#!/usr/bin/env python3
"""Bounded native prefilter for long-horizon SimBench candidates.

This builder is candidate-only. It consumes locked SimBench topology/profile
windows through the existing native pandapower backend, checks deterministic
source-driven evolution and an isolated native actuator effect, and stops
before Protocol-2.1 replay when native task headroom is absent. Declared
perturbations are recorded as transformations and never contribute source
independence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.scenario_validator import validate_scenario_yaml  # noqa: E402
from core.source_asset_contract import virtual_source_identity_sha256  # noqa: E402
from core.suite_identity import recompute_signature_with_seed  # noqa: E402
from domains.power_grid.backends.cigre_distribution import (  # noqa: E402
    CigreDistributionBackend,
)
from evaluation.task_completion import task_completion_contract  # noqa: E402
from scripts.grounded_candidate_pipeline import (  # noqa: E402
    derive_simbench_seed,
    rank_simbench_windows,
)

PIPELINE_VERSION = "simbench_protocol21_long_horizon_prefilter_v1"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "simbench_long_horizon_prefilter_current.json"
DEFAULT_SURVIVOR_REPORT = (
    REPO_ROOT / "reports" / "simbench_protocol21_survivor_candidate_report.json"
)
DEFAULT_SURVIVOR_YAML = (
    REPO_ROOT
    / "scenarios"
    / "staging"
    / "simbench_protocol21_long_horizon"
    / "power_grid"
    / "simbench_mv_semiurban_timeseries_control"
    / "deep_planning"
    / "extreme"
    / "grounded_1_MV_semiurb_0_sw_extreme_p19584_r0p6_h72.yaml"
)
DEFAULT_INTENSITY = 0.6
DEFAULT_SPECS = (
    {
        "network": "simbench:1-MV-semiurb--0-sw",
        "difficulty": "high",
        "horizon_ticks": 48,
        "window_rank": 0,
    },
    {
        "network": "simbench:1-MV-rural--0-sw",
        "difficulty": "high",
        "horizon_ticks": 48,
        "window_rank": 0,
    },
    {
        "network": "simbench:1-MV-semiurb--0-sw",
        "difficulty": "extreme",
        "horizon_ticks": 72,
        "window_rank": 1,
    },
)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def effective_source_key(seed: dict[str, Any]) -> str:
    """Identify a physical graph/profile window; ignore stress transformations."""
    config = seed.get("backend_config") or {}
    window = (seed.get("provenance") or {}).get("time_window") or {}
    return _stable_hash(
        {
            "dataset": (seed.get("provenance") or {}).get("data_source"),
            "backend": seed.get("backend_kind"),
            "network": config.get("network"),
            "profile_start_index": window.get("profile_start_index"),
            "source_window_sha256": window.get("source_window_sha256"),
        }
    )


def _native_loss(records: list[Any]) -> float:
    return sum(
        float(record.n_voltage_violations)
        * CigreDistributionBackend.VOLTAGE_VIOLATION_COST_PER_TICK
        + float(record.n_overloads) * CigreDistributionBackend.OVERLOAD_COST_PER_TICK
        + float(record.n_disconnected_lines)
        * CigreDistributionBackend.DISCONNECTION_COST_PER_LINE_TICK
        for record in records
    )


def _trace_digest(records: list[Any]) -> str:
    return _stable_hash(
        [
            {
                "tick": record.tick,
                "demand": record.aggregate_demand_mw,
                "generation": record.aggregate_generation_mw,
                "rho_max": round(float(record.rho_max), 9),
                "voltage_violations": record.n_voltage_violations,
                "overloads": record.n_overloads,
                "disconnected": record.n_disconnected_lines,
                "converged": record.converged,
                "events": record.realized_events,
            }
            for record in records
        ]
    )


def _run_wait(seed_obj: Any) -> tuple[list[Any], CigreDistributionBackend]:
    backend = CigreDistributionBackend()
    backend.reset(seed_obj)
    records = [backend.tick(tick) for tick in range(seed_obj.horizon_ticks)]
    return records, backend


def _run_reference(seed_obj: Any, action_tick: int) -> dict[str, Any]:
    """Apply an isolated source-conditioned Volt/Var curtailment probe."""
    backend = CigreDistributionBackend()
    control = CigreDistributionBackend()
    backend.reset(seed_obj)
    control.reset(deepcopy(seed_obj))
    records = []
    action: dict[str, Any] | None = None
    native_action_effect = False
    for tick in range(seed_obj.horizon_ticks):
        record = backend.tick(tick)
        control.tick(tick)
        records.append(record)
        if action is not None and tick == int(action["tick"]) + 1:
            native_action_effect = backend._solved_state_digest() != control._solved_state_digest()
        max_voltage = float(backend._net.res_bus.vm_pu.max())
        should_act = max_voltage >= 1.045 or tick == action_tick
        if not should_act or action is not None or tick >= seed_obj.horizon_ticks - 1:
            continue
        generation = backend._net.sgen.p_mw.astype(float)
        if generation.empty or float(generation.max()) <= 0.0:
            continue
        results = []
        for generator_index in generation.nlargest(3).index:
            result = backend.apply_tool_effect(
                "redispatch_generation",
                {"generator_index": int(generator_index), "target_mw": 0.0},
            )
            results.append(
                {
                    "generator_index": int(generator_index),
                    "target_mw": 0.0,
                    "backend_result": result,
                }
            )
        action = {
            "tick": tick,
            "tool": "redispatch_generation",
            "trigger": (
                "observed_overvoltage_margin" if max_voltage >= 1.045 else "source_event_probe_tick"
            ),
            "max_voltage_pu": max_voltage,
            "commands": results,
        }
    return {
        "records": records,
        "action": action,
        "native_action_effect": native_action_effect,
    }


def _runtime_prefilter(seed_obj: Any) -> dict[str, Any]:
    wait_first, first_backend = _run_wait(seed_obj)
    wait_second, _ = _run_wait(seed_obj)
    event_ticks = list(
        (seed_obj.backend_config.get("derivation_recipe") or {}).get("event_ticks") or []
    )
    action_tick = min(
        max(0, int(event_ticks[0]) if event_ticks else seed_obj.horizon_ticks // 2),
        seed_obj.horizon_ticks - 2,
    )
    reference = _run_reference(seed_obj, action_tick)
    reference_records = reference["records"]
    realized_events = [event for record in wait_first for event in record.realized_events]
    demand = [float(record.aggregate_demand_mw) for record in wait_first]
    generation = [float(record.aggregate_generation_mw) for record in wait_first]
    wait_loss = _native_loss(wait_first)
    reference_loss = _native_loss(reference_records)
    headroom = wait_loss - reference_loss
    headroom_floor = max(1.0, 0.05 * wait_loss)
    family = str(seed_obj.family)
    return {
        "source_profile_consumed": any(
            event.get("type") == "simbench_profile_window_started"
            and int(event.get("profile_start_index", -1))
            == int(seed_obj.backend_config.get("profile_start_index", -2))
            for event in realized_events
        ),
        "source_material_change": bool(
            demand
            and generation
            and (max(demand) - min(demand) > 1e-6)
            and (max(generation) - min(generation) > 1e-6)
        ),
        "deterministic_replay": _trace_digest(wait_first) == _trace_digest(wait_second),
        "native_action_effect": bool(reference["native_action_effect"]),
        "wait_native_loss": wait_loss,
        "reference_native_loss": reference_loss,
        "native_headroom": headroom,
        "headroom_floor": headroom_floor,
        "task_contract": task_completion_contract(seed_obj.domain, family),
        "task_contract_supported": task_completion_contract(seed_obj.domain, family)
        != "unsupported",
        "horizon_ticks": seed_obj.horizon_ticks,
        "demand_range_mw": [min(demand), max(demand)],
        "generation_range_mw": [min(generation), max(generation)],
        "max_line_loading_percent": 100.0 * max(float(record.rho_max) for record in wait_first),
        "total_voltage_violation_bus_ticks": sum(
            int(record.n_voltage_violations) for record in wait_first
        ),
        "total_overload_line_ticks": sum(int(record.n_overloads) for record in wait_first),
        "reference_probe": reference["action"],
        "source_constructor_hash": first_backend._source_constructor_hash,
        "source_solver_state_digest": first_backend._source_solver_state_digest,
    }


def _terminal_disposition(summary: dict[str, Any]) -> tuple[str, list[str]]:
    checks = {
        "source_profile_consumed": bool(summary.get("source_profile_consumed")),
        "source_material_change": bool(summary.get("source_material_change")),
        "deterministic_replay": bool(summary.get("deterministic_replay")),
        "native_action_effect": bool(summary.get("native_action_effect")),
        "positive_wait_task_loss": float(summary.get("wait_native_loss") or 0.0) > 0.0,
        "positive_native_headroom": float(summary.get("native_headroom") or 0.0)
        >= float(summary.get("headroom_floor") or 0.0),
        "task_contract_supported": bool(summary.get("task_contract_supported")),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if not failures:
        failures = ["full_protocol21_pending"]
    return "held_repair", failures


def summarize_candidate(
    *,
    scenario_id: str,
    seed: dict[str, Any],
    runtime_summary: dict[str, Any],
) -> dict[str, Any]:
    disposition, failures = _terminal_disposition(runtime_summary)
    return {
        "scenario_id": scenario_id,
        "stage": "native_prefilter",
        "work_state": "terminal",
        "disposition": disposition,
        "effective_source_key": effective_source_key(seed),
        "declared_perturbations_count_toward_source_independence": False,
        "gate_failures": failures,
        "runtime_evidence": runtime_summary,
        "full_protocol21_calls": 0,
        "core_admission_claimed": False,
    }


def build_report(*, specs: tuple[dict[str, Any], ...] = DEFAULT_SPECS) -> dict[str, Any]:
    rows = []
    seen_effective_sources: set[str] = set()
    for spec in specs:
        horizon = int(spec["horizon_ticks"])
        windows = rank_simbench_windows(
            str(spec["network"]),
            horizon_ticks=horizon,
            limit=int(spec["window_rank"]) + 1,
        )
        window = windows[int(spec["window_rank"])]
        seed_obj = derive_simbench_seed(
            network=str(spec["network"]),
            difficulty=str(spec["difficulty"]),
            source_window=window,
            intensity=DEFAULT_INTENSITY,
        )
        seed_obj.horizon_ticks = horizon
        seed_obj.backend_config["long_horizon_candidate"] = {
            "horizon_ticks": horizon,
            "profile_window_is_effective_source": True,
            "declared_perturbation_is_source_independence": False,
        }
        seed = seed_obj.to_dict()
        source_key = effective_source_key(seed)
        scenario_id = (
            f"power_grid/{seed_obj.family}/deep_planning/{spec['difficulty']}/"
            f"{seed_obj.seed_id}_h{horizon}"
        )
        if source_key in seen_effective_sources:
            row = summarize_candidate(
                scenario_id=scenario_id,
                seed=seed,
                runtime_summary={
                    "task_contract_supported": False,
                    "duplicate_effective_source": True,
                },
            )
            row["disposition"] = "secondary_duplicate"
            row["gate_failures"] = ["duplicate_effective_source"]
        else:
            seen_effective_sources.add(source_key)
            row = summarize_candidate(
                scenario_id=scenario_id,
                seed=seed,
                runtime_summary=_runtime_prefilter(seed_obj),
            )
        row.update(
            {
                "network": spec["network"],
                "difficulty_target": spec["difficulty"],
                "horizon_ticks": horizon,
                "profile_start_index": window["profile_start_index"],
                "source_window_sha256": window["source_window_sha256"],
            }
        )
        rows.append(row)
    dispositions: dict[str, int] = {}
    for row in rows:
        key = str(row["disposition"])
        dispositions[key] = dispositions.get(key, 0) + 1
    return {
        "schema_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete",
        "scope": "candidate_only_non_release",
        "runtime_versions": {
            name: importlib.metadata.version(name) for name in ("simbench", "pandapower")
        },
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_suite_sha256": _stable_hash(specs),
        "n_inputs": len(specs),
        "n_terminal": len(rows),
        "n_full_protocol21_calls": 0,
        "disposition_counts": dispositions,
        "release_membership_changed": False,
        "results": rows,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _survivor_row(prefilter: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row
        for row in prefilter.get("results") or []
        if row.get("network") == "simbench:1-MV-semiurb--0-sw"
        and row.get("difficulty_target") == "extreme"
        and int(row.get("horizon_ticks") or 0) == 72
        and int(row.get("profile_start_index") or -1) == 19584
    ]
    if len(matches) != 1:
        raise ValueError("exactly one bounded SimBench survivor is required")
    row = matches[0]
    evidence = row.get("runtime_evidence") or {}
    required = {
        "source_profile_consumed": bool(evidence.get("source_profile_consumed")),
        "source_material_change": bool(evidence.get("source_material_change")),
        "deterministic_replay": bool(evidence.get("deterministic_replay")),
        "native_action_effect": bool(evidence.get("native_action_effect")),
        "task_contract_supported": bool(evidence.get("task_contract_supported")),
        "positive_wait_native_loss": float(evidence.get("wait_native_loss") or 0.0) > 0.0,
        "positive_native_headroom": float(evidence.get("native_headroom") or 0.0) > 0.0,
        "only_full_protocol21_pending": row.get("gate_failures") == ["full_protocol21_pending"],
    }
    failures = [name for name, passed in required.items() if not passed]
    if failures:
        raise ValueError(f"SimBench survivor native gates drifted: {failures}")
    return row


def build_protocol21_survivor(
    *,
    prefilter_path: Path,
    implementation_tree_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one non-release survivor report and its scenario body."""
    prefilter = _load_json_object(prefilter_path)
    row = _survivor_row(prefilter)
    runtime_versions = prefilter.get("runtime_versions") or {}
    actual_versions = {
        name: importlib.metadata.version(name) for name in ("simbench", "pandapower")
    }
    if runtime_versions != actual_versions:
        raise ValueError("SimBench survivor runtime versions drifted")
    horizon = int(row["horizon_ticks"])
    windows = rank_simbench_windows(str(row["network"]), horizon_ticks=horizon, limit=2)
    source_window = next(
        (
            window
            for window in windows
            if int(window["profile_start_index"]) == int(row["profile_start_index"])
        ),
        None,
    )
    if source_window is None:
        raise ValueError("locked survivor window is no longer ranked")
    if source_window["source_window_sha256"] != row["source_window_sha256"]:
        raise ValueError("locked survivor source window hash drifted")
    seed_obj = derive_simbench_seed(
        network=str(row["network"]),
        difficulty="extreme",
        source_window=source_window,
        intensity=DEFAULT_INTENSITY,
    )
    seed_obj.horizon_ticks = horizon
    seed_obj.difficulty_mode = "deep_planning"
    seed_obj.provenance.time_window = {
        **seed_obj.provenance.time_window,
        "horizon_ticks": horizon,
        "source_window_sha256": row["source_window_sha256"],
    }
    body = seed_obj.to_dict()
    profile_uri = str(body["provenance"]["files"][0])
    constructor_uri = "pandapower-simbench://1-MV-semiurb--0-sw@1.6.2"
    body["provenance"]["files"] = [constructor_uri, profile_uri]
    virtual_sha256 = virtual_source_identity_sha256(constructor_uri)
    if virtual_sha256 is None:
        raise ValueError("SimBench virtual source URI is not lockable")
    source_key = effective_source_key(body)
    effective_key = f"simbench_official:{source_key}"
    physical_source_key = "simbench_official:simbench:1-MV-semiurb--0-sw"
    body["backend_config"]["source_denominator_key"] = effective_key
    body["backend_config"]["long_horizon_candidate"] = {
        "pipeline_version": PIPELINE_VERSION,
        "prefilter_artifact_sha256": hashlib.sha256(prefilter_path.read_bytes()).hexdigest(),
        "implementation_tree_sha256": implementation_tree_sha256,
        "profile_window_is_effective_source": True,
        "declared_perturbation_is_source_independence": False,
    }
    body["source_contract"] = {
        "runtime_input": [],
        "derivation_input": [constructor_uri],
        "implementation_asset": ["domains/power_grid/backends/cigre_distribution.py"],
        "metadata": ["domains/power_grid/seeds/from_cigre.py"],
        "license": [],
        "file_sha256s": {constructor_uri: virtual_sha256},
        "derived_window": {
            "sha256": str(row["source_window_sha256"]),
            "recipe_version": PIPELINE_VERSION,
        },
    }
    scenario_id = (
        "power_grid/simbench_mv_semiurban_timeseries_control/deep_planning/"
        "extreme/grounded_1_MV_semiurb_0_sw_extreme_p19584_r0p6_h72"
    )
    body["scenario_id"] = scenario_id
    body["scenario_signature"] = recompute_signature_with_seed(body, int(body["seed"]))
    errors = validate_scenario_yaml(body)
    if errors:
        raise ValueError(f"survivor scenario validation failed: {errors}")
    report = {
        "schema_version": "candidate-report-v1",
        "status": "staging_candidates_pending_full_admission",
        "scope": "candidate_only_non_release",
        "release_membership_changed": False,
        "constraints": {
            "one_per_effective_source_identity": True,
            "declared_perturbations_count_toward_independence": False,
            "full_protocol21_required": True,
            "core_admission_claimed": False,
        },
        "bindings": {
            "implementation_tree_sha256": implementation_tree_sha256,
            "prefilter_path": str(prefilter_path.resolve()),
            "prefilter_sha256": hashlib.sha256(prefilter_path.read_bytes()).hexdigest(),
            "source_window_sha256": row["source_window_sha256"],
            "runtime_versions": actual_versions,
        },
        "scenarios": [
            {
                "scenario_id": scenario_id,
                "scenario_signature": body["scenario_signature"],
                "source_denominator_key": body["backend_config"]["source_denominator_key"],
                "effective_source_key": effective_key,
                "physical_source_key": physical_source_key,
                "path": "",
                "native_prefilter": row["runtime_evidence"],
            }
        ],
    }
    return report, body


def materialize_protocol21_survivor(
    *,
    prefilter_path: Path,
    report_path: Path,
    scenario_path: Path,
    implementation_tree_sha256: str,
) -> dict[str, Any]:
    report_path = report_path.resolve()
    scenario_path = scenario_path.resolve()
    reports_root = (REPO_ROOT / "reports").resolve()
    staging_root = (REPO_ROOT / "scenarios" / "staging").resolve()
    if not report_path.resolve().is_relative_to(reports_root):
        raise ValueError("survivor report must stay under reports/")
    if not scenario_path.resolve().is_relative_to(staging_root):
        raise ValueError("survivor YAML must stay under scenarios/staging/")
    report, body = build_protocol21_survivor(
        prefilter_path=prefilter_path,
        implementation_tree_sha256=implementation_tree_sha256,
    )
    import yaml

    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    report["scenarios"][0]["path"] = scenario_path.relative_to(REPO_ROOT).as_posix()
    _write_json(report_path, report)
    return report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.resolve()
    reports_root = (REPO_ROOT / "reports").resolve()
    if not resolved.is_relative_to(reports_root):
        raise ValueError("candidate output must stay under reports/")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--materialize-survivor", action="store_true")
    parser.add_argument("--prefilter", type=Path)
    parser.add_argument("--survivor-report", type=Path, default=DEFAULT_SURVIVOR_REPORT)
    parser.add_argument("--survivor-yaml", type=Path, default=DEFAULT_SURVIVOR_YAML)
    parser.add_argument("--implementation-tree-sha256")
    args = parser.parse_args()
    if args.materialize_survivor:
        if args.prefilter is None or not args.implementation_tree_sha256:
            parser.error(
                "--materialize-survivor requires --prefilter and --implementation-tree-sha256"
            )
        report = materialize_protocol21_survivor(
            prefilter_path=args.prefilter.resolve(),
            report_path=args.survivor_report,
            scenario_path=args.survivor_yaml,
            implementation_tree_sha256=args.implementation_tree_sha256,
        )
        print(json.dumps({"status": report["status"], "n_scenarios": 1}, indent=2))
        return
    report = build_report()
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "n_inputs", "n_terminal", "disposition_counts")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
