#!/usr/bin/env python3
"""Bounded, candidate-only SimBench Extreme strategy-depth probe.

The probe replays the locked 72-tick semi-urban SimBench profile directly on
the native pandapower backend.  It tests a feedback policy with exactly one
physical call after the first observed voltage violation.  A successful
one-call trace disproves the three-stage *necessary* depth required by the
declared Extreme tier; it is not used to manufacture a multi-stage task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from domains.power_grid.backends.cigre_distribution import (  # noqa: E402
    CigreDistributionBackend,
)
from scripts.grounded_candidate_pipeline import (  # noqa: E402
    derive_simbench_seed,
    rank_simbench_windows,
)

SCENARIO_PATH = REPO_ROOT / (
    "scenarios/staging/simbench_protocol21_current_20260814/power_grid/"
    "simbench_mv_semiurban_timeseries_control/deep_planning/extreme/"
    "simbench_mv_semiurban_extreme_p19584_h72_current_8603b2.yaml"
)
SOURCE_SUITE_PATH = (
    REPO_ROOT / "reports/simbench_protocol21_current_20260814/source_suite_current_8603b2.json"
)
PRIOR_DEPTH_PATH = REPO_ROOT / (
    "reports/simbench_protocol21_current_20260814/"
    "protocol21_full_bounded_current_8603b2/strategy_depth_protocol2_v21.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "reports/simbench_quality_refine_20260814/terminal_ledger.json"
NETWORK = "simbench:1-MV-semiurb--0-sw"
PROFILE_START_INDEX = 19584
HORIZON_TICKS = 72
EXTREME_DEPTH_FLOOR = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _native_loss(records: list[dict[str, Any]]) -> float:
    return sum(
        float(record["n_voltage_violations"])
        * CigreDistributionBackend.VOLTAGE_VIOLATION_COST_PER_TICK
        + float(record["n_overloads"]) * CigreDistributionBackend.OVERLOAD_COST_PER_TICK
        + float(record["n_disconnected_lines"])
        * CigreDistributionBackend.DISCONNECTION_COST_PER_LINE_TICK
        for record in records
    )


def _run_native(seed: Any, *, feedback_control: bool) -> dict[str, Any]:
    backend = CigreDistributionBackend()
    backend.reset(deepcopy(seed))
    records: list[dict[str, Any]] = []
    action: dict[str, Any] | None = None
    for tick in range(HORIZON_TICKS):
        native = backend.tick(tick)
        record = {
            "tick": tick,
            "aggregate_demand_mw": round(float(native.aggregate_demand_mw), 9),
            "aggregate_generation_mw": round(float(native.aggregate_generation_mw), 9),
            "rho_max": round(float(native.rho_max), 9),
            "n_voltage_violations": int(native.n_voltage_violations),
            "n_overloads": int(native.n_overloads),
            "n_disconnected_lines": int(native.n_disconnected_lines),
            "converged": bool(native.converged),
            "solved_state_digest": backend._solved_state_digest(),
            "events": list(native.realized_events),
        }
        records.append(record)
        if (
            feedback_control
            and action is None
            and record["n_voltage_violations"] > 0
            and tick < HORIZON_TICKS - 1
        ):
            generation = backend._net.sgen.p_mw.astype(float)
            generator_index = int(generation.idxmax())
            result = backend.apply_tool_effect(
                "redispatch_generation",
                {"generator_index": generator_index, "target_mw": 0.0},
            )
            action = {
                "tick": tick,
                "observation": {
                    "n_voltage_violations": record["n_voltage_violations"],
                    "rho_max": record["rho_max"],
                },
                "tool": "redispatch_generation",
                "args": {"generator_index": generator_index, "target_mw": 0.0},
                "backend_result": result,
            }
    source_trace = backend.protocol21_source_trace()
    return {
        "records": records,
        "trace_sha256": _stable_hash(records),
        "native_loss": _native_loss(records),
        "action": action,
        "source_trace": source_trace,
    }


def _probe() -> dict[str, Any]:
    scenario = yaml.safe_load(SCENARIO_PATH.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict):
        raise ValueError("SimBench staging scenario must be a mapping")
    expected = {
        "backend_kind": "cigre_distribution",
        "difficulty_level": "extreme",
        "horizon_ticks": HORIZON_TICKS,
    }
    drift = [key for key, value in expected.items() if scenario.get(key) != value]
    if drift:
        raise ValueError(f"SimBench Extreme input drifted: {drift}")
    window_hash = str(
        ((scenario.get("provenance") or {}).get("time_window") or {}).get("source_window_sha256")
        or ""
    )
    windows = rank_simbench_windows(NETWORK, horizon_ticks=HORIZON_TICKS, limit=2)
    window = next(
        (row for row in windows if int(row["profile_start_index"]) == PROFILE_START_INDEX),
        None,
    )
    if window is None or str(window["source_window_sha256"]) != window_hash:
        raise ValueError("locked SimBench source window no longer reproduces")
    seed = derive_simbench_seed(
        network=NETWORK,
        difficulty="extreme",
        source_window=window,
        intensity=0.6,
    )
    seed.horizon_ticks = HORIZON_TICKS
    seed.provenance.files = list((scenario.get("provenance") or {}).get("files") or [])

    wait_first = _run_native(seed, feedback_control=False)
    wait_second = _run_native(seed, feedback_control=False)
    control_first = _run_native(seed, feedback_control=True)
    control_second = _run_native(seed, feedback_control=True)
    action = control_first["action"]
    if not isinstance(action, dict):
        raise ValueError("bounded feedback probe found no response opportunity")
    action_tick = int(action["tick"])
    outcome_tick = action_tick + 1
    wait_records = wait_first["records"]
    control_records = control_first["records"]
    wait_loss = float(wait_first["native_loss"])
    control_loss = float(control_first["native_loss"])
    source_events = [
        event for record in wait_records for event in record["events"] if isinstance(event, dict)
    ]
    demands = [float(record["aggregate_demand_mw"]) for record in wait_records]
    generations = [float(record["aggregate_generation_mw"]) for record in wait_records]
    headroom_floor = max(1.0, 0.05 * wait_loss)
    return {
        "source_profile_consumed": any(
            event.get("type") == "simbench_profile_window_started"
            and int(event.get("profile_start_index", -1)) == PROFILE_START_INDEX
            for event in source_events
        ),
        "source_material_change": (
            max(demands) - min(demands) > 1e-6 and max(generations) - min(generations) > 1e-6
        ),
        "deterministic_wait_replay": (wait_first["trace_sha256"] == wait_second["trace_sha256"]),
        "deterministic_control_replay": (
            control_first["trace_sha256"] == control_second["trace_sha256"]
            and control_first["action"] == control_second["action"]
        ),
        "native_control_effect": (
            control_records[outcome_tick]["solved_state_digest"]
            != wait_records[outcome_tick]["solved_state_digest"]
        ),
        "wait_trace_sha256": wait_first["trace_sha256"],
        "control_trace_sha256": control_first["trace_sha256"],
        "wait_native_loss": wait_loss,
        "control_native_loss": control_loss,
        "native_headroom": wait_loss - control_loss,
        "headroom_floor": headroom_floor,
        "successful_control_call_count": 1,
        "successful_control_tick_count": 1,
        "safe_control": bool(
            all(record["converged"] for record in control_records)
            and max(record["n_overloads"] for record in control_records) == 0
            and max(record["n_disconnected_lines"] for record in control_records) == 0
            and control_loss < wait_loss
        ),
        "feedback_policy": {
            "trigger": "first_observed_native_voltage_violation",
            "future_source_knowledge_used": False,
            "action": action,
            "outcome_tick": outcome_tick,
            "outcome_native_state_digest": control_records[outcome_tick]["solved_state_digest"],
            "post_control_review_ticks": list(range(outcome_tick, HORIZON_TICKS)),
        },
        "source_trace": control_first["source_trace"],
    }


def terminal_disposition(
    probe: dict[str, Any],
    *,
    perturbation_kind_count: int,
    extreme_depth_floor: int,
) -> tuple[str, list[str]]:
    native_checks = {
        "source_profile_consumed": bool(probe.get("source_profile_consumed")),
        "source_material_change": bool(probe.get("source_material_change")),
        "deterministic_wait_replay": bool(probe.get("deterministic_wait_replay")),
        "deterministic_control_replay": bool(probe.get("deterministic_control_replay")),
        "native_control_effect": bool(probe.get("native_control_effect")),
        "positive_native_headroom": float(probe.get("native_headroom") or 0.0)
        >= float(probe.get("headroom_floor") or 0.0),
        "safe_control": bool(probe.get("safe_control")),
    }
    native_failures = [name for name, passed in native_checks.items() if not passed]
    if native_failures:
        return "held_native_gate_failure", native_failures
    blockers = []
    if perturbation_kind_count < 3:
        blockers.append("extreme_perturbation_kind_floor_not_met")
    successful_calls = int(probe.get("successful_control_call_count") or 0)
    successful_ticks = int(probe.get("successful_control_tick_count") or 0)
    if successful_calls and min(successful_calls, successful_ticks) < extreme_depth_floor:
        blockers.append("single_control_trace_disproves_extreme_depth")
    if not blockers:
        blockers.append("necessary_dependency_depth_not_proven")
    return "held_strategy_depth_unproven", blockers


def build_terminal_ledger() -> dict[str, Any]:
    start_identity = implementation_identity(REPO_ROOT)
    scenario = yaml.safe_load(SCENARIO_PATH.read_text(encoding="utf-8"))
    perturbation_kinds = sorted(
        {
            str(row.get("kind"))
            for row in scenario.get("perturbations") or []
            if isinstance(row, dict) and row.get("kind")
        }
    )
    probe = _probe()
    scientific_disposition, scientific_blockers = terminal_disposition(
        probe,
        perturbation_kind_count=len(perturbation_kinds),
        extreme_depth_floor=EXTREME_DEPTH_FLOOR,
    )
    end_identity = implementation_identity(REPO_ROOT)
    stable = (
        start_identity["implementation_tree_sha256"] == end_identity["implementation_tree_sha256"]
    )
    disposition = scientific_disposition if stable else "held_implementation_drift"
    blockers = list(scientific_blockers)
    if not stable:
        blockers.insert(0, "implementation_tree_changed_during_bounded_probe")
    prior_depth = json.loads(PRIOR_DEPTH_PATH.read_text(encoding="utf-8"))
    prior_sample = (prior_depth.get("samples") or [{}])[0]
    return {
        "schema_version": "simbench-quality-refine-terminal-v1",
        "status": "complete",
        "scope": "candidate_only_non_release",
        "release_membership_changed": False,
        "core_admission_claimed": False,
        "full_protocol21_calls": 0,
        "full_protocol21_skipped_reason": (
            "bounded_native_trace_disproves_extreme_necessary_depth"
        ),
        "implementation_identity": {
            "start": start_identity,
            "end": end_identity,
            "stable": stable,
        },
        "input_bindings": {
            "scenario_path": str(SCENARIO_PATH.relative_to(REPO_ROOT)),
            "scenario_sha256": _sha256(SCENARIO_PATH),
            "source_suite_path": str(SOURCE_SUITE_PATH.relative_to(REPO_ROOT)),
            "source_suite_sha256": _sha256(SOURCE_SUITE_PATH),
            "prior_strategy_depth_path": str(PRIOR_DEPTH_PATH.relative_to(REPO_ROOT)),
            "prior_strategy_depth_sha256": _sha256(PRIOR_DEPTH_PATH),
        },
        "n_inputs": 1,
        "n_terminal": 1,
        "disposition_counts": {disposition: 1},
        "results": [
            {
                "scenario_id": scenario["scenario_id"],
                "scenario_signature": scenario["scenario_signature"],
                "difficulty_level": "extreme",
                "horizon_ticks": HORIZON_TICKS,
                "work_state": "terminal",
                "disposition": disposition,
                "scientific_disposition": scientific_disposition,
                "blockers": blockers,
                "perturbation_kinds": perturbation_kinds,
                "perturbation_kind_count": len(perturbation_kinds),
                "extreme_depth_floor": EXTREME_DEPTH_FLOOR,
                "bounded_native_probe": probe,
                "prior_protocol21_depth_diagnostic": {
                    "implementation_tree_sha256": prior_depth.get("implementation_tree_sha256"),
                    "disposition": prior_sample.get("disposition"),
                    "successful_strategy_tick_upper_bound": prior_sample.get(
                        "successful_strategy_tick_upper_bound"
                    ),
                    "required_depth_lower_bound": prior_sample.get("required_depth_lower_bound"),
                    "depth_proof_kinds": prior_sample.get("depth_proof_kinds"),
                },
                "refine_decision": {
                    "new_scenario_materialized": False,
                    "reason": (
                        "Adding forced milestones would not make them necessary: "
                        "one source-responsive native call already clears the task "
                        "headroom gate. Such a layout would be score shaping rather "
                        "than an Extreme scheduling task."
                    ),
                },
            }
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = path.resolve()
    resolved.relative_to((REPO_ROOT / "reports").resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ledger = build_terminal_ledger()
    _write_json(args.output, ledger)
    print(
        json.dumps(
            {
                "status": ledger["status"],
                "n_terminal": ledger["n_terminal"],
                "disposition_counts": ledger["disposition_counts"],
                "implementation_stable": ledger["implementation_identity"]["stable"],
                "full_protocol21_calls": ledger["full_protocol21_calls"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
