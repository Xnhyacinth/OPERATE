#!/usr/bin/env python3
"""Run locked-source Traffic and Datacenter operational-agency controls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from runner.episode import run_one  # noqa: E402

AGENT_NAME = "operational_agency_positive_control"
SUMO_VERSION = "1.27.1"
DEFAULT_TRAFFIC = (
    REPO_ROOT / "scenarios/operate_v0_58_0/traffic/"
    "signal_coordination/deep_planning/medium/"
    "resco_cologne1_demand_surge_medium_s9414.yaml"
)
DEFAULT_DATACENTER = (
    REPO_ROOT / "scenarios/operate_v0_58_0/datacenter/"
    "gpu_cluster_queue_control/time_pressure/basic/"
    "alibaba_gpu_w052_377_382_basic.yaml"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "reports/operate_v0_58_0/agency/traffic_datacenter.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_scenario(path: Path, *, expected_domain: str) -> dict[str, Any]:
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict):
        raise ValueError(f"scenario must be an object: {path}")
    if str(scenario.get("domain") or "") != expected_domain:
        raise ValueError(f"scenario domain is not {expected_domain}: {path}")
    return scenario


def _source_bindings(scenario: dict[str, Any]) -> dict[str, str]:
    contract = scenario.get("source_contract") or {}
    declared = [
        *(contract.get("runtime_input") or []),
        *(contract.get("derivation_input") or []),
        *((scenario.get("provenance") or {}).get("files") or []),
    ]
    bindings: dict[str, str] = {}
    for value in declared:
        if not isinstance(value, str) or not value:
            raise ValueError("source path must be a non-empty string")
        path = (REPO_ROOT / value).resolve()
        path.relative_to(REPO_ROOT.resolve())
        if not path.is_file():
            raise FileNotFoundError(value)
        bindings[value] = _sha256(path)
    if not bindings:
        raise ValueError("positive control requires at least one locked source file")
    return dict(sorted(bindings.items()))


def _sumo_runtime_contract() -> dict[str, Any]:
    required_packages = ("traci", "sumolib", "sumo-data")
    versions: dict[str, str | None] = {}
    for package in ("libsumo", "traci", "sumolib", "sumo-data"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    spec = importlib.util.find_spec("sumo")
    binary: Path | None = None
    if spec is not None and spec.submodule_search_locations:
        candidate = Path(next(iter(spec.submodule_search_locations))) / "bin" / "sumo"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            binary = candidate.resolve()
    version_line = ""
    if binary is not None:
        completed = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        version_line = (completed.stdout or completed.stderr).splitlines()[0].strip()
    exact = bool(
        binary is not None
        and version_line.endswith(SUMO_VERSION)
        and all(versions[package] == SUMO_VERSION for package in required_packages)
    )
    return {
        "target_version": SUMO_VERSION,
        "packages": versions,
        "required_packages": list(required_packages),
        "optional_packages": ["libsumo"],
        "binary": str(binary) if binary else None,
        "binary_sha256": _sha256(binary) if binary else None,
        "binary_version_line": version_line,
        "transport": "traci_tcp",
        "exact_version_match": exact,
    }


def _configure_sumo_runtime(contract: dict[str, Any]) -> None:
    if contract.get("exact_version_match") is not True:
        raise RuntimeError("SUMO runtime does not match locked version 1.27.1")
    binary = Path(str(contract["binary"]))
    os.environ["OPERATE_TRAFFIC_BACKEND_REAL"] = "1"
    os.environ["OPERATE_TRAFFIC_FORCE_TRANSPORT"] = "traci"
    os.environ["PATH"] = f"{binary.parent}{os.pathsep}{os.environ.get('PATH', '')}"


def _sumo_process_ids() -> list[int]:
    completed = subprocess.run(
        ["pgrep", "-f", r"(^|/)sumo( |$)"],
        check=False,
        capture_output=True,
        text=True,
    )
    return sorted(int(value) for value in completed.stdout.split() if value.strip().isdigit())


def summarize_domain_control(
    result: dict[str, Any],
    *,
    domain: str,
) -> dict[str, Any]:
    """Fail closed unless native completion and exact masked evidence pass."""

    blockers: list[str] = []
    counterfactual = result.get("counterfactual") or {}
    for prefix in ("per_action", "per_action_group"):
        expected = int(counterfactual.get(f"{prefix}_expected") or 0)
        attempted = int(counterfactual.get(f"{prefix}_attempted") or 0)
        completed = int(counterfactual.get(f"{prefix}_completed") or 0)
        failures = counterfactual.get(f"{prefix}_failures") or []
        if (
            counterfactual.get(f"{prefix}_status") != "complete"
            or expected != attempted
            or attempted != completed
            or failures
        ):
            blockers.append(f"{prefix}_attribution_not_complete")

    task = result.get("task_completion") or {}
    if task.get("completed") is not True:
        blockers.append(f"native_task_incomplete:{task.get('reason_code') or 'unknown'}")

    trajectory = result.get("trajectory_summary") or {}
    valid_ids = {
        str(value)
        for value in trajectory.get("operational_agency_valid_evidence_ids") or []
        if value
    }
    records = trajectory.get("event_response_records") or []
    origins = {
        str(event.get("event_id")): str(event.get("origin") or "")
        for event in trajectory.get("world_evolution_records") or []
        if isinstance(event, dict) and event.get("event_id")
    }

    def event_origin(record: dict[str, Any]) -> str | None:
        value = origins.get(str(record.get("event_id") or "")) or str(
            record.get("event_origin") or ""
        )
        return value or None

    positive_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or record.get("response_status") != "causal":
            continue
        trigger_ids = {str(value) for value in record.get("trigger_evidence_ids") or [] if value}
        consumed_ids = {
            str(value) for value in record.get("action_consumes_evidence_ids") or [] if value
        }
        effect_ids = {
            str(value) for value in record.get("backend_effect_evidence_ids") or [] if value
        }
        try:
            masked_delta = float(record.get("masked_action_group_delta"))
        except (TypeError, ValueError):
            continue
        if (
            masked_delta > 0.0
            and trigger_ids.intersection(consumed_ids).intersection(valid_ids)
            and effect_ids
            and effect_ids.issubset(valid_ids)
        ):
            positive_records.append(
                {
                    **dict(record),
                    "event_origin": event_origin(record),
                    "declared_perturbation": (
                        event_origin(record) == "declared_perturbation"
                    ),
                }
            )
    if not positive_records:
        blockers.append("no_positive_masked_causal_record")

    profile = trajectory.get("operational_agency_profile") or {}
    outcome = (profile.get("dimensions") or {}).get("outcome_influence") or {}
    if (
        profile.get("runtime_binding_verified") is not True
        or profile.get("runtime_evidence_binding_verified") is not True
        or profile.get("masked_replay_binding_verified") is not True
        or int(profile.get("causal_record_count") or 0) < 1
        or outcome.get("applicable") is not True
        or not outcome.get("evidence_ids")
    ):
        blockers.append("operational_agency_profile_not_positive")

    terminal = dict(trajectory.get("terminal_integrity") or {})
    terminal.update(
        {
            "terminal": True,
            "fatal": bool(
                (result.get("ground_truth_summary") or {}).get(
                    "chose_fatal_option", False
                )
            ),
            "fatal_error": (
                result.get("error") if result.get("status") == "error" else None
            ),
            "orphan_process_count": 0,
        }
    )
    return {
        "domain": domain,
        "status": "held" if blockers else "passed",
        "blockers": blockers,
        "scenario_id": result.get("scenario_id"),
        "scenario_signature": result.get("scenario_signature"),
        "agent_name": result.get("agent_name"),
        "seed": result.get("seed"),
        "task_completion": task,
        "counterfactual": {
            key: counterfactual.get(key)
            for key in (
                "actual_cost",
                "counterfactual_cost",
                "prevented_loss",
                "per_action_status",
                "per_action_expected",
                "per_action_attempted",
                "per_action_completed",
                "per_action_failures",
                "per_action_capped",
                "per_action",
                "per_action_group_status",
                "per_action_group_expected",
                "per_action_group_attempted",
                "per_action_group_completed",
                "per_action_group_failures",
                "per_action_groups",
            )
        },
        "positive_causal_record_count": len(positive_records),
        "positive_causal_records": positive_records,
        "event_response_records": [
            {
                **dict(record),
                "event_origin": event_origin(record),
                "declared_perturbation": (
                    event_origin(record) == "declared_perturbation"
                ),
            }
            for record in records
            if isinstance(record, dict)
        ],
        "operational_agency_valid_evidence_ids": sorted(valid_ids),
        "operational_agency_profile": profile,
        "terminal_integrity": terminal,
    }


def _run_domain(
    domain: str,
    scenario: dict[str, Any],
    *,
    repeats: int,
) -> dict[str, Any]:
    summaries = [
        summarize_domain_control(
            run_one(
                scenario,
                AGENT_NAME,
                per_action_attribution=True,
                per_action_cap=None,
                per_action_group_attribution=True,
                per_action_group_cap=None,
                within_tick_interaction=True,
            ),
            domain=domain,
        )
        for _ in range(repeats)
    ]
    deterministic = all(summary == summaries[0] for summary in summaries[1:])
    if not deterministic:
        for summary in summaries:
            summary["status"] = "held"
            summary["blockers"].append("deterministic_repeat_mismatch")
    return {
        "status": summaries[0]["status"],
        "determinism": {"repeats": repeats, "passed": deterministic},
        "result": summaries[0],
    }


def run_controls(
    *,
    traffic_path: Path,
    datacenter_path: Path,
    output_path: Path,
    repeats: int,
) -> dict[str, Any]:
    if repeats != 2:
        raise ValueError("domain positive controls require exactly two repeats")
    implementation_before = implementation_identity(REPO_ROOT)
    paths = {"traffic": traffic_path, "datacenter": datacenter_path}
    scenarios = {
        domain: _load_scenario(path, expected_domain=domain) for domain, path in paths.items()
    }
    runtime = _sumo_runtime_contract()
    controls: dict[str, Any] = {}
    before_pids = _sumo_process_ids()
    try:
        _configure_sumo_runtime(runtime)
        for domain in ("traffic", "datacenter"):
            scenario = scenarios[domain]
            try:
                control = _run_domain(domain, scenario, repeats=repeats)
            except Exception as exc:  # noqa: BLE001 - artifact records exact blocker
                control = {
                    "status": "held",
                    "determinism": {"repeats": repeats, "passed": False},
                    "result": {
                        "domain": domain,
                        "status": "held",
                        "blockers": [f"runtime_error:{type(exc).__name__}:{exc}"],
                    },
                }
            control["scenario_binding"] = {
                "path": paths[domain].relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(paths[domain]),
                "scenario_id": scenario.get("scenario_id"),
            }
            control["source_file_bindings"] = _source_bindings(scenario)
            controls[domain] = control
    finally:
        after_pids = _sumo_process_ids()
    orphan_pids = sorted(set(after_pids).difference(before_pids))
    if orphan_pids:
        controls.setdefault("traffic", {}).setdefault("result", {}).setdefault(
            "blockers", []
        ).append("sumo_orphan_process_detected")
        controls["traffic"]["status"] = "held"
        controls["traffic"]["result"]["status"] = "held"

    implementation_after = implementation_identity(REPO_ROOT)
    implementation_stable = (
        implementation_before["implementation_tree_sha256"]
        == implementation_after["implementation_tree_sha256"]
    )
    if not implementation_stable:
        for control in controls.values():
            control["status"] = "held"
            result = control.setdefault("result", {})
            result["status"] = "held"
            result.setdefault("blockers", []).append("implementation_hash_changed_during_run")

    report = {
        "schema_version": "domain-operational-agency-runtime-positive-controls-v1",
        "status": (
            "passed"
            if controls.keys() == {"traffic", "datacenter"}
            and all(control.get("status") == "passed" for control in controls.values())
            and not orphan_pids
            and implementation_stable
            else "held"
        ),
        "diagnostic_only": True,
        "release_admission": False,
        "core_admission_claimed": False,
        "agent_name": AGENT_NAME,
        "attribution_contract": {
            "per_action": "complete_uncapped",
            "per_action_group": "complete_uncapped",
            "mode": "complete_uncapped",
            "per_action_cap": None,
            "per_action_group_cap": None,
            "event_evidence": "authoritative_visible_source_event",
            "positive_masked_replay_required": True,
        },
        "implementation_identity": implementation_after,
        "implementation_stability": {
            "before": implementation_before,
            "after": implementation_after,
            "passed": implementation_stable,
        },
        "traffic_runtime_contract": runtime,
        "sumo_process_check": {
            "before": before_pids,
            "after": after_pids,
            "orphan_pids": orphan_pids,
            "passed": not orphan_pids,
        },
        "controls": controls,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traffic-scenario", type=Path, default=DEFAULT_TRAFFIC)
    parser.add_argument("--datacenter-scenario", type=Path, default=DEFAULT_DATACENTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    report = run_controls(
        traffic_path=args.traffic_scenario.resolve(),
        datacenter_path=args.datacenter_scenario.resolve(),
        output_path=args.output.resolve(),
        repeats=args.repeats,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "controls": {
                    domain: {
                        "status": control["status"],
                        "positive_causal_record_count": control["result"].get(
                            "positive_causal_record_count", 0
                        ),
                    }
                    for domain, control in report["controls"].items()
                },
                "orphan_pids": report["sumo_process_check"]["orphan_pids"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
