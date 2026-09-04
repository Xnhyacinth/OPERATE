#!/usr/bin/env python3
"""Build a fail-closed pilot queue for external benchmark transfer candidates.

The queue is an audit artifact, not a scenario materializer.  Original candidate
fields are preserved verbatim.  External raw assets can never become ready; the
single DynaSched/JSPLIB row is ready only when source-locked native runtime,
state-changing controls, response-window, replay, depth, and independence
evidence all pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

SCENARIO_ID = (
    "logistics/job_shop_dispatch/time_pressure/high/"
    "jobshop_la09_dynamic_recovery_high_s44"
)
SOURCE_PATH = "works/JSPLIB-Instances/instances/la09"
SOURCE_SHA256 = "0c1730baa9e9480efff7d062d5af5758f056a5889326d877bc7d5b43d71a2cc4"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = REPO_ROOT / "release/dt_sched_bench_v0_52_0_candidate/external_transfer_candidates_v3/queue.json"
DEFAULT_EVIDENCE_ROOT = REPO_ROOT / "release/dt_sched_bench_v0_52_0_candidate/protocol21_unified_five_domain_v8_sumo_runtime_revalidation"
DEFAULT_RUNTIME_PROBE = REPO_ROOT / "release/dt_sched_bench_v0_52_0_candidate/protocol21_expansion_trials/external_dyna_native_backend_probe_v1.json"
DEFAULT_OUTPUT = REPO_ROOT / "release/dt_sched_bench_v0_52_0_candidate/protocol21_expansion_trials/external_native_pilot_queue_v1.json"


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find(rows: Any, scenario_id: str) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None
    return next(
        (row for row in rows if isinstance(row, dict) and row.get("scenario_id") == scenario_id),
        None,
    )


def _evidence_for_dyna(
    repo_root: Path, evidence_root: Path, runtime_probe_path: Path | None
) -> dict[str, Any]:
    paths = {
        "source_consumption": evidence_root / "source_consumption_protocol2_v21.json",
        "source_grounded": evidence_root / "source_grounded_protocol2_v21.json",
        "task_contracts": evidence_root / "task_contracts_protocol2_v21.json",
        "strategy_depth": evidence_root / "strategy_depth_protocol2_v21.json",
    }
    selection_path = evidence_root / "core_selection.json"
    if not selection_path.is_file():
        selection_path = evidence_root / "refined_core_selection_protocol2_v21.json"
    paths["core_selection"] = selection_path
    payloads = {name: _read(path) for name, path in paths.items()}
    source_consumption = payloads["source_consumption"]
    source_grounded = payloads["source_grounded"]
    task_contracts = payloads["task_contracts"]
    strategy_depth = payloads["strategy_depth"]
    core_selection = payloads["core_selection"]

    consumption = _find(source_consumption.get("results"), SCENARIO_ID)
    grounded = _find(source_grounded.get("results"), SCENARIO_ID)
    task = _find(task_contracts.get("results"), SCENARIO_ID)
    depth = _find(strategy_depth.get("samples"), SCENARIO_ID)
    core = _find(core_selection.get("scenarios"), SCENARIO_ID)
    source = repo_root / SOURCE_PATH
    runtime_probe = _read(runtime_probe_path) if runtime_probe_path else {}
    runtime = runtime_probe.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    controls = runtime_probe.get("state_changing_controls")
    controls = controls if isinstance(controls, dict) else {}
    repair = controls.get("machine_repair")
    repair = repair if isinstance(repair, dict) else {}

    consumed_hash = (consumption or {}).get("consumed_source_hashes", {}).get(SOURCE_PATH)
    grounded_gates = (grounded or {}).get("gates") or {}
    task_evidence = (task or {}).get("evidence") or {}
    terminal = (task or {}).get("terminal_integrity") or {}
    source_lock = source.is_file() and _sha256(source) == SOURCE_SHA256
    native_runtime = bool(
        consumption
        and consumption.get("status") == "passed"
        and consumption.get("proof_kind") == "direct_runtime_files"
        and consumption.get("direct_runtime_match") is True
        and consumed_hash == SOURCE_SHA256
        and consumption.get("runtime_trace_observed") is True
        and runtime_probe.get("ready_for_method_transfer_probe") is True
        and runtime.get("source_hash_matches_expected") is True
        and runtime.get("runtime_trace_observed") is True
    )
    state_effect = bool(
        consumption
        and consumption.get("state_effect_observed") is True
        and consumption.get("derived_backend_state_fields")
        and repair.get("state_changed") is True
    )
    response_window = bool(
        task
        and task.get("status") == "passed"
        and task.get("completed") is True
        and terminal.get("release_ready") is True
        and int(terminal.get("response_window_extensions", 0) or 0) > 0
        and task_evidence.get("ordered_tool_milestones_met") is True
    )
    replay = bool(
        grounded_gates.get("deterministic_replay") is True
        and grounded_gates.get("counterfactual") is True
        and consumption
        and consumption.get("deterministic_across_replays") is True
    )
    depth_ok = bool(
        grounded_gates.get("difficulty_proof") is True
        and depth
        and depth.get("disposition") == "required_depth_lower_bound_met"
        and int(depth.get("required_depth_lower_bound", 0) or 0)
        >= int(depth.get("tier_floor", 1) or 1)
    )
    independence = bool(
        grounded_gates.get("independence") is True
        and (core or {}).get("protocol21_lineage", {}).get("ready") is True
    )
    gates = {
        "source_lock": source_lock and grounded_gates.get("source_lock") is True,
        "native_runtime": native_runtime,
        "state_changing_control": state_effect,
        "response_window": response_window,
        "deterministic_counterfactual_replay": replay,
        "difficulty_depth": depth_ok,
        "effective_source_independence": independence,
    }
    return {
        "scenario_id": SCENARIO_ID,
        "native_source": {"path": SOURCE_PATH, "sha256": SOURCE_SHA256},
        "gates": gates,
        "ready": all(gates.values()),
        "evidence_refs": {name: path.name for name, path in paths.items()},
        "evidence_file_sha256": {
            name: _sha256(path) for name, path in paths.items() if _sha256(path)
        },
        "runtime_probe_sha256": _sha256(runtime_probe_path) if runtime_probe_path else None,
        "observed": {
            "consumed_hash": consumed_hash,
            "operations_scheduled": task_evidence.get("operations_scheduled"),
            "physical_tools": task_evidence.get("distinct_physical_tools"),
            "milestone_ticks": task_evidence.get("selected_milestone_ticks"),
            "depth_lower_bound": (depth or {}).get("required_depth_lower_bound"),
        },
        "runtime_probe": runtime_probe,
    }


def _rank_key(row: dict[str, Any]) -> tuple[int, int, str]:
    mode = row.get("transfer_mode")
    source = str(row.get("physical_source_key") or "")
    candidate_id = str(row.get("candidate_id") or "")
    if mode == "external_raw_asset":
        return (3, 3, candidate_id)
    if candidate_id == "dynaschedbench__jsplib__high":
        return (0, 0, candidate_id)
    if str(row.get("external_source_id")) == "realm_bench" and source.startswith("jsplib:"):
        return (1, 1, candidate_id)
    if str(row.get("external_source_id")) == "frontier_eng" and source.startswith("nrel_microgrid:"):
        return (2, 1, candidate_id)
    if mode == "native_method_transfer" and row.get("source_assets_exist"):
        return (2, 2, candidate_id)
    return (3, 2, candidate_id)


def build_report(
    repo_root: Path,
    queue_path: Path,
    evidence_root: Path,
    runtime_probe_path: Path | None = None,
) -> dict[str, Any]:
    queue = _read(queue_path)
    rows = queue.get("candidates")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("queue candidates must be a list of objects")
    dyna = _evidence_for_dyna(repo_root, evidence_root, runtime_probe_path)
    ranked = sorted((copy.deepcopy(row) for row in rows), key=_rank_key)
    output_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        row["pilot_rank"] = rank
        row["pilot_status"] = "held"
        row["feasibility_class"] = "held_external_or_missing_evidence"
        row["pilot_gate_evidence"] = {}
        if row.get("transfer_mode") == "external_raw_asset":
            row["feasibility_class"] = "external_raw_forbidden"
            row["pilot_gate_evidence"] = {"source_lock": "not_admissible"}
        elif row.get("candidate_id") == "dynaschedbench__jsplib__high":
            row["pilot_status"] = "ready_method_transfer" if dyna["ready"] else "held"
            row["feasibility_class"] = "native_method_transfer"
            row["pilot_gate_evidence"] = dyna
        elif row.get("transfer_mode") == "native_method_transfer":
            row["feasibility_class"] = (
                "native_adapter_available_evidence_pending"
                if row.get("source_assets_exist") and row.get("native_tools")
                else "native_adapter_or_asset_missing"
            )
            row["pilot_gate_evidence"] = {
                "source_lock": "local_asset_present_only" if row.get("source_assets_exist") else "missing",
                "native_runtime": "pending_runtime_consumption",
                "state_changing_control": "pending",
                "response_window": "pending",
                "deterministic_counterfactual_replay": "pending",
                "difficulty_depth": "pending",
            }
        output_rows.append(row)
    ready = [row for row in output_rows if row["pilot_status"] != "held"]
    return {
        "schema_version": "protocol21-external-native-pilot-queue-v1",
        "status": "pilot_queue_ready_no_core_admission",
        "direct_core_admission": False,
        "external_raw_direct_admission": False,
        "n_input_candidates": len(rows),
        "n_ready_method_transfer": len(ready),
        "n_held": sum(row["pilot_status"] == "held" for row in output_rows),
        "queue_sha256": _sha256(queue_path),
        "evidence_root": str(evidence_root),
        "source_evidence": dyna,
        "promotion_policy": "Only rows with all native evidence gates pass may be marked ready_method_transfer; readiness never implies Core admission.",
        "candidates": output_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument("--runtime-probe", type=Path, default=DEFAULT_RUNTIME_PROBE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        args.repo_root.resolve(),
        args.queue.resolve(),
        args.evidence_root.resolve(),
        args.runtime_probe.resolve() if args.runtime_probe else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("n_input_candidates", "n_ready_method_transfer", "n_held")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
