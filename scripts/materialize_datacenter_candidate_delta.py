#!/usr/bin/env python3
"""Materialize source-locked Datacenter candidates without editing active Core."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.oracle_offline import OracleOfflineAgent  # noqa: E402
from baselines.wait_only import WaitOnlyAgent  # noqa: E402
from core import Action, ToolCall  # noqa: E402
from core.protocol21_evidence import canonicalize_repo_owned_paths  # noqa: E402
from core.scenario_validator import validate_scenario_yaml  # noqa: E402
from core.source_asset_contract import resolve_source_asset_contract  # noqa: E402
from core.suite_identity import recompute_signature_with_seed  # noqa: E402
from domains.datacenter.adapter import DatacenterEnvironment  # noqa: E402
from domains.datacenter.source_native_candidates import (  # noqa: E402
    build_openb_candidate,
    datacenter_dimension_applicability,
)
from domains.registry import (  # noqa: E402
    get_backend_capability,
    resolve_backend_source_contract_builder,
)
from evaluation.dimension_applicability import (  # noqa: E402
    dimension_applicability_contract_issue,
)
from scripts.build_protocol21_candidate_source_suite import (  # noqa: E402
    build_suite,
)

DEFAULT_REFINEMENT = (
    ROOT / ".hl/artifacts/operate_v058_datacenter_archive_candidate_refinement.json"
)
DEFAULT_SPOT_LEDGER = (
    ROOT / ".hl/artifacts/datacenter_spot_candidate_ledger_20260828.json"
)
DEFAULT_ACTIVE_SUITE = ROOT / "release/operate_v0_61_0/protocol21_source_suite.json"
DEFAULT_OUTPUT_ROOT = ROOT / ".hl/artifacts/operate_v058_datacenter_delta"
ALIBABA_CLUSTERDATA_COMMIT = "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71"
ALIBABA_TRACE_LICENSE = "Apache-2.0 upstream repository; trace terms apply"
READY_FOR_FULL_ADMISSION = "ready_for_full_admission"
LEGACY_READY_FOR_FULL_ADMISSION = "core_ready"
READABLE_REFINEMENT_DISPOSITIONS = {
    READY_FOR_FULL_ADMISSION,
    LEGACY_READY_FOR_FULL_ADMISSION,
}


def _load(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _safe_output_root(output_root: Path) -> Path:
    resolved = output_root.resolve()
    protected = (
        ROOT / "release/operate_v0_58_0",
        ROOT / "scenarios/operate_v0_58_0",
    )
    if any(
        resolved == item.resolve() or item.resolve() in resolved.parents
        for item in protected
    ):
        raise ValueError("candidate delta output cannot be inside active Core")
    return resolved


def _source_event_contract(backend_kind: str) -> dict[str, Any]:
    event_type = (
        "pod_arrival"
        if backend_kind == "alibaba_openb_gpu_placement"
        else "job_arrival"
    )
    return {
        "schema_version": "datacenter_source_event_contract_v1",
        "registry": [
            {
                "event_type": event_type,
                "event_class": "task",
                "origin": "source_schedule",
                "actionability": "runtime_materiality_and_remaining_horizon",
            }
        ],
        "unknown_events_actionable": False,
    }


def _task_requirements(backend_kind: str, horizon: int) -> dict[str, Any]:
    if backend_kind == "alibaba_openb_gpu_placement":
        return {
            "min_distinct_control_ticks": 2,
            "min_distinct_physical_tools": 2,
            "ordered_tool_milestones": [
                {"tool": "set_placement_policy", "not_after_tick": 0},
                {
                    "tool": "place_pod",
                    "not_before_tick": 1,
                    "not_after_tick": max(1, horizon - 2),
                },
            ],
        }
    return {
        "min_distinct_control_ticks": 1,
        "min_distinct_physical_tools": 1,
        "ordered_tool_milestones": [
            {"tool": "set_queue_policy", "not_after_tick": min(3, horizon - 2)}
        ],
    }


def _finalize_body(
    body: dict[str, Any],
    *,
    candidate_id: str,
    repo_root: Path,
) -> dict[str, Any]:
    body = copy.deepcopy(body)
    backend_kind = str(body.get("backend_kind") or "")
    body["scenario_id"] = candidate_id
    body["seed_id"] = candidate_id
    body["candidate_only"] = True
    body["release_admission"] = False
    body["difficulty_mode"] = "deep_planning"
    body["difficulty_level"] = "high"
    tick_minutes = body.get("tick_minutes")
    if isinstance(tick_minutes, float) and tick_minutes.is_integer():
        body["tick_minutes"] = int(tick_minutes)
    body["policy_contract"] = {
        "strict_prompt": True,
        "benchmark_side_hints": False,
    }
    provenance = body.setdefault("provenance", {})
    if not provenance.get("commit"):
        provenance["commit"] = ALIBABA_CLUSTERDATA_COMMIT
    if not provenance.get("license"):
        provenance["license"] = ALIBABA_TRACE_LICENSE
    body.setdefault("perturbations", [])
    config = body.setdefault("backend_config", {})
    applicability = datacenter_dimension_applicability(backend_kind)
    issue = dimension_applicability_contract_issue(applicability)
    if issue is not None:
        raise ValueError(f"dimension_applicability_invalid:{backend_kind}:{issue}")
    config["dimension_applicability"] = applicability
    config["candidate_materialization"] = {
        "candidate_id": candidate_id,
        "active_core": False,
    }
    config["source_event_contract"] = _source_event_contract(backend_kind)
    config["task_requirements"] = _task_requirements(
        backend_kind, int(body["horizon_ticks"])
    )
    if backend_kind == "alibaba_openb_gpu_placement":
        # The LLM must own placement; autonomous placement would erase the
        # native decision axis this candidate is intended to measure.
        config["autonomous_placement"] = False
        capability = get_backend_capability(backend_kind)
        builder = resolve_backend_source_contract_builder(capability)
        body["source_contract"] = builder(body, repo_root)
    contract = body.get("source_contract")
    if not isinstance(contract, dict):
        raise ValueError("source_contract_missing")
    required = list(contract.get("runtime_input") or []) + list(
        contract.get("derivation_input") or []
    )
    hashes: dict[str, str] = {}
    for raw in required:
        source = Path(str(raw))
        if not source.is_absolute():
            source = repo_root / source
        if not source.is_file():
            raise FileNotFoundError(source)
        hashes[str(raw)] = _sha256(source)
    contract["file_sha256s"] = hashes
    body.pop("scenario_signature", None)
    body["scenario_signature"] = recompute_signature_with_seed(body, int(body["seed"]))
    errors = validate_scenario_yaml(body)
    if errors:
        raise ValueError(f"scenario_schema_invalid:{'|'.join(errors)}")
    resolved = resolve_source_asset_contract(body, repo_root=repo_root)
    if resolved.contract_errors or resolved.missing_required_files:
        raise ValueError(
            "source_contract_invalid:"
            f"{list(resolved.contract_errors)}:{list(resolved.missing_required_files)}"
        )
    return body


def _run_agent(body: dict[str, Any], agent_type: type[Any]) -> dict[str, Any]:
    env = DatacenterEnvironment()
    observation = env.reset(body, seed=int(body["seed"]))
    agent = agent_type()
    agent.reset(env, body, seed=int(body["seed"]))
    calls: list[str] = []
    while True:
        action = agent.act(observation, env.get_tool_specs())
        calls.extend(call.name for call in action.tool_calls)
        step = env.step(action)
        observation = step.observation
        if step.done:
            break
    truth = env.ground_truth()
    result = {
        "cost": round(sum(truth["cost_components"].values()), 9),
        "tool_names": sorted(set(calls)),
        "control_summary": truth["control_summary"],
    }
    env.close()
    return result


def _run_spot_strategy(
    body: dict[str, Any], strategy: dict[str, Any]
) -> dict[str, Any]:
    env = DatacenterEnvironment()
    observation = env.reset(body, seed=int(body["seed"]))
    tool_results: list[dict[str, Any]] = []
    while True:
        tick = int(env.tick)
        calls: list[ToolCall] = []
        if tick == int(strategy.get("tick", -1)):
            kind = str(strategy.get("kind") or "")
            if kind == "queue_policy":
                calls.append(
                    ToolCall(
                        name="set_queue_policy",
                        args={"policy": strategy["policy"]},
                        idempotency_key=f"cal_policy_{tick}",
                    )
                )
            elif kind == "preempt_low_criticality":
                running = [
                    (job_id, job)
                    for job_id, job in (env.ground_truth().get("jobs") or {}).items()
                    if job.get("status") == "running"
                ]
                if running:
                    job_id, _ = min(
                        running,
                        key=lambda item: (
                            float(item[1].get("criticality") or 0.0),
                            -int(item[1].get("remaining_ticks") or 0),
                            item[0],
                        ),
                    )
                    calls.append(
                        ToolCall(
                            name="preempt_job",
                            args={"job_id": job_id},
                            idempotency_key=f"cal_preempt_{tick}",
                        )
                    )
            elif kind == "reserve_one_gpu":
                calls.append(
                    ToolCall(
                        name="reserve_gpu_capacity",
                        args={"gpu_units": 1.0, "duration_ticks": 2},
                        idempotency_key=f"cal_reserve_{tick}",
                    )
                )
        if not calls:
            calls = [ToolCall(name="wait", idempotency_key=f"cal_wait_{tick}")]
        step = env.step(Action(tool_calls=calls, dominant=calls[0].name))
        tool_results.extend(
            {
                "tick": tick,
                "name": result.name,
                "ok": result.ok,
                "state_changing": result.state_changing,
            }
            for result in step.tool_results
        )
        observation = step.observation
        if step.done:
            break
    truth = env.ground_truth()
    result = {
        "cost": round(sum(truth["cost_components"].values()), 9),
        "tool_results": tool_results,
        "control_summary": truth["control_summary"],
    }
    env.close()
    del observation
    return result


def bounded_spot_reference_probe(body: dict[str, Any]) -> dict[str, Any]:
    """Search bounded native action/timing choices outside the model prompt."""
    horizon = int(body.get("horizon_ticks") or 1)
    response_window_end = min(4, max(0, horizon - 2))
    strategies = [
        {"kind": "queue_policy", "policy": policy, "tick": tick}
        for tick in range(response_window_end + 1)
        for policy in (
            "shortest_job_first",
            "least_gpu_first",
            "deadline_criticality_first",
        )
    ]
    strategies.extend(
        {"kind": kind, "tick": tick}
        for tick in range(response_window_end + 1)
        for kind in ("preempt_low_criticality", "reserve_one_gpu")
    )
    wait = _run_spot_strategy(body, {"kind": "wait", "tick": -1})
    attempts = [
        {"strategy": strategy, **_run_spot_strategy(body, strategy)}
        for strategy in strategies
    ]
    selected = min(
        attempts,
        key=lambda item: (
            float(item["cost"]),
            json.dumps(item["strategy"], sort_keys=True),
        ),
    )
    headroom = float(wait["cost"]) - float(selected["cost"])
    threshold = max(1.0, float(wait["cost"]) * 0.001)
    material = headroom > threshold
    return {
        "status": "passed" if material else "failed",
        "method": "bounded_native_action_timing_search_v1",
        "search_space": {
            "response_window_ticks": [0, response_window_end],
            "native_action_families": [
                "set_queue_policy",
                "preempt_job",
                "reserve_gpu_capacity",
            ],
            "n_candidates": len(strategies),
        },
        "selected_strategy": selected["strategy"],
        "selected_timing": selected["strategy"]["tick"],
        "wait_cost": wait["cost"],
        "reference_cost": selected["cost"],
        "headroom": round(headroom, 9),
        "materiality_threshold": round(threshold, 9),
        "material_headroom": material,
        "counterfactual_evidence": {
            "wait_control_summary": wait["control_summary"],
            "reference_control_summary": selected["control_summary"],
            "reference_tool_results": selected["tool_results"],
        },
    }


def _default_reference_probe(body: dict[str, Any]) -> dict[str, Any]:
    if body.get("backend_kind") == "alibaba_trace_sim":
        return bounded_spot_reference_probe(body)
    wait = _run_agent(body, WaitOnlyAgent)
    reference = _run_agent(body, OracleOfflineAgent)
    headroom = float(wait["cost"]) - float(reference["cost"])
    threshold = max(1.0, float(wait["cost"]) * 0.001)
    return {
        "status": "passed" if headroom > threshold else "failed",
        "method": "oracle_offline_vs_wait_v1",
        "search_space": {"agent": "oracle_offline"},
        "selected_timing": reference["control_summary"].get("distinct_control_ticks"),
        "wait_cost": wait["cost"],
        "reference_cost": reference["cost"],
        "headroom": round(headroom, 9),
        "materiality_threshold": round(threshold, 9),
        "material_headroom": headroom > threshold,
        "counterfactual_evidence": {
            "wait": wait,
            "reference": reference,
        },
    }


def _runtime_probe(body: dict[str, Any]) -> dict[str, Any]:
    env = DatacenterEnvironment()
    observation = env.reset(body, seed=int(body["seed"]))
    capability = get_backend_capability(str(body["backend_kind"]))
    tool_names = {row["function"]["name"] for row in env.get_tool_specs()}
    typed: list[dict[str, Any]] = []
    while env.tick < env.horizon and not typed:
        step = env.step(
            Action(
                tool_calls=[
                    ToolCall(
                        name="wait",
                        idempotency_key=f"runtime_probe_wait_{env.tick}",
                    )
                ]
            )
        )
        typed.extend(
            row
            for row in step.info.extra["world_evolution_records"]
            if row.get("event_type") in capability.source_scheduled_event_types
        )
        if step.done:
            break
    evidence = env.source_consumption_evidence(scenario=body)
    env.close()
    del observation
    return {
        "reset": True,
        "native_tools_registered": set(capability.control_tools) <= tool_names,
        "typed_source_events": bool(typed)
        and all(not row.get("event_contract_violations") for row in typed),
        "source_evidence_status": evidence.get("status"),
        "source_event_types": sorted({str(row.get("event_type")) for row in typed}),
    }


def _blocker(row: dict[str, Any], code: str, detail: str) -> dict[str, Any]:
    return {
        "candidate_id": str(row.get("candidate_id") or ""),
        "source_id": str(row.get("source_id") or ""),
        "source_unit": str(row.get("source_unit") or ""),
        "status": "blocked",
        "blocker_code": code,
        "detail": detail,
    }


def materialize_datacenter_candidate_delta(
    *,
    refinement_ledger: Path = DEFAULT_REFINEMENT,
    spot_ledger: Path = DEFAULT_SPOT_LEDGER,
    active_suite: Path = DEFAULT_ACTIVE_SUITE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    execute: bool = False,
    repo_root: Path = ROOT,
    reference_probe: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    refinement = _load(refinement_ledger, "refinement ledger")
    spot = _load(spot_ledger, "Spot ledger")
    rows = refinement.get("rows")
    spot_rows = spot.get("candidates")
    if not isinstance(rows, list) or not isinstance(spot_rows, list):
        raise ValueError("candidate ledgers must contain rows/candidates lists")
    if not active_suite.is_file():
        raise FileNotFoundError(active_suite)
    selected = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("final_disposition") in READABLE_REFINEMENT_DISPOSITIONS
        and row.get("classification_scope") == "candidate"
    ]
    identities = [str(row.get("candidate_id") or "") for row in selected]
    invalid = sorted(
        candidate_id
        for candidate_id, count in Counter(identities).items()
        if not candidate_id or count > 1
    )
    if invalid:
        raise ValueError(f"duplicate selected candidate_id: {invalid[0]!r}")
    by_spot_id = {
        str(row.get("candidate_id") or ""): row
        for row in spot_rows
        if isinstance(row, dict)
    }
    output_root = _safe_output_root(output_root)
    probe = reference_probe or _default_reference_probe
    scenarios: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    bodies: list[tuple[Path, dict[str, Any]]] = []
    for row in sorted(selected, key=lambda item: str(item["candidate_id"])):
        candidate_id = str(row["candidate_id"])
        try:
            candidate_recipe = (row.get("evidence") or {}).get("candidate_recipe") or {}
            if (
                row.get("source_family") == "alibaba_cluster_trace_gpu_v2023_openb"
                or candidate_recipe.get("backend_kind") == "alibaba_openb_gpu_placement"
            ):
                roles = (
                    (candidate_recipe.get("backend_config") or {}).get(
                        "source_transform"
                    )
                    or {}
                ).get("source_file_roles") or {}
                units = [
                    Path(str(value))
                    for value in (
                        roles.get("node_inventory"),
                        roles.get("pod_trace"),
                    )
                    if value
                ]
                if not units:
                    units = [
                        repo_root / "works/clusterdata" / value
                        for value in str(row["source_unit"]).split("+")
                    ]
                if len(units) != 2:
                    raise ValueError("openb_source_unit_invalid")
                resolved_units = [
                    unit if unit.is_absolute() else repo_root / unit for unit in units
                ]
                body = build_openb_candidate(
                    node_path=resolved_units[0],
                    pod_path=resolved_units[1],
                    repo_root=repo_root,
                )
                scenario_id = body["scenario_id"]
            else:
                source = by_spot_id.get(candidate_id)
                if source is None or not isinstance(source.get("suite_recipe"), dict):
                    raise ValueError("spot_suite_recipe_missing")
                body = copy.deepcopy(source["suite_recipe"])
                scenario_id = candidate_id.replace(
                    "/structural_prefilter/", "/deep_planning/high/"
                )
            body = _finalize_body(
                body,
                candidate_id=scenario_id,
                repo_root=repo_root,
            )
            runtime = _runtime_probe(body)
            if not all(
                (
                    runtime["reset"],
                    runtime["native_tools_registered"],
                    runtime["typed_source_events"],
                    runtime["source_evidence_status"] == "passed",
                )
            ):
                raise ValueError(f"runtime_contract_probe_failed:{runtime}")
            reference = probe(body)
            if (
                reference.get("status") != "passed"
                or reference.get("material_headroom") is not True
            ):
                blockers.append(
                    _blocker(
                        row,
                        "reference_headroom_unproven",
                        json.dumps(reference, sort_keys=True),
                    )
                )
                continue
            path = output_root / "scenarios" / f"{_slug(scenario_id)}.yaml"
            bodies.append((path, body))
            scenarios.append(
                {
                    "candidate_id": candidate_id,
                    "scenario_id": scenario_id,
                    "scenario_signature": body["scenario_signature"],
                    "path": str(path),
                    "domain": "datacenter",
                    "family": body["family"],
                    "backend_kind": body["backend_kind"],
                    "difficulty_mode": body["difficulty_mode"],
                    "difficulty_level": body["difficulty_level"],
                    "runtime_probe": runtime,
                    "reference_probe": reference,
                    "status": "materialized_candidate",
                }
            )
        except Exception as exc:  # noqa: BLE001
            blockers.append(
                _blocker(
                    row,
                    "candidate_materialization_failed",
                    f"{type(exc).__name__}: {exc}",
                )
            )
    terminal = len(scenarios) + len(blockers)
    if terminal != len(selected):
        raise RuntimeError("selected Datacenter candidate accounting failure")
    report = {
        "schema_version": "operate-datacenter-candidate-delta-v1",
        "status": "candidate_only_requires_protocol21_admission",
        "input_bindings": {
            "refinement_ledger": {
                "path": str(refinement_ledger.resolve()),
                "sha256": _sha256(refinement_ledger),
            },
            "spot_ledger": {
                "path": str(spot_ledger.resolve()),
                "sha256": _sha256(spot_ledger),
            },
            "active_suite": {
                "path": str(active_suite.resolve()),
                "sha256": _sha256(active_suite),
                "read_only": True,
            },
        },
        "selection": {
            "final_disposition": READY_FOR_FULL_ADMISSION,
            "candidate_scope_plus_executable_openb_source": True,
        },
        "summary": {
            "n_selected_ready_for_full_admission": len(selected),
            "n_materialized": len(scenarios),
            "n_blocked": len(blockers),
            "n_terminal": terminal,
            "n_unresolved": 0,
            "active_core_modified": False,
        },
        "scenarios": scenarios,
        "blockers": blockers,
    }
    portable_report = canonicalize_repo_owned_paths(report, repo_root=repo_root)
    if not execute:
        return portable_report
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite candidate delta: {output_root}")
    for path, body in bodies:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(body, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    report_path = output_root / "candidate_report.json"
    report_path.write_text(
        json.dumps({"scenarios": scenarios}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        suite = canonicalize_repo_owned_paths(
            build_suite(report_path), repo_root=repo_root
        )
    finally:
        report_path.write_text(
            json.dumps(
                {"scenarios": portable_report["scenarios"]},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    suite["constraints"] = {
        **suite["constraints"],
        "prompt_mode": "strict",
        "active_core_modified": False,
        "reference_calibration_hidden_from_model": True,
    }
    (output_root / "source_suite.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "materialization_ledger.json").write_text(
        json.dumps(portable_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return portable_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement-ledger", type=Path, default=DEFAULT_REFINEMENT)
    parser.add_argument("--spot-ledger", type=Path, default=DEFAULT_SPOT_LEDGER)
    parser.add_argument("--active-suite", type=Path, default=DEFAULT_ACTIVE_SUITE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = materialize_datacenter_candidate_delta(
        refinement_ledger=args.refinement_ledger,
        spot_ledger=args.spot_ledger,
        active_suite=args.active_suite,
        output_root=args.output_root,
        execute=args.execute,
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
