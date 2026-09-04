#!/usr/bin/env python3
"""Replay-calibrate sample-level trajectory complexity for a core suite.

Each row is checkpointed as an atomic shard; the final artifact is written
atomically once. Reported minimal traces are deterministic 1-minimal upper
bounds, never global-shortest claims.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import signal
import sys
import time
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit._common import _resolve_scenario_path  # noqa: E402
from audit.episode_cache import AUDIT_EPISODE_CONTRACT_VERSION  # noqa: E402
from baselines import make_agent  # noqa: E402
from core import Action  # noqa: E402
from core.counterfactual import run_counterfactual  # noqa: E402
from core.implementation_identity import implementation_identity  # noqa: E402
from data import (  # noqa: E402
    analyze_trajectory_steps,
    exhaustive_trace_subset_minimum,
    minimize_successful_action_sequence,
)
from data.trajectory_analysis import (  # noqa: E402
    TRAJECTORY_ANALYSIS_CONTRACT_VERSION,
    validated_event_action_edges,
)
from domains.registry import get_backend_capability, get_domain_spec  # noqa: E402
from evaluation import (  # noqa: E402
    SCORING_VERSION,
    domain_cost_extractor,
    domain_counterfactual_report,
    evaluate_task_completion,
    separate_task_outcome_and_process,
)
from runner import (  # noqa: E402
    EVALUATION_IMPLEMENTATION_FINGERPRINT,
    EVALUATION_PROTOCOL_VERSION,
)
from runner.episode import _run_episode_loop  # noqa: E402
from scripts.calibrate_core_candidate import (  # noqa: E402
    HARD_ISOLATION_BACKENDS,
)

DEFAULT_SUITE = (
    REPO_ROOT / "release" / "operate_v0_58_0" / "protocol21_source_suite.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release"
    / "operate_v0_58_0_candidate"
    / "operate_v058_formal"
    / "complexity_protocol2_v21.json"
)
REPLAY_CONFIG_KEYS = (
    "agent_names",
    "prevention_ratio",
    "per_action_cap",
    "audit_episode_contract_version",
    "evaluation_protocol_version",
    "evaluation_implementation_fingerprint",
    "scoring_version",
    "success_contract_version",
    "trajectory_analysis_contract_version",
)
BUDGET_CONFIG_KEYS = (
    "max_replays",
    "max_replay_work_ticks",
    "exact_max_calls",
    "exact_max_replays",
)
BOUNDED_ISOLATION_WORKERS = {"sumo": 1}
SUCCESS_CONTRACT_VERSION = "task_outcome_process_and_prevention_v2"
CHECKPOINT_CONTRACT_VERSION = "complexity_result_shards_v1"
CHECKPOINT_CLEAR_ATTEMPTS = 5
CHECKPOINT_CLEAR_RETRY_SECONDS = 0.01


def _runtime_actuator_endpoints_for_actions(
    actions: list[Any], observed: dict[str, Any]
) -> list[str]:
    """Bind retained calls to actuator endpoints proven by runtime results."""
    observed_endpoints = set(
        observed.get("observed_physical_actuator_endpoint_set") or []
    )
    retained: set[str] = set()
    for action in actions:
        for call in getattr(action, "tool_calls", []):
            name = str(getattr(call, "name", "") or "")
            args = getattr(call, "args", {}) or {}
            endpoint_id = args.get("physical_actuator_id")
            if endpoint_id is None:
                endpoint_id = args.get("sumo_tls_id") or args.get("tls_id")
            if endpoint_id is None:
                continue
            token = f"{name}|{endpoint_id}"
            if token in observed_endpoints:
                retained.add(token)
    return sorted(retained)


def _finite_int_tick(value: Any) -> int | None:
    """Coerce a runtime tick without letting NaN/Inf crash calibration."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or numeric != math.trunc(numeric):
        return None
    try:
        return int(numeric)
    except (OverflowError, ValueError):
        return None


def _finite_tick_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return sorted(
        {tick for raw in value if (tick := _finite_int_tick(raw)) is not None}
    )


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    domains: set[str] | None,
    levels: set[str] | None,
    scenario_ids: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if (not domains or str(row.get("domain")) in domains)
        and (not levels or str(row.get("difficulty_level")) in levels)
        and (not scenario_ids or str(row.get("scenario_id")) in scenario_ids)
    ]
    return selected[:limit] if limit is not None else selected


def _compatible_config(current: dict[str, Any], prior: dict[str, Any]) -> bool:
    if not all(current.get(key) == prior.get(key) for key in REPLAY_CONFIG_KEYS):
        return False
    for key in BUDGET_CONFIG_KEYS:
        current_value = int(current.get(key, -1))
        if key == "max_replay_work_ticks" and key not in prior:
            continue
        if int(prior.get(key, -1)) < current_value:
            return False
    return True


def _passed_task_contract_ids(report: dict[str, Any]) -> set[str]:
    """Return only rows with a completed reference-policy task contract."""
    return {
        str(row["scenario_id"])
        for row in report.get("results") or []
        if row.get("scenario_id") and row.get("status") == "passed"
    }


def _task_contract_dependency_proof(
    *,
    scenario: dict[str, Any],
    task_completion: dict[str, Any],
    minimization_status: str,
    retained_decision_ticks: list[int],
) -> dict[str, Any]:
    """Prove ordered task-stage depth for an irreducible successful trace.

    This proof is intentionally restricted to task contracts with explicit
    ordered milestones, or multiple phase outcomes with a later control
    reversal. A generic phase-count proxy is not accepted.
    """
    contract = (scenario.get("backend_config") or {}).get("task_contract") or {}
    requirements = (scenario.get("backend_config") or {}).get("task_requirements") or {}
    phase_ticks = sorted({int(value) for value in contract.get("phase_ticks") or []})
    ordered_milestones = list(requirements.get("ordered_tool_milestones") or [])
    evidence = task_completion.get("evidence") or {}
    reductions = evidence.get("phase_reductions") or {}
    completed_phases = [
        tick for tick in phase_ticks if float(reductions.get(str(tick)) or 0.0) > 0.0
    ]
    retained_ticks = sorted({int(value) for value in retained_decision_ticks})
    phase_proven = bool(
        minimization_status == "one_minimal"
        and task_completion.get("completed")
        and len(phase_ticks) >= 2
        and completed_phases == phase_ticks
        and contract.get("reversal")
        and evidence.get("plan_reversal_observed") is True
        and len(retained_ticks) >= len(phase_ticks)
    )
    ordered_proven = bool(
        minimization_status == "one_minimal"
        and task_completion.get("completed")
        and ordered_milestones
        and evidence.get("ordered_tool_milestones_met") is True
        and len(evidence.get("selected_milestone_ticks") or [])
        == len(ordered_milestones)
        and len(retained_ticks) >= len(ordered_milestones)
    )
    selected_milestone_ticks = [
        int(value)
        for value in evidence.get("selected_milestone_ticks") or []
        if isinstance(value, (int, float))
    ]
    ordered_contract_met = bool(
        task_completion.get("completed")
        and ordered_milestones
        and evidence.get("ordered_tool_milestones_met") is True
        and len(selected_milestone_ticks) == len(ordered_milestones)
        and len(set(selected_milestone_ticks)) == len(selected_milestone_ticks)
    )
    required_depth_lower_bound = (
        len(ordered_milestones) if ordered_contract_met else None
    )
    proven_depth = (
        len(phase_ticks)
        if phase_proven
        else len(ordered_milestones)
        if ordered_proven
        else None
    )
    single_stage_proven = bool(
        minimization_status == "one_minimal"
        and task_completion.get("completed")
        and not phase_ticks
        and not ordered_milestones
        and retained_ticks
    )
    return {
        "exact_dependency_depth": (
            proven_depth
            if proven_depth is not None
            else (1 if single_stage_proven else None)
        ),
        "required_depth_lower_bound": required_depth_lower_bound,
        "depth_proof_kind": (
            "task_contract_ordered_milestone_lower_bound"
            if required_depth_lower_bound is not None
            else None
        ),
        "dependency_depth_status": (
            "one_minimal_task_contract_phase_dag"
            if proven_depth is not None
            else (
                "one_minimal_single_stage_action_dag"
                if single_stage_proven
                else (
                    "task_contract_ordered_milestone_lower_bound"
                    if required_depth_lower_bound is not None
                    else "task_contract_proof_incomplete"
                )
            )
        ),
        "task_contract_phase_ticks": phase_ticks,
        "ordered_tool_milestones": ordered_milestones,
        "ordered_tool_milestones_met": ordered_contract_met,
        "selected_milestone_ticks": selected_milestone_ticks,
        "completed_phase_ticks": completed_phases,
        "retained_decision_ticks": retained_ticks,
        "plan_reversal_observed": evidence.get("plan_reversal_observed"),
    }


def _normalize_minimization_result(result: dict[str, Any]) -> dict[str, Any]:
    """Upgrade legacy successful traces without strengthening their claim."""
    minimization = result.get("replay_minimization")
    if not isinstance(minimization, dict):
        return result
    if "successful_decision_tick_upper_bound" not in minimization:
        minimization["successful_decision_tick_upper_bound"] = minimization.get(
            "one_minimal_decision_ticks"
        )
        minimization["successful_tool_set_upper_bound"] = minimization.get(
            "one_minimal_successful_tool_set"
        )
        minimization["successful_distinct_tool_count_upper_bound"] = minimization.get(
            "one_minimal_distinct_tool_count"
        )
        minimization["successful_non_meta_call_count_upper_bound"] = minimization.get(
            "one_minimal_non_meta_call_count"
        )
    if minimization.get("status") != "one_minimal":
        minimization["claim"] = "bounded_successful_upper_bound_not_one_minimal"
        minimization["one_minimal_decision_ticks"] = None
        minimization["one_minimal_successful_tool_set"] = None
        minimization["one_minimal_distinct_tool_count"] = None
        minimization["one_minimal_non_meta_call_count"] = None
    return result


def _replay_success_contract(
    *,
    scenario: dict[str, Any],
    completion: dict[str, Any],
    actual_cost: float,
    prevented_loss: float,
    prevention_ratio: float,
    reference_completion: dict[str, Any] | None = None,
    reference_actual_cost: float | None = None,
    reference_prevented_loss: float | None = None,
) -> tuple[bool, float, float]:
    """Evaluate a replay against the reference objective in native units.

    Most contracts use aggregate economic cost, but native task contracts may
    intentionally trade cost for a safer or more feasible state.  In that
    case, using ``prevented_loss`` mixes incompatible units and incorrectly
    marks a successful native recovery as an unsuccessful baseline trace.
    The reference completion fixes the target once; each replay is then
    compared with that same target.
    """
    reference = reference_completion or completion
    process_capable = bool(
        completion.get("process_capability_applicable") is not True
        or completion.get("process_capability_satisfied") is True
    )
    reference_evidence = reference.get("evidence") or {}
    candidate_evidence = completion.get("evidence") or {}
    native_contract = str(reference.get("contract") or "") == (
        "microgrid.native_state_loss.v1"
    ) or str(completion.get("contract") or "") == ("microgrid.native_state_loss.v1")
    if native_contract:
        reference_task_loss = float(
            reference_evidence.get("counterfactual_task_loss") or 0.0
        )
        reference_reduction = float(
            reference_evidence.get("task_loss_reduction") or 0.0
        )
        target = reference_task_loss - prevention_ratio * reference_reduction
        candidate_task_loss = float(candidate_evidence.get("actual_task_loss") or 0.0)
        success = (
            bool(completion.get("completed"))
            and process_capable
            and (candidate_task_loss <= target + 1e-9)
        )
        return success, target, candidate_task_loss

    reference_actual = (
        float(reference_actual_cost)
        if reference_actual_cost is not None
        else float(actual_cost)
    )
    reference_prevented = (
        float(reference_prevented_loss)
        if reference_prevented_loss is not None
        else float(prevented_loss)
    )
    reference_counterfactual = reference_actual + reference_prevented
    target = reference_counterfactual - prevention_ratio * reference_prevented
    success = (
        bool(completion.get("completed"))
        and process_capable
        and (float(prevented_loss) > 0.0 and float(actual_cost) <= target + 1e-9)
    )
    return success, target, float(actual_cost)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _checkpoint_root(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".checkpoint")


def _checkpoint_fingerprint(
    *,
    config: dict[str, Any],
    implementation_tree_sha256: str,
    core_release_pipeline_sha256: str,
) -> str:
    payload = {
        "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
        "config": config,
        "implementation_tree_sha256": implementation_tree_sha256,
        "core_release_pipeline_sha256": core_release_pipeline_sha256,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_checkpoint(
    output_path: Path,
    *,
    config: dict[str, Any],
    implementation_tree_sha256: str,
    core_release_pipeline_sha256: str,
) -> Path:
    """Create or reopen the exact-run checkpoint shard directory."""
    root = _checkpoint_root(output_path)
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError(f"complexity checkpoint root is not a directory: {root}")
    fingerprint = _checkpoint_fingerprint(
        config=config,
        implementation_tree_sha256=implementation_tree_sha256,
        core_release_pipeline_sha256=core_release_pipeline_sha256,
    )
    directory = root / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(
            f"complexity checkpoint generation is not a directory: {directory}"
        )
    metadata = {
        "schema_version": "1.0",
        "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
        "fingerprint": fingerprint,
        "config": config,
        "implementation_tree_sha256": implementation_tree_sha256,
        "core_release_pipeline_sha256": core_release_pipeline_sha256,
    }
    metadata_path = directory / "manifest.json"
    if metadata_path.exists():
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        if observed != metadata:
            raise ValueError("complexity checkpoint manifest mismatch")
    else:
        _atomic_write(metadata_path, metadata)
    return directory


def _checkpoint_result_path(directory: Path, key: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return directory / f"result-{digest}.json"


def _write_checkpoint_result(
    directory: Path,
    *,
    key: str,
    result: dict[str, Any],
) -> None:
    _atomic_write(
        _checkpoint_result_path(directory, key),
        {
            "schema_version": "1.0",
            "checkpoint_contract_version": CHECKPOINT_CONTRACT_VERSION,
            "result_key": key,
            "result": result,
        },
    )


def _load_checkpoint_results(
    directory: Path,
    *,
    expected_signatures: dict[str, str],
    agent_names: set[str],
) -> dict[str, dict[str, Any]]:
    """Load only complete, identity-matching shards from this exact run."""
    restored: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("result-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or payload.get("checkpoint_contract_version") != CHECKPOINT_CONTRACT_VERSION
            or not isinstance(payload.get("result"), dict)
        ):
            raise ValueError(f"invalid complexity checkpoint shard: {path}")
        result = payload["result"]
        scenario_id = str(result.get("scenario_id") or "")
        agent_name = str(result.get("agent_name") or "")
        key = f"{scenario_id}::{agent_name}"
        if payload.get("result_key") != key or path != _checkpoint_result_path(
            directory, key
        ):
            raise ValueError(f"complexity checkpoint shard identity mismatch: {path}")
        if (
            result.get("status") == "complete"
            and scenario_id in expected_signatures
            and agent_name in agent_names
            and str(result.get("scenario_signature") or "")
            == expected_signatures[scenario_id]
        ):
            restored[key] = _normalize_minimization_result(result)
    return restored


def _clear_checkpoint(output_path: Path) -> None:
    root = _checkpoint_root(output_path)
    for attempt in range(CHECKPOINT_CLEAR_ATTEMPTS):
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise ValueError(f"complexity checkpoint root is not a directory: {root}")
        if not root.exists():
            return
        try:
            shutil.rmtree(root)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if (
                exc.errno not in {errno.EBUSY, errno.ENOTEMPTY}
                or attempt == CHECKPOINT_CLEAR_ATTEMPTS - 1
            ):
                raise
            time.sleep(CHECKPOINT_CLEAR_RETRY_SECONDS * (attempt + 1))


def _graph_acyclic(graph: dict[str, Any]) -> bool:
    nodes = [
        str(node.get("id") or "")
        for node in graph.get("nodes") or []
        if isinstance(node, dict)
    ]
    if not nodes or not all(nodes) or len(nodes) != len(set(nodes)):
        return False
    adjacency = {node: [] for node in nodes}
    indegree = {node: 0 for node in nodes}
    for edge in graph.get("edges") or []:
        if not isinstance(edge, dict):
            return False
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source not in adjacency or target not in adjacency:
            return False
        adjacency[source].append(target)
        indegree[target] += 1
    frontier = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while frontier:
        node = frontier.pop()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
    return visited == len(nodes)


def _build_agentic_evidence(
    *,
    scenario: dict[str, Any] | None = None,
    loop: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, Any]:
    """Export only runtime-observed clock, event, control, and graph evidence."""
    scenario = scenario or {}
    autonomy = loop.get("event_adaptive_autonomy") or {}
    records = [row for row in autonomy.get("records") or [] if isinstance(row, dict)]
    final_ground_truth = loop.get("final_ground_truth") or {}
    final_control_summary = final_ground_truth.get("control_summary") or {}
    persistent_policy_review_records = [
        dict(row)
        for row in final_control_summary.get("policy_review_ledger") or []
        if isinstance(row, dict)
    ]
    persistent_policy_review_bindings: list[dict[str, Any]] = []
    persistent_policy_effect_bindings: list[dict[str, Any]] = []
    try:
        capability = get_backend_capability(scenario.get("backend_kind"))
        control_tools = tuple(capability.control_tools)
        persistent_control_tools = set(map(str, capability.persistent_control_tools))
        clock_semantics = str(capability.clock_semantics)
        source_scheduled_event_types = tuple(capability.source_scheduled_event_types)
    except KeyError:
        # Legacy diagnostics may omit a descriptor. Production preflight
        # rejects that case; here we retain only directly observed controls.
        control_tools = tuple(
            map(
                str,
                observed.get("observed_state_changing_tool_set") or (),
            )
        )
        persistent_control_tools = set()
        clock_semantics = ""
        source_scheduled_event_types = ()
    wake_reason_counts = Counter(
        str(reason)
        for row in records
        if row.get("kind") in {"early_wake", "mandatory_interrupt_budget_overrun"}
        for reason in row.get("reasons") or []
    )
    state_changing_calls = 0
    successful_control_names: set[str] = set()
    persistent_control_names_with_effect: set[str] = set()
    agent_state_ticks: set[int] = set()
    persistent_control_ticks: set[int] = set()
    realized_event_ids: list[str] = []
    realized_event_ticks: set[int] = set()
    for step in loop.get("analysis_steps") or []:
        tick = _finite_int_tick(step.get("tick"))
        if tick is None:
            tick = 0
        for result in step.get("tool_results") or []:
            if (
                isinstance(result, dict)
                and result.get("ok") is True
                and result.get("state_changing") is True
                and str((result.get("payload") or {}).get("_status") or "") != "pending"
            ):
                state_changing_calls += 1
                result_name = str(result.get("name") or "")
                successful_control_names.add(result_name)
                agent_state_ticks.add(tick)
                if (
                    result_name in persistent_control_tools
                    and _finite_int_tick(result.get("effect_tick")) is not None
                    and bool(result.get("produces_evidence_ids"))
                ):
                    persistent_control_ticks.add(tick)
                    persistent_control_names_with_effect.add(result_name)
            if not isinstance(result, dict) or result.get("ok") is not True:
                continue
            result_name = str(result.get("name") or "")
            payload = result.get("payload") or {}
            result_evidence_ids = {
                str(value)
                for value in [
                    result.get("evidence_id"),
                    *(result.get("produces_evidence_ids") or []),
                ]
                if str(value)
            }
            trace_edges = list(step.get("tool_trace_edges") or [])
            if not trace_edges:
                trace_edges = list(
                    (step.get("info") or {}).get("extra", {}).get("tool_trace_edges")
                    or []
                )
            call_id = str(result.get("call_id") or "")
            graph_evidence_ids = {
                str(value)
                for edge in trace_edges
                if isinstance(edge, dict) and str(edge.get("call_id") or "") == call_id
                for value in edge.get("produces_evidence_ids") or []
                if str(value)
            }
            if (
                result_name == "review_persistent_policy"
                and isinstance(payload, dict)
                and payload.get("review_status") == "accepted"
                and str(payload.get("review_id") or "")
                and call_id
            ):
                persistent_policy_review_bindings.append(
                    {
                        "review_id": str(payload["review_id"]),
                        "review_tool_name": result_name,
                        "call_id": call_id,
                        "accepted": True,
                        "evidence_ids": sorted(result_evidence_ids),
                        "action_graph_evidence_ids": sorted(graph_evidence_ids),
                    }
                )
            if (
                result_name in persistent_control_tools
                and _finite_int_tick(result.get("effect_tick")) is not None
                and result_evidence_ids
                and isinstance(payload, dict)
            ):
                generation = _finite_int_tick(payload.get("policy_generation"))
                if generation is not None and generation > 0:
                    persistent_policy_effect_bindings.append(
                        {
                            "policy_generation": generation,
                            "policy_tool_name": result_name,
                            "accepted": True,
                            "effect_tick": _finite_int_tick(result.get("effect_tick")),
                            "call_id": call_id,
                            "evidence_ids": sorted(result_evidence_ids),
                        }
                    )
        info = step.get("info") or {}
        for event in list(info.get("realized_events") or []) + list(
            info.get("fault_injections") or []
        ):
            if not isinstance(event, dict):
                continue
            event_id = str(
                event.get("event_id")
                or event.get("id")
                or event.get("type")
                or event.get("kind")
                or ""
            )
            if event_id:
                realized_event_ids.append(event_id)
                realized_event_ticks.add(tick)
    graph = observed.get("evidence_action_graph") or {}
    visible_interrupt_count = sum(
        row.get("kind") in {"early_wake", "mandatory_interrupt_budget_overrun"}
        for row in records
    )
    declared_events = [
        event
        for event in scenario.get("perturbations") or []
        if isinstance(event, dict)
    ]
    declared_event_ids = [
        str(
            event.get("event_id")
            or event.get("id")
            or event.get("kind")
            or event.get("type")
            or ""
        )
        for event in declared_events
    ]
    declared_event_ticks = sorted(
        {
            tick
            for event in declared_events
            if (tick := _finite_int_tick(event.get("trigger_tick"))) is not None
        }
    )
    hold_kinds = {
        "pending_action_hold",
        "autonomous_hold",
        "native_idle_hold",
        "model_decision_budget_exhausted",
    }
    simulator_advance_without_model_ticks = sorted(
        {
            tick
            for row in records
            if str(row.get("kind") or "") in hold_kinds
            if (tick := _finite_int_tick(row.get("tick"))) is not None
        }
    )
    source_evidence = loop.get("source_consumption_evidence") or {}
    source_ticks = list(source_evidence.get("consumption_ticks") or [])
    interaction_stats = loop.get("llm_interaction_stats") or {}
    scheduled_review_ticks = _finite_tick_list(autonomy.get("scheduled_review_ticks"))
    periodic_scan_ticks = _finite_tick_list(autonomy.get("periodic_scan_ticks"))
    standing_plan_commit_ticks = sorted(
        {
            tick
            for record in records
            if str(record.get("kind") or "") == "autonomy_window_opened"
            and str(record.get("schedule_status") or "") == "accepted"
            if (tick := _finite_int_tick(record.get("tick"))) is not None
        }
    )
    explicit_continuous_review_ticks = _finite_tick_list(
        autonomy.get("continuous_supervisory_review_ticks")
        or observed.get("continuous_supervisory_review_ticks")
    )
    legacy_environment_ticks = _finite_tick_list(
        observed.get("environment_change_ticks")
    )
    simulator_tick_sequence = []
    for index, step in enumerate(loop.get("analysis_steps") or []):
        if not isinstance(step, dict):
            continue
        simulator_tick_sequence.append(
            _finite_int_tick(step.get("tick"))
            if _finite_int_tick(step.get("tick")) is not None
            else index
        )
    simulator_clock_progressed = bool(
        len(simulator_tick_sequence) > 1
        and all(
            current > previous
            for previous, current in zip(
                simulator_tick_sequence,
                simulator_tick_sequence[1:],
                strict=False,
            )
        )
    )
    event_to_action_edges = validated_event_action_edges(
        observed.get("event_to_action_edges"),
        material_exogenous_records=observed.get("material_exogenous_event_records"),
        agent_caused_records=observed.get("agent_caused_event_records"),
    )
    return {
        "simulator_ticks": int(observed.get("simulator_ticks") or 0),
        "clock_semantics": clock_semantics,
        "simulator_tick_sequence": simulator_tick_sequence,
        "simulator_owned_clock_observed": bool(
            clock_semantics == "simulator_owned" and simulator_clock_progressed
        ),
        "model_decision_ticks": int(autonomy.get("model_decision_ticks") or 0),
        "provider_calls": int(interaction_stats.get("llm_calls_ok") or 0)
        + int(interaction_stats.get("llm_calls_failed") or 0)
        + int(interaction_stats.get("llm_fc_retries") or 0),
        "simulator_advance_without_model_ticks": (
            simulator_advance_without_model_ticks
        ),
        "autonomous_hold_ticks": int(autonomy.get("autonomous_hold_ticks") or 0),
        "pending_action_hold_ticks": int(
            autonomy.get("pending_action_hold_ticks") or 0
        ),
        "wake_reason_counts": dict(sorted(wake_reason_counts.items())),
        "visible_interrupt_count": int(visible_interrupt_count),
        "declared_predesigned_event_ids": declared_event_ids,
        "declared_predesigned_event_ticks": declared_event_ticks,
        "realized_predesigned_event_ids": sorted(set(realized_event_ids)),
        "realized_predesigned_event_ticks": sorted(realized_event_ticks),
        "exogenous_state_change_ticks": sorted(
            realized_event_ticks.union(legacy_environment_ticks)
        ),
        "agent_caused_state_change_ticks": sorted(agent_state_ticks),
        "source_consumption_ticks": source_ticks,
        "realized_event_ticks": sorted(
            realized_event_ticks.union(legacy_environment_ticks)
        ),
        "state_change_ticks": list(observed.get("state_change_ticks") or []),
        "state_changing_tool_calls": state_changing_calls,
        "available_native_control_tool_names": list(control_tools),
        "successful_native_control_tool_names": sorted(
            successful_control_names.intersection(control_tools)
        ),
        "native_control_tool_names": list(control_tools),
        "decision_opportunity_ticks": list(
            autonomy.get("decision_opportunity_ticks") or []
        ),
        # Keep the actual supervision edges, not just the backend's cadence
        # declaration. The agentic diagnostic must distinguish a standing plan
        # that was genuinely reviewed from a one-shot reactive action.
        "scheduled_review_ticks": sorted(set(scheduled_review_ticks)),
        "periodic_scan_ticks": sorted(set(periodic_scan_ticks)),
        # Continuous supervision must be emitted by an explicit runtime review
        # record. Consecutive model decisions plus a one-shot action are not a
        # standing scan/hold diagnostic.
        "continuous_supervisory_review_ticks": explicit_continuous_review_ticks,
        "actual_supervisory_review_observed": bool(
            scheduled_review_ticks
            or periodic_scan_ticks
            or explicit_continuous_review_ticks
        ),
        "standing_plan_committed": any(standing_plan_commit_ticks),
        "standing_plan_commit_ticks": standing_plan_commit_ticks,
        # Only backend-declared persistent controls can substitute for the
        # optional meta-tool.  One-shot dispatch/shed/repair calls remain
        # ordinary actions and cannot masquerade as a standing plan.
        "standing_control_commit_ticks": sorted(persistent_control_ticks),
        "persistent_control_tool_names": sorted(
            persistent_control_names_with_effect.intersection(persistent_control_tools)
        ),
        "persistent_policy_review_records": persistent_policy_review_records,
        "persistent_policy_review_bindings": persistent_policy_review_bindings,
        "persistent_policy_effect_bindings": persistent_policy_effect_bindings,
        "persistent_policy_attribution": dict(
            loop.get("persistent_policy_attribution") or {}
        ),
        "decision_graph_nodes": len(graph.get("nodes") or []),
        "decision_graph_edges": len(graph.get("edges") or []),
        "decision_graph_acyclic": _graph_acyclic(graph),
        "event_adaptive_cadence_declared": bool(
            autonomy.get("cadence_contract_declared")
        ),
        "periodic_cadence_observed": bool(
            autonomy.get("cadence_contract_declared")
            and (scheduled_review_ticks or periodic_scan_ticks)
        ),
        "world_change_contract_declared": bool(
            declared_events or source_scheduled_event_types
        ),
        "material_exogenous_event_records": list(
            observed.get("material_exogenous_event_records") or []
        ),
        "source_scheduled_event_records": list(
            observed.get("source_scheduled_event_records") or []
        ),
        "source_scheduled_change_records": list(
            observed.get("source_scheduled_event_records") or []
        ),
        "declared_perturbation_event_records": list(
            observed.get("declared_perturbation_event_records") or []
        ),
        "endogenous_completion_event_records": list(
            observed.get("endogenous_completion_event_records") or []
        ),
        "endogenous_completion_records": list(
            observed.get("endogenous_completion_event_records") or []
        ),
        "post_change_decision_ticks": list(
            observed.get("post_change_decision_ticks") or []
        ),
        "event_to_decision_action_edges": list(
            observed.get("event_to_decision_action_edges") or []
        ),
        "event_to_action_edges": [dict(row) for row in event_to_action_edges],
        "action_to_outcome_edges": [
            dict(row.get("action_to_outcome_edge") or {})
            for row in observed.get("agent_caused_event_records") or []
            if row.get("action_to_outcome_edge")
        ],
        "agent_action_effect_records": list(
            observed.get("agent_caused_event_records") or []
        ),
        "adaptive_replanning_observed": bool(
            observed.get("adaptive_replanning_observed") and event_to_action_edges
        ),
        "valid_plan_delegation_observed": bool(
            int(autonomy.get("delegated_plan_opportunities") or 0) > 0
        ),
        "delegated_plan_opportunity_ticks": [
            parsed_tick
            for raw_tick in autonomy.get("delegated_plan_opportunity_ticks") or []
            if (parsed_tick := _finite_int_tick(raw_tick)) is not None
        ],
        "agent_action_backend_effect_observed": bool(
            observed.get("agent_action_backend_effect_observed")
            or (
                observed.get("effective_control_tick_status")
                == "counterfactual_attributed"
                and int(observed.get("n_effective_control_ticks") or 0) > 0
            )
        ),
    }


def _collect_actions(scenario: dict[str, Any], agent_name: str) -> dict[str, Any]:
    seed = int(scenario.get("seed", 42))
    spec = get_domain_spec(scenario.get("domain"))
    env = spec.env_factory()()
    try:
        env.reset(scenario, seed=seed)
        agent = make_agent(agent_name)
        agent.reset(env, scenario, seed=seed)
        loop = _run_episode_loop(env=env, agent=agent, logger=None)
        if hasattr(agent, "get_interaction_stats"):
            loop["llm_interaction_stats"] = agent.get_interaction_stats()
        loop["source_consumption_evidence"] = env.source_consumption_evidence(
            scenario=scenario,
        )
        loop["final_ground_truth"] = env.ground_truth()
        return loop
    finally:
        env.close()


def _persistent_policy_state_digest(ground_truth: dict[str, Any]) -> str:
    """Hash policy-independent native outcomes for policy attribution.

    The active policy name and generation are metadata about the treatment, not
    an outcome.  Including them here would make every actual-vs-masked replay
    look causal even when dispatch order and costs are identical.
    """
    payload = _persistent_policy_outcome_payload(ground_truth)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _persistent_policy_queue_digest(ground_truth: dict[str, Any]) -> str:
    """Hash policy-independent arrived-job ordering and statuses."""
    outcome = _persistent_policy_outcome_payload(ground_truth)
    payload = {
        "dispatch_order": outcome["dispatch_order"],
        "running_job_ids": outcome["running_job_ids"],
        "queued_jobs": outcome["queued_jobs"],
        "job_status": [
            {
                "job_id": row["job_id"],
                "status": row["status"],
                "dispatch_order": row["dispatch_order"],
                "wait_ticks": row["wait_ticks"],
            }
            for row in outcome["jobs"]
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _persistent_policy_outcome_payload(
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Return policy-independent queue, state, and cost outcomes."""
    queue = ground_truth.get("queue") or {}
    jobs = ground_truth.get("jobs") or {}
    job_rows = []
    for job_id, raw_job in sorted(jobs.items()):
        if not isinstance(raw_job, dict):
            continue
        job_rows.append(
            {
                "job_id": str(job_id),
                "status": str(raw_job.get("status") or ""),
                "submit_tick": raw_job.get("submit_tick"),
                "due_tick": raw_job.get("due_tick"),
                "remaining_ticks": raw_job.get("remaining_ticks"),
                "wait_ticks": raw_job.get("wait_ticks"),
                "dispatch_order": raw_job.get("dispatch_order"),
                "preemptions": raw_job.get("preemptions"),
            }
        )
    queued_jobs = []
    for raw_job in queue.get("queued_jobs") or []:
        if not isinstance(raw_job, dict):
            continue
        queued_jobs.append(
            {
                key: raw_job.get(key)
                for key in (
                    "job_id",
                    "remaining_ticks",
                    "gpu_units",
                    "criticality",
                    "due_tick",
                    "dispatch_order",
                )
            }
        )
    capacity = ground_truth.get("capacity") or {}
    capacity_outcome = {
        key: capacity.get(key)
        for key in (
            "gpu_capacity_units",
            "cpu_capacity_units",
            "gpu_allocated_units",
            "cpu_allocated_units",
            "capacity_factor",
            "reserved_gpu_units",
            "pending_reserved_gpu_units",
        )
    }
    return {
        "dispatch_order": list(queue.get("dispatch_order") or []),
        "running_job_ids": list(queue.get("running_job_ids") or []),
        "queued_jobs": queued_jobs,
        "jobs": job_rows,
        "capacity": capacity_outcome,
        "cost_components": dict(ground_truth.get("cost_components") or {}),
    }


def _persistent_policy_mask(action: Action, _tick: int) -> Action:
    """Mask queue-policy changes while retaining review and other controls."""
    return Action(
        tool_calls=[
            call for call in action.tool_calls if call.name != "set_queue_policy"
        ],
        dominant=action.dominant,
        assistant_text=action.assistant_text,
        rationale=action.rationale,
    )


def _persistent_policy_attribution(
    *,
    scenario: dict[str, Any],
    seed: int,
    actions: list[Action],
    final_ground_truth: dict[str, Any],
) -> dict[str, Any]:
    """Run a bounded actual-vs-policy-masked replay for a reviewed policy."""
    summary = final_ground_truth.get("control_summary") or {}
    ledger = [
        row
        for row in summary.get("policy_review_ledger") or []
        if isinstance(row, dict)
        and str(row.get("review_id") or "")
        and row.get("outcome_effect_ticks")
    ]
    if not ledger or not any(
        any(call.name == "set_queue_policy" for call in action.tool_calls)
        for action in actions
    ):
        return {}
    spec = get_domain_spec(scenario.get("domain"))
    probe = spec.env_factory()()
    try:
        probe.reset(scenario, seed)
        readonly_tool_names = probe.readonly_tool_names()
    finally:
        probe.close()
    replay = run_counterfactual(
        env_factory=spec.env_factory(),
        scenario_config=scenario,
        seed=seed,
        actual_actions=actions,
        cost_extractor=domain_cost_extractor,
        masking_policy=_persistent_policy_mask,
        masking_label="persistent_policy_mask",
        readonly_tool_names=readonly_tool_names,
    )
    actual_ground_truth = replay.actual_ground_truth or {}
    masked_ground_truth = replay.counterfactual_ground_truth or {}
    actual_ledger = list(
        (actual_ground_truth.get("control_summary") or {}).get("policy_review_ledger")
        or []
    )
    masked_ledger = list(
        (masked_ground_truth.get("control_summary") or {}).get("policy_review_ledger")
        or []
    )

    def _review_for(rows: list[object], review_id: str) -> dict[str, Any] | None:
        return next(
            (
                row
                for row in rows
                if isinstance(row, dict)
                and str(row.get("review_id") or "") == review_id
            ),
            None,
        )

    def _lineage_verified(candidate: dict[str, Any]) -> bool:
        review_id = str(candidate.get("review_id") or "")
        replayed = _review_for(actual_ledger, review_id)
        masked = _review_for(masked_ledger, review_id)
        return bool(
            actual_ground_truth
            and isinstance(replayed, dict)
            and replayed.get("evidence_ids")
            and replayed.get("policy_effect_evidence_id")
            and replayed.get("outcome_effect_ticks")
            and int(replayed.get("policy_generation") or 0)
            == int(candidate.get("policy_generation") or 0)
            and str(replayed.get("policy_tool_name") or "")
            == str(candidate.get("policy_tool_name") or "")
            and str(replayed.get("policy_effect_evidence_id") or "")
            == str(candidate.get("policy_effect_evidence_id") or "")
            and not (
                isinstance(masked, dict)
                and masked.get("evidence_ids")
                and masked.get("policy_effect_evidence_id")
            )
        )

    # Do not let an earlier stale review hide a later replayable review.  Keep
    # the first ledger row as a held diagnostic when no row has full lineage.
    review = next(
        (candidate for candidate in ledger if _lineage_verified(candidate)),
        ledger[0],
    )
    replay_lineage_verified = _lineage_verified(review)
    actual_queue_digest = _persistent_policy_queue_digest(actual_ground_truth)
    masked_queue_digest = _persistent_policy_queue_digest(masked_ground_truth)
    actual_state_digest = _persistent_policy_state_digest(actual_ground_truth)
    masked_state_digest = _persistent_policy_state_digest(masked_ground_truth)
    material_delta = float(replay.counterfactual_cost - replay.actual_cost)
    threshold = max(1.0, abs(float(replay.actual_cost)) * 0.001)
    passed = bool(
        replay.applicable
        and replay_lineage_verified
        and material_delta > threshold
        and actual_state_digest != masked_state_digest
        and actual_queue_digest != masked_queue_digest
    )
    return {
        "status": "passed" if passed else "held",
        "review_id": str(review["review_id"]),
        "selected_review_id": str(review["review_id"]),
        "event_ids": [str(value) for value in review.get("event_ids") or []],
        "review_tool_name": str(review.get("review_tool_name") or ""),
        "policy_tool_name": str(review.get("policy_tool_name") or ""),
        "policy_generation": int(review.get("policy_generation") or 0),
        "policy_effect_evidence_id": str(review.get("policy_effect_evidence_id") or ""),
        "deterministic_replay": bool(replay.applicable),
        "replay_lineage_verified": replay_lineage_verified,
        "actual_state_digest": actual_state_digest,
        "masked_state_digest": masked_state_digest,
        "actual_queue_order_digest": actual_queue_digest,
        "masked_queue_order_digest": masked_queue_digest,
        "material_delta": material_delta,
        "materiality_threshold": threshold,
        "effect_ticks": [
            int(value) for value in review.get("outcome_effect_ticks") or []
        ],
        "reason_code": (
            ""
            if passed
            else (
                "persistent_policy_replay_lineage_unproven"
                if not replay_lineage_verified
                else "persistent_policy_attribution_unproven"
            )
        ),
    }


def _bounded_reduced_trace_proof(
    *,
    full_trace_non_meta_call_count: int,
    reduced_trace_status: str,
    reduced_trace_tool_set: list[str],
    reduced_trace_non_meta_call_count: int,
    reduced_trace_decision_ticks: list[int],
    has_successful_upper_bound: bool,
    exact_found: bool,
    exact_max_calls: int,
    exact_max_replays: int,
) -> dict[str, Any] | None:
    """Describe a reduced trace when full exhaustive proof is budgeted out.

    The record is diagnostic only.  In particular, it never changes the
    existing ``trace_too_long`` status or turns a 1-minimal trace into a
    global-shortest claim.  ``exact_found`` is supplied by the caller so this
    helper cannot accidentally emit a bounded record alongside an exact proof.
    """
    if not has_successful_upper_bound or exact_found:
        return None
    reason = (
        "source_call_cap_exceeded"
        if full_trace_non_meta_call_count > exact_max_calls
        else "replay_budget_exhausted"
    )
    proof_kind = (
        "one_minimal"
        if reduced_trace_status == "one_minimal"
        else "bounded_successful_upper_bound"
    )
    return {
        "status": "bounded_reduced_trace_candidate",
        "proof_scope": "reduced_trace_only",
        "global_minimum_proven": False,
        "full_trace_exhaustive_status": "trace_too_long",
        "exhaustive_budget_reason": reason,
        "configured_max_calls": int(exact_max_calls),
        "configured_max_replays": int(exact_max_replays),
        "full_trace_non_meta_call_count": int(full_trace_non_meta_call_count),
        "reduced_trace_proof_kind": proof_kind,
        "reduced_trace_status": reduced_trace_status,
        "reduced_trace_tool_set": list(reduced_trace_tool_set),
        "reduced_trace_non_meta_call_count": int(reduced_trace_non_meta_call_count),
        "reduced_trace_decision_ticks": list(reduced_trace_decision_ticks),
    }


def _calibrate_one(
    row: dict[str, Any],
    *,
    agent_name: str,
    prevention_ratio: float,
    max_replays: int,
    max_replay_work_ticks: int,
    per_action_cap: int | None,
    exact_max_calls: int,
    exact_max_replays: int,
    replay_cache_path: Path,
) -> dict[str, Any]:
    path = _resolve_scenario_path(row["path"])
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    identity = implementation_identity()
    implementation_tree_sha256 = identity["implementation_tree_sha256"]
    core_release_pipeline_sha256 = identity["core_release_pipeline_sha256"]
    seed = int(scenario.get("seed", 42))
    spec = get_domain_spec(scenario.get("domain"))
    loop = _collect_actions(scenario, agent_name)
    actions = loop["actions"]
    analysis_steps = loop["analysis_steps"]
    report = domain_counterfactual_report(
        spec.env_factory(),
        scenario,
        seed,
        actions,
        per_action=True,
        per_action_cap=per_action_cap,
    )
    persistent_policy_attribution = _persistent_policy_attribution(
        scenario=scenario,
        seed=seed,
        actions=actions,
        final_ground_truth=loop.get("final_ground_truth") or {},
    )
    if persistent_policy_attribution:
        loop["persistent_policy_attribution"] = persistent_policy_attribution
    observed = analyze_trajectory_steps(
        analysis_steps,
        per_action_attribution=(
            report.per_action
            if report.per_action_status in {"complete", "capped"}
            else None
        ),
    )
    task_counterfactual = report.to_dict()
    task_counterfactual["_counterfactual_task_tick_records"] = (
        report.counterfactual_ground_truth.get("_task_tick_records") or []
    )
    reference_task_completion = separate_task_outcome_and_process(
        evaluate_task_completion(
            scenario=scenario,
            ground_truth=report.actual_ground_truth,
            counterfactual=task_counterfactual,
            score={"dimensions": []},
        ),
        scenario=scenario,
    )

    _, target_metric, _ = _replay_success_contract(
        scenario=scenario,
        completion=reference_task_completion,
        actual_cost=report.actual_cost,
        prevented_loss=report.prevented_loss,
        prevention_ratio=prevention_ratio,
    )
    native_task_loss_replay = (
        str(reference_task_completion.get("contract") or "")
        == "microgrid.native_state_loss.v1"
    )
    replay_outcomes: dict[str, dict[str, Any]] = {}
    if replay_cache_path.exists():
        cached = json.loads(replay_cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("schema_version") == "0.2"
            and cached.get("success_contract_version") == SUCCESS_CONTRACT_VERSION
            and cached.get("implementation_tree_sha256") == implementation_tree_sha256
            and cached.get("core_release_pipeline_sha256")
            == core_release_pipeline_sha256
            and cached.get("scenario_signature") == row["scenario_signature"]
        ):
            replay_outcomes = {
                str(key): dict(value)
                for key, value in (cached.get("outcomes") or {}).items()
                if isinstance(value, dict)
            }

    def succeeds(candidate: list[Any]) -> bool:
        key = repr(
            [
                [(call.name, sorted(call.args.items())) for call in action.tool_calls]
                for action in candidate
            ]
        )
        if key not in replay_outcomes:
            candidate_report = domain_counterfactual_report(
                spec.env_factory(), scenario, seed, candidate
            )
            task_counterfactual = candidate_report.to_dict()
            task_counterfactual["_counterfactual_task_tick_records"] = (
                candidate_report.counterfactual_ground_truth.get("_task_tick_records")
                or []
            )
            completion = separate_task_outcome_and_process(
                evaluate_task_completion(
                    scenario=scenario,
                    ground_truth=candidate_report.actual_ground_truth,
                    counterfactual=task_counterfactual,
                    score={"dimensions": []},
                ),
                scenario=scenario,
            )
            success, _, metric = _replay_success_contract(
                scenario=scenario,
                completion=completion,
                actual_cost=candidate_report.actual_cost,
                prevented_loss=candidate_report.prevented_loss,
                prevention_ratio=prevention_ratio,
                reference_completion=reference_task_completion,
                reference_actual_cost=report.actual_cost,
                reference_prevented_loss=report.prevented_loss,
            )
            replay_outcomes[key] = {
                "cost": float(candidate_report.actual_cost),
                "success_metric": float(metric),
                "success": bool(success),
                "task_completed": bool(completion.get("completed")),
                "task_reason_code": completion.get("reason_code"),
            }
            _atomic_write(
                replay_cache_path,
                {
                    "schema_version": "0.2",
                    "success_contract_version": SUCCESS_CONTRACT_VERSION,
                    "scenario_id": row["scenario_id"],
                    "scenario_signature": row["scenario_signature"],
                    "agent_name": agent_name,
                    "implementation_tree_sha256": implementation_tree_sha256,
                    "core_release_pipeline_sha256": (core_release_pipeline_sha256),
                    "outcomes": replay_outcomes,
                },
            )
        outcome = replay_outcomes[key]
        if "success" in outcome:
            return bool(outcome["success"])
        if native_task_loss_replay:
            # A cache written before native-unit replay semantics is not
            # evidence for the new contract, even if it contains a task ack.
            return False
        return (
            bool(report.applicable)
            and float(report.prevented_loss) > 0
            and (
                bool(outcome.get("task_completed"))
                and float(outcome["cost"]) <= target_metric + 1e-9
            )
        )

    horizon_ticks = max(1, int(scenario.get("horizon_ticks") or len(actions) or 1))
    effective_max_replays = max(
        1,
        min(
            max_replays,
            max(1, int(max_replay_work_ticks)) // horizon_ticks,
        ),
    )
    minimized = minimize_successful_action_sequence(
        actions,
        succeeds,
        max_replays=effective_max_replays,
    )
    has_successful_upper_bound = minimized.status != "initial_trace_not_successful"
    exact = (
        exhaustive_trace_subset_minimum(
            actions,
            succeeds,
            max_calls=exact_max_calls,
            max_replays=exact_max_replays,
        )
        if has_successful_upper_bound
        else None
    )
    non_meta = {"wait", "noop"}
    retained_ticks = [
        index
        for index, action in enumerate(minimized.actions)
        if any(call.name not in non_meta for call in action.tool_calls)
    ]
    retained_calls = sum(
        call.name not in non_meta
        for action in minimized.actions
        for call in action.tool_calls
    )
    full_trace_non_meta_call_count = sum(
        call.name not in non_meta for action in actions for call in action.tool_calls
    )
    exact_calls = (
        sum(
            call.name not in non_meta
            for action in exact.actions
            for call in action.tool_calls
        )
        if exact and exact.status == "trace_subset_global_minimum"
        else None
    )
    lower_bound = (
        exact_calls
        if exact_calls is not None
        else (1 if has_successful_upper_bound else None)
    )
    one_minimal_proven = minimized.status == "one_minimal"
    minimized_actuator_endpoints = _runtime_actuator_endpoints_for_actions(
        minimized.actions,
        observed,
    )
    bounded_reduced_trace_proof = _bounded_reduced_trace_proof(
        full_trace_non_meta_call_count=full_trace_non_meta_call_count,
        reduced_trace_status=minimized.status,
        reduced_trace_tool_set=minimized.tool_set,
        reduced_trace_non_meta_call_count=retained_calls,
        reduced_trace_decision_ticks=retained_ticks,
        has_successful_upper_bound=has_successful_upper_bound,
        exact_found=exact is not None,
        exact_max_calls=exact_max_calls,
        exact_max_replays=exact_max_replays,
    )
    dependency_proof = _task_contract_dependency_proof(
        scenario=scenario,
        task_completion=reference_task_completion,
        minimization_status=minimized.status,
        retained_decision_ticks=retained_ticks,
    )
    return {
        "scenario_id": row["scenario_id"],
        "scenario_signature": row["scenario_signature"],
        "domain": row.get("domain", "power_grid"),
        "backend_kind": row["backend_kind"],
        "family": row["family"],
        "difficulty_level": row["difficulty_level"],
        "agent_name": agent_name,
        "status": "complete",
        "observed": observed,
        "agentic_evidence": _build_agentic_evidence(
            scenario=scenario,
            loop=loop,
            observed=observed,
        ),
        "counterfactual": {
            "actual_cost": report.actual_cost,
            "wait_cost": report.counterfactual_cost,
            "prevented_loss": report.prevented_loss,
            "per_action_capped": report.per_action_capped,
            "per_action_status": report.per_action_status,
            "per_action_expected": report.per_action_expected,
            "per_action_attempted": report.per_action_attempted,
            "per_action_completed": report.per_action_completed,
            "per_action_failures": report.per_action_failures,
        },
        "replay_minimization": {
            "status": minimized.status,
            "claim": (
                "one_minimal_upper_bound_not_global_shortest"
                if one_minimal_proven
                else "bounded_successful_upper_bound_not_one_minimal"
            ),
            "success_prevention_ratio": prevention_ratio,
            "target_max_cost": target_metric,
            "success_metric": (
                "native_task_loss" if native_task_loss_replay else "actual_cost"
            ),
            "target_max_metric": target_metric,
            "configured_max_replays": max_replays,
            "max_replay_work_ticks": max_replay_work_ticks,
            "effective_max_replays": effective_max_replays,
            "successful_tool_set_upper_bound": minimized.tool_set,
            "successful_physical_actuator_endpoint_set_upper_bound": (
                minimized_actuator_endpoints
            ),
            "successful_distinct_tool_count_upper_bound": (
                minimized.distinct_tool_count
            ),
            "successful_non_meta_call_count_upper_bound": retained_calls,
            "successful_decision_tick_upper_bound": retained_ticks,
            "one_minimal_successful_tool_set": (
                minimized.tool_set if one_minimal_proven else None
            ),
            "one_minimal_physical_actuator_endpoint_set": (
                minimized_actuator_endpoints if one_minimal_proven else None
            ),
            "one_minimal_distinct_tool_count": (
                minimized.distinct_tool_count if one_minimal_proven else None
            ),
            "one_minimal_non_meta_call_count": (
                retained_calls if one_minimal_proven else None
            ),
            "one_minimal_decision_ticks": (
                retained_ticks if one_minimal_proven else None
            ),
            "n_replays": minimized.n_replays,
            "replay_cache_entries": len(replay_outcomes),
            "success_contract_version": SUCCESS_CONTRACT_VERSION,
            "non_meta_call_count_lower_bound": lower_bound,
            "non_meta_call_count_upper_bound": (
                exact_calls
                if exact_calls is not None
                else (retained_calls if has_successful_upper_bound else None)
            ),
            "bound_scope": (
                "observed_trace_subsets_exact"
                if exact_calls is not None
                else (
                    (
                        "policy_lower_bound_and_one_minimal_upper_bound"
                        if one_minimal_proven
                        else "policy_lower_bound_and_bounded_successful_upper_bound"
                    )
                    if has_successful_upper_bound
                    else "no_successful_baseline_trace"
                )
            ),
            "trace_subset_minimum_status": (
                exact.status
                if exact
                else (
                    "trace_too_long"
                    if has_successful_upper_bound
                    else "initial_trace_not_successful"
                )
            ),
            "trace_subset_minimum_tool_set": exact.tool_set if exact else None,
            "trace_subset_minimum_distinct_tool_count": (
                exact.distinct_tool_count if exact else None
            ),
            "trace_subset_minimum_non_meta_call_count": exact_calls,
            "trace_subset_minimum_n_replays": exact.n_replays if exact else 0,
            "global_shortest_successful_tool_set": None,
            "bounded_reduced_trace_proof": bounded_reduced_trace_proof,
            **dependency_proof,
        },
    }


def _complexity_error_result(
    row: dict[str, Any],
    agent_name: str,
    exc: BaseException | str,
) -> dict[str, Any]:
    error = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return {
        "scenario_id": row["scenario_id"],
        "scenario_signature": row.get("scenario_signature"),
        "agent_name": agent_name,
        "status": "error",
        "error": str(error),
    }


def _claim_process_group() -> None:
    """Put an isolated replay and its native children in one killable group."""
    if not hasattr(os, "setpgid"):
        return
    try:
        os.setpgid(0, 0)
    except OSError:
        # The process may already be a group leader on platforms where the
        # multiprocessing start method assigns one.  Cleanup still falls back
        # to terminating the Python child below.
        return


def _terminate_process_group(process: Any) -> None:
    """Terminate native descendants that outlive an isolated Python child."""
    process_id = getattr(process, "pid", None)
    if process_id is None or not hasattr(os, "killpg"):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(int(process_id), sig)
        except OSError:
            # The process group may already have exited; cleanup is
            # intentionally best effort and must not hide the replay result.
            continue


@contextmanager
def _sample_timeout(seconds: int | None):
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"complexity sample exceeded {seconds}s")

    prior_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prior_handler)


def _calibrate_one_child(
    connection: Any,
    row: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    _claim_process_group()
    try:
        connection.send(_calibrate_one(row, **kwargs))
    except BaseException as exc:
        connection.send(_complexity_error_result(row, str(kwargs["agent_name"]), exc))
    finally:
        connection.close()


def _calibrate_one_isolated(
    row: dict[str, Any],
    *,
    sample_timeout_seconds: int | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run replay minimization behind a process boundary killable on timeout."""
    if not sample_timeout_seconds or sample_timeout_seconds <= 0:
        try:
            return _calibrate_one(row, **kwargs)
        except BaseException as exc:
            return _complexity_error_result(row, str(kwargs["agent_name"]), exc)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_calibrate_one_child,
        args=(child, row, kwargs),
        daemon=False,
    )
    process.start()
    child.close()
    try:
        if parent.poll(float(sample_timeout_seconds)):
            try:
                return parent.recv()
            except EOFError:
                pass
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=5.0)
        _terminate_process_group(process)
        return _complexity_error_result(
            row,
            str(kwargs["agent_name"]),
            (
                "TimeoutError: complexity sample exceeded "
                f"{sample_timeout_seconds}s in isolated process"
            ),
        )
    finally:
        parent.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5.0)
        _terminate_process_group(process)


def _calibrate_one_dispatched(
    row: dict[str, Any],
    *,
    sample_timeout_seconds: int | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Isolate native solvers while reusing workers for safe Python backends."""
    if str(row.get("backend_kind") or "") in HARD_ISOLATION_BACKENDS or (
        sample_timeout_seconds
        and sample_timeout_seconds > 0
        and not hasattr(signal, "SIGALRM")
    ):
        return _calibrate_one_isolated(
            row,
            sample_timeout_seconds=sample_timeout_seconds,
            **kwargs,
        )
    try:
        with _sample_timeout(sample_timeout_seconds):
            return _calibrate_one(row, **kwargs)
    except Exception as exc:
        return _complexity_error_result(row, str(kwargs["agent_name"]), exc)


def _run_hard_isolated_batch(
    pending: list[tuple[dict[str, Any], str, Path]],
    *,
    workers: int,
    sample_timeout_seconds: int | None,
    kwargs_for: Any,
    on_result: Any,
) -> None:
    """Run native backends in one killable process layer."""
    context = multiprocessing.get_context("spawn")
    queued = deque(pending)
    active: dict[int, tuple[Any, Any, dict[str, Any], str, float]] = {}

    def terminate(process: Any) -> None:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=5.0)
        _terminate_process_group(process)

    try:
        while queued or active:
            while queued and len(active) < max(1, workers):
                row, agent_name, cache_path = queued.popleft()
                kwargs = kwargs_for(agent_name, cache_path)
                parent, child = context.Pipe(duplex=False)
                process = context.Process(
                    target=_calibrate_one_child,
                    args=(child, row, kwargs),
                    daemon=False,
                )
                process.start()
                child.close()
                active[int(process.pid)] = (
                    process,
                    parent,
                    row,
                    agent_name,
                    time.monotonic(),
                )

            progressed = False
            for pid, (
                process,
                parent,
                row,
                agent_name,
                started,
            ) in list(active.items()):
                result = None
                if parent.poll():
                    try:
                        result = parent.recv()
                    except EOFError:
                        result = _complexity_error_result(
                            row,
                            agent_name,
                            "complexity worker exited without a result",
                        )
                elif not process.is_alive():
                    result = _complexity_error_result(
                        row,
                        agent_name,
                        "complexity worker exited without a result",
                    )
                elif (
                    sample_timeout_seconds
                    and sample_timeout_seconds > 0
                    and time.monotonic() - started >= sample_timeout_seconds
                ):
                    result = _complexity_error_result(
                        row,
                        agent_name,
                        (
                            "TimeoutError: complexity sample exceeded "
                            f"{sample_timeout_seconds}s in isolated process"
                        ),
                    )
                if result is None:
                    continue
                terminate(process)
                parent.close()
                del active[pid]
                on_result(result)
                progressed = True
            if not progressed and active:
                time.sleep(0.05)
    finally:
        for process, parent, _row, _agent_name, _started in active.values():
            terminate(process)
            parent.close()


def _terminate_executor(executor: ProcessPoolExecutor) -> None:
    """Cancel and terminate a process pool without waiting on stuck solvers."""
    processes = list((getattr(executor, "_processes", None) or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=5.0)


def calibrate(
    suite_path: Path,
    output_path: Path,
    *,
    agent_names: list[str],
    prevention_ratio: float,
    max_replays: int,
    max_replay_work_ticks: int,
    per_action_cap: int | None,
    exact_max_calls: int,
    exact_max_replays: int,
    cache_dir: Path,
    limit: int | None,
    domains: set[str] | None = None,
    levels: set[str] | None = None,
    scenario_ids: set[str] | None = None,
    eligible_task_contracts_path: Path | None = None,
    supplemental_suite_paths: list[Path] | None = None,
    import_result_paths: list[Path] | None = None,
    workers: int = 1,
    sample_timeout_seconds: int | None = 900,
) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    identity = implementation_identity()
    implementation_tree_sha256 = identity["implementation_tree_sha256"]
    core_release_pipeline_sha256 = identity["core_release_pipeline_sha256"]
    source_rows = list(suite["scenarios"])
    seen_ids = {str(row["scenario_id"]) for row in source_rows}
    for supplemental_path in supplemental_suite_paths or []:
        supplemental = json.loads(supplemental_path.read_text(encoding="utf-8"))
        for row in supplemental.get("scenarios") or []:
            if str(row["scenario_id"]) not in seen_ids:
                source_rows.append(row)
                seen_ids.add(str(row["scenario_id"]))
    if eligible_task_contracts_path is not None:
        task_contracts = json.loads(
            eligible_task_contracts_path.read_text(encoding="utf-8")
        )
        if task_contracts.get("status") != "complete":
            raise ValueError("eligible task-contract report must be complete")
        if (
            task_contracts.get("implementation_tree_sha256")
            != implementation_tree_sha256
            or task_contracts.get("core_release_pipeline_sha256")
            != core_release_pipeline_sha256
        ):
            raise ValueError("eligible task-contract report is stale")
        task_pass_ids = _passed_task_contract_ids(task_contracts)
        scenario_ids = (
            task_pass_ids
            if scenario_ids is None
            else scenario_ids.intersection(task_pass_ids)
        )
    rows = _filter_rows(
        source_rows,
        domains=domains,
        levels=levels,
        scenario_ids=scenario_ids,
        limit=limit,
    )
    config = {
        "suite_id": suite.get("suite_id") or suite.get("candidate_id"),
        "selected_scenario_ids": [str(row["scenario_id"]) for row in rows],
        "agent_names": agent_names,
        "prevention_ratio": prevention_ratio,
        "max_replays": max_replays,
        "max_replay_work_ticks": max_replay_work_ticks,
        "per_action_cap": per_action_cap,
        "exact_max_calls": exact_max_calls,
        "exact_max_replays": exact_max_replays,
        "audit_episode_contract_version": AUDIT_EPISODE_CONTRACT_VERSION,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "evaluation_implementation_fingerprint": (
            EVALUATION_IMPLEMENTATION_FINGERPRINT
        ),
        "scoring_version": SCORING_VERSION,
        "success_contract_version": SUCCESS_CONTRACT_VERSION,
        "trajectory_analysis_contract_version": (TRAJECTORY_ANALYSIS_CONTRACT_VERSION),
        "sample_timeout_seconds": sample_timeout_seconds,
    }
    expected_signatures = {
        str(row["scenario_id"]): str(row["scenario_signature"]) for row in rows
    }
    results: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            prior.get("implementation_tree_sha256") == implementation_tree_sha256
            and prior.get("core_release_pipeline_sha256")
            == core_release_pipeline_sha256
            and _compatible_config(config, prior.get("config") or {})
        ):
            wanted = {
                (row["scenario_id"], agent_name)
                for row in rows
                for agent_name in agent_names
            }
            for result in prior.get("results", []):
                scenario_id = str(result.get("scenario_id") or "")
                agent_name = str(result.get("agent_name") or "")
                if (
                    (scenario_id, agent_name) in wanted
                    and result.get("status") == "complete"
                    and str(result.get("scenario_signature") or "")
                    == expected_signatures[scenario_id]
                ):
                    results[f"{scenario_id}::{agent_name}"] = (
                        _normalize_minimization_result(result)
                    )
    wanted_agents = set(agent_names)
    for import_path in import_result_paths or []:
        if not import_path.is_file():
            continue
        imported = json.loads(import_path.read_text(encoding="utf-8"))
        if (
            imported.get("implementation_tree_sha256") != implementation_tree_sha256
            or imported.get("core_release_pipeline_sha256")
            != core_release_pipeline_sha256
            or not _compatible_config(config, imported.get("config") or {})
        ):
            raise ValueError(f"incompatible replay calibration import: {import_path}")
        for result in imported.get("results") or []:
            scenario_id = str(result.get("scenario_id") or "")
            agent_name = str(result.get("agent_name") or "")
            if (
                scenario_id in expected_signatures
                and agent_name in wanted_agents
                and result.get("status") == "complete"
                and str(result.get("scenario_signature") or "")
                == expected_signatures[scenario_id]
            ):
                results.setdefault(
                    f"{scenario_id}::{agent_name}",
                    _normalize_minimization_result(result),
                )

    checkpoint_dir = _prepare_checkpoint(
        output_path,
        config=config,
        implementation_tree_sha256=implementation_tree_sha256,
        core_release_pipeline_sha256=core_release_pipeline_sha256,
    )
    for key, result in _load_checkpoint_results(
        checkpoint_dir,
        expected_signatures=expected_signatures,
        agent_names=wanted_agents,
    ).items():
        results.setdefault(key, result)

    def write() -> dict[str, Any]:
        ordered = [results[key] for key in sorted(results)]
        summary = {
            "status": (
                "complete"
                if len(ordered) == len(rows) * len(agent_names)
                else "partial"
            ),
            "n_expected": len(rows) * len(agent_names),
            "n_completed": len(ordered),
        }
        _atomic_write(
            output_path,
            {
                "schema_version": "0.2",
                "status": summary["status"],
                "config": config,
                "evaluation_semantics": {
                    "protocol_version": EVALUATION_PROTOCOL_VERSION,
                    "implementation_fingerprint": (
                        EVALUATION_IMPLEMENTATION_FINGERPRINT
                    ),
                    "scoring_version": SCORING_VERSION,
                },
                "implementation_tree_sha256": implementation_tree_sha256,
                "core_release_pipeline_sha256": core_release_pipeline_sha256,
                "n_expected": summary["n_expected"],
                "n_completed": summary["n_completed"],
                "results": ordered,
            },
        )
        return summary

    pending = [
        (
            row,
            agent_name,
            cache_dir / f"{row['scenario_signature']}--{agent_name}.json",
        )
        for row in rows
        for agent_name in agent_names
        if f"{row['scenario_id']}::{agent_name}" not in results
    ]

    def kwargs_for(agent_name: str, cache_path: Path) -> dict[str, Any]:
        return {
            "agent_name": agent_name,
            "prevention_ratio": prevention_ratio,
            "max_replays": max_replays,
            "max_replay_work_ticks": max_replay_work_ticks,
            "per_action_cap": per_action_cap,
            "exact_max_calls": exact_max_calls,
            "exact_max_replays": exact_max_replays,
            "replay_cache_path": cache_path,
        }

    def save_result(result: dict[str, Any]) -> None:
        key = f"{result['scenario_id']}::{result['agent_name']}"
        _write_checkpoint_result(checkpoint_dir, key=key, result=result)
        results[key] = result

    hard_pending = [
        item
        for item in pending
        if str(item[0].get("backend_kind") or "") in HARD_ISOLATION_BACKENDS
        or (
            sample_timeout_seconds
            and sample_timeout_seconds > 0
            and not hasattr(signal, "SIGALRM")
        )
    ]
    hard_keys = {(str(row["scenario_id"]), agent) for row, agent, _path in hard_pending}
    safe_pending = [
        item
        for item in pending
        if (str(item[0]["scenario_id"]), item[1]) not in hard_keys
    ]

    if workers <= 1:
        for row, agent_name, cache_path in pending:
            result = _calibrate_one_dispatched(
                row,
                sample_timeout_seconds=sample_timeout_seconds,
                **kwargs_for(agent_name, cache_path),
            )
            save_result(result)
    else:
        if safe_pending:
            executor = ProcessPoolExecutor(max_workers=workers)
            futures = {
                executor.submit(
                    _calibrate_one_dispatched,
                    row,
                    sample_timeout_seconds=sample_timeout_seconds,
                    **kwargs_for(agent_name, cache_path),
                ): (row, agent_name)
                for row, agent_name, cache_path in safe_pending
            }
            try:
                for future in as_completed(futures):
                    row, agent_name = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = _complexity_error_result(row, agent_name, exc)
                    save_result(result)
            except BaseException:
                for future in futures:
                    future.cancel()
                _terminate_executor(executor)
                raise
            else:
                executor.shutdown(wait=True)
        bounded_hard_pending = [
            item
            for item in hard_pending
            if str(item[0].get("backend_kind") or "") in BOUNDED_ISOLATION_WORKERS
        ]
        parallel_hard_pending = [
            item
            for item in hard_pending
            if str(item[0].get("backend_kind") or "") not in BOUNDED_ISOLATION_WORKERS
        ]
        if parallel_hard_pending:
            _run_hard_isolated_batch(
                parallel_hard_pending,
                workers=workers,
                sample_timeout_seconds=sample_timeout_seconds,
                kwargs_for=kwargs_for,
                on_result=save_result,
            )
        if bounded_hard_pending:
            _run_hard_isolated_batch(
                bounded_hard_pending,
                workers=min(workers, BOUNDED_ISOLATION_WORKERS["sumo"]),
                sample_timeout_seconds=sample_timeout_seconds,
                kwargs_for=kwargs_for,
                on_result=save_result,
            )
    summary = write()
    _clear_checkpoint(output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["oracle_offline", "greedy_heuristic"],
    )
    parser.add_argument("--prevention-ratio", type=float, default=0.95)
    parser.add_argument("--max-replays", type=int, default=100)
    parser.add_argument(
        "--max-replay-work-ticks",
        type=int,
        default=512,
        help=(
            "Bound minimization work as horizon_ticks × replay_count; "
            "long trajectories retain successful upper/lower bounds."
        ),
    )
    parser.add_argument("--exact-max-calls", type=int, default=6)
    parser.add_argument("--exact-max-replays", type=int, default=256)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / ".audit-cache" / "complexity-replays",
    )
    parser.add_argument(
        "--per-action-cap",
        type=int,
        default=20,
        help="Maximum leave-one-out replays per row; use -1 for all actions.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-timeout-seconds", type=int, default=900)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--levels", nargs="+")
    parser.add_argument(
        "--scenario-ids-file",
        type=Path,
        help="Optional JSON array or newline-delimited list of scenario IDs.",
    )
    parser.add_argument(
        "--eligible-task-contracts",
        type=Path,
        help="Complete task-contract report; replay-calibrate only passed rows.",
    )
    parser.add_argument("--supplemental-suite", type=Path, action="append")
    parser.add_argument("--import-results", type=Path, action="append")
    args = parser.parse_args()
    scenario_ids = None
    if args.scenario_ids_file:
        text = args.scenario_ids_file.read_text(encoding="utf-8")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [line.strip() for line in text.splitlines() if line.strip()]
        if not isinstance(parsed, list) or not all(
            isinstance(value, str) for value in parsed
        ):
            raise ValueError(
                "scenario IDs file must contain a JSON string array or one ID per line"
            )
        scenario_ids = set(parsed)
    report = calibrate(
        args.suite.resolve(),
        args.output.resolve(),
        agent_names=args.agents,
        prevention_ratio=args.prevention_ratio,
        max_replays=args.max_replays,
        max_replay_work_ticks=args.max_replay_work_ticks,
        per_action_cap=None if args.per_action_cap < 0 else args.per_action_cap,
        exact_max_calls=args.exact_max_calls,
        exact_max_replays=args.exact_max_replays,
        cache_dir=args.cache_dir.resolve(),
        limit=args.limit,
        domains=set(args.domains) if args.domains else None,
        levels=set(args.levels) if args.levels else None,
        scenario_ids=scenario_ids,
        eligible_task_contracts_path=(
            args.eligible_task_contracts.resolve()
            if args.eligible_task_contracts
            else None
        ),
        supplemental_suite_paths=(
            [path.resolve() for path in args.supplemental_suite]
            if args.supplemental_suite
            else None
        ),
        import_result_paths=(
            [path.resolve() for path in args.import_results]
            if args.import_results
            else None
        ),
        workers=max(1, args.workers),
        sample_timeout_seconds=args.sample_timeout_seconds,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "n_expected", "n_completed")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
