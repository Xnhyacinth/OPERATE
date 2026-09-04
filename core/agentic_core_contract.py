"""Fail-closed Protocol-2.1 agentic Core admission contract.

The contract joins runtime evidence by ``scenario_id`` and
``scenario_signature``. Scenario metadata may describe an intended task, but it
never substitutes for replay evidence that the world evolved, native controls
were used, and the task remained solvable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from core.implementation_identity import implementation_identity
from core.protocol21_admission import (
    QUALITY_CORE_V2_ADMISSION_PROFILE,
    agentic_admission_check_names,
    declared_protocol21_admission_profile,
    resolve_protocol21_admission_profile,
)
from core.protocol21_evidence import artifact_binding as _artifact_binding
from evaluation.scorer import SCORING_VERSION
from runner.episode import (
    EVALUATION_IMPLEMENTATION_FINGERPRINT,
    EVALUATION_PROTOCOL_VERSION,
)

REQUIRED_SEMANTICS = {
    "protocol_version": EVALUATION_PROTOCOL_VERSION,
    "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
    "scoring_version": SCORING_VERSION,
}
SOURCE_GATES = {
    "source_lock_passed": "source_lock",
    "source_consumption_passed": "source_consumption",
    "source_independence_passed": "source_independence",
    "deterministic_replay_passed": "deterministic_replay",
}
RETIRED_BLOCKERS = {
    "deterministic_replay_failed",
    "task_contract_failed",
    "reference_headroom_nonpositive",
    "source_consumption_failed",
    "predesigned_event_unreachable",
    "strategy_depth_contradicted",
}


def artifact_binding(path: Path) -> dict[str, Any]:
    """Return the immutable identity of a JSON input artifact."""
    return _artifact_binding(path)


def _semantics(report: dict[str, Any]) -> dict[str, str]:
    raw = report.get("evaluation_semantics") or {}
    protocol = report.get("evaluation_protocol") or {}
    config = report.get("config") or {}
    return {
        "protocol_version": str(
            raw.get("protocol_version")
            or raw.get("evaluation_protocol_version")
            or raw.get("version")
            or protocol.get("version")
            or report.get("evaluation_protocol_version")
            or config.get("evaluation_protocol_version")
            or ""
        ),
        "implementation_fingerprint": str(
            raw.get("implementation_fingerprint")
            or raw.get("evaluation_implementation_fingerprint")
            or protocol.get("implementation_fingerprint")
            or report.get("evaluation_implementation_fingerprint")
            or config.get("evaluation_implementation_fingerprint")
            or ""
        ),
        "scoring_version": str(
            raw.get("scoring_version")
            or report.get("scoring_version")
            or config.get("scoring_version")
            or ""
        ),
    }


def _is_current(report: dict[str, Any]) -> bool:
    return _semantics(report) == REQUIRED_SEMANTICS


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "samples", "scenarios"):
        value = report.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _index(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in _rows(report):
        grouped.setdefault(str(row.get("scenario_id") or ""), []).append(row)
    return grouped


def _signature_matches(
    signature: str,
    rows: list[dict[str, Any]],
) -> bool:
    return bool(signature) and all(
        str(row.get("scenario_signature") or "") == signature for row in rows
    )


def _source_gate_names(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    passed = set(row.get("passed_gates") or [])
    failed = set(row.get("failed_gates") or [])
    gates = row.get("gates") or {}
    for name, value in gates.items():
        canonical = {
            "counterfactual": "counterfactual_replay",
            "independence": "source_independence",
        }.get(str(name), str(name))
        (passed if value is True else failed).add(canonical)
    return passed, failed


def _agent_name(row: dict[str, Any]) -> str:
    return str(row.get("agent_name") or row.get("agent") or "").lower()


def _has_runtime_tick_evidence(value: Any, *, horizon: int | None = None) -> bool:
    """Accept only concrete numeric tick lists as supervision evidence."""
    if not isinstance(value, list):
        return False
    for tick in value:
        if not isinstance(tick, (int, float)) or isinstance(tick, bool):
            continue
        try:
            numeric = float(tick)
        except (OverflowError, TypeError, ValueError):
            continue
        if not math.isfinite(numeric) or numeric != int(numeric):
            continue
        try:
            parsed = int(numeric)
        except (OverflowError, TypeError, ValueError):
            continue
        if parsed < 0 or (horizon is not None and parsed >= horizon):
            continue
        return True
    return False


def _runtime_ticks(value: Any, *, horizon: int | None = None) -> list[int]:
    """Return finite, integral runtime ticks within the episode horizon."""
    if not isinstance(value, list):
        return []
    ticks: set[int] = set()
    for tick in value:
        if not isinstance(tick, (int, float)) or isinstance(tick, bool):
            continue
        try:
            numeric = float(tick)
        except (OverflowError, TypeError, ValueError):
            continue
        if not math.isfinite(numeric) or numeric != int(numeric):
            continue
        try:
            parsed = int(numeric)
        except (OverflowError, TypeError, ValueError):
            continue
        if parsed < 0 or (horizon is not None and parsed >= horizon):
            continue
        ticks.add(parsed)
    return sorted(ticks)


def _runtime_int(value: Any, *, default: int = 0) -> int:
    """Parse one finite integral runtime value without raising on JSON NaN."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError):
            return default
    elif isinstance(value, str):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return default
    else:
        return default
    if not math.isfinite(numeric) or numeric != int(numeric):
        return default
    try:
        return int(numeric)
    except (OverflowError, TypeError, ValueError):
        return default


def _persistent_policy_evidence(
    *,
    evidence: dict[str, Any],
    material_exogenous_records: list[dict[str, Any]],
    post_change_decision_ticks: list[int],
    horizon: int,
    source_row: dict[str, Any] | None = None,
    control_tools: list[str] | None = None,
    successful_control_tools: set[str] | None = None,
) -> dict[str, Any]:
    """Validate runtime evidence for a persistent-policy review path.

    A review is supervisory evidence, not a physical action.  It is admitted
    only when the backend binds it to observed event ids and the separate
    masked replay proves that keeping the policy changed a native outcome.
    YAML claims, free-form rationales and aggregate cost deltas are not enough.
    """
    source_row = source_row or {}
    source_capability = source_row.get("capability") or {}
    policy_contract = source_capability.get("persistent_policy_review")
    policy_contract = policy_contract if isinstance(policy_contract, dict) else {}
    contract_backend = str(policy_contract.get("backend_kind") or "")
    source_backend = str(source_row.get("backend_kind") or "")
    review_tool_name = str(policy_contract.get("review_tool_name") or "")
    policy_tool_names = {
        str(value)
        for value in policy_contract.get("policy_tool_names") or []
        if str(value)
    }
    declared_control_tools = {
        str(value)
        for value in (source_capability.get("control_tools") or [])
        if str(value)
    }
    observed_control_tools = {
        str(value) for value in (control_tools or []) if str(value)
    }
    successful_control_tools = {
        str(value) for value in (successful_control_tools or set()) if str(value)
    }
    persistent_control_tools = {
        str(value)
        for value in (evidence.get("persistent_control_tool_names") or [])
        if str(value)
    }
    contract_declared = bool(
        source_backend
        and contract_backend == source_backend
        and review_tool_name
        and policy_tool_names
        and (
            not declared_control_tools
            or policy_tool_names.issubset(declared_control_tools)
        )
    )
    event_ticks = {
        str(record.get("event_id") or ""): _runtime_int(
            record.get("applied_tick", record.get("tick")),
            default=-1,
        )
        for record in material_exogenous_records
        if isinstance(record, dict) and str(record.get("event_id") or "")
    }
    raw_reviews = evidence.get("persistent_policy_review_records")
    reviews = raw_reviews if isinstance(raw_reviews, list) else []
    raw_bindings = evidence.get("persistent_policy_review_bindings")
    bindings = raw_bindings if isinstance(raw_bindings, list) else []
    binding_by_review_id = {
        str(binding.get("review_id") or ""): binding
        for binding in bindings
        if isinstance(binding, dict) and str(binding.get("review_id") or "")
    }
    policy_effect_bindings = evidence.get("persistent_policy_effect_bindings")
    policy_effect_bindings = (
        policy_effect_bindings if isinstance(policy_effect_bindings, list) else []
    )
    effect_bindings_by_generation: dict[int, list[dict[str, Any]]] = {}
    for effect in policy_effect_bindings:
        if not isinstance(effect, dict):
            continue
        generation = _runtime_int(effect.get("policy_generation"), default=0)
        effect_tool_name = str(effect.get("policy_tool_name") or "")
        effect_tick = _runtime_int(effect.get("effect_tick"), default=-1)
        effect_call_id = str(effect.get("call_id") or "")
        effect_ids = {
            str(value) for value in effect.get("evidence_ids") or [] if str(value)
        }
        if (
            generation > 0
            and effect.get("accepted") is True
            and effect_tool_name
            and effect_tick >= 0
            and effect_call_id
            and effect_ids
        ):
            effect_bindings_by_generation.setdefault(generation, []).append(
                {
                    "policy_tool_name": effect_tool_name,
                    "effect_tick": effect_tick,
                    "call_id": effect_call_id,
                    "evidence_ids": effect_ids,
                }
            )
    valid_reviews: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    for review in reviews:
        if not isinstance(review, dict):
            continue
        review_id = str(review.get("review_id") or "")
        review_tick = _runtime_int(review.get("review_tick"), default=-1)
        generation = _runtime_int(review.get("policy_generation"), default=0)
        event_ids = [
            str(event_id).strip()
            for event_id in review.get("event_ids") or []
            if str(event_id).strip()
        ]
        queue_digest = str(review.get("queue_order_digest") or "")
        decision = str(review.get("decision") or "keep")
        policy_tool_name = str(review.get("policy_tool_name") or "")
        review_tool = str(review.get("review_tool_name") or "")
        policy_effect_evidence_id = str(review.get("policy_effect_evidence_id") or "")
        review_evidence_ids = {
            str(value) for value in review.get("evidence_ids") or [] if str(value)
        }
        binding = binding_by_review_id.get(review_id) or {}
        matching_effect = next(
            (
                effect
                for effect in effect_bindings_by_generation.get(generation, [])
                if policy_effect_evidence_id in effect["evidence_ids"]
                and effect["policy_tool_name"] == policy_tool_name
                and effect["effect_tick"] <= review_tick
            ),
            None,
        )
        binding_evidence_ids = {
            str(value) for value in binding.get("evidence_ids") or [] if str(value)
        }
        graph_evidence_ids = {
            str(value)
            for value in binding.get("action_graph_evidence_ids") or []
            if str(value)
        }
        effect_ticks = _runtime_ticks(
            review.get("outcome_effect_ticks"), horizon=horizon
        )
        if (
            not contract_declared
            or not review_id
            or review_tick < 0
            or review_tick >= horizon
            or generation <= 0
            or not event_ids
            or len(event_ids) != len(set(event_ids))
            or seen_event_ids.intersection(event_ids)
            or decision not in {"keep", "replace", "retire"}
            or review_tool != review_tool_name
            or policy_tool_name not in policy_tool_names
            or policy_tool_name not in persistent_control_tools
            or policy_tool_name not in observed_control_tools
            or policy_tool_name not in successful_control_tools
            or not policy_effect_evidence_id
            or matching_effect is None
            or not review_evidence_ids
            or binding.get("accepted") is not True
            or str(binding.get("review_tool_name") or "") != review_tool_name
            or not str(binding.get("call_id") or "")
            or not (review_evidence_ids & binding_evidence_ids & graph_evidence_ids)
            or not str(review.get("policy_digest") or "")
            or not queue_digest
            or any(event_id not in event_ticks for event_id in event_ids)
            or any(event_ticks[event_id] < 0 for event_id in event_ids)
            or any(event_ticks[event_id] >= review_tick for event_id in event_ids)
            or not effect_ticks
        ):
            continue
        declared_ticks = _runtime_ticks(review.get("event_ticks"), horizon=horizon)
        if declared_ticks and declared_ticks != sorted(
            {event_ticks[event_id] for event_id in event_ids}
        ):
            continue
        valid_reviews.append(
            {
                **review,
                "review_id": review_id,
                "review_tick": review_tick,
                "policy_generation": generation,
                "event_ids": event_ids,
                "event_ticks": sorted(event_ticks[event_id] for event_id in event_ids),
                "outcome_effect_ticks": effect_ticks,
                "decision": decision,
                "review_tool_name": review_tool,
                "policy_tool_name": policy_tool_name,
                "policy_effect_evidence_id": policy_effect_evidence_id,
                "evidence_ids": sorted(review_evidence_ids),
            }
        )
        seen_event_ids.update(event_ids)

    attribution = evidence.get("persistent_policy_attribution")
    attribution = attribution if isinstance(attribution, dict) else {}
    attribution_review_id = str(attribution.get("review_id") or "")
    matched_review = next(
        (
            review
            for review in valid_reviews
            if review["review_id"] == attribution_review_id
        ),
        None,
    )
    attribution_effect_ticks = _runtime_ticks(
        attribution.get("effect_ticks"), horizon=horizon
    )
    attribution_policy_tool = str(attribution.get("policy_tool_name") or "")
    attribution_review_tool = str(attribution.get("review_tool_name") or "")
    attribution_generation = _runtime_int(
        attribution.get("policy_generation"), default=0
    )
    attribution_effect_evidence_id = str(
        attribution.get("policy_effect_evidence_id") or ""
    )
    actual_state_digest = str(attribution.get("actual_state_digest") or "")
    masked_state_digest = str(attribution.get("masked_state_digest") or "")
    actual_queue_digest = str(attribution.get("actual_queue_order_digest") or "")
    masked_queue_digest = str(attribution.get("masked_queue_order_digest") or "")
    material_delta = attribution.get("material_delta")
    threshold = attribution.get("materiality_threshold")
    try:
        material_delta_value = float(material_delta)
        threshold_value = float(threshold)
    except (TypeError, ValueError):
        material_delta_value = 0.0
        threshold_value = math.inf
    attribution_passed = bool(
        matched_review
        and attribution.get("status") == "passed"
        and attribution_review_tool == review_tool_name
        and attribution_policy_tool == matched_review.get("policy_tool_name")
        and attribution_generation == matched_review.get("policy_generation")
        and attribution_effect_evidence_id
        == matched_review.get("policy_effect_evidence_id")
        and attribution.get("deterministic_replay") is True
        and attribution_effect_ticks
        and set(attribution_effect_ticks).intersection(
            matched_review["outcome_effect_ticks"]
        )
        and actual_state_digest
        and masked_state_digest
        and actual_state_digest != masked_state_digest
        and actual_queue_digest
        and masked_queue_digest
        and actual_queue_digest != masked_queue_digest
        and math.isfinite(material_delta_value)
        and math.isfinite(threshold_value)
        and material_delta_value > max(0.0, threshold_value)
    )
    timeline_observed = False
    if attribution_passed and matched_review:
        for event_tick in matched_review["event_ticks"]:
            for decision_tick in post_change_decision_ticks:
                if not event_tick < decision_tick <= matched_review["review_tick"]:
                    continue
                if any(
                    effect_tick >= matched_review["review_tick"]
                    for effect_tick in attribution_effect_ticks
                ):
                    timeline_observed = True
                    break
            if timeline_observed:
                break
    return {
        "reviews": valid_reviews,
        "review_ticks": sorted(int(review["review_tick"]) for review in valid_reviews),
        "review_observed": bool(valid_reviews),
        "attribution_passed": bool(attribution_passed),
        "timeline_observed": timeline_observed,
        "matched_review_id": (matched_review["review_id"] if matched_review else ""),
        "contract_declared": contract_declared,
        "review_tool_name": review_tool_name,
        "policy_tool_names": sorted(policy_tool_names),
    }


def _hard_checks() -> tuple[str, ...]:
    return (
        "current_protocol_semantics",
        "identity_bound_across_artifacts",
        "scenario_signature_current",
        "native_backend_executable",
        "source_lock_passed",
        "source_consumption_passed",
        "source_independence_passed",
        "deterministic_replay_passed",
        "multi_tick_horizon",
        "world_change_contract_declared",
        "material_exogenous_change_observed",
        "post_change_decision_observed",
        "difficulty_appropriate_control_response_observed",
        "agent_action_backend_effect_observed",
        "exogenous_state_evolution_observed",
        "predesigned_change_or_disruption_reached",
        "event_or_change_occurs_after_initial_state",
        "event_adaptive_cadence_declared",
        "actual_supervisory_review_observed",
        "parallel_simulator_agent_clock_observed",
        "native_state_changing_control_available",
        "successful_reference_used_native_control",
        "task_contract_passed",
        "terminal_integrity_passed",
        "oracle_result_present",
        "greedy_result_present",
        "wait_reference_present",
        "material_reference_headroom",
        "decision_graph_present",
        "decision_graph_acyclic",
        "bounded_or_exact_depth_available",
        "difficulty_not_contradicted",
    )


def validate_agentic_row(
    *,
    source_row: dict[str, Any],
    behavioral_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    complexity_rows: list[dict[str, Any]],
    observed_depth_rows: list[dict[str, Any]],
    strategy_depth_rows: list[dict[str, Any]],
    source_grounded_rows: list[dict[str, Any]],
    source_consumption_rows: list[dict[str, Any]],
    semantics_current: bool,
    admission_profile: str = "strict_v1",
) -> dict[str, Any]:
    """Validate one identity without deriving runtime facts from YAML names."""
    scenario_id = str(source_row.get("scenario_id") or "")
    signature = str(source_row.get("scenario_signature") or "")
    joined_rows = (
        behavioral_rows
        + task_rows
        + complexity_rows
        + observed_depth_rows
        + strategy_depth_rows
        + source_grounded_rows
        + source_consumption_rows
    )
    complexity_agents = [_agent_name(row) for row in complexity_rows]
    identity_bound = bool(joined_rows) and (
        len(behavioral_rows) == 1
        and len(task_rows) == 1
        and len(observed_depth_rows) == 1
        and len(strategy_depth_rows) == 1
        and len(source_grounded_rows) == 1
        and len(source_consumption_rows) == 1
        and sorted(complexity_agents)
        == ["greedy_heuristic", "oracle_offline", "wait_only"]
    )
    signature_matches = identity_bound and _signature_matches(signature, joined_rows)

    behavior = behavioral_rows[0] if behavioral_rows else {}
    task = task_rows[0] if task_rows else {}
    observed = observed_depth_rows[0] if observed_depth_rows else {}
    strategy = strategy_depth_rows[0] if strategy_depth_rows else {}
    source_gate = source_grounded_rows[0] if source_grounded_rows else {}
    source_consumption = source_consumption_rows[0] if source_consumption_rows else {}
    reference_agent = _agent_name(task)
    reference_complexity_rows = [
        row
        for row in complexity_rows
        if reference_agent and _agent_name(row) == reference_agent
    ]
    if not reference_complexity_rows:
        reference_complexity_rows = [
            row
            for row in complexity_rows
            if _agent_name(row) in {"oracle_offline", "greedy_heuristic"}
        ]
    complexity_evidence = next(
        (
            row.get("agentic_evidence")
            for row in reference_complexity_rows
            if isinstance(row.get("agentic_evidence"), dict)
        ),
        {},
    )
    evidence = {
        **(behavior.get("agentic_evidence") or {}),
        **complexity_evidence,
    }
    passed_gates, _ = _source_gate_names(source_gate)
    source_gate_evidence = source_gate.get("gate_evidence") or {}

    oracle_rows = [row for row in complexity_rows if "oracle" in _agent_name(row)]
    greedy_rows = [row for row in complexity_rows if "greedy" in _agent_name(row)]
    wait_rows = [row for row in complexity_rows if "wait" in _agent_name(row)]
    task_material_headroom = task.get("material_headroom") or {}
    material_headroom = task_material_headroom.get("status") == "passed"

    graph_rows = [
        row.get("replay_minimization") or row.get("decision_graph") or {}
        for row in reference_complexity_rows
    ]
    graph_present = bool(evidence.get("decision_graph_nodes", 0)) or any(
        graph.get("decision_graph_present") is True or bool(graph.get("nodes"))
        for graph in graph_rows
    )
    graph_acyclic = evidence.get("decision_graph_acyclic") is True or any(
        graph.get("decision_graph_acyclic") is True for graph in graph_rows
    )
    depth_available = any(
        value is not None
        for value in (
            strategy.get("exact_task_dependency_depth"),
            strategy.get("successful_strategy_tick_upper_bound"),
            strategy.get("non_meta_call_count_lower_bound"),
        )
    ) or any(
        graph.get("bounded_or_exact_depth_available") is True for graph in graph_rows
    )
    contradicted = (
        str(strategy.get("core_action") or "") in {"replace_or_retire", "retire"}
        or "contradicted" in str(strategy.get("disposition") or "")
        or "contradicted" in str(observed.get("disposition") or "")
    )
    control_tools = list(
        evidence.get("available_native_control_tool_names")
        or (source_row.get("capability") or {}).get("control_tools")
        or []
    )
    successful_control_tools = set(
        map(
            str,
            evidence.get("successful_native_control_tool_names") or [],
        )
    )
    holds = _runtime_int(evidence.get("autonomous_hold_ticks")) + _runtime_int(
        evidence.get("pending_action_hold_ticks")
    )
    horizon = _runtime_int(
        evidence.get("simulator_ticks") or source_row.get("horizon_ticks")
    )
    event_ticks = _runtime_ticks(
        evidence.get("realized_predesigned_event_ticks"), horizon=horizon
    )
    exogenous_ticks = _runtime_ticks(
        evidence.get("exogenous_state_change_ticks"), horizon=horizon
    )
    source_consumption_ticks = _runtime_ticks(
        evidence.get("source_consumption_ticks"), horizon=horizon
    )
    declared_event_ticks = _runtime_ticks(
        evidence.get("declared_predesigned_event_ticks"), horizon=horizon
    )
    event_unreachable = bool(declared_event_ticks) and not any(
        0 < tick < horizon for tick in declared_event_ticks
    )
    combined_world_ticks = event_ticks + exogenous_ticks + source_consumption_ticks
    material_exogenous_records = list(
        evidence.get("material_exogenous_event_records") or []
    )
    post_change_decision_ticks = _runtime_ticks(
        evidence.get("post_change_decision_ticks"), horizon=horizon
    )
    persistent_policy = _persistent_policy_evidence(
        evidence=evidence,
        material_exogenous_records=material_exogenous_records,
        post_change_decision_ticks=post_change_decision_ticks,
        horizon=horizon,
        source_row=source_row,
        control_tools=control_tools,
        successful_control_tools=successful_control_tools,
    )
    material_event_ticks = _runtime_ticks(
        [
            record.get("applied_tick", record.get("tick"))
            for record in material_exogenous_records
            if isinstance(record, dict)
        ],
        horizon=horizon,
    )
    delegated_plan_opportunity_ticks = _runtime_ticks(
        evidence.get("delegated_plan_opportunity_ticks"), horizon=horizon
    )
    delegation_after_change_observed = any(
        event_tick < delegation_tick
        for event_tick in material_event_ticks
        for delegation_tick in delegated_plan_opportunity_ticks
    )
    difficulty_level = str(source_row.get("difficulty_level") or "").lower()
    active_replanning_required = difficulty_level in {"high", "extreme"}
    adaptive_control_observed = bool(
        evidence.get("adaptive_replanning_observed")
        or (
            evidence.get("valid_plan_delegation_observed")
            and delegation_after_change_observed
        )
    )
    # Every release row must show a concrete supervision edge.  A scheduled
    # review or periodic scan is preferred; continuous native-control rows may
    # instead provide the actual model-decision ticks captured by the runner.
    scheduled_review_ticks = _runtime_ticks(
        evidence.get("scheduled_review_ticks"), horizon=horizon
    )
    periodic_scan_ticks = _runtime_ticks(
        evidence.get("periodic_scan_ticks"), horizon=horizon
    )
    continuous_supervisory_review_ticks = _runtime_ticks(
        evidence.get("continuous_supervisory_review_ticks"), horizon=horizon
    )
    actual_supervisory_review_observed = bool(
        scheduled_review_ticks
        or periodic_scan_ticks
        or continuous_supervisory_review_ticks
        or persistent_policy["review_observed"]
    )
    standing_plan_commit_ticks = _runtime_ticks(
        evidence.get("standing_plan_commit_ticks"), horizon=horizon
    )
    standing_control_commit_ticks = _runtime_ticks(
        evidence.get("standing_control_commit_ticks"), horizon=horizon
    )
    standing_plan_committed = bool(
        standing_plan_commit_ticks or standing_control_commit_ticks
    )
    plan_commit_ticks = sorted(
        set(standing_plan_commit_ticks + standing_control_commit_ticks)
    )
    supervisory_review_ticks = sorted(
        set(
            scheduled_review_ticks
            + periodic_scan_ticks
            + continuous_supervisory_review_ticks
        )
    )
    agent_effect_ticks = _runtime_ticks(
        evidence.get("agent_caused_state_change_ticks"), horizon=horizon
    )
    standing_plan_timeline_observed = any(
        event_tick < decision_tick
        and commit_tick <= decision_tick
        and review_tick >= max(event_tick, commit_tick)
        and effect_tick >= decision_tick
        for event_tick in material_event_ticks
        for decision_tick in post_change_decision_ticks
        for commit_tick in plan_commit_ticks
        for review_tick in supervisory_review_ticks
        for effect_tick in agent_effect_ticks
    )
    direct_standing_plan_observed = bool(
        material_exogenous_records
        and post_change_decision_ticks
        and standing_plan_committed
        and actual_supervisory_review_observed
        and standing_plan_timeline_observed
        and evidence.get("agent_action_backend_effect_observed")
        and successful_control_tools.intersection(map(str, control_tools))
    )
    standing_plan_observed = bool(
        direct_standing_plan_observed
        or (
            persistent_policy["timeline_observed"]
            and evidence.get("agent_action_backend_effect_observed")
            and successful_control_tools.intersection(map(str, control_tools))
        )
    )
    required_control_response = (
        "active_replanning_or_valid_delegation"
        if active_replanning_required
        else "standing_plan_with_post_change_monitoring"
    )

    checks = {
        "current_protocol_semantics": semantics_current,
        "identity_bound_across_artifacts": identity_bound and signature_matches,
        "scenario_signature_current": bool(signature) and signature_matches,
        "native_backend_executable": bool(
            (behavior.get("checks") or {}).get("native_backend_executable")
        ),
        "source_lock_passed": "source_lock" in passed_gates,
        "source_consumption_passed": bool(
            "source_consumption" in passed_gates
            and source_consumption.get("status") == "passed"
        ),
        "source_independence_passed": "source_independence" in passed_gates,
        "deterministic_replay_passed": "deterministic_replay" in passed_gates,
        "multi_tick_horizon": horizon > 1,
        "world_change_contract_declared": bool(
            evidence.get("world_change_contract_declared")
        ),
        "material_exogenous_change_observed": bool(material_exogenous_records),
        "post_change_decision_observed": bool(
            material_exogenous_records and post_change_decision_ticks
        ),
        "difficulty_appropriate_control_response_observed": bool(
            adaptive_control_observed
            if active_replanning_required
            else standing_plan_observed
        ),
        "agent_action_backend_effect_observed": bool(
            evidence.get("agent_action_backend_effect_observed")
        ),
        "exogenous_state_evolution_observed": bool(
            exogenous_ticks or source_consumption_ticks
        ),
        "predesigned_change_or_disruption_reached": bool(
            event_ticks and not event_unreachable
        ),
        "event_or_change_occurs_after_initial_state": any(
            tick > 0 for tick in combined_world_ticks
        ),
        "event_adaptive_cadence_declared": bool(
            evidence.get("event_adaptive_cadence_declared")
        ),
        "actual_supervisory_review_observed": bool(actual_supervisory_review_observed),
        "parallel_simulator_agent_clock_observed": bool(
            evidence.get("simulator_owned_clock_observed") is True
            or evidence.get("simulator_advance_without_model_ticks")
            or holds > 0
            or (
                evidence.get("periodic_cadence_observed") is True
                and evidence.get("decision_opportunity_ticks")
            )
        ),
        "native_state_changing_control_available": bool(
            control_tools and "capability_contract" in passed_gates
        ),
        "successful_reference_used_native_control": bool(
            successful_control_tools.intersection(map(str, control_tools))
        ),
        "task_contract_passed": bool(
            task.get("status") == "passed" and task.get("completed") is True
        ),
        "terminal_integrity_passed": bool(
            (task.get("terminal_integrity") or {}).get("release_ready")
        ),
        "oracle_result_present": bool(oracle_rows),
        "greedy_result_present": bool(greedy_rows),
        "wait_reference_present": bool(wait_rows),
        "material_reference_headroom": material_headroom,
        "decision_graph_present": graph_present,
        "decision_graph_acyclic": graph_acyclic,
        "bounded_or_exact_depth_available": depth_available,
        "difficulty_not_contradicted": not contradicted,
    }

    blockers: list[str] = []
    if not semantics_current:
        blockers.append("artifact_semantics_stale")
    if not identity_bound or not signature_matches:
        blockers.append("artifact_identity_mismatch")
    if not checks["native_backend_executable"]:
        blockers.append("native_backend_execution_unproven")
    if not checks["source_consumption_passed"]:
        blockers.append(
            "source_consumption_failed"
            if (source_gate_evidence.get("source_consumption") or {}).get("status")
            == "failed"
            else "source_consumption_unproven"
        )
    if not checks["deterministic_replay_passed"]:
        blockers.append(
            "deterministic_replay_failed"
            if (source_gate_evidence.get("deterministic_replay") or {}).get("status")
            == "failed"
            else "deterministic_replay_evidence_missing"
        )
    if not checks["task_contract_passed"]:
        blockers.append(
            "task_contract_failed"
            if task and task.get("status") == "failed"
            else "task_contract_evidence_missing"
        )
    if not checks["terminal_integrity_passed"]:
        blockers.append("terminal_integrity_failed")
    if not checks["exogenous_state_evolution_observed"]:
        blockers.append("evidence_missing_runtime_evolution")
    if not checks["world_change_contract_declared"]:
        blockers.append("world_change_contract_missing")
    if not checks["material_exogenous_change_observed"]:
        blockers.append("material_exogenous_change_unproven")
    if not checks["post_change_decision_observed"]:
        blockers.append("post_change_decision_unproven")
    if not checks["difficulty_appropriate_control_response_observed"]:
        blockers.append(
            "adaptive_replanning_or_delegation_unproven"
            if active_replanning_required
            else "standing_plan_response_unproven"
        )
    if not checks["agent_action_backend_effect_observed"]:
        blockers.append("agent_action_backend_effect_unproven")
    if not checks["predesigned_change_or_disruption_reached"]:
        blockers.append(
            "predesigned_event_unreachable"
            if event_unreachable
            else "predesigned_event_not_reached"
        )
    if not checks["event_adaptive_cadence_declared"]:
        blockers.append("event_adaptive_cadence_undeclared")
    if not checks["actual_supervisory_review_observed"]:
        blockers.append("actual_supervisory_review_unobserved")
    if not checks["parallel_simulator_agent_clock_observed"]:
        blockers.append("parallel_clock_not_observed")
    if not checks["successful_reference_used_native_control"]:
        blockers.append("evidence_missing_native_control_use")
    if not checks["material_reference_headroom"]:
        blockers.append(
            "reference_headroom_nonpositive"
            if (source_gate_evidence.get("task_headroom") or {}).get("status")
            == "failed"
            else "reference_headroom_unproven"
        )
    if contradicted:
        blockers.append("strategy_depth_contradicted")
    for check in _hard_checks():
        if not checks[check] and check not in {
            "current_protocol_semantics",
            "identity_bound_across_artifacts",
            "scenario_signature_current",
            "native_backend_executable",
            "source_consumption_passed",
            "deterministic_replay_passed",
            "task_contract_passed",
            "terminal_integrity_passed",
            "exogenous_state_evolution_observed",
            "world_change_contract_declared",
            "material_exogenous_change_observed",
            "post_change_decision_observed",
            "difficulty_appropriate_control_response_observed",
            "agent_action_backend_effect_observed",
            "predesigned_change_or_disruption_reached",
            "parallel_simulator_agent_clock_observed",
            "successful_reference_used_native_control",
            "material_reference_headroom",
            "difficulty_not_contradicted",
        }:
            blockers.append(f"check_failed:{check}")
    blockers = sorted(set(blockers))
    if admission_profile == QUALITY_CORE_V2_ADMISSION_PROFILE:
        admission_blockers = sorted(
            f"check_failed:{check}"
            for check in agentic_admission_check_names(
                difficulty_level=difficulty_level
            )
            if not checks.get(check, False)
        )
    else:
        admission_blockers = list(blockers)
    status = (
        "retired"
        if RETIRED_BLOCKERS.intersection(admission_blockers)
        else ("passed" if not admission_blockers else "held")
    )
    return {
        "scenario_id": scenario_id,
        "scenario_signature": signature,
        "domain": source_row.get("domain"),
        "backend_kind": source_row.get("backend_kind"),
        "status": status,
        "admission_profile": admission_profile,
        "admission_blockers": admission_blockers,
        "diagnostic_blockers": blockers,
        "blockers": blockers,
        "checks": checks,
        "agentic_contract": {
            "policy_contract": dict(source_row.get("policy_contract") or {}),
            "tool_contract": {
                "native_control_tool_names": sorted(set(map(str, control_tools))),
                "state_changing_tool_calls": int(
                    evidence.get("state_changing_tool_calls") or 0
                ),
            },
            "task_contract": dict(task),
            "world_evolution_contract": {
                key: evidence.get(key)
                for key in (
                    "simulator_ticks",
                    "clock_semantics",
                    "simulator_tick_sequence",
                    "simulator_owned_clock_observed",
                    "model_decision_ticks",
                    "provider_calls",
                    "simulator_advance_without_model_ticks",
                    "autonomous_hold_ticks",
                    "pending_action_hold_ticks",
                    "scheduled_review_ticks",
                    "periodic_scan_ticks",
                    "continuous_supervisory_review_ticks",
                    "actual_supervisory_review_observed",
                    "standing_plan_committed",
                    "standing_plan_commit_ticks",
                    "standing_control_commit_ticks",
                    "persistent_policy_review_records",
                    "persistent_policy_review_bindings",
                    "persistent_policy_effect_bindings",
                    "persistent_policy_attribution",
                    "persistent_control_tool_names",
                    "wake_reason_counts",
                    "visible_interrupt_count",
                    "declared_predesigned_event_ids",
                    "declared_predesigned_event_ticks",
                    "realized_predesigned_event_ids",
                    "realized_predesigned_event_ticks",
                    "exogenous_state_change_ticks",
                    "agent_caused_state_change_ticks",
                    "source_consumption_ticks",
                    "state_change_ticks",
                    "world_change_contract_declared",
                    "material_exogenous_event_records",
                    "source_scheduled_event_records",
                    "declared_perturbation_event_records",
                    "endogenous_completion_event_records",
                    "post_change_decision_ticks",
                    "event_to_decision_action_edges",
                    "adaptive_replanning_observed",
                    "valid_plan_delegation_observed",
                    "delegated_plan_opportunity_ticks",
                    "agent_action_backend_effect_observed",
                )
            },
            "decision_process_contract": {
                "decision_graph_present": graph_present,
                "decision_graph_acyclic": graph_acyclic,
                "bounded_or_exact_depth_available": depth_available,
                "difficulty_level": difficulty_level,
                "required_control_response": required_control_response,
                "active_replanning_required": active_replanning_required,
                "standing_plan_observed": standing_plan_observed,
                "persistent_policy_review_observed": bool(
                    persistent_policy["review_observed"]
                ),
                "persistent_policy_attribution_passed": bool(
                    persistent_policy["attribution_passed"]
                ),
                "persistent_policy_timeline_observed": bool(
                    persistent_policy["timeline_observed"]
                ),
                "standing_plan_timeline_observed": standing_plan_timeline_observed,
                "adaptive_control_observed": adaptive_control_observed,
                "delegation_after_change_observed": delegation_after_change_observed,
            },
            "source_contract": dict(source_gate),
            "source_consumption_contract": dict(source_consumption),
        },
    }


def build_agentic_contract_report(
    *,
    source_suite: dict[str, Any],
    behavioral: dict[str, Any],
    task_contracts: dict[str, Any],
    complexity: dict[str, Any],
    observed_depth: dict[str, Any],
    strategy_depth: dict[str, Any],
    source_grounded: dict[str, Any],
    source_consumption: dict[str, Any],
    input_bindings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the Protocol-2.1 per-row agentic admission report."""
    admission_profile = resolve_protocol21_admission_profile(source_suite)
    source_row_profile_mismatches = [
        str(row.get("scenario_id") or "")
        for row in _rows(source_suite)
        if declared_protocol21_admission_profile(row) not in (None, admission_profile)
    ]
    if source_row_profile_mismatches:
        raise ValueError(
            "source row admission profile mismatch: "
            + ", ".join(sorted(source_row_profile_mismatches))
        )
    reports = {
        "behavioral": behavioral,
        "task_contracts": task_contracts,
        "complexity": complexity,
        "observed_depth": observed_depth,
        "strategy_depth": strategy_depth,
        "source_grounded": source_grounded,
        "source_consumption": source_consumption,
    }
    semantics_current = all(_is_current(report) for report in reports.values())
    indexes = {name: _index(report) for name, report in reports.items()}
    results = []
    for source_row in _rows(source_suite):
        scenario_id = str(source_row.get("scenario_id") or "")
        results.append(
            validate_agentic_row(
                source_row=source_row,
                behavioral_rows=indexes["behavioral"].get(scenario_id, []),
                task_rows=indexes["task_contracts"].get(scenario_id, []),
                complexity_rows=indexes["complexity"].get(scenario_id, []),
                observed_depth_rows=indexes["observed_depth"].get(scenario_id, []),
                strategy_depth_rows=indexes["strategy_depth"].get(scenario_id, []),
                source_grounded_rows=indexes["source_grounded"].get(scenario_id, []),
                source_consumption_rows=indexes["source_consumption"].get(
                    scenario_id, []
                ),
                semantics_current=semantics_current,
                admission_profile=admission_profile,
            )
        )
    counts = {
        status: sum(row["status"] == status for row in results)
        for status in ("passed", "held", "retired")
    }
    reports_complete = all(
        report.get("status") == "complete" or report.get("complete") is True
        for report in reports.values()
    )
    return {
        "schema_version": "1.0",
        "admission_profile": admission_profile,
        "status": "complete" if reports_complete else "partial",
        "evaluation_semantics": dict(REQUIRED_SEMANTICS),
        "implementation_tree_sha256": implementation_identity()[
            "implementation_tree_sha256"
        ],
        "input_bindings": input_bindings,
        "n_expected": len(_rows(source_suite)),
        "n_completed": len(results),
        "n_passed": counts["passed"],
        "n_held": counts["held"],
        "n_retired": counts["retired"],
        "summary": {
            "status_counts": counts,
            "blocker_counts": {
                blocker: sum(blocker in row["blockers"] for row in results)
                for blocker in sorted(
                    {blocker for row in results for blocker in row["blockers"]}
                )
            },
        },
        "results": sorted(results, key=lambda row: row["scenario_id"]),
    }
