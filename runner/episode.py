"""runner.episode — single-episode runner (P3-2).

Moved verbatim from ``run.py``: ``run_one`` plus its tightly-coupled
helpers (``_run_episode_loop``, ``_public_agent_config``,
``_summarize_trajectory``, ``_record_stale_observations``,
``_collect_multi_turn_drafts``, ``_multi_turn_draft_to_dict``,
``_maybe_lp_optimum``, ``_LP_OPTIMUM_CACHE``, ``_recompute_signature``).
``run.py`` re-imports these to preserve every existing public/private
import path.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from datetime import UTC
from pathlib import Path
from typing import Any

from baselines import make_agent
from baselines.llm_agent import public_provider_url, redact_provider_error
from core import Action, EvidenceLogger, ToolCall
from core.event_protocol import (
    EVENT_DECISION_CONTRACT_VERSION,
    OPTIONAL_PLAN_WAKE_REASONS,
    audit_event_decision_contract,
    resolve_event_decision,
)
from core.world_evolution_contract import canonicalize_runtime_events
from core.tool_protocol import is_infrastructure_tool_failure
from data import EpisodeHeader, TrajectoryLogger, analyze_trajectory_steps
from domains.registry import (
    DomainSpec,
    build_backend_records,
    get_backend_capability,
    get_domain_spec,
    reference_optimum_from_backend_config,
    reference_optimum_objective_component,
)
from evaluation import (
    ScoringInputs,
    classify_tool_semantic_histogram,
    domain_counterfactual_report,
    evaluate_foresight,
    evaluate_operational_agency,
    evaluate_task_completion,
    score_episode,
    separate_task_outcome_and_process,
    summarize_decision_impact,
)
from runner.resume import recompute_signature_with_seed

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger(__name__)
EVALUATION_PROTOCOL_VERSION = "2.1"
EVALUATION_IMPLEMENTATION_FINGERPRINT = (
    "protocol-2.1-v21-quality-maximal-five-group-v14"
)
MAX_WITHIN_TICK_INVESTIGATION_CALLS = 2
WITHIN_TICK_COMMIT_CALL_RESERVE = 1
def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_value(value: Any) -> Any:
    """Return a detached JSON value or fail closed on non-JSON input."""
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    )


_NON_COMMIT_TOOL_NAMES = {
    "wait",
    "noop",
    "commit_to_plan",
    "moral_choice",
}


def _tool_surface_contract(
    env: Any,
    scenario: dict[str, Any],
) -> dict[str, Any]:
    """Bind one sample's actual exposed tools to its backend capability."""

    schemas = list(env.get_tool_specs() or [])
    exposed = sorted(
        {
            str((schema.get("function") or {}).get("name") or "")
            for schema in schemas
            if isinstance(schema, dict)
            and str((schema.get("function") or {}).get("name") or "")
        }
    )
    backend_kind = str(scenario.get("backend_kind") or "")
    try:
        capability = get_backend_capability(backend_kind)
    except KeyError as exc:
        return {
            "schema_version": "tool-surface-contract-v1",
            "backend_kind": backend_kind,
            "complete": False,
            "reason": str(exc),
            "exposed_tool_names": exposed,
            "exposed_schema_sha256": _canonical_sha256(schemas),
        }
    declared_observation = sorted(set(capability.observation_tools))
    declared_control = sorted(set(capability.control_tools))
    exposed_names = set(exposed)
    readonly = set(env.readonly_tool_names() or set()).intersection(exposed_names)
    effective_commit = sorted(
        exposed_names - readonly - _NON_COMMIT_TOOL_NAMES
    )
    missing_observation = sorted(set(declared_observation) - exposed_names)
    missing_control = sorted(set(declared_control) - exposed_names)
    missing_commit_control = sorted(
        set(declared_control) - set(effective_commit)
    )
    declared = set(declared_observation).union(declared_control)
    return {
        "schema_version": "tool-surface-contract-v1",
        "backend_kind": backend_kind,
        "complete": not (
            missing_observation or missing_control or missing_commit_control
        ),
        "exposed_tool_names": exposed,
        "declared_observation_tool_names": declared_observation,
        "declared_control_tool_names": declared_control,
        "effective_commit_tool_names": effective_commit,
        "missing_observation_tool_names": missing_observation,
        "missing_control_tool_names": missing_control,
        "missing_commit_control_tool_names": missing_commit_control,
        "exposed_undeclared_tool_names": sorted(
            exposed_names - declared - _NON_COMMIT_TOOL_NAMES
        ),
        "exposed_schema_sha256": _canonical_sha256(schemas),
    }


def _agent_visible_step_evidence_ids(
    *,
    realized_events: list[dict[str, Any]],
    step_evidence_ids: list[str],
    tool_results: list[Any],
) -> list[str]:
    """Filter evidence IDs to what the agent can actually observe.

    Some adapters log a hidden realized event and return the logger-generated
    ID only through ``StepInfo.evidence_ids`` without attaching it to the
    event.  When that happens the ID cannot be distinguished from other
    same-tick evidence, so fail closed and expose only IDs proven visible by
    a ToolResult or a non-hidden event.
    """
    tool_ids = {
        str(evidence_id)
        for result in tool_results
        for evidence_id in [
            getattr(result, "evidence_id", None),
            *(getattr(result, "produces_evidence_ids", None) or []),
        ]
        if evidence_id
    }
    hidden_present = any(event.get("hidden") for event in realized_events)
    if not hidden_present:
        allowed = {str(value) for value in step_evidence_ids if value}
    else:
        visible_event_ids = {
            str(evidence_id)
            for event in realized_events
            if not event.get("hidden")
            for evidence_id in [
                event.get("evidence_id"),
                *(event.get("evidence_ids") or []),
            ]
            if evidence_id
        }
        allowed = tool_ids | visible_event_ids
    ordered = [
        str(value)
        for value in step_evidence_ids
        if value and str(value) in allowed
    ]
    for evidence_id in sorted(tool_ids):
        if evidence_id not in ordered:
            ordered.append(evidence_id)
    return ordered


def _configured_model_decision_budget(observation: dict[str, Any]) -> int | None:
    cadence = observation.get("decision_cadence") or {}
    if not isinstance(cadence, dict):
        return None
    try:
        value = int(cadence.get("model_decision_budget"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _baseline_scan_interval(
    observation: dict[str, Any],
    configured_interval: int | None,
) -> int | None:
    if configured_interval is not None:
        return _positive_int(configured_interval)
    cadence = observation.get("decision_cadence") or {}
    if not isinstance(cadence, dict):
        return None
    intervals = [
        interval
        for interval in (
            _positive_int(cadence.get("periodic_scan_every_ticks")),
            _positive_int(cadence.get("max_review_after_ticks")),
        )
        if interval is not None
    ]
    return min(intervals) if intervals else None


def _hold_while_actions_pending(observation: dict[str, Any]) -> bool:
    cadence = observation.get("decision_cadence") or {}
    return bool(
        isinstance(cadence, dict)
        and cadence.get("hold_while_actions_pending", False)
    )


def _update_pending_action_deadlines(
    deadlines: dict[str, int],
    tool_results: list[Any],
) -> None:
    """Track asynchronous calls until their result materializes."""
    for index, result in enumerate(tool_results):
        call_id = str(getattr(result, "call_id", "") or f"anonymous-{index}")
        payload = getattr(result, "payload", None) or {}
        if isinstance(payload, dict) and payload.get("_status") == "pending":
            try:
                deadlines[call_id] = int(payload["due_tick"])
            except (KeyError, TypeError, ValueError):
                continue
        elif call_id in deadlines:
            del deadlines[call_id]


def _tool_trace_edges(
    action: Action,
    tool_results: list[Any],
    *,
    realized_events: list[dict[str, Any]] | None = None,
    applied_tick: int | None = None,
    request_tick: int | None = None,
    known_calls: dict[str, ToolCall] | None = None,
    known_call_ticks: dict[str, int] | None = None,
    visible_evidence_ids: set[str] | None = None,
    evidence_logger: EvidenceLogger | None = None,
) -> list[dict[str, Any]]:
    """Attach explicit action/evidence edges to materialized tool results.

    Adapters already emit native agent_caused events when a physical effect is
    observed. This runner-level normalization joins those events back to the
    originating call for every backend, including read-only and pending
    results. An acknowledgement alone never creates an effect_tick; that
    field is set only by a matching native action-effect event.
    """
    calls_by_id = {
        str(call.call_id): call
        for call in action.tool_calls
        if call.call_id is not None
    }
    calls_by_id_with_history = dict(known_calls or {})
    calls_by_id_with_history.update(calls_by_id)
    calls_by_name: dict[str, list[ToolCall]] = {}
    for call in [*(known_calls or {}).values(), *action.tool_calls]:
        calls_by_name.setdefault(call.name, []).append(call)
    used_call_ids: set[str] = set()
    effect_by_call: dict[str, list[dict[str, Any]]] = {}
    for event in realized_events or []:
        if not isinstance(event, dict):
            continue
        # Native logistics backends use the canonical contract's string
        # ``origin`` field, while a few older adapters still emit the
        # boolean compatibility flag.  Accept both representations here so
        # an acknowledged tool is only promoted to an effect edge when the
        # backend has explicitly attributed the later state change to it.
        if not (
            event.get("agent_caused") is True
            or str(event.get("origin") or "") == "agent_caused"
        ):
            continue
        call_id = str(event.get("call_id") or "")
        if not call_id:
            continue
        effect_by_call.setdefault(call_id, []).append(event)
    authoritative_effect_ids_by_call: dict[str, list[str]] = {}
    for item in evidence_logger.items() if evidence_logger is not None else []:
        payload = item.payload or {}
        call_id = str(payload.get("call_id") or "")
        if not (
            call_id
            and item.kind == "realized_event"
            and item.source == "engine"
            and (
                payload.get("agent_caused") is True
                or str(payload.get("origin") or "") == "agent_caused"
            )
        ):
            continue
        authoritative_effect_ids_by_call.setdefault(call_id, []).append(
            item.evidence_id
        )

    edges: list[dict[str, Any]] = []
    for index, result in enumerate(tool_results):
        call_id = str(getattr(result, "call_id", "") or "")
        call = calls_by_id_with_history.get(call_id) if call_id else None
        if call is None and not call_id:
            candidates = calls_by_name.get(str(getattr(result, "name", "")), [])
            call = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.call_id and str(candidate.call_id) not in used_call_ids
                ),
                None,
            )
        if call is not None:
            call_id = str(call.call_id or "")
            used_call_ids.add(call_id)
            if not getattr(result, "call_id", None):
                result.call_id = call.call_id
            consumed = list(call.consumes_evidence_ids or [])
            if visible_evidence_ids is not None:
                consumed = [
                    evidence_id
                    for evidence_id in consumed
                    if evidence_id in visible_evidence_ids
                ]
            result.consumes_evidence_ids = consumed or None
            result.depends_on_call_ids = list(call.depends_on_call_ids or []) or None

        produced: list[str] = []
        for evidence_id in list(getattr(result, "produces_evidence_ids", None) or []) + [
            getattr(result, "evidence_id", None)
        ]:
            if isinstance(evidence_id, str) and evidence_id and evidence_id not in produced:
                produced.append(evidence_id)
        native_effects = effect_by_call.get(call_id, [])
        backend_effect_evidence_ids = list(
            authoritative_effect_ids_by_call.get(call_id, [])
        )
        effect_tick: int | None = None
        for event in native_effects:
            for evidence_id in event.get("evidence_ids") or []:
                if (
                    isinstance(evidence_id, str)
                    and evidence_id
                    and evidence_id not in produced
                ):
                    produced.append(evidence_id)
            event_tick = event.get("outcome_tick", event.get("applied_tick"))
            try:
                effect_tick = int(event_tick)
            except (TypeError, ValueError):
                if applied_tick is not None:
                    effect_tick = int(applied_tick)
        result.produces_evidence_ids = produced or None
        result.effect_tick = effect_tick
        original_request_tick = (
            (known_call_ticks or {}).get(call_id)
            if call_id
            else None
        )
        if original_request_tick is None:
            original_request_tick = request_tick
        if original_request_tick is None:
            original_request_tick = applied_tick
        edges.append(
            {
                "index": index,
                "call_id": call_id or None,
                "tool_name": str(getattr(result, "name", "")),
                "consumes_evidence_ids": list(
                    getattr(result, "consumes_evidence_ids", None) or []
                ),
                "produces_evidence_ids": list(
                    getattr(result, "produces_evidence_ids", None) or []
                ),
                "backend_effect_evidence_ids": backend_effect_evidence_ids,
                "depends_on_call_ids": list(
                    getattr(result, "depends_on_call_ids", None) or []
                ),
                "request_tick": original_request_tick,
                "effect_tick": effect_tick,
                "state_changing": bool(getattr(result, "state_changing", False)),
                "effect_proven": bool(
                    native_effects
                    and backend_effect_evidence_ids
                    and effect_tick is not None
                ),
            }
        )
    return edges


def _build_event_response_records(
    analysis_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join native exogenous events to proven backend action effects.

    The join is deliberately strict: an acknowledgement is insufficient.  A
    record is emitted only when a native ``agent_caused`` event names both its
    causal parent event and originating call, and that call has a proven
    state-changing effect edge.
    """

    events: dict[str, dict[str, Any]] = {}
    effects: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for step in analysis_steps:
        action_tick = int(step.get("action_tick", step.get("tick", 0)) or 0)
        info = step.get("info") or {}
        extra = info.get("extra") if isinstance(info, dict) else {}
        world_records = (
            extra.get("world_evolution_records")
            if isinstance(extra, dict)
            else []
        ) or []
        trace_edges = {
            str(edge.get("call_id") or ""): edge
            for edge in (step.get("tool_trace_edges") or [])
            if isinstance(edge, dict) and edge.get("call_id")
        }
        for event in world_records:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id") or "")
            if not event_id:
                continue
            if str(event.get("origin") or "") != "agent_caused":
                events[event_id] = event
                continue
            call_id = str(event.get("call_id") or "")
            edge = trace_edges.get(call_id)
            if (
                call_id
                and edge is not None
                and edge.get("state_changing") is True
                and edge.get("effect_proven") is True
            ):
                effects.append((event, edge, action_tick))

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for effect, edge, action_tick in effects:
        parent_id = str(effect.get("causal_parent_event_id") or "")
        call_id = str(effect.get("call_id") or "")
        parent = events.get(parent_id)
        if parent is None or (parent_id, call_id) in seen:
            continue
        seen.add((parent_id, call_id))
        parent_ids = [
            str(value)
            for value in (parent.get("evidence_ids") or [])
            if value
        ]
        effect_ids = list(
            dict.fromkeys(
                str(value)
                for value in [
                    *(effect.get("evidence_ids") or []),
                    *(edge.get("produces_evidence_ids") or []),
                ]
                if value
            )
        )
        if not effect_ids:
            continue
        event_tick = int(
            parent.get(
                "applied_tick",
                parent.get("trigger_tick", parent.get("tick", 0)),
            )
            or 0
        )
        effect_tick = int(
            edge.get(
                "effect_tick",
                effect.get("outcome_tick", effect.get("applied_tick", action_tick)),
            )
            or action_tick
        )
        request_tick = edge.get("request_tick")
        request_tick = int(action_tick if request_tick is None else request_tick)
        consumes_parent_evidence = bool(
            set(parent_ids).intersection(edge.get("consumes_evidence_ids") or [])
        )
        deadline = parent.get(
            "response_deadline_tick",
            parent.get("mandatory_response_tick"),
        )
        records.append(
            {
                "event_id": parent_id,
                "causal_parent_event_id": parent_id,
                "call_id": call_id,
                "event_origin": str(parent.get("origin") or "") or None,
                "declared_perturbation": (
                    str(parent.get("origin") or "") == "declared_perturbation"
                ),
                "event_tick": event_tick,
                "visibility": parent.get("visibility") or (
                    "hidden" if parent.get("hidden") else "visible"
                ),
                "surprise": bool(parent.get("surprise", False)),
                "first_observed_tick": (
                    request_tick if consumes_parent_evidence else None
                ),
                "first_investigation_tick": None,
                "first_control_call_tick": request_tick,
                "first_effect_tick": effect_tick,
                "mandatory_response_tick": (
                    int(deadline) if deadline is not None else None
                ),
                "response_status": "causal",
                "observation_evidence_ids": parent_ids,
                "trigger_evidence_ids": parent_ids,
                "action_consumes_evidence_ids": list(
                    edge.get("consumes_evidence_ids") or []
                ),
                "action_evidence_ids": effect_ids,
                "backend_effect_evidence_ids": effect_ids,
                "outcome_evidence_ids": effect_ids,
            }
        )
    return records


def _normalize_event_response_evidence_ids(
    event_response_records: list[dict[str, Any]],
    *,
    valid_evidence_ids: set[str],
) -> list[dict[str, Any]]:
    """Keep event-response evidence claims bound to the episode logger."""

    normalized: list[dict[str, Any]] = []
    for raw_record in event_response_records:
        record = dict(raw_record)
        for key, values in record.items():
            if not key.endswith("_evidence_ids"):
                continue
            raw_ids = values if isinstance(values, list) else []
            record[key] = list(
                dict.fromkeys(
                    evidence_id
                    for evidence_id in raw_ids
                    if isinstance(evidence_id, str)
                    and evidence_id in valid_evidence_ids
                )
            )
        normalized.append(record)
    return normalized


def _bind_event_response_masked_replays(
    event_response_records: list[dict[str, Any]],
    *,
    per_action: list[dict[str, Any]],
    per_action_groups: list[dict[str, Any]],
) -> dict[str, float]:
    """Bind each runtime response to an exact masked replay result.

    A repeated-control group is eligible only when it contains the response
    call and every grouped call occurs after the event and no later than the
    proven backend effect.  This prevents a same-tool action from a later,
    unrelated event being folded into the response.  Ambiguous groups fail
    closed to the ordinary single-call attribution.
    """

    individual: dict[str, float] = {}
    for row in per_action:
        call_id = str(row.get("call_id") or "")
        value = row.get("marginal_prevented_loss")
        if not call_id or isinstance(value, bool):
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed == parsed and abs(parsed) != float("inf"):
            individual[call_id] = parsed

    bindings: dict[str, float] = {}
    for record in event_response_records:
        call_id = str(record.get("call_id") or "")
        if not call_id:
            continue
        try:
            event_tick = int(record.get("event_tick"))
            effect_tick = int(record.get("first_effect_tick"))
        except (TypeError, ValueError):
            event_tick = -1
            effect_tick = -1
        eligible: list[tuple[dict[str, Any], float]] = []
        for group in per_action_groups:
            group_id = str(group.get("group_id") or "")
            call_ids = group.get("call_ids")
            ticks = group.get("ticks")
            value = group.get("masked_action_group_delta")
            if (
                not group_id
                or not isinstance(call_ids, list)
                or call_id not in call_ids
                or not isinstance(ticks, list)
                or not ticks
                or isinstance(value, bool)
            ):
                continue
            try:
                parsed_ticks = [int(tick) for tick in ticks]
                parsed_value = float(value)
            except (TypeError, ValueError):
                continue
            if (
                event_tick < 0
                or effect_tick < event_tick
                or min(parsed_ticks) < event_tick
                or max(parsed_ticks) > effect_tick
                or parsed_value != parsed_value
                or abs(parsed_value) == float("inf")
            ):
                continue
            eligible.append((group, parsed_value))
        if len(eligible) == 1:
            group, value = eligible[0]
            record["masked_action_group_id"] = str(group["group_id"])
            record["masked_action_group_call_ids"] = [
                str(value) for value in group["call_ids"]
            ]
            record["masked_action_group_delta"] = value
            record["first_control_call_tick"] = min(
                int(tick) for tick in group["ticks"]
            )
            bindings[call_id] = value
            continue
        control_tick = record.get("first_control_call_tick")
        try:
            parsed_control_tick = int(control_tick)
        except (TypeError, ValueError):
            parsed_control_tick = -1
        if (
            call_id in individual
            and event_tick >= 0
            and event_tick <= parsed_control_tick <= effect_tick
        ):
            record["masked_action_group_delta"] = individual[call_id]
            bindings[call_id] = individual[call_id]
    return bindings


def _operational_agency_artifacts(
    *,
    env: Any,
    ground_truth: dict[str, Any],
    analysis_steps: list[dict[str, Any]],
    counterfactual: Any,
) -> tuple[list[dict[str, Any]], set[str], dict[str, Any]]:
    """Build the authoritative causal profile before headline scoring."""
    evidence_logger = getattr(env, "evidence", None)
    valid_evidence_ids = (
        {item.evidence_id for item in evidence_logger.items()}
        if evidence_logger is not None
        else set()
    )
    raw_records = ground_truth.get("event_response_records") or []
    records = (
        [dict(record) for record in raw_records]
        if isinstance(raw_records, list)
        and all(isinstance(record, dict) for record in raw_records)
        else []
    )
    if not records:
        records = _build_event_response_records(analysis_steps)
    records = _normalize_event_response_evidence_ids(
        records,
        valid_evidence_ids=valid_evidence_ids,
    )
    masked_replay_by_call_id = _bind_event_response_masked_replays(
        records,
        per_action=counterfactual.per_action,
        per_action_groups=counterfactual.per_action_groups,
    )
    if counterfactual.applicable:
        for record in records:
            record["native_burden_before"] = float(
                counterfactual.counterfactual_cost
            )
            record["native_burden_basis"] = (
                "episode_replay_no_action_cost"
            )
    profile = evaluate_operational_agency(
        records,
        valid_evidence_ids=valid_evidence_ids,
        masked_replay_by_call_id=masked_replay_by_call_id,
    )
    return records, valid_evidence_ids, profile


def _proven_tool_effect_evidence_by_call_id(
    analysis_steps: list[dict[str, Any]],
    *,
    evidence_logger: Any,
) -> dict[str, list[str]]:
    """Return only state changes with runner-proven, in-ledger effects."""

    effect_evidence_by_call_id: dict[str, list[str]] = {}
    for item in evidence_logger.items() if evidence_logger is not None else []:
        payload = item.payload or {}
        call_id = str(payload.get("call_id") or "")
        if not (
            call_id
            and item.kind == "realized_event"
            and item.source == "engine"
            and (
                payload.get("agent_caused") is True
                or str(payload.get("origin") or "") == "agent_caused"
            )
        ):
            continue
        effect_evidence_by_call_id.setdefault(call_id, []).append(
            item.evidence_id
        )

    proven: dict[str, list[str]] = {}
    for step in analysis_steps:
        for edge in step.get("tool_trace_edges") or []:
            if not isinstance(edge, dict) or not (
                edge.get("state_changing") is True
                and edge.get("effect_proven") is True
            ):
                continue
            call_id = str(edge.get("call_id") or "")
            effect_ids = effect_evidence_by_call_id.get(call_id, [])
            if call_id and effect_ids:
                proven[call_id] = list(
                    dict.fromkeys([*proven.get(call_id, []), *effect_ids])
                )
    return proven


def _raise_on_infrastructure_tool_failure(tool_results: list[Any]) -> None:
    infrastructure_failure = next(
        (
            result
            for result in tool_results
            if is_infrastructure_tool_failure(result)
        ),
        None,
    )
    if infrastructure_failure is not None:
        raise RuntimeError(
            "infrastructure_tool_failure:"
            f"{infrastructure_failure.name}:"
            f"{infrastructure_failure.error_code}"
        )


def _tool_feedback_interrupt_reasons(
    observation: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    for result in observation.get("__last_tool_results__") or []:
        if not bool(result.get("ok")):
            reasons.append("tool_failure")
        elif (
            str((result.get("payload") or {}).get("_status") or "").lower()
            != "pending"
            and int(result.get("latency_ticks", 0) or 0) > 0
            and str(result.get("name") or "") != "commit_to_plan"
        ):
            reasons.append("delayed_tool")
    return sorted(set(reasons))


def _autonomy_interrupt_reasons(
    observation: dict[str, Any],
    *,
    include_tool_feedback: bool = True,
) -> list[str]:
    reasons: list[str] = []
    for event in observation.get("__last_realized_events__") or []:
        resolution = resolve_event_decision(event)
        if resolution.requires_decision and resolution.interrupt_reason:
            reasons.append(resolution.interrupt_reason)
    if observation.get("__last_early_stop_warnings__"):
        reasons.append("safety_warning")
    if observation.get("__last_forecast_updates__"):
        reasons.append("forecast_update")
    if observation.get("active_dilemmas"):
        reasons.append("active_dilemma")
    if include_tool_feedback:
        reasons.extend(_tool_feedback_interrupt_reasons(observation))
    return sorted(set(reasons))


def _terminal_response_window_reasons(
    events: list[dict[str, Any]] | None,
) -> list[str]:
    """Return release blockers carried by the full, unfiltered event stream.

    Hidden events are intentionally removed from the agent observation, but a
    backend must still be able to prove that a terminal response window was
    served.  This helper is only used at the terminal boundary and therefore
    cannot leak hidden event contents into the prompt.
    """
    return sorted(
        {
            "terminal_response_window_missing"
            for event in events or []
            if isinstance(event, dict)
            and event.get("response_window_required") is True
            and event.get("terminal_response_window_missing") is True
        }
    )


def _native_decision_opportunity(
    observation: dict[str, Any],
    *,
    initial: bool,
    interrupt_reasons: list[str],
    agent_owns_review_schedule: bool,
) -> bool:
    """Whether the canonical wakeup policy warrants a model decision."""
    if initial or interrupt_reasons:
        return True
    cadence = observation.get("decision_cadence") or {}
    if agent_owns_review_schedule and isinstance(cadence, dict) and (
        cadence.get("cadence_contract") == "agent_scheduled_v1"
        or cadence.get("harness_periodic_supervisory_scan") is False
    ):
        return False
    explicit = observation.get("decision_opportunity")
    if isinstance(explicit, bool):
        return explicit
    if not isinstance(cadence, dict):
        return False
    explicit = cadence.get("native_opportunity")
    return explicit if isinstance(explicit, bool) else False


def _cadence_contract_declared(observation: dict[str, Any]) -> bool:
    if isinstance(observation.get("decision_opportunity"), bool):
        return True
    cadence = observation.get("decision_cadence") or {}
    return bool(
        isinstance(cadence, dict)
        and (
            cadence.get("cadence_contract") == "agent_scheduled_v1"
            or cadence.get("review_owner") == "agent"
            or isinstance(cadence.get("native_opportunity"), bool)
            or _positive_int(cadence.get("periodic_scan_every_ticks"))
            is not None
            or _positive_int(cadence.get("max_review_after_ticks")) is not None
        )
    )


def _confirmed_autonomy_window(
    action: Action,
    tool_results: list[Any],
) -> tuple[int, str | None]:
    for call in reversed(action.tool_calls):
        if call.name != "commit_to_plan":
            continue
        matching_results = [
            result
            for result in tool_results
            if result.name == "commit_to_plan"
            and result.ok
            and (
                call.call_id is None
                or result.call_id is None
                or result.call_id == call.call_id
            )
            and str((result.payload or {}).get("_status") or "").lower()
            != "pending"
        ]
        if not matching_results:
            continue
        try:
            review_after_ticks = int(call.args.get("review_after_ticks", 1) or 1)
        except (TypeError, ValueError):
            return 0, None
        hold_ticks = max(0, review_after_ticks - 1)
        return hold_ticks, str(call.args.get("plan_id") or "") or None
    return 0, None


def _materialized_autonomy_window(
    action: Action,
    tool_results: list[Any],
    pending_plan_requests: dict[str, ToolCall],
) -> tuple[int, str | None, str | None, dict[str, Any]]:
    """Activate terminal plan acknowledgements while preserving lineage.

    A single simulator tick can materialize a delayed plan and acknowledge a
    new plan submitted in the current action.  Consume every terminal result
    before returning so stale requests cannot be matched by a later result;
    the latest successful acknowledgement wins, which mirrors the backend's
    ordered action stream and lets a current plan supersede a delayed one.
    """
    current_plan_calls = [
        call for call in action.tool_calls if call.name == "commit_to_plan"
    ]
    for call in current_plan_calls:
        # ToolProtocol normally assigns a call_id before returning a result,
        # but the core ToolCall contract keeps it optional for lightweight
        # adapters and replay fixtures.  Retain anonymous plans as well so a
        # delayed acknowledgement cannot silently lose the standing plan.
        key = str(call.call_id or f"anonymous-plan-{id(call)}")
        pending_plan_requests[key] = call

    matched_call_ids: set[int] = set()
    selected: tuple[int, str | None, str | None, dict[str, Any]] | None = None
    for result in tool_results:
        if result.name != "commit_to_plan":
            continue
        status = str((result.payload or {}).get("_status") or "").lower()
        if status == "pending":
            continue
        call_id = str(result.call_id or "") or None
        pending_key = call_id
        call = pending_plan_requests.get(pending_key or "")
        if call is None and call_id is None:
            # A protocol-compliant result has a call_id, but preserve
            # deterministic matching for adapters that omit it.  There is at
            # most one anonymous standing-plan request in normal operation;
            # choose the oldest retained request if a fixture violates that
            # expectation rather than dropping the acknowledgement.
            anonymous_key, call = next(
                (
                    (key, candidate)
                    for key, candidate in pending_plan_requests.items()
                    if key.startswith("anonymous-plan-")
                ),
                (None, None),
            )
            if anonymous_key is not None:
                pending_key = anonymous_key
        if call is None and len(current_plan_calls) == 1:
            call = current_plan_calls[0]
            if id(call) not in matched_call_ids:
                pending_key = str(call.call_id or "") or None
            else:
                call = None
        if pending_key:
            pending_plan_requests.pop(pending_key, None)
        if call is None:
            continue
        matched_call_ids.add(id(call))
        if not result.ok or status in {"error", "failed", "rejected"}:
            continue
        hold_ticks, plan_id = _confirmed_autonomy_window(
            Action(tool_calls=[call]),
            [result],
        )
        selected = (hold_ticks, plan_id, call_id, dict(call.args))
    return selected or (0, None, None, {})


def _public_agent_config(agent_kwargs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a trajectory-header-safe copy of agent kwargs."""
    from baselines import LLMConfig

    if agent_kwargs is None:
        return None

    def scrub(value: Any) -> Any:
        if isinstance(value, LLMConfig):
            data = {
                field_name: scrub(getattr(value, field_name))
                for field_name in value.__dataclass_fields__
            }
            if data.get("extra_headers"):
                data["extra_headers"] = {
                    str(k): "[redacted]" for k in dict(data["extra_headers"])
                }
            for url_field in ("base_url", "responses_base_url"):
                data[url_field] = public_provider_url(data.get(url_field))
            return data
        if isinstance(value, dict):
            return {str(k): scrub(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [scrub(v) for v in value]
        return value

    public_config = scrub(agent_kwargs)
    config = agent_kwargs.get("config")
    if (
        isinstance(public_config, dict)
        and isinstance(config, LLMConfig)
        and config.interaction_mode == "logical_stateless"
    ):
        public_config["session_context_contract"] = {
            "schema_version": "session-context-contract/1.0",
            "interaction_mode": "logical_stateless",
            "provider_transcript": "fresh_bounded_request_per_decision_epoch",
            "benchmark_managed_episode_projection": [
                "decision_ledger",
                "plan_state",
            ],
            "cross_decision_memoryless": False,
            "intended_use": "historical_protocol21_compatibility",
        }
    return public_config


def _collect_agent_session_artifacts(
    *,
    agent: Any,
    logger: TrajectoryLogger | None,
) -> tuple[Any | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Snapshot bounded agent memory and persist its semantic ledger.

    Accessors are optional so non-LLM baselines remain unaffected.  Agents
    that expose them must return JSON values; malformed audit state fails
    closed instead of silently producing an unverifiable long-horizon run.
    """

    structured_memory: Any | None = None
    get_structured_memory = getattr(agent, "get_structured_memory", None)
    if callable(get_structured_memory):
        structured_memory = _canonical_json_value(get_structured_memory())

    semantic_ledger_artifact: dict[str, Any] | None = None
    get_session_ledger = getattr(agent, "get_session_ledger", None)
    if logger is not None and callable(get_session_ledger):
        ledger = _canonical_json_value(get_session_ledger())
        if not isinstance(ledger, list) or not all(
            isinstance(item, dict) for item in ledger
        ):
            raise TypeError("agent semantic session ledger must be list[dict]")
        semantic_ledger_artifact = logger.write_semantic_ledger(ledger)

    provider_audit_artifact: dict[str, Any] | None = None
    get_interaction_stats = getattr(agent, "get_interaction_stats", None)
    if logger is not None and callable(get_interaction_stats):
        stats = _canonical_json_value(get_interaction_stats() or {})
        if not isinstance(stats, dict):
            raise TypeError("agent interaction stats must be a JSON object")
        audit_items: list[dict[str, Any]] = []
        for record_kind, field in (
            ("provider_request", "provider_request_records"),
            ("provider_response", "provider_response_records"),
        ):
            records = stats.get(field) or []
            if not isinstance(records, list) or not all(
                isinstance(item, dict) for item in records
            ):
                raise TypeError(f"agent {field} must be list[dict]")
            audit_items.extend(
                {"record_kind": record_kind, **item} for item in records
            )
        if audit_items:
            provider_audit_artifact = logger.write_provider_audit(audit_items)

    return structured_memory, semantic_ledger_artifact, provider_audit_artifact


def _finalize_failed_episode_audit(
    *,
    agent: Any,
    logger: TrajectoryLogger | None,
    error: BaseException,
    error_stage: str,
) -> dict[str, Any]:
    """Persist partial agent/provider state before an episode abort escapes."""

    details: dict[str, Any] = {
        "status": "error",
        "error_type": type(error).__name__,
        "error_stage": error_stage,
    }
    request_sequence = getattr(error, "request_sequence", None)
    if request_sequence is not None:
        details["provider_request_sequence"] = int(request_sequence)
    reset_at = getattr(error, "reset_at", None)
    if reset_at:
        details["provider_reset_at"] = str(reset_at)
    if logger is not None:
        (
            structured_memory,
            semantic_ledger_artifact,
            provider_audit_artifact,
        ) = _collect_agent_session_artifacts(agent=agent, logger=logger)
        if structured_memory is not None:
            details["structured_memory"] = structured_memory
        if semantic_ledger_artifact is not None:
            details["semantic_ledger_artifact"] = semantic_ledger_artifact
        if provider_audit_artifact is not None:
            details["provider_audit_artifact"] = provider_audit_artifact
        logger.finalize(final_score=None, trajectory_summary=details)
    return details


def _run_episode_loop(
    *,
    env: Any,
    agent: Any,
    logger: TrajectoryLogger | None,
    multi_turn: bool = False,
    multi_turn_rounds: int = 3,
    within_tick_interaction: bool = True,
    baseline_scan_interval: int | None = None,
) -> dict[str, Any]:
    """Run the env <-> agent tick loop.

    Extracted from ``run_one`` so v0.2.2 regression tests (T-01, F-03)
    can drive a fake env+agent without spinning up the full PowerGrid
    backend. Returns a dict with:
        - actions: list of Action emitted by the agent
        - final_observation: last observation seen
        - tool_results_ok / tool_results_failed: counters
        - episode_reward_total: sum of per-tick rewards across executed ticks
    """
    if multi_turn and callable(getattr(agent, "start_decision_epoch", None)):
        raise ValueError(
            "multi_turn is incompatible with decision-epoch agents; draft "
            "requests would not be consumed by the already computed action"
        )
    actions: list[Action] = []
    multi_turn_records: list[dict[str, Any]] = []
    rounds_per_tick = max(1, int(multi_turn_rounds))
    tool_results_ok = 0
    tool_results_failed = 0
    stale_observation_records: list[dict[str, Any]] = []
    analysis_steps: list[dict[str, Any]] = []
    within_tick_records: list[dict[str, Any]] = []
    control_reconciliation_records: list[dict[str, Any]] = []
    autonomy_records: list[dict[str, Any]] = []
    autonomy_ticks_remaining = 0
    autonomy_plan_id: str | None = None
    autonomy_plan_active = False
    active_wake_if = set(OPTIONAL_PLAN_WAKE_REASONS)
    active_plan_expires_at_tick: int | None = None
    requested_review_tick: int | None = None
    scheduled_review_ticks: list[int] = []
    periodic_scan_ticks: list[int] = []
    decision_epochs: list[dict[str, Any]] = []
    provider_retry_tick: int | None = None
    provider_failure_count = 0
    provider_request_record_count = 0
    provider_response_record_count = 0
    next_baseline_scan_tick: int | None = None
    model_decision_ticks = 0
    autonomous_hold_ticks = 0
    decision_budget_hold_ticks = 0
    decision_budget_interrupt_overruns = 0
    pending_action_hold_ticks = 0
    native_idle_hold_ticks = 0
    decision_opportunities = 0
    decision_opportunity_ticks: list[int] = []
    missed_actionable_opportunities = 0
    delegated_plan_opportunities = 0
    delegated_plan_opportunity_ticks: list[int] = []
    deferred_pending_opportunities = 0
    pending_action_deadlines: dict[str, int] = {}
    known_tool_calls: dict[str, ToolCall] = {}
    known_tool_call_ticks: dict[str, int] = {}
    agent_visible_evidence_ids: set[str] = set()
    pending_plan_requests: dict[str, ToolCall] = {}
    terminal_unanswered_interrupt_reasons: list[str] = []
    terminal_response_window_reasons_seen: list[str] = []
    terminal_response_window_extended_reasons: set[str] = set()
    response_window_extensions = 0
    decision_budget_exhaustion_recorded = False
    world_evolution_records: list[dict[str, Any]] = []
    event_contract_violations: list[dict[str, Any]] = []
    transition_ingestion_attempted = 0
    transition_ingestion_completed = 0
    transition_ingestion_failures: list[dict[str, Any]] = []
    # v0.2.2 F-03: accumulate the true per-episode return for Reflexion-style
    # agents instead of summing over a single-element list at episode end.
    episode_reward_total = 0.0
    obs = env.snapshot()
    # v0.2.1: surface structured per-tick feedback on the NEXT observation
    # under reserved keys so ReAct / Reflexion agents can observe the
    # actual outcome of their prior tools. Pre-existing baselines that
    # ignore these keys are unaffected (purely additive).
    obs["__last_tool_results__"] = []
    obs["__last_realized_events__"] = []
    obs["__last_evidence_ids__"] = []
    obs["__last_reward__"] = 0.0
    obs["__last_early_stop_warnings__"] = []
    obs["__last_forecast_updates__"] = {}
    cadence_contract_declared = _cadence_contract_declared(obs)
    model_decision_budget = _configured_model_decision_budget(obs)
    hold_while_pending = _hold_while_actions_pending(obs)
    obs["__model_decision_budget__"] = model_decision_budget
    obs["__model_decisions_remaining__"] = model_decision_budget
    persistent_session_check = getattr(agent, "_uses_persistent_session", None)
    agent_owns_review_schedule = bool(
        callable(persistent_session_check) and persistent_session_check()
    )
    reference_scan_interval = (
        None
        if agent_owns_review_schedule
        else _baseline_scan_interval(obs, baseline_scan_interval)
    )
    for _ in range(env.horizon):
        within_tick_record: dict[str, Any] | None = None
        control_reconciliation_record: dict[str, Any] | None = None
        decision_envelope: dict[str, Any] | None = None
        investigation_results: list[Any] = []
        current_tick = int(obs.get("tick", getattr(env, "tick", 0)) or 0)
        interrupt_reasons = _autonomy_interrupt_reasons(obs)
        if (
            provider_retry_tick is not None
            and current_tick >= provider_retry_tick
        ):
            interrupt_reasons.append("provider_retry")
        if (
            next_baseline_scan_tick is not None
            and current_tick >= next_baseline_scan_tick
        ):
            interrupt_reasons.append("periodic_scan")
        if autonomy_plan_active:
            interrupt_reasons = [
                reason
                for reason in interrupt_reasons
                if reason not in OPTIONAL_PLAN_WAKE_REASONS
                or reason in active_wake_if
            ]
        if (
            active_plan_expires_at_tick is not None
            and current_tick >= active_plan_expires_at_tick
        ):
            interrupt_reasons.append("plan_expiry")
        if (
            requested_review_tick is not None
            and current_tick >= requested_review_tick
        ):
            interrupt_reasons.append("scheduled_review")
        interrupt_reasons = sorted(set(interrupt_reasons))
        raw_decision_budget_exhausted = (
            model_decision_budget is not None
            and model_decision_ticks >= model_decision_budget
        )
        # Planned supervision is part of the configured decision budget. Only
        # safety/task/terminal/provider-recovery interrupts may force a model
        # call after that budget is exhausted.
        planned_review_reasons = {
            "periodic_scan",
            "scheduled_review",
            "plan_expiry",
        }
        budget_interrupt_reasons = sorted(
            set(interrupt_reasons) - planned_review_reasons
        )
        decision_budget_exhausted = (
            raw_decision_budget_exhausted and not budget_interrupt_reasons
        )
        if raw_decision_budget_exhausted and not decision_budget_exhaustion_recorded:
            autonomy_records.append(
                {
                    "tick": obs.get("tick", getattr(env, "tick", 0)),
                    "kind": "model_decision_budget_exhausted",
                    "configured_budget": model_decision_budget,
                    "forced_interrupt_reasons": budget_interrupt_reasons,
                }
            )
            # Record exhaustion independently of whether a mandatory
            # interrupt temporarily overruns the budget.  Otherwise a forced
            # visible/safety wake would leave the episode telemetry claiming
            # that the budget was never exhausted.
            decision_budget_exhaustion_recorded = True
        if raw_decision_budget_exhausted and budget_interrupt_reasons:
            decision_budget_interrupt_overruns += 1
            autonomy_records.append(
                {
                    "tick": obs.get("tick", getattr(env, "tick", 0)),
                    "kind": "mandatory_interrupt_budget_overrun",
                    "configured_budget": model_decision_budget,
                    "reasons": budget_interrupt_reasons,
                }
            )
        native_decision_opportunity = _native_decision_opportunity(
            obs,
            initial=(len(actions) == 0),
            interrupt_reasons=interrupt_reasons,
            agent_owns_review_schedule=agent_owns_review_schedule,
        )
        if native_decision_opportunity:
            decision_opportunities += 1
            decision_opportunity_ticks.append(current_tick)
        pending_action_hold = bool(
            hold_while_pending
            and pending_action_deadlines
            and current_tick <= max(pending_action_deadlines.values())
            and not interrupt_reasons
        )
        autonomous_hold = autonomy_plan_active and not interrupt_reasons
        native_idle_hold = bool(
            not native_decision_opportunity
            and not pending_action_hold
            and not decision_budget_exhausted
            and not autonomous_hold
        )
        if native_decision_opportunity and decision_budget_exhausted:
            missed_actionable_opportunities += 1
        if native_decision_opportunity and pending_action_hold:
            deferred_pending_opportunities += 1
        if native_decision_opportunity and autonomous_hold:
            delegated_plan_opportunities += 1
            delegated_plan_opportunity_ticks.append(current_tick)
        if autonomy_plan_active and interrupt_reasons:
            planned_review_reasons = {
                "scheduled_review",
                "periodic_scan",
                "plan_expiry",
            }
            autonomy_records.append(
                {
                    "tick": obs.get("tick", getattr(env, "tick", 0)),
                    "kind": (
                        "planned_review"
                        if set(interrupt_reasons).issubset(
                            planned_review_reasons
                        )
                        else "early_wake"
                    ),
                    "plan_id": autonomy_plan_id,
                    "reasons": interrupt_reasons,
                }
            )
            autonomy_ticks_remaining = 0
            autonomy_plan_active = False
            requested_review_tick = None
            active_plan_expires_at_tick = None
        stale_observation_records.extend(
            _record_stale_observations(
                evidence=getattr(env, "evidence", None),
                observation=obs,
                tick=obs.get("tick", getattr(env, "tick", 0)),
            )
        )
        tool_specs = env.get_tool_specs()
        prospective_decision_reasons = list(interrupt_reasons)
        if len(actions) == 0:
            prospective_decision_reasons = ["initial"]
        elif not prospective_decision_reasons:
            prospective_decision_reasons = ["native_opportunity"]
        act_obs = dict(obs)
        act_obs["__decision_epoch__"] = {
            "decision_id": f"decision-{model_decision_ticks + 1}",
            "model_decision_index": model_decision_ticks + 1,
            "reasons": prospective_decision_reasons,
            "state_version": current_tick,
            "simulator_tick": current_tick,
            "deadline_tick": min(
                (
                    tick
                    for tick in (
                        requested_review_tick,
                        active_plan_expires_at_tick,
                    )
                    if tick is not None
                ),
                default=None,
            ),
        }
        precomputed_action: Action | None = None
        precomputed_provider_observation: Any | None = None
        start_decision_epoch = getattr(agent, "start_decision_epoch", None)
        continue_decision_epoch = getattr(agent, "continue_decision_epoch", None)
        investigate = getattr(agent, "investigate", None)
        execute_investigation = getattr(env, "execute_investigation", None)
        readonly_names_fn = getattr(env, "readonly_tool_names", None)
        readonly_names = (
            set(readonly_names_fn() or set())
            if callable(readonly_names_fn)
            else set()
        )
        if pending_action_hold:
            action = Action(
                tool_calls=[],
                dominant="pending_action_hold",
                assistant_text=(
                    "No model call: a submitted domain action is still pending; "
                    "the simulator advances until its result materializes."
                ),
                rationale="domain-native asynchronous action cadence",
            )
            pending_action_hold_ticks += 1
            autonomy_records.append(
                {
                    "tick": current_tick,
                    "kind": "pending_action_hold",
                    "pending_call_ids": sorted(pending_action_deadlines),
                    "next_due_tick": min(pending_action_deadlines.values()),
                }
            )
            drafts: list[dict[str, Any]] = []
        elif decision_budget_exhausted:
            tick = int(obs.get("tick", getattr(env, "tick", 0)) or 0)
            action = Action(
                tool_calls=[],
                dominant="decision_budget_hold",
                assistant_text=(
                    "No model call: the domain-native supervisory decision "
                    "budget is exhausted while the simulator completes."
                ),
                rationale="bounded supervisory decision cadence",
            )
            decision_budget_hold_ticks += 1
            drafts: list[dict[str, Any]] = []
        elif autonomous_hold:
            tick = int(obs.get("tick", getattr(env, "tick", 0)) or 0)
            action = Action(
                # This is a runner scheduling decision, not a model tool call.
                # An empty action still advances the deterministic backend and
                # materializes delayed results, without polluting tool-use
                # efficiency or wait-dominance evidence.
                tool_calls=[],
                dominant="autonomous_plan_hold",
                assistant_text=(
                    "No model call: standing plan remains in force until the "
                    "scheduled review or a mandatory interrupt."
                ),
                rationale="event-adaptive long-horizon plan hold",
            )
            autonomy_ticks_remaining -= 1
            autonomous_hold_ticks += 1
            autonomy_records.append(
                {
                    "tick": tick,
                    "kind": "autonomous_hold",
                    "plan_id": autonomy_plan_id,
                    "remaining_after_tick": autonomy_ticks_remaining,
                }
            )
            drafts: list[dict[str, Any]] = []
        elif native_idle_hold:
            action = Action(
                tool_calls=[],
                dominant="native_idle_hold",
                assistant_text=(
                    "No model call: the simulator has no domain-native "
                    "decision opportunity at this tick."
                ),
                rationale="simulator-owned time progression",
            )
            native_idle_hold_ticks += 1
            autonomy_records.append(
                {"tick": current_tick, "kind": "native_idle_hold"}
            )
            drafts: list[dict[str, Any]] = []
        elif (
            within_tick_interaction
            and (callable(start_decision_epoch) or callable(investigate))
            and callable(execute_investigation)
        ):
            investigation_action: Action | None = None
            if callable(start_decision_epoch):
                candidate = start_decision_epoch(act_obs, tool_specs)
                candidate_names = {call.name for call in candidate.tool_calls}
                investigative_names = candidate_names - {
                    "wait",
                    "noop",
                    "commit_to_plan",
                }
                if (
                    investigative_names
                    and candidate_names.issubset(readonly_names)
                ):
                    investigation_action = candidate
                else:
                    precomputed_action = candidate
                    precomputed_provider_observation = _canonical_json_value(
                        act_obs
                    )
            elif callable(investigate):
                investigation_action = investigate(act_obs, tool_specs)
            if investigation_action is not None and investigation_action.tool_calls:
                requested_calls = list(investigation_action.tool_calls)
                tick_budget = getattr(env, "budget", None)
                max_calls_per_tick = getattr(
                    tick_budget,
                    "max_tool_calls_per_tick",
                    None,
                )
                investigation_call_limit = MAX_WITHIN_TICK_INVESTIGATION_CALLS
                if isinstance(max_calls_per_tick, int):
                    investigation_call_limit = min(
                        investigation_call_limit,
                        max(
                            0,
                            max_calls_per_tick - WITHIN_TICK_COMMIT_CALL_RESERVE,
                        ),
                    )
                executed_calls = requested_calls[:investigation_call_limit]
                dropped_calls = requested_calls[investigation_call_limit:]
                bounded_investigation_action = Action(
                    tool_calls=executed_calls,
                    dominant=investigation_action.dominant,
                    assistant_text=investigation_action.assistant_text,
                    rationale=investigation_action.rationale,
                )
                query_obs, query_results = execute_investigation(
                    bounded_investigation_action
                )
                investigation_results = list(query_results)
                _raise_on_infrastructure_tool_failure(
                    investigation_results
                )
                investigation_trace_edges = _tool_trace_edges(
                    bounded_investigation_action,
                    investigation_results,
                    applied_tick=current_tick,
                    request_tick=current_tick,
                    visible_evidence_ids=agent_visible_evidence_ids,
                    evidence_logger=getattr(env, "evidence", None),
                )
                for result in investigation_results:
                    agent_visible_evidence_ids.update(
                        str(evidence_id)
                        for evidence_id in [
                            getattr(result, "evidence_id", None),
                            *(getattr(result, "produces_evidence_ids", None) or []),
                        ]
                        if evidence_id
                    )
                for result in investigation_results:
                    if result.ok:
                        tool_results_ok += 1
                    else:
                        tool_results_failed += 1
                # A read-only investigation may materialize a delayed
                # state-changing call before the commit stage. Keep the
                # runner's terminal-integrity ledger in sync with those
                # results just as we do for ``env.step`` results below.
                _update_pending_action_deadlines(
                    pending_action_deadlines,
                    investigation_results,
                )
                act_obs = dict(query_obs)
                act_obs["__decision_epoch__"] = {
                    "decision_id": f"decision-{model_decision_ticks + 1}",
                    "model_decision_index": model_decision_ticks + 1,
                    "reasons": prospective_decision_reasons,
                    "state_version": current_tick,
                    "simulator_tick": current_tick,
                }
                act_obs["__last_tool_results__"] = obs.get("__last_tool_results__", [])
                act_obs["__last_realized_events__"] = obs.get(
                    "__last_realized_events__", []
                )
                act_obs["__last_evidence_ids__"] = obs.get("__last_evidence_ids__", [])
                act_obs["__last_reward__"] = obs.get("__last_reward__", 0.0)
                act_obs["__within_tick_tool_results__"] = [
                    result.to_dict() for result in query_results
                ]
                act_obs["__within_tick_budget__"] = {
                    "requested_calls": len(requested_calls),
                    "executed_calls": len(executed_calls),
                    "dropped_calls": len(dropped_calls),
                    "dropped_call_ids": [
                        str(call.call_id)
                        for call in dropped_calls
                        if call.call_id not in (None, "")
                    ],
                    "investigation_call_limit": investigation_call_limit,
                    "commit_call_reserve": WITHIN_TICK_COMMIT_CALL_RESERVE,
                }
                act_obs["__interaction_stage__"] = "commit"
                within_tick_record = {
                    "tick": obs.get("tick", getattr(env, "tick", 0)),
                    "investigation_action": bounded_investigation_action.to_dict(),
                    "requested_investigation_tool_calls": [
                        call.name for call in requested_calls
                    ],
                    "investigation_tool_calls": [
                        call.name for call in executed_calls
                    ],
                    "investigation_call_ids": [
                        str(call.call_id)
                        for call in executed_calls
                        if call.call_id not in (None, "")
                    ],
                    "dropped_investigation_tool_calls": [
                        call.name for call in dropped_calls
                    ],
                    "dropped_investigation_call_ids": [
                        str(call.call_id)
                        for call in dropped_calls
                        if call.call_id not in (None, "")
                    ],
                    "tool_results": [result.to_dict() for result in query_results],
                    "tool_trace_edges": investigation_trace_edges,
                }
                within_tick_records.append(within_tick_record)
            commit_allowed_names = sorted(
                name
                for spec in tool_specs
                for name in [
                    str((spec.get("function") or {}).get("name") or "")
                ]
                if name
                and (
                    name not in readonly_names
                    or name in {"wait", "noop", "commit_to_plan"}
                )
            )
            if commit_allowed_names:
                act_obs = dict(act_obs)
                act_obs["__allowed_tool_names__"] = commit_allowed_names
                act_obs["__interaction_stage__"] = "commit"
        if (
            not pending_action_hold
            and not decision_budget_exhausted
            and not autonomous_hold
            and not native_idle_hold
        ):
            drafts = []
            model_decision_ticks += 1
            decision_reasons = prospective_decision_reasons
            if "scheduled_review" in decision_reasons:
                scheduled_review_ticks.append(current_tick)
            if "periodic_scan" in decision_reasons:
                periodic_scan_ticks.append(current_tick)
            if "provider_retry" in decision_reasons:
                provider_retry_tick = None
            if reference_scan_interval is not None and (
                next_baseline_scan_tick is None
                or "periodic_scan" in decision_reasons
            ):
                next_baseline_scan_tick = current_tick + reference_scan_interval
            decision_epochs.append(
                {
                    "tick": current_tick,
                    "model_decision_index": model_decision_ticks,
                    "reasons": decision_reasons,
                    "plan_id": autonomy_plan_id,
                }
            )
            act_obs = dict(act_obs)
            act_obs["__model_decision_budget__"] = model_decision_budget
            act_obs["__model_decisions_remaining__"] = (
                model_decision_budget - model_decision_ticks + 1
                if model_decision_budget is not None
                else None
            )
            if multi_turn:
                drafts = _collect_multi_turn_drafts(
                    agent=agent,
                    observation=obs,
                    tool_specs=tool_specs,
                    n_rounds=rounds_per_tick,
                )
                multi_turn_records.append(
                    {
                        "tick": obs.get("tick", getattr(env, "tick", 0)),
                        "rounds": drafts,
                    }
                )
                act_obs = dict(act_obs)
                act_obs["__multi_turn_drafts__"] = list(drafts)
            canonical_observation = (
                precomputed_provider_observation
                if precomputed_provider_observation is not None
                else _canonical_json_value(act_obs)
            )
            canonical_tool_specs = _canonical_json_value(tool_specs)
            decision_envelope = {
                "simulator_tick": current_tick,
                "model_decision_index": model_decision_ticks,
                "decision_reasons": decision_reasons,
                "active_plan_id": autonomy_plan_id,
                "pre_action_observation_sha256": _canonical_sha256(
                    canonical_observation
                ),
                "available_tool_schema_sha256": _canonical_sha256(
                    canonical_tool_specs
                ),
                "pre_action_observation": canonical_observation,
                "available_tool_schema": canonical_tool_specs,
                "requested_review_tick": requested_review_tick,
                "backend_review_deadline_tick": None,
                "next_periodic_scan_tick": next_baseline_scan_tick,
            }
            if precomputed_action is not None:
                action = precomputed_action
            elif callable(continue_decision_epoch) and investigation_results:
                action = continue_decision_epoch(act_obs, tool_specs)
            else:
                action = agent.act(act_obs, tool_specs)
            get_interaction_stats = getattr(
                agent, "get_interaction_stats", None
            )
            if callable(get_interaction_stats):
                interaction_stats = get_interaction_stats() or {}
                provider_records = list(
                    interaction_stats.get("provider_request_records") or []
                )
                decision_envelope["provider_requests"] = (
                    provider_records[provider_request_record_count:]
                )
                provider_request_record_count = len(provider_records)
                response_records = list(
                    interaction_stats.get("provider_response_records") or []
                )
                decision_envelope["provider_responses"] = (
                    response_records[provider_response_record_count:]
                )
                provider_response_record_count = len(response_records)
        get_provider_outcome = getattr(
            agent,
            "get_last_provider_outcome",
            None,
        )
        provider_outcome = (
            get_provider_outcome()
            if callable(get_provider_outcome)
            else {"status": "not_applicable"}
        )
        provider_status = str(
            (provider_outcome or {}).get("status") or "not_applicable"
        )
        if provider_status == "failed":
            provider_failure_count += 1
            provider_retry_tick = current_tick + 1
            autonomy_records.append(
                {
                    "tick": current_tick,
                    "kind": "provider_retry_scheduled",
                    "retry_tick": provider_retry_tick,
                }
            )
            if decision_envelope is not None:
                decision_envelope["provider_status"] = "failed"
        elif decision_envelope is not None:
            decision_envelope["provider_status"] = provider_status
        actions.append(action)
        reconcile_control_receipts = getattr(
            agent,
            "reconcile_control_receipts",
            None,
        )
        supports_control_reconciliation = getattr(
            env,
            "supports_control_reconciliation",
            None,
        )
        use_control_reconciliation = bool(
            callable(reconcile_control_receipts)
            and callable(supports_control_reconciliation)
            and supports_control_reconciliation()
        )
        if use_control_reconciliation:
            control_observation, initial_control_results = env.stage_control(action)
            # ToolProtocol assigns any missing call IDs at the stage boundary.
            # Capture the request only afterward so the agent can link a retry
            # to the exact receipt without inventing an identifier.
            initial_action = action.to_dict()
            initial_control_results = list(initial_control_results)
            _raise_on_infrastructure_tool_failure(initial_control_results)
            retryable_receipts = [
                result
                for result in initial_control_results
                if result.ok is False
                and result.error_code == "INJECTED_FAILURE"
                and result.state_changing
                and result.call_id
            ]
            retry_action: Action | None = None
            retry_results: list[Any] = []
            if retryable_receipts:
                receipt_observation = dict(act_obs)
                receipt_observation.update(control_observation)
                receipt_observation["__interaction_stage__"] = (
                    "control_reconciliation"
                )
                receipt_observation["__control_receipts__"] = [
                    result.to_dict() for result in initial_control_results
                ]
                receipt_observation["__control_calls__"] = [
                    dict(call) for call in initial_action["actions"]
                ]
                receipt_observation["__retryable_call_ids__"] = [
                    str(result.call_id) for result in retryable_receipts
                ]
                receipt_observation["__allowed_tool_names__"] = sorted(
                    {str(result.name) for result in retryable_receipts}
                )
                candidate_retry = reconcile_control_receipts(
                    receipt_observation,
                    tool_specs,
                )
                if not isinstance(candidate_retry, Action):
                    raise TypeError(
                        "agent reconcile_control_receipts must return Action"
                    )
                retry_action = candidate_retry
                _retry_observation, retry_results = env.stage_control(retry_action)
                retry_results = list(retry_results)
                _raise_on_infrastructure_tool_failure(retry_results)
                action.tool_calls.extend(retry_action.tool_calls)
            ret = env.advance_staged_control()
            control_reconciliation_record = {
                "tick": current_tick,
                "initial_action": initial_action,
                "initial_receipts": [
                    result.to_dict() for result in initial_control_results
                ],
                "retryable_call_ids": [
                    str(result.call_id) for result in retryable_receipts
                ],
                "retry_action": retry_action.to_dict() if retry_action else None,
                "retry_receipts": [result.to_dict() for result in retry_results],
            }
            control_reconciliation_records.append(
                control_reconciliation_record
            )
        else:
            ret = env.step(action)
        _raise_on_infrastructure_tool_failure(list(ret.tool_results))
        episode_reward_total += float(ret.reward)
        applied_tick = int(ret.observation.get("tick", env.tick - 1))
        # ToolProtocol assigns missing call IDs during ``env.step``.  Register
        # calls only after that boundary so delayed materializations retain
        # their original request identity and tick.
        for call in action.tool_calls:
            if call.call_id is not None:
                known_tool_calls[str(call.call_id)] = call
                known_tool_call_ticks[str(call.call_id)] = current_tick
        tool_trace_edges = _tool_trace_edges(
            action,
            list(ret.tool_results),
            realized_events=list(ret.info.realized_events or []),
            applied_tick=applied_tick,
            request_tick=current_tick,
            known_calls=known_tool_calls,
            known_call_ticks=known_tool_call_ticks,
            visible_evidence_ids=agent_visible_evidence_ids,
            evidence_logger=getattr(env, "evidence", None),
        )
        for tr in ret.tool_results:
            if tr.ok:
                tool_results_ok += 1
            else:
                tool_results_failed += 1
        _update_pending_action_deadlines(
            pending_action_deadlines,
            list(ret.tool_results),
        )
        info_payload = ret.info.to_dict()
        info_extra = dict(info_payload.get("extra") or {})
        info_extra["tool_trace_edges"] = tool_trace_edges
        info_payload["extra"] = info_extra
        if decision_envelope is not None:
            decision_envelope["action_tick"] = current_tick
            decision_envelope["applied_tick"] = int(
                applied_tick
            )
            decision_envelope["observation_tick"] = int(
                current_tick
            )
            decision_envelope["post_observation_tick"] = int(
                applied_tick
            )
            decision_envelope["next_state_sha256"] = _canonical_sha256(
                ret.observation
            )
            info_payload["decision_envelope"] = decision_envelope
        canonical_events = canonicalize_runtime_events(
            list(ret.info.realized_events or []),
            # Exogenous events are applied during the transition selected at
            # ``current_tick``. Agent-caused records retain their explicit
            # later ``outcome_tick`` for delayed native effects.
            applied_tick=current_tick,
        )
        world_evolution_records.extend(canonical_events)
        extra = dict(info_payload.get("extra") or {})
        extra["world_evolution_records"] = canonical_events
        step_event_contract_violations: list[dict[str, Any]] = []
        for event_index, event in enumerate(ret.info.realized_events or []):
            if not isinstance(event, dict):
                continue
            audit_row = audit_event_decision_contract(
                event,
                event_index=event_index,
            )
            if audit_row is None:
                continue
            audit_row["applied_tick"] = current_tick
            step_event_contract_violations.append(audit_row)
        if step_event_contract_violations:
            event_contract_violations.extend(step_event_contract_violations)
            extra["event_contract_violations"] = (
                step_event_contract_violations
            )
        info_payload["extra"] = extra
        terminal_response_window_reasons_seen.extend(
            _terminal_response_window_reasons(
                list(ret.info.realized_events or [])
            )
        )
        if within_tick_record is not None:
            extra = dict(info_payload.get("extra") or {})
            extra["within_tick_investigation"] = within_tick_record
            info_payload["extra"] = extra
        if control_reconciliation_record is not None:
            extra = dict(info_payload.get("extra") or {})
            extra["control_reconciliation"] = control_reconciliation_record
            info_payload["extra"] = extra
        analysis_steps.append(
            {
                # Analysis is decision-centric: this is the tick whose
                # observation the agent consumed and on which it chose the
                # action.  The resulting post-step observation has its own
                # explicit tick so evidence consumers cannot conflate the two.
                "tick": current_tick,
                "action_tick": current_tick,
                "applied_tick": applied_tick,
                "post_observation_tick": applied_tick,
                "action": action.to_dict(),
                "tool_results": [result.to_dict() for result in ret.tool_results],
                "tool_trace_edges": tool_trace_edges,
                "info": info_payload,
            }
        )
        if logger is not None:
            if drafts:
                extra = dict(info_payload.get("extra") or {})
                extra["multi_turn_drafts"] = list(drafts)
                info_payload["extra"] = extra
            logger.log_step(
                tick=applied_tick,
                observation=ret.observation,
                action=action.to_dict(),
                reward=ret.reward,
                tool_results=[
                    r.to_dict() for r in investigation_results + list(ret.tool_results)
                ],
                evidence_ids=ret.info.evidence_ids,
                info=info_payload,
                assistant_text=action.assistant_text,
            )
        obs = ret.observation
        # Surface structured feedback for the NEXT agent.act() call.
        obs["__last_tool_results__"] = [r.to_dict() for r in ret.tool_results]
        # v0.3.1 P1 fix: respect fog-of-war on the agent-facing event stream.
        # A hidden perturbation (hidden generator/line outage) must NOT be
        # announced to the agent through this passive channel — that defeats
        # the paid ``investigate_substation`` mechanic. The SCORER's
        # realized-event stream is built separately from the evidence log
        # (see ``items_by_kind("realized_event")`` below), so it still sees
        # the full ground-truth set; only the agent observation is filtered.
        obs["__last_realized_events__"] = [
            ev for ev in (ret.info.realized_events or []) if not ev.get("hidden")
        ]
        tool_result_evidence_ids = {
            str(evidence_id)
            for result in ret.tool_results
            for evidence_id in [
                getattr(result, "evidence_id", None),
                *(getattr(result, "produces_evidence_ids", None) or []),
            ]
            if evidence_id
        }
        visible_step_evidence_ids = _agent_visible_step_evidence_ids(
            realized_events=list(ret.info.realized_events or []),
            step_evidence_ids=list(ret.info.evidence_ids or []),
            tool_results=list(ret.tool_results),
        )
        agent_visible_evidence_ids.update(visible_step_evidence_ids)
        agent_visible_evidence_ids.update(tool_result_evidence_ids)
        obs["__last_evidence_ids__"] = visible_step_evidence_ids
        obs["__last_reward__"] = float(ret.reward)
        obs["__last_early_stop_warnings__"] = list(
            ret.info.early_stop_warnings or []
        )
        obs["__last_forecast_updates__"] = dict(ret.info.forecast_updates or {})
        obs["__model_decision_budget__"] = model_decision_budget
        obs["__model_decisions_remaining__"] = (
            max(0, model_decision_budget - model_decision_ticks)
            if model_decision_budget is not None
            else None
        )
        observe_transition = getattr(agent, "observe_transition", None)
        if callable(observe_transition):
            transition_ingestion_attempted += 1
            try:
                observe_transition(obs)
                transition_ingestion_completed += 1
            except Exception as exc:
                failure = {
                    "applied_tick": applied_tick,
                    "error_type": type(exc).__name__,
                    "error_message": redact_provider_error(exc),
                    "violation_codes": ["agent_transition_ingest_failed"],
                }
                transition_ingestion_failures.append(failure)
                event_contract_violations.append(failure)
                LOGGER.warning(
                    "agent.observe_transition raised %s: %s; recorded as an "
                    "event-contract violation",
                    type(exc).__name__,
                    redact_provider_error(exc),
                )
        (
            hold_ticks,
            plan_id,
            plan_call_id,
            plan_args,
        ) = _materialized_autonomy_window(
            action,
            list(ret.tool_results),
            pending_plan_requests,
        )
        if plan_args and plan_args.get("review_after_ticks") is None:
            autonomy_ticks_remaining = 0
            autonomy_plan_id = plan_id
            autonomy_plan_active = False
            requested_review_tick = None
            active_plan_expires_at_tick = None
            autonomy_records.append(
                {
                    "tick": ret.observation.get("tick", env.tick - 1),
                    "kind": "plan_committed_without_scheduled_review",
                    "plan_id": plan_id,
                    "call_id": plan_call_id,
                }
            )
        elif plan_args:
            if "wake_if" in plan_args:
                active_wake_if = {
                    str(value)
                    for value in plan_args.get("wake_if") or []
                    if str(value) in OPTIONAL_PLAN_WAKE_REASONS
                }
            else:
                active_wake_if = set(OPTIONAL_PLAN_WAKE_REASONS)
            active_plan_expires_at_tick = _positive_int(
                plan_args.get("plan_expires_at_tick")
            )
            review_interval = hold_ticks + 1
            autonomy_ticks_remaining = hold_ticks
            autonomy_plan_id = plan_id
            autonomy_plan_active = True
            requested_review_tick = current_tick + review_interval
            autonomy_records.append(
                {
                    "tick": ret.observation.get("tick", env.tick - 1),
                    "kind": "autonomy_window_opened",
                    "plan_id": plan_id,
                    "call_id": plan_call_id,
                    "requested_hold_ticks": hold_ticks,
                    "requested_review_tick": current_tick + review_interval,
                    "backend_review_deadline_tick": None,
                    "schedule_status": "accepted",
                }
            )
        if ret.done:
            response_reasons = _autonomy_interrupt_reasons(obs)
            response_reasons.extend(terminal_response_window_reasons_seen)
            if provider_retry_tick is not None:
                response_reasons.append("provider_retry")
            response_reasons = sorted(set(response_reasons))
            extension_reasons = [
                reason
                for reason in response_reasons
                if reason not in terminal_response_window_extended_reasons
            ]
            terminal_response_window_extended_reasons.update(
                reason
                for reason in response_reasons
                if reason == "terminal_response_window_missing"
            )
            pending_drain_due = (
                max(pending_action_deadlines.values())
                if pending_action_deadlines
                else None
            )
            pending_drain_available = bool(
                pending_drain_due is not None
                and int(env.tick) <= pending_drain_due
                and pending_drain_due < int(env.horizon)
            )
            if (
                extension_reasons or pending_drain_available
            ) and int(env.tick) < int(env.horizon):
                response_window_extensions += 1
                autonomy_records.append(
                    {
                        "tick": int(env.tick),
                        "kind": "terminal_response_window_extended",
                        "reasons": [
                            *response_reasons,
                            *(
                                ["pending_action_drain"]
                                if pending_drain_available
                                else []
                            ),
                        ],
                    }
                )
                continue
            break

    # Whether the backend returned ``done`` or the configured horizon was
    # exhausted, the final observation has no subsequent agent response
    # window. Preserve any newly visible interrupt as a release blocker.
    terminal_unanswered_interrupt_reasons = _autonomy_interrupt_reasons(
        obs,
        include_tool_feedback=False,
    )
    terminal_unanswered_interrupt_reasons.extend(
        terminal_response_window_reasons_seen
    )
    if provider_retry_tick is not None:
        terminal_unanswered_interrupt_reasons.append("provider_retry")
    terminal_unanswered_interrupt_reasons = sorted(
        set(terminal_unanswered_interrupt_reasons)
    )
    terminal_feedback_reasons = _tool_feedback_interrupt_reasons(obs)

    # v0.2.1: per-episode reflection hook for Reflexion-style agents.
    # v0.2.2 F-03: pass the accumulated episode return (not the last tick).
    # Baselines without this method are unaffected.
    if hasattr(agent, "on_episode_end"):
        try:
            agent.on_episode_end(
                final_observation=obs,
                actions=actions,
                episode_reward=episode_reward_total,
            )
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "agent.on_episode_end raised %s: %s — ignoring",
                type(exc).__name__,
                redact_provider_error(exc),
            )

    return {
        "actions": actions,
        "final_observation": obs,
        "tool_results_ok": tool_results_ok,
        "tool_results_failed": tool_results_failed,
        "episode_reward_total": episode_reward_total,
        "stale_observation_records": stale_observation_records,
        "analysis_steps": analysis_steps,
        "multi_turn": {
            "enabled": bool(multi_turn),
            "rounds_per_tick": rounds_per_tick if multi_turn else 1,
        },
        "multi_turn_records": multi_turn_records,
        "within_tick_interaction": {
            "enabled": bool(within_tick_interaction),
            "investigation_rounds": 1 if within_tick_interaction else 0,
        },
        "within_tick_records": within_tick_records,
        "control_reconciliation_records": control_reconciliation_records,
        "world_evolution_records": world_evolution_records,
        "transition_ingestion": {
            "status": (
                "failed"
                if transition_ingestion_failures
                else "complete"
                if transition_ingestion_attempted
                else "not_applicable"
            ),
            "attempted": transition_ingestion_attempted,
            "completed": transition_ingestion_completed,
            "failed": len(transition_ingestion_failures),
            "failures": transition_ingestion_failures,
        },
        "event_contract": {
            "schema_version": EVENT_DECISION_CONTRACT_VERSION,
            "violation_count": len(event_contract_violations),
            "violations": event_contract_violations,
        },
        "terminal_integrity": {
            "release_ready": not (
                pending_action_deadlines
                or terminal_unanswered_interrupt_reasons
            ),
            "unresolved_pending_actions": dict(
                sorted(pending_action_deadlines.items())
            ),
            "unanswered_interrupt_reasons": (
                terminal_unanswered_interrupt_reasons
            ),
            "terminal_feedback_reasons": terminal_feedback_reasons,
            "response_window_extensions": response_window_extensions,
        },
        "event_adaptive_autonomy": {
            "enabled": True,
            "cadence_contract_declared": cadence_contract_declared,
            "model_decision_ticks": model_decision_ticks,
            "autonomous_hold_ticks": autonomous_hold_ticks,
            "model_decision_budget": model_decision_budget,
            "decision_budget_exhausted": decision_budget_exhaustion_recorded,
            "decision_budget_fully_consumed": bool(
                model_decision_budget is not None
                and model_decision_ticks >= model_decision_budget
            ),
            "decision_budget_hold_ticks": decision_budget_hold_ticks,
            "decision_budget_interrupt_overruns": (
                decision_budget_interrupt_overruns
            ),
            "pending_action_hold_ticks": pending_action_hold_ticks,
            "native_idle_hold_ticks": native_idle_hold_ticks,
            "decision_opportunities": decision_opportunities,
            "decision_opportunity_ticks": decision_opportunity_ticks,
            "scheduled_review_ticks": scheduled_review_ticks,
            "periodic_scan_ticks": periodic_scan_ticks,
            "decision_epochs": decision_epochs,
            "provider_failure_count": provider_failure_count,
            "missed_actionable_opportunities": missed_actionable_opportunities,
            "delegated_plan_opportunities": delegated_plan_opportunities,
            "delegated_plan_opportunity_ticks": delegated_plan_opportunity_ticks,
            "deferred_pending_opportunities": deferred_pending_opportunities,
            "hold_while_actions_pending": hold_while_pending,
            "records": autonomy_records,
        },
    }


def _collect_multi_turn_drafts(
    *,
    agent: Any,
    observation: dict[str, Any],
    tool_specs: list[dict[str, Any]],
    n_rounds: int,
) -> list[dict[str, Any]]:
    drafts: list[dict[str, Any]] = []
    deliberate = getattr(agent, "deliberate", None)
    for round_index in range(1, n_rounds + 1):
        draft_obs = dict(observation)
        draft_obs["__multi_turn_drafts__"] = list(drafts)
        if callable(deliberate):
            draft_action = deliberate(
                draft_obs,
                tool_specs,
                round_index=round_index,
                n_rounds=n_rounds,
                previous_drafts=list(drafts),
            )
        else:
            draft_action = Action(
                tool_calls=[],
                dominant="deliberate",
                assistant_text=(
                    f"draft round {round_index}/{n_rounds}; no deliberate hook"
                ),
            )
        drafts.append(_multi_turn_draft_to_dict(draft_action, round_index, n_rounds))
    return drafts


def _multi_turn_draft_to_dict(
    action: Action, round_index: int, n_rounds: int
) -> dict[str, Any]:
    assistant_text = action.assistant_text or ""
    return {
        "round_index": int(round_index),
        "n_rounds": int(n_rounds),
        "tool_calls": [call.to_dict() for call in action.tool_calls],
        "dominant_action": action.dominant,
        "assistant_text": assistant_text,
        "reflection_summary": assistant_text[:240],
    }


def _record_stale_observations(
    *,
    evidence: Any,
    observation: dict[str, Any],
    tick: Any,
) -> list[dict[str, Any]]:
    entities = observation.get("entities") if isinstance(observation, dict) else None
    if not isinstance(entities, dict):
        return []
    records: list[dict[str, Any]] = []
    try:
        tick_int = int(tick)
    except (TypeError, ValueError):
        tick_int = 0
    for entity_id, entity in sorted(entities.items()):
        if not isinstance(entity, dict):
            continue
        stale_attrs = entity.get("_stale_attrs")
        if not isinstance(stale_attrs, dict):
            continue
        for attr, staleness_ticks in sorted(stale_attrs.items()):
            record = {
                "tick": tick_int,
                "entity_id": str(entity_id),
                "attr": str(attr),
                "staleness_ticks": staleness_ticks,
            }
            if evidence is not None:
                record["evidence_id"] = evidence.log(
                    kind="stale_observation",
                    tick=tick_int,
                    payload=record,
                    source="engine",
                )
            records.append(record)
    return records


def run_one(
    scenario: dict[str, Any],
    agent_name: str,
    *,
    agent_kwargs: dict[str, Any] | None = None,
    trajectory_dir: Path | None = None,
    counterfactual_masking: str = "wait_only",
    seed_override: int | None = None,
    multi_turn: bool = False,
    multi_turn_rounds: int = 3,
    per_action_attribution: bool = False,
    per_action_cap: int | None = 20,
    per_action_group_attribution: bool = False,
    per_action_group_cap: int | None = 20,
    within_tick_interaction: bool = True,
) -> dict[str, Any]:
    """Run a single episode and return the result blob (no disk write here)."""
    seed = int(seed_override if seed_override is not None else scenario.get("seed", 42))

    # T0: dispatch the environment / oracle / equity map by scenario domain.
    # A missing ``domain`` defaults to power_grid (v0.1–v0.6 scenarios).
    spec = get_domain_spec(scenario.get("domain"))
    env = spec.env_factory()()
    try:
        return _run_one_with_environment(
            scenario,
            agent_name,
            env=env,
            spec=spec,
            seed=seed,
            agent_kwargs=agent_kwargs,
            trajectory_dir=trajectory_dir,
            counterfactual_masking=counterfactual_masking,
            multi_turn=multi_turn,
            multi_turn_rounds=multi_turn_rounds,
            per_action_attribution=per_action_attribution,
            per_action_cap=per_action_cap,
            per_action_group_attribution=per_action_group_attribution,
            per_action_group_cap=per_action_group_cap,
            within_tick_interaction=within_tick_interaction,
        )
    finally:
        env.close()


def _snapshot_and_close_completed_environment(
    env: Any,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[Any], list[dict[str, Any]]]:
    """Capture completed runtime state before starting fresh replay environments.

    Native runtimes such as libsumo expose one process-global connection.  The
    completed live environment must release that connection before the
    counterfactual probe starts another environment in the same process.
    """
    try:
        ground_truth = env.ground_truth()
        foresight = (
            evaluate_foresight(env.evidence).to_dict() if env.evidence else None
        )
        backend_records = build_backend_records(env)
        realized = [
            {**event.payload, "tick": event.tick}
            for event in (
                env.evidence.items_by_kind("realized_event") if env.evidence else []
            )
        ]
        return ground_truth, foresight, backend_records, realized
    finally:
        env.close()


def _run_one_with_environment(
    scenario: dict[str, Any],
    agent_name: str,
    *,
    env: Any,
    spec: DomainSpec,
    seed: int,
    agent_kwargs: dict[str, Any] | None,
    trajectory_dir: Path | None,
    counterfactual_masking: str,
    multi_turn: bool,
    multi_turn_rounds: int,
    per_action_attribution: bool,
    per_action_cap: int | None,
    per_action_group_attribution: bool,
    per_action_group_cap: int | None,
    within_tick_interaction: bool,
) -> dict[str, Any]:
    env.reset(scenario, seed=seed)
    agent = make_agent(agent_name, **(agent_kwargs or {}))
    agent.reset(env, scenario, seed=seed)

    # v0.2.2 (P1-3): capture Reflexion's lessons-file fingerprint *after*
    # `agent.reset(...)` (which loaded the lessons) but *before* the
    # episode runs (which may append a new lesson via on_episode_end).
    # This pins down WHICH past lessons influenced this episode.
    agent_extras: dict[str, Any] | None = None
    if hasattr(agent, "lessons_fingerprint"):
        try:
            agent_extras = {
                "reflexion_lessons_state": agent.lessons_fingerprint(),
            }
        except Exception:  # noqa: BLE001 — fingerprint is best-effort
            agent_extras = None

    logger: TrajectoryLogger | None = None
    if trajectory_dir is not None:
        logger = TrajectoryLogger(
            episode_id=f"{agent_name}_{scenario.get('seed_id', 'anon')}_s{seed}",
            output_dir=trajectory_dir,
        )
        from datetime import datetime

        logger.set_header(
            EpisodeHeader(
                episode_id=logger.episode_id,
                scenario_id=str(scenario.get("seed_id", "anon")),
                scenario_signature=recompute_signature_with_seed(scenario, seed, spec),
                domain=str(scenario.get("domain", "power_grid")),
                family=str(scenario.get("family", "")),
                difficulty_mode=str(scenario.get("difficulty_mode", "time_pressure")),
                difficulty_level=str(scenario.get("difficulty_level", "basic")),
                backend_kind=str(scenario.get("backend_kind", "")),
                horizon_ticks=int(scenario.get("horizon_ticks", 0)),
                tick_minutes=(
                    int(scenario["tick_minutes"])
                    if scenario.get("tick_minutes") is not None
                    else None
                ),
                agent_name=agent_name,
                agent_config=_public_agent_config(agent_kwargs),
                seed=seed,
                start_time_utc=datetime.now(UTC).isoformat(),
                agent_extras=agent_extras,
                tick_seconds=(
                    float(scenario["tick_seconds"])
                    if scenario.get("tick_seconds") is not None
                    else None
                ),
                clock_contract=(
                    dict(scenario["clock_contract"])
                    if isinstance(scenario.get("clock_contract"), dict)
                    else None
                ),
            )
        )

    try:
        capability = get_backend_capability(str(scenario.get("backend_kind") or ""))
        baseline_intervals = [
            interval
            for interval in (
                _positive_int(capability.periodic_scan_every_ticks),
                _positive_int(capability.max_review_after_ticks),
            )
            if interval is not None
        ]
        loop_result = _run_episode_loop(
            env=env,
            agent=agent,
            logger=logger,
            multi_turn=multi_turn,
            multi_turn_rounds=multi_turn_rounds,
            within_tick_interaction=within_tick_interaction,
            baseline_scan_interval=(
                min(baseline_intervals) if baseline_intervals else None
            ),
        )
    except Exception as exc:
        try:
            details = _finalize_failed_episode_audit(
                agent=agent,
                logger=logger,
                error=exc,
                error_stage="interaction_loop",
            )
        except Exception:
            LOGGER.exception("failed to persist aborted episode audit")
            details = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_stage": "interaction_loop",
                "audit_persistence": "failed",
            }
        exc.episode_error_details = details  # type: ignore[attr-defined]
        raise
    actions = loop_result["actions"]
    tool_results_ok = loop_result["tool_results_ok"]
    tool_results_failed = loop_result["tool_results_failed"]
    stale_observation_records = loop_result["stale_observation_records"]

    gt, foresight, backend_records, realized = (
        _snapshot_and_close_completed_environment(env)
    )

    cf = domain_counterfactual_report(
        env_factory=spec.env_factory(),
        scenario_config=scenario,
        seed=seed,
        actual_actions=actions,
        masking_policy=counterfactual_masking,
        per_action=per_action_attribution,
        per_action_cap=per_action_cap,
        per_action_groups=per_action_group_attribution,
        per_action_group_cap=per_action_group_cap,
    )

    # T0: ``load_assignments`` is the cross-domain stakeholder-class field
    # name (kept identical across all 5 domains, see each seeds/schema.py),
    # so this extraction stays domain-agnostic.
    load_classes = {
        la["load_id"]: la["stakeholder_class"]
        for la in scenario.get("load_assignments", [])
    }
    load_criticalities = {
        la["load_id"]: float(la["criticality"])
        for la in scenario.get("load_assignments", [])
        if la.get("criticality") is not None
    }

    # T0: optimality-gap reference (``ScoringInputs.lp_optimum``) is
    # dispatched by domain. power_grid keeps the in-runner LP / AC-OPF
    # oracle; the v0.7 domains read the oracle envelope cached on
    # ``seed.backend_config['reference_optimum']`` during ``env.reset``.
    lp_optimum: float | None
    if spec.uses_runner_lp_oracle:
        # v0.2: compute LP economic-dispatch optimum for pglib-uc scenarios.
        # Storm (Grid2Op) and CIGRE distribution scenarios skip this since
        # the LP model doesn't cover their physics. Cached per case-path.
        lp_optimum = _maybe_lp_optimum(scenario)
        # v0.4 (P0-1): the pandapower AC-OPF backend exposes a TRUE per-tick
        # AC-OPF reference optimum (redispatch pins released), computed during
        # the episode. Wire it into optimality_gap so the AC-OPF family scores
        # an *operational* dispatch-efficiency gap (agent's realized cost vs
        # the best cost achievable given the grid state it faced). Without
        # this, optimality_gap was applicable=False for all AC-OPF scenarios
        # despite the manifest descriptor claiming it was scored.
        if (
            lp_optimum is None
            and str(scenario.get("backend_kind", "")) == "pandapower_acopf"
        ):
            backend = getattr(env, "_backend", None)
            ref_fn = getattr(backend, "acopf_reference_optimum", None)
            if callable(ref_fn):
                try:
                    ref_val = float(ref_fn())
                    if ref_val > 0:
                        lp_optimum = ref_val
                except Exception:
                    lp_optimum = None
        optimality_objective_component = spec.objective_cost_component
    else:
        lp_optimum = reference_optimum_from_backend_config(env)
        optimality_objective_component = reference_optimum_objective_component(
            env,
            default=spec.objective_cost_component,
        )
    # v0.2.1: log evidence so audit can verify scoring traceability.
    if env.evidence is not None:
        env.evidence.log(
            kind="counterfactual_result",
            tick=env.tick,
            payload={
                "actual_cost": float(cf.actual_cost),
                "counterfactual_cost": float(cf.counterfactual_cost),
                "prevented_loss": float(cf.prevented_loss),
                "applicable": bool(cf.applicable),
                "reason_code": cf.reason_code,
                "masking_policy": cf.masking_policy,
                "per_action_status": cf.per_action_status,
                "per_action": cf.per_action,
                "per_action_group_status": cf.per_action_group_status,
                "per_action_groups": cf.per_action_groups,
            },
            source="engine",
        )
        if lp_optimum is not None:
            objective_component = optimality_objective_component
            env.evidence.log(
                kind="lp_oracle",
                tick=env.tick,
                payload={
                    "lp_optimum_cost": float(lp_optimum),
                    "objective_component": objective_component,
                    "actual_objective_cost": (
                        float(gt["cost_components"][objective_component])
                        if objective_component
                        and objective_component in gt["cost_components"]
                        else None
                    ),
                },
                source="engine",
            )
        foresight = evaluate_foresight(env.evidence).to_dict()

    (
        event_response_records,
        valid_evidence_ids,
        operational_agency_profile,
    ) = _operational_agency_artifacts(
        env=env,
        ground_truth=gt,
        analysis_steps=loop_result["analysis_steps"],
        counterfactual=cf,
    )
    backend_config = scenario.get("backend_config")
    dimension_applicability = (
        backend_config.get("dimension_applicability")
        if isinstance(backend_config, dict)
        else None
    ) or scenario.get("dimension_applicability") or {}
    if not isinstance(dimension_applicability, dict):
        dimension_applicability = {}
    applicability_evidence_ids: list[str] = []
    if dimension_applicability and env.evidence is not None:
        applicability_evidence_ids.append(
            env.evidence.log(
                kind="dimension_applicability_contract",
                tick=env.tick,
                payload={
                    "dimensions": _canonical_json_value(
                        dimension_applicability
                    ),
                },
                source="engine",
            )
        )

    inputs = ScoringInputs(
        backend_tick_records=backend_records,
        realized_events=realized,
        cost_components=gt["cost_components"],
        # T0: per-entity unmet/shed/delay map under the domain's canonical
        # ground_truth key (per_load_shed_mwh / per_customer_unmet_units /
        # per_corridor_delay_minutes / per_zone_unserved_minutes).
        per_load_shed_mwh=gt.get(spec.equity_shed_key, {}),
        load_classes=load_classes,
        load_criticalities=load_criticalities,
        evidence_logger=env.evidence,
        stakeholder_mgr=env.stakeholders,
        dilemma_mgr=env.dilemmas,
        chose_fatal_option=gt.get("chose_fatal_option", False),
        counterfactual_report=cf.to_dict(),
        foresight_summary=foresight,
        lp_optimum=lp_optimum,
        optimality_objective_component=optimality_objective_component,
        difficulty_level=str(scenario.get("difficulty_level", "basic")),
        scenario_signature=recompute_signature_with_seed(scenario, seed, spec),
        stale_observation_records=stale_observation_records,
        adaptive_recovery_signal_key=spec.adaptive_recovery_signal_key,
        adaptive_recovery_signal_name=spec.adaptive_recovery_signal_name,
        causal_adaptation=(
            operational_agency_profile.get("dimensions", {}).get(
                "surprise_adaptation"
            )
        ),
        dimension_applicability=dimension_applicability,
        dimension_applicability_evidence_ids=applicability_evidence_ids,
        proven_tool_effect_evidence_by_call_id=(
            _proven_tool_effect_evidence_by_call_id(
                loop_result["analysis_steps"],
                evidence_logger=env.evidence,
            )
        ),
    )
    score = score_episode(inputs)

    evidence_path = None
    if logger is not None and env.evidence is not None:
        evidence_path = logger.write_evidence(env.evidence.to_jsonable())

    llm_stats = (
        agent.get_interaction_stats()
        if hasattr(agent, "get_interaction_stats")
        else None
    )
    trajectory_summary = _summarize_trajectory(actions, llm_stats)
    trajectory_summary["complexity"] = analyze_trajectory_steps(
        loop_result["analysis_steps"],
        per_action_attribution=(
            cf.per_action
            if per_action_attribution
            and cf.per_action_status in {"complete", "capped"}
            else None
        ),
        llm_stats=llm_stats,
    )
    trajectory_summary["tool_results_ok"] = tool_results_ok
    trajectory_summary["tool_results_failed"] = tool_results_failed
    trajectory_summary["multi_turn"] = loop_result["multi_turn"]
    trajectory_summary["within_tick_interaction"] = loop_result[
        "within_tick_interaction"
    ]
    trajectory_summary["event_adaptive_autonomy"] = loop_result[
        "event_adaptive_autonomy"
    ]
    trajectory_summary["terminal_integrity"] = loop_result[
        "terminal_integrity"
    ]
    trajectory_summary["event_contract"] = loop_result["event_contract"]
    trajectory_summary["transition_ingestion"] = loop_result["transition_ingestion"]
    if loop_result["within_tick_records"]:
        trajectory_summary["within_tick_records"] = loop_result[
            "within_tick_records"
        ]
        investigation_hist = Counter(
            tool_name
            for record in loop_result["within_tick_records"]
            for tool_name in record["investigation_tool_calls"]
        )
        trajectory_summary["n_tool_calls"] += sum(investigation_hist.values())
        combined_hist = Counter(trajectory_summary.get("tool_histogram") or {})
        combined_hist.update(investigation_hist)
        trajectory_summary["tool_histogram"] = dict(combined_hist)
    if loop_result["multi_turn_records"]:
        trajectory_summary["multi_turn_records"] = loop_result["multi_turn_records"]
    trajectory_summary["event_response_records"] = event_response_records
    trajectory_summary["operational_agency_valid_evidence_ids"] = sorted(
        valid_evidence_ids
    )
    trajectory_summary["operational_agency_profile"] = (
        operational_agency_profile
    )
    tool_registry = getattr(env, "_tools", None)
    if tool_registry is not None:
        tool_histogram = trajectory_summary.get("tool_histogram") or {}
        tool_surface_contract = _tool_surface_contract(env, scenario)
        trajectory_summary["tool_surface_contract"] = tool_surface_contract
        trajectory_summary["tool_semantic_coverage"] = (
            tool_registry.validate_semantic_coverage(
                tool_surface_contract.get("exposed_tool_names") or []
            )
        )
        trajectory_summary["tool_semantic_histogram"] = (
            classify_tool_semantic_histogram(
                tool_histogram,
                registry=tool_registry,
            )
        )
    (
        structured_memory,
        semantic_ledger_artifact,
        provider_audit_artifact,
    ) = _collect_agent_session_artifacts(agent=agent, logger=logger)
    if structured_memory is not None:
        trajectory_summary["structured_memory"] = structured_memory
    if semantic_ledger_artifact is not None:
        trajectory_summary["semantic_ledger_artifact"] = (
            semantic_ledger_artifact
        )
    if provider_audit_artifact is not None:
        trajectory_summary["provider_audit_artifact"] = (
            provider_audit_artifact
        )
    if logger is not None:
        trajectory_summary["trajectory_path"] = str(trajectory_dir / logger.episode_id)
        if evidence_path is not None:
            trajectory_summary["evidence_path"] = str(evidence_path)

    decision_impact = summarize_decision_impact(
        trajectory_summary.get("tool_histogram") or {},
        cf.to_dict(),
        tool_results_ok=tool_results_ok,
        tool_results_failed=tool_results_failed,
    )
    task_counterfactual = cf.to_dict()
    task_counterfactual["_counterfactual_task_tick_records"] = (
        cf.counterfactual_ground_truth.get("_task_tick_records") or []
    )
    task_completion = separate_task_outcome_and_process(
        evaluate_task_completion(
            scenario=scenario,
            ground_truth=gt,
            counterfactual=task_counterfactual,
            score=score.to_dict(),
        ),
        scenario=scenario,
    )
    if logger is not None:
        logger.finalize(
            final_score=score.total_score,
            trajectory_summary=trajectory_summary,
        )

    ground_truth_summary = {
        "cost_components": gt["cost_components"],
        "stakeholder_equity_gini": gt.get("stakeholder_equity_gini"),
        "chose_fatal_option": gt.get("chose_fatal_option"),
        "dilemmas_triggered": gt.get("dilemmas_triggered", []),
        # Audit-only: completed runs must expose realized backend events
        # so source-consumption calibration cannot infer runtime behavior
        # from YAML/provenance declarations.  This is emitted after the
        # interaction loop and is never inserted into an agent prompt.
        "realized_events": _canonical_json_value(realized),
    }
    if str(scenario.get("domain") or "") == "autonomous_driving":
        task_records = [
            dict(record)
            for record in gt.get("_task_tick_records") or []
            if isinstance(record, dict)
        ]
        backend_config = scenario.get("backend_config") or {}
        ground_truth_summary.update(
            {
                "candidate_id": str(backend_config.get("candidate_id") or ""),
                "source_window_sha256": str(
                    scenario.get("source_window_sha256") or ""
                ),
                "source_event_chain_sha256": str(
                    (scenario.get("provenance") or {}).get("source_event_chain_sha256")
                    or ""
                ),
                "collision_count": max(
                    (int(record.get("collision_count") or 0) for record in task_records),
                    default=0,
                ),
                "road_departure_count": max(
                    (int(record.get("road_departure_count") or 0) for record in task_records),
                    default=0,
                ),
                "runtime_assurance": _canonical_json_value(
                    gt.get("runtime_assurance") or {}
                ),
                "tactical_action_trace": _canonical_json_value(
                    gt.get("tactical_action_trace") or []
                ),
                "investigation_trace": _canonical_json_value(
                    gt.get("investigation_trace") or []
                ),
            }
        )

    result = {
        "scenario_id": scenario.get("seed_id"),
        "scenario_signature": recompute_signature_with_seed(scenario, seed, spec),
        "family": scenario.get("family"),
        "difficulty_mode": scenario.get("difficulty_mode"),
        "difficulty_level": scenario.get("difficulty_level"),
        "agent_name": agent_name,
        "agent_config": _public_agent_config(agent_kwargs),
        "seed": seed,
        "n_ticks_ran": len(actions),
        "score": score.to_dict(),
        "counterfactual": cf.to_dict(),
        "foresight": foresight,
        "ground_truth_summary": ground_truth_summary,
        "trajectory_summary": trajectory_summary,
        "structured_memory": structured_memory,
        "decision_impact": decision_impact,
        "task_completion": task_completion,
        "evaluation_protocol": {
            "version": EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
            "within_tick_interaction": bool(within_tick_interaction),
            "event_adaptive_autonomy": True,
            "task_completion_schema": task_completion["schema_version"],
            "construct_contract": scenario.get("construct_contract"),
            "operational_agency_profile_schema": (
                operational_agency_profile["schema_version"]
            ),
        },

    # ── v0.35 Self-Reflection ──
    # Post-episode structured critique stored in trajectory but not affecting score.
        # v0.2.2 (P1-3): None for non-Reflexion agents; identical
        # structure as the trajectory header's `agent_extras`.
        "agent_extras": agent_extras,
    }
    return result


# v0.2: per-case LP optimum cache. Keyed by case_file path so the
# expensive linprog call runs once per pglib-uc case across a batch run.
# v0.2.2 (P2-3): widen the value type so the negative-result sentinel
# `None` is type-correct (we deliberately cache failures so subsequent
# scenarios that share the same case file don't repeatedly retry).
_LP_OPTIMUM_CACHE: dict[str, float | None] = {}


def _maybe_lp_optimum(scenario: dict[str, Any]) -> float | None:
    """Return the LP economic-dispatch optimum for pglib-uc scenarios.

    Returns None for Grid2Op storms, CIGRE distribution, or any case
    file that fails to load — the optimality_gap dimension is then
    marked applicable=False for the episode.
    """
    backend_kind = str(scenario.get("backend_kind", ""))
    if backend_kind != "pglib_uc_synthetic":
        return None
    case_rel = str((scenario.get("backend_config") or {}).get("case_file", ""))
    if not case_rel:
        return None
    horizon = int(scenario.get("horizon_ticks", 24))
    cache_key = f"{case_rel}@{horizon}"
    if cache_key in _LP_OPTIMUM_CACHE:
        return _LP_OPTIMUM_CACHE[cache_key]
    try:
        from evaluation.lp_oracle import lp_dispatch_optimum

        raw_case_path = Path(case_rel)
        if raw_case_path.is_absolute() or ".." in raw_case_path.parts:
            return None
        case_path = (REPO_ROOT / raw_case_path).resolve()
        try:
            case_path.relative_to(REPO_ROOT)
        except ValueError:
            return None
        if not case_path.is_file():
            return None
        with open(case_path) as f:
            case = json.load(f)
        result = lp_dispatch_optimum(case, n_periods=horizon)
        if not result.feasible or result.optimum_cost <= 0:
            _LP_OPTIMUM_CACHE[cache_key] = None
            return None
        _LP_OPTIMUM_CACHE[cache_key] = result.optimum_cost
        return result.optimum_cost
    except Exception:
        return None


def _summarize_trajectory(
    actions: list[Action],
    llm_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    hist: Counter[str] = Counter()
    n_wait = 0
    n_tools = 0
    for action in actions:
        for call in action.tool_calls:
            n_tools += 1
            hist[call.name] += 1
            if call.name == "wait":
                n_wait += 1
    summary: dict[str, Any] = {
        "n_ticks": len(actions),
        "n_tool_calls": n_tools,
        "n_wait_actions": n_wait,
        "tool_histogram": dict(hist),
    }
    if llm_stats:
        llm_stats = dict(llm_stats)
        provider_request_records = list(
            llm_stats.pop("provider_request_records", []) or []
        )
        provider_response_records = list(
            llm_stats.pop("provider_response_records", []) or []
        )
        if provider_request_records:
            llm_stats["provider_request_count"] = len(
                provider_request_records
            )
            llm_stats["provider_request_sha256"] = [
                str(record.get("sha256") or "")
                for record in provider_request_records
                if isinstance(record, dict) and record.get("sha256")
            ]
        if provider_response_records:
            llm_stats["provider_response_count"] = len(
                provider_response_records
            )
            llm_stats["provider_response_sha256"] = [
                str(record.get("sha256") or "")
                for record in provider_response_records
                if isinstance(record, dict) and record.get("sha256")
            ]
        total_ticks = max(len(actions), 1)
        llm_stats["fallback_wait_ratio"] = round(
            float(llm_stats.get("ticks_wait_fallback", 0) or 0) / total_ticks,
            4,
        )
        summary["llm"] = llm_stats
    return summary


def _recompute_signature(
    scenario: dict[str, Any], spec: DomainSpec | None = None
) -> str:
    """Recompute the scenario signature so the result blob is self-checking."""
    spec = spec or get_domain_spec(scenario.get("domain"))
    return spec.scenario_signature(scenario, int(scenario.get("seed", 42)))


# Backward-compat alias: ``run._recompute_signature_with_seed`` must keep
# resolving for every existing caller. Kept as a plain name binding so
# ``from run import _recompute_signature_with_seed`` still works once
# ``run.py`` re-exports this name.
_recompute_signature_with_seed = recompute_signature_with_seed
