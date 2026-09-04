"""Metrics for the independent soft real-time supervision treatment.

These metrics intentionally do not modify ``SCORING_VERSION`` or any of the
frozen thirteen leaderboard dimensions. They measure interaction-system
behaviour that cannot be represented by deterministic logical-tick episodes.
Formal publication is decided only by the release-bound realtime batch gate;
a single episode never declares itself leaderboard eligible.
"""

from __future__ import annotations

from collections import Counter
from statistics import fmean
from typing import Any

SCHEMA_VERSION = "realtime-diagnostics/1.6"
RUNTIME_ASSURANCE_EVIDENCE_KINDS = frozenset(
    {
        "runtime_assurance_initialized",
        "runtime_assurance",
        "runtime_assurance_observation",
    }
)
ACTIONABLE_TRIGGER_KINDS = frozenset(
    {
        "environment_alarm",
        "safety_warning",
        "forecast_update",
        "tool_failure",
        "delayed_tool",
        "tool_result",
        "action_receipt",
        "native_opportunity",
        "scheduled_review",
        "supervisory_scan",
    }
)
ALARM_TRIGGER_KINDS = frozenset(
    {
        "environment_alarm",
        "safety_warning",
        "forecast_update",
        "tool_failure",
        "delayed_tool",
    }
)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _exact_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        normalized = int(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if isinstance(value, float) and value != normalized:
        return None
    return normalized


def takeover_evidence_is_causal(
    transition: dict[str, Any],
    *,
    evidence_by_id: dict[str, dict[str, Any]],
) -> bool:
    """Join a native takeover to its decision snapshot and realized effect."""

    if not str(transition.get("action_id") or "") or not str(
        transition.get("decision_id") or ""
    ):
        return False
    state_before = _exact_int(transition.get("state_version_before"))
    state_after = _exact_int(transition.get("state_version_after"))
    tick_before = _exact_int(transition.get("simulator_tick_before"))
    tick_after = _exact_int(transition.get("simulator_tick"))
    if (
        state_before is None
        or state_after is None
        or tick_before is None
        or tick_after is None
        or state_before < 0
        or state_after <= state_before
        or tick_before < 0
        or tick_after < tick_before
        or transition.get("simulator_time_advanced") is not True
    ):
        return False

    safety_ids = {
        str(value) for value in transition.get("safety_evidence_ids") or [] if value
    }
    decision_safety_ids = {
        str(value)
        for value in (transition.get("safety_decision") or {}).get("evidence_ids")
        or []
        if value
    }
    if not safety_ids or decision_safety_ids != safety_ids:
        return False
    expected_observation_tick = max(0, tick_before - 1)
    for evidence_id in safety_ids:
        item = evidence_by_id.get(evidence_id)
        if not isinstance(item, dict) or item.get("source") != "engine":
            return False
        evidence_tick = _exact_int(item.get("tick"))
        kind = item.get("kind")
        if kind == "runtime_assurance_initialized":
            if tick_before != 0 or evidence_tick != 0:
                return False
        elif (
            kind not in RUNTIME_ASSURANCE_EVIDENCE_KINDS
            or evidence_tick != expected_observation_tick
        ):
            return False

    applied_call_ids = {
        str(call.get("call_id") or "")
        for call in (transition.get("applied_action") or {}).get("actions") or []
        if isinstance(call, dict) and call.get("call_id")
    }
    effect_ids = {
        str(value) for value in transition.get("effect_evidence_ids") or [] if value
    }
    if not applied_call_ids or not effect_ids:
        return False
    proven_effect_calls: dict[str, str] = {}
    for edge in transition.get("tool_trace_edges") or []:
        if not isinstance(edge, dict) or edge.get("effect_proven") is not True:
            continue
        call_id = str(edge.get("call_id") or "")
        if call_id not in applied_call_ids:
            continue
        for evidence_id in edge.get("effect_evidence_ids") or []:
            if evidence_id:
                proven_effect_calls[str(evidence_id)] = call_id
    for evidence_id in effect_ids:
        item = evidence_by_id.get(evidence_id)
        if not (
            isinstance(item, dict)
            and item.get("source") == "engine"
            and item.get("kind") == "realized_event"
            and isinstance(item.get("payload"), dict)
        ):
            return False
        payload = item["payload"]
        call_id = str(payload.get("call_id") or "")
        before_digest = str(payload.get("before_state_digest") or "")
        after_digest = str(payload.get("after_state_digest") or "")
        effect_tick = _exact_int(item.get("tick"))
        if (
            not str(payload.get("event_id") or "")
            or str(payload.get("origin") or "") != "agent_caused"
            or call_id not in applied_call_ids
            or proven_effect_calls.get(evidence_id) != call_id
            or effect_tick is None
            or not tick_before <= effect_tick <= tick_after
            or not before_digest
            or not after_digest
            or before_digest == after_digest
        ):
            return False
    return True


def _turn_event_ids(turn: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in [
            turn.get("trigger_event_id"),
            *(turn.get("delivered_event_ids") or []),
            *(turn.get("steered_event_ids") or []),
            *(turn.get("causal_event_ids") or []),
        ]
        if value
    }


def _turn_acted(turn: dict[str, Any]) -> bool:
    return (
        turn.get("decision_valid") is not False
        and turn.get("deliberate_wait") is not True
        and bool(turn.get("action_id"))
        and turn.get("receipt_status") in {
            "confirmed",
            "effected",
            "no_effect",
        }
    )


def _turn_decision_no_action(turn: dict[str, Any]) -> bool:
    return (
        turn.get("status") == "completed"
        and turn.get("decision_valid") is not False
        and turn.get("decision_tick") is not None
        and (
            turn.get("deliberate_wait") is True
            or (
                not turn.get("action_id")
                and (
                    turn.get("action_is_wait") is True
                    or turn.get("receipt_status") == "no_action"
                )
            )
        )
    )


def _matching_effect(
    transitions: list[dict[str, Any]],
    *,
    turn: dict[str, Any],
    effected_actions: set[tuple[str, str]],
) -> dict[str, Any] | None:
    identity = (str(turn.get("decision_id") or ""), str(turn.get("action_id") or ""))
    if identity not in effected_actions:
        return None
    candidates: list[dict[str, Any]] = []
    for row in transitions:
        deferred_call_ids = {
            str(edge.get("call_id") or "")
            for outcome in row.get("deferred_action_outcomes") or []
            if isinstance(outcome, dict)
            for edge in outcome.get("tool_trace_edges") or []
            if isinstance(edge, dict) and edge.get("call_id")
        }
        immediate_match = (
            row.get("action_source") == "model"
            and row.get("effect_observed") is True
            and bool(row.get("effect_evidence_ids"))
            and str(row.get("decision_id") or "") == identity[0]
            and str(row.get("action_id") or "") == identity[1]
            and any(
                edge.get("effect_proven") is True
                for edge in row.get("tool_trace_edges") or []
                if isinstance(edge, dict)
                and str(edge.get("call_id") or "") not in deferred_call_ids
            )
        )
        deferred_match = any(
            str(outcome.get("decision_id") or "") == identity[0]
            and str(outcome.get("action_id") or "") == identity[1]
            and outcome.get("effect_observed") is True
            and bool(outcome.get("effect_evidence_ids"))
            and any(
                edge.get("effect_proven") is True
                for edge in outcome.get("tool_trace_edges") or []
                if isinstance(edge, dict)
            )
            for outcome in row.get("deferred_action_outcomes") or []
            if isinstance(outcome, dict)
        )
        if immediate_match or deferred_match:
            candidates.append(row)
    return min(candidates, key=lambda row: int(row.get("monotonic_ns", 0)), default=None)


def _turn_attempted_state_change(
    transitions: list[dict[str, Any]], *, turn: dict[str, Any]
) -> bool:
    """Distinguish a quiet-window intervention from a read-only investigation."""

    identity = (str(turn.get("decision_id") or ""), str(turn.get("action_id") or ""))
    state_change_flags: list[bool] = []
    for row in transitions:
        top_level_match = (
            str(row.get("decision_id") or ""),
            str(row.get("action_id") or ""),
        ) == identity
        if top_level_match:
            attempted = row.get("model_attempted_state_change")
            if isinstance(attempted, bool):
                if attempted:
                    return True
            elif row.get("action_source") == "model":
                state_change_flags.extend(
                    bool(result["state_changing"])
                    for result in row.get("tool_results") or []
                    if isinstance(result, dict)
                    and isinstance(result.get("state_changing"), bool)
                    and str(result.get("call_id") or "")
                    not in {
                        str(edge.get("call_id") or "")
                        for outcome in row.get("deferred_action_outcomes") or []
                        if isinstance(outcome, dict)
                        for edge in outcome.get("tool_trace_edges") or []
                        if isinstance(edge, dict)
                    }
                )
        if any(
            str(outcome.get("decision_id") or "") == identity[0]
            and str(outcome.get("action_id") or "") == identity[1]
            and (
                outcome.get("effect_observed") is True
                or outcome.get("control_confirmed") is True
            )
            for outcome in row.get("deferred_action_outcomes") or []
            if isinstance(outcome, dict)
        ):
            return True
    if state_change_flags:
        return any(state_change_flags)
    return turn.get("receipt_status") in {"confirmed", "effected"}


def evaluate_realtime_diagnostics(
    *,
    events: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    lifecycle: list[dict[str, Any]],
    polling_events: int = 0,
    interaction_stats: dict[str, Any] | None = None,
    evidence_ledger: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an evidence-like scorecard from append-only coordinator ledgers.

    ``polling_events`` counts only evaluator-identified polling beyond a
    declared plan or backend supervisory cadence. Required ``supervisory_scan``
    events are reported separately and must not be included in that value.
    """

    provider_stats = interaction_stats or {}
    has_native_counts = all(
        key in provider_stats
        for key in (
            "native_decision_responses",
            "native_tool_protocol_valid_responses",
            "native_tool_protocol_invalid_responses",
        )
    )
    if has_native_counts:
        logical_calls = max(
            0, int(provider_stats.get("native_decision_responses") or 0)
        )
    else:
        logical_calls = max(
            0,
            int(provider_stats.get("llm_calls_ok") or 0)
            + int(provider_stats.get("llm_calls_failed") or 0),
        )
    repair_attempts = max(
        0, int(provider_stats.get("protocol_repair_attempts") or 0)
    )
    repair_successes = max(
        0, int(provider_stats.get("protocol_repair_successes") or 0)
    )
    repair_dependent_calls = min(logical_calls, repair_attempts)
    native_valid = max(
        0,
        int(provider_stats.get("native_tool_protocol_valid_responses") or 0),
    ) if has_native_counts else max(0, logical_calls - repair_dependent_calls)
    native_invalid = max(
        0,
        int(provider_stats.get("native_tool_protocol_invalid_responses") or 0),
    ) if has_native_counts else repair_dependent_calls

    actionable_triggers = [
        event
        for event in events
        if event.get("kind") in ACTIONABLE_TRIGGER_KINDS
        and event.get("decision_required") is True
    ]
    trigger_ids = {str(event.get("event_id")) for event in actionable_triggers}
    alarm_triggers = [
        event
        for event in actionable_triggers
        if event.get("kind") in ALARM_TRIGGER_KINDS
    ]
    alarm_ids = {str(event.get("event_id")) for event in alarm_triggers}
    merged_ids_by_event = {
        str(event.get("event_id")): {
            str(value) for value in event.get("merged_event_ids") or [] if value
        }
        for event in actionable_triggers
    }

    def expand_merged(event_ids: set[str]) -> set[str]:
        expanded = set(event_ids)
        pending = list(event_ids)
        while pending:
            event_id = pending.pop()
            for merged_id in merged_ids_by_event.get(event_id, set()):
                if merged_id not in expanded:
                    expanded.add(merged_id)
                    pending.append(merged_id)
        return expanded

    delivered_ids = {
        event_id
        for turn in turns
        for event_id in _turn_event_ids(turn)
        if event_id in trigger_ids
    }
    delivered_ids = expand_merged(delivered_ids) & trigger_ids
    acknowledged_ids = {
        event_id
        for turn in turns
        if turn.get("status") == "completed"
        and turn.get("decision_valid") is not False
        for event_id in _turn_event_ids(turn)
        if event_id in trigger_ids
    }
    acknowledged_ids = expand_merged(acknowledged_ids) & trigger_ids
    decided_ids = {
        event_id
        for turn in turns
        if turn.get("status") == "completed"
        and turn.get("decision_valid") is not False
        and turn.get("decision_tick") is not None
        for event_id in _turn_event_ids(turn)
        if event_id in trigger_ids
    }
    decided_ids = expand_merged(decided_ids) & trigger_ids
    acted_ids = {
        event_id
        for turn in turns
        if _turn_acted(turn)
        for event_id in _turn_event_ids(turn)
        if event_id in trigger_ids
    }
    acted_ids = expand_merged(acted_ids) & trigger_ids
    decision_no_action_ids = {
        event_id
        for turn in turns
        if _turn_decision_no_action(turn)
        for event_id in _turn_event_ids(turn)
        if event_id in trigger_ids
    }
    decision_no_action_ids = expand_merged(decision_no_action_ids) & trigger_ids
    responded_ids = acted_ids | decision_no_action_ids
    quiet_windows = [event for event in events if event.get("kind") == "quiet_window"]
    quiet_ticks = {
        int(event.get("simulator_tick", -1))
        for event in quiet_windows
        if event.get("simulator_tick") is not None
    }
    evaluator_necessity_by_event = {
        str(event.get("event_id")): event.get(
            "evaluator_only_proactive_action_necessary"
        )
        for event in events
        if event.get("event_id")
    }
    false_alarm_turns: list[dict[str, Any]] = []
    false_alarm_assessed_turns: list[dict[str, Any]] = []
    false_alarm_unassessed_turns: list[dict[str, Any]] = []
    for turn in turns:
        if (
            turn.get("status") != "completed"
            or turn.get("decision_valid") is False
            or turn.get("deliberate_wait") is True
            or not turn.get("action_id")
        ):
            continue
        necessity = turn.get("proactive_action_necessary")
        if necessity is None:
            direct_event_ids = {
                str(value)
                for value in [
                    turn.get("trigger_event_id"),
                    *(turn.get("delivered_event_ids") or []),
                    *(turn.get("steered_event_ids") or []),
                ]
                if value
            }
            direct_necessities = [
                evaluator_necessity_by_event[event_id]
                for event_id in direct_event_ids
                if event_id in evaluator_necessity_by_event
            ]
            if any(value is True for value in direct_necessities):
                necessity = True
            elif direct_necessities and all(
                value is False for value in direct_necessities
            ):
                necessity = False
        if (
            int(turn.get("started_tick", -1)) not in quiet_ticks
            or not _turn_attempted_state_change(transitions, turn=turn)
        ):
            continue
        if isinstance(necessity, bool):
            false_alarm_assessed_turns.append(turn)
        else:
            false_alarm_unassessed_turns.append(turn)
        if necessity is False:
            false_alarm_turns.append(turn)
    delegated_hold_turns = [
        turn
        for turn in turns
        if turn.get("status") == "completed"
        and turn.get("decision_valid") is not False
        and turn.get("deliberate_wait") is True
        and turn.get("receipt_status") in {"no_effect", "confirmed"}
        and int(turn.get("started_tick", -1)) in quiet_ticks
    ]
    standing_plan_quiet_ticks = {
        int(event.get("simulator_tick", -1))
        for event in quiet_windows
        if event.get("model_confirmed_standing_plan") is True
    }
    delegated_hold_ticks = {
        int(turn.get("started_tick", -1)) for turn in delegated_hold_turns
    }
    model_silence_opportunity_ticks = (
        standing_plan_quiet_ticks | delegated_hold_ticks
    )
    correct_silence_ticks = {
        tick
        for tick in model_silence_opportunity_ticks
        if not any(
            int(turn.get("started_tick", -1)) == tick
            and _turn_attempted_state_change(transitions, turn=turn)
            for turn in turns
        )
    }
    autonomous_quiet_ticks = quiet_ticks - {
        int(turn.get("started_tick", -1))
        for turn in turns
        if int(turn.get("started_tick", -1)) in quiet_ticks
    }
    unattributed_quiet_ticks = quiet_ticks - model_silence_opportunity_ticks
    alarm_intervention_identities = {
        (
            str(turn.get("decision_id") or ""),
            str(turn.get("action_id") or ""),
        )
        for turn in turns
        if turn.get("decision_valid") is not False
        and turn.get("action_id")
        and expand_merged(_turn_event_ids(turn)) & alarm_ids
        and _turn_attempted_state_change(transitions, turn=turn)
    }
    effected_actions = {
        (str(row.get("decision_id") or ""), str(row.get("action_id") or ""))
        for row in lifecycle
        if row.get("status") == "effected"
        and row.get("decision_id")
        and row.get("action_id")
    }

    effected_ids: set[str] = set()
    for turn in turns:
        if (
            _matching_effect(
                transitions,
                turn=turn,
                effected_actions=effected_actions,
            )
            is None
        ):
            continue
        effected_ids.update(expand_merged(_turn_event_ids(turn)) & trigger_ids)

    alarm_to_decision_ticks: list[float] = []
    alarm_to_decision_wall_ms: list[float] = []
    alarm_to_effect_ticks: list[float] = []
    alarm_to_effect_wall_ms: list[float] = []
    for alarm in alarm_triggers:
        alarm_id = str(alarm.get("event_id"))
        matching = [
            turn
            for turn in turns
            if alarm_id in _turn_event_ids(turn)
            and turn.get("status") == "completed"
            and turn.get("decision_valid") is not False
            and turn.get("decision_tick") is not None
        ]
        decision_turn = (
            min(
                matching,
                key=lambda row: int(
                    row.get("decision_monotonic_ns")
                    or row.get("started_monotonic_ns", 0)
                ),
            )
            if matching
            else None
        )
        if decision_turn is None or decision_turn.get("decision_tick") is None:
            continue
        alarm_tick = int(alarm.get("simulator_tick", 0))
        alarm_ns = int(alarm.get("monotonic_ns", 0))
        decision_tick = int(decision_turn["decision_tick"])
        decision_ns = int(decision_turn.get("decision_monotonic_ns", alarm_ns))
        alarm_to_decision_ticks.append(float(max(0, decision_tick - alarm_tick)))
        alarm_to_decision_wall_ms.append(float(max(0, decision_ns - alarm_ns)) / 1e6)
        effects = [
            effect
            for turn in matching
            for effect in [
                _matching_effect(
                    transitions,
                    turn=turn,
                    effected_actions=effected_actions,
                )
            ]
            if effect is not None
        ]
        effect = min(
            effects,
            key=lambda row: int(row.get("monotonic_ns", 0)),
            default=None,
        )
        if effect is not None:
            effect_tick = int(
                effect.get(
                    "simulator_tick",
                    effect.get("state_version_after", decision_tick),
                )
            )
            effect_ns = int(effect.get("monotonic_ns", decision_ns))
            alarm_to_effect_ticks.append(float(max(0, effect_tick - alarm_tick)))
            alarm_to_effect_wall_ms.append(float(max(0, effect_ns - alarm_ns)) / 1e6)

    lifecycle_counts = Counter(str(row.get("status") or "unknown") for row in lifecycle)
    late_discarded = sum(bool(turn.get("late_response_discarded")) for turn in turns)
    turn_canceled = sum(bool(turn.get("cancel_requested")) for turn in turns)
    turn_superseded = sum(turn.get("status") == "superseded" for turn in turns)
    timeout_invalidated = sum(
        turn.get("invalidated_reason") == "EPISODE_WALL_TIMEOUT" for turn in turns
    )
    pending_late_responses = sum(
        bool(turn.get("late_response_pending")) for turn in turns
    )
    stale_or_discarded = (
        lifecycle_counts["stale"]
        + lifecycle_counts["canceled"]
        + lifecycle_counts["superseded"]
        + late_discarded
    )
    safety_supervisor_transitions = [
        row
        for row in transitions
        if row.get("action_source") == "safety_supervisor"
        and row.get("safety_supervisor_failed") is not True
        and row.get("applied_action") is not None
    ]
    takeover_modes = Counter(
        str((row.get("safety_decision") or {}).get("mode") or "unknown")
        for row in safety_supervisor_transitions
    )
    controlled_holds = takeover_modes["controlled_hold"]
    takeover_candidates = [
        row
        for row in safety_supervisor_transitions
        if (
            str((row.get("safety_decision") or {}).get("mode") or "")
            == "minimum_risk_fallback"
            or str((row.get("safety_decision") or {}).get("mode") or "")
            .startswith("native_")
            or "takeover"
            in str((row.get("safety_decision") or {}).get("mode") or "")
        )
    ]
    evidence_by_id = {
        str(row.get("evidence_id")): row
        for row in evidence_ledger or []
        if isinstance(row, dict) and row.get("evidence_id")
    }

    verified_takeover_transitions = [
        row
        for row in takeover_candidates
        if takeover_evidence_is_causal(row, evidence_by_id=evidence_by_id)
        and row.get("effect_observed") is True
    ]
    takeover_modes_eligible = dict(
        Counter(
            str((row.get("safety_decision") or {}).get("mode") or "unknown")
            for row in verified_takeover_transitions
        )
    )
    takeovers = sum(takeover_modes_eligible.values())
    unverified_takeovers = len(takeover_candidates) - takeovers
    safety_failures = sum(
        row.get("safety_supervisor_failed") is True for row in transitions
    )
    scheduled_review_events = [
        event for event in events if event.get("kind") == "scheduled_review"
    ]
    supervisory_scan_events = [
        event for event in events if event.get("kind") == "supervisory_scan"
    ]
    completed_review_ids = {
        str(turn.get("trigger_event_id"))
        for turn in turns
        if turn.get("trigger_kind") == "scheduled_review"
        and turn.get("status") == "completed"
        and turn.get("decision_valid") is not False
    }
    completed_scan_ids = {
        str(turn.get("trigger_event_id"))
        for turn in turns
        if turn.get("trigger_kind") == "supervisory_scan"
        and turn.get("status") == "completed"
        and turn.get("decision_valid") is not False
    }
    transition_demands = [
        event
        for event in actionable_triggers
        if str((event.get("payload") or {}).get("type") or "")
        == "transition_demand"
    ]
    environment_ticks = sum(
        row.get("simulator_time_advanced") is not False for row in transitions
    )

    stage_by_kind: dict[str, dict[str, int]] = {}
    for kind in sorted(ACTIONABLE_TRIGGER_KINDS):
        kind_ids = {
            str(event.get("event_id"))
            for event in actionable_triggers
            if event.get("kind") == kind
        }
        if kind_ids:
            stage_by_kind[kind] = {
                "actionable": len(kind_ids),
                "detected": len(kind_ids & acknowledged_ids),
                "delivered": len(kind_ids & delivered_ids),
                "acknowledged": len(kind_ids & acknowledged_ids),
                "decided": len(kind_ids & decided_ids),
                "acted": len(kind_ids & acted_ids),
                "effected": len(kind_ids & effected_ids),
                "decision_no_action": len(kind_ids & decision_no_action_ids),
                "delivery_missed": len(kind_ids - delivered_ids),
                "decision_missed": len(kind_ids - decided_ids),
                "response_missed": len(kind_ids - responded_ids),
            }

    return {
        "schema_version": SCHEMA_VERSION,
        "treatment": "realtime_persistent",
        "leaderboard_eligible": False,
        "measurement_contract": {
            "detected_field_semantics": (
                "completed_valid_transport_acknowledgement"
            ),
            "semantic_detection_supported": False,
            "correct_silence_semantics": (
                "model_confirmed_standing_plan_or_explicit_delegated_hold_"
                "without_attempting_state_change"
            ),
            "false_alarm_rate_semantics": (
                "false_alarms_per_evaluator_assessed_quiet_window_intervention"
            ),
            "unassessed_interventions_excluded_from_false_alarm_rate": True,
            "harness_environment_quiet_excluded_from_model_silence": True,
        },
        "provider_protocol": {
            "logical_calls": logical_calls,
            "native_valid_without_repair": native_valid,
            "native_invalid_responses": native_invalid,
            "repair_attempts": repair_attempts,
            "repair_successes": repair_successes,
            "repair_failures": max(0, repair_attempts - repair_successes),
            "repair_dependent_call_rate": (
                float(repair_dependent_calls / logical_calls)
                if logical_calls
                else None
            ),
            "repair_success_rate": (
                float(repair_successes / repair_attempts)
                if repair_attempts
                else None
            ),
        },
        "trigger_response": {
            "actionable": len(actionable_triggers),
            "detected": len(acknowledged_ids),
            "delivered": len(delivered_ids),
            "acknowledged": len(acknowledged_ids),
            "decided": len(decided_ids),
            "acted": len(acted_ids),
            "effected": len(effected_ids),
            "decision_no_action": len(decision_no_action_ids),
            "missed": len(actionable_triggers) - len(responded_ids),
            "delivery_missed": len(trigger_ids - delivered_ids),
            "decision_missed": len(trigger_ids - decided_ids),
            "response_missed": len(trigger_ids - responded_ids),
            "by_kind": stage_by_kind,
        },
        "alarm_response": {
            "actionable_alarms": len(alarm_triggers),
            "detected": len(alarm_ids & acknowledged_ids),
            "missed": len(alarm_ids - responded_ids),
            "delivery_missed": len(alarm_ids - delivered_ids),
            "decision_missed": len(alarm_ids - decided_ids),
            "response_missed": len(alarm_ids - responded_ids),
            "false_alarms": len(false_alarm_turns),
            "false_alarm_assessed_interventions": len(
                false_alarm_assessed_turns
            ),
            "false_alarm_unassessed_interventions": len(
                false_alarm_unassessed_turns
            ),
            "false_alarm_rate": (
                float(len(false_alarm_turns) / len(false_alarm_assessed_turns))
                if false_alarm_assessed_turns
                else None
            ),
            "model_attempted_interventions": len(
                alarm_intervention_identities
            ),
            "quiet_windows": len(quiet_windows),
            "agent_silence_opportunities": len(
                model_silence_opportunity_ticks
            ),
            "correct_silence": len(correct_silence_ticks),
            "model_standing_plan_quiet_windows": len(
                standing_plan_quiet_ticks
            ),
            "model_delegated_hold_windows": len(delegated_hold_ticks),
            "autonomous_quiet_windows": len(autonomous_quiet_ticks),
        },
        "harness_environment": {
            "quiet_windows": len(quiet_windows),
            "quiet_windows_without_model_turn": len(autonomous_quiet_ticks),
            "unattributed_quiet_windows": len(unattributed_quiet_ticks),
        },
        "latency": {
            "alarm_to_decision_ticks": _summary(alarm_to_decision_ticks),
            "alarm_to_decision_wall_ms": _summary(alarm_to_decision_wall_ms),
            "alarm_to_effect_ticks": _summary(alarm_to_effect_ticks),
            "alarm_to_effect_wall_ms": _summary(alarm_to_effect_wall_ms),
        },
        "autonomy": {
            "unnecessary_polling": int(polling_events),
            "invalid_model_responses": sum(
                turn.get("decision_valid") is False for turn in turns
            ),
            "model_turns": len(turns),
            "environment_ticks": environment_ticks,
            "environment_ticks_per_model_turn": (
                float(environment_ticks / len(turns)) if turns else None
            ),
            "scheduled_reviews": len(scheduled_review_events),
            "scheduled_reviews_served": sum(
                str(event.get("event_id")) in completed_review_ids
                for event in scheduled_review_events
            ),
            "supervisory_scans": len(supervisory_scan_events),
            "supervisory_scans_served": sum(
                str(event.get("event_id")) in completed_scan_ids
                for event in supervisory_scan_events
            ),
        },
        "action_lifecycle": {
            "stale": lifecycle_counts["stale"],
            "expired": lifecycle_counts["expired"],
            "canceled": lifecycle_counts["canceled"],
            "superseded": lifecycle_counts["superseded"],
            "rejected": lifecycle_counts["rejected"],
            "failed": lifecycle_counts["failed"],
            "no_effect": lifecycle_counts["no_effect"],
            "confirmed": lifecycle_counts["confirmed"],
            "effected": lifecycle_counts["effected"],
            "late_response_discarded": late_discarded,
            "turn_cancel_requested": turn_canceled,
            "turn_superseded": turn_superseded,
            "timeout_invalidated_turns": timeout_invalidated,
            "pending_late_responses_at_return": pending_late_responses,
            "stale_or_discarded": stale_or_discarded,
        },
        "safety": {
            "takeovers": takeovers,
            "takeover_modes": dict(sorted(takeover_modes_eligible.items())),
            "unverified_takeovers": unverified_takeovers,
            "controlled_holds": controlled_holds,
            "transition_demands": len(transition_demands),
            "transition_demands_acknowledged": sum(
                str(event.get("event_id")) in delivered_ids
                for event in transition_demands
            ),
            "minimum_risk_fallbacks": takeover_modes_eligible.get(
                "minimum_risk_fallback", 0
            ),
            "supervisor_failures": safety_failures,
        },
    }
