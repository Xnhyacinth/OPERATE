"""Objective trajectory-complexity metrics and replay-based minimization.

Observed metrics are computed from executed tool results.  Metrics that require
causal claims stay explicitly unknown until a deterministic backend replay is
performed; a single successful trajectory cannot prove global minimality.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from core import Action, ToolCall
from core.world_evolution_contract import canonicalize_runtime_events

META_TOOLS = frozenset({"wait", "noop"})
PLAN_TOOL = "commit_to_plan"
TRAJECTORY_ANALYSIS_CONTRACT_VERSION = "0.7"
RUNNER_AUTONOMY_ACTIONS = frozenset(
    {
        "autonomous_plan_hold",
        "decision_budget_hold",
        "native_idle_hold",
        "pending_action_hold",
    }
)


def _finite_int_tick(value: Any) -> int | None:
    """Coerce a finite integral tick; malformed evidence is ignored."""
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


def _record_tick(record: dict[str, Any]) -> int | None:
    """Read a runtime tick without letting malformed evidence crash audit."""
    return _finite_int_tick(record.get("applied_tick"))


def _effect_tick(record: dict[str, Any]) -> int | None:
    return _finite_int_tick(
        record.get(
            "effect_tick",
            record.get("outcome_tick", record.get("applied_tick", record.get("tick"))),
        )
    )


def _string_ids(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def _evidence_ids(record: dict[str, Any]) -> set[str]:
    values = _string_ids(record.get("produces_evidence_ids"))
    values.update(_string_ids(record.get("evidence_ids")))
    singular = record.get("evidence_id")
    if isinstance(singular, str) and singular:
        values.add(singular)
    return values


def _observed_event_evidence(
    decision_envelope: dict[str, Any],
) -> tuple[set[str], set[str]]:
    observation = decision_envelope.get("pre_action_observation")
    if not isinstance(observation, dict):
        return set(), set()
    event_ids: set[str] = set()
    event_evidence_ids: set[str] = set()
    raw_events = observation.get("__last_realized_events__")
    for event in raw_events if isinstance(raw_events, list) else []:
        if not isinstance(event, dict):
            continue
        event_id = event.get("event_id", event.get("id"))
        if isinstance(event_id, str) and event_id:
            event_ids.add(event_id)
        event_evidence_ids.update(_evidence_ids(event))
    event_evidence_ids.update(
        _string_ids(observation.get("__last_evidence_ids__"))
    )
    return event_ids, event_evidence_ids


def _action_outcome_edge_matches(
    edge: Any, *, call_id: str, outcome_event_id: str
) -> bool:
    if not isinstance(edge, dict):
        return False
    kind = str(edge.get("kind") or "")
    if kind == "action_to_outcome":
        return (
            edge.get("source") == f"call:{call_id}"
            and edge.get("target") == f"outcome:{outcome_event_id}"
        )
    if kind == "native_control_to_state_effect":
        return (
            edge.get("source_call_id") == call_id
            and edge.get("target_event_id") == outcome_event_id
        )
    return False


def _result_proves_state_change(result: dict[str, Any]) -> bool:
    payload = result.get("payload")
    if payload is not None and not isinstance(payload, dict):
        return False
    return bool(
        result.get("ok") is True
        and result.get("state_changing") is True
        and str((payload or {}).get("_status") or "") != "pending"
    )


def validated_event_action_edges(
    edges: Any,
    *,
    material_exogenous_records: Any,
    agent_caused_records: Any,
) -> list[dict[str, Any]]:
    """Return only exact runtime event-decision-call-outcome chains."""
    if not isinstance(edges, list):
        return []
    material_by_id: dict[str, dict[str, Any]] = {}
    duplicate_material_ids: set[str] = set()
    for record in (
        material_exogenous_records
        if isinstance(material_exogenous_records, list)
        else []
    ):
        if not isinstance(record, dict) or record.get("material_exogenous") is not True:
            continue
        event_id = record.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            continue
        if event_id in material_by_id:
            duplicate_material_ids.add(event_id)
        else:
            material_by_id[event_id] = record
    effects_by_id: dict[str, dict[str, Any]] = {}
    duplicate_effect_ids: set[str] = set()
    for record in agent_caused_records if isinstance(agent_caused_records, list) else []:
        if not isinstance(record, dict):
            continue
        event_id = record.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            continue
        if event_id in effects_by_id:
            duplicate_effect_ids.add(event_id)
        else:
            effects_by_id[event_id] = record

    validated: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        source_event_id = edge.get("source_event_id")
        outcome_event_id = edge.get("outcome_event_id")
        call_id = edge.get("call_id")
        tool_name = edge.get("tool_name")
        if not all(
            isinstance(value, str) and value
            for value in (
                source_event_id,
                outcome_event_id,
                call_id,
                tool_name,
            )
        ):
            continue
        if (
            source_event_id in duplicate_material_ids
            or outcome_event_id in duplicate_effect_ids
        ):
            continue
        material = material_by_id.get(source_event_id)
        effect = effects_by_id.get(outcome_event_id)
        if material is None or effect is None:
            continue
        source_tick = _finite_int_tick(edge.get("source_event_tick"))
        decision_tick = _finite_int_tick(edge.get("decision_tick"))
        effect_tick = _finite_int_tick(edge.get("effect_tick"))
        material_tick = _record_tick(material)
        recorded_effect_tick = _effect_tick(effect)
        event_evidence_ids = _string_ids(edge.get("event_evidence_ids"))
        matched_event_evidence_ids = _string_ids(
            edge.get("matched_event_evidence_ids")
        )
        call_evidence_ids = _string_ids(edge.get("call_evidence_ids"))
        effect_evidence_ids = _string_ids(edge.get("effect_evidence_ids"))
        recorded_event_evidence_ids = _evidence_ids(material)
        recorded_effect_evidence_ids = _evidence_ids(effect)
        changed_state_fields = edge.get("changed_state_fields")
        recorded_changed_state_fields = effect.get("changed_state_fields")
        material_changed_state_fields = material.get("changed_state_fields")
        if (
            edge.get("kind") != "event_to_post_event_action_outcome"
            or source_tick is None
            or decision_tick is None
            or effect_tick is None
            or material_tick != source_tick
            or not isinstance(material_changed_state_fields, list)
            or not material_changed_state_fields
            or not source_tick < decision_tick <= effect_tick
            or recorded_effect_tick != effect_tick
            or str(effect.get("origin") or "") != "agent_caused"
            or str(effect.get("call_id") or "") != call_id
            or str(effect.get("tool_name") or "") != tool_name
            or event_evidence_ids != recorded_event_evidence_ids
            or (
                recorded_event_evidence_ids
                and not matched_event_evidence_ids
            )
            or not matched_event_evidence_ids.issubset(
                recorded_event_evidence_ids
            )
            or not call_evidence_ids
            or effect_evidence_ids != recorded_effect_evidence_ids
            or not call_evidence_ids.intersection(effect_evidence_ids)
            or not isinstance(changed_state_fields, list)
            or not changed_state_fields
            or changed_state_fields != recorded_changed_state_fields
            or not str(effect.get("before_state_digest") or "")
            or not str(effect.get("after_state_digest") or "")
            or effect.get("before_state_digest") == effect.get("after_state_digest")
            or not _action_outcome_edge_matches(
                effect.get("action_to_outcome_edge"),
                call_id=call_id,
                outcome_event_id=outcome_event_id,
            )
        ):
            continue
        identity = (source_event_id, decision_tick, call_id, outcome_event_id)
        if identity in seen:
            continue
        seen.add(identity)
        validated.append(dict(edge))
    return validated


@dataclass
class MinimizationResult:
    actions: list[Action]
    tool_set: list[str]
    distinct_tool_count: int
    n_replays: int
    status: str


def analyze_trajectory_steps(
    steps: list[dict[str, Any]],
    *,
    per_action_attribution: list[dict[str, Any]] | None = None,
    llm_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute evidence-bounded interaction metrics from serialized steps."""
    successful: set[str] = set()
    state_changing: set[str] = set()
    physical_actuator_endpoints: set[str] = set()
    state_change_ticks: set[int] = set()
    information_ticks: set[int] = set()
    model_decision_ticks: set[int] = set()
    environment_ticks: set[int] = set()
    material_exogenous_records: list[dict[str, Any]] = []
    source_scheduled_records: list[dict[str, Any]] = []
    perturbation_records: list[dict[str, Any]] = []
    endogenous_records: list[dict[str, Any]] = []
    agent_caused_records: list[dict[str, Any]] = []
    control_sequence: list[str] = []
    phase_sequence: list[str] = []
    plan_revisions = 0
    plan_revision_ticks: list[int] = []
    dependency_nodes: dict[str, set[str]] = {}
    evidence_producers: dict[str, str] = {}
    graph_nodes: dict[str, dict[str, Any]] = {}
    graph_edges: set[tuple[str, str, str]] = set()
    dependency_metadata_complete = True
    non_meta_call_count = 0
    decision_calls_by_id: dict[str, dict[str, Any]] = {}
    ambiguous_decision_call_ids: set[str] = set()
    results_by_call_id: dict[str, list[dict[str, Any]]] = {}

    for index, step in enumerate(steps):
        tick = _finite_int_tick(step.get("tick", index))
        if tick is None:
            tick = index
        action = step.get("action") or {}
        calls = action.get("actions") or []
        if str(action.get("dominant_action") or "") not in (
            RUNNER_AUTONOMY_ACTIONS
        ):
            model_decision_ticks.add(tick)
        results = step.get("tool_results") or []
        info = step.get("info") or {}
        decision_envelope = info.get("decision_envelope") or {}
        observation_node_id = None
        if isinstance(decision_envelope, dict):
            observation_sha256 = str(
                decision_envelope.get("pre_action_observation_sha256") or ""
            )
            if observation_sha256:
                observation_node_id = f"observation:{observation_sha256}"
                graph_nodes.setdefault(
                    observation_node_id,
                    {
                        "id": observation_node_id,
                        "kind": "observation",
                        "tick": (
                            _finite_int_tick(
                                decision_envelope.get("observation_tick")
                            )
                            if decision_envelope.get("observation_tick") is not None
                            and _finite_int_tick(
                                decision_envelope.get("observation_tick")
                            )
                            is not None
                            else tick
                        ),
                        "observation_sha256": observation_sha256,
                    },
                )
        world_records = [
            row
            for row in (
                step.get("world_evolution_records")
                or (info.get("extra") or {}).get("world_evolution_records")
                or []
            )
            if isinstance(row, dict)
        ]
        if not world_records:
            world_records = canonicalize_runtime_events(
                [
                    row
                    for row in (
                        list(info.get("realized_events") or [])
                        + list(info.get("fault_injections") or [])
                    )
                    if isinstance(row, dict)
                ],
                applied_tick=tick,
            )
        for record in world_records:
            origin = str(record.get("origin") or "")
            if origin == "source_schedule":
                source_scheduled_records.append(record)
            elif origin == "declared_perturbation":
                perturbation_records.append(record)
            elif origin == "endogenous_completion":
                endogenous_records.append(record)
            elif origin == "agent_caused":
                agent_caused_records.append(record)
            if record.get("material_exogenous") is True:
                material_exogenous_records.append(record)
        within_tick = ((info.get("extra") or {}).get("within_tick_investigation") or {})
        if isinstance(within_tick, dict):
            query_calls = (
                (within_tick.get("investigation_action") or {}).get("actions") or []
            )
            query_results = within_tick.get("tool_results") or []
            calls = list(query_calls) + list(calls)
            results = list(query_results) + list(results)
        for result in results:
            if not isinstance(result, dict):
                continue
            result_call_id = result.get("call_id")
            if isinstance(result_call_id, str) and result_call_id:
                results_by_call_id.setdefault(result_call_id, []).append(result)
        if (
            isinstance(decision_envelope, dict)
            and str(action.get("dominant_action") or "")
            not in RUNNER_AUTONOMY_ACTIONS
        ):
            decision_tick = tick
            observed_event_ids, observed_event_evidence_ids = (
                _observed_event_evidence(decision_envelope)
            )
            if decision_tick is not None:
                for call in calls:
                    if not isinstance(call, dict):
                        continue
                    call_id = call.get("call_id")
                    if not isinstance(call_id, str) or not call_id:
                        continue
                    record = {
                        "decision_tick": decision_tick,
                        "tool_name": str(
                            call.get("name") or call.get("action") or ""
                        ),
                        "consumes_evidence_ids": _string_ids(
                            call.get("consumes_evidence_ids")
                        ),
                        "observed_event_ids": observed_event_ids,
                        "observed_event_evidence_ids": (
                            observed_event_evidence_ids
                        ),
                    }
                    if call_id in decision_calls_by_id:
                        ambiguous_decision_call_ids.add(call_id)
                    else:
                        decision_calls_by_id[call_id] = record
        for result in results:
            if isinstance(result, dict) and result.get("call_id") and result.get("evidence_id"):
                evidence_id = str(result["evidence_id"])
                call_id = str(result["call_id"])
                evidence_producers[evidence_id] = call_id
                graph_nodes.setdefault(
                    f"call:{call_id}",
                    {
                        "id": f"call:{call_id}",
                        "kind": "action",
                        "tick": tick,
                        "tool": str(result.get("name") or ""),
                        "state_changing": bool(
                            result.get("state_changing")
                        ),
                    },
                )
                graph_nodes[f"evidence:{evidence_id}"] = {
                    "id": f"evidence:{evidence_id}",
                    "kind": (
                        "outcome"
                        if bool(result.get("state_changing"))
                        else "observation"
                    ),
                    "tick": tick,
                    "evidence_id": evidence_id,
                }
                graph_edges.add(
                    (f"call:{call_id}", f"evidence:{evidence_id}", "produces")
                )
        for result in results:
            if not isinstance(result, dict) or not result.get("ok"):
                continue
            name = str(result.get("name") or "")
            if not name or name in META_TOOLS:
                continue
            successful.add(name)
            if bool(result.get("state_changing")):
                state_changing.add(name)
                endpoint = _physical_actuator_endpoint_token(result, name)
                if endpoint is not None:
                    physical_actuator_endpoints.add(endpoint)
                effect_tick = _effect_tick(result)
                if effect_tick is None:
                    effect_tick = _finite_int_tick(step.get("applied_tick"))
                state_change_ticks.add(
                    tick if effect_tick is None else effect_tick
                )
                strategy_token = _control_strategy_token(result, name)
                if strategy_token is not None:
                    control_sequence.append(strategy_token)
                _append_phase(phase_sequence, "control")
            elif (result.get("payload") or {}).get("_status") != "pending" and name != PLAN_TOOL:
                information_ticks.add(tick)
                _append_phase(phase_sequence, "information")
        for call_index, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or call.get("action") or "")
            result = _matching_result(results, call_index, name)
            if not name or name in META_TOOLS or not result or not result.get("ok"):
                continue
            non_meta_call_count += 1
            call_id = call.get("call_id")
            if observation_node_id and call_id is not None:
                graph_edges.add(
                    (
                        observation_node_id,
                        f"call:{call_id}",
                        "consumes_context",
                    )
                )
            consumes = call.get("consumes_evidence_ids")
            depends = call.get("depends_on_call_ids")
            if call_id is None or consumes is None or depends is None:
                dependency_metadata_complete = False
            else:
                direct_dependencies = {str(value) for value in depends}
                direct = set(direct_dependencies)
                direct.update(
                    evidence_producers[str(value)]
                    for value in consumes
                    if str(value) in evidence_producers
                )
                call_id = str(call_id)
                dependency_nodes[call_id] = direct
                graph_nodes[f"call:{call_id}"] = {
                    "id": f"call:{call_id}",
                    "kind": "action",
                    "tick": tick,
                    "tool": name,
                    "state_changing": bool(result.get("state_changing")),
                }
                for dependency in direct_dependencies:
                    graph_edges.add(
                        (
                            f"call:{dependency}",
                            f"call:{call_id}",
                            "depends_on",
                        )
                    )
                for evidence_id in {str(value) for value in consumes}:
                    graph_edges.add(
                        (
                            f"evidence:{evidence_id}",
                            f"call:{call_id}",
                            "consumes",
                        )
                    )
            if name == PLAN_TOOL:
                _append_phase(phase_sequence, "plan")
                args = call.get("args") or {}
                if args.get("replaces_plan_id"):
                    plan_revisions += 1
                    plan_revision_ticks.append(tick)

        if info.get("realized_events") or info.get("fault_injections"):
            environment_ticks.add(tick)

    attributed = per_action_attribution is not None
    if attributed:
        step_ticks = [
            (
                _finite_int_tick(step.get("tick", index))
                if _finite_int_tick(step.get("tick", index)) is not None
                else index
            )
            for index, step in enumerate(steps)
        ]
        effective_ticks = sorted(
            {
                _trajectory_tick_for_action_index(int(row["tick"]), step_ticks)
                for row in per_action_attribution or []
                if float(row.get("marginal_prevented_loss", 0.0)) > 0.0
            }
        )
        effective_status = "counterfactual_attributed"
    else:
        effective_ticks = sorted(state_change_ticks)
        effective_status = "state_change_proxy"

    simulator_ticks = len(steps)
    runner_autonomy_ticks = sum(
        str((step.get("action") or {}).get("dominant_action") or "")
        in RUNNER_AUTONOMY_ACTIONS
        for step in steps
    )
    interaction_turns = simulator_ticks - runner_autonomy_ticks
    if llm_stats:
        reported = int(llm_stats.get("llm_calls_ok", 0) or 0) + int(
            llm_stats.get("llm_calls_failed", 0) or 0
        )
        if reported > 0:
            interaction_turns = reported

    exact_depth = None
    dependency_status = "metadata_incomplete"
    if non_meta_call_count and dependency_metadata_complete:
        exact_depth = _dependency_depth(dependency_nodes)
        dependency_status = "declared_evidence_action_dag"
    post_change_decision_ticks = sorted(
        {
            tick
            for event in material_exogenous_records
            if (event_tick := _record_tick(event)) is not None
            for tick in model_decision_ticks
            if tick > event_tick
        }
    )
    for event in material_exogenous_records:
        event_tick = _record_tick(event)
        later_ticks = [
            tick
            for tick in model_decision_ticks
            if event_tick is not None and tick > event_tick
        ]
        event["later_decision_opportunity"] = bool(later_ticks)
        event["next_decision_tick"] = min(later_ticks) if later_ticks else None
    event_to_decision_edges = [
        {
            "source_event_id": str(event.get("event_id") or ""),
            "target_tick": next_decision_tick,
            "kind": "event_to_post_change_decision",
        }
        for event in material_exogenous_records
        if (event_tick := _record_tick(event)) is not None
        if (
            next_decision_tick := _finite_int_tick(event.get("next_decision_tick"))
        )
        is not None
        and next_decision_tick > event_tick
    ]
    event_to_action_edges: list[dict[str, Any]] = []
    seen_event_action_edges: set[tuple[str, int, str, str]] = set()
    for event in material_exogenous_records:
        event_id = str(event.get("event_id") or "")
        event_tick = _record_tick(event)
        event_evidence_ids = _evidence_ids(event)
        if not event_id or event_tick is None:
            continue
        for call_id, decision in decision_calls_by_id.items():
            decision_tick = int(decision["decision_tick"])
            if (
                call_id in ambiguous_decision_call_ids
                or decision_tick <= event_tick
                or event_id not in decision["observed_event_ids"]
            ):
                continue
            observed_event_evidence_ids = set(
                decision["observed_event_evidence_ids"]
            )
            consumed_evidence_ids = set(decision["consumes_evidence_ids"])
            matched_event_evidence_ids = event_evidence_ids.intersection(
                observed_event_evidence_ids.union(consumed_evidence_ids)
            )
            if event_evidence_ids and not matched_event_evidence_ids:
                continue
            call_results = [
                result
                for result in results_by_call_id.get(call_id, [])
                if _result_proves_state_change(result)
            ]
            call_evidence_ids = set().union(
                *(_evidence_ids(result) for result in call_results)
            ) if call_results else set()
            if not call_evidence_ids:
                continue
            for effect in agent_caused_records:
                outcome_event_id = str(effect.get("event_id") or "")
                effect_tick = _effect_tick(effect)
                changed_state_fields = [
                    str(field)
                    for field in (
                        effect.get("changed_state_fields")
                        if isinstance(effect.get("changed_state_fields"), list)
                        else []
                    )
                    if str(field)
                ]
                before_digest = str(effect.get("before_state_digest") or "")
                after_digest = str(effect.get("after_state_digest") or "")
                effect_evidence_ids = _evidence_ids(effect)
                if (
                    str(effect.get("call_id") or "") != call_id
                    or not outcome_event_id
                    or effect_tick is None
                    or effect_tick < decision_tick
                    or not changed_state_fields
                    or not before_digest
                    or not after_digest
                    or before_digest == after_digest
                    or not effect_evidence_ids.intersection(call_evidence_ids)
                    or not _action_outcome_edge_matches(
                        effect.get("action_to_outcome_edge"),
                        call_id=call_id,
                        outcome_event_id=outcome_event_id,
                    )
                ):
                    continue
                identity = (event_id, decision_tick, call_id, outcome_event_id)
                if identity in seen_event_action_edges:
                    continue
                seen_event_action_edges.add(identity)
                event_to_action_edges.append(
                    {
                        "source_event_id": event_id,
                        "source_event_tick": event_tick,
                        "event_evidence_ids": sorted(event_evidence_ids),
                        "matched_event_evidence_ids": sorted(
                            matched_event_evidence_ids
                        ),
                        "decision_tick": decision_tick,
                        "call_id": call_id,
                        "tool_name": str(decision["tool_name"]),
                        "call_evidence_ids": sorted(call_evidence_ids),
                        "outcome_event_id": outcome_event_id,
                        "effect_tick": effect_tick,
                        "effect_evidence_ids": sorted(effect_evidence_ids),
                        "changed_state_fields": changed_state_fields,
                        "kind": "event_to_post_event_action_outcome",
                    }
                )
    event_to_action_edges = validated_event_action_edges(
        event_to_action_edges,
        material_exogenous_records=material_exogenous_records,
        agent_caused_records=agent_caused_records,
    )
    return {
        "schema_version": TRAJECTORY_ANALYSIS_CONTRACT_VERSION,
        "observed_successful_tool_set": sorted(successful),
        "observed_successful_distinct_tool_count": len(successful),
        "observed_state_changing_tool_set": sorted(state_changing),
        "observed_state_changing_distinct_tool_count": len(state_changing),
        "observed_physical_actuator_endpoint_set": sorted(
            physical_actuator_endpoints
        ),
        "observed_physical_actuator_endpoint_count": len(
            physical_actuator_endpoints
        ),
        "state_change_ticks": sorted(state_change_ticks),
        "information_gathering_ticks": sorted(information_ticks),
        "model_decision_ticks": sorted(model_decision_ticks),
        "environment_change_ticks": sorted(environment_ticks),
        "material_exogenous_event_records": material_exogenous_records,
        "source_scheduled_event_records": source_scheduled_records,
        "declared_perturbation_event_records": perturbation_records,
        "endogenous_completion_event_records": endogenous_records,
        "agent_caused_event_records": agent_caused_records,
        "post_change_decision_ticks": post_change_decision_ticks,
        "event_to_decision_action_edges": event_to_decision_edges,
        "event_to_action_edges": event_to_action_edges,
        "adaptive_replanning_observed": bool(event_to_action_edges),
        "agent_action_backend_effect_observed": bool(
            agent_caused_records
            and state_changing
        ),
        "effective_control_ticks": effective_ticks,
        "n_effective_control_ticks": len(effective_ticks),
        "effective_control_tick_status": effective_status,
        "explicit_plan_revision_count": plan_revisions,
        "control_strategy_switch_count": _count_switches(control_sequence),
        "observed_tool_phase_depth_proxy": len(phase_sequence),
        "observed_tool_phase_sequence": phase_sequence,
        "actual_interaction_turns": interaction_turns,
        "simulator_ticks": simulator_ticks,
        "runner_autonomy_ticks": runner_autonomy_ticks,
        "shortest_successful_tool_set": None,
        "required_distinct_tool_count": None,
        "exact_dependency_depth": exact_depth,
        "dependency_depth_status": dependency_status,
        "dependency_metadata_coverage": (
            len(dependency_nodes) / non_meta_call_count if non_meta_call_count else 0.0
        ),
        "evidence_action_graph": {
            "nodes": [
                graph_nodes[node_id] for node_id in sorted(graph_nodes)
            ],
            "edges": [
                {"source": source, "target": target, "kind": kind}
                for source, target, kind in sorted(graph_edges)
            ],
        },
        "minimality_status": "requires_deterministic_replay",
    }


def minimize_successful_action_sequence(
    actions: list[Action],
    succeeds: Callable[[list[Action]], bool],
    *,
    max_replays: int = 100,
) -> MinimizationResult:
    """Find a deterministic 1-minimal successful trace by tool and call ablation.

    The result is locally irreducible: removing any one remaining call breaks
    the supplied success predicate.  It is not claimed to be the global
    shortest trace unless the caller separately performs exhaustive search.
    """
    retained = {
        (action_index, call_index)
        for action_index, action in enumerate(actions)
        for call_index, call in enumerate(action.tool_calls)
        if call.name not in META_TOOLS
    }
    n_replays = 1
    if not succeeds(_materialize_actions(actions, retained)):
        return MinimizationResult(
            actions=_materialize_actions(actions, retained),
            tool_set=sorted(_retained_tool_names(actions, retained)),
            distinct_tool_count=len(_retained_tool_names(actions, retained)),
            n_replays=n_replays,
            status="initial_trace_not_successful",
        )

    for tool_name in sorted(_retained_tool_names(actions, retained)):
        if n_replays >= max_replays:
            break
        candidate = {
            key
            for key in retained
            if actions[key[0]].tool_calls[key[1]].name != tool_name
        }
        n_replays += 1
        if succeeds(_materialize_actions(actions, candidate)):
            retained = candidate

    changed = True
    while changed and n_replays < max_replays:
        changed = False
        for key in sorted(retained):
            if n_replays >= max_replays:
                break
            candidate = retained - {key}
            n_replays += 1
            if succeeds(_materialize_actions(actions, candidate)):
                retained = candidate
                changed = True

    names = sorted(_retained_tool_names(actions, retained))
    return MinimizationResult(
        actions=_materialize_actions(actions, retained),
        tool_set=names,
        distinct_tool_count=len(names),
        n_replays=n_replays,
        status="one_minimal" if n_replays < max_replays else "replay_budget_exhausted",
    )


def exhaustive_trace_subset_minimum(
    actions: list[Action],
    succeeds: Callable[[list[Action]], bool],
    *,
    max_calls: int = 10,
    max_replays: int = 2048,
) -> MinimizationResult | None:
    """Prove the shortest successful subset of one observed trace.

    This cannot discover a different policy or tool. ``None`` means the trace
    was too long or the replay budget ended before a proof was obtained.
    """
    retained = sorted(
        (action_index, call_index)
        for action_index, action in enumerate(actions)
        for call_index, call in enumerate(action.tool_calls)
        if call.name not in META_TOOLS
    )
    if len(retained) > max_calls:
        return None

    n_replays = 0
    for size in range(len(retained) + 1):
        for selected in combinations(retained, size):
            if n_replays >= max_replays:
                return None
            n_replays += 1
            selected_set = set(selected)
            if succeeds(_materialize_actions(actions, selected_set)):
                names = sorted(_retained_tool_names(actions, selected_set))
                return MinimizationResult(
                    actions=_materialize_actions(actions, selected_set),
                    tool_set=names,
                    distinct_tool_count=len(names),
                    n_replays=n_replays,
                    status="trace_subset_global_minimum",
                )
    return MinimizationResult(
        actions=_materialize_actions(actions, set()),
        tool_set=[],
        distinct_tool_count=0,
        n_replays=n_replays,
        status="no_successful_trace_subset",
    )


def _matching_result(
    results: list[Any], call_index: int, tool_name: str
) -> dict[str, Any] | None:
    if call_index < len(results) and isinstance(results[call_index], dict):
        candidate = results[call_index]
        if str(candidate.get("name") or "") == tool_name:
            return candidate
    return next(
        (
            result
            for result in results
            if isinstance(result, dict) and str(result.get("name") or "") == tool_name
        ),
        None,
    )


def _append_phase(phases: list[str], phase: str) -> None:
    if not phases or phases[-1] != phase:
        phases.append(phase)


def _count_switches(sequence: list[str]) -> int:
    return sum(
        left != right
        for left, right in zip(sequence, sequence[1:], strict=False)
    )


def _control_strategy_token(
    result: dict[str, Any], name: str
) -> str | None:
    """Return a token that reflects the native control policy, not just its tool.

    Traffic phase-duration and CityLearn storage supervision each use one
    state-changing tool with different native policies. Treating every call as
    the same strategy hides real reversals. CityLearn direction is credited
    only when the successful result is linked to an observed physical effect.
    """
    if name == "set_storage_dispatch":
        payload = result.get("payload")
        if not isinstance(payload, dict):
            return None
        policy = str(payload.get("native_control_policy") or "")
        endpoint = str(payload.get("physical_actuator_id") or "")
        try:
            signed_value = float(payload.get("signed_control_value"))
        except (TypeError, ValueError):
            return None
        effect_proven = bool(
            result.get("evidence_id")
            and result.get("produces_evidence_ids")
            and _finite_int_tick(result.get("effect_tick")) is not None
        )
        direction_consistent = (
            policy == "charge" and signed_value > 0.0
        ) or (
            policy == "discharge" and signed_value < 0.0
        )
        if (
            not effect_proven
            or not endpoint
            or not math.isfinite(signed_value)
            or not direction_consistent
        ):
            return None
        return f"{name}:{policy}"
    if name != "set_signal_phase_duration":
        return name
    payload = result.get("payload")
    if not isinstance(payload, dict):
        payload = result
    raw_duration = payload.get("sumo_phase_duration_s")
    if raw_duration is None:
        after = payload.get("after_runtime_state")
        if isinstance(after, dict):
            raw_duration = after.get("remaining_phase_duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        return name
    if not math.isfinite(duration):
        return name
    return f"{name}:{duration:.6g}"


def _physical_actuator_endpoint_token(
    result: dict[str, Any], name: str
) -> str | None:
    """Return a runtime-proven actuator identity when the backend exposes one.

    Configuration declarations are deliberately ignored.  SUMO currently
    emits ``sumo_tls_id`` in the state-changing result payload; the generic
    ``physical_actuator_id`` fallback supports other native backends without
    changing their historical name-only accounting.
    """
    payload = result.get("payload")
    if not isinstance(payload, dict):
        return None
    endpoint = payload.get("physical_actuator_id")
    if endpoint is None:
        endpoint = payload.get("sumo_tls_id") or payload.get("tls_id")
    if endpoint is None:
        return None
    endpoint = str(endpoint).strip()
    if not endpoint:
        return None
    return f"{name}|{endpoint}"


def _dependency_depth(nodes: dict[str, set[str]]) -> int:
    memo: dict[str, int] = {}
    visiting: set[str] = set()

    def depth(node: str) -> int:
        if node in memo:
            return memo[node]
        if node in visiting:
            raise ValueError("tool dependency graph contains a cycle")
        visiting.add(node)
        parents = {parent for parent in nodes.get(node, set()) if parent in nodes}
        value = 1 + max((depth(parent) for parent in parents), default=0)
        visiting.remove(node)
        memo[node] = value
        return value

    return max((depth(node) for node in nodes), default=0)


def _trajectory_tick_for_action_index(action_index: int, step_ticks: list[int]) -> int:
    """Map zero-based counterfactual action positions to logged tick labels."""
    if 0 <= action_index < len(step_ticks):
        return step_ticks[action_index]
    return action_index


def _retained_tool_names(
    actions: list[Action], retained: set[tuple[int, int]]
) -> set[str]:
    return {actions[i].tool_calls[j].name for i, j in retained}


def _materialize_actions(
    actions: list[Action], retained: set[tuple[int, int]]
) -> list[Action]:
    materialized: list[Action] = []
    for action_index, action in enumerate(actions):
        calls = [
            ToolCall(
                name=call.name,
                args=dict(call.args),
                idempotency_key=call.idempotency_key,
                rationale=call.rationale,
                call_id=call.call_id,
                consumes_evidence_ids=call.consumes_evidence_ids,
                depends_on_call_ids=call.depends_on_call_ids,
            )
            for call_index, call in enumerate(action.tool_calls)
            if (action_index, call_index) in retained
        ]
        if not calls:
            calls = [ToolCall(name="wait")]
        materialized.append(
            Action(
                tool_calls=calls,
                dominant=calls[0].name,
                assistant_text=action.assistant_text,
                rationale=action.rationale,
            )
        )
    return materialized
