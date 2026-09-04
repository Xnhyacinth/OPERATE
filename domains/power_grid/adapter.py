"""
domains.power_grid.adapter — Power-grid POMDP environment.

Wires together:

- A backend (``pglib_uc_synthetic`` or ``grid2op``)
- ``core.FogOfWarPolicy`` for partial observability + investigation reveal
- ``core.BeliefStateTracker`` (optional, scoring side-channel)
- ``core.ToolRegistry`` populated from ``domains.power_grid.native_tools``
- ``core.StakeholderTrustManager`` populated from
  ``domains.power_grid.native_stakeholders``
- ``core.EthicalDilemmaManager`` armed from the seed's ``dilemmas``
- ``core.EvidenceLogger`` to make every score auditable
- ``core.CascadeBus`` publication on grid faults (v0.2 will plug subscribers)

Implements ``core.POMDPEnvironment`` so the runner + counterfactual replay
treat it uniformly with future domains.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from contextlib import suppress
from typing import Any

from core import (
    Action,
    BeliefStateTracker,
    CascadeBus,
    CascadeEvent,
    EthicalDilemmaManager,
    EvidenceLogger,
    FogOfWarPolicy,
    HideRule,
    NoiseRule,
    POMDPEnvironment,
    StakeholderTrustManager,
    StalenessRule,
    StepInfo,
    StepReturn,
    TickBudget,
    ToolCall,
    ToolContext,
    ToolRegistry,
    ToolResult,
    arm_dilemmas,
    safe_dataclass_to_dict,
)
from core.difficulty_levels import canonical_difficulty_level
from core.evidence import control_summary_from_evidence
from core.world_evolution_contract import canonicalize_runtime_events
from domains.registry import apply_supervisory_cadence

from .backends.pglib_uc_synthetic import PglibUcSyntheticBackend
from .seeds.schema import DilemmaSeed, ScenarioSeed

_control_summary_from_evidence = control_summary_from_evidence

_NON_EFFECT_TOOL_RESULT_STATUSES = {
    "canceled",
    "cancelled",
    "error",
    "expired",
    "failed",
    "pending",
    "rejected",
    "superseded",
}


def _tool_result_status_blocks_effect(
    result: Any,
    *,
    extra_statuses: set[str] | None = None,
) -> bool:
    payload = getattr(result, "payload", None)
    if not isinstance(payload, dict):
        return True
    raw_status = payload.get("_status")
    if not isinstance(raw_status, str):
        return False
    tokens = {
        token
        for token in raw_status.strip().lower().replace("-", "_").split("_")
        if token
    }
    return bool(
        tokens.intersection(
            _NON_EFFECT_TOOL_RESULT_STATUSES.union(extra_statuses or set())
        )
    )


def _state_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _bind_delayed_mutual_aid_effects(
    *,
    backend_events: list[dict[str, Any]],
    tool_results: list[Any],
    calls_by_id: dict[str, Any],
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    applied_tick: int,
) -> list[dict[str, Any]]:
    """Bind native reserve arrivals to exact materialized tool calls."""
    arrivals = [
        event
        for event in backend_events
        if (event.get("kind") or event.get("type")) == "mutual_aid_arrived"
        and not event.get("call_id")
    ]
    materialized = [
        result
        for result in tool_results
        if result.name == "request_mutual_aid"
        and result.ok
        and result.state_changing
        and str((result.payload or {}).get("_status") or "") == "materialized"
        and str(result.call_id or "") in calls_by_id
    ]
    before_digest = _state_digest(before_state)
    after_digest = _state_digest(after_state)
    if not arrivals or not materialized or before_digest == after_digest:
        return backend_events

    retained = [event for event in backend_events if event not in arrivals]
    template = arrivals[0]
    for result in materialized:
        call_id = str(result.call_id)
        call = calls_by_id[call_id]
        event_id = f"mutual_aid_arrived:{call_id}:{applied_tick}"
        retained.append(
            {
                **template,
                "event_id": event_id,
                "origin": "agent_caused",
                "agent_caused": True,
                "event_class": "agent_outcome",
                "decision_required": False,
                "actionable": False,
                "mw": float(call.args.get("mw", 0.0) or 0.0),
                "call_id": call_id,
                "tool_name": "request_mutual_aid",
                "requested_action": {
                    "name": "request_mutual_aid",
                    "args": dict(call.args),
                },
                "before_state_digest": before_digest,
                "after_state_digest": after_digest,
                "effect_tick": applied_tick,
                "outcome_tick": applied_tick,
            }
        )
    return retained


def _acopf_control_state_for_call(
    snapshot: dict[str, Any],
    *,
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any] | None:
    """Project only native AC-OPF control state used for causal evidence."""
    entities = snapshot.get("entities") or {}
    totals = snapshot.get("totals") or {}
    if not isinstance(entities, dict) or not isinstance(totals, dict):
        return None
    if tool_name == "shed_load":
        load_id = str(args.get("load_id") or "")
        load = entities.get(load_id)
        if not isinstance(load, dict):
            return None
        return {
            "load_id": load_id,
            "current_demand_mw": load.get("current_demand_mw"),
            "cumulative_shed_mwh": load.get("cumulative_shed_mwh"),
            "aggregate_demand_mw": totals.get("aggregate_demand_mw"),
        }
    if tool_name == "commit_reserve":
        reserve = entities.get("reserve_commitment")
        if not isinstance(reserve, dict):
            return None
        return {
            "committed_reserve_mw": reserve.get("committed_reserve_mw"),
            "pending_reserve_mw": reserve.get("pending_reserve_mw"),
            "reserves_procured_mw": totals.get("reserves_procured_mw"),
        }
    if tool_name == "redispatch_generation":
        generator_id = str(args.get("generator_id") or "")
        if generator_id not in entities:
            try:
                generator_id = f"gen_{int(args.get('generator_index', 0))}"
            except (TypeError, ValueError):
                return None
        generator = entities.get(generator_id)
        if not isinstance(generator, dict):
            return None
        return {
            "generator_id": generator_id,
            "commanded_target_mw": generator.get("commanded_target_mw"),
            "actual_dispatch_mw": generator.get("actual_dispatch_mw"),
        }
    return None


def _native_acopf_action_effect_events(
    *,
    tool_results: list[Any],
    calls_by_id: dict[str, Any],
    before_control_state: dict[str, Any],
    applied_control_state: dict[str, Any],
    post_tick_control_state: dict[str, Any],
    applied_tick: int,
) -> list[dict[str, Any]]:
    """Emit causal edges only when a native AC-OPF control persists post-solve."""
    effects: list[dict[str, Any]] = []
    for result in tool_results:
        if (
            not getattr(result, "ok", False)
            or not getattr(result, "state_changing", False)
            or not isinstance(getattr(result, "payload", None), dict)
        ):
            continue
        if _tool_result_status_blocks_effect(result):
            continue
        call_id = str(getattr(result, "call_id", "") or "")
        call = calls_by_id.get(call_id)
        tool_name = str(getattr(result, "name", ""))
        if call is None or tool_name not in {
            "shed_load",
            "commit_reserve",
            "redispatch_generation",
        }:
            continue
        before = _acopf_control_state_for_call(
            before_control_state,
            tool_name=tool_name,
            args=dict(call.args),
        )
        applied = _acopf_control_state_for_call(
            applied_control_state,
            tool_name=tool_name,
            args=dict(call.args),
        )
        post = _acopf_control_state_for_call(
            post_tick_control_state,
            tool_name=tool_name,
            args=dict(call.args),
        )
        if before is None or applied is None or post is None or before == applied:
            continue
        if tool_name == "shed_load":
            before_shed = float(before.get("cumulative_shed_mwh") or 0.0)
            applied_shed = float(applied.get("cumulative_shed_mwh") or 0.0)
            post_shed = float(post.get("cumulative_shed_mwh") or 0.0)
            physical_effect = applied_shed > before_shed
            command_persisted = post_shed >= applied_shed
            changed_state_fields = [
                "load_current_demand_mw",
                "load_cumulative_shed_mwh",
            ]
            event_type = "acopf_load_shed_applied"
        elif tool_name == "commit_reserve":
            command_persisted = applied.get("committed_reserve_mw") == post.get(
                "committed_reserve_mw"
            )
            physical_effect = post.get("reserves_procured_mw") != before.get(
                "reserves_procured_mw"
            )
            changed_state_fields = [
                "committed_reserve_mw",
                "reserves_procured_mw",
            ]
            event_type = "acopf_reserve_commitment_applied"
        else:
            command_persisted = applied.get("commanded_target_mw") == post.get(
                "commanded_target_mw"
            )
            physical_effect = post.get("actual_dispatch_mw") != before.get(
                "actual_dispatch_mw"
            )
            changed_state_fields = [
                "generator_commanded_target_mw",
                "generator_actual_dispatch_mw",
            ]
            event_type = "acopf_redispatch_applied"
        if not command_persisted or not physical_effect:
            continue
        event_id = f"acopf-agent-control:{call_id}:{applied_tick}"
        evidence_ids = [
            evidence_id
            for evidence_id in [getattr(result, "evidence_id", None)]
            if isinstance(evidence_id, str) and evidence_id
        ]
        effects.append(
            {
                "event_id": event_id,
                "type": event_type,
                "origin": "agent_caused",
                "agent_caused": True,
                "tick": applied_tick,
                "actionable": False,
                "decision_required": False,
                "changed_state_fields": changed_state_fields,
                "call_id": call_id,
                "tool_name": tool_name,
                "requested_action": {
                    "name": str(call.name),
                    "args": dict(call.args),
                },
                "applied_action": {
                    "control_state_before_tool": before,
                    "control_state_after_tool": applied,
                    "post_tick_control_state": post,
                },
                "before_state_digest": _state_digest(before),
                "after_state_digest": _state_digest(
                    {
                        "control_state_after_tool": applied,
                        "post_tick_control_state": post,
                    }
                ),
                "effect_tick": applied_tick + 1,
                "outcome_tick": applied_tick + 1,
                "evidence_ids": evidence_ids,
                "action_to_outcome_edge": {
                    "source": f"call:{call_id}",
                    "target": f"outcome:{event_id}",
                    "kind": "action_to_outcome",
                },
            }
        )
        result.effect_tick = applied_tick + 1
    return effects


def _opendss_fresh_control_state_for_call(
    snapshot: dict[str, Any],
    *,
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any] | None:
    """Project a native OpenDSS control and its physical observables."""
    if tool_name == "set_transformer_tap":
        target = int(args.get("trafo_id", -1))
        controls = snapshot.get("regcontrols") or []
        match = next(
            (
                row
                for row in controls
                if isinstance(row, dict) and int(row.get("trafo_id", -1)) == target
            ),
            None,
        )
        if not isinstance(match, dict):
            return None
        control = {
            "trafo_id": target,
            "regcontrol": match.get("name"),
            "tap_number": match.get("tap_number"),
        }
    elif tool_name == "switch_capacitor":
        target = int(args.get("cap_id", -1))
        controls = snapshot.get("capacitors") or []
        match = next(
            (
                row
                for row in controls
                if isinstance(row, dict) and int(row.get("cap_id", -1)) == target
            ),
            None,
        )
        if not isinstance(match, dict):
            return None
        control = {
            "cap_id": target,
            "capacitor": match.get("name"),
            "states": list(match.get("states") or []),
        }
    elif tool_name == "switch_branch":
        target = int(args.get("line_index", -1))
        controls = snapshot.get("lines") or []
        match = next(
            (
                row
                for row in controls
                if isinstance(row, dict) and int(row.get("line_index", -1)) == target
            ),
            None,
        )
        if not isinstance(match, dict):
            return None
        control = {
            "line_index": target,
            "line": match.get("name"),
            "in_service": match.get("in_service"),
        }
    else:
        return None
    return {
        "control": control,
        "native_observables": {
            key: snapshot.get(key)
            for key in (
                "voltage_min_pu",
                "voltage_max_pu",
                "n_voltage_violations",
                "line_current_max_a",
            )
        },
    }


def _native_opendss_fresh_action_effect_events(
    *,
    tool_results: list[Any],
    calls_by_id: dict[str, Any],
    before_control_state: dict[str, Any],
    applied_control_state: dict[str, Any],
    post_tick_control_state: dict[str, Any],
    applied_tick: int,
    event_id_prefix: str = "opendss-fresh",
) -> list[dict[str, Any]]:
    """Link applied OpenDSS controls to observed native state changes."""
    effects: list[dict[str, Any]] = []
    for result in tool_results:
        if (
            not getattr(result, "ok", False)
            or not getattr(result, "state_changing", False)
            or not isinstance(getattr(result, "payload", None), dict)
        ):
            continue
        if _tool_result_status_blocks_effect(result):
            continue
        call_id = str(getattr(result, "call_id", "") or "")
        call = calls_by_id.get(call_id)
        tool_name = str(getattr(result, "name", ""))
        if call is None or tool_name not in {
            "set_transformer_tap",
            "switch_capacitor",
            "switch_branch",
        }:
            continue
        before = _opendss_fresh_control_state_for_call(
            before_control_state,
            tool_name=tool_name,
            args=dict(call.args),
        )
        applied = _opendss_fresh_control_state_for_call(
            applied_control_state,
            tool_name=tool_name,
            args=dict(call.args),
        )
        post = _opendss_fresh_control_state_for_call(
            post_tick_control_state,
            tool_name=tool_name,
            args=dict(call.args),
        )
        if (
            before is None
            or applied is None
            or post is None
            or before["control"] == applied["control"]
            or before["native_observables"] == applied["native_observables"]
            or applied["control"] != post["control"]
            or before["native_observables"] == post["native_observables"]
        ):
            continue
        control_field = {
            "set_transformer_tap": "regcontrol_tap_number",
            "switch_capacitor": "capacitor_states",
            "switch_branch": "line_in_service",
        }[tool_name]
        changed_observables = [
            name
            for name, value in applied["native_observables"].items()
            if value != before["native_observables"].get(name)
        ]
        type_prefix = event_id_prefix.replace("-", "_")
        event_id = f"{event_id_prefix}-agent-control:{call_id}:{applied_tick}"
        evidence_ids = [
            evidence_id
            for evidence_id in [getattr(result, "evidence_id", None)]
            if isinstance(evidence_id, str) and evidence_id
        ]
        effects.append(
            {
                "event_id": event_id,
                "type": f"{type_prefix}_{tool_name}_applied",
                "origin": "agent_caused",
                "agent_caused": True,
                "tick": applied_tick,
                "actionable": False,
                "decision_required": False,
                "changed_state_fields": [control_field, *changed_observables],
                "call_id": call_id,
                "tool_name": tool_name,
                "requested_action": {
                    "name": str(call.name),
                    "args": dict(call.args),
                },
                "applied_action": {
                    "control_state_after_tool": applied["control"],
                    "native_state_after_tool": applied["native_observables"],
                    "post_tick_native_state": post["native_observables"],
                    "control_persisted_to_next_tick": True,
                },
                "before_state_digest": _state_digest(before),
                "after_state_digest": _state_digest(
                    {
                        "control_state_after_tool": applied["control"],
                        "native_state_after_tool": applied["native_observables"],
                        "post_tick_native_state": post["native_observables"],
                    }
                ),
                "effect_tick": applied_tick + 1,
                "outcome_tick": applied_tick + 1,
                "evidence_ids": evidence_ids,
                "action_to_outcome_edge": {
                    "source": f"call:{call_id}",
                    "target": f"outcome:{event_id}",
                    "kind": "action_to_outcome",
                },
            }
        )
        result.effect_tick = applied_tick + 1
    return effects


def _cigre_control_state_for_call(
    snapshot: dict[str, Any],
    *,
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any] | None:
    """Project a CIGRE native command and its post-solve observables."""
    entities = snapshot.get("entities") or {}
    totals = snapshot.get("totals") or {}
    if not isinstance(entities, dict) or not isinstance(totals, dict):
        return None

    control: dict[str, Any] | None = None
    if tool_name == "switch_capacitor":
        cap_id = int(args.get("cap_id", args.get("capacitor_id", -1)))
        control = entities.get(f"capacitor_{cap_id}")
        if not isinstance(control, dict):
            return None
        control = {
            "cap_id": cap_id,
            "on": control.get("on"),
            "q_mvar": control.get("q_mvar"),
        }
    elif tool_name == "set_transformer_tap":
        trafo_id = int(args.get("trafo_id", args.get("trafo_index", -1)))
        row = entities.get(f"trafo_{trafo_id}")
        if not isinstance(row, dict):
            return None
        control = {"trafo_id": trafo_id, "tap_pos": row.get("tap_pos")}
    elif tool_name == "set_der_reactive_power":
        der_id = int(args.get("der_id", args.get("sgen_index", -1)))
        row = entities.get(f"sgen_{der_id}")
        if not isinstance(row, dict):
            return None
        control = {"der_id": der_id, "q_mvar": row.get("q_mvar")}
    elif tool_name == "set_battery_dispatch":
        storage_id = int(args.get("storage_id", args.get("storage_index", -1)))
        row = entities.get(f"storage_{storage_id}")
        if not isinstance(row, dict):
            return None
        control = {"storage_id": storage_id, "p_mw": row.get("p_mw")}
    elif tool_name == "switch_branch":
        line_id = int(args.get("line_index", -1))
        row = entities.get(f"line_{line_id}")
        if not isinstance(row, dict):
            return None
        control = {"line_index": line_id, "in_service": row.get("in_service")}
    elif tool_name == "redispatch_generation":
        try:
            der_id = int(args.get("generator_index", args.get("der_id", -1)))
        except (TypeError, ValueError):
            return None
        row = entities.get(f"sgen_{der_id}")
        if not isinstance(row, dict):
            return None
        control = {
            "generator_index": der_id,
            "effective_cap_mw": row.get("effective_cap_mw"),
            "output_mw": row.get("output_mw"),
        }
    elif tool_name == "shed_load":
        load_id = str(args.get("load_id") or "")
        row = entities.get(load_id)
        if not isinstance(row, dict):
            return None
        control = {
            "load_id": load_id,
            "current_demand_mw": row.get("current_demand_mw"),
            "cumulative_shed_mwh": row.get("cumulative_shed_mwh"),
        }
    elif tool_name == "commit_reserve":
        control = {
            "pending_reserve_mw": totals.get("pending_reserve_mw"),
            "reserves_procured_mw": totals.get("reserves_procured_mw"),
        }
    else:
        return None

    native_observables = {
        key: totals.get(key)
        for key in (
            "demand_mw",
            "der_generation_mw",
            "reserves_procured_mw",
            "n_voltage_violations",
            "n_overloads",
            "n_disconnected_lines",
            "rho_max",
            "telemetry_confidence",
        )
    }
    return {"control": control, "native_observables": native_observables}


def _cigre_call_requests_control_change(
    *,
    tool_name: str,
    args: dict[str, Any],
    before_control: dict[str, Any],
) -> bool:
    """Reject idempotent setters before later same-target calls are joined."""
    requested_and_current: tuple[Any, Any] | None = None
    if tool_name == "switch_capacitor":
        requested_and_current = (bool(args.get("status", True)), before_control.get("on"))
    elif tool_name == "set_transformer_tap":
        requested_and_current = (
            int(args.get("tap_pos", 0)),
            before_control.get("tap_pos"),
        )
    elif tool_name == "set_der_reactive_power":
        requested_and_current = (
            float(args.get("q_mvar", 0.0)),
            before_control.get("q_mvar"),
        )
    elif tool_name == "set_battery_dispatch":
        requested_and_current = (
            float(args.get("p_mw", 0.0)),
            before_control.get("p_mw"),
        )
    elif tool_name == "switch_branch":
        requested_and_current = (
            bool(args.get("connect", True)),
            before_control.get("in_service"),
        )
    elif tool_name == "redispatch_generation":
        requested_and_current = (
            float(args.get("target_mw", 0.0)),
            before_control.get("effective_cap_mw"),
        )
    if requested_and_current is None or requested_and_current[1] is None:
        return True
    return requested_and_current[0] != requested_and_current[1]


def _native_cigre_action_effect_events(
    *,
    tool_results: list[Any],
    calls_by_id: dict[str, Any],
    before_control_state: dict[str, Any],
    applied_control_state: dict[str, Any],
    post_tick_control_state: dict[str, Any],
    applied_tick: int,
) -> list[dict[str, Any]]:
    """Link CIGRE commands to state observed after the next power-flow tick."""
    effects: list[dict[str, Any]] = []
    supported = {
        "switch_capacitor",
        "set_der_reactive_power",
        "set_transformer_tap",
        "set_battery_dispatch",
        "shed_load",
        "switch_branch",
        "redispatch_generation",
        "commit_reserve",
    }
    for result in tool_results:
        if (
            not getattr(result, "ok", False)
            or not getattr(result, "state_changing", False)
            or str(getattr(result, "name", "")) not in supported
            or not isinstance(getattr(result, "payload", None), dict)
        ):
            continue
        if _tool_result_status_blocks_effect(result, extra_statuses={"ack"}):
            continue
        call_id = str(getattr(result, "call_id", "") or "")
        call = calls_by_id.get(call_id)
        if call is None or not call_id:
            continue
        tool_name = str(getattr(result, "name", ""))
        before = _cigre_control_state_for_call(
            before_control_state, tool_name=tool_name, args=dict(call.args)
        )
        applied = _cigre_control_state_for_call(
            applied_control_state, tool_name=tool_name, args=dict(call.args)
        )
        post = _cigre_control_state_for_call(
            post_tick_control_state, tool_name=tool_name, args=dict(call.args)
        )
        if before is None or applied is None or post is None:
            continue
        if not _cigre_call_requests_control_change(
            tool_name=tool_name,
            args=dict(call.args),
            before_control=before["control"],
        ):
            continue

        # CIGRE queues Volt-Var setpoints and applies them immediately before
        # the next power-flow solve, so the pre-tick ``applied`` snapshot may
        # still show the old control.  Attribute the effect to the command
        # only when the post-tick native control reflects a change.
        control_changed = before["control"] != post["control"]
        # shed_load mutates the cumulative native quantity only at the tick;
        # commit_reserve exposes the pending reserve immediately and is
        # materialised in reserves_procured_mw after the solve.
        if tool_name == "shed_load":
            control_changed = float(
                post["control"].get("cumulative_shed_mwh") or 0.0
            ) > float(before["control"].get("cumulative_shed_mwh") or 0.0)
        elif tool_name == "commit_reserve":
            control_changed = float(
                post["control"].get("pending_reserve_mw") or 0.0
            ) > float(before["control"].get("pending_reserve_mw") or 0.0)
        if not control_changed:
            continue

        control_persisted = True
        if not control_persisted:
            continue
        changed_observables = [
            name
            for name, value in post["native_observables"].items()
            if value != before["native_observables"].get(name)
        ]
        control_field = {
            "switch_capacitor": "capacitor_in_service",
            "set_der_reactive_power": "der_reactive_power_mvar",
            "set_transformer_tap": "transformer_tap_pos",
            "set_battery_dispatch": "storage_dispatch_mw",
            "shed_load": "load_demand_mw",
            "switch_branch": "line_in_service",
            "redispatch_generation": "der_dispatch_cap_mw",
            "commit_reserve": "pending_reserve_mw",
        }[tool_name]
        event_id = f"cigre-agent-control:{call_id}:{applied_tick}"
        evidence_ids = [
            evidence_id
            for evidence_id in [getattr(result, "evidence_id", None)]
            if isinstance(evidence_id, str) and evidence_id
        ]
        event = {
            "event_id": event_id,
            "type": f"cigre_{tool_name}_applied",
            "origin": "agent_caused",
            "agent_caused": True,
            "tick": applied_tick,
            "actionable": False,
            "decision_required": False,
            "changed_state_fields": [control_field, *changed_observables],
            "call_id": call_id,
            "tool_name": tool_name,
            "requested_action": {"name": str(call.name), "args": dict(call.args)},
            "applied_action": {
                "control_state_after_tool": applied["control"],
                "post_tick_control_state": post["control"],
                "post_tick_native_state": post["native_observables"],
                "control_persisted_to_next_tick": control_persisted,
            },
            "before_state_digest": _state_digest(before),
            "after_state_digest": _state_digest(post),
            "effect_tick": applied_tick + 1,
            "outcome_tick": applied_tick + 1,
            "evidence_ids": evidence_ids,
            "action_to_outcome_edge": {
                "source": f"call:{call_id}",
                "target": f"outcome:{event_id}",
                "kind": "action_to_outcome",
            },
        }
        effects.append(event)
        result.effect_tick = applied_tick + 1
    return effects


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────


class PowerGridEnvironment(POMDPEnvironment):
    """OPERATE environment for the power-grid domain.

    Constructor takes no arguments. State is set up by ``reset()``, which
    expects ``scenario_config`` to be the dict serialization of a
    ``ScenarioSeed`` (so that counterfactual replay can pass the same JSON
    blob to a fresh env factory).
    """

    domain = "power_grid"

    def __init__(self, cascade_bus: CascadeBus | None = None) -> None:
        self._seed_obj: ScenarioSeed | None = None
        self._tick: int = 0
        self._horizon: int = 24
        self._backend: Any = None
        self._fog: FogOfWarPolicy | None = None
        self._belief: BeliefStateTracker | None = None
        self._tools: ToolRegistry | None = None
        self._stakeholders: StakeholderTrustManager | None = None
        self._dilemmas: EthicalDilemmaManager | None = None
        self._evidence: EvidenceLogger | None = None
        self._cascade_bus = cascade_bus or CascadeBus()
        self._episode_id: str = ""
        # D-04 wiring (v0.2.2): cumulative per-load shed snapshots taken
        # just before each dilemma fires, keyed by dilemma_id. Used by
        # ``dilemma_consequences.record_dilemma_consequences`` to scope
        # the harm-proxy window to the dilemma's resolution window.
        self._pre_dilemma_shed: dict[str, dict[str, float]] = {}
        self._consequences_recorded: bool = False
        # Phase 3 (Part D): cascade subscriptions — traffic.signal.failure
        # causes EV charger shedding at power-grid loads.
        self._cascade_bus.subscribe(
            "traffic.signal.failure", self._on_traffic_signal_failure
        )
        self._pending_cascade_perturbations: list[dict[str, Any]] = []
        # Retain the originating call while a delayed native control waits
        # for its physical execution tick.  A later acknowledgement alone
        # must never be mistaken for a backend state effect.
        self._pending_native_control_calls: dict[str, Any] = {}
        self._agency_source_events: list[dict[str, Any]] = []
        self._agency_parent_by_call: dict[str, dict[str, Any]] = {}
        self._staged_control: dict[str, Any] | None = None

    # ── POMDPEnvironment surface ────────────────────────────────────────

    def reset(self, scenario_config: dict[str, Any], seed: int) -> dict[str, Any]:
        seed_obj = _rebuild_seed_from_dict(scenario_config, override_seed=seed)
        self._seed_obj = seed_obj
        self._tick = 0
        self._horizon = seed_obj.horizon_ticks
        self._episode_id = f"pg_{seed_obj.signature()}_s{seed}"

        # Backend
        self._backend = _build_backend(seed_obj.backend_kind)
        self._backend.reset(seed_obj)

        # Fog
        self._fog = _build_fog(seed_obj)
        self._fog.reset(seed=seed)

        # Belief
        self._belief = BeliefStateTracker()
        self._belief.reset()

        # Tool registry
        self._tools = ToolRegistry(
            budget=_build_tick_budget(seed_obj),
            seed=seed,
            difficulty_level=seed_obj.difficulty_level,
        )
        self._tools.reset(seed=seed)
        _register_native_tools(self._tools, self._backend, self)

        # Stakeholders
        self._stakeholders = StakeholderTrustManager()
        _register_native_stakeholders(self._stakeholders, seed_obj)

        # Dilemmas
        self._dilemmas = EthicalDilemmaManager()
        self._dilemmas.reset()
        arm_dilemmas(self._dilemmas, seed_obj.dilemmas)

        # Evidence
        self._evidence = EvidenceLogger(episode_id=self._episode_id)

        # D-04 wiring: reset per-dilemma snapshot bookkeeping
        self._pre_dilemma_shed = {}
        self._consequences_recorded = False
        self._pending_cascade_perturbations = []
        self._pending_native_control_calls = {}
        self._agency_source_events = []
        self._agency_parent_by_call = {}
        self._staged_control = None

        obs = self.snapshot()
        self._belief.update_from_observation(obs, tick=0)
        return obs

    def step(self, action: Action) -> StepReturn:
        if self._staged_control is not None:
            raise RuntimeError("a staged control must be advanced before step")
        self.stage_control(action)
        return self.advance_staged_control()

    def supports_control_reconciliation(self) -> bool:
        return bool(
            self._seed_obj is not None
            and self._seed_obj.backend_kind == "pglib_uc_synthetic"
        )

    def stage_control(
        self, action: Action
    ) -> tuple[dict[str, Any], list[ToolResult]]:
        """Execute control calls and expose receipts before simulator advance."""
        assert self._tools is not None and self._backend is not None
        assert self._fog is not None and self._evidence is not None
        assert self._dilemmas is not None and self._stakeholders is not None
        assert self._belief is not None

        ctx = ToolContext(
            tick=self._tick,
            seed=int(self._seed_obj.seed if self._seed_obj else 0),
            backend=self._backend,
            extra={
                "fog": self._fog,
                "stakeholders": self._stakeholders,
                "dilemmas": self._dilemmas,
                "evidence": self._evidence,
                "cascade_bus": self._cascade_bus,
                "env": self,
            },
        )
        first_stage = self._staged_control is None
        if first_stage:
            before_control_state = self._backend.snapshot()
            investigation_stage_open = self._consume_within_tick_budget_state()
            tool_results = self._tools.execute_action(
                action,
                ctx,
                begin_tick=not investigation_stage_open,
            )
            self._staged_control = {
                "before_control_state": before_control_state,
                "tool_results": [],
                "calls_by_id": dict(self._pending_native_control_calls),
                "allow_same_tick_reveal": investigation_stage_open,
                "reconciliation_attempted": False,
            }
        else:
            assert self._staged_control is not None
            if self._staged_control["reconciliation_attempted"]:
                raise RuntimeError("only one control reconciliation is allowed per tick")
            self._staged_control["reconciliation_attempted"] = True
            tool_results = self._tools.execute_injected_failure_retry(action, ctx)

        assert self._staged_control is not None
        allow_same_tick_reveal = bool(
            self._staged_control["allow_same_tick_reveal"]
        )
        for call in action.tool_calls:
            spec = self._tools.get(call.name)
            if spec is None or not spec.state_changing or call.call_id is None:
                continue
            source_parent = self._agency_source_parent_for_call(
                call=call,
                request_tick=self._tick,
                allow_same_tick_reveal=allow_same_tick_reveal,
            )
            if source_parent is not None:
                self._agency_parent_by_call[str(call.call_id)] = source_parent
        calls_by_id = self._staged_control["calls_by_id"]
        calls_by_id.update(
            {
                str(call.call_id): call
                for call in action.tool_calls
                if call.call_id is not None
            }
        )
        for r in tool_results:
            call = calls_by_id.get(str(r.call_id or ""))
            r.evidence_id = self._evidence.log(
                kind="tool_call",
                tick=self._tick,
                payload={
                    "name": r.name,
                    "ok": r.ok,
                    "error_code": r.error_code,
                    "cost_units": r.cost_units,
                    "call_id": r.call_id,
                    "state_changing": r.state_changing,
                    "args": dict(call.args) if call is not None else {},
                    "payload": r.payload,
                    **self.tool_dependency_payload(action, r),
                },
                source="tool",
            )
            bind_tool_result = getattr(self._backend, "bind_tool_result", None)
            if r.ok and callable(bind_tool_result):
                bind_tool_result(
                    name=r.name,
                    call_id=r.call_id,
                    evidence_id=r.evidence_id,
                    payload=r.payload,
                )
            call_id = str(r.call_id or "")
            if not call_id:
                continue
            if isinstance(r.payload, dict) and r.payload.get("_status") == "pending":
                call = calls_by_id.get(call_id)
                if call is not None:
                    self._pending_native_control_calls[call_id] = call
            else:
                self._pending_native_control_calls.pop(call_id, None)

        self._staged_control["tool_results"].extend(tool_results)
        self._staged_control["applied_control_state"] = self._backend.snapshot()
        observation = self.snapshot()
        observation["tick"] = self._tick
        observation["__tool_budget__"] = self._tools.budget_status()
        return observation, list(tool_results)

    def advance_staged_control(self) -> StepReturn:
        """Advance the backend exactly once after the staged control epoch."""
        assert self._tools is not None and self._backend is not None
        assert self._fog is not None and self._evidence is not None
        assert self._dilemmas is not None and self._stakeholders is not None
        assert self._belief is not None
        if self._staged_control is None:
            raise RuntimeError("no staged control is available to advance")
        staged_control = self._staged_control
        self._staged_control = None
        before_control_state = staged_control["before_control_state"]
        applied_control_state = staged_control["applied_control_state"]
        tool_results = list(staged_control["tool_results"])
        calls_by_id = dict(staged_control["calls_by_id"])

        # Drain pending cascade perturbations (traffic -> power_grid, etc.)
        # before the backend ticks, so the cascaded effect lands this tick.
        if self._pending_cascade_perturbations:
            for p in self._pending_cascade_perturbations:
                try:
                    result = self._backend.apply_tool_effect("shed_load", p)
                    if result.get("_status") != "error" and self._evidence:
                        self._evidence.log(
                            "cascade_effect",
                            self._tick,
                            payload={**p, "result": result},
                            source="cascade_bus",
                        )
                except Exception:
                    pass
            self._pending_cascade_perturbations = []

        # Advance backend chronics + perturbations
        backend_tick_record = self._backend.tick(self._tick)
        backend_tick_payload = safe_dataclass_to_dict(
            backend_tick_record, dict_fallback=True
        )
        backend_events = list(getattr(backend_tick_record, "realized_events", []) or [])
        post_tick_control_state = self._backend.snapshot()
        backend_events = _bind_delayed_mutual_aid_effects(
            backend_events=backend_events,
            tool_results=tool_results,
            calls_by_id=calls_by_id,
            before_state=before_control_state,
            after_state=post_tick_control_state,
            applied_tick=self._tick,
        )
        native_action_effect_events: list[dict[str, Any]] = []
        if (
            self._seed_obj is not None
            and self._seed_obj.backend_kind == "pandapower_acopf"
        ):
            native_action_effect_events = _native_acopf_action_effect_events(
                tool_results=tool_results,
                calls_by_id=calls_by_id,
                before_control_state=before_control_state,
                applied_control_state=applied_control_state,
                post_tick_control_state=post_tick_control_state,
                applied_tick=self._tick,
            )
        elif self._seed_obj is not None and self._seed_obj.backend_kind in {
            "opendss_fresh_feeders",
            "opendss_ieee13",
        }:
            native_action_effect_events = _native_opendss_fresh_action_effect_events(
                tool_results=tool_results,
                calls_by_id=calls_by_id,
                before_control_state=before_control_state,
                applied_control_state=applied_control_state,
                post_tick_control_state=post_tick_control_state,
                applied_tick=self._tick,
                event_id_prefix=(
                    "opendss-ieee13"
                    if self._seed_obj.backend_kind == "opendss_ieee13"
                    else "opendss-fresh"
                ),
            )
        elif (
            self._seed_obj is not None
            and self._seed_obj.backend_kind == "cigre_distribution"
        ):
            native_action_effect_events = _native_cigre_action_effect_events(
                tool_results=tool_results,
                calls_by_id=calls_by_id,
                before_control_state=before_control_state,
                applied_control_state=applied_control_state,
                post_tick_control_state=post_tick_control_state,
                applied_tick=self._tick,
            )
        realized_events = [*backend_events, *native_action_effect_events]
        for event in realized_events:
            if str(event.get("origin") or "") == "agent_caused":
                event.setdefault("agent_caused", True)
                try:
                    outcome_tick = int(event.get("outcome_tick", self._tick))
                except (TypeError, ValueError):
                    outcome_tick = self._tick
                if outcome_tick < self._tick:
                    outcome_tick = self._tick
                event["effect_tick"] = outcome_tick
                event["outcome_tick"] = outcome_tick
                matching_results = [
                    result
                    for result in tool_results
                    if str(result.call_id or "") == str(event.get("call_id") or "")
                    and str(result.name or "") == str(event.get("tool_name") or "")
                    and result.ok
                    and result.state_changing
                    and not _tool_result_status_blocks_effect(result)
                ]
                if len(matching_results) == 1:
                    matching_results[0].effect_tick = outcome_tick
            call_id = str(event.get("call_id") or "")
            parent = self._agency_parent_by_call.get(call_id)
            if str(event.get("origin") or "") == "agent_caused" and parent is not None:
                event["causal_parent_event_id"] = parent["event_id"]
        world_evolution_records = canonicalize_runtime_events(
            realized_events,
            applied_tick=self._tick,
        )
        self._evidence.log(
            kind="backend_tick",
            tick=self._tick,
            payload=backend_tick_payload,
            source="engine",
        )
        self._evidence.log(
            kind="cost_summary",
            tick=self._tick,
            payload=_cost_summary_payload(self._backend, backend_tick_payload),
            source="engine",
        )
        for index, ev in enumerate(realized_events):
            evidence_id = self._evidence.log(
                kind="realized_event",
                tick=self._tick,
                payload=dict(ev),
                source="engine",
            )
            evidence_ids = list(ev.get("evidence_ids") or [])
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
            ev["evidence_ids"] = evidence_ids
            if index < len(world_evolution_records):
                world_evolution_records[index]["evidence_ids"] = list(evidence_ids)
                if world_evolution_records[index]["origin"] in {
                    "source_schedule",
                    "declared_perturbation",
                }:
                    self._agency_source_events.append(
                        {
                            "event_id": world_evolution_records[index]["event_id"],
                            "evidence_id": evidence_id,
                            "applied_tick": self._tick,
                            "hidden": world_evolution_records[index]["visibility"]
                            == "hidden",
                            "reveal_targets": self._agency_event_reveal_targets(ev),
                            "revealed_tick": None,
                            "reveal_evidence_ids": [],
                        }
                    )
            self._publish_cascade_for_event(ev, evidence_id)

        # Dilemma triggers
        current_snap = self.snapshot()
        triggered = self._dilemmas.maybe_trigger(self._tick, current_snap)
        for d in triggered:
            # D-04 wiring: snapshot per-load cumulative shed at the
            # moment the dilemma fires so the consequence proxy can
            # later scope to the WINDOW shed (delta), not the
            # cumulative-since-episode-start total.
            if hasattr(self._backend, "per_load_shed_mwh"):
                self._pre_dilemma_shed[d.dilemma_id] = dict(
                    self._backend.per_load_shed_mwh() or {}
                )
            self._evidence.log(
                kind="dilemma_triggered",
                tick=self._tick,
                payload={"dilemma_id": d.dilemma_id, "description": d.description},
                source="engine",
            )
        # Enforce dilemma deadlines: fire the default option on any
        # triggered-but-unresolved dilemma whose deadline has passed.
        # Makes the "default fires on missed deadline" contract in
        # docs/PROTOCOL.md actually true.
        defaults_fired = self._dilemmas.fire_defaults_for_missed_deadlines(self._tick)
        for choice in defaults_fired:
            self._evidence.log(
                kind="moral_choice",
                tick=self._tick,
                payload={
                    "dilemma_id": choice.dilemma_id,
                    "option_id": choice.chosen_option_id,
                    "rationale": choice.rationale,
                    "source": "default_option_fired",
                },
                source="engine",
            )

        # Stakeholder natural drift
        self._stakeholders.tick(self._tick)

        self._tick += 1
        backend_done = bool(getattr(backend_tick_record, "done", False))
        done = backend_done or self._tick >= self._horizon

        # D-04 wiring: at episode end, compute per-dilemma harm proxies
        # so ``EthicalDilemmaManager.score_consequences()`` is no longer
        # structurally zero. We do this exactly once (guarded by
        # ``_consequences_recorded``) so repeated calls to ``step()``
        # after ``done`` cannot inflate the recorded values.
        if done and not self._consequences_recorded and self._seed_obj is not None:
            from .dilemma_consequences import record_dilemma_consequences

            record_dilemma_consequences(
                seed_obj=self._seed_obj,
                dilemma_mgr=self._dilemmas,
                backend=self._backend,
                evidence=self._evidence,
                final_tick=self._tick,
                pre_dilemma_shed=self._pre_dilemma_shed,
            )
            self._consequences_recorded = True

        snap = self.snapshot()
        snap["tick"] = self._tick

        # Belief update from the observation returned for the next decision tick.
        self._belief.update_from_observation(snap, tick=self._tick)

        # Per-tick reward (informative only — official score is post-hoc)
        reward = self._reward_signal(backend_tick_record)

        info = StepInfo(
            realized_events=realized_events,
            evidence_ids=[
                i.evidence_id
                for i in self._evidence.items()
                if i.tick == self._tick - 1
            ],
            extra={
                "dilemmas_triggered": [d.dilemma_id for d in triggered],
                "stakeholder_trust": {
                    gid: r.trust for gid, r in self._stakeholders.snapshot().items()
                },
                "backend_tick_record": backend_tick_payload,
                "world_evolution_records": world_evolution_records,
            },
        )
        return StepReturn(
            observation=snap,
            tool_results=tool_results,
            reward=float(reward),
            done=done,
            info=info,
        )

    @staticmethod
    def _agency_event_reveal_targets(event: dict[str, Any]) -> set[str]:
        targets = {
            str(event[key])
            for key in (
                "asset_id",
                "generator_id",
                "line_id",
                "load_id",
                "substation_id",
                "target_id",
            )
            if event.get(key) not in (None, "")
        }
        declared_target = (event.get("declared_event") or {}).get("target")
        if isinstance(declared_target, dict):
            targets.update(
                str(value)
                for value in declared_target.values()
                if isinstance(value, (str, int))
            )
        return targets

    def _agency_source_parent_for_call(
        self,
        *,
        call: ToolCall,
        request_tick: int,
        allow_same_tick_reveal: bool = False,
    ) -> dict[str, Any] | None:
        consumed = {str(value) for value in call.consumes_evidence_ids or []}
        if not consumed:
            return None
        eligible = [
            event
            for event in self._agency_source_events
            if int(event["applied_tick"]) < request_tick
            and (
                (not event["hidden"] and str(event["evidence_id"]) in consumed)
                or (
                    event["hidden"]
                    and event["revealed_tick"] is not None
                    and (
                        int(event["revealed_tick"]) < request_tick
                        or (
                            allow_same_tick_reveal
                            and int(event["revealed_tick"]) == request_tick
                        )
                    )
                    and bool(consumed.intersection(event["reveal_evidence_ids"]))
                )
            )
        ]
        return eligible[-1] if eligible else None

    def mark_agency_source_events_revealed(
        self,
        *,
        target_id: str,
        reveal_tick: int,
        evidence_id: str | None = None,
    ) -> None:
        for event in self._agency_source_events:
            if event["hidden"] and target_id in event["reveal_targets"]:
                if event["revealed_tick"] is None:
                    event["revealed_tick"] = int(reveal_tick)
                if evidence_id and evidence_id not in event["reveal_evidence_ids"]:
                    event["reveal_evidence_ids"].append(evidence_id)

    def execute_investigation(
        self, action: Action
    ) -> tuple[dict[str, Any], list[ToolResult]]:
        observation, results = super().execute_investigation(action)
        calls_by_id = {call.call_id: call for call in action.tool_calls if call.call_id}
        for result in results:
            call = calls_by_id.get(result.call_id)
            if (
                result.ok
                and result.evidence_id
                and call is not None
                and call.name == "investigate_substation"
            ):
                self.mark_agency_source_events_revealed(
                    target_id=str(call.args.get("target_id", "")),
                    reveal_tick=self._tick,
                    evidence_id=str(result.evidence_id),
                )
        return observation, results

    def snapshot(self) -> dict[str, Any]:
        assert self._backend is not None
        raw = self._backend.snapshot()
        if self._fog:
            # Lock fog to current tick so repeated reads are byte-identical
            self._fog.set_tick(self._tick)
            raw = self._fog.filter(raw)
        backend_kind = str(getattr(self._backend, "backend_kind", "") or "")
        if backend_kind:
            with suppress(KeyError):
                raw["decision_cadence"] = apply_supervisory_cadence(
                    backend_kind,
                    raw.get("decision_cadence"),
                )
        # Augment with stakeholder + dilemma summary (these are intentionally
        # visible — they are the agent's social context).
        if self._stakeholders:
            raw["stakeholder_trust"] = {
                gid: {"trust": r.trust, "tier": r.tier}
                for gid, r in self._stakeholders.snapshot().items()
            }
        if self._dilemmas:
            raw["active_dilemmas"] = [
                {
                    "dilemma_id": d.dilemma_id,
                    "description": d.description,
                    "options": [
                        {"option_id": o.option_id, "label": o.label, "fatal": o.fatal}
                        for o in d.options
                    ],
                    "deadline_tick": d.trigger_tick + d.resolution_deadline_ticks,
                }
                for d in self._dilemmas.record.dilemmas_triggered
                if d.dilemma_id not in self._dilemmas.record.choices
            ]
        return raw

    def ground_truth(self) -> dict[str, Any]:
        """Full state, including hidden perturbations and per-load shed."""
        assert self._backend is not None and self._seed_obj is not None
        gt = self._backend.snapshot()
        gt["per_load_shed_mwh"] = self._backend.per_load_shed_mwh()
        gt["cost_components"] = self._backend.ground_truth_costs()
        # v0.6 (scoped, frozen-safe): a per-cell opt-in flag on the oracle-only
        # ground truth (NOT the agent-facing snapshot). When a scenario sets
        # ``backend_config.oracle_opf_inert``, the offline oracle mirrors
        # ``wait_only`` exactly. This is used for structurally OPF-saturated
        # AC-OPF cells where ``wait_only`` already solves a full per-tick
        # ``pp.runopp`` optimum the oracle cannot beat with dispatch/shed/commit
        # levers (without it, those levers PIN setpoints and make the oracle
        # WORSE than wait — a failure no frozen release ships). Frozen seeds
        # never carry this key, so their oracle behaviour is byte-identical.
        if bool(self._seed_obj.backend_config.get("oracle_opf_inert", False)):
            gt["oracle_opf_inert"] = True
        if self._evidence and (
            self._seed_obj.backend_config.get("task_requirements")
            or (
                self._seed_obj.backend_config.get("decision_axis")
                and self._seed_obj.backend_config.get("control_action_probe")
            )
        ):
            gt["control_summary"] = _control_summary_from_evidence(
                self._evidence,
                include_lifecycle=True,
            )
            requirements = self._seed_obj.backend_config.get("task_requirements") or {}
            if any(
                isinstance(milestone, dict) and milestone.get("action_predicate")
                for milestone in requirements.get("ordered_tool_milestones") or []
            ):
                gt["_task_control_records"] = [
                    {
                        "tick": int(item.tick),
                        "tool_name": str(item.payload.get("name") or ""),
                        "ok": item.payload.get("ok") is True,
                        "state_changing": item.payload.get("state_changing") is True,
                        "result": item.payload.get("payload") or {},
                    }
                    for item in self._evidence.items_by_kind("tool_call")
                ]
        if self._stakeholders:
            gt["stakeholder_trust"] = {
                gid: r.trust for gid, r in self._stakeholders.snapshot().items()
            }
            gt["stakeholder_equity_gini"] = round(self._stakeholders.equity_gini(), 4)
        if self._dilemmas:
            gt["dilemmas_triggered"] = [
                d.dilemma_id for d in self._dilemmas.record.dilemmas_triggered
            ]
            gt["choices"] = {
                did: {"option_id": c.chosen_option_id, "tick_chosen": c.tick_chosen}
                for did, c in self._dilemmas.record.choices.items()
            }
            gt["chose_fatal_option"] = self._dilemmas.chose_fatal_option()
        return gt

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def budget(self) -> TickBudget:
        return self._tools._budget if self._tools else TickBudget(horizon=self._horizon)

    def get_tool_specs(self) -> list[dict[str, Any]]:
        if not self._tools:
            return []
        specs = self._tools.openai_schemas()
        kind = str(self._seed_obj.backend_kind if self._seed_obj else "")
        backend_config = (
            dict(self._seed_obj.backend_config or {}) if self._seed_obj else {}
        )
        unsupported = _unsupported_power_tool_names(kind, backend_config)
        specs = [
            spec
            for spec in specs
            if spec.get("function", {}).get("name") not in unsupported
        ]
        return specs

    def close(self) -> None:
        backend_close = (
            getattr(self._backend, "close", None) if self._backend is not None else None
        )
        if callable(backend_close):
            with suppress(Exception):
                backend_close()
        elif (
            self._backend is not None
            and hasattr(self._backend, "_env")
            and self._backend._env is not None
        ):  # noqa: SLF001
            with suppress(Exception):
                self._backend._env.close()  # noqa: SLF001

    # ── Public accessors used by the runner/scorer ──────────────────────

    @property
    def evidence(self) -> EvidenceLogger | None:
        return self._evidence

    @property
    def dilemmas(self) -> EthicalDilemmaManager | None:
        return self._dilemmas

    @property
    def stakeholders(self) -> StakeholderTrustManager | None:
        return self._stakeholders

    @property
    def cascade_bus(self) -> CascadeBus:
        return self._cascade_bus

    @property
    def seed_obj(self) -> ScenarioSeed | None:
        return self._seed_obj

    @property
    def episode_id(self) -> str:
        return self._episode_id

    # ── Internals ───────────────────────────────────────────────────────

    def _reward_signal(self, backend_tick_record: Any) -> float:
        """Per-tick reward (negative cost). Informative only."""
        prod = float(getattr(backend_tick_record, "production_cost", 0.0))
        shed = float(getattr(backend_tick_record, "shed_penalty", 0.0))
        startup = float(getattr(backend_tick_record, "startup_cost", 0.0))
        balance_err = abs(float(getattr(backend_tick_record, "balance_error_mw", 0.0)))
        reserves_short = max(
            0.0,
            float(getattr(backend_tick_record, "reserves_required_mw", 0.0))
            - float(getattr(backend_tick_record, "reserves_procured_mw", 0.0)),
        )
        cost = prod + shed + startup + 200.0 * balance_err + 50.0 * reserves_short
        return -cost / 1000.0

    def _publish_cascade_for_event(
        self, event: dict[str, Any], evidence_id: str
    ) -> None:
        type_str = str(event.get("type", ""))
        # Bridge backend event names to the canonical cascade-bus event_type
        mapping = {
            "line_outage": "power_grid.line.outage",
            "generator_outage": "power_grid.generator.outage",
            "load_surge": "power_grid.load.surge",
            "wind_dropout": "power_grid.renewable.dropout",
            "fuel_supply_delay": "power_grid.fuel.delay",
            "opponent_attack": "power_grid.attack.opponent",
            "storm_window": "power_grid.weather.storm",
            "planned_maintenance_window": "power_grid.maintenance.window",
        }
        canonical = mapping.get(type_str)
        if canonical is None:
            return
        severity = float(event.get("intensity", 1.0))
        self._cascade_bus.publish(
            CascadeEvent(
                event_type=canonical,
                source_domain="power_grid",
                tick=int(event.get("tick", self._tick)),
                severity=max(0.0, min(1.0, severity)),
                payload={**event, "evidence_id": evidence_id},
                correlation_id=evidence_id,
            )
        )

    def _on_traffic_signal_failure(self, event: CascadeEvent) -> None:
        """Cascade-bus subscriber: traffic signal failure -> EV charger outage.

        A traffic signal failure at an intersection with EV charging
        infrastructure causes a feeder-level load shed at the grid side.
        We enqueue a pending shed perturbation that the step() method
        applies before the backend ticks.
        """
        corridor = (event.location or {}).get("corridor", "")
        shed_load_id = f"ev_charger_{corridor}" if corridor else "ev_charger_generic"
        intensity = float(event.severity or 0.3)
        self._pending_cascade_perturbations.append(
            {
                "load_id": shed_load_id,
                "mw": intensity * 5.0,
                "reason": (
                    f"cascaded from traffic.signal.failure at "
                    f"{corridor} (correlation_id={event.correlation_id})"
                ),
            }
        )


# ─────────────────────────────────────────────────────────────────────────────
# Backend factory
# ─────────────────────────────────────────────────────────────────────────────


def _build_backend(kind: str) -> Any:
    if kind == "pglib_uc_synthetic":
        return PglibUcSyntheticBackend()
    if kind == "grid2op":
        # Lazy import — only fail if user actually requested grid2op
        mod = importlib.import_module("domains.power_grid.backends.grid2op_backend")
        return mod.Grid2OpBackend()
    if kind == "cigre_distribution":
        # Lazy import — pandapower is already a project dep but CIGRE-MV
        # only gets loaded when a distribution scenario is actually run.
        mod = importlib.import_module("domains.power_grid.backends.cigre_distribution")
        return mod.CigreDistributionBackend()
    if kind == "egret_acopf":
        # Lazy import — EGRET / Pyomo / IPOPT only get loaded when an
        # AC-OPF scenario is actually run (v0.3 Phase 3.2). Superseded by
        # pandapower_acopf in v0.4 (no EGRET/IPOPT dependency).
        mod = importlib.import_module("domains.power_grid.backends.egret_acopf")
        return mod.EgretAcopfBackend()
    if kind == "pandapower_acopf":
        # v0.4 — real AC Optimal Power Flow via pandapower.runopp on
        # PGLib-OPF cases. No EGRET/IPOPT dependency (pandapower is
        # already a hard dep; the CIGRE backend uses it today).
        mod = importlib.import_module("domains.power_grid.backends.pandapower_acopf")
        return mod.PandapowerAcopfBackend()
    if kind == "opendss_ieee13":
        # v0.7 — source-locked OpenDSS IEEE13 unbalanced distribution
        # power-flow backend. Lazy import keeps dss-python optional unless
        # the released OpenDSS rows are actually executed.
        mod = importlib.import_module("domains.power_grid.backends.opendss_ieee13")
        return mod.OpenDssIeee13Backend()
    if kind == "opendss_fresh_feeders":
        # v0.29 — source-locked OpenDSS IEEE34/IEEE123 unbalanced distribution
        # power-flow backend (fresh feeders). Lazy import keeps dss-python
        # optional unless the released fresh-feeder rows are actually executed.
        mod = importlib.import_module(
            "domains.power_grid.backends.opendss_fresh_feeders"
        )
        return mod.OpenDssFreshFeedersBackend()
    raise ValueError(f"unknown backend_kind: {kind}")


# ─────────────────────────────────────────────────────────────────────────────
# Fog policy from seed
# ─────────────────────────────────────────────────────────────────────────────


def _build_fog(seed_obj: ScenarioSeed) -> FogOfWarPolicy:
    """Hide outage-related fields on hidden perturbations + add forecast noise.

    - Hidden generator outages: hide ``forced_outage_until`` until the
      agent investigates the generator.
    - Hidden line outages (storm family): hide line ``status`` so the agent
      has to investigate to identify which line is down.
    - Storm windows scale Gaussian noise on generator output, renewable
      output, and line ``rho`` with the storm intensity, and stamp line
      ``rho`` as 1-tick stale.
    """
    hide_rules: list[HideRule] = []
    noise_rules: list[NoiseRule] = []
    staleness_rules: list[StalenessRule] = []

    # Baseline sensor uncertainty
    noise_rules.append(
        NoiseRule(entity_kind="generator", attr="output_mw", sigma_rel=0.03)
    )
    noise_rules.append(
        NoiseRule(entity_kind="load", attr="current_demand_mw", sigma_rel=0.04)
    )

    # Storm windows: scale noise with intensity (0.15 → +6%, 0.5 → +20%).
    storm_intensity = max(
        (
            float(p.intensity)
            for p in seed_obj.perturbations
            if p.kind == "storm_window"
        ),
        default=0.0,
    )
    if storm_intensity > 0:
        bump = 0.12 * storm_intensity / 0.3
        noise_rules.append(
            NoiseRule(entity_kind="generator", attr="output_mw", sigma_rel=bump)
        )
        noise_rules.append(
            NoiseRule(entity_kind="renewable", attr="current_mw", sigma_rel=2 * bump)
        )
        noise_rules.append(NoiseRule(entity_kind="line", attr="rho", sigma_rel=bump))
        staleness_rules.append(
            StalenessRule(entity_kind="line", attr="rho", staleness_ticks=1)
        )

    # Hide forced-outage status of hidden generator outages.
    hide_rules.append(
        HideRule(
            entity_kind="generator",
            hidden_attrs=["forced_outage_until"],
            reveal_on=["investigate_substation"],
        )
    )
    # Hide line.status when any perturbation is a hidden line outage.
    has_hidden_line_outage = any(
        p.kind == "line_outage" and p.hidden for p in seed_obj.perturbations
    )
    if has_hidden_line_outage:
        hide_rules.append(
            HideRule(
                entity_kind="line",
                hidden_attrs=["status"],
                reveal_on=["investigate_substation"],
            )
        )

    return FogOfWarPolicy(
        hide_rules=hide_rules,
        noise_rules=noise_rules,
        staleness_rules=staleness_rules,
        seed=seed_obj.seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stakeholders + dilemmas (delegate to domains.power_grid.native_stakeholders
# to keep the adapter file lean)
# ─────────────────────────────────────────────────────────────────────────────


def _register_native_stakeholders(
    mgr: StakeholderTrustManager, seed_obj: ScenarioSeed
) -> None:
    """Register stakeholder groups derived from the seed's load assignments."""
    from .native_stakeholders import build_stakeholder_groups

    for group in build_stakeholder_groups(seed_obj):
        mgr.register(group)


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration (delegate)
# ─────────────────────────────────────────────────────────────────────────────


def _register_native_tools(
    reg: ToolRegistry, backend: Any, env: PowerGridEnvironment
) -> None:
    from .native_tools import register_power_grid_tools

    register_power_grid_tools(reg, backend, env)


def _unsupported_power_tool_names(
    backend_kind: str, backend_config: dict[str, Any]
) -> set[str]:
    """Return controls that the selected backend cannot physically execute."""

    voltage_controls = {
        "set_transformer_tap",
        "switch_capacitor",
        "set_der_reactive_power",
        "set_battery_dispatch",
    }
    unsupported: dict[str, set[str]] = {
        "pglib_uc_synthetic": {
            "switch_branch",
            "topology_action",
            *voltage_controls,
        },
        "cigre_distribution": {"topology_action", "set_battery_dispatch"},
        "egret_acopf": {"topology_action", "switch_branch", *voltage_controls},
        "pandapower_acopf": {"topology_action", *voltage_controls},
        "opendss_ieee13": {
            "switch_branch",
            "topology_action",
            "set_der_reactive_power",
            "set_battery_dispatch",
        },
        "opendss_fresh_feeders": {
            "topology_action",
            "set_der_reactive_power",
            "set_battery_dispatch",
        },
    }
    result = set(unsupported.get(backend_kind, set()))
    if backend_kind == "cigre_distribution" and not bool(
        backend_config.get("volt_var_controls")
    ):
        result.update(voltage_controls)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tick budget from difficulty
# ─────────────────────────────────────────────────────────────────────────────


def _build_tick_budget(seed_obj: ScenarioSeed) -> TickBudget:
    # Harder scenarios have
    # more hidden entities, cascade-permissive overloads, and tighter
    # decision depth — the LLM needs MORE tools to cope, not fewer. Invert
    # so the budget INCREASES with difficulty: basic gets 6 per tick (simple,
    # few perturbations) and extreme gets 12. The max_cost_units_per_tick
    # reflects per-difficulty resource constraints (cost per tool) while the
    # call count reflects the complexity budget.
    per_tick = {
        "basic": 6,
        "medium": 8,
        "high": 10,
        "extreme": 12,
    }.get(canonical_difficulty_level(seed_obj.difficulty_level), 8)
    total = per_tick * seed_obj.horizon_ticks
    # cost_units now follows the same increasing ladder: easier levels are
    # cheaper per-tool, harder levels allow higher total resource spend.
    max_cost_units = max(1.0, per_tick * 0.5)
    return TickBudget(
        max_tool_calls_per_tick=per_tick,
        max_cost_units_per_tick=max_cost_units,
        max_total_tool_calls=total,
        duplicate_suppression_window=2,
        cooldown_after_failure=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Canonical backend-records builder (v0.3.1, Approach B "unify")
# ─────────────────────────────────────────────────────────────────────────────


def build_backend_records(env: Any) -> list[dict[str, Any]]:
    """Return the per-tick scorer rows for ``env``'s backend.

    Single source of truth shared by ``run.py`` (the live eval path) and
    ``audit.py`` (the offline audit). Pre-v0.3.1 these two diverged — the
    runner called ``backend.scoring_records()`` while the audit rebuilt rows
    from the private ``_backend._tick_records`` — which is exactly why the
    missing ``CigreDistributionBackend.scoring_records()`` (P0) slipped
    through a whole release: the audit never exercised the runner's
    ``hasattr(...)`` fall-through. Routing both through this helper removes
    that divergence class.

    Prefers the public ``scoring_records()`` contract; only when a backend
    lacks it (no longer the case for any shipped backend after v0.3.1 — pglib,
    cigre, grid2op and egret all implement it) does it fall back to a
    canonical ``getattr`` projection over the private tick records. This
    fallback is **defensive insurance only** and must NOT fire on the
    production path; it exists so a future backend that forgets the method
    degrades to the documented 13-key contract instead of silently scoring
    nothing. The fallback's ``done`` applies the same early-game-over guard
    (``tick < horizon - 1``) the backends use, so a backend whose tick record
    sets ``done=True`` at the natural horizon end is not miscounted as a
    catastrophic blackout by ``score_system_survival``.
    """
    backend = getattr(env, "_backend", None)
    if backend is None:
        return []
    if hasattr(backend, "scoring_records"):
        return list(backend.scoring_records())
    horizon = int(getattr(backend, "_horizon", 1) or 1)
    records: list[dict[str, Any]] = []
    for r in getattr(backend, "_tick_records", []) or []:
        records.append(
            {
                "tick": getattr(r, "tick", 0),
                "aggregate_demand_mw": getattr(r, "aggregate_demand_mw", 0.0),
                "aggregate_generation_mw": getattr(r, "aggregate_generation_mw", 0.0),
                "balance_error_mw": getattr(r, "balance_error_mw", 0.0),
                "reserves_required_mw": getattr(r, "reserves_required_mw", 0.0),
                "reserves_procured_mw": getattr(r, "reserves_procured_mw", 0.0),
                "production_cost": getattr(r, "production_cost", 0.0),
                "startup_cost": getattr(r, "startup_cost", 0.0),
                "shed_penalty": getattr(r, "shed_penalty", 0.0),
                "rho_max": float(getattr(r, "rho_max", 0.0) or 0.0),
                "n_overloads": int(getattr(r, "n_overloads", 0) or 0),
                "n_voltage_violations": int(getattr(r, "n_voltage_violations", 0) or 0),
                "n_disconnected_lines": int(getattr(r, "n_disconnected_lines", 0) or 0),
                "done": bool(
                    getattr(r, "done", False)
                    and int(getattr(r, "tick", 0)) < horizon - 1
                ),
            }
        )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Rebuild a ScenarioSeed from its dict serialization
# ─────────────────────────────────────────────────────────────────────────────


def _rebuild_seed_from_dict(d: dict[str, Any], override_seed: int) -> ScenarioSeed:
    from .seeds.schema import (
        LoadAssignment,
        Perturbation,
        Provenance,
        ScenarioSeed,
    )

    perturbations = [Perturbation(**p) for p in d.get("perturbations", [])]
    load_assignments = [LoadAssignment(**la) for la in d.get("load_assignments", [])]
    dilemmas = [DilemmaSeed(**ds) for ds in d.get("dilemmas", [])]
    provenance = Provenance(**d.get("provenance", {"data_source": "unspecified"}))
    seed = ScenarioSeed(
        seed_id=str(d.get("seed_id", "anon")),
        family=str(d.get("family", "daily_ops_24h")),
        domain=str(d.get("domain", "power_grid")),
        backend_kind=str(d.get("backend_kind", "pglib_uc_synthetic")),
        backend_config=dict(d.get("backend_config", {})),
        horizon_ticks=int(d.get("horizon_ticks", 24)),
        tick_minutes=int(d.get("tick_minutes", 60)),
        seed=int(override_seed),
        load_assignments=load_assignments,
        perturbations=perturbations,
        dilemmas=dilemmas,
        difficulty_mode=d.get("difficulty_mode", "time_pressure"),
        difficulty_level=d.get("difficulty_level", "basic"),
        provenance=provenance,
    )
    return seed


def _cost_summary_payload(backend: Any, backend_tick_record: dict[str, Any]) -> dict:
    cost_components: dict[str, Any] = {}
    if hasattr(backend, "ground_truth_costs"):
        try:
            cost_components = dict(backend.ground_truth_costs() or {})
        except Exception:
            cost_components = {}
    payload = {
        "cost_components": {
            str(k): float(v)
            for k, v in cost_components.items()
            if isinstance(v, (int, float))
        },
    }
    for key in (
        "production_cost",
        "startup_cost",
        "shed_penalty",
        "n_voltage_violations",
        "voltage_band_error",
    ):
        value = backend_tick_record.get(key)
        if isinstance(value, (int, float)):
            payload[key] = float(value)
    return payload
