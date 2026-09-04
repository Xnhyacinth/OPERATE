"""
domains.microgrid.native_tools — Microgrid-domain native EMS tool surface.

``register_microgrid_tools(reg, backend, env)`` populates a
``core.ToolRegistry`` with the backend-supported subset of the spec §6 EMS
tool set. Categories:

- Dispatch / control (state-changing):
    set_battery_dispatch, dispatch_genset (delay 1, start-up), set_grid_exchange
    (DOMAIN_REJECTED when islanded), connect_pcc (delay 1, re-sync),
    curtail_der, shed_load, set_der_reactive_power (LV-only effective)
- Investigation (read-only, paid, noised — NO active-recovery credit):
    forecast_query, investigate_asset
- Record / foresight:
    commit_to_plan
- Meta:
    wait, noop

Per Red Line #3 this module does NOT import from another domain. Domain
rejections return ``{"_status": "error", ...}`` which the protocol surfaces
as ``DOMAIN_REJECTED`` (not a silent ``ok=True``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core import (
    ToolContext,
    ToolRegistry,
    ToolSpec,
    commit_to_plan_handler,
    noop_tool_spec,
    plan_autonomy_properties,
    wait_tool_spec,
)

from .native_stakeholders import trust_event_for_shed

if TYPE_CHECKING:  # pragma: no cover
    from .adapter import MicrogridEnvironment


def register_microgrid_tools(
    reg: ToolRegistry,
    backend: Any,
    env: MicrogridEnvironment,
) -> None:
    """Register only the native tools implemented by the selected backend."""

    supported = getattr(backend, "supported_tool_names", None)

    def register(spec: ToolSpec) -> None:
        if supported is None or spec.name in supported:
            reg.register(spec)

    try:
        visible_entities = (backend.snapshot() or {}).get("entities") or {}
    except (AttributeError, RuntimeError):
        visible_entities = {}
    battery_ids = sorted(
        entity_id
        for entity_id, row in visible_entities.items()
        if row.get("kind") == "battery"
    )
    der_ids = sorted(
        entity_id
        for entity_id, row in visible_entities.items()
        if row.get("kind") in {"pv", "wind", "der"}
    )
    investigable_ids = sorted(
        entity_id
        for entity_id, row in visible_entities.items()
        if row.get("kind") in {"battery", "pv", "wind", "der", "genset"}
    )
    battery_id_schema: dict[str, Any] = {"type": "string"}
    der_id_schema: dict[str, Any] = {"type": "string"}
    asset_id_schema: dict[str, Any] = {"type": "string"}
    if battery_ids:
        battery_id_schema["enum"] = battery_ids
    if der_ids:
        der_id_schema["enum"] = der_ids
    if investigable_ids:
        asset_id_schema["enum"] = investigable_ids

    # ── DISPATCH / CONTROL ─────────────────────────────────────────────

    register(
        ToolSpec(
            name="set_battery_dispatch",
            description=(
                "Set the battery dispatch setpoint in MW: p_mw>0 charges "
                "(stores), p_mw<0 discharges (supplies the bus). SoC updates "
                "next tick. DOMAIN_REJECTED if the SoC/rate window cannot "
                "accommodate the request."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "battery_id": dict(battery_id_schema),
                    "p_mw": {"type": "number"},
                },
                "required": ["battery_id", "p_mw"],
            },
            handler=_h_apply(backend, env, "set_battery_dispatch"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="storage_unit",
            actuator_family="battery_dispatch",
            cost_units=0.5,
        )
    )

    register(
        ToolSpec(
            name="dispatch_genset",
            description=(
                "Commit + ramp the controllable diesel genset to p_mw. "
                "Acknowledged immediately (status=pending, effect_due_tick="
                "t+1); the genset physically produces one tick later "
                "(start-up). Emits startup_cost on first commit. "
                "DOMAIN_REJECTED if the genset is unavailable this scenario; "
                "small fail_rate → INJECTED_FAILURE."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "genset_id": {"type": "string"},
                    "p_mw": {"type": "number"},
                },
                "required": ["genset_id", "p_mw"],
            },
            handler=_h_dispatch_genset(backend),
            state_changing=True,
            semantic_role="control",
            native_target_kind="dispatchable_generator",
            actuator_family="unit_commitment_dispatch",
            delay_ticks=1,
            fail_rate=0.05,
            cost_units=1.0,
        )
    )

    register(
        ToolSpec(
            name="set_grid_exchange",
            description=(
                "Set the PCC exchange setpoint in MW: p_mw>0 imports, p_mw<0 "
                "exports. DOMAIN_REJECTED when the microgrid is islanded "
                "(PCC open)."
            ),
            parameters={
                "type": "object",
                "properties": {"p_mw": {"type": "number"}},
                "required": ["p_mw"],
            },
            handler=_h_set_grid_exchange(backend, env),
            state_changing=True,
            semantic_role="control",
            native_target_kind="point_of_common_coupling",
            actuator_family="grid_exchange_setpoint",
            cost_units=0.25,
        )
    )

    register(
        ToolSpec(
            name="connect_pcc",
            description=(
                "Reconnect (connect=true) or disconnect (connect=false) the "
                "main grid at the PCC. Acknowledged immediately (status="
                "pending, effect_due_tick=t+1); the breaker re-syncs one tick "
                "later. fail_rate models a breaker mis-operation."
            ),
            parameters={
                "type": "object",
                "properties": {"connect": {"type": "boolean"}},
                "required": ["connect"],
            },
            handler=_h_connect_pcc(backend),
            state_changing=True,
            semantic_role="control",
            native_target_kind="point_of_common_coupling_breaker",
            actuator_family="breaker_switching",
            delay_ticks=1,
            fail_rate=0.05,
            cost_units=1.0,
        )
    )

    register(
        ToolSpec(
            name="curtail_der",
            description=(
                "Cap a controllable DER (PV/wind) output at target_mw. "
                "DOMAIN_REJECTED on an unknown der_id or a negative target."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "der_id": dict(der_id_schema),
                    "target_mw": {"type": "number"},
                },
                "required": ["der_id", "target_mw"],
            },
            handler=_h_apply(backend, env, "curtail_der"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="distributed_energy_resource",
            actuator_family="active_power_curtailment",
            cost_units=0.25,
        )
    )

    register(
        ToolSpec(
            name="shed_load",
            description=(
                "Reduce a feeder load by mw (criticality-weighted). "
                "DOMAIN_REJECTED on an unknown load_id or a non-positive mw. "
                "Shedding a critical (hospital/water) load drives stakeholder "
                "trust negatively in islanding scenarios."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "load_id": {"type": "string"},
                    "mw": {"type": "number"},
                },
                "required": ["load_id", "mw"],
            },
            handler=_h_shed_load(backend, env),
            state_changing=True,
            semantic_role="control",
            native_target_kind="feeder_load",
            actuator_family="load_shedding",
            cost_units=0.0,
        )
    )

    register(
        ToolSpec(
            name="set_der_reactive_power",
            description=(
                "Set a DER's reactive-power setpoint for LV Volt-Var control. "
                "Pandapower generator convention: q_mvar<0 absorbs reactive "
                "power and lowers over-voltage; q_mvar>0 injects reactive "
                "power and supports under-voltage. Keep |q_mvar| within the "
                "entity's visible max_abs_q_mvar. Only effective on the "
                "microgrid_lv_voltage_6h power-flow tier; a no-op "
                "(status=unsupported_off_lv) on aggregate EMS families."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "der_id": dict(der_id_schema),
                    "q_mvar": {"type": "number"},
                },
                "required": ["der_id", "q_mvar"],
            },
            handler=_h_apply(backend, env, "set_der_reactive_power"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="der_inverter",
            actuator_family="reactive_power_setpoint",
            cost_units=0.25,
        )
    )

    # ── INVESTIGATION (read-only, paid, noised) ────────────────────────

    register(
        ToolSpec(
            name="forecast_query",
            description=(
                "Query a NOISED PV/load/price forecast over a horizon "
                "(documented bias/variance; never ground truth). Paid query."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "horizon_ticks": {"type": "integer", "minimum": 1},
                },
                "required": ["horizon_ticks"],
            },
            handler=_h_forecast_query(backend, env),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="microgrid_forecast",
            cost_units=1.0,
        )
    )

    register(
        ToolSpec(
            name="investigate_asset",
            description=(
                "Reveal the fogged SoC / availability of one asset (battery, "
                "DER, genset). Use the exact entity ID shown in the current "
                "observation, such as batt0 or der0; do not invent aliases "
                "such as battery_0 or pv_0. Paid query; also discovers a "
                "hidden DER failure."
            ),
            parameters={
                "type": "object",
                "properties": {"asset_id": dict(asset_id_schema)},
                "required": ["asset_id"],
            },
            handler=_h_investigate_asset(backend, env),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="microgrid_asset",
            cost_units=1.0,
        )
    )

    # ── RECORD / FORESIGHT ─────────────────────────────────────────────

    register(
        ToolSpec(
            name="commit_to_plan",
            description=(
                "Record foresight predictions about ramps / islanding so the "
                "foresight scorer can verify follow-through."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan_id": {"type": "string"},
                    "rationale": {"type": "string"},
                    "replaces_plan_id": {"type": "string"},
                    "revision_reason": {"type": "string"},
                    "trigger_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    **plan_autonomy_properties(),
                    "predictions": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["plan_id"],
            },
            handler=commit_to_plan_handler(
                env, events_key="predictions", include_horizon_ticks=False
            ),
            state_changing=False,
            semantic_role="planning",
            native_target_kind="standing_plan",
            cost_units=0.0,
        )
    )

    # ── META ───────────────────────────────────────────────────────────

    register(
        wait_tool_spec(
            "Decline intervention for this interval; simulator time advances independently."
        )
    )
    register(noop_tool_spec())


# ─────────────────────────────────────────────────────────────────────────────
# Handler factories
# ─────────────────────────────────────────────────────────────────────────────


def _h_apply(backend: Any, env: MicrogridEnvironment, tool_name: str):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        result = backend.apply_tool_effect(tool_name, args)
        status = result.get("_status")
        applied = status not in {
            "error",
            "unsupported",
            "unsupported_on_lv",
            "no_effect",
        }
        if applied and hasattr(backend, "record_applied_control"):
            backend.record_applied_control(
                tick=ctx.tick,
                tool_name=tool_name,
                args=args,
            )
        if applied and env.evidence is not None:
            env.evidence.log(tool_name, ctx.tick, payload={**result}, source="tool")
        return result

    return handler


def _h_dispatch_genset(backend: Any):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        gid = str(args.get("genset_id", "genset0"))
        p_mw = float(args.get("p_mw", 0.0))
        genset = getattr(backend, "_genset", None)
        if genset is None or not getattr(genset, "available", False):
            return {"_status": "error", "error": "genset_unavailable", "genset_id": gid}
        if not hasattr(backend, "queue_effect"):
            return {"_status": "unsupported", "error": "genset_not_modeled"}
        due = int(ctx.extra.get("materialization_tick", int(ctx.tick) + 1))
        backend.queue_effect(
            due_tick=due,
            kind="genset_commit",
            payload={"genset_id": gid, "p_mw": p_mw},
        )
        return {
            "_status": "pending",
            "due_tick": due,
            "genset_id": gid,
            "p_mw": p_mw,
            "info": "genset start-up; produces at due_tick",
        }

    return handler


def _h_set_grid_exchange(backend: Any, env: MicrogridEnvironment):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        if hasattr(backend, "is_islanded_at") and backend.is_islanded_at(ctx.tick):
            return {"_status": "error", "error": "islanded_pcc_open"}
        result = backend.apply_tool_effect("set_grid_exchange", args)
        if result.get("_status") != "error" and env.evidence is not None:
            env.evidence.log(
                "set_grid_exchange", ctx.tick, payload={**result}, source="tool"
            )
        return result

    return handler


def _h_connect_pcc(backend: Any):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        connect = bool(args.get("connect", True))
        # Voltage-control backends (e.g. pandapower_lv) do not model a PCC
        # breaker and lack the delayed-effect queue. Report an honest
        # ``unsupported`` no-effect (symmetric with ``set_der_reactive_power``
        # on EMS backends) instead of raising an AttributeError.
        if not hasattr(backend, "queue_effect"):
            return {"_status": "unsupported", "error": "pcc_not_modeled"}
        due = int(ctx.extra.get("materialization_tick", int(ctx.tick) + 1))
        backend.queue_effect(
            due_tick=due, kind="pcc_reconnect", payload={"connect": connect}
        )
        return {
            "_status": "pending",
            "due_tick": due,
            "connect": connect,
            "info": "PCC breaker re-sync; effective at due_tick",
        }

    return handler


def _h_shed_load(backend: Any, env: MicrogridEnvironment):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        result = backend.apply_tool_effect("shed_load", args)
        if result.get("_status") == "error":
            return result
        if env.evidence is not None:
            env.evidence.log("shed_load", ctx.tick, payload={**result}, source="tool")
        # Stakeholder trust event (only registered on islanding scenarios).
        if env.stakeholders is not None:
            gid, ev = trust_event_for_shed(
                stakeholder_class=str(result.get("stakeholder_class", "residential")),
                criticality=float(result.get("criticality", 0.2) or 0.2),
            )
            if gid in env.stakeholders.snapshot():
                env.stakeholders.record_event(group_id=gid, event=ev, tick=ctx.tick)
                env.evidence.log(
                    "trust_event",
                    ctx.tick,
                    payload={
                        "group_id": gid,
                        "event": ev,
                        "source_tool": "shed_load",
                        "load_id": result.get("load_id"),
                    },
                    source="tool",
                )
        return result

    return handler


def _h_forecast_query(backend: Any, env: MicrogridEnvironment):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        horizon = int(args.get("horizon_ticks", 3) or 3)
        result = backend.forecast_for(horizon)
        if env.evidence is not None:
            env.evidence.log(
                "investigation",
                ctx.tick,
                payload={"tool": "forecast_query", "horizon_ticks": horizon},
                source="tool",
            )
        return {"horizon_ticks": horizon, "forecast": result}

    return handler


def _h_investigate_asset(backend: Any, env: MicrogridEnvironment):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        aid = str(args.get("asset_id", ""))
        result = backend.investigate_asset(aid)
        if result.get("_status") != "error":
            env.mark_agency_source_events_revealed(
                target_id=aid,
                reveal_tick=ctx.tick,
            )
        if env.evidence is not None:
            env.evidence.log(
                "investigation",
                ctx.tick,
                payload={"target_id": aid, "tool": "investigate_asset"},
                source="tool",
            )
        return result

    return handler


# ``commit_to_plan`` handler factory moved to ``core.common_tools``
# (domain-agnostic; see ``register_microgrid_tools`` above for the call
# site: ``commit_to_plan_handler(env, events_key="predictions",
# include_horizon_ticks=False)`` — microgrid uses ``predictions`` (not
# ``predicted_events``) as both its args key and evidence-payload key, and
# never surfaced ``horizon_ticks``, so those two flags keep this domain's
# behavior byte-identical to its pre-extraction handler).
