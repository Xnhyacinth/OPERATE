"""
domains.disaster.native_tools — Disaster-domain native tool surface.

Mirrors the power-grid native_tools shape: ``register_disaster_tools(reg,
backend, env)`` populates a ``core.ToolRegistry`` with ~10 OpenAI-style
function tools. Categories:

- Investigation (no state change, low cost):
    survey_zone, dispatch_recon

- Coordination / dispatch (state-changing):
    dispatch_ambulance, dispatch_fire_brigade, dispatch_police_cordon

- Resource (state-changing):
    assign_triage, evacuate_zone

- Mutual aid (F-01 delayed-effect):
    request_mutual_aid_team — ack at tick t, materializes at tick t+2

- Ethics / record (mirror power_grid):
    moral_choice, commit_to_plan

- Meta:
    wait, noop

Per Red Line #3, this module does NOT import from
``domains.power_grid``. The mirror is structural — same handler-factory
pattern, same ``ToolSpec(...)`` calls, disaster-native names.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from core import (
    ToolContext,
    ToolRegistry,
    ToolSpec,
    commit_to_plan_handler,
    moral_choice_handler,
    noop_tool_spec,
    wait_tool_spec,
)

from .native_stakeholders import (
    trust_event_for_dispatch,
)

if TYPE_CHECKING:  # pragma: no cover
    from .adapter import DisasterEnvironment


# Map dispatch tool name → which stakeholder responder class to credit
# for the trust event when the dispatch lands.
_DISPATCH_TOOL_TO_RESPONDER: dict[str, str] = {
    "dispatch_ambulance": "responder_ems",
    "dispatch_fire_brigade": "responder_fire",
    "dispatch_police_cordon": "responder_police",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point — adapter calls this at reset
# ─────────────────────────────────────────────────────────────────────────────


def register_disaster_tools(
    reg: ToolRegistry,
    backend: Any,
    env: DisasterEnvironment,
) -> None:
    """Register the full native-tool surface for the disaster domain."""

    # ── INVESTIGATION ──────────────────────────────────────────────────

    reg.register(
        ToolSpec(
            name="survey_zone",
            description=(
                "Pay one tick of attention to survey a zone's current "
                "ground truth (buried count, fire intensity, evacuation "
                "status). Counts against the tick budget. Reveals hidden "
                "perturbations in that zone via the fog-of-war policy."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_zone": {"type": "string"},
                },
                "required": ["target_zone"],
            },
            handler=_h_survey_zone(backend, env),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="disaster_zone",
            cost_units=1.5,
        )
    )

    reg.register(
        ToolSpec(
            name="dispatch_recon",
            description=(
                "Send a recon asset (drone, motorcycle scout, satellite "
                "pass) to a zone to reveal hidden hazards. More expensive "
                "than survey_zone but unlocks more ground-truth fields."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_zone": {"type": "string"},
                    "asset": {
                        "type": "string",
                        "enum": ["drone", "scout", "satellite"],
                    },
                },
                "required": ["target_zone", "asset"],
            },
            handler=_h_dispatch_recon(backend, env),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="hazard_observation",
            cost_units=2.0,
        )
    )

    # ── COORDINATION / DISPATCH ────────────────────────────────────────

    reg.register(
        ToolSpec(
            name="dispatch_ambulance",
            description=(
                "Dispatch ``n_teams`` ambulance/EMS teams to ``target_zone``. "
                "Reduces buried-civilian count by ~4 per team per tick once "
                "they arrive. Adds an EMS trust event proportional to the "
                "criticality of the target zone."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_zone": {"type": "string"},
                    "n_teams": {"type": "integer", "minimum": 1},
                },
                "required": ["target_zone"],
            },
            handler=_h_dispatch(backend, env, "dispatch_ambulance"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="ambulance_team",
            actuator_family="emergency_response_dispatch",
            cost_units=2.0,
        )
    )

    reg.register(
        ToolSpec(
            name="dispatch_fire_brigade",
            description=(
                "Dispatch ``n_teams`` fire-brigade teams to ``target_zone``. "
                "Suppresses fire_intensity in that zone and blocks "
                "fire_spread to adjacent zones. No-op in zones with "
                "fire_intensity == 0."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_zone": {"type": "string"},
                    "n_teams": {"type": "integer", "minimum": 1},
                },
                "required": ["target_zone"],
            },
            handler=_h_dispatch(backend, env, "dispatch_fire_brigade"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="fire_brigade_team",
            actuator_family="emergency_response_dispatch",
            cost_units=2.0,
        )
    )

    reg.register(
        ToolSpec(
            name="dispatch_police_cordon",
            description=(
                "Dispatch police units to ``target_zone`` to cordon "
                "(``purpose=traffic`` or ``purpose=security``). Reduces "
                "secondary casualties from looting / panic. Adds a "
                "police-responder trust event."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_zone": {"type": "string"},
                    "purpose": {
                        "type": "string",
                        "enum": ["traffic", "security", "evac_route"],
                    },
                    "n_teams": {"type": "integer", "minimum": 1},
                },
                "required": ["target_zone", "purpose"],
            },
            handler=_h_dispatch(backend, env, "dispatch_police_cordon"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="police_team",
            actuator_family="emergency_response_dispatch",
            cost_units=1.5,
        )
    )

    # ── RESOURCE ───────────────────────────────────────────────────────

    reg.register(
        ToolSpec(
            name="assign_triage",
            description=(
                "Assign a START-style triage priority (RED / YELLOW / "
                "GREEN / BLACK) to a zone's casualties. Drives downstream "
                "ambulance routing priority and feeds the equity scorer."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "zone": {"type": "string"},
                    "priority": {
                        "type": "string",
                        "enum": ["RED", "YELLOW", "GREEN", "BLACK"],
                    },
                },
                "required": ["zone", "priority"],
            },
            handler=_h_assign_triage(backend, env),
            state_changing=True,
            semantic_role="control",
            native_target_kind="casualty_group",
            actuator_family="triage_assignment",
            cost_units=0.5,
        )
    )

    reg.register(
        ToolSpec(
            name="evacuate_zone",
            description=(
                "Evacuate residents of ``zone`` to ``destination`` along "
                "``route_hint``. Sets the zone's evacuated flag; remaining "
                "buried civilians still accumulate unserved minutes until "
                "rescued."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "zone": {"type": "string"},
                    "destination": {"type": "string"},
                    "route_hint": {"type": "string"},
                },
                "required": ["zone", "destination"],
            },
            handler=_h_evacuate_zone(backend, env),
            state_changing=True,
            semantic_role="control",
            native_target_kind="population_zone",
            actuator_family="evacuation_dispatch",
            cost_units=2.5,
        )
    )

    # ── MUTUAL AID (F-01 delayed effect) ───────────────────────────────

    reg.register(
        ToolSpec(
            name="request_mutual_aid_team",
            description=(
                "Request mutual-aid teams from a neighboring region. "
                "The call is acknowledged immediately at tick ``t`` "
                "(status=pending, effect_due_tick=t+2); the physical "
                "team count enters the backend at ``t+2`` and the "
                "materialized result is surfaced as a realized event "
                "then. Identical contract to power_grid's "
                "request_mutual_aid (F-01)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "team_type": {
                        "type": "string",
                        "enum": ["ambulance", "fire", "police", "medical"],
                    },
                    "source_region": {"type": "string"},
                    "n_teams": {
                        "type": "integer",
                        "minimum": 1,
                        "default": 1,
                    },
                },
                "required": ["team_type", "source_region"],
            },
            handler=_h_request_mutual_aid_team(backend),
            state_changing=True,
            semantic_role="control",
            native_target_kind="mutual_aid_capacity",
            actuator_family="mutual_aid_dispatch",
            delay_ticks=2,
            cost_units=2.0,
        )
    )

    # ── ETHICS / RECORD (mirror power_grid) ────────────────────────────

    reg.register(
        ToolSpec(
            name="moral_choice",
            description=(
                "Resolve an active dilemma by recording the chosen option "
                "and rationale. Rationale should mention the trade-off "
                "and the affected stakeholders."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "dilemma_id": {"type": "string"},
                    "option_id": {"type": "string"},
                    "rationale": {"type": "string"},
                    "tradeoffs_considered": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "affected_stakeholders": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "reversibility_assessment": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "required": [
                    "dilemma_id",
                    "option_id",
                    "rationale",
                    "tradeoffs_considered",
                    "affected_stakeholders",
                    "reversibility_assessment",
                ],
            },
            handler=moral_choice_handler(env),
            state_changing=True,
            semantic_role="communication",
            native_target_kind="active_dilemma",
            actuator_family="dilemma_resolution",
            cost_units=0.0,
        )
    )

    reg.register(
        ToolSpec(
            name="commit_to_plan",
            description=(
                "Record a multi-tick plan with predicted events. Used by "
                "the foresight scorer to verify the agent followed "
                "through on declared intent."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "plan_id": {"type": "string"},
                    "horizon_ticks": {"type": "integer"},
                    "rationale": {"type": "string"},
                    "replaces_plan_id": {"type": "string"},
                    "revision_reason": {"type": "string"},
                    "trigger_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "predicted_events": {
                        "type": "array",
                        "description": (
                            "List of predicted events. Each item has "
                            "``event_type`` (one of building_collapse, "
                            "aftershock, fire_spread, medical_surge, "
                            "comms_blackout), optional ``target_zone``, "
                            "required ``tick_offset`` (ticks ahead), "
                            "optional ``confidence`` in [0, 1]."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "event_type": {
                                    "type": "string",
                                    "enum": [
                                        "building_collapse",
                                        "aftershock",
                                        "fire_spread",
                                        "medical_surge",
                                        "comms_blackout",
                                        "road_blockage",
                                        "bridge_failure",
                                        "tsunami_inundation",
                                    ],
                                },
                                "target_zone": {"type": "string"},
                                "tick_offset": {
                                    "type": "integer",
                                    "minimum": 1,
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                },
                            },
                            "required": ["event_type", "tick_offset"],
                        },
                    },
                },
                "required": ["plan_id", "rationale"],
            },
            handler=commit_to_plan_handler(env),
            state_changing=False,
            semantic_role="planning",
            native_target_kind="standing_plan",
            cost_units=0.0,
        )
    )

    # ── META ───────────────────────────────────────────────────────────

    reg.register(wait_tool_spec())
    reg.register(noop_tool_spec())


# ─────────────────────────────────────────────────────────────────────────────
# Handler factories
# ─────────────────────────────────────────────────────────────────────────────


def _h_survey_zone(backend: Any, env: DisasterEnvironment):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        zone_id = str(args.get("target_zone", ""))
        result = backend.apply_tool_effect("survey_zone", {"target_zone": zone_id})
        # Mark revealed in fog of war so subsequent snapshots include
        # hidden fields for this zone.
        if ctx.extra.get("fog") is not None:
            with contextlib.suppress(Exception):
                ctx.extra["fog"].mark_revealed(zone_id)
        if env.evidence is not None:
            env.evidence.log(
                "investigation",
                ctx.tick,
                payload={"target_id": zone_id, "tool": "survey_zone"},
                source="tool",
            )
        return {"target_zone": zone_id, "ground_truth": result}

    return handler


def _h_dispatch_recon(backend: Any, env: DisasterEnvironment):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        zone_id = str(args.get("target_zone", ""))
        asset = str(args.get("asset", "drone"))
        result = backend.apply_tool_effect(
            "dispatch_recon", {"target_zone": zone_id, "asset": asset}
        )
        if ctx.extra.get("fog") is not None:
            with contextlib.suppress(Exception):
                ctx.extra["fog"].mark_revealed(zone_id)
        if env.evidence is not None:
            env.evidence.log(
                "investigation",
                ctx.tick,
                payload={
                    "target_id": zone_id,
                    "tool": "dispatch_recon",
                    "asset": asset,
                },
                source="tool",
            )
        return {"target_zone": zone_id, "asset": asset, "ground_truth": result}

    return handler


def _h_dispatch(backend: Any, env: DisasterEnvironment, tool_name: str):
    """Factory for the three dispatch tools.

    Records a per-dispatch trust event against the responder class
    (responder_ems / responder_fire / responder_police) using the same
    ``trust_event`` evidence pattern as power_grid's shed_load.
    """
    responder_class = _DISPATCH_TOOL_TO_RESPONDER[tool_name]

    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        # Resolve target zone for trust-event payload BEFORE mutating.
        zone_id = str(args.get("target_zone") or args.get("zone") or "")
        n_teams = int(args.get("n_teams", 1) or 1)
        gt_zone = env.ground_truth().get("entities", {}).get(zone_id, {})
        zone_buried = int(gt_zone.get("buried", 0) or 0)
        zone_population = int(gt_zone.get("population_initial", 1) or 1)
        criticality = float(gt_zone.get("criticality", 0.5) or 0.5)

        result = backend.apply_tool_effect(tool_name, args)

        # Trust transition only if dispatch was accepted.
        if result.get("_status") != "error" and env.stakeholders is not None:
            ev = trust_event_for_dispatch(
                responder_class,
                criticality,
                n_teams=n_teams,
                zone_buried=zone_buried,
                zone_population=zone_population,
            )
            env.stakeholders.record_event(
                group_id=responder_class,
                event=ev,
                tick=ctx.tick,
            )
            if env.evidence is not None:
                env.evidence.log(
                    "dispatch",
                    ctx.tick,
                    payload={**result, "trust_event": ev},
                    source="tool",
                )
                env.evidence.log(
                    "trust_event",
                    ctx.tick,
                    payload={
                        "group_id": responder_class,
                        "event": ev,
                        "source_tool": tool_name,
                        "zone": zone_id,
                        "n_teams": n_teams,
                    },
                    source="tool",
                )
        return result

    return handler


def _h_assign_triage(backend: Any, env: DisasterEnvironment):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        result = backend.apply_tool_effect("assign_triage", args)
        if result.get("_status") != "error" and env.evidence is not None:
            env.evidence.log(
                "triage_assignment",
                ctx.tick,
                payload={**result},
                source="tool",
            )
        return result

    return handler


def _h_evacuate_zone(backend: Any, env: DisasterEnvironment):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        zone_id = str(args.get("zone") or args.get("target_zone") or "")
        result = backend.apply_tool_effect("evacuate_zone", args)
        # Evacuation is a civilian-facing promise; tag as ``timely_response``
        # on the ``civilian`` group. The cross-zone fairness check lives in
        # the equity scorer, not here.
        if result.get("_status") != "error" and env.stakeholders is not None:
            ev = "timely_response"
            env.stakeholders.record_event("civilian", ev, ctx.tick)
            if env.evidence is not None:
                env.evidence.log(
                    "evacuation",
                    ctx.tick,
                    payload={**result, "trust_event": ev},
                    source="tool",
                )
                env.evidence.log(
                    "trust_event",
                    ctx.tick,
                    payload={
                        "group_id": "civilian",
                        "event": ev,
                        "source_tool": "evacuate_zone",
                        "zone": zone_id,
                    },
                    source="tool",
                )
        return result

    return handler


def _h_request_mutual_aid_team(backend: Any):
    """F-01 unified delayed-effect handler.

    Disaster reuses ``queue_mutual_aid_effect(due_tick, mw)`` — ``mw``
    here means "team_units" (the n_teams the caller requested). The
    ack pattern, ``due_tick=tick+delay_ticks``, and the
    ``_status='pending'`` payload all match power_grid byte-for-byte so
    cross-domain tests are uniform.
    """

    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        team_type = str(args.get("team_type", ""))
        source_region = str(args.get("source_region", ""))
        n_teams = int(args.get("n_teams", 1) or 1)
        delay_ticks = 2
        due_tick = int(
            ctx.extra.get("materialization_tick", int(ctx.tick) + delay_ticks)
        )
        if not hasattr(backend, "queue_mutual_aid_effect"):
            return {
                "_status": "error",
                "error": "backend_missing_queue_mutual_aid_effect",
                "backend": type(backend).__name__,
            }
        tool_call = ctx.extra.get("materializing_tool_call")
        queue_kwargs = {"due_tick": due_tick, "mw": float(n_teams)}
        if isinstance(tool_call, dict):
            queue_kwargs["tool_call"] = dict(tool_call)
        try:
            backend.queue_mutual_aid_effect(**queue_kwargs)
        except TypeError:
            backend.queue_mutual_aid_effect(due_tick=due_tick, mw=float(n_teams))
        return {
            "_status": "pending",
            "due_tick": due_tick,
            "team_type": team_type,
            "source_region": source_region,
            "n_teams": n_teams,
            "mw": float(n_teams),  # mirrored field so cross-domain tests pass
            "info": (
                "mutual-aid queued; physical team arrival enters the "
                "backend at due_tick"
            ),
        }

    return handler


# ``moral_choice`` / ``commit_to_plan`` handler factories moved to
# ``core.common_tools`` (domain-agnostic; see ``register_disaster_tools``
# above for the call sites: ``moral_choice_handler(env)`` /
# ``commit_to_plan_handler(env)``).
