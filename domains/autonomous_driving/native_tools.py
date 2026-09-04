"""Tactical and read-only tools for autonomous-driving supervision."""

from __future__ import annotations

from collections.abc import Callable
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

if TYPE_CHECKING:
    from .adapter import AutonomousDrivingEnvironment


def _backend_handler(
    backend: Any,
    method_name: str,
) -> Callable[[dict[str, Any], ToolContext], dict[str, Any]]:
    def handler(args: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        method = getattr(backend, method_name, None)
        if not callable(method):
            return {
                "_status": "error",
                "error": f"backend_missing_{method_name}",
            }
        if method_name.startswith("inspect_"):
            payload = method()
            if method_name == "inspect_local_scene":
                fog = context.extra.get("fog")
                mark_revealed = getattr(fog, "mark_revealed", None)
                if callable(mark_revealed):
                    for actor in payload.get("actors") or []:
                        mark_revealed(
                            str(actor.get("actor_id") or ""),
                            ["relative_speed_mps"],
                        )
            record_investigation = getattr(backend, "record_investigation", None)
            if callable(record_investigation):
                record_investigation(method_name)
            return payload
        return method(args)

    return handler


def register_autonomous_driving_tools(
    registry: ToolRegistry,
    backend: Any,
    env: AutonomousDrivingEnvironment,
) -> None:
    """Register only observations and tactical intents, never raw actuators."""
    for name, description, target in (
        (
            "inspect_ego_state",
            "Inspect the current ego-vehicle kinematic state.",
            "ego_vehicle_state",
        ),
        (
            "inspect_local_scene",
            "Inspect currently observable nearby road actors and lane geometry.",
            "local_driving_scene",
        ),
        (
            "inspect_odd_status",
            "Inspect operational-design-domain and sensing availability.",
            "operational_design_domain",
        ),
        (
            "inspect_safety_state",
            "Inspect runtime-assurance mode and current safety margins.",
            "runtime_assurance_state",
        ),
    ):
        registry.register(
            ToolSpec(
                name=name,
                description=description,
                parameters={"type": "object", "properties": {}},
                handler=_backend_handler(backend, name),
                state_changing=False,
                semantic_role="investigation",
                native_target_kind=target,
                cost_units=0.5,
            )
        )

    registry.register(
        ToolSpec(
            name="set_driving_envelope",
            description=(
                "Set bounded tactical speed and following constraints. The "
                "backend controller and safety shield retain low-level control."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "target_speed_min_mps": {
                        "type": "number",
                        "minimum": 0.0,
                    },
                    "target_speed_max_mps": {
                        "type": "number",
                        "minimum": 0.0,
                    },
                    "min_time_headway_s": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 10.0,
                    },
                    "max_acceleration_mps2": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 4.0,
                    },
                    "max_deceleration_mps2": {
                        "type": "number",
                        "minimum": 0.1,
                        "maximum": 10.0,
                    },
                    "command_sequence": {"type": "integer", "minimum": 1},
                    "expires_at_tick": {"type": "integer", "minimum": 0},
                },
                "required": [
                    "target_speed_min_mps",
                    "target_speed_max_mps",
                    "command_sequence",
                    "expires_at_tick",
                ],
            },
            handler=_backend_handler(backend, "set_driving_envelope"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="tactical_driving_envelope",
            actuator_family="supervisory_envelope",
            delay_ticks=0,
            cost_units=1.0,
        )
    )
    registry.register(
        ToolSpec(
            name="request_tactical_maneuver",
            description=(
                "Request a bounded tactical maneuver. The backend may reject, "
                "clip, or safely execute it; no steering value is accepted."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "maneuver": {
                        "type": "string",
                        "enum": [
                            "keep_lane",
                            "change_lane_left",
                            "change_lane_right",
                            "slow_for_hazard",
                        ],
                    },
                    "target_lane": {"type": "integer", "minimum": 0},
                    "command_sequence": {"type": "integer", "minimum": 1},
                    "expires_at_tick": {"type": "integer", "minimum": 0},
                },
                "required": ["maneuver", "command_sequence", "expires_at_tick"],
            },
            handler=_backend_handler(backend, "request_tactical_maneuver"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="ego_tactical_maneuver",
            actuator_family="tactical_maneuver_request",
            delay_ticks=0,
            cost_units=1.0,
        )
    )
    registry.register(
        ToolSpec(
            name="request_minimal_risk_maneuver",
            description=(
                "Request the backend runtime-assurance system to transition "
                "toward a minimal-risk condition."
            ),
            parameters={
                "type": "object",
                "properties": {"reason": {"type": "string", "minLength": 1}},
                "required": ["reason"],
            },
            handler=_backend_handler(backend, "request_minimal_risk_maneuver"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="minimal_risk_maneuver",
            actuator_family="runtime_assurance_mode_request",
            delay_ticks=0,
            cost_units=1.0,
        )
    )
    registry.register(
        ToolSpec(
            name="request_recovery_check",
            description=(
                "Ask the backend to check recovery readiness and, only when "
                "eligible, issue the episode-scoped recovery token."
            ),
            parameters={"type": "object", "properties": {}},
            handler=_backend_handler(backend, "request_recovery_check"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="runtime_assurance_recovery_readiness",
            actuator_family="recovery_token_issuance",
            delay_ticks=0,
            cost_units=0.5,
        )
    )
    registry.register(
        ToolSpec(
            name="authorize_recovery",
            description=(
                "Authorize recovery from a minimal-risk condition using an "
                "token previously issued by request_recovery_check; runtime "
                "assurance still validates the health dwell."
            ),
            parameters={
                "type": "object",
                "properties": {"recovery_token": {"type": "string", "minLength": 1}},
                "required": ["recovery_token"],
            },
            handler=_backend_handler(backend, "authorize_recovery"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="runtime_assurance_recovery",
            actuator_family="mode_recovery_authorization",
            delay_ticks=0,
            cost_units=1.0,
        )
    )

    plan_properties = plan_autonomy_properties()
    plan_properties["review_after_ticks"] = {
        **plan_properties["review_after_ticks"],
        "maximum": 2,
    }
    registry.register(
        ToolSpec(
            name="commit_to_plan",
            description="Record or revise a bounded tactical supervision plan.",
            parameters={
                "type": "object",
                "properties": {
                    "plan_id": {"type": "string"},
                    "rationale": {"type": "string"},
                    "replaces_plan_id": {"type": "string"},
                    "revision_reason": {"type": "string"},
                    **plan_properties,
                },
                "required": ["plan_id", "rationale"],
            },
            handler=commit_to_plan_handler(env, include_horizon_ticks=False),
            state_changing=False,
            semantic_role="planning",
            native_target_kind="standing_plan",
            cost_units=0.0,
        )
    )
    registry.register(
        wait_tool_spec(
            "Decline tactical intervention; backend control and the safety shield continue."
        )
    )
    registry.register(noop_tool_spec())
