"""CityLearn-native tools for Building Energy candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from core import (
    ToolRegistry,
    ToolSpec,
    commit_to_plan_handler,
    noop_tool_spec,
    plan_autonomy_properties,
    wait_tool_spec,
)

if TYPE_CHECKING:
    from .adapter import BuildingEnergyEnvironment


def register_building_energy_tools(
    registry: ToolRegistry, backend: Any, env: BuildingEnergyEnvironment
) -> None:
    building_ids = backend.buildings
    seed_obj = getattr(env, "seed_obj", None)
    backend_config = getattr(seed_obj, "backend_config", {}) or {}
    storage_control_delay_ticks = int(
        backend_config.get("storage_control_delay_ticks") or 0
    )
    if not 0 <= storage_control_delay_ticks <= 4:
        raise ValueError("CityLearn storage_control_delay_ticks must be in [0, 4]")
    building_schema: dict[str, Any] = {"type": "string", "enum": building_ids}

    registry.register(
        ToolSpec(
            name="inspect_building_state",
            description=(
                "Inspect current native CityLearn storage, net-load and capacity "
                "state for one building or the full bounded cluster."
            ),
            parameters={
                "type": "object",
                "properties": {"building_id": dict(building_schema)},
            },
            handler=lambda args, ctx: backend.inspect_building_state(
                args.get("building_id")
            ),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="building_energy_state",
            cost_units=0.5,
        )
    )
    registry.register(
        ToolSpec(
            name="set_storage_dispatch",
            description=(
                "Set one or more buildings' native electrical-storage rates in one "
                "atomic simulator receipt. Use building_id/rate for one endpoint or "
                "dispatches for a sparse vector. Rates are bounded to [-1, 1]; "
                "positive charges and negative discharges."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "building_id": dict(building_schema),
                    "rate": {"type": "number", "minimum": -1.0, "maximum": 1.0},
                    "dispatches": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": len(building_ids),
                        "items": {
                            "type": "object",
                            "properties": {
                                "building_id": dict(building_schema),
                                "rate": {
                                    "type": "number",
                                    "minimum": -1.0,
                                    "maximum": 1.0,
                                },
                            },
                            "required": ["building_id", "rate"],
                        },
                    },
                },
                "anyOf": [
                    {"required": ["building_id", "rate"]},
                    {"required": ["dispatches"]},
                ],
            },
            handler=lambda args, ctx: (
                backend.queue_storage_rates(args["dispatches"])
                if isinstance(args.get("dispatches"), list)
                else backend.queue_storage_rate(
                    str(args.get("building_id") or ""), float(args.get("rate"))
                )
            ),
            state_changing=True,
            semantic_role="control",
            native_target_kind="electrical_storage",
            actuator_family="storage_dispatch",
            # The default preserves the 24-hour pilot. Long-horizon candidates
            # may source-declare a bounded protocol delay; ToolRegistry invokes
            # the native handler only at its due simulator tick.
            delay_ticks=storage_control_delay_ticks,
            cost_units=0.5,
        )
    )
    plan_properties = plan_autonomy_properties()
    plan_properties["review_after_ticks"] = {
        **plan_properties["review_after_ticks"],
        "maximum": 4,
    }
    registry.register(
        ToolSpec(
            name="commit_to_plan",
            description="Record or revise a standing building-energy supervision plan.",
            parameters={
                "type": "object",
                "properties": {
                    "plan_id": {"type": "string"},
                    "rationale": {"type": "string"},
                    "replaces_plan_id": {"type": "string"},
                    "revision_reason": {"type": "string"},
                    **plan_properties,
                },
                "required": ["plan_id"],
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
            "Decline intervention for this interval; CityLearn advances independently."
        )
    )
    registry.register(noop_tool_spec())
