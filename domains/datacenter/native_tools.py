"""Datacenter-native investigation and scheduling controls."""

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

if TYPE_CHECKING:
    from .adapter import DatacenterEnvironment


def register_datacenter_tools(
    registry: ToolRegistry,
    backend: Any,
    env: DatacenterEnvironment,
) -> None:
    if getattr(backend, "backend_kind", "") == "alibaba_openb_gpu_placement":
        _register_openb_tools(registry, backend, env)
        return
    plan_properties = plan_autonomy_properties()
    # This backend exposes a four-tick native review bound. Keep the shared
    # contract dynamic, while making the Datacenter tool surface validate the
    # backend-specific limit before a call is sent.
    plan_properties["review_after_ticks"] = {
        **plan_properties["review_after_ticks"],
        "maximum": 4,
    }
    registry.register(
        ToolSpec(
            name="query_job_queue",
            description=(
                "Inspect arrived GPU jobs, remaining duration, resource request, "
                "deadline and current queue policy. Future arrivals remain hidden."
            ),
            parameters={"type": "object", "properties": {}},
            handler=_query(backend, env, "queue"),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="job_queue",
            cost_units=1.0,
        )
    )
    registry.register(
        ToolSpec(
            name="query_cluster_capacity",
            description=(
                "Inspect available and allocated GPU/CPU capacity, including "
                "active service reservations or capacity reductions."
            ),
            parameters={"type": "object", "properties": {}},
            handler=_query(backend, env, "capacity"),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="cluster_capacity",
            cost_units=1.0,
        )
    )
    registry.register(
        ToolSpec(
            name="forecast_trace_arrivals",
            description=(
                "Return aggregate arrivals from the locked trace over a bounded "
                "future horizon; individual future job identities stay hidden."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "horizon_ticks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 4,
                    }
                },
                "required": ["horizon_ticks"],
            },
            handler=_forecast(backend, env),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="trace_arrivals",
            cost_units=1.0,
        )
    )
    registry.register(
        ToolSpec(
            name="set_queue_policy",
            description=(
                "Change the live cluster allocation order. Valid policies are "
                "fifo, shortest_job_first, least_gpu_first and "
                "deadline_criticality_first. The deadline policy orders arrived "
                "jobs by due slack, descending criticality, remaining duration, "
                "submit tick and job id; future arrivals stay hidden."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "policy": {
                        "type": "string",
                        "enum": [
                            "fifo",
                            "shortest_job_first",
                            "least_gpu_first",
                            "deadline_criticality_first",
                        ],
                    }
                },
                "required": ["policy"],
            },
            handler=_apply(backend, env, "set_queue_policy"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="queue_policy",
            actuator_family="queue_scheduler",
            # This is atomic local scheduler state, unlike a capacity
            # reservation whose resources arrive asynchronously.
            delay_ticks=0,
            cost_units=0.5,
        )
    )
    registry.register(
        ToolSpec(
            name="preempt_job",
            description=(
                "Preempt one running job and return it to the queue. Its progress "
                "is lost and charged, so use only when a critical arrival warrants it."
            ),
            parameters={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
            handler=_apply(backend, env, "preempt_job"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="running_job",
            actuator_family="running_job_pool",
            cost_units=1.0,
        )
    )
    registry.register(
        ToolSpec(
            name="review_persistent_policy",
            description=(
                "Review the active queue policy after one or more observed runtime "
                "events. Cite only event ids from prior observations and the "
                "currently active policy generation; this records supervision "
                "evidence without changing scheduler state."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "event_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "policy_generation": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "rationale": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
                "required": ["event_ids", "policy_generation", "rationale"],
            },
            handler=_apply(backend, env, "review_persistent_policy"),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="queue_policy",
            cost_units=0.5,
        )
    )
    registry.register(
        ToolSpec(
            name="reserve_gpu_capacity",
            description=(
                "Reserve temporary GPU capacity. The reservation arrives one tick "
                "later and incurs a per-unit capacity charge."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "gpu_units": {"type": "number", "minimum": 1.0},
                    "duration_ticks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 4,
                    },
                },
                "required": ["gpu_units", "duration_ticks"],
            },
            handler=_apply(backend, env, "reserve_gpu_capacity"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="gpu_capacity",
            actuator_family="gpu_capacity_pool",
            delay_ticks=1,
            cost_units=1.5,
        )
    )
    registry.register(
        ToolSpec(
            name="commit_to_plan",
            description="Record or revise an evidence-linked cluster scheduling plan.",
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
                    **plan_properties,
                },
                "required": ["plan_id"],
            },
            handler=commit_to_plan_handler(env, include_horizon_ticks=False),
            state_changing=False,
            semantic_role="planning",
            native_target_kind="scheduling_plan",
            cost_units=0.0,
        )
    )
    registry.register(
        wait_tool_spec(
            "Decline intervention for this interval; simulator time advances independently."
        )
    )
    registry.register(noop_tool_spec())


def _register_openb_tools(
    registry: ToolRegistry,
    backend: Any,
    env: DatacenterEnvironment,
) -> None:
    """Register the source-native OpenB placement and migration surface."""
    registry.register(
        ToolSpec(
            name="query_node_placements",
            description=(
                "Inspect visible OpenB pods, node CPU/memory allocation, physical GPU "
                "model compatibility, per-GPU milli-share fragmentation and assignments."
            ),
            parameters={"type": "object", "properties": {}},
            handler=_query_openb(backend, env),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="gpu_node_placement",
            cost_units=1.0,
        )
    )
    registry.register(
        ToolSpec(
            name="forecast_pod_arrivals",
            description=(
                "Return aggregate future pod arrivals from the locked OpenB trace; "
                "future pod identities remain hidden."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "horizon_ticks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 4,
                    }
                },
                "required": ["horizon_ticks"],
            },
            handler=_forecast(backend, env),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="openb_pod_arrivals",
            cost_units=1.0,
        )
    )
    registry.register(
        ToolSpec(
            name="set_placement_policy",
            description=(
                "Change the live OpenB placement policy. Valid policies are "
                "first_fit and fragmentation_aware."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "policy": {
                        "type": "string",
                        "enum": [
                            "first_fit",
                            "fragmentation_aware",
                        ],
                    }
                },
                "required": ["policy"],
            },
            handler=_apply(backend, env, "set_placement_policy"),
            state_changing=True,
            semantic_role="control",
            native_target_kind="placement_policy",
            actuator_family="placement_policy_controller",
            delay_ticks=0,
            cost_units=0.5,
        )
    )
    for name, description in (
        (
            "place_pod",
            "Place one currently queued OpenB pod on a compatible source node.",
        ),
        (
            "migrate_pod",
            "Migrate one running OpenB pod to a compatible source node and charge migration cost.",
        ),
    ):
        registry.register(
            ToolSpec(
                name=name,
                description=description,
                parameters={
                    "type": "object",
                    "properties": {
                        "pod_id": {"type": "string"},
                        "node_id": {"type": "string"},
                    },
                    "required": ["pod_id", "node_id"],
                },
                handler=_apply(backend, env, name),
                state_changing=True,
                semantic_role="control",
                native_target_kind="gpu_pod_assignment",
                actuator_family="node_placement_engine",
                delay_ticks=0,
                cost_units=1.0 if name == "place_pod" else 1.5,
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
            description="Record or revise an evidence-linked GPU placement plan.",
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
                    **plan_properties,
                },
                "required": ["plan_id"],
            },
            handler=commit_to_plan_handler(env, include_horizon_ticks=False),
            state_changing=False,
            semantic_role="planning",
            native_target_kind="placement_plan",
            cost_units=0.0,
        )
    )
    registry.register(
        wait_tool_spec(
            "Decline placement intervention for this interval; OpenB pod arrivals continue."
        )
    )
    registry.register(noop_tool_spec())


def _log(
    env: DatacenterEnvironment, tick: int, tool: str, payload: dict[str, Any]
) -> None:
    if env.evidence is not None:
        env.evidence.log(
            kind="investigation" if tool.startswith(("query", "forecast")) else tool,
            tick=tick,
            payload={"tool": tool, **payload},
            source="tool",
        )


def _query(backend: Any, env: DatacenterEnvironment, target: str):
    def handler(_args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        result = (
            backend.queue_state() if target == "queue" else backend.capacity_state()
        )
        _log(env, ctx.tick, f"query_{target}", result)
        return result

    return handler


def _query_openb(backend: Any, env: DatacenterEnvironment):
    def handler(_args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        result = backend.placement_state()
        _log(env, ctx.tick, "query_node_placements", result)
        return result

    return handler


def _forecast(backend: Any, env: DatacenterEnvironment):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        result = backend.arrival_forecast(int(args.get("horizon_ticks") or 1))
        _log(env, ctx.tick, "forecast_trace_arrivals", result)
        return result

    return handler


def _apply(backend: Any, env: DatacenterEnvironment, name: str):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        result = backend.apply_tool_effect(name, args, current_tick=ctx.tick)
        if result.get("_status") != "error":
            _log(env, ctx.tick, name, result)
        return result

    return handler
