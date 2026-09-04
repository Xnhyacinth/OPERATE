"""Classify power-grid tool calls and summarize decision impact vs wait-only CF."""

from __future__ import annotations

from typing import Any

from core.tool_protocol import ToolRegistry

# Read-only / non-physics information-gathering tools (real investigation work).
# v0.2.2: ``wait`` and ``noop`` are explicitly NOT investigation; they are
# meta no-op signals (see ``META_TOOL_NAMES``). A ``wait_only`` agent must
# not register any investigation calls.
INVESTIGATION_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "query_grid_state",
        "query_chronics_window",
        "forecast_query",
        "investigate_substation",
        "stakeholder_query",
        "query_active_dilemmas",
        # v0.7 logistics-native investigation/meta tools — read-only, paid,
        # noised; they must NOT earn adaptive_replanning active-recovery credit.
        "query_eta",
        "forecast_demand",
        # v0.7 microgrid-native read-only investigation tool (forecast_query
        # is already listed above; reused unchanged).
        "investigate_asset",
        # v0.7 disaster-native read-only investigation tools — survey/recon
        # reveal hidden hazards (recon is delayed fog); forecast_aftershock is
        # a noised forecast. They must NOT earn adaptive_replanning credit.
        "survey_zone",
        "dispatch_recon",
        "forecast_aftershock",
        # v0.7 traffic-native read-only investigation tools — they reveal
        # fog-hidden queue/delay attributes (query_network_state /
        # query_detector / inspect_intersection). They must NOT earn
        # adaptive_replanning active-recovery credit.
        "query_network_state",
        "query_detector",
        "inspect_intersection",
        # v0.52 native traffic control-surface readback.  This is a
        # state-observation tool, not a state-changing action, so it belongs
        # in the investigation bucket for decision-impact accounting.
        "query_signal_control",
        # Datacenter-native read-only scheduling observations.
        "query_job_queue",
        "query_cluster_capacity",
        "forecast_trace_arrivals",
        "review_persistent_policy",
    }
)

# Meta / no-op tools (do nothing, change nothing, learn nothing).
# Splitting these out of ``INVESTIGATION_TOOL_NAMES`` is the v0.2.2 fix that
# stops ``wait_only`` episodes from being labelled "investigation_only".
META_TOOL_NAMES: frozenset[str] = frozenset({"wait", "noop"})

# Tools that can change simulator state or operational posture.
CONTROL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "redispatch_generation",
        "commit_reserve",
        "shed_load",
        "switch_branch",
        "topology_action",
        "request_mutual_aid",
        "negotiate_with_stakeholder",
        "escalate_to_human",
        "moral_choice",
        "commit_to_plan",
        # v0.7 logistics-native state-changing dispatch/order tools.
        "assign_stop",
        "reroute_vehicle",
        "dispatch_vehicle",
        "hire_spot_carrier",
        "hold_order",
        "drop_order",
        "dispatch_ready_operations",
        "dispatch_job_operation",
        "place_replenishment_order",
        # v0.7 microgrid-native state-changing EMS tools (set_battery_dispatch
        # / set_der_reactive_power are reused distribution-native names; the
        # rest are microgrid-native). shed_load / commit_to_plan are already
        # listed above and reused unchanged.
        "set_transformer_tap",
        "switch_capacitor",
        "set_battery_dispatch",
        "set_der_reactive_power",
        "dispatch_genset",
        "set_grid_exchange",
        "connect_pcc",
        "curtail_der",
        # v0.7 disaster-native state-changing response tools (moral_choice /
        # commit_to_plan are already listed above and reused unchanged).
        "dispatch_ambulance",
        "dispatch_fire_brigade",
        "dispatch_police_cordon",
        "assign_triage",
        "evacuate_zone",
        "request_mutual_aid_team",
        # v0.7 traffic-native state-changing control tools (moral_choice /
        # commit_to_plan are already listed above and reused unchanged).
        "change_signal_plan",
        "extend_current_green_phase",
        "reroute_flow",
        "close_lane",
        "meter_inflow",
        "dispatch_emergency_priority",
        "request_incident_response_team",
        # v0.52 live-SUMO state-changing control surface.
        "set_signal_program",
        "set_signal_phase_duration",
        # Datacenter-native scheduler controls.
        "set_queue_policy",
        "preempt_job",
        "reserve_gpu_capacity",
    }
)

PLANNING_TOOL_NAMES: frozenset[str] = frozenset({"commit_to_plan"})
COMMUNICATION_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "stakeholder_query",
        "negotiate_with_stakeholder",
        "escalate_to_human",
        "moral_choice",
    }
)

_COST_EPSILON = 1.0


def classify_tool_semantic_histogram(
    tool_histogram: dict[str, int],
    *,
    registry: ToolRegistry | None = None,
    formal: bool = False,
) -> dict[str, int | list[str]]:
    """Classify calls into the five semantic roles plus unknown.

    Live evaluation should pass the environment's registry so ToolSpec metadata
    is authoritative. The name sets remain a compatibility path for historical
    trajectories that did not persist their registered specs.
    """
    if registry is not None:
        registry.validate_semantic_coverage(set(tool_histogram), formal=formal)
        roles = registry.semantic_roles()
    else:
        roles = {}
        for name in tool_histogram:
            if name in META_TOOL_NAMES:
                roles[name] = "meta"
            elif name in PLANNING_TOOL_NAMES:
                roles[name] = "planning"
            elif name in COMMUNICATION_TOOL_NAMES:
                roles[name] = "communication"
            elif name in INVESTIGATION_TOOL_NAMES:
                roles[name] = "investigation"
            elif name in CONTROL_TOOL_NAMES:
                roles[name] = "control"

    counts = {
        "investigation": 0,
        "control": 0,
        "planning": 0,
        "communication": 0,
        "meta": 0,
    }
    unknown_names: list[str] = []
    unknown_calls = 0
    for name, count in tool_histogram.items():
        role = roles.get(name)
        if role in counts:
            counts[role] += count
        else:
            unknown_names.append(name)
            unknown_calls += count
    unknown_names.sort()
    if formal and unknown_names:
        raise ValueError(f"unknown_tool_semantics: {', '.join(unknown_names)}")
    return {
        "n_investigation_calls": counts["investigation"],
        "n_control_calls": counts["control"],
        "n_planning_calls": counts["planning"],
        "n_communication_calls": counts["communication"],
        "n_meta_calls": counts["meta"],
        "n_unknown_calls": unknown_calls,
        "unknown_tool_names": unknown_names,
    }


def classify_tool_histogram(
    tool_histogram: dict[str, int],
    *,
    registry: ToolRegistry | None = None,
    formal: bool = False,
) -> dict[str, int]:
    """Split counts into legacy decision-impact buckets.

    Planning and communication remain in ``other`` for API compatibility.
    When a registry is supplied, its ToolSpec semantics replace name matching.
    """
    if registry is not None:
        semantic = classify_tool_semantic_histogram(
            tool_histogram, registry=registry, formal=formal
        )
        return {
            "n_investigation_calls": int(semantic["n_investigation_calls"]),
            "n_control_calls": int(semantic["n_control_calls"]),
            "n_meta_calls": int(semantic["n_meta_calls"]),
            "n_other_calls": (
                int(semantic["n_planning_calls"])
                + int(semantic["n_communication_calls"])
                + int(semantic["n_unknown_calls"])
            ),
        }
    inv = sum(tool_histogram.get(n, 0) for n in INVESTIGATION_TOOL_NAMES)
    ctrl = sum(tool_histogram.get(n, 0) for n in CONTROL_TOOL_NAMES)
    meta = sum(tool_histogram.get(n, 0) for n in META_TOOL_NAMES)
    other = sum(tool_histogram.values()) - inv - ctrl - meta
    return {
        "n_investigation_calls": inv,
        "n_control_calls": ctrl,
        "n_meta_calls": meta,
        "n_other_calls": max(0, other),
    }


def summarize_decision_impact(
    tool_histogram: dict[str, int],
    counterfactual: dict[str, Any] | None,
    *,
    tool_results_ok: int = 0,
    tool_results_failed: int = 0,
) -> dict[str, Any]:
    """Whether the agent's decisions measurably changed episode cost vs wait-only."""
    cf = counterfactual or {}
    prevented = float(cf.get("prevented_loss", 0.0))
    actual = float(cf.get("actual_cost", 0.0))
    cf_cost = float(cf.get("counterfactual_cost", 0.0))
    norm = float(cf.get("normalized_prevention", 0.0))

    buckets = classify_tool_histogram(tool_histogram)
    n_control = buckets["n_control_calls"]
    n_investigation = buckets["n_investigation_calls"]
    outcome_changed = abs(prevented) > _COST_EPSILON
    helped = prevented > _COST_EPSILON
    hurt = prevented < -_COST_EPSILON

    return {
        **buckets,
        "actual_cost": actual,
        "counterfactual_cost": cf_cost,
        "prevented_loss": prevented,
        "normalized_prevention": norm,
        "outcome_changed": outcome_changed,
        "agent_helped": helped,
        "agent_hurt": hurt,
        # An episode is "investigation-only" if the agent really did
        # information-gathering work but took no control actions. A pure
        # wait/noop episode is NOT investigation-only — that's a no-op
        # episode and is classified as such by ``_interpret()`` below.
        "investigation_only_episode": n_control == 0 and n_investigation > 0,
        "tool_results_ok": tool_results_ok,
        "tool_results_failed": tool_results_failed,
        "interpretation": _interpret(
            n_control=n_control,
            n_investigation=n_investigation,
            outcome_changed=outcome_changed,
            helped=helped,
            hurt=hurt,
            tool_results_failed=tool_results_failed,
        ),
    }


def _interpret(
    *,
    n_control: int,
    n_investigation: int,
    outcome_changed: bool,
    helped: bool,
    hurt: bool,
    tool_results_failed: int,
) -> str:
    if tool_results_failed > 0:
        return "some_tool_executions_failed"
    if n_control == 0 and not outcome_changed:
        # v0.2.2: separate pure-meta no-op from real investigation-only.
        if n_investigation > 0:
            return "investigation_only_no_cost_delta_vs_wait_only"
        return "no_action_taken_meta_only"
    if n_control > 0 and not outcome_changed:
        return "control_tools_called_but_cost_matches_wait_only"
    if helped:
        return "control_actions_reduced_cost_vs_wait_only"
    if hurt:
        return "control_actions_increased_cost_vs_wait_only"
    return "unchanged"
