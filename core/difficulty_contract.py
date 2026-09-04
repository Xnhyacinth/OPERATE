"""Machine-checkable four-level difficulty contract.

The contract deliberately separates scenario configuration from observed
behavior.  A label is not accepted merely because a YAML contains more tools
or more fog: Medium and above require deterministic replay evidence, while
High and Extreme additionally require an evidence/action dependency DAG.
"""

from __future__ import annotations

from collections.abc import Collection
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

DIFFICULTY_CONTRACT_VERSION = "source_grounded_behavioral_v3"


@dataclass(frozen=True)
class DifficultyRequirements:
    min_horizon_ticks: int
    min_task_milestones: int
    min_perturbations: int
    min_visible_perturbations: int
    min_hidden_perturbations: int
    min_perturbation_kinds: int
    min_event_span_ticks: int
    min_effective_ticks: int
    min_physical_tools: int
    min_strategy_switches: int
    min_interaction_turns: int
    min_dependency_depth: int


def difficulty_calibration_matches_level(
    calibration: dict[str, Any],
    level: str,
) -> bool:
    """Return whether replay calibration is current and bound to ``level``."""

    return bool(
        calibration.get("version") == DIFFICULTY_CONTRACT_VERSION
        and calibration.get("status") == "passed"
        and calibration.get("declared_level_matches_evidence") is True
        and calibration.get("declared_difficulty_level") == level
        and calibration.get("calibrated_difficulty_level") == level
    )


DIFFICULTY_REQUIREMENTS = {
    "basic": DifficultyRequirements(
        min_horizon_ticks=3,
        min_task_milestones=1,
        min_perturbations=0,
        min_visible_perturbations=0,
        min_hidden_perturbations=0,
        min_perturbation_kinds=0,
        min_event_span_ticks=0,
        min_effective_ticks=1,
        min_physical_tools=1,
        min_strategy_switches=0,
        min_interaction_turns=1,
        min_dependency_depth=1,
    ),
    "medium": DifficultyRequirements(
        min_horizon_ticks=6,
        min_task_milestones=1,
        min_perturbations=1,
        min_visible_perturbations=1,
        min_hidden_perturbations=0,
        min_perturbation_kinds=1,
        min_event_span_ticks=0,
        min_effective_ticks=2,
        min_physical_tools=1,
        min_strategy_switches=0,
        min_interaction_turns=2,
        min_dependency_depth=1,
    ),
    "high": DifficultyRequirements(
        min_horizon_ticks=6,
        min_task_milestones=2,
        min_perturbations=2,
        min_visible_perturbations=1,
        min_hidden_perturbations=1,
        min_perturbation_kinds=2,
        min_event_span_ticks=2,
        min_effective_ticks=2,
        min_physical_tools=2,
        min_strategy_switches=1,
        min_interaction_turns=3,
        min_dependency_depth=2,
    ),
    "extreme": DifficultyRequirements(
        min_horizon_ticks=10,
        min_task_milestones=3,
        min_perturbations=3,
        min_visible_perturbations=1,
        min_hidden_perturbations=1,
        min_perturbation_kinds=3,
        min_event_span_ticks=4,
        min_effective_ticks=3,
        min_physical_tools=2,
        min_strategy_switches=2,
        min_interaction_turns=4,
        min_dependency_depth=3,
    ),
}

_NON_EVENT_KINDS = frozenset({"storm_window"})
_STATIC_SCHEDULING_FAMILIES = frozenset(
    {"inventory_replenishment", "job_shop_dispatch"}
)
_STATIC_CONSTRAINT_AXES = {
    "inventory_replenishment": (
        "n_stages",
        "lead_time_days",
        "forecast_horizon",
        "demand_to_capacity_ratio",
    ),
    "job_shop_dispatch": (
        "n_jobs",
        "n_machines",
        "machine_conflict_density",
        "processing_time_density",
    ),
}
_STATIC_PLANNING_DEPTH_FLOORS = {
    "basic": 1,
    "medium": 2,
    "high": 3,
    "extreme": 4,
}
_STATIC_CONSTRAINT_BREADTH_FLOORS = {
    "basic": 1,
    "medium": 2,
    "high": 2,
    "extreme": 3,
}
_ORDERED_MILESTONE_DEPTH_PROOF = "task_contract_ordered_milestone_lower_bound"
_BOUNDED_REPLAY_STATUSES = frozenset(
    {"replay_budget_exhausted", "bounded_successful_upper_bound", "proved_lower_bound"}
)


def _task_milestones(scenario: dict[str, Any]) -> list[int]:
    config = scenario.get("backend_config") or {}
    profiles = [
        config.get("task_contract_profile"),
        config.get("task_contract") if isinstance(config.get("task_contract"), dict) else None,
    ]
    for profile in profiles:
        if not isinstance(profile, dict):
            continue
        for key in ("milestone_ticks", "phase_ticks"):
            values = profile.get(key)
            if isinstance(values, list) and values:
                return sorted({int(value) for value in values})
    event_ticks = sorted(
        {
            int(row.get("trigger_tick") or 0)
            for row in scenario.get("perturbations") or []
            if str(row.get("kind") or "") not in _NON_EVENT_KINDS
        }
    )
    if event_ticks:
        return event_ticks
    return [0]


def _configuration_features(scenario: dict[str, Any]) -> dict[str, Any]:
    perturbations = [
        row
        for row in scenario.get("perturbations") or []
        if str(row.get("kind") or "") not in _NON_EVENT_KINDS
    ]
    ticks = [int(row.get("trigger_tick") or 0) for row in perturbations]
    hidden = sum(bool(row.get("hidden")) for row in perturbations)
    complexity = scenario.get("complexity_metrics") or {}
    constraint_keys = _STATIC_CONSTRAINT_AXES.get(
        str(scenario.get("family") or ""),
        (),
    )
    task_milestones = _task_milestones(scenario)
    return {
        "horizon_ticks": int(scenario.get("horizon_ticks") or 0),
        "task_milestones": task_milestones,
        "n_task_milestones": len(task_milestones),
        "n_perturbations": len(perturbations),
        "n_visible_perturbations": len(perturbations) - hidden,
        "n_hidden_perturbations": hidden,
        "n_perturbation_kinds": len(
            {str(row.get("kind") or "") for row in perturbations}
        ),
        "event_span_ticks": max(ticks) - min(ticks) if ticks else 0,
        "observability_burden": int(
            complexity.get("observability_burden") or 0
        ),
        "planning_depth": int(complexity.get("decision_depth") or 0),
        "constraint_breadth": sum(
            float(complexity.get(key) or 0.0) > 0.0
            for key in constraint_keys
        ),
        "constraint_axes": [
            key
            for key in constraint_keys
            if float(complexity.get(key) or 0.0) > 0.0
        ],
        "source_task_size": int(
            complexity.get("n_operations")
            or complexity.get("n_periods")
            or scenario.get("horizon_ticks")
            or 0
        ),
    }


def _behavior_features(
    trajectory_complexity: dict[str, Any] | None,
    replay_minimality: dict[str, Any] | None,
    physical_tool_names: Collection[str] | None,
) -> dict[str, Any]:
    complexity = trajectory_complexity or {}
    minimality = replay_minimality or {}
    allowed = set(physical_tool_names or ())
    observed_tools = set(
        complexity.get("observed_physical_actuator_endpoint_set")
        or complexity.get("observed_state_changing_tool_set")
        or []
    )
    minimal_tools = set(
        minimality.get("one_minimal_physical_actuator_endpoint_set")
        or minimality.get("one_minimal_successful_tool_set")
        or []
    )
    exact_dependency_depth = complexity.get("exact_dependency_depth")
    if exact_dependency_depth is None:
        exact_dependency_depth = minimality.get("exact_dependency_depth")
        dependency_depth_status = minimality.get("dependency_depth_status")
    else:
        dependency_depth_status = complexity.get("dependency_depth_status")
    if allowed:
        observed_tools &= allowed
        minimal_tools &= allowed
    successful_tools = set(
        minimality.get(
            "successful_physical_actuator_endpoint_set_upper_bound"
        )
        or minimality.get("successful_tool_set_upper_bound")
        or []
    )
    if not successful_tools:
        successful_tools = set(minimality.get("one_minimal_successful_tool_set") or [])
    if allowed:
        successful_tools &= allowed
    successful_ticks = minimality.get("successful_decision_tick_upper_bound")
    if successful_ticks is None:
        successful_ticks = minimality.get("one_minimal_decision_ticks")
    required_depth_lower_bound = complexity.get("required_depth_lower_bound")
    if required_depth_lower_bound is None:
        required_depth_lower_bound = minimality.get("required_depth_lower_bound")
    effective_ticks = {
        int(value) for value in complexity.get("effective_control_ticks") or []
    }
    if minimality.get("status") == "one_minimal":
        effective_ticks.update(
            int(value)
            for value in minimality.get("one_minimal_decision_ticks") or []
        )
    return {
        "effective_control_ticks": sorted(effective_ticks),
        "effective_control_tick_proof": (
            "trajectory_or_one_minimal_replay"
            if minimality.get("status") == "one_minimal"
            else "trajectory_attribution_only"
        ),
        "observed_physical_tools": sorted(observed_tools),
        "control_strategy_switch_count": int(
            complexity.get("control_strategy_switch_count") or 0
        ),
        "actual_interaction_turns": int(
            complexity.get("actual_interaction_turns") or 0
        ),
        "exact_dependency_depth": exact_dependency_depth,
        "dependency_depth_status": dependency_depth_status,
        "required_depth_lower_bound": (
            int(required_depth_lower_bound)
            if required_depth_lower_bound is not None
            else None
        ),
        "depth_proof_kind": (
            complexity.get("depth_proof_kind")
            or minimality.get("depth_proof_kind")
        ),
        "one_minimal_status": minimality.get("status"),
        "one_minimal_decision_ticks": sorted(
            {int(value) for value in minimality.get("one_minimal_decision_ticks") or []}
        ),
        "one_minimal_physical_tools": sorted(minimal_tools),
        "successful_decision_tick_upper_bound": sorted(
            {int(value) for value in successful_ticks or []}
        ),
        "successful_tool_set_upper_bound": sorted(successful_tools),
    }


def evaluate_difficulty_contract(
    scenario: dict[str, Any],
    *,
    trajectory_complexity: dict[str, Any] | None = None,
    replay_minimality: dict[str, Any] | None = None,
    physical_tool_names: Collection[str] | None = None,
    require_behavior: bool = False,
    require_minimality: bool = False,
) -> dict[str, Any]:
    """Evaluate static and replay-backed difficulty evidence.

    ``pending`` is used only when the caller intentionally performs a static
    preflight.  Once behavior or minimality is required, missing evidence is a
    failure and the candidate is held.
    """
    level = str(scenario.get("difficulty_level") or "")
    requirements = DIFFICULTY_REQUIREMENTS.get(level)
    if requirements is None:
        return {
            "version": DIFFICULTY_CONTRACT_VERSION,
            "difficulty_level": level,
            "status": "held",
            "checks": {"canonical_four_level_difficulty": False},
            "failures": ["canonical_four_level_difficulty"],
            "pending": [],
        }

    track = (
        "static_scheduling"
        if str(scenario.get("family") or "") in _STATIC_SCHEDULING_FAMILIES
        and not scenario.get("perturbations")
        else "adaptive_control"
    )
    static = _configuration_features(scenario)
    behavior = _behavior_features(
        trajectory_complexity,
        replay_minimality,
        physical_tool_names,
    )
    checks = {
        "canonical_four_level_difficulty": True,
        "horizon_floor": static["horizon_ticks"] >= requirements.min_horizon_ticks,
    }
    if track == "static_scheduling":
        checks.update(
            {
                "adaptive_control_required_for_high_extreme": level
                not in {"high", "extreme"},
                "planning_depth_floor": (
                    static["planning_depth"]
                    >= _STATIC_PLANNING_DEPTH_FLOORS[level]
                ),
                "constraint_breadth_floor": (
                    static["constraint_breadth"]
                    >= _STATIC_CONSTRAINT_BREADTH_FLOORS[level]
                ),
                "source_task_size_floor": (
                    static["source_task_size"]
                    >= requirements.min_horizon_ticks
                ),
            }
        )
    else:
        checks.update(
            {
                "task_milestone_floor": (
                    static["n_task_milestones"]
                    >= requirements.min_task_milestones
                ),
                "perturbation_floor": (
                    static["n_perturbations"]
                    >= requirements.min_perturbations
                ),
                "visible_perturbation_floor": (
                    static["n_visible_perturbations"]
                    >= requirements.min_visible_perturbations
                ),
                "hidden_perturbation_floor": (
                    static["n_hidden_perturbations"]
                    >= requirements.min_hidden_perturbations
                    or (
                        requirements.min_hidden_perturbations > 0
                        and static["observability_burden"] > 0
                    )
                ),
                "perturbation_kind_floor": (
                    static["n_perturbation_kinds"]
                    >= requirements.min_perturbation_kinds
                ),
                "event_span_floor": (
                    static["event_span_ticks"]
                    >= requirements.min_event_span_ticks
                ),
            }
        )
    pending: list[str] = []
    if trajectory_complexity is None and not require_behavior:
        pending.extend(
            [
                "observed_effective_tick_floor",
                "observed_physical_tool_floor",
                "strategy_switch_floor",
                "interaction_turn_floor",
                "exact_dependency_depth_floor",
            ]
        )
    else:
        exact_depth = behavior["exact_dependency_depth"]
        exact_depth_proof = bool(
            exact_depth is not None
            and behavior["dependency_depth_status"]
            in {
                "declared_evidence_action_dag",
                "one_minimal_task_contract_phase_dag",
                "one_minimal_single_stage_action_dag",
            }
            and int(exact_depth) >= requirements.min_dependency_depth
        )
        bounded_depth_proof = bool(
            behavior["dependency_depth_status"]
            == _ORDERED_MILESTONE_DEPTH_PROOF
            and behavior["depth_proof_kind"]
            == _ORDERED_MILESTONE_DEPTH_PROOF
            and behavior["required_depth_lower_bound"] is not None
            and int(behavior["required_depth_lower_bound"])
            >= requirements.min_dependency_depth
            and behavior["one_minimal_status"] in _BOUNDED_REPLAY_STATUSES
            and behavior["successful_decision_tick_upper_bound"]
            and behavior["successful_tool_set_upper_bound"]
        )
        checks.update(
            {
                "observed_effective_tick_floor": (
                    len(behavior["effective_control_ticks"])
                    >= requirements.min_effective_ticks
                ),
                "observed_physical_tool_floor": (
                    len(behavior["observed_physical_tools"])
                    >= requirements.min_physical_tools
                ),
                "strategy_switch_floor": (
                    behavior["control_strategy_switch_count"]
                    >= requirements.min_strategy_switches
                ),
                "interaction_turn_floor": (
                    behavior["actual_interaction_turns"]
                    >= requirements.min_interaction_turns
                ),
                "exact_dependency_depth_floor": (
                    exact_depth is not None
                    and behavior["dependency_depth_status"]
                    in {
                        "declared_evidence_action_dag",
                        "one_minimal_task_contract_phase_dag",
                        "one_minimal_single_stage_action_dag",
                    }
                    and int(exact_depth) >= requirements.min_dependency_depth
                ),
                # A bounded replay is deliberately not reported as exact or
                # one-minimal.  The ordered task contract proves a necessary
                # stage-count lower bound, while the replay records only a
                # successful upper-bound policy trace.
                "bounded_replay_depth_floor": (
                    bounded_depth_proof or exact_depth_proof
                ),
                "bounded_replay_tick_floor": (
                    exact_depth_proof
                    or (
                        bounded_depth_proof
                        and len(behavior["successful_decision_tick_upper_bound"])
                        >= requirements.min_effective_ticks
                    )
                ),
                "bounded_replay_physical_tool_floor": (
                    exact_depth_proof
                    or (
                        bounded_depth_proof
                        and len(behavior["successful_tool_set_upper_bound"])
                        >= requirements.min_physical_tools
                    )
                ),
            }
        )
    if replay_minimality is None and not require_minimality:
        pending.extend(
            ["one_minimal_replay_proven", "one_minimal_tick_floor", "one_minimal_physical_tool_floor"]
        )
    else:
        checks.update(
            {
                "one_minimal_replay_proven": (
                    behavior["one_minimal_status"] == "one_minimal"
                ),
                "one_minimal_tick_floor": (
                    len(behavior["one_minimal_decision_ticks"])
                    >= requirements.min_effective_ticks
                ),
                "one_minimal_physical_tool_floor": (
                    len(behavior["one_minimal_physical_tools"])
                    >= requirements.min_physical_tools
                ),
            }
        )
    failures = []
    bounded_replay = bool(
        checks.get("bounded_replay_depth_floor")
        and not checks.get("exact_dependency_depth_floor")
    )
    bounded_exemptions = {
        "exact_dependency_depth_floor",
        "one_minimal_replay_proven",
        "one_minimal_tick_floor",
        "one_minimal_physical_tool_floor",
    }
    for name, passed in checks.items():
        if passed or (bounded_replay and name in bounded_exemptions):
            continue
        failures.append(name)
    status = "held" if failures else ("pending" if pending else "passed")
    return {
        "version": DIFFICULTY_CONTRACT_VERSION,
        "difficulty_level": level,
        "track": track,
        "status": status,
        "requirements": asdict(requirements),
        "configuration": static,
        "behavior": behavior,
        "checks": checks,
        "failures": failures,
        "pending": pending,
        "configuration_complexity_is_not_sufficient": True,
    }


def calibrate_difficulty_level(
    scenario: dict[str, Any],
    *,
    trajectory_complexity: dict[str, Any],
    replay_minimality: dict[str, Any],
    physical_tool_names: Collection[str] | None = None,
) -> dict[str, Any]:
    """Return the highest public level supported by replay evidence.

    The declared label is deliberately ignored while evaluating each level.
    This prevents a configuration-only YAML label from promoting an item and
    caps static scheduling items at Medium until they are genuinely dynamic.
    """

    declared = str(scenario.get("difficulty_level") or "")
    evaluations: dict[str, dict[str, Any]] = {}
    calibrated: str | None = None
    for level in DIFFICULTY_REQUIREMENTS:
        candidate = deepcopy(scenario)
        candidate["difficulty_level"] = level
        result = evaluate_difficulty_contract(
            candidate,
            trajectory_complexity=trajectory_complexity,
            replay_minimality=replay_minimality,
            physical_tool_names=physical_tool_names,
            require_behavior=True,
            require_minimality=True,
        )
        evaluations[level] = result
        if result["status"] == "passed":
            calibrated = level

    return {
        "version": DIFFICULTY_CONTRACT_VERSION,
        "declared_difficulty_level": declared,
        "calibrated_difficulty_level": calibrated,
        "declared_level_matches_evidence": declared == calibrated,
        "status": "passed" if calibrated is not None else "held",
        "evaluations": evaluations,
    }
