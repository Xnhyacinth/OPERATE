"""Versioned Protocol-2.1 candidate admission profiles.

Frozen and legacy working sets keep the original strict contract.  New
candidate-only working sets may explicitly select ``quality_core_v2`` to avoid
making the same invariant a hard gate in more than one pipeline stage.
"""

from __future__ import annotations

from typing import Any

STRICT_ADMISSION_PROFILE = "strict_v1"
QUALITY_CORE_V2_ADMISSION_PROFILE = "quality_core_v2"
QUALITY_CORE_V2_DIAGNOSTIC_SOURCE_GATES = frozenset(
    {"decision_graph", "difficulty_proof"}
)
QUALITY_CORE_V2_DIAGNOSTIC_BEHAVIORAL_CHECKS = frozenset(
    {"reference_process_capability_satisfied"}
)
SUPPORTED_ADMISSION_PROFILES = frozenset(
    {STRICT_ADMISSION_PROFILE, QUALITY_CORE_V2_ADMISSION_PROFILE}
)


def declared_protocol21_admission_profile(artifact: dict[str, Any]) -> str | None:
    """Return one explicit profile, rejecting contradictory declarations."""
    raw_values = [
        artifact.get("core_admission_profile"),
        artifact.get("admission_profile"),
    ]
    for key in ("constraints", "selection_constraints"):
        constraints = artifact.get(key)
        if isinstance(constraints, dict):
            raw_values.append(constraints.get("core_admission_profile"))
    profiles = {str(value).strip() for value in raw_values if value not in (None, "")}
    unsupported = profiles - SUPPORTED_ADMISSION_PROFILES
    if unsupported:
        raise ValueError(
            "unsupported Protocol-2.1 admission profile: "
            + ", ".join(sorted(unsupported))
        )
    if len(profiles) > 1:
        raise ValueError(
            "conflicting Protocol-2.1 admission profiles: "
            + ", ".join(sorted(profiles))
        )
    return next(iter(profiles), None)


def resolve_protocol21_admission_profile(artifact: dict[str, Any]) -> str:
    """Resolve and validate an explicitly versioned candidate profile."""
    return declared_protocol21_admission_profile(artifact) or STRICT_ADMISSION_PROFILE


def source_admission_failures(
    failures: list[str],
    *,
    profile: str,
) -> list[str]:
    """Return failures owned by the source-grounded stage.

    Under ``quality_core_v2``, decision-graph shape and exact
    shortest-strategy/minimality evidence are diagnostics.  Source integrity,
    environment closure, deterministic replay, task headroom, counterfactual
    support, domain boundaries, and source independence remain blockers.
    """
    if profile == STRICT_ADMISSION_PROFILE:
        return list(failures)
    if profile == QUALITY_CORE_V2_ADMISSION_PROFILE:
        return [
            failure
            for failure in failures
            if failure not in QUALITY_CORE_V2_DIAGNOSTIC_SOURCE_GATES
        ]
    raise ValueError(f"unsupported Protocol-2.1 admission profile: {profile}")


def requires_exact_strategy_minimality(*, profile: str) -> bool:
    """Return whether exact shortest-strategy calibration gates admission."""
    if profile == STRICT_ADMISSION_PROFILE:
        return True
    if profile == QUALITY_CORE_V2_ADMISSION_PROFILE:
        return False
    raise ValueError(f"unsupported Protocol-2.1 admission profile: {profile}")


def partition_behavioral_check_failures(
    checks: dict[str, Any],
    *,
    profile: str,
) -> tuple[list[str], list[str]]:
    """Partition behavioral failures under one versioned admission profile."""
    failures = sorted(name for name, passed in checks.items() if passed is not True)
    if profile == STRICT_ADMISSION_PROFILE:
        return failures, []
    if profile == QUALITY_CORE_V2_ADMISSION_PROFILE:
        diagnostics = [
            name
            for name in failures
            if name in QUALITY_CORE_V2_DIAGNOSTIC_BEHAVIORAL_CHECKS
        ]
        diagnostic_set = set(diagnostics)
        return [name for name in failures if name not in diagnostic_set], diagnostics
    raise ValueError(f"unsupported Protocol-2.1 admission profile: {profile}")


def treats_observed_depth_floor_as_diagnostic(*, profile: str) -> bool:
    """Return whether High/Extreme tick-floor paperwork is diagnostic.

    ``quality_core_v2`` establishes native control effect and task headroom in
    the behavioral stage. Reference-agent timing and the observed oracle tick
    count remain diagnostics, not suite-level formal-run blockers.
    """
    if profile == STRICT_ADMISSION_PROFILE:
        return False
    if profile == QUALITY_CORE_V2_ADMISSION_PROFILE:
        return True
    raise ValueError(f"unsupported Protocol-2.1 admission profile: {profile}")


def agentic_admission_check_names(*, difficulty_level: str) -> tuple[str, ...]:
    """Return task/environment checks owned by ``quality_core_v2`` admission.

    Reference-agent behavior is diagnostic here. Native effect and task
    headroom are already established by the behavioral stage, while review,
    post-change response, and parallel-clock behavior are capabilities of the
    evaluated persistent agent rather than intrinsic task-quality gates.
    """
    checks = [
        "current_protocol_semantics",
        "identity_bound_across_artifacts",
        "scenario_signature_current",
        "native_backend_executable",
        "multi_tick_horizon",
        "world_change_contract_declared",
        "material_exogenous_change_observed",
        "exogenous_state_evolution_observed",
        "event_or_change_occurs_after_initial_state",
        "event_adaptive_cadence_declared",
        "native_state_changing_control_available",
    ]
    # High/Extreme adaptive-replanning cadence remains a diagnostic check
    # on the agentic contract; it is not an admission blocker.
    _ = difficulty_level
    return tuple(checks)
