"""Fail-closed evidence contract for Protocol-2.1 Traffic candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

TRAFFIC_CANDIDATE_SCHEMA_VERSION = "1.0"


def classify_native_control_surface(
    *,
    program_ids: Sequence[str],
    phase_duration_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify only native controls observed on the exact SUMO runtime."""
    programs = list(dict.fromkeys(str(value) for value in program_ids if str(value)))
    durations = [dict(value) for value in phase_duration_candidates]
    program_selection = len(programs) > 1
    phase_duration = bool(durations)
    if program_selection and phase_duration:
        surface = "both"
    elif program_selection:
        surface = "program_selection"
    elif phase_duration:
        surface = "phase_duration"
    else:
        surface = "none"
    return {
        "program_ids": programs,
        "program_selection_available": program_selection,
        "phase_duration_available": phase_duration,
        "phase_duration_candidates": durations,
        "native_control_surface": surface,
    }


def world_evolution_is_material(
    *,
    source_schedule: Sequence[Mapping[str, Any]],
    external_events: Sequence[Mapping[str, Any]],
    environment_transition_contract: Mapping[str, Any],
    vehicle_positions_changed: bool,
) -> bool:
    """Reject ordinary vehicle motion as the sole exogenous-world proof."""
    del vehicle_positions_changed
    return bool(
        source_schedule
        or external_events
        or environment_transition_contract
    )


def native_control_effect_observed(
    *,
    agent_action: Mapping[str, Any],
    runtime_mutation: Mapping[str, Any],
    state_digest_before: str,
    state_digest_after: str,
) -> bool:
    """Require an action, native acknowledgement, and observable state delta."""
    return bool(
        agent_action
        and runtime_mutation.get("sumo_state_mutated") is True
        and state_digest_before
        and state_digest_after
        and state_digest_before != state_digest_after
    )


def _runtime_complete(runtime_identity: Mapping[str, Any]) -> bool:
    return bool(
        runtime_identity.get("native_launch_passed") is True
        and runtime_identity.get("complete") is True
        and runtime_identity.get("sumocfg_sha256")
        and runtime_identity.get("network_sha256")
        and runtime_identity.get("ordered_route_sha256s")
        and runtime_identity.get("sumo_version")
        and runtime_identity.get("transport")
    )


def build_traffic_candidate(
    *,
    runtime_identity: Mapping[str, Any],
    source_contract: Mapping[str, Any],
    world_evolution_contract: Mapping[str, Any],
    control_surface: Mapping[str, Any],
    safety_baseline: Mapping[str, Any],
    deterministic_replay: Mapping[str, Any],
    native_control_effect: Mapping[str, Any],
    task_contract: Mapping[str, Any] | None = None,
    safety_attribution: Mapping[str, Any] | None = None,
    event_context_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build diagnostic evidence without granting release admission."""
    blockers: set[str] = set()
    runtime_ready = _runtime_complete(runtime_identity)
    if not runtime_ready:
        blockers.add("runtime_identity_incomplete")

    source_schedule = source_contract.get("source_schedule") or []
    source_ready = bool(
        source_contract.get("status") == "complete"
        and source_contract.get("runtime_consumption_observed") is True
        and source_schedule
    )
    if not source_ready:
        blockers.add("source_trace_missing")

    material_events = (
        world_evolution_contract.get("material_exogenous_events") or []
    )
    transition_contract = (
        world_evolution_contract.get("environment_transition_contract") or {}
    )
    world_ready = bool(
        world_evolution_contract.get("status") == "complete"
        and world_evolution_is_material(
            source_schedule=source_schedule,
            external_events=material_events,
            environment_transition_contract=transition_contract,
            vehicle_positions_changed=bool(
                world_evolution_contract.get("vehicle_positions_changed")
            ),
        )
    )
    if not world_ready:
        blockers.add("world_evolution_missing")

    canonical_control = {
        **dict(control_surface),
        **classify_native_control_surface(
            program_ids=control_surface.get("program_ids") or [],
            phase_duration_candidates=(
                control_surface.get("phase_duration_candidates") or []
            ),
        ),
    }
    native_surface = str(
        canonical_control.get("native_control_surface") or "none"
    )
    control_ready = native_surface in {
        "program_selection",
        "phase_duration",
        "both",
    }
    if not control_ready:
        blockers.add("native_control_surface_missing")

    effect_ready = native_control_effect_observed(
        agent_action=native_control_effect.get("action") or {},
        runtime_mutation=native_control_effect.get("runtime_mutation") or {},
        state_digest_before=str(
            native_control_effect.get("state_digest_before") or ""
        ),
        state_digest_after=str(
            native_control_effect.get("state_digest_after") or ""
        ),
    )
    if not effect_ready:
        blockers.add("native_control_not_material")

    replay_ready = bool(
        deterministic_replay.get("baseline_deterministic") is True
        and deterministic_replay.get("reference_deterministic") is True
    )
    if not replay_ready:
        blockers.add("deterministic_paired_replay_missing")
    if safety_baseline.get("status") != "complete":
        blockers.add("safety_baseline_incomplete")

    evidence_blockers = set(blockers)
    task_contract = task_contract or {}
    task_contract_status = str(
        task_contract.get("status") or "missing"
    )
    if task_contract_status != "valid":
        blockers.add(f"task_contract_{task_contract_status}")
        task_schema_fields = {
            "objective",
            "decision_space",
            "metrics",
            "success_condition",
            "failure_condition",
            "evaluation_horizon",
            "action_constraints",
            "capture_contract_version",
        }
        if not task_schema_fields <= set(task_contract):
            blockers.add("task_contract_schema_missing")

    safety_attribution = safety_attribution or {}
    safety_attribution_status = str(
        safety_attribution.get("status")
        or safety_attribution.get("release_status")
        or "missing"
    )
    if safety_attribution_status != "passed":
        blockers.add(
            f"safety_attribution_{safety_attribution_status}"
        )

    event_context_contract = event_context_contract or {}
    event_context_status = str(
        event_context_contract.get("status") or "missing"
    )
    event_context_complete = event_context_status in {
        "complete",
        "evaluable",
    }
    if not event_context_complete:
        blockers.add("safety_event_context_missing")

    if not runtime_ready:
        status = "runtime_only"
        evidence_stage = "runtime_only"
    elif evidence_blockers:
        status = "blocked"
        evidence_stage = "runtime_only"
    elif not event_context_complete:
        status = "blocked"
        evidence_stage = "diagnostic_ready"
    elif safety_attribution_status == "failed":
        status = "blocked"
        evidence_stage = "safety_attribution_evaluable"
    elif task_contract_status == "valid":
        status = "task_safety_evaluable"
        evidence_stage = "task_safety_evaluable"
    else:
        status = "safety_attribution_evaluable"
        evidence_stage = "safety_attribution_evaluable"
    if status == "blocked" and not event_context_complete:
        blocking_category = "blocked_due_capture_schema"
    elif status == "blocked":
        blocking_category = "blocked_due_contracts"
    else:
        blocking_category = None
    return {
        "schema_version": TRAFFIC_CANDIDATE_SCHEMA_VERSION,
        "candidate_kind": "protocol21_traffic_runtime_candidate",
        "runtime_identity": dict(runtime_identity),
        "source_contract": dict(source_contract),
        "world_evolution_contract": dict(world_evolution_contract),
        "control_surface": canonical_control,
        "safety_baseline": dict(safety_baseline),
        "deterministic_replay": dict(deterministic_replay),
        "native_control_effect": dict(native_control_effect),
        "task_contract": dict(task_contract),
        "task_contract_status": task_contract_status,
        "safety_attribution": dict(safety_attribution),
        "safety_attribution_status": safety_attribution_status,
        "event_context_contract": dict(event_context_contract),
        "event_context_status": event_context_status,
        "evidence_stage": evidence_stage,
        "blocking_category": blocking_category,
        "candidate_status": status,
        "status": status,
        "blockers": sorted(blockers),
    }
