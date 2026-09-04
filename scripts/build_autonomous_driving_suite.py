#!/usr/bin/env python3
"""Build held, source-bound autonomous-driving difficulty slices.

The suite builder deliberately fails closed when a bundle has no material
source event or no complete prevention/protection/recovery window.  It creates
one YAML per public difficulty level for a single source window, but marks
those slices as variance-only until separate source windows are supplied.
That distinction prevents difficulty variants from inflating the Core
denominator while still making fog, budget, and planning-depth behavior
testable during calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DIFFICULTIES = ("basic", "medium", "high", "extreme")
NGSIM_DATA_SOURCE = (
    "U.S. DOT FHWA Next Generation Simulation (NGSIM) US-101 Vehicle "
    "Trajectories and Supporting Data"
)
NGSIM_LICENSE = (
    "CC-BY-SA-3.0 dataset API metadata; CC-BY-SA-4.0 Common Core metadata "
    "(operator-reviewed)"
)
NGSIM_LOCK_STRATEGY = (
    "doi+canonical_query_or_archive+raw_sha256+row_semantic_sha256"
)
_TACTICAL_DELAY_TICKS = {
    "basic": 0,
    "medium": 1,
    # High adds stricter observation and review obligations without claiming
    # an impossible two-epoch delay on a short source window.  Extreme keeps
    # the two-epoch profile for future long-window source slices.
    "high": 1,
    "extreme": 2,
}
_POST_RECOVERY_DWELL_TICKS = {
    "basic": 2,
    "medium": 3,
    "high": 4,
    "extreme": 6,
}
_PLAN_DEPENDENCY_DEPTH = {
    "basic": 1,
    "medium": 2,
    "high": 3,
    "extreme": 4,
}
_DECISION_DILEMMA = {
    "basic": "maintain_progress_while_preserving_headway",
    "medium": "early_slowdown_vs_route_progress",
    "high": "paid_information_vs_prevention_deadline",
    "extreme": "guarded_recovery_vs_premature_resume",
}

_REQUIRED_SUPERVISORY_TOOLS = {
    "basic": ["commit_to_plan", "set_driving_envelope"],
    "medium": ["inspect_local_scene", "commit_to_plan", "set_driving_envelope"],
    "high": [
        "inspect_local_scene",
        "inspect_safety_state",
        "commit_to_plan",
        "set_driving_envelope",
        "request_tactical_maneuver",
    ],
    "extreme": [
        "inspect_local_scene",
        "inspect_safety_state",
        "inspect_odd_status",
        "commit_to_plan",
        "set_driving_envelope",
        "request_tactical_maneuver",
        "request_minimal_risk_maneuver",
        "request_recovery_check",
        "authorize_recovery",
    ],
}
_REQUIRED_OBSERVATION_TOOLS = {
    "basic": [],
    "medium": ["inspect_local_scene"],
    "high": ["inspect_local_scene", "inspect_safety_state"],
    "extreme": ["inspect_local_scene", "inspect_safety_state", "inspect_odd_status"],
}

_SOURCE_EVENT_SEQUENCE = {
    "lead_vehicle_braking": ("lead_vehicle_braking", "actor_state_update"),
    "lane_change_conflict": ("lane_change_conflict", "cut_in_gap_boundary"),
    "minimum_time_headway_conflict": ("short_time_headway_boundary",),
    # Kept for small synthetic adapter fixtures; published candidates use one
    # of the three source-native miner labels above.
    "cut_in": ("actor_state_update",),
}


def _source_event_contract(hazard_kind: str, fixture: dict[str, Any]) -> dict[str, Any]:
    expected = _SOURCE_EVENT_SEQUENCE.get(hazard_kind)
    if expected is None:
        raise ValueError("autonomous_driving_source_event_hazard_unsupported")
    events = [row for row in fixture.get("source_events") or [] if isinstance(row, dict)]
    ordered = sorted(
        events,
        key=lambda row: (
            int(row.get("trigger_tick") or 0),
            int(row.get("trigger_offset_ms_within_tick") or 0),
            str(row.get("event_id") or ""),
        ),
    )
    if ordered != events:
        raise ValueError("autonomous_driving_source_events_not_time_ordered")
    kinds = [str(row.get("kind") or "") for row in events]
    cursor = 0
    for kind in kinds:
        if cursor < len(expected) and kind == expected[cursor]:
            cursor += 1
    if cursor != len(expected):
        raise ValueError(
            "autonomous_driving_source_event_sequence_incomplete:"
            f"expected={','.join(expected)}:observed={','.join(kinds)}"
        )
    return {
        "source_native": True,
        "expected_event_kinds": list(expected),
        "observed_event_kinds": kinds,
        "event_count": len(events),
        "ordered_by_source_time": True,
        "interruptible_after_standing_plan": True,
    }


def _difficulty_profile(level: str) -> dict[str, Any]:
    """Return the explicit supervisory contract for one public difficulty.

    Keeping this in one place prevents the YAML, report, and readiness auditor
    from silently drifting apart as the long-horizon task is hardened.
    """
    return {
        "review_interval_ticks": 1 if level in {"high", "extreme"} else 2,
        "requires_paid_safety_inspection": level in {"high", "extreme"},
        "max_tool_calls_per_tick": {"basic": 5, "medium": 7, "high": 9, "extreme": 11}[level],
        "plan_dependency_depth": _PLAN_DEPENDENCY_DEPTH[level],
        "worst_case_declared_tactical_delay_ticks": _TACTICAL_DELAY_TICKS[level],
        "post_recovery_stable_dwell_ticks": _POST_RECOVERY_DWELL_TICKS[level],
        "decision_dilemma": _DECISION_DILEMMA[level],
        "recovery_sequence": [
            "request_minimal_risk_maneuver",
            "request_recovery_check",
            "authorize_recovery",
        ],
        "observation_regime": (
            "full_local_scene"
            if level in {"basic", "medium"}
            else "partial_actor_velocity_until_paid_inspection"
        ),
        "required_supervisory_tools": list(_REQUIRED_SUPERVISORY_TOOLS[level]),
        "required_observation_tools": list(_REQUIRED_OBSERVATION_TOOLS[level]),
        "requires_preventive_action": True,
        "preventive_action_tools": [
            "set_driving_envelope",
            "request_tactical_maneuver",
        ],
        "minimum_decision_epochs": {
            "basic": 1,
            "medium": 2,
            "high": 2,
            "extreme": 3,
        }[level],
    }


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _materialized_source_contract(
    *, bundle: Path, manifest: dict[str, Any], source_window_sha256: str
) -> dict[str, Any] | None:
    raw = manifest.get("source_contract")
    if not isinstance(raw, dict):
        return None
    contract: dict[str, Any] = {}
    for role in (
        "runtime_input",
        "derivation_input",
        "implementation_asset",
        "metadata",
        "license",
    ):
        values = raw.get(role) or []
        if not isinstance(values, list):
            raise ValueError(f"autonomous_driving_source_contract_{role}_invalid")
        contract[role] = [_relative(bundle / str(value)) for value in values]
    required = [*contract["runtime_input"], *contract["derivation_input"]]
    hashes: dict[str, str] = {}
    for declared in required:
        path = Path(declared)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise ValueError(f"autonomous_driving_source_contract_file_missing:{declared}")
        hashes[declared] = hashlib.sha256(path.read_bytes()).hexdigest()
    contract["file_sha256s"] = hashes
    contract["derived_window"] = {
        "sha256": source_window_sha256,
        "recipe_version": "ngsim_phase_complete_window_v1",
    }
    return contract


def _require_complete_fixture(
    bundle: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from domains.autonomous_driving.data.ngsim import verify_bundle

    verify_bundle(bundle)
    manifest = _load(bundle / "bundle.json")
    source_lock_path = bundle / "source/source.lock.json"
    if source_lock_path.is_file():
        source_lock = _load(source_lock_path)
        manifest["source_recording_id"] = (source_lock.get("source_plan") or {}).get(
            "recording_id"
        ) or (source_lock.get("source_plan") or {}).get("location")
    mining = _load(bundle / "mining/candidates.json")
    fixture = _load(bundle / "runtime/fixture.json")
    derivation = dict(fixture.get("derivation") or {})
    candidate_id = str(derivation.get("candidate_id") or "")
    if (
        "selected_candidate_id" in manifest
        and str(manifest.get("selected_candidate_id") or "") != candidate_id
    ):
        raise ValueError("autonomous_driving_suite_manifest_candidate_mismatch")
    selected = next(
        (
            dict(row)
            for row in mining.get("candidates") or []
            if row.get("candidate_id") == candidate_id
        ),
        None,
    )
    if selected is None:
        raise ValueError("autonomous_driving_suite_fixture_candidate_missing")
    context = dict(selected.get("hazard_context") or {})
    if not bool(context.get("phase_window_complete")):
        raise ValueError("autonomous_driving_suite_requires_phase_complete_candidate")
    events = [row for row in fixture.get("source_events") or [] if isinstance(row, dict)]
    if not events:
        raise ValueError("autonomous_driving_suite_requires_source_grounded_events")
    required = (
        "first_observable_time_ms",
        "latest_preventive_command_time_ms",
        "risk_boundary_proxy_time_ms",
        "recovery_window_end_time_ms",
    )
    if any(context.get(key) is None for key in required):
        raise ValueError("autonomous_driving_suite_response_window_incomplete")
    if int(context["latest_preventive_command_time_ms"]) <= int(
        context["first_observable_time_ms"]
    ):
        raise ValueError("autonomous_driving_suite_prevention_window_empty")
    _source_event_contract(
        str(context.get("hazard_kind") or selected.get("hazard_kind") or ""), fixture
    )
    return manifest, selected, fixture


def _horizon_ticks(candidate: dict[str, Any], tick_seconds: float = 5.0) -> int:
    start = int(candidate["start_time_ms"])
    end = int(candidate["end_time_ms_exclusive"])
    ticks = math.ceil((end - start) / (tick_seconds * 1000.0))
    return max(4, ticks)


def _response_window_ticks(
    *,
    candidate: dict[str, Any],
    level: str,
    tick_seconds: float,
) -> dict[str, int]:
    """Convert source times into auditable logical-decision deadlines."""
    context = dict(candidate.get("hazard_context") or {})
    start_ms = int(candidate["start_time_ms"])
    tick_ms = int(round(tick_seconds * 1000.0))
    first_ms = int(context["first_observable_time_ms"])
    latest_ms = int(context["latest_preventive_command_time_ms"])
    boundary_ms = int(context["risk_boundary_proxy_time_ms"])
    first_tick = max(0, (first_ms - start_ms) // tick_ms)
    latest_tick = max(0, (latest_ms - start_ms) // tick_ms)
    boundary_tick = max(0, (boundary_ms - start_ms + tick_ms - 1) // tick_ms)
    delay_ticks = _TACTICAL_DELAY_TICKS[level]
    available_epochs = latest_tick - first_tick
    if available_epochs < delay_ticks + 1:
        raise ValueError(
            "autonomous_driving_response_window_too_short:"
            f"{level}:available={available_epochs}:required={delay_ticks + 1}"
        )
    if boundary_tick <= latest_tick:
        raise ValueError("autonomous_driving_protective_deadline_precedes_prevention")
    return {
        "first_observable_tick": first_tick,
        "latest_preventive_command_tick": latest_tick,
        "protective_response_deadline_tick": boundary_tick,
        "worst_case_declared_tactical_delay_ticks": delay_ticks,
        "available_prevention_epochs": available_epochs,
    }


def _scenario(
    *,
    bundle: Path,
    manifest: dict[str, Any],
    candidate: dict[str, Any],
    fixture: dict[str, Any],
    level: str,
    execution_mode: str = "emulated_source_initialized",
) -> dict[str, Any]:
    context = dict(candidate.get("hazard_context") or {})
    derivation = dict(fixture.get("derivation") or {})
    candidate_id = str(candidate["candidate_id"])
    source_digest = str(candidate["source_window_sha256"])
    hazard_kind = str(context.get("hazard_kind") or "source_grounded_conflict")
    event_contract = _source_event_contract(hazard_kind, fixture)
    stem = f"ngsim_{candidate_id.split(':', 2)[1]}_{level}_s42"
    family = {
        "cut_in": "cut_in_prevention_and_emergency",
        "lane_change_conflict": "cut_in_prevention_and_emergency",
        "leader_braking": "sustained_highway_risk_supervision",
        "short_headway": "sustained_highway_risk_supervision",
        "minimum_time_headway_conflict": "sustained_highway_risk_supervision",
    }.get(hazard_kind, "sustained_highway_risk_supervision")
    mode = "deep_planning" if level in {"high", "extreme"} else "time_pressure"
    scenario_id = f"autonomous_driving/{family}/{mode}/{level}/{stem}"
    if execution_mode not in {"emulated_source_initialized", "live"}:
        raise ValueError("autonomous_driving_suite_execution_mode_invalid")
    ego_actor_id = str(derivation.get("ego_actor_id") or "")
    backend_config: dict[str, Any] = {
        "source_bundle": _relative(bundle),
        "candidate_id": candidate_id,
        "ego_actor_id": ego_actor_id,
        "execution_mode": execution_mode,
    }
    if execution_mode == "live":
        backend_config.update(
            {
                "ego_vehicle_id": ego_actor_id,
                "sumo_config_path": _relative(bundle / "sumo/run.sumocfg"),
                "sumo_net_path": _relative(bundle / "sumo/network.net.xml"),
                "sumo_route_path": _relative(bundle / "sumo/routes.rou.xml"),
            }
        )
    backend_config["task_requirements"] = {
        "required_stable_dwell_ticks": _POST_RECOVERY_DWELL_TICKS[level],
        "guarded_recovery_required_if_mrm": True,
        "recovery_state": "nominal_after_guarded_recovery",
        "recovery_sequence": list(_difficulty_profile(level)["recovery_sequence"]),
        "required_review_interval_ticks": _difficulty_profile(level)["review_interval_ticks"],
        "decision_dilemma": _difficulty_profile(level)["decision_dilemma"],
        "requires_paid_safety_inspection": _difficulty_profile(level)[
            "requires_paid_safety_inspection"
        ],
        "required_supervisory_tools": list(
            _difficulty_profile(level)["required_supervisory_tools"]
        ),
        "required_observation_tools": list(
            _difficulty_profile(level)["required_observation_tools"]
        ),
        "requires_preventive_action": True,
        "preventive_action_tools": list(_difficulty_profile(level)["preventive_action_tools"]),
        "minimum_decision_epochs": _difficulty_profile(level)["minimum_decision_epochs"],
    }
    first_observable_ms = int(context["first_observable_time_ms"])
    latest_preventive_ms = int(context["latest_preventive_command_time_ms"])
    risk_boundary_ms = int(context["risk_boundary_proxy_time_ms"])
    response_ticks = _response_window_ticks(
        candidate=candidate,
        level=level,
        tick_seconds=5.0,
    )
    backend_config["task_requirements"]["paid_safety_inspection_deadline_tick"] = response_ticks[
        "latest_preventive_command_tick"
    ]
    backend_config["task_requirements"]["latest_preventive_command_tick"] = response_ticks[
        "latest_preventive_command_tick"
    ]
    backend_config["dimension_applicability"] = {
        "system_survival": {
            "applicable": True,
            "reason": "measured_by_native_evidence",
        },
        "safety_violation": {
            "applicable": True,
            "reason": "measured_by_native_evidence",
        },
        "economic_cost": {
            "applicable": True,
            "reason": "measured_by_native_evidence",
        },
        "adaptive_replanning": {
            "applicable": True,
            "reason": "measured_by_native_evidence",
        },
        "information_efficiency": {
            "applicable": True,
            "reason": "measured_by_native_evidence",
        },
        "foresight_score": {
            "applicable": True,
            "reason": "measured_by_native_evidence",
        },
        "counterfactual_prevention": {
            "applicable": True,
            "reason": "deterministic_wait_only_counterfactual_replay_available",
        },
        "tool_use_efficiency": {
            "applicable": True,
            "reason": "measured_by_native_evidence",
        },
        "optimality_gap": {
            "applicable": False,
            "reason": "no_validated_native_trajectory_optimum",
        },
        "weighted_equity_score": {
            "applicable": False,
            "reason": "no_source_grounded_criticality_classes",
        },
        "ethical_quality": {
            "applicable": False,
            "reason": "no_source_grounded_ethical_dilemma",
        },
        "stakeholder_management": {
            "applicable": False,
            "reason": "no_source_grounded_stakeholder_model",
        },
        "stakeholder_equity": {
            "applicable": False,
            "reason": "no_source_grounded_stakeholder_classes",
        },
    }
    worst_case_tactical_delay_ticks = response_ticks["worst_case_declared_tactical_delay_ticks"]
    source_ticks = _horizon_ticks(candidate)
    post_recovery_dwell_ticks = _POST_RECOVERY_DWELL_TICKS[level]
    horizon_ticks = max(8, source_ticks + post_recovery_dwell_ticks)
    profile = _difficulty_profile(level)
    scenario = {
        "seed_id": scenario_id,
        "scenario_id": scenario_id,
        "family": family,
        "domain": "autonomous_driving",
        "backend_kind": "sumo_ego",
        "backend_config": backend_config,
        "task_contract": {
            "contract": "autonomous_driving.risk_progress_mitigation.v1",
            "standing_plan_required": True,
            "hazard_kind": hazard_kind,
            "first_observable_time_ms": first_observable_ms,
            "latest_preventive_command_time_ms": latest_preventive_ms,
            "risk_boundary_proxy_time_ms": risk_boundary_ms,
            "supervisory_prevention_window_ms": int(context["latest_preventive_command_time_ms"])
            - int(context["first_observable_time_ms"]),
            "protective_response_window_ms": int(context.get("protective_response_window_ms") or 0),
            "recovery_window_end_time_ms": int(context["recovery_window_end_time_ms"]),
            "first_observable_tick": response_ticks["first_observable_tick"],
            "latest_preventive_command_tick": response_ticks["latest_preventive_command_tick"],
            "protective_response_deadline_tick": response_ticks[
                "protective_response_deadline_tick"
            ],
            "worst_case_declared_tactical_delay_ticks": worst_case_tactical_delay_ticks,
            "available_prevention_epochs": response_ticks["available_prevention_epochs"],
            "post_recovery_stable_dwell_ticks": post_recovery_dwell_ticks,
            "required_recovery_state": "nominal_after_guarded_recovery",
            "decision_dilemma": profile["decision_dilemma"],
            "required_review_interval_ticks": profile["review_interval_ticks"],
            "required_recovery_sequence": list(profile["recovery_sequence"]),
            "response_window_semantics": (
                "logical supervisory deadline excludes declared tool/plan delay; "
                "substep shield owns the protective deadline"
            ),
            "required_supervisory_tools": list(profile["required_supervisory_tools"]),
            "required_observation_tools": list(profile["required_observation_tools"]),
            "requires_preventive_action": True,
            "preventive_action_tools": list(profile["preventive_action_tools"]),
            "minimum_decision_epochs": profile["minimum_decision_epochs"],
            "environment_evolution": event_contract,
        },
        "horizon_ticks": horizon_ticks,
        "tick_seconds": 5.0,
        "clock_contract": {
            "schema_version": "driving_clock_v1",
            "physics_step_seconds": 0.1,
            "shield_step_seconds": 0.1,
            "substeps_per_supervisory_tick": 50,
            "provider_wall_clock_advances_simulation": False,
        },
        "seed": 42,
        "difficulty_level": level,
        "difficulty_mode": mode,
        "difficulty_params": {
            **profile,
            "source_event_count": len(fixture.get("source_events") or []),
            "source_event_sequence": list(event_contract["observed_event_kinds"]),
            "source_event_expected_sequence": list(event_contract["expected_event_kinds"]),
            "environment_evolution": event_contract,
            "available_prevention_epochs": response_ticks["available_prevention_epochs"],
            "protective_response_deadline_tick": response_ticks[
                "protective_response_deadline_tick"
            ],
        },
        "formal_core_allowed": False,
        "release_admission": "held_diagnostic_pilot",
        "held_reasons": [
            "live_sumo_reactive_rollout_not_validated",
            "shield_only_material_headroom_not_yet_demonstrated",
            "native_reactive_runtime_not_verified",
        ],
        "source_window_sha256": source_digest,
        "provenance": {
            "data_source": NGSIM_DATA_SOURCE,
            "dataset_id": manifest.get("source_dataset_id"),
            "url": (
                "https://data.transportation.gov/Automobiles/"
                "Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj"
            ),
            "license": NGSIM_LICENSE,
            "lock_strategy": NGSIM_LOCK_STRATEGY,
            "source_release": manifest.get("source_release"),
            "recording_id": manifest.get("source_recording_id"),
            "license_id": manifest.get("license_id"),
            "source_evidence_sha256": (manifest.get("evidence") or {}).get(
                "source_evidence_sha256"
            ),
            "source_event_chain_sha256": (manifest.get("evidence") or {}).get(
                "runtime_source_events_sha256"
            ),
            "bundle_id": manifest.get("bundle_id"),
            "candidate_id": candidate_id,
            "hazard_kind": hazard_kind,
        },
    }
    source_contract = _materialized_source_contract(
        bundle=bundle,
        manifest=manifest,
        source_window_sha256=source_digest,
    )
    if source_contract is not None:
        scenario["source_contract"] = source_contract
    return scenario


def build_suite(
    bundle: Path,
    output: Path | None,
    *,
    execution_mode: str = "emulated_source_initialized",
    primary_difficulty: str | None = None,
) -> dict[str, Any]:
    """Return a source-feasible suite or one calibrated primary scenario."""
    if primary_difficulty is not None and primary_difficulty not in DIFFICULTIES:
        raise ValueError("autonomous_driving_primary_difficulty_invalid")
    manifest, candidate, fixture = _require_complete_fixture(bundle)
    event_contract = _source_event_contract(
        str((candidate.get("hazard_context") or {}).get("hazard_kind") or ""), fixture
    )
    if output is not None:
        output = output.resolve()
    if output is not None and output.exists() and any(output.iterdir()):
        raise FileExistsError("autonomous_driving_suite_output_not_empty")
    rows: list[str] = []
    skipped: dict[str, str] = {}
    included_levels: list[str] = []
    selected_levels = (primary_difficulty,) if primary_difficulty is not None else DIFFICULTIES
    for level in selected_levels:
        try:
            scenario = _scenario(
                bundle=bundle,
                manifest=manifest,
                candidate=candidate,
                fixture=fixture,
                level=level,
                execution_mode=execution_mode,
            )
        except ValueError as error:
            if not str(error).startswith("autonomous_driving_response_window_too_short"):
                raise
            skipped[level] = str(error)
            continue
        relative = Path(scenario["scenario_id"] + ".yaml")
        rows.append(relative.as_posix())
        included_levels.append(level)
        if output is not None:
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                yaml.safe_dump(scenario, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )
    report = {
        "schema_version": "autonomous_driving_suite_report_v2",
        "status": "held",
        "bundle_id": manifest.get("bundle_id"),
        "candidate_id": candidate.get("candidate_id"),
        "primary_difficulty": primary_difficulty,
        "execution_mode": execution_mode,
        "source_window_sha256": candidate.get("source_window_sha256"),
        "recording_id": manifest.get("source_recording_id"),
        "hazard_kind": (candidate.get("hazard_context") or {}).get("hazard_kind"),
        "source_window_seconds": round(
            (int(candidate["end_time_ms_exclusive"]) - int(candidate["start_time_ms"])) / 1000.0,
            3,
        ),
        "difficulty_slices": list(rows),
        "skipped_difficulty_slices": skipped,
        "difficulty_feasibility": {
            level: {
                "status": "included" if level in included_levels else "held",
                "reason": (
                    "source_window_has_declared_prevention_epochs"
                    if level in included_levels
                    else skipped.get(level, "not_generated")
                ),
            }
            for level in DIFFICULTIES
        },
        "difficulty_profiles": {level: _difficulty_profile(level) for level in included_levels},
        "long_horizon": {
            "source_window_ticks": _horizon_ticks(candidate),
            "source_window_seconds": round(
                (int(candidate["end_time_ms_exclusive"]) - int(candidate["start_time_ms"]))
                / 1000.0,
                3,
            ),
            "post_recovery_dwell_ticks": {
                level: _POST_RECOVERY_DWELL_TICKS[level] for level in included_levels
            },
            "horizon_ticks": {
                level: max(
                    8,
                    _horizon_ticks(candidate) + _POST_RECOVERY_DWELL_TICKS[level],
                )
                for level in included_levels
            },
            "minimum_supervisory_ticks": 10,
        },
        "core_denominator_eligible": False,
        "core_denominator_reason": "difficulty_slices_share_one_source_window",
        "admission": {
            "formal_core_allowed": False,
            "source_events_verified": True,
            "reactive_closed_loop_validated": False,
            "native_runtime_validation": False,
            "shield_only_material_headroom": "pending",
            "license_review": manifest.get("license_review_status")
            or "pending_metadata_discrepancy",
        },
        "response_windows_ms": {
            "supervisory_prevention": int(
                (candidate.get("hazard_context") or {}).get("supervisory_prevention_window_ms", 0)
            ),
            "protective_response": int(
                (candidate.get("hazard_context") or {}).get("protective_response_window_ms", 0)
            ),
            "recovery": int((candidate.get("hazard_context") or {}).get("recovery_window_ms", 0)),
        },
        "source_event_chain_sha256": (manifest.get("evidence") or {}).get(
            "runtime_source_events_sha256"
        ),
        "source_event_sequence": list(event_contract["observed_event_kinds"]),
        "source_event_expected_sequence": list(event_contract["expected_event_kinds"]),
        "environment_evolution": event_contract,
    }
    if output is not None:
        (output / "suite_report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        (output / "candidate_report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=("emulated_source_initialized", "live"),
        default="emulated_source_initialized",
    )
    parser.add_argument("--primary-difficulty", choices=DIFFICULTIES)
    args = parser.parse_args(argv)
    bundle = args.bundle if args.bundle.is_absolute() else REPO_ROOT / args.bundle
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    try:
        report = build_suite(
            bundle.resolve(),
            None if args.check else output.resolve(),
            execution_mode=args.execution_mode,
            primary_difficulty=args.primary_difficulty,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, indent=2))
        return 1
    if args.check:
        expected = [output / Path(value) for value in report["difficulty_slices"]]
        report_path = output / "suite_report.json"
        ok = all(path.is_file() for path in expected) and report_path.is_file()
        print(json.dumps({"status": "verified" if ok else "stale", "paths": len(expected) + 1}))
        return 0 if ok else 1
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
