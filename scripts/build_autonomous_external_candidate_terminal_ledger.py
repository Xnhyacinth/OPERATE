#!/usr/bin/env python3
"""Build one terminal candidate row per driving window and external family.

The ledger separates a bounded native prefilter from formal Core admission.
It consumes existing live-SUMO calibration/replay evidence, validates exact
NGSIM source/ego/conflict/window bindings and shield attribution, and preserves
external benchmark holds without copying or redistributing raw data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_ROOT = REPO_ROOT / "works/autonomous_driving/ngsim/derived/ngsim_multisite_core_slice_v13"
DEFAULT_CATALOG = DEFAULT_ROOT / "catalog_full_v1.json"
DEFAULT_READINESS = DEFAULT_ROOT / "readiness_multisite_v36_runtime_complete.json"
DEFAULT_CALIBRATIONS = DEFAULT_ROOT / "portable_inputs_v1/calibration_batch"
DEFAULT_REPLAYS = DEFAULT_ROOT / "portable_inputs_v1/replay_batch"
DEFAULT_EXTERNAL = REPO_ROOT / "reports/track_c_external_conversion_current_20260812.json"
DEFAULT_ADDITIONAL_EXTERNAL = REPO_ROOT / "reports/latest_benchmark_candidate_wave_20260813.json"
DEFAULT_OUTPUT = (
    REPO_ROOT / "reports/autonomous_external_candidate_terminal_20260814" / "terminal_ledger.json"
)

LEG_NAMES = {"shield_only", "rule_tactical", "oracle_offline"}
REQUIRED_NATIVE_GATES = (
    "calibration_report_valid",
    "source_event_materiality",
    "three_leg_presence",
    "collision_departure_safety",
    "reactive_closed_loop",
    "deterministic_replay",
    "difficulty_contract",
)
ADDITIONAL_EXTERNAL_SOURCE_IDS = {
    "alibaba_clusterdata",
    "boptest",
    "building_data_genome_2",
}
RESEARCHED_DRIVING_ROUTES = [
    {
        "source_id": "waymo_open_motion_v1_3_1",
        "title": "Waymo Open Motion Dataset v1.3.1",
        "source_url": "https://waymo.com/open/download/",
        "backend_target": "waymax",
        "disposition": "held_access_terms",
        "access_status": "login_and_non_commercial_terms_required",
        "source_role": "closed_loop_branching_source",
        "integration_rung": "pre_cloned",
        "source_type_core_eligible": True,
        "next_stage": "license_review_then_single_shard_fetch_build",
        "blockers": [
            "source_assets_missing",
            "license_terms_review_pending",
            "single_shard_native_conversion_pending",
            "waymax_runtime_or_adapter_missing",
        ],
        "source_facts": {
            "version": "1.3.1",
            "release_date": "2025-10",
            "segments": 103_354,
            "segment_seconds": 20,
            "frequency_hz": 10,
            "sdc_paths_available": True,
        },
    },
    {
        "source_id": "nuplan",
        "title": "nuPlan planning benchmark",
        "source_url": "https://www.nuplan.org/nuplan",
        "backend_target": "nuplan_closed_loop_devkit",
        "disposition": "held_access_terms",
        "access_status": "registration_required_non_commercial",
        "source_role": "closed_loop_planning_source",
        "integration_rung": "pre_cloned",
        "source_type_core_eligible": True,
        "next_stage": "license_review_then_single_shard_fetch_build",
        "blockers": [
            "source_assets_missing",
            "registration_required",
            "non_commercial_terms_review_pending",
            "single_shard_native_conversion_pending",
            "closed_loop_adapter_missing",
        ],
        "source_facts": {
            "driving_hours_approx": 1_200,
            "cities": ["Boston", "Pittsburgh", "Las Vegas", "Singapore"],
            "closed_loop_devkit_available": True,
        },
    },
    {
        "source_id": "highd",
        "title": "highD highway trajectory dataset",
        "source_url": "https://levelxdata.com/highd-dataset/",
        "backend_target": "source_native_ego_reactive_adapter",
        "disposition": "held_access_terms",
        "access_status": "manual_application_non_commercial_non_redistributable",
        "source_role": "naturalistic_trajectory_source",
        "integration_rung": "pre_cloned",
        "source_type_core_eligible": True,
        "next_stage": "manual_access_then_single_recording_conversion",
        "blockers": [
            "source_assets_missing",
            "manual_access_application_required",
            "non_commercial_terms_review_pending",
            "raw_data_redistribution_forbidden",
            "single_recording_native_conversion_pending",
            "closed_loop_adapter_missing",
        ],
        "source_facts": {
            "recordings": 60,
            "recording_locations": 6,
            "vehicles": 110_500,
        },
    },
    {
        "source_id": "commonroad_reach",
        "title": "CommonRoad Reach",
        "source_url": "https://github.com/CommonRoad/commonroad-reachable-set",
        "backend_target": "commonroad_reach_shadow",
        "disposition": "held_runtime_or_integration",
        "access_status": "public_bsd_3_clause_code_python_below_3_12",
        "source_role": "shadow_validator",
        "integration_rung": "pre_cloned",
        "source_type_core_eligible": False,
        "next_stage": "isolated_python311_shadow_validator",
        "blockers": [
            "commonroad_reach_runtime_missing",
            "shadow_validator_integration_pending",
            "not_a_source_dataset",
        ],
        "source_facts": {"version": "2025.2.1", "blocking_authority": False},
    },
    {
        "source_id": "commonroad_crime",
        "title": "CommonRoad CriMe",
        "source_url": "https://github.com/CommonRoad/commonroad-crime",
        "backend_target": "commonroad_crime_shadow",
        "disposition": "held_runtime_or_integration",
        "access_status": "public_bsd_3_clause_code",
        "source_role": "shadow_monitor",
        "integration_rung": "pre_cloned",
        "source_type_core_eligible": False,
        "next_stage": "isolated_shadow_metric_integration",
        "blockers": [
            "commonroad_crime_runtime_missing",
            "shadow_monitor_integration_pending",
            "not_a_source_dataset",
        ],
        "source_facts": {"blocking_authority": False},
    },
    {
        "source_id": "carla_rss",
        "title": "CARLA RSS",
        "source_url": "https://carla.readthedocs.io/en/latest/adv_rss/",
        "backend_target": "carla_rss_dev_only",
        "disposition": "dev_only",
        "access_status": "public_code_separate_simulator_assets",
        "source_role": "dev_only_simulator",
        "integration_rung": "pre_cloned",
        "source_type_core_eligible": False,
        "next_stage": "optional_isolated_high_fidelity_track",
        "blockers": [
            "carla_runtime_missing",
            "rss_experimental_ground_truth_sensor",
            "synthetic_source_not_core_eligible",
        ],
        "source_facts": {"blocking_authority": False},
    },
    {
        "source_id": "bench2drive",
        "title": "Bench2Drive 0.0.4",
        "source_url": "https://github.com/Thinklab-SJTU/Bench2Drive",
        "backend_target": "carla_0_9_15_dev_only",
        "disposition": "method_transfer_only",
        "access_status": "non_commercial_terms_review_pending",
        "source_role": "method_transfer_only",
        "integration_rung": "pre_cloned",
        "source_type_core_eligible": False,
        "next_stage": "reuse_evaluation_patterns_without_core_rows",
        "blockers": [
            "carla_runtime_missing",
            "synthetic_source_not_core_eligible",
            "license_terms_review_pending",
        ],
        "source_facts": {"version": "0.0.4", "carla_version": "0.9.15"},
    },
]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _digest_without(value: dict[str, Any], *fields: str) -> str:
    unsigned = dict(value)
    for field in fields:
        unsigned.pop(field, None)
    return _object_digest(unsigned)


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _runtime_tick(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _total_cost(row: dict[str, Any]) -> float | None:
    values = row.get("cost_components")
    if not isinstance(values, dict) or not values:
        return None
    try:
        return round(sum(float(value) for value in values.values()), 6)
    except (TypeError, ValueError):
        return None


def _path_identity(value: Any) -> str:
    text = str(value or "")
    path = Path(text)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            return str(path.resolve())
    return path.as_posix()


def _valid_post_change_runtime_chain(leg: dict[str, Any], identity_fields: dict[str, str]) -> bool:
    material_event_ids = {
        str(event.get("event_id") or "")
        for event in leg.get("source_events") or []
        if isinstance(event, dict) and event.get("materiality_passed") is True
    }
    evidence_rows = leg.get("post_change_runtime_evidence")
    if not material_event_ids or not isinstance(evidence_rows, list):
        return False
    leg_name = str(leg.get("leg") or "")
    for evidence in evidence_rows:
        if not isinstance(evidence, dict):
            continue
        material_tick = _runtime_tick(evidence.get("material_event_tick"))
        decision_tick = _runtime_tick(evidence.get("decision_tick"))
        control_tick = _runtime_tick(evidence.get("control_tick"))
        effect_tick = _runtime_tick(evidence.get("native_effect_tick"))
        if (
            material_tick is None
            or decision_tick is None
            or control_tick is None
            or effect_tick is None
        ):
            continue
        decision = evidence.get("decision")
        control = evidence.get("control")
        native_effect = evidence.get("native_effect")
        if (
            not isinstance(decision, dict)
            or not isinstance(control, dict)
            or not isinstance(native_effect, dict)
        ):
            continue
        action_id = str(decision.get("action_id") or "")
        before_digest = str(native_effect.get("state_digest_before") or "")
        after_digest = str(native_effect.get("state_digest_after") or "")
        if (
            str(evidence.get("source_event_id") or "") in material_event_ids
            and str(evidence.get("ego_actor_id") or "") == identity_fields["ego_actor_id"]
            and str(evidence.get("conflict_actor_id") or "") == identity_fields["conflict_actor_id"]
            and 0 <= material_tick < decision_tick <= control_tick < effect_tick
            and decision.get("origin") == "agent_policy"
            and decision.get("policy_leg") == leg_name
            and bool(action_id)
            and str(control.get("action_id") or "") == action_id
            and control.get("applied_by_native_backend") is True
            and control.get("backend_kind") == "sumo_ego"
            and native_effect.get("observed_from_backend_step") is True
            and _valid_sha256(before_digest)
            and _valid_sha256(after_digest)
            and before_digest != after_digest
        ):
            return True
    return False


def _native_evidence(
    catalog_row: dict[str, Any],
    readiness_row: dict[str, Any],
    calibration: dict[str, Any],
    replay: dict[str, Any],
    fixture: dict[str, Any],
    bundle_evidence: dict[str, Any],
    implementation_binding: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(catalog_row.get("candidate_id") or "")
    derivation = dict(fixture.get("derivation") or {})
    identity_fields = {
        "candidate_id": candidate_id,
        "ego_actor_id": str(catalog_row.get("ego_actor_id") or ""),
        "conflict_actor_id": str(catalog_row.get("conflict_actor_id") or ""),
        "hazard_kind": str(catalog_row.get("hazard_kind") or ""),
        "source_window_sha256": str(catalog_row.get("source_window_sha256") or ""),
    }
    fixture_identity = (
        all(
            str(derivation.get(name) or "") == expected
            for name, expected in identity_fields.items()
        )
        and (derivation.get("window_semantics") or {}).get("configured_window_complete") is True
    )
    bundle_bytes_and_opened_hashes = (
        bundle_evidence.get("status") == "verified"
        and bundle_evidence.get("full_bundle_verified") is True
        and bundle_evidence.get("opened_source_hashes_verified") is True
        and str(bundle_evidence.get("source_window_sha256") or "")
        == identity_fields["source_window_sha256"]
        and str(bundle_evidence.get("source_event_chain_sha256") or "")
        == str(catalog_row.get("source_event_chain_sha256") or "")
        and _valid_sha256(bundle_evidence.get("input_binding_sha256"))
    )
    source_lock_valid = (
        fixture_identity
        and bundle_bytes_and_opened_hashes
        and _valid_sha256(catalog_row.get("source_window_sha256"))
        and _valid_sha256(catalog_row.get("source_event_chain_sha256"))
        and int(catalog_row.get("end_time_ms_exclusive") or 0)
        > int(catalog_row.get("start_time_ms") or 0)
    )

    scenario = dict(calibration.get("scenario") or {})
    backend = dict(scenario.get("backend_config") or {})
    calibration_digest = str(calibration.get("report_digest_sha256") or "")
    calibration_marker = calibration.get("_ledger_report_digest_valid")
    calibration_digest_valid = calibration_digest == _digest_without(
        calibration,
        "report_digest_sha256",
        "_ledger_report_digest_valid",
    ) and (calibration_marker is None or calibration_marker is True)
    replay_digest = str(replay.get("replay_digest_sha256") or "")
    replay_marker = replay.get("_ledger_replay_digest_valid")
    replay_digest_valid = replay_digest == _digest_without(
        replay,
        "replay_digest_sha256",
        "_ledger_replay_digest_valid",
    ) and (replay_marker is None or replay_marker is True)
    portable_artifact_digest_integrity = (
        _valid_sha256(calibration_digest)
        and calibration_digest_valid
        and _valid_sha256(replay_digest)
        and replay_digest_valid
    )
    calibration_identity = (
        calibration.get("schema_version") == "autonomous_driving_calibration_legs_v1"
        and calibration.get("status") == "diagnostic_complete"
        and calibration_digest_valid
        and scenario.get("domain") == "autonomous_driving"
        and scenario.get("backend_kind") == "sumo_ego"
        and str(backend.get("candidate_id") or "") == candidate_id
        and str(backend.get("ego_actor_id") or "") == identity_fields["ego_actor_id"]
        and (
            str(backend.get("source_bundle") or "") == "."
            or _path_identity(backend.get("source_bundle"))
            == _path_identity(catalog_row.get("bundle_path"))
        )
        and backend.get("execution_mode") == "live"
    )
    active_runtime_assurance = (
        str(backend.get("diagnostic_shield_mode") or "active").lower() == "active"
    )
    legs = {
        str(value.get("leg") or ""): value
        for value in calibration.get("legs") or []
        if isinstance(value, dict)
    }
    three_legs = set(legs) == LEG_NAMES and len(calibration.get("legs") or []) == 3
    source_runtime = (
        calibration_identity
        and three_legs
        and bundle_bytes_and_opened_hashes
        and all(
            leg.get("status") == "completed"
            and (leg.get("source_consumption") or {}).get("status") == "verified"
            and (leg.get("source_consumption") or {}).get("runtime_fidelity")
            == "native_live_sumo_reactive"
            and (leg.get("source_consumption") or {}).get("scenario_domain") == "autonomous_driving"
            and (leg.get("source_consumption") or {}).get("deterministic_source_trace") is True
            and (leg.get("source_consumption") or {}).get("evidence_from_scenario_config_only")
            is False
            and bool((leg.get("source_consumption") or {}).get("opened_source_sha256"))
            and bool(leg.get("source_events"))
            and all(
                isinstance(event, dict) and event.get("materiality_passed") is True
                for event in leg.get("source_events") or []
            )
            for leg in legs.values()
        )
    )
    safe = three_legs and all(
        int(leg.get("collision_count") or 0) == 0 and int(leg.get("road_departure_count") or 0) == 0
        for leg in legs.values()
    )
    shield_cost = _total_cost(legs.get("shield_only", {}))
    oracle_cost = _total_cost(legs.get("oracle_offline", {}))
    headroom = (
        round(shield_cost - oracle_cost, 6)
        if shield_cost is not None and oracle_cost is not None
        else None
    )
    attribution = dict(calibration.get("attribution") or {})
    oracle_attribution = dict((attribution.get("comparisons") or {}).get("oracle_offline") or {})
    raw_attributed_headroom = oracle_attribution.get("agent_incremental_value_vs_shield_only")
    raw_attributed_shield_cost = attribution.get("shield_only_total_cost")
    if isinstance(raw_attributed_headroom, int | float) and isinstance(
        raw_attributed_shield_cost, int | float
    ):
        attributed_headroom = round(float(raw_attributed_headroom), 6)
        attributed_shield_cost = round(float(raw_attributed_shield_cost), 6)
    else:
        attributed_headroom = None
        attributed_shield_cost = None
    shield_separated = (
        attribution.get("status") == "diagnostic"
        and headroom is not None
        and attributed_headroom == headroom
        and attributed_shield_cost == shield_cost
        and oracle_attribution.get("safety_regression_vs_shield_only") is False
        and oracle_attribution.get("prevention_credit_eligible") is True
    )
    replay_valid = (
        replay.get("schema_version") == "autonomous_driving_replay_audit_v1"
        and replay.get("status") == "verified"
        and str(replay.get("candidate_id") or "") == candidate_id
        and replay.get("deterministic_semantic_replay") is True
        and replay_digest_valid
    )
    portable_binding = {
        "candidate_id": candidate_id,
        "implementation_sha256": implementation_binding.get("autonomous_driving_slice_sha256"),
        "semantics_sha256": implementation_binding.get("semantics_sha256"),
        "runtime_sha256": implementation_binding.get("runtime_sha256"),
        "input_sha256": bundle_evidence.get("input_binding_sha256"),
        "source_window_sha256": catalog_row.get("source_window_sha256"),
        "source_event_chain_sha256": catalog_row.get("source_event_chain_sha256"),
    }
    calibration_report_digest = calibration_digest
    current_portable_evidence_binding = (
        all(_valid_sha256(value) for value in portable_binding.values() if value != candidate_id)
        and calibration.get("evidence_binding") == portable_binding
        and replay.get("evidence_binding")
        == {
            **portable_binding,
            "calibration_report_digest_sha256": calibration_report_digest,
        }
        and _valid_sha256(calibration_report_digest)
        and portable_artifact_digest_integrity
    )
    task_requirements = dict(backend.get("task_requirements") or {})
    post_change_runtime_evidence = {
        leg_name: _valid_post_change_runtime_chain(legs.get(leg_name, {}), identity_fields)
        for leg_name in ("rule_tactical", "oracle_offline")
    }
    high_extreme_post_change_response = (
        scenario.get("difficulty_level") in {"high", "extreme"}
        and int(task_requirements.get("required_review_interval_ticks") or 0) in {1, 2}
        and int(task_requirements.get("required_stable_dwell_ticks") or 0) >= 2
        and task_requirements.get("recovery_sequence")
        == [
            "request_minimal_risk_maneuver",
            "request_recovery_check",
            "authorize_recovery",
        ]
        and int(scenario.get("horizon_ticks") or 0) > 0
        and all(
            int(leg.get("records") or 0) >= int(scenario.get("horizon_ticks") or 0)
            for leg in legs.values()
        )
        and all(post_change_runtime_evidence.values())
    )
    readiness_gates = dict(readiness_row.get("gates") or {})
    readiness_match = (
        str(readiness_row.get("candidate_id") or "") == candidate_id
        and headroom is not None
        and round(float(readiness_row.get("oracle_headroom_vs_shield_only") or 0.0), 6) == headroom
    )
    independent_native_gates = {
        "source_identity_lock": source_lock_valid,
        "bundle_bytes_and_opened_hashes": bundle_bytes_and_opened_hashes,
        "portable_artifact_digest_integrity": portable_artifact_digest_integrity,
        "calibration_identity": calibration_identity,
        "active_runtime_assurance": active_runtime_assurance,
        "three_leg_presence": three_legs,
        "native_source_consumption": source_runtime,
        "collision_departure_safety": safe,
        "shield_attribution_separated": shield_separated,
        "deterministic_replay": replay_valid,
        "current_portable_evidence_binding": current_portable_evidence_binding,
        "readiness_binding": readiness_match,
        "difficulty_contract": bool(readiness_gates.get("difficulty_contract")),
        "high_extreme_post_change_response": high_extreme_post_change_response,
        "positive_oracle_headroom": headroom is not None and headroom > 0.0,
    }
    readiness_native_gates = all(bool(readiness_gates.get(name)) for name in REQUIRED_NATIVE_GATES)
    native_prefilter_passed = all(independent_native_gates.values()) and readiness_native_gates
    non_headroom_gates = {
        name: value
        for name, value in independent_native_gates.items()
        if name != "positive_oracle_headroom"
    }
    if native_prefilter_passed:
        blockers: list[str] = []
    elif all(non_headroom_gates.values()) and readiness_native_gates and (headroom or 0.0) <= 0.0:
        blockers = ["positive_oracle_headroom_missing"]
    else:
        blockers = [
            f"{name}_failed" for name, passed in independent_native_gates.items() if not passed
        ]
        if not readiness_native_gates:
            blockers.append("declared_native_readiness_gate_failed")
    return {
        "native_prefilter_passed": native_prefilter_passed,
        "gates": independent_native_gates,
        "blockers": sorted(set(blockers)),
        "shield_attribution_separated": shield_separated,
        "post_change_runtime_evidence": post_change_runtime_evidence,
        "bundle_bytes_and_opened_hashes": bundle_bytes_and_opened_hashes,
        "portable_artifact_digest_integrity": portable_artifact_digest_integrity,
        "current_portable_evidence_binding": current_portable_evidence_binding,
        "headroom": {
            "shield_only_cost": shield_cost,
            "oracle_cost": oracle_cost,
            "oracle_vs_shield_only": headroom,
        },
    }


def build_ledger(
    *,
    catalog: dict[str, Any],
    readiness: dict[str, Any],
    calibrations: dict[str, dict[str, Any]],
    replays: dict[str, dict[str, Any]],
    fixtures: dict[str, dict[str, Any]],
    bundle_evidence: dict[str, dict[str, Any]],
    external: dict[str, Any],
    additional_external: dict[str, Any],
    researched_driving_routes: list[dict[str, Any]],
    core_candidate_ids: set[str],
    implementation_binding: dict[str, Any],
    input_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    catalog_rows = catalog.get("bundles")
    readiness_rows = readiness.get("bundles")
    external_rows = external.get("recipes")
    additional_rows = additional_external.get("sources")
    if not isinstance(catalog_rows, list) or not isinstance(readiness_rows, list):
        raise ValueError("autonomous catalog/readiness rows missing")
    if not isinstance(external_rows, list):
        raise ValueError("external recipes missing")
    if not isinstance(additional_rows, list):
        raise ValueError("additional external sources missing")
    additional_rows = [
        row
        for row in additional_rows
        if isinstance(row, dict)
        and str(row.get("source_id") or "") in ADDITIONAL_EXTERNAL_SOURCE_IDS
    ]
    readiness_by_id = {
        str(row.get("candidate_id") or ""): row for row in readiness_rows if isinstance(row, dict)
    }
    rows: list[dict[str, Any]] = []
    for catalog_row in catalog_rows:
        if not isinstance(catalog_row, dict):
            raise ValueError("autonomous catalog row must be an object")
        candidate_id = str(catalog_row.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("autonomous candidate identity missing")
        if any(
            candidate_id not in mapping
            for mapping in (
                readiness_by_id,
                calibrations,
                replays,
                fixtures,
                bundle_evidence,
            )
        ):
            raise ValueError(f"autonomous candidate evidence missing: {candidate_id}")
        evidence = _native_evidence(
            catalog_row,
            readiness_by_id[candidate_id],
            calibrations[candidate_id],
            replays[candidate_id],
            fixtures[candidate_id],
            bundle_evidence[candidate_id],
            implementation_binding,
        )
        existing_core = candidate_id in core_candidate_ids
        if existing_core:
            disposition = "excluded_existing_core"
            blockers: list[str] = []
        elif not evidence["bundle_bytes_and_opened_hashes"]:
            disposition = "held_source_evidence"
            blockers = list(evidence["blockers"])
        elif not evidence["current_portable_evidence_binding"]:
            disposition = "held_stale_evidence"
            blockers = list(evidence["blockers"])
        elif evidence["native_prefilter_passed"]:
            disposition = "candidate_prefilter_passed"
            blockers = []
        else:
            disposition = "held_repair"
            blockers = list(evidence["blockers"])
        rows.append(
            {
                "input_id": candidate_id,
                "input_kind": "autonomous_driving_source_window",
                "domain": "autonomous_driving",
                "work_state": "terminal",
                "candidate_only": True,
                "core_admission_claimed": False,
                "existing_core_identity": existing_core,
                "disposition": disposition,
                "blockers": blockers,
                "recording_id": catalog_row.get("recording_id"),
                "hazard_kind": catalog_row.get("hazard_kind"),
                "source_identity": {
                    "bundle_id": catalog_row.get("bundle_id"),
                    "bundle_path": catalog_row.get("bundle_path"),
                    "source_window_sha256": catalog_row.get("source_window_sha256"),
                    "source_event_chain_sha256": catalog_row.get("source_event_chain_sha256"),
                    "start_time_ms": catalog_row.get("start_time_ms"),
                    "end_time_ms_exclusive": catalog_row.get("end_time_ms_exclusive"),
                    "ego_actor_id": catalog_row.get("ego_actor_id"),
                    "conflict_actor_id": catalog_row.get("conflict_actor_id"),
                },
                "native_gates": evidence["gates"],
                "shield_attribution_separated": evidence["shield_attribution_separated"],
                "post_change_runtime_evidence": evidence["post_change_runtime_evidence"],
                "headroom": evidence["headroom"],
                "next_stage": (
                    "license_resolution_then_bounded_protocol21"
                    if disposition == "candidate_prefilter_passed"
                    else "none_existing_core"
                    if disposition == "excluded_existing_core"
                    else "controllability_or_candidate_replacement"
                ),
            }
        )

    for external_row in external_rows:
        if not isinstance(external_row, dict):
            raise ValueError("external recipe must be an object")
        source_id = str(external_row.get("source_id") or "")
        if not source_id:
            raise ValueError("external source identity missing")
        raw_consumed = bool((external_row.get("external_source") or {}).get("raw_asset_consumed"))
        direct_core = bool(external_row.get("direct_core_admission"))
        if raw_consumed or direct_core:
            raise ValueError(f"external raw/core admission forbidden: {source_id}")
        rows.append(
            {
                "input_id": f"external::{source_id}",
                "input_kind": "external_benchmark_family",
                "domain": str((external_row.get("target") or {}).get("domain") or "unknown"),
                "work_state": "terminal",
                "candidate_only": True,
                "core_admission_claimed": False,
                "disposition": str(external_row.get("disposition") or "held"),
                "upstream_status": str(external_row.get("status") or "held"),
                "ready_for_full_protocol21": bool(external_row.get("ready_for_full_protocol21")),
                "blockers": sorted(
                    set(str(value) for value in external_row.get("blocker_codes") or [])
                ),
                "raw_external_asset_consumed": False,
                "local_asset_count": sum(
                    bool(asset.get("exists"))
                    for asset in external_row.get("native_source_assets") or []
                    if isinstance(asset, dict)
                ),
                "title": external_row.get("title"),
            }
        )
    for external_row in additional_rows:
        source_id = str(external_row.get("source_id") or "")
        rows.append(
            {
                "input_id": f"external::{source_id}",
                "input_kind": "external_benchmark_family",
                "domain": str(external_row.get("domain") or "unknown"),
                "backend_kind": external_row.get("backend_kind"),
                "work_state": "terminal",
                "candidate_only": True,
                "core_admission_claimed": False,
                "disposition": str(external_row.get("disposition") or "held"),
                "upstream_status": str(external_row.get("disposition") or "held"),
                "ready_for_full_protocol21": False,
                "blockers": sorted(set(str(value) for value in external_row.get("blockers") or [])),
                "raw_external_asset_consumed": False,
                "local_asset_count": int(
                    (
                        ((external_row.get("source") or {}).get("local_assets") or {}).get(
                            "asset_count"
                        )
                    )
                    or 0
                ),
                "title": external_row.get("title") or source_id,
            }
        )
    for route in researched_driving_routes:
        if not isinstance(route, dict):
            raise ValueError("researched autonomous-driving route must be an object")
        source_id = str(route.get("source_id") or "")
        blockers = sorted(set(str(value) for value in route.get("blockers") or []))
        if not source_id or not blockers:
            raise ValueError("researched autonomous-driving route identity/blockers missing")
        rows.append(
            {
                "input_id": f"driving-route::{source_id}",
                "input_kind": "researched_autonomous_driving_source_route",
                "domain": "autonomous_driving",
                "work_state": "terminal",
                "candidate_only": True,
                "core_admission_claimed": False,
                "disposition": str(route.get("disposition") or "held_source_access"),
                "blockers": blockers,
                "title": route.get("title") or source_id,
                "source_url": route.get("source_url"),
                "backend_target": route.get("backend_target"),
                "access_status": route.get("access_status"),
                "source_role": route.get("source_role"),
                "integration_rung": route.get("integration_rung", "pre_cloned"),
                "source_type_core_eligible": bool(route.get("source_type_core_eligible", False)),
                "current_core_eligible": False,
                "source_facts": dict(route.get("source_facts") or {}),
                "local_assets_present": False,
                "raw_external_asset_consumed": False,
                "bulk_download_permitted": False,
                "next_stage": route.get(
                    "next_stage", "license_review_then_single_shard_fetch_build"
                ),
            }
        )
    rows.sort(key=lambda row: str(row["input_id"]))
    identities = [str(row["input_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("terminal ledger input identity duplicate")
    dispositions = Counter(str(row["disposition"]) for row in rows)
    report: dict[str, Any] = {
        "schema_version": "autonomous-external-candidate-terminal-ledger-v1",
        "status": "complete_candidate_only",
        "candidate_only": True,
        "core_admission_claimed": False,
        "frozen_core_modified": False,
        "release_artifacts_modified": False,
        "implementation_binding": dict(implementation_binding),
        "input_bindings": dict(input_bindings or {}),
        "policy": {
            "one_terminal_row_per_input": True,
            "native_prefilter_is_not_core_admission": True,
            "shield_only_loss_is_separate_from_agent_incremental_value": True,
            "raw_restricted_external_data_copied": False,
            "bulk_external_driving_download_permitted": False,
            "researched_driving_routes_require_license_review": True,
            "provider_llm_is_not_a_native_data_prefilter_gate": True,
            "license_and_full_protocol_remain_admission_gates": True,
        },
        "constraints": {
            "core_admission_profile": "quality_core_v2",
            "candidate_only": True,
            "retained_final_core_gates": ["strategy_depth"],
            "removed_redundant_checks": [
                "provider_llm_as_native_data_prefilter",
                "duplicate_source_grounding_checks_outside_source_stage",
                "duplicate_agentic_checks_already_proven_by_native_stages",
            ],
            "retained_autonomous_driving_gates": [
                "ngsim_identity_and_source_window_lock",
                "native_live_sumo_source_consumption",
                "deterministic_semantic_replay",
                "native_beneficial_headroom",
                "collision_departure_safety",
                "native_task_and_difficulty_contract",
                "shield_vs_agent_attribution",
                "high_extreme_post_change_response",
            ],
        },
        "counts": {
            "input_rows": (
                len(catalog_rows)
                + len(external_rows)
                + len(additional_rows)
                + len(researched_driving_routes)
            ),
            "terminal_rows": len(rows),
            "autonomous_input_rows": len(catalog_rows),
            "external_family_rows": len(external_rows) + len(additional_rows),
            "researched_driving_route_rows": len(researched_driving_routes),
            "autonomous_native_prefilter_passed": sum(
                row["input_kind"] == "autonomous_driving_source_window"
                and row["disposition"] == "candidate_prefilter_passed"
                for row in rows
            ),
            "autonomous_held_repair": sum(
                row["input_kind"] == "autonomous_driving_source_window"
                and row["disposition"] == "held_repair"
                for row in rows
            ),
            "existing_core_excluded": sum(
                row["disposition"] == "excluded_existing_core" for row in rows
            ),
            "full_core_ready": 0,
            "dispositions": dict(sorted(dispositions.items())),
        },
        "global_admission_blockers": [
            "ngsim_license_metadata_discrepancy",
            "fresh_bounded_protocol21_pending_for_native_survivors",
        ],
        "rows": rows,
    }
    report["ledger_digest_sha256"] = _object_digest(report)
    return report


def _index_calibrations(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for candidate in sorted(path.glob("*.json")):
        report = _load(candidate)
        candidate_id = str(
            ((report.get("scenario") or {}).get("backend_config") or {}).get("candidate_id") or ""
        )
        if not candidate_id or candidate_id in rows:
            raise ValueError("calibration candidate identity missing or duplicate")
        expected_digest = str(report.get("report_digest_sha256") or "")
        report["_ledger_report_digest_valid"] = _valid_sha256(
            expected_digest
        ) and expected_digest == _digest_without(
            report,
            "report_digest_sha256",
            "_ledger_report_digest_valid",
        )
        rows[candidate_id] = report
    return rows


def _index_replays(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for candidate in sorted(path.glob("*.json")):
        report = _load(candidate)
        candidate_id = str(report.get("candidate_id") or "")
        if not candidate_id or candidate_id in rows:
            raise ValueError("replay candidate identity missing or duplicate")
        expected_digest = str(report.get("replay_digest_sha256") or "")
        report["_ledger_replay_digest_valid"] = _valid_sha256(
            expected_digest
        ) and expected_digest == _digest_without(
            report,
            "replay_digest_sha256",
            "_ledger_replay_digest_valid",
        )
        rows[candidate_id] = report
    return rows


def _fixtures(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for value in catalog.get("bundles") or []:
        candidate_id = str(value.get("candidate_id") or "")
        bundle = Path(str(value.get("bundle_path") or ""))
        if not bundle.is_absolute():
            bundle = REPO_ROOT / bundle
        rows[candidate_id] = _load(bundle / "runtime/fixture.json")
    return rows


def _bundle_evidence(
    catalog: dict[str, Any], calibrations: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    from domains.autonomous_driving.evidence_binding import verified_bundle_input_binding

    rows: dict[str, dict[str, Any]] = {}
    for catalog_row in catalog.get("bundles") or []:
        candidate_id = str(catalog_row.get("candidate_id") or "")
        bundle = Path(str(catalog_row.get("bundle_path") or ""))
        if not bundle.is_absolute():
            bundle = REPO_ROOT / bundle
        try:
            evidence = verified_bundle_input_binding(
                bundle,
                [dict(value) for value in calibrations[candidate_id].get("legs") or []],
            )
            rows[candidate_id] = {
                "status": "verified",
                "full_bundle_verified": True,
                "opened_source_hashes_verified": evidence["opened_source_hashes_verified"],
                "source_window_sha256": evidence["source_window_sha256"],
                "source_event_chain_sha256": evidence["source_event_chain_sha256"],
                "input_binding_sha256": evidence["input_sha256"],
            }
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
            rows[candidate_id] = {
                "status": "invalid",
                "full_bundle_verified": False,
                "opened_source_hashes_verified": False,
                "source_window_sha256": None,
                "source_event_chain_sha256": None,
                "input_binding_sha256": None,
                "blocker": str(error),
            }
    return rows


def _collect_candidate_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        candidate_id = value.get("candidate_id")
        if isinstance(candidate_id, str) and candidate_id.startswith("ngsim:"):
            found.add(candidate_id)
        for nested in value.values():
            found.update(_collect_candidate_ids(nested))
    elif isinstance(value, list):
        for nested in value:
            found.update(_collect_candidate_ids(nested))
    return found


def _implementation_binding() -> dict[str, Any]:
    from domains.autonomous_driving.evidence_binding import (
        runtime_implementation_binding,
        sumo_runtime_binding,
    )

    binding = runtime_implementation_binding(REPO_ROOT)
    runtime = sumo_runtime_binding(REPO_ROOT)
    return {
        **binding,
        "runtime_sha256": runtime["runtime_sha256"],
        "ledger_builder_sha256": _sha256(
            REPO_ROOT / "scripts/build_autonomous_external_candidate_terminal_ledger.py"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATIONS)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAYS)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--additional-external", type=Path, default=DEFAULT_ADDITIONAL_EXTERNAL)
    parser.add_argument("--core-suite", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    paths = [args.catalog, args.readiness, args.external, args.additional_external]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError([str(path) for path in paths if not path.is_file()])
    catalog = _load(args.catalog)
    calibrations = _index_calibrations(args.calibration_dir)
    core_ids: set[str] = set()
    for core_path in args.core_suite:
        core_ids.update(_collect_candidate_ids(_load(core_path)))
    input_bindings = {
        "catalog": {"path": str(args.catalog), "sha256": _sha256(args.catalog)},
        "readiness": {"path": str(args.readiness), "sha256": _sha256(args.readiness)},
        "external": {"path": str(args.external), "sha256": _sha256(args.external)},
        "additional_external": {
            "path": str(args.additional_external),
            "sha256": _sha256(args.additional_external),
        },
        "core_suites": [{"path": str(path), "sha256": _sha256(path)} for path in args.core_suite],
    }
    report = build_ledger(
        catalog=catalog,
        readiness=_load(args.readiness),
        calibrations=calibrations,
        replays=_index_replays(args.replay_dir),
        fixtures=_fixtures(catalog),
        bundle_evidence=_bundle_evidence(catalog, calibrations),
        external=_load(args.external),
        additional_external=_load(args.additional_external),
        researched_driving_routes=RESEARCHED_DRIVING_ROUTES,
        core_candidate_ids=core_ids,
        implementation_binding=_implementation_binding(),
        input_bindings=input_bindings,
    )
    if args.check:
        verified = args.output.is_file() and _load(args.output) == report
    else:
        if args.output.exists():
            raise FileExistsError("autonomous external terminal ledger output exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verified = True
    print(
        json.dumps(
            {
                "status": "verified" if verified else "stale",
                **report["counts"],
            },
            sort_keys=True,
        )
    )
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
