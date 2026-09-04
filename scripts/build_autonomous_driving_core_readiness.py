#!/usr/bin/env python3
"""Build a fail-closed Core-readiness report for autonomous-driving bundles.

This auditor consumes the independently built bundle catalog and native
calibration-leg reports.  It never edits the scenario registry and never
changes ``formal_core_allowed``; promotion remains a separate reviewed act.
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
REQUIRED_LEGS = ("oracle_offline", "rule_tactical", "shield_only")
REQUIRED_DIFFICULTIES = ("basic", "medium", "high", "extreme")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _blocker(code: str, scope: str, detail: str) -> dict[str, str]:
    return {"code": code, "scope": scope, "detail": detail}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _total_cost(leg: dict[str, Any]) -> float | None:
    costs = leg.get("cost_components")
    if not isinstance(costs, dict) or not costs:
        return None
    values = [_finite_number(value) for value in costs.values()]
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _calibration_candidate(report: dict[str, Any]) -> str:
    scenario = report.get("scenario")
    if not isinstance(scenario, dict):
        return ""
    backend = scenario.get("backend_config")
    if not isinstance(backend, dict):
        return ""
    return str(backend.get("candidate_id") or "")


def _catalog_bundle_path(catalog_row: dict[str, Any]) -> Path:
    path = Path(str(catalog_row.get("bundle_path") or ""))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _current_calibration_evidence_binding(
    *, catalog_row: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, str]:
    from domains.autonomous_driving.evidence_binding import calibration_evidence_binding

    legs = [dict(value) for value in calibration.get("legs") or [] if isinstance(value, dict)]
    return calibration_evidence_binding(
        repo_root=REPO_ROOT,
        bundle=_catalog_bundle_path(catalog_row),
        candidate_id=str(catalog_row.get("candidate_id") or ""),
        legs=legs,
    )


def _report_digest_valid(report: dict[str, Any], field: str) -> bool:
    expected = str(report.get(field) or "")
    unsigned = dict(report)
    unsigned.pop(field, None)
    return _valid_sha256(expected) and expected == _digest(unsigned)


def _evidence_binding_current(
    calibration: dict[str, Any],
    replay: dict[str, Any] | None,
    catalog_row: dict[str, Any],
) -> bool:
    if not _catalog_source_identity_present(catalog_row):
        return False
    if replay is None:
        return False
    try:
        expected = _current_calibration_evidence_binding(
            catalog_row=catalog_row, calibration=calibration
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False
    calibration_digest = str(calibration.get("report_digest_sha256") or "")
    replay_binding = {
        **expected,
        "calibration_report_digest_sha256": calibration_digest,
    }
    return bool(
        calibration.get("evidence_binding") == expected
        and replay.get("evidence_binding") == replay_binding
        and _report_digest_valid(calibration, "report_digest_sha256")
        and _report_digest_valid(replay, "replay_digest_sha256")
    )


def _calibration_identity_valid(report: dict[str, Any], catalog_row: dict[str, Any]) -> bool:
    """Bind native calibration to the exact source bundle and ego identity.

    Formal readiness requires the actor and bundle identity carried by the
    source catalog. Historical candidate-only fixtures remain diagnostic and
    cannot open this gate.
    """
    if not _catalog_source_identity_present(catalog_row):
        return False
    scenario = report.get("scenario")
    if not isinstance(scenario, dict):
        return False
    backend = scenario.get("backend_config")
    if not isinstance(backend, dict):
        return False
    candidate_id = str(catalog_row.get("candidate_id") or "")
    if str(backend.get("candidate_id") or "") != candidate_id:
        return False
    expected_ego = str(catalog_row.get("ego_actor_id") or "")
    if expected_ego and str(backend.get("ego_actor_id") or "") != expected_ego:
        return False
    expected_bundle = str(catalog_row.get("bundle_path") or "")
    actual_bundle = str(backend.get("source_bundle") or "")
    if expected_bundle and actual_bundle:
        if actual_bundle != "." and Path(actual_bundle).expanduser().resolve() != (
            _catalog_bundle_path(catalog_row)
        ):
            return False
    elif expected_bundle:
        return False
    if backend.get("execution_mode") != "live":
        return False
    requirements = backend.get("task_requirements")
    if not isinstance(requirements, dict):
        return False
    if int(requirements.get("required_review_interval_ticks") or 0) not in {1, 2}:
        return False
    if requirements.get("recovery_sequence") != [
        "request_minimal_risk_maneuver",
        "request_recovery_check",
        "authorize_recovery",
    ]:
        return False
    return int(requirements.get("required_stable_dwell_ticks") or 0) >= 2


def _catalog_source_identity_present(catalog_row: dict[str, Any]) -> bool:
    return bool(
        str(catalog_row.get("candidate_id") or "")
        and str(catalog_row.get("ego_actor_id") or "")
        and str(catalog_row.get("bundle_path") or "")
        and _valid_sha256(catalog_row.get("source_window_sha256"))
        and _valid_sha256(catalog_row.get("source_event_chain_sha256"))
    )


def _replay_contract_passed(
    report: dict[str, Any] | None,
    *,
    candidate_id: str,
    artifact_validated: bool,
    require_artifact_validation: bool,
) -> bool:
    if report is None:
        return False
    repeats = report.get("repeats")
    sources = report.get("repeat_sources")
    evidence = report.get("repeat_evidence")
    digests = report.get("leg_semantic_digests")
    schema_version = report.get("schema_version")
    structural = bool(
        schema_version
        in {
            "autonomous_driving_replay_audit_v1",
            "autonomous_driving_replay_audit_v2",
        }
        and (
            schema_version == "autonomous_driving_replay_audit_v1"
            or report.get("evidence_tier") == "formal_yaml_bound_v1"
        )
        and report.get("status") == "verified"
        and report.get("deterministic_semantic_replay") is True
        and str(report.get("candidate_id") or "") == candidate_id
        and repeats == 3
        and sources
        == [
            "reference_calibration",
            "fresh_native_replay",
            "fresh_native_replay",
        ]
        and isinstance(evidence, list)
        and len(evidence) == 3
        and len({str(value) for value in evidence}) == 3
        and isinstance(digests, dict)
        and set(digests) == set(REQUIRED_LEGS)
        and all(
            isinstance(values, list)
            and len(values) == 3
            and len(set(str(value) for value in values)) == 1
            and bool(str(values[0]))
            for values in digests.values()
        )
    )
    return structural and (artifact_validated or not require_artifact_validation)


def _index_legs(report: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], bool]:
    rows = report.get("legs")
    if not isinstance(rows, list):
        return {}, False
    indexed: dict[str, dict[str, Any]] = {}
    unique = True
    for value in rows:
        if not isinstance(value, dict):
            unique = False
            continue
        name = str(value.get("leg") or "")
        if not name or name in indexed:
            unique = False
            continue
        indexed[name] = dict(value)
    return indexed, unique


def _material_source_events(legs: dict[str, dict[str, Any]]) -> bool:
    for name in REQUIRED_LEGS:
        events = legs.get(name, {}).get("source_events")
        if not isinstance(events, list) or not events:
            return False
        for event in events:
            if not isinstance(event, dict) or event.get("materiality_passed") is not True:
                return False
            before = event.get("before_state_digest")
            after = event.get("after_state_digest")
            if (before is not None or after is not None) and (
                not _valid_sha256(before) or not _valid_sha256(after) or before == after
            ):
                return False
    return True


def _safe_legs(legs: dict[str, dict[str, Any]], names: tuple[str, ...]) -> bool:
    return all(
        legs.get(name, {}).get("status") == "completed"
        and legs.get(name, {}).get("collision_count") == 0
        and legs.get(name, {}).get("road_departure_count") == 0
        for name in names
    )


def _positive_oracle_headroom(legs: dict[str, dict[str, Any]]) -> tuple[bool, float | None]:
    shield_cost = _total_cost(legs.get("shield_only", {}))
    oracle_cost = _total_cost(legs.get("oracle_offline", {}))
    if shield_cost is None or oracle_cost is None:
        return False, None
    headroom = round(shield_cost - oracle_cost, 6)
    safe = _safe_legs(legs, ("shield_only", "oracle_offline"))
    return safe and headroom > 0.0, headroom


def _catalog_non_overlap(rows: list[dict[str, Any]], catalog: dict[str, Any]) -> bool:
    ordered: list[tuple[str, int, int, str]] = []
    for row in rows:
        try:
            start = int(row["start_time_ms"])
            end = int(row["end_time_ms_exclusive"])
        except (KeyError, TypeError, ValueError):
            return False
        if end <= start:
            return False
        ordered.append(
            (
                str(row.get("recording_id") or ""),
                start,
                end,
                str(row.get("candidate_id") or ""),
            )
        )
    ordered.sort()
    recomputed = all(
        current[0] != previous[0] or current[1] >= previous[2]
        for previous, current in zip(ordered, ordered[1:], strict=False)
    )
    declared = (catalog.get("structural_dedup") or {}).get("non_overlapping_windows")
    return bool(ordered) and recomputed and declared is True


def _catalog_recording_diversity(rows: list[dict[str, Any]], catalog: dict[str, Any]) -> bool:
    """Require independently declared recording/site coverage when present.

    Older pilot catalogs do not carry this optional structural field and keep
    their historical readiness shape.  Any catalog that declares it opts into
    the stricter Core gate, which prevents several windows from one recording
    being mistaken for independent naturalistic coverage.
    """
    structural = catalog.get("structural_dedup") or {}
    if "unique_recordings" not in structural:
        return True
    observed = {
        str(row.get("recording_id") or "") for row in rows if str(row.get("recording_id") or "")
    }
    try:
        declared = int(structural.get("unique_recordings") or 0)
    except (TypeError, ValueError):
        return False
    return declared >= 2 and len(observed) >= 2 and declared == len(observed)


def _license_review_passed(catalog: dict[str, Any]) -> bool:
    admission = catalog.get("admission") or {}
    passed = admission.get("license_review") in {"passed", "approved", "verified"}
    reasons = [str(value).lower() for value in catalog.get("core_denominator_reason") or []]
    return passed and not any("license" in reason for reason in reasons)


def _inventory_core_target_passed(
    inventory: dict[str, Any], catalog_ids: set[str], expected_catalog_sha256: str | None
) -> tuple[bool, list[dict[str, str]]]:
    """Require the versioned source-pool inventory before Core admission.

    The inventory is an independent, fail-closed accounting of source windows.
    It prevents a catalog with four YAML slices per window from appearing to
    meet a multi-recording/multi-hazard Core target.
    """
    blockers: list[dict[str, str]] = []
    if not _report_digest_valid(inventory, "inventory_digest_sha256"):
        blockers.append(
            _blocker(
                "core_inventory_digest_invalid",
                "inventory",
                "source-pool inventory self-digest is missing or stale",
            )
        )
    if inventory.get("verification_mode") != "full_bundle_verify":
        blockers.append(
            _blocker(
                "core_inventory_full_verification_required",
                "inventory",
                "Core admission requires full bundle verification",
            )
        )
    catalog_input = (inventory.get("inputs") or {}).get("catalog") or {}
    if (
        not expected_catalog_sha256
        or str(catalog_input.get("sha256") or "") != expected_catalog_sha256
    ):
        blockers.append(
            _blocker(
                "core_inventory_catalog_binding_stale",
                "inventory",
                "inventory must bind the exact readiness catalog bytes",
            )
        )
    if inventory.get("schema_version") != "autonomous_driving_core_inventory_v1":
        blockers.append(
            _blocker(
                "core_inventory_schema_invalid",
                "inventory",
                "source-pool inventory schema is not recognized",
            )
        )
    if (
        inventory.get("status") != "held"
        or inventory.get("registry_mutation_performed") is not False
    ):
        blockers.append(
            _blocker(
                "core_inventory_mutation_or_status_invalid",
                "inventory",
                "inventory must be a held, non-mutating audit artifact",
            )
        )
    discovery_errors = inventory.get("discovery_errors") or []
    if discovery_errors:
        blockers.append(
            _blocker(
                "core_inventory_discovery_errors",
                "inventory",
                f"{len(discovery_errors)} bundle artifacts failed manifest identity checks",
            )
        )
    tier = dict((inventory.get("tier_assessment") or {}).get("minimal_core") or {})
    if tier.get("structurally_sufficient") is not True:
        blockers.append(
            _blocker(
                "source_pool_core_target_missing",
                "inventory",
                "minimal Core target requires at least 12 non-duplicated source windows, "
                "3 recordings, and 3 hazard families",
            )
        )
    selected = dict(inventory.get("selected_catalog") or {})
    selected_count = int(selected.get("source_window_denominator_keys") or 0)
    if selected_count < 12:
        blockers.append(
            _blocker(
                "selected_catalog_denominator_too_small",
                "inventory",
                f"selected catalog has {selected_count} source-window keys; 12 are required",
            )
        )
    if set(str(value) for value in selected.get("candidate_ids") or []) != catalog_ids:
        blockers.append(
            _blocker(
                "inventory_catalog_identity_mismatch",
                "inventory",
                "inventory selected candidate IDs do not equal readiness catalog IDs",
            )
        )
    if selected_count and not bool(selected.get("difficulty_slices_are_not_denominator_keys")):
        blockers.append(
            _blocker(
                "difficulty_denominator_contract_missing",
                "inventory",
                "difficulty slices must be explicitly excluded from the source denominator",
            )
        )
    difficulty_counts = dict(selected.get("difficulty_slice_count_by_level") or {})
    if selected_count and any(
        int(difficulty_counts.get(level) or 0) != selected_count for level in REQUIRED_DIFFICULTIES
    ):
        blockers.append(
            _blocker(
                "difficulty_slice_coverage_missing",
                "inventory",
                "each selected source-window key must have basic, medium, high, and extreme slices",
            )
        )
    passed = not blockers
    return passed, blockers


def _declared_suite_levels(report: dict[str, Any]) -> tuple[str, ...]:
    """Return the scenario levels that this candidate declares for admission.

    Historical suite reports omit ``primary_difficulty`` and therefore retain
    their four-slice contract.  New candidate reports may bind one calibrated
    primary scenario; additional difficulty slices are release diagnostics.
    """
    primary = report.get("primary_difficulty")
    if primary is None:
        return REQUIRED_DIFFICULTIES
    level = str(primary)
    return (level,) if level in REQUIRED_DIFFICULTIES else ()


def _suite_contract_passed(report: dict[str, Any]) -> bool:
    """Check the source-bound scenario contract declared for this candidate."""
    schema_version = str(report.get("schema_version") or "")
    if schema_version not in {
        "autonomous_driving_suite_report_v1",
        "autonomous_driving_suite_report_v2",
    }:
        return False
    if report.get("status") != "held":
        return False
    feasibility = report.get("difficulty_feasibility")
    if not isinstance(feasibility, dict):
        return False
    expected_levels = list(_declared_suite_levels(report))
    if not expected_levels:
        return False
    if any((feasibility.get(level) or {}).get("status") != "included" for level in expected_levels):
        return False
    slices = report.get("difficulty_slices")
    if not isinstance(slices, list) or len(slices) != len(expected_levels):
        return False
    if any(not isinstance(path, str) or not path.endswith(".yaml") for path in slices):
        return False
    profiles = report.get("difficulty_profiles")
    if not isinstance(profiles, dict):
        return False
    if any(level not in profiles for level in expected_levels):
        return False
    profile_rows = [profiles[level] for level in expected_levels]
    if any(not isinstance(row, dict) for row in profile_rows):
        return False
    tool_budgets = [int(row.get("max_tool_calls_per_tick") or 0) for row in profile_rows]
    plan_depths = [int(row.get("plan_dependency_depth") or 0) for row in profile_rows]
    delays = [int(row.get("worst_case_declared_tactical_delay_ticks") or 0) for row in profile_rows]
    review_intervals = [int(row.get("review_interval_ticks") or 0) for row in profile_rows]
    if (
        tool_budgets != sorted(tool_budgets)
        or plan_depths != sorted(plan_depths)
        or review_intervals != sorted(review_intervals, reverse=True)
    ):
        return False
    if delays != sorted(delays):
        return False
    if not all(
        bool(profiles[level].get("requires_paid_safety_inspection"))
        is (level in {"high", "extreme"})
        for level in expected_levels
    ):
        return False
    if any(
        str(profiles[level].get("observation_regime") or "")
        != (
            "full_local_scene"
            if level in {"basic", "medium"}
            else "partial_actor_velocity_until_paid_inspection"
        )
        for level in expected_levels
    ):
        return False
    if schema_version == "autonomous_driving_suite_report_v2":
        if str(report.get("execution_mode") or "") != "live":
            return False
        observed_events = report.get("source_event_sequence")
        expected_events = report.get("source_event_expected_sequence")
        evolution = report.get("environment_evolution")
        if (
            not isinstance(observed_events, list)
            or not observed_events
            or not all(isinstance(value, str) and value for value in observed_events)
            or not isinstance(expected_events, list)
            or not expected_events
            or not all(isinstance(value, str) and value for value in expected_events)
            or not isinstance(evolution, dict)
            or evolution.get("source_native") is not True
            or evolution.get("ordered_by_source_time") is not True
            or evolution.get("interruptible_after_standing_plan") is not True
            or list(evolution.get("expected_event_kinds") or []) != expected_events
            or list(evolution.get("observed_event_kinds") or []) != observed_events
        ):
            return False
        expected_sequence = [
            "request_minimal_risk_maneuver",
            "request_recovery_check",
            "authorize_recovery",
        ]
        dilemmas = [str(profiles[level].get("decision_dilemma") or "") for level in expected_levels]
        if not all(dilemmas) or len(set(dilemmas)) != len(dilemmas):
            return False
        required_tools = [
            [str(tool) for tool in profiles[level].get("required_supervisory_tools") or []]
            for level in expected_levels
        ]
        if any(not tools for tools in required_tools):
            return False
        if any(
            not set(required_tools[index]).issubset(set(required_tools[index + 1]))
            for index in range(len(required_tools) - 1)
        ):
            return False
        observation_tools = [
            [str(tool) for tool in profiles[level].get("required_observation_tools") or []]
            for level in expected_levels
        ]
        if any(
            not set(observation_tools[index]).issubset(set(observation_tools[index + 1]))
            for index in range(len(observation_tools) - 1)
        ):
            return False
        expected_observation_tools = {
            "basic": [],
            "medium": ["inspect_local_scene"],
            "high": ["inspect_local_scene", "inspect_safety_state"],
            "extreme": [
                "inspect_local_scene",
                "inspect_safety_state",
                "inspect_odd_status",
            ],
        }
        if any(
            observation_tools[index] != expected_observation_tools[level]
            for index, level in enumerate(expected_levels)
        ):
            return False
        if any(
            profiles[level].get("requires_preventive_action") is not True
            or set(profiles[level].get("preventive_action_tools") or {})
            != {"set_driving_envelope", "request_tactical_maneuver"}
            or int(profiles[level].get("minimum_decision_epochs") or 0) < 1
            for level in expected_levels
        ):
            return False
        if any(
            str(profiles[level].get("decision_dilemma") or "") == ""
            or profiles[level].get("recovery_sequence") != expected_sequence
            or int(profiles[level].get("review_interval_ticks") or 0) < 1
            for level in expected_levels
        ):
            return False
    long_horizon = report.get("long_horizon")
    if not isinstance(long_horizon, dict):
        return False
    horizons = long_horizon.get("horizon_ticks")
    dwell = long_horizon.get("post_recovery_dwell_ticks")
    if not isinstance(horizons, dict) or not isinstance(dwell, dict):
        return False
    if not all(
        isinstance(horizons.get(level), int)
        and horizons[level] >= 8
        and isinstance(dwell.get(level), int)
        and dwell[level] >= 2
        for level in expected_levels
    ):
        return False
    ordered_horizons = [horizons[level] for level in expected_levels]
    ordered_dwell = [dwell[level] for level in expected_levels]
    if ordered_horizons != sorted(ordered_horizons) or ordered_dwell != sorted(ordered_dwell):
        return False
    if schema_version == "autonomous_driving_suite_report_v2":
        source_seconds = long_horizon.get("source_window_seconds")
        minimum_ticks = long_horizon.get("minimum_supervisory_ticks")
        if (
            not isinstance(source_seconds, int | float)
            or isinstance(source_seconds, bool)
            or float(source_seconds) < 40.0
            or not isinstance(minimum_ticks, int)
            or minimum_ticks < 10
            or any(horizons[level] < minimum_ticks for level in expected_levels)
        ):
            return False
    return True


def _suite_yaml_contract_passed(
    report_path: Path,
    report: dict[str, Any],
    catalog_row: dict[str, Any],
) -> tuple[bool, list[Path]]:
    from core.scenario_validator import validate_scenario_yaml

    slices = report.get("difficulty_slices")
    long_horizon = report.get("long_horizon")
    if not isinstance(slices, list) or not isinstance(long_horizon, dict):
        return False, []
    horizons = long_horizon.get("horizon_ticks")
    if not isinstance(horizons, dict):
        return False, []
    suite_root = report_path.parent.resolve()
    expected_candidate = str(catalog_row.get("candidate_id") or "")
    expected_ego = str(catalog_row.get("ego_actor_id") or "")
    expected_bundle = _catalog_bundle_path(catalog_row)
    expected_window = str(catalog_row.get("source_window_sha256") or "")
    expected_event_chain = str(catalog_row.get("source_event_chain_sha256") or "")
    expected_levels = set(_declared_suite_levels(report))
    if not expected_levels:
        return False, []
    observed_levels: set[str] = set()
    resolved_files: list[Path] = []
    for value in slices:
        if not isinstance(value, str):
            return False, resolved_files
        path = (suite_root / value).resolve()
        try:
            path.relative_to(suite_root)
        except ValueError:
            return False, resolved_files
        if not path.is_file():
            return False, resolved_files
        try:
            scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            return False, resolved_files
        if not isinstance(scenario, dict) or validate_scenario_yaml(scenario, path):
            return False, resolved_files
        level = str(scenario.get("difficulty_level") or "")
        backend = scenario.get("backend_config")
        provenance = scenario.get("provenance")
        if (
            level not in expected_levels
            or level in observed_levels
            or scenario.get("domain") != "autonomous_driving"
            or scenario.get("horizon_ticks") != horizons.get(level)
            or str(scenario.get("source_window_sha256") or "") != expected_window
            or not isinstance(backend, dict)
            or str(backend.get("candidate_id") or "") != expected_candidate
            or str(backend.get("ego_actor_id") or "") != expected_ego
            or backend.get("execution_mode") != "live"
            or not isinstance(provenance, dict)
            or str(provenance.get("candidate_id") or "") != expected_candidate
            or str(provenance.get("source_event_chain_sha256") or "") != expected_event_chain
        ):
            return False, resolved_files
        source_bundle = Path(str(backend.get("source_bundle") or ""))
        resolved_bundle = (
            source_bundle.resolve()
            if source_bundle.is_absolute()
            else (REPO_ROOT / source_bundle).resolve()
        )
        if resolved_bundle != expected_bundle:
            return False, resolved_files
        observed_levels.add(level)
        resolved_files.append(path)
    return observed_levels == expected_levels, resolved_files


def _calibration_horizon_covers_suite(
    calibration: dict[str, Any] | None,
    replay: dict[str, Any] | None,
    suite: dict[str, Any] | None,
) -> bool:
    if calibration is None or replay is None or suite is None:
        return False
    scenario = calibration.get("scenario")
    long_horizon = suite.get("long_horizon")
    if not isinstance(scenario, dict) or not isinstance(long_horizon, dict):
        return False
    horizons = long_horizon.get("horizon_ticks")
    if not isinstance(horizons, dict):
        return False
    declared_levels = _declared_suite_levels(suite)
    if not declared_levels:
        return False
    calibration_level = str(scenario.get("difficulty_level") or "")
    primary_level = str(suite.get("primary_difficulty") or calibration_level)
    if primary_level not in declared_levels:
        return False
    required = horizons.get(primary_level)
    observed = scenario.get("horizon_ticks")
    return bool(
        isinstance(required, int)
        and isinstance(observed, int)
        and observed >= required
        and calibration_level == primary_level
        and replay.get("difficulty_level") == primary_level
        and replay.get("ticks") == observed
    )


def _calibration_scenario_contract_bound(
    calibration: dict[str, Any] | None,
    suite: dict[str, Any] | None,
) -> bool:
    if calibration is None or suite is None:
        return False
    expected_by_level = suite.get("__scenario_artifact_by_difficulty__")
    artifact = calibration.get("scenario_artifact")
    if not isinstance(expected_by_level, dict) or not isinstance(artifact, dict):
        return False
    scenario = calibration.get("scenario")
    if not isinstance(scenario, dict):
        return False
    level = str(suite.get("primary_difficulty") or scenario.get("difficulty_level") or "")
    if level not in _declared_suite_levels(suite):
        return False
    expected = expected_by_level.get(level)
    return bool(
        isinstance(expected, dict)
        and artifact.get("schema_version") == "autonomous_driving_scenario_artifact_v1"
        and _valid_sha256(artifact.get("scenario_yaml_sha256"))
        and _valid_sha256(artifact.get("semantic_contract_sha256"))
        and artifact.get("scenario_yaml_sha256") == expected.get("scenario_yaml_sha256")
        and artifact.get("semantic_contract_sha256") == expected.get("semantic_contract_sha256")
    )


def _llm_evidence_candidate_ids(
    reports: list[tuple[str, dict[str, Any]]] | None,
    *,
    expected_scenario_artifacts: dict[str, dict[str, dict[str, Any]]] | None = None,
    require_suite_binding: bool = False,
) -> set[str]:
    """Return only candidates covered by a complete provider-backed report."""
    eligible: set[str] = set()
    for _name, report in reports or []:
        if report.get("schema_version") != "autonomous_driving_llm_evidence_v1":
            continue
        if report.get("status") != "verified" or report.get("blockers"):
            continue
        if require_suite_binding and not _report_digest_valid(report, "evidence_digest_sha256"):
            continue
        required = {str(level) for level in report.get("required_difficulties") or []}
        if required != set(REQUIRED_DIFFICULTIES):
            continue
        runs = report.get("runs")
        if not isinstance(runs, list) or any(
            not isinstance(run, dict) or run.get("status") != "verified" for run in runs
        ):
            continue
        candidate_ids = {str(value) for value in report.get("candidate_ids") or [] if str(value)}
        if require_suite_binding:
            expected_binding = {
                f"{candidate_id}:{level}": artifact
                for candidate_id in sorted(candidate_ids)
                for level, artifact in sorted(
                    (expected_scenario_artifacts or {}).get(candidate_id, {}).items()
                )
            }
            if report.get("suite_semantic_coverage_sha256") != _digest(expected_binding):
                continue
        for candidate_id in candidate_ids:
            expected_levels = (expected_scenario_artifacts or {}).get(candidate_id, {})
            levels = {
                str(run.get("difficulty_level") or "")
                for run in runs
                if str(run.get("candidate_id") or "") == candidate_id
            }
            run_bindings_match = not require_suite_binding or all(
                any(
                    str(run.get("candidate_id") or "") == candidate_id
                    and str(run.get("difficulty_level") or "") == level
                    and run.get("scenario_yaml_sha256") == artifact.get("scenario_yaml_sha256")
                    and run.get("semantic_contract_sha256")
                    == artifact.get("semantic_contract_sha256")
                    and run.get("horizon_ticks") == artifact.get("horizon_ticks")
                    for run in runs
                )
                for level, artifact in expected_levels.items()
            )
            if (
                levels == set(REQUIRED_DIFFICULTIES)
                and (
                    not require_suite_binding or set(expected_levels) == set(REQUIRED_DIFFICULTIES)
                )
                and run_bindings_match
            ):
                eligible.add(candidate_id)
    return eligible


def _bundle_readiness(
    catalog_row: dict[str, Any],
    calibration: dict[str, Any] | None,
    calibration_file: str | None,
    replay: dict[str, Any] | None,
    replay_file: str | None,
    suite: dict[str, Any] | None,
    suite_file: str | None,
    suite_required: bool,
    llm_candidate_ids: set[str],
    replay_artifact_validated: bool,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    candidate_id = str(catalog_row.get("candidate_id") or "")
    try:
        catalog_source_event_count = int(catalog_row.get("source_event_count") or 0)
    except (TypeError, ValueError):
        catalog_source_event_count = 0
    blockers: list[dict[str, str]] = []
    if calibration is None:
        gates = {
            "calibration_report_valid": False,
            "source_event_materiality": False,
            "three_leg_presence": False,
            "collision_departure_safety": False,
            "positive_oracle_headroom": False,
            "active_runtime_assurance": False,
            "reactive_closed_loop": False,
            "actual_llm_evaluation": False,
            "deterministic_replay": False,
            "evidence_binding_current": False,
            "suite_primary_execution_binding": False,
        }
        blockers.append(
            _blocker(
                "calibration_report_missing",
                candidate_id,
                "no calibration JSON binds this catalog candidate",
            )
        )
        for code, detail in (
            (
                "source_event_materiality_failed",
                "source-event materiality has no calibration evidence",
            ),
            (
                "three_leg_calibration_missing",
                "shield_only, rule_tactical, and oracle_offline are missing",
            ),
            (
                "collision_departure_safety_failed",
                "collision/departure safety has no calibration evidence",
            ),
            (
                "positive_oracle_headroom_missing",
                "oracle headroom has no calibration evidence",
            ),
            (
                "active_runtime_assurance_required",
                "formal admission requires an enforcing, non-diagnostic shield",
            ),
            (
                "reactive_closed_loop_validation_missing",
                "native reactive replay has no calibration evidence",
            ),
            (
                "deterministic_replay_missing",
                "three same-seed isolated replays with identical semantic digests are missing",
            ),
            (
                "evidence_binding_stale",
                "calibration and replay must match current bundle, runtime, and implementation hashes",
            ),
            (
                "scenario_contract_binding_missing",
                "calibration must execute the exact current primary YAML contract",
            ),
        ):
            blockers.append(_blocker(code, candidate_id, detail))
        legs: dict[str, dict[str, Any]] = {}
        headroom = None
    else:
        legs, unique_legs = _index_legs(calibration)
        three_legs = unique_legs and all(name in legs for name in REQUIRED_LEGS)
        catalog_materiality = bool(
            catalog_source_event_count > 0
            and _valid_sha256(catalog_row.get("source_event_chain_sha256"))
            and _valid_sha256(catalog_row.get("source_window_sha256"))
        )
        materiality = catalog_materiality and three_legs and _material_source_events(legs)
        safety = three_legs and _safe_legs(legs, REQUIRED_LEGS)
        headroom_passed, headroom = _positive_oracle_headroom(legs)
        scenario = calibration.get("scenario") or {}
        backend = scenario.get("backend_config") or {}
        active_runtime_assurance = bool(
            str(backend.get("diagnostic_shield_mode") or "active").lower() == "active"
            and backend.get("unsafe_diagnostic_acknowledged") is not True
            and backend.get("diagnostic_only") is not True
        )
        calibration_schema = calibration.get("schema_version")
        calibration_schema_valid = bool(
            calibration_schema
            in {
                "autonomous_driving_calibration_legs_v1",
                "autonomous_driving_calibration_legs_v2",
            }
            and (
                calibration_schema == "autonomous_driving_calibration_legs_v1"
                or calibration.get("evidence_tier") == "formal_yaml_bound_v1"
            )
        )
        reactive = bool(
            calibration_schema_valid
            and calibration.get("status") == "diagnostic_complete"
            and calibration.get("leg_isolation") == "subprocess_per_leg"
            and backend.get("execution_mode") == "live"
            and three_legs
            and all(legs[name].get("status") == "completed" for name in REQUIRED_LEGS)
        )
        source_traces = [legs[name].get("source_consumption") for name in REQUIRED_LEGS]
        if any(isinstance(trace, dict) for trace in source_traces):
            reactive = reactive and all(
                isinstance(trace, dict)
                and trace.get("status") == "verified"
                and trace.get("runtime_fidelity") == "native_live_sumo_reactive"
                and trace.get("blockers") == []
                for trace in source_traces
            )
        # These reports come from the baseline calibration harness.  A leg
        # label cannot establish provider/model identity or strict-prompt LLM
        # execution, so actual LLM evidence is intentionally out of scope.
        actual_llm = candidate_id in llm_candidate_ids
        replay_passed = _replay_contract_passed(
            replay,
            candidate_id=candidate_id,
            artifact_validated=replay_artifact_validated,
            require_artifact_validation=_catalog_source_identity_present(catalog_row),
        )
        calibration_valid = bool(
            calibration_schema_valid
            and _calibration_candidate(calibration) == candidate_id
            and _calibration_identity_valid(calibration, catalog_row)
        )
        evidence_binding_current = _evidence_binding_current(calibration, replay, catalog_row)
        gates = {
            "calibration_report_valid": calibration_valid,
            "source_event_materiality": materiality,
            "three_leg_presence": three_legs,
            "collision_departure_safety": safety,
            "positive_oracle_headroom": headroom_passed,
            "active_runtime_assurance": active_runtime_assurance,
            "reactive_closed_loop": reactive,
            "actual_llm_evaluation": actual_llm,
            "deterministic_replay": replay_passed,
            "evidence_binding_current": evidence_binding_current,
            "suite_primary_execution_binding": True,
        }
        failure_details = {
            "calibration_report_valid": (
                "calibration schema or candidate binding is invalid",
                "calibration_report_invalid",
            ),
            "source_event_materiality": (
                "each baseline leg must observe only material source events",
                "source_event_materiality_failed",
            ),
            "three_leg_presence": (
                "shield_only, rule_tactical, and oracle_offline must appear once",
                "three_leg_calibration_missing",
            ),
            "collision_departure_safety": (
                "a required leg collided or departed the drivable surface",
                "collision_departure_safety_failed",
            ),
            "positive_oracle_headroom": (
                "oracle cost must be strictly below shield-only cost without safety regression",
                "positive_oracle_headroom_missing",
            ),
            "active_runtime_assurance": (
                "formal admission requires active enforcement and forbids unsafe diagnostics",
                "active_runtime_assurance_required",
            ),
            "reactive_closed_loop": (
                "isolated native live-SUMO three-leg replay is missing",
                "reactive_closed_loop_validation_missing",
            ),
            "deterministic_replay": (
                "three same-seed isolated replays must have identical semantic digests",
                "deterministic_replay_missing",
            ),
            "evidence_binding_current": (
                "calibration and replay must match current bundle, runtime, and implementation hashes",
                "evidence_binding_stale",
            ),
            "suite_primary_execution_binding": (
                "calibration must execute the exact current primary YAML contract",
                "scenario_contract_binding_missing",
            ),
        }
        for gate, passed in gates.items():
            if gate != "actual_llm_evaluation" and not passed:
                detail, code = failure_details[gate]
                blockers.append(_blocker(code, candidate_id, detail))

    suite_passed = not suite_required
    horizon_coverage = not suite_required
    if suite_required:
        suite_passed = bool(
            suite is not None
            and str(suite.get("candidate_id") or "") == candidate_id
            and str(suite.get("source_window_sha256") or "")
            == str(catalog_row.get("source_window_sha256") or "")
            and str(suite.get("source_event_chain_sha256") or "")
            == str(catalog_row.get("source_event_chain_sha256") or "")
            and _suite_contract_passed(suite)
            and (
                suite.get("__yaml_contract_validated__") is True
                or not _catalog_source_identity_present(catalog_row)
            )
        )
        if not suite_passed:
            blockers.append(
                _blocker(
                    "difficulty_contract_missing",
                    candidate_id,
                    "one source-bound primary scenario with recovery dwell is required",
                )
            )
        horizon_coverage = suite_passed and _calibration_horizon_covers_suite(
            calibration, replay, suite
        )
        if not horizon_coverage:
            blockers.append(
                _blocker(
                    "calibration_horizon_coverage_missing",
                    candidate_id,
                    "calibration and replay must cover the full declared primary scenario horizon",
                )
            )
        scenario_binding = bool(
            suite_passed
            and (
                not _catalog_source_identity_present(catalog_row)
                or _calibration_scenario_contract_bound(calibration, suite)
            )
        )
        gates["suite_primary_execution_binding"] = scenario_binding
        if not scenario_binding:
            blockers.append(
                _blocker(
                    "scenario_contract_binding_missing",
                    candidate_id,
                    "calibration must execute the exact current primary YAML contract",
                )
            )

    candidate_admission_gates = {
        key: value for key, value in gates.items() if key != "actual_llm_evaluation"
    }
    candidate_admission_gates.update(
        {
            "difficulty_contract": suite_passed,
            "calibration_horizon_coverage": horizon_coverage,
        }
    )

    return (
        {
            "bundle_path": catalog_row.get("bundle_path"),
            "candidate_id": candidate_id,
            "source_window_sha256": catalog_row.get("source_window_sha256"),
            "catalog_source_event_count": catalog_source_event_count,
            "calibration_file": calibration_file,
            "replay_file": replay_file,
            "suite_file": suite_file,
            "oracle_headroom_vs_shield_only": headroom,
            "observed_legs": sorted(legs),
            "gates": {
                **gates,
                "difficulty_contract": suite_passed,
                "calibration_horizon_coverage": horizon_coverage,
            },
            "candidate_admission_gates": candidate_admission_gates,
            "diagnostics": {"actual_llm_evaluation": gates["actual_llm_evaluation"]},
            "ready_for_full_admission": all(candidate_admission_gates.values()),
        },
        blockers,
    )


def build_readiness(
    catalog: dict[str, Any],
    calibrations: list[tuple[str, dict[str, Any]]],
    *,
    replays: list[tuple[str, dict[str, Any]]] | None = None,
    suites: list[tuple[str, dict[str, Any]]] | None = None,
    llm_evidence: list[tuple[str, dict[str, Any]]] | None = None,
    inventory: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
    validated_replay_files: set[str] | None = None,
) -> dict[str, Any]:
    rows = [dict(value) for value in catalog.get("bundles") or [] if isinstance(value, dict)]
    rows.sort(key=lambda row: str(row.get("candidate_id") or ""))
    blockers: list[dict[str, str]] = []
    if catalog.get("schema_version") != "autonomous_driving_candidate_catalog_v1":
        blockers.append(_blocker("catalog_schema_invalid", "catalog", "unexpected catalog schema"))
    candidate_ids = [str(row.get("candidate_id") or "") for row in rows]
    if (
        not rows
        or any(not value for value in candidate_ids)
        or len(set(candidate_ids)) != len(rows)
    ):
        blockers.append(
            _blocker(
                "catalog_candidate_identity_invalid",
                "catalog",
                "catalog candidates must be present and unique",
            )
        )
    source_identity = bool(rows) and all(_catalog_source_identity_present(row) for row in rows)
    if not source_identity:
        blockers.append(
            _blocker(
                "catalog_source_identity_missing",
                "catalog",
                "every Core candidate requires ego, bundle, source-window, and source-event identity",
            )
        )
    non_overlap = _catalog_non_overlap(rows, catalog)
    if not non_overlap:
        blockers.append(
            _blocker(
                "source_window_overlap_detected",
                "catalog",
                "catalog source windows overlap or the declared overlap gate is false",
            )
        )
    recording_diversity_required = "unique_recordings" in (catalog.get("structural_dedup") or {})
    recording_diversity = _catalog_recording_diversity(rows, catalog)
    indexed: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    unbound: list[str] = []
    for name, calibration_report in sorted(calibrations, key=lambda value: value[0]):
        candidate_id = _calibration_candidate(calibration_report)
        if candidate_id:
            indexed.setdefault(candidate_id, []).append((name, calibration_report))
        else:
            unbound.append(name)
    for name in unbound:
        blockers.append(
            _blocker(
                "calibration_candidate_binding_missing",
                "calibration",
                name,
            )
        )
    replay_indexed: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for name, replay_report in sorted(replays or [], key=lambda value: value[0]):
        candidate_id = str(replay_report.get("candidate_id") or "")
        if candidate_id:
            replay_indexed.setdefault(candidate_id, []).append((name, replay_report))
    suite_required = suites is not None or source_identity
    suite_indexed: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for name, suite_report in sorted(suites or [], key=lambda value: value[0]):
        candidate_id = str(suite_report.get("candidate_id") or "")
        if candidate_id:
            suite_indexed.setdefault(candidate_id, []).append((name, suite_report))
    expected_scenario_artifacts = {
        candidate_id: dict(matches[0][1].get("__scenario_artifact_by_difficulty__") or {})
        for candidate_id, matches in suite_indexed.items()
        if len(matches) == 1
    }
    llm_candidate_ids = _llm_evidence_candidate_ids(
        llm_evidence,
        expected_scenario_artifacts=expected_scenario_artifacts,
        require_suite_binding=source_identity,
    )
    bundle_reports: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "")
        matches = indexed.get(candidate_id, [])
        if len(matches) > 1:
            blockers.append(
                _blocker(
                    "duplicate_calibration_report",
                    candidate_id,
                    ",".join(name for name, _report in matches),
                )
            )
        selected = matches[0] if len(matches) == 1 else None
        replay_matches = replay_indexed.get(candidate_id, [])
        if len(replay_matches) > 1:
            blockers.append(
                _blocker(
                    "duplicate_replay_report",
                    candidate_id,
                    ",".join(name for name, _report in replay_matches),
                )
            )
        replay_selected = replay_matches[0] if len(replay_matches) == 1 else None
        suite_matches = suite_indexed.get(candidate_id, [])
        if len(suite_matches) > 1:
            blockers.append(
                _blocker(
                    "duplicate_suite_report",
                    candidate_id,
                    ",".join(name for name, _report in suite_matches),
                )
            )
        suite_selected = suite_matches[0] if len(suite_matches) == 1 else None
        bundle_report, bundle_blockers = _bundle_readiness(
            row,
            selected[1] if selected else None,
            selected[0] if selected else None,
            replay_selected[1] if replay_selected else None,
            replay_selected[0] if replay_selected else None,
            suite_selected[1] if suite_selected else None,
            suite_selected[0] if suite_selected else None,
            suite_required,
            llm_candidate_ids,
            replay_selected is not None and replay_selected[0] in (validated_replay_files or set()),
        )
        bundle_reports.append(bundle_report)
        blockers.extend(bundle_blockers)
    unknown_candidates = sorted(set(indexed) - set(candidate_ids))
    for candidate_id in unknown_candidates:
        blockers.append(
            _blocker(
                "calibration_candidate_not_in_catalog",
                candidate_id,
                "calibration candidate is outside the catalog",
            )
        )
    unknown_replay_candidates = sorted(set(replay_indexed) - set(candidate_ids))
    for candidate_id in unknown_replay_candidates:
        blockers.append(
            _blocker(
                "replay_candidate_not_in_catalog",
                candidate_id,
                "replay candidate is outside the catalog",
            )
        )
    unknown_suite_candidates = sorted(set(suite_indexed) - set(candidate_ids))
    for candidate_id in unknown_suite_candidates:
        blockers.append(
            _blocker(
                "suite_candidate_not_in_catalog",
                candidate_id,
                "suite candidate is outside the catalog",
            )
        )
    llm_report_candidates = {
        candidate_id
        for _name, report in llm_evidence or []
        for candidate_id in report.get("candidate_ids") or []
        if str(candidate_id)
    }
    for candidate_id in sorted(llm_report_candidates - set(candidate_ids)):
        blockers.append(
            _blocker(
                "llm_evidence_candidate_not_in_catalog",
                candidate_id,
                "provider-backed evidence is outside the selected catalog",
            )
        )
    license_passed = _license_review_passed(catalog)
    if not license_passed:
        blockers.append(
            _blocker(
                "license_review_pending",
                "catalog",
                "redistribution/license compatibility is not approved in the catalog",
            )
        )
    global_gates = {
        "catalog_source_identity": source_identity,
        "source_event_materiality": bool(bundle_reports)
        and (catalog.get("admission") or {}).get("source_event_materiality") == "passed"
        and all(row["gates"]["source_event_materiality"] for row in bundle_reports),
        "three_leg_presence": bool(bundle_reports)
        and all(row["gates"]["three_leg_presence"] for row in bundle_reports),
        "collision_departure_safety": bool(bundle_reports)
        and all(row["gates"]["collision_departure_safety"] for row in bundle_reports),
        "non_overlapping_windows": non_overlap,
        "positive_oracle_headroom": bool(bundle_reports)
        and all(row["gates"]["positive_oracle_headroom"] for row in bundle_reports),
        "active_runtime_assurance": bool(bundle_reports)
        and all(row["gates"]["active_runtime_assurance"] for row in bundle_reports),
        "reactive_closed_loop": bool(bundle_reports)
        and all(row["gates"]["reactive_closed_loop"] for row in bundle_reports),
        "license_review": license_passed,
        "deterministic_replay": bool(bundle_reports)
        and all(row["gates"]["deterministic_replay"] for row in bundle_reports),
        "evidence_binding_current": bool(bundle_reports)
        and all(row["gates"]["evidence_binding_current"] for row in bundle_reports),
    }
    if suite_required:
        global_gates["difficulty_contract"] = bool(bundle_reports) and all(
            row["gates"]["difficulty_contract"] for row in bundle_reports
        )
        global_gates["calibration_horizon_coverage"] = bool(bundle_reports) and all(
            row["gates"]["calibration_horizon_coverage"] for row in bundle_reports
        )
        global_gates["suite_primary_execution_binding"] = bool(bundle_reports) and all(
            row["gates"]["suite_primary_execution_binding"] for row in bundle_reports
        )
    release_maturity_diagnostics = {
        "actual_llm_evaluation": bool(bundle_reports)
        and all(row["gates"]["actual_llm_evaluation"] for row in bundle_reports),
        "recording_diversity": recording_diversity
        if recording_diversity_required
        else None,
    }
    if inventory is not None:
        inventory_passed, inventory_blockers = _inventory_core_target_passed(
            inventory,
            set(candidate_ids),
            str((inputs or {}).get("catalog_sha256") or "") or None,
        )
        release_maturity_diagnostics["source_pool_core_target"] = inventory_passed
        release_maturity_diagnostics["source_pool_core_target_reasons"] = inventory_blockers
    elif source_identity:
        release_maturity_diagnostics["source_pool_core_target"] = False
        release_maturity_diagnostics["source_pool_core_target_reasons"] = [
            _blocker(
                "source_pool_inventory_missing",
                "inventory",
                "release composition inventory is not supplied",
            )
        ]
    candidate_admission_gates = dict(global_gates)
    evidence_gates_passed = all(candidate_admission_gates.values())
    blockers = sorted(
        blockers,
        key=lambda value: (value["scope"], value["code"], value["detail"]),
    )
    report: dict[str, Any] = {
        "schema_version": "autonomous_driving_core_readiness_v2"
        if inventory is not None
        else "autonomous_driving_core_readiness_v1",
        "status": "held",
        "formal_core_allowed": False,
        "registry_mutation_performed": False,
        "candidate_count": len(bundle_reports),
        "inputs": dict(inputs or {}),
        "global_gates": global_gates,
        "candidate_admission_gates": candidate_admission_gates,
        "release_maturity_diagnostics": release_maturity_diagnostics,
        "bundles": bundle_reports,
        "blockers": blockers,
        "evidence_gates_passed": evidence_gates_passed,
        "admission_disposition": (
            "manual_registry_review_required"
            if evidence_gates_passed
            else "held_until_blockers_resolved"
        ),
    }
    report["readiness_digest_sha256"] = _digest(report)
    return report


def build_readiness_from_paths(
    catalog_path: Path,
    calibration_dir: Path,
    replay_dir: Path | None = None,
    suite_dir: Path | None = None,
    llm_evidence_dir: Path | None = None,
    inventory_path: Path | None = None,
) -> dict[str, Any]:
    catalog_path = catalog_path.resolve()
    catalog = _load(catalog_path)
    catalog_rows = {
        str(row.get("candidate_id") or ""): row
        for row in catalog.get("bundles") or []
        if isinstance(row, dict)
    }
    calibration_dir = calibration_dir.resolve()
    if replay_dir is not None:
        replay_dir = replay_dir.resolve()
    if suite_dir is not None:
        suite_dir = suite_dir.resolve()
    if llm_evidence_dir is not None:
        llm_evidence_dir = llm_evidence_dir.resolve()
    if inventory_path is not None:
        inventory_path = inventory_path.resolve()
    # Replay audits live in a separate directory and must not be mistaken for
    # baseline calibration reports.  Core calibration inputs are one level
    # deep so the file-to-candidate binding remains explicit.
    calibration_paths = sorted(calibration_dir.glob("*.json"))
    calibrations = [
        (path.relative_to(calibration_dir).as_posix(), _load(path)) for path in calibration_paths
    ]
    replay_paths = sorted(replay_dir.glob("*.json")) if replay_dir is not None else []
    replay_evidence_paths = sorted(replay_dir.rglob("*.json")) if replay_dir is not None else []
    replays = (
        [(path.relative_to(replay_dir).as_posix(), _load(path)) for path in replay_paths]
        if replay_dir is not None
        else []
    )
    suite_paths = sorted(suite_dir.glob("*/suite_report.json")) if suite_dir is not None else []
    suite_evidence_paths = list(suite_paths)
    suites: list[tuple[str, dict[str, Any]]] | None = None
    if suite_dir is not None:
        suites = []
        for path in suite_paths:
            suite = _load(path)
            row = catalog_rows.get(str(suite.get("candidate_id") or ""))
            valid = False
            declared_paths: list[Path] = []
            if row is not None:
                valid, declared_paths = _suite_yaml_contract_passed(path, suite, row)
            suite["__yaml_contract_validated__"] = valid
            if valid:
                scenario_artifacts: dict[str, dict[str, str]] = {}
                for scenario_path in declared_paths:
                    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
                    scenario_artifacts[str(scenario["difficulty_level"])] = {
                        "scenario_yaml_sha256": _sha256(scenario_path),
                        "semantic_contract_sha256": _digest(scenario),
                        "horizon_ticks": scenario["horizon_ticks"],
                    }
                suite["__scenario_artifact_by_difficulty__"] = scenario_artifacts
            suite_evidence_paths.extend(declared_paths)
            suites.append((path.relative_to(suite_dir).as_posix(), suite))
    llm_paths = sorted(llm_evidence_dir.glob("*.json")) if llm_evidence_dir is not None else []
    llm_evidence = (
        [(path.relative_to(llm_evidence_dir).as_posix(), _load(path)) for path in llm_paths]
        if llm_evidence_dir is not None
        else None
    )
    validated_replay_files: set[str] = set()
    if replay_dir is not None:
        from scripts.run_autonomous_driving_core_calibration_batch import (
            validate_replay_report,
        )

        calibration_by_candidate = {
            _calibration_candidate(report): calibration_dir / name
            for name, report in calibrations
            if _calibration_candidate(report)
        }
        for name, replay_report in replays:
            candidate_id = str(replay_report.get("candidate_id") or "")
            row = catalog_rows.get(candidate_id)
            calibration_path = calibration_by_candidate.get(candidate_id)
            if row is None or calibration_path is None:
                continue
            try:
                calibration = _load(calibration_path)
                scenario = dict(calibration.get("scenario") or {})
                validate_replay_report(
                    replay_dir / name,
                    candidate_id=candidate_id,
                    ego_actor_id=str(row.get("ego_actor_id") or ""),
                    bundle=_catalog_bundle_path(row),
                    calibration_path=calibration_path,
                    seed=int(calibration.get("seed") or -1),
                    ticks=int(scenario.get("horizon_ticks") or 0),
                    repeats=3,
                    difficulty_level=str(scenario.get("difficulty_level") or ""),
                    scenario_artifact=(dict(calibration.get("scenario_artifact") or {}) or None),
                )
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
                continue
            validated_replay_files.add(name)
    return build_readiness(
        catalog,
        calibrations,
        replays=replays,
        suites=suites,
        llm_evidence=llm_evidence,
        inventory=_load(inventory_path) if inventory_path is not None else None,
        validated_replay_files=validated_replay_files,
        inputs={
            "catalog_sha256": _sha256(catalog_path),
            "calibration_files": [
                {
                    "path": path.relative_to(calibration_dir).as_posix(),
                    "sha256": _sha256(path),
                }
                for path in calibration_paths
            ],
            "replay_files": [
                {
                    "path": path.relative_to(replay_dir).as_posix(),
                    "sha256": _sha256(path),
                }
                for path in replay_evidence_paths
            ]
            if replay_dir is not None
            else [],
            "suite_files": [
                {
                    "path": path.relative_to(suite_dir).as_posix(),
                    "sha256": _sha256(path),
                }
                for path in sorted(set(suite_evidence_paths))
            ]
            if suite_dir is not None
            else [],
            "suite_coverage_sha256": _digest(
                {
                    path.relative_to(suite_dir).as_posix(): _sha256(path)
                    for path in sorted(set(suite_evidence_paths))
                }
            )
            if suite_dir is not None
            else None,
            "llm_evidence_files": [
                {
                    "path": path.relative_to(llm_evidence_dir).as_posix(),
                    "sha256": _sha256(path),
                }
                for path in llm_paths
            ]
            if llm_evidence_dir is not None
            else [],
            "inventory": {
                "path": inventory_path.relative_to(REPO_ROOT).as_posix()
                if inventory_path is not None
                else None,
                "sha256": _sha256(inventory_path) if inventory_path is not None else None,
            },
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--llm-evidence-dir", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    catalog = args.catalog if args.catalog.is_absolute() else REPO_ROOT / args.catalog
    calibration_dir = (
        args.calibration_dir
        if args.calibration_dir.is_absolute()
        else REPO_ROOT / args.calibration_dir
    )
    replay_dir = (
        (args.replay_dir if args.replay_dir.is_absolute() else REPO_ROOT / args.replay_dir)
        if args.replay_dir is not None
        else None
    )
    suite_dir = (
        (args.suite_dir if args.suite_dir.is_absolute() else REPO_ROOT / args.suite_dir)
        if args.suite_dir is not None
        else None
    )
    llm_evidence_dir = (
        (
            args.llm_evidence_dir
            if args.llm_evidence_dir.is_absolute()
            else REPO_ROOT / args.llm_evidence_dir
        )
        if args.llm_evidence_dir is not None
        else None
    )
    inventory_path = (
        (args.inventory if args.inventory.is_absolute() else REPO_ROOT / args.inventory)
        if args.inventory is not None
        else None
    )
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    try:
        report = build_readiness_from_paths(
            catalog,
            calibration_dir,
            replay_dir,
            suite_dir,
            llm_evidence_dir,
            inventory_path,
        )
        if args.check:
            if not output.is_file():
                raise ValueError("autonomous_driving_core_readiness_report_missing")
            current = _load(output)
            verified = current == report
        else:
            if output.exists():
                raise FileExistsError("autonomous_driving_core_readiness_output_exists")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            verified = True
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "verified" if verified else "stale",
                "candidate_count": report["candidate_count"],
                "formal_core_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
