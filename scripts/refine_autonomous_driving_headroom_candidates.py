#!/usr/bin/env python3
"""Build a reports-only, fail-closed NGSIM headroom replacement ledger.

Measured headroom is eligible only when an exact catalog source identity,
source-native miner row, native three-leg calibration, and deterministic replay
agree.  Unmeasured mined windows are queued for materialization and native
probing; they never satisfy the headroom target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
REQUIRED_LEGS = ("oracle_offline", "rule_tactical", "shield_only")
SOURCE_EVENT_TYPES_BY_HAZARD = {
    "lead_vehicle_braking": {"lead_vehicle_braking"},
    "lane_change_conflict": {"lane_change_conflict"},
    "minimum_time_headway_conflict": {"short_time_headway_boundary"},
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _digest(payload: dict[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _resolved(value: Any) -> Path:
    path = Path(str(value or ""))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _recording_id(row: dict[str, Any]) -> str:
    declared = str(row.get("recording_id") or "")
    if declared:
        return declared
    context = dict(row.get("hazard_context") or {})
    ego = str(context.get("ego_actor_id") or row.get("ego_actor_id") or "")
    return ego.split(":", 1)[0] if ":" in ego else ""


def _source_identity(row: dict[str, Any]) -> tuple[str, str, int, int, str, str, str, str]:
    context = dict(row.get("hazard_context") or {})
    return (
        str(row.get("candidate_id") or ""),
        str(row.get("source_window_sha256") or ""),
        int(row.get("start_time_ms") or 0),
        int(row.get("end_time_ms_exclusive") or 0),
        str(context.get("hazard_kind") or row.get("hazard_kind") or ""),
        str(context.get("ego_actor_id") or row.get("ego_actor_id") or ""),
        str(context.get("conflict_actor_id") or row.get("conflict_actor_id") or ""),
        _recording_id(row),
    )


def _hazard_identity_sha256(row: dict[str, Any]) -> str:
    identity = _source_identity(row)
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def _source_row_valid(row: dict[str, Any]) -> bool:
    identity = _source_identity(row)
    context = dict(row.get("hazard_context") or {})
    return bool(
        identity[0]
        and _valid_sha256(identity[1])
        and identity[2] > 0
        and identity[3] > identity[2]
        and all(identity[index] for index in (4, 5, 6, 7))
        and context.get("phase_window_complete") is True
        and int(context.get("supervisory_prevention_window_ms") or 0) >= 15_000
        and int(context.get("recovery_window_ms") or 0) >= 20_000
    )


def _index_miner_reports(
    miner_reports: list[tuple[str, dict[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    indexed: dict[str, dict[str, Any]] = {}
    evidence: dict[str, list[str]] = {}
    for name, report in sorted(miner_reports, key=lambda value: value[0]):
        if report.get("schema_version") != "ngsim_mining_report_v1":
            raise ValueError("autonomous_driving_refine_miner_schema_invalid")
        rows = report.get("candidates")
        if not isinstance(rows, list):
            raise ValueError("autonomous_driving_refine_miner_candidates_invalid")
        for value in rows:
            if not isinstance(value, dict):
                raise ValueError("autonomous_driving_refine_miner_candidate_invalid")
            row = dict(value)
            candidate_id = str(row.get("candidate_id") or "")
            if not candidate_id:
                raise ValueError("autonomous_driving_refine_miner_candidate_id_missing")
            previous = indexed.get(candidate_id)
            if previous is not None and _source_identity(previous) != _source_identity(row):
                raise ValueError("autonomous_driving_refine_miner_candidate_identity_conflict")
            if previous is None:
                indexed[candidate_id] = row
            evidence.setdefault(candidate_id, []).append(name)
    return indexed, evidence


def _index_catalog(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    if catalog.get("schema_version") != "autonomous_driving_candidate_catalog_v1":
        raise ValueError("autonomous_driving_refine_catalog_schema_invalid")
    raw_rows = catalog.get("bundles")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("autonomous_driving_refine_catalog_empty")
    rows: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    source_keys: set[str] = set()
    for value in raw_rows:
        if not isinstance(value, dict):
            raise ValueError("autonomous_driving_refine_catalog_row_invalid")
        row = dict(value)
        candidate_id, source_key, start, end, hazard, ego, conflict, recording = _source_identity(
            row
        )
        if (
            not candidate_id
            or candidate_id in candidate_ids
            or not _valid_sha256(source_key)
            or source_key in source_keys
            or start <= 0
            or end <= start
            or not all((hazard, ego, conflict, recording))
            or not _valid_sha256(row.get("source_event_chain_sha256"))
        ):
            raise ValueError("autonomous_driving_refine_catalog_identity_invalid")
        candidate_ids.add(candidate_id)
        source_keys.add(source_key)
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["candidate_id"]))


def _cost(leg: dict[str, Any]) -> float | None:
    components = leg.get("cost_components")
    if not isinstance(components, dict) or not components:
        return None
    values: list[float] = []
    for value in components.values():
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        values.append(number)
    return round(sum(values), 6)


def _expected_evidence_binding(
    row: dict[str, Any],
    implementation_binding: dict[str, Any],
    current_evidence_bindings: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    implementation_sha256 = str(implementation_binding.get("autonomous_driving_slice_sha256") or "")
    semantics_sha256 = str(implementation_binding.get("semantics_sha256") or "")
    if not _valid_sha256(implementation_sha256) or not _valid_sha256(semantics_sha256):
        raise ValueError("autonomous_driving_refine_implementation_binding_invalid")
    identity = {
        "candidate_id": str(row.get("candidate_id") or ""),
        "implementation_sha256": implementation_sha256,
        "semantics_sha256": semantics_sha256,
        "source_window_sha256": str(row.get("source_window_sha256") or ""),
        "source_event_chain_sha256": str(row.get("source_event_chain_sha256") or ""),
    }
    current = current_evidence_bindings.get(identity["candidate_id"])
    if (
        not isinstance(current, dict)
        or set(current)
        != {
            *identity,
            "input_sha256",
            "runtime_sha256",
        }
        or any(current.get(name) != value for name, value in identity.items())
        or not _valid_sha256(current.get("input_sha256"))
        or not _valid_sha256(current.get("runtime_sha256"))
    ):
        return None
    return dict(current)


def _calibration_binding_valid(
    row: dict[str, Any],
    report: dict[str, Any],
    implementation_binding: dict[str, Any],
    current_evidence_bindings: dict[str, dict[str, str]],
) -> bool:
    observed = dict(report.get("evidence_binding") or {})
    expected = _expected_evidence_binding(row, implementation_binding, current_evidence_bindings)
    return expected is not None and observed == expected


def _index_legs(report: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    values = report.get("legs")
    if not isinstance(values, list) or len(values) != len(REQUIRED_LEGS):
        return None
    indexed = {
        str(value.get("leg") or ""): dict(value) for value in values if isinstance(value, dict)
    }
    return indexed if set(indexed) == set(REQUIRED_LEGS) else None


def _calibration_assessment(
    row: dict[str, Any],
    name: str,
    report: dict[str, Any],
    implementation_binding: dict[str, Any],
    current_evidence_bindings: dict[str, dict[str, str]],
) -> tuple[list[str], dict[str, dict[str, Any]] | None, dict[str, float | None]]:
    blockers: list[str] = []
    if (
        report.get("schema_version") != "autonomous_driving_calibration_legs_v1"
        or report.get("status") != "diagnostic_complete"
        or str(report.get("report_digest_sha256") or "") != _digest(report, "report_digest_sha256")
    ):
        blockers.append("calibration_report_invalid")
    if not _calibration_binding_valid(
        row, report, implementation_binding, current_evidence_bindings
    ):
        blockers.append("calibration_runtime_fingerprint_stale")
    scenario = dict(report.get("scenario") or {})
    backend = dict(scenario.get("backend_config") or {})
    candidate_id, _source_key, _start, _end, hazard, ego, _conflict, _recording = _source_identity(
        row
    )
    if (
        str(backend.get("candidate_id") or "") != candidate_id
        or str(backend.get("ego_actor_id") or "") != ego
        or (
            str(backend.get("source_bundle") or "") != "."
            and _resolved(backend.get("source_bundle")) != _resolved(row.get("bundle_path"))
        )
        or backend.get("execution_mode") != "live"
    ):
        blockers.append("calibration_source_identity_mismatch")
    if str(backend.get("diagnostic_shield_mode") or "active").lower() != "active":
        blockers.append("active_runtime_assurance_required")
    if scenario.get("difficulty_level") not in {"high", "extreme"}:
        blockers.append("high_or_extreme_calibration_missing")
    legs = _index_legs(report)
    if legs is None:
        blockers.append("three_leg_attribution_missing")
        return sorted(set(blockers)), None, {}
    for leg in legs.values():
        source = dict(leg.get("source_consumption") or {})
        events = leg.get("source_events")
        expected_event_types = SOURCE_EVENT_TYPES_BY_HAZARD.get(hazard, {hazard})
        source_event_valid = (
            isinstance(events, list)
            and bool(events)
            and all(
                isinstance(event, dict)
                and event.get("materiality_passed") is True
                and str(event.get("event_id") or "").startswith(candidate_id)
                for event in events
            )
            and any(
                isinstance(event, dict) and str(event.get("type") or "") in expected_event_types
                for event in events
            )
        )
        if (
            leg.get("status") != "completed"
            or source.get("status") != "verified"
            or source.get("runtime_fidelity") != "native_live_sumo_reactive"
            or source.get("runtime_trace_observed") is not True
            or source.get("deterministic_source_trace") is not True
            or not source_event_valid
        ):
            blockers.append("source_native_three_leg_evidence_missing")
            break
    if any(
        int(leg.get("collision_count") or 0) != 0 or int(leg.get("road_departure_count") or 0) != 0
        for leg in legs.values()
    ):
        blockers.append("collision_or_departure_detected")
    costs = {name: _cost(legs[name]) for name in REQUIRED_LEGS}
    if any(value is None for value in costs.values()):
        blockers.append("three_leg_cost_missing")
    else:
        shield = float(costs["shield_only"] or 0.0)
        oracle = float(costs["oracle_offline"] or 0.0)
        measured = round(shield - oracle, 6)
        comparison = dict(
            ((report.get("attribution") or {}).get("comparisons") or {}).get("oracle_offline") or {}
        )
        declared = comparison.get("agent_incremental_value_vs_shield_only")
        if (
            isinstance(declared, bool)
            or not isinstance(declared, int | float)
            or round(float(declared), 6) != measured
            or comparison.get("safety_regression_vs_shield_only") is not False
        ):
            blockers.append("attribution_headroom_mismatch")
    return sorted(set(blockers)), legs, costs


def _replay_blockers(
    row: dict[str, Any],
    calibration_name: str,
    legs: dict[str, dict[str, Any]] | None,
    report: dict[str, Any],
    implementation_binding: dict[str, Any],
    current_evidence_bindings: dict[str, dict[str, str]],
    calibration_report: dict[str, Any],
) -> list[str]:
    candidate_id, _source_key, _start, _end, _hazard, ego, _conflict, _recording = _source_identity(
        row
    )
    blockers: list[str] = []
    if (
        report.get("schema_version") != "autonomous_driving_replay_audit_v1"
        or report.get("status") != "verified"
        or report.get("deterministic_semantic_replay") is not True
        or str(report.get("replay_digest_sha256") or "") != _digest(report, "replay_digest_sha256")
    ):
        blockers.append("deterministic_replay_invalid")
    expected_replay_binding = {
        **dict(calibration_report.get("evidence_binding") or {}),
        "calibration_report_digest_sha256": str(
            calibration_report.get("report_digest_sha256") or ""
        ),
    }
    if (
        not _calibration_binding_valid(
            row,
            calibration_report,
            implementation_binding,
            current_evidence_bindings,
        )
        or report.get("evidence_binding") != expected_replay_binding
        or not _valid_sha256(expected_replay_binding["calibration_report_digest_sha256"])
    ):
        blockers.append("replay_runtime_fingerprint_stale")
    if (
        str(report.get("candidate_id") or "") != candidate_id
        or str(report.get("ego_actor_id") or "") != ego
        or (
            str(report.get("bundle") or "") != "."
            and _resolved(report.get("bundle")) != _resolved(row.get("bundle_path"))
        )
        or Path(str(report.get("reference_calibration") or "")).name != Path(calibration_name).name
    ):
        blockers.append("replay_source_identity_mismatch")
    repeats = int(report.get("repeats") or 0)
    digests = report.get("leg_semantic_digests")
    if (
        repeats < 3
        or report.get("repeat_sources")
        != ["reference_calibration", *["fresh_native_replay"] * (repeats - 1)]
        or not isinstance(digests, dict)
        or set(digests) != set(REQUIRED_LEGS)
        or any(
            not isinstance(values, list)
            or len(values) != repeats
            or len(set(str(value) for value in values)) != 1
            or not str(values[0])
            for values in digests.values()
        )
    ):
        blockers.append("three_repeat_replay_missing")
    elif legs is not None and any(
        str(digests[name][0]) != str(legs[name].get("semantic_digest") or "")
        for name in REQUIRED_LEGS
    ):
        blockers.append("replay_calibration_semantic_mismatch")
    return sorted(set(blockers))


def _overlaps(row: dict[str, Any], accepted: list[dict[str, Any]]) -> bool:
    _candidate, _source, start, end, _hazard, _ego, _conflict, recording = _source_identity(row)
    for other in accepted:
        (
            _other_candidate,
            _other_source,
            other_start,
            other_end,
            _other_hazard,
            _other_ego,
            _other_conflict,
            other_recording,
        ) = _source_identity(other)
        if recording == other_recording and start < other_end and other_start < end:
            return True
    return False


def _summary_row(
    row: dict[str, Any], *, miner_evidence: list[str], disposition: str
) -> dict[str, Any]:
    candidate_id, source_key, start, end, hazard, ego, conflict, recording = _source_identity(row)
    return {
        "candidate_id": candidate_id,
        "source_denominator_key": source_key,
        "hazard_identity_sha256": _hazard_identity_sha256(row),
        "recording_id": recording,
        "hazard_kind": hazard,
        "ego_actor_id": ego,
        "conflict_actor_id": conflict,
        "start_time_ms": start,
        "end_time_ms_exclusive": end,
        "miner_evidence": sorted(set(miner_evidence)),
        "disposition": disposition,
    }


def build_refinement_report(
    *,
    catalog: dict[str, Any],
    calibrations: dict[str, tuple[str, dict[str, Any]]],
    replays: dict[str, tuple[str, dict[str, Any]]],
    miner_reports: list[tuple[str, dict[str, Any]]],
    implementation_binding: dict[str, Any],
    current_evidence_bindings: dict[str, dict[str, str]],
    target_count: int = 12,
    minimum_recordings: int = 3,
    minimum_hazard_families: int = 3,
    minimum_headroom_exclusive: float = 0.0,
    queue_limit: int | None = None,
) -> dict[str, Any]:
    if target_count < 1 or minimum_recordings < 1 or minimum_hazard_families < 1:
        raise ValueError("autonomous_driving_refine_target_invalid")
    if not math.isfinite(minimum_headroom_exclusive):
        raise ValueError("autonomous_driving_refine_headroom_threshold_invalid")
    catalog_rows = _index_catalog(catalog)
    miner_index, miner_evidence = _index_miner_reports(miner_reports)
    known_ids = {str(row["candidate_id"]) for row in catalog_rows}
    if (
        set(calibrations) - known_ids
        or set(replays) - known_ids
        or set(current_evidence_bindings) - known_ids
    ):
        raise ValueError("autonomous_driving_refine_runtime_candidate_outside_catalog")

    selected: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    selected_source_rows: list[dict[str, Any]] = []
    for row in catalog_rows:
        candidate_id = str(row["candidate_id"])
        blockers: list[str] = []
        miner_row = miner_index.get(candidate_id)
        if miner_row is None:
            blockers.append("source_native_miner_candidate_missing")
        elif not _source_row_valid(miner_row) or _source_identity(miner_row) != _source_identity(
            row
        ):
            blockers.append("source_native_miner_identity_mismatch")
        calibration_entry = calibrations.get(candidate_id)
        replay_entry = replays.get(candidate_id)
        legs: dict[str, dict[str, Any]] | None = None
        costs: dict[str, float | None] = {}
        if calibration_entry is None:
            blockers.append("calibration_missing")
        else:
            calibration_name, calibration = calibration_entry
            calibration_blockers, legs, costs = _calibration_assessment(
                row,
                calibration_name,
                calibration,
                implementation_binding,
                current_evidence_bindings,
            )
            blockers.extend(calibration_blockers)
            if replay_entry is None:
                blockers.append("replay_missing")
            else:
                _replay_name, replay = replay_entry
                blockers.extend(
                    _replay_blockers(
                        row,
                        calibration_name,
                        legs,
                        replay,
                        implementation_binding,
                        current_evidence_bindings,
                        calibration,
                    )
                )
        headroom: float | None = None
        rule_headroom: float | None = None
        if costs and all(costs.get(name) is not None for name in REQUIRED_LEGS):
            shield = float(costs["shield_only"] or 0.0)
            headroom = round(shield - float(costs["oracle_offline"] or 0.0), 6)
            rule_headroom = round(shield - float(costs["rule_tactical"] or 0.0), 6)
            if headroom <= minimum_headroom_exclusive:
                blockers.append("positive_oracle_headroom_missing")
        if not blockers and _overlaps(row, selected_source_rows):
            blockers.append("overlaps_selected_source_window")
        summary = {
            **_summary_row(
                row,
                miner_evidence=miner_evidence.get(candidate_id, []),
                disposition="selected_positive_headroom" if not blockers else "held",
            ),
            "bundle_path": row.get("bundle_path"),
            "calibration_evidence": calibration_entry[0] if calibration_entry else None,
            "replay_evidence": replay_entry[0] if replay_entry else None,
            "oracle_headroom_vs_shield_only": headroom,
            "rule_headroom_vs_shield_only": rule_headroom,
            "headroom_threshold_exclusive": minimum_headroom_exclusive,
        }
        if blockers:
            held.append({**summary, "blockers": sorted(set(blockers))})
        else:
            selected.append(summary)
            selected_source_rows.append(row)

    selected_recordings = {str(row["recording_id"]) for row in selected}
    selected_hazards = {str(row["hazard_kind"]) for row in selected}
    deficits = {
        "positive_headroom_source_keys": max(0, target_count - len(selected)),
        "recording_count": max(0, minimum_recordings - len(selected_recordings)),
        "hazard_family_count": max(0, minimum_hazard_families - len(selected_hazards)),
    }
    if queue_limit is None:
        queue_limit = max(12, deficits["positive_headroom_source_keys"] * 3)
    if queue_limit < 1:
        raise ValueError("autonomous_driving_refine_queue_limit_invalid")
    pool = [row for candidate_id, row in miner_index.items() if candidate_id not in known_ids]
    pool.sort(
        key=lambda row: (
            -int(row.get("risk_score_milli") or 0),
            -int((row.get("hazard_context") or {}).get("supervisory_prevention_window_ms") or 0),
            str(row.get("candidate_id") or ""),
        )
    )
    queue: list[dict[str, Any]] = []
    pool_held: list[dict[str, Any]] = []
    accepted = list(selected_source_rows)
    for row in pool:
        candidate_id = str(row.get("candidate_id") or "")
        pool_blockers: list[str] = []
        if not _source_row_valid(row):
            pool_blockers.append("source_native_phase_window_invalid")
        elif _overlaps(row, selected_source_rows):
            pool_blockers.append("overlaps_selected_source_window")
        elif _overlaps(row, accepted):
            pool_blockers.append("overlaps_higher_priority_queue_window")
        elif len(queue) >= queue_limit:
            pool_blockers.append("queue_limit_reached")
        summary = {
            **_summary_row(
                row,
                miner_evidence=miner_evidence.get(candidate_id, []),
                disposition="held" if pool_blockers else "queued_for_native_probe",
            ),
            "risk_score_milli": int(row.get("risk_score_milli") or 0),
            "headroom_status": "unmeasured",
            "headroom_credit": 0,
        }
        if pool_blockers:
            pool_held.append({**summary, "blockers": pool_blockers})
        else:
            queue.append(summary)
            accepted.append(row)

    gates = {
        "positive_headroom_source_key_target": deficits["positive_headroom_source_keys"] == 0,
        "recording_diversity_target": deficits["recording_count"] == 0,
        "hazard_family_diversity_target": deficits["hazard_family_count"] == 0,
        "source_native_identity_binding": bool(selected)
        and all(row["miner_evidence"] for row in selected),
    }
    report: dict[str, Any] = {
        "schema_version": "autonomous_driving_headroom_refinement_v1",
        "status": "target_satisfied" if all(gates.values()) else "held",
        "formal_core_allowed": False,
        "registry_mutation_performed": False,
        "selection_contract": {
            "target_positive_headroom_source_keys": target_count,
            "minimum_recordings": minimum_recordings,
            "minimum_hazard_families": minimum_hazard_families,
            "oracle_headroom_threshold_exclusive": minimum_headroom_exclusive,
            "unmeasured_candidates_contribute_to_target": False,
            "overlap_scope": "within_recording_id",
            "runtime_evidence_binding": {
                "implementation_sha256": implementation_binding.get(
                    "autonomous_driving_slice_sha256"
                ),
                "semantics_sha256": implementation_binding.get("semantics_sha256"),
            },
        },
        "summary": {
            "catalog_candidates": len(catalog_rows),
            "selected_positive_headroom": len(selected),
            "held_catalog_candidates": len(held),
            "queued_unmeasured_candidates": len(queue),
            "held_miner_pool_candidates": len(pool_held),
            "selected_recordings": len(selected_recordings),
            "selected_hazard_families": len(selected_hazards),
        },
        "gates": gates,
        "deficits": deficits,
        "selected_positive_headroom": selected,
        "held": held,
        "next_candidate_queue": queue,
        "miner_pool_held": pool_held,
        "next_miner_round": {
            "required_additional_positive_headroom_keys": deficits["positive_headroom_source_keys"],
            "probe_queue_count": len(queue),
            "parameters": {
                "window_ms": 60_000,
                "stride_ms": 5_000,
                "minimum_supervisory_prevention_ms": 15_000,
                "minimum_recovery_ms": 20_000,
                "phase_window_complete": True,
                "prioritize_source_native_lead_braking_and_headway": True,
                "lane_change_requires_verified_lateral_control_before_core_credit": True,
                "recommended_candidates_to_probe": min(
                    len(queue), max(12, deficits["positive_headroom_source_keys"] * 3)
                ),
            },
        },
        "blockers": [
            code
            for code, missing in (
                ("positive_headroom_source_key_deficit", deficits["positive_headroom_source_keys"]),
                ("positive_headroom_recording_deficit", deficits["recording_count"]),
                ("positive_headroom_hazard_family_deficit", deficits["hazard_family_count"]),
            )
            if missing
        ],
    }
    report["report_digest_sha256"] = _digest(report, "report_digest_sha256")
    return report


def _index_runtime_reports(
    directory: Path, *, calibration: bool
) -> dict[str, tuple[str, dict[str, Any]]]:
    indexed: dict[str, tuple[str, dict[str, Any]]] = {}
    for path in sorted(directory.glob("*.json")):
        report = _load(path)
        if calibration:
            candidate_id = str(
                ((report.get("scenario") or {}).get("backend_config") or {}).get("candidate_id")
                or ""
            )
        else:
            candidate_id = str(report.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("autonomous_driving_refine_runtime_candidate_binding_missing")
        if candidate_id in indexed:
            raise ValueError("autonomous_driving_refine_runtime_candidate_duplicate")
        indexed[candidate_id] = (str(path.resolve()), report)
    return indexed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--miner-report", type=Path, action="append", required=True)
    parser.add_argument("--target-count", type=int, default=12)
    parser.add_argument("--minimum-recordings", type=int, default=3)
    parser.add_argument("--minimum-hazard-families", type=int, default=3)
    parser.add_argument("--minimum-headroom-exclusive", type=float, default=0.0)
    parser.add_argument("--queue-limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    catalog_path = args.catalog if args.catalog.is_absolute() else REPO_ROOT / args.catalog
    calibration_dir = (
        args.calibration_dir
        if args.calibration_dir.is_absolute()
        else REPO_ROOT / args.calibration_dir
    )
    replay_dir = args.replay_dir if args.replay_dir.is_absolute() else REPO_ROOT / args.replay_dir
    miner_paths = [path if path.is_absolute() else REPO_ROOT / path for path in args.miner_report]
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    try:
        from scripts.build_autonomous_external_candidate_terminal_ledger import (
            _implementation_binding,
        )

        catalog = _load(catalog_path)
        calibrations = _index_runtime_reports(calibration_dir, calibration=True)
        replays = _index_runtime_reports(replay_dir, calibration=False)
        from domains.autonomous_driving.evidence_binding import (
            calibration_evidence_binding,
        )

        current_evidence_bindings: dict[str, dict[str, str]] = {}
        for row in _index_catalog(catalog):
            candidate_id = str(row["candidate_id"])
            calibration_entry = calibrations.get(candidate_id)
            if calibration_entry is None:
                continue
            bundle = _resolved(row.get("bundle_path"))
            try:
                current_evidence_bindings[candidate_id] = calibration_evidence_binding(
                    repo_root=REPO_ROOT,
                    bundle=bundle,
                    candidate_id=candidate_id,
                    legs=[
                        dict(value)
                        for value in calibration_entry[1].get("legs") or []
                        if isinstance(value, dict)
                    ],
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        report = build_refinement_report(
            catalog=catalog,
            calibrations=calibrations,
            replays=replays,
            miner_reports=[(str(path.resolve()), _load(path)) for path in miner_paths],
            implementation_binding=_implementation_binding(),
            current_evidence_bindings=current_evidence_bindings,
            target_count=args.target_count,
            minimum_recordings=args.minimum_recordings,
            minimum_hazard_families=args.minimum_hazard_families,
            minimum_headroom_exclusive=args.minimum_headroom_exclusive,
            queue_limit=args.queue_limit,
        )
        if args.check:
            if not output.is_file() or _load(output) != report:
                raise ValueError("autonomous_driving_refine_report_stale_or_missing")
        else:
            if output.exists():
                raise FileExistsError("autonomous_driving_refine_output_exists")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"status": report["status"], **report["summary"]}, sort_keys=True))
    return 0 if report["status"] == "target_satisfied" else 2


if __name__ == "__main__":
    raise SystemExit(main())
