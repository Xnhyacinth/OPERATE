#!/usr/bin/env python3
"""Machine-readable release diff and reason-code helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

REASON_CODE_ALLOWLIST = {
    "traffic_sumo365_live_headroom_passed",
    "traffic_sumo365_source_lock_failed",
    "traffic_sumo365_mock_filter_failed",
    "traffic_sumo365_live_headroom_missing",
    "traffic_sumo365_decision_fingerprint_duplicate",
    "traffic_sumo365_not_selected_for_core",
    "retained_from_v0_32",
    "retained_from_v0_33_rc1",
    "retained_from_v0_33_rc2",
    "retained_from_v0_33_rc3",
    "retained_from_v0_33_rc4",
    "retained_from_v0_33_rc5",
    "retained_from_v0_33_rc6",
    "retained_from_v0_33_rc7",
    "retained_from_v0_33_rc8",
    "retained_from_v0_33_rc9",
    "retained_from_v0_33_rc10",
    "retained_from_v0_33_rc11",
    "retained_from_v0_33_rc12",
    "retained_from_v0_33_rc13",
    "retained_from_v0_33_rc14",
    "logistics_job_shop_shape_decision_overdense",
    "logistics_routing_exact_headroom_blocked",
    "retain_domain_coverage_anchor",
    "diagnostic_cell_excluded",
    "same_effective_source_key_secondary_variant",
    "demote_same_physical_source_secondary_variant",
    "demote_low_headroom_or_acknowledged_diagnostic",
    "demote_low_score_headroom",
    "demote_near_duplicate_structural_fingerprint",
    "demote_quantity_only_variant",
    "audit_missing_structural_fingerprint",
    "audit_missing_domain",
    "audit_missing_reason_code",
}


def _rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("scenario_id")): row for row in rows}


def _ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("scenario_id")) for row in rows}


def _ledger(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("case_ledger") or {}


def _source_key(row: dict[str, Any]) -> str:
    return str(
        row.get("source_denominator_key")
        or _ledger(row).get("source_denominator_key")
        or row.get("source_key")
        or ""
    )


def _first_nonempty_source_key(*rows: dict[str, Any] | None) -> str:
    for row in rows:
        if not row:
            continue
        key = _source_key(row)
        if key:
            return key
    return ""


def _first_nonempty_decision_fingerprint(*rows: dict[str, Any] | None) -> str:
    for row in rows:
        if not row:
            continue
        fingerprint = _decision_fingerprint(row)
        if fingerprint:
            return fingerprint
    return ""


def _decision_fingerprint(row: dict[str, Any]) -> str:
    return str(
        _ledger(row).get("decision_fingerprint")
        or row.get("decision_fingerprint")
        or ""
    )


def _membership(
    suites: dict[str, list[dict[str, Any]]], scenario_id: str
) -> dict[str, bool]:
    return {
        "registry": scenario_id in _ids(suites.get("registry", [])),
        "primary": scenario_id in _ids(suites.get("primary", [])),
        "core": scenario_id in _ids(suites.get("core", [])),
    }


def _change_type(before: dict[str, bool], after: dict[str, bool]) -> str:
    if before["registry"] and not after["registry"]:
        return "removed"
    if not before["registry"] and after["registry"]:
        return "added"
    if before["primary"] and not after["primary"]:
        return "downgraded"
    if before["core"] and not after["core"]:
        return "core_demoted"
    if not before["core"] and after["core"]:
        return "core_selected"
    if before == after:
        return "retained"
    return "membership_changed"


def build_release_diff(
    *,
    source_release_id: str,
    target_release_id: str,
    source_suites: dict[str, list[dict[str, Any]]],
    target_suites: dict[str, list[dict[str, Any]]],
    reason_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_registry = _rows_by_id(source_suites.get("registry", []))
    target_registry = _rows_by_id(target_suites.get("registry", []))
    source_primary = _rows_by_id(source_suites.get("primary", []))
    target_primary = _rows_by_id(target_suites.get("primary", []))
    source_core = _rows_by_id(source_suites.get("core", []))
    target_core = _rows_by_id(target_suites.get("core", []))
    all_ids = sorted(
        set(source_registry)
        | set(target_registry)
        | set(source_primary)
        | set(target_primary)
        | set(source_core)
        | set(target_core)
    )

    rows: list[dict[str, Any]] = []
    for scenario_id in all_ids:
        row = (
            target_registry.get(scenario_id)
            or target_primary.get(scenario_id)
            or target_core.get(scenario_id)
            or source_registry.get(scenario_id)
            or source_primary.get(scenario_id)
            or source_core.get(scenario_id)
            or {}
        )
        source_denominator_key = _first_nonempty_source_key(
            target_registry.get(scenario_id),
            target_primary.get(scenario_id),
            target_core.get(scenario_id),
            source_registry.get(scenario_id),
            source_primary.get(scenario_id),
            source_core.get(scenario_id),
        )
        decision_fingerprint = _first_nonempty_decision_fingerprint(
            target_registry.get(scenario_id),
            target_primary.get(scenario_id),
            target_core.get(scenario_id),
            source_registry.get(scenario_id),
            source_primary.get(scenario_id),
            source_core.get(scenario_id),
        )
        before = _membership(source_suites, scenario_id)
        after = _membership(target_suites, scenario_id)
        reason = reason_records.get(scenario_id) or {}
        rows.append(
            {
                "scenario_id": scenario_id,
                "domain": row.get("domain"),
                "family": row.get("family"),
                "backend_kind": row.get("backend_kind"),
                "difficulty_mode": row.get("difficulty_mode"),
                "difficulty_level": row.get("difficulty_level"),
                "source_denominator_key": source_denominator_key,
                "structural_fingerprint": row.get("structural_fingerprint"),
                "decision_fingerprint": decision_fingerprint,
                "membership_before": before,
                "membership_after": after,
                "change_type": _change_type(before, after),
                "reason_code": reason.get("reason_code"),
                "reason_detail": reason.get("reason_detail"),
                "evidence_refs": list(reason.get("evidence_refs") or []),
                "source_lock_refs": list(reason.get("source_lock_refs") or []),
            }
        )

    counts = Counter(row["change_type"] for row in rows)
    return {
        "schema_version": "0.1",
        "source_release_id": source_release_id,
        "target_release_id": target_release_id,
        "rows": rows,
        "summary": {f"n_{key}": counts.get(key, 0) for key in sorted(counts)},
        "validation_issues": [],
    }


def validate_release_diff(diff: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for row in diff.get("rows") or []:
        reason_code = row.get("reason_code")
        change_type = row.get("change_type")
        scenario_id = row.get("scenario_id")
        if change_type != "retained" and not reason_code:
            issues.append({"code": "missing_reason_code", "scenario_id": scenario_id})
        if reason_code and reason_code not in REASON_CODE_ALLOWLIST:
            issues.append(
                {
                    "code": "unknown_reason_code",
                    "scenario_id": scenario_id,
                    "reason_code": reason_code,
                }
            )
        if not row.get("domain"):
            issues.append({"code": "missing_domain", "scenario_id": scenario_id})
        if (row.get("membership_after") or {}).get("primary") and not row.get(
            "structural_fingerprint"
        ):
            issues.append(
                {
                    "code": "missing_structural_fingerprint",
                    "scenario_id": scenario_id,
                }
            )
    return issues
