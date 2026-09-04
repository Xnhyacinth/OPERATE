"""Fail-closed lineage and selection contracts for Protocol-2.1 working sets."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from typing import Any

_REQUIRED_LINEAGE_FIELDS = (
    "source_key",
    "case_ledger",
    "structural_fingerprint",
    "semantic_fingerprint",
)


def _present(value: Any) -> bool:
    return value not in (None, "", {})


def _decision_axis_present(row: dict[str, Any]) -> bool:
    ledger = row.get("case_ledger")
    ledger = ledger if isinstance(ledger, dict) else {}
    return any(
        _present(value)
        for value in (
            row.get("decision_axis"),
            ledger.get("decision_axis"),
            ledger.get("decision_pressure_axis"),
            ledger.get("decision_variant_key"),
            ledger.get("additional_decision_axis"),
        )
    ) or (
        _present(row.get("family"))
        and _present(row.get("difficulty_mode"))
    )


def extract_protocol21_selection_constraints(
    artifact: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """Return the source artifact's constraint key and a defensive copy."""
    for key in ("constraints", "selection_constraints"):
        value = artifact.get(key)
        if isinstance(value, dict) and value:
            return key, deepcopy(value)
    return None, {}


def preserve_protocol21_row_lineage(
    source: dict[str, Any],
    output: dict[str, Any],
) -> dict[str, Any]:
    """Copy only lineage explicitly present in the bound source row."""
    preserved = deepcopy(output)
    for field in _REQUIRED_LINEAGE_FIELDS:
        if field in source:
            preserved[field] = deepcopy(source[field])
    ledger = source.get("case_ledger")
    if isinstance(ledger, dict) and _present(
        ledger.get("source_denominator_key")
    ):
        preserved["source_denominator_key"] = deepcopy(
            ledger["source_denominator_key"]
        )
    if _present(source.get("physical_source_key")):
        preserved["physical_source_key"] = deepcopy(
            source["physical_source_key"]
        )
    return preserved


def validate_protocol21_row_lineage(row: dict[str, Any]) -> list[str]:
    """Return machine-readable blockers for one candidate row."""
    blockers = [
        f"working_set_{field}_missing"
        for field in _REQUIRED_LINEAGE_FIELDS
        if not _present(row.get(field))
    ]
    if not _present(row.get("source_denominator_key")):
        blockers.append("working_set_source_denominator_key_missing")
    if not _present(row.get("scenario_signature")):
        blockers.append("working_set_scenario_signature_missing")
    if not _decision_axis_present(row):
        blockers.append("working_set_decision_axis_missing")
    ledger = row.get("case_ledger")
    nested_physical_key = (
        ledger.get("physical_source_key")
        if isinstance(ledger, dict)
        else None
    )
    physical_lock = (
        ledger.get("physical_source_lock")
        if isinstance(ledger, dict)
        else None
    )
    if not any(
        _present(value)
        for value in (
            row.get("physical_source_key"),
            nested_physical_key,
            physical_lock,
        )
    ):
        blockers.append("working_set_physical_source_identity_missing")
    return sorted(blockers)


def summarize_effective_source_groups(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[str]] = {}
    missing: list[dict[str, Any]] = []
    for row in rows:
        key = row.get("source_denominator_key")
        if not _present(key):
            missing.append(
                {
                    "scenario_id": str(row.get("scenario_id") or ""),
                    "backend_kind": str(row.get("backend_kind") or ""),
                    "reason_code": (
                        "working_set_source_denominator_key_missing"
                    ),
                }
            )
            continue
        grouped.setdefault(str(key), []).append(
            str(row.get("scenario_id") or "")
        )
    groups = [
        {
            "source_denominator_key": key,
            "n_rows": len(scenario_ids),
            "scenario_ids": sorted(scenario_ids),
        }
        for key, scenario_ids in sorted(grouped.items())
    ]
    return groups, sorted(missing, key=lambda item: item["scenario_id"])


def summarize_physical_source_lock_groups(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for row in rows:
        ledger = row.get("case_ledger")
        ledger = ledger if isinstance(ledger, dict) else {}
        explicit_key = row.get("physical_source_key")
        nested_key = ledger.get("physical_source_key")
        lock = ledger.get("physical_source_lock")
        identity = next(
            (
                value
                for value in (explicit_key, nested_key, lock)
                if _present(value)
            ),
            None,
        )
        if identity is None:
            missing.append(
                {
                    "scenario_id": str(row.get("scenario_id") or ""),
                    "domain": str(row.get("domain") or ""),
                    "backend_kind": str(row.get("backend_kind") or ""),
                    "reason_code": (
                        "working_set_physical_source_identity_missing"
                    ),
                }
            )
            continue
        canonical = json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        group = grouped.setdefault(
            canonical,
            {
                "physical_source_lock": deepcopy(identity),
                "scenario_ids": [],
            },
        )
        group["scenario_ids"].append(str(row.get("scenario_id") or ""))
    groups: list[dict[str, Any]] = []
    for canonical in sorted(grouped):
        group = grouped[canonical]
        scenario_ids = sorted(group["scenario_ids"])
        groups.append(
            {
                "physical_source_lock": group["physical_source_lock"],
                "n_rows": len(scenario_ids),
                "scenario_ids": scenario_ids,
            }
        )
    return groups, sorted(missing, key=lambda item: item["scenario_id"])


def validate_protocol21_working_set_contract(
    rows: list[dict[str, Any]],
    *,
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """Summarize explicit lineage without inventing fallback identities."""
    effective_groups, missing_effective = summarize_effective_source_groups(
        rows
    )
    physical_groups, missing_physical = (
        summarize_physical_source_lock_groups(rows)
    )

    counts: dict[str, int] = {}
    for field in _REQUIRED_LINEAGE_FIELDS:
        present = sum(_present(row.get(field)) for row in rows)
        counts[f"n_rows_with_{field}"] = present
        counts[f"n_rows_missing_{field}"] = len(rows) - present
    denominator_count = sum(
        _present(row.get("source_denominator_key")) for row in rows
    )
    signature_count = sum(
        _present(row.get("scenario_signature")) for row in rows
    )
    decision_axis_count = sum(_decision_axis_present(row) for row in rows)
    physical_key_count = sum(
        _present(row.get("physical_source_key")) for row in rows
    )
    physical_lock_count = sum(
        isinstance(row.get("case_ledger"), dict)
        and _present(row["case_ledger"].get("physical_source_lock"))
        for row in rows
    )
    row_blockers = {
        str(row.get("scenario_id") or ""): validate_protocol21_row_lineage(row)
        for row in rows
    }
    row_blockers = {
        scenario_id: blockers
        for scenario_id, blockers in row_blockers.items()
        if blockers
    }
    metadata_complete = not row_blockers
    constraints_present = bool(constraints)
    reason_codes: set[str] = {
        reason for reasons in row_blockers.values() for reason in reasons
    }
    if not constraints_present:
        reason_codes.add("working_set_selection_constraints_missing")
    if reason_codes:
        reason_codes.add("working_set_source_identity_metadata_missing")

    return {
        "n_rows": len(rows),
        **counts,
        "n_rows_with_source_denominator_key": denominator_count,
        "n_rows_missing_source_denominator_key": (
            len(rows) - denominator_count
        ),
        "n_rows_with_scenario_signature": signature_count,
        "n_rows_missing_scenario_signature": len(rows) - signature_count,
        "n_rows_with_decision_axis": decision_axis_count,
        "n_rows_missing_decision_axis": len(rows) - decision_axis_count,
        "n_rows_with_physical_source_key": physical_key_count,
        "n_rows_with_physical_source_lock": physical_lock_count,
        "n_rows_missing_physical_source_identity": len(missing_physical),
        "n_rows_grouped_by_effective_source": sum(
            group["n_rows"] for group in effective_groups
        ),
        "n_rows_grouped_by_physical_source_lock": sum(
            group["n_rows"] for group in physical_groups
        ),
        "effective_source_group_count": len(effective_groups),
        "physical_source_lock_group_count": len(physical_groups),
        "effective_source_groups": effective_groups,
        "missing_effective_source_identity": missing_effective,
        "physical_source_lock_groups": physical_groups,
        "missing_physical_source_identity": missing_physical,
        "missing_source_denominator_by_backend": dict(
            sorted(
                Counter(
                    item["backend_kind"] for item in missing_effective
                ).items()
            )
        ),
        "missing_physical_source_identity_by_domain": dict(
            sorted(
                Counter(item["domain"] for item in missing_physical).items()
            )
        ),
        "row_blockers": row_blockers,
        "reason_codes": sorted(reason_codes),
        "constraints_present": constraints_present,
        "source_identity_metadata_complete": metadata_complete,
        "formal_lineage_ready": metadata_complete and constraints_present,
    }
