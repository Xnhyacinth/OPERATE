#!/usr/bin/env python3
"""Restore and audit source-bound Protocol-2.1 working-set lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.protocol21_evidence import report_rows  # noqa: E402
from core.working_set_contract import (  # noqa: E402
    extract_protocol21_selection_constraints,
    preserve_protocol21_row_lineage,
    validate_protocol21_working_set_contract,
)


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_lineage_preview(
    *,
    source_suite: dict[str, Any],
    working_set: dict[str, Any],
    migration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_rows = report_rows(source_suite)
    working_rows = report_rows(working_set)
    source_by_identity: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in source_rows:
        identity = (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        source_by_identity.setdefault(identity, []).append(row)

    migration_by_new: dict[
        tuple[str, str], list[tuple[str, str]]
    ] = {}
    for item in migration.get("migration_map") or []:
        if not isinstance(item, dict):
            continue
        new_identity = (
            str(item.get("new_scenario_id") or ""),
            str(item.get("new_scenario_signature") or ""),
        )
        old_identity = (
            str(item.get("old_scenario_id") or ""),
            str(item.get("old_scenario_signature") or ""),
        )
        migration_by_new.setdefault(new_identity, []).append(old_identity)

    preview_rows: list[dict[str, Any]] = []
    binding_failures: list[dict[str, Any]] = []
    for row in working_rows:
        new_identity = (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        old_identities = migration_by_new.get(new_identity, [])
        source_matches = [
            source_row
            for old_identity in old_identities
            for source_row in source_by_identity.get(old_identity, [])
        ]
        if len(old_identities) != 1 or len(source_matches) != 1:
            binding_failures.append(
                {
                    "scenario_id": new_identity[0],
                    "scenario_signature": new_identity[1],
                    "n_migration_matches": len(old_identities),
                    "n_source_matches": len(source_matches),
                    "reason_code": "working_set_source_binding_not_unique",
                }
            )
            preview_rows.append(deepcopy(row))
            continue
        preview_rows.append(
            preserve_protocol21_row_lineage(source_matches[0], row)
        )

    constraint_key, source_constraints = (
        extract_protocol21_selection_constraints(source_suite)
    )
    contract = validate_protocol21_working_set_contract(
        preview_rows,
        constraints=source_constraints,
    )
    reason_codes = set(contract["reason_codes"])
    if binding_failures:
        reason_codes.add("working_set_source_binding_not_unique")
        reason_codes.add("working_set_source_identity_metadata_missing")
    blockers = set(working_set.get("blockers") or [])
    blockers.update(reason_codes)
    if contract["formal_lineage_ready"] and not binding_failures:
        blockers.discard("working_set_source_identity_metadata_missing")

    preview = deepcopy(working_set)
    preview["leaderboard_eligible"] = False
    preview["scenarios"] = preview_rows
    preview["n_scenarios"] = len(preview_rows)
    preview["blockers"] = sorted(blockers)
    if constraint_key is not None:
        preview.pop(
            "selection_constraints"
            if constraint_key == "constraints"
            else "constraints",
            None,
        )
        preview[constraint_key] = deepcopy(source_constraints)
    preview["lineage_contract"] = {
        "status": (
            "ready"
            if contract["formal_lineage_ready"] and not binding_failures
            else "blocked"
        ),
        "source_identity_metadata_complete": contract[
            "source_identity_metadata_complete"
        ],
        "formal_lineage_ready": (
            contract["formal_lineage_ready"] and not binding_failures
        ),
        "reason_codes": sorted(reason_codes),
    }
    source_digest = _digest(source_constraints)
    preview_key, preview_constraints = (
        extract_protocol21_selection_constraints(preview)
    )
    preview_digest = _digest(preview_constraints)
    constraint_preservation = {
        "source_key": constraint_key,
        "preview_key": preview_key,
        "source_constraints": source_constraints,
        "preview_constraints": preview_constraints,
        "source_canonical_sha256": source_digest,
        "preview_canonical_sha256": preview_digest,
        "matches": bool(
            constraint_key
            and constraint_key == preview_key
            and source_digest == preview_digest
        ),
        "working_set_constraints_were_missing": not bool(
            extract_protocol21_selection_constraints(working_set)[1]
        ),
    }
    report = {
        "schema_version": "2.1",
        "status": preview["lineage_contract"]["status"],
        "n_source_rows": len(source_rows),
        "n_rows_bound_to_source": len(preview_rows) - len(binding_failures),
        "binding_failures": binding_failures,
        **contract,
        "constraint_preservation": constraint_preservation,
    }
    report["formal_lineage_ready"] = preview["lineage_contract"][
        "formal_lineage_ready"
    ]
    return report, preview


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--working-set", type=Path, required=True)
    parser.add_argument("--migration", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-preview", type=Path, required=True)
    args = parser.parse_args()

    report, preview = build_lineage_preview(
        source_suite=json.loads(
            args.source_suite.read_text(encoding="utf-8")
        ),
        working_set=json.loads(
            args.working_set.read_text(encoding="utf-8")
        ),
        migration=json.loads(args.migration.read_text(encoding="utf-8")),
    )
    for path, payload in (
        (args.output_report, report),
        (args.output_preview, preview),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
