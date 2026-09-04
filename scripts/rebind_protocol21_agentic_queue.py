#!/usr/bin/env python3
"""Rebind a legacy Protocol-2.1 agentic queue to a current source suite.

This is an identity audit, not a remediation runner.  It copies every legacy
queue item, preserves its original ``disposition``, and records whether its
scenario identity is present in the supplied current source suite.  No
replay, evidence, score, or Core-admission claim is made by this utility.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON input: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return payload


def _legacy_items(queue: dict[str, Any]) -> list[dict[str, Any]]:
    items = queue.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError("legacy queue must contain items: list[object]")
    return items


def _source_rows(source_suite: dict[str, Any]) -> list[dict[str, Any]]:
    rows = source_suite.get("scenarios")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("current source suite must contain scenarios: non-empty list[object]")
    identities = [
        (str(row.get("scenario_id") or ""), str(row.get("scenario_signature") or ""))
        for row in rows
    ]
    if any(not scenario_id or not signature for scenario_id, signature in identities):
        raise ValueError("current source suite identities must contain scenario_id and scenario_signature")
    if len(set(identities)) != len(identities):
        raise ValueError("current source suite identities must be unique")
    return rows


def source_suite_identity_fingerprint(rows: list[dict[str, Any]]) -> str:
    """Hash sorted current-suite identities, independent of JSON formatting/order."""

    identities = [
        {
            "scenario_id": str(row.get("scenario_id") or ""),
            "scenario_signature": str(row.get("scenario_signature") or ""),
        }
        for row in rows
    ]
    identities.sort(
        key=lambda identity: (identity["scenario_id"], identity["scenario_signature"])
    )
    return hashlib.sha256(_canonical_json(identities).encode("utf-8")).hexdigest()


def _identity_overlap(
    items: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    current_ids = {str(row.get("scenario_id") or "") for row in rows}
    current_signatures = {str(row.get("scenario_signature") or "") for row in rows}
    id_matches = sorted(
        {
            str(item.get("scenario_id") or "")
            for item in items
            if str(item.get("scenario_id") or "") in current_ids
        }
    )
    signature_matches = sorted(
        {
            str(item.get("scenario_signature") or "")
            for item in items
            if str(item.get("scenario_signature") or "") in current_signatures
        }
    )
    row_indexes = [
        index
        for index, item in enumerate(items)
        if str(item.get("scenario_id") or "") in current_ids
        or str(item.get("scenario_signature") or "") in current_signatures
    ]
    return {
        "scenario_overlap": len(row_indexes),
        "scenario_overlap_is_zero": not row_indexes,
        "scenario_id_overlap": {"n": len(id_matches), "values": id_matches},
        "scenario_signature_overlap": {
            "n": len(signature_matches),
            "values": signature_matches,
        },
        "legacy_row_indexes_with_any_identity_overlap": row_indexes,
    }


def build_rebind_report(
    legacy_queue: dict[str, Any],
    current_source_suite: dict[str, Any],
    *,
    legacy_queue_path: str | None = None,
    current_source_suite_path: str | None = None,
    current_source_suite_sha256: str | None = None,
    expected_legacy_items: int | None = None,
) -> dict[str, Any]:
    """Build a fail-closed identity report without changing either input."""

    items = _legacy_items(legacy_queue)
    rows = _source_rows(current_source_suite)
    if expected_legacy_items is not None and len(items) != expected_legacy_items:
        raise ValueError(
            "legacy queue item count mismatch: "
            f"expected {expected_legacy_items}, observed {len(items)}"
        )

    overlap = _identity_overlap(items, rows)
    source_sha256 = current_source_suite_sha256
    if source_sha256 is None:
        source_sha256 = str(current_source_suite.get("source_suite_sha256") or "") or None
    source_fingerprint = source_suite_identity_fingerprint(rows)

    disposition_values = [item.get("disposition") for item in items]
    disposition_counts = Counter(str(value) for value in disposition_values)
    id_set = {str(row.get("scenario_id") or "") for row in rows}
    signature_set = {str(row.get("scenario_signature") or "") for row in rows}

    rebound_items: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for original in items:
        item = copy.deepcopy(original)
        scenario_id = str(original.get("scenario_id") or "")
        scenario_signature = str(original.get("scenario_signature") or "")
        id_match = scenario_id in id_set
        signature_match = scenario_signature in signature_set
        reasons = [
            "legacy_disposition_preserved",
            "current_v5_rebind_not_proven",
            "current_v5_pipeline_evidence_not_attached",
            "core_admission_forbidden",
        ]
        if not id_match:
            reasons.append("scenario_id_not_in_current_source_suite")
        if not signature_match:
            reasons.append("scenario_signature_not_in_current_source_suite")
        if id_match or signature_match:
            reasons.append("identity_overlap_requires_full_artifact_binding")
        reasons = sorted(set(reasons))
        reason_counts.update(reasons)
        item["current_v5_rebind"] = {
            "status": "held",
            "rebound": False,
            "legacy_disposition": original.get("disposition"),
            "scenario_id_match": id_match,
            "scenario_signature_match": signature_match,
            "reason_codes": reasons,
        }
        rebound_items.append(item)

    fail_closed_reasons = [
        "legacy_dispositions_preserved_without_remediation_claim",
        "current_v5_pipeline_evidence_not_attached",
        "native_replay_and_all_protocol21_gates_not_reproven",
        "core_admission_forbidden",
    ]
    if current_source_suite.get("status") != "working_set":
        fail_closed_reasons.append("current_source_suite_not_working_set")
    if current_source_suite.get("leaderboard_eligible") is not False:
        fail_closed_reasons.append("current_source_suite_leaderboard_eligibility_unproven")
    if overlap["scenario_overlap"] == 0:
        fail_closed_reasons.append("scenario_overlap_zero")
    fail_closed_reasons = sorted(set(fail_closed_reasons))
    return {
        "schema_version": "protocol21-agentic-queue-rebind-v1",
        "status": "diagnostic_only_rebind_pending",
        "rebind_claimed": False,
        "direct_core_admission": False,
        "current_source_suite_sha256": source_sha256,
        "current_source_suite_fingerprint": source_fingerprint,
        "scenario_overlap": overlap["scenario_overlap"],
        "legacy_queue": {
            "path": legacy_queue_path,
            "schema_version": legacy_queue.get("schema_version"),
            "n_items": len(items),
            "disposition_counts": dict(sorted(disposition_counts.items())),
        },
        "current_source_suite": {
            "path": current_source_suite_path,
            "sha256": source_sha256,
            "identity_fingerprint": source_fingerprint,
            "status": current_source_suite.get("status"),
            "leaderboard_eligible": current_source_suite.get("leaderboard_eligible"),
            "release_ready": current_source_suite.get("release_ready"),
            "n_scenarios": len(rows),
        },
        "overlap": overlap,
        "disposition_preservation": {
            "identical": all(
                original.get("disposition") == rebound.get("disposition")
                for original, rebound in zip(items, rebound_items, strict=True)
            ),
            "n_preserved": len(items),
            "counts": dict(sorted(disposition_counts.items())),
        },
        "fail_closed_reasons": fail_closed_reasons,
        "reason_counts": dict(sorted(reason_counts.items())),
        "items": rebound_items,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-queue", type=Path, required=True)
    parser.add_argument("--current-source-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-legacy-items", type=int, default=109)
    args = parser.parse_args(argv)

    legacy_queue = _load_json_object(args.legacy_queue)
    current_source_suite = _load_json_object(args.current_source_suite)
    report = build_rebind_report(
        legacy_queue,
        current_source_suite,
        legacy_queue_path=str(args.legacy_queue),
        current_source_suite_path=str(args.current_source_suite),
        current_source_suite_sha256=_sha256(args.current_source_suite),
        expected_legacy_items=args.expected_legacy_items,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_items": report["legacy_queue"]["n_items"],
                "scenario_overlap": report["overlap"]["scenario_overlap"],
                "rebind_claimed": report["rebind_claimed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
