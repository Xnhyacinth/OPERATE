#!/usr/bin/env python3
"""Classify external transfer rows before native Protocol-2.1 admission.

This command is deliberately conservative.  It never materializes a scenario
and never marks a row as Core-ready.  A complete row is only eligible to enter
the normal candidate pipeline, where native execution, scoring, depth, and
replay gates still run.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_TRANSFER_MODES = {"native_method_transfer", "external_raw_asset"}


def _external_source_url(row: dict[str, Any]) -> str | None:
    """Return the canonical external reference when the row carries one."""

    value = row.get("external_source_url", row.get("source_url"))
    if value is None:
        return None
    return str(value).strip()


def _external_transfer_reasons(row: dict[str, Any]) -> set[str]:
    """Fail closed on explicit external transfer metadata.

    Older hand-authored unit fixtures do not carry transfer metadata and remain
    compatible.  Generated external queues do carry it, so a raw-source row
    can never become ``candidate_pending_full_pipeline`` merely because its
    native-looking fields were filled in.
    """

    reasons: set[str] = set()
    raw_mode = row.get("transfer_mode")
    mode = raw_mode if isinstance(raw_mode, str) else None
    has_external_metadata = "external_source_id" in row or raw_mode is not None
    if not has_external_metadata:
        return reasons
    if raw_mode is not None and mode not in _TRANSFER_MODES:
        reasons.add("unsupported_transfer_mode")
    if mode == "external_raw_asset":
        reasons.add("external_raw_not_admissible")
    url = _external_source_url(row)
    if not url:
        reasons.add("external_source_url_missing")
    else:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            reasons.add("external_source_url_invalid")
    return reasons


def _reason_codes(row: dict[str, Any]) -> list[str]:
    reasons = {str(reason) for reason in row.get("reason_codes") or []}
    reasons.update(_external_transfer_reasons(row))
    promotion = row.get("promotion") or {}
    if not row.get("source_assets_exist"):
        reasons.add("external_source_unresolved")
    physical = str(row.get("physical_source_key") or "")
    if not physical or "unresolved" in physical:
        reasons.add("independent_physical_source_unresolved")
    if not (row.get("native_tools") or []) or row.get("native_tools") == [
        "backend_native_tools_only"
    ]:
        reasons.add("native_tools_unresolved")
    if not (row.get("difficulty_contract") or {}).get("observed"):
        reasons.add("difficulty_depth_unproven")
    if not (row.get("event_contract") or {}).get("response_window_proven"):
        reasons.add("response_window_unproven")
    if promotion.get("direct_core_admission") is not False:
        reasons.add("direct_core_admission_forbidden")
    if promotion.get("requires_all_gates") is not True:
        reasons.add("all_protocol21_gates_not_required")
    return sorted(reasons)


def classify_queue(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify rows and retain every decision/reason in machine-readable form."""
    result_rows: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    for original in rows:
        row = copy.deepcopy(original)
        reasons = _reason_codes(row)
        source = str(row.get("source_denominator_key") or row.get("physical_source_key") or "")
        if source and source in seen_sources:
            reasons.append("duplicate_effective_source")
        if source:
            seen_sources.add(source)
        reasons = sorted(set(reasons))
        row["reason_codes"] = reasons
        row["promotion"] = {
            **(row.get("promotion") or {}),
            "direct_core_admission": False,
            "requires_all_gates": True,
            "requires_native_replay": True,
        }
        row["status"] = "candidate_pending_full_pipeline" if not reasons else "held"
        raw_mode = row.get("transfer_mode")
        mode = raw_mode if isinstance(raw_mode, str) else None
        row["transfer_audit"] = {
            "mode": mode if mode in _TRANSFER_MODES else "unspecified",
            "external_source_url": _external_source_url(row),
            "external_source_url_valid": not any(
                reason in reasons
                for reason in {
                    "external_source_url_missing",
                    "external_source_url_invalid",
                }
            )
            if ("external_source_id" in row or mode is not None)
            else None,
            "raw_source_claim": mode == "external_raw_asset",
            "direct_core_admission_blocked": True,
        }
        result_rows.append(row)
    transfer_modes = Counter(
        str(row.get("transfer_mode") or "unspecified") for row in result_rows
    )
    return {
        "schema_version": "protocol21_external_conversion_v1",
        "direct_core_admission": False,
        "promotion_policy": "complete_candidate_enters_full_protocol21_pipeline",
        "n_candidates": len(result_rows),
        "n_ready": sum(row["status"] == "candidate_pending_full_pipeline" for row in result_rows),
        "n_held": sum(row["status"] == "held" for row in result_rows),
        "transfer_mode_counts": dict(sorted(transfer_modes.items())),
        "n_external_raw_held": sum(
            row.get("transfer_mode") == "external_raw_asset"
            and row["status"] == "held"
            for row in result_rows
        ),
        "reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in result_rows
                    for reason in row.get("reason_codes") or []
                ).items()
            )
        ),
        "candidates": result_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.queue.read_text(encoding="utf-8"))
    rows = payload.get("candidates")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise SystemExit("queue must contain candidates: list[object]")
    result = classify_queue(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("n_candidates", "n_ready", "n_held")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
