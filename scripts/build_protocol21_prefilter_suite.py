#!/usr/bin/env python3
"""Build a non-admitting Protocol-2.1 suite from prefilter shortlist reports.

This helper only selects rows that already passed a bounded native prefilter.
It preserves the source-suite row bytes, requires exact scenario identities, and
marks the result as a fresh-replay working set.  It never marks a row Core and
never changes a frozen release artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

PREFILTER_DISPOSITION = "preflight_candidate"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("scenario_id") or ""), str(row.get("scenario_signature") or "")


def _prefilter_rows(
    report: dict[str, Any], report_path: Path
) -> list[dict[str, Any]]:
    rows = report.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"prefilter report rows missing: {report_path}")
    selected = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("disposition") == PREFILTER_DISPOSITION
    ]
    identities = [_identity(row) for row in selected]
    if any(not scenario_id or not signature for scenario_id, signature in identities):
        raise ValueError(f"prefilter identity incomplete: {report_path}")
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate prefilter identity: {report_path}")
    return selected


def _validate_report_binding(
    report: dict[str, Any],
    report_path: Path,
    *,
    source_suite_sha256: str,
) -> None:
    if report.get("source_suite_sha256") != source_suite_sha256:
        raise ValueError(f"prefilter source suite hash mismatch: {report_path}")
    implementation = report.get("implementation_tree_sha256")
    if not isinstance(implementation, str) or not implementation:
        raise ValueError(f"prefilter implementation binding missing: {report_path}")
    runtime_version = report.get("runtime_version")
    if runtime_version in (None, "", {}):
        raise ValueError(f"prefilter runtime version missing: {report_path}")


def build_prefilter_suite(
    source_suite_path: Path,
    report_paths: list[Path],
    *,
    allow_unmatched: bool = False,
) -> dict[str, Any]:
    source = _load(source_suite_path)
    source_suite_sha256 = _sha256(source_suite_path)
    if source.get("status") != "working_set" or source.get("leaderboard_eligible") is not False:
        raise ValueError("source suite must be a non-eligible working_set")
    source_rows = source.get("scenarios")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("source suite scenarios missing")
    source_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for row in source_rows:
        if not isinstance(row, dict):
            raise ValueError("source suite row must be an object")
        identity = _identity(row)
        if not all(identity) or identity in source_by_identity:
            raise ValueError("source suite identities must be complete and unique")
        source_by_identity[identity] = row

    selected: list[dict[str, Any]] = []
    selected_identities: set[tuple[str, str]] = set()
    unmatched_rows: list[dict[str, Any]] = []
    report_bindings: list[dict[str, Any]] = []
    terminal_identities: dict[tuple[str, str], Path] = {}
    report_implementation_sha256: str | None = None
    for report_path in report_paths:
        report = _load(report_path)
        _validate_report_binding(
            report,
            report_path,
            source_suite_sha256=source_suite_sha256,
        )
        implementation_sha256 = str(report["implementation_tree_sha256"])
        if report_implementation_sha256 is None:
            report_implementation_sha256 = implementation_sha256
        elif report_implementation_sha256 != implementation_sha256:
            raise ValueError("prefilter reports bind different implementations")
        source_implementation = source.get("implementation_tree_sha256")
        if source_implementation and source_implementation != implementation_sha256:
            raise ValueError("prefilter implementation does not match source suite")
        raw_rows = report.get("rows")
        if not isinstance(raw_rows, list):
            raise ValueError(f"prefilter report rows missing: {report_path}")
        for raw_row in raw_rows:
            if not isinstance(raw_row, dict):
                raise ValueError(f"prefilter report row must be an object: {report_path}")
            identity = _identity(raw_row)
            if not all(identity):
                raise ValueError(f"prefilter identity incomplete: {report_path}")
            disposition = raw_row.get("disposition")
            if not isinstance(disposition, str) or not disposition:
                raise ValueError(
                    f"prefilter terminal disposition missing: {identity[0]}"
                )
            prior_path = terminal_identities.get(identity)
            if prior_path is not None:
                raise ValueError(
                    "duplicate terminal disposition across reports: "
                    f"{identity[0]}: {prior_path}, {report_path}"
                )
            terminal_identities[identity] = report_path
        report_rows = _prefilter_rows(report, report_path)
        report_bindings.append(
            {
                "path": str(report_path),
                "sha256": _sha256(report_path),
                "source_suite_sha256": report["source_suite_sha256"],
                "implementation_tree_sha256": report[
                    "implementation_tree_sha256"
                ],
                "runtime_version": copy.deepcopy(report["runtime_version"]),
                "n_rows": len(raw_rows),
                "n_selected": len(report_rows),
            }
        )
        for report_row in report_rows:
            identity = _identity(report_row)
            if identity not in source_by_identity:
                if allow_unmatched:
                    unmatched_rows.append(
                        {
                            "scenario_id": identity[0],
                            "scenario_signature": identity[1],
                            "reason": "prefilter_identity_missing_from_source_suite",
                            "report": str(report_path),
                        }
                    )
                    continue
                raise ValueError(f"prefilter identity not found in source suite: {identity[0]}")
            if identity in selected_identities:
                raise ValueError(f"duplicate selected identity across reports: {identity[0]}")
            selected_identities.add(identity)
            selected.append(copy.deepcopy(source_by_identity[identity]))

    missing_terminal = sorted(set(source_by_identity) - set(terminal_identities))
    if missing_terminal:
        raise ValueError(
            "source inputs missing terminal disposition: "
            + ", ".join(scenario_id for scenario_id, _ in missing_terminal)
        )

    if not selected:
        raise ValueError("prefilter reports selected no rows")
    output = copy.deepcopy(source)
    output["scenarios"] = selected
    output["n_scenarios"] = len(selected)
    output["release_ready"] = False
    output["leaderboard_eligible"] = False
    output["status"] = "working_set"
    constraints = dict(output.get("constraints") or {})
    constraints["formal_evaluation_ready"] = False
    constraints["model_outcomes_used_for_filtering"] = False
    output["constraints"] = constraints
    output["prefilter"] = {
        "schema_version": "protocol21-prefilter-suite-v1",
        "core_admission": False,
        "full_protocol21_required": True,
        "source_suite_sha256": source_suite_sha256,
        "input_reports": report_bindings,
        "unmatched_rows": unmatched_rows,
        "selected_identities": [
            {"scenario_id": scenario_id, "scenario_signature": signature}
            for scenario_id, signature in sorted(selected_identities)
        ],
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="record prefilter rows absent from this source suite instead of silently dropping them",
    )
    args = parser.parse_args()
    suite = build_prefilter_suite(
        args.source_suite.resolve(),
        [path.resolve() for path in args.report],
        allow_unmatched=args.allow_unmatched,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": suite["status"], "n_scenarios": suite["n_scenarios"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
