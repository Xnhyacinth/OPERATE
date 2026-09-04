#!/usr/bin/env python3
"""Build a fail-closed repair/replace/retire queue from contract calibrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = REPO_ROOT / "release" / "dt_sched_bench_v0_52_0_candidate"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(
    report_paths: list[Path],
    registry_path: Path,
    core_path: Path,
) -> dict[str, Any]:
    registry = {
        str(row["scenario_id"]): row
        for row in (_read(registry_path).get("scenarios") or [])
    }
    core_ids = {
        str(row["scenario_id"])
        for row in (_read(core_path).get("scenarios") or [])
    }
    failed: dict[str, dict[str, Any]] = {}
    for report_path in report_paths:
        report = _read(report_path)
        if report.get("schema_version") != "1.0" or report.get("status") != "complete":
            raise ValueError(f"incomplete task calibration: {report_path}")
        for result in report.get("results") or []:
            if result.get("status") != "failed":
                continue
            scenario_id = str(result["scenario_id"])
            source = registry.get(scenario_id) or {}
            ledger = source.get("case_ledger") or {}
            failed[scenario_id] = {
                **result,
                "disposition": "retire",
                "remediation_status": (
                    "retired_from_core_candidate"
                    if scenario_id not in core_ids
                    else "blocking_still_in_core"
                ),
                "independent_source_lock": source.get("source_lock") or {},
                "source_denominator_key": (
                    ledger.get("source_denominator_key")
                    or source.get("source_key")
                ),
            }
    items = [failed[key] for key in sorted(failed)]
    blocking = [
        row["scenario_id"]
        for row in items
        if row["remediation_status"] == "blocking_still_in_core"
    ]
    return {
        "schema_version": "1.1",
        "status": "complete" if not blocking else "blocked",
        "suite_id": _read(core_path).get("suite_id"),
        "policy": (
            "repair only with new behavioral evidence; otherwise replace from "
            "an independently locked source or retire without weakening gates"
        ),
        "n_failed": len(items),
        "n_repaired": 0,
        "n_replaced": 0,
        "n_retired": len(items) - len(blocking),
        "all_failed_absent_from_core": not blocking,
        "blocking_scenario_ids": blocking,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument(
        "--registry", type=Path, default=CANDIDATE_DIR / "candidate_registry.json"
    )
    parser.add_argument(
        "--core", type=Path, default=CANDIDATE_DIR / "validated_core_suite.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CANDIDATE_DIR / "task_contract_remediation_queue.json",
    )
    args = parser.parse_args()
    queue = build(
        [path.resolve() for path in args.report],
        args.registry.resolve(),
        args.core.resolve(),
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                key: queue[key]
                for key in (
                    "status",
                    "n_failed",
                    "n_retired",
                    "all_failed_absent_from_core",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
