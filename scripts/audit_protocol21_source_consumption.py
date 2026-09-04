#!/usr/bin/env python3
"""Build current, replay-bound source-consumption evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import (  # noqa: E402
    artifact_binding,
    extract_semantics,
    report_rows,
    required_semantics,
    row_identity,
)
from core.source_consumption_contract import (  # noqa: E402
    normalize_runtime_source_evidence,
)


def build_report(
    *,
    suite: dict[str, Any],
    behavioral: dict[str, Any],
    suite_path: Path,
    behavioral_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    rows = report_rows(suite)
    tree_hash = implementation_identity(repo_root)[
        "implementation_tree_sha256"
    ]
    behavior_rows = report_rows(behavioral)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in behavior_rows:
        grouped.setdefault(row_identity(row), []).append(row)
    results: list[dict[str, Any]] = []
    structural_errors: list[str] = []
    if extract_semantics(behavioral) != required_semantics():
        structural_errors.append("behavioral_semantics_stale")
    if behavioral.get("status") != "complete":
        structural_errors.append("behavioral_incomplete")
    if behavioral.get("implementation_tree_sha256") != tree_hash:
        structural_errors.append("implementation_tree_mismatch")

    for row in rows:
        identity = row_identity(row)
        matches = grouped.get(identity, [])
        if len(matches) != 1:
            structural_errors.append("behavioral_identity_multiplicity")
            results.append(
                {
                    "scenario_id": identity[0],
                    "scenario_signature": identity[1],
                    "backend_kind": row.get("backend_kind"),
                    "status": "held",
                    "blockers": ["behavioral_identity_multiplicity"],
                }
            )
            continue
        path = Path(str(row.get("path") or ""))
        scenario_path = path if path.is_absolute() else repo_root / path
        try:
            scenario = yaml.safe_load(
                scenario_path.read_text(encoding="utf-8")
            )
            if not isinstance(scenario, dict):
                raise ValueError("scenario YAML is not a mapping")
        except Exception:
            structural_errors.append("scenario_yaml_unreadable")
            results.append(
                {
                    "scenario_id": identity[0],
                    "scenario_signature": identity[1],
                    "backend_kind": row.get("backend_kind"),
                    "status": "held",
                    "blockers": ["scenario_yaml_unreadable"],
                }
            )
            continue
        replay = matches[0].get("replay_evidence") or {}
        results.append(
            normalize_runtime_source_evidence(
                row=row,
                scenario=scenario,
                replay_evidence=[
                    replay.get("source_consumption_first") or {},
                    replay.get("source_consumption_second") or {},
                ],
                repo_root=repo_root,
            )
        )

    status = (
        "complete"
        if not structural_errors and len(results) == len(rows)
        else "partial"
    )
    counts = Counter(str(row.get("status") or "held") for row in results)

    def grouped_counts(field: str) -> dict[str, dict[str, int]]:
        output: dict[str, Counter[str]] = {}
        source_by_identity = {row_identity(row): row for row in rows}
        for result in results:
            source_row = source_by_identity.get(row_identity(result), {})
            key = str(source_row.get(field) or result.get(field) or "")
            output.setdefault(key, Counter()).update(
                [str(result.get("status") or "held")]
            )
        return {
            key: dict(sorted(value.items()))
            for key, value in sorted(output.items())
        }

    return {
        "schema_version": "1.0",
        "status": status,
        "evaluation_semantics": required_semantics(),
        "implementation_tree_sha256": tree_hash,
        "input_bindings": {
            "source_suite": artifact_binding(
                suite_path,
                implementation_tree_sha256=tree_hash,
            ),
            "behavioral": artifact_binding(
                behavioral_path,
                implementation_tree_sha256=tree_hash,
            ),
        },
        "n_expected": len(rows),
        "n_completed": len(results),
        "n_passed": counts["passed"],
        "n_held": counts["held"],
        "n_failed": counts["failed"],
        "structural_errors": sorted(set(structural_errors)),
        "results": results,
        "by_backend": grouped_counts("backend_kind"),
        "by_domain": grouped_counts("domain"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--behavioral", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite_path = args.suite.resolve()
    behavioral_path = args.behavioral.resolve()
    report = build_report(
        suite=json.loads(suite_path.read_text(encoding="utf-8")),
        behavioral=json.loads(behavioral_path.read_text(encoding="utf-8")),
        suite_path=suite_path,
        behavioral_path=behavioral_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
