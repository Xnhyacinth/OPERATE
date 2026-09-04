#!/usr/bin/env python3
"""Merge independently produced Protocol-2.1 candidate artifacts.

This utility only composes a source suite.  It does not promote rows, reuse
old evidence, or run a quality gate.  The full Protocol-2.1 pipeline remains
the authority for admission under the current implementation fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    for key in ("scenarios", "results", "samples", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            rows = [row for row in value if isinstance(row, dict)]
            if len(rows) != len(value):
                raise ValueError(f"{path}: non-object row in {key}")
            return rows
    raise ValueError(f"{path}: no candidate row list")


def _canonical(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"))


def build_source_suite(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one candidate artifact is required")

    by_identity: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    by_scenario_id: dict[str, set[str]] = defaultdict(set)
    source_bindings: list[dict[str, Any]] = []
    input_counts: dict[str, int] = {}
    constraint_candidates: list[dict[str, Any]] = []

    for raw_path in paths:
        path = raw_path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: root must be an object")
        rows = _rows(payload, path)
        constraints = payload.get("constraints")
        if isinstance(constraints, dict) and constraints:
            constraint_candidates.append(constraints)
        input_counts[str(path)] = len(rows)
        source_bindings.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "schema_version": payload.get("schema_version"),
                "status": payload.get("status"),
                "implementation_tree_sha256": payload.get(
                    "implementation_tree_sha256"
                ),
                "n_rows": len(rows),
            }
        )
        for row in rows:
            scenario_id = str(row.get("scenario_id") or "")
            signature = str(row.get("scenario_signature") or "")
            if not scenario_id or not signature:
                raise ValueError(
                    f"{path}: row missing scenario_id or scenario_signature"
                )
            identity = (scenario_id, signature)
            by_scenario_id[scenario_id].add(signature)
            existing = by_identity.get(identity)
            if existing is None:
                by_identity[identity] = (row, path)
                continue
            previous, previous_path = existing
            if _canonical(previous) != _canonical(row):
                raise ValueError(
                    "identity maps to different content: "
                    f"{scenario_id}|{signature}: "
                    f"{previous_path} != {path}"
                )

    signature_conflicts = {
        scenario_id: sorted(signatures)
        for scenario_id, signatures in by_scenario_id.items()
        if len(signatures) > 1
    }
    if signature_conflicts:
        raise ValueError(
            "scenario_id maps to multiple signatures: "
            + json.dumps(signature_conflicts, sort_keys=True)
        )

    rows = [
        row
        for _, (row, _) in sorted(
            by_identity.items(), key=lambda item: item[0]
        )
    ]
    effective_sources = [
        str(row.get("source_denominator_key") or "") for row in rows
    ]
    effective_source_counts = Counter(
        key for key in effective_sources if key
    )
    duplicate_groups = {
        key: count
        for key, count in sorted(effective_source_counts.items())
        if count > 1
    }
    constraints = constraint_candidates[0] if constraint_candidates else {}
    constraint_conflict = any(
        json.dumps(candidate, sort_keys=True) != json.dumps(constraints, sort_keys=True)
        for candidate in constraint_candidates[1:]
    )
    return {
        "schema_version": "protocol21-expansion-source-suite-v1",
        "status": "working_set",
        "selection_policy": "quality_maximal_v1",
        "leaderboard_eligible": False,
        "constraints": constraints,
        "n_scenarios": len(rows),
        "scenarios": rows,
        "source_artifacts": source_bindings,
        "merge_audit": {
            "n_input_artifacts": len(paths),
            "input_counts": input_counts,
            "n_input_rows": sum(input_counts.values()),
            "n_unique_identity_rows": len(rows),
            "n_exact_identity_duplicates_removed": (
                sum(input_counts.values()) - len(rows)
            ),
            "effective_source_duplicate_groups": duplicate_groups,
            "identity_variant_groups": {},
            "identity_variant_policy": "fail_closed_on_content_mismatch",
            "duplicate_effective_sources_are_diagnostic_only": True,
            "model_outcomes_used_for_filtering": False,
            "constraints_carried_from_input": bool(constraints),
            "constraint_input_conflict": constraint_conflict,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = build_source_suite(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: output[k] for k in ("status", "n_scenarios")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
