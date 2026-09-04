#!/usr/bin/env python3
"""Persist held Protocol-2.1 candidates and their remediation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.working_set_contract import (  # noqa: E402
    validate_protocol21_row_lineage,
)

RERUN_GATES = (
    "static_preflight",
    "source_consumption",
    "material_headroom",
    "world_evolution",
    "task_contract",
    "baseline_comparison",
    "complexity",
    "observed_depth",
    "strategy_depth",
    "counterfactual_replay",
    "duplicate_detection",
    "provenance",
    "selection",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"output directory must be inside repository: {path}") from exc


def _remediation_class(
    difficulty_evidence: dict[str, Any],
    strategy_evidence: dict[str, Any],
) -> str:
    exact_depth = difficulty_evidence.get("exact_dependency_depth")
    tier_floor = strategy_evidence.get("tier_floor")
    if (
        isinstance(exact_depth, int)
        and isinstance(tier_floor, int)
        and exact_depth < tier_floor
        and difficulty_evidence.get("minimality_status")
        != "replay_budget_exhausted"
    ):
        return "redesign_task_dependency_structure"
    return "prove_required_depth_or_redesign"


def materialize_remediation_queue(
    *,
    working_set: dict[str, Any],
    source_grounded: dict[str, Any],
    strategy_depth: dict[str, Any],
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Copy held rows and emit a non-release remediation queue."""
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    rows = [
        row
        for row in working_set.get("scenarios") or []
        if isinstance(row, dict)
    ]
    rows_by_identity = {_identity(row): row for row in rows}
    strategy_by_identity = {
        _identity(row): row
        for row in strategy_depth.get("samples") or []
        if isinstance(row, dict)
    }
    held = [
        row
        for row in source_grounded.get("held") or []
        if isinstance(row, dict)
    ]
    if not held:
        raise ValueError("source-grounded artifact contains no held rows")
    missing = sorted(
        identity
        for identity in map(_identity, held)
        if identity not in rows_by_identity
    )
    if missing:
        raise ValueError(f"held identity missing from working set: {missing}")

    scenario_dir = output_dir / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for held_row in sorted(held, key=lambda item: _identity(item)):
        identity = _identity(held_row)
        working_row = rows_by_identity[identity]
        lineage_errors = validate_protocol21_row_lineage(working_row)
        if lineage_errors:
            raise ValueError(
                f"scenario lineage incomplete for {identity[0]}: {lineage_errors}"
            )
        source_path = Path(str(working_row.get("path") or ""))
        if not source_path.is_absolute():
            source_path = repo_root / source_path
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        destination = scenario_dir / source_path.name
        if destination.exists():
            raise ValueError(f"duplicate scenario filename: {source_path.name}")
        shutil.copy2(source_path, destination)

        stable_row = dict(working_row)
        stable_row["path"] = _repo_relative(destination, repo_root=repo_root)
        candidate_rows.append(stable_row)
        difficulty_evidence = held_row.get("difficulty_evidence")
        difficulty_evidence = (
            difficulty_evidence
            if isinstance(difficulty_evidence, dict)
            else {}
        )
        strategy_evidence = strategy_by_identity.get(identity, {})
        source_lock = (
            (working_row.get("case_ledger") or {}).get("physical_source_lock")
            or working_row.get("source_denominator_key")
            or working_row.get("source_key")
        )
        items.append(
            {
                "scenario_id": identity[0],
                "scenario_signature": identity[1],
                "domain": working_row.get("domain"),
                "backend_kind": working_row.get("backend_kind"),
                "difficulty_level": working_row.get("difficulty_level"),
                "current_disposition": "repair",
                "allowed_final_dispositions": [
                    "repair",
                    "replace",
                    "retire",
                ],
                "remediation_class": _remediation_class(
                    difficulty_evidence,
                    strategy_evidence,
                ),
                "failed_gates": sorted(
                    str(gate) for gate in held_row.get("failed_gates") or []
                ),
                "difficulty_evidence": difficulty_evidence,
                "strategy_depth_evidence": {
                    "tier_floor": strategy_evidence.get("tier_floor"),
                    "exact_task_dependency_depth": strategy_evidence.get(
                        "exact_task_dependency_depth"
                    ),
                    "core_action": strategy_evidence.get("core_action"),
                },
                "independent_source_lock": source_lock or "",
                "candidate_scenario_path": stable_row["path"],
                "candidate_scenario_sha256": _sha256(destination),
                "model_performance_used_for_disposition": False,
                "required_rerun_gates": list(RERUN_GATES),
            }
        )

    candidate_working_set = {
        "schema_version": "2.1",
        "status": "working_set",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(candidate_rows),
        "constraints": {
            "remediation_candidates_only": True,
            "one_per_effective_source_identity": True,
        },
        "scenarios": candidate_rows,
    }
    queue = {
        "schema_version": "2.1",
        "status": "remediation_required",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_items": len(items),
        "disposition_counts": dict(
            sorted(Counter(item["current_disposition"] for item in items).items())
        ),
        "items": items,
    }
    candidate_path = output_dir / "candidate_working_set.json"
    queue_path = output_dir / "remediation_queue.json"
    candidate_path.write_text(
        json.dumps(candidate_working_set, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    queue_path.write_text(
        json.dumps(queue, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "2.1",
        "status": "remediation_materialized",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_candidates": len(items),
        "candidate_working_set_sha256": _sha256(candidate_path),
        "remediation_queue_sha256": _sha256(queue_path),
    }
    (output_dir / "materialization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-set", type=Path, required=True)
    parser.add_argument("--source-grounded", type=Path, required=True)
    parser.add_argument("--strategy-depth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = materialize_remediation_queue(
            working_set=json.loads(
                args.working_set.read_text(encoding="utf-8")
            ),
            source_grounded=json.loads(
                args.source_grounded.read_text(encoding="utf-8")
            ),
            strategy_depth=json.loads(
                args.strategy_depth.read_text(encoding="utf-8")
            ),
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
