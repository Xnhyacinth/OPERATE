#!/usr/bin/env python3
"""Materialize the strict refined v0.52 core from the audited selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = REPO_ROOT / "release" / "dt_sched_bench_v0_52_0_candidate"
DEFAULT_SELECTION = CANDIDATE_DIR / "refined_core_selection.json"
DEFAULT_SUITE = CANDIDATE_DIR / "validated_core_suite.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def materialize(
    selection_path: Path,
    previous_suite_path: Path,
    *,
    allow_distribution_blocked: bool = False,
) -> dict[str, Any]:
    selection = _read(selection_path)
    previous = _read(previous_suite_path)
    scenarios = []
    for row in selection.get("scenarios") or []:
        kept = dict(row)
        native = kept.pop("native_behavioral_validation", None)
        if native is not None:
            kept["behavioral_validation"] = {
                "status": "passed",
                **native,
            }
        scenarios.append(kept)

    if len(scenarios) != int(selection.get("n_selected", -1)):
        raise ValueError("selection count does not match selected scenarios")
    constraints = selection.get("constraint_validation") or {}
    constraints_passed = bool(constraints.get("passed"))
    if not constraints_passed and not allow_distribution_blocked:
        raise ValueError("selection diversity constraints are not satisfied")
    if (
        not constraints_passed
        and constraints.get("missing_family_difficulty_cells")
    ):
        raise ValueError("selection drops eligible family/difficulty cells")
    ladder = selection.get("behavioral_difficulty_ladder") or {}
    if int(ladder.get("n_promotion_blocking_groups", -1)) != 0:
        raise ValueError("selection still contains unresolved difficulty inversions")

    previous_ids = {str(row["scenario_id"]) for row in previous.get("scenarios") or []}
    selected_ids = {str(row["scenario_id"]) for row in scenarios}
    rejection_reason = {
        str(row["scenario_id"]): str(row.get("reason") or "not_selected")
        for row in selection.get("rejected") or []
    }
    removed = [
        {"scenario_id": scenario_id, "reason": rejection_reason.get(scenario_id, "not_selected")}
        for scenario_id in sorted(previous_ids - selected_ids)
    ]
    added = sorted(selected_ids - previous_ids)
    leaderboard_blockers = [
        "multi_model_screening_pending_on_strict_refined_suite",
        "bounded_replay_minimality_pending",
    ]
    if not constraints_passed:
        leaderboard_blockers.append("distribution_constraints_unsatisfied")
    return {
        "suite_id": "dt_sched_bench_v0_52_0_strict_refined_core_candidate",
        "status": (
            "strict_refined_core_pending_multi_model_and_minimality_validation"
            if constraints_passed
            else "quality_filtered_core_distribution_blocked"
        ),
        "leaderboard_eligible": False,
        "leaderboard_blockers": leaderboard_blockers,
        "public_difficulty_levels": ["basic", "medium", "high", "extreme"],
        "selection_schema_version": selection.get("schema_version"),
        "selection_constraints": selection.get("constraints"),
        "constraint_validation": constraints,
        "n_scenarios": len(scenarios),
        "n_removed_from_previous_core": len(removed),
        "n_added_from_candidate_pool": len(added),
        "by_domain": dict(sorted(Counter(str(row.get("domain")) for row in scenarios).items())),
        "by_backend": dict(
            sorted(Counter(str(row.get("backend_kind")) for row in scenarios).items())
        ),
        "by_difficulty_level": dict(
            sorted(Counter(str(row.get("difficulty_level")) for row in scenarios).items())
        ),
        "behavioral_difficulty_ladder": ladder,
        "removed_from_previous_core": removed,
        "added_from_candidate_pool": added,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--previous-suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--allow-distribution-blocked", action="store_true")
    args = parser.parse_args()
    report = materialize(
        args.selection.resolve(),
        args.previous_suite.resolve(),
        allow_distribution_blocked=args.allow_distribution_blocked,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "n_scenarios",
                    "n_removed_from_previous_core",
                    "n_added_from_candidate_pool",
                    "by_domain",
                    "by_difficulty_level",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
