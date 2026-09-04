#!/usr/bin/env python3
"""Mine additional disjoint CityLearn 72-hour candidate windows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from scripts.build_protocol21_candidate_source_suite import (  # noqa: E402
    build_suite as build_candidate_source_suite,
)
from scripts.refine_citylearn_long_horizon_candidates import (  # noqa: E402
    DEFAULT_INPUT,
    SEED,
    WindowPlan,
    _evaluate_one,
    _load_base_rows,
    _repo_ref,
    _sha256,
)

EXPANSION_PLANS = tuple(
    WindowPlan(dataset_id, start)
    for dataset_id in (
        "citylearn_challenge_2022_phase_1",
        "citylearn_challenge_2022_phase_2",
        "citylearn_challenge_2022_phase_3",
    )
    for start in (216, 720)
)
DEFAULT_STAGING = (
    REPO_ROOT / "scenarios" / "staging" / "citylearn_expansion_20260814_v58"
)
DEFAULT_REPORT = (
    REPO_ROOT / "reports" / "citylearn_expansion_20260814_v58" / "terminal_ledger.json"
)


def run_expansion(
    *,
    input_suite: Path,
    staging_root: Path,
    report_path: Path,
    plans: tuple[WindowPlan, ...] | None = None,
    suite_id: str = "citylearn_expansion_20260814_v58",
    rerun_command: str = (
        ".venv/bin/python scripts/refine_citylearn_expansion_candidates.py"
    ),
    extra_report: dict[str, Any] | None = None,
) -> dict[str, object]:
    selected_plans = tuple(plans) if plans is not None else EXPANSION_PLANS
    start_identity = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    bases = _load_base_rows(input_suite)
    rows: list[dict[str, object]] = []
    accepted: list[tuple[dict[str, object], dict[str, object]]] = []
    for plan in selected_plans:
        row, scenario = _evaluate_one(
            plan=plan,
            base=bases[plan.dataset_id]["scenario"],
            implementation_start=start_identity,
        )
        rows.append(row)
        if scenario is not None:
            accepted.append((row, scenario))

    end_identity = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    if end_identity != start_identity:
        accepted.clear()
        for row in rows:
            blockers = row.setdefault("blockers", [])
            if "implementation_tree_changed_during_run" not in blockers:
                blockers.append("implementation_tree_changed_during_run")
            row["status"] = "held_runtime"
            row["disposition"] = "held_runtime"

    staging_root.mkdir(parents=True, exist_ok=True)
    scenario_rows: list[dict[str, object]] = []
    for row, scenario in accepted:
        source_window = list(row["source_window"])
        path = staging_root / (
            f"{row['dataset_id']}_w{source_window[0]}_{source_window[1]}_72h_extreme.yaml"
        )
        path.write_text(yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8")
        source_identity = dict(row["source_identity"])
        scenario_rows.append(
            {
                "scenario_id": row["scenario_id"],
                "scenario_signature": row["scenario_signature"],
                "path": _repo_ref(path),
                "domain": "building_energy",
                "backend_kind": "citylearn",
                "family": "citylearn_der_storage_control",
                "difficulty_level": "extreme",
                "difficulty_mode": "source_locked_long_horizon",
                "horizon_ticks": 72,
                "seed": SEED,
                "source_denominator_key": source_identity["effective_source_key"],
                "physical_source_key": source_identity["physical_source_key"],
                "effective_source_key": source_identity["effective_source_key"],
                "candidate_only": True,
                "status": "pending_full_protocol21",
                "readiness_blockers": ["full_protocol21_gate_chain_pending"],
            }
        )

    candidate_report = {
        "schema_version": "citylearn_expansion_candidate_report_v1",
        "suite_id": suite_id,
        "status": "staging_candidates_pending_full_admission",
        "candidate_only": True,
        "leaderboard_eligible": False,
        "release_ready": False,
        "full_protocol21_executed": False,
        "n_scenarios": len(scenario_rows),
        "one_per_effective_source_identity": True,
        "scenarios": scenario_rows,
    }
    candidate_report_path = staging_root / "candidate_report.json"
    candidate_report_path.write_text(
        json.dumps(candidate_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    suite = build_candidate_source_suite(candidate_report_path)
    suite["suite_id"] = suite_id
    suite_path = staging_root / "source_suite.json"
    suite_path.write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": "citylearn_expansion_terminal_ledger_v1",
        "status": "candidate_survivors" if accepted else "held",
        "candidate_only": True,
        "core_admission_claimed": False,
        "full_protocol21_executed": False,
        "input": {"path": _repo_ref(input_suite), "sha256": _sha256(input_suite)},
        "implementation_stability": {
            "start": start_identity,
            "end": end_identity,
            "stable": start_identity == end_identity,
        },
        "window_plans": [asdict(plan) for plan in selected_plans],
        "summary": {
            "attempted": len(rows),
            "candidate_survivors": len(accepted),
            "held": len(rows) - len(accepted),
        },
        "rows": rows,
        "source_suite": {"path": _repo_ref(suite_path), "sha256": _sha256(suite_path)},
        "rerun_command": rerun_command,
    }
    if extra_report:
        report.update(extra_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-suite", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = run_expansion(
        input_suite=args.input_suite,
        staging_root=args.staging_root,
        report_path=args.report,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0 if report["summary"]["candidate_survivors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
