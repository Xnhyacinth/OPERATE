#!/usr/bin/env python3
"""Merge fully gated grounded Microgrid rows into the v0.52 Core candidate."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SELECTION = (
    REPO_ROOT
    / "release/dt_sched_bench_v0_52_0_candidate"
    / "refined_core_selection_v4_source_grounded.json"
)
ADMISSION = (
    REPO_ROOT
    / "release/dt_sched_bench_v0_52_0_candidate"
    / "grounded_microgrid_candidate_admission.json"
)
POST_MINIMALITY = (
    REPO_ROOT
    / "release/dt_sched_bench_v0_52_0_candidate"
    / "grounded_microgrid_post_minimality.json"
)
MODEL_DIAGNOSTIC = (
    REPO_ROOT
    / "release/dt_sched_bench_v0_52_0_candidate"
    / "grounded_microgrid_model_diagnostic.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release/dt_sched_bench_v0_52_0_candidate"
    / "refined_core_selection_v5_grounded_recovery.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    dimensions = {
        "by_domain": "domain",
        "by_backend": "backend_kind",
        "by_family": "family",
        "by_difficulty_level": "difficulty_level",
    }
    return {
        name: dict(sorted(Counter(str(row[field]) for row in rows).items()))
        for name, field in dimensions.items()
    }


def _candidate_row(
    scenario: dict[str, Any],
    admission: dict[str, Any],
    minimality: dict[str, Any],
) -> dict[str, Any]:
    path = REPO_ROOT / str(scenario["path"])
    seed = yaml.safe_load(path.read_text(encoding="utf-8"))
    backend = seed.get("backend_config") or {}
    recipe = backend.get("derivation_recipe") or {}
    source_identity = {
        "backend": seed["backend_kind"],
        "site": backend.get("site"),
        "source_window_sha256": recipe.get("source_window_sha256"),
        "network": recipe.get("network"),
    }
    task = (admission.get("admission") or {}).get("task_contract") or {}
    return {
        "scenario_id": scenario["scenario_id"],
        "path": scenario["path"],
        "domain": seed["domain"],
        "backend_kind": seed["backend_kind"],
        "family": seed["family"],
        "difficulty_mode": seed["difficulty_mode"],
        "difficulty_level": seed["difficulty_level"],
        "horizon_ticks": seed["horizon_ticks"],
        "seed": seed["seed"],
        "source_key": json.dumps(
            source_identity,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "provenance_files": list((seed.get("provenance") or {}).get("files") or []),
        "case_ledger": {
            "site": backend.get("site"),
            "profile_start_index": backend.get("profile_start_index"),
            "source_window_sha256": recipe.get("source_window_sha256"),
            "task_contract": task.get("contract"),
            "admission_report": str(ADMISSION.relative_to(REPO_ROOT)),
            "minimality_report": str(POST_MINIMALITY.relative_to(REPO_ROOT)),
        },
        "scenario_signature": scenario["scenario_signature"],
        "structural_fingerprint": (admission.get("admission") or {})[
            "structural_fingerprint"
        ],
        "semantic_fingerprint": (admission.get("admission") or {})[
            "semantic_fingerprint"
        ],
        "source_difficulty_label": seed["difficulty_level"],
        "stress_profile": [
            {
                key: event.get(key)
                for key in (
                    "kind",
                    "trigger_tick",
                    "duration_ticks",
                    "hidden",
                    "intensity",
                    "target",
                )
            }
            for event in seed.get("perturbations") or []
        ],
        "candidate_gate": {
            "status": "passed",
            "checks": (admission.get("admission") or {}).get("checks") or {},
        },
        "native_behavioral_validation": {
            "status": "passed",
            "event_reachability": (
                (admission.get("admission") or {}).get("event_reachability")
                or []
            ),
        },
        "task_contract_validation": {
            "schema_version": task.get("schema_version"),
            "status": "passed",
            "contract": task.get("contract"),
            "completed": task.get("completed"),
            "reason_code": task.get("reason_code"),
        },
        "strategy_depth_validation": {
            "status": "passed",
            "one_minimal": minimality.get("one_minimal") or {},
        },
    }


def build_selection() -> dict[str, Any]:
    base = _load(BASE_SELECTION)
    admission = _load(ADMISSION)
    post = _load(POST_MINIMALITY)
    diagnostic = _load(MODEL_DIAGNOSTIC)
    admission_by_id = {
        str(row["scenario_id"]): row for row in admission.get("results") or []
    }
    post_by_id = {
        str(row["scenario_id"]): row for row in post.get("results") or []
    }
    diagnostic_by_id = {
        str(row["scenario_id"]): row
        for row in diagnostic.get("results") or []
    }

    additions: list[dict[str, Any]] = []
    for scenario in post.get("scenarios") or []:
        scenario_id = str(scenario["scenario_id"])
        admission_row = admission_by_id[scenario_id]
        post_row = post_by_id[scenario_id]
        diagnostic_row = diagnostic_by_id.get(scenario_id) or {}
        if admission_row.get("admission", {}).get("status") != (
            "preadmitted_pending_one_minimal"
        ):
            raise ValueError(f"{scenario_id}: pre-admission is not complete")
        if post_row.get("status") != "pending_model_discrimination":
            raise ValueError(f"{scenario_id}: one-minimal gate is not complete")
        if diagnostic_row.get("status") == "difficulty_label_review":
            raise ValueError(f"{scenario_id}: unresolved difficulty contradiction")
        additions.append(_candidate_row(scenario, admission_row, post_row))

    rows = list(base.get("scenarios") or []) + additions
    scenario_ids = [str(row["scenario_id"]) for row in rows]
    structural = [str(row["structural_fingerprint"]) for row in rows]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("scenario_id duplicate after grounded merge")
    if len(structural) != len(set(structural)):
        raise ValueError("structural duplicate after grounded merge")

    result = dict(base)
    result.update(
        {
            "schema_version": "0.3",
            "status": (
                "quality_gated_core_candidate_not_leaderboard_eligible_"
                "pending_current_semantics_recalibration"
            ),
            "leaderboard_eligible": False,
            "n_selected": len(rows),
            "selection_shortfall": max(
                0, int(base.get("target_size") or 300) - len(rows)
            ),
            "n_grounded_microgrid_added": len(additions),
            "grounded_microgrid_evidence": {
                "admission": str(ADMISSION.relative_to(REPO_ROOT)),
                "minimality": str(POST_MINIMALITY.relative_to(REPO_ROOT)),
                "model_diagnostic": str(MODEL_DIAGNOSTIC.relative_to(REPO_ROOT)),
                "model_role": (
                    "difficulty/discrimination diagnostic; model failure is "
                    "not a data-quality rejection"
                ),
            },
            "distribution": _distribution(rows),
            "scenarios": rows,
        }
    )
    domain_counts = result["distribution"]["by_domain"]
    backend_counts = result["distribution"]["by_backend"]
    n_rows = max(1, len(rows))
    validation = dict(result.get("constraint_validation") or {})
    selected_cells = {
        f"{row['family']}/{row['difficulty_level']}" for row in rows
    }
    source_rejected_cells = [
        str(cell)
        for cell in validation.get("source_rejected_family_difficulty_cells")
        or []
        if str(cell) not in selected_cells
    ]
    validation.update(
        {
            "max_domain_share_actual": max(domain_counts.values()) / n_rows,
            "max_backend_share_actual": max(backend_counts.values()) / n_rows,
            "eligible_family_difficulty_cells": len(selected_cells),
            "selected_family_difficulty_cells": len(selected_cells),
            "source_rejected_family_difficulty_cells": source_rejected_cells,
            "passed": False,
            "promotion_blockers": [
                "target_size_shortfall",
                "max_domain_share",
                "max_backend_share",
                "current_semantics_full-suite_recalibration_pending",
            ],
        }
    )
    result["constraint_validation"] = validation
    result["behavioral_difficulty_ladder"] = {
        "status": "pending_current_semantics_full_suite_recalibration",
        "recalibration_required": True,
        "groups": [],
    }
    return result


def main() -> None:
    result = build_selection()
    DEFAULT_OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_selected": result["n_selected"],
                "selection_shortfall": result["selection_shortfall"],
                "n_grounded_microgrid_added": result[
                    "n_grounded_microgrid_added"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
