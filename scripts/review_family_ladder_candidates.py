#!/usr/bin/env python3
"""Review missing family tiers against all Core admission evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = REPO_ROOT / "release" / "dt_sched_bench_v0_52_0_candidate"
PUBLIC_LEVELS = ("basic", "medium", "high", "extreme")


def _source_key(row: dict[str, Any]) -> str:
    return str(
        (row.get("case_ledger") or {}).get("source_denominator_key")
        or row.get("source_key")
        or ""
    )


def build_review(
    *,
    operational: dict[str, Any],
    selected: dict[str, Any],
    candidates: dict[str, Any],
    calibrations: list[dict[str, Any]],
    contracts: list[dict[str, Any]],
    strategy_depth_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_rows = list(selected.get("scenarios") or [])
    candidate_rows = list(candidates.get("scenarios") or [])
    calibration_by_id = {
        str(row["scenario_id"]): row
        for report in calibrations
        for row in report.get("results") or []
    }
    contract_by_id = {
        str(row["scenario_id"]): row
        for report in contracts
        for row in report.get("results") or []
    }
    depth_by_id = {
        str(row["scenario_id"]): row
        for row in (strategy_depth_audit or {}).get("samples") or []
    }
    selected_semantics = {
        str(row.get("semantic_fingerprint") or "") for row in selected_rows
    }
    selected_structures = {
        str(row.get("structural_fingerprint") or "") for row in selected_rows
    }
    selected_sources = {_source_key(row) for row in selected_rows}
    families: list[dict[str, Any]] = []
    for target in operational.get("prioritized_gap_targets") or []:
        if target.get("code") != "incomplete_family_ladder":
            continue
        scope = str(target["scope"])
        domain, family = scope.split("/", 1)
        present = set((target.get("evidence") or {}).get("present_levels") or [])
        missing = [level for level in PUBLIC_LEVELS if level not in present]
        reviews: list[dict[str, Any]] = []
        for row in candidate_rows:
            if (
                str(row.get("domain")) != domain
                or str(row.get("family")) != family
                or str(row.get("difficulty_level")) not in missing
            ):
                continue
            scenario_id = str(row["scenario_id"])
            calibration = calibration_by_id.get(scenario_id) or {}
            contract = contract_by_id.get(scenario_id) or {}
            signature = str(row.get("scenario_signature") or "")
            semantic = str(row.get("semantic_fingerprint") or "")
            structural = str(row.get("structural_fingerprint") or "")
            source = _source_key(row)
            depth = depth_by_id.get(scenario_id)
            checks = {
                "native_behavior_passed": (
                    bool(signature)
                    and calibration.get("scenario_signature") == signature
                    and calibration.get("status") == "passed"
                    and bool(
                        (calibration.get("checks") or {}).get(
                            "native_state_changing_leverage"
                        )
                    )
                ),
                "task_contract_passed": (
                    bool(signature)
                    and contract.get("scenario_signature") == signature
                    and contract.get("status") == "passed"
                    and bool(contract.get("applicable"))
                    and bool(contract.get("completed"))
                ),
                "semantic_unique": bool(semantic)
                and semantic not in selected_semantics,
                "structural_unique": bool(structural)
                and structural not in selected_structures,
                "source_independent": bool(source) and source not in selected_sources,
                "strategy_depth_valid": (
                    depth is None or depth.get("core_action") != "replace_or_retire"
                ),
            }
            reviews.append(
                {
                    "scenario_id": scenario_id,
                    "difficulty_level": row.get("difficulty_level"),
                    "checks": checks,
                    "promotion_ready": all(checks.values()),
                    "blocking_checks": [
                        name for name, passed in checks.items() if not passed
                    ],
                }
            )
        if not reviews:
            status = "independent_candidate_missing"
        elif any(row["promotion_ready"] for row in reviews):
            status = "promotion_ready_pending_full_reselection"
        else:
            status = "candidate_gate_blocked"
        families.append(
            {
                "scope": scope,
                "present_levels": sorted(present, key=PUBLIC_LEVELS.index),
                "missing_levels": missing,
                "status": status,
                "candidates": reviews,
            }
        )
    return {
        "schema_version": "1.0",
        "scope": "incomplete_family_ladder_candidate_review",
        "n_families": len(families),
        "n_promotion_ready": sum(
            row["status"] == "promotion_ready_pending_full_reselection"
            for row in families
        ),
        "policy": {
            "behavior_and_contract_pass_do_not_override_duplicate_gate": True,
            "mode_or_difficulty_relabel_is_not_a_repair": True,
        },
        "families": families,
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RELEASE_DIR / "family_ladder_candidate_review.json",
    )
    args = parser.parse_args()
    selected = _load(RELEASE_DIR / "refined_core_selection.json")
    n_selected = int(selected.get("n_selected", 0) or 0)
    candidates = _load(RELEASE_DIR / "candidate_registry.json")
    candidates["scenarios"] = list(candidates.get("scenarios") or []) + list(
        _load(RELEASE_DIR / "coverage_candidates.json").get("scenarios") or []
    )
    report = build_review(
        operational=_load(
            RELEASE_DIR / f"core_{n_selected}_operational_validity.json"
        ),
        selected=selected,
        candidates=candidates,
        calibrations=[
            _load(RELEASE_DIR / "behavioral_calibration_v3_native.json"),
            _load(RELEASE_DIR / "traffic_ladder_candidate_calibration.json"),
        ],
        contracts=[
            _load(RELEASE_DIR / "task_contract_calibration_final.json"),
            _load(RELEASE_DIR / "traffic_ladder_candidate_task_contracts.json"),
        ],
        strategy_depth_audit=_load(
            RELEASE_DIR / "microgrid_candidate_strategy_depth_audit.json"
        ),
    )
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_families": report["n_families"],
                "n_promotion_ready": report["n_promotion_ready"],
                "statuses": {
                    row["scope"]: row["status"] for row in report["families"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
