#!/usr/bin/env python3
"""Audit a Core candidate against the four-level difficulty contract.

This pass checks source configuration only.  Passing rows remain pending
replay-backed behavior and 1-minimal gates; failing rows are excluded from the
strict working set instead of being silently relabelled.
"""

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

from core.difficulty_contract import (  # noqa: E402
    DIFFICULTY_CONTRACT_VERSION,
    evaluate_difficulty_contract,
)

CANDIDATE_DIR = REPO_ROOT / "release" / "dt_sched_bench_v0_52_0_candidate"
DEFAULT_SELECTION = (
    CANDIDATE_DIR / "refined_core_selection_v6_grounded_recovery.json"
)
DEFAULT_AUDIT = CANDIDATE_DIR / "difficulty_contract_audit_v6.json"
DEFAULT_FILTERED = (
    CANDIDATE_DIR
    / "refined_core_selection_v7_four_domain_configuration_gate.json"
)
RELEASED_DOMAINS = frozenset(
    {"power_grid", "logistics", "traffic", "microgrid", "datacenter"}
)


def _distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    dimensions = {
        "by_domain": "domain",
        "by_backend": "backend_kind",
        "by_family": "family",
        "by_difficulty_level": "difficulty_level",
    }
    return {
        name: dict(sorted(Counter(str(row.get(field)) for row in rows).items()))
        for name, field in dimensions.items()
    }


def audit_selection(
    selection: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    four_domain_failure_counts: Counter[str] = Counter()
    four_domain_disposition_by_cell: Counter[tuple[str, str, str]] = Counter()
    disposition_counts: Counter[str] = Counter()
    track_counts: Counter[str] = Counter()

    for row in selection.get("scenarios") or []:
        scenario_path = repo_root / str(row.get("path") or "")
        if not scenario_path.is_file():
            result = {
                "scenario_id": row.get("scenario_id"),
                "path": row.get("path"),
                "domain": row.get("domain"),
                "difficulty_level": row.get("difficulty_level"),
                "track": None,
                "disposition": "held_missing_scenario",
                "failures": ["scenario_file_exists"],
                "contract": None,
            }
        else:
            seed = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
            contract = evaluate_difficulty_contract(seed)
            domain = str(seed.get("domain") or row.get("domain") or "")
            failures = list(contract.get("failures") or [])
            if domain not in RELEASED_DOMAINS:
                disposition = "excluded_outside_candidate_domains"
            elif failures:
                disposition = "held_repair_relabel_or_retire"
            else:
                disposition = "configuration_pass_pending_behavior"
                retained.append(row)
            result = {
                "scenario_id": row.get("scenario_id"),
                "path": row.get("path"),
                "domain": domain,
                "family": seed.get("family"),
                "difficulty_level": seed.get("difficulty_level"),
                "track": contract.get("track"),
                "disposition": disposition,
                "failures": failures,
                "contract": contract,
            }
            track_counts[str(contract.get("track"))] += 1
            if domain in RELEASED_DOMAINS:
                four_domain_failure_counts.update(str(value) for value in failures)
                four_domain_disposition_by_cell[
                    (
                        domain,
                        str(seed.get("difficulty_level") or ""),
                        disposition,
                    )
                ] += 1
        disposition_counts[str(result["disposition"])] += 1
        failure_counts.update(str(value) for value in result["failures"])
        results.append(result)

    audit = {
        "schema_version": "0.1",
        "difficulty_contract_version": DIFFICULTY_CONTRACT_VERSION,
        "scope": "configuration_preflight_only",
        "source_selection_status": selection.get("status"),
        "n_input": len(results),
        "n_retained_candidate_configuration_pass": len(retained),
        "n_excluded": len(results) - len(retained),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "track_counts": dict(sorted(track_counts.items())),
        "failure_counts": dict(sorted(failure_counts.items())),
        "four_domain_failure_counts": dict(
            sorted(four_domain_failure_counts.items())
        ),
        "four_domain_disposition_by_cell": [
            {
                "domain": domain,
                "difficulty_level": level,
                "disposition": disposition,
                "count": count,
            }
            for (domain, level, disposition), count in sorted(
                four_domain_disposition_by_cell.items()
            )
        ],
        "important_limit": (
            "configuration pass is necessary but not sufficient; every "
            "retained row still requires deterministic replay, task-contract "
            "success, 1-minimal calibration, exact dependency evidence where "
            "required, duplicate/provenance gates, and model diagnostics"
        ),
        "results": results,
    }
    return audit, retained


def build_filtered_selection(
    selection: dict[str, Any],
    retained: list[dict[str, Any]],
    *,
    audit_path: Path,
) -> dict[str, Any]:
    result = dict(selection)
    distribution = _distribution(retained)
    n_retained = max(1, len(retained))
    result.update(
        {
            "schema_version": "0.5",
            "status": (
                "strict_candidate_domain_configuration_gated_candidate_"
                "pending_full_behavioral_recalibration"
            ),
            "leaderboard_eligible": False,
            "n_selected": len(retained),
            "selection_shortfall": max(
                0, int(selection.get("target_size") or 300) - len(retained)
            ),
            "difficulty_contract_audit": str(
                audit_path.relative_to(REPO_ROOT)
                if audit_path.is_relative_to(REPO_ROOT)
                else audit_path
            ),
            "distribution": distribution,
            "scenarios": retained,
        }
    )
    result["behavioral_difficulty_ladder"] = {
        "status": "pending_full_behavioral_recalibration",
        "recalibration_required": True,
        "groups": [],
    }
    validation = dict(result.get("constraint_validation") or {})
    validation.update(
        {
            "max_domain_share_actual": (
                max(distribution["by_domain"].values(), default=0) / n_retained
            ),
            "max_backend_share_actual": (
                max(distribution["by_backend"].values(), default=0) / n_retained
            ),
            "selected_family_difficulty_cells": len(
                {
                    (
                        str(row.get("family") or ""),
                        str(row.get("difficulty_level") or ""),
                    )
                    for row in retained
                }
            ),
            "passed": False,
            "promotion_blockers": [
                "target_size_shortfall",
                "full_deterministic_behavioral_recalibration_pending",
                "formal_three_family_three_repeat_model_gate_pending",
            ],
        }
    )
    result["constraint_validation"] = validation
    return result


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--filtered-output", type=Path, default=DEFAULT_FILTERED)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    audit, retained = audit_selection(selection)
    filtered = build_filtered_selection(
        selection,
        retained,
        audit_path=args.audit_output,
    )
    _atomic_write_json(args.audit_output, audit)
    _atomic_write_json(args.filtered_output, filtered)
    print(
        json.dumps(
            {
                "n_input": audit["n_input"],
                "n_retained": len(retained),
                "disposition_counts": audit["disposition_counts"],
                "failure_counts": audit["failure_counts"],
                "distribution": filtered["distribution"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
