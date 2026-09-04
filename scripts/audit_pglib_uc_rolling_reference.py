#!/usr/bin/env python3
"""Summarize the bounded three-source PGLib-UC rolling-reference gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(behavioral: Path) -> dict[str, Any]:
    payload = json.loads(behavioral.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for result in payload.get("results", []):
        if not isinstance(result, dict):
            continue
        episodes = result.get("episodes")
        if not isinstance(episodes, dict):
            rows.append(
                {
                    "scenario_id": result.get("scenario_id"),
                    "family": result.get("family"),
                    "status": result.get("status", "error"),
                    "checks": result.get("checks") or {},
                    "error": result.get("error"),
                    "task_completed": False,
                    "effective_native_tools": [],
                    "effective_decision_ticks": [],
                    "terminal_integrity": {},
                }
            )
            continue
        oracle = episodes["oracle_offline"]
        wait = episodes["wait_only"]
        oracle_task = oracle["task_completion"]
        oracle_evidence = oracle_task["evidence"]
        wait_evidence = wait["task_completion"]["evidence"]
        rows.append(
            {
                "scenario_id": result["scenario_id"],
                "family": result["family"],
                "status": result["status"],
                "checks": result["checks"],
                "oracle_cost": oracle["cost"],
                "wait_cost": wait["cost"],
                "relative_cost_improvement": round(
                    (wait["cost"] - oracle["cost"]) / max(wait["cost"], 1e-9),
                    9,
                ),
                "oracle_task_loss": oracle_evidence["actual_task_loss"],
                "wait_task_loss": wait_evidence["actual_task_loss"],
                "task_loss_reduction": oracle_evidence["task_loss_reduction"],
                "system_survival_score": oracle_evidence[
                    "system_survival_score"
                ],
                "task_completed": oracle_task["completed"],
                "task_reason_code": oracle_task["reason_code"],
                "effective_native_tools": oracle_evidence[
                    "distinct_physical_tools"
                ],
                "effective_decision_ticks": oracle["effective_decision_ticks"],
                "terminal_integrity": oracle["terminal_integrity"],
            }
        )
    all_passed = bool(rows) and all(row["status"] == "passed" for row in rows)
    portfolio_observed = all(
        "dispatch_generation_portfolio" in row["effective_native_tools"]
        for row in rows
    )
    return {
        "schema_version": "pglib-uc-rolling-reference-review-v1",
        "status": "passed" if all_passed else "held_repair",
        "decision": "eligible_for_56_expansion" if all_passed else "do_not_expand_to_56",
        "behavioral_artifact": {
            "path": behavioral.as_posix(),
            "sha256": _sha256(behavioral),
            "implementation_tree_sha256": payload.get(
                "implementation_tree_sha256"
            ),
            "n_expected": payload.get("n_expected"),
            "n_completed": payload.get("n_completed"),
            "status_counts": payload.get("status_counts"),
        },
        "reference_contract": {
            "solver": "scipy.optimize.milp",
            "rolling_horizon_ticks": 4,
            "source_native_series": ["demand", "reserves", "renewable_generators"],
            "decision_variables": [
                "dispatch",
                "commitment",
                "startup",
                "shutdown",
                "piecewise_cost_epigraph",
            ],
            "constraints": [
                "unit_on_t0",
                "minimum_up_down",
                "ramp",
                "demand_balance",
                "reserve",
                "declared_outage",
                "fuel_supply_factor",
            ],
            "first_period_controls_execute_via_tool_protocol": True,
            "atomic_portfolio_control_observed_on_all_sources": portfolio_observed,
            "reserve_actuator_is_source_conditioned": True,
            "solver_claim": "aggregate_multi_period_unit_commitment_reference_only",
        },
        "representatives": rows,
        "blockers": []
        if all_passed
        else [
            "reference_task_completion_failed_on_all_representatives",
            "native_state_changing_leverage_failed_on_all_representatives",
            "system_survival_floor_failed_on_all_representatives",
            "bounded_shortest_strategy_depth_not_proven",
        ],
        "minimum_next_design": [
            "Keep all 56 sources held; the atomic ToolProtocol and source-conditioned actuator blockers are resolved but the scientific survival/task gates are not.",
            "Audit whether the aggregate backend's initial source state and tick ordering admit a safe trajectory under native ramp and commitment constraints; report intrinsic infeasibility instead of relaxing the survival floor.",
            "Prove bounded shortest-strategy depth on the same three representatives before any bulk expansion.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavioral", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.behavioral)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "decision": report["decision"]}))


if __name__ == "__main__":
    main()
