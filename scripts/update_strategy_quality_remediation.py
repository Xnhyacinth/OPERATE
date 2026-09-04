#!/usr/bin/env python3
"""Merge replay-proven difficulty contradictions into the retirement queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = REPO_ROOT / "release" / "dt_sched_bench_v0_52_0_candidate"


def merge_depth_retirements(
    *, queue: dict[str, Any], depth_audit: dict[str, Any]
) -> dict[str, Any]:
    if not (
        depth_audit.get("complete") is True
        or depth_audit.get("status") == "complete"
    ):
        raise ValueError("strategy-depth audit must be complete before retirement")
    items = {
        str(row["scenario_id"]): dict(row) for row in queue.get("items") or []
    }
    depth_reason_codes = {
        "successful_strategy_depth_below_tier_floor",
        "task_completing_reference_trace_below_tier_floor",
    }
    # A complete fresh audit supersedes older depth-only decisions for the
    # same scenario. Unrelated retirement categories remain immutable.
    for row in depth_audit.get("samples") or []:
        scenario_id = str(row["scenario_id"])
        prior = items.get(scenario_id)
        if (
            prior is not None
            and prior.get("reason_code") in depth_reason_codes
            and row.get("core_action") in {"keep", "hold_pending_lower_bound"}
        ):
            del items[scenario_id]
    for row in depth_audit.get("samples") or []:
        replay_contradiction = row.get("core_action") == "replace_or_retire"
        task_contract_contradiction = (
            row.get("disposition")
            == "replace_or_retire_depth_contradicted"
        )
        if not (replay_contradiction or task_contract_contradiction):
            continue
        scenario_id = str(row["scenario_id"])
        reason_code = (
            "task_completing_reference_trace_below_tier_floor"
            if task_contract_contradiction
            else "successful_strategy_depth_below_tier_floor"
        )
        prior = items.get(scenario_id)
        if prior is None or prior.get("decision") != "retire":
            items[scenario_id] = {
                "scenario_id": scenario_id,
                "decision": "retire",
                "reason_code": reason_code,
            }
    ordered = [items[key] for key in sorted(items)]
    prior_reason = queue.get("reason") or {}
    reason_catalog = dict(queue.get("reason_catalog") or {})
    if prior_reason.get("code") == "destructive_shortcut_only":
        reason_catalog["destructive_shortcut_only"] = {
            key: value for key, value in prior_reason.items() if key != "code"
        }
    reason_catalog["successful_strategy_depth_below_tier_floor"] = {
        "evidence": (
            "Bounded deterministic replay found a successful strategy "
            "tick upper bound below the public difficulty tier floor."
        ),
        "repair_requirement": (
            "Redesign with source-supported temporal dependencies, then "
            "rerun behavior, task contract, minimality, difficulty, "
            "duplicate, provenance and model gates."
        ),
    }
    reason_catalog["task_completing_reference_trace_below_tier_floor"] = {
        "evidence": (
            "A deterministic reference episode completed the native task "
            "contract with fewer effective decision ticks than the public "
            "difficulty tier floor."
        ),
        "repair_requirement": (
            "Redesign the task with source-supported temporal dependencies "
            "or relabel it, then rerun behavior, task contract, minimality, "
            "difficulty, duplicate, provenance and model gates."
        ),
    }
    merged = dict(queue)
    merged.update(
        {
            "status": "complete",
            "policy": (
                "Core excludes samples whose only calibrated value is a "
                "destructive accounting shortcut or whose replay-proven "
                "successful strategy is shallower than the declared tier."
            ),
            "n_reviewed": len(ordered),
            "n_retired": len(ordered),
            "reason": {
                "code": "multiple_strategy_quality_failures",
                "categories": sorted(reason_catalog),
            },
            "reason_catalog": reason_catalog,
            "items": ordered,
        }
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queue",
        type=Path,
        default=RELEASE_DIR / "strategy_quality_remediation_queue.json",
    )
    parser.add_argument(
        "--depth-audit",
        type=Path,
        default=RELEASE_DIR / "microgrid_strategy_depth_audit.json",
    )
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    audit = json.loads(args.depth_audit.read_text(encoding="utf-8"))
    merged = merge_depth_retirements(queue=queue, depth_audit=audit)
    args.queue.write_text(
        json.dumps(merged, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "n_reviewed": merged["n_reviewed"],
                "n_retired": merged["n_retired"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
