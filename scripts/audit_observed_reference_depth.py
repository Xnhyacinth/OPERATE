#!/usr/bin/env python3
"""Reject task-completing reference traces shallower than their tier."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402

EXPANSION_DIR = (
    REPO_ROOT / "release" / "operate_v0_58_0_candidate" / "operate_v058_formal"
)
DEPTH_FLOORS = {"basic": 1, "medium": 1, "high": 2, "extreme": 3}


def build_report(
    *,
    behavioral: dict[str, Any],
    task_contracts: dict[str, Any],
) -> dict[str, Any]:
    behavior_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in behavioral.get("results") or []:
        behavior_grouped[
            (
                str(row.get("scenario_id") or ""),
                str(row.get("scenario_signature") or ""),
            )
        ].append(row)
    task_identities: Counter[tuple[str, str]] = Counter(
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in task_contracts.get("results") or []
    )
    multiplicity_error = any(
        len(rows) != 1 for rows in behavior_grouped.values()
    ) or any(count != 1 for count in task_identities.values())
    samples: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for task in task_contracts.get("results") or []:
        scenario_id = str(task["scenario_id"])
        identity = (
            scenario_id,
            str(task.get("scenario_signature") or ""),
        )
        matches = behavior_grouped.get(identity, [])
        behavior = matches[0] if len(matches) == 1 else None
        agent_name = str(task.get("agent_name") or "oracle_offline")
        episode = ((behavior or {}).get("episodes") or {}).get(agent_name) or {}
        level = str(
            task.get("difficulty_level")
            or (behavior or {}).get("difficulty_level")
            or ""
        )
        floor = DEPTH_FLOORS.get(level, 1)
        ticks = episode.get("effective_decision_ticks")
        task_success = (
            task.get("status") == "passed"
            and bool(task.get("applicable"))
            and bool(task.get("completed"))
        )
        contradicted = (
            task_success
            and isinstance(ticks, int)
            and ticks < floor
        )
        if behavior is None:
            disposition = "behavioral_result_missing"
        elif not task_success:
            disposition = "task_contract_not_completed"
        elif not isinstance(ticks, int):
            disposition = "reference_depth_missing"
        elif contradicted:
            disposition = "replace_or_retire_depth_contradicted"
            items.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_signature": task.get("scenario_signature"),
                    "decision": "retire",
                    "reason_code": "task_completing_reference_trace_below_tier_floor",
                }
            )
        else:
            disposition = "bounded_replay_required"
        samples.append(
            {
                "scenario_id": scenario_id,
                "scenario_signature": task.get("scenario_signature"),
                "difficulty_level": level,
                "tier_floor": floor,
                "agent_name": agent_name,
                "task_contract_completed": task_success,
                "observed_effective_decision_ticks": ticks,
                "disposition": disposition,
            }
        )
    complete = (
        behavioral.get("status") == "complete"
        and task_contracts.get("status") == "complete"
        and not multiplicity_error
        and all(row["disposition"] != "behavioral_result_missing" for row in samples)
    )
    counts = Counter(row["disposition"] for row in samples)
    return {
        "schema_version": "1.0",
        "status": "complete" if complete else "partial",
        "evaluation_semantics": {
            "protocol_version": (
                behavioral.get("evaluation_protocol_version")
                or task_contracts.get("evaluation_protocol_version")
                or ""
            ),
            "implementation_fingerprint": (
                behavioral.get("evaluation_implementation_fingerprint")
                or task_contracts.get("evaluation_implementation_fingerprint")
                or ""
            ),
            "scoring_version": (
                behavioral.get("scoring_version")
                or task_contracts.get("scoring_version")
                or ""
            ),
        },
        "implementation_tree_sha256": implementation_identity()[
            "implementation_tree_sha256"
        ],
        "scope": "task_completing_reference_trace_depth_gate",
        "n_expected": len(samples),
        "n_completed": len(samples),
        "n_reviewed": len(samples),
        "n_retired": len(items),
        "reason": {
            "code": "task_completing_reference_trace_below_tier_floor",
            "evidence": (
                "A reference episode completed the native task contract with fewer "
                "effective decision ticks than the public tier floor."
            ),
        },
        "summary": {"disposition_counts": dict(sorted(counts.items()))},
        "policy": {
            "successful_trace_below_floor_is_contradictory": True,
            "trace_at_or_above_floor_still_requires_bounded_replay": True,
        },
        "items": sorted(items, key=lambda row: row["scenario_id"]),
        "samples": sorted(samples, key=lambda row: row["scenario_id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behavioral",
        type=Path,
        default=EXPANSION_DIR / "behavioral_calibration_v3_native.json",
    )
    parser.add_argument(
        "--task-contracts",
        type=Path,
        default=EXPANSION_DIR / "task_contracts_protocol1_4.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPANSION_DIR / "observed_reference_depth_gate.json",
    )
    args = parser.parse_args()
    report = build_report(
        behavioral=json.loads(args.behavioral.read_text(encoding="utf-8")),
        task_contracts=json.loads(args.task_contracts.read_text(encoding="utf-8")),
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_reviewed": report["n_reviewed"],
                "n_retired": report["n_retired"],
                "dispositions": report["summary"]["disposition_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
