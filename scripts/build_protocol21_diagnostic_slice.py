#!/usr/bin/env python3
"""Build the deterministic 40-row Protocol-2.1 diagnostic LLM smoke slice."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_DOMAIN_COUNTS = {
    "power_grid": 4,
    "microgrid": 5,
    "traffic": 8,
    "logistics": 11,
    "datacenter": 12,
}
DIFFICULTY_PRIORITY = {"extreme": 0, "high": 1, "medium": 2, "basic": 3}


def _physical_identity(row: dict[str, Any]) -> str:
    ledger = row.get("case_ledger") or {}
    value = ledger.get("physical_source_lock") or row.get("physical_source_key")
    if value in (None, "", {}, []):
        raise ValueError(f"physical source identity missing: {row.get('scenario_id')}")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _priority(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        DIFFICULTY_PRIORITY.get(str(row.get("difficulty_level")), 99),
        0 if row.get("difficulty_mode") == "deep_planning" else 1,
        str(row.get("source_denominator_key") or ""),
        str(row.get("scenario_id") or ""),
    )


def build_diagnostic_slice(
    readiness: dict[str, Any],
    *,
    domain_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    counts = dict(domain_counts or DEFAULT_DOMAIN_COUNTS)
    rows = readiness.get("scenarios")
    if readiness.get("formal_evaluation_ready") is not True:
        raise ValueError("diagnostic source readiness is not formal-evaluation green")
    if not isinstance(rows, list):
        raise ValueError("readiness scenarios missing")

    selected: list[dict[str, Any]] = []
    for domain, target in counts.items():
        candidates = sorted(
            (
                row
                for row in rows
                if isinstance(row, dict)
                and row.get("domain") == domain
                and row.get("status") == "core_locked"
            ),
            key=_priority,
        )
        if len(candidates) < target:
            raise ValueError(
                f"diagnostic domain {domain} requires {target}, found {len(candidates)}"
            )
        unique_first: list[dict[str, Any]] = []
        repeated: list[dict[str, Any]] = []
        seen_physical: set[str] = set()
        for row in candidates:
            physical = _physical_identity(row)
            if physical in seen_physical:
                repeated.append(row)
            else:
                seen_physical.add(physical)
                unique_first.append(row)
        selected.extend(copy.deepcopy((unique_first + repeated)[:target]))

    selected.sort(key=lambda row: str(row.get("scenario_id") or ""))
    identities = [
        (str(row.get("scenario_id") or ""), str(row.get("scenario_signature") or ""))
        for row in selected
    ]
    if any(not all(identity) for identity in identities) or len(set(identities)) != len(
        identities
    ):
        raise ValueError("diagnostic identities must be complete and unique")
    return {
        "schema_version": "protocol21-diagnostic-slice-v1",
        "status": "working_set",
        "leaderboard_eligible": False,
        "release_ready": False,
        "diagnostic_only": True,
        "selection_policy": (
            "physical_source_first_then_extreme_high_deep_planning_v1"
        ),
        "source_readiness_sha256": None,
        "n_scenarios": len(selected),
        "domain_counts": counts,
        "scenarios": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    readiness = json.loads(args.readiness.read_text(encoding="utf-8"))
    result = build_diagnostic_slice(readiness)
    result["source_readiness_sha256"] = hashlib.sha256(
        args.readiness.read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n_scenarios": result["n_scenarios"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
