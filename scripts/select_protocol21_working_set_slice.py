#!/usr/bin/env python3
"""Select a lineage-safe, non-release Protocol-2.1 evidence slice."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.working_set_contract import (  # noqa: E402
    validate_protocol21_row_lineage,
)


def select_working_set_slice(
    *,
    source: dict[str, Any],
    domains: set[str],
    backends: set[str],
    freeze_scope: str,
    require_lineage_ready: bool,
) -> dict[str, Any]:
    """Return selected rows without granting release or leaderboard status."""
    if freeze_scope not in {"backend", "domain"}:
        raise ValueError("freeze_scope must be backend or domain")
    matching = [
        row
        for row in source.get("scenarios") or []
        if isinstance(row, dict)
        and (not domains or str(row.get("domain") or "") in domains)
        and (not backends or str(row.get("backend_kind") or "") in backends)
    ]
    held = [
        {
            "scenario_id": str(row.get("scenario_id") or ""),
            "scenario_signature": str(row.get("scenario_signature") or ""),
            "reason_codes": validate_protocol21_row_lineage(row),
        }
        for row in matching
        if validate_protocol21_row_lineage(row)
    ]
    selected = [
        row
        for row in matching
        if not require_lineage_ready
        or not validate_protocol21_row_lineage(row)
    ]
    selected.sort(
        key=lambda row: (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
    )
    if not selected:
        raise ValueError("working-set filters selected no scenarios")
    identities = [
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in selected
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("selected slice contains duplicate scenario identity")
    return {
        "schema_version": "2.1",
        "status": "working_set",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(selected),
        "n_matching_before_lineage_gate": len(matching),
        "n_held_lineage_incomplete": len(held),
        "held_lineage_incomplete": held,
        "filters": {
            "domains": sorted(domains),
            "backends": sorted(backends),
            "require_lineage_ready": require_lineage_ready,
        },
        "constraints": {
            f"{freeze_scope}_evidence_freeze_only": True,
            "one_per_effective_source_identity": True,
        },
        "scenarios": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", action="append", default=[])
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument(
        "--freeze-scope",
        choices=("backend", "domain"),
        required=True,
    )
    parser.add_argument("--allow-lineage-incomplete", action="store_true")
    args = parser.parse_args()
    try:
        report = select_working_set_slice(
            source=json.loads(args.source.read_text(encoding="utf-8")),
            domains=set(args.domain),
            backends=set(args.backend),
            freeze_scope=args.freeze_scope,
            require_lineage_ready=not args.allow_lineage_incomplete,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_scenarios": report["n_scenarios"],
                "n_held_lineage_incomplete": report[
                    "n_held_lineage_incomplete"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
