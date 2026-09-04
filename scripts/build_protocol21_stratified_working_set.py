#!/usr/bin/env python3
"""Build the largest source-locked Protocol-2.1 candidate under share caps."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.merge_protocol21_evidence_freeze_selections import (  # noqa: E402
    merge_evidence_freeze_selections,
)

STRATIFIED_WORKING_SET_POLICY = "source_locked_stratified_maximal_v1"
DEFAULT_MAX_DOMAIN_SHARE = 0.40
DEFAULT_MAX_BACKEND_SHARE = 0.25


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _cell(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("family") or ""),
        str(row.get("difficulty_level") or ""),
    )


def _physical_lock(row: Mapping[str, Any]) -> str:
    ledger = row.get("case_ledger")
    ledger = ledger if isinstance(ledger, Mapping) else {}
    value = next(
        (
            candidate
            for candidate in (
                row.get("physical_source_key"),
                ledger.get("physical_source_key"),
                ledger.get("physical_source_lock"),
            )
            if candidate not in (None, "", {}, [])
        ),
        "",
    )
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _validate_share(value: float, *, name: str) -> float:
    if not 0 < value <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return value


def _cap(target: int, share: float) -> int:
    return int(math.floor(target * share + 1e-12))


def _candidate_key(
    row: Mapping[str, Any],
    *,
    domain_counts: Counter[str],
    backend_counts: Counter[str],
    physical_counts: Counter[str],
) -> tuple[Any, ...]:
    domain = str(row.get("domain") or "")
    backend = str(row.get("backend_kind") or "")
    return (
        backend_counts[backend],
        domain_counts[domain],
        physical_counts[_physical_lock(row)],
        str(row.get("source_denominator_key") or ""),
        *_identity(row),
    )


def _select_for_target(
    rows: Sequence[dict[str, Any]],
    *,
    target: int,
    max_domain_share: float,
    max_backend_share: float,
) -> list[dict[str, Any]] | None:
    domain_cap = _cap(target, max_domain_share)
    backend_cap = _cap(target, max_backend_share)
    if domain_cap < 1 or backend_cap < 1:
        return None
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[_cell(row)].append(row)
    selected: list[dict[str, Any]] = []
    selected_identities: set[tuple[str, str]] = set()
    domain_counts: Counter[str] = Counter()
    backend_counts: Counter[str] = Counter()
    physical_counts: Counter[str] = Counter()

    def feasible(row: Mapping[str, Any]) -> bool:
        domain = str(row.get("domain") or "")
        backend = str(row.get("backend_kind") or "")
        return (
            domain_counts[domain] < domain_cap
            and backend_counts[backend] < backend_cap
        )

    def add(row: dict[str, Any]) -> None:
        selected.append(row)
        selected_identities.add(_identity(row))
        domain_counts[str(row.get("domain") or "")] += 1
        backend_counts[str(row.get("backend_kind") or "")] += 1
        physical_counts[_physical_lock(row)] += 1

    for _cell_key, candidates in sorted(
        cells.items(), key=lambda item: (len(item[1]), item[0])
    ):
        choices = [candidate for candidate in candidates if feasible(candidate)]
        if not choices:
            return None
        add(
            min(
                choices,
                key=lambda item: _candidate_key(
                    item,
                    domain_counts=domain_counts,
                    backend_counts=backend_counts,
                    physical_counts=physical_counts,
                ),
            )
        )

    while len(selected) < target:
        choices = [
            row
            for row in rows
            if _identity(row) not in selected_identities and feasible(row)
        ]
        if not choices:
            return None
        add(
            min(
                choices,
                key=lambda item: _candidate_key(
                    item,
                    domain_counts=domain_counts,
                    backend_counts=backend_counts,
                    physical_counts=physical_counts,
                ),
            )
        )
    return sorted(selected, key=_identity)


def build_stratified_working_set(
    *,
    source_paths: list[Path],
    max_domain_share: float = DEFAULT_MAX_DOMAIN_SHARE,
    max_backend_share: float = DEFAULT_MAX_BACKEND_SHARE,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Select the largest balanced subset without discarding the evidence pool."""
    max_domain_share = _validate_share(
        max_domain_share, name="max_domain_share"
    )
    max_backend_share = _validate_share(
        max_backend_share, name="max_backend_share"
    )
    merged = merge_evidence_freeze_selections(
        source_paths=source_paths,
        freeze_scope="backend",
        repo_root=repo_root,
    )
    rows = [dict(row) for row in merged["scenarios"]]
    cells = {_cell(row) for row in rows}
    selected: list[dict[str, Any]] | None = None
    for target in range(len(rows), len(cells) - 1, -1):
        selected = _select_for_target(
            rows,
            target=target,
            max_domain_share=max_domain_share,
            max_backend_share=max_backend_share,
        )
        if selected is not None:
            break
    if selected is None:
        raise ValueError(
            "no feasible stratified working set preserves every family/difficulty cell"
        )
    selected_ids = {_identity(row) for row in selected}
    domain_counts = Counter(str(row.get("domain") or "") for row in selected)
    backend_counts = Counter(
        str(row.get("backend_kind") or "") for row in selected
    )
    denominator = len(selected)
    max_domain_actual = max(domain_counts.values(), default=0) / denominator
    max_backend_actual = max(backend_counts.values(), default=0) / denominator
    selected_cells = {_cell(row) for row in selected}
    return {
        "schema_version": "2.1",
        "status": "working_set",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": denominator,
        "selection_policy": STRATIFIED_WORKING_SET_POLICY,
        "constraints": {
            "candidate_evidence_merge_only": True,
            "one_per_effective_source_identity": True,
            "preserve_each_eligible_family_difficulty_cell": True,
            "max_domain_share": max_domain_share,
            "max_backend_share": max_backend_share,
        },
        "source_artifacts": merged["source_artifacts"],
        "selection_audit": {
            "n_source_rows": len(rows),
            "n_selected_rows": denominator,
            "n_held_out_distribution_reserve": len(rows) - denominator,
            "eligible_family_difficulty_cells": [list(cell) for cell in sorted(cells)],
            "selected_family_difficulty_cells": [
                list(cell) for cell in sorted(selected_cells)
            ],
            "preserve_each_eligible_family_difficulty_cell": (
                selected_cells == cells
            ),
            "max_domain_share_actual": round(max_domain_actual, 9),
            "max_backend_share_actual": round(max_backend_actual, 9),
            "max_domain_share_passed": max_domain_actual <= max_domain_share,
            "max_backend_share_passed": max_backend_actual <= max_backend_share,
            "by_domain": dict(sorted(domain_counts.items())),
            "by_backend_kind": dict(sorted(backend_counts.items())),
        },
        "held_out": [
            {
                "scenario_id": scenario_id,
                "scenario_signature": signature,
                "reason_code": "distribution_stratified_reserve",
            }
            for scenario_id, signature in sorted(
                _identity(row) for row in rows if _identity(row) not in selected_ids
            )
        ],
        "scenarios": selected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-domain-share", type=float, default=DEFAULT_MAX_DOMAIN_SHARE
    )
    parser.add_argument(
        "--max-backend-share", type=float, default=DEFAULT_MAX_BACKEND_SHARE
    )
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        report = build_stratified_working_set(
            source_paths=args.source,
            max_domain_share=args.max_domain_share,
            max_backend_share=args.max_backend_share,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
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
                "selection_audit": report["selection_audit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
