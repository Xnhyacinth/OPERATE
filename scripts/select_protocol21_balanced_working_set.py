#!/usr/bin/env python3
"""Select a balanced Protocol-2.1 working set without weakening row gates.

The pipeline may discover more fully qualified rows than the release
distribution contract can carry.  This selector keeps the source-grounded
base suite, admits the maximum number of additional rows that respect the
declared domain/backend caps, and records the remaining qualified rows as
held-out concentration reserves.  It never changes a scenario or turns a
failed row into a passing row.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("source suite must be a JSON object")
    return payload


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _counts(rows: list[dict[str, Any]], field: str) -> Counter[str]:
    return Counter(str(row.get(field) or "") for row in rows)


def _share(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return max(_counts(rows, field).values(), default=0) / len(rows)


def _resolve_base(source: dict[str, Any], source_path: Path) -> set[tuple[str, str]]:
    raw = str((source.get("selection_audit") or {}).get("base_suite") or "")
    if not raw:
        return set()
    base_path = Path(raw)
    if not base_path.is_absolute():
        base_path = REPO_ROOT / base_path
    if not base_path.is_file():
        # A source suite can be assembled outside the repository.  Resolve
        # relative base paths against that suite's parent as a fallback.
        base_path = source_path.parent / raw
    if not base_path.is_file():
        return set()
    return {_identity(row) for row in (_load(base_path).get("scenarios") or [])}


def select(source: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    rows = [row for row in source.get("scenarios") or [] if isinstance(row, dict)]
    constraints = dict(source.get("constraints") or {})
    max_domain = float(constraints.get("max_domain_share", 1.0))
    max_backend = float(constraints.get("max_backend_share", 1.0))
    base_ids = _resolve_base(source, source_path)
    base = [row for row in rows if _identity(row) in base_ids]
    extras = [row for row in rows if _identity(row) not in base_ids]
    if not base:
        # Without an independently identified base, preserve the input order
        # and apply the same fail-closed greedy selection to all rows.
        extras = rows

    selected = list(base)
    selected_ids = {_identity(row) for row in selected}
    eligible_cells = {
        (
            str(row.get("family") or ""),
            str(row.get("difficulty_level") or ""),
        )
        for row in rows
    }

    def admissible(row: dict[str, Any]) -> bool:
        trial = [*selected, row]
        return _share(trial, "domain") <= max_domain and _share(
            trial, "backend_kind"
        ) <= max_backend

    # First preserve a newly discovered family/difficulty cell when possible;
    # this keeps the coverage contract meaningful while the caps remain hard.
    selected_cells = {
        (str(row.get("family") or ""), str(row.get("difficulty_level") or ""))
        for row in selected
    }
    def cell(row: dict[str, Any]) -> tuple[str, str]:
        return (
            str(row.get("family") or ""),
            str(row.get("difficulty_level") or ""),
        )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sorted(extras, key=lambda item: str(item.get("scenario_id") or "")):
        grouped.setdefault(cell(row), []).append(row)
    # Round one admits at most one row from each new family/difficulty cell.
    # This avoids spending the whole concentration budget on the first
    # alphabetically sorted family and dropping an otherwise admissible cell.
    ordered: list[dict[str, Any]] = []
    for key in sorted(grouped):
        ordered.extend(grouped[key][:1])
    for key in sorted(grouped):
        ordered.extend(grouped[key][1:])
    for row in ordered:
        identity = _identity(row)
        if identity in selected_ids:
            continue
        row_cell = cell(row)
        # Prefer new cells but do not violate a hard share cap.  A cell that
        # cannot fit is held with a machine-readable reason below.
        if admissible(row):
            selected.append(row)
            selected_ids.add(identity)
            selected_cells.add(row_cell)

    held = []
    for row in rows:
        if _identity(row) in selected_ids:
            continue
        held.append(
            {
                "scenario_id": row.get("scenario_id"),
                "scenario_signature": row.get("scenario_signature"),
                "domain": row.get("domain"),
                "backend_kind": row.get("backend_kind"),
                "difficulty_level": row.get("difficulty_level"),
                "reason_code": "selection_concentration_guard",
                "quality_gates_preserved": True,
            }
        )

    output = deepcopy(source)
    output["scenarios"] = selected
    output["n_scenarios"] = len(selected)
    output["held_out"] = [*(source.get("held_out") or []), *held]
    audit = dict(source.get("selection_audit") or {})
    # The input audit may describe the immutable base suite.  Preserve that
    # information under explicit names, then make the top-level counts and
    # distributions describe this selector's actual output.  Leaving base
    # counts under ambiguous keys makes a machine-readable source lock lie
    # about the rows that the pipeline will consume.
    if "by_domain" in audit and "base_by_domain" not in audit:
        audit["base_by_domain"] = audit["by_domain"]
    if "by_backend_kind" in audit and "base_by_backend_kind" not in audit:
        audit["base_by_backend_kind"] = audit["by_backend_kind"]
    if "n_source_rows" in audit and "base_n_source_rows" not in audit:
        audit["base_n_source_rows"] = audit["n_source_rows"]
    audit.update(
        {
            "selection_policy": "protocol21_balanced_candidate_v1",
            "input_rows": len(rows),
            "n_source_rows": len(rows),
            "n_base_rows": len(base),
            "n_selected_rows": len(selected),
            "n_held_out_concentration_reserve": len(held),
            "held_out_reason_code": "selection_concentration_guard",
            "eligible_family_difficulty_cells": sorted(eligible_cells),
            "selected_family_difficulty_cells": sorted(selected_cells),
            "by_domain": dict(sorted(_counts(selected, "domain").items())),
            "by_backend_kind": dict(
                sorted(_counts(selected, "backend_kind").items())
            ),
            "by_domain_selected": dict(sorted(_counts(selected, "domain").items())),
            "by_backend_selected": dict(
                sorted(_counts(selected, "backend_kind").items())
            ),
            "max_domain_share_actual": round(_share(selected, "domain"), 9),
            "max_backend_share_actual": round(_share(selected, "backend_kind"), 9),
            "max_domain_share_passed": _share(selected, "domain") <= max_domain,
            "max_backend_share_passed": _share(selected, "backend_kind") <= max_backend,
            "preserve_each_eligible_family_difficulty_cell": eligible_cells.issubset(
                selected_cells
            ),
        }
    )
    output["selection_audit"] = audit
    output["status"] = "working_set"
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = _load(args.source_suite)
    output = select(source, source_path=args.source_suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": output["status"],
                "n_input": output["selection_audit"]["input_rows"],
                "n_selected": output["n_scenarios"],
                "n_held_out": output["selection_audit"][
                    "n_held_out_concentration_reserve"
                ],
                "max_domain_share_passed": output["selection_audit"][
                    "max_domain_share_passed"
                ],
                "max_backend_share_passed": output["selection_audit"][
                    "max_backend_share_passed"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
