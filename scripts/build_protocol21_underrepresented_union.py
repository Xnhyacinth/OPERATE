#!/usr/bin/env python3
"""Merge a selected Core candidate with lineage-clean expansion survivors."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import required_semantics  # noqa: E402
from core.working_set_contract import validate_protocol21_row_lineage  # noqa: E402
from scripts.build_protocol21_incremental_union import (  # noqa: E402
    _identity,
    _source_identity_key,
)
from scripts.build_protocol21_selected_union import (  # noqa: E402
    _physical_key,
    _selected_source_rows,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def build_underrepresented_union(
    *,
    base_source: dict[str, Any],
    base_selection: dict[str, Any],
    base_source_sha256: str,
    additions: list[dict[str, Any]],
    implementation_tree_sha256: str,
) -> dict[str, Any]:
    base_rows = _selected_source_rows(
        base_source,
        base_selection,
        source_sha256=base_source_sha256,
    )
    rows = [copy.deepcopy(row) for row in base_rows]
    for suite in additions:
        if suite.get("status") not in {
            "working_set",
            "candidate_prefilter_survivors",
        }:
            raise ValueError("addition source suite has unsupported candidate status")
        if suite.get("leaderboard_eligible") is not False:
            raise ValueError("addition source suite must be candidate-only")
        if suite.get("release_ready") is not False:
            raise ValueError("addition source suite must not be release-ready")
        rows.extend(copy.deepcopy(row) for row in suite.get("scenarios") or [])

    identities: set[tuple[str, str]] = set()
    effective_sources: set[str] = set()
    physical_sources: set[str] = set()
    for row in rows:
        blockers = validate_protocol21_row_lineage(row)
        if blockers:
            raise ValueError(
                f"lineage invalid for {row.get('scenario_id')}: {','.join(blockers)}"
            )
        identity = _identity(row)
        if not all(identity) or identity in identities:
            raise ValueError(f"duplicate scenario identity: {identity}")
        effective = _source_identity_key(row)
        if not effective or effective in effective_sources:
            raise ValueError(f"duplicate effective source identity: {identity[0]}")
        identities.add(identity)
        effective_sources.add(effective)
        physical_sources.add(_physical_key(row))
        row["status"] = "pending_fresh_union_replay"

    rows.sort(key=lambda row: str(row["scenario_id"]))
    return {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "working_set",
        "selection_policy": "v57_selected_plus_underrepresented_survivors_v1",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(rows),
        "n_base_selected": len(base_rows),
        "n_additions": len(rows) - len(base_rows),
        "n_effective_sources": len(effective_sources),
        "n_physical_sources": len(physical_sources),
        "implementation_tree_sha256": implementation_tree_sha256,
        "evaluation_semantics": required_semantics(),
        "constraints": {
            "core_admission_profile": "quality_core_v2",
            "candidate_only": True,
            "formal_evaluation_ready": False,
            "model_outcomes_used_for_filtering": False,
            "one_per_effective_source_identity": True,
            "physical_sources_are_inference_clusters": True,
            "fresh_protocol21_replay_required": True,
        },
        "scenarios": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--base-selection", type=Path, required=True)
    parser.add_argument("--addition", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_source = args.base_source.resolve()
    base_selection = args.base_selection.resolve()
    additions = [path.resolve() for path in args.addition]
    result = build_underrepresented_union(
        base_source=_read(base_source),
        base_selection=_read(base_selection),
        base_source_sha256=_sha256(base_source),
        additions=[_read(path) for path in additions],
        implementation_tree_sha256=implementation_identity(REPO_ROOT)[
            "implementation_tree_sha256"
        ],
    )
    result["input_bindings"] = {
        "base_source": {"path": str(args.base_source), "sha256": _sha256(base_source)},
        "base_selection": {
            "path": str(args.base_selection),
            "sha256": _sha256(base_selection),
        },
        "additions": [
            {"path": str(raw), "sha256": _sha256(path)}
            for raw, path in zip(args.addition, additions, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_scenarios": result["n_scenarios"],
                "n_additions": result["n_additions"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
