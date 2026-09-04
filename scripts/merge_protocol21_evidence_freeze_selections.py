#!/usr/bin/env python3
"""Merge independently replayed non-release Protocol-2.1 selections."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.working_set_contract import (  # noqa: E402
    validate_protocol21_row_lineage,
)

_ALLOWED_SOURCE_STATUSES = {"working_set", "protocol21_core_candidate"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"source artifact must be inside repository: {path}"
        ) from exc


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _load_source(path: Path, *, repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    status = str(payload.get("status") or "")
    if status not in _ALLOWED_SOURCE_STATUSES:
        raise ValueError(f"unsupported source selection status: {status}")
    if payload.get("leaderboard_eligible") is not False:
        raise ValueError("source selection must not be leaderboard eligible")
    if payload.get("formal_evaluation_ready") is True or payload.get(
        "release_ready"
    ) is True:
        raise ValueError("source selection must remain non-release")
    constraints = payload.get("constraints")
    if not isinstance(constraints, dict) or constraints.get(
        "one_per_effective_source_identity"
    ) is not True:
        raise ValueError(
            "source selection must require one_per_effective_source_identity"
        )
    rows = payload.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source selection must contain scenarios")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("source selection scenarios must be mappings")
    return payload, {
        "path": _repo_relative(resolved, repo_root=repo_root),
        "sha256": _sha256(resolved),
        "status": status,
        "n_scenarios": len(rows),
    }


def merge_evidence_freeze_selections(
    *,
    source_paths: list[Path],
    freeze_scope: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build a lineage-safe working set without granting release status."""
    if freeze_scope not in {"backend", "domain"}:
        raise ValueError("freeze_scope must be backend or domain")
    if not source_paths:
        raise ValueError("at least one source selection is required")

    loaded = [
        _load_source(path, repo_root=repo_root) for path in source_paths
    ]
    loaded.sort(key=lambda item: str(item[1]["path"]))
    rows = [
        deepcopy(row)
        for payload, _source in loaded
        for row in payload["scenarios"]
    ]
    identities = [_identity(row) for row in rows]
    if any(not scenario_id or not signature for scenario_id, signature in identities):
        raise ValueError("scenario identities must be complete")
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate scenario identity across source selections")
    lineage_errors = {
        scenario_id: errors
        for row, (scenario_id, _signature) in zip(rows, identities, strict=True)
        if (errors := validate_protocol21_row_lineage(row))
    }
    if lineage_errors:
        raise ValueError(f"scenario lineage incomplete: {lineage_errors}")
    effective_keys = [str(row["source_denominator_key"]) for row in rows]
    if len(set(effective_keys)) != len(effective_keys):
        raise ValueError(
            "duplicate effective source identity across source selections"
        )
    rows.sort(key=_identity)

    scope_constraint = f"{freeze_scope}_evidence_freeze_only"
    return {
        "schema_version": "2.1",
        "status": "working_set",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(rows),
        "constraints": {
            scope_constraint: True,
            "merged_candidate_sources": True,
            "one_per_effective_source_identity": True,
        },
        "source_artifacts": [source for _payload, source in loaded],
        "scenarios": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--freeze-scope",
        choices=("backend", "domain"),
        required=True,
    )
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError(f"output already exists: {args.output}")
        merged = merge_evidence_freeze_selections(
            source_paths=args.source,
            freeze_scope=args.freeze_scope,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(merged, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": merged["status"],
                "n_scenarios": merged["n_scenarios"],
                "source_artifacts": merged["source_artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
