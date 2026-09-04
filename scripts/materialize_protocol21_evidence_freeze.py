#!/usr/bin/env python3
"""Promote passing rows into a stable Protocol-2.1 evidence-freeze suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.working_set_contract import (  # noqa: E402
    validate_protocol21_row_lineage,
)


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
        raise ValueError(f"output directory must be inside repository: {path}") from exc


def materialize_evidence_freeze(
    *,
    source_path: Path,
    output_dir: Path,
    repo_root: Path = REPO_ROOT,
    freeze_scope: str,
) -> dict[str, Any]:
    """Copy selected YAMLs and emit a stable, non-release working set."""
    if freeze_scope not in {"backend", "domain"}:
        raise ValueError("freeze_scope must be backend or domain")
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    rows = payload.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source selection must contain scenarios")
    if payload.get("leaderboard_eligible") is not False:
        raise ValueError("source selection must not be leaderboard eligible")
    if (output_dir / "working_set.json").exists():
        raise ValueError(f"output working set already exists: {output_dir}")
    scenario_dir = output_dir / "scenarios"
    if scenario_dir.exists() and any(scenario_dir.iterdir()):
        raise ValueError(f"output scenario directory is not empty: {scenario_dir}")

    names = [Path(str(row.get("path") or "")).name for row in rows]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("duplicate scenario filename")
    identities = [
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in rows
    ]
    if (
        any(not scenario_id or not signature for scenario_id, signature in identities)
        or len(set(identities)) != len(rows)
    ):
        raise ValueError("scenario identities must be complete and unique")
    lineage_errors = {
        scenario_id: errors
        for row, (scenario_id, _signature) in zip(rows, identities, strict=True)
        if (errors := validate_protocol21_row_lineage(row))
    }
    if lineage_errors:
        raise ValueError(f"scenario lineage incomplete: {lineage_errors}")

    scenario_dir.mkdir(parents=True, exist_ok=True)
    stable_rows: list[dict[str, Any]] = []
    scenario_files: list[dict[str, Any]] = []
    for row, name in zip(rows, names, strict=True):
        source_scenario = Path(str(row["path"]))
        if not source_scenario.is_absolute():
            source_scenario = repo_root / source_scenario
        if not source_scenario.is_file():
            raise FileNotFoundError(source_scenario)
        destination = scenario_dir / name
        shutil.copy2(source_scenario, destination)
        stable_row = dict(row)
        stable_row["path"] = _repo_relative(destination, repo_root=repo_root)
        stable_rows.append(stable_row)
        scenario_files.append(
            {
                "path": stable_row["path"],
                "sha256": _sha256(destination),
            }
        )

    scope_constraint = f"{freeze_scope}_evidence_freeze_only"
    working_set = {
        "schema_version": "2.1",
        "status": "working_set",
        "leaderboard_eligible": False,
        "blockers": [],
        "n_scenarios": len(stable_rows),
        "constraints": {
            scope_constraint: True,
            "one_per_effective_source_identity": True,
        },
        "scenarios": stable_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    working_set_path = output_dir / "working_set.json"
    working_set_path.write_text(
        json.dumps(working_set, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "2.1",
        "status": "materialized_for_evidence_replay",
        "freeze_scope": freeze_scope,
        "leaderboard_eligible": False,
        "release_ready": False,
        "source_artifact_sha256": _sha256(source_path),
        "working_set_sha256": _sha256(working_set_path),
        "n_scenarios": len(stable_rows),
        "scenario_files": scenario_files,
    }
    (output_dir / "materialization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--freeze-scope",
        choices=("backend", "domain"),
        required=True,
    )
    args = parser.parse_args()
    try:
        report = materialize_evidence_freeze(
            source_path=args.source,
            output_dir=args.output_dir,
            freeze_scope=args.freeze_scope,
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
