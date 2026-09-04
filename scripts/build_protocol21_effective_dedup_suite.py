#!/usr/bin/env python3
"""Remove explicitly adjudicated duplicate effective identities from a working set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.build_protocol21_incremental_union import _source_identity_key

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "protocol21-effective-dedup-decisions-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("scenario_id") or ""), str(
        row.get("scenario_signature") or ""
    )


def _repo_path(path: Path, *, repo_root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"path outside repository: {path}") from exc
    return resolved


def build_dedup_suite(
    source: dict[str, Any],
    decisions: dict[str, Any],
    *,
    source_sha256: str,
    decisions_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if source.get("status") != "working_set":
        raise ValueError("source suite must have status=working_set")
    if source.get("leaderboard_eligible") is not False:
        raise ValueError("source suite must not be leaderboard eligible")
    rows = source.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source suite scenarios missing")
    if decisions.get("schema_version") != SCHEMA:
        raise ValueError(f"decision schema_version must be {SCHEMA}")
    raw_decisions = decisions.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("decisions list missing")

    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    groups: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source suite row must be an object")
        identity = _identity(row)
        if not all(identity) or identity in by_identity:
            raise ValueError("source suite row identities must be complete and unique")
        key = _source_identity_key(row)
        if not key:
            raise ValueError(f"effective source identity missing: {identity[0]}")
        by_identity[identity] = row
        groups.setdefault(key, []).append(identity)

    duplicates = {key: ids for key, ids in groups.items() if len(ids) > 1}
    adjudicated: set[str] = set()
    removed: dict[tuple[str, str], dict[str, Any]] = {}
    for decision in raw_decisions:
        if not isinstance(decision, dict):
            raise ValueError("dedup decision must be an object")
        keep = (
            str(decision.get("keep_scenario_id") or ""),
            str(decision.get("keep_scenario_signature") or ""),
        )
        raw_removed = decision.get("remove")
        if not all(keep) or not isinstance(raw_removed, list) or not raw_removed:
            raise ValueError("dedup decision identity incomplete")
        if keep not in by_identity:
            raise ValueError(f"dedup keep identity missing: {keep[0]}")
        key = _source_identity_key(by_identity[keep])
        group = duplicates.get(key)
        if group is None:
            raise ValueError(f"dedup keep identity is not duplicated: {keep[0]}")
        if key in adjudicated:
            raise ValueError(f"duplicate decision for effective identity: {keep[0]}")
        expected_removed = set(group) - {keep}
        declared_removed: set[tuple[str, str]] = set()
        for item in raw_removed:
            if not isinstance(item, dict):
                raise ValueError("removed identity must be an object")
            identity = (
                str(item.get("scenario_id") or ""),
                str(item.get("scenario_signature") or ""),
            )
            if str(item.get("disposition") or "") != "secondary_duplicate":
                raise ValueError("removed disposition must be secondary_duplicate")
            declared_removed.add(identity)
            removed[identity] = {
                "scenario_id": identity[0],
                "scenario_signature": identity[1],
                "disposition": "secondary_duplicate",
                "primary_scenario_id": keep[0],
                "reason_code": "duplicate_canonical_effective_source",
            }
        if declared_removed != expected_removed:
            raise ValueError("dedup decision does not exactly cover duplicate group")
        adjudicated.add(key)

    unlisted = set(duplicates) - adjudicated
    if unlisted:
        raise ValueError(f"unlisted duplicate effective identities: {len(unlisted)}")

    kept_rows = [
        copy.deepcopy(row) for identity, row in by_identity.items() if identity not in removed
    ]
    kept_keys = [_source_identity_key(row) for row in kept_rows]
    if len(kept_keys) != len(set(kept_keys)):
        raise ValueError("deduplicated suite still contains duplicate effective identities")
    output = copy.deepcopy(source)
    output["scenarios"] = sorted(
        kept_rows, key=lambda row: str(row.get("scenario_id") or "")
    )
    output["n_scenarios"] = len(kept_rows)
    output["release_ready"] = False
    output["leaderboard_eligible"] = False
    constraints = dict(output.get("constraints") or {})
    constraints["formal_evaluation_ready"] = False
    constraints["one_per_effective_source_identity"] = True
    output["constraints"] = constraints
    report = {
        "schema_version": "protocol21-effective-dedup-report-v1",
        "status": "working_set_deduplicated_requires_fresh_replay",
        "core_admission": False,
        "source_suite_sha256": source_sha256,
        "decisions_sha256": decisions_sha256,
        "n_input": len(rows),
        "n_output": len(kept_rows),
        "n_removed": len(removed),
        "removed": sorted(removed.values(), key=lambda row: row["scenario_id"]),
    }
    output["effective_identity_dedup"] = copy.deepcopy(report)
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    source_path = _repo_path(args.source_suite, repo_root=REPO_ROOT)
    decisions_path = _repo_path(args.decisions, repo_root=REPO_ROOT)
    output_path = _repo_path(args.output, repo_root=REPO_ROOT)
    report_path = _repo_path(args.report, repo_root=REPO_ROOT)
    output, report = build_dedup_suite(
        _load(source_path),
        _load(decisions_path),
        source_sha256=_sha256(source_path),
        decisions_sha256=_sha256(decisions_path),
    )
    for path, payload in ((output_path, output), (report_path, report)):
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if path.exists() and path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"refusing to overwrite differing artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(encoded, encoding="utf-8")
    print(json.dumps({"status": report["status"], "n_output": report["n_output"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
