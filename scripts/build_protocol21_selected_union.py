#!/usr/bin/env python3
"""Build a non-admitting union from exact Protocol-2.1 selections.

The output is a fresh-replay input, never a release or Core proof.  Every
selected identity is copied from the exact source suite bound by its selection
artifact, and effective/physical source collisions fail closed.
"""

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

from core.protocol21_admission import (  # noqa: E402
    STRICT_ADMISSION_PROFILE,
    SUPPORTED_ADMISSION_PROFILES,
)
from core.source_asset_contract import (  # noqa: E402
    canonical_physical_source_asset_key,
)
from scripts.build_protocol21_incremental_union import (  # noqa: E402
    _source_identity_key,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("scenario_id") or ""), str(row.get("scenario_signature") or "")


def _physical_key(row: dict[str, Any]) -> str:
    ledger = row.get("case_ledger") or {}
    lock = row.get("_physical_source_lock") or ledger.get("physical_source_lock")
    if not isinstance(lock, (dict, list, str)) or not lock:
        raise ValueError(f"physical source lock missing: {_identity(row)[0]}")
    return canonical_physical_source_asset_key(lock)


def _selected_source_rows(
    source_suite: dict[str, Any],
    selection: dict[str, Any],
    *,
    source_sha256: str,
) -> list[dict[str, Any]]:
    if source_suite.get("status") != "working_set":
        raise ValueError("source suite must have status=working_set")
    if selection.get("status") != "protocol21_core_candidate":
        raise ValueError("selection must be a Protocol-2.1 Core candidate")
    binding = ((selection.get("input_bindings") or {}).get("source_suite") or {}).get(
        "sha256"
    ) or ""
    if binding != source_sha256:
        raise ValueError("selection source-suite hash mismatch")
    source_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for row in source_suite.get("scenarios") or []:
        identity = _identity(row)
        if not all(identity) or identity in source_by_identity:
            raise ValueError(f"invalid or duplicate source identity: {identity}")
        source_by_identity[identity] = row
    selected = selection.get("scenarios") or []
    if int(selection.get("n_selected", -1)) != len(selected):
        raise ValueError("selection count mismatch")
    rows = []
    for selected_row in selected:
        if not isinstance(selected_row, dict):
            raise ValueError("selection scenario row must be an object")
        if (
            selected_row.get("status") != "core_locked"
            or selected_row.get("core_disposition") != "core_locked"
        ):
            raise ValueError(
                "selection row lacks core_locked terminal proof: "
                f"{selected_row.get('scenario_id') or '<missing>'}"
            )
        identity = _identity(selected_row)
        source_row = source_by_identity.get(identity)
        if source_row is None:
            raise ValueError(f"selected identity missing from source suite: {identity}")
        rows.append(copy.deepcopy(source_row))
    return rows


def build_selected_union(
    inputs: list[tuple[dict[str, Any], dict[str, Any], str]],
    *,
    admission_profile: str = STRICT_ADMISSION_PROFILE,
) -> dict[str, Any]:
    if admission_profile not in SUPPORTED_ADMISSION_PROFILES:
        raise ValueError(f"unsupported admission profile: {admission_profile}")
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    effective: set[str] = set()
    physical: set[str] = set()
    for source_suite, selection, source_sha256 in inputs:
        for row in _selected_source_rows(source_suite, selection, source_sha256=source_sha256):
            identity = _identity(row)
            effective_key = _source_identity_key(row)
            physical_key = _physical_key(row)
            if identity in identities:
                raise ValueError(f"duplicate scenario identity: {identity}")
            if not effective_key or effective_key in effective:
                raise ValueError(f"duplicate effective source identity: {identity[0]}")
            identities.add(identity)
            effective.add(effective_key)
            physical.add(physical_key)
            row["status"] = "pending_fresh_union_replay"
            rows.append(row)
    rows.sort(key=lambda row: str(row["scenario_id"]))
    return {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "working_set",
        "selection_policy": "selected_union_fresh_replay_required_v1",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(rows),
        "n_physical_sources": len(physical),
        "constraints": {
            "core_admission_profile": admission_profile,
            "candidate_only": True,
            "formal_evaluation_ready": False,
            "model_outcomes_used_for_filtering": False,
            "one_per_effective_source_identity": True,
            "physical_sources_are_inference_clusters": True,
            "fresh_protocol21_replay_required": True,
        },
        "scenarios": rows,
    }


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("SOURCE_SUITE", "SELECTION"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--admission-profile",
        choices=sorted(SUPPORTED_ADMISSION_PROFILES),
        default=STRICT_ADMISSION_PROFILE,
    )
    args = parser.parse_args()
    inputs = []
    bindings = []
    for source_raw, selection_raw in args.pair:
        source = Path(source_raw).resolve()
        selection = Path(selection_raw).resolve()
        inputs.append((_read(source), _read(selection), _sha256(source)))
        bindings.append(
            {
                "source_suite": {"path": source_raw, "sha256": _sha256(source)},
                "selection": {"path": selection_raw, "sha256": _sha256(selection)},
            }
        )
    result = build_selected_union(
        inputs,
        admission_profile=args.admission_profile,
    )
    result["input_bindings"] = bindings
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "n_scenarios": result["n_scenarios"]}))


if __name__ == "__main__":
    main()
