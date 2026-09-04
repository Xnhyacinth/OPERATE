#!/usr/bin/env python3
"""Merge previously admitted rows into a current Protocol-2.1 candidate set.

The old admission artifact is used only as an allow-list.  Rows are copied
from the exact source suite bound by that artifact and must pass the current
pipeline again before they become Core.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.source_asset_contract import (  # noqa: E402
    canonical_physical_source_asset_key,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _serialized_identities(
    identities: set[tuple[str, str]],
) -> list[dict[str, str]]:
    return [
        {"scenario_id": scenario_id, "scenario_signature": signature}
        for scenario_id, signature in sorted(identities)
    ]


def _partition_identity_set(value: Any, *, label: str) -> set[tuple[str, str]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"candidate import {label} identities are invalid")
    identities = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in value
    }
    if len(identities) != len(value) or any(
        not all(identity) for identity in identities
    ):
        raise ValueError(f"candidate import {label} identities are invalid")
    if value != _serialized_identities(identities):
        raise ValueError(f"candidate import {label} identities are not canonical")
    return identities


def validate_candidate_import_partition(
    suite: dict[str, Any],
    *,
    allow_unpartitioned_base: bool = False,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    rows = suite.get("scenarios")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("candidate import suite scenarios are invalid")
    row_identities = {_identity(row) for row in rows}
    if len(row_identities) != len(rows) or any(
        not all(identity) for identity in row_identities
    ):
        raise ValueError("candidate import suite identities are invalid")
    partition = suite.get("candidate_import_partition")
    if partition is None and allow_unpartitioned_base:
        if any(row.get("historical_admission") is not None for row in rows):
            raise ValueError("unpartitioned base has candidate import markers")
        return sorted(row_identities), []
    if not isinstance(partition, dict) or set(partition) != {
        "schema_version",
        "status",
        "n_base",
        "n_imported",
        "base_identities",
        "imported_identities",
    }:
        raise ValueError("candidate import partition is missing or invalid")
    base = _partition_identity_set(partition["base_identities"], label="base")
    imported = _partition_identity_set(
        partition["imported_identities"], label="imported"
    )
    marker_identities = {
        _identity(row)
        for row in rows
        if (row.get("historical_admission") or {}).get("status")
        == "previously_core_locked_requires_current_replay"
    }
    invalid_markers = [
        row
        for row in rows
        if row.get("historical_admission") is not None
        and (row.get("historical_admission") or {}).get("status")
        != "previously_core_locked_requires_current_replay"
    ]
    if invalid_markers or marker_identities != imported:
        raise ValueError("candidate import marker mismatch")
    if (
        base.intersection(imported)
        or base | imported != row_identities
        or partition.get("schema_version") != "operate-candidate-import-partition-v1"
        or partition.get("status") != "complete"
        or partition.get("n_base") != len(base)
        or partition.get("n_imported") != len(imported)
    ):
        raise ValueError("candidate import partition accounting mismatch")
    return sorted(base), sorted(imported)


def _source_key(row: dict[str, Any]) -> str:
    return str(row.get("source_denominator_key") or "")


def _normalized_source_value(value: Any, *, field: str = "") -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalized_source_value(item, field=str(key))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_normalized_source_value(item, field=field) for item in value]
    if isinstance(value, str) and ("file" in field.lower() or "path" in field.lower()):
        return Path(value).name
    return value


def _normalized_physical_lock(lock: dict[str, Any]) -> dict[str, Any]:
    """Preserve the complete structured lock while normalizing asset paths."""
    normalized = _normalized_source_value(lock)
    assets = normalized.get("required_source_assets") or []
    if isinstance(assets, list):
        normalized["required_source_assets"] = sorted(
            assets,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    return normalized


def _source_identity_key(row: dict[str, Any]) -> str:
    raw_key = _source_key(row)
    if not raw_key:
        return ""
    try:
        parsed_key = json.loads(raw_key)
    except json.JSONDecodeError:
        parsed_key = raw_key
    ledger = row.get("case_ledger") or {}
    lock = row.get("_physical_source_lock") or ledger.get("physical_source_lock") or {}
    if not isinstance(lock, dict):
        lock = {}
    identity = {
        "source": _normalized_source_value(parsed_key),
        "physical_source_lock": _normalized_physical_lock(lock),
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _physical_source_key(row: dict[str, Any]) -> str:
    ledger = row.get("case_ledger") or {}
    lock = row.get("_physical_source_lock") or ledger.get("physical_source_lock")
    if not isinstance(lock, (dict, list, str)) or not lock:
        raise ValueError(f"physical source lock missing: {_identity(row)[0]}")
    return canonical_physical_source_asset_key(lock)


def _physical_source_lock(row: dict[str, Any]) -> dict[str, Any]:
    ledger = row.get("case_ledger") or {}
    lock = row.get("_physical_source_lock") or ledger.get("physical_source_lock")
    if not isinstance(lock, dict):
        raise ValueError(f"physical source lock missing: {_identity(row)[0]}")
    return lock


def _repo_file(raw_path: Any, *, repo_root: Path, label: str) -> Path:
    path = Path(str(raw_path or ""))
    if not path.is_absolute():
        path = repo_root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} outside repository: {raw_path}") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} missing: {raw_path}")
    return resolved


def _routing_source_edge_weight_type(
    row: dict[str, Any], *, repo_root: Path
) -> str | None:
    if not str(row.get("backend_kind") or "").startswith("pyvrp_"):
        return None
    assets = _physical_source_lock(row).get("required_source_assets") or []
    if not isinstance(assets, list) or len(assets) != 1 or not isinstance(assets[0], dict):
        raise ValueError(f"routing source asset binding invalid: {_identity(row)[0]}")
    asset = assets[0]
    path = _repo_file(
        asset.get("declared_path"), repo_root=repo_root, label="routing source asset"
    )
    expected_sha256 = str(asset.get("sha256") or "").removeprefix("sha256:")
    if expected_sha256 and _sha256(path) != expected_sha256:
        raise ValueError(f"routing source asset hash mismatch: {_identity(row)[0]}")
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = raw_line.partition(":")
        if separator and key.strip().upper() == "EDGE_WEIGHT_TYPE":
            return value.strip().upper()
        if raw_line.strip().upper().endswith("_SECTION"):
            break
    return None


def _routing_executable_key(
    row: dict[str, Any], *, repo_root: Path
) -> tuple[str, str] | None:
    backend_kind = str(row.get("backend_kind") or "")
    if not backend_kind.startswith("pyvrp_"):
        return None
    path = _repo_file(row.get("path"), repo_root=repo_root, label="scenario path")
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"scenario YAML must be a mapping: {_identity(row)[0]}")
    network = (body.get("backend_config") or {}).get("network")
    if not isinstance(network, dict) or not network:
        raise ValueError(f"routing executable network missing: {_identity(row)[0]}")
    encoded = json.dumps(network, sort_keys=True, separators=(",", ":")).encode()
    return backend_kind, hashlib.sha256(encoded).hexdigest()


def _canonicalize_scenario_path(row: dict[str, Any], repo_root: Path) -> None:
    """Make scenario paths portable while rejecting paths outside the repo."""
    raw_path = str(row.get("path") or "")
    if not raw_path:
        return
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"scenario path outside repository: {row.get('scenario_id')}: {raw_path}"
        ) from exc
    row["path"] = relative.as_posix()


def _unique_index(
    rows: list[dict[str, Any]],
    *,
    key,
    label: str,
) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for row in rows:
        value = key(row)
        if not value or (isinstance(value, tuple) and not all(value)):
            raise ValueError(f"{label} missing")
        if value in result:
            raise ValueError(f"duplicate {label}: {value}")
        result[value] = row
    return result


def build_incremental_union(
    *,
    base_working_set: dict[str, Any],
    addition_source_suite: dict[str, Any],
    admission_selection: dict[str, Any],
    addition_source_sha256: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Return a staging union whose imported rows require current re-audit."""
    if base_working_set.get("status") != "working_set":
        raise ValueError("base working set must have status=working_set")
    if base_working_set.get("leaderboard_eligible") is not False:
        raise ValueError("base working set must not be leaderboard eligible")
    if addition_source_suite.get("status") != "working_set":
        raise ValueError("addition source suite must have status=working_set")
    if admission_selection.get("schema_version") != "2.1":
        raise ValueError("admission selection schema_version must be 2.1")
    if admission_selection.get("status") != "protocol21_core_candidate":
        raise ValueError("admission selection status must be protocol21_core_candidate")
    if admission_selection.get("selection_policy") != "quality_maximal_v1":
        raise ValueError("admission selection policy must be quality_maximal_v1")

    bound_sha = str(
        (
            (admission_selection.get("input_bindings") or {}).get("source_suite") or {}
        ).get("sha256")
        or ""
    )
    if bound_sha != addition_source_sha256:
        raise ValueError(
            "admission source-suite hash mismatch: "
            f"expected {bound_sha or '<missing>'}, got {addition_source_sha256}"
        )

    base_identity_rows, prior_imported_rows = validate_candidate_import_partition(
        base_working_set,
        allow_unpartitioned_base=True,
    )
    base_partition = set(base_identity_rows)
    prior_imported = set(prior_imported_rows)
    base_rows = [copy.deepcopy(row) for row in base_working_set.get("scenarios") or []]
    source_rows = [
        copy.deepcopy(row) for row in addition_source_suite.get("scenarios") or []
    ]
    for row in [*base_rows, *source_rows]:
        _canonicalize_scenario_path(row, repo_root)
    selected_rows = admission_selection.get("scenarios") or []
    for selected in selected_rows:
        if (
            selected.get("status") != "core_locked"
            or selected.get("core_disposition") != "core_locked"
        ):
            raise ValueError(
                "admission selection row is not core_locked: "
                f"{selected.get('scenario_id') or '<missing>'}"
            )
    if admission_selection.get("n_selected") != len(selected_rows):
        raise ValueError("admission selection n_selected does not match scenarios")
    source_by_identity = _unique_index(
        source_rows, key=_identity, label="source-suite scenario identity"
    )
    _unique_index(base_rows, key=_identity, label="base scenario identity")
    base_by_source: dict[str, list[dict[str, Any]]] = {}
    base_by_routing_executable: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in base_rows:
        source_identity = _source_identity_key(row)
        if not source_identity:
            raise ValueError("base effective source identity missing")
        if source_identity in base_by_source:
            raise ValueError(
                "base working set contains duplicate effective source identity: "
                f"{source_identity}"
            )
        base_by_source[source_identity] = [row]
        routing_key = _routing_executable_key(row, repo_root=repo_root)
        if routing_key is not None:
            if routing_key in base_by_routing_executable:
                raise ValueError(
                    "base working set contains duplicate executable routing network: "
                    f"{routing_key}"
                )
            base_by_routing_executable[routing_key] = [row]

    imported: list[dict[str, Any]] = []
    already_present: list[str] = []
    secondary: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    base_identities = {_identity(row) for row in base_rows}
    for selected in selected_rows:
        identity = _identity(selected)
        source_row = source_by_identity.get(identity)
        if source_row is None:
            raise ValueError(
                "selected identity missing from source suite: "
                f"{identity[0]}@{identity[1]}"
            )
        if identity in base_identities:
            already_present.append(identity[0])
            continue
        edge_weight_type = _routing_source_edge_weight_type(
            source_row, repo_root=repo_root
        )
        if edge_weight_type == "EXACT_2D":
            excluded.append(
                {
                    "scenario_id": identity[0],
                    "scenario_signature": identity[1],
                    "reason_code": "unsupported_edge_weight_semantics",
                    "detail": "EDGE_WEIGHT_TYPE=EXACT_2D",
                }
            )
            continue
        routing_key = _routing_executable_key(source_row, repo_root=repo_root)
        routing_conflicts = (
            base_by_routing_executable.get(routing_key) if routing_key else None
        ) or []
        if routing_conflicts:
            excluded.append(
                {
                    "scenario_id": identity[0],
                    "scenario_signature": identity[1],
                    "primary_scenario_ids": ",".join(
                        sorted(
                            str(row.get("scenario_id")) for row in routing_conflicts
                        )
                    ),
                    "reason_code": (
                        "executable_source_duplicate_after_consumed_window"
                    ),
                }
            )
            continue
        source_key = _source_key(source_row)
        if not source_key:
            raise ValueError(f"effective source identity missing: {identity[0]}")
        source_identity = _source_identity_key(source_row)
        conflicts = base_by_source.get(source_identity) or []
        if conflicts:
            secondary.append(
                {
                    "scenario_id": identity[0],
                    "source_denominator_key": source_key,
                    "primary_scenario_ids": ",".join(
                        sorted(str(row.get("scenario_id")) for row in conflicts)
                    ),
                    "reason_code": "secondary_duplicate_effective_source",
                }
            )
            continue
        source_row["status"] = "pending_current_protocol21_readmission"
        source_row["historical_admission"] = {
            "status": "previously_core_locked_requires_current_replay",
            "selection_artifact_implementation_tree_sha256": admission_selection.get(
                "implementation_tree_sha256"
            ),
        }
        imported.append(source_row)
        base_rows.append(source_row)
        base_by_source.setdefault(source_identity, []).append(source_row)
        if routing_key is not None:
            base_by_routing_executable.setdefault(routing_key, []).append(source_row)
        base_identities.add(identity)

    effective_sources = {_source_identity_key(row) for row in base_rows}
    physical_sources = {_physical_source_key(row) for row in base_rows}
    result = copy.deepcopy(base_working_set)
    constraints = dict(result.get("constraints") or {})
    constraints.pop("max_domain_share", None)
    constraints.pop("max_backend_share", None)
    constraints.update(
        {
            "formal_evaluation_ready": False,
            "model_outcomes_used_for_filtering": False,
            "one_per_effective_source_identity": True,
            "quality_maximal_selection": True,
        }
    )
    result.update(
        {
            "status": "working_set",
            "leaderboard_eligible": False,
            "release_ready": False,
            "n_scenarios": len(base_rows),
            "n_effective_sources": len(effective_sources),
            "n_physical_sources": len(physical_sources),
            "constraints": constraints,
            "scenarios": sorted(base_rows, key=lambda row: str(row.get("scenario_id"))),
            "incremental_import": {
                "policy": "historical_full_gate_allowlist_current_replay_required_v1",
                "n_selected_in_admission": len(selected_rows),
                "n_already_present": len(already_present),
                "n_added": len(imported),
                "n_secondary": len(secondary),
                "n_excluded": len(excluded),
                "added_scenario_ids": sorted(
                    str(row.get("scenario_id")) for row in imported
                ),
                "secondary": secondary,
                "excluded": excluded,
            },
            "candidate_import_partition": {
                "schema_version": "operate-candidate-import-partition-v1",
                "status": "complete",
                "n_base": len(base_partition),
                "n_imported": len(prior_imported) + len(imported),
                "base_identities": _serialized_identities(base_partition),
                "imported_identities": _serialized_identities(
                    prior_imported | {_identity(row) for row in imported}
                ),
            },
        }
    )
    validate_candidate_import_partition(result)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _portable_binding_path(path: Path, repo_root: Path) -> str:
    """Return a checkout-independent path for a repository input artifact."""

    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"input binding path outside repository: {path}") from exc


def _validate_paths(rows: list[dict[str, Any]], repo_root: Path) -> None:
    for row in rows:
        path = Path(str(row.get("path") or ""))
        if not path.is_absolute():
            path = repo_root / path
        try:
            path = path.resolve()
            path.relative_to(repo_root.resolve())
        except ValueError as exc:
            raise ValueError(
                "scenario path outside repository: "
                f"{row.get('scenario_id')}: {row.get('path')}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"scenario path missing: {row.get('scenario_id')}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--addition-source", type=Path, required=True)
    parser.add_argument("--admission-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    try:
        repo_root = args.repo_root.resolve()
        base_path = args.base.resolve()
        source_path = args.addition_source.resolve()
        selection_path = args.admission_selection.resolve()
        result = build_incremental_union(
            base_working_set=_read_json(base_path),
            addition_source_suite=_read_json(source_path),
            admission_selection=_read_json(selection_path),
            addition_source_sha256=_sha256(source_path),
            repo_root=repo_root,
        )
        _validate_paths(result["scenarios"], repo_root)
        result["input_bindings"] = {
            "base": {
                "path": _portable_binding_path(base_path, repo_root),
                "sha256": _sha256(base_path),
            },
            "addition_source": {
                "path": _portable_binding_path(source_path, repo_root),
                "sha256": _sha256(source_path),
            },
            "admission_selection": {
                "path": _portable_binding_path(selection_path, repo_root),
                "sha256": _sha256(selection_path),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_scenarios": result["n_scenarios"],
                "n_added": result["incremental_import"]["n_added"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
