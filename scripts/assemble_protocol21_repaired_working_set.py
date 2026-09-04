#!/usr/bin/env python3
"""Assemble source-locked Protocol-2.1 staging replacements.

This utility updates a non-release working set only when a candidate preserves
the same effective source identity and exact physical source lock as the row
it replaces.  It does not grant any behavioral, depth, or formal-readiness
gate; the assembled set must be rerun through the complete pipeline.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.source_asset_contract import (  # noqa: E402
    canonical_physical_source_asset_key,
    physical_source_lock_from_contract,
    resolve_source_asset_contract,
)
from core.working_set_contract import validate_protocol21_row_lineage  # noqa: E402
from runner.resume import recompute_signature_with_seed  # noqa: E402
from scripts.audit_core_difficulty import _semantic_fingerprint  # noqa: E402
from scripts.build_primary_suite import structural_fingerprint  # noqa: E402


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _load_body(path: Path, *, source_bytes: bytes | None = None) -> dict[str, Any]:
    raw = path.read_bytes() if source_bytes is None else source_bytes
    payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"candidate is not a mapping: {path}")
    payload["source_contract"] = payload.get("source_contract") or {}
    expected = recompute_signature_with_seed(payload, int(payload.get("seed") or 0))
    if str(payload.get("scenario_signature") or "") != expected:
        raise ValueError(f"candidate signature mismatch: {path}")
    contract = resolve_source_asset_contract(payload, repo_root=REPO_ROOT)
    if contract.contract_errors or contract.missing_required_files:
        raise ValueError(
            f"candidate source contract invalid: {path}: "
            f"{sorted(contract.contract_errors)} {contract.missing_required_files}"
        )
    lock = physical_source_lock_from_contract(
        contract, backend_kind=str(payload.get("backend_kind") or "")
    )
    if lock is None:
        raise ValueError(f"candidate physical source lock missing: {path}")
    payload["_physical_source_lock"] = lock
    return payload


def _load_abandonment_spec(
    path: Path,
    *,
    base_sha256: str,
    source_bytes: bytes | None = None,
) -> list[dict[str, Any]]:
    raw = path.read_bytes() if source_bytes is None else source_bytes
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("abandonment spec must be a mapping")
    if payload.get("schema_version") != "operate_working_set_abandonment_v1":
        raise ValueError("abandonment spec schema mismatch")
    if payload.get("status") != "terminal":
        raise ValueError("abandonment spec must have status=terminal")
    if payload.get("base_working_set_sha256") != base_sha256:
        raise ValueError("abandonment spec base hash mismatch")
    abandonments = payload.get("abandonments")
    if not isinstance(abandonments, list) or not abandonments:
        raise ValueError("abandonment spec must contain abandonments")
    if not all(isinstance(item, dict) for item in abandonments):
        raise ValueError("abandonment spec entries must be mappings")
    return abandonments


def _normalize_abandonment(raw: dict[str, Any]) -> dict[str, Any]:
    scenario_id = str(raw.get("scenario_id") or "")
    signature = str(raw.get("scenario_signature") or "")
    reasons = raw.get("reason_codes")
    valid = (
        bool(scenario_id)
        and bool(signature)
        and raw.get("disposition") == "abandoned_terminal"
        and raw.get("included") is False
        and isinstance(reasons, list)
        and bool(reasons)
        and all(isinstance(reason, str) and reason for reason in reasons)
    )
    if not valid:
        raise ValueError(f"invalid abandonment: {scenario_id or '<missing>'}")
    return {
        "scenario_id": scenario_id,
        "scenario_signature": signature,
        "disposition": "abandoned_terminal",
        "included": False,
        "reason_codes": list(dict.fromkeys(reasons)),
    }


def _validate_existing_abandonments(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("invalid existing abandonment inventory")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("invalid existing abandonment inventory")
        identity = (
            str(raw.get("scenario_id") or ""),
            str(raw.get("scenario_signature") or ""),
        )
        reasons = raw.get("reason_codes")
        if (
            not all(identity)
            or raw.get("disposition") != "abandoned_terminal"
            or raw.get("included") is not False
            or not isinstance(reasons, list)
            or not reasons
            or not all(isinstance(reason, str) and reason for reason in reasons)
            or identity in seen
        ):
            raise ValueError("invalid existing abandonment inventory")
        seen.add(identity)
        result.append(copy.deepcopy(raw))
    return result


def _row_physical_source_key(row: dict[str, Any]) -> str:
    ledger = row.get("case_ledger") or {}
    lock = row.get("_physical_source_lock") or ledger.get("physical_source_lock")
    if not isinstance(lock, (dict, list, str)) or not lock:
        raise ValueError(
            "assembled working set has missing physical source lock: "
            f"{row.get('scenario_id')}"
        )
    return canonical_physical_source_asset_key(lock)


def _resolve_replacement_target(
    *,
    rows_by_id: dict[str, dict[str, Any]],
    rows_by_source: dict[str, dict[str, Any]],
    scenario_id: str,
    declared_source_key: str,
) -> tuple[dict[str, Any] | None, str]:
    base = rows_by_id.get(scenario_id)
    declared_owner = rows_by_source.get(declared_source_key)
    if base is not None:
        if declared_owner is not None and declared_owner is not base:
            raise ValueError(f"candidate identity conflicts: {scenario_id}")
        effective_key = str(base.get("source_denominator_key") or "")
        if not effective_key:
            raise ValueError(f"base source denominator missing: {scenario_id}")
        return base, effective_key
    return declared_owner, declared_source_key


def _build_row(
    *,
    body: dict[str, Any],
    path: Path,
    base: dict[str, Any] | None,
    repo_root: Path,
    source_denominator: str | None = None,
) -> dict[str, Any]:
    config = dict(body.get("backend_config") or {})
    source_denominator = source_denominator or str(
        config.get("source_denominator_key") or ""
    )
    if not source_denominator:
        raise ValueError(f"candidate source denominator missing: {path}")
    row = copy.deepcopy(base) if base is not None else {}
    if base is not None:
        base_lock = (base.get("case_ledger") or {}).get("physical_source_lock")
        if _canonical(base_lock) != _canonical(body["_physical_source_lock"]):
            raise ValueError(
                f"replacement physical source lock mismatch: {body.get('scenario_id')}"
            )
    # ``_physical_source_lock`` is an assembly-time verification value, not
    # scenario semantics.  Excluding it keeps fingerprints stable between a
    # candidate emitted by a generator and the assembled working-set row.
    fingerprint_body = copy.deepcopy(body)
    fingerprint_body.pop("_physical_source_lock", None)
    row.update(
        {
            "scenario_id": str(body.get("scenario_id") or body.get("seed_id") or ""),
            "path": _relative(path, repo_root),
            "backend_kind": body.get("backend_kind"),
            "domain": body.get("domain"),
            "family": body.get("family"),
            "difficulty_mode": body.get("difficulty_mode"),
            "difficulty_level": body.get("difficulty_level"),
            "horizon_ticks": int(body.get("horizon_ticks") or 0),
            "seed": int(body.get("seed") or 0),
            "scenario_signature": body.get("scenario_signature"),
            "source_denominator_key": source_denominator,
            "structural_fingerprint": structural_fingerprint(fingerprint_body),
            "semantic_fingerprint": _semantic_fingerprint(fingerprint_body),
        }
    )
    if not row.get("source_key"):
        row["source_key"] = _canonical(
            {
                "backend": body.get("backend_kind"),
                "source": (body.get("provenance") or {}).get("data_source"),
                "instance_name": config.get("instance_name"),
            }
        )
    ledger = copy.deepcopy(row.get("case_ledger") or {})
    ledger.update(
        {
            "schema_version": "0.1",
            "source_denominator_key": source_denominator,
            "physical_source_lock": body["_physical_source_lock"],
            "behavioral_validation": "pending_protocol21_recalibration",
            "additional_decision_axis": (
                f"difficulty={body.get('difficulty_mode')}/{body.get('difficulty_level')}"
            ),
            "decision_pressure_axis": (
                "native_dynamic_event_recovery_and_long_horizon_scheduling"
            ),
            "decision_variant_key": str(row["semantic_fingerprint"]),
            "complexity_tags": [
                f"n_perturbations={len(body.get('perturbations') or [])}",
                "procedural_events_source_locked_targets",
                "response_window_required",
                "pending_full_protocol21_gates",
            ],
            "event_repairs": [],
            "source_refinement": {
                "pipeline": "protocol21_dynamic_repair_assembly_v1",
                "candidate_path": _relative(path, repo_root),
                "replaces_scenario_id": (
                    base.get("scenario_id") if base is not None else None
                ),
            },
        }
    )
    row["case_ledger"] = ledger
    row["protocol21_lineage"] = {
        "physical_identity_origin": "verified_source_asset_graph",
        "ready": True,
        "status": "ready",
        "reason_codes": [],
        "rematerialized_from_scenario_id": (
            base.get("scenario_id") if base is not None else None
        ),
    }
    row.pop("_physical_source_lock", None)
    errors = validate_protocol21_row_lineage(row)
    if errors:
        raise ValueError(
            f"candidate row lineage invalid: {row['scenario_id']}: {errors}"
        )
    return row


def assemble_repaired_working_set(
    *,
    base_working_set: dict[str, Any],
    candidate_paths: Iterable[Path],
    abandonments: Iterable[dict[str, Any]] = (),
    candidate_source_bytes: dict[Path, bytes] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if base_working_set.get("status") != "working_set":
        raise ValueError("base working set must have status=working_set")
    if base_working_set.get("leaderboard_eligible") is not False:
        raise ValueError("base working set must not be leaderboard eligible")
    if base_working_set.get("candidate_import_partition") is not None:
        raise ValueError("assembler requires an unpartitioned base working set")
    rows = [copy.deepcopy(row) for row in base_working_set.get("scenarios") or []]
    by_source = {str(row.get("source_denominator_key")): row for row in rows}
    if len(by_source) != len(rows):
        raise ValueError("base working set has duplicate effective source identities")
    by_id = {str(row.get("scenario_id") or ""): row for row in rows}
    if len(by_id) != len(rows) or "" in by_id:
        raise ValueError("base working set has invalid scenario identities")
    replacements: list[dict[str, Any]] = []
    seen_candidates: set[str] = set()
    for candidate_path in candidate_paths:
        path = candidate_path.resolve()
        source_bytes = None
        if candidate_source_bytes is not None:
            try:
                source_bytes = candidate_source_bytes[path]
            except KeyError as exc:
                raise ValueError(f"candidate bytes missing: {path}") from exc
        body = _load_body(path, source_bytes=source_bytes)
        declared_key = str(
            (body.get("backend_config") or {}).get("source_denominator_key") or ""
        )
        scenario_id = str(body.get("scenario_id") or body.get("seed_id") or "")
        base, key = _resolve_replacement_target(
            rows_by_id=by_id,
            rows_by_source=by_source,
            scenario_id=scenario_id,
            declared_source_key=declared_key,
        )
        if key in seen_candidates:
            raise ValueError(f"duplicate candidate effective source: {key}")
        seen_candidates.add(key)
        row = _build_row(
            body=body,
            path=path,
            base=base,
            source_denominator=key,
            repo_root=repo_root,
        )
        if base is None:
            rows.append(row)
        else:
            index = rows.index(base)
            rows[index] = row
            by_id.pop(str(base.get("scenario_id") or ""), None)
        by_source[key] = row
        if row["scenario_id"] in by_id:
            raise ValueError(
                f"duplicate candidate scenario identity: {row['scenario_id']}"
            )
        by_id[row["scenario_id"]] = row
        replacements.append(
            {
                "base_scenario_id": base.get("scenario_id") if base else None,
                "candidate_scenario_id": row["scenario_id"],
                "source_denominator_key": key,
                "physical_source_lock_preserved": base is not None,
                "candidate_path": _relative(path, repo_root),
            }
        )
    existing_abandoned = _validate_existing_abandonments(
        base_working_set.get("abandoned_candidates", [])
    )
    abandoned_identities = {
        (
            str(item.get("scenario_id") or ""),
            str(item.get("scenario_signature") or ""),
        )
        for item in existing_abandoned
    }
    terminalized: list[dict[str, Any]] = []
    for raw in abandonments:
        abandonment = _normalize_abandonment(raw)
        identity = (
            abandonment["scenario_id"],
            abandonment["scenario_signature"],
        )
        matches = [
            row
            for row in rows
            if (
                str(row.get("scenario_id") or ""),
                str(row.get("scenario_signature") or ""),
            )
            == identity
        ]
        if len(matches) != 1:
            raise ValueError(
                f"abandonment identity not active: {identity[0]}@{identity[1]}"
            )
        if identity in abandoned_identities:
            raise ValueError(
                f"abandonment identity already terminal: {identity[0]}@{identity[1]}"
            )
        rows.remove(matches[0])
        existing_abandoned.append(abandonment)
        abandoned_identities.add(identity)
        terminalized.append(abandonment)
    final_keys = [str(row.get("source_denominator_key") or "") for row in rows]
    if any(not key for key in final_keys) or len(set(final_keys)) != len(final_keys):
        raise ValueError("assembled working set has invalid source denominator keys")
    physical_keys = [_row_physical_source_key(row) for row in rows]
    active_identities = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in rows
    }
    if active_identities.intersection(abandoned_identities):
        raise ValueError("assembled active and abandoned identities overlap")
    result = copy.deepcopy(base_working_set)
    result.update(
        {
            "status": "working_set",
            "leaderboard_eligible": False,
            "release_ready": False,
            "n_scenarios": len(rows),
            "n_effective_sources": len(set(final_keys)),
            "n_physical_sources": len(set(physical_keys)),
            "candidate_replacements": replacements,
            "abandoned_candidates": sorted(
                existing_abandoned,
                key=lambda item: (
                    str(item.get("scenario_id") or ""),
                    str(item.get("scenario_signature") or ""),
                ),
            ),
            "scenarios": sorted(rows, key=lambda item: str(item.get("scenario_id"))),
        }
    )
    constraints = dict(result.get("constraints") or {})
    constraints.update(
        {
            "one_per_effective_source_identity": True,
            "candidate_replacements_staging_only": True,
            "formal_evaluation_ready": False,
        }
    )
    result["constraints"] = constraints
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-working-set", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--abandonment-spec", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.candidate and args.abandonment_spec is None:
        parser.error("at least one --candidate or --abandonment-spec is required")
    try:
        base_path = args.base_working_set.resolve()
        base_bytes = base_path.read_bytes()
        base_sha256 = _sha256_bytes(base_bytes)
        candidate_paths = [path.resolve() for path in args.candidate]
        candidate_source_bytes = {path: path.read_bytes() for path in candidate_paths}
        abandonment_spec_path = (
            args.abandonment_spec.resolve() if args.abandonment_spec else None
        )
        abandonment_spec_bytes = (
            abandonment_spec_path.read_bytes()
            if abandonment_spec_path is not None
            else None
        )
        abandonments = (
            _load_abandonment_spec(
                abandonment_spec_path,
                base_sha256=base_sha256,
                source_bytes=abandonment_spec_bytes,
            )
            if abandonment_spec_path is not None
            else []
        )
        result = assemble_repaired_working_set(
            base_working_set=json.loads(base_bytes),
            candidate_paths=candidate_paths,
            abandonments=abandonments,
            candidate_source_bytes=candidate_source_bytes,
            repo_root=REPO_ROOT,
        )
        result["input_bindings"] = {
            "base_working_set": {
                "path": _relative(base_path, REPO_ROOT),
                "sha256": base_sha256,
            },
            "candidate_paths": [
                {
                    "path": _relative(path, REPO_ROOT),
                    "sha256": _sha256_bytes(candidate_source_bytes[path]),
                }
                for path in candidate_paths
            ],
        }
        if abandonment_spec_path is not None:
            result["input_bindings"]["abandonment_spec"] = {
                "path": _relative(abandonment_spec_path, REPO_ROOT),
                "sha256": _sha256_bytes(abandonment_spec_bytes or b""),
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_scenarios": result["n_scenarios"],
                "n_replacements": len(result["candidate_replacements"]),
                "n_abandonments": len(abandonments),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
