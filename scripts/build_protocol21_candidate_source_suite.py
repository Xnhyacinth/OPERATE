#!/usr/bin/env python3
"""Convert a staging candidate report into a Protocol-2.1 working set.

The converter is deliberately backend-neutral: it copies no source data and
does not call a simulator.  It verifies that report rows and YAMLs agree on
identity, source contract, and native metadata, then leaves all behavioral,
headroom, depth, and agentic decisions to the standard Protocol-2.1 pipeline.
"""

from __future__ import annotations

import argparse
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
    physical_source_lock_from_contract,
    resolve_source_asset_contract,
)
from core.suite_identity import recompute_signature_with_seed  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any, *, length: int = 16) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:length]


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        raise ValueError("candidate report must contain a scenarios list")
    return payload


def _load_yaml(row: dict[str, Any], *, report_path: Path) -> tuple[dict[str, Any], Path]:
    raw_path = row.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("candidate row path is required")
    path = Path(raw_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
        if not path.is_file():
            path = (report_path.parent / raw_path).resolve()
    if not path.is_file():
        raise ValueError(f"candidate YAML does not exist: {raw_path}")
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"candidate YAML is not an object: {path}")
    return body, path


def _row(raw: dict[str, Any], body: dict[str, Any], path: Path) -> dict[str, Any]:
    scenario_id = str(body.get("scenario_id") or body.get("seed_id") or "").strip()
    if not scenario_id or scenario_id != str(raw.get("scenario_id") or ""):
        raise ValueError("candidate report/YAML scenario_id mismatch")
    domain = str(body.get("domain") or "").strip()
    backend = str(body.get("backend_kind") or "").strip()
    if not domain or not backend:
        raise ValueError(f"{scenario_id}: domain/backend are required")
    source_contract = body.get("source_contract")
    if not isinstance(source_contract, dict):
        raise ValueError(f"{scenario_id}: source_contract is required")
    runtime_input = source_contract.get("runtime_input")
    derivation_input = source_contract.get("derivation_input")
    if not isinstance(runtime_input, list) or not isinstance(derivation_input, list):
        raise ValueError(f"{scenario_id}: source contract input lists are invalid")
    required_input = [*runtime_input, *derivation_input]
    if not required_input:
        raise ValueError(f"{scenario_id}: source contract has no required input")
    hashes = source_contract.get("file_sha256s")
    if not isinstance(hashes, dict) or set(hashes) != set(required_input):
        raise ValueError(f"{scenario_id}: source_contract.file_sha256s is incomplete")
    resolved_contract = resolve_source_asset_contract(body, repo_root=REPO_ROOT)
    if resolved_contract.contract_errors:
        raise ValueError(
            f"{scenario_id}: source contract errors: {','.join(resolved_contract.contract_errors)}"
        )
    if resolved_contract.missing_required_files:
        raise ValueError(
            f"{scenario_id}: source files missing: "
            f"{','.join(resolved_contract.missing_required_files)}"
        )
    physical_source_lock = physical_source_lock_from_contract(
        resolved_contract,
        backend_kind=backend,
    )
    if physical_source_lock is None:
        raise ValueError(f"{scenario_id}: verified physical source lock is unavailable")
    physical_source_key = canonical_physical_source_asset_key(physical_source_lock)
    # Physical identity stays asset-graph only. Effective source identity may
    # also include a contract-verified derived window so disjoint time slices
    # of the same files remain distinct Core rows.
    window = None
    if isinstance(physical_source_lock, dict):
        window = physical_source_lock.get("derived_window")
    source_identity = {"physical": physical_source_key}
    if isinstance(window, dict) and window.get("sha256"):
        source_identity["derived_window"] = window
    source_key = f"{backend}:{_digest(source_identity, length=32)}"
    seed = body.get("seed")
    horizon = body.get("horizon_ticks")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{scenario_id}: seed is invalid")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 1:
        raise ValueError(f"{scenario_id}: horizon_ticks is invalid")
    signature = recompute_signature_with_seed(body, seed)
    declared_signature = str(raw.get("scenario_signature") or "")
    if declared_signature and declared_signature != signature:
        raise ValueError(f"{scenario_id}: scenario_signature mismatch")
    config = body.get("backend_config")
    if not isinstance(config, dict):
        raise ValueError(f"{scenario_id}: backend_config is required")
    row = {
        "scenario_id": scenario_id,
        "path": _relative(path),
        "domain": domain,
        "backend_kind": backend,
        "family": str(body.get("family") or raw.get("family") or ""),
        "difficulty_mode": str(body.get("difficulty_mode") or raw.get("difficulty_mode") or ""),
        "difficulty_level": str(body.get("difficulty_level") or raw.get("difficulty_level") or ""),
        "horizon_ticks": horizon,
        "seed": seed,
        "scenario_signature": signature,
        "source_key": str(source_key),
        "source_denominator_key": str(source_key),
        "physical_source_key": physical_source_key,
        "structural_fingerprint": _digest(
            {"backend": backend, "physical_source_lock": physical_source_lock}
        ),
        "semantic_fingerprint": _digest(
            {
                "difficulty": body.get("difficulty_level"),
                "perturbations": body.get("perturbations"),
                "task_contract": body.get("task_contract") or config.get("task_requirements"),
            }
        ),
        "construct_contract": "operational_agency.v1",
        "case_ledger": {
            "schema_version": "0.1",
            "physical_source_lock": physical_source_lock,
            "physical_source_key": physical_source_key,
            "source_denominator_key": str(source_key),
            "candidate_report": "staging_report",
        },
        "protocol21_lineage": {
            "physical_identity_origin": "verified_source_asset_graph",
            "ready": True,
            "status": "ready_for_full_protocol21_replay",
            "reason_codes": ["candidate_only_requires_behavioral_and_depth_gates"],
        },
        "status": "pending_protocol21_full_admission",
        "reason_codes": ["candidate_only", "requires_full_protocol21_replay"],
    }
    reported_source_label = (
        config.get("source_denominator_key")
        or raw.get("source_denominator_key")
        or raw.get("source_key")
    )
    if reported_source_label not in (None, ""):
        row["reported_source_label"] = str(reported_source_label)
    for lineage_field in ("candidate_id", "historical_candidate_id"):
        lineage_value = raw.get(lineage_field)
        if lineage_value not in (None, ""):
            row[lineage_field] = str(lineage_value)
    return row


def build_suite(report_path: Path) -> dict[str, Any]:
    report = _load_report(report_path)
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    source_keys: set[str] = set()
    for raw in report["scenarios"]:
        if not isinstance(raw, dict):
            raise ValueError("candidate report rows must be objects")
        body, path = _load_yaml(raw, report_path=report_path)
        row = _row(raw, body, path)
        identity = (row["scenario_id"], row["scenario_signature"])
        if identity in identities:
            raise ValueError(f"duplicate candidate identity: {identity}")
        source_key = str(row["source_key"])
        if source_key in source_keys:
            raise ValueError(
                f"duplicate effective source identity: {source_key}"
            )
        identities.add(identity)
        source_keys.add(source_key)
        rows.append(row)
    rows.sort(key=lambda row: (row["scenario_id"], row["scenario_signature"]))
    return {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "working_set",
        "selection_policy": "candidate_report_source_locked_v1",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(rows),
        "constraints": {
            "core_admission_profile": "quality_core_v2",
            "candidate_only": True,
            "candidate_evidence_merge_only": True,
            "candidate_replacements_staging_only": True,
            "formal_evaluation_ready": False,
            "model_outcomes_used_for_filtering": False,
            "one_per_effective_source_identity": True,
        },
        "source_artifacts": [
            {
                "kind": "staging_candidate_report",
                "path": _relative(report_path),
                "sha256": _sha256(report_path),
            }
        ],
        "scenarios": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    suite = build_suite(args.report.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_scenarios": suite["n_scenarios"],
                "sha256": _sha256(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
