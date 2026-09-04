#!/usr/bin/env python3
"""Materialize a Protocol-2.1 working set from a SUMO candidate queue.

This bridge performs no simulator calls.  It only admits source-locked,
runtime-ready candidate YAMLs into the existing Protocol-2.1 pipeline shape;
the pipeline remains responsible for native replay, headroom, depth, and Core
selection.  A queue item with a terminal disposition is rejected so a held or
failed candidate cannot be mistaken for a runnable source suite.
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

from core.suite_identity import recompute_signature_with_seed  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_queue(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate queue must be an object")
    if payload.get("schema_version") != "candidate-batch-queue-v1":
        raise ValueError("queue schema must be candidate-batch-queue-v1")
    if payload.get("candidate_only") is not True:
        raise ValueError("queue must remain candidate_only")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("queue must contain non-empty items")
    return payload


def _load_candidate(item: dict[str, Any], *, queue_path: Path) -> tuple[dict[str, Any], Path]:
    raw_path = item.get("candidate_yaml")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("candidate_yaml is required")
    path = Path(raw_path)
    if not path.is_absolute():
        path = (queue_path.parent / path).resolve()
        if not path.is_file():
            path = (REPO_ROOT / raw_path).resolve()
    if not path.is_file():
        raise ValueError(f"candidate YAML does not exist: {raw_path}")
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"candidate YAML is not an object: {path}")
    return body, path


def _row(item: dict[str, Any], body: dict[str, Any], path: Path) -> dict[str, Any]:
    if item.get("work_state") != "pending" or item.get("disposition") != "held_repair":
        raise ValueError(
            f"{item.get('work_id')}: only pending/held_repair items may enter a suite"
        )
    if body.get("domain") != "traffic" or body.get("backend_kind") != "sumo":
        raise ValueError("traffic suite accepts only domain=traffic/backend_kind=sumo")
    scenario_id = str(body.get("scenario_id") or body.get("seed_id") or "").strip()
    if not scenario_id or scenario_id != str(item.get("scenario_id") or ""):
        raise ValueError("queue/YAML scenario identity mismatch")
    source_contract = body.get("source_contract")
    if not isinstance(source_contract, dict):
        raise ValueError(f"{scenario_id}: source_contract is required")
    runtime_inputs = source_contract.get("runtime_input")
    hashes = source_contract.get("file_sha256s")
    if not isinstance(runtime_inputs, list) or not runtime_inputs:
        raise ValueError(f"{scenario_id}: runtime_input source lock is empty")
    if not isinstance(hashes, dict) or set(hashes) != set(runtime_inputs):
        raise ValueError(f"{scenario_id}: source file hash binding is incomplete")
    seed = body.get("seed")
    horizon = body.get("horizon_ticks")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{scenario_id}: seed must be a non-negative integer")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 1:
        raise ValueError(f"{scenario_id}: horizon_ticks must exceed one")
    config = body.get("backend_config")
    if not isinstance(config, dict):
        raise ValueError(f"{scenario_id}: backend_config is required")
    source_schedule = config.get("source_event_schedule")
    declared = config.get("declared_pressure_schedule")
    if not isinstance(source_schedule, list) or not source_schedule:
        raise ValueError(f"{scenario_id}: source schedule is empty")
    if any(event.get("origin") != "source_schedule" for event in source_schedule if isinstance(event, dict)):
        raise ValueError(f"{scenario_id}: source schedule origin is invalid")
    if not isinstance(declared, list) or not declared:
        raise ValueError(f"{scenario_id}: declared perturbation schedule is empty")
    if any(
        event.get("origin") != "declared_perturbation"
        or event.get("source_independence_credit") is not False
        for event in declared
        if isinstance(event, dict)
    ):
        raise ValueError(f"{scenario_id}: declared perturbation independence is invalid")
    signature = recompute_signature_with_seed(body, seed)
    physical = str(item.get("physical_source_identity") or "").strip()
    effective = str(item.get("effective_source_identity") or "").strip()
    if not physical or not effective:
        raise ValueError(f"{scenario_id}: source identities are required")
    return {
        "scenario_id": scenario_id,
        "path": _relative(path),
        "domain": "traffic",
        "backend_kind": "sumo",
        "family": str(body.get("family") or ""),
        "difficulty_mode": str(body.get("difficulty_mode") or ""),
        "difficulty_level": str(body.get("difficulty_level") or ""),
        "horizon_ticks": horizon,
        "seed": seed,
        "scenario_signature": signature,
        "source_key": effective,
        "source_denominator_key": str(
            config.get("source_denominator_key") or effective
        ),
        "structural_fingerprint": _canonical_sha256(
            {"physical": physical, "source_contract": source_contract}
        )[:16],
        "semantic_fingerprint": _canonical_sha256(
            {
                "difficulty": body.get("difficulty_level"),
                "source_schedule": source_schedule,
                "declared_perturbations": declared,
            }
        )[:16],
        "case_ledger": {
            "schema_version": "0.1",
            "physical_source_lock": source_contract,
            "source_denominator_key": str(
                config.get("source_denominator_key") or effective
            ),
            "source_schedule": source_schedule,
            "declared_perturbations": declared,
            "candidate_queue_item": str(item.get("work_id") or ""),
        },
        "protocol21_lineage": {
            "physical_identity_origin": "traffic_source_contract",
            "physical_source_identity": physical,
            "effective_source_identity": effective,
            "ready": True,
            "status": "ready_for_full_protocol21_replay",
            "reason_codes": [],
        },
        "status": "pending_protocol21_full_admission",
        "reason_codes": [
            "candidate_queue_runtime_ready",
            "declared_stressors_do_not_create_source_independence",
            "requires_behavior_task_depth_agentic_gates",
        ],
    }


def build_suite(queue_path: Path) -> dict[str, Any]:
    queue = _load_queue(queue_path)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw_item in queue["items"]:
        if not isinstance(raw_item, dict):
            raise ValueError("queue item must be an object")
        body, path = _load_candidate(raw_item, queue_path=queue_path)
        row = _row(raw_item, body, path)
        identity = (row["scenario_id"], row["scenario_signature"])
        if identity in seen:
            raise ValueError(f"duplicate suite identity: {identity}")
        seen.add(identity)
        rows.append(row)
    rows.sort(key=lambda value: (value["scenario_id"], value["scenario_signature"]))
    return {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "working_set",
        "selection_policy": "traffic_sumo365_candidate_v1",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(rows),
        "constraints": {
            "candidate_only": True,
            "candidate_evidence_merge_only": True,
            "candidate_replacements_staging_only": True,
            "formal_evaluation_ready": False,
            "model_outcomes_used_for_filtering": False,
            "one_per_effective_source_identity": True,
        },
        "source_artifacts": [
            {
                "kind": "traffic_candidate_queue",
                "path": _relative(queue_path),
                "sha256": _sha256(queue_path),
            }
        ],
        "scenarios": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    suite = build_suite(args.queue.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
