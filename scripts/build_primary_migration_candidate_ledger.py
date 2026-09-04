#!/usr/bin/env python3
"""Reuse terminal Primary-migration evidence in a candidate-only queue.

The v57 migration has already converted and replayed most prefilter-required
sources.  This wrapper binds that evidence to the migration inventory and the
locked base Core instead of spending simulator calls a second time.  Every
inventory item receives exactly one terminal disposition; no row is admitted
to Core and the resulting queue schedules zero commands in the shared batch
coordinator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIGRATION_PLAN = REPO_ROOT / "reports/protocol21_primary_migration_plan_v57.json"
DEFAULT_BASE_CORE = (
    REPO_ROOT
    / "reports/protocol21_pending_union_fresh_e18_realtraffic_v1"
    / "refined_core_selection_protocol2_v21.json"
)
DEFAULT_CONVERTED_SUITE = (
    REPO_ROOT / "reports/protocol21_primary_prefilter_all_v1/converted_source_suite.json"
)
DEFAULT_FULL_GATE = (
    REPO_ROOT
    / "reports/protocol21_primary_converted_full_e18_realtraffic_v1"
    / "refined_core_selection_protocol2_v21.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "reports/protocol21_primary_migration_terminal_queue_v1.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _identity(row: Mapping[str, Any], *, label: str) -> tuple[str, str]:
    identity = (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )
    if not all(identity):
        raise ValueError(f"{label} lacks exact scenario identity")
    return identity


def _migration_identity(row: Mapping[str, Any], *, label: str) -> str:
    value = str(
        row.get("canonical_effective_identity_sha256") or row.get("migration_identity_sha256") or ""
    )
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} lacks canonical migration SHA-256 identity")
    return value


def _unique_by_migration(rows: list[Any], *, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        identity = _migration_identity(raw, label=f"{label}[{index}]")
        if identity in result:
            raise ValueError(f"duplicate migration identity in {label}: {identity}")
        result[identity] = raw
    return result


def _gate_results(
    gate: dict[str, Any],
    *,
    converted_identities: set[tuple[str, str]],
) -> dict[tuple[str, str], tuple[str, list[str]]]:
    if gate.get("status") != "protocol21_core_candidate":
        raise ValueError("full gate must be a terminal Protocol-2.1 candidate artifact")
    result: dict[tuple[str, str], tuple[str, list[str]]] = {}
    fields = (
        ("scenarios", "candidate_survivor_requires_fresh_union"),
        ("rejected", None),
        ("secondary", "secondary_duplicate"),
    )
    for field, forced in fields:
        rows = gate.get(field) or []
        if not isinstance(rows, list):
            raise ValueError(f"full gate {field} must be a list")
        for index, raw in enumerate(rows):
            if not isinstance(raw, dict):
                raise ValueError(f"full gate {field}[{index}] must be an object")
            identity = _identity(raw, label=f"full gate {field}[{index}]")
            if identity not in converted_identities:
                raise ValueError(f"gate identity absent from converted suite: {identity[0]}")
            if identity in result:
                raise ValueError(f"duplicate terminal gate identity: {identity[0]}")
            reasons = raw.get("reason_codes") or []
            if not isinstance(reasons, list):
                reasons = []
            if raw.get("reason_code"):
                reasons = [*reasons, str(raw["reason_code"])]
            if forced == "candidate_survivor_requires_fresh_union":
                disposition = "held_repair"
                reasons = [*reasons, forced]
            else:
                disposition = forced or str(raw.get("disposition") or "held_repair")
            if disposition not in {
                "held_repair",
                "held_runtime",
                "held_license_or_terms",
                "transfer_only",
                "secondary_duplicate",
                "retired_intrinsic",
            }:
                raise ValueError(f"unsupported terminal gate disposition: {disposition}")
            result[identity] = (disposition, sorted(set(map(str, reasons))))
    if len(result) != len(converted_identities):
        missing = sorted(converted_identities - set(result))
        raise ValueError(f"converted input lacks terminal gate result: {missing[0][0]}")
    declared = gate.get("n_source")
    if declared is not None and int(declared) != len(converted_identities):
        raise ValueError("full gate n_source does not match converted source suite")
    return result


def build_terminal_ledger(
    *,
    migration_plan_path: Path = DEFAULT_MIGRATION_PLAN,
    base_core_path: Path = DEFAULT_BASE_CORE,
    converted_suite_path: Path = DEFAULT_CONVERTED_SUITE,
    full_gate_path: Path = DEFAULT_FULL_GATE,
) -> dict[str, Any]:
    """Return a coordinator-compatible terminal queue with no simulator work."""
    paths = {
        "migration_plan": migration_plan_path.resolve(),
        "base_core": base_core_path.resolve(),
        "converted_suite": converted_suite_path.resolve(),
        "full_gate": full_gate_path.resolve(),
    }
    payloads = {name: _load(path) for name, path in paths.items()}
    migration = payloads["migration_plan"]
    if migration.get("schema_version") != "protocol21-primary-migration-plan-v1":
        raise ValueError("unsupported migration plan schema")
    if migration.get("status") != "migration_plan_non_admitting":
        raise ValueError("migration plan must be non-admitting")
    migration_items = _unique_by_migration(
        list(migration.get("items") or []), label="migration items"
    )
    summary = migration.get("summary") or {}
    if int(summary.get("planned_items") or -1) != len(migration_items):
        raise ValueError("migration plan item count mismatch")

    base_rows = payloads["base_core"].get("scenarios") or []
    if not isinstance(base_rows, list):
        raise ValueError("base Core scenarios must be a list")
    base_identities = {
        _identity(row, label=f"base Core[{index}]")
        for index, row in enumerate(base_rows)
        if isinstance(row, dict)
    }

    converted_payload = payloads["converted_suite"]
    if converted_payload.get("status") != "working_set":
        raise ValueError("converted suite must be a candidate working_set")
    converted = _unique_by_migration(
        list(converted_payload.get("scenarios") or []), label="converted suite"
    )
    unknown_converted = sorted(set(converted) - set(migration_items))
    if unknown_converted:
        raise ValueError(f"converted migration identity absent from plan: {unknown_converted[0]}")
    converted_by_exact: dict[tuple[str, str], str] = {}
    for migration_id, row in converted.items():
        exact = _identity(row, label=f"converted suite {migration_id}")
        if exact in converted_by_exact:
            raise ValueError(f"duplicate converted exact identity: {exact[0]}")
        converted_by_exact[exact] = migration_id
    gate_results = _gate_results(
        payloads["full_gate"], converted_identities=set(converted_by_exact)
    )

    items: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    for migration_id, source in sorted(migration_items.items()):
        source_exact = _identity(source, label=f"migration item {migration_id}")
        converted_row = converted.get(migration_id)
        evidence_reused = False
        if source_exact in base_identities:
            disposition = "secondary_duplicate"
            reasons = ["exact_identity_already_in_locked_core"]
            counters["core_overlap"] += 1
        elif source.get("terminal_lane") == "transfer_only":
            disposition = "transfer_only"
            reasons = ["migration_plan_transfer_only_lane"]
            counters["transfer_only"] += 1
        elif converted_row is None:
            disposition = "held_runtime"
            reasons = [
                "converted_candidate_missing_current_runtime_contract",
                f"backend:{source.get('backend_kind') or 'unknown'}",
            ]
            counters["held_runtime"] += 1
        else:
            exact = _identity(converted_row, label=f"converted row {migration_id}")
            disposition, reasons = gate_results[exact]
            evidence_reused = True
            counters["reused"] += 1
        reason_counts.update(reasons)
        items.append(
            {
                "work_id": f"primary-migration-{migration_id}",
                "stage": "evidence_freeze",
                "work_state": "terminal",
                "disposition": disposition,
                "domain": str(source.get("domain") or "unknown"),
                "backend": str(source.get("backend_kind") or "unknown"),
                "scenario_id": source_exact[0],
                "scenario_signature": source_exact[1],
                "migration_identity_sha256": migration_id,
                "reason_codes": reasons,
                "evidence_reused": evidence_reused,
                "simulator_calls": 0,
                "core_admission": False,
            }
        )

    if len(items) != len(migration_items) or len({item["work_id"] for item in items}) != len(items):
        raise ValueError("migration terminal accounting is not one-to-one")
    observed_lane_counts = Counter(
        str(row.get("terminal_lane")) for row in migration_items.values()
    )
    if observed_lane_counts["transfer_only"] != int(summary.get("transfer_only") or 0):
        raise ValueError("migration transfer-only count mismatch")
    if observed_lane_counts["prefilter_required"] != int(summary.get("prefilter_required") or 0):
        raise ValueError("migration prefilter-required count mismatch")
    materialization_counts = Counter(
        str(row.get("scenario_materialization_status") or "unspecified")
        for row in migration_items.values()
    )
    for status in ("historical_yaml_present", "historical_yaml_archived"):
        declared = summary.get(status)
        if declared is not None and int(declared) != materialization_counts[status]:
            raise ValueError(f"migration {status} count mismatch")

    return {
        "schema_version": "candidate-batch-queue-v1",
        "queue_kind": "protocol21_primary_migration_terminal_evidence_reuse_v1",
        "status": "terminal",
        "release_admission": False,
        "candidate_only": True,
        "bindings": {name: _binding(path) for name, path in paths.items()},
        "summary": {
            "n_input": len(items),
            "n_terminal": len(items),
            "n_reused_full_protocol21": counters["reused"],
            "n_core_overlap_zero_calls": counters["core_overlap"],
            "n_held_runtime": counters["held_runtime"],
            "n_native_pending": counters["held_runtime"],
            "n_transfer_only": counters["transfer_only"],
            "n_historical_yaml_present": materialization_counts["historical_yaml_present"],
            "n_historical_yaml_archived": materialization_counts["historical_yaml_archived"],
            "simulator_calls_requested": 0,
        },
        "disposition_counts": dict(sorted(Counter(item["disposition"] for item in items).items())),
        "held_reason_counts": dict(sorted(reason_counts.items())),
        "policy": {
            "base_core_locked": True,
            "core_overlap_simulator_calls": 0,
            "all_inputs_have_one_terminal_disposition": True,
            "full_protocol21_evidence_is_hash_bound_and_reused": True,
            "final_union_required_for_any_future_admission": True,
        },
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migration-plan", type=Path, default=DEFAULT_MIGRATION_PLAN)
    parser.add_argument("--base-core", type=Path, default=DEFAULT_BASE_CORE)
    parser.add_argument("--converted-suite", type=Path, default=DEFAULT_CONVERTED_SUITE)
    parser.add_argument("--full-gate", type=Path, default=DEFAULT_FULL_GATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ledger = build_terminal_ledger(
        migration_plan_path=args.migration_plan,
        base_core_path=args.base_core,
        converted_suite_path=args.converted_suite,
        full_gate_path=args.full_gate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **ledger["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
