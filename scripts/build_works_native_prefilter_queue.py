#!/usr/bin/env python3
"""Build a hash-bound, candidate-only native-prefilter queue for local works.

Only already materialized Protocol-2.1 rows are schedulable.  Raw source units
without a converted row receive an explicit terminal ledger entry; the command
never treats inventory presence as conversion or admission evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from scripts.build_works_candidate_batch_queue import SUITE_SPECS  # noqa: E402
from scripts.build_works_candidate_inventory import (  # noqa: E402
    DEFAULT_LOCKED_CORE,
    SOURCE_SPECS,
)
from scripts.build_works_candidate_inventory import (  # noqa: E402
    DEFAULT_OUTPUT as DEFAULT_INVENTORY,
)

FAMILY_TERMINALS = {
    "citylearn": ("held_repair", "converted_row_not_materialized"),
    "datacenter": ("held_repair", "converted_row_not_materialized"),
    "dynasched": ("transfer_only", "generated_method_transfer_source"),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    identity = (str(row.get("scenario_id") or ""), str(row.get("scenario_signature") or ""))
    if not all(identity):
        raise ValueError("candidate row lacks exact identity")
    return identity


def _manifest_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _source_unit_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    inventory_by_id = {str(row["source_id"]): row for row in inventory["sources"]}
    rows: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        if spec.source_id not in {
            "pglib_opf",
            "pglib_uc",
            "sumo365_ingolstadt",
            "resco",
            "citylearn",
            "dynaschedbench",
            "alibaba_clusterdata",
        }:
            continue
        source = inventory_by_id[spec.source_id]
        root = ROOT / spec.path
        for path in sorted(root.glob(spec.unit_glob)):
            if not (path.is_file() or path.is_dir()):
                continue
            relative = path.relative_to(ROOT).as_posix()
            disposition = str(source["disposition"])
            reason = disposition
            if spec.source_id == "pglib_opf":
                disposition, reason = "held_repair", "conversion_not_materialized"
            elif spec.source_id == "pglib_uc":
                disposition, reason = "held_repair", "source_unit_not_materialized"
            elif spec.source_id == "dynaschedbench":
                disposition, reason = "transfer_only", "generated_method_transfer_source"
            elif disposition == "held_license_or_terms":
                disposition, reason = (
                    "held_license_or_terms",
                    "license_or_terms_review",
                )
            elif disposition == "held_runtime":
                disposition, reason = "held_runtime", "native_runtime_unavailable"
            unit_hash = (
                _sha256(path) if path.is_file() else hashlib.sha256(relative.encode()).hexdigest()
            )
            unit_id = hashlib.sha256(f"{spec.source_id}:{relative}".encode()).hexdigest()[:16]
            rows.append(
                {
                    "unit_id": f"{spec.source_id}:{unit_id}",
                    "source_id": spec.source_id,
                    "source_unit": relative,
                    "source_unit_sha256": unit_hash,
                    "work_state": "terminal",
                    "disposition": disposition,
                    "reason": reason,
                    "simulator_calls": 0,
                }
            )
    return rows


def build_queue(
    *,
    inventory_path: Path,
    base_core_path: Path,
    output_root: Path,
    queue_path: Path,
    ledger_path: Path,
    shard_size: int = 64,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 1 <= shard_size <= 64:
        raise ValueError("shard_size must be in [1, 64]")
    inventory = _load(inventory_path)
    core = _load(base_core_path)
    locked = {_identity(row) for row in core.get("scenarios", [])}
    source_by_id = {str(row["source_id"]): row for row in inventory["sources"]}
    implementation = implementation_identity(ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    queue_items: list[dict[str, Any]] = []
    candidate_ledger: list[dict[str, Any]] = []

    for family, spec in sorted(SUITE_SPECS.items()):
        suite_path = Path(spec["suite"])
        suite = _load(suite_path)
        rows = suite.get("scenarios")
        if not isinstance(rows, list):
            raise ValueError(f"{suite_path}: scenarios missing")
        source_id = {"datacenter": "alibaba_clusterdata", "dynasched": "dynaschedbench"}.get(
            family, family
        )
        source = source_by_id[source_id]
        schedulable: list[dict[str, Any]] = []
        for row in rows:
            identity = _identity(row)
            base = {
                "family": family,
                "scenario_id": identity[0],
                "scenario_signature": identity[1],
                "suite_sha256": _sha256(suite_path),
            }
            if identity in locked:
                candidate_ledger.append(
                    {
                        **base,
                        "work_state": "terminal",
                        "disposition": "secondary_duplicate",
                        "reason": "exact_identity_already_in_locked_core",
                        "simulator_calls": 0,
                    }
                )
            elif family in FAMILY_TERMINALS:
                disposition, reason = FAMILY_TERMINALS[family]
                candidate_ledger.append(
                    {
                        **base,
                        "work_state": "terminal",
                        "disposition": disposition,
                        "reason": reason,
                        "simulator_calls": 0,
                    }
                )
            elif not bool(source["runtime_binding"]["available"]):
                candidate_ledger.append(
                    {
                        **base,
                        "work_state": "terminal",
                        "disposition": "held_runtime",
                        "reason": "native_runtime_unavailable",
                        "simulator_calls": 0,
                    }
                )
            else:
                schedulable.append(row)
                candidate_ledger.append(
                    {
                        **base,
                        "work_state": "pending",
                        "disposition": None,
                        "reason": "native_prefilter_scheduled",
                        "simulator_calls": 0,
                    }
                )

        for offset in range(0, len(schedulable), shard_size):
            shard = schedulable[offset : offset + shard_size]
            if not shard:
                continue
            shard_path = output_root / f"{family}_native_{offset // shard_size:03d}.json"
            shard_payload = {
                **suite,
                "scenarios": shard,
                "n_scenarios": len(shard),
                "release_ready": False,
                "leaderboard_eligible": False,
            }
            shard_path.write_text(
                json.dumps(shard_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            result_path = output_root / f"{family}_native_{offset // shard_size:03d}_preflight.json"
            first = shard[0]
            license_path = ROOT / str(source["license"].get("evidence_path") or "")
            queue_items.append(
                {
                    "work_id": f"works-native:{family}:{offset // shard_size:03d}",
                    "stage": "native_prefilter",
                    "work_state": "pending",
                    "disposition": None,
                    "domain": str(spec["domain"]),
                    "backend": str(spec["backend"]),
                    "scenario_id": str(first["scenario_id"]),
                    "scenario_signature": str(first["scenario_signature"]),
                    "command": [
                        sys.executable,
                        "scripts/preflight_protocol21_working_set.py",
                        "--source-suite",
                        str(shard_path),
                        "--output",
                        str(result_path),
                        "--expected-count",
                        str(len(shard)),
                        "--require-source-consumption-adapters",
                        "--exercise-source-adapters",
                    ],
                    "metadata": {
                        "candidate_only": True,
                        "source_suite_sha256": _sha256(shard_path),
                        "source_inventory_sha256": _sha256(inventory_path),
                        "base_core_sha256": _sha256(base_core_path),
                        "implementation_binding": implementation,
                        "runtime_binding": source["runtime_binding"],
                        "license_binding": {
                            **source["license"],
                            "evidence_sha256": _sha256(license_path)
                            if license_path.is_file()
                            else None,
                        },
                        "n_scenarios": len(shard),
                    },
                }
            )

    units = _source_unit_rows(inventory)
    queue = {
        "schema_version": "candidate-batch-queue-v1",
        "queue_kind": "works_native_prefilter_v1",
        "status": "pending" if queue_items else "complete",
        "candidate_only": True,
        "release_admission": False,
        "created_with": {"script": Path(__file__).name, "python": platform.python_version()},
        "items": queue_items,
    }
    terminal_counts = Counter(
        row["disposition"] for row in units if row["work_state"] == "terminal"
    )
    ledger = {
        "schema_version": "works-bulk-ledger-v1",
        "status": "native_prefilter_pending" if queue_items else "complete",
        "candidate_only": True,
        "release_admission": False,
        "bindings": {
            "inventory": {
                "path": _manifest_path(inventory_path),
                "sha256": _sha256(inventory_path),
            },
            "base_core": {
                "path": _manifest_path(base_core_path),
                "sha256": _sha256(base_core_path),
                "n_rows": len(core.get("scenarios", [])),
                "mutated": False,
            },
        },
        "source_units": units,
        "candidate_rows": candidate_ledger,
        "summary": {
            "n_source_units": len(units),
            "source_units_by_family": dict(
                sorted(Counter(row["source_id"] for row in units).items())
            ),
            "source_unit_terminal_dispositions": dict(sorted(terminal_counts.items())),
            "n_materialized_candidate_rows": len(candidate_ledger),
            "n_native_prefilter_scheduled_rows": sum(
                row["work_state"] == "pending" for row in candidate_ledger
            ),
            "n_core_overlap_simulator_calls": 0,
            "all_source_units_have_one_terminal": len({row["unit_id"] for row in units})
            == len(units),
        },
    }
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return queue, ledger


def finalize_ledger(
    *,
    ledger_path: Path,
    coordinator_ledger_path: Path,
    preflight_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Close every materialized row after a bounded native-prefilter run."""
    ledger = _load(ledger_path)
    coordinator = _load(coordinator_ledger_path)
    results: dict[tuple[str, str], dict[str, Any]] = {}
    result_bindings: list[dict[str, str]] = []
    for path in sorted(preflight_root.glob("*_preflight.json")):
        report = _load(path)
        result_bindings.append({"path": _manifest_path(path), "sha256": _sha256(path)})
        for row in report.get("results", []):
            results[_identity(row)] = row
    for row in ledger["candidate_rows"]:
        if row["work_state"] == "terminal":
            continue
        result = results.get((row["scenario_id"], row["scenario_signature"]))
        row["work_state"] = "terminal"
        if result is None:
            row["disposition"] = "held_runtime"
            row["reason"] = "native_prefilter_result_missing"
        elif result.get("fatal_blockers"):
            row["disposition"] = "held_repair"
            row["reason"] = "native_prefilter_fatal"
        else:
            row["disposition"] = "held_repair"
            row["reason"] = "source_state_effect_or_full_protocol21_unproven"
        row["native_prefilter_status"] = result.get("status") if result else None
    if any(row["work_state"] != "terminal" for row in ledger["candidate_rows"]):
        raise ValueError("candidate row missing terminal state")
    ledger["status"] = "complete_non_admitting"
    ledger["bindings"]["coordinator_ledger"] = {
        "path": _manifest_path(coordinator_ledger_path),
        "sha256": _sha256(coordinator_ledger_path),
        "status": coordinator.get("status"),
    }
    ledger["bindings"]["native_prefilter_results"] = result_bindings
    ledger["summary"]["n_native_prefilter_scheduled_rows"] = 0
    ledger["summary"]["n_materialized_terminal_rows"] = len(ledger["candidate_rows"])
    ledger["summary"]["materialized_terminal_dispositions"] = dict(
        sorted(Counter(row["disposition"] for row in ledger["candidate_rows"]).items())
    )
    ledger["summary"]["all_materialized_rows_have_one_terminal"] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--base-core", type=Path, default=DEFAULT_LOCKED_CORE)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--finalize-from", type=Path)
    parser.add_argument("--final-ledger", type=Path)
    args = parser.parse_args(argv)
    queue, ledger = build_queue(
        inventory_path=args.inventory.resolve(),
        base_core_path=args.base_core.resolve(),
        output_root=args.output_root.resolve(),
        queue_path=args.queue.resolve(),
        ledger_path=args.ledger.resolve(),
        shard_size=args.shard_size,
    )
    if bool(args.finalize_from) != bool(args.final_ledger):
        parser.error("--finalize-from and --final-ledger must be supplied together")
    if args.finalize_from and args.final_ledger:
        ledger = finalize_ledger(
            ledger_path=args.ledger.resolve(),
            coordinator_ledger_path=args.finalize_from.resolve(),
            preflight_root=args.output_root.resolve(),
            output_path=args.final_ledger.resolve(),
        )
    print(json.dumps({"n_queue_items": len(queue["items"]), **ledger["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
