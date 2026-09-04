#!/usr/bin/env python3
"""Build deterministic native-prefilter shards for the Power Grid batch.

The queue points at an already materialized candidate working set.  It binds
the suite and frozen Core hashes, writes one immutable shard per <=64 rows, and
never executes a simulator.  ``run_protocol21_candidate_batches.py`` is the
only executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def build_queue(
    *,
    suite_path: Path,
    base_core_path: Path,
    output_root: Path,
    queue_path: Path,
    shard_size: int = 64,
) -> dict[str, Any]:
    if not 1 <= shard_size <= 64:
        raise ValueError("shard_size must be in [1, 64]")
    suite = _load(suite_path)
    if suite.get("status") != "working_set":
        raise ValueError("candidate suite must be a working_set")
    if suite.get("release_ready") is not False or suite.get("leaderboard_eligible") is not False:
        raise ValueError("candidate suite must remain non-release")
    rows = suite.get("scenarios")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("candidate suite scenarios must be a non-empty list")
    identities = {(str(row.get("scenario_id") or ""), str(row.get("scenario_signature") or "")) for row in rows}
    if any(not scenario_id or not signature for scenario_id, signature in identities):
        raise ValueError("every candidate row needs an exact identity")
    if len(identities) != len(rows):
        raise ValueError("candidate identities must be unique")
    if any(str(row.get("domain") or "") != "power_grid" for row in rows):
        raise ValueError("Power Grid queue cannot contain another domain")

    output_root.mkdir(parents=True, exist_ok=True)
    shard_rows = [rows[offset : offset + shard_size] for offset in range(0, len(rows), shard_size)]
    items: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for index, shard in enumerate(shard_rows):
        shard_path = output_root / "suites" / f"power_grid_native_{index:03d}.json"
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        shard_payload = {
            **suite,
            "n_scenarios": len(shard),
            "scenarios": shard,
            "release_ready": False,
            "leaderboard_eligible": False,
        }
        shard_path.write_text(json.dumps(shard_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result_path = output_root / "prefilter" / f"power_grid_native_{index:03d}.json"
        first = shard[0]
        item = {
            "work_id": f"powergrid:native_prefilter:{index:03d}",
            "stage": "native_prefilter",
            "work_state": "pending",
            "disposition": None,
            "domain": "power_grid",
            "backend": "pglib_uc_synthetic",
            "scenario_id": str(first["scenario_id"]),
            "scenario_signature": str(first["scenario_signature"]),
            "command": [
                sys.executable,
                "scripts/preflight_protocol21_working_set.py",
                "--source-suite",
                str(shard_path.resolve()),
                "--output",
                str(result_path.resolve()),
                "--expected-count",
                str(len(shard)),
                "--require-source-consumption-adapters",
                "--exercise-source-adapters",
            ],
            "metadata": {
                "candidate_only": True,
                "suite_sha256": _sha256(shard_path),
                "n_scenarios": len(shard),
                "source_families": sorted({str(row.get("family") or "") for row in shard}),
                "runtime_resource_tokens": 1,
                "release_admission": False,
            },
        }
        items.append(item)
        bindings.append(
            {
                "shard": str(shard_path.resolve()),
                "sha256": _sha256(shard_path),
                "n_scenarios": len(shard),
            }
        )

    queue = {
        "schema_version": "candidate-batch-queue-v1",
        "queue_kind": "powergrid_native_prefilter_v1",
        "status": "pending",
        "candidate_only": True,
        "release_admission": False,
        "created_with": {"script": Path(__file__).name, "python": platform.python_version()},
        "input_bindings": {
            "candidate_suite": {"path": str(suite_path.resolve()), "sha256": _sha256(suite_path)},
            "base_core": {"path": str(base_core_path.resolve()), "sha256": _sha256(base_core_path)},
        },
        "shard_bindings": bindings,
        "items": items,
        "policy": {
            "frozen_core_untouched": True,
            "model_outcomes_used_for_filtering": False,
            "native_prefilter_before_full_protocol21": True,
            "final_union_required_for_admission": True,
        },
    }
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return queue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--base-core", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=64)
    args = parser.parse_args(argv)
    queue = build_queue(
        suite_path=args.suite.resolve(),
        base_core_path=args.base_core.resolve(),
        output_root=args.output_root.resolve(),
        queue_path=args.queue.resolve(),
        shard_size=args.shard_size,
    )
    print(json.dumps({"status": queue["status"], "n_items": len(queue["items"]), "n_rows": sum(binding["n_scenarios"] for binding in queue["shard_bindings"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
