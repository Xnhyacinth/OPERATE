#!/usr/bin/env python3
"""Build a candidate-only static-preflight queue for converted ``works`` suites.

This bridge deliberately does not convert raw assets or admit scenarios.  It
binds already materialized candidate Protocol-2.1 suites to the shared batch
coordinator, one immutable preflight command per suite.  Native replay remains
the next stage and is only scheduled after a preflight result is reviewed.
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

SUITE_SPECS = {
    "dynasched": {
        "domain": "logistics",
        "backend": "dynasched_flexible_job_shop",
        "suite": ROOT / "reports/works_dynasched_protocol21_candidate_suite_20260812.json",
        "preflight": ROOT / "reports/works_dynasched_preflight_20260812.json",
    },
    "citylearn": {
        "domain": "building_energy",
        "backend": "citylearn",
        "suite": ROOT / "reports/works_citylearn_protocol21_candidate_suite_20260812.json",
        "preflight": ROOT / "reports/works_citylearn_preflight_20260812.json",
    },
    "datacenter": {
        "domain": "datacenter",
        "backend": "alibaba_trace_sim",
        "suite": ROOT / "reports/works_datacenter_protocol21_candidate_suite_20260812.json",
        "preflight": ROOT / "reports/works_datacenter_preflight_20260812.json",
    },
    "pglib_uc": {
        "domain": "power_grid",
        "backend": "pglib_uc_synthetic",
        "suite": ROOT / "reports/works_pglib_uc_protocol21_candidate_suite_20260812.json",
        "preflight": ROOT / "reports/works_pglib_uc_preflight_20260812.json",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _suite_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load(path)
    if payload.get("status") != "working_set":
        raise ValueError(f"{path}: candidate suite must be a working_set")
    if (
        payload.get("release_ready") is not False
        or payload.get("leaderboard_eligible") is not False
    ):
        raise ValueError(f"{path}: candidate suite must remain non-release")
    rows = payload.get("scenarios")
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: scenarios must be a non-empty list")
    return rows


def build_queue(*, output: Path, selected: tuple[str, ...] = tuple(SUITE_SPECS)) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for name in selected:
        if name not in SUITE_SPECS:
            raise ValueError(f"unsupported works suite: {name}")
        spec = SUITE_SPECS[name]
        suite = Path(spec["suite"])
        if not suite.is_file():
            raise ValueError(f"candidate suite is missing: {suite}")
        rows = _suite_rows(suite)
        preflight = Path(spec["preflight"])
        command = [
            sys.executable,
            "scripts/preflight_protocol21_working_set.py",
            "--source-suite",
            str(suite),
            "--output",
            str(preflight),
            "--expected-count",
            str(len(rows)),
            "--require-source-consumption-adapters",
        ]
        bindings.append(
            {
                "name": name,
                "suite": str(suite.resolve()),
                "suite_sha256": _sha256(suite),
                "expected_count": len(rows),
            }
        )
        # One queue item per suite keeps a suite's source/backend atomic while
        # allowing independent domains to run in parallel.  This is an
        # aggregate command over every row in the suite, so it deliberately
        # carries no scenario identity.  Binding the first row would let an
        # exact locked-Core overlap incorrectly skip the entire suite.
        items.append(
            {
                "work_id": f"works-static-preflight:{name}",
                "stage": "static_preflight",
                "work_state": "pending",
                "disposition": None,
                "domain": str(spec["domain"]),
                "backend": str(spec["backend"]),
                "command": command,
                "metadata": {
                    "candidate_only": True,
                    "identity_scope": "suite_aggregate",
                    "suite_name": name,
                    "suite_sha256": _sha256(suite),
                    "n_scenarios": len(rows),
                    "release_admission": False,
                },
            }
        )
    items.sort(key=lambda row: str(row["work_id"]))
    queue = {
        "schema_version": "candidate-batch-queue-v1",
        "queue_kind": "works_protocol21_static_preflight_v1",
        "status": "pending",
        "candidate_only": True,
        "release_admission": False,
        "created_with": {
            "script": "build_works_candidate_batch_queue.py",
            "python": platform.python_version(),
        },
        "suite_bindings": bindings,
        "items": items,
        "policy": {
            "raw_assets_redistributed": False,
            "model_outcomes_used_for_filtering": False,
            "native_replay_after_static_preflight": True,
            "final_union_required_for_admission": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return queue


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--suites", nargs="+", choices=tuple(SUITE_SPECS), default=tuple(SUITE_SPECS)
    )
    args = parser.parse_args(argv)
    queue = build_queue(output=args.output.resolve(), selected=tuple(args.suites))
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "n_items": len(queue["items"]),
                "candidate_only": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
