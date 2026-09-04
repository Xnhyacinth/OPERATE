#!/usr/bin/env python3
"""Create a deterministic candidate-batch queue for the existing Wave-1 conversions.

The queue is intentionally candidate-only.  Each family is converted into an
isolated staging directory, source-preflighted, and then sent through the
Protocol-2.1 pipeline.  No step writes the frozen Core or a release manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "reports/protocol21_wave1_candidate_queue.json"
DEFAULT_MATERIALIZATION_ROOT = REPO_ROOT / "reports/protocol21_wave1_candidate_batches"
PYTHON = REPO_ROOT / ".venv/bin/python"

FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "name": "power_grid_acopf",
        "domain": "power_grid",
        "backend": "pandapower_acopf",
        "converter": "scripts/convert_primary_acopf_wave1.py",
        "source_suite": "scenarios/staging/v0_57_primary_acopf_wave1/source_suite.json",
    },
    {
        "name": "microgrid_lv",
        "domain": "microgrid",
        "backend": "pandapower_lv",
        "converter": "scripts/convert_primary_lv_wave1.py",
        "source_suite": "scenarios/staging/v0_57_primary_lv_wave1/source_suite.json",
    },
    {
        "name": "logistics_orgym",
        "domain": "logistics",
        "backend": "orgym_invmgmt",
        "converter": "scripts/convert_primary_orgym_wave1.py",
        "source_suite": "scenarios/staging/v0_57_primary_orgym_wave1/source_suite.json",
    },
    {
        "name": "traffic_sumo",
        "domain": "traffic",
        "backend": "sumo",
        "converter": "scripts/convert_primary_sumo_wave1.py",
        "source_suite": "scenarios/staging/v0_57_primary_sumo_wave1/source_suite.json",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        raise ValueError(f"{path}: source suite must contain scenarios list")
    return payload


def _python_argv() -> str:
    return str(PYTHON if PYTHON.is_file() else Path(sys.executable))


def _item(
    *,
    work_id: str,
    stage: str,
    family: dict[str, Any],
    command: list[str],
    source_suite: Path,
    output_root: Path,
    expected_count: int,
) -> dict[str, Any]:
    return {
        "work_id": work_id,
        "stage": stage,
        "work_state": "pending",
        "disposition": None,
        "domain": family["domain"],
        "backend": family["backend"],
        "command": command,
        "metadata": {
            "family": family["name"],
            "source_suite": str(source_suite),
            "source_suite_sha256": _sha256(source_suite),
            "expected_count": expected_count,
            "identity_scope": "suite_aggregate",
            "candidate_output_root": str(output_root),
            "release_admission": False,
        },
    }


def build_queue(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    materialization_root: Path = DEFAULT_MATERIALIZATION_ROOT,
    families: tuple[dict[str, Any], ...] = FAMILIES,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    family_summary: list[dict[str, Any]] = []
    python = _python_argv()
    for family in families:
        source_suite = (REPO_ROOT / family["source_suite"]).resolve()
        converter = (REPO_ROOT / family["converter"]).resolve()
        if not source_suite.is_file():
            raise FileNotFoundError(source_suite)
        if not converter.is_file():
            raise FileNotFoundError(converter)
        suite = _load_suite(source_suite)
        expected_count = len(suite["scenarios"])
        family_root = (materialization_root / family["name"]).resolve()
        pipeline_root = family_root / "protocol21"
        suite_target = family_root / "source_suite.json"
        preflight_target = family_root / "preflight.json"
        items.append(
            _item(
                work_id=f"wave1-{family['name']}-conversion",
                stage="conversion",
                family=family,
                command=[
                    python,
                    str(converter),
                    "--output-dir",
                    str(family_root),
                ],
                source_suite=source_suite,
                output_root=family_root,
                expected_count=expected_count,
            )
        )
        items.append(
            _item(
                work_id=f"wave1-{family['name']}-static-preflight",
                stage="static_preflight",
                family=family,
                command=[
                    python,
                    str(REPO_ROOT / "scripts/preflight_protocol21_working_set.py"),
                    "--source-suite",
                    str(suite_target),
                    "--output",
                    str(preflight_target),
                    "--expected-count",
                    str(expected_count),
                    "--require-source-consumption-adapters",
                    "--exercise-source-adapters",
                ],
                source_suite=source_suite,
                output_root=family_root,
                expected_count=expected_count,
            )
        )
        items.append(
            _item(
                work_id=f"wave1-{family['name']}-full-protocol21",
                stage="full_protocol21",
                family=family,
                command=[
                    python,
                    str(REPO_ROOT / "scripts/run_protocol21_core_pipeline.py"),
                    "--source-suite",
                    str(suite_target),
                    "--release-dir",
                    str(pipeline_root),
                    "--expected-count",
                    str(expected_count),
                    "--execute",
                ],
                source_suite=source_suite,
                output_root=pipeline_root,
                expected_count=expected_count,
            )
        )
        family_summary.append(
            {
                "family": family["name"],
                "domain": family["domain"],
                "backend": family["backend"],
                "source_suite": _display_path(source_suite),
                "source_suite_sha256": _sha256(source_suite),
                "n_source_rows": expected_count,
                "candidate_output_root": str(family_root),
            }
        )
    return {
        "schema_version": "candidate-batch-queue-v1",
        "queue_kind": "protocol21_wave1_existing_conversion_candidates",
        "release_admission": False,
        "status": "pending",
        "created_with": {
            "python": platform.python_version(),
            "implementation": "build_wave1_candidate_queue.py",
        },
        "base_core_policy": {
            "locked_core_is_not_mutated": True,
            "final_union_required_for_admission": True,
            "exact_identity_skip_happens_in_coordinator": True,
        },
        "families": family_summary,
        "items": items,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--materialization-root", type=Path, default=DEFAULT_MATERIALIZATION_ROOT
    )
    args = parser.parse_args(argv)
    queue = build_queue(
        output_path=args.output,
        materialization_root=args.materialization_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": queue["status"],
                "n_items": len(queue["items"]),
                "n_families": len(queue["families"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
