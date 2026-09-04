#!/usr/bin/env python3
"""Stage source-preserving Protocol-2.1 depth repairs for native UC rows.

The RTS-GMLC/pglib-uc bytes and event schedules are unchanged.  This utility
only adds the replay-verifiable ordered reserve/redispatch task contract to
existing High/Extreme rows and recomputes their scenario signatures.  The
full Protocol-2.1 pipeline remains the admission authority.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.power_grid.seeds.from_pglib_uc import (  # noqa: E402
    load_case,
    ordered_uc_task_requirements,
)
from runner.resume import recompute_signature_with_seed  # noqa: E402

DEFAULT_SOURCE_SUITE = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "protocol21_expansion_trials"
    / "working_set_dynamic_repaired_v1"
    / "source_suite.json"
)
DEFAULT_STAGING = REPO_ROOT / "scenarios" / "staging" / "v0_52_protocol21_pglib_uc_depth"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "pglib_uc_depth_candidates_v1.json"
)


def _load_suite(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("scenarios"), list):
        raise ValueError(f"invalid source suite: {path}")
    return payload


def build(
    *,
    source_suite: Path = DEFAULT_SOURCE_SUITE,
    staging_root: Path = DEFAULT_STAGING,
    repo_root: Path = REPO_ROOT,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    suite = _load_suite(source_suite)
    rows: list[dict[str, Any]] = []
    files: dict[Path, dict[str, Any]] = {}
    for original in suite["scenarios"]:
        if not isinstance(original, dict):
            continue
        if original.get("backend_kind") != "pglib_uc_synthetic":
            continue
        level = str(original.get("difficulty_level") or "").lower()
        if level not in {"high", "extreme"}:
            continue
        source_path = repo_root / str(original.get("path") or "")
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        body = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        if not isinstance(body, dict):
            raise ValueError(f"scenario is not a mapping: {source_path}")
        candidate = copy.deepcopy(body)
        case_file = str(
            (candidate.get("backend_config") or {}).get("case_file") or ""
        )
        if not case_file:
            raise ValueError(f"PGLib-UC scenario missing case_file: {source_path}")
        case_path = Path(case_file)
        if not case_path.is_absolute():
            case_path = repo_root / case_path
        if not case_path.is_file():
            raise FileNotFoundError(case_path)
        case = load_case(case_path)
        candidate.setdefault("backend_config", {})[
            "task_requirements"
        ] = ordered_uc_task_requirements(
            difficulty_level=level,
            horizon_ticks=int(candidate.get("horizon_ticks") or 0),
            case=case,
        )
        source_key = original.get("source_denominator_key")
        if source_key not in (None, ""):
            candidate["backend_config"]["source_denominator_key"] = source_key
        candidate.pop("scenario_signature", None)
        candidate["scenario_signature"] = recompute_signature_with_seed(
            candidate, int(candidate.get("seed") or 42)
        )
        relative = staging_root.relative_to(repo_root) / source_path.name
        output_path = repo_root / relative
        files[output_path] = candidate
        source_key = original.get("source_denominator_key")
        if source_key in (None, ""):
            source_key = candidate.get("source_denominator_key") or (
                candidate.get("backend_config") or {}
            ).get("source_denominator_key")
        rows.append(
            {
                "scenario_id": candidate.get("scenario_id"),
                "path": relative.as_posix(),
                "domain": candidate.get("domain"),
                "backend_kind": candidate.get("backend_kind"),
                "family": candidate.get("family"),
                "difficulty_level": level,
                "scenario_signature": candidate.get("scenario_signature"),
                "source_denominator_key": source_key,
                "replaces_scenario_id": original.get("scenario_id"),
                "status": "pending_protocol21_full_admission",
                "reason_codes": [
                    "ordered_native_task_contract_added",
                    "source_bytes_unchanged",
                ],
            }
        )
    return (
        {
            "schema_version": "protocol21-pglib-uc-depth-candidates-v1",
            "status": "staging_candidates_pending_protocol21",
            "leaderboard_eligible": False,
            "release_ready": False,
            "n_candidates": len(rows),
            "scenarios": rows,
        },
        files,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", type=Path, default=DEFAULT_SOURCE_SUITE)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report, files = build(
        source_suite=args.source_suite.resolve(),
        staging_root=args.staging_root.resolve(),
        repo_root=REPO_ROOT,
    )
    if args.execute:
        for path, body in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
