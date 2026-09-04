#!/usr/bin/env python3
"""Build the source-locked PGLib case73 + RTS-GMLC load candidate.

The pairing is deliberately narrow: RTS-GMLC supplies only its real regional
load trajectory. PGLib-OPF supplies the matching 73-bus network, constraints,
and generator cost/capability data. Renewable trajectories are not claimed
because the two source packages do not expose the same generator fleet.
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
from domains.power_grid.seeds.from_pglib_opf import (  # noqa: E402
    build_acopf_dispatch_24h_seed,
)

DEFAULT_STAGING = REPO_ROOT / "scenarios/staging/powergrid_case73_paired_20260814"
DEFAULT_REPORT = REPO_ROOT / "reports/powergrid_case73_paired_20260814/candidate_report.json"
CASE_FILE = "works/PGLib-OPF/pglib_opf_case73_ieee_rts.m"
LOAD_CSV = (
    "works/RTS-GMLC/RTS_Data/timeseries_data_files/Load/"
    "REAL_TIME_regional_Load.csv"
)
BUS_CSV = "works/RTS-GMLC/RTS_Data/SourceData/bus.csv"
BRANCH_CSV = "works/RTS-GMLC/RTS_Data/SourceData/branch.csv"
SCENARIO_ID = (
    "power_grid/acopf_rts_paired/deep_planning/high/"
    "case73_rts_gmlc_2020_07_20_s42"
)


def _sha256(repo_root: Path, declared_path: str) -> str:
    return hashlib.sha256((repo_root / declared_path).read_bytes()).hexdigest()


def _reported_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _source_contract(repo_root: Path) -> dict[str, Any]:
    runtime = [CASE_FILE, LOAD_CSV]
    derivation = [BUS_CSV, BRANCH_CSV]
    required = [*runtime, *derivation]
    return {
        "runtime_input": runtime,
        "derivation_input": derivation,
        "implementation_asset": [],
        "metadata": [],
        "license": [],
        "file_sha256s": {path: _sha256(repo_root, path) for path in required},
        "pairing_contract": {
            "version": "rts_gmlc_case73_regional_load_v1",
            "case_identity_basis": [
                "73 ordered bus identifiers and base active loads",
                "ordered branch endpoints, impedance, and three ratings",
            ],
            "source_window": "2020-07-20 periods 1-288",
            "aggregation": "mean of 12 consecutive 5-minute values per hour",
            "claim_limit": "load timeseries only; no renewable fleet pairing claim",
        },
    }


def _scenario(repo_root: Path) -> dict[str, Any]:
    body = build_acopf_dispatch_24h_seed(
        case_name="pglib_opf_case73_ieee_rts",
        seed_id=SCENARIO_ID,
        seed=42,
        difficulty_mode="deep_planning",
        difficulty_level="high",
        backend_kind="pandapower_acopf",
        structural_seed=True,
    ).to_dict()
    body["scenario_id"] = SCENARIO_ID
    body["backend_config"]["paired_timeseries"] = {
        "contract": "rts_gmlc_case73_regional_load_v1",
        "calendar_date": "2020-07-20",
        "periods_per_tick": 12,
        "load_csv": LOAD_CSV,
        "bus_csv": BUS_CSV,
        "branch_csv": BRANCH_CSV,
    }
    body["backend_config"]["source_denominator_key"] = (
        "pandapower_acopf:pglib_case73:rts_gmlc_load:2020-07-20"
    )
    body["provenance"] = {
        "data_source": "PGLib-OPF case73 + RTS-GMLC regional load",
        "files": [CASE_FILE, LOAD_CSV, BUS_CSV, BRANCH_CSV],
        "url": [
            "https://github.com/power-grid-lib/pglib-opf",
            "https://github.com/GridMod/RTS-GMLC",
        ],
        "lock_strategy": "file_sha256+physical_case_pairing_v1",
        "time_window": {
            "calendar_date": "2020-07-20",
            "source_periods": "1-288",
            "source_resolution_minutes": 5,
            "backend_tick_minutes": 60,
        },
        "license": "CC-BY-4.0 (PGLib-OPF) + BSD-3-Clause (RTS-GMLC)",
        "notes": (
            "The real RTS-GMLC regional load drives each AC-OPF tick after "
            "an exact bus/load and branch-parameter pairing check. PGLib "
            "case73 remains the source of network and generator constraints. "
            "No RTS-GMLC renewable trajectory is consumed or claimed."
        ),
    }
    body["source_contract"] = _source_contract(repo_root)
    body["scenario_signature"] = recompute_signature_with_seed(body, 42)
    return body


def build(
    *,
    repo_root: Path = REPO_ROOT,
    staging_root: Path = DEFAULT_STAGING,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    body = _scenario(repo_root)
    path = staging_root / f"{SCENARIO_ID.replace('/', '__')}.yaml"
    row = {
        "scenario_id": SCENARIO_ID,
        "scenario_signature": body["scenario_signature"],
        "path": _reported_path(path, repo_root),
        "domain": "power_grid",
        "backend_kind": "pandapower_acopf",
        "family": body["family"],
        "difficulty_mode": body["difficulty_mode"],
        "difficulty_level": body["difficulty_level"],
        "horizon_ticks": body["horizon_ticks"],
        "seed": body["seed"],
        "source_denominator_key": body["backend_config"][
            "source_denominator_key"
        ],
        "status": "pending_protocol21_full_admission",
        "essential_gate_order": [
            "source_pair_and_hash_lock",
            "deterministic_native_bounded_probe",
            "tool_protocol_state_effect",
            "positive_native_reward_and_safe_completion",
            "high_event_response",
        ],
        "deferred_gates": ["exact_minimality", "release_coverage"],
        "claim_limit": "candidate only until repaired quality_core_v2 replay passes",
    }
    report = {
        "schema_version": "candidate-report-v1",
        "status": "staging_candidates_pending_full_admission",
        "pipeline_version": "powergrid_essential_gates_v1",
        "n_candidates": 1,
        "scenarios": [row],
    }
    return report, {path: body}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report, files = build(
        repo_root=REPO_ROOT,
        staging_root=args.staging_root.resolve(),
    )
    if args.execute:
        for path, body in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
