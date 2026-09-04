#!/usr/bin/env python3
"""Build source-locked PGLib small-case + RTS-GMLC load candidates.

RTS-GMLC supplies only its real regional load trajectory (hourly mean of 12
five-minute periods). PGLib-OPF supplies the network, constraints, and
generator costs. The composition is labelled synthetic topology-profile
pairing: it is not a historical co-occurrence and does not claim a shared
renewable fleet. Case73 identity pairing is intentionally not reused.
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
from domains.power_grid.paired_rts_load import (  # noqa: E402
    BUS_CSV,
    EXCLUDED_DATES,
    LOAD_CSV,
    RTS_GMLC_COMMIT,
    SYNTHETIC_CASES,
    SYNTHETIC_CONTRACT,
    WINDOW_RECIPE,
    load_day_rows,
    mine_stress_days,
    parse_calendar_date,
    select_horizon_rows,
    window_sha256,
)
from domains.power_grid.seeds.from_pglib_opf import (  # noqa: E402
    build_acopf_dispatch_24h_seed,
)

DEFAULT_STAGING = REPO_ROOT / "scenarios/staging/powergrid_paired_20260815"
DEFAULT_REPORT = REPO_ROOT / "reports/powergrid_paired_20260815/candidate_report.json"
DEFAULT_CASE = "pglib_opf_case14_ieee"
CASE_FILE_BY_NAME = {
    "pglib_opf_case14_ieee": "works/PGLib-OPF/pglib_opf_case14_ieee.m",
    "pglib_opf_case30_ieee": "works/PGLib-OPF/pglib_opf_case30_ieee.m",
}


def _sha256(repo_root: Path, declared_path: str) -> str:
    return hashlib.sha256((repo_root / declared_path).read_bytes()).hexdigest()


def _reported_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _case_slug(case_name: str) -> str:
    return case_name.replace("pglib_opf_", "").replace("_ieee", "")


def _scenario_id(case_name: str, calendar_date: str) -> str:
    slug = _case_slug(case_name)
    day = calendar_date.replace("-", "_")
    return (
        "power_grid/acopf_rts_paired/deep_planning/high/"
        f"{slug}_rts_gmlc_{day}_s42"
    )


def _derived_window(
    repo_root: Path, calendar_date: str, *, horizon_ticks: int = 24
) -> dict[str, str]:
    rows = select_horizon_rows(
        load_day_rows(
            repo_root / LOAD_CSV, parse_calendar_date(calendar_date)
        ),
        horizon_ticks=horizon_ticks,
        periods_per_tick=12,
    )
    return {
        "sha256": window_sha256(
            rows, calendar_date=calendar_date, periods_per_tick=12
        ),
        "recipe_version": WINDOW_RECIPE,
    }


def _source_contract(
    repo_root: Path, *, case_file: str, calendar_date: str
) -> dict[str, Any]:
    runtime = [case_file, LOAD_CSV]
    derivation = [BUS_CSV]
    required = [*runtime, *derivation]
    return {
        "runtime_input": runtime,
        "derivation_input": derivation,
        "implementation_asset": [],
        "metadata": [],
        "license": [],
        "file_sha256s": {path: _sha256(repo_root, path) for path in required},
        "derived_window": _derived_window(repo_root, calendar_date),
        "pairing_contract": {
            "version": SYNTHETIC_CONTRACT,
            "composition": "synthetic_topology_profile",
            "scale": (
                "hourly total RTS regional load / RTS declared base MW "
                "applied to PGLib native loads"
            ),
            "source_window": f"{calendar_date} periods 1-288",
            "aggregation": "mean of 12 consecutive 5-minute values per hour",
            "claim_limit": (
                "synthetic topology-profile composition; load timeseries only; "
                "no historical pairing; no bus/branch identity claim; "
                "no renewable fleet pairing claim"
            ),
        },
    }


def _scenario(
    repo_root: Path, *, case_name: str, calendar_date: str
) -> dict[str, Any]:
    if case_name not in SYNTHETIC_CASES:
        raise ValueError(f"unsupported synthetic pairing case: {case_name}")
    if calendar_date in EXCLUDED_DATES:
        raise ValueError(f"excluded exhausted pairing date: {calendar_date}")
    case_file = CASE_FILE_BY_NAME[case_name]
    scenario_id = _scenario_id(case_name, calendar_date)
    body = build_acopf_dispatch_24h_seed(
        case_name=case_name,
        seed_id=scenario_id,
        seed=42,
        difficulty_mode="deep_planning",
        difficulty_level="high",
        backend_kind="pandapower_acopf",
        structural_seed=True,
    ).to_dict()
    body["scenario_id"] = scenario_id
    body["backend_config"]["paired_timeseries"] = {
        "contract": SYNTHETIC_CONTRACT,
        "calendar_date": calendar_date,
        "periods_per_tick": 12,
        "load_csv": LOAD_CSV,
        "bus_csv": BUS_CSV,
        "composition": "synthetic_topology_profile",
    }
    body["backend_config"]["source_denominator_key"] = (
        f"pandapower_acopf:{_case_slug(case_name)}:rts_gmlc_load_profile:{calendar_date}"
    )
    body["provenance"] = {
        "data_source": (
            f"PGLib-OPF {case_name} + RTS-GMLC regional load "
            "(synthetic topology-profile composition)"
        ),
        "files": [case_file, LOAD_CSV, BUS_CSV],
        "url": [
            "https://github.com/power-grid-lib/pglib-opf",
            "https://github.com/GridMod/RTS-GMLC",
        ],
        "commit": RTS_GMLC_COMMIT,
        "lock_strategy": "file_sha256+rts_gmlc_commit+derived_load_window_v1",
        "time_window": {
            "calendar_date": calendar_date,
            "source_periods": "1-288",
            "source_resolution_minutes": 5,
            "backend_tick_minutes": 60,
        },
        "license": "CC-BY-4.0 (PGLib-OPF) + BSD-3-Clause (RTS-GMLC)",
        "notes": (
            "RTS-GMLC real regional load supplies the 24-hour demand profile "
            "after hourly mean-of-12 aggregation. PGLib-OPF supplies the "
            "network, constraints, and generator costs. This is synthetic "
            "topology-profile composition, not a historical pairing. No "
            "RTS-GMLC renewable trajectory is consumed or claimed."
        ),
    }
    body["source_contract"] = _source_contract(
        repo_root, case_file=case_file, calendar_date=calendar_date
    )
    body["scenario_signature"] = recompute_signature_with_seed(body, 42)
    return body


def _row(
    *,
    repo_root: Path,
    staging_root: Path,
    case_name: str,
    calendar_date: str,
    selection_reason: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    body = _scenario(
        repo_root, case_name=case_name, calendar_date=calendar_date
    )
    path = staging_root / f"{body['scenario_id'].replace('/', '__')}.yaml"
    row = {
        "scenario_id": body["scenario_id"],
        "scenario_signature": body["scenario_signature"],
        "path": _reported_path(path, repo_root),
        "domain": "power_grid",
        "backend_kind": "pandapower_acopf",
        "family": body["family"],
        "difficulty_mode": body["difficulty_mode"],
        "difficulty_level": body["difficulty_level"],
        "horizon_ticks": body["horizon_ticks"],
        "seed": body["seed"],
        "source_denominator_key": body["backend_config"]["source_denominator_key"],
        "calendar_date": calendar_date,
        "case_name": case_name,
        "selection_reason": selection_reason,
        "status": "pending_protocol21_full_admission",
        "essential_gate_order": [
            "source_pair_and_hash_lock",
            "deterministic_native_bounded_probe",
            "tool_protocol_state_effect",
            "positive_native_reward_and_safe_completion",
            "high_event_response",
        ],
        "deferred_gates": ["exact_minimality", "release_coverage"],
        "claim_limit": "candidate only; no Core admission",
    }
    return row, path, body


def build(
    *,
    repo_root: Path = REPO_ROOT,
    staging_root: Path = DEFAULT_STAGING,
    case_name: str = DEFAULT_CASE,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    mining = mine_stress_days(repo_root / LOAD_CSV)
    selected = mining["selected"]
    if not selected:
        raise ValueError("window mining produced no winter stress days")
    files: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for item in selected:
        row, path, body = _row(
            repo_root=repo_root,
            staging_root=staging_root,
            case_name=case_name,
            calendar_date=str(item["calendar_date"]),
            selection_reason=str(item["selection_reason"]),
        )
        files[path] = body
        rows.append(row)
    report = {
        "schema_version": "candidate-report-v1",
        "status": "staging_candidates_pending_full_admission",
        "pipeline_version": "powergrid_essential_gates_v1",
        "candidate_only": True,
        "release_admission": False,
        "core_admission_claimed": False,
        "case_name": case_name,
        "rts_gmlc_commit": RTS_GMLC_COMMIT,
        "rts_gmlc_retagged": False,
        "excluded_dates": sorted(EXCLUDED_DATES),
        "composition": "synthetic_topology_profile",
        "window_mining": mining,
        "n_candidates": len(rows),
        "scenarios": rows,
    }
    return report, files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--case-name", default=DEFAULT_CASE, choices=sorted(SYNTHETIC_CASES))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report, files = build(
        repo_root=REPO_ROOT,
        staging_root=args.staging_root.resolve(),
        case_name=args.case_name,
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
