#!/usr/bin/env python3
"""Build a fail-closed Traffic Protocol21 low-coverage report.

The report only promotes rows backed by a native runtime graph, native control
readback, a nonterminal response window, deterministic paired replay, and
positive headroom.  Diagnostic miner outputs are retained as blockers when a
source has no legal native control or regresses against its paired baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "traffic_protocol21_low_coverage_v1"
NATIVE_ROOT = (
    REPO_ROOT
    / "release/dt_sched_bench_v0_52_0_candidate/protocol21_expansion_trials"
    / "traffic_sumo365_native_v1"
)
DEFAULT_MINER_DIR = NATIVE_ROOT / "miner_snapshot"
DEFAULT_HIGH_SOURCE_GROUNDED = (
    REPO_ROOT
    / "release/dt_sched_bench_v0_52_0_candidate/protocol21_expansion_trials"
    / "traffic_resco_cologne1_high_v1/current_replay"
    / "source_grounded_protocol2_v21.json"
)
DEFAULT_HIGH_STRATEGY = (
    REPO_ROOT
    / "release/dt_sched_bench_v0_52_0_candidate/protocol21_expansion_trials"
    / "traffic_resco_cologne1_high_v1/current_replay"
    / "strategy_depth_protocol2_v21.json"
)
DEFAULT_OUTPUT = NATIVE_ROOT / "low_coverage_report_protocol21.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _graph_asset_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    graph = source.get("source_assets") or {}
    rows: list[dict[str, Any]] = []
    for key in ("sumocfg", "network"):
        row = graph.get(key)
        if isinstance(row, dict):
            rows.append(row)
    for key in ("route_files", "additional_files", "recursive_inputs"):
        rows.extend(row for row in graph.get(key, []) if isinstance(row, dict))
    return rows


def _validate_source_graphs(source_document: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in source_document.get("results", [])
        if isinstance(row, dict)
    ]
    identities = [str(row.get("complete_source_identity_sha256") or "") for row in rows]
    graph_checks: list[dict[str, Any]] = []
    for source in rows:
        assets = _graph_asset_rows(source)
        mismatches = []
        for asset in assets:
            path = Path(str(asset.get("path") or ""))
            expected = str(asset.get("sha256") or "")
            if not path.is_file():
                mismatches.append({"path": str(path), "reason": "asset_missing"})
            elif _sha256(path) != expected:
                mismatches.append({"path": str(path), "reason": "asset_hash_mismatch"})
        graph_checks.append(
            {
                "service_date": source.get("service_date"),
                "source_identity": source.get("complete_source_identity_sha256"),
                "asset_count": len(assets),
                "status": "passed" if not mismatches else "held",
                "mismatches": mismatches,
            }
        )
    return {
        "n_source_identities": len(set(identity for identity in identities if identity)),
        "n_source_rows": len(rows),
        "duplicate_identity_count": len(identities) - len(set(identities)),
        "asset_graph_checks": graph_checks,
        "all_asset_graphs_match": bool(rows)
        and all(row["status"] == "passed" for row in graph_checks),
    }


def _load_high_difficulty(
    source_grounded_path: Path, strategy_path: Path
) -> dict[str, Any]:
    source = _read_json(source_grounded_path)
    rows = [row for row in source.get("held", []) if isinstance(row, dict)]
    high = next(
        (
            row
            for row in rows
            if str(row.get("scenario_id") or "").endswith("resco_cologne1_phase_control_high_s9411")
        ),
        {},
    )
    strategy = _read_json(strategy_path)
    sample: dict[str, Any] = next(iter(strategy.get("samples", [])), {})
    difficulty = high.get("difficulty_evidence") or {}
    return {
        "scenario_id": high.get("scenario_id"),
        "status": high.get("status", "missing"),
        "failed_gates": high.get("failed_gates", []),
        "exact_dependency_depth": difficulty.get("exact_dependency_depth"),
        "required_tools": difficulty.get("required_tools", []),
        "plan_reversal_count": difficulty.get("plan_reversal_count"),
        "strategy_disposition": sample.get("disposition"),
        "blockers": [
            "difficulty_proof_plan_reversal_below_high_floor"
            if difficulty.get("plan_reversal_count") == 0
            and "difficulty_proof" in (high.get("failed_gates") or [])
            else None,
            "high_source_grounded_row_missing" if not high else None,
        ],
    }


def build_report(
    *,
    miner_dir: Path = DEFAULT_MINER_DIR,
    high_source_grounded_path: Path = DEFAULT_HIGH_SOURCE_GROUNDED,
    high_strategy_path: Path = DEFAULT_HIGH_STRATEGY,
) -> dict[str, Any]:
    summary = _read_json(miner_dir / "summary.json")
    source_identities = _read_json(miner_dir / "source_identities.json")
    crosscheck = _read_json(miner_dir / "source_identity_crosscheck.json")
    trials = _read_json(miner_dir / "headroom_trials.json")
    retirements = _read_json(miner_dir / "retirement_ledger.json")
    graph = _validate_source_graphs(source_identities)
    high = _load_high_difficulty(high_source_grounded_path, high_strategy_path)

    trial_rows = [row for row in trials.get("results", []) if isinstance(row, dict)]
    retirement_rows = [
        row for row in retirements.get("results", []) if isinstance(row, dict)
    ]
    release_positive = [
        row
        for row in trial_rows
        if row.get("status") == "passed"
        and row.get("release_candidate_status") == "passed"
    ]
    blocker_counts = Counter(
        str(row.get("reason_code") or "unknown")
        for row in [*trial_rows, *retirement_rows]
        if row.get("reason_code")
    )
    nested_blockers = Counter(
        str(code)
        for row in trial_rows
        for code in row.get("reason_codes", [])
        if code and code != row.get("reason_code")
    )
    response_windows_passed = bool(trial_rows) and all(
        row.get("world_evolution", {})
        .get("response_window", {})
        .get("status")
        == "passed"
        and row.get("world_evolution", {}).get("post_change_decision_observed") is True
        and row.get("world_evolution", {}).get("tail_seconds_observed") is True
        for row in trial_rows
    )
    deterministic_replay_passed = bool(trial_rows) and all(
        row.get("deterministic_replay", {}).get("status") == "passed"
        and row.get("baseline_repeat_deterministic") is True
        and row.get("reference_repeat_deterministic") is True
        for row in trial_rows
    )
    source_identity_passed = bool(
        crosscheck.get("scope_kind") == "bounded_request"
        and crosscheck.get("all_match") is True
        and not crosscheck.get("missing_service_dates")
        and not crosscheck.get("unexpected_service_dates")
    )
    checks = {
        "source_identity_crosscheck_passed": source_identity_passed,
        "runtime_asset_graphs_match": graph["all_asset_graphs_match"],
        "native_replay_attempted": bool(trial_rows),
        "response_windows_passed": response_windows_passed,
        "deterministic_replay_passed": deterministic_replay_passed,
        "no_release_positive_without_headroom": not bool(
            [row for row in trial_rows if row.get("status") == "passed"
        ])
        or bool(release_positive),
        "high_candidate_has_no_unresolved_gate": not bool(high.get("failed_gates")),
    }
    candidate_ids = [str(row.get("service_date")) for row in release_positive]
    blockers = sorted(
        {
            *blocker_counts,
            *nested_blockers,
            *[value for value in high.get("blockers", []) if value],
            *(
                ["source_identity_crosscheck_failed"]
                if not source_identity_passed
                else []
            ),
            *(
                ["runtime_asset_graph_hash_mismatch"]
                if not graph["all_asset_graphs_match"]
                else []
            ),
        }
    )
    status = "candidate_ready" if candidate_ids and all(checks.values()) else "blocked"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "candidate_count": len(candidate_ids),
        "candidate_ids": candidate_ids,
        "leaderboard_eligible": False,
        "core_mutation": "none",
        "checks": checks,
        "blockers": blockers,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "nested_blocker_counts": dict(sorted(nested_blockers.items())),
        "native_miner_summary": {
            "status": summary.get("status"),
            "primary_dates": summary.get("primary_dates", []),
            "n_headroom_trials": len(trial_rows),
            "n_release_positive": len(release_positive),
            "n_retirements": len(retirement_rows),
        },
        "asset_graph_audit": graph,
        "high_extreme_audit": high,
        "evidence_bindings": {
            "summary_sha256": _sha256(miner_dir / "summary.json"),
            "source_identities_sha256": _sha256(miner_dir / "source_identities.json"),
            "source_identity_crosscheck_sha256": _sha256(
                miner_dir / "source_identity_crosscheck.json"
            ),
            "headroom_trials_sha256": _sha256(miner_dir / "headroom_trials.json"),
            "retirement_ledger_sha256": _sha256(miner_dir / "retirement_ledger.json"),
            "high_source_grounded_sha256": _sha256(high_source_grounded_path),
            "high_strategy_depth_sha256": _sha256(high_strategy_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--miner-dir", type=Path, default=DEFAULT_MINER_DIR)
    parser.add_argument("--high-source-grounded", type=Path, default=DEFAULT_HIGH_SOURCE_GROUNDED)
    parser.add_argument("--high-strategy", type=Path, default=DEFAULT_HIGH_STRATEGY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        miner_dir=args.miner_dir,
        high_source_grounded_path=args.high_source_grounded,
        high_strategy_path=args.high_strategy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["status"] == "candidate_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
