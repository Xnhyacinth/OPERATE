#!/usr/bin/env python3
"""Calibrate source-locked SimBench windows before any core promotion.

The report is a staging artifact.  A live backend and real SimBench profiles
are necessary but not sufficient: every candidate must also pass the same
native state-changing leverage gate used by the main Core calibration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.power_grid.seeds.from_cigre import (  # noqa: E402
    _DISTRIBUTION_NETWORKS,
    build_distribution_volt_var_seed,
)
from domains.power_grid.seeds.source_locks import SOURCE_LOCKS  # noqa: E402
from run import run_one  # noqa: E402
from scripts.calibrate_core_candidate import _classify_result  # noqa: E402

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "simbench_candidate_calibration.json"
)
NETWORKS = (
    "simbench:1-LV-rural1--0-sw",
    "simbench:1-MV-rural--0-sw",
    "simbench:1-MV-semiurb--0-sw",
)
WINDOWS = (
    ("basic", 0),
    ("medium", 96 * 90),
    ("high", 96 * 180),
    ("extreme", 96 * 270),
)
AGENTS = ("wait_only", "greedy_heuristic", "oracle_offline")


def candidate_id(network: str, difficulty: str, profile_start_index: int) -> str:
    code = network.split(":", 1)[1].replace("--", "_").replace("-", "_")
    return f"simbench_{code}_{difficulty}_p{profile_start_index}"


def scenario_id(candidate: str, family: str, difficulty: str) -> str:
    return f"power_grid/{family}/time_pressure/{difficulty}/{candidate}"


def runtime_profile_consumption_observed(raw_episode: dict[str, Any]) -> bool:
    """Return whether a completed run emitted the native SimBench trace.

    Provenance and scenario configuration identify an intended input, but only
    the post-episode runtime event proves that the active backend consumed its
    profile window.  This output is audit-only and is never agent-visible.
    """
    summary = raw_episode.get("ground_truth_summary") or {}
    events = summary.get("realized_events") if isinstance(summary, dict) else None
    if not isinstance(events, list):
        return False
    return any(
        isinstance(event, dict)
        and event.get("type") == "simbench_profile_window_started"
        and isinstance(event.get("profile_start_index"), int)
        and isinstance(event.get("profile_step"), int)
        for event in events
    )


def classify(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized["checks"] = {
        "deterministic_wait_replay": bool(result["deterministic_wait_replay"]),
        "source_profile_consumed": bool(result["source_profile_consumed"]),
    }
    normalized["episodes"] = {
        name: {
            **episode,
            "cost": float(episode["actual_cost"]),
            "successful_state_changing_calls": (
                int(episode["n_control_calls"])
                if bool(episode["outcome_changed"])
                else 0
            ),
        }
        for name, episode in result["episodes"].items()
    }
    classified = _classify_result(normalized)
    classified["core_eligible"] = classified["status"] == "passed"
    return classified


def _episode(seed: dict[str, Any], agent: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = run_one(seed, agent_name=agent)
    impact = raw.get("decision_impact") or {}
    summary = raw.get("trajectory_summary") or {}
    score = raw["score"]
    native_dimension_names = {
        "system_survival",
        "economic_cost",
        "safety_violation",
        "optimality_gap",
        "counterfactual_prevention",
    }
    return raw, {
        "total_score": round(float(score["total_score"]), 9),
        "raw_total": round(float(score["raw_total"]), 9),
        "actual_cost": round(float(raw["counterfactual"]["actual_cost"]), 9),
        "prevented_loss": round(float(raw["counterfactual"]["prevented_loss"]), 9),
        "n_ticks": int(summary.get("n_ticks", 0) or 0),
        "n_tool_calls": int(summary.get("n_tool_calls", 0) or 0),
        "n_control_calls": int(impact.get("n_control_calls", 0) or 0),
        "outcome_changed": bool(impact.get("outcome_changed", False)),
        "native_dimension_scores": {
            str(dimension["name"]): round(float(dimension["raw_score"]), 9)
            for dimension in score.get("dimensions") or []
            if dimension.get("name") in native_dimension_names
            and bool(dimension.get("applicable"))
        },
    }


def calibrate_one(network: str, difficulty: str, profile_start_index: int) -> dict[str, Any]:
    seed_obj = build_distribution_volt_var_seed(
        seed_id=candidate_id(network, difficulty, profile_start_index),
        network=network,
        difficulty_level=difficulty,
        profile_start_index=profile_start_index,
    )
    seed = seed_obj.to_dict()
    wait_raw_a, wait_a = _episode(seed, "wait_only")
    wait_raw_b, wait_b = _episode(seed, "wait_only")
    episodes = {"wait_only": wait_a}
    for agent in AGENTS[1:]:
        _, episodes[agent] = _episode(seed, agent)
    source_profile_consumed = (
        runtime_profile_consumption_observed(wait_raw_a)
        and runtime_profile_consumption_observed(wait_raw_b)
    )
    return classify(
        {
            "candidate_id": candidate_id(network, difficulty, profile_start_index),
            "network": network,
            "family": str(seed["family"]),
            "difficulty_level": difficulty,
            "profile_start_index": profile_start_index,
            "scenario_signature": seed_obj.signature(),
            "deterministic_wait_replay": wait_a == wait_b,
            "source_profile_consumed": source_profile_consumed,
            "episodes": episodes,
        }
    )


def _write(path: Path, results: dict[str, dict[str, Any]], expected: int) -> dict[str, Any]:
    rows = [results[key] for key in sorted(results)]
    for row in rows:
        family = str(
            row.get("family")
            or _DISTRIBUTION_NETWORKS[str(row["network"])]["family"]
        )
        row["family"] = family
        row["scenario_id"] = scenario_id(
            str(row["candidate_id"]),
            family,
            str(row["difficulty_level"]),
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lock = SOURCE_LOCKS["simbench"]
    report = {
        "schema_version": "0.3",
        "scope": "simbench_source_locked_behavioral_staging",
        "status": "complete" if len(rows) == expected else "partial",
        "release_membership_changed": False,
        "source_lock": {
            "url": lock.url,
            "tag": lock.version,
            "commit": lock.commit,
            "lock_strategy": lock.lock_strategy,
        },
        "gate_policy": {
            "policy": "native_state_changing_leverage_v3",
            "requires_deterministic_replay": True,
            "requires_state_changing_control": True,
            "requires_real_profile_consumption": True,
            "minimum_relative_cost_gap": 0.005,
            "minimum_prevented_loss": 1.0,
            "minimum_native_dimension_improvement": 1.0,
            "maximum_critical_native_regression": 5.0,
        },
        "n_expected": expected,
        "n_completed": len(rows),
        "n_core_eligible": sum(bool(row["core_eligible"]) for row in rows),
        "status_counts": dict(sorted(counts.items())),
        "results": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
    return report


def calibrate(output: Path, *, limit: int | None = None, resume: bool = True) -> dict[str, Any]:
    candidates = [(network, level, start) for network in NETWORKS for level, start in WINDOWS]
    if limit is not None:
        candidates = candidates[:limit]
    results: dict[str, dict[str, Any]] = {}
    if resume and output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        desired = {candidate_id(*row) for row in candidates}
        results = {
            row["candidate_id"]: row
            for row in prior.get("results", [])
            if row.get("candidate_id") in desired
        }
    for network, level, start in candidates:
        key = candidate_id(network, level, start)
        if key in results:
            continue
        results[key] = calibrate_one(network, level, start)
        _write(output, results, len(candidates))
    return _write(output, results, len(candidates))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    report = calibrate(args.output.resolve(), limit=args.limit, resume=not args.no_resume)
    print(
        json.dumps(
            {key: report[key] for key in ("status", "n_expected", "n_completed", "n_core_eligible", "status_counts")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
