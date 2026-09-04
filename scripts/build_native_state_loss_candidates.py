#!/usr/bin/env python3
"""Build source-locked Power Grid native state-loss staging candidates.

The source network remains the public pandapower asset.  The severe load
excursion is a deterministic procedural overlay and is intentionally labelled
as such; it is not claimed as a source-observed event.  Candidates remain
staging-only until the ordinary source, task, headroom, depth, replay, and
agentic gates pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.power_grid.seeds.from_cigre import (  # noqa: E402
    build_distribution_volt_var_seed,
)
from runner.resume import recompute_signature_with_seed  # noqa: E402
from scripts.prepare_protocol21_working_set import _source_contract  # noqa: E402

DEFAULT_STAGING = REPO_ROOT / "scenarios/staging/v0_52_protocol21_native_state_loss"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release/dt_sched_bench_v0_52_0_candidate"
    / "native_state_loss_candidates_v1.json"
)


DEFAULT_CASES: tuple[dict[str, Any], ...] = (
    {
        "network": "cigre_mv_with_der_all",
        "level": "high",
        "seed": 73,
        "event_tick": 8,
        "duration_ticks": 5,
        "intensity": 1.0,
        "cleanup_tick": 14,
        "slug": "cigre_high_s73",
    },
    {
        "network": "mv_oberrhein",
        "level": "extreme",
        "seed": 74,
        "event_tick": 12,
        "duration_ticks": 6,
        "intensity": 0.70,
        # The native oracle observes the first post-event state at tick
        # event_tick+4 and releases the capacitive bank at tick 29.  Keep a
        # real recovery window instead of declaring an impossible tick-19
        # milestone that would reject otherwise effective control.
        "cleanup_tick": 27,
        "slug": "oberrhein_extreme_s74",
    },
)


def _source_key(body: dict[str, Any]) -> str:
    provenance = dict(body.get("provenance") or {})
    return json.dumps(
        {
            "backend": body["backend_kind"],
            "network": body["backend_config"]["network"],
            "source_files": provenance.get("files") or [],
            "source_commit": provenance.get("commit"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _build_body(case: dict[str, Any]) -> dict[str, Any]:
    network = str(case["network"])
    level = str(case["level"])
    seed = int(case["seed"])
    event_tick = int(case["event_tick"])
    duration = int(case["duration_ticks"])
    cleanup_tick = int(case["cleanup_tick"])
    horizon = 30 if level == "high" else 42
    seed_id = (
        "power_grid/distribution_volt_var_native_state_loss/"
        f"deep_planning/{level}/native_{case['slug']}"
    )
    seed_obj = build_distribution_volt_var_seed(
        seed_id=seed_id,
        seed=seed,
        difficulty_mode="deep_planning",
        difficulty_level=level,
        network=network,
    )
    body = seed_obj.to_dict()
    body["seed_id"] = seed_id
    body["scenario_id"] = seed_id
    body["horizon_ticks"] = horizon
    body["difficulty_level"] = level

    event = {
        "kind": "load_surge",
        "trigger_tick": event_tick,
        "duration_ticks": duration,
        "hidden": level == "extreme",
        "target": {},
        "intensity": float(case["intensity"]),
        "notes": (
            "Deterministic all-feeder stress overlay anchored to the locked "
            "pandapower network; not a source-observed event."
        ),
    }
    # Keep the candidate's task-driving stress first and retain the
    # source-seed perturbations as independent background events.  This makes
    # the response window unambiguous in the serialized contract.
    body["perturbations"] = [event, *body.get("perturbations", [])]

    config = copy.deepcopy(body.get("backend_config") or {})
    if network == "mv_oberrhein":
        # The Oberrhein reference controller safely mitigates the procedural
        # surge with three source-specific native stages: reserve commitment,
        # reactive support, and a later capacitor transition. The stages are
        # intentionally separated across ticks so Extreme has a real
        # cross-tick dependency rather than a same-tick tool bundle.
        task_milestones = [
            {
                "tool": "commit_reserve",
                "not_before_tick": event_tick + 1,
                "not_after_tick": event_tick + 4,
            },
            {
                "tool": "set_der_reactive_power",
                "not_before_tick": event_tick + 4,
                "not_after_tick": event_tick + 10,
            },
            {
                "tool": "switch_capacitor",
                "not_before_tick": cleanup_tick,
                "not_after_tick": cleanup_tick + 3,
            },
        ]
        required_native_actions = [
            {
                "tick": event_tick + 1,
                "tool": "commit_reserve",
            },
            {
                "tick": event_tick + 4,
                "tool": "set_der_reactive_power",
            },
            {
                "tick": cleanup_tick,
                "tool": "switch_capacitor",
            },
        ]
    else:
        task_milestones = [
            {
                "tool": "switch_capacitor",
                "args": {"cap_id": 0, "status": True},
                "not_before_tick": event_tick + 1,
                "not_after_tick": event_tick + 4,
            },
            {
                "tool": "set_der_reactive_power",
                "not_before_tick": cleanup_tick,
                "not_after_tick": cleanup_tick + 3,
            },
        ]
        required_native_actions = [
            {
                "tick": event_tick + 1,
                "tool": "switch_capacitor",
                "args": {"cap_id": 0, "status": True},
            },
            {
                "tick": event_tick + 1,
                "tool": "switch_capacitor",
                "args": {"cap_id": 1, "status": True},
            },
            {
                "tick": cleanup_tick,
                "tool": "switch_capacitor",
                "args": {"cap_id": 0, "status": False},
            },
            {
                "tick": cleanup_tick,
                "tool": "switch_capacitor",
                "args": {"cap_id": 1, "status": False},
            },
        ]

    config.update(
        {
            "volt_var_controls": True,
            "release_ready": False,
            "release_reentry_ready": False,
            "source_integration_rung": "staging_native_state_loss_v1",
            "source_denominator_key": _source_key(body),
            "procedural_stress": {
                "origin": "deterministic_overlay",
                "source_observed": False,
                "source_anchor": "native_feeder_load_state",
                "event_kind": "load_surge",
                "event_tick": event_tick,
                "duration_ticks": duration,
                "intensity": float(case["intensity"]),
                "native_loss_required": True,
            },
            "response_window_recipe": {
                "version": "native_state_loss_response_v1",
                "event_trigger_tick": event_tick,
                "response_opportunity_tick": event_tick + 1,
                "response_window_end_tick": event_tick + duration - 1,
                "cleanup_tick": cleanup_tick,
                "later_decision_opportunity": True,
                "simulator_advances_time": True,
            },
            "native_state_loss_task": {
                "contract": "power_grid.native_state_loss.v1",
                "required_task_loss_keys": [
                    "voltage_violation_cost",
                    "overload_cost",
                    "shed_penalty",
                    "disconnection_cost",
                ],
                "production_cost_only_rejected": True,
                "minimum_baseline_task_loss": 100.0,
                "minimum_task_loss_reduction": 100.0,
                "response_window_required": True,
                "source_vs_procedural_separation": True,
            },
            "task_requirements": {
                "min_distinct_control_ticks": 2 if level == "high" else 3,
                "min_distinct_physical_tools": (
                    2 if level in {"high", "extreme"} else 1
                ),
                "ordered_tool_milestones": task_milestones,
                "required_native_actions": required_native_actions,
            },
        }
    )
    body["backend_config"] = config
    body["source_contract"] = _source_contract(body)
    body["complexity_metrics"] = seed_obj.complexity_metrics()
    body["complexity_metrics"].update(
        {
            "native_task_loss_required": True,
            "response_window_ticks": [event_tick + 1, event_tick + duration - 1],
            "procedural_stress_count": 1,
        }
    )
    body["scenario_signature"] = recompute_signature_with_seed(body, seed)
    return body


def _row(body: dict[str, Any], relative_path: str) -> dict[str, Any]:
    config = dict(body.get("backend_config") or {})
    return {
        "scenario_id": body["scenario_id"],
        "path": relative_path,
        "domain": body["domain"],
        "backend_kind": body["backend_kind"],
        "family": body["family"],
        "difficulty_mode": body["difficulty_mode"],
        "difficulty_level": body["difficulty_level"],
        "horizon_ticks": body["horizon_ticks"],
        "seed": body["seed"],
        "scenario_signature": body["scenario_signature"],
        "source_denominator_key": config["source_denominator_key"],
        "status": "pending_protocol21_full_admission",
        "reason_codes": [
            "source_locked_native_feeder",
            "procedural_stress_overlay_explicit",
            "native_task_loss_required",
            "native_constructor_runtime_trace_proven",
            "requires_behavior_task_depth_replay_gates",
        ],
    }


def build(
    *,
    repo_root: Path = REPO_ROOT,
    staging_root: Path = DEFAULT_STAGING,
    cases: Iterable[dict[str, Any]] = DEFAULT_CASES,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    del repo_root
    files: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for case in cases:
        body = _build_body(dict(case))
        relative = staging_root / f"{body['scenario_id'].replace('/', '__')}.yaml"
        files[relative] = body
        rows.append(_row(body, str(relative)))
    source_keys = [row["source_denominator_key"] for row in rows]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("native state-loss source keys must be independent")
    return (
        {
            "schema_version": "protocol21-native-state-loss-candidates-v1",
            "status": "staging_native_state_loss_candidates_pending_full_admission",
            "leaderboard_eligible": False,
            "release_ready": False,
            "n_candidates": len(rows),
            "source_key_count": len(source_keys),
            "scenarios": rows,
        },
        files,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report, files = build(staging_root=args.staging_root.resolve())
    if args.execute:
        for path, body in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
