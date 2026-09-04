#!/usr/bin/env python3
"""Build source-locked Pymgrid-family native state-loss candidates.

The EMS backend consumes a locked NSRDB/OEDI profile window at runtime.  The
candidate events below are deterministic, source-anchored stress overlays;
they are not presented as source-observed outages.  The builder only emits
staging rows.  Behavioral, task, depth, replay, and model gates remain
necessary before any release admission.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.microgrid.seeds.from_pymgrid import (  # noqa: E402
    build_microgrid_economic_dispatch_24h_seed,
)
from domains.microgrid.seeds.schema import Perturbation  # noqa: E402
from runner.resume import recompute_signature_with_seed  # noqa: E402
from scripts.prepare_protocol21_working_set import _source_contract  # noqa: E402

DEFAULT_STAGING = (
    REPO_ROOT / "scenarios" / "staging" / "v0_52_microgrid_native_state_loss"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "microgrid_native_state_loss_candidates_v1.json"
)

DEFAULT_CASES: tuple[dict[str, Any], ...] = (
    {
        "level": "high",
        "seed": 57,
        "site": "miami_fl",
        "event_tick": 4,
        "duration_ticks": 4,
        "load_spike_intensity": 0.30,
        "slug": "miami_high_s57",
    },
    {
        "level": "extreme",
        "seed": 59,
        "site": "denver_co",
        "event_tick": 3,
        "duration_ticks": 6,
        "load_spike_intensity": 0.50,
        "slug": "denver_extreme_s59",
    },
    {
        "level": "high",
        "seed": 61,
        "site": "chicago_il",
        "event_tick": 4,
        "duration_ticks": 4,
        "load_spike_intensity": 0.35,
        "slug": "chicago_high_s61",
    },
    {
        "level": "extreme",
        "seed": 63,
        "site": "columbus_oh",
        "event_tick": 3,
        "duration_ticks": 8,
        "load_spike_intensity": 0.55,
        "slug": "columbus_extreme_s63",
    },
)


def _source_key(body: dict[str, Any]) -> str:
    config = dict(body.get("backend_config") or {})
    recipe = dict(config.get("derivation_recipe") or {})
    provenance = dict(body.get("provenance") or {})
    return json.dumps(
        {
            "backend": body["backend_kind"],
            "site": config.get("site"),
            "profile_start_index": recipe.get("profile_start_index"),
            "source_window_sha256": recipe.get("source_window_sha256"),
            "source_files": provenance.get("files") or [],
            "source_commit": provenance.get("commit"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _events(case: dict[str, Any]) -> list[Perturbation]:
    level = str(case["level"])
    event_tick = int(case["event_tick"])
    duration = int(case["duration_ticks"])
    events = [
        Perturbation(
            kind="grid_outage",
            trigger_tick=event_tick,
            duration_ticks=duration,
            hidden=True,
            target={},
            intensity=1.0,
            notes=(
                "Deterministic PCC outage over a locked NREL/OEDI EMS window; "
                "procedural stress overlay, not a source-observed outage."
            ),
        ),
        Perturbation(
            kind="load_spike",
            trigger_tick=event_tick + 2,
            duration_ticks=2,
            hidden=False,
            target={},
            intensity=float(case["load_spike_intensity"]),
            notes=(
                "Deterministic demand excursion on the native EMS profile; "
                "the source profile remains the runtime state driver."
            ),
        ),
    ]
    if level == "extreme":
        events.append(
            Perturbation(
                kind="der_failure",
                trigger_tick=event_tick + 4,
                duration_ticks=max(2, duration - 2),
                hidden=True,
                target={"der_index": int(case["seed"]) % 5},
                intensity=1.0,
                notes=(
                    "Deterministic DER failure during the outage; hidden state "
                    "loss requiring a later native recovery decision."
                ),
            )
        )
    return events


def _build_body(case: dict[str, Any]) -> dict[str, Any]:
    level = str(case["level"])
    seed = int(case["seed"])
    event_tick = int(case["event_tick"])
    duration = int(case["duration_ticks"])
    horizon = 24
    seed_id = (
        "microgrid/microgrid_economic_dispatch_24h/"
        f"deep_planning/{level}/native_state_loss_{case['slug']}"
    )
    seed_obj = build_microgrid_economic_dispatch_24h_seed(
        seed=seed,
        seed_id=seed_id,
        difficulty_level=level,
        difficulty_mode="deep_planning",
        site=str(case["site"]),
        source_profile_start_index=(
            int(case["start_index"]) if case.get("start_index") is not None else None
        ),
    )
    seed_obj.perturbations = _events(case)
    body = seed_obj.to_dict()
    body["scenario_id"] = seed_id
    body["seed_id"] = seed_id
    body["horizon_ticks"] = horizon
    body["difficulty_level"] = level

    config = copy.deepcopy(body.get("backend_config") or {})
    recipe = dict(config.get("derivation_recipe") or {})
    preposition_tick = max(0, event_tick - 1)
    recovery_tick = event_tick + 1
    restoration_tick = event_tick + duration - 1
    restoration_effect_tick = restoration_tick + 1
    required_tools = [
        "dispatch_genset",
        "set_battery_dispatch",
        "connect_pcc",
    ]
    milestones = [
        {
            "tool": "dispatch_genset",
            "not_before_tick": preposition_tick,
            "not_after_tick": event_tick,
        },
        {
            "tool": "set_battery_dispatch",
            "not_before_tick": recovery_tick,
            "not_after_tick": event_tick + 3,
        },
        {
            "tool": "connect_pcc",
            "not_before_tick": restoration_effect_tick,
            "not_after_tick": restoration_effect_tick,
        },
    ]
    config.update(
        {
            "release_ready": False,
            "release_reentry_ready": False,
            "source_integration_rung": "staging_native_state_loss_ems_v1",
            "source_denominator_key": _source_key(body),
            "procedural_stress": {
                "origin": "deterministic_overlay",
                "source_observed": False,
                "source_anchor": "nrel_oedi_profile_window",
                "event_kind": "grid_outage",
                "event_tick": event_tick,
                "duration_ticks": duration,
                "native_loss_required": True,
            },
            "response_window_recipe": {
                "version": "pymgrid_native_state_loss_response_v1",
                "event_trigger_tick": event_tick,
                "preposition_opportunity_tick": preposition_tick,
                "recovery_opportunity_tick": recovery_tick,
                "response_window_end_tick": restoration_tick,
                "restoration_opportunity_tick": restoration_tick,
                "simulator_advances_time": True,
                "state_changing_tools": required_tools,
            },
            "native_state_loss_task": {
                "contract": "microgrid.native_state_loss.v1",
                "required_task_loss_keys": [
                    "balance_error_mw",
                    "shed_penalty",
                ],
                "task_loss_formula": "sum(abs(balance_error_mw) * 200 + shed_penalty)",
                "production_cost_only_rejected": True,
                "minimum_baseline_task_loss": 1000.0,
                "minimum_task_loss_reduction": 100.0,
                "response_window_required": True,
                "source_vs_procedural_separation": True,
            },
            "task_requirements": {
                "min_distinct_control_ticks": 3,
                "min_distinct_physical_tools": 3,
                "ordered_tool_milestones": milestones,
            },
            "source_lock": {
                "files": list((body.get("provenance") or {}).get("files") or []),
                "commit": (body.get("provenance") or {}).get("commit"),
                "url": (body.get("provenance") or {}).get("url"),
                "window_sha256": recipe.get("source_window_sha256"),
            },
        }
    )
    body["backend_config"] = config
    body["source_contract"] = _source_contract(body)
    body["scenario_signature"] = recompute_signature_with_seed(body, seed)
    return body


def _row(body: dict[str, Any], relative_path: str) -> dict[str, Any]:
    config = dict(body.get("backend_config") or {})
    task = dict(config.get("native_state_loss_task") or {})
    response = dict(config.get("response_window_recipe") or {})
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
        "source_key": config["source_denominator_key"],
        "source_denominator_key": config["source_denominator_key"],
        "source_window_sha256": config["source_lock"]["window_sha256"],
        "response_window": {
            "event_trigger_tick": response["event_trigger_tick"],
            "preposition_opportunity_tick": response[
                "preposition_opportunity_tick"
            ],
            "recovery_opportunity_tick": response["recovery_opportunity_tick"],
            "response_window_end_tick": response["response_window_end_tick"],
            "restoration_opportunity_tick": response[
                "restoration_opportunity_tick"
            ],
        },
        "task_loss_keys": list(task["required_task_loss_keys"]),
        "status": "pending_protocol21_full_admission",
        "reason_codes": [
            "source_locked_nrel_oedi_profile",
            "procedural_stress_overlay_explicit",
            "native_ems_task_loss_required",
            "cross_tick_response_window_required",
            "requires_behavior_task_depth_replay_gates",
        ],
    }


def build(
    *,
    repo_root: Path = REPO_ROOT,
    staging_root: Path = DEFAULT_STAGING,
    cases: Iterable[dict[str, Any]] = DEFAULT_CASES,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    files: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for case in cases:
        body = _build_body(dict(case))
        path = staging_root / f"{body['scenario_id'].replace('/', '__')}.yaml"
        files[path] = body
        relative_path = path.resolve().relative_to(repo_root.resolve())
        rows.append(_row(body, relative_path.as_posix()))
    source_keys = [row["source_denominator_key"] for row in rows]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("microgrid native state-loss source keys must be independent")
    return (
        {
            "schema_version": "protocol21-microgrid-native-state-loss-v1",
            "status": "staging_native_state_loss_candidates_pending_full_admission",
            "leaderboard_eligible": False,
            "release_ready": False,
            "n_candidates": len(rows),
            "difficulty_counts": dict(
                sorted(Counter(str(row["difficulty_level"]) for row in rows).items())
            ),
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
