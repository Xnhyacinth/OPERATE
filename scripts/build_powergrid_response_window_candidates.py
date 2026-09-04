#!/usr/bin/env python3
"""Build source-anchored OpenDSS response-window replacement candidates.

The candidates retain the exact IEEE13/IEEE123 source graphs and only add
deterministic procedural perturbations on top of the native feeder.  They are
staging artifacts: behavioral, task, depth, and agentic gates still decide
whether a row can enter Core.
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
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "powergrid_response_window_candidates_v1.json"
)
DEFAULT_STAGING = REPO_ROOT / "scenarios" / "staging" / "v0_52_powergrid_response_v1"


def _load(path: Path) -> dict[str, Any]:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"scenario is not a mapping: {path}")
    return body


def _canonicalize(body: dict[str, Any]) -> dict[str, Any]:
    from runner.resume import recompute_signature_with_seed
    from scripts.prepare_protocol21_working_set import _source_contract

    result = copy.deepcopy(body)
    result.pop("scenario_signature", None)
    result["source_contract"] = _source_contract(result)
    result["scenario_signature"] = recompute_signature_with_seed(
        result, int(result.get("seed") or 42)
    )
    return result


def _candidate(
    source_path: Path,
    *,
    suffix: str,
    level: str,
    horizon: int,
    perturbations: list[dict[str, Any]],
    milestone_ticks: list[int],
) -> dict[str, Any]:
    body = _load(source_path)
    source_id = str(body["scenario_id"])
    source_parts = source_id.split("/")
    if len(source_parts) >= 5:
        source_parts[3] = level
        promoted_id = "/".join(source_parts)
    else:
        promoted_id = source_id
    body["seed_id"] = f"{promoted_id}_response_{suffix}"
    body["scenario_id"] = body["seed_id"]
    body["difficulty_level"] = level
    body["horizon_ticks"] = horizon
    body["perturbations"] = copy.deepcopy(perturbations)
    config = body.setdefault("backend_config", {})
    config["task_contract_profile"] = {"milestone_ticks": milestone_ticks}
    if level == "high":
        # The high row must respond in two distinct native stages: pre-position
        # the regulator, then switch a capacitor after the hidden outage.  The
        # ordered contract is replayed from successful tool evidence and is
        # what proves temporal depth; the event declarations alone do not.
        config["task_requirements"] = {
            "min_distinct_control_ticks": 2,
            "min_distinct_physical_tools": 2,
            "ordered_tool_milestones": [
                {
                    "tool": "set_transformer_tap",
                    "not_after_tick": max(1, milestone_ticks[0] + 1),
                },
                {
                    "tool": "switch_capacitor",
                    "not_before_tick": milestone_ticks[1] + 1,
                    "not_after_tick": horizon - 1,
                },
            ],
        }
        config["control_action_follow_up"] = {
            "tool": "switch_capacitor",
            # IEEE123's first native capacitor is on at reset; switching it
            # off after the hidden outage is therefore a real state-changing
            # control rather than an acknowledgement/no-op.
            "args": {"cap_id": 0, "status": False},
        }
    config["response_window_recipe"] = {
        "version": "opendss_native_response_window_v1",
        "source_scenario_id": source_id,
        "native_event_tools": ["set_transformer_tap", "switch_capacitor"],
        "claim_limit": (
            "procedural events are source-anchored overlays; source consumption "
            "must still be proven from the compiled OpenDSS include graph"
        ),
    }
    config["source_denominator_key"] = config.get("source_denominator_key")
    return _canonicalize(body)


def build(
    *,
    repo_root: Path = REPO_ROOT,
    staging_root: Path = DEFAULT_STAGING,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    ieee13 = repo_root / (
        "scenarios/staging/v0_52_core_candidates/power_grid/"
        "opendss_fresh_feeders_volt_var/deep_planning/basic/"
        "opendss_fresh_ieee34_volt_var_basic_s42.yaml"
    )
    ieee123 = repo_root / (
        "scenarios/staging/v0_52_protocol2_v21_working_set/"
        "power_grid__opendss_fresh_feeders_volt_var__deep_planning__basic__"
        "opendss_fresh_ieee123_volt_var_basic_s42.yaml"
    )
    rows: list[dict[str, Any]] = []
    files: dict[Path, dict[str, Any]] = {}
    cases = [
        (
            ieee13,
            "medium",
            "medium_load_surge",
            6,
            [
                {
                    "kind": "load_surge",
                    "trigger_tick": 1,
                    "duration_ticks": 2,
                    "target": {"load_fraction": 0.25},
                    "intensity": 1.0,
                    "hidden": False,
                    "notes": "Deterministic native load overlay after feeder reset.",
                }
            ],
            [1, 3],
        ),
        (
            ieee123,
            "high",
            "high_multi_event",
            10,
            [
                {
                    "kind": "load_surge",
                    "trigger_tick": 1,
                    "duration_ticks": 2,
                    "target": {"load_fraction": 0.20},
                    "intensity": 1.0,
                    "hidden": False,
                    "notes": "Visible demand excursion on the locked feeder.",
                },
                {
                    "kind": "line_outage",
                    "trigger_tick": 5,
                    "duration_ticks": 2,
                    "target": {"line_index": 0},
                    "intensity": 1.0,
                    "hidden": True,
                    "notes": "Hidden native line outage with a later response window.",
                },
            ],
            [1, 5, 7],
        ),
    ]
    for source_path, level, suffix, horizon, events, milestones in cases:
        body = _candidate(
            source_path,
            suffix=suffix,
            level=level,
            horizon=horizon,
            perturbations=events,
            milestone_ticks=milestones,
        )
        relative = (
            staging_root.relative_to(repo_root)
            / f"{str(body['seed_id']).replace('/', '__')}.yaml"
        )
        files[repo_root / relative] = body
        config = body["backend_config"]
        rows.append(
            {
                "scenario_id": body["scenario_id"],
                "path": str(relative),
                "domain": body["domain"],
                "backend_kind": body["backend_kind"],
                "family": body["family"],
                "difficulty_mode": body["difficulty_mode"],
                "difficulty_level": body["difficulty_level"],
                "horizon_ticks": body["horizon_ticks"],
                "seed": body["seed"],
                "scenario_signature": body["scenario_signature"],
                "source_key": config.get("source_denominator_key"),
                "source_denominator_key": config.get("source_denominator_key"),
                "replaces_scenario_id": str(
                    (config.get("response_window_recipe") or {}).get(
                        "source_scenario_id"
                    )
                ),
                "status": "pending_protocol21_full_admission",
                "claim_limit": "held until behavioral/task/depth/agentic gates pass",
            }
        )
    return (
        {
            "schema_version": "0.1",
            "status": "staging_candidates_pending_full_admission",
            "pipeline_version": "opendss_native_response_window_v1",
            "n_candidates": len(rows),
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
    report, files = build(
        repo_root=REPO_ROOT,
        staging_root=args.staging_root.resolve(),
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
