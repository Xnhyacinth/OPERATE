#!/usr/bin/env python3
"""Build source-locked OpenDSS response-window candidates (Protocol-2.1).

The legacy candidate rows pointed at a removed ``OpenDSS-IEEE34-IEEE123``
layout.  This builder resolves the exact runtime include graph present in the
repository, records its paths in provenance, and declares native startup
operating points plus two executable response windows.  Rows remain staging
only until the normal replay/depth gates admit them.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SOURCE_ROOT_REL = "works/OpenDSS-IEEE13/Version8/Distrib/IEEETestCases"
DEFAULT_STAGING = REPO_ROOT / "scenarios/staging/v0_52_powergrid_response_v2"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release/dt_sched_bench_v0_52_0_candidate"
    / "powergrid_response_window_candidates_v2.json"
)


def _load(path: Path) -> dict[str, Any]:
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"scenario is not a mapping: {path}")
    return body


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _include_graph_paths(master: Path) -> list[str]:
    from domains.power_grid.backends.opendss_fresh_feeders import (
        _resolve_native_include_graph,
    )

    assets, _inventory = _resolve_native_include_graph(master.resolve())
    paths: list[str] = []
    for asset in assets:
        path = Path(str(asset["path"]))
        if path.is_file():
            paths.append(_repo_relative(path))
    return list(dict.fromkeys(paths))


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
    feeder: str,
    level: str,
    suffix: str,
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
    body["seed_id"] = f"{promoted_id}_native_response_{suffix}"
    body["scenario_id"] = body["seed_id"]
    body["difficulty_level"] = level
    body["horizon_ticks"] = horizon
    body["perturbations"] = copy.deepcopy(perturbations)

    config = body.setdefault("backend_config", {})
    master_file = {
        "ieee34": "34Bus/ieee34Mod1.dss",
        "ieee123": "123Bus/IEEE123Master.dss",
    }[feeder]
    config.update(
        {
            "feeder": feeder,
            "source_root": SOURCE_ROOT_REL,
            "master_file": master_file,
            "task_contract_profile": {"milestone_ticks": milestone_ticks},
            "response_window_recipe": {
                "version": "opendss_native_response_window_v2",
                "source_identity": f"{SOURCE_ROOT_REL}/{master_file}",
                "runtime_consumption_required": True,
                "startup_controls_are_not_agent_actions": True,
            },
        }
    )

    if feeder == "ieee34":
        # Native IEEE34 starts below the safety floor with all taps at zero.
        # This source-grounded operating point keeps the feeder alive while
        # leaving two distinct tap positions for the response policy.
        config["initial_native_controls"] = [
            {
                "tool": "set_transformer_tap",
                "args": {"trafo_id": index, "tap_pos": value},
            }
            for index, value in enumerate([8, 8, 8, 8, 8, 8])
        ]
        config["control_action_probe"] = {
            "tool": "set_transformer_tap",
            "args": {"trafo_id": 0, "tap_pos": 12},
        }
        config["control_action_follow_up"] = {
            "tool": "set_transformer_tap",
            "args": {"trafo_id": 3, "tap_pos": 6},
        }
        config["task_requirements"] = {
            "min_distinct_control_ticks": 2,
            "min_distinct_physical_tools": 1,
            "ordered_tool_milestones": [
                {
                    "tool": "set_transformer_tap",
                    "args": {"trafo_id": 0, "tap_pos": 12},
                    "not_after_tick": 2,
                },
                {
                    "tool": "set_transformer_tap",
                    "args": {"trafo_id": 3, "tap_pos": 6},
                    "not_before_tick": 3,
                    "not_after_tick": horizon - 1,
                },
            ],
        }
    else:
        # IEEE123 is natively safe after one modest regulator pre-positioning;
        # the two different tools then remain genuinely state-changing.
        config["initial_native_controls"] = [
            {
                "tool": "set_transformer_tap",
                "args": {"trafo_id": 0, "tap_pos": 6},
            },
            {
                "tool": "switch_capacitor",
                "args": {"cap_id": 0, "status": False},
            },
        ]
        config["control_action_probe"] = {
            "tool": "set_transformer_tap",
            "args": {"trafo_id": 0, "tap_pos": 8},
        }
        config["control_action_follow_up"] = {
            "tool": "switch_capacitor",
            "args": {"cap_id": 0, "status": True},
        }
        config["task_requirements"] = {
            "min_distinct_control_ticks": 2,
            "min_distinct_physical_tools": 2,
            "ordered_tool_milestones": [
                {
                    "tool": "set_transformer_tap",
                    "args": {"trafo_id": 0, "tap_pos": 8},
                    "not_after_tick": milestone_ticks[0] + 1,
                },
                {
                    "tool": "switch_capacitor",
                    "args": {"cap_id": 0, "status": True},
                    "not_before_tick": milestone_ticks[1] + 1,
                    "not_after_tick": horizon - 1,
                },
            ],
        }

    graph_master = REPO_ROOT / SOURCE_ROOT_REL / master_file
    graph_paths = _include_graph_paths(graph_master)
    config["runtime_source_lock"] = {
        "master_file": f"{SOURCE_ROOT_REL}/{master_file}",
        "master_sha256": _sha256(graph_master),
        "include_graph_paths": graph_paths,
    }
    source_axes = dict(config.get("source_axes") or {})
    source_axes.update(
        {
            "source": "dss_extensions_electricdss_tst",
            "source_root": SOURCE_ROOT_REL,
            "master_file": master_file,
            "master_sha256": _sha256(graph_master),
            "runtime_include_graph": graph_paths,
        }
    )
    config["source_axes"] = source_axes
    provenance = dict(body.get("provenance") or {})
    provenance["files"] = graph_paths
    provenance["data_source"] = "dss_extensions_electricdss_tst"
    time_window = dict(provenance.get("time_window") or {})
    time_window.update(
        {
            "feeder": feeder,
            "mode": "deep_planning",
            "level": level,
            "master_file": master_file,
        }
    )
    provenance["time_window"] = time_window
    provenance["notes"] = (
        "Runtime source paths are the exact OpenDSS include graph consumed by "
        "dss-python; procedural events are deterministic response overlays."
    )
    body["provenance"] = provenance

    complexity = dict(body.get("complexity_metrics") or {})
    complexity.update(
        {
            "horizon_minutes": horizon * int(body.get("tick_minutes") or 1),
            "n_perturbations": len(perturbations),
            "suddenness_ticks": min(
                (int(event.get("trigger_tick") or 0) for event in perturbations),
                default=horizon,
            ),
            "decision_depth": len(milestone_ticks),
            "n_effective_control_ticks": len(milestone_ticks),
        }
    )
    body["complexity_metrics"] = complexity
    return _canonicalize(body)


def build(
    *, repo_root: Path = REPO_ROOT, staging_root: Path = DEFAULT_STAGING
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    global REPO_ROOT
    REPO_ROOT = repo_root.resolve()
    ieee34_source = (
        REPO_ROOT
        / "scenarios/staging/v0_52_core_candidates/power_grid/"
        "opendss_fresh_feeders_volt_var/deep_planning/basic/"
        "opendss_fresh_ieee34_volt_var_basic_s42.yaml"
    )
    ieee123_source = (
        REPO_ROOT
        / "scenarios/staging/v0_52_protocol2_v21_working_set/"
        "power_grid__opendss_fresh_feeders_volt_var__deep_planning__basic__"
        "opendss_fresh_ieee123_volt_var_basic_s42.yaml"
    )
    cases = [
        (
            ieee34_source,
            "ieee34",
            "medium",
            "medium_two_tap_windows",
            6,
            [
                {
                    "kind": "load_surge",
                    "trigger_tick": 1,
                    "duration_ticks": 2,
                    "target": {"load_fraction": 0.05},
                    "intensity": 1.0,
                    "hidden": False,
                    "notes": "Visible demand excursion on the locked feeder.",
                },
                {
                    "kind": "load_surge",
                    "trigger_tick": 3,
                    "duration_ticks": 2,
                    "target": {"load_fraction": 0.05},
                    "intensity": 1.0,
                    "hidden": False,
                    "notes": "Second native demand excursion opens the later tap window.",
                },
            ],
            [1, 3],
        ),
        (
            ieee123_source,
            "ieee123",
            "high",
            "high_tap_cap_windows",
            10,
            [
                {
                    "kind": "load_surge",
                    "trigger_tick": 1,
                    "duration_ticks": 2,
                    "target": {"load_fraction": 0.10},
                    "intensity": 1.0,
                    "hidden": False,
                    "notes": "Visible demand excursion on the locked feeder.",
                },
                {
                    "kind": "line_outage",
                    "trigger_tick": 3,
                    "duration_ticks": 2,
                    "target": {"line_index": 2},
                    "intensity": 1.0,
                    "hidden": True,
                    "notes": "Hidden native branch outage with response window.",
                },
                {
                    "kind": "load_surge",
                    "trigger_tick": 5,
                    "duration_ticks": 2,
                    "target": {"load_fraction": 0.20},
                    "intensity": 1.0,
                    "hidden": True,
                    "notes": "Hidden demand excursion requires capacitor recovery.",
                },
            ],
            [1, 5, 7],
        ),
    ]
    files: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for source, feeder, level, suffix, horizon, events, milestones in cases:
        body = _candidate(
            source,
            feeder=feeder,
            level=level,
            suffix=suffix,
            horizon=horizon,
            perturbations=events,
            milestone_ticks=milestones,
        )
        filename = (
            f"power_grid__opendss_fresh_feeders_volt_var__deep_planning__"
            f"{level}__{feeder}_{suffix}_s42.yaml"
        )
        path = staging_root / filename
        files[path] = body
        rows.append(
            {
                "scenario_id": body["scenario_id"],
                "path": (
                    _repo_relative(path)
                    if path.is_absolute()
                    else str(path)
                ),
                "difficulty_level": level,
                "feeder": feeder,
                "source_contract": body["source_contract"],
                "scenario_signature": body["scenario_signature"],
                "status": "staging_pending_behavioral_and_depth_gates",
            }
        )
    return {
        "schema_version": "protocol2.1_powergrid_response_candidates_v2",
        "status": "staging_candidates_pending_full_admission",
        "n_candidates": len(rows),
        "scenarios": rows,
    }, files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report, files = build(staging_root=args.staging_root)
    args.staging_root.mkdir(parents=True, exist_ok=True)
    for path, body in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
