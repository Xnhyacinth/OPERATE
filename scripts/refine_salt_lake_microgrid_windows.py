#!/usr/bin/env python3
"""Bounded candidate-only source-window screening for Salt Lake Microgrid.

The three windows are predeclared seasonal daylight strata from the same
locked NREL site profile.  The screen does not alter difficulty, perturbation
schedule, stress intensity, task contract, backend, or structural seed.
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

from core import Action, ToolCall  # noqa: E402
from domains.microgrid.adapter import MicrogridEnvironment  # noqa: E402
from scripts.build_microgrid_protocol21_candidates import _build_body  # noqa: E402
from scripts.calibrate_core_candidate import _episode  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/microgrid_salt_lake_window_refine_v1"
SOURCE_PROFILE = REPO_ROOT / "works/nrel-microgrid/salt_lake_city_ut.npz"
SOURCE_PROVENANCE = REPO_ROOT / "works/nrel-microgrid/salt_lake_city_ut.provenance.json"
HORIZON_TICKS = 10

# These strata are fixed before native replay.  They cover separated daylight
# periods in the annual Salt Lake profile and are not selected from outcomes.
SALT_LAKE_WINDOWS: tuple[dict[str, Any], ...] = (
    {
        "window_id": "early_year_daylight_p1543",
        "start_index": 1543,
        "selection_basis": "predeclared_annual_stratum_early_year_daylight",
    },
    {
        "window_id": "mid_year_daylight_p4040",
        "start_index": 4040,
        "selection_basis": "predeclared_annual_stratum_mid_year_daylight",
    },
    {
        "window_id": "late_summer_daylight_p5217",
        "start_index": 5217,
        "selection_basis": "predeclared_annual_stratum_late_summer_daylight",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_non_overlapping_windows(
    windows: tuple[dict[str, Any], ...] | list[dict[str, Any]], *, horizon_ticks: int
) -> None:
    ordered = sorted((int(window["start_index"]), str(window["window_id"])) for window in windows)
    if len(ordered) != len({start for start, _window_id in ordered}):
        raise ValueError("duplicate Salt Lake source-window start")
    for (left, _), (right, _) in zip(ordered, ordered[1:], strict=False):
        if left + horizon_ticks > right:
            raise ValueError("Salt Lake source windows overlap")


def _classify_probe(probe: dict[str, Any]) -> dict[str, Any]:
    wait_cost = float(probe.get("wait_cost") or 0.0)
    reference_cost = float(probe.get("reference_cost") or 0.0)
    headroom = wait_cost - reference_cost
    required_headroom = max(1.0, 0.05 * wait_cost)
    difficulty = str(probe.get("difficulty_level") or "")
    depth_floor = 4 if difficulty == "extreme" else 3
    gates = {
        "source_consumption": probe.get("source_consumption_passed") is True,
        "deterministic_replay": probe.get("deterministic_replay_passed") is True,
        "reference_safety": probe.get("reference_safety_passed") is True,
        "native_task_completion": probe.get("native_task_completion_passed") is True,
        "beneficial_headroom": wait_cost > 0.0 and headroom >= required_headroom,
        "partial_observation": probe.get("partial_observation_passed") is True,
        "reference_depth": int(probe.get("reference_depth") or 0) >= depth_floor,
    }
    reason_map = {
        "source_consumption": "source_consumption_unproven",
        "deterministic_replay": "deterministic_replay_unproven",
        "reference_safety": "reference_safety_unproven",
        "native_task_completion": "native_task_completion_unproven",
        "beneficial_headroom": "beneficial_headroom_below_five_percent",
        "partial_observation": "partial_observation_not_runtime_enforced",
        "reference_depth": f"reference_depth_below_{difficulty}_floor",
    }
    return {
        "gates": gates,
        "headroom": headroom,
        "required_headroom": required_headroom,
        "depth_floor": depth_floor,
        "all_scientific_gates_passed": all(gates.values()),
        "reason_codes": [reason_map[name] for name, passed in gates.items() if not passed],
    }


def _body_for_window(window: dict[str, Any]) -> dict[str, Any]:
    start = int(window["start_index"])
    body = _build_body(
        {
            "site": "salt_lake_city_ut",
            "level": "extreme",
            "seed": 58,
            "start_index": start,
            "slug": f"salt_lake_city_ut_extreme_p{start}",
        }
    )
    config = body["backend_config"]
    config["salt_lake_source_window_screen"] = {
        "method": "predeclared_nonoverlapping_source_window_screen_v1",
        "window_id": str(window["window_id"]),
        "selection_basis": str(window["selection_basis"]),
        "profile_start_index": start,
        "source_profile_sha256": _sha256(SOURCE_PROFILE),
        "difficulty_unchanged": True,
        "event_schedule_unchanged": True,
        "stress_intensity_unchanged": True,
        "candidate_only": True,
    }
    return body


def _investigation_reveals_hidden_state(
    *, before: dict[str, Any], investigation: dict[str, Any]
) -> bool:
    hidden_state_fields = {"failed", "available"}
    return not hidden_state_fields.intersection(before) and bool(
        hidden_state_fields.intersection(investigation)
    )


def _partial_observation_probe(body: dict[str, Any]) -> dict[str, Any]:
    env = MicrogridEnvironment()
    observations: list[dict[str, Any]] = []
    hidden_event: dict[str, Any] | None = None
    hidden_asset_id: str | None = None
    hidden_asset_before_investigation: dict[str, Any] = {}
    investigation_payload: dict[str, Any] = {}
    leaked_asset_id: str | None = None
    try:
        env.reset(body, seed=int(body["seed"]))
        for _tick in range(int(body["horizon_ticks"])):
            if hidden_asset_id and not investigation_payload:
                action = Action(
                    tool_calls=[
                        ToolCall(
                            name="investigate_asset",
                            args={"asset_id": hidden_asset_id},
                        )
                    ],
                    dominant="investigate_asset",
                )
            else:
                action = Action(tool_calls=[ToolCall(name="wait")], dominant="wait")
            result = env.step(action)
            for tool_result in result.tool_results:
                if tool_result.name == "investigate_asset" and isinstance(
                    tool_result.payload, dict
                ):
                    investigation_payload = dict(tool_result.payload)
            records = list(result.info.extra.get("world_evolution_records") or [])
            observations.append(
                {
                    "tick": result.observation.get("tick"),
                    "hidden_event_ids": [
                        row.get("event_id") for row in records if row.get("visibility") == "hidden"
                    ],
                }
            )
            for row in records:
                if row.get("event_type") != "der_failure" or row.get("visibility") != "hidden":
                    continue
                hidden_event = dict(row)
                hidden_asset_id = str(row.get("event_id") or "").rsplit(":", 1)[-1]
                hidden_asset_before_investigation = dict(
                    (result.observation.get("entities") or {}).get(hidden_asset_id) or {}
                )
                failed_assets = [
                    str(asset_id)
                    for asset_id, entity in (result.observation.get("entities") or {}).items()
                    if isinstance(entity, dict) and entity.get("failed") is True
                ]
                if failed_assets:
                    leaked_asset_id = failed_assets[0]
            if result.done:
                break
    finally:
        env.close()
    selective_reveal = _investigation_reveals_hidden_state(
        before=hidden_asset_before_investigation,
        investigation=investigation_payload,
    )
    passed = hidden_event is not None and leaked_asset_id is None and selective_reveal
    return {
        "passed": passed,
        "hidden_event_observed": hidden_event is not None,
        "hidden_event": hidden_event,
        "pre_investigation_failed_asset_leaked": leaked_asset_id is not None,
        "leaked_asset_id": leaked_asset_id,
        "hidden_asset_id": hidden_asset_id,
        "hidden_asset_before_investigation": hidden_asset_before_investigation,
        "investigation_payload": investigation_payload,
        "selective_reveal_passed": selective_reveal,
        "runtime_observation_ticks": observations,
        "reason_code": (
            "hidden_state_withheld_before_investigation"
            if passed
            else "hidden_der_failure_visible_without_investigation"
            if leaked_asset_id
            else "investigation_does_not_reveal_hidden_availability"
            if hidden_event is not None and not selective_reveal
            else "hidden_der_failure_not_realized"
        ),
    }


def _source_consumption_passed(*episodes: dict[str, Any]) -> bool:
    return bool(episodes) and all(
        (episode.get("source_consumption_evidence") or {}).get("status") == "passed"
        and (episode.get("source_consumption_evidence") or {}).get("source_state_effect_observed")
        is True
        for episode in episodes
    )


def _probe_window(window: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    body = _body_for_window(window)
    row = {
        "scenario_id": body["scenario_id"],
        "scenario_signature": body["scenario_signature"],
        "path": "candidate-only-in-memory",
        "domain": body["domain"],
        "backend_kind": body["backend_kind"],
        "family": body["family"],
        "difficulty_mode": body["difficulty_mode"],
        "difficulty_level": body["difficulty_level"],
    }
    wait_a = _episode(row, body, "wait_only", replay_index=0)
    wait_b = _episode(row, body, "wait_only", replay_index=1)
    reference = _episode(row, body, "oracle_offline", replay_index=0)
    partial = _partial_observation_probe(body)
    reference_task = dict(reference.get("task_completion") or {})
    survival = float((reference.get("native_dimension_scores") or {}).get("system_survival") or 0.0)
    probe = {
        "window_id": window["window_id"],
        "selection_basis": window["selection_basis"],
        "start_index": window["start_index"],
        "difficulty_level": body["difficulty_level"],
        "scenario_id": body["scenario_id"],
        "scenario_signature": body["scenario_signature"],
        "source_window_sha256": body["backend_config"]["derivation_recipe"]["source_window_sha256"],
        "source_consumption_passed": _source_consumption_passed(wait_a, wait_b, reference),
        "deterministic_replay_passed": wait_a == wait_b,
        "reference_safety_passed": (
            survival >= 100.0
            and reference_task.get("reason_code") != "catastrophic_outcome"
            and (reference.get("terminal_integrity") or {}).get("release_ready") is True
        ),
        "native_task_completion_passed": reference_task.get("completed") is True,
        "wait_cost": wait_a.get("cost"),
        "reference_cost": reference.get("cost"),
        "reference_system_survival": survival,
        "reference_task_completion": reference_task,
        "reference_depth": int(reference.get("phase_depth_proxy") or 0),
        "reference_effective_tool_names": reference.get("effective_tool_names"),
        "reference_effective_state_changing_ticks": reference.get("effective_state_changing_ticks"),
        "partial_observation_passed": partial["passed"],
        "partial_observation_probe": partial,
    }
    classified = _classify_probe(probe)
    probe.update(classified)
    return probe, body


def build_report(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT
) -> tuple[dict[str, Any], dict[str, Any], dict[Path, dict[str, Any]]]:
    _validate_non_overlapping_windows(SALT_LAKE_WINDOWS, horizon_ticks=HORIZON_TICKS)
    probes: list[dict[str, Any]] = []
    bodies: dict[str, dict[str, Any]] = {}
    for window in SALT_LAKE_WINDOWS:
        probe, body = _probe_window(window)
        probes.append(probe)
        bodies[str(window["window_id"])] = body
    survivors = [probe for probe in probes if probe["all_scientific_gates_passed"]]
    files: dict[Path, dict[str, Any]] = {}
    scenarios: list[dict[str, Any]] = []
    for probe in survivors:
        body = bodies[str(probe["window_id"])]
        path = output_root / "candidates" / f"{body['scenario_id'].split('/')[-1]}.yaml"
        files[path] = body
        scenarios.append(
            {
                "scenario_id": body["scenario_id"],
                "scenario_signature": body["scenario_signature"],
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "domain": body["domain"],
                "backend_kind": body["backend_kind"],
                "family": body["family"],
                "difficulty_mode": body["difficulty_mode"],
                "difficulty_level": body["difficulty_level"],
                "seed": body["seed"],
                "source_denominator_key": body["backend_config"]["source_denominator_key"],
                "status": "pending_full_protocol21",
                "candidate_only": True,
            }
        )
    suite = {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "working_set" if scenarios else "terminal_empty",
        "candidate_only": True,
        "release_ready": False,
        "n_scenarios": len(scenarios),
        "scenarios": scenarios,
    }
    report = {
        "schema_version": "salt-lake-microgrid-window-refine-v1",
        "status": "candidate_only_probe_complete",
        "candidate_only": True,
        "release_admission": False,
        "source_profile": {
            "path": SOURCE_PROFILE.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(SOURCE_PROFILE),
            "provenance_path": SOURCE_PROVENANCE.relative_to(REPO_ROOT).as_posix(),
            "provenance_sha256": _sha256(SOURCE_PROVENANCE),
        },
        "policy": {
            "predeclared_non_overlapping_windows": True,
            "difficulty_unchanged": True,
            "event_schedule_unchanged": True,
            "stress_intensity_unchanged": True,
            "model_performance_used_for_admission": False,
            "full_protocol21_only_after_all_scientific_gates": True,
        },
        "summary": {
            "n_windows": len(probes),
            "n_survivors": len(survivors),
            "disposition": (
                "candidate_pending_full_protocol21" if survivors else "terminal_held_repair"
            ),
        },
        "probes": probes,
        "terminal_reason_codes": (
            []
            if survivors
            else sorted({code for probe in probes for code in probe["reason_codes"]})
        ),
        "repair_prescription": (
            None
            if survivors
            else "Keep Salt Lake held. Source-window changes cannot repair the runtime observation contract or insufficient native reference evidence; do not lower difficulty or add events."
        ),
    }
    return report, suite, files


def _write_immutable(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.relative_to(REPO_ROOT.resolve())
    report, suite, files = build_report(output_root=output_root)
    if args.execute:
        for path, body in files.items():
            _write_immutable(path, yaml.safe_dump(body, sort_keys=False))
        _write_immutable(
            output_root / "probe_report.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        _write_immutable(
            output_root / "source_suite.json",
            json.dumps(suite, indent=2, sort_keys=True) + "\n",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
