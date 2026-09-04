#!/usr/bin/env python3
"""Build a fail-closed ledger for one source-grounded Traffic candidate.

This is deliberately a ledger, not a Core-selection tool.  It binds the
already-qualified RESCO Ingolstadt21 replay to the live SUMO asset graph and
records the native phase-duration, response-window, and baseline/headroom
evidence without mutating a frozen release or copying a mock scenario ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.traffic.source_identity import (  # noqa: E402
    build_sumo_source_identity_payload,
    compute_sumo_source_identity,
    resolve_sumo_input_graph,
)

SCHEMA_VERSION = "traffic_protocol21_recovery_candidate_v1"
SCENARIO_ID = (
    "traffic/signal_coordination/deep_planning/medium/"
    "resco_ingolstadt21_phase_control_medium_s9413"
)
SOURCE_KEY = "resco:ingolstadt21:phase_control"
NATIVE_CONTROL_TOOL = "set_signal_phase_duration"

DEFAULT_ROOT = (
    REPO_ROOT
    / "release/dt_sched_bench_v0_52_0_candidate/protocol21_expansion_trials"
    / "traffic_resco_additional_sources_v1"
)
DEFAULT_SCENARIO = (
    DEFAULT_ROOT
    / "working_set_materialization/scenarios/"
    "traffic__signal_coordination__deep_planning__medium__"
    "resco_ingolstadt21_phase_control_medium_s9413.yaml"
)
DEFAULT_SOURCE_GROUNDED = DEFAULT_ROOT / "current_replay/source_grounded_protocol2_v21.json"
DEFAULT_BEHAVIORAL = DEFAULT_ROOT / "current_replay/behavioral_calibration_protocol2_v21.json"
DEFAULT_AGENTIC = DEFAULT_ROOT / "current_replay/agentic_core_contract_protocol2_v21.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "recovery_candidate_protocol21.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _one_result(document: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in document.get("results", [])
        if isinstance(row, dict) and row.get("scenario_id") == scenario_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one result for {scenario_id!r}, found {len(matches)}"
        )
    return matches[0]


def _asset_graph(scenario: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    backend = scenario.get("backend_config")
    if not isinstance(backend, dict):
        raise ValueError("scenario backend_config is missing")
    config_path = REPO_ROOT / str(backend["sumo_config_path"])
    graph = resolve_sumo_input_graph(config_path)
    expected = {
        str(graph["sumocfg"]["path"]): str(graph["sumocfg"]["sha256"]),
        str(graph["network"]["path"]): str(graph["network"]["sha256"]),
    }
    expected.update(
        {
            str(entry["path"]): str(entry["sha256"])
            for entry in graph.get("route_files", [])
        }
    )
    if not all("ingolstadt21" in path for path in expected):
        raise ValueError("candidate asset graph is not locked to Ingolstadt21")
    return graph, expected


def build_recovery_candidate(
    *,
    scenario_path: Path = DEFAULT_SCENARIO,
    source_grounded_path: Path = DEFAULT_SOURCE_GROUNDED,
    behavioral_path: Path = DEFAULT_BEHAVIORAL,
    agentic_path: Path = DEFAULT_AGENTIC,
    scenario_id: str = SCENARIO_ID,
) -> dict[str, Any]:
    """Return a source-locked candidate ledger, or a fail-closed blocker."""
    scenario = _load_yaml(scenario_path)
    source_grounded = _one_result(_read_json(source_grounded_path), scenario_id)
    behavioral = _one_result(_read_json(behavioral_path), scenario_id)
    agentic = _one_result(_read_json(agentic_path), scenario_id)

    graph, graph_hashes = _asset_graph(scenario)
    source_identity_payload = build_sumo_source_identity_payload(
        graph,
        service_date=str(scenario["backend_config"]["service_date"]),
        sumo_version="SUMO 1.27.1",
        transport="traci_tcp",
    )
    source_identity_sha256 = compute_sumo_source_identity(source_identity_payload)
    locked_hashes = {
        str(path): str(value)
        for path, value in dict(source_grounded.get("source_file_hashes") or {}).items()
    }
    graph_paths_match = set(locked_hashes) == set(graph_hashes)
    graph_hashes_match = graph_paths_match and all(
        locked_hashes[path] == graph_hashes[path] for path in graph_hashes
    )
    files_match = graph_hashes_match and all(
        _sha256(Path(path)) == digest for path, digest in graph_hashes.items()
    )

    episodes = behavioral.get("episodes") or {}
    wait = episodes.get("wait_only") or {}
    reference = episodes.get("oracle_offline") or {}
    task = agentic.get("agentic_contract", {}).get("task_contract", {})
    task_evidence = task.get("evidence") or {}
    source_contract = agentic.get("agentic_contract", {}).get("source_contract", {})
    tool_contract = agentic.get("agentic_contract", {}).get("tool_contract", {})
    world = agentic.get("agentic_contract", {}).get("world_evolution_contract", {})
    checks = agentic.get("checks") or {}
    replay_source_identity = (
        behavioral.get("replay_evidence", {})
        .get("source_consumption_first", {})
        .get("complete_source_identity_sha256")
    )
    edges = world.get("event_to_decision_action_edges") or []
    horizon = int(scenario.get("horizon_ticks", 0))
    response_ticks = sorted(
        int(edge["target_tick"])
        for edge in edges
        if isinstance(edge, dict)
        and edge.get("kind") == "event_to_post_change_decision"
        and isinstance(edge.get("target_tick"), int)
    )
    response_window_passed = bool(
        checks.get("post_change_decision_observed")
        and response_ticks
        and min(response_ticks) > 0
        and max(response_ticks) < horizon
    )
    baseline_headroom = {
        "wait_only_cost": wait.get("cost"),
        "reference_cost": reference.get("cost"),
        "absolute_improvement": float(wait.get("cost", 0.0))
        - float(reference.get("cost", 0.0)),
        "relative_improvement": (
            float(wait.get("cost", 0.0)) - float(reference.get("cost", 0.0))
        )
        / float(wait.get("cost", 1.0)),
        "threshold": task_evidence.get("task_loss_reduction_threshold"),
        "status": "passed"
        if task_evidence.get("native_control_requirements_met")
        and float(task_evidence.get("prevented_loss", 0.0))
        > float(task_evidence.get("task_loss_reduction_threshold", float("inf")))
        else "held",
    }
    native_control = {
        "tool": NATIVE_CONTROL_TOOL,
        "required_tools": source_grounded.get("difficulty_evidence", {}).get(
            "required_tools", []
        ),
        "available_tools": tool_contract.get("native_control_tool_names", []),
        "distinct_control_ticks": task_evidence.get("distinct_control_ticks", []),
        "native_control_requirements_met": task_evidence.get(
            "native_control_requirements_met", False
        ),
        "status": "passed"
        if NATIVE_CONTROL_TOOL
        in source_grounded.get("difficulty_evidence", {}).get("required_tools", [])
        and NATIVE_CONTROL_TOOL in tool_contract.get("native_control_tool_names", [])
        and task_evidence.get("native_control_requirements_met")
        and checks.get("successful_reference_used_native_control")
        else "held",
    }
    checks_out = {
        "source_grounded_admitted": source_grounded.get("status") == "admitted",
        "source_key_exact": scenario.get("backend_config", {}).get(
            "source_denominator_key"
        )
        == SOURCE_KEY,
        "asset_graph_exact": graph_hashes_match and files_match,
        "source_identity_matches_replay": source_identity_sha256
        == replay_source_identity,
        "source_consumption_passed": checks.get("source_consumption_passed") is True,
        "source_independence_passed": checks.get("source_independence_passed") is True,
        "deterministic_replay_passed": checks.get("deterministic_replay_passed") is True,
        "native_phase_duration_control": native_control["status"] == "passed",
        "baseline_headroom_passed": baseline_headroom["status"] == "passed",
        "response_window_passed": response_window_passed,
        "no_mock_identifiers": scenario.get("backend_kind") == "sumo"
        and not any(
            "mock" in str(value).lower()
            for value in (
                scenario_id,
                SOURCE_KEY,
                scenario.get("backend_config", {}).get("source_denominator_key"),
                source_grounded.get("path"),
            )
        ),
    }
    blockers = [name for name, passed in checks_out.items() if not passed]
    status = "candidate_ready" if not blockers else "blocked"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "candidate_status": "source_locked_protocol21" if not blockers else "held",
        "leaderboard_eligible": False,
        "core_mutation": "none",
        "scenario_id": scenario_id,
        "source_denominator_key": SOURCE_KEY,
        "checks": checks_out,
        "blockers": blockers,
        "asset_graph": {
            "schema_version": "source_asset_graph_v1",
            "sumocfg": graph["sumocfg"],
            "network": graph["network"],
            "route_files": graph.get("route_files", []),
            "source_file_hashes": graph_hashes,
        },
        "native_control": native_control,
        "baseline_headroom": baseline_headroom,
        "response_window": {
            "status": "passed" if response_window_passed else "held",
            "horizon_ticks": horizon,
            "n_post_change_edges": len(response_ticks),
            "first_response_tick": min(response_ticks) if response_ticks else None,
            "last_response_tick": max(response_ticks) if response_ticks else None,
        },
        "evidence_bindings": {
            "scenario_file_sha256": _sha256(scenario_path),
            "source_grounded_file_sha256": _sha256(source_grounded_path),
            "behavioral_file_sha256": _sha256(behavioral_path),
            "agentic_file_sha256": _sha256(agentic_path),
            "scenario_signature": source_grounded.get("scenario_signature"),
            "source_identity_sha256": source_contract.get(
                "source_identity_sha256"
            )
            or source_identity_sha256,
            "source_identity_payload": source_identity_payload,
        },
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--source-grounded", type=Path, default=DEFAULT_SOURCE_GROUNDED)
    parser.add_argument("--behavioral", type=Path, default=DEFAULT_BEHAVIORAL)
    parser.add_argument("--agentic", type=Path, default=DEFAULT_AGENTIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_recovery_candidate(
        scenario_path=args.scenario,
        source_grounded_path=args.source_grounded,
        behavioral_path=args.behavioral,
        agentic_path=args.agentic,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["status"] == "candidate_ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
