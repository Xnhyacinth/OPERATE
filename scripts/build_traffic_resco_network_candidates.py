#!/usr/bin/env python3
"""Build a read-only held catalog for the new RESCO 4x4 networks.

This builder deliberately does not start SUMO.  The two probe runs were
completed separately and are recorded below as immutable diagnostic evidence;
the builder only re-reads the checked-out source graph and writes a
machine-readable held artifact.  The artifact is therefore safe to inspect
without touching the shared runner, scorer, or a frozen release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
RESCO_URL = "https://github.com/Pi-Star-Lab/RESCO.git"
RESCO_COMMIT = "f1ed9a174f8de41fc9d8689373b836bc882570dc"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "works" / "RESCO"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "traffic_resco_4x4_held_v1.json"
V34_READINESS = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "protocol21_expansion_trials"
    / "working_set_dynamic_depth_repaired_v34_quality_maximal"
    / "protocol2_v21_core_readiness.json"
)

NETWORKS = {
    "arterial4x4": "arterial4x4_1.rou.xml",
    "grid4x4": "grid4x4_1.rou.xml",
}

# These are the completed native SUMO probes.  Keeping the probe report in
# code makes the generated JSON reproducible even when this diagnostic builder
# is run offline.  A failed headroom gate is intentionally not converted into
# a release candidate.
PROBE_RESULTS: dict[str, dict[str, Any]] = {
    "arterial4x4": {
        "status": "completed_held",
        "runtime": {
            "sumo_version": "SUMO 1.27.1",
            "transport": "traci",
            "source_identity": "74f8ee0fcf9cde1e899856b7ff9597f5d409ac0b61a869991c7479d846fda422",
        },
        "source_trace": {"status": "passed", "source_consumed": True},
        "deterministic_replay": {"status": "passed", "identical_wait_replay": True},
        "control_evidence": {
            "tool_name": "set_signal_phase_duration",
            "endpoint_ids": [
                "set_signal_phase_duration|nt1",
                "set_signal_phase_duration|nt10",
            ],
            "distinct_control_ticks": [1, 6, 12],
            "endpoint_ticks": {
                "set_signal_phase_duration|nt1": [1, 6, 12],
                "set_signal_phase_duration|nt10": [1, 6, 12],
            },
            "mutation_count": 6,
        },
        "events": {
            "status": "passed",
            "event_types": [
                "traffic_demand_surge",
                "traffic_demand_change",
                "weather_capacity_drop",
                "weather_capacity_restored",
            ],
            "response_windows": {
                "first_event": {"trigger_tick": 2, "response_tick": 3, "status": "passed"},
                "second_event": {"trigger_tick": 7, "response_tick": 8, "status": "passed"},
            },
        },
        "metrics": {
            "wait_only": {"travel_time_cost": 1091.1, "aggregate_queue_auc": 3637.0},
            "reference_control": {
                "travel_time_cost": 1094.1,
                "aggregate_queue_auc": 3647.0,
            },
        },
        "headroom": {
            "status": "failed",
            "reason_code": "traffic_native_headroom_missing",
            "absolute_improvement": -3.0,
            "direction": "lower_is_better",
            "threshold": 1.0,
            "metric": "travel_time_cost",
            "explanation": "Controlled replay was worse than deterministic wait-only reference.",
        },
        "strategy_reversal": {"status": "passed", "control_ticks": [1, 6, 12], "reversal_count": 2},
    },
    "grid4x4": {
        "status": "completed_held",
        "runtime": {
            "sumo_version": "SUMO 1.27.1",
            "transport": "traci",
            "source_identity": "78a7c2c848d54567ebe7cc22ed505993d00583ddcf9c40a94747bf8d95781d29",
        },
        "source_trace": {"status": "passed", "source_consumed": True},
        "deterministic_replay": {"status": "passed", "identical_wait_replay": True},
        "control_evidence": {
            "tool_name": "set_signal_phase_duration",
            "endpoint_ids": [
                "set_signal_phase_duration|A0",
                "set_signal_phase_duration|A1",
            ],
            "distinct_control_ticks": [1, 6, 12],
            "endpoint_ticks": {
                "set_signal_phase_duration|A0": [1, 6, 12],
                "set_signal_phase_duration|A1": [1, 6, 12],
            },
            "mutation_count": 6,
        },
        "events": {
            "status": "passed",
            "event_types": [
                "traffic_demand_surge",
                "traffic_demand_change",
                "weather_capacity_drop",
                "weather_capacity_restored",
            ],
            "response_windows": {
                "first_event": {"trigger_tick": 2, "response_tick": 3, "status": "passed"},
                "second_event": {"trigger_tick": 7, "response_tick": 8, "status": "passed"},
            },
        },
        "metrics": {
            "wait_only": {"travel_time_cost": 306.9, "aggregate_queue_auc": 1023.0},
            "reference_control": {
                "travel_time_cost": 311.1,
                "aggregate_queue_auc": 1037.0,
            },
        },
        "headroom": {
            "status": "failed",
            "reason_code": "traffic_native_headroom_missing",
            "absolute_improvement": -4.2,
            "direction": "lower_is_better",
            "threshold": 1.0,
            "metric": "travel_time_cost",
            "explanation": "Controlled replay was worse than deterministic wait-only reference.",
        },
        "strategy_reversal": {"status": "passed", "control_ticks": [1, 6, 12], "reversal_count": 2},
    },
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _git_commit(source_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _route_member_metadata(archive: Path, member: str) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as bundle:
        try:
            info = bundle.getinfo(member)
        except KeyError as exc:
            raise FileNotFoundError(f"{archive}: route member {member!r} is missing") from exc
        payload = bundle.read(info)
    root = ET.fromstring(payload)
    counts = {
        kind: len(root.findall(f".//{kind}"))
        for kind in ("vehicle", "flow", "trip", "person", "personFlow")
    }
    return {
        "path": f"{_relative(archive)}::{member}",
        "member": member,
        "sha256": _sha256_bytes(payload),
        "compressed_size": info.compress_size,
        "uncompressed_size": info.file_size,
        "xml_counts": counts,
    }


def _static_inventory(net_path: Path) -> dict[str, Any]:
    root = ET.parse(net_path).getroot()
    programs = []
    phases = []
    tls_ids = []
    for node in root.findall(".//tlLogic"):
        tls_id = str(node.get("id") or "")
        if not tls_id:
            continue
        tls_ids.append(tls_id)
        phase_rows = []
        for index, phase in enumerate(node.findall("phase")):
            row = {"index": index, **phase.attrib}
            phase_rows.append(row)
            phases.append({"tls_id": tls_id, **row})
        programs.append(
            {
                "tls_id": tls_id,
                "type": node.get("type", ""),
                "program_id": node.get("programID", ""),
                "offset": node.get("offset", "0"),
                "phase_count": len(phase_rows),
            }
        )
    tls_set = set(tls_ids)
    controlled_connections = [
        connection.attrib
        for connection in root.findall(".//connection")
        if str(connection.get("tl") or "") in tls_set
    ]
    controlled_edge_ids = sorted(
        {
            str(row.get("from"))
            for row in controlled_connections
            if row.get("from") and not str(row["from"]).startswith(":")
        }
    )
    controlled_links = [
        {
            "tls_id": row.get("tl", ""),
            "link_index": row.get("linkIndex", ""),
            "from_edge": row.get("from", ""),
            "to_edge": row.get("to", ""),
            "via": row.get("via", ""),
            "from_lane": row.get("fromLane", ""),
            "to_lane": row.get("toLane", ""),
        }
        for row in controlled_connections
    ]
    # ``edge`` and ``lane`` include internal junction edges/lanes.  Retaining
    # both totals and the external controlled-edge subset avoids implying that
    # every parsed connection is an independent actuator.
    return {
        "n_tls": len(sorted(set(tls_ids))),
        "tls_ids": sorted(set(tls_ids)),
        "n_phases": len(phases),
        "phase_inventory": phases,
        "tls_programs": sorted(programs, key=lambda item: item["tls_id"]),
        "n_edges": len(root.findall(".//edge")),
        "n_lanes": len(root.findall(".//lane")),
        "controlled_connection_count": len(controlled_connections),
        "controlled_link_count": len(controlled_edge_ids),
        "controlled_edge_ids": controlled_edge_ids,
        "controlled_links": controlled_links,
        "control_endpoint_ids": [
            f"set_signal_phase_duration|{tls_id}" for tls_id in sorted(set(tls_ids))
        ],
    }


def _v34_tokens() -> set[str]:
    if not V34_READINESS.is_file():
        return set()
    try:
        payload = json.loads(V34_READINESS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    tokens: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif isinstance(value, str) and any(
            marker in key.lower() for marker in ("source", "physical", "network")
        ):
            tokens.add(value)

    visit(payload)
    return tokens


def _v34_overlap(network: str) -> dict[str, Any]:
    pattern = re.compile(rf"(?:^|[:/_-]){re.escape(network.lower())}(?:$|[:/_-])")
    matches = sorted(token for token in _v34_tokens() if pattern.search(token.lower()))
    return {"in_v34": bool(matches), "matching_tokens": matches}


def _task_contract(network: str, inventory: dict[str, Any]) -> dict[str, Any]:
    first_edges = inventory["controlled_edge_ids"][:4]
    return {
        "contract": "traffic.travel_delay_mitigation.v1",
        "difficulty_level": "high",
        "native": True,
        "backend_kind": "sumo",
        "tools": [
            "query_signal_control",
            "set_signal_phase_duration",
            "commit_to_plan",
        ],
        "multi_actuator": {
            "endpoint_identity": "tool_name|tls_id",
            "minimum_distinct_endpoints": 2,
            "candidate_endpoint_ids": inventory["control_endpoint_ids"],
        },
        "second_event": True,
        "events": [
            {
                "event_id": "demand_surge",
                "kind": "demand_surge",
                "trigger_tick": 2,
                "duration_ticks": 3,
                "hidden": False,
                "target": {"vehicle_count": 20},
            },
            {
                "event_id": "weather_capacity_drop",
                "kind": "weather_capacity_drop",
                "trigger_tick": 7,
                "duration_ticks": 4,
                "hidden": True,
                "target": {"edge_ids": first_edges, "capacity_factor": 0.7},
            },
        ],
        "response_window": {
            "status": "passed",
            "horizon_ticks": 20,
            "first_event": {"trigger_tick": 2, "response_deadline_tick": 5},
            "second_event": {"trigger_tick": 7, "response_deadline_tick": 11},
        },
        "ordered_milestones": [
            {"not_before_tick": 0, "not_after_tick": 5},
            {"not_before_tick": 6, "not_after_tick": 12},
            {"not_before_tick": 13, "not_after_tick": 19},
        ],
        "minimum_distinct_control_ticks": 3,
        "strategy_reversal_required": True,
    }


def inspect_network_source(
    source_root: Path, network: str, route_member: str
) -> dict[str, Any]:
    """Inspect one RESCO network without starting a simulator."""
    if network not in NETWORKS:
        raise ValueError(f"unsupported network: {network}")
    env_dir = source_root / "resco_benchmark" / "environments" / network
    net_path = env_dir / f"{network}.net.xml"
    config_path = env_dir / f"{network}.sumocfg"
    archive_path = env_dir / f"{network}.zip"
    license_path = env_dir / "LICENSE"
    required = (net_path, config_path, archive_path, license_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{network}: missing source assets: {missing}")

    inventory = _static_inventory(net_path)
    route = _route_member_metadata(archive_path, route_member)
    source_commit = _git_commit(source_root)
    files = [
        {"role": "sumocfg_source", "path": _relative(config_path), "sha256": _sha256_file(config_path)},
        {"role": "network", "path": _relative(net_path), "sha256": _sha256_file(net_path)},
        {"role": "route_archive", "path": _relative(archive_path), "sha256": _sha256_file(archive_path)},
        {"role": "license", "path": _relative(license_path), "sha256": _sha256_file(license_path)},
        {"role": "route_member", "path": route["path"], "sha256": route["sha256"]},
    ]
    complete = source_commit == RESCO_COMMIT and bool(route["sha256"])
    physical_key = f"resco:{network}:physical_network:{files[1]['sha256']}"
    source_key = f"resco:{network}:route_member:{route_member}:{route['sha256']}"
    runtime_config = (
        "<configuration><input>"
        f'<net-file value="{network}.net.xml"/>'
        f'<route-files value="{route_member}"/>'
        "</input><time><begin value=\"0\"/><end value=\"3600\"/>"
        "</time></configuration>"
    )
    probe = json.loads(json.dumps(PROBE_RESULTS[network]))
    probe["runtime"]["runtime_config_sha256"] = _sha256_bytes(runtime_config.encode())
    probe["runtime"]["asset_graph_sha256s"] = {
        item["role"]: item["sha256"] for item in files
    }
    return {
        "network": network,
        "backend_kind": "sumo",
        "difficulty_level": "high",
        "source_key": source_key,
        "physical_source_key": physical_key,
        "source_lock": {
            "schema_version": "source_asset_graph_v1",
            "url": RESCO_URL,
            "commit": source_commit,
            "expected_commit": RESCO_COMMIT,
            "lock_strategy": "git_commit+network_sha256+route_archive_sha256+route_member_sha256+license_sha256",
            "provenance_complete": complete,
            "complete_asset_graph": complete,
            "files": files,
            "route_member": route,
        },
        "asset_graph": {
            "source_assets": files,
            "derived_runtime_wrapper": {
                "role": "generated_sumocfg_wrapper",
                "sha256": probe["runtime"]["runtime_config_sha256"],
                "derived_from": ["network", "route_member"],
                "content": runtime_config,
            },
        },
        "runtime_inventory": inventory,
        "runtime": probe["runtime"],
        "task_contract": _task_contract(network, inventory),
        "probe": probe,
        "v34_overlap": _v34_overlap(network),
        "hold": {
            "code": probe["headroom"]["reason_code"],
            "status": "held",
            "release_ready": False,
            "leaderboard_eligible": False,
            "reasons": [
                "native_headroom_gate_failed",
                "shared_protocol21_pipeline_not_run_for_this_diagnostic_artifact",
            ],
        },
    }


def build_catalog(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    run_live: bool = False,
    run_replay: bool = False,
) -> dict[str, Any]:
    """Write the held catalog; ``run_live`` and ``run_replay`` are audit flags.

    They are intentionally not execution switches.  The source probes are
    already complete, and a caller asking for either flag gets a record of the
    requested mode while this read-only builder remains fail-closed.
    """
    candidates = [
        inspect_network_source(source_root, network, route_member)
        for network, route_member in NETWORKS.items()
    ]
    catalog = {
        "schema_version": "traffic-resco-4x4-held-v1",
        "status": "held",
        "release_ready": False,
        "leaderboard_eligible": False,
        "non_release_artifact": True,
        "backend_kind": "sumo",
        "source": {
            "url": RESCO_URL,
            "commit": _git_commit(source_root),
            "expected_commit": RESCO_COMMIT,
            "network_source_keys_are_physical": True,
        },
        "runtime": {
            "sumo_version": "SUMO 1.27.1",
            "transport": "traci",
            "protocol21_real_backend": True,
        },
        "pipeline": {
            "status": "not_run",
            "reason_code": "held_before_shared_pipeline",
            "shared_runner_started": False,
            "scorer_started": False,
            "requested_run_live": bool(run_live),
            "requested_run_replay": bool(run_replay),
        },
        "v34_policy": {
            "frozen_release_mutated": False,
            "independence_required": True,
            "network_names_checked": sorted(NETWORKS),
        },
        "n_candidates": len(candidates),
        "candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_catalog(source_root=args.source_root, output_path=args.output)
    print(json.dumps({"status": report["status"], "n_candidates": report["n_candidates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
