#!/usr/bin/env python3
"""Reusable, non-release preflight for the next logistics migration slice.

The script intentionally stops before the full Protocol-2.1 pipeline.  It
binds a scenario to its declared source files, exercises the current native
logistics adapter, compares a small observation-derived control probe with a
wait-only replay, and checks deterministic replay.  A passing row is only a
``preflight_candidate``; admission still requires the full source-consumption,
task, counterfactual, depth, agentic, difficulty and deduplication stages.

No scenario YAML, release manifest, core suite, or queue is written by this
module.  The only default output is an auditable JSON report under ``reports``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import Action, ToolCall  # noqa: E402
from domains.logistics.adapter import LogisticsEnvironment  # noqa: E402

REPORT_SCHEMA = "protocol21-logistics-preflight-v1"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "protocol21_logistics_preflight.json"
DEFAULT_SOURCE_SUITE = (
    REPO_ROOT / "scenarios/staging/v0_52_protocol21_v52_quality_maximal/source_suite.json"
)
DEFAULT_ROUTING_ROOT = REPO_ROOT / "scenarios/staging/v0_52_routing_recalibration"

# These are the six positive-headroom rows named by the migration queue.  They
# are kept as explicit source paths so a changed/removed candidate fails closed
# instead of silently selecting a different routing instance.
ROUTING_CANDIDATE_PATHS = (
    "logistics/cvrp_dispatch/time_pressure/basic/cvrp_A-n32-k5_time_pressure_basic_s7701.yaml",
    "logistics/cvrp_dispatch/time_pressure/basic/cvrp_B-n31-k5_time_pressure_basic_s7702.yaml",
    "logistics/cvrp_dispatch/time_pressure/basic/cvrp_CMT6_time_pressure_basic_s7703.yaml",
    "logistics/cvrp_dispatch/time_pressure/basic/cvrp_Golden_1_time_pressure_basic_s7705.yaml",
    "logistics/cvrp_dispatch/time_pressure/basic/cvrp_Li_21_time_pressure_basic_s7706.yaml",
    "logistics/vrptw_dispatch/time_pressure/basic/vrptw_C101_time_pressure_basic_s8801.yaml",
)


def stable_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible values."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path, *, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _scenario_path(row: dict[str, Any], *, repo_root: Path) -> Path:
    path = row.get("path") or row.get("scenario_path")
    if not path:
        raise ValueError("scenario_path_missing")
    return _resolve_path(str(path), repo_root=repo_root)


def _expected_source_hashes(scenario: dict[str, Any]) -> dict[str, str]:
    """Collect optional per-file hashes from all supported lock locations."""

    out: dict[str, str] = {}
    provenance = dict(scenario.get("provenance") or {})
    backend = dict(scenario.get("backend_config") or {})
    locks: list[dict[str, Any]] = []
    for candidate in (
        provenance.get("source_lock"),
        provenance.get("file_sha256s"),
        backend.get("file_sha256s"),
        backend.get("m5_source_lock"),
    ):
        if isinstance(candidate, dict):
            locks.append(candidate)
    for lock in locks:
        nested = lock.get("file_sha256s") if isinstance(lock.get("file_sha256s"), dict) else lock
        if not isinstance(nested, dict):
            continue
        for key, value in nested.items():
            if isinstance(value, str) and len(value.removeprefix("sha256:")) == 64:
                out[str(key)] = value.removeprefix("sha256:")
    # M5 keeps the canonical lock in backend_config.m5_source_lock.
    m5_lock = backend.get("m5_source_lock")
    if isinstance(m5_lock, dict):
        for key, value in dict(m5_lock.get("file_sha256s") or {}).items():
            if isinstance(value, str):
                out[str(key)] = value.removeprefix("sha256:")
    return out


def build_row_identity(
    row: dict[str, Any], *, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    """Load one candidate and bind its scenario/source identities.

    The function is deliberately usable without importing a simulator, which
    lets unit tests and batch callers reject missing source assets cheaply.
    """

    scenario_path = _scenario_path(row, repo_root=repo_root)
    out: dict[str, Any] = {
        "scenario_id": str(row.get("scenario_id") or ""),
        "path": str(scenario_path),
        "path_relative": str(
            scenario_path.relative_to(repo_root)
            if scenario_path.is_relative_to(repo_root)
            else scenario_path
        ),
        "scenario_sha256": sha256_file(scenario_path),
        "scenario_exists": scenario_path.is_file(),
    }
    if not scenario_path.is_file():
        out["source_identity"] = {
            "status": "held",
            "reason_code": "scenario_yaml_missing",
            "files": [],
        }
        return out
    try:
        scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive parser boundary
        out["source_identity"] = {
            "status": "held",
            "reason_code": "scenario_yaml_parse_error",
            "error": f"{type(exc).__name__}: {exc}",
            "files": [],
        }
        return out
    if not isinstance(scenario, dict):
        out["source_identity"] = {
            "status": "held",
            "reason_code": "scenario_yaml_not_mapping",
            "files": [],
        }
        return out
    out["scenario"] = scenario
    provenance = dict(scenario.get("provenance") or {})
    source_files = [str(item) for item in provenance.get("files") or []]
    expected_hashes = _expected_source_hashes(scenario)
    file_rows: list[dict[str, Any]] = []
    for source in source_files:
        path = _resolve_path(source, repo_root=repo_root)
        actual = sha256_file(path)
        expected = expected_hashes.get(source)
        file_rows.append(
            {
                "declared_path": source,
                "path": str(path),
                "exists": path.is_file(),
                "sha256": actual,
                "expected_sha256": expected,
                "hash_matches": (
                    actual == expected if expected is not None else None
                ),
            }
        )
    metadata_ok = all(
        bool(provenance.get(key))
        for key in ("data_source", "url", "license", "commit", "lock_strategy")
    )
    files_ok = bool(file_rows) and all(item["exists"] for item in file_rows)
    hashes_ok = all(
        item["hash_matches"] is not False for item in file_rows
    )
    source_status = "passed" if metadata_ok and files_ok and hashes_ok else "held"
    reason = None
    if not metadata_ok:
        reason = "source_lock_metadata_incomplete"
    elif not files_ok:
        reason = "source_asset_missing"
    elif not hashes_ok:
        reason = "source_asset_hash_mismatch"
    out["source_identity"] = {
        "status": source_status,
        "reason_code": reason,
        "data_source": provenance.get("data_source"),
        "url": provenance.get("url"),
        "license": provenance.get("license"),
        "commit": provenance.get("commit"),
        "lock_strategy": provenance.get("lock_strategy"),
        "files": file_rows,
        # A missing expected hash is visible and does not silently become a
        # hash match; source locks with an explicit expected hash are checked.
        "hash_binding": "explicit" if expected_hashes else "observed_at_preflight",
    }
    out["backend_kind"] = str(scenario.get("backend_kind") or "")
    out["family"] = str(scenario.get("family") or "")
    out["seed"] = int(scenario.get("seed") or 0)
    out["horizon_ticks"] = int(scenario.get("horizon_ticks") or 0)
    out["difficulty_level"] = str(scenario.get("difficulty_level") or "")
    out["scenario_signature"] = str(scenario.get("scenario_signature") or "")
    return out


def _wait_action(tick: int) -> Action:
    return Action(
        tool_calls=[ToolCall(name="wait", call_id=f"preflight-wait-{tick}")],
        dominant="wait",
    )


def _routing_action(snapshot: dict[str, Any], tick: int) -> Action:
    entities = dict(snapshot.get("entities") or {})
    vehicles = [
        (key, value)
        for key, value in entities.items()
        if isinstance(value, dict)
        and value.get("kind") == "vehicle"
        and value.get("active") is True
        and value.get("broken") is not True
        and isinstance(value.get("route"), list)
        and value.get("route")
    ]
    customer_meta = {
        key: value
        for key, value in entities.items()
        if isinstance(value, dict) and value.get("kind") == "customer"
    }
    if not vehicles:
        # Investigation is still useful for a hidden breakdown, but no native
        # route control is possible until a vehicle is visible.
        return _wait_action(tick)

    # First repair a visible breakdown / blocked route.  ``query_eta`` is the
    # native paid investigation which reveals hidden vehicle failures; the
    # next tick then supplies an observation-derived assign_stop opportunity.
    broken_ids = {
        key
        for key, value in entities.items()
        if isinstance(value, dict)
        and value.get("kind") == "vehicle"
        and value.get("broken") is True
    }
    active = [
        (key, value)
        for key, value in vehicles
        if key not in broken_ids
    ]
    if broken_ids and active:
        failed_queue = {
            str(cid)
            for key, value in entities.items()
            if key in broken_ids and isinstance(value, dict)
            for cid in value.get("route") or []
        }
        blocked_queue = {
            str(cid)
            for cid, value in customer_meta.items()
            if value.get("blocked") is True
        }
        queued = failed_queue | blocked_queue
        candidates = [
            (cid, value)
            for cid, value in customer_meta.items()
            if value.get("served") is not True and value.get("dropped") is not True
            and cid in queued
        ]
        for vehicle_id, vehicle in sorted(active, key=lambda item: str(item[0])):
            remaining = float(vehicle.get("remaining_capacity") or 0.0)
            fitting = [
                (cid, value)
                for cid, value in candidates
                if float(value.get("demand") or 0.0) <= remaining + 1e-9
            ]
            if fitting:
                cid, _ = sorted(
                    fitting,
                    key=lambda item: (
                        int(item[1].get("due_tick", 10**9)),
                        -float(item[1].get("criticality", 0.0)),
                        item[0],
                    ),
                )[0]
                return Action(
                    tool_calls=[
                        ToolCall(
                            name="assign_stop",
                            args={"vehicle_id": vehicle_id, "customer_id": cid},
                            call_id=f"preflight-repair-{tick}",
                            idempotency_key=f"preflight-repair-{tick}-{vehicle_id}-{cid}",
                        )
                    ],
                    dominant="assign_stop",
                )
        # If every individual stop exceeds remaining capacity, a native
        # re-plan is still an observable control axis.  Keep one failed-fleet
        # stop in an active route so the simulator records the route mutation;
        # the later headroom gate will decide whether that mutation helps.
        if failed_queue:
            vehicle_id, vehicle = sorted(active, key=lambda item: str(item[0]))[0]
            route = [str(value) for value in vehicle.get("route") or []]
            candidate = sorted(failed_queue)[0]
            if candidate not in route:
                return Action(
                    tool_calls=[
                        ToolCall(
                            name="reroute_vehicle",
                            args={
                                "vehicle_id": vehicle_id,
                                "stop_sequence": route + [candidate],
                            },
                            call_id=f"preflight-reroute-repair-{tick}",
                            idempotency_key=(
                                f"preflight-reroute-repair-{tick}-{vehicle_id}-{candidate}"
                            ),
                        )
                    ],
                    dominant="reroute_vehicle",
                )

    # Reveal a hidden failure before trying a re-plan.  The query is
    # deliberately bounded to the current fleet (at most the native fleet
    # size, well below the basic per-tick budget) so a hidden failure on v1/v2
    # is not missed merely because v0 was healthy.
    if tick >= 0 and not broken_ids:
        return Action(
            tool_calls=[
                ToolCall(
                    name="query_eta",
                    args={"vehicle_id": vehicle_id},
                    call_id=f"preflight-investigate-{tick}-{vehicle_id}",
                    idempotency_key=f"preflight-investigate-{tick}-{vehicle_id}",
                )
                for vehicle_id, _vehicle in sorted(vehicles, key=lambda item: str(item[0]))
            ],
            dominant="query_eta",
        )

    # If no failure has been observed, leave the source-derived route intact;
    # changing an otherwise feasible route would make this screening probe
    # optimize its own artifact instead of testing native repair affordances.
    return _wait_action(tick)


def _orgym_action(snapshot: dict[str, Any], tick: int) -> Action:
    capacity_values = [float(value) for value in snapshot.get("supply_capacity") or []]
    capacity = int(max(capacity_values)) if capacity_values else 0
    on_hand = sum(float(value) for value in snapshot.get("inventory_on_hand") or [])
    pipeline = sum(float(value) for value in snapshot.get("pipeline_inventory") or [])
    next_demand = float(snapshot.get("next_demand_units") or 0.0)
    if capacity <= 0 or next_demand <= on_hand + pipeline:
        return _wait_action(tick)
    quantity = max(1, min(capacity, int(math.ceil(next_demand - on_hand - pipeline))))
    return Action(
        tool_calls=[
            ToolCall(
                name="place_replenishment_order",
                args={"quantity": quantity, "stage": 0},
                call_id=f"preflight-order-{tick}",
                idempotency_key=f"preflight-order-{tick}",
            )
        ],
        dominant="place_replenishment_order",
    )


def _scalar_cost(costs: dict[str, Any]) -> float:
    total = 0.0
    for value in costs.values():
        try:
            total += float(value)
        except (TypeError, ValueError):
            continue
    return round(total, 6)


def _rollout(
    scenario: dict[str, Any],
    *,
    policy: Callable[[dict[str, Any], int], Action],
) -> dict[str, Any]:
    env = LogisticsEnvironment()
    observations: list[dict[str, Any]] = []
    raw_events: list[dict[str, Any]] = []
    canonical_events: list[dict[str, Any]] = []
    tool_effects: list[dict[str, Any]] = []
    done = False
    error: str | None = None
    try:
        observation = env.reset(scenario, seed=int(scenario.get("seed") or 0))
        for tick in range(max(1, int(scenario.get("horizon_ticks") or 1))):
            observations.append(
                {
                    "tick": tick,
                    "snapshot_digest": stable_digest(observation),
                }
            )
            result = env.step(policy(observation, tick))
            raw_events.extend(list(result.info.realized_events or []))
            canonical_events.extend(
                list(result.info.extra.get("world_evolution_records") or [])
            )
            for tool_result in result.tool_results:
                if tool_result.ok and tool_result.state_changing:
                    tool_effects.append(
                        {
                            "name": tool_result.name,
                            "tick": tick,
                            "payload": dict(tool_result.payload),
                            "effect_tick": tool_result.effect_tick,
                        }
                    )
            done = bool(result.done)
            observation = result.observation
            if done:
                break
        ground_truth = env.ground_truth()
        costs = dict(ground_truth.get("cost_components") or {})
        return {
            "status": "passed",
            "n_ticks": len(observations),
            "done": done,
            "observations": observations,
            "raw_events": raw_events,
            "canonical_events": canonical_events,
            "tool_effects": tool_effects,
            "cost_components": costs,
            "scalar_cost": _scalar_cost(costs),
            "final_state_digest": stable_digest(ground_truth),
            "trajectory_digest": stable_digest(
                {
                    "observations": observations,
                    "raw_events": raw_events,
                    "canonical_events": canonical_events,
                    "tool_effects": tool_effects,
                    "ground_truth": ground_truth,
                }
            ),
        }
    except Exception as exc:  # keep one bad source from aborting the batch
        error = f"{type(exc).__name__}: {exc}"
        return {
            "status": "held",
            "reason_code": "runtime_preflight_error",
            "error": error,
            "n_ticks": len(observations),
            "observations": observations,
            "raw_events": raw_events,
            "canonical_events": canonical_events,
            "tool_effects": tool_effects,
        }
    finally:
        env.close()


def _behavior_summary(wait: dict[str, Any]) -> dict[str, Any]:
    records = list(wait.get("canonical_events") or [])
    material = []
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("material_exogenous") is True or record.get("materiality", {}).get("passed") is True:
            material.append(record)
    response_windows = [
        record
        for record in material
        if record.get("response_opportunity_tick") is not None
        or record.get("decision_required") is True
    ]
    return {
        "status": "passed" if material and response_windows else "held",
        "reason_code": None if material and response_windows else "material_event_or_response_window_missing",
        "material_event_count": len(material),
        "response_window_count": len(response_windows),
        "event_types": sorted({str(record.get("event_type") or "") for record in material}),
        "event_ticks": sorted({int(record.get("applied_tick", record.get("tick", 0))) for record in material}),
    }


def _task_summary(controlled: dict[str, Any]) -> dict[str, Any]:
    effects = list(controlled.get("tool_effects") or [])
    successful = [effect for effect in effects if effect.get("name") not in {"wait", "noop"}]
    return {
        "status": "passed" if successful else "held",
        "reason_code": None if successful else "native_control_effect_not_observed",
        "native_control_effect_count": len(successful),
        "native_control_tools": sorted({str(effect.get("name")) for effect in successful}),
    }


def _headroom_summary(wait: dict[str, Any], controlled: dict[str, Any]) -> dict[str, Any]:
    if wait.get("status") != "passed" or controlled.get("status") != "passed":
        return {
            "status": "held",
            "reason_code": "runtime_rollout_missing",
            "wait_cost": wait.get("scalar_cost"),
            "controlled_cost": controlled.get("scalar_cost"),
            "raw_headroom": None,
        }
    wait_cost = float(wait.get("scalar_cost") or 0.0)
    controlled_cost = float(controlled.get("scalar_cost") or 0.0)
    raw_headroom = round(wait_cost - controlled_cost, 6)
    return {
        "status": "passed" if raw_headroom > 1e-9 else "held",
        "reason_code": None if raw_headroom > 1e-9 else "native_control_no_positive_headroom",
        "wait_cost": wait_cost,
        "controlled_cost": controlled_cost,
        "raw_headroom": raw_headroom,
        "probe_only": True,
    }


def classify_preflight(row: dict[str, Any], runtime: dict[str, Any] | None) -> dict[str, Any]:
    """Assign a fail-closed disposition without ever admitting Core rows."""

    if row.get("source_identity", {}).get("status") != "passed":
        disposition = "held_source_lock"
    elif not runtime or runtime.get("status") != "passed":
        disposition = "held_runtime"
    elif row.get("behavior", {}).get("status") != "passed":
        disposition = "held_behavior"
    elif row.get("task", {}).get("status") != "passed":
        disposition = "held_task"
    elif row.get("headroom", {}).get("status") != "passed":
        disposition = "held_headroom"
    elif row.get("determinism", {}).get("status") != "passed":
        disposition = "held_determinism"
    else:
        disposition = "preflight_candidate"
    return {
        **row,
        "disposition": disposition,
        "core_admission": False,
        "full_protocol21_required": True,
    }


def _determinism_summary(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    passed = (
        first.get("status") == "passed"
        and second.get("status") == "passed"
        and first.get("trajectory_digest") == second.get("trajectory_digest")
    )
    return {
        "status": "passed" if passed else "held",
        "reason_code": None if passed else "wait_replay_digest_mismatch",
        "first_trajectory_digest": first.get("trajectory_digest"),
        "second_trajectory_digest": second.get("trajectory_digest"),
        "first_final_state_digest": first.get("final_state_digest"),
        "second_final_state_digest": second.get("final_state_digest"),
    }


def _load_target_rows(
    *,
    source_suite: Path,
    routing_root: Path,
    repo_root: Path,
    include_routing: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if include_routing:
        for relative in ROUTING_CANDIDATE_PATHS:
            rows.append(
                {
                    "scenario_id": relative.removesuffix(".yaml"),
                    "path": str(routing_root / relative),
                    "lane": "P1_routing_v9_positive_headroom_replay",
                }
            )
    if source_suite.is_file():
        payload = json.loads(source_suite.read_text(encoding="utf-8"))
        for row in payload.get("scenarios") or []:
            if isinstance(row, dict) and row.get("backend_kind") == "orgym_invmgmt":
                rows.append({**row, "lane": "P1_orgym_effect_lineage_recertification"})
    return rows


def build_report(
    *,
    source_suite: Path = DEFAULT_SOURCE_SUITE,
    routing_root: Path = DEFAULT_ROUTING_ROOT,
    repo_root: Path = REPO_ROOT,
    include_routing: bool = True,
) -> dict[str, Any]:
    targets = _load_target_rows(
        source_suite=source_suite,
        routing_root=routing_root,
        repo_root=repo_root,
        include_routing=include_routing,
    )
    rows: list[dict[str, Any]] = []
    for target in targets:
        identity = build_row_identity(target, repo_root=repo_root)
        row: dict[str, Any] = {
            key: value
            for key, value in identity.items()
            if key not in {"scenario"}
        }
        row["lane"] = str(target.get("lane") or "")
        scenario = identity.get("scenario")
        if not isinstance(scenario, dict):
            rows.append(classify_preflight(row, runtime=None))
            continue
        wait_first = _rollout(scenario, policy=lambda _snap, tick: _wait_action(tick))
        wait_second = _rollout(scenario, policy=lambda _snap, tick: _wait_action(tick))
        controlled_policy = (
            _orgym_action
            if str(scenario.get("backend_kind") or "") == "orgym_invmgmt"
            else _routing_action
        )
        controlled = _rollout(scenario, policy=controlled_policy)
        row["runtime"] = {
            "status": wait_first.get("status"),
            "n_ticks": wait_first.get("n_ticks"),
            "done": wait_first.get("done"),
            "error": wait_first.get("error"),
        }
        row["behavior"] = _behavior_summary(wait_first)
        row["task"] = _task_summary(controlled)
        row["headroom"] = _headroom_summary(wait_first, controlled)
        row["determinism"] = _determinism_summary(wait_first, wait_second)
        row["probe"] = {
            "controlled_trajectory_digest": controlled.get("trajectory_digest"),
            "controlled_final_state_digest": controlled.get("final_state_digest"),
            "controlled_cost_components": controlled.get("cost_components"),
            "wait_cost_components": wait_first.get("cost_components"),
        }
        rows.append(classify_preflight(row, runtime=wait_first))
    dispositions = Counter(str(row.get("disposition")) for row in rows)
    source_sha = sha256_file(source_suite)
    script_sha = sha256_file(Path(__file__).resolve())
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "preflight_complete",
        "release_ready": False,
        "core_admission": False,
        "full_protocol21_required": True,
        "target": {
            "name": (
                "routing_plus_source_suite" if include_routing else "source_suite_only"
            ),
            "n_expected": len(targets),
            "source_suite": str(source_suite),
            "source_suite_sha256": source_sha,
            "routing_root": str(routing_root),
            "converter_sha256": script_sha,
        },
        "n_rows": len(rows),
        "disposition_counts": dict(sorted(dispositions.items())),
        "rows": rows,
        "non_goals": [
            "does_not_modify_scenario_yaml",
            "does_not_modify_release_or_core_suite",
            "does_not_use_historical_calibration_for_core_admission",
            "does_not_replace_full_protocol21_replay",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", type=Path, default=DEFAULT_SOURCE_SUITE)
    parser.add_argument("--routing-root", type=Path, default=DEFAULT_ROUTING_ROOT)
    parser.add_argument("--exclude-routing", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report(
        source_suite=args.source_suite.resolve(),
        routing_root=args.routing_root.resolve(),
        include_routing=not args.exclude_routing,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_rows": report["n_rows"],
                "disposition_counts": report["disposition_counts"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
