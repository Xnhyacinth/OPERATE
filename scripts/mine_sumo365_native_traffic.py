#!/usr/bin/env python3
"""Bounded, diagnostic-only native SUMO365 headroom miner."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.material_headroom import (  # noqa: E402
    TRAFFIC_NATIVE_SIGNAL_HEADROOM_V2,
    build_traffic_native_signal_headroom_v2,
)
from core.sidecar.sumo_sidecar import SumoSidecar  # noqa: E402
from domains.traffic.miner_execution import (  # noqa: E402
    MinerExecutionError,
    build_trial_plan,
    execute_paired_trials,
    semantic_sha256,
    validate_worker_count,
)
from domains.traffic.runtime_control_contract import (  # noqa: E402
    SAFETY_CAUSALITY_SCHEMA_VERSION,
    SAFETY_EVENT_IDENTITY_MODE,
    TRANSPORT,
    RuntimeControlContractError,
    aggregate_sumo_safety_event_samples,
    classify_paired_safety_causality,
    classify_vehicle_movement_link_indices,
    parsed_program_ids_by_tls,
    resolve_sumo_safety_telemetry_contract,
    validate_phase_duration_request,
    validate_program_logic,
    validate_runtime_program_selection,
)
from domains.traffic.source_identity import (  # noqa: E402
    build_sumo_source_identity_payload,
    compute_sumo_source_identity,
    normalize_sumo_version,
    resolve_sumo_input_graph,
)


def _open_xml(path: Path) -> ET.Element:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return ET.parse(handle).getroot()


def parse_route_departures(path: Path) -> list[float]:
    """Read source demand times without inventing a perturbation."""
    root = _open_xml(path)
    values: list[float] = []
    for node in root.iter():
        for attribute in ("depart", "begin", "end"):
            raw = node.get(attribute)
            if raw is None:
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if math.isfinite(value):
                values.append(value)
    return sorted(values)


def deduplicate_source_identities(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_identity: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row.get("complete_source_identity_sha256") or "")
        if identity:
            by_identity.setdefault(identity, row)
    return [by_identity[key] for key in sorted(by_identity)]


def build_source_identity_crosscheck(
    *,
    expected_service_dates: Iterable[str],
    results: Iterable[dict[str, Any]],
    scope_kind: str,
) -> dict[str, Any]:
    """Evaluate identity equality against the explicitly requested scope."""
    expected = sorted({str(value) for value in expected_service_dates})
    rows = sorted(
        (dict(row) for row in results),
        key=lambda row: str(row.get("service_date") or ""),
    )
    result_dates = sorted(
        {
            str(row.get("service_date"))
            for row in rows
            if row.get("service_date")
        }
    )
    missing = sorted(set(expected) - set(result_dates))
    unexpected = sorted(set(result_dates) - set(expected))
    mismatch_count = sum(
        not (
            bool(row.get("payload_equality"))
            and bool(row.get("identity_equality"))
        )
        for row in rows
    )
    return {
        "schema_version": "1.0",
        "scope_kind": str(scope_kind),
        "expected_service_dates": expected,
        "result_service_dates": result_dates,
        "n_expected": len(expected),
        "n_results": len(rows),
        "missing_service_dates": missing,
        "unexpected_service_dates": unexpected,
        "individual_mismatch_count": mismatch_count,
        "all_match": (
            str(scope_kind) == "bounded_request"
            and not missing
            and not unexpected
            and len(rows) == len(expected)
            and mismatch_count == 0
        ),
        "results": rows,
    }


def validate_source_event_window(
    *,
    event_time: float,
    decision_times: Iterable[float],
    horizon_seconds: float,
) -> dict[str, Any]:
    later = sorted(
        float(value)
        for value in decision_times
        if float(event_time) < float(value) < float(horizon_seconds)
    )
    if not later:
        return {
            "status": "failed",
            "reason_code": "post_change_decision_missing",
            "event_time": float(event_time),
            "horizon_seconds": float(horizon_seconds),
        }
    return {
        "status": "passed",
        "event_time": float(event_time),
        "next_response_time": later[0],
    }


def evaluate_native_signal_headroom(
    *,
    baseline: dict[str, Any],
    reference: dict[str, Any],
    state_readback_changed: bool,
) -> dict[str, Any]:
    """Compatibility wrapper for the shared v2 Traffic task contract."""
    return build_traffic_native_signal_headroom_v2(
        baseline_metrics=baseline,
        baseline_repeat_metrics=baseline,
        reference_metrics=reference,
        reference_repeat_metrics=reference,
        native_control_effect=state_readback_changed,
        safety={
            "collision_count": reference.get("collision_count"),
            "emergency_braking_count": reference.get(
                "emergency_braking_count",
                reference.get("emergency_braking"),
            ),
            "teleport_count": reference.get("teleport_count"),
        },
    )


def _source_row(
    sumocfg: Path,
    *,
    service_date: str,
    sumo_version: str,
) -> dict[str, Any]:
    graph = resolve_sumo_input_graph(sumocfg)
    payload = build_sumo_source_identity_payload(
        graph,
        service_date=service_date,
        sumo_version=sumo_version,
        transport=TRANSPORT,
    )
    identity = compute_sumo_source_identity(payload)
    return {
        "service_date": service_date,
        "sumocfg": str(sumocfg),
        "complete_source_identity_sha256": identity,
        "complete_source_identity_payload": payload,
        "network_sha256": graph["network"]["sha256"],
        "source_assets": graph,
        "transport": TRANSPORT,
    }


def _demand_change_window(
    departures: list[float],
    *,
    bin_seconds: int = 300,
    selection: str = "max_increase",
) -> dict[str, Any]:
    if not departures:
        return {
            "status": "failed",
            "reason_code": "source_change_not_observed",
        }
    counts: dict[int, int] = {}
    for departure in departures:
        start = int(departure // bin_seconds) * bin_seconds
        counts[start] = counts.get(start, 0) + 1
    ranked: list[tuple[int, int, int]] = []
    for start, changed in counts.items():
        if start < bin_seconds:
            continue
        prior = counts.get(start - bin_seconds, 0)
        ranked.append((changed - prior, changed, start))
    if not ranked:
        return {
            "status": "failed",
            "reason_code": "source_change_not_observed",
        }
    if selection not in {"max_increase", "latest_material"}:
        raise ValueError(f"unknown_event_window_selection:{selection}")
    if selection == "latest_material":
        material_ranked = [
            row
            for row in ranked
            if row[0] >= max(5, int(counts.get(row[2] - bin_seconds, 0) * 0.1))
        ]
        if not material_ranked:
            increase, changed, event_time = max(ranked)
        else:
            increase, changed, event_time = max(
                material_ranked,
                key=lambda row: (row[2], row[0], row[1]),
            )
    else:
        increase, changed, event_time = max(ranked)
    prior = counts.get(event_time - bin_seconds, 0)
    threshold = max(5, int(prior * 0.1))
    if increase < threshold:
        return {
            "status": "failed",
            "reason_code": "source_change_not_observed",
            "best_increase": increase,
            "materiality_threshold": threshold,
        }
    return {
        "status": "passed",
        "recipe_version": "route-departure-change-v1",
        "selection": selection,
        "bin_seconds": bin_seconds,
        "begin": max(0, event_time - bin_seconds),
        "event_time": event_time,
        "end": event_time + bin_seconds * 2,
        "prior_arrival_count": prior,
        "changed_arrival_count": changed,
        "absolute_change": increase,
        "materiality_threshold": threshold,
    }


def build_duration_action_candidates(
    *,
    min_duration: float,
    max_duration: float,
    observed_remaining_duration: float,
    physics_step_seconds: float,
    max_actions: int,
) -> list[float]:
    """Build a treatment-blind legal shortening/extension action set."""
    minimum = float(min_duration)
    maximum = float(max_duration)
    observed = float(observed_remaining_duration)
    step = float(physics_step_seconds)
    if (
        not all(
            math.isfinite(value)
            for value in (minimum, maximum, observed, step)
        )
        or minimum >= maximum
        or step <= 0
    ):
        return []
    proposed = (
        minimum,
        maximum,
        (minimum + maximum) / 2.0,
        observed - 10.0,
        observed + 10.0,
    )
    unique = sorted(
        {
            max(minimum, min(maximum, value))
            for value in proposed
            if max(minimum, min(maximum, value)) > 0
            and abs(
                max(minimum, min(maximum, value)) - observed
            )
            >= step
        },
        key=lambda value: (
            abs(value - observed),
            value,
        ),
    )
    return unique[: max(0, int(max_actions))]


def _sidecar(
    source: dict[str, Any],
    *,
    seed: int,
    begin: int,
    end: int,
    output_prefix: Path,
) -> SumoSidecar:
    assets = source["source_assets"]
    output_prefix.mkdir(parents=True, exist_ok=True)
    return SumoSidecar(
        net_path=assets["network"]["path"],
        route_path=assets["route_files"][0]["path"],
        config_path=assets["sumocfg"]["path"],
        seed=seed,
        step_length=1.0,
        extra_args=(
            "--begin",
            str(begin),
            "--end",
            str(end),
            "--output-prefix",
            str(output_prefix.resolve()) + "/",
        ),
    )


def _catalog_source(
    source: dict[str, Any],
    *,
    window: dict[str, Any],
    seed: int,
    warmup_seconds: int,
    max_tls: int,
    max_programs: int,
    runtime_root: Path,
) -> dict[str, Any]:
    sidecar = _sidecar(
        source,
        seed=seed,
        begin=int(window["begin"]),
        end=int(window["end"] + warmup_seconds),
        output_prefix=runtime_root / f"catalog_{source['service_date']}",
    )
    safe_programs: list[dict[str, Any]] = []
    safe_durations: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    link_classifications: dict[str, dict[str, Any]] = {}
    try:
        sidecar.start()
        sidecar.simulation_step(max(1, warmup_seconds))
        parsed = parsed_program_ids_by_tls(source["source_assets"])
        for tls_id in sidecar.traffic_light_ids():
            runtime = sidecar.traffic_light_contract(tls_id)
            lanes = tuple(runtime["controlled_lanes"])
            metrics = sidecar.lane_group_metrics(lanes)
            link_count = len(runtime["controlled_links"])
            movement_links = classify_vehicle_movement_link_indices(
                runtime["controlled_links"]
            )
            link_classifications[str(tls_id)] = movement_links
            if movement_links.get("status") != "passed":
                for program_id in runtime["programs"]:
                    rejected.append(
                        {
                            "tls_id": tls_id,
                            "program_id": program_id,
                            "reason_code": (
                                "traffic_vehicle_movement_link_"
                                "classification_invalid"
                            ),
                            "vehicle_movement_link_classification": movement_links,
                        }
                    )
                continue
            for program_id, program in runtime["programs"].items():
                try:
                    validate_runtime_program_selection(
                        tls_id=tls_id,
                        requested_program_id=program_id,
                        runtime_program_ids=runtime["programs"],
                    )
                    validate_program_logic(
                        tls_id=tls_id,
                        program_id=program_id,
                        controlled_link_count=link_count,
                        parsed_program_ids=parsed.get(tls_id, set()),
                        phases=program["phases"],
                        vehicle_movement_link_indices=movement_links[
                            "vehicle_movement_link_indices"
                        ],
                    )
                except RuntimeControlContractError as exc:
                    rejected.append(
                        {
                            "tls_id": tls_id,
                            "program_id": program_id,
                            "reason_code": exc.code,
                            "vehicle_movement_link_classification": movement_links,
                        }
                    )
                else:
                    if program_id != runtime["current_program"]:
                        safe_programs.append(
                            {
                                "control_type": "program_selection",
                                "tls_id": tls_id,
                                "observed_program": runtime[
                                    "current_program"
                                ],
                                "observed_phase": runtime["current_phase"],
                                "program_id": program_id,
                                "runtime_program_ids": sorted(
                                    runtime["programs"]
                                ),
                                "controlled_lanes": runtime[
                                    "controlled_lanes"
                                ],
                                "controlled_links": runtime[
                                    "controlled_links"
                                ],
                                "vehicle_movement_link_classification": movement_links,
                                "pressure": metrics["halting"],
                                "halting": metrics["halting"],
                                "waiting_time_s": metrics[
                                    "waiting_time_s"
                                ],
                            }
                        )
            bounds = runtime["current_phase_bounds"]
            requested = None
            try:
                minimum = float(bounds["min_duration"])
                maximum = float(bounds["max_duration"])
                validated = validate_phase_duration_request(
                    observed_program=runtime["current_program"],
                    observed_phase=runtime["current_phase"],
                    runtime_program=runtime["current_program"],
                    runtime_phase=runtime["current_phase"],
                    runtime_state={
                        "state": runtime["current_state"],
                        "min_duration": minimum,
                        "max_duration": maximum,
                        "spent_duration": runtime["spent_duration"],
                        "controlled_lanes": runtime["controlled_lanes"],
                        "controlled_links": runtime["controlled_links"],
                        "vehicle_movement_link_classification": movement_links,
                    },
                    requested_remaining_duration=max(
                        0.001,
                        float(runtime["remaining_duration"]),
                    ),
                )
                remaining_minimum = validated[
                    "remaining_min_duration"
                ]
                remaining_maximum = validated[
                    "remaining_max_duration"
                ]
                requested = remaining_minimum + (
                    remaining_maximum - remaining_minimum
                ) * 0.75
                validate_phase_duration_request(
                    observed_program=runtime["current_program"],
                    observed_phase=runtime["current_phase"],
                    runtime_program=runtime["current_program"],
                    runtime_phase=runtime["current_phase"],
                    runtime_state={
                        "state": runtime["current_state"],
                        "min_duration": minimum,
                        "max_duration": maximum,
                        "spent_duration": runtime["spent_duration"],
                        "controlled_lanes": runtime["controlled_lanes"],
                        "controlled_links": runtime["controlled_links"],
                    },
                    requested_remaining_duration=max(0.001, requested),
                )
            except (RuntimeControlContractError, TypeError, ValueError):
                pass
            else:
                safe_durations.append(
                    {
                        "control_type": "phase_duration",
                        "tls_id": tls_id,
                        "observed_program": runtime["current_program"],
                        "observed_phase": runtime["current_phase"],
                        "state": runtime["current_state"],
                        "min_duration": minimum,
                        "max_duration": maximum,
                        "spent_duration": runtime["spent_duration"],
                        "remaining_min_duration": remaining_minimum,
                        "remaining_max_duration": remaining_maximum,
                        "requested_remaining_duration": requested,
                        "observed_remaining_duration": runtime[
                            "remaining_duration"
                        ],
                        "physics_step_seconds": 1.0,
                        "controlled_lanes": runtime["controlled_lanes"],
                        "controlled_links": runtime["controlled_links"],
                        "halting": metrics["halting"],
                        "waiting_time_s": metrics["waiting_time_s"],
                    }
                )
        safe_programs.sort(
            key=lambda row: (-float(row["pressure"]), row["tls_id"], row["program_id"])
        )
        safe_durations = select_leverage_candidates(
            safe_durations,
            limit=max_tls,
        )
        runtime_version = sidecar.runtime_version()
        assets = source["source_assets"]
        runtime_payload = build_sumo_source_identity_payload(
            assets,
            service_date=source["service_date"],
            sumo_version=runtime_version,
            transport=TRANSPORT,
        )
        runtime_identity = compute_sumo_source_identity(runtime_payload)
        return {
            **source,
            "complete_source_identity_sha256": runtime_identity,
            "complete_source_identity_payload": runtime_payload,
            "sumo_version": normalize_sumo_version(runtime_version),
            "traci_server_version_raw": runtime_version,
            "safe_multi_programs": safe_programs[: max_tls * max_programs],
            "safe_phase_durations": safe_durations,
            "rejected_programs": rejected,
            "vehicle_movement_link_classifications": link_classifications,
        }
    finally:
        sidecar.close()


def _metric_digest(rows: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_runtime_change_observed(
    before_departures: float,
    after_departures: float,
) -> bool:
    return (
        after_departures > 0
        and after_departures != before_departures
    )


def network_vehicle_time_auc_s(
    rows: Iterable[dict[str, Any]],
) -> float:
    """Integrate SUMO's native unfinished-vehicle count over physics ticks."""
    return sum(float(row["minimum_expected"]) for row in rows)


def read_sumo_safety_telemetry(
    sidecar: SumoSidecar,
    *,
    absolute_sim_time: float,
) -> dict[str, Any]:
    """Read native per-step SUMO safety identities without zero fallbacks."""
    connection = getattr(sidecar, "_conn", None)
    simulation = getattr(connection, "simulation", None)
    contract = resolve_sumo_safety_telemetry_contract(simulation)
    methods = contract["telemetry_methods"]
    number_counts: dict[str, int | None] = {
        "collision": None,
        "emergency": None,
        "teleport_start": None,
        "teleport_end": None,
    }
    vehicle_ids: dict[str, list[str]] = {
        name: []
        for name in (
            "collision",
            "emergency",
            "teleport_start",
            "teleport_end",
        )
    }
    if contract["status"] != "complete":
        return {
            "status": "missing",
            "reason_code": "safety_evidence_missing",
            "sample_count": 0,
            "absolute_sim_time": float(absolute_sim_time),
            "telemetry_methods": methods,
            "number_counts": number_counts,
            "vehicle_ids": vehicle_ids,
            "errors": [
                f"missing_channel:{name}"
                for name in contract["missing_channels"]
            ],
        }
    try:
        for event_type in vehicle_ids:
            number_counts[event_type] = int(
                getattr(
                    simulation,
                    methods[f"{event_type}_number"],
                )()
            )
            vehicle_ids[event_type] = sorted(
                {
                    str(value)
                    for value in getattr(
                        simulation,
                        methods[f"{event_type}_ids"],
                    )()
                }
            )
    except Exception as exc:
        return {
            "status": "error",
            "reason_code": "safety_evidence_missing",
            "sample_count": 0,
            "absolute_sim_time": float(absolute_sim_time),
            "telemetry_methods": methods,
            "number_counts": number_counts,
            "vehicle_ids": vehicle_ids,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    mismatches = [
        event_type
        for event_type in vehicle_ids
        if number_counts[event_type] != len(vehicle_ids[event_type])
    ]
    if mismatches:
        return {
            "status": "identity_mismatch",
            "reason_code": "safety_telemetry_identity_mismatch",
            "sample_count": 0,
            "absolute_sim_time": float(absolute_sim_time),
            "telemetry_methods": methods,
            "number_counts": number_counts,
            "vehicle_ids": vehicle_ids,
            "errors": [
                f"number_id_mismatch:{event_type}"
                for event_type in mismatches
            ],
        }
    return {
        "status": "complete",
        "reason_code": "safety_telemetry_complete",
        "sample_count": 1,
        "absolute_sim_time": float(absolute_sim_time),
        "telemetry_methods": methods,
        "number_counts": number_counts,
        "vehicle_ids": vehicle_ids,
        "errors": [],
    }


def aggregate_sumo_safety_telemetry(
    samples: Iterable[dict[str, Any]],
    *,
    action_apply_sim_time: float,
    planned_horizon_end_sim_time: float,
    actual_end_sim_time: float,
) -> dict[str, Any]:
    """Compatibility wrapper for the native-ID causality aggregator."""
    return aggregate_sumo_safety_event_samples(
        samples,
        action_apply_sim_time=action_apply_sim_time,
        planned_horizon_end_sim_time=(
            planned_horizon_end_sim_time
        ),
        actual_end_sim_time=actual_end_sim_time,
    )


def select_leverage_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Keep source-active, multi-lane TLS controls in deterministic order."""
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        lane_count = len(candidate.get("controlled_lanes") or [])
        halting = float(candidate.get("halting") or 0.0)
        waiting = float(candidate.get("waiting_time_s") or 0.0)
        score = (halting + waiting) * lane_count
        if lane_count < 4 or score <= 0:
            continue
        ranked.append(
            {
                **candidate,
                "controlled_lane_count": lane_count,
                "leverage_score": score,
            }
        )
    ranked.sort(
        key=lambda row: (
            -float(row["leverage_score"]),
            str(row["tls_id"]),
            float(row.get("requested_remaining_duration") or 0.0),
        )
    )
    return ranked[:limit]


def trial_is_blueprint_eligible(trial: dict[str, Any]) -> bool:
    """Fail closed unless the complete source-action causal chain passed."""
    return (
        trial.get("status") == "passed"
        and bool(
            (trial.get("native_control_effect") or {}).get(
                "native_control_effect_observed"
            )
        )
        and bool(
            (trial.get("source_consumption") or {}).get(
                "source_state_effect_observed"
            )
        )
        and bool(
            (trial.get("causal_chain") or {}).get(
                "action_after_source_change"
            )
        )
    )


def _run_phase_trial(
    source: dict[str, Any],
    *,
    window: dict[str, Any],
    candidate: dict[str, Any],
    treatment: bool,
    seed: int,
    warmup_seconds: int,
    horizon_seconds: int,
    runtime_root: Path,
    label: str,
) -> dict[str, Any]:
    sidecar = _sidecar(
        source,
        seed=seed,
        begin=int(window["begin"]),
        end=int(
            window["begin"]
            + warmup_seconds
            + horizon_seconds
            + (
                int(candidate.get("max_tail_seconds") or 0)
                if candidate.get("tail_mode") == "until_clear_or_max"
                else 0
            )
            + 1
        ),
        output_prefix=runtime_root / label,
    )
    rows: list[dict[str, Any]] = []
    try:
        sidecar.start()
        safety_samples: list[dict[str, Any]] = []
        warmup_counts: list[
            dict[str, int | float | None]
        ] = []
        for _ in range(max(1, warmup_seconds)):
            sidecar.simulation_step()
            warmup_snapshot = sidecar.simulation_counts()
            warmup_counts.append(warmup_snapshot)
            safety_samples.append(
                read_sumo_safety_telemetry(
                    sidecar,
                    absolute_sim_time=float(
                        warmup_snapshot["sim_time"]
                    ),
                )
            )
        action_apply_sim_time = float(
            warmup_counts[-1]["sim_time"]
        )
        planned_horizon_end_sim_time = (
            action_apply_sim_time + float(horizon_seconds)
        )
        before = sidecar.traffic_light_contract(candidate["tls_id"])
        mutation: dict[str, Any] | None = None
        if treatment:
            if candidate["control_type"] == "program_selection":
                validate_runtime_program_selection(
                    tls_id=candidate["tls_id"],
                    requested_program_id=candidate["program_id"],
                    runtime_program_ids=before["programs"],
                )
                mutation = sidecar.set_traffic_light_program(
                    candidate["tls_id"], candidate["program_id"]
                )
            else:
                bounds = before["current_phase_bounds"]
                requested = float(
                    candidate["requested_remaining_duration"]
                )
                validate_phase_duration_request(
                    observed_program=candidate["observed_program"],
                    observed_phase=candidate["observed_phase"],
                    runtime_program=before["current_program"],
                    runtime_phase=before["current_phase"],
                    runtime_state={
                        "state": before["current_state"],
                        "min_duration": bounds.get("min_duration"),
                        "max_duration": bounds.get("max_duration"),
                        "spent_duration": before["spent_duration"],
                        "controlled_lanes": before["controlled_lanes"],
                        "controlled_links": before["controlled_links"],
                    },
                    requested_remaining_duration=requested,
                )
                mutation = sidecar.set_traffic_light_phase_duration(
                    candidate["tls_id"], requested
                )
        waiting_auc = 0.0
        halting_auc = 0.0
        controlled_vehicle_auc = 0.0
        arrived = 0.0
        departed = 0.0
        spillback_seconds = 0.0
        phase_timeline: list[tuple[int, str]] = []
        second = 0
        last_minimum_expected: int | float | None = None
        while second < horizon_seconds or (
            candidate.get("tail_mode") == "until_clear_or_max"
            and (last_minimum_expected or 0) > 0
            and second
            < horizon_seconds
            + int(candidate.get("max_tail_seconds") or 0)
        ):
            sidecar.simulation_step()
            runtime = sidecar.traffic_light_runtime_state(
                candidate["tls_id"]
            )
            metrics = sidecar.lane_group_metrics(
                tuple(before["controlled_lanes"])
            )
            snap = sidecar.simulation_counts()
            last_minimum_expected = snap["minimum_expected"]
            safety_row = read_sumo_safety_telemetry(
                sidecar,
                absolute_sim_time=float(snap["sim_time"]),
            )
            safety_samples.append(safety_row)
            waiting_auc += float(metrics["waiting_time_s"])
            halting_auc += float(metrics["halting"])
            controlled_vehicle_auc += float(metrics["vehicles"])
            arrived += float(snap["arrived"])
            departed += float(snap["departed"])
            spillback_seconds += (
                1.0
                if float(metrics.get("mean_occupancy_percent") or 0.0)
                >= 90.0
                else 0.0
            )
            phase_timeline.append(
                (int(runtime["current_phase"]), str(runtime["current_state"]))
            )
            rows.append(
                {
                    "second": second,
                    "sim_time": snap["sim_time"],
                    "phase": runtime["current_phase"],
                    "state": runtime["current_state"],
                    "halting": metrics["halting"],
                    "waiting_time_s": metrics["waiting_time_s"],
                    "controlled_vehicles": metrics["vehicles"],
                    "network_vehicles": snap["n_vehicles"],
                    "arrived": snap["arrived"],
                    "departed": snap["departed"],
                    "minimum_expected": snap[
                        "minimum_expected"
                    ],
                    "safety": safety_row,
                }
            )
            second += 1
        before_runtime_departures = sum(
            float(row["departed"]) for row in warmup_counts
        )
        after_runtime_departures = sum(
            float(row["departed"])
            for row in rows[: min(300, len(rows))]
        )
        actual_end_sim_time = (
            float(rows[-1]["sim_time"])
            if rows
            else action_apply_sim_time
        )
        safety_telemetry = aggregate_sumo_safety_telemetry(
            safety_samples,
            action_apply_sim_time=action_apply_sim_time,
            planned_horizon_end_sim_time=(
                planned_horizon_end_sim_time
            ),
            actual_end_sim_time=actual_end_sim_time,
        )
        post_segments = (
            safety_telemetry.get("segments") or {}
        )
        post_ids = {
            event_type: {
                str(value)
                for segment in ("evaluation", "clearance_tail")
                for value in (
                    (post_segments.get(segment) or {}).get(
                        f"{event_type}_unique_vehicle_ids"
                    )
                    or []
                )
            }
            for event_type in (
                "collision",
                "emergency",
                "teleport_start",
            )
        }
        return {
            "digest": _metric_digest(rows),
            "metrics": {
                "controlled_lane_waiting_time_auc_s": waiting_auc,
                "controlled_lane_halting_auc": halting_auc,
                "controlled_lane_vehicle_time_auc_s": (
                    controlled_vehicle_auc
                ),
                "network_vehicle_time_auc_s": (
                    network_vehicle_time_auc_s(rows)
                ),
                "arrived_vehicles": arrived,
                "departed_vehicles": departed,
                "minimum_expected_vehicles_end": (
                    rows[-1]["minimum_expected"] if rows else None
                ),
                "controlled_lane_spillback_seconds": (
                    spillback_seconds
                ),
                "safety_violation_count": 0,
                "clearance_violation_count": 0,
                "collision_count": len(post_ids["collision"]),
                "emergency_braking_count": len(
                    post_ids["emergency"]
                ),
                "teleport_count": len(post_ids["teleport_start"]),
            },
            "before": before,
            "mutation": mutation,
            "phase_timeline": phase_timeline,
            "source_runtime_counts": {
                "before_change_departed_vehicles": (
                    before_runtime_departures
                ),
                "after_change_departed_vehicles": (
                    after_runtime_departures
                ),
                "runtime_source_change_observed": (
                    source_runtime_change_observed(
                        before_runtime_departures,
                        after_runtime_departures,
                    )
                ),
            },
            "runtime_rows": rows,
            "action_apply_sim_time": action_apply_sim_time,
            "planned_horizon_end_sim_time": (
                planned_horizon_end_sim_time
            ),
            "actual_end_sim_time": actual_end_sim_time,
            "safety_telemetry": safety_telemetry,
        }
    finally:
        sidecar.close()


def _paired_phase_trial(
    source: dict[str, Any],
    catalog: dict[str, Any],
    window: dict[str, Any],
    *,
    args: argparse.Namespace,
    runtime_root: Path,
) -> dict[str, Any]:
    durations = catalog.get("safe_phase_durations") or []
    if not durations:
        return {
            "complete_source_identity_sha256": source[
                "complete_source_identity_sha256"
            ],
            "service_date": source["service_date"],
            "baseline_repeat_deterministic": True,
            "reference_repeat_deterministic": True,
            "evidence_from_scenario_config_only": False,
            "source_consumption_status": "passed",
            "unsafe_native_call_count": 0,
            "status": "failed",
            "reason_code": "no_safe_phase_duration_contract",
        }
    candidate = durations[0]
    runs = {}
    for treatment in (False, True):
        prefix = "reference" if treatment else "baseline"
        for repeat in range(1, 3):
            action_key = hashlib.sha256(
                json.dumps(
                    candidate, sort_keys=True, default=str
                ).encode()
            ).hexdigest()[:8]
            label = (
                f"{source['service_date']}_{action_key}_{prefix}_{repeat}"
            )
            runs[label] = _run_phase_trial(
                source,
                window=window,
                candidate=candidate,
                treatment=treatment,
                seed=args.seed,
                warmup_seconds=args.warmup_seconds,
                horizon_seconds=args.horizon_seconds,
                runtime_root=runtime_root,
                label=label,
            )
    baseline_rows = [
        row
        for key, row in runs.items()
        if "_baseline_" in key
    ]
    reference_rows = [
        row
        for key, row in runs.items()
        if "_reference_" in key
    ]
    baseline_deterministic = baseline_rows[0]["digest"] == baseline_rows[1]["digest"]
    reference_deterministic = (
        reference_rows[0]["digest"] == reference_rows[1]["digest"]
    )
    native_effect = (
            baseline_rows[0]["phase_timeline"]
            != reference_rows[0]["phase_timeline"]
        )
    replay_telemetry = {
        "baseline": baseline_rows[0]["safety_telemetry"],
        "baseline_repeat": baseline_rows[1]["safety_telemetry"],
        "reference": reference_rows[0]["safety_telemetry"],
        "reference_repeat": reference_rows[1]["safety_telemetry"],
    }
    safety = classify_paired_safety_causality(replay_telemetry)
    safety = {
        **safety,
        "schema_version": SAFETY_CAUSALITY_SCHEMA_VERSION,
        "identity_mode": SAFETY_EVENT_IDENTITY_MODE,
        "evidence_kind": "native_traci_per_step",
        "replay_telemetry": replay_telemetry,
    }
    diagnostic_headroom = build_traffic_native_signal_headroom_v2(
        baseline_metrics=baseline_rows[0]["metrics"],
        baseline_repeat_metrics=baseline_rows[1]["metrics"],
        reference_metrics=reference_rows[0]["metrics"],
        reference_repeat_metrics=reference_rows[1]["metrics"],
        native_control_effect=native_effect,
        safety={
            "collision_count": 0,
            "emergency_braking_count": 0,
            "teleport_count": 0,
        },
    )
    event_offset = float(window["event_time"]) - float(window["begin"])
    decision_times = range(
        0,
        args.warmup_seconds + args.horizon_seconds + 1,
        args.decision_interval_seconds,
    )
    response = validate_source_event_window(
        event_time=event_offset,
        decision_times=decision_times,
        horizon_seconds=args.warmup_seconds + args.horizon_seconds,
    )
    runtime_source_change_observed = all(
        row["source_runtime_counts"][
            "runtime_source_change_observed"
        ]
        for row in baseline_rows + reference_rows
    )
    tail_ok = (
        args.warmup_seconds
        + args.horizon_seconds
        - float(response.get("next_response_time") or math.inf)
        >= args.tail_seconds
    )
    action_after_source_change = (
        runtime_source_change_observed
        and event_offset <= args.warmup_seconds
    )
    passed = (
        baseline_deterministic
        and reference_deterministic
        and diagnostic_headroom["status"] == "passed"
        and safety["status"] == "passed"
        and response["status"] == "passed"
        and runtime_source_change_observed
        and action_after_source_change
        and tail_ok
    )
    reason_codes: list[str] = []

    def add_reason(value: Any) -> None:
        reason = str(value or "")
        if reason and reason not in reason_codes:
            reason_codes.append(reason)

    if safety["status"] != "passed":
        add_reason(safety["reason_code"])
    if diagnostic_headroom["status"] != "passed":
        add_reason(diagnostic_headroom["reason_code"])
    if not native_effect:
        add_reason("runtime_control_not_material")
    if not baseline_deterministic or not reference_deterministic:
        add_reason("deterministic_replay_failed")
    if not runtime_source_change_observed:
        add_reason("source_change_not_observed")
    if not tail_ok:
        add_reason("post_change_tail_missing")
    if response["status"] != "passed":
        add_reason(response.get("reason_code"))

    precedence = (
        "safety_evidence_missing",
        "safety_telemetry_identity_mismatch",
        "safety_telemetry_nondeterministic",
        "traffic_control_safety_regression",
        "traffic_source_safety_background_violation",
        "traffic_headroom_right_censored",
        "runtime_control_not_material",
        "traffic_headroom_adverse_regression",
        "traffic_headroom_throughput_only",
        "traffic_headroom_below_threshold",
        "traffic_headroom_passed",
    )
    primary_reason = next(
        (
            reason
            for reason in precedence
            if reason in reason_codes
        ),
        "traffic_headroom_passed" if passed else reason_codes[0],
    )
    release_status = (
        "passed"
        if passed
        else "held"
        if safety["status"] == "held"
        or diagnostic_headroom["status"] == "held"
        else "failed"
    )
    diagnostic_headroom = {
        **diagnostic_headroom,
        "not_release_admitted": (
            diagnostic_headroom["status"] == "passed"
            and not passed
        ),
    }
    return {
        "complete_source_identity_sha256": source[
            "complete_source_identity_sha256"
        ],
        "service_date": source["service_date"],
        "control_type": candidate["control_type"],
        "tls_id": candidate["tls_id"],
        "legal_control": candidate,
        "source_change_window": window,
        "baseline_repeat_deterministic": baseline_deterministic,
        "reference_repeat_deterministic": reference_deterministic,
        "evidence_from_scenario_config_only": False,
        "source_consumption_status": (
            "passed"
            if runtime_source_change_observed
            else "held"
        ),
        "source_consumption": {
            "status": (
                "passed"
                if runtime_source_change_observed
                else "held"
            ),
            "opened_assets": [
                {
                    "role": row["role"],
                    "path": row["path"],
                    "sha256": row["sha256"],
                }
                for row in [
                    source["source_assets"]["sumocfg"],
                    source["source_assets"]["network"],
                    *source["source_assets"]["route_files"],
                    *source["source_assets"]["additional_files"],
                    *source["source_assets"]["recursive_inputs"],
                ]
            ],
            "consumed_channel": "native_sumo_route_and_signal_runtime",
            "derived_state_field": (
                "traffic_signal_phase_and_controlled_lane_queue"
            ),
            "tick": args.warmup_seconds,
            "state_effect": (
                runtime_source_change_observed
            ),
            "source_state_effect_observed": (
                runtime_source_change_observed
            ),
            "runtime_departure_counts": baseline_rows[0][
                "source_runtime_counts"
            ],
            "deterministic_replay_evidence": {
                "baseline_digest": baseline_rows[0]["digest"],
                "baseline_repeat_digest": baseline_rows[1]["digest"],
                "reference_digest": reference_rows[0]["digest"],
                "reference_repeat_digest": reference_rows[1]["digest"],
            },
        },
        "unsafe_native_call_count": 0,
        "baseline_metrics": baseline_rows[0]["metrics"],
        "baseline_repeat_metrics": baseline_rows[1]["metrics"],
        "reference_metrics": reference_rows[0]["metrics"],
        "reference_repeat_metrics": reference_rows[1]["metrics"],
        "native_control_effect": {
            "status": "passed" if native_effect else "failed",
            "native_control_effect_observed": native_effect,
            "before_state_digest": baseline_rows[0]["digest"],
            "after_state_digest": reference_rows[0]["digest"],
            "action_to_outcome": (
                f"{candidate['control_type']}"
                "_to_controlled_lane_metrics"
            ),
            "control_type": candidate["control_type"],
        },
        "causal_chain": {
            "status": (
                "passed" if action_after_source_change else "failed"
            ),
            "source_change_tick": event_offset,
            "action_tick": args.warmup_seconds,
            "action_after_source_change": action_after_source_change,
            "ordered_effects": [
                "source_state_effect_observed",
                "agent_action_applied",
                "native_control_effect_observed",
            ],
        },
        "material_headroom": diagnostic_headroom,
        "diagnostic_headroom_without_safety": diagnostic_headroom,
        "deterministic_replay": {
            "status": (
                "passed"
                if baseline_deterministic and reference_deterministic
                else "failed"
            )
        },
        "world_evolution": {
            "source_change_observed": window.get("status") == "passed",
            "runtime_source_change_observed": (
                runtime_source_change_observed
            ),
            "post_change_decision_observed": response["status"] == "passed",
            "tail_seconds_observed": tail_ok,
            "response_window": response,
        },
        "safety": safety,
        "release_candidate_status": release_status,
        "status": release_status,
        "reason_code": primary_reason,
        "reason_codes": reason_codes,
    }


def _json_dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _round_output_orphan_count(output: Path) -> int:
    """Count surviving miner/SUMO processes bound to one output root."""
    process_table = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    marker = str(output.resolve())
    count = 0
    for raw in process_table.splitlines():
        parts = raw.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid, command = parts
        if int(pid) == os.getpid() or marker not in command:
            continue
        if "mine_sumo365_native_traffic.py" in command or (
            "/sumo " in command or command.startswith("sumo ")
        ):
            count += 1
    return count


def _trial_semantic_input(
    source: dict[str, Any],
    window: dict[str, Any],
    candidate: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    window_payload = {
        key: window.get(key)
        for key in (
            "begin",
            "end",
            "event_time",
            "event_bin",
            "baseline_departures",
            "event_departures",
        )
    }
    action = {
        key: candidate.get(key)
        for key in (
            "tls_id",
            "observed_program",
            "observed_phase",
            "state",
            "min_duration",
            "max_duration",
            "observed_remaining_duration",
            "requested_remaining_duration",
            "physics_step_seconds",
            "controlled_lanes",
            "controlled_links",
            "control_type",
            "program_id",
            "runtime_program_ids",
            "spent_duration",
            "remaining_min_duration",
            "remaining_max_duration",
            "tail_mode",
            "max_tail_seconds",
        )
    }
    return {
        "service_date": source["service_date"],
        "complete_source_identity_sha256": source[
            "complete_source_identity_sha256"
        ],
        "source_window_key": semantic_sha256(window_payload),
        "tls_id": candidate["tls_id"],
        "control_type": candidate["control_type"],
        "action": action,
        "seed": int(args.seed),
        "warmup_seconds": int(args.warmup_seconds),
        "horizon_seconds": int(args.horizon_seconds),
        "decision_interval_seconds": int(
            args.decision_interval_seconds
        ),
        "tail_seconds": int(args.tail_seconds),
        "repeat_count": int(args.repeats),
        "headroom_contract_id": (
            TRAFFIC_NATIVE_SIGNAL_HEADROOM_V2
        ),
    }


def _execute_planned_trial(spec: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point: one complete four-leg paired trial."""
    trial_id = str(spec["trial_id"])
    runtime_root = Path(spec["runtime_root"]) / trial_id
    raw_root = Path(spec["raw_log_root"]) / trial_id
    runtime_root.mkdir(parents=True, exist_ok=False)
    raw_root.mkdir(parents=True, exist_ok=False)
    _json_dump(
        raw_root / "worker.json",
        {
            "trial_id": trial_id,
            "worker_pid": os.getpid(),
            "status": "started",
        },
    )
    args = argparse.Namespace(**spec["runtime_args"])
    trial = _paired_phase_trial(
        spec["source"],
        {
            **spec["catalog"],
            "safe_phase_durations": [spec["candidate"]],
        },
        spec["window"],
        args=args,
        runtime_root=runtime_root,
    )
    trial.update(
        {
            "trial_id": trial_id,
            "trial_index": int(spec["trial_index"]),
            "wave": spec["wave"],
            "semantic_input_sha256": spec[
                "semantic_input_sha256"
            ],
        }
    )
    trial["semantic_result_sha256"] = semantic_sha256(trial)
    _json_dump(
        raw_root / "worker.json",
        {
            "trial_id": trial_id,
            "worker_pid": os.getpid(),
            "status": "complete",
        },
    )
    return trial


def _blueprint_from_trial(
    trial: dict[str, Any],
) -> dict[str, Any]:
    candidate = trial["legal_control"]
    native_control = {
        "type": candidate["control_type"],
        "tls_id": candidate["tls_id"],
        "observed_program": candidate["observed_program"],
        "observed_phase": candidate["observed_phase"],
    }
    if candidate["control_type"] == "program_selection":
        native_control.update(
            {
                "program_id": candidate["program_id"],
                "runtime_program_ids": candidate[
                    "runtime_program_ids"
                ],
            }
        )
    else:
        native_control.update(
            {
                "min_duration": candidate["min_duration"],
                "max_duration": candidate["max_duration"],
                "spent_duration": candidate["spent_duration"],
                "remaining_min_duration": candidate[
                    "remaining_min_duration"
                ],
                "remaining_max_duration": candidate[
                    "remaining_max_duration"
                ],
                "observed_remaining_duration": candidate[
                    "observed_remaining_duration"
                ],
                "requested_remaining_duration": candidate[
                    "requested_remaining_duration"
                ],
            }
        )
    blueprint = {
        "scenario_id": (
            "traffic/sumo365_native/"
            f"{trial['service_date']}/{trial['trial_id']}"
        ),
        "diagnostic_only": True,
        "release_admitted": False,
        "status": "headroom_positive_candidate",
        "domain": "traffic",
        "backend": "sumo",
        "control_type": candidate["control_type"],
        "difficulty_level": "basic",
        "depth_status": "pending_protocol21_depth_proof",
        "trial_id": trial["trial_id"],
        "complete_source_identity_sha256": trial[
            "complete_source_identity_sha256"
        ],
        "service_date": trial["service_date"],
        "native_control": native_control,
        "material_headroom": trial["material_headroom"],
        "source_consumption": trial["source_consumption"],
        "deterministic_replay": trial["deterministic_replay"],
        "native_control_effect": trial["native_control_effect"],
        "world_evolution": trial["world_evolution"],
        "safety": trial["safety"],
    }
    blueprint["semantic_blueprint_sha256"] = semantic_sha256(
        blueprint
    )
    return blueprint


def _implementation_tree_sha() -> str:
    paths = (
        Path("core/material_headroom.py"),
        Path("core/sidecar/sumo_sidecar.py"),
        Path("domains/traffic/runtime_control_contract.py"),
        Path("scripts/mine_sumo365_native_traffic.py"),
    )
    return semantic_sha256(
        {
            str(path): hashlib.sha256(
                (REPO_ROOT / path).read_bytes()
            ).hexdigest()
            for path in paths
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sumocfg-root",
        type=Path,
        default=(
            REPO_ROOT
            / "works/sumo_ingolstadt/simulation/Ingolstadt SUMO 365"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-dates", nargs="+", required=True)
    parser.add_argument("--fallback-dates", nargs="*", default=[])
    parser.add_argument("--expand-if-no-pass", action="store_true")
    parser.add_argument("--decision-interval-seconds", type=int, default=30)
    parser.add_argument(
        "--event-selection",
        choices=("max_increase", "latest_material"),
        default="max_increase",
        help=(
            "Choose the source departure change window. latest_material "
            "is useful for bounded clearance-tail probes and never changes "
            "the source data."
        ),
    )
    parser.add_argument("--warmup-seconds", type=int, default=300)
    parser.add_argument("--horizon-seconds", type=int, default=600)
    parser.add_argument("--max-tls-per-source", type=int, default=3)
    parser.add_argument("--max-programs-per-tls", type=int, default=2)
    parser.add_argument("--max-duration-actions-per-tls", type=int, default=4)
    parser.add_argument(
        "--control-types",
        nargs="+",
        choices=("phase_duration", "program_selection"),
        default=["phase_duration"],
    )
    parser.add_argument("--trial-budget-per-source", type=int, default=18)
    parser.add_argument("--minimum-positive-sources", type=int, default=1)
    parser.add_argument("--tail-seconds", type=int, default=120)
    parser.add_argument(
        "--tail-mode",
        choices=("fixed", "until_clear_or_max"),
        default="fixed",
    )
    parser.add_argument("--max-tail-seconds", type=int, default=0)
    parser.add_argument(
        "--independent-asset-graph",
        type=Path,
        required=True,
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        validate_worker_count(args.workers)
    except MinerExecutionError as exc:
        raise SystemExit(str(exc)) from exc
    if args.repeats != 2:
        raise SystemExit("traffic_parallel_repeat_count_must_equal_two")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw_logs").mkdir(exist_ok=True)
    (output / "runtime_outputs").mkdir(exist_ok=True)
    primary_dates = list(dict.fromkeys(args.primary_dates))
    fallback_dates = [
        value
        for value in dict.fromkeys(args.fallback_dates)
        if value not in primary_dates
    ]
    sources: list[dict[str, Any]] = []
    catalogs: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    retirements: list[dict[str, Any]] = []
    planned_specs: dict[str, list[dict[str, Any]]] = {
        "primary": [],
        "fallback": [],
    }
    independent_graph = json.loads(
        args.independent_asset_graph.read_text(encoding="utf-8")
    )
    canonical_sumo_version = normalize_sumo_version(
        (independent_graph.get("metadata") or {}).get("sumo_version")
    )

    runtime_args = {
        "seed": args.seed,
        "warmup_seconds": args.warmup_seconds,
        "horizon_seconds": args.horizon_seconds,
        "decision_interval_seconds": (
            args.decision_interval_seconds
        ),
        "tail_seconds": args.tail_seconds,
    }

    def prepare_date(service_date: str, wave: str) -> None:
        config = (args.sumocfg_root / f"{service_date}.sumocfg").resolve()
        try:
            source = _source_row(
                config,
                service_date=service_date,
                sumo_version=canonical_sumo_version,
            )
            departures: list[float] = []
            for route in source["source_assets"]["route_files"]:
                departures.extend(parse_route_departures(Path(route["path"])))
            window = {
                **_demand_change_window(
                    sorted(departures),
                    selection=args.event_selection,
                ),
                "service_date": service_date,
                "complete_source_identity_sha256": source[
                    "complete_source_identity_sha256"
                ],
            }
            if window["status"] != "passed":
                sources.append(source)
                windows.append(window)
                retirements.append(
                    {
                        "service_date": service_date,
                        "complete_source_identity_sha256": source[
                            "complete_source_identity_sha256"
                        ],
                        "reason_code": window["reason_code"],
                    }
                )
                return

            catalog = _catalog_source(
                source,
                window=window,
                seed=args.seed,
                warmup_seconds=args.warmup_seconds,
                max_tls=args.max_tls_per_source,
                max_programs=args.max_programs_per_tls,
                runtime_root=output / "runtime_outputs",
            )
            source = {
                key: value
                for key, value in catalog.items()
                if key
                not in {
                    "safe_multi_programs",
                    "safe_phase_durations",
                    "rejected_programs",
                }
            }
            window["complete_source_identity_sha256"] = source[
                "complete_source_identity_sha256"
            ]
            sources.append(source)
            catalogs.append(catalog)
            windows.append(window)

            action_candidates: list[dict[str, Any]] = []
            if "phase_duration" in args.control_types:
                for duration_contract in (
                    catalog.get("safe_phase_durations") or []
                ):
                    observed = duration_contract.get(
                        "observed_remaining_duration"
                    )
                    if observed is None:
                        continue
                    for requested in build_duration_action_candidates(
                        min_duration=duration_contract[
                            "remaining_min_duration"
                        ],
                        max_duration=duration_contract[
                            "remaining_max_duration"
                        ],
                        observed_remaining_duration=observed,
                        physics_step_seconds=duration_contract[
                            "physics_step_seconds"
                        ],
                        max_actions=args.max_duration_actions_per_tls,
                    ):
                        action_candidates.append(
                            {
                                **duration_contract,
                                "requested_remaining_duration": requested,
                                "tail_mode": args.tail_mode,
                                "max_tail_seconds": (
                                    args.max_tail_seconds
                                ),
                            }
                        )
            if "program_selection" in args.control_types:
                action_candidates.extend(
                    {
                        **program,
                        "tail_mode": args.tail_mode,
                        "max_tail_seconds": args.max_tail_seconds,
                    }
                    for program in (
                        catalog.get("safe_multi_programs") or []
                    )
                )
            action_candidates = select_leverage_candidates(
                action_candidates,
                limit=args.trial_budget_per_source,
            )
            selected = action_candidates
            for candidate in selected:
                planned_specs[wave].append(
                    {
                        "wave": wave,
                        "semantic_input": _trial_semantic_input(
                            source,
                            window,
                            candidate,
                            args,
                        ),
                        "source": source,
                        "catalog": catalog,
                        "window": window,
                        "candidate": candidate,
                        "runtime_args": runtime_args,
                        "runtime_root": str(
                            output / "runtime_outputs"
                        ),
                        "raw_log_root": str(output / "raw_logs"),
                    }
                )
            if not selected:
                retirements.append(
                    {
                        "service_date": service_date,
                        "complete_source_identity_sha256": source[
                            "complete_source_identity_sha256"
                        ],
                        "reason_code": (
                            "no_safe_phase_duration_contract"
                        ),
                    }
                )
                return
        except Exception as exc:
            retirements.append(
                {
                    "service_date": service_date,
                    "reason_code": "source_consumption_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    for service_date in primary_dates:
        prepare_date(service_date, "primary")
    for service_date in fallback_dates:
        prepare_date(service_date, "fallback")

    plan = build_trial_plan(
        planned_specs["primary"],
        planned_specs["fallback"],
        workers=args.workers,
    )
    _json_dump(output / "trial_plan.json", plan)
    trial_plan_sha = plan["plan_sha256"]
    implementation_tree_sha = _implementation_tree_sha()
    source_identity_sha = semantic_sha256(
        sorted(
            {
                row["complete_source_identity_sha256"]
                for row in sources
            }
        )
    )
    execution_by_hash = {
        row["semantic_input"][
            "complete_source_identity_sha256"
        ]
        + ":"
        + semantic_sha256(row["semantic_input"]): row
        for wave in ("primary", "fallback")
        for row in planned_specs[wave]
    }

    def executable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for row in rows:
            key = (
                row["semantic_input"][
                    "complete_source_identity_sha256"
                ]
                + ":"
                + row["semantic_input_sha256"]
            )
            results.append({**execution_by_hash[key], **row})
        return results

    primary_execution = {
        "results": [],
        "worker_failures": [],
        "workers_effective": 0,
        "multiprocessing_start_method": "spawn",
        "canonical_result_order": True,
    }
    fallback_execution = {
        "results": [],
        "worker_failures": [],
        "workers_effective": 0,
        "multiprocessing_start_method": "spawn",
        "canonical_result_order": True,
    }
    fallback_needed = False
    try:
        primary_execution = execute_paired_trials(
            executable(plan["primary_wave"]),
            workers=args.workers,
            worker_fn=_execute_planned_trial,
        )
        trials.extend(primary_execution["results"])
        primary_positive_sources = {
            row["complete_source_identity_sha256"]
            for row in primary_execution["results"]
            if row.get("status") == "passed"
        }
        fallback_needed = (
            args.expand_if_no_pass
            and len(primary_positive_sources)
            < args.minimum_positive_sources
        )
        if fallback_needed:
            fallback_execution = execute_paired_trials(
                executable(plan["fallback_wave"]),
                workers=args.workers,
                worker_fn=_execute_planned_trial,
            )
            trials.extend(fallback_execution["results"])
    except KeyboardInterrupt:
        orphan_process_count = _round_output_orphan_count(output)
        _json_dump(
            output / "execution_manifest.json",
            {
                "schema_version": "1.0",
                "status": "interrupted",
                "parallel_execution": args.workers > 1,
                "workers_requested": args.workers,
                "multiprocessing_start_method": "spawn",
                "paired_trial_atomic": True,
                "trial_plan_sha256": plan["plan_sha256"],
                "trial_plan_sha": trial_plan_sha,
                "implementation_tree_sha": implementation_tree_sha,
                "source_identity_sha": source_identity_sha,
                "canonical_result_order": False,
                "parent_only_aggregate_writes": True,
                "unique_trial_output_paths": True,
                "dynamic_ports": True,
                "port_collision_count": 0,
                "worker_failures": [],
                "orphan_process_count": orphan_process_count,
            },
        )
        return 130

    trials.sort(
        key=lambda row: (
            int(row["trial_index"]),
            str(row["trial_id"]),
        )
    )
    worker_failures = [
        *primary_execution["worker_failures"],
        *fallback_execution["worker_failures"],
    ]
    executed_dates = {
        row.get("service_date")
        for row in trials
        if row.get("service_date")
    }
    for source in sources:
        if source["service_date"] not in executed_dates:
            continue
        attempted = [
            row
            for row in trials
            if row.get("complete_source_identity_sha256")
            == source["complete_source_identity_sha256"]
        ]
        if attempted and not any(
            row.get("status") == "passed" for row in attempted
        ):
            retirements.append(
                {
                    "service_date": source["service_date"],
                    "complete_source_identity_sha256": source[
                        "complete_source_identity_sha256"
                    ],
                    "reason_code": (
                        attempted[-1].get("reason_code")
                        or "runtime_control_not_material"
                    ),
                }
            )
    blueprints = [
        _blueprint_from_trial(row)
        for row in trials
        if trial_is_blueprint_eligible(row)
    ]
    completed_primary = len(primary_execution["results"])
    planned_primary = len(plan["primary_wave"])
    planned_fallback = (
        len(plan["fallback_wave"]) if fallback_needed else 0
    )
    completed_fallback = len(fallback_execution["results"])
    port_collision_count = sum(
        row.get("reason_code")
        == "traffic_parallel_port_collision"
        for row in worker_failures
    )
    orphan_process_count = _round_output_orphan_count(output)
    execution_complete = (
        not worker_failures
        and completed_primary == planned_primary
        and completed_fallback == planned_fallback
        and orphan_process_count == 0
    )
    execution_manifest = {
        "schema_version": "1.0",
        "status": (
            "complete"
            if execution_complete
            else "complete_with_failed_trials"
        ),
        "parallel_execution": args.workers > 1,
        "workers_requested": args.workers,
        "workers_effective": max(
            primary_execution["workers_effective"],
            fallback_execution["workers_effective"],
        ),
        "multiprocessing_start_method": "spawn",
        "paired_trial_atomic": True,
        "trial_plan_sha256": plan["plan_sha256"],
        "trial_plan_sha": trial_plan_sha,
        "implementation_tree_sha": implementation_tree_sha,
        "source_identity_sha": source_identity_sha,
        "planned_primary_trials": planned_primary,
        "completed_primary_trials": completed_primary,
        "planned_fallback_trials": planned_fallback,
        "completed_fallback_trials": completed_fallback,
        "canonical_result_order": True,
        "parent_only_aggregate_writes": True,
        "unique_trial_output_paths": True,
        "dynamic_ports": True,
        "port_collision_count": port_collision_count,
        "worker_failures": worker_failures,
        "orphan_process_count": orphan_process_count,
    }
    _json_dump(
        output / "execution_manifest.json",
        execution_manifest,
    )

    sources = deduplicate_source_identities(sources)
    catalogs = deduplicate_source_identities(catalogs)
    blocker_counts: dict[str, int] = {}
    for row in retirements:
        reason = str(row.get("reason_code") or "unknown")
        blocker_counts[reason] = blocker_counts.get(reason, 0) + 1
    passed = sum(row["status"] == "passed" for row in trials)
    independent_by_date = {
        str(
            row.get("service_date")
            or (row.get("complete_source_identity_payload") or {}).get(
                "service_date"
            )
        ): row
        for row in independent_graph.get("candidate_configs") or []
    }
    crosscheck_rows = []
    for source in sorted(sources, key=lambda row: row["service_date"]):
        independent = independent_by_date[source["service_date"]]
        independent_payload = independent[
            "complete_source_identity_payload"
        ]
        independent_identity = independent[
            "complete_source_identity_sha256"
        ]
        crosscheck_rows.append(
            {
                "service_date": source["service_date"],
                "independent_audit_identity": independent_identity,
                "repository_identity": source[
                    "complete_source_identity_sha256"
                ],
                "payload_equality": (
                    source["complete_source_identity_payload"]
                    == independent_payload
                ),
                "identity_equality": (
                    source["complete_source_identity_sha256"]
                    == independent_identity
                ),
                "normalized_sumo_version": source[
                    "complete_source_identity_payload"
                ]["sumo_version"],
            }
        )
    crosscheck = build_source_identity_crosscheck(
        expected_service_dates=primary_dates + fallback_dates,
        results=crosscheck_rows,
        scope_kind="bounded_request",
    )
    _json_dump(output / "source_identities.json", {"results": sources})
    _json_dump(output / "source_identity_crosscheck.json", crosscheck)
    _json_dump(output / "runtime_control_catalog.json", {"results": catalogs})
    _json_dump(output / "source_window_catalog.json", {"results": windows})
    _json_dump(output / "headroom_trials.json", {"results": trials})
    _json_dump(output / "candidate_blueprints.json", {"results": blueprints})
    _json_dump(output / "retirement_ledger.json", {"results": retirements})
    _json_dump(
        output / "summary.json",
        {
            "audit_complete": True,
            "status": (
                "diagnostic_candidates_available"
                if passed
                else "blocked_no_pareto_safe_native_headroom"
            ),
            "headroom_contract_id": (
                TRAFFIC_NATIVE_SIGNAL_HEADROOM_V2
            ),
            "paid_model_calls": 0,
            "old_2020_binding_used_for_2023": False,
            "unsafe_native_calls": 0,
            "config_only_evidence_passes": 0,
            "primary_dates": primary_dates,
            "fallback_dates_attempted": sorted(
                {
                    row.get("service_date")
                    for row in fallback_execution["results"]
                    if row.get("service_date")
                }
            ),
            "runtime_sources_cataloged": len(catalogs),
            "headroom_trials_passed": passed,
            "candidate_blueprints": len(blueprints),
            "strict_positive_source_count": len(
                {
                    row["complete_source_identity_sha256"]
                    for row in blueprints
                }
            ),
            "best_non_passing_trial": next(
                (
                    row
                    for row in trials
                    if row["status"] != "passed"
                ),
                None,
            ),
            "blocker_counts": blocker_counts,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
