"""Exact-source contracts for native SUMO signal control.

The contract is deliberately independent of benchmark corridor labels.  It is
keyed by the files and runtime that SUMO actually consumed, then validates only
traffic-light identifiers, programs, phases, lanes, and links reported for that
same source identity.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .source_identity import (
    build_sumo_source_identity_payload,
    compute_sumo_source_identity,
    resolve_sumo_input_graph,
)

SCHEMA_VERSION = "1.0"
TRANSPORT = "traci_tcp"
SAFETY_CAUSALITY_SCHEMA_VERSION = "traffic-safety-causality-v1"
SAFETY_EVENT_IDENTITY_MODE = "native_traci_id_list"
SAFETY_SEGMENTS = (
    "initialization",
    "evaluation",
    "clearance_tail",
)
SAFETY_EVENT_TYPES = (
    "collision",
    "emergency",
    "teleport_start",
    "teleport_end",
)
PRIMARY_SAFETY_EVENT_TYPES = (
    "collision",
    "emergency",
    "teleport_start",
)


class RuntimeControlContractError(ValueError):
    """Stable fail-closed error for an invalid or stale native control."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def resolve_sumo_safety_telemetry_contract(
    simulation_api: Any,
) -> dict[str, Any]:
    """Resolve the native SUMO safety counters without zero fallbacks."""
    methods = {
        "collision_number": "getCollidingVehiclesNumber",
        "collision_ids": "getCollidingVehiclesIDList",
        "emergency_number": "getEmergencyStoppingVehiclesNumber",
        "emergency_ids": "getEmergencyStoppingVehiclesIDList",
        "teleport_start_number": "getStartingTeleportNumber",
        "teleport_start_ids": "getStartingTeleportIDList",
        "teleport_end_number": "getEndingTeleportNumber",
        "teleport_end_ids": "getEndingTeleportIDList",
    }
    missing = [
        channel
        for channel, method in methods.items()
        if not callable(getattr(simulation_api, method, None))
    ]
    return {
        "status": "complete" if not missing else "missing",
        "telemetry_methods": methods,
        "identity_mode": SAFETY_EVENT_IDENTITY_MODE,
        "missing_channels": missing,
    }


def classify_sumo_safety_segment(
    sim_time: Any,
    *,
    action_apply_sim_time: Any,
    planned_horizon_end_sim_time: Any,
    actual_end_sim_time: Any,
) -> str | None:
    """Classify one native physics step against explicit trial boundaries."""
    values = (
        sim_time,
        action_apply_sim_time,
        planned_horizon_end_sim_time,
        actual_end_sim_time,
    )
    try:
        current, action, planned_end, actual_end = (
            float(value) for value in values
        )
    except (TypeError, ValueError):
        return None
    if not all(
        math.isfinite(value)
        for value in (current, action, planned_end, actual_end)
    ):
        return None
    if action > planned_end or planned_end > actual_end:
        return None
    if current < action:
        return "initialization"
    if current <= planned_end:
        return "evaluation"
    if current <= actual_end:
        return "clearance_tail"
    return None


def _empty_segment() -> dict[str, Any]:
    row: dict[str, Any] = {"sample_count": 0}
    for event_type in SAFETY_EVENT_TYPES:
        row[f"{event_type}_unique_vehicle_ids"] = []
        row[f"{event_type}_unique_vehicle_count"] = 0
        row[f"{event_type}_event_step_count"] = 0
    row["teleport_primary_safety_event_count"] = 0
    return row


def aggregate_sumo_safety_event_samples(
    samples: Iterable[Mapping[str, Any]],
    *,
    action_apply_sim_time: Any,
    planned_horizon_end_sim_time: Any,
    actual_end_sim_time: Any,
) -> dict[str, Any]:
    """Build stable segment-level native-ID safety evidence for one leg."""
    boundaries = {
        "action_apply_sim_time": action_apply_sim_time,
        "planned_horizon_end_sim_time": (
            planned_horizon_end_sim_time
        ),
        "actual_end_sim_time": actual_end_sim_time,
    }
    boundary_probe = classify_sumo_safety_segment(
        action_apply_sim_time,
        **boundaries,
    )
    rows = [dict(row) for row in samples]
    segments = {name: _empty_segment() for name in SAFETY_SEGMENTS}
    if boundary_probe is None:
        return {
            "schema_version": SAFETY_CAUSALITY_SCHEMA_VERSION,
            "identity_mode": SAFETY_EVENT_IDENTITY_MODE,
            "status": "missing",
            "reason_code": "safety_evidence_missing",
            "sample_count": len(rows),
            "boundaries": boundaries,
            "segments": segments,
            "events": [],
            "semantic_digest": None,
            "errors": ["invalid_or_missing_trial_boundary"],
        }

    event_ids = {
        segment: {
            event_type: set()
            for event_type in SAFETY_EVENT_TYPES
        }
        for segment in SAFETY_SEGMENTS
    }
    event_steps = {
        segment: {
            event_type: 0
            for event_type in SAFETY_EVENT_TYPES
        }
        for segment in SAFETY_SEGMENTS
    }
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    statuses: list[str] = []
    for row in rows:
        status = str(row.get("status") or "missing")
        statuses.append(status)
        errors.extend(str(value) for value in row.get("errors") or [])
        sim_time = row.get("absolute_sim_time")
        segment = classify_sumo_safety_segment(
            sim_time,
            **boundaries,
        )
        if segment is None:
            statuses.append("missing")
            errors.append(f"unclassified_sim_time:{sim_time}")
            continue
        segments[segment]["sample_count"] += 1
        for event_type in SAFETY_EVENT_TYPES:
            ids = sorted(
                {
                    str(value)
                    for value in (
                        (row.get("vehicle_ids") or {}).get(event_type)
                        or []
                    )
                }
            )
            if not ids:
                continue
            event_ids[segment][event_type].update(ids)
            event_steps[segment][event_type] += 1
            events.append(
                {
                    "relative_time": (
                        float(sim_time)
                        - float(action_apply_sim_time)
                    ),
                    "absolute_sim_time": float(sim_time),
                    "segment": segment,
                    "event_type": event_type,
                    "vehicle_ids": ids,
                }
            )
    for segment in SAFETY_SEGMENTS:
        for event_type in SAFETY_EVENT_TYPES:
            ids = sorted(event_ids[segment][event_type])
            segments[segment][
                f"{event_type}_unique_vehicle_ids"
            ] = ids
            segments[segment][
                f"{event_type}_unique_vehicle_count"
            ] = len(ids)
            segments[segment][
                f"{event_type}_event_step_count"
            ] = event_steps[segment][event_type]
        segments[segment]["teleport_primary_safety_event_count"] = (
            segments[segment][
                "teleport_start_unique_vehicle_count"
            ]
        )
    semantic_events = [
        {
            "relative_time": row["relative_time"],
            "segment": row["segment"],
            "event_type": row["event_type"],
            "vehicle_ids": row["vehicle_ids"],
        }
        for row in sorted(
            events,
            key=lambda row: (
                row["relative_time"],
                row["event_type"],
                row["vehicle_ids"],
            ),
        )
    ]
    digest = hashlib.sha256(
        json.dumps(
            semantic_events,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    status = (
        "complete"
        if rows and all(value == "complete" for value in statuses)
        else "identity_mismatch"
        if "identity_mismatch" in statuses
        else "error"
        if "error" in statuses
        else "missing"
    )
    reason = (
        "safety_telemetry_identity_mismatch"
        if status == "identity_mismatch"
        else "safety_evidence_missing"
        if status != "complete"
        else "safety_telemetry_complete"
    )
    return {
        "schema_version": SAFETY_CAUSALITY_SCHEMA_VERSION,
        "identity_mode": SAFETY_EVENT_IDENTITY_MODE,
        "status": status,
        "reason_code": reason,
        "sample_count": len(rows),
        "boundaries": {
            name: float(value) for name, value in boundaries.items()
        },
        "segments": segments,
        "events": semantic_events,
        "semantic_digest": digest,
        "errors": errors,
    }


def _post_control_primary_ids(
    telemetry: Mapping[str, Any],
) -> dict[str, set[str]]:
    result = {
        event_type: set()
        for event_type in PRIMARY_SAFETY_EVENT_TYPES
    }
    segments = telemetry.get("segments") or {}
    for segment in ("evaluation", "clearance_tail"):
        row = segments.get(segment) or {}
        for event_type in PRIMARY_SAFETY_EVENT_TYPES:
            result[event_type].update(
                str(value)
                for value in row.get(
                    f"{event_type}_unique_vehicle_ids"
                )
                or []
            )
    return result


def classify_paired_safety_causality(
    replay_telemetry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify absolute and control-relative safety for four replay legs."""
    legs = (
        "baseline",
        "baseline_repeat",
        "reference",
        "reference_repeat",
    )
    missing = [name for name in legs if name not in replay_telemetry]
    rows = {
        name: replay_telemetry.get(name) or {}
        for name in legs
    }
    noncomplete = [
        name
        for name, row in rows.items()
        if row.get("status") != "complete"
    ]
    boundaries = {
        (
            (row.get("boundaries") or {}).get(
                "action_apply_sim_time"
            ),
            (row.get("boundaries") or {}).get(
                "planned_horizon_end_sim_time"
            ),
        )
        for row in rows.values()
    }
    initialization_present = any(
        any(
            int(
                ((row.get("segments") or {}).get("initialization") or {}).get(
                    f"{event_type}_unique_vehicle_count"
                )
                or 0
            )
            > 0
            for event_type in PRIMARY_SAFETY_EVENT_TYPES
        )
        for row in rows.values()
    )
    if missing or noncomplete or len(boundaries) != 1 or None in next(
        iter(boundaries), (None,)
    ):
        mismatch = any(
            row.get("reason_code")
            == "safety_telemetry_identity_mismatch"
            for row in rows.values()
        )
        return {
            "status": "held",
            "reason_code": (
                "safety_telemetry_identity_mismatch"
                if mismatch
                else "safety_evidence_missing"
            ),
            "telemetry_status": "held",
            "source_episode_safety_status": "held",
            "control_safety_noninferiority_status": "held",
            "telemetry_complete": False,
            "initialization_background_events_present": (
                initialization_present
            ),
        }
    deterministic = (
        rows["baseline"].get("semantic_digest")
        == rows["baseline_repeat"].get("semantic_digest")
        and rows["reference"].get("semantic_digest")
        == rows["reference_repeat"].get("semantic_digest")
    )
    if not deterministic:
        return {
            "status": "held",
            "reason_code": "safety_telemetry_nondeterministic",
            "telemetry_status": "held",
            "source_episode_safety_status": "held",
            "control_safety_noninferiority_status": "held",
            "telemetry_complete": True,
            "initialization_background_events_present": (
                initialization_present
            ),
        }
    primary = {
        name: _post_control_primary_ids(row)
        for name, row in rows.items()
    }
    baseline_max = {
        event_type: max(
            len(primary["baseline"][event_type]),
            len(primary["baseline_repeat"][event_type]),
        )
        for event_type in PRIMARY_SAFETY_EVENT_TYPES
    }
    reference_max = {
        event_type: max(
            len(primary["reference"][event_type]),
            len(primary["reference_repeat"][event_type]),
        )
        for event_type in PRIMARY_SAFETY_EVENT_TYPES
    }
    regression = any(
        reference_max[event_type] > baseline_max[event_type]
        for event_type in PRIMARY_SAFETY_EVENT_TYPES
    )
    background = any(
        values[event_type]
        for values in primary.values()
        for event_type in PRIMARY_SAFETY_EVENT_TYPES
    )
    status = "passed"
    reason = "traffic_safety_passed"
    if regression:
        status = "failed"
        reason = "traffic_control_safety_regression"
    elif background:
        status = "held"
        reason = "traffic_source_safety_background_violation"
    return {
        "status": status,
        "reason_code": reason,
        "telemetry_status": "passed",
        "source_episode_safety_status": (
            "failed" if background else "passed"
        ),
        "control_safety_noninferiority_status": (
            "failed" if regression else "passed"
        ),
        "telemetry_complete": True,
        "deterministic": True,
        "initialization_background_events_present": (
            initialization_present
        ),
        "baseline_conservative_max": baseline_max,
        "reference_conservative_max": reference_max,
    }


def compute_complete_source_identity(
    *,
    sumocfg_sha256: str,
    network_sha256: str,
    route_sha256s: Sequence[str],
    additional_sha256s: Sequence[str],
    recursive_include_sha256s: Sequence[str],
    service_date: str,
    sumo_version: str,
    transport: str,
) -> str:
    """Compatibility wrapper around the canonical SUMO identity contract."""
    payload = build_sumo_source_identity_payload(
        {
            "sumocfg": {"sha256": sumocfg_sha256},
            "network": {"sha256": network_sha256},
            "route_files": [
                {"sha256": value} for value in route_sha256s
            ],
            "additional_files": [
                {"sha256": value} for value in additional_sha256s
            ],
            "recursive_inputs": [
                {"sha256": value}
                for value in recursive_include_sha256s
            ],
        },
        service_date=service_date,
        sumo_version=sumo_version,
        transport=transport,
    )
    return compute_sumo_source_identity(payload)


def require_compatible_binding(
    *,
    binding_network_sha256: str,
    runtime_network_sha256: str,
    declared_tls_id: str,
    declared_program_ids: Iterable[str],
    runtime_programs_by_tls: Mapping[str, Iterable[str]],
) -> None:
    """Reject cross-network, cross-TLS, or program-string binding reuse."""
    if str(binding_network_sha256) != str(runtime_network_sha256):
        raise RuntimeControlContractError(
            "traffic_binding_network_mismatch",
            "the binding network does not match the runtime network",
            binding_network_sha256=binding_network_sha256,
            runtime_network_sha256=runtime_network_sha256,
        )
    tls = str(declared_tls_id)
    if tls not in runtime_programs_by_tls:
        raise RuntimeControlContractError(
            "traffic_binding_tls_missing",
            f"declared TLS {tls!r} is absent from the runtime",
            tls_id=tls,
        )
    runtime_programs = {str(value) for value in runtime_programs_by_tls[tls]}
    missing = sorted({str(value) for value in declared_program_ids} - runtime_programs)
    if missing:
        raise RuntimeControlContractError(
            "traffic_binding_program_missing_on_tls",
            f"declared programs are absent from TLS {tls!r}",
            tls_id=tls,
            missing_program_ids=missing,
            runtime_program_ids=sorted(runtime_programs),
        )


def _finite_positive(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def classify_vehicle_movement_link_indices(
    controlled_links: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Classify links from native SUMO controlled-link identities.

    SUMO may expose internal junction connections as controlled links.  Such
    links have an internal ``incoming_lane`` (``:junction_*``) and are not
    independent vehicle movements that need a green phase in every selectable
    program.  The classification is derived solely from the runtime link
    identities; it does not use a program id, corridor label, or hard-coded
    network index.
    """
    vehicle: list[int] = []
    structural: list[int] = []
    invalid: list[int] = []
    for index, group in enumerate(controlled_links):
        if not isinstance(group, Sequence) or isinstance(group, (str, bytes)):
            invalid.append(index)
            continue
        if not group or any(not isinstance(link, Mapping) for link in group):
            invalid.append(index)
            continue
        incoming_lanes = {
            str(link.get("incoming_lane") or "")
            for link in group
        }
        # Missing lane identity is malformed runtime evidence, not an
        # internal structural link.  Treating a mixed group as structural
        # could exempt it from the green-phase safety checks.
        if not incoming_lanes or not all(incoming_lanes):
            invalid.append(index)
        elif all(lane.startswith(":") for lane in incoming_lanes):
            structural.append(index)
        else:
            vehicle.append(index)
    return {
        "schema_version": "traffic-vehicle-movement-links-v1",
        "classification_method": "native_controlled_links_incoming_lane_identity",
        "vehicle_movement_link_indices": vehicle,
        "structural_internal_link_indices": structural,
        "invalid_link_indices": invalid,
        "controlled_link_count": len(controlled_links),
        "status": "passed" if vehicle and not invalid else "failed",
    }


def validate_program_logic(
    *,
    tls_id: str,
    program_id: str,
    controlled_link_count: int,
    parsed_program_ids: set[str],
    phases: Sequence[Mapping[str, Any]],
    vehicle_movement_link_indices: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Validate a runtime-reported selectable program without applying it."""
    if str(program_id) not in parsed_program_ids:
        raise RuntimeControlContractError(
            "traffic_binding_program_missing_on_tls",
            "runtime program is absent from parsed assets for this source",
            tls_id=str(tls_id),
            program_id=str(program_id),
        )
    link_count = int(controlled_link_count)
    if link_count <= 0 or not phases:
        raise RuntimeControlContractError(
            "traffic_program_logic_invalid",
            "program has no controlled links or phases",
            tls_id=str(tls_id),
            program_id=str(program_id),
        )
    if vehicle_movement_link_indices is None:
        required_indices = list(range(link_count))
        structural_indices: list[int] = []
    else:
        required_indices = sorted({int(value) for value in vehicle_movement_link_indices})
        if not required_indices or any(
            value < 0 or value >= link_count for value in required_indices
        ):
            raise RuntimeControlContractError(
                "traffic_vehicle_movement_link_set_invalid",
                "vehicle-movement link indices must be non-empty and in range",
                tls_id=str(tls_id),
                program_id=str(program_id),
                controlled_link_count=link_count,
                vehicle_movement_link_indices=required_indices,
            )
        structural_indices = [
            index for index in range(link_count) if index not in required_indices
        ]
    green_seen = [False] * link_count
    for index, phase in enumerate(phases):
        duration = _finite_positive(phase.get("duration"))
        state = str(phase.get("state") or "")
        if duration is None or len(state) != link_count:
            raise RuntimeControlContractError(
                "traffic_program_logic_invalid",
                "phase duration or state cardinality is invalid",
                tls_id=str(tls_id),
                program_id=str(program_id),
                phase_index=index,
            )
        for link_index, signal in enumerate(state):
            if signal in {"g", "G"}:
                green_seen[link_index] = True
        for target in phase.get("next") or []:
            try:
                next_index = int(target)
            except (TypeError, ValueError) as exc:
                raise RuntimeControlContractError(
                    "traffic_program_logic_invalid",
                    "phase next index is not integral",
                    tls_id=str(tls_id),
                    program_id=str(program_id),
                    phase_index=index,
                ) from exc
            if next_index < 0 or next_index >= len(phases):
                raise RuntimeControlContractError(
                    "traffic_program_logic_invalid",
                    "phase next index is outside the program",
                    tls_id=str(tls_id),
                    program_id=str(program_id),
                    phase_index=index,
                    next_phase=next_index,
                )
    missing_indices = [
        index for index in required_indices if not green_seen[index]
    ]
    if missing_indices:
        raise RuntimeControlContractError(
            "traffic_program_missing_green",
            "one or more controlled links never receive a green movement",
            tls_id=str(tls_id),
            program_id=str(program_id),
            permanently_red_link_indices=[
                index for index in missing_indices
            ],
            structural_internal_link_indices=structural_indices,
            vehicle_movement_link_indices=required_indices,
        )
    return {
        "status": "passed",
        "tls_id": str(tls_id),
        "program_id": str(program_id),
        "controlled_link_count": link_count,
        "phase_count": len(phases),
        "vehicle_movement_link_indices": required_indices,
        "structural_internal_link_indices": structural_indices,
    }


def validate_runtime_program_selection(
    *,
    tls_id: str,
    requested_program_id: str,
    runtime_program_ids: Iterable[str],
) -> dict[str, Any]:
    """Require the selected program in the live TLS inventory."""
    requested = str(requested_program_id)
    available = sorted({str(value) for value in runtime_program_ids})
    if requested not in available:
        raise RuntimeControlContractError(
            "program_not_available_on_runtime_tls",
            "requested program is absent from the live TLS inventory",
            tls_id=str(tls_id),
            requested_program_id=requested,
            runtime_program_ids=available,
        )
    return {
        "status": "passed",
        "tls_id": str(tls_id),
        "program_id": requested,
        "runtime_program_ids": available,
    }


def validate_phase_duration_request(
    *,
    observed_program: str,
    observed_phase: int,
    runtime_program: str,
    runtime_phase: int,
    runtime_state: Mapping[str, Any],
    requested_remaining_duration: float,
) -> dict[str, Any]:
    """Validate TraCI remaining-duration semantics before any mutation."""
    if (
        str(observed_program) != str(runtime_program)
        or int(observed_phase) != int(runtime_phase)
    ):
        raise RuntimeControlContractError(
            "traffic_runtime_contract_stale",
            "the runtime program or phase changed after observation",
            observed_program=observed_program,
            runtime_program=runtime_program,
            observed_phase=observed_phase,
            runtime_phase=runtime_phase,
        )
    state = str(runtime_state.get("state") or "")
    if not any(signal in {"g", "G"} for signal in state) or set(state) <= {"r", "R"}:
        raise RuntimeControlContractError(
            "traffic_phase_not_green",
            "phase-duration control requires a live green phase",
            state=state,
        )
    if any(signal in {"y", "Y"} for signal in state):
        raise RuntimeControlContractError(
            "traffic_phase_not_green",
            "yellow phases cannot be extended",
            state=state,
        )
    minimum = _finite_positive(runtime_state.get("min_duration"))
    maximum = _finite_positive(runtime_state.get("max_duration"))
    if minimum is None or maximum is None or minimum > maximum:
        raise RuntimeControlContractError(
            "traffic_phase_bounds_invalid",
            "phase-duration bounds must be explicit, finite, and positive",
            min_duration=runtime_state.get("min_duration"),
            max_duration=runtime_state.get("max_duration"),
        )
    spent_raw = runtime_state.get("spent_duration", 0.0)
    try:
        spent = float(spent_raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeControlContractError(
            "traffic_phase_bounds_invalid",
            "current phase elapsed duration must be finite and non-negative",
            spent_duration=spent_raw,
        ) from exc
    if not math.isfinite(spent) or spent < 0:
        raise RuntimeControlContractError(
            "traffic_phase_bounds_invalid",
            "current phase elapsed duration must be finite and non-negative",
            spent_duration=spent_raw,
        )
    remaining_minimum = max(0.0, minimum - spent)
    remaining_maximum = max(remaining_minimum, maximum - spent)
    requested = _finite_positive(requested_remaining_duration)
    if (
        requested is None
        or requested < remaining_minimum
        or requested > remaining_maximum
    ):
        raise RuntimeControlContractError(
            "traffic_phase_duration_out_of_range",
            "requested remaining duration is outside runtime bounds",
            requested_remaining_duration=requested_remaining_duration,
            remaining_min_duration=remaining_minimum,
            remaining_max_duration=remaining_maximum,
        )
    if not runtime_state.get("controlled_lanes") or not runtime_state.get(
        "controlled_links"
    ):
        raise RuntimeControlContractError(
            "traffic_program_logic_invalid",
            "phase-duration control requires controlled lanes and links",
        )
    return {
        "status": "passed",
        "program": str(runtime_program),
        "phase": int(runtime_phase),
        "requested_remaining_duration": requested,
        "min_duration": minimum,
        "max_duration": maximum,
        "spent_duration": spent,
        "remaining_min_duration": remaining_minimum,
        "remaining_max_duration": remaining_maximum,
    }


def _open_xml(path: Path) -> ET.Element:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return ET.parse(handle).getroot()


def resolve_sumocfg_asset_graph(sumocfg: Path) -> dict[str, Any]:
    """Compatibility wrapper for the canonical recursive input resolver."""
    try:
        return resolve_sumo_input_graph(sumocfg)
    except ValueError as exc:
        raise RuntimeControlContractError(
            "traffic_source_identity_mismatch",
            str(exc),
        ) from exc


def parsed_program_ids_by_tls(source_assets: Mapping[str, Any]) -> dict[str, set[str]]:
    """Index ``tlLogic`` programs from the exact parsed source graph."""
    paths: list[Path] = []
    network = source_assets.get("network") or {}
    if network.get("path"):
        paths.append(Path(str(network["path"])))
    for key in ("additional_files", "recursive_inputs"):
        for row in source_assets.get(key) or []:
            raw = row.get("path")
            if raw:
                paths.append(Path(str(raw)))
    programs: dict[str, set[str]] = {}
    for path in dict.fromkeys(paths):
        try:
            root = _open_xml(path)
        except (OSError, ET.ParseError):
            continue
        for logic in root.iter("tlLogic"):
            tls_id = str(logic.get("id") or "")
            program_id = str(logic.get("programID") or "")
            if tls_id and program_id:
                programs.setdefault(tls_id, set()).add(program_id)
    return programs
