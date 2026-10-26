"""Future-capture contract for Protocol-2.1 Traffic evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TRAFFIC_CAPTURE_SCHEMA_VERSION = "1.0"
VEHICLE_EVENT_FIELDS = (
    "vehicle_id",
    "simulation_time",
    "edge_id",
    "lane_id",
    "route_id",
    "trip_id",
    "source_event_ids",
    "tls_context",
    "controlled_link_context",
    "phase_context",
)
SOURCE_LINEAGE_FIELDS = (
    "source_trip_id",
    "runtime_vehicle_id",
    "event_vehicle_id",
    "route_id",
    "edge_sequence",
    "evidence_ids",
)
CONTROL_CONTEXT_FIELDS = (
    "tls_context",
    "controlled_link_context",
    "phase_context",
)


def traffic_capture_schema() -> dict[str, Any]:
    """Return the canonical future-capture requirements."""
    return {
        "schema_version": TRAFFIC_CAPTURE_SCHEMA_VERSION,
        "vehicle_event_schema": {
            "required_fields": list(VEHICLE_EVENT_FIELDS),
        },
        "source_lineage_schema": {
            "required_fields": list(SOURCE_LINEAGE_FIELDS),
            "required_chain": [
                "source_trip",
                "runtime_vehicle",
                "event",
            ],
        },
        "control_context_schema": {
            "required_fields": list(CONTROL_CONTEXT_FIELDS),
            "association": (
                "vehicle_lane_to_controlled_link_to_tls"
            ),
        },
        "safety_attribution_categories": {
            "control_regression_candidate": (
                "same vehicle or same source lineage, and reference worse"
            ),
            "traffic_variance_candidate": (
                "different vehicle with same source lineage"
            ),
            "background_variance": "different source lineage",
            "unknown": "capture context missing",
        },
        "required_for_safety_attribution": [
            "complete_vehicle_event",
            "verified_source_lineage",
            "complete_control_context",
        ],
        "required_for_headroom": [
            "verified_source_lineage",
            "valid_task_contract",
        ],
    }


def _missing_fields(
    payload: Mapping[str, Any],
    required: tuple[str, ...],
) -> list[str]:
    return [
        field
        for field in required
        if payload.get(field) in (None, "", [], {})
    ]


def validate_vehicle_event_capture(
    event: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one captured safety event without filling gaps."""
    missing = _missing_fields(event, VEHICLE_EVENT_FIELDS)
    return {
        "status": "missing" if missing else "complete",
        "missing_fields": missing,
    }


def validate_source_lineage(
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Require an evidence-linked source-trip to event chain."""
    missing = _missing_fields(lineage, SOURCE_LINEAGE_FIELDS)
    identifiers_match = (
        lineage.get("runtime_vehicle_id")
        == lineage.get("event_vehicle_id")
    )
    lineage_verified = not missing and identifiers_match
    return {
        "status": "complete" if lineage_verified else "missing",
        "missing_fields": missing,
        "identifiers_match": identifiers_match,
        "lineage_verified": lineage_verified,
    }


def classify_capture_attribution(
    *,
    baseline_event: Mapping[str, Any],
    reference_event: Mapping[str, Any],
    reference_worse: bool,
) -> dict[str, Any]:
    """Classify captured events only after the required context exists."""
    baseline = validate_vehicle_event_capture(baseline_event)
    reference = validate_vehicle_event_capture(reference_event)
    if baseline["status"] != "complete" or reference["status"] != "complete":
        return {
            "status": "held",
            "classification": "unknown",
            "reason": "missing_capture_contract",
        }

    same_vehicle = (
        baseline_event["vehicle_id"] == reference_event["vehicle_id"]
    )
    same_source_lineage = bool(
        set(str(value) for value in baseline_event["source_event_ids"])
        & set(str(value) for value in reference_event["source_event_ids"])
    )
    if reference_worse and (same_vehicle or same_source_lineage):
        classification = "control_regression_candidate"
    elif not same_vehicle and same_source_lineage:
        classification = "traffic_variance_candidate"
    elif not same_source_lineage:
        classification = "background_variance"
    else:
        classification = "unknown"
    return {
        "status": "evaluable",
        "classification": classification,
        "reason": classification,
    }


def evaluate_headroom_capture_prerequisites(
    *,
    vehicle_lineage_status: str,
    task_contract_status: str,
) -> dict[str, Any]:
    """Block headroom until capture lineage and task semantics exist."""
    allowed = (
        vehicle_lineage_status == "complete"
        and task_contract_status == "valid"
    )
    return {
        "status": "ready" if allowed else "blocked",
        "reason": (
            "capture_contract_complete"
            if allowed
            else "missing_capture_contract"
        ),
        "headroom_allowed": allowed,
    }
