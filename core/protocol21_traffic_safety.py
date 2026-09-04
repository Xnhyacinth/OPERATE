"""Identity-aware safety causality for Protocol-2.1 Traffic diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SAFETY_EVENT_TYPES = ("collision", "emergency")
SAFETY_EVENT_CONTEXT_SCHEMA_VERSION = "1.1"
SAFETY_EVENT_CONTEXT_FIELDS = (
    "vehicle_id",
    "simulation_time",
    "edge_id",
    "lane_id",
    "route_id",
    "trip_id",
    "source_event_ids",
    "tls_context",
    "phase_context",
)


def _event_ids(
    leg: Mapping[str, Sequence[str]],
) -> dict[str, set[str]]:
    return {
        event_type: {
            str(vehicle_id)
            for vehicle_id in leg.get(event_type, ())
            if str(vehicle_id)
        }
        for event_type in SAFETY_EVENT_TYPES
    }


def classify_safety_identity_causality(
    *,
    baseline: Mapping[str, Sequence[str]],
    baseline_repeat: Mapping[str, Sequence[str]],
    reference: Mapping[str, Sequence[str]],
    reference_repeat: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Compare native safety-event identities across paired replay legs."""
    legs = {
        "baseline": _event_ids(baseline),
        "baseline_repeat": _event_ids(baseline_repeat),
        "reference": _event_ids(reference),
        "reference_repeat": _event_ids(reference_repeat),
    }
    repeat_deterministic = (
        legs["baseline"] == legs["baseline_repeat"]
        and legs["reference"] == legs["reference_repeat"]
    )
    count_delta = {
        event_type: (
            len(legs["reference"][event_type])
            - len(legs["baseline"][event_type])
        )
        for event_type in SAFETY_EVENT_TYPES
    }
    identity_delta = {
        event_type: {
            "baseline_only": sorted(
                legs["baseline"][event_type]
                - legs["reference"][event_type]
            ),
            "reference_only": sorted(
                legs["reference"][event_type]
                - legs["baseline"][event_type]
            ),
            "overlap": sorted(
                legs["baseline"][event_type]
                & legs["reference"][event_type]
            ),
        }
        for event_type in SAFETY_EVENT_TYPES
    }

    reason_codes: list[str] = []
    if not repeat_deterministic:
        status = "held"
        reason_codes.append("telemetry_nondeterministic")
    elif any(delta > 0 for delta in count_delta.values()):
        status = "failed"
        reason_codes.append("traffic_control_safety_regression")
    elif any(
        delta["baseline_only"] or delta["reference_only"]
        for delta in identity_delta.values()
    ):
        status = "held"
        reason_codes.append("safety_event_identity_shift")
    elif any(legs["baseline"][event_type] for event_type in SAFETY_EVENT_TYPES):
        status = "held"
        reason_codes.append("traffic_source_safety_background_violation")
    else:
        status = "passed"

    return {
        "status": status,
        "reason_codes": reason_codes,
        "count_delta": count_delta,
        "identity_delta": identity_delta,
        "repeat_deterministic": repeat_deterministic,
    }


def classify_safety_attribution(
    *,
    baseline_ids: Sequence[str],
    baseline_repeat_ids: Sequence[str],
    reference_ids: Sequence[str],
    reference_repeat_ids: Sequence[str],
) -> dict[str, Any]:
    """Classify paired event identities without inventing causal context."""
    baseline = {str(value) for value in baseline_ids if str(value)}
    baseline_repeat = {
        str(value) for value in baseline_repeat_ids if str(value)
    }
    reference = {str(value) for value in reference_ids if str(value)}
    reference_repeat = {
        str(value) for value in reference_repeat_ids if str(value)
    }
    repeat_deterministic = (
        baseline == baseline_repeat
        and reference == reference_repeat
    )
    if not repeat_deterministic:
        status = "held"
        reason = "safety_telemetry_nondeterministic"
    elif len(reference) > len(baseline):
        status = "failed"
        reason = "traffic_control_safety_regression"
    elif baseline != reference:
        status = "held"
        reason = "safety_event_identity_shift"
    elif baseline:
        status = "held"
        reason = "traffic_source_safety_background_violation"
    else:
        status = "passed"
        reason = "traffic_safety_passed"
    return {
        "status": status,
        "reason": reason,
        "repeat_deterministic": repeat_deterministic,
        "baseline_ids": sorted(baseline),
        "reference_ids": sorted(reference),
        "identity_overlap": sorted(baseline & reference),
        "baseline_only_ids": sorted(baseline - reference),
        "reference_only_ids": sorted(reference - baseline),
        "identity_shift": baseline != reference,
    }


def validate_safety_event_context(
    event: Mapping[str, Any],
) -> dict[str, Any]:
    """Require vehicle-specific context for causal safety attribution."""
    missing_fields = []
    for field in SAFETY_EVENT_CONTEXT_FIELDS:
        value = event.get(field)
        if value is None or value == "" or value == [] or value == {}:
            missing_fields.append(field)
    status = "held" if missing_fields else "complete"
    return {
        "schema_version": SAFETY_EVENT_CONTEXT_SCHEMA_VERSION,
        "event_context_required": True,
        "event_fields": list(SAFETY_EVENT_CONTEXT_FIELDS),
        "status": status,
        "reason": (
            "safety_event_context_missing"
            if missing_fields
            else "safety_event_context_complete"
        ),
        "missing_fields": missing_fields,
        "event": dict(event),
    }


def classify_safety_event_attribution(
    *,
    baseline_event: Mapping[str, Any],
    reference_event: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify complete paired event records without claiming causality."""
    baseline = validate_safety_event_context(baseline_event)
    reference = validate_safety_event_context(reference_event)
    if baseline["status"] != "complete" or reference["status"] != "complete":
        return {
            "status": "held",
            "reason": "safety_event_context_missing",
            "classification": "missing_context",
            "baseline_context": baseline,
            "reference_context": reference,
        }

    if baseline_event["vehicle_id"] == reference_event["vehicle_id"]:
        classification = "control_safety_regression_candidate"
    else:
        baseline_sources = {
            str(value) for value in baseline_event["source_event_ids"]
        }
        reference_sources = {
            str(value) for value in reference_event["source_event_ids"]
        }
        same_flow = bool(
            baseline_event["route_id"] == reference_event["route_id"]
            and baseline_sources & reference_sources
        )
        classification = (
            "traffic_variance_candidate"
            if same_flow
            else "background_variance_candidate"
        )
    return {
        "status": "evaluable",
        "reason": classification,
        "classification": classification,
        "baseline_context": baseline,
        "reference_context": reference,
    }
