"""Canonical runtime event records for protocol-2.1."""

from __future__ import annotations

import math
from typing import Any

from .event_protocol import resolve_event_decision

VALID_ORIGINS = frozenset(
    {
        "source_schedule",
        "declared_perturbation",
        "agent_caused",
        "endogenous_completion",
        "unknown",
    }
)
_ENDOGENOUS_TYPES = frozenset(
    {
        "job_completed",
        "gpu_reservation_expired",
        "delivery_arrived",
        "operation_completed",
    }
)
_SOURCE_SCHEDULE_TYPES = frozenset(
    {
        "job_arrival",
        "inventory_demand_realized",
        "demand_realization",
        "load_change",
        "generation_change",
        "generation_ramp",
        "pv_update",
        "tariff_change",
        "traffic_demand_change",
        "vehicle_arrival",
        "order_arrival",
    }
)


def _origin(event: dict[str, Any]) -> str:
    explicit = str(event.get("origin") or "").strip()
    if explicit in VALID_ORIGINS:
        return explicit
    event_type = str(event.get("type") or event.get("kind") or "")
    if event_type in _SOURCE_SCHEDULE_TYPES:
        return "source_schedule"
    if event_type in _ENDOGENOUS_TYPES or event_type.endswith("_completed"):
        return "endogenous_completion"
    source = str(event.get("source") or "").lower()
    if source in {"tool", "agent", "control"} or event.get("agent_caused") is True:
        return "agent_caused"
    if event.get("declared_perturbation") is True:
        return "declared_perturbation"
    # Fail closed: an unlabelled backend event is not credited as exogenous
    # or silently relabelled as an endogenous completion.
    return "unknown"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def canonicalize_runtime_events(
    events: list[dict[str, Any]],
    *,
    applied_tick: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(events):
        if not isinstance(raw, dict):
            continue
        event_type = str(raw.get("type") or raw.get("kind") or "unknown")
        origin = _origin(raw)
        value = _finite(
            raw.get("materiality_value", raw.get("impact_value"))
        )
        threshold = _finite(
            raw.get("materiality_threshold", raw.get("impact_threshold"))
        )
        if raw.get("materiality_passed") is not None:
            materiality_passed = raw.get("materiality_passed") is True
        elif value is not None and threshold is not None:
            materiality_passed = abs(value) >= abs(threshold)
        else:
            materiality_passed = False
        changed = [
            str(field)
            for field in (
                raw.get("changed_state_fields")
                or raw.get("affected_state_fields")
                or []
            )
            if str(field)
        ]
        event_decision = resolve_event_decision(raw)
        record = {
            "event_id": str(
                raw.get("event_id")
                or raw.get("id")
                or f"{event_type}@{applied_tick}:{index}"
            ),
            "event_type": event_type,
            "origin": origin,
            "declared_event": dict(raw.get("declared_event") or {}),
            "applied_tick": int(applied_tick),
            "visibility": "hidden" if raw.get("hidden") else "visible",
            "event_class": event_decision.decision_class.value,
            "decision_required": event_decision.requires_decision,
            "event_decision_declared_by": event_decision.declared_by,
            "event_contract_violations": list(
                event_decision.violation_codes
            ),
            "changed_state_fields": changed,
            "materiality_metric": raw.get(
                "materiality_metric", raw.get("impact_metric")
            ),
            "materiality_value": value,
            "materiality_threshold": threshold,
            "materiality_passed": materiality_passed,
            "materiality": {
                "metric": raw.get(
                    "materiality_metric", raw.get("impact_metric")
                ),
                "value": value,
                "threshold": threshold,
                "passed": materiality_passed,
            },
            "material_exogenous": bool(
                origin in {"source_schedule", "declared_perturbation"}
                and materiality_passed
                and changed
            ),
        }
        if origin == "agent_caused":
            for key in (
                "call_id",
                "tool_name",
                "requested_action",
                "applied_action",
                "before_state_digest",
                "after_state_digest",
                "outcome_tick",
                "evidence_ids",
                "action_to_outcome_edge",
            ):
                if key in raw:
                    record[key] = raw[key]
        for key in (
            "causal_parent_event_id",
            "causal_call_id",
            "evidence_ids",
            "before_state_digest",
            "after_state_digest",
            "response_window_required",
            "response_opportunity_tick",
            "terminal_response_window_missing",
        ):
            if key in raw:
                record[key] = raw[key]
        records.append(record)
    return records
