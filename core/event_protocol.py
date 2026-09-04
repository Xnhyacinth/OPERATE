"""Backend-agnostic decision semantics for realized environment events.

The simulator may emit arbitrarily rich domain-native event payloads.  The
runner must not guess whether an unfamiliar payload warrants another model
turn: doing so turns routine telemetry into an implicit per-tick prompt.  This
module is the narrow contract between those payloads and the supervisory
coordinator.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

EVENT_DECISION_CONTRACT_VERSION = "1.0"
OPTIONAL_PLAN_WAKE_REASONS = frozenset(
    {"visible_event", "forecast_update", "delayed_tool"}
)


class EventDecisionClass(str, Enum):
    """Portable event classes understood by the runner."""

    ALARM = "alarm"
    TASK = "task"
    SAFETY = "safety"
    FORECAST = "forecast"
    ROUTINE = "routine"
    TELEMETRY = "telemetry"
    LIFECYCLE = "lifecycle"
    AGENT_OUTCOME = "agent_outcome"
    UNDECLARED = "undeclared"


EVENT_CLASS_REGISTRY = MappingProxyType(
    {
        EventDecisionClass.ALARM: (True, "visible_event"),
        EventDecisionClass.TASK: (True, "mandatory_task_event"),
        EventDecisionClass.SAFETY: (True, "safety_warning"),
        EventDecisionClass.FORECAST: (True, "forecast_update"),
        EventDecisionClass.ROUTINE: (False, None),
        EventDecisionClass.TELEMETRY: (False, None),
        EventDecisionClass.LIFECYCLE: (False, None),
        EventDecisionClass.AGENT_OUTCOME: (False, None),
        EventDecisionClass.UNDECLARED: (False, None),
    }
)


@dataclass(frozen=True, slots=True)
class EventDecisionResolution:
    """Resolved wake decision plus machine-readable contract diagnostics."""

    requires_decision: bool
    decision_class: EventDecisionClass
    interrupt_reason: str | None
    declared_by: str
    violation_codes: tuple[str, ...] = ()


def _declared_class(event: dict[str, Any]) -> tuple[EventDecisionClass, list[str]]:
    violations: list[str] = []
    raw_class = event.get("event_class")
    if raw_class is not None:
        if isinstance(raw_class, EventDecisionClass):
            return raw_class, violations
        try:
            return EventDecisionClass(str(raw_class)), violations
        except ValueError:
            violations.append("unknown_event_class")
            return EventDecisionClass.UNDECLARED, violations

    origin = str(event.get("origin") or "")
    if origin == "agent_caused":
        return EventDecisionClass.AGENT_OUTCOME, violations
    if origin == "endogenous_completion":
        return EventDecisionClass.LIFECYCLE, violations
    return EventDecisionClass.UNDECLARED, violations


def resolve_event_decision(event: dict[str, Any]) -> EventDecisionResolution:
    """Resolve whether a visible event warrants a supervisory model turn.

    Explicit boolean flags may refine a valid typed class.  Conflicting flags
    are audited and resolve conservatively to actionable only when that class
    is registered as actionable.  An unknown or undeclared event fails closed
    to *no wake* and is reported as a contract violation.
    """

    decision_class, violations = _declared_class(event)
    explicit: dict[str, bool] = {}
    for field in ("decision_required", "actionable"):
        if field not in event:
            continue
        value = event[field]
        if isinstance(value, bool):
            explicit[field] = value
        else:
            violations.append(f"invalid_{field}_flag")

    if len(set(explicit.values())) > 1:
        violations.append("conflicting_actionability_flags")

    if explicit and decision_class is EventDecisionClass.UNDECLARED:
        requires_decision = False
        interrupt_reason = None
        declared_by = "explicit_flags"
        if "unknown_event_class" not in violations:
            violations.append("missing_event_decision_contract")
    elif explicit:
        requested_decision = any(explicit.values())
        class_requires_decision, class_interrupt_reason = EVENT_CLASS_REGISTRY[
            decision_class
        ]
        requires_decision = requested_decision and class_requires_decision
        declared_by = "explicit_flags"
        if requested_decision and not class_requires_decision:
            violations.append("actionability_class_mismatch")
        if not requires_decision:
            interrupt_reason = None
        else:
            interrupt_reason = class_interrupt_reason or (
                "mandatory_task_event"
                if explicit.get("decision_required") is True
                else "visible_event"
            )
    elif decision_class is not EventDecisionClass.UNDECLARED:
        requires_decision, interrupt_reason = EVENT_CLASS_REGISTRY[decision_class]
        declared_by = "event_class" if event.get("event_class") is not None else "origin"
    else:
        requires_decision = False
        interrupt_reason = None
        declared_by = "missing"
        violations.append("missing_event_decision_contract")

    return EventDecisionResolution(
        requires_decision=requires_decision,
        decision_class=decision_class,
        interrupt_reason=interrupt_reason,
        declared_by=declared_by,
        violation_codes=tuple(sorted(set(violations))),
    )


def audit_event_decision_contract(
    event: dict[str, Any],
    *,
    event_index: int,
) -> dict[str, Any] | None:
    """Return an audit row for a malformed event, redacting hidden identity."""

    resolution = resolve_event_decision(event)
    if not resolution.violation_codes:
        return None
    hidden = event.get("hidden") is True
    event_type = str(event.get("type") or event.get("kind") or "unknown")
    event_id = str(event.get("event_id") or event.get("id") or "")
    row: dict[str, Any] = {
        "event_index": int(event_index),
        "event_type": "redacted" if hidden else event_type,
        "visibility": "hidden" if hidden else "visible",
        "violation_codes": list(resolution.violation_codes),
    }
    if hidden:
        opaque_ref = f"{event_index}:{event_type}:{event_id}".encode()
        row["event_ref_sha256"] = hashlib.sha256(opaque_ref).hexdigest()
    elif event_id:
        row["event_id"] = event_id
    return row


__all__ = [
    "EVENT_DECISION_CONTRACT_VERSION",
    "EVENT_CLASS_REGISTRY",
    "OPTIONAL_PLAN_WAKE_REASONS",
    "EventDecisionClass",
    "EventDecisionResolution",
    "audit_event_decision_contract",
    "resolve_event_decision",
]
