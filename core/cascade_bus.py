"""
core.cascade_bus — Versioned message bus for cross-domain coupling.

OPERATE v0.1 ships only one domain (power_grid), but the bus is
reserved from day one so v0.2 can plug in SUMO traffic, v0.3 can plug in
RoboCup-Rescue, and a single Grid2Op fault can cascade into:

  - SUMO traffic signal failure (`signal_outage`)
  - RCRS hospital ICU shortfall (`critical_load_loss`)
  - downstream LLM agent observation containing all of the above.

The bus is intentionally a typed schema (NOT an opaque dict) so cross-
runtime (Python ↔ Java RCRS over JSON RPC, Python ↔ Julia
PowerModelsRestoration over docker) is feasible in v0.3 without rewriting
the contract.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

CASCADE_BUS_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CascadeEvent:
    """A message published by one domain that other domains may subscribe to.

    ``event_type`` is a free string but conventionally a triple
    ``"<source_domain>.<noun>.<verb>"``, e.g. ``"power_grid.line.outage"``,
    ``"power_grid.bus.voltage_violation"``, ``"traffic.signal.failed"``,
    ``"disaster.hazard.earthquake"``.

    ``severity`` is bounded ``[0, 1]`` so subscribers across domains can
    threshold uniformly.
    """

    schema_version: str = CASCADE_BUS_SCHEMA_VERSION
    event_type: str = ""
    source_domain: str = ""
    tick: int = 0
    severity: float = 0.0
    location: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None  # ties together a cascade chain

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_type": self.event_type,
            "source_domain": self.source_domain,
            "tick": self.tick,
            "severity": float(self.severity),
            "location": dict(self.location),
            "payload": dict(self.payload),
            "correlation_id": self.correlation_id,
        }


Subscriber = Callable[[CascadeEvent], None]


class CascadeBus:
    """In-process pub/sub for v0.1; pluggable transports for v0.3.

    v0.1 implementation is a simple in-memory queue + topic dispatcher.
    v0.3 will swap the in-memory transport for an out-of-process JSON-RPC
    bridge so non-Python simulators can join. The public API stays the same.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._history: list[CascadeEvent] = []
        # Subscriber callbacks that raised during delivery are swallowed so a
        # buggy subscriber cannot crash a domain, but they are *counted* here so
        # the failure is observable (audits/tests can assert
        # ``failed_deliveries == 0`` instead of silently losing a cascade edge).
        self._failed_deliveries: int = 0
        self._last_delivery_errors: list[dict[str, Any]] = []
        self._delivery_batch_depth = 0
        self._pending_delivery: list[CascadeEvent] = []

    # ── Pub / sub ───────────────────────────────────────────────────────

    def subscribe(self, topic: str, fn: Subscriber) -> None:
        """Subscribe to a topic pattern. ``"*"`` matches everything."""
        self._subscribers[topic].append(fn)

    def publish(self, event: CascadeEvent) -> int:
        """Publish an event. Returns the number of subscribers it fanned out to."""
        self._history.append(event)
        if self._delivery_batch_depth:
            self._pending_delivery.append(event)
            return 0
        return self._deliver(event)

    def begin_delivery_batch(self) -> None:
        """Defer callbacks until a synchronous multi-environment tick ends."""
        self._delivery_batch_depth += 1

    def end_delivery_batch(self) -> int:
        """Flush one delivery batch and return total successful callbacks."""
        if self._delivery_batch_depth <= 0:
            raise RuntimeError("cascade delivery batch is not active")
        self._delivery_batch_depth -= 1
        if self._delivery_batch_depth:
            return 0
        pending, self._pending_delivery = self._pending_delivery, []
        return sum(self._deliver(event) for event in pending)

    def _deliver(self, event: CascadeEvent) -> int:
        delivered = 0
        for topic, fns in self._subscribers.items():
            if (
                topic == "*"
                or topic == event.event_type
                or _topic_matches(topic, event.event_type)
            ):
                for fn in fns:
                    try:
                        fn(event)
                        delivered += 1
                    except Exception as exc:
                        # Bus must never crash a domain, but the dropped edge
                        # must not vanish silently: count it and keep a bounded
                        # trail so it surfaces in audits/tests.
                        self._failed_deliveries += 1
                        if len(self._last_delivery_errors) < 64:
                            self._last_delivery_errors.append(
                                {
                                    "event_type": event.event_type,
                                    "topic": topic,
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "tick": event.tick,
                                }
                            )
        return delivered

    # ── Inspection ──────────────────────────────────────────────────────

    @property
    def history(self) -> list[CascadeEvent]:
        return list(self._history)

    def history_for(self, tick: int) -> list[CascadeEvent]:
        return [e for e in self._history if e.tick == tick]

    @property
    def failed_deliveries(self) -> int:
        """Number of subscriber callbacks that raised during ``publish``.

        A healthy run should keep this at ``0``; a non-zero value means a
        cross-domain cascade edge was dropped because a subscriber raised.
        """
        return self._failed_deliveries

    @property
    def last_delivery_errors(self) -> list[dict[str, Any]]:
        """Bounded trail (max 64) of recent swallowed subscriber failures."""
        return list(self._last_delivery_errors)

    def reset(self) -> None:
        self._subscribers.clear()
        self._history.clear()
        self._failed_deliveries = 0
        self._last_delivery_errors.clear()


def _topic_matches(pattern: str, event_type: str) -> bool:
    """Glob-style topic match: ``power_grid.*`` matches any power_grid event."""
    if pattern == event_type:
        return True
    if pattern.endswith(".*"):
        prefix = pattern[:-2]
        return event_type.startswith(prefix + ".")
    return False
