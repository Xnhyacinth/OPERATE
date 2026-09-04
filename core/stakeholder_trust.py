"""
core.stakeholder_trust — Abstract stakeholder trust dynamics.

Concept inherited from ``dispatch-benchmark/engine/stakeholder_trust.py``
but rewritten without emergency-domain assumptions. Each domain defines
its own stakeholder groups (power: hospital/water/transit/industrial/
residential; traffic: emergency_services/commuters/freight/pedestrians)
and registers them with the manager.

Trust evolves in response to events:

- Positive: promise_kept, timely_response, resource_shared, info_shared,
  fair_treatment, successful_collaboration → +trust
- Negative (asymmetric, larger magnitude): promise_broken, delayed_response,
  resource_withheld, info_withheld, unfair_treatment, failed_collaboration
  → −trust
- Natural drift toward stakeholder-specific baseline each tick.

Trust then modulates downstream behaviour: information quality (e.g.,
stakeholder_query returns less detail), cooperation_modifier (e.g., mutual
aid takes longer), and final ``stakeholder_management`` dimension score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StakeholderGroup:
    """Definition of a stakeholder group, domain-specific."""

    group_id: str
    display_name: str
    baseline_trust: float = 0.6  # ``[0, 1]``
    volatility: float = 0.05  # drift speed per tick
    # how much each event class moves trust (signed deltas in [-1, 1])
    positive_delta: dict[str, float] = field(default_factory=dict)
    negative_delta: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrustReading:
    group_id: str
    trust: float
    last_event: str | None = None
    last_event_tick: int = -1

    @property
    def tier(self) -> str:
        if self.trust >= 0.75:
            return "high"
        if self.trust >= 0.50:
            return "medium"
        if self.trust >= 0.25:
            return "low"
        return "critical"


class StakeholderTrustManager:
    """Tracks trust per stakeholder group; emits TrustReading snapshots."""

    DEFAULT_POSITIVE: dict[str, float] = {
        "promise_kept": 0.08,
        "timely_response": 0.05,
        "resource_shared": 0.04,
        "info_shared": 0.03,
        "fair_treatment": 0.07,
        "successful_collaboration": 0.10,
    }

    DEFAULT_NEGATIVE: dict[str, float] = {
        "promise_broken": -0.18,
        "delayed_response": -0.10,
        "resource_withheld": -0.12,
        "info_withheld": -0.08,
        "unfair_treatment": -0.15,
        "failed_collaboration": -0.20,
    }

    def __init__(self) -> None:
        self._groups: dict[str, StakeholderGroup] = {}
        self._trust: dict[str, float] = {}
        self._last_event: dict[str, tuple[str, int]] = {}

    # ── Registration ────────────────────────────────────────────────────

    def register(self, group: StakeholderGroup) -> None:
        # merge defaults so domain only needs to override the deltas it
        # cares about
        merged_pos = {**self.DEFAULT_POSITIVE, **group.positive_delta}
        merged_neg = {**self.DEFAULT_NEGATIVE, **group.negative_delta}
        group = StakeholderGroup(
            group_id=group.group_id,
            display_name=group.display_name,
            baseline_trust=group.baseline_trust,
            volatility=group.volatility,
            positive_delta=merged_pos,
            negative_delta=merged_neg,
            metadata=dict(group.metadata),
        )
        self._groups[group.group_id] = group
        self._trust[group.group_id] = group.baseline_trust

    def reset(self) -> None:
        self._trust = {gid: g.baseline_trust for gid, g in self._groups.items()}
        self._last_event.clear()

    # ── Events ──────────────────────────────────────────────────────────

    def record_event(self, group_id: str, event: str, tick: int) -> float:
        """Apply an event to a group; return new trust value."""
        group = self._groups.get(group_id)
        if group is None:
            return 0.0
        delta = group.positive_delta.get(event, group.negative_delta.get(event, 0.0))
        new_trust = max(0.0, min(1.0, self._trust[group_id] + delta))
        self._trust[group_id] = new_trust
        self._last_event[group_id] = (event, tick)
        return new_trust

    def tick(self, current_tick: int) -> None:
        """Natural drift toward baseline."""
        for gid, group in self._groups.items():
            cur = self._trust[gid]
            target = group.baseline_trust
            self._trust[gid] = cur + (target - cur) * group.volatility

    # ── Read-out ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, TrustReading]:
        out: dict[str, TrustReading] = {}
        for gid in self._groups:
            last_evt, last_tick = self._last_event.get(gid, (None, -1))
            out[gid] = TrustReading(
                group_id=gid,
                trust=round(self._trust[gid], 3),
                last_event=last_evt,
                last_event_tick=last_tick,
            )
        return out

    def cooperation_modifier(self, group_id: str) -> float:
        """Return a multiplier for tasks involving the group.

        Mirrors dispatch-benchmark's tiers: high=1.25, medium=1.0, low=0.75,
        critical=0.5.
        """
        reading = self.snapshot().get(group_id)
        if reading is None:
            return 1.0
        return {"high": 1.25, "medium": 1.0, "low": 0.75, "critical": 0.5}.get(
            reading.tier, 1.0
        )

    def info_quality(self, group_id: str) -> float:
        reading = self.snapshot().get(group_id)
        if reading is None:
            return 1.0
        return {"high": 1.0, "medium": 0.8, "low": 0.6, "critical": 0.4}.get(
            reading.tier, 1.0
        )

    def equity_gini(self) -> float:
        """Inequality of trust across groups (0 = perfect equality)."""
        vals = sorted(self._trust.values())
        n = len(vals)
        if n <= 1:
            return 0.0
        s = sum(vals)
        if s <= 0:
            return 0.0
        cumulative = 0.0
        for i, v in enumerate(vals, start=1):
            cumulative += i * v
        return (2 * cumulative) / (n * s) - (n + 1) / n
