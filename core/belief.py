"""
core.belief — Backend-agnostic belief-state tracker.

Maintains the agent's belief over hidden world state (since observations
are partial) and computes belief accuracy against ground truth.

Forked-and-refactored from ``dispatch-benchmark/engine/belief_state.py``:
removed all emergency-specific fields (incidents, units, hospitals,
ambulance, etc.); generalized to a tagged-entity model so each domain
declares its own entity kinds (substations, generators, lines, loads for
power; intersections, vehicles for traffic; buildings, civilians for RCRS).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EntityBelief:
    """The agent's belief about a single tagged entity.

    The ``kind`` is a domain-defined tag like ``"substation"``, ``"line"``,
    ``"load"``, ``"intersection"``, ``"vehicle"``. ``believed_attrs`` holds
    whatever attributes the agent has observed/inferred, and ``confidence``
    decays for stale beliefs.
    """

    entity_id: str
    kind: str
    believed_attrs: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    last_updated_tick: int = -1
    information_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "attrs": dict(self.believed_attrs),
            "confidence": round(self.confidence, 3),
            "last_updated_tick": self.last_updated_tick,
            "information_sources": list(self.information_sources),
        }


@dataclass
class BeliefState:
    """The full belief vector at a given tick."""

    tick: int = 0
    entities: dict[str, EntityBelief] = field(default_factory=dict)
    forecasts: list[dict[str, Any]] = field(default_factory=list)
    proactive_actions_taken: list[str] = field(default_factory=list)
    global_uncertainty: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "entities": {eid: b.to_dict() for eid, b in self.entities.items()},
            "forecasts": list(self.forecasts),
            "proactive_actions_taken": list(self.proactive_actions_taken),
            "global_uncertainty": round(self.global_uncertainty, 3),
        }


class BeliefStateTracker:
    """Tracks belief over time and computes accuracy vs ground truth.

    The agent does NOT have direct access to ground truth — this class only
    receives ground truth from the env *for scoring*. The agent's view goes
    through observations.
    """

    def __init__(
        self,
        decay_rate: float = 0.02,
        observation_confidence_gain: float = 0.2,
        investigation_confidence_gain: float = 0.4,
    ):
        self._belief = BeliefState()
        self._accuracy_history: list[float] = []
        self.decay_rate = decay_rate
        self.observation_gain = observation_confidence_gain
        self.investigation_gain = investigation_confidence_gain

    @property
    def belief(self) -> BeliefState:
        return self._belief

    def reset(self) -> None:
        self._belief = BeliefState()
        self._accuracy_history.clear()

    # ── Updates ─────────────────────────────────────────────────────────

    def update_from_observation(
        self,
        observation: dict[str, Any],
        tick: int,
        entity_key: str = "entities",
        source: str = "observation",
    ) -> None:
        """Update beliefs from a domain observation.

        Expects ``observation[entity_key]`` to be ``{eid: {kind, ...attrs}}``.
        """
        self._belief.tick = tick
        gain = (
            self.observation_gain
            if source == "observation"
            else self.investigation_gain
        )

        entities = observation.get(entity_key) or {}
        if isinstance(entities, list):
            iter_items: Iterable[tuple[str, dict[str, Any]]] = (
                (e.get("id", str(i)), e)
                for i, e in enumerate(entities)
                if isinstance(e, dict)
            )
        elif isinstance(entities, dict):
            iter_items = (
                (str(eid), data)
                for eid, data in entities.items()
                if isinstance(data, dict)
            )
        else:
            iter_items = iter(())

        for eid, data in iter_items:
            kind = str(data.get("kind") or data.get("type") or "unknown")
            if eid not in self._belief.entities:
                self._belief.entities[eid] = EntityBelief(entity_id=eid, kind=kind)
            belief = self._belief.entities[eid]
            belief.kind = kind
            for k, v in data.items():
                if k in {"id", "kind", "type"}:
                    continue
                belief.believed_attrs[k] = v
            belief.confidence = min(1.0, belief.confidence + gain)
            belief.last_updated_tick = tick
            if source not in belief.information_sources:
                belief.information_sources.append(source)

        self._decay_stale_beliefs(tick)
        self._update_global_uncertainty()

    def record_forecast(self, forecast: dict[str, Any]) -> None:
        self._belief.forecasts.append(dict(forecast))

    def record_proactive_action(self, action_type: str) -> None:
        self._belief.proactive_actions_taken.append(action_type)

    # ── Internal mechanics ──────────────────────────────────────────────

    def _decay_stale_beliefs(self, current_tick: int) -> None:
        for belief in self._belief.entities.values():
            stale = max(0, current_tick - belief.last_updated_tick)
            belief.confidence = max(0.05, belief.confidence - self.decay_rate * stale)

    def _update_global_uncertainty(self) -> None:
        if not self._belief.entities:
            self._belief.global_uncertainty = 1.0
            return
        mean_conf = sum(b.confidence for b in self._belief.entities.values()) / len(
            self._belief.entities
        )
        self._belief.global_uncertainty = max(0.0, 1.0 - mean_conf)

    # ── Accuracy vs ground truth ────────────────────────────────────────

    def compute_accuracy(self, ground_truth: dict[str, Any]) -> dict[str, float]:
        """Per-kind belief accuracy versus the ground-truth entity table.

        ``ground_truth`` is expected to be ``{entity_id: {kind, ...attrs}}``.
        """
        gt_entities = ground_truth.get("entities") or {}
        if not gt_entities:
            return {"overall": 1.0, "per_kind": {}}

        by_kind: dict[str, list[float]] = {}
        for eid, gt in gt_entities.items():
            kind = str(gt.get("kind") or gt.get("type") or "unknown")
            score = self._score_entity(eid, gt)
            by_kind.setdefault(kind, []).append(score)

        per_kind = {k: round(sum(v) / len(v), 3) for k, v in by_kind.items() if v}
        overall = round(sum(per_kind.values()) / max(len(per_kind), 1), 3)
        self._accuracy_history.append(overall)
        return {"overall": overall, "per_kind": per_kind}

    def _score_entity(self, eid: str, ground_truth: dict[str, Any]) -> float:
        belief = self._belief.entities.get(eid)
        if not belief:
            return 0.0
        # Generic attribute-overlap score with confidence weighting.
        score = 0.0
        matched_attrs = 0
        total_attrs = 0
        for k, gt_v in ground_truth.items():
            if k in {"id", "kind", "type"}:
                continue
            total_attrs += 1
            bv = belief.believed_attrs.get(k)
            if bv is None:
                continue
            if isinstance(gt_v, (int, float)) and isinstance(bv, (int, float)):
                denom = max(abs(gt_v), 1.0)
                rel_err = abs(bv - gt_v) / denom
                score += max(0.0, 1.0 - rel_err)
            elif bv == gt_v:
                score += 1.0
            else:
                score += 0.1
            matched_attrs += 1
        if total_attrs == 0:
            return 1.0
        raw = score / total_attrs
        return raw * belief.confidence

    def accuracy_trend(self) -> float:
        """Late-half mean minus early-half mean. Positive = improving."""
        if len(self._accuracy_history) < 4:
            return 0.0
        half = len(self._accuracy_history) // 2
        early = sum(self._accuracy_history[:half]) / half
        late = sum(self._accuracy_history[half:]) / (len(self._accuracy_history) - half)
        return round(late - early, 3)
