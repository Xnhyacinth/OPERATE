"""
core.fog_of_war — Policy that maps full ground-truth state to partial obs.

Two responsibilities:

1. Apply hide rules so the agent never sees fields it has not paid the
   cost to investigate.
2. Add noise/staleness to readings the agent has technically seen but is
   meant to be uncertain about (e.g., distant substations, last-tick
   forecasts beyond the forecast horizon).

PURE filter contract (added in v0.1.1):

The original v0.1 ``filter`` used a single stateful ``random.Random`` that
advanced on every call, which meant repeated ``query_grid_state`` calls
within one tick would yield DIFFERENT noised observations and would
subtly poison downstream trust calculations that read the snapshot. v0.1.1
makes ``filter`` deterministic in ``(seed, tick, entity_id, attr)`` so
repeated reads at the same tick are identical.

Forked-and-refactored from dispatch-benchmark's ad-hoc obs filtering inside
``engine/environment.py`` into a typed, deterministic policy object.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Iterable

# NIT-4: removed `import struct` — was kept under noqa for "future use"
# that never materialised; speculative imports rot silently.
from dataclasses import dataclass, field
from typing import Any

from .difficulty_levels import canonical_difficulty_level

# ── Unified Fog-of-War Strategy (v0.35) ──

FOG_LEVELS = {
    "basic": {"visibility": 1.0, "noise_std": 0.0, "delay_ticks": 0, "hidden_attrs": []},
    "medium": {"visibility": 0.7, "noise_std": 0.1, "delay_ticks": 1, "hidden_attrs": ["exact_capacity", "competitor_actions"]},
    "high": {"visibility": 0.4, "noise_std": 0.25, "delay_ticks": 2, "hidden_attrs": ["exact_capacity", "competitor_actions", "future_demand", "stakeholder_trust"]},
    "extreme": {"visibility": 0.15, "noise_std": 0.5, "delay_ticks": 3, "hidden_attrs": ["exact_capacity", "competitor_actions", "future_demand", "stakeholder_trust", "cascade_risk", "own_impact"]},
}

def get_fog_config(difficulty_level: str, domain: str | None = None) -> dict:
    """Return the unified fog-of-war configuration for a given difficulty level and domain."""
    level = canonical_difficulty_level(difficulty_level)
    base = FOG_LEVELS.get(level, FOG_LEVELS["basic"]).copy()
    # Domain-specific adjustments
    if domain == "traffic":
        base["hidden_attrs"] = base.get("hidden_attrs", []) + ["exact_vehicle_count"]
    elif domain == "microgrid":
        base["hidden_attrs"] = base.get("hidden_attrs", []) + ["exact_pv_forecast"]
    return base


@dataclass
class HideRule:
    """A rule that hides specific attributes of specific entity kinds."""

    entity_kind: str
    hidden_attrs: list[str] = field(default_factory=list)
    reveal_on: list[str] = field(default_factory=list)  # tool names that unlock


@dataclass
class NoiseRule:
    """Add Gaussian noise to numeric attributes."""

    entity_kind: str
    attr: str
    sigma_rel: float = 0.05  # relative std-dev as fraction of value
    bias: float = 0.0  # additive bias


@dataclass
class StalenessRule:
    """Mark attribute readings as `staleness_ticks` ticks old.

    Used for distance-decay observability: a remote substation's voltage
    reading might be 3 ticks stale because telemetry lags.
    """

    entity_kind: str
    attr: str
    staleness_ticks: int = 0


@dataclass
class FogOfWarPolicy:
    """Stateful policy that filters and noises an observation snapshot.

    Deterministic given a seed; safe to call inside counterfactual replay
    because all randomness flows through the constructor-provided RNG.
    """

    hide_rules: list[HideRule] = field(default_factory=list)
    noise_rules: list[NoiseRule] = field(default_factory=list)
    staleness_rules: list[StalenessRule] = field(default_factory=list)
    seed: int = 0
    current_tick: int = 0

    def __post_init__(self) -> None:
        # entities for which the agent has paid investigation cost this episode
        self._revealed: dict[str, set[str]] = {}

    def reset(self, seed: int) -> None:
        self.seed = seed
        self.current_tick = 0
        self._revealed.clear()

    def set_tick(self, tick: int) -> None:
        """Adapter calls this once per tick BEFORE any snapshot is filtered,
        so deterministic per-(entity, tick) noise can be reproduced exactly
        by repeated read-only queries within the same tick."""
        self.current_tick = int(tick)

    def mark_revealed(self, entity_id: str, attrs: Iterable[str] | None = None) -> None:
        """Record that the agent paid to investigate ``entity_id``.

        Subsequent observations of that entity's attrs are no longer hidden
        (but may still be noised). ``attrs=None`` reveals every hidden attr.
        """
        bucket = self._revealed.setdefault(entity_id, set())
        if attrs is None:
            bucket.add("*")
        else:
            bucket.update(attrs)

    def filter(
        self,
        observation: dict[str, Any],
        entity_key: str = "entities",
    ) -> dict[str, Any]:
        """Return a sanitized copy of the observation.

        Pure in ``(seed, current_tick, observation)``: calling this multiple
        times within the same tick yields byte-identical results, so a
        read-only ``query_grid_state`` cannot perturb downstream state.
        """
        out = copy.deepcopy(observation)
        entities = out.get(entity_key)
        if not isinstance(entities, dict):
            return out

        for eid, ent in entities.items():
            if not isinstance(ent, dict):
                continue
            kind = str(ent.get("kind") or ent.get("type") or "unknown")
            self._apply_hide_rules(eid, kind, ent)
            self._apply_noise_rules(eid, kind, ent)
            self._apply_staleness_rules(kind, ent)
        return out

    # ── Rule application ────────────────────────────────────────────────

    def _apply_hide_rules(self, eid: str, kind: str, ent: dict[str, Any]) -> None:
        revealed = self._revealed.get(eid, set())
        for rule in self.hide_rules:
            if rule.entity_kind not in {kind, "*"}:
                continue
            for attr in rule.hidden_attrs:
                if "*" in revealed or attr in revealed:
                    continue
                if attr in ent:
                    ent[attr] = None
                    ent.setdefault("_hidden_attrs", []).append(attr)

    def _apply_noise_rules(self, eid: str, kind: str, ent: dict[str, Any]) -> None:
        for rule in self.noise_rules:
            if rule.entity_kind not in {kind, "*"}:
                continue
            val = ent.get(rule.attr)
            if val is None or not isinstance(val, (int, float)) or math.isnan(val):
                continue
            sigma = abs(val) * rule.sigma_rel
            # Deterministic noise: hash (seed, tick, entity_id, attr) → N(0,1)
            noise = (
                _deterministic_gauss(self.seed, self.current_tick, eid, rule.attr)
                * sigma
                + rule.bias
            )
            ent[rule.attr] = val + noise
            ent.setdefault("_noisy_attrs", []).append(rule.attr)

    def _apply_staleness_rules(self, kind: str, ent: dict[str, Any]) -> None:
        for rule in self.staleness_rules:
            if rule.entity_kind not in {kind, "*"}:
                continue
            if rule.attr in ent and rule.staleness_ticks > 0:
                ent.setdefault("_stale_attrs", {})[rule.attr] = rule.staleness_ticks


def _deterministic_gauss(seed: int, tick: int, entity_id: str, attr: str) -> float:
    """Box-Muller transform over a SHA-256 hash of (seed, tick, eid, attr).

    Produces a standard-normal sample that is byte-identical for the same
    inputs. Used by ``FogOfWarPolicy._apply_noise_rules`` to keep the
    filter pure.
    """
    key = f"{seed}|{tick}|{entity_id}|{attr}".encode()
    h = hashlib.sha256(key).digest()
    u1_int = int.from_bytes(h[:8], "big")
    u2_int = int.from_bytes(h[8:16], "big")
    # Avoid u1 == 0 (would blow up log)
    denom = float(1 << 64)
    u1 = max((u1_int + 1) / (denom + 1), 1e-12)
    u2 = (u2_int + 1) / (denom + 1)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
