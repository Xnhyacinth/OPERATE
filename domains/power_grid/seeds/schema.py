"""
domains.power_grid.seeds.schema — Typed seed/scenario contract.

A ``ScenarioSeed`` is the canonical handoff between the real-data parsers
(``from_l2rpn``, ``from_rts_gmlc``, ``from_pglib_uc``) and the runtime
adapter (``domains.power_grid.adapter``). The seed is the unit that gets
scenario-hashed and lands in the release manifest.

Design tenets:

- Every field is JSON-safe (dataclasses, primitives, dicts of primitives).
- The seed declares WHICH real backend to spin up, WHICH time window /
  chronics to load, and WHAT perturbations to inject — but does NOT
  pre-compute simulator state. State is materialized at ``adapter.reset()``.
- ``provenance`` carries the chain back to the real data file (URL/commit/
  time window/license) so audits can verify nothing was synthesized.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────────────
# Perturbations
# ─────────────────────────────────────────────────────────────────────────────


PerturbationKind = Literal[
    "planned_maintenance",
    "line_outage",
    "load_surge",
    "generator_forced_outage",
    "fuel_supply_delay",
    "forecast_bias",
    "renewable_output_error",
    "opponent_attack",
    "wind_dropout",
    "storm_window",
]


@dataclass
class Perturbation:
    """A deterministic perturbation applied during the episode.

    All perturbations carry a ``trigger_tick`` (when they fire) and a typed
    payload. ``hidden`` means the agent does NOT see this perturbation in
    initial observations — it has to pay ``investigate_*`` to find it.
    """

    kind: PerturbationKind
    trigger_tick: int
    duration_ticks: int = 1
    hidden: bool = False
    target: dict[str, Any] = field(default_factory=dict)
    intensity: float = 1.0
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Stakeholder assignment (load classification)
# ─────────────────────────────────────────────────────────────────────────────


StakeholderClass = Literal[
    "hospital",
    "water",
    "transit",
    "industrial",
    "residential",
    "commercial",
    "data_center",
]


@dataclass
class LoadAssignment:
    """Assign one load bus to a stakeholder class with optional criticality."""

    load_id: str
    stakeholder_class: StakeholderClass
    criticality: float = 0.5  # 0.0 (lowest) – 1.0 (cannot be shed)
    bus_id: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Dilemmas (pre-armed; adapter fires them under predicates)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DilemmaSeed:
    """Pre-armed dilemma the adapter may surface to the agent."""

    dilemma_id: str
    trigger_tick: int
    description: str
    options: list[dict[str, Any]] = field(default_factory=list)
    expected_tradeoff_tokens: list[str] = field(default_factory=list)
    expected_stakeholder_tokens: list[str] = field(default_factory=list)
    resolution_deadline_ticks: int = 3
    default_option_id: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Provenance (audit chain back to real data)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Provenance:
    """Audit trail back to the originating real-data file(s).

    ``data_source`` is one of ``"rts_gmlc"``, ``"pglib_uc"``,
    ``"grid2op_l2rpn"``. ``files`` lists the actual file paths consumed
    (relative to ``benchmark/works/``). ``commit`` records the upstream
    SHA when known; ``url`` and ``lock_strategy`` make the upstream lock
    auditable at scenario level; ``license`` records the data licence to
    keep the release manifest auditable.
    """

    data_source: str
    files: list[str] = field(default_factory=list)
    commit: str | None = None
    url: str | None = None
    lock_strategy: str | None = None
    time_window: dict[str, Any] = field(default_factory=dict)
    license: str = "see docs/DATA_PROVENANCE.md"
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Top-level ScenarioSeed
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ScenarioSeed:
    seed_id: str
    family: str  # "daily_ops_24h" | "storm_emergency_6h" | ...
    domain: str = "power_grid"
    backend_kind: str = (
        "grid2op"  # "grid2op" | "pglib_uc_synthetic" | "synthetic_l2rpn14"
    )
    backend_config: dict[str, Any] = field(default_factory=dict)
    horizon_ticks: int = 96
    tick_minutes: int = 15
    seed: int = 42  # deterministic RNG seed for adapter + fog + tools

    load_assignments: list[LoadAssignment] = field(default_factory=list)
    perturbations: list[Perturbation] = field(default_factory=list)
    dilemmas: list[DilemmaSeed] = field(default_factory=list)

    difficulty_mode: Literal["time_pressure", "deep_planning"] = "time_pressure"
    # Stronger mechanics within ``extreme`` belong in
    # ``backend_config.stress_profile``; they are not public difficulty tiers.
    difficulty_level: Literal["basic", "medium", "high", "extreme"] = "basic"

    provenance: Provenance = field(
        default_factory=lambda: Provenance(data_source="unspecified")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        """Stable SHA-256 over the normalized JSON body.

        The release manifest stores this; ``audit.py`` recomputes it during
        verification and refuses to publish a scenario whose hash drifts.
        """
        body = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def complexity_metrics(self) -> dict[str, Any]:
        """Family-agnostic complexity vector derived from the seed itself.

        The taxonomy decouples "physics hardness" from the scorer's
        ``DIFFICULTY_CAL`` ceiling and gives the leaderboard / paper a
        consistent way to talk about depth, suddenness, observability and
        chronic length. See ``docs/REVIEW_v0.1_comprehensive.md`` §5.

        Returned keys:

        - ``horizon_minutes`` (H) — wall-clock planning window.
        - ``n_perturbations`` (N) — number of injected perturbations.
        - ``suddenness_ticks`` (S) — minimum non-ambient trigger_tick;
          smaller → less warning before first shock.
        - ``observability_burden`` (O) — count of hidden perturbations.
        - ``decision_depth`` (D) — chained line outages + dilemmas + hidden
          surprises that require sequential reasoning.
        - ``cascade_permissiveness`` (C) — 1 if the backend allows
          overload-driven disconnects, else 0.
        - ``persistence_ratio`` (P) — mean perturbation duration / horizon.
        - ``ambient_fraction`` — fraction of perturbations starting at
          tick 0 (background noise vs discrete shocks).
        """
        h_min = self.horizon_ticks * self.tick_minutes
        n_pert = len(self.perturbations)
        # Ambient noise (forecast_bias / storm_window starting at 0)
        ambient_kinds = {"forecast_bias", "storm_window"}
        non_ambient = [
            p
            for p in self.perturbations
            if not (p.kind in ambient_kinds and p.trigger_tick == 0)
        ]
        suddenness = min(p.trigger_tick for p in non_ambient) if non_ambient else self.horizon_ticks
        observability = sum(1 for p in self.perturbations if p.hidden)
        n_line_outages = sum(1 for p in self.perturbations if p.kind == "line_outage")
        decision_depth = (
            len(self.dilemmas)
            + max(0, n_line_outages - 1)  # chained outages add depth
            + observability  # hidden surprises require reasoning
        )
        cascade_permissive = int(
            not bool(self.backend_config.get("no_overflow_disconnection", True))
        )
        if n_pert > 0:
            persistence = sum(p.duration_ticks for p in self.perturbations) / (
                n_pert * max(1, self.horizon_ticks)
            )
            ambient_fraction = (
                sum(
                    1
                    for p in self.perturbations
                    if p.kind in ambient_kinds and p.trigger_tick == 0
                )
                / n_pert
            )
        else:
            persistence = 0.0
            ambient_fraction = 0.0
        return {
            "horizon_minutes": h_min,
            "n_perturbations": n_pert,
            "suddenness_ticks": suddenness,
            "observability_burden": observability,
            "decision_depth": decision_depth,
            "cascade_permissiveness": cascade_permissive,
            "persistence_ratio": round(persistence, 4),
            "ambient_fraction": round(ambient_fraction, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for the seed factories
# ─────────────────────────────────────────────────────────────────────────────


def default_stakeholder_distribution() -> dict[StakeholderClass, float]:
    """Default fractional mix used when a seed factory has no real metadata
    to classify loads. Picks a realistic-ish urban mix and is overridable.
    """
    return {
        "hospital": 0.05,
        "water": 0.05,
        "transit": 0.08,
        "industrial": 0.22,
        "commercial": 0.20,
        "residential": 0.35,
        "data_center": 0.05,
    }


def criticality_default(stakeholder_class: StakeholderClass) -> float:
    return {
        "hospital": 0.95,
        "water": 0.85,
        "transit": 0.80,
        "data_center": 0.70,
        "industrial": 0.50,
        "commercial": 0.30,
        "residential": 0.25,
    }.get(stakeholder_class, 0.5)
