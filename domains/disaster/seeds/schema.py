"""
domains.disaster.seeds.schema — Disaster-native scenario seed contract.

Mirrors ``domains.power_grid.seeds.schema`` in shape, but does NOT import
from it. Per ``.hl/policy.md`` Red Line #3 the disaster domain has its own
``Perturbation`` kinds, its own stakeholder classes, and its own zone /
hospital / responder concepts. Cross-domain coupling goes through
``core.cascade_bus.CascadeEvent``, never through direct schema reuse.

Design tenets:

- Every field is JSON-safe (dataclasses, primitives, dicts of primitives).
- The seed declares WHICH RCRS map / hazard time series / kernel flags to
  spin up — but does NOT pre-compute simulator state. State is
  materialized at ``adapter.reset()``.
- ``provenance`` carries the chain back to the real data file (RCRS
  upstream + commit, OpenQuake-baked .npz UUID, USGS ShakeMap event id)
  so audits can verify nothing was hand-written.

Per the v0.3 disaster design doc §1, BSD-3-Clause runtime + AGPL-only-at-
build-time is the licence posture; the schema records both for every seed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────────────
# Perturbations (disaster-native — see design doc §5.1)
# ─────────────────────────────────────────────────────────────────────────────


DisasterPerturbationKind = Literal[
    "hazard_shake",  # primary seismic event (PGA/PGV from baked .npz)
    "building_collapse",  # triggered by hazard_shake when intensity > threshold
    "aftershock",  # follow-on seismic event (timing hidden, see fog)
    "fire_spread",  # urban-fire propagation
    "tsunami_inundation",  # NTHMP-driven flood wavefront
    "road_blockage",  # debris / damage cuts a transport edge
    "bridge_failure",  # higher-criticality road blockage analogue
    "gas_leak",  # ignites if fire reaches the same cell
    "medical_surge",  # casualty arrival rate at a hospital_id spikes
    "comms_blackout",  # zone-wide observation staleness ≥ 5 ticks
]


@dataclass
class Perturbation:
    """A deterministic disaster-domain perturbation.

    Same shape as ``power_grid.seeds.schema.Perturbation`` but typed against
    the disaster-native ``DisasterPerturbationKind`` literal. Adapter code
    never cross-types between the two domains' Perturbation classes.
    """

    kind: DisasterPerturbationKind
    trigger_tick: int
    duration_ticks: int = 1
    hidden: bool = False
    target: dict[str, Any] = field(default_factory=dict)
    intensity: float = 1.0
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Stakeholder classes (disaster-native — see design doc §5.2)
# ─────────────────────────────────────────────────────────────────────────────


DisasterStakeholderClass = Literal[
    "civilian",  # affected residents; most volatile
    "responder_ems",  # paramedics / ambulance teams
    "responder_fire",  # fire-brigade teams
    "responder_police",  # police; cordon, traffic, security
    "hospital",  # ED / ICU / surge capacity
    "media",  # press; trust drives narrative pressure
    "volunteer_org",  # CERT / Red Cross-style volunteer groups
    "local_government",  # mayor / emergency management agency
]


# ─────────────────────────────────────────────────────────────────────────────
# Zone assignment (disaster analogue of LoadAssignment)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ZoneAssignment:
    """Assign one map zone to a district + demographic profile.

    Used by:

    - The mock backend to compute per-zone population at risk and
      casualty time courses.
    - ``equitable_resource_allocation`` (v0.3 Gini-based equity scorer)
      to weight burden across districts.
    - Stakeholder-class derivation (a zone with ``has_hospital=True``
      anchors the ``hospital`` stakeholder group).
    """

    zone_id: str
    district: str
    population: int
    income_bracket: Literal["low", "mid", "high"] = "mid"
    elderly_fraction: float = 0.15
    has_hospital: bool = False
    has_school: bool = False
    criticality: float = 0.5  # 0.0 (lowest) – 1.0 (cannot be evacuated late)


# ─────────────────────────────────────────────────────────────────────────────
# Dilemmas (pre-armed; adapter fires them under predicates)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DilemmaSeed:
    """Pre-armed dilemma the adapter may surface to the agent.

    Identical shape to ``power_grid.seeds.schema.DilemmaSeed`` for
    cross-domain scoring uniformity, but defined locally so the disaster
    package never imports from power_grid.
    """

    dilemma_id: str
    trigger_tick: int
    description: str
    options: list[dict[str, Any]] = field(default_factory=list)
    expected_tradeoff_tokens: list[str] = field(default_factory=list)
    expected_stakeholder_tokens: list[str] = field(default_factory=list)
    resolution_deadline_ticks: int = 3
    default_option_id: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Provenance
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Provenance:
    """Audit trail back to the originating real-data file(s).

    For the disaster domain, ``data_source`` is one of ``"rcrs_kobe"``,
    ``"rcrs_berlin"``, ``"rcrs_vc"``, ``"rcrs_kobe_spike"`` (Phase 3.3
    procedural spike), ``"openquake_baked"``, ``"usgs_shakemap"``,
    ``"nthmp_tsunami"``, ``"eea_industrial"``, ``"hazus_tables"``,
    ``"em_dat"``. The Phase 3.3 spike uses the procedural ``rcrs_kobe_spike``
    source — see ``from_rcrs_kobe.py`` for what that means.
    """

    data_source: str
    files: list[str] = field(default_factory=list)
    commit: str | None = None
    time_window: dict[str, Any] = field(default_factory=dict)
    license: str = "see docs/v0.3_disaster_design.md §1"
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Top-level DisasterScenarioSeed
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DisasterScenarioSeed:
    """Disaster-domain analogue of ``power_grid.ScenarioSeed``.

    Distinct dataclass to honor Red Line #3 (no schema leakage). The two
    domains share the abstract concept of a ``Perturbation`` / ``DilemmaSeed``
    / ``Provenance`` (same field names) but the per-domain classes are
    independent.
    """

    seed_id: str
    family: str  # "urban_earthquake_M6_24h" | "coastal_tsunami_first_3h" | ...
    domain: str = "disaster"
    backend_kind: str = "mock_rcrs"  # "mock_rcrs" (default) | "rcrs"
    backend_config: dict[str, Any] = field(default_factory=dict)
    horizon_ticks: int = 144  # 24h @ 10min/tick by default (design doc §5.4)
    tick_minutes: int = 10
    seed: int = 42  # deterministic RNG seed for adapter + fog + tools

    zone_assignments: list[ZoneAssignment] = field(default_factory=list)
    perturbations: list[Perturbation] = field(default_factory=list)
    dilemmas: list[DilemmaSeed] = field(default_factory=list)

    # Path (relative to repo root) to a baked OpenQuake .npz hazard time
    # series, OR ``None`` for the spike's procedurally-generated series.
    hazard_time_series_ref: str | None = None

    difficulty_mode: Literal["time_pressure", "deep_planning"] = "time_pressure"
    difficulty_level: Literal["basic", "medium", "high", "extreme"] = "basic"

    provenance: Provenance = field(
        default_factory=lambda: Provenance(data_source="unspecified")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        """Stable SHA-256 over the normalized JSON body.

        Mirrors ``power_grid.ScenarioSeed.signature()`` exactly so the
        release manifest's audit gate works uniformly across domains.
        """
        body = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def complexity_metrics(self) -> dict[str, Any]:
        """Disaster-native complexity vector derived from the seed.

        Returned keys (per the v0.3 design doc §5.4):

        - ``horizon_minutes`` — wall-clock planning window.
        - ``n_perturbations`` — number of injected perturbations.
        - ``suddenness_ticks`` — minimum non-ambient trigger_tick.
        - ``observability_burden`` — count of hidden perturbations.
        - ``decision_depth`` — chained collapses + dilemmas + hidden surprises.
        - ``n_hazard_modes`` — distinct ``Perturbation.kind`` values present.
        - ``persistence_ratio`` — mean perturbation duration / horizon.
        - ``ambient_fraction`` — fraction of perturbations active at tick 0.
        """
        h_min = self.horizon_ticks * self.tick_minutes
        n_pert = len(self.perturbations)
        # Ambient = persistent throughout (e.g. comms_blackout that starts
        # at tick 0 and runs the full horizon, or a hazard_shake whose
        # ground motion decays over many ticks).
        ambient_kinds: set[str] = {"comms_blackout", "fire_spread"}
        non_ambient = [
            p
            for p in self.perturbations
            if not (p.kind in ambient_kinds and p.trigger_tick == 0)
        ]
        suddenness = min(p.trigger_tick for p in non_ambient) if non_ambient else self.horizon_ticks
        observability = sum(1 for p in self.perturbations if p.hidden)
        n_collapses = sum(
            1 for p in self.perturbations if p.kind == "building_collapse"
        )
        decision_depth = len(self.dilemmas) + max(0, n_collapses - 1) + observability
        n_hazard_modes = len({p.kind for p in self.perturbations})
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
            "n_hazard_modes": n_hazard_modes,
            "persistence_ratio": round(persistence, 4),
            "ambient_fraction": round(ambient_fraction, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def criticality_default(stakeholder_class: DisasterStakeholderClass) -> float:
    """Default per-class criticality. Used by zone-to-stakeholder mapping
    and by the equity scorer's burden weighting.

    ``hospital`` and ``responder_ems`` are highest because depriving them
    of resources directly drives realized-casualty deltas; ``media`` is
    lowest because their trust drives narrative pressure rather than
    physical harm.
    """
    return {
        "hospital": 0.95,
        "responder_ems": 0.85,
        "responder_fire": 0.80,
        "responder_police": 0.70,
        "civilian": 0.60,
        "local_government": 0.55,
        "volunteer_org": 0.45,
        "media": 0.30,
    }.get(stakeholder_class, 0.5)
