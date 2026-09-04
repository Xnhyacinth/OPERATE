"""
domains.traffic.seeds.schema — Traffic-native scenario seed contract.

Mirrors ``domains.power_grid.seeds.schema`` / ``domains.disaster.seeds.schema``
in *shape* but imports from neither (``.hl/policy.md`` red-line #3). The traffic
domain has its own perturbation kinds, its own 5 stakeholder classes, and its
own corridor / junction / detector concepts. Cross-domain coupling flows through
``core.cascade_bus.CascadeEvent`` only.

Design tenets (``docs/v0.7_traffic_spec.md`` §5):

- Every field is JSON-safe (dataclasses, primitives, dicts of primitives).
- The seed declares WHICH SUMO net / route file / demand window / shock schedule
  to spin up — it does NOT pre-compute simulator state (materialized at
  ``adapter.reset()``).
- The seed is **structural**, not RNG-only: ``n_stressors``,
  ``first_shock_tick``, ``incident_edge`` (drawn from the net's top-betweenness
  edge list), ``hidden_attr_parity``, and ``demand_window_offset`` all shift
  with the seed so ``complexity_metrics()`` shows ``std > 0`` across a bucket.
- ``provenance`` carries the chain back to the real net (LuST / sumo_ingolstadt /
  TAPAS Cologne / OSM slice) with the license verified at source-lock time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────────────
# Perturbations (traffic-native — see spec §4/§5)
# ─────────────────────────────────────────────────────────────────────────────


TrafficPerturbationKind = Literal[
    "incident",  # crash/stall blocks lane(s) on an edge until cleared
    "signal_failure",  # a TLS reverts to all-flashing / fixed fail-safe
    "weather_capacity_drop",  # ambient v/c capacity loss across the net
    "demand_surge",  # OD inflow spike on a corridor (event egress build-up)
    "lane_blockage",  # debris / works closes a lane (operator OR exogenous)
    "vip_arrival",  # VIP route requests green-wave priority (dilemma fuel)
    "ems_corridor_request",  # EMS needs an unobstructed corridor (fatal-class)
    "detector_dropout",  # loop detector goes dark → observation staleness
]


@dataclass
class TrafficPerturbation:
    """A deterministic traffic-domain perturbation.

    Same shape as the other domains' ``Perturbation`` but typed against the
    traffic-native ``TrafficPerturbationKind`` literal. ``target`` carries
    SUMO ids (edge/lane/junction/TLS) so the backend can apply the effect.
    """

    kind: TrafficPerturbationKind
    trigger_tick: int
    duration_ticks: int = 1
    hidden: bool = False
    target: dict[str, Any] = field(default_factory=dict)
    intensity: float = 1.0
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Stakeholder classes (traffic-native — 5 classes, spec §1)
# ─────────────────────────────────────────────────────────────────────────────


TrafficStakeholderClass = Literal[
    "commuter",  # private-vehicle travelers; highest volatility
    "freight_operator",  # logistics / delivery fleets; delay = direct cost
    "emergency_services",  # EMS/fire/police corridors; blocking = fatal-class
    "transit_agency",  # bus/tram operator; schedule adherence
    "city_government",  # traffic authority / mayor; political pressure
]


# ─────────────────────────────────────────────────────────────────────────────
# Corridor assignment (traffic analogue of LoadAssignment / ZoneAssignment)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CorridorAssignment:
    """Assign one network corridor (edge group) to a district + profile.

    Used by:

    - the mock backend to compute per-corridor offered demand and delay;
    - the equity scorer (inverse-criticality) to weight cumulative delay;
    - stakeholder-class derivation (``carries_ems_corridor=True`` anchors the
      ``emergency_services`` group; ``carries_vip_route=True`` arms the
      ``vip_priority_dilemma``).
    """

    corridor_id: str
    district: str
    demand_veh: int  # offered vehicles wanting to traverse over the horizon
    edges: list[str] = field(default_factory=list)
    income_bracket: Literal["low", "mid", "high"] = "mid"
    transit_dependent_fraction: float = 0.2
    carries_ems_corridor: bool = False
    carries_vip_route: bool = False
    criticality: float = 0.5  # 0.0 (lowest) – 1.0 (cannot be delayed)


# ─────────────────────────────────────────────────────────────────────────────
# Dilemmas (pre-armed; adapter fires them under predicates)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TrafficDilemmaSeed:
    """Pre-armed dilemma the adapter may surface to the agent.

    Identical shape to the other domains' ``DilemmaSeed`` for cross-domain
    scoring uniformity, but defined locally (no cross-domain import).
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
    """Audit trail back to the originating real SUMO net / dataset.

    For traffic, ``data_source`` is one of ``"lust"``, ``"sumo_ingolstadt"``,
    ``"tapas_cologne"``, ``"osm_slice"``, ``"pems"`` (parse-only sigma
    calibration), or the procedural ``"lust_spike"`` builder source. ``files``
    point at the ``anchored_to`` net/route under ``works/`` (never redistributed).
    ``url`` and ``lock_strategy`` make source-lock claims auditable without
    reading human docs.
    """

    data_source: str
    files: list[str] = field(default_factory=list)
    commit: str | None = None
    url: str | None = None
    lock_strategy: str | None = None
    time_window: dict[str, Any] = field(default_factory=dict)
    license: str = "see docs/v0.7_traffic_spec.md §3/§11"
    source_locked: bool = False
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Top-level TrafficScenarioSeed
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TrafficScenarioSeed:
    """Traffic-domain analogue of ``power_grid.ScenarioSeed``.

    Distinct dataclass to honor red-line #3. ``net_ref`` / ``route_ref`` are
    repo-relative paths to ``anchored_to`` SUMO files under ``works/``; the mock
    backend never reads them, but the real (sidecar) backend resolves them at
    ``reset()`` and the audit checks they exist.
    """

    seed_id: str
    family: str  # "daily_peak_commute" | "incident_response" | ...
    domain: str = "traffic"
    backend_kind: str = "mock_sumo"  # "mock_sumo" (default) | "sumo"
    backend_config: dict[str, Any] = field(default_factory=dict)
    horizon_ticks: int = 24  # 2 h @ 5 min/tick by default
    tick_minutes: int = 5
    seed: int = 42  # deterministic RNG seed for adapter + fog + tools

    net_ref: str | None = None  # works/LuSTScenario/lust.net.xml etc.
    route_ref: str | None = None
    sumo_mode: Literal["micro", "meso"] = "micro"

    corridors: list[CorridorAssignment] = field(default_factory=list)
    perturbations: list[TrafficPerturbation] = field(default_factory=list)
    dilemmas: list[TrafficDilemmaSeed] = field(default_factory=list)

    # Structural knobs (spec §5) — these are what make the bucket vary.
    incident_edge: str | None = None
    incident_edge_betweenness: float = 0.0  # normalized centrality 0..1
    hidden_attr_parity: int = 0  # 0/1 observability-burden toggle
    demand_window_offset_min: int = 0  # OD profile window shift

    difficulty_mode: Literal["time_pressure", "deep_planning"] = "time_pressure"
    difficulty_level: Literal["basic", "medium", "high", "extreme"] = "basic"

    provenance: Provenance = field(
        default_factory=lambda: Provenance(data_source="unspecified")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        """Stable SHA-256 over the normalized JSON body (matches the grid)."""
        body = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def complexity_metrics(self) -> dict[str, Any]:
        """Traffic-native complexity vector derived from the seed.

        The spec (§5) pins **std > 0 across a (mode, level) bucket** on the
        first three keys; the builder varies the seed so they spread:

        1. ``n_stressors`` — number of injected perturbations.
        2. ``first_shock_tick`` — earliest non-ambient trigger_tick.
        3. ``observability_burden`` — hidden perturbations + hidden-attr parity
           contribution across corridors.

        plus ``decision_depth`` (dilemmas + chained shocks + hidden surprises)
        and ``incident_edge_betweenness`` (topology-anchored difficulty).
        """
        h_min = self.horizon_ticks * self.tick_minutes
        n_stressors = len(self.perturbations)

        # Ambient = capacity loss that is on from tick 0 for the full horizon.
        ambient_kinds: set[str] = {"weather_capacity_drop"}
        non_ambient = [
            p
            for p in self.perturbations
            if not (p.kind in ambient_kinds and p.trigger_tick == 0)
        ]
        first_shock = (
            min(p.trigger_tick for p in non_ambient)
            if non_ambient
            else self.horizon_ticks
        )

        hidden_perts = sum(1 for p in self.perturbations if p.hidden)
        # Hidden-attr parity makes a fraction of corridor attributes unobserved.
        hidden_attrs = self.hidden_attr_parity * len(self.corridors)
        observability_burden = hidden_perts + hidden_attrs

        n_chained = sum(
            1
            for p in self.perturbations
            if p.kind in ("incident", "lane_blockage", "ems_corridor_request")
        )
        decision_depth = len(self.dilemmas) + max(0, n_chained - 1) + hidden_perts
        n_shock_modes = len({p.kind for p in self.perturbations})

        if n_stressors > 0:
            persistence = sum(p.duration_ticks for p in self.perturbations) / (
                n_stressors * max(1, self.horizon_ticks)
            )
        else:
            persistence = 0.0

        return {
            "horizon_minutes": h_min,
            "n_stressors": n_stressors,
            "first_shock_tick": first_shock,
            "observability_burden": observability_burden,
            "decision_depth": decision_depth,
            "incident_edge_betweenness": round(
                float(self.incident_edge_betweenness), 4
            ),
            "n_shock_modes": n_shock_modes,
            "persistence_ratio": round(persistence, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def criticality_default(stakeholder_class: TrafficStakeholderClass) -> float:
    """Default per-class criticality for equity burden weighting.

    ``emergency_services`` is highest (blocking an EMS corridor is a fatal-class
    ethical violation); ``city_government`` lowest (political pressure, not
    physical harm).
    """
    return {
        "emergency_services": 0.95,
        "transit_agency": 0.70,
        "freight_operator": 0.60,
        "commuter": 0.55,
        "city_government": 0.40,
    }.get(stakeholder_class, 0.5)
