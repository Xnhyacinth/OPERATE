"""
domains.logistics.seeds.schema — Logistics-native scenario seed contract.

Mirrors ``domains.power_grid.seeds.schema`` / ``domains.disaster.seeds.schema``
in *shape* (``signature()`` / ``to_dict()`` / ``complexity_metrics()`` are
hash-compatible) but does NOT import from them. Per ``.hl/policy.md`` Red
Line #3 the logistics domain has its own ``Perturbation`` kinds, its own
priority/stakeholder classes, and its own depot / vehicle / customer / arc
concepts. Cross-domain coupling goes through ``core.cascade_bus.CascadeEvent``.

Design tenets:

- Every field is JSON-safe (dataclasses, primitives, dicts/lists of
  primitives) so ``signature()`` is a stable SHA-256 over the seed body.
- The seed declares WHICH parsed VRP network to spin up (in
  ``backend_config['network']``), WHICH structural disruptions to inject
  (``perturbations``), and the customer priority classes
  (``load_assignments`` — kept as the field name the cross-domain
  scorer/runner expect, §7) — but does NOT pre-compute simulator state.
- ``provenance`` carries the chain back to the real instance file
  (URL / commit / license / file path) so audits verify nothing was
  synthesized.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────────────
# Perturbations (logistics-native — spec §5/§6 stressors)
# ─────────────────────────────────────────────────────────────────────────────


LogisticsPerturbationKind = Literal[
    "vehicle_breakdown",  # a vehicle is disabled (visible or discovered on arrival)
    "demand_surge",  # outstanding order demand spikes
    "urgent_order",  # high-priority order injection (last-mile dilemma/equity)
    "blocked_arc",  # an arc becomes untraversable (reroute pressure)
    "traffic_delay",  # travel-time multiplier on a region (time-window pressure)
    "machine_breakdown",  # deterministic machine outage in a dynamic job-shop
]


@dataclass
class Perturbation:
    """A deterministic logistics-domain perturbation.

    Same shape as the power-grid / disaster ``Perturbation`` but typed
    against ``LogisticsPerturbationKind``. ``hidden`` means the agent does
    NOT see it in initial observations (e.g. a breakdown discovered only on
    arrival — ``hidden_parity`` in §5).
    """

    kind: LogisticsPerturbationKind
    trigger_tick: int
    duration_ticks: int = 1
    hidden: bool = False
    target: dict[str, Any] = field(default_factory=dict)
    intensity: float = 1.0
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Customer priority class (logistics analogue of LoadAssignment)
# ─────────────────────────────────────────────────────────────────────────────


LogisticsPriorityClass = Literal[
    "medical",  # perishable-medical (highest criticality; dilemma anchor)
    "perishable",  # cold-chain / time-critical goods
    "commercial",  # standard commercial delivery
    "standard",  # general parcels
    "bulk",  # low-priority bulk freight
]


@dataclass
class CustomerPriority:
    """Assign one customer/order to a priority class with criticality.

    Field-name parity with ``power_grid.LoadAssignment`` (``load_id`` /
    ``stakeholder_class`` / ``criticality``) so the cross-domain equity
    scorer reads ``criticality`` unchanged; the logistics-native id is
    ``customer_id`` and the demand is recorded for the equity / drop
    decisions.
    """

    load_id: str  # canonical customer/order id (kept as load_id for scorer parity)
    stakeholder_class: LogisticsPriorityClass = "standard"
    criticality: float = 0.3  # 0.0 (lowest) – 1.0 (cannot be dropped)
    demand: float = 0.0
    bus_id: str | None = None  # unused; kept for shape parity


# ─────────────────────────────────────────────────────────────────────────────
# Dilemmas (pre-armed; adapter fires them under predicates)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DilemmaSeed:
    """Pre-armed dilemma the adapter may surface to the agent.

    Identical shape to the power-grid / disaster ``DilemmaSeed`` for
    cross-domain scoring uniformity, defined locally so logistics never
    imports another domain's schema.
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
    """Audit trail back to the originating real VRP instance file(s)."""

    data_source: str  # "vrplib" | "amazon_lmrrc" | ...
    files: list[str] = field(default_factory=list)
    commit: str | None = None
    url: str | None = None
    lock_strategy: str | None = None
    time_window: dict[str, Any] = field(default_factory=dict)
    license: str = "see docs/v0.7_logistics_spec.md §3/§11"
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Top-level LogisticsScenarioSeed
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class LogisticsScenarioSeed:
    """Logistics-domain analogue of ``power_grid.ScenarioSeed``.

    Distinct dataclass to honor Red Line #3 (no schema leakage). The two
    domains share the abstract concept of a ``Perturbation`` / ``DilemmaSeed``
    / ``Provenance`` (same field names) but the per-domain classes are
    independent.
    """

    seed_id: str
    family: str  # "cvrp_dispatch" | "vrptw_dispatch" | "lastmile_priority" | ...
    domain: str = "logistics"
    backend_kind: str = "pyvrp_cvrp"  # "pyvrp_cvrp" | "pyvrp_vrptw" | "pyvrp_lastmile"
    backend_config: dict[str, Any] = field(default_factory=dict)
    horizon_ticks: int = 8
    tick_minutes: int = 30  # one dispatch wave per tick
    seed: int = 42  # STRUCTURAL seed (§5) + fog/tool RNG

    # ``load_assignments`` is kept as the field name the cross-domain
    # scorer/runner expect (§7); for logistics each entry is a customer
    # priority class (not an electrical load).
    load_assignments: list[CustomerPriority] = field(default_factory=list)
    perturbations: list[Perturbation] = field(default_factory=list)
    dilemmas: list[DilemmaSeed] = field(default_factory=list)

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
        """Logistics-native complexity vector derived from the seed.

        Keeps the cross-domain shape (``horizon_minutes`` … ``ambient_fraction``)
        AND adds the four spec-§5 routing metrics pinned by the structural
        std>0 unit test:

        - ``n_stressors`` — number of injected disruptions (≈ n_perturbations).
        - ``first_disruption_tick`` — minimum non-ambient trigger_tick.
        - ``n_priority_customers`` — count of above-standard priority customers.
        - ``demand_to_capacity_ratio`` — total order demand / total fleet
          capacity (with the per-seed demand multiplier folded in by the
          builder so same-instance seeds still vary).
        """
        h_min = self.horizon_ticks * self.tick_minutes
        n_pert = len(self.perturbations)
        ambient_kinds = {"traffic_delay"}
        non_ambient = [
            p
            for p in self.perturbations
            if not (p.kind in ambient_kinds and p.trigger_tick == 0)
        ]
        if non_ambient:
            first_disruption = min(p.trigger_tick for p in non_ambient)
        else:
            first_disruption = self.horizon_ticks
        observability = sum(1 for p in self.perturbations if p.hidden)
        n_breakdowns = sum(
            1 for p in self.perturbations if p.kind == "vehicle_breakdown"
        )
        decision_depth = len(self.dilemmas) + max(0, n_breakdowns - 1) + observability
        # Routing-native quantities.
        n_priority_customers = sum(
            1
            for c in self.load_assignments
            if c.stakeholder_class in {"medical", "perishable"}
        )
        net = self.backend_config.get("network", {}) or {}
        total_demand = float(net.get("total_demand", 0.0) or 0.0)
        total_capacity = float(net.get("total_capacity", 0.0) or 0.0)
        demand_mult = float(self.backend_config.get("demand_multiplier", 1.0) or 1.0)
        demand_to_capacity = (
            (total_demand * demand_mult) / total_capacity if total_capacity > 0 else 0.0
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
        metrics = {
            "horizon_minutes": h_min,
            "n_perturbations": n_pert,
            "suddenness_ticks": first_disruption,
            "observability_burden": observability,
            "decision_depth": decision_depth,
            "cascade_permissiveness": int(
                bool(self.backend_config.get("cascade_permissive", False))
            ),
            "persistence_ratio": round(persistence, 4),
            "ambient_fraction": round(ambient_fraction, 4),
            # ── spec §5 routing metrics (std>0 across a (mode,level) bucket) ──
            "n_stressors": n_pert,
            "first_disruption_tick": first_disruption,
            "n_priority_customers": n_priority_customers,
            "demand_to_capacity_ratio": round(demand_to_capacity, 4),
        }
        job_shop = self.backend_config.get("job_shop") or {}
        if job_shop:
            jobs = int(job_shop.get("jobs", 0) or 0)
            machines = int(job_shop.get("machines", 0) or 0)
            operations = int(job_shop.get("operations", 0) or 0)
            total_processing = float(job_shop.get("total_processing_time", 0.0) or 0.0)
            metrics.update(
                {
                    "n_jobs": jobs,
                    "n_machines": machines,
                    "n_operations": operations,
                    "decision_depth": operations,
                    "machine_conflict_density": round(operations / max(1, machines), 4),
                    "processing_time_density": round(
                        total_processing / max(1, jobs * machines), 4
                    ),
                    "reference_type": (self.backend_config.get("reference") or {}).get(
                        "type"
                    ),
                }
            )
        elif self.backend_kind == "orgym_invmgmt":
            metrics.update(self._inventory_complexity_metrics())
        return metrics

    def _inventory_complexity_metrics(self) -> dict[str, Any]:
        """OR-Gym multi-period inventory decision-complexity metrics.

        Inventory replenishment carries no injected ``perturbations`` or
        ``dilemmas``, so the family-agnostic derivation collapses every
        ``decision_depth`` / ``observability_burden`` / ``persistence_ratio``
        to zero (a misleading constant ``decision_value``). The real
        decision pressure instead lives in the simulator knobs that already
        vary across release rows:

        - ``decision_depth`` — forward replenishment-commitment depth: an
          order placed now lands ``lead_time`` ticks later across
          ``stages`` echelons, so each decision must reason
          ``lead_time + (stages - 1)`` ticks ahead.
        - ``observability_burden`` — demand days hidden beyond the bounded
          forecast horizon (``periods - observation_forecast_horizon``);
          the agent cannot see realized demand past that window.
        - ``persistence_ratio`` — empirical demand density
          (nonzero-demand days / window length): sustained nonzero demand
          is stress that cannot be cleared by a single order.
        """
        env = self.backend_config.get("orgym_env_config") or {}
        lead_times = [
            int(x) for x in (self.backend_config.get("lead_times") or env.get("L") or [])
        ]
        lead_time = (
            max(lead_times)
            if lead_times
            else int(self.backend_config.get("m5_lead_time_days") or 1)
        )
        stages = int(self.backend_config.get("stages", 2) or 2)
        window_length = int(
            self.backend_config.get("m5_window_length_days")
            or env.get("periods")
            or self.horizon_ticks
        )
        forecast_horizon = int(
            self.backend_config.get("observation_forecast_horizon", 0) or 0
        )
        nonzero_days = int(self.backend_config.get("m5_nonzero_demand_days", 0) or 0)
        demand_sum = int(self.backend_config.get("m5_demand_sum_units", 0) or 0)
        capacities = [
            int(x) for x in (self.backend_config.get("capacities") or env.get("c") or [])
        ]
        capacity_units = max(capacities) if capacities else 0
        demand_density = (
            round(nonzero_days / window_length, 4) if window_length > 0 else 0.0
        )
        return {
            "decision_depth": lead_time + max(0, stages - 1),
            "observability_burden": max(0, window_length - forecast_horizon),
            "persistence_ratio": demand_density,
            "n_stages": stages,
            "lead_time_days": lead_time,
            "n_periods": window_length,
            "forecast_horizon": forecast_horizon,
            "demand_density": demand_density,
            "demand_sum_units": demand_sum,
            "capacity_units": capacity_units,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def criticality_default(priority_class: LogisticsPriorityClass) -> float:
    """Default per-class criticality (drives equity weighting + drop order)."""
    return {
        "medical": 0.95,
        "perishable": 0.75,
        "commercial": 0.45,
        "standard": 0.30,
        "bulk": 0.15,
    }.get(priority_class, 0.3)
