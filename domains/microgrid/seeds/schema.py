"""
domains.microgrid.seeds.schema — Microgrid-native scenario seed contract.

Mirrors ``domains.power_grid.seeds.schema`` / ``domains.logistics.seeds.schema``
in *shape* (``signature()`` / ``to_dict()`` / ``complexity_metrics()`` are
hash-compatible) but does NOT import from them. Per ``.hl/policy.md`` Red
Line #3 the microgrid domain has its own ``Perturbation`` kinds, its own
EMS asset concepts (battery / genset / PCC / DER), and its own load
criticality classes (hospital / water vs residential). No power-bus
re-typing.

Design tenets:

- Every field is JSON-safe (dataclasses, primitives, dicts/lists of
  primitives) so ``signature()`` is a stable SHA-256 over the seed body.
- The seed declares WHICH backend to spin up (``backend_kind``), the baked
  weather/load/price overlays (``backend_config['profiles']``), WHICH
  structural disruptions to inject (``perturbations``) and the load
  criticality classes (``load_assignments`` — kept as the field name the
  cross-domain scorer/runner expect, §7) — but does NOT pre-compute
  simulator state.
- ``provenance`` carries the chain back to the real baked dataset (NSRDB /
  OEDI / pymgrid) so audits verify nothing was synthesized live.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# ─────────────────────────────────────────────────────────────────────────────
# Perturbations (microgrid-native — spec §4/§5 stressors)
# ─────────────────────────────────────────────────────────────────────────────


MicrogridPerturbationKind = Literal[
    "grid_outage",  # PCC trips → island (ride-through on battery+genset)
    "pv_ramp",  # large PV ramp (cloud edge / sunrise-sunset)
    "der_failure",  # a controllable DER drops out (visible or hidden)
    "load_spike",  # demand surge on the served feeder
    "price_spike",  # grid-import price spike (arbitrage pressure)
    "forecast_bias",  # forecast under/over-estimates (silent until queried)
]


@dataclass
class Perturbation:
    """A deterministic microgrid-domain perturbation.

    Same shape as the power-grid / logistics ``Perturbation`` but typed
    against ``MicrogridPerturbationKind``. ``hidden`` means the agent does
    NOT see it in initial observations (e.g. a DER failure discovered only
    via ``investigate_asset`` — ``hidden_parity`` in §5).
    """

    kind: MicrogridPerturbationKind
    trigger_tick: int
    duration_ticks: int = 1
    hidden: bool = False
    target: dict[str, Any] = field(default_factory=dict)
    intensity: float = 1.0
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Load criticality class (microgrid analogue of LoadAssignment)
# ─────────────────────────────────────────────────────────────────────────────


MicrogridLoadClass = Literal[
    "hospital",  # life-critical (highest criticality; dilemma anchor)
    "water",  # water/sanitation pumping (critical)
    "data_center",  # high-value commercial load
    "commercial",  # standard commercial
    "residential",  # residential (lowest criticality; shed first)
]


@dataclass
class MicrogridLoad:
    """Assign one feeder load to a criticality class.

    Field-name parity with ``power_grid.LoadAssignment`` (``load_id`` /
    ``stakeholder_class`` / ``criticality``) so the cross-domain equity
    scorer reads ``criticality`` unchanged. ``demand_fraction`` is this
    load's share of aggregate feeder demand.
    """

    load_id: str  # canonical load id (kept as load_id for scorer parity)
    stakeholder_class: MicrogridLoadClass = "residential"
    criticality: float = 0.25  # 0.0 (shed first) – 1.0 (never shed)
    demand_fraction: float = 0.0
    bus_id: str | None = None  # LV-family bus binding (None on EMS families)


# ─────────────────────────────────────────────────────────────────────────────
# Dilemmas (pre-armed; adapter fires them under predicates)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DilemmaSeed:
    """Pre-armed dilemma the adapter may surface to the agent.

    Identical shape to the power-grid / logistics ``DilemmaSeed`` for
    cross-domain scoring uniformity, defined locally so microgrid never
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
    """Audit trail back to the baked NSRDB / OEDI / pymgrid series."""

    data_source: str  # "nsrdb+oedi" | "pymgrid_bundled" | "nsrdb_rooftop" | ...
    files: list[str] = field(default_factory=list)
    commit: str | None = None
    url: str | None = None
    lock_strategy: str | None = None
    time_window: dict[str, Any] = field(default_factory=dict)
    license: str = "see docs/v0.7_microgrid_spec.md §3/§11"
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Top-level MicrogridScenarioSeed
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MicrogridScenarioSeed:
    """Microgrid-domain analogue of ``power_grid.ScenarioSeed``.

    Distinct dataclass to honor Red Line #3 (no schema leakage). The
    domains share the abstract concept of a ``Perturbation`` /
    ``DilemmaSeed`` / ``Provenance`` (same field names) but the per-domain
    classes are independent.
    """

    seed_id: str
    family: str  # microgrid_islanding_24h | microgrid_economic_dispatch_24h | ...
    domain: str = "microgrid"
    backend_kind: str = "pymgrid_islanding"  # see backends/
    backend_config: dict[str, Any] = field(default_factory=dict)
    horizon_ticks: int = 24
    tick_minutes: int = 60  # one supervisory tick per hour (LV uses 6h horizon)
    seed: int = 42  # STRUCTURAL seed (§5) + fog/tool RNG

    # ``load_assignments`` kept as the field name the cross-domain
    # scorer/runner expect (§7); for microgrid each entry is a feeder load
    # criticality class.
    load_assignments: list[MicrogridLoad] = field(default_factory=list)
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
        """Microgrid-native complexity vector derived from the seed.

        Keeps the cross-domain shape (``horizon_minutes`` … ``ambient_fraction``)
        AND adds the spec-§5 EMS metrics pinned by the structural std>0 unit
        test:

        - ``n_islanding_events`` — number of ``grid_outage`` disruptions.
        - ``controllable_asset_count`` — controllable DER + battery + genset.
        - ``forecast_error_sigma`` — forecast-regime noise sigma.
        - ``decision_depth`` — battery planning horizon proxy.
        - ``critical_load_fraction`` — share of demand on critical classes.
        """
        h_min = self.horizon_ticks * self.tick_minutes
        n_pert = len(self.perturbations)
        ambient_kinds = {"forecast_bias"}
        non_ambient = [
            p
            for p in self.perturbations
            if not (p.kind in ambient_kinds and p.trigger_tick == 0)
        ]
        first_disruption = (
            min(p.trigger_tick for p in non_ambient)
            if non_ambient
            else self.horizon_ticks
        )
        observability = sum(1 for p in self.perturbations if p.hidden)

        # EMS-native quantities.
        n_islanding = sum(1 for p in self.perturbations if p.kind == "grid_outage")
        bc = self.backend_config or {}
        controllable_asset_count = int(bc.get("controllable_der_count", 0)) + int(
            bool(bc.get("genset_available", False))
        )
        if bc.get("battery"):
            controllable_asset_count += 1
        forecast_error_sigma = float(bc.get("forecast_error_sigma", 0.0) or 0.0)
        critical_fraction = round(
            sum(
                la.demand_fraction
                for la in self.load_assignments
                if la.stakeholder_class in {"hospital", "water"}
            ),
            4,
        )
        # Battery planning horizon proxy: deeper in deep_planning + islanding.
        decision_depth = (
            n_islanding * 2
            + len(self.dilemmas)
            + observability
            + (2 if self.difficulty_mode == "deep_planning" else 0)
        )
        recovery_phases = list(
            (bc.get("task_contract") or {}).get("phase_ticks") or []
        )
        if recovery_phases:
            decision_depth = max(decision_depth, len(recovery_phases))

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
            "suddenness_ticks": first_disruption,
            "observability_burden": observability,
            "decision_depth": decision_depth,
            "cascade_permissiveness": int(
                bool(self.backend_config.get("cascade_permissive", False))
            ),
            "persistence_ratio": round(persistence, 4),
            "ambient_fraction": round(ambient_fraction, 4),
            # ── spec §5 EMS metrics (std>0 across a (mode,level) bucket) ──
            "n_stressors": n_pert,
            "first_disruption_tick": first_disruption,
            "n_islanding_events": n_islanding,
            "controllable_asset_count": controllable_asset_count,
            "forecast_error_sigma": round(forecast_error_sigma, 4),
            "critical_load_fraction": critical_fraction,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def criticality_default(load_class: MicrogridLoadClass) -> float:
    """Default per-class criticality (drives equity weighting + shed order)."""
    return {
        "hospital": 0.97,
        "water": 0.85,
        "data_center": 0.55,
        "commercial": 0.40,
        "residential": 0.20,
    }.get(load_class, 0.25)
