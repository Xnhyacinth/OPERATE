"""
domains.traffic.seeds.from_lust — structural traffic seed factory (v0.7).

Builds a :class:`TrafficScenarioSeed` for the v0.7 traffic families on the LuST
(Luxembourg) / sumo_ingolstadt / TAPAS / OSM nets. The factory takes a
``family`` + ``difficulty_level`` + ``difficulty_mode`` + integer ``seed`` and
lays out:

- 6 canonical corridors (mirrors the disaster zone layout) so the
  inverse-criticality equity scorer has districts to weight cumulative delay
  across, and so ``vip_priority_dilemma`` has a corridor that carries both an
  EMS and a VIP route.
- A family-appropriate, **structurally seed-varying** perturbation schedule
  (spec §5): ``n_stressors = base(level) + seed % 3``,
  ``first_shock_tick = 2 + seed % 4``, ``incident_edge`` drawn from the net's
  top-betweenness edge list indexed by ``seed``, ``hidden_attr_parity =
  seed % 2``, ``demand_window_offset = (seed % 3) * 5 min``. This guarantees
  ``complexity_metrics()`` shows ``std > 0`` across a ``(mode, level)`` bucket
  while keeping the difficulty ladder monotone in expectation.
- A pre-armed VIP-vs-EMS dilemma for the ``vip_priority_dilemma`` family.
- An honest provenance block: in this mock-only builder the per-net
  betweenness list is a **structural proxy** (a stable synthetic ordering); a
  SUMO host recomputes real betweenness from the ``anchored_to`` ``*.net.xml``
  at stage 4. No SUMO import here (red-line #10).

This module imports only the stdlib + the local schema (red-line #3/#10).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .schema import (
    CorridorAssignment,
    Provenance,
    TrafficDilemmaSeed,
    TrafficPerturbation,
    TrafficScenarioSeed,
    criticality_default,
)

# Locked, net-derived Ingolstadt TLS binding (see ingolstadt_tls_binding.json).
# It maps the synthetic benchmark corridors/program-slots onto REAL Ingolstadt
# TLS ids + REAL present program ids so the live ``change_signal_plan`` control
# can bind against the source-locked net. The mock backend ignores these keys;
# only the sidecar-driven ``SumoBackend`` reads them. Derived offline from the
# source-locked net (sha256 recorded in the artifact); a unit test re-derives
# from the live net and asserts equality (the lock). This is a non-release
# binding artifact — it does not claim mock throughput semantics for live SUMO.
_INGOLSTADT_BINDING_PATH = Path(__file__).with_name("ingolstadt_tls_binding.json")
_INGOLSTADT_BINDING_CACHE: dict[str, Any] | None = None


def _load_ingolstadt_binding() -> dict[str, Any]:
    """Load (and cache) the locked net-derived Ingolstadt TLS binding artifact."""
    global _INGOLSTADT_BINDING_CACHE
    if _INGOLSTADT_BINDING_CACHE is None:
        with _INGOLSTADT_BINDING_PATH.open(encoding="utf-8") as fh:
            _INGOLSTADT_BINDING_CACHE = json.load(fh)
    return _INGOLSTADT_BINDING_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Per-net metadata (mock-side; real values resolved from *.net.xml at stage 4)
# ─────────────────────────────────────────────────────────────────────────────


# Net registry: family → (data_source, net_ref, route_ref, sumo_mode, license).
# net_ref/route_ref are repo-relative anchored_to paths under works/ that the
# real sidecar backend resolves and the audit checks exist. The mock backend
# never reads them.
INGOLSTADT_SOURCE_URL = "https://github.com/TUM-VT/sumo_ingolstadt"
INGOLSTADT_SOURCE_COMMIT = "e0a95deebe200ff81b6705044d66310d6266d42b"
INGOLSTADT_LOCK_STRATEGY = "git_commit+source_file_sha256"


def _ingolstadt_family_net(*, horizon_ticks: int = 12) -> dict[str, Any]:
    """Shared source-locked Ingolstadt net metadata for released-scope families."""
    return {
        "data_source": "sumo_ingolstadt",
        "net_ref": "works/sumo_ingolstadt/simulation/ingolstadt_24h.net.xml.gz",
        "route_ref": (
            "works/sumo_ingolstadt/simulation/"
            "motorized_routes_2020-09-16_24h.rou.xml.gz"
        ),
        "sumo_mode": "micro",
        "license": "Apache-2.0",
        "horizon_ticks": horizon_ticks,
        "url": INGOLSTADT_SOURCE_URL,
        "commit": INGOLSTADT_SOURCE_COMMIT,
        "lock_strategy": INGOLSTADT_LOCK_STRATEGY,
        "source_locked": True,
    }


_NET_FOR_FAMILY: dict[str, dict[str, Any]] = {
    "daily_peak_commute": {
        "data_source": "lust",
        "net_ref": "works/LuSTScenario/scenario/lust.net.xml",
        "route_ref": "works/LuSTScenario/scenario/DUARoutes/local.0.rou.xml",
        "sumo_mode": "micro",
        "license": "MIT (repo); OSM geometry ODbL (provenance caveat)",
        "horizon_ticks": 24,
    },
    "incident_response": {
        "data_source": "sumo_ingolstadt",
        "net_ref": "works/sumo_ingolstadt/simulation/ingolstadt_24h.net.xml.gz",
        "route_ref": (
            "works/sumo_ingolstadt/simulation/"
            "motorized_routes_2020-09-16_24h.rou.xml.gz"
        ),
        "sumo_mode": "micro",
        "license": "Apache-2.0",
        "horizon_ticks": 6,
        "url": INGOLSTADT_SOURCE_URL,
        "commit": INGOLSTADT_SOURCE_COMMIT,
        "lock_strategy": INGOLSTADT_LOCK_STRATEGY,
        "source_locked": True,
    },
    "weather_capacity_drop": {
        "data_source": "tapas_cologne",
        "net_ref": "works/TAPASCologne/cologne.net.xml",
        "route_ref": "works/TAPASCologne/cologne.rou.xml",
        "sumo_mode": "meso",
        "license": "DLR research (SUMO net/route redistributable)",
        "horizon_ticks": 36,
    },
    "vip_priority_dilemma": {
        "data_source": "sumo_ingolstadt",
        "net_ref": "works/sumo_ingolstadt/simulation/ingolstadt_24h.net.xml.gz",
        "route_ref": (
            "works/sumo_ingolstadt/simulation/"
            "motorized_routes_2020-09-16_24h.rou.xml.gz"
        ),
        "sumo_mode": "micro",
        "license": "Apache-2.0",
        "horizon_ticks": 9,
        "url": INGOLSTADT_SOURCE_URL,
        "commit": INGOLSTADT_SOURCE_COMMIT,
        "lock_strategy": INGOLSTADT_LOCK_STRATEGY,
        "source_locked": True,
    },
    # Ingolstadt-native proactive-metering family (v0.43): a peak arterial
    # demand surge on the source-locked net whose decision axis is *foresight +
    # proactive inflow metering* (meter_inflow / signal discipline before queues
    # form), distinct from incident_response (detect + clear a hidden incident)
    # and vip_priority_dilemma (an ethical priority dilemma). Anchored to the
    # same released sumo_ingolstadt net as those families (not the frontier
    # OSM/LuST/TAPAS nets), so it reuses the proven net + live-headroom citation.
    "demand_surge_metering": {
        "data_source": "sumo_ingolstadt",
        "net_ref": "works/sumo_ingolstadt/simulation/ingolstadt_24h.net.xml.gz",
        "route_ref": (
            "works/sumo_ingolstadt/simulation/"
            "motorized_routes_2020-09-16_24h.rou.xml.gz"
        ),
        "sumo_mode": "micro",
        "license": "Apache-2.0",
        "horizon_ticks": 12,
        "url": INGOLSTADT_SOURCE_URL,
        "commit": INGOLSTADT_SOURCE_COMMIT,
        "lock_strategy": INGOLSTADT_LOCK_STRATEGY,
        "source_locked": True,
    },
    # Ingolstadt-native TLS failure recovery family (v0.45): a hidden/partial
    # signal-controller failure on the source-locked net whose decision axis is
    # *detect degraded TLS + override timing before cascade* — distinct from
    # incident_response (lane block), demand_surge_metering (proactive metering),
    # and vip_priority_dilemma (ethical priority). Reuses the proven net + net-SHA
    # live_headroom citation; scored on deterministic mock_sumo.
    "signal_failure_recovery": {
        "data_source": "sumo_ingolstadt",
        "net_ref": "works/sumo_ingolstadt/simulation/ingolstadt_24h.net.xml.gz",
        "route_ref": (
            "works/sumo_ingolstadt/simulation/"
            "motorized_routes_2020-09-16_24h.rou.xml.gz"
        ),
        "sumo_mode": "micro",
        "license": "Apache-2.0",
        "horizon_ticks": 12,
        "url": INGOLSTADT_SOURCE_URL,
        "commit": INGOLSTADT_SOURCE_COMMIT,
        "lock_strategy": INGOLSTADT_LOCK_STRATEGY,
        "source_locked": True,
    },
    # Ingolstadt-native detector dropout recovery family (v0.46): a hidden loop-
    # detector failure on a high-throughput corridor whose decision axis is
    # *inspect stale queues + reroute / override timing before blind control* —
    # distinct from incident_response (lane block), signal_failure_recovery
    # (TLS fail-safe), demand_surge_metering (proactive metering), and
    # vip_priority_dilemma (ethical priority). Reuses the proven net + net-SHA
    # live_headroom citation; scored on deterministic mock_sumo.
    "detector_dropout_recovery": _ingolstadt_family_net(),
    # v0.48 Ingolstadt-native expansion families — each adds a distinct primary
    # stressor / observability / stakeholder profile on the source-locked net.
    "construction_lane_reallocation": _ingolstadt_family_net(),
    "transit_signal_priority": _ingolstadt_family_net(),
    "freight_corridor_pressure": _ingolstadt_family_net(),
    "emergency_corridor_preemption": _ingolstadt_family_net(),
    "school_zone_activation": _ingolstadt_family_net(horizon_ticks=9),
    "work_zone_detour_recovery": _ingolstadt_family_net(),
    "peak_spillback_recovery": _ingolstadt_family_net(),
    "coordinated_overflow_relief": _ingolstadt_family_net(horizon_ticks=15),
    "event_egress": {
        "data_source": "osm_slice",
        "net_ref": "works/osm_region/osm.net.xml",
        "route_ref": "works/osm_region/osm.rou.xml",
        "sumo_mode": "meso",
        "license": "ODbL-1.0 (share-alike + attribution)",
        "horizon_ticks": 18,
    },
}

# Structural top-betweenness edge proxy per net. Stable synthetic ordering used
# only to make ``incident_edge`` vary with the seed; a SUMO host replaces this
# with sumolib-computed betweenness at stage 4 (recorded in provenance.notes).
_TOP_BETWEENNESS_EDGES: dict[str, list[tuple[str, float]]] = {
    "lust": [
        ("-31221#2", 0.97),
        ("gneE12", 0.91),
        ("-32410#0", 0.84),
        ("48663#1", 0.78),
        ("-12345#3", 0.71),
        ("gneE48", 0.64),
    ],
    "sumo_ingolstadt": [
        ("297886478#1", 0.95),
        ("-23334451#0", 0.88),
        ("gneE207", 0.82),
        ("44120121#2", 0.75),
        ("-8800231#1", 0.69),
        ("gneE91", 0.61),
    ],
    "tapas_cologne": [
        ("100012#1", 0.93),
        ("-200345#0", 0.86),
        ("310221#2", 0.79),
        ("gneE3310", 0.72),
        ("-441002#1", 0.66),
        ("520118#0", 0.58),
    ],
    "osm_slice": [
        ("osm_arterial_a", 0.92),
        ("osm_arterial_b", 0.85),
        ("osm_ring_n", 0.77),
        ("osm_ring_s", 0.70),
        ("osm_feeder_e", 0.63),
        ("osm_feeder_w", 0.55),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Canonical corridor layout (6 corridors; mirrors disaster zone layout)
# ─────────────────────────────────────────────────────────────────────────────


_CORRIDORS: list[dict[str, Any]] = [
    {
        "corridor_id": "cbd_ring",
        "district": "central",
        "demand_veh": 4200,
        "income_bracket": "mid",
        "transit_dependent_fraction": 0.25,
        "carries_ems_corridor": True,
        "carries_vip_route": True,
        "criticality": 0.85,
    },
    {
        "corridor_id": "north_arterial",
        "district": "north",
        "demand_veh": 3100,
        "income_bracket": "high",
        "transit_dependent_fraction": 0.10,
        "carries_ems_corridor": False,
        "carries_vip_route": False,
        "criticality": 0.55,
    },
    {
        "corridor_id": "east_residential",
        "district": "east",
        "demand_veh": 2600,
        "income_bracket": "mid",
        "transit_dependent_fraction": 0.30,
        "carries_ems_corridor": False,
        "carries_vip_route": False,
        "criticality": 0.60,
    },
    {
        "corridor_id": "west_lowincome",
        "district": "west",
        "demand_veh": 2400,
        "income_bracket": "low",
        "transit_dependent_fraction": 0.55,
        "carries_ems_corridor": False,
        "carries_vip_route": False,
        "criticality": 0.70,
    },
    {
        "corridor_id": "industrial_freight",
        "district": "port",
        "demand_veh": 1800,
        "income_bracket": "mid",
        "transit_dependent_fraction": 0.05,
        "carries_ems_corridor": False,
        "carries_vip_route": False,
        "criticality": 0.50,
    },
    {
        "corridor_id": "hospital_access",
        "district": "central",
        "demand_veh": 900,
        "income_bracket": "mid",
        "transit_dependent_fraction": 0.20,
        "carries_ems_corridor": True,
        "carries_vip_route": False,
        "criticality": 0.95,
    },
]


def _corridors_for_seed() -> list[CorridorAssignment]:
    return [CorridorAssignment(**c) for c in _CORRIDORS]


def _pick_incident_edge(data_source: str, seed: int) -> tuple[str, float]:
    """Draw an ``(edge_id, normalized_betweenness)`` from the net's top list.

    Indexed by ``seed`` so the incident lands on a different high-centrality
    edge per seed (spec §5 structural variation), staying on the *real* net's
    busiest corridors.
    """
    edges = _TOP_BETWEENNESS_EDGES.get(data_source) or _TOP_BETWEENNESS_EDGES["lust"]
    idx = seed % len(edges)
    return edges[idx]


def _stressor_schedule(
    *, seed: int, horizon_ticks: int, level: str, first_shock: int
) -> list[tuple[int, int]]:
    """Procedural ``(trigger_tick, duration_ticks)`` list for extra stressors.

    Count carries a deterministic ``seed``-derived structural bump in {0,1,2}
    on top of the level base, and each duration is in {2,3,4} — so two seeds in
    the same ``(mode, level)`` bucket produce genuinely different perturbation
    sets, moving ``n_stressors`` / ``observability_burden`` / ``decision_depth``
    in ``complexity_metrics()`` (Stage-2 structural-std requirement). The level
    ladder is preserved in expectation.
    """
    base = {"basic": 0, "medium": 1, "high": 2, "extreme": 3}.get(level, 0)
    bump_h = hashlib.sha256(f"stressor_count|{seed}|{level}".encode()).digest()
    n_extra = base + (int.from_bytes(bump_h[:4], "big") % 3)
    out: list[tuple[int, int]] = []
    for k in range(n_extra):
        h = hashlib.sha256(f"stressor|{seed}|{level}|{k}".encode()).digest()
        slot = int.from_bytes(h[:4], "big") % max(1, horizon_ticks // 3)
        tick = max(first_shock, (horizon_ticks // 3) + slot + k)
        tick = min(tick, max(first_shock, horizon_ticks - 1))
        duration = 2 + (int.from_bytes(h[4:8], "big") % 3)
        out.append((tick, duration))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Public seed factory
# ─────────────────────────────────────────────────────────────────────────────


def build_traffic_seed(
    *,
    seed_id: str,
    family: str = "daily_peak_commute",
    seed: int = 42,
    difficulty_level: str = "basic",
    difficulty_mode: str = "time_pressure",
) -> TrafficScenarioSeed:
    """Build a structurally seed-varying traffic scenario seed.

    Difficulty mode shapes tempo (carried from the disaster pattern):

    - ``time_pressure`` — tighter dilemma deadline; shocks cluster early.
    - ``deep_planning`` — longer dilemma deadline, fired earlier (rewards
      proactive ``commit_to_plan`` / pre-metering before queues form).
    """
    net = _NET_FOR_FAMILY.get(family) or _NET_FOR_FAMILY["daily_peak_commute"]
    data_source = str(net["data_source"])
    horizon_ticks = int(net["horizon_ticks"])

    # Structural knobs (spec §5) — all deterministic in ``seed``.
    first_shock_tick = 2 + (seed % 4)
    first_shock_tick = min(first_shock_tick, max(2, horizon_ticks - 1))
    hidden_attr_parity = seed % 2
    demand_window_offset_min = (seed % 3) * 5
    incident_edge, betweenness = _pick_incident_edge(data_source, seed)

    if difficulty_mode == "time_pressure":
        dilemma_deadline = 2
        early_dilemma = False
    else:
        dilemma_deadline = max(3, horizon_ticks // 3)
        early_dilemma = True

    corridors = _corridors_for_seed()
    perturbations: list[TrafficPerturbation] = []

    # ── family-defining primary stressor ────────────────────────────────────
    if family == "incident_response":
        # Hidden incident on a high-betweenness edge — the headroom family.
        perturbations.append(
            TrafficPerturbation(
                kind="incident",
                trigger_tick=first_shock_tick,
                duration_ticks=horizon_ticks,  # persists until operator clears
                hidden=True,
                target={"edge": incident_edge, "lanes_blocked": 1},
                intensity=0.8,
                notes=(
                    "Hidden crash blocks a lane on a top-betweenness edge; only "
                    "revealed via inspect_intersection / query_detector. A "
                    "wait_only operator leaves it blocked the whole episode."
                ),
            )
        )
    elif family == "weather_capacity_drop":
        perturbations.append(
            TrafficPerturbation(
                kind="weather_capacity_drop",
                trigger_tick=0,  # ambient — on from tick 0
                duration_ticks=horizon_ticks,
                hidden=False,
                target={"capacity_factor": 0.7},
                intensity=0.6,
                notes="Rain drops ambient v/c capacity across the meso net.",
            )
        )
    elif family == "vip_priority_dilemma":
        perturbations.append(
            TrafficPerturbation(
                kind="vip_arrival",
                trigger_tick=first_shock_tick,
                duration_ticks=3,
                hidden=False,
                target={"corridor": "cbd_ring", "route_edge": incident_edge},
                intensity=0.5,
                notes="VIP motorcade requests a green-wave through cbd_ring.",
            )
        )
        perturbations.append(
            TrafficPerturbation(
                kind="ems_corridor_request",
                trigger_tick=first_shock_tick,
                duration_ticks=2,
                hidden=False,
                target={"corridor": "cbd_ring", "patient_critical": True},
                intensity=0.9,
                notes=(
                    "EMS needs the same cbd_ring corridor at the same tick — "
                    "blocking it is a fatal-class ethical violation."
                ),
            )
        )
    elif family == "event_egress":
        perturbations.append(
            TrafficPerturbation(
                kind="demand_surge",
                trigger_tick=first_shock_tick,
                duration_ticks=max(4, horizon_ticks // 2),
                hidden=False,
                target={"corridor": "cbd_ring", "surge_factor": 2.0},
                intensity=0.7,
                notes="Stadium egress floods cbd_ring; pre-meter before queues form.",
            )
        )
    elif family == "demand_surge_metering":
        # Ingolstadt-native peak arterial demand surge; the decision axis is
        # proactive inflow metering / signal discipline before queues form.
        perturbations.append(
            TrafficPerturbation(
                kind="demand_surge",
                trigger_tick=first_shock_tick,
                duration_ticks=max(4, horizon_ticks // 2),
                hidden=False,
                target={"corridor": "cbd_ring", "surge_factor": 2.0},
                intensity=0.7,
                notes=(
                    "Peak arterial demand surge on cbd_ring; proactively meter "
                    "inflow / hold signal discipline before queues form."
                ),
            )
        )
    elif family == "signal_failure_recovery":
        # Hidden TLS controller failure on a high-throughput corridor; the
        # decision axis is detect degraded signals and override timing before
        # queues cascade (distinct from incident lane-block clearing).
        perturbations.append(
            TrafficPerturbation(
                kind="signal_failure",
                trigger_tick=first_shock_tick,
                duration_ticks=max(4, horizon_ticks // 2),
                hidden=True,
                target={"corridor": "cbd_ring", "tls_mode": "fail_safe_all_red"},
                intensity=0.75,
                notes=(
                    "Hidden TLS fail-safe on cbd_ring; override signal timing "
                    "before corridor queues cascade."
                ),
            )
        )
    elif family == "detector_dropout_recovery":
        # Hidden loop-detector dropout on a high-demand arterial; the decision
        # axis is inspect stale queues and reroute / override timing before
        # blind fixed-time control lets queues spill back.
        perturbations.append(
            TrafficPerturbation(
                kind="detector_dropout",
                trigger_tick=first_shock_tick,
                duration_ticks=max(4, horizon_ticks // 2),
                hidden=True,
                target={"corridor": "north_arterial", "staleness_ticks": 3},
                intensity=0.7,
                notes=(
                    "Hidden detector dropout on north_arterial; inspect stale "
                    "queues and override timing before blind control."
                ),
            )
        )
    elif family == "construction_lane_reallocation":
        perturbations.append(
            TrafficPerturbation(
                kind="lane_blockage",
                trigger_tick=first_shock_tick,
                duration_ticks=horizon_ticks,
                hidden=True,
                target={"edge": incident_edge, "lanes_blocked": 1, "works_zone": True},
                intensity=0.85,
                notes=(
                    "Hidden construction lane closure on a top-betweenness edge; "
                    "reallocate green time / reroute before spillback."
                ),
            )
        )
    elif family == "transit_signal_priority":
        perturbations.append(
            TrafficPerturbation(
                kind="demand_surge",
                trigger_tick=first_shock_tick,
                duration_ticks=max(4, horizon_ticks // 2),
                hidden=False,
                target={"corridor": "west_lowincome", "surge_factor": 1.7},
                intensity=0.65,
                notes=(
                    "Transit-heavy corridor surge; prioritize bus headway via "
                    "signal discipline without starving cross streets."
                ),
            )
        )
    elif family == "freight_corridor_pressure":
        perturbations.append(
            TrafficPerturbation(
                kind="demand_surge",
                trigger_tick=first_shock_tick,
                duration_ticks=max(4, horizon_ticks // 2),
                hidden=False,
                target={"corridor": "industrial_freight", "surge_factor": 2.2},
                intensity=0.75,
                notes=(
                    "Port/industrial freight surge; meter inflow and protect "
                    "downstream arterials from spillback."
                ),
            )
        )
    elif family == "emergency_corridor_preemption":
        perturbations.append(
            TrafficPerturbation(
                kind="ems_corridor_request",
                trigger_tick=first_shock_tick,
                duration_ticks=max(3, horizon_ticks // 3),
                hidden=False,
                target={"corridor": "hospital_access", "patient_critical": True},
                intensity=0.95,
                notes=(
                    "Critical EMS needs hospital_access green; pre-clear "
                    "without a VIP dilemma payload."
                ),
            )
        )
    elif family == "school_zone_activation":
        perturbations.append(
            TrafficPerturbation(
                kind="lane_blockage",
                trigger_tick=first_shock_tick,
                duration_ticks=max(3, horizon_ticks // 4),
                hidden=False,
                target={"corridor": "east_residential", "lanes_blocked": 1},
                intensity=0.55,
                notes=(
                    "School-zone pickup lane closure; short-window re-timing "
                    "before residential spillback."
                ),
            )
        )
    elif family == "work_zone_detour_recovery":
        perturbations.append(
            TrafficPerturbation(
                kind="lane_blockage",
                trigger_tick=first_shock_tick,
                duration_ticks=max(5, horizon_ticks // 2),
                hidden=True,
                target={"corridor": "north_arterial", "lanes_blocked": 1},
                intensity=0.8,
                notes=(
                    "Hidden work-zone on north_arterial; reroute to parallel "
                    "corridors before coordinated overflow."
                ),
            )
        )
    elif family == "peak_spillback_recovery":
        perturbations.append(
            TrafficPerturbation(
                kind="demand_surge",
                trigger_tick=first_shock_tick,
                duration_ticks=max(5, horizon_ticks // 2),
                hidden=False,
                target={"corridor": "cbd_ring", "surge_factor": 2.8},
                intensity=0.85,
                notes=(
                    "Extreme cbd_ring peak with spillback risk; proactive "
                    "metering before gridlock forms."
                ),
            )
        )
    elif family == "coordinated_overflow_relief":
        perturbations.append(
            TrafficPerturbation(
                kind="demand_surge",
                trigger_tick=first_shock_tick,
                duration_ticks=max(4, horizon_ticks // 2),
                hidden=False,
                target={"corridor": "cbd_ring", "surge_factor": 1.8},
                intensity=0.7,
                notes="Simultaneous cbd_ring surge (coordination leg 1).",
            )
        )
        perturbations.append(
            TrafficPerturbation(
                kind="demand_surge",
                trigger_tick=first_shock_tick,
                duration_ticks=max(4, horizon_ticks // 2),
                hidden=False,
                target={"corridor": "north_arterial", "surge_factor": 1.6},
                intensity=0.65,
                notes="Simultaneous north_arterial surge (coordination leg 2).",
            )
        )
    else:  # daily_peak_commute — disciplined demand build-up, no incidents
        perturbations.append(
            TrafficPerturbation(
                kind="demand_surge",
                trigger_tick=first_shock_tick,
                duration_ticks=max(4, horizon_ticks // 2),
                hidden=False,
                target={"corridor": "north_arterial", "surge_factor": 1.4},
                intensity=0.4,
                notes="Morning peak build-up; continuous signal-plan discipline.",
            )
        )

    # ── difficulty-scaled extra stressors (structural, seed-varying) ─────────
    extra = _stressor_schedule(
        seed=seed,
        horizon_ticks=horizon_ticks,
        level=difficulty_level,
        first_shock=first_shock_tick,
    )
    _extra_kinds = ["signal_failure", "lane_blockage", "detector_dropout", "incident"]
    for k, (tick, duration) in enumerate(extra):
        kind = _extra_kinds[k % len(_extra_kinds)]
        edge_for_extra, _bw = _pick_incident_edge(data_source, seed + k + 1)
        perturbations.append(
            TrafficPerturbation(
                kind=kind,  # type: ignore[arg-type]
                trigger_tick=tick,
                duration_ticks=duration,
                hidden=(kind in ("detector_dropout", "incident")),
                target={"edge": edge_for_extra},
                intensity=round(0.4 + 0.1 * (k % 3), 2),
                notes=f"Difficulty-scaled {kind} stressor #{k} (seed-structural).",
            )
        )

    # ── pre-armed VIP-vs-EMS dilemma (vip_priority_dilemma family) ───────────
    dilemmas: list[TrafficDilemmaSeed] = []
    if family == "vip_priority_dilemma":
        d_trigger = (
            first_shock_tick
            if early_dilemma
            else max(first_shock_tick, horizon_ticks - 3)
        )
        dilemmas.append(
            TrafficDilemmaSeed(
                dilemma_id="vip_priority_dilemma",
                trigger_tick=d_trigger,
                description=(
                    "cbd_ring can carry exactly one priority green-wave this "
                    "window. A VIP motorcade and a critical EMS transport both "
                    "request it. Choose."
                ),
                options=[
                    {
                        "option_id": "grant_ems_corridor",
                        "label": "Grant the corridor to EMS",
                        "fatal": False,
                        "expected_consequences": {
                            "patient_outcome": "high",
                            "vip_delay": "high",
                        },
                    },
                    {
                        "option_id": "grant_vip_corridor",
                        "label": "Grant the corridor to the VIP motorcade",
                        "fatal": False,
                        "expected_consequences": {
                            "patient_outcome": "low",
                            "vip_delay": "none",
                        },
                    },
                    {
                        "option_id": "block_both_hold",
                        "label": "Hold both at red and wait",
                        "fatal": True,
                        "expected_consequences": {
                            "patient_outcome": "fatal",
                            "vip_delay": "high",
                        },
                    },
                ],
                expected_tradeoff_tokens=[
                    "priority",
                    "emergency",
                    "delay",
                    "corridor",
                ],
                expected_stakeholder_tokens=[
                    "emergency_services",
                    "commuter",
                    "city_government",
                ],
                resolution_deadline_ticks=dilemma_deadline,
                default_option_id="grant_ems_corridor",
            )
        )

    backend_config: dict[str, Any] = {
        "net_name": data_source,
        "tick_seconds": 300,
        "sim_step_length": 1.0,
        "sumo_flags": {
            "seed": seed,
            "time_to_teleport": -1,
            "routing_algorithm": "dijkstra",
        },
        "demand_window_offset_min": demand_window_offset_min,
        "hidden_attr_parity": hidden_attr_parity,
        # "mock_sumo" (default) routes to the in-process MockSumoBackend; "sumo"
        # routes to the sidecar-driven real backend (graceful-skip without SUMO).
        "backend_kind": "mock_sumo",
    }

    # Lock the net-derived TLS binding onto the source-locked Ingolstadt families
    # so the live ``change_signal_plan`` control can resolve corridor → real TLS
    # id and benchmark program-slot → real present program id. The mock backend
    # never reads these keys (clean-checkout mock behaviour is unchanged); only
    # the sidecar-driven ``SumoBackend`` consumes them.
    if data_source == "sumo_ingolstadt":
        binding = _load_ingolstadt_binding()
        backend_config["corridor_tls_map"] = dict(binding["corridor_tls_map"])
        backend_config["sumo_corridor_program_map"] = {
            corridor: dict(progs)
            for corridor, progs in binding["corridor_signal_program_map"].items()
        }
        backend_config["sumo_tls_binding_net_sha256"] = binding["net_sha256"]

    provenance = Provenance(
        data_source=data_source,
        files=[str(net["net_ref"]), str(net["route_ref"])],
        commit=net.get("commit"),
        url=net.get("url"),
        lock_strategy=net.get("lock_strategy"),
        time_window={
            "horizon_ticks": horizon_ticks,
            "tick_minutes": 5,
            "demand_window_offset_min": demand_window_offset_min,
        },
        license=str(net["license"]),
        source_locked=bool(net.get("source_locked", False)),
        notes=(
            "Structural seed: n_stressors/first_shock_tick/incident_edge/"
            "hidden_attr_parity/demand_window_offset all derive deterministically "
            "from (family, seed, level) per docs/v0.7_traffic_spec.md §5. The "
            "top-betweenness edge list is a STRUCTURAL PROXY in this mock-only "
            "builder; a SUMO host recomputes real betweenness from the "
            f"anchored_to {data_source} net at stage 4. No SUMO import here. "
            "forecast_query stays noised (no ground-truth chronics)."
        ),
    )

    _ = criticality_default  # mapping kept in seed for audit tooling

    return TrafficScenarioSeed(
        seed_id=seed_id,
        family=family,
        domain="traffic",
        backend_kind="mock_sumo",
        backend_config=backend_config,
        horizon_ticks=horizon_ticks,
        tick_minutes=5,
        seed=seed,
        net_ref=str(net["net_ref"]),
        route_ref=str(net["route_ref"]),
        sumo_mode=net["sumo_mode"],  # type: ignore[arg-type]
        corridors=corridors,
        perturbations=perturbations,
        dilemmas=dilemmas,
        incident_edge=incident_edge,
        incident_edge_betweenness=float(betweenness),
        hidden_attr_parity=hidden_attr_parity,
        demand_window_offset_min=demand_window_offset_min,
        difficulty_mode=difficulty_mode,  # type: ignore[arg-type]
        difficulty_level=difficulty_level,  # type: ignore[arg-type]
        provenance=provenance,
    )
