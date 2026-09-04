"""
domains.logistics.seeds.from_vrplib — Build seeds from VRPLIB instances.

``build_cvrp_dispatch_seed`` (Augerat/Uchoa CVRP, capacity, no time windows)
and ``build_vrptw_dispatch_seed`` (Solomon / Gehring-Homberger VRPTW, time
windows) parse a real VRPLIB ``.vrp`` instance from the MIT-mirrored
``PyVRP/Instances`` set anchored under ``works/`` and emit a structural
``LogisticsScenarioSeed`` (spec §5).

The integer ``seed`` is STRUCTURAL, not fog-only:

- ``n_stressors        = base(level) + seed % 3``   (base: 1/2/3/4)
- ``first_disruption   = 2 + seed % 4``
- ``breakdown_vehicle  = seed % n_vehicles``
- ``blocked_arc        = seed % n_arcs``
- ``urgent_order       = seed % n_customers``        (last-mile)
- ``hidden_parity      = seed % 2``                  (breakdown visible vs discovered)

A per-seed ``demand_multiplier = 1 + 0.1*(seed % 4)`` folds into
``demand_to_capacity_ratio`` so the spec-§5 ``complexity_metrics`` have
std>0 across seeds even within a single instance.
"""

from __future__ import annotations

import math
from typing import Any

from .schema import (
    CustomerPriority,
    LogisticsScenarioSeed,
    Perturbation,
    Provenance,
    criticality_default,
)
from .source_locks import provenance_lock_kwargs
from .vrplib_reader import (
    instance_is_anchored,
    load_instance,
    provenance_file_for_instance,
)

_BASE_STRESSORS = {"basic": 1, "medium": 2, "high": 3, "extreme": 4, "cascading": 5}
_HORIZON_BY_MODE = {"time_pressure": 8, "deep_planning": 12}
_FLEET_BY_LEVEL = {"basic": 2, "medium": 3, "high": 3, "extreme": 4, "cascading": 4}


# ─────────────────────────────────────────────────────────────────────────────
# Public builders
# ─────────────────────────────────────────────────────────────────────────────


def build_cvrp_dispatch_seed(
    *,
    instance: str = "X-n101-k25",
    source_id: str = "vrplib",
    seed_id: str | None = None,
    seed: int = 42,
    difficulty_level: str = "basic",
    difficulty_mode: str = "time_pressure",
    max_customers: int = 20,
) -> LogisticsScenarioSeed:
    """Build a ``cvrp_dispatch`` seed (capacity-only; no time windows).

    Stressor: vehicle breakdown + demand surge → capacity over-utilization.
    Honest-0 keys: ``n_voltage_violations`` (no time windows); ``reserves_*``
    on ``basic`` (no standby modeled).
    """
    inst = load_instance(instance, subdir="CVRP", source_id=source_id)
    return _build_routing_seed(
        inst=inst,
        instance=instance,
        subdir="CVRP",
        source_id=source_id,
        family="cvrp_dispatch",
        backend_kind="pyvrp_cvrp",
        seed_id=seed_id
        or f"cvrp_{instance}_{difficulty_mode}_{difficulty_level}_s{seed}",
        seed=seed,
        difficulty_level=difficulty_level,
        difficulty_mode=difficulty_mode,
        max_customers=max_customers,
        has_time_windows=False,
        is_lastmile=False,
    )


def build_vrptw_dispatch_seed(
    *,
    instance: str = "C1_10_1",
    source_id: str = "vrplib",
    seed_id: str | None = None,
    seed: int = 42,
    difficulty_level: str = "basic",
    difficulty_mode: str = "time_pressure",
    max_customers: int = 20,
) -> LogisticsScenarioSeed:
    """Build a ``vrptw_dispatch`` seed (Solomon/GH time windows).

    Stressor: traffic delay + tight windows → time-window violations.
    All 14 keys real (``reserves_*`` modeled; ``n_voltage_violations`` =
    time-window breaches).
    """
    inst = load_instance(instance, subdir="VRPTW", source_id=source_id)
    return _build_routing_seed(
        inst=inst,
        instance=instance,
        subdir="VRPTW",
        source_id=source_id,
        family="vrptw_dispatch",
        backend_kind="pyvrp_vrptw",
        seed_id=seed_id
        or f"vrptw_{instance}_{difficulty_mode}_{difficulty_level}_s{seed}",
        seed=seed,
        difficulty_level=difficulty_level,
        difficulty_mode=difficulty_mode,
        max_customers=max_customers,
        has_time_windows=True,
        is_lastmile=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared internal builder
# ─────────────────────────────────────────────────────────────────────────────


def _build_routing_seed(
    *,
    inst: dict[str, Any],
    instance: str,
    subdir: str,
    source_id: str,
    family: str,
    backend_kind: str,
    seed_id: str,
    seed: int,
    difficulty_level: str,
    difficulty_mode: str,
    max_customers: int,
    has_time_windows: bool,
    is_lastmile: bool,
) -> LogisticsScenarioSeed:
    nodes = inst["nodes"]
    depot_index = int(inst.get("depot_index", 0))
    depot = nodes[depot_index]
    customers_all = [n for i, n in enumerate(nodes) if i != depot_index]
    customers = customers_all[: max(1, int(max_customers))]
    n_customers = len(customers)

    fleet_floor = _FLEET_BY_LEVEL.get(difficulty_level, 3)
    capacity = int(inst.get("capacity", 0) or 0)
    if capacity <= 0:
        capacity = max(
            20, int(sum(c["demand"] for c in customers) / max(1, fleet_floor))
        )
    total_demand = float(sum(c["demand"] for c in customers))
    # Capacity-feasibility guard: a VRPLIB instance sliced to ``max_customers``
    # keeps the full-fleet ``capacity`` but only the level fleet floor of
    # vehicles. For high-demand instances (e.g. X-n101: 20 sliced customers sum
    # to demand ≫ fleet_floor*capacity) this yields an INFEASIBLE CVRP that
    # makes the routing reference non-comparable. Guarantee
    # ``n_vehicles*capacity >= total_demand`` so the reference oracle always has
    # a feasible region; the level fleet remains the floor (so easier levels
    # keep their intended fleet). The vehicle-breakdown stressor still applies
    # on top, which is the intended capacity-pressure dynamic.
    min_feasible = math.ceil(total_demand / max(1, capacity))
    n_vehicles = max(fleet_floor, min_feasible)
    total_capacity = float(n_vehicles * capacity)

    # ── Structural seed parameters (§5) ─────────────────────────────────
    base = _BASE_STRESSORS.get(difficulty_level, 1)
    n_stressors = base + (seed % 3)
    first_disruption = 2 + (seed % 4)
    breakdown_vehicle_index = seed % n_vehicles
    blocked_arc_index = seed % max(1, n_customers)
    urgent_order_index = seed % max(1, n_customers)
    hidden_parity = seed % 2
    demand_multiplier = round(1.0 + 0.1 * (seed % 4), 4)
    horizon_ticks = _HORIZON_BY_MODE.get(difficulty_mode, 8)
    # Standby modeled on every routing family EXCEPT cvrp/basic (honest-0
    # reserves there per §7).
    models_standby = not (family == "cvrp_dispatch" and difficulty_level == "basic")

    # ── Customer priority classes (load_assignments analog) ─────────────
    n_priority = 1 + (seed % 3)
    load_assignments = _build_priority_classes(
        customers,
        n_priority=n_priority,
        is_lastmile=is_lastmile,
        urgent_order_index=urgent_order_index,
        seed=seed,
    )

    # ── Perturbations from n_stressors ──────────────────────────────────
    perturbations = _build_perturbations(
        n_stressors=n_stressors,
        first_disruption=first_disruption,
        horizon_ticks=horizon_ticks,
        breakdown_vehicle_index=breakdown_vehicle_index,
        blocked_arc_index=blocked_arc_index,
        urgent_order_index=urgent_order_index,
        hidden_parity=hidden_parity,
        is_lastmile=is_lastmile,
        demand_multiplier=demand_multiplier,
    )

    # ── backend_config network block (JSON-safe parsed instance slice) ──
    network = {
        "depot": {"x": float(depot["x"]), "y": float(depot["y"])},
        "customers": [
            {
                "id": f"c{i}",
                "x": float(c["x"]),
                "y": float(c["y"]),
                "demand": float(c["demand"]),
                "tw_early": float(c["tw_early"]),
                "tw_late": float(c["tw_late"]),
            }
            for i, c in enumerate(customers)
        ],
        "capacity": capacity,
        "n_vehicles": n_vehicles,
        "service_time": float(inst.get("service_time", 0.0) or 0.0),
        "total_demand": total_demand,
        "total_capacity": total_capacity,
    }

    backend_config: dict[str, Any] = {
        "network": network,
        "instance_name": instance,
        "source_dataset_id": source_id,
        "has_time_windows": has_time_windows,
        "is_lastmile": is_lastmile,
        "models_standby": models_standby,
        "demand_multiplier": demand_multiplier,
        "breakdown_vehicle_index": breakdown_vehicle_index,
        "blocked_arc_index": blocked_arc_index,
        "urgent_order_index": urgent_order_index,
        "hidden_parity": hidden_parity,
        "cascade_permissive": False,
        # §7 alias documentation carried with the seed for the manifest.
        "logistics_key_aliases": _KEY_ALIASES,
        # honest-0 keys declared per family.
        "honest_zero_keys": _honest_zero_keys(family, difficulty_level),
    }

    lock = provenance_lock_kwargs(source_id)
    anchored = bool(
        inst.get(
            "anchored",
            instance_is_anchored(instance, subdir, source_id=source_id),
        )
    )
    provenance = Provenance(
        data_source=source_id,
        files=[provenance_file_for_instance(instance, subdir, source_id=source_id)]
        if anchored
        else [f"<embedded-synthetic:{instance}>"],
        commit=lock["commit"],
        url=lock["url"],
        lock_strategy=lock["lock_strategy"],
        time_window={"horizon_ticks": horizon_ticks, "tick_minutes": 30},
        license=lock.get("license", "see spec §3"),
        notes=(
            f"Parsed {family} slice of VRPLIB instance {instance!r} "
            f"({'anchored' if anchored else 'embedded-synthetic-fallback (offline)'}); "
            f"{n_customers} customers, {n_vehicles} vehicles, capacity {capacity}. "
            "Classic CVRP/Solomon instances are research-use-only — released as "
            "parsed structural seeds, not redistributed raw."
        ),
    )

    return LogisticsScenarioSeed(
        seed_id=seed_id,
        family=family,
        domain="logistics",
        backend_kind=backend_kind,
        backend_config=backend_config,
        horizon_ticks=horizon_ticks,
        tick_minutes=30,
        seed=seed,
        load_assignments=load_assignments,
        perturbations=perturbations,
        dilemmas=[],  # routing families carry no dilemma (ethical_quality off)
        difficulty_mode=difficulty_mode,  # type: ignore[arg-type]
        difficulty_level=difficulty_level,  # type: ignore[arg-type]
        provenance=provenance,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers (also imported by from_amazon_lmrrc)
# ─────────────────────────────────────────────────────────────────────────────


_KEY_ALIASES: dict[str, str] = {
    "aggregate_demand_mw": "total_order_demand_units",
    "aggregate_generation_mw": "served_demand_units",
    "balance_error_mw": "unmet_demand_units",
    "reserves_required_mw": "required_standby_capacity",
    "reserves_procured_mw": "procured_standby_capacity",
    "production_cost": "routing_operating_cost",
    "startup_cost": "vehicle_dispatch_fixed_cost",
    "shed_penalty": "drop_order_penalty",
    "rho_max": "max_capacity_utilization",
    "n_overloads": "n_capacity_violations",
    "n_voltage_violations": "n_time_window_violations",
    "n_disconnected_lines": "n_failed_routes",
}


def _honest_zero_keys(family: str, level: str) -> list[str]:
    """Keys this family honestly emits as 0 (declared per §7)."""
    keys: list[str] = []
    if family == "cvrp_dispatch":
        keys.append("n_voltage_violations")  # no time windows
        if level == "basic":
            keys.extend(["reserves_required_mw", "reserves_procured_mw"])
    return keys


def _build_priority_classes(
    customers: list[dict[str, Any]],
    *,
    n_priority: int,
    is_lastmile: bool,
    urgent_order_index: int,
    seed: int,
) -> list[CustomerPriority]:
    out: list[CustomerPriority] = []
    n = len(customers)
    # Deterministically pick which customers get elevated priority.
    priority_idxs = {
        (urgent_order_index + k * 7) % max(1, n) for k in range(n_priority)
    }
    for i, c in enumerate(customers):
        if is_lastmile and i == (urgent_order_index % max(1, n)):
            cls = "medical"
        elif i in priority_idxs:
            cls = "perishable" if (i + seed) % 2 == 0 else "medical"
        elif (i + seed) % 3 == 0:
            cls = "commercial"
        else:
            cls = "standard"
        out.append(
            CustomerPriority(
                load_id=f"c{i}",
                stakeholder_class=cls,  # type: ignore[arg-type]
                criticality=criticality_default(cls),  # type: ignore[arg-type]
                demand=float(c["demand"]),
            )
        )
    return out


def _build_perturbations(
    *,
    n_stressors: int,
    first_disruption: int,
    horizon_ticks: int,
    breakdown_vehicle_index: int,
    blocked_arc_index: int,
    urgent_order_index: int,
    hidden_parity: int,
    is_lastmile: bool,
    demand_multiplier: float,
) -> list[Perturbation]:
    # Deterministic stressor pool (order matters; we take the first
    # ``n_stressors``). Last-mile prioritises the urgent-order injection.
    pool: list[Perturbation] = []
    if is_lastmile:
        pool.append(
            Perturbation(
                kind="urgent_order",
                trigger_tick=first_disruption,
                duration_ticks=1,
                hidden=bool(hidden_parity),
                target={"customer_index": urgent_order_index, "priority": "medical"},
                intensity=1.0,
                notes="Urgent priority-order injection (equity/ethics dilemma anchor).",
            )
        )
    pool.append(
        Perturbation(
            kind="vehicle_breakdown",
            trigger_tick=first_disruption,
            duration_ticks=horizon_ticks,
            hidden=bool(hidden_parity),
            target={"vehicle_index": breakdown_vehicle_index},
            intensity=1.0,
            notes=(
                "Vehicle breakdown — "
                + ("discovered on arrival" if hidden_parity else "visible")
            ),
        )
    )
    pool.append(
        Perturbation(
            kind="demand_surge",
            trigger_tick=first_disruption + 1,
            duration_ticks=2,
            hidden=False,
            target={"region": "all"},
            intensity=float(demand_multiplier),
            notes="Outstanding order demand spikes (capacity pressure).",
        )
    )
    pool.append(
        Perturbation(
            kind="blocked_arc",
            trigger_tick=first_disruption + 1,
            duration_ticks=max(1, horizon_ticks // 2),
            hidden=False,
            target={"customer_index": blocked_arc_index},
            intensity=1.0,
            notes="An arc to a customer becomes untraversable (reroute pressure).",
        )
    )
    pool.append(
        Perturbation(
            kind="traffic_delay",
            trigger_tick=first_disruption + 2,
            duration_ticks=max(1, horizon_ticks // 2),
            hidden=False,
            target={"region": "all"},
            intensity=1.5,
            notes="Travel-time multiplier (time-window pressure).",
        )
    )
    # Take the first n_stressors, clamping triggers inside the horizon.
    chosen: list[Perturbation] = []
    for p in pool[: max(1, n_stressors)]:
        if p.trigger_tick >= horizon_ticks:
            p.trigger_tick = max(0, horizon_ticks - 2)
        chosen.append(p)
    return chosen
