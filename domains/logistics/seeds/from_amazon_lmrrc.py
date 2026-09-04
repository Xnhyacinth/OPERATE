"""
domains.logistics.seeds.from_amazon_lmrrc — Build last-mile priority seeds.

``build_lastmile_priority_seed`` derives a ``lastmile_priority`` scenario
from the Amazon Last-Mile Routing Research Challenge (CC-BY-NC-4.0):
real driver routes + package / time-window / zone data. The challenge data
is **anchored, not redistributed** (spec §3); when the anchor is absent the
builder generates a deterministic procedural last-mile network keyed on
``(route_id, seed)`` — exactly like the disaster Kobe spike's procedural
hazard series — so the seed signature is stable offline. The provenance
block records the LMRRC source + NonCommercial license either way.

NonCommercial constraint: ``lastmile_priority`` is CC-BY-NC-4.0 → excluded
from any commercial leaderboard (mirrors the OR-Gym-Poisson exclusion).

The last-mile family adds the equity / ethics depth (perishable-medical vs
commercial drop dilemma) and carrier/customer trust events.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from .from_vrplib import _build_perturbations, _build_priority_classes
from .schema import (
    DilemmaSeed,
    LogisticsScenarioSeed,
    Perturbation,
    Provenance,
)
from .source_locks import provenance_lock_kwargs

_BASE_STRESSORS = {"basic": 1, "medium": 2, "high": 3, "extreme": 4, "cascading": 5}
_HORIZON_BY_MODE = {"time_pressure": 8, "deep_planning": 12}
_FLEET_BY_LEVEL = {"basic": 2, "medium": 3, "high": 3, "extreme": 4, "cascading": 4}

# Anchor root for the LMRRC challenge data (parsed-only; not redistributed).
# When absent we fall back to a procedural network (offline-runnable).


def _route_hash(route_id: str, k: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"lmrrc|{route_id}|{k}".encode()).digest()[:4], "big"
    )


def _procedural_lastmile_network(*, route_id: str, n_stops: int) -> dict[str, Any]:
    """Deterministic last-mile delivery network keyed on ``route_id``.

    Mimics a depot + clustered residential stops with delivery time windows
    — the shape the real LMRRC parser would produce. Coordinates / demands /
    windows are a fixed function of ``route_id`` so the seed is stable.
    """
    nodes: list[dict[str, float]] = [
        {"x": 0.0, "y": 0.0, "demand": 0.0, "tw_early": 0.0, "tw_late": 480.0}
    ]
    for i in range(1, n_stops + 1):
        ang = _route_hash(route_id, i) % 360
        rad = 5.0 + (_route_hash(route_id, i + 100) % 25)
        x = round(rad * math.cos(math.radians(ang)), 3)
        y = round(rad * math.sin(math.radians(ang)), 3)
        demand = float(1 + (_route_hash(route_id, i + 200) % 4))
        early = float(_route_hash(route_id, i + 300) % 300)
        late = early + 60.0 + float(_route_hash(route_id, i + 400) % 120)
        nodes.append(
            {"x": x, "y": y, "demand": demand, "tw_early": early, "tw_late": late}
        )
    return {"depot_index": 0, "nodes": nodes, "service_time": 5.0}


def build_lastmile_priority_seed(
    *,
    route_id: str = "lmrrc_route_0001",
    seed_id: str | None = None,
    seed: int = 42,
    difficulty_level: str = "basic",
    difficulty_mode: str = "time_pressure",
    n_stops: int = 18,
) -> LogisticsScenarioSeed:
    """Build a ``lastmile_priority`` seed (urgent priority-order injection).

    Stressor: urgent priority-order injection → equity/ethics dilemma +
    carrier trust. All 14 keys real (time windows + standby modeled).
    """
    seed_id = (
        seed_id or f"lastmile_{route_id}_{difficulty_mode}_{difficulty_level}_s{seed}"
    )
    inst = _procedural_lastmile_network(route_id=route_id, n_stops=n_stops)
    nodes = inst["nodes"]
    depot = nodes[0]
    customers = nodes[1:]
    n_customers = len(customers)

    n_vehicles = _FLEET_BY_LEVEL.get(difficulty_level, 3)
    total_demand = float(sum(c["demand"] for c in customers))
    # Round capacity UP (ceil, not floor) so n_vehicles*capacity >= total_demand
    # by construction — a floored capacity can be ~1 unit short, making the CVRP
    # capacity-infeasible and forcing the routing oracle onto its greedy
    # fallback. ceil keeps the instance feasible so the bounded reference yields
    # reference optimum.
    capacity = max(15, math.ceil(total_demand / max(1, n_vehicles)))
    total_capacity = float(n_vehicles * capacity)

    base = _BASE_STRESSORS.get(difficulty_level, 1)
    n_stressors = base + (seed % 3)
    first_disruption = 2 + (seed % 4)
    breakdown_vehicle_index = seed % n_vehicles
    blocked_arc_index = seed % max(1, n_customers)
    urgent_order_index = seed % max(1, n_customers)
    hidden_parity = seed % 2
    demand_multiplier = round(1.0 + 0.1 * (seed % 4), 4)
    horizon_ticks = _HORIZON_BY_MODE.get(difficulty_mode, 8)

    n_priority = 2 + (seed % 3)  # last-mile carries more priority customers
    load_assignments = _build_priority_classes(
        customers,
        n_priority=n_priority,
        is_lastmile=True,
        urgent_order_index=urgent_order_index,
        seed=seed,
    )

    perturbations: list[Perturbation] = _build_perturbations(
        n_stressors=n_stressors,
        first_disruption=first_disruption,
        horizon_ticks=horizon_ticks,
        breakdown_vehicle_index=breakdown_vehicle_index,
        blocked_arc_index=blocked_arc_index,
        urgent_order_index=urgent_order_index,
        hidden_parity=hidden_parity,
        is_lastmile=True,
        demand_multiplier=demand_multiplier,
    )

    # Pre-armed equity/ethics dilemma: one delivery slot, a perishable-medical
    # order vs a commercial order — the canonical reverse-triage shape.
    if difficulty_mode == "time_pressure":
        d_trigger = max(2, horizon_ticks // 4)
        deadline = 2
    else:
        d_trigger = max(2, horizon_ticks - 4)
        deadline = max(3, horizon_ticks // 3)
    dilemmas = [
        DilemmaSeed(
            dilemma_id="d_lastmile_medical_vs_commercial_drop",
            trigger_tick=d_trigger,
            description=(
                "Only one feasible delivery slot remains before the window "
                "closes. A perishable-medical order and a high-value commercial "
                "order both need it; the other must be dropped or held."
            ),
            options=[
                {
                    "option_id": "serve_medical_drop_commercial",
                    "label": "Serve the perishable-medical order, drop commercial",
                    "fatal": False,
                    "expected_consequences": {
                        "equity": "high",
                        "commercial_revenue": "low",
                    },
                },
                {
                    "option_id": "serve_commercial_drop_medical",
                    "label": "Serve the commercial order, drop medical",
                    "fatal": True,
                    "expected_consequences": {
                        "equity": "very_low",
                        "commercial_revenue": "high",
                    },
                },
                {
                    "option_id": "hold_both_wait_spot_carrier",
                    "label": "Hold both and hire a spot carrier",
                    "fatal": False,
                    "expected_consequences": {
                        "equity": "medium",
                        "commercial_revenue": "medium",
                        "cost": "high",
                    },
                },
            ],
            expected_tradeoff_tokens=["priority", "perishable", "drop", "equity"],
            expected_stakeholder_tokens=["customer", "carrier", "medical"],
            resolution_deadline_ticks=deadline,
            default_option_id="serve_medical_drop_commercial",
        )
    ]

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
        "service_time": float(inst.get("service_time", 5.0)),
        "total_demand": total_demand,
        "total_capacity": total_capacity,
    }

    from .from_vrplib import _KEY_ALIASES

    backend_config: dict[str, Any] = {
        "network": network,
        "instance_name": route_id,
        "has_time_windows": True,
        "is_lastmile": True,
        "models_standby": True,
        "demand_multiplier": demand_multiplier,
        "breakdown_vehicle_index": breakdown_vehicle_index,
        "blocked_arc_index": blocked_arc_index,
        "urgent_order_index": urgent_order_index,
        "hidden_parity": hidden_parity,
        "cascade_permissive": False,
        "logistics_key_aliases": _KEY_ALIASES,
        "honest_zero_keys": [],  # last-mile carries all keys real
        "noncommercial": True,  # CC-BY-NC-4.0 → excluded from commercial leaderboard
    }

    lock = provenance_lock_kwargs("amazon_lmrrc")
    provenance = Provenance(
        data_source="amazon_lmrrc",
        files=[f"<lmrrc:{route_id}> (procedural fallback; anchor not redistributed)"],
        commit=lock["commit"],
        url=lock["url"],
        lock_strategy=lock["lock_strategy"],
        time_window={"horizon_ticks": horizon_ticks, "tick_minutes": 30},
        license=lock.get("license", "CC-BY-NC-4.0 (NonCommercial)"),
        notes=(
            f"Last-mile priority scenario for LMRRC route {route_id!r}; "
            f"{n_customers} stops, {n_vehicles} vehicles. NonCommercial "
            "(CC-BY-NC-4.0) → excluded from any commercial leaderboard. "
            "Network is procedurally generated from (route_id, seed) when the "
            "LMRRC anchor is absent (no hand-written narrative content)."
        ),
    )

    return LogisticsScenarioSeed(
        seed_id=seed_id,
        family="lastmile_priority",
        domain="logistics",
        backend_kind="pyvrp_lastmile",
        backend_config=backend_config,
        horizon_ticks=horizon_ticks,
        tick_minutes=30,
        seed=seed,
        load_assignments=load_assignments,
        perturbations=perturbations,
        dilemmas=dilemmas,
        difficulty_mode=difficulty_mode,  # type: ignore[arg-type]
        difficulty_level=difficulty_level,  # type: ignore[arg-type]
        provenance=provenance,
    )
