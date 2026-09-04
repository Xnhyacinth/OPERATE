"""domains.traffic.oracle.wardrop_ue — deterministic system-optimal reference.

Traffic analogue of ``domains.microgrid.oracle.economic_dispatch``. It computes
a replay-stable lower bound on total travel-time cost (``production_cost`` in
the 14-key scorer contract) so ``evaluation/scorer.py:score_optimality_gap``
can report how far an agent's realized delay sits above the best achievable.

Reference policy (the "system optimum" a perfectly-playing operator reaches):

- **Optimal signal timing** — the best feasible signal program (max capacity
  bonus, ``incident_relief`` = ×1.30) is held on every corridor for the whole
  horizon. This is a configuration optimization with no production-cost (it
  only books ``startup_cost`` actuation lost-time, which is *not* in
  ``production_cost``), so a good agent can match it for free.
- **Optimal routing (Wardrop UE / system-optimal assignment)** — flow is
  pooled across substitutable corridors (achievable in the mock via
  ``reroute_flow``). Pooling aggregate demand against aggregate capacity is a
  valid *lower bound*: per-corridor ``min(inflow, cap)`` summed is always ≥ the
  pooled ``min(Σ inflow, Σ cap)``.
- **No discretionary surge** — mutual-aid relief crews and EMS/VIP priority
  bonuses are *excluded* (they are bounded, costed surge resources, the traffic
  analogue of "don't assume you can build new generation"). Including them would
  collapse the bound toward zero and make the gap meaningless.

Exogenous shocks the operator *cannot* avoid are applied exactly as the backend
does: ambient ``weather_capacity_drop`` (network-wide capacity factor),
``incident`` / ``lane_blockage`` capacity floors, and ``demand_surge`` inflow
spikes. ``signal_failure`` is overridden by the optimal signal program (the
operator restores timing), so it has no effect on the bound — matching the
backend, where ``change_signal_plan`` clears a ``fail_safe`` program.

The result is cached into ``seed.backend_config['reference_optimum']`` with the
same envelope shape the microgrid oracle uses (``reference_optimum``,
``method``, ``horizon``, ``deterministic_stop``) so the runner/scorer read a
single domain-agnostic scalar.
"""

from __future__ import annotations

from typing import Any

from domains.traffic.backends.mock_sumo import (
    _CAP_SLACK,
    _INCIDENT_CAP_FLOOR,
    _PROGRAM_BONUS,
    _SURGE_FAMILY_FACTOR_KEY,
    _VOT_PER_VEH_MIN,
    _det_hash,
)
from domains.traffic.seeds.schema import TrafficScenarioSeed

__all__ = ["compute_reference_optimum", "reference_optimum_cost"]

# Best feasible signal-program capacity bonus (operator restores optimal timing).
_BEST_PROGRAM_BONUS: float = max(_PROGRAM_BONUS.values())


def _peak_tick(seed_obj: TrafficScenarioSeed) -> int:
    """Replicate ``MockSumoBackend`` triangular-peak placement."""
    horizon = max(1, int(seed_obj.horizon_ticks))
    offset_ticks = int(seed_obj.demand_window_offset_min // 5)
    return min(horizon - 1, max(0, int(horizon * 0.4) + offset_ticks))


def _demand_weight(tick: int, horizon: int, peak: int) -> float:
    """Triangular weight, normalized to sum≈1 — byte-identical to the backend."""
    horizon = max(1, horizon)
    if tick <= peak:
        w = 0.2 + 0.8 * ((tick + 1) / (peak + 1))
    else:
        span = max(1, horizon - 1 - peak)
        w = 0.2 + 0.8 * max(0.0, (horizon - 1 - tick) / span)
    total = 0.0
    for t in range(horizon):
        if t <= peak:
            total += 0.2 + 0.8 * ((t + 1) / (peak + 1))
        else:
            span = max(1, horizon - 1 - peak)
            total += 0.2 + 0.8 * max(0.0, (horizon - 1 - t) / span)
    return w / max(total, 1e-9)


def _edge_to_corridor_map(seed_obj: TrafficScenarioSeed) -> dict[str, str]:
    """Replicate the backend's deterministic edge→corridor attribution."""
    cids = sorted(c.corridor_id for c in seed_obj.corridors)
    mapping: dict[str, str] = {}
    if not cids:
        return mapping
    if seed_obj.incident_edge:
        idx = _det_hash(seed_obj.seed, 0, seed_obj.incident_edge) % len(cids)
        mapping[seed_obj.incident_edge] = cids[idx]
    for p in seed_obj.perturbations:
        edge = str(p.target.get("edge", "")) if isinstance(p.target, dict) else ""
        if edge and edge not in mapping:
            idx = _det_hash(seed_obj.seed, 0, edge) % len(cids)
            mapping[edge] = cids[idx]
    return mapping


def _perturbation_corridor(
    p: Any, corridor_ids: set[str], edge_map: dict[str, str]
) -> str | None:
    tgt = p.target if isinstance(p.target, dict) else {}
    cid = str(tgt.get("corridor", ""))
    if cid and cid in corridor_ids:
        return cid
    edge = str(tgt.get("edge", ""))
    return edge_map.get(edge)


def reference_optimum_cost(seed_obj: TrafficScenarioSeed) -> float:
    """Minimum achievable total travel-time cost (production_cost lower bound).

    Pure function of the seed — no RNG, no backend mutation — so it is
    replay-stable across runs (counterfactual-replay + scorer contract).
    """
    horizon = max(1, int(seed_obj.horizon_ticks))
    tick_minutes = int(seed_obj.tick_minutes or 5)
    peak = _peak_tick(seed_obj)
    edge_map = _edge_to_corridor_map(seed_obj)
    corridor_ids = {c.corridor_id for c in seed_obj.corridors}

    base_cap = {
        c.corridor_id: (int(c.demand_veh) / horizon) * _CAP_SLACK
        for c in seed_obj.corridors
    }
    demand_veh = {c.corridor_id: int(c.demand_veh) for c in seed_obj.corridors}

    agg_queue = 0.0
    travel_cost = 0.0
    for t in range(horizon):
        # Exogenous shocks active at this tick.
        weather_factor = 1.0
        surge: dict[str, float] = {cid: 1.0 for cid in corridor_ids}
        incident_floor: dict[str, float] = {cid: 1.0 for cid in corridor_ids}
        for p in seed_obj.perturbations:
            active = p.trigger_tick <= t < p.trigger_tick + p.duration_ticks
            if not active:
                continue
            kind = str(p.kind)
            if kind == "weather_capacity_drop":
                weather_factor = float(p.target.get("capacity_factor", 0.7))
            elif kind in ("incident", "lane_blockage"):
                cid = _perturbation_corridor(p, corridor_ids, edge_map)
                if cid is not None:
                    intensity = float(p.intensity)
                    incident_floor[cid] = _INCIDENT_CAP_FLOOR + (
                        1.0 - _INCIDENT_CAP_FLOOR
                    ) * (1.0 - intensity)
            elif kind == "demand_surge":
                cid = _perturbation_corridor(p, corridor_ids, edge_map)
                if cid is not None:
                    surge[cid] = float(p.target.get(_SURGE_FAMILY_FACTOR_KEY, 1.5))
            # signal_failure overridden by optimal program; detector_dropout /
            # vip_arrival / ems_corridor_request have no capacity effect.

        weight = _demand_weight(t, horizon, peak)
        agg_demand = sum(demand_veh[cid] * weight * surge[cid] for cid in corridor_ids)
        agg_cap = sum(
            base_cap[cid] * weather_factor * _BEST_PROGRAM_BONUS * incident_floor[cid]
            for cid in corridor_ids
        )
        agg_cap = max(agg_cap, 1e-6)

        inflow = agg_demand + agg_queue
        served = min(inflow, agg_cap)
        agg_queue = max(0.0, inflow - served)
        travel_cost += agg_queue * tick_minutes * _VOT_PER_VEH_MIN

    return round(travel_cost, 2)


def compute_reference_optimum(
    seed_obj: TrafficScenarioSeed, *, write_back: bool = True
) -> dict[str, Any]:
    """Compute and (optionally) cache the system-optimal reference envelope.

    Mirrors the microgrid oracle's return/cache contract::

        seed_obj.backend_config['reference_optimum'] = {
            'reference_optimum': <float>,   # min total travel-time cost
            'method': 'wardrop_system_optimal',
            'horizon': <int>,
            'deterministic_stop': True,
        }
    """
    optimum = reference_optimum_cost(seed_obj)
    envelope = {
        "reference_optimum": optimum,
        "objective_component": "travel_time_cost",
        "method": "wardrop_system_optimal",
        "horizon": int(seed_obj.horizon_ticks),
        "deterministic_stop": True,
    }
    if write_back:
        if not isinstance(seed_obj.backend_config, dict):
            seed_obj.backend_config = {}
        seed_obj.backend_config["reference_optimum"] = envelope
    return envelope
