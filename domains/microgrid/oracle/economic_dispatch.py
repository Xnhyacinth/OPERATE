"""
domains.microgrid.oracle.economic_dispatch — offline EMS reference optimum.

Computes the offline LP economic-dispatch optimum for a microgrid scenario
and caches it into the seed's ``backend_config['reference_optimum']``. This
is the ``optimality_gap`` oracle and the headroom-gate upper bound
(spec §8/§9).

Determinism (spec §9 + Risk #2): the LP is **convex** so its optimal
*objective value* is unique and solver-independent — no wall-clock time
limit is ever used. The oracle is computed **offline once** and the
per-tick scorer reads the cached scalar — fully replay-stable. It is NEVER
invoked live per tick.

When cvxpy is absent the deterministic pure-Python greedy fallback (serve
load from cheapest source per tick: renewables → battery → genset/grid by
price) yields the cached optimum so the cache/determinism contract holds on
a deps-absent host (clearly labeled ``method="greedy_fallback"``).
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..seeds.schema import MicrogridScenarioSeed

CVXPY_AVAILABLE = importlib.util.find_spec("cvxpy") is not None


def _islanded_ticks(seed_obj: MicrogridScenarioSeed) -> set[int]:
    out: set[int] = set()
    for p in seed_obj.perturbations:
        if str(p.kind) == "grid_outage":
            for t in range(p.trigger_tick, p.trigger_tick + p.duration_ticks):
                out.add(t)
    return out


def _profiles(
    seed_obj: MicrogridScenarioSeed, horizon: int
) -> dict[str, list[float]]:
    prof = (seed_obj.backend_config or {}).get("profiles", {}) or {}

    def _arr(name: str, default: float) -> list[float]:
        a = [float(x) for x in (prof.get(name) or [])]
        if len(a) >= horizon:
            return a[:horizon]
        return a + [default] * (horizon - len(a))

    return {
        "load": _arr("load_mw", 0.0),
        "pv": _arr("pv_mw", 0.0),
        "wind": _arr("wind_mw", 0.0),
        "price": _arr("price", 40.0),
    }


def compute_reference_optimum(
    seed_obj: MicrogridScenarioSeed, *, cache: bool = True
) -> dict[str, Any]:
    """Compute + (optionally) cache the deterministic EMS reference optimum.

    Returns a JSON-safe dict ``{reference_optimum, method, horizon,
    deterministic_stop}``. Two calls with the same seed return identical
    results (replay-stable).
    """
    horizon = int(seed_obj.horizon_ticks)
    cfg = seed_obj.backend_config or {}
    prof = _profiles(seed_obj, horizon)
    islanded = _islanded_ticks(seed_obj)

    batt = dict(cfg.get("battery", {}) or {})
    cap = float(batt.get("capacity_mwh", 0.0) or 0.0)
    max_ch = float(batt.get("max_charge_mw", cap) or cap)
    max_dis = float(batt.get("max_discharge_mw", cap) or cap)
    eff = float(batt.get("efficiency", 0.95) or 0.95)
    init_soc = float(batt.get("init_soc", 0.5) or 0.5) * cap

    gen = dict(cfg.get("genset", {}) or {})
    g_max = float(gen.get("max_mw", 0.0) or 0.0) if gen.get("available") else 0.0
    g_cost = float(gen.get("fuel_cost_per_mwh", 120.0) or 120.0)

    grid = dict(cfg.get("grid", {}) or {})
    max_imp = float(grid.get("max_import_mw", 0.0) or 0.0)
    max_exp = float(grid.get("max_export_mw", 0.0) or 0.0)
    degr = 8.0  # battery degradation $/MWh (matches ems_sim)

    if CVXPY_AVAILABLE:
        opt, method = _solve_cvxpy(
            horizon,
            prof,
            islanded,
            cap,
            max_ch,
            max_dis,
            eff,
            init_soc,
            g_max,
            g_cost,
            max_imp,
            max_exp,
            degr,
        )
    else:  # pragma: no cover - exercised only when cvxpy absent
        opt, method = _solve_greedy(
            horizon,
            prof,
            islanded,
            cap,
            max_ch,
            max_dis,
            eff,
            init_soc,
            g_max,
            g_cost,
            max_imp,
            max_exp,
            degr,
        )

    result = {
        "reference_optimum": round(float(opt), 4),
        "objective_component": "production_cost",
        "method": method,
        "horizon": horizon,
        "deterministic_stop": {"time_limit": None, "convex_lp": True},
    }
    if cache:
        seed_obj.backend_config["reference_optimum"] = dict(result)
    return result


def _solve_cvxpy(
    horizon,
    prof,
    islanded,
    cap,
    max_ch,
    max_dis,
    eff,
    init_soc,
    g_max,
    g_cost,
    max_imp,
    max_exp,
    degr,
) -> tuple[float, str]:
    import cvxpy as cp  # type: ignore[import]

    g = cp.Variable(horizon, nonneg=True)
    imp = cp.Variable(horizon, nonneg=True)
    exp = cp.Variable(horizon, nonneg=True)
    ch = cp.Variable(horizon, nonneg=True)
    dis = cp.Variable(horizon, nonneg=True)
    soc = cp.Variable(horizon + 1, nonneg=True)

    cons = [soc[0] == init_soc, soc <= cap, g <= g_max, ch <= max_ch, dis <= max_dis]
    for t in range(horizon):
        cons.append(soc[t + 1] == soc[t] + eff * ch[t] - dis[t] / max(0.01, eff))
        if t in islanded:
            cons.append(imp[t] == 0)
            cons.append(exp[t] == 0)
        else:
            cons.append(imp[t] <= max_imp)
            cons.append(exp[t] <= max_exp)
        # Power balance (oracle serves all load; no shedding).
        supply = prof["pv"][t] + prof["wind"][t] + g[t] + imp[t] + dis[t]
        cons.append(supply - exp[t] - ch[t] == prof["load"][t])

    cost = cp.sum(
        cp.multiply(g, g_cost) + cp.multiply(imp, prof["price"]) + degr * (ch + dis)
    )
    prob = cp.Problem(cp.Minimize(cost), cons)
    try:
        prob.solve()
        if prob.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
            return float(prob.value), "cvxpy_lp"
    except Exception:
        pass
    return _solve_greedy(
        horizon,
        prof,
        islanded,
        cap,
        max_ch,
        max_dis,
        eff,
        init_soc,
        g_max,
        g_cost,
        max_imp,
        max_exp,
        degr,
    )


def _solve_greedy(
    horizon,
    prof,
    islanded,
    cap,
    max_ch,
    max_dis,
    eff,
    init_soc,
    g_max,
    g_cost,
    max_imp,
    max_exp,
    degr,
) -> tuple[float, str]:
    """Deterministic per-tick greedy dispatch (renewable → battery → grid/genset)."""
    soc = init_soc
    total = 0.0
    for t in range(horizon):
        residual = prof["load"][t] - prof["pv"][t] - prof["wind"][t]
        if residual <= 0:
            # Surplus → charge battery (deterministic), rest exported free.
            charge = min(max_ch, (cap - soc) / max(0.01, 1.0))
            charge = min(charge, -residual)
            soc = min(cap, soc + eff * charge)
            total += degr * charge
            continue
        # Discharge battery first (cheap relative to grid at peak).
        dis = min(max_dis, soc, residual)
        soc -= dis
        total += degr * dis
        residual -= dis
        if residual <= 0:
            continue
        if t in islanded:
            # No grid → genset covers the rest (greedy ignores infeasibility).
            g = min(g_max, residual)
            total += g * g_cost
            residual -= g
        else:
            imp = min(max_imp, residual)
            total += imp * prof["price"][t]
            residual -= imp
            if residual > 0 and g_max > 0:
                g = min(g_max, residual)
                total += g * g_cost
    return float(total), "greedy_fallback"
