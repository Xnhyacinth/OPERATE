"""
evaluation.lp_oracle — Lightweight LP economic-dispatch oracle.

v0.2 deliverable. Given a `ScenarioSeed` + the realized chronics, this
module computes the minimum-cost dispatch the system could have achieved
*if* all decisions were made jointly with perfect foresight by a
linear-programming solver. The resulting cost is used as an
optimality-floor reference for the new `optimality_gap` scoring
dimension:

    optimality_gap = (actual_cost − lp_optimum) / lp_optimum

A score of 0 means the agent matched the perfect-info LP; large gaps
mean the agent left substantial cost on the table. We deliberately use
a DC linear approximation rather than a full AC OPF so the oracle is:

- deterministic (no nonlinear solver convergence quirks)
- fast (sub-second per scenario on pglib-uc cases)
- Python-only (`scipy.optimize.linprog`, already a dependency).

This is NOT the agent's oracle for action selection — that is
`baselines.oracle_offline.OracleOfflineAgent` which uses heuristic
rules. The LP oracle is purely an evaluation tool that ships a cost
floor for the audit and the leaderboard.

References:
- Wood & Wollenberg, "Power Generation, Operation, and Control" §3.7
  (economic dispatch as a linear program with piecewise marginal cost)
- Knueven et al., pglib-uc README §3 (the published cases are exactly
  the right scale for an LP relaxation: ~50–200 thermal units, 24–48
  hourly periods).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# scipy is imported lazily inside ``lp_dispatch_optimum`` so that the
# ``evaluation`` package (and any module that re-exports from it) remains
# importable in environments where scipy is not installed. The LP oracle is an
# optional evaluation feature; consumers that never call it should not pay an
# import-time dependency on scipy.


@dataclass
class LpOracleResult:
    optimum_cost: float
    n_periods: int
    n_units: int
    feasible: bool
    notes: str = ""


def lp_dispatch_optimum(
    pglib_case: dict[str, Any],
    *,
    n_periods: int | None = None,
) -> LpOracleResult:
    """Compute the LP economic-dispatch optimum for a pglib-uc case.

    Variables: ``p[i, t]`` = output of unit i in period t (MW).
    Objective: minimise sum_i sum_t c_i * p[i, t] where c_i is the
    average marginal cost of the unit's piecewise-linear cost curve.
    Constraints (per period):
        - sum_i p[i, t] = demand[t]   (power balance)
        - 0 <= p[i, t] <= power_max[i]
    Renewable production is held at its nameplate (best-case) so it
    becomes a free injection that reduces effective thermal demand.

    Notes:
    - We deliberately ignore startup costs, ramp limits, min-up/down
      times, and reserves. The LP is therefore an OPTIMISTIC bound:
      every agent's cost should be >= this number. This is the
      property the optimality_gap dimension exploits.
    - When the case is infeasible (renewable exceeds demand somewhere),
      we relax the equality to an inequality (over-generation is free).
    """
    try:
        from scipy.optimize import linprog  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover — exercised only when scipy missing
        raise ImportError(
            "scipy is required for the LP oracle (lp_dispatch_optimum). "
            "Install it via `pip install scipy` to enable optimality_gap "
            "scoring."
        ) from exc

    thermals = pglib_case.get("thermal_generators", {}) or {}
    renewables = pglib_case.get("renewable_generators", {}) or {}
    demand = list(pglib_case.get("demand", []) or [])
    if not demand:
        return LpOracleResult(
            optimum_cost=0.0,
            n_periods=0,
            n_units=0,
            feasible=False,
            notes="empty demand series",
        )
    T = n_periods if n_periods is not None else len(demand)
    T = min(T, len(demand))
    if T <= 0:
        return LpOracleResult(
            optimum_cost=0.0,
            n_periods=0,
            n_units=0,
            feasible=False,
            notes="non-positive horizon",
        )

    unit_ids = sorted(thermals.keys())
    n_units = len(unit_ids)
    if n_units == 0:
        return LpOracleResult(
            optimum_cost=0.0,
            n_periods=T,
            n_units=0,
            feasible=False,
            notes="no thermal units in case",
        )

    # Per-unit average marginal cost: take the piecewise-linear cost
    # curve's mean slope.
    avg_marginal: list[float] = []
    power_max: list[float] = []
    for uid in unit_ids:
        spec = thermals[uid]
        pmax = float(spec.get("power_output_maximum", 0.0))
        power_max.append(pmax)
        cps = spec.get("piecewise_production", []) or []
        if cps and len(cps) >= 2:
            # average slope across all interior segments
            slopes = []
            for i in range(len(cps) - 1):
                p0, c0 = float(cps[i]["mw"]), float(cps[i]["cost"])
                p1, c1 = float(cps[i + 1]["mw"]), float(cps[i + 1]["cost"])
                if p1 > p0:
                    slopes.append((c1 - c0) / (p1 - p0))
            avg_marginal.append(float(np.mean(slopes)) if slopes else 50.0)
        else:
            avg_marginal.append(50.0)

    # Per-period renewable injection ceiling (best case = nameplate).
    ren_per_period = np.zeros(T, dtype=float)
    for _rid, rspec in renewables.items():
        # `power_output_maximum` is the upper envelope per period.
        pmax_series = list(rspec.get("power_output_maximum", []) or [])
        for t in range(min(T, len(pmax_series))):
            ren_per_period[t] += float(pmax_series[t])

    # Solve T independent LPs (one per period) — faster than one big LP.
    total_cost = 0.0
    feasible_all = True
    notes_parts: list[str] = []
    for t in range(T):
        effective_demand = max(0.0, float(demand[t]) - ren_per_period[t])
        # If renewable alone covers demand, thermal cost = 0.
        if effective_demand <= 1e-6:
            continue
        # Decision vars: p_i for thermal i.
        c = np.asarray(avg_marginal, dtype=float)
        # 0 <= p_i <= power_max[i]
        bounds = [(0.0, pmax) for pmax in power_max]
        # sum(p_i) >= effective_demand   (LP inequality for slack)
        A_ub = -np.ones((1, n_units), dtype=float)
        b_ub = np.array([-effective_demand], dtype=float)
        # Make sure capacity is enough — otherwise infeasible.
        if sum(power_max) < effective_demand - 1e-3:
            feasible_all = False
            notes_parts.append(
                f"t={t}: capacity {sum(power_max):.1f} < demand {effective_demand:.1f}"
            )
            # Fall back to clamping demand to total capacity.
            b_ub = np.array([-sum(power_max)], dtype=float)
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
            if res.success:
                total_cost += float(res.fun)
            else:
                feasible_all = False
                notes_parts.append(f"t={t}: linprog status={res.status}")
        except Exception as exc:  # pragma: no cover — defensive
            feasible_all = False
            notes_parts.append(f"t={t}: linprog raised {type(exc).__name__}")

    return LpOracleResult(
        optimum_cost=round(total_cost, 2),
        n_periods=T,
        n_units=n_units,
        feasible=feasible_all,
        notes="; ".join(notes_parts[:5]),
    )


def optimality_gap_score(
    actual_production_cost: float,
    lp_optimum: float,
) -> dict[str, Any]:
    """Compute the optimality-gap scoring blob.

    IMPORTANT: compares ONLY the ``production_cost`` component, NOT
    the aggregate cost. The LP doesn't model balance_error_cost,
    shed_penalty, or reserve_violation_cost — those are captured by
    other dimensions (safety_violation, equity_fairness). Mixing
    them here would double-count safety failures and produce gaps in
    the 30–50× range that obscure actual dispatch efficiency.

    Returns:
        {"gap": float, "raw_score": 0..100, "notes": str}

    Scoring curve: ``100 / (1 + gap)`` so a perfect agent scores 100,
    a 2×-suboptimal agent scores 33, a 10× one scores 9.
    """
    if lp_optimum <= 0:
        return {
            "gap": float("inf"),
            "raw_score": 0.0,
            "notes": "no LP optimum (case infeasible or empty)",
        }
    gap = max(0.0, (actual_production_cost - lp_optimum) / lp_optimum)
    score = 100.0 / (1.0 + gap)
    return {
        "gap": round(gap, 4),
        "raw_score": round(score, 2),
        "notes": (
            f"production_cost={actual_production_cost:.1f} vs lp_opt={lp_optimum:.1f}"
        ),
    }
