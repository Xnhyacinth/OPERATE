"""
domains.power_grid.backends.pglib_uc_synthetic — Pure-Python UC backend.

Drives a deterministic, lightweight unit-commitment-style simulator from
a pglib-uc JSON case. State per tick:

- For each thermal generator: committed (0/1), power_output (MW), hours_up,
  hours_down, off_time_for_startup_cost.
- For each load bus: assigned demand fraction × aggregate demand (with
  perturbations applied), shed_mw, cumulative_shed_mwh.
- For each renewable: min/max envelope, current_output.
- Reserves: required (from case + perturbations), procured.
- Aggregate metrics: production_cost, startup_cost, shed_penalty,
  reserve_violation_mw, voltage_violation_tick_count (placeholder for the
  AC-flow backend).

The simulator does NOT solve OPF; it accepts tool-call deltas from the
agent (commit/decommit, redispatch, shed) and applies the chronics + the
perturbations. Generation is constrained by ramps and min-up/down times.
Aggregate generation may not equal aggregate demand — the residual is
reported as ``balance_error`` and counted toward the safety dimension.

This keeps the backend honest: the agent must actively maintain the
balance, and "doing nothing" leads to measurable balance error during
demand swings or generator outages.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

from ..seeds.schema import LoadAssignment, Perturbation, ScenarioSeed
from ..source_paths import resolve_source_ref

PGLIB_UC_ROOT_REL = "works/pglib-uc"


def _semantic_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


@dataclass
class GeneratorState:
    gen_id: str
    committed: bool
    output_mw: float
    hours_up: int
    hours_down: int
    power_min: float
    power_max: float
    ramp_up: float
    ramp_down: float
    time_up_minimum: int
    time_down_minimum: int
    must_run: bool
    forced_outage_until: int = -1
    fuel_supply_factor: float = 1.0  # 1.0 = full supply; <1 = throttled


@dataclass
class LoadState:
    load_id: str
    bus_id: str | None
    stakeholder_class: str
    criticality: float
    demand_fraction: float  # share of aggregate demand
    current_demand_mw: float = 0.0
    shed_mw: float = 0.0
    cumulative_shed_mwh: float = 0.0


@dataclass
class TickRecord:
    tick: int
    aggregate_demand_mw: float
    aggregate_generation_mw: float
    balance_error_mw: float
    reserves_required_mw: float
    reserves_procured_mw: float
    production_cost: float
    startup_cost: float
    shed_penalty: float
    realized_events: list[dict[str, Any]] = field(default_factory=list)


class PglibUcSyntheticBackend:
    """Self-contained UC backend.

    Public surface:

    - ``reset(scenario_seed)`` initializes state from the case JSON + seed.
    - ``snapshot()`` returns the full ground-truth state dict.
    - ``apply_tool_effect(name, args)`` mutates state in response to a tool
      handler (the tool layer translates LLM intents into these calls).
    - ``tick(current_tick)`` advances chronics, applies perturbations and
      pending forced outages, recomputes balance + costs, returns a
      ``TickRecord``.
    """

    # Keep the runtime identity on the backend itself.  The power-grid
    # adapter uses this value to attach the registry's bounded supervisory
    # cadence; without it a real PGLib instance exposed only its native
    # source-change opportunities and silently lost periodic review evidence.
    backend_kind = "pglib_uc_synthetic"

    # Per-class "Value of Lost Load" tariffs (USD per MWh of unserved
    # energy). Calibrated so 15 minutes of residential shed costs ~$50
    # and hospital shed costs ~$1250 (catastrophic). Approximate
    # lower-bound VoLL values from Lawrence Berkeley Lab studies.
    SHED_PENALTY_PER_MWH_BY_CLASS: dict[str, float] = {
        "residential": 200.0,
        "commercial": 400.0,
        "industrial": 600.0,
        "data_center": 1500.0,
        "transit": 1500.0,
        "water": 2500.0,
        "hospital": 5000.0,
    }
    SHED_PENALTY_DEFAULT = 1000.0
    # Backwards-compatible legacy constant; new code should use the
    # ``shed_penalty_for(load)`` helper instead.
    SHED_PENALTY_PER_MWH = SHED_PENALTY_DEFAULT
    RESERVE_VIOLATION_PENALTY_PER_MW = 50.0
    BALANCE_ERROR_PENALTY_PER_MW = 200.0

    def __init__(self) -> None:
        self._case: dict[str, Any] | None = None
        self._seed_obj: ScenarioSeed | None = None
        self._rng: random.Random = random.Random(0)
        self._tick: int = 0
        self._horizon: int = 24
        self._loads: dict[str, LoadState] = {}
        self._gens: dict[str, GeneratorState] = {}
        self._renew: dict[str, dict[str, Any]] = {}
        self._tick_records: list[TickRecord] = []
        # forecast bias (per perturbation) — applied to forecast_query
        self._forecast_bias: float = 0.0
        # v0.2.1: when a seed provides a real per-tick DA→RT error
        # profile (e.g. from RTS-GMLC), `forecast_for()` consumes it
        # tick-by-tick instead of the scalar `_forecast_bias`. The list
        # is direction-signed and indexed by absolute tick.
        self._forecast_bias_profile: list[float] = []
        self._case_source_path: str = ""
        self._case_source_declared: str = ""
        self._case_source_sha256: str = ""
        self._initial_source_state_digest: str = ""
        self._source_consumption_ticks: list[int] = []
        self._post_source_state_digests: list[dict[str, Any]] = []
        self._runtime_source_events: list[dict[str, Any]] = []
        self._pending_action_effects: list[dict[str, Any]] = []
        self._action_effects: list[dict[str, Any]] = []
        self._native_reference_diagnostics: dict[str, Any] = {}
        self._native_reference_plan: dict[str, Any] = {}

    # ── Reset ───────────────────────────────────────────────────────────

    def reset(self, scenario_seed: ScenarioSeed) -> None:
        self._seed_obj = scenario_seed
        self._rng = random.Random(scenario_seed.seed)
        self._tick = 0
        self._horizon = scenario_seed.horizon_ticks
        self._tick_records.clear()
        self._forecast_bias = 0.0
        self._forecast_bias_profile = []
        self._source_consumption_ticks = []
        self._post_source_state_digests = []
        self._runtime_source_events = []
        self._pending_action_effects = []
        self._action_effects = []
        self._native_reference_diagnostics = {}
        self._native_reference_plan = {}
        # v0.2.2 F-01: per-episode delayed-effect queue for request_mutual_aid
        # entries are (due_tick, mw) and drained at the start of tick().
        self._pending_mutual_aid: list[tuple[int, float]] = []

        case_rel = scenario_seed.backend_config.get("case_file")
        if not case_rel:
            raise ValueError("backend_config.case_file missing for pglib_uc_synthetic")
        case_path = resolve_source_ref(case_rel, description="pglib-uc case")
        with open(case_path, encoding="utf-8") as f:
            self._case = json.load(f)
        self._case_source_path = str(case_path.resolve())
        self._case_source_declared = str(case_rel)
        self._case_source_sha256 = hashlib.sha256(case_path.read_bytes()).hexdigest()

        self._init_loads(scenario_seed.load_assignments)
        self._init_gens()
        self._init_renewables()
        self._initial_source_state_digest = self._source_state_digest()
        # apply persistent perturbations that fire at tick 0
        self._apply_perturbations_at_tick(0)

    # ── Initialization helpers ──────────────────────────────────────────

    def _init_loads(self, assignments: list[LoadAssignment]) -> None:
        self._loads.clear()
        # Normalize fractions: split equally among assignments using the
        # default distribution weights.
        n = max(len(assignments), 1)
        # Use stakeholder distribution if available
        from ..seeds.schema import default_stakeholder_distribution

        weights = default_stakeholder_distribution()
        for a in assignments:
            w = weights.get(a.stakeholder_class, 1.0 / n)
            self._loads[a.load_id] = LoadState(
                load_id=a.load_id,
                bus_id=a.bus_id,
                stakeholder_class=a.stakeholder_class,
                criticality=a.criticality,
                demand_fraction=w,
            )
        total = sum(load.demand_fraction for load in self._loads.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            for load in self._loads.values():
                load.demand_fraction = load.demand_fraction / total

    def _init_gens(self) -> None:
        self._gens.clear()
        assert self._case is not None
        for gid, g in self._case.get("thermal_generators", {}).items():
            self._gens[gid] = GeneratorState(
                gen_id=gid,
                committed=bool(g.get("unit_on_t0", False)) or bool(g.get("must_run", False)),
                output_mw=float(g.get("power_output_t0", 0.0) or 0.0),
                hours_up=int(g.get("time_up_t0", 0) or 0),
                hours_down=int(g.get("time_down_t0", 0) or 0),
                power_min=float(g.get("power_output_minimum", 0.0)),
                power_max=float(g.get("power_output_maximum", 0.0)),
                ramp_up=float(g.get("ramp_up_limit", 0.0)),
                ramp_down=float(g.get("ramp_down_limit", 0.0)),
                time_up_minimum=int(g.get("time_up_minimum", 1)),
                time_down_minimum=int(g.get("time_down_minimum", 1)),
                must_run=bool(g.get("must_run", False)),
            )

    def _init_renewables(self) -> None:
        self._renew.clear()
        assert self._case is not None
        for gid, r in self._case.get("renewable_generators", {}).items():
            self._renew[gid] = {
                "min": list(r.get("power_output_minimum", [])),
                "max": list(r.get("power_output_maximum", [])),
                "current": 0.0,
            }

    def native_oracle_reference_dispatch(self, *, max_calls: int = 4) -> list[dict[str, Any]]:
        """Solve a bounded rolling-horizon, source-native UC reference.

        The MILP is an upper-reference policy, not a power-flow solver.  It
        binds the locked PGLib-UC schedules and generator constraints over a
        short future horizon, then returns only first-period controls.  Those
        controls still pass through ToolProtocol and the backend's native
        validation path.
        """
        if max_calls <= 0 or self._case is None:
            return []
        target_tick = len(self._tick_records)
        if target_tick >= self._horizon:
            return []
        solution: dict[str, Any] | None = None
        plan_offset = 0
        plan = self._native_reference_plan
        if plan:
            plan_offset = target_tick - int(plan.get("start_tick", -1))
            dispatch = list(plan.get("dispatch_mw") or [])
            commitment = list(plan.get("commitment") or [])
            if (
                0 < plan_offset < len(dispatch)
                and plan_offset < len(commitment)
                and self._native_reference_plan_state_matches(
                    plan,
                    completed_offset=plan_offset - 1,
                )
            ):
                solution = {
                    "dispatch_mw": dispatch[plan_offset:],
                    "commitment": commitment[plan_offset:],
                }
                self._native_reference_diagnostics = {
                    **dict(plan.get("diagnostics") or {}),
                    "plan_reused": True,
                    "plan_start_tick": int(plan["start_tick"]),
                    "plan_target_tick": target_tick,
                }
            else:
                self._native_reference_plan = {}
        if solution is None:
            horizon = min(4, self._horizon - target_tick)
            solution = self._solve_native_reference_uc(
                start_tick=target_tick,
                horizon=horizon,
            )
            if solution is not None:
                self._native_reference_plan = {
                    "start_tick": target_tick,
                    "dispatch_mw": list(solution["dispatch_mw"]),
                    "commitment": list(solution["commitment"]),
                    "diagnostics": dict(self._native_reference_diagnostics),
                }
                self._native_reference_diagnostics = {
                    **self._native_reference_diagnostics,
                    "plan_reused": False,
                    "plan_start_tick": target_tick,
                    "plan_target_tick": target_tick,
                }
        if solution is None:
            return []

        first_dispatch = solution["dispatch_mw"][0]
        first_commitment = solution["commitment"][0]
        horizon = len(solution["dispatch_mw"])
        changes: list[dict[str, Any]] = []
        for gid in sorted(self._gens):
            generator = self._gens[gid]
            target = float(first_dispatch[gid])
            committed = bool(first_commitment[gid])
            if committed == generator.committed and abs(target - generator.output_mw) <= max(
                0.01, generator.power_max * 1e-6
            ):
                continue
            changes.append(
                {
                    "tool": "redispatch_generation",
                    "args": {
                        "generator_id": gid,
                        "target_mw": round(target, 6),
                        **({"commit": committed} if committed != generator.committed else {}),
                    },
                    "delta_mw": target - generator.output_mw,
                }
            )
        changes.sort(
            key=lambda item: (
                -abs(float(item["delta_mw"])),
                str(item["args"]["generator_id"]),
            )
        )
        if not changes:
            return []
        return [
            {
                "tool": "dispatch_generation_portfolio",
                "args": {
                    "dispatches": [dict(item["args"]) for item in changes],
                    "source_tick": target_tick,
                    "rolling_horizon_ticks": horizon,
                },
            }
        ]

    def _native_reference_plan_state_matches(
        self,
        plan: dict[str, Any],
        *,
        completed_offset: int,
    ) -> bool:
        """Require executed state to match the cached plan before reuse."""
        dispatch = list(plan.get("dispatch_mw") or [])
        commitment = list(plan.get("commitment") or [])
        if not (0 <= completed_offset < len(dispatch) and completed_offset < len(commitment)):
            return False
        expected_dispatch = dispatch[completed_offset]
        expected_commitment = commitment[completed_offset]
        for gid, generator in self._gens.items():
            if gid not in expected_dispatch or gid not in expected_commitment:
                return False
            if bool(expected_commitment[gid]) != generator.committed:
                return False
            tolerance = max(0.01, generator.power_max * 1e-6)
            if abs(float(expected_dispatch[gid]) - generator.output_mw) > tolerance:
                return False
        return True

    def native_oracle_reference_diagnostics(self) -> dict[str, Any]:
        """Return immutable diagnostics for the most recent UC solve."""
        return dict(self._native_reference_diagnostics)

    def _solve_native_reference_uc(
        self,
        *,
        start_tick: int,
        horizon: int,
    ) -> dict[str, Any] | None:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix

        assert self._case is not None
        gids = sorted(self._gens)
        n_gen = len(gids)
        n_period = max(1, horizon)
        n_fields = 5
        n_vars = n_gen * n_period * n_fields

        def variable(field: int, period: int, generator: int) -> int:
            return (period * n_gen + generator) * n_fields + field

        p_field, u_field, start_field, stop_field, cost_field = range(n_fields)
        objective = np.zeros(n_vars)
        lower_bounds = np.zeros(n_vars)
        upper_bounds = np.full(n_vars, np.inf)
        integrality: Any = np.zeros(n_vars, dtype=np.uint8)
        specs = self._case.get("thermal_generators", {})

        for period in range(n_period):
            for generator, gid in enumerate(gids):
                state = self._gens[gid]
                upper_bounds[variable(p_field, period, generator)] = state.power_max
                for variable_field in (u_field, start_field, stop_field):
                    index = variable(variable_field, period, generator)
                    upper_bounds[index] = 1.0
                    integrality[index] = 1
                objective[variable(cost_field, period, generator)] = 1.0
                objective[variable(u_field, period, generator)] += float(
                    specs.get(gid, {}).get("no_load_cost", 0.0)
                )
                startup = list(specs.get(gid, {}).get("startup") or [])
                objective[variable(start_field, period, generator)] += float(
                    startup[0].get("cost", 0.0) if startup else 0.0
                )

        row_indices: list[int] = []
        column_indices: list[int] = []
        coefficients: list[float] = []
        row_lower: list[float] = []
        row_upper: list[float] = []

        def add_constraint(terms: dict[int, float], minimum: float, maximum: float) -> None:
            row = len(row_lower)
            for column, coefficient in terms.items():
                if coefficient:
                    row_indices.append(row)
                    column_indices.append(column)
                    coefficients.append(coefficient)
            row_lower.append(minimum)
            row_upper.append(maximum)

        interval_data = [
            self._native_reference_interval(start_tick + period) for period in range(n_period)
        ]
        for period, (demand, renewable, reserve) in enumerate(interval_data):
            thermal = max(0.0, demand - renewable)
            add_constraint(
                {variable(p_field, period, generator): 1.0 for generator in range(n_gen)},
                thermal,
                thermal,
            )
            reserve_terms: dict[int, float] = {}
            for generator, gid in enumerate(gids):
                maximum = self._native_reference_available_max(gid, start_tick + period)
                reserve_terms[variable(u_field, period, generator)] = maximum
                reserve_terms[variable(p_field, period, generator)] = -1.0
            add_constraint(reserve_terms, reserve, np.inf)

        for generator, gid in enumerate(gids):
            state = self._gens[gid]
            spec = specs.get(gid, {})
            startup_ramp = float(spec.get("ramp_startup_limit", state.ramp_up))
            shutdown_ramp = float(spec.get("ramp_shutdown_limit", state.ramp_down))
            for period in range(n_period):
                absolute_tick = start_tick + period
                p_index = variable(p_field, period, generator)
                u_index = variable(u_field, period, generator)
                start_index = variable(start_field, period, generator)
                stop_index = variable(stop_field, period, generator)
                available_max = self._native_reference_available_max(gid, absolute_tick)
                add_constraint({p_index: 1.0, u_index: -available_max}, -np.inf, 0.0)
                add_constraint({p_index: 1.0, u_index: -state.power_min}, 0.0, np.inf)
                previous_u = 1.0 if state.committed else 0.0
                transition = {
                    u_index: 1.0,
                    start_index: -1.0,
                    stop_index: 1.0,
                }
                if period:
                    transition[variable(u_field, period - 1, generator)] = -1.0
                    transition_rhs = 0.0
                else:
                    transition_rhs = previous_u
                add_constraint(transition, transition_rhs, transition_rhs)

                ramp_up_terms = {
                    p_index: 1.0,
                    start_index: -startup_ramp,
                }
                ramp_down_terms = {
                    p_index: -1.0,
                    stop_index: -shutdown_ramp,
                }
                if period:
                    previous_p_index = variable(p_field, period - 1, generator)
                    previous_u_index = variable(u_field, period - 1, generator)
                    ramp_up_terms[previous_p_index] = -1.0
                    ramp_up_terms[previous_u_index] = -state.ramp_up
                    ramp_down_terms[previous_p_index] = 1.0
                    ramp_down_terms[u_index] = -state.ramp_down
                    ramp_up_rhs = 0.0
                    ramp_down_rhs = 0.0
                else:
                    ramp_down_terms[u_index] = -state.ramp_down
                    ramp_up_rhs = state.output_mw + state.ramp_up * previous_u
                    ramp_down_rhs = -state.output_mw

                forced_off = available_max + 1e-9 < state.power_min
                if not forced_off:
                    add_constraint(ramp_up_terms, -np.inf, ramp_up_rhs)
                    add_constraint(ramp_down_terms, -np.inf, ramp_down_rhs)
                if forced_off:
                    add_constraint({u_index: 1.0}, 0.0, 0.0)
                elif state.must_run:
                    add_constraint({u_index: 1.0}, 1.0, 1.0)

                if not forced_off:
                    up_start = max(0, period - state.time_up_minimum + 1)
                    add_constraint(
                        {
                            **{
                                variable(start_field, t, generator): 1.0
                                for t in range(up_start, period + 1)
                            },
                            u_index: -1.0,
                        },
                        -np.inf,
                        0.0,
                    )
                down_start = max(0, period - state.time_down_minimum + 1)
                add_constraint(
                    {
                        **{
                            variable(stop_field, t, generator): 1.0
                            for t in range(down_start, period + 1)
                        },
                        u_index: 1.0,
                    },
                    -np.inf,
                    1.0,
                )
                if not forced_off and state.committed and state.hours_up < state.time_up_minimum:
                    locked = state.time_up_minimum - state.hours_up
                    if period < locked:
                        add_constraint({u_index: 1.0}, 1.0, 1.0)
                if not state.committed and state.hours_down < state.time_down_minimum:
                    locked = state.time_down_minimum - state.hours_down
                    if period < locked:
                        add_constraint({u_index: 1.0}, 0.0, 0.0)

                cost_index = variable(cost_field, period, generator)
                points = sorted(
                    list(spec.get("piecewise_production") or []),
                    key=lambda point: float(point.get("mw", 0.0)),
                )
                segments = list(zip(points, points[1:], strict=False))
                if not segments and points:
                    point = points[0]
                    mw = max(float(point.get("mw", 0.0)), 1e-9)
                    segments = [
                        (
                            {"mw": 0.0, "cost": 0.0},
                            {"mw": mw, "cost": float(point.get("cost", 0.0))},
                        )
                    ]
                for left, right in segments:
                    left_mw = float(left.get("mw", 0.0))
                    right_mw = float(right.get("mw", 0.0))
                    if right_mw <= left_mw:
                        continue
                    slope = (float(right.get("cost", 0.0)) - float(left.get("cost", 0.0))) / (
                        right_mw - left_mw
                    )
                    intercept = float(left.get("cost", 0.0)) - slope * left_mw
                    add_constraint(
                        {
                            cost_index: 1.0,
                            p_index: -slope,
                            u_index: -intercept,
                        },
                        0.0,
                        np.inf,
                    )

        matrix = coo_matrix(
            (coefficients, (row_indices, column_indices)),
            shape=(len(row_lower), n_vars),
        ).tocsr()
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=LinearConstraint(matrix, row_lower, row_upper),
            options={"time_limit": 20.0, "mip_rel_gap": 1e-6},
        )
        constraint_contract = {
            "binary_commitment_startup_shutdown": True,
            "demand_balance": True,
            "forced_outage": True,
            "minimum_up_down": True,
            "native_initial_state": True,
            "ramp": True,
            "reserve": True,
        }
        if not result.success or result.x is None:
            self._native_reference_diagnostics = {
                "solver": "scipy.optimize.milp",
                "status": "infeasible_or_timeout",
                "message": str(result.message),
                "rolling_horizon_ticks": n_period,
                "n_thermal_generators": n_gen,
                "constraints": constraint_contract,
            }
            return None

        dispatch: list[dict[str, float]] = []
        commitment: list[dict[str, bool]] = []
        maximum_residual = 0.0
        for period, (demand, renewable, _reserve) in enumerate(interval_data):
            dispatch_row = {
                gid: float(result.x[variable(p_field, period, generator)])
                for generator, gid in enumerate(gids)
            }
            commitment_row = {
                gid: bool(round(result.x[variable(u_field, period, generator)]))
                for generator, gid in enumerate(gids)
            }
            dispatch.append(dispatch_row)
            commitment.append(commitment_row)
            maximum_residual = max(
                maximum_residual,
                abs(sum(dispatch_row.values()) + renewable - demand),
            )
        self._native_reference_diagnostics = {
            "solver": "scipy.optimize.milp",
            "status": "optimal",
            "message": str(result.message),
            "rolling_horizon_ticks": n_period,
            "n_thermal_generators": n_gen,
            "objective": float(result.fun),
            "maximum_balance_residual_mw": maximum_residual,
            "constraints": constraint_contract,
        }
        return {"dispatch_mw": dispatch, "commitment": commitment}

    def _native_reference_available_max(self, gid: str, tick: int) -> float:
        generator = self._gens[gid]
        if tick < generator.forced_outage_until:
            return 0.0
        # Reconstruct each reference interval from the source schedule.  The
        # live factor describes only the current interval and must not leak
        # into later rolling-horizon periods after its perturbation expires.
        factor = 1.0
        sorted_gids = sorted(self._gens)
        for perturbation in self._seed_obj.perturbations if self._seed_obj else []:
            if not self._perturbation_active(perturbation, tick):
                continue
            if perturbation.kind == "generator_forced_outage":
                index = int(perturbation.target.get("index", 0)) % max(len(sorted_gids), 1)
                if sorted_gids[index] == gid:
                    return 0.0
            elif perturbation.kind == "planned_maintenance":
                fraction = float(perturbation.target.get("fraction", 0.05))
                cutoff = max(1, int(len(sorted_gids) * fraction))
                if gid in sorted_gids[:cutoff]:
                    return 0.0
            elif perturbation.kind == "fuel_supply_delay":
                target_fuels = {
                    str(fuel).lower()
                    for fuel in perturbation.target.get("fuels", ["natural_gas", "gas", "ng"])
                }
                fuel_tag = str(
                    (self._case or {})
                    .get("thermal_generators", {})
                    .get(gid, {})
                    .get("fuel", "natural_gas")
                ).lower()
                if not fuel_tag or fuel_tag in target_fuels:
                    factor = min(
                        factor,
                        max(0.1, 1.0 - float(perturbation.intensity)),
                    )
        effective_max = max(0.0, generator.power_max * factor)
        # A fuel restriction must not make a must-run unit impossible to
        # commit.  PGLib cases commonly mark nuclear baseload units as
        # ``must_run`` with a long minimum-down time.  If a temporary fuel
        # factor drives ``effective_max`` below ``power_min``, the old model
        # forced the unit off, then made the rolling MILP infeasible when the
        # restriction expired because the unit could not restart before its
        # minimum-down timer.  Keep the native must-run commitment and cap it
        # at its technical minimum; forced outages still take precedence via
        # the early return above.
        if generator.must_run:
            effective_max = max(effective_max, generator.power_min)
        return effective_max

    def _native_reference_interval(self, tick: int) -> tuple[float, float, float]:
        case = self._case or {}

        def _at(values: list[Any]) -> float:
            return float(values[min(tick, len(values) - 1)]) if values else 0.0

        demand = _at(list(case.get("demand") or []))
        reserve = _at(list(case.get("reserves") or []))
        renewable = 0.0
        wind_factor = 1.0
        for perturbation in self._seed_obj.perturbations if self._seed_obj else []:
            if not self._perturbation_active(perturbation, tick):
                continue
            if perturbation.kind == "load_surge":
                demand *= 1.0 + float(perturbation.intensity)
            elif perturbation.kind == "wind_dropout":
                wind_factor = max(0.1, 1.0 - float(perturbation.intensity))
        for spec in case.get("renewable_generators", {}).values():
            minimum = _at(list(spec.get("power_output_minimum") or []))
            maximum = _at(list(spec.get("power_output_maximum") or []))
            renewable += max(minimum, min(maximum, (minimum + maximum) / 2.0 * wind_factor))
        return demand, renewable, reserve

    # ── Tool effects (called by tool handlers) ──────────────────────────

    def apply_tool_effect(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Mutate state from a tool call. Returns a dict the handler echoes."""
        before_digest = self._action_state_digest()
        if name == "redispatch_generation":
            result = self._redispatch(args)
        elif name == "dispatch_generation_portfolio":
            result = self._dispatch_generation_portfolio(args)
        elif name == "commit_reserve":
            result = self._commit_reserve(args)
        elif name == "shed_load":
            result = self._shed_load(args)
        elif name == "switch_branch":
            # No branch model in synthetic backend; record intent only
            result = {
                "_status": "noop",
                "reason": "no branches in synthetic UC backend",
            }
        elif name == "topology_action":
            result = {
                "_status": "noop",
                "reason": "no topology in synthetic UC backend",
            }
        elif name == "request_mutual_aid":
            # v0.2.2 F-01: request_mutual_aid no longer goes through
            # apply_tool_effect — it has its own handler that queues a
            # delayed effect on the backend. Keep this branch as a defensive
            # no-op so any caller still routing through apply_tool_effect()
            # gets a clear, non-mutating ack.
            result = {
                "_status": "ack",
                "info": (
                    "mutual-aid uses the dedicated delayed-effect path; "
                    "this code path no longer mutates state"
                ),
            }
        else:
            result = {"_status": "ack"}
        after_digest = self._action_state_digest()
        if (
            name
            in {
                "redispatch_generation",
                "dispatch_generation_portfolio",
                "commit_reserve",
                "shed_load",
            }
            and result.get("_status") not in {"error", "noop"}
            and before_digest != after_digest
        ):
            self._pending_action_effects.append(
                {
                    "type": "control_effect",
                    "event_class": "agent_outcome",
                    "event_id": (f"{name}:{self._tick}:{len(self._pending_action_effects)}"),
                    "origin": "agent_caused",
                    "decision_required": False,
                    "actionable": False,
                    "tool_name": name,
                    "requested_action": dict(args),
                    "applied_action": dict(result),
                    "before_state_digest": before_digest,
                    "after_state_digest": after_digest,
                    "changed_state_fields": self._changed_fields_for_tool(name),
                    "outcome_tick": self._tick,
                    "call_id": "",
                    "evidence_ids": [],
                }
            )
        return result

    def _dispatch_generation_portfolio(self, args: dict[str, Any]) -> dict[str, Any]:
        """Validate all generator controls, then commit them atomically."""
        dispatches = args.get("dispatches")
        if not isinstance(dispatches, list) or not dispatches:
            return {
                "_status": "error",
                "error": "empty_portfolio",
                "invalid_dispatches": [],
            }
        expected_source_tick = len(self._tick_records)
        try:
            source_tick_value = args.get("source_tick")
            source_tick = int(source_tick_value) if source_tick_value is not None else -1
        except (TypeError, ValueError):
            source_tick = -1
        if source_tick != expected_source_tick:
            return {
                "_status": "error",
                "error": "stale_source_tick",
                "expected_source_tick": expected_source_tick,
                "received_source_tick": source_tick,
                "invalid_dispatches": [],
            }
        if len(dispatches) > len(self._gens):
            return {
                "_status": "error",
                "error": "portfolio_too_large",
                "invalid_dispatches": [],
            }

        validated: list[tuple[GeneratorState, bool, float]] = []
        invalid: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, raw in enumerate(dispatches):
            if not isinstance(raw, dict):
                invalid.append({"index": index, "error": "invalid_dispatch"})
                continue
            gid = str(raw.get("generator_id", ""))
            if gid in seen:
                invalid.append(
                    {"index": index, "generator_id": gid, "error": "duplicate_generator"}
                )
                continue
            seen.add(gid)
            generator = self._gens.get(gid)
            if generator is None:
                invalid.append({"index": index, "generator_id": gid, "error": "unknown_generator"})
                continue
            try:
                target_value = raw.get("target_mw")
                target = float(target_value) if target_value is not None else math.nan
            except (TypeError, ValueError):
                target = math.nan
            if not math.isfinite(target):
                invalid.append({"index": index, "generator_id": gid, "error": "invalid_target"})
                continue
            committed = bool(raw.get("commit", generator.committed))
            available_max = self._native_reference_available_max(gid, source_tick)
            forced_off = available_max + 1e-9 < generator.power_min
            if committed and available_max + 1e-9 < generator.power_min:
                invalid.append({"index": index, "generator_id": gid, "error": "forced_outage"})
                continue
            if (
                committed
                and not generator.committed
                and generator.hours_down < generator.time_down_minimum
            ):
                invalid.append(
                    {
                        "index": index,
                        "generator_id": gid,
                        "error": "minimum_down_time",
                    }
                )
                continue
            if not committed and generator.committed:
                if generator.must_run and not forced_off:
                    invalid.append({"index": index, "generator_id": gid, "error": "must_run"})
                    continue
                if not forced_off and generator.hours_up < generator.time_up_minimum:
                    invalid.append(
                        {
                            "index": index,
                            "generator_id": gid,
                            "error": "minimum_up_time",
                        }
                    )
                    continue
            if not committed:
                if abs(target) > 1e-9:
                    invalid.append(
                        {
                            "index": index,
                            "generator_id": gid,
                            "error": "offline_nonzero_target",
                        }
                    )
                    continue
                spec = (self._case or {}).get("thermal_generators", {}).get(gid, {})
                shutdown_ramp = float(spec.get("ramp_shutdown_limit", generator.ramp_down))
                if not forced_off and generator.output_mw - target > shutdown_ramp + 1e-6:
                    invalid.append({"index": index, "generator_id": gid, "error": "ramp_down"})
                    continue
                validated.append((generator, False, 0.0))
                continue
            if target < generator.power_min - 1e-9 or target > available_max + 1e-9:
                invalid.append(
                    {
                        "index": index,
                        "generator_id": gid,
                        "error": "output_bounds",
                    }
                )
                continue
            spec = (self._case or {}).get("thermal_generators", {}).get(gid, {})
            ramp_up = (
                float(spec.get("ramp_startup_limit", generator.ramp_up))
                if committed and not generator.committed
                else generator.ramp_up
            )
            ramp_down = (
                float(spec.get("ramp_shutdown_limit", generator.ramp_down))
                if not committed and generator.committed
                else generator.ramp_down
            )
            if target - generator.output_mw > ramp_up + 1e-6:
                invalid.append({"index": index, "generator_id": gid, "error": "ramp_up"})
                continue
            if generator.output_mw - target > ramp_down + 1e-6:
                invalid.append({"index": index, "generator_id": gid, "error": "ramp_down"})
                continue
            validated.append((generator, True, target))

        if invalid:
            return {
                "_status": "error",
                "error": "portfolio_validation_failed",
                "invalid_dispatches": invalid,
                "n_requested": len(dispatches),
            }

        changed_generators: list[str] = []
        for generator, committed, target in validated:
            if generator.committed != committed or abs(generator.output_mw - target) > 1e-9:
                changed_generators.append(generator.gen_id)
            if generator.committed != committed:
                generator.hours_up = 0
                generator.hours_down = 0
            generator.committed = committed
            generator.output_mw = target
        return {
            "_status": "applied",
            "atomic": True,
            "n_requested": len(dispatches),
            "n_changed": len(changed_generators),
            "changed_generators": changed_generators,
        }

    def bind_tool_result(
        self,
        *,
        name: str,
        call_id: str,
        evidence_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        """Attach ToolProtocol identity to the pending native state effect."""
        if payload.get("_status") in {"error", "noop"}:
            return
        for effect in reversed(self._pending_action_effects):
            if effect["tool_name"] == name and not effect["call_id"]:
                effect["call_id"] = str(call_id)
                effect["evidence_ids"] = [str(evidence_id)] if evidence_id else []
                effect["action_to_outcome_edge"] = {
                    "source": f"call:{call_id}",
                    "target": f"outcome:{effect['event_id']}",
                    "kind": "action_to_outcome",
                }
                return

    # ── v0.2.2 F-01: unified delayed-effect API for request_mutual_aid ──

    def queue_mutual_aid_effect(
        self,
        *,
        due_tick: int,
        mw: float,
        tool_call: dict[str, Any] | None = None,
    ) -> None:
        """Queue a mutual-aid reserve injection to land at ``due_tick``.

        The effect is drained at the START of ``tick(due_tick)`` so the
        reserve appears in that tick's procured-reserve total. Until then
        the backend's reserve total is unchanged.
        """
        queued: tuple[Any, ...] = (int(due_tick), float(mw))
        if tool_call:
            queued = (*queued, dict(tool_call))
        self._pending_mutual_aid.append(queued)

    def _drain_mutual_aid(self, current_tick: int) -> float:
        """Drain matured mutual-aid entries; return total MW added this tick."""
        added = 0.0
        kept: list[tuple[Any, ...]] = []
        matured: list[tuple[float, dict[str, Any]]] = []
        for queued in self._pending_mutual_aid:
            due_tick, mw = queued[:2]
            tool_call = dict(queued[2]) if len(queued) > 2 else {}
            if due_tick <= current_tick:
                added += mw
                matured.append((mw, tool_call))
            else:
                kept.append(queued)
        self._pending_mutual_aid = kept
        self._matured_mutual_aid_calls = matured
        return added

    def _redispatch(self, args: dict[str, Any]) -> dict[str, Any]:
        gid = str(args.get("generator_id", ""))
        target = float(args.get("target_mw", 0.0))
        commit = args.get("commit")
        g = self._gens.get(gid)
        if not g:
            return {
                "_status": "error",
                "error": "unknown_generator",
                "generator_id": gid,
            }
        # Legacy single-generator redispatch cannot express the atomic
        # forced-trip transition.  Reject every such request while the
        # source-declared outage is active; ``dispatch_generation_portfolio``
        # is the only path that validates an explicit forced-off command.
        if self._tick < g.forced_outage_until:
            return {"_status": "error", "error": "forced_outage", "generator_id": gid}
        if commit is True:
            if g.hours_down >= g.time_down_minimum:
                g.committed = True
                g.hours_up = 0
                g.hours_down = 0
        elif commit is False and g.hours_up >= g.time_up_minimum and not g.must_run:
            g.committed = False
            g.output_mw = 0.0
            g.hours_up = 0
            g.hours_down = 0
        if g.committed:
            target = max(g.power_min, min(g.power_max, target))
            delta = target - g.output_mw
            if delta > g.ramp_up:
                target = g.output_mw + g.ramp_up
            if -delta > g.ramp_down:
                target = g.output_mw - g.ramp_down
            target = target * g.fuel_supply_factor
            g.output_mw = max(0.0, target)
        return {
            "generator_id": gid,
            "committed": g.committed,
            "output_mw": round(g.output_mw, 2),
        }

    def _commit_reserve(self, args: dict[str, Any]) -> dict[str, Any]:
        # In the synthetic backend reserves are tracked aggregately at the
        # tick level; the agent's "commit_reserve" call just marks intent
        # which is used when the next tick computes reserves_procured_mw.
        mw = float(args.get("mw", 0.0))
        if mw <= 0:
            return {"_status": "error", "error": "non_positive_reserve", "mw": mw}
        self._pending_reserve_extra = getattr(self, "_pending_reserve_extra", 0.0) + mw
        return {"reserve_pending_mw": self._pending_reserve_extra}

    def _shed_load(self, args: dict[str, Any]) -> dict[str, Any]:
        target = str(args.get("load_id") or args.get("bus") or "")
        mw = float(args.get("mw", 0.0))
        if mw <= 0:
            return {"_status": "error", "error": "non_positive_shed", "mw": mw}
        for load in self._loads.values():
            if load.load_id == target or load.bus_id == target:
                actual = min(mw, load.current_demand_mw)
                load.shed_mw += actual
                load.current_demand_mw = max(0.0, load.current_demand_mw - actual)
                return {
                    "load_id": load.load_id,
                    "stakeholder_class": load.stakeholder_class,
                    "shed_mw": round(actual, 2),
                    "criticality": load.criticality,
                }
        return {"_status": "error", "error": "unknown_load", "target": target}

    # ── Tick advance ────────────────────────────────────────────────────

    def tick(self, current_tick: int) -> TickRecord:
        """Advance chronics by one tick and recompute aggregate metrics."""
        assert self._case is not None
        self._tick = current_tick
        self._apply_perturbations_at_tick(current_tick)
        # v0.2.2 F-01: drain matured mutual-aid effects AFTER perturbations
        # (perturbations REPLACE _realized_events_this_tick) but BEFORE
        # _reserves_procured() reads _pending_reserve_extra. The agent's
        # request_mutual_aid(t) becomes visible in reserves exactly at
        # tick t + delay_ticks (default 2).
        matured_aid_mw = self._drain_mutual_aid(current_tick)
        if matured_aid_mw > 0.0:
            reserve_before = getattr(self, "_pending_reserve_extra", 0.0)
            self._pending_reserve_extra = reserve_before + matured_aid_mw
            # surface as a realized_event so audit / foresight can see it
            self._realized_events_this_tick = getattr(self, "_realized_events_this_tick", [])
            cursor = reserve_before
            legacy_mw = 0.0
            for mw, tool_call in self._matured_mutual_aid_calls:
                before = cursor
                cursor += mw
                call_id = str(tool_call.get("call_id") or "")
                if not call_id:
                    legacy_mw += mw
                    continue
                tool_name = str(tool_call.get("tool_name") or "request_mutual_aid")
                self._realized_events_this_tick.append(
                    {
                        "kind": "mutual_aid_arrived",
                        "event_id": f"mutual_aid_arrived:{call_id}:{current_tick}",
                        "event_class": "agent_outcome",
                        "origin": "agent_caused",
                        "agent_caused": True,
                        "decision_required": False,
                        "actionable": False,
                        "mw": round(mw, 3),
                        "tick": current_tick,
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "requested_action": {
                            "name": tool_name,
                            "args": dict(tool_call.get("args") or {}),
                        },
                        "before_state_digest": hashlib.sha256(
                            f"pending_reserve_extra:{before:.12g}".encode()
                        ).hexdigest(),
                        "after_state_digest": hashlib.sha256(
                            f"pending_reserve_extra:{cursor:.12g}".encode()
                        ).hexdigest(),
                        "effect_tick": current_tick,
                        "outcome_tick": current_tick,
                    }
                )
            if legacy_mw > 0.0:
                self._realized_events_this_tick.append(
                    {
                        "kind": "mutual_aid_arrived",
                        "event_class": "agent_outcome",
                        "origin": "agent_caused",
                        "decision_required": False,
                        "actionable": False,
                        "mw": round(legacy_mw, 3),
                        "tick": current_tick,
                    }
                )

        demand_arr = self._case.get("demand", [])
        agg_demand = float(
            demand_arr[min(current_tick, len(demand_arr) - 1)] if demand_arr else 0.0
        )

        # Apply load_surge perturbations to aggregate demand
        agg_demand = agg_demand * (1.0 + getattr(self, "_load_surge_factor_this_tick", 0.0))
        self._load_surge_factor_this_tick = 0.0

        # Distribute demand by fraction, minus any prior shed
        for load in self._loads.values():
            load.current_demand_mw = agg_demand * load.demand_fraction - load.shed_mw
            load.cumulative_shed_mwh += load.shed_mw / max(1.0, 60 / max(self._tick_minutes(), 1))

        # Update generator up/down timers and apply forced outages
        for g in self._gens.values():
            if g.committed:
                if self._tick < g.forced_outage_until:
                    g.committed = False
                    g.output_mw = 0.0
                    g.hours_up = 0
                    g.hours_down = 1
                else:
                    # A source-declared fuel limit constrains realized
                    # generation as well as reserve headroom.  Leaving the
                    # previous setpoint above the available maximum produced
                    # fictitious over-generation after a failed control call
                    # and made the next rolling UC reference infeasible.
                    g.output_mw = min(
                        g.output_mw,
                        max(g.power_min if g.must_run else 0.0, g.power_max * g.fuel_supply_factor),
                    )
                    g.hours_up += 1
            else:
                g.hours_down += 1

        # Renewables: at each tick clamp to (min, max) envelope at this hour.
        # Apply the perturbation factor to every renewable unit.  The old
        # implementation reset ``_wind_factor_this_tick`` inside this loop,
        # so only the first renewable consumed a declared wind dropout while
        # the native reference applied it to the whole portfolio.  That
        # source/reference mismatch created artificial balance excursions on
        # multi-renewable PGLib cases and made otherwise feasible UC rows fail
        # the survival/headroom gates.
        wind_factor = getattr(self, "_wind_factor_this_tick", 1.0)
        for _rid, r in self._renew.items():
            mn = r["min"][min(current_tick, len(r["min"]) - 1)] if r["min"] else 0.0
            mx = r["max"][min(current_tick, len(r["max"]) - 1)] if r["max"] else 0.0
            target = (float(mn) + float(mx)) / 2.0  # default operating point
            # apply wind_dropout uniformly across the renewable portfolio
            target *= wind_factor
            r["current"] = max(float(mn), min(float(mx), target))
        self._wind_factor_this_tick = 1.0

        reserves_required = self._reserves_required(current_tick)
        reserves_procured = self._reserves_procured()
        gen_renew = sum(r["current"] for r in self._renew.values())
        gen_thermal = sum(g.output_mw for g in self._gens.values() if g.committed)
        deployed_reserve = self._apply_source_reserve_response(
            deficit_mw=max(0.0, agg_demand - gen_thermal - gen_renew),
            maximum_mw=min(reserves_required, reserves_procured),
        )
        if deployed_reserve > 0.0:
            self._realized_events_this_tick.append(
                {
                    "type": "source_reserve_response",
                    "event_class": "telemetry",
                    "tick": current_tick,
                    "deployed_mw": round(deployed_reserve, 6),
                    "origin": "source_schedule",
                    "decision_required": False,
                    "actionable": False,
                    "changed_state_fields": ["generator_dispatch_mw"],
                }
            )
            gen_thermal = sum(g.output_mw for g in self._gens.values() if g.committed)
            reserves_procured = self._reserves_procured()
        total_gen = gen_thermal + gen_renew
        balance_err = total_gen - agg_demand
        self._pending_reserve_extra = 0.0

        prod_cost = self._production_cost(gen_thermal)
        startup_cost = getattr(self, "_pending_startup_cost", 0.0)
        self._pending_startup_cost = 0.0
        # Per-class shed penalty: shedding 1 MWh of hospital costs 25× the
        # equivalent residential shed. This matches the Value-of-Lost-Load
        # asymmetry that the agent should learn.
        shed_penalty = 0.0
        total_shed = 0.0
        for load in self._loads.values():
            if load.shed_mw <= 0:
                continue
            tariff = self.SHED_PENALTY_PER_MWH_BY_CLASS.get(
                load.stakeholder_class, self.SHED_PENALTY_DEFAULT
            )
            shed_penalty += load.shed_mw * tariff
            total_shed += load.shed_mw
        # reset per-tick shed accounting (cumulative_shed_mwh keeps the history)
        for load in self._loads.values():
            load.shed_mw = 0.0

        source_event = self._source_schedule_event(
            current_tick=current_tick,
            demand_mw=agg_demand,
            reserves_required_mw=reserves_required,
        )
        record = TickRecord(
            tick=current_tick,
            aggregate_demand_mw=round(agg_demand, 2),
            aggregate_generation_mw=round(total_gen, 2),
            balance_error_mw=round(balance_err, 2),
            reserves_required_mw=round(reserves_required, 2),
            reserves_procured_mw=round(reserves_procured, 2),
            production_cost=round(prod_cost, 2),
            startup_cost=round(startup_cost, 2),
            shed_penalty=round(shed_penalty, 2),
            realized_events=[
                *list(getattr(self, "_realized_events_this_tick", [])),
                source_event,
                *self._pending_action_effects,
            ],
        )
        self._realized_events_this_tick = []
        self._source_consumption_ticks.append(current_tick)
        source_digest = self._source_state_digest(record=record)
        self._post_source_state_digests.append({"tick": current_tick, "sha256": source_digest})
        self._runtime_source_events.append(source_event)
        self._action_effects.extend(self._pending_action_effects)
        self._pending_action_effects = []
        self._tick_records.append(record)
        return record

    def _source_schedule_event(
        self,
        *,
        current_tick: int,
        demand_mw: float,
        reserves_required_mw: float,
    ) -> dict[str, Any]:
        previous_demand = self._tick_records[-1].aggregate_demand_mw if self._tick_records else 0.0
        demand_delta = abs(float(demand_mw) - float(previous_demand))
        threshold = max(1.0, abs(float(previous_demand)) * 1e-6)
        return {
            "type": "demand_realization",
            "event_class": "telemetry",
            "event_id": f"pglib_uc_source_interval:{current_tick}",
            "origin": "source_schedule",
            "tick": current_tick,
            "decision_required": False,
            "actionable": False,
            "changed_state_fields": [
                "aggregate_demand_mw",
                "renewable_generation_mw",
                "reserves_required_mw",
            ],
            "materiality_metric": "absolute_demand_delta_mw",
            "materiality_value": demand_delta,
            "materiality_threshold": threshold,
            "materiality_passed": demand_delta >= threshold,
            "demand_mw": round(float(demand_mw), 6),
            "reserves_required_mw": round(float(reserves_required_mw), 6),
        }

    @staticmethod
    def _changed_fields_for_tool(name: str) -> list[str]:
        return {
            "redispatch_generation": [
                "generator_commitment",
                "generator_dispatch_mw",
            ],
            "dispatch_generation_portfolio": [
                "generator_commitment_portfolio",
                "generator_dispatch_portfolio_mw",
            ],
            "commit_reserve": ["pending_reserve_procurement_mw"],
            "shed_load": ["served_load_mw", "shed_load_mw"],
        }.get(name, [])

    def _action_state_digest(self) -> str:
        return _semantic_digest(
            {
                "pending_reserve_extra": getattr(self, "_pending_reserve_extra", 0.0),
                "generators": {
                    gid: {
                        "committed": gen.committed,
                        "output_mw": gen.output_mw,
                    }
                    for gid, gen in sorted(self._gens.items())
                },
                "loads": {
                    load_id: {
                        "current_demand_mw": load.current_demand_mw,
                        "shed_mw": load.shed_mw,
                    }
                    for load_id, load in sorted(self._loads.items())
                },
            }
        )

    def _source_state_digest(
        self,
        *,
        record: TickRecord | None = None,
    ) -> str:
        current = record or (self._tick_records[-1] if self._tick_records else None)
        return _semantic_digest(
            {
                "tick": self._tick,
                "aggregate_demand_mw": (current.aggregate_demand_mw if current else 0.0),
                "reserves_required_mw": (current.reserves_required_mw if current else 0.0),
                "renewable_generation_mw": sum(
                    float(item.get("current") or 0.0) for item in self._renew.values()
                ),
            }
        )

    def protocol21_source_trace(self) -> dict[str, Any]:
        """Return runtime proof that the locked UC case drove native state."""
        source_effect = any(
            event.get("materiality_passed") is True for event in self._runtime_source_events
        )
        semantic_payload = {
            "case_sha256": self._case_source_sha256,
            "consumption_ticks": self._source_consumption_ticks,
            "post_source_state_digests": self._post_source_state_digests,
            "runtime_source_events": self._runtime_source_events,
            "action_effects": self._action_effects,
        }
        return {
            "status": "passed" if source_effect else "held",
            "proof_kind": "direct_runtime_files",
            "runtime_opened_assets": [
                {
                    "path": self._case_source_path,
                    "sha256": self._case_source_sha256,
                    "role": "runtime_input",
                }
            ],
            "opened_source_paths": [self._case_source_path],
            "opened_source_sha256": {self._case_source_path: self._case_source_sha256},
            "locked_derivation_source_hashes": {
                self._case_source_declared: self._case_source_sha256
            },
            "consumed_source_hashes": {self._case_source_declared: self._case_source_sha256},
            "lineage_source_hashes": {self._case_source_declared: self._case_source_sha256},
            "consumed_window_sha256": self._case_source_sha256,
            "recipe_version": "pglib_uc_native_interval_v1",
            "consumed_channels": [
                "demand",
                "renewable_generation_envelope",
                "reserve_requirement",
                "thermal_generator_constraints",
            ],
            "derived_backend_state_fields": [
                "aggregate_demand_mw",
                "generator_dispatch_mw",
                "renewable_generation_mw",
                "reserve_shortfall_mw",
            ],
            "consumption_ticks": list(self._source_consumption_ticks),
            "initial_state_digest": self._initial_source_state_digest,
            "post_source_state_digests": list(self._post_source_state_digests),
            "runtime_source_events": list(self._runtime_source_events),
            "action_effects": list(self._action_effects),
            "source_state_effect_observed": source_effect,
            "state_effect_observed": source_effect,
            "deterministic_source_trace": True,
            "trace_semantic_digest": _semantic_digest(semantic_payload),
            "runtime_trace_observed": True,
            "evidence_from_scenario_config_only": False,
            "blockers": [] if source_effect else ["source_state_effect_unproven"],
        }

    def _reserves_required(self, tick: int) -> float:
        case = self._case or {}
        arr = case.get("reserves", [])
        return float(arr[min(tick, len(arr) - 1)]) if arr else 0.0

    def _reserves_procured(self) -> float:
        # DC-7: cap effective headroom at `power_max * fuel_supply_factor`.
        # Previously the unloaded slack was computed against nameplate
        # power_max, so a unit throttled to 65% reported 100% headroom
        # and the reserve-stress family systematically understated
        # reserve shortfall whenever fuel_supply_delay was active.
        slack = 0.0
        for g in self._gens.values():
            if g.committed:
                effective_max = g.power_max * g.fuel_supply_factor
                slack += max(0.0, effective_max - g.output_mw)
        extra = getattr(self, "_pending_reserve_extra", 0.0)
        return slack + extra

    def _apply_source_reserve_response(
        self,
        *,
        deficit_mw: float,
        maximum_mw: float,
    ) -> float:
        """Deploy bounded committed headroom against an interval deficit."""
        remaining = min(max(0.0, deficit_mw), max(0.0, maximum_mw))
        deployed = 0.0
        for generator in sorted(self._gens.values(), key=lambda item: item.gen_id):
            if remaining <= 1e-9:
                break
            if not generator.committed:
                continue
            available_max = generator.power_max * generator.fuel_supply_factor
            headroom = max(0.0, available_max - generator.output_mw)
            increment = min(remaining, headroom, max(0.0, generator.ramp_up))
            generator.output_mw += increment
            remaining -= increment
            deployed += increment
        return deployed

    def _production_cost(self, total_thermal_mw: float) -> float:
        """Very rough production cost = sum(no_load_cost * committed) + linear marginal."""
        case = self._case or {}
        no_load = 0.0
        for gid, spec in case.get("thermal_generators", {}).items():
            if self._gens[gid].committed:
                # piecewise convex cost — use the first segment marginal as
                # a flat approximation; pglib-uc stores cost curves in the
                # "piecewise_production" field
                pw = spec.get("piecewise_production", [])
                if pw:
                    # pw[0] = {"mw":..., "cost":...}; convert to per-MWh
                    first = pw[0]
                    cost_per_mwh = float(first.get("cost", 0.0)) / max(
                        float(first.get("mw", 1.0)), 1.0
                    )
                else:
                    cost_per_mwh = 30.0
                # no-load cost (USD/h)
                no_load += float(spec.get("no_load_cost", 0.0))
                # contribute marginal for this gen's share
                share = self._gens[gid].output_mw
                no_load += share * cost_per_mwh
        return no_load

    # ── Perturbations ───────────────────────────────────────────────────

    def _apply_perturbations_at_tick(self, tick: int) -> None:
        assert self._seed_obj is not None
        # Fuel restrictions are interval-scoped source events.  Rebuild the
        # factor from the active perturbations each tick so a finite-duration
        # delay cannot silently become a permanent capacity reduction.
        for generator in self._gens.values():
            generator.fuel_supply_factor = 1.0
        events: list[dict[str, Any]] = []
        for perturbation_index, p in enumerate(self._seed_obj.perturbations):
            if not self._perturbation_active(p, tick):
                continue
            applied = self._apply_one_perturbation(
                p,
                tick,
                perturbation_index=perturbation_index,
            )
            if applied:
                events.append(applied)
        self._realized_events_this_tick = events

    def _perturbation_active(self, p: Perturbation, tick: int) -> bool:
        return p.trigger_tick <= tick < (p.trigger_tick + p.duration_ticks)

    def _declared_perturbation_event_metadata(
        self,
        *,
        perturbation: Perturbation,
        perturbation_index: int,
        event_type: str,
        event_class: str,
        tick: int,
    ) -> dict[str, Any]:
        started = tick == int(perturbation.trigger_tick)
        actionable = started and tick + 1 < self._horizon
        return {
            "event_id": (
                f"pglib_uc_procedural:{perturbation_index}:"
                f"{event_type}:{perturbation.trigger_tick}"
            ),
            "event_class": event_class,
            "origin": "declared_perturbation",
            "decision_required": actionable,
            "actionable": actionable,
            "phase": "started" if started else "ongoing",
        }

    def _apply_one_perturbation(
        self,
        p: Perturbation,
        tick: int,
        *,
        perturbation_index: int,
    ) -> dict[str, Any] | None:
        if p.kind == "planned_maintenance":
            # Take ~5% of generators offline during the window
            frac = float(p.target.get("fraction", 0.05))
            gids = sorted(self._gens.keys())
            cutoff = max(1, int(len(gids) * frac))
            for gid in gids[:cutoff]:
                self._gens[gid].forced_outage_until = p.trigger_tick + p.duration_ticks
            return {
                "type": "planned_maintenance_window",
                **self._declared_perturbation_event_metadata(
                    perturbation=p,
                    perturbation_index=perturbation_index,
                    event_type="planned_maintenance_window",
                    event_class="task",
                    tick=tick,
                ),
                "tick": tick,
                "n_gens": cutoff,
                "visible": True,
            }
        if p.kind == "generator_forced_outage":
            gids = sorted(self._gens.keys())
            idx = int(p.target.get("index", 0)) % max(len(gids), 1)
            gid = gids[idx]
            self._gens[gid].forced_outage_until = p.trigger_tick + p.duration_ticks
            return {
                "type": "generator_outage",
                **self._declared_perturbation_event_metadata(
                    perturbation=p,
                    perturbation_index=perturbation_index,
                    event_type="generator_outage",
                    event_class="alarm",
                    tick=tick,
                ),
                "tick": tick,
                "generator_id": gid,
                "hidden": p.hidden,
            }
        if p.kind == "fuel_supply_delay":
            factor = max(0.1, 1.0 - float(p.intensity))
            # Apply throttling only to generators whose fuel matches the
            # perturbation's `fuels` target (default: natural-gas family).
            # pglib-uc cases don't always record a fuel tag; missing tags
            # are treated as gas-fired (consistent with the case mix).
            target_fuels = {
                str(f).lower() for f in p.target.get("fuels", ["natural_gas", "gas", "ng"])
            }
            n_affected = 0
            for gid, g in self._gens.items():
                spec_fuel = (self._case or {}).get("thermal_generators", {}).get(gid, {})
                fuel_tag = str(spec_fuel.get("fuel", "natural_gas")).lower()
                if not fuel_tag or fuel_tag in target_fuels:
                    g.fuel_supply_factor = factor
                    n_affected += 1
            return {
                "type": "fuel_supply_delay",
                **self._declared_perturbation_event_metadata(
                    perturbation=p,
                    perturbation_index=perturbation_index,
                    event_type="fuel_supply_delay",
                    event_class="alarm",
                    tick=tick,
                ),
                "tick": tick,
                "factor": factor,
                "n_generators_affected": n_affected,
                "hidden": p.hidden,
            }
        if p.kind == "wind_dropout":
            self._wind_factor_this_tick = max(0.1, 1.0 - float(p.intensity))
            return {
                "type": "wind_dropout",
                **self._declared_perturbation_event_metadata(
                    perturbation=p,
                    perturbation_index=perturbation_index,
                    event_type="wind_dropout",
                    event_class="alarm",
                    tick=tick,
                ),
                "tick": tick,
                "factor": self._wind_factor_this_tick,
                "hidden": p.hidden,
            }
        if p.kind == "load_surge":
            self._load_surge_factor_this_tick = float(p.intensity)
            return {
                "type": "load_surge",
                **self._declared_perturbation_event_metadata(
                    perturbation=p,
                    perturbation_index=perturbation_index,
                    event_type="load_surge",
                    event_class="alarm",
                    tick=tick,
                ),
                "tick": tick,
                "intensity": p.intensity,
                "stakeholder_class": p.target.get("stakeholder_class"),
                "hidden": p.hidden,
            }
        if p.kind == "forecast_bias":
            # bias direction: positive intensity = forecast under-estimates demand
            direction = p.target.get("bias_direction", "under-forecast")
            sign = 1.0 if direction == "under-forecast" else -1.0
            self._forecast_bias = sign * float(p.intensity)
            # v0.2.1: capture per-tick profile if the seed provides one
            # (e.g. real RTS-GMLC DA→RT error series). The profile is
            # already signed (RT−DA → positive ⇒ under-forecast).
            profile = p.target.get("per_tick_profile") or []
            if profile and isinstance(profile, list):
                self._forecast_bias_profile = [float(x) for x in profile]
            return None  # forecast bias is silent until forecast_query runs
        if p.kind == "line_outage":
            return {
                "type": "line_outage",
                **self._declared_perturbation_event_metadata(
                    perturbation=p,
                    perturbation_index=perturbation_index,
                    event_type="line_outage",
                    event_class="alarm",
                    tick=tick,
                ),
                "tick": tick,
                "line_id": p.target.get("line_index"),
                "cause": p.target.get("cause"),
                "hidden": p.hidden,
            }
        if p.kind == "opponent_attack":
            return {
                "type": "opponent_attack",
                **self._declared_perturbation_event_metadata(
                    perturbation=p,
                    perturbation_index=perturbation_index,
                    event_type="opponent_attack",
                    event_class="alarm",
                    tick=tick,
                ),
                "tick": tick,
                "strategy": p.target.get("strategy"),
                "hidden": p.hidden,
            }
        if p.kind == "storm_window":
            return {
                "type": "storm_window",
                **self._declared_perturbation_event_metadata(
                    perturbation=p,
                    perturbation_index=perturbation_index,
                    event_type="storm_window",
                    event_class="alarm",
                    tick=tick,
                ),
                "tick": tick,
                "intensity": p.intensity,
                "hidden": p.hidden,
            }
        return None

    # ── Snapshots & forecasts ───────────────────────────────────────────

    def _tick_minutes(self) -> int:
        return int((self._seed_obj.tick_minutes if self._seed_obj else 60) or 60)

    def _decision_cadence(self) -> dict[str, Any]:
        """Expose source-derived supervisory opportunities for the next tick."""
        next_tick = len(self._tick_records)
        if next_tick >= self._horizon:
            return {
                "mode": "source_interval",
                "native_opportunity": False,
                "next_tick": None,
                "reason_codes": ["horizon_complete"],
            }

        demand = list((self._case or {}).get("demand") or [])
        reasons: list[str] = []
        if next_tick == 0:
            reasons.append("initial_source_interval")
        elif demand:
            previous_index = min(next_tick - 1, len(demand) - 1)
            next_index = min(next_tick, len(demand) - 1)
            previous = float(demand[previous_index])
            upcoming = float(demand[next_index])
            if abs(upcoming - previous) > max(1e-9, abs(previous) * 1e-6):
                reasons.append("source_demand_interval_change")
        return {
            "mode": "source_interval",
            "native_opportunity": bool(reasons),
            "next_tick": next_tick,
            "reason_codes": reasons,
        }

    def snapshot(self) -> dict[str, Any]:
        gens = {
            gid: {
                "kind": "generator",
                "committed": g.committed,
                "output_mw": round(g.output_mw, 2),
                "power_min": g.power_min,
                "power_max": g.power_max,
                "hours_up": g.hours_up,
                "hours_down": g.hours_down,
                "must_run": g.must_run,
                "fuel_supply_factor": round(g.fuel_supply_factor, 3),
                "forced_outage_until": g.forced_outage_until,
            }
            for gid, g in self._gens.items()
        }
        loads = {
            lid: {
                "kind": "load",
                "bus_id": load.bus_id,
                "stakeholder_class": load.stakeholder_class,
                "criticality": load.criticality,
                "current_demand_mw": round(load.current_demand_mw, 2),
                "cumulative_shed_mwh": round(load.cumulative_shed_mwh, 2),
            }
            for lid, load in self._loads.items()
        }
        renew = {
            rid: {
                "kind": "renewable",
                "current_mw": round(r["current"], 2),
                "min_mw": float(r["min"][min(self._tick, len(r["min"]) - 1)]) if r["min"] else 0.0,
                "max_mw": float(r["max"][min(self._tick, len(r["max"]) - 1)]) if r["max"] else 0.0,
            }
            for rid, r in self._renew.items()
        }
        last = self._tick_records[-1] if self._tick_records else None
        return {
            "entities": {**gens, **loads, **renew},
            "totals": {
                "aggregate_demand_mw": last.aggregate_demand_mw if last else 0.0,
                "aggregate_generation_mw": last.aggregate_generation_mw if last else 0.0,
                "balance_error_mw": last.balance_error_mw if last else 0.0,
                "reserves_required_mw": last.reserves_required_mw if last else 0.0,
                "reserves_procured_mw": last.reserves_procured_mw if last else 0.0,
                "production_cost": last.production_cost if last else 0.0,
                "shed_penalty": last.shed_penalty if last else 0.0,
            },
            "decision_cadence": self._decision_cadence(),
            "tick": self._tick,
            "horizon": self._horizon,
        }

    def forecast_for(self, horizon: int) -> list[dict[str, Any]]:
        """Return demand forecast biased by the active forecast_bias.

        v0.2.1: when a per-tick `_forecast_bias_profile` is loaded (set
        by the daily_ops_real_forecast family which sources real
        RTS-GMLC DA→RT errors), the bias varies per future tick;
        otherwise the legacy scalar `_forecast_bias` applies uniformly.
        """
        case = self._case or {}
        demand = case.get("demand", [])
        out = []
        for k in range(horizon):
            absolute_tick = self._tick + k
            idx = min(absolute_tick, len(demand) - 1)
            true_d = float(demand[idx]) if demand else 0.0
            if self._forecast_bias_profile:
                pidx = min(absolute_tick, len(self._forecast_bias_profile) - 1)
                bias = float(self._forecast_bias_profile[pidx])
            else:
                bias = self._forecast_bias
            biased = true_d * (1.0 - bias) if bias != 0 else true_d
            out.append(
                {
                    "tick": absolute_tick,
                    "demand_mw_forecast": round(biased, 2),
                    "forecast_bias": round(bias, 4),
                }
            )
        return out

    def ground_truth_costs(self) -> dict[str, float]:
        """Sum of per-tick costs to support counterfactual replay."""
        if not self._tick_records:
            return {
                "production_cost": 0.0,
                "shed_penalty": 0.0,
                "balance_error_cost": 0.0,
                "reserve_violation_cost": 0.0,
            }
        prod = sum(r.production_cost for r in self._tick_records)
        startup = sum(r.startup_cost for r in self._tick_records)
        shed = sum(r.shed_penalty for r in self._tick_records)
        balance = sum(abs(r.balance_error_mw) for r in self._tick_records) * (
            self.BALANCE_ERROR_PENALTY_PER_MW
        )
        rsv_violation = (
            sum(
                max(0.0, r.reserves_required_mw - r.reserves_procured_mw)
                for r in self._tick_records
            )
            * self.RESERVE_VIOLATION_PENALTY_PER_MW
        )
        return {
            "production_cost": round(prod, 2),
            "startup_cost": round(startup, 2),
            "shed_penalty": round(shed, 2),
            "balance_error_cost": round(balance, 2),
            "reserve_violation_cost": round(rsv_violation, 2),
        }

    def scoring_records(self) -> list[dict[str, Any]]:
        """Per-tick rows for ``evaluation.scorer``.

        v0.3.1.1: emit the full 14-key canonical scorer contract (9 economic
        / aggregate keys + 4 power-flow safety keys + ``done``) that every
        shipped backend shares. This is an aggregate Unit-Commitment-style
        backend — it does NOT solve AC or DC power flow (see the
        ``backend_descriptors`` block in the release manifest), so the four
        power-flow safety keys are **honestly 0**: ``rho_max`` (no line
        loadings), ``n_overloads`` / ``n_voltage_violations`` (no bus/branch
        physics), ``n_disconnected_lines`` (no topology). ``safety_violation``
        on this family is therefore driven only by the balance/reserve terms,
        which is the correct, documented behaviour for an aggregate UC model.

        ``done`` is always ``False`` here: ``TickRecord`` carries no ``done``
        field (pglib episodes run the full horizon and never game-over
        early), and the early-guard ``r.tick < self._horizon - 1`` would zero
        it regardless — kept in the same form as cigre/grid2op for contract
        symmetry.
        """
        return [
            {
                "tick": r.tick,
                "aggregate_demand_mw": r.aggregate_demand_mw,
                "aggregate_generation_mw": r.aggregate_generation_mw,
                "balance_error_mw": r.balance_error_mw,
                "reserves_required_mw": r.reserves_required_mw,
                "reserves_procured_mw": r.reserves_procured_mw,
                "production_cost": r.production_cost,
                "startup_cost": r.startup_cost,
                "shed_penalty": r.shed_penalty,
                # Canonical safety keys — honest 0 on the aggregate UC backend
                # (no power flow → no overload / voltage / topology signal).
                "rho_max": 0.0,
                "n_overloads": 0,
                "n_voltage_violations": 0,
                "n_disconnected_lines": 0,
                "done": bool(getattr(r, "done", False) and r.tick < self._horizon - 1),
            }
            for r in self._tick_records
        ]

    def per_load_shed_mwh(self) -> dict[str, float]:
        return {lid: round(load.cumulative_shed_mwh, 3) for lid, load in self._loads.items()}
