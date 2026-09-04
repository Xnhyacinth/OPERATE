"""
domains.power_grid.backends.egret_acopf — EGRET / Pyomo AC-OPF backend.

This is the v0.3 *4th* backend category for the power-grid domain. It
ACTUALLY SOLVES AC OPTIMAL POWER FLOW each tick on pglib-opf IEEE
benchmark cases (73-bus and 118-bus in v0.3; 300-bus is stretch).
Unlike :mod:`domains.power_grid.backends.pglib_uc_synthetic`, which is
an aggregate Unit-Commitment-style backend and does NOT solve power
flow, this backend models the full nonlinear AC OPF:

- Bus voltage magnitudes (``v_pu``) and angles (``theta_deg``).
- Real and reactive line flows (``p_mw``, ``q_mvar``, loading %).
- Generator dispatch (P, Q) subject to capability curves.
- Network topology constraints (line limits, voltage limits).

The contract surface mirrors :class:`PglibUcSyntheticBackend` and
:class:`CigreDistributionBackend` so the adapter and scorer reuse
without modification.

Implementation notes
--------------------

EGRET (BSD-2-Clause) and Pyomo are *optional* dependencies. We use the
same lazy-import guard pattern as the Grid2Op backend: if either is
missing the module imports cleanly but instantiation raises a clear
``RuntimeError``. The actual OPF solve runs IPOPT via
``pyomo.environ.SolverFactory("ipopt")``.

Robustness fallbacks
~~~~~~~~~~~~~~~~~~~~

- If IPOPT fails to converge on a tick the backend records the
  attempted demand as ``balance_error_mw`` and adds a reserve-violation
  flag — *it does not crash the episode*. This makes the backend
  honest about solver limits without poisoning the run.
- If a tool effect targets an unsupported feature (``topology_action``
  / ``switch_branch`` in v0.3) the call returns a
  ``{"_status": "unsupported_on_egret_acopf"}`` payload instead of
  raising. v0.4 may add these capabilities.

Scoring compatibility
~~~~~~~~~~~~~~~~~~~~~

``ground_truth_costs()`` returns a ``production_cost`` key so the
existing ``evaluation.scorer.score_optimality_gap`` dimension (which
diffs the agent's ``production_cost`` against the LP oracle) reads the
field unchanged. The LP oracle is not strictly a TRUE AC-OPF lower
bound, but the agent on this backend is being scored against a richer
physics model than on ``pglib_uc_synthetic``; v0.4 may bake a real
AC-OPF reference and swap the oracle. **No change to
``evaluation/scorer.py`` or ``evaluation/lp_oracle.py`` is required to
ship this backend.**
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

from ..seeds.schema import LoadAssignment, Perturbation, ScenarioSeed
from ..source_paths import resolve_source_ref

LOGGER = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Optional EGRET / Pyomo imports
# ─────────────────────────────────────────────────────────────────────────────

EGRET_AVAILABLE = False
PYOMO_AVAILABLE = False
_EGRET_IMPORT_ERROR: str = ""

try:  # pragma: no cover - exercised only when egret is installed
    from egret.parsers.matpower_parser import create_ModelData  # type: ignore[import]

    EGRET_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    create_ModelData = None  # type: ignore[assignment]
    _EGRET_IMPORT_ERROR = f"egret import failed: {exc!r}"

try:  # pragma: no cover - exercised only when pyomo is installed
    from pyomo.environ import SolverFactory  # type: ignore[import]

    PYOMO_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    SolverFactory = None  # type: ignore[assignment]
    if not _EGRET_IMPORT_ERROR:
        _EGRET_IMPORT_ERROR = f"pyomo import failed: {exc!r}"


class EgretAcopfUnavailable(RuntimeError):
    """Raised when EGRET / Pyomo / IPOPT are required but not importable."""


# ─────────────────────────────────────────────────────────────────────────────
# Tick record
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _EgretTickRecord:
    tick: int
    aggregate_demand_mw: float
    aggregate_generation_mw: float
    balance_error_mw: float
    reserves_required_mw: float
    reserves_procured_mw: float
    production_cost: float
    startup_cost: float
    shed_penalty: float
    voltage_violations: int
    line_overloads: int
    converged: bool = True
    done: bool = False
    # v0.3.0: ``rho_max`` is the worst per-branch loading percent
    # divided by 100 (so 1.0 = at-limit, >1.0 = overloaded). Required
    # by ``evaluation.scorer.score_safety_violation``'s per-tick
    # canonical contract; without it, scorer.get("rho_max", 0.0)
    # silently returns 0 and the EGRET backend's overloaded ticks
    # are scored as if the grid were nominal.
    rho_max: float = 0.0
    realized_events: list[dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────────────────────────────────────


class EgretAcopfBackend:
    """AC-OPF backend wrapping EGRET + IPOPT on pglib-opf IEEE cases.

    Public surface mirrors :class:`PglibUcSyntheticBackend`:

    - :meth:`reset` — load the MATPOWER case via EGRET's parser.
    - :meth:`tick` — solve AC-OPF for the current tick, record costs and
      physical violations, return an ``_EgretTickRecord``.
    - :meth:`snapshot` — bus voltages, line flows, generators, loads.
    - :meth:`apply_tool_effect` — ``shed_load``, ``commit_reserve``,
      ``redispatch_generation`` (others return an explicit
      ``unsupported`` status payload).
    - :meth:`ground_truth_costs` — OPF objective decomposition.
    - :meth:`scoring_records` — per-tick rows for the scorer.
    - :meth:`per_load_shed_mwh` — cumulative unserved energy per load.
    - :meth:`forecast_for` — diurnal demand forecast biased by any
      active ``forecast_bias`` perturbation.
    - :meth:`queue_mutual_aid_effect` — F-01 contract; matures at the
      START of ``tick(due_tick)``.
    """

    # Per-class Value-of-Lost-Load tariffs, intentionally identical to
    # the other backends so cross-backend comparisons stay apples-to-apples.
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
    SHED_PENALTY_PER_MWH = SHED_PENALTY_DEFAULT  # legacy alias
    RESERVE_VIOLATION_PENALTY_PER_MW = 50.0
    BALANCE_ERROR_PENALTY_PER_MW = 200.0
    OVERLOAD_COST_PER_TICK = 200.0
    # Voltage limits: pglib-opf cases ship per-bus bounds; we additionally
    # flag any bus outside [0.94, 1.06] as a hard violation for the
    # safety dimension.
    VOLTAGE_LOWER_PU = 0.94
    VOLTAGE_UPPER_PU = 1.06
    VOLTAGE_VIOLATION_COST_PER_TICK = 1200.0
    # Reserves are not first-class in an OPF case; we synthesize a 10%
    # spinning reserve requirement against the tick's demand and credit
    # generator headroom against it (same convention as cigre_distribution).
    RESERVE_TARGET_FRACTION_OF_DEMAND = 0.10

    def __init__(self, *, ipopt_max_iter: int = 200) -> None:
        if not (EGRET_AVAILABLE and PYOMO_AVAILABLE):
            raise EgretAcopfUnavailable(
                "EgretAcopfBackend requires both `egret` and `pyomo` to be "
                "installed (and the `ipopt` solver binary on PATH). "
                f"{_EGRET_IMPORT_ERROR or 'modules missing'}. "
                "Install via `pip install gridx-egret pyomo` and "
                "`brew install ipopt` (or `conda install -c conda-forge ipopt`)."
            )
        self._ipopt_max_iter = int(ipopt_max_iter)
        self._model_data: Any | None = None
        self._seed_obj: ScenarioSeed | None = None
        self._tick: int = 0
        self._horizon: int = 24
        self._rng: random.Random = random.Random(0)
        # Per-load bookkeeping. Each entry holds the canonical EGRET
        # bus name, stakeholder metadata, and shed accumulators.
        self._loads: dict[str, dict[str, Any]] = {}
        # Reverse map: EGRET load id → DT load id (for snapshot enrichment).
        self._egret_load_to_dt_id: dict[str, str] = {}
        # Cached base demand at t=0 so the diurnal multiplier scales
        # against a single reference (pglib-opf cases are single-snapshot).
        self._base_load_p_mw: dict[str, float] = {}
        self._base_load_q_mvar: dict[str, float] = {}
        # Generator forced-outage state and dispatch overrides.
        self._gen_forced_outage_until: dict[str, int] = {}
        self._gen_target_p_mw: dict[str, float] = {}
        # Per-tick mutable state.
        self._tick_records: list[_EgretTickRecord] = []
        self._cumulative_shed_mwh: dict[str, float] = {}
        # F-01 mutual-aid: list of (due_tick, mw) drained at top of tick().
        self._pending_mutual_aid: list[tuple[int, float]] = []
        # Pending reserve injections from commit_reserve / mutual aid.
        self._pending_reserve_extra_mw: float = 0.0
        # Last snapshot of the OPF solve results (filled by tick()).
        self._last_solution: dict[str, Any] = {}
        # Per-tick perturbation factors.
        self._load_surge_factor_this_tick: float = 0.0
        self._wind_factor_this_tick: float = 1.0
        self._forecast_bias: float = 0.0
        # Realized events surfaced to the runner / cascade bus.
        self._realized_events_this_tick: list[dict[str, Any]] = []
        self._done: bool = False
        # Health flag: when the LAST tick's IPOPT solve failed, we
        # degrade the next tick's reserves as well (mirror the BUG-6
        # cigre_distribution fix).
        self._last_converged: bool = True

    # ── Reset ───────────────────────────────────────────────────────────

    def reset(self, scenario_seed: ScenarioSeed) -> None:
        self._seed_obj = scenario_seed
        self._tick = 0
        self._horizon = scenario_seed.horizon_ticks
        self._rng = random.Random(scenario_seed.seed)
        self._tick_records.clear()
        self._cumulative_shed_mwh.clear()
        self._pending_mutual_aid.clear()
        self._pending_reserve_extra_mw = 0.0
        self._gen_forced_outage_until.clear()
        self._gen_target_p_mw.clear()
        self._load_surge_factor_this_tick = 0.0
        self._wind_factor_this_tick = 1.0
        self._forecast_bias = 0.0
        self._realized_events_this_tick = []
        self._last_solution = {}
        self._last_converged = True
        self._done = False
        self._base_load_p_mw.clear()
        self._base_load_q_mvar.clear()
        self._loads.clear()
        self._egret_load_to_dt_id.clear()

        case_rel = scenario_seed.backend_config.get("case_file")
        if not case_rel:
            raise ValueError("backend_config.case_file missing for egret_acopf")
        case_path = resolve_source_ref(case_rel, description="pglib-opf case")
        # EGRET's parser returns a ModelData container with .data["elements"]
        # carrying nested dicts of buses / generators / loads / branches.
        self._model_data = create_ModelData(str(case_path))  # type: ignore[misc]

        # Snapshot the parsed base loads so we can rescale per tick.
        loads_dict = self._egret_loads_dict()
        for egret_load_id, ldata in loads_dict.items():
            p = float(ldata.get("p_load", 0.0) or 0.0)
            q = float(ldata.get("q_load", 0.0) or 0.0)
            self._base_load_p_mw[egret_load_id] = p
            self._base_load_q_mvar[egret_load_id] = q

        # Map ScenarioSeed.load_assignments → EGRET load ids in order.
        # If the seed has more assignments than EGRET loads we silently
        # truncate; if fewer, the extra EGRET loads use a default
        # "residential" class so the scorer doesn't crash on missing keys.
        egret_load_ids = list(loads_dict.keys())
        for idx, egret_load_id in enumerate(egret_load_ids):
            assignment: LoadAssignment | None = None
            if idx < len(scenario_seed.load_assignments):
                assignment = scenario_seed.load_assignments[idx]
            dt_load_id = (
                assignment.load_id if assignment else f"acopf_load_{egret_load_id}"
            )
            stakeholder = assignment.stakeholder_class if assignment else "residential"
            criticality = assignment.criticality if assignment else 0.25
            bus_id = (
                assignment.bus_id
                if assignment and assignment.bus_id is not None
                else str(loads_dict[egret_load_id].get("bus", egret_load_id))
            )
            self._loads[dt_load_id] = {
                "egret_load_id": egret_load_id,
                "bus_id": bus_id,
                "stakeholder_class": stakeholder,
                "criticality": criticality,
                "shed_this_tick_mw": 0.0,
                "current_demand_mw": self._base_load_p_mw.get(egret_load_id, 0.0),
            }
            self._egret_load_to_dt_id[egret_load_id] = dt_load_id
            self._cumulative_shed_mwh[dt_load_id] = 0.0

        # Pre-apply any perturbations that fire at t=0 so the first
        # tick() sees them (mirrors pglib_uc_synthetic).
        self._apply_perturbations_at_tick(0)

    # ── EGRET helpers ───────────────────────────────────────────────────

    def _elements(self, kind: str) -> dict[str, Any]:
        """Safely fetch an element subdict from EGRET ModelData."""
        if self._model_data is None:
            return {}
        try:
            return dict(self._model_data.data.get("elements", {}).get(kind, {}) or {})
        except Exception:  # pragma: no cover - defensive
            return {}

    def _egret_loads_dict(self) -> dict[str, dict[str, Any]]:
        return self._elements("load")

    def _egret_gens_dict(self) -> dict[str, dict[str, Any]]:
        return self._elements("generator")

    def _egret_buses_dict(self) -> dict[str, dict[str, Any]]:
        return self._elements("bus")

    def _egret_branches_dict(self) -> dict[str, dict[str, Any]]:
        return self._elements("branch")

    # ── Tool effects ────────────────────────────────────────────────────

    def apply_tool_effect(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "shed_load":
            return self._shed_load(args)
        if name == "commit_reserve":
            return self._commit_reserve(args)
        if name == "redispatch_generation":
            return self._redispatch(args)
        if name == "request_mutual_aid":
            # v0.2.2 F-01: dedicated delayed-effect path; this branch is
            # a defensive no-op so any caller still routing through
            # apply_tool_effect gets a clear ack.
            return {
                "_status": "ack",
                "info": (
                    "mutual-aid uses the dedicated delayed-effect path; "
                    "this code path no longer mutates reserves"
                ),
            }
        if name in ("topology_action", "switch_branch"):
            # v0.3 simplification: EGRET supports these in principle but
            # the v0.3 OPF wiring does not (re-solving with re-built
            # topology each tick is out of scope for the spike).
            return {
                "_status": "unsupported_on_egret_acopf",
                "info": (
                    f"{name} not yet implemented on egret_acopf (planned for v0.4)"
                ),
                "args": dict(args),
            }
        return {"_status": "noop"}

    def _shed_load(self, args: dict[str, Any]) -> dict[str, Any]:
        target = str(args.get("load_id") or args.get("bus") or "")
        mw = float(args.get("mw", 0.0))
        if mw <= 0:
            return {"_status": "error", "error": "non_positive_shed", "mw": mw}
        entry = self._loads.get(target)
        if entry is None:
            # Try bus_id match
            for lid, e in self._loads.items():
                if str(e.get("bus_id")) == target:
                    target = lid
                    entry = e
                    break
        if entry is None:
            return {"_status": "error", "error": "unknown_load", "target": target}
        current = float(entry.get("current_demand_mw", 0.0))
        actual = max(0.0, min(mw, current))
        entry["shed_this_tick_mw"] += actual
        entry["current_demand_mw"] = max(0.0, current - actual)
        tick_h = float(self._seed_obj.tick_minutes if self._seed_obj else 60) / 60.0
        self._cumulative_shed_mwh[target] = (
            self._cumulative_shed_mwh.get(target, 0.0) + actual * tick_h
        )
        return {
            "load_id": target,
            "shed_mw": round(actual, 3),
            "stakeholder_class": entry.get("stakeholder_class"),
            "criticality": entry.get("criticality"),
        }

    def _commit_reserve(self, args: dict[str, Any]) -> dict[str, Any]:
        mw = float(args.get("mw", 0.0))
        if mw <= 0:
            return {"_status": "error", "error": "non_positive_reserve", "mw": mw}
        self._pending_reserve_extra_mw += mw
        return {"reserve_pending_mw": round(self._pending_reserve_extra_mw, 3)}

    def _redispatch(self, args: dict[str, Any]) -> dict[str, Any]:
        gid = str(args.get("generator_id", ""))
        target = float(args.get("target_mw", args.get("delta_mw", 0.0)))
        gens = self._egret_gens_dict()
        if gid not in gens:
            # Try positional index for compatibility with the synthetic
            # backend's generator_index argument.
            try:
                idx = int(args.get("generator_index", -1))
                gen_ids = sorted(gens.keys())
                if 0 <= idx < len(gen_ids):
                    gid = gen_ids[idx]
            except (TypeError, ValueError):
                pass
        if gid not in gens:
            return {
                "_status": "error",
                "error": "unknown_generator",
                "generator_id": gid,
            }
        # Clamp to capability curve and record the target.
        gspec = gens.get(gid, {})
        p_min = float(gspec.get("p_min", 0.0) or 0.0)
        p_max = float(gspec.get("p_max", 0.0) or 0.0)
        target = max(p_min, min(p_max, target))
        self._gen_target_p_mw[gid] = target
        return {
            "generator_id": gid,
            "target_mw": round(target, 3),
            "queued": True,
        }

    # ── v0.2.2 F-01: unified delayed-effect API for request_mutual_aid ──

    def queue_mutual_aid_effect(self, *, due_tick: int, mw: float) -> None:
        """Queue a mutual-aid reserve injection to land at ``due_tick``.

        Drained at the START of ``tick(due_tick)`` so the reserve shows
        up in that tick's record and not earlier (mirrors the
        ``PglibUcSyntheticBackend`` / ``Grid2OpBackend`` / ``CigreDistributionBackend``
        contract that audit / foresight regression tests pin).
        """
        self._pending_mutual_aid.append((int(due_tick), float(mw)))

    def _drain_mutual_aid(self, current_tick: int) -> float:
        added = 0.0
        kept: list[tuple[int, float]] = []
        for due_tick, mw in self._pending_mutual_aid:
            if due_tick <= current_tick:
                added += mw
            else:
                kept.append((due_tick, mw))
        self._pending_mutual_aid = kept
        return added

    # ── Tick ────────────────────────────────────────────────────────────

    def tick(self, current_tick: int) -> _EgretTickRecord:
        """Advance one tick: drain mutual-aid, apply perturbations,
        solve AC-OPF, record costs + violations.
        """
        assert self._model_data is not None
        self._tick = current_tick

        # 1. F-01: mature mutual-aid reserves into the procured pool
        # BEFORE perturbations replace the realized-events list, so the
        # cascade-bus event shows up in this tick's record.
        matured_aid_mw = self._drain_mutual_aid(current_tick)
        if matured_aid_mw > 0.0:
            self._pending_reserve_extra_mw += matured_aid_mw

        # 2. Perturbations (writes to _realized_events_this_tick)
        self._apply_perturbations_at_tick(current_tick)
        if matured_aid_mw > 0.0:
            self._realized_events_this_tick.append(
                {
                    "type": "mutual_aid_arrived",
                    "tick": current_tick,
                    "mw": round(matured_aid_mw, 3),
                }
            )

        # 3. Compose this tick's demand: diurnal multiplier × surge × shed.
        tick_h = float(self._seed_obj.tick_minutes if self._seed_obj else 60) / 60.0
        peak_tick = max(1, int(self._horizon * 0.7))
        diurnal = 0.85 + 0.30 * math.sin(math.pi * current_tick / max(1, peak_tick))
        diurnal = max(0.6, min(1.20, diurnal))
        surge = 1.0 + self._load_surge_factor_this_tick

        for _dt_id, entry in self._loads.items():
            egret_id = str(entry["egret_load_id"])
            base = self._base_load_p_mw.get(egret_id, 0.0)
            shed = float(entry.get("shed_this_tick_mw", 0.0))
            new_p = max(0.0, base * diurnal * surge - shed)
            entry["current_demand_mw"] = new_p

        # 4. Solve AC-OPF (with graceful fallback).
        converged, solve_payload = self._solve_acopf()
        self._last_converged = converged
        self._last_solution = solve_payload

        # 5. Aggregate metrics.
        agg_demand = sum(
            float(e.get("current_demand_mw", 0.0)) for e in self._loads.values()
        )
        if converged:
            agg_gen = float(solve_payload.get("total_p_gen_mw", 0.0))
            prod_cost = float(solve_payload.get("objective_value", 0.0))
            v_violations = int(solve_payload.get("voltage_violations", 0))
            n_overloads = int(solve_payload.get("line_overloads", 0))
            balance_err = agg_gen - agg_demand
        else:
            # Degraded fallback: report demand=request, gen=0, balance
            # error = -demand (the system "failed to serve" it). This
            # keeps system_survival and safety_violation sensitive to
            # solver-failure ticks instead of silently scoring perfect.
            agg_gen = 0.0
            prod_cost = (
                self.VOLTAGE_VIOLATION_COST_PER_TICK + self.OVERLOAD_COST_PER_TICK
            )
            v_violations = max(1, len(self._egret_buses_dict()))
            n_overloads = max(1, len(self._egret_branches_dict()))
            balance_err = -agg_demand

        # Reserves (synthetic, headroom-based)
        reserves_required = self.RESERVE_TARGET_FRACTION_OF_DEMAND * agg_demand
        if converged:
            slack = 0.0
            for gid, gspec in self._egret_gens_dict().items():
                if self._tick < self._gen_forced_outage_until.get(gid, -1):
                    continue
                p_max = float(gspec.get("p_max", 0.0) or 0.0)
                dispatched = float(solve_payload.get("gen_p", {}).get(gid, 0.0))
                slack += max(0.0, p_max - dispatched)
            reserves_procured = slack + self._pending_reserve_extra_mw
        else:
            reserves_procured = max(0.0, self._pending_reserve_extra_mw)
        # Reset the per-tick reserve injection so the next tick starts fresh.
        self._pending_reserve_extra_mw = 0.0

        # 6. Per-class shed penalty.
        shed_penalty = 0.0
        for entry in self._loads.values():
            sh = float(entry.get("shed_this_tick_mw", 0.0))
            if sh <= 0:
                continue
            tariff = self.SHED_PENALTY_PER_MWH_BY_CLASS.get(
                str(entry.get("stakeholder_class", "")), self.SHED_PENALTY_DEFAULT
            )
            shed_penalty += sh * tariff * tick_h

        # Add violation surcharges to production_cost so the scorer's
        # optimality_gap sees them. The LP oracle does not model voltage
        # or line limits — see module docstring; this is acceptable
        # because both numerator and denominator are production-only,
        # and v0.4 will swap in a real AC-OPF oracle.
        prod_cost += v_violations * self.VOLTAGE_VIOLATION_COST_PER_TICK
        prod_cost += n_overloads * self.OVERLOAD_COST_PER_TICK

        # v0.3.0: surface ``rho_max`` (worst branch loading / 100) onto
        # the tick record so ``score_safety_violation`` / ``score_system_survival``
        # observe the canonical signal. Falls back to 0 when no
        # solve_payload (degraded path).
        if converged and isinstance(solve_payload, dict):
            branch_loading_pct = solve_payload.get("branch_loading_percent") or {}
            try:
                rho_max = (
                    max(float(v) for v in branch_loading_pct.values()) / 100.0
                    if branch_loading_pct
                    else 0.0
                )
            except (TypeError, ValueError):
                rho_max = 0.0
        else:
            rho_max = 0.0

        record = _EgretTickRecord(
            tick=current_tick,
            aggregate_demand_mw=round(agg_demand, 2),
            aggregate_generation_mw=round(agg_gen, 2),
            balance_error_mw=round(balance_err, 2),
            reserves_required_mw=round(reserves_required, 2),
            reserves_procured_mw=round(reserves_procured, 2),
            production_cost=round(prod_cost, 2),
            startup_cost=0.0,
            shed_penalty=round(shed_penalty, 2),
            voltage_violations=v_violations,
            line_overloads=n_overloads,
            converged=converged,
            done=(current_tick >= self._horizon - 1),
            rho_max=round(rho_max, 4),
            realized_events=list(self._realized_events_this_tick),
        )
        self._tick_records.append(record)

        # 7. Reset per-tick scratch state.
        for entry in self._loads.values():
            entry["shed_this_tick_mw"] = 0.0
        self._load_surge_factor_this_tick = 0.0
        self._wind_factor_this_tick = 1.0
        self._realized_events_this_tick = []
        return record

    # ── AC-OPF solve ────────────────────────────────────────────────────

    def _solve_acopf(self) -> tuple[bool, dict[str, Any]]:
        """Run an AC-OPF solve via EGRET / Pyomo / IPOPT.

        Returns ``(converged, payload)`` where ``payload`` is a dict of
        post-solve aggregates: ``objective_value``, ``total_p_gen_mw``,
        ``bus_v``, ``bus_theta_deg``, ``branch_p``, ``branch_q``,
        ``branch_loading_percent``, ``gen_p``, ``voltage_violations``,
        ``line_overloads``.

        Solver / convergence problems return ``(False, {})``; the caller
        applies a degraded "balance-error" approximation rather than
        crashing the episode.
        """
        if self._model_data is None or not (EGRET_AVAILABLE and PYOMO_AVAILABLE):
            return False, {}

        # Patch base loads (rescaled by diurnal/surge) into the ModelData
        # before invoking the OPF builder.
        try:
            loads = self._model_data.data["elements"].get("load", {})
            for egret_id, ldata in loads.items():
                dt_id = self._egret_load_to_dt_id.get(egret_id)
                if dt_id is None:
                    continue
                entry = self._loads.get(dt_id)
                if entry is None:
                    continue
                new_p = float(entry.get("current_demand_mw", 0.0))
                base_p = self._base_load_p_mw.get(egret_id, 0.0)
                # Scale Q with P (constant power factor).
                base_q = self._base_load_q_mvar.get(egret_id, 0.0)
                q_scale = (new_p / base_p) if base_p > 1e-9 else 1.0
                ldata["p_load"] = new_p
                ldata["q_load"] = base_q * q_scale
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("egret_acopf: load patching failed: %s", exc)

        # Disable generators inside their forced-outage window by
        # clamping p_max and q_max to zero for this solve. EGRET's
        # ModelData is a mutable container; we save/restore the original
        # bounds around the solve.
        gen_overrides: list[tuple[str, float, float, float, float]] = []
        try:
            gens = self._model_data.data["elements"].get("generator", {})
            for gid, gspec in gens.items():
                if self._tick < self._gen_forced_outage_until.get(gid, -1):
                    gen_overrides.append(
                        (
                            gid,
                            float(gspec.get("p_min", 0.0) or 0.0),
                            float(gspec.get("p_max", 0.0) or 0.0),
                            float(gspec.get("q_min", 0.0) or 0.0),
                            float(gspec.get("q_max", 0.0) or 0.0),
                        )
                    )
                    gspec["p_min"] = 0.0
                    gspec["p_max"] = 0.0
                    gspec["q_min"] = 0.0
                    gspec["q_max"] = 0.0
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("egret_acopf: gen outage patching failed: %s", exc)

        try:
            # Lazy import: only loaded when a real solve is attempted, so
            # tests that mock EGRET don't have to provide solve_acopf.
            from egret.models.acopf import solve_acopf  # type: ignore[import]

            solver_options = {"max_iter": self._ipopt_max_iter}
            solved_md, results = solve_acopf(
                self._model_data,
                solver="ipopt",
                solver_tee=False,
                return_results=True,
                solver_options=solver_options,
            )
            converged = self._results_converged(results)
            if not converged:
                return False, {}
            payload = self._extract_solve_payload(solved_md)
            return True, payload
        except Exception as exc:
            LOGGER.warning("egret_acopf: IPOPT solve failed: %s", exc)
            return False, {}
        finally:
            # Restore generator bounds for the next tick.
            for gid, p_min, p_max, q_min, q_max in gen_overrides:
                try:
                    gens = self._model_data.data["elements"]["generator"]
                    gens[gid]["p_min"] = p_min
                    gens[gid]["p_max"] = p_max
                    gens[gid]["q_min"] = q_min
                    gens[gid]["q_max"] = q_max
                except Exception:  # pragma: no cover - defensive
                    pass

    @staticmethod
    def _results_converged(results: Any) -> bool:  # pragma: no cover - mock-only
        """Best-effort check on a Pyomo solver Results object."""
        try:
            cond = str(results.solver.termination_condition).lower()
            return "optimal" in cond or "feasible" in cond
        except Exception:
            return True  # if we can't inspect, trust the model_data round-trip

    def _extract_solve_payload(
        self, solved_md: Any
    ) -> dict[str, Any]:  # pragma: no cover - real solve only
        """Pull bus / branch / gen results from a solved ModelData."""
        try:
            elements = solved_md.data["elements"]
            buses = elements.get("bus", {})
            branches = elements.get("branch", {})
            gens = elements.get("generator", {})
            bus_v = {bid: float(b.get("vm", 1.0) or 1.0) for bid, b in buses.items()}
            bus_theta = {
                bid: math.degrees(float(b.get("va", 0.0) or 0.0))
                for bid, b in buses.items()
            }
            branch_p = {
                brid: float(br.get("pf", 0.0) or 0.0) for brid, br in branches.items()
            }
            branch_q = {
                brid: float(br.get("qf", 0.0) or 0.0) for brid, br in branches.items()
            }
            branch_loading = {}
            for brid, br in branches.items():
                rating = float(
                    br.get("rating_long_term", br.get("rating_a", 0.0)) or 0.0
                )
                if rating <= 0:
                    branch_loading[brid] = 0.0
                else:
                    s = math.hypot(
                        float(br.get("pf", 0.0) or 0.0),
                        float(br.get("qf", 0.0) or 0.0),
                    )
                    branch_loading[brid] = 100.0 * s / rating
            n_overloads = sum(1 for x in branch_loading.values() if x > 100.0)
            v_violations = sum(
                1
                for v in bus_v.values()
                if v < self.VOLTAGE_LOWER_PU or v > self.VOLTAGE_UPPER_PU
            )
            gen_p = {gid: float(g.get("pg", 0.0) or 0.0) for gid, g in gens.items()}
            total_p_gen_mw = sum(gen_p.values())
            obj = 0.0
            try:
                obj = float(solved_md.data["system"].get("total_cost", 0.0) or 0.0)
            except Exception:
                obj = 0.0
            return {
                "objective_value": obj,
                "total_p_gen_mw": total_p_gen_mw,
                "bus_v": bus_v,
                "bus_theta_deg": bus_theta,
                "branch_p": branch_p,
                "branch_q": branch_q,
                "branch_loading_percent": branch_loading,
                "gen_p": gen_p,
                "voltage_violations": v_violations,
                "line_overloads": n_overloads,
            }
        except Exception as exc:
            LOGGER.warning("egret_acopf: payload extraction failed: %s", exc)
            return {}

    # ── Perturbations ───────────────────────────────────────────────────

    def _apply_perturbations_at_tick(self, tick: int) -> None:
        events: list[dict[str, Any]] = []
        if self._seed_obj is None:
            self._realized_events_this_tick = events
            return
        for p in self._seed_obj.perturbations:
            if not self._perturbation_active(p, tick):
                continue
            applied = self._apply_one_perturbation(p, tick)
            if applied:
                events.append(applied)
        self._realized_events_this_tick = events

    @staticmethod
    def _perturbation_active(p: Perturbation, tick: int) -> bool:
        return p.trigger_tick <= tick < (p.trigger_tick + p.duration_ticks)

    def _apply_one_perturbation(
        self, p: Perturbation, tick: int
    ) -> dict[str, Any] | None:
        if p.kind == "load_surge":
            self._load_surge_factor_this_tick = float(p.intensity)
            return {
                "type": "load_surge",
                "tick": tick,
                "intensity": p.intensity,
                "stakeholder_class": p.target.get("stakeholder_class"),
                "hidden": p.hidden,
            }
        if p.kind == "generator_forced_outage":
            gen_ids = sorted(self._egret_gens_dict().keys())
            if not gen_ids:
                return None
            idx = int(p.target.get("index", 0)) % len(gen_ids)
            gid = gen_ids[idx]
            self._gen_forced_outage_until[gid] = p.trigger_tick + p.duration_ticks
            if tick == p.trigger_tick:
                return {
                    "type": "generator_outage",
                    "tick": tick,
                    "generator_id": gid,
                    "hidden": p.hidden,
                }
            return None
        if p.kind == "line_outage":
            return {
                "type": "line_outage",
                "tick": tick,
                "line_id": p.target.get("line_index"),
                "hidden": p.hidden,
            }
        if p.kind == "forecast_bias":
            direction = p.target.get("bias_direction", "under-forecast")
            sign = 1.0 if direction == "under-forecast" else -1.0
            self._forecast_bias = sign * float(p.intensity)
            return None
        if p.kind == "wind_dropout":
            self._wind_factor_this_tick = max(0.05, 1.0 - float(p.intensity))
            return {
                "type": "wind_dropout",
                "tick": tick,
                "factor": self._wind_factor_this_tick,
                "hidden": p.hidden,
            }
        if p.kind == "storm_window":
            return {
                "type": "storm_window",
                "tick": tick,
                "intensity": p.intensity,
                "hidden": p.hidden,
            }
        if p.kind == "planned_maintenance":
            gen_ids = sorted(self._egret_gens_dict().keys())
            if not gen_ids:
                return None
            frac = float(p.target.get("fraction", 0.05))
            cutoff = max(1, int(len(gen_ids) * frac))
            for gid in gen_ids[:cutoff]:
                self._gen_forced_outage_until[gid] = p.trigger_tick + p.duration_ticks
            return {
                "type": "planned_maintenance_window",
                "tick": tick,
                "n_gens": cutoff,
                "visible": True,
            }
        return None

    # ── Snapshot ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        if self._model_data is None or self._seed_obj is None:
            return {"entities": {}, "totals": {}, "tick": self._tick}
        entities: dict[str, dict[str, Any]] = {}

        bus_v = (self._last_solution.get("bus_v") if self._last_solution else {}) or {}
        bus_theta = (
            self._last_solution.get("bus_theta_deg") if self._last_solution else {}
        ) or {}
        branch_loading = (
            self._last_solution.get("branch_loading_percent")
            if self._last_solution
            else {}
        ) or {}
        branch_p = (
            self._last_solution.get("branch_p") if self._last_solution else {}
        ) or {}
        branch_q = (
            self._last_solution.get("branch_q") if self._last_solution else {}
        ) or {}
        gen_p = (self._last_solution.get("gen_p") if self._last_solution else {}) or {}

        # Buses
        for bid, bdata in self._egret_buses_dict().items():
            entities[f"bus_{bid}"] = {
                "kind": "bus",
                "bus_id": bid,
                "v_pu": float(bus_v.get(bid, 1.0)),
                "theta_deg": float(bus_theta.get(bid, 0.0)),
                "v_min": float(bdata.get("v_min", 0.0) or 0.0),
                "v_max": float(bdata.get("v_max", 0.0) or 0.0),
            }
        # Branches (lines)
        for brid, brdata in self._egret_branches_dict().items():
            entities[f"line_{brid}"] = {
                "kind": "line",
                "from_bus": brdata.get("from_bus"),
                "to_bus": brdata.get("to_bus"),
                "p_mw": float(branch_p.get(brid, 0.0)),
                "q_mvar": float(branch_q.get(brid, 0.0)),
                "loading_percent": float(branch_loading.get(brid, 0.0)),
                "rho": float(branch_loading.get(brid, 0.0)) / 100.0,
                "in_service": True,
            }
        # Generators
        for gid, gspec in self._egret_gens_dict().items():
            entities[gid] = {
                "kind": "generator",
                "bus_id": gspec.get("bus"),
                "output_mw": float(gen_p.get(gid, 0.0)),
                "power_min": float(gspec.get("p_min", 0.0) or 0.0),
                "power_max": float(gspec.get("p_max", 0.0) or 0.0),
                "forced_outage_until": self._gen_forced_outage_until.get(gid, -1),
                "committed": self._tick >= self._gen_forced_outage_until.get(gid, -1),
            }
        # Loads
        for dt_id, entry in self._loads.items():
            entities[dt_id] = {
                "kind": "load",
                "bus_id": entry.get("bus_id"),
                "current_demand_mw": round(
                    float(entry.get("current_demand_mw", 0.0)), 3
                ),
                "stakeholder_class": entry.get("stakeholder_class"),
                "criticality": entry.get("criticality"),
                "cumulative_shed_mwh": round(
                    self._cumulative_shed_mwh.get(dt_id, 0.0), 3
                ),
            }

        last = self._tick_records[-1] if self._tick_records else None
        totals = {
            "aggregate_demand_mw": last.aggregate_demand_mw if last else 0.0,
            "aggregate_generation_mw": last.aggregate_generation_mw if last else 0.0,
            "balance_error_mw": last.balance_error_mw if last else 0.0,
            "reserves_required_mw": last.reserves_required_mw if last else 0.0,
            "reserves_procured_mw": last.reserves_procured_mw if last else 0.0,
            "production_cost": last.production_cost if last else 0.0,
            "shed_penalty": last.shed_penalty if last else 0.0,
            "voltage_violations": last.voltage_violations if last else 0,
            "line_overloads": last.line_overloads if last else 0,
            "converged": (last.converged if last else True),
        }
        return {
            "tick": self._tick,
            "horizon": self._horizon,
            "entities": entities,
            "totals": totals,
        }

    # ── Forecast / costs / scoring ──────────────────────────────────────

    def forecast_for(self, horizon: int) -> list[dict[str, Any]]:
        """Diurnal demand forecast biased by the active forecast_bias.

        pglib-opf cases are single-snapshot (no chronics), so the
        forecast is the same diurnal curve we drive in ``tick()``,
        biased per the active perturbation. This matches the
        ``forecast_query`` contract on the other backends.
        """
        out: list[dict[str, Any]] = []
        if self._seed_obj is None:
            return out
        peak_tick = max(1, int(self._horizon * 0.7))
        base_total = sum(self._base_load_p_mw.values())
        for offset in range(1, max(1, horizon) + 1):
            t = self._tick + offset
            diurnal = 0.85 + 0.30 * math.sin(math.pi * t / max(1, peak_tick))
            diurnal = max(0.6, min(1.20, diurnal))
            true_demand = base_total * diurnal
            biased = true_demand * max(0.0, 1.0 - self._forecast_bias)
            out.append(
                {
                    "tick": t,
                    "demand_mw_forecast": round(biased, 2),
                    "forecast_bias": round(self._forecast_bias, 4),
                }
            )
        return out

    def ground_truth_costs(self) -> dict[str, float]:
        """OPF objective decomposition for the scorer.

        Returns the same key set used by every other backend so the
        scorer's ``optimality_gap`` dimension reads ``production_cost``
        unchanged (see module docstring for why we still ship a real
        production-cost number even though the LP oracle is DC-style).
        """
        if not self._tick_records:
            return {
                "production_cost": 0.0,
                "startup_cost": 0.0,
                "shed_penalty": 0.0,
                "balance_error_cost": 0.0,
                "reserve_violation_cost": 0.0,
            }
        prod = sum(r.production_cost for r in self._tick_records)
        startup = sum(r.startup_cost for r in self._tick_records)
        shed = sum(r.shed_penalty for r in self._tick_records)
        balance = (
            sum(abs(r.balance_error_mw) for r in self._tick_records)
            * self.BALANCE_ERROR_PENALTY_PER_MW
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

        IMPORTANT: keys MUST match the canonical names the scorer reads
        (see ``evaluation/scorer.py::score_safety_violation`` and
        ``score_system_survival`` for the canonical contract). Pre-v0.3
        this method emitted backend-private attribute names
        (``voltage_violations``, ``line_overloads``) which the scorer's
        ``r.get("n_voltage_violations", 0)`` calls silently mapped to
        0 — producing perfect-safety scores for every EGRET scenario
        regardless of solver state. v0.3 maps the private attrs to the
        scorer-facing keys here.

        ``n_disconnected_lines`` is honestly emitted as 0 because
        EGRET-OPF does not model discrete branch-status transitions;
        ``rho_max`` is derived from per-branch loading percent.
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
                # Canonical scorer-facing keys (v0.3.0 fix):
                "n_voltage_violations": int(r.voltage_violations),
                "n_overloads": int(r.line_overloads),
                "n_disconnected_lines": 0,
                "rho_max": float(getattr(r, "rho_max", 0.0) or 0.0),
                # v0.3.4: early-guard `done` (align with cigre/grid2op/pglib)
                # so a normal horizon-end tick is never miscounted as a
                # catastrophic blackout by score_system_survival.
                "done": bool(r.done and r.tick < self._horizon - 1),
                # Internal diagnostics retained for trajectory logs:
                "converged": r.converged,
            }
            for r in self._tick_records
        ]

    def per_load_shed_mwh(self) -> dict[str, float]:
        return {lid: round(v, 3) for lid, v in self._cumulative_shed_mwh.items()}

    def realized_events_for_tick(self) -> list[dict[str, Any]]:
        """Surface the per-tick realized events (used by some tools)."""
        return list(self._realized_events_this_tick)
