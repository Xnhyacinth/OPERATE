"""
domains.power_grid.backends.pandapower_acopf — real AC Optimal Power Flow.

v0.4 fourth backend category. Solves a FULL nonlinear AC Optimal Power
Flow every tick on PGLib-OPF IEEE benchmark cases (case14 / case30 /
case57 / case118 in v0.4; ≤300-bus stretch). Unlike
``pglib_uc_synthetic`` (aggregate UC, NO power flow), this backend models
the real grid physics:

- bus voltage magnitudes (``v_pu``) and the per-bus voltage band
  [VOLTAGE_LOWER_PU, VOLTAGE_UPPER_PU] enforced as hard OPF constraints
- real + reactive line/trafo flows and branch loading percentage
- generator real/reactive dispatch co-optimized by the OPF
- network topology constraints (line limits)

Why pandapower and not EGRET: pandapower is already a hard project
dependency (the CIGRE distribution backend uses ``pp.runpp`` today) and
ships a native AC-OPF (``pp.runopp``) that needs no EGRET/Pyomo/IPOPT
install. The earlier ``egret_acopf`` backend required an unavailable
solver stack and shipped 0 release scenarios; this backend supersedes it
and is always runnable.

Contract surface mirrors ``CigreDistributionBackend`` /
``PglibUcSyntheticBackend`` exactly (``reset / tick / snapshot /
apply_tool_effect / ground_truth_costs / scoring_records /
per_load_shed_mwh / forecast_for / queue_mutual_aid_effect``) so the
adapter and scorer reuse without modification.

Scoring separation (vs the EGRET stub): ``production_cost`` is the PURE
generation cost (``net.res_cost``); voltage / overload / disconnection
surcharges live in separate ``ground_truth_costs()`` keys so
``optimality_gap`` stays a clean dispatch-efficiency metric. The backend
exposes ``acopf_reference_optimum()`` (the sum of per-tick OPF optima
solved with the agent's redispatch pins RELEASED) so the runner can feed
a TRUE per-tick AC-OPF optimality gap via ``ScoringInputs.lp_optimum``.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import pandapower as pp  # type: ignore[import-untyped]
from pandapower.converter.matpower import from_mpc  # type: ignore[import-untyped]

from ..paired_rts_load import (
    CASE73_CASE,
    CASE73_CONTRACT,
    SYNTHETIC_CASES,
    SYNTHETIC_CONTRACT,
    WINDOW_RECIPE,
    hourly_region_means,
    load_day_rows,
    parse_calendar_date,
    region_base_mw,
    select_horizon_rows,
    synthetic_scaled_profile,
    window_sha256,
)
from ..seeds.schema import ScenarioSeed
from ..source_paths import resolve_source_ref

ACOPF_PERTURBATION_EVENT_CLASS = MappingProxyType(
    {
        "line_outage": "alarm",
        "generator_outage": "alarm",
        "planned_maintenance": "alarm",
        "load_surge": "alarm",
        "forecast_bias": "forecast",
        "storm_window": "alarm",
    }
)
ACOPF_PERTURBATION_KIND_REGISTRY = MappingProxyType(
    {
        "line_outage": "line_outage",
        "generator_forced_outage": "generator_outage",
        "planned_maintenance": "planned_maintenance",
        "load_surge": "load_surge",
        "forecast_bias": "forecast_bias",
        "storm_window": "storm_window",
    }
)


@dataclass
class PandapowerAcopfTickRecord:
    tick: int
    aggregate_demand_mw: float
    aggregate_generation_mw: float
    balance_error_mw: float
    reserves_required_mw: float
    reserves_procured_mw: float
    production_cost: float  # PURE generation cost = net.res_cost
    startup_cost: float  # 0.0 — AC-OPF is continuous, no UC binaries
    shed_penalty: float
    rho_max: float
    n_overloads: int
    n_voltage_violations: int
    n_disconnected_lines: int
    converged: bool = True
    done: bool = False
    realized_events: list[dict[str, Any]] = field(default_factory=list)


class PandapowerAcopfBackend:
    """pandapower-native AC-OPF backend on PGLib-OPF cases."""

    # Keep the adapter's supervisory cadence tied to the registered native
    # backend contract rather than to scenario prose.
    backend_kind = "pandapower_acopf"

    # Tariffs identical to CIGRE/pglib backends so cross-backend
    # comparisons stay apples-to-apples.
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
    OVERLOAD_COST_PER_TICK = 200.0
    VOLTAGE_VIOLATION_COST_PER_TICK = 1200.0
    DISCONNECTION_COST_PER_LINE_TICK = 500.0
    RESERVE_TARGET_FRACTION_OF_DEMAND = 0.10
    # Transmission voltage band — pglib cases ship per-bus vmin/vmax; we
    # use a [0.95, 1.05] band for the post-hoc violation count so the
    # system_survival catastrophic "voltage_collapse>=5" threshold keys
    # off the same band the other backends use.
    VOLTAGE_LOWER_PU = 0.95
    VOLTAGE_UPPER_PU = 1.05
    # Finite cap for branches whose MATPOWER rateA is 0 (unlimited);
    # without this loading_percent is NaN/inf and the safety scorers
    # silently misread it.
    UNLIMITED_BRANCH_CAP_PCT = 1000.0

    def __init__(self) -> None:
        self._net: Any = None
        self._seed_obj: ScenarioSeed | None = None
        self._tick: int = 0
        self._horizon: int = 24
        self._tick_records: list[PandapowerAcopfTickRecord] = []
        self._loads: dict[str, dict[str, Any]] = {}
        self._base_load_p_mw: dict[int, float] = {}
        self._base_load_q_mvar: dict[int, float] = {}
        self._cumulative_shed_mwh: dict[str, float] = {}
        # redispatch pins: gen index → forced target MW (realized solve).
        self._gen_pins: dict[int, float] = {}
        self._line_status_overrides: dict[int, bool] = {}
        self._gen_outage_until: dict[int, int] = {}
        self._class_surge_factors: dict[str, float] = {}
        self._forecast_bias: float = 0.0
        self._pending_reserve_extra_mw: float = 0.0
        self._committed_reserve_mw: float = 0.0
        self._reserve_procurement_cost_total: float = 0.0
        self._reserve_shortfall_cost_total: float = 0.0
        self._last_balance_error: float = 0.0
        self._done: bool = False
        self._pending_mutual_aid: list[tuple[int, float]] = []
        self._idx_to_load_id: dict[int, str] = {}
        # Sum of per-tick reference OPF optima (pins released) — the TRUE
        # AC-OPF optimality-gap denominator.
        self._reference_optimum_total: float = 0.0
        # Restorable base gen capability so released-pin reference solves
        # use native limits.
        self._base_gen_min_p: dict[int, float] = {}
        self._base_gen_max_p: dict[int, float] = {}
        self._base_ext_grid_max_p: dict[int, float] = {}
        self._case_source_declared: str = ""
        self._case_source_path: str = ""
        self._case_source_sha256: str = ""
        self._native_net_fields: dict[str, int] = {}
        self._parser_output_digest: str = ""
        self._initial_solver_state: dict[str, Any] | None = None
        self._paired_load_profile: list[dict[int, float]] = []
        self._paired_timeseries_contract: str = ""
        self._paired_source_assets: dict[str, str] = {}
        self._paired_source_hashes: dict[str, str] = {}
        self._paired_consumption_ticks: list[int] = []
        self._paired_window_sha256: str = ""
        self._paired_recipe_version: str = ""
        self._paired_calendar_date: str = ""

    # ── Reset ──────────────────────────────────────────────────────────

    def reset(self, scenario_seed: ScenarioSeed) -> None:
        self._seed_obj = scenario_seed
        self._tick = 0
        self._horizon = scenario_seed.horizon_ticks
        self._tick_records.clear()
        self._cumulative_shed_mwh.clear()
        self._gen_pins.clear()
        self._line_status_overrides.clear()
        self._gen_outage_until.clear()
        self._class_surge_factors.clear()
        self._forecast_bias = 0.0
        self._pending_reserve_extra_mw = 0.0
        self._committed_reserve_mw = 0.0
        self._reserve_procurement_cost_total = 0.0
        self._reserve_shortfall_cost_total = 0.0
        self._last_balance_error = 0.0
        self._done = False
        self._pending_mutual_aid = []
        self._idx_to_load_id.clear()
        self._reference_optimum_total = 0.0
        self._base_gen_min_p.clear()
        self._base_gen_max_p.clear()
        self._base_ext_grid_max_p.clear()
        self._initial_solver_state = None
        self._paired_load_profile = []
        self._paired_timeseries_contract = ""
        self._paired_source_assets = {}
        self._paired_source_hashes = {}
        self._paired_consumption_ticks = []
        self._paired_window_sha256 = ""
        self._paired_recipe_version = ""
        self._paired_calendar_date = ""

        for perturbation in scenario_seed.perturbations:
            self._validate_perturbation_kind(perturbation)

        case_file = scenario_seed.backend_config.get("case_file", "")
        case_path = resolve_source_ref(case_file, description="PGLib-OPF case")
        self._case_source_declared = str(case_file)
        self._case_source_path = str(case_path.resolve())
        self._case_source_sha256 = hashlib.sha256(case_path.read_bytes()).hexdigest()
        self._net = from_mpc(str(case_path))
        parser_payload = self._native_parser_payload(self._net)
        self._native_net_fields = parser_payload["native_net_fields"]
        self._parser_output_digest = self._semantic_digest(parser_payload)
        self._make_opf_ready(self._net)

        # Cache base profiles for deterministic per-tick rebuild.
        self._base_load_p_mw = {
            int(i): float(self._net.load.p_mw.iloc[i])
            for i in range(len(self._net.load))
        }
        self._base_load_q_mvar = {
            int(i): float(self._net.load.q_mvar.iloc[i])
            for i in range(len(self._net.load))
        }
        for i in range(len(self._net.gen)):
            self._base_gen_min_p[int(i)] = float(self._net.gen.min_p_mw.iloc[i])
            self._base_gen_max_p[int(i)] = float(self._net.gen.max_p_mw.iloc[i])
        for i in range(len(self._net.ext_grid)):
            self._base_ext_grid_max_p[int(i)] = float(
                self._net.ext_grid.max_p_mw.iloc[i]
            )

        # Map load assignments → pandapower load indices in seed order.
        self._loads = {}
        for idx, assignment in enumerate(scenario_seed.load_assignments):
            if idx >= len(self._net.load):
                break
            self._loads[assignment.load_id] = {
                "load_index": idx,
                "bus_id": int(self._net.load.bus.iloc[idx]),
                "stakeholder_class": assignment.stakeholder_class,
                "criticality": assignment.criticality,
                "shed_this_tick_mw": 0.0,
            }
            self._cumulative_shed_mwh[assignment.load_id] = 0.0
            self._idx_to_load_id[idx] = assignment.load_id

        self._configure_paired_timeseries(scenario_seed)

        # Pre-apply perturbations firing at t=0.
        self._apply_perturbations_at_tick(0)
        self._reserve_decision_lever()
        self._emergency_reserve_protection()

    def _configure_paired_timeseries(self, scenario_seed: ScenarioSeed) -> None:
        """Load an RTS-GMLC regional-load window onto a PGLib-OPF case."""
        raw = scenario_seed.backend_config.get("paired_timeseries")
        if raw is None:
            return
        if not isinstance(raw, dict):
            raise ValueError("paired_timeseries must be a mapping")
        contract = str(raw.get("contract", ""))
        case_name = str(scenario_seed.backend_config.get("case_name", ""))
        identity_pair = contract == CASE73_CONTRACT and case_name == CASE73_CASE
        synthetic_pair = (
            contract == SYNTHETIC_CONTRACT and case_name in SYNTHETIC_CASES
        )
        if not identity_pair and not synthetic_pair:
            raise ValueError(
                "paired_timeseries is source-locked to RTS-GMLC regional load "
                "with either pglib_opf_case73_ieee_rts identity pairing or "
                "synthetic topology-profile composition on "
                "pglib_opf_case14_ieee / pglib_opf_case30_ieee"
            )

        required = ("load_csv", "bus_csv")
        if identity_pair:
            required = ("load_csv", "bus_csv", "branch_csv")
        declared_assets = {name: str(raw.get(name, "")) for name in required}
        if any(not value for value in declared_assets.values()):
            raise ValueError(
                "paired_timeseries requires " + ", ".join(required)
            )
        resolved_assets = {
            name: resolve_source_ref(value, description=f"paired {name}")
            for name, value in declared_assets.items()
        }
        self._paired_source_assets = {
            declared_assets[name]: str(path.resolve())
            for name, path in resolved_assets.items()
        }
        self._paired_source_hashes = {
            declared_assets[name]: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in resolved_assets.items()
        }

        with resolved_assets["bus_csv"].open(encoding="utf-8", newline="") as stream:
            bus_rows = list(csv.DictReader(stream))
        region_base = region_base_mw(bus_rows)
        calendar_day = parse_calendar_date(str(raw.get("calendar_date", "")))
        periods_per_tick = int(raw.get("periods_per_tick", 0))
        selected_rows = select_horizon_rows(
            load_day_rows(resolved_assets["load_csv"], calendar_day),
            horizon_ticks=self._horizon,
            periods_per_tick=periods_per_tick,
        )
        hourly = hourly_region_means(
            selected_rows,
            periods_per_tick=periods_per_tick,
            regions=region_base,
        )
        self._paired_calendar_date = calendar_day.isoformat()
        self._paired_recipe_version = WINDOW_RECIPE
        self._paired_window_sha256 = window_sha256(
            selected_rows,
            calendar_date=self._paired_calendar_date,
            periods_per_tick=periods_per_tick,
        )

        if identity_pair:
            if len(bus_rows) != 73:
                raise ValueError(
                    "RTS-GMLC case73 pairing requires exactly 73 source buses"
                )
            bus_by_id = {int(row["Bus ID"]): row for row in bus_rows}
            net_load_by_bus: dict[int, float] = {}
            for idx, base_mw in self._base_load_p_mw.items():
                bus_id = int(self._net.load.bus.iloc[idx]) + 1
                net_load_by_bus[bus_id] = net_load_by_bus.get(bus_id, 0.0) + base_mw
            for bus_id, row in bus_by_id.items():
                if not math.isclose(
                    net_load_by_bus.get(bus_id, 0.0),
                    float(row["MW Load"]),
                    abs_tol=1e-6,
                ):
                    raise ValueError(
                        f"RTS-GMLC bus load does not match PGLib case73 at bus {bus_id}"
                    )

            branch_rows = self._matpower_matrix_rows(
                resolve_source_ref(
                    self._case_source_declared, description="PGLib-OPF case"
                ).read_text(encoding="utf-8"),
                "branch",
            )
            with resolved_assets["branch_csv"].open(
                encoding="utf-8", newline=""
            ) as stream:
                paired_branches = list(csv.DictReader(stream))
            if len(branch_rows) != len(paired_branches):
                raise ValueError("RTS-GMLC branches do not match PGLib case73")
            branch_columns = (
                ("From Bus", 0),
                ("To Bus", 1),
                ("R", 2),
                ("X", 3),
                ("B", 4),
                ("Cont Rating", 5),
                ("LTE Rating", 6),
                ("STE Rating", 7),
            )
            for source_row, case_row in zip(paired_branches, branch_rows, strict=True):
                if any(
                    not math.isclose(
                        float(source_row[column]), case_row[position], abs_tol=1e-9
                    )
                    for column, position in branch_columns
                ):
                    raise ValueError(
                        "RTS-GMLC branch parameters do not match PGLib case73 "
                        f"at {source_row.get('UID', '<unknown>')}"
                    )
            profile: list[dict[int, float]] = []
            for region_mean_mw in hourly:
                tick_profile: dict[int, float] = {}
                for idx, base_mw in self._base_load_p_mw.items():
                    bus_id = int(self._net.load.bus.iloc[idx]) + 1
                    region = int(bus_by_id[bus_id]["Area"])
                    tick_profile[idx] = (
                        region_mean_mw[region] * base_mw / region_base[region]
                    )
                profile.append(tick_profile)
            self._paired_load_profile = profile
        else:
            self._paired_load_profile = synthetic_scaled_profile(
                base_load_p_mw=self._base_load_p_mw,
                hourly_region_mw=hourly,
                region_base=region_base,
            )
        self._paired_timeseries_contract = contract

    @staticmethod
    def _matpower_matrix_rows(text: str, name: str) -> list[list[float]]:
        marker = f"mpc.{name} = ["
        start = text.find(marker)
        if start < 0:
            raise ValueError(f"MATPOWER source is missing {name} matrix")
        end = text.find("];", start)
        if end < 0:
            raise ValueError(f"MATPOWER source has unterminated {name} matrix")
        rows: list[list[float]] = []
        for line in text[start + len(marker) : end].splitlines():
            values = line.strip().rstrip(";").split()
            if values:
                rows.append([float(value) for value in values])
        return rows

    def _make_opf_ready(self, net: Any) -> None:
        """Guard the converted net so ``runopp`` always has finite bounds."""
        # Controllable gens/ext_grid so the OPF can dispatch them.
        if len(net.gen) > 0:
            net.gen["controllable"] = True
        if len(net.ext_grid) > 0:
            net.ext_grid["controllable"] = True
        # Voltage bands.
        if "min_vm_pu" not in net.bus or net.bus["min_vm_pu"].isna().any():
            net.bus["min_vm_pu"] = net.bus.get("min_vm_pu", 0.9)
            net.bus["min_vm_pu"] = net.bus["min_vm_pu"].fillna(0.9)
        if "max_vm_pu" not in net.bus or net.bus["max_vm_pu"].isna().any():
            net.bus["max_vm_pu"] = net.bus.get("max_vm_pu", 1.1)
            net.bus["max_vm_pu"] = net.bus["max_vm_pu"].fillna(1.1)
        # Branch loading caps — replace 0 / NaN with a large finite cap.
        for tbl in ("line", "trafo"):
            df = getattr(net, tbl)
            if len(df) == 0:
                continue
            if "max_loading_percent" not in df:
                df["max_loading_percent"] = self.UNLIMITED_BRANCH_CAP_PCT
            df["max_loading_percent"] = (
                df["max_loading_percent"]
                .fillna(self.UNLIMITED_BRANCH_CAP_PCT)
                .replace(0, self.UNLIMITED_BRANCH_CAP_PCT)
            )
        # Ensure poly_cost exists for every controllable unit; from_mpc
        # usually populates this from gencost. If a gen lacks a cost row,
        # inject a default linear cost so the OPF objective is defined.
        try:
            costed_gens = set(net.poly_cost[net.poly_cost.et == "gen"].element.tolist())
        except Exception:
            costed_gens = set()
        for i in range(len(net.gen)):
            if i not in costed_gens:
                with contextlib.suppress(Exception):
                    pp.create_poly_cost(net, i, "gen", cp1_eur_per_mw=50.0, cp0_eur=0.0)

    # ── Tool effects ────────────────────────────────────────────────────

    def apply_tool_effect(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "shed_load":
            load_id = str(args.get("load_id", ""))
            mw = float(args.get("mw", 0.0))
            entry = self._loads.get(load_id)
            if entry is None:
                return {"_status": "error", "error": "unknown_load", "load_id": load_id}
            entry["shed_this_tick_mw"] += mw
            tick_h = float(self._seed_obj.tick_minutes if self._seed_obj else 60) / 60.0
            self._cumulative_shed_mwh[load_id] += mw * tick_h
            return {
                "load_id": load_id,
                "shed_mw": mw,
                "stakeholder_class": entry["stakeholder_class"],
                "criticality": entry["criticality"],
            }
        if name == "redispatch_generation":
            try:
                gen_idx = int(args.get("generator_index", args.get("generator_id", 0)))
            except (TypeError, ValueError):
                gen_id = str(args.get("generator_id", ""))
                gen_idx = next(
                    (i for i, n in enumerate(self._net.gen.name) if str(n) == gen_id),
                    0,
                )
            target = float(args.get("target_mw", args.get("delta_mw", 0.0)))
            self._gen_pins[gen_idx] = max(0.0, target)
            return {
                "generator_index": gen_idx,
                "target_mw": round(target, 3),
                "queued": True,
            }
        if name == "switch_branch":
            line_index = int(args.get("line_index", 0))
            connect = bool(args.get("connect", True))
            self._line_status_overrides[line_index] = connect
            return {"line_id": line_index, "connect": connect, "queued": True}
        if name == "commit_reserve":
            mw = float(args.get("mw", 0.0))
            if not math.isfinite(mw):
                return {
                    "_status": "error",
                    "error": "non_finite_reserve",
                    "reason_code": "non_finite_reserve",
                    "requested_mw": mw,
                }
            if mw <= 0.0:
                return {"_status": "error", "error": "non_positive_reserve", "mw": mw}
            capacity = self._reserve_capacity_state()
            max_deliverable_mw = float(capacity["remaining_deliverable_mw"])
            evidence = {
                "requested_mw": round(mw, 6),
                "max_deliverable_mw": round(max_deliverable_mw, 6),
                "physical_headroom_mw": round(
                    float(capacity["physical_headroom_mw"]), 6
                ),
                "source_generation_capacity_mw": round(
                    float(capacity["source_generation_capacity_mw"]), 6
                ),
                "source_gen_capacity_mw": round(
                    float(capacity["source_gen_capacity_mw"]), 6
                ),
                "source_ext_grid_capacity_mw": round(
                    float(capacity["source_ext_grid_capacity_mw"]), 6
                ),
                "current_dispatch_mw": round(
                    float(capacity["current_dispatch_mw"]), 6
                ),
                "capacity_basis": str(capacity["capacity_basis"]),
                "dispatch_basis": str(capacity["dispatch_basis"]),
                "source_case_sha256": self._case_source_sha256,
            }
            if mw > max_deliverable_mw + 1e-9:
                return {
                    "_status": "error",
                    "error": "reserve_request_exceeds_physical_headroom",
                    "reason_code": "reserve_request_exceeds_physical_headroom",
                    **evidence,
                }
            self._pending_reserve_extra_mw += mw
            self._committed_reserve_mw += mw
            return {
                "reserve_pending_mw": round(self._pending_reserve_extra_mw, 3),
                "committed_reserve_mw": round(self._committed_reserve_mw, 3),
                "info": "acopf reserve commitment queued",
                **evidence,
            }
        if name == "request_mutual_aid":
            return {
                "_status": "ack",
                "info": "mutual-aid uses the dedicated delayed-effect path",
            }
        if name == "topology_action":
            return {
                "_status": "unsupported_on_pandapower_acopf",
                "info": "substation bus-splitting deferred to v0.5",
            }
        return {"_status": "noop"}

    # ── F-01 delayed-effect API ─────────────────────────────────────────

    def queue_mutual_aid_effect(self, *, due_tick: int, mw: float) -> None:
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

    def _reserve_capacity_state(self) -> dict[str, float | str]:
        """Return the single physical capacity basis for reserve accounting."""
        source_gen_capacity_mw = sum(
            max(0.0, maximum)
            for index, maximum in self._base_gen_max_p.items()
            if 0 <= index < len(self._net.gen)
            and bool(self._net.gen.in_service.iloc[index])
            and not (
                self._gen_outage_until.get(index) is not None
                and self._tick < int(self._gen_outage_until[index])
            )
        )
        source_ext_grid_capacity_mw = sum(
            max(0.0, maximum)
            for index, maximum in self._base_ext_grid_max_p.items()
            if 0 <= index < len(self._net.ext_grid)
            and bool(self._net.ext_grid.in_service.iloc[index])
        )
        source_generation_capacity_mw = (
            source_gen_capacity_mw + source_ext_grid_capacity_mw
        )

        has_solver_dispatch = bool(
            self._tick_records or getattr(self._net, "OPF_converged", False)
        )
        if has_solver_dispatch:
            gen_dispatch_mw = sum(
                max(0.0, float(value))
                for value in self._net.res_gen.p_mw.tolist()
                if math.isfinite(float(value))
            )
            ext_grid_dispatch_mw = sum(
                max(0.0, float(value))
                for value in self._net.res_ext_grid.p_mw.tolist()
                if math.isfinite(float(value))
            )
            dispatch_basis = "pandapower_solver_results"
        else:
            gen_dispatch_mw = sum(
                max(0.0, float(value))
                for value in self._net.gen.p_mw.tolist()
                if math.isfinite(float(value))
            )
            # from_mpc represents the source slack generator as ext_grid and
            # does not retain its PG setpoint.  Before the first solve, infer
            # only the balancing portion from converted source demand.
            demand_mw = sum(
                max(0.0, float(value))
                for value in self._net.load.p_mw.tolist()
                if math.isfinite(float(value))
            )
            ext_grid_dispatch_mw = max(0.0, demand_mw - gen_dispatch_mw)
            dispatch_basis = "converted_source_setpoints_and_balance"
        current_dispatch_mw = gen_dispatch_mw + ext_grid_dispatch_mw
        physical_headroom_mw = max(
            0.0, source_generation_capacity_mw - current_dispatch_mw
        )
        remaining_deliverable_mw = max(
            0.0, physical_headroom_mw - self._pending_reserve_extra_mw
        )
        return {
            "source_gen_capacity_mw": source_gen_capacity_mw,
            "source_ext_grid_capacity_mw": source_ext_grid_capacity_mw,
            "source_generation_capacity_mw": source_generation_capacity_mw,
            "current_dispatch_mw": current_dispatch_mw,
            "physical_headroom_mw": physical_headroom_mw,
            "remaining_deliverable_mw": remaining_deliverable_mw,
            "capacity_basis": (
                "converted_source_gen_and_ext_grid_pmax_minus_current_dispatch_v1"
            ),
            "dispatch_basis": dispatch_basis,
        }

    # ── Tick ────────────────────────────────────────────────────────────

    def tick(self, current_tick: int) -> PandapowerAcopfTickRecord:
        assert self._net is not None
        self._tick = current_tick
        matured_aid = self._drain_mutual_aid(current_tick)
        if matured_aid > 0.0:
            self._pending_reserve_extra_mw += matured_aid
        realized_events = self._apply_perturbations_at_tick(current_tick)
        if matured_aid > 0.0:
            realized_events.append(
                {
                    "kind": "mutual_aid_arrived",
                    "event_class": "agent_outcome",
                    "origin": "agent_caused",
                    "decision_required": False,
                    "actionable": False,
                    "tick": current_tick,
                    "mw": round(matured_aid, 3),
                }
            )

        tick_h = float(self._seed_obj.tick_minutes if self._seed_obj else 60) / 60.0
        peak_tick = max(1, int(self._horizon * 0.7))
        diurnal = 0.85 + 0.30 * math.sin(math.pi * current_tick / max(1, peak_tick))
        diurnal = max(0.6, min(1.20, diurnal))
        paired_tick_profile = (
            self._paired_load_profile[current_tick]
            if current_tick < len(self._paired_load_profile)
            else None
        )
        if paired_tick_profile is not None:
            base_total = sum(self._base_load_p_mw.values())
            diurnal = sum(paired_tick_profile.values()) / max(base_total, 1e-9)
        self._apply_emergency_reserve_protection(
            current_tick=current_tick,
            tick_h=tick_h,
            diurnal=diurnal,
            events=realized_events,
        )

        # Compose demand with per-class surges + shed.
        for idx, base in self._base_load_p_mw.items():
            load_id = self._idx_to_load_id.get(idx, "")
            entry = self._loads.get(load_id, {})
            cls = str(entry.get("stakeholder_class", ""))
            surge = 1.0 + self._class_surge_factors.get(cls, 0.0)
            shed_mw = float(entry.get("shed_this_tick_mw", 0.0))
            source_target = (
                paired_tick_profile[idx]
                if paired_tick_profile is not None
                else base * diurnal
            )
            target = source_target * surge
            new_p = max(0.0, target - shed_mw)
            self._net.load.at[idx, "p_mw"] = new_p
            q_base = self._base_load_q_mvar.get(idx, 0.0)
            p_factor = new_p / base if base > 1e-9 else 1.0
            self._net.load.at[idx, "q_mvar"] = q_base * p_factor
        if paired_tick_profile is not None:
            self._paired_consumption_ticks.append(current_tick)
            realized_events.append(
                {
                    "kind": "paired_load_realization",
                    "event_class": "telemetry",
                    "decision_required": False,
                    "actionable": False,
                    "tick": current_tick,
                    "contract": self._paired_timeseries_contract,
                    "aggregate_demand_mw": round(
                        sum(paired_tick_profile.values()), 6
                    ),
                }
            )

        # Apply outages + line overrides.
        self._apply_gen_outages(current_tick)
        for line_idx, connect in self._line_status_overrides.items():
            if 0 <= line_idx < len(self._net.line):
                self._net.line.at[line_idx, "in_service"] = connect

        # Realized solve: apply redispatch pins, then OPF.
        self._apply_gen_pins()
        converged, prod_cost = self._solve()

        # Reference solve (pins released) for the operational optimality
        # gap — only when the agent actually pinned a generator this tick.
        # P2-8 fix: never add an infeasible reference (0.0) to the
        # denominator — that silently shrinks the optimum and distorts the
        # gap. If the reference solve fails, fall back to the realized
        # cost (gap contribution 0 for that tick).
        if self._gen_pins:
            ref_cost = self._reference_solve()
            if ref_cost is not None and ref_cost > 0.0:
                self._reference_optimum_total += ref_cost
            else:
                self._reference_optimum_total += prod_cost if converged else 0.0
        else:
            self._reference_optimum_total += prod_cost if converged else 0.0

        record = self._build_record(
            current_tick, converged, prod_cost, realized_events, tick_h
        )
        self._tick_records.append(record)
        if current_tick == 0:
            solver_state = self._native_solver_state(record)
            self._initial_solver_state = {
                **solver_state,
                "state_digest": self._semantic_digest(solver_state),
            }

        # Reset per-tick scratch.
        for entry in self._loads.values():
            entry["shed_this_tick_mw"] = 0.0
        self._class_surge_factors.clear()
        return record

    @staticmethod
    def _semantic_digest(payload: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _finite(value: Any) -> float | str:
        number = float(value)
        return round(number, 9) if math.isfinite(number) else str(number)

    def _native_parser_payload(self, net: Any) -> dict[str, Any]:
        bus_rows = [
            {
                "index": int(index),
                "vn_kv": self._finite(row["vn_kv"]),
                "min_vm_pu": self._finite(row.get("min_vm_pu", 0.0)),
                "max_vm_pu": self._finite(row.get("max_vm_pu", 0.0)),
            }
            for index, row in net.bus.iterrows()
        ]
        load_rows = [
            {
                "index": int(index),
                "bus": int(row["bus"]),
                "p_mw": self._finite(row["p_mw"]),
                "q_mvar": self._finite(row["q_mvar"]),
            }
            for index, row in net.load.iterrows()
        ]
        generator_rows = [
            {
                "index": int(index),
                "bus": int(row["bus"]),
                "p_mw": self._finite(row["p_mw"]),
                "min_p_mw": self._finite(row.get("min_p_mw", 0.0)),
                "max_p_mw": self._finite(row.get("max_p_mw", 0.0)),
                "min_q_mvar": self._finite(row.get("min_q_mvar", 0.0)),
                "max_q_mvar": self._finite(row.get("max_q_mvar", 0.0)),
            }
            for index, row in net.gen.iterrows()
        ]
        external_grid_rows = [
            {
                "index": int(index),
                "bus": int(row["bus"]),
                "min_p_mw": self._finite(row.get("min_p_mw", 0.0)),
                "max_p_mw": self._finite(row.get("max_p_mw", 0.0)),
                "min_q_mvar": self._finite(row.get("min_q_mvar", 0.0)),
                "max_q_mvar": self._finite(row.get("max_q_mvar", 0.0)),
            }
            for index, row in net.ext_grid.iterrows()
        ]
        line_rows = [
            {
                "index": int(index),
                "from_bus": int(row["from_bus"]),
                "to_bus": int(row["to_bus"]),
                "max_i_ka": self._finite(row["max_i_ka"]),
            }
            for index, row in net.line.iterrows()
        ]
        trafo_rows = [
            {
                "index": int(index),
                "hv_bus": int(row["hv_bus"]),
                "lv_bus": int(row["lv_bus"]),
                "sn_mva": self._finite(row["sn_mva"]),
            }
            for index, row in net.trafo.iterrows()
        ]
        native_fields = {
            "bus_count": len(bus_rows),
            "load_count": len(load_rows),
            "generator_count": len(generator_rows) + len(net.ext_grid),
            "branch_count": len(line_rows) + len(trafo_rows),
            "line_count": len(line_rows),
            "transformer_count": len(trafo_rows),
            "cost_row_count": len(net.poly_cost),
        }
        return {
            "native_net_fields": native_fields,
            "buses": bus_rows,
            "loads": load_rows,
            "generators": generator_rows,
            "external_grids": external_grid_rows,
            "source_generation_capacity_mw": self._finite(
                sum(float(row.get("max_p_mw", 0.0)) for _, row in net.gen.iterrows())
                + sum(
                    float(row.get("max_p_mw", 0.0))
                    for _, row in net.ext_grid.iterrows()
                )
            ),
            "lines": line_rows,
            "transformers": trafo_rows,
        }

    def _native_solver_state(
        self, record: PandapowerAcopfTickRecord
    ) -> dict[str, Any]:
        return {
            "tick": int(record.tick),
            "converged": bool(record.converged),
            "bus_voltage_pu": [
                self._finite(value) for value in self._net.res_bus.vm_pu.tolist()
            ],
            "generator_dispatch_mw": [
                self._finite(value) for value in self._net.res_gen.p_mw.tolist()
            ],
            "fixed_generator_dispatch_mw": [
                self._finite(value) for value in self._net.res_sgen.p_mw.tolist()
            ],
            "external_grid_dispatch_mw": [
                self._finite(value)
                for value in self._net.res_ext_grid.p_mw.tolist()
            ],
            "line_loading_percent": [
                self._finite(value)
                for value in self._net.res_line.loading_percent.tolist()
            ],
            "transformer_loading_percent": [
                self._finite(value)
                for value in self._net.res_trafo.loading_percent.tolist()
            ],
            "production_cost": self._finite(record.production_cost),
        }

    def protocol21_source_trace(self) -> dict[str, Any]:
        """Bind the opened PGLib case to parsed net and initial solve state."""
        initial = dict(self._initial_solver_state or {})
        state_effect = bool(initial and initial.get("converged") is True)
        paired_effect = bool(self._paired_consumption_ticks)
        semantic_payload = {
            "case_source_sha256": self._case_source_sha256,
            "parser_output_digest": self._parser_output_digest,
            "initial_solver_state": initial,
            "paired_timeseries_contract": self._paired_timeseries_contract,
            "paired_source_hashes": self._paired_source_hashes,
            "paired_consumption_ticks": self._paired_consumption_ticks,
            "paired_window_sha256": self._paired_window_sha256,
            "paired_calendar_date": self._paired_calendar_date,
        }
        source_gen_capacity_mw = sum(
            max(0.0, value) for value in self._base_gen_max_p.values()
        )
        source_ext_grid_capacity_mw = sum(
            max(0.0, value) for value in self._base_ext_grid_max_p.values()
        )
        runtime_assets = [
            {
                "path": self._case_source_path,
                "sha256": self._case_source_sha256,
                "role": "runtime_input",
            }
        ]
        runtime_assets.extend(
            {
                "path": resolved,
                "sha256": self._paired_source_hashes[declared],
                "role": self._paired_asset_role(declared),
            }
            for declared, resolved in self._paired_source_assets.items()
        )
        consumed_hashes = {
            self._case_source_declared: self._case_source_sha256,
            **self._paired_source_hashes,
        }
        consumed_channels = [
            "branch_topology_and_limits",
            "bus_voltage_limits",
            "generator_capability_and_cost",
            "load_active_and_reactive_power",
        ]
        if self._paired_timeseries_contract:
            consumed_channels.append("paired_regional_load_series")
            if self._paired_timeseries_contract == CASE73_CONTRACT:
                consumed_channels.append("paired_bus_and_branch_identity")
        paired_window = bool(self._paired_window_sha256)
        return {
            "status": "passed" if state_effect else "held",
            "proof_kind": "direct_runtime_files",
            "runtime_opened_assets": runtime_assets,
            "opened_source_paths": [asset["path"] for asset in runtime_assets],
            "opened_source_sha256": {
                asset["path"]: asset["sha256"] for asset in runtime_assets
            },
            "consumed_source_hashes": consumed_hashes,
            "lineage_source_hashes": consumed_hashes,
            "consumed_window_sha256": (
                self._paired_window_sha256
                if paired_window
                else self._case_source_sha256
            ),
            "recipe_version": (
                self._paired_recipe_version
                if paired_window
                else "pandapower_from_mpc_acopf_v1"
            ),
            "parser_output_digest": self._parser_output_digest,
            "native_net_fields": dict(self._native_net_fields),
            "source_generation_capacity_mw": round(
                source_gen_capacity_mw + source_ext_grid_capacity_mw, 6
            ),
            "source_gen_capacity_mw": round(source_gen_capacity_mw, 6),
            "source_ext_grid_capacity_mw": round(
                source_ext_grid_capacity_mw, 6
            ),
            "source_capacity_basis": (
                "converted_source_gen_and_ext_grid_pmax_v1"
            ),
            "consumed_channels": consumed_channels,
            "derived_backend_state_fields": [
                "bus_voltage_pu",
                "generator_dispatch_mw",
                "line_loading_percent",
                "solver_converged",
            ],
            "consumption_ticks": sorted(
                set(([0] if state_effect else []) + self._paired_consumption_ticks)
            ),
            "initial_solver_state": initial,
            "post_source_state_digests": (
                [str(initial["state_digest"])] if state_effect else []
            ),
            "source_state_effect_observed": state_effect,
            "state_effect_observed": state_effect,
            "deterministic_source_trace": True,
            "trace_semantic_digest": self._semantic_digest(semantic_payload),
            "runtime_trace_observed": True,
            "evidence_from_scenario_config_only": False,
            "source_time_variation_claimed": paired_effect,
            "paired_timeseries_contract": self._paired_timeseries_contract,
            "paired_consumption_ticks": list(self._paired_consumption_ticks),
            "source_window": (
                {
                    "calendar_date": self._paired_calendar_date,
                    "source_periods": "1-288",
                    "recipe_version": self._paired_recipe_version,
                }
                if self._paired_timeseries_contract
                else {}
            ),
            "blockers": [] if state_effect else ["initial_solver_state_unproven"],
        }

    def _paired_asset_role(self, declared: str) -> str:
        if declared.endswith("REAL_TIME_regional_Load.csv"):
            return "runtime_input"
        if self._paired_timeseries_contract == CASE73_CONTRACT:
            return "physical_pairing_input"
        return "derivation_input"

    def _apply_gen_pins(self) -> None:
        # Restore native limits first, then pin requested gens.
        for i, lo in self._base_gen_min_p.items():
            if 0 <= i < len(self._net.gen):
                self._net.gen.at[i, "min_p_mw"] = lo
                self._net.gen.at[i, "max_p_mw"] = self._base_gen_max_p[i]
        for gen_idx, target in self._gen_pins.items():
            if 0 <= gen_idx < len(self._net.gen):
                lo = self._base_gen_min_p.get(gen_idx, 0.0)
                hi = self._base_gen_max_p.get(gen_idx, target)
                center = max(lo, min(hi, target))
                # P2-10 fix: pin as a NARROW BAND, not strict equality.
                # min_p==max_p can make the OPF infeasible (forces a unit
                # to an exact MW that may violate balance/limits), which
                # silently dropped the solve to the runpp fallback or to
                # non-convergence. A small band honors the redispatch
                # intent while leaving the OPF a feasible interval.
                band = max(1.0, 0.05 * abs(center))
                self._net.gen.at[gen_idx, "min_p_mw"] = max(lo, center - band)
                self._net.gen.at[gen_idx, "max_p_mw"] = min(hi, center + band)

    def _solve(self) -> tuple[bool, float]:
        try:
            pp.runopp(self._net, numba=False, init="flat")
            cost = float(self._net.res_cost)
            # P2-9 guard: a "converged" runopp can return a NaN/inf cost;
            # propagating it poisons economic_cost (0.5+0.5*NaN resolves to
            # a spurious perfect score). Treat as a failed solve.
            if not math.isfinite(cost):
                return False, 0.0
            return True, cost
        except Exception:
            try:
                pp.runpp(self._net, numba=False)
                cost = self._eval_poly_cost()
                if not math.isfinite(cost):
                    return False, 0.0
                return True, cost
            except Exception:
                return False, 0.0

    def _reference_solve(self) -> float | None:
        """OPF with redispatch pins RELEASED (native gen limits).

        Returns the reference optimum cost, or ``None`` if the released
        solve did not converge (so the caller can avoid polluting the
        optimality-gap denominator with a 0).
        """
        for i, lo in self._base_gen_min_p.items():
            if 0 <= i < len(self._net.gen):
                self._net.gen.at[i, "min_p_mw"] = lo
                self._net.gen.at[i, "max_p_mw"] = self._base_gen_max_p[i]
        ref: float | None
        try:
            pp.runopp(self._net, numba=False, init="flat")
            ref = float(self._net.res_cost)
            if not math.isfinite(ref):
                ref = None
        except Exception:
            ref = None
        # Re-apply pins so the realized state stands for snapshot/extraction.
        self._apply_gen_pins()
        with contextlib.suppress(Exception):
            pp.runopp(self._net, numba=False, init="flat")
        return ref

    def _eval_poly_cost(self) -> float:
        """Evaluate the polynomial cost on the current PF dispatch."""
        total = 0.0
        try:
            for _, row in self._net.poly_cost.iterrows():
                et, el = row.get("et"), int(row.get("element"))
                if et == "gen" and el < len(self._net.res_gen):
                    p = float(self._net.res_gen.p_mw.iloc[el])
                elif et == "ext_grid" and el < len(self._net.res_ext_grid):
                    p = float(self._net.res_ext_grid.p_mw.iloc[el])
                else:
                    continue
                total += (
                    float(row.get("cp0_eur", 0.0))
                    + float(row.get("cp1_eur_per_mw", 0.0)) * p
                    + float(row.get("cp2_eur_per_mw2", 0.0)) * p * p
                )
        except Exception:
            return 0.0
        return total

    def _build_record(
        self,
        current_tick: int,
        converged: bool,
        prod_cost: float,
        events: list[dict[str, Any]],
        tick_h: float,
    ) -> PandapowerAcopfTickRecord:
        if converged:
            demand_mw = float(self._net.load.p_mw.sum())
            gen_mw = 0.0
            if "p_mw" in self._net.res_gen.columns:
                gen_mw += float(self._net.res_gen.p_mw.sum())
            if "p_mw" in self._net.res_sgen.columns:
                gen_mw += float(self._net.res_sgen.p_mw.sum())
            if "p_mw" in self._net.res_ext_grid.columns:
                gen_mw += float(self._net.res_ext_grid.p_mw.sum())
            line_load = (
                self._net.res_line.loading_percent.to_numpy()
                if len(self._net.res_line)
                else []
            )
            trafo_load = (
                self._net.res_trafo.loading_percent.to_numpy()
                if len(self._net.res_trafo)
                else []
            )
            import numpy as np

            all_load = (
                np.concatenate([np.asarray(line_load), np.asarray(trafo_load)])
                if (len(line_load) or len(trafo_load))
                else np.asarray([0.0])
            )
            # Guard all-NaN loading (a degenerate "converged" solve where
            # branch loadings come back NaN). Treat as worst-case so the
            # safety dimensions SEE the failure instead of reading NaN
            # (NaN compares False against the >1.0 threshold and would
            # silently score the tick as safe).
            finite_load = all_load[np.isfinite(all_load)]
            if finite_load.size:
                rho_max = float(np.max(finite_load)) / 100.0
                n_overloads = int(np.sum(finite_load > 100.0))
            else:
                converged = False
                rho_max = 2.0
                n_overloads = len(self._net.line)
            v_pu = self._net.res_bus.vm_pu.to_numpy()
            # P1-3 guard: NaN bus voltages (a degenerate "converged"
            # solve). NaN comparisons are False, so the violation count
            # would silently read 0 — undercounting a voltage collapse.
            # Treat all-NaN voltages as a failed solve (catastrophic).
            finite_v = v_pu[np.isfinite(v_pu)]
            if finite_v.size == 0:
                converged = False
                n_voltage = len(self._net.bus)
            else:
                # Use the case's OWN per-bus operational band (pglib
                # transmission cases ship vmin/vmax, often 0.94/1.06). A
                # "violation" = outside the case's actual limits, not a
                # hardcoded distribution band — otherwise a feasible OPF
                # at 1.06 pu would be false-flagged against a 1.05 ceiling.
                try:
                    vmin = self._net.bus.min_vm_pu.to_numpy()
                    vmax = self._net.bus.max_vm_pu.to_numpy()
                    finite_mask = np.isfinite(v_pu)
                    # Small tolerance so a solution exactly at the OPF
                    # limit is not counted as a violation.
                    below = (v_pu < vmin - 1e-3) & finite_mask
                    above = (v_pu > vmax + 1e-3) & finite_mask
                    # Any non-finite bus voltage is itself a violation.
                    n_voltage = int((below | above | ~finite_mask).sum())
                except Exception:
                    n_voltage = int(
                        (
                            (v_pu < self.VOLTAGE_LOWER_PU)
                            | (v_pu > self.VOLTAGE_UPPER_PU)
                        ).sum()
                    )
            n_disc = int((~self._net.line.in_service.to_numpy()).sum())
        if not converged:
            # P1-3: a non-converged AC solve is a real grid failure
            # (≈ voltage collapse / infeasible dispatch). Set catastrophic
            # sentinels so score_system_survival SEES it (n_voltage>=5 and
            # rho_max>1 fire the catastrophic path) instead of carrying
            # forward the previous healthy tick's observables.
            demand_mw = float(self._net.load.p_mw.sum())
            last = self._tick_records[-1] if self._tick_records else None
            gen_mw = last.aggregate_generation_mw if last else 0.0
            rho_max = 2.0
            n_overloads = max(len(self._net.line), 1)
            n_voltage = max(len(self._net.bus), 5)
            n_disc = (
                last.n_disconnected_lines
                if last
                else int((~self._net.line.in_service.to_numpy()).sum())
            )

        shed_penalty = 0.0
        for entry in self._loads.values():
            sh = float(entry.get("shed_this_tick_mw", 0.0))
            if sh <= 0:
                continue
            tariff = self.SHED_PENALTY_PER_MWH_BY_CLASS.get(
                str(entry.get("stakeholder_class", "")), self.SHED_PENALTY_DEFAULT
            )
            shed_penalty += sh * tariff * tick_h

        reserves_required = self.RESERVE_TARGET_FRACTION_OF_DEMAND * demand_mw
        reserve_capacity = self._reserve_capacity_state()
        physical_headroom_mw = float(reserve_capacity["physical_headroom_mw"])
        # A commitment allocates existing source-native headroom; it never
        # adds MW on top of that same headroom.  Mutual aid shares the pending
        # reserve ledger and is bounded by this physical delivery limit too.
        reserves_procured = min(
            max(0.0, self._pending_reserve_extra_mw), physical_headroom_mw
        )
        reserve_lever = self._reserve_decision_lever()
        if reserve_lever is not None:
            reserves_required = max(
                reserves_required,
                self._reserve_required_for_lever(reserve_lever, current_tick),
            )
            procurement_cost = (
                min(self._committed_reserve_mw, physical_headroom_mw)
                * float(reserve_lever["procurement_cost_per_mw_tick"])
                * tick_h
            )
            shortfall_cost = (
                max(0.0, reserves_required - reserves_procured)
                * float(reserve_lever["shortfall_cost_per_mw_tick"])
                * tick_h
            )
            self._reserve_procurement_cost_total += procurement_cost
            self._reserve_shortfall_cost_total += shortfall_cost
        # P1-2 fix: on a converged AC-OPF the system is balanced BY
        # CONSTRUCTION — generation = demand + network losses. The raw
        # gen-demand difference is therefore ~losses (2-5% of demand),
        # NOT a supply/demand imbalance; charging it against the scorer's
        # >100/>200 MW thresholds spuriously penalised large cases (e.g.
        # case118) every tick, uniformly across agents. Subtract modeled
        # losses so balance_error reflects true unserved/over-served
        # energy (~0 when converged). A non-converged solve is the real
        # imbalance signal: report the full served demand as the error so
        # system_survival's >200 MW catastrophic path also fires.
        if converged:
            try:
                losses_mw = 0.0
                if len(self._net.res_line) and "pl_mw" in self._net.res_line.columns:
                    losses_mw += float(self._net.res_line.pl_mw.sum())
                if len(self._net.res_trafo) and "pl_mw" in self._net.res_trafo.columns:
                    losses_mw += float(self._net.res_trafo.pl_mw.sum())
                if not math.isfinite(losses_mw):
                    losses_mw = 0.0
            except Exception:
                losses_mw = 0.0
            balance_error = gen_mw - demand_mw - losses_mw
        else:
            balance_error = -demand_mw
        self._last_balance_error = balance_error

        return PandapowerAcopfTickRecord(
            tick=current_tick,
            aggregate_demand_mw=round(demand_mw, 2),
            aggregate_generation_mw=round(gen_mw, 2),
            balance_error_mw=round(balance_error, 2),
            reserves_required_mw=round(reserves_required, 2),
            reserves_procured_mw=round(reserves_procured, 2),
            production_cost=round(prod_cost, 2),
            startup_cost=0.0,
            shed_penalty=round(shed_penalty, 2),
            rho_max=round(rho_max, 4),
            n_overloads=n_overloads,
            n_voltage_violations=n_voltage,
            n_disconnected_lines=n_disc,
            converged=converged,
            done=current_tick >= self._horizon - 1,
            realized_events=events,
        )

    def _reserve_decision_lever(self) -> dict[str, float] | None:
        if self._seed_obj is None:
            return None
        if "acopf_reserve_decision_lever" not in self._seed_obj.backend_config:
            return None
        raw = self._seed_obj.backend_config["acopf_reserve_decision_lever"]
        if not isinstance(raw, dict):
            raise ValueError("acopf_reserve_decision_lever must be a non-empty mapping")
        if not raw:
            raise ValueError("acopf_reserve_decision_lever must be a non-empty mapping")
        required = {
            "window_start_tick",
            "window_duration_ticks",
            "required_mw",
            "shortfall_cost_per_mw_tick",
            "procurement_cost_per_mw_tick",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(
                "acopf_reserve_decision_lever missing required keys: "
                + ", ".join(missing)
            )
        window_start_raw = raw["window_start_tick"]
        window_duration_raw = raw["window_duration_ticks"]
        if isinstance(window_start_raw, bool) or not isinstance(window_start_raw, int):
            raise ValueError(
                "acopf_reserve_decision_lever window_start_tick must be int"
            )
        if isinstance(window_duration_raw, bool) or not isinstance(
            window_duration_raw, int
        ):
            raise ValueError(
                "acopf_reserve_decision_lever window_duration_ticks must be int"
            )
        window_start = int(window_start_raw)
        window_duration = int(window_duration_raw)
        required_mw = float(raw["required_mw"])
        shortfall_cost = float(raw["shortfall_cost_per_mw_tick"])
        procurement_cost = float(raw["procurement_cost_per_mw_tick"])
        if window_start < 0:
            raise ValueError("acopf_reserve_decision_lever window_start_tick < 0")
        if window_duration <= 0:
            raise ValueError(
                "acopf_reserve_decision_lever window_duration_ticks must be positive"
            )
        if window_start >= self._horizon:
            raise ValueError(
                "acopf_reserve_decision_lever window must overlap the scenario horizon"
            )
        if required_mw <= 0:
            raise ValueError(
                "acopf_reserve_decision_lever required_mw must be positive"
            )
        if shortfall_cost <= 0:
            raise ValueError(
                "acopf_reserve_decision_lever "
                "shortfall_cost_per_mw_tick must be positive"
            )
        if procurement_cost < 0:
            raise ValueError(
                "acopf_reserve_decision_lever "
                "procurement_cost_per_mw_tick must be non-negative"
            )
        return {
            "window_start_tick": float(window_start),
            "window_duration_ticks": float(window_duration),
            "required_mw": required_mw,
            "shortfall_cost_per_mw_tick": shortfall_cost,
            "procurement_cost_per_mw_tick": procurement_cost,
        }

    @staticmethod
    def _reserve_required_for_lever(
        lever: dict[str, float], current_tick: int
    ) -> float:
        start = int(lever["window_start_tick"])
        end = start + int(lever["window_duration_ticks"])
        if start <= current_tick < end:
            return float(lever["required_mw"])
        return 0.0

    def _emergency_reserve_protection(self) -> dict[str, Any] | None:
        if self._seed_obj is None:
            return None
        if "acopf_emergency_reserve_protection" not in self._seed_obj.backend_config:
            return None
        raw = self._seed_obj.backend_config["acopf_emergency_reserve_protection"]
        if not isinstance(raw, dict):
            raise ValueError(
                "acopf_emergency_reserve_protection must be a non-empty mapping"
            )
        if not raw:
            raise ValueError(
                "acopf_emergency_reserve_protection must be a non-empty mapping"
            )
        required = {
            "window_start_tick",
            "window_duration_ticks",
            "required_mw",
            "shed_mw_per_tick",
            "protected_load_classes",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(
                "acopf_emergency_reserve_protection missing required keys: "
                + ", ".join(missing)
            )
        window_start_raw = raw["window_start_tick"]
        window_duration_raw = raw["window_duration_ticks"]
        if isinstance(window_start_raw, bool) or not isinstance(window_start_raw, int):
            raise ValueError(
                "acopf_emergency_reserve_protection window_start_tick must be int"
            )
        if isinstance(window_duration_raw, bool) or not isinstance(
            window_duration_raw, int
        ):
            raise ValueError(
                "acopf_emergency_reserve_protection window_duration_ticks must be int"
            )
        window_start = int(window_start_raw)
        window_duration = int(window_duration_raw)
        required_mw = float(raw["required_mw"])
        shed_mw = float(raw["shed_mw_per_tick"])
        classes = raw["protected_load_classes"]
        if window_start < 0:
            raise ValueError("acopf_emergency_reserve_protection window_start_tick < 0")
        if window_duration <= 0:
            raise ValueError(
                "acopf_emergency_reserve_protection "
                "window_duration_ticks must be positive"
            )
        if window_start >= self._horizon:
            raise ValueError(
                "acopf_emergency_reserve_protection window must overlap the horizon"
            )
        if required_mw <= 0:
            raise ValueError(
                "acopf_emergency_reserve_protection required_mw must be positive"
            )
        if shed_mw <= 0:
            raise ValueError(
                "acopf_emergency_reserve_protection shed_mw_per_tick must be positive"
            )
        if not isinstance(classes, list) or not classes:
            raise ValueError(
                "acopf_emergency_reserve_protection "
                "protected_load_classes must be a non-empty list"
            )
        normalized_classes = [str(cls) for cls in classes]
        if any(not cls for cls in normalized_classes):
            raise ValueError(
                "acopf_emergency_reserve_protection "
                "protected_load_classes must contain non-empty strings"
            )
        return {
            "window_start_tick": window_start,
            "window_duration_ticks": window_duration,
            "required_mw": required_mw,
            "shed_mw_per_tick": shed_mw,
            "protected_load_classes": normalized_classes,
        }

    def _apply_emergency_reserve_protection(
        self,
        *,
        current_tick: int,
        tick_h: float,
        diurnal: float,
        events: list[dict[str, Any]],
    ) -> None:
        protection = self._emergency_reserve_protection()
        if protection is None:
            return
        start = int(protection["window_start_tick"])
        end = start + int(protection["window_duration_ticks"])
        if not (start <= current_tick < end):
            return
        required_mw = float(protection["required_mw"])
        reserve_capacity = self._reserve_capacity_state()
        deliverable_committed_mw = min(
            self._committed_reserve_mw,
            float(reserve_capacity["physical_headroom_mw"]),
        )
        reserve_shortfall = max(0.0, required_mw - deliverable_committed_mw)
        if reserve_shortfall <= 0.0:
            return

        remaining_shed_mw = float(protection["shed_mw_per_tick"])
        target_classes = set(protection["protected_load_classes"])
        shed_by_load: dict[str, float] = {}
        candidates = sorted(
            (
                (lid, entry)
                for lid, entry in self._loads.items()
                if str(entry.get("stakeholder_class")) in target_classes
            ),
            key=lambda item: (
                list(protection["protected_load_classes"]).index(
                    str(item[1].get("stakeholder_class"))
                ),
                int(item[1].get("load_index", 0)),
            ),
        )
        for load_id, entry in candidates:
            if remaining_shed_mw <= 0.0:
                break
            idx = int(entry["load_index"])
            cls = str(entry.get("stakeholder_class", ""))
            surge = 1.0 + self._class_surge_factors.get(cls, 0.0)
            demand_mw = max(0.0, self._base_load_p_mw.get(idx, 0.0) * diurnal * surge)
            shed_mw = min(remaining_shed_mw, demand_mw)
            if shed_mw <= 0.0:
                continue
            entry["shed_this_tick_mw"] += shed_mw
            self._cumulative_shed_mwh[load_id] += shed_mw * tick_h
            shed_by_load[load_id] = round(shed_mw, 3)
            remaining_shed_mw -= shed_mw

        total_shed = sum(shed_by_load.values())
        if total_shed <= 0.0:
            return
        events.append(
            {
                "type": "emergency_reserve_protection_failed",
                "event_class": "safety",
                "decision_required": current_tick + 1 < self._horizon,
                "actionable": current_tick + 1 < self._horizon,
                "tick": current_tick,
                "reserve_shortfall_mw": round(reserve_shortfall, 3),
                "required_mw": round(required_mw, 3),
                "committed_reserve_mw": round(self._committed_reserve_mw, 3),
                "deliverable_committed_reserve_mw": round(
                    deliverable_committed_mw, 3
                ),
                "physical_headroom_mw": round(
                    float(reserve_capacity["physical_headroom_mw"]), 3
                ),
                "shed_mw": round(total_shed, 3),
                "shed_by_load": shed_by_load,
                "protected_load_classes": list(protection["protected_load_classes"]),
            }
        )

    def _apply_gen_outages(self, tick: int) -> None:
        for i in range(len(self._net.gen)):
            until = self._gen_outage_until.get(i)
            if until is not None and tick < until:
                self._net.gen.at[i, "in_service"] = False
            else:
                self._net.gen.at[i, "in_service"] = True

    # ── Perturbations ───────────────────────────────────────────────────

    @staticmethod
    def _validate_perturbation_kind(perturbation: Any) -> str:
        kind = str(getattr(perturbation, "kind", "") or "")
        try:
            return ACOPF_PERTURBATION_KIND_REGISTRY[kind]
        except KeyError as exc:
            raise ValueError(
                f"unsupported ACOPF perturbation kind: {kind or '<missing>'}"
            ) from exc

    def _declared_perturbation_event(
        self,
        *,
        perturbation_index: int,
        perturbation: Any,
        event_type: str,
        tick: int,
        changed_state_fields: list[str],
        materiality_metric: str,
        materiality_value: float,
        materiality_threshold: float,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Describe a seeded procedural change after the backend applies it.

        PGLib-OPF contributes the native network and OPF physics; these
        disruptions are intentionally *procedural* variants over that locked
        source, not source-observed time series.  The explicit origin keeps
        that distinction machine-readable while giving the agentic gate a
        runtime event with concrete state fields to audit.
        """
        start = int(perturbation.trigger_tick)
        try:
            event_class = ACOPF_PERTURBATION_EVENT_CLASS[event_type]
        except KeyError as exc:
            raise ValueError(
                f"unknown ACOPF perturbation event type: {event_type}"
            ) from exc
        actionable = (
            tick + 1 < self._horizon and not bool(perturbation.hidden)
        )
        return {
            "event_id": f"acopf-procedural:{perturbation_index}:{event_type}:{start}",
            "type": event_type,
            "tick": tick,
            "origin": "declared_perturbation",
            "event_class": event_class,
            "declared_perturbation": True,
            "declared_event": {
                "kind": str(perturbation.kind),
                "trigger_tick": start,
                "duration_ticks": int(perturbation.duration_ticks),
                "procedural_variant": True,
            },
            "hidden": bool(perturbation.hidden),
            "decision_required": actionable,
            "actionable": actionable,
            "response_window_required": actionable,
            "response_opportunity_tick": tick + 1 if actionable else None,
            "changed_state_fields": changed_state_fields,
            "materiality_metric": materiality_metric,
            "materiality_value": materiality_value,
            "materiality_threshold": materiality_threshold,
            "materiality_passed": abs(materiality_value) >= abs(
                materiality_threshold
            ),
            **payload,
        }

    def _apply_perturbations_at_tick(self, tick: int) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if self._seed_obj is None:
            return events
        for perturbation_index, p in enumerate(self._seed_obj.perturbations):
            start = int(p.trigger_tick)
            end = start + max(1, int(p.duration_ticks))
            if not (start <= tick < end):
                if tick == end and p.kind == "line_outage":
                    line_id = int(p.target.get("line_index", 0))
                    self._line_status_overrides[line_id] = True
                    events.append(
                        {
                            "type": "line_restored",
                            "event_class": "lifecycle",
                            "tick": tick,
                            "line_id": line_id,
                            "origin": "endogenous_completion",
                            "decision_required": False,
                            "actionable": False,
                            "changed_state_fields": ["line_in_service"],
                        }
                    )
                continue
            if p.kind == "line_outage":
                if tick == start:
                    line_id = int(p.target.get("line_index", 0)) % max(
                        len(self._net.line), 1
                    )
                    self._line_status_overrides[line_id] = False
                    events.append(
                        self._declared_perturbation_event(
                            perturbation_index=perturbation_index,
                            perturbation=p,
                            event_type="line_outage",
                            tick=tick,
                            changed_state_fields=[
                                "line_in_service",
                                "line_loading_percent",
                                "bus_voltage_pu",
                            ],
                            materiality_metric="outaged_line_count",
                            materiality_value=1.0,
                            materiality_threshold=1.0,
                            payload={
                                "line_id": line_id,
                                "intensity": p.intensity,
                            },
                        )
                    )
            elif p.kind == "generator_forced_outage":
                gen_idx = int(p.target.get("index", 0)) % max(len(self._net.gen), 1)
                self._gen_outage_until[gen_idx] = end
                if tick == start:
                    events.append(
                        self._declared_perturbation_event(
                            perturbation_index=perturbation_index,
                            perturbation=p,
                            event_type="generator_outage",
                            tick=tick,
                            changed_state_fields=[
                                "generator_availability",
                                "aggregate_generation_mw",
                                "reserves_procured_mw",
                            ],
                            materiality_metric="outaged_generator_count",
                            materiality_value=1.0,
                            materiality_threshold=1.0,
                            payload={
                                "generator_id": f"gen_{gen_idx}",
                                "intensity": p.intensity,
                            },
                        )
                    )
            elif p.kind == "planned_maintenance":
                fraction = max(
                    0.0,
                    min(1.0, float(p.target.get("fraction", 0.05) or 0.05)),
                )
                count = min(
                    len(self._net.gen),
                    max(1, round(len(self._net.gen) * fraction)),
                )
                ordered = sorted(
                    range(len(self._net.gen)),
                    key=lambda index: (
                        float(self._base_gen_max_p.get(index, 0.0)),
                        index,
                    ),
                )
                selected = ordered[:count]
                for gen_idx in selected:
                    self._gen_outage_until[gen_idx] = end
                if tick == start:
                    events.append(
                        self._declared_perturbation_event(
                            perturbation_index=perturbation_index,
                            perturbation=p,
                            event_type="planned_maintenance",
                            tick=tick,
                            changed_state_fields=[
                                "generator_availability",
                                "aggregate_generation_mw",
                                "reserves_procured_mw",
                            ],
                            materiality_metric="maintenance_generator_count",
                            materiality_value=float(len(selected)),
                            materiality_threshold=1.0,
                            payload={
                                "generator_ids": [
                                    f"gen_{gen_idx}" for gen_idx in selected
                                ],
                                "fraction": fraction,
                            },
                        )
                    )
            elif p.kind == "load_surge":
                target_class = p.target.get("stakeholder_class")
                if target_class:
                    self._class_surge_factors[str(target_class)] = float(p.intensity)
                else:
                    for cls in self.SHED_PENALTY_PER_MWH_BY_CLASS:
                        self._class_surge_factors[cls] = float(p.intensity)
                if tick == start:
                    events.append(
                        self._declared_perturbation_event(
                            perturbation_index=perturbation_index,
                            perturbation=p,
                            event_type="load_surge",
                            tick=tick,
                            changed_state_fields=[
                                "aggregate_demand_mw",
                                "line_loading_percent",
                                "reserves_procured_mw",
                            ],
                            materiality_metric="demand_surge_fraction",
                            materiality_value=abs(float(p.intensity)),
                            materiality_threshold=0.01,
                            payload={
                                "intensity": p.intensity,
                                "stakeholder_class": target_class,
                            },
                        )
                    )
            elif p.kind == "forecast_bias":
                direction = p.target.get("bias_direction", "under-forecast")
                sign = 1.0 if direction == "under-forecast" else -1.0
                self._forecast_bias = sign * float(p.intensity)
                if tick == start:
                    events.append(
                        self._declared_perturbation_event(
                            perturbation_index=perturbation_index,
                            perturbation=p,
                            event_type="forecast_bias",
                            tick=tick,
                            changed_state_fields=["demand_forecast_mw"],
                            materiality_metric="forecast_bias_fraction",
                            materiality_value=abs(self._forecast_bias),
                            materiality_threshold=0.01,
                            payload={"bias_direction": direction},
                        )
                    )
            elif p.kind == "storm_window":
                actionable = (
                    tick == start
                    and tick + 1 < self._horizon
                    and not bool(p.hidden)
                )
                events.append(
                    {
                        "type": "storm_window",
                        "event_class": ACOPF_PERTURBATION_EVENT_CLASS[
                            "storm_window"
                        ],
                        "tick": tick,
                        "intensity": p.intensity,
                        "hidden": p.hidden,
                        "origin": "declared_perturbation",
                        "declared_perturbation": True,
                        "decision_required": actionable,
                        "actionable": actionable,
                        "response_window_required": actionable,
                        "response_opportunity_tick": (
                            tick + 1 if actionable else None
                        ),
                    }
                )
        return events

    # ── Snapshot ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        if self._net is None or self._seed_obj is None:
            return {"entities": {}, "totals": {}}
        entities: dict[str, dict[str, Any]] = {}
        for lid, entry in self._loads.items():
            idx = int(entry["load_index"])
            entities[lid] = {
                "kind": "load",
                "bus_id": int(self._net.load.bus.iloc[idx]),
                "current_demand_mw": float(self._net.load.p_mw.iloc[idx]),
                "stakeholder_class": entry["stakeholder_class"],
                "criticality": entry["criticality"],
                "cumulative_shed_mwh": round(
                    self._cumulative_shed_mwh.get(lid, 0.0), 3
                ),
            }
        for i in range(len(self._net.gen)):
            in_svc = bool(self._net.gen.in_service.iloc[i])
            actual_dispatch_mw: float | None = None
            if self._net.res_gen is not None and i < len(self._net.res_gen):
                with contextlib.suppress(Exception):
                    actual_dispatch_mw = round(float(self._net.res_gen.p_mw.iloc[i]), 6)
            entities[f"gen_{i}"] = {
                "kind": "generator",
                "bus_id": int(self._net.gen.bus.iloc[i]),
                "power_max": float(self._base_gen_max_p.get(i, 0.0)),
                "committed": in_svc,
                "forced_outage_until": int(self._gen_outage_until.get(i, -1)),
                "commanded_target_mw": self._gen_pins.get(i),
                "actual_dispatch_mw": actual_dispatch_mw,
            }
        entities["reserve_commitment"] = {
            "kind": "reserve_commitment",
            "committed_reserve_mw": round(self._committed_reserve_mw, 6),
            "pending_reserve_mw": round(self._pending_reserve_extra_mw, 6),
            **{
                key: round(float(value), 6) if isinstance(value, float) else value
                for key, value in self._reserve_capacity_state().items()
            },
            "source_case_sha256": self._case_source_sha256,
        }
        # Bus voltages (post-solve, if available).
        if self._net.res_bus is not None and len(self._net.res_bus) > 0:
            for i in range(len(self._net.bus)):
                try:
                    vm = float(self._net.res_bus.vm_pu.iloc[i])
                except Exception:
                    continue
                entities[f"bus_{i}"] = {"kind": "bus", "vm_pu": round(vm, 4)}
        totals = {}
        if self._tick_records:
            r = self._tick_records[-1]
            totals = {
                "aggregate_demand_mw": r.aggregate_demand_mw,
                "aggregate_generation_mw": r.aggregate_generation_mw,
                "balance_error_mw": r.balance_error_mw,
                "reserves_required_mw": r.reserves_required_mw,
                "reserves_procured_mw": r.reserves_procured_mw,
                "committed_reserve_mw": round(self._committed_reserve_mw, 6),
                "pending_reserve_mw": round(self._pending_reserve_extra_mw, 6),
                "physical_headroom_mw": entities["reserve_commitment"][
                    "physical_headroom_mw"
                ],
                "source_generation_capacity_mw": entities["reserve_commitment"][
                    "source_generation_capacity_mw"
                ],
                "rho_max": r.rho_max,
                "n_voltage_violations": r.n_voltage_violations,
            }
        return {"entities": entities, "totals": totals}

    # ── Cost / scoring surfaces ─────────────────────────────────────────

    def ground_truth_costs(self) -> dict[str, float]:
        if not self._tick_records:
            out = {
                "production_cost": 0.0,
                "shed_penalty": 0.0,
                "voltage_violation_cost": 0.0,
                "overload_cost": 0.0,
                "disconnection_cost": 0.0,
            }
            if self._reserve_decision_lever() is not None:
                out["reserve_procurement_cost"] = 0.0
                out["reserve_shortfall_cost"] = 0.0
            return out
        prod = sum(r.production_cost for r in self._tick_records)
        shed = sum(r.shed_penalty for r in self._tick_records)
        volt = sum(
            r.n_voltage_violations * self.VOLTAGE_VIOLATION_COST_PER_TICK
            for r in self._tick_records
        )
        over = sum(
            r.n_overloads * self.OVERLOAD_COST_PER_TICK for r in self._tick_records
        )
        disc = sum(
            r.n_disconnected_lines * self.DISCONNECTION_COST_PER_LINE_TICK
            for r in self._tick_records
        )
        out = {
            "production_cost": round(prod, 2),
            "shed_penalty": round(shed, 2),
            "voltage_violation_cost": round(volt, 2),
            "overload_cost": round(over, 2),
            "disconnection_cost": round(disc, 2),
        }
        if self._reserve_decision_lever() is not None:
            out["reserve_procurement_cost"] = round(
                self._reserve_procurement_cost_total, 2
            )
            out["reserve_shortfall_cost"] = round(self._reserve_shortfall_cost_total, 2)
        return out

    def scoring_records(self) -> list[dict[str, Any]]:
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
                "rho_max": r.rho_max,
                "n_overloads": r.n_overloads,
                "n_voltage_violations": r.n_voltage_violations,
                "n_disconnected_lines": r.n_disconnected_lines,
                # The scorer treats ``done`` as a catastrophic early terminal
                # signal.  Natural horizon completion is owned here because
                # this backend knows the configured horizon; do not ask the
                # backend-agnostic scorer to infer it from list position.
                "done": bool(r.done and r.tick < self._horizon - 1),
                "converged": r.converged,
            }
            for r in self._tick_records
        ]

    def per_load_shed_mwh(self) -> dict[str, float]:
        return {lid: round(v, 3) for lid, v in self._cumulative_shed_mwh.items()}

    def acopf_reference_optimum(self) -> float:
        """Sum of per-tick reference OPF optima (pins released) — the
        TRUE AC-OPF optimality-gap denominator. Feed via
        ``ScoringInputs.lp_optimum``."""
        return round(self._reference_optimum_total, 2)

    def forecast_for(self, horizon: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        base_total = sum(self._base_load_p_mw.values())
        peak_tick = max(1, int(self._horizon * 0.7))
        for t in range(self._tick, min(self._tick + horizon, self._horizon)):
            diurnal = max(
                0.6, min(1.20, 0.85 + 0.30 * math.sin(math.pi * t / peak_tick))
            )
            true_d = base_total * diurnal
            biased = true_d * (1.0 - self._forecast_bias)
            out.append(
                {
                    "tick": t,
                    "demand_mw_forecast": round(biased, 2),
                    "forecast_bias": round(self._forecast_bias, 4),
                }
            )
        return out
