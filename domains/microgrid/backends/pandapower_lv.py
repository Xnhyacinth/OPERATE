"""
domains.microgrid.backends.pandapower_lv — LV power-flow EMS tier.

The ``microgrid_lv_voltage_6h`` family's backend. Built on pandapower's
public **synthetic voltage-control LV network**
(``create_synthetic_voltage_control_lv_network``; BSD-3-Clause, already in
tree) and solves AC power flow each tick via ``pp.runpp``. Rooftop PV is
scaled so mid-horizon reverse power flow drives **over-voltage**; the agent
holds the [0.95, 1.05] pu band by curtailing PV (``curtail_der``), absorbing
reactive power (``set_der_reactive_power``), charging the battery
(``set_battery_dispatch``) or shedding load.

Unlike the pymgrid (aggregate energy-balance) families, this tier fills the
four AC power-flow keys **honestly** from the real solve: ``rho_max``
(``res_line.loading_percent/100``), ``n_overloads``, ``n_voltage_violations``
(buses outside [0.95, 1.05] pu), ``n_disconnected_lines``. ``startup_cost``
is honestly 0 (no genset on the LV feeder).

The contract surface mirrors the EMS simulator + the power-grid
``CigreDistributionBackend`` (reset / tick / snapshot / apply_tool_effect /
ground_truth_costs / scoring_records / per_load_shed_mwh / forecast_for) so
the adapter and scorer reuse it without modification. This is a
microgrid-domain file; it does NOT import or modify any power-grid module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import pandapower as pp  # type: ignore[import-untyped]

if TYPE_CHECKING:  # pragma: no cover
    from ..seeds.schema import MicrogridScenarioSeed

_SHED_TARIFF_BY_CLASS: dict[str, float] = {
    "hospital": 5000.0,
    "water": 2500.0,
    "data_center": 1500.0,
    "commercial": 400.0,
    "residential": 200.0,
}
_SHED_TARIFF_DEFAULT = 1000.0
PANDAPOWER_LV_PERTURBATION_EVENT_CLASS = MappingProxyType(
    {
        "pv_ramp": "alarm",
        "der_failure": "alarm",
        "load_spike": "alarm",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass
class _LvTickRecord:
    tick: int
    aggregate_demand_mw: float
    aggregate_generation_mw: float
    balance_error_mw: float
    reserves_required_mw: float
    reserves_procured_mw: float
    production_cost: float
    startup_cost: float
    shed_penalty: float
    rho_max: float
    n_overloads: int
    n_voltage_violations: int
    n_disconnected_lines: int
    converged: bool = True
    done: bool = False
    realized_events: list[dict[str, Any]] = field(default_factory=list)


class PandapowerLvBackend:
    """pandapower synthetic-LV voltage-control backend (real AC power flow)."""

    backend_kind = "pandapower_lv"
    supported_tool_names = frozenset(
        {
            "set_battery_dispatch",
            "curtail_der",
            "shed_load",
            "set_der_reactive_power",
            "forecast_query",
            "investigate_asset",
            "commit_to_plan",
            "wait",
            "noop",
        }
    )

    PRODUCTION_COST_PER_MWH = 50.0
    OVERLOAD_COST_PER_TICK = 200.0
    VOLTAGE_VIOLATION_COST_PER_TICK = 1200.0
    DISCONNECTION_COST_PER_LINE_TICK = 500.0
    RESERVE_TARGET_FRACTION_OF_DEMAND = 0.10
    VOLTAGE_LOWER_PU = 0.95
    VOLTAGE_UPPER_PU = 1.05

    def __init__(self) -> None:
        self._net: Any = None
        self._seed_obj: MicrogridScenarioSeed | None = None
        self._tick = 0
        self._horizon = 6
        self._tick_minutes = 60
        self._pv_scale = 1.0
        self._active_pv_scale = 1.0
        self._active_load_multiplier = 1.0
        self._failed_sgen_indices: set[int] = set()
        self._base_load_p_mw: dict[int, float] = {}
        self._base_load_q_mvar: dict[int, float] = {}
        self._base_sgen_p_mw: dict[int, float] = {}
        self._loads: dict[str, dict[str, Any]] = {}
        self._idx_to_load_id: dict[int, str] = {}
        self._der_to_sgen: dict[str, int] = {}
        self._der_q_limits: dict[int, float] = {}
        self._cumulative_shed_mwh: dict[str, float] = {}
        self._pending_curtail: dict[int, float] = {}
        self._der_q_targets: dict[int, float] = {}
        self._storage_idx: int | None = None
        self._storage_p_target = 0.0
        self._storage_p_applied = 0.0
        self._storage_capacity_mwh = 0.05
        self._storage_energy_mwh = 0.025
        self._storage_max_charge_mw = 0.025
        self._storage_max_discharge_mw = 0.025
        self._storage_efficiency = 0.95
        self._source_profile_applied = False
        self._source_profile_start_index = 0
        self._source_load_mw: list[float] = []
        self._source_pv_mw: list[float] = []
        self._source_load_reference_mw = 1.0
        self._source_pv_reference_mw = 1.0
        self._forecast_bias = 0.0
        self._tick_records: list[_LvTickRecord] = []
        self._applied_control_records: list[dict[str, Any]] = []
        self._realized_events_this_tick: list[dict[str, Any]] = []
        self._revealed_assets: set[str] = set()
        self._runtime_source_assets: list[dict[str, str]] = []
        self._runtime_source_verified = False
        self._source_consumption_ticks: list[int] = []
        self._runtime_source_events: list[dict[str, Any]] = []
        self._post_source_state_digests: list[dict[str, Any]] = []
        self._initial_source_state_digest = ""

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self, scenario_seed: MicrogridScenarioSeed) -> None:
        import pandapower.networks as pn  # type: ignore[import-untyped]

        self._seed_obj = scenario_seed
        self._tick = 0
        self._horizon = int(scenario_seed.horizon_ticks)
        self._tick_minutes = int(scenario_seed.tick_minutes)
        cfg = scenario_seed.backend_config or {}
        self._pv_scale = float(cfg.get("pv_scale", 6.0) or 6.0)
        self._active_pv_scale = self._pv_scale
        self._active_load_multiplier = 1.0
        self._failed_sgen_indices = set()
        self._forecast_bias = float(cfg.get("forecast_bias", 0.0) or 0.0)
        self._source_profile_applied = bool(cfg.get("source_profile_applied", False))
        self._source_profile_start_index = int(cfg.get("profile_start_index", 0) or 0)
        self._source_load_mw = []
        self._source_pv_mw = []
        self._source_load_reference_mw = 1.0
        self._source_pv_reference_mw = 1.0
        self._runtime_source_assets = []
        self._runtime_source_verified = False
        self._source_consumption_ticks = []
        self._runtime_source_events = []
        self._post_source_state_digests = []
        self._initial_source_state_digest = ""
        if self._source_profile_applied:
            profiles = cfg.get("source_profiles") or {}
            references = cfg.get("source_profile_reference") or {}
            self._source_load_mw = [
                float(value) for value in profiles.get("load_mw") or []
            ]
            self._source_pv_mw = [
                float(value) for value in profiles.get("pv_mw") or []
            ]
            if (
                len(self._source_load_mw) < self._horizon
                or len(self._source_pv_mw) < self._horizon
                or not all(
                    math.isfinite(value) and value >= 0
                    for value in [*self._source_load_mw, *self._source_pv_mw]
                )
            ):
                raise ValueError(
                    "source_profile_applied requires finite non-negative "
                    "load_mw/pv_mw arrays covering the horizon"
                )
            self._source_load_reference_mw = float(
                references.get("load_mw") or 0.0
            )
            self._source_pv_reference_mw = float(
                references.get("pv_mw") or 0.0
            )
            if (
                not math.isfinite(self._source_load_reference_mw)
                or not math.isfinite(self._source_pv_reference_mw)
                or self._source_load_reference_mw <= 0
                or self._source_pv_reference_mw <= 0
            ):
                raise ValueError(
                    "source_profile_reference load_mw/pv_mw must be positive"
                )
            self._bind_runtime_source_asset()

        self._net = pn.create_synthetic_voltage_control_lv_network()
        self._base_load_p_mw = {
            int(i): float(self._net.load.p_mw.iloc[i])
            for i in range(len(self._net.load))
        }
        self._base_load_q_mvar = {
            int(i): float(self._net.load.q_mvar.iloc[i])
            for i in range(len(self._net.load))
        }
        self._base_sgen_p_mw = {
            int(i): float(self._net.sgen.p_mw.iloc[i])
            for i in range(len(self._net.sgen))
        }
        # If the bundled net carries no PV, seed a deterministic rooftop PV
        # at the highest-load buses so reverse flow / over-voltage exists.
        if not self._base_sgen_p_mw:
            for li, _mw in sorted(self._base_load_p_mw.items(), key=lambda kv: -kv[1])[
                :4
            ]:
                bus = int(self._net.load.bus.iloc[li])
                idx = pp.create_sgen(
                    self._net,
                    bus=bus,
                    p_mw=0.02,
                    q_mvar=0.0,
                    name=f"rooftop_pv_{bus}",
                    type="PV",
                )
                self._base_sgen_p_mw[int(idx)] = 0.02

        # Map DER ids → sgen indices.
        self._der_to_sgen = {
            f"der{i}": idx for i, idx in enumerate(sorted(self._base_sgen_p_mw))
        }
        q_capability_fraction = float(
            cfg.get("der_q_capability_fraction", 0.5) or 0.5
        )
        if not 0 < q_capability_fraction <= 1:
            raise ValueError("der_q_capability_fraction must be in (0, 1]")
        self._der_q_limits = {
            idx: max(
                0.001,
                base * self._pv_scale * q_capability_fraction,
            )
            for idx, base in self._base_sgen_p_mw.items()
        }

        # Add a controllable battery (storage) at the worst-PV bus.
        self._storage_idx = None
        battery_cfg = cfg.get("battery") or {}
        self._storage_capacity_mwh = float(
            battery_cfg.get("capacity_mwh")
            or cfg.get("battery_e_mwh", 0.05)
            or 0.05
        )
        init_soc = float(battery_cfg.get("init_soc", 0.5))
        self._storage_max_charge_mw = float(
            battery_cfg.get("max_charge_mw", self._storage_capacity_mwh / 2.0)
        )
        self._storage_max_discharge_mw = float(
            battery_cfg.get("max_discharge_mw", self._storage_capacity_mwh / 2.0)
        )
        self._storage_efficiency = float(battery_cfg.get("efficiency", 0.95))
        if (
            self._storage_capacity_mwh <= 0
            or not 0 <= init_soc <= 1
            or self._storage_max_charge_mw <= 0
            or self._storage_max_discharge_mw <= 0
            or not 0 < self._storage_efficiency <= 1
        ):
            raise ValueError("invalid LV battery capacity, SoC, rate, or efficiency")
        self._storage_energy_mwh = self._storage_capacity_mwh * init_soc
        if self._base_sgen_p_mw:
            worst = max(self._base_sgen_p_mw, key=lambda k: self._base_sgen_p_mw[k])
            bus = int(self._net.sgen.bus.iloc[worst])
            self._storage_idx = int(
                pp.create_storage(
                    self._net,
                    bus=bus,
                    p_mw=0.0,
                    max_e_mwh=self._storage_capacity_mwh,
                    soc_percent=init_soc * 100.0,
                    name="ems_battery",
                )
            )

        # Map load assignments → pandapower load indices in seed order.
        self._loads = {}
        self._idx_to_load_id = {}
        self._cumulative_shed_mwh = {}
        for idx, la in enumerate(scenario_seed.load_assignments):
            if idx >= len(self._net.load):
                break
            self._loads[la.load_id] = {
                "load_index": idx,
                "bus_id": int(self._net.load.bus.iloc[idx]),
                "stakeholder_class": str(la.stakeholder_class),
                "criticality": float(la.criticality),
                "shed_this_tick_mw": 0.0,
            }
            self._idx_to_load_id[idx] = la.load_id
            self._cumulative_shed_mwh[la.load_id] = 0.0

        self._pending_curtail = {}
        self._der_q_targets = {}
        self._storage_p_target = 0.0
        self._storage_p_applied = 0.0
        self._tick_records = []
        self._applied_control_records = []
        self._realized_events_this_tick = []
        self._revealed_assets = set()
        self._initial_source_state_digest = self._source_state_digest()

    def _bind_runtime_source_asset(self) -> None:
        """Reopen and verify the exact NPZ window consumed by the backend."""
        if self._seed_obj is None:
            return
        repo_root = Path(__file__).resolve().parents[3]
        declared_npz = next(
            (
                str(value)
                for value in self._seed_obj.provenance.files
                if str(value).endswith(".npz")
            ),
            "",
        )
        if not declared_npz:
            return
        path = (repo_root / declared_npz).resolve()
        if not path.is_file():
            return
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            start = self._source_profile_start_index
            stop = start + self._horizon
            if "load_mw" not in data or "pv_mw" not in data:
                return
            runtime_load = [
                round(float(value), 5)
                for value in data["load_mw"].ravel()[start:stop]
            ]
            runtime_pv = [
                round(float(value), 5)
                for value in data["pv_mw"].ravel()[start:stop]
            ]
        expected_load = [
            round(float(value), 5) for value in self._source_load_mw
        ]
        expected_pv = [
            round(float(value), 5) for value in self._source_pv_mw
        ]
        if (
            len(runtime_load) != self._horizon
            or len(runtime_pv) != self._horizon
            or runtime_load != expected_load
            or runtime_pv != expected_pv
        ):
            return
        relative = str(path.relative_to(repo_root))
        self._runtime_source_assets = [
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "role": "runtime_input",
            }
        ]
        self._runtime_source_verified = True

    # ── Tool effects ─────────────────────────────────────────────────────

    def apply_tool_effect(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "shed_load":
            return self._shed_load(args)
        if name == "curtail_der":
            return self._curtail_der(args)
        if name == "set_der_reactive_power":
            return self._set_der_q(args)
        if name == "set_battery_dispatch":
            return self._set_battery(args)
        if name in ("set_grid_exchange", "dispatch_genset", "connect_pcc"):
            # The LV feeder is a grid-connected Volt-Var tier with no
            # islanding / genset / PCC arbitrage — these EMS levers are
            # not effective here (honest unsupported, not a hard rejection).
            return {
                "_status": "unsupported_on_lv",
                "info": f"{name} is only effective on the pymgrid EMS families",
            }
        return {"_status": "ack"}

    def record_applied_control(
        self,
        *,
        tick: int,
        tool_name: str,
        args: dict[str, Any],
    ) -> None:
        """Record a successfully materialized native control for task gates."""
        self._applied_control_records.append(
            {
                "tick": int(tick),
                "tool_name": str(tool_name),
                "args": dict(args),
            }
        )

    def applied_control_records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._applied_control_records]

    def _shed_load(self, args: dict[str, Any]) -> dict[str, Any]:
        lid = str(args.get("load_id", ""))
        mw = float(args.get("mw", 0.0))
        if mw <= 0:
            return {"_status": "error", "error": "non_positive_shed", "mw": mw}
        entry = self._loads.get(lid)
        if entry is None:
            return {"_status": "error", "error": "unknown_load", "load_id": lid}
        entry["shed_this_tick_mw"] += mw
        tick_h = self._tick_minutes / 60.0
        self._cumulative_shed_mwh[lid] += mw * tick_h
        return {
            "load_id": lid,
            "shed_mw": round(mw, 4),
            "stakeholder_class": entry["stakeholder_class"],
            "criticality": entry["criticality"],
        }

    def _curtail_der(self, args: dict[str, Any]) -> dict[str, Any]:
        did = str(args.get("der_id", ""))
        if did not in self._der_to_sgen:
            return {"_status": "error", "error": "unknown_der", "der_id": did}
        target = float(args.get("target_mw", 0.0))
        if target < 0:
            return {"_status": "error", "error": "negative_target", "der_id": did}
        self._pending_curtail[self._der_to_sgen[did]] = target
        return {"der_id": did, "target_mw": round(target, 4), "queued": True}

    def _set_der_q(self, args: dict[str, Any]) -> dict[str, Any]:
        did = str(args.get("der_id", ""))
        if did not in self._der_to_sgen:
            return {"_status": "error", "error": "unknown_der", "der_id": did}
        q = float(args.get("q_mvar", 0.0))
        if not math.isfinite(q):
            return {"_status": "error", "error": "non_finite_reactive_setpoint"}
        sgen_idx = self._der_to_sgen[did]
        limit = self._der_q_limits[sgen_idx]
        if abs(q) > limit + 1e-12:
            return {
                "_status": "error",
                "error": "der_reactive_limit_exceeded",
                "der_id": did,
                "q_mvar": q,
                "max_abs_q_mvar": round(limit, 6),
            }
        self._der_q_targets[sgen_idx] = q
        return {"der_id": did, "q_mvar": round(q, 4), "queued": True}

    def _set_battery(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._storage_idx is None:
            return {"_status": "error", "error": "no_battery"}
        bid = str(args.get("battery_id", "batt0"))
        p_mw = float(args.get("p_mw", 0.0))
        if bid != "batt0":
            return {"_status": "error", "error": "unknown_battery", "battery_id": bid}
        if not math.isfinite(p_mw):
            return {"_status": "error", "error": "non_finite_dispatch"}
        rate_limit = (
            self._storage_max_charge_mw
            if p_mw >= 0
            else self._storage_max_discharge_mw
        )
        if abs(p_mw) > rate_limit + 1e-12:
            return {
                "_status": "error",
                "error": "battery_rate_exceeded",
                "p_mw": p_mw,
                "rate_limit_mw": rate_limit,
            }
        tick_h = self._tick_minutes / 60.0
        if p_mw > 0:
            feasible = (
                self._storage_capacity_mwh - self._storage_energy_mwh
            ) / (self._storage_efficiency * tick_h)
        elif p_mw < 0:
            feasible = (
                self._storage_energy_mwh * self._storage_efficiency / tick_h
            )
        else:
            feasible = 0.0
        if abs(p_mw) > feasible + 1e-12:
            return {
                "_status": "error",
                "error": "battery_soc_window_exceeded",
                "p_mw": p_mw,
                "feasible_mw": round(feasible, 6),
            }
        self._storage_p_target = p_mw  # +charge (absorbs reverse PV) / −discharge
        return {"battery_id": bid, "p_mw": round(p_mw, 4), "queued": True}

    # ── Tick ─────────────────────────────────────────────────────────────

    def tick(self, current_tick: int) -> _LvTickRecord:
        assert self._net is not None
        self._tick = int(current_tick)
        self._realized_events_this_tick = self._apply_perturbations_at_tick(self._tick)
        tick_h = self._tick_minutes / 60.0

        # Source-grounded candidates consume a locked load/PV window. Legacy
        # scenarios retain their historical deterministic diurnal path.
        if self._source_profile_applied:
            diurnal = (
                self._source_load_mw[current_tick]
                / self._source_load_reference_mw
            )
            solar = self._source_pv_mw[current_tick] / self._source_pv_reference_mw
        else:
            diurnal = 0.85 + 0.30 * math.sin(
                math.pi * current_tick / max(1, self._horizon)
            )
            solar = max(
                0.0,
                math.sin(math.pi * current_tick / max(1, self._horizon - 1)),
            )

        for idx, base in self._base_load_p_mw.items():
            lid = self._idx_to_load_id.get(idx, "")
            shed = float(self._loads.get(lid, {}).get("shed_this_tick_mw", 0.0))
            new_p = max(
                0.0,
                base * diurnal * self._active_load_multiplier - shed,
            )
            self._net.load.at[idx, "p_mw"] = new_p
            q_base = self._base_load_q_mvar.get(idx, 0.0)
            self._net.load.at[idx, "q_mvar"] = (
                q_base * diurnal * self._active_load_multiplier
            )

        # PV: locked NSRDB factor (new candidates) or legacy solar bell.
        for idx, base in self._base_sgen_p_mw.items():
            new_p = (
                0.0
                if idx in self._failed_sgen_indices
                else base * self._active_pv_scale * solar
            )
            cap = self._pending_curtail.get(idx)
            if cap is not None and cap < new_p:
                new_p = cap
            self._net.sgen.at[idx, "p_mw"] = new_p
            if idx in self._der_q_targets and "q_mvar" in self._net.sgen:
                self._net.sgen.at[idx, "q_mvar"] = self._der_q_targets[idx]

        if self._storage_idx is not None and len(self._net.storage):
            if self._storage_p_target >= 0:
                feasible = (
                    self._storage_capacity_mwh - self._storage_energy_mwh
                ) / (self._storage_efficiency * tick_h)
                self._storage_p_applied = min(
                    self._storage_p_target,
                    self._storage_max_charge_mw,
                    max(0.0, feasible),
                )
                self._storage_energy_mwh += (
                    self._storage_p_applied * self._storage_efficiency * tick_h
                )
            else:
                feasible = (
                    self._storage_energy_mwh * self._storage_efficiency / tick_h
                )
                discharge = min(
                    -self._storage_p_target,
                    self._storage_max_discharge_mw,
                    max(0.0, feasible),
                )
                self._storage_p_applied = -discharge
                self._storage_energy_mwh -= (
                    discharge / self._storage_efficiency * tick_h
                )
            self._storage_energy_mwh = min(
                self._storage_capacity_mwh,
                max(0.0, self._storage_energy_mwh),
            )
            self._net.storage.at[
                self._storage_idx, "p_mw"
            ] = self._storage_p_applied
            self._net.storage.at[self._storage_idx, "soc_percent"] = (
                100.0 * self._storage_energy_mwh / self._storage_capacity_mwh
            )

        converged = True
        try:
            pp.runpp(self._net, numba=False)
        except Exception:
            converged = False

        if converged:
            demand_mw = float(self._net.load.p_mw.sum())
            gen_mw = float(self._net.sgen.p_mw.sum()) + float(
                self._net.res_ext_grid.p_mw.sum()
                if "p_mw" in self._net.res_ext_grid.columns
                else 0.0
            )
            rho = self._net.res_line.loading_percent.to_numpy() / 100.0
            rho_max = float(rho.max()) if len(rho) else 0.0
            n_overload = int((rho > 1.0).sum())
            v_pu = self._net.res_bus.vm_pu.to_numpy()
            n_v_viol = int(
                ((v_pu < self.VOLTAGE_LOWER_PU) | (v_pu > self.VOLTAGE_UPPER_PU)).sum()
            )
            n_disc = int((~self._net.line.in_service.to_numpy()).sum())
        else:
            last = self._tick_records[-1] if self._tick_records else None
            demand_mw = float(self._net.load.p_mw.sum())
            gen_mw = last.aggregate_generation_mw if last else 0.0
            rho_max = last.rho_max if last else 2.0
            n_overload = last.n_overloads if last else len(self._net.line)
            n_v_viol = last.n_voltage_violations if last else len(self._net.bus)
            n_disc = last.n_disconnected_lines if last else 0

        shed_penalty = 0.0
        for entry in self._loads.values():
            sh = float(entry.get("shed_this_tick_mw", 0.0))
            if sh <= 0:
                continue
            tariff = _SHED_TARIFF_BY_CLASS.get(
                str(entry.get("stakeholder_class", "")), _SHED_TARIFF_DEFAULT
            )
            shed_penalty += sh * tariff * tick_h

        prod_cost = self.PRODUCTION_COST_PER_MWH * demand_mw * tick_h
        prod_cost += n_v_viol * self.VOLTAGE_VIOLATION_COST_PER_TICK
        prod_cost += n_overload * self.OVERLOAD_COST_PER_TICK
        prod_cost += n_disc * self.DISCONNECTION_COST_PER_LINE_TICK

        reserves_required = self.RESERVE_TARGET_FRACTION_OF_DEMAND * demand_mw
        # ext-grid acts as slack → effectively ample headroom.
        reserves_procured = reserves_required + 0.01

        balance_error = gen_mw - demand_mw
        source_event = self._source_schedule_event(
            current_tick=current_tick,
            demand_mw=demand_mw,
        )

        record = _LvTickRecord(
            tick=current_tick,
            aggregate_demand_mw=round(demand_mw, 4),
            aggregate_generation_mw=round(gen_mw, 4),
            balance_error_mw=round(balance_error, 4),
            reserves_required_mw=round(reserves_required, 4),
            reserves_procured_mw=round(reserves_procured, 4),
            production_cost=round(prod_cost, 3),
            startup_cost=0.0,  # honest 0 — no genset on the LV feeder
            shed_penalty=round(shed_penalty, 3),
            rho_max=round(rho_max, 4),
            n_overloads=n_overload,
            n_voltage_violations=n_v_viol,
            n_disconnected_lines=n_disc,
            converged=converged,
            done=current_tick >= self._horizon - 1,
            realized_events=[
                *list(self._realized_events_this_tick),
                source_event,
            ],
        )
        self._tick_records.append(record)
        self._source_consumption_ticks.append(current_tick)
        self._runtime_source_events.append(source_event)
        self._post_source_state_digests.append(
            {
                "tick": current_tick,
                "sha256": self._source_state_digest(record=record),
            }
        )
        for entry in self._loads.values():
            entry["shed_this_tick_mw"] = 0.0
        return record

    def _source_schedule_event(
        self,
        *,
        current_tick: int,
        demand_mw: float,
    ) -> dict[str, Any]:
        previous_load = (
            self._source_load_mw[current_tick - 1]
            if self._source_profile_applied and current_tick > 0
            else 0.0
        )
        current_load = (
            self._source_load_mw[current_tick]
            if self._source_profile_applied
            else demand_mw
        )
        current_pv = (
            self._source_pv_mw[current_tick]
            if self._source_profile_applied
            else 0.0
        )
        delta = abs(float(current_load) - float(previous_load))
        threshold = max(1e-9, abs(float(previous_load)) * 1e-6)
        return {
            "type": "source_profile_interval",
            "event_id": f"pandapower_lv_source_interval:{current_tick}",
            "origin": "source_schedule",
            "event_class": "telemetry",
            "tick": current_tick,
            "decision_required": False,
            "actionable": False,
            "changed_state_fields": [
                "aggregate_demand_mw",
                "aggregate_generation_mw",
                "bus_voltage_pu",
                "line_loading_percent",
            ],
            "materiality_metric": "absolute_source_load_delta_mw",
            "materiality_value": delta,
            "materiality_threshold": threshold,
            "materiality_passed": delta >= threshold,
            "source_load_mw": round(float(current_load), 6),
            "source_pv_mw": round(float(current_pv), 6),
        }

    # ── Perturbations ────────────────────────────────────────────────────

    def _apply_perturbations_at_tick(self, tick: int) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        self._active_pv_scale = self._pv_scale
        self._active_load_multiplier = 1.0
        self._failed_sgen_indices = set()
        if self._seed_obj is None:
            return events
        for p in self._seed_obj.perturbations:
            kind = str(p.kind)
            try:
                event_class = PANDAPOWER_LV_PERTURBATION_EVENT_CLASS[kind]
            except KeyError as exc:
                raise ValueError(
                    f"unknown pandapower LV perturbation event type: {kind}"
                ) from exc
            if not (p.trigger_tick <= tick < p.trigger_tick + p.duration_ticks):
                continue
            actionable = not bool(p.hidden) and tick + 1 < self._horizon
            if kind == "pv_ramp":
                # Boost PV scale during the window to provoke over-voltage.
                if self._source_profile_applied:
                    self._active_pv_scale = max(
                        self._active_pv_scale,
                        self._pv_scale * float(p.intensity),
                    )
                else:
                    self._active_pv_scale = max(
                        self._active_pv_scale,
                        6.0 * float(p.intensity),
                    )
                if tick == p.trigger_tick:
                    events.append(
                        {
                            "type": "pv_ramp",
                            "event_id": f"pandapower_lv:pv_ramp:{tick}",
                            "origin": "declared_perturbation",
                            "event_class": event_class,
                            "tick": tick,
                            "factor": p.intensity,
                            "hidden": p.hidden,
                            "decision_required": actionable,
                            "actionable": actionable,
                            "changed_state_fields": [
                                "der_generation_mw",
                                "bus_voltage_pu",
                            ],
                            "materiality_metric": "pv_scale_factor",
                            "materiality_value": float(p.intensity),
                            "materiality_threshold": 1.0,
                            "materiality_passed": float(p.intensity) > 1.0,
                        }
                    )
            elif kind == "der_failure":
                der_ids = sorted(self._der_to_sgen)
                if der_ids:
                    idx = int(p.target.get("der_index", 0)) % len(der_ids)
                    self._failed_sgen_indices.add(
                        self._der_to_sgen[der_ids[idx]]
                    )
                    if tick == p.trigger_tick:
                        events.append(
                            {
                                "type": "der_failure",
                                "event_id": (
                                    f"pandapower_lv:der_failure:{tick}:{der_ids[idx]}"
                                ),
                                "origin": "declared_perturbation",
                                "event_class": event_class,
                                "tick": tick,
                                "der_id": der_ids[idx],
                                "hidden": p.hidden,
                                "decision_required": actionable,
                                "actionable": actionable,
                                "changed_state_fields": [
                                    "der_generation_mw",
                                    "bus_voltage_pu",
                                ],
                                "materiality_metric": "failed_der_count",
                                "materiality_value": 1,
                                "materiality_threshold": 1,
                                "materiality_passed": True,
                            }
                        )
            elif kind == "load_spike":
                self._active_load_multiplier = max(
                    self._active_load_multiplier,
                    1.0 + float(p.intensity),
                )
                if tick == p.trigger_tick:
                    events.append(
                        {
                            "type": "load_spike",
                            "event_id": f"pandapower_lv:load_spike:{tick}",
                            "origin": "declared_perturbation",
                            "event_class": event_class,
                            "tick": tick,
                            "intensity": p.intensity,
                            "hidden": p.hidden,
                            "decision_required": actionable,
                            "actionable": actionable,
                            "changed_state_fields": [
                                "aggregate_demand_mw",
                                "bus_voltage_pu",
                            ],
                            "materiality_metric": "load_multiplier_delta",
                            "materiality_value": float(p.intensity),
                            "materiality_threshold": 0.0,
                            "materiality_passed": float(p.intensity) > 0.0,
                        }
                    )
        return events

    # ── Read-only helpers ────────────────────────────────────────────────

    def forecast_for(self, horizon: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        base_total = sum(self._base_load_p_mw.values())
        for k in range(1, max(1, int(horizon)) + 1):
            t = self._tick + k
            if self._source_profile_applied:
                idx = min(t, len(self._source_load_mw) - 1)
                diurnal = (
                    self._source_load_mw[idx] / self._source_load_reference_mw
                )
                solar = self._source_pv_mw[idx] / self._source_pv_reference_mw
            else:
                diurnal = 0.85 + 0.30 * math.sin(
                    math.pi * t / max(1, self._horizon)
                )
                solar = max(
                    0.0,
                    math.sin(math.pi * t / max(1, self._horizon - 1)),
                )
            out.append(
                {
                    "tick": t,
                    "load_mw_forecast": round(
                        base_total * diurnal * (1.0 + self._forecast_bias), 5
                    ),
                    "pv_factor_forecast": round(solar, 4),
                    "noised": True,
                }
            )
        return out

    def investigate_asset(self, asset_id: str) -> dict[str, Any]:
        self._revealed_assets.add(asset_id)
        if (
            asset_id == "batt0"
            and self._storage_idx is not None
            and len(self._net.storage)
        ):
            idx = self._storage_idx
            return {
                "asset_id": asset_id,
                "kind": "battery",
                "p_mw": float(self._net.storage.p_mw.iloc[idx]),
                "soc_percent": float(self._net.storage.soc_percent.iloc[idx]),
                "max_e_mwh": float(self._net.storage.max_e_mwh.iloc[idx]),
                "soc_mwh": round(self._storage_energy_mwh, 6),
                "max_charge_mw": self._storage_max_charge_mw,
                "max_discharge_mw": self._storage_max_discharge_mw,
                "efficiency": self._storage_efficiency,
            }
        if asset_id in self._der_to_sgen:
            idx = self._der_to_sgen[asset_id]
            return {
                "asset_id": asset_id,
                "kind": "pv",
                "output_mw": float(self._net.sgen.p_mw.iloc[idx]),
                "curtail_cap_mw": self._pending_curtail.get(idx),
                "max_abs_q_mvar": round(self._der_q_limits[idx], 6),
            }
        return {"_status": "error", "error": "unknown_asset", "asset_id": asset_id}

    def reveal_asset(self, asset_id: str) -> None:
        self._revealed_assets.add(asset_id)

    def realized_events_for_tick(self) -> list[dict[str, Any]]:
        return list(self._realized_events_this_tick)

    # ── Snapshot ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        if self._net is None or self._seed_obj is None:
            return {"entities": {}, "totals": {}, "tick": self._tick}
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
                    self._cumulative_shed_mwh.get(lid, 0.0), 4
                ),
            }
        for did, idx in self._der_to_sgen.items():
            entities[did] = {
                "kind": "pv",
                "bus_id": int(self._net.sgen.bus.iloc[idx]),
                "output_mw": float(self._net.sgen.p_mw.iloc[idx]),
                "curtail_cap_mw": self._pending_curtail.get(idx),
                "max_abs_q_mvar": round(self._der_q_limits[idx], 6),
            }
        if self._storage_idx is not None and len(self._net.storage):
            entities["batt0"] = {
                "kind": "battery",
                "bus_id": int(self._net.storage.bus.iloc[self._storage_idx]),
                "p_mw": float(self._net.storage.p_mw.iloc[self._storage_idx]),
                "commanded_p_mw": round(self._storage_p_target, 6),
                "applied_p_mw": round(self._storage_p_applied, 6),
                "soc_mwh": round(self._storage_energy_mwh, 6),
                "soc_percent": round(
                    100.0
                    * self._storage_energy_mwh
                    / self._storage_capacity_mwh,
                    4,
                ),
                "max_e_mwh": self._storage_capacity_mwh,
                "max_charge_mw": self._storage_max_charge_mw,
                "max_discharge_mw": self._storage_max_discharge_mw,
                "efficiency": self._storage_efficiency,
            }
        if hasattr(self._net, "res_bus") and len(self._net.res_bus):
            for b in range(len(self._net.bus)):
                try:
                    vm = float(self._net.res_bus.vm_pu.iloc[b])
                except Exception:
                    continue
                if math.isfinite(vm):
                    entities[f"bus_{b}"] = {"kind": "bus", "vm_pu": round(vm, 4)}
        last = self._tick_records[-1] if self._tick_records else None
        totals = {
            "demand_mw": float(self._net.load.p_mw.sum()),
            "der_generation_mw": float(self._net.sgen.p_mw.sum()),
            "rho_max": last.rho_max if last else 0.0,
            "n_voltage_violations": last.n_voltage_violations if last else 0,
            "n_overloads": last.n_overloads if last else 0,
            "balance_error_mw": last.balance_error_mw if last else 0.0,
        }
        source_profile = {
            "applied": self._source_profile_applied,
            "start_index": self._source_profile_start_index,
            "source_index": (
                self._source_profile_start_index + self._tick
                if self._source_profile_applied
                else None
            ),
            "load_mw": (
                self._source_load_mw[self._tick]
                if self._source_profile_applied
                else None
            ),
            "pv_mw": (
                self._source_pv_mw[self._tick]
                if self._source_profile_applied
                else None
            ),
        }
        return {
            "tick": self._tick,
            "horizon": self._horizon,
            "entities": entities,
            "totals": totals,
            "source_profile": source_profile,
        }

    # ── Cost roll-up / scoring ───────────────────────────────────────────

    def ground_truth_costs(self) -> dict[str, float]:
        volt = sum(
            r.n_voltage_violations * self.VOLTAGE_VIOLATION_COST_PER_TICK
            for r in self._tick_records
        )
        overload = sum(
            r.n_overloads * self.OVERLOAD_COST_PER_TICK
            for r in self._tick_records
        )
        disconnection = sum(
            r.n_disconnected_lines * self.DISCONNECTION_COST_PER_LINE_TICK
            for r in self._tick_records
        )
        production = (
            sum(r.production_cost for r in self._tick_records)
            - volt
            - overload
            - disconnection
        )
        shed = sum(r.shed_penalty for r in self._tick_records)
        return {
            "production_cost": round(production, 3),
            "startup_cost": 0.0,
            "shed_penalty": round(shed, 3),
            "voltage_violation_cost": round(volt, 3),
            "overload_cost": round(overload, 3),
            "disconnection_cost": round(disconnection, 3),
        }

    def scoring_records(self) -> list[dict[str, Any]]:
        """Per-tick rows: 14 canonical keys with the four power-flow keys
        filled **honestly** from the real ``pp.runpp`` solve. ``startup_cost``
        is honest-0 (no genset). ``done`` is early-guarded.
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
                "rho_max": float(r.rho_max),
                "n_overloads": int(r.n_overloads),
                "n_voltage_violations": int(r.n_voltage_violations),
                "n_disconnected_lines": int(r.n_disconnected_lines),
                "done": bool(r.done and r.tick < self._horizon - 1),
            }
            for r in self._tick_records
        ]

    def per_load_shed_mwh(self) -> dict[str, float]:
        return dict(self._cumulative_shed_mwh)

    def _source_state_digest(
        self,
        *,
        record: _LvTickRecord | None = None,
    ) -> str:
        current = record or (
            self._tick_records[-1] if self._tick_records else None
        )
        return _semantic_digest(
            {
                "tick": current.tick if current else self._tick,
                "source_window_sha256": (
                    (self._seed_obj.backend_config or {})
                    .get("derivation_recipe", {})
                    .get("source_window_sha256")
                    if self._seed_obj
                    else None
                ),
                "aggregate_demand_mw": (
                    current.aggregate_demand_mw if current else None
                ),
                "aggregate_generation_mw": (
                    current.aggregate_generation_mw if current else None
                ),
                "rho_max": current.rho_max if current else None,
                "n_voltage_violations": (
                    current.n_voltage_violations if current else None
                ),
            }
        )

    def protocol21_source_trace(self) -> dict[str, Any]:
        """Prove that a locked NPZ window drove native LV power-flow state."""
        source_effect = bool(
            self._runtime_source_verified
            and self._source_consumption_ticks
            and any(
                event.get("materiality_passed") is True
                for event in self._runtime_source_events
            )
        )
        opened_paths = [
            asset["path"] for asset in self._runtime_source_assets
        ]
        opened_hashes = {
            asset["path"]: asset["sha256"]
            for asset in self._runtime_source_assets
        }
        recipe = (
            (self._seed_obj.backend_config or {}).get(
                "derivation_recipe", {}
            )
            if self._seed_obj
            else {}
        )
        semantic = {
            "opened_source_sha256": opened_hashes,
            "consumed_window_sha256": recipe.get("source_window_sha256"),
            "consumption_ticks": self._source_consumption_ticks,
            "post_source_state_digests": self._post_source_state_digests,
            "runtime_source_events": self._runtime_source_events,
        }
        blockers = []
        if not self._runtime_source_verified:
            blockers.append("runtime_source_asset_window_mismatch")
        if not self._source_consumption_ticks:
            blockers.append("runtime_source_consumption_unobserved")
        if not source_effect:
            blockers.append("source_state_effect_unproven")
        return {
            "status": "passed" if not blockers else "held",
            "proof_kind": "direct_runtime_files",
            "runtime_opened_assets": list(self._runtime_source_assets),
            "opened_source_paths": opened_paths,
            "opened_source_sha256": opened_hashes,
            "consumed_source_hashes": opened_hashes,
            "lineage_source_hashes": opened_hashes,
            "consumed_window_sha256": recipe.get("source_window_sha256"),
            "recipe_version": recipe.get("pipeline_version"),
            "consumed_channels": (
                ["load_mw", "pv_mw"]
                if self._runtime_source_verified
                else []
            ),
            "derived_backend_state_fields": (
                [
                    "aggregate_demand_mw",
                    "aggregate_generation_mw",
                    "bus_voltage_pu",
                    "line_loading_percent",
                ]
                if self._runtime_source_verified
                else []
            ),
            "consumption_ticks": list(self._source_consumption_ticks),
            "initial_state_digest": self._initial_source_state_digest,
            "post_source_state_digests": list(
                self._post_source_state_digests
            ),
            "runtime_source_events": list(self._runtime_source_events),
            "source_state_effect_observed": source_effect,
            "state_effect_observed": source_effect,
            "deterministic_source_trace": True,
            "trace_semantic_digest": _semantic_digest(semantic),
            "runtime_trace_observed": bool(
                self._runtime_source_verified
                and self._source_consumption_ticks
            ),
            "evidence_from_scenario_config_only": False,
            "blockers": sorted(set(blockers)),
        }

    def source_consumption_evidence(
        self,
        *,
        scenario: dict[str, Any],
    ) -> dict[str, Any]:
        """Trace the baked, source-locked load/PV window into solved states."""
        config = scenario.get("backend_config") or {}
        recipe = config.get("derivation_recipe") or {}
        payload = {
            "load_mw": [
                round(float(value), 9) for value in self._source_load_mw
            ],
            "pv_mw": [
                round(float(value), 9) for value in self._source_pv_mw
            ],
        }
        window_hash = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        expected_hash = str(recipe.get("source_window_sha256") or "")
        applied_ticks = list(
            range(min(len(self._tick_records), self._horizon))
        )
        valid = bool(
            self._source_profile_applied
            and self._source_load_mw
            and self._source_pv_mw
            and expected_hash
            and window_hash == expected_hash
            and applied_ticks
        )
        return {
            "status": "passed" if valid else "held",
            "proof_kind": "baked_source_window_backend_trace",
            "consumed_source_hashes": {},
            "consumed_window_sha256": window_hash if valid else None,
            "consumed_channels": (
                ["source_load_mw", "source_pv_mw"] if valid else []
            ),
            "derived_backend_state_fields": (
                [
                    "aggregate_demand_mw",
                    "aggregate_generation_mw",
                    "bus_voltage_pu",
                    "line_loading_percent",
                ]
                if valid
                else []
            ),
            "consumption_ticks": applied_ticks if valid else [],
            "state_effect_observed": valid,
            "blockers": [] if valid else ["source_window_trace_unproven"],
        }

    def apply_perturbation(self, perturbation: dict[str, Any]) -> None:
        """Apply a cascade perturbation to the LV network (no-op stub).

        The pandapower LV backend does not currently support dynamic
        cascade perturbations. Cascade events are logged but do not
        modify the network state.
        """
        import logging

        _LOGGER = logging.getLogger(__name__)
        _LOGGER.warning(
            "PandapowerLvBackend.apply_perturbation not implemented for: %s",
            perturbation.get("event_type", "unknown"),
        )

    def close(self) -> None:
        """Release pandapower network resources."""
        pass
