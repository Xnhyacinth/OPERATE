"""
domains.microgrid.backends.ems_sim — Deterministic seeded EMS simulator.

This **pure-Python** energy-management simulator IS the microgrid
environment (spec §9, mirroring ``domains.logistics.backends.route_sim``):
no pymgrid, no cvxpy, no wall-clock dependency to step. The baked NSRDB/OEDI
overlays, islanding/PV-ramp/DER-failure timing, and all fog/tool RNG are
locked to ``seed_id`` so two ``reset`` + ``tick`` loops with identical
inputs produce byte-identical records — the contract counterfactual replay
relies on.

Method surface mirrors
``domains.power_grid.backends.pglib_uc_synthetic.PglibUcSyntheticBackend``
(reset / tick / snapshot / apply_tool_effect / ground_truth_costs /
scoring_records / forecast_for / per_load_shed_mwh / realized_events_for_tick
+ a delayed-effect queue for start-up / PCC re-sync delays) so the microgrid
adapter is structurally identical to the other adapters.

pymgrid (LGPL-3.0) is imported lazily and used ONLY on the optional
cross-check path (``evaluate_with_pymgrid``); when absent the typed
``MicrogridBackendUnavailable`` is raised there only — the simulator itself
never needs it (spec §"Runtime gate" graceful-skip; mirrors the
``egret_acopf`` IPOPT gate and the logistics PyVRP cost-eval path).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..seeds.schema import MicrogridScenarioSeed


def _pymgrid_available() -> bool:
    """Mirror ``audit._runtime_unavailable`` detection for pymgrid."""
    return importlib.util.find_spec("pymgrid") is not None


PYMGRID_AVAILABLE = _pymgrid_available()


class MicrogridBackendUnavailable(RuntimeError):
    """Raised when pymgrid is required (cross-check path) but not importable."""


# ── Cost / penalty calibration (MW-equivalent proxy; spec §7) ───────────────
_BATTERY_DEGRADATION_PER_MWH = 8.0  # $/MWh throughput (cycle wear)
_RESERVE_TARGET_FRACTION = 0.10
_SHED_TARIFF_BY_CLASS: dict[str, float] = {
    "hospital": 5000.0,
    "water": 2500.0,
    "data_center": 1500.0,
    "commercial": 400.0,
    "residential": 200.0,
}
_SHED_TARIFF_DEFAULT = 1000.0
EMS_PERTURBATION_EVENT_CLASS = MappingProxyType(
    {
        "grid_outage": "alarm",
        "pv_ramp": "alarm",
        "der_failure": "alarm",
        "load_spike": "alarm",
        "price_spike": "alarm",
        "forecast_bias": "forecast",
    }
)


def _det_hash(seed: int, tick: int, key: str) -> int:
    body = f"{int(seed)}|{int(tick)}|{key}".encode()
    return int.from_bytes(hashlib.sha256(body).digest()[:4], "big") % 1000


def _semantic_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class _Battery:
    capacity_mwh: float
    soc_mwh: float
    max_charge_mw: float
    max_discharge_mw: float
    efficiency: float = 0.95
    setpoint_mw: float = 0.0  # +charge / −discharge requested by the agent


@dataclass
class _Genset:
    min_mw: float
    max_mw: float
    fuel_cost_per_mwh: float
    startup_cost: float
    available: bool
    committed: bool = False
    output_mw: float = 0.0


@dataclass
class _Der:
    der_id: str
    kind: str  # "pv" | "wind"
    curtail_cap_mw: float | None = None  # None ⇒ uncurtailed
    failed_until: int = -1


@dataclass
class _Load:
    load_id: str
    stakeholder_class: str
    criticality: float
    demand_fraction: float
    current_demand_mw: float = 0.0
    shed_this_tick_mw: float = 0.0
    cumulative_shed_mwh: float = 0.0


@dataclass
class _EmsTickRecord:
    tick: int
    aggregate_demand_mw: float = 0.0
    aggregate_generation_mw: float = 0.0
    balance_error_mw: float = 0.0
    reserves_required_mw: float = 0.0
    reserves_procured_mw: float = 0.0
    production_cost: float = 0.0
    startup_cost: float = 0.0
    shed_penalty: float = 0.0
    collapsed: bool = False
    done: bool = False
    realized_events: list[dict[str, Any]] = field(default_factory=list)


class EmsSimulator:
    """Pure-Python deterministic microgrid EMS simulator.

    Family behaviour is configured by the seed's ``backend_config`` flags
    (baked ``profiles``, ``battery`` / ``genset`` / ``grid`` sizing,
    ``honest_zero_keys``). The three ``pymgrid_*`` subclasses
    (see ``pymgrid_backend.py``) only set their canonical ``backend_kind``
    label.
    """

    backend_kind = "ems_sim"
    supported_tool_names = frozenset(
        {
            "set_battery_dispatch",
            "dispatch_genset",
            "set_grid_exchange",
            "connect_pcc",
            "curtail_der",
            "shed_load",
            "forecast_query",
            "investigate_asset",
            "commit_to_plan",
            "wait",
            "noop",
        }
    )

    def __init__(self) -> None:
        self._seed_obj: MicrogridScenarioSeed | None = None
        self._tick = 0
        self._horizon = 24
        self._seed = 0
        self._tick_minutes = 60
        self._profiles: dict[str, list[float]] = {}
        self._battery: _Battery | None = None
        self._genset: _Genset | None = None
        self._ders: dict[str, _Der] = {}
        self._loads: dict[str, _Load] = {}
        self._grid_max_import = 0.0
        self._grid_max_export = 0.0
        self._grid_setpoint: float | None = None  # agent override at PCC
        self._pcc_connected = True
        self._islanded_until = -1
        self._honest_zero_keys: set[str] = set()
        self._forecast_bias = 0.0
        self._forecast_sigma = 0.0
        self._pv_factor = 1.0
        self._wind_factor = 1.0
        self._load_spike_factor = 0.0
        self._price_spike_factor = 0.0
        self._battery_applied_mw = 0.0
        self._pending_effects: list[tuple[int, str, dict[str, Any]]] = []
        self._tick_records: list[_EmsTickRecord] = []
        self._realized_events_this_tick: list[dict[str, Any]] = []
        self._pending_startup_cost = 0.0
        self._revealed_assets: set[str] = set()
        self._collapsed = False
        self._runtime_source_assets: list[dict[str, str]] = []
        self._runtime_source_verified = False
        self._source_consumption_ticks: list[int] = []
        self._post_source_state_digests: list[str] = []
        self._runtime_source_events: list[dict[str, Any]] = []

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self, scenario_seed: MicrogridScenarioSeed) -> None:
        self._seed_obj = scenario_seed
        self._tick = 0
        self._horizon = int(scenario_seed.horizon_ticks)
        self._seed = int(scenario_seed.seed)
        self._tick_minutes = int(scenario_seed.tick_minutes)
        cfg = scenario_seed.backend_config or {}

        profiles = dict(cfg.get("profiles", {}) or {})
        self._profiles = {
            k: [float(x) for x in (profiles.get(k) or [])]
            for k in ("load_mw", "pv_mw", "wind_mw", "price")
        }
        self._runtime_source_assets = []
        self._runtime_source_verified = False
        self._source_consumption_ticks = []
        self._post_source_state_digests = []
        self._runtime_source_events = []
        self._bind_runtime_source_asset()
        self._forecast_bias = float(cfg.get("forecast_bias", 0.0) or 0.0)
        self._forecast_sigma = float(cfg.get("forecast_error_sigma", 0.0) or 0.0)
        self._honest_zero_keys = set(cfg.get("honest_zero_keys", []) or [])

        bcfg = dict(cfg.get("battery", {}) or {})
        if bcfg:
            cap = float(bcfg.get("capacity_mwh", 0.0) or 0.0)
            init_soc = float(bcfg.get("init_soc", 0.5) or 0.5)
            self._battery = _Battery(
                capacity_mwh=cap,
                soc_mwh=max(0.0, min(cap, init_soc * cap)),
                max_charge_mw=float(bcfg.get("max_charge_mw", cap) or cap),
                max_discharge_mw=float(bcfg.get("max_discharge_mw", cap) or cap),
                efficiency=float(bcfg.get("efficiency", 0.95) or 0.95),
            )
        else:
            self._battery = None

        gcfg = dict(cfg.get("genset", {}) or {})
        if gcfg:
            self._genset = _Genset(
                min_mw=float(gcfg.get("min_mw", 0.0) or 0.0),
                max_mw=float(gcfg.get("max_mw", 0.0) or 0.0),
                fuel_cost_per_mwh=float(gcfg.get("fuel_cost_per_mwh", 120.0) or 120.0),
                startup_cost=float(gcfg.get("startup_cost", 500.0) or 500.0),
                available=bool(gcfg.get("available", False)),
            )
        else:
            self._genset = None

        grid = dict(cfg.get("grid", {}) or {})
        self._grid_max_import = float(grid.get("max_import_mw", 0.0) or 0.0)
        self._grid_max_export = float(grid.get("max_export_mw", 0.0) or 0.0)
        self._grid_setpoint = None
        self._pcc_connected = True
        self._islanded_until = -1

        self._ders = {}
        for d in cfg.get("ders", []) or []:
            did = str(d["der_id"])
            self._ders[did] = _Der(der_id=did, kind=str(d.get("kind", "pv")))

        self._loads = {}
        for la in scenario_seed.load_assignments:
            self._loads[la.load_id] = _Load(
                load_id=la.load_id,
                stakeholder_class=str(la.stakeholder_class),
                criticality=float(la.criticality),
                demand_fraction=float(la.demand_fraction),
            )
        # Normalize fractions so aggregate demand maps onto load_mw profile.
        total = sum(load.demand_fraction for load in self._loads.values())
        if total > 0 and abs(total - 1.0) > 1e-9:
            for load in self._loads.values():
                load.demand_fraction /= total

        self._pending_effects = []
        self._tick_records = []
        self._realized_events_this_tick = []
        self._pending_startup_cost = 0.0
        self._pv_factor = 1.0
        self._wind_factor = 1.0
        self._load_spike_factor = 0.0
        self._price_spike_factor = 0.0
        self._battery_applied_mw = 0.0
        self._revealed_assets = set()
        self._collapsed = False

    # ── Delayed-effect queue (genset start-up / PCC resync) ─────────────

    def queue_effect(
        self, *, due_tick: int, kind: str, payload: dict[str, Any]
    ) -> None:
        self._pending_effects.append((int(due_tick), str(kind), dict(payload)))

    def _drain_effects(self, current_tick: int) -> None:
        kept: list[tuple[int, str, dict[str, Any]]] = []
        for due, kind, payload in self._pending_effects:
            if due > current_tick:
                kept.append((due, kind, payload))
                continue
            if kind == "genset_commit" and self._genset is not None:
                if not self._genset.committed:
                    self._pending_startup_cost += self._genset.startup_cost
                self._genset.committed = True
                self._genset.output_mw = float(payload.get("p_mw", 0.0))
                self._realized_events_this_tick.append(
                    {
                        "type": "genset_started",
                        "origin": "endogenous_completion",
                        "decision_required": False,
                        "actionable": False,
                        "tick": current_tick,
                        "genset_id": payload.get("genset_id", "genset0"),
                        "p_mw": self._genset.output_mw,
                    }
                )
            elif kind == "pcc_reconnect":
                connect = bool(payload.get("connect", True))
                if connect and not self._external_grid_available_at(current_tick):
                    continue
                before = self._pcc_connected
                self._pcc_connected = connect
                if before != self._pcc_connected:
                    self._realized_events_this_tick.append(
                        {
                            "type": (
                                "pcc_reconnected" if connect else "pcc_disconnected"
                            ),
                            "origin": "endogenous_completion",
                            "decision_required": False,
                            "actionable": False,
                            "tick": current_tick,
                        }
                    )
        self._pending_effects = kept

    # ── Islanding helper ─────────────────────────────────────────────────

    def _external_grid_available_at(self, tick: int) -> bool:
        if tick < self._islanded_until:
            return False
        if self._seed_obj is None:
            return True
        return not any(
            str(p.kind) == "grid_outage"
            and p.trigger_tick <= tick < p.trigger_tick + p.duration_ticks
            for p in self._seed_obj.perturbations
        )

    def _is_islanded(self) -> bool:
        return (
            not self._external_grid_available_at(self._tick) or not self._pcc_connected
        )

    def is_islanded_at(self, tick: int) -> bool:
        """Tick-aware islanding check for tool handlers.

        Tools execute *before* the tick's perturbations apply, so a handler
        cannot rely on ``_is_islanded`` (which reflects post-tick state).
        This scans the seed's ``grid_outage`` windows + the live PCC state so
        ``set_grid_exchange`` is ``DOMAIN_REJECTED`` exactly on islanded ticks.
        """
        if not self._pcc_connected:
            return True
        return not self._external_grid_available_at(tick)

    # ── Tick ─────────────────────────────────────────────────────────────

    def tick(self, current_tick: int) -> _EmsTickRecord:
        assert self._seed_obj is not None
        self._tick = int(current_tick)
        self._realized_events_this_tick = []
        self._pv_factor = 1.0
        self._wind_factor = 1.0
        self._load_spike_factor = 0.0
        self._price_spike_factor = 0.0

        self._drain_effects(self._tick)
        self._apply_perturbations_at_tick(self._tick)

        # 1) renewable output (baked profile × factor − curtail).
        pv_raw = self._profile_at("pv_mw") * self._pv_factor
        wind_raw = self._profile_at("wind_mw") * self._wind_factor
        pv_total, wind_total = self._apportion_renewables(pv_raw, wind_raw)
        renewable = pv_total + wind_total

        # 2) load (baked profile × spike − shed).
        load_total = self._profile_at("load_mw") * (1.0 + self._load_spike_factor)
        served_load = 0.0
        for load in self._loads.values():
            base = load_total * load.demand_fraction
            load.current_demand_mw = max(0.0, base - load.shed_this_tick_mw)
            served_load += load.current_demand_mw

        # 3) battery dispatch (clamped to rate + SoC).
        batt_charge, batt_discharge = self._resolve_battery()
        self._battery_applied_mw = batt_charge - batt_discharge

        # 4) genset output (committed only).
        genset_out = 0.0
        if self._genset is not None and self._genset.committed:
            genset_out = max(
                self._genset.min_mw, min(self._genset.max_mw, self._genset.output_mw)
            )

        # 5) grid exchange at PCC.
        demand_local = served_load + batt_charge
        supply_local = renewable + batt_discharge + genset_out
        islanded = self._is_islanded()
        if islanded:
            grid_import = 0.0
        elif self._grid_setpoint is not None:
            grid_import = max(
                -self._grid_max_export, min(self._grid_max_import, self._grid_setpoint)
            )
        else:
            # Slack: auto-import the residual (wait_only pays full price here).
            grid_import = max(
                -self._grid_max_export,
                min(self._grid_max_import, demand_local - supply_local),
            )

        total_supply = supply_local + grid_import
        unmet = max(0.0, demand_local - total_supply)
        overgen = max(0.0, total_supply - demand_local)
        balance_error = unmet - overgen

        # 6) SoC update (charge stores eff·E; discharge drains E/eff).
        if self._battery is not None:
            self._battery.soc_mwh = max(
                0.0,
                min(
                    self._battery.capacity_mwh,
                    self._battery.soc_mwh
                    + batt_charge * self._battery.efficiency * self._tick_h()
                    - (batt_discharge / max(0.01, self._battery.efficiency))
                    * self._tick_h(),
                ),
            )

        # 7) costs.
        price = self._profile_at("price") * (1.0 + self._price_spike_factor)
        prod_cost = 0.0
        prod_cost += genset_out * self._genset_fuel_cost() * self._tick_h()
        prod_cost += max(0.0, grid_import) * price * self._tick_h()
        prod_cost += (
            (batt_charge + batt_discharge)
            * _BATTERY_DEGRADATION_PER_MWH
            * (self._tick_h())
        )
        startup_cost = self._pending_startup_cost
        self._pending_startup_cost = 0.0

        shed_penalty = 0.0
        for load in self._loads.values():
            if load.shed_this_tick_mw <= 0:
                continue
            tariff = _SHED_TARIFF_BY_CLASS.get(
                load.stakeholder_class, _SHED_TARIFF_DEFAULT
            )
            shed_penalty += load.shed_this_tick_mw * tariff * self._tick_h()
            load.cumulative_shed_mwh += load.shed_this_tick_mw * self._tick_h()

        # 8) reserves.
        reserves_required = _RESERVE_TARGET_FRACTION * served_load
        reserves_procured = 0.0
        if self._battery is not None:
            reserves_procured += min(
                self._battery.max_discharge_mw,
                self._battery.soc_mwh / max(0.01, self._tick_h()),
            )
        if self._genset is not None and self._genset.available:
            reserves_procured += max(0.0, self._genset.max_mw - genset_out)
        if not islanded:
            reserves_procured += max(0.0, self._grid_max_import - max(0.0, grid_import))

        # 9) catastrophic islanding collapse → terminal sentinel.
        critical_unmet = unmet > 0.0 and any(
            load.stakeholder_class in {"hospital", "water"}
            and load.current_demand_mw > 0.0
            for load in self._loads.values()
        )
        battery_depleted = self._battery is None or self._battery.soc_mwh <= 1e-6
        genset_dead = self._genset is None or not (
            self._genset.available and self._genset.committed
        )
        collapsed = bool(
            islanded and critical_unmet and battery_depleted and genset_dead
        )
        if collapsed:
            self._collapsed = True

        done = self._collapsed or (self._tick >= self._horizon - 1)

        record = _EmsTickRecord(
            tick=self._tick,
            aggregate_demand_mw=round(served_load + batt_charge, 3),
            aggregate_generation_mw=round(total_supply, 3),
            balance_error_mw=round(balance_error, 3),
            reserves_required_mw=round(reserves_required, 3),
            reserves_procured_mw=round(reserves_procured, 3),
            production_cost=round(prod_cost, 3),
            startup_cost=round(startup_cost, 3),
            shed_penalty=round(shed_penalty, 3),
            collapsed=collapsed,
            done=bool(done),
            realized_events=list(self._realized_events_this_tick),
        )
        source_event = self._source_schedule_event(record=record)
        if source_event is not None:
            record.realized_events.append(source_event)
        self._tick_records.append(record)
        if self._runtime_source_verified:
            self._source_consumption_ticks.append(self._tick)
            state_digest = self._source_state_digest(record=record)
            self._post_source_state_digests.append(state_digest)
            self._runtime_source_events.append(
                {
                    "tick": self._tick,
                    "source_values": {
                        key: round(self._profile_at(key), 5)
                        for key in ("load_mw", "pv_mw", "wind_mw", "price")
                    },
                    "derived_state": {
                        "aggregate_demand_mw": record.aggregate_demand_mw,
                        "aggregate_generation_mw": record.aggregate_generation_mw,
                        "production_cost": record.production_cost,
                        "balance_error_mw": record.balance_error_mw,
                    },
                    "materiality_passed": True,
                    "post_state_digest": state_digest,
                }
            )

        # reset per-tick shed accounting; battery setpoint persists until reset.
        for load in self._loads.values():
            load.shed_this_tick_mw = 0.0
        return record

    # ── Renewable apportionment ──────────────────────────────────────────

    def _apportion_renewables(
        self, pv_raw: float, wind_raw: float
    ) -> tuple[float, float]:
        pv_ders = [d for d in self._ders.values() if d.kind == "pv"]
        wind_ders = [d for d in self._ders.values() if d.kind == "wind"]
        pv = self._apportion_kind(pv_raw, pv_ders)
        wind = self._apportion_kind(wind_raw, wind_ders)
        return pv, wind

    def _apportion_kind(self, raw_total: float, ders: list[_Der]) -> float:
        if not ders:
            return max(0.0, raw_total)
        share = raw_total / len(ders)
        out = 0.0
        for d in ders:
            if self._tick < d.failed_until:
                continue
            val = share
            if d.curtail_cap_mw is not None:
                val = min(val, d.curtail_cap_mw)
            out += max(0.0, val)
        return out

    # ── Battery resolution ───────────────────────────────────────────────

    def _resolve_battery(self) -> tuple[float, float]:
        """Return (charge_mw, discharge_mw) clamped to rate + SoC."""
        if self._battery is None:
            return 0.0, 0.0
        sp = self._battery.setpoint_mw
        if sp > 0:  # charge
            room = (self._battery.capacity_mwh - self._battery.soc_mwh) / max(
                0.01, self._tick_h()
            )
            return min(sp, self._battery.max_charge_mw, max(0.0, room)), 0.0
        if sp < 0:  # discharge
            avail = self._battery.soc_mwh / max(0.01, self._tick_h())
            return 0.0, min(-sp, self._battery.max_discharge_mw, max(0.0, avail))
        return 0.0, 0.0

    # ── Perturbations ────────────────────────────────────────────────────

    def _apply_perturbations_at_tick(self, tick: int) -> None:
        assert self._seed_obj is not None
        for ordinal, p in enumerate(self._seed_obj.perturbations):
            kind = str(p.kind)
            if kind not in EMS_PERTURBATION_EVENT_CLASS:
                raise ValueError(f"unknown EMS perturbation event type: {kind}")
            if not (p.trigger_tick <= tick < p.trigger_tick + p.duration_ticks):
                continue
            self._apply_one(p, tick, ordinal)

    def _declared_perturbation_event(
        self,
        *,
        p: Any,
        tick: int,
        ordinal: int,
        changed_state_fields: list[str],
        materiality_metric: str,
        materiality_value: float,
        materiality_threshold: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Describe a simulator-applied perturbation at its first live tick."""
        kind = str(p.kind)
        event_class = EMS_PERTURBATION_EVENT_CLASS[kind]
        actionable = not bool(p.hidden) and tick + 1 < self._horizon
        return {
            "type": kind,
            "event_id": f"ems:{p.kind}:{p.trigger_tick}:{ordinal}",
            "origin": "declared_perturbation",
            "event_class": event_class,
            "declared_perturbation": True,
            "declared_event": {
                "kind": str(p.kind),
                "trigger_tick": int(p.trigger_tick),
                "duration_ticks": int(p.duration_ticks),
            },
            "tick": tick,
            "hidden": bool(p.hidden),
            "decision_required": actionable,
            "actionable": actionable,
            "changed_state_fields": changed_state_fields,
            "materiality_metric": materiality_metric,
            "materiality_value": materiality_value,
            "materiality_threshold": materiality_threshold,
            "materiality_passed": (
                abs(materiality_value) >= abs(materiality_threshold)
            ),
            **(extra or {}),
        }

    def _apply_one(self, p: Any, tick: int, ordinal: int) -> None:
        kind = str(p.kind)
        if kind == "grid_outage":
            self._islanded_until = max(
                self._islanded_until,
                p.trigger_tick + p.duration_ticks,
            )
            if tick == p.trigger_tick:
                self._pcc_connected = False
                self._realized_events_this_tick.append(
                    self._declared_perturbation_event(
                        p=p,
                        tick=tick,
                        ordinal=ordinal,
                        changed_state_fields=[
                            "external_grid_available",
                            "pcc_connected",
                            "pcc_islanded",
                            "grid_import_mw",
                        ],
                        materiality_metric="islanding_duration_ticks",
                        materiality_value=float(p.duration_ticks),
                        materiality_threshold=1.0,
                    )
                )
        elif kind == "pv_ramp":
            self._pv_factor = max(0.0, float(p.intensity))
            if tick == p.trigger_tick:
                self._realized_events_this_tick.append(
                    self._declared_perturbation_event(
                        p=p,
                        tick=tick,
                        ordinal=ordinal,
                        changed_state_fields=[
                            "der_generation_mw",
                            "aggregate_generation_mw",
                        ],
                        materiality_metric="pv_output_multiplier_delta",
                        materiality_value=abs(1.0 - float(p.intensity)),
                        materiality_threshold=0.01,
                        extra={"factor": float(p.intensity)},
                    )
                )
        elif kind == "der_failure":
            if tick != p.trigger_tick:
                return
            der_ids = list(self._ders.keys())
            if not der_ids:
                return
            idx = int(p.target.get("der_index", 0)) % len(der_ids)
            d = self._ders[der_ids[idx]]
            d.failed_until = p.trigger_tick + p.duration_ticks
            self._realized_events_this_tick.append(
                self._declared_perturbation_event(
                    p=p,
                    tick=tick,
                    ordinal=ordinal,
                    changed_state_fields=[
                        "der_availability",
                        "aggregate_generation_mw",
                    ],
                    materiality_metric="failed_der_count",
                    materiality_value=1.0,
                    materiality_threshold=1.0,
                    extra={"der_id": d.der_id},
                )
            )
        elif kind == "load_spike":
            self._load_spike_factor = max(0.0, float(p.intensity))
            if tick == p.trigger_tick:
                self._realized_events_this_tick.append(
                    self._declared_perturbation_event(
                        p=p,
                        tick=tick,
                        ordinal=ordinal,
                        changed_state_fields=[
                            "aggregate_demand_mw",
                            "balance_error_mw",
                        ],
                        materiality_metric="load_multiplier_delta",
                        materiality_value=float(p.intensity),
                        materiality_threshold=0.01,
                        extra={"intensity": float(p.intensity)},
                    )
                )
        elif kind == "price_spike":
            self._price_spike_factor = max(0.0, float(p.intensity))
            if tick == p.trigger_tick:
                self._realized_events_this_tick.append(
                    self._declared_perturbation_event(
                        p=p,
                        tick=tick,
                        ordinal=ordinal,
                        changed_state_fields=[
                            "grid_import_price_per_mwh",
                            "production_cost",
                        ],
                        materiality_metric="price_multiplier_delta",
                        materiality_value=float(p.intensity),
                        materiality_threshold=0.01,
                        extra={"intensity": float(p.intensity)},
                    )
                )
        elif kind == "forecast_bias":
            direction = p.target.get("bias_direction", "under-forecast")
            sign = 1.0 if direction == "under-forecast" else -1.0
            self._forecast_bias = sign * float(p.intensity)
            if tick == p.trigger_tick:
                self._realized_events_this_tick.append(
                    self._declared_perturbation_event(
                        p=p,
                        tick=tick,
                        ordinal=ordinal,
                        changed_state_fields=["forecast_bias"],
                        materiality_metric="forecast_bias_fraction",
                        materiality_value=abs(float(p.intensity)),
                        materiality_threshold=0.01,
                        extra={"bias_direction": str(direction)},
                    )
                )

    # ── Tool effects ─────────────────────────────────────────────────────

    def apply_tool_effect(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "set_battery_dispatch":
            return self._set_battery_dispatch(args)
        if name == "set_grid_exchange":
            return self._set_grid_exchange(args)
        if name == "curtail_der":
            return self._curtail_der(args)
        if name == "shed_load":
            return self._shed_load(args)
        if name == "set_der_reactive_power":
            # EMS backend has no AC power flow → Volt-Var is a no-op here.
            return {
                "_status": "unsupported_off_lv",
                "info": (
                    "set_der_reactive_power is only effective on the "
                    "microgrid_lv_voltage_6h power-flow tier"
                ),
            }
        return {"_status": "ack"}

    def _set_battery_dispatch(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._battery is None:
            return {"_status": "error", "error": "no_battery"}
        bid = str(args.get("battery_id", "batt0"))
        p_mw = float(args.get("p_mw", 0.0))
        # Reject only if the SoC/rate window cannot accommodate any of it.
        if p_mw > 0 and self._battery.soc_mwh >= self._battery.capacity_mwh - 1e-9:
            return {"_status": "error", "error": "soc_full", "battery_id": bid}
        if p_mw < 0 and self._battery.soc_mwh <= 1e-9:
            return {"_status": "error", "error": "soc_empty", "battery_id": bid}
        self._battery.setpoint_mw = p_mw
        return {
            "battery_id": bid,
            "p_mw": round(p_mw, 3),
            "soc_mwh": round(self._battery.soc_mwh, 3),
        }

    def _set_grid_exchange(self, args: dict[str, Any]) -> dict[str, Any]:
        # Tick-aware islanding rejection is enforced by the tool handler
        # (``_h_set_grid_exchange``) using ``is_islanded_at(ctx.tick)`` because
        # perturbations apply after tools. The stored setpoint is ignored by
        # ``tick()`` while islanded regardless.
        p_mw = float(args.get("p_mw", 0.0))
        self._grid_setpoint = p_mw
        return {"p_mw": round(p_mw, 3), "queued": True}

    def _curtail_der(self, args: dict[str, Any]) -> dict[str, Any]:
        did = str(args.get("der_id", ""))
        d = self._ders.get(did)
        if d is None:
            return {"_status": "error", "error": "unknown_der", "der_id": did}
        target = float(args.get("target_mw", 0.0))
        if target < 0:
            return {"_status": "error", "error": "negative_target", "der_id": did}
        d.curtail_cap_mw = target
        return {"der_id": did, "target_mw": round(target, 3), "queued": True}

    def _shed_load(self, args: dict[str, Any]) -> dict[str, Any]:
        lid = str(args.get("load_id", ""))
        mw = float(args.get("mw", 0.0))
        if mw <= 0:
            return {"_status": "error", "error": "non_positive_shed", "mw": mw}
        load = self._loads.get(lid)
        if load is None:
            return {"_status": "error", "error": "unknown_load", "load_id": lid}
        load.shed_this_tick_mw += mw
        return {
            "load_id": lid,
            "shed_mw": round(mw, 3),
            "stakeholder_class": load.stakeholder_class,
            "criticality": load.criticality,
        }

    def _genset_fuel_cost(self) -> float:
        return self._genset.fuel_cost_per_mwh if self._genset else 0.0

    # ── Read-only helpers (noised forecast / investigate) ───────────────

    def forecast_for(self, horizon: int) -> list[dict[str, Any]]:
        """Noised PV/load/price forecast (documented bias/variance)."""
        out: list[dict[str, Any]] = []
        for k in range(max(1, int(horizon))):
            t = self._tick + k
            jitter = (
                ((_det_hash(self._seed, t, "fc") / 1000.0) - 0.5)
                * 2.0
                * self._forecast_sigma
            )
            load_true = self._profile_at("load_mw", t)
            pv_true = self._profile_at("pv_mw", t)
            price_true = self._profile_at("price", t)
            out.append(
                {
                    "tick": t,
                    "load_mw_forecast": round(
                        load_true * (1.0 + self._forecast_bias + jitter), 3
                    ),
                    "pv_mw_forecast": round(
                        max(0.0, pv_true * (1.0 - self._forecast_bias + jitter)), 3
                    ),
                    "price_forecast": round(price_true, 3),
                    "noised": True,
                }
            )
        return out

    def investigate_asset(self, asset_id: str) -> dict[str, Any]:
        """Reveal fogged SoC / availability for one asset."""
        self._revealed_assets.add(asset_id)
        if asset_id in ("batt0", "battery") and self._battery is not None:
            return {
                "asset_id": asset_id,
                "kind": "battery",
                "soc_mwh": round(self._battery.soc_mwh, 3),
                "capacity_mwh": self._battery.capacity_mwh,
            }
        if asset_id in self._ders:
            d = self._ders[asset_id]
            return {
                "asset_id": asset_id,
                "kind": d.kind,
                "failed": self._tick < d.failed_until,
                "curtail_cap_mw": d.curtail_cap_mw,
            }
        if asset_id in ("genset0", "genset") and self._genset is not None:
            return {
                "asset_id": asset_id,
                "kind": "genset",
                "available": self._genset.available,
                "committed": self._genset.committed,
            }
        return {"_status": "error", "error": "unknown_asset", "asset_id": asset_id}

    def reveal_asset(self, asset_id: str) -> None:
        self._revealed_assets.add(asset_id)

    def realized_events_for_tick(self) -> list[dict[str, Any]]:
        return list(self._realized_events_this_tick)

    # ── Snapshot ─────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        entities: dict[str, dict[str, Any]] = {}
        if self._battery is not None:
            entities["batt0"] = {
                "kind": "battery",
                "soc_mwh": round(self._battery.soc_mwh, 3),
                "capacity_mwh": self._battery.capacity_mwh,
                "max_charge_mw": self._battery.max_charge_mw,
                "max_discharge_mw": self._battery.max_discharge_mw,
                "commanded_p_mw": round(self._battery.setpoint_mw, 3),
                "applied_p_mw": round(self._battery_applied_mw, 3),
            }
        if self._genset is not None:
            entities["genset0"] = {
                "kind": "genset",
                "available": self._genset.available,
                "committed": self._genset.committed,
                "output_mw": round(self._genset.output_mw, 3),
                "max_mw": self._genset.max_mw,
            }
        entities["pcc"] = {
            "kind": "pcc",
            "external_grid_available": self._external_grid_available_at(self._tick),
            "connected": self._pcc_connected,
            "islanded": self._is_islanded(),
            "max_import_mw": self._grid_max_import,
            "commanded_exchange_mw": self._grid_setpoint,
        }
        for did, d in self._ders.items():
            entities[did] = {
                "kind": d.kind,
                "failed": self._tick < d.failed_until,
                "curtail_cap_mw": d.curtail_cap_mw,
            }
        for lid, load in self._loads.items():
            entities[lid] = {
                "kind": "load",
                "stakeholder_class": load.stakeholder_class,
                "criticality": load.criticality,
                "current_demand_mw": round(load.current_demand_mw, 3),
                "cumulative_shed_mwh": round(load.cumulative_shed_mwh, 3),
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
            "islanded": self._is_islanded(),
        }
        return {
            "tick": self._tick,
            "horizon": self._horizon,
            "entities": entities,
            "totals": totals,
        }

    # ── Cost roll-up / scoring ───────────────────────────────────────────

    def ground_truth_costs(self) -> dict[str, float]:
        if not self._tick_records:
            return {
                "production_cost": 0.0,
                "startup_cost": 0.0,
                "shed_penalty": 0.0,
                "balance_error_cost": 0.0,
            }
        prod = sum(r.production_cost for r in self._tick_records)
        startup = sum(r.startup_cost for r in self._tick_records)
        shed = sum(r.shed_penalty for r in self._tick_records)
        balance = sum(abs(r.balance_error_mw) for r in self._tick_records) * 200.0
        return {
            "production_cost": round(prod, 3),
            "startup_cost": round(startup, 3),
            "shed_penalty": round(shed, 3),
            "balance_error_cost": round(balance, 3),
        }

    def scoring_records(self) -> list[dict[str, Any]]:
        """Per-tick rows: the canonical 14-key contract with the §7 alias
        mapping. Keys declared ``honest_zero`` for this family (the four
        power-flow keys on pymgrid families) are forced 0. ``done`` is
        early-guarded ``bool(done and tick < horizon-1)`` so a normal
        horizon-end tick is never miscounted as a catastrophic collapse —
        but a genuine mid-horizon islanding collapse is surfaced.
        """
        hz = self._honest_zero_keys
        rows: list[dict[str, Any]] = []
        for r in self._tick_records:
            row = {
                "tick": r.tick,
                "aggregate_demand_mw": r.aggregate_demand_mw,
                "aggregate_generation_mw": r.aggregate_generation_mw,
                "balance_error_mw": r.balance_error_mw,
                "reserves_required_mw": r.reserves_required_mw,
                "reserves_procured_mw": r.reserves_procured_mw,
                "production_cost": r.production_cost,
                "startup_cost": r.startup_cost,
                "shed_penalty": r.shed_penalty,
                # pymgrid is an aggregate energy-balance EMS — no AC/DC power
                # flow → the four power-flow keys are honestly 0 (§7).
                "rho_max": 0.0,
                "n_overloads": 0,
                "n_voltage_violations": 0,
                "n_disconnected_lines": 0,
                "done": bool(r.done and r.tick < self._horizon - 1),
            }
            for k in hz:
                if k in row:
                    row[k] = 0
            rows.append(row)
        return rows

    def per_load_shed_mwh(self) -> dict[str, float]:
        """Microgrid analogue of ``per_load_shed_mwh`` — deterministic map."""
        return {
            lid: round(load.cumulative_shed_mwh, 3) for lid, load in self._loads.items()
        }

    def _bind_runtime_source_asset(self) -> None:
        """Reopen the locked NPZ and verify the exact four-channel window."""
        if self._seed_obj is None:
            return
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
        repo_root = Path(__file__).resolve().parents[3]
        path = (repo_root / declared_npz).resolve()
        if not path.is_file():
            return
        recipe = (self._seed_obj.backend_config or {}).get("derivation_recipe") or {}
        start = int(recipe.get("profile_start_index", 0) or 0)
        stop = start + self._horizon
        import numpy as np

        runtime_profiles: dict[str, list[float]] = {}
        with np.load(path, allow_pickle=False) as data:
            for key in ("load_mw", "pv_mw", "wind_mw", "price"):
                if key not in data:
                    return
                values = data[key].ravel()
                if stop > len(values):
                    return
                runtime_profiles[key] = [
                    round(float(value), 5) for value in values[start:stop]
                ]
        expected_profiles = {
            key: [round(float(value), 5) for value in self._profiles.get(key, [])]
            for key in ("load_mw", "pv_mw", "wind_mw", "price")
        }
        if runtime_profiles != expected_profiles:
            return
        from ..seeds.from_pymgrid import ems_source_window_sha256

        window_sha256 = ems_source_window_sha256(**runtime_profiles)
        if window_sha256 != str(recipe.get("source_window_sha256") or ""):
            return
        relative = str(path.relative_to(repo_root))
        self._runtime_source_assets = [
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "role": "derivation_input",
            }
        ]
        self._runtime_source_verified = True

    def _source_state_digest(
        self,
        *,
        record: _EmsTickRecord,
    ) -> str:
        return _semantic_digest(
            {
                "tick": record.tick,
                "aggregate_demand_mw": record.aggregate_demand_mw,
                "aggregate_generation_mw": record.aggregate_generation_mw,
                "production_cost": record.production_cost,
                "balance_error_mw": record.balance_error_mw,
                "source_values": {
                    key: round(self._profile_at(key, record.tick), 5)
                    for key in ("load_mw", "pv_mw", "wind_mw", "price")
                },
            }
        )

    def _source_schedule_event(
        self,
        *,
        record: _EmsTickRecord,
    ) -> dict[str, Any] | None:
        """Expose an actually consumed profile interval as native state.

        This is emitted only after reset reopened and matched the locked NPZ
        window, so scenario configuration cannot masquerade as a source
        schedule transition.
        """
        if not self._runtime_source_verified or self._seed_obj is None:
            return None
        current = {
            key: float(self._profile_at(key, record.tick))
            for key in ("load_mw", "pv_mw", "wind_mw", "price")
        }
        previous = {
            key: (
                float(self._profile_at(key, record.tick - 1))
                if record.tick > 0
                else current[key]
            )
            for key in current
        }
        relative_delta = {
            key: abs(current[key] - previous[key]) / max(abs(previous[key]), 1e-9)
            for key in current
        }
        changed_state_fields: list[str] = []
        if relative_delta["load_mw"] > 1e-9:
            changed_state_fields.append("aggregate_demand_mw")
        if relative_delta["pv_mw"] > 1e-9 or relative_delta["wind_mw"] > 1e-9:
            changed_state_fields.append("aggregate_generation_mw")
        if relative_delta["price"] > 1e-9:
            changed_state_fields.append("grid_import_price_per_mwh")
        value = max(relative_delta.values(), default=0.0)
        return {
            "type": "source_profile_interval",
            "event_id": (f"ems-source-profile:{self._seed_obj.seed_id}:{record.tick}"),
            "origin": "source_schedule",
            "event_class": "telemetry",
            "tick": record.tick,
            # This is provenance/telemetry for an interval already consumed by
            # the simulator, not a new task or alarm.  Native control cadence,
            # declared perturbations and the agent's standing-plan review own
            # decision wakeups.
            "decision_required": False,
            "actionable": False,
            "changed_state_fields": changed_state_fields,
            "materiality_metric": "max_relative_profile_channel_delta",
            "materiality_value": value,
            "materiality_threshold": 1e-9,
            "materiality_passed": bool(changed_state_fields),
            "source_values": {key: round(value, 5) for key, value in current.items()},
            "derived_state": {
                "aggregate_demand_mw": record.aggregate_demand_mw,
                "aggregate_generation_mw": record.aggregate_generation_mw,
                "production_cost": record.production_cost,
            },
        }

    def protocol21_source_trace(self) -> dict[str, Any]:
        """Prove locked NSRDB/OEDI windows drove native EMS state."""
        recipe = (
            (self._seed_obj.backend_config or {}).get("derivation_recipe", {})
            if self._seed_obj
            else {}
        )
        opened_paths = [asset["path"] for asset in self._runtime_source_assets]
        opened_hashes = {
            asset["path"]: asset["sha256"] for asset in self._runtime_source_assets
        }
        source_effect = bool(
            self._runtime_source_verified
            and self._source_consumption_ticks
            and self._runtime_source_events
        )
        blockers = (
            []
            if self._runtime_source_verified
            else ["runtime_source_asset_window_mismatch"]
        )
        if self._runtime_source_verified and not self._source_consumption_ticks:
            blockers.append("runtime_source_consumption_unobserved")
        semantic = {
            "opened_source_sha256": opened_hashes,
            "consumed_window_sha256": recipe.get("source_window_sha256"),
            "consumption_ticks": self._source_consumption_ticks,
            "post_source_state_digests": self._post_source_state_digests,
            "runtime_source_events": self._runtime_source_events,
        }
        return {
            "status": "passed" if not blockers else "held",
            "proof_kind": "baked_source_window_backend_trace",
            "runtime_opened_assets": list(self._runtime_source_assets),
            "opened_source_paths": opened_paths,
            "opened_source_sha256": opened_hashes,
            "consumed_source_hashes": {},
            "lineage_source_hashes": opened_hashes,
            "consumed_window_sha256": recipe.get("source_window_sha256"),
            "recipe_version": recipe.get("pipeline_version"),
            "consumed_channels": (
                ["load_mw", "pv_mw", "wind_mw", "price"]
                if self._runtime_source_verified
                else []
            ),
            "derived_backend_state_fields": (
                [
                    "aggregate_demand_mw",
                    "aggregate_generation_mw",
                    "production_cost",
                    "balance_error_mw",
                ]
                if self._runtime_source_verified
                else []
            ),
            "consumption_ticks": list(self._source_consumption_ticks),
            "post_source_state_digests": list(self._post_source_state_digests),
            "runtime_source_events": list(self._runtime_source_events),
            "source_state_effect_observed": source_effect,
            "state_effect_observed": source_effect,
            "deterministic_source_trace": True,
            "trace_semantic_digest": _semantic_digest(semantic),
            "runtime_trace_observed": bool(
                self._runtime_source_verified and self._source_consumption_ticks
            ),
            "evidence_from_scenario_config_only": False,
            "blockers": blockers,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    def _tick_h(self) -> float:
        return self._tick_minutes / 60.0

    def _profile_at(self, key: str, tick: int | None = None) -> float:
        arr = self._profiles.get(key) or []
        if not arr:
            return 0.0
        t = self._tick if tick is None else tick
        return float(arr[min(max(0, t), len(arr) - 1)])

    # ── Optional pymgrid cross-check (LGPL-3.0; dynamic-link) ───────────

    def evaluate_with_pymgrid(self) -> dict[str, Any]:
        """Cross-check this scenario against a pymgrid ``Microgrid`` run.

        This is the ONLY path that requires pymgrid; the simulator never
        needs it to step. Raises the typed ``MicrogridBackendUnavailable``
        when pymgrid is absent so callers / tests skip cleanly
        (graceful-skip, spec §"Runtime gate").
        """
        if not _pymgrid_available():
            raise MicrogridBackendUnavailable(
                "pymgrid is required for the EMS cross-check path but is not "
                "importable. Install via `pip install pymgrid` (LGPL-3.0, "
                "dynamic-link). The deterministic EMS simulator does NOT "
                "require pymgrid to run."
            )
        from pymgrid import Microgrid  # type: ignore[import]
        from pymgrid.modules import (  # type: ignore[import]
            BatteryModule,
            GensetModule,
            GridModule,
            LoadModule,
            RenewableModule,
        )

        load = self._profiles.get("load_mw") or [1.0]
        pv = self._profiles.get("pv_mw") or [0.0]
        price = self._profiles.get("price") or [40.0]
        n = max(1, self._horizon)

        modules: list[Any] = []
        modules.append(
            LoadModule(
                time_series=[[-abs(load[min(i, len(load) - 1)])] for i in range(n)]
            )
        )
        modules.append(
            RenewableModule(
                time_series=[[abs(pv[min(i, len(pv) - 1)])] for i in range(n)]
            )
        )
        if self._battery is not None:
            modules.append(
                BatteryModule(
                    min_capacity=0.0,
                    max_capacity=max(0.1, self._battery.capacity_mwh),
                    max_charge=max(0.1, self._battery.max_charge_mw),
                    max_discharge=max(0.1, self._battery.max_discharge_mw),
                    efficiency=self._battery.efficiency,
                    init_soc=0.5,
                )
            )
        if self._genset is not None and self._genset.available:
            modules.append(
                GensetModule(
                    running_min_production=self._genset.min_mw,
                    running_max_production=max(0.1, self._genset.max_mw),
                    genset_cost=self._genset.fuel_cost_per_mwh,
                )
            )
        import numpy as np  # local import

        grid_ts = np.array(
            [
                [
                    self._grid_max_import,
                    self._grid_max_export,
                    price[min(i, len(price) - 1)],
                    1.0,
                ]
                for i in range(n)
            ],
            dtype=float,
        )
        modules.append(
            GridModule(
                max_import=self._grid_max_import,
                max_export=self._grid_max_export,
                time_series=grid_ts,
            )
        )
        mg = Microgrid(modules)
        mg.reset()
        return {
            "pymgrid_n_modules": int(mg.n_modules),
            "horizon": n,
            "lock": "pymgrid-dynamic-link-LGPL-3.0",
        }
