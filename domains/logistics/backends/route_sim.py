"""
domains.logistics.backends.route_sim — Deterministic seeded route/demand sim.

This **pure-Python** simulator IS the logistics environment (spec §9): no
PyVRP, no OR-Tools, no wall-clock dependency. Demand realization, disruption
timing, breakdown/arc/urgent selection are all locked to ``seed_id`` so two
``reset`` + ``tick`` loops with identical inputs produce byte-identical
records — the contract counterfactual replay relies on.

Method surface mirrors
``domains.power_grid.backends.pglib_uc_synthetic.PglibUcSyntheticBackend`` /
``domains.disaster.backends.mock_rcrs.MockRcrsBackend`` (reset / tick /
snapshot / apply_tool_effect / ground_truth_costs / scoring_records /
forecast_for / per-customer accumulator + a delayed-effect queue) so the
logistics adapter is structurally identical to the other adapters.

PyVRP (MIT) is imported lazily and used ONLY on the fixed-plan route-cost
*evaluation* path (``evaluate_fixed_plan_cost``); when absent the typed
``LogisticsBackendUnavailable`` is raised there only — the simulator itself
never needs it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .pyvrp_source import resolve_pyvrp_source_instance

if TYPE_CHECKING:  # pragma: no cover
    from ..seeds.schema import LogisticsScenarioSeed


# ── Optional PyVRP (MIT) — cost-eval path only ──────────────────────────────
PYVRP_AVAILABLE = False
_PYVRP_IMPORT_ERROR = ""
try:  # pragma: no cover - exercised only when pyvrp is installed
    from pyvrp import Model as _PyVRPModel  # type: ignore[import]

    PYVRP_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _PyVRPModel = None  # type: ignore[assignment]
    _PYVRP_IMPORT_ERROR = f"pyvrp import failed: {exc!r}"


class LogisticsBackendUnavailable(RuntimeError):
    """Raised when PyVRP is required (fixed-plan cost eval) but not importable."""


# ── Cost / penalty calibration (proxy units; same scale as other domains) ───
_COST_PER_DISTANCE = 1.0
_TRAVEL_TIME_PREMIUM = 0.5  # extra cost per distance unit under traffic delay
_DISPATCH_FIXED_COST = 100.0
_SPOT_PREMIUM_PER_UNIT = 50.0
_DROP_PENALTY_BASE = 200.0
_RESERVE_TARGET_FRACTION = 0.10
_ROUTE_EVENT_REGISTRY = MappingProxyType(
    {
        "vehicle_breakdown": "safety",
        "demand_surge": "alarm",
        "blocked_arc": "alarm",
        "traffic_delay": "alarm",
        "urgent_order": "task",
    }
)


def _det_hash(seed: int, tick: int, key: str) -> int:
    body = f"{int(seed)}|{int(tick)}|{key}".encode()
    return int.from_bytes(hashlib.sha256(body).digest()[:4], "big") % 1000


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class _Vehicle:
    vid: str
    capacity: float
    remaining_capacity: float
    route: list[str] = field(default_factory=list)  # remaining customer ids
    pos: tuple[float, float] = (0.0, 0.0)
    active: bool = True
    is_standby: bool = False
    broken: bool = False
    broken_hidden: bool = False
    service_rate: int = 2


@dataclass
class _Customer:
    cid: str
    x: float
    y: float
    demand: float
    tw_early: float
    tw_late: float
    due_tick: int
    priority_class: str = "standard"
    criticality: float = 0.3
    served: bool = False
    dropped: bool = False
    held_until: int = -1
    blocked: bool = False
    assigned_vehicle: str | None = None


@dataclass
class _RouteTickRecord:
    tick: int
    aggregate_demand: float = 0.0
    served_demand: float = 0.0
    unmet_demand: float = 0.0
    required_standby: float = 0.0
    procured_standby: float = 0.0
    routing_cost: float = 0.0
    dispatch_fixed_cost: float = 0.0
    drop_penalty: float = 0.0
    max_utilization: float = 0.0
    n_capacity_violations: int = 0
    n_time_window_violations: int = 0
    n_failed_routes: int = 0
    done: bool = False
    realized_events: list[dict[str, Any]] = field(default_factory=list)


class RouteDemandSimulator:
    """Pure-Python deterministic fleet-dispatch simulator.

    Family behaviour is configured by the seed's ``backend_config`` flags
    (``has_time_windows`` / ``models_standby`` / ``is_lastmile`` /
    ``honest_zero_keys``); the three ``pyvrp_*`` subclasses simply set their
    canonical ``backend_kind`` label.
    """

    backend_kind = "route_sim"

    def __init__(self) -> None:
        self._seed_obj: LogisticsScenarioSeed | None = None
        self._tick = 0
        self._horizon = 8
        self._seed = 0
        self._depot: tuple[float, float] = (0.0, 0.0)
        self._vehicles: dict[str, _Vehicle] = {}
        self._customers: dict[str, _Customer] = {}
        self._tick_records: list[_RouteTickRecord] = []
        self._has_time_windows = False
        self._models_standby = False
        self._is_lastmile = False
        self._honest_zero_keys: set[str] = set()
        self._capacity = 0.0
        self._service_time = 0.0
        # disruption state
        self._traffic_mult = 1.0
        self._traffic_until = -1
        self._demand_surge_mult = 1.0
        self._surge_until = -1
        # procured standby capacity (from spot carrier / standby dispatch)
        self._procured_standby = 0.0
        self._pending_effects: list[tuple[int, str, dict[str, Any]]] = []
        self._cumulative_unmet: dict[str, float] = {}
        self._realized_events_this_tick: list[dict[str, Any]] = []
        self._urgent_counter = 0
        self._drop_accum = 0.0
        self._tick_extra_cost = {"dispatch_fixed": 0.0, "spot_premium": 0.0}
        self._revealed_vehicles: set[str] = set()
        self._source_resolution: dict[str, Any] | None = None
        self._initial_state_digest = ""
        self._post_source_state_digests: list[dict[str, Any]] = []
        self._world_evolution_records: list[dict[str, Any]] = []
        self._action_effects: list[dict[str, Any]] = []
        self._unbound_action_effects: list[dict[str, Any]] = []
        self._pending_agent_events: list[dict[str, Any]] = []

    # ── Reset ───────────────────────────────────────────────────────────

    def reset(self, scenario_seed: LogisticsScenarioSeed) -> None:
        self._seed_obj = scenario_seed
        self._tick = 0
        self._horizon = int(scenario_seed.horizon_ticks)
        self._seed = int(scenario_seed.seed)
        cfg = scenario_seed.backend_config
        net = cfg.get("network", {})
        self._source_resolution = None
        source_files = list(scenario_seed.provenance.files or [])
        has_direct_source = bool(
            source_files and not str(source_files[0]).startswith("<embedded-synthetic:")
        )
        if has_direct_source and self.backend_kind in {
            "pyvrp_cvrp",
            "pyvrp_vrptw",
        }:
            self._source_resolution = resolve_pyvrp_source_instance(
                source_path=str(source_files[0]),
                source_sha256=None,
                instance_kind=(
                    "vrptw" if self.backend_kind == "pyvrp_vrptw" else "cvrp"
                ),
                backend_config=cfg,
                repo_root=Path(__file__).resolve().parents[3],
            )
            net = self._source_resolution["parser_representation"]["network"]
        self._has_time_windows = bool(cfg.get("has_time_windows", False))
        self._models_standby = bool(cfg.get("models_standby", False))
        self._is_lastmile = bool(cfg.get("is_lastmile", False))
        self._honest_zero_keys = set(cfg.get("honest_zero_keys", []) or [])
        self._capacity = float(net.get("capacity", 0.0) or 0.0)
        self._service_time = float(net.get("service_time", 0.0) or 0.0)
        depot = net.get("depot", {"x": 0.0, "y": 0.0})
        self._depot = (float(depot["x"]), float(depot["y"]))

        self._tick_records.clear()
        self._traffic_mult = 1.0
        self._traffic_until = -1
        self._demand_surge_mult = 1.0
        self._surge_until = -1
        self._procured_standby = 0.0
        self._pending_effects.clear()
        self._cumulative_unmet.clear()
        self._realized_events_this_tick = []
        self._urgent_counter = 0
        self._drop_accum = 0.0
        self._tick_extra_cost = {"dispatch_fixed": 0.0, "spot_premium": 0.0}
        self._revealed_vehicles = set()
        self._post_source_state_digests = []
        self._world_evolution_records = []
        self._action_effects = []
        self._unbound_action_effects = []
        self._pending_agent_events = []

        # Customers (with priority classes mapped from load_assignments).
        prio_by_id = {
            c.load_id: (c.stakeholder_class, c.criticality)
            for c in scenario_seed.load_assignments
        }
        self._customers = {}
        raw_customers = list(net.get("customers", []))
        for i, rc in enumerate(raw_customers):
            cid = str(rc["id"])
            cls, crit = prio_by_id.get(cid, ("standard", 0.3))
            # Deterministic delivery deadline spread across the horizon.
            due_tick = 1 + (i % max(1, self._horizon - 1))
            self._customers[cid] = _Customer(
                cid=cid,
                x=float(rc["x"]),
                y=float(rc["y"]),
                demand=float(rc["demand"]),
                tw_early=float(rc.get("tw_early", 0.0)),
                tw_late=float(rc.get("tw_late", 1e9)),
                due_tick=due_tick,
                priority_class=str(cls),
                criticality=float(crit),
            )
            self._cumulative_unmet[cid] = 0.0

        # Vehicles: active fleet + (optionally) one standby launched on demand.
        n_active = int(net.get("n_vehicles", 2) or 2)
        cust_ids = list(self._customers.keys())
        self._vehicles = {}
        for v in range(n_active):
            vid = f"v{v}"
            assigned = cust_ids[v::n_active]
            ordered = self._nearest_neighbor_order(assigned)
            svc_rate = max(1, math.ceil(len(ordered) / max(1, self._horizon - 1)))
            self._vehicles[vid] = _Vehicle(
                vid=vid,
                capacity=self._capacity,
                remaining_capacity=self._capacity,
                route=ordered,
                pos=self._depot,
                active=True,
                is_standby=False,
                service_rate=svc_rate,
            )
            for cid in assigned:
                self._customers[cid].assigned_vehicle = vid
        if self._models_standby:
            self._vehicles[f"v{n_active}"] = _Vehicle(
                vid=f"v{n_active}",
                capacity=self._capacity,
                remaining_capacity=self._capacity,
                route=[],
                pos=self._depot,
                active=False,
                is_standby=True,
                service_rate=2,
            )
        self._initial_state_digest = self._state_digest()
        self._post_source_state_digests.append(
            {"tick": 0, "sha256": self._initial_state_digest}
        )

    def _nearest_neighbor_order(self, cust_ids: list[str]) -> list[str]:
        remaining = list(cust_ids)
        ordered: list[str] = []
        pos = self._depot
        while remaining:
            nxt = min(
                remaining,
                key=lambda cid: (
                    _dist(pos, (self._customers[cid].x, self._customers[cid].y)),
                    cid,
                ),
            )
            ordered.append(nxt)
            pos = (self._customers[nxt].x, self._customers[nxt].y)
            remaining.remove(nxt)
        return ordered

    # ── Delayed-effect queue (dispatch_vehicle / hire_spot_carrier) ─────

    def queue_capacity_effect(
        self, *, due_tick: int, kind: str, payload: dict[str, Any]
    ) -> None:
        """Queue a delayed capacity arrival, materialized at ``due_tick``."""
        self._pending_effects.append((int(due_tick), str(kind), dict(payload)))

    def _drain_effects(self, current_tick: int) -> None:
        kept: list[tuple[int, str, dict[str, Any]]] = []
        for due, kind, payload in self._pending_effects:
            if due > current_tick:
                kept.append((due, kind, payload))
                continue
            if kind == "dispatch_vehicle":
                vid = str(payload.get("vehicle_id", ""))
                veh = self._vehicles.get(vid)
                if veh is not None and veh.is_standby and not veh.active:
                    before_digest = self._state_digest()
                    veh.active = True
                    self._procured_standby += veh.capacity
                    self._record_tick_cost("dispatch_fixed", _DISPATCH_FIXED_COST)
                    after_digest = self._state_digest()
                    tool_call = payload.get("_tool_call") or {}
                    call_id = str(tool_call.get("call_id") or "")
                    requested_action = {
                        "name": str(tool_call.get("tool_name") or kind),
                        "args": dict(tool_call.get("args") or {}),
                    }
                    self._realized_events_this_tick.append(
                        {
                            "type": "vehicle_dispatched",
                            "event_id": (
                                f"vehicle_dispatched:{call_id}:{current_tick}"
                                if call_id
                                else f"vehicle_dispatched:{vid}:{current_tick}"
                            ),
                            "origin": "agent_caused",
                            "agent_caused": True,
                            "event_class": "agent_outcome",
                            "decision_required": False,
                            "actionable": False,
                            "tick": current_tick,
                            "vehicle_id": vid,
                            "capacity": veh.capacity,
                            **(
                                {
                                    "call_id": call_id,
                                    "tool_name": requested_action["name"],
                                    "requested_action": requested_action,
                                    "before_state_digest": before_digest,
                                    "after_state_digest": after_digest,
                                    "effect_tick": current_tick,
                                    "outcome_tick": current_tick,
                                }
                                if call_id and before_digest != after_digest
                                else {}
                            ),
                        }
                    )
            elif kind == "hire_spot_carrier":
                units = float(payload.get("capacity_units", 0.0) or 0.0)
                before_digest = self._state_digest()
                self._procured_standby += units
                self._record_tick_cost("spot_premium", units * _SPOT_PREMIUM_PER_UNIT)
                after_digest = self._state_digest()
                tool_call = payload.get("_tool_call") or {}
                call_id = str(tool_call.get("call_id") or "")
                requested_action = {
                    "name": str(tool_call.get("tool_name") or kind),
                    "args": dict(tool_call.get("args") or {}),
                }
                self._realized_events_this_tick.append(
                    {
                        "type": "spot_carrier_arrived",
                        "event_id": (
                            f"spot_carrier_arrived:{call_id}:{current_tick}"
                            if call_id
                            else f"spot_carrier_arrived:{current_tick}"
                        ),
                        "origin": "agent_caused",
                        "agent_caused": True,
                        "event_class": "agent_outcome",
                        "decision_required": False,
                        "actionable": False,
                        "tick": current_tick,
                        "region": payload.get("region"),
                        "capacity_units": units,
                        **(
                            {
                                "call_id": call_id,
                                "tool_name": requested_action["name"],
                                "requested_action": requested_action,
                                "before_state_digest": before_digest,
                                "after_state_digest": after_digest,
                                "effect_tick": current_tick,
                                "outcome_tick": current_tick,
                            }
                            if call_id and before_digest != after_digest
                            else {}
                        ),
                    }
                )
        self._pending_effects = kept

    # ── Tick ────────────────────────────────────────────────────────────

    def tick(self, current_tick: int) -> _RouteTickRecord:
        assert self._seed_obj is not None
        self._tick = int(current_tick)
        self._realized_events_this_tick = list(self._pending_agent_events)
        self._pending_agent_events = []
        self._tick_extra_cost = {"dispatch_fixed": 0.0, "spot_premium": 0.0}

        # 1) drain matured delayed effects.
        self._drain_effects(self._tick)

        # 2) apply perturbations firing this tick.
        self._apply_perturbations_at_tick(self._tick)
        # expire transient disruptions.
        if self._traffic_until >= 0 and self._tick >= self._traffic_until:
            self._traffic_mult = 1.0
        if self._surge_until >= 0 and self._tick >= self._surge_until:
            self._demand_surge_mult = 1.0

        # 3) vehicle movement / service.
        routing_cost = 0.0
        served_demand = 0.0
        n_cap_viol = 0
        n_tw_viol = 0
        max_util = 0.0
        for vid in sorted(self._vehicles.keys()):
            veh = self._vehicles[vid]
            if not veh.active or veh.broken or not veh.route:
                if veh.capacity > 0:
                    util = (veh.capacity - veh.remaining_capacity) / veh.capacity
                    max_util = max(max_util, util)
                continue
            served_count = 0
            new_route: list[str] = []
            for cid in veh.route:
                cust = self._customers.get(cid)
                if cust is None or cust.served or cust.dropped:
                    continue
                if served_count >= veh.service_rate:
                    new_route.append(cid)
                    continue
                if cust.held_until > self._tick:
                    new_route.append(cid)
                    continue
                if cust.blocked:
                    new_route.append(cid)  # cannot traverse; stays stranded
                    continue
                leg = _dist(veh.pos, (cust.x, cust.y))
                routing_cost += leg * _COST_PER_DISTANCE * self._traffic_mult
                if self._traffic_mult > 1.0:
                    routing_cost += (
                        leg * _TRAVEL_TIME_PREMIUM * (self._traffic_mult - 1.0)
                    )
                if veh.remaining_capacity + 1e-9 < cust.demand:
                    n_cap_viol += 1  # over-capacity service
                veh.remaining_capacity = max(0.0, veh.remaining_capacity - cust.demand)
                veh.pos = (cust.x, cust.y)
                cust.served = True
                served_demand += cust.demand
                served_count += 1
                completion = {
                    "type": "customer_service_completed",
                    "event_id": f"customer_service_completed:{cid}:{self._tick}",
                    "origin": "endogenous_completion",
                    "tick": self._tick,
                    "customer_id": cid,
                    "vehicle_id": vid,
                    "changed_state_fields": [
                        "unassigned_customers",
                        "route_assignments",
                        "vehicle_loads",
                        "remaining_capacity",
                        "route_distance",
                    ],
                    "materiality_passed": False,
                }
                self._realized_events_this_tick.append(completion)
                self._world_evolution_records.append(dict(completion))
                if self._has_time_windows and self._tick > cust.due_tick:
                    n_tw_viol += 1
            veh.route = new_route
            if veh.capacity > 0:
                util = (veh.capacity - veh.remaining_capacity) / veh.capacity
                max_util = max(max_util, util)

        # 4) aggregate demand / unmet.
        surge = self._demand_surge_mult
        aggregate_demand = round(
            sum(
                c.demand
                for c in self._customers.values()
                if c.due_tick <= self._tick and not c.dropped
            )
            * surge,
            3,
        )
        unmet = 0.0
        for c in self._customers.values():
            if c.served or c.dropped:
                continue
            if c.due_tick <= self._tick:
                unmet += c.demand
                self._cumulative_unmet[c.cid] = (
                    self._cumulative_unmet.get(c.cid, 0.0) + c.demand
                )
        unmet = round(unmet * surge, 3)

        # 5) failed routes: broken active vehicles + blocked unreachable stops.
        broken_active = sum(1 for v in self._vehicles.values() if v.active and v.broken)
        blocked_unreached = sum(
            1
            for c in self._customers.values()
            if c.blocked and not c.served and not c.dropped
        )
        n_failed = broken_active + blocked_unreached

        # 6) reserves (honest-0 when not modeled).
        total_active_capacity = sum(
            v.capacity for v in self._vehicles.values() if v.active and not v.broken
        )
        if self._models_standby:
            required_standby = round(
                _RESERVE_TARGET_FRACTION * total_active_capacity, 3
            )
        else:
            required_standby = 0.0
        procured_standby = (
            round(self._procured_standby, 3) if self._models_standby else 0.0
        )

        # 7) terminal / infeasible.
        all_done = all(c.served or c.dropped for c in self._customers.values())
        active_alive = any(v.active and not v.broken for v in self._vehicles.values())
        infeasible = not active_alive and not all_done
        done = (self._tick >= self._horizon - 1) or all_done or infeasible

        record = _RouteTickRecord(
            tick=self._tick,
            aggregate_demand=aggregate_demand,
            served_demand=round(served_demand, 3),
            unmet_demand=unmet,
            required_standby=required_standby,
            procured_standby=procured_standby,
            routing_cost=round(routing_cost + self._tick_extra_cost["spot_premium"], 3),
            dispatch_fixed_cost=round(self._tick_extra_cost["dispatch_fixed"], 3),
            drop_penalty=0.0,  # drops are recorded at call time via _drop accumulator
            max_utilization=round(max_util, 4),
            n_capacity_violations=int(n_cap_viol),
            n_time_window_violations=int(n_tw_viol),
            n_failed_routes=int(n_failed),
            done=bool(done),
            realized_events=list(self._realized_events_this_tick),
        )
        # Fold any drop penalties accrued (via apply_tool_effect) this tick.
        record.drop_penalty = round(self._drop_penalty_this_tick(), 3)
        self._reset_drop_penalty_accumulator()
        self._tick_records.append(record)
        return record

    # drop-penalty accumulator (drops happen during apply_tool_effect, which
    # runs before tick() in the adapter step loop).
    _drop_accum: float = 0.0

    def _drop_penalty_this_tick(self) -> float:
        return self._drop_accum

    def _reset_drop_penalty_accumulator(self) -> None:
        self._drop_accum = 0.0

    def _record_tick_cost(self, kind: str, amount: float) -> None:
        if not hasattr(self, "_tick_extra_cost"):
            self._tick_extra_cost = {"dispatch_fixed": 0.0, "spot_premium": 0.0}
        self._tick_extra_cost[kind] = self._tick_extra_cost.get(kind, 0.0) + amount

    # ── Perturbations ───────────────────────────────────────────────────

    def _apply_perturbations_at_tick(self, tick: int) -> None:
        assert self._seed_obj is not None
        for p in self._seed_obj.perturbations:
            if not (p.trigger_tick <= tick < p.trigger_tick + p.duration_ticks):
                continue
            self._apply_one_perturbation(p, tick)

    def _apply_one_perturbation(self, p: Any, tick: int) -> None:
        kind = str(p.kind)
        if kind == "vehicle_breakdown":
            if tick != p.trigger_tick:
                return
            idx = int(p.target.get("vehicle_index", 0))
            vid = f"v{idx}"
            veh = self._vehicles.get(vid)
            if veh is not None and not veh.broken:
                before = self._state_digest()
                veh.broken = True
                veh.broken_hidden = bool(p.hidden)
                self._append_declared_perturbation_event(
                    p,
                    tick,
                    event_type="vehicle_breakdown",
                    target={"vehicle_id": vid, **dict(p.target or {})},
                    changed_state_fields=[
                        "vehicle_status",
                        "available_vehicle_capacity",
                        "failed_routes",
                    ],
                    materiality_metric="vehicle_breakdown_duration_ticks",
                    materiality_value=float(max(1, int(p.duration_ticks))),
                    materiality_threshold=1.0,
                    before_digest=before,
                )
        elif kind == "demand_surge":
            before = self._state_digest()
            self._demand_surge_mult = max(1.0, float(p.intensity))
            self._surge_until = p.trigger_tick + p.duration_ticks
            if tick == p.trigger_tick:
                self._append_declared_perturbation_event(
                    p,
                    tick,
                    event_type="demand_surge",
                    target=dict(p.target or {}),
                    changed_state_fields=[
                        "demand_surge_multiplier",
                        "aggregate_demand",
                        "unmet_demand",
                    ],
                    materiality_metric="demand_surge_multiplier",
                    materiality_value=float(p.intensity),
                    materiality_threshold=1.0,
                    before_digest=before,
                )
        elif kind == "blocked_arc":
            if tick != p.trigger_tick:
                return
            idx = int(p.target.get("customer_index", 0))
            cust_ids = list(self._customers.keys())
            if cust_ids:
                cid = cust_ids[idx % len(cust_ids)]
                before = self._state_digest()
                self._customers[cid].blocked = True
                self._append_declared_perturbation_event(
                    p,
                    tick,
                    event_type="blocked_arc",
                    target={"customer_id": cid, **dict(p.target or {})},
                    changed_state_fields=[
                        "blocked_customers",
                        "route_feasibility",
                        "failed_routes",
                    ],
                    materiality_metric="blocked_customer_count",
                    materiality_value=1.0,
                    materiality_threshold=1.0,
                    before_digest=before,
                )
        elif kind == "traffic_delay":
            before = self._state_digest()
            self._traffic_mult = max(1.0, float(p.intensity))
            self._traffic_until = p.trigger_tick + p.duration_ticks
            if tick == p.trigger_tick:
                self._append_declared_perturbation_event(
                    p,
                    tick,
                    event_type="traffic_delay",
                    target=dict(p.target or {}),
                    changed_state_fields=[
                        "travel_time_multiplier",
                        "route_cost",
                    ],
                    materiality_metric="traffic_delay_multiplier",
                    materiality_value=float(p.intensity),
                    materiality_threshold=1.0,
                    before_digest=before,
                )
        elif kind == "urgent_order":
            if tick != p.trigger_tick:
                return
            self._urgent_counter += 1
            new_cid = f"urgent_{self._urgent_counter}"
            # Deterministic location near the depot; tight deadline.
            jitter = _det_hash(self._seed, tick, f"urgent|{new_cid}")
            ang = jitter % 360
            rad = 8.0 + (jitter % 12)
            cust = _Customer(
                cid=new_cid,
                x=round(self._depot[0] + rad * math.cos(math.radians(ang)), 3),
                y=round(self._depot[1] + rad * math.sin(math.radians(ang)), 3),
                demand=float(2 + jitter % 4),
                tw_early=float(tick),
                tw_late=float(tick + 2),
                due_tick=min(self._horizon - 1, tick + 1),
                priority_class="medical",
                criticality=0.95,
            )
            before = self._state_digest()
            self._customers[new_cid] = cust
            self._cumulative_unmet[new_cid] = 0.0
            self._append_declared_perturbation_event(
                p,
                tick,
                event_type="urgent_order",
                target={"customer_id": new_cid, **dict(p.target or {})},
                changed_state_fields=[
                    "unassigned_customers",
                    "priority_orders",
                    "sla_risk",
                ],
                materiality_metric="urgent_order_demand_units",
                materiality_value=float(cust.demand),
                materiality_threshold=1.0,
                before_digest=before,
                extra={"customer_id": new_cid, "priority": "medical"},
            )

    def _append_declared_perturbation_event(
        self,
        perturbation: Any,
        tick: int,
        *,
        event_type: str,
        target: dict[str, Any],
        changed_state_fields: list[str],
        materiality_metric: str,
        materiality_value: float,
        materiality_threshold: float,
        before_digest: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Emit the auditable runtime record for a declared disruption.

        Route perturbations used to be emitted as bare dictionaries.  That
        made a real simulator mutation indistinguishable from an unlabelled
        log message to the Protocol-2.1 canonicalizer, so the event could not
        establish a material exogenous change or a later response opportunity.
        Keep the source declaration and the observed state transition together
        at the point where the simulator applies the mutation.
        """
        before = before_digest or self._state_digest()
        after = self._state_digest()
        value = float(materiality_value)
        threshold = float(materiality_threshold)
        has_response_window = tick + 1 < self._horizon
        event_class = _ROUTE_EVENT_REGISTRY.get(event_type)
        actionable = bool(
            event_class is not None
            and not bool(perturbation.hidden)
            and has_response_window
        )
        event: dict[str, Any] = {
            "type": event_type,
            "event_id": f"{event_type}:{self._seed}:{tick}",
            "origin": "declared_perturbation",
            "declared_perturbation": True,
            "declared_event": {
                "kind": event_type,
                "trigger_tick": int(perturbation.trigger_tick),
                "duration_ticks": int(perturbation.duration_ticks),
                "target": dict(perturbation.target or {}),
                "intensity": float(perturbation.intensity),
            },
            "tick": int(tick),
            "hidden": bool(perturbation.hidden),
            "event_class": event_class or "telemetry",
            "decision_required": actionable,
            "actionable": actionable,
            "changed_state_fields": list(changed_state_fields),
            "materiality_metric": materiality_metric,
            "materiality_value": value,
            "materiality_threshold": threshold,
            "materiality_passed": bool(before != after and value >= threshold),
            "response_window_required": actionable,
            "response_opportunity_tick": tick + 1 if actionable else None,
            "terminal_response_window_missing": not has_response_window,
            "before_state_digest": before,
            "after_state_digest": after,
            "evidence_ids": [],
        }
        if extra:
            event.update(extra)
        self._realized_events_this_tick.append(event)
        self._world_evolution_records.append(dict(event))

    # ── Tool effects ────────────────────────────────────────────────────

    def apply_tool_effect(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        before = self._state_digest()
        if name == "assign_stop":
            result = self._assign_stop(args)
        elif name == "reroute_vehicle":
            result = self._reroute_vehicle(args)
        elif name == "hold_order":
            result = self._hold_order(args)
        elif name == "drop_order":
            result = self._drop_order(args)
        else:
            result = {"_status": "ack"}
        after = self._state_digest()
        if result.get("_status") != "error" and before != after:
            changed = {
                "assign_stop": ["route_assignments", "route_distance"],
                "reroute_vehicle": ["route_assignments", "route_distance"],
                "hold_order": ["unassigned_customers"],
                "drop_order": ["unassigned_customers", "route_assignments"],
            }.get(name, ["route_assignments"])
            self._unbound_action_effects.append(
                {
                    "type": f"{name}_applied",
                    "event_id": f"{name}_applied:{self._tick}:{len(self._action_effects)}",
                    "status": "passed",
                    "origin": "agent_caused",
                    "agent_caused": True,
                    "tool_name": name,
                    "requested_action": dict(args),
                    "applied_action": {
                        key: value
                        for key, value in result.items()
                        if not str(key).startswith("_")
                    },
                    "before_state_digest": before,
                    "after_state_digest": after,
                    "changed_state_fields": changed,
                    "outcome_tick": self._tick,
                    "call_id": None,
                    "evidence_ids": [],
                }
            )
        return result

    def bind_tool_result(
        self,
        *,
        name: str,
        call_id: str | None,
        evidence_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        if payload.get("_status") == "error":
            return
        effect = next(
            (
                row
                for row in self._unbound_action_effects
                if row["tool_name"] == name and row["call_id"] is None
            ),
            None,
        )
        if effect is None:
            return
        effect["call_id"] = call_id
        effect["evidence_ids"] = [evidence_id] if evidence_id else []
        effect["action_to_outcome_edge"] = {
            "source": f"call:{call_id}",
            "target": f"outcome:{effect['event_id']}",
            "kind": "action_to_outcome",
            "call_id": call_id,
        }
        self._unbound_action_effects.remove(effect)
        self._action_effects.append(effect)
        self._world_evolution_records.append(dict(effect))
        self._pending_agent_events.append(dict(effect))

    def _assign_stop(self, args: dict[str, Any]) -> dict[str, Any]:
        vid = str(args.get("vehicle_id", ""))
        cid = str(args.get("customer_id", ""))
        veh = self._vehicles.get(vid)
        if veh is None:
            return {"_status": "error", "error": "unknown_vehicle", "vehicle_id": vid}
        if not veh.active or veh.broken:
            return {
                "_status": "error",
                "error": "vehicle_unavailable",
                "vehicle_id": vid,
            }
        cust = self._customers.get(cid)
        if cust is None:
            return {"_status": "error", "error": "unknown_customer", "customer_id": cid}
        if cust.served:
            return {"_status": "error", "error": "already_served", "customer_id": cid}
        if cust.dropped:
            return {"_status": "error", "error": "already_dropped", "customer_id": cid}
        if veh.remaining_capacity + 1e-9 < cust.demand:
            return {
                "_status": "error",
                "error": "over_capacity",
                "vehicle_id": vid,
                "remaining_capacity": round(veh.remaining_capacity, 3),
                "demand": cust.demand,
            }
        # Move customer onto this vehicle's route (front).
        for ov in self._vehicles.values():
            if cid in ov.route:
                ov.route.remove(cid)
        veh.route.insert(0, cid)
        cust.assigned_vehicle = vid
        return {
            "vehicle_id": vid,
            "customer_id": cid,
            "priority_class": cust.priority_class,
            "criticality": cust.criticality,
        }

    def _reroute_vehicle(self, args: dict[str, Any]) -> dict[str, Any]:
        vid = str(args.get("vehicle_id", ""))
        seq = args.get("stop_sequence", [])
        veh = self._vehicles.get(vid)
        if veh is None:
            return {"_status": "error", "error": "unknown_vehicle", "vehicle_id": vid}
        if not isinstance(seq, list):
            return {"_status": "error", "error": "stop_sequence_not_list"}
        new_route: list[str] = []
        for cid in seq:
            cid = str(cid)
            cust = self._customers.get(cid)
            if cust is None or cust.served or cust.dropped:
                return {"_status": "error", "error": "invalid_stop", "customer_id": cid}
            new_route.append(cid)
        veh.route = new_route
        return {"vehicle_id": vid, "stop_sequence": new_route}

    def _hold_order(self, args: dict[str, Any]) -> dict[str, Any]:
        cid = str(args.get("customer_id", ""))
        until = int(args.get("until_tick", self._tick + 1))
        cust = self._customers.get(cid)
        if cust is None:
            return {"_status": "error", "error": "unknown_customer", "customer_id": cid}
        if cust.served:
            return {"_status": "error", "error": "already_served", "customer_id": cid}
        if self._has_time_windows and until > cust.due_tick:
            return {
                "_status": "error",
                "error": "past_hard_window",
                "customer_id": cid,
                "due_tick": cust.due_tick,
            }
        cust.held_until = until
        return {"customer_id": cid, "held_until": until}

    def _drop_order(self, args: dict[str, Any]) -> dict[str, Any]:
        cid = str(args.get("customer_id", ""))
        cust = self._customers.get(cid)
        if cust is None:
            return {"_status": "error", "error": "unknown_customer", "customer_id": cid}
        if cust.served:
            return {"_status": "error", "error": "already_served", "customer_id": cid}
        if cust.dropped:
            return {"_status": "error", "error": "already_dropped", "customer_id": cid}
        cust.dropped = True
        for ov in self._vehicles.values():
            if cid in ov.route:
                ov.route.remove(cid)
        penalty = _DROP_PENALTY_BASE * (1.0 + cust.criticality)
        self._drop_accum = getattr(self, "_drop_accum", 0.0) + penalty
        return {
            "customer_id": cid,
            "priority_class": cust.priority_class,
            "criticality": cust.criticality,
            "drop_penalty": round(penalty, 3),
        }

    # ── Read-only helpers (noised ETA / forecast) ───────────────────────

    def eta_for(self, vehicle_id: str) -> dict[str, Any]:
        """Noised ETA for a vehicle's next stop (documented bias/variance)."""
        veh = self._vehicles.get(vehicle_id)
        if veh is None:
            return {"_status": "error", "error": "unknown_vehicle"}
        if not veh.route:
            return {"vehicle_id": vehicle_id, "eta_ticks": None, "next_stop": None}
        cid = veh.route[0]
        cust = self._customers.get(cid)
        true_eta = 1
        if cust is not None:
            true_eta = max(1, math.ceil(_dist(veh.pos, (cust.x, cust.y)) / 20.0))
        # Deterministic +/- bias from the seed (documented; no ground truth).
        noise = (_det_hash(self._seed, self._tick, f"eta|{vehicle_id}") % 5) - 2
        return {
            "vehicle_id": vehicle_id,
            "next_stop": cid,
            "eta_ticks": max(1, true_eta + noise),
            "noised": True,
        }

    def reveal_vehicle(self, vehicle_id: str) -> None:
        """Mark a vehicle as investigated (reveals a hidden breakdown)."""
        if vehicle_id in self._vehicles:
            self._revealed_vehicles.add(vehicle_id)

    def forecast_for(self, region: str, horizon: int) -> list[dict[str, Any]]:
        """Noised demand forecast (bias schedule; no ground truth)."""
        out: list[dict[str, Any]] = []
        base = sum(c.demand for c in self._customers.values() if not c.served)
        for k in range(max(1, int(horizon))):
            t = self._tick + k
            jitter = _det_hash(self._seed, t, f"forecast|{region}|{k}") / 1000.0
            out.append(
                {
                    "tick": t,
                    "region": region,
                    "demand_forecast_units": round(base * (0.85 + 0.3 * jitter), 3),
                    "noised": True,
                }
            )
        return out

    # ── Snapshot ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        entities: dict[str, dict[str, Any]] = {}
        entities["depot"] = {
            "kind": "depot",
            "x": self._depot[0],
            "y": self._depot[1],
        }
        for vid, v in self._vehicles.items():
            # A hidden breakdown is concealed until discovered via query_eta
            # (which calls ``reveal_vehicle``); we expose the flag only when
            # visible or already revealed.
            broken_visible = v.broken and (
                not v.broken_hidden or vid in self._revealed_vehicles
            )
            entities[vid] = {
                "kind": "vehicle",
                "capacity": v.capacity,
                "remaining_capacity": round(v.remaining_capacity, 3),
                "route": list(v.route),
                "active": v.active,
                "is_standby": v.is_standby,
                "broken": broken_visible,
            }
        for cid, c in self._customers.items():
            entities[cid] = {
                "kind": "customer",
                "x": c.x,
                "y": c.y,
                "demand": c.demand,
                "priority_class": c.priority_class,
                "criticality": c.criticality,
                "served": c.served,
                "dropped": c.dropped,
                "blocked": c.blocked,
                "held_until": c.held_until,
                "due_tick": c.due_tick,
                "tw_late": c.tw_late,
            }
        last = self._tick_records[-1] if self._tick_records else None
        totals = {
            "aggregate_demand_units": last.aggregate_demand if last else 0.0,
            "served_demand_units": last.served_demand if last else 0.0,
            "unmet_demand_units": last.unmet_demand if last else 0.0,
            "routing_operating_cost": last.routing_cost if last else 0.0,
            "max_capacity_utilization": last.max_utilization if last else 0.0,
            "n_failed_routes": last.n_failed_routes if last else 0,
            "procured_standby_capacity": (last.procured_standby if last else 0.0),
        }
        return {
            "tick": self._tick,
            "horizon": self._horizon,
            "entities": entities,
            "totals": totals,
        }

    # ── Cost roll-up / scoring ──────────────────────────────────────────

    def ground_truth_costs(self) -> dict[str, float]:
        if not self._tick_records:
            return {
                "routing_operating_cost": 0.0,
                "vehicle_dispatch_fixed_cost": 0.0,
                "drop_order_penalty": 0.0,
                "unmet_demand_cost": 0.0,
            }
        routing = sum(r.routing_cost for r in self._tick_records)
        dispatch = sum(r.dispatch_fixed_cost for r in self._tick_records)
        drop = sum(r.drop_penalty for r in self._tick_records)
        unmet = sum(r.unmet_demand for r in self._tick_records) * _DROP_PENALTY_BASE
        return {
            "routing_operating_cost": round(routing, 3),
            "vehicle_dispatch_fixed_cost": round(dispatch, 3),
            "drop_order_penalty": round(drop, 3),
            "unmet_demand_cost": round(unmet, 3),
        }

    def scoring_records(self) -> list[dict[str, Any]]:
        """Per-tick rows: the canonical 14-key contract with the §7 alias
        mapping. Keys declared ``honest_zero`` for this family are forced 0.
        ``done`` is early-guarded ``bool(done and tick < horizon-1)`` so a
        normal horizon-end tick is never miscounted as a catastrophic
        collapse by ``score_system_survival``.
        """
        hz = self._honest_zero_keys
        rows: list[dict[str, Any]] = []
        for r in self._tick_records:
            entity_count = max(1, len(self._customers))
            violation_density = (
                r.n_capacity_violations + r.n_time_window_violations + r.n_failed_routes
            ) / entity_count
            utilization_excess = max(0.0, r.max_utilization - 1.0)
            row = {
                "tick": r.tick,
                "aggregate_demand_mw": r.aggregate_demand,
                "aggregate_generation_mw": r.served_demand,
                "balance_error_mw": r.unmet_demand,
                "reserves_required_mw": r.required_standby,
                "reserves_procured_mw": r.procured_standby,
                "production_cost": r.routing_cost,
                "startup_cost": r.dispatch_fixed_cost,
                "shed_penalty": r.drop_penalty,
                "rho_max": r.max_utilization,
                "n_overloads": r.n_capacity_violations,
                "n_voltage_violations": (
                    r.n_time_window_violations if self._has_time_windows else 0
                ),
                "n_disconnected_lines": r.n_failed_routes,
                "done": bool(r.done and r.tick < self._horizon - 1),
                "catastrophic_failure": bool(r.done and r.tick < self._horizon - 1),
                "safety_violation_severity": min(
                    1.0, max(utilization_excess, violation_density)
                ),
            }
            for k in hz:
                if k in row:
                    row[k] = 0
            rows.append(row)
        return rows

    def per_customer_unmet_units(self) -> dict[str, float]:
        """Logistics analogue of ``per_load_shed_mwh`` — deterministic map."""
        return {cid: round(v, 3) for cid, v in self._cumulative_unmet.items()}

    def realized_events_for_tick(self) -> list[dict[str, Any]]:
        return list(self._realized_events_this_tick)

    def protocol21_source_trace(self) -> dict[str, Any]:
        if self._source_resolution is None:
            return {
                "status": "held",
                "proof_kind": "direct_runtime_files",
                "runtime_trace_observed": False,
                "evidence_from_scenario_config_only": True,
                "blockers": ["source_instance_path_missing"],
            }
        resolution = self._source_resolution
        fields = [
            "unassigned_customers",
            "route_assignments",
            "vehicle_loads",
            "remaining_capacity",
            "route_distance",
        ]
        if resolution["instance_kind"] == "vrptw":
            fields.extend(["route_arrival_times", "time_window_feasibility"])
        semantic = {
            "source_sha256": resolution["source_sha256"],
            "parser_output_digest": resolution["parser_output_digest"],
            "initial_state_digest": self._initial_state_digest,
            "post_source_state_digests": self._post_source_state_digests,
            "action_effects": self._action_effects,
        }
        path = resolution["source_path"]
        declared_path = resolution["declared_source_path"]
        return {
            "status": "passed",
            "proof_kind": "direct_runtime_files",
            "runtime_opened_assets": [path],
            "opened_source_paths": [path],
            "opened_source_sha256": {path: resolution["source_sha256"]},
            "consumed_source_hashes": {declared_path: resolution["source_sha256"]},
            "parser_output_digest": resolution["parser_output_digest"],
            "instance_kind": resolution["instance_kind"],
            "consumed_channels": list(resolution["consumed_channels"]),
            "derived_backend_state_fields": fields,
            "consumption_ticks": [0],
            "initial_state_digest": self._initial_state_digest,
            "post_source_state_digests": list(self._post_source_state_digests),
            "source_state_effect_observed": True,
            "state_effect_observed": True,
            "deterministic_source_trace": True,
            "trace_semantic_digest": hashlib.sha256(
                json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "runtime_trace_observed": True,
            "evidence_from_scenario_config_only": False,
            "blockers": [],
        }

    def protocol21_world_evolution_records(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._world_evolution_records))

    def protocol21_agent_action_effect_records(self) -> list[dict[str, Any]]:
        return json.loads(json.dumps(self._action_effects))

    def _state_digest(self) -> str:
        state = {
            "depot": self._depot,
            "vehicles": {
                vid: {
                    "remaining_capacity": vehicle.remaining_capacity,
                    "route": vehicle.route,
                    "active": vehicle.active,
                    "broken": vehicle.broken,
                }
                for vid, vehicle in sorted(self._vehicles.items())
            },
            "customers": {
                cid: {
                    "served": customer.served,
                    "dropped": customer.dropped,
                    "held_until": customer.held_until,
                    "assigned_vehicle": customer.assigned_vehicle,
                    "blocked": customer.blocked,
                }
                for cid, customer in sorted(self._customers.items())
            },
            "traffic_multiplier": self._traffic_mult,
            "demand_surge_multiplier": self._demand_surge_mult,
            "traffic_until": self._traffic_until,
            "surge_until": self._surge_until,
            "procured_standby": self._procured_standby,
        }
        return hashlib.sha256(
            json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    # ── PyVRP fixed-plan cost eval (optional; cost-eval path only) ──────

    def evaluate_fixed_plan_cost(self) -> float:
        """Evaluate the current routes' total distance via PyVRP (MIT).

        This is the ONLY path that requires PyVRP; the simulator never needs
        it to step. Raises the typed ``LogisticsBackendUnavailable`` when
        PyVRP is absent so callers / tests can skip cleanly.
        """
        if not PYVRP_AVAILABLE:
            raise LogisticsBackendUnavailable(
                "PyVRP is required for fixed-plan route-cost evaluation but is "
                f"not importable: {_PYVRP_IMPORT_ERROR or 'module missing'}. "
                "Install via `pip install pyvrp`. The deterministic simulator "
                "does NOT require PyVRP to run."
            )
        m = _PyVRPModel()  # type: ignore[operator]
        depot = m.add_depot(x=int(round(self._depot[0])), y=int(round(self._depot[1])))
        n_active = sum(1 for v in self._vehicles.values() if v.active)
        m.add_vehicle_type(
            max(1, n_active), capacity=int(max(1, round(self._capacity)))
        )
        clients = {}
        for cid, c in self._customers.items():
            if c.dropped:
                continue
            clients[cid] = m.add_client(
                x=int(round(c.x)),
                y=int(round(c.y)),
                delivery=int(max(0, round(c.demand))),
            )
        locs = [depot] + list(clients.values())
        for a in locs:
            for b in locs:
                if a is b:
                    continue
                m.add_edge(a, b, distance=int(round(_dist((a.x, a.y), (b.x, b.y)))))
        res = m.solve(stop=_pyvrp_stop(), display=False)
        return float(res.cost())


def _pyvrp_stop():  # pragma: no cover - only when pyvrp present
    """Deterministic PyVRP stopping rule (max-iteration count, no wall clock)."""
    from pyvrp.stop import MaxIterations  # type: ignore[import]

    return MaxIterations(50)
