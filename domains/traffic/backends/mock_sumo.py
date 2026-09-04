"""
domains.traffic.backends.mock_sumo — Pure-Python deterministic SUMO mock.

Mirrors the method surface of
``domains.disaster.backends.mock_rcrs.MockRcrsBackend`` /
``domains.power_grid.backends.pglib_uc_synthetic`` (reset / tick / snapshot /
apply_tool_effect / ground_truth_costs / scoring_records / forecast_for /
queue_mutual_aid_effect / per_corridor_delay_minutes) so the traffic adapter is
structurally identical to the other domains' adapters (red-line #3: matching
*shape*, not names). No SUMO / traci / libsumo import lives here — the real
sidecar-driven backend is a separate class.

Determinism (spec §9): all randomness flows through ``_det_hash(seed, tick,
key) → [0,1000)`` (pure SHA-256). Two ``reset`` + ``tick`` loops with identical
inputs produce byte-identical per-corridor delay-minute maps — the contract the
adaptive-replanning + counterfactual-replay scorers rely on.

Mock physics (deliberate simplifications — documented to stay honest; a SUMO
host re-derives these from ``env.step()`` micro-simulation at stage 4):

1. Each corridor has an offered-demand profile (triangular peak shifted by the
   seed's ``demand_window_offset_min``), a per-tick service capacity, a
   carried-over ``queue`` (spillback), and a cumulative ``delay_minutes``
   accumulator. Vehicles that cannot be served this tick wait and accrue delay.
2. ``capacity`` is modulated multiplicatively by: ambient weather factor,
   active incident (lane blockage), operator ``close_lane`` actions, the active
   signal program (``change_signal_plan``), and granted priority.
3. ``meter_inflow`` defers a fraction of a corridor's offered demand upstream —
   reducing downstream spillback at the cost of *metered* delay booked to
   ``shed_penalty`` (the VOT-weighted delay the operator deliberately imposes).
4. ``reroute_flow`` diverts a fraction of one corridor's offered demand to a
   sibling corridor (deterministic target) — mitigating an incident corridor.
5. ``vip_arrival`` / ``ems_corridor_request`` arrive as perturbations and are
   surfaced via ``realized_events``; ``detector_dropout`` is hidden by the
   adapter's fog layer (we only surface the structured event here).
6. Delayed effect: ``queue_mutual_aid_effect(due_tick, mw)`` releases ``mw``
   network-wide capacity-relief units (e.g., signal-retiming crews) that
   materialize at ``due_tick`` exactly (F-01 unified-delay contract; the arg
   name ``mw`` is preserved so all backends share one cross-domain test).

Cost model (cost-units, calibrated onto the same scale as the grid/disaster
proxies so the cross-domain dashboard stays comparable):

- ``travel_time_cost`` = sum over ticks of queued-veh × tick_minutes × VOT.
- ``shed_delay_cost``  = sum over ticks of metered-veh × tick_minutes × VOT.
- ``actuation_cost``   = signal re-program / reroute / priority lost-time.
- ``mutual_aid_cost``  = relief units × per-unit cost.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..seeds.schema import TrafficScenarioSeed


# ── Calibration constants (small, contract-shaped — v0.8 swaps in PeMS/VOT) ──
_CAP_SLACK: float = 1.15  # base capacity headroom vs mean demand (no-shock clears)
_VOT_PER_VEH_MIN: float = 0.30  # value-of-time per vehicle-minute
_CLOSED_LANE_CAP_MULT: float = 0.80  # each operator-closed lane cuts capacity
_INCIDENT_CAP_FLOOR: float = 0.35  # capacity multiplier at full incident intensity
_PRIORITY_CAP_BONUS: float = 1.25  # granted EMS/VIP green-wave throughput bump
_GRIDLOCK_QUEUE_MULT: float = 4.0  # queue > mult × base_cap ⇒ junction gridlock
_ACTUATION_LOST_TIME_COST: float = 40.0  # per signal-replan / reroute / priority
_MUTUAL_AID_COST_PER_UNIT: float = 60.0
_SURGE_FAMILY_FACTOR_KEY: str = "surge_factor"

# Signal-program capacity multipliers (spec §6 native-tool vocabulary).
_PROGRAM_BONUS: dict[str, float] = {
    "default": 1.00,
    "incident_relief": 1.30,
    "ems_priority": 1.25,
    "vip_greenwave": 1.20,
    "fail_safe": 0.60,  # signal_failure fallback
    "peak_coordination": 1.10,
}


def _det_hash(seed: int, tick: int, key: str) -> int:
    """Deterministic integer in ``[0, 1000)`` from ``(seed, tick, key)``."""
    body = f"{int(seed)}|{int(tick)}|{key}".encode()
    digest = hashlib.sha256(body).digest()
    return int.from_bytes(digest[:4], "big") % 1000


@dataclass
class _CorridorState:
    corridor_id: str
    district: str
    demand_veh: int
    base_cap_per_tick: float
    criticality: float
    income_bracket: str = "mid"
    transit_dependent_fraction: float = 0.2
    carries_ems_corridor: bool = False
    carries_vip_route: bool = False
    # live control / shock state
    queue: float = 0.0
    delay_minutes: float = 0.0
    capacity_factor: float = 1.0  # ambient weather
    incident_active: bool = False
    incident_intensity: float = 0.0
    surge_factor: float = 1.0
    closed_lanes: int = 0
    signal_program: str = "default"
    metered_fraction: float = 0.0
    reroute_out_fraction: float = 0.0
    reroute_target: str | None = None
    priority_granted: str | None = None  # "ems" | "vip" | None
    detector_dark_until: int = -1


@dataclass
class _TrafficTickRecord:
    tick: int
    aggregate_offered: float = 0.0
    aggregate_served: float = 0.0
    aggregate_queue: float = 0.0
    aggregate_delay_minutes: float = 0.0
    travel_cost_this_tick: float = 0.0
    shed_cost_this_tick: float = 0.0
    actuation_cost_this_tick: float = 0.0
    mutual_aid_cost_this_tick: float = 0.0
    reserves_required: float = 0.0
    reserves_procured: float = 0.0
    rho_max: float = 0.0
    n_overloads: int = 0
    n_gridlocked: int = 0
    n_blocked_edges: int = 0
    realized_events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False


class MockSumoBackend:
    """Pure-Python deterministic SUMO substitute (default for stages 1–3)."""

    def __init__(self) -> None:
        self._seed_obj: TrafficScenarioSeed | None = None
        self._tick: int = 0
        self._horizon: int = 24
        self._corridors: dict[str, _CorridorState] = {}
        self._tick_records: list[_TrafficTickRecord] = []
        self._mutual_aid_queue: list[tuple[int, float]] = []
        self._mutual_aid_units_total: float = 0.0
        self._actuation_events_pending: int = 0
        self._edge_to_corridor: dict[str, str] = {}
        self._peak_tick: int = 0
        self._realized_events_this_tick: list[dict[str, Any]] = []

    # ── Reset ───────────────────────────────────────────────────────────────

    def reset(self, scenario_seed: TrafficScenarioSeed) -> None:
        self._seed_obj = scenario_seed
        self._tick = 0
        self._horizon = int(scenario_seed.horizon_ticks)
        self._tick_records.clear()
        self._mutual_aid_queue.clear()
        self._mutual_aid_units_total = 0.0
        self._actuation_events_pending = 0
        self._realized_events_this_tick = []

        horizon = max(1, self._horizon)
        self._corridors = {
            c.corridor_id: _CorridorState(
                corridor_id=c.corridor_id,
                district=c.district,
                demand_veh=int(c.demand_veh),
                base_cap_per_tick=(int(c.demand_veh) / horizon) * _CAP_SLACK,
                criticality=float(c.criticality),
                income_bracket=str(c.income_bracket),
                transit_dependent_fraction=float(c.transit_dependent_fraction),
                carries_ems_corridor=bool(c.carries_ems_corridor),
                carries_vip_route=bool(c.carries_vip_route),
            )
            for c in scenario_seed.corridors
        }

        # Deterministic edge→corridor map so incidents named only by SUMO edge
        # id land on a stable corridor (operator reveals it via inspect tools).
        cids = sorted(self._corridors.keys())
        self._edge_to_corridor = {}
        if cids and scenario_seed.incident_edge:
            idx = _det_hash(scenario_seed.seed, 0, scenario_seed.incident_edge) % len(
                cids
            )
            self._edge_to_corridor[scenario_seed.incident_edge] = cids[idx]
        for p in scenario_seed.perturbations:
            edge = str(p.target.get("edge", "")) if isinstance(p.target, dict) else ""
            if edge and edge not in self._edge_to_corridor and cids:
                idx = _det_hash(scenario_seed.seed, 0, edge) % len(cids)
                self._edge_to_corridor[edge] = cids[idx]

        # Triangular demand peak shifted by the seed's window offset.
        offset_ticks = int(scenario_seed.demand_window_offset_min // 5)
        self._peak_tick = min(horizon - 1, max(0, int(horizon * 0.4) + offset_ticks))

    # ── Demand profile ───────────────────────────────────────────────────────

    def _demand_weight(self, tick: int) -> float:
        """Triangular weight over the horizon, normalized so weights sum≈1."""
        horizon = max(1, self._horizon)
        peak = self._peak_tick
        # Tent: rises to 1.0 at peak, falls to a 0.2 floor at the edges.
        if tick <= peak:
            rise = (tick + 1) / (peak + 1)
            w = 0.2 + 0.8 * rise
        else:
            span = max(1, horizon - 1 - peak)
            fall = (horizon - 1 - tick) / span
            w = 0.2 + 0.8 * max(0.0, fall)
        # Normalize by the analytic sum of the tent across the horizon.
        total = 0.0
        for t in range(horizon):
            if t <= peak:
                total += 0.2 + 0.8 * ((t + 1) / (peak + 1))
            else:
                span = max(1, horizon - 1 - peak)
                total += 0.2 + 0.8 * max(0.0, (horizon - 1 - t) / span)
        return w / max(total, 1e-9)

    # ── Tick advance ───────────────────────────────────────────────────────

    def tick(self, current_tick: int) -> _TrafficTickRecord:
        assert self._seed_obj is not None
        self._tick = int(current_tick)
        tick_minutes = int(self._seed_obj.tick_minutes or 5)

        # 1) Apply perturbations firing at this tick.
        self._apply_perturbations_at_tick(self._tick)

        # 2) Drain matured capacity-relief arrivals (F-01).
        mutual_aid_before = self._mutual_aid_units_total
        added_units = self._drain_mutual_aid(self._tick)
        if added_units > 0.0:
            self._mutual_aid_units_total += added_units
            cursor = mutual_aid_before
            legacy_units = 0.0
            for units, tool_call in self._matured_mutual_aid_calls:
                before = cursor
                cursor += units
                call_id = str(tool_call.get("call_id") or "")
                if not call_id:
                    legacy_units += units
                    continue
                tool_name = str(
                    tool_call.get("tool_name") or "request_incident_response_team"
                )
                self._realized_events_this_tick.append(
                    {
                        "type": "relief_crew_arrived",
                        "event_id": f"relief_crew_arrived:{call_id}:{self._tick}",
                        "origin": "agent_caused",
                        "agent_caused": True,
                        "event_class": "agent_outcome",
                        "decision_required": False,
                        "actionable": False,
                        "units": round(units, 3),
                        "tick": self._tick,
                        "call_id": call_id,
                        "tool_name": tool_name,
                        "requested_action": {
                            "name": tool_name,
                            "args": dict(tool_call.get("args") or {}),
                        },
                        "before_state_digest": hashlib.sha256(
                            f"mutual_aid_units_total:{before:.12g}".encode()
                        ).hexdigest(),
                        "after_state_digest": hashlib.sha256(
                            f"mutual_aid_units_total:{cursor:.12g}".encode()
                        ).hexdigest(),
                        "effect_tick": self._tick,
                        "outcome_tick": self._tick,
                    }
                )
            if legacy_units > 0.0:
                self._realized_events_this_tick.append(
                    {
                        "type": "relief_crew_arrived",
                        "units": round(legacy_units, 3),
                        "tick": self._tick,
                    }
                )
        # Relief crews lift capacity network-wide for this + later ticks.
        relief_bonus = 1.0 + min(0.5, 0.05 * self._mutual_aid_units_total)

        # 3) Compute reroute diversions (offered moved between corridors).
        diverted_in: dict[str, float] = {cid: 0.0 for cid in self._corridors}
        offered_raw: dict[str, float] = {}
        weight = self._demand_weight(self._tick)
        for cid, c in self._corridors.items():
            offered_raw[cid] = c.demand_veh * weight * c.surge_factor
        for cid, c in self._corridors.items():
            if c.reroute_out_fraction > 0.0 and c.reroute_target in self._corridors:
                moved = offered_raw[cid] * c.reroute_out_fraction
                offered_raw[cid] -= moved
                diverted_in[c.reroute_target] += moved

        # 4) Per-corridor service + queue + delay.
        agg_offered = agg_served = agg_queue = 0.0
        travel_cost = shed_cost = 0.0
        rho_max = 0.0
        n_overloads = n_gridlocked = 0
        for cid, c in self._corridors.items():
            metered = offered_raw[cid] * c.metered_fraction
            offered_eff = max(0.0, offered_raw[cid] - metered + diverted_in[cid])

            cap = c.base_cap_per_tick * c.capacity_factor * relief_bonus
            cap *= _PROGRAM_BONUS.get(c.signal_program, 1.0)
            cap *= _CLOSED_LANE_CAP_MULT**c.closed_lanes
            if c.incident_active:
                cap *= _INCIDENT_CAP_FLOOR + (1.0 - _INCIDENT_CAP_FLOOR) * (
                    1.0 - c.incident_intensity
                )
            if c.priority_granted:
                cap *= _PRIORITY_CAP_BONUS
            cap = max(cap, 1e-6)

            inflow = offered_eff + c.queue
            served = min(inflow, cap)
            queue_out = max(0.0, inflow - served)
            c.queue = queue_out

            tick_delay = queue_out * tick_minutes
            c.delay_minutes += tick_delay
            travel_cost += tick_delay * _VOT_PER_VEH_MIN
            shed_cost += metered * tick_minutes * _VOT_PER_VEH_MIN

            rho = inflow / cap
            rho_max = max(rho_max, rho)
            if rho > 1.0:
                n_overloads += 1
            if queue_out > c.base_cap_per_tick * _GRIDLOCK_QUEUE_MULT:
                n_gridlocked += 1

            agg_offered += offered_eff
            agg_served += served
            agg_queue += queue_out

        # 5) Actuation lost-time booked when control actions fired this tick.
        actuation_cost = self._actuation_events_pending * _ACTUATION_LOST_TIME_COST
        self._actuation_events_pending = 0
        mutual_aid_cost_this_tick = added_units * _MUTUAL_AID_COST_PER_UNIT

        n_blocked = sum(1 for c in self._corridors.values() if c.incident_active)
        n_blocked += sum(c.closed_lanes for c in self._corridors.values())

        reserves_required = sum(
            0.1 * c.base_cap_per_tick for c in self._corridors.values()
        )
        reserves_procured = max(0.0, agg_served - agg_offered) + max(
            0.0,
            sum(c.base_cap_per_tick for c in self._corridors.values()) - agg_offered,
        )

        all_gridlocked = len(self._corridors) > 0 and n_gridlocked == len(
            self._corridors
        )

        record = _TrafficTickRecord(
            tick=self._tick,
            aggregate_offered=round(agg_offered, 3),
            aggregate_served=round(agg_served, 3),
            aggregate_queue=round(agg_queue, 3),
            aggregate_delay_minutes=round(
                sum(c.delay_minutes for c in self._corridors.values()), 2
            ),
            travel_cost_this_tick=round(travel_cost, 2),
            shed_cost_this_tick=round(shed_cost, 2),
            actuation_cost_this_tick=round(actuation_cost, 2),
            mutual_aid_cost_this_tick=round(mutual_aid_cost_this_tick, 2),
            reserves_required=round(reserves_required, 3),
            reserves_procured=round(reserves_procured, 3),
            rho_max=round(rho_max, 4),
            n_overloads=int(n_overloads),
            n_gridlocked=int(n_gridlocked),
            n_blocked_edges=int(n_blocked),
            realized_events=list(self._realized_events_this_tick),
            done=bool(all_gridlocked),
        )
        self._realized_events_this_tick = []
        self._tick_records.append(record)
        return record

    # ── Perturbations ────────────────────────────────────────────────────────

    def _apply_perturbations_at_tick(self, tick: int) -> None:
        assert self._seed_obj is not None
        events: list[dict[str, Any]] = []
        for p in self._seed_obj.perturbations:
            window = p.trigger_tick <= tick < p.trigger_tick + p.duration_ticks
            applied = self._apply_one_perturbation(p, tick, active=window)
            if applied is not None:
                events.append(applied)
        self._realized_events_this_tick = events

    def _target_corridor(self, p: Any) -> _CorridorState | None:
        tgt = p.target if isinstance(p.target, dict) else {}
        cid = str(tgt.get("corridor", ""))
        if cid and cid in self._corridors:
            return self._corridors[cid]
        edge = str(tgt.get("edge", ""))
        mapped = self._edge_to_corridor.get(edge)
        if mapped:
            return self._corridors.get(mapped)
        return None

    def _apply_one_perturbation(
        self, p: Any, tick: int, *, active: bool
    ) -> dict[str, Any] | None:
        kind = str(p.kind)
        c = self._target_corridor(p)

        if kind == "weather_capacity_drop":
            factor = float(p.target.get("capacity_factor", 0.7))
            for cc in self._corridors.values():
                cc.capacity_factor = factor if active else 1.0
            if tick == p.trigger_tick:
                return {
                    "type": "weather_capacity_drop",
                    "tick": tick,
                    "capacity_factor": factor,
                    "hidden": bool(p.hidden),
                }
            return None

        if kind in ("incident", "lane_blockage"):
            if c is not None:
                c.incident_active = active
                c.incident_intensity = float(p.intensity) if active else 0.0
            if tick == p.trigger_tick:
                return {
                    "type": kind,
                    "tick": tick,
                    "corridor": c.corridor_id if c else None,
                    "edge": str(p.target.get("edge", "")),
                    "hidden": bool(p.hidden),
                }
            return None

        if kind == "signal_failure":
            if c is not None:
                if active:
                    c.signal_program = "fail_safe"
                elif c.signal_program == "fail_safe":
                    c.signal_program = "default"
            if tick == p.trigger_tick:
                return {
                    "type": "signal_failure",
                    "tick": tick,
                    "corridor": c.corridor_id if c else None,
                    "hidden": bool(p.hidden),
                }
            return None

        if kind == "demand_surge":
            factor = float(p.target.get(_SURGE_FAMILY_FACTOR_KEY, 1.5))
            if c is not None:
                c.surge_factor = factor if active else 1.0
            if tick == p.trigger_tick:
                return {
                    "type": "demand_surge",
                    "tick": tick,
                    "corridor": c.corridor_id if c else None,
                    "surge_factor": factor,
                    "hidden": bool(p.hidden),
                }
            return None

        if kind == "detector_dropout":
            if c is not None and active:
                c.detector_dark_until = p.trigger_tick + p.duration_ticks
            if tick == p.trigger_tick:
                return {
                    "type": "detector_dropout",
                    "tick": tick,
                    "corridor": c.corridor_id if c else None,
                    "hidden": bool(p.hidden),
                }
            return None

        if kind in ("vip_arrival", "ems_corridor_request"):
            if tick == p.trigger_tick:
                return {
                    "type": kind,
                    "tick": tick,
                    "corridor": str(p.target.get("corridor", "")),
                    "hidden": bool(p.hidden),
                }
            return None

        return None

    # ── F-01 delayed capacity relief ──────────────────────────────────────────

    def queue_mutual_aid_effect(
        self,
        *,
        due_tick: int,
        mw: float,
        tool_call: dict[str, Any] | None = None,
    ) -> None:
        """F-01 unified-delay contract — ``mw`` reused as relief crew units."""
        self._mutual_aid_queue.append(
            (int(due_tick), float(mw), dict(tool_call or {}))
        )

    def _drain_mutual_aid(self, current_tick: int) -> float:
        added = 0.0
        kept: list[tuple[int, float, dict[str, Any]]] = []
        matured: list[tuple[float, dict[str, Any]]] = []
        for queued in self._mutual_aid_queue:
            due_tick, units = queued[:2]
            tool_call = dict(queued[2]) if len(queued) > 2 else {}
            if due_tick <= current_tick:
                added += units
                matured.append((units, tool_call))
            else:
                kept.append((due_tick, units, tool_call))
        self._mutual_aid_queue = kept
        self._matured_mutual_aid_calls = matured
        return added

    # ── Tool effects ──────────────────────────────────────────────────────────

    def apply_tool_effect(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "change_signal_plan":
            return self._change_signal_plan(args)
        if name == "reroute_flow":
            return self._reroute_flow(args)
        if name == "close_lane":
            return self._close_lane(args)
        if name == "meter_inflow":
            return self._meter_inflow(args)
        if name == "dispatch_emergency_priority":
            return self._dispatch_priority(args)
        if name in ("query_network_state", "query_detector", "inspect_intersection"):
            return self._inspect(args)
        return {"_status": "ack"}

    def _corridor_arg(self, args: dict[str, Any]) -> _CorridorState | None:
        cid = str(args.get("corridor") or args.get("corridor_id") or "")
        if cid in self._corridors:
            return self._corridors[cid]
        edge = str(args.get("edge", ""))
        mapped = self._edge_to_corridor.get(edge)
        return self._corridors.get(mapped) if mapped else None

    def _change_signal_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        c = self._corridor_arg(args)
        if c is None:
            return {"_status": "error", "error": "unknown_corridor"}
        program = str(args.get("program") or args.get("program_id") or "default")
        if program not in _PROGRAM_BONUS:
            return {"_status": "error", "error": "unknown_program", "program": program}
        c.signal_program = program
        self._actuation_events_pending += 1
        return {
            "corridor": c.corridor_id,
            "program": program,
            "capacity_multiplier": _PROGRAM_BONUS[program],
            "stakeholder_class": _stakeholder_for_corridor(c),
        }

    def _reroute_flow(self, args: dict[str, Any]) -> dict[str, Any]:
        c = self._corridor_arg(args)
        if c is None:
            return {"_status": "error", "error": "unknown_corridor"}
        fraction = float(args.get("fraction", 0.2) or 0.2)
        fraction = max(0.0, min(1.0, fraction))
        target = str(args.get("to_corridor") or args.get("target") or "")
        if target not in self._corridors:
            siblings = [k for k in sorted(self._corridors) if k != c.corridor_id]
            target = siblings[0] if siblings else c.corridor_id
        c.reroute_out_fraction = fraction
        c.reroute_target = target
        self._actuation_events_pending += 1
        return {
            "corridor": c.corridor_id,
            "to_corridor": target,
            "fraction": fraction,
            "stakeholder_class": _stakeholder_for_corridor(c),
        }

    def _close_lane(self, args: dict[str, Any]) -> dict[str, Any]:
        c = self._corridor_arg(args)
        if c is None:
            return {"_status": "error", "error": "unknown_corridor"}
        n = int(args.get("n_lanes", 1) or 1)
        if n <= 0:
            return {"_status": "error", "error": "non_positive_n_lanes", "n_lanes": n}
        c.closed_lanes += n
        # Physically removing the blocked lane clears incident churn on it.
        c.incident_active = False
        c.incident_intensity = 0.0
        return {
            "corridor": c.corridor_id,
            "closed_lanes": c.closed_lanes,
            "stakeholder_class": _stakeholder_for_corridor(c),
        }

    def _meter_inflow(self, args: dict[str, Any]) -> dict[str, Any]:
        c = self._corridor_arg(args)
        if c is None:
            return {"_status": "error", "error": "unknown_corridor"}
        rate = float(args.get("meter_fraction", args.get("rate", 0.3)) or 0.3)
        rate = max(0.0, min(1.0, rate))
        c.metered_fraction = rate
        return {
            "corridor": c.corridor_id,
            "meter_fraction": rate,
            "stakeholder_class": _stakeholder_for_corridor(c),
        }

    def _dispatch_priority(self, args: dict[str, Any]) -> dict[str, Any]:
        c = self._corridor_arg(args)
        if c is None:
            return {"_status": "error", "error": "unknown_corridor"}
        mode = str(args.get("mode", "ems")).lower()
        if mode not in ("ems", "vip"):
            return {"_status": "error", "error": "unknown_priority_mode", "mode": mode}
        c.priority_granted = mode
        c.signal_program = "ems_priority" if mode == "ems" else "vip_greenwave"
        self._actuation_events_pending += 1
        return {
            "corridor": c.corridor_id,
            "mode": mode,
            "fatal_class": bool(c.carries_ems_corridor and mode == "vip"),
            "stakeholder_class": _stakeholder_for_corridor(c),
        }

    def _inspect(self, args: dict[str, Any]) -> dict[str, Any]:
        c = self._corridor_arg(args)
        if c is None:
            return {"_status": "error", "error": "unknown_corridor"}
        return {
            "corridor": c.corridor_id,
            "district": c.district,
            "queue": round(c.queue, 2),
            "delay_minutes": round(c.delay_minutes, 2),
            "incident_active": c.incident_active,
            "signal_program": c.signal_program,
            "closed_lanes": c.closed_lanes,
            "metered_fraction": c.metered_fraction,
            "criticality": c.criticality,
            "detector_dark": (
                c.detector_dark_until >= 0 and self._tick < c.detector_dark_until
            ),
            "stakeholder_class": _stakeholder_for_corridor(c),
        }

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        corridors = {
            cid: {
                "kind": "corridor",
                "district": c.district,
                "demand_veh": c.demand_veh,
                "queue": round(c.queue, 2),
                "delay_minutes": round(c.delay_minutes, 2),
                "capacity_factor": round(c.capacity_factor, 3),
                "incident_active": c.incident_active,
                "signal_program": c.signal_program,
                "closed_lanes": c.closed_lanes,
                "metered_fraction": round(c.metered_fraction, 3),
                "priority_granted": c.priority_granted,
                "criticality": c.criticality,
                "carries_ems_corridor": c.carries_ems_corridor,
                "carries_vip_route": c.carries_vip_route,
                "detector_dark_active": (
                    c.detector_dark_until >= 0 and self._tick < c.detector_dark_until
                ),
            }
            for cid, c in self._corridors.items()
        }
        last = self._tick_records[-1] if self._tick_records else None
        return {
            "entities": corridors,
            "totals": {
                "aggregate_offered": (last.aggregate_offered if last else 0.0),
                "aggregate_served": (last.aggregate_served if last else 0.0),
                "aggregate_queue": (last.aggregate_queue if last else 0.0),
                "aggregate_delay_minutes": (
                    last.aggregate_delay_minutes if last else 0.0
                ),
                "mutual_aid_units_total": round(self._mutual_aid_units_total, 3),
            },
            "tick": self._tick,
            "horizon": self._horizon,
        }

    # ── Forecasts ─────────────────────────────────────────────────────────────

    def forecast_for(self, horizon: int) -> list[dict[str, Any]]:
        """Noised per-tick demand forecast (deterministic in ``(seed, tick, k)``)."""
        if self._seed_obj is None:
            return []
        out: list[dict[str, Any]] = []
        for k in range(int(horizon)):
            absolute_tick = self._tick + k
            base_weight = self._demand_weight(min(absolute_tick, self._horizon - 1))
            total_demand = sum(c.demand_veh for c in self._corridors.values())
            jitter = (
                _det_hash(self._seed_obj.seed, self._tick, f"forecast|{k}") / 1000.0
            )
            out.append(
                {
                    "tick": absolute_tick,
                    "predicted_offered_veh": round(
                        total_demand * base_weight * (0.9 + 0.2 * jitter), 2
                    ),
                    "forecast_index": k,
                }
            )
        return out

    # ── Cost roll-up ──────────────────────────────────────────────────────────

    def ground_truth_costs(self) -> dict[str, float]:
        if not self._tick_records:
            return {
                "travel_time_cost": 0.0,
                "shed_delay_cost": 0.0,
                "actuation_cost": 0.0,
                "mutual_aid_cost": 0.0,
            }
        travel = sum(r.travel_cost_this_tick for r in self._tick_records)
        shed = sum(r.shed_cost_this_tick for r in self._tick_records)
        actuation = sum(r.actuation_cost_this_tick for r in self._tick_records)
        mutual_aid = self._mutual_aid_units_total * _MUTUAL_AID_COST_PER_UNIT
        return {
            "travel_time_cost": round(travel, 2),
            "shed_delay_cost": round(shed, 2),
            "actuation_cost": round(actuation, 2),
            "mutual_aid_cost": round(mutual_aid, 2),
        }

    def native_scoring_records(self) -> list[dict[str, Any]]:
        return [
            {
                "tick": r.tick,
                "aggregate_offered": r.aggregate_offered,
                "aggregate_served": r.aggregate_served,
                "aggregate_queue": r.aggregate_queue,
                "aggregate_delay_minutes": r.aggregate_delay_minutes,
                "travel_cost_this_tick": r.travel_cost_this_tick,
                "shed_cost_this_tick": r.shed_cost_this_tick,
            }
            for r in self._tick_records
        ]

    def scoring_records(self) -> list[dict[str, Any]]:
        """Per-tick rows in the canonical 14-key scorer contract (spec §7).

        Traffic→canonical proxy mapping:

        - ``aggregate_demand_mw``     ← offered demand (veh)
        - ``aggregate_generation_mw`` ← served throughput (veh)
        - ``balance_error_mw``        ← spillback / unmet demand (veh)
        - ``reserves_required_mw``    ← target spare capacity
        - ``reserves_procured_mw``    ← realized spare capacity
        - ``production_cost``         ← travel-time cost (delay × VOT)
        - ``startup_cost``            ← signal-replan / reroute actuation lost-time
        - ``shed_penalty``            ← VOT delay imposed by ``meter_inflow``
        - ``rho_max`` / ``n_overloads`` ← max v/c + oversaturated-edge count
        - ``n_voltage_violations``    ← junction gridlock count (alias)
        - ``n_disconnected_lines``    ← blocked/closed edge count
        - ``done`` is early-guarded ``bool(done and tick < horizon-1)`` so a
          normal horizon-end tick is never miscounted as a catastrophic collapse.
        """
        rows: list[dict[str, Any]] = []
        for r in self._tick_records:
            corridor_count = max(1, len(self._corridors))
            queue_fraction = r.aggregate_queue / max(1.0, r.aggregate_offered)
            network_violation_fraction = (
                r.n_overloads + r.n_gridlocked + r.n_blocked_edges
            ) / (3.0 * corridor_count)
            rows.append(
                {
                    "tick": r.tick,
                    "aggregate_demand_mw": float(r.aggregate_offered),
                    "aggregate_generation_mw": float(r.aggregate_served),
                    "balance_error_mw": float(r.aggregate_queue),
                    "reserves_required_mw": float(r.reserves_required),
                    "reserves_procured_mw": float(r.reserves_procured),
                    "production_cost": float(r.travel_cost_this_tick),
                    "startup_cost": float(r.actuation_cost_this_tick),
                    "shed_penalty": float(r.shed_cost_this_tick),
                    "rho_max": float(r.rho_max),
                    "n_overloads": int(r.n_overloads),
                    "n_voltage_violations": int(r.n_gridlocked),
                    "n_disconnected_lines": int(r.n_blocked_edges),
                    "done": bool(r.done and r.tick < self._horizon - 1),
                    "catastrophic_failure": bool(
                        r.done and r.tick < self._horizon - 1
                    ),
                    "safety_violation_severity": min(
                        1.0,
                        max(
                            0.0,
                            queue_fraction,
                            network_violation_fraction,
                            r.rho_max - 1.0,
                        ),
                    ),
                }
            )
        return rows

    def per_corridor_delay_minutes(self) -> dict[str, float]:
        """Traffic analogue of ``per_zone_unserved_minutes`` / ``per_load_shed``.

        Deterministic — two ``reset`` + ``tick`` loops with the same seed return
        identical maps (counterfactual-replay + adaptive-replanning contract).
        """
        return {cid: round(c.delay_minutes, 3) for cid, c in self._corridors.items()}


def _stakeholder_for_corridor(c: _CorridorState) -> str:
    """Map a corridor to its canonical traffic stakeholder class.

    ``emergency_services`` if the corridor carries an EMS route (highest
    criticality); freight for the industrial/port district; transit for
    high transit-dependence; otherwise commuter.
    """
    if c.carries_ems_corridor:
        return "emergency_services"
    if c.district == "port":
        return "freight_operator"
    if c.transit_dependent_fraction >= 0.4:
        return "transit_agency"
    return "commuter"
