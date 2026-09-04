"""
domains.disaster.backends.mock_rcrs — Pure-Python deterministic RCRS mock.

Mirrors the method surface of
``domains.power_grid.backends.pglib_uc_synthetic.PglibUcSyntheticBackend``
(reset / tick / snapshot / apply_tool_effect / ground_truth_costs /
scoring_records / forecast_for / queue_mutual_aid_effect) so the disaster
adapter is structurally identical to the power-grid adapter (Red Line #3:
matching shape, not names).

Determinism design (per the v0.3 disaster design doc §5.5):

- All randomness flows through ``_det_hash(seed, tick, key) → [0, 1000)``
  — never ``random.random()`` and never wall-clock state. Two ``reset``
  + ``tick`` loops with identical inputs produce byte-identical per-zone
  unserved-minutes maps. This is the contract the adaptive-replanning and
  counterfactual-replay scorers rely on.
- Hazard time series, aftershock schedule, building-collapse triggers,
  fire spread, medical surge thresholds — all derived from the seed's
  ``backend_config`` block plus the seed's perturbation list.

Mock physics (deliberate simplifications — documented to stay honest):

1. Each zone has a ``population`` (initial), a ``buried`` counter, a
   ``fire_intensity`` in ``[0, 1]``, a per-zone ``unserved_minutes``
   accumulator, and lists of dispatched ``ambulance_teams``,
   ``fire_brigade_teams``, ``police_units``.
2. ``hazard_shake`` injects buried civilians proportional to population ×
   intensity. ``aftershock`` and hidden ``building_collapse`` add more.
3. Ambulance teams stationed in a zone reduce ``buried`` proportional to
   team count each tick (rescue throughput). Buried civilians count
   ``tick_minutes`` against ``unserved_minutes`` every tick they remain
   unrescued.
4. Fire brigade teams suppress ``fire_intensity`` toward 0. Without
   suppression, fire spreads to an adjacent zone (deterministic adjacency
   graph keyed on zone_id pairs) once per tick.
5. ``medical_surge`` and ``comms_blackout`` arrive as perturbations and
   are surfaced via ``realized_events``. ``comms_blackout`` is hidden by
   the adapter's fog layer (we only surface the structured event here).
6. Mutual aid: ``queue_mutual_aid_effect(due_tick, mw)`` adds team-count
   equivalents that materialize at ``due_tick`` exactly (F-01 contract).
   The mock interprets ``mw`` as ``team_units × 1.0`` so the disaster
   side reuses the unified delayed-effect plumbing.

Cost model (USD-like proxy units, calibrated to fit on the same scale as
power-grid costs so the cross-domain dashboard makes sense):

- ``response_cost``           = sum over ticks of dispatched_team_cost.
- ``unserved_population_cost``= unserved_minutes × per_minute_VoL.
- ``secondary_casualty_cost`` = casualties from un-suppressed fire spread.
- ``mutual_aid_cost``         = mutual_aid_units × per_unit_aid_cost.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..seeds.schema import DisasterScenarioSeed


# Per-tick cost calibration (deliberately small constants — the spike is
# about contract shape, not absolute economic realism; v0.4 swaps in
# HAZUS / EM-DAT-derived tariffs).
_PER_TEAM_DISPATCH_COST_PER_TICK: float = 50.0
_UNSERVED_COST_PER_PERSON_MINUTE: float = 0.5
_SECONDARY_CASUALTY_COST_PER_PERSON: float = 5000.0
# Per-tick secondary-loss proxy: scales fire intensity into the
# production_cost roll-up (spec §7 production_cost = response + unserved +
# secondary). Magnitude aligned with the spread-based ground_truth roll-up.
_SECONDARY_COST_PER_FIRE_INTENSITY: float = 250.0
_MUTUAL_AID_COST_PER_UNIT: float = 250.0

# Fire spread threshold + adjacency table (canonical Kobe spike layout).
_FIRE_SPREAD_THRESHOLD: float = 0.35
_ZONE_ADJACENCY: dict[str, list[str]] = {
    "downtown": ["transit_hub", "gov_district", "residential_west"],
    "port_north": ["port_south", "industrial_north"],
    "port_south": ["port_north", "industrial_south"],
    "residential_east": ["transit_hub", "hospital_east", "hillside"],
    "residential_west": ["downtown", "hillside", "industrial_north"],
    "hillside": ["residential_east", "residential_west"],
    "industrial_north": ["residential_west", "port_north"],
    "industrial_south": ["port_south"],
    "hospital_central": ["downtown", "gov_district"],
    "hospital_east": ["residential_east"],
    "transit_hub": ["downtown", "residential_east"],
    "gov_district": ["downtown", "hospital_central"],
}

# Medical surge triggers when buried in any hospital-anchored district
# crosses this fraction of initial population.
_MEDICAL_SURGE_BURIED_FRAC: float = 0.02


def _det_hash(seed: int, tick: int, key: str) -> int:
    """Deterministic integer in ``[0, 1000)`` from ``(seed, tick, key)``.

    Pure SHA-256 (no ``random.Random``) so two backends with identical
    inputs produce identical streams across Python versions.
    """
    body = f"{int(seed)}|{int(tick)}|{key}".encode()
    digest = hashlib.sha256(body).digest()
    return int.from_bytes(digest[:4], "big") % 1000


@dataclass
class _ZoneState:
    zone_id: str
    district: str
    population_initial: int
    buried: int = 0
    fire_intensity: float = 0.0
    unserved_minutes: float = 0.0
    has_hospital: bool = False
    has_school: bool = False
    criticality: float = 0.5
    ambulance_teams: int = 0
    fire_brigade_teams: int = 0
    police_units: int = 0
    triage_priority: str | None = None
    evacuated: bool = False
    cordoned: bool = False
    comms_blackout_until: int = -1


@dataclass
class _DisasterTickRecord:
    """Mock-RCRS analogue of ``pglib_uc_synthetic.TickRecord``.

    The disaster scorer keys off the same field names where they make
    sense (``tick``, ``realized_events``, ``done``) plus disaster-native
    fields (``aggregate_buried``, ``aggregate_fire_intensity`` etc.).
    """

    tick: int
    aggregate_buried: int = 0
    aggregate_fire_intensity: float = 0.0
    aggregate_unserved_minutes: float = 0.0
    aggregate_dispatched_teams: int = 0
    response_cost_this_tick: float = 0.0
    unserved_cost_this_tick: float = 0.0
    realized_events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    # ── Canonical 14-key proxies (computed at the source where per-zone
    # state is live; see spec §7). The native aggregates above feed the
    # reward signal; these feed the scorer contract via scoring_records().
    rescued_this_tick: int = 0
    mutual_aid_cost_this_tick: float = 0.0
    secondary_cost_this_tick: float = 0.0
    reserves_required: float = 0.0
    reserves_procured: float = 0.0
    rho_max: float = 0.0
    n_overloads: int = 0


class MockRcrsBackend:
    """Pure-Python deterministic RCRS substitute.

    The adapter constructs this when the seed's ``backend_config`` has
    ``backend_kind == "mock_rcrs"`` (the default for the spike). The
    real Java/Docker-backed ``RcrsBackend`` is a separate class that
    shares this method surface.
    """

    def __init__(self) -> None:
        self._seed_obj: DisasterScenarioSeed | None = None
        self._tick: int = 0
        self._horizon: int = 144
        self._zones: dict[str, _ZoneState] = {}
        self._tick_records: list[_DisasterTickRecord] = []
        self._mutual_aid_queue: list[tuple[int, float]] = []
        self._mutual_aid_units_total: float = 0.0
        # team pool — total dispatch units the agent has spent (used by the
        # response-cost component); each tick the live dispatched teams
        # are read from per-zone counters.
        self._dispatch_events: list[dict[str, Any]] = []

    # ── Reset ───────────────────────────────────────────────────────────

    def reset(self, scenario_seed: DisasterScenarioSeed) -> None:
        self._seed_obj = scenario_seed
        self._tick = 0
        self._horizon = scenario_seed.horizon_ticks
        self._tick_records.clear()
        self._mutual_aid_queue.clear()
        self._mutual_aid_units_total = 0.0
        self._dispatch_events.clear()
        self._zones = {
            za.zone_id: _ZoneState(
                zone_id=za.zone_id,
                district=za.district,
                population_initial=int(za.population),
                buried=0,
                fire_intensity=0.0,
                unserved_minutes=0.0,
                has_hospital=bool(za.has_hospital),
                has_school=bool(za.has_school),
                criticality=float(za.criticality),
            )
            for za in scenario_seed.zone_assignments
        }
        # Unlike pglib_uc (whose perturbations are idempotent), disaster
        # perturbations like ``hazard_shake`` ADD buried civilians on the
        # tick they fire. Applying them at reset AND again at tick(0)
        # would double-count, so we leave the realized-events list empty
        # at reset and let the first ``tick()`` call run them once.
        self._realized_events_this_tick: list[dict[str, Any]] = []

    # ── Tick advance ────────────────────────────────────────────────────

    def tick(self, current_tick: int) -> _DisasterTickRecord:
        assert self._seed_obj is not None
        self._tick = int(current_tick)

        # 1) Apply perturbations that fire at this tick (replaces
        # _realized_events_this_tick with the new list).
        self._apply_perturbations_at_tick(self._tick)

        # 2) Drain matured mutual-aid arrivals BEFORE rescue/fire so the
        # team count is visible in this tick's record.
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
                    tool_call.get("tool_name") or "request_mutual_aid_team"
                )
                self._realized_events_this_tick.append(
                    {
                        "type": "mutual_aid_arrived",
                        "event_id": f"mutual_aid_arrived:{call_id}:{self._tick}",
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
                        "type": "mutual_aid_arrived",
                        "units": round(legacy_units, 3),
                        "tick": self._tick,
                    }
                )

        # 3) Rescue + medical surge.
        tick_minutes = int(self._seed_obj.tick_minutes or 10)
        rescued_this_tick = 0
        for z in self._zones.values():
            if z.buried > 0:
                # rescue rate: 4 buried per ambulance team per tick,
                # capped at current buried count. Deterministic — no RNG.
                rescue_cap = 4 * z.ambulance_teams
                rescued = min(z.buried, rescue_cap)
                if rescued > 0:
                    z.buried -= rescued
                    rescued_this_tick += rescued
                    self._realized_events_this_tick.append(
                        {
                            "type": "civilians_rescued",
                            "tick": self._tick,
                            "zone": z.zone_id,
                            "rescued": rescued,
                            "remaining_buried": z.buried,
                        }
                    )
                # Unserved minutes accumulate on whatever remains.
                z.unserved_minutes += z.buried * tick_minutes
            # Medical surge auto-trigger
            frac = z.buried / max(z.population_initial, 1)
            if (
                z.has_hospital
                and frac >= _MEDICAL_SURGE_BURIED_FRAC
                and not any(
                    e.get("type") == "medical_surge" and e.get("zone") == z.zone_id
                    for e in self._realized_events_this_tick
                )
            ):
                self._realized_events_this_tick.append(
                    {
                        "type": "medical_surge",
                        "tick": self._tick,
                        "zone": z.zone_id,
                        "buried": z.buried,
                    }
                )

        # 4) Fire propagation + suppression.
        self._propagate_fire(self._tick)

        # 5) Aggregate + cost roll-up.
        agg_buried = sum(z.buried for z in self._zones.values())
        agg_fire = sum(z.fire_intensity for z in self._zones.values())
        agg_unserved = sum(z.unserved_minutes for z in self._zones.values())
        agg_dispatched = sum(
            z.ambulance_teams + z.fire_brigade_teams + z.police_units
            for z in self._zones.values()
        )
        mutual_aid_cost_this_tick = added_units * _MUTUAL_AID_COST_PER_UNIT
        response_cost_this_tick = (
            agg_dispatched * _PER_TEAM_DISPATCH_COST_PER_TICK
            + mutual_aid_cost_this_tick
        )
        unserved_cost_this_tick = (
            sum(z.buried for z in self._zones.values())
            * tick_minutes
            * _UNSERVED_COST_PER_PERSON_MINUTE
        )
        # Secondary-casualty proxy: ongoing fire drives additional loss.
        secondary_cost_this_tick = agg_fire * _SECONDARY_COST_PER_FIRE_INTENSITY

        # ── Canonical 14-key proxies (spec §7). Computed here where per-zone
        # state is live, post-rescue/post-fire. rho/overloads use the local
        # rescue capacity (4 buried per ambulance team).
        rho_max = 0.0
        n_overloads = 0
        for z in self._zones.values():
            cap_z = 4 * z.ambulance_teams
            if z.buried <= 0:
                continue
            rho_z = z.buried / float(max(cap_z, 1))
            if rho_z > rho_max:
                rho_max = rho_z
            if z.buried > cap_z:
                n_overloads += 1
        reserves_required = agg_buried / 4.0
        reserves_procured = float(agg_dispatched)

        record = _DisasterTickRecord(
            tick=self._tick,
            aggregate_buried=int(agg_buried),
            aggregate_fire_intensity=round(agg_fire, 3),
            aggregate_unserved_minutes=round(agg_unserved, 2),
            aggregate_dispatched_teams=int(agg_dispatched),
            response_cost_this_tick=round(response_cost_this_tick, 2),
            unserved_cost_this_tick=round(unserved_cost_this_tick, 2),
            realized_events=list(self._realized_events_this_tick),
            done=False,
            rescued_this_tick=int(rescued_this_tick),
            mutual_aid_cost_this_tick=round(mutual_aid_cost_this_tick, 2),
            secondary_cost_this_tick=round(secondary_cost_this_tick, 2),
            reserves_required=round(reserves_required, 3),
            reserves_procured=reserves_procured,
            rho_max=round(rho_max, 4),
            n_overloads=int(n_overloads),
        )
        self._realized_events_this_tick = []
        self._tick_records.append(record)
        return record

    def _propagate_fire(self, tick: int) -> None:
        assert self._seed_obj is not None
        # Suppression: each fire_brigade team reduces fire_intensity by 0.25.
        for z in self._zones.values():
            if z.fire_brigade_teams > 0 and z.fire_intensity > 0:
                reduction = min(z.fire_intensity, 0.25 * z.fire_brigade_teams)
                z.fire_intensity = round(max(0.0, z.fire_intensity - reduction), 3)
        # Spread: any zone with intensity > threshold and NO brigade leaks
        # to a deterministic adjacent zone (lowest zone_id alphabetically
        # among neighbours present in the seed). Spread = +0.2 capped at 1.
        for src_id in sorted(self._zones.keys()):
            z = self._zones[src_id]
            if z.fire_intensity <= _FIRE_SPREAD_THRESHOLD:
                continue
            if z.fire_brigade_teams > 0:
                continue
            neighbours = [
                n for n in _ZONE_ADJACENCY.get(src_id, []) if n in self._zones
            ]
            if not neighbours:
                continue
            # Deterministic pick: hash chooses one neighbour.
            idx = _det_hash(self._seed_obj.seed, tick, f"fire|{src_id}") % len(
                neighbours
            )
            tgt = neighbours[idx]
            before = self._zones[tgt].fire_intensity
            self._zones[tgt].fire_intensity = round(min(1.0, before + 0.2), 3)
            if before <= _FIRE_SPREAD_THRESHOLD < self._zones[tgt].fire_intensity:
                self._realized_events_this_tick.append(
                    {
                        "type": "fire_spread",
                        "tick": tick,
                        "from_zone": src_id,
                        "to_zone": tgt,
                        "intensity": self._zones[tgt].fire_intensity,
                    }
                )

    # ── Perturbations ───────────────────────────────────────────────────

    def _apply_perturbations_at_tick(self, tick: int) -> None:
        assert self._seed_obj is not None
        events: list[dict[str, Any]] = []
        for p in self._seed_obj.perturbations:
            if not (p.trigger_tick <= tick < p.trigger_tick + p.duration_ticks):
                continue
            applied = self._apply_one_perturbation(p, tick)
            if applied:
                events.append(applied)
        self._realized_events_this_tick = events

    def _apply_one_perturbation(self, p: Any, tick: int) -> dict[str, Any] | None:
        assert self._seed_obj is not None
        kind = str(p.kind)
        if kind == "hazard_shake":
            # On the tick the shake fires, inject buried proportional to
            # intensity in the epicenter and its neighbours.
            if tick != p.trigger_tick:
                return None
            epicenter = str(p.target.get("epicenter_zone", "downtown"))
            primary_buried_frac = 0.012 * float(p.intensity)
            for zid, z in self._zones.items():
                if zid == epicenter:
                    new = int(z.population_initial * primary_buried_frac)
                elif zid in _ZONE_ADJACENCY.get(epicenter, []):
                    new = int(z.population_initial * primary_buried_frac * 0.4)
                else:
                    new = int(z.population_initial * primary_buried_frac * 0.05)
                z.buried += new
            return {
                "type": "hazard_shake",
                "tick": tick,
                "epicenter": epicenter,
                "intensity": float(p.intensity),
                "hidden": bool(p.hidden),
            }
        if kind == "building_collapse":
            if tick != p.trigger_tick:
                return None
            zone_id = str(p.target.get("zone", "downtown"))
            n_buildings = int(p.target.get("n_buildings", 1))
            if zone_id in self._zones:
                # ~30 occupants per collapsed building deterministically.
                added = int(n_buildings * 30 * float(p.intensity))
                # Slight per-collapse jitter (deterministic) so two
                # different collapse events don't collide on the same
                # buried count.
                jitter = _det_hash(self._seed_obj.seed, tick, f"collapse|{zone_id}")
                added += jitter % 5
                self._zones[zone_id].buried += added
                return {
                    "type": "building_collapse",
                    "tick": tick,
                    "zone": zone_id,
                    "n_buildings": n_buildings,
                    "buried_added": added,
                    "hidden": bool(p.hidden),
                }
        if kind == "aftershock":
            if tick != p.trigger_tick:
                return None
            epicenter = str(p.target.get("epicenter_zone", "downtown"))
            magnitude = float(p.target.get("magnitude", 4.5))
            scale = 0.004 * (magnitude / 5.0)
            for zid, z in self._zones.items():
                if zid == epicenter or zid in _ZONE_ADJACENCY.get(epicenter, []):
                    z.buried += int(z.population_initial * scale)
            return {
                "type": "aftershock",
                "tick": tick,
                "epicenter": epicenter,
                "magnitude": magnitude,
                "hidden": bool(p.hidden),
            }
        if kind == "fire_spread":
            # Igniting a zone — only on first tick of the perturbation.
            if tick != p.trigger_tick:
                return None
            zone_id = str(p.target.get("zone", "downtown"))
            if zone_id in self._zones:
                self._zones[zone_id].fire_intensity = round(
                    min(1.0, self._zones[zone_id].fire_intensity + 0.5), 3
                )
            return {
                "type": "fire_spread",
                "tick": tick,
                "from_zone": zone_id,
                "to_zone": zone_id,
                "intensity": float(p.intensity),
                "hidden": bool(p.hidden),
            }
        if kind == "medical_surge":
            if tick != p.trigger_tick:
                return None
            return {
                "type": "medical_surge",
                "tick": tick,
                "zone": str(p.target.get("hospital_id", "hospital_central")),
                "casualty_rate": int(p.target.get("casualty_rate", 5)),
                "hidden": bool(p.hidden),
            }
        if kind == "comms_blackout":
            # Mark zone with comms_blackout_until so the adapter / fog
            # layer can flag stale observations.
            if tick != p.trigger_tick:
                return None
            zone_id = str(p.target.get("zone", "downtown"))
            if zone_id in self._zones:
                self._zones[zone_id].comms_blackout_until = (
                    p.trigger_tick + p.duration_ticks
                )
            return {
                "type": "comms_blackout",
                "tick": tick,
                "zone": zone_id,
                "intensity": float(p.intensity),
                "hidden": bool(p.hidden),
            }
        if kind == "road_blockage":
            if tick != p.trigger_tick:
                return None
            return {
                "type": "road_blockage",
                "tick": tick,
                "edge": str(p.target.get("edge", "")),
                "hidden": bool(p.hidden),
            }
        if kind == "bridge_failure":
            if tick != p.trigger_tick:
                return None
            return {
                "type": "bridge_failure",
                "tick": tick,
                "bridge": str(p.target.get("bridge", "")),
                "hidden": bool(p.hidden),
            }
        if kind == "gas_leak":
            if tick != p.trigger_tick:
                return None
            return {
                "type": "gas_leak",
                "tick": tick,
                "zone": str(p.target.get("zone", "")),
                "hidden": bool(p.hidden),
            }
        if kind == "tsunami_inundation":
            if tick != p.trigger_tick:
                return None
            zone_id = str(p.target.get("zone", "port_south"))
            if zone_id in self._zones:
                # Tsunami injects buried + evacuates that zone immediately.
                self._zones[zone_id].buried += int(
                    self._zones[zone_id].population_initial * 0.08
                )
                self._zones[zone_id].evacuated = True
            return {
                "type": "tsunami_inundation",
                "tick": tick,
                "zone": zone_id,
                "depth_m": float(p.target.get("depth_m", 1.0)),
                "hidden": bool(p.hidden),
            }
        return None

    # ── F-01 Mutual aid (delayed effect) ────────────────────────────────

    def queue_mutual_aid_effect(
        self,
        *,
        due_tick: int,
        mw: float,
        tool_call: dict[str, Any] | None = None,
    ) -> None:
        """F-01 contract — disaster side reuses ``mw`` as team_units.

        The arg name is preserved (``mw``) so the native tool plumbing
        and the cross-domain test ``test_mutual_aid_unified_delay_contract``
        can call all backends with identical kwargs.
        """
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

    # ── Tool effects ────────────────────────────────────────────────────

    def apply_tool_effect(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Handle one disaster-domain state-changing tool call."""
        if name == "dispatch_ambulance":
            return self._dispatch_teams(args, kind="ambulance")
        if name == "dispatch_fire_brigade":
            return self._dispatch_teams(args, kind="fire_brigade")
        if name == "dispatch_police_cordon":
            return self._dispatch_teams(args, kind="police")
        if name == "assign_triage":
            return self._assign_triage(args)
        if name == "evacuate_zone":
            return self._evacuate_zone(args)
        if name == "survey_zone" or name == "dispatch_recon":
            # No state mutation — survey/recon are investigation tools;
            # the adapter routes them through fog_of_war reveal. We still
            # return a ground-truth peek so the handler can echo it.
            zone_id = str(args.get("target_zone") or args.get("zone") or "")
            z = self._zones.get(zone_id)
            if not z:
                return {"_status": "error", "error": "unknown_zone", "zone": zone_id}
            return {
                "zone": zone_id,
                "buried": z.buried,
                "fire_intensity": z.fire_intensity,
                "evacuated": z.evacuated,
                "population_initial": z.population_initial,
                "ambulance_teams": z.ambulance_teams,
                "fire_brigade_teams": z.fire_brigade_teams,
                "police_units": z.police_units,
            }
        if name == "request_mutual_aid_team":
            # Defensive — the dedicated handler in native_tools.py queues
            # the delayed effect; this path stays as a non-mutating ack.
            return {
                "_status": "ack",
                "info": (
                    "mutual-aid uses the dedicated delayed-effect path; "
                    "this code path no longer mutates state"
                ),
            }
        return {"_status": "ack"}

    def _dispatch_teams(self, args: dict[str, Any], *, kind: str) -> dict[str, Any]:
        zone_id = str(args.get("target_zone") or args.get("zone") or "")
        n_teams = int(args.get("n_teams", 1) or 1)
        if n_teams <= 0:
            return {
                "_status": "error",
                "error": "non_positive_n_teams",
                "n_teams": n_teams,
            }
        z = self._zones.get(zone_id)
        if not z:
            return {"_status": "error", "error": "unknown_zone", "zone": zone_id}
        if kind == "ambulance":
            z.ambulance_teams += n_teams
        elif kind == "fire_brigade":
            z.fire_brigade_teams += n_teams
        elif kind == "police":
            z.police_units += n_teams
            z.cordoned = True
        self._dispatch_events.append(
            {"tick": self._tick, "zone": zone_id, "kind": kind, "n_teams": n_teams}
        )
        return {
            "zone": zone_id,
            "kind": kind,
            "n_teams_added": n_teams,
            "ambulance_teams": z.ambulance_teams,
            "fire_brigade_teams": z.fire_brigade_teams,
            "police_units": z.police_units,
            "criticality": z.criticality,
            "stakeholder_class": _stakeholder_for_zone(z),
        }

    def _assign_triage(self, args: dict[str, Any]) -> dict[str, Any]:
        zone_id = str(args.get("zone") or args.get("target_zone") or "")
        priority = str(args.get("priority", "GREEN")).upper()
        z = self._zones.get(zone_id)
        if not z:
            return {"_status": "error", "error": "unknown_zone", "zone": zone_id}
        if priority not in {"RED", "YELLOW", "GREEN", "BLACK"}:
            return {
                "_status": "error",
                "error": "unknown_priority",
                "priority": priority,
            }
        z.triage_priority = priority
        return {
            "zone": zone_id,
            "priority": priority,
            "stakeholder_class": _stakeholder_for_zone(z),
            "criticality": z.criticality,
        }

    def _evacuate_zone(self, args: dict[str, Any]) -> dict[str, Any]:
        zone_id = str(args.get("zone") or args.get("target_zone") or "")
        destination = str(args.get("destination", ""))
        route_hint = str(args.get("route_hint", ""))
        z = self._zones.get(zone_id)
        if not z:
            return {"_status": "error", "error": "unknown_zone", "zone": zone_id}
        z.evacuated = True
        return {
            "zone": zone_id,
            "destination": destination,
            "route_hint": route_hint,
            "stakeholder_class": _stakeholder_for_zone(z),
            "criticality": z.criticality,
        }

    # ── Snapshot ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        zones = {
            zid: {
                "kind": "zone",
                "district": z.district,
                "population_initial": z.population_initial,
                "buried": z.buried,
                "fire_intensity": z.fire_intensity,
                "unserved_minutes": round(z.unserved_minutes, 2),
                "ambulance_teams": z.ambulance_teams,
                "fire_brigade_teams": z.fire_brigade_teams,
                "police_units": z.police_units,
                "triage_priority": z.triage_priority,
                "evacuated": z.evacuated,
                "cordoned": z.cordoned,
                "has_hospital": z.has_hospital,
                "criticality": z.criticality,
                "comms_blackout_active": (
                    z.comms_blackout_until >= 0 and self._tick < z.comms_blackout_until
                ),
            }
            for zid, z in self._zones.items()
        }
        last = self._tick_records[-1] if self._tick_records else None
        return {
            "entities": zones,
            "totals": {
                "aggregate_buried": (last.aggregate_buried if last else 0),
                "aggregate_fire_intensity": (
                    last.aggregate_fire_intensity if last else 0.0
                ),
                "aggregate_unserved_minutes": (
                    last.aggregate_unserved_minutes if last else 0.0
                ),
                "aggregate_dispatched_teams": (
                    last.aggregate_dispatched_teams if last else 0
                ),
                "mutual_aid_units_total": round(self._mutual_aid_units_total, 3),
            },
            "tick": self._tick,
            "horizon": self._horizon,
        }

    # ── Forecasts ───────────────────────────────────────────────────────

    def forecast_for(self, horizon: int) -> list[dict[str, Any]]:
        """Synthetic aftershock + casualty arrival forecast.

        Deterministic in ``(seed, current_tick, k)``. Used by the
        foresight scorer + the agent's ``commit_to_plan`` tool to
        check whether the agent predicted a hidden aftershock.
        """
        if self._seed_obj is None:
            return []
        out: list[dict[str, Any]] = []
        for k in range(int(horizon)):
            absolute_tick = self._tick + k
            # Aftershock probability decays slowly from 5% → 1% over the
            # next 24 ticks. Deterministic per-tick.
            base_prob = max(0.01, 0.05 * (0.95**k))
            casualty_rate = max(0, 6 - k // 2)
            jitter = (
                _det_hash(self._seed_obj.seed, self._tick, f"forecast|{k}") / 1000.0
            )  # [0, 1)
            out.append(
                {
                    "tick": absolute_tick,
                    "aftershock_probability": round(
                        base_prob + 0.01 * (jitter - 0.5), 4
                    ),
                    "casualty_rate_per_hospital": casualty_rate,
                    "forecast_index": k,
                }
            )
        return out

    # ── Cost roll-up ────────────────────────────────────────────────────

    def ground_truth_costs(self) -> dict[str, float]:
        if not self._tick_records:
            return {
                "response_cost": 0.0,
                "unserved_population_cost": 0.0,
                "secondary_casualty_cost": 0.0,
                "mutual_aid_cost": 0.0,
            }
        response = sum(r.response_cost_this_tick for r in self._tick_records)
        unserved = sum(r.unserved_cost_this_tick for r in self._tick_records)
        # Secondary casualty proxy: each spreading fire event in
        # realized_events implies ~5 person-equivalent secondary casualties.
        spread_count = 0
        for r in self._tick_records:
            for ev in r.realized_events:
                if ev.get("type") == "fire_spread":
                    spread_count += 1
        secondary = spread_count * 5 * _SECONDARY_CASUALTY_COST_PER_PERSON / 100.0
        mutual_aid = self._mutual_aid_units_total * _MUTUAL_AID_COST_PER_UNIT
        return {
            "response_cost": round(response, 2),
            "unserved_population_cost": round(unserved, 2),
            "secondary_casualty_cost": round(secondary, 2),
            "mutual_aid_cost": round(mutual_aid, 2),
        }

    # Disaster-native per-tick rows (buried / fire / teams). Retained for
    # diagnostics and the skeleton test; NOT the scorer contract.
    def native_scoring_records(self) -> list[dict[str, Any]]:
        return [
            {
                "tick": r.tick,
                "aggregate_buried": r.aggregate_buried,
                "aggregate_fire_intensity": r.aggregate_fire_intensity,
                "aggregate_unserved_minutes": r.aggregate_unserved_minutes,
                "aggregate_dispatched_teams": r.aggregate_dispatched_teams,
                "response_cost_this_tick": r.response_cost_this_tick,
                "unserved_cost_this_tick": r.unserved_cost_this_tick,
            }
            for r in self._tick_records
        ]

    def scoring_records(self) -> list[dict[str, Any]]:
        """Per-tick rows in the canonical 14-key scorer contract, with the
        §7 disaster→canonical proxy mapping. ``n_voltage_violations`` and
        ``n_disconnected_lines`` are honest-0 (disaster has no electrical
        network). ``done`` is early-guarded ``bool(done and tick <
        horizon-1)`` so a normal horizon-end tick is never miscounted as a
        catastrophic collapse by ``score_system_survival``.
        """
        rows: list[dict[str, Any]] = []
        for r in self._tick_records:
            rows.append(
                {
                    "tick": r.tick,
                    # rescue-throughput proxies (person-units, not MW)
                    "aggregate_demand_mw": float(
                        r.aggregate_buried + r.rescued_this_tick
                    ),
                    "aggregate_generation_mw": float(r.rescued_this_tick),
                    "balance_error_mw": float(r.aggregate_buried),
                    "reserves_required_mw": float(r.reserves_required),
                    "reserves_procured_mw": float(r.reserves_procured),
                    # casualty + property-loss proxies
                    "production_cost": round(
                        r.response_cost_this_tick
                        + r.unserved_cost_this_tick
                        + r.secondary_cost_this_tick,
                        2,
                    ),
                    "startup_cost": float(r.mutual_aid_cost_this_tick),
                    "shed_penalty": float(r.unserved_cost_this_tick),
                    # zone-overload proxies
                    "rho_max": float(r.rho_max),
                    "n_overloads": int(r.n_overloads),
                    # honest-0: disaster has no electrical network
                    "n_voltage_violations": 0,
                    "n_disconnected_lines": 0,
                    "done": bool(r.done and r.tick < self._horizon - 1),
                }
            )
        return rows

    def per_zone_unserved_minutes(self) -> dict[str, float]:
        """Disaster analogue of ``per_load_shed_mwh``.

        Used by the cross-domain audit gate + the adaptive-replanning
        scorer. MUST be deterministic — two ``reset``+``tick`` loops
        with the same seed return identical maps.
        """
        return {zid: round(z.unserved_minutes, 3) for zid, z in self._zones.items()}


def _stakeholder_for_zone(z: _ZoneState) -> str:
    """Map a zone to its canonical disaster stakeholder class.

    Mirrors ``power_grid.native_stakeholders`` shed-class derivation:
    ``hospital`` if the zone has a hospital; otherwise ``civilian``.
    Responder classes are populated by the adapter from the team-type
    of the tool that fired (ambulance ⇒ responder_ems, etc.).
    """
    if z.has_hospital:
        return "hospital"
    return "civilian"
