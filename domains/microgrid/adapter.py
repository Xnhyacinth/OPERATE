"""
domains.microgrid.adapter — Microgrid EMS POMDP environment.

Mirrors the structure of ``domains.logistics.adapter`` /
``domains.power_grid.adapter`` — backend + fog + belief + tool registry +
(family-gated) stakeholder manager + dilemma manager + evidence logger +
cascade-bus publisher — but uses microgrid-native types and never imports
from another domain (Red Line #3).

The backend is selected by ``backend_kind``: the three ``pymgrid_*`` labels
all map to the pure-Python ``EmsSimulator`` (the simulator IS the
environment; pymgrid is never required to step), and ``pandapower_lv`` maps
to the LV power-flow tier.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from typing import Any

from core import (
    Action,
    BeliefStateTracker,
    CascadeBus,
    CascadeEvent,
    EthicalDilemmaManager,
    EvidenceLogger,
    FogOfWarPolicy,
    POMDPEnvironment,
    StakeholderTrustManager,
    StepInfo,
    StepReturn,
    TickBudget,
    ToolCall,
    ToolContext,
    ToolRegistry,
    ToolResult,
    arm_dilemmas,
    safe_dataclass_to_dict,
)
from core.difficulty_levels import canonical_difficulty_level
from core.evidence import control_summary_from_evidence
from core.world_evolution_contract import canonicalize_runtime_events
from domains.registry import apply_supervisory_cadence

from .backends.ems_sim import EmsSimulator
from .oracle import compute_reference_optimum
from .seeds.schema import (
    DilemmaSeed,
    MicrogridLoad,
    MicrogridScenarioSeed,
    Perturbation,
    Provenance,
)

# Backend-emitted event type → canonical cascade event_type.
_CASCADE_EVENT_MAPPING: dict[str, str] = {
    "grid_outage": "microgrid.pcc.islanded",
    "pcc_reconnected": "microgrid.pcc.reconnected",
    "pcc_disconnected": "microgrid.pcc.disconnected",
    "der_failure": "microgrid.der.failure",
    "pv_ramp": "microgrid.pv.ramp",
    "load_spike": "microgrid.load.spike",
    "price_spike": "microgrid.price.spike",
    "genset_started": "microgrid.genset.started",
}


def _state_digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _control_state_for_call(
    snapshot: dict[str, Any],
    *,
    tool_name: str,
    args: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the bounded native state a supported control can affect."""
    entities = snapshot.get("entities") or {}
    if not isinstance(entities, dict):
        return None
    if tool_name == "set_battery_dispatch":
        battery_id = str(args.get("battery_id") or "batt0")
        battery = entities.get(battery_id)
        if not isinstance(battery, dict):
            return None
        return {
            "battery_id": battery_id,
            "commanded_p_mw": battery.get("commanded_p_mw"),
            "applied_p_mw": battery.get("applied_p_mw", battery.get("p_mw")),
            "soc_mwh": battery.get("soc_mwh"),
        }
    if tool_name == "dispatch_genset":
        genset_id = str(args.get("genset_id") or "genset0")
        genset = entities.get(genset_id)
        if not isinstance(genset, dict):
            return None
        return {
            "genset_id": genset_id,
            "available": genset.get("available"),
            "committed": genset.get("committed"),
            "output_mw": genset.get("output_mw"),
        }
    if tool_name == "connect_pcc":
        pcc = entities.get("pcc")
        if not isinstance(pcc, dict):
            return None
        return {
            "pcc_id": "pcc",
            "connected": pcc.get("connected"),
            "islanded": pcc.get("islanded"),
        }
    return None


def _native_microgrid_action_effect_events(
    *,
    tool_results: list[Any],
    calls_by_id: dict[str, Any],
    before_control_state: dict[str, Any],
    applied_control_state: dict[str, Any],
    post_tick_control_state: dict[str, Any],
    applied_tick: int,
) -> list[dict[str, Any]]:
    """Emit a post-physics edge only for a proven native control effect."""
    effects: list[dict[str, Any]] = []
    for result in tool_results:
        if (
            not getattr(result, "ok", False)
            or not getattr(result, "state_changing", False)
            or not isinstance(getattr(result, "payload", None), dict)
        ):
            continue
        payload = result.payload
        if payload.get("_status") == "pending":
            continue
        call_id = str(getattr(result, "call_id", "") or "")
        call = calls_by_id.get(call_id)
        tool_name = str(result.name)
        if call is None or tool_name not in {
            "set_battery_dispatch",
            "dispatch_genset",
            "connect_pcc",
        }:
            continue
        before = _control_state_for_call(
            before_control_state,
            tool_name=str(result.name),
            args=dict(call.args),
        )
        applied = _control_state_for_call(
            applied_control_state,
            tool_name=str(result.name),
            args=dict(call.args),
        )
        post = _control_state_for_call(
            post_tick_control_state,
            tool_name=str(result.name),
            args=dict(call.args),
        )
        if before is None or applied is None or post is None:
            continue
        if tool_name == "set_battery_dispatch":
            physical_effect = bool(
                before != applied
                and applied.get("commanded_p_mw") == post.get("commanded_p_mw")
                and (
                    post.get("applied_p_mw") != before.get("applied_p_mw")
                    or post.get("soc_mwh") != before.get("soc_mwh")
                )
            )
            event_type = "microgrid_battery_dispatch_applied"
            changed_state_fields = [
                "battery_commanded_p_mw",
                "battery_applied_p_mw",
                "battery_soc_mwh",
            ]
        elif tool_name == "dispatch_genset":
            requested_output = float(call.args.get("p_mw") or 0.0)
            try:
                actual_output = float(post.get("output_mw") or 0.0)
            except (TypeError, ValueError):
                continue
            physical_effect = bool(
                before != post
                and post.get("committed") is True
                and abs(actual_output - requested_output) <= 1e-9
            )
            event_type = "microgrid_genset_dispatch_applied"
            changed_state_fields = [
                "genset_committed",
                "genset_output_mw",
            ]
        else:
            requested_connection = bool(call.args.get("connect", True))
            physical_effect = bool(
                before.get("connected") != post.get("connected")
                and post.get("connected") is requested_connection
            )
            event_type = "microgrid_pcc_connection_applied"
            changed_state_fields = ["pcc_connected", "pcc_islanded"]
        if not physical_effect:
            continue
        event_id = f"microgrid-agent-control:{call_id}:{applied_tick}"
        evidence_ids = [
            evidence_id
            for evidence_id in [getattr(result, "evidence_id", None)]
            if isinstance(evidence_id, str) and evidence_id
        ]
        effects.append(
            {
                "event_id": event_id,
                "type": event_type,
                "origin": "agent_caused",
                "agent_caused": True,
                "tick": applied_tick,
                "actionable": False,
                "decision_required": False,
                "changed_state_fields": changed_state_fields,
                "call_id": call_id,
                "tool_name": str(result.name),
                "requested_action": {
                    "name": str(call.name),
                    "args": dict(call.args),
                },
                "applied_action": {
                    "control_state_after_tool": applied,
                    "post_tick_control_state": post,
                },
                "before_state_digest": _state_digest(before),
                "after_state_digest": _state_digest(
                    {
                        "control_state_after_tool": applied,
                        "post_tick_control_state": post,
                    }
                ),
                "outcome_tick": applied_tick + 1,
                "evidence_ids": evidence_ids,
                "action_to_outcome_edge": {
                    "source": f"call:{call_id}",
                    "target": f"outcome:{event_id}",
                    "kind": "action_to_outcome",
                },
            }
        )
    return effects


class MicrogridEnvironment(POMDPEnvironment):
    """OPERATE environment for the microgrid EMS domain."""

    domain = "microgrid"

    def __init__(self, cascade_bus: CascadeBus | None = None) -> None:
        self._seed_obj: MicrogridScenarioSeed | None = None
        self._tick = 0
        self._horizon = 24
        self._backend: Any = None
        self._fog: FogOfWarPolicy | None = None
        self._belief: BeliefStateTracker | None = None
        self._tools: ToolRegistry | None = None
        self._stakeholders: StakeholderTrustManager | None = None
        self._dilemmas: EthicalDilemmaManager | None = None
        self._evidence: EvidenceLogger | None = None
        self._cascade_bus = cascade_bus or CascadeBus()
        self._episode_id = ""
        # Delayed controls materialize in a later native control interval.
        # Retain only their original protocol calls until that interval so a
        # later state effect can be tied to the initiating call_id.
        self._pending_native_control_calls: dict[str, Any] = {}
        self._agency_source_events: list[dict[str, Any]] = []
        self._agency_parent_by_call: dict[str, dict[str, Any]] = {}
        # Phase 3 (Part D): logistics -> microgrid cascade
        self._pending_cascade_perturbations: list[Any] = []
        self._cascade_bus.subscribe(
            "logistics.fuel.delay", self._on_logistics_fuel_delay
        )
        self._cascade_bus.subscribe(
            "traffic.signal.failure", self._on_traffic_signal_failure
        )

    # ── POMDPEnvironment surface ────────────────────────────────────────

    def reset(self, scenario_config: dict[str, Any], seed: int) -> dict[str, Any]:
        seed_obj = _rebuild_seed_from_dict(scenario_config, override_seed=seed)
        self._seed_obj = seed_obj
        self._tick = 0
        self._horizon = seed_obj.horizon_ticks
        self._episode_id = f"dt_{seed_obj.signature()}_s{seed}"
        self._pending_cascade_perturbations = []
        self._pending_native_control_calls = {}
        self._agency_source_events = []
        self._agency_parent_by_call = {}

        self._backend = _build_backend(seed_obj.backend_kind)
        self._backend.reset(seed_obj)

        # Cache the deterministic economic-dispatch reference optimum so the
        # scorer's optimality_gap reads a replay-stable cost floor off
        # ``backend_config['reference_optimum']`` (convex LP via cvxpy, or the
        # greedy fallback when cvxpy is absent). Pure function of the seed.
        with contextlib.suppress(Exception):
            compute_reference_optimum(seed_obj, cache=True)

        self._fog = FogOfWarPolicy(
            hide_rules=[], noise_rules=[], staleness_rules=[], seed=seed
        )
        self._fog.reset(seed=seed)

        self._belief = BeliefStateTracker()
        self._belief.reset()

        self._tools = ToolRegistry(
            budget=_build_tick_budget(seed_obj),
            seed=seed,
            difficulty_level=seed_obj.difficulty_level,
        )
        self._tools.reset(seed=seed)
        _register_native_tools(self._tools, self._backend, self)

        self._stakeholders = StakeholderTrustManager()
        _register_native_stakeholders(self._stakeholders, seed_obj)

        self._dilemmas = EthicalDilemmaManager()
        self._dilemmas.reset()
        arm_dilemmas(self._dilemmas, seed_obj.dilemmas)

        self._evidence = EvidenceLogger(episode_id=self._episode_id)

        obs = self.snapshot()
        self._belief.update_from_observation(obs, tick=0)
        return obs

    def step(self, action: Action) -> StepReturn:
        assert self._tools is not None and self._backend is not None
        assert self._fog is not None and self._evidence is not None
        assert self._dilemmas is not None and self._belief is not None

        ctx = ToolContext(
            tick=self._tick,
            seed=int(self._seed_obj.seed if self._seed_obj else 0),
            backend=self._backend,
            extra={
                "fog": self._fog,
                "stakeholders": self._stakeholders,
                "dilemmas": self._dilemmas,
                "evidence": self._evidence,
                "cascade_bus": self._cascade_bus,
                "env": self,
            },
        )
        before_control_state = self._backend.snapshot()
        investigation_stage_open = self._consume_within_tick_budget_state()
        tool_results = self._tools.execute_action(
            action, ctx, begin_tick=not investigation_stage_open
        )
        for call in action.tool_calls:
            spec = self._tools.get(call.name)
            if spec is None or not spec.state_changing or call.call_id is None:
                continue
            source_parent = self._agency_source_parent_for_call(
                call=call,
                request_tick=self._tick,
                allow_same_tick_reveal=investigation_stage_open,
            )
            if source_parent is not None:
                self._agency_parent_by_call[str(call.call_id)] = source_parent
        calls_by_id = dict(self._pending_native_control_calls)
        calls_by_id.update(
            {
                str(call.call_id): call
                for call in action.tool_calls
                if call.call_id is not None
            }
        )
        pending_call_ids = set(self._pending_native_control_calls)
        current_call_ids = {
            str(call.call_id) for call in action.tool_calls if call.call_id is not None
        }
        for r in tool_results:
            state_changing = bool(r.state_changing)
            call_id = str(r.call_id or "")
            materialized_pending = bool(
                call_id
                and call_id in pending_call_ids
                and call_id not in current_call_ids
                and isinstance(r.payload, dict)
                and r.payload.get("_status") == "pending"
            )
            if materialized_pending:
                r.payload = {
                    **r.payload,
                    "_status": "materialized",
                    "materialized_from": "pending",
                }
            r.evidence_id = self._evidence.log(
                kind="tool_call",
                tick=self._tick,
                payload={
                    "name": r.name,
                    "ok": r.ok,
                    "error_code": r.error_code,
                    "cost_units": r.cost_units,
                    "call_id": r.call_id,
                    "state_changing": state_changing,
                    "payload": r.payload,
                    **self.tool_dependency_payload(action, r),
                },
                source="tool",
            )
            if not call_id:
                continue
            if isinstance(r.payload, dict) and r.payload.get("_status") == "pending":
                call = calls_by_id.get(call_id)
                if call is not None:
                    self._pending_native_control_calls[call_id] = call
            else:
                self._pending_native_control_calls.pop(call_id, None)

        applied_control_state = self._backend.snapshot()

        # Drain pending cascade perturbations (logistics fuel delay -> microgrid)
        # before the backend ticks, so the cascaded effect lands this tick.
        if self._pending_cascade_perturbations:
            for p in self._pending_cascade_perturbations:
                try:
                    p_kind = getattr(p, "kind", "der_failure")
                    p_intensity = getattr(p, "intensity", 0.5)
                    if not hasattr(self._backend, "apply_perturbation"):
                        continue
                    self._backend.apply_perturbation(p)
                    if self._evidence:
                        self._evidence.log(
                            "cascade_effect",
                            self._tick,
                            payload={
                                "kind": p_kind,
                                "intensity": p_intensity,
                                "source": "logistics.fuel.delay",
                            },
                            source="cascade_bus",
                        )
                except Exception:
                    pass
            self._pending_cascade_perturbations = []

        backend_tick_record = self._backend.tick(self._tick)
        backend_tick_payload = safe_dataclass_to_dict(backend_tick_record)
        if not isinstance(backend_tick_payload, dict):
            backend_tick_payload = {}
        backend_events = list(getattr(backend_tick_record, "realized_events", []) or [])
        post_tick_control_state = self._backend.snapshot()
        native_action_effect_events = _native_microgrid_action_effect_events(
            tool_results=tool_results,
            calls_by_id=calls_by_id,
            before_control_state=before_control_state,
            applied_control_state=applied_control_state,
            post_tick_control_state=post_tick_control_state,
            applied_tick=self._tick,
        )
        realized_events = [*backend_events, *native_action_effect_events]
        for event in realized_events:
            call_id = str(event.get("call_id") or "")
            parent = self._agency_parent_by_call.get(call_id)
            if str(event.get("origin") or "") == "agent_caused" and parent is not None:
                event["causal_parent_event_id"] = parent["event_id"]
        world_evolution_records = canonicalize_runtime_events(
            realized_events,
            applied_tick=self._tick,
        )
        self._evidence.log(
            kind="backend_tick",
            tick=self._tick,
            payload=backend_tick_payload,
            source="engine",
        )
        self._evidence.log(
            kind="cost_summary",
            tick=self._tick,
            payload=self._cost_summary_payload(backend_tick_record),
            source="engine",
        )
        for index, ev in enumerate(realized_events):
            evidence_id = self._evidence.log(
                kind="realized_event",
                tick=self._tick,
                payload=dict(ev),
                source="engine",
            )
            evidence_ids = list(ev.get("evidence_ids") or [])
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
            ev["evidence_ids"] = evidence_ids
            if index < len(world_evolution_records):
                world_evolution_records[index]["evidence_ids"] = list(evidence_ids)
                if world_evolution_records[index]["origin"] in {
                    "source_schedule",
                    "declared_perturbation",
                }:
                    self._agency_source_events.append(
                        {
                            "event_id": world_evolution_records[index]["event_id"],
                            "evidence_id": evidence_id,
                            "applied_tick": self._tick,
                            "hidden": world_evolution_records[index]["visibility"]
                            == "hidden",
                            "reveal_targets": self._agency_event_reveal_targets(ev),
                            "revealed_tick": None,
                            "reveal_evidence_ids": [],
                        }
                    )
            self._publish_cascade_for_event(ev, evidence_id)

        current_snap = self.snapshot()
        triggered = self._dilemmas.maybe_trigger(self._tick, current_snap)
        for d in triggered:
            self._evidence.log(
                kind="dilemma_triggered",
                tick=self._tick,
                payload={"dilemma_id": d.dilemma_id, "description": d.description},
                source="engine",
            )
        defaults_fired = self._dilemmas.fire_defaults_for_missed_deadlines(self._tick)
        for choice in defaults_fired:
            self._evidence.log(
                kind="moral_choice",
                tick=self._tick,
                payload={
                    "dilemma_id": choice.dilemma_id,
                    "option_id": choice.chosen_option_id,
                    "rationale": choice.rationale,
                    "source": "default_option_fired",
                },
                source="engine",
            )

        if self._stakeholders is not None:
            self._stakeholders.tick(self._tick)

        self._tick += 1
        backend_done = bool(getattr(backend_tick_record, "done", False))
        done = backend_done or self._tick >= self._horizon

        snap = self.snapshot()
        snap["tick"] = self._tick
        self._belief.update_from_observation(snap, tick=self._tick)

        reward = self._reward_signal(backend_tick_record)
        info = StepInfo(
            realized_events=realized_events,
            evidence_ids=[
                i.evidence_id
                for i in self._evidence.items()
                if i.tick == self._tick - 1
            ],
            extra={
                "dilemmas_triggered": [d.dilemma_id for d in triggered],
                "backend_tick_record": safe_dataclass_to_dict(backend_tick_record),
                "world_evolution_records": world_evolution_records,
            },
        )
        return StepReturn(
            observation=snap,
            tool_results=tool_results,
            reward=float(reward),
            done=done,
            info=info,
        )

    @staticmethod
    def _agency_event_reveal_targets(event: dict[str, Any]) -> set[str]:
        targets = {
            str(event[key])
            for key in (
                "asset_id",
                "battery_id",
                "der_id",
                "genset_id",
                "load_id",
                "target_id",
            )
            if event.get(key) not in (None, "")
        }
        declared_target = (event.get("declared_event") or {}).get("target")
        if isinstance(declared_target, dict):
            targets.update(
                str(value)
                for value in declared_target.values()
                if isinstance(value, (str, int))
            )
        return targets

    def _agency_source_parent_for_call(
        self,
        *,
        call: ToolCall,
        request_tick: int,
        allow_same_tick_reveal: bool = False,
    ) -> dict[str, Any] | None:
        consumed = {str(value) for value in call.consumes_evidence_ids or []}
        if not consumed:
            return None
        eligible = [
            event
            for event in self._agency_source_events
            if int(event["applied_tick"]) < request_tick
            and (
                (not event["hidden"] and str(event["evidence_id"]) in consumed)
                or (
                    event["hidden"]
                    and event["revealed_tick"] is not None
                    and (
                        int(event["revealed_tick"]) < request_tick
                        or (
                            allow_same_tick_reveal
                            and int(event["revealed_tick"]) == request_tick
                        )
                    )
                    and bool(consumed.intersection(event["reveal_evidence_ids"]))
                )
            )
        ]
        return eligible[-1] if eligible else None

    def mark_agency_source_events_revealed(
        self,
        *,
        target_id: str,
        reveal_tick: int,
        evidence_id: str | None = None,
    ) -> None:
        for event in self._agency_source_events:
            if event["hidden"] and target_id in event["reveal_targets"]:
                if event["revealed_tick"] is None:
                    event["revealed_tick"] = int(reveal_tick)
                if evidence_id and evidence_id not in event["reveal_evidence_ids"]:
                    event["reveal_evidence_ids"].append(evidence_id)

    def execute_investigation(
        self, action: Action
    ) -> tuple[dict[str, Any], list[ToolResult]]:
        observation, results = super().execute_investigation(action)
        calls_by_id = {call.call_id: call for call in action.tool_calls if call.call_id}
        for result in results:
            call = calls_by_id.get(result.call_id)
            if (
                result.ok
                and result.evidence_id
                and call is not None
                and call.name == "investigate_asset"
            ):
                self.mark_agency_source_events_revealed(
                    target_id=str(call.args.get("asset_id", "")),
                    reveal_tick=self._tick,
                    evidence_id=str(result.evidence_id),
                )
        return observation, results

    def snapshot(self) -> dict[str, Any]:
        assert self._backend is not None
        raw = self._backend.snapshot()
        # EMS setpoints are revisable at every native control interval.
        # Declaring this here keeps the runner fail-closed for backends that
        # omit cadence while preserving Microgrid's real periodic control.
        raw.setdefault("decision_opportunity", True)
        raw.setdefault(
            "decision_cadence",
            {
                "mode": "native_control_interval",
                "native_opportunity": True,
            },
        )
        if self._fog:
            self._fog.set_tick(self._tick)
            raw = self._fog.filter(raw)
        backend_kind = str(getattr(self._backend, "backend_kind", "") or "")
        if backend_kind:
            with contextlib.suppress(KeyError):
                raw["decision_cadence"] = apply_supervisory_cadence(
                    backend_kind,
                    raw.get("decision_cadence"),
                )
        if self._stakeholders is not None and self._stakeholders.snapshot():
            raw["stakeholder_trust"] = {
                gid: {"trust": r.trust, "tier": r.tier}
                for gid, r in self._stakeholders.snapshot().items()
            }
        if self._dilemmas is not None:
            raw["active_dilemmas"] = [
                {
                    "dilemma_id": d.dilemma_id,
                    "description": d.description,
                    "options": [
                        {"option_id": o.option_id, "label": o.label, "fatal": o.fatal}
                        for o in d.options
                    ],
                    "deadline_tick": d.trigger_tick + d.resolution_deadline_ticks,
                }
                for d in self._dilemmas.record.dilemmas_triggered
                if d.dilemma_id not in self._dilemmas.record.choices
            ]
        return raw

    def ground_truth(self) -> dict[str, Any]:
        assert self._backend is not None and self._seed_obj is not None
        gt = self._backend.snapshot()
        gt["per_load_shed_mwh"] = self._backend.per_load_shed_mwh()
        gt["cost_components"] = self._backend.ground_truth_costs()
        if self._seed_obj.backend_config.get("task_requirements") and self._evidence:
            gt["control_summary"] = control_summary_from_evidence(self._evidence)
        # Private native records for phase-aware task completion. The runner
        # publishes only the compact ground_truth_summary, so these rows
        # remain evaluator-internal while still being replay-verifiable.
        gt["_task_tick_records"] = self._backend.scoring_records()
        if hasattr(self._backend, "applied_control_records"):
            gt["_task_control_records"] = self._backend.applied_control_records()
        if self._stakeholders is not None and self._stakeholders.snapshot():
            gt["stakeholder_trust"] = {
                gid: r.trust for gid, r in self._stakeholders.snapshot().items()
            }
        if self._dilemmas is not None:
            gt["dilemmas_triggered"] = [
                d.dilemma_id for d in self._dilemmas.record.dilemmas_triggered
            ]
            gt["choices"] = {
                did: {"option_id": c.chosen_option_id, "tick_chosen": c.tick_chosen}
                for did, c in self._dilemmas.record.choices.items()
            }
        return gt

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def budget(self) -> TickBudget:
        return self._tools._budget if self._tools else TickBudget(horizon=self._horizon)

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return self._tools.openai_schemas() if self._tools else []

    def close(self) -> None:
        pass

    # ── Public accessors (runner / scorer / tools) ──────────────────────

    @property
    def evidence(self) -> EvidenceLogger | None:
        return self._evidence

    @property
    def dilemmas(self) -> EthicalDilemmaManager | None:
        return self._dilemmas

    @property
    def stakeholders(self) -> StakeholderTrustManager | None:
        return self._stakeholders

    @property
    def cascade_bus(self) -> CascadeBus:
        return self._cascade_bus

    @property
    def seed_obj(self) -> MicrogridScenarioSeed | None:
        return self._seed_obj

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def _on_logistics_fuel_delay(self, event: CascadeEvent) -> None:
        """Cascade-bus subscriber: logistics fuel delivery delay -> microgrid DER failure.

        When a logistics fuel delivery is delayed, the microgrid's diesel
        genset may become unavailable due to fuel shortage. We enqueue a
        der_failure perturbation on the genset.
        """
        intensity = float(event.severity or 0.5)
        # Try to import Perturbation from the schema
        try:
            from .seeds.schema import Perturbation as MgPerturbation

            p = MgPerturbation(
                kind="der_failure",
                trigger_tick=self._tick + 1,
                duration_ticks=max(1, int(3 * intensity)),
                hidden=False,
                target={"der_index": 0},
                intensity=intensity,
                notes=(
                    f"Cascaded from logistics.fuel.delay "
                    f"(correlation_id={event.correlation_id})"
                ),
            )
            self._pending_cascade_perturbations.append(p)
        except Exception:
            pass

    def _on_traffic_signal_failure(self, event: CascadeEvent) -> None:
        """Cascade-bus subscriber: traffic disruption -> delayed genset fuel access."""
        fuel_event = CascadeEvent(
            event_type="logistics.fuel.delay",
            source_domain="traffic",
            tick=event.tick,
            severity=event.severity,
            location=event.location,
            payload={
                **dict(event.payload or {}),
                "cause": "traffic.signal.failure",
                "fuel_supply_style_direction": "traffic_to_microgrid",
            },
            correlation_id=event.correlation_id,
        )
        self._on_logistics_fuel_delay(fuel_event)

    # ── Internals ───────────────────────────────────────────────────────

    def _cost_summary_payload(self, rec: Any) -> dict[str, float]:
        return {
            "generation_cost": float(getattr(rec, "generation_cost", 0.0)),
            "grid_import_cost": float(getattr(rec, "grid_import_cost", 0.0)),
            "shed_penalty": float(getattr(rec, "shed_penalty", 0.0)),
            "battery_cycles": float(getattr(rec, "battery_cycles", 0.0)),
            "is_islanded": float(getattr(rec, "is_islanded", 0.0)),
        }

    def _reward_signal(self, rec: Any) -> float:
        prod = float(getattr(rec, "production_cost", 0.0))
        startup = float(getattr(rec, "startup_cost", 0.0))
        shed = float(getattr(rec, "shed_penalty", 0.0))
        balance = abs(float(getattr(rec, "balance_error_mw", 0.0)))
        return -(prod + startup + shed + balance * 200.0) / 1000.0

    def _publish_cascade_for_event(
        self, event: dict[str, Any], evidence_id: str
    ) -> None:
        canonical = _CASCADE_EVENT_MAPPING.get(str(event.get("type", "")))
        if canonical is None:
            return
        severity = float(event.get("intensity", 0.5) or 0.5)
        self._cascade_bus.publish(
            CascadeEvent(
                event_type=canonical,
                source_domain="microgrid",
                tick=int(event.get("tick", self._tick)),
                severity=max(0.0, min(1.0, severity)),
                location={
                    "site": (self._seed_obj.backend_config or {}).get("site")
                    if self._seed_obj
                    else None
                },
                payload={**event, "evidence_id": evidence_id},
                correlation_id=evidence_id,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factories / helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_backend(kind: str) -> Any:
    # Import backends at call time so collecting tests that only need the
    # adapter (or a pymgrid-family EMS backend) does not require pandapower.
    if kind == "pymgrid_islanding":
        from .backends.pymgrid_backend import PymgridIslandingBackend

        return PymgridIslandingBackend()
    if kind == "pymgrid_economic_dispatch":
        from .backends.pymgrid_backend import PymgridEconomicDispatchBackend

        return PymgridEconomicDispatchBackend()
    if kind == "pymgrid_solar_ramp":
        from .backends.pymgrid_backend import PymgridSolarRampBackend

        return PymgridSolarRampBackend()
    if kind == "pandapower_lv":
        from .backends.pandapower_lv import PandapowerLvBackend

        return PandapowerLvBackend()
    if kind == "ems_sim":
        return EmsSimulator()
    raise ValueError(f"unknown microgrid backend_kind: {kind}")


def _register_native_tools(
    reg: ToolRegistry, backend: Any, env: MicrogridEnvironment
) -> None:
    from .native_tools import register_microgrid_tools

    register_microgrid_tools(reg, backend, env)


def _register_native_stakeholders(
    mgr: StakeholderTrustManager, seed_obj: MicrogridScenarioSeed
) -> None:
    from .native_stakeholders import build_stakeholder_groups

    for group in build_stakeholder_groups(seed_obj):
        mgr.register(group)


def _build_tick_budget(seed_obj: MicrogridScenarioSeed) -> TickBudget:
    per_tick = {"basic": 6, "medium": 8, "high": 10, "extreme": 12}.get(
        canonical_difficulty_level(seed_obj.difficulty_level), 8
    )
    max_cost_units = max(1.0, per_tick * 0.5)
    return TickBudget(
        max_tool_calls_per_tick=per_tick,
        max_cost_units_per_tick=max_cost_units,
        max_total_tool_calls=per_tick * seed_obj.horizon_ticks,
        duplicate_suppression_window=2,
        cooldown_after_failure=1,
    )


def _rebuild_seed_from_dict(
    d: dict[str, Any], override_seed: int
) -> MicrogridScenarioSeed:
    perturbations = [Perturbation(**p) for p in d.get("perturbations", [])]
    load_assignments = [MicrogridLoad(**c) for c in d.get("load_assignments", [])]
    dilemmas = [DilemmaSeed(**ds) for ds in d.get("dilemmas", [])]
    provenance = Provenance(**d.get("provenance", {"data_source": "unspecified"}))
    return MicrogridScenarioSeed(
        seed_id=str(d.get("seed_id", "anon")),
        family=str(d.get("family", "microgrid_islanding_24h")),
        domain=str(d.get("domain", "microgrid")),
        backend_kind=str(d.get("backend_kind", "pymgrid_islanding")),
        backend_config=dict(d.get("backend_config", {})),
        horizon_ticks=int(d.get("horizon_ticks", 24)),
        tick_minutes=int(d.get("tick_minutes", 60)),
        seed=int(override_seed),
        load_assignments=load_assignments,
        perturbations=perturbations,
        dilemmas=dilemmas,
        difficulty_mode=d.get("difficulty_mode", "time_pressure"),
        difficulty_level=d.get("difficulty_level", "basic"),
        provenance=provenance,
    )
