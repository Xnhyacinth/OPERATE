"""
domains.traffic.adapter — Traffic-control POMDP environment.

Mirrors ``domains.disaster.adapter`` / ``domains.power_grid.adapter``
line-for-line — backend + fog + belief + tool registry + stakeholder manager +
dilemma manager + evidence logger + cascade-bus subscriber — but uses
traffic-native types and never imports from another ``domains.*`` package
(Red Line #3, ``.hl/policy.md``).

The default backend is ``MockSumoBackend`` (pure-Python, deterministic). The
seed's ``backend_kind == "sumo"`` switches to the real ``SumoBackend`` (which
drives SUMO through ``core.sidecar.sumo_sidecar`` and gracefully skips when no
SUMO transport is installed).

Cross-domain coupling: subscribes to ``power_grid.line.outage`` on the cascade
bus and injects a ``signal_failure`` perturbation on EMS / high-criticality
corridors, so a grid fault that de-energizes traffic signals propagates to a
traffic-side signal outage. Realized traffic shocks are published back onto the
bus as ``traffic.*`` events for other domains to consume.
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
    HideRule,
    NoiseRule,
    POMDPEnvironment,
    StakeholderTrustManager,
    StalenessRule,
    StepInfo,
    StepReturn,
    TickBudget,
    ToolContext,
    ToolRegistry,
    arm_dilemmas,
    safe_dataclass_to_dict,
)
from core.evidence import control_summary_from_evidence
from core.world_evolution_contract import canonicalize_runtime_events
from domains.registry import apply_supervisory_cadence

from .backends.mock_sumo import MockSumoBackend
from .oracle import compute_reference_optimum
from .seeds.schema import (
    CorridorAssignment,
    Provenance,
    TrafficDilemmaSeed,
    TrafficPerturbation,
    TrafficScenarioSeed,
)

try:  # real backend is optional — only present once the sidecar stub lands
    from .backends.sumo_backend import SumoBackend  # type: ignore
except Exception:  # pragma: no cover - real backend not yet implemented
    SumoBackend = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Event-name mapping (backend → canonical cascade event_type)
# ─────────────────────────────────────────────────────────────────────────────


_CASCADE_EVENT_MAPPING: dict[str, str] = {
    "incident": "traffic.incident.blocking",
    "lane_blockage": "traffic.lane.blocked",
    "signal_failure": "traffic.signal.failure",
    "demand_surge": "traffic.demand.surge",
    "weather_capacity_drop": "traffic.weather.capacity_drop",
    "detector_dropout": "traffic.detector.dropout",
    "vip_arrival": "traffic.vip.arrival",
    "ems_corridor_request": "traffic.ems.corridor_request",
    "relief_crew_arrived": "traffic.aid.arrived",
}

_ACTIONABLE_SOURCE_EVENT_CLASSES = frozenset({"alarm", "forecast", "safety", "task"})


def _apply_typed_source_event_registry(
    events: list[dict[str, Any]],
    *,
    registry: dict[str, Any] | None,
    current_tick: int,
    horizon: int,
) -> list[dict[str, Any]]:
    """Apply scenario-declared decision semantics to source-schedule events.

    Native SUMO flow telemetry occurs at most ticks.  Only a source transition
    pre-registered by the scenario contract may wake the supervisor; an
    unknown or malformed declaration remains visible telemetry.
    """

    declarations = registry if isinstance(registry, dict) else {}
    resolved: list[dict[str, Any]] = []
    for raw in events:
        event = dict(raw)
        if str(event.get("origin") or "") != "source_schedule":
            resolved.append(event)
            continue
        event_type = str(event.get("type") or event.get("kind") or "")
        declaration = declarations.get(event_type)
        if not isinstance(declaration, dict):
            event.update(
                {
                    "event_class": "telemetry",
                    "actionable": False,
                    "decision_required": False,
                    "response_window_required": False,
                }
            )
            resolved.append(event)
            continue

        event_class = str(declaration.get("event_class") or "")
        actionable_ticks = {
            int(value)
            for value in declaration.get("actionable_ticks") or []
            if isinstance(value, int) and not isinstance(value, bool)
        }
        try:
            threshold = float(declaration.get("materiality_threshold"))
        except (TypeError, ValueError):
            threshold = float("inf")
        raw_value = event.get("materiality_value")
        if raw_value is None:
            raw_value = int(event.get("interval_arrived") or 0) + int(
                event.get("interval_departed") or 0
            )
        try:
            materiality_value = float(raw_value)
        except (TypeError, ValueError):
            materiality_value = 0.0
        response_window_ticks = declaration.get("response_window_ticks")
        valid_window = (
            isinstance(response_window_ticks, int)
            and not isinstance(response_window_ticks, bool)
            and response_window_ticks > 0
            and current_tick + 1 < horizon
        )
        actionable = bool(
            event_class in _ACTIONABLE_SOURCE_EVENT_CLASSES
            and current_tick in actionable_ticks
            and event.get("hidden") is not True
            and materiality_value >= threshold
            and valid_window
        )
        event.update(
            {
                "event_class": event_class if actionable else "telemetry",
                "actionable": actionable,
                "decision_required": actionable,
                "materiality_metric": declaration.get("materiality_metric"),
                "materiality_value": materiality_value,
                "materiality_threshold": threshold,
                "materiality_passed": materiality_value >= threshold,
                "response_window_required": actionable,
                "response_opportunity_tick": (
                    current_tick + 1 if actionable else None
                ),
                "response_deadline_tick": (
                    min(horizon - 1, current_tick + response_window_ticks)
                    if actionable
                    else None
                ),
                "terminal_response_window_missing": bool(
                    current_tick in actionable_ticks and not valid_window
                ),
                "source_event_registry_version": "traffic-source-events/1.0",
            }
        )
        resolved.append(event)
    return resolved


def _state_digest(value: dict[str, Any]) -> str:
    """Return a deterministic digest for a bounded native-state observation."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _post_tick_native_state(
    backend_tick_payload: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract the native SUMO state observed after a supervisory action.

    The phase-duration tool changes TLS state before the backend advances.
    This bounded snapshot captures the state after SUMO has completed its
    native physics substeps, so an acknowledgement alone cannot be used as
    evidence of an action effect.
    """
    events = backend_tick_payload.get("realized_events") or []
    if not isinstance(events, list):
        return None
    snapshot = next(
        (
            row
            for row in events
            if isinstance(row, dict) and row.get("type") == "sumo_live_snapshot"
        ),
        None,
    )
    if snapshot is None:
        return None
    return {
        key: snapshot[key]
        for key in (
            "tick",
            "n_vehicles",
            "arrived",
            "departed",
            "per_corridor",
            "native_physics_step_count",
            "decision_interval_seconds",
            "runtime_signal_control",
        )
        if key in snapshot
    }


def _authoritative_source_event(
    event: dict[str, Any],
    evidence_id: str,
    visible_source_events_by_evidence_id: dict[str, dict[str, Any]],
) -> None:
    """Attach logger evidence and retain only passively visible source events."""

    if str(event.get("origin") or "") == "agent_caused":
        return
    event.setdefault("event_id", f"traffic-source-event:{evidence_id}")
    evidence_ids = event.setdefault("evidence_ids", [])
    if evidence_id not in evidence_ids:
        evidence_ids.append(evidence_id)
    if event.get("hidden") is True:
        return
    if event.get("actionable") is not True or event.get("decision_required") is not True:
        return
    visible_source_events_by_evidence_id[evidence_id] = {
        "event_id": str(event["event_id"]),
        "visible_from_request_tick": int(event.get("tick") or 0) + 1,
    }


def _visible_causal_parent_event_id(
    action: Action,
    call_id: str | None,
    request_tick: int,
    visible_source_events_by_evidence_id: dict[str, dict[str, Any]],
) -> str | None:
    """Resolve an explicitly consumed, already-visible source event."""

    call = next(
        (candidate for candidate in action.tool_calls if candidate.call_id == call_id),
        None,
    )
    if call is None:
        return None
    for evidence_id in call.consumes_evidence_ids or []:
        event = visible_source_events_by_evidence_id.get(str(evidence_id))
        if event is not None and int(event["visible_from_request_tick"]) <= int(
            request_tick
        ):
            return str(event["event_id"])
    return None


def _native_signal_action_effect_events(
    *,
    action: Action,
    tool_results: list[Any],
    backend_tick_payload: dict[str, Any],
    applied_tick: int,
    visible_source_events_by_evidence_id: dict[str, dict[str, Any]] | None = None,
    pending_signal_program_calls: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Emit evidence-linked live-SUMO signal effects after native physics."""
    post_tick_state = _post_tick_native_state(backend_tick_payload)
    visible_source_events_by_evidence_id = visible_source_events_by_evidence_id or {}
    program_requests = (
        pending_signal_program_calls if pending_signal_program_calls is not None else {}
    )
    calls_by_id = {
        call.call_id: call for call in action.tool_calls if call.call_id is not None
    }
    effects: list[dict[str, Any]] = []
    for result in tool_results:
        if getattr(result, "name", None) == "set_signal_program":
            call_id = str(getattr(result, "call_id", None) or "")
            if not getattr(result, "ok", False):
                for key, pending in list(program_requests.items()):
                    if pending.get("call_id") == call_id:
                        program_requests.pop(key, None)
                continue
            program_payload = (
                result.payload
                if isinstance(getattr(result, "payload", None), dict)
                else {}
            )
            program_call = calls_by_id.get(call_id)
            tls_id = str(program_payload.get("tls_id") or "")
            program_id = str(program_payload.get("program_id") or "")
            if program_call is not None:
                tls_id = tls_id or str(program_call.args.get("tls_id") or "")
                program_id = program_id or str(
                    program_call.args.get("program_id") or ""
                )
            key = (tls_id, program_id)
            pending = program_requests.get(key)
            if pending is None and program_call is not None and all(key):
                pending = {
                    "call_id": call_id,
                    "tool_name": "set_signal_program",
                    "requested_action": {
                        "name": program_call.name,
                        "args": dict(program_call.args),
                    },
                    "request_tick": applied_tick,
                    "payload": {},
                    "evidence_ids": [],
                    "causal_parent_event_id": _visible_causal_parent_event_id(
                        action,
                        call_id,
                        applied_tick,
                        visible_source_events_by_evidence_id,
                    ),
                }
                program_requests[key] = pending
            if pending is not None and pending.get("call_id") == call_id:
                pending["payload"] = {
                    **pending.get("payload", {}),
                    **program_payload,
                }
                evidence_ids = pending["evidence_ids"]
                for evidence_id in [
                    getattr(result, "evidence_id", None),
                    *(program_payload.get("evidence_ids") or []),
                ]:
                    if (
                        isinstance(evidence_id, str)
                        and evidence_id
                        and evidence_id not in evidence_ids
                    ):
                        evidence_ids.append(evidence_id)
        if (
            getattr(result, "name", None) != "set_signal_phase_duration"
            or not getattr(result, "ok", False)
            or not getattr(result, "state_changing", False)
        ):
            continue
        payload = getattr(result, "payload", None)
        call_id = getattr(result, "call_id", None)
        call = calls_by_id.get(call_id)
        if (
            not isinstance(payload, dict)
            or payload.get("sumo_state_mutated") is not True
            or call is None
            or post_tick_state is None
        ):
            continue
        before = payload.get("before_runtime_state")
        after = payload.get("after_runtime_state")
        tls_id = str(payload.get("sumo_tls_id") or call.args.get("tls_id") or "")
        post_tick_tls_state = (
            post_tick_state.get("runtime_signal_control", {}).get(tls_id)
            if isinstance(post_tick_state.get("runtime_signal_control"), dict)
            else None
        )
        source_identity = str(payload.get("complete_source_identity_sha256") or "")
        if (
            not isinstance(before, dict)
            or not isinstance(after, dict)
            or not isinstance(post_tick_tls_state, dict)
            or not source_identity
            or _state_digest(
                {
                    key: before.get(key)
                    for key in (
                        "current_program",
                        "current_phase",
                        "current_state",
                        "remaining_duration",
                    )
                }
            )
            == _state_digest(
                {
                    key: post_tick_tls_state.get(key)
                    for key in (
                        "current_program",
                        "current_phase",
                        "current_state",
                        "remaining_duration",
                    )
                }
            )
        ):
            continue
        event_id = (
            f"sumo-agent-phase-duration:{source_identity}:{call_id}:{applied_tick}"
        )
        evidence_ids: list[str] = []
        for evidence_id in [
            getattr(result, "evidence_id", None),
            *(payload.get("evidence_ids") or []),
        ]:
            if (
                isinstance(evidence_id, str)
                and evidence_id
                and evidence_id not in evidence_ids
            ):
                evidence_ids.append(evidence_id)
        applied_action = {
            "sumo_tls_id": payload.get("sumo_tls_id"),
            "sumo_phase_duration_s": payload.get("sumo_phase_duration_s"),
            "runtime_state_after_action": after,
            "post_tick_tls_state": post_tick_tls_state,
            "post_tick_native_state": post_tick_state,
        }
        causal_parent_event_id = _visible_causal_parent_event_id(
            action,
            call_id,
            applied_tick,
            visible_source_events_by_evidence_id,
        )
        effects.append(
            {
                "event_id": event_id,
                "type": "traffic_signal_phase_duration_applied",
                "origin": "agent_caused",
                "agent_caused": True,
                "tick": applied_tick,
                "actionable": False,
                "decision_required": False,
                "changed_state_fields": [
                    "tls_remaining_phase_duration",
                    "tls_runtime_state",
                ],
                "call_id": call_id,
                "tool_name": result.name,
                "requested_action": {
                    "name": call.name,
                    "args": dict(call.args),
                },
                "applied_action": applied_action,
                "before_state_digest": _state_digest(
                    {
                        key: before.get(key)
                        for key in (
                            "current_program",
                            "current_phase",
                            "current_state",
                            "remaining_duration",
                        )
                    }
                ),
                "after_state_digest": _state_digest(
                    {
                        key: post_tick_tls_state.get(key)
                        for key in (
                            "current_program",
                            "current_phase",
                            "current_state",
                            "remaining_duration",
                        )
                    }
                ),
                "outcome_tick": applied_tick + 1,
                "evidence_ids": evidence_ids,
                **(
                    {"causal_parent_event_id": causal_parent_event_id}
                    if causal_parent_event_id
                    else {}
                ),
                "action_to_outcome_edge": {
                    "source": f"call:{call_id}",
                    "target": f"outcome:{event_id}",
                    "kind": "action_to_outcome",
                },
            }
        )
    if post_tick_state is None:
        return effects
    live_snapshot = next(
        (
            row
            for row in backend_tick_payload.get("realized_events") or []
            if isinstance(row, dict) and row.get("type") == "sumo_live_snapshot"
        ),
        {},
    )
    materialized_controls = live_snapshot.get("materialized_signal_controls") or []
    for materialized in materialized_controls:
        if not isinstance(materialized, dict):
            continue
        tls_id = str(materialized.get("tls_id") or "")
        program_id = str(materialized.get("program_id") or "")
        request = program_requests.get((tls_id, program_id))
        if request is None:
            continue
        payload = request.get("payload", {})
        effect_tick = int(materialized.get("applied_at_tick", applied_tick))
        before = {
            "tls_id": tls_id,
            "current_program": materialized.get(
                "prior_program", payload.get("prior_program")
            ),
            "current_phase": materialized.get(
                "prior_phase", payload.get("prior_phase")
            ),
            "current_state": materialized.get(
                "prior_state", payload.get("prior_state")
            ),
        }
        after = materialized.get("resulting_runtime_state")
        source_identity = str(payload.get("complete_source_identity_sha256") or "")
        if (
            materialized.get("sumo_state_mutated") is not True
            or materialized.get("sumo_program_readback") != program_id
            or not isinstance(after, dict)
            or after.get("current_program") != program_id
            or not source_identity
            or effect_tick != applied_tick
            or effect_tick < int(request.get("request_tick", applied_tick))
            or _state_digest(before) == _state_digest(after)
        ):
            continue
        evidence_ids: list[str] = list(request.get("evidence_ids") or [])
        for evidence_id in [
            *(materialized.get("evidence_ids") or []),
        ]:
            if (
                isinstance(evidence_id, str)
                and evidence_id
                and evidence_id not in evidence_ids
            ):
                evidence_ids.append(evidence_id)
        call_id = str(request["call_id"])
        event_id = f"sumo-agent-program:{source_identity}:{call_id}:{effect_tick}"
        effects.append(
            {
                "event_id": event_id,
                "type": "traffic_signal_program_applied",
                "origin": "agent_caused",
                "agent_caused": True,
                "tick": effect_tick,
                "actionable": False,
                "decision_required": False,
                "changed_state_fields": [
                    "tls_current_program",
                    "tls_current_phase",
                    "tls_current_state",
                ],
                "call_id": call_id,
                "tool_name": str(request["tool_name"]),
                "requested_action": dict(request["requested_action"]),
                "applied_action": {
                    "tls_id": tls_id,
                    "program_id": program_id,
                    "resulting_runtime_state": after,
                    "post_tick_native_state": post_tick_state,
                },
                "before_state_digest": _state_digest(before),
                "after_state_digest": _state_digest(after),
                "outcome_tick": effect_tick + 1,
                "evidence_ids": evidence_ids,
                **(
                    {"causal_parent_event_id": request["causal_parent_event_id"]}
                    if request.get("causal_parent_event_id")
                    else {}
                ),
                "action_to_outcome_edge": {
                    "source": f"call:{call_id}",
                    "target": f"outcome:{event_id}",
                    "kind": "action_to_outcome",
                },
            }
        )
        program_requests.pop((tls_id, program_id), None)
    return effects


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────


class TrafficEnvironment(POMDPEnvironment):
    """OPERATE environment for the traffic-control domain.

    Constructor takes an optional ``cascade_bus`` so a cross-domain run can
    share a single bus across PowerGridEnvironment + TrafficEnvironment.
    """

    domain = "traffic"

    def __init__(self, cascade_bus: CascadeBus | None = None) -> None:
        self._seed_obj: TrafficScenarioSeed | None = None
        self._tick: int = 0
        self._horizon: int = 24
        self._backend: Any = None
        self._fog: FogOfWarPolicy | None = None
        self._belief: BeliefStateTracker | None = None
        self._tools: ToolRegistry | None = None
        self._stakeholders: StakeholderTrustManager | None = None
        self._dilemmas: EthicalDilemmaManager | None = None
        self._evidence: EvidenceLogger | None = None
        self._cascade_bus = cascade_bus or CascadeBus()
        self._episode_id: str = ""
        self._visible_source_events_by_evidence_id: dict[str, dict[str, Any]] = {}
        self._pending_signal_program_calls: dict[tuple[str, str], dict[str, Any]] = {}
        # Pending grid→traffic cascade injections spliced in at next tick().
        self._pending_cascade_perturbations: list[TrafficPerturbation] = []
        self._cascade_bus.subscribe(
            "power_grid.line.outage", self._on_power_grid_line_outage
        )

    # ── POMDPEnvironment surface ────────────────────────────────────────

    def reset(self, scenario_config: dict[str, Any], seed: int) -> dict[str, Any]:
        seed_obj = _rebuild_seed_from_dict(scenario_config, override_seed=seed)
        self._seed_obj = seed_obj
        self._tick = 0
        self._horizon = seed_obj.horizon_ticks
        self._episode_id = f"dt_{seed_obj.signature()}_s{seed}"
        self._visible_source_events_by_evidence_id = {}
        self._pending_signal_program_calls = {}

        kind = str(seed_obj.backend_kind or "mock_sumo")
        self._backend = _build_backend(kind, seed_obj.backend_config)
        try:
            self._backend.reset(seed_obj)

            # The Wardrop reference models MockSumoBackend's aggregate corridor
            # equations. It is not an aligned optimum for a native SUMO network.
            if kind == "mock_sumo":
                with contextlib.suppress(Exception):
                    compute_reference_optimum(seed_obj, write_back=True)

            self._fog = _build_fog(seed_obj)
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
        except BaseException:
            self.close()
            raise

    def step(self, action: Action) -> StepReturn:
        assert self._tools is not None and self._backend is not None
        assert self._fog is not None and self._evidence is not None
        assert self._dilemmas is not None and self._stakeholders is not None
        assert self._belief is not None

        # 1) Execute tool calls (mutates backend via tool handlers).
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
        tool_results = self._tools.execute_action(
            action, ctx, begin_tick=not self._consume_within_tick_budget_state()
        )
        calls_by_id = {
            str(call.call_id): call
            for call in action.tool_calls
            if call.call_id is not None
        }
        for r in tool_results:
            call = calls_by_id.get(str(r.call_id or ""))
            r.evidence_id = self._evidence.log(
                kind="tool_call",
                tick=self._tick,
                payload={
                    "name": r.name,
                    "ok": r.ok,
                    "error_code": r.error_code,
                    "cost_units": r.cost_units,
                    "call_id": r.call_id,
                    "state_changing": r.state_changing,
                    "args": dict(call.args) if call is not None else {},
                    "payload": r.payload,
                    **self.tool_dependency_payload(action, r),
                },
                source="tool",
            )

        # 2) Splice any pending cascade-bus perturbations onto the seed BEFORE
        # the backend ticks so it sees them this tick.
        if self._pending_cascade_perturbations and self._seed_obj is not None:
            self._seed_obj.perturbations.extend(self._pending_cascade_perturbations)
            self._pending_cascade_perturbations = []

        # 3) Advance backend.
        backend_tick_record = self._backend.tick(self._tick)
        backend_tick_payload = safe_dataclass_to_dict(backend_tick_record)
        if not isinstance(backend_tick_payload, dict):
            backend_tick_payload = {}
        backend_events = _apply_typed_source_event_registry(
            list(getattr(backend_tick_record, "realized_events", []) or []),
            registry=(
                self._seed_obj.backend_config.get("source_event_registry")
                if self._seed_obj is not None
                else None
            ),
            current_tick=self._tick,
            horizon=self._horizon,
        )
        native_action_effect_events = _native_signal_action_effect_events(
            action=action,
            tool_results=tool_results,
            backend_tick_payload=backend_tick_payload,
            applied_tick=self._tick,
            visible_source_events_by_evidence_id=(
                self._visible_source_events_by_evidence_id
            ),
            pending_signal_program_calls=self._pending_signal_program_calls,
        )
        realized_events = [*backend_events, *native_action_effect_events]
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
        for ev in realized_events:
            evidence_id = self._evidence.log(
                kind="realized_event",
                tick=self._tick,
                payload=dict(ev),
                source="engine",
            )
            _authoritative_source_event(
                ev,
                evidence_id,
                self._visible_source_events_by_evidence_id,
            )
            self._publish_cascade_for_event(ev, evidence_id)
        world_evolution_records = canonicalize_runtime_events(
            realized_events,
            applied_tick=self._tick,
        )

        # 4) Dilemma triggers.
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

        # 5) Stakeholder natural drift.
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
                "stakeholder_trust": {
                    gid: r.trust for gid, r in self._stakeholders.snapshot().items()
                },
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

    def snapshot(self) -> dict[str, Any]:
        assert self._backend is not None
        raw = self._backend.snapshot()
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
        if self._stakeholders:
            raw["stakeholder_trust"] = {
                gid: {"trust": r.trust, "tier": r.tier}
                for gid, r in self._stakeholders.snapshot().items()
            }
        if self._dilemmas:
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
        """Full state, including hidden perturbations and per-corridor delay."""
        assert self._backend is not None and self._seed_obj is not None
        gt = self._backend.snapshot()
        gt["per_corridor_delay_minutes"] = self._backend.per_corridor_delay_minutes()
        gt["cost_components"] = self._backend.ground_truth_costs()
        if self._seed_obj.backend_config.get("task_requirements") and self._evidence:
            gt["control_summary"] = control_summary_from_evidence(
                self._evidence,
                include_lifecycle=True,
            )
        if self._stakeholders:
            gt["stakeholder_trust"] = {
                gid: r.trust for gid, r in self._stakeholders.snapshot().items()
            }
            gt["stakeholder_equity_gini"] = round(self._stakeholders.equity_gini(), 4)
        if self._dilemmas:
            gt["dilemmas_triggered"] = [
                d.dilemma_id for d in self._dilemmas.record.dilemmas_triggered
            ]
            gt["choices"] = {
                did: {"option_id": c.chosen_option_id, "tick_chosen": c.tick_chosen}
                for did, c in self._dilemmas.record.choices.items()
            }
            gt["chose_fatal_option"] = self._dilemmas.chose_fatal_option()
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
        if not self._tools:
            return []
        return self._tools.openai_schemas()

    def close(self) -> None:
        # Mock backend has no external resources; the real SUMO backend closes
        # its sidecar transport here.
        if self._backend is not None and hasattr(self._backend, "close"):
            with contextlib.suppress(Exception):
                self._backend.close()

    # ── Public accessors used by the runner/scorer ──────────────────────

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
    def seed_obj(self) -> TrafficScenarioSeed | None:
        return self._seed_obj

    @property
    def episode_id(self) -> str:
        return self._episode_id

    # ── Internals ───────────────────────────────────────────────────────

    def _cost_summary_payload(self, backend_tick_record: Any) -> dict[str, float]:
        return {
            "total_delay_minutes": float(
                getattr(backend_tick_record, "total_delay_minutes", 0.0)
            ),
            "queue_length": float(getattr(backend_tick_record, "queue_length", 0.0)),
            "signal_violations": float(
                getattr(backend_tick_record, "signal_violations", 0.0)
            ),
            "unserved_vehicles": float(
                getattr(backend_tick_record, "unserved_vehicles", 0.0)
            ),
        }

    def _reward_signal(self, backend_tick_record: Any) -> float:
        """Per-tick reward (negative cost). Informative only."""
        travel = float(getattr(backend_tick_record, "travel_cost_this_tick", 0.0))
        shed = float(getattr(backend_tick_record, "shed_cost_this_tick", 0.0))
        return -(travel + shed) / 1000.0

    def _publish_cascade_for_event(
        self, event: dict[str, Any], evidence_id: str
    ) -> None:
        """Map a backend-emitted event onto a canonical cascade event_type
        and publish it. Mirrors the power-grid / disaster adapter helper."""
        type_str = str(event.get("type", ""))
        canonical = _CASCADE_EVENT_MAPPING.get(type_str)
        if canonical is None:
            return
        severity = float(event.get("intensity", 0.5) or 0.5)
        self._cascade_bus.publish(
            CascadeEvent(
                event_type=canonical,
                source_domain="traffic",
                tick=int(event.get("tick", self._tick)),
                severity=max(0.0, min(1.0, severity)),
                location={"corridor": event.get("corridor") or event.get("edge")},
                payload={**event, "evidence_id": evidence_id},
                correlation_id=evidence_id,
            )
        )

    def _on_power_grid_line_outage(self, event: CascadeEvent) -> None:
        """Cascade-bus subscriber: grid line outage → traffic signal failure.

        A de-energized substation drops the traffic signals it powers. We
        enqueue a ``signal_failure`` perturbation (next tick, 6-tick duration)
        on EMS / high-criticality corridors so the dark-signal shock flows
        through the normal evidence + cascade path rather than mutating backend
        state directly. Falls back to the single most-critical corridor when no
        corridor is flagged EMS / high-criticality.
        """
        if self._seed_obj is None:
            return
        trigger_tick = max(self._tick, int(event.tick) + 1)
        targets = [
            c
            for c in self._seed_obj.corridors
            if c.carries_ems_corridor or c.criticality >= 0.7
        ]
        if not targets and self._seed_obj.corridors:
            targets = [max(self._seed_obj.corridors, key=lambda c: c.criticality)]
        for c in targets:
            self._pending_cascade_perturbations.append(
                TrafficPerturbation(
                    kind="signal_failure",
                    trigger_tick=trigger_tick,
                    duration_ticks=6,
                    hidden=False,
                    target={
                        "corridor": c.corridor_id,
                        "cause": "cascaded_from_power_grid.line.outage",
                    },
                    intensity=float(event.severity or 0.5),
                    notes=(
                        f"Cascaded from power_grid.line.outage "
                        f"(correlation_id={event.correlation_id})"
                    ),
                )
            )


# ─────────────────────────────────────────────────────────────────────────────
# Backend factory
# ─────────────────────────────────────────────────────────────────────────────


def _build_backend(kind: str, cfg: dict[str, Any]) -> Any:
    if kind == "mock_sumo":
        return MockSumoBackend()
    if kind == "sumo":
        if SumoBackend is None:
            raise ValueError(
                "traffic backend_kind='sumo' requested but SumoBackend is not "
                "available (real sidecar stub not installed)."
            )
        return SumoBackend(cfg)
    raise ValueError(f"unknown traffic backend_kind: {kind}")


# ─────────────────────────────────────────────────────────────────────────────
# Fog policy from seed
# ─────────────────────────────────────────────────────────────────────────────


def _build_fog(seed_obj: TrafficScenarioSeed) -> FogOfWarPolicy:
    """Hide corridor congestion under hidden shocks + add detector noise.

    - Corridors under a hidden ``incident`` / ``lane_blockage`` /
      ``signal_failure`` / ``detector_dropout`` perturbation hide their
      ``queue`` and ``delay_minutes`` until investigated.
    - Baseline detector uncertainty: ``queue`` ±8% noise, ``delay_minutes``
      ±5% noise (loop-detector imprecision).
    - ``detector_dropout`` perturbations add a staleness rule on ``queue`` so
      the reading lags until the detector recovers.
    """
    hide_rules: list[HideRule] = []
    noise_rules: list[NoiseRule] = [
        NoiseRule(entity_kind="corridor", attr="queue", sigma_rel=0.08),
        NoiseRule(entity_kind="corridor", attr="delay_minutes", sigma_rel=0.05),
    ]
    staleness_rules: list[StalenessRule] = []

    hidden_kinds = {
        "incident",
        "lane_blockage",
        "signal_failure",
        "detector_dropout",
        "weather_capacity_drop",
    }
    has_hidden_shock = any(
        p.kind in hidden_kinds and p.hidden for p in seed_obj.perturbations
    )
    if has_hidden_shock:
        hide_rules.append(
            HideRule(
                entity_kind="corridor",
                hidden_attrs=["queue", "delay_minutes"],
                reveal_on=[
                    "query_network_state",
                    "query_detector",
                    "inspect_intersection",
                    "query_signal_control",
                ],
            )
        )

    has_dropout = any(p.kind == "detector_dropout" for p in seed_obj.perturbations)
    if has_dropout:
        staleness_rules.append(
            StalenessRule(entity_kind="corridor", attr="queue", staleness_ticks=2)
        )

    return FogOfWarPolicy(
        hide_rules=hide_rules,
        noise_rules=noise_rules,
        staleness_rules=staleness_rules,
        seed=seed_obj.seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stakeholders + dilemmas (delegate)
# ─────────────────────────────────────────────────────────────────────────────


def _register_native_stakeholders(
    mgr: StakeholderTrustManager, seed_obj: TrafficScenarioSeed
) -> None:
    from .native_stakeholders import build_stakeholder_groups

    for group in build_stakeholder_groups(seed_obj):
        mgr.register(group)


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration (delegate)
# ─────────────────────────────────────────────────────────────────────────────


def _register_native_tools(
    reg: ToolRegistry, backend: Any, env: TrafficEnvironment
) -> None:
    from .native_tools import register_traffic_tools

    register_traffic_tools(reg, backend, env)


# ─────────────────────────────────────────────────────────────────────────────
# Tick budget from difficulty
# ─────────────────────────────────────────────────────────────────────────────


def _build_tick_budget(seed_obj: TrafficScenarioSeed) -> TickBudget:
    per_tick = {"basic": 6, "medium": 8, "high": 10, "extreme": 12}.get(
        seed_obj.difficulty_level, 8
    )
    total = per_tick * seed_obj.horizon_ticks
    max_cost_units = max(1.0, per_tick * 0.5)
    return TickBudget(
        max_tool_calls_per_tick=per_tick,
        max_cost_units_per_tick=max_cost_units,
        max_total_tool_calls=total,
        duplicate_suppression_window=2,
        cooldown_after_failure=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rebuild a TrafficScenarioSeed from its dict serialization
# ─────────────────────────────────────────────────────────────────────────────


def _rebuild_seed_from_dict(
    d: dict[str, Any], override_seed: int
) -> TrafficScenarioSeed:
    corridors = [CorridorAssignment(**c) for c in d.get("corridors", [])]
    perturbations = [TrafficPerturbation(**p) for p in d.get("perturbations", [])]
    dilemmas = [TrafficDilemmaSeed(**ds) for ds in d.get("dilemmas", [])]
    provenance = Provenance(**d.get("provenance", {"data_source": "unspecified"}))
    return TrafficScenarioSeed(
        seed_id=str(d.get("seed_id", "anon")),
        family=str(d.get("family", "daily_peak_commute")),
        domain=str(d.get("domain", "traffic")),
        backend_kind=str(d.get("backend_kind", "mock_sumo")),
        backend_config=dict(d.get("backend_config", {})),
        horizon_ticks=int(d.get("horizon_ticks", 24)),
        tick_minutes=int(d.get("tick_minutes", 5)),
        seed=int(override_seed),
        net_ref=d.get("net_ref"),
        route_ref=d.get("route_ref"),
        sumo_mode=d.get("sumo_mode", "micro"),
        corridors=corridors,
        perturbations=perturbations,
        dilemmas=dilemmas,
        incident_edge=d.get("incident_edge"),
        incident_edge_betweenness=float(d.get("incident_edge_betweenness", 0.0)),
        hidden_attr_parity=int(d.get("hidden_attr_parity", 0)),
        demand_window_offset_min=int(d.get("demand_window_offset_min", 0)),
        difficulty_mode=d.get("difficulty_mode", "time_pressure"),
        difficulty_level=d.get("difficulty_level", "basic"),
        provenance=provenance,
    )
