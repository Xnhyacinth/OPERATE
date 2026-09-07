from __future__ import annotations

from core import (
    Action,
    EvidenceLogger,
    TickBudget,
    ToolCall,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from core.evidence import control_summary_from_evidence
from core.world_evolution_contract import canonicalize_runtime_events
from domains.traffic.adapter import (
    _apply_typed_source_event_registry,
    _authoritative_source_event,
    _native_signal_action_effect_events,
)


def _backend_tick_payload() -> dict[str, object]:
    return {
        "aggregate_queue": 17.0,
        "aggregate_delay_minutes": 8.5,
        "realized_events": [
            {
                "type": "sumo_live_snapshot",
                "tick": 0,
                "n_vehicles": 17,
                "arrived": 4,
                "departed": 2,
                "per_corridor": {"corridor-a": {"queue": 3.0}},
                "runtime_signal_control": {
                    "tls-a": {
                        "current_program": "0",
                        "current_phase": 3,
                        "remaining_duration": 11.0,
                    }
                },
            }
        ],
    }


def test_source_registry_makes_only_pre_registered_flow_transition_actionable() -> (
    None
):
    registry = {
        "traffic_demand_change": {
            "event_class": "task",
            "actionable_ticks": [3],
            "materiality_metric": "interval_vehicle_flow",
            "materiality_threshold": 1,
            "response_window_ticks": 4,
        }
    }
    source_event = {
        "event_id": "flow@3",
        "type": "traffic_demand_change",
        "origin": "source_schedule",
        "tick": 3,
        "interval_arrived": 4,
        "interval_departed": 2,
        "changed_state_fields": ["n_vehicles", "controlled_lane_queues"],
        "materiality_value": 6,
    }

    [typed] = _apply_typed_source_event_registry(
        [source_event],
        registry=registry,
        current_tick=3,
        horizon=12,
    )
    [routine] = _apply_typed_source_event_registry(
        [{**source_event, "event_id": "flow@2", "tick": 2}],
        registry=registry,
        current_tick=2,
        horizon=12,
    )

    assert typed["event_class"] == "task"
    assert typed["actionable"] is True
    assert typed["decision_required"] is True
    assert typed["response_opportunity_tick"] == 4
    assert typed["materiality_passed"] is True
    assert routine["event_class"] == "telemetry"
    assert routine["actionable"] is False
    assert routine["decision_required"] is False


def test_unknown_source_event_registry_entry_stays_non_actionable() -> None:
    [event] = _apply_typed_source_event_registry(
        [
            {
                "event_id": "unknown@1",
                "type": "unregistered_source_event",
                "origin": "source_schedule",
                "tick": 1,
                "actionable": True,
                "decision_required": True,
            }
        ],
        registry={},
        current_tick=1,
        horizon=8,
    )

    assert event["event_class"] == "telemetry"
    assert event["actionable"] is False
    assert event["decision_required"] is False


def test_registered_source_event_can_parent_later_tls_effect() -> None:
    source_event = {
        "event_id": "flow@3",
        "type": "traffic_demand_change",
        "origin": "source_schedule",
        "tick": 3,
        "event_class": "task",
        "actionable": True,
        "decision_required": True,
    }
    visible: dict[str, dict[str, object]] = {}
    _authoritative_source_event(source_event, "source-evidence", visible)
    action = Action(
        tool_calls=[
            ToolCall(
                name="set_signal_phase_duration",
                args={"tls_id": "tls-a"},
                call_id="control-call",
                consumes_evidence_ids=["source-evidence"],
            )
        ]
    )
    result = ToolResult(
        name="set_signal_phase_duration",
        ok=True,
        state_changing=True,
        call_id="control-call",
        evidence_id="tool-evidence",
        payload={
            "sumo_state_mutated": True,
            "before_runtime_state": {"remaining_duration": 5.0},
            "after_runtime_state": {"remaining_duration": 15.0},
            "complete_source_identity_sha256": "source-a",
        },
    )

    [effect] = _native_signal_action_effect_events(
        action=action,
        tool_results=[result],
        backend_tick_payload=_backend_tick_payload(),
        applied_tick=4,
        visible_source_events_by_evidence_id=visible,
    )

    assert effect["causal_parent_event_id"] == "flow@3"


def test_routine_source_telemetry_cannot_parent_tls_effect() -> None:
    source_event = {
        "event_id": "flow@2",
        "type": "traffic_demand_change",
        "origin": "source_schedule",
        "tick": 2,
        "event_class": "telemetry",
        "actionable": False,
        "decision_required": False,
        "materiality_passed": True,
    }
    visible: dict[str, dict[str, object]] = {}
    _authoritative_source_event(source_event, "source-evidence", visible)

    assert visible == {}


def test_live_phase_duration_effect_has_native_before_after_and_post_tick_edge() -> (
    None
):
    action = Action(
        tool_calls=[
            ToolCall(
                name="set_signal_phase_duration",
                args={
                    "tls_id": "tls-a",
                    "observed_program": "0",
                    "observed_phase": 2,
                    "remaining_duration_seconds": 15.0,
                },
                call_id="tick-0-call-0",
            )
        ]
    )
    result = ToolResult(
        name="set_signal_phase_duration",
        ok=True,
        state_changing=True,
        call_id="tick-0-call-0",
        evidence_id="tool-evidence",
        payload={
            "sumo_state_mutated": True,
            "sumo_tls_id": "tls-a",
            "sumo_phase_duration_s": 15.0,
            "before_runtime_state": {
                "current_program": "0",
                "current_phase": 2,
                "remaining_duration": 5.0,
            },
            "after_runtime_state": {
                "current_program": "0",
                "current_phase": 2,
                "remaining_duration": 15.0,
            },
            "evidence_ids": ["native-control-evidence"],
            "complete_source_identity_sha256": "source-a",
        },
    )

    records = canonicalize_runtime_events(
        _native_signal_action_effect_events(
            action=action,
            tool_results=[result],
            backend_tick_payload=_backend_tick_payload(),
            applied_tick=0,
        ),
        applied_tick=0,
    )

    assert len(records) == 1
    effect = records[0]
    assert effect["origin"] == "agent_caused"
    assert effect["tool_name"] == "set_signal_phase_duration"
    assert effect["call_id"] == "tick-0-call-0"
    assert effect["before_state_digest"] != effect["after_state_digest"]
    assert effect["outcome_tick"] == 1
    assert effect["evidence_ids"] == [
        "tool-evidence",
        "native-control-evidence",
    ]
    assert effect["applied_action"]["post_tick_tls_state"] == {
        "current_program": "0",
        "current_phase": 3,
        "remaining_duration": 11.0,
    }
    assert effect["action_to_outcome_edge"] == {
        "source": "call:tick-0-call-0",
        "target": f"outcome:{effect['event_id']}",
        "kind": "action_to_outcome",
    }


def test_signal_ack_or_missing_post_tick_state_does_not_claim_action_effect() -> None:
    action = Action(
        tool_calls=[
            ToolCall(
                name="set_signal_phase_duration",
                args={"tls_id": "tls-a"},
                call_id="tick-0-call-0",
            )
        ]
    )
    acknowledged = ToolResult(
        name="set_signal_phase_duration",
        ok=True,
        state_changing=True,
        call_id="tick-0-call-0",
        payload={"sumo_state_mutated": False},
    )
    changed_without_outcome = ToolResult(
        name="set_signal_phase_duration",
        ok=True,
        state_changing=True,
        call_id="tick-0-call-0",
        payload={
            "sumo_state_mutated": True,
            "before_runtime_state": {"remaining_duration": 5.0},
            "after_runtime_state": {"remaining_duration": 15.0},
            "complete_source_identity_sha256": "source-a",
        },
    )

    assert (
        _native_signal_action_effect_events(
            action=action,
            tool_results=[acknowledged],
            backend_tick_payload=_backend_tick_payload(),
            applied_tick=0,
        )
        == []
    )
    assert (
        _native_signal_action_effect_events(
            action=action,
            tool_results=[changed_without_outcome],
            backend_tick_payload={"realized_events": []},
            applied_tick=0,
        )
        == []
    )


def test_materialized_signal_program_emits_native_state_effect() -> None:
    action = Action(
        tool_calls=[
            ToolCall(
                name="set_signal_program",
                args={"tls_id": "tls-a", "program_id": "alternate"},
                call_id="program-call",
            )
        ]
    )
    result = ToolResult(
        name="set_signal_program",
        ok=True,
        state_changing=True,
        call_id="program-call",
        evidence_id="tool-evidence",
        payload={
            "_status": "pending",
            "tls_id": "tls-a",
            "program_id": "alternate",
            "prior_program": "0",
            "prior_phase": 1,
            "prior_state": "Gr",
            "complete_source_identity_sha256": "source-a",
            "evidence_ids": ["native-control-evidence"],
        },
    )
    backend_tick_payload = _backend_tick_payload()
    backend_tick_payload["realized_events"][0]["materialized_signal_controls"] = [
        {
            "tls_id": "tls-a",
            "program_id": "alternate",
            "prior_program": "0",
            "prior_phase": 1,
            "prior_state": "Gr",
            "applied_at_tick": 0,
            "sumo_state_mutated": True,
            "sumo_program_readback": "alternate",
            "resulting_runtime_state": {
                "current_program": "alternate",
                "current_phase": 0,
                "current_state": "rG",
            },
            "evidence_ids": ["program-materialized-evidence"],
        }
    ]

    [effect] = _native_signal_action_effect_events(
        action=action,
        tool_results=[result],
        backend_tick_payload=backend_tick_payload,
        applied_tick=0,
    )

    assert effect["origin"] == "agent_caused"
    assert effect["agent_caused"] is True
    assert effect["tool_name"] == "set_signal_program"
    assert effect["call_id"] == "program-call"
    assert effect["tick"] == 0
    assert effect["before_state_digest"] != effect["after_state_digest"]
    assert effect["evidence_ids"] == [
        "tool-evidence",
        "native-control-evidence",
        "program-materialized-evidence",
    ]


def test_program_effect_keeps_tool_protocol_lineage_until_native_boundary() -> None:
    def schedule_program(args: dict[str, object], _ctx: ToolContext):
        return {
            "_status": "pending",
            "tls_id": args["tls_id"],
            "program_id": args["program_id"],
            "prior_program": "0",
            "prior_phase": 1,
            "prior_state": "Gr",
            "complete_source_identity_sha256": "source-a",
            "evidence_ids": ["native-control-evidence"],
        }

    registry = ToolRegistry(
        budget=TickBudget(max_tool_calls_per_tick=2, max_total_tool_calls=4),
        seed=42,
        difficulty_level="high",
    )
    registry.register(
        ToolSpec(
            name="set_signal_program",
            description="test program control",
            parameters={
                "type": "object",
                "properties": {
                    "tls_id": {"type": "string"},
                    "program_id": {"type": "string"},
                },
                "required": ["tls_id", "program_id"],
            },
            handler=schedule_program,
            state_changing=True,
        )
    )
    action = Action(
        tool_calls=[
            ToolCall(
                name="set_signal_program",
                args={"tls_id": "tls-a", "program_id": "alternate"},
                call_id="program-call",
            )
        ]
    )
    lineage: dict[tuple[str, str], dict[str, object]] = {}
    request_results = registry.execute_action(
        action,
        ToolContext(tick=0, seed=42, backend=object()),
    )
    request_results[0].evidence_id = "request-evidence"
    assert (
        _native_signal_action_effect_events(
            action=action,
            tool_results=request_results,
            backend_tick_payload=_backend_tick_payload(),
            applied_tick=0,
            pending_signal_program_calls=lineage,
        )
        == []
    )

    scheduled_results = registry.execute_action(
        Action(),
        ToolContext(tick=1, seed=42, backend=object()),
    )
    scheduled_results[0].evidence_id = "scheduled-evidence"
    assert (
        _native_signal_action_effect_events(
            action=Action(),
            tool_results=scheduled_results,
            backend_tick_payload=_backend_tick_payload(),
            applied_tick=1,
            pending_signal_program_calls=lineage,
        )
        == []
    )

    materialized_payload = _backend_tick_payload()
    materialized_payload["realized_events"][0]["tick"] = 2
    materialized_payload["realized_events"][0]["materialized_signal_controls"] = [
        {
            "tls_id": "tls-a",
            "program_id": "alternate",
            "prior_program": "0",
            "prior_phase": 1,
            "prior_state": "Gr",
            "applied_at_tick": 2,
            "sumo_state_mutated": True,
            "sumo_program_readback": "alternate",
            "resulting_runtime_state": {
                "current_program": "alternate",
                "current_phase": 0,
                "current_state": "rG",
            },
            "evidence_ids": ["program-materialized-evidence"],
        }
    ]
    [effect] = _native_signal_action_effect_events(
        action=Action(),
        tool_results=[],
        backend_tick_payload=materialized_payload,
        applied_tick=2,
        pending_signal_program_calls=lineage,
    )

    assert effect["call_id"] == "program-call"
    assert effect["tool_name"] == "set_signal_program"
    assert effect["tick"] == 2
    assert effect["evidence_ids"] == [
        "request-evidence",
        "scheduled-evidence",
        "native-control-evidence",
        "program-materialized-evidence",
    ]
    assert lineage == {}


def test_program_no_effect_or_rejection_cannot_leave_effect_lineage() -> None:
    action = Action(
        tool_calls=[
            ToolCall(
                name="set_signal_program",
                args={"tls_id": "tls-a", "program_id": "alternate"},
                call_id="program-call",
            )
        ]
    )
    for error_code in ("NO_EFFECT", "DOMAIN_REJECTED"):
        lineage: dict[tuple[str, str], dict[str, object]] = {}
        pending = ToolResult(
            name="set_signal_program",
            ok=True,
            call_id="program-call",
            payload={"_status": "pending"},
        )
        assert (
            _native_signal_action_effect_events(
                action=action,
                tool_results=[pending],
                backend_tick_payload=_backend_tick_payload(),
                applied_tick=0,
                pending_signal_program_calls=lineage,
            )
            == []
        )
        rejected = ToolResult(
            name="set_signal_program",
            ok=False,
            call_id="program-call",
            error_code=error_code,
            state_changing=True,
        )
        assert (
            _native_signal_action_effect_events(
                action=Action(),
                tool_results=[rejected],
                backend_tick_payload=_backend_tick_payload(),
                applied_tick=1,
                pending_signal_program_calls=lineage,
            )
            == []
        )
        assert lineage == {}


def test_control_summary_links_request_to_later_first_physical_effect() -> None:
    evidence = EvidenceLogger("traffic-cross-tick")
    evidence.log(
        kind="tool_call",
        tick=0,
        source="tool",
        payload={
            "name": "set_signal_program",
            "ok": True,
            "call_id": "program-call",
            "state_changing": False,
            "payload": {"_status": "pending", "due_tick": 1},
        },
    )
    evidence.log(
        kind="tool_call",
        tick=1,
        source="tool",
        payload={
            "name": "set_signal_program",
            "ok": True,
            "call_id": "program-call",
            "state_changing": True,
            "payload": {
                "_status": "pending",
                "tls_id": "tls-a",
                "program_id": "alternate",
            },
        },
    )
    for tick in (2, 3):
        evidence.log(
            kind="realized_event",
            tick=tick,
            source="engine",
            payload={
                "origin": "agent_caused",
                "agent_caused": True,
                "call_id": "program-call",
                "tool_name": "set_signal_program",
                "before_state_digest": f"before-{tick}",
                "after_state_digest": f"after-{tick}",
            },
        )

    assert control_summary_from_evidence(evidence) == {
        "distinct_control_ticks": [2],
        "distinct_physical_tools": ["set_signal_program"],
        "tool_ticks": {"set_signal_program": [0]},
        "effect_tool_ticks": {"set_signal_program": [2]},
        "distinct_physical_actuator_endpoints": ["set_signal_program|tls-a"],
        "actuator_endpoint_ticks": {"set_signal_program|tls-a": [2]},
    }


def test_control_summary_rejects_unrequested_or_pre_request_effects() -> None:
    evidence = EvidenceLogger("traffic-invalid-effects")
    evidence.log(
        kind="realized_event",
        tick=0,
        source="engine",
        payload={
            "origin": "agent_caused",
            "agent_caused": True,
            "call_id": "program-call",
            "tool_name": "set_signal_program",
            "before_state_digest": "before",
            "after_state_digest": "after",
        },
    )
    evidence.log(
        kind="tool_call",
        tick=1,
        source="tool",
        payload={
            "name": "set_signal_program",
            "ok": True,
            "call_id": "program-call",
            "state_changing": True,
            "payload": {"tls_id": "tls-a"},
        },
    )
    evidence.log(
        kind="realized_event",
        tick=2,
        source="engine",
        payload={
            "origin": "agent_caused",
            "agent_caused": True,
            "call_id": "unrequested-call",
            "tool_name": "set_signal_program",
            "before_state_digest": "other-before",
            "after_state_digest": "other-after",
        },
    )

    assert control_summary_from_evidence(evidence) == {
        "distinct_control_ticks": [],
        "distinct_physical_tools": [],
        "tool_ticks": {"set_signal_program": [1]},
        "effect_tool_ticks": {},
    }


def test_backend_source_record_ids_are_not_claimed_as_logger_evidence():
    raw = {'type': 'traffic_demand_change', 'origin': 'source_schedule',
           'event_id': 'source-flow:0', 'tick': 0,
           'evidence_ids': ['sumo:source-sha:source-flow:0']}
    [event] = _apply_typed_source_event_registry([raw], registry={}, current_tick=0, horizon=4)
    evidence = EvidenceLogger('source-event-binding')
    evidence_id = evidence.log('realized_event', 0, payload=dict(event), source='engine')
    _authoritative_source_event(event, evidence_id, {})
    assert event['evidence_ids'] == [evidence_id]
    assert event['source_record_ids'] == raw['evidence_ids']
    assert event['event_id'] == raw['event_id']
    assert evidence.items_by_kind('realized_event')[0].payload['evidence_ids'] == []
