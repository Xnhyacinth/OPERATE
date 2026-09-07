from __future__ import annotations

import pytest

from core import Action, EvidenceLogger, ToolCall, ToolResult
from runner.episode import _build_event_response_records, _tool_trace_edges


def test_event_response_records_require_native_parent_and_effect_lineage() -> None:
    records = _build_event_response_records(
        [
            {
                "tick": 2,
                "tool_trace_edges": [],
                "info": {
                    "extra": {
                        "world_evolution_records": [
                            {
                                "event_id": "outage-1",
                                "type": "machine_breakdown",
                                "applied_tick": 2,
                                "hidden": False,
                                "response_window_required": True,
                                "response_deadline_tick": 5,
                                "evidence_ids": ["event-evidence"],
                            }
                        ]
                    }
                },
            },
            {
                "tick": 3,
                "tool_trace_edges": [
                    {
                        "call_id": "call-1",
                        "consumes_evidence_ids": ["event-evidence"],
                        "produces_evidence_ids": ["effect-evidence"],
                        "request_tick": 3,
                        "effect_tick": 4,
                        "state_changing": True,
                        "effect_proven": True,
                    }
                ],
                "info": {
                    "extra": {
                        "world_evolution_records": [
                            {
                                "event_id": "repair-1",
                                "origin": "agent_caused",
                                "causal_parent_event_id": "outage-1",
                                "call_id": "call-1",
                                "applied_tick": 4,
                                "evidence_ids": ["effect-evidence"],
                            }
                        ]
                    }
                },
            },
        ]
    )

    assert records == [
        {
            "event_id": "outage-1",
            "causal_parent_event_id": "outage-1",
            "call_id": "call-1",
            "event_origin": None,
            "declared_perturbation": False,
            "event_tick": 2,
            "visibility": "visible",
            "surprise": False,
            "first_observed_tick": 3,
            "first_investigation_tick": None,
            "first_control_call_tick": 3,
            "first_effect_tick": 4,
            "mandatory_response_tick": 5,
            "response_status": "causal",
            "observation_evidence_ids": ["event-evidence"],
            "trigger_evidence_ids": ["event-evidence"],
            "action_consumes_evidence_ids": ["event-evidence"],
            "action_evidence_ids": ["effect-evidence"],
            "backend_effect_evidence_ids": ["effect-evidence"],
            "outcome_evidence_ids": ["effect-evidence"],
        }
    ]


def test_trace_edges_join_consumed_and_produced_evidence_to_native_effect() -> None:
    action = Action(
        tool_calls=[
            ToolCall(
                name="inspect_state",
                call_id="inspect-1",
                consumes_evidence_ids=["ev_prior"],
                depends_on_call_ids=["plan-1"],
            ),
            ToolCall(
                name="set_dispatch",
                call_id="control-1",
                consumes_evidence_ids=["ev_prior", "ev_state"],
            ),
        ]
    )
    results = [
        ToolResult(
            name="inspect_state",
            ok=True,
            call_id="inspect-1",
            evidence_id="ev_inspect",
        ),
        ToolResult(
            name="set_dispatch",
            ok=True,
            call_id="control-1",
            state_changing=True,
            evidence_id="ev_ack",
        ),
    ]
    edges = _tool_trace_edges(
        action,
        results,
        realized_events=[
            {
                "agent_caused": True,
                "call_id": "control-1",
                "outcome_tick": 7,
                "evidence_ids": ["ev_effect"],
            }
        ],
        applied_tick=6,
        request_tick=3,
    )

    assert edges[0]["consumes_evidence_ids"] == ["ev_prior"]
    assert edges[0]["produces_evidence_ids"] == ["ev_inspect"]
    assert edges[0]["effect_proven"] is False
    assert edges[1]["consumes_evidence_ids"] == ["ev_prior", "ev_state"]
    assert edges[1]["produces_evidence_ids"] == ["ev_ack", "ev_effect"]
    assert edges[1]["depends_on_call_ids"] == []
    assert edges[1]["effect_tick"] == 7
    assert edges[1]["request_tick"] == 3
    assert results[1].to_dict()["produces_evidence_ids"] == [
        "ev_ack",
        "ev_effect",
    ]


def test_delayed_effect_retains_original_request_tick() -> None:
    call = ToolCall(name="repair_machine", call_id="tick-2-call-1")
    result = ToolResult(
        name="repair_machine",
        ok=True,
        call_id="tick-2-call-1",
        state_changing=True,
    )

    edge = _tool_trace_edges(
        Action(),
        [result],
        realized_events=[
            {
                "origin": "agent_caused",
                "call_id": "tick-2-call-1",
                "outcome_tick": 5,
                "evidence_ids": ["ev-effect"],
            }
        ],
        applied_tick=4,
        known_calls={"tick-2-call-1": call},
        known_call_ticks={"tick-2-call-1": 2},
    )[0]

    assert edge["request_tick"] == 2
    assert edge["effect_tick"] == 5


def test_nonempty_unknown_result_call_id_is_not_remapped_by_tool_name() -> None:
    current = ToolCall(name="repair_machine", call_id="tick-4-call-3")
    result = ToolResult(
        name="repair_machine",
        ok=True,
        call_id="tick-2-call-1",
        state_changing=True,
    )

    edge = _tool_trace_edges(
        Action(tool_calls=[current]),
        [result],
        applied_tick=4,
        request_tick=4,
    )[0]

    assert edge["call_id"] == "tick-2-call-1"
    assert result.call_id == "tick-2-call-1"


def test_self_reported_hidden_evidence_is_filtered_until_revealed() -> None:
    action = Action(
        tool_calls=[
            ToolCall(
                name="set_dispatch",
                call_id="control-1",
                consumes_evidence_ids=["visible-1", "hidden-1"],
            )
        ]
    )
    result = ToolResult(
        name="set_dispatch",
        ok=True,
        call_id="control-1",
        state_changing=True,
    )

    edge = _tool_trace_edges(
        action,
        [result],
        applied_tick=2,
        request_tick=1,
        visible_evidence_ids={"visible-1"},
    )[0]

    assert edge["consumes_evidence_ids"] == ["visible-1"]


def test_acknowledgement_without_native_effect_has_no_effect_tick() -> None:
    action = Action(
        tool_calls=[ToolCall(name="set_dispatch", call_id="control-2")]
    )
    result = ToolResult(
        name="set_dispatch",
        ok=True,
        call_id="control-2",
        state_changing=True,
        evidence_id="ev_ack_only",
    )

    edge = _tool_trace_edges(action, [result], applied_tick=4)[0]

    assert edge["produces_evidence_ids"] == ["ev_ack_only"]
    assert edge["effect_tick"] is None
    assert edge["effect_proven"] is False


@pytest.mark.parametrize("evidence_backed", [False, True])
def test_canonical_origin_only_event_requires_ledger_to_prove_effect(evidence_backed) -> None:
    """Logistics adapters may emit the canonical string origin field only."""
    action = Action(
        tool_calls=[ToolCall(name="repair_machine", call_id="repair-1")]
    )
    evidence = EvidenceLogger("canonical-origin")
    ack_id = evidence.log("tool_call", 8, {"call_id": "repair-1"}, source="tool")
    event = {"origin": "agent_caused", "call_id": "repair-1", "outcome_tick": 9}
    effect_id = evidence.log("realized_event", 9, dict(event), source="engine")
    event["evidence_ids"] = [effect_id]
    result = ToolResult(
        name="repair_machine",
        ok=True,
        call_id="repair-1",
        state_changing=True,
        evidence_id=ack_id,
    )

    edge = _tool_trace_edges(
        action,
        [result],
        realized_events=[event],
        applied_tick=8,
        evidence_logger=evidence if evidence_backed else None,
    )[0]

    assert edge["produces_evidence_ids"] == [ack_id, effect_id]
    assert edge["effect_tick"] == 9
    assert edge["effect_proven"] is evidence_backed
