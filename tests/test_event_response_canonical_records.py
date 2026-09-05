"""Regressions for canonical world records emitted by persistent episodes."""

from runner.episode import _build_event_response_records


def _steps(*, visibility="visible", request_tick=101, proven=True):
    # Shape of hy3 ta24: alarm at 100, delayed repair requested at 101,
    # native effect at 104. A pending acknowledgement is not an effect.
    return [
        {
            "tick": 100,
            "info": {
                "extra": {
                    "world_evolution_records": [
                        {
                            "event_id": "breakdown",
                            "origin": "declared_perturbation",
                            "applied_tick": 0 if request_tick == 0 else 100,
                            "visibility": visibility,
                            "evidence_ids": ["alarm-evidence"],
                        }
                    ]
                }
            },
        },
        {
            "tick": 103,
            "action_tick": 103,
            "info": {
                "extra": {
                    "world_evolution_records": [
                        {
                            "event_id": "repair",
                            "origin": "agent_caused",
                            "causal_parent_event_id": "breakdown",
                            "call_id": "repair-call",
                            "outcome_tick": 104,
                            "evidence_ids": ["native-effect"],
                        }
                    ],
                }
            },
            "tool_trace_edges": [
                {
                    "call_id": "repair-call",
                    "request_tick": request_tick,
                    "effect_tick": 104,
                    "state_changing": True,
                    "effect_proven": proven,
                    "consumes_evidence_ids": ["alarm-evidence"],
                    "produces_evidence_ids": ["receipt", "native-effect"],
                    "backend_effect_evidence_ids": ["native-effect"],
                }
            ],
        },
    ]


def test_delayed_repair_retains_causal_event_and_request_time():
    (record,) = _build_event_response_records(_steps())
    assert record["causal_parent_event_id"] == "breakdown"
    assert record["first_control_call_tick"] == 101
    assert record["first_effect_tick"] == 104
    assert _build_event_response_records(_steps(proven=False)) == []


def test_canonical_hidden_visibility_survives_response_join():
    (record,) = _build_event_response_records(_steps(visibility="hidden"))
    assert record["visibility"] == "hidden"


def test_tick_zero_request_is_not_replaced_by_delayed_effect_tick():
    (record,) = _build_event_response_records(_steps(request_tick=0))
    assert record["first_control_call_tick"] == 0
    assert record["first_observed_tick"] == 0
