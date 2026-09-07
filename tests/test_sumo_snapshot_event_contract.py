from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

from baselines.llm_agent import LLMAgent, LLMConfig
from core.event_protocol import EventDecisionClass, resolve_event_decision
from domains.traffic.adapter import (
    _apply_typed_source_event_registry,
    _post_tick_native_state,
)
from domains.traffic.backends.sumo_backend import SumoBackend


def _backend() -> SumoBackend:
    backend = SumoBackend({})
    backend._horizon = 4
    backend._sidecar = SimpleNamespace(
        simulation_step=lambda: None,
        snapshot=lambda *, tick: {
            "tick": tick,
            "sim_time": 30.0,
            "n_vehicles": 7,
            "arrived": 2,
            "departed": 3,
            "network_counts": {"n_lanes": 2, "n_edges": 1},
        },
    )
    return backend


def test_live_snapshot_is_valid_telemetry_without_a_model_wakeup() -> None:
    record = _backend().tick(0)
    events = _apply_typed_source_event_registry(
        asdict(record)["realized_events"], registry={}, current_tick=0, horizon=4,
    )
    snapshot = next(event for event in events if event["type"] == "sumo_live_snapshot")
    resolution = resolve_event_decision(snapshot)

    assert resolution.requires_decision is False
    assert resolution.interrupt_reason is None
    assert resolution.violation_codes == ()
    assert resolution.decision_class is EventDecisionClass.TELEMETRY


def test_snapshot_annotation_preserves_native_state_and_other_event_contracts() -> None:
    backend = _backend()
    record = backend.tick(0)
    native_payload = asdict(record)
    legacy_payload = deepcopy(native_payload)
    legacy_payload["realized_events"][0].pop("event_class")

    assert _post_tick_native_state(native_payload) == _post_tick_native_state(legacy_payload)
    state = backend.snapshot()
    costs = backend.ground_truth_costs()
    scores = backend.scoring_records()
    # The old event representation differs only in this annotation; neither
    # native state nor numeric scoring records may depend on its presence.
    record.realized_events[0].pop("event_class")
    assert asdict(record) == legacy_payload
    assert backend.snapshot() == state
    assert backend.ground_truth_costs() == costs
    assert backend.scoring_records() == scores

    events = _apply_typed_source_event_registry(
        native_payload["realized_events"],
        registry={"traffic_demand_change": {
            "event_class": "alarm", "actionable_ticks": [0],
            "materiality_threshold": 1, "response_window_ticks": 1,
        }},
        current_tick=0, horizon=4,
    )
    demand = next(event for event in events if event["type"] == "traffic_demand_change")
    assert resolve_event_decision(demand).requires_decision is True
    unknown = resolve_event_decision({
        "type": "unregistered_event", "actionable": False, "decision_required": False,
    })
    assert unknown.requires_decision is False
    assert unknown.violation_codes == ("missing_event_decision_contract",)


def test_telemetry_annotation_is_visible_in_prompt_projection() -> None:
    backend = _backend()
    record = backend.tick(0)
    events = deepcopy(record.realized_events)
    legacy_events = deepcopy(events)
    legacy_events[0].pop("event_class")
    observation = backend.snapshot()
    agent = LLMAgent(LLMConfig(interaction_mode="logical_persistent"))
    current = agent._observation_summary({
        **observation, "__last_realized_events__": events,
    })
    previous = agent._observation_summary({
        **observation, "__last_realized_events__": legacy_events,
    })
    # Wake semantics and physics are unchanged, but exact model-visible bytes
    # are not equivalent. Old trajectories cannot be relabelled as a new run.
    current_prompt = agent._serialize_prompt_body(current, max_chars=128_000)
    previous_prompt = agent._serialize_prompt_body(previous, max_chars=128_000)
    assert current_prompt != previous_prompt
    assert current["last_realized_events"][0].pop("event_class") == "telemetry"
    assert current == previous
