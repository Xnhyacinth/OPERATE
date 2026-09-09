from core import Action, ToolCall
from core.event_protocol import EventDecisionClass, resolve_event_decision
from domains.logistics.adapter import LogisticsEnvironment
from domains.logistics.backends.route_sim import RouteDemandSimulator, _Vehicle
from domains.logistics.seeds.from_vrplib import build_cvrp_dispatch_seed
from runner.realtime_actor import _action_effect_closure


def test_delayed_capacity_arrivals_are_typed_agent_outcomes() -> None:
    simulator = RouteDemandSimulator()
    simulator._vehicles = {
        "standby": _Vehicle(
            vid="standby",
            capacity=10.0,
            remaining_capacity=10.0,
            active=False,
            is_standby=True,
        )
    }
    simulator.queue_capacity_effect(
        due_tick=1,
        kind="dispatch_vehicle",
        payload={"vehicle_id": "standby"},
    )
    simulator.queue_capacity_effect(
        due_tick=1,
        kind="hire_spot_carrier",
        payload={"region": "north", "capacity_units": 4.0},
    )

    simulator._drain_effects(1)

    assert [event["type"] for event in simulator._realized_events_this_tick] == [
        "vehicle_dispatched",
        "spot_carrier_arrived",
    ]
    for event in simulator._realized_events_this_tick:
        resolution = resolve_event_decision(event)
        assert resolution.decision_class is EventDecisionClass.AGENT_OUTCOME
        assert resolution.requires_decision is False
        assert resolution.violation_codes == ()


def test_visible_route_disruptions_are_typed_interrupts() -> None:
    seed = build_cvrp_dispatch_seed(
        instance="A-n32-k5",
        seed=42,
        difficulty_level="medium",
        difficulty_mode="time_pressure",
    )
    scenario = seed.to_dict()
    scenario["perturbations"].extend(
        [
            {
                "kind": "blocked_arc",
                "trigger_tick": 2,
                "duration_ticks": 2,
                "hidden": False,
                "target": {"customer_index": 0},
                "intensity": 1.0,
            },
            {
                "kind": "traffic_delay",
                "trigger_tick": 3,
                "duration_ticks": 2,
                "hidden": False,
                "target": {},
                "intensity": 1.5,
            },
            {
                "kind": "urgent_order",
                "trigger_tick": 6,
                "duration_ticks": 1,
                "hidden": False,
                "target": {},
                "intensity": 1.0,
            },
        ]
    )
    env = LogisticsEnvironment()
    env.reset(scenario, seed=seed.seed)

    realized_events: list[dict] = []
    for _ in range(7):
        realized_events.extend(
            env.step(Action(tool_calls=[ToolCall(name="wait")])).info.realized_events
        )

    disruptions = {
        event["type"]: event
        for event in realized_events
        if event.get("type")
        in {
            "vehicle_breakdown",
            "demand_surge",
            "blocked_arc",
            "traffic_delay",
            "urgent_order",
        }
    }
    assert set(disruptions) == {
        "vehicle_breakdown",
        "demand_surge",
        "blocked_arc",
        "traffic_delay",
        "urgent_order",
    }
    assert disruptions["vehicle_breakdown"]["event_class"] == "safety"
    assert disruptions["demand_surge"]["event_class"] == "alarm"
    assert disruptions["blocked_arc"]["event_class"] == "alarm"
    assert disruptions["traffic_delay"]["event_class"] == "alarm"
    assert disruptions["urgent_order"]["event_class"] == "task"
    for event in disruptions.values():
        resolution = resolve_event_decision(event)
        assert resolution.requires_decision is True
        assert resolution.violation_codes == ()


def test_hidden_route_disruption_is_typed_without_waking() -> None:
    seed = build_cvrp_dispatch_seed(
        instance="A-n32-k5",
        seed=42,
        difficulty_level="medium",
        difficulty_mode="time_pressure",
    )
    scenario = seed.to_dict()
    scenario["perturbations"] = [
        {
            "kind": "urgent_order",
            "trigger_tick": 1,
            "duration_ticks": 1,
            "hidden": True,
            "target": {},
            "intensity": 1.0,
        }
    ]
    env = LogisticsEnvironment()
    env.reset(scenario, seed=seed.seed)
    env.step(Action(tool_calls=[ToolCall(name="wait")]))

    event = next(
        event
        for event in env.step(
            Action(tool_calls=[ToolCall(name="wait")])
        ).info.realized_events
        if event.get("type") == "urgent_order"
    )

    assert event["event_class"] == "task"
    assert event["actionable"] is False
    assert event["decision_required"] is False
    assert resolve_event_decision(event).requires_decision is False


def test_terminal_route_disruption_has_no_response_opportunity() -> None:
    seed = build_cvrp_dispatch_seed(
        instance="A-n32-k5",
        seed=42,
        difficulty_level="medium",
        difficulty_mode="time_pressure",
    )
    scenario = seed.to_dict()
    scenario["horizon_ticks"] = 2
    scenario["perturbations"] = [
        {
            "kind": "urgent_order",
            "trigger_tick": 1,
            "duration_ticks": 1,
            "hidden": False,
            "target": {},
            "intensity": 1.0,
        }
    ]
    env = LogisticsEnvironment()
    env.reset(scenario, seed=seed.seed)
    env.step(Action(tool_calls=[ToolCall(name="wait")]))

    event = next(
        event
        for event in env.step(
            Action(tool_calls=[ToolCall(name="wait")])
        ).info.realized_events
        if event.get("type") == "urgent_order"
    )

    assert event["event_class"] == "task"
    assert event["actionable"] is False
    assert event["decision_required"] is False
    assert event["response_opportunity_tick"] is None
    assert resolve_event_decision(event).requires_decision is False


def test_unknown_route_perturbation_does_not_create_an_actionable_event() -> None:
    seed = build_cvrp_dispatch_seed(
        instance="A-n32-k5",
        seed=42,
        difficulty_level="medium",
        difficulty_mode="time_pressure",
    )
    scenario = seed.to_dict()
    scenario["perturbations"] = [
        {
            "kind": "unknown_route_condition",
            "trigger_tick": 1,
            "duration_ticks": 2,
            "hidden": False,
            "target": {},
            "intensity": 2.0,
        }
    ]
    env = LogisticsEnvironment()
    env.reset(scenario, seed=seed.seed)

    events: list[dict] = []
    for _ in range(3):
        events.extend(
            env.step(Action(tool_calls=[ToolCall(name="wait")])).info.realized_events
        )

    assert not [event for event in events if event.get("decision_required") is True]


def test_delayed_vehicle_effect_carries_exact_tool_provenance() -> None:
    seed = build_cvrp_dispatch_seed(
        instance="A-n32-k5",
        seed=42,
        difficulty_level="medium",
        difficulty_mode="time_pressure",
    )
    env = LogisticsEnvironment()
    observation = env.reset(seed.to_dict(), seed=seed.seed)
    standby_id = next(
        entity_id
        for entity_id, entity in observation["entities"].items()
        if entity.get("kind") == "vehicle" and entity.get("is_standby") is True
    )
    args = {"vehicle_id": standby_id, "depot_id": "depot"}

    request = Action(
        tool_calls=[
            ToolCall(
                name="dispatch_vehicle",
                args=args,
                call_id="dispatch-standby",
            )
        ]
    )
    requested = env.step(request)
    assert requested.tool_results[0].payload["_status"] == "pending"
    materialized = env.step(Action(tool_calls=[ToolCall(name="wait")]))
    effect = next(
        event
        for event in materialized.info.realized_events
        if event.get("type") == "vehicle_dispatched"
    )

    assert effect["call_id"] == "dispatch-standby"
    assert effect["tool_name"] == "dispatch_vehicle"
    assert effect["requested_action"] == {
        "name": "dispatch_vehicle",
        "args": args,
    }
    assert effect["before_state_digest"] != effect["after_state_digest"]
    assert effect["effect_tick"] == 1
    _, effect_observed, _, _ = _action_effect_closure(
        action=request,
        tool_results=[result.to_dict() for result in materialized.tool_results],
        realized_events=[effect],
        evidence_ledger=[item.to_dict() for item in env.evidence.items()],
        request_tick=0,
        simulator_tick=1,
    )
    assert effect_observed is True


def test_investigation_materialized_control_keeps_its_own_effect_identity() -> None:
    seed = build_cvrp_dispatch_seed(
        instance="A-n32-k5", seed=42, difficulty_level="medium",
        difficulty_mode="time_pressure",
    )
    env = LogisticsEnvironment()
    observation = env.reset(seed.to_dict(), seed=seed.seed)
    customers = [
        key for key, entity in observation["entities"].items()
        if entity.get("kind") == "customer"
    ]
    # A registry delay can mature during a later read-only decision stage.
    env._tools.get("hold_order").delay_ticks = 1
    env._tools.get("hold_order").fail_rate = 0.0
    queued = env.step(Action(tool_calls=[ToolCall(
        name="hold_order", args={"customer_id": customers[0], "until_tick": 5},
        call_id="delayed-old",
    )]))
    assert queued.tool_results[0].payload["_status"] == "pending"
    _, receipts = env.execute_investigation(Action(tool_calls=[]))
    [materialized] = receipts
    assert materialized.call_id == "delayed-old"
    assert materialized.ok
    first = env.step(Action(tool_calls=[]))
    effects = [event for event in first.info.realized_events
               if event.get("tool_name") == "hold_order"]
    assert len(effects) == 1
    assert effects[0]["call_id"] == "delayed-old"
    assert effects[0]["outcome_tick"] == 1
    assert materialized.evidence_id in effects[0]["evidence_ids"]

    env._tools.get("hold_order").delay_ticks = 0
    other_customer = next(
        key for key, entity in first.observation["entities"].items()
        if entity.get("kind") == "customer" and not entity.get("served")
        and key != customers[0]
    )
    second = env.step(Action(tool_calls=[ToolCall(
        name="hold_order", args={"customer_id": other_customer, "until_tick": 6},
        call_id="new-call",
    )]))
    effects = [event for event in second.info.realized_events
               if event.get("tool_name") == "hold_order"]
    assert len(effects) == 1
    assert effects[0]["call_id"] == "new-call"
    assert effects[0]["outcome_tick"] == 2
    assert effects[0]["requested_action"]["customer_id"] == other_customer
    assert not [event for event in env.step(Action(tool_calls=[])).info.realized_events
                if event.get("tool_name") == "hold_order"]
