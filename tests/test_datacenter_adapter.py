from pathlib import Path

import yaml

from baselines.oracle_offline import OracleOfflineAgent
from core import Action, ToolCall
from domains.datacenter.adapter import DatacenterEnvironment
from domains.datacenter.backends.alibaba_trace_backend import AlibabaTraceBackend
from domains.datacenter.seeds.schema import (
    DatacenterPerturbation,
    DatacenterScenarioSeed,
    Provenance,
)
from run import run_one


def _scenario() -> dict:
    jobs = [
        {
            "job_id": "long",
            "user": "u1",
            "start_time": 0,
            "duration_seconds": 300,
            "requested_gpu_units": 1,
            "requested_cpu_percent": 1,
            "instance_count": 1,
        },
        {
            "job_id": "short-a",
            "user": "u2",
            "start_time": 0,
            "duration_seconds": 60,
            "requested_gpu_units": 1,
            "requested_cpu_percent": 1,
            "instance_count": 1,
        },
        {
            "job_id": "short-b",
            "user": "u3",
            "start_time": 0,
            "duration_seconds": 60,
            "requested_gpu_units": 1,
            "requested_cpu_percent": 1,
            "instance_count": 1,
        },
    ]
    return DatacenterScenarioSeed(
        seed_id="datacenter_test",
        family="gpu_cluster_queue_control",
        horizon_ticks=8,
        tick_minutes=1,
        backend_config={
            "jobs": jobs,
            "gpu_capacity_units": 1,
            "cpu_capacity_units": 1,
            "initial_queue_policy": "fifo",
        },
        provenance=Provenance(
            data_source="alibaba_cluster_trace_gpu_v2020",
            files=["locked.csv"],
        ),
    ).to_dict()


def _arrival_queue_scenario() -> DatacenterScenarioSeed:
    """Small native queue with a hidden future arrival for policy tests."""
    return DatacenterScenarioSeed(
        seed_id="datacenter_queue_policy_test",
        family="gpu_cluster_queue_control",
        horizon_ticks=8,
        tick_minutes=1,
        backend_config={
            "jobs": [
                {
                    "job_id": "critical-tight",
                    "user": "critical-tenant",
                    "start_time": 0,
                    "duration_seconds": 60,
                    "requested_gpu_units": 1,
                    "requested_cpu_percent": 1,
                    "instance_count": 1,
                    "priority_class": "HP",
                },
                {
                    "job_id": "ordinary-tight",
                    "user": "ordinary-tenant",
                    "start_time": 0,
                    "duration_seconds": 60,
                    "requested_gpu_units": 1,
                    "requested_cpu_percent": 1,
                    "instance_count": 1,
                    "priority_class": "Spot",
                },
                {
                    "job_id": "critical-loose",
                    "user": "critical-tenant-2",
                    "start_time": 0,
                    "duration_seconds": 120,
                    "requested_gpu_units": 1,
                    "requested_cpu_percent": 1,
                    "instance_count": 1,
                    "priority_class": "HP",
                },
                {
                    "job_id": "future-secret",
                    "user": "future-tenant",
                    "start_time": 120,
                    "duration_seconds": 60,
                    "requested_gpu_units": 1,
                    "requested_cpu_percent": 1,
                    "instance_count": 1,
                    "priority_class": "HP",
                },
            ],
            "gpu_capacity_units": 1,
            "cpu_capacity_units": 1,
            "initial_queue_policy": "fifo",
            "source_time_scale": {
                "mode": "source_seconds",
                "seconds_per_tick": 60,
            },
        },
        provenance=Provenance(
            data_source="alibaba_cluster_trace_gpu_v2020",
            files=["locked.csv"],
        ),
    )


def test_deadline_criticality_first_orders_visible_jobs_deterministically() -> None:
    backend = AlibabaTraceBackend()
    backend.reset(_arrival_queue_scenario())
    result = backend.apply_tool_effect(
        "set_queue_policy", {"policy": "deadline_criticality_first"}
    )
    assert result["queue_policy"] == "deadline_criticality_first"

    backend.tick(0)
    queue = backend.queue_state()

    assert queue["queue_policy"] == "deadline_criticality_first"
    assert queue["dispatch_order"] == [
        "critical-tight",
        "ordinary-tight",
        "critical-loose",
    ]
    assert [job["job_id"] for job in queue["queued_jobs"]] == [
        "ordinary-tight",
        "critical-loose",
    ]
    rationale = queue["dispatch_rationale"]["ordinary-tight"]
    assert rationale["order"] == [
        "due_slack",
        "criticality_desc",
        "remaining_ticks",
        "submit_tick",
        "job_id",
    ]
    assert rationale["due_slack"] == 2
    assert rationale["criticality"] == 0.25


def test_queue_observation_hides_future_arrivals_until_their_submit_tick() -> None:
    env = DatacenterEnvironment()
    env.reset(_arrival_queue_scenario().to_dict(), seed=42)

    before_arrival = env.step(Action(tool_calls=[ToolCall(name="wait")]))
    assert "future-secret" not in before_arrival.observation["queue"]["dispatch_order"]
    assert "future-secret" not in before_arrival.observation["jobs"]
    assert (
        "future-secret" not in before_arrival.observation["queue"]["dispatch_rationale"]
    )

    arrival_tick = env.step(Action(tool_calls=[ToolCall(name="wait")]))
    assert "future-secret" not in arrival_tick.observation["queue"]["dispatch_order"]
    assert "future-secret" not in arrival_tick.observation["jobs"]

    arrived = env.step(Action(tool_calls=[ToolCall(name="wait")]))
    assert "future-secret" in arrived.observation["queue"]["dispatch_order"]
    assert "future-secret" in arrived.observation["jobs"]
    env.close()


def test_runtime_event_observation_exposes_visible_ids_but_not_hidden_events() -> None:
    visible_env = DatacenterEnvironment()
    visible_env.reset(_review_scenario().to_dict(), seed=42)
    visible_env.step(Action(tool_calls=[ToolCall(name="wait")]))
    visible = visible_env.step(Action(tool_calls=[ToolCall(name="wait")]))
    visible_events = visible.observation["runtime_events"]
    assert any(
        event.get("type") == "queue_burst" and event.get("event_id")
        for event in visible_events
    )
    visible_env.close()

    hidden_scenario = _arrival_queue_scenario()
    hidden_scenario.perturbations = [
        DatacenterPerturbation(
            kind="queue_burst",
            trigger_tick=1,
            duration_ticks=2,
            hidden=True,
            intensity=1.0,
        )
    ]
    hidden_env = DatacenterEnvironment()
    hidden_env.reset(hidden_scenario.to_dict(), seed=42)
    hidden_env.step(Action(tool_calls=[ToolCall(name="wait")]))
    hidden = hidden_env.step(Action(tool_calls=[ToolCall(name="wait")]))
    assert all(
        event.get("type") != "queue_burst"
        for event in hidden.observation["runtime_events"]
    )
    hidden_env.close()


def test_oracle_reviews_only_visible_datacenter_runtime_events() -> None:
    env = DatacenterEnvironment()
    scenario = _review_scenario()
    observation = env.reset(scenario.to_dict(), seed=42)
    agent = OracleOfflineAgent()
    agent.reset(env, scenario.to_dict(), seed=42)
    tool_specs = env.get_tool_specs()

    for _ in range(4):
        result = env.step(agent.act(observation, tool_specs))
        observation = result.observation

    reviews = env.ground_truth()["control_summary"]["policy_review_ledger"]
    assert reviews
    assert all(review["event_ids"] for review in reviews)
    assert all("future-secret" not in event_id for event_id in reviews[0]["event_ids"])
    assert any(
        any(":queue_burst:" in event_id for event_id in review["event_ids"])
        for review in reviews
    )
    env.close()


def test_w030_terminal_review_remains_unattributed_without_tail_tick() -> None:
    """A terminal review stays held; the backend must not infer an effect."""
    scenario_path = Path(
        "scenarios/operate_v0_58_0/datacenter/"
        "gpu_cluster_queue_control/time_pressure/basic/"
        "datacenter__gpu_cluster_queue_control__time_pressure__medium__"
        "alibaba_gpu_w030_f8aff532191f_546b09c9c415_medium__6447f64f__"
        "relabel_v1.yaml"
    )
    scenario = yaml.safe_load(scenario_path.read_text())
    env = DatacenterEnvironment()
    observation = env.reset(scenario, seed=int(scenario["seed"]))
    agent = OracleOfflineAgent()
    agent.reset(env, scenario, seed=int(scenario["seed"]))
    tool_specs = env.get_tool_specs()

    while True:
        result = env.step(agent.act(observation, tool_specs))
        observation = result.observation
        if result.done:
            break

    reviews = env.ground_truth()["control_summary"]["policy_review_ledger"]
    terminal_reviews = [
        review for review in reviews if not review["outcome_effect_ticks"]
    ]
    assert terminal_reviews
    assert terminal_reviews[-1]["review_tick"] == 7
    assert terminal_reviews[-1]["event_ticks"] == [6]
    env.close()


def test_dispatch_order_matches_first_running_job_after_policy_change() -> None:
    env = DatacenterEnvironment()
    env.reset(_arrival_queue_scenario().to_dict(), seed=42)

    result = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="set_queue_policy",
                    args={"policy": "deadline_criticality_first"},
                    idempotency_key="deadline-policy",
                )
            ]
        )
    )

    queue = result.observation["queue"]
    running = [
        job_id
        for job_id, job in result.observation["jobs"].items()
        if job["status"] == "running"
    ]
    assert queue["dispatch_order"]
    assert running == ["critical-tight"]
    assert running[0] == queue["dispatch_order"][0]
    assert queue["running_job_ids"] == running
    env.close()


def _review_scenario() -> DatacenterScenarioSeed:
    scenario = _arrival_queue_scenario()
    scenario.perturbations = [
        DatacenterPerturbation(
            kind="queue_burst",
            trigger_tick=1,
            duration_ticks=2,
            intensity=1.0,
        )
    ]
    return scenario


def _event_after_queue_burst() -> tuple[DatacenterEnvironment, str]:
    env = DatacenterEnvironment()
    env.reset(_review_scenario().to_dict(), seed=42)
    env.step(Action(tool_calls=[ToolCall(name="wait")]))
    event_tick = env.step(Action(tool_calls=[ToolCall(name="wait")]))
    event = next(
        item
        for item in event_tick.info.realized_events
        if item.get("type") == "queue_burst"
    )
    return env, str(event["event_id"])


def test_review_persistent_policy_records_runtime_lineage_without_effect() -> None:
    env, event_id = _event_after_queue_burst()

    result = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="review_persistent_policy",
                    args={
                        "event_ids": [event_id],
                        "policy_generation": 1,
                        "rationale": "queue pressure remains within the active policy envelope",
                    },
                    idempotency_key="review-queue-burst",
                )
            ]
        )
    )

    review = result.tool_results[0]
    assert review.ok is True
    assert review.state_changing is False
    assert review.payload["review_status"] == "accepted"
    assert review.payload["event_ids"] == [event_id]
    assert review.payload["policy_generation"] == 1
    assert review.payload["decision"] == "keep"
    assert len(review.payload["policy_digest"]) == 64
    assert review.payload["queue_order_digest"]
    assert review.payload["evidence_ids"]
    assert not any(
        event.get("type") == "control_effect" for event in result.info.realized_events
    )
    summary = env.ground_truth()["control_summary"]
    assert summary["policy_review_count"] == 1
    ledger = summary["policy_review_ledger"][0]
    assert ledger["event_ids"] == [event_id]
    assert ledger["decision"] == "keep"
    assert ledger["policy_digest"] == review.payload["policy_digest"]
    assert ledger["evidence_ids"] == review.payload["evidence_ids"]
    assert ledger["outcome_effect_ticks"] == []
    env.step(Action(tool_calls=[ToolCall(name="wait")]))
    outcome_ticks = env.ground_truth()["control_summary"]["policy_review_ledger"][0][
        "outcome_effect_ticks"
    ]
    assert outcome_ticks
    assert all(tick > review.payload["review_tick"] for tick in outcome_ticks)
    env.close()


def test_review_persistent_policy_rejects_empty_fake_stale_and_duplicate_inputs() -> (
    None
):
    env, event_id = _event_after_queue_burst()

    empty = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="review_persistent_policy",
                    args={
                        "event_ids": [],
                        "policy_generation": 1,
                        "rationale": "empty review must fail",
                    },
                    idempotency_key="review-empty",
                )
            ]
        )
    )
    assert empty.tool_results[0].ok is False

    fake = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="review_persistent_policy",
                    args={
                        "event_ids": ["forged-event-id"],
                        "policy_generation": 1,
                        "rationale": "fake review must fail",
                    },
                    idempotency_key="review-fake",
                )
            ]
        )
    )
    assert fake.tool_results[0].ok is False

    stale = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="review_persistent_policy",
                    args={
                        "event_ids": [event_id],
                        "policy_generation": 0,
                        "rationale": "stale generation must fail",
                    },
                    idempotency_key="review-stale",
                )
            ]
        )
    )
    assert stale.tool_results[0].ok is False

    accepted = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="review_persistent_policy",
                    args={
                        "event_ids": [event_id],
                        "policy_generation": 1,
                        "rationale": "the first valid review",
                    },
                    idempotency_key="review-first",
                )
            ]
        )
    )
    assert accepted.tool_results[0].ok is True

    duplicate = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="review_persistent_policy",
                    args={
                        "event_ids": [event_id],
                        "policy_generation": 1,
                        "rationale": "the same event cannot be reviewed twice",
                    },
                    idempotency_key="review-duplicate",
                )
            ]
        )
    )
    assert duplicate.tool_results[0].ok is False
    assert env.ground_truth()["control_summary"]["policy_review_count"] == 1
    env.close()


def test_review_persistent_policy_rejects_review_at_event_tick() -> None:
    backend = AlibabaTraceBackend()
    backend.reset(_review_scenario())
    backend.tick(0)
    record = backend.tick(1)
    event = next(
        item for item in record.realized_events if item.get("type") == "queue_burst"
    )

    rejected = backend.apply_tool_effect(
        "review_persistent_policy",
        {
            "event_ids": [event["event_id"]],
            "policy_generation": 1,
            "rationale": "same-tick review must fail",
        },
        current_tick=1,
    )

    assert rejected["_status"] == "error"
    assert rejected["error"] == "review_must_follow_event"
    assert backend.control_summary()["policy_review_count"] == 0


def test_review_persistent_policy_rejects_hidden_runtime_event() -> None:
    scenario = _arrival_queue_scenario()
    scenario.perturbations = [
        DatacenterPerturbation(
            kind="queue_burst",
            trigger_tick=1,
            duration_ticks=2,
            hidden=True,
            intensity=1.0,
        )
    ]
    backend = AlibabaTraceBackend()
    backend.reset(scenario)
    backend.tick(0)
    record = backend.tick(1)
    hidden_event = next(
        event for event in record.realized_events if event.get("type") == "queue_burst"
    )

    rejected = backend.apply_tool_effect(
        "review_persistent_policy",
        {
            "event_ids": [hidden_event["event_id"]],
            "policy_generation": 1,
            "rationale": "hidden event must not be citeable",
        },
        current_tick=2,
    )

    assert rejected["_status"] == "error"
    assert rejected["error"] == "event_not_visible"
    assert backend.control_summary()["policy_review_count"] == 0


def test_datacenter_adapter_exposes_native_tools_and_replays() -> None:
    env = DatacenterEnvironment()
    observation = env.reset(_scenario(), seed=42)

    assert observation["queue"]["queue_policy"] == "fifo"
    assert set(observation["stakeholder_trust"]) == {"u1", "u2", "u3"}
    assert "set_queue_policy" in {
        tool["function"]["name"] for tool in env.get_tool_specs()
    }
    policy_tool = next(
        tool
        for tool in env.get_tool_specs()
        if tool["function"]["name"] == "set_queue_policy"
    )
    assert (
        "deadline_criticality_first"
        in policy_tool["function"]["parameters"]["properties"]["policy"]["enum"]
    )
    reserve = next(
        tool
        for tool in env.get_tool_specs()
        if tool["function"]["name"] == "reserve_gpu_capacity"
    )
    assert set(reserve["function"]["parameters"]["required"]) == {
        "gpu_units",
        "duration_ticks",
    }
    plan = next(
        tool
        for tool in env.get_tool_specs()
        if tool["function"]["name"] == "commit_to_plan"
    )
    assert plan["function"]["parameters"]["properties"]["review_after_ticks"] == {
        "type": "integer",
        "minimum": 1,
        "maximum": 4,
        "description": (
            "Request the next model review after this many simulator ticks "
            "while current controls remain in force. The runner advances "
            "the backend autonomously and wakes early for visible events, "
            "tool failures, safety warnings, or active dilemmas."
        ),
    }
    result = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="set_queue_policy",
                    args={"policy": "shortest_job_first"},
                    idempotency_key="policy-1",
                )
            ]
        )
    )
    assert result.tool_results[0].ok
    assert env.ground_truth()["cost_components"]
    env.close()


def test_gpu_reservation_materializes_one_tick_after_request() -> None:
    env = DatacenterEnvironment()
    env.reset(_scenario(), seed=42)

    first = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="reserve_gpu_capacity",
                    args={"gpu_units": 2.0, "duration_ticks": 2},
                    idempotency_key="reserve-delayed",
                )
            ]
        )
    )

    assert first.tool_results[0].payload["_status"] == "pending"
    assert first.observation["capacity"]["reserved_gpu_units"] == 0.0

    second = env.step(Action(tool_calls=[ToolCall(name="wait")]))

    assert second.observation["capacity"]["reserved_gpu_units"] == 2.0
    materialized = next(
        result
        for result in second.tool_results
        if result.name == "reserve_gpu_capacity"
    )
    assert materialized.payload.get("_status") is None
    assert materialized.payload["reserved_gpu_units"] == 2.0
    assert materialized.payload["duration_ticks"] == 2
    assert materialized.payload["reservation_cost"] > 0.0
    assert materialized.state_changing is True
    control_effect = next(
        event
        for event in second.info.realized_events
        if event.get("type") == "control_effect"
        and event.get("tool_name") == "reserve_gpu_capacity"
    )
    assert control_effect["call_id"] == materialized.call_id
    assert control_effect["agent_caused"] is True
    assert control_effect["applied_action"]["gpu_units"] == 2.0
    assert control_effect["changed_state_fields"] == ["reserved_gpu_units"]
    assert control_effect["before_state_digest"] != control_effect["after_state_digest"]
    assert env.ground_truth()["control_summary"]["effect_tool_ticks"] == {
        "reserve_gpu_capacity": [1]
    }
    env.close()


def test_direct_backend_control_cannot_bypass_evidence_join() -> None:
    env = DatacenterEnvironment()
    env.reset(_scenario(), seed=42)

    env._backend.apply_tool_effect(
        "set_queue_policy",
        {"policy": "shortest_job_first"},
    )
    env._backend.tick(0)

    summary = env.ground_truth()["control_summary"]
    assert summary["distinct_control_ticks"] == []
    assert summary["distinct_physical_tools"] == []
    assert summary["tool_ticks"] == {}
    assert summary["effect_tool_ticks"] == {}
    assert summary["distinct_physical_actuator_endpoints"] == []
    env.close()


def test_delayed_gpu_reservation_fails_when_capacity_recovers_at_due_tick() -> None:
    scenario_path = Path(
        "scenarios/operate_v0_58_0/datacenter/"
        "gpu_cluster_spot_sla_control/deep_planning/high/"
        "alibaba_spot_gpu_a100_w448554_high.yaml"
    )
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario["perturbations"][0]["duration_ticks"] = 2
    scenario["perturbations"] = scenario["perturbations"][:1]

    env = DatacenterEnvironment()
    env.reset(scenario, seed=52)
    env.step(Action(tool_calls=[ToolCall(name="wait")]))
    submitted = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="reserve_gpu_capacity",
                    args={"gpu_units": 6.0, "duration_ticks": 2},
                    idempotency_key="restore-boundary-reserve",
                )
            ]
        )
    )
    assert submitted.tool_results[0].payload["_status"] == "pending"

    due = env.step(Action(tool_calls=[ToolCall(name="wait")]))
    materialized = next(
        result for result in due.tool_results if result.name == "reserve_gpu_capacity"
    )

    assert materialized.ok is False
    assert materialized.error_code == "DOMAIN_REJECTED"
    assert due.observation["capacity"]["reserved_gpu_units"] == 0.0
    assert not any(
        event.get("type") == "control_effect"
        and event.get("tool_name") == "reserve_gpu_capacity"
        for event in due.info.realized_events
    )
    tool_evidence = [
        item
        for item in env.evidence.items_by_kind("tool_call")
        if item.payload.get("call_id") == materialized.call_id
    ]
    assert tool_evidence
    assert all(item.payload["ok"] is False for item in tool_evidence[-1:])
    env.close()


def test_datacenter_sla_outcomes_drive_native_stakeholder_evidence() -> None:
    env = DatacenterEnvironment()
    env.reset(_scenario(), seed=42)

    for _ in range(5):
        result = env.step(Action(tool_calls=[]))
        if result.done:
            break

    evidence = env.evidence
    assert evidence is not None
    trust_events = evidence.items_by_kind("trust_event")
    assert trust_events
    assert all(item.payload["group_id"] in {"u1", "u2", "u3"} for item in trust_events)
    assert "stakeholder_trust" in env.ground_truth()
    env.close()


def test_oracle_completes_datacenter_task_contract() -> None:
    oracle = run_one(_scenario(), agent_name="oracle_offline")
    wait = run_one(_scenario(), agent_name="wait_only")

    assert oracle["task_completion"]["contract"] == (
        "datacenter.queue_sla_mitigation.v2"
    )
    assert oracle["task_completion"]["completed"] is True
    assert oracle["counterfactual"]["prevented_loss"] > 0
    assert (
        oracle["counterfactual"]["actual_cost"] < wait["counterfactual"]["actual_cost"]
    )


def test_oracle_replans_across_separate_extreme_capacity_intervals() -> None:
    scenario = _scenario()
    scenario["difficulty_level"] = "extreme"
    scenario["difficulty_mode"] = "deep_planning"
    scenario["backend_config"]["jobs"][0]["duration_seconds"] = 600
    scenario["perturbations"] = [
        {
            "kind": "capacity_reduction",
            "trigger_tick": 2,
            "duration_ticks": 3,
            "hidden": False,
            "intensity": 0.5,
        },
        {
            "kind": "capacity_reduction",
            "trigger_tick": 5,
            "duration_ticks": 3,
            "hidden": True,
            "intensity": 0.5,
        },
    ]

    oracle = run_one(scenario, agent_name="oracle_offline")

    assert oracle["task_completion"]["completed"] is True
    assert oracle["trajectory_summary"]["tool_histogram"]["reserve_gpu_capacity"] >= 2
    complexity = oracle["trajectory_summary"]["complexity"]
    assert complexity["n_effective_control_ticks"] >= 2
    # State-change ticks are native action/effect ticks, not the following
    # post-step observation tick.  The offline reference reserves at tick 1
    # for the visible capacity interval that begins at tick 2.
    assert 1 in complexity["state_change_ticks"]
    requirements = oracle["task_completion"]["evidence"]
    assert requirements["native_control_requirements_met"] is True
    assert requirements["reservation_arrivals"] >= 2
    assert len(requirements["distinct_control_ticks"]) >= 3


def test_spot_runtime_oracle_does_not_change_legacy_trace_policy() -> None:
    scenario_path = Path(
        "scenarios/operate_v0_58_0/datacenter/"
        "gpu_cluster_queue_control/time_pressure/basic/"
        "datacenter__gpu_cluster_queue_control__time_pressure__medium__"
        "alibaba_gpu_w000_d1c907cbfb6d_af68ac870a61_medium__"
        "c5c42329__relabel_v1.yaml"
    )
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))

    oracle = run_one(scenario, agent_name="oracle_offline")

    assert oracle["task_completion"]["completed"] is True
    assert oracle["counterfactual"]["prevented_loss"] > 0.0


def test_investigation_binds_delayed_reservation_before_backend_advance() -> None:
    env = DatacenterEnvironment()
    env.reset(_scenario(), seed=42)
    try:
        env.step(Action(tool_calls=[ToolCall(
            name="reserve_gpu_capacity",
            args={"gpu_units": 2.0, "duration_ticks": 2},
            call_id="delayed-reserve",
            consumes_evidence_ids=["known-input"],
        )]))
        _, [receipt] = env.execute_investigation(Action(tool_calls=[]))
        assert env.tick == 1
        assert receipt.ok
        result = env.step(Action(tool_calls=[]))
        [effect] = [event for event in result.info.realized_events
                    if event.get("tool_name") == "reserve_gpu_capacity"]
        assert effect["call_id"] == "delayed-reserve"
        assert receipt.evidence_id in effect["evidence_ids"]
        assert env.ground_truth()["control_summary"]["effect_tool_ticks"] == {
            "reserve_gpu_capacity": [1]
        }
        [evidence] = [item for item in env.evidence.items_by_kind("tool_call")
                      if item.evidence_id == receipt.evidence_id]
        assert evidence.payload["consumes_evidence_ids"] == ["known-input"]
    finally:
        env.close()
