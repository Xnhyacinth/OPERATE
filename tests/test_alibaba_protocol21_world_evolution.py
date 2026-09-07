from __future__ import annotations

from pathlib import Path

import yaml

from core import Action, ToolCall
from domains.datacenter.adapter import DatacenterEnvironment

REPO_ROOT = Path(__file__).resolve().parents[1]
BASIC_SCENARIO = (
    "scenarios/"
    "datacenter/gpu_cluster_queue_control/time_pressure/basic/"
    "alibaba_gpu_w052_377_382_basic.yaml"
)
RESERVATION_SCENARIO = (
    "scenarios/datacenter/gpu_cluster_sla_control/time_pressure/basic/"
    "datacenter__gpu_cluster_sla_control__time_pressure__medium__"
    "alibaba_gpu_w953_6918_6923_medium__c96f0427__relabel_v1.yaml"
)
ACTIVE_MEDIUM_SCENARIO = (
    "scenarios/datacenter/gpu_cluster_queue_control/"
    "time_pressure/medium/datacenter__gpu_cluster_queue_control__time_pressure__"
    "medium__alibaba_gpu_w904_6531_6537_medium__6715e80d__relabel_v1__"
    "b38f3183__relabel_v1.yaml"
)


def _scenario(path: str) -> dict:
    loaded = yaml.safe_load((REPO_ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _records(result) -> list[dict]:
    return list(result.info.extra.get("world_evolution_records") or [])


def test_source_arrivals_and_completions_are_separate_runtime_origins() -> None:
    env = DatacenterEnvironment()
    env.reset(_scenario(BASIC_SCENARIO), seed=94)

    results = [
        env.step(
            Action(
                tool_calls=[
                    ToolCall(
                        name="set_queue_policy",
                        args={"policy": "shortest_job_first"},
                        idempotency_key="policy-change",
                    )
                ]
            )
        )
    ]
    results.extend(
        env.step(Action(tool_calls=[ToolCall(name="wait")]))
        for _ in range(3)
    )
    records = [record for result in results for record in _records(result)]
    assert not any(
        record["event_type"] == "job_arrival"
        and record["applied_tick"] == 0
        for record in records
    )
    arrivals = [
        record
        for record in records
        if record["event_type"] == "job_arrival"
        and record["applied_tick"] > 0
    ]
    completions = [
        record for record in records if record["event_type"] == "job_completed"
    ]

    assert arrivals
    assert all(record["origin"] == "source_schedule" for record in arrivals)
    assert all(record["changed_state_fields"] for record in arrivals)
    assert all(record["materiality"]["threshold"] == 1 for record in arrivals)
    assert all(record["materiality"]["passed"] is True for record in arrivals)
    assert completions
    assert all(
        record["origin"] == "endogenous_completion"
        for record in completions
    )
    assert not any(
        record["material_exogenous"] is True for record in completions
    )
    env.close()


def test_queue_policy_effect_keeps_call_and_evidence_lineage() -> None:
    env = DatacenterEnvironment()
    env.reset(_scenario(BASIC_SCENARIO), seed=94)

    result = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="set_queue_policy",
                    args={"policy": "shortest_job_first"},
                    idempotency_key="policy-effect",
                )
            ]
        )
    )
    effect = next(
        record
        for record in _records(result)
        if record["origin"] == "agent_caused"
    )

    assert effect["tool_name"] == "set_queue_policy"
    assert effect["call_id"]
    assert effect["before_state_digest"] != effect["after_state_digest"]
    assert effect["changed_state_fields"]
    assert effect["evidence_ids"]
    assert effect["action_to_outcome_edge"] == {
        "source": f"call:{effect['call_id']}",
        "target": f"outcome:{effect['event_id']}",
        "kind": "action_to_outcome",
    }
    env.close()


def test_delayed_reservation_is_not_effective_until_materialization() -> None:
    env = DatacenterEnvironment()
    env.reset(_scenario(RESERVATION_SCENARIO), seed=995)
    # The same source window was calibrated to basic on promotion. Exercise
    # deferred receipt semantics explicitly without changing its source jobs.
    env._tools.get("reserve_gpu_capacity").delay_ticks = 1

    pending = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="reserve_gpu_capacity",
                    args={"gpu_units": 2.0, "duration_ticks": 2},
                    idempotency_key="delayed-reservation",
                )
            ]
        )
    )
    assert not any(
        record["origin"] == "agent_caused" for record in _records(pending)
    )

    materialized = env.step(Action(tool_calls=[ToolCall(name="wait")]))
    effect = next(
        record
        for record in _records(materialized)
        if record["origin"] == "agent_caused"
        and record["tool_name"] == "reserve_gpu_capacity"
    )
    assert effect["call_id"]
    assert effect["outcome_tick"] == 1
    assert "reserved_gpu_units" in effect["changed_state_fields"]
    env.close()


def test_noop_queue_policy_does_not_create_passed_action_effect() -> None:
    env = DatacenterEnvironment()
    env.reset(_scenario(BASIC_SCENARIO), seed=94)

    result = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="set_queue_policy",
                    args={"policy": "fifo"},
                    idempotency_key="policy-noop",
                )
            ]
        )
    )

    assert not any(
        record["origin"] == "agent_caused" for record in _records(result)
    )
    env.close()


def test_capacity_reduction_is_a_material_declared_event_with_response_window() -> None:
    scenario = _scenario(ACTIVE_MEDIUM_SCENARIO)
    scenario["perturbations"] = [
        {
            "kind": "capacity_reduction",
            "trigger_tick": 4,
            "duration_ticks": 2,
            "hidden": False,
            "target": {"resource": "shared_gpu_service_maintenance"},
            "intensity": 0.2,
        }
    ]
    env = DatacenterEnvironment()
    env.reset(scenario, seed=995)

    results = [
        env.step(Action(tool_calls=[ToolCall(name="wait")]))
        for _ in range(5)
    ]
    records = [record for result in results for record in _records(result)]
    event = next(
        record for record in records if record["event_type"] == "capacity_reduction"
    )

    assert event["origin"] == "declared_perturbation"
    assert event["event_class"] == "alarm"
    assert event["decision_required"] is True
    assert event["material_exogenous"] is True
    assert event["changed_state_fields"]
    assert event["materiality"]["passed"] is True
    assert event["before_state_digest"] != event["after_state_digest"]
    assert event["response_opportunity_tick"] == 5
    assert event["terminal_response_window_missing"] is False
    env.close()
