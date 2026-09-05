"""Regression coverage for long operational sessions, without provider calls."""

import json
from copy import deepcopy

import pytest

from baselines.llm_agent import LLMConfig, LLMAgent
from core import Action, ToolCall


def agent():
    return LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            model="hy3-ioa",
            max_tokens=32768,
            model_context_window_tokens=192000,
            model_max_output_tokens=64000,
            persistent_context_max_chars=512000,
            provider_failure_policy="abort",
        )
    )


def test_persistent_observation_uses_session_budget(monkeypatch):
    subject = agent()
    body = {"native_state": {f"control_{i}": i for i in range(1000)}}
    monkeypatch.setattr(LLMAgent, "_observation_summary", lambda *args: body)
    monkeypatch.setattr(
        subject,
        "_call_openai_compatible",
        lambda messages: Action(tool_calls=[ToolCall(name="wait")], dominant="wait"),
    )
    subject._call_llm({"tick": 0, "entities": {}})
    request = subject.get_interaction_stats()["provider_request_records"][0]
    payload = json.loads(request["envelope"]["messages"][-1]["content"])
    assert payload["event_context"]["native_state"] == body["native_state"]


def test_small_session_reserves_briefing_and_memory_before_observation(monkeypatch):
    subject = agent()
    subject.config.persistent_context_max_chars = 16000
    subject._system_prompt = "mission " * 700
    body = {
        "ready_operations": {
            f"op-{i}": {
                "machine_id": f"machine-{i}",
                "duration": 20,
                "metadata": "x" * 600,
            }
            for i in range(20)
        }
    }
    monkeypatch.setattr(LLMAgent, "_observation_summary", lambda *args: deepcopy(body))
    monkeypatch.setattr(
        subject,
        "_call_openai_compatible",
        lambda messages: Action(tool_calls=[ToolCall(name="wait")], dominant="wait"),
    )
    subject._call_llm({"tick": 0, "entities": {}})
    request = subject.get_interaction_stats()["provider_request_records"][0]
    messages = request["envelope"]["messages"]
    assert sum(len(m["content"]) for m in messages) <= 16000
    operations = json.loads(messages[-1]["content"])["event_context"][
        "ready_operations"
    ]
    assert set(operations) == set(body["ready_operations"])
    assert all(
        operations[key]["machine_id"] == row["machine_id"]
        and operations[key]["duration"] == row["duration"]
        for key, row in body["ready_operations"].items()
    )


@pytest.mark.parametrize(
    "bucket,detail", [("confirmed_facts", "x"), ("open_obligations", "查")]
)
def test_provider_cap_projects_latest_memory_without_losing_ledger(bucket, detail):
    subject = agent()
    subject._system_prompt = "mission"
    subject._structured_memory[bucket] = [
        {
            "id": f"fact-{i}",
            "status": "pending",
            "call_id": f"call-{i}",
            "detail": detail * 6000,
        }
        for i in range(32)
    ]
    original_memory = deepcopy(subject._structured_memory)
    payload = subject._fit_persistent_event_memory(
        event={"kind": "agent_scheduled_review"},
        event_context={"native_state": {"load": 20}},
    )
    subject._append_persistent_message(
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
    )
    original_ledger = deepcopy(subject.get_session_ledger())
    kwargs = dict(
        tools=[],
        max_tokens=32768,
        effective_tool_choice=None,
        effective_wire_stream=True,
        effective_temperature=0.7,
    )
    messages, projection = subject._provider_cap_aware_projection(
        messages=subject._persistent_provider_messages(), **kwargs
    )
    assert (
        subject._request_budget_audit(messages=messages, **kwargs)["status"]
        == "within_budget"
    )
    assert subject.get_session_ledger() == original_ledger
    assert subject._structured_memory == original_memory
    latest = json.loads(messages[-1]["content"])
    assert latest["event_context"]["native_state"] == {"load": 20}
    assert [
        row["id"] for row in latest["event_context"]["structured_memory"][bucket]
    ] == [f"fact-{i}" for i in range(32)]
    assert [
        row["call_id"] for row in latest["event_context"]["structured_memory"][bucket]
    ] == [f"call-{i}" for i in range(32)]
    assert all(
        row["status"] == "pending"
        for row in latest["event_context"]["structured_memory"][bucket]
    )
    assert projection["provider_cap_applied"]


def test_strict_local_prompt_failure_is_not_wait(monkeypatch):
    subject = agent()
    subject._has_api_key = True

    def fail(*args, **kwargs):
        raise ValueError(
            "mandatory prompt state exceeds max_chars; refusing to omit action-critical fields"
        )

    monkeypatch.setattr(subject, "_call_llm", fail)
    with pytest.raises(ValueError, match="mandatory prompt"):
        subject.act({"tick": 0}, [])
    assert subject.get_interaction_stats()["ticks_wait_fallback"] == 0
