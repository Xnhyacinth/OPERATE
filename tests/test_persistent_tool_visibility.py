import json
from types import SimpleNamespace

import pytest

from baselines.llm_agent import LLMAgent, LLMConfig
from core import Action, ToolCall


@pytest.mark.parametrize("realtime", [False, True])
def test_persistent_decision_receives_complete_investigation_results(
    monkeypatch, realtime,
):
    agent = LLMAgent(LLMConfig(
        model_context_window_tokens=192000,
        model_max_output_tokens=32768,
        persistent_context_max_chars=128000,
        provider_failure_policy="abort",
    ))
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    monkeypatch.setattr(agent, "_make_client", lambda _: object())
    tools = [{"type": "function", "function": {
        "name": "wait", "parameters": {"type": "object", "properties": {}},
    }}]
    agent.reset(SimpleNamespace(
        get_tool_specs=lambda: tools,
        readonly_tool_names=lambda: set(),
        budget=SimpleNamespace(max_tool_calls_per_tick=8, max_cost_units_per_tick=8),
    ), {"horizon_ticks": 3, "tick_minutes": 1}, 42)
    captured = []

    def invoke(messages):
        captured.append(json.loads(messages[-1]["content"]))
        return Action(tool_calls=[ToolCall(name="wait")], dominant="wait")

    monkeypatch.setattr(agent, "_invoke_decision_provider", invoke)
    results = [{
        "name": "query_job_queue", "ok": True,
        "call_id": f"call-{i}", "evidence_id": f"ev-{i}",
        "payload": {"jobs": [{"job_id": f"job-{i}-{j}", "remaining_ticks": j}
                             for j in range(30)]},
    } for i in range(6)]
    observation = {"tick": 1, "__within_tick_tool_results__": results}
    if realtime:
        observation["__realtime_event__"] = {
            "kind": "tool_result", "event_id": "realtime-1",
            "payload": {"tool_results": results},
        }
    agent.act(observation, tools)
    context = captured[-1]["event_context"]
    visible = (context["realtime_event"]["payload"]["tool_results"]
               if realtime else context["within_tick_tool_results"])
    assert [row["call_id"] for row in visible] == [row["call_id"] for row in results]
    assert [row["payload"] for row in visible] == [row["payload"] for row in results]


def test_budget_compaction_preserves_every_tool_result_identity():
    results = [{"name": "inspect", "ok": True, "call_id": f"call-{i}",
                "payload": {"detail": "x" * 10000}} for i in range(6)]
    rendered = LLMAgent._serialize_prompt_body(
        {"tick": 1, "within_tick_tool_results": results},
        max_chars=2000, include_cost_units=True,
    )
    assert len(rendered) <= 2000
    assert [r["call_id"] for r in json.loads(rendered)["within_tick_tool_results"]] == [
        r["call_id"] for r in results
    ]
