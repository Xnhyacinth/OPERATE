"""Reasoning controls must reach each Chat Completions wire request."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from baselines.llm_agent import LLMAgent, LLMConfig

httpx = pytest.importorskip("httpx")
openai = pytest.importorskip("openai")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Wait",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


@pytest.mark.parametrize(
    ("provider", "effort_format", "thinking_type", "effort_field"),
    [
        ("openai_compatible", "native", "enabled", "reasoning_effort"),
        ("openai_compatible", "auto", None, "reasoning"),
        ("openai", "openrouter", None, "reasoning"),
        ("openai", "auto", None, "reasoning_effort"),
    ],
)
def test_reasoning_profile_reaches_decision_and_repair_wire(
    monkeypatch, provider, effort_format, thinking_type, effort_field
):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        message = {"role": "assistant", "content": "Wait for the next event."}
        if len(requests) == 2:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_wait",
                        "type": "function",
                        "function": {"name": "wait", "arguments": "{}"},
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "hy3-ioa",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if len(requests) == 2 else "stop",
                        "message": message,
                    }
                ],
            },
        )

    sdk_class = openai.OpenAI
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: sdk_class(
            **kwargs, http_client=httpx.Client(transport=transport)
        ),
    )
    monkeypatch.setenv("OPERATE_WIRE_TEST_KEY", "test-key")
    config = LLMConfig(
        provider=provider,
        model="hy3-ioa",
        base_url="https://copilot.tencent.com/v2",
        api_key_env="OPERATE_WIRE_TEST_KEY",
        api_mode="chat_completions",
        interaction_mode="logical_stateless",
        reasoning_effort="high",
        reasoning_effort_format=effort_format,
        thinking_type=thinking_type,
        provider_failure_policy="abort",
    )
    agent = LLMAgent(config)
    env = SimpleNamespace(
        get_tool_specs=lambda: TOOLS,
        readonly_tool_names=lambda: set(),
        budget=SimpleNamespace(max_tool_calls_per_tick=6, max_cost_units_per_tick=3.0),
    )
    agent.reset(
        env,
        {
            "domain": "logistics",
            "family": "wire_test",
            "horizon_ticks": 1,
            "tick_minutes": 1,
        },
        seed=42,
    )

    action = agent.act({"tick": 0}, TOOLS)

    assert action.dominant == "wait"
    assert len(requests) == 2
    for payload in requests:
        expected = "high" if effort_field == "reasoning_effort" else {"effort": "high"}
        assert payload.get(effort_field) == expected
        omitted_field = (
            "reasoning" if effort_field == "reasoning_effort" else "reasoning_effort"
        )
        assert omitted_field not in payload
        if thinking_type is None:
            assert "thinking" not in payload
        else:
            assert payload["thinking"] == {"type": thinking_type}
    envelopes = agent.get_interaction_stats()["provider_request_records"]
    for request in envelopes:
        assert request["envelope"]["reasoning_effort_format"] == effort_format
        assert request["envelope"]["thinking_type"] == thinking_type


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"reasoning_effort_format": "flat_typo"}, "reasoning_effort_format"),
        ({"thinking_type": "auto"}, "thinking_type"),
        ({"api_mode": "responses", "thinking_type": "enabled"}, "Chat Completions"),
        (
            {"provider": "anthropic", "reasoning_effort_format": "native"},
            "Chat Completions",
        ),
    ],
)
def test_unsupported_reasoning_controls_fail_before_transport(overrides, error):
    config = LLMConfig(interaction_mode="logical_stateless", **overrides)
    agent = LLMAgent(config)
    env = SimpleNamespace(
        get_tool_specs=lambda: TOOLS,
        readonly_tool_names=lambda: set(),
        budget=SimpleNamespace(max_tool_calls_per_tick=6, max_cost_units_per_tick=3.0),
    )
    with pytest.raises(ValueError, match=error):
        agent.reset(env, {"horizon_ticks": 1, "tick_minutes": 1}, seed=42)
