from types import SimpleNamespace

import httpx
import httpx2
import pytest
from openai import APIConnectionError, APITimeoutError

from baselines.llm_agent import LLMAgent, LLMConfig, classify_provider_error


@pytest.mark.parametrize("error", [
    ConnectionError("connection dropped"), TimeoutError("timed out"),
    APIConnectionError(request=httpx.Request("POST", "https://example.test")),
    APITimeoutError(request=httpx.Request("POST", "https://example.test")),
    httpx.RemoteProtocolError("peer closed while streaming function_call"),
    httpx.ReadError("connection reset"),
    httpx.ReadTimeout("read timed out"),
    httpx2.RemoteProtocolError("peer closed while streaming function_call"),
    httpx2.ReadError("connection reset"),
    httpx2.ReadTimeout("read timed out"),
])
def test_network_errors_are_transport_failures(error):
    assert classify_provider_error(error) == "provider_transport_error"


@pytest.mark.parametrize("transport", [httpx, httpx2], ids=["httpx", "httpx2"])
def test_streaming_disconnect_from_real_sdk_retries_identical_request(monkeypatch, transport):
    import json
    from openai import OpenAI

    requests = []
    partial = {"id": "rsp", "model": "fixture", "choices": [{
        "index": 0, "finish_reason": None, "delta": {"tool_calls": [{
            "index": 0, "id": "call", "type": "function",
            "function": {"name": "wait", "arguments": "{}"},
        }]},
    }]}

    class InterruptedStream(transport.SyncByteStream):
        def __iter__(self):
            yield ("data: " + json.dumps(partial) + "\n\n").encode()
            raise transport.RemoteProtocolError("peer closed connection")

    def respond(request):
        requests.append(request.content)
        if len(requests) == 1:
            return transport.Response(200, stream=InterruptedStream(), headers={"content-type": "text/event-stream"})
        complete = json.loads(json.dumps(partial))
        complete["choices"][0]["finish_reason"] = "tool_calls"
        return transport.Response(200, text="data: " + json.dumps(complete) + "\n\ndata: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})

    agent = LLMAgent(LLMConfig(
        model="fixture", interaction_mode="logical_stateless",
        stream_chat_completions=True, provider_failure_policy="abort",
    ))
    agent._has_api_key = True
    agent._system_prompt = "fixture"
    agent._tool_specs = [{"type": "function", "function": {"name": "wait", "parameters": {}}}]
    with OpenAI(api_key="fixture", base_url="https://example.test/v1", max_retries=0,
                http_client=transport.Client(transport=transport.MockTransport(respond))) as client:
        agent._client = client
        monkeypatch.setattr(agent, "_sleep_before_provider_retry", lambda _: None)
        assert agent.act({"tick": 0}, agent._tool_specs).dominant == "wait"
    assert len(requests) == 2 and requests[0] == requests[1]
    assert [row["response"]["status"] for row in agent.get_interaction_stats()["provider_response_records"]] == ["failed", "success"]


def test_abort_profile_refuses_missing_credentials_before_an_episode(monkeypatch):
    monkeypatch.delenv("OPERATE_TEST_MISSING_KEY", raising=False)
    agent = LLMAgent(LLMConfig(
        interaction_mode="logical_stateless", provider_failure_policy="abort",
        api_key_env="OPERATE_TEST_MISSING_KEY",
    ))
    env = SimpleNamespace(
        get_tool_specs=lambda: [], readonly_tool_names=lambda: set(),
        budget=SimpleNamespace(max_tool_calls_per_tick=6, max_cost_units_per_tick=3),
    )
    with pytest.raises(ValueError, match="credential.*OPERATE_TEST_MISSING_KEY"):
        agent.reset(env, {"horizon_ticks": 2, "tick_minutes": 1}, 42)


def test_abort_profile_never_executes_a_stream_without_terminal_marker(monkeypatch):
    agent = LLMAgent(LLMConfig(
        interaction_mode="logical_stateless", stream_chat_completions=True,
        provider_failure_policy="abort",
    ))
    sent = []
    closed = []

    def create(**kwargs):
        sent.append(kwargs)

        class Stream:
            def __iter__(self):
                yield SimpleNamespace(choices=[SimpleNamespace(
                    finish_reason=None if len(sent) == 1 else "tool_calls",
                    delta=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                        index=0, function=SimpleNamespace(name="wait", arguments="{}"),
                    )]),
                )])

            def close(self):
                closed.append(True)

        return Stream()

    agent._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    agent._has_api_key = True
    agent._system_prompt = "fixture"
    agent._tool_specs = [{"function": {"name": "wait", "parameters": {}}}]
    monkeypatch.setattr(agent, "_sleep_before_provider_retry", lambda _: None)
    action = agent.act({"tick": 0}, agent._tool_specs)
    assert len(sent) == 2
    assert sent[0] == sent[1]
    assert len(closed) == 2
    assert action.dominant == "wait"
    responses = agent.get_interaction_stats()["provider_response_records"]
    assert [r["response"]["status"] for r in responses] == ["failed", "success"]
    assert responses[0]["response"]["error_reason"] == "provider_transport_error"
