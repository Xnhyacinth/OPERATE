"""Offline provider-boundary regressions using the installed SDK transport."""

import json
from datetime import UTC, datetime
from email.utils import format_datetime

import httpx2
import pytest
from openai import OpenAI

from baselines.llm_agent import (
    LLMAgent,
    LLMConfig,
    ProviderRetryBudgetExhaustedError,
    RealtimeTurnCanceledError,
    classify_provider_error,
)


def _agent(client, **config):
    agent = LLMAgent(LLMConfig(
        model="fixture", interaction_mode="logical_persistent",
        model_context_window_tokens=100_000, model_max_output_tokens=4096,
        stream_chat_completions=True, provider_failure_policy="abort",
        max_consecutive_provider_failures=1, **config,
    ))
    agent._client = client
    agent._has_api_key = True
    agent._system_prompt = "offline fixture"
    agent._tool_specs = [{
        "type": "function", "function": {"name": "wait", "parameters": {}},
    }]
    return agent


def _client(respond):
    return OpenAI(
        api_key="fixture", base_url="https://example.test/v1", max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    )


def _frame(*, finish_reason="tool_calls", model="fixture", arguments="{}", name="wait"):
    return "data: " + json.dumps({
        "id": "fixture-response", "model": model, "choices": [{
            "index": 0, "finish_reason": finish_reason, "delta": {
                "tool_calls": [{
                    "index": 0, "id": "fixture-call", "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }],
            },
        }],
    }) + "\n\n"


def _success():
    return httpx2.Response(
        200, text=_frame() + "data: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


@pytest.mark.parametrize("header_kind", ["seconds", "http_date"])
def test_server_retry_after_is_a_lower_bound(monkeypatch, header_kind):
    now = 1_800_000_000.0
    elapsed = [0.0]
    monkeypatch.setattr("baselines.llm_agent.time.time", lambda: now + elapsed[0])
    monkeypatch.setattr(
        "baselines.llm_agent.time.sleep",
        lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
    )
    requests = []

    def respond(request):
        requests.append((elapsed[0], request.content))
        if len(requests) == 1:
            retry_after = "120" if header_kind == "seconds" else format_datetime(
                datetime.fromtimestamp(now + 120, tz=UTC), usegmt=True,
            )
            return httpx2.Response(
                429, json={"error": {"message": "rate limited"}},
                headers={"retry-after": retry_after},
            )
        return _success()

    with _client(respond) as client:
        agent = _agent(client)
        action = agent.act({"tick": 7}, agent._tool_specs)
    assert action.dominant == "wait"
    assert len(requests) == 2
    assert requests[1][0] >= 120
    assert requests[0][1] == requests[1][1]


def test_disconnect_after_terminal_frame_preserves_completed_action(monkeypatch):
    requests = []
    closed = []

    class TailDisconnect(httpx2.SyncByteStream):
        def __iter__(self):
            yield _frame().encode()
            raise httpx2.RemoteProtocolError("usage trailer disconnected")

        def close(self):
            closed.append(True)

    def respond(request):
        requests.append(request.content)
        return httpx2.Response(
            200, stream=TailDisconnect(),
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr("baselines.llm_agent.time.sleep", lambda _: None)
    with _client(respond) as client:
        agent = _agent(client)
        action = agent.act({"tick": 7}, agent._tool_specs)
    assert [call.name for call in action.tool_calls] == ["wait"]
    assert len(requests) == 1
    assert closed
    stats = agent.get_interaction_stats()
    assert stats["retry_attempts_total"] == 0
    response = stats["provider_response_records"][0]["response"]
    assert response["status"] == "success"
    assert response["provider_metadata"]["finish_reason"] == "tool_calls"
    assert response["provider_metadata"]["stream_tail_error"]["reason"] == "provider_transport_error"


def test_observed_model_mismatch_aborts_before_action_or_retry(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request.content)
        return httpx2.Response(
            200, text=_frame(model="unexpected-model") + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr("baselines.llm_agent.time.sleep", lambda _: None)
    with _client(respond) as client:
        agent = _agent(client)
        with pytest.raises(Exception, match="model identity mismatch") as caught:
            agent.act({"tick": 7}, agent._tool_specs)
    assert classify_provider_error(caught.value) == "provider_model_identity_mismatch"
    assert len(requests) == 1
    assert agent.get_interaction_stats()["retry_attempts_total"] == 0
    assert agent.get_last_provider_outcome()["status"] == "failed"
    assert not any(row["role"] == "assistant" for row in agent.get_session_ledger())
    response = agent.get_interaction_stats()["provider_response_records"][0]["response"]
    assert response["error_reason"] == "provider_model_identity_mismatch"
    assert response["model_identity_closure"]["closure"] == "mismatch"


def test_attempt_budget_stops_recovery_and_keeps_every_wire_record(monkeypatch):
    requests = []
    monkeypatch.setattr("baselines.llm_agent.time.sleep", lambda _: None)

    def respond(request):
        requests.append(request.content)
        raise httpx2.ConnectError("offline outage")

    with _client(respond) as client:
        agent = _agent(client, provider_retry_max_attempts=3)
        with pytest.raises(ProviderRetryBudgetExhaustedError) as caught:
            agent.act({"tick": 7}, agent._tool_specs)
    assert caught.value.budget_reason == "max_attempts"
    assert caught.value.attempts == 3
    assert len(requests) == 3
    stats = agent.get_interaction_stats()
    assert len(stats["provider_request_records"]) == 3
    assert len(stats["provider_response_records"]) == 3
    assert stats["ticks_wait_fallback"] == 0
    for index, record in enumerate(stats["provider_request_records"], start=1):
        budget = record["envelope"]["provider_retry_budget"]
        assert budget["attempt"] == index
        assert budget["max_attempts"] == 3


def test_stream_generation_obeys_recovery_deadline(monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr("baselines.llm_agent.time.monotonic", lambda: elapsed[0])
    requests = []

    class SlowStream(httpx2.SyncByteStream):
        def __iter__(self):
            yield _frame(finish_reason=None, arguments="{").encode()
            elapsed[0] = 11.0
            yield _frame().encode()

    def respond(request):
        requests.append(request)
        return httpx2.Response(
            200, stream=SlowStream(), headers={"content-type": "text/event-stream"},
        )

    with _client(respond) as client:
        agent = _agent(client, provider_retry_max_elapsed_s=10.0)
        with pytest.raises(ProviderRetryBudgetExhaustedError) as caught:
            agent.act({"tick": 7}, agent._tool_specs)
    assert caught.value.budget_reason == "max_elapsed"
    assert len(requests) == 1
    assert requests[0].extensions["timeout"]["read"] <= 10.0
    assert not any(row["role"] == "assistant" for row in agent.get_session_ledger())


def test_recovery_backoff_still_honors_realtime_cancel(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        raise httpx2.ConnectError("offline outage")

    with _client(respond) as client:
        agent = _agent(client)
        monkeypatch.setattr(
            "baselines.llm_agent.time.sleep",
            lambda _: agent.cancel_realtime_turn(turn_id="turn-1", reason="superseded"),
        )
        with pytest.raises(RealtimeTurnCanceledError):
            agent.act({"tick": 7, "__decision_epoch__": {"turn_id": "turn-1"}}, agent._tool_specs)
    assert len(requests) == 1


def test_quota_error_in_stream_tail_is_not_ignored(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx2.Response(
            200, text=_frame() + 'data: {"error":{"code":6004,"message":"quota exhausted"}}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr("baselines.llm_agent.time.sleep", lambda _: None)
    with _client(respond) as client:
        agent = _agent(client)
        with pytest.raises(Exception) as caught:
            agent.act({"tick": 7}, agent._tool_specs)
    assert classify_provider_error(caught.value) == "provider_quota_exhausted"
    assert len(requests) == 1


def test_long_episode_recovers_pending_request_without_repeating_prior_turn(monkeypatch):
    requests = []

    def respond(request):
        requests.append(request.content)
        if 2 <= len(requests) <= 6:
            raise httpx2.ConnectError("offline outage")
        return _success()

    monkeypatch.setattr("baselines.llm_agent.time.sleep", lambda _: None)
    with _client(respond) as client:
        agent = _agent(client)
        agent.config.provider_retry_max_attempts = 6
        agent.config.provider_retry_max_elapsed_s = 1800.0
        agent.act({"tick": 0}, agent._tool_specs)
        recovered = agent.act({"tick": 7}, agent._tool_specs)
    assert recovered.dominant == "wait"
    assert len(requests) == 7
    assert all(body == requests[1] for body in requests[2:])
    ledger = agent.get_session_ledger()
    assert sum(row["role"] == "assistant" for row in ledger) == 2
    assert sum(row["role"] == "user" for row in ledger) == 2
    records = agent.get_interaction_stats()["provider_request_records"]
    assert [row["envelope"]["provider_retry_index"] for row in records[1:]] == list(range(6))
    assert all(row["envelope"]["retry_of_request_sequence"] == 2 for row in records[2:])


def test_recovery_window_exhaustion_defers_without_an_early_request(monkeypatch):
    elapsed = [0.0]
    now = 1_800_000_000.0
    monkeypatch.setattr("baselines.llm_agent.time.monotonic", lambda: elapsed[0])
    monkeypatch.setattr("baselines.llm_agent.time.time", lambda: now + elapsed[0])
    monkeypatch.setattr(
        "baselines.llm_agent.time.sleep",
        lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
    )
    requests = []

    def respond(request):
        requests.append(request.content)
        raise httpx2.ConnectError("offline outage")

    with _client(respond) as client:
        agent = _agent(client)
        agent.config.provider_retry_max_attempts = 10
        agent.config.provider_retry_max_elapsed_s = 12.0
        with pytest.raises(Exception) as caught:
            agent.act({"tick": 7}, agent._tool_specs)
    assert len(requests) == 2
    assert elapsed[0] == 5.0
    assert caught.value.budget_reason == "max_elapsed"
    assert classify_provider_error(caught.value) == "provider_transport_error"
    assert datetime.fromisoformat(caught.value.retry_at).timestamp() >= now + 15
    assert not any(row["role"] == "assistant" for row in agent.get_session_ledger())


def test_json_resume_state_continues_without_losing_history_or_request_ids():
    requests = []

    def respond(request):
        requests.append(request.content)
        return _success()

    with _client(respond) as client:
        original = _agent(client)
        original.act({"tick": 0}, original._tool_specs)
        state = json.loads(json.dumps(original.export_resume_state()))
        expected = original.act({"tick": 7}, original._tool_specs)
        resumed = _agent(client)
        resumed.import_resume_state(state)
        actual = resumed.act({"tick": 7}, resumed._tool_specs)
    assert actual.to_dict() == expected.to_dict()
    assert requests[1] == requests[2]
    assert len(resumed.get_interaction_stats()["provider_request_records"]) == 2
    assert resumed.get_session_ledger() == original.get_session_ledger()
    assert '"config"' not in json.dumps(state)
    assert '"api_key"' not in json.dumps(state)


def test_resume_state_rejects_a_different_request_binding():
    with _client(lambda _: _success()) as client:
        original = _agent(client)
        original.act({"tick": 0}, original._tool_specs)
        state = original.export_resume_state()
        resumed = _agent(client, temperature=0.9)
        before = resumed.snapshot_behavioral_state()
        with pytest.raises(ValueError, match="binding"):
            resumed.import_resume_state(state)
        assert resumed.snapshot_behavioral_state() == before


def test_resume_plan_receipt_resolves_every_alias_after_json_roundtrip():
    def respond(_request):
        return httpx2.Response(
            200, text=_frame(name="commit_to_plan", arguments='{"plan_id":"p","horizon_ticks":3}') + "data: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    specs = [{"type": "function", "function": {"name": "commit_to_plan", "parameters": {}}}]
    with _client(respond) as client:
        original = _agent(client)
        original._tool_specs = specs
        action = original.act({"tick": 0}, specs)
        state = json.loads(json.dumps(original.export_resume_state()))
        assert len(state["pending_plans"]) == 1
        assert len(state["pending_plans"][0]["aliases"]) == 2
        resumed = _agent(client)
        resumed._tool_specs = specs
        resumed.import_resume_state(state)
        call = action.tool_calls[0]
        resumed.observe_transition({"tick": 1, "__last_tool_results__": [{
            "name": "commit_to_plan", "ok": True, "call_id": call.call_id,
            "idempotency_key": call.idempotency_key, "payload": {"plan_id": "p"},
        }]})
    after = resumed.snapshot_behavioral_state()
    assert after["pending_plan_calls"] == {}
    assert after["active_plan"]["plan_id"] == "p"
    assert resumed.get_interaction_stats()["plan_commits_confirmed"] == 1


def test_resume_snapshot_is_rejected_while_provider_request_is_in_flight():
    def respond(_request):
        with pytest.raises(ValueError, match="settled"):
            agent.export_resume_state()
        return _success()

    with _client(respond) as client:
        agent = _agent(client)
        agent.act({"tick": 0}, agent._tool_specs)
