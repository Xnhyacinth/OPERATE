import json
from types import SimpleNamespace

import pytest

from baselines.llm_agent import LLMAgent, LLMConfig, RealtimeTurnCanceledError
from core.provider_request_limiter import ProviderRequestLimiter


@pytest.mark.parametrize("before_reservation", [False, True])
def test_canceled_turn_never_sends_a_request_after_waiting_for_quota(
    monkeypatch, tmp_path, before_reservation,
):
    agent = LLMAgent(LLMConfig(
        stream_chat_completions=True, provider_failure_policy="abort",
        model_context_window_tokens=192000, model_max_output_tokens=32768,
        provider_rpm_limit=1, provider_rate_limit_scope="fixture",
    ))
    agent._has_api_key = True
    agent._system_prompt = "fixture"
    agent._tool_specs = [{"function": {"name": "wait", "parameters": {}}}]
    agent._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **_: pytest.fail("canceled request reached provider"),
    )))
    agent._begin_realtime_turn("turn-1")
    if before_reservation:
        agent.cancel_realtime_turn(turn_id="turn-1", reason="superseded")
    ProviderRequestLimiter(
        rpm_limit=1, scope="fixture", state_dir=tmp_path, now=lambda: 1000,
    ).acquire()

    def limiter(**kwargs):
        kwargs.setdefault("sleep", lambda _: agent.cancel_realtime_turn(
            turn_id="turn-1", reason="superseded",
        ))
        return ProviderRequestLimiter(**kwargs, state_dir=tmp_path, now=lambda: 1000)

    monkeypatch.setattr("baselines.llm_agent.ProviderRequestLimiter", limiter)
    monkeypatch.setattr("baselines.llm_agent.time.sleep", lambda _: agent.cancel_realtime_turn(
        turn_id="turn-1", reason="superseded",
    ))
    with pytest.raises(RealtimeTurnCanceledError):
        agent.act({"tick": 1, "__decision_epoch__": {"turn_id": "turn-1"}}, [])
    stats = agent.get_interaction_stats()
    if before_reservation:
        assert stats["provider_request_records"] == []
    else:
        assert len(stats["provider_request_records"]) == 1
        audit = stats["provider_request_records"][0]["envelope"]["provider_rate_limit"]
        assert audit["status"] == "wait_interrupted"
        assert audit["scheduled_wait_seconds"] == 60
        assert len(stats["provider_response_records"]) == 1
    state = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert state["day_count"] == (1 if before_reservation else 2)
