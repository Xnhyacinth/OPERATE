import pytest

from baselines.llm_agent import LLMAgent
from core import Action, ToolCall


@pytest.mark.parametrize("identity", [
    {"call_id": "old-call", "idempotency_key": "old-key"},
    {"call_id": "new-call", "idempotency_key": "old-key"},
    {"call_id": "old-call", "idempotency_key": "new-key"},
])
def test_conflicting_receipt_does_not_activate_or_consume_pending_plan(identity):
    agent = LLMAgent()
    agent._record_pending_plans(Action(tool_calls=[ToolCall(
        name="commit_to_plan", call_id="new-call", idempotency_key="new-key",
        args={"plan_id": "reused-plan-name", "review_after_ticks": 3},
    )]), {"tick": 2})
    agent.observe_transition({"tick": 3, "__last_tool_results__": [{
        "name": "commit_to_plan", "ok": True,
        "payload": {"plan_id": "reused-plan-name", "ack": True}, **identity,
    }]})
    assert agent._active_plan is None
    assert "new-call" in agent._pending_plan_calls
    assert agent.get_interaction_stats()["plan_commits_confirmed"] == 0


def test_acknowledged_plan_preserves_complete_agent_review_contract():
    agent = LLMAgent()
    args = {"plan_id": "p", "review_after_ticks": 3,
            "plan_expires_at_tick": 9, "wake_if": [],
            "trigger_evidence_ids": ["ev-1"]}
    agent._record_pending_plans(Action(tool_calls=[ToolCall(
        name="commit_to_plan", call_id="call", idempotency_key="key", args=args,
    )]), {"tick": 2})
    agent.observe_transition({"tick": 3, "__last_tool_results__": [{
        "name": "commit_to_plan", "ok": True, "call_id": "call",
        "idempotency_key": "key", "payload": {"plan_id": "p", "ack": True},
    }]})
    assert all(agent._active_plan.get(key) == value for key, value in args.items())


def test_pending_receipt_does_not_hide_terminal_receipt_in_same_observation():
    agent = LLMAgent()
    agent._record_pending_plans(Action(tool_calls=[ToolCall(
        name="commit_to_plan", call_id="call", idempotency_key="key",
        args={"plan_id": "p", "review_after_ticks": 3},
    )]), {"tick": 0})
    receipt = {"name": "commit_to_plan", "ok": True,
               "call_id": "call", "idempotency_key": "key"}
    agent.observe_transition({
        "tick": 1,
        "__last_tool_results__": [{**receipt, "payload": {"plan_id": "p", "_status": "pending"}}],
        "__within_tick_tool_results__": [{**receipt, "payload": {"plan_id": "p", "ack": True}}],
    })
    assert agent._active_plan is not None
    assert agent.get_interaction_stats()["plan_commits_confirmed"] == 1
    assert not agent._pending_plan_calls
