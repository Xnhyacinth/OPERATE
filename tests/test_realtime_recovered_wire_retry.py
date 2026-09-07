from copy import deepcopy
import hashlib
import json

import pytest

from runner import realtime_episode as runtime


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def retry_row():
    row = {
        "provider_turn_settled": True,
        "provider_started": True,
        "provider_audit_status": "completed",
        "behavioral_transaction_consistent": True,
        "behavioral_transaction_status": "committed",
        "behavioral_state_outcome": "committed",
        "provider_requests": [],
        "provider_responses": [],
        "provider_model_identities": [],
    }
    for seq in (1, 2):
        identity = {
            "schema_version": "provider_model_identity_closure_v1",
            "request_sequence": seq,
            "requested_model": "model",
            "observed_models": [] if seq == 1 else ["model"],
            "closure": "request_failed" if seq == 1 else "exact",
        }
        envelope = {
            "model": "model",
            "messages": [{"role": "user", "content": "same"}],
            "tools": [],
            "max_tokens": 100,
            "temperature": 0,
            "provider_retry_index": seq - 1,
            "retry_of_request_sequence": None if seq == 1 else 1,
            "provider_transient_retry_policy": {
                "max_retries": 4,
                "retry_reasons": [
                    "provider_rate_limit",
                    "provider_server_error",
                    "provider_transport_error",
                ],
            },
            "request_budget": {"status": "within_budget"},
            "provider_rate_limit": {"status": "acquired"},
        }
        response = {
            "status": "failed" if seq == 1 else "success",
            "model_identity_closure": deepcopy(identity),
        }
        if seq == 1:
            response["error_reason"] = "provider_transport_error"
        row["provider_requests"].append(
            {"sequence": seq, "envelope": envelope, "sha256": digest(envelope)}
        )
        row["provider_responses"].append(
            {
                "sequence": seq,
                "request_sequence": seq,
                "response": response,
                "sha256": digest(response),
            }
        )
        row["provider_model_identities"].append(identity)
    return row


def test_exact_recovered_retry_is_not_failed_provider_turn():
    assert runtime._provider_turn_audit_violations(retry_row()) == set()


@pytest.mark.parametrize(
    "defect",
    ["messages", "index", "root", "quota", "hash", "identity", "unrecovered", "budget"],
)
def test_unproven_retry_chain_remains_failed(defect):
    row = retry_row()
    request = row["provider_requests"][-1]
    response = row["provider_responses"][0]
    if defect == "messages":
        request["envelope"]["messages"][0]["content"] = "different"
    if defect == "index":
        request["envelope"]["provider_retry_index"] = 2
    if defect == "root":
        request["envelope"]["retry_of_request_sequence"] = 10
    if defect == "quota":
        response["response"]["error_reason"] = "provider_quota_exhausted"
    if defect == "identity":
        row["provider_model_identities"][-1]["observed_models"] = ["other"]
    if defect == "unrecovered":
        row["provider_responses"][-1]["response"]["status"] = "failed"
    if defect == "budget":
        request["envelope"]["request_budget"]["status"] = "preflight_rejected"
    request["sha256"] = digest(request["envelope"])
    response["sha256"] = digest(response["response"])
    if defect == "hash":
        request["sha256"] = "0" * 64
    assert "PROVIDER_RESPONSE_FAILED" in runtime._provider_turn_audit_violations(row)


def test_real_agent_retry_audit_preserves_unicode_wire_identity(monkeypatch):
    from baselines.llm_agent import LLMAgent, LLMConfig
    from core import Action

    agent = LLMAgent(
        LLMConfig(
            interaction_mode="logical_persistent",
            model_context_window_tokens=192000,
            model_max_output_tokens=64000,
        )
    )
    monkeypatch.setattr(agent, "_sleep_before_provider_retry", lambda _: None)
    calls = []

    def invoke():
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("stream断开")
        agent._record_provider_response_identity(
            type("Response", (), {"model": agent.config.model})()
        )
        return Action(dominant="wait")

    action, started, sequence = agent._call_with_transient_provider_retries(
        invoke=invoke,
        messages=[{"role": "user", "content": "调度任务"}],
        tools=[],
        fallback_without_tools=False,
    )
    agent._record_provider_action_response(
        action, started_ns=started, request_sequence=sequence
    )
    stats = agent.get_interaction_stats()
    row = {
        **retry_row(),
        "provider_requests": stats["provider_request_records"],
        "provider_responses": stats["provider_response_records"],
        "provider_model_identities": stats["provider_model_identity_records"],
    }
    assert runtime.recovered_provider_retry_sequences(row) == {1}
    assert runtime._provider_turn_audit_violations(row) == set()


@pytest.mark.parametrize("retries,accepted", [(4, True), (5, False)])
def test_retry_bound_is_enforced(retries, accepted):
    row = retry_row()
    base = deepcopy(row)
    row["provider_requests"] = []
    row["provider_responses"] = []
    row["provider_model_identities"] = []
    for index in range(retries + 1):
        terminal = index == retries
        request = deepcopy(base["provider_requests"][0])
        response = deepcopy(base["provider_responses"][1 if terminal else 0])
        identity = deepcopy(base["provider_model_identities"][1 if terminal else 0])
        seq = index + 1
        request["sequence"] = seq
        request["envelope"].update(
            provider_retry_index=index, retry_of_request_sequence=1 if index else None
        )
        identity["request_sequence"] = seq
        response.update(sequence=seq, request_sequence=seq)
        response["response"]["model_identity_closure"] = identity
        request["sha256"] = digest(request["envelope"])
        response["sha256"] = digest(response["response"])
        row["provider_requests"].append(request)
        row["provider_responses"].append(response)
        row["provider_model_identities"].append(identity)
    assert bool(runtime.recovered_provider_retry_sequences(row)) is accepted


def test_strict_cancellation_closes_prior_retry_failures():
    row = retry_row()
    row.update(
        provider_audit_status="superseded_completed",
        turn_status="superseded",
        cancel_requested=True,
        cancel_acknowledged=True,
        cancellation_mode="provider_stream_canceled",
        hard_cancel_performed=True,
        execution_fence="late_response_audit_only",
        late_response_discarded=True,
        behavioral_transaction_status="rolled_back",
        behavioral_state_outcome="rolled_back",
    )
    identity = row["provider_model_identities"][-1]
    identity.update(closure="request_failed")
    payload = row["provider_responses"][-1]["response"]
    payload.update(
        status="failed",
        error_reason="realtime_turn_canceled",
        error_summary="realtime provider stream canceled: turn-1",
        model_identity_closure=deepcopy(identity),
    )
    row["provider_responses"][-1]["sha256"] = digest(payload)
    assert runtime.recovered_provider_retry_sequences(row) == {1}
    assert runtime._provider_turn_audit_violations(row) == set()
    row["cancel_acknowledged"] = False
    assert runtime.recovered_provider_retry_sequences(row) == set()
