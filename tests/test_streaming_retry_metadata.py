from __future__ import annotations

import httpx
import pytest
from openai import APIError

import run
from baselines.llm_agent import ProviderCircuitOpenError
from runner import batch as runner
from scripts.batch_llm_eval import _retryable_infrastructure_row


@pytest.mark.parametrize("code,retryable", [(429, True), (502, True), ("503", True), (403, False), ("invalid_model", False)])
def test_stream_error_code_survives_circuit_wrapper(monkeypatch, tmp_path, code, retryable):
    def fail(**kwargs):
        cause = APIError("stream failed", request=httpx.Request("POST", "https://example.test/v1/chat/completions"), body={"code": code})
        raise ProviderCircuitOpenError("provider circuit opened") from cause

    monkeypatch.setattr(run, "load_scenario_yaml", lambda slug: {})
    monkeypatch.setattr(runner, "run_one", fail)
    row = runner.run_one_safe(("case", "llm_agent", 42, {}, {"episode_log_path": tmp_path/"episode.log"}))
    assert row["error_cause_type"] == "APIError"
    if str(code).isdigit():
        assert row.get("error_http_status") == int(code)
    else:
        assert "error_http_status" not in row
    assert _retryable_infrastructure_row(row) is retryable
