from types import SimpleNamespace

import httpx
import pytest
from openai import APIError

from baselines.llm_agent import classify_provider_error


@pytest.mark.parametrize("code,expected", [
    (502, "provider_server_error"),
    ("503", "provider_server_error"),
    (429, "provider_rate_limit"),
    (403, "provider_other_error"),
])
def test_stream_error_classification_uses_structured_status(code, expected):
    error = APIError(
        "opaque upstream failure",
        request=httpx.Request("POST", "https://example.test/v1"),
        body={"code": code},
    )
    assert classify_provider_error(error) == expected


@pytest.mark.parametrize("message", [
    "maximum context size is 150000 tokens",
    "unknown entity job_429 and task_500",
    "invalid input: value must be less than 6004",
])
def test_unrelated_numbers_do_not_trigger_provider_retries(message):
    assert classify_provider_error(ValueError(message)) == "provider_other_error"


def test_http_auth_error_has_priority_over_incidental_server_error_text():
    error = RuntimeError("403 access denied: server error for request 500")
    error.status_code = 403
    error.response = SimpleNamespace(status_code=403)
    assert classify_provider_error(error) == "provider_other_error"


@pytest.mark.parametrize("body", [{"code": 6004}, {"error": {"code": "6004"}}])
def test_structured_hard_quota_code_is_not_retried_as_unknown_error(body):
    error = APIError("opaque upstream failure", request=httpx.Request(
        "POST", "https://example.test/v1",
    ), body=body)
    assert classify_provider_error(error) == "provider_quota_exhausted"
