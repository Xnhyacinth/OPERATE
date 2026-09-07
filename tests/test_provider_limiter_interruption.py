"""Canceled limiter sleeps retain reservation evidence without a transport call."""

import json

import pytest

from core.provider_request_limiter import ProviderRequestLimiter


def test_interrupted_wait_attaches_elapsed_audit_and_keeps_reservation(
    tmp_path, monkeypatch
):
    class TurnCanceled(RuntimeError):
        pass

    epoch = 1_800_000_000.0
    limiter = ProviderRequestLimiter(
        rpm_limit=1,
        rpd_limit=10,
        scope="test-cancel",
        state_dir=tmp_path,
        now=lambda: epoch,
    )
    limiter.acquire()
    canceled = TurnCanceled("canceled while reserving request")

    def interrupted_sleep(seconds):
        assert seconds == 60.0
        raise canceled

    monotonic = iter([100.0, 103.25])
    monkeypatch.setattr(
        "core.provider_request_limiter.time.monotonic", lambda: next(monotonic)
    )
    waiting = ProviderRequestLimiter(
        rpm_limit=1,
        rpd_limit=10,
        scope="test-cancel",
        state_dir=tmp_path,
        now=lambda: epoch,
        sleep=interrupted_sleep,
    )
    with pytest.raises(TurnCanceled) as raised:
        waiting.acquire()
    assert raised.value is canceled
    audit = canceled.provider_rate_limit_audit
    assert audit["status"] == "wait_interrupted"
    assert audit["scheduled_wait_seconds"] == 60.0
    assert audit["wait_seconds"] == 3.25
    assert audit["utc_day_request_count"] == 2
    state = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert state["day_count"] == 2
    assert state["request_times"] == [epoch, epoch + 60]


def test_successful_wait_keeps_existing_acquired_audit(tmp_path):
    epoch = 1_800_000_000.0
    waits = []
    limiter = ProviderRequestLimiter(
        rpm_limit=1,
        scope="test-success",
        state_dir=tmp_path,
        now=lambda: epoch,
        sleep=waits.append,
    )
    limiter.acquire()
    audit = limiter.acquire()
    assert waits == [60.0]
    assert audit["status"] == "acquired"
    assert audit["wait_seconds"] == 60.0
    assert "scheduled_wait_seconds" not in audit
