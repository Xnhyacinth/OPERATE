"""An unrelated receipt cannot activate or supersede a standing plan."""

from core import Action, ToolCall, ToolResult
from runner.episode import _materialized_autonomy_window
from runner.realtime_episode import RealtimeEpisodeCoordinator
from tests.test_realtime_episode import (
    _AlarmEnvironment,
    _DelayedStubDriver,
    _SafetySupervisor,
)


def test_logical_unrelated_explicit_plan_receipt_does_not_activate_plan():
    call = ToolCall(
        name="commit_to_plan",
        call_id="requested-plan",
        args={"plan_id": "requested", "review_after_ticks": 4},
    )
    pending = {}
    result = _materialized_autonomy_window(
        Action(tool_calls=[call]),
        [ToolResult(name="commit_to_plan", call_id="unrelated-plan", ok=True)],
        pending,
    )
    assert result == (0, None, None, {})
    assert pending == {"requested-plan": call}


def test_realtime_unrelated_explicit_plan_receipt_does_not_activate_plan():
    driver = _DelayedStubDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=1,
    )
    try:
        coordinator._ingest_confirmed_plan_reviews(
            {
                "simulator_tick": 1,
                "submitted_action": {
                    "actions": [
                        {
                            "name": "commit_to_plan",
                            "call_id": "requested-plan",
                            "args": {"review_after_ticks": 4},
                        }
                    ]
                },
                "tool_results": [
                    {"name": "commit_to_plan", "call_id": "unrelated-plan", "ok": True}
                ],
            }
        )
        assert coordinator._active_plan is False
        assert coordinator._scheduled_review_ticks == []
        assert "requested-plan" in coordinator._pending_plan_requests
    finally:
        driver.close()
