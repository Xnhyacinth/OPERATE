"""Real-time terminal warning outcomes retain their answered event lineage."""

from concurrent.futures import Future

import pytest

from core import Action
from runner.realtime_episode import RealtimeEpisodeCoordinator
from tests.test_realtime_episode import (
    _AlarmEnvironment,
    _DelayedStubDriver,
    _SafetySupervisor,
)


class WarningEnvironment(_AlarmEnvironment):
    def __init__(self, warning_rows):
        super().__init__(alarm_tick=None, horizon=len(warning_rows))
        self.warning_rows = warning_rows

    def step(self, action):
        result = super().step(action)
        result.info.early_stop_warnings = self.warning_rows[self.tick - 1]
        return result


class NoActionDriver(_DelayedStubDriver):
    def start_turn(self, *, turn_id, observation, event):
        self.started.append(event)
        future = Future()
        future.set_result(Action())
        return future


def test_answered_realtime_terminal_warning_is_residual_state():
    driver = NoActionDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=WarningEnvironment([["MRM"]] * 3),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2)
    terminal = [
        event
        for event in artifact["events"]
        if event["kind"] == "safety_warning" and event["simulator_tick"] == 3
    ]
    assert len(terminal) == 1
    assert terminal[0]["decision_required"] is False
    assert terminal[0]["answered_by_turn_ids"]
    assert terminal[0].get("terminal_unanswerable") is not True
    assert [event.simulator_tick for event in driver.started] == [0, 1, 2]
    assert not any(
        event["kind"] == "quiet_window" and event["simulator_tick"] == 3
        for event in artifact["events"]
    )


@pytest.mark.parametrize(
    "warnings",
    [
        [[], [], ["MRM"]],
        [["MRM"], ["MRM"], ["MRM", "new_hazard"]],
        [["MRM"], [], ["MRM"]],
    ],
)
def test_unanswered_realtime_terminal_warning_still_blocks(warnings):
    artifact = RealtimeEpisodeCoordinator(
        env=WarningEnvironment(warnings),
        turn_driver=NoActionDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2)
    terminal = next(
        event
        for event in artifact["events"]
        if event["kind"] == "safety_warning" and event["simulator_tick"] == 3
    )
    assert terminal["decision_required"] is True
    assert terminal["terminal_unanswerable"] is True
    assert terminal["terminal_formal_blocker"] is True


@pytest.mark.parametrize("outcome", ["failed", "hard_error_fallback"])
def test_failed_or_invalid_realtime_turn_does_not_answer_warning(outcome):
    class InvalidDriver(NoActionDriver):
        def start_turn(self, *, turn_id, observation, event):
            self.started.append(event)
            future = Future()
            if outcome == "failed":
                future.set_exception(RuntimeError("provider failed"))
            else:
                future.set_result(Action(dominant=outcome))
            return future

    artifact = RealtimeEpisodeCoordinator(
        env=WarningEnvironment([["MRM"]] * 3),
        turn_driver=InvalidDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2)
    terminal = next(
        event
        for event in artifact["events"]
        if event["kind"] == "safety_warning" and event["simulator_tick"] == 3
    )
    assert terminal["decision_required"] is True
    assert terminal["terminal_unanswerable"] is True


def test_uncompleted_realtime_warning_turn_remains_terminal_blocker():
    class UncompletedDriver(NoActionDriver):
        def start_turn(self, *, turn_id, observation, event):
            self.started.append(event)
            return Future()

    artifact = RealtimeEpisodeCoordinator(
        env=WarningEnvironment([["MRM"]] * 3),
        turn_driver=UncompletedDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2)
    terminal = next(
        event
        for event in artifact["events"]
        if event["kind"] == "safety_warning" and event["simulator_tick"] == 3
    )
    assert terminal["terminal_formal_blocker"] is True


def test_realtime_native_terminal_alarm_survives_residual_warning():
    class NativeAlarmEnvironment(WarningEnvironment):
        def step(self, action):
            result = super().step(action)
            if result.done:
                result.info.realized_events = [
                    {
                        "event_id": "new-terminal-hazard",
                        "event_class": "safety",
                        "type": "new_native_safety_alarm",
                    }
                ]
            return result

    artifact = RealtimeEpisodeCoordinator(
        env=NativeAlarmEnvironment([["MRM"]] * 3),
        turn_driver=NoActionDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2)
    native = next(
        event
        for event in artifact["events"]
        if event.get("payload", {}).get("event_id") == "new-terminal-hazard"
    )
    assert native["decision_required"] is True
    assert native["terminal_formal_blocker"] is True


def test_hidden_missing_terminal_response_window_is_audit_only_blocker():
    driver = NoActionDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=WarningEnvironment([[]]),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=1,
    )
    try:
        coordinator._process_transition(
            {
                "state_version_after": 1,
                "simulator_tick": 1,
                "environment_done": True,
                "realized_events": [
                    {
                        "event_id": "hidden-required-window",
                        "event_class": "safety",
                        "hidden": True,
                        "response_window_required": True,
                        "terminal_response_window_missing": True,
                    }
                ],
            }
        )
        assert any(
            "terminal_response_window_missing" in row["violation_codes"]
            for row in coordinator._event_contract_violations
        )
        assert not any(
            event.get("payload", {}).get("event_id") == "hidden-required-window"
            for event in coordinator._events
        )
        assert driver.started == []
    finally:
        driver.close()
