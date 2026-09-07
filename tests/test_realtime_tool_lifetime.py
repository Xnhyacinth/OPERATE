"""Decision response deadlines must not expire timely delayed tool effects."""

from concurrent.futures import Future

import pytest

from core import (
    Action,
    StepInfo,
    StepReturn,
    ToolCall,
    ToolContext,
    ToolRegistry,
    ToolSpec,
)
from runner.realtime_actor import HoldSafetySupervisor
from runner.realtime_episode import RealtimeEpisodeCoordinator
from tests.test_realtime_episode import _DelayedStubDriver


class DelayedEnvironment:
    def __init__(self):
        self.tick = 0
        self.horizon = 4
        self.value = 0
        self._tools = ToolRegistry(seed=42)
        self._tools.register(
            ToolSpec(
                name="delayed_control",
                description="delayed native control",
                parameters={
                    "type": "object",
                    "properties": {"expires_at_tick": {"type": "integer"}},
                },
                handler=self.mutate,
                state_changing=True,
                delay_ticks=1,
                fail_rate=0,
            )
        )
        self._tools.register(
            ToolSpec(
                name="wait",
                description="wait",
                parameters={"type": "object", "properties": {}},
                handler=lambda *_: {},
                delay_ticks=0,
                fail_rate=0,
            )
        )

    def mutate(self, args, ctx):
        self.value += 1
        return {"value": self.value}

    def snapshot(self):
        return {"tick": self.tick, "value": self.value}

    def get_tool_specs(self):
        return [
            self._tools.get(name).to_openai_schema() for name in self._tools.names()
        ]

    def step(self, action):
        results = self._tools.execute_action(
            action,
            ToolContext(
                tick=self.tick,
                seed=42,
                extra={"episode_horizon": self.horizon, "env": self},
            ),
        )
        self.tick += 1
        return StepReturn(
            observation=self.snapshot(),
            tool_results=results,
            reward=0,
            done=self.tick >= self.horizon,
            info=StepInfo(),
        )


@pytest.mark.parametrize("explicit_expiry,expected_value", [(None, 1), (1, 0)])
def test_registered_tool_delay_has_effect_window_without_relaxing_expiry(
    explicit_expiry, expected_value
):
    class Driver(_DelayedStubDriver):
        def start_turn(self, *, turn_id, observation, event):
            future = Future()
            if event.kind == "session_start":
                args = (
                    {}
                    if explicit_expiry is None
                    else {"expires_at_tick": explicit_expiry}
                )
                future.set_result(
                    Action(tool_calls=[ToolCall(name="delayed_control", args=args)])
                )
            else:
                future.set_result(Action())
            return future

    env = DelayedEnvironment()
    result = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=Driver(),
        safety_supervisor=HoldSafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2)
    assert env.value == expected_value
    assert result["turns"][0]["active_deadline_tick"] == 1
    if explicit_expiry is None:
        assert result["turns"][0]["action_expires_at_tick"] == 2
    else:
        assert result["turns"][0]["receipt_status"] == "expired"


@pytest.mark.parametrize(
    "deadline_field",
    ["deadline_tick", "response_deadline_tick", "mandatory_response_tick"],
)
def test_explicit_native_deadline_is_not_extended_for_tool_delay(deadline_field):
    class AlarmEnvironment(DelayedEnvironment):
        def step(self, action):
            result = super().step(action)
            if self.tick == 1:
                result.info.realized_events = [
                    {
                        "type": "capacity_alarm",
                        "event_class": "alarm",
                        "event_id": "hard-deadline-alarm",
                        deadline_field: 2,
                    }
                ]
            return result

    class Driver(_DelayedStubDriver):
        def start_turn(self, *, turn_id, observation, event):
            future = Future()
            calls = (
                [ToolCall(name="delayed_control")]
                if event.kind == "environment_alarm"
                else []
            )
            future.set_result(Action(tool_calls=calls))
            return future

    env = AlarmEnvironment()
    result = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=Driver(),
        safety_supervisor=HoldSafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2)
    assert env.value == 0
    alarm = next(
        turn for turn in result["turns"] if turn["trigger_kind"] == "environment_alarm"
    )
    assert alarm["active_deadline_tick"] == 2
    assert alarm["action_expires_at_tick"] == 2
    assert alarm["receipt_status"] == "expired"


def test_malformed_tool_expiry_reaches_protocol_rejection_without_hanging_turn():
    class Driver(_DelayedStubDriver):
        def start_turn(self, *, turn_id, observation, event):
            future = Future()
            calls = (
                [ToolCall(name="delayed_control", args={"expires_at_tick": "invalid"})]
                if event.kind == "session_start"
                else []
            )
            future.set_result(Action(tool_calls=calls))
            return future

    env = DelayedEnvironment()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=Driver(),
        safety_supervisor=HoldSafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2)
    assert env.value == 0
    assert artifact["turns"][0]["status"] == "completed"
    assert any(
        result.get("ok") is False
        for transition in artifact["transitions"]
        for result in transition["tool_results"]
    )
