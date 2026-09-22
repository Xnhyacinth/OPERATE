from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import runner.realtime_episode as realtime_episode
from baselines.llm_agent import (
    LLMAgent,
    LLMConfig,
)
from core import (
    Action,
    StepInfo,
    StepReturn,
    ToolCall,
    ToolContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from core.evidence import EvidenceLogger
from evaluation.realtime_diagnostics import evaluate_realtime_diagnostics
from runner.realtime_actor import SafetyDecision
from runner.realtime_episode import (
    AgentTurnDriver,
    RealtimeEpisodeCoordinator,
    RealtimeEvent,
    _build_evidence_closure,
    _continuation_tool_results,
    _write_realtime_artifact_exclusive,
    build_realtime_treatment_identity,
    run_realtime,
)


class _AlarmEnvironment:
    def __init__(self, *, alarm_tick: int | None = 1, horizon: int = 5) -> None:
        self.tick = 0
        self.horizon = horizon
        self.alarm_tick = alarm_tick
        self.applied: list[str] = []

    def snapshot(self) -> dict:
        return {"tick": self.tick, "entities": {"queue": {"depth": self.tick}}}

    def get_tool_specs(self) -> list[dict]:
        return [{"type": "function", "function": {"name": "reroute"}}]

    def step(self, action: Action) -> StepReturn:
        names = [call.name for call in action.tool_calls]
        self.applied.extend(names)
        events = []
        if self.alarm_tick is not None and self.tick + 1 == self.alarm_tick:
            events.append(
                {
                    "event_id": "alarm-1",
                    "type": "capacity_alarm",
                    "event_class": "alarm",
                    "decision_required": True,
                    "hidden": False,
                }
            )
        results = [
            ToolResult(name=name, ok=True, state_changing=name == "reroute")
            for name in names
        ]
        self.tick += 1
        return StepReturn(
            observation=self.snapshot(),
            tool_results=results,
            reward=0.0,
            done=self.tick >= self.horizon,
            info=StepInfo(realized_events=events),
        )


class _SafetySupervisor:
    def decide(self, *, observation: dict, simulator_tick: int, reason: str) -> SafetyDecision:
        del observation, simulator_tick
        return SafetyDecision(
            action=Action(tool_calls=[ToolCall(name="minimum_risk_hold")]),
            mode="minimum_risk_fallback",
            reason_code=reason,
        )


@dataclass
class _Turn:
    turn_id: str
    future: Future[Action]


class _DelayedStubDriver:
    """Deterministic provider stub: initial turn is slow, alarm response is fast."""

    def __init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=2)
        self.started: list[RealtimeEvent] = []
        self.canceled: list[str] = []
        self.closed = False

    def start_turn(self, *, turn_id: str, observation: dict, event: RealtimeEvent) -> Future[Action]:
        del observation
        self.started.append(event)

        def complete() -> Action:
            if event.kind == "session_start":
                time.sleep(0.11)
                return Action(tool_calls=[ToolCall(name="late_initial_command")])
            time.sleep(0.004)
            return Action(tool_calls=[ToolCall(name="reroute")])

        return self._pool.submit(complete)

    def steer_turn(self, *, turn_id: str, event: RealtimeEvent) -> bool:
        del turn_id, event
        return False

    def cancel_turn(self, *, turn_id: str, reason: str) -> None:
        del reason
        self.canceled.append(turn_id)

    def commit_turn(self, turn_id: str) -> bool:
        del turn_id
        return True

    def rollback_turn(self, turn_id: str) -> bool:
        del turn_id
        return True

    def close(self, *, wait: bool = True) -> None:
        self.closed = True
        self._pool.shutdown(wait=wait, cancel_futures=True)


class _PlanReviewDriver(_DelayedStubDriver):
    def start_turn(
        self, *, turn_id: str, observation: dict, event: RealtimeEvent
    ) -> Future[Action]:
        del turn_id, observation
        self.started.append(event)
        future: Future[Action] = Future()
        if event.kind == "session_start":
            future.set_result(
                Action(
                    tool_calls=[
                        ToolCall(
                            name="commit_to_plan",
                            args={"review_after_ticks": 2},
                        )
                    ]
                )
            )
        else:
            future.set_result(Action())
        return future


class _NativeSteerDriver(_DelayedStubDriver):
    def __init__(self) -> None:
        super().__init__()
        self._steered = False

    def start_turn(
        self, *, turn_id: str, observation: dict, event: RealtimeEvent
    ) -> Future[Action]:
        del turn_id, observation
        self.started.append(event)

        def complete() -> Action:
            deadline = time.monotonic() + 1.0
            while not self._steered and time.monotonic() < deadline:
                time.sleep(0.001)
            return Action(tool_calls=[ToolCall(name="reroute")])

        return self._pool.submit(complete)

    def steer_turn(self, *, turn_id: str, event: RealtimeEvent) -> bool:
        del turn_id, event
        self._steered = True
        return True


class _ConcurrencyProbeAgent:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def snapshot_behavioral_state(self) -> dict:
        return {"active": self.active}

    def restore_behavioral_state(self, snapshot: dict) -> None:
        self.active = int(snapshot["active"])

    def act(self, observation: dict, tool_specs: list[dict]) -> Action:
        del observation, tool_specs
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)
        with self._lock:
            self.active -= 1
        return Action()


class _TransactionalAgent:
    def __init__(self) -> None:
        self.history: list[str] = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def snapshot_behavioral_state(self) -> dict:
        return {"history": list(self.history)}

    def restore_behavioral_state(self, snapshot: dict) -> None:
        self.history = list(snapshot["history"])

    def act(self, observation: dict, tool_specs: list[dict]) -> Action:
        del tool_specs
        self.history.append(str(observation["__decision_epoch__"]["turn_id"]))
        self.entered.set()
        assert self.release.wait(timeout=1.0)
        return Action(tool_calls=[ToolCall(name="mutating_action")])


class _TimeoutDriver:
    def __init__(self) -> None:
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.release = threading.Event()
        self.future: Future[Action] | None = None
        self.close_wait: bool | None = None
        self.cancel_requests: list[str] = []

    def start_turn(
        self, *, turn_id: str, observation: dict, event: RealtimeEvent
    ) -> Future[Action]:
        del turn_id, observation, event

        def complete() -> Action:
            assert self.release.wait(timeout=1.0)
            return Action(tool_calls=[ToolCall(name="too_late")])

        self.future = self.pool.submit(complete)
        return self.future

    def steer_turn(self, *, turn_id: str, event: RealtimeEvent) -> bool:
        del turn_id, event
        return False

    def cancel_turn(self, *, turn_id: str, reason: str) -> bool:
        del reason
        self.cancel_requests.append(turn_id)
        return bool(self.future.cancel()) if self.future is not None else False

    def commit_turn(self, turn_id: str) -> bool:
        del turn_id
        return True

    def rollback_turn(self, turn_id: str) -> bool:
        del turn_id
        return True

    def outstanding_turn_count(self) -> int:
        return int(self.future is not None and not self.future.done())

    def close(self, *, wait: bool = True) -> None:
        self.close_wait = wait
        self.pool.shutdown(wait=wait, cancel_futures=True)


class _InspectEnvironment(_AlarmEnvironment):
    def step(self, action: Action) -> StepReturn:
        names = [call.name for call in action.tool_calls]
        self.applied.extend(names)
        results = [
            ToolResult(
                name=name,
                ok=True,
                payload={"queue_depth": 7} if name == "inspect_queue" else {},
                state_changing=name == "commit_control",
                evidence_id=f"evidence-{self.tick}-{name}",
            )
            for name in names
        ]
        self.tick += 1
        return StepReturn(
            observation=self.snapshot(),
            tool_results=results,
            reward=1.0 if "commit_control" in names else 0.0,
            done=self.tick >= self.horizon,
            info=StepInfo(evidence_ids=[f"tick-{self.tick}"]),
        )


class _InspectThenCommitDriver(_DelayedStubDriver):
    def start_turn(
        self, *, turn_id: str, observation: dict, event: RealtimeEvent
    ) -> Future[Action]:
        del turn_id
        self.started.append(event)
        future: Future[Action] = Future()
        if event.kind == "session_start":
            future.set_result(Action(tool_calls=[ToolCall(name="inspect_queue")]))
        elif event.kind == "tool_result" and not any(
            call.kind == "tool_result" for call in self.started[:-1]
        ):
            assert observation["__last_tool_results__"][0]["name"] == "inspect_queue"
            future.set_result(Action(tool_calls=[ToolCall(name="commit_control")]))
        else:
            future.set_result(Action())
        return future


def _event(sequence: int) -> RealtimeEvent:
    return RealtimeEvent(
        event_id=f"e{sequence}",
        event_seq=sequence,
        kind="environment_alarm",
        priority=100,
        decision_id=f"d{sequence}",
        state_version=sequence,
        simulator_tick=sequence,
        deadline_tick=sequence + 1,
        deadline_monotonic_ns=None,
        decision_required=True,
    )


def test_realtime_alarm_cancels_inflight_turn_and_late_response_never_executes() -> None:
    env = _AlarmEnvironment()
    driver = _DelayedStubDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    )

    artifact = coordinator.run(timeout_s=2.0)

    assert artifact["interaction_mode"] == "realtime_persistent"
    assert "reroute" in env.applied
    assert "late_initial_command" not in env.applied
    assert driver.canceled == ["turn-1"]
    assert artifact["turns"][0]["status"] == "superseded"
    assert artifact["turns"][0]["late_response_discarded"] is True
    assert any(event["kind"] == "environment_alarm" for event in artifact["events"])
    assert artifact["diagnostics"]["alarm_response"]["detected"] == 1
    assert artifact["diagnostics"]["action_lifecycle"]["stale_or_discarded"] >= 1


def test_blocking_native_provider_is_canceled_before_alarm_supersession_executes() -> None:
    class BlockingCancelDriver(_DelayedStubDriver):
        def __init__(self) -> None:
            super().__init__()
            self.initial_started = threading.Event()
            self.initial_release = threading.Event()
            self._cancel_outcomes: dict[str, dict[str, bool]] = {}

        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del observation
            self.started.append(event)

            def complete() -> Action:
                if event.kind == "session_start":
                    self.initial_started.set()
                    assert self.initial_release.wait(timeout=1.0)
                    return Action(tool_calls=[ToolCall(name="late_initial_command")])
                return Action(tool_calls=[ToolCall(name="reroute")])

            return self._pool.submit(complete)

        def cancel_turn(self, *, turn_id: str, reason: str) -> bool:
            del reason
            assert self.initial_started.is_set()
            self.canceled.append(turn_id)
            self._cancel_outcomes[turn_id] = {
                "provider_stream_canceled": True,
                "queued_future_canceled": False,
            }
            self.initial_release.set()
            return True

        def cancellation_outcome(self, turn_id: str) -> dict[str, bool]:
            return self._cancel_outcomes.get(turn_id, {})

        def capabilities(self) -> dict[str, bool]:
            return {"provider_turn_hard_cancel_supported": True}

    env = _AlarmEnvironment(horizon=3)
    driver = BlockingCancelDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2.0)

    first = artifact["turns"][0]
    assert first["status"] == "superseded"
    assert first["cancel_acknowledged"] is True
    assert first["cancellation_mode"] == "provider_stream_canceled"
    assert first["hard_cancel_performed"] is True
    assert "late_initial_command" not in env.applied
    assert "reroute" in env.applied


def test_stream_registered_after_supersession_updates_eventual_cancel_audit() -> None:
    class UnreadStream:
        def __init__(self) -> None:
            self.closed = False
            self.iterated = False

        def __iter__(self):
            self.iterated = True
            return iter(())

        def close(self) -> None:
            self.closed = True

    class StreamCreationRaceAgent(LLMAgent):
        def __init__(self) -> None:
            super().__init__(
                LLMConfig(
                    provider="openai_compatible",
                    api_mode="chat_completions",
                    stream_chat_completions=True,
                    interaction_mode="logical_persistent",
                )
            )
            self.creation_blocked = threading.Event()
            self.cancel_requested = threading.Event()
            self.release_creation = threading.Event()
            self.stream = UnreadStream()
            self.call_count = 0

        def act(self, observation: dict, tool_specs: list[dict]) -> Action:
            del tool_specs
            self.call_count += 1
            turn_id = str(observation["__decision_epoch__"]["turn_id"])
            self._begin_realtime_turn(turn_id)  # noqa: SLF001
            try:
                if self.call_count == 1:
                    self.creation_blocked.set()
                    assert self.release_creation.wait(timeout=1.0)
                    return self._action_from_openai_stream(  # noqa: SLF001
                        self.stream
                    )
                return Action(tool_calls=[ToolCall(name="reroute")])
            finally:
                self._end_realtime_turn(turn_id)  # noqa: SLF001

        def cancel_realtime_turn(self, *, turn_id: str, reason: str) -> bool:
            canceled = super().cancel_realtime_turn(
                turn_id=turn_id, reason=reason
            )
            self.cancel_requested.set()
            self.release_creation.set()
            return canceled

    env = _AlarmEnvironment(horizon=3)
    agent = StreamCreationRaceAgent()
    driver = AgentTurnDriver(agent, env.get_tool_specs())
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2.0)

    first = artifact["turns"][0]
    assert agent.creation_blocked.is_set()
    assert agent.cancel_requested.is_set()
    assert agent.stream.closed is True
    assert agent.stream.iterated is False
    assert first["status"] == "superseded"
    assert first["late_response_discarded"] is True
    assert first["cancel_acknowledged"] is True
    assert first["cancellation_mode"] == "provider_stream_canceled"
    assert first["hard_cancel_performed"] is True


def test_agent_turn_driver_stream_cancel_is_expected_audit_not_failure() -> None:
    class BlockingStream:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.released = threading.Event()
            self.closed = False

        def __iter__(self):
            self.started.set()
            assert self.released.wait(timeout=1.0)
            return iter(())

        def close(self) -> None:
            self.closed = True
            self.released.set()

    class ImmediateToolStream:
        def __iter__(self):
            function = SimpleNamespace(name="reroute", arguments="{}")
            tool_call = SimpleNamespace(index=0, id="call-reroute", function=function)
            delta = SimpleNamespace(
                content=None,
                reasoning_content=None,
                tool_calls=[tool_call],
            )
            choice = SimpleNamespace(finish_reason="tool_calls", delta=delta)
            return iter(
                [SimpleNamespace(model="test-model", choices=[choice])]
            )

        def close(self) -> None:
            return None

    class FakeCompletions:
        def __init__(self, blocking: BlockingStream) -> None:
            self.blocking = blocking
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            return self.blocking if self.calls == 1 else ImmediateToolStream()

    env = _AlarmEnvironment(horizon=3)
    tool_specs = [
        {
            "type": "function",
            "function": {
                "name": "reroute",
                "description": "Reroute queued work.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    blocking = BlockingStream()
    completions = FakeCompletions(blocking)
    agent = LLMAgent(
        LLMConfig(
            provider="openai_compatible",
            model="test-model",
            stream_chat_completions=True,
            interaction_mode="logical_persistent",
            tool_choice="required",
            max_tokens=1_024,
            model_context_window_tokens=16_384,
            model_max_output_tokens=4_096,
        )
    )
    agent._has_api_key = True  # noqa: SLF001
    agent._client = SimpleNamespace(  # noqa: SLF001
        chat=SimpleNamespace(completions=completions)
    )
    agent._system_prompt = "Use the provided tools."  # noqa: SLF001
    agent._tool_specs = tool_specs  # noqa: SLF001
    agent._readonly_tools = set()  # noqa: SLF001

    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=AgentTurnDriver(agent, tool_specs),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2.0)

    first_turn = artifact["turns"][0]
    first_audit = artifact["provider_audit"][0]
    assert blocking.started.is_set()
    assert blocking.closed is True
    assert first_turn["status"] == "superseded"
    assert first_turn["cancellation_mode"] == "provider_stream_canceled"
    assert first_audit["provider_audit_status"] == "superseded_completed"
    assert first_audit["provider_responses"][0]["response"]["status"] == "failed"
    assert first_audit["provider_model_identities"][0]["closure"] == (
        "request_failed"
    )
    assert artifact["provider_audit_contract"]["complete"] is True

    tampered = {**first_audit, "cancellation_mode": "logical_supersession"}
    assert "PROVIDER_RESPONSE_FAILED" in (
        realtime_episode._provider_turn_audit_violations(tampered)  # noqa: SLF001
    )


@pytest.mark.parametrize("transport_canceled", [False, True])
def test_final_settlement_refreshes_delivery_cancel_transport_audit(
    transport_canceled: bool,
) -> None:
    class DeliveryCancelDriver(_DelayedStubDriver):
        def __init__(self) -> None:
            super().__init__()
            self.delivery: Future[Action] = Future()
            self.settled = False

        def start_turn(self, *, turn_id, observation, event):
            self.started.append(event)
            return self.delivery

        def cancel_turn(self, *, turn_id, reason):
            self.canceled.append(turn_id)
            # A delivery wrapper settles before its provider transport closes.
            return self.delivery.cancel()

        def cancellation_outcome(self, turn_id):
            return {
                "provider_stream_canceled": self.settled and transport_canceled,
                "delivery_future_canceled": self.delivery.cancelled(),
            }

        def capabilities(self):
            return {"provider_turn_hard_cancel_supported": True}

        def wait_for_behavioral_settlement(self, *, timeout_s):
            self.settled = True
            return True

    env = _AlarmEnvironment(alarm_tick=None, horizon=1)
    driver = DeliveryCancelDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.01,
    ).run(timeout_s=1.0)

    turn = artifact["turns"][0]
    assert driver.settled is True
    assert turn["status"] == "superseded"
    assert turn["late_response_discarded"] is True
    assert turn["hard_cancel_performed"] is transport_canceled
    assert turn["cancellation_mode"] == (
        "provider_stream_canceled" if transport_canceled else "logical_supersession"
    )
    assert "late_initial_command" not in env.applied


def test_realtime_quiet_cell_measures_correct_silence_without_polling() -> None:
    env = _AlarmEnvironment(alarm_tick=None, horizon=3)
    driver = _DelayedStubDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.015,
    )
    artifact = coordinator.run(timeout_s=2.0)

    assert [event.kind for event in driver.started] == ["session_start"]
    assert artifact["diagnostics"]["alarm_response"] == {
        "terminal_unanswerable": 0,
        "actionable_alarms": 0,
        "detected": 0,
        "missed": 0,
        "delivery_missed": 0,
        "decision_missed": 0,
            "response_missed": 0,
            "false_alarms": 0,
            "false_alarm_assessed_interventions": 0,
            "false_alarm_unassessed_interventions": 0,
            "false_alarm_rate": None,
            "model_attempted_interventions": 0,
        "quiet_windows": 3,
        "agent_silence_opportunities": 0,
        "correct_silence": 0,
        "model_standing_plan_quiet_windows": 0,
        "model_delegated_hold_windows": 0,
        "autonomous_quiet_windows": 3,
    }
    assert artifact["diagnostics"]["autonomy"]["unnecessary_polling"] == 0
    assert artifact["diagnostics"]["safety"]["takeovers"] == 0
    assert artifact["diagnostics"]["safety"]["unverified_takeovers"] == 3


def test_quiet_cadence_does_not_create_turn_without_agent_scheduled_review() -> None:
    class QuietCadenceEnvironment(_AlarmEnvironment):
        def snapshot(self) -> dict:
            return {
                **super().snapshot(),
                "decision_cadence": {
                    "periodic_scan_every_ticks": 1,
                    "max_review_after_ticks": 2,
                },
            }

    class NoPlanDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(Action())
            return future

    env = QuietCadenceEnvironment(alarm_tick=None, horizon=4)
    driver = NoPlanDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.015,
    ).run(timeout_s=2.0)

    assert [event.kind for event in driver.started] == ["session_start"]
    assert not [
        event
        for event in artifact["events"]
        if event["kind"] == "supervisory_scan" and event["decision_required"]
    ]
    assert artifact["episode_status"] == "complete"
    assert env.tick == env.horizon
    assert len(artifact["transitions"]) == env.horizon
    assert all(
        transition["simulator_time_advanced"]
        and not transition["realized_events"]
        for transition in artifact["transitions"]
    )


def test_backend_cadence_does_not_replace_agent_scheduled_review() -> None:
    class CadenceEnvironment(_AlarmEnvironment):
        def snapshot(self) -> dict:
            return {
                **super().snapshot(),
                "decision_cadence": {
                    "periodic_scan_every_ticks": 2,
                    "max_review_after_ticks": 4,
                },
            }

    class ScheduledReviewDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(
                Action(
                    tool_calls=[
                        ToolCall(
                            name="commit_to_plan",
                            args={"review_after_ticks": 4},
                        )
                    ]
                )
                if event.kind == "session_start"
                else Action()
            )
            return future

    driver = ScheduledReviewDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=CadenceEnvironment(alarm_tick=None, horizon=6),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.015,
    ).run(timeout_s=2.0)

    assert not [
        event for event in artifact["events"] if event["kind"] == "supervisory_scan"
    ]
    assert [event.kind for event in driver.started] == [
        "session_start",
        "tool_result",
        "scheduled_review",
    ]
    review = next(
        event for event in artifact["events"] if event["kind"] == "scheduled_review"
    )
    assert review["simulator_tick"] == 5
    assert artifact["diagnostics"]["autonomy"]["unnecessary_polling"] == 0


def test_explicit_native_opportunity_does_not_start_realtime_turn() -> None:
    class NativeOpportunityEnvironment(_AlarmEnvironment):
        def snapshot(self) -> dict:
            return {
                **super().snapshot(),
                "decision_opportunity": self.tick == 2,
            }

    class ImmediateDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(Action())
            return future

    driver = ImmediateDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=NativeOpportunityEnvironment(alarm_tick=None, horizon=4),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.015,
    ).run(timeout_s=2.0)

    assert [event.kind for event in driver.started] == ["session_start"]
    assert not [
        event for event in artifact["events"] if event["kind"] == "native_opportunity"
    ]


def test_standing_plan_wake_if_controls_optional_realtime_alarm() -> None:
    class OptionalAlarmEnvironment(_AlarmEnvironment):
        def step(self, action: Action) -> StepReturn:
            result = super().step(action)
            for event in result.info.realized_events:
                event.pop("decision_required", None)
                event["event_class"] = "alarm"
            return result

    class PlanDriver(_PlanReviewDriver):
        def __init__(self, wake_if: list[str]) -> None:
            super().__init__()
            self.wake_if = wake_if

        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            if event.kind == "session_start":
                future.set_result(
                    Action(
                        tool_calls=[
                            ToolCall(
                                name="commit_to_plan",
                                args={
                                    "review_after_ticks": 3,
                                    "wake_if": list(self.wake_if),
                                },
                            )
                        ]
                    )
                )
            else:
                future.set_result(Action())
            return future

    unsubscribed = PlanDriver([])
    unsubscribed_artifact = RealtimeEpisodeCoordinator(
        env=OptionalAlarmEnvironment(alarm_tick=2, horizon=4),
        turn_driver=unsubscribed,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    ).run(timeout_s=2.0)

    subscribed = PlanDriver(["visible_event"])
    subscribed_artifact = RealtimeEpisodeCoordinator(
        env=OptionalAlarmEnvironment(alarm_tick=2, horizon=4),
        turn_driver=subscribed,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    ).run(timeout_s=2.0)

    assert "environment_alarm" not in [event.kind for event in unsubscribed.started]
    suppressed = next(
        event
        for event in unsubscribed_artifact["events"]
        if event["kind"] == "environment_alarm"
    )
    assert suppressed["dispatch_suppressed_reason"] == "PLAN_WAKE_NOT_SUBSCRIBED"
    assert "environment_alarm" in [event.kind for event in subscribed.started]
    assert subscribed_artifact["turns"]


def test_revised_standing_plan_supersedes_prior_review_schedule() -> None:
    driver = _DelayedStubDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=2),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    )
    coordinator._turns = [  # noqa: SLF001
        {"turn_id": "turn-1"},
        {"turn_id": "turn-2"},
    ]
    try:
        coordinator._ingest_confirmed_plan_reviews(  # noqa: SLF001
            {
                "turn_id": "turn-1",
                "simulator_tick": 1,
                "submitted_action": {
                    "actions": [
                        {
                            "name": "commit_to_plan",
                            "call_id": "plan-1",
                            "args": {"review_after_ticks": 3},
                        }
                    ]
                },
                "tool_results": [
                    {"name": "commit_to_plan", "call_id": "plan-1", "ok": True}
                ],
            }
        )
        coordinator._ingest_confirmed_plan_reviews(  # noqa: SLF001
            {
                "turn_id": "turn-2",
                "simulator_tick": 2,
                "submitted_action": {
                    "actions": [
                        {
                            "name": "commit_to_plan",
                            "call_id": "plan-2",
                            "args": {"review_after_ticks": 5},
                        }
                    ]
                },
                "tool_results": [
                    {"name": "commit_to_plan", "call_id": "plan-2", "ok": True}
                ],
            }
        )

        assert coordinator._scheduled_review_ticks == [7]  # noqa: SLF001
        assert coordinator._turns[1]["superseded_scheduled_review_ticks"] == [  # noqa: SLF001
            4
        ]
    finally:
        driver.close()


def test_same_tick_investigation_does_not_create_periodic_provider_turn() -> None:
    class InvestigationCadenceEnvironment(_AlarmEnvironment):
        def snapshot(self) -> dict:
            return {
                **super().snapshot(),
                "decision_cadence": {
                    "periodic_scan_every_ticks": 2,
                    "max_review_after_ticks": 4,
                },
            }

        def readonly_tool_names(self) -> set[str]:
            return {"inspect_queue"}

        def execute_investigation(
            self, action: Action
        ) -> tuple[dict, list[ToolResult]]:
            return self.snapshot(), [
                ToolResult(
                    name="inspect_queue",
                    ok=True,
                    state_changing=False,
                    call_id=action.tool_calls[0].call_id,
                )
            ]

    class InvestigateThenObserveDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(
                Action(tool_calls=[ToolCall(name="inspect_queue")])
                if event.kind == "session_start"
                else Action()
            )
            return future

    driver = InvestigateThenObserveDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=InvestigationCadenceEnvironment(alarm_tick=None, horizon=3),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2.0)

    investigation = next(
        row for row in artifact["transitions"] if row["simulator_time_advanced"] is False
    )
    assert investigation["simulator_tick_before"] == 0
    assert investigation["simulator_tick"] == 0
    assert [event.kind for event in driver.started] == ["session_start", "tool_result"]
    assert not [
        event for event in artifact["events"] if event["kind"] == "supervisory_scan"
    ]
    assert artifact["episode_status"] == "complete"


def test_slow_direct_turn_is_terminal_fenced_without_synthetic_deadline() -> None:
    env = _AlarmEnvironment(alarm_tick=None, horizon=3)
    driver = _DelayedStubDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    ).run(timeout_s=2.0)

    turn = artifact["turns"][0]
    assert turn["deadline_met"] is None
    assert turn["decision_latency_ticks"] >= 1
    assert turn["execution_fence"] == "late_response_audit_only"
    assert turn["hard_cancel_performed"] is False
    assert "late_initial_command" not in env.applied
    assert artifact["harness"]["provider_turn_hard_cancel_supported"] is False
    assert artifact["harness"]["cancellation_semantics"] == (
        "logical_supersession_with_execution_fence"
    )


def test_scheduled_review_control_remains_unlabeled_without_evaluator_evidence() -> None:
    class ProactiveControlDriver(_PlanReviewDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            if event.kind != "scheduled_review":
                return super().start_turn(
                    turn_id=turn_id,
                    observation=observation,
                    event=event,
                )
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(Action(tool_calls=[ToolCall(name="reroute")]))
            return future

    env = _AlarmEnvironment(alarm_tick=None, horizon=5)
    driver = ProactiveControlDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.015,
    ).run(timeout_s=2.0)

    review_turn = next(
        turn for turn in artifact["turns"] if turn["trigger_kind"] == "scheduled_review"
    )
    assert "proactive_action_necessary" not in review_turn
    assert "proactive_action_basis" not in review_turn
    assert artifact["diagnostics"]["alarm_response"]["false_alarms"] == 0
    assert artifact["diagnostics"]["alarm_response"][
        "false_alarm_unassessed_interventions"
    ] == 1
    assert artifact["diagnostics"]["alarm_response"]["false_alarm_rate"] is None
    assert artifact["diagnostics"]["alarm_response"]["correct_silence"] == 1
    assert artifact["diagnostics"]["alarm_response"][
        "model_standing_plan_quiet_windows"
    ] == 2


def test_explicit_model_wait_runs_tool_protocol_and_is_not_takeover() -> None:
    class WaitDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(Action(tool_calls=[ToolCall(name="wait")]))
            return future

    env = _AlarmEnvironment(alarm_tick=None, horizon=1)
    driver = WaitDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    ).run(timeout_s=2.0)

    assert env.applied == ["wait"]
    assert artifact["transitions"][0]["action_source"] == "model"
    assert artifact["turns"][0]["deliberate_wait"] is True
    assert artifact["turns"][0]["decision_no_action"] is False
    assert artifact["turns"][0]["action_id"]
    assert artifact["turns"][0]["receipt_status"] == "no_effect"
    assert artifact["action_receipts"][0]["status"] == "no_effect"
    assert artifact["diagnostics"]["safety"]["takeovers"] == 0


def test_realtime_unknown_event_fails_closed_without_alarm_turn() -> None:
    class UnknownEventEnvironment(_AlarmEnvironment):
        def step(self, action: Action) -> StepReturn:
            result = super().step(action)
            result.info.realized_events = [
                {
                    "event_id": "unknown-1",
                    "type": "unregistered_telemetry_shape",
                    "hidden": False,
                }
            ]
            return result

    env = UnknownEventEnvironment(alarm_tick=None, horizon=2)
    driver = _DelayedStubDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.015,
    ).run(timeout_s=2.0)

    assert [event.kind for event in driver.started] == ["session_start"]
    assert artifact["diagnostics"]["alarm_response"]["actionable_alarms"] == 0
    assert artifact["event_contract"]["violation_count"] == 2
    assert artifact["event_contract"]["violations"][0]["violation_codes"] == [
        "missing_event_decision_contract"
    ]


def test_agent_owned_review_resumes_without_tick_polling() -> None:
    env = _AlarmEnvironment(alarm_tick=None, horizon=4)
    driver = _PlanReviewDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    ).run(timeout_s=2.0)

    assert [event.kind for event in driver.started] == [
        "session_start",
        "tool_result",
        "scheduled_review",
    ]
    assert "commit_to_plan" in env.applied
    assert artifact["diagnostics"]["autonomy"]["scheduled_reviews"] == 1
    assert artifact["diagnostics"]["autonomy"]["scheduled_reviews_served"] == 1
    assert artifact["diagnostics"]["autonomy"]["unnecessary_polling"] == 0


def test_alarm_does_not_drop_future_agent_owned_review() -> None:
    env = _AlarmEnvironment(alarm_tick=1, horizon=5)
    driver = _PlanReviewDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    ).run(timeout_s=2.0)

    review = next(
        event for event in artifact["events"] if event["kind"] == "scheduled_review"
    )
    assert review["simulator_tick"] == 3
    assert artifact["diagnostics"]["autonomy"]["scheduled_reviews"] == 1
    assert artifact["diagnostics"]["autonomy"]["scheduled_reviews_served"] == 1


def test_due_review_is_coalesced_when_alarm_arrives_on_same_tick() -> None:
    env = _AlarmEnvironment(alarm_tick=3, horizon=5)
    driver = _PlanReviewDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    ).run(timeout_s=2.0)

    same_tick = [
        event
        for event in artifact["events"]
        if event["simulator_tick"] == 3
        and event["kind"] in {"environment_alarm", "scheduled_review"}
    ]
    assert [event["kind"] for event in same_tick] == [
        "environment_alarm",
        "scheduled_review",
    ]
    assert same_tick[1]["queued_behind_event_id"] == same_tick[0]["event_id"]
    assert same_tick[1]["dispatched_from_pending"] is True


def test_failed_plan_commit_uses_tool_protocol_feedback_without_scheduling() -> None:
    class RejectedPlanEnvironment(_AlarmEnvironment):
        def step(self, action: Action) -> StepReturn:
            result = super().step(action)
            for tool_result in result.tool_results:
                if tool_result.name == "commit_to_plan":
                    tool_result.ok = False
                    tool_result.error_code = "PLAN_REJECTED"
            return result

    env = RejectedPlanEnvironment(alarm_tick=None, horizon=4)
    driver = _PlanReviewDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    ).run(timeout_s=2.0)

    assert "commit_to_plan" in env.applied
    assert "tool_failure" in [event.kind for event in driver.started]
    assert "scheduled_review" not in [event.kind for event in driver.started]
    assert artifact["turns"][0].get("scheduled_review_tick") is None


def test_native_steer_updates_turn_version_without_canceling_it() -> None:
    class EvidenceAlarmEnvironment(_AlarmEnvironment):
        def step(self, action: Action) -> StepReturn:
            result = super().step(action)
            for event in result.info.realized_events:
                event["evidence_ids"] = ["alarm-evidence"]
            return result

    env = EvidenceAlarmEnvironment(horizon=4)
    driver = _NativeSteerDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2.0)

    assert driver.canceled == []
    assert "reroute" in env.applied
    assert len(artifact["turns"]) == 1
    assert artifact["turns"][0]["steered_event_ids"]
    assert artifact["turns"][0]["trigger_kind"] == "session_start"
    assert artifact["turns"][0]["delivered_event_ids"] == [
        artifact["events"][0]["event_id"],
        artifact["turns"][0]["steered_event_ids"][0],
    ]
    assert artifact["turns"][0]["steer_envelopes"][0]["kind"] == (
        "environment_alarm"
    )
    assert artifact["turns"][0]["late_response_discarded"] is False
    assert artifact["turns"][0]["based_on_visible_evidence_ids"] == [
        "alarm-evidence"
    ]


def test_direct_agent_driver_never_mutates_one_semantic_session_concurrently() -> None:
    agent = _ConcurrencyProbeAgent()
    driver = AgentTurnDriver(agent, [])
    try:
        first = driver.start_turn(turn_id="t1", observation={}, event=_event(1))
        second = driver.start_turn(turn_id="t2", observation={}, event=_event(2))
        first.result(timeout=1.0)
        second.result(timeout=1.0)
        assert agent.max_active == 1
        assert driver.capabilities()["fallback_interrupt"] == (
            "logical_supersession_serial_resume"
        )
    finally:
        driver.close()


def test_agent_driver_waits_boundedly_for_canceled_turn_settlement() -> None:
    class CancellableAgent(_TransactionalAgent):
        def realtime_capabilities(self) -> dict[str, bool]:
            return {"stream_cancel_supported": True}

        def cancel_realtime_turn(self, *, turn_id: str, reason: str) -> bool:
            del turn_id, reason
            self.release.set()
            return True

    agent = CancellableAgent()
    driver = AgentTurnDriver(agent, [])
    try:
        future = driver.start_turn(
            turn_id="terminal-turn",
            observation={},
            event=_event(1),
        )
        assert agent.entered.wait(timeout=1.0)
        assert driver.cancel_turn(
            turn_id="terminal-turn",
            reason="ENVIRONMENT_CLOSED",
        ) is True

        assert driver.wait_for_behavioral_settlement(timeout_s=0.5) is True
        assert future.done() is True
        assert driver.outstanding_turn_count() == 0
        assert agent.history == []
    finally:
        driver.close()


def test_coordinator_returns_when_serialized_observation_ingest_exceeds_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingIngestAgent:
        def __init__(self) -> None:
            self.ingest_started = threading.Event()
            self.release_ingest = threading.Event()
            self.ingest_finished = threading.Event()
            self.closed = threading.Event()
            self.closed_while_ingest = False

        def snapshot_behavioral_state(self) -> dict:
            return {}

        def restore_behavioral_state(self, snapshot: dict) -> None:
            del snapshot

        def act(self, observation: dict, tool_specs: list[dict]) -> Action:
            del observation, tool_specs
            return Action()

        def get_interaction_stats(self) -> dict:
            if self.ingest_started.is_set() and not self.ingest_finished.is_set():
                raise AssertionError("must not snapshot stats during ingest")
            return {}

        def ingest_realtime_observation(self, observation: dict) -> None:
            del observation
            self.ingest_started.set()
            try:
                assert self.release_ingest.wait(timeout=1.0)
            finally:
                self.ingest_finished.set()

        def close(self) -> None:
            self.closed_while_ingest = not self.ingest_finished.is_set()
            self.closed.set()

    monkeypatch.setattr(
        realtime_episode,
        "_PROVIDER_CANCEL_SETTLEMENT_GRACE_S",
        0.02,
    )
    agent = HangingIngestAgent()
    driver = AgentTurnDriver(agent, [])
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.005,
    )

    started = time.monotonic()
    artifact = coordinator.run(timeout_s=0.2)

    assert time.monotonic() - started < 0.5
    assert agent.ingest_started.is_set()
    assert artifact["teardown"]["behavioral_settlement_complete"] is False
    assert artifact["environment_observation_ingestion"]["pending"] == 1
    assert agent.closed.is_set() is False

    agent.release_ingest.set()
    assert driver.wait_for_behavioral_settlement(timeout_s=0.5) is True
    assert agent.closed.wait(timeout=0.5)
    assert agent.closed_while_ingest is False


def test_environment_done_waits_remaining_budget_for_outstanding_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SlowAgent(_TransactionalAgent):
        def act(self, observation: dict, tool_specs: list) -> Action:
            del observation, tool_specs
            time.sleep(0.12)
            return Action()

    monkeypatch.setattr(
        realtime_episode,
        "_PROVIDER_CANCEL_SETTLEMENT_GRACE_S",
        0.02,
    )
    driver = AgentTurnDriver(SlowAgent(), [])
    try:
        started = time.monotonic()
        artifact = RealtimeEpisodeCoordinator(
            env=_AlarmEnvironment(alarm_tick=None, horizon=1),
            turn_driver=driver,
            safety_supervisor=_SafetySupervisor(),
            tick_interval_s=0.005,
        ).run(timeout_s=0.4)
        elapsed = time.monotonic() - started
    finally:
        driver.close(wait=True)

    assert elapsed < 0.35
    assert artifact["clock"]["timed_out"] is False
    assert artifact["teardown"]["behavioral_settlement_complete"] is True
    assert artifact["teardown"]["behavioral_settlement_grace_s"] > 0.02
    assert artifact["clock"]["outstanding_provider_turns_at_return"] == 0


def test_non_waking_observation_ingestion_is_serialized_after_provider_turn() -> None:
    class IngestAgent(_TransactionalAgent):
        def __init__(self) -> None:
            super().__init__()
            self.ingested: list[str] = []

        def ingest_realtime_observation(self, observation: dict) -> None:
            event = observation["__last_realized_events__"][0]
            self.ingested.append(str(event["type"]))

    agent = IngestAgent()
    driver = AgentTurnDriver(agent, [])
    try:
        active = driver.start_turn(turn_id="active", observation={}, event=_event(1))
        assert agent.entered.wait(timeout=1.0)
        ingested = driver.ingest_observation(
            {
                "__last_realized_events__": [
                    {
                        "type": "load_surge_cleared",
                        "decision_required": False,
                        "hidden": False,
                    }
                ]
            }
        )
        assert ingested.done() is False
        assert driver.cancel_turn(turn_id="active", reason="clear-event") is False
        agent.release.set()
        active.result(timeout=1.0)
        ingested.result(timeout=1.0)
        assert agent.history == []
        assert agent.ingested == ["load_surge_cleared"]
    finally:
        driver.close()


def test_coordinator_ingests_each_transition_observation_not_latest_snapshot() -> None:
    class CapturingDriver(_DelayedStubDriver):
        def __init__(self) -> None:
            super().__init__()
            self.ingested_ticks: list[int] = []

        def ingest_observation(self, observation: dict) -> Future[None]:
            self.ingested_ticks.append(int(observation["tick"]))
            future: Future[None] = Future()
            future.set_result(None)
            return future

    driver = CapturingDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=1.0,
    )
    try:
        coordinator._ingest_transition_observation(  # noqa: SLF001
            {"agent_visible_observation_after": {"tick": 1}}
        )
        coordinator._ingest_transition_observation(  # noqa: SLF001
            {"agent_visible_observation_after": {"tick": 2}}
        )
        coordinator._ingest_transition_observation(  # noqa: SLF001
            {"simulator_time_advanced": False}
        )
    finally:
        driver.close()

    assert driver.ingested_ticks == [1, 2]

    class AsyncIngestDriver(_DelayedStubDriver):
        def __init__(self) -> None:
            super().__init__()
            self.ingest_started = threading.Event()
            self.release_ingest = threading.Event()

        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(
                Action(
                    tool_calls=[
                        ToolCall(
                            name="wait" if event.kind == "session_start" else "reroute"
                        )
                    ]
                )
            )
            return future

        def ingest_observation(self, observation: dict) -> Future[None]:
            del observation

            def ingest() -> None:
                self.ingest_started.set()
                assert self.release_ingest.wait(timeout=1.0)

            return self._pool.submit(ingest)

    async_driver = AsyncIngestDriver()
    async_coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=1, horizon=2),
        turn_driver=async_driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        run = pool.submit(async_coordinator.run, timeout_s=2.0)
        assert async_driver.ingest_started.wait(timeout=1.0)
        assert [event.kind for event in async_driver.started] == ["session_start"]
        deadline = time.monotonic() + 1.0
        while not any(
            event.get("queued_reason") == "OBSERVATION_INGEST_PENDING"
            for event in async_coordinator._events  # noqa: SLF001
        ):
            assert time.monotonic() < deadline
            time.sleep(0.005)
        async_driver.release_ingest.set()
        artifact = run.result(timeout=2.0)

    assert "environment_alarm" in [event.kind for event in async_driver.started]
    assert any(
        event.get("queued_reason") == "OBSERVATION_INGEST_PENDING"
        for event in artifact["events"]
    )


def test_nonadvancing_rejection_does_not_create_quiet_window() -> None:
    driver = _DelayedStubDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=1.0,
    )
    try:
        coordinator._process_transition(  # noqa: SLF001
            {
                "state_version_after": 0,
                "simulator_tick": 0,
                "simulator_time_advanced": False,
                "rejection_reason": "INVISIBLE_EVIDENCE_REFERENCE",
            }
        )
        events = list(coordinator._events)  # noqa: SLF001
    finally:
        driver.close()

    assert events == []


def test_nonadvancing_rejection_reconciles_before_next_clock_tick() -> None:
    class ReadonlyEnvironment(_AlarmEnvironment):
        def __init__(self) -> None:
            super().__init__(alarm_tick=None, horizon=1)
            self.step_started = threading.Event()

        def readonly_tool_names(self) -> set[str]:
            return {"inspect"}

        def step(self, action: Action) -> StepReturn:
            self.step_started.set()
            return super().step(action)

    class RejectingSupervisor(_SafetySupervisor):
        def arbitrate(
            self,
            *,
            observation: dict,
            simulator_tick: int,
            candidate_action: Action,
        ) -> SafetyDecision:
            del observation, simulator_tick, candidate_action
            return SafetyDecision(
                action=Action(tool_calls=[ToolCall(name="minimum_risk_hold")]),
                mode="native_runtime_takeover",
                reason_code="READ_BLOCKED",
                disposition="reject",
            )

    class ReceiptProbeDriver(_DelayedStubDriver):
        def __init__(self) -> None:
            super().__init__()
            self.receipt_seen = threading.Event()

        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            if event.kind == "session_start":
                future.set_result(Action(tool_calls=[ToolCall(name="inspect")]))
            else:
                if event.kind == "action_receipt":
                    self.receipt_seen.set()
                future.set_result(Action())
            return future

    env = ReadonlyEnvironment()
    driver = ReceiptProbeDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=RejectingSupervisor(),
        tick_interval_s=0.4,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        run_future = pool.submit(coordinator.run, timeout_s=2.0)
        assert driver.receipt_seen.wait(timeout=0.2)
        assert env.step_started.is_set() is False
        assert env.tick == 0
        artifact = run_future.result(timeout=2.0)

    assert any(event.kind == "action_receipt" for event in driver.started)
    assert artifact["action_receipts"][0]["status"] == "rejected"


def test_rejected_action_rolls_back_behavioral_state_but_keeps_provider_audit() -> None:
    class AuditedTransactionalAgent:
        def __init__(self) -> None:
            self.history: list[str] = []
            self.requests: list[dict] = []
            self.responses: list[dict] = []
            self.identities: list[dict] = []

        def snapshot_behavioral_state(self) -> dict:
            return {"history": list(self.history)}

        def restore_behavioral_state(self, snapshot: dict) -> None:
            self.history = list(snapshot["history"])

        def get_interaction_stats(self) -> dict:
            return {
                "provider_request_records": self.requests,
                "provider_response_records": self.responses,
                "provider_model_identity_records": self.identities,
            }

        def act(self, observation: dict, tool_specs: list[dict]) -> Action:
            del tool_specs
            turn_id = str(observation["__decision_epoch__"]["turn_id"])
            event_kind = str(observation["__realtime_event__"]["kind"])
            self.history.append(turn_id)
            sequence = len(self.requests) + 1
            self.requests.append({"sequence": sequence, "turn_marker": turn_id})
            self.responses.append(
                {
                    "request_sequence": sequence,
                    "response": {"status": "success"},
                }
            )
            self.identities.append(
                {
                    "schema_version": "provider_model_identity_closure_v1",
                    "request_sequence": sequence,
                    "requested_model": "transaction-test-model",
                    "observed_models": ["transaction-test-model"],
                    "closure": "exact",
                }
            )
            return Action(
                tool_calls=[
                    ToolCall(
                        name="reroute" if event_kind == "session_start" else "wait"
                    )
                ]
            )

    class RejectFirstSupervisor(_SafetySupervisor):
        def __init__(self) -> None:
            self.rejected = False

        def arbitrate(
            self,
            *,
            observation: dict,
            simulator_tick: int,
            candidate_action: Action,
        ) -> SafetyDecision:
            del observation, simulator_tick
            if not self.rejected:
                self.rejected = True
                return SafetyDecision(
                    action=Action(tool_calls=[ToolCall(name="minimum_risk_hold")]),
                    mode="native_runtime_takeover",
                    reason_code="TEST_REJECTION",
                    disposition="reject",
                )
            return SafetyDecision(
                action=candidate_action,
                mode="model_action",
                reason_code="MODEL_ACTION_ACCEPTED",
                disposition="pass",
            )

    agent = AuditedTransactionalAgent()
    driver = AgentTurnDriver(agent, [])
    artifact = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=2),
        turn_driver=driver,
        safety_supervisor=RejectFirstSupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2.0)

    rejected_turn = next(
        turn for turn in artifact["turns"] if turn.get("receipt_status") == "rejected"
    )
    assert rejected_turn["behavioral_transaction_status"] == "rolled_back"
    assert rejected_turn["turn_id"] not in agent.history
    rejected_audit = next(
        row
        for row in artifact["provider_audit"]
        if row["turn_id"] == rejected_turn["turn_id"]
    )
    assert rejected_audit["provider_requests"][0]["turn_marker"] == (
        rejected_turn["turn_id"]
    )
    assert rejected_audit["behavioral_state_outcome"] == "rolled_back"
    assert rejected_audit["behavioral_transaction_consistent"] is True


def test_direct_driver_requires_behavioral_transaction_hooks() -> None:
    class UnsafeAgent:
        def act(self, observation: dict, tool_specs: list[dict]) -> Action:
            del observation, tool_specs
            return Action()

    with pytest.raises(ValueError, match="behavioral transaction hooks"):
        AgentTurnDriver(UnsafeAgent(), [])

    class UnsafeCustomDriver(_DelayedStubDriver):
        commit_turn = None
        rollback_turn = None

    unsafe_driver = UnsafeCustomDriver()
    try:
        with pytest.raises(ValueError, match="behavioral transaction hooks"):
            RealtimeEpisodeCoordinator(
                env=_AlarmEnvironment(alarm_tick=None, horizon=1),
                turn_driver=unsafe_driver,
                safety_supervisor=_SafetySupervisor(),
                tick_interval_s=0.02,
            )
    finally:
        unsafe_driver.close()


def test_running_superseded_turn_rolls_back_behavioral_state_before_next_turn() -> None:
    agent = _TransactionalAgent()
    driver = AgentTurnDriver(agent, [])
    try:
        first = driver.start_turn(turn_id="t1", observation={}, event=_event(1))
        assert agent.entered.wait(timeout=1.0)
        assert driver.cancel_turn(turn_id="t1", reason="alarm") is False
        agent.release.set()
        first.result(timeout=1.0)
        assert agent.history == []

        agent.entered.clear()
        second = driver.start_turn(turn_id="t2", observation={}, event=_event(2))
        assert agent.entered.wait(timeout=1.0)
        second.result(timeout=1.0)
        driver.commit_turn("t2")
        assert agent.history == ["t2"]
    finally:
        driver.close()


def test_completed_uncommitted_turn_can_rollback_after_driver_forgets_future() -> None:
    agent = _TransactionalAgent()
    agent.release.set()
    driver = AgentTurnDriver(agent, [])
    try:
        completed = driver.start_turn(
            turn_id="t-complete",
            observation={},
            event=_event(1),
        )
        completed.result(timeout=1.0)
        assert agent.history == ["t-complete"]
        assert driver.outstanding_turn_count() == 0

        assert driver.cancel_turn(turn_id="t-complete", reason="late-alarm") is False
        assert agent.history == []
        assert driver.commit_turn("t-complete") is False
    finally:
        driver.close()


def test_arbitration_winner_commits_before_delayed_terminal_receipt() -> None:
    agent = _TransactionalAgent()
    agent.release.set()
    driver = AgentTurnDriver(agent, [])
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=2),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=1.0,
    )
    try:
        turn_id = "winner"
        completed = driver.start_turn(
            turn_id=turn_id,
            observation={},
            event=_event(1),
        )
        completed.result(timeout=1.0)
        coordinator._turns.append(  # noqa: SLF001
            {
                "turn_id": turn_id,
                "action_id": "action-winner",
                "status": "awaiting_arbitration",
                "behavioral_transaction_status": "awaiting_arbitration",
            }
        )
        coordinator._pending_behavioral_turn_id = turn_id  # noqa: SLF001

        assert (
            coordinator._invalidate_all_turns(  # noqa: SLF001
                reason="ENVIRONMENT_CLOSED",
                defer_pending_arbitration=True,
            )
            == 0
        )

        coordinator._process_transition(  # noqa: SLF001
            {
                "turn_id": turn_id,
                "action_id": "action-winner",
                "state_version_after": 1,
                "simulator_tick": 1,
                "simulator_time_advanced": True,
                "action_source": "model",
                "safety_decision": {"disposition": "pass"},
                "tool_results": [
                    {
                        "call_id": "delayed-call",
                        "name": "mutating_action",
                        "ok": True,
                        "state_changing": True,
                        "latency_ticks": 2,
                        "payload": {"_status": "pending", "due_tick": 3},
                    }
                ],
            }
        )

        turn = coordinator._turn_record(turn_id)  # noqa: SLF001
        assert turn["behavioral_transaction_status"] == "committed"
        assert turn["behavioral_transaction_reason"] == "ACTION_WON_ARBITRATION"
        assert coordinator._pending_behavioral_turn_id is None  # noqa: SLF001
        assert agent.history == [turn_id]
        assert driver.commit_turn(turn_id) is False
        coordinator._receipt_queue.put(  # noqa: SLF001
            (
                turn_id,
                {
                    "status": "failed",
                    "turn_id": turn_id,
                    "action_id": "action-winner",
                    "decision_id": "decision-winner",
                },
            )
        )
        coordinator._drain_receipts()  # noqa: SLF001

        assert turn["behavioral_transaction_status"] == "committed"
        assert turn["execution_status"] == "failed"
        assert agent.history == [turn_id]
        assert any(event["kind"] == "action_receipt" for event in coordinator._events)  # noqa: SLF001
        assert (
            coordinator._invalidate_all_turns(  # noqa: SLF001
                reason="ENVIRONMENT_CLOSED"
            )
            == 0
        )
    finally:
        driver.close()


def test_queued_canceled_turn_has_complete_zero_request_audit_range() -> None:
    agent = _TransactionalAgent()
    driver = AgentTurnDriver(agent, [])
    try:
        first = driver.start_turn(turn_id="running", observation={}, event=_event(1))
        assert agent.entered.wait(timeout=1.0)
        queued = driver.start_turn(turn_id="queued", observation={}, event=_event(2))
        assert driver.cancel_turn(turn_id="queued", reason="superseded") is True
        agent.release.set()
        first.result(timeout=1.0)
        with pytest.raises(CancelledError):
            queued.result(timeout=1.0)

        queued_audit = next(
            row
            for row in driver.provider_audit_records()
            if row["turn_id"] == "queued"
        )
        assert queued_audit["provider_started"] is False
        assert queued_audit["provider_turn_settled"] is True
        assert queued_audit["provider_requests"] == []
        assert queued_audit["provider_responses"] == []
    finally:
        driver.close()


def test_queued_cancel_audit_has_no_phantom_late_response() -> None:
    class QueuedCancelDriver(_DelayedStubDriver):
        def cancellation_outcome(self, turn_id: str) -> dict[str, bool]:
            del turn_id
            return {
                "queued_future_canceled": True,
                "provider_stream_canceled": False,
            }

        def capabilities(self) -> dict[str, bool]:
            return {"provider_turn_hard_cancel_supported": True}

    driver = QueuedCancelDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=1.0,
    )
    record = {
        "late_response_pending": True,
        "late_response_discarded": True,
    }
    try:
        coordinator._record_cancel_audit(  # noqa: SLF001
            record,
            turn_id="queued-turn",
            cancel_acknowledged=True,
        )
    finally:
        driver.close()

    assert record["cancellation_mode"] == "queued_future_canceled"
    assert record["late_response_pending"] is False
    assert record["late_response_discarded"] is False


def test_terminal_model_tool_failure_is_unanswerable_without_blocking_artifact() -> None:
    driver = _DelayedStubDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=1.0,
    )
    try:
        coordinator._process_transition(  # noqa: SLF001
            {
                "state_version_after": 1,
                "simulator_tick": 1,
                "simulator_time_advanced": True,
                "environment_done": True,
                "action_source": "model",
                "tool_results": [
                    {
                        "name": "control",
                        "ok": False,
                        "error_code": "DEADLINE_EXCEEDS_EPISODE",
                        "state_changing": True,
                    }
                ],
                "realized_events": [],
            }
        )
    finally:
        driver.close()

    [failure] = coordinator._events  # noqa: SLF001
    assert failure["kind"] == "tool_failure"
    assert failure["decision_required"] is True
    assert failure["terminal_dispatch_suppressed"] is True
    assert failure["dispatch_suppressed_reason"] == "ENVIRONMENT_DONE"
    assert failure["terminal_unanswerable"] is True
    assert failure["terminal_trigger_origin"] == "model_action_feedback"
    assert failure["terminal_formal_blocker"] is False
    assert driver.started == []

    diagnostics = evaluate_realtime_diagnostics(
        events=coordinator._events,  # noqa: SLF001
        turns=[],
        transitions=[],
        lifecycle=[],
    )
    assert diagnostics["trigger_response"]["actionable"] == 0
    assert diagnostics["trigger_response"]["delivery_missed"] == 0
    assert diagnostics["trigger_response"]["response_missed"] == 0
    assert diagnostics["trigger_response"]["terminal_unanswerable"] == 1

    artifact = {
        "episode_status": "complete",
        "provider_audit_contract": {"complete": True},
        "tool_surface_contract": {"complete": True},
        "evidence_closure": {"closure_complete": True},
        "event_contract": {"violation_count": 0},
        "events": coordinator._events,  # noqa: SLF001
        "teardown": {
            "actor_stopped": True,
            "unsafe_teardown": False,
            "environment_close_allowed": True,
        },
    }
    realtime_episode._apply_realtime_artifact_validation(
        artifact, behavioral_state_settled=True
    )
    assert "TERMINAL_ACTIONABLE_TRIGGER_UNDELIVERABLE" not in artifact[
        "artifact_validation"
    ]["blocker_codes"]


@pytest.mark.parametrize(
    ("transition_feedback", "expected_kind"),
    [
        (
            {
                "realized_events": [
                    {
                        "event_id": "terminal-alarm",
                        "type": "capacity_alarm",
                        "event_class": "alarm",
                        "decision_required": True,
                        "hidden": False,
                    }
                ]
            },
            "environment_alarm",
        ),
        ({"early_stop_warnings": ["unsafe_margin"]}, "safety_warning"),
    ],
)
def test_terminal_alarm_and_warning_remain_in_actionable_denominator(
    transition_feedback: dict,
    expected_kind: str,
) -> None:
    driver = _DelayedStubDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=1.0,
    )
    try:
        coordinator._process_transition(  # noqa: SLF001
            {
                "state_version_after": 1,
                "simulator_tick": 1,
                "simulator_time_advanced": True,
                "environment_done": True,
                "action_source": "safety_supervisor",
                "tool_results": [],
                "realized_events": [],
                **transition_feedback,
            }
        )
    finally:
        driver.close()

    [event] = coordinator._events  # noqa: SLF001
    assert event["kind"] == expected_kind
    assert event["decision_required"] is True
    assert event["terminal_unanswerable"] is True
    assert event["terminal_trigger_origin"] == "environment_or_harness"
    assert event["terminal_formal_blocker"] is True
    assert event["dispatch_suppressed_reason"] == "ENVIRONMENT_DONE"
    assert driver.started == []

    artifact = {
        "episode_status": "complete",
        "provider_audit_contract": {"complete": True},
        "tool_surface_contract": {"complete": True},
        "evidence_closure": {"closure_complete": True},
        "event_contract": {"violation_count": 0},
        "events": coordinator._events,  # noqa: SLF001
        "teardown": {
            "actor_stopped": True,
            "unsafe_teardown": False,
            "environment_close_allowed": True,
        },
    }
    realtime_episode._apply_realtime_artifact_validation(
        artifact, behavioral_state_settled=True
    )
    assert "TERMINAL_ACTIONABLE_TRIGGER_UNDELIVERABLE" not in artifact[
        "artifact_validation"
    ]["blocker_codes"]


def test_episode_timeout_invalidates_turn_without_claiming_hard_provider_cancel() -> None:
    env = _AlarmEnvironment(alarm_tick=None, horizon=100)
    driver = _TimeoutDriver()
    started = time.monotonic()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.01,
    ).run(timeout_s=0.04)
    elapsed = time.monotonic() - started
    try:
        assert elapsed < 0.3
        assert artifact["episode_status"] == "timed_out"
        assert artifact["clock"]["timed_out"] is True
        assert artifact["clock"]["provider_turn_hard_timeout_enforced"] is False
        assert artifact["clock"]["process_exit_hard_deadline"] is False
        assert artifact["clock"]["outstanding_provider_turns_at_return"] == 1
        assert artifact["turns"][0]["invalidated_reason"] == "EPISODE_WALL_TIMEOUT"
        assert artifact["turns"][0]["late_response_pending"] is True
        assert driver.close_wait is False
        assert "too_late" not in env.applied
    finally:
        driver.release.set()
        driver.pool.shutdown(wait=True)
    assert "too_late" not in env.applied


def test_tool_result_continuation_supports_inspect_then_commit_without_tick_prompt() -> None:
    env = _InspectEnvironment(alarm_tick=None, horizon=4)
    driver = _InspectThenCommitDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2.0)

    assert [event.kind for event in driver.started] == [
        "session_start",
        "tool_result",
    ]
    assert "inspect_queue" in env.applied
    assert "commit_control" in env.applied
    tool_event = next(
        event for event in artifact["events"] if event["kind"] == "tool_result"
    )
    assert tool_event["payload"]["tool_results"][0]["name"] == "inspect_queue"
    assert artifact["episode_outcome"]["cumulative_reward"] == 1.0
    assert "evidence-0-inspect_queue" in artifact["episode_outcome"][
        "authoritative_evidence_ids"
    ]


def test_successful_state_changing_control_does_not_self_trigger_new_turn() -> None:
    class ControlOnlyDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(
                Action(tool_calls=[ToolCall(name="commit_control")])
                if event.kind == "session_start"
                else Action()
            )
            return future

    env = _InspectEnvironment(alarm_tick=None, horizon=3)
    driver = ControlOnlyDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2.0)

    assert [event.kind for event in driver.started] == ["session_start"]
    assert "commit_control" in env.applied
    assert not {
        "tool_result",
        "delayed_tool",
    }.intersection(event["kind"] for event in artifact["events"])


def test_stale_action_receipt_reconciles_and_triggers_replan() -> None:
    class StaleThenReplanDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)

            def complete() -> Action:
                if event.kind == "session_start":
                    time.sleep(0.05)
                    return Action(tool_calls=[ToolCall(name="late_control")])
                if event.kind == "action_receipt":
                    return Action(tool_calls=[ToolCall(name="reroute")])
                return Action()

            return self._pool.submit(complete)

    env = _AlarmEnvironment(alarm_tick=None, horizon=8)
    driver = StaleThenReplanDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.015,
    ).run(timeout_s=2.0)

    assert "late_control" not in env.applied
    assert "reroute" in env.applied
    assert "action_receipt" in [event.kind for event in driver.started]
    stale_turn = next(
        turn
        for turn in artifact["turns"]
        if turn.get("receipt_status") == "stale"
    )
    receipt_event = next(
        event for event in artifact["events"] if event["kind"] == "action_receipt"
    )
    assert stale_turn["deadline_met"] is None
    assert stale_turn["behavioral_transaction_status"] == "rolled_back"
    assert receipt_event["payload"]["receipt"]["turn_id"] == stale_turn["turn_id"]
    assert receipt_event["payload"]["submitted_tool_calls"][0]["name"] == (
        "late_control"
    )


def test_delayed_tool_feedback_uses_typed_delayed_continuation() -> None:
    class DelayedEnvironment(_InspectEnvironment):
        def step(self, action: Action) -> StepReturn:
            result = super().step(action)
            for tool_result in result.tool_results:
                tool_result.latency_ticks = 2
            return result

    env = DelayedEnvironment(alarm_tick=None, horizon=3)
    driver = _InspectThenCommitDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2.0)

    assert "delayed_tool" in [event.kind for event in driver.started]
    delayed_event = next(
        event for event in artifact["events"] if event["kind"] == "delayed_tool"
    )
    assert delayed_event["payload"]["event_class"] == "task"
    assert delayed_event["payload"]["tool_results"][0]["latency_ticks"] == 2


def test_pending_delayed_ack_is_silent_but_terminal_result_wakes() -> None:
    pending, pending_delayed = _continuation_tool_results(
        [
            {
                "name": "inspect_queue",
                "ok": True,
                "state_changing": False,
                "latency_ticks": 2,
                "payload": {"_status": "pending", "due_tick": 3},
            }
        ]
    )
    terminal, terminal_delayed = _continuation_tool_results(
        [
            {
                "name": "inspect_queue",
                "ok": True,
                "state_changing": False,
                "latency_ticks": 2,
                "payload": {"queue_depth": 7},
            }
        ]
    )

    assert pending == []
    assert pending_delayed is False
    assert terminal[0]["payload"] == {"queue_depth": 7}
    assert terminal_delayed is True


def test_terminal_delayed_result_wakes_from_later_safety_transition() -> None:
    driver = _DelayedStubDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=2),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=1.0,
    )
    coordinator._accept_turn_results = False  # noqa: SLF001
    try:
        coordinator._process_transition(  # noqa: SLF001
            {
                "state_version_after": 1,
                "simulator_tick": 0,
                "simulator_time_advanced": False,
                "action_source": "model",
                "tool_results": [
                    {
                        "call_id": "inspect-call",
                        "name": "inspect_queue",
                        "ok": True,
                        "state_changing": False,
                        "latency_ticks": 2,
                        "payload": {"_status": "pending", "due_tick": 2},
                    }
                ],
            }
        )
        coordinator._process_transition(  # noqa: SLF001
            {
                "state_version_after": 2,
                "simulator_tick": 1,
                "simulator_time_advanced": True,
                "action_source": "safety_supervisor",
                "tool_results": [
                    {
                        "call_id": "inspect-call",
                        "name": "inspect_queue",
                        "ok": True,
                        "state_changing": False,
                        "latency_ticks": 2,
                        "payload": {"queue_depth": 7},
                    }
                ],
            }
        )
    finally:
        driver.close()

    delayed_events = [
        event
        for event in coordinator._events  # noqa: SLF001
        if event["kind"] == "delayed_tool"
    ]
    assert len(delayed_events) == 1
    assert delayed_events[0]["payload"]["tool_results"][0]["payload"] == {
        "queue_depth": 7
    }


def test_superseded_delayed_a_is_canceled_before_materialization() -> None:
    class MixedDelayedEnvironment(_AlarmEnvironment):
        def __init__(self) -> None:
            super().__init__(alarm_tick=1, horizon=3)
            self.evidence = EvidenceLogger("mixed-delayed-causality")
            self._tools = ToolRegistry(seed=42)

            def delayed_handler(_args: dict, ctx: ToolContext) -> dict:
                del ctx
                return {"_status": "applied"}

            self._tools.register(
                ToolSpec(
                    name="delayed_control",
                    description="Delayed control.",
                    parameters={"type": "object", "properties": {}},
                    handler=delayed_handler,
                    state_changing=True,
                    semantic_role="control",
                    fail_rate=0.0,
                    delay_ticks=1,
                )
            )
            self._tools.register(
                ToolSpec(
                    name="wait",
                    description="Wait.",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _args, _ctx: {"_status": "observed"},
                    state_changing=False,
                    fail_rate=0.0,
                    delay_ticks=0,
                )
            )

        def get_tool_specs(self) -> list[dict]:
            return [
                self._tools.get(name).to_openai_schema()
                for name in self._tools.names()
                if self._tools.get(name) is not None
            ]

        def step(self, action: Action) -> StepReturn:
            results = self._tools.execute_action(
                action,
                ToolContext(
                    tick=self.tick,
                    seed=42,
                    extra={"episode_horizon": self.horizon, "env": self},
                ),
            )
            events: list[dict] = []
            evidence_ids: list[str] = []
            if self.tick == 0:
                events.append(
                    {
                        "event_id": "alarm-for-b",
                        "type": "capacity_alarm",
                        "event_class": "task",
                        "decision_required": True,
                        "actionable": True,
                        "hidden": False,
                    }
                )
            for result in results:
                if (
                    result.name != "delayed_control"
                    or str(result.payload.get("_status") or "") == "pending"
                ):
                    continue
                event = {
                    "event_id": "effect-a",
                    "type": "control_effect",
                    "event_class": "observation",
                    "decision_required": False,
                    "actionable": False,
                    "origin": "agent_caused",
                    "agent_caused": True,
                    "call_id": result.call_id,
                    "tool_name": "delayed_control",
                    "requested_action": {"name": "delayed_control", "args": {}},
                    "before_state_digest": "before",
                    "after_state_digest": "after",
                }
                result.evidence_id = self.evidence.log(
                    "realized_event", tick=self.tick, payload=event, source="engine"
                )
                evidence_ids.append(result.evidence_id)
                events.append(event)
            self.tick += 1
            return StepReturn(
                observation=self.snapshot(),
                tool_results=results,
                reward=0.0,
                done=self.tick >= self.horizon,
                info=StepInfo(
                    realized_events=events,
                    evidence_ids=evidence_ids,
                ),
            )

    class AThenBDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(
                Action(
                    tool_calls=[
                        ToolCall(
                            name=(
                                "delayed_control"
                                if event.kind == "session_start"
                                else "wait"
                            )
                        )
                    ]
                )
            )
            return future

    env = MixedDelayedEnvironment()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=AThenBDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.05,
    ).run(timeout_s=2.0)

    turn_a, turn_b = artifact["turns"][:2]
    assert turn_a["receipt_status"] == "superseded"
    assert turn_b["receipt_status"] == "no_effect"
    cancellation_audit = [
        entry
        for row in artifact["transitions"]
        for entry in row.get("cancellation_audit", [])
    ]
    assert len(cancellation_audit) == 1
    assert cancellation_audit[0]["call_id"]
    assert cancellation_audit[0] | {"call_id": "<normalized>"} == {
        "call_id": "<normalized>",
        "queue_kind": "registry_invocation",
        "outcome": "canceled",
        "callback_invoked": False,
        "callback_error_type": None,
    }
    assert not any(
        event.get("event_id") == "effect-a"
        for row in artifact["transitions"]
        for event in row.get("realized_events") or []
    )
    assert not any(
        event["kind"] == "delayed_tool"
        and (event.get("payload") or {}).get("source_turn_id") == turn_a["turn_id"]
        for event in artifact["events"]
    )
    assert not any(
        event["kind"] == "action_receipt"
        and (event.get("payload") or {}).get("receipt", {}).get("action_id")
        == turn_a["action_id"]
        for event in artifact["events"]
    )


def test_real_registry_delayed_readonly_query_is_quiet_until_terminal_once() -> None:
    class DelayedReadonlyEnvironment(_AlarmEnvironment):
        def __init__(self) -> None:
            super().__init__(alarm_tick=None, horizon=3)
            self._tools = ToolRegistry(seed=42)
            self._tools.register(
                ToolSpec(
                    name="forecast_query",
                    description="Delayed forecast query.",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _args, _ctx: {"forecast": 7.0},
                    state_changing=False,
                    semantic_role="investigation",
                    fail_rate=0.0,
                    delay_ticks=1,
                )
            )
            self._tools.register(
                ToolSpec(
                    name="wait",
                    description="Wait.",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _args, _ctx: {"_status": "observed"},
                    state_changing=False,
                    fail_rate=0.0,
                    delay_ticks=0,
                )
            )

        def get_tool_specs(self) -> list[dict]:
            return [
                self._tools.get(name).to_openai_schema()
                for name in self._tools.names()
                if self._tools.get(name) is not None
            ]

        def execute_investigation(
            self, action: Action
        ) -> tuple[dict, list[ToolResult]]:
            return self.snapshot(), self._tools.execute_action(
                action,
                ToolContext(
                    tick=self.tick,
                    seed=42,
                    extra={"episode_horizon": self.horizon, "env": self},
                ),
            )

        def step(self, action: Action) -> StepReturn:
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
                reward=0.0,
                done=self.tick >= self.horizon,
                info=StepInfo(),
            )

    class ForecastDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(
                Action(
                    tool_calls=[
                        ToolCall(
                            name=(
                                "forecast_query"
                                if event.kind == "session_start"
                                else "wait"
                            )
                        )
                    ]
                )
            )
            return future

    driver = ForecastDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=DelayedReadonlyEnvironment(),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.04,
    ).run(timeout_s=2.0)

    first_turn = artifact["turns"][0]
    assert first_turn["receipt_status"] == "no_effect"
    assert [
        row["status"]
        for row in artifact["action_lifecycle"]
        if row["action_id"] == first_turn["action_id"]
    ] == ["queued", "accepted", "pending", "applied", "no_effect"]
    assert [event.kind for event in driver.started].count("delayed_tool") == 1
    assert sum(
        event["kind"] == "delayed_tool" for event in artifact["events"]
    ) == 1
    assert not any(
        event["kind"] == "action_receipt"
        and (event.get("payload") or {}).get("receipt", {}).get("action_id")
        == first_turn["action_id"]
        for event in artifact["events"]
    )


def test_coordinator_wires_native_protocol_repair_metrics() -> None:
    class RepairStatsDriver(_DelayedStubDriver):
        def interaction_stats(self) -> dict[str, int]:
            return {
                "llm_calls_ok": 6,
                "llm_calls_failed": 0,
                "protocol_repair_attempts": 5,
                "protocol_repair_successes": 5,
            }

    artifact = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=RepairStatsDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.01,
    ).run(timeout_s=1.0)

    protocol = artifact["diagnostics"]["provider_protocol"]
    assert protocol["native_valid_without_repair"] == 1
    assert protocol["repair_dependent_call_rate"] == pytest.approx(5 / 6)


@pytest.mark.parametrize(
    ("info", "expected_kind"),
    [
        (StepInfo(forecast_updates={"load": "rising"}), "forecast_update"),
        (StepInfo(early_stop_warnings=["unsafe_margin"]), "safety_warning"),
    ],
)
def test_typed_forecast_and_safety_feedback_trigger_via_declared_contract(
    info: StepInfo,
    expected_kind: str,
) -> None:
    class FeedbackEnvironment(_AlarmEnvironment):
        def step(self, action: Action) -> StepReturn:
            result = super().step(action)
            if self.tick == 1:
                result.info = info
            return result

    env = FeedbackEnvironment(alarm_tick=None, horizon=3)
    driver = _DelayedStubDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    ).run(timeout_s=2.0)

    assert expected_kind in [event.kind for event in driver.started]
    typed_event = next(
        event for event in artifact["events"] if event["kind"] == expected_kind
    )
    assert typed_event["payload"]["event_class"] in {"forecast", "safety"}


def test_same_transition_dispatches_typed_triggers_in_priority_order() -> None:
    class ConcurrentFeedbackEnvironment(_InspectEnvironment):
        def step(self, action: Action) -> StepReturn:
            result = super().step(action)
            if self.tick == 1:
                result.info.early_stop_warnings = ["unsafe_margin"]
                result.info.forecast_updates = {"load": "rising"}
            return result

    env = ConcurrentFeedbackEnvironment(alarm_tick=None, horizon=3)
    driver = _InspectThenCommitDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.03,
    ).run(timeout_s=2.0)

    assert [event.kind for event in driver.started] == [
        "session_start",
        "safety_warning",
        "forecast_update",
        "tool_result",
    ]
    same_tick = [
        event
        for event in artifact["events"]
        if event["simulator_tick"] == 1
        and event["kind"] in {"safety_warning", "forecast_update", "tool_result"}
    ]
    assert [event["kind"] for event in same_tick] == [
        "safety_warning",
        "forecast_update",
        "tool_result",
    ]
    assert all(
        event.get("queued_behind_event_id") == same_tick[0]["event_id"]
        for event in same_tick[1:]
    )
    assert all(event.get("dispatched_from_pending") for event in same_tick[1:])


def test_pending_trigger_without_domain_deadline_keeps_original_identity() -> None:
    class SlowSafetyDriver(_InspectThenCommitDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            if event.kind != "safety_warning":
                return super().start_turn(
                    turn_id=turn_id,
                    observation=observation,
                    event=event,
                )
            self.started.append(event)

            def complete() -> Action:
                time.sleep(0.06)
                return Action()

            return self._pool.submit(complete)

    class ConcurrentFeedbackEnvironment(_InspectEnvironment):
        def step(self, action: Action) -> StepReturn:
            result = super().step(action)
            if self.tick == 1:
                result.info.early_stop_warnings = ["unsafe_margin"]
                result.info.forecast_updates = {"load": "rising"}
            return result

    env = ConcurrentFeedbackEnvironment(alarm_tick=None, horizon=20)
    driver = SlowSafetyDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.01,
    ).run(timeout_s=2.0)

    original = next(
        event
        for event in artifact["events"]
        if event["kind"] == "forecast_update"
        and event.get("current_state_continuation") is not True
    )
    assert original["deadline_tick"] is None
    assert original.get("pending_expired") is not True
    assert original["event_id"] in {
        event_id
        for turn in artifact["turns"]
        for event_id in turn["delivered_event_ids"]
    }
    assert not any(event.get("current_state_continuation") for event in artifact["events"])


def test_pending_event_is_audited_without_dispatch_after_environment_done() -> None:
    driver = _DelayedStubDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=1.0,
    )
    try:
        pending = coordinator._new_event(  # noqa: SLF001
            kind="delayed_tool",
            state_version=0,
            simulator_tick=0,
            decision_required=True,
            priority=10,
            payload={"type": "delayed_tool"},
        )
        coordinator._queue_pending_event(  # noqa: SLF001
            pending,
            reason="LOWER_OR_EQUAL_PRIORITY",
        )
        with coordinator._actor._condition:  # noqa: SLF001
            coordinator._actor._done = True  # noqa: SLF001

        coordinator._dispatch_next_pending()  # noqa: SLF001

        event = next(
            row
            for row in coordinator._events  # noqa: SLF001
            if row["event_id"] == pending.event_id
        )
        assert event["terminal_dispatch_suppressed"] is True
        assert event["dispatch_suppressed_reason"] == "ENVIRONMENT_DONE"
        assert event["terminal_unanswerable"] is True
        assert coordinator._pending_events == []  # noqa: SLF001
        assert driver.started == []
    finally:
        driver.close()


def test_direct_event_is_not_admitted_after_environment_done() -> None:
    driver = _DelayedStubDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=1.0,
    )
    try:
        event = coordinator._new_event(  # noqa: SLF001
            kind="environment_alarm",
            state_version=1,
            simulator_tick=1,
            decision_required=True,
            priority=100,
        )
        with coordinator._actor._condition:  # noqa: SLF001
            coordinator._actor._done = True  # noqa: SLF001

        coordinator._interrupt_or_steer(event)  # noqa: SLF001

        record = next(
            row
            for row in coordinator._events  # noqa: SLF001
            if row["event_id"] == event.event_id
        )
        assert record["terminal_dispatch_suppressed"] is True
        assert record["dispatch_suppressed_reason"] == "ENVIRONMENT_DONE"
        assert record["terminal_unanswerable"] is True
        assert coordinator._turns == []  # noqa: SLF001
        assert driver.started == []
    finally:
        driver.close()


def test_turn_admission_defers_without_blocking_during_clock_dispatch() -> None:
    driver = _DelayedStubDriver()
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=1.0,
    )
    event = coordinator._new_event(  # noqa: SLF001
        kind="delayed_tool",
        state_version=0,
        simulator_tick=0,
        decision_required=True,
        priority=10,
    )
    with coordinator._actor._condition:  # noqa: SLF001
        coordinator._actor._active_clock_dispatch = {  # noqa: SLF001
            "dispatch_seq": 1,
            "scheduled_monotonic_ns": time.monotonic_ns(),
            "started_monotonic_ns": time.monotonic_ns(),
        }
    try:
        started = time.monotonic()
        coordinator._interrupt_or_steer(event)  # noqa: SLF001

        assert time.monotonic() - started < 0.1
        assert driver.started == []
        assert coordinator._pending_events == [event]  # noqa: SLF001

        with coordinator._actor._condition:  # noqa: SLF001
            coordinator._actor._done = True  # noqa: SLF001
            coordinator._actor._active_clock_dispatch = None  # noqa: SLF001
            coordinator._actor._condition.notify_all()  # noqa: SLF001
        coordinator._dispatch_next_pending()  # noqa: SLF001

        record = next(
            row
            for row in coordinator._events  # noqa: SLF001
            if row["event_id"] == event.event_id
        )
        assert record["terminal_dispatch_suppressed"] is True
        assert record["dispatch_suppressed_reason"] == "ENVIRONMENT_DONE"
        assert record["terminal_unanswerable"] is True
        assert coordinator._turns == []  # noqa: SLF001
        assert driver.started == []
    finally:
        driver.close()


def test_realtime_diagnostics_reports_latency_and_all_lifecycle_rates() -> None:
    report = evaluate_realtime_diagnostics(
        events=[
            {
                "event_id": "a1",
                "decision_id": "d1",
                "kind": "environment_alarm",
                "simulator_tick": 2,
                "monotonic_ns": 100,
                "decision_required": True,
            },
            {
                "event_id": "q1",
                "kind": "quiet_window",
                "simulator_tick": 3,
                "monotonic_ns": 200,
                "decision_required": False,
            },
        ],
        turns=[
            {
                "trigger_event_id": "a1",
                "started_tick": 2,
                "started_monotonic_ns": 120,
                "decision_tick": 3,
                "decision_monotonic_ns": 150,
                "status": "completed",
                "decision_id": "d1",
                "action_id": "action-1",
                "action_is_wait": False,
                "receipt_status": "effected",
            }
        ],
        transitions=[
            {
                "state_version_after": 4,
                "monotonic_ns": 190,
                "action_source": "model",
                "decision_id": "d1",
                "action_id": "action-1",
                "effect_observed": True,
                "effect_evidence_ids": ["effect-1"],
                "tool_trace_edges": [
                    {"call_id": "call-1", "effect_proven": True}
                ],
            },
            {"state_version_after": 5, "action_source": "safety_supervisor"},
        ],
        lifecycle=[
            {"status": "stale"},
            {"status": "canceled"},
            {"status": "superseded"},
            {"status": "no_effect"},
            {
                "status": "effected",
                "decision_id": "d1",
                "action_id": "action-1",
            },
        ],
        polling_events=1,
    )

    assert report["schema_version"] == "realtime-diagnostics/1.7"
    assert report["alarm_response"]["detected"] == 1
    assert report["alarm_response"]["correct_silence"] == 0
    assert report["latency"]["alarm_to_decision_ticks"]["mean"] == 1.0
    assert report["latency"]["alarm_to_effect_wall_ms"]["mean"] == 0.00009
    assert report["action_lifecycle"]["stale"] == 1
    assert report["action_lifecycle"]["canceled"] == 1
    assert report["action_lifecycle"]["superseded"] == 1
    assert report["action_lifecycle"]["no_effect"] == 1
    assert report["safety"]["takeovers"] == 0
    assert report["autonomy"]["unnecessary_polling"] == 1


def test_diagnostics_do_not_credit_failed_wait_or_unrelated_effects() -> None:
    events = [
        {
            "event_id": f"alarm-{index}",
            "decision_id": f"decision-{index}",
            "kind": "environment_alarm",
            "simulator_tick": index,
            "monotonic_ns": index * 100,
            "decision_required": True,
        }
        for index in range(1, 4)
    ]
    turns = [
        {
            "trigger_event_id": "alarm-1",
            "decision_id": "decision-1",
            "status": "failed",
        },
        {
            "trigger_event_id": "alarm-2",
            "decision_id": "decision-2",
            "status": "completed",
            "action_id": None,
            "action_is_wait": True,
            "receipt_status": "no_action",
        },
        {
            "trigger_event_id": "alarm-3",
            "decision_id": "decision-3",
            "status": "completed",
            "action_id": "action-3",
            "action_is_wait": False,
            "receipt_status": "no_effect",
            "decision_tick": 3,
            "decision_monotonic_ns": 350,
            "started_monotonic_ns": 320,
        },
    ]
    report = evaluate_realtime_diagnostics(
        events=events,
        turns=turns,
        transitions=[
            {
                "action_source": "model",
                "decision_id": "unrelated-decision",
                "action_id": "unrelated-action",
                "effect_observed": True,
                "state_version_after": 4,
                "monotonic_ns": 400,
            }
        ],
        lifecycle=[
            {
                "status": "effected",
                "decision_id": "unrelated-decision",
                "action_id": "unrelated-action",
            },
            {
                "status": "no_effect",
                "decision_id": "decision-3",
                "action_id": "action-3",
            },
        ],
    )

    assert report["alarm_response"]["detected"] == 2
    assert report["alarm_response"]["missed"] == 2
    assert report["latency"]["alarm_to_decision_ticks"]["count"] == 1
    assert report["latency"]["alarm_to_effect_ticks"]["count"] == 0


def test_diagnostics_treat_truncated_provider_output_as_missed_not_silence() -> None:
    report = evaluate_realtime_diagnostics(
        events=[
            {
                "event_id": "alarm-1",
                "kind": "environment_alarm",
                "simulator_tick": 1,
                "monotonic_ns": 100,
                "decision_required": True,
            }
        ],
        turns=[
            {
                "trigger_event_id": "alarm-1",
                "delivered_event_ids": ["alarm-1"],
                "status": "completed",
                "decision_tick": 1,
                "decision_valid": False,
                "invalid_decision_reason": "provider_output_truncated",
                "receipt_status": "invalid_model_output",
            }
        ],
        transitions=[],
        lifecycle=[],
    )

    assert report["trigger_response"]["delivered"] == 1
    assert report["trigger_response"]["acknowledged"] == 0
    assert report["trigger_response"]["decided"] == 0
    assert report["trigger_response"]["decision_no_action"] == 0
    assert report["trigger_response"]["missed"] == 1
    assert report["autonomy"]["invalid_model_responses"] == 1


def test_alarm_scorecard_does_not_label_tool_results_as_alarms() -> None:
    report = evaluate_realtime_diagnostics(
        events=[
            {
                "event_id": "tool-result-1",
                "kind": "tool_result",
                "simulator_tick": 1,
                "monotonic_ns": 100,
                "decision_required": True,
            }
        ],
        turns=[],
        transitions=[],
        lifecycle=[],
    )

    assert report["trigger_response"]["actionable"] == 1
    assert report["alarm_response"]["actionable_alarms"] == 0


def test_hung_provider_is_delivered_but_not_detected_or_decided() -> None:
    report = evaluate_realtime_diagnostics(
        events=[
            {
                "event_id": "alarm-hung",
                "kind": "environment_alarm",
                "decision_required": True,
                "simulator_tick": 1,
                "monotonic_ns": 100,
            }
        ],
        turns=[
            {
                "turn_id": "turn-hung",
                "trigger_event_id": "alarm-hung",
                "delivered_event_ids": ["alarm-hung"],
                "status": "superseded",
                "decision_tick": None,
            }
        ],
        transitions=[],
        lifecycle=[],
    )

    assert report["trigger_response"]["delivered"] == 1
    assert report["trigger_response"]["detected"] == 0
    assert report["trigger_response"]["decision_missed"] == 1
    assert report["trigger_response"]["response_missed"] == 1


def test_diagnostics_stage_typed_triggers_without_requiring_effect() -> None:
    kinds = [
        "environment_alarm",
        "safety_warning",
        "forecast_update",
        "tool_failure",
        "delayed_tool",
    ]
    events = [
        {
            "event_id": f"event-{index}",
            "decision_id": f"decision-{index}",
            "kind": kind,
            "simulator_tick": index,
            "monotonic_ns": index * 100,
            "decision_required": True,
        }
        for index, kind in enumerate(kinds, start=1)
    ]
    turns = [
        {
            "turn_id": f"turn-{index}",
            "trigger_event_id": event["event_id"],
            "decision_id": event["decision_id"],
            "trigger_kind": event["kind"],
            "status": "completed",
            "decision_tick": index + 1,
            "decision_monotonic_ns": index * 100 + 50,
            "action_id": f"action-{index}" if index < 4 else None,
            "action_is_wait": index >= 4,
            "receipt_status": "no_effect" if index < 4 else "no_action",
        }
        for index, event in enumerate(events, start=1)
    ]

    report = evaluate_realtime_diagnostics(
        events=events,
        turns=turns,
        transitions=[],
        lifecycle=[],
    )

    assert report["trigger_response"]["actionable"] == 5
    assert report["trigger_response"]["delivered"] == 5
    assert report["trigger_response"]["decided"] == 5
    assert report["trigger_response"]["acted"] == 3
    assert report["trigger_response"]["effected"] == 0
    assert report["alarm_response"]["detected"] == 5
    assert report["alarm_response"]["missed"] == 0


def test_diagnostics_layers_no_action_and_quiet_interval_false_alarm() -> None:
    report = evaluate_realtime_diagnostics(
        events=[
            {
                "event_id": "alarm-action",
                "kind": "environment_alarm",
                "decision_required": True,
                "simulator_tick": 1,
                "monotonic_ns": 100,
            },
            {
                "event_id": "alarm-no-action",
                "kind": "environment_alarm",
                "decision_required": True,
                "simulator_tick": 2,
                "monotonic_ns": 200,
            },
            {
                "event_id": "quiet-3",
                "kind": "quiet_window",
                "decision_required": False,
                "simulator_tick": 3,
            },
            {
                "event_id": "quiet-4",
                "kind": "quiet_window",
                "decision_required": False,
                "simulator_tick": 4,
            },
        ],
        turns=[
            {
                "trigger_event_id": "alarm-action",
                "delivered_event_ids": ["alarm-action"],
                "status": "completed",
                "decision_tick": 1,
                "action_id": "action-1",
                "receipt_status": "confirmed",
                "action_is_wait": False,
            },
            {
                "trigger_event_id": "alarm-no-action",
                "delivered_event_ids": ["alarm-no-action"],
                "status": "completed",
                "decision_tick": 2,
                "action_id": None,
                "receipt_status": "no_action",
                "action_is_wait": True,
            },
            {
                "trigger_event_id": "scheduled-3",
                "trigger_kind": "scheduled_review",
                "started_tick": 3,
                "status": "completed",
                "decision_tick": 3,
                "action_id": "proactive-3",
                "receipt_status": "confirmed",
                "action_is_wait": False,
                "proactive_action_necessary": False,
            },
        ],
        transitions=[],
        lifecycle=[],
    )

    assert report["trigger_response"] == {
        "terminal_unanswerable": 0,
        "actionable": 2,
        "detected": 2,
        "delivered": 2,
        "acknowledged": 2,
        "decided": 2,
        "acted": 1,
        "effected": 0,
        "decision_no_action": 1,
        "missed": 0,
        "delivery_missed": 0,
        "decision_missed": 0,
        "response_missed": 0,
        "by_kind": {
            "environment_alarm": {
                "actionable": 2,
                "detected": 2,
                "delivered": 2,
                "acknowledged": 2,
                "decided": 2,
                "acted": 1,
                "effected": 0,
                "decision_no_action": 1,
                "delivery_missed": 0,
                "decision_missed": 0,
                "response_missed": 0,
            }
        },
    }
    assert report["alarm_response"]["detected"] == 2
    assert report["alarm_response"]["missed"] == 0
    assert report["alarm_response"]["false_alarms"] == 1
    assert report["alarm_response"]["correct_silence"] == 0


def test_correct_silence_requires_model_confirmed_standing_plan_or_hold() -> None:
    report = evaluate_realtime_diagnostics(
        events=[
            {
                "event_id": "quiet-harness",
                "kind": "quiet_window",
                "decision_required": False,
                "simulator_tick": 1,
            },
            {
                "event_id": "quiet-plan",
                "kind": "quiet_window",
                "decision_required": False,
                "simulator_tick": 2,
                "model_confirmed_standing_plan": True,
                "silence_attribution": "model_standing_plan",
            },
        ],
        turns=[
            {
                "trigger_event_id": "scheduled-1",
                "trigger_kind": "scheduled_review",
                "started_tick": 1,
                "status": "completed",
                "decision_valid": True,
                "decision_tick": 1,
                "action_id": None,
                "receipt_status": "no_action",
                "action_is_wait": True,
            }
        ],
        transitions=[],
        lifecycle=[],
    )

    assert report["alarm_response"]["agent_silence_opportunities"] == 1
    assert report["alarm_response"]["correct_silence"] == 1
    assert report["harness_environment"] == {
        "quiet_windows": 2,
        "quiet_windows_without_model_turn": 1,
        "unattributed_quiet_windows": 1,
    }


def test_pending_coalescing_requires_explicit_key_and_preserves_merge_closure() -> None:
    coordinator = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=2),
        turn_driver=_DelayedStubDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    )

    first_unkeyed = coordinator._new_event(  # noqa: SLF001
        kind="environment_alarm",
        state_version=1,
        simulator_tick=1,
        decision_required=True,
        priority=100,
        payload={"type": "capacity_alarm", "generator": "a"},
        evidence_ids=["evidence-a"],
        deadline_tick=3,
    )
    second_unkeyed = coordinator._new_event(  # noqa: SLF001
        kind="environment_alarm",
        state_version=1,
        simulator_tick=1,
        decision_required=True,
        priority=100,
        payload={"type": "capacity_alarm", "generator": "b"},
        evidence_ids=["evidence-b"],
        deadline_tick=4,
    )
    coordinator._queue_pending_event(first_unkeyed, reason="test")  # noqa: SLF001
    coordinator._queue_pending_event(second_unkeyed, reason="test")  # noqa: SLF001

    first_keyed = coordinator._new_event(  # noqa: SLF001
        kind="environment_alarm",
        state_version=1,
        simulator_tick=1,
        decision_required=True,
        priority=100,
        payload={"type": "capacity_alarm", "coalesce_key": "generator:c"},
        evidence_ids=["evidence-c1"],
        deadline_tick=5,
    )
    second_keyed = coordinator._new_event(  # noqa: SLF001
        kind="environment_alarm",
        state_version=2,
        simulator_tick=2,
        decision_required=True,
        priority=100,
        payload={"type": "capacity_alarm", "coalesce_key": "generator:c"},
        evidence_ids=["evidence-c2"],
        deadline_tick=4,
    )
    coordinator._queue_pending_event(first_keyed, reason="test")  # noqa: SLF001
    coordinator._queue_pending_event(second_keyed, reason="test")  # noqa: SLF001

    pending = coordinator._pending_events  # noqa: SLF001
    assert [event.event_id for event in pending[:2]] == [
        first_unkeyed.event_id,
        second_unkeyed.event_id,
    ]
    assert len(pending) == 3
    merged = pending[-1]
    assert merged.event_id == second_keyed.event_id
    assert merged.evidence_ids == ("evidence-c1", "evidence-c2")
    assert merged.deadline_tick == 4
    merged_record = next(
        event
        for event in coordinator._events  # noqa: SLF001
        if event["event_id"] == merged.event_id
    )
    assert merged_record["merged_event_ids"] == [first_keyed.event_id]
    assert merged_record["merged_deadline_ticks"] == [4, 5]

    report = evaluate_realtime_diagnostics(
        events=coordinator._events,  # noqa: SLF001
        turns=[
            {
                "trigger_event_id": merged.event_id,
                "delivered_event_ids": [merged.event_id],
                "status": "completed",
                "decision_tick": 2,
                "action_id": None,
                "receipt_status": "no_action",
                "action_is_wait": True,
            }
        ],
        transitions=[],
        lifecycle=[],
    )
    assert report["trigger_response"]["decision_no_action"] == 2
    assert report["trigger_response"]["missed"] == 2


def test_controlled_hold_is_not_counted_as_a_safety_takeover() -> None:
    report = evaluate_realtime_diagnostics(
        events=[],
        turns=[],
        transitions=[
                {
                    "action_source": "safety_supervisor",
                    "safety_decision": {"mode": "controlled_hold"},
                    "applied_action": {"actions": [{"name": "wait"}]},
                },
                {
                    "decision_id": "decision-1",
                    "action_id": "action-1",
                    "state_version_before": 1,
                    "state_version_after": 2,
                    "simulator_tick_before": 1,
                    "simulator_tick": 2,
                    "simulator_time_advanced": True,
                    "action_source": "safety_supervisor",
                    "safety_decision": {
                        "mode": "minimum_risk_fallback",
                        "evidence_ids": ["safety-1"],
                    },
                    "safety_evidence_ids": ["safety-1"],
                    "applied_action": {
                        "actions": [
                            {"name": "minimum_risk_hold", "call_id": "mrm-1"}
                        ]
                    },
                    "effect_observed": True,
                    "effect_evidence_ids": ["effect-1"],
                    "tool_trace_edges": [
                        {
                            "call_id": "mrm-1",
                            "effect_proven": True,
                            "effect_evidence_ids": ["effect-1"],
                        }
                    ],
                },
        ],
        lifecycle=[],
        evidence_ledger=[
            {
                "evidence_id": "safety-1",
                    "kind": "runtime_assurance_observation",
                    "source": "engine",
                    "tick": 0,
                },
                {
                    "evidence_id": "effect-1",
                    "kind": "realized_event",
                    "source": "engine",
                    "tick": 1,
                    "payload": {
                        "event_id": "mrm-effect-1",
                        "call_id": "mrm-1",
                        "origin": "agent_caused",
                        "before_state_digest": "before",
                        "after_state_digest": "after",
                    },
                },
            ],
    )

    assert report["safety"]["controlled_holds"] == 1
    assert report["safety"]["takeovers"] == 1


def test_provider_audit_survives_behavioral_rollback_and_is_turn_joined() -> None:
    class AuditedAgent(_TransactionalAgent):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[dict] = []
            self.responses: list[dict] = []
            self.identities: list[dict] = []

        def get_interaction_stats(self) -> dict:
            return {
                "provider_request_records": self.requests,
                "provider_response_records": self.responses,
                "provider_model_identity_records": self.identities,
            }

        def act(self, observation: dict, tool_specs: list[dict]) -> Action:
            turn_id = str(observation["__decision_epoch__"]["turn_id"])
            self.history.append(turn_id)
            self.requests.append({"sequence": 1, "turn_marker": turn_id})
            self.identities.append(
                {
                    "schema_version": "provider_model_identity_closure_v1",
                    "request_sequence": 1,
                    "requested_model": "requested-model",
                    "observed_models": ["requested-model"],
                    "closure": "exact",
                }
            )
            self.entered.set()
            assert self.release.wait(timeout=1.0)
            self.responses.append({"sequence": 1, "request_sequence": 1})
            return Action(tool_calls=[ToolCall(name="mutating_action")])

    agent = AuditedAgent()
    driver = AgentTurnDriver(agent, [])
    try:
        future = driver.start_turn(
            turn_id="paid-superseded",
            observation={},
            event=_event(1),
        )
        assert agent.entered.wait(timeout=1.0)
        driver.cancel_turn(turn_id="paid-superseded", reason="new-alarm")
        agent.release.set()
        future.result(timeout=1.0)

        assert agent.history == []
        audit = driver.provider_audit_records()
        assert audit[0]["turn_id"] == "paid-superseded"
        assert audit[0]["provider_requests"][0]["turn_marker"] == (
            "paid-superseded"
        )
        assert audit[0]["provider_responses"][0]["request_sequence"] == 1
        assert audit[0]["provider_model_identities"][0]["closure"] == "exact"
    finally:
        driver.close()


def test_settled_completed_turn_with_zero_provider_requests_fails_audit() -> None:
    class EmptyFallbackDriver(_DelayedStubDriver):
        def __init__(self) -> None:
            super().__init__()
            self.turn_ids: list[str] = []

        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del observation
            self.turn_ids.append(turn_id)
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(Action())
            return future

        def provider_audit_records(self) -> list[dict]:
            return [
                {
                    "turn_id": turn_id,
                    "provider_requests": [],
                    "provider_responses": [],
                    "provider_turn_settled": True,
                    "provider_started": True,
                    "provider_audit_status": "completed",
                }
                for turn_id in self.turn_ids
            ]

    artifact = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=EmptyFallbackDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    ).run(timeout_s=1.0)

    assert artifact["provider_audit_contract"]["complete"] is False
    assert artifact["provider_audit_contract"]["invalid_turn_ids"]
    assert "PROVIDER_REQUEST_AUDIT_INVALID" in artifact[
        "provider_audit_contract"
    ]["blocker_codes"]


def test_provider_audit_status_and_zero_request_cancellation_are_explicit() -> None:
    valid_transport = {
        "provider_turn_settled": True,
        "provider_started": True,
        "provider_audit_status": "completed",
        "behavioral_transaction_status": "committed",
        "behavioral_state_outcome": "committed",
        "behavioral_transaction_consistent": True,
        "provider_requests": [{"sequence": 1}],
        "provider_responses": [
            {
                "request_sequence": 1,
                "response": {"status": "success"},
            }
        ],
        "provider_model_identities": [
            {
                "schema_version": "provider_model_identity_closure_v1",
                "request_sequence": 1,
                "requested_model": "requested-model",
                "observed_models": ["requested-model"],
                "closure": "exact",
            }
        ],
    }
    assert "PROVIDER_AUDIT_STATUS_INVALID" in (
        realtime_episode._provider_turn_audit_violations(  # noqa: SLF001
            {**valid_transport, "provider_audit_status": "unknown"}
        )
    )
    assert "BEHAVIORAL_TRANSACTION_CLOSURE_INVALID" in (
        realtime_episode._provider_turn_audit_violations(  # noqa: SLF001
            {
                **valid_transport,
                "behavioral_transaction_status": "rolled_back",
            }
        )
    )
    forged_cancellation = {
        "provider_turn_settled": True,
        "provider_started": True,
        "provider_audit_status": "canceled_before_provider_call",
        "provider_requests": [],
        "provider_responses": [],
        "provider_model_identities": [],
    }
    assert "CANCELED_PROVIDER_TURN_LIFECYCLE_INVALID" in (
        realtime_episode._provider_turn_audit_violations(  # noqa: SLF001
            forged_cancellation
        )
    )
    valid_cancellation = {
        **forged_cancellation,
        "provider_started": False,
        "behavioral_transaction_status": "rolled_back",
        "behavioral_state_outcome": "rolled_back",
        "behavioral_transaction_consistent": True,
        "turn_status": "superseded",
        "cancel_requested": True,
        "cancel_acknowledged": True,
        "cancellation_mode": "queued_future_canceled",
        "hard_cancel_performed": False,
        "execution_fence": "late_response_audit_only",
        "late_response_discarded": False,
    }
    assert (
        realtime_episode._provider_turn_audit_violations(  # noqa: SLF001
            valid_cancellation
        )
        == set()
    )


@pytest.mark.parametrize(
    ("response", "identity", "expected_field"),
    [
        (
            {
                "request_sequence": 1,
                "response": {"status": "failed"},
            },
            {
                "schema_version": "provider_model_identity_closure_v1",
                "request_sequence": 1,
                "requested_model": "requested-model",
                "observed_models": [],
                "closure": "request_failed",
            },
            "failed_response_turn_ids",
        ),
        (
            {
                "request_sequence": 1,
                "response": {"status": "success"},
            },
            {
                "schema_version": "provider_model_identity_closure_v1",
                "request_sequence": 1,
                "requested_model": "requested-model",
                "observed_models": ["replacement-model"],
                "closure": "mismatch",
            },
            "model_identity_mismatch_turn_ids",
        ),
    ],
)
def test_provider_audit_fails_closed_on_failed_response_or_model_mismatch(
    response: dict,
    identity: dict,
    expected_field: str,
) -> None:
    class ContaminatedProviderDriver(_DelayedStubDriver):
        def __init__(self) -> None:
            super().__init__()
            self.turn_ids: list[str] = []

        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del observation
            self.turn_ids.append(turn_id)
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(Action())
            return future

        def provider_audit_records(self) -> list[dict]:
            return [
                {
                    "turn_id": turn_id,
                    "provider_requests": [{"sequence": 1}],
                    "provider_responses": [response],
                    "provider_model_identities": [identity],
                    "provider_turn_settled": True,
                    "provider_started": True,
                    "provider_audit_status": "completed",
                }
                for turn_id in self.turn_ids
            ]

    artifact = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=ContaminatedProviderDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    ).run(timeout_s=1.0)

    assert artifact["provider_audit_contract"]["complete"] is False
    assert artifact["provider_audit_contract"][expected_field]


@pytest.mark.parametrize(
    ("response_sequences", "expected_complete"),
    [([2], False), ([1, 2], True)],
)
def test_provider_audit_requires_terminal_response_for_every_request(
    response_sequences: list[int], expected_complete: bool
) -> None:
    class MultiRequestDriver(_DelayedStubDriver):
        def __init__(self) -> None:
            super().__init__()
            self.turn_ids: list[str] = []

        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del observation
            self.turn_ids.append(turn_id)
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(Action())
            return future

        def provider_audit_records(self) -> list[dict]:
            return [
                {
                    "turn_id": turn_id,
                    "provider_requests": [
                        {"sequence": 1},
                        {"sequence": 2},
                    ],
                    "provider_responses": [
                        {
                            "request_sequence": sequence,
                            "response": {"status": "success"},
                        }
                        for sequence in response_sequences
                    ],
                    "provider_model_identities": [
                        {
                            "schema_version": (
                                "provider_model_identity_closure_v1"
                            ),
                            "request_sequence": sequence,
                            "requested_model": "requested-model",
                            "observed_models": ["requested-model"],
                            "closure": "exact",
                        }
                        for sequence in (1, 2)
                    ],
                    "provider_turn_settled": True,
                    "provider_started": True,
                    "provider_audit_status": "completed",
                }
                for turn_id in self.turn_ids
            ]

    artifact = RealtimeEpisodeCoordinator(
        env=_AlarmEnvironment(alarm_tick=None, horizon=1),
        turn_driver=MultiRequestDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.02,
    ).run(timeout_s=1.0)

    assert artifact["provider_audit_contract"]["complete"] is expected_complete


def test_evidence_closure_resolves_every_realtime_reference() -> None:
    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger("realtime-test")

    env = EvidenceEnvironment()
    evidence_id = env.evidence.log("tool_call", tick=1, payload={"ok": True})
    closure = _build_evidence_closure(
        env,
        {
            "events": [{"evidence_ids": [evidence_id]}],
            "transitions": [
                {
                    "step_evidence_ids": [evidence_id],
                    "tool_results": [{"evidence_id": evidence_id}],
                }
            ],
        },
    )

    assert closure["closure_complete"] is True
    assert closure["ledger_count"] == 1
    assert closure["referenced_evidence_ids"] == [evidence_id]
    assert closure["unresolved_evidence_ids"] == []
    assert len(closure["ledger_sha256"]) == 64


def test_evidence_closure_checks_visible_consumes_dependencies_and_effect_join() -> None:
    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger("realtime-consume")

    env = EvidenceEnvironment()
    visible_id = env.evidence.log("observation", tick=0)
    realized_event = {
        "event_id": "control:call-control@1",
        "origin": "agent_caused",
        "call_id": "call-control",
        "outcome_tick": 1,
    }
    effect_id = env.evidence.log(
        "realized_event",
        tick=1,
        payload=realized_event,
        source="engine",
    )
    valid_transition = {
        "action_id": "action-valid",
        "based_on_visible_evidence_ids": [visible_id],
        "submitted_action": {
            "actions": [
                {
                    "name": "control",
                    "call_id": "call-control",
                    "consumes_evidence_ids": [visible_id],
                    "depends_on_call_ids": [],
                }
            ]
        },
        "tool_results": [
            {
                "name": "control",
                "call_id": "call-control",
                "ok": True,
                "state_changing": True,
            }
        ],
        "realized_events": [{**realized_event, "evidence_ids": [effect_id]}],
        "tool_trace_edges": [
            {
                "call_id": "call-control",
                "effect_proven": True,
                "effect_evidence_ids": [effect_id],
            }
        ],
        "effect_observed": True,
    }

    valid = _build_evidence_closure(
        env,
        {"events": [], "transitions": [valid_transition]},
    )
    assert valid["closure_complete"] is True

    invalid_transition = json.loads(json.dumps(valid_transition))
    invalid_transition["submitted_action"]["actions"][0][
        "consumes_evidence_ids"
    ] = [effect_id]
    invalid_transition["submitted_action"]["actions"][0][
        "depends_on_call_ids"
    ] = ["missing-call"]
    invalid = _build_evidence_closure(
        env,
        {"events": [], "transitions": [invalid_transition]},
    )
    assert invalid["closure_complete"] is False
    assert invalid["invisible_consumed_evidence_ids"] == [effect_id]
    assert invalid["dangling_dependency_call_ids"] == ["missing-call"]

    missing_safety = json.loads(json.dumps(valid_transition))
    missing_safety["safety_evidence_ids"] = ["missing-shield-evidence"]
    invalid_safety = _build_evidence_closure(
        env,
        {"events": [], "transitions": [missing_safety]},
    )
    assert invalid_safety["closure_complete"] is False
    assert invalid_safety["unresolved_evidence_ids"] == [
        "missing-shield-evidence"
    ]

    unrelated_safety_id = env.evidence.log("tool_call", tick=1)
    wrong_safety_kind = json.loads(json.dumps(valid_transition))
    wrong_safety_kind["safety_evidence_ids"] = [unrelated_safety_id]
    invalid_safety_kind = _build_evidence_closure(
        env,
        {"events": [], "transitions": [wrong_safety_kind]},
    )
    assert invalid_safety_kind["closure_complete"] is False
    assert invalid_safety_kind["invalid_safety_evidence_ids"] == [
        unrelated_safety_id
    ]

    unproven_mutation = json.loads(json.dumps(valid_transition))
    unproven_mutation["realized_events"][0].update(
        event_id="native-mutation-without-proof",
        before_state_digest="before",
        after_state_digest="after",
    )
    unproven_mutation["tool_trace_edges"] = []
    unproven_mutation["effect_observed"] = False
    invalid_mutation = _build_evidence_closure(
        env,
        {"events": [], "transitions": [unproven_mutation]},
    )
    assert invalid_mutation["closure_complete"] is False
    assert invalid_mutation["unproven_agent_mutation_event_ids"] == [
        "native-mutation-without-proof"
    ]

    missing_digest_mutation = json.loads(json.dumps(valid_transition))
    missing_digest_mutation["realized_events"][0].update(
        event_id="marked-mutation-without-digest",
        changed_state_fields=["native_capacity"],
    )
    missing_digest_mutation["tool_trace_edges"] = []
    missing_digest_mutation["effect_observed"] = False
    invalid_missing_digest = _build_evidence_closure(
        env,
        {"events": [], "transitions": [missing_digest_mutation]},
    )
    assert invalid_missing_digest["closure_complete"] is False
    assert invalid_missing_digest["unproven_agent_mutation_event_ids"] == [
        "marked-mutation-without-digest"
    ]

    notification = json.loads(json.dumps(valid_transition))
    notification["realized_events"][0].update(
        event_id="action-receipt-notification",
        event_class="agent_outcome",
    )
    notification["tool_trace_edges"] = []
    notification["effect_observed"] = False
    notification_closure = _build_evidence_closure(
        env,
        {"events": [], "transitions": [notification]},
    )
    assert notification_closure["closure_complete"] is True
    assert notification_closure["unproven_agent_mutation_event_ids"] == []


def test_evidence_closure_keyed_idempotency_conflict_to_first_registration() -> None:
    """A reused idempotency key must not borrow tick-0 visibility.

    Regression 2026-09-21: the model re-issued ``call-llm_t1_n1`` at tick 8 as
    a different tool; the protocol answered with ``IDEMPOTENCY_KEY_CONFLICT``
    carrying the *new* call's ``consumes_evidence_ids``. The closure compared
    them against the visibility snapshot of the first registration (tick 0),
    where the cited evidence did not exist yet, and failed six otherwise-valid
    driving episodes with ``EVIDENCE_CLOSURE_INCOMPLETE``. The result's own
    transition is the materialization point: evidence visible there is legal.
    """

    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger("realtime-idem-conflict")

    env = EvidenceEnvironment()
    supervisor_wait_id = env.evidence.log(
        "tool_call",
        tick=1,
        payload={"name": "wait", "ok": True, "_status": "waited"},
        source="tool",
    )

    # Transition 0 (tick 0): first registration — inspect_ego_state, no consumes.
    first_transition = {
        "action_id": "action-1",
        "decision_id": "decision-1",
        "simulator_tick": 0,
        "based_on_visible_evidence_ids": [],
        "submitted_action": {
            "actions": [
                {
                    "name": "inspect_ego_state",
                    "call_id": "call-llm_t1_n1",
                    "consumes_evidence_ids": None,
                    "depends_on_call_ids": None,
                }
            ]
        },
        "tool_results": [
            {
                "name": "inspect_ego_state",
                "call_id": "call-llm_t1_n1",
                "ok": True,
                "consumes_evidence_ids": None,
                "payload": {"_status": "pending"},
            }
        ],
    }
    # Transition 9 (tick 8): the key is reused for inspect_safety_state; the
    # protocol materializes the conflict result HERE, and the model's decision
    # legitimately cites evidence that first appeared at tick 1.
    conflict_transition = {
        "action_id": "action-3",
        "decision_id": "decision-3",
        "simulator_tick": 8,
        "based_on_visible_evidence_ids": [supervisor_wait_id],
        "submitted_action": {
            "actions": [
                {
                    "name": "inspect_safety_state",
                    "call_id": "call-llm_t1_n1",
                    "consumes_evidence_ids": [supervisor_wait_id],
                    "depends_on_call_ids": None,
                }
            ]
        },
        "tool_results": [
            {
                "name": "inspect_safety_state",
                "call_id": "call-llm_t1_n1",
                "ok": False,
                "error_code": "IDEMPOTENCY_KEY_CONFLICT",
                "consumes_evidence_ids": [supervisor_wait_id],
                "depends_on_call_ids": None,
                "payload": {
                    "_status": "idempotency_key_conflict",
                    "prior_tool": "inspect_ego_state",
                    "current_tool": "inspect_safety_state",
                },
            }
        ],
    }

    closure = _build_evidence_closure(
        env,
        {"events": [], "transitions": [first_transition, conflict_transition]},
    )
    assert closure["invisible_consumed_evidence_ids"] == []
    assert closure["closure_complete"] is True

    # Control: evidence invisible at BOTH registration and materialization
    # is still a violation — the fix only widens to the materializing
    # transition, it never waives the check.
    unknown_id = "ev_not_in_any_ledger"
    leaked = json.loads(json.dumps(conflict_transition))
    leaked["tool_results"][0]["consumes_evidence_ids"] = [unknown_id]
    leaked["submitted_action"]["actions"][0]["consumes_evidence_ids"] = [
        unknown_id
    ]
    leaked_closure = _build_evidence_closure(
        env,
        {"events": [], "transitions": [first_transition, leaked]},
    )
    assert leaked_closure["invisible_consumed_evidence_ids"] == [unknown_id]
    assert leaked_closure["closure_complete"] is False


def test_evidence_closure_rejects_stale_native_takeover_proof() -> None:
    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger("realtime-native-takeover")

    env = EvidenceEnvironment()
    safety_id = env.evidence.log(
        "runtime_assurance_observation",
        tick=0,
        payload={"mode": "mrm_active"},
        source="engine",
    )
    tool_id = env.evidence.log(
        "tool_call",
        tick=1,
        payload={"call_id": "mrm-1", "ok": True},
        source="tool",
    )
    realized_event = {
        "event_id": "mrm-effect-1",
        "origin": "agent_caused",
        "agent_caused": True,
        "call_id": "mrm-1",
        "tool_name": "request_minimal_risk_maneuver",
        "before_state_digest": "before",
        "after_state_digest": "after",
        "changed_state_fields": ["mrm_active"],
        "outcome_tick": 1,
    }
    effect_id = env.evidence.log(
        "realized_event",
        tick=1,
        payload=realized_event,
        source="engine",
    )
    transition = {
        "decision_id": "decision-1",
        "action_id": "action-1",
        "state_version_before": 1,
        "state_version_after": 2,
        "simulator_tick_before": 1,
        "simulator_tick": 2,
        "simulator_time_advanced": True,
        "action_source": "safety_supervisor",
        "safety_decision": {
            "mode": "native_runtime_takeover",
            "evidence_ids": [safety_id],
        },
        "safety_evidence_ids": [safety_id],
        "applied_action": {
            "actions": [
                {
                    "name": "request_minimal_risk_maneuver",
                    "call_id": "mrm-1",
                    "consumes_evidence_ids": [],
                    "depends_on_call_ids": [],
                }
            ]
        },
        "tool_results": [
            {
                "name": "request_minimal_risk_maneuver",
                "call_id": "mrm-1",
                "ok": True,
                "state_changing": True,
                "evidence_id": tool_id,
            }
        ],
        "realized_events": [{**realized_event, "evidence_ids": [effect_id]}],
        "tool_trace_edges": [
            {
                "call_id": "mrm-1",
                "effect_proven": True,
                "effect_evidence_ids": [effect_id],
            }
        ],
        "effect_observed": True,
        "effect_evidence_ids": [effect_id],
    }

    valid = _build_evidence_closure(
        env,
        {"events": [], "transitions": [transition]},
    )
    assert valid["closure_complete"] is True

    stale = json.loads(json.dumps(transition))
    stale.update(
        state_version_before=99,
        state_version_after=100,
        simulator_tick_before=99,
        simulator_tick=100,
    )
    invalid = _build_evidence_closure(
        env,
        {"events": [], "transitions": [stale]},
    )

    assert invalid["closure_complete"] is False
    assert invalid["invalid_safety_evidence_ids"] == [safety_id]


def test_evidence_closure_joins_realized_event_ledger_identity() -> None:
    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger("realtime-ledger-effect")

    env = EvidenceEnvironment()
    tool_evidence_id = env.evidence.log(
        "tool_call",
        tick=0,
        payload={"call_id": "call-control", "ok": True},
        source="tool",
    )
    realized_event = {
        "event_id": "control:0@0",
        "origin": "agent_caused",
        "call_id": "call-control",
        "tool_name": "control",
        "requested_action": {"setting": "safe"},
        "before_state_digest": "before",
        "after_state_digest": "after",
        "changed_state_fields": ["setting"],
        "evidence_ids": [tool_evidence_id],
        "outcome_tick": 0,
    }
    effect_evidence_id = env.evidence.log(
        "realized_event",
        tick=0,
        payload=realized_event,
        source="engine",
    )
    transition = {
        "action_id": "action-control",
        "submitted_action": {
            "actions": [
                {
                    "name": "control",
                    "call_id": "call-control",
                    "consumes_evidence_ids": [],
                    "depends_on_call_ids": [],
                }
            ]
        },
        "tool_results": [
            {
                "name": "control",
                "call_id": "call-control",
                "ok": True,
                "state_changing": True,
                "evidence_id": tool_evidence_id,
            }
        ],
        "realized_events": [realized_event],
        "tool_trace_edges": [
            {
                "call_id": "call-control",
                "effect_proven": True,
                "effect_evidence_ids": [effect_evidence_id],
            }
        ],
        "effect_observed": True,
    }

    closure = _build_evidence_closure(
        env,
        {"events": [], "transitions": [transition]},
    )

    assert closure["closure_complete"] is True
    assert closure["invalid_effect_action_ids"] == []
    assert closure["unproven_agent_mutation_event_ids"] == []


def test_evidence_closure_reuses_prior_effect_proof_for_deferred_action_receipt() -> (
    None
):
    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger("realtime-deferred-action-effect")

    env = EvidenceEnvironment()
    tool_evidence_id = env.evidence.log(
        "tool_call",
        tick=1,
        payload={"call_id": "call-immediate", "ok": True},
        source="tool",
    )
    realized_event = {
        "event_id": "control:1@1",
        "origin": "agent_caused",
        "call_id": "call-immediate",
        "tool_name": "control",
        "requested_action": {"setting": "safe"},
        "before_state_digest": "before",
        "after_state_digest": "after",
        "changed_state_fields": ["setting"],
        "evidence_ids": [tool_evidence_id],
        "outcome_tick": 1,
    }
    effect_evidence_id = env.evidence.log(
        "realized_event",
        tick=1,
        payload=realized_event,
        source="engine",
    )
    immediate = {
        "action_id": "action-mixed",
        "submitted_action": {
            "actions": [{"name": "control", "call_id": "call-immediate"}]
        },
        "tool_results": [
            {
                "name": "control",
                "call_id": "call-immediate",
                "ok": True,
                "state_changing": True,
                "evidence_id": tool_evidence_id,
            }
        ],
        "realized_events": [realized_event],
        "tool_trace_edges": [
            {
                "call_id": "call-immediate",
                "effect_proven": True,
                "effect_evidence_ids": [effect_evidence_id],
            }
        ],
        "effect_observed": True,
    }
    deferred_receipt = {
        "action_id": "action-mixed",
        "submitted_action": None,
        "tool_results": [],
        "realized_events": [],
        "tool_trace_edges": immediate["tool_trace_edges"],
        "deferred_action_outcomes": [
            {
                "action_id": "action-mixed",
                "effect_observed": True,
                "effect_evidence_ids": [effect_evidence_id],
                "tool_trace_edges": immediate["tool_trace_edges"],
            }
        ],
        "effect_observed": True,
    }

    closure = _build_evidence_closure(
        env,
        {"events": [], "transitions": [immediate, deferred_receipt]},
    )

    assert closure["closure_complete"] is True
    assert closure["invalid_effect_action_ids"] == []


def test_evidence_closure_allows_non_agent_post_log_identity_enrichment() -> None:
    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger("realtime-source-event")

    env = EvidenceEnvironment()
    source_event = {
        "origin": "source_schedule",
        "event_class": "task_arrival",
        "tick": 1,
        "task_id": "task-1",
    }
    evidence_id = env.evidence.log(
        "realized_event",
        tick=1,
        payload=source_event,
        source="engine",
    )
    enriched_event = {
        **source_event,
        "event_id": f"source:{evidence_id}",
        "evidence_ids": [evidence_id],
    }

    closure = _build_evidence_closure(
        env,
        {
            "events": [],
            "transitions": [{"realized_events": [enriched_event]}],
        },
    )

    assert closure["closure_complete"] is True
    assert closure["invalid_realized_event_evidence_ids"] == []


def test_evidence_closure_rejects_mismatched_realized_event_ledger_identity() -> None:
    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger("realtime-mismatched-effect")

    env = EvidenceEnvironment()
    tool_evidence_id = env.evidence.log(
        "tool_call",
        tick=0,
        payload={"call_id": "call-control", "ok": True},
        source="tool",
    )
    ledger_event = {
        "event_id": "shared-event",
        "origin": "endogenous_completion",
        "event_class": "lifecycle",
        "call_id": "call-control",
        "tool_name": "control",
        "requested_action": {"setting": "safe"},
        "before_state_digest": "before",
        "after_state_digest": "after",
        "evidence_ids": [tool_evidence_id],
        "outcome_tick": 0,
    }
    effect_evidence_id = env.evidence.log(
        "realized_event",
        tick=0,
        payload=ledger_event,
        source="engine",
    )
    transition_event = {
        **ledger_event,
        "origin": "agent_caused",
        "event_class": "control_effect",
        "changed_state_fields": ["setting"],
        "evidence_ids": [tool_evidence_id, effect_evidence_id],
    }
    transition = {
        "action_id": "action-control",
        "submitted_action": {
            "actions": [{"name": "control", "call_id": "call-control"}]
        },
        "tool_results": [
            {
                "name": "control",
                "call_id": "call-control",
                "ok": True,
                "state_changing": True,
                "evidence_id": tool_evidence_id,
            }
        ],
        "realized_events": [transition_event],
        "tool_trace_edges": [
            {
                "call_id": "call-control",
                "effect_proven": True,
                "effect_evidence_ids": [effect_evidence_id],
            }
        ],
        "effect_observed": True,
    }

    closure = _build_evidence_closure(
        env,
        {"events": [], "transitions": [transition]},
    )

    assert closure["closure_complete"] is False
    assert closure["invalid_realized_event_evidence_ids"] == [effect_evidence_id]
    assert closure["invalid_effect_action_ids"] == ["action-control"]
    assert closure["unproven_agent_mutation_event_ids"] == ["shared-event"]


def test_evidence_closure_rejects_tool_receipt_as_effect_proof() -> None:
    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger("realtime-tool-receipt-effect")

    env = EvidenceEnvironment()
    tool_evidence_id = env.evidence.log(
        "tool_call",
        tick=0,
        payload={"call_id": "call-control", "ok": True},
        source="tool",
    )
    transition = {
        "action_id": "action-control",
        "submitted_action": {
            "actions": [{"name": "control", "call_id": "call-control"}]
        },
        "tool_results": [
            {
                "name": "control",
                "call_id": "call-control",
                "ok": True,
                "state_changing": True,
                "evidence_id": tool_evidence_id,
            }
        ],
        "realized_events": [
            {
                "event_id": "control:0@0",
                "origin": "agent_caused",
                "event_class": "control_effect",
                "call_id": "call-control",
                "changed_state_fields": ["setting"],
                "evidence_ids": [tool_evidence_id],
                "outcome_tick": 0,
            }
        ],
        "tool_trace_edges": [
            {
                "call_id": "call-control",
                "effect_proven": True,
                "effect_evidence_ids": [tool_evidence_id],
            }
        ],
        "effect_observed": True,
    }

    closure = _build_evidence_closure(
        env,
        {"events": [], "transitions": [transition]},
    )

    assert closure["closure_complete"] is False
    assert closure["invalid_effect_action_ids"] == ["action-control"]
    assert closure["unproven_agent_mutation_event_ids"] == ["control:0@0"]


@pytest.mark.parametrize("variant", ["tick_mismatch", "duplicate_identity"])
def test_evidence_closure_rejects_ambiguous_realized_event_ledger_identity(
    variant: str,
) -> None:
    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger(f"realtime-{variant}")

    env = EvidenceEnvironment()
    tool_evidence_id = env.evidence.log(
        "tool_call",
        tick=0,
        payload={"call_id": "call-control", "ok": True},
        source="tool",
    )
    realized_event = {
        "event_id": "control:0@0",
        "origin": "agent_caused",
        "event_class": "control_effect",
        "call_id": "call-control",
        "tool_name": "control",
        "requested_action": {"setting": "safe"},
        "before_state_digest": "before",
        "after_state_digest": "after",
        "changed_state_fields": ["setting"],
        "evidence_ids": [tool_evidence_id],
        "outcome_tick": 0,
    }
    effect_evidence_id = env.evidence.log(
        "realized_event",
        tick=1 if variant == "tick_mismatch" else 0,
        payload=realized_event,
        source="engine",
    )
    if variant == "duplicate_identity":
        env.evidence.log(
            "realized_event",
            tick=0,
            payload=realized_event,
            source="engine",
        )
    transition = {
        "action_id": "action-control",
        "submitted_action": {
            "actions": [{"name": "control", "call_id": "call-control"}]
        },
        "tool_results": [
            {
                "name": "control",
                "call_id": "call-control",
                "ok": True,
                "state_changing": True,
                "evidence_id": tool_evidence_id,
            }
        ],
        "realized_events": [realized_event],
        "tool_trace_edges": [
            {
                "call_id": "call-control",
                "effect_proven": True,
                "effect_evidence_ids": [effect_evidence_id],
            }
        ],
        "effect_observed": True,
    }

    closure = _build_evidence_closure(
        env,
        {"events": [], "transitions": [transition]},
    )

    assert closure["closure_complete"] is False
    assert closure["invalid_effect_action_ids"] == ["action-control"]
    assert closure["unproven_agent_mutation_event_ids"] == ["control:0@0"]


def test_evidence_closure_uses_submission_visibility_for_delayed_result() -> None:
    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger("realtime-delayed-result")

    env = EvidenceEnvironment()
    consumed_id = env.evidence.log("observation", tick=1)
    result_id = env.evidence.log("tool_call", tick=3)
    call = {
        "name": "investigate_asset",
        "call_id": "call-delayed",
        "consumes_evidence_ids": [consumed_id],
        "depends_on_call_ids": [],
    }
    closure = _build_evidence_closure(
        env,
        {
            "events": [],
            "transitions": [
                {
                    "simulator_tick": 1,
                    "based_on_visible_evidence_ids": [consumed_id],
                    "submitted_action": {"actions": [call]},
                    "tool_results": [],
                },
                {
                    "simulator_tick": 3,
                    "based_on_visible_evidence_ids": [],
                    "submitted_action": {"actions": []},
                    "tool_results": [
                        {
                            **call,
                            "ok": True,
                            "evidence_id": result_id,
                        }
                    ],
                },
            ],
        },
    )

    assert closure["closure_complete"] is True
    assert closure["invisible_consumed_evidence_ids"] == []

    orphan = _build_evidence_closure(
        env,
        {
            "events": [],
            "transitions": [
                {
                    "based_on_visible_evidence_ids": [consumed_id],
                    "submitted_action": {"actions": []},
                    "tool_results": [
                        {
                            **call,
                            "ok": True,
                            "evidence_id": result_id,
                        }
                    ],
                }
            ],
        },
    )
    assert orphan["closure_complete"] is False
    assert orphan["invisible_consumed_evidence_ids"] == [consumed_id]


def test_realtime_artifact_validation_fails_closed_after_complete_execution() -> None:
    artifact = {
        "episode_status": "complete",
        "provider_audit_contract": {"complete": True},
        "tool_surface_contract": {"complete": True},
        "evidence_closure": {"closure_complete": False},
        "teardown": {
            "actor_stopped": True,
            "unsafe_teardown": False,
            "environment_close_allowed": True,
        },
    }

    realtime_episode._apply_realtime_artifact_validation(
        artifact, behavioral_state_settled=True
    )

    assert artifact["episode_status"] == "invalid_evidence_closure"
    assert artifact["evaluation_ready"] is False
    assert artifact["artifact_validation"]["blocker_codes"] == [
        "EVIDENCE_CLOSURE_INCOMPLETE"
    ]


def test_realtime_artifact_validation_blocks_event_contract_violation() -> None:
    artifact = {
        "episode_status": "complete",
        "provider_audit_contract": {"complete": True},
        "tool_surface_contract": {"complete": True},
        "evidence_closure": {"closure_complete": True},
        "event_contract": {"violation_count": 1},
        "teardown": {
            "actor_stopped": True,
            "unsafe_teardown": False,
            "environment_close_allowed": True,
        },
    }

    realtime_episode._apply_realtime_artifact_validation(
        artifact, behavioral_state_settled=True
    )

    assert artifact["episode_status"] == "invalid_artifact"
    assert artifact["evaluation_ready"] is False
    assert artifact["artifact_validation"]["blocker_codes"] == [
        "EVENT_CONTRACT_VIOLATION"
    ]


def test_evidence_closure_rejects_orphan_result_without_consumes() -> None:
    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger("realtime-orphan-result")

    env = EvidenceEnvironment()
    result_id = env.evidence.log("tool_call", tick=1)
    closure = _build_evidence_closure(
        env,
        {
            "events": [],
            "transitions": [
                {
                    "submitted_action": {"actions": []},
                    "applied_action": {"actions": []},
                    "tool_results": [
                        {
                            "name": "orphan_control",
                            "call_id": "orphan-call",
                            "ok": True,
                            "evidence_id": result_id,
                            "consumes_evidence_ids": [],
                        }
                    ],
                }
            ],
        },
    )

    assert closure["closure_complete"] is False
    assert closure["orphan_result_call_ids"] == ["orphan-call"]


@pytest.mark.parametrize(
    ("calls", "expected_edge"),
    [
        (
            [
                {
                    "name": "first",
                    "call_id": "call-first",
                    "depends_on_call_ids": ["call-second"],
                },
                {"name": "second", "call_id": "call-second"},
            ],
            {
                "call_id": "call-first",
                "depends_on_call_id": "call-second",
            },
        ),
        (
            [
                {
                    "name": "self",
                    "call_id": "call-self",
                    "depends_on_call_ids": ["call-self"],
                }
            ],
            {
                "call_id": "call-self",
                "depends_on_call_id": "call-self",
            },
        ),
    ],
)
def test_evidence_closure_rejects_self_and_forward_dependencies(
    calls: list[dict], expected_edge: dict
) -> None:
    class EvidenceEnvironment:
        evidence = EvidenceLogger("realtime-noncausal-dependency")

    closure = _build_evidence_closure(
        EvidenceEnvironment(),
        {
            "events": [],
            "transitions": [
                {
                    "submitted_action": {"actions": calls},
                    "applied_action": {"actions": calls},
                    "tool_results": [],
                }
            ],
        },
    )

    assert closure["closure_complete"] is False
    assert expected_edge in closure["noncausal_dependency_edges"]


def test_evidence_closure_allows_earlier_dependency_on_delayed_result_copy() -> None:
    class EvidenceEnvironment:
        evidence = EvidenceLogger("realtime-causal-dependency")

    first = {"name": "inspect", "call_id": "call-first"}
    second = {
        "name": "control",
        "call_id": "call-second",
        "depends_on_call_ids": ["call-first"],
    }
    closure = _build_evidence_closure(
        EvidenceEnvironment(),
        {
            "events": [],
            "transitions": [
                {
                    "submitted_action": {"actions": [first, second]},
                    "applied_action": {"actions": [first, second]},
                    "tool_results": [],
                },
                {
                    "submitted_action": {"actions": []},
                    "applied_action": {"actions": []},
                    "tool_results": [{**second, "ok": True}],
                },
            ],
        },
    )

    assert closure["closure_complete"] is True
    assert closure["noncausal_dependency_edges"] == []


def test_evidence_closure_uses_applied_action_visibility_for_safety_result() -> None:
    class EvidenceEnvironment:
        def __init__(self) -> None:
            self.evidence = EvidenceLogger("realtime-safety-result")

    env = EvidenceEnvironment()
    consumed_id = env.evidence.log("safety_observation", tick=1)
    result_id = env.evidence.log("safety_tool_call", tick=1)
    call = {
        "name": "controlled_hold",
        "call_id": "call-safety",
        "consumes_evidence_ids": [consumed_id],
        "depends_on_call_ids": [],
    }

    closure = _build_evidence_closure(
        env,
        {
            "events": [],
            "transitions": [
                {
                    "based_on_visible_evidence_ids": [consumed_id],
                    "submitted_action": None,
                    "applied_action": {"actions": [call]},
                    "tool_results": [
                        {**call, "ok": True, "evidence_id": result_id}
                    ],
                }
            ],
        },
    )

    assert closure["closure_complete"] is True
    assert closure["invisible_consumed_evidence_ids"] == []


def test_realtime_treatment_hash_binds_clock_safety_and_public_provider_config() -> None:
    kwargs = {
        "config": {
            "provider": "openai_compatible",
            "model": "hy3-ioa",
            "base_url": "https://user:password@example.test/v1?api_key=secret",
            "extra_headers": {
                "Authorization": "Bearer private-token",
                "X-Route": "canary",
            },
            "temperature": 0.2,
        }
    }
    identity, digest = build_realtime_treatment_identity(
        agent_name="llm_agent",
        agent_kwargs=kwargs,
        tick_interval_s=0.5,
        episode_timeout_s=60.0,
        safety_supervisor=_SafetySupervisor(),
    )
    changed_identity, changed_digest = build_realtime_treatment_identity(
        agent_name="llm_agent",
        agent_kwargs=kwargs,
        tick_interval_s=1.0,
        episode_timeout_s=60.0,
        safety_supervisor=_SafetySupervisor(),
    )

    serialized = json.dumps(identity, sort_keys=True)
    assert len(digest) == 64
    assert digest != changed_digest
    assert identity != changed_identity
    assert "password" not in serialized
    assert "private-token" not in serialized
    assert identity["provider_public_config"]["base_url"] == "https://example.test/v1"
    assert identity["interrupt_contract"] == {
        "behavioral_state_transactional": True,
        "direct_api_turn_concurrency": 1,
        "established_provider_stream_cancel_supported": False,
        "fallback_interrupt": "logical_supersession_with_execution_fence",
        "late_response_execution_allowed": False,
        "late_response_execution_fence": True,
    }
    assert identity["implementation_contract"]["realtime_coordinator"] == (
        "realtime_episode_v6"
    )
    assert identity["wakeup_policy"] == {
        "session_start": True,
        "typed_actionable_events": True,
        "agent_scheduled_reviews": True,
        "harness_periodic_supervisory_scan": False,
        "unknown_events_actionable": False,
    }

    rotated_secret = json.loads(json.dumps(kwargs))
    rotated_secret["config"]["extra_headers"]["Authorization"] = "Bearer rotated"
    _, rotated_secret_digest = build_realtime_treatment_identity(
        agent_name="llm_agent",
        agent_kwargs=rotated_secret,
        tick_interval_s=0.5,
        episode_timeout_s=60.0,
        safety_supervisor=_SafetySupervisor(),
    )
    changed_route = json.loads(json.dumps(kwargs))
    changed_route["config"]["extra_headers"]["X-Route"] = "stable"
    _, changed_route_digest = build_realtime_treatment_identity(
        agent_name="llm_agent",
        agent_kwargs=changed_route,
        tick_interval_s=0.5,
        episode_timeout_s=60.0,
        safety_supervisor=_SafetySupervisor(),
    )
    assert rotated_secret_digest == digest
    assert changed_route_digest != digest


def test_realtime_treatment_hash_binds_effective_api_version(monkeypatch) -> None:
    kwargs = {
        "config": {
            "provider": "azure",
            "model": "deployment",
            "api_version": None,
            "api_version_env": "TEST_EFFECTIVE_API_VERSION",
        }
    }
    monkeypatch.setenv("TEST_EFFECTIVE_API_VERSION", "2026-01-01")
    first, first_digest = build_realtime_treatment_identity(
        agent_name="llm_agent",
        agent_kwargs=kwargs,
        tick_interval_s=1.0,
        episode_timeout_s=60.0,
        safety_supervisor=_SafetySupervisor(),
    )
    monkeypatch.setenv("TEST_EFFECTIVE_API_VERSION", "2026-02-01")
    second, second_digest = build_realtime_treatment_identity(
        agent_name="llm_agent",
        agent_kwargs=kwargs,
        tick_interval_s=1.0,
        episode_timeout_s=60.0,
        safety_supervisor=_SafetySupervisor(),
    )

    assert first["provider_public_config"]["effective_api_version"] == "2026-01-01"
    assert second["provider_public_config"]["effective_api_version"] == "2026-02-01"
    assert first_digest != second_digest


def test_realtime_artifact_write_is_exclusive_durable_and_never_overwrites(
    tmp_path: Path,
) -> None:
    target = tmp_path / "episode.json"
    first = {"treatment_sha256": "a" * 64, "value": 1}
    _write_realtime_artifact_exclusive(target=target, artifact=first)
    original_bytes = target.read_bytes()

    with pytest.raises(FileExistsError):
        _write_realtime_artifact_exclusive(
            target=target,
            artifact={"treatment_sha256": "b" * 64, "value": 2},
        )

    assert target.read_bytes() == original_bytes
    assert json.loads(original_bytes) == first
    assert not list(tmp_path.glob("*.tmp"))


def test_realtime_artifact_write_uses_portable_short_temporary_basename(
    tmp_path: Path,
) -> None:
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    target = tmp_path / ("e" * (name_max - len(".json") - 8) + ".json")
    assert len(target.name.encode("utf-8")) < name_max

    _write_realtime_artifact_exclusive(
        target=target,
        artifact={"treatment_sha256": "a" * 64, "value": 1},
    )

    assert json.loads(target.read_text(encoding="utf-8"))["value"] == 1
    assert not list(tmp_path.glob(".*.tmp"))


def test_realtime_artifact_target_compacts_long_scenario_id_deterministically(
    tmp_path: Path,
) -> None:
    treatment_sha256 = "a" * 64
    common_prefix = "inventory/" + "source-segment-" * 30
    first = realtime_episode._realtime_artifact_target(
        trajectory_dir=tmp_path,
        agent_name="llm_agent",
        scenario_id=f"{common_prefix}first",
        seed=42,
        treatment_sha256=treatment_sha256,
    )
    first_repeat = realtime_episode._realtime_artifact_target(
        trajectory_dir=tmp_path,
        agent_name="llm_agent",
        scenario_id=f"{common_prefix}first",
        seed=42,
        treatment_sha256=treatment_sha256,
    )
    second = realtime_episode._realtime_artifact_target(
        trajectory_dir=tmp_path,
        agent_name="llm_agent",
        scenario_id=f"{common_prefix}second",
        seed=42,
        treatment_sha256=treatment_sha256,
    )

    assert first == first_repeat
    assert first != second
    assert len(first.name.encode("utf-8")) <= 200
    assert treatment_sha256 in first.name

    artifact = {
        "scenario_id": f"{common_prefix}first",
        "treatment_sha256": treatment_sha256,
    }
    _write_realtime_artifact_exclusive(target=first, artifact=artifact)
    assert json.loads(first.read_text(encoding="utf-8")) == artifact


def test_realtime_artifact_publish_failure_never_exposes_empty_final(
    tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "episode.json"
    artifact = {"treatment_sha256": "a" * 64, "value": 1}
    real_link = realtime_episode.os.link

    def fail_link(_source, _target):
        raise OSError("simulated publish failure")

    monkeypatch.setattr(realtime_episode.os, "link", fail_link)
    with pytest.raises(OSError, match="simulated publish failure"):
        _write_realtime_artifact_exclusive(target=target, artifact=artifact)

    assert target.exists() is False
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))

    monkeypatch.setattr(realtime_episode.os, "link", real_link)
    _write_realtime_artifact_exclusive(target=target, artifact=artifact)
    assert json.loads(target.read_text()) == artifact


def test_actor_loop_exception_marks_episode_failed_in_artifact() -> None:
    class MalformedResult:
        @property
        def observation(self):
            raise TypeError("private malformed observation")

    class MalformedEnvironment(_AlarmEnvironment):
        def step(self, action: Action):
            del action
            return MalformedResult()

    driver = _DelayedStubDriver()
    artifact = RealtimeEpisodeCoordinator(
        env=MalformedEnvironment(alarm_tick=None, horizon=1),
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.01,
    ).run(timeout_s=1.0)

    assert artifact["episode_status"] == "failed"
    assert artifact["clock"]["actor_failed"] is True
    assert artifact["actor_failure"] == {
        "error_type": "TypeError",
        "stage": "environment_actor_loop",
    }
    assert artifact["episode_outcome"]["actor_failure"] == artifact["actor_failure"]
    assert "private malformed observation" not in json.dumps(artifact)


def test_nonadvancing_environment_tick_marks_episode_failed_in_artifact() -> None:
    class NonAdvancingEnvironment(_AlarmEnvironment):
        def step(self, action: Action) -> StepReturn:
            del action
            return StepReturn(
                observation=self.snapshot(),
                tool_results=[],
                reward=0.0,
                done=False,
                info=StepInfo(),
            )

    artifact = RealtimeEpisodeCoordinator(
        env=NonAdvancingEnvironment(alarm_tick=None, horizon=1),
        turn_driver=_DelayedStubDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.01,
    ).run(timeout_s=1.0)

    assert artifact["episode_status"] == "failed"
    assert artifact["actor_failure"] == {
        "error_type": "ValueError",
        "stage": "environment_actor_loop",
    }
    assert artifact["transitions"][-1]["simulator_time_advanced"] is False


def test_blocking_environment_step_returns_fail_closed_teardown_artifact() -> None:
    class BlockingEnvironment(_AlarmEnvironment):
        def __init__(self) -> None:
            super().__init__(alarm_tick=None, horizon=1)
            self.step_entered = threading.Event()
            self.release_step = threading.Event()

        def step(self, action: Action) -> StepReturn:
            self.step_entered.set()
            assert self.release_step.wait(timeout=5.0)
            return super().step(action)

    class NoActionDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(Action())
            return future

    env = BlockingEnvironment()
    coordinator = RealtimeEpisodeCoordinator(
        env=env,
        turn_driver=NoActionDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.005,
    )

    artifact = coordinator.run(timeout_s=0.03)

    assert env.step_entered.is_set()
    assert artifact["episode_status"] == "timed_out"
    assert artifact["clock"]["timed_out"] is True
    assert artifact["teardown"]["actor_stopped"] is False
    assert artifact["teardown"]["unsafe_teardown"] is True
    assert artifact["teardown"]["environment_close_allowed"] is False
    assert artifact["teardown"]["exception"].startswith("TimeoutError:")

    env.release_step.set()
    coordinator._actor.stop(timeout_s=1.0)
    assert coordinator.environment_actor_stopped is True


def test_alarm_admission_does_not_block_timeout_during_next_environment_step() -> None:
    class AlarmThenBlockingEnvironment(_AlarmEnvironment):
        def __init__(self) -> None:
            super().__init__(alarm_tick=1, horizon=3)
            self.second_step_entered = threading.Event()
            self.release_step = threading.Event()

        def step(self, action: Action) -> StepReturn:
            if self.tick == 1:
                self.second_step_entered.set()
                assert self.release_step.wait(timeout=5.0)
            return super().step(action)

    class NoActionDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(Action())
            return future

    class DelayedTransitionCoordinator(RealtimeEpisodeCoordinator):
        def _process_transition(self, transition: dict) -> None:
            if transition.get("simulator_tick") == 1:
                assert env.second_step_entered.wait(timeout=1.0)
            super()._process_transition(transition)

    env = AlarmThenBlockingEnvironment()
    driver = NoActionDriver()
    coordinator = DelayedTransitionCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.005,
    )

    started = time.monotonic()
    artifact = coordinator.run(timeout_s=0.05)

    assert time.monotonic() - started < 3.0
    assert artifact["episode_status"] == "timed_out"
    assert artifact["teardown"]["unsafe_teardown"] is True
    assert [event.kind for event in driver.started] == ["session_start"]
    alarm = next(
        event
        for event in artifact["events"]
        if event["kind"] == "environment_alarm"
    )
    assert alarm["queued_reason"] == "CLOCK_DISPATCH_IN_FLIGHT"
    assert alarm["pending_at_shutdown"] is True
    assert alarm["dispatch_suppressed_reason"] == "EPISODE_WALL_TIMEOUT"

    env.release_step.set()
    coordinator._actor.stop(timeout_s=1.0)
    assert coordinator.environment_actor_stopped is True


def test_clock_deferred_alarm_is_retried_after_next_transition_settles() -> None:
    class AlarmThenPausedEnvironment(_AlarmEnvironment):
        def __init__(self) -> None:
            super().__init__(alarm_tick=1, horizon=3)
            self.second_step_entered = threading.Event()
            self.release_step = threading.Event()

        def step(self, action: Action) -> StepReturn:
            if self.tick == 1:
                self.second_step_entered.set()
                assert self.release_step.wait(timeout=1.0)
            return super().step(action)

    class NoActionDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(Action())
            return future

    class ReleaseAfterAlarmQueuedCoordinator(RealtimeEpisodeCoordinator):
        def _process_transition(self, transition: dict) -> None:
            if transition.get("simulator_tick") == 1:
                assert env.second_step_entered.wait(timeout=1.0)
                super()._process_transition(transition)
                env.release_step.set()
                return
            super()._process_transition(transition)

    env = AlarmThenPausedEnvironment()
    driver = NoActionDriver()
    artifact = ReleaseAfterAlarmQueuedCoordinator(
        env=env,
        turn_driver=driver,
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.005,
    ).run(timeout_s=1.0)

    assert artifact["episode_status"] == "complete"
    assert [event.kind for event in driver.started].count("environment_alarm") == 1
    alarm = next(
        event
        for event in artifact["events"]
        if event["kind"] == "environment_alarm"
    )
    assert alarm["queued_reason"] == "CLOCK_DISPATCH_IN_FLIGHT"
    assert alarm["dispatched_from_pending"] is True
    assert alarm.get("pending_at_shutdown") is not True


def test_shutdown_settles_candidates_from_transition_completed_during_stop() -> None:
    class SlowFeedbackEnvironment(_AlarmEnvironment):
        def __init__(self) -> None:
            super().__init__(alarm_tick=None, horizon=3)

        def step(self, action: Action) -> StepReturn:
            time.sleep(0.08)
            result = super().step(action)
            result.info.early_stop_warnings = ["unsafe_margin"]
            result.info.forecast_updates = {"load": "rising"}
            return result

    class NoActionDriver(_DelayedStubDriver):
        def start_turn(
            self, *, turn_id: str, observation: dict, event: RealtimeEvent
        ) -> Future[Action]:
            del turn_id, observation
            self.started.append(event)
            future: Future[Action] = Future()
            future.set_result(Action())
            return future

    artifact = RealtimeEpisodeCoordinator(
        env=SlowFeedbackEnvironment(),
        turn_driver=NoActionDriver(),
        safety_supervisor=_SafetySupervisor(),
        tick_interval_s=0.005,
    ).run(timeout_s=0.03)

    assert artifact["episode_status"] == "timed_out"
    feedback = [
        event
        for event in artifact["events"]
        if event["kind"] in {"safety_warning", "forecast_update"}
    ]
    assert [event["kind"] for event in feedback] == [
        "safety_warning",
        "forecast_update",
    ]
    assert all(
        event["dispatch_suppressed_reason"] == "EPISODE_WALL_TIMEOUT"
        for event in feedback
    )
    assert feedback[1]["pending_at_shutdown"] is True


def test_run_realtime_does_not_close_environment_while_actor_is_unstopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        interaction_mode = "logical_persistent"

    class Agent:
        def reset(self, env: object, scenario: dict, *, seed: int) -> None:
            del env, scenario, seed

        def snapshot_behavioral_state(self) -> dict:
            return {}

        def restore_behavioral_state(self, snapshot: dict) -> None:
            del snapshot

    class Environment:
        def __init__(self) -> None:
            self.closed = False

        def reset(self, scenario: dict, *, seed: int) -> None:
            del scenario, seed

        def get_tool_specs(self) -> list[dict]:
            return []

        def close(self) -> None:
            self.closed = True

    class UnsafeCoordinator:
        environment_actor_stopped = False

        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def run(self, *, timeout_s: float) -> dict:
            del timeout_s
            return {
                "clock": {"outstanding_provider_turns_at_return": 0},
                "teardown": {
                    "actor_stopped": False,
                    "unsafe_teardown": True,
                    "environment_close_allowed": False,
                },
            }

    env = Environment()
    monkeypatch.setattr(
        realtime_episode,
        "get_domain_spec",
        lambda domain: type(
            "Spec",
            (),
            {"env_factory": staticmethod(lambda: lambda: env)},
        )(),
    )
    monkeypatch.setattr(realtime_episode, "make_agent", lambda *args, **kwargs: Agent())
    monkeypatch.setattr(
        realtime_episode,
        "build_realtime_treatment_identity",
        lambda **kwargs: ({"test": True}, "a" * 64),
    )
    monkeypatch.setattr(
        realtime_episode,
        "RealtimeEpisodeCoordinator",
        UnsafeCoordinator,
    )
    monkeypatch.setattr(
        realtime_episode,
        "_build_evidence_closure",
        lambda env, artifact: {"closure_complete": False},
    )
    monkeypatch.setattr(
        realtime_episode,
        "recompute_signature_with_seed",
        lambda scenario, seed, spec: "signature",
    )

    artifact = run_realtime(
        {"domain": "test", "seed_id": "test/unsafe", "seed": 1},
        "llm_agent",
        agent_kwargs={"config": Config()},
        tick_interval_s=0.01,
        timeout_s=0.02,
        safety_supervisor=_SafetySupervisor(),
    )

    assert artifact["teardown"]["unsafe_teardown"] is True
    assert "UNSAFE_OR_INCOMPLETE_TEARDOWN" in artifact["artifact_validation"][
        "blocker_codes"
    ]
    assert artifact["evaluation_ready"] is False
    assert env.closed is False


def test_run_realtime_binds_suite_scenario_id_not_seed_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        interaction_mode = "logical_persistent"

    class Agent:
        def reset(self, env: object, scenario: dict, *, seed: int) -> None:
            del env, scenario, seed

        def snapshot_behavioral_state(self) -> dict:
            return {}

        def restore_behavioral_state(self, snapshot: dict) -> None:
            del snapshot

        def get_session_ledger(self) -> list[dict]:
            return [{"role": "system", "content": "mission"}]

        def get_structured_memory(self) -> dict:
            return {"schema_version": "persistent_working_memory_v2"}

        def get_interaction_stats(self) -> dict:
            return {}

    class Environment:
        def reset(self, scenario: dict, *, seed: int) -> None:
            del scenario, seed

        def get_tool_specs(self) -> list[dict]:
            return []

        def close(self) -> None:
            return None

    class SettledCoordinator:
        environment_actor_stopped = True

        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def run(self, *, timeout_s: float) -> dict:
            del timeout_s
            return {
                "episode_status": "complete",
                "clock": {"outstanding_provider_turns_at_return": 0},
                "teardown": {
                    "actor_stopped": True,
                    "unsafe_teardown": False,
                    "environment_close_allowed": True,
                    "behavioral_settlement_complete": True,
                },
                "environment_observation_ingestion": {
                    "queued": 0,
                    "settled": 0,
                    "completed": 0,
                    "failed": 0,
                    "canceled": 0,
                    "pending": 0,
                },
                "provider_audit_contract": {"complete": True},
                "event_contract": {"violation_count": 0},
                "events": [],
            }

    monkeypatch.setattr(
        realtime_episode,
        "get_domain_spec",
        lambda domain: type(
            "Spec",
            (),
            {"env_factory": staticmethod(lambda: lambda: Environment())},
        )(),
    )
    monkeypatch.setattr(realtime_episode, "make_agent", lambda *args, **kwargs: Agent())
    monkeypatch.setattr(
        realtime_episode,
        "build_realtime_treatment_identity",
        lambda **kwargs: ({"test": True}, "b" * 64),
    )
    monkeypatch.setattr(
        realtime_episode,
        "RealtimeEpisodeCoordinator",
        SettledCoordinator,
    )
    monkeypatch.setattr(
        realtime_episode,
        "_build_evidence_closure",
        lambda env, artifact: {"closure_complete": True},
    )
    monkeypatch.setattr(
        realtime_episode,
        "recompute_signature_with_seed",
        lambda scenario, seed, spec: "signature",
    )

    artifact = run_realtime(
        {
            "domain": "test",
            "seed_id": "resco_ingolstadt7_phase_control_high_s9427",
            "scenario_id": (
                "traffic/signal_coordination/deep_planning/high/"
                "resco_ingolstadt7_phase_control_high_s9427"
            ),
            "seed": 1,
        },
        "llm_agent",
        agent_kwargs={"config": Config()},
        tick_interval_s=0.01,
        timeout_s=0.02,
        safety_supervisor=_SafetySupervisor(),
    )

    assert artifact["scenario_id"] == (
        "traffic/signal_coordination/deep_planning/high/"
        "resco_ingolstadt7_phase_control_high_s9427"
    )
    assert artifact["semantic_ledger"]["schema_version"] == (
        realtime_episode.SEMANTIC_SESSION_LEDGER_SCHEMA_VERSION
    )
    assert artifact["semantic_ledger"]["records"] == [
        {"role": "system", "content": "mission"}
    ]


def test_run_realtime_does_not_read_memory_with_pending_behavioral_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        interaction_mode = "logical_persistent"

    class Agent:
        def __init__(self) -> None:
            self.ledger_reads = 0
            self.memory_reads = 0
            self.stats_reads = 0

        def reset(self, env: object, scenario: dict, *, seed: int) -> None:
            del env, scenario, seed

        def snapshot_behavioral_state(self) -> dict:
            return {}

        def restore_behavioral_state(self, snapshot: dict) -> None:
            del snapshot

        def get_session_ledger(self) -> list[dict]:
            self.ledger_reads += 1
            return [{"must_not": "be read concurrently"}]

        def get_structured_memory(self) -> dict:
            self.memory_reads += 1
            return {"must_not": "be read concurrently"}

        def get_interaction_stats(self) -> dict:
            self.stats_reads += 1
            return {}

    class Environment:
        def reset(self, scenario: dict, *, seed: int) -> None:
            del scenario, seed

        def get_tool_specs(self) -> list[dict]:
            return []

        def close(self) -> None:
            return None

    class UnsettledCoordinator:
        environment_actor_stopped = True

        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def run(self, *, timeout_s: float) -> dict:
            del timeout_s
            return {
                "episode_status": "complete",
                "clock": {"outstanding_provider_turns_at_return": 0},
                "teardown": {
                    "actor_stopped": True,
                    "unsafe_teardown": False,
                    "environment_close_allowed": True,
                    "behavioral_settlement_complete": False,
                },
                "environment_observation_ingestion": {
                    "queued": 1,
                    "settled": 0,
                    "completed": 0,
                    "failed": 0,
                    "canceled": 0,
                    "pending": 1,
                },
                "provider_audit_contract": {"complete": False},
                "event_contract": {"violation_count": 0},
            }

    env = Environment()
    agent = Agent()
    monkeypatch.setattr(
        realtime_episode,
        "get_domain_spec",
        lambda domain: type(
            "Spec",
            (),
            {"env_factory": staticmethod(lambda: lambda: env)},
        )(),
    )
    monkeypatch.setattr(realtime_episode, "make_agent", lambda *args, **kwargs: agent)
    monkeypatch.setattr(
        realtime_episode,
        "build_realtime_treatment_identity",
        lambda **kwargs: ({"test": True}, "b" * 64),
    )
    monkeypatch.setattr(
        realtime_episode,
        "RealtimeEpisodeCoordinator",
        UnsettledCoordinator,
    )
    monkeypatch.setattr(
        realtime_episode,
        "_build_evidence_closure",
        lambda env, artifact: {"closure_complete": True},
    )
    monkeypatch.setattr(
        realtime_episode,
        "recompute_signature_with_seed",
        lambda scenario, seed, spec: "signature",
    )

    artifact = run_realtime(
        {"domain": "test", "seed_id": "test/unsettled", "seed": 1},
        "llm_agent",
        agent_kwargs={"config": Config()},
        tick_interval_s=0.01,
        timeout_s=0.02,
        safety_supervisor=_SafetySupervisor(),
    )

    assert artifact["behavioral_state_artifact_status"] == (
        "unavailable_pending_observation_ingest"
    )
    assert artifact["semantic_ledger"] is None
    assert artifact["structured_memory"] is None
    assert agent.ledger_reads == 0
    assert agent.memory_reads == 0
    assert agent.stats_reads == 0
    assert artifact["llm_interaction_stats"] is None
    assert "BEHAVIORAL_STATE_UNSETTLED" in artifact["artifact_validation"][
        "blocker_codes"
    ]
    assert artifact["evaluation_ready"] is False


def test_environment_done_waits_for_ingest_after_provider_has_settled(monkeypatch):
    class IngestAgent(_TransactionalAgent):
        def act(self, observation, tool_specs):
            return Action()

        def ingest_realtime_observation(self, observation):
            time.sleep(0.12)

    monkeypatch.setattr(realtime_episode, '_PROVIDER_CANCEL_SETTLEMENT_GRACE_S', 0.02)
    driver = AgentTurnDriver(IngestAgent(), [])
    try:
        artifact = RealtimeEpisodeCoordinator(
            env=_AlarmEnvironment(alarm_tick=None, horizon=2), turn_driver=driver,
            safety_supervisor=_SafetySupervisor(), tick_interval_s=0.01,
        ).run(timeout_s=0.6)
    finally:
        driver.close(wait=True)
    assert artifact['clock']['outstanding_provider_turns_at_return'] == 0
    assert artifact['teardown']['behavioral_settlement_complete'] is True
    assert artifact['environment_observation_ingestion']['pending'] == 0
    assert artifact['environment_observation_ingestion']['canceled'] == 0
