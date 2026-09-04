from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

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
from runner.realtime_actor import (
    HoldSafetySupervisor,
    RealtimeEnvironmentActor,
    SafetyDecision,
)


@dataclass
class _StepResult:
    observation: dict
    done: bool = False


class _ClockEnvironment:
    def __init__(self, horizon: int = 20) -> None:
        self.tick = 0
        self.horizon = horizon
        self.applied_names: list[str | None] = []

    def snapshot(self) -> dict:
        return {"tick": self.tick}

    def step(self, action: Action) -> _StepResult:
        self.applied_names.append(
            action.tool_calls[0].name if action.tool_calls else None
        )
        self.tick += 1
        return _StepResult(
            observation={"tick": self.tick},
            done=self.tick >= self.horizon,
        )


class _FailingEnvironment(_ClockEnvironment):
    def step(self, action: Action) -> _StepResult:
        raise RuntimeError("private backend details")


class _BlockingEnvironment(_ClockEnvironment):
    def __init__(self) -> None:
        super().__init__(horizon=1)
        self.entered_step = threading.Event()
        self.release_step = threading.Event()

    def step(self, action: Action) -> _StepResult:
        self.entered_step.set()
        assert self.release_step.wait(timeout=1.0)
        return super().step(action)


class _MinimumRiskSupervisor:
    def decide(self, *, observation: dict, simulator_tick: int, reason: str) -> SafetyDecision:
        del observation, simulator_tick
        return SafetyDecision(
            action=Action(tool_calls=[ToolCall(name="minimum_risk_hold")]),
            mode="minimum_risk_fallback",
            reason_code=reason,
        )


@pytest.mark.parametrize("tick_interval_s", [float("nan"), float("inf"), 1e-12])
def test_environment_actor_rejects_invalid_clock_interval(
    tick_interval_s: float,
) -> None:
    with pytest.raises(ValueError, match="finite and at least 1ns"):
        RealtimeEnvironmentActor(
            _ClockEnvironment(), tick_interval_s=tick_interval_s
        )


@pytest.mark.parametrize(
    "snapshot",
    [{}, {"tick": True}, {"tick": "0"}, {"tick": 1}],
)
def test_environment_actor_rejects_invalid_initial_clock_snapshot(
    snapshot: dict,
) -> None:
    class InvalidSnapshotEnvironment(_ClockEnvironment):
        def snapshot(self) -> dict:
            return snapshot

    actor = RealtimeEnvironmentActor(
        InvalidSnapshotEnvironment(), tick_interval_s=0.01
    )
    with pytest.raises(ValueError, match="initial environment tick"):
        actor.start()


def test_environment_actor_rejects_nonmapping_initial_snapshot() -> None:
    class InvalidSnapshotEnvironment(_ClockEnvironment):
        def snapshot(self) -> list[dict]:
            return []

    actor = RealtimeEnvironmentActor(
        InvalidSnapshotEnvironment(), tick_interval_s=0.01
    )
    with pytest.raises(TypeError, match="snapshot must be a mapping"):
        actor.start()


def test_actor_fatal_error_settles_active_submission_exactly_once() -> None:
    class MalformedResult:
        @property
        def observation(self):
            raise TypeError("private malformed observation")

    class MalformedEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=1)
            self.step_entered = threading.Event()

        def step(self, action: Action):
            del action
            self.step_entered.set()
            return MalformedResult()

    env = MalformedEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.01)
    actor.start()
    future = actor.submit_current(
        Action(tool_calls=[ToolCall(name="control")]),
        action_id="fatal-action",
        decision_id="fatal-decision",
        turn_id="fatal-turn",
    )
    assert env.step_entered.wait(timeout=1.0)
    receipt = future.result(timeout=1.0)
    actor.stop()

    assert receipt.status == "failed"
    assert receipt.reason_code == "ACTOR_FATAL_ERROR"
    assert actor.fatal_error() == {
        "error_type": "TypeError",
        "stage": "environment_actor_loop",
    }
    receipts = actor.receipt_records()
    assert len(receipts) == 1
    assert receipts[0]["action_id"] == "fatal-action"
    assert [
        row["status"]
        for row in actor.lifecycle_records()
        if row["action_id"] == "fatal-action"
    ] == ["queued", "accepted", "failed"]


@pytest.mark.parametrize("returned_tick", [0, -1, "invalid", None])
def test_actor_fails_closed_when_environment_tick_does_not_advance(
    returned_tick: object,
) -> None:
    class NonAdvancingEnvironment(_ClockEnvironment):
        def step(self, action: Action) -> _StepResult:
            del action
            observation = {} if returned_tick is None else {"tick": returned_tick}
            return _StepResult(observation=observation)

    actor = RealtimeEnvironmentActor(
        NonAdvancingEnvironment(horizon=1), tick_interval_s=0.01
    )
    actor.start()
    future = actor.submit_current(
        Action(tool_calls=[ToolCall(name="control")]),
        action_id="nonadvancing-action",
    )
    receipt = future.result(timeout=1.0)
    actor.stop()

    assert receipt.status == "failed"
    assert receipt.reason_code == "ACTOR_FATAL_ERROR"
    assert actor.fatal_error() is not None
    fatal = actor.transition_records()[-1]
    assert fatal["actor_fatal"] is True
    assert fatal["simulator_time_advanced"] is False


class _FailingSupervisor:
    def decide(self, *, observation: dict, simulator_tick: int, reason: str) -> SafetyDecision:
        del observation, simulator_tick, reason
        raise RuntimeError("private shield failure")


def test_environment_advances_while_provider_is_slow_and_rejects_stale_action() -> None:
    env = _ClockEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.01)
    actor.start()
    try:
        observed_version, _ = actor.snapshot()
        time.sleep(0.04)  # provider remains in flight while the world advances
        assert actor.wait_for_version(3, timeout_s=1.0)

        late = actor.submit(
            Action(tool_calls=[ToolCall(name="late_control")]),
            action_id="late",
            based_on_state_version=observed_version,
        ).result(timeout=1.0)
        assert late.status == "stale"
        assert late.reason_code == "STALE_STATE"

        current = actor.submit_current(
            Action(tool_calls=[ToolCall(name="current_control")]),
            action_id="current",
        ).result(timeout=1.0)
        assert current.status == "no_effect"
        assert "current_control" in env.applied_names
        assert "wait" in env.applied_names
        assert any(
            row["used_safety_fallback"] for row in actor.transition_records()
        )
    finally:
        actor.stop()


def test_readonly_investigation_returns_without_consuming_simulator_tick() -> None:
    class InvestigationEnvironment(_ClockEnvironment):
        def readonly_tool_names(self) -> set[str]:
            return {"inspect"}

        def execute_investigation(
            self, action: Action
        ) -> tuple[dict, list[ToolResult]]:
            assert [call.name for call in action.tool_calls] == ["inspect"]
            return self.snapshot(), [
                ToolResult(
                    name="inspect",
                    ok=True,
                    payload={"tick_seen": self.tick},
                    state_changing=False,
                    call_id=action.tool_calls[0].call_id,
                    evidence_id="inspect-evidence",
                )
            ]

    env = InvestigationEnvironment(horizon=2)
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.2)
    actor.start()
    try:
        receipt = actor.submit_current(
            Action(tool_calls=[ToolCall(name="inspect", call_id="inspect-call")]),
            action_id="inspect-action",
        ).result(timeout=0.1)
        second_receipt = actor.submit_current(
            Action(tool_calls=[ToolCall(name="inspect", call_id="inspect-call-2")]),
            action_id="inspect-action-2",
        ).result(timeout=0.1)
        state_version, observation = actor.snapshot()
    finally:
        actor.stop()

    assert receipt.status == "no_effect"
    assert receipt.request_tick == 0
    assert second_receipt.request_tick == 0
    assert state_version == 2
    assert observation["tick"] == 0
    assert env.tick == 0
    transitions = actor.transition_records()
    assert len(transitions) == 2
    assert all(row["simulator_time_advanced"] is False for row in transitions)
    assert all(row["simulator_tick_before"] == 0 for row in transitions)
    assert all(row["simulator_tick"] == 0 for row in transitions)


def test_stale_investigation_snapshot_cannot_regress_authoritative_clock() -> None:
    class StaleSnapshotInvestigationEnvironment(_ClockEnvironment):
        def readonly_tool_names(self) -> set[str]:
            return {"inspect"}

        def execute_investigation(
            self, action: Action
        ) -> tuple[dict, list[ToolResult]]:
            return {"tick": self.tick - 1}, [
                ToolResult(
                    name="inspect",
                    ok=True,
                    state_changing=False,
                    call_id=action.tool_calls[0].call_id,
                )
            ]

    env = StaleSnapshotInvestigationEnvironment(horizon=3)
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.08)
    actor.start()
    try:
        assert actor.wait_for_version(1, timeout_s=1.0)
        investigation = actor.submit_current(
            Action(tool_calls=[ToolCall(name="inspect", call_id="inspect-call")]),
            action_id="inspect-action",
            decision_id="inspect-decision",
            turn_id="inspect-turn",
        ).result(timeout=0.05)
        state_version, observation = actor.snapshot()
        control = actor.submit(
            Action(tool_calls=[ToolCall(name="control")]),
            action_id="control-action",
            decision_id="control-decision",
            turn_id="control-turn",
            based_on_state_version=state_version,
            valid_from_tick=1,
            expires_at_tick=2,
        ).result(timeout=1.0)
    finally:
        actor.stop()

    assert investigation.status == "no_effect"
    assert observation["tick"] == 1
    assert control.status == "no_effect"
    assert "control" in env.applied_names
    clock_ticks = [
        (row["simulator_tick_before"], row["simulator_tick"])
        for row in actor.transition_records()
    ]
    assert all(after >= before for before, after in clock_ticks)
    assert all(
        later[0] >= earlier[1]
        for earlier, later in zip(clock_ticks, clock_ticks[1:], strict=False)
    )


def test_slow_readonly_investigation_freeze_is_explicitly_audited() -> None:
    class SlowInvestigationEnvironment(_ClockEnvironment):
        def readonly_tool_names(self) -> set[str]:
            return {"inspect"}

        def execute_investigation(
            self, action: Action
        ) -> tuple[dict, list[ToolResult]]:
            time.sleep(0.045)
            return self.snapshot(), [
                ToolResult(
                    name="inspect",
                    ok=True,
                    state_changing=False,
                    call_id=action.tool_calls[0].call_id,
                )
            ]

    env = SlowInvestigationEnvironment(horizon=2)
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.01)
    actor.start()
    try:
        receipt = actor.submit_current(
            Action(tool_calls=[ToolCall(name="inspect", call_id="slow-inspect")]),
            action_id="slow-inspect-action",
        ).result(timeout=1.0)
        assert receipt.status == "no_effect"
        assert actor.wait_for_version(2, timeout_s=1.0)
    finally:
        actor.stop()

    investigation = next(
        row
        for row in actor.transition_records()
        if row.get("clock_advance_reason") == "READ_ONLY_INVESTIGATION"
    )
    assert investigation["clock_semantics"] == "soft_realtime_monotonic_single_writer"
    assert investigation["environment_progress_during_investigation"] is False
    assert investigation["investigation_duration_ns"] >= 30_000_000
    assert investigation["investigation_clock_deadline_overrun_ns"] > 0
    assert investigation["investigation_elapsed_tick_intervals"] >= 3


def test_slow_investigation_overrun_forces_clock_before_next_investigation() -> None:
    class FairInvestigationEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=1)
            self.first_started = threading.Event()
            self.release_first = threading.Event()
            self.executions: list[str] = []

        def readonly_tool_names(self) -> set[str]:
            return {"inspect"}

        def execute_investigation(
            self, action: Action
        ) -> tuple[dict, list[ToolResult]]:
            call_id = str(action.tool_calls[0].call_id)
            self.executions.append(call_id)
            if call_id == "first":
                self.first_started.set()
                assert self.release_first.wait(timeout=1.0)
            return self.snapshot(), [
                ToolResult(
                    name="inspect",
                    ok=True,
                    state_changing=False,
                    call_id=call_id,
                )
            ]

    env = FairInvestigationEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.02)
    actor.start()
    first = actor.submit_current(
        Action(tool_calls=[ToolCall(name="inspect", call_id="first")]),
        action_id="first-action",
    )
    assert env.first_started.wait(timeout=1.0)
    actor.submit_current(
        Action(tool_calls=[ToolCall(name="inspect", call_id="second")]),
        action_id="second-action",
    )
    time.sleep(0.03)
    env.release_first.set()
    assert first.result(timeout=1.0).status == "no_effect"
    assert actor.wait_until_done(timeout_s=1.0)
    actor.stop()

    records = actor.transition_records()
    first_index = next(
        index for index, row in enumerate(records) if row.get("action_id") == "first-action"
    )
    clock_index = next(
        index
        for index, row in enumerate(records)
        if row.get("simulator_time_advanced") is True
    )
    assert first_index < clock_index
    assert env.executions == ["first"]


def test_delayed_state_change_settles_only_after_materialized_effect() -> None:
    class DelayedEffectEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=2)
            self.evidence = EvidenceLogger("delayed-effect")
            self.call_id = "delayed-control-call"

        def step(self, action: Action) -> StepReturn:
            del action
            self.tick += 1
            if self.tick == 1:
                result = ToolResult(
                    name="delayed_control",
                    ok=True,
                    payload={"_status": "pending", "due_tick": 2},
                    state_changing=True,
                    call_id=self.call_id,
                    latency_ticks=1,
                )
                events: list[dict] = []
                evidence_ids: list[str] = []
            else:
                event = {
                    "event_id": "effect-event",
                    "origin": "agent_caused",
                    "agent_caused": True,
                    "call_id": self.call_id,
                    "tool_name": "delayed_control",
                    "requested_action": {"name": "delayed_control", "args": {}},
                    "before_state_digest": "before",
                    "after_state_digest": "after",
                }
                effect_id = self.evidence.log(
                    "realized_event", tick=self.tick, payload=event, source="engine"
                )
                result = ToolResult(
                    name="delayed_control",
                    ok=True,
                    payload={"_status": "applied"},
                    state_changing=True,
                    call_id=self.call_id,
                    evidence_id=effect_id,
                    effect_tick=self.tick,
                )
                events = [event]
                evidence_ids = [effect_id]
            return StepReturn(
                observation=self.snapshot(),
                tool_results=[result],
                reward=0.0,
                done=self.tick >= self.horizon,
                info=StepInfo(realized_events=events, evidence_ids=evidence_ids),
            )

    env = DelayedEffectEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.04)
    actor.start()
    future = actor.submit_current(
        Action(
            tool_calls=[
                ToolCall(
                    name="delayed_control",
                    call_id=env.call_id,
                    idempotency_key="delayed-control-key",
                )
            ]
        ),
        action_id="delayed-action",
        decision_id="delayed-decision",
        turn_id="delayed-turn",
    )
    assert actor.wait_for_transition_count(1, timeout_s=1.0)
    assert future.done() is False
    assert actor.transition_records()[0]["applied_action"] is None

    receipt = future.result(timeout=1.0)
    actor.stop()

    assert receipt.status == "effected"
    lifecycle = [
        row["status"]
        for row in actor.lifecycle_records()
        if row["action_id"] == "delayed-action"
    ]
    assert lifecycle == ["queued", "accepted", "pending", "applied", "effected"]
    assert len(actor.receipt_records()) == 1
    materialized = actor.transition_records()[1]
    assert materialized["action_source"] == "safety_supervisor"
    assert materialized["action_id"] == "delayed-action"
    assert materialized["effect_observed"] is True
    assert materialized["deferred_action_outcomes"][0]["status"] == "effected"


def test_mixed_immediate_and_pending_batch_preserves_terminal_effect_state() -> None:
    class MixedBatchEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=2)
            self.evidence = EvidenceLogger("mixed-batch")

        def step(self, action: Action) -> StepReturn:
            del action
            self.tick += 1
            if self.tick == 1:
                event = {
                    "event_id": "immediate-effect-event",
                    "origin": "agent_caused",
                    "agent_caused": True,
                    "call_id": "immediate-call",
                    "tool_name": "immediate_control",
                    "requested_action": {
                        "name": "immediate_control",
                        "args": {"value": 1},
                    },
                    "before_state_digest": "before",
                    "after_state_digest": "after",
                    "effect_tick": 1,
                }
                evidence_id = self.evidence.log(
                    "realized_event", tick=1, payload=event, source="engine"
                )
                results = [
                    ToolResult(
                        name="immediate_control",
                        ok=True,
                        payload={"_status": "applied"},
                        state_changing=True,
                        call_id="immediate-call",
                        evidence_id=evidence_id,
                        effect_tick=1,
                    ),
                    ToolResult(
                        name="delayed_control",
                        ok=True,
                        payload={"_status": "pending", "due_tick": 2},
                        state_changing=False,
                        call_id="delayed-call",
                        latency_ticks=1,
                    ),
                ]
                events = [event]
                evidence_ids = [evidence_id]
            else:
                results = [
                    ToolResult(
                        name="delayed_control",
                        ok=False,
                        payload={"_status": "error"},
                        error_code="DOMAIN_REJECTED",
                        state_changing=True,
                        call_id="delayed-call",
                        effect_tick=2,
                    )
                ]
                events = []
                evidence_ids = []
            return StepReturn(
                observation=self.snapshot(),
                tool_results=results,
                reward=0.0,
                done=self.tick >= self.horizon,
                info=StepInfo(realized_events=events, evidence_ids=evidence_ids),
            )

    actor = RealtimeEnvironmentActor(
        MixedBatchEnvironment(), tick_interval_s=0.04
    )
    actor.start()
    receipt = actor.submit_current(
        Action(
            tool_calls=[
                ToolCall(
                    name="immediate_control",
                    args={"value": 1},
                    call_id="immediate-call",
                ),
                ToolCall(name="delayed_control", call_id="delayed-call"),
            ]
        ),
        action_id="mixed-action",
    ).result(timeout=1.0)
    actor.stop()

    assert receipt.status == "effected"
    assert receipt.applied_tick == 0
    materialized = actor.transition_records()[1]
    [outcome] = materialized["deferred_action_outcomes"]
    assert outcome["status"] == "effected"
    assert outcome["control_confirmed"] is True
    assert outcome["effect_observed"] is True
    assert outcome["effect_evidence_ids"]
    assert any(
        edge["call_id"] == "immediate-call"
        for edge in outcome["tool_trace_edges"]
    )


def test_delayed_state_change_without_cancellation_fails_closed_before_expiry() -> None:
    class LateEffectEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=2)
            self.evidence = EvidenceLogger("late-effect")
            self.call_id = "late-control-call"

        def step(self, action: Action) -> StepReturn:
            del action
            self.tick += 1
            if self.tick == 1:
                result = ToolResult(
                    name="delayed_control",
                    ok=True,
                    payload={"_status": "pending", "due_tick": 2},
                    state_changing=True,
                    call_id=self.call_id,
                    latency_ticks=1,
                )
                events: list[dict] = []
            else:
                event = {
                    "event_id": "late-effect-event",
                    "origin": "agent_caused",
                    "agent_caused": True,
                    "call_id": self.call_id,
                    "tool_name": "delayed_control",
                    "requested_action": {"name": "delayed_control", "args": {}},
                    "before_state_digest": "before",
                    "after_state_digest": "after",
                    "effect_tick": self.tick,
                }
                evidence_id = self.evidence.log(
                    "realized_event", tick=self.tick, payload=event, source="engine"
                )
                result = ToolResult(
                    name="delayed_control",
                    ok=True,
                    payload={"_status": "applied"},
                    state_changing=True,
                    call_id=self.call_id,
                    evidence_id=evidence_id,
                    effect_tick=self.tick,
                )
                events = [event]
            return StepReturn(
                observation=self.snapshot(),
                tool_results=[result],
                reward=0.0,
                done=self.tick >= self.horizon,
                info=StepInfo(realized_events=events),
            )

    actor = RealtimeEnvironmentActor(
        LateEffectEnvironment(), tick_interval_s=0.04
    )
    actor.start()
    future = actor.submit_current(
        Action(
            tool_calls=[ToolCall(name="delayed_control", call_id="late-control-call")]
        ),
        action_id="late-action",
        expires_at_tick=1,
    )
    receipt = future.result(timeout=1.0)
    actor.stop()

    assert receipt.status == "failed"
    assert receipt.reason_code == "EXECUTION_FENCE_FAILED"
    assert receipt.applied_tick is None
    assert [
        row["status"]
        for row in actor.lifecycle_records()
        if row["action_id"] == "late-action"
    ] == ["queued", "accepted", "pending", "failed"]
    assert actor._env.tick == 1
    assert not any(
        transition.get("realized_events")
        for transition in actor.transition_records()
    )
    fence = actor.transition_records()[-1]
    assert fence["execution_fence_failed"] is True
    assert fence["rejection_reason"] == "EXECUTION_FENCE_FAILED"


def test_uncancellable_supersession_fails_closed_before_deferred_effect() -> None:
    class SupersededEffectEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=2)
            self.evidence = EvidenceLogger("superseded-effect")
            self.old_call_id = "old-control-call"

        def step(self, action: Action) -> StepReturn:
            del action
            self.tick += 1
            if self.tick == 1:
                result = ToolResult(
                    name="delayed_control",
                    ok=True,
                    payload={"_status": "pending", "due_tick": 2},
                    state_changing=True,
                    call_id=self.old_call_id,
                    latency_ticks=1,
                )
                events: list[dict] = []
            else:
                event = {
                    "event_id": "superseded-effect-event",
                    "origin": "agent_caused",
                    "agent_caused": True,
                    "call_id": self.old_call_id,
                    "tool_name": "delayed_control",
                    "requested_action": {"name": "delayed_control", "args": {}},
                    "before_state_digest": "before",
                    "after_state_digest": "after",
                    "effect_tick": self.tick,
                }
                evidence_id = self.evidence.log(
                    "realized_event", tick=self.tick, payload=event, source="engine"
                )
                result = ToolResult(
                    name="delayed_control",
                    ok=True,
                    payload={"_status": "applied"},
                    state_changing=True,
                    call_id=self.old_call_id,
                    evidence_id=evidence_id,
                    effect_tick=self.tick,
                )
                events = [event]
            return StepReturn(
                observation=self.snapshot(),
                tool_results=[result],
                reward=0.0,
                done=self.tick >= self.horizon,
                info=StepInfo(realized_events=events),
            )

    actor = RealtimeEnvironmentActor(
        SupersededEffectEnvironment(), tick_interval_s=0.08
    )
    actor.start()
    old = actor.submit_current(
        Action(
            tool_calls=[ToolCall(name="delayed_control", call_id="old-control-call")]
        ),
        action_id="old-action",
    )
    assert actor.wait_for_transition_count(1, timeout_s=1.0)
    new = actor.submit_current(
        Action(tool_calls=[ToolCall(name="replacement", call_id="new-call")]),
        action_id="new-action",
        supersedes_action_id="old-action",
    )

    old_receipt = old.result(timeout=1.0)
    new_receipt = new.result(timeout=1.0)
    actor.stop()

    assert old_receipt.status == "failed"
    assert old_receipt.reason_code == "EXECUTION_FENCE_FAILED"
    assert old_receipt.applied_tick is None
    assert new_receipt.status == "failed"
    assert new_receipt.reason_code == "EXECUTION_FENCE_FAILED"
    assert [
        row["status"]
        for row in actor.lifecycle_records()
        if row["action_id"] == "old-action"
    ] == ["queued", "accepted", "pending", "failed"]
    late_transition = actor.transition_records()[1]
    assert late_transition["simulator_time_advanced"] is False
    assert late_transition["execution_fence_failed"] is True
    assert late_transition["cancellation_audit"] == [
        {
            "call_id": "old-control-call",
            "queue_kind": None,
            "outcome": "registry_unavailable",
            "callback_invoked": False,
            "callback_error_type": None,
        }
    ]


def test_explicit_supersession_cancels_queued_handler_before_backend_mutation() -> None:
    class QueuedHandlerEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=2)
            self.value = 0
            self._tools = ToolRegistry(seed=42)

            def mutate(args: dict, _ctx: ToolContext) -> dict:
                self.value += int(args["increment"])
                return {"_status": "applied", "value": self.value}

            self._tools.register(
                ToolSpec(
                    name="delayed_increment",
                    description="Delayed mutation.",
                    parameters={
                        "type": "object",
                        "properties": {"increment": {"type": "integer"}},
                        "required": ["increment"],
                    },
                    handler=mutate,
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
                    semantic_role="meta",
                    fail_rate=0.0,
                    delay_ticks=0,
                )
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

    env = QueuedHandlerEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.08)
    actor.start()
    old = actor.submit_current(
        Action(
            tool_calls=[
                ToolCall(
                    name="delayed_increment",
                    args={"increment": 1},
                    call_id="queued-mutation",
                )
            ]
        ),
        action_id="queued-action",
    )
    assert actor.wait_for_transition_count(1, timeout_s=1.0)
    replacement = actor.submit_current(
        Action(tool_calls=[ToolCall(name="wait", call_id="replacement-wait")]),
        action_id="replacement-action",
        supersedes_action_id="queued-action",
    )

    assert old.result(timeout=1.0).status == "superseded"
    assert replacement.result(timeout=1.0).status == "no_effect"
    actor.stop()

    assert env.value == 0


def test_superseding_investigation_fences_due_control_before_tool_execution() -> None:
    class InvestigationFenceEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=2)
            self.value = 0
            self._tools = ToolRegistry(seed=42)
            self._tools.register(
                ToolSpec(
                    name="delayed_increment",
                    description="Delayed mutation.",
                    parameters={"type": "object", "properties": {}},
                    handler=self._mutate,
                    state_changing=True,
                    semantic_role="control",
                    fail_rate=0.0,
                    delay_ticks=1,
                )
            )
            self._tools.register(
                ToolSpec(
                    name="inspect",
                    description="Inspect current state.",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _args, _ctx: {"value": self.value},
                    state_changing=False,
                    semantic_role="investigation",
                    fail_rate=0.0,
                    delay_ticks=0,
                )
            )

        def _mutate(self, _args: dict, _ctx: ToolContext) -> dict:
            self.value += 1
            return {"_status": "applied", "value": self.value}

        def step(self, action: Action) -> StepReturn:
            results = self._execute(action)
            self.tick += 1
            return StepReturn(
                observation=self.snapshot(),
                tool_results=results,
                reward=0.0,
                done=self.tick >= self.horizon,
                info=StepInfo(),
            )

        def execute_investigation(
            self, action: Action
        ) -> tuple[dict, list[ToolResult]]:
            return self.snapshot(), self._execute(action)

        def _execute(self, action: Action) -> list[ToolResult]:
            return self._tools.execute_action(
                action,
                ToolContext(
                    tick=self.tick,
                    seed=42,
                    extra={"episode_horizon": self.horizon, "env": self},
                ),
            )

    env = InvestigationFenceEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.15)
    actor.start()
    old = actor.submit_current(
        Action(
            tool_calls=[
                ToolCall(name="delayed_increment", call_id="due-control")
            ]
        ),
        action_id="old-control",
    )
    assert actor.wait_for_transition_count(1, timeout_s=1.0)
    inspect = actor.submit_current(
        Action(tool_calls=[ToolCall(name="inspect", call_id="inspect-call")]),
        action_id="inspect-action",
        supersedes_action_id="old-control",
    )

    assert old.result(timeout=1.0).status == "superseded"
    assert inspect.result(timeout=1.0).status == "no_effect"
    actor.stop()

    assert env.value == 0
    investigation = next(
        row
        for row in actor.transition_records()
        if row.get("action_id") == "inspect-action"
    )
    assert investigation["cancellation_audit"] == [
        {
            "call_id": "due-control",
            "queue_kind": "registry_invocation",
            "outcome": "canceled",
            "callback_invoked": False,
            "callback_error_type": None,
        }
    ]


def test_supersession_during_materialization_does_not_retroactively_cancel() -> None:
    class MaterializationBoundaryEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=3)
            self.value = 0
            self.materialized = threading.Event()
            self.release_materialization = threading.Event()
            self._tools = ToolRegistry(seed=42)

            def mutate(args: dict, _ctx: ToolContext) -> dict:
                self.value += int(args["increment"])
                return {"_status": "applied", "value": self.value}

            self._tools.register(
                ToolSpec(
                    name="delayed_increment",
                    description="Delayed mutation.",
                    parameters={
                        "type": "object",
                        "properties": {"increment": {"type": "integer"}},
                        "required": ["increment"],
                    },
                    handler=mutate,
                    state_changing=True,
                    semantic_role="control",
                    fail_rate=0.0,
                    delay_ticks=1,
                )
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
            if self.tick == 1:
                self.materialized.set()
                assert self.release_materialization.wait(timeout=1.0)
            self.tick += 1
            return StepReturn(
                observation=self.snapshot(),
                tool_results=results,
                reward=0.0,
                done=self.tick >= self.horizon,
                info=StepInfo(),
            )

    env = MaterializationBoundaryEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.04)
    actor.start()
    old = actor.submit_current(
        Action(
            tool_calls=[
                ToolCall(
                    name="delayed_increment",
                    args={"increment": 1},
                    call_id="materializing-call",
                )
            ]
        ),
        action_id="materializing-action",
    )
    assert actor.wait_for_transition_count(1, timeout_s=1.0)
    assert env.materialized.wait(timeout=1.0)
    replacement = actor.submit_current(
        Action(),
        action_id="replacement-action",
        supersedes_action_id="materializing-action",
    )
    env.release_materialization.set()

    assert old.result(timeout=1.0).status == "confirmed"
    assert replacement.result(timeout=1.0).status == "stale"
    assert actor.wait_until_done(timeout_s=1.0)
    actor.stop()

    assert env.value == 1
    audit = [
        row
        for transition in actor.transition_records()
        for row in transition.get("cancellation_audit", [])
    ]
    assert audit == [
        {
            "call_id": "materializing-call",
            "queue_kind": None,
            "outcome": "already_materialized_or_resolved",
            "callback_invoked": False,
            "callback_error_type": None,
        }
    ]


def test_expiry_cancels_queued_handler_before_backend_mutation() -> None:
    class ExpiringQueuedEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=2)
            self.value = 0
            self._tools = ToolRegistry(seed=42)

            def mutate(args: dict, _ctx: ToolContext) -> dict:
                self.value += int(args["increment"])
                return {"_status": "applied", "value": self.value}

            self._tools.register(
                ToolSpec(
                    name="delayed_increment",
                    description="Delayed mutation.",
                    parameters={
                        "type": "object",
                        "properties": {"increment": {"type": "integer"}},
                        "required": ["increment"],
                    },
                    handler=mutate,
                    state_changing=True,
                    semantic_role="control",
                    fail_rate=0.0,
                    delay_ticks=1,
                )
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

    env = ExpiringQueuedEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.04)
    actor.start()
    receipt = actor.submit_current(
        Action(
            tool_calls=[
                ToolCall(
                    name="delayed_increment",
                    args={"increment": 1},
                    call_id="expiring-mutation",
                )
            ]
        ),
        action_id="expiring-action",
        expires_at_tick=0,
    ).result(timeout=1.0)
    actor.stop()

    assert receipt.status == "expired"
    assert env.value == 0


def test_realtime_rejects_uncancellable_handler_managed_delay_before_queueing() -> None:
    class UncancellableManagedEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=1)
            self.queued = 0
            self._tools = ToolRegistry(seed=42)

            def queue_native(_args: dict, ctx: ToolContext) -> dict:
                self.queued += 1
                return {"_status": "pending", "due_tick": ctx.tick + 1}

            self._tools.register(
                ToolSpec(
                    name="uncancellable_native",
                    description="Native backend manages the delay.",
                    parameters={"type": "object", "properties": {}},
                    handler=queue_native,
                    state_changing=True,
                    semantic_role="control",
                    fail_rate=0.0,
                    delay_ticks=1,
                    handler_manages_delay=True,
                )
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
                done=True,
                info=StepInfo(),
            )

    env = UncancellableManagedEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.01)
    actor.start()
    receipt = actor.submit_current(
        Action(
            tool_calls=[
                ToolCall(name="uncancellable_native", call_id="native-call")
            ]
        ),
        action_id="native-action",
    ).result(timeout=1.0)
    actor.stop()

    assert receipt.status == "rejected"
    assert receipt.reason_code == "UNCANCELLABLE_MANAGED_DELAY"
    assert env.queued == 0


def test_realtime_allows_handler_managed_delay_with_cancellation_contract() -> None:
    class CancellableManagedEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=2)
            self.queued: set[str] = set()
            self._tools = ToolRegistry(seed=42)

            def queue_native(args: dict, ctx: ToolContext) -> dict:
                self.queued.add(str(args["job_id"]))
                return {"_status": "pending", "due_tick": ctx.tick + 1}

            def cancel_native(call: ToolCall) -> bool:
                job_id = str(call.args["job_id"])
                if job_id not in self.queued:
                    return False
                self.queued.remove(job_id)
                return True

            self._tools.register(
                ToolSpec(
                    name="cancellable_native",
                    description="Native backend manages a cancellable delay.",
                    parameters={
                        "type": "object",
                        "properties": {"job_id": {"type": "string"}},
                        "required": ["job_id"],
                    },
                    handler=queue_native,
                    state_changing=True,
                    semantic_role="control",
                    fail_rate=0.0,
                    delay_ticks=1,
                    handler_manages_delay=True,
                    cancel_pending=cancel_native,
                )
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

    env = CancellableManagedEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.01)
    actor.start()
    receipt = actor.submit_current(
        Action(
            tool_calls=[
                ToolCall(
                    name="cancellable_native",
                    args={"job_id": "job-1"},
                    call_id="native-call",
                )
            ]
        ),
        action_id="native-action",
    ).result(timeout=1.0)
    actor.stop()

    assert receipt.status == "confirmed"
    assert env.queued == {"job-1"}


@pytest.mark.parametrize(
    ("failure_mode", "expected_outcome"),
    [("return_false", "callback_false"), ("raise", "callback_exception")],
)
def test_partial_native_cancellation_fails_closed_on_actor_step_boundary(
    failure_mode: str,
    expected_outcome: str,
) -> None:
    class PartialCancellationEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=4)
            self.value = 0
            self.queued: dict[str, int] = {}
            self.callback_thread_ids: list[int] = []
            self.step_thread_ids: list[int] = []
            self._tools = ToolRegistry(seed=42)

            def queue_native(args: dict, ctx: ToolContext) -> dict:
                job_id = str(args["job_id"])
                self.queued[job_id] = ctx.tick + 2
                return {"_status": "pending", "due_tick": ctx.tick + 2}

            def cancel_native(call: ToolCall) -> bool:
                self.callback_thread_ids.append(threading.get_ident())
                job_id = str(call.args["job_id"])
                if job_id == "refuse":
                    if failure_mode == "raise":
                        raise RuntimeError("private cancellation detail")
                    return False
                self.queued.pop(job_id, None)
                return True

            self._tools.register(
                ToolSpec(
                    name="managed_delay",
                    description="Native backend manages a cancellable delay.",
                    parameters={
                        "type": "object",
                        "properties": {"job_id": {"type": "string"}},
                        "required": ["job_id"],
                    },
                    handler=queue_native,
                    state_changing=True,
                    semantic_role="control",
                    fail_rate=0.0,
                    delay_ticks=2,
                    handler_manages_delay=True,
                    cancel_pending=cancel_native,
                )
            )

        def step(self, action: Action) -> StepReturn:
            self.step_thread_ids.append(threading.get_ident())
            results = self._tools.execute_action(
                action,
                ToolContext(
                    tick=self.tick,
                    seed=42,
                    extra={"episode_horizon": self.horizon, "env": self},
                ),
            )
            self.value += sum(
                1 for due_tick in self.queued.values() if due_tick <= self.tick
            )
            self.tick += 1
            return StepReturn(
                observation=self.snapshot(),
                tool_results=results,
                reward=0.0,
                done=self.tick >= self.horizon,
                info=StepInfo(),
            )

    env = PartialCancellationEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.08)
    actor.start()
    old = actor.submit_current(
        Action(
            tool_calls=[
                ToolCall(
                    name="managed_delay",
                    args={"job_id": job_id},
                    call_id=f"call-{job_id}",
                )
                for job_id in ("cancel", "refuse")
            ]
        ),
        action_id="old-action",
    )
    assert actor.wait_for_transition_count(1, timeout_s=1.0)
    submitter_thread_id = threading.get_ident()
    replacement = actor.submit_current(
        Action(),
        action_id="replacement-action",
        supersedes_action_id="old-action",
    )

    assert actor.wait_until_done(timeout_s=1.0)
    old_receipt = old.result(timeout=1.0)
    replacement_receipt = replacement.result(timeout=1.0)
    actor.stop()

    assert old_receipt.status == "failed"
    assert old_receipt.reason_code == "EXECUTION_FENCE_FAILED"
    assert replacement_receipt.status == "failed"
    assert replacement_receipt.reason_code == "EXECUTION_FENCE_FAILED"
    assert env.tick == 1
    assert env.value == 0
    assert set(env.callback_thread_ids) == {env.step_thread_ids[0]}
    assert submitter_thread_id not in env.callback_thread_ids
    audit = next(
        row["cancellation_audit"]
        for row in actor.transition_records()
        if row.get("cancellation_audit")
    )
    assert {row["call_id"]: row["outcome"] for row in audit} == {
        "call-cancel": "canceled",
        "call-refuse": expected_outcome,
    }


def test_real_tool_registry_pending_control_ack_waits_for_materialization() -> None:
    class RegistryDelayedEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=2)
            self._tools = ToolRegistry(seed=42)
            self._tools.register(
                ToolSpec(
                    name="delayed_control",
                    description="Delayed control.",
                    parameters={"type": "object", "properties": {}},
                    handler=lambda _args, _ctx: {"_status": "applied"},
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

    actor = RealtimeEnvironmentActor(
        RegistryDelayedEnvironment(), tick_interval_s=0.04
    )
    actor.start()
    future = actor.submit_current(
        Action(tool_calls=[ToolCall(name="delayed_control")]),
        action_id="registry-delayed-action",
    )
    assert actor.wait_for_transition_count(1, timeout_s=1.0)
    first = actor.transition_records()[0]
    assert first["tool_results"][0]["state_changing"] is False
    assert first["tool_results"][0]["payload"]["_status"] == "pending"
    assert future.done() is False

    receipt = future.result(timeout=1.0)
    actor.stop()

    assert receipt.status == "confirmed"
    assert [
        row["status"]
        for row in actor.lifecycle_records()
        if row["action_id"] == "registry-delayed-action"
    ] == ["queued", "accepted", "pending", "applied", "confirmed"]


def test_pending_state_change_at_terminal_boundary_fails_without_applied() -> None:
    class UnresolvedDelayedEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=1)

        def step(self, action: Action) -> StepReturn:
            del action
            self.tick += 1
            return StepReturn(
                observation=self.snapshot(),
                tool_results=[
                    ToolResult(
                        name="delayed_control",
                        ok=True,
                        payload={"_status": "pending", "due_tick": 2},
                        state_changing=True,
                        call_id="terminal-pending-call",
                    )
                ],
                reward=0.0,
                done=True,
                info=StepInfo(),
            )

    actor = RealtimeEnvironmentActor(
        UnresolvedDelayedEnvironment(), tick_interval_s=0.03
    )
    actor.start()
    receipt = actor.submit_current(
        Action(
            tool_calls=[
                ToolCall(name="delayed_control", call_id="terminal-pending-call")
            ]
        ),
        action_id="terminal-pending-action",
    ).result(timeout=1.0)
    actor.stop()

    assert receipt.status == "failed"
    assert receipt.reason_code == "DELAYED_ACTION_UNRESOLVED_AT_EPISODE_END"
    lifecycle = [
        row["status"]
        for row in actor.lifecycle_records()
        if row["action_id"] == "terminal-pending-action"
    ]
    assert lifecycle == ["queued", "accepted", "pending", "failed"]


def test_safety_supervisor_arbitrates_pass_override_and_reject() -> None:
    class ArbitratingSupervisor(HoldSafetySupervisor):
        def arbitrate(
            self,
            *,
            observation: dict,
            simulator_tick: int,
            candidate_action: Action,
        ) -> SafetyDecision:
            del observation
            name = candidate_action.tool_calls[0].name
            if name == "safe":
                return SafetyDecision(
                    action=candidate_action,
                    mode="native_shield",
                    reason_code="WITHIN_ENVELOPE",
                    supervisor_id="test-shield",
                    disposition="pass",
                    evidence_ids=("safety-pass",),
                )
            disposition = "override" if name == "adjustable" else "reject"
            return SafetyDecision(
                action=Action(tool_calls=[ToolCall(name="minimum_risk_hold")]),
                mode="native_shield",
                reason_code=f"{disposition.upper()}_UNSAFE",
                supervisor_id="test-shield",
                disposition=disposition,
                evidence_ids=(f"safety-{disposition}",),
            )

    for nominal_name, expected_disposition, expected_applied in (
        ("safe", "pass", "safe"),
        ("adjustable", "override", "minimum_risk_hold"),
        ("unsafe", "reject", "minimum_risk_hold"),
    ):
        env = _ClockEnvironment(horizon=1)
        actor = RealtimeEnvironmentActor(
            env,
            tick_interval_s=0.03,
            safety_supervisor=ArbitratingSupervisor(),
        )
        actor.start()
        receipt = actor.submit_current(
            Action(tool_calls=[ToolCall(name=nominal_name)]),
            action_id=f"action-{nominal_name}",
        ).result(timeout=1.0)
        actor.stop()

        transition = actor.transition_records()[0]
        assert transition["safety_arbitration"]["disposition"] == expected_disposition
        assert transition["submitted_action"]["actions"][0]["name"] == nominal_name
        assert transition["nominal_action"]["actions"][0]["name"] == nominal_name
        assert transition["applied_action"]["actions"][0]["name"] == expected_applied
        assert transition["model_attempted_state_change"] is True
        assert transition["safety_evidence_ids"] == [
            f"safety-{expected_disposition}"
        ]
        if expected_disposition == "pass":
            assert receipt.status == "no_effect"
            assert transition["action_source"] == "model"
        else:
            assert receipt.status == "rejected"
            assert receipt.reason_code == f"SAFETY_{expected_disposition.upper()}"
            assert transition["action_source"] == "safety_supervisor"


def test_rejected_readonly_investigation_settles_immediately_with_safety_audit() -> None:
    class ReadonlyEnvironment(_ClockEnvironment):
        def readonly_tool_names(self) -> set[str]:
            return {"inspect"}

    class RejectingSupervisor(HoldSafetySupervisor):
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
                mode="native_shield",
                reason_code="READ_BLOCKED",
                disposition="reject",
                evidence_ids=("shield-evidence",),
            )

    actor = RealtimeEnvironmentActor(
        ReadonlyEnvironment(horizon=1),
        tick_interval_s=5.0,
        safety_supervisor=RejectingSupervisor(),
    )
    actor.start()
    try:
        receipt = actor.submit_current(
            Action(tool_calls=[ToolCall(name="inspect")]),
            action_id="blocked-inspection",
        ).result(timeout=0.2)
    finally:
        actor.stop()

    assert receipt.status == "rejected"
    assert receipt.reason_code == "SAFETY_REJECT"
    transition = actor.transition_records()[0]
    assert transition["simulator_time_advanced"] is False
    assert transition["safety_arbitration"]["disposition"] == "reject"
    assert transition["safety_evidence_ids"] == ["shield-evidence"]
    assert transition["model_attempted_state_change"] is False


def test_slow_environment_step_records_clock_overrun_without_catchup_burst() -> None:
    class SlowEnvironment(_ClockEnvironment):
        def step(self, action: Action) -> _StepResult:
            time.sleep(0.035)
            return super().step(action)

    actor = RealtimeEnvironmentActor(
        SlowEnvironment(horizon=2),
        tick_interval_s=0.01,
    )
    actor.start()
    assert actor.wait_until_done(timeout_s=1.0)
    actor.stop()

    transitions = actor.transition_records()
    assert len(transitions) == 2
    assert all(row["simulator_time_advanced"] is True for row in transitions)
    assert all(row["clock_execution_overrun_ns"] > 0 for row in transitions)
    assert all(row["clock_catchup_policy"] == "skip_missed_intervals" for row in transitions)
    assert transitions[0]["clock_catchup_skipped_intervals"] >= 1
    assert (
        transitions[1]["clock_started_monotonic_ns"]
        - transitions[0]["clock_finished_monotonic_ns"]
        >= 8_000_000
    )


def test_hidden_event_evidence_is_not_injected_into_realtime_agent_snapshot() -> None:
    class HiddenEvidenceEnvironment(_ClockEnvironment):
        def step(self, action: Action) -> StepReturn:
            del action
            self.tick += 1
            return StepReturn(
                observation={"tick": self.tick},
                tool_results=[
                    ToolResult(
                        name="inspect",
                        ok=True,
                        evidence_id="visible-tool-evidence",
                    )
                ],
                reward=0.0,
                done=self.tick >= 1,
                info=StepInfo(
                    realized_events=[
                        {
                            "event_id": "hidden-event",
                            "type": "hidden_fault",
                            "hidden": True,
                            "evidence_ids": ["hidden-evidence"],
                        }
                    ],
                    evidence_ids=["hidden-evidence", "visible-tool-evidence"],
                ),
            )

    actor = RealtimeEnvironmentActor(
        HiddenEvidenceEnvironment(horizon=1),
        tick_interval_s=0.01,
    )
    actor.start()
    assert actor.wait_until_done(timeout_s=1.0)
    _, observation = actor.snapshot()
    actor.stop()

    assert observation["__last_realized_events__"] == []
    assert observation["__last_evidence_ids__"] == ["visible-tool-evidence"]


def test_latest_valid_action_supersedes_older_submission() -> None:
    env = _ClockEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.05)
    actor.start()
    try:
        version, _ = actor.snapshot()
        first = actor.submit(
            Action(tool_calls=[ToolCall(name="first")]),
            action_id="first",
            based_on_state_version=version,
        )
        second = actor.submit(
            Action(tool_calls=[ToolCall(name="second")]),
            action_id="second",
            based_on_state_version=version,
        )

        assert first.result(timeout=1.0).status == "superseded"
        assert second.result(timeout=1.0).status == "no_effect"
        assert env.applied_names[0] == "second"
    finally:
        actor.stop()


def test_expired_action_is_distinct_from_unexpired_stale_action() -> None:
    actor = RealtimeEnvironmentActor(_ClockEnvironment(horizon=4), tick_interval_s=0.03)
    actor.start()
    try:
        assert actor.wait_for_version(1, timeout_s=1.0)
        expired = actor.submit(
            Action(tool_calls=[ToolCall(name="expired-control")]),
            action_id="expired-action",
            based_on_state_version=0,
            valid_from_tick=0,
            expires_at_tick=0,
        )
        stale = actor.submit(
            Action(tool_calls=[ToolCall(name="stale-control")]),
            action_id="stale-action",
            based_on_state_version=0,
            valid_from_tick=0,
            expires_at_tick=3,
        )
        assert expired.result(timeout=1.0).status == "expired"
        assert stale.result(timeout=1.0).status == "stale"
    finally:
        actor.stop()


def test_transition_uses_decision_time_visible_evidence_snapshot() -> None:
    env = _ClockEnvironment(horizon=2)
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.05)
    actor.start()
    try:
        version, _ = actor.snapshot()
        receipt = actor.submit(
            Action(tool_calls=[ToolCall(name="control")]),
            action_id="decision-evidence",
            based_on_state_version=version,
            based_on_visible_evidence_ids=["visible-at-decision"],
        ).result(timeout=1.0)
        assert receipt.status == "no_effect"
    finally:
        actor.stop()

    model_transition = next(
        row
        for row in actor.transition_records()
        if row.get("action_id") == "decision-evidence"
    )
    assert model_transition["based_on_visible_evidence_ids"] == [
        "visible-at-decision"
    ]


def test_action_with_invisible_evidence_reference_is_rejected_before_execution() -> None:
    env = _ClockEnvironment(horizon=2)
    actor = RealtimeEnvironmentActor(
        env,
        tick_interval_s=0.05,
        safety_supervisor=_MinimumRiskSupervisor(),
    )
    actor.start()
    try:
        version, _ = actor.snapshot()
        receipt = actor.submit(
            Action(
                tool_calls=[
                    ToolCall(
                        name="control",
                        consumes_evidence_ids=["memory-key-not-evidence"],
                    )
                ]
            ),
            action_id="invalid-evidence",
            based_on_state_version=version,
            based_on_visible_evidence_ids=["ev-visible"],
        ).result(timeout=1.0)
    finally:
        actor.stop()

    assert receipt.status == "rejected"
    assert receipt.reason_code == "INVISIBLE_EVIDENCE_REFERENCE"
    assert "control" not in env.applied_names


def test_environment_step_failure_settles_selected_action_without_error_leak() -> None:
    actor = RealtimeEnvironmentActor(
        _FailingEnvironment(),
        tick_interval_s=0.02,
    )
    actor.start()
    future = actor.submit_current(
        Action(tool_calls=[ToolCall(name="unsafe_if_lost")]),
        action_id="selected",
    )

    receipt = future.result(timeout=1.0)
    actor.stop()

    assert receipt.status == "failed"
    assert receipt.reason_code == "ENVIRONMENT_STEP_FAILED"
    assert actor.snapshot()[0] == 0
    transition = actor.transition_records()[0]
    assert transition["environment_step_failed"] is True
    assert transition["environment_error_type"] == "RuntimeError"
    assert "private backend details" not in str(transition)


def test_canceled_future_cannot_kill_actor_or_strand_valid_submission() -> None:
    env = _ClockEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.05)
    actor.start()
    try:
        version, _ = actor.snapshot()
        canceled = actor.submit(
            Action(tool_calls=[ToolCall(name="canceled")]),
            action_id="canceled",
            based_on_state_version=version,
        )
        selected = actor.submit(
            Action(tool_calls=[ToolCall(name="selected")]),
            action_id="selected",
            based_on_state_version=version,
        )
        assert canceled.cancel() is True

        assert selected.result(timeout=1.0).status == "no_effect"
        assert env.applied_names[0] == "selected"
        assert actor.wait_for_version(2, timeout_s=1.0)
        receipts = actor.receipt_records()[:2]
        assert [row["action_id"] for row in receipts] == ["canceled", "selected"]
        assert [row["status"] for row in receipts] == ["canceled", "no_effect"]
        assert receipts[0]["reason_code"] == "CALLER_CANCELED"
        assert receipts[1]["accepted_state_version"] == version
        assert all(row["decision_id"] and row["turn_id"] for row in receipts)
    finally:
        actor.stop()


def test_submission_owns_action_snapshot_and_records_supersession() -> None:
    env = _ClockEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.05)
    actor.start()
    try:
        version, _ = actor.snapshot()
        original = Action(tool_calls=[ToolCall(name="original")])
        superseded = actor.submit(
            Action(tool_calls=[ToolCall(name="older")]),
            action_id="older",
            based_on_state_version=version,
        )
        selected = actor.submit(
            original,
            action_id="immutable",
            based_on_state_version=version,
        )
        original.tool_calls[0].name = "mutated-after-submit"

        assert superseded.result(timeout=1.0).status == "superseded"
        assert selected.result(timeout=1.0).status == "no_effect"
        assert env.applied_names[0] == "original"
        assert [row["status"] for row in actor.receipt_records()[:2]] == [
            "superseded",
            "no_effect",
        ]
    finally:
        actor.stop()


def test_safety_supervisor_is_explicit_and_action_lifecycle_is_identity_bound() -> None:
    env = _ClockEnvironment(horizon=4)
    actor = RealtimeEnvironmentActor(
        env,
        tick_interval_s=0.03,
        safety_supervisor=_MinimumRiskSupervisor(),
    )
    actor.start()
    try:
        assert actor.wait_for_version(1, timeout_s=1.0)
        version, _ = actor.snapshot()
        receipt = actor.submit(
            Action(tool_calls=[ToolCall(name="dispatch")]),
            action_id="action-7",
            decision_id="decision-3",
            turn_id="turn-5",
            based_on_state_version=version,
            valid_from_tick=version,
            expires_at_tick=version + 1,
            supersedes_action_id="action-6",
            idempotency_key="episode/action-7",
        ).result(timeout=1.0)

        assert receipt.status in {"applied", "effected", "no_effect"}
        assert receipt.decision_id == "decision-3"
        assert receipt.turn_id == "turn-5"
        assert receipt.valid_from_tick == version
        assert receipt.supersedes_action_id == "action-6"
        assert receipt.idempotency_key == "episode/action-7"
        assert env.applied_names[0] == "minimum_risk_hold"
        first_transition = actor.transition_records()[0]
        assert first_transition["action_source"] == "safety_supervisor"
        assert first_transition["safety_decision"]["mode"] == "minimum_risk_fallback"
        lifecycle = [
            row["status"]
            for row in actor.lifecycle_records()
            if row["action_id"] == "action-7"
        ]
        assert lifecycle[:2] == ["queued", "accepted"]
        assert lifecycle[-1] in {"applied", "effected", "no_effect"}
    finally:
        actor.stop()


def test_duplicate_idempotency_key_is_rejected_without_execution() -> None:
    env = _ClockEnvironment(horizon=5)
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.04)
    actor.start()
    try:
        version, _ = actor.snapshot()
        first = actor.submit(
            Action(tool_calls=[ToolCall(name="first")]),
            action_id="first",
            decision_id="d1",
            turn_id="t1",
            based_on_state_version=version,
            idempotency_key="same-key",
        )
        duplicate = actor.submit(
            Action(tool_calls=[ToolCall(name="duplicate")]),
            action_id="duplicate",
            decision_id="d2",
            turn_id="t2",
            based_on_state_version=version,
            idempotency_key="same-key",
        )

        assert duplicate.result(timeout=1.0).reason_code == "DUPLICATE_IDEMPOTENCY_KEY"
        assert first.result(timeout=1.0).status in {"applied", "effected", "no_effect"}
        assert "duplicate" not in env.applied_names
    finally:
        actor.stop()


def test_safety_supervisor_failure_stops_fail_closed_without_detail_leak() -> None:
    env = _ClockEnvironment()
    actor = RealtimeEnvironmentActor(
        env,
        tick_interval_s=0.01,
        safety_supervisor=_FailingSupervisor(),
    )
    actor.start()
    assert actor.wait_until_done(timeout_s=1.0)
    actor.stop()

    assert env.tick == 0
    transition = actor.transition_records()[0]
    assert transition["safety_supervisor_failed"] is True
    assert transition["safety_error_type"] == "RuntimeError"
    assert "private shield failure" not in str(transition)


def test_accepted_action_future_cannot_be_canceled_while_environment_executes() -> None:
    env = _BlockingEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.01)
    actor.start()
    future = actor.submit_current(
        Action(tool_calls=[ToolCall(name="committed")]),
        action_id="committed",
        decision_id="decision-accepted",
        turn_id="turn-accepted",
    )
    assert env.entered_step.wait(timeout=1.0)

    assert future.cancel() is False
    env.release_step.set()
    receipt = future.result(timeout=1.0)
    actor.stop()

    assert receipt.status == "no_effect"
    assert env.applied_names == ["committed"]
    assert [
        row["status"]
        for row in actor.lifecycle_records()
        if row["action_id"] == "committed"
    ] == ["queued", "accepted", "applied", "no_effect"]


def test_all_failed_tool_calls_are_not_labeled_applied() -> None:
    class RejectedToolEnvironment(_ClockEnvironment):
        def step(self, action: Action) -> StepReturn:
            call = action.tool_calls[0]
            self.applied_names.append(call.name)
            self.tick += 1
            return StepReturn(
                observation=self.snapshot(),
                tool_results=[
                    ToolResult(
                        name=call.name,
                        ok=False,
                        error_code="DOMAIN_REJECTED",
                        state_changing=True,
                        call_id=call.call_id,
                    )
                ],
                reward=0.0,
                done=True,
                info=StepInfo(),
            )

    actor = RealtimeEnvironmentActor(
        RejectedToolEnvironment(horizon=1), tick_interval_s=0.01
    )
    actor.start()
    receipt = actor.submit_current(
        Action(tool_calls=[ToolCall(name="unsafe", call_id="unsafe-call")]),
        action_id="unsafe-action",
    ).result(timeout=1.0)
    actor.stop()

    assert receipt.status == "no_effect"
    assert receipt.applied_tick is None
    assert [
        row["status"]
        for row in actor.lifecycle_records()
        if row["action_id"] == "unsafe-action"
    ] == ["queued", "accepted", "no_effect"]


def test_receipt_callback_never_runs_under_actor_condition_lock() -> None:
    env = _BlockingEnvironment()
    actor = RealtimeEnvironmentActor(env, tick_interval_s=0.01)
    actor.start()
    future = actor.submit_current(
        Action(tool_calls=[ToolCall(name="committed")]),
        action_id="committed",
    )
    callback_lock = threading.Lock()
    callback_entered = threading.Event()
    callback_finished = threading.Event()
    snapshot_finished = threading.Event()

    def receipt_callback(_: object) -> None:
        callback_entered.set()
        with callback_lock:
            callback_finished.set()

    def take_snapshot() -> None:
        actor.snapshot()
        snapshot_finished.set()

    future.add_done_callback(receipt_callback)
    assert env.entered_step.wait(timeout=1.0)
    try:
        with callback_lock:
            env.release_step.set()
            assert callback_entered.wait(timeout=1.0)
            snapshot_thread = threading.Thread(target=take_snapshot)
            snapshot_thread.start()
            assert snapshot_finished.wait(timeout=0.5)
        snapshot_thread.join(timeout=1.0)
        assert callback_finished.wait(timeout=1.0)
    finally:
        env.release_step.set()
        actor.stop()


def test_state_changing_ack_requires_matching_agent_caused_evidence_for_effect() -> None:
    class EvidenceEffectEnvironment(_ClockEnvironment):
        def __init__(
            self,
            *,
            event_call_id: str | None,
            include_evidence: bool,
            requested_args: dict | None = None,
            after_digest: str = "after",
            evidence_tick: int = 0,
        ) -> None:
            super().__init__(horizon=1)
            self.event_call_id = event_call_id
            self.include_evidence = include_evidence
            self.requested_args = requested_args
            self.after_digest = after_digest
            self.evidence_tick = evidence_tick
            self.evidence = EvidenceLogger("realtime-effect")

        def step(self, action: Action) -> StepReturn:
            call = action.tool_calls[0]
            evidence_ids = []
            event = {
                "event_id": "native-effect",
                "origin": "agent_caused",
                "agent_caused": True,
                "call_id": self.event_call_id,
                "tool_name": call.name,
                "requested_action": {
                    "name": call.name,
                    "args": (
                        dict(call.args)
                        if self.requested_args is None
                        else dict(self.requested_args)
                    ),
                },
                "before_state_digest": "before",
                "after_state_digest": self.after_digest,
            }
            if self.include_evidence:
                evidence_ids.append(
                    self.evidence.log(
                        "realized_event",
                        tick=self.evidence_tick,
                        payload=event,
                        source="engine",
                    )
                )
            self.tick += 1
            return StepReturn(
                observation=self.snapshot(),
                tool_results=[
                    ToolResult(
                        name=call.name,
                        ok=True,
                        state_changing=True,
                        call_id=call.call_id,
                    )
                ],
                reward=0.0,
                done=True,
                info=StepInfo(
                    realized_events=[
                        {
                            **event,
                            "evidence_ids": evidence_ids,
                        }
                    ]
                ),
            )

    def run_case(
        *,
        event_call_id: str | None,
        include_evidence: bool,
        requested_args: dict | None = None,
        after_digest: str = "after",
        evidence_tick: int = 0,
    ) -> tuple:
        actor = RealtimeEnvironmentActor(
            EvidenceEffectEnvironment(
                event_call_id=event_call_id,
                include_evidence=include_evidence,
                requested_args=requested_args,
                after_digest=after_digest,
                evidence_tick=evidence_tick,
            ),
            tick_interval_s=0.01,
        )
        actor.start()
        receipt = actor.submit_current(
            Action(
                tool_calls=[
                    ToolCall(name="control", call_id="call-control")
                ]
            ),
            action_id="action-control",
            decision_id="decision-control",
            turn_id="turn-control",
        ).result(timeout=1.0)
        transition = actor.transition_records()[0]
        actor.stop()
        return receipt, transition

    confirmed, confirmed_transition = run_case(
        event_call_id="different-call",
        include_evidence=True,
    )
    unproven, unproven_transition = run_case(
        event_call_id="call-control",
        include_evidence=False,
    )
    effected, effected_transition = run_case(
        event_call_id="call-control",
        include_evidence=True,
    )
    wrong_args, wrong_args_transition = run_case(
        event_call_id="call-control",
        include_evidence=True,
        requested_args={"target": "other"},
    )
    unchanged, unchanged_transition = run_case(
        event_call_id="call-control",
        include_evidence=True,
        after_digest="before",
    )
    future_tick, future_tick_transition = run_case(
        event_call_id="call-control",
        include_evidence=True,
        evidence_tick=2,
    )

    assert confirmed.status == "confirmed"
    assert confirmed_transition["effect_observed"] is False
    assert unproven.status == "confirmed"
    assert unproven_transition["effect_observed"] is False
    assert effected.status == "effected"
    assert effected_transition["effect_observed"] is True
    assert effected_transition["tool_trace_edges"][0]["effect_proven"] is True
    assert effected_transition["effect_evidence_ids"]
    for receipt, transition in (
        (wrong_args, wrong_args_transition),
        (unchanged, unchanged_transition),
        (future_tick, future_tick_transition),
    ):
        assert receipt.status == "confirmed"
        assert transition["effect_observed"] is False


def test_state_changing_ack_rejects_non_realized_effect_evidence() -> None:
    class UnprovenEffectEnvironment(_ClockEnvironment):
        def __init__(self) -> None:
            super().__init__(horizon=1)
            self.evidence = EvidenceLogger("realtime-unproven-effect")

        def step(self, action: Action) -> StepReturn:
            call = action.tool_calls[0]
            evidence_id = self.evidence.log(
                "observation",
                tick=0,
                payload={"call_id": call.call_id},
            )
            self.tick += 1
            return StepReturn(
                observation=self.snapshot(),
                tool_results=[
                    ToolResult(
                        name=call.name,
                        ok=True,
                        state_changing=True,
                        call_id=call.call_id,
                    )
                ],
                reward=0.0,
                done=True,
                info=StepInfo(
                    realized_events=[
                        {
                            "event_id": "unproven-effect",
                            "origin": "agent_caused",
                            "call_id": call.call_id,
                            "evidence_ids": [evidence_id],
                        }
                    ]
                ),
            )

    actor = RealtimeEnvironmentActor(
        UnprovenEffectEnvironment(), tick_interval_s=0.01
    )
    actor.start()
    receipt = actor.submit_current(
        Action(tool_calls=[ToolCall(name="control", call_id="call-control")]),
        action_id="action-control",
    ).result(timeout=1.0)
    transition = actor.transition_records()[0]
    actor.stop()

    assert receipt.status == "confirmed"
    assert transition["effect_observed"] is False
    assert transition["effect_evidence_ids"] == []
