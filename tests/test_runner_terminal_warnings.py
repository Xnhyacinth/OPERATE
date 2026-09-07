"""Terminal safety state must not be confused with missing model decisions."""

import pytest

from core import Action, ToolCall
from runner.episode import _run_episode_loop
from tests.test_runner import FakeEnv, PersistentCaptureAgent


@pytest.mark.parametrize("done_at_tick", [None, 2])
def test_answered_persistent_terminal_warning_is_a_task_outcome(done_at_tick):
    warnings = ["runtime_assurance_intervention", "minimal_risk_maneuver_active"]
    env = FakeEnv(
        rewards=[0.0] * 3, warnings_per_tick=[warnings] * 3, done_at_tick=done_at_tick
    )
    agent = PersistentCaptureAgent()
    result = _run_episode_loop(env=env, agent=agent, logger=None)
    assert len(agent.observations_seen) == 3
    assert result["final_observation"]["__last_early_stop_warnings__"] == warnings
    assert result["terminal_integrity"]["release_ready"] is True
    assert result["terminal_integrity"]["unanswered_interrupt_reasons"] == []


@pytest.mark.parametrize(
    "warnings",
    [
        [[], [], ["new_hazard"]],
        [["old_hazard"], ["old_hazard"], ["old_hazard", "new_hazard"]],
        [["old_hazard"], [], ["old_hazard"]],
    ],
)
def test_new_or_reappearing_terminal_warning_remains_unanswered(warnings):
    env = FakeEnv(rewards=[0.0] * 3, warnings_per_tick=warnings)
    result = _run_episode_loop(env=env, agent=PersistentCaptureAgent(), logger=None)
    assert result["terminal_integrity"]["release_ready"] is False
    assert result["terminal_integrity"]["unanswered_interrupt_reasons"] == [
        "safety_warning"
    ]


def test_terminal_typed_safety_event_not_suppressed_by_answered_warning():
    env = FakeEnv(
        rewards=[0.0] * 2,
        warnings_per_tick=[["MRM"]] * 2,
        realized_per_tick=[
            [],
            [{"type": "new_collision_risk", "event_class": "safety"}],
        ],
    )
    result = _run_episode_loop(env=env, agent=PersistentCaptureAgent(), logger=None)
    assert result["terminal_integrity"]["release_ready"] is False
    assert result["terminal_integrity"]["unanswered_interrupt_reasons"] == [
        "safety_warning"
    ]


def test_provider_failure_does_not_count_as_answering_terminal_warning():
    class FailingAgent(PersistentCaptureAgent):
        def act(self, observation, tool_specs):
            super().act(observation, tool_specs)
            return Action(tool_calls=[ToolCall(name="wait")])

        def get_last_provider_outcome(self):
            return {"status": "failed"}

    env = FakeEnv(rewards=[0.0] * 2, warnings_per_tick=[["MRM"]] * 2)
    result = _run_episode_loop(env=env, agent=FailingAgent(), logger=None)
    assert result["terminal_integrity"]["release_ready"] is False
    assert result["terminal_integrity"]["unanswered_interrupt_reasons"] == [
        "provider_retry",
        "safety_warning",
    ]


def test_invalid_successful_provider_response_does_not_answer_terminal_warning():
    class InvalidAgent(PersistentCaptureAgent):
        def act(self, observation, tool_specs):
            super().act(observation, tool_specs)
            return Action(dominant="protocol_repair_no_tool_call")

        def get_last_provider_outcome(self):
            return {"status": "success"}

    env = FakeEnv(rewards=[0.0] * 2, warnings_per_tick=[["MRM"]] * 2)
    result = _run_episode_loop(env=env, agent=InvalidAgent(), logger=None)
    assert result["terminal_integrity"]["release_ready"] is False
    assert result["terminal_integrity"]["unanswered_interrupt_reasons"] == [
        "safety_warning"
    ]
    assert result["event_adaptive_autonomy"]["provider_failure_count"] == 0


def test_seen_warning_invalid_model_response_is_complete_collection():
    class InvalidAgent(PersistentCaptureAgent):
        def act(self, observation, tool_specs):
            super().act(observation, tool_specs)
            return Action(dominant="protocol_repair_no_tool_call")

        def get_last_provider_outcome(self):
            return {"status": "success"}

    env = FakeEnv(rewards=[0.0] * 2, warnings_per_tick=[["MRM"]] * 2)
    result = _run_episode_loop(env=env, agent=InvalidAgent(), logger=None)
    terminal = result["terminal_integrity"]
    assert terminal["release_ready"] is False
    assert terminal["unanswered_interrupt_reasons"] == ["safety_warning"]
    assert terminal["answered_persistent_warnings"] == []
    assert terminal["collection_complete"] is True
    assert (
        terminal["terminal_disposition"] == "observed_invalid_terminal_model_response"
    )
    assert terminal["model_response_failure"]["presented_warnings"] == ["MRM"]


def test_invalid_response_does_not_close_new_terminal_warning_collection():
    class InvalidAgent(PersistentCaptureAgent):
        def act(self, observation, tool_specs):
            super().act(observation, tool_specs)
            return Action(dominant="protocol_repair_no_tool_call")

        def get_last_provider_outcome(self):
            return {"status": "success"}

    env = FakeEnv(rewards=[0.0] * 2, warnings_per_tick=[["MRM"], ["MRM", "new"]])
    result = _run_episode_loop(env=env, agent=InvalidAgent(), logger=None)
    assert result["terminal_integrity"]["collection_complete"] is False
