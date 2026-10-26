"""Terminal safety state must not be confused with missing model decisions."""

import pytest

from core import Action, ToolCall
from runner.episode import _run_episode_loop
from dataclasses import dataclass, field
from typing import Any

from core import EvidenceLogger
from core.pomdp import StepInfo, StepReturn, ToolResult


@dataclass
class FakeEnv:
    """Minimal POMDPEnvironment subset used by ``_run_episode_loop``."""

    rewards: list[float]
    tool_results_per_tick: list[list[ToolResult]] = field(default_factory=list)
    realized_per_tick: list[list[dict[str, Any]]] = field(default_factory=list)
    evidence_per_tick: list[list[str]] = field(default_factory=list)
    forecast_per_tick: list[dict[str, Any]] = field(default_factory=list)
    warnings_per_tick: list[list[str]] = field(default_factory=list)
    done_at_tick: int | None = None  # 0-indexed
    snapshot_payload: dict[str, Any] = field(default_factory=dict)
    declare_periodic_opportunity: bool = True
    tool_specs: list[dict[str, Any]] = field(default_factory=list)
    evidence: EvidenceLogger | None = None

    horizon: int = 0
    tick: int = 0

    def __post_init__(self) -> None:
        self.horizon = len(self.rewards)
        if self.evidence is None:
            self.evidence = EvidenceLogger("fake_runner")
        # default empty lists for sidecars if caller didn't pass them
        n = self.horizon
        if not self.tool_results_per_tick:
            self.tool_results_per_tick = [[] for _ in range(n)]
        if not self.realized_per_tick:
            self.realized_per_tick = [[] for _ in range(n)]
        if not self.evidence_per_tick:
            self.evidence_per_tick = [[] for _ in range(n)]
        if not self.forecast_per_tick:
            self.forecast_per_tick = [{} for _ in range(n)]
        if not self.warnings_per_tick:
            self.warnings_per_tick = [[] for _ in range(n)]

    def snapshot(self) -> dict[str, Any]:
        payload = dict(self.snapshot_payload)
        if self.declare_periodic_opportunity:
            payload.setdefault("decision_opportunity", True)
            payload.setdefault(
                "decision_cadence",
                {
                    "periodic_scan_every_ticks": 4,
                    "max_review_after_ticks": 4,
                },
            )
        return payload

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return list(self.tool_specs)

    def step(self, _action: Action) -> StepReturn:
        i = self.tick
        observation = {"tick": i + 1, **self.snapshot_payload}
        if self.declare_periodic_opportunity:
            observation.setdefault("decision_opportunity", True)
            observation.setdefault(
                "decision_cadence",
                {
                    "periodic_scan_every_ticks": 4,
                    "max_review_after_ticks": 4,
                },
            )
        ret = StepReturn(
            observation=observation,
            tool_results=list(self.tool_results_per_tick[i]),
            reward=self.rewards[i],
            done=(self.done_at_tick is not None and i >= self.done_at_tick),
            info=StepInfo(
                realized_events=list(self.realized_per_tick[i]),
                evidence_ids=list(self.evidence_per_tick[i]),
                forecast_updates=dict(self.forecast_per_tick[i]),
                early_stop_warnings=list(self.warnings_per_tick[i]),
            ),
        )
        self.tick += 1
        return ret



class CaptureAgent:
    """Captures every observation it sees and what ``on_episode_end`` got."""

    def __init__(self) -> None:
        self.observations_seen: list[dict[str, Any]] = []
        self.episode_end_call: dict[str, Any] | None = None

    def reset(self, *_a: Any, **_kw: Any) -> None:  # pragma: no cover - unused here
        pass

    def act(
        self, observation: dict[str, Any], _tool_specs: list[dict[str, Any]]
    ) -> Action:
        # IMPORTANT: copy so later mutations to obs don't retroactively
        # change what we recorded.
        self.observations_seen.append(dict(observation))
        return Action(tool_calls=[ToolCall(name="wait")])

    def on_episode_end(
        self,
        final_observation: dict[str, Any],
        actions: list[Action],
        episode_reward: float = 0.0,
    ) -> None:
        self.episode_end_call = {
            "final_observation": dict(final_observation),
            "n_actions": len(actions),
            "episode_reward": float(episode_reward),
        }



class PersistentCaptureAgent(CaptureAgent):
    def _uses_persistent_session(self) -> bool:
        return True



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
