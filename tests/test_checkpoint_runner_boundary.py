from types import SimpleNamespace

import pytest

from core import Action
from core.pomdp import StepInfo, StepReturn
from runner.checkpoint import CheckpointIntegrityError
from runner.episode import _run_episode_loop


@pytest.mark.parametrize("hook", ["observe_transition", "on_episode_end"])
def test_final_tick_checkpoint_divergence_is_never_downgraded_to_warning(hook):
    env = SimpleNamespace(horizon=1, tick=0, evidence=None,
                          snapshot=lambda: {"tick": 0}, get_tool_specs=lambda: [])

    def step(action):
        env.tick = 1
        return StepReturn(observation={"tick": 1}, tool_results=[], reward=0.0,
                          done=True, info=StepInfo())

    def diverged(*args, **kwargs):
        raise CheckpointIntegrityError("terminal boundary differs")

    env.step = step
    agent = SimpleNamespace(act=lambda *args: Action())
    setattr(agent, hook, diverged)
    with pytest.raises(CheckpointIntegrityError, match="terminal boundary differs"):
        _run_episode_loop(env=env, agent=agent, logger=None)
