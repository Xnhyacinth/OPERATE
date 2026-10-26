"""
baselines.wait_only — Emit only ``wait`` every tick.

Lower-bound baseline: quantifies the cost of doing nothing. Counterfactual
``prevented_loss`` is reported relative to this agent's policy by default.
"""

from __future__ import annotations

from typing import Any

from core import Action, ToolCall
from core.pomdp_env import POMDPEnvironment

from .base import BaselineAgent


class WaitOnlyAgent(BaselineAgent):
    name = "wait_only"

    def __init__(self) -> None:
        self._tick = 0

    def reset(
        self, env: POMDPEnvironment, scenario_config: dict[str, Any], seed: int
    ) -> None:
        self._tick = 0
        self._reset_idem_seq()

    def act(
        self, observation: dict[str, Any], tool_specs: list[dict[str, Any]]
    ) -> Action:
        self._tick += 1
        return Action(
            tool_calls=[
                ToolCall(name="wait", idempotency_key=self._next_idem_key("wait"))
            ],
            dominant="wait",
        )
