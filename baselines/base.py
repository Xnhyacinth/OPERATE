"""
baselines.base — Common Agent contract for OPERATE baselines.

Every baseline implements:

    def reset(self, env: POMDPEnvironment, scenario_config: dict, seed: int) -> None
    def act(self, observation: dict, tool_specs: list[dict]) -> Action

This matches the runner's loop in ``run.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core import Action
from core.pomdp_env import POMDPEnvironment


class BaselineAgent(ABC):
    """Minimal agent interface."""

    name: str = "abstract"

    @abstractmethod
    def reset(
        self, env: POMDPEnvironment, scenario_config: dict[str, Any], seed: int
    ) -> None: ...

    @abstractmethod
    def act(
        self, observation: dict[str, Any], tool_specs: list[dict[str, Any]]
    ) -> Action: ...

    def deliberate(
        self,
        observation: dict[str, Any],
        tool_specs: list[dict[str, Any]],
        *,
        round_index: int,
        n_rounds: int,
        previous_drafts: list[dict[str, Any]],
    ) -> Action:
        """Optional draft-only multi-turn hook.

        The runner may call this before ``act`` when multi-turn
        deliberation is explicitly enabled. Draft actions are never
        executed against the environment; they are surfaced back to the
        final ``act`` call under ``observation['__multi_turn_drafts__']``.
        Default behavior is a no-op draft so legacy agents stay compatible.
        """
        return Action(
            tool_calls=[],
            dominant="deliberate",
            assistant_text=(
                f"draft round {round_index}/{n_rounds}; "
                f"previous_drafts={len(previous_drafts)}"
            ),
        )

    def investigate(
        self, observation: dict[str, Any], tool_specs: list[dict[str, Any]]
    ) -> Action:
        """Optional read-only stage before the tick's committed action."""
        return Action(tool_calls=[], dominant="investigate")

    def _reset_idem_seq(self) -> None:
        """Reset the per-episode idempotency-key sequence counter.

        Every ``reset()`` implementation should call this alongside
        resetting ``self._tick`` so idempotency keys stay unique for the
        life of the episode. A tick- or loop-index-only key (e.g.
        ``f"wait_{self._tick}"`` or ``f"llm_{self._tick}_{i}"``) collides
        if the same code path fires twice for one nominal tick — e.g. a
        provider-error retry, a fallback branch, or a future multi-turn
        deliberation round re-entering the same tick. A monotonic
        counter that only resets once per episode cannot collide.
        """
        self._idem_seq = 0

    def _next_idem_key(self, prefix: str) -> str:
        """Return a per-episode-unique idempotency key.

        Embeds the current tick (for readability in logs/trajectories)
        plus a monotonic sequence number that never resets except via
        ``_reset_idem_seq()``. This guarantees uniqueness across however
        many times a code path fires within one episode, instead of
        relying on a tick value or loop-local index that can repeat.
        """
        self._idem_seq = getattr(self, "_idem_seq", 0) + 1
        tick = getattr(self, "_tick", 0)
        return f"{prefix}_t{tick}_n{self._idem_seq}"

    def on_episode_end(
        self,
        final_observation: dict[str, Any],
        actions: list[Action],
        episode_reward: float = 0.0,
    ) -> None:
        """Optional Reflexion-style episode-end hook.

        Called by ``run.py`` after the loop terminates. Default no-op so
        existing agents are unaffected. Override to:
        - Inspect what happened (`final_observation['__last_*__']` keys).
        - Update episodic memory / reflection notes carried into the
          next episode (handled per-agent, not by the runner).
        - Persist lessons to disk (e.g. Reflexion-style memory file).

        The runner catches and logs exceptions from this hook so a
        flawed reflection cannot fail the episode.
        """
        return None

    def close(self) -> None:
        """Optional cleanup hook."""
        return None
