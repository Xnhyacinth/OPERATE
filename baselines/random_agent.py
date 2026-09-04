"""
baselines.random_agent — Random valid tool calls.

Noise-floor baseline. Picks a random tool each tick with reasonable
argument defaults. Deterministic given ``seed``.
"""

from __future__ import annotations

import random
from typing import Any

from core import Action, ToolCall
from core.pomdp_env import POMDPEnvironment

from .base import BaselineAgent


class RandomAgent(BaselineAgent):
    name = "random"

    def __init__(self) -> None:
        self._rng: random.Random | None = None
        self._tools: list[str] = []
        self._gen_ids: list[str] = []
        self._load_ids: list[str] = []
        self._tick = 0
        self._last_observation: dict[str, Any] = {}

    def reset(
        self, env: POMDPEnvironment, scenario_config: dict[str, Any], seed: int
    ) -> None:
        self._rng = random.Random(seed + 7919)
        self._tick = 0
        self._reset_idem_seq()
        self._last_observation = {}
        self._tools = [spec["function"]["name"] for spec in env.get_tool_specs()]
        obs = env.snapshot()
        entities = obs.get("entities", {})
        self._gen_ids = [
            eid for eid, e in entities.items() if e.get("kind") == "generator"
        ]
        self._load_ids = [eid for eid, e in entities.items() if e.get("kind") == "load"]

    def act(
        self, observation: dict[str, Any], tool_specs: list[dict[str, Any]]
    ) -> Action:
        assert self._rng is not None
        self._tick += 1
        self._last_observation = dict(observation or {})
        name = self._rng.choice(self._tools)
        args = self._random_args_for(name)
        return Action(
            tool_calls=[
                ToolCall(
                    name=name, args=args, idempotency_key=self._next_idem_key("rand")
                )
            ],
            dominant=name,
        )

    def _random_args_for(self, name: str) -> dict[str, Any]:
        assert self._rng is not None
        if name == "redispatch_generation" and self._gen_ids:
            return {
                "generator_id": self._rng.choice(self._gen_ids),
                "target_mw": float(self._rng.randint(0, 200)),
                "commit": True,
            }
        if name == "shed_load" and self._load_ids:
            return {
                "load_id": self._rng.choice(self._load_ids),
                "mw": float(self._rng.randint(0, 50)),
                "reason": "random",
            }
        if name == "commit_reserve":
            return {"mw": float(self._rng.randint(0, 100))}
        if name == "switch_branch":
            return {
                "line_index": self._rng.randint(0, 19),
                "connect": bool(self._rng.randint(0, 1)),
            }
        if name == "forecast_query":
            return {"horizon": self._rng.choice([2, 4, 6])}
        if name == "query_chronics_window":
            return {"window": self._rng.choice([1, 3, 6])}
        if name == "investigate_substation" and self._gen_ids:
            return {"target_id": self._rng.choice(self._gen_ids)}
        if name == "stakeholder_query":
            return {
                "group": self._rng.choice(["hospital", "residential", "industrial"])
            }
        if name == "request_mutual_aid":
            return {"neighbor": "iso_east", "mw": float(self._rng.randint(0, 100))}
        if name == "negotiate_with_stakeholder":
            return {"group": "residential", "offer": "5% curtailment for credit"}
        if name == "moral_choice":
            return {
                "dilemma_id": "anon",
                "option_id": "shed_residential",
                "rationale": "random baseline",
            }
        if name == "commit_to_plan":
            return {
                "plan_id": f"p_{self._tick}",
                "horizon_ticks": 3,
                "rationale": "random baseline",
                "predicted_events": [],
            }
        if name == "place_replenishment_order":
            capacity = self._last_observation.get("supply_capacity") or [10]
            try:
                cap = max(1, int(float(capacity[0])))
            except (TypeError, ValueError, IndexError):
                cap = 10
            return {"quantity": int(self._rng.randint(1, cap)), "stage": 0}
        return {}
