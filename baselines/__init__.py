"""Baseline agents — noise floor and oracle upper bound for the leaderboard."""

from __future__ import annotations

from .base import BaselineAgent
from .greedy_heuristic import GreedyHeuristicAgent
from .llm_agent import LLMAgent, LLMConfig
from .operational_agency_positive_control import OperationalAgencyPositiveControlAgent
from .oracle_offline import OracleOfflineAgent
from .random_agent import RandomAgent
from .react_agent import ReActLLMAgent
from .reflexion_agent import ReflexionLLMAgent
from .wait_only import WaitOnlyAgent

REGISTRY: dict[str, type[BaselineAgent]] = {
    WaitOnlyAgent.name: WaitOnlyAgent,
    RandomAgent.name: RandomAgent,
    GreedyHeuristicAgent.name: GreedyHeuristicAgent,
    OracleOfflineAgent.name: OracleOfflineAgent,
    OperationalAgencyPositiveControlAgent.name: OperationalAgencyPositiveControlAgent,
    LLMAgent.name: LLMAgent,
    ReActLLMAgent.name: ReActLLMAgent,
    ReflexionLLMAgent.name: ReflexionLLMAgent,
}


def make_agent(name: str, **kwargs) -> BaselineAgent:
    if name not in REGISTRY:
        raise ValueError(f"unknown agent {name}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = [
    "BaselineAgent",
    "GreedyHeuristicAgent",
    "LLMAgent",
    "LLMConfig",
    "OracleOfflineAgent",
    "OperationalAgencyPositiveControlAgent",
    "RandomAgent",
    "ReActLLMAgent",
    "ReflexionLLMAgent",
    "WaitOnlyAgent",
    "REGISTRY",
    "make_agent",
]
