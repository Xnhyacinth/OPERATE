"""
core.pomdp — Backend-agnostic POMDP contract for OPERATE.

Formalization (U, S, A, O, T, R):

    U : agent / decision-maker capabilities (tool surface, budget, role)
    S : full ground-truth world state held by the simulator
    A : action space (typed tool calls produced by the agent each tick)
    O : observation space (partial snapshot of S, filtered by FogOfWar)
    T : state transition (computed by the real simulator backend)
    R : reward / scoring (evidence-linked, computed post-hoc by core.evidence
        and evaluation.scorer; the env may emit per-tick reward signals but
        the official benchmark score is the multi-dim evaluation, not the
        Grid2Op / Gym reward).

This module DOES NOT depend on Grid2Op, SUMO, RCRS, pandapower, or any
specific backend. Each domain implements ``POMDPEnvironment`` for its
simulator.

Forked-and-refactored from
``dispatch-benchmark/engine/pomdp_formalization.py`` so it can host multiple
backends without inheriting any emergency-domain assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """A single typed tool call produced by the agent within one tick.

    Multiple ``ToolCall`` objects may be grouped into a single ``Action``
    because real dispatch centers issue many parallel commands per tick.
    """

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
    rationale: str | None = None
    call_id: str | None = None
    consumes_evidence_ids: list[str] | None = None
    depends_on_call_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "args": dict(self.args),
            "idempotency_key": self.idempotency_key,
            "rationale": self.rationale,
            "call_id": self.call_id,
            "consumes_evidence_ids": self.consumes_evidence_ids,
            "depends_on_call_ids": self.depends_on_call_ids,
        }


@dataclass
class Action:
    """An action this tick is a (possibly empty) sequence of tool calls."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    dominant: str | None = None  # for trajectory diagnostics
    assistant_text: str | None = None  # raw text the LLM emitted
    # P1-2: free-form reasoning accompanying the action. Populated by every
    # LLM provider path (``""`` when the provider exposes no separate
    # reasoning field). Never ``None`` so failure-recipe mining can treat it
    # as a present (possibly empty) string.
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [c.to_dict() for c in self.tool_calls],
            "dominant_action": self.dominant,
            "assistant_text": self.assistant_text,
            "rationale": self.rationale,
        }

    @property
    def is_noop(self) -> bool:
        return not self.tool_calls or all(
            c.name in {"wait", "noop"} for c in self.tool_calls
        )


@dataclass
class ToolResult:
    """Outcome of executing one ``ToolCall``.

    Carries enough metadata for both the agent's next-turn observation and
    the evidence logger / scorer.
    """

    name: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    latency_ticks: int = 0  # if delayed, when result will materialize
    idempotency_key: str | None = None
    state_changing: bool = False
    evidence_id: str | None = None  # links back to evidence.EvidenceLogger
    call_id: str | None = None  # explicit producing action edge
    # Explicit causal metadata is populated by the episode runner after a
    # backend step.  Keeping it on the result (rather than only in an
    # adapter-specific payload) makes dependency-graph reconstruction
    # backend-independent while remaining backward compatible with existing
    # tool handlers.
    consumes_evidence_ids: list[str] | None = None
    produces_evidence_ids: list[str] | None = None
    depends_on_call_ids: list[str] | None = None
    effect_tick: int | None = None
    cost_units: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": self.ok,
            "payload": self.payload,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "latency_ticks": self.latency_ticks,
            "idempotency_key": self.idempotency_key,
            "state_changing": self.state_changing,
            "evidence_id": self.evidence_id,
            "call_id": self.call_id,
            "consumes_evidence_ids": self.consumes_evidence_ids,
            "produces_evidence_ids": self.produces_evidence_ids,
            "depends_on_call_ids": self.depends_on_call_ids,
            "effect_tick": self.effect_tick,
            "cost_units": float(self.cost_units),
        }


@dataclass
class StepInfo:
    """Extra side-channel returned by ``POMDPEnvironment.step``."""

    realized_events: list[dict[str, Any]] = field(default_factory=list)
    fault_injections: list[dict[str, Any]] = field(default_factory=list)
    forecast_updates: dict[str, Any] = field(default_factory=dict)
    early_stop_warnings: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "realized_events": self.realized_events,
            "fault_injections": self.fault_injections,
            "forecast_updates": self.forecast_updates,
            "early_stop_warnings": self.early_stop_warnings,
            "evidence_ids": self.evidence_ids,
            "extra": self.extra,
        }


@dataclass
class StepReturn:
    """Bundle returned by ``POMDPEnvironment.step``.

    Per-tick reward is informative ONLY — the official benchmark score is
    the multi-dim evaluator that runs over the full trajectory.
    """

    observation: dict[str, Any]
    tool_results: list[ToolResult]
    reward: float
    done: bool
    info: StepInfo


# ─────────────────────────────────────────────────────────────────────────────
# Tick budget
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TickBudget:
    """Per-tick budget for tool calls, to prevent tool-call storms.

    Mitigates the wall-timeout pathology observed in dispatch-benchmark's
    gpt-5-2025-08-07 reference runs (22/88 wall_timeout).
    """

    max_tool_calls_per_tick: int = 8
    max_total_tool_calls: int | None = None  # episode-wide hard cap; None → computed default
    max_cost_units_per_tick: float | None = None
    duplicate_suppression_window: int = 3  # ticks
    cooldown_after_failure: int = 1
    horizon: int | None = None  # scenario tick horizon; used to compute the default cap

    def __post_init__(self) -> None:
        # P1-1: bound tool storms. When unset, cap = per_tick * horizon * 1.5
        # (1.5x headroom over a steady max-calls/tick agent). If horizon is
        # unknown, fall back to a generous absolute cap so storms are always
        # bounded — never None.
        if self.max_total_tool_calls is None:
            if self.horizon is not None and self.horizon > 0:
                self.max_total_tool_calls = int(self.max_tool_calls_per_tick * self.horizon * 1.5)
            else:
                self.max_total_tool_calls = self.max_tool_calls_per_tick * 48  # fallback: 8*48=384
