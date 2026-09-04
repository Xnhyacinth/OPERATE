"""
baselines.react_agent — ReAct-style LLM agent for OPERATE.

Implements the **THINK → ACT → OBSERVE → THINK** loop within and across
ticks (v0.2.1). Distinguishing features vs the plain ``LLMAgent``:

- Reads the runner-provided reserved observation keys
  (``__last_tool_results__``, ``__last_realized_events__``,
  ``__last_reward__``, ``__last_evidence_ids__``) and surfaces them to
  the LLM as structured feedback on the previous tick.
- Maintains a per-episode **scratchpad** (chain-of-thought log) so the
  LLM can read back its own earlier reasoning.
- Tracks a longer rolling window (10 ticks vs 3 in plain LLMAgent) of
  recent (tool, result) pairs.
- Prompts the LLM to explicitly emit a ``THOUGHT:`` rationale before
  tool calls — captured in ``Action.assistant_text``.
- No env-API changes; falls back to wait_only without an API key.

ReAct paper reference: Yao et al., "ReAct: Synergizing Reasoning and
Acting in Language Models" (ICLR 2023).
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

from core import Action, ToolCall
from core.pomdp_env import POMDPEnvironment

from .base import BaselineAgent  # noqa: F401 — kept for back-compat
from .llm_agent import (  # noqa: F401
    DOMAIN_ROLES,
    LLMAgent,
    LLMConfig,
    ProviderCircuitOpenError,
    ProviderQuotaExhaustedError,
    _empty_interaction_stats,
    _event_prompt_view,
    _prompt_safe_entity,
    _with_dependency_metadata,
    build_visible_belief_summary,
    redact_provider_error,
)

LOGGER = logging.getLogger(__name__)


_REACT_SYSTEM_PROMPT_STRICT = """You are the real-time {domain_role} agent.

Use the visible observation and function schemas to choose up to {max_tools}
tool calls within a shared cost budget of {max_cost_units} units. Each function
schema states its resource cost. Some ticks contain a bounded `investigation`
stage followed by a `commit` stage. Investigation exposes read-only functions
and returns results without advancing simulator time; commit advances the
backend once. Both stages share one tool budget.

Output a concise decision rationale and tool calls through the function-calling
interface. When a call relies on visible evidence or a prior call, cite it with
the optional `_consumes_evidence_ids` and `_depends_on_call_ids` audit fields.
Do not invent entity, evidence, or call identifiers. Follow the current stage's
`allowed_tool_names` when it is present.

Scenario briefing (visible once):
- family       : {family}
- horizon      : {horizon_ticks} ticks × {tick_minutes} min
- tool budget  : {max_tools} tool calls per tick
- cost budget  : {max_cost_units} resource units per tick
"""

_REACT_SYSTEM_PROMPT_DEBUG = """You are the real-time {domain_role} ReAct agent.

Each tick you operate a THINK → ACT → OBSERVE loop:

  THINK : reason in 1-3 sentences about (a) what the previous tools
          returned, (b) what the current observation implies, (c) what
          you predict will happen if you do nothing.
  ACT   : choose up to {max_tools} tool calls. If you have a multi-tick
          plan, you MAY emit `commit_to_plan` with `predicted_events`
          listing the future events you expect (each item is an object
          {{event_type, target_id?, tick_offset>=1, confidence?}}).
  OBSERVE: the runner will surface the resulting tool_results and
          realized_events on the NEXT observation under the reserved
          keys `__last_tool_results__`, `__last_realized_events__`,
          `__last_evidence_ids__`, `__last_reward__`. Inspect them
          before your next THINK.

Output rules:
- Always start with a single line beginning `THOUGHT:` containing your
  THINK step. The runner records it as the assistant rationale.
- Then emit tool calls via the standard function-calling interface.
- It is acceptable to call only `wait` if the analysis says so, but
  every `wait`-tick you spend is one tick the environment is free to
  evolve without you.
- Avoid duplicate tool calls within the same tick.

Scenario briefing (visible once):
- family       : {family}
- mode         : {difficulty_mode}
- level        : {difficulty_level}
- horizon      : {horizon_ticks} ticks × {tick_minutes} min
- tool budget  : {max_tools} tool calls per tick
- cost budget  : {max_cost_units} resource units per tick
"""

# Back-compat re-export for callers that imported the old name.
_REACT_SYSTEM_PROMPT = _REACT_SYSTEM_PROMPT_STRICT


@dataclass
class _ReActMemory:
    """Per-episode scratchpad + rolling tool-result history."""

    scratchpad: list[str] = field(default_factory=list)  # one entry per tick
    recent_actions: list[dict[str, Any]] = field(default_factory=list)
    recent_results: list[dict[str, Any]] = field(default_factory=list)
    recent_evidence_ids: list[dict[str, Any]] = field(default_factory=list)
    realized_events: list[dict[str, Any]] = field(default_factory=list)


class ReActLLMAgent(LLMAgent):
    """ReAct-style LLM agent with intra-tick structured feedback."""

    name = "react_llm"
    RECENT_WINDOW = 10  # vs 3 in plain LLMAgent

    def __init__(self, config: LLMConfig | None = None) -> None:
        super().__init__(config=config)
        self._memory = _ReActMemory()
        self._stats = self._new_interaction_stats()

    @staticmethod
    def _new_interaction_stats() -> dict[str, Any]:
        stats = _empty_interaction_stats()
        stats["commit_to_plan_calls"] = 0
        return stats

    def reset(
        self, env: POMDPEnvironment, scenario_config: dict[str, Any], seed: int
    ) -> None:
        if self.config.interaction_mode != "logical_stateless":
            raise ValueError(
                "ReAct/Reflexion agents have not qualified the "
                "logical_persistent session contract; use llm_agent"
            )
        if (
            self.config.provider in {"openai", "azure", "openai_compatible"}
            and self._resolved_api_mode() != "chat_completions"
        ):
            raise ValueError(
                "ReAct/Reflexion agents currently support chat_completions "
                "only; use llm_agent for Responses API treatments"
            )
        prompt_mode = (
            getattr(self.config, "prompt_mode", "strict") or "strict"
        ).lower()
        if prompt_mode not in {"strict", "debug"}:
            raise ValueError(
                f"Invalid prompt_mode: {prompt_mode!r}. Must be 'strict' or 'debug'."
            )
        self._tick = 0
        self._consecutive_provider_failures = 0
        self._reset_idem_seq()
        self._tool_specs = _with_dependency_metadata(
            env.get_tool_specs(), required=False
        )
        self._readonly_tools = set(env.readonly_tool_names() or set()) - {
            "wait",
            "noop",
        }
        self._max_tools = env.budget.max_tool_calls_per_tick
        self._max_cost_units = env.budget.max_cost_units_per_tick
        self._memory = _ReActMemory()
        self._stats = self._new_interaction_stats()
        self._last_provider_outcome = {"status": "not_called"}
        domain = str(scenario_config.get("domain", "power_grid"))
        domain_role = DOMAIN_ROLES.get(domain, "decision support system")
        if prompt_mode == "strict":
            # D-01: publishable run, no difficulty leakage.
            self._system_prompt = _REACT_SYSTEM_PROMPT_STRICT.format(
                domain_role=domain_role,
                max_tools=self._max_tools,
                max_cost_units=self._max_cost_units,
                family=scenario_config.get("family", "unknown"),
                horizon_ticks=scenario_config.get("horizon_ticks", "?"),
                tick_minutes=scenario_config.get("tick_minutes", "?"),
            )
        else:
            self._system_prompt = _REACT_SYSTEM_PROMPT_DEBUG.format(
                domain_role=domain_role,
                max_tools=self._max_tools,
                max_cost_units=self._max_cost_units,
                family=scenario_config.get("family", "unknown"),
                difficulty_mode=scenario_config.get("difficulty_mode", "unknown"),
                difficulty_level=scenario_config.get("difficulty_level", "unknown"),
                horizon_ticks=scenario_config.get("horizon_ticks", "?"),
                tick_minutes=scenario_config.get("tick_minutes", "?"),
            )
        api_key = os.getenv(self.config.api_key_env)
        if not api_key:
            warnings.warn(
                f"{self.config.api_key_env} not set — ReActLLMAgent will "
                "fall back to wait_only.",
                stacklevel=2,
            )
            self._has_api_key = False
            return
        self._has_api_key = True
        self._client = self._make_client(api_key)

    def act(
        self, observation: dict[str, Any], tool_specs: list[dict[str, Any]]
    ) -> Action:
        self._tick += 1
        # Pull v0.2.1 reserved keys (set by run.py).
        last_results = observation.get("__last_tool_results__", []) or []
        last_realized = observation.get("__last_realized_events__", []) or []
        last_evidence_ids = observation.get("__last_evidence_ids__", []) or []
        last_reward = float(observation.get("__last_reward__", 0.0))
        if last_results:
            self._memory.recent_results.append(
                {"tick": self._tick - 1, "results": last_results}
            )
            self._memory.recent_results = self._memory.recent_results[
                -self.RECENT_WINDOW :
            ]
        if last_evidence_ids:
            self._memory.recent_evidence_ids.append(
                {"tick": self._tick - 1, "evidence_ids": list(last_evidence_ids)}
            )
            self._memory.recent_evidence_ids = self._memory.recent_evidence_ids[
                -self.RECENT_WINDOW :
            ]
        if last_realized:
            self._memory.realized_events.extend(
                {**ev, "observed_at_tick": self._tick} for ev in last_realized
            )
        if not self._has_api_key:
            self._last_provider_outcome = {"status": "not_called"}
            self._stats["ticks_wait_fallback"] += 1
            return Action(
                tool_calls=[
                    ToolCall(
                        name="wait", idempotency_key=self._next_idem_key("fallback")
                    )
                ],
                dominant="wait",
                assistant_text="THOUGHT: no API key configured; falling back to wait.",
            )
        try:
            action = self._call_llm(observation, last_reward)
            self._last_provider_outcome = {"status": "success"}
            self._stats["llm_calls_ok"] += 1
            self._consecutive_provider_failures = 0
            self._stats["tool_calls_requested"] += len(action.tool_calls)
            self._stats["commit_to_plan_calls"] += sum(
                1 for c in action.tool_calls if c.name == "commit_to_plan"
            )
            return action
        except (ProviderQuotaExhaustedError, ProviderCircuitOpenError):
            raise
        except Exception as exc:
            reason = self._note_failed_llm_call(exc)
            self._stats["ticks_wait_fallback"] += 1
            exc_summary = redact_provider_error(exc)
            LOGGER.warning(
                "ReAct LLM call failed at tick %d (%s): %s; wait fallback.",
                self._tick,
                reason,
                exc_summary,
            )
            return Action(
                tool_calls=[
                    ToolCall(name="wait", idempotency_key=self._next_idem_key("err"))
                ],
                dominant="hard_error_fallback",
                assistant_text=(
                    f"THOUGHT: LLM error {type(exc).__name__}: {exc_summary}"
                ),
            )

    def on_episode_end(
        self,
        final_observation: dict[str, Any],
        actions: list[Action],
        episode_reward: float = 0.0,
    ) -> None:
        """ReAct does not persist memory across episodes (Reflexion does)."""
        return

    def get_interaction_stats(self) -> dict[str, Any]:
        return super().get_interaction_stats()

    # ── Provider plumbing (mirrors LLMAgent's openai-compatible path) ──

    def _make_client(self, api_key: str) -> Any:
        # Reuse LLMAgent's client construction logic.
        return LLMAgent._make_client(self, api_key)

    def _call_llm(
        self,
        observation: dict[str, Any],
        last_reward: float = 0.0,
    ) -> Action:
        body = self._observation_summary(observation, last_reward)
        body["interaction_stage"] = observation.get(
            "__interaction_stage__", "commit"
        )
        allowed_tools = observation.get("__allowed_tool_names__")
        if allowed_tools:
            body["allowed_tool_names"] = list(allowed_tools)
        user_msg = {
            "role": "user",
            "content": (
                f"Tick {observation.get('tick', self._tick)}. "
                "Observation + ReAct context:\n"
                + self._serialize_prompt_body(
                    body, max_chars=self._observation_budget_chars
                )
            ),
        }
        messages = [
            {"role": "system", "content": self._system_prompt},
            user_msg,
        ]
        original_tool_specs = self._tool_specs
        if allowed_tools is not None:
            allowed = {str(name) for name in allowed_tools}
            self._tool_specs = [
                spec
                for spec in original_tool_specs
                if str((spec.get("function") or {}).get("name")) in allowed
            ]
        request_tool_specs = list(self._tool_specs)
        available_prior_call_ids = {
            str(call.get("call_id"))
            for decision in self._memory.recent_actions
            if isinstance(decision, dict)
            for call in (decision.get("tool_calls") or [])
            if isinstance(call, dict) and call.get("call_id")
        }
        provider_started_ns = time.monotonic_ns()
        self._last_provider_response_metadata = {}
        self._record_provider_request(
            messages=messages,
            tools=self._tool_specs,
            fallback_without_tools=False,
        )
        try:
            action = self._provider_dispatch(messages)
        except Exception as exc:
            self._record_provider_action_response(
                None,
                started_ns=provider_started_ns,
                error=exc,
            )
            raise
        finally:
            self._tool_specs = original_tool_specs
        native_protocol_violation = self._native_action_protocol_violation(
            action, request_tool_specs
        )
        self._record_provider_action_response(
            action,
            started_ns=provider_started_ns,
            decision_valid=native_protocol_violation is None,
        )
        self._stats["native_decision_responses"] += 1
        if native_protocol_violation is None:
            self._stats["native_tool_protocol_valid_responses"] += 1
        else:
            self._stats["native_tool_protocol_invalid_responses"] += 1
        if native_protocol_violation is None:
            action = self._bound_native_action_dependencies(
                action, available_prior_call_ids
            )
        elif native_protocol_violation in {
            "malformed_tool_arguments",
            "unknown_tool",
        }:
            action = Action(
                tool_calls=[],
                dominant=(
                    "native_malformed_tool_rejected"
                    if native_protocol_violation == "malformed_tool_arguments"
                    else "native_unknown_tool_rejected"
                ),
                assistant_text=action.assistant_text,
                rationale=action.rationale,
            )
        # Record this tick's reasoning + tool plan in the scratchpad.
        self._memory.scratchpad.append(
            f"tick={self._tick} thought={action.assistant_text[:200]} "
            f"tools={[c.name for c in action.tool_calls]}"
        )
        self._memory.recent_actions.append(
            {
                "tick": self._tick,
                "tool_calls": [
                    {
                        "name": c.name,
                        "args": c.args,
                        "call_id": c.call_id,
                    }
                    for c in action.tool_calls
                ],
            }
        )
        self._memory.recent_actions = self._memory.recent_actions[-self.RECENT_WINDOW :]
        return action

    def _observation_summary(
        self, observation: dict[str, Any], last_reward: float
    ) -> dict[str, Any]:
        entities = observation.get("entities", {})
        gens = {eid: e for eid, e in entities.items() if e.get("kind") == "generator"}
        loads = {eid: e for eid, e in entities.items() if e.get("kind") == "load"}
        by_kind: dict[str, dict[str, Any]] = {}
        for eid, entity in entities.items():
            by_kind.setdefault(str(entity.get("kind", "unknown")), {})[eid] = entity
        # ReAct: expose more entities than plain LLMAgent (25 vs 5)
        return {
            "tick": observation.get("tick"),
            "horizon": observation.get("horizon"),
            "totals": observation.get("totals", {}),
            "n_generators": len(gens),
            "n_loads": len(loads),
            "entity_kind_counts": {
                kind: len(kind_entities)
                for kind, kind_entities in sorted(by_kind.items())
            },
            "sample_generators": {
                eid: _prompt_safe_entity(entity) if isinstance(entity, dict) else entity
                for eid, entity in list(gens.items())[:8]
            },
            "sample_loads": {
                eid: _prompt_safe_entity(entity) if isinstance(entity, dict) else entity
                for eid, entity in list(loads.items())[:8]
            },
            "sample_entities_by_kind": {
                kind: {
                    eid: _prompt_safe_entity(entity)
                    if isinstance(entity, dict)
                    else entity
                    for eid, entity in list(kind_entities.items())[:8]
                }
                for kind, kind_entities in sorted(by_kind.items())
            },
            "stakeholder_trust": observation.get("stakeholder_trust", {}),
            "active_dilemmas": observation.get("active_dilemmas", []),
            "belief_summary": build_visible_belief_summary(observation),
            "tool_budget": observation.get("__tool_budget__", {}),
            "react_context": {
                "last_reward": last_reward,
                "last_tool_results": self._memory.recent_results[-1:],
                "last_evidence_ids": (
                    self._memory.recent_evidence_ids[-1]["evidence_ids"]
                    if self._memory.recent_evidence_ids
                    else []
                ),
                "recent_evidence_ids": self._memory.recent_evidence_ids[-3:],
                "recently_realized_events": [
                    _event_prompt_view(event)
                    for event in self._memory.realized_events[-6:]
                ],
                "recent_actions": self._memory.recent_actions[-3:],
                "scratchpad_tail": self._memory.scratchpad[-3:],
            },
        }

    # The three provider-specific calls reuse LLMAgent's implementations.
    def _call_openai_compatible(self, messages: list[dict[str, Any]]) -> Action:
        return self._provider_dispatch(messages)

    def _call_anthropic(self, messages: list[dict[str, Any]]) -> Action:
        return self._provider_dispatch(messages)

    def _call_google(self, messages: list[dict[str, Any]]) -> Action:
        return self._provider_dispatch(messages)

    @staticmethod
    def _is_tool_calling_provider_error(exc: BaseException) -> bool:
        from .llm_agent import LLMAgent

        return LLMAgent._is_tool_calling_provider_error(exc)

    def _extract_openai_calls(self, msg: Any) -> list[ToolCall]:
        from .llm_agent import LLMAgent

        return LLMAgent.__dict__["_extract_openai_calls"](self, msg)
