"""
core.pomdp_env — Abstract base for domain-specific POMDP environments.

Every domain (power_grid, traffic, disaster, ...) implements this class on
top of its real simulator backend. The runner (run.py / batch_eval.py) only
talks to this contract.

Split from ``core.pomdp`` to keep the dataclasses importable without forcing
backend SDK imports just to read the types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .pomdp import Action, StepReturn, TickBudget, ToolResult


class POMDPEnvironment(ABC):
    """Domain-agnostic environment contract.

    Implementations MUST be deterministic given ``(scenario_config, seed)``
    so counterfactual replay (`core.counterfactual`) works correctly. The
    only acceptable source of randomness is the seeded RNG that the
    implementation seeds in ``reset``.
    """

    #: human-readable domain name, e.g. "power_grid"
    domain: str = "abstract"

    @abstractmethod
    def reset(self, scenario_config: dict[str, Any], seed: int) -> dict[str, Any]:
        """Reset to a fresh episode. Returns the initial observation."""

    @abstractmethod
    def step(self, action: Action) -> StepReturn:
        """Apply an action for one tick; return obs + tool_results + reward."""

    @abstractmethod
    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the agent-visible observation.

        Implementations MUST filter through the FogOfWarPolicy so hidden
        state is not leaked.
        """

    @abstractmethod
    def ground_truth(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the FULL state (for the evaluator
        and counterfactual replay — NEVER shown to the agent)."""

    @property
    @abstractmethod
    def tick(self) -> int:
        """Current tick index (0-based, incremented after each ``step``)."""

    @property
    @abstractmethod
    def horizon(self) -> int:
        """Maximum number of ticks for this episode."""

    @property
    def budget(self) -> TickBudget:
        """Tick-level tool budget (override to customize)."""
        return TickBudget()

    def close(self) -> None:  # noqa: B027 - optional no-op capability hook
        """Free backend resources. Default is a no-op."""
        pass

    # ── Optional capability hooks (override in domain adapters) ──────────
    def get_tool_specs(self) -> list[dict[str, Any]]:
        """OpenAI-style function schemas for all tools, used by llm_agent.

        Default returns an empty list; domain adapters should override.
        """
        return []

    def supports_counterfactual(self) -> bool:
        """Whether this env supports masked-action replay. Default True for
        any deterministic simulator backend."""
        return True

    def source_consumption_evidence(
        self,
        *,
        scenario: dict[str, Any],
    ) -> dict[str, Any]:
        """Return backend-native source load/window evidence when available.

        The default deliberately reports no proof. A backend must expose a
        trace hook rather than receiving credit from provenance metadata.
        """
        from domains.registry import (
            get_backend_capability,
            resolve_backend_source_evidence_adapter,
        )

        try:
            capability = get_backend_capability(scenario.get("backend_kind"))
            extractor = resolve_backend_source_evidence_adapter(capability)
        except (ImportError, AttributeError, KeyError, TypeError, ValueError) as exc:
            return {
                "status": "held",
                "blockers": ["backend_source_evidence_adapter_unimplemented"],
                "detail": str(exc),
            }
        try:
            evidence = extractor(env=self, scenario=scenario)
        except Exception as exc:
            return {
                "status": "held",
                "blockers": ["backend_source_evidence_adapter_exception"],
                "detail": f"{type(exc).__name__}: {exc}",
            }
        return evidence if isinstance(evidence, dict) else {
            "status": "held",
            "blockers": ["backend_source_evidence_adapter_invalid"],
        }

    def readonly_tool_names(self) -> set[str] | None:
        """Names of registered read-only (non-state-changing) tools.

        Used by the counterfactual ``keep_investigations`` masking policy so
        it stays domain-agnostic instead of hardcoding a per-domain tool list.
        The default inspects the conventional ``self._tools`` ``ToolRegistry``
        that every domain adapter builds in ``reset``; it returns ``None`` when
        the registry is unavailable (e.g. before reset) so callers can fall
        back to a safe default.
        """
        reg = getattr(self, "_tools", None)
        getter = getattr(reg, "readonly_names", None)
        if callable(getter):
            try:
                return set(getter())
            except Exception:
                return None
        return None

    def execute_investigation(
        self, action: Action
    ) -> tuple[dict[str, Any], list[ToolResult]]:
        """Execute one read-only stage without advancing simulator time.

        The following ``step`` shares the already-open per-tick budget. Domain
        adapters use the conventional registry/context attributes required by
        the benchmark contract; unsupported environments fail explicitly.
        """
        from .tool_protocol import ToolContext

        registry = getattr(self, "_tools", None)
        backend = getattr(self, "_backend", None)
        evidence = getattr(self, "_evidence", None)
        if registry is None or backend is None:
            raise NotImplementedError("environment has no tool registry/backend")
        readonly = self.readonly_tool_names() or set()
        rejected = [call for call in action.tool_calls if call.name not in readonly]
        allowed = [call for call in action.tool_calls if call.name in readonly]
        ctx = ToolContext(
            tick=int(self.tick),
            seed=int(getattr(getattr(self, "_seed_obj", None), "seed", 0)),
            backend=backend,
            extra={
                "fog": getattr(self, "_fog", None),
                "stakeholders": getattr(self, "_stakeholders", None),
                "dilemmas": getattr(self, "_dilemmas", None),
                "evidence": evidence,
                "cascade_bus": getattr(self, "_cascade_bus", None),
                "env": self,
            },
        )
        allowed_results = registry.execute_action(
            Action(tool_calls=allowed, dominant="investigate"),
            ctx,
            begin_tick=not bool(
                getattr(self, "_within_tick_budget_open", False)
            ),
        )
        rejected_results = {
            id(call): ToolResult(
                name=call.name,
                ok=False,
                error_code="STATE_CHANGE_NOT_ALLOWED_DURING_INVESTIGATION",
                error_message="only read-only tools may run before the commit stage",
                call_id=call.call_id,
                state_changing=True,
            )
            for call in rejected
        }
        results: list[ToolResult] = []
        used_result_ids: set[int] = set()
        for call in action.tool_calls:
            if id(call) in rejected_results:
                results.append(rejected_results[id(call)])
                continue
            match = next(
                (
                    result
                    for result in allowed_results
                    if id(result) not in used_result_ids
                    and result.call_id == call.call_id
                ),
                None,
            )
            if match is not None:
                used_result_ids.add(id(match))
                results.append(match)
        results.extend(
            result for result in allowed_results if id(result) not in used_result_ids
        )
        calls_by_id = {call.call_id: call for call in action.tool_calls if call.call_id}
        if evidence is not None:
            for result in results:
                call = calls_by_id.get(result.call_id)
                linked_result_evidence_id = result.evidence_id
                payload = {
                    "name": result.name,
                    "ok": result.ok,
                    "error_code": result.error_code,
                    "cost_units": result.cost_units,
                    "call_id": result.call_id,
                    "state_changing": result.state_changing,
                    "payload": result.payload,
                    "consumes_evidence_ids": (
                        call.consumes_evidence_ids if call is not None else None
                    ),
                    "depends_on_call_ids": (
                        call.depends_on_call_ids if call is not None else None
                    ),
                    "interaction_stage": "investigation",
                }
                if linked_result_evidence_id:
                    payload["linked_result_evidence_id"] = linked_result_evidence_id
                result.evidence_id = evidence.log(
                    kind="tool_call",
                    tick=int(self.tick),
                    payload=payload,
                    source="tool",
                )
        self._within_tick_budget_open = True
        observation = self.snapshot()
        budget_status = getattr(registry, "budget_status", None)
        if callable(budget_status):
            observation["__tool_budget__"] = budget_status()
        return observation, results

    def supports_control_reconciliation(self) -> bool:
        """Whether the environment exposes a two-phase control boundary.

        The default remains false so existing adapters keep their atomic
        ``step(action)`` behavior.  Adapters that opt in must execute and
        return state-changing tool receipts without advancing simulator time,
        accept at most one explicitly linked injected-failure retry, and then
        advance exactly once through ``advance_staged_control``.
        """
        return False

    def stage_control(
        self, action: Action
    ) -> tuple[dict[str, Any], list[ToolResult]]:
        """Execute one control stage without advancing simulator time."""
        raise NotImplementedError("environment has no two-phase control boundary")

    def advance_staged_control(self) -> StepReturn:
        """Advance once after a staged control attempt and optional retry."""
        raise NotImplementedError("environment has no staged control to advance")

    def _consume_within_tick_budget_state(self) -> bool:
        """Return whether a query stage opened this tick, then close it."""
        was_open = bool(getattr(self, "_within_tick_budget_open", False))
        self._within_tick_budget_open = False
        return was_open

    @staticmethod
    def tool_dependency_payload(action: Action, result: ToolResult) -> dict[str, Any]:
        """Recover explicit evidence/action edges for a materialized result."""
        call = next(
            (candidate for candidate in action.tool_calls if candidate.call_id == result.call_id),
            None,
        )
        if call is None:
            return {
                "consumes_evidence_ids": (
                    list(result.consumes_evidence_ids or []) or None
                ),
                "depends_on_call_ids": (
                    list(result.depends_on_call_ids or []) or None
                ),
            }
        return {
            "consumes_evidence_ids": call.consumes_evidence_ids,
            "depends_on_call_ids": call.depends_on_call_ids,
        }
