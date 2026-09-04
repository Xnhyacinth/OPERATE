"""
core.tool_protocol — Typed tool execution layer.

Every domain-native tool (e.g., ``shed_load``, ``redispatch_generation``,
``investigate_substation``) is registered as a ``ToolSpec`` against this
registry. The runner calls ``ToolRegistry.execute(name, args, ctx)`` and
the protocol handles:

- Argument validation against the OpenAI-style JSON schema
- Per-tool ``fail_rate`` and ``delay_ticks`` injection (deterministic per
  ``(seed, tick, name, idempotency_key)`` triple)
- Per-tick ``TickBudget`` accounting
- Duplicate suppression within a window
- Recent-failure cooldown
- Pending action tracking when ``delay_ticks > 0``
- Structured ``ToolResult`` output for the trajectory logger

Originally proposed in
``ai_sched_bench/docs/TOOL_MODEL_SPEC.md`` (tool imperfection contract);
this file is a clean, audit-friendly implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from .difficulty_levels import canonical_difficulty_level
from .pomdp import Action, TickBudget, ToolCall, ToolResult

# ─────────────────────────────────────────────────────────────────────────────
# Tool spec
# ─────────────────────────────────────────────────────────────────────────────


ToolSemanticRole = Literal[
    "investigation",
    "control",
    "planning",
    "communication",
    "meta",
]
TOOL_SEMANTIC_ROLES: frozenset[str] = frozenset(
    {"investigation", "control", "planning", "communication", "meta"}
)
INFRASTRUCTURE_TOOL_ERROR_CODES: frozenset[str] = frozenset(
    {"HANDLER_EXCEPTION"}
)


def is_infrastructure_tool_failure(result: ToolResult) -> bool:
    """Return whether a tool result reflects harness/runtime failure.

    Domain rejection, injected failure, validation failure, and budget failure
    are part of the task contract. An exception escaping a registered handler
    is an invalid execution environment and must never become model score.
    """

    return bool(
        result.ok is False
        and str(result.error_code or "") in INFRASTRUCTURE_TOOL_ERROR_CODES
    )


@dataclass
class ToolSpec:
    """OpenAI-style tool spec plus imperfection knobs and a handler.

    ``fail_rate`` / ``delay_ticks`` default to ``None`` (P1-4) so the
    protocol can distinguish "unset — inherit per-difficulty profile
    default" from "explicitly zero". Explicit values always override the
    profile; ``None`` is resolved at execution via
    ``ToolRegistry.resolve_imperfection``. The registry normally invokes a
    delayed handler only at its due tick. Set ``handler_manages_delay`` only
    for legacy/native handlers that already queue their own future backend
    effect; their acknowledgement is still withheld until the same due tick.
    Such handlers must provide ``cancel_pending`` before realtime execution
    can safely fence a superseded or expired call.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema, OpenAI function-tool style
    handler: Callable[[dict[str, Any], ToolContext], dict[str, Any]]
    state_changing: bool = False
    fail_rate: float | None = None
    delay_ticks: int | None = None
    handler_manages_delay: bool = False
    cancel_pending: Callable[[ToolCall], bool] | None = None
    cost_units: float = 0.0  # for budget tracking / leaderboard
    semantic_role: ToolSemanticRole | None = None
    native_target_kind: str | None = None
    actuator_family: str | None = None

    def __post_init__(self) -> None:
        if self.semantic_role is not None and self.semantic_role not in TOOL_SEMANTIC_ROLES:
            raise ValueError(f"invalid semantic_role: {self.semantic_role}")
        for field_name in ("native_target_kind", "actuator_family"):
            value = getattr(self, field_name)
            if value is not None and not value.strip():
                raise ValueError(f"{field_name} must be non-empty when provided")

    def resolved_semantic_role(self) -> ToolSemanticRole:
        """Return explicit semantics or a backward-compatible legacy inference.

        Historical domain registrations predate semantic metadata. Their existing
        ``state_changing`` contract is authoritative enough to distinguish control
        from read-only investigation. Shared no-op names retain meta semantics.
        Formal trajectory validation still fails for names absent from the
        registry, rather than silently placing them in an ``other`` bucket.
        """
        if self.semantic_role is not None:
            return self.semantic_role
        if self.name in {"wait", "noop"}:
            return "meta"
        if self.name == "commit_to_plan":
            return "planning"
        if self.name in {
            "stakeholder_query",
            "negotiate_with_stakeholder",
            "escalate_to_human",
            "moral_choice",
        }:
            return "communication"
        return "control" if self.state_changing else "investigation"

    def to_openai_schema(self) -> dict[str, Any]:
        resource_note = f"Resource cost: {float(self.cost_units):g} units per call."
        description = f"{self.description.rstrip()} {resource_note}"
        return {
            "type": "function",
            "x-cost-units": float(self.cost_units),
            "function": {
                "name": self.name,
                "description": description,
                "parameters": sanitize_openai_parameters(self.parameters),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-difficulty imperfection profile (P1-4)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DifficultyImperfectionProfile:
    """P1-4: per-difficulty default ``fail_rate`` / ``delay_ticks`` for tools
    that do not set them explicitly. Makes tool imperfection a default contract
    (Hard Red Line #7), not an opt-in. Explicit per-tool values always override.

    Levels are monotonic in imperfection: basic is clean, extreme is the
    harshest. Applied at execution time via ``ToolRegistry.resolve_imperfection``.
    """

    basic: dict[str, float] = field(
        default_factory=lambda: {"fail_rate": 0.0, "delay_ticks": 0}
    )
    medium: dict[str, float] = field(
        default_factory=lambda: {"fail_rate": 0.03, "delay_ticks": 1}
    )
    high: dict[str, float] = field(
        default_factory=lambda: {"fail_rate": 0.05, "delay_ticks": 1}
    )
    extreme: dict[str, float] = field(
        default_factory=lambda: {"fail_rate": 0.08, "delay_ticks": 2}
    )

    def for_level(self, level: str) -> dict[str, float]:
        """Return the imperfection dict for ``level``.

        Frozen-release aliases canonicalize to ``extreme`` via
        ``core.difficulty_levels``, so old artifacts get the harshest
        imperfection rather than silently falling back to ``basic``.
        Unknown levels fall back to ``basic``.
        """
        canon = canonical_difficulty_level(level)
        return getattr(self, canon, self.basic) if hasattr(self, canon) else self.basic


# ─────────────────────────────────────────────────────────────────────────────
# Context object passed to handlers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolContext:
    """Side-channel passed to every tool handler.

    Lets a handler mutate the underlying simulator's state without forcing
    every tool signature to take a backend reference. Each domain adapter
    populates ``backend`` with whatever object handlers need (e.g., Grid2Op
    env, pandapower net, action_space helper).
    """

    tick: int
    seed: int
    backend: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class _PendingInvocation:
    call: ToolCall
    tool_name: str
    cooldown_key: str
    latency_ticks: int


@dataclass
class _PendingManagedResult:
    result: ToolResult
    call: ToolCall
    cancel_pending: Callable[[ToolCall], bool] | None


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────


class ToolRegistry:
    """Holds all domain-native tools and enforces the protocol invariants."""

    def __init__(
        self,
        budget: TickBudget | None = None,
        seed: int = 0,
        difficulty_level: str = "basic",
    ):
        self._tools: dict[str, ToolSpec] = {}
        self._budget = budget or TickBudget()
        self._seed = seed
        self._rng = random.Random(seed)
        self._attempted_total = 0
        self._call_seq = 0
        self._attempts_this_tick = 0
        self._cost_this_tick = 0.0
        self._recent_failures: deque[tuple[int, str]] = deque(maxlen=32)
        # ring of recent (tick, name+argshash) to suppress duplicates
        self._recent_calls: deque[tuple[int, str]] = deque(maxlen=128)
        self._idempotency_keys: dict[str, tuple[str, str]] = {}
        self._pending_invocations: dict[int, list[_PendingInvocation]] = defaultdict(
            list
        )
        self._pending_results: dict[int, list[_PendingManagedResult]] = defaultdict(
            list
        )
        self._retryable_injected_failures_this_tick: dict[str, tuple[int, str]] = {}
        self._linked_retry_parent_ids_this_tick: set[str] = set()
        # P1-4: per-difficulty default fail_rate/delay_ticks applied when a
        # tool registers without explicit imperfection kwargs. Frozen profile
        # cached once; resolution is a pure lookup at execution time.
        self._difficulty_level = difficulty_level
        self._imperfection_profile = DifficultyImperfectionProfile()

    # ── Registration ────────────────────────────────────────────────────

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def semantic_roles(
        self, *, explicit_only: bool = False
    ) -> dict[str, ToolSemanticRole]:
        """Return registered tool roles for evaluation and release audits."""
        roles: dict[str, ToolSemanticRole] = {}
        for name, spec in sorted(self._tools.items()):
            if explicit_only:
                if spec.semantic_role is None:
                    continue
                roles[name] = spec.semantic_role
            else:
                roles[name] = spec.resolved_semantic_role()
        return roles

    def validate_semantic_coverage(
        self,
        tool_names: list[str] | tuple[str, ...] | set[str] | None = None,
        *,
        formal: bool = False,
    ) -> dict[str, Any]:
        """Audit semantic coverage, failing closed for formal trajectories.

        ``tool_names`` normally comes from the trajectory histogram. This makes
        a provider-emitted or stale tool name a formal contract violation even
        though diagnostic analyses may still report it as unknown.
        """
        requested = set(self._tools) if tool_names is None else {str(n) for n in tool_names}
        unknown = sorted(requested - self._tools.keys())
        unclassified = sorted(
            name
            for name in requested.intersection(self._tools)
            if self._tools[name].resolved_semantic_role() not in TOOL_SEMANTIC_ROLES
        )
        missing_explicit = sorted(
            name
            for name in requested.intersection(self._tools)
            if self._tools[name].semantic_role is None
        )
        missing_native_target = sorted(
            name
            for name in requested.intersection(self._tools)
            if self._tools[name].native_target_kind is None
        )
        missing_actuator_family = sorted(
            name
            for name in requested.intersection(self._tools)
            if self._tools[name].state_changing
            and self._tools[name].actuator_family is None
        )
        report = {
            "covered": not unknown and not unclassified,
            "explicit_semantic_roles_complete": not missing_explicit,
            "native_targets_complete": not missing_native_target,
            "state_changing_actuators_complete": not missing_actuator_family,
            "registered_tool_names": sorted(requested.intersection(self._tools)),
            "unknown_tool_names": unknown,
            "unclassified_tool_names": unclassified,
            "missing_explicit_semantic_role_names": missing_explicit,
            "missing_native_target_kind_names": missing_native_target,
            "missing_actuator_family_names": missing_actuator_family,
        }
        if formal and (
            not report["covered"]
            or missing_explicit
            or missing_native_target
            or missing_actuator_family
        ):
            failures = (
                unknown
                + unclassified
                + missing_explicit
                + missing_native_target
                + missing_actuator_family
            )
            raise ValueError(f"unknown_tool_semantics: {', '.join(failures)}")
        return report

    def readonly_names(self) -> list[str]:
        """Names of registered tools that do not change state.

        This is the domain-agnostic signal the counterfactual
        ``keep_investigations`` masking policy uses to keep investigative
        (read-only) tools while dropping state-changing actions, without
        hardcoding any per-domain tool list.
        """
        return sorted(n for n, s in self._tools.items() if not s.state_changing)

    def openai_schemas(self) -> list[dict[str, Any]]:
        return [s.to_openai_schema() for s in self._tools.values()]

    def budget_status(self) -> dict[str, int | float | None]:
        """Return the remaining shared tick/episode budget for agent feedback."""
        max_episode = self._budget.max_total_tool_calls
        max_cost = self._budget.max_cost_units_per_tick
        return {
            "max_calls_per_tick": self._budget.max_tool_calls_per_tick,
            "attempted_calls_this_tick": self._attempts_this_tick,
            "remaining_calls_this_tick": max(
                0,
                self._budget.max_tool_calls_per_tick - self._attempts_this_tick,
            ),
            "max_calls_per_episode": max_episode,
            "attempted_calls_this_episode": self._attempted_total,
            "remaining_calls_this_episode": (
                max(0, max_episode - self._attempted_total)
                if max_episode is not None
                else None
            ),
            "max_cost_units_per_tick": max_cost,
            "used_cost_units_this_tick": round(self._cost_this_tick, 6),
            "remaining_cost_units_this_tick": (
                round(max(0.0, max_cost - self._cost_this_tick), 6)
                if max_cost is not None
                else None
            ),
        }

    # ── P1-4: per-difficulty imperfection resolution ────────────────────

    @property
    def difficulty_level(self) -> str:
        return self._difficulty_level

    def resolve_imperfection(self, name: str) -> dict[str, float]:
        """Return the effective ``{fail_rate, delay_ticks}`` for a tool.

        Explicit per-tool values (``ToolSpec.fail_rate`` / ``delay_ticks``
        set at registration) always override the per-difficulty profile
        default. ``None`` (unset) inherits the profile default for the
        active difficulty level. Unknown tools resolve to the zero-imperfection
        default so callers cannot crash on a missing registration.
        """
        # Meta-actions advance the interaction protocol without representing
        # an external service request. Their semantics are globally immediate
        # and infallible even when a domain registers a custom evidence-logging
        # handler and leaves imperfection fields unset.
        if name in {"wait", "noop"}:
            return {"fail_rate": 0.0, "delay_ticks": 0}
        spec = self._tools.get(name)
        if spec is None:
            return {"fail_rate": 0.0, "delay_ticks": 0}
        profile = self._imperfection_profile.for_level(self._difficulty_level)
        return {
            "fail_rate": spec.fail_rate
            if spec.fail_rate is not None
            else profile["fail_rate"],
            "delay_ticks": spec.delay_ticks
            if spec.delay_ticks is not None
            else profile["delay_ticks"],
        }

    # ── Episode lifecycle ───────────────────────────────────────────────

    def reset(self, seed: int) -> None:
        self._seed = seed
        self._rng = random.Random(seed)
        self._attempted_total = 0
        self._call_seq = 0
        self._attempts_this_tick = 0
        self._recent_failures.clear()
        self._recent_calls.clear()
        self._idempotency_keys.clear()
        self._pending_invocations.clear()
        self._pending_results.clear()
        self._retryable_injected_failures_this_tick.clear()
        self._linked_retry_parent_ids_this_tick.clear()

    def begin_tick(self) -> None:
        self._attempts_this_tick = 0
        self._cost_this_tick = 0.0
        self._retryable_injected_failures_this_tick.clear()
        self._linked_retry_parent_ids_this_tick.clear()

    # ── Execution ───────────────────────────────────────────────────────

    def execute_action(
        self,
        action: Action,
        ctx: ToolContext,
        *,
        begin_tick: bool = True,
    ) -> list[ToolResult]:
        """Execute all tool calls in ``action`` for the current tick.

        Returns results materialized at this tick followed by acknowledgements
        or immediate results for calls submitted in the current action.
        """
        if begin_tick:
            self.begin_tick()
        results = self._materialize_due_invocations(ctx)
        for index, call in enumerate(action.tool_calls):
            if call.call_id is None:
                call.call_id = f"tick-{ctx.tick}-call-{self._call_seq}"
                self._call_seq += 1
            if self._episode_budget_exhausted():
                results.append(_episode_budget_exhausted_result())
                break
            if self._attempts_this_tick >= self._budget.max_tool_calls_per_tick:
                results.append(_tick_budget_exhausted_result(call.name, self._budget))
                continue
            self._attempts_this_tick += 1
            self._attempted_total += 1
            result = self._execute_one(call, ctx)
            results.append(result)
            spec = self._tools.get(call.name)
            if (
                result.error_code == "INJECTED_FAILURE"
                and spec is not None
                and spec.state_changing
                and call.call_id
            ):
                self._retryable_injected_failures_this_tick[str(call.call_id)] = (
                    int(ctx.tick),
                    call.name,
                )
            if (
                self._episode_budget_exhausted()
                and index + 1 < len(action.tool_calls)
            ):
                results.append(_episode_budget_exhausted_result())
                break
        return results

    def execute_injected_failure_retry(
        self,
        action: Action,
        ctx: ToolContext,
    ) -> list[ToolResult]:
        """Execute one explicit same-tick retry linked to an injected failure.

        This is deliberately narrower than ``execute_action``.  It never opens
        a new tick budget and accepts exactly one state-changing call whose
        sole dependency is a same-tick ``INJECTED_FAILURE`` receipt for the
        same tool.  The dependency may be consumed only once.  Validation,
        cost, idempotency, and handler/domain failures still use the ordinary
        fail-closed path; only the post-failure cooldown is bypassed.
        """
        calls = list(action.tool_calls)
        if len(calls) != 1:
            return [
                self._retry_rejected_result(
                    call,
                    error_code="RECONCILIATION_RETRY_LIMIT",
                    error_message="exactly one linked retry is allowed",
                )
                for call in calls
            ] or [
                ToolResult(
                    name="__reconciliation__",
                    ok=False,
                    error_code="RECONCILIATION_RETRY_LIMIT",
                    error_message="exactly one linked retry is required",
                )
            ]

        call = calls[0]
        if call.call_id is None:
            call.call_id = f"tick-{ctx.tick}-call-{self._call_seq}"
            self._call_seq += 1
        dependencies = list(call.depends_on_call_ids or [])
        parent_id = str(dependencies[0]) if len(dependencies) == 1 else ""
        parent = self._retryable_injected_failures_this_tick.get(parent_id)
        spec = self._tools.get(call.name)
        if (
            not parent_id
            or parent is None
            or parent[0] != int(ctx.tick)
            or parent[1] != call.name
            or parent_id in self._linked_retry_parent_ids_this_tick
            or spec is None
            or not spec.state_changing
        ):
            return [
                self._retry_rejected_result(
                    call,
                    error_code="INVALID_RETRY_DEPENDENCY",
                    error_message=(
                        "retry must depend on one unused same-tick injected "
                        "failure receipt for the same state-changing tool"
                    ),
                )
            ]
        if self._episode_budget_exhausted():
            result = _episode_budget_exhausted_result()
            result.call_id = call.call_id
            result.depends_on_call_ids = dependencies
            return [result]
        if self._attempts_this_tick >= self._budget.max_tool_calls_per_tick:
            result = _tick_budget_exhausted_result(call.name, self._budget)
            result.call_id = call.call_id
            result.depends_on_call_ids = dependencies
            return [result]

        self._linked_retry_parent_ids_this_tick.add(parent_id)
        self._attempts_this_tick += 1
        self._attempted_total += 1
        return [self._execute_one(call, ctx, bypass_failure_cooldown=True)]

    def _retry_rejected_result(
        self,
        call: ToolCall,
        *,
        error_code: str,
        error_message: str,
    ) -> ToolResult:
        if call.call_id is None:
            call.call_id = f"tick-retry-call-{self._call_seq}"
            self._call_seq += 1
        spec = self._tools.get(call.name)
        return ToolResult(
            name=call.name,
            ok=False,
            error_code=error_code,
            error_message=error_message,
            state_changing=bool(spec is not None and spec.state_changing),
            idempotency_key=call.idempotency_key,
            call_id=call.call_id,
            consumes_evidence_ids=list(call.consumes_evidence_ids or []) or None,
            depends_on_call_ids=list(call.depends_on_call_ids or []) or None,
        )

    def cancel_pending_calls(self, call_ids: set[str]) -> set[str]:
        """Cancel queued calls before their state-changing handler can run.

        Handler-managed delays are cancellable only when their ``ToolSpec``
        supplies an explicit native cancellation callback.  Missing or failed
        native cancellation therefore remains pending instead of being
        reported as safely fenced.
        """

        return {
            str(row["call_id"])
            for row in self.cancel_pending_calls_with_audit(call_ids)
            if row["outcome"] == "canceled"
        }

    def cancel_pending_calls_with_audit(
        self, call_ids: set[str]
    ) -> list[dict[str, Any]]:
        """Cancel pending calls and return one sanitized audit row per call."""

        requested = {str(call_id) for call_id in call_ids if call_id}
        audit = {
            call_id: {
                "call_id": call_id,
                "queue_kind": None,
                "outcome": "not_pending",
                "callback_invoked": False,
                "callback_error_type": None,
            }
            for call_id in requested
        }
        for due_tick, invocations in list(self._pending_invocations.items()):
            retained: list[_PendingInvocation] = []
            for pending in invocations:
                call_id = str(pending.call.call_id or "")
                if call_id in requested:
                    audit[call_id].update(
                        queue_kind="registry_invocation",
                        outcome="canceled",
                    )
                else:
                    retained.append(pending)
            if retained:
                self._pending_invocations[due_tick] = retained
            else:
                self._pending_invocations.pop(due_tick, None)

        for due_tick, managed_results in list(self._pending_results.items()):
            retained_results: list[_PendingManagedResult] = []
            for pending in managed_results:
                call_id = str(pending.call.call_id or "")
                if call_id not in requested:
                    retained_results.append(pending)
                    continue
                audit[call_id]["queue_kind"] = "native_managed"
                if pending.cancel_pending is None:
                    audit[call_id]["outcome"] = "missing_callback"
                    retained_results.append(pending)
                    continue
                audit[call_id]["callback_invoked"] = True
                try:
                    native_canceled = pending.cancel_pending(
                        copy.deepcopy(pending.call)
                    )
                except Exception as exc:  # noqa: BLE001 - cancellation fails closed
                    audit[call_id].update(
                        outcome="callback_exception",
                        callback_error_type=type(exc).__name__,
                    )
                    retained_results.append(pending)
                    continue
                if native_canceled is True:
                    audit[call_id]["outcome"] = "canceled"
                else:
                    audit[call_id]["outcome"] = "callback_false"
                    retained_results.append(pending)
            if retained_results:
                self._pending_results[due_tick] = retained_results
            else:
                self._pending_results.pop(due_tick, None)
        return [audit[call_id] for call_id in sorted(audit)]

    def _materialize_due_invocations(self, ctx: ToolContext) -> list[ToolResult]:
        due = self._pending_invocations.pop(ctx.tick, [])
        results = [
            pending.result
            for pending in self._pending_results.pop(ctx.tick, [])
        ]
        for result in results:
            if result.payload.get("_status") == "pending":
                result.payload = {
                    **result.payload,
                    "_status": "materialized",
                    "materialized_from": "pending",
                }
        for pending in due:
            spec = self._tools.get(pending.tool_name)
            if spec is None:  # pragma: no cover - registry entries are immutable
                continue
            materialization_ctx = ToolContext(
                tick=ctx.tick,
                seed=ctx.seed,
                backend=ctx.backend,
                extra={
                    **ctx.extra,
                    "materialization_tick": ctx.tick,
                    "materializing_tool_call": {
                        "call_id": str(pending.call.call_id or ""),
                        "tool_name": pending.call.name,
                        "args": copy.deepcopy(pending.call.args),
                    },
                },
            )
            result = self._invoke_handler(
                pending.call,
                spec,
                materialization_ctx,
                pending.cooldown_key,
                cost_units=0.0,
            )
            result.call_id = pending.call.call_id
            result.latency_ticks = pending.latency_ticks
            result.consumes_evidence_ids = list(
                pending.call.consumes_evidence_ids or []
            ) or None
            result.depends_on_call_ids = list(
                pending.call.depends_on_call_ids or []
            ) or None
            if result.payload.get("_status") == "pending":
                result.payload = {
                    **result.payload,
                    "_status": "materialized",
                    "materialized_from": "pending",
                }
            results.append(result)
        return results

    # ── Internals ───────────────────────────────────────────────────────

    def _execute_one(
        self,
        call: ToolCall,
        ctx: ToolContext,
        *,
        bypass_failure_cooldown: bool = False,
    ) -> ToolResult:
        result = self._execute_one_unlinked(
            call,
            ctx,
            bypass_failure_cooldown=bypass_failure_cooldown,
        )
        result.call_id = call.call_id
        result.idempotency_key = call.idempotency_key
        result.consumes_evidence_ids = list(call.consumes_evidence_ids or []) or None
        result.depends_on_call_ids = list(call.depends_on_call_ids or []) or None
        return result

    def _execute_one_unlinked(
        self,
        call: ToolCall,
        ctx: ToolContext,
        *,
        bypass_failure_cooldown: bool = False,
    ) -> ToolResult:
        spec = self._tools.get(call.name)
        if spec is None:
            return ToolResult(
                name=call.name,
                ok=False,
                error_code="UNKNOWN_TOOL",
                error_message=f"tool not registered: {call.name}",
            )
        if call.args.get("__protocol_error__") == "MALFORMED_ARGUMENTS":
            return ToolResult(
                name=call.name,
                ok=False,
                payload={
                    "reason": str(
                        call.args.get("__protocol_error_reason__") or "invalid_json"
                    ),
                    "source": str(
                        call.args.get("__protocol_error_source__") or "provider"
                    ),
                },
                error_code="MALFORMED_ARGUMENTS",
                error_message="provider emitted malformed tool arguments",
                state_changing=spec.state_changing,
            )

        dedup_result, sig, cd_key = self._check_duplicate_and_idempotency(
            call, ctx, spec
        )
        if dedup_result is not None:
            return dedup_result

        if not bypass_failure_cooldown:
            cooldown_result = self._check_cooldown(call, ctx, cd_key)
            if cooldown_result is not None:
                return cooldown_result

        validation_result = self._validate_args_or_fail(call, spec, ctx, cd_key)
        if validation_result is not None:
            return validation_result

        cost_result = self._check_cost_budget(call, spec)
        if cost_result is not None:
            return cost_result
        cost_units = float(spec.cost_units)
        self._cost_this_tick += cost_units

        failure_result = self._maybe_inject_failure(call, spec, ctx, cd_key, cost_units)
        if failure_result is not None:
            return failure_result

        return self._invoke_handler_and_record(call, spec, ctx, sig, cd_key, cost_units)

    def _check_duplicate_and_idempotency(
        self, call: ToolCall, ctx: ToolContext, spec: ToolSpec
    ) -> tuple[ToolResult | None, str, str]:
        """Idempotency-key conflict check, then within-window duplicate suppression.

        Returns ``(result, sig, cd_key)``. ``result`` is ``None`` when both
        checks pass; ``sig``/``cd_key`` are the call/cooldown signatures the
        rest of ``_execute_one`` needs downstream (empty strings when an
        idempotency conflict short-circuits before they're computed, matching
        the original control flow).
        """
        conflict = self._idempotency_conflict(call)
        if conflict is not None:
            prior_name, _prior_args = conflict
            return (
                ToolResult(
                    name=call.name,
                    ok=False,
                    payload={
                        "prior_tool": prior_name,
                        "current_tool": call.name,
                        "_status": "idempotency_key_conflict",
                    },
                    error_code="IDEMPOTENCY_KEY_CONFLICT",
                    error_message=(
                        "idempotency_key reused for a different tool call "
                        f"(prior_tool={prior_name})"
                    ),
                    idempotency_key=call.idempotency_key,
                    state_changing=spec.state_changing,
                ),
                "",
                "",
            )
        sig = _call_signature(call)
        cd_key = _cooldown_key(call)
        if call.name in {"wait", "noop"}:
            return None, sig, cd_key
        cooldown_lo = ctx.tick - self._budget.cooldown_after_failure
        if any(
            tick >= cooldown_lo and key == cd_key
            for tick, key in self._recent_failures
        ):
            # Let the following cooldown check return the more specific
            # anti-storm result for an exact retry after failure.
            return None, sig, cd_key
        recent_window_lo = ctx.tick - self._budget.duplicate_suppression_window
        for t, s in self._recent_calls:
            if t >= recent_window_lo and s == sig:
                return (
                    ToolResult(
                        name=call.name,
                        ok=False,
                        error_code="DUPLICATE_SUPPRESSED",
                        error_message=(
                            "identical call repeated within suppression window"
                        ),
                    ),
                    sig,
                    cd_key,
                )
        return None, sig, cd_key

    def _check_cooldown(
        self, call: ToolCall, ctx: ToolContext, cd_key: str
    ) -> ToolResult | None:
        """Cooldown after a recent failure of the SAME call (name + args).

        Keying on the call — not just the tool name — is deliberate: a
        single malformed argument must not freeze an entire state-changing
        tool for the whole cooldown window. Otherwise one bad ``job_id``
        bricks every subsequent (valid) ``dispatch_job_operation`` call and
        collapses the family to the wait-floor. Repeating the exact failing
        call is still cooled down; residual storm risk is bounded by the
        per-tick budget and duplicate suppression.
        """
        cooldown_lo = ctx.tick - self._budget.cooldown_after_failure
        for t, k in self._recent_failures:
            if t >= cooldown_lo and k == cd_key:
                return ToolResult(
                    name=call.name,
                    ok=False,
                    error_code="COOLDOWN",
                    error_message=(f"{call.name} is in cooldown after recent failure"),
                )
        return None

    def _validate_args_or_fail(
        self, call: ToolCall, spec: ToolSpec, ctx: ToolContext, cd_key: str
    ) -> ToolResult | None:
        """Validate args against schema (lightweight — no jsonschema dep)."""
        valid, err = _validate_args(call.args, spec.parameters)
        if not valid:
            self._recent_failures.append((ctx.tick, cd_key))
            return ToolResult(
                name=call.name,
                ok=False,
                error_code="VALIDATION_ERROR",
                error_message=err,
            )
        return None

    def _check_cost_budget(self, call: ToolCall, spec: ToolSpec) -> ToolResult | None:
        if (
            self._budget.max_cost_units_per_tick is not None
            and self._cost_this_tick + float(spec.cost_units)
            > float(self._budget.max_cost_units_per_tick)
        ):
            return ToolResult(
                name=call.name,
                ok=False,
                payload={
                    "attempted_cost_units": float(spec.cost_units),
                    "used_cost_units": round(self._cost_this_tick, 6),
                    "max_cost_units_per_tick": float(
                        self._budget.max_cost_units_per_tick
                    ),
                },
                error_code="TICK_COST_BUDGET_EXHAUSTED",
                error_message=(
                    "per-tick tool cost budget exhausted "
                    f"(used={self._cost_this_tick}, "
                    f"attempted={spec.cost_units}, "
                    f"max={self._budget.max_cost_units_per_tick})"
                ),
                cost_units=0.0,
            )
        return None

    def _maybe_inject_failure(
        self,
        call: ToolCall,
        spec: ToolSpec,
        ctx: ToolContext,
        cd_key: str,
        cost_units: float,
    ) -> ToolResult | None:
        """Deterministic failure injection keyed on (seed, tick, name, idem).

        P1-4: the effective ``fail_rate`` is resolved through
        ``resolve_imperfection`` so tools that did not set one explicitly
        inherit the per-difficulty profile default at execution time.
        """
        fail_rate = self.resolve_imperfection(call.name)["fail_rate"]
        if (
            fail_rate > 0.0
            and _seeded_uniform(self._seed, ctx.tick, call.name, call.idempotency_key)
            < fail_rate
        ):
            self._recent_failures.append((ctx.tick, cd_key))
            return ToolResult(
                name=call.name,
                ok=False,
                error_code="INJECTED_FAILURE",
                error_message="tool failed (injected by tool_protocol)",
                state_changing=spec.state_changing,
                idempotency_key=call.idempotency_key,
                cost_units=cost_units,
            )
        return None

    def _invoke_handler_and_record(
        self,
        call: ToolCall,
        spec: ToolSpec,
        ctx: ToolContext,
        sig: str,
        cd_key: str,
        cost_units: float,
    ) -> ToolResult:
        """Invoke the domain handler, classify its outcome, and record the call.

        Delayed calls are queued without invoking the handler, so backend state
        cannot change before the declared due tick. Immediate calls invoke the
        handler here. Both paths record duplicate/idempotency bookkeeping at
        submission time.
        """
        delay_ticks = int(self.resolve_imperfection(call.name)["delay_ticks"])
        future_tick = ctx.tick + delay_ticks
        episode_horizon = ctx.extra.get("episode_horizon")
        if episode_horizon is None:
            episode_horizon = getattr(ctx.extra.get("env"), "horizon", None)
        if (
            delay_ticks > 0
            and episode_horizon is not None
            and future_tick >= int(episode_horizon)
        ):
            return ToolResult(
                name=call.name,
                ok=False,
                payload={
                    "due_tick": future_tick,
                    "episode_horizon": int(episode_horizon),
                },
                error_code="DEADLINE_EXCEEDS_EPISODE",
                error_message=(
                    f"delayed result due at tick {future_tick}, outside "
                    f"episode horizon {int(episode_horizon)}"
                ),
                state_changing=spec.state_changing,
                idempotency_key=call.idempotency_key,
                cost_units=cost_units,
            )
        if delay_ticks > 0 and not spec.handler_manages_delay:
            self._pending_invocations[future_tick].append(
                _PendingInvocation(
                    call=copy.deepcopy(call),
                    tool_name=call.name,
                    cooldown_key=cd_key,
                    latency_ticks=delay_ticks,
                )
            )
            ack = ToolResult(
                name=call.name,
                ok=True,
                payload={"_status": "pending", "due_tick": future_tick},
                latency_ticks=delay_ticks,
                state_changing=False,
                idempotency_key=call.idempotency_key,
                cost_units=cost_units,
            )
            self._record_call(call, sig, ctx.tick)
            return ack

        result = self._invoke_handler(
            call,
            spec,
            ctx,
            cd_key,
            cost_units=cost_units,
        )
        if delay_ticks > 0:
            result.call_id = call.call_id
            result.latency_ticks = delay_ticks
            result.cost_units = 0.0
            result.consumes_evidence_ids = list(
                call.consumes_evidence_ids or []
            ) or None
            result.depends_on_call_ids = list(
                call.depends_on_call_ids or []
            ) or None
            self._pending_results[future_tick].append(
                _PendingManagedResult(
                    result=result,
                    call=copy.deepcopy(call),
                    cancel_pending=spec.cancel_pending,
                )
            )
            ack = ToolResult(
                name=call.name,
                ok=True,
                payload={"_status": "pending", "due_tick": future_tick},
                latency_ticks=delay_ticks,
                state_changing=False,
                idempotency_key=call.idempotency_key,
                cost_units=cost_units,
            )
            self._record_call(call, sig, ctx.tick)
            return ack
        self._record_call(call, sig, ctx.tick)
        return result

    def _invoke_handler(
        self,
        call: ToolCall,
        spec: ToolSpec,
        ctx: ToolContext,
        cd_key: str,
        *,
        cost_units: float,
    ) -> ToolResult:
        """Invoke one domain handler at its physical execution tick."""
        try:
            payload = spec.handler(dict(call.args), ctx) or {}
        except Exception as exc:  # defensive — never let a domain tool crash the run
            self._recent_failures.append((ctx.tick, cd_key))
            return ToolResult(
                name=call.name,
                ok=False,
                error_code="HANDLER_EXCEPTION",
                error_message=f"{type(exc).__name__}: {exc}",
                cost_units=cost_units,
            )
        evidence_id = _pop_payload_evidence_id(payload)

        # v0.3.1 P2 fix: backend-side no-effect statuses must not earn
        # successful state-changing action credit. A hard domain rejection
        # returns ``{"_status": "error", ...}``; a backend that cannot model
        # the requested state-changing effect may return an unsupported,
        # out-of-range, or no-effect status. True meta ``wait`` / ``noop``
        # tools and read-only tools remain successful.
        if isinstance(payload, dict) and payload.get("_status") == "error":
            self._recent_failures.append((ctx.tick, cd_key))
            return ToolResult(
                name=call.name,
                ok=False,
                error_code="DOMAIN_REJECTED",
                error_message=str(
                    payload.get("error") or payload.get("info") or "domain rejection"
                ),
                payload=payload,
                state_changing=spec.state_changing,
                idempotency_key=call.idempotency_key,
                evidence_id=evidence_id,
                cost_units=cost_units,
            )
        no_effect_code = _no_effect_error_code(call, spec, payload)
        if no_effect_code is not None:
            self._recent_failures.append((ctx.tick, cd_key))
            return ToolResult(
                name=call.name,
                ok=False,
                error_code=no_effect_code,
                error_message=_payload_failure_message(payload),
                payload=payload,
                state_changing=spec.state_changing,
                idempotency_key=call.idempotency_key,
                evidence_id=evidence_id,
                cost_units=cost_units,
            )

        result = ToolResult(
            name=call.name,
            ok=True,
            payload=payload,
            state_changing=spec.state_changing,
            idempotency_key=call.idempotency_key,
            evidence_id=evidence_id,
            cost_units=cost_units,
        )
        return result

    def _record_call(self, call: ToolCall, sig: str, tick: int) -> None:
        self._recent_calls.append((tick, sig))
        if call.idempotency_key:
            self._idempotency_keys.setdefault(
                call.idempotency_key, _logical_call_signature(call)
            )
    def _episode_budget_exhausted(self) -> bool:
        return self._budget.max_total_tool_calls is not None and (
            self._attempted_total >= self._budget.max_total_tool_calls
        )

    def _idempotency_conflict(self, call: ToolCall) -> tuple[str, str] | None:
        if not call.idempotency_key:
            return None
        current = _logical_call_signature(call)
        prior = self._idempotency_keys.get(call.idempotency_key)
        if prior is None or prior == current:
            return None
        return prior


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _call_signature(call: ToolCall) -> str:
    """Stable signature for duplicate suppression.

    Includes ``idempotency_key`` when set so two calls with the same
    semantics but different intent (different keys) are NOT treated as
    duplicates. This matches the standard idempotency contract where the
    key is the deduplication anchor; absent a key we fall back to
    name+args (any two identical-by-content calls collapse).
    """
    if call.idempotency_key is not None:
        body = f"{call.name}|idem={call.idempotency_key}"
    else:
        body = f"{call.name}|" + "|".join(
            f"{k}={call.args[k]}" for k in sorted(call.args)
        )
    return hashlib.sha1(body.encode()).hexdigest()


def _cooldown_key(call: ToolCall) -> str:
    """Key for post-failure cooldown: tool name + args, ignoring the
    ``idempotency_key``.

    A call that just failed should not be retried immediately with the
    *same* arguments, but a *different* valid call to the same tool must
    not be blocked. Keying cooldown on name-only (the historical behaviour)
    froze an entire state-changing tool after a single bad argument, which
    collapsed families like ``job_shop_dispatch`` to the wait-floor once any
    call mis-formatted a ``job_id``. Ignoring ``idempotency_key`` here means
    a semantic retry (same args, new key) is still cooled down.
    """
    body = f"{call.name}|" + "|".join(f"{k}={call.args[k]}" for k in sorted(call.args))
    return hashlib.sha1(body.encode()).hexdigest()


def _logical_call_signature(call: ToolCall) -> tuple[str, str]:
    return (
        call.name,
        json.dumps(call.args, sort_keys=True, separators=(",", ":"), default=str),
    )


def _episode_budget_exhausted_result() -> ToolResult:
    return ToolResult(
        name="__budget_exhausted__",
        ok=False,
        error_code="EPISODE_BUDGET_EXHAUSTED",
        error_message=(
            "Episode-wide tool budget exhausted; remaining calls this tick were skipped."
        ),
    )


def _tick_budget_exhausted_result(name: str, budget: TickBudget) -> ToolResult:
    return ToolResult(
        name=name,
        ok=False,
        error_code="TICK_BUDGET_EXHAUSTED",
        error_message=(
            "per-tick tool budget exhausted "
            f"(max={budget.max_tool_calls_per_tick})"
        ),
    )


def _pop_payload_evidence_id(payload: dict[str, Any]) -> str | None:
    """Lift handler-emitted evidence id into the ToolResult metadata field."""
    if not isinstance(payload, dict):
        return None
    evidence_id = payload.pop("evidence_id", None)
    return evidence_id if isinstance(evidence_id, str) and evidence_id else None


def _no_effect_error_code(
    call: ToolCall,
    spec: ToolSpec,
    payload: dict[str, Any],
) -> str | None:
    if not spec.state_changing or call.name in {"wait", "noop"}:
        return None
    if not isinstance(payload, dict):
        return None
    status = payload.get("_status")
    if not isinstance(status, str):
        return None
    normalized = status.strip().lower().replace("-", "_")
    if normalized.startswith(("unsupported", "not_supported")):
        return "UNSUPPORTED_TOOL_EFFECT"
    if normalized.startswith("out_of_range"):
        return "OUT_OF_RANGE_TOOL_EFFECT"
    if normalized in {"noop", "no_op", "no_effect", "ignored", "skipped"}:
        return "NO_EFFECT"
    return None


def _payload_failure_message(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return "state-changing tool produced no effect"
    return str(
        payload.get("error")
        or payload.get("reason")
        or payload.get("info")
        or "state-changing tool produced no effect"
    )


def sanitize_openai_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """Make JSON Schema safe for strict OpenAI / Gemini tool APIs.

    - ``exclusiveMinimum`` / ``exclusiveMaximum`` → ``minimum`` / ``maximum``
    - Arrays without ``items`` get a generic item schema
    """

    def _walk(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        out = copy.deepcopy(node)
        if out.get("type") == "array" and "items" not in out:
            out["items"] = {"type": "string"}
        if "exclusiveMinimum" in out:
            base = out.pop("exclusiveMinimum")
            if isinstance(base, (int, float)):
                out.setdefault("minimum", float(base) + 1e-6)
        if "exclusiveMaximum" in out:
            base = out.pop("exclusiveMaximum")
            if isinstance(base, (int, float)):
                out.setdefault("maximum", float(base) - 1e-6)
        for key, val in list(out.items()):
            if key == "properties" and isinstance(val, dict):
                out[key] = {k: _walk(v) for k, v in val.items()}
            elif key == "items":
                out[key] = _walk(val)
            elif key in {"anyOf", "oneOf", "allOf"} and isinstance(val, list):
                out[key] = [_walk(v) for v in val]
        return out

    return _walk(schema)


def _seeded_uniform(seed: int, tick: int, name: str, idem: str | None) -> float:
    """Deterministic uniform draw based on seed+tick+name+idem."""
    key = f"{seed}|{tick}|{name}|{idem or ''}"
    h = hashlib.sha256(key.encode()).digest()
    n = int.from_bytes(h[:8], "big", signed=False)
    return n / float(1 << 64)


def _validate_args(
    args: dict[str, Any], schema: dict[str, Any]
) -> tuple[bool, str | None]:
    """Validate tool arguments against the supported JSON-Schema subset.

    The core intentionally avoids a runtime ``jsonschema`` dependency, but
    dynamic ``enum`` values are part of the benchmark's fog-of-war boundary.
    Validation therefore fails closed for objects that declare ``properties``
    and recursively enforces types, enums, numeric bounds, and array items.
    A deliberately open object can omit/empty ``properties`` or set
    ``additionalProperties: true``.
    """

    def _path(parent: str, child: str) -> str:
        return f"{parent}.{child}" if parent else child

    def _validate(value: Any, node: dict[str, Any], path: str) -> str | None:
        expected = node.get("type")
        label = path or "arguments"

        if expected == "string" and not isinstance(value, str):
            return f"{label} must be string"
        if expected == "integer" and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            return f"{label} must be integer"
        if expected == "number" and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            return f"{label} must be number"
        if expected == "boolean" and not isinstance(value, bool):
            return f"{label} must be boolean"
        if expected == "array" and not isinstance(value, list):
            return f"{label} must be array"
        if expected == "object" and not isinstance(value, dict):
            return f"{label} must be object"

        if "enum" in node and value not in node["enum"]:
            return f"{label} must be one of {node['enum']}"

        if expected in {"integer", "number"}:
            try:
                numeric = float(value)
            except OverflowError:
                return f"{label} must be finite"
            if not math.isfinite(numeric):
                return f"{label} must be finite"
            if "minimum" in node and value < node["minimum"]:
                return f"{label} must be >= {node['minimum']}"
            if "maximum" in node and value > node["maximum"]:
                return f"{label} must be <= {node['maximum']}"
            if "exclusiveMinimum" in node and value <= node["exclusiveMinimum"]:
                return f"{label} must be > {node['exclusiveMinimum']}"
            if "exclusiveMaximum" in node and value >= node["exclusiveMaximum"]:
                return f"{label} must be < {node['exclusiveMaximum']}"

        if expected == "array":
            if "minItems" in node and len(value) < int(node["minItems"]):
                return f"{label} must contain at least {node['minItems']} items"
            if "maxItems" in node and len(value) > int(node["maxItems"]):
                return f"{label} must contain at most {node['maxItems']} items"
            item_schema = node.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    error = _validate(item, item_schema, f"{label}[{index}]")
                    if error:
                        return error

        if expected == "object" or (
            isinstance(value, dict)
            and any(
                keyword in node
                for keyword in ("properties", "required", "additionalProperties")
            )
        ):
            properties = node.get("properties")
            required = node.get("required") or []
            for key in required:
                if key not in value:
                    return f"missing required parameter: {_path(path, str(key))}"
            if isinstance(properties, dict) or "additionalProperties" in node:
                known_properties = properties if isinstance(properties, dict) else {}
                # ``properties: {}`` is used by legacy test/meta tools as an
                # intentionally open payload. Schemas that enumerate at least
                # one property are closed unless they explicitly opt back in.
                additional = node.get(
                    "additionalProperties", not bool(known_properties)
                )
                for key, item in value.items():
                    child_path = _path(path, str(key))
                    child_schema = known_properties.get(key)
                    if isinstance(child_schema, dict):
                        error = _validate(item, child_schema, child_path)
                        if error:
                            return error
                    elif additional is False:
                        return f"{child_path} is not allowed"
                    elif isinstance(additional, dict):
                        error = _validate(item, additional, child_path)
                        if error:
                            return error

        all_of = node.get("allOf")
        if isinstance(all_of, list):
            for branch in all_of:
                if isinstance(branch, dict):
                    error = _validate(value, branch, path)
                    if error:
                        return f"{label} must satisfy allOf: {error}"

        any_of = node.get("anyOf")
        if isinstance(any_of, list):
            branch_errors = [
                _validate(value, branch, path)
                for branch in any_of
                if isinstance(branch, dict)
            ]
            if not branch_errors or all(error is not None for error in branch_errors):
                return f"{label} must satisfy at least one anyOf branch"

        one_of = node.get("oneOf")
        if isinstance(one_of, list):
            matches = sum(
                _validate(value, branch, path) is None
                for branch in one_of
                if isinstance(branch, dict)
            )
            if matches != 1:
                return f"{label} must satisfy exactly one oneOf branch"
        return None

    if not isinstance(schema, dict):
        return True, None
    error = _validate(args, schema, "")
    return error is None, error
