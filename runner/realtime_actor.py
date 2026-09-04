"""Single-writer environment actor for the separate soft real-time treatment.

The deterministic logical runner remains authoritative for the frozen scorer.
This actor lets simulator time advance while a provider turn is in flight and
keeps every state-changing command behind a compare-and-swap action lifecycle.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from core import Action, ToolCall


def _agent_visible_evidence_ids(
    *,
    realized_events: list[dict[str, Any]],
    step_evidence_ids: list[Any],
    tool_results: list[dict[str, Any]],
) -> list[str]:
    """Fail closed when a step mixes visible and hidden evidence."""

    tool_ids = {
        str(evidence_id)
        for result in tool_results
        for evidence_id in [
            result.get("evidence_id"),
            *(result.get("produces_evidence_ids") or []),
        ]
        if evidence_id
    }
    visible_event_ids = {
        str(evidence_id)
        for event in realized_events
        if event.get("hidden") is not True
        for evidence_id in [
            event.get("evidence_id"),
            *(event.get("evidence_ids") or []),
        ]
        if evidence_id
    }
    if not any(event.get("hidden") is True for event in realized_events):
        allowed = {
            str(value) for value in step_evidence_ids if value
        } | visible_event_ids
    else:
        allowed = tool_ids | visible_event_ids
    ordered = [
        str(value)
        for value in step_evidence_ids
        if value and str(value) in allowed
    ]
    for evidence_id in sorted(tool_ids):
        if evidence_id not in ordered:
            ordered.append(evidence_id)
    for evidence_id in sorted(visible_event_ids):
        if evidence_id not in ordered:
            ordered.append(evidence_id)
    return ordered


def _action_effect_closure(
    *,
    action: Action,
    tool_results: list[dict[str, Any]],
    realized_events: list[dict[str, Any]],
    evidence_ledger: list[dict[str, Any]],
    request_tick: int,
    simulator_tick: int,
) -> tuple[bool, bool, list[str], list[dict[str, Any]]]:
    """Join control acknowledgements to native, evidence-backed effects."""

    action_call_ids = {
        str(call.call_id)
        for call in action.tool_calls
        if call.call_id is not None
    }
    action_calls_by_name: dict[str, list[Any]] = {}
    for call in action.tool_calls:
        action_calls_by_name.setdefault(call.name, []).append(call)
    confirmed_result_indexes: set[int] = set()
    confirmed_call_ids = {
        str(result.get("call_id"))
        for result in tool_results
        if result.get("ok") is True
        and result.get("state_changing") is True
        and str((result.get("payload") or {}).get("_status") or "").lower()
        != "pending"
        and result.get("call_id") is not None
        and str(result.get("call_id")) in action_call_ids
    }
    for index, result in enumerate(tool_results):
        if result.get("ok") is not True or result.get("state_changing") is not True:
            continue
        call_id = str(result.get("call_id") or "")
        if call_id in confirmed_call_ids or (
            not call_id
            and len(action_calls_by_name.get(str(result.get("name") or ""), []))
            == 1
        ):
            confirmed_result_indexes.add(index)
    calls_by_id = {
        str(call.call_id): call
        for call in action.tool_calls
        if call.call_id is not None
    }
    realized_evidence = [
        item
        for item in evidence_ledger
        if isinstance(item, dict)
        and item.get("kind") == "realized_event"
        and item.get("source") == "engine"
        and isinstance(item.get("payload"), dict)
    ]
    non_effect_statuses = {
        "canceled",
        "cancelled",
        "error",
        "expired",
        "failed",
        "rejected",
        "superseded",
    }

    def blocks_effect(value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        for key in ("_status", "status"):
            status = value.get(key)
            if not isinstance(status, str):
                continue
            tokens = {
                token
                for token in status.strip().lower().replace("-", "_").split("_")
                if token
            }
            if tokens.intersection(non_effect_statuses):
                return True
        return blocks_effect(value.get("applied_action"))

    effect_evidence_by_call: dict[str, list[str]] = {}
    for event in realized_events:
        if (
            not isinstance(event, dict)
            or str(event.get("origin") or "") != "agent_caused"
            or not str(event.get("event_id") or "")
        ):
            continue
        call_id = str(event.get("call_id") or "")
        if call_id not in confirmed_call_ids:
            continue
        call = calls_by_id.get(call_id)
        if call is None:
            continue
        for item in realized_evidence:
            payload = item["payload"]
            requested_action = payload.get("requested_action") or {}
            if isinstance(requested_action, dict):
                event_tool_name = str(
                    requested_action.get("name") or payload.get("tool_name") or ""
                )
                event_args = requested_action.get("args", requested_action)
            else:
                event_tool_name = str(payload.get("tool_name") or "")
                event_args = None
            before_digest = str(payload.get("before_state_digest") or "")
            after_digest = str(payload.get("after_state_digest") or "")
            raw_item_tick = item.get("tick")
            raw_effect_tick = payload.get("effect_tick", raw_item_tick)
            try:
                item_tick = int(raw_item_tick)
                effect_tick = int(raw_effect_tick)
            except (OverflowError, TypeError, ValueError):
                continue
            if (
                isinstance(raw_item_tick, bool)
                or isinstance(raw_effect_tick, bool)
                or (isinstance(raw_item_tick, float) and raw_item_tick != item_tick)
                or (isinstance(raw_effect_tick, float) and raw_effect_tick != effect_tick)
                or str(payload.get("origin") or "") != "agent_caused"
                or str(payload.get("call_id") or "") != call_id
                or event_tool_name != call.name
                or not isinstance(event_args, dict)
                or dict(event_args) != dict(call.args)
                or not before_digest
                or not after_digest
                or before_digest == after_digest
                or blocks_effect(payload)
                or item_tick < request_tick
                or effect_tick < item_tick
                or effect_tick > simulator_tick
            ):
                continue
            if any(
                event.get(key) != payload.get(key)
                for key in (
                    "event_id",
                    "call_id",
                    "tool_name",
                    "requested_action",
                    "before_state_digest",
                    "after_state_digest",
                    "tick",
                    "effect_tick",
                    "outcome_tick",
                )
                if key in event or key in payload
            ):
                continue
            evidence_id = str(item.get("evidence_id") or "")
            if evidence_id:
                effect_evidence_by_call.setdefault(call_id, []).append(evidence_id)

    edges: list[dict[str, Any]] = []
    for index, result in enumerate(tool_results):
        call_id = str(result.get("call_id") or "")
        effect_ids = list(dict.fromkeys(effect_evidence_by_call.get(call_id, [])))
        edges.append(
            {
                "index": index,
                "call_id": call_id or None,
                "tool_name": str(result.get("name") or ""),
                "state_changing": result.get("state_changing") is True,
                "confirmed": index in confirmed_result_indexes,
                "effect_proven": bool(effect_ids),
                "effect_evidence_ids": effect_ids,
            }
        )
    effect_evidence_ids = list(
        dict.fromkeys(
            evidence_id
            for evidence_ids in effect_evidence_by_call.values()
            for evidence_id in evidence_ids
        )
    )
    return (
        bool(confirmed_result_indexes),
        bool(effect_evidence_ids),
        effect_evidence_ids,
        edges,
    )


def _deferred_materialization_tick(
    result: dict[str, Any],
    *,
    request_tick: int,
    simulator_tick: int,
) -> int:
    """Return the causal tick at which a deferred result became available."""

    def exact_tick(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            tick = int(value)
        except (OverflowError, TypeError, ValueError):
            return None
        if isinstance(value, float) and value != tick:
            return None
        return tick

    payload = result.get("payload") or {}
    due_tick = exact_tick(
        payload.get("due_tick") if isinstance(payload, dict) else None
    )
    if due_tick is not None and due_tick >= request_tick:
        return due_tick
    latency_ticks = exact_tick(result.get("latency_ticks"))
    if latency_ticks is not None and latency_ticks > 0:
        return request_tick + latency_ticks
    effect_tick = exact_tick(result.get("effect_tick"))
    if effect_tick is not None and effect_tick >= request_tick:
        return effect_tick
    return simulator_tick


def _uncredited_tool_trace_edges(
    tool_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Preserve terminal call provenance after an execution fence."""

    return [
        {
            "index": index,
            "call_id": str(result.get("call_id") or "") or None,
            "tool_name": str(result.get("name") or ""),
            "state_changing": result.get("state_changing") is True,
            "confirmed": False,
            "effect_proven": False,
            "effect_evidence_ids": [],
        }
        for index, result in enumerate(tool_results)
    ]


def _merge_tool_trace_edges(
    existing: list[dict[str, Any]],
    terminal: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one latest lifecycle edge per call while preserving call order."""

    merged = deepcopy(existing)
    positions = {str(row.get("call_id") or ""): index for index, row in enumerate(merged)}
    for row in terminal:
        call_id = str(row.get("call_id") or "")
        if call_id and call_id in positions:
            merged[positions[call_id]] = deepcopy(row)
        else:
            positions[call_id] = len(merged)
            merged.append(deepcopy(row))
    return merged


@dataclass(frozen=True)
class SafetyDecision:
    """A safety arbitration or fallback decision with auditable evidence."""

    action: Action
    mode: str
    reason_code: str
    supervisor_id: str = "default_safety_supervisor"
    disposition: Literal["pass", "override", "reject"] = "override"
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.to_dict()
        return payload


class SafetySupervisor(Protocol):
    def arbitrate(
        self,
        *,
        observation: dict[str, Any],
        simulator_tick: int,
        candidate_action: Action,
    ) -> SafetyDecision: ...

    def decide(
        self,
        *,
        observation: dict[str, Any],
        simulator_tick: int,
        reason: str,
    ) -> SafetyDecision: ...


class HoldSafetySupervisor:
    """Domain-neutral, explicitly labelled controlled-hold policy.

    Domain tracks such as autonomous driving should inject their native shield.
    """

    def arbitrate(
        self,
        *,
        observation: dict[str, Any],
        simulator_tick: int,
        candidate_action: Action,
    ) -> SafetyDecision:
        del observation, simulator_tick
        return SafetyDecision(
            action=deepcopy(candidate_action),
            mode="domain_neutral_pass",
            reason_code="NO_DOMAIN_SHIELD_CONFIGURED",
            supervisor_id="domain_neutral_hold",
            disposition="pass",
        )

    def decide(
        self,
        *,
        observation: dict[str, Any],
        simulator_tick: int,
        reason: str,
    ) -> SafetyDecision:
        del observation
        return SafetyDecision(
            action=Action(
                tool_calls=[
                    ToolCall(
                        name="wait",
                        idempotency_key=f"realtime_safety_hold_t{simulator_tick}",
                    )
                ],
                dominant="controlled_hold",
            ),
            mode="controlled_hold",
            reason_code=reason,
            supervisor_id="domain_neutral_hold",
            disposition="override",
        )


@dataclass(frozen=True)
class RealtimeActionReceipt:
    action_id: str
    decision_id: str
    turn_id: str
    status: str
    reason_code: str | None
    based_on_state_version: int
    accepted_state_version: int | None
    request_tick: int
    valid_from_tick: int
    expires_at_tick: int | None
    applied_tick: int | None
    supersedes_action_id: str | None
    idempotency_key: str
    lifecycle_seq: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _PendingSubmission:
    sequence: int
    action_id: str
    decision_id: str
    turn_id: str
    action: Action
    based_on_state_version: int
    request_tick: int
    valid_from_tick: int
    expires_at_tick: int | None
    supersedes_action_id: str | None
    idempotency_key: str
    based_on_visible_evidence_ids: tuple[str, ...]
    future: Future[RealtimeActionReceipt]


@dataclass
class _DeferredSubmission:
    submission: _PendingSubmission
    pending_call_ids: set[str]
    terminal_results: list[dict[str, Any]]
    control_confirmed: bool = False
    effect_observed: bool = False
    effect_evidence_ids: tuple[str, ...] = ()
    tool_trace_edges: list[dict[str, Any]] = field(default_factory=list)
    applied_tick: int | None = None
    fenced_status: str | None = None
    fenced_reason_code: str | None = None


@dataclass
class _DeferredFenceRequest:
    target_action_id: str
    target_call_ids: tuple[str, ...]
    requesting_submission: _PendingSubmission


class RealtimeEnvironmentActor:
    """Advance one environment on a soft real-time monotonic clock.

    Only this thread calls ``env.step``. Model submissions carry the exact
    observation version they were based on; stale, canceled, superseded,
    expired, and duplicate submissions are recorded but never executed. A
    read-only investigation also runs on this thread because the current
    environment contract mutates tool budgets and evidence ledgers; its wall
    time is therefore audited as an explicit clock stall.
    """

    def __init__(
        self,
        env: Any,
        *,
        tick_interval_s: float,
        safety_supervisor: SafetySupervisor | None = None,
    ) -> None:
        if not math.isfinite(tick_interval_s) or tick_interval_s < 1e-9:
            raise ValueError("tick_interval_s must be finite and at least 1ns")
        self._env = env
        self._tick_interval_s = float(tick_interval_s)
        self._safety_supervisor = safety_supervisor or HoldSafetySupervisor()
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state_version = 0
        self._observation: dict[str, Any] = {}
        self._pending: list[_PendingSubmission] = []
        self._submission_sequence = 0
        self._lifecycle_sequence = 0
        self._done = False
        self._transitions: list[dict[str, Any]] = []
        self._receipts: list[dict[str, Any]] = []
        self._lifecycle: list[dict[str, Any]] = []
        self._future_completions: list[
            tuple[Future[RealtimeActionReceipt], RealtimeActionReceipt]
        ] = []
        self._settled_sequences: set[int] = set()
        self._idempotency_keys: set[str] = set()
        self._cumulative_reward = 0.0
        self._authoritative_evidence_ids: list[str] = []
        self._clock_dispatch_seq = 0
        self._active_clock_dispatch: dict[str, int] | None = None
        self._active_submission: _PendingSubmission | None = None
        self._deferred_submissions: dict[str, _DeferredSubmission] = {}
        self._deferred_action_by_call_id: dict[str, str] = {}
        self._deferred_fence_requests: list[_DeferredFenceRequest] = []
        self._fatal_error_type: str | None = None
        self._fatal_error_stage: str | None = None

    @property
    def done(self) -> bool:
        with self._condition:
            return self._done

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("environment actor already started")
            initial_observation = self._env.snapshot()
            if not isinstance(initial_observation, dict):
                raise TypeError("initial environment snapshot must be a mapping")
            initial_tick = initial_observation.get("tick")
            if not isinstance(initial_tick, int) or isinstance(initial_tick, bool):
                raise ValueError("initial environment tick must be integer zero")
            if initial_tick != 0:
                raise ValueError("initial environment tick must equal zero")
            self._observation = deepcopy(initial_observation)
            self._thread = threading.Thread(
                target=self._run,
                name="dt-sched-realtime-environment",
                daemon=True,
            )
            self._thread.start()

    def snapshot(self) -> tuple[int, dict[str, Any]]:
        with self._condition:
            return self._state_version, deepcopy(self._observation)

    def try_admit_provider_turn(
        self, start: Callable[[], None]
    ) -> Literal[
        "admitted",
        "clock_dispatch_in_flight",
        "environment_done",
        "environment_stopped",
    ]:
        """Atomically admit non-blocking provider work or return why it cannot.

        The method never waits for ``env.step``. The coordinator can therefore
        keep enforcing its wall deadline while an unusually slow backend step
        is in flight, then retry the queued event after the transition settles.
        """

        with self._condition:
            if self._done:
                return "environment_done"
            if self._stop.is_set():
                return "environment_stopped"
            if self._active_clock_dispatch is not None:
                return "clock_dispatch_in_flight"
            start()
            return "admitted"

    def submit(
        self,
        action: Action,
        *,
        action_id: str,
        based_on_state_version: int,
        decision_id: str | None = None,
        turn_id: str | None = None,
        valid_from_tick: int | None = None,
        expires_at_tick: int | None = None,
        supersedes_action_id: str | None = None,
        idempotency_key: str | None = None,
        based_on_visible_evidence_ids: list[str] | tuple[str, ...] | None = None,
    ) -> Future[RealtimeActionReceipt]:
        with self._condition:
            future = self._submit_locked(
                action,
                action_id=action_id,
                decision_id=decision_id or action_id,
                turn_id=turn_id or action_id,
                based_on_state_version=based_on_state_version,
                valid_from_tick=(
                    int(valid_from_tick)
                    if valid_from_tick is not None
                    else self._simulator_tick_locked()
                ),
                expires_at_tick=expires_at_tick,
                supersedes_action_id=supersedes_action_id,
                idempotency_key=idempotency_key or action_id,
                based_on_visible_evidence_ids=(
                    based_on_visible_evidence_ids
                    if based_on_visible_evidence_ids is not None
                    else tuple(self._observation.get("__last_evidence_ids__") or [])
                ),
            )
        self._drain_future_completions()
        return future

    def submit_current(
        self,
        action: Action,
        *,
        action_id: str,
        decision_id: str | None = None,
        turn_id: str | None = None,
        valid_from_tick: int | None = None,
        expires_at_tick: int | None = None,
        supersedes_action_id: str | None = None,
        idempotency_key: str | None = None,
        based_on_visible_evidence_ids: list[str] | tuple[str, ...] | None = None,
    ) -> Future[RealtimeActionReceipt]:
        with self._condition:
            future = self._submit_locked(
                action,
                action_id=action_id,
                decision_id=decision_id or action_id,
                turn_id=turn_id or action_id,
                based_on_state_version=self._state_version,
                valid_from_tick=(
                    int(valid_from_tick)
                    if valid_from_tick is not None
                    else self._simulator_tick_locked()
                ),
                expires_at_tick=expires_at_tick,
                supersedes_action_id=supersedes_action_id,
                idempotency_key=idempotency_key or action_id,
                based_on_visible_evidence_ids=(
                    based_on_visible_evidence_ids
                    if based_on_visible_evidence_ids is not None
                    else tuple(self._observation.get("__last_evidence_ids__") or [])
                ),
            )
        self._drain_future_completions()
        return future

    def _submit_locked(
        self,
        action: Action,
        *,
        action_id: str,
        decision_id: str,
        turn_id: str,
        based_on_state_version: int,
        valid_from_tick: int,
        expires_at_tick: int | None,
        supersedes_action_id: str | None,
        idempotency_key: str,
        based_on_visible_evidence_ids: list[str] | tuple[str, ...],
    ) -> Future[RealtimeActionReceipt]:
        future: Future[RealtimeActionReceipt] = Future()
        self._submission_sequence += 1
        submission = _PendingSubmission(
            sequence=self._submission_sequence,
            action_id=str(action_id),
            decision_id=str(decision_id),
            turn_id=str(turn_id),
            action=deepcopy(action),
            based_on_state_version=int(based_on_state_version),
            request_tick=self._simulator_tick_locked(),
            valid_from_tick=int(valid_from_tick),
            expires_at_tick=(int(expires_at_tick) if expires_at_tick is not None else None),
            supersedes_action_id=(
                str(supersedes_action_id) if supersedes_action_id is not None else None
            ),
            idempotency_key=str(idempotency_key),
            based_on_visible_evidence_ids=tuple(
                str(value) for value in based_on_visible_evidence_ids if value
            ),
            future=future,
        )
        if self._done or self._stop.is_set():
            self._settle_locked(submission, status="failed", reason="ENVIRONMENT_CLOSED")
            return future
        if submission.idempotency_key in self._idempotency_keys:
            self._settle_locked(
                submission,
                status="rejected",
                reason="DUPLICATE_IDEMPOTENCY_KEY",
            )
            return future
        if (
            submission.expires_at_tick is not None
            and submission.expires_at_tick < submission.valid_from_tick
        ):
            self._settle_locked(
                submission,
                status="rejected",
                reason="INVALID_VALIDITY_WINDOW",
            )
            return future
        self._idempotency_keys.add(submission.idempotency_key)
        if submission.supersedes_action_id:
            deferred = self._deferred_submissions.get(
                submission.supersedes_action_id,
            )
            if deferred is not None:
                self._deferred_fence_requests.append(
                    _DeferredFenceRequest(
                        target_action_id=deferred.submission.action_id,
                        target_call_ids=tuple(sorted(deferred.pending_call_ids)),
                        requesting_submission=submission,
                    )
                )
            retained: list[_PendingSubmission] = []
            for pending in self._pending:
                if pending.action_id == submission.supersedes_action_id:
                    self._settle_locked(
                        pending,
                        status="superseded",
                        reason="EXPLICIT_SUPERSESSION",
                    )
                else:
                    retained.append(pending)
            self._pending = retained
        self._pending.append(submission)
        self._record_lifecycle_locked(submission, status="queued", reason=None)
        self._condition.notify_all()
        return future

    def wait_for_version(self, version: int, *, timeout_s: float) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while self._state_version < int(version) and not self._done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return self._state_version >= int(version)

    def wait_for_transition_count(self, count: int, *, timeout_s: float) -> bool:
        """Wait for append-only actor activity, including non-advancing failures."""

        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while len(self._transitions) < int(count) and not self._done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return len(self._transitions) >= int(count)

    def wait_until_done(self, *, timeout_s: float) -> bool:
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while not self._done:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def transition_records(self) -> list[dict[str, Any]]:
        with self._condition:
            return deepcopy(self._transitions)

    def receipt_records(self) -> list[dict[str, Any]]:
        """Return one terminal receipt per submitted action."""
        with self._condition:
            return deepcopy(self._receipts)

    def lifecycle_records(self) -> list[dict[str, Any]]:
        """Return append-only queued/accepted/terminal action transitions."""
        with self._condition:
            return deepcopy(self._lifecycle)

    def episode_summary(self) -> dict[str, Any]:
        with self._condition:
            return {
                "final_state_version": self._state_version,
                "final_observation": deepcopy(self._observation),
                "cumulative_reward": float(self._cumulative_reward),
                "authoritative_evidence_ids": list(self._authoritative_evidence_ids),
                "done": bool(self._done),
                "actor_failure": (
                    {
                        "error_type": self._fatal_error_type,
                        "stage": self._fatal_error_stage,
                    }
                    if self._fatal_error_type is not None
                    else None
                ),
            }

    def fatal_error(self) -> dict[str, str] | None:
        """Return a sanitized actor-loop failure, if one occurred."""

        with self._condition:
            if self._fatal_error_type is None:
                return None
            return {
                "error_type": self._fatal_error_type,
                "stage": str(self._fatal_error_stage or "unknown"),
            }

    @property
    def stopped(self) -> bool:
        """Whether the actor thread has conclusively stopped."""

        thread = self._thread
        return thread is None or not thread.is_alive()

    def stop(self, *, timeout_s: float = 2.0) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                raise TimeoutError("environment actor did not stop")

    def _simulator_tick_locked(self) -> int:
        try:
            return int(self._observation.get("tick", 0))
        except (TypeError, ValueError):
            return 0

    def _is_investigation_action(self, action: Action) -> bool:
        if not action.tool_calls:
            return False
        registry = getattr(self._env, "_tools", None)
        semantic_roles = getattr(registry, "semantic_roles", None)
        if callable(semantic_roles):
            roles = semantic_roles()
            return all(roles.get(call.name) == "investigation" for call in action.tool_calls)
        readonly_names = getattr(self._env, "readonly_tool_names", None)
        readonly = readonly_names() if callable(readonly_names) else None
        if not readonly:
            return False
        return all(
            call.name in readonly and call.name not in {"wait", "noop", "commit_to_plan"}
            for call in action.tool_calls
        )

    def _nominal_action_semantics(
        self, action: Action | None
    ) -> tuple[list[dict[str, str]], bool]:
        if action is None:
            return [], False
        registry = getattr(self._env, "_tools", None)
        semantic_roles = getattr(registry, "semantic_roles", None)
        roles = semantic_roles() if callable(semantic_roles) else {}
        semantics = [
            {
                "name": call.name,
                "semantic_role": str(roles.get(call.name) or "unknown"),
            }
            for call in action.tool_calls
        ]
        deliberate_wait = bool(action.tool_calls) and all(
            call.name in {"wait", "noop"} for call in action.tool_calls
        )
        if deliberate_wait or not action.tool_calls:
            return semantics, False
        if roles:
            return semantics, any(
                item["semantic_role"] in {"control", "unknown"}
                for item in semantics
            )
        return semantics, not self._is_investigation_action(action)

    def _uncancellable_managed_delay_calls_locked(
        self,
        action: Action,
    ) -> list[str]:
        registry = getattr(self._env, "_tools", None)
        get_spec = getattr(registry, "get", None)
        resolve_imperfection = getattr(registry, "resolve_imperfection", None)
        if not callable(get_spec) or not callable(resolve_imperfection):
            return []
        rejected: list[str] = []
        for call in action.tool_calls:
            spec = get_spec(call.name)
            if spec is None:
                continue
            imperfection = resolve_imperfection(call.name)
            delay_ticks = int((imperfection or {}).get("delay_ticks", 0))
            if (
                spec.state_changing is True
                and spec.handler_manages_delay is True
                and delay_ticks > 0
                and not callable(spec.cancel_pending)
            ):
                rejected.append(str(call.call_id or call.name))
        return rejected

    def _has_current_investigation_locked(self) -> bool:
        simulator_tick = self._simulator_tick_locked()
        return any(
            submission.based_on_state_version == self._state_version
            and submission.valid_from_tick <= simulator_tick
            and (
                submission.expires_at_tick is None
                or simulator_tick <= submission.expires_at_tick
            )
            and self._is_investigation_action(submission.action)
            for submission in self._pending
        )

    def _arbitrate_locked(
        self,
        candidate_action: Action,
    ) -> SafetyDecision:
        arbitrate = getattr(self._safety_supervisor, "arbitrate", None)
        if not callable(arbitrate):
            return SafetyDecision(
                action=deepcopy(candidate_action),
                mode="legacy_supervisor_pass",
                reason_code="ARBITRATION_INTERFACE_UNAVAILABLE",
                supervisor_id=(
                    f"{type(self._safety_supervisor).__module__}."
                    f"{type(self._safety_supervisor).__qualname__}"
                ),
                disposition="pass",
            )
        decision = arbitrate(
            observation=deepcopy(self._observation),
            simulator_tick=self._simulator_tick_locked(),
            candidate_action=deepcopy(candidate_action),
        )
        if not isinstance(decision, SafetyDecision):
            raise TypeError("supervisor returned an invalid arbitration decision")
        if decision.disposition not in {"pass", "override", "reject"}:
            raise ValueError("supervisor returned an invalid arbitration disposition")
        return decision

    def _clock_audit_fields(self) -> dict[str, Any]:
        dispatch = self._active_clock_dispatch
        if dispatch is None:
            return {}
        finished_ns = time.monotonic_ns()
        scheduled_ns = dispatch["scheduled_monotonic_ns"]
        started_ns = dispatch["started_monotonic_ns"]
        interval_ns = int(self._tick_interval_s * 1e9)
        return {
            "clock_dispatch_seq": dispatch["dispatch_seq"],
            "clock_scheduled_monotonic_ns": scheduled_ns,
            "clock_started_monotonic_ns": started_ns,
            "clock_finished_monotonic_ns": finished_ns,
            "clock_schedule_lateness_ns": max(0, started_ns - scheduled_ns),
            "clock_execution_duration_ns": max(0, finished_ns - started_ns),
            "clock_execution_overrun_ns": max(
                0, finished_ns - started_ns - interval_ns
            ),
            "clock_total_lateness_ns": max(
                0, finished_ns - scheduled_ns - interval_ns
            ),
            "clock_catchup_policy": "skip_missed_intervals",
        }

    def _record_skipped_clock_intervals(
        self, *, dispatch_seq: int, skipped_intervals: int
    ) -> None:
        with self._condition:
            transition = next(
                (
                    row
                    for row in reversed(self._transitions)
                    if row.get("clock_dispatch_seq") == dispatch_seq
                ),
                None,
            )
            if transition is not None:
                transition["clock_catchup_skipped_intervals"] = int(
                    skipped_intervals
                )

    def _run(self) -> None:
        deadline = time.monotonic() + self._tick_interval_s
        try:
            while not self._stop.is_set() and not self._done:
                with self._condition:
                    remaining = deadline - time.monotonic()
                    while (
                        remaining > 0
                        and not self._has_current_investigation_locked()
                        and not self._stop.is_set()
                    ):
                        self._condition.wait(timeout=remaining)
                        remaining = deadline - time.monotonic()
                    if self._stop.is_set():
                        break
                    investigation_due = (
                        remaining > 0 and self._has_current_investigation_locked()
                    )
                if investigation_due:
                    self._execute_investigation_once(
                        clock_deadline_monotonic_s=deadline
                    )
                    continue
                with self._condition:
                    self._clock_dispatch_seq += 1
                    dispatch_seq = self._clock_dispatch_seq
                    self._active_clock_dispatch = {
                        "dispatch_seq": dispatch_seq,
                        "scheduled_monotonic_ns": int(deadline * 1e9),
                        "started_monotonic_ns": time.monotonic_ns(),
                    }
                self._advance_once()
                with self._condition:
                    self._active_clock_dispatch = None
                    self._condition.notify_all()
                nominal_next_deadline = deadline + self._tick_interval_s
                finished = time.monotonic()
                if finished > nominal_next_deadline:
                    skipped_intervals = max(
                        1,
                        int(
                            (finished - nominal_next_deadline)
                            // self._tick_interval_s
                        )
                        + 1,
                    )
                    deadline = finished + self._tick_interval_s
                else:
                    skipped_intervals = 0
                    deadline = nominal_next_deadline
                self._record_skipped_clock_intervals(
                    dispatch_seq=dispatch_seq,
                    skipped_intervals=skipped_intervals,
                )
        except Exception as exc:  # noqa: BLE001 - thread boundary must fail closed
            with self._condition:
                self._fatal_error_type = type(exc).__name__
                self._fatal_error_stage = "environment_actor_loop"
                active_submission = self._active_submission
                if (
                    active_submission is not None
                    and active_submission.sequence not in self._settled_sequences
                ):
                    self._settle_locked(
                        active_submission,
                        status="failed",
                        reason="ACTOR_FATAL_ERROR",
                    )
                self._active_submission = None
                self._transitions.append(
                    {
                        **self._clock_audit_fields(),
                        "state_version_before": self._state_version,
                        "state_version_after": self._state_version,
                        "simulator_tick_before": self._simulator_tick_locked(),
                        "simulator_tick": self._simulator_tick_locked(),
                        "simulator_time_advanced": False,
                        "action_id": (
                            active_submission.action_id
                            if active_submission is not None
                            else None
                        ),
                        "decision_id": (
                            active_submission.decision_id
                            if active_submission is not None
                            else None
                        ),
                        "turn_id": (
                            active_submission.turn_id
                            if active_submission is not None
                            else None
                        ),
                        "action_source": "environment_actor",
                        "actor_fatal": True,
                        "actor_fatal_error_type": self._fatal_error_type,
                        "actor_fatal_stage": self._fatal_error_stage,
                        "environment_step_failed": True,
                        "monotonic_ns": time.monotonic_ns(),
                    }
                )
                self._done = True
                self._condition.notify_all()
        finally:
            with self._condition:
                self._done = True
                for submission in self._pending:
                    self._settle_locked(
                        submission,
                        status="failed",
                        reason="ENVIRONMENT_CLOSED",
                    )
                self._pending.clear()
                for deferred in self._deferred_submissions.values():
                    self._settle_locked(
                        deferred.submission,
                        status="failed",
                        reason="DELAYED_ACTION_UNRESOLVED_AT_EPISODE_END",
                    )
                self._deferred_submissions.clear()
                self._deferred_action_by_call_id.clear()
                self._condition.notify_all()
            self._drain_future_completions()

    def _record_lifecycle_locked(
        self,
        submission: _PendingSubmission,
        *,
        status: str,
        reason: str | None,
        applied_tick: int | None = None,
    ) -> dict[str, Any]:
        self._lifecycle_sequence += 1
        record = {
            "lifecycle_seq": self._lifecycle_sequence,
            "action_id": submission.action_id,
            "decision_id": submission.decision_id,
            "turn_id": submission.turn_id,
            "status": status,
            "reason_code": reason,
            "based_on_state_version": submission.based_on_state_version,
            "accepted_state_version": (
                self._state_version if status == "accepted" else None
            ),
            "request_tick": submission.request_tick,
            "valid_from_tick": submission.valid_from_tick,
            "expires_at_tick": submission.expires_at_tick,
            "applied_tick": applied_tick,
            "supersedes_action_id": submission.supersedes_action_id,
            "idempotency_key": submission.idempotency_key,
            "monotonic_ns": time.monotonic_ns(),
        }
        self._lifecycle.append(record)
        return record

    def _receipt(
        self,
        submission: _PendingSubmission,
        *,
        status: str,
        reason: str | None,
        applied_tick: int | None = None,
    ) -> RealtimeActionReceipt:
        self._lifecycle_sequence += 1
        return RealtimeActionReceipt(
            action_id=submission.action_id,
            decision_id=submission.decision_id,
            turn_id=submission.turn_id,
            status=status,
            reason_code=reason,
            based_on_state_version=submission.based_on_state_version,
            accepted_state_version=(
                submission.based_on_state_version
                if status in {"applied", "confirmed", "effected", "no_effect"}
                else None
            ),
            request_tick=submission.request_tick,
            valid_from_tick=submission.valid_from_tick,
            expires_at_tick=submission.expires_at_tick,
            applied_tick=applied_tick,
            supersedes_action_id=submission.supersedes_action_id,
            idempotency_key=submission.idempotency_key,
            lifecycle_seq=self._lifecycle_sequence,
        )

    def _settle_locked(
        self,
        submission: _PendingSubmission,
        *,
        status: str,
        reason: str | None,
        applied_tick: int | None = None,
    ) -> RealtimeActionReceipt:
        if (
            status != "canceled"
            and not submission.future.running()
            and not submission.future.done()
            and not submission.future.set_running_or_notify_cancel()
        ):
            status = "canceled"
            reason = "CALLER_CANCELED"
        receipt = self._receipt(
            submission,
            status=status,
            reason=reason,
            applied_tick=applied_tick,
        )
        if submission.sequence not in self._settled_sequences:
            self._settled_sequences.add(submission.sequence)
            terminal = receipt.to_dict()
            terminal["monotonic_ns"] = time.monotonic_ns()
            self._receipts.append(terminal)
            self._lifecycle.append(dict(terminal))
            if not submission.future.done():
                self._future_completions.append((submission.future, receipt))
        return receipt

    def _drain_future_completions(self) -> None:
        """Resolve receipts only after releasing the actor condition lock."""

        while True:
            with self._condition:
                completions, self._future_completions = (
                    self._future_completions,
                    [],
                )
            if not completions:
                return
            for future, receipt in completions:
                if not future.done():
                    future.set_result(receipt)

    @staticmethod
    def _pending_call_ids(
        *,
        action: Action,
        tool_results: list[dict[str, Any]],
    ) -> set[str]:
        del action
        return {
            str(result["call_id"])
            for result in tool_results
            if result.get("call_id")
            and str((result.get("payload") or {}).get("_status") or "").lower()
            == "pending"
        }

    def _cancel_pending_tool_calls_locked(
        self, call_ids: set[str]
    ) -> list[dict[str, Any]]:
        if self._thread is None or threading.get_ident() != self._thread.ident:
            raise RuntimeError("pending tool cancellation requires the actor thread")
        registry = getattr(self._env, "_tools", None)
        cancel = getattr(registry, "cancel_pending_calls_with_audit", None)
        if not callable(cancel):
            return [
                {
                    "call_id": call_id,
                    "queue_kind": None,
                    "outcome": "registry_unavailable",
                    "callback_invoked": False,
                    "callback_error_type": None,
                }
                for call_id in sorted(call_ids)
            ]
        try:
            return list(cancel(set(call_ids)))
        except Exception as exc:  # noqa: BLE001 - execution fence fails closed
            return [
                {
                    "call_id": call_id,
                    "queue_kind": None,
                    "outcome": "registry_exception",
                    "callback_invoked": False,
                    "callback_error_type": type(exc).__name__,
                }
                for call_id in sorted(call_ids)
            ]

    @staticmethod
    def _cancellation_succeeded(
        call_ids: set[str], audit: list[dict[str, Any]]
    ) -> bool:
        outcomes = {
            str(row.get("call_id") or ""): str(row.get("outcome") or "")
            for row in audit
        }
        return all(outcomes.get(call_id) == "canceled" for call_id in call_ids)

    def _fail_execution_fence_locked(
        self,
        *,
        deferred: _DeferredSubmission,
        audit: list[dict[str, Any]],
        requesting_submission: _PendingSubmission | None,
        simulator_tick: int,
    ) -> None:
        self._done = True
        self._settle_locked(
            deferred.submission,
            status="failed",
            reason="EXECUTION_FENCE_FAILED",
            applied_tick=deferred.applied_tick,
        )
        if requesting_submission is not None:
            self._settle_locked(
                requesting_submission,
                status="failed",
                reason="EXECUTION_FENCE_FAILED",
            )
            self._pending = [
                pending
                for pending in self._pending
                if pending.sequence != requesting_submission.sequence
            ]
        for call_id in deferred.pending_call_ids:
            self._deferred_action_by_call_id.pop(call_id, None)
        self._deferred_submissions.pop(deferred.submission.action_id, None)
        self._transitions.append(
            {
                **self._clock_audit_fields(),
                "state_version_before": self._state_version,
                "state_version_after": self._state_version,
                "simulator_tick_before": simulator_tick,
                "simulator_tick": simulator_tick,
                "simulator_time_advanced": False,
                "action_id": deferred.submission.action_id,
                "decision_id": deferred.submission.decision_id,
                "turn_id": deferred.submission.turn_id,
                "action_source": "environment_actor",
                "rejection_reason": "EXECUTION_FENCE_FAILED",
                "execution_fence_failed": True,
                "cancellation_audit": deepcopy(audit),
                "environment_step_failed": False,
                "monotonic_ns": time.monotonic_ns(),
            }
        )
        self._condition.notify_all()

    def _process_deferred_fence_requests_locked(
        self, *, simulator_tick: int
    ) -> tuple[list[dict[str, Any]], bool]:
        requests, self._deferred_fence_requests = (
            self._deferred_fence_requests,
            [],
        )
        outcomes: list[dict[str, Any]] = []
        for request in requests:
            deferred = self._deferred_submissions.get(request.target_action_id)
            if deferred is None:
                audit = [
                    {
                        "call_id": call_id,
                        "queue_kind": None,
                        "outcome": "already_materialized_or_resolved",
                        "callback_invoked": False,
                        "callback_error_type": None,
                    }
                    for call_id in request.target_call_ids
                ]
                outcomes.append(
                    {
                        "action_id": request.target_action_id,
                        "decision_id": None,
                        "turn_id": None,
                        "status": "already_materialized_or_resolved",
                        "reason_code": "SUPERSESSION_AFTER_MATERIALIZATION",
                        "pending_call_ids": [],
                        "canceled_call_ids": [],
                        "cancellation_audit": audit,
                        "control_confirmed": False,
                        "effect_observed": False,
                        "effect_evidence_ids": [],
                        "tool_trace_edges": [],
                    }
                )
                continue
            audit = self._cancel_pending_tool_calls_locked(
                deferred.pending_call_ids
            )
            if not self._cancellation_succeeded(
                deferred.pending_call_ids, audit
            ):
                self._fail_execution_fence_locked(
                    deferred=deferred,
                    audit=audit,
                    requesting_submission=request.requesting_submission,
                    simulator_tick=simulator_tick,
                )
                return outcomes, True
            canceled_call_ids = {
                str(row["call_id"])
                for row in audit
                if row.get("outcome") == "canceled"
            }
            for call_id in canceled_call_ids:
                deferred.pending_call_ids.discard(call_id)
                self._deferred_action_by_call_id.pop(call_id, None)
            deferred.fenced_status = "superseded"
            deferred.fenced_reason_code = "EXPLICIT_SUPERSESSION"
            self._deferred_submissions.pop(deferred.submission.action_id, None)
            self._settle_locked(
                deferred.submission,
                status="superseded",
                reason="EXPLICIT_SUPERSESSION",
                applied_tick=deferred.applied_tick,
            )
            outcomes.append(
                {
                    "action_id": deferred.submission.action_id,
                    "decision_id": deferred.submission.decision_id,
                    "turn_id": deferred.submission.turn_id,
                    "status": "superseded",
                    "reason_code": "EXPLICIT_SUPERSESSION",
                    "pending_call_ids": [],
                    "canceled_call_ids": sorted(canceled_call_ids),
                    "cancellation_audit": audit,
                    "control_confirmed": deferred.control_confirmed,
                    "effect_observed": deferred.effect_observed,
                    "effect_evidence_ids": list(deferred.effect_evidence_ids),
                    "tool_trace_edges": [],
                }
            )
        return outcomes, False

    def _process_deferred_boundary_locked(
        self,
        *,
        simulator_tick: int,
        expiry_fence_tick: int | None = None,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Fence superseded/expired work before any environment dispatch."""

        outcomes, failed = self._process_deferred_fence_requests_locked(
            simulator_tick=simulator_tick,
        )
        if failed:
            return outcomes, True
        outcomes.extend(
            self._expire_deferred_submissions_locked(
                simulator_tick=simulator_tick,
                expiry_fence_tick=expiry_fence_tick,
            )
        )
        return outcomes, self._done

    def _expire_deferred_submissions_locked(
        self,
        *,
        simulator_tick: int,
        expiry_fence_tick: int | None = None,
    ) -> list[dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        for action_id, deferred in list(self._deferred_submissions.items()):
            expires_at_tick = deferred.submission.expires_at_tick
            fence_tick = simulator_tick
            if (
                expiry_fence_tick is not None
                and not self._is_investigation_action(deferred.submission.action)
            ):
                fence_tick = expiry_fence_tick
            if (
                deferred.fenced_status is not None
                or expires_at_tick is None
                or fence_tick <= expires_at_tick
            ):
                continue
            deferred.fenced_status = "expired"
            deferred.fenced_reason_code = "DELAYED_ACTION_EXPIRED"
            audit = self._cancel_pending_tool_calls_locked(deferred.pending_call_ids)
            if not self._cancellation_succeeded(deferred.pending_call_ids, audit):
                self._fail_execution_fence_locked(
                    deferred=deferred,
                    audit=audit,
                    requesting_submission=None,
                    simulator_tick=simulator_tick,
                )
                return outcomes
            canceled_call_ids = {
                str(row["call_id"])
                for row in audit
                if row.get("outcome") == "canceled"
            }
            for call_id in canceled_call_ids:
                deferred.pending_call_ids.discard(call_id)
                self._deferred_action_by_call_id.pop(call_id, None)
            self._settle_locked(
                deferred.submission,
                status="expired",
                reason="DELAYED_ACTION_EXPIRED",
                applied_tick=deferred.applied_tick,
            )
            outcomes.append(
                {
                    "action_id": deferred.submission.action_id,
                    "decision_id": deferred.submission.decision_id,
                    "turn_id": deferred.submission.turn_id,
                    "status": "expired",
                    "reason_code": "DELAYED_ACTION_EXPIRED",
                    "pending_call_ids": sorted(deferred.pending_call_ids),
                    "canceled_call_ids": sorted(canceled_call_ids),
                    "cancellation_audit": audit,
                    "control_confirmed": deferred.control_confirmed,
                    "effect_observed": deferred.effect_observed,
                    "effect_evidence_ids": list(deferred.effect_evidence_ids),
                    "tool_trace_edges": deepcopy(deferred.tool_trace_edges),
                }
            )
            if not deferred.pending_call_ids:
                self._deferred_submissions.pop(action_id, None)
        return outcomes

    def _resolve_deferred_results_locked(
        self,
        *,
        tool_results: list[dict[str, Any]],
        realized_events: list[dict[str, Any]],
        evidence_ledger: list[dict[str, Any]],
        simulator_tick: int,
    ) -> list[dict[str, Any]]:
        results_by_action: dict[str, list[dict[str, Any]]] = {}
        for result in tool_results:
            call_id = str(result.get("call_id") or "")
            action_id = self._deferred_action_by_call_id.get(call_id)
            if not action_id or str(
                (result.get("payload") or {}).get("_status") or ""
            ).lower() == "pending":
                continue
            results_by_action.setdefault(action_id, []).append(result)

        outcomes: list[dict[str, Any]] = []
        for action_id, terminal_results in results_by_action.items():
            deferred = self._deferred_submissions[action_id]
            if deferred.fenced_status is not None:
                for result in terminal_results:
                    call_id = str(result.get("call_id") or "")
                    deferred.pending_call_ids.discard(call_id)
                    self._deferred_action_by_call_id.pop(call_id, None)
                outcomes.append(
                    {
                        "action_id": deferred.submission.action_id,
                        "decision_id": deferred.submission.decision_id,
                        "turn_id": deferred.submission.turn_id,
                        "status": deferred.fenced_status,
                        "reason_code": deferred.fenced_reason_code,
                        "pending_call_ids": sorted(deferred.pending_call_ids),
                        "control_confirmed": False,
                        "effect_observed": False,
                        "effect_evidence_ids": [],
                        "tool_trace_edges": _uncredited_tool_trace_edges(
                            terminal_results
                        ),
                    }
                )
                if not deferred.pending_call_ids:
                    self._deferred_submissions.pop(action_id, None)
                continue
            expires_at_tick = deferred.submission.expires_at_tick
            materialization_tick = max(
                _deferred_materialization_tick(
                    result,
                    request_tick=deferred.submission.request_tick,
                    simulator_tick=simulator_tick,
                )
                for result in terminal_results
            )
            if (
                expires_at_tick is not None
                and materialization_tick > expires_at_tick
            ):
                for call_id in deferred.pending_call_ids:
                    self._deferred_action_by_call_id.pop(call_id, None)
                self._settle_locked(
                    deferred.submission,
                    status="expired",
                    reason="DELAYED_ACTION_EXPIRED",
                    applied_tick=deferred.applied_tick,
                )
                self._deferred_submissions.pop(action_id, None)
                outcomes.append(
                    {
                        "action_id": deferred.submission.action_id,
                        "decision_id": deferred.submission.decision_id,
                        "turn_id": deferred.submission.turn_id,
                        "status": "expired",
                        "reason_code": "DELAYED_ACTION_EXPIRED",
                        "pending_call_ids": sorted(deferred.pending_call_ids),
                        "control_confirmed": deferred.control_confirmed,
                        "effect_observed": deferred.effect_observed,
                        "effect_evidence_ids": list(
                            deferred.effect_evidence_ids
                        ),
                        "tool_trace_edges": _merge_tool_trace_edges(
                            deferred.tool_trace_edges,
                            _uncredited_tool_trace_edges(terminal_results),
                        ),
                    }
                )
                continue
            (
                control_confirmed,
                effect_observed,
                effect_evidence_ids,
                tool_trace_edges,
            ) = _action_effect_closure(
                action=deferred.submission.action,
                tool_results=terminal_results,
                realized_events=realized_events,
                evidence_ledger=evidence_ledger,
                request_tick=deferred.submission.request_tick,
                simulator_tick=simulator_tick,
            )
            deferred.terminal_results.extend(deepcopy(terminal_results))
            deferred.control_confirmed |= control_confirmed
            deferred.effect_observed |= effect_observed
            deferred.effect_evidence_ids = tuple(
                dict.fromkeys(
                    [*deferred.effect_evidence_ids, *effect_evidence_ids]
                )
            )
            deferred.tool_trace_edges = _merge_tool_trace_edges(
                deferred.tool_trace_edges,
                tool_trace_edges,
            )
            if any(result.get("ok") is True for result in terminal_results):
                deferred.applied_tick = (
                    simulator_tick
                    if deferred.applied_tick is None
                    else deferred.applied_tick
                )
            for result in terminal_results:
                call_id = str(result.get("call_id") or "")
                deferred.pending_call_ids.discard(call_id)
                self._deferred_action_by_call_id.pop(call_id, None)

            terminal_status: str | None = None
            reason: str | None = None
            if not deferred.pending_call_ids:
                any_success = any(
                    result.get("ok") is True
                    for result in deferred.terminal_results
                )
                if not any_success:
                    terminal_status = "failed"
                    reason = "DELAYED_ACTION_FAILED"
                else:
                    terminal_status = (
                        "effected"
                        if deferred.effect_observed
                        else "confirmed"
                        if deferred.control_confirmed
                        else "no_effect"
                    )
                    self._record_lifecycle_locked(
                        deferred.submission,
                        status="applied",
                        reason=None,
                        applied_tick=deferred.applied_tick,
                    )
                self._settle_locked(
                    deferred.submission,
                    status=terminal_status,
                    reason=reason,
                    applied_tick=deferred.applied_tick,
                )
                self._deferred_submissions.pop(action_id, None)
            outcomes.append(
                {
                    "action_id": deferred.submission.action_id,
                    "decision_id": deferred.submission.decision_id,
                    "turn_id": deferred.submission.turn_id,
                    "status": terminal_status or "pending",
                    "pending_call_ids": sorted(deferred.pending_call_ids),
                    "control_confirmed": deferred.control_confirmed,
                    "effect_observed": deferred.effect_observed,
                    "effect_evidence_ids": list(deferred.effect_evidence_ids),
                    "tool_trace_edges": deepcopy(deferred.tool_trace_edges),
                }
            )
        return outcomes

    def _execute_investigation_once(
        self, *, clock_deadline_monotonic_s: float
    ) -> None:
        """Execute a read-only tool batch on the actor thread without a clock tick."""

        early_exit = False
        with self._condition:
            simulator_tick = self._simulator_tick_locked()
            (
                pre_dispatch_deferred_outcomes,
                execution_fence_failed,
            ) = self._process_deferred_boundary_locked(
                simulator_tick=simulator_tick,
            )
            if execution_fence_failed:
                return
            candidates = [
                submission
                for submission in self._pending
                if submission.based_on_state_version == self._state_version
                and submission.valid_from_tick <= simulator_tick
                and (
                    submission.expires_at_tick is None
                    or simulator_tick <= submission.expires_at_tick
                )
                and self._is_investigation_action(submission.action)
            ]
            if not candidates:
                return
            selected = candidates[-1]
            nominal_semantics, model_attempted_state_change = (
                self._nominal_action_semantics(selected.action)
            )
            candidate_sequences = {submission.sequence for submission in candidates}
            self._pending = [
                submission
                for submission in self._pending
                if submission.sequence not in candidate_sequences
            ]
            for submission in candidates[:-1]:
                self._settle_locked(
                    submission,
                    status="superseded",
                    reason="NEWER_VALID_INVESTIGATION",
                )
            if not selected.future.set_running_or_notify_cancel():
                self._settle_locked(
                    selected,
                    status="canceled",
                    reason="CALLER_CANCELED",
                )
                self._condition.notify_all()
                early_exit = True
            visible_evidence = set(selected.based_on_visible_evidence_ids)
            if not early_exit and any(
                str(evidence_id) not in visible_evidence
                for call in selected.action.tool_calls
                for evidence_id in call.consumes_evidence_ids or []
            ):
                self._settle_locked(
                    selected,
                    status="rejected",
                    reason="INVISIBLE_EVIDENCE_REFERENCE",
                )
                self._transitions.append(
                    {
                        "state_version_before": self._state_version,
                        "state_version_after": self._state_version,
                        "simulator_tick_before": simulator_tick,
                        "simulator_tick": simulator_tick,
                        "simulator_time_advanced": False,
                        "action_id": selected.action_id,
                        "decision_id": selected.decision_id,
                        "turn_id": selected.turn_id,
                        "submitted_action": selected.action.to_dict(),
                        "nominal_action": selected.action.to_dict(),
                        "nominal_tool_semantics": nominal_semantics,
                        "model_attempted_state_change": (
                            model_attempted_state_change
                        ),
                        "applied_action": None,
                        "action_source": "model",
                        "rejection_reason": "INVISIBLE_EVIDENCE_REFERENCE",
                        "safety_supervisor_failed": False,
                        "environment_step_failed": False,
                        "monotonic_ns": time.monotonic_ns(),
                    }
                )
                self._condition.notify_all()
                early_exit = True
            if not early_exit:
                try:
                    safety_decision = self._arbitrate_locked(selected.action)
                except Exception as exc:  # noqa: BLE001 - fail closed safety boundary
                    self._settle_locked(
                        selected,
                        status="rejected",
                        reason="SAFETY_ARBITRATION_FAILED",
                    )
                    self._transitions.append(
                        {
                            "state_version_before": self._state_version,
                            "state_version_after": self._state_version,
                            "simulator_tick_before": simulator_tick,
                            "simulator_tick": simulator_tick,
                            "simulator_time_advanced": False,
                            "action_id": selected.action_id,
                            "decision_id": selected.decision_id,
                            "turn_id": selected.turn_id,
                            "submitted_action": selected.action.to_dict(),
                            "nominal_action": selected.action.to_dict(),
                            "nominal_tool_semantics": nominal_semantics,
                            "model_attempted_state_change": (
                                model_attempted_state_change
                            ),
                            "applied_action": None,
                            "action_source": "safety_supervisor",
                            "safety_supervisor_failed": True,
                            "safety_error_type": type(exc).__name__,
                            "environment_step_failed": False,
                            "monotonic_ns": time.monotonic_ns(),
                        }
                    )
                    self._condition.notify_all()
                    early_exit = True
            if not early_exit and safety_decision.disposition != "pass":
                self._settle_locked(
                    selected,
                    status="rejected",
                    reason=f"SAFETY_{safety_decision.disposition.upper()}",
                )
                self._transitions.append(
                    {
                        "state_version_before": self._state_version,
                        "state_version_after": self._state_version,
                        "simulator_tick_before": simulator_tick,
                        "simulator_tick": simulator_tick,
                        "simulator_time_advanced": False,
                        "action_id": selected.action_id,
                        "decision_id": selected.decision_id,
                        "turn_id": selected.turn_id,
                        "submitted_action": selected.action.to_dict(),
                        "nominal_action": selected.action.to_dict(),
                        "nominal_tool_semantics": nominal_semantics,
                        "model_attempted_state_change": (
                            model_attempted_state_change
                        ),
                        "applied_action": None,
                        "action_source": "safety_supervisor",
                        "safety_decision": safety_decision.to_dict(),
                        "safety_arbitration": safety_decision.to_dict(),
                        "safety_evidence_ids": list(safety_decision.evidence_ids),
                        "used_safety_fallback": True,
                        "environment_step_failed": False,
                        "monotonic_ns": time.monotonic_ns(),
                    }
                )
                self._condition.notify_all()
                early_exit = True
            if not early_exit:
                self._record_lifecycle_locked(selected, status="accepted", reason=None)
                request_version = self._state_version
                nominal_action = deepcopy(selected.action)
                action = deepcopy(safety_decision.action)
                self._active_submission = selected
        self._drain_future_completions()
        if early_exit:
            return
        investigation_started_ns = time.monotonic_ns()
        clock_deadline_ns = int(clock_deadline_monotonic_s * 1e9)
        try:
            observation, raw_results = self._env.execute_investigation(action)
        except Exception as exc:  # noqa: BLE001 - tool boundary fails closed
            investigation_ended_ns = time.monotonic_ns()
            investigation_duration_ns = max(
                0, investigation_ended_ns - investigation_started_ns
            )
            with self._condition:
                self._settle_locked(
                    selected,
                    status="failed",
                    reason="INVESTIGATION_EXECUTION_FAILED",
                )
                self._active_submission = None
                self._transitions.append(
                    {
                        "state_version_before": request_version,
                        "state_version_after": request_version,
                        "simulator_tick_before": simulator_tick,
                        "simulator_tick": simulator_tick,
                        "simulator_time_advanced": False,
                        "action_id": selected.action_id,
                        "decision_id": selected.decision_id,
                        "turn_id": selected.turn_id,
                        "nominal_action": nominal_action.to_dict(),
                        "nominal_tool_semantics": nominal_semantics,
                        "model_attempted_state_change": (
                            model_attempted_state_change
                        ),
                        "applied_action": None,
                        "action_source": "model",
                        "safety_arbitration": safety_decision.to_dict(),
                        "safety_evidence_ids": list(safety_decision.evidence_ids),
                        "environment_step_failed": False,
                        "investigation_failed": True,
                        "investigation_error_type": type(exc).__name__,
                        "clock_semantics": "soft_realtime_monotonic_single_writer",
                        "environment_progress_during_investigation": False,
                        "investigation_started_monotonic_ns": investigation_started_ns,
                        "investigation_ended_monotonic_ns": investigation_ended_ns,
                        "investigation_duration_ns": investigation_duration_ns,
                        "investigation_clock_deadline_monotonic_ns": clock_deadline_ns,
                        "investigation_clock_deadline_overrun_ns": max(
                            0, investigation_ended_ns - clock_deadline_ns
                        ),
                        "investigation_elapsed_tick_intervals": int(
                            investigation_duration_ns
                            // int(self._tick_interval_s * 1e9)
                        ),
                        "monotonic_ns": time.monotonic_ns(),
                    }
                )
                self._condition.notify_all()
            self._drain_future_completions()
            return
        investigation_ended_ns = time.monotonic_ns()
        investigation_duration_ns = max(
            0, investigation_ended_ns - investigation_started_ns
        )
        with self._condition:
            tool_results = [
                item.to_dict() if hasattr(item, "to_dict") else deepcopy(item)
                for item in raw_results
            ]
            returned_tick = observation.get("tick")
            self._state_version += 1
            self._observation = deepcopy(observation)
            # A staged/read-only backend may return its last completed backend
            # tick. Investigation is non-advancing, so the actor clock remains
            # authoritative and must never regress to that snapshot tick.
            self._observation["tick"] = simulator_tick
            self._observation["__last_tool_results__"] = deepcopy(tool_results)
            self._observation["__last_realized_events__"] = []
            visible_evidence_ids = [
                str(value)
                for result in tool_results
                for value in [
                    result.get("evidence_id"),
                    *(result.get("produces_evidence_ids") or []),
                ]
                if value
            ]
            self._observation["__last_evidence_ids__"] = list(
                dict.fromkeys(visible_evidence_ids)
            )
            self._observation["__last_reward__"] = 0.0
            self._observation["__last_early_stop_warnings__"] = []
            self._observation["__last_forecast_updates__"] = {}
            for evidence_id in visible_evidence_ids:
                if evidence_id not in self._authoritative_evidence_ids:
                    self._authoritative_evidence_ids.append(evidence_id)
            _, _, _, tool_trace_edges = _action_effect_closure(
                action=action,
                tool_results=tool_results,
                realized_events=[],
                evidence_ledger=[],
                request_tick=selected.request_tick,
                simulator_tick=simulator_tick,
            )
            pending_call_ids = self._pending_call_ids(
                action=action,
                tool_results=tool_results,
            )
            if pending_call_ids:
                selected.action = deepcopy(action)
                self._deferred_submissions[selected.action_id] = _DeferredSubmission(
                    submission=selected,
                    pending_call_ids=set(pending_call_ids),
                    terminal_results=[
                        deepcopy(result)
                        for result in tool_results
                        if str(
                            (result.get("payload") or {}).get("_status") or ""
                        ).lower()
                        != "pending"
                    ],
                    tool_trace_edges=deepcopy(tool_trace_edges),
                    applied_tick=(
                        simulator_tick
                        if any(
                            result.get("ok") is True
                            and str(
                                (result.get("payload") or {}).get("_status") or ""
                            ).lower()
                            != "pending"
                            for result in tool_results
                        )
                        else None
                    ),
                )
                for call_id in pending_call_ids:
                    self._deferred_action_by_call_id[call_id] = selected.action_id
                self._record_lifecycle_locked(
                    selected,
                    status="pending",
                    reason="DELAYED_ACTION_PENDING",
                )
                self._active_submission = None
            self._transitions.append(
                {
                    "state_version_before": request_version,
                    "state_version_after": self._state_version,
                    "simulator_tick_before": simulator_tick,
                    "simulator_tick": simulator_tick,
                    "simulator_time_advanced": False,
                    "clock_advance_reason": "READ_ONLY_INVESTIGATION",
                    "clock_semantics": "soft_realtime_monotonic_single_writer",
                    "environment_progress_during_investigation": False,
                    "investigation_started_monotonic_ns": investigation_started_ns,
                    "investigation_ended_monotonic_ns": investigation_ended_ns,
                    "investigation_duration_ns": investigation_duration_ns,
                    "investigation_clock_deadline_monotonic_ns": clock_deadline_ns,
                    "investigation_clock_deadline_overrun_ns": max(
                        0, investigation_ended_ns - clock_deadline_ns
                    ),
                    "investigation_elapsed_tick_intervals": int(
                        investigation_duration_ns // int(self._tick_interval_s * 1e9)
                    ),
                    "investigation_snapshot_tick": returned_tick,
                    "investigation_tick_corrected": returned_tick != simulator_tick,
                    "action_id": selected.action_id,
                    "decision_id": selected.decision_id,
                    "turn_id": selected.turn_id,
                    "submitted_action": (
                        selected.action.to_dict()
                        if pending_call_ids
                        else nominal_action.to_dict()
                    ),
                    "nominal_action": nominal_action.to_dict(),
                    "nominal_tool_semantics": nominal_semantics,
                    "model_attempted_state_change": model_attempted_state_change,
                    "applied_action": None if pending_call_ids else action.to_dict(),
                    "dispatched_action": action.to_dict(),
                    "based_on_visible_evidence_ids": list(
                        selected.based_on_visible_evidence_ids
                    ),
                    "visible_evidence_ids_after": list(
                        self._observation["__last_evidence_ids__"]
                    ),
                    "agent_visible_observation_after": deepcopy(
                        self._observation
                    ),
                    "action_source": "model",
                    "safety_decision": safety_decision.to_dict(),
                    "safety_arbitration": safety_decision.to_dict(),
                    "safety_evidence_ids": list(safety_decision.evidence_ids),
                    "used_safety_fallback": False,
                    "environment_step_failed": False,
                    "control_confirmed": False,
                    "effect_observed": False,
                    "effect_evidence_ids": [],
                    "tool_trace_edges": tool_trace_edges,
                    "deferred_action_outcomes": pre_dispatch_deferred_outcomes,
                    "cancellation_audit": [
                        row
                        for outcome in pre_dispatch_deferred_outcomes
                        for row in outcome.get("cancellation_audit", [])
                    ],
                    "tool_results": tool_results,
                    "realized_events": [],
                    "forecast_updates": {},
                    "early_stop_warnings": [],
                    "step_evidence_ids": [],
                    "reward": 0.0,
                    "monotonic_ns": time.monotonic_ns(),
                }
            )
            investigation_applied = not pending_call_ids and any(
                result.get("ok") is True for result in tool_results
            )
            if investigation_applied:
                self._record_lifecycle_locked(
                    selected,
                    status="applied",
                    reason=None,
                    applied_tick=simulator_tick,
                )
            if not pending_call_ids:
                self._settle_locked(
                    selected,
                    status="no_effect",
                    reason=None,
                    applied_tick=(
                        simulator_tick if investigation_applied else None
                    ),
                )
                self._active_submission = None
            self._condition.notify_all()
        self._drain_future_completions()

    def _advance_once(self) -> None:
        safety_failed = False
        with self._condition:
            simulator_tick_before = self._simulator_tick_locked()
            (
                pre_step_deferred_outcomes,
                execution_fence_failed,
            ) = self._process_deferred_boundary_locked(
                simulator_tick=simulator_tick_before,
                expiry_fence_tick=simulator_tick_before + 1,
            )
            if execution_fence_failed:
                return
            pending, self._pending = self._pending, []
            valid: list[_PendingSubmission] = []
            future_submissions: list[_PendingSubmission] = []
            for submission in pending:
                if submission.future.cancelled():
                    self._settle_locked(
                        submission,
                        status="canceled",
                        reason="CALLER_CANCELED",
                    )
                elif (
                    submission.expires_at_tick is not None
                    and simulator_tick_before > submission.expires_at_tick
                ):
                    self._settle_locked(submission, status="expired", reason="EXPIRED")
                elif submission.based_on_state_version != self._state_version:
                    self._settle_locked(submission, status="stale", reason="STALE_STATE")
                elif simulator_tick_before < submission.valid_from_tick:
                    future_submissions.append(submission)
                else:
                    valid.append(submission)
            self._pending.extend(future_submissions)
            selected = valid[-1] if valid else None
            for submission in valid[:-1]:
                self._settle_locked(
                    submission,
                    status="superseded",
                    reason="NEWER_VALID_ACTION",
                )
            if selected is not None:
                uncancellable_call_ids = (
                    self._uncancellable_managed_delay_calls_locked(
                        selected.action
                    )
                )
                if uncancellable_call_ids:
                    rejected = selected
                    self._settle_locked(
                        rejected,
                        status="rejected",
                        reason="UNCANCELLABLE_MANAGED_DELAY",
                    )
                    self._transitions.append(
                        {
                            "state_version_before": self._state_version,
                            "state_version_after": self._state_version,
                            "simulator_tick_before": simulator_tick_before,
                            "simulator_tick": simulator_tick_before,
                            "simulator_time_advanced": False,
                            "action_id": rejected.action_id,
                            "decision_id": rejected.decision_id,
                            "turn_id": rejected.turn_id,
                            "submitted_action": rejected.action.to_dict(),
                            "nominal_action": rejected.action.to_dict(),
                            "applied_action": None,
                            "action_source": "model",
                            "rejection_reason": "UNCANCELLABLE_MANAGED_DELAY",
                            "uncancellable_call_ids": uncancellable_call_ids,
                            "environment_step_failed": False,
                            "monotonic_ns": time.monotonic_ns(),
                        }
                    )
                    selected = None
            if selected is not None and not selected.future.set_running_or_notify_cancel():
                self._settle_locked(
                    selected,
                    status="canceled",
                    reason="CALLER_CANCELED",
                )
                selected = None
            request_version = self._state_version
            based_on_visible_evidence_ids = (
                list(selected.based_on_visible_evidence_ids)
                if selected is not None
                else list(self._observation.get("__last_evidence_ids__") or [])
            )
            if selected is not None:
                visible_evidence = set(based_on_visible_evidence_ids)
                invisible_references = {
                    str(evidence_id)
                    for call in selected.action.tool_calls
                    for evidence_id in call.consumes_evidence_ids or []
                    if str(evidence_id) not in visible_evidence
                }
                if invisible_references:
                    nominal_semantics, model_attempted_state_change = (
                        self._nominal_action_semantics(selected.action)
                    )
                    self._settle_locked(
                        selected,
                        status="rejected",
                        reason="INVISIBLE_EVIDENCE_REFERENCE",
                    )
                    self._transitions.append(
                        {
                            "state_version_before": request_version,
                            "state_version_after": request_version,
                            "simulator_tick_before": simulator_tick_before,
                            "simulator_tick": simulator_tick_before,
                            "simulator_time_advanced": False,
                            "action_id": selected.action_id,
                            "decision_id": selected.decision_id,
                            "turn_id": selected.turn_id,
                            "submitted_action": selected.action.to_dict(),
                            "nominal_action": selected.action.to_dict(),
                            "nominal_tool_semantics": nominal_semantics,
                            "model_attempted_state_change": (
                                model_attempted_state_change
                            ),
                            "applied_action": None,
                            "action_source": "model",
                            "rejection_reason": "INVISIBLE_EVIDENCE_REFERENCE",
                            "safety_supervisor_failed": False,
                            "environment_step_failed": False,
                            "monotonic_ns": time.monotonic_ns(),
                        }
                    )
                    selected = None
            nominal_semantics, model_attempted_state_change = (
                self._nominal_action_semantics(
                    selected.action if selected is not None else None
                )
            )
            if selected is not None:
                try:
                    safety_decision = self._arbitrate_locked(selected.action)
                except Exception as exc:  # noqa: BLE001 - fail closed at safety boundary
                    self._done = True
                    self._settle_locked(
                        selected,
                        status="rejected",
                        reason="SAFETY_ARBITRATION_FAILED",
                    )
                    self._transitions.append(
                        {
                            **self._clock_audit_fields(),
                            "state_version_before": request_version,
                            "state_version_after": request_version,
                            "simulator_tick_before": simulator_tick_before,
                            "simulator_tick": simulator_tick_before,
                            "simulator_time_advanced": False,
                            "action_id": selected.action_id,
                            "decision_id": selected.decision_id,
                            "turn_id": selected.turn_id,
                            "nominal_action": selected.action.to_dict(),
                            "nominal_tool_semantics": nominal_semantics,
                            "model_attempted_state_change": (
                                model_attempted_state_change
                            ),
                            "applied_action": None,
                            "action_source": "safety_supervisor",
                            "safety_decision": None,
                            "safety_arbitration": None,
                            "used_safety_fallback": True,
                            "safety_supervisor_failed": True,
                            "safety_error_type": type(exc).__name__,
                            "environment_step_failed": False,
                            "monotonic_ns": time.monotonic_ns(),
                        }
                    )
                    self._condition.notify_all()
                    safety_failed = True
                if not safety_failed:
                    assert safety_decision is not None
                    if safety_decision.disposition == "pass":
                        self._record_lifecycle_locked(
                            selected, status="accepted", reason=None
                        )
                    action = deepcopy(safety_decision.action)
            else:
                try:
                    safety_decision = self._safety_supervisor.decide(
                        observation=deepcopy(self._observation),
                        simulator_tick=simulator_tick_before,
                        reason="NO_VALID_MODEL_ACTION",
                    )
                    if not isinstance(safety_decision, SafetyDecision):
                        raise TypeError("supervisor returned an invalid decision")
                except Exception as exc:  # noqa: BLE001 - fail closed at safety boundary
                    self._done = True
                    self._transitions.append(
                        {
                            **self._clock_audit_fields(),
                            "state_version_before": request_version,
                            "state_version_after": request_version,
                            "simulator_tick_before": simulator_tick_before,
                            "simulator_tick": simulator_tick_before,
                            "simulator_time_advanced": False,
                            "action_id": None,
                            "decision_id": None,
                            "turn_id": None,
                            "action_source": "safety_supervisor",
                            "safety_decision": None,
                            "used_safety_fallback": True,
                            "safety_supervisor_failed": True,
                            "safety_error_type": type(exc).__name__,
                            "environment_step_failed": False,
                            "monotonic_ns": time.monotonic_ns(),
                        }
                    )
                    self._condition.notify_all()
                    safety_failed = True
                if not safety_failed:
                    assert safety_decision is not None
                    action = deepcopy(safety_decision.action)
            if not safety_failed and selected is not None:
                self._active_submission = selected
        self._drain_future_completions()
        if safety_failed:
            return
        try:
            result = self._env.step(action)
        except Exception as exc:  # noqa: BLE001 - boundary settles without detail leak
            with self._condition:
                self._done = True
                self._transitions.append(
                    {
                        **self._clock_audit_fields(),
                        "state_version_before": request_version,
                        "state_version_after": request_version,
                        "simulator_tick_before": simulator_tick_before,
                        "simulator_tick": simulator_tick_before,
                        "simulator_time_advanced": False,
                        "action_id": selected.action_id if selected else None,
                        "decision_id": selected.decision_id if selected else None,
                        "turn_id": selected.turn_id if selected else None,
                        "nominal_action": (
                            selected.action.to_dict() if selected else None
                        ),
                        "nominal_tool_semantics": nominal_semantics,
                        "model_attempted_state_change": model_attempted_state_change,
                        "applied_action": action.to_dict(),
                        "action_source": (
                            "model"
                            if selected is not None
                            and safety_decision is not None
                            and safety_decision.disposition == "pass"
                            else "safety_supervisor"
                        ),
                        "safety_decision": (
                            safety_decision.to_dict() if safety_decision else None
                        ),
                        "safety_arbitration": (
                            safety_decision.to_dict() if safety_decision else None
                        ),
                        "safety_evidence_ids": (
                            list(safety_decision.evidence_ids)
                            if safety_decision is not None
                            else []
                        ),
                        "used_safety_fallback": (
                            selected is None
                            or safety_decision is None
                            or safety_decision.disposition != "pass"
                        ),
                        "environment_step_failed": True,
                        "environment_error_type": type(exc).__name__,
                        "monotonic_ns": time.monotonic_ns(),
                    }
                )
                if selected is not None:
                    self._settle_locked(
                        selected,
                        status="failed",
                        reason="ENVIRONMENT_STEP_FAILED",
                    )
                    self._active_submission = None
                self._condition.notify_all()
            self._drain_future_completions()
            return
        result_observation = result.observation
        if not isinstance(result_observation, dict):
            raise TypeError("environment step observation must be a mapping")
        result_tick_raw = result_observation.get("tick")
        if isinstance(result_tick_raw, bool):
            raise ValueError("environment step tick must be an integer")
        if not isinstance(result_tick_raw, (int, str)):
            raise ValueError("environment step tick must be an integer")
        try:
            result_tick = int(result_tick_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("environment step tick must be an integer") from exc
        if result_tick <= simulator_tick_before:
            raise ValueError("environment step must strictly advance simulator tick")
        with self._condition:
            self._state_version += 1
            self._observation = deepcopy(result_observation)
            self._done = bool(result.done)
            tool_results = [
                item.to_dict() if hasattr(item, "to_dict") else deepcopy(item)
                for item in getattr(result, "tool_results", [])
            ]
            info = getattr(result, "info", None)
            realized_events = deepcopy(getattr(info, "realized_events", []) or [])
            step_evidence_ids = list(getattr(info, "evidence_ids", []) or [])
            authoritative_ids = [
                *step_evidence_ids,
                *[
                    value
                    for item in tool_results
                    for value in [
                        item.get("evidence_id"),
                        *(item.get("produces_evidence_ids") or []),
                    ]
                    if value
                ],
                *[
                    value
                    for event in realized_events
                    if isinstance(event, dict)
                    for value in [
                        event.get("evidence_id"),
                        *(event.get("evidence_ids") or []),
                    ]
                    if value
                ],
            ]
            for evidence_id in authoritative_ids:
                normalized = str(evidence_id)
                if normalized not in self._authoritative_evidence_ids:
                    self._authoritative_evidence_ids.append(normalized)
            visible_events = [
                event
                for event in realized_events
                if isinstance(event, dict) and event.get("hidden") is not True
            ]
            self._observation["__last_tool_results__"] = deepcopy(tool_results)
            self._observation["__last_realized_events__"] = deepcopy(visible_events)
            self._observation["__last_evidence_ids__"] = _agent_visible_evidence_ids(
                realized_events=realized_events,
                step_evidence_ids=list(getattr(info, "evidence_ids", []) or []),
                tool_results=tool_results,
            )
            self._observation["__last_reward__"] = float(
                getattr(result, "reward", 0.0)
            )
            self._observation["__last_early_stop_warnings__"] = list(
                getattr(info, "early_stop_warnings", []) or []
            )
            self._observation["__last_forecast_updates__"] = dict(
                getattr(info, "forecast_updates", {}) or {}
            )
            reward = float(getattr(result, "reward", 0.0))
            self._cumulative_reward += reward
            evidence_logger = getattr(self._env, "evidence", None)
            evidence_ledger = (
                evidence_logger.to_jsonable()
                if evidence_logger is not None
                and callable(getattr(evidence_logger, "to_jsonable", None))
                else []
            )
            (
                control_confirmed,
                effect_observed,
                effect_evidence_ids,
                tool_trace_edges,
            ) = _action_effect_closure(
                action=action,
                tool_results=tool_results,
                realized_events=realized_events,
                evidence_ledger=evidence_ledger,
                request_tick=(
                    selected.request_tick
                    if selected is not None
                    else simulator_tick_before
                ),
                simulator_tick=result_tick,
            )
            pending_call_ids = (
                self._pending_call_ids(
                    action=action,
                    tool_results=tool_results,
                )
                if selected is not None
                and safety_decision is not None
                and safety_decision.disposition == "pass"
                else set()
            )
            if selected is not None and pending_call_ids:
                selected.action = deepcopy(action)
                deferred_submission = _DeferredSubmission(
                    submission=selected,
                    pending_call_ids=set(pending_call_ids),
                    terminal_results=[
                        deepcopy(result)
                        for result in tool_results
                        if str(
                            (result.get("payload") or {}).get("_status") or ""
                        ).lower()
                        != "pending"
                    ],
                    control_confirmed=control_confirmed,
                    effect_observed=effect_observed,
                    effect_evidence_ids=tuple(effect_evidence_ids),
                    tool_trace_edges=deepcopy(tool_trace_edges),
                    applied_tick=(
                        simulator_tick_before
                        if any(
                            result.get("ok") is True
                            and str(
                                (result.get("payload") or {}).get("_status") or ""
                            ).lower()
                            != "pending"
                            for result in tool_results
                        )
                        else None
                    ),
                )
                self._deferred_submissions[selected.action_id] = deferred_submission
                for call_id in pending_call_ids:
                    self._deferred_action_by_call_id[call_id] = selected.action_id
                self._record_lifecycle_locked(
                    selected,
                    status="pending",
                    reason="DELAYED_ACTION_PENDING",
                )
                self._active_submission = None
            deferred_outcomes = [
                *pre_step_deferred_outcomes,
                *self._resolve_deferred_results_locked(
                    tool_results=tool_results,
                    realized_events=realized_events,
                    evidence_ledger=evidence_ledger,
                    simulator_tick=result_tick,
                ),
            ]
            if self._done and self._deferred_submissions:
                for action_id, deferred in list(
                    self._deferred_submissions.items()
                ):
                    terminal_status = deferred.fenced_status or "failed"
                    terminal_reason = (
                        deferred.fenced_reason_code
                        or "DELAYED_ACTION_UNRESOLVED_AT_EPISODE_END"
                    )
                    if deferred.fenced_status is None:
                        self._settle_locked(
                            deferred.submission,
                            status=terminal_status,
                            reason=terminal_reason,
                            applied_tick=deferred.applied_tick,
                        )
                    deferred_outcomes.append(
                        {
                            "action_id": deferred.submission.action_id,
                            "decision_id": deferred.submission.decision_id,
                            "turn_id": deferred.submission.turn_id,
                            "status": terminal_status,
                            "reason_code": terminal_reason,
                            "pending_call_ids": sorted(
                                deferred.pending_call_ids
                            ),
                            "control_confirmed": deferred.control_confirmed,
                            "effect_observed": deferred.effect_observed,
                            "effect_evidence_ids": list(
                                deferred.effect_evidence_ids
                            ),
                            "tool_trace_edges": deepcopy(
                                deferred.tool_trace_edges
                            ),
                        }
                    )
                    for call_id in deferred.pending_call_ids:
                        self._deferred_action_by_call_id.pop(call_id, None)
                    self._deferred_submissions.pop(action_id, None)
            associated_deferred = (
                deferred_outcomes[0]
                if selected is None and len(deferred_outcomes) == 1
                else None
            )
            deferred_edges = [
                edge
                for outcome in deferred_outcomes
                for edge in outcome["tool_trace_edges"]
            ]
            deferred_effect_ids = [
                evidence_id
                for outcome in deferred_outcomes
                for evidence_id in outcome["effect_evidence_ids"]
            ]
            cancellation_audit = [
                row
                for outcome in deferred_outcomes
                for row in outcome.get("cancellation_audit", [])
            ]
            self._transitions.append(
                {
                    **self._clock_audit_fields(),
                    "state_version_before": request_version,
                    "state_version_after": self._state_version,
                    "simulator_tick_before": simulator_tick_before,
                    "simulator_tick": self._observation.get("tick"),
                    "simulator_time_advanced": True,
                    "environment_done": self._done,
                    "action_id": (
                        selected.action_id
                        if selected
                        else associated_deferred["action_id"]
                        if associated_deferred
                        else None
                    ),
                    "decision_id": (
                        selected.decision_id
                        if selected
                        else associated_deferred["decision_id"]
                        if associated_deferred
                        else None
                    ),
                    "turn_id": (
                        selected.turn_id
                        if selected
                        else associated_deferred["turn_id"]
                        if associated_deferred
                        else None
                    ),
                    "submitted_action": (
                        selected.action.to_dict() if selected is not None else None
                    ),
                    "nominal_action": (
                        selected.action.to_dict() if selected else None
                    ),
                    "nominal_tool_semantics": nominal_semantics,
                    "model_attempted_state_change": model_attempted_state_change,
                    "applied_action": (
                        None
                        if pending_call_ids
                        and all(
                            str(
                                (result.get("payload") or {}).get("_status")
                                or ""
                            ).lower()
                            == "pending"
                            for result in tool_results
                        )
                        else action.to_dict()
                    ),
                    "dispatched_action": action.to_dict(),
                    "based_on_visible_evidence_ids": (
                        based_on_visible_evidence_ids
                    ),
                    "visible_evidence_ids_after": list(
                        self._observation["__last_evidence_ids__"]
                    ),
                    "agent_visible_observation_after": deepcopy(
                        self._observation
                    ),
                    "action_source": (
                        "model"
                        if selected is not None
                        and safety_decision is not None
                        and safety_decision.disposition == "pass"
                        else "safety_supervisor"
                    ),
                    "safety_decision": (
                        safety_decision.to_dict() if safety_decision else None
                    ),
                    "safety_arbitration": (
                        safety_decision.to_dict() if safety_decision else None
                    ),
                    "safety_evidence_ids": (
                        list(safety_decision.evidence_ids)
                        if safety_decision is not None
                        else []
                    ),
                    "used_safety_fallback": (
                        selected is None
                        or safety_decision is None
                        or safety_decision.disposition != "pass"
                    ),
                    "environment_step_failed": False,
                    "control_confirmed": control_confirmed
                    or any(
                        outcome["control_confirmed"]
                        for outcome in deferred_outcomes
                    ),
                    "effect_observed": effect_observed
                    or any(
                        outcome["effect_observed"]
                        for outcome in deferred_outcomes
                    ),
                    "effect_action_source": (
                        "model_deferred"
                        if any(
                            outcome["effect_observed"]
                            for outcome in deferred_outcomes
                        )
                        else (
                            "model"
                            if selected is not None
                            and safety_decision is not None
                            and safety_decision.disposition == "pass"
                            else None
                        )
                    ),
                    "effect_evidence_ids": list(
                        dict.fromkeys(
                            [*effect_evidence_ids, *deferred_effect_ids]
                        )
                    ),
                    "tool_trace_edges": [*tool_trace_edges, *deferred_edges],
                    "deferred_action_outcomes": deferred_outcomes,
                    "cancellation_audit": cancellation_audit,
                    "tool_results": tool_results,
                    "realized_events": realized_events,
                    "forecast_updates": dict(
                        getattr(info, "forecast_updates", {}) or {}
                    ),
                    "early_stop_warnings": list(
                        getattr(info, "early_stop_warnings", []) or []
                    ),
                    "step_evidence_ids": step_evidence_ids,
                    "reward": reward,
                    "monotonic_ns": time.monotonic_ns(),
                }
            )
            if selected is not None:
                assert safety_decision is not None
                if safety_decision.disposition == "pass" and not pending_call_ids:
                    action_applied = not tool_results or any(
                        result.get("ok") is True for result in tool_results
                    )
                    if action_applied:
                        self._record_lifecycle_locked(
                            selected,
                            status="applied",
                            reason=None,
                            applied_tick=simulator_tick_before,
                        )
                    self._settle_locked(
                        selected,
                        status=(
                            "effected"
                            if effect_observed
                            else "confirmed" if control_confirmed else "no_effect"
                        ),
                        reason=None,
                        applied_tick=(
                            simulator_tick_before if action_applied else None
                        ),
                    )
                elif safety_decision.disposition != "pass":
                    self._settle_locked(
                        selected,
                        status="rejected",
                        reason=f"SAFETY_{safety_decision.disposition.upper()}",
                    )
                if not pending_call_ids:
                    self._active_submission = None
            self._condition.notify_all()
        self._drain_future_completions()
