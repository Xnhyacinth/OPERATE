"""
core.evidence — Evidence-linked scoring contract.

The single biggest scoring weakness in dispatch-benchmark was that some
dimensions emitted scores without auditable evidence: an LLM-judged
``ethical_reasoning`` score could not be re-derived from the trajectory.

OPERATE enforces a triple per dimension:

    applicable     : bool       # was this dimension exercised at all?
    support_count  : int        # how many evidence items back the score?
    evidence_ids   : list[str]  # pointers into the trajectory / event log
    reason         : str        # short human-readable rationale

Every event the env emits — a tool call, a fault injection, a load shed,
a moral choice, a casualty number — gets a stable evidence id. Scores then
cite those ids; the audit script (`audit.py`) verifies that every score
above zero has at least one valid evidence id.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceItem:
    """One auditable piece of evidence captured during an episode."""

    evidence_id: str
    tick: int
    kind: str  # e.g. "tool_call", "fault", "shed", "moral_choice", "casualty"
    payload: dict[str, Any] = field(default_factory=dict)
    source: str = "engine"  # "engine" | "tool" | "scenario" | "agent"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DimensionScore:
    """Evidence-linked score for one evaluation dimension."""

    name: str
    raw_score: float = 0.0
    calibrated_score: float = 0.0
    applicable: bool = True
    support_count: int = 0
    evidence_ids: list[str] = field(default_factory=list)
    reason: str = ""
    weight: float = 1.0  # may be zeroed if not applicable
    # Set by ``system_survival`` when >=1 catastrophic-failure tick occurred
    # (see ``evaluation.scorer.score_system_survival``). Unused by every other
    # dimension; exists so a view-computed objective pass/fail (e.g.
    # ``task_completion_for_row``) can key off a real floor rather than
    # thresholding the continuous ``calibrated_score`` density.
    floor_violation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "raw_score": round(float(self.raw_score), 4),
            "calibrated_score": round(float(self.calibrated_score), 4),
            "applicable": bool(self.applicable),
            "support_count": int(self.support_count),
            "evidence_ids": list(self.evidence_ids),
            "reason": self.reason,
            "weight": float(self.weight),
            "floor_violation": bool(self.floor_violation),
        }


class EvidenceLogger:
    """Append-only structured log of episode evidence.

    Each item gets a deterministic id derived from ``(episode_id, tick,
    kind, ordinal)`` so trajectory-replay tests get stable diff output.
    """

    def __init__(self, episode_id: str):
        self._episode_id = episode_id
        self._items: list[EvidenceItem] = []
        self._counters: dict[str, int] = {}

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def log(
        self,
        kind: str,
        tick: int,
        payload: dict[str, Any] | None = None,
        source: str = "engine",
    ) -> str:
        idx = self._counters.get(kind, 0)
        self._counters[kind] = idx + 1
        eid = _evidence_id(self._episode_id, tick, kind, idx)
        self._items.append(
            EvidenceItem(
                evidence_id=eid,
                tick=tick,
                kind=kind,
                payload=dict(payload or {}),
                source=source,
            )
        )
        return eid

    def items(self) -> list[EvidenceItem]:
        return list(self._items)

    def items_by_kind(self, kind: str) -> list[EvidenceItem]:
        return [i for i in self._items if i.kind == kind]

    def to_jsonable(self) -> list[dict[str, Any]]:
        return [i.to_dict() for i in self._items]


def control_summary_from_evidence(
    evidence: EvidenceLogger,
    *,
    include_lifecycle: bool = False,
) -> dict[str, Any]:
    """Separate accepted control requests from materialized physical effects."""
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
            raw_status = value.get(key)
            if not isinstance(raw_status, str):
                continue
            tokens = {
                token
                for token in raw_status.strip().lower().replace("-", "_").split("_")
                if token
            }
            if tokens.intersection(non_effect_statuses):
                return True
        applied_action = value.get("applied_action")
        return isinstance(applied_action, dict) and blocks_effect(applied_action)

    request_tool_ticks: dict[str, set[int]] = {}
    effect_tool_ticks: dict[str, set[int]] = {}
    endpoint_ticks: dict[str, set[int]] = {}
    request_ticks_by_call: dict[tuple[str, str], int] = {}
    request_args_by_call: dict[tuple[str, str], dict[str, Any]] = {}
    request_results_by_call: dict[tuple[str, str], dict[str, Any]] = {}
    effect_results_by_call: dict[tuple[str, str], dict[str, Any]] = {}
    endpoint_by_call: dict[tuple[str, str], str] = {}
    tool_rows_by_call: dict[str, list[tuple[EvidenceItem, dict[str, Any]]]] = {}
    for item in evidence.items_by_kind("tool_call"):
        payload = item.payload
        name = str(payload.get("name") or "")
        call_id = str(payload.get("call_id") or "")
        if not name or not call_id:
            continue
        tool_rows_by_call.setdefault(call_id, []).append((item, payload))

    ignored_names = {"wait", "noop", "moral_choice", "commit_to_plan"}
    for call_id, rows in tool_rows_by_call.items():
        names = {str(payload.get("name") or "") for _, payload in rows}
        if len(names) != 1:
            continue
        name = next(iter(names))
        if name in ignored_names or len(rows) > 2:
            continue
        normalized_rows: list[
            tuple[EvidenceItem, dict[str, Any], dict[str, Any], str]
        ] = []
        invalid = False
        for item, payload in rows:
            result = payload.get("payload") or {}
            if not isinstance(result, dict):
                result = {}
            status = str(result.get("_status") or "").strip().lower()
            if payload.get("ok") is not True or blocks_effect(result):
                invalid = True
                break
            normalized_rows.append((item, payload, result, status))
        if invalid:
            continue
        request_row = normalized_rows[0]
        terminal_row: tuple[EvidenceItem, dict[str, Any], dict[str, Any], str] | None
        if len(normalized_rows) == 1:
            terminal_row = None if request_row[3] == "pending" else request_row
            if (
                terminal_row is not None
                and request_row[1].get("state_changing") is not True
            ):
                continue
        else:
            terminal_row = normalized_rows[1]
            request_args = request_row[1].get("args") or {}
            terminal_args = terminal_row[1].get("args") or {}
            due_tick = request_row[2].get("due_tick")
            if (
                request_row[3] != "pending"
                or isinstance(due_tick, bool)
                or not isinstance(due_tick, int)
                or due_tick != int(terminal_row[0].tick)
                or terminal_row[1].get("state_changing") is not True
                or int(terminal_row[0].tick) < int(request_row[0].tick)
                or (
                    isinstance(request_args, dict)
                    and isinstance(terminal_args, dict)
                    and dict(request_args) != dict(terminal_args)
                )
            ):
                continue
        call_key = (call_id, name)
        request_ticks_by_call[call_key] = int(request_row[0].tick)
        args = request_row[1].get("args") or {}
        if isinstance(args, dict):
            request_args_by_call[call_key] = dict(args)
        request_results_by_call[call_key] = dict(request_row[2])
        endpoint_rows = [request_row]
        if terminal_row is not None:
            effect_results_by_call[call_key] = dict(terminal_row[2])
            endpoint_rows.append(terminal_row)
        for _, _, result, _ in endpoint_rows:
            endpoint = str(result.get("sumo_tls_id") or result.get("tls_id") or "")
            if endpoint:
                endpoint_by_call.setdefault(call_key, endpoint)
    for (_call_id, name), request_tick in request_ticks_by_call.items():
        request_tool_ticks.setdefault(name, set()).add(request_tick)

    first_effect_by_call: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for item in evidence.items_by_kind("realized_event"):
        payload = item.payload
        before_digest = str(payload.get("before_state_digest") or "")
        after_digest = str(payload.get("after_state_digest") or "")
        call_id = str(payload.get("call_id") or "")
        tool_name = str(payload.get("tool_name") or "")
        call_key = (call_id, tool_name)
        request_tick = request_ticks_by_call.get(call_key)
        raw_effect_tick = payload.get("effect_tick")
        if raw_effect_tick is None:
            raw_effect_tick = item.tick
        if isinstance(raw_effect_tick, bool):
            continue
        try:
            effect_tick = int(raw_effect_tick)
        except (OverflowError, TypeError, ValueError):
            continue
        if isinstance(raw_effect_tick, float) and raw_effect_tick != effect_tick:
            continue
        requested_action = payload.get("requested_action") or {}
        if isinstance(requested_action, dict):
            event_tool_name = str(requested_action.get("name") or tool_name)
            event_args = requested_action.get("args", requested_action)
        else:
            event_tool_name = tool_name
            event_args = None
        request_args = request_args_by_call.get(call_key, {})
        if (
            item.source != "engine"
            or payload.get("origin") != "agent_caused"
            or payload.get("agent_caused") is not True
            or not call_id
            or not tool_name
            or not before_digest
            or not after_digest
            or before_digest == after_digest
            or blocks_effect(payload)
            or request_tick is None
            or effect_tick < request_tick
            or effect_tick < int(item.tick)
            or event_tool_name != tool_name
            or (
                include_lifecycle
                and isinstance(event_args, dict)
                and dict(event_args) != request_args
            )
        ):
            continue
        previous = first_effect_by_call.get(call_key)
        if previous is None or effect_tick < previous[0]:
            effect_payload = dict(payload)
            effect_payload.setdefault("event_id", item.evidence_id)
            first_effect_by_call[call_key] = (effect_tick, effect_payload)

    lifecycle_records: list[dict[str, Any]] = []
    for call_key, (effect_tick, event_payload) in first_effect_by_call.items():
        call_id, name = call_key
        effect_tool_ticks.setdefault(name, set()).add(effect_tick)
        endpoint = endpoint_by_call.get(call_key)
        if endpoint:
            endpoint_ticks.setdefault(f"{name}|{endpoint}", set()).add(effect_tick)
        lifecycle = {
            "call_id": call_id,
            "tool_name": name,
            "request_tick": request_ticks_by_call[call_key],
            "effect_tick": effect_tick,
            "request_args": request_args_by_call.get(call_key, {}),
            "request_result": request_results_by_call.get(call_key, {}),
            "effect_result": effect_results_by_call.get(call_key, {}),
            "effect_event_id": str(event_payload.get("event_id") or ""),
        }
        if endpoint:
            lifecycle["actuator_endpoint"] = endpoint
        lifecycle_records.append(lifecycle)
    summary = {
        "distinct_control_ticks": sorted(
            {tick for ticks in effect_tool_ticks.values() for tick in ticks}
        ),
        "distinct_physical_tools": sorted(effect_tool_ticks),
        "tool_ticks": {
            tool: sorted(ticks) for tool, ticks in sorted(request_tool_ticks.items())
        },
        "effect_tool_ticks": {
            tool: sorted(ticks) for tool, ticks in sorted(effect_tool_ticks.items())
        },
    }
    if include_lifecycle:
        summary["control_lifecycle_records"] = sorted(
            lifecycle_records,
            key=lambda row: (
                int(row["effect_tick"]),
                int(row["request_tick"]),
                str(row["tool_name"]),
                str(row["call_id"]),
            ),
        )
    if endpoint_ticks:
        summary.update(
            {
                "distinct_physical_actuator_endpoints": sorted(endpoint_ticks),
                "actuator_endpoint_ticks": {
                    endpoint: sorted(ticks)
                    for endpoint, ticks in sorted(endpoint_ticks.items())
                },
            }
        )
    return summary


def _evidence_id(episode_id: str, tick: int, kind: str, ordinal: int) -> str:
    body = f"{episode_id}|{tick}|{kind}|{ordinal}"
    return "ev_" + hashlib.sha1(body.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers used by evaluation.scorer
# ─────────────────────────────────────────────────────────────────────────────


def aggregate(
    dimensions: Iterable[DimensionScore],
    drop_non_applicable: bool = True,
) -> float:
    """Compute weighted mean over applicable dimensions.

    ``drop_non_applicable=True`` means non-applicable dimensions are
    completely excluded (their weight is zeroed) rather than counted as
    zero — this is the lesson from dispatch-benchmark where free zeros
    silently inflated scores on simple scenarios.
    """
    total_weight = 0.0
    total = 0.0
    for d in dimensions:
        if drop_non_applicable and not d.applicable:
            continue
        weight = d.weight if d.applicable else 0.0
        total_weight += weight
        total += weight * d.calibrated_score
    if total_weight <= 0.0:
        return 0.0
    return total / total_weight
