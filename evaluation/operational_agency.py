"""Diagnostic, causally grounded operational-agency profile.

This module is deliberately separate from the outcome headline scorer.  It
only assigns credit when an event-response record links observation or action
to a native backend effect and a positive action-group masked replay delta.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

PROFILE_VERSION = "operational_agency_profile_v1"
DIMENSIONS = (
    "outcome_influence",
    "initiative",
    "surprise_adaptation",
    "epistemic_control",
    "temporal_planning",
    "trade_off_quality",
)


def operational_agency_profile_is_consistent(
    trajectory_summary: Mapping[str, Any],
    *,
    counterfactual: Mapping[str, Any] | None = None,
) -> bool:
    """Validate the diagnostic construct and its episode-local evidence.

    This validator is shared by diagnostic readiness and the formal runner so
    an older or self-reported profile cannot pass one entry point but fail the
    other.
    """
    records = trajectory_summary.get("event_response_records")
    profile = trajectory_summary.get("operational_agency_profile")
    if (
        not isinstance(records, list)
        or not all(isinstance(record, Mapping) for record in records)
        or not isinstance(profile, Mapping)
        or profile.get("schema_version") != PROFILE_VERSION
        or profile.get("diagnostic_only") is not True
        or profile.get("headline_score_included") is not False
        or profile.get("runtime_binding_verified") is not True
        or profile.get("runtime_evidence_binding_verified") is not True
        or profile.get("masked_replay_binding_verified") is not True
    ):
        return False

    record_count = profile.get("event_response_record_count")
    causal_count = profile.get("causal_record_count")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count != len(records)
        or isinstance(causal_count, bool)
        or not isinstance(causal_count, int)
        or not 0 <= causal_count <= record_count
    ):
        return False

    authoritative_ids = trajectory_summary.get(
        "operational_agency_valid_evidence_ids"
    )
    if not isinstance(authoritative_ids, list) or not all(
        isinstance(evidence_id, str) and evidence_id
        for evidence_id in authoritative_ids
    ):
        return False
    episode_evidence_ids = set(authoritative_ids)
    claimed_ids = {
        evidence_id
        for record in records
        for key, values in record.items()
        if key.endswith("evidence_ids") and isinstance(values, list)
        for evidence_id in values
        if isinstance(evidence_id, str) and evidence_id
    }
    if not claimed_ids.issubset(episode_evidence_ids):
        return False
    replay_by_call_id: dict[str, float] = {}
    if isinstance(counterfactual, Mapping):
        individual_by_call_id: dict[str, float] = {}
        for row in counterfactual.get("per_action") or []:
            if not isinstance(row, Mapping):
                continue
            try:
                parsed = float(row.get("marginal_prevented_loss"))
            except (TypeError, ValueError):
                continue
            call_id = str(row.get("call_id") or "")
            if call_id and math.isfinite(parsed):
                individual_by_call_id[call_id] = parsed
        groups = [
            row
            for row in counterfactual.get("per_action_groups") or []
            if isinstance(row, Mapping)
        ]
        for record in records:
            call_id = str(record.get("call_id") or "")
            record_delta = _number(record, "masked_action_group_delta")
            if not call_id or record_delta is None:
                continue
            group_id = str(record.get("masked_action_group_id") or "")
            if group_id:
                candidates = [
                    group
                    for group in groups
                    if str(group.get("group_id") or "") == group_id
                    and call_id in [str(value) for value in group.get("call_ids") or []]
                ]
                declared_call_ids = [
                    str(value)
                    for value in record.get("masked_action_group_call_ids") or []
                ]
                if declared_call_ids:
                    candidates = [
                        group
                        for group in candidates
                        if [str(value) for value in group.get("call_ids") or []]
                        == declared_call_ids
                    ]
            elif call_id in individual_by_call_id:
                candidates = [
                    {"masked_action_group_delta": individual_by_call_id[call_id]}
                ]
            else:
                candidates = [
                    group
                    for group in groups
                    if call_id in [str(value) for value in group.get("call_ids") or []]
                ]
            matching = []
            for candidate in candidates:
                try:
                    value = float(candidate.get("masked_action_group_delta"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and abs(value - record_delta) <= 1e-9:
                    matching.append(value)
            if len(matching) == 1:
                replay_by_call_id[call_id] = record_delta
    if not replay_by_call_id and records:
        return False
    recomputed = evaluate_operational_agency(
        records,
        valid_evidence_ids=episode_evidence_ids,
        masked_replay_by_call_id=replay_by_call_id,
    )
    if dict(profile) != recomputed:
        return False
    dimensions = profile.get("dimensions")
    if not isinstance(dimensions, Mapping) or set(dimensions) != set(DIMENSIONS):
        return False
    for name in DIMENSIONS:
        dimension = dimensions[name]
        if not isinstance(dimension, Mapping):
            return False
        applicable = dimension.get("applicable")
        score = dimension.get("score")
        support_count = dimension.get("support_count")
        evidence_ids = dimension.get("evidence_ids")
        reason = dimension.get("reason")
        if applicable is True:
            if not (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and math.isfinite(float(score))
                and 0.0 <= float(score) <= 100.0
                and isinstance(support_count, int)
                and not isinstance(support_count, bool)
                and 1 <= support_count <= causal_count
                and isinstance(evidence_ids, list)
                and bool(evidence_ids)
                and all(
                    isinstance(evidence_id, str)
                    and bool(evidence_id)
                    and evidence_id in episode_evidence_ids
                    for evidence_id in evidence_ids
                )
                and reason is None
            ):
                return False
            continue
        if not (
            applicable is False
            and score is None
            and support_count == 0
            and evidence_ids == []
            and isinstance(reason, str)
            and bool(reason)
        ):
            return False
    outcome = dimensions["outcome_influence"]
    if causal_count == 0:
        return (
            outcome.get("applicable") is False
            and outcome.get("support_count") == 0
        )
    return outcome.get("applicable") is True and outcome.get(
        "support_count"
    ) == causal_count


def _tick(record: Mapping[str, Any], key: str) -> int | None:
    value = record.get(key)
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _number(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _ids(
    record: Mapping[str, Any],
    *keys: str,
    valid_evidence_ids: set[str] | None = None,
) -> list[str]:
    values: list[str] = []
    for key in keys:
        raw = record.get(key) or []
        if not isinstance(raw, list):
            continue
        values.extend(
            str(value)
            for value in raw
            if value is not None
            and str(value)
            and (
                valid_evidence_ids is None
                or str(value) in valid_evidence_ids
            )
        )
    return list(dict.fromkeys(values))


def _causal(
    record: Mapping[str, Any],
    *,
    valid_evidence_ids: set[str] | None,
    masked_replay_by_call_id: Mapping[str, float] | None,
) -> bool:
    delta = _number(record, "masked_action_group_delta")
    event_tick = _tick(record, "event_tick")
    control_tick = _tick(record, "first_control_call_tick")
    effect_tick = _tick(record, "first_effect_tick")
    effect_ids = _ids(
        record,
        "backend_effect_evidence_ids",
        "outcome_evidence_ids",
        valid_evidence_ids=valid_evidence_ids,
    )
    event_id = str(record.get("event_id") or "")
    call_id = str(record.get("call_id") or "")
    parent_event_id = str(record.get("causal_parent_event_id") or "")
    trigger_ids = set(
        _ids(
            record,
            "trigger_evidence_ids",
            valid_evidence_ids=valid_evidence_ids,
        )
    )
    consumed_ids = set(
        _ids(
            record,
            "action_consumes_evidence_ids",
            valid_evidence_ids=valid_evidence_ids,
        )
    )
    replay_verified = False
    if masked_replay_by_call_id is not None:
        replay_delta = masked_replay_by_call_id.get(call_id)
        replay_verified = bool(
            replay_delta is not None
            and delta is not None
            and abs(float(replay_delta) - delta) <= 1e-9
        )
    linked_to_event = bool(
        event_id
        and call_id
        and (
            parent_event_id == event_id
            or bool(trigger_ids.intersection(consumed_ids))
        )
    )
    return bool(
        delta is not None
        and delta > 0.0
        and event_tick is not None
        and control_tick is not None
        and event_tick <= control_tick
        and effect_tick is not None
        and effect_tick >= control_tick
        and effect_ids
        and linked_to_event
        and valid_evidence_ids is not None
        and replay_verified
        and record.get("response_status") == "causal"
    )


def _bounded_score(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 6)


def _dimension(
    qualifying: Iterable[tuple[float, list[str]]],
    *,
    reason: str,
) -> dict[str, Any]:
    rows = list(qualifying)
    if not rows:
        return {
            "applicable": False,
            "score": None,
            "support_count": 0,
            "evidence_ids": [],
            "reason": reason,
        }
    evidence_ids = list(
        dict.fromkeys(evidence_id for _, ids in rows for evidence_id in ids)
    )
    if not evidence_ids:
        return {
            "applicable": False,
            "score": None,
            "support_count": 0,
            "evidence_ids": [],
            "reason": "native_evidence_missing",
        }
    return {
        "applicable": True,
        "score": _bounded_score(sum(value for value, _ in rows) / len(rows)),
        "support_count": len(rows),
        "evidence_ids": evidence_ids,
        "reason": None,
    }


def _causal_score(record: Mapping[str, Any]) -> float | None:
    delta = _number(record, "masked_action_group_delta")
    burden = _number(record, "native_burden_before")
    if (
        delta is None
        or delta <= 0.0
        or burden is None
        or burden < 0.0
        or record.get("native_burden_basis") != "episode_replay_no_action_cost"
    ):
        return None
    return 100.0 * delta / max(delta + burden, 1e-12)


def evaluate_operational_agency(
    event_response_records: Iterable[Mapping[str, Any]],
    *,
    valid_evidence_ids: set[str] | None = None,
    masked_replay_by_call_id: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build the independent six-dimensional diagnostic profile.

    Agent text, declared dilemma labels, and evidence identifiers without a
    positive masked replay effect are never sufficient for credit.
    """
    records = [dict(record) for record in event_response_records]
    evidence_binding_verified = valid_evidence_ids is not None
    masked_replay_binding_verified = masked_replay_by_call_id is not None
    runtime_binding_verified = (
        evidence_binding_verified and masked_replay_binding_verified
    )
    if valid_evidence_ids is not None:
        for record in records:
            for key, value in list(record.items()):
                if key.endswith("evidence_ids") and isinstance(value, list):
                    record[key] = [
                        evidence_id
                        for evidence_id in value
                        if isinstance(evidence_id, str)
                        and evidence_id in valid_evidence_ids
                    ]
    causal = [
        record
        for record in records
        if _causal(
            record,
            valid_evidence_ids=valid_evidence_ids,
            masked_replay_by_call_id=masked_replay_by_call_id,
        )
    ]

    outcome = []
    for record in causal:
        score = _causal_score(record)
        if score is not None:
            outcome.append(
                (
                    score,
                    _ids(
                        record,
                        "backend_effect_evidence_ids",
                        "outcome_evidence_ids",
                    ),
                )
            )

    initiative: list[tuple[float, list[str]]] = []
    for record in causal:
        action_tick = _tick(record, "first_control_call_tick")
        mandatory_tick = _tick(record, "mandatory_response_tick")
        event_tick = _tick(record, "event_tick")
        observed_tick = _tick(record, "first_observed_tick")
        if (
            action_tick is None
            or mandatory_tick is None
            or event_tick is None
            or observed_tick is None
            or observed_tick > action_tick
            or action_tick >= mandatory_tick
        ):
            continue
        span = max(1, mandatory_tick - event_tick)
        lead = mandatory_tick - action_tick
        initiative.append(
            (
                100.0 * lead / span,
                _ids(
                    record,
                    "action_evidence_ids",
                    "backend_effect_evidence_ids",
                ),
            )
        )

    adaptation: list[tuple[float, list[str]]] = []
    for record in causal:
        visibility = str(record.get("visibility") or "")
        surprise = bool(record.get("surprise")) or visibility in {
            "hidden",
            "stale",
            "delayed",
        }
        observed = _tick(record, "first_observed_tick")
        action = _tick(record, "first_control_call_tick")
        if surprise and observed is not None and action is not None and action >= observed:
            score = _causal_score(record)
            if score is None:
                continue
            adaptation.append(
                (
                    score,
                    _ids(
                        record,
                        "observation_evidence_ids",
                        "backend_effect_evidence_ids",
                    ),
                )
            )

    epistemic: list[tuple[float, list[str]]] = []
    for record in causal:
        investigation = _tick(record, "first_investigation_tick")
        action = _tick(record, "first_control_call_tick")
        observation_ids = _ids(record, "observation_evidence_ids")
        if (
            investigation is not None
            and action is not None
            and investigation <= action
            and observation_ids
        ):
            score = _causal_score(record)
            if score is None:
                continue
            epistemic.append(
                (
                    score,
                    observation_ids
                    + _ids(record, "backend_effect_evidence_ids"),
                )
            )

    planning: list[tuple[float, list[str]]] = []
    for record in causal:
        plan_replaced = bool(record.get("replaces_plan_id"))
        committed = bool(record.get("irreversible_commitment"))
        depth = _number(record, "dependency_depth") or 0.0
        plan_ids = _ids(record, "plan_evidence_ids")
        if plan_ids and (plan_replaced or committed or depth >= 2.0):
            planning.append(
                (
                    min(100.0, 25.0 * max(1.0, depth)),
                    plan_ids + _ids(record, "backend_effect_evidence_ids"),
                )
            )

    # Trade-off credit remains unavailable until records carry structured,
    # replay-bound policy outcomes from which this evaluator can independently
    # recompute feasibility, fatality, Pareto dominance, and regret.  Boolean
    # option labels supplied by scenarios/backends are not score evidence.
    tradeoffs: list[tuple[float, list[str]]] = []

    dimensions = {
        "outcome_influence": _dimension(
            outcome, reason="no_positive_masked_backend_effect"
        ),
        "initiative": _dimension(
            initiative, reason="no_preemptive_causal_response"
        ),
        "surprise_adaptation": _dimension(
            adaptation, reason="no_causal_surprise_response"
        ),
        "epistemic_control": _dimension(
            epistemic, reason="no_investigation_to_effect_chain"
        ),
        "temporal_planning": _dimension(
            planning, reason="no_causal_multi_stage_plan_evidence"
        ),
        "trade_off_quality": _dimension(
            tradeoffs, reason="no_replay_backed_non_dominated_tradeoff"
        ),
    }
    return {
        "schema_version": PROFILE_VERSION,
        "diagnostic_only": True,
        "headline_score_included": False,
        "runtime_binding_verified": runtime_binding_verified,
        "runtime_evidence_binding_verified": evidence_binding_verified,
        "masked_replay_binding_verified": masked_replay_binding_verified,
        "event_response_record_count": len(records),
        "causal_record_count": len(causal),
        "dimensions": dimensions,
    }
