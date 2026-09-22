"""Evidence-linked companion to offline 0.22 outcomes, never a new ranking.

Input artifact authentication belongs to the caller. Episode-local causal joins
are independently checked here, even when native calibration is unavailable.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from evaluation.native_objectives import extract_native_objective
from evaluation.operational_agency import (
    evaluate_operational_agency,
    operational_agency_profile_is_consistent,
)


def _ticks(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def build_capability_report(
    episode: dict[str, Any], *, scenario_spec: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Expose measured outcomes, causal timing and measurement boundaries.

    ``scenario_spec`` is the caller's matched suite specification, not model
    text. Recorded tick counts are metadata, not proof of memory retention or
    authenticated trace coverage. No opportunity success rate is invented from
    a profile whose support contains successful chains only.
    """
    summary = episode.get("trajectory_summary") or {}
    counterfactual = episode.get("counterfactual") or {}
    verified = operational_agency_profile_is_consistent(
        summary, counterfactual=counterfactual
    )
    profile = summary.get("operational_agency_profile") if verified else None
    chains = None
    if verified:
        chains = []
        ids = set(summary["operational_agency_valid_evidence_ids"])
        for record in summary["event_response_records"]:
            # Validate each successful chain independently, so a valid aggregate
            # cannot confer causal status on another noncausal record.
            delta = record.get("masked_action_group_delta")
            single_profile = evaluate_operational_agency(
                [record], valid_evidence_ids=ids,
                masked_replay_by_call_id={record.get("call_id"): delta},
            )
            single = {
                "event_response_records": [record],
                "operational_agency_valid_evidence_ids": sorted(ids),
                "operational_agency_profile": single_profile,
            }
            if not operational_agency_profile_is_consistent(
                single, counterfactual=counterfactual
            ) or single_profile["causal_record_count"] != 1:
                continue
            observed = _ticks(record.get("first_observed_tick"))
            action = _ticks(record.get("first_control_call_tick"))
            effect = _ticks(record.get("first_effect_tick"))
            if None in (observed, action, effect):
                continue
            chains.append({
                "event_id": record["event_id"],
                "call_id": record["call_id"],
                "observed_tick": observed,
                "action_tick": action,
                "effect_tick": effect,
                "observation_to_action_ticks": action - observed,
                "action_to_effect_ticks": effect - action,
                "masked_replay_delta": delta,
                "plan_evidence_ids": list(record.get("plan_evidence_ids") or []),
                "effect_evidence_ids": list(dict.fromkeys(
                    (record.get("backend_effect_evidence_ids") or [])
                    + (record.get("outcome_evidence_ids") or [])
                )),
            })
    configured = _ticks((scenario_spec or {}).get("horizon_ticks"))
    recorded = _ticks(episode.get("n_ticks_ran"))
    summary_ticks = _ticks(summary.get("n_ticks"))
    reason = "recorded_episode_metadata" if recorded is not None else "tick_count_missing"
    if recorded is not None and summary_ticks is not None and recorded != summary_ticks:
        recorded = None
        reason = "recorded_tick_counts_disagree"
    mode = episode.get("interaction_mode")
    return {
        "schema_version": "capability_report.v1",
        "evaluation_version": "0.22.0",
        "diagnostic_only": True,
        "headline_score_included": False,
        "formal_run_certified": False,
        "interaction_mode": mode,
        "native_outcome": extract_native_objective(episode),
        "operational_agency": {
            "verified": verified,
            "validation_scope": "episode_internal_consistency",
            "dimensions": deepcopy(profile["dimensions"]) if verified else None,
            "causal_record_count": profile["causal_record_count"] if verified else None,
            "opportunity_count": None,
            "success_rate": None,
            "reason": "conditional_successful_chain_support_not_opportunity_rate"
            if verified else "missing_or_inconsistent_episode_causal_profile",
        },
        "temporal_evidence": {
            "chains": chains,
            "interpretation": "observed_action_effect_spans_not_memory_retention",
        },
        "horizon": {
            "configured_ticks": configured,
            "configured_stratum": None if configured is None else (
                "above_192_ticks" if configured > 192 else "at_most_192_ticks"
            ),
            "recorded_ticks": recorded,
            "reason": reason,
            "trace_coverage_verified": False,
            "native_elapsed_time": None,
            "memory_retention_score": None,
            "memory_retention_reason": "requires_cross_boundary_retention_evidence",
            "stratum_semantics": "descriptive_suite_split_not_capability_threshold",
        },
        "realtime_supervision": {
            "applicable": mode == "realtime_persistent",
            "score": None,
            "reason": "use_independent_realtime_ledger_scorecard"
            if mode == "realtime_persistent" else "not_a_realtime_treatment",
        },
    }


def build_long_task_report(
    graded_rows: list[dict[str, Any]], suite_rows: list[dict[str, Any]], model: str
) -> dict[str, Any]:
    """Fixed declared >192-tick outcome slice; caller enforces execution cohort.

    This is a small-sample scheduling diagnostic, not a memory capability score.
    Missing rows never reduce the declared denominator.
    """
    from evaluation.native_quality import aggregate_native_rows

    scope = [row for row in suite_rows if (_ticks(row.get("horizon_ticks")) or 0) > 192]
    result: dict[str, Any] = {
        "schema_version": "long_task_diagnostic.v1",
        "selection": "declared_suite_horizon_above_192_ticks",
        "interpretation": "selected_long_task_outcomes_not_memory_retention",
        "formal_run_certified": False,
        "expected_cases": len(scope), "n_authenticated_native_outcomes": 0,
        "n_reference_normalized_cases": 0, "complete": False, "index": None,
        "cases": [], "reason": "no_declared_long_task_cases",
    }
    if not scope:
        return result
    keys = {(row["scenario_signature"], row["seed"]) for row in scope}
    selected = []
    for row in graded_rows:
        if row.get("model") == model and (row["scenario_signature"], row["seed"]) in keys:
            bound = (row.get("artifact_binding") or {}).get("verified") is True
            selected.append({**row, "native_quality": row["native_quality"] if bound else {
                "score": None, "reason": "unbound_episode_artifacts",
            }})
    aggregate = aggregate_native_rows(selected, scope, [model])
    summary = aggregate["models"][model]
    for spec in scope:
        attempts = [row for row in selected if (row["scenario_signature"], row["seed"]) ==
                    (spec["scenario_signature"], spec["seed"])]
        case = {
            "scenario_signature": spec["scenario_signature"], "seed": spec["seed"],
            "configured_ticks": spec["horizon_ticks"], "recorded_ticks": None,
            "native_outcome": None, "native_quality": None,
            "reason": "missing_or_duplicate_attempt",
        }
        if len(attempts) == 1:
            row = attempts[0]
            binding = row.get("artifact_binding") or {}
            case["native_quality"] = deepcopy(row["native_quality"])
            case["reason"] = row["native_quality"].get("reason")
            if binding.get("verified") is True:
                case["recorded_ticks"] = binding.get("trace_ticks")
                outcome = (row.get("capability_report") or {}).get("native_outcome")
                if outcome and outcome.get("applicable") is True:
                    case["native_outcome"] = deepcopy(outcome)
                    result["n_authenticated_native_outcomes"] += 1
        result["cases"].append(case)
    result.update(
        index=summary["index"], complete=summary["complete"],
        n_reference_normalized_cases=summary["n_measured"],
        aggregation=aggregate["aggregation"],
        reason=None if summary["complete"] else "incomplete_fixed_long_task_scope",
    )
    return result
