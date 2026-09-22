"""Outcome harm remains visible beside the clipped wait-relative primary."""

from collections import defaultdict
import math
from statistics import mean
from typing import Any

from evaluation.leaderboard import _macro
from evaluation.scorer import native_outcome_diagnostics


def summarize_native_outcomes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Report harm on measured outcomes, with missingness and lineage explicit.

    This does not grant eligibility or sum incomparable native cost units.
    Callers own treatment matching and qualification of the input rows.
    """
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_model[
            str(
                row.get("model_id")
                or row.get("model")
                or row.get("agent_name")
                or "unknown"
            )
        ].append(row)
    report = {}
    schedule_reports = summarize_completed_schedule_quality(rows)
    for model, samples in sorted(by_model.items()):
        counts = {"harmed": 0, "parity": 0, "improved": 0}
        harm_by_source: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        missing_lineage = 0
        signed_changes: list[float] = []
        changes_by_source: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        severity_by_source: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for row in samples:
            outcome = native_outcome_diagnostics(row.get("counterfactual"))
            if not outcome["applicable"]:
                continue
            category = outcome["outcome_vs_wait"]
            counts[category] += 1
            relative = outcome["wait_relative_change"]
            if relative is not None:
                signed_changes.append(relative)
            ledger = row.get("case_ledger") or {}
            source = row.get("source_denominator_key") or ledger.get(
                "source_denominator_key"
            )
            key = (row.get("domain"), row.get("backend_kind"), source)
            if not all(isinstance(value, str) and value.strip() for value in key):
                missing_lineage += 1
                continue
            harm_by_source[key].append(float(category == "harmed"))
            if relative is not None:
                changes_by_source[key].append(relative)
                severity_by_source[key].append(max(0.0, -relative))
        measured = sum(counts.values())
        report[model] = {
            "schema_version": "native_outcome_summary_v2",
            "headline_score_included": False,
            "n_input": len(samples),
            "n_measured": measured,
            "n_unavailable": len(samples) - measured,
            "counts": counts,
            "sample_harm_rate": counts["harmed"] / measured if measured else None,
            "macro_harm_rate": (
                _macro(harm_by_source)[0]
                if harm_by_source and not missing_lineage
                else None
            ),
            "mean_wait_relative_change": mean(signed_changes)
            if signed_changes
            else None,
            "macro_wait_relative_change": (
                _macro(changes_by_source)[0]
                if changes_by_source and not missing_lineage
                else None
            ),
            "mean_harm_severity": (
                mean(max(0.0, -value) for value in signed_changes)
                if signed_changes
                else None
            ),
            "macro_harm_severity": (
                _macro(severity_by_source)[0]
                if severity_by_source and not missing_lineage
                else None
            ),
            "n_ratio_measured": len(signed_changes),
            "n_ratio_unavailable": len(samples) - len(signed_changes),
            "ratio_denominator": "finite_comparable_outcomes_with_positive_wait_cost",
            **schedule_reports[model],
            "n_measured_missing_lineage": missing_lineage,
            "denominator": "measured_native_outcomes_only",
        }
    return report


def completed_schedule_quality(row: dict[str, Any]) -> dict[str, Any]:
    """Expose DynaSched makespan after feasibility, never an optimality score.

    This backend's production cost is makespan plus 1000 per remaining
    operation. Once every required operation is completed, the penalty is zero.
    Native time is not assumed to equal runner ticks or physical minutes.
    """
    base = {
        "schema_version": "completed_schedule_quality_v1",
        "headline_score_included": False,
        "reference_is_optimum": False,
    }
    unavailable = {
        **base,
        "applicable": False,
        "reason": "completed_native_schedule_unproven",
    }
    completion = row.get("task_completion") or {}
    evidence = completion.get("evidence") or {}
    counts = [
        evidence.get(key)
        for key in (
            "operations_required",
            "operations_completed",
            "operations_scheduled",
            "operations_total",
            "operations_cancelled",
        )
    ]
    if not (
        row.get("status") == "ok"
        and row.get("domain") == "logistics"
        and row.get("backend_kind") == "dynasched_flexible_job_shop"
        and completion.get("applicable") is True
        and completion.get("completed") is True
        and completion.get("contract")
        == "logistics.job_shop.all_operations_scheduled.v1"
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in counts
        )
        and counts[0] > 0
        and counts[0] == counts[1] == counts[2] == counts[3] - counts[4]
        and not evidence.get("survival_floor_violation")
        and not evidence.get("chose_fatal_option")
    ):
        return unavailable
    native = (row.get("ground_truth_summary") or {}).get("cost_components") or {}
    replay = (row.get("counterfactual") or {}).get("actual_components") or {}
    makespan = native.get("production_cost")
    if not (
        set(native) == {"production_cost"}
        and native == replay
        and isinstance(makespan, (int, float))
        and not isinstance(makespan, bool)
        and math.isfinite(makespan)
        and makespan >= 0
    ):
        return {**unavailable, "reason": "native_completed_makespan_unavailable"}
    valid_ids = set(
        (row.get("trajectory_summary") or {}).get(
            "operational_agency_valid_evidence_ids"
        )
        or []
    )
    cost_ids = list(
        dict.fromkeys(
            evidence_id
            for dimension in (row.get("score") or {}).get("dimensions") or []
            if dimension.get("name") in {"economic_cost", "counterfactual_prevention"}
            for evidence_id in dimension.get("evidence_ids") or []
            if isinstance(evidence_id, str) and evidence_id in valid_ids
        )
    )
    if not cost_ids:
        return {**unavailable, "reason": "native_cost_evidence_unavailable"}
    return {
        **base,
        "applicable": True,
        "makespan": float(makespan),
        "unit": "native_simulator_time",
        "lower_is_better": True,
        "operations_completed": counts[1],
        "operations_cancelled": counts[4],
        "evidence_ids": cost_ids,
    }


def compare_completed_schedule_to_reference(
    row: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Compare two bound feasible runs against an executed policy, not an optimum."""
    base = {
        "schema_version": "executed_schedule_reference_v1",
        "reference_kind": "executed_policy",
        "reference_is_optimum": False,
        "headline_score_included": False,
    }
    unavailable = {**base, "applicable": False, "reason": "reference_identity_mismatch"}
    identity_fields = (
        "scenario_signature",
        "seed",
        "implementation_tree_sha256",
        "suite_manifest_sha256",
        "source_denominator_key",
        "interaction_mode",
    )
    if any(row.get(key) != reference.get(key) for key in identity_fields):
        return unavailable
    if any(
        not isinstance(row.get(key), str) or not row[key].strip()
        for key in identity_fields
        if key != "seed"
    ) or (
        not isinstance(row.get("seed"), int)
        or isinstance(row["seed"], bool)
        or not isinstance(reference.get("seed"), int)
        or isinstance(reference["seed"], bool)
    ):
        return unavailable
    actual_protocol = row.get("evaluation_protocol") or {}
    reference_protocol = reference.get("evaluation_protocol") or {}
    for key in ("version", "implementation_fingerprint"):
        actual_value = actual_protocol.get(key)
        reference_value = reference_protocol.get(key)
        if (
            not isinstance(actual_value, str)
            or not actual_value
            or actual_value != reference_value
        ):
            return unavailable
    for key in ("within_tick_interaction",):
        actual_value = actual_protocol.get(key)
        reference_value = reference_protocol.get(key)
        if (
            type(actual_value) is not bool
            or type(reference_value) is not bool
            or actual_value != reference_value
        ):
            return unavailable
    actual_fingerprint = row.get("evaluation_implementation_fingerprint")
    reference_fingerprint = reference.get("evaluation_implementation_fingerprint")
    if (
        not isinstance(actual_fingerprint, str)
        or not actual_fingerprint
        or actual_fingerprint != reference_fingerprint
    ):
        return unavailable
    scoring_version = (row.get("score") or {}).get("scoring_version")
    if not scoring_version or scoring_version != (reference.get("score") or {}).get(
        "scoring_version"
    ):
        return unavailable
    policy = reference.get("model") or reference.get("agent_name")
    if not isinstance(policy, str) or not policy:
        return {**unavailable, "reason": "reference_policy_identity_missing"}
    actual, baseline = (
        completed_schedule_quality(row),
        completed_schedule_quality(reference),
    )
    if not actual["applicable"] or not baseline["applicable"]:
        return {
            **unavailable,
            "reason": "both_schedules_must_be_completed_and_evidenced",
        }
    if any(
        actual[key] != baseline[key]
        for key in ("operations_completed", "operations_cancelled")
    ):
        return {**unavailable, "reason": "required_workload_mismatch"}
    delta = actual["makespan"] - baseline["makespan"]
    relative = delta / baseline["makespan"] if baseline["makespan"] > 0 else None
    if not math.isfinite(delta) or (
        relative is not None and not math.isfinite(relative)
    ):
        return {**unavailable, "reason": "nonfinite_reference_difference"}
    return {
        **base,
        "applicable": True,
        "reference_policy": policy,
        "actual_makespan": actual["makespan"],
        "reference_makespan": baseline["makespan"],
        "makespan_delta": delta,
        "relative_makespan_gap": relative,
        "lower_is_better": True,
        "unit": "native_simulator_time",
        "actual_evidence_ids": actual["evidence_ids"],
        "reference_evidence_ids": baseline["evidence_ids"],
    }


def summarize_completed_schedule_quality(
    rows: list[dict[str, Any]],
    *,
    reference_policy: str = "oracle_offline",
) -> dict[str, dict[str, Any]]:
    """Publish native times per task and only matched dimensionless policy gaps.

    A macro comparison is withheld when any feasible candidate lacks a unique
    comparable reference. Missing policy execution is not zero relative gap.
    """
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model = str(
            row.get("model_id")
            or row.get("model")
            or row.get("agent_name")
            or "unknown"
        )
        by_model[model].append(row)
        if model == reference_policy:
            references[str(row.get("scenario_signature") or "")].append(row)
    report = {}
    for model, samples in by_model.items():
        schedules = []
        gaps: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        n_applicable = 0
        n_matched = 0
        for row in samples:
            if row.get("backend_kind") != "dynasched_flexible_job_shop":
                continue
            quality = completed_schedule_quality(row)
            comparison = {
                "applicable": False,
                "reason": "completed_native_schedule_unproven",
            }
            if quality["applicable"]:
                n_applicable += 1
                matches = [
                    result
                    for reference in references[
                        str(row.get("scenario_signature") or "")
                    ]
                    for result in [
                        compare_completed_schedule_to_reference(row, reference)
                    ]
                    if result["applicable"]
                ]
                comparison = (
                    matches[0]
                    if len(matches) == 1
                    else {
                        "applicable": False,
                        "reason": "ambiguous_matching_reference"
                        if matches
                        else "matching_reference_unavailable",
                    }
                )
                if comparison["applicable"]:
                    n_matched += 1
                    relative = comparison["relative_makespan_gap"]
                    if relative is not None:
                        gaps[
                            (
                                row["domain"],
                                row["backend_kind"],
                                row["source_denominator_key"],
                            )
                        ].append(relative)
            schedules.append(
                {
                    "scenario_signature": row.get("scenario_signature"),
                    "source_denominator_key": row.get("source_denominator_key"),
                    **quality,
                    "reference_comparison": comparison,
                }
            )
        n_ratios = sum(len(values) for values in gaps.values())
        report[model] = {
            "completed_schedule_quality": schedules,
            "n_schedule_quality_applicable": n_applicable,
            "n_schedule_quality_unavailable": len(schedules) - n_applicable,
            "required_reference_policy": reference_policy,
            "reference_is_optimum": False,
            "n_matched_reference": n_matched,
            "reference_coverage": n_matched / n_applicable if n_applicable else None,
            "macro_relative_makespan_gap": (
                _macro(gaps)[0] if gaps and n_ratios == n_applicable else None
            ),
            "reference_ratio_denominator": "all_completed_evidenced_candidate_schedules",
        }
    return report
