"""Fixed-policy calibrated operational quality; independent of the model cohort.

This is a scoring protocol, not run/publication certification. Reference
measurements must be authenticated by the reference-bundle reader before use.
"""

from __future__ import annotations

import math
from statistics import mean

VERSION = "native_quality.v1"
REFERENCE_POLICIES = ("wait_only", "greedy_heuristic", "oracle_offline", "random")
# Native double-precision accounting tolerance; not fitted to model results.
ABS_TOLERANCE = 1e-7
REL_TOLERANCE = 1e-10
MAKESPAN_OBJECTIVE = "dynasched_flexible_job_shop.makespan_unfinished_penalty.v1"


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _valid(measurement):
    return (
        measurement.get("applicable") is True
        and _number(measurement.get("actual_cost"))
        and all(type(measurement.get(k)) is bool for k in ("feasible", "hard_failure"))
        and all(
            isinstance(measurement.get(k), str) and measurement[k]
            for k in ("objective_id", "unit")
        )
        and isinstance(measurement.get("evidence_ids"), list)
        and bool(measurement["evidence_ids"])
        and all(isinstance(v, str) and v for v in measurement["evidence_ids"])
    )


def calibrate_row(references):
    """Freeze extrema of four predeclared policies, never extrema of LLM scores.

    All policies need two measured repetitions. Infeasible/hard-failed policies
    are documented but cannot set the feasible quality scale. An unstable or
    degenerate reference set is unavailable, not silently replaced with wait.
    """
    result = {
        "schema_version": VERSION,
        "status": "unavailable",
        "reference_policies": list(REFERENCE_POLICIES),
        "reference_is_optimum": False,
    }

    def fail(reason):
        return {**result, "reason": reason}

    if set(references) != set(REFERENCE_POLICIES):
        return fail("incomplete_fixed_reference_set")
    identities, costs, evidence = set(), {}, []
    max_noise = 0.0
    for policy in REFERENCE_POLICIES:
        pair = references[policy]
        if len(pair) != 2 or not all(_valid(m) for m in pair):
            return fail("reference_measurement_unavailable")
        for m in pair:
            identities.add((m["objective_id"], m["unit"]))
        a, b = pair
        if any(
            a.get(k) != b.get(k) for k in ("feasible", "hard_failure", "task_success")
        ):
            return fail("reference_constraint_instability")
        noise = abs(a["actual_cost"] - b["actual_cost"])
        max_noise = max(max_noise, noise)
        if not math.isclose(
            a["actual_cost"],
            b["actual_cost"],
            rel_tol=REL_TOLERANCE,
            abs_tol=ABS_TOLERANCE,
        ):
            return fail("reference_cost_instability")
        evidence.extend(v for m in pair for v in m["evidence_ids"])
        if a["feasible"] and not a["hard_failure"]:
            costs[policy] = mean(m["actual_cost"] for m in pair)
    if len(identities) != 1:
        return fail("reference_objective_mismatch")
    if len(costs) < 2:
        return fail("insufficient_feasible_references")
    weak, strong = max(costs.values()), min(costs.values())
    gap = weak - strong
    minimum = max(ABS_TOLERANCE, 100 * max_noise)
    objective, unit = next(iter(identities))
    # Completed makespan has a natural zero. Nearly identical dispatchers must
    # not turn a tiny elapsed-time improvement into almost the maximum utility.
    scale = max(gap, strong) if objective == MAKESPAN_OBJECTIVE else gap
    if not math.isfinite(scale) or scale <= minimum:
        return fail("degenerate_reference_gap")
    return {
        **result,
        "status": "ready",
        "objective_id": objective,
        "unit": unit,
        "weak_cost": weak,
        "strong_cost": strong,
        "normalization_scale": scale,
        "normalization_basis": "makespan_reference_floor"
        if objective == MAKESPAN_OBJECTIVE
        else "fixed_policy_gap",
        "minimum_anchor_gap": minimum,
        "eligible_policy_costs": costs,
        "weak_policies": [p for p, c in costs.items() if c == weak],
        "strong_policies": [p for p, c in costs.items() if c == strong],
        "anchor_evidence_ids": sorted(set(evidence)),
        "max_repeat_cost_difference": max_noise,
    }


def score_native_quality(measurement, contract):
    """Signed bounded utility; missing measurement is never zero-filled."""
    base = {"schema_version": VERSION, "score": None, "reference_is_optimum": False}
    if contract.get("status") != "ready":
        return {**base, "reason": "reference_not_calibrated"}
    if not _valid(measurement):
        return {**base, "reason": "native_measurement_unavailable"}
    if any(measurement[k] != contract.get(k) for k in ("objective_id", "unit")):
        return {**base, "reason": "native_objective_mismatch"}
    weak, strong = contract.get("weak_cost"), contract.get("strong_cost")
    if not _number(weak) or not _number(strong) or weak < strong:
        return {**base, "reason": "invalid_reference_scale"}
    actual, gap = measurement["actual_cost"], weak - strong
    scale = (
        max(gap, strong) if measurement["objective_id"] == MAKESPAN_OBJECTIVE else gap
    )
    minimum = contract.get("minimum_anchor_gap")
    evidence = contract.get("anchor_evidence_ids")
    if (
        not math.isfinite(gap)
        or not _number(minimum)
        or minimum <= 0
        or scale <= minimum
        or not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(v, str) and v for v in evidence)
    ):
        return {**base, "reason": "invalid_reference_scale_or_evidence"}
    if contract.get("normalization_scale") != scale:
        return {**base, "reason": "invalid_reference_normalization"}
    difference = weak - actual
    if math.isfinite(difference):
        normalized = 100 * (difference / scale)
    else:
        magnitude = max(abs(weak), abs(strong), abs(actual))
        normalized = (
            100 * ((weak / magnitude - actual / magnitude) / (scale / magnitude))
            if scale / magnitude
            else math.inf
        )
    if not math.isfinite(normalized):
        return {**base, "reason": "unrepresentable_quality"}
    raw = 100 * (difference / gap) if gap > minimum else None
    if raw is not None and not math.isfinite(raw):
        raw = None
    failed = measurement["hard_failure"] or not measurement["feasible"]
    return {
        **base,
        "score": -100.0 if failed else 100.0 * (normalized / (100.0 + abs(normalized))),
        "utility_mapping": "signed_rational_v1",
        "normalization_scale": scale,
        "raw_score": raw,
        "raw_score_unavailable_reason": "degenerate_or_unrepresentable_reference_gap"
        if raw is None
        else None,
        "native_cost": actual,
        "constraint_failed": failed,
        "task_success": measurement.get("task_success"),
        "range_clipped": False,
        "above_strong_reference": actual < strong,
        "evidence_ids": list(measurement["evidence_ids"]),
        "anchor_evidence_ids": list(contract["anchor_evidence_ids"]),
    }


def aggregate_native_rows(rows, suite_rows, models, *, bootstrap=0):
    """Complete native-score denominator; same source/backend/domain estimand.

    Callers isolate runtime/prompt/treatment cohorts before invoking this API.
    Cluster inference uses an affine conversion solely to reuse the existing
    bounded-score engine; all emitted score intervals are converted back.
    """
    from collections import defaultdict
    from evaluation.leaderboard import _macro, infer_primary_leaderboard

    expected = {(r["scenario_signature"], r["seed"]): r for r in suite_rows}
    if (
        not expected
        or len(expected) != len(suite_rows)
        or not models
        or len(set(models)) != len(models)
    ):
        raise ValueError("nonempty unique model and suite scopes required")
    indexed = defaultdict(list)
    for row in rows:
        key = (row["scenario_signature"], row["seed"])
        if key not in expected or row["model"] not in models:
            raise ValueError("native result outside declared scope")
        indexed[(row["model"], *key)].append(row)
    output, inference_rows = {}, []
    for model in models:
        costs, successes, failures = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        missing, graded, model_inference = [], [], []
        for key, spec in expected.items():
            attempts = indexed[(model, *key)]
            if len(attempts) != 1:
                missing.append(
                    {
                        "scenario_signature": key[0],
                        "reason": "missing_or_duplicate_attempt",
                    }
                )
                continue
            item = attempts[0]["native_quality"]
            value = item.get("score")
            if not _number(value) or not -100 <= value <= 100:
                missing.append(
                    {
                        "scenario_signature": key[0],
                        "reason": item.get("reason", "score_unavailable"),
                    }
                )
                continue
            source = tuple(
                spec.get(k)
                for k in ("domain", "backend_kind", "source_denominator_key")
            )
            if not all(isinstance(v, str) and v for v in source):
                raise ValueError("missing aggregation lineage")
            costs[source].append(value)
            failures[source].append(float(item["constraint_failed"]))
            if type(item.get("task_success")) is bool:
                successes[source].append(float(item["task_success"]))
            graded.append({"scenario_signature": key[0], "seed": key[1], **item})
            model_inference.append(
                {
                    "model": model,
                    **{
                        k: spec.get(k)
                        for k in (
                            "domain",
                            "backend_kind",
                            "source_denominator_key",
                            "physical_source_key",
                        )
                    },
                    "seed": key[1],
                    "discriminative_core_score": (value + 100) / 2,
                    "task_completion_raw": 0,
                }
            )
        complete = not missing
        score, sources, backends, domains = (
            _macro(costs) if costs else (None, {}, {}, {})
        )
        output[model] = {
            "index": score if complete else None,
            "complete": complete,
            "n_expected": len(expected),
            "n_measured": len(graded),
            "missing": missing,
            "rows": graded,
            "domain_scores": domains if complete else None,
            "backend_scores": backends if complete else None,
            "source_scores": sources if complete else None,
            "task_success_rate": _macro(successes)[0]
            if complete and sum(map(len, successes.values())) == len(expected)
            else None,
            "constraint_failure_rate": _macro(failures)[0] if complete else None,
            "above_strong_reference_fraction": mean(
                x.get("above_strong_reference", False) for x in graded
            )
            if complete
            else None,
            "upper_bound_fraction": mean(x["score"] == 100 for x in graded)
            if complete
            else None,
            "lower_bound_fraction": mean(x["score"] == -100 for x in graded)
            if complete
            else None,
        }
        if complete:
            inference_rows.extend(model_inference)
    pairwise, inference_error = [], None
    if bootstrap and all(x["complete"] for x in output.values()):
        try:
            inferred = infer_primary_leaderboard(
                inference_rows, n_bootstrap=bootstrap, seed=1729
            )
            for item in inferred["leaderboard"]:
                ci = item["primary_cluster_ci"]
                output[item["model"]]["ci"] = {
                    "lo": 2 * ci["lo"] - 100,
                    "hi": 2 * ci["hi"] - 100,
                }
            for pair in inferred["primary_pairwise"]:
                pairwise.append(
                    {
                        **pair,
                        "mean_diff": pair["mean_diff"] * 2,
                        "ci_lo": pair["ci_lo"] * 2,
                        "ci_hi": pair["ci_hi"] * 2,
                        "estimand": VERSION,
                    }
                )
        except ValueError as exc:
            inference_error = str(exc)
    complete_all = all(x["complete"] for x in output.values())
    return {
        "schema_version": VERSION,
        "aggregation": "effective_source_backend_domain_macro_v1",
        "models": output,
        "ranking": sorted(models, key=lambda m: (-output[m]["index"], m))
        if complete_all
        else [],
        "complete": complete_all,
        "pairwise": pairwise,
        "inference_error": inference_error,
    }
