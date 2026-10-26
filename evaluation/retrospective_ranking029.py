"""Fixed-cohort retrospective Q ordering under recorded measurement bounds.

These are interval-compatible competition ranks, not sampling confidence or
future-run predictions. Overlap is unresolved ordering, not an equivalence class.
The caller supplies authenticated output from build_outcome_analysis.
"""

from copy import deepcopy
from itertools import combinations
import math


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _bounds(value, *, low, high):
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_finite(v) for v in value)
        and low <= value[0] <= value[1] <= high
    )


def build_retrospective_ranking(analysis: dict) -> dict:
    """Retain the existing point ranking and expose measurement ambiguity."""
    if analysis.get("schema_version") != "operate_outcome_analysis029.v1":
        raise ValueError("retrospective_requires_outcome_analysis029")
    models = {}
    for model in analysis["models"]:
        name = model["model"]
        if not isinstance(name, str) or not name or name in models:
            raise ValueError("retrospective_duplicate_or_invalid_model")
        q = model["overall"]["Q"]
        if not _bounds(q.get("bounds"), low=0, high=100) or (
            q.get("score") is not None
            and (
                not _finite(q["score"])
                or not q["bounds"][0] <= q["score"] <= q["bounds"][1]
            )
        ):
            raise ValueError("retrospective_invalid_measurement_bounds")
        models[name] = q
    cohort = analysis["rank_sensitivity"]["cohort"]
    baseline = analysis["rank_sensitivity"]["baseline_ranks"]
    if (
        not isinstance(cohort, list)
        or len(cohort) != len(set(cohort))
        or not set(cohort) <= models.keys()
        or set(baseline) != set(cohort)
    ):
        raise ValueError("retrospective_invalid_cohort")
    for name in cohort:
        q = models[name]
        if (
            q["score"] is None
            or q["measured_cases"] != q["expected_cases"]
            or type(baseline[name]) is not int
            or not 1 <= baseline[name] <= len(cohort)
        ):
            raise ValueError("retrospective_incomplete_ranked_model")
    cohort = sorted(cohort)
    rows = []
    for name in sorted(models):
        q = models[name]
        low, high = q["bounds"]
        others = [models[m]["bounds"] for m in cohort if m != name]
        declared_others = [v["bounds"] for m, v in models.items() if m != name]
        rows.append(
            {
                "model": name,
                "Q": q["score"],
                "complete_eligible": name in cohort,
                "point_rank": baseline.get(name),
                "measurement_bounds": list(q["bounds"]),
                "declared_panel_rank_range": [
                    1 + sum(v[0] > high for v in declared_others),
                    1 + sum(v[1] > low for v in declared_others),
                ],
                "measurement_rank_range": [
                    1 + sum(v[0] > high for v in others),
                    1 + sum(v[1] > low for v in others),
                ]
                if name in cohort
                else None,
            }
        )
    pairs = {}
    for pair in analysis["pairwise"]:
        a, b = pair["a"], pair["b"]
        if a not in models or b not in models or a == b:
            raise ValueError("retrospective_foreign_pair")
        key = tuple(sorted((a, b)))
        if key in pairs:
            raise ValueError("retrospective_duplicate_pair")
        bounds = pair["Q_difference_bounds"]
        if not _bounds(bounds, low=-100, high=100):
            raise ValueError("retrospective_invalid_pair_bounds")
        pairs[key] = pair
    if set(pairs) != set(combinations(sorted(models), 2)):
        raise ValueError("retrospective_pair_population_mismatch")
    ordering = []
    for a, b in combinations(sorted(cohort), 2):
        pair = pairs[a, b]
        bounds = pair["Q_difference_bounds"]
        if pair["a"] != a:
            bounds = [-bounds[1], -bounds[0]]
        decision = (
            "a_above"
            if bounds[0] > 0
            else "b_above"
            if bounds[1] < 0
            else "exact_tie"
            if bounds == [0, 0]
            else "unresolved"
        )
        ordering.append(
            {"a": a, "b": b, "Q_difference_bounds": list(bounds), "ordering": decision}
        )
    return {
        "schema_version": "operate_retrospective_ranking029.v1",
        "evaluation_version": "0.29.1",
        "base_evaluation_version": "0.28.0",
        "headline_score_changed": False,
        "rank_cohort_count": len(cohort),
        "total_declared_model_count": len(models),
        "cohort": deepcopy(cohort),
        "rows": rows,
        "pairwise": ordering,
        "interpretation": "fixed_panel_recorded_outcome_ordering_not_general_ability_ranking",
        "rank_range_interpretation": "interval_compatible_competition_ranks_not_sampling_confidence_intervals",
        "declared_panel_rank_range_interpretation": "hypothetical_interval_compatible_ranks_including_missing_models_not_assigned_ranks",
        "overlap_interpretation": "unresolved_pairwise_order_not_transitive_tie_groups",
    }
