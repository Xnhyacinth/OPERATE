"""Offline, fixed-suite native comparisons; no absolute quality calibration.

The reported win share is cohort relative. Fifty means a pairwise tie, never
an assumed quality baseline. No missing case is dropped or assigned zero.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from statistics import mean
from typing import Any

from evaluation.leaderboard import (
    PRIMARY_LEADERBOARD_FORMULA_VERSION,
    _macro,
    _source_denominator_key,
)

NATIVE_COMPARISON_VERSION = "0.24.0"


def _text(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or invalid {field}")
    return value.strip()


def _key(row: dict[str, Any]) -> tuple[str, int]:
    signature = _text(row, "scenario_signature")
    seed = row.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return signature, seed


def _case_id(key: tuple[str, int]) -> dict[str, Any]:
    return {"scenario_signature": key[0], "seed": key[1]}


def _outcome_issue(row: dict[str, Any]) -> str | None:
    if row.get("blocker"):
        return str(row["blocker"])
    native = row.get("native_outcome")
    if not isinstance(native, dict) or native.get("applicable") is not True:
        return "native_outcome_unavailable"
    cost = native.get("actual_cost")
    if (
        isinstance(cost, bool)
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
    ):
        return "native_cost_not_finite"
    if not all(isinstance(native.get(k), bool) for k in ("hard_failure", "feasible")):
        return "native_feasibility_unknown"
    if not all(
        isinstance(native.get(k), str) and native[k].strip()
        for k in ("objective_id", "unit")
    ):
        return "native_objective_identity_missing"
    evidence = native.get("evidence_ids")
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(e, str) and e.strip() for e in evidence)
    ):
        return "native_evidence_missing"
    return None


def _compare(left: dict[str, Any], right: dict[str, Any]) -> tuple[float, str]:
    if left["hard_failure"] != right["hard_failure"]:
        return (0.0 if left["hard_failure"] else 100.0), "hard_failure"
    if left["hard_failure"]:
        return 50.0, "both_hard_failure"
    if left["feasible"] != right["feasible"]:
        return (100.0 if left["feasible"] else 0.0), "feasibility"
    a, b = left["actual_cost"], right["actual_cost"]
    if math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7):
        return 50.0, "cost_tie"
    return (100.0 if a < b else 0.0), "native_cost"


def aggregate_native_comparisons(
    rows: list[dict[str, Any]],
    suite_rows: list[dict[str, Any]],
    *,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Compare complete models on a fixed, validated suite without execution.

    Rows may omit strata (the suite supplies authoritative strata); any supplied
    strata must agree. Duplicates and outside-suite rows disqualify that model.
    Objective or unit mismatches invalidate the whole pair, not just one case.
    """
    if not suite_rows:
        raise ValueError("suite must not be empty")
    suite = {}
    sources: dict[str, tuple[str, str]] = {}
    groups: dict[tuple[str, str, str], list[tuple[str, int]]] = defaultdict(list)
    for row in suite_rows:
        key = _key(row)
        if key in suite:
            raise ValueError(f"duplicate suite case: {key}")
        domain, backend = _text(row, "domain"), _text(row, "backend_kind")
        source = _source_denominator_key(row)
        if not source.strip():
            raise ValueError("empty source denominator key")
        stratum = domain, backend
        if source in sources and sources[source] != stratum:
            raise ValueError("source identity spans domain/backend strata")
        sources[source] = stratum
        suite[key] = (domain, backend, source)
        groups[(domain, backend, source)].append(key)
    keys = sorted(suite)
    all_models = set(models or [])
    if any(not isinstance(m, str) or not m.strip() for m in all_models):
        raise ValueError("models must contain non-empty names")
    by_model: dict[str, dict[tuple[str, int], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    issues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model = _text(row, "model")
        all_models.add(model)
        try:
            key = _key(row)
        except ValueError as exc:
            issues[model].append({"reason": str(exc)})
            continue
        if key not in suite:
            issues[model].append({**_case_id(key), "reason": "outside_suite"})
            continue
        by_model[model][key].append(row)
    coverage = {}
    for model in sorted(all_models):
        usable = 0
        missing = []
        for key in keys:
            candidates = by_model[model].get(key, [])
            if not candidates:
                missing.append(_case_id(key))
                continue
            if len(candidates) != 1:
                issues[model].append({**_case_id(key), "reason": "duplicate_case"})
                continue
            row = candidates[0]
            mismatch = any(
                row.get(field) is not None and row[field] != expected
                for field, expected in zip(
                    ("domain", "backend_kind", "source_denominator_key"),
                    suite[key],
                    strict=True,
                )
            )
            if (
                isinstance(row.get("case_ledger"), dict)
                and row["case_ledger"].get("source_denominator_key") is not None
            ):
                mismatch |= (
                    row["case_ledger"]["source_denominator_key"] != suite[key][2]
                )
            reason = "suite_identity_mismatch" if mismatch else _outcome_issue(row)
            if reason:
                issues[model].append({**_case_id(key), "reason": reason})
            else:
                usable += 1
        coverage[model] = {
            "complete": usable == len(suite) and not issues[model],
            "usable_cases": usable,
            "required_cases": len(suite),
            "missing_cases": missing,
            "issues": issues[model],
        }
    cohort = [m for m in sorted(all_models) if coverage[m]["complete"]]
    pairwise = []
    opponent_scores: dict[str, list[float]] = defaultdict(list)
    for left, right in combinations(cohort, 2):
        cases = []
        values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        blockers = []
        for key in keys:
            a = by_model[left][key][0]["native_outcome"]
            b = by_model[right][key][0]["native_outcome"]
            if (a["objective_id"], a["unit"]) != (b["objective_id"], b["unit"]):
                blockers.append(
                    {**_case_id(key), "reason": "objective_or_unit_mismatch"}
                )
                continue
            score, reason = _compare(a, b)
            values[suite[key]].append(score)
            difference = a["actual_cost"] - b["actual_cost"]
            cases.append(
                {
                    **_case_id(key),
                    "left_cost": a["actual_cost"],
                    "right_cost": b["actual_cost"],
                    "cost_difference": difference
                    if math.isfinite(difference)
                    else None,
                    "unit": a["unit"],
                    "objective_id": a["objective_id"],
                    "left_win_share": score,
                    "reason": reason,
                    "left_evidence_ids": a["evidence_ids"],
                    "right_evidence_ids": b["evidence_ids"],
                }
            )
        pair = {
            "pair": [left, right],
            "applicable": not blockers,
            "blockers": blockers,
            "cases": cases,
            "left_win_share": None,
            "domain_scores": {},
            "backend_scores": {},
            "source_scores": {},
            "required_cases": len(suite),
        }
        if not blockers:
            score, source_scores, backend_scores, domain_scores = _macro(values)
            pair.update(
                left_win_share=score,
                source_scores=source_scores,
                backend_scores=backend_scores,
                domain_scores=domain_scores,
            )
            opponent_scores[left].append(score)
            opponent_scores[right].append(100.0 - score)
        pairwise.append(pair)
    leaderboard = []
    for model in sorted(all_models):
        eligible = (
            len(cohort) > 1
            and coverage[model]["complete"]
            and len(opponent_scores[model]) == len(cohort) - 1
        )
        leaderboard.append(
            {
                "model": model,
                "score": mean(opponent_scores[model]) if eligible else None,
                "rank": None,
                "complete": coverage[model]["complete"],
                "compared_opponents": len(opponent_scores[model]),
            }
        )
    leaderboard.sort(
        key=lambda r: (r["score"] is None, -(r["score"] or 0.0), r["model"])
    )
    previous = None
    rank = None
    for index, row in enumerate(leaderboard, 1):
        if row["score"] is None:
            continue
        if previous is None or not math.isclose(
            row["score"], previous, rel_tol=1e-10, abs_tol=1e-7
        ):
            rank = index
        row["rank"] = rank
        previous = row["score"]
    domains = {d for d, _, _ in groups}
    weights = []
    for key in keys:
        domain, backend, source = suite[key]
        backends = {b for d, b, _ in groups if d == domain}
        n_sources = sum(d == domain and b == backend for d, b, _ in groups)
        weight = 1.0 / (
            len(domains)
            * len(backends)
            * n_sources
            * len(groups[(domain, backend, source)])
        )
        weights.append({**_case_id(key), "weight": weight})
    return {
        "schema_version": "1.0",
        "scoring_version": NATIVE_COMPARISON_VERSION,
        "score_type": "cohort_relative_win_share",
        "absolute_primary_score": False,
        "aggregation": PRIMARY_LEADERBOARD_FORMULA_VERSION,
        "comparison_order": ["hard_failure", "feasibility", "lower_native_cost"],
        "cost_tie_tolerance": {"relative": 1e-10, "absolute": 1e-7},
        "requested_models": sorted(all_models),
        "complete_cohort": cohort,
        "suite_size": len(suite),
        "case_weights": weights,
        "coverage": coverage,
        "pairwise": pairwise,
        "leaderboard": leaderboard,
    }
