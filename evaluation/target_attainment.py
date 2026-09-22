"""Fixed-target attainment, with complete-denominator hierarchical aggregation.

Targets are established upstream independently of model results. This module
accepts evidence-linked binary measurements; it never estimates a reference
from competing models or policies. Missing evidence is not a failed task.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evaluation.leaderboard import PRIMARY_LEADERBOARD_FORMULA_VERSION, _macro

SCHEMA_VERSION = "operate_target_attainment.v1"
EVALUATION_VERSION = "0.23.1"


def _identity(row: dict[str, Any]) -> tuple[str, int]:
    signature, seed = row.get("scenario_signature"), row.get("seed")
    if not isinstance(signature, str) or not signature.strip():
        raise ValueError("scenario_signature must be nonempty text")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    return signature, seed


def _evidence(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _measurement(row: dict[str, Any]) -> tuple[float | None, bool, str | None]:
    safety = row.get("safety")
    if not isinstance(safety, dict) or safety.get("verified") is not True:
        return None, False, "unverified_safety"
    if not isinstance(safety.get("hard_failure"), bool) or not _evidence(
        safety.get("evidence_ids")
    ):
        return None, False, "invalid_safety_evidence"
    measurement = row.get("measurement")
    if not isinstance(measurement, dict):
        return None, False, "missing_measurement"
    score = measurement.get("score")
    attained = measurement.get("attained")
    if (
        isinstance(score, bool)
        or not isinstance(score, (float, int))
        or score not in (0.0, 100.0)
        or not isinstance(attained, bool)
        or score != (100.0 if attained else 0.0)
    ):
        return None, False, "invalid_binary_measurement"
    if not _evidence(measurement.get("evidence_ids")):
        return None, False, "missing_measurement_evidence"
    reason = measurement.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return None, False, "missing_measurement_reason"
    if safety["hard_failure"]:
        return 0.0, True, None
    return float(score), False, None


def aggregate_target_attainment(
    rows: list[dict[str, Any]],
    suite_rows: list[dict[str, Any]],
    *,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Return complete-model ranks and explicit incomplete-model coverage.

    ``models`` includes expected models with zero observations. Rows outside
    the canonical suite invalidate their model rather than being silently
    dropped. Malformed suite identities or unidentified models raise an error.
    Safety and target measurements are verified upstream against bound artifacts;
    proven hard failure gates a valid measurement, never hides missing targets.
    """
    if not suite_rows:
        raise ValueError("suite_rows must be nonempty")
    suite: dict[tuple[str, int], dict[str, Any]] = {}
    for case in suite_rows:
        key = _identity(case)
        if key in suite:
            raise ValueError(f"duplicate suite case: {key}")
        for field in ("domain", "backend_kind", "source_denominator_key"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"suite case missing {field}: {key}")
        suite[key] = case

    grouped: dict[str, list[dict[str, Any]]] = {}
    if models is not None and not models:
        raise ValueError("declared models must be nonempty")
    for model in models or []:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be nonempty text")
        if model in grouped:
            raise ValueError("duplicate declared model")
        grouped[model] = []
    for row in rows:
        model = row.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be nonempty text")
        if models is not None and model not in grouped:
            raise ValueError("row model outside declared models")
        grouped.setdefault(model, []).append(row)

    results = []
    for model, observations in sorted(grouped.items()):
        issues: list[dict[str, Any]] = []
        by_case: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in observations:
            try:
                key = _identity(row)
            except ValueError as error:
                issues.append({"reason": "invalid_case_identity", "detail": str(error)})
                continue
            if key not in suite:
                issues.append({"reason": "unexpected_case", "case": list(key)})
            else:
                by_case[key].append(row)
        scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        n_passed = n_scored = n_hard_failures = 0
        cases = []
        for key, case in suite.items():
            candidates = by_case.get(key, [])
            score, hard_failure, reason = None, False, None
            if len(candidates) != 1:
                reason = "missing_case" if not candidates else "duplicate_case"
            else:
                score, hard_failure, reason = _measurement(candidates[0])
            if reason:
                issues.append({"reason": reason, "case": list(key)})
            else:
                n_scored += 1
                n_passed += int(score == 100.0)
                n_hard_failures += int(hard_failure)
                scores[
                    (
                        case["domain"],
                        case["backend_kind"],
                        case["source_denominator_key"],
                    )
                ].append(score)
            cases.append(
                {
                    "scenario_signature": key[0],
                    "seed": key[1],
                    "score": score,
                    "hard_failure": hard_failure,
                    "reason": reason,
                    "measurement": candidates[0].get("measurement")
                    if len(candidates) == 1
                    else None,
                    "safety": candidates[0].get("safety")
                    if len(candidates) == 1
                    else None,
                }
            )
        complete = not issues and n_scored == len(suite)
        if complete:
            score, sources, backends, domains = _macro(scores)
        else:
            score, sources, backends, domains = None, {}, {}, {}
        results.append(
            {
                "model": model,
                "score": score,
                "rank": None,
                "complete": complete,
                "n_expected": len(suite),
                "n_observed": len(observations),
                "n_scored": n_scored,
                "n_passed": n_passed,
                "n_hard_failures": n_hard_failures,
                "micro_pass_rate": 100.0 * n_passed / len(suite) if complete else None,
                "source_scores": sources,
                "backend_scores": backends,
                "domain_scores": domains,
                "issues": issues,
                "cases": cases,
            }
        )
    leaderboard = sorted(
        (result for result in results if result["complete"]),
        key=lambda result: (-result["score"], result["model"]),
    )
    previous_score, previous_rank = None, None
    for position, result in enumerate(leaderboard, 1):
        if result["score"] != previous_score:
            previous_rank = position
        result["rank"] = previous_rank
        previous_score = result["score"]
    return {
        "schema_version": SCHEMA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "aggregation": PRIMARY_LEADERBOARD_FORMULA_VERSION,
        "score_units": "percent_target_attainment",
        "uncertainty": "single_observation_per_case; repeated-run reliability not estimated",
        "n_expected": len(suite),
        "models": results,
        "leaderboard": leaderboard,
        "incomplete_models": [result for result in results if not result["complete"]],
    }
