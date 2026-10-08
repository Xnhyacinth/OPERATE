"""Descriptive diagnostics of authenticated 0.28 outcomes; no new ability score.

The caller authenticates source policies and episode artifacts. This module
validates their report-level relationships, preserves fixed weights and exposes
information hidden by clipping, bottlenecks and the headline mean.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from itertools import combinations
import math

from evaluation.operational_outcome028 import aggregate_outcome028
from evaluation.operational_weights import _key, _unique, validate_task_weights

VERSION = "0.29.0"


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _interval(value, point=None):
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_finite(v) and 0 <= v <= 100 for v in value)
        and value[0] <= value[1]
        and (point is None or value[0] <= point <= value[1])
    )


def _applicable(contract):
    return not (
        contract["backend_kind"] == "citylearn"
        and contract.get("service", {}).get("kind")
        == "automatic_exogenous_load_service"
        and contract["service"].get("applicable") is False
    )


def _validate(report):
    if report.get("evaluation_version") != "0.28.0":
        raise ValueError("analysis_requires_outcome028_report")
    contracts = _unique(report["source_contracts"]["contracts"])
    weights = validate_task_weights(
        report["task_weight_manifest"], report["source_contracts"]
    )
    if report.get("n_expected") != len(contracts):
        raise ValueError("analysis_declared_population_mismatch")
    models = [row["model"] for row in report["models"]]
    if (
        not models
        or any(not isinstance(m, str) or not m for m in models)
        or len(set(models)) != len(models)
    ):
        raise ValueError("analysis_invalid_model_population")
    rows = report["graded_episodes"]
    indexed = {}
    for row in rows:
        key = _key(row)
        identity = row["model"], *key
        if identity in indexed or row["model"] not in models or key not in contracts:
            raise ValueError("analysis_duplicate_or_foreign_case")
        contract, outcome = contracts[key], row["outcome028"]
        for field in (
            "domain",
            "backend_kind",
            "task_family",
            "physical_source_cluster",
            "scenario_sha256",
        ):
            if field in row and row[field] != contract.get(field):
                raise ValueError("analysis_foreign_case_metadata")
        for field in ("Q", "N", "F", "C"):
            value = row.get(field)
            if value is not None and (
                not _finite(value) or (field != "C" and not 0 <= value <= 100)
            ):
                raise ValueError("analysis_invalid_numeric_component")
        if (
            row.get("C") is not None
            and row["C"] < 0
            and contract["native_cost_value_domain"] != "signed"
        ):
            raise ValueError("analysis_unsigned_negative_cost")
        if row.get("Q") != outcome.get("score") or row.get("N") != outcome.get(
            "native_quality"
        ):
            raise ValueError("analysis_component_outcome_mismatch")
        if outcome.get("completion_applicable") is not _applicable(contract):
            raise ValueError("analysis_completion_applicability_mismatch")
        measured = outcome.get("completion_measured")
        if type(measured) is not bool or (
            measured
            and (row.get("F") is None or row["F"] != outcome.get("completion_score"))
        ):
            raise ValueError("analysis_completion_measurement_mismatch")
        if not measured and outcome.get("completion_score") is not None:
            raise ValueError("analysis_unmeasured_completion_value")
        if not _applicable(contract) and (measured or row.get("F") is not None):
            raise ValueError("analysis_inapplicable_completion_value")
        for field in ("hard_failure", "native_catastrophe_applicable"):
            if outcome.get(field) is not None and type(outcome[field]) is not bool:
                raise ValueError("analysis_invalid_safety_state")
        evidence = outcome.get("evidence_ids")
        if not isinstance(evidence, list) or any(
            not isinstance(v, str) or not v for v in evidence
        ):
            raise ValueError("analysis_invalid_evidence_ids")
        if outcome.get("evidence_qualified") and not evidence:
            raise ValueError("analysis_missing_evidence_ids")
        if row.get("Q") is not None and not _interval(
            outcome.get("score_interval"), row["Q"]
        ):
            raise ValueError("analysis_invalid_score_interval")
        mission = row.get("native_measurement027") or {}
        if mission.get("native_objective") != row.get("C"):
            raise ValueError("analysis_native_cost_mismatch")
        if any(not _finite(v) for v in (mission.get("components") or {}).values()):
            raise ValueError("analysis_invalid_raw_component")
        indexed[identity] = row
    # Reuse the frozen qualification, missingness and aggregate validation.
    summary = aggregate_outcome028(rows, contracts, weights, models=models)
    return contracts, weights, sorted(models), indexed, summary


def _native_bounds(row):
    if row is None or not row["outcome028"].get("evidence_qualified"):
        return [0.0, 100.0]
    outcome = row["outcome028"]
    if outcome.get("hard_failure") is True:
        return [0.0, 0.0]
    quality = outcome.get("native_quality_if_no_hard_failure")
    if quality is None:
        quality = row.get("N")
    if quality is None:
        return [0.0, 100.0]
    normalized = row.get("normalization028")
    interval = (
        normalized.get("quality_interval")
        if normalized is not None
        else (row.get("native_measurement027") or {}).get(
            "native_quality_interval", [quality, quality]
        )
    )
    # The underlying 0.27 reader permits last-bit differences between the
    # overflow-resistant point formula and the recorded interval expression.
    if (
        not _interval(interval)
        or not interval[0] - 1e-8 <= quality <= interval[1] + 1e-8
    ):
        raise ValueError("analysis_invalid_native_precision_interval")
    return [
        0.0 if outcome.get("hard_failure") is None else min(interval[0], quality),
        max(interval[1], quality),
    ]


def _bounds(row):
    if row is None:
        return [0.0, 100.0]
    outcome = row["outcome028"]
    if row["Q"] is not None:
        return outcome["score_interval"]
    if not outcome.get("evidence_qualified"):
        return [0.0, 100.0]
    # Unlike the frozen 0.28 identification bounds (conditioned on point C),
    # these diagnostics also cover authenticated native rounding uncertainty.
    upper = _native_bounds(row)[1]
    if outcome.get("completion_measured"):
        upper = min(upper, outcome["completion_score"])
    return [0.0, upper]


def _metric(keys, rows, contracts, weights, metric):
    eligible = [k for k in keys if metric != "F" or _applicable(contracts[k])]
    mass = math.fsum(weights[k] for k in eligible)
    known, bounds = {}, {}
    for key in eligible:
        row = rows.get(key)
        outcome = row["outcome028"] if row else {}
        value = row.get(metric) if row and outcome.get("evidence_qualified") else None
        if metric == "F" and not outcome.get("completion_measured"):
            value = None
        if metric == "Q" and not outcome.get("publish_eligible"):
            value = None
        if value is not None:
            known[key] = value
        bounds[key] = (
            _bounds(row)
            if metric == "Q"
            else _native_bounds(row)
            if metric == "N"
            else [value, value]
            if value is not None
            else [0, 100]
        )
    complete = bool(eligible) and len(known) == len(eligible)
    return {
        "score": math.fsum(weights[k] * known[k] for k in eligible) / mass
        if complete
        else None,
        "bounds": [
            math.fsum(weights[k] * bounds[k][i] for k in eligible) / mass
            for i in (0, 1)
        ]
        if mass
        else None,
        "expected_cases": len(eligible),
        "measured_cases": len(known),
        "expected_weight_mass": mass,
        "measured_weight_mass": math.fsum(weights[k] for k in known),
        "applicable": bool(eligible),
    }


def _slice(keys, rows, contracts, weights):
    return {m: _metric(keys, rows, contracts, weights, m) for m in ("Q", "N", "F")}


def _case(row, contract, weight):
    outcome = row["outcome028"] if row else {}
    mission = row.get("native_measurement027") or {} if row else {}
    q, n, f = (row.get(m) if row else None for m in ("Q", "N", "F"))
    if outcome.get("hard_failure") is True:
        bottleneck = "hard_failure"
    elif q is None:
        bottleneck = "unknown"
    elif not _applicable(contract):
        bottleneck = "native_only"
    else:
        bottleneck = "fulfillment" if f < n else "native_quality" if n < f else "joint"
    return {
        "scenario_signature": contract["scenario_signature"],
        "seed": contract["seed"],
        "source_contract_sha256": contract.get("contract_sha256"),
        "domain": contract["domain"],
        "task_family": contract["task_family"],
        "physical_source_cluster": contract.get("physical_source_cluster"),
        "weight": weight,
        "C": row.get("C") if row else None,
        "Q": q,
        "N": n,
        "F": outcome.get("completion_score"),
        "completion_applicable": _applicable(contract),
        "hard_failure": outcome.get("hard_failure"),
        "native_catastrophe_applicable": outcome.get("native_catastrophe_applicable"),
        "objective_id": mission.get("scored_objective_id")
        or mission.get("native_objective_id"),
        "raw_cost_components": deepcopy(mission.get("components")),
        "evidence_qualified": outcome.get("evidence_qualified") is True,
        "evidence_ids": list(outcome.get("evidence_ids", [])),
        "origin": deepcopy(row.get("origin")) if row else None,
        "bottleneck": bottleneck,
        "negative_cost_saturation": bool(
            row and row.get("C") is not None and row["C"] < 0 and n == 100
        ),
    }


def _relation(a, b):
    # Native catastrophe has priority only when both native gates are measured.
    if not a["evidence_qualified"] or not b["evidence_qualified"]:
        return "unknown", "evidence_missing"
    if (
        a["native_catastrophe_applicable"] is True
        and b["native_catastrophe_applicable"] is True
    ):
        if (
            type(a["hard_failure"]) is bool
            and type(b["hard_failure"]) is bool
            and a["hard_failure"] != b["hard_failure"]
        ):
            return (
                "b_dominates" if a["hard_failure"] else "a_dominates"
            ), "native_hard_failure_priority"
    if (
        a["hard_failure"] is None
        or b["hard_failure"] is None
        or a["C"] is None
        or b["C"] is None
        or not a["objective_id"]
        or a["objective_id"] != b["objective_id"]
        or (a["completion_applicable"] and (a["F"] is None or b["F"] is None))
    ):
        return "unknown", "cost_completion_or_objective_unavailable"
    deltas = [b["C"] - a["C"]]
    if a["completion_applicable"]:
        deltas.append(a["F"] - b["F"])
    if all(d == 0 for d in deltas):
        return "tie", "equal_recorded_components"
    if all(d >= 0 for d in deltas):
        return "a_dominates", "cost_fulfillment_pareto"
    if all(d <= 0 for d in deltas):
        return "b_dominates", "cost_fulfillment_pareto"
    return "tradeoff", "cost_fulfillment_tradeoff"


def _pairwise(models, ledgers, indexed, contracts, weights):
    result = []
    for a, b in combinations(models, 2):
        cases, differences, intervals = [], [], []
        for key in contracts:
            left, right = ledgers[a][key], ledgers[b][key]
            relation, reason = _relation(left, right)
            ar, br = indexed.get((a, *key)), indexed.get((b, *key))
            ab, bb = _bounds(ar), _bounds(br)
            interval = [ab[0] - bb[1], ab[1] - bb[0]]
            difference = (
                ar["Q"] - br["Q"]
                if ar and br and ar["Q"] is not None and br["Q"] is not None
                else None
            )
            differences.append(difference)
            intervals.append(interval)
            cases.append(
                {
                    "scenario_signature": key[0],
                    "seed": key[1],
                    "weight": weights[key],
                    "relation": relation,
                    "reason": reason,
                    "Q_difference": difference,
                    "Q_difference_bounds": interval,
                    "Q_plateau_hides_component_difference": difference == 0
                    and relation in {"a_dominates", "b_dominates", "tradeoff"},
                    "a_evidence_ids": left["evidence_ids"],
                    "b_evidence_ids": right["evidence_ids"],
                    "a_objective_id": left["objective_id"],
                    "b_objective_id": right["objective_id"],
                }
            )
        result.append(
            {
                "a": a,
                "b": b,
                "cases": cases,
                "relation_weight_mass": {
                    label: math.fsum(
                        c["weight"] for c in cases if c["relation"] == label
                    )
                    for label in (
                        "a_dominates",
                        "b_dominates",
                        "tradeoff",
                        "tie",
                        "unknown",
                    )
                },
                "Q_difference": math.fsum(
                    weights[k] * d for k, d in zip(contracts, differences)
                )
                if all(d is not None for d in differences)
                else None,
                "Q_difference_bounds": [
                    math.fsum(weights[k] * v[i] for k, v in zip(contracts, intervals))
                    for i in (0, 1)
                ],
            }
        )
    return result


def _ranks(scores):
    ordered = sorted(scores, key=lambda m: (-scores[m], m))
    ranks, previous, rank = {}, None, None
    for position, model in enumerate(ordered, 1):
        if scores[model] != previous:
            rank = position
        ranks[model], previous = rank, scores[model]
    return ranks


def _sensitivity(contracts, weights, indexed, summary):
    cohort = sorted(
        m["model"]
        for m in summary["models"]
        if m["complete"] and m["valid_for_comparison"]
    )
    baseline = _ranks(
        {
            m["model"]: m["primary_score"]
            for m in summary["models"]
            if m["model"] in cohort
        }
    )
    outputs = {}
    for field in ("physical_source_cluster", "domain"):
        groups = {}
        for key, c in contracts.items():
            label = c.get(field)
            if label is None:
                label = "unidentified_case:" + repr(key)
            groups.setdefault(label, set()).add(key)
        variants = []
        for label, excluded in sorted(groups.items()):
            retained = [k for k in contracts if k not in excluded]
            mass = math.fsum(weights[k] for k in retained)
            if not mass:
                continue
            scores = {
                m: math.fsum(weights[k] * indexed[(m, *k)]["Q"] for k in retained)
                / mass
                for m in cohort
            }
            variants.append(
                {
                    "excluded": label,
                    "removed_weight_mass": math.fsum(weights[k] for k in excluded),
                    "retained_weight_mass": mass,
                    "scores": scores,
                    "ranks": _ranks(scores),
                }
            )
        outputs[field] = {
            "variants": variants,
            "rank_ranges": {
                m: [
                    min([baseline[m], *[v["ranks"][m] for v in variants]]),
                    max([baseline[m], *[v["ranks"][m] for v in variants]]),
                ]
                for m in cohort
            },
        }
    return {
        "cohort": cohort,
        "baseline_ranks": baseline,
        "interpretation": "deterministic_fixed_weight_deletion_sensitivity_not_confidence_interval",
        **outputs,
    }


def build_outcome_analysis(report: dict) -> dict:
    """Expose outcome tradeoffs on the declared fixed panel, without rescoring."""
    contracts, weights, models, indexed, summary = _validate(report)
    output, ledgers = [], {}
    summaries = {m["model"]: m for m in summary["models"]}
    for model in models:
        rows = {k: indexed.get((model, *k)) for k in contracts}
        ledger = {k: _case(rows[k], c, weights[k]) for k, c in contracts.items()}
        ledgers[model] = ledger
        groups = {}
        for field in ("domain", "task_family"):
            labels = sorted({(c["domain"], c[field]) for c in contracts.values()})
            groups[field] = [
                {
                    "domain": domain,
                    "label": label,
                    **_slice(
                        [
                            k
                            for k, c in contracts.items()
                            if (c["domain"], c[field]) == (domain, label)
                        ],
                        rows,
                        contracts,
                        weights,
                    ),
                }
                for domain, label in labels
            ]
        output.append(
            {
                "model": model,
                "overall": _slice(list(contracts), rows, contracts, weights),
                **groups,
                "safety_coverage": summaries[model]["safety_coverage"],
                "recorded_hard_failure_rate": summaries[model][
                    "recorded_hard_failure_rate"
                ],
                "bottleneck_counts": dict(
                    Counter(c["bottleneck"] for c in ledger.values())
                ),
                "negative_cost_saturation_cases": sum(
                    c["negative_cost_saturation"] for c in ledger.values()
                ),
                "cases": list(ledger.values()),
            }
        )
    return {
        "schema_version": "operate_outcome_analysis029.v1",
        "evaluation_version": VERSION,
        "base_evaluation_version": "0.28.0",
        "headline_score_changed": False,
        "n_expected": len(contracts),
        "weight_manifest_sha256": report["task_weight_manifest"]["manifest_sha256"],
        "suite_sha256": report["source_contracts"]["suite_sha256"],
        "interpretation": "fixed_panel_descriptive_outcomes_not_isolated_ability_or_statistical_significance",
        "bounds_interpretation": "recorded_precision_and_missing_measurement_bounds_not_sampling_confidence_intervals",
        "models": output,
        "pairwise": _pairwise(models, ledgers, indexed, contracts, weights),
        "rank_sensitivity": _sensitivity(contracts, weights, indexed, summary),
    }
