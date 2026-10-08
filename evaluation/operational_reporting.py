"""Fixed explicit task weights for 0.25 acceptance and separate diagnostics."""

from __future__ import annotations

from collections import Counter
import math

REVISION = "frozen_operational_acceptance.v4"


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def aggregate_acceptance(rows, contracts, weights, *, models):
    suite = {(c["scenario_signature"], c["seed"]): c for c in contracts}
    if not suite or len(suite) != len(contracts) or set(weights) != set(suite):
        raise ValueError("invalid or duplicate weighted suite")
    if any(not _finite(w) or w <= 0 for w in weights.values()) or not math.isclose(
        math.fsum(weights.values()), 1, abs_tol=1e-12, rel_tol=0
    ):
        raise ValueError("invalid frozen task weights")
    if not models or len(set(models)) != len(models):
        raise ValueError("empty or duplicate model population")
    indexed = {}
    for row in rows:
        key = (row["model"], row["scenario_signature"], row["seed"])
        if key in indexed:
            raise ValueError("duplicate weighted observation")
        if key[0] not in models or key[1:] not in suite:
            raise ValueError("foreign weighted observation")
        indexed[key] = row

    def aggregate(scope):
        total_weight = math.fsum(weights[k] for k in scope)
        output = []
        for model in models:
            valid = {}
            strict = {}
            legacy = {}
            hard = {}
            measured = {}
            issues = []
            for key in scope:
                m = (indexed.get((model, *key)) or {}).get("mission") or {}
                measured[key] = m
                bound = bool(m.get("evidence_ids"))
                native_bound = bool(m.get("native_evidence_ids", m.get("evidence_ids")))
                score = m.get("score")
                if bound and _finite(score) and 0 <= score <= 100:
                    valid[key] = score
                else:
                    issues.append(
                        dict(case=list(key), reason=m.get("reason", "missing_case"))
                    )
                if bound and type(m.get("attained")) is bool:
                    strict[key] = 100.0 * m["attained"]
                value = m.get("legacy_utility_score")
                if native_bound and _finite(value) and 0 <= value <= 100:
                    legacy[key] = value
                safety = m.get("safety") or {}
                if (
                    safety.get("verified") is True
                    and type(safety.get("hard_failure")) is bool
                ):
                    hard[key] = 100.0 * safety["hard_failure"]

            def mean(values):
                # Only source-defined slices normalize their frozen mass. Missing
                # model evidence never changes the denominator or weight vector.
                return (
                    math.fsum(weights[k] * values[k] for k in scope) / total_weight
                    if scope and len(values) == len(scope)
                    else None
                )

            interval = None
            if scope and len(valid) == len(scope):
                bounds = [
                    measured[k].get("score_interval", [valid[k], valid[k]])
                    for k in scope
                ]
                if any(
                    not isinstance(b, list)
                    or len(b) != 2
                    or not all(_finite(v) for v in b)
                    or not 0 <= b[0] <= valid[k] <= b[1] <= 100
                    for k, b in zip(scope, bounds, strict=True)
                ):
                    raise ValueError("invalid acceptance precision interval")
                interval = [
                    mean({k: b[i] for k, b in zip(scope, bounds, strict=True)})
                    for i in (0, 1)
                ]
            output.append(
                dict(
                    model=model,
                    primary_score=mean(valid),
                    primary_rank=None,
                    strict_attainment_score=mean(strict),
                    score_interval=interval,
                    complete=bool(scope) and len(valid) == len(scope),
                    n_expected=len(scope),
                    n_scored=len(valid),
                    n_strict_known=len(strict),
                    n_native_measured=sum(
                        _finite(m.get("native_objective"))
                        and bool(m.get("native_evidence_ids", m.get("evidence_ids")))
                        for m in measured.values()
                    ),
                    n_observed=sum((model, *k) in indexed for k in scope),
                    hard_failure_rate=mean(hard),
                    legacy_utility_fixed_weight_score=mean(legacy),
                    legacy_utility_included_in_primary=False,
                    retrospective_only=True,
                    issues=issues,
                )
            )
        ordered = sorted(
            (m for m in output if m["complete"]),
            key=lambda m: (-m["primary_score"], m["model"]),
        )
        previous = rank = None
        for i, m in enumerate(ordered, 1):
            if previous != m["primary_score"]:
                rank = i
            m["primary_rank"] = rank
            previous = m["primary_score"]
        return dict(
            models=output,
            leaderboard=ordered,
            n_expected=len(scope),
            frozen_weight_mass=total_weight,
        )

    result = aggregate(suite)
    result.update(
        evaluation_version="0.25.0",
        protocol_revision=REVISION,
        aggregation="frozen_explicit_task_weights.v1",
        suite_cases=len(suite),
        full_suite_leaderboard=result["leaderboard"],
        formal_run_certified=False,
        leaderboard_eligible=False,
        primary_interpretation="frozen_operational_acceptance_not_behavior_reward",
        slices={
            name: aggregate(
                {
                    k: c
                    for k, c in suite.items()
                    if c.get("strata", {}).get(name, {}).get("status")
                    == "source_declared"
                }
            )
            for name in sorted({n for c in contracts for n in c.get("strata", {})})
        },
        reason_counts=dict(Counter(r["mission"]["reason"] for r in rows)),
    )
    return result


def aggregate_native_outcome(rows, contracts, weights, *, models):
    """Fixed-denominator ranking of settled source-normalized native outcomes."""
    suite = {(c["scenario_signature"], c["seed"]): c for c in contracts}
    if not suite or len(suite) != len(contracts) or set(weights) != set(suite):
        raise ValueError("invalid or duplicate weighted suite")
    if any(not _finite(w) or w <= 0 for w in weights.values()) or not math.isclose(
        math.fsum(weights.values()), 1, abs_tol=1e-12, rel_tol=0
    ):
        raise ValueError("invalid frozen task weights")
    if not models or len(set(models)) != len(models):
        raise ValueError("empty or duplicate model population")
    indexed = {}
    for row in rows:
        key = (row["model"], row["scenario_signature"], row["seed"])
        if key in indexed or key[0] not in models or key[1:] not in suite:
            raise ValueError("duplicate or foreign weighted observation")
        indexed[key] = row

    def aggregate(scope):
        total_weight = math.fsum(weights[k] for k in scope)
        output = []
        for model in models:
            scores, intervals, hard, ratios, issues = {}, {}, {}, {}, []
            native_count = 0
            for key in scope:
                mission = (indexed.get((model, *key)) or {}).get("mission") or {}
                score = mission.get("score")
                evidence = mission.get("native_evidence_ids") or mission.get(
                    "evidence_ids"
                )
                safety = mission.get("safety") or {}
                if _finite(mission.get("native_objective")) and evidence:
                    native_count += 1
                if not (
                    evidence
                    and _finite(score)
                    and 0 <= score <= 100
                    and safety.get("verified") is True
                    and type(safety.get("hard_failure")) is bool
                ):
                    issues.append(
                        dict(
                            case=list(key), reason=mission.get("reason", "missing_case")
                        )
                    )
                    continue
                if safety["hard_failure"] and score != 0:
                    raise ValueError("hard_failure_must_score_zero")
                interval = mission.get("score_interval")
                if not (
                    isinstance(interval, list)
                    and len(interval) == 2
                    and all(_finite(v) for v in interval)
                    and 0 <= interval[0] <= score <= interval[1] <= 100
                ):
                    raise ValueError("invalid native outcome precision interval")
                scores[key] = score
                intervals[key] = interval
                hard[key] = 100.0 * safety["hard_failure"]
                cost, scale = (
                    mission.get("native_objective"),
                    mission.get("source_scale"),
                )
                if _finite(cost) and _finite(scale) and scale > 0:
                    ratio = cost / scale
                    if _finite(ratio):
                        ratios[key] = ratio

            def mean(values):
                return (
                    math.fsum(weights[k] * values[k] for k in scope) / total_weight
                    if scope and len(values) == len(scope)
                    else None
                )

            def weighted_quantile(values, fraction):
                if not scope or len(values) != len(scope):
                    return None
                cumulative = 0.0
                for value, weight in sorted((values[k], weights[k]) for k in scope):
                    cumulative += weight
                    if cumulative >= fraction * total_weight:
                        return value
                return max(values.values())

            def bottom_decile(values):
                if not scope or len(values) != len(scope):
                    return None
                remaining = total_weight * 0.1
                contributions = []
                for value, weight in sorted((values[k], weights[k]) for k in scope):
                    taken = min(remaining, weight)
                    contributions.append(value * taken)
                    remaining -= taken
                    if remaining <= 0:
                        break
                return math.fsum(contributions) / (total_weight * 0.1)

            output.append(
                dict(
                    model=model,
                    primary_score=mean(scores),
                    primary_rank=None,
                    score_interval=[
                        mean({k: intervals[k][i] for k in scope}) for i in (0, 1)
                    ]
                    if len(scores) == len(scope) and scope
                    else None,
                    strict_attainment_score=None,
                    hard_failure_rate=mean(hard),
                    normalized_cost_ratio_quantiles={
                        "p50": weighted_quantile(ratios, 0.5),
                        "p90": weighted_quantile(ratios, 0.9),
                    },
                    bottom_decile_quality=bottom_decile(scores),
                    complete=bool(scope) and len(scores) == len(scope),
                    n_expected=len(scope),
                    n_scored=len(scores),
                    n_native_measured=native_count,
                    n_observed=sum((model, *k) in indexed for k in scope),
                    retrospective_only=True,
                    issues=issues,
                )
            )
        ordered = sorted(
            (m for m in output if m["complete"]),
            key=lambda m: (-m["primary_score"], m["model"]),
        )
        previous = rank = None
        for i, item in enumerate(ordered, 1):
            if item["primary_score"] != previous:
                rank = i
            item["primary_rank"] = rank
            previous = item["primary_score"]
        return dict(
            models=output,
            leaderboard=ordered,
            n_expected=len(scope),
            frozen_weight_mass=total_weight,
        )

    result = aggregate(suite)
    result.update(
        evaluation_version="0.25.0",
        protocol_revision="source_grounded_native_outcome.v3",
        aggregation="frozen_explicit_task_weights.v2",
        suite_cases=len(suite),
        full_suite_leaderboard=result["leaderboard"],
        formal_run_certified=False,
        leaderboard_eligible=False,
        primary_interpretation="source_normalized_settled_native_outcome_quality_not_task_attainment",
        slices={
            name: aggregate(
                {
                    k: c
                    for k, c in suite.items()
                    if c.get("strata", {}).get(name, {}).get("status")
                    == "source_declared"
                }
            )
            for name in sorted({n for c in contracts for n in c.get("strata", {})})
        },
        reason_counts=dict(Counter(r["mission"]["reason"] for r in rows)),
    )
    return result


def result_ledger(report):
    """Export recorded outcomes separately from the rule that scores them."""
    contracts = {
        (c["scenario_signature"], c["seed"]): c
        for c in report["source_contracts"]["contracts"]
    }
    weights = {
        (c["scenario_signature"], c["seed"]): c
        for c in report.get("task_weight_manifest", {}).get("cases", [])
    }
    for row in report["graded_episodes"]:
        key = (row["scenario_signature"], row["seed"])
        contract = contracts[key]
        m = row["mission"]
        weight = weights.get(key) or {}
        yield dict(
            model=row["model"],
            scenario_signature=key[0],
            seed=key[1],
            scenario_sha256=contract["scenario_sha256"],
            domain=contract["domain"],
            backend_kind=contract["backend_kind"],
            physical_source_cluster=weight.get("physical_source_cluster"),
            score_weight=weight.get("score_weight"),
            recorded_native_cost=m.get("recorded_native_objective"),
            settled_native_cost=m.get("native_objective"),
            objective_id=m.get("scored_objective_id"),
            native_component_amounts=m.get("components"),
            native_component_definition=(row.get("native_outcome") or {}).get(
                "definition_source"
            ),
            component_semantics="already_weighted_native_amounts_not_unweighted_physical_quantities",
            settlement_recovery=m.get("terminal_settlement_recovery"),
            obligation_settlement=m.get("obligation_settlement"),
            safety=m.get("safety"),
            service=m.get("service"),
            quality_score=m.get("score"),
            strict_attained=m.get("attained"),
            requirement_results=m.get("requirement_results"),
            measurement_reason=m.get("reason"),
            source_scale=m.get("source_scale"),
            scale_sensitivity=m.get("scale_sensitivity"),
            legacy_source_scale=m.get("source_scale"),
            legacy_utility=m.get(
                "legacy_utility_score",
                m.get("score")
                if report.get("scoring_mode") == "legacy_utility"
                else None,
            ),
            evidence_ids=m.get("evidence_ids"),
            native_evidence_ids=m.get("native_evidence_ids"),
            artifact_binding=row.get("artifact_binding"),
            execution_provenance=row.get("execution_provenance"),
            failure_attribution="observed_outcome_not_proof_of_model_only_causality",
        )
