"""Fixed-denominator R/A/L aggregation, with task-local safety gates.

Inputs are evidence-linked measurements produced by authenticated readers.
The aggregator never infers opportunities, fills missing scores or changes the
suite to match observed successful runs. Compatibility is owned by the caller.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from math import isfinite, isclose
import random

from evaluation.leaderboard import _macro

VERSION = "0.23.0"
PROTOCOL = "operate_quality.v2"
DIMENSIONS = ("R", "A", "L")
DEFAULT_WEIGHTS = {"R": 0.5, "A": 0.25, "L": 0.25}


def _number(value):
    try:
        return type(value) in (float, int) and isfinite(value)
    except OverflowError:
        return False


def _ids(value):
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(x, str) and x for x in value)
    )


def _weights(value):
    value = DEFAULT_WEIGHTS if value is None else value
    if (
        set(value) != set(DIMENSIONS)
        or any(not _number(v) or v <= 0 for v in value.values())
        or not isclose(sum(value.values()), 1, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError(
            "dimension weights must be positive finite R/A/L weights summing to one"
        )
    return dict(value)


def _measurement(row):
    measurement = row.get("measurement") or {}
    safety = row.get("safety") or {}
    if (
        safety.get("verified") is not True
        or type(safety.get("hard_failure")) is not bool
        or not _ids(safety.get("evidence_ids"))
    ):
        return None, "safety_measurement_unavailable"
    # Safety alone cannot manufacture a measured opportunity or fulfilled task.
    value = measurement.get("score")
    if not _number(value) or not 0 <= value <= 100:
        return None, measurement.get("reason") or "measurement_unavailable"
    if not _ids(measurement.get("evidence_ids")):
        return None, "measurement_evidence_missing"
    return (0.0 if safety["hard_failure"] else float(value)), None


def _dimension(values, expected_count, missing):
    complete = expected_count > 0 and not missing
    score, sources, backends, domains = _macro(values) if values else (None, {}, {}, {})
    return dict(
        score=score if complete else None,
        complete=complete,
        n_expected=expected_count,
        n_measured=sum(len(v) for v in values.values()),
        missing=missing,
        source_scores=sources if complete else None,
        backend_scores=backends if complete else None,
        domain_scores=domains if complete else None,
        reason=None
        if complete
        else (
            "dimension_not_in_declared_suite"
            if not expected_count
            else "incomplete_fixed_dimension"
        ),
    )


def _domain_weights(domain_axes, weights):
    return {
        domain: {
            d: weights[d] / sum(weights[a] for a in axes)
            for d in DIMENSIONS
            if d in axes
        }
        for domain, axes in sorted(domain_axes.items())
    }


def _composite(dimensions, weights, domain_axes, mode):
    if mode == "axis_first":
        return sum(weights[d] * dimensions[d]["score"] for d in DIMENSIONS)
    domain_weights = _domain_weights(domain_axes, weights)
    return sum(
        sum(
            weight * dimensions[d]["domain_scores"][domain]
            for d, weight in selected.items()
        )
        for domain, selected in domain_weights.items()
    ) / len(domain_weights)


def aggregate_operate_rows(
    rows,
    suite_rows,
    models,
    *,
    weights=None,
    bootstrap=0,
    seed=1729,
    aggregation_mode="axis_first",
):
    weights = _weights(weights)
    if aggregation_mode not in {"axis_first", "domain_balanced"}:
        raise ValueError("unknown aggregation mode")
    if type(bootstrap) is not int or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    if not suite_rows or not models or len(set(models)) != len(models):
        raise ValueError("nonempty unique suite and model scopes required")
    specs = {}
    for spec in suite_rows:
        key = (
            spec.get("scenario_signature"),
            spec.get("seed"),
            spec.get("primary_dimension"),
        )
        if (
            not isinstance(key[0], str)
            or not key[0]
            or type(key[1]) is not int
            or key in specs
            or spec.get("primary_dimension") not in DIMENSIONS
        ):
            raise ValueError(
                "each scenario/seed/dimension requires one unique declared measurement"
            )
        if any(
            not isinstance(spec.get(k), str) or not spec[k]
            for k in ("domain", "backend_kind", "source_denominator_key")
        ):
            raise ValueError("aggregation lineage missing")
        specs[key] = spec
    lineages = {}
    for key, spec in specs.items():
        lineage = tuple(
            spec.get(k)
            for k in (
                "domain",
                "backend_kind",
                "source_denominator_key",
                "physical_source_key",
            )
        )
        if key[:2] in lineages and lineages[key[:2]] != lineage:
            raise ValueError("same case has inconsistent dimension lineage")
        lineages[key[:2]] = lineage
    case_dimensions = defaultdict(list)
    for signature, case_seed, dimension in specs:
        case_dimensions[(signature, case_seed)].append(dimension)
    domain_axes = defaultdict(set)
    for spec in specs.values():
        domain_axes[spec["domain"]].add(spec["primary_dimension"])
    domain_weights = _domain_weights(domain_axes, weights)
    indexed = defaultdict(list)
    for row in rows:
        case = (row.get("scenario_signature"), row.get("seed"))
        dimension = row.get("primary_dimension")
        if dimension is None:
            choices = case_dimensions[case]
            if len(choices) != 1:
                raise ValueError("measurement dimension is ambiguous or outside scope")
            dimension = choices[0]
        key = (*case, dimension)
        if key not in specs or row.get("model") not in models:
            raise ValueError("measurement outside declared scope")
        indexed[(row["model"], key)].append(row)
    result = {}
    measured = {}
    for model in models:
        by_dimension = {d: defaultdict(list) for d in DIMENSIONS}
        missing = {d: [] for d in DIMENSIONS}
        safety_by_case = defaultdict(list)
        values = {}
        for key, spec in specs.items():
            dimension = spec["primary_dimension"]
            attempts = indexed[(model, key)]
            value, reason = None, "missing_or_duplicate_attempt"
            if len(attempts) == 1:
                row = attempts[0]
                safety = row.get("safety") or {}
                if (
                    safety.get("verified") is True
                    and type(safety.get("hard_failure")) is bool
                    and _ids(safety.get("evidence_ids"))
                ):
                    safety_by_case[key[:2]].append(safety["hard_failure"])
                value, reason = _measurement(row)
            if value is None:
                missing[dimension].append(
                    dict(scenario_signature=key[0], seed=key[1], reason=reason)
                )
            else:
                lineage = tuple(
                    spec[k]
                    for k in ("domain", "backend_kind", "source_denominator_key")
                )
                by_dimension[dimension][lineage].append(value)
                values[key] = value
        if any(len(set(flags)) > 1 for flags in safety_by_case.values()):
            raise ValueError("same case has conflicting safety measurements")
        safety_known = sum(
            len(flags) == len(case_dimensions[key]) and len(set(flags)) == 1
            for key, flags in safety_by_case.items()
        )
        safety_failed = sum(
            len(flags) == len(case_dimensions[key]) and all(flags)
            for key, flags in safety_by_case.items()
        )
        expected = Counter(s["primary_dimension"] for s in specs.values())
        dimensions = {
            d: _dimension(by_dimension[d], expected[d], missing[d]) for d in DIMENSIONS
        }
        complete = all(d["complete"] for d in dimensions.values())
        result[model] = dict(
            index=_composite(dimensions, weights, domain_axes, aggregation_mode)
            if complete
            else None,
            axis_first_diagnostic_index=sum(
                weights[d] * dimensions[d]["score"] for d in DIMENSIONS
            )
            if complete
            else None,
            complete=complete,
            dimensions=dimensions,
            n_expected=len(specs),
            n_measured=len(values),
            safety=dict(
                n_expected=len(case_dimensions),
                n_measured=safety_known,
                hard_failures=safety_failed,
                hard_failure_rate=safety_failed / len(case_dimensions)
                if safety_known == len(case_dimensions)
                else None,
                scope="declared_task_hard_constraints_not_deployment_certification",
            ),
        )
        measured[model] = values
    complete = all(v["complete"] for v in result.values())
    report = dict(
        schema_version=PROTOCOL,
        evaluation_version=VERSION,
        weights=weights,
        aggregation="fixed_domain_axis_then_source_backend_macro_then_equal_domains"
        if aggregation_mode == "domain_balanced"
        else "fixed_dimension_then_effective_source_backend_domain_macro",
        aggregation_mode=aggregation_mode,
        domain_dimension_weights=domain_weights,
        effective_dimension_weights={
            d: sum(w.get(d, 0) for w in domain_weights.values()) / len(domain_weights)
            for d in DIMENSIONS
        }
        if aggregation_mode == "domain_balanced"
        else dict(weights),
        safety_gate="task_local_before_aggregation",
        models=result,
        complete=complete,
        ranking=sorted(models, key=lambda m: (-result[m]["index"], m))
        if complete
        else [],
        pairwise=[],
        inference_reason="not_requested",
        formal_run_certified=False,
    )
    # Sensitivity changes only declared weights, never input selection or scores.
    report["weight_sensitivity"] = []
    candidates = [dict(R=r, A=(1 - r) / 2, L=(1 - r) / 2) for r in (0.4, 0.5, 0.6)]
    candidates.append(dict.fromkeys(DIMENSIONS, 1 / 3))
    for candidate in candidates:
        scores = {
            m: _composite(
                result[m]["dimensions"], candidate, domain_axes, aggregation_mode
            )
            if result[m]["complete"]
            else None
            for m in models
        }
        report["weight_sensitivity"].append(
            dict(
                weights=candidate,
                scores=scores,
                ranking=sorted(models, key=lambda m: (-scores[m], m))
                if complete
                else [],
            )
        )
    if bootstrap:
        if complete:
            _bootstrap(report, specs, measured, bootstrap, seed)
        else:
            report["inference_reason"] = "incomplete_fixed_scope"
    return report


def _bootstrap(report, specs, measured, count, seed):
    """Paired physical-source bootstrap stratified by construct/domain footprint.

    All cases of a physical source share a draw, including cases assigned to
    different constructs. Footprint strata keep every declared construct and
    backend represented rather than silently dropping empty strata in a draw.
    """
    physical = defaultdict(list)
    source_physical = defaultdict(set)
    for key, spec in specs.items():
        name = spec.get("physical_source_key")
        if not isinstance(name, str) or not name:
            report["inference_reason"] = "physical_source_identity_missing"
            return
        physical[name].append(key)
        source_physical[
            tuple(
                spec[k]
                for k in (
                    "primary_dimension",
                    "domain",
                    "backend_kind",
                    "source_denominator_key",
                )
            )
        ].add(name)
    if any(len(names) != 1 for names in source_physical.values()):
        report["inference_reason"] = "effective_source_spans_physical_clusters"
        return
    strata = defaultdict(list)
    for name, keys in physical.items():
        footprint = tuple(
            sorted(
                {
                    (
                        specs[k]["primary_dimension"],
                        specs[k]["domain"],
                        specs[k]["backend_kind"],
                    )
                    for k in keys
                }
            )
        )
        strata[footprint].append(name)
    rng = random.Random(seed)
    sampled = {m: [] for m in measured}
    for _ in range(count):
        draw = [rng.choice(names) for _, names in sorted(strata.items()) for _ in names]
        for model, scores in measured.items():
            values = {d: defaultdict(list) for d in DIMENSIONS}
            for instance, name in enumerate(draw):
                for key in physical[name]:
                    spec = specs[key]
                    source = (
                        spec["domain"],
                        spec["backend_kind"],
                        f"{instance}:{spec['source_denominator_key']}",
                    )
                    values[spec["primary_dimension"]][source].append(scores[key])
            dimensions = {
                d: {
                    "score": _macro(values[d])[0],
                    "domain_scores": _macro(values[d])[3],
                }
                for d in DIMENSIONS
            }
            sampled[model].append(
                _composite(
                    dimensions,
                    report["weights"],
                    report["domain_dimension_weights"],
                    report["aggregation_mode"],
                )
            )

    def interval(values):
        values = sorted(values)

        def quantile(p):
            index = (len(values) - 1) * p
            low = int(index)
            high = min(low + 1, len(values) - 1)
            return values[low] + (index - low) * (values[high] - values[low])

        return quantile(0.025), quantile(0.975)

    limited = min(map(len, strata.values())) < 10
    for model in measured:
        low, high = interval(sampled[model])
        report["models"][model]["ci"] = dict(
            lo=low,
            hi=high,
            level=0.95,
            n_bootstrap=count,
            n_physical_clusters=len(physical),
            min_stratum_clusters=min(map(len, strata.values())),
            limited_cluster_support=limited,
            method="paired_physical_cluster_footprint_stratified_bootstrap",
        )
    models = sorted(measured)
    for i, left in enumerate(models):
        for right in models[i + 1 :]:
            low, high = interval(
                [a - b for a, b in zip(sampled[left], sampled[right], strict=True)]
            )
            report["pairwise"].append(
                dict(
                    left=left,
                    right=right,
                    mean_difference=report["models"][left]["index"]
                    - report["models"][right]["index"],
                    ci_lo=low,
                    ci_hi=high,
                    distinguishable=(low > 0 or high < 0) and not limited,
                    limited_cluster_support=limited,
                )
            )
    report["inference_reason"] = None
