"""Task-sampling uncertainty, tiers and domain diagnostics for 0.31 results.

The bootstrap resamples headline-eligible cases within each domain, keeping
each domain's fixed weight mass. It measures sensitivity to which tasks were
sampled, not repeated-run noise; a separate test-retest summary reports the
only repeated executions available. Diagnostics never feed back into weights.
"""

from __future__ import annotations

from collections import Counter
import math
import random


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def _spearman(x, y):
    return _pearson(_ranks(x), _ranks(y))


def _percentile(sorted_values: list[float], q: float) -> float:
    position = q * (len(sorted_values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (
        position - low
    )


def compact_letter_display(ordered: list[str], resolved: set[frozenset]) -> dict:
    """Insert-absorb letters: models sharing a letter are not resolved apart.

    ``resolved`` holds unordered pairs separated at the chosen probability.
    Letters may overlap; unlike chained tiers this never hides an unresolved
    pair across groups.
    """
    columns = [frozenset(ordered)]
    for pair in sorted(resolved, key=lambda p: sorted(ordered.index(m) for m in p)):
        a, b = tuple(pair)
        split = []
        for column in columns:
            if a in column and b in column:
                split += [column - {a}, column - {b}]
            else:
                split.append(column)
        columns = [c for c in split if c and not any(c < other for other in split)]
        columns = list(dict.fromkeys(columns))
    columns.sort(key=lambda c: min(ordered.index(m) for m in c))
    letters = {m: "" for m in ordered}
    for index, column in enumerate(columns):
        for m in column:
            letters[m] += chr(ord("a") + index) if index < 26 else f"[{index}]"
    return letters


def bootstrap_uncertainty(
    report: dict,
    *,
    replicates: int = 2000,
    seed: int = 0,
    separation: float = 0.9,
    resolution: float = 0.95,
    clusters: dict | None = None,
) -> dict:
    """Domain-stratified bootstrap over complete models only.

    With ``clusters`` (case key -> physical source cluster) whole clusters are
    resampled within each domain, so same-source cases move together; otherwise
    cases are resampled. Each domain keeps its fixed weight mass.
    """
    weights = {
        (w["scenario_signature"], w["seed"]): w["weight"]
        for w in report["headline_weights"]
    }
    complete = [m["model"] for m in report["models"] if m["complete"]]
    if len(complete) < 2:
        return dict(
            models=[],
            heuristic_groups=[],
            letters={},
            pairwise={},
            replicates=0,
            resolution_probability=resolution,
            resolved_pairs=0,
            total_pairs=0,
            interpretation="insufficient_complete_models_for_comparative_uncertainty",
        )
    scores = {
        (c["model"], c["scenario_signature"], c["seed"]): c["S"]
        for c in report["cases"]
        if c["model"] in complete
    }
    domain_of = {
        (c["scenario_signature"], c["seed"]): c["domain"] for c in report["cases"]
    }
    strata = {}
    for key in sorted(weights):
        unit = clusters[key] if clusters is not None else key
        strata.setdefault(domain_of[key], {}).setdefault(unit, []).append(key)
    rng = random.Random(seed)
    draws = {m: [] for m in complete}
    for _ in range(replicates):
        total = dict.fromkeys(complete, 0.0)
        for units in strata.values():
            names = list(units)
            mass = math.fsum(weights[k] for keys in units.values() for k in keys)
            sample = [k for _ in names for k in units[names[rng.randrange(len(names))]]]
            sample_mass = math.fsum(weights[k] for k in sample)
            for k in sample:
                share = weights[k] / sample_mass * mass
                for m in complete:
                    total[m] += share * scores[(m, *k)]
        for m in complete:
            draws[m].append(total[m])
    point = {m["model"]: m["primary_score"] for m in report["models"] if m["complete"]}
    ordered = sorted(complete, key=lambda m: (-point[m], m))
    position = {m: [] for m in complete}
    for b in range(replicates):
        for rank, m in enumerate(sorted(complete, key=lambda m: -draws[m][b]), 1):
            position[m].append(rank)
    pairwise = {
        a: {
            b: sum(x > y for x, y in zip(draws[a], draws[b])) / replicates
            for b in complete
            if b != a
        }
        for a in complete
    }
    groups, current = [], []
    for model in ordered:
        if current and pairwise[current[0]][model] >= separation:
            groups.append(current)
            current = []
        current.append(model)
    groups.append(current)
    group_of = {m: i + 1 for i, group in enumerate(groups) for m in group}
    resolved = {
        frozenset((a, b))
        for i, a in enumerate(ordered)
        for b in ordered[i + 1 :]
        if max(pairwise[a][b], pairwise[b][a]) >= resolution
    }
    letters = compact_letter_display(ordered, resolved)
    rows = []
    for m in ordered:
        values, ranks = sorted(draws[m]), sorted(position[m])
        rows.append(
            dict(
                model=m,
                point=point[m],
                interval95=[_percentile(values, 0.025), _percentile(values, 0.975)],
                bootstrap_sd=math.sqrt(
                    sum((v - sum(values) / len(values)) ** 2 for v in values)
                    / (len(values) - 1)
                ),
                rank_interval95=[
                    ranks[math.floor(0.025 * (len(ranks) - 1))],
                    ranks[math.ceil(0.975 * (len(ranks) - 1))],
                ],
                heuristic_group=group_of[m],
                letters=letters[m],
            )
        )
    units_per_domain = {d: len(u) for d, u in sorted(strata.items())}
    return dict(
        method=(
            "domain_stratified_source_cluster_bootstrap_fixed_domain_mass"
            if clusters is not None
            else "domain_stratified_case_bootstrap_fixed_domain_mass"
        ),
        replicates=replicates,
        seed=seed,
        resampling_units_per_domain=units_per_domain,
        coarse_domains=[d for d, n in units_per_domain.items() if n < 5],
        resolution_probability=resolution,
        resolved_pairs=len(resolved),
        total_pairs=len(ordered) * (len(ordered) - 1) // 2,
        letters_rule="compact_letter_display;shared_letter_means_not_resolved_at_resolution_probability",
        heuristic_group_probability=separation,
        heuristic_group_rule="heuristic_only;new_group_when_top_member_of_current_group_exceeds_model_with_probability_at_least_threshold;cross_group_pairs_may_be_unresolved",
        unresolved_cross_group_pairs=sum(
            1
            for a in ordered
            for b in ordered
            if group_of[a] < group_of[b] and pairwise[a][b] < separation
        ),
        interpretation="task_sampling_uncertainty_not_repeated_run_variance",
        models=rows,
        heuristic_groups=groups,
        letters=letters,
        pairwise=pairwise,
    )


def _complete_scores(report: dict, weights: dict) -> dict:
    complete = [m["model"] for m in report["models"] if m["complete"]]
    scores = {
        (c["model"], c["scenario_signature"], c["seed"]): c["S"]
        for c in report["cases"]
    }
    mass = math.fsum(weights.values())
    return {
        m: math.fsum(w * scores[(m, *k)] for k, w in weights.items()) / mass
        for m in complete
    }


def _order(scores: dict) -> dict:
    return {m: i for i, m in enumerate(sorted(scores, key=lambda m: (-scores[m], m)))}


def influence_diagnostics(report: dict, *, top: int = 10) -> dict:
    """How much single cases and families move the complete-model index."""
    weights = {
        (w["scenario_signature"], w["seed"]): w["weight"]
        for w in report["headline_weights"]
    }
    meta = {(c["scenario_signature"], c["seed"]): c for c in report["cases"]}
    base = _complete_scores(report, weights)
    if len(base) < 3:
        return dict(interpretation="insufficient_complete_models")
    base_rank, models = _order(base), sorted(base)
    removals = []
    for key in weights:
        rest = {k: w for k, w in weights.items() if k != key}
        scores = _complete_scores(report, rest)
        rank = _order(scores)
        removals.append(
            dict(
                scenario_signature=key[0],
                seed=key[1],
                domain=meta[key]["domain"],
                task_family=meta[key]["task_family"],
                weight=weights[key],
                max_rank_shift=max(abs(rank[m] - base_rank[m]) for m in models),
                max_score_change=max(abs(scores[m] - base[m]) for m in models),
                spearman=_spearman(
                    [base[m] for m in models], [scores[m] for m in models]
                ),
            )
        )
    removals.sort(
        key=lambda r: (
            -r["max_rank_shift"],
            -r["max_score_change"],
            r["scenario_signature"],
        )
    )
    totals = [base[m] for m in models]
    mean_total = sum(totals) / len(totals)
    variance = sum((t - mean_total) ** 2 for t in totals)
    case_scores = {
        (c["model"], c["scenario_signature"], c["seed"]): c["S"]
        for c in report["cases"]
    }
    mass = math.fsum(weights.values())
    shares = []
    for key, w in weights.items():
        values = [w / mass * case_scores[(m, *key)] for m in models]
        mean = sum(values) / len(values)
        share = (
            sum((v - mean) * (t - mean_total) for v, t in zip(values, totals))
            / variance
            if variance
            else 0.0
        )
        shares.append((share, key))
    shares.sort(key=lambda s: (-s[0], s[1]))
    family_share = Counter()
    for share, key in shares:
        family_share[meta[key]["task_family"]] += share
    low, high = report["rules"]["score_bounds"]
    clips = {}
    for c in report["cases"]:
        key = (c["scenario_signature"], c["seed"])
        if (
            c["model"] not in base
            or key not in weights
            or c["binding_axis"] == "hard_failure"
        ):
            continue
        bucket = clips.setdefault(c["domain"], dict(rows=0, at_lower=0, at_upper=0))
        bucket["rows"] += 1
        bucket["at_lower"] += c["S_unclipped"] < low
        bucket["at_upper"] += c["S_unclipped"] > high
    return dict(
        leave_one_case_out=dict(
            rank_shift_distribution=dict(
                sorted(Counter(r["max_rank_shift"] for r in removals).items())
            ),
            min_spearman=min(r["spearman"] for r in removals),
            most_influential=removals[:top],
        ),
        case_variance_share=dict(
            top=[
                dict(
                    scenario_signature=k[0],
                    seed=k[1],
                    task_family=meta[k]["task_family"],
                    share=s,
                )
                for s, k in shares[:top]
            ],
            top_share_total=sum(s for s, _ in shares[:top]),
            cases=len(shares),
        ),
        family_variance_share=dict(family_share.most_common()),
        clipped_rows_by_domain=dict(sorted(clips.items())),
        interpretation="diagnostic_only_weights_are_not_tuned_on_these_results",
    )


def domain_variance_share(report: dict) -> dict:
    """Share of between-model index variance attributable to each domain."""
    weights = {
        (w["scenario_signature"], w["seed"]): w["weight"]
        for w in report["headline_weights"]
    }
    complete = [m["model"] for m in report["models"] if m["complete"]]
    contributions = {m: {} for m in complete}
    for c in report["cases"]:
        key = (c["scenario_signature"], c["seed"])
        if c["model"] in contributions and key in weights:
            bucket = contributions[c["model"]]
            bucket[c["domain"]] = bucket.get(c["domain"], 0.0) + weights[key] * c["S"]
    if len(complete) < 3:
        return dict(shares={}, interpretation="insufficient_complete_models")
    domains = sorted({d for row in contributions.values() for d in row})
    totals = [sum(contributions[m].values()) for m in complete]
    mean_total = sum(totals) / len(totals)
    variance = sum((t - mean_total) ** 2 for t in totals)
    shares = {}
    for d in domains:
        values = [contributions[m].get(d, 0.0) for m in complete]
        mean = sum(values) / len(values)
        shares[d] = (
            sum((v - mean) * (t - mean_total) for v, t in zip(values, totals))
            / variance
            if variance
            else None
        )
    return dict(
        shares=shares,
        n_models=len(complete),
        interpretation="covariance_share_of_between_model_variance_reported_only_never_used_for_weighting",
    )


def retest_agreement(
    historical: list[dict], substituted: list[dict], *, domain: str
) -> dict:
    """Compare historical and corrected-retest case scores for one domain."""
    old = {
        (c["model"], c["scenario_signature"], c["seed"]): c["S"]
        for c in historical
        if c["domain"] == domain and c["determined"]
    }
    new = {
        (c["model"], c["scenario_signature"], c["seed"]): c["S"]
        for c in substituted
        if c["domain"] == domain and c["determined"]
    }
    pairs = sorted(old.keys() & new.keys())
    by_model = {}
    for key in pairs:
        by_model.setdefault(key[0], []).append(key)
    models = [
        m
        for m, keys in by_model.items()
        if len(keys) == len([k for k in new if k[0] == m])
    ]
    mean_old = [sum(old[k] for k in by_model[m]) / len(by_model[m]) for m in models]
    mean_new = [sum(new[k] for k in by_model[m]) / len(by_model[m]) for m in models]
    return dict(
        domain=domain,
        paired_cases=len(pairs),
        case_pearson=_pearson([old[k] for k in pairs], [new[k] for k in pairs]),
        model_mean_spearman=_spearman(mean_old, mean_new),
        models=[
            dict(model=m, historical_mean=a, retest_mean=b)
            for m, a, b in zip(models, mean_old, mean_new)
        ],
        interpretation="historical_and_corrected_feedback_executions_differ_in_harness_not_a_pure_noise_estimate",
    )
