"""Primary Protocol-2.1 leaderboard aggregation.

The release score is a four-level macro: samples within effective source,
effective sources within backend, backends within domain, and equal-weight
domains within model. Flat episode means remain diagnostics only.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from copy import deepcopy
from statistics import mean
from typing import Any

from core.source_asset_contract import canonical_physical_source_asset_key
from evaluation.scorer import SCORING_VERSION

PRIMARY_LEADERBOARD_FORMULA_VERSION = (
    "effective_source_backend_domain_macro_v1"
)
PHYSICAL_SOURCE_DIAGNOSTIC_VERSION = (
    "physical_source_backend_domain_macro_v1"
)


class PrimaryLeaderboardContractError(ValueError):
    """A formal aggregation row is missing canonical score or lineage."""


def _required_text(
    row: dict[str, Any],
    *fields: str,
) -> str:
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise PrimaryLeaderboardContractError(
        f"missing required field: {'/'.join(fields)}"
    )


def _source_denominator_key(row: dict[str, Any]) -> str:
    direct = row.get("source_denominator_key")
    ledger = row.get("case_ledger")
    nested = (
        ledger.get("source_denominator_key")
        if isinstance(ledger, dict)
        else None
    )
    value = direct if direct not in (None, "") else nested
    if value in (None, ""):
        raise PrimaryLeaderboardContractError(
            "missing required field: source_denominator_key"
        )
    return str(value)


def _physical_source_identity(row: dict[str, Any]) -> str | None:
    ledger = row.get("case_ledger")
    ledger = ledger if isinstance(ledger, dict) else {}
    physical_lock = ledger.get("physical_source_lock")
    if physical_lock not in (None, "", {}, []):
        return canonical_physical_source_asset_key(physical_lock)
    identity = next(
        (
            value
            for value in (
                row.get("physical_source_key"),
                ledger.get("physical_source_key"),
            )
            if value not in (None, "", {}, [])
        ),
        None,
    )
    if identity is None:
        return None
    try:
        return json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        return None


def _bounded_number(
    row: dict[str, Any],
    field: str,
    *,
    low: float,
    high: float,
) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrimaryLeaderboardContractError(
            f"{field} must be numeric within [{low}, {high}]"
        )
    numeric = float(value)
    if not math.isfinite(numeric) or not low <= numeric <= high:
        raise PrimaryLeaderboardContractError(
            f"{field} must be finite within [{low}, {high}]"
        )
    return numeric


def _macro(
    values: dict[tuple[str, str, str], list[float]],
) -> tuple[
    float,
    dict[str, float],
    dict[str, float],
    dict[str, float],
]:
    source_scores = {
        key: mean(samples)
        for key, samples in values.items()
        if samples
    }
    backend_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (domain, backend, _source), score in source_scores.items():
        backend_groups[(domain, backend)].append(score)
    backend_scores = {
        key: mean(scores)
        for key, scores in backend_groups.items()
        if scores
    }
    domain_groups: dict[str, list[float]] = defaultdict(list)
    for (domain, _backend), score in backend_scores.items():
        domain_groups[domain].append(score)
    domain_scores = {
        domain: mean(scores)
        for domain, scores in domain_groups.items()
        if scores
    }
    primary = mean(domain_scores.values()) if domain_scores else 0.0
    return (
        primary,
        {
            f"{domain}/{backend}/{source}": score
            for (domain, backend, source), score in sorted(source_scores.items())
        },
        {
            f"{domain}/{backend}": score
            for (domain, backend), score in sorted(backend_scores.items())
        },
        dict(sorted(domain_scores.items())),
    )


def aggregate_primary_leaderboard(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate formal sample rows using the sole primary score formula."""

    by_model_score: dict[
        str, dict[tuple[str, str, str], list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    by_model_completion: dict[
        str, dict[tuple[str, str, str], list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    sample_scores: dict[str, list[float]] = defaultdict(list)
    sample_completion: dict[str, list[float]] = defaultdict(list)
    by_model_physical_score: dict[
        str, dict[tuple[str, str, str], list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    by_model_physical_completion: dict[
        str, dict[tuple[str, str, str], list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    missing_physical_identity: dict[str, int] = defaultdict(int)

    for row in rows:
        model = _required_text(row, "model_id", "model")
        domain = _required_text(row, "domain")
        backend = _required_text(row, "backend_kind")
        source = _source_denominator_key(row)
        score = _bounded_number(
            row,
            "discriminative_core_score",
            low=0.0,
            high=100.0,
        )
        completion = _bounded_number(
            row,
            "task_completion_raw",
            low=0.0,
            high=1.0,
        )
        key = (domain, backend, source)
        by_model_score[model][key].append(score)
        by_model_completion[model][key].append(completion)
        physical_source = _physical_source_identity(row)
        if physical_source is None:
            missing_physical_identity[model] += 1
        else:
            physical_key = (domain, backend, physical_source)
            by_model_physical_score[model][physical_key].append(score)
            by_model_physical_completion[model][physical_key].append(
                completion
            )
        sample_scores[model].append(score)
        sample_completion[model].append(completion)

    leaderboard: list[dict[str, Any]] = []
    for model in sorted(by_model_score):
        primary, source_scores, backend_scores, domain_scores = _macro(
            by_model_score[model]
        )
        (
            completion,
            completion_sources,
            completion_backends,
            completion_domains,
        ) = _macro(by_model_completion[model])
        physical_complete = missing_physical_identity[model] == 0
        if physical_complete:
            (
                physical_score,
                physical_source_scores,
                _physical_backend_scores,
                _physical_domain_scores,
            ) = _macro(by_model_physical_score[model])
            (
                physical_completion,
                _physical_completion_sources,
                _physical_completion_backends,
                _physical_completion_domains,
            ) = _macro(by_model_physical_completion[model])
            n_physical_sources: int | None = len(physical_source_scores)
        else:
            physical_score = None
            physical_completion = None
            physical_source_scores = {}
            n_physical_sources = None
        leaderboard.append(
            {
                "model": model,
                "primary_leaderboard_score": primary,
                "primary_task_completion_rate": completion,
                "primary_leaderboard_formula_version": (
                    PRIMARY_LEADERBOARD_FORMULA_VERSION
                ),
                "scoring_version": SCORING_VERSION,
                "effective_source_scores": source_scores,
                "backend_scores": backend_scores,
                "domain_scores": domain_scores,
                "task_completion_effective_source_rates": completion_sources,
                "task_completion_backend_rates": completion_backends,
                "task_completion_domain_rates": completion_domains,
                "n_samples": len(sample_scores[model]),
                "n_effective_sources": len(source_scores),
                "diagnostic_physical_source_macro_version": (
                    PHYSICAL_SOURCE_DIAGNOSTIC_VERSION
                ),
                "diagnostic_physical_source_macro_applicable": (
                    physical_complete
                ),
                "diagnostic_physical_source_macro_score": physical_score,
                "diagnostic_physical_source_task_completion_rate": (
                    physical_completion
                ),
                "diagnostic_physical_source_scores": physical_source_scores,
                "n_physical_sources": n_physical_sources,
                "n_samples_missing_physical_source_identity": (
                    missing_physical_identity[model]
                ),
                "n_backends": len(backend_scores),
                "n_domains": len(domain_scores),
                "diagnostic_sample_weighted_mean": mean(sample_scores[model]),
                "diagnostic_sample_weighted_task_completion_rate": mean(
                    sample_completion[model]
                ),
            }
        )
    leaderboard.sort(
        key=lambda row: (
            -float(row["primary_leaderboard_score"]),
            str(row["model"]),
        )
    )
    return {
        "schema_version": "1.0",
        "scoring_version": SCORING_VERSION,
        "primary_leaderboard_formula_version": (
            PRIMARY_LEADERBOARD_FORMULA_VERSION
        ),
        "leaderboard": leaderboard,
        "n_input_samples": len(rows),
    }


def _percentile_interval(
    values: list[float],
    *,
    alpha: float,
) -> tuple[float, float]:
    ordered = sorted(values)
    if not ordered:
        return 0.0, 0.0
    lo_index = min(int(len(ordered) * alpha / 2.0), len(ordered) - 1)
    hi_index = min(
        max(int(len(ordered) * (1.0 - alpha / 2.0)) - 1, 0),
        len(ordered) - 1,
    )
    return ordered[lo_index], ordered[hi_index]


def _clustered_rows_for_draw(
    by_model_cluster: dict[str, dict[tuple[str, str, str], list[dict[str, Any]]]],
    draw: dict[tuple[str, str], list[tuple[str, str, str]]],
) -> list[dict[str, Any]]:
    sampled: list[dict[str, Any]] = []
    for model in sorted(by_model_cluster):
        for stratum in sorted(draw):
            for draw_index, cluster in enumerate(draw[stratum]):
                for original in by_model_cluster[model][cluster]:
                    row = deepcopy(original)
                    source = _source_denominator_key(row)
                    row["source_denominator_key"] = (
                        f"{source}::physical_cluster_draw:{draw_index}"
                    )
                    sampled.append(row)
    return sampled


def _paired_randomization_p_value(
    *,
    left: str,
    right: str,
    by_model_cluster: dict[
        str, dict[tuple[str, str, str], list[dict[str, Any]]]
    ],
    observed_diff: float,
    n_randomization: int,
    seed: int,
) -> tuple[float, int, str]:
    """Paired cluster label-swap test for the exact primary estimand."""
    clusters = sorted(by_model_cluster[left])
    exact = len(clusters) <= 16
    n_draws = (1 << len(clusters)) if exact else n_randomization
    rng = random.Random(seed)
    extreme = 0
    for draw_index in range(n_draws):
        swapped_rows: list[dict[str, Any]] = []
        for cluster_index, cluster in enumerate(clusters):
            swap = (
                bool((draw_index >> cluster_index) & 1)
                if exact
                else bool(rng.getrandbits(1))
            )
            for source_model, target_model in (
                (left, right if swap else left),
                (right, left if swap else right),
            ):
                for original in by_model_cluster[source_model][cluster]:
                    row = deepcopy(original)
                    if "model_id" in row:
                        row["model_id"] = target_model
                    if "model" in row:
                        row["model"] = target_model
                    swapped_rows.append(row)
        randomized = aggregate_primary_leaderboard(swapped_rows)
        score_by_model = {
            str(row["model"]): float(row["primary_leaderboard_score"])
            for row in randomized["leaderboard"]
        }
        null_diff = score_by_model[left] - score_by_model[right]
        if abs(null_diff) >= abs(observed_diff) - 1e-12:
            extreme += 1
    if exact:
        return extreme / n_draws, n_draws, "exact_cluster_label_swap"
    return (
        (extreme + 1) / (n_draws + 1),
        n_draws,
        "monte_carlo_cluster_label_swap",
    )


def infer_primary_leaderboard(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    min_clusters: int = 10,
    n_randomization: int | None = None,
) -> dict[str, Any]:
    """Clustered inference for the exact formal primary estimand.

    Physical sources are resampled within domain/backend strata.  Every
    resample then recomputes the complete effective-source → backend → domain
    macro, so confidence intervals and pairwise comparisons estimate the same
    quantity used to rank models.  All models must have the same physical
    cluster support; formal coverage is fail-closed rather than intersected.
    """
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1")
    if n_randomization is None:
        n_randomization = n_bootstrap
    if n_randomization < 1:
        raise ValueError("n_randomization must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be within (0, 1)")

    point = aggregate_primary_leaderboard(rows)
    by_model_cluster: dict[
        str, dict[tuple[str, str, str], list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    physical_strata: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        model = _required_text(row, "model_id", "model")
        domain = _required_text(row, "domain")
        backend = _required_text(row, "backend_kind")
        physical = _physical_source_identity(row)
        if physical is None:
            raise PrimaryLeaderboardContractError(
                "formal primary inference requires physical source identity"
            )
        physical_strata[physical].add((domain, backend))
        by_model_cluster[model][(domain, backend, physical)].append(row)

    if any(len(strata) != 1 for strata in physical_strata.values()):
        raise PrimaryLeaderboardContractError(
            "physical source identity may not span domain/backend strata"
        )

    models = sorted(by_model_cluster)
    if not models:
        return {
            **point,
            "primary_inference_version": "physical_cluster_hierarchical_bootstrap_randomization_v1",
            "primary_pairwise": [],
        }
    reference_clusters = set(by_model_cluster[models[0]])
    reference_membership = {
        cluster: sorted(
            (
                _source_denominator_key(row),
                str(row.get("repeat_id", row.get("seed", ""))),
            )
            for row in by_model_cluster[models[0]][cluster]
        )
        for cluster in reference_clusters
    }
    for model in models[1:]:
        if set(by_model_cluster[model]) != reference_clusters:
            raise PrimaryLeaderboardContractError(
                "formal primary inference requires identical physical cluster coverage"
            )
        membership = {
            cluster: sorted(
                (
                    _source_denominator_key(row),
                    str(row.get("repeat_id", row.get("seed", ""))),
                )
                for row in by_model_cluster[model][cluster]
            )
            for cluster in reference_clusters
        }
        if membership != reference_membership:
            raise PrimaryLeaderboardContractError(
                "formal primary inference requires identical effective-source "
                "and repeat membership within physical clusters"
            )

    strata: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for cluster in sorted(reference_clusters):
        strata[(cluster[0], cluster[1])].append(cluster)
    min_stratum_clusters = min(
        (len(clusters) for clusters in strata.values()),
        default=0,
    )
    rng = random.Random(seed)
    bootstrap_scores: dict[str, list[float]] = {model: [] for model in models}
    for _ in range(n_bootstrap):
        draw = {
            stratum: [rng.choice(clusters) for _ in range(len(clusters))]
            for stratum, clusters in strata.items()
        }
        sampled_report = aggregate_primary_leaderboard(
            _clustered_rows_for_draw(by_model_cluster, draw)
        )
        for row in sampled_report["leaderboard"]:
            bootstrap_scores[str(row["model"])].append(
                float(row["primary_leaderboard_score"])
            )

    point_by_model = {
        str(row["model"]): float(row["primary_leaderboard_score"])
        for row in point["leaderboard"]
    }
    for row in point["leaderboard"]:
        model = str(row["model"])
        lo, hi = _percentile_interval(bootstrap_scores[model], alpha=alpha)
        row["primary_cluster_ci"] = {
            "point": point_by_model[model],
            "lo": lo,
            "hi": hi,
            "alpha": alpha,
            "n_bootstrap": n_bootstrap,
            "n_physical_clusters": len(reference_clusters),
            "cluster_unit": "physical_source_lock",
            "estimand": PRIMARY_LEADERBOARD_FORMULA_VERSION,
            "degenerate": lo == hi,
        }

    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(models):
        for right in models[index + 1 :]:
            bootstrap_diffs = [
                a - b
                for a, b in zip(
                    bootstrap_scores[left],
                    bootstrap_scores[right],
                    strict=True,
                )
            ]
            lo, hi = _percentile_interval(bootstrap_diffs, alpha=alpha)
            mean_diff = point_by_model[left] - point_by_model[right]
            pair_seed = seed + sum(
                (index + 1) * ord(character)
                for index, character in enumerate(f"{left}\0{right}")
            )
            raw_p, randomization_draws, randomization_method = (
                _paired_randomization_p_value(
                    left=left,
                    right=right,
                    by_model_cluster=by_model_cluster,
                    observed_diff=mean_diff,
                    n_randomization=n_randomization,
                    seed=pair_seed,
                )
            )
            pairs.append(
                {
                    "pair": [left, right],
                    "mean_diff": mean_diff,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "raw_p_value": raw_p,
                    "raw_p_value_method": randomization_method,
                    "n_randomization": randomization_draws,
                    "n_physical_clusters": len(reference_clusters),
                    "min_clusters_per_stratum": min_stratum_clusters,
                    "low_power": min_stratum_clusters < min_clusters,
                    "cluster_unit": "physical_source_lock",
                    "estimand": PRIMARY_LEADERBOARD_FORMULA_VERSION,
                }
            )

    holm_pairs = sorted(
        pairs,
        key=lambda pair: (pair["raw_p_value"], pair["pair"]),
    )
    running_adjusted = 0.0
    for rank, pair in enumerate(holm_pairs, 1):
        multiplier = len(holm_pairs) - rank + 1
        running_adjusted = max(
            running_adjusted,
            min(1.0, multiplier * float(pair["raw_p_value"])),
        )
        pair["holm_rank"] = rank
        pair["holm_adjusted_p_value"] = running_adjusted
        pair["significant"] = (
            not bool(pair["low_power"]) and running_adjusted <= alpha
        )

    point.update(
        {
            "primary_inference_version": "physical_cluster_hierarchical_bootstrap_randomization_v1",
            "primary_inference_alpha": alpha,
            "primary_inference_n_bootstrap": n_bootstrap,
            "primary_inference_n_physical_clusters": len(reference_clusters),
            "primary_pairwise": pairs,
        }
    )
    return point
