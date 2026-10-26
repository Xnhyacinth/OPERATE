"""
evaluation.statistical — Statistical analysis layer.

Forked from ``dispatch-benchmark/evaluation/statistical_evaluator.py``:
- Bootstrap confidence intervals over per-episode totals
- Cohen's d effect size
- Cronbach-alpha-style internal consistency over per-dim raw scores
- Per-model leaderboard table with CI

Pure stdlib + numpy where helpful. No external statistics dependency.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass
class BootstrapCI:
    point: float
    lo: float
    hi: float
    n_bootstrap: int
    alpha: float
    # True when the input was empty or all-identical (zero variance). In that
    # case ``point``/``lo``/``hi`` collapse to the (single) value or 0 and the
    # CI is not a real interval — a downstream consumer must not read a tight
    # ``hi - lo`` as precision. Surfaced so leaderboard/report code can filter
    # or flag degenerate cells instead of silently reporting a misleading CI.
    degenerate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": round(self.point, 3),
            "lo": round(self.lo, 3),
            "hi": round(self.hi, 3),
            "n_bootstrap": self.n_bootstrap,
            "alpha": self.alpha,
            "degenerate": bool(self.degenerate),
        }


def _make_rng(seed: int) -> random.Random:
    """Deterministic RNG factory. Use this instead of bare ``random.random()``
    so every stochastic call site in the pairwise path is seed-controlled and
    replayable."""
    return random.Random(seed)


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    if not values:
        return BootstrapCI(
            point=0.0, lo=0.0, hi=0.0, n_bootstrap=n_bootstrap, alpha=alpha,
            degenerate=True,
        )
    n = len(values)
    # Zero-variance input: every value identical. The CI collapses to a point
    # but a naive percentile cut would still report ``lo == hi == point`` and
    # look like a high-precision measurement. Flag it so consumers can filter.
    first = values[0]
    if all(v == first for v in values):
        return BootstrapCI(
            point=first, lo=first, hi=first, n_bootstrap=n_bootstrap, alpha=alpha,
            degenerate=True,
        )
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(n_bootstrap * (alpha / 2))]
    hi = means[int(n_bootstrap * (1 - alpha / 2)) - 1]
    return BootstrapCI(
        point=sum(values) / n, lo=lo, hi=hi, n_bootstrap=n_bootstrap, alpha=alpha,
        degenerate=False,
    )


def holm_pairwise_ci(
    per_agent_totals: dict[str, list[float]],
    *,
    seed: int = 0,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    min_observations: int = 10,
) -> list[dict[str, Any]]:
    """Paired comparisons with Holm-Bonferroni multiplicity correction.

    Returns one dict per unordered pair (a, b) with a < b lexicographically:
    ``mean_diff`` is a minus b; ``ci_lo``/``ci_hi`` are the pointwise bootstrap
    CI; ``raw_p_value`` comes from a paired sign-flip randomization test; and
    ``significant`` is True iff the monotone Holm-adjusted p-value is at most
    ``alpha``. ``cohens_dz`` standardizes the paired differences by their
    sample standard deviation; it is ``None`` when that effect size is not
    finite. ``n_observations`` reports the number of aligned cells.

    A pair sharing fewer than ``min_observations`` aligned cells is reported
    with ``low_power: True`` and NO ``significant`` verdict — a thin intersection
    (e.g. two partially-collected models in a refresh) yields an
    over-confident CI, so we refuse to label it significant rather than ship a
    fragile ranking claim. The CI/mean_diff are still returned for inspection.
    """
    agents = sorted(per_agent_totals)
    pairs: list[tuple[str, str]] = [
        (a, b) for i, a in enumerate(agents) for b in agents[i + 1:]
    ]
    raw: list[dict[str, Any]] = []
    for idx, (a, b) in enumerate(pairs):
        ta, tb = per_agent_totals[a], per_agent_totals[b]
        n = min(len(ta), len(tb))
        diffs = [ta[i] - tb[i] for i in range(n)]
        rng = _make_rng(seed + idx * 7919)
        dz = cohens_dz(diffs)
        if n < 2:
            effect_size_status = "insufficient_observations"
        elif dz is None:
            effect_size_status = "undefined_zero_variance"
        else:
            effect_size_status = "estimated"
        entry: dict[str, Any] = {
            "pair": (a, b),
            "mean_diff": (sum(diffs) / n) if n > 0 else 0.0,
            "cohens_dz": round(dz, 4) if dz is not None else None,
            "effect_size_kind": "cohens_dz_paired_difference",
            "effect_size_status": effect_size_status,
            "n_observations": int(n),
            "low_power": n < min_observations,
            "_idx": idx,
        }
        if n == 0:
            entry.update(ci_lo=0.0, ci_hi=0.0, raw_p_value=1.0)
            raw.append(entry)
            continue
        entry["raw_p_value"] = _paired_sign_flip_p_value(
            diffs,
            rng=_make_rng(seed + idx * 7919 + 104729),
            n_resamples=n_bootstrap,
        )
        # Zero-variance diffs: the descriptive CI is the observed point.
        if all(x == diffs[0] for x in diffs):
            entry.update(ci_lo=diffs[0], ci_hi=diffs[0])
            raw.append(entry)
            continue
        boot = sorted(
            sum(rng.choice(diffs) for _ in range(n)) / n
            for _ in range(n_bootstrap)
        )
        lo = boot[int(n_bootstrap * (alpha / 2))]
        hi = boot[int(n_bootstrap * (1 - alpha / 2)) - 1]
        entry.update(ci_lo=lo, ci_hi=hi)
        raw.append(entry)
    # Holm step-down orders eligible hypotheses by raw p-value.  The running
    # maximum makes adjusted p-values monotone in that order.
    eligible = [r for r in raw if not r.get("low_power") and r["n_observations"] > 0]
    order = sorted(eligible, key=lambda r: (r["raw_p_value"], r["pair"]))
    m_eligible = len(order)
    running_adjusted_p = 0.0
    for rank_index, r in enumerate(order):
        multiplier = m_eligible - rank_index
        adjusted_p = min(1.0, multiplier * float(r["raw_p_value"]))
        running_adjusted_p = max(running_adjusted_p, adjusted_p)
        r["holm_rank"] = rank_index + 1
        r["holm_threshold"] = alpha / multiplier
        r["holm_adjusted_p_value"] = running_adjusted_p
        r["significant"] = running_adjusted_p <= alpha
    for r in raw:
        r.pop("_idx", None)
    return raw


def _paired_sign_flip_p_value(
    diffs: list[float],
    *,
    rng: random.Random,
    n_resamples: int,
) -> float:
    """Two-sided Monte Carlo randomization p-value for paired differences."""
    if not diffs:
        return 1.0
    observed = abs(sum(diffs) / len(diffs))
    if observed == 0.0:
        return 1.0
    extreme = 0
    tolerance = 1e-12
    for _ in range(n_resamples):
        randomized = abs(
            sum(value if rng.random() < 0.5 else -value for value in diffs)
            / len(diffs)
        )
        if randomized >= observed - tolerance:
            extreme += 1
    # The plus-one correction prevents a zero Monte Carlo p-value.
    return (extreme + 1) / (n_resamples + 1)


def _diffs_degenerate(entry: dict[str, Any]) -> bool:
    """Kept for callers that introspect pairwise entries. True iff the entry's
    CI collapsed to a point (zero-variance diffs)."""
    return entry.get("ci_lo") == entry.get("ci_hi") and entry["n_observations"] > 0


def cohens_d(a: list[float], b: list[float]) -> float:
    """Effect size between two groups (no pooled SD assumption)."""
    if not a or not b:
        return 0.0
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / max(len(a) - 1, 1)
    vb = sum((x - mb) ** 2 for x in b) / max(len(b) - 1, 1)
    pooled = math.sqrt((va + vb) / 2.0)
    if pooled <= 0:
        return 0.0
    return (ma - mb) / pooled


def cohens_dz(diffs: list[float]) -> float | None:
    """Cohen's dz: mean paired difference divided by its sample SD."""
    if len(diffs) < 2:
        return None
    mean = sum(diffs) / len(diffs)
    variance = sum((value - mean) ** 2 for value in diffs) / (len(diffs) - 1)
    standard_deviation = math.sqrt(variance)
    if standard_deviation <= 0.0:
        return None
    return mean / standard_deviation


def cronbach_alpha(items: list[list[float]]) -> float:
    """``items`` is a list of per-item score lists (each sub-list = one episode).
    Returns Cronbach alpha — internal consistency of the scoring dimensions.
    """
    if not items or len(items) < 2:
        return 0.0
    k = len(items[0])
    if k < 2:
        return 0.0
    # transpose to per-dimension columns
    cols: list[list[float]] = [
        [row[i] for row in items if i < len(row)] for i in range(k)
    ]
    total_per_episode = [sum(row[:k]) for row in items]
    var_total = _variance(total_per_episode)
    if var_total <= 0:
        return 0.0
    sum_var_items = sum(_variance(col) for col in cols)
    return (k / (k - 1)) * (1.0 - sum_var_items / var_total)


def _variance(xs: list[float]) -> float:
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1)


@dataclass
class AgentLeaderboardRow:
    agent_id: str
    n_episodes: int
    mean: float
    ci_lo: float
    ci_hi: float
    n_finished: int
    n_fatal: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "n_episodes": self.n_episodes,
            "mean": round(self.mean, 2),
            "ci_lo": round(self.ci_lo, 2),
            "ci_hi": round(self.ci_hi, 2),
            "n_finished": self.n_finished,
            "n_fatal": self.n_fatal,
            "notes": self.notes,
        }


def build_leaderboard(
    per_agent_totals: dict[str, list[float]],
    finished_flags: dict[str, list[bool]] | None = None,
    fatal_flags: dict[str, list[bool]] | None = None,
    seed: int = 0,
) -> list[AgentLeaderboardRow]:
    rows: list[AgentLeaderboardRow] = []
    for agent_id, totals in per_agent_totals.items():
        ci = bootstrap_ci(totals, seed=seed)
        n_fin = sum(
            1 for x in (finished_flags or {}).get(agent_id, [True] * len(totals)) if x
        )
        n_fat = sum(
            1 for x in (fatal_flags or {}).get(agent_id, [False] * len(totals)) if x
        )
        rows.append(
            AgentLeaderboardRow(
                agent_id=agent_id,
                n_episodes=len(totals),
                mean=ci.point,
                ci_lo=ci.lo,
                ci_hi=ci.hi,
                n_finished=n_fin,
                n_fatal=n_fat,
            )
        )
    # Rank by the conservative bootstrap lower bound first, not just the point
    # mean. This keeps the public ordering CI-aware while preserving the mean as
    # the next tie-breaker for agents with comparable uncertainty.
    rows.sort(key=lambda r: (-r.ci_lo, -r.mean, -r.ci_hi, r.agent_id))
    return rows
