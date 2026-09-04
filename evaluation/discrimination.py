"""
evaluation.discrimination — score-discrimination analysis (additive, read-only).

This module answers the audit question "why do model scores cluster?" without
touching the scoring contract. It consumes already-scored episode records (the
rows written to ``episodes.jsonl`` / the batch-eval ``results`` list) and reports
how much of the observed score variance separates *models* from how much merely
reflects *scenario/task* difficulty.

Design notes / correctness assumptions:

- The simulator backends are deterministic given ``seed`` and every agent runs
  each cell with the same seed. Therefore, for a single-pass matrix, the
  variance of agent scores *within one cell* is purely model-attributable — it
  is not stochastic replication noise. That makes the within-cell between-model
  variance a clean discrimination signal.
- When multiple passes per (agent, cell) exist (``pass_k > 1`` with a non-zero
  temperature), they are averaged per (agent, cell) before the between-model
  decomposition, and the mean within-(agent, cell) spread is reported separately
  as the stochastic noise floor.
- A dimension whose ``task_dominance_ratio`` is ~1.0 only reflects task
  difficulty (its outcome barely moves with the agent's actions); it is flagged
  ``degenerate`` so it can be down-weighted or marked N/A rather than silently
  flooring every model. A cell whose model score range is ~0 is flagged
  ``uninformative`` (it does not separate models).

Nothing here changes any published score; it is a reporting layer.

Pure stdlib. No external statistics dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Canonical 11-dimension scoring surface (plus the two derived process
# dimensions that the live scorer also emits). Kept here only to give a stable
# ordering / membership check; the analysis works on whatever dimensions the
# episodes actually emit.
DEFAULT_VIEW = "adaptive_applicable"

# Reporting thresholds (flagging only — never used to change a score).
DEFAULT_UNINFORMATIVE_CELL_RANGE = 1.0
DEFAULT_DEGENERATE_DIM_RANGE = 1.0
DEFAULT_DEGENERATE_TASK_DOMINANCE = 0.98


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pvariance(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def _pstdev(xs: list[float]) -> float:
    return _pvariance(xs) ** 0.5


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx, my = _mean(xs), _mean(ys)
    dx = [value - mx for value in xs]
    dy = [value - my for value in ys]
    denominator = (sum(value * value for value in dx) * sum(value * value for value in dy)) ** 0.5
    if denominator == 0:
        return None
    return sum(left * right for left, right in zip(dx, dy, strict=False)) / denominator


def _dimension_correlations(
    episodes: list[dict[str, Any]], dim_names: list[str], cell_key: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(dim_names):
        for right in dim_names[left_index + 1 :]:
            matched = [
                (cell_of(ep, cell_key), dimension_value(ep, left), dimension_value(ep, right))
                for ep in episodes
            ]
            matched = [
                (cell, x, y)
                for cell, x, y in matched
                if cell is not None and x is not None and y is not None
            ]
            xs = [float(x) for _, x, _ in matched]
            ys = [float(y) for _, _, y in matched]
            cell_x: dict[str, list[float]] = {}
            cell_y: dict[str, list[float]] = {}
            for cell, x, y in matched:
                cell_x.setdefault(str(cell), []).append(float(x))
                cell_y.setdefault(str(cell), []).append(float(y))
            residual_x = [float(x) - _mean(cell_x[str(cell)]) for cell, x, _ in matched]
            residual_y = [float(y) - _mean(cell_y[str(cell)]) for cell, _, y in matched]
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "n_paired": len(matched),
                    "raw_pearson": _rounded_or_none(_pearson(xs, ys)),
                    "within_cell_model_pearson": _rounded_or_none(
                        _pearson(residual_x, residual_y)
                    ),
                }
            )
    return rows


def _rounded_or_none(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _dimension_only_agent_ranks(
    episodes: list[dict[str, Any]], excluded: str | None = None
) -> dict[str, int]:
    values: dict[str, list[float]] = {}
    for episode in episodes:
        score = episode.get("score")
        if not isinstance(score, dict):
            continue
        weighted_sum = 0.0
        denominator = 0.0
        for dim in score.get("dimensions") or []:
            if not isinstance(dim, dict) or dim.get("name") == excluded:
                continue
            if not dim.get("applicable", False):
                continue
            value = dim.get("calibrated_score", dim.get("raw_score"))
            if not isinstance(value, (int, float)):
                continue
            weight = float(dim.get("weight", 1.0) or 1.0)
            weighted_sum += float(value) * weight
            denominator += weight
        if denominator:
            values.setdefault(agent_of(episode), []).append(weighted_sum / denominator)
    means = {agent: _mean(agent_values) for agent, agent_values in values.items()}
    ordered = sorted(means, key=lambda agent: (-means[agent], agent))
    return {agent: rank for rank, agent in enumerate(ordered, start=1)}


def agent_of(episode: dict[str, Any]) -> str:
    """Canonical model/agent identity for grouping."""
    model = episode.get("model")
    if isinstance(model, str) and model:
        return model
    name = episode.get("agent_name")
    if isinstance(name, str) and name:
        return name
    legacy_name = episode.get("agent")
    if isinstance(legacy_name, str) and legacy_name:
        return legacy_name
    return "unknown_agent"


def cell_of(episode: dict[str, Any], cell_key: str) -> str | None:
    value = episode.get(cell_key)
    if isinstance(value, str) and value:
        return value
    return None


def view_total(episode: dict[str, Any], view: str) -> float | None:
    """Total score for a named scoring view, falling back to the flat total."""
    score = episode.get("score")
    if not isinstance(score, dict):
        return None
    views = score.get("score_views")
    if isinstance(views, dict):
        v = views.get(view)
        if isinstance(v, dict) and isinstance(v.get("total_score"), (int, float)):
            return float(v["total_score"])
    total = score.get("total_score")
    if isinstance(total, (int, float)):
        return float(total)
    return None


def dimension_value(episode: dict[str, Any], dim_name: str) -> float | None:
    """Calibrated score for a dimension, only when it is applicable."""
    score = episode.get("score")
    if not isinstance(score, dict):
        return None
    for dim in score.get("dimensions") or []:
        if not isinstance(dim, dict) or dim.get("name") != dim_name:
            continue
        if not dim.get("applicable", False):
            return None
        val = dim.get("calibrated_score")
        if val is None:
            val = dim.get("raw_score")
        return float(val) if isinstance(val, (int, float)) else None
    return None


def _emitted_dimension_names(episodes: list[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for ep in episodes:
        score = ep.get("score")
        if not isinstance(score, dict):
            continue
        for dim in score.get("dimensions") or []:
            if isinstance(dim, dict) and isinstance(dim.get("name"), str):
                seen.setdefault(dim["name"], None)
    return list(seen)


def _cell_agent_matrix(
    episodes: list[dict[str, Any]],
    value_fn,
    cell_key: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    """Build cell -> {agent -> mean value} plus cell -> {agent -> pass count}.

    Multiple passes per (agent, cell) are averaged; ``counts`` records how many
    passes contributed so the caller can report a stochastic noise floor.
    """
    accum: dict[str, dict[str, list[float]]] = {}
    for ep in episodes:
        cell = cell_of(ep, cell_key)
        if cell is None:
            continue
        value = value_fn(ep)
        if value is None:
            continue
        agent = agent_of(ep)
        accum.setdefault(cell, {}).setdefault(agent, []).append(value)
    matrix: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for cell, agents in accum.items():
        matrix[cell] = {a: _mean(vs) for a, vs in agents.items()}
        counts[cell] = {a: len(vs) for a, vs in agents.items()}
    return matrix, counts


def _variance_decomposition(
    matrix: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Two-way (task x model) SS decomposition over cells with >=2 agents.

    Returns SS_total = SS_task + SS_within (within-cell between-agent), plus the
    normalized ``discrimination_ratio`` (SS_within / SS_total) and its
    complement ``task_dominance_ratio`` (SS_task / SS_total).
    """
    usable = {c: a for c, a in matrix.items() if len(a) >= 2}
    all_values: list[float] = [v for agents in usable.values() for v in agents.values()]
    n = len(all_values)
    if n < 2:
        return {
            "n_cells": len(usable),
            "n_values": n,
            "ss_total": 0.0,
            "ss_task": 0.0,
            "ss_within_cell_between_model": 0.0,
            "discrimination_ratio": 0.0,
            "task_dominance_ratio": 0.0,
            "mean_within_cell_std": 0.0,
            "mean_within_cell_range": 0.0,
        }
    grand_mean = _mean(all_values)
    ss_total = sum((v - grand_mean) ** 2 for v in all_values)
    ss_task = 0.0
    ss_within = 0.0
    within_stds: list[float] = []
    within_ranges: list[float] = []
    for agents in usable.values():
        vals = list(agents.values())
        cell_mean = _mean(vals)
        ss_task += len(vals) * (cell_mean - grand_mean) ** 2
        ss_within += sum((v - cell_mean) ** 2 for v in vals)
        within_stds.append(_pstdev(vals))
        within_ranges.append(max(vals) - min(vals))
    discrimination_ratio = ss_within / ss_total if ss_total > 0 else 0.0
    task_dominance_ratio = ss_task / ss_total if ss_total > 0 else 0.0
    return {
        "n_cells": len(usable),
        "n_values": n,
        "ss_total": round(ss_total, 4),
        "ss_task": round(ss_task, 4),
        "ss_within_cell_between_model": round(ss_within, 4),
        "discrimination_ratio": round(discrimination_ratio, 4),
        "task_dominance_ratio": round(task_dominance_ratio, 4),
        "mean_within_cell_std": round(_mean(within_stds), 4),
        "mean_within_cell_range": round(_mean(within_ranges), 4),
    }


@dataclass
class DimensionDiscrimination:
    name: str
    applicability_rate: float
    n_applicable_episodes: int
    decomposition: dict[str, Any]
    assessable: bool
    degenerate: bool | None
    not_assessable_reason: str | None = None
    degenerate_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "applicability_rate": round(self.applicability_rate, 4),
            "n_applicable_episodes": self.n_applicable_episodes,
            "assessable": self.assessable,
            "degenerate": self.degenerate,
            "not_assessable_reason": self.not_assessable_reason,
            "degenerate_reasons": list(self.degenerate_reasons),
            **self.decomposition,
        }


@dataclass
class CellDiscrimination:
    cell: str
    n_agents: int
    model_score_mean: float
    model_score_std: float
    model_score_range: float
    uninformative: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "n_agents": self.n_agents,
            "model_score_mean": round(self.model_score_mean, 3),
            "model_score_std": round(self.model_score_std, 3),
            "model_score_range": round(self.model_score_range, 3),
            "uninformative": self.uninformative,
        }


def build_discrimination_report(
    episodes: list[dict[str, Any]],
    *,
    view: str = DEFAULT_VIEW,
    cell_key: str = "scenario_signature",
    uninformative_cell_range: float = DEFAULT_UNINFORMATIVE_CELL_RANGE,
    degenerate_dim_range: float = DEFAULT_DEGENERATE_DIM_RANGE,
    degenerate_task_dominance: float = DEFAULT_DEGENERATE_TASK_DOMINANCE,
    max_flagged_cells: int = 50,
) -> dict[str, Any]:
    """Compute per-dimension and per-cell discrimination stats.

    ``episodes`` is a list of scored episode dicts (as emitted to
    ``episodes.jsonl`` or the batch-eval results list). The report is purely
    descriptive: it never mutates a score and requires no scoring-version bump.
    """
    input_episodes = [ep for ep in episodes if isinstance(ep, dict)]
    episodes = [
        ep
        for ep in input_episodes
        if ep.get("status") == "ok" and view_total(ep, view) is not None
    ]
    agents = sorted({agent_of(ep) for ep in episodes})
    cell_ids = sorted({c for ep in episodes if (c := cell_of(ep, cell_key))})

    # ---- per-cell discrimination on the headline view total ----
    total_matrix, total_counts = _cell_agent_matrix(
        episodes, lambda ep: view_total(ep, view), cell_key
    )
    cells: list[CellDiscrimination] = []
    max_passes = 1
    for cell, agent_scores in total_matrix.items():
        max_passes = max(max_passes, *total_counts[cell].values())
        vals = list(agent_scores.values())
        if len(vals) < 2:
            continue
        rng = max(vals) - min(vals)
        cells.append(
            CellDiscrimination(
                cell=cell,
                n_agents=len(vals),
                model_score_mean=_mean(vals),
                model_score_std=_pstdev(vals),
                model_score_range=rng,
                uninformative=rng <= uninformative_cell_range,
            )
        )
    cells.sort(key=lambda c: (c.model_score_range, c.cell))
    uninformative_cells = [c for c in cells if c.uninformative]
    if len(agents) < 2:
        not_assessable_reason = "fewer_than_two_scored_treatments"
    elif not cells:
        not_assessable_reason = "no_shared_scored_cells_across_treatments"
    else:
        not_assessable_reason = None
    assessable = not_assessable_reason is None

    # ---- per-dimension discrimination ----
    n_eps = len(episodes)
    dim_reports: list[DimensionDiscrimination] = []
    dim_names = _emitted_dimension_names(episodes)
    for dim_name in dim_names:
        n_applicable = sum(
            1 for ep in episodes if dimension_value(ep, dim_name) is not None
        )
        matrix, _ = _cell_agent_matrix(
            episodes, lambda ep, d=dim_name: dimension_value(ep, d), cell_key
        )
        decomp = _variance_decomposition(matrix)
        reasons: list[str] = []
        if n_applicable == 0:
            reasons.append("never_applicable")
        elif decomp["n_values"] >= 2:
            if decomp["mean_within_cell_range"] <= degenerate_dim_range:
                reasons.append("models_indistinguishable_within_cell")
            if decomp["task_dominance_ratio"] >= degenerate_task_dominance:
                reasons.append("variance_dominated_by_task_not_model")
        dim_reports.append(
            DimensionDiscrimination(
                name=dim_name,
                applicability_rate=(n_applicable / n_eps) if n_eps else 0.0,
                n_applicable_episodes=n_applicable,
                decomposition=decomp,
                assessable=assessable,
                degenerate=bool(reasons) if assessable else None,
                not_assessable_reason=not_assessable_reason,
                degenerate_reasons=reasons,
            )
        )
    # Most discriminating first; degenerate dimensions sink to the bottom.
    dim_reports.sort(
        key=lambda d: (
            -d.decomposition["discrimination_ratio"],
            -d.decomposition["mean_within_cell_range"],
            d.name,
        )
    )

    degenerate_dims = [d.name for d in dim_reports if d.degenerate is True]
    correlations = _dimension_correlations(episodes, dim_names, cell_key)
    high_correlations = [
        row
        for row in correlations
        if row["within_cell_model_pearson"] is not None
        and abs(row["within_cell_model_pearson"]) >= 0.9
    ]
    base_ranks = _dimension_only_agent_ranks(episodes)
    rank_sensitivity = []
    for dim_name in dim_names:
        ranks = _dimension_only_agent_ranks(episodes, excluded=dim_name)
        shifts = {
            agent: ranks[agent] - base_rank
            for agent, base_rank in base_ranks.items()
            if agent in ranks
        }
        rank_sensitivity.append(
            {
                "excluded_dimension": dim_name,
                "max_absolute_rank_shift": max(
                    (abs(value) for value in shifts.values()), default=0
                ),
                "rank_shifts": shifts,
            }
        )
    return {
        "schema_version": "0.3",
        "scope": "score_discrimination",
        "view": view,
        "cell_key": cell_key,
        "thresholds": {
            "uninformative_cell_range": uninformative_cell_range,
            "degenerate_dim_range": degenerate_dim_range,
            "degenerate_task_dominance": degenerate_task_dominance,
        },
        "n_input_episodes": len(input_episodes),
        "n_episodes": n_eps,
        "n_excluded_unscored_episodes": len(input_episodes) - n_eps,
        "n_agents": len(agents),
        "agents": agents,
        "n_cells": len(cell_ids),
        "max_passes_per_agent_cell": max_passes,
        "single_pass_matrix": max_passes == 1,
        "summary": {
            "assessment_status": "assessable" if assessable else "not_assessable",
            "not_assessable_reason": not_assessable_reason,
            "n_degenerate_dimensions": len(degenerate_dims),
            "degenerate_dimensions": degenerate_dims,
            "n_uninformative_cells": len(uninformative_cells),
            "n_discriminating_cells": len(cells) - len(uninformative_cells),
            "most_discriminating_dimension": (
                dim_reports[0].name if assessable and dim_reports else None
            ),
        },
        "per_dimension": [d.to_dict() for d in dim_reports],
        "metric_redundancy": {
            "pairwise_correlations": correlations,
            "high_within_cell_correlations": high_correlations,
            "leave_one_dimension_out_rank_sensitivity": rank_sensitivity,
            "rank_sensitivity_scope": "dimension_only_applicable_weighted_mean_diagnostic",
        },
        "per_cell": {
            "n_cells_with_multiple_agents": len(cells),
            "uninformative_cells": [c.to_dict() for c in uninformative_cells[
                :max_flagged_cells
            ]],
        },
        "caveat": (
            f"Discrimination not assessable: {not_assessable_reason}."
            if not assessable
            else (
                "Single-pass matrix: within-cell between-model variance is purely "
                "model-attributable because backends are deterministic given seed."
                if max_passes == 1
                else "Multi-pass matrix: passes averaged per (agent, cell) before "
                "decomposition; stochastic noise is not separated from model effect."
            )
        ),
    }
