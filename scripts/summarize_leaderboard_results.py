#!/usr/bin/env python3
"""Summarize clean OPERATE leaderboard results from batch artifacts.

This script is report-only. It reads a release ``core_suite.json`` plus a
``batch_llm_eval.py`` ``summary.csv`` and emits deterministic aggregate tables
for public leaderboard review. It never calls model APIs and never mutates
release artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.leaderboard import aggregate_primary_leaderboard  # noqa: E402
from evaluation.scorer import (  # noqa: E402
    CANONICAL_DIMENSION_WEIGHTS,
    DIFFICULTY_CAL,
)


def _load_episodes_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(raw_lines) - 1:
                break
            raise
    return rows


def _ok_row_cleanliness(row: dict[str, Any]) -> tuple[bool, str | None]:
    llm = (row.get("trajectory_summary") or {}).get("llm") or {}
    if int(llm.get("llm_calls_failed", 0) or 0) > 0:
        return False, "llm_failure_or_fallback_wait"
    if int(llm.get("ticks_wait_fallback", 0) or 0) > 0:
        return False, "llm_failure_or_fallback_wait"
    if float(llm.get("fallback_wait_ratio", 0.0) or 0.0) > 0.0:
        return False, "llm_failure_or_fallback_wait"
    return True, None

DEFAULT_RELEASE = REPO_ROOT / "release" / "operate_v0_61_0"
DEFAULT_SUMMARY_CSV = (
    REPO_ROOT
    / "batch_results"
    / "operate_v0_61_0"
    / "formal"
    / "logical_persistent"
    / "summary.csv"
)
DEFAULT_OUTPUT_JSON = REPO_ROOT / ".hl/artifacts/operate_v061_leaderboard_results.json"
DEFAULT_OUTPUT_MARKDOWN = REPO_ROOT / ".hl/artifacts/operate_v061_leaderboard_results.md"
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 1729
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_FLAKY_SCORE_RANGE_THRESHOLD = 5.0
EXAMPLE_LIMIT = 20


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_core_rows(release_dir: Path) -> list[dict[str, Any]]:
    core = _load_json(release_dir / "core_suite.json")
    rows = core.get("scenarios")
    if not isinstance(rows, list):
        raise ValueError(f"{release_dir / 'core_suite.json'} has no scenarios list")
    return rows


def _load_summary_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _maybe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: object, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _model_label(row: dict[str, Any]) -> str:
    return str(row.get("model") or row.get("agent_name") or "").replace(
        "llm_agent/", ""
    )


def _row_key(row: dict[str, Any]) -> tuple[str, str, int] | None:
    scenario_id = str(row.get("scenario_id") or "")
    model = _model_label(row)
    seed = row.get("seed")
    if not scenario_id or not model or seed is None or seed == "":
        return None
    return scenario_id, model, _as_int(seed)


def _episode_key(row: dict[str, Any]) -> tuple[str, str, int] | None:
    return _row_key(row)


def _pass_id(row: dict[str, Any]) -> str | None:
    for key in ("pass_id", "replicate_id", "sample_id", "_episode_pass_id"):
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _summary_cleanliness(
    row: dict[str, Any],
    *,
    episode_cleanliness: dict[tuple[str, str, int], tuple[bool, str | None]] | None,
) -> tuple[bool, str | None]:
    key = _row_key(row)
    if str(row.get("status") or "ok") != "ok":
        return False, "status_not_ok"
    if _as_int(row.get("llm_calls_failed")) > 0:
        return False, "llm_failure_or_fallback_wait"
    if (
        key is not None
        and episode_cleanliness is not None
        and key in episode_cleanliness
    ):
        return episode_cleanliness[key]
    return True, None


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = max(0.0, min(1.0, fraction)) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _bootstrap_mean_ci(
    values: list[float],
    *,
    samples: int,
    rng: random.Random,
    confidence_level: float,
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1 or samples <= 0:
        value = mean(values)
        return value, value
    n = len(values)
    draws = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _sample in range(samples)
    ]
    alpha = max(0.0, min(1.0, 1.0 - confidence_level))
    return _percentile(draws, alpha / 2.0), _percentile(draws, 1.0 - alpha / 2.0)


def _summarize_rows(
    rows: list[dict[str, Any]], model: str | None = None
) -> dict[str, Any]:
    if not rows:
        return {
            "model": model,
            "clean_cells": 0,
            "mean_total_score": None,
            "median_total_score": None,
            "min_total_score": None,
            "max_total_score": None,
            "std_total_score": None,
            "mean_raw_total": None,
            "mean_prevented_loss": None,
            "mean_foresight_score": None,
            "total_n_tool_calls": 0,
            "mean_n_tool_calls": None,
            "total_n_control_calls": 0,
            "mean_n_control_calls": None,
            "outcome_changed_rate": None,
            "zero_control_call_rate": None,
        }
    scores = [_as_float(row.get("total_score")) for row in rows]
    raw_totals = [_as_float(row.get("raw_total")) for row in rows]
    prevented = [_as_float(row.get("prevented_loss")) for row in rows]
    foresight = [_as_float(row.get("foresight_score")) for row in rows]
    tool_calls = [_as_int(row.get("n_tool_calls")) for row in rows]
    control_calls = [_as_int(row.get("n_control_calls")) for row in rows]
    changed = [_as_bool(row.get("outcome_changed")) for row in rows]
    total_tools = sum(tool_calls)
    total_controls = sum(control_calls)
    out: dict[str, Any] = {
        "clean_cells": len(rows),
        "mean_total_score": _round(mean(scores)),
        "median_total_score": _round(median(scores)),
        "min_total_score": _round(min(scores)),
        "max_total_score": _round(max(scores)),
        "std_total_score": _round(pstdev(scores) if len(scores) > 1 else 0.0),
        "mean_raw_total": _round(mean(raw_totals)),
        "mean_prevented_loss": _round(mean(prevented)),
        "mean_foresight_score": _round(mean(foresight)),
        "total_n_tool_calls": total_tools,
        "mean_n_tool_calls": _round(total_tools / len(rows)),
        "total_n_control_calls": total_controls,
        "mean_n_control_calls": _round(total_controls / len(rows)),
        "outcome_changed_rate": _round(sum(changed) / len(rows)),
        "zero_control_call_rate": _round(
            sum(1 for count in control_calls if count == 0) / len(rows)
        ),
    }
    if model is not None:
        out = {"model": model, **out}
    return out


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            _model_label(row),
            str(row.get("scenario_id") or ""),
            _as_int(row.get("seed")),
        ),
    )


def _group_summary(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    models: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {model: [] for model in models}
    )
    for row in rows:
        group = str(row.get(group_key) or "")
        model = _model_label(row)
        if model in models:
            grouped[group][model].append(row)
    return {
        group: {
            model: _summarize_rows(grouped[group].get(model, [])) for model in models
        }
        for group in sorted(grouped)
    }


def _calibrate_score(raw: float, difficulty_level: str) -> float:
    floor, ceiling, power = DIFFICULTY_CAL.get(difficulty_level, (0.0, 100.0, 1.0))
    if raw <= 0.0:
        return floor
    return floor + (ceiling - floor) * ((raw / 100.0) ** power)


def _dimension_weight(dim: dict[str, Any]) -> float:
    explicit = _maybe_float(dim.get("weight"))
    if explicit is not None:
        return explicit
    return float(CANONICAL_DIMENSION_WEIGHTS.get(str(dim.get("name") or ""), 1.0))


def _normalize_score_views(
    views: dict[str, Any],
) -> dict[str, dict[str, Any]] | None:
    adaptive = views.get("adaptive_applicable")
    fixed = views.get("fixed_all_dimensions")
    if not isinstance(adaptive, dict) or not isinstance(fixed, dict):
        return None
    return {
        "adaptive_applicable": dict(adaptive),
        "fixed_all_dimensions": dict(fixed),
    }


def _reconstruct_score_views(score: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    dims_raw = score.get("dimensions")
    if not isinstance(dims_raw, list) or not dims_raw:
        return None
    dims = [dim for dim in dims_raw if isinstance(dim, dict)]
    if not dims:
        return None

    difficulty_level = str(score.get("difficulty_level") or "")
    applicable_dims = [dim for dim in dims if _as_bool(dim.get("applicable"))]
    adaptive_denominator = sum(_dimension_weight(dim) for dim in applicable_dims)
    adaptive_raw = _maybe_float(score.get("raw_total"))
    if adaptive_raw is None and adaptive_denominator > 0.0:
        adaptive_raw = (
            sum(
                _dimension_weight(dim) * _as_float(dim.get("calibrated_score"))
                for dim in applicable_dims
            )
            / adaptive_denominator
        )
    if adaptive_raw is None:
        adaptive_raw = 0.0
    adaptive_total = _maybe_float(score.get("total_score"))
    if adaptive_total is None:
        adaptive_total = _calibrate_score(adaptive_raw, difficulty_level)

    dims_by_name = {str(dim.get("name") or ""): dim for dim in dims}
    fixed_denominator = sum(CANONICAL_DIMENSION_WEIGHTS.values())
    fixed_numerator = 0.0
    for name, weight in CANONICAL_DIMENSION_WEIGHTS.items():
        dim = dims_by_name.get(name)
        if dim is None or not _as_bool(dim.get("applicable")):
            continue
        fixed_numerator += float(weight) * _as_float(dim.get("calibrated_score"))
    fixed_raw = fixed_numerator / fixed_denominator if fixed_denominator else 0.0
    non_applicable = [
        str(dim.get("name") or "")
        for dim in dims
        if str(dim.get("name") or "") and not _as_bool(dim.get("applicable"))
    ]
    return {
        "adaptive_applicable": {
            "aggregation": "drop_non_applicable",
            "raw_total": adaptive_raw,
            "total_score": adaptive_total,
            "weight_denominator": adaptive_denominator,
            "n_dimensions": len(dims),
            "n_applicable_dimensions": len(applicable_dims),
            "non_applicable_dimensions": non_applicable,
        },
        "fixed_all_dimensions": {
            "aggregation": "fixed_all_dimensions_non_applicable_zero",
            "raw_total": fixed_raw,
            "total_score": _calibrate_score(fixed_raw, difficulty_level),
            "weight_denominator": fixed_denominator,
            "n_dimensions": len(dims),
            "n_applicable_dimensions": len(applicable_dims),
            "non_applicable_dimensions": non_applicable,
        },
    }


def _score_views_from_episode(row: dict[str, Any]) -> dict[str, Any] | None:
    score = row.get("score")
    if not isinstance(score, dict):
        return None
    explicit = score.get("score_views")
    if isinstance(explicit, dict):
        normalized = _normalize_score_views(explicit)
        if normalized is not None:
            return {"source": "explicit", "views": normalized}
    reconstructed = _reconstruct_score_views(score)
    if reconstructed is not None:
        return {"source": "reconstructed", "views": reconstructed}
    return None


def _build_episode_score_views(
    episodes_jsonl: Path | None,
) -> dict[tuple[str, str, int], dict[str, Any]] | None:
    if episodes_jsonl is None:
        return None
    rows = _load_episodes_jsonl(episodes_jsonl)
    score_views: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = _episode_key(row)
        if key is None:
            continue
        item = _score_views_from_episode(row)
        if item is not None:
            score_views[key] = item
    return score_views


def _score_view_metric(
    row: dict[str, Any],
    view_name: str,
    field: str,
) -> float | None:
    views = row.get("_score_views")
    if not isinstance(views, dict):
        return None
    view = views.get(view_name)
    if not isinstance(view, dict):
        return None
    return _maybe_float(view.get(field))


def _score_view_non_applicable(row: dict[str, Any]) -> list[str]:
    views = row.get("_score_views")
    if not isinstance(views, dict):
        return []
    adaptive = views.get("adaptive_applicable")
    if not isinstance(adaptive, dict):
        return []
    raw = adaptive.get("non_applicable_dimensions")
    if not isinstance(raw, list):
        return []
    return sorted(str(item) for item in raw if str(item))


def _summarize_score_view_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    view_rows = [row for row in rows if isinstance(row.get("_score_views"), dict)]
    explicit_rows = [
        row for row in view_rows if row.get("_score_view_source") == "explicit"
    ]
    reconstructed_rows = [
        row for row in view_rows if row.get("_score_view_source") == "reconstructed"
    ]

    def mean_metric(view_name: str, field: str) -> float | None:
        values = [
            value
            for row in view_rows
            if (value := _score_view_metric(row, view_name, field)) is not None
        ]
        return _round(mean(values)) if values else None

    deltas: list[float] = []
    raw_deltas: list[float] = []
    denominator_deltas: list[float] = []
    for row in view_rows:
        adaptive_total = _score_view_metric(row, "adaptive_applicable", "total_score")
        fixed_total = _score_view_metric(row, "fixed_all_dimensions", "total_score")
        adaptive_raw = _score_view_metric(row, "adaptive_applicable", "raw_total")
        fixed_raw = _score_view_metric(row, "fixed_all_dimensions", "raw_total")
        adaptive_denominator = _score_view_metric(
            row, "adaptive_applicable", "weight_denominator"
        )
        fixed_denominator = _score_view_metric(
            row, "fixed_all_dimensions", "weight_denominator"
        )
        if adaptive_total is not None and fixed_total is not None:
            deltas.append(adaptive_total - fixed_total)
        if adaptive_raw is not None and fixed_raw is not None:
            raw_deltas.append(adaptive_raw - fixed_raw)
        if adaptive_denominator is not None and fixed_denominator is not None:
            denominator_deltas.append(fixed_denominator - adaptive_denominator)

    non_applicable_counts: Counter[str] = Counter()
    for row in view_rows:
        non_applicable_counts.update(_score_view_non_applicable(row))

    return {
        "clean_cells": len(rows),
        "score_view_cells": len(view_rows),
        "explicit_score_view_cells": len(explicit_rows),
        "reconstructed_score_view_cells": len(reconstructed_rows),
        "missing_score_view_cells": len(rows) - len(view_rows),
        "mean_adaptive_total_score": mean_metric("adaptive_applicable", "total_score"),
        "mean_fixed_total_score": mean_metric("fixed_all_dimensions", "total_score"),
        "mean_total_score_delta": _round(mean(deltas)) if deltas else None,
        "mean_adaptive_raw_total": mean_metric("adaptive_applicable", "raw_total"),
        "mean_fixed_raw_total": mean_metric("fixed_all_dimensions", "raw_total"),
        "mean_raw_total_delta": _round(mean(raw_deltas)) if raw_deltas else None,
        "mean_adaptive_weight_denominator": mean_metric(
            "adaptive_applicable", "weight_denominator"
        ),
        "mean_fixed_weight_denominator": mean_metric(
            "fixed_all_dimensions", "weight_denominator"
        ),
        "mean_weight_denominator_delta": _round(mean(denominator_deltas))
        if denominator_deltas
        else None,
        "non_applicable_dimension_counts": dict(sorted(non_applicable_counts.items())),
    }


def _group_score_view_summary(
    rows: list[dict[str, Any]],
    *,
    group_key: str,
    models: list[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {model: [] for model in models}
    )
    for row in rows:
        group = str(row.get(group_key) or "")
        model = _model_label(row)
        if model in models:
            grouped[group][model].append(row)
    return {
        group: {
            model: _summarize_score_view_rows(grouped[group].get(model, []))
            for model in models
        }
        for group in sorted(grouped)
    }


def _score_view_by_model(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
) -> dict[str, dict[str, Any]]:
    return {
        model: _summarize_score_view_rows(
            [row for row in rows if _model_label(row) == model]
        )
        for model in models
    }


def _domain_for_row(row: dict[str, Any]) -> str:
    return str(row.get("domain") or "power_grid")


def _build_domain_buckets(core_rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(_domain_for_row(row) for row in core_rows)
    return {"core_scenarios_by_domain": dict(sorted(counts.items()))}


def _descriptor_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower().startswith("yes")


def _capability_bucket_for_descriptor(descriptor: dict[str, Any]) -> str:
    category = str(descriptor.get("category") or "").lower()
    if _descriptor_truthy(descriptor.get("solves_power_flow")):
        return "power_flow_capable"
    if category == "aggregate_uc" or "unit_commitment" in category:
        return "aggregate_uc_non_power_flow"
    if "logistics" in category:
        return "logistics_non_power_flow"
    return "other_non_power_flow"


def _build_capability_buckets(
    manifest: dict[str, Any],
    core_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    descriptors = manifest.get("backend_descriptors") or {}
    by_backend: dict[str, str] = {}
    for backend, descriptor in descriptors.items():
        if isinstance(descriptor, dict):
            by_backend[str(backend)] = _capability_bucket_for_descriptor(descriptor)

    counts: Counter[str] = Counter()
    uncategorized: list[str] = []
    for row in core_rows:
        backend = str(row.get("backend_kind") or "")
        bucket = by_backend.get(backend, "unknown_backend_capability")
        counts[bucket] += 1
        if bucket == "unknown_backend_capability":
            uncategorized.append(backend)

    return {
        "by_backend": dict(sorted(by_backend.items())),
        "core_scenarios_by_bucket": dict(sorted(counts.items())),
        "uncategorized_backends": sorted(set(uncategorized)),
    }


def _capability_macro_leaderboard(
    by_capability_bucket: dict[str, dict[str, dict[str, Any]]],
    *,
    models: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        bucket_scores: dict[str, float] = {}
        bucket_clean_cells: dict[str, int] = {}
        for bucket, by_model in by_capability_bucket.items():
            summary = by_model.get(model) or {}
            score = summary.get("mean_total_score")
            clean_cells = _as_int(summary.get("clean_cells"))
            if score is None or clean_cells <= 0:
                continue
            bucket_scores[bucket] = float(score)
            bucket_clean_cells[bucket] = clean_cells
        macro_score = _round(mean(bucket_scores.values())) if bucket_scores else None
        rows.append(
            {
                "model": model,
                "macro_mean_total_score": macro_score,
                "n_buckets": len(bucket_scores),
                "bucket_scores": dict(sorted(bucket_scores.items())),
                "bucket_clean_cells": dict(sorted(bucket_clean_cells.items())),
            }
        )
    rows.sort(
        key=lambda item: (
            -(item["macro_mean_total_score"] or -1),
            str(item["model"]),
        )
    )
    return rows


def _hierarchical_domain_backend_macro_leaderboard(
    rows: list[dict[str, Any]],
    *,
    models: list[str],
) -> list[dict[str, Any]]:
    """Average backends within domains, then domains within each model.

    This prevents a large family or a single prolific backend from dominating
    the headline while retaining every quality-passing source in the Core.
    The sample-weighted leaderboard remains available as a diagnostic view.
    """
    expected_domains = {
        str(row.get("domain") or "") for row in rows if row.get("domain")
    }
    output: list[dict[str, Any]] = []
    for model in models:
        cells: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in rows:
            if _model_label(row) != model:
                continue
            domain = str(row.get("domain") or "")
            backend = str(row.get("backend_kind") or "")
            if domain and backend:
                cells[(domain, backend)].append(_as_float(row.get("total_score")))
        backend_scores = {
            (domain, backend): mean(scores)
            for (domain, backend), scores in cells.items()
            if scores
        }
        domain_scores: dict[str, float] = {}
        for domain in sorted({key[0] for key in backend_scores}):
            values = [
                score
                for (cell_domain, _backend), score in backend_scores.items()
                if cell_domain == domain
            ]
            if values:
                domain_scores[domain] = mean(values)
        coverage_complete = set(domain_scores) == expected_domains
        macro_score = _round(mean(domain_scores.values())) if domain_scores else None
        output.append(
            {
                "model": model,
                "macro_mean_total_score": macro_score,
                "n_domains": len(domain_scores),
                "n_backends": len(backend_scores),
                "domain_coverage_complete": coverage_complete,
                "headline_eligible": coverage_complete,
                "domain_scores": {
                    domain: _round(score)
                    for domain, score in sorted(domain_scores.items())
                },
                "backend_scores": {
                    f"{domain}/{backend}": _round(score)
                    for (domain, backend), score in sorted(backend_scores.items())
                },
            }
        )
    output.sort(
        key=lambda item: (
            not item["headline_eligible"],
            -(item["macro_mean_total_score"] or -1),
            str(item["model"]),
        )
    )
    return output


def _primary_leaderboard_payload(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Delegate formal aggregation to the sole primary-score implementation."""

    return aggregate_primary_leaderboard(rows)


def _capability_macro_statistical_ranking(
    macro_leaderboard: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    rng = random.Random(bootstrap_seed)
    bucket_score_by_model: dict[str, dict[str, float]] = {
        str(item["model"]): {
            str(bucket): float(score)
            for bucket, score in (item.get("bucket_scores") or {}).items()
        }
        for item in macro_leaderboard
    }

    model_rank_rows: list[dict[str, Any]] = []
    for rank, item in enumerate(macro_leaderboard, start=1):
        model = str(item["model"])
        values = list(bucket_score_by_model.get(model, {}).values())
        low, high = _bootstrap_mean_ci(
            values,
            samples=bootstrap_samples,
            rng=rng,
            confidence_level=confidence_level,
        )
        model_rank_rows.append(
            {
                "rank": rank,
                "model": model,
                "macro_mean_total_score": item["macro_mean_total_score"],
                "macro_mean_ci_low": _round(low),
                "macro_mean_ci_high": _round(high),
                "n_buckets": len(values),
                "significance_group": None,
                "group_label": None,
                "rank_reason_code": None,
            }
        )

    pairwise_by_order: dict[tuple[str, str], dict[str, Any]] = {}
    pairwise: list[dict[str, Any]] = []
    for higher_index, higher in enumerate(model_rank_rows):
        for lower in model_rank_rows[higher_index + 1 :]:
            higher_model = higher["model"]
            lower_model = lower["model"]
            higher_scores = bucket_score_by_model.get(higher_model, {})
            lower_scores = bucket_score_by_model.get(lower_model, {})
            shared_buckets = sorted(set(higher_scores) & set(lower_scores))
            differences = [
                higher_scores[bucket] - lower_scores[bucket]
                for bucket in shared_buckets
            ]
            diff_mean = mean(differences) if differences else None
            low, high = _bootstrap_mean_ci(
                differences,
                samples=bootstrap_samples,
                rng=rng,
                confidence_level=confidence_level,
            )
            if len(shared_buckets) < 2:
                reason_code = "insufficient_buckets"
                significant = False
            elif low is not None and high is not None and low > 0.0:
                reason_code = "significant"
                significant = True
            else:
                reason_code = "overlapping_ci"
                significant = False
            row = {
                "higher_macro_mean_model": higher_model,
                "lower_macro_mean_model": lower_model,
                "macro_mean_difference": _round(diff_mean),
                "diff_ci_low": _round(low),
                "diff_ci_high": _round(high),
                "n_shared_buckets": len(shared_buckets),
                "significant": significant,
                "reason_code": reason_code,
            }
            pairwise.append(row)
            pairwise_by_order[(higher_model, lower_model)] = row

    current_group = 1
    group_anchor = model_rank_rows[0] if model_rank_rows else None
    for index, row in enumerate(model_rank_rows):
        if index == 0:
            row["significance_group"] = current_group
            row["group_label"] = _group_label(current_group)
            row["rank_reason_code"] = "top_macro_mean"
            continue
        assert group_anchor is not None
        comparison = pairwise_by_order.get((group_anchor["model"], row["model"]))
        if comparison and comparison["reason_code"] == "significant":
            current_group += 1
            group_anchor = row
            row["rank_reason_code"] = "significantly_below_previous_macro_group"
        else:
            row["rank_reason_code"] = (
                comparison["reason_code"]
                if comparison
                else "missing_pairwise_comparison"
            )
        row["significance_group"] = current_group
        row["group_label"] = _group_label(current_group)

    return {
        "method": "paired_bootstrap_macro_mean_difference",
        "parameters": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "unit": "capability_bucket",
        },
        "models": model_rank_rows,
        "pairwise": pairwise,
    }


def _group_label(index: int) -> str:
    label = ""
    value = index
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _statistical_ranking(
    per_model_rows: dict[str, list[dict[str, Any]]],
    leaderboard: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    rng = random.Random(bootstrap_seed)
    score_by_model_unit: dict[str, dict[tuple[str, int], float]] = {}
    for model, rows in per_model_rows.items():
        score_by_model_unit[model] = {
            (str(row.get("scenario_id") or ""), _as_int(row.get("seed"))): _as_float(
                row.get("total_score")
            )
            for row in rows
        }

    model_rank_rows: list[dict[str, Any]] = []
    for rank, item in enumerate(leaderboard, start=1):
        model = str(item["model"])
        values = list(score_by_model_unit.get(model, {}).values())
        low, high = _bootstrap_mean_ci(
            values,
            samples=bootstrap_samples,
            rng=rng,
            confidence_level=confidence_level,
        )
        model_rank_rows.append(
            {
                "rank": rank,
                "model": model,
                "mean_total_score": item["mean_total_score"],
                "mean_ci_low": _round(low),
                "mean_ci_high": _round(high),
                "n_clean_units": len(values),
                "significance_group": None,
                "group_label": None,
                "rank_reason_code": None,
            }
        )

    pairwise_by_order: dict[tuple[str, str], dict[str, Any]] = {}
    pairwise: list[dict[str, Any]] = []
    for higher_index, higher in enumerate(model_rank_rows):
        for lower in model_rank_rows[higher_index + 1 :]:
            higher_model = higher["model"]
            lower_model = lower["model"]
            higher_scores = score_by_model_unit.get(higher_model, {})
            lower_scores = score_by_model_unit.get(lower_model, {})
            shared_units = sorted(set(higher_scores) & set(lower_scores))
            differences = [
                higher_scores[unit] - lower_scores[unit] for unit in shared_units
            ]
            diff_mean = mean(differences) if differences else None
            low, high = _bootstrap_mean_ci(
                differences,
                samples=bootstrap_samples,
                rng=rng,
                confidence_level=confidence_level,
            )
            if len(shared_units) < 2:
                reason_code = "insufficient_samples"
                significant = False
            elif low is not None and high is not None and low > 0.0:
                reason_code = "significant"
                significant = True
            else:
                reason_code = "overlapping_ci"
                significant = False
            row = {
                "higher_mean_model": higher_model,
                "lower_mean_model": lower_model,
                "mean_difference": _round(diff_mean),
                "diff_ci_low": _round(low),
                "diff_ci_high": _round(high),
                "n_shared_units": len(shared_units),
                "significant": significant,
                "reason_code": reason_code,
            }
            pairwise.append(row)
            pairwise_by_order[(higher_model, lower_model)] = row

    current_group = 1
    group_anchor = model_rank_rows[0] if model_rank_rows else None
    for index, row in enumerate(model_rank_rows):
        if index == 0:
            row["significance_group"] = current_group
            row["group_label"] = _group_label(current_group)
            row["rank_reason_code"] = "top_mean"
            continue
        assert group_anchor is not None
        comparison = pairwise_by_order.get((group_anchor["model"], row["model"]))
        if comparison and comparison["reason_code"] == "significant":
            current_group += 1
            group_anchor = row
            row["rank_reason_code"] = "significantly_below_previous_group"
        else:
            row["rank_reason_code"] = (
                comparison["reason_code"]
                if comparison
                else "missing_pairwise_comparison"
            )
        row["significance_group"] = current_group
        row["group_label"] = _group_label(current_group)

    return {
        "method": "paired_bootstrap_mean_difference",
        "parameters": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "unit": "scenario_id_seed",
        },
        "models": model_rank_rows,
        "pairwise": pairwise,
    }


def _compact_row(row: dict[str, Any], *, reason: str | None = None) -> dict[str, Any]:
    out = {
        "model": _model_label(row),
        "scenario_id": str(row.get("scenario_id") or ""),
        "seed": _as_int(row.get("seed")),
    }
    if reason is not None:
        out["reason"] = reason
    return out


def _build_episode_cleanliness(
    episodes_jsonl: Path | None,
) -> dict[tuple[str, str, int], tuple[bool, str | None]] | None:
    if episodes_jsonl is None:
        return None
    rows = _load_episodes_jsonl(episodes_jsonl)
    clean: dict[tuple[str, str, int], tuple[bool, str | None]] = {}
    for row in rows:
        key = _episode_key(row)
        if key is None:
            continue
        clean[key] = _ok_row_cleanliness(row)
    return clean


def _build_episode_pass_ids(
    episodes_jsonl: Path | None,
) -> dict[tuple[str, str, int], list[str]]:
    if episodes_jsonl is None:
        return {}
    pass_ids: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for row in _load_episodes_jsonl(episodes_jsonl):
        key = _episode_key(row)
        if key is None:
            continue
        clean, _reason = _ok_row_cleanliness(row)
        if not clean:
            continue
        row_pass_id = _pass_id(row)
        pass_ids[key].append(row_pass_id or "")
    return pass_ids


def _aggregate_base_cell_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) == 1:
        return rows[0]
    out = dict(rows[-1])
    numeric_fields = (
        "total_score",
        "raw_total",
        "prevented_loss",
        "foresight_score",
        "n_tool_calls",
        "n_control_calls",
    )
    for field in numeric_fields:
        values = [
            value for row in rows if (value := _maybe_float(row.get(field))) is not None
        ]
        if values:
            out[field] = mean(values)
    out["outcome_changed"] = any(_as_bool(row.get("outcome_changed")) for row in rows)
    out["_pass_unit_count"] = len(rows)
    out["_pass_unit_ids"] = sorted(_pass_id(row) or "" for row in rows)
    return out


def _build_pass_k_reliability(
    clean_rows_by_key: dict[tuple[str, str, int], list[dict[str, Any]]],
    *,
    models: list[str],
    flaky_score_range_threshold: float = DEFAULT_FLAKY_SCORE_RANGE_THRESHOLD,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[
    dict[str, Any],
    dict[tuple[str, str, int], dict[str, Any]],
    list[tuple[tuple[str, str, int], int]],
]:
    by_model_cells: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    clean_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicate_clean_items: list[tuple[tuple[str, str, int], int]] = []
    missing_pass_id_examples: list[dict[str, Any]] = []
    duplicate_pass_id_examples: list[dict[str, Any]] = []
    clean_replicate_rows_missing_pass_id = 0
    duplicate_pass_id_rows = 0
    observed_pass_units = 0
    observed_replicated_base_cells = 0
    score_by_model_pass_unit: dict[str, dict[tuple[str, int, str], float]] = {
        model: {} for model in models
    }

    for key, rows in sorted(clean_rows_by_key.items()):
        scenario_id, model, seed = key
        by_pass_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        missing_rows: list[dict[str, Any]] = []
        for row in rows:
            row_pass_id = _pass_id(row)
            if row_pass_id is None:
                missing_rows.append(row)
            else:
                by_pass_id[row_pass_id].append(row)

        unique_pass_rows = [
            items[0] for _pass_id_value, items in sorted(by_pass_id.items())
        ]
        duplicate_for_key = sum(max(len(items) - 1, 0) for items in by_pass_id.values())
        if rows and len(rows) > 1 and missing_rows:
            clean_replicate_rows_missing_pass_id += len(missing_rows)
            missing_duplicate_count = (
                len(missing_rows) if unique_pass_rows else max(len(missing_rows) - 1, 0)
            )
            duplicate_for_key += missing_duplicate_count
            if len(missing_pass_id_examples) < EXAMPLE_LIMIT:
                missing_pass_id_examples.append(
                    {
                        "model": model,
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "missing_rows": len(missing_rows),
                    }
                )

        for pass_id_value, items in sorted(by_pass_id.items()):
            if len(items) <= 1:
                continue
            duplicate_pass_id_rows += len(items) - 1
            if len(duplicate_pass_id_examples) < EXAMPLE_LIMIT:
                duplicate_pass_id_examples.append(
                    {
                        "model": model,
                        "scenario_id": scenario_id,
                        "seed": seed,
                        "pass_id": pass_id_value,
                        "duplicate_rows": len(items),
                    }
                )

        if duplicate_for_key:
            duplicate_clean_items.append((key, duplicate_for_key))

        pass_unit_rows = unique_pass_rows
        if not pass_unit_rows:
            pass_unit_rows = [rows[-1]]
        if len(pass_unit_rows) >= 2:
            observed_replicated_base_cells += 1
        observed_pass_units += len(pass_unit_rows)
        for row in pass_unit_rows:
            row_pass_id = _pass_id(row)
            if row_pass_id is not None:
                score_by_model_pass_unit.setdefault(model, {})[
                    (scenario_id, seed, row_pass_id)
                ] = _as_float(row.get("total_score"))
        clean_by_key[key] = _aggregate_base_cell_rows(pass_unit_rows)
        by_model_cells.setdefault(model, []).append(
            {
                "key": key,
                "pass_unit_rows": pass_unit_rows,
            }
        )

    by_model: dict[str, dict[str, Any]] = {}
    for model in models:
        cells = by_model_cells.get(model, [])
        pass_counts = [len(cell["pass_unit_rows"]) for cell in cells]
        pass_scores = [
            _as_float(row.get("total_score"))
            for cell in cells
            for row in cell["pass_unit_rows"]
        ]
        cell_stddevs: list[float] = []
        flaky_cells = 0
        for cell in cells:
            scores = [
                _as_float(row.get("total_score")) for row in cell["pass_unit_rows"]
            ]
            if len(scores) >= 2:
                cell_stddevs.append(pstdev(scores))
                if max(scores) - min(scores) > flaky_score_range_threshold:
                    flaky_cells += 1
        replicated_cells = sum(1 for count in pass_counts if count >= 2)
        by_model[model] = {
            "n_base_cells": len(cells),
            "n_replicated_base_cells": replicated_cells,
            "n_pass_units": sum(pass_counts),
            "mean_passes_per_cell": _round(mean(pass_counts)) if pass_counts else None,
            "min_passes_per_cell": min(pass_counts) if pass_counts else 0,
            "max_passes_per_cell": max(pass_counts) if pass_counts else 0,
            "score_mean_across_pass_units": _round(mean(pass_scores))
            if pass_scores
            else None,
            "score_stddev_across_pass_units": _round(
                pstdev(pass_scores) if len(pass_scores) > 1 else 0.0
            )
            if pass_scores
            else None,
            "mean_cell_score_stddev": _round(mean(cell_stddevs))
            if cell_stddevs
            else 0.0,
            "flaky_cell_rate": _round(flaky_cells / replicated_cells)
            if replicated_cells
            else 0.0,
        }

    rng = random.Random(bootstrap_seed)
    paired_pass_unit_comparisons: list[dict[str, Any]] = []
    for left_index, left_model in enumerate(models):
        for right_model in models[left_index + 1 :]:
            left_scores = score_by_model_pass_unit.get(left_model, {})
            right_scores = score_by_model_pass_unit.get(right_model, {})
            shared_units = sorted(set(left_scores) & set(right_scores))
            differences = [
                left_scores[unit] - right_scores[unit] for unit in shared_units
            ]
            diff_mean = mean(differences) if differences else None
            low, high = _bootstrap_mean_ci(
                differences,
                samples=bootstrap_samples,
                rng=rng,
                confidence_level=confidence_level,
            )
            if len(shared_units) < 2:
                reason_code = "insufficient_shared_pass_units"
                significant = False
            elif low is not None and high is not None and (low > 0.0 or high < 0.0):
                reason_code = "significant"
                significant = True
            else:
                reason_code = "overlapping_ci"
                significant = False
            paired_pass_unit_comparisons.append(
                {
                    "model_a": left_model,
                    "model_b": right_model,
                    "mean_difference_a_minus_b": _round(diff_mean),
                    "diff_ci_low": _round(low),
                    "diff_ci_high": _round(high),
                    "n_shared_pass_units": len(shared_units),
                    "significant": significant,
                    "reason_code": reason_code,
                }
            )

    if clean_replicate_rows_missing_pass_id or duplicate_pass_id_rows:
        status = "uncertified_replicates"
    elif observed_replicated_base_cells:
        status = "certified_replicates"
    else:
        status = "single_pass"

    reliability = {
        "schema_version": "0.1",
        "status": status,
        "unit": "scenario_id_model_seed_pass_id",
        "structural_denominator": "scenario_id_model_seed",
        "accepted_pass_id_fields": ["pass_id", "replicate_id", "sample_id"],
        "flaky_score_range_threshold": flaky_score_range_threshold,
        "paired_pass_unit_confidence": {
            "method": "paired_bootstrap_mean_difference_over_shared_pass_units",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "confidence_level": confidence_level,
            "comparisons": paired_pass_unit_comparisons,
        },
        "observed_base_cells": len(clean_rows_by_key),
        "observed_replicated_base_cells": observed_replicated_base_cells,
        "observed_pass_units": observed_pass_units,
        "clean_replicate_rows_missing_pass_id": clean_replicate_rows_missing_pass_id,
        "missing_pass_id_examples": missing_pass_id_examples,
        "duplicate_pass_id_rows": duplicate_pass_id_rows,
        "duplicate_pass_id_examples": duplicate_pass_id_examples,
        "by_model": by_model,
    }
    return reliability, clean_by_key, duplicate_clean_items


def _models_from_summary(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({_model_label(row) for row in rows if _model_label(row)})


def _seeds_from_summary(rows: list[dict[str, Any]]) -> list[int]:
    return sorted({_as_int(row.get("seed")) for row in rows if row.get("seed") != ""})


def build_leaderboard_results_summary(
    *,
    release_dir: Path = DEFAULT_RELEASE,
    summary_csv: Path = DEFAULT_SUMMARY_CSV,
    episodes_jsonl: Path | None = None,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    manifest = _load_json(release_dir / "manifest.json")
    core = _load_json(release_dir / "core_suite.json")
    core_rows = _load_core_rows(release_dir)
    core_by_id: dict[str, dict[str, Any]] = {}
    for row in core_rows:
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id:
            raise ValueError("core_suite row missing scenario_id")
        if scenario_id in core_by_id:
            raise ValueError(f"duplicate core scenario_id: {scenario_id}")
        core_by_id[scenario_id] = row

    summary_rows = _load_summary_rows(summary_csv)
    domain_buckets = _build_domain_buckets(core_rows)
    capability_buckets = _build_capability_buckets(manifest, core_rows)
    backend_capability_bucket = capability_buckets["by_backend"]
    configured_models = list(models or _models_from_summary(summary_rows))
    configured_seeds = sorted(
        {
            int(seed)
            for seed in (seeds or _seeds_from_summary(summary_rows))
        }
    )
    if not configured_models:
        raise ValueError("no model identity was supplied or observed in summary rows")
    if not configured_seeds:
        raise ValueError("no scenario seed was supplied or observed in summary rows")
    expected_keys = {
        (scenario_id, model, seed)
        for scenario_id in core_by_id
        for model in configured_models
        for seed in configured_seeds
    }

    if episodes_jsonl is None:
        sibling = summary_csv.parent / "episodes.jsonl"
        episodes_jsonl = sibling if sibling.exists() else None
    episode_cleanliness = _build_episode_cleanliness(episodes_jsonl)
    episode_score_views = _build_episode_score_views(episodes_jsonl)
    episode_pass_ids = _build_episode_pass_ids(episodes_jsonl)
    episode_pass_offsets: Counter[tuple[str, str, int]] = Counter()

    clean_rows_by_key: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    unclean_expected: list[tuple[dict[str, Any], str]] = []
    extra_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()

    for index, row in enumerate(summary_rows):
        row["_csv_index"] = str(index)
        status_counts[str(row.get("status") or "")] += 1
        model_counts[_model_label(row)] += 1
        key = _row_key(row)
        if key not in expected_keys:
            extra_rows.append(row)
            continue
        scenario_id, _model, _seed = key
        metadata = core_by_id[scenario_id]
        row["domain"] = _domain_for_row(metadata)
        row["backend_kind"] = str(metadata.get("backend_kind") or "")
        row["capability_bucket"] = backend_capability_bucket.get(
            row["backend_kind"], "unknown_backend_capability"
        )
        row["source_denominator_key"] = str(
            (metadata.get("case_ledger") or {}).get("source_denominator_key") or ""
        )
        row["independence_axis"] = str(
            (metadata.get("case_ledger") or {}).get("independence_axis") or ""
        )
        clean, reason = _summary_cleanliness(
            row,
            episode_cleanliness=episode_cleanliness,
        )
        if not clean:
            unclean_expected.append((row, reason or "unclean"))
            continue
        if _pass_id(row) is None and key in episode_pass_ids:
            offset = episode_pass_offsets[key]
            if offset < len(episode_pass_ids[key]):
                row["_episode_pass_id"] = episode_pass_ids[key][offset]
            episode_pass_offsets[key] += 1
        if episode_score_views is not None and key in episode_score_views:
            score_view_item = episode_score_views[key]
            row["_score_views"] = score_view_item["views"]
            row["_score_view_source"] = score_view_item["source"]
        clean_rows_by_key[key].append(row)

    pass_k_reliability, clean_by_key, duplicate_items = _build_pass_k_reliability(
        clean_rows_by_key,
        models=configured_models,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )

    missing_keys = sorted(expected_keys - set(clean_by_key))
    clean_rows = _sorted_rows(list(clean_by_key.values()))
    per_model_rows: dict[str, list[dict[str, Any]]] = {
        model: [] for model in configured_models
    }
    for row in clean_rows:
        model = _model_label(row)
        if model in per_model_rows:
            per_model_rows[model].append(row)

    leaderboard = [
        _summarize_rows(per_model_rows[model], model=model)
        for model in configured_models
    ]
    leaderboard.sort(
        key=lambda item: (
            -(item["mean_total_score"] or -1),
            str(item["model"]),
        )
    )

    blockers: list[dict[str, Any]] = []
    if missing_keys:
        blockers.append(
            {
                "code": "missing_clean_model_cells",
                "message": "At least one expected core-suite model cell has no clean row.",
                "count": len(missing_keys),
            }
        )
    if unclean_expected:
        blockers.append(
            {
                "code": "unclean_expected_rows",
                "message": "At least one expected row is excluded by the cleanliness predicate.",
                "count": len(unclean_expected),
            }
        )
    if duplicate_items:
        blockers.append(
            {
                "code": "duplicate_clean_cells",
                "message": "At least one expected model cell has duplicate clean rows that are not distinct certified pass units.",
                "count": sum(count for _key, count in duplicate_items),
            }
        )
    if pass_k_reliability["clean_replicate_rows_missing_pass_id"]:
        blockers.append(
            {
                "code": "pass_k_replicates_need_explicit_pass_id",
                "message": "Repeated clean rows for a model cell require pass_id, replicate_id, or sample_id before they can be used as reliability evidence.",
                "count": pass_k_reliability["clean_replicate_rows_missing_pass_id"],
            }
        )
    if pass_k_reliability["duplicate_pass_id_rows"]:
        blockers.append(
            {
                "code": "duplicate_pass_ids_in_replicates",
                "message": "Duplicate pass ids are deduplicated for reliability and cannot inflate pass-unit coverage.",
                "count": pass_k_reliability["duplicate_pass_id_rows"],
            }
        )
    if extra_rows:
        blockers.append(
            {
                "code": "extra_rows_outside_core_scope",
                "message": "The summary CSV contains rows outside the configured core-suite/model/seed scope.",
                "count": len(extra_rows),
            }
        )

    rows_by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        rows_by_scenario[str(row.get("scenario_id") or "")].append(row)

    largest_gaps: list[dict[str, Any]] = []
    for scenario_id, rows in rows_by_scenario.items():
        if len(rows) < 2:
            continue
        ordered = sorted(rows, key=lambda item: _as_float(item.get("total_score")))
        worst = ordered[0]
        best = ordered[-1]
        gap = _as_float(best.get("total_score")) - _as_float(worst.get("total_score"))
        largest_gaps.append(
            {
                "scenario_id": scenario_id,
                "backend_kind": str(best.get("backend_kind") or ""),
                "family": str(best.get("family") or ""),
                "gap": _round(gap),
                "best_model": _model_label(best),
                "best_score": _round(_as_float(best.get("total_score"))),
                "worst_model": _model_label(worst),
                "worst_score": _round(_as_float(worst.get("total_score"))),
            }
        )
    largest_gaps.sort(
        key=lambda item: (-float(item["gap"] or 0.0), item["scenario_id"])
    )

    worst_by_model: dict[str, list[dict[str, Any]]] = {}
    for model in configured_models:
        worst_by_model[model] = [
            {
                "scenario_id": str(row.get("scenario_id") or ""),
                "backend_kind": str(row.get("backend_kind") or ""),
                "family": str(row.get("family") or ""),
                "total_score": _round(_as_float(row.get("total_score"))),
            }
            for row in sorted(
                per_model_rows[model],
                key=lambda item: (
                    _as_float(item.get("total_score")),
                    str(item.get("scenario_id") or ""),
                ),
            )[:EXAMPLE_LIMIT]
        ]

    high_tool_outliers = [
        {
            "model": _model_label(row),
            "scenario_id": str(row.get("scenario_id") or ""),
            "backend_kind": str(row.get("backend_kind") or ""),
            "family": str(row.get("family") or ""),
            "n_tool_calls": _as_int(row.get("n_tool_calls")),
            "total_score": _round(_as_float(row.get("total_score"))),
        }
        for row in sorted(
            clean_rows,
            key=lambda item: (
                -_as_int(item.get("n_tool_calls")),
                _model_label(item),
                str(item.get("scenario_id") or ""),
            ),
        )[:EXAMPLE_LIMIT]
    ]
    zero_control_examples = [
        {
            "model": _model_label(row),
            "scenario_id": str(row.get("scenario_id") or ""),
            "backend_kind": str(row.get("backend_kind") or ""),
            "family": str(row.get("family") or ""),
            "total_score": _round(_as_float(row.get("total_score"))),
        }
        for row in clean_rows
        if _as_int(row.get("n_control_calls")) == 0
    ][:EXAMPLE_LIMIT]

    status = "ready" if not blockers else "not_ready"
    by_capability_bucket = _group_summary(
        clean_rows,
        group_key="capability_bucket",
        models=configured_models,
    )
    capability_macro_leaderboard = _capability_macro_leaderboard(
        by_capability_bucket,
        models=configured_models,
    )
    hierarchical_domain_backend_macro = (
        _hierarchical_domain_backend_macro_leaderboard(
            clean_rows,
            models=configured_models,
        )
    )
    score_view_rows = [
        row for row in clean_rows if isinstance(row.get("_score_views"), dict)
    ]
    score_view_largest_deltas: list[dict[str, Any]] = []
    for row in score_view_rows:
        adaptive_total = _score_view_metric(row, "adaptive_applicable", "total_score")
        fixed_total = _score_view_metric(row, "fixed_all_dimensions", "total_score")
        if adaptive_total is None or fixed_total is None:
            continue
        score_view_largest_deltas.append(
            {
                "model": _model_label(row),
                "scenario_id": str(row.get("scenario_id") or ""),
                "backend_kind": str(row.get("backend_kind") or ""),
                "family": str(row.get("family") or ""),
                "adaptive_total_score": _round(adaptive_total),
                "fixed_total_score": _round(fixed_total),
                "total_score_delta": _round(adaptive_total - fixed_total),
            }
        )
    score_view_largest_deltas.sort(
        key=lambda item: (
            -abs(float(item["total_score_delta"] or 0.0)),
            str(item["model"]),
            str(item["scenario_id"]),
        )
    )
    score_view_summary = {
        "views": ["adaptive_applicable", "fixed_all_dimensions"],
        "availability": {
            "clean_cells": len(clean_rows),
            "with_score_views": len(score_view_rows),
            "explicit_score_views": sum(
                1
                for row in score_view_rows
                if row.get("_score_view_source") == "explicit"
            ),
            "reconstructed_score_views": sum(
                1
                for row in score_view_rows
                if row.get("_score_view_source") == "reconstructed"
            ),
            "missing_score_views": len(clean_rows) - len(score_view_rows),
        },
        "by_model": _score_view_by_model(clean_rows, models=configured_models),
        "by_domain": _group_score_view_summary(
            clean_rows,
            group_key="domain",
            models=configured_models,
        ),
        "by_capability_bucket": _group_score_view_summary(
            clean_rows,
            group_key="capability_bucket",
            models=configured_models,
        ),
        "by_backend": _group_score_view_summary(
            clean_rows,
            group_key="backend_kind",
            models=configured_models,
        ),
    }
    return {
        "schema_version": "operate-leaderboard-results-v1",
        "scope": "operate_core_logical_persistent_primary",
        "release_id": str(manifest.get("release_id")),
        "scoring_version": str(manifest.get("scoring_version")),
        "status": status,
        "release_ready": status == "ready",
        "release_reentry_ready": status == "ready",
        "proceed_commands": [],
        "inputs": {
            "release_dir": str(release_dir),
            "summary_csv": str(summary_csv),
            "summary_csv_sha256": _sha256_file(summary_csv),
            "episodes_jsonl": str(episodes_jsonl) if episodes_jsonl else None,
            "episodes_jsonl_sha256": _sha256_file(episodes_jsonl)
            if episodes_jsonl
            else None,
            "cleanliness_basis": "summary_csv_plus_episodes_jsonl"
            if episodes_jsonl
            else "summary_csv_only",
            "core_suite_sha256": _sha256_file(release_dir / "core_suite.json"),
        },
        "configured_evaluation": {
            "suite": "core_suite",
            "suite_id": core.get("suite_id"),
            "models": configured_models,
            "seeds": configured_seeds,
            "n_core_scenarios": len(core_by_id),
        },
        "coverage": {
            "expected_model_cells": len(expected_keys),
            "summary_rows": len(summary_rows),
            "clean_model_cells": len(clean_by_key),
            "missing_model_cells": len(missing_keys),
            "unclean_expected_rows": len(unclean_expected),
            "extra_rows": len(extra_rows),
            "duplicate_clean_cells": sum(count for _key, count in duplicate_items),
            "status_counts": dict(sorted(status_counts.items())),
            "rows_by_model": dict(sorted(model_counts.items())),
        },
        "domain_buckets": domain_buckets,
        "capability_buckets": capability_buckets,
        "diagnostic_sample_weighted_leaderboard": leaderboard,
        "diagnostic_legacy_hierarchical_macro": {
            "policy": "hierarchical_domain_backend_macro_v1",
            "reason": (
                "legacy total_score aggregation retained for diagnostics only; "
                "formal results use evaluation.leaderboard"
            ),
            "leaderboard": hierarchical_domain_backend_macro,
        },
        "pass_k_reliability": pass_k_reliability,
        "statistical_ranking": _statistical_ranking(
            per_model_rows,
            leaderboard,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
        "capability_macro_leaderboard": capability_macro_leaderboard,
        "capability_macro_statistical_ranking": _capability_macro_statistical_ranking(
            capability_macro_leaderboard,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
        "score_view_summary": score_view_summary,
        "breakdowns": {
            "by_domain": _group_summary(
                clean_rows,
                group_key="domain",
                models=configured_models,
            ),
            "by_capability_bucket": by_capability_bucket,
            "by_backend": _group_summary(
                clean_rows,
                group_key="backend_kind",
                models=configured_models,
            ),
            "by_family": _group_summary(
                clean_rows,
                group_key="family",
                models=configured_models,
            ),
            "by_difficulty_mode": _group_summary(
                clean_rows,
                group_key="difficulty_mode",
                models=configured_models,
            ),
            "by_difficulty_level": _group_summary(
                clean_rows,
                group_key="difficulty_level",
                models=configured_models,
            ),
        },
        "diagnostics": {
            "missing_cell_examples": [
                {"scenario_id": scenario_id, "model": model, "seed": seed}
                for scenario_id, model, seed in missing_keys[:EXAMPLE_LIMIT]
            ],
            "unclean_row_examples": [
                {**_compact_row(row, reason=reason)}
                for row, reason in unclean_expected[:EXAMPLE_LIMIT]
            ],
            "extra_row_examples": [
                _compact_row(row) for row in extra_rows[:EXAMPLE_LIMIT]
            ],
            "duplicate_clean_cell_examples": [
                {
                    "scenario_id": scenario_id,
                    "model": model,
                    "seed": seed,
                    "clean_rows": count + 1,
                }
                for (scenario_id, model, seed), count in duplicate_items[:EXAMPLE_LIMIT]
            ],
            "largest_model_gaps": largest_gaps[:EXAMPLE_LIMIT],
            "worst_scenarios_by_model": worst_by_model,
            "high_tool_call_outliers": high_tool_outliers,
            "zero_control_call_examples": zero_control_examples,
            "score_view_largest_deltas": score_view_largest_deltas[:EXAMPLE_LIMIT],
        },
        "blockers": blockers,
    }


def write_leaderboard_results_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# {report['release_id']} Leaderboard Results",
        "",
        f"- Status: `{report['status']}`",
        f"- Release: `{report['release_id']}`",
        f"- Scoring version: `{report['scoring_version']}`",
        f"- Core scenarios: `{report['configured_evaluation']['n_core_scenarios']}`",
        f"- Models: `{len(report['configured_evaluation']['models'])}`",
        f"- Expected model cells: `{report['coverage']['expected_model_cells']}`",
        f"- Clean model cells: `{report['coverage']['clean_model_cells']}`",
        f"- Missing model cells: `{report['coverage']['missing_model_cells']}`",
        f"- Cleanliness basis: `{report['inputs']['cleanliness_basis']}`",
        f"- Summary CSV SHA-256: `{report['inputs']['summary_csv_sha256']}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        ".venv/bin/python scripts/summarize_leaderboard_results.py "
        f"--release-dir {report['inputs']['release_dir']} "
        f"--summary-csv {report['inputs']['summary_csv']} "
        "--output-json .hl/artifacts/operate_v059_leaderboard_results.json "
        "--output-markdown .hl/artifacts/operate_v059_leaderboard_results.md",
        "```",
        "",
        "## Leaderboard",
        "",
        "| rank | model | clean cells | mean | median | min | max | tools/cell | controls/cell | outcome changed |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    diagnostic_leaderboard = report["diagnostic_sample_weighted_leaderboard"]
    for rank, item in enumerate(diagnostic_leaderboard, start=1):
        lines.append(
            "| "
            f"{rank} | `{item['model']}` | {item['clean_cells']} | "
            f"{item['mean_total_score']} | {item['median_total_score']} | "
            f"{item['min_total_score']} | {item['max_total_score']} | "
            f"{item['mean_n_tool_calls']} | {item['mean_n_control_calls']} | "
            f"{item['outcome_changed_rate']} |"
        )

    lines.extend(["", "## Statistical Ranking", ""])
    lines.extend(
        [
            "| group | rank | model | mean | 95% CI | reason |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    for item in report["statistical_ranking"]["models"]:
        ci = (
            f"[{item['mean_ci_low']}, {item['mean_ci_high']}]"
            if item["mean_ci_low"] is not None
            else "n/a"
        )
        lines.append(
            "| "
            f"{item['group_label']} | {item['rank']} | `{item['model']}` | "
            f"{item['mean_total_score']} | {ci} | "
            f"`{item['rank_reason_code']}` |"
        )

    lines.extend(["", "## Capability Macro Statistical Ranking", ""])
    lines.extend(
        [
            "| group | rank | model | macro mean | 95% CI | buckets | reason |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for item in report["capability_macro_statistical_ranking"]["models"]:
        ci = (
            f"[{item['macro_mean_ci_low']}, {item['macro_mean_ci_high']}]"
            if item["macro_mean_ci_low"] is not None
            else "n/a"
        )
        lines.append(
            "| "
            f"{item['group_label']} | {item['rank']} | `{item['model']}` | "
            f"{item['macro_mean_total_score']} | {ci} | "
            f"{item['n_buckets']} | `{item['rank_reason_code']}` |"
        )

    reliability = report["pass_k_reliability"]
    lines.extend(["", "## Pass^k Reliability", ""])
    lines.extend(
        [
            f"- Status: `{reliability['status']}`",
            f"- Base cells: `{reliability['observed_base_cells']}`",
            f"- Replicated base cells: `{reliability['observed_replicated_base_cells']}`",
            f"- Pass units: `{reliability['observed_pass_units']}`",
            f"- Missing pass ids on repeated rows: `{reliability['clean_replicate_rows_missing_pass_id']}`",
            f"- Duplicate pass-id rows: `{reliability['duplicate_pass_id_rows']}`",
            "",
            "| model | cells | replicated cells | pass units | passes/cell | score mean | score std | flaky rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, item in reliability["by_model"].items():
        lines.append(
            f"| `{model}` | {item['n_base_cells']} | "
            f"{item['n_replicated_base_cells']} | {item['n_pass_units']} | "
            f"{item['mean_passes_per_cell']} | "
            f"{item['score_mean_across_pass_units']} | "
            f"{item['score_stddev_across_pass_units']} | "
            f"{item['flaky_cell_rate']} |"
        )
    comparisons = reliability["paired_pass_unit_confidence"]["comparisons"]
    if comparisons:
        lines.extend(
            [
                "",
                "Paired pass-unit comparisons:",
                "",
                "| model A | model B | mean A-B | CI | shared pass units | reason |",
                "|---|---|---:|---:|---:|---|",
            ]
        )
        for item in comparisons:
            ci = (
                f"[{item['diff_ci_low']}, {item['diff_ci_high']}]"
                if item["diff_ci_low"] is not None
                else "n/a"
            )
            lines.append(
                f"| `{item['model_a']}` | `{item['model_b']}` | "
                f"{item['mean_difference_a_minus_b']} | {ci} | "
                f"{item['n_shared_pass_units']} | `{item['reason_code']}` |"
            )

    lines.extend(["", "## Score View Denominator Check", ""])
    availability = report["score_view_summary"]["availability"]
    lines.extend(
        [
            f"- Score-view cells: `{availability['with_score_views']}` / `{availability['clean_cells']}`",
            f"- Explicit score views: `{availability['explicit_score_views']}`",
            f"- Reconstructed score views: `{availability['reconstructed_score_views']}`",
            f"- Missing score views: `{availability['missing_score_views']}`",
            "",
            "| model | cells | adaptive mean | fixed mean | adaptive - fixed | adaptive denom | fixed denom | denom gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, item in report["score_view_summary"]["by_model"].items():
        lines.append(
            f"| `{model}` | {item['score_view_cells']} | "
            f"{item['mean_adaptive_total_score']} | "
            f"{item['mean_fixed_total_score']} | "
            f"{item['mean_total_score_delta']} | "
            f"{item['mean_adaptive_weight_denominator']} | "
            f"{item['mean_fixed_weight_denominator']} | "
            f"{item['mean_weight_denominator_delta']} |"
        )
    lines.extend(["", "Largest score-view deltas:"])
    for item in report["diagnostics"]["score_view_largest_deltas"][:10]:
        lines.append(
            f"- `{item['model']}` `{item['scenario_id']}` "
            f"adaptive=`{item['adaptive_total_score']}` "
            f"fixed=`{item['fixed_total_score']}` "
            f"delta=`{item['total_score_delta']}`"
        )

    lines.extend(["", "## Capability Macro Leaderboard", ""])
    lines.extend(
        [
            "| rank | model | macro mean | buckets | headline mean |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    headline_by_model = {
        item["model"]: item for item in diagnostic_leaderboard
    }
    for rank, item in enumerate(report["capability_macro_leaderboard"], start=1):
        headline = headline_by_model.get(item["model"], {})
        lines.append(
            "| "
            f"{rank} | `{item['model']}` | {item['macro_mean_total_score']} | "
            f"{item['n_buckets']} | {headline.get('mean_total_score')} |"
        )

    lines.extend(["", "## Capability Bucket Breakdown", ""])
    lines.extend(
        [
            "| bucket | core scenarios | model | clean cells | mean | tools/cell | controls/cell |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    core_counts_by_bucket = report["capability_buckets"]["core_scenarios_by_bucket"]
    for bucket, by_model in report["breakdowns"]["by_capability_bucket"].items():
        core_count = core_counts_by_bucket.get(bucket, 0)
        for model, item in by_model.items():
            lines.append(
                f"| `{bucket}` | {core_count} | `{model}` | "
                f"{item['clean_cells']} | {item['mean_total_score']} | "
                f"{item['mean_n_tool_calls']} | {item['mean_n_control_calls']} |"
            )

    lines.extend(["", "## Domain Breakdown", ""])
    lines.extend(
        [
            "| domain | core scenarios | model | clean cells | mean | tools/cell | controls/cell |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    core_counts_by_domain = report["domain_buckets"]["core_scenarios_by_domain"]
    for domain, by_model in report["breakdowns"]["by_domain"].items():
        core_count = core_counts_by_domain.get(domain, 0)
        for model, item in by_model.items():
            lines.append(
                f"| `{domain}` | {core_count} | `{model}` | "
                f"{item['clean_cells']} | {item['mean_total_score']} | "
                f"{item['mean_n_tool_calls']} | {item['mean_n_control_calls']} |"
            )

    lines.extend(["", "## Backend Breakdown", ""])
    for backend, by_model in report["breakdowns"]["by_backend"].items():
        lines.extend(
            [
                f"### `{backend}`",
                "",
                "| model | clean cells | mean | tools/cell | controls/cell |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for model, item in by_model.items():
            lines.append(
                f"| `{model}` | {item['clean_cells']} | "
                f"{item['mean_total_score']} | {item['mean_n_tool_calls']} | "
                f"{item['mean_n_control_calls']} |"
            )
        lines.append("")

    lines.extend(["", "## Diagnostics", ""])
    if report["blockers"]:
        lines.append("Blockers:")
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker['code']}`: {blocker['message']}")
    else:
        lines.append("- No result-integrity blockers.")
    lines.extend(["", "Largest model gaps:"])
    for item in report["diagnostics"]["largest_model_gaps"][:10]:
        lines.append(
            f"- `{item['scenario_id']}` gap `{item['gap']}`: "
            f"`{item['best_model']}` {item['best_score']} vs "
            f"`{item['worst_model']}` {item['worst_score']}"
        )
    lines.extend(["", "High tool-call outliers:"])
    for item in report["diagnostics"]["high_tool_call_outliers"][:10]:
        lines.append(
            f"- `{item['model']}` `{item['scenario_id']}` "
            f"tools=`{item['n_tool_calls']}` score=`{item['total_score']}`"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_models(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
        help="One immutable formal shard summary.csv; model and seed axes are inferred unless supplied.",
    )
    parser.add_argument(
        "--episodes-jsonl",
        type=Path,
        help="Optional episodes.jsonl for fallback-wait cleanliness checks. Defaults to a sibling episodes.jsonl when present.",
    )
    parser.add_argument("--models", help="Comma-separated model ids.")
    parser.add_argument("--seeds", help="Comma-separated seeds.")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Paired bootstrap resamples for statistical ranking.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Deterministic seed for statistical-ranking bootstrap.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help="Confidence level for bootstrap intervals.",
    )
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--no-markdown", action="store_true")
    args = parser.parse_args(argv)

    report = build_leaderboard_results_summary(
        release_dir=args.release_dir,
        summary_csv=args.summary_csv,
        episodes_jsonl=args.episodes_jsonl,
        models=_parse_models(args.models),
        seeds=_parse_seeds(args.seeds),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if not args.no_markdown:
        write_leaderboard_results_markdown(report, args.output_markdown)
    print(
        f"{report['status']} clean={report['coverage']['clean_model_cells']}/"
        f"{report['coverage']['expected_model_cells']} "
        f"missing={report['coverage']['missing_model_cells']} "
        f"duplicates={report['coverage']['duplicate_clean_cells']}"
    )
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
