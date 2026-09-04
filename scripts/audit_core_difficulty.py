#!/usr/bin/env python3
"""Read-only per-sample audit of core difficulty and decision-value evidence.

Static pressure scores are within-family review proxies. They flag possible
label inversions but do not replace baseline, model, or human calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.difficulty_levels import (  # noqa: E402
    canonical_difficulty_level,
    raw_level_rank,
)
from core.fog_of_war import get_fog_config  # noqa: E402
from core.tool_protocol import DifficultyImperfectionProfile, ToolRegistry  # noqa: E402
from run import load_scenario_yaml  # noqa: E402

SCHEMA_VERSION = "0.2"
STATIC_FEATURES = (
    "n_perturbations",
    "hidden_perturbations",
    "n_dilemmas",
    "decision_depth",
    "observability_burden",
    "persistence_ratio",
    "task_units_log",
    "shock_earliness",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _domain(row: dict[str, Any]) -> str:
    return str(row.get("domain") or "power_grid")


def _scenario_slug(row: dict[str, Any]) -> str:
    path = str(row.get("path") or "")
    if path.startswith("scenarios/"):
        path = path[len("scenarios/") :]
    return path[:-5] if path.endswith(".yaml") else path


def _domain_runtime(domain: str) -> tuple[Any, Any, Any]:
    if domain == "power_grid":
        from domains.power_grid.adapter import _build_tick_budget, _rebuild_seed_from_dict
        from domains.power_grid.native_tools import register_power_grid_tools

        return _rebuild_seed_from_dict, _build_tick_budget, register_power_grid_tools
    if domain == "logistics":
        from domains.logistics.adapter import _build_tick_budget, _rebuild_seed_from_dict
        from domains.logistics.native_tools import register_logistics_tools

        return _rebuild_seed_from_dict, _build_tick_budget, register_logistics_tools
    if domain == "traffic":
        from domains.traffic.adapter import _build_tick_budget, _rebuild_seed_from_dict
        from domains.traffic.native_tools import register_traffic_tools

        return _rebuild_seed_from_dict, _build_tick_budget, register_traffic_tools
    if domain == "microgrid":
        from domains.microgrid.adapter import _build_tick_budget, _rebuild_seed_from_dict
        from domains.microgrid.native_tools import register_microgrid_tools

        return _rebuild_seed_from_dict, _build_tick_budget, register_microgrid_tools
    raise ValueError(f"unsupported released domain: {domain}")


def _tool_inventory(domain: str) -> list[str]:
    _, _, register = _domain_runtime(domain)
    registry = ToolRegistry()
    register(registry, object(), object())
    return registry.names()


def _task_scale(domain: str, body: dict[str, Any]) -> tuple[int, str]:
    config = body.get("backend_config") or {}
    if domain == "traffic":
        return len(body.get("corridors") or []), "corridors"
    if domain == "logistics":
        job_shop = config.get("job_shop") or {}
        if job_shop:
            return int(job_shop.get("operations", 0) or 0), "job_shop_operations"
        orgym = config.get("orgym_env_config") or {}
        if orgym:
            return int(orgym.get("periods", body.get("horizon_ticks", 0)) or 0), "inventory_periods"
        return len(body.get("load_assignments") or []), "customers"
    if domain == "microgrid":
        return len(body.get("load_assignments") or []), "microgrid_loads"

    assignments = len(body.get("load_assignments") or [])
    if assignments:
        return assignments, "grid_load_assignments"
    axes = config.get("source_axes") or {}
    for key, basis in (("n_buses", "network_buses"), ("n_loads", "network_loads")):
        if axes.get(key):
            return int(axes[key]), basis
    return 0, "unresolved"


def _source_locked(row: dict[str, Any]) -> bool:
    lock = row.get("source_lock") or {}
    required = ("data_source", "files", "url", "license", "lock_strategy")
    return bool(lock.get("provenance_complete")) and all(lock.get(k) for k in required)


def _behavioral_evidence(
    row: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None, float | None]:
    ledger = row.get("case_ledger") or {}
    legacy = ledger.get("behavioral_headroom")
    if legacy:
        selection = row.get("core_selection") or {}
        value = selection.get("decision_value_score")
        return True, legacy, float(value) if value is not None else None

    validation = row.get("behavioral_validation") or {}
    if validation.get("status") != "passed":
        return False, None, None
    metrics = dict(validation.get("metrics") or {})
    raw_headroom = metrics.get("best_greedy_or_oracle_raw_headroom")
    if raw_headroom is not None:
        decision_value = float(raw_headroom)
    elif metrics.get("best_greedy_or_oracle_score_headroom") is not None:
        decision_value = float(metrics["best_greedy_or_oracle_score_headroom"])
    elif metrics.get("best_beneficial_relative_cost_gap") is not None:
        decision_value = 100.0 * float(
            metrics["best_beneficial_relative_cost_gap"]
        )
    else:
        decision_value = None
    return True, metrics, decision_value


_SEMANTIC_ID_KEYS = {
    "complexity_metrics",
    "difficulty_level",
    "difficulty_mode",
    "seed",
    "seed_id",
    "scenario_id",
    "scenario_signature",
    "source_denominator_key",
}


def _semantic_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _semantic_value(item)
            for key, item in sorted(value.items())
            if key not in _SEMANTIC_ID_KEYS
        }
    if isinstance(value, list):
        return [_semantic_value(item) for item in value]
    return value


def _semantic_fingerprint(body: dict[str, Any]) -> str:
    payload = json.dumps(
        _semantic_value(body), sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _difficulty_feature_fingerprint(sample: dict[str, Any]) -> str:
    payload = {
        "domain": sample["domain"],
        "backend_kind": sample["backend_kind"],
        "family": sample["family"],
        "horizon_ticks": sample["horizon_ticks"],
        "tick_minutes": sample["tick_minutes"],
        "task_units": sample["task_units"],
        "task_scale_basis": sample["task_scale_basis"],
        "n_perturbations": sample["n_perturbations"],
        "hidden_perturbations": sample["hidden_perturbations"],
        "n_dilemmas": sample["n_dilemmas"],
        "event_kinds": sample["event_kinds"],
        "event_mechanics": sample["event_mechanics"],
        "first_post_start_event_tick": sample["first_post_start_event_tick"],
        "complexity_metrics": sample["complexity_metrics"],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _event_mechanics(perturbations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "kind": str(item.get("kind")),
            "trigger_tick": int(item.get("trigger_tick", 0)),
            "duration_ticks": int(item.get("duration_ticks", 0)),
            "hidden": bool(item.get("hidden")),
            "intensity": float(item.get("intensity", 0.0) or 0.0),
            "target": item.get("target") or {},
        }
        for item in perturbations
    ]


def _sample(row: dict[str, Any], inventories: dict[str, list[str]]) -> dict[str, Any]:
    domain = _domain(row)
    body = load_scenario_yaml(_scenario_slug(row))
    rebuild, build_budget, _ = _domain_runtime(domain)
    seed_obj = rebuild(body, int(body.get("seed", 42)))
    budget = build_budget(seed_obj)
    complexity = seed_obj.complexity_metrics()
    perturbations = list(body.get("perturbations") or [])
    dilemmas = list(body.get("dilemmas") or [])
    hidden = sum(bool(item.get("hidden")) for item in perturbations)
    trigger_ticks = [
        int(item.get("trigger_tick", 0))
        for item in perturbations
        if int(item.get("trigger_tick", 0)) > 0
    ]
    horizon = int(body.get("horizon_ticks", 0) or 0)
    invalid_event_triggers = [
        {
            "kind": str(item.get("kind")),
            "trigger_tick": int(item.get("trigger_tick", 0)),
        }
        for item in perturbations
        if not 0 <= int(item.get("trigger_tick", 0)) < horizon
    ]
    overlong_events = [
        {
            "kind": str(item.get("kind")),
            "trigger_tick": int(item.get("trigger_tick", 0)),
            "duration_ticks": int(item.get("duration_ticks", 0)),
        }
        for item in perturbations
        if int(item.get("trigger_tick", 0)) + int(item.get("duration_ticks", 0))
        > horizon
    ]
    task_units, task_basis = _task_scale(domain, body)
    ledger = row.get("case_ledger") or {}
    measured, headroom, decision_value = _behavioral_evidence(row)
    level = str(row.get("difficulty_level"))
    fog = get_fog_config(level, domain)
    imperfection = DifficultyImperfectionProfile().for_level(level)

    return {
        "scenario_id": row.get("scenario_id"),
        "path": row.get("path"),
        "domain": domain,
        "backend_kind": row.get("backend_kind"),
        "family": row.get("family"),
        "difficulty_mode": row.get("difficulty_mode"),
        "difficulty_level": canonical_difficulty_level(level),
        "source_difficulty_label": row.get("difficulty_level"),
        "difficulty_rank": raw_level_rank(level),
        "horizon_ticks": horizon,
        "tick_minutes": int(body.get("tick_minutes", 0) or 0),
        "max_decision_rounds": horizon,
        "interaction_measurement": "configured_upper_bound_not_observed",
        "available_tool_count": len(inventories[domain]),
        "available_tools": inventories[domain],
        "max_tool_calls_per_tick": int(budget.max_tool_calls_per_tick),
        "max_total_tool_calls": int(budget.max_total_tool_calls),
        "tool_fail_rate": float(imperfection["fail_rate"]),
        "tool_delay_ticks": int(imperfection["delay_ticks"]),
        "fog_visibility": float(fog["visibility"]),
        "fog_noise_std": float(fog["noise_std"]),
        "fog_delay_ticks": int(fog["delay_ticks"]),
        "n_perturbations": len(perturbations),
        "hidden_perturbations": hidden,
        "n_dilemmas": len(dilemmas),
        "event_kinds": sorted({str(item.get("kind")) for item in perturbations}),
        "event_mechanics": _event_mechanics(perturbations),
        "first_post_start_event_tick": min(trigger_ticks) if trigger_ticks else None,
        "invalid_event_triggers": invalid_event_triggers,
        "events_extending_past_horizon": overlong_events,
        "task_units": task_units,
        "task_scale_basis": task_basis,
        "complexity_metrics": complexity,
        "structurally_independent": bool(
            row.get("structural_fingerprint")
            and ledger.get("source_denominator_key")
            and ledger.get("independence_axis")
            and ledger.get("decision_pressure_axis")
        ),
        "source_locked": _source_locked(row),
        "decision_value_score": decision_value,
        "behavioral_headroom_measured": measured,
        "behavioral_headroom": headroom,
        "raw_behavioral_headroom_measured": bool(
            isinstance(headroom, dict)
            and (
                "best_greedy_or_oracle_raw_headroom" in headroom
                or "oracle_wait_raw_headroom" in headroom
            )
        ),
        "live_headroom_cited": bool(ledger.get("live_headroom_citation")),
        "semantic_fingerprint": _semantic_fingerprint(body),
    }


def _proxy_features(sample: dict[str, Any]) -> dict[str, float]:
    complexity = sample.get("complexity_metrics") or {}
    horizon = max(1, int(sample["horizon_ticks"]))
    first = sample.get("first_post_start_event_tick")
    earliness = 0.0 if first is None else 1.0 - min(1.0, float(first) / horizon)
    return {
        "n_perturbations": float(sample["n_perturbations"]),
        "hidden_perturbations": float(sample["hidden_perturbations"]),
        "n_dilemmas": float(sample["n_dilemmas"]),
        "decision_depth": float(complexity.get("decision_depth", 0) or 0),
        "observability_burden": float(complexity.get("observability_burden", 0) or 0),
        "persistence_ratio": float(complexity.get("persistence_ratio", 0) or 0),
        "task_units_log": math.log1p(max(0, int(sample["task_units"]))),
        "shock_earliness": earliness,
    }


def _assign_static_pressure(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_family[(str(sample["domain"]), str(sample["family"]))].append(sample)

    ladders = []
    for (domain, family), rows in sorted(by_family.items()):
        vectors = [_proxy_features(row) for row in rows]
        varying = [
            key
            for key in STATIC_FEATURES
            if max(v[key] for v in vectors) > min(v[key] for v in vectors)
        ]
        for row, vector in zip(rows, vectors, strict=False):
            normalized = []
            for key in varying:
                values = [v[key] for v in vectors]
                normalized.append((vector[key] - min(values)) / (max(values) - min(values)))
            row["static_pressure_score"] = round(
                100.0 * (sum(normalized) / len(normalized) if normalized else 0.0), 3
            )
            row["static_pressure_features"] = vector

        by_rank: dict[int, list[float]] = defaultdict(list)
        labels: dict[int, set[str]] = defaultdict(set)
        for row in rows:
            rank = int(row["difficulty_rank"])
            if rank >= 0:
                by_rank[rank].append(float(row["static_pressure_score"]))
                labels[rank].add(str(row["difficulty_level"]))
        medians = {rank: round(median(values), 3) for rank, values in sorted(by_rank.items())}
        inversions = []
        ordered = sorted(medians)
        for lower, upper in zip(ordered, ordered[1:], strict=False):
            if medians[upper] + 5.0 < medians[lower]:
                inversions.append(
                    {
                        "lower_rank": lower,
                        "upper_rank": upper,
                        "lower_median": medians[lower],
                        "upper_median": medians[upper],
                    }
                )
        review = bool(inversions)
        for row in rows:
            row["static_ladder_review"] = review
            row["static_ladder_promotion_blocking"] = False
        ladders.append(
            {
                "domain": domain,
                "family": family,
                "n_rows": len(rows),
                "varying_static_features": varying,
                "level_labels": {str(k): sorted(v) for k, v in sorted(labels.items())},
                "median_static_pressure_by_rank": {str(k): v for k, v in medians.items()},
                "inversions": inversions,
                "status": "review" if review else "no_static_inversion_detected",
            }
        )
    return {
        "method": "within_family_minmax_mean_static_proxy",
        "warning": "Review signal only; empirical baseline/model runs decide final labels.",
        "n_families": len(ladders),
        "n_families_with_inversions": sum(bool(x["inversions"]) for x in ladders),
        "families": ladders,
    }


def _stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "min": round(ordered[0], 4),
        "median": round(float(median(ordered)), 4),
        "max": round(ordered[-1], 4),
    }


def _distribution(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        grouped[str(sample["domain"])][str(sample["difficulty_level"])].append(sample)
    domain_level = {}
    for domain, levels in sorted(grouped.items()):
        domain_level[domain] = {}
        for level, rows in sorted(levels.items(), key=lambda item: raw_level_rank(item[0])):
            domain_level[domain][level] = {
                "n": len(rows),
                "horizon_ticks": _stats([float(x["horizon_ticks"]) for x in rows]),
                "max_tool_calls_per_tick": _stats([float(x["max_tool_calls_per_tick"]) for x in rows]),
                "n_perturbations": _stats([float(x["n_perturbations"]) for x in rows]),
                "hidden_perturbations": _stats([float(x["hidden_perturbations"]) for x in rows]),
                "n_dilemmas": _stats([float(x["n_dilemmas"]) for x in rows]),
                "task_units": _stats([float(x["task_units"]) for x in rows]),
                "static_pressure_score": _stats([float(x["static_pressure_score"]) for x in rows]),
            }
    counters = {
        "by_domain": Counter(str(x["domain"]) for x in samples),
        "by_backend": Counter(str(x["backend_kind"]) for x in samples),
        "by_family": Counter(str(x["family"]) for x in samples),
        "by_difficulty_level": Counter(str(x["difficulty_level"]) for x in samples),
        "by_difficulty_mode": Counter(str(x["difficulty_mode"]) for x in samples),
    }
    return {
        **{name: dict(counter.most_common()) for name, counter in counters.items()},
        "by_domain_level": domain_level,
    }


def audit_core_difficulty(
    release_dir: Path | None = None,
    *,
    suite_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    if suite_path is None:
        if release_dir is None:
            raise ValueError("release_dir or suite_path is required")
        suite_path = release_dir / "core_suite.json"
    suite = _load_json(suite_path)
    rows = list(suite.get("scenarios") or [])
    if manifest_path is not None:
        manifest = _load_json(manifest_path)
    elif release_dir is not None and (release_dir / "manifest.json").exists():
        manifest = _load_json(release_dir / "manifest.json")
    else:
        manifest = {}
    audit_id = (
        release_dir.name
        if release_dir is not None
        else str(suite.get("suite_id") or suite_path.stem)
    )
    domains = sorted({_domain(row) for row in rows})
    inventories = {domain: _tool_inventory(domain) for domain in domains}
    samples = []
    load_failures = []
    for row in rows:
        try:
            samples.append(_sample(row, inventories))
        except Exception as exc:
            load_failures.append(
                {"scenario_id": str(row.get("scenario_id")), "error": f"{type(exc).__name__}: {exc}"}
            )

    ladder = _assign_static_pressure(samples)
    eligibility = manifest.get("leaderboard_eligibility") or {}

    def _cell_set(key: str) -> set[tuple[str, str, str]]:
        return {
            (
                str(item.get("family")),
                str(item.get("difficulty_mode")),
                str(item.get("difficulty_level")),
            )
            for item in eligibility.get(key, [])
        }

    diagnostic_cells = _cell_set("diagnostic_cells") | _cell_set("uninformative_cells")
    wait_cells = _cell_set("wait_dominant_cells")
    semantic_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        semantic_groups[str(sample["semantic_fingerprint"])].append(sample)
    duplicate_groups = [group for group in semantic_groups.values() if len(group) > 1]
    for group_index, group in enumerate(duplicate_groups, start=1):
        group_id = f"semantic_dup_{group_index:03d}"
        for sample in group:
            sample["semantic_duplicate_group"] = group_id
            sample["semantic_duplicate_group_size"] = len(group)
    feature_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        feature_groups[_difficulty_feature_fingerprint(sample)].append(sample)
    equivalent_groups = [
        group
        for group in feature_groups.values()
        if len(group) > 1
        and len({str(sample["difficulty_level"]) for sample in group}) > 1
    ]
    for group_index, group in enumerate(equivalent_groups, start=1):
        group_id = f"difficulty_equiv_{group_index:03d}"
        for sample in group:
            sample["difficulty_feature_equivalent_group"] = group_id
            sample["difficulty_feature_equivalent_group_size"] = len(group)

    for sample in samples:
        cell = (
            str(sample["family"]),
            str(sample["difficulty_mode"]),
            str(sample["difficulty_level"]),
        )
        sample["diagnostic_cell"] = cell in diagnostic_cells
        sample["wait_dominant_cell"] = cell in wait_cells
        sample["eligibility_match_scope"] = "family/difficulty_mode/difficulty_level"
        hard_issues = []
        if not sample["source_locked"]:
            hard_issues.append("source_lock_incomplete")
        if not sample["structurally_independent"]:
            hard_issues.append("structural_independence_evidence_missing")
        if sample["invalid_event_triggers"]:
            hard_issues.append("event_trigger_out_of_horizon")
        attention = []
        if not sample["behavioral_headroom_measured"]:
            attention.append("behavioral_headroom_not_measured")
        elif not sample["raw_behavioral_headroom_measured"]:
            attention.append("raw_behavioral_headroom_not_measured")
        if float(sample.get("decision_value_score") or 0.0) <= 0:
            attention.append("nonpositive_decision_value")
        if sample["static_ladder_promotion_blocking"]:
            attention.append("matched_family_static_difficulty_inversion")
        if sample.get("semantic_duplicate_group"):
            attention.append("semantic_duplicate_candidate")
        if sample.get("difficulty_feature_equivalent_group"):
            attention.append("cross_label_difficulty_feature_equivalent")
        if sample["events_extending_past_horizon"]:
            attention.append("event_duration_extends_past_horizon")
        if sample["diagnostic_cell"]:
            attention.append("diagnostic_leaderboard_cell")
        if sample["wait_dominant_cell"]:
            attention.append("inherited_wait_dominant_cell")
        sample["hard_issues"] = hard_issues
        sample["attention"] = attention
        sample["quality_status"] = "fail" if hard_issues else ("attention" if attention else "pass")

    measured = sum(bool(x["behavioral_headroom_measured"]) for x in samples)
    raw_measured = sum(
        bool(x["raw_behavioral_headroom_measured"]) for x in samples
    )
    status_counts = Counter(str(x["quality_status"]) for x in samples)
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": audit_id,
        "scope": "read_only_core_difficulty_audit",
        "changes_suite_membership": False,
        "rubric": {
            "difficulty_order": ["basic", "medium", "high", "extreme"],
            "configured_dimensions": [
                "max_decision_rounds",
                "tool_budget",
                "tool_inventory",
                "tool_imperfection",
                "fog_of_war",
                "event_load",
                "hidden_information",
                "task_scale",
            ],
            "empirical_requirements": [
                "wait/random/greedy/oracle headroom",
                "multi-model solve-rate confidence intervals",
                "repeat-trial stability",
                "effective tool chain and decision-point counts",
            ],
            "static_proxy_is_not_empirical_difficulty": True,
        },
        "summary": {
            "n_rows": len(rows),
            "n_rows_audited": len(samples),
            "n_yaml_load_failures": len(load_failures),
            "n_source_locked": sum(bool(x["source_locked"]) for x in samples),
            "n_structurally_independent": sum(bool(x["structurally_independent"]) for x in samples),
            "n_behavioral_headroom_measured": measured,
            "n_behavioral_headroom_missing": len(samples) - measured,
            "n_raw_behavioral_headroom_measured": raw_measured,
            "n_raw_behavioral_headroom_missing": len(samples) - raw_measured,
            "n_live_headroom_cited": sum(bool(x["live_headroom_cited"]) for x in samples),
            "n_static_ladder_review": sum(bool(x["static_ladder_review"]) for x in samples),
            "n_rows_with_invalid_event_triggers": sum(
                bool(x["invalid_event_triggers"]) for x in samples
            ),
            "n_rows_with_events_extending_past_horizon": sum(
                bool(x["events_extending_past_horizon"]) for x in samples
            ),
            "n_semantic_duplicate_rows": sum(
                bool(x.get("semantic_duplicate_group")) for x in samples
            ),
            "n_cross_label_difficulty_equivalent_rows": sum(
                bool(x.get("difficulty_feature_equivalent_group")) for x in samples
            ),
            "n_diagnostic_cell_rows": sum(bool(x["diagnostic_cell"]) for x in samples),
            "n_wait_dominant_cell_rows": sum(bool(x["wait_dominant_cell"]) for x in samples),
            "quality_status_counts": dict(sorted(status_counts.items())),
            "release_quality_claim": (
                "structurally_valid_but_behaviorally_under_verified"
                if measured < len(samples)
                else (
                    "behavioral_headroom_measured_but_raw_difficulty_pending"
                    if raw_measured < len(samples)
                    else "structurally_and_behaviorally_verified"
                )
            ),
        },
        "load_failures": load_failures,
        "tool_inventory_by_domain": inventories,
        "difficulty_ladder_review": ladder,
        "semantic_duplicates": {
            "method": "sha256_of_yaml_after_removing_ids_seed_mode_level_and_denominator_key",
            "n_groups": len(duplicate_groups),
            "groups": [
                {
                    "group_id": group[0]["semantic_duplicate_group"],
                    "size": len(group),
                    "members": [str(sample["scenario_id"]) for sample in group],
                    "labels": sorted(
                        {
                            f"{sample['difficulty_mode']}/{sample['difficulty_level']}"
                            for sample in group
                        }
                    ),
                }
                for group in duplicate_groups
            ],
        },
        "cross_label_difficulty_equivalence": {
            "method": "exact_match_on_runtime_independent_difficulty_feature_vector",
            "warning": "Candidate review signal; equal static features do not prove equal empirical difficulty.",
            "n_groups": len(equivalent_groups),
            "groups": [
                {
                    "group_id": group[0]["difficulty_feature_equivalent_group"],
                    "size": len(group),
                    "members": [str(sample["scenario_id"]) for sample in group],
                    "levels": sorted({str(sample["difficulty_level"]) for sample in group}),
                }
                for group in equivalent_groups
            ],
        },
        "distribution": _distribution(samples),
        "samples": samples,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# {report['release_id']} Core Difficulty Audit",
        "",
        "This is a read-only audit. Static pressure is a review proxy, not a replacement for baseline and model runs.",
        "",
        "## Verdict",
        "",
        f"- Rows audited: {summary['n_rows_audited']}/{summary['n_rows']}",
        f"- Source-locked: {summary['n_source_locked']}/{summary['n_rows']}",
        f"- Structurally independent: {summary['n_structurally_independent']}/{summary['n_rows']}",
        f"- Behavioral headroom measured: {summary['n_behavioral_headroom_measured']}/{summary['n_rows']}",
        f"- Raw-score headroom measured: {summary['n_raw_behavioral_headroom_measured']}/{summary['n_rows']}",
        f"- Rows requiring static ladder review: {summary['n_static_ladder_review']}",
        f"- Claim: `{summary['release_quality_claim']}`",
        "",
        "## Domain Distribution",
        "",
        "| Domain | Rows | Share |",
        "| --- | ---: | ---: |",
    ]
    total = max(1, int(summary["n_rows"]))
    for domain, count in report["distribution"]["by_domain"].items():
        lines.append(f"| {domain} | {count} | {100 * count / total:.1f}% |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Configured ticks and tool-call limits are upper bounds. Effective interactions, shortest successful tool chains, and empirical difficulty require trajectory runs. Rows without measured behavioral headroom should not yet be described as individually proven discriminative tests.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    if args.release_dir is None and args.suite is None:
        parser.error("one of --release-dir or --suite is required")
    report = audit_core_difficulty(
        args.release_dir.resolve() if args.release_dir else None,
        suite_path=args.suite.resolve() if args.suite else None,
        manifest_path=args.manifest.resolve() if args.manifest else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 1 if report["summary"]["n_yaml_load_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
