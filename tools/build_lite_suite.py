#!/usr/bin/env python3
"""Build the deterministic, coverage-closed OPERATE-Lite development suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PREFERRED_RANGE = (150, 200)
QUALITY_PREFERRED_RANGE = (100, 165)
QUALITY_EVIDENCE_NAME = "lite_quality_evidence.json"
QUALITY_ALGORITHM = "quality_gated_cpu_headroom_hy3_v9"
TRUE_EASY_FLAGS = frozenset(
    {
        "too_easy_ceiling",
        "too_easy_saturated",
        "dead_headroom",
        "wait_parity_collapse",
    }
)
WEAK_ORACLE_FLAGS = frozenset({"cpu_oracle_dead", "oracle_yardstick_broken"})
HARD_EXCLUDE_FLAGS = TRUE_EASY_FLAGS | {"lite_cost_extreme_horizon"} | WEAK_ORACLE_FLAGS
SOFT_EXCLUDE_FLAGS = frozenset(
    {
        "short_and_easy",
        "lite_cost_long_horizon",
        "cpu_oracle_dead",
        "oracle_yardstick_broken",
    }
)
QUALITY_RESCUE_FLAGS = frozenset(
    {
        "keep_headroom",
        "all_models_zero_oracle_open",
        "hy3_hard_strict",
        "hy3_hard_band",
        "hy3_zero_unresolved",
        "cpu_greedy_peer_strong",
    }
)
MODEL_LAG_MEAN = 40.0
DOMAIN_FLOOR = {
    "autonomous_driving": 7,
    "building_energy": 3,
    "datacenter": 10,
    "logistics": 20,
    "microgrid": 10,
    "power_grid": 6,
    "traffic": 0,
}
FAMILY_SOFT_CAP = {
    "job_shop_dispatch": 8,
    "citylearn_der_storage_control": 10,
    "signal_coordination": 4,
    "microgrid_economic_dispatch_24h": 26,
    "vrptw_dispatch": 22,
    "cvrp_dispatch": 16,
    "gpu_cluster_sla_control": 16,
    "gpu_cluster_queue_control": 14,
    "inventory_replenishment": 12,
}
MUST_KEEP_FLAGS = frozenset(
    {
        "keep_headroom",
        "all_models_zero_oracle_open",
        "keep_discriminative",
        "cpu_greedy_peer_strong",
    }
)
MUST_KEEP_IGNORE_CAP = frozenset(MUST_KEEP_FLAGS)
FULL_RETENTION_DOMAINS = (
    "autonomous_driving",
    "building_energy",
    "microgrid",
    "power_grid",
    "traffic",
)
REQUIRED_CORE_FIELDS = (
    "backend_kind",
    "difficulty_level",
    "difficulty_mode",
    "domain",
    "family",
    "horizon_ticks",
    "path",
    "physical_source_key",
    "scenario_id",
    "scenario_signature",
    "semantic_fingerprint",
    "source_denominator_key",
    "structural_fingerprint",
    "yaml_sha256",
)
SCALE_FIELDS = (
    "n_jobs",
    "n_machines",
    "n_operations",
    "n_buses",
    "n_lines",
    "n_loads",
    "n_vehicles",
    "n_buildings",
    "n_nodes",
    "n_edges",
)
HORIZON_BUCKETS = (
    (1, 8, "1-8"),
    (9, 16, "9-16"),
    (17, 48, "17-48"),
    (49, 96, "49-96"),
    (97, 192, "97-192"),
    (193, None, "193+"),
)


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _horizon_bucket(value: int) -> str:
    for lower, upper, label in HORIZON_BUCKETS:
        if value >= lower and (upper is None or value <= upper):
            return label
    raise ValueError(f"invalid horizon: {value}")


def _validate_core_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Core contains no scenarios")
    for field in ("scenario_id", "path", "structural_fingerprint"):
        if len(rows) != len({str(row.get(field)) for row in rows}):
            raise ValueError(f"Core contains duplicate {field}")
    for row in rows:
        missing = [field for field in REQUIRED_CORE_FIELDS if not row.get(field)]
        if missing:
            raise ValueError(f"{row.get('scenario_id')}: missing {', '.join(missing)}")
        if (
            row.get("status") != "core_locked"
            or row.get("core_disposition") != "core_locked"
        ):
            raise ValueError(f"{row['scenario_id']}: row is not Core-locked")


def _read_scenario(row: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    relative = PurePosixPath(row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe scenario path: {row['path']}")
    path = (repo_root / relative).resolve()
    if not path.is_relative_to(repo_root.resolve()):
        raise ValueError(f"scenario path escapes repository: {row['path']}")
    raw = path.read_bytes()
    if _sha256_bytes(raw) != row["yaml_sha256"]:
        raise ValueError(f"scenario YAML hash mismatch: {row['path']}")
    body = yaml.safe_load(raw)
    if not isinstance(body, dict) or any(
        body.get(field) != row[field]
        for field in (
            "scenario_id",
            "domain",
            "backend_kind",
            "family",
            "difficulty_level",
            "difficulty_mode",
            "horizon_ticks",
        )
    ):
        raise ValueError(f"scenario identity mismatch: {row['path']}")
    return body


def _required(mapping: dict[str, Any], *fields: str) -> tuple[Any, ...]:
    if any(mapping.get(field) in (None, "") for field in fields):
        raise ValueError(f"required Lite coverage metadata missing: {fields}")
    return tuple(mapping[field] for field in fields)


def _coverage_tokens(row: dict[str, Any], body: dict[str, Any]) -> set[str]:
    config = body.get("backend_config") or {}
    provenance = body.get("provenance") or {}
    metrics = body.get("complexity_metrics") or {}
    axes = config.get("source_axes") or {}
    task = config.get("task_contract") or {}
    requirements = config.get("task_requirements") or {}
    base = (row["domain"], row["backend_kind"], row["family"])
    events = sorted(
        {
            (str(event.get("kind")), bool(event.get("hidden")))
            for event in body.get("perturbations", [])
            if isinstance(event, dict)
        }
    )
    controls = sorted(
        set(task.get("native_controls") or [])
        | {
            str(item.get("tool"))
            for item in requirements.get("ordered_tool_milestones", [])
            if isinstance(item, dict)
        }
    )
    sizes = [
        (field, metrics.get(field, axes.get(field)))
        for field in SCALE_FIELDS
        if type(metrics.get(field, axes.get(field))) in (int, float)
        and metrics.get(field, axes.get(field)) > 0
    ]
    features = [
        (
            "cell",
            *base,
            row["difficulty_mode"],
            row["difficulty_level"],
            _horizon_bucket(int(row["horizon_ticks"])),
        ),
        ("dataset", *base, str(provenance.get("data_source", "unspecified"))),
        ("events", *base, events),
        ("controls", *base, str(task.get("contract", "unspecified")), controls),
        ("scale", *base, sizes),
    ]
    variation: list[tuple[Any, ...]] = []
    if row["domain"] == "autonomous_driving":
        variation.append(
            (
                "hazard_response",
                *_required(provenance, "hazard_kind"),
                *_required(
                    requirements,
                    "latest_preventive_command_tick",
                    "paid_safety_inspection_deadline_tick",
                ),
            )
        )
    elif row["domain"] == "microgrid":
        variation.append(
            (
                "site_forecast_supply",
                *_required(
                    config,
                    "site",
                    "forecast_bias",
                    "forecast_error_sigma",
                    "genset_available",
                ),
            )
        )
    elif row["domain"] == "building_energy":
        native_events = config.get("native_source_events")
        if not isinstance(native_events, list) or not native_events:
            raise ValueError(
                "required Lite coverage metadata missing: native_source_events"
            )
        for event in native_events:
            source, kind, channel = _required(event, "source_asset", "kind", "channel")
            variation.append(
                (
                    "dataset_native_change",
                    str(PurePosixPath(source).parent),
                    kind,
                    channel,
                    bool(event.get("hidden")),
                )
            )
    elif row["domain"] == "power_grid":
        for field in ("network", "feeder", "decision_axis"):
            if field in config:
                variation.append((field, str(config[field])))
    elif row["domain"] == "traffic":
        corridors = config.get("corridor_tls_map")
        if not isinstance(corridors, dict) or not corridors:
            raise ValueError(
                "required Lite coverage metadata missing: corridor_tls_map"
            )
        variation.append(("controlled_topology", sorted(corridors)))
    features.extend(("source_variation", *base, *value) for value in variation)
    return {_canonical_bytes(feature).decode("utf-8") for feature in features}


def _select_with_audit(
    rows: list[dict[str, Any]],
    *,
    repo_root: Path,
    preferred_range: tuple[int, int] = PREFERRED_RANGE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    minimum, maximum = preferred_range
    if not 0 <= minimum <= maximum:
        raise ValueError("invalid Lite preferred size range")
    _validate_core_rows(rows)
    remaining = sorted(rows, key=lambda row: row["scenario_id"])
    tokens = {
        row["scenario_id"]: _coverage_tokens(row, _read_scenario(row, repo_root))
        for row in remaining
    }
    features = sorted(set().union(*tokens.values()))
    feature_ids = {feature: index for index, feature in enumerate(features)}
    covered: set[str] = set()
    sources: set[str] = set()
    selected: list[dict[str, Any]] = []
    gains: dict[str, set[str]] = {}
    owners: dict[str, str] = {}
    while remaining:
        # max() keeps the first scenario ID on an exact tie.
        best = max(
            remaining,
            key=lambda row: (
                len(tokens[row["scenario_id"]] - covered),
                row["physical_source_key"] not in sources,
            ),
        )
        new = tokens[best["scenario_id"]] - covered
        if not new:
            break
        selected.append(best)
        gains[best["scenario_id"]] = new
        owners.update({feature: best["scenario_id"] for feature in new})
        covered.update(new)
        sources.add(best["physical_source_key"])
        remaining.remove(best)
    core_ids = [row["scenario_id"] for row in selected]
    available: dict[str, set[str]] = defaultdict(set)
    supported: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for token in tokens[row["scenario_id"]]:
            available[token].add(row["physical_source_key"])
    for row in selected:
        for token in tokens[row["scenario_id"]]:
            supported[token].add(row["physical_source_key"])
    enrichment: dict[str, set[str]] = {}
    rounds = []
    for target in range(2, max(map(len, available.values()), default=1) + 1):
        if len(selected) >= minimum:
            break
        added = []

        def support_gain(row: dict[str, Any]) -> set[str]:
            return {
                token
                for token in tokens[row["scenario_id"]]
                if len(supported[token]) < min(target, len(available[token]))
                and row["physical_source_key"] not in supported[token]
            }

        while remaining and len(selected) < maximum:
            best = max(
                remaining,
                key=lambda row: (
                    len(support_gain(row)),
                    row["physical_source_key"] not in sources,
                ),
            )
            new_support = support_gain(best)
            if not new_support:
                break
            selected.append(best)
            enrichment[best["scenario_id"]] = new_support
            added.append(best["scenario_id"])
            sources.add(best["physical_source_key"])
            for token in tokens[best["scenario_id"]]:
                supported[token].add(best["physical_source_key"])
            remaining.remove(best)
        rounds.append(
            {
                "target_distinct_sources": target,
                "added_scenario_ids": added,
                "completed": all(
                    len(supported[token]) >= min(target, len(source_set))
                    for token, source_set in available.items()
                ),
            }
        )
    # Preserve admitted window/condition variants in the current small domains.
    # Add the scarce medium/high datacenter cases without replacing basic rows.
    retained = set()
    datacenter_retained = set()
    for row in list(remaining):
        keep_datacenter = (
            row["domain"] == "datacenter"
            and row["difficulty_level"] in {"medium", "high"}
        )
        if row["domain"] in FULL_RETENTION_DOMAINS or keep_datacenter:
            selected.append(row)
            (datacenter_retained if keep_datacenter else retained).add(row["scenario_id"])
            for token in tokens[row["scenario_id"]]:
                supported[token].add(row["physical_source_key"])
            remaining.remove(row)
    audit_rows = []
    for row in sorted(rows, key=lambda row: row["scenario_id"]):
        scenario_id = row["scenario_id"]
        in_core = scenario_id in gains
        included = (
            in_core or scenario_id in enrichment
            or scenario_id in retained or scenario_id in datacenter_retained
        )
        can_add_support = any(
            row["physical_source_key"] not in supported[token]
            for token in tokens[scenario_id]
        )
        audit_rows.append(
            {
                "scenario_id": scenario_id,
                "included": included,
                "selection_stage": "coverage_core"
                if in_core
                else (
                    "datacenter_difficulty_retention" if scenario_id in datacenter_retained
                    else (
                        "small_domain_retention" if scenario_id in retained
                        else ("diversity_enrichment" if included else "excluded")
                    )
                ),
                "reason": "adds_coverage"
                if in_core
                else (
                    "preserves_admitted_datacenter_medium_high"
                    if scenario_id in datacenter_retained
                    else (
                        "preserves_admitted_small_domain_variation"
                        if scenario_id in retained
                        else (
                            "adds_independent_source_support"
                            if included
                            else (
                                "preferred_budget_reached"
                                if can_add_support
                                else "coverage_already_represented"
                            )
                        )
                    )
                ),
                "feature_ids": sorted(
                    feature_ids[token] for token in tokens[scenario_id]
                ),
                "new_feature_ids": sorted(
                    feature_ids[token] for token in gains.get(scenario_id, set())
                ),
                "new_source_support_feature_ids": sorted(
                    feature_ids[token] for token in enrichment.get(scenario_id, set())
                ),
                "covered_by": sorted({owners[token] for token in tokens[scenario_id]}),
            }
        )
    return sorted(selected, key=lambda row: row["scenario_id"]), {
        "coverage_complete": covered == set(features),
        "coverage_core_scenario_ids": core_ids,
        "n_coverage_core": len(core_ids),
        "n_diversity_enrichment": len(enrichment),
        "n_small_domain_retention": len(retained),
        "n_datacenter_difficulty_retention": len(datacenter_retained),
        "preferred_size_range": list(preferred_range),
        "budget_satisfied": minimum <= len(selected) <= maximum,
        "enrichment_rounds": rounds,
        "feature_source_support": [
            {
                "feature_id": feature_ids[token],
                "available_sources": len(available[token]),
                "selected_sources": len(supported[token]),
            }
            for token in features
        ],
        "features": [json.loads(feature) for feature in features],
        "selection_order": [row["scenario_id"] for row in selected],
        "rows": audit_rows,
    }


def select_lite(
    rows: list[dict[str, Any]],
    *,
    repo_root: Path = REPO_ROOT,
) -> list[dict[str, Any]]:
    return _select_with_audit(rows, repo_root=repo_root)[0]


def _quality_evidence_path(core_path: Path) -> Path:
    return core_path.parent / QUALITY_EVIDENCE_NAME


def load_quality_evidence(core_path: Path) -> dict[str, Any]:
    path = _quality_evidence_path(core_path)
    if not path.is_file():
        raise ValueError(f"Lite quality evidence missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "operate-lite-quality-evidence-v1":
        raise ValueError(f"unsupported Lite quality evidence schema: {path}")
    if not isinstance(payload.get("rows"), dict) or not payload["rows"]:
        raise ValueError(f"Lite quality evidence has no rows: {path}")
    return payload


def _quality_record(evidence: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    rec = evidence.get("rows", {}).get(scenario_id)
    return rec if isinstance(rec, dict) else {}


def _quality_flags(evidence: dict[str, Any], scenario_id: str) -> set[str]:
    flags = _quality_record(evidence, scenario_id).get("flags") or []
    return {str(flag) for flag in flags}


def _quality_rank(row: dict[str, Any], evidence: dict[str, Any]) -> float:
    rec = _quality_record(evidence, row["scenario_id"])
    flags = _quality_flags(evidence, row["scenario_id"])
    score = 0.0
    if "keep_headroom" in flags:
        score += 40.0
    if "keep_discriminative" in flags:
        score += 25.0
    if "all_models_zero_oracle_open" in flags:
        score += 30.0
    if "cpu_headroom" in flags:
        score += 20.0
    if "cpu_greedy_peer_strong" in flags:
        four_mean = rec.get("four_mean")
        greedy = rec.get("cpu_greedy")
        if isinstance(four_mean, (int, float)) and isinstance(greedy, (int, float)):
            if float(four_mean) <= 0.55 * float(greedy):
                score += 20.0
            else:
                score += 5.0
        else:
            score += 5.0
    if "high_disagreement" in flags:
        score += 10.0
    if "hy3_hard_strict" in flags:
        score += 24.0
    elif "hy3_hard_band" in flags:
        score += 12.0
    if "hy3_zero_unresolved" in flags:
        score += 18.0
    if not rec.get("in_cpu_baseline", rec.get("in_parent_lite")) and "hy3_hard_strict" in flags:
        score += 12.0
    four_mean = rec.get("four_mean")
    if isinstance(four_mean, (int, float)) and 5 <= float(four_mean) < 50:
        score += 10.0
    hy3 = rec.get("hy3_primary")
    if isinstance(hy3, (int, float)) and 50 <= float(hy3) < 70:
        score -= 10.0
    if _is_hard_exclude(row, evidence):
        score -= 80.0
    elif _is_soft_exclude(row, evidence):
        score -= 20.0
    if "beats_oracle" in flags:
        score -= 5.0
    score -= min(int(row["horizon_ticks"]), 400) / 50.0
    score += {
        "extreme": 8.0,
        "high": 5.0,
        "medium": 2.0,
        "basic": 0.0,
    }.get(str(row["difficulty_level"]), 0.0)
    return score


def _models_still_lag(rec: dict[str, Any]) -> bool:
    mean = rec.get("four_mean")
    if mean is None:
        return True
    return float(mean) < MODEL_LAG_MEAN


def _is_quality_rescue(row: dict[str, Any], evidence: dict[str, Any]) -> bool:
    flags = _quality_flags(evidence, row["scenario_id"])
    rec = _quality_record(evidence, row["scenario_id"])
    if not (flags & QUALITY_RESCUE_FLAGS):
        return False
    if "keep_headroom" in flags or "all_models_zero_oracle_open" in flags:
        return True
    return _models_still_lag(rec)


def _is_hardness_eligible(row: dict[str, Any], evidence: dict[str, Any]) -> bool:
    flags = _quality_flags(evidence, row["scenario_id"])
    rec = _quality_record(evidence, row["scenario_id"])
    mean = rec.get("four_mean")
    hy3 = rec.get("hy3_primary")
    mn = rec.get("four_min")
    mx = rec.get("four_max")
    oracle = rec.get("oracle_primary")
    if flags & TRUE_EASY_FLAGS:
        return False
    if isinstance(mean, (int, float)) and float(mean) >= 50.0:
        return False
    if (
        isinstance(mean, (int, float))
        and isinstance(mn, (int, float))
        and isinstance(mx, (int, float))
        and isinstance(oracle, (int, float))
        and (float(mx) - float(mn)) < 2.0
        and abs(float(mean) - float(oracle)) < 2.0
        and float(mean) >= 30.0
    ):
        return False
    if isinstance(hy3, (int, float)) and float(hy3) >= 50.0:
        if mean is None or float(mean) >= 30.0:
            return False
    if "keep_headroom" in flags and (
        mean is None or (isinstance(mean, (int, float)) and float(mean) < 50.0)
    ):
        return True
    if "all_models_zero_oracle_open" in flags:
        return True
    if "hy3_hard_strict" in flags:
        return True
    if "hy3_zero_unresolved" in flags:
        return True
    if "cpu_greedy_peer_strong" in flags:
        greedy = rec.get("cpu_greedy")
        if (
            isinstance(greedy, (int, float))
            and isinstance(mean, (int, float))
            and float(mean) <= 0.55 * float(greedy)
        ):
            return True
        if (
            isinstance(greedy, (int, float))
            and mean is None
            and isinstance(hy3, (int, float))
            and float(hy3) <= 0.55 * float(greedy)
        ):
            return True
    if (
        "keep_discriminative" in flags
        and isinstance(mean, (int, float))
        and float(mean) < MODEL_LAG_MEAN
        and (mn is None or float(mn) == 0.0)
        and (hy3 is None or float(hy3) < 50.0)
    ):
        return True
    return False


def _is_hard_exclude(row: dict[str, Any], evidence: dict[str, Any]) -> bool:
    flags = _quality_flags(evidence, row["scenario_id"])
    if flags & TRUE_EASY_FLAGS:
        return True
    rescued = _is_quality_rescue(row, evidence)
    if "lite_cost_extreme_horizon" in flags:
        if not (
            rescued
            and (
                "keep_headroom" in flags or "all_models_zero_oracle_open" in flags
            )
        ):
            return True
    elif flags & WEAK_ORACLE_FLAGS and not rescued:
        return True
    return not _is_hardness_eligible(row, evidence)


def _is_soft_exclude(row: dict[str, Any], evidence: dict[str, Any]) -> bool:
    if _is_quality_rescue(row, evidence):
        return False
    return bool(_quality_flags(evidence, row["scenario_id"]) & SOFT_EXCLUDE_FLAGS)


def _stratum_value(row: dict[str, Any], field: str) -> str:
    if field == "horizon_bucket":
        return _horizon_bucket(int(row["horizon_ticks"]))
    return str(row[field])


def _select_quality_gated_v6(
    rows: list[dict[str, Any]],
    *,
    repo_root: Path,
    evidence: dict[str, Any],
    preferred_range: tuple[int, int] = QUALITY_PREFERRED_RANGE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    minimum, maximum = preferred_range
    if not 0 <= minimum <= maximum:
        raise ValueError("invalid Lite preferred size range")
    _validate_core_rows(rows)
    ordered = sorted(rows, key=lambda row: row["scenario_id"])
    tokens = {
        row["scenario_id"]: _coverage_tokens(row, _read_scenario(row, repo_root))
        for row in ordered
    }
    by_id = {row["scenario_id"]: row for row in ordered}

    hard_ids = {
        row["scenario_id"] for row in ordered if _is_hard_exclude(row, evidence)
    }
    eligible_ids = {row["scenario_id"] for row in ordered} - hard_ids
    restored_ids: set[str] = set()
    eligible_rows = [by_id[sid] for sid in eligible_ids]
    for field in ("backend_kind", "family", "difficulty_level", "horizon_bucket"):
        have = {_stratum_value(by_id[sid], field) for sid in eligible_ids}
        need = {_stratum_value(row, field) for row in eligible_rows}
        for missing in sorted(need - have):
            candidates = [
                by_id[sid]
                for sid in sorted(hard_ids)
                if _stratum_value(by_id[sid], field) == missing
            ]
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda row: (
                    _quality_rank(row, evidence),
                    -int(row["horizon_ticks"]),
                    row["scenario_id"],
                ),
            )
            hard_ids.remove(best["scenario_id"])
            eligible_ids.add(best["scenario_id"])
            restored_ids.add(best["scenario_id"])

    eligible_features = sorted(
        set().union(*(tokens[sid] for sid in eligible_ids)) if eligible_ids else set()
    )
    feature_ids = {feature: index for index, feature in enumerate(eligible_features)}
    covered: set[str] = set()
    sources: set[str] = set()
    selected: list[dict[str, Any]] = []
    gains: dict[str, set[str]] = {}
    owners: dict[str, str] = {}
    remaining = [row for row in ordered if row["scenario_id"] in eligible_ids]
    while remaining:
        best = max(
            remaining,
            key=lambda row: (
                len(tokens[row["scenario_id"]] - covered),
                _quality_rank(row, evidence),
                row["physical_source_key"] not in sources,
                -int(row["horizon_ticks"]),
            ),
        )
        new = tokens[best["scenario_id"]] - covered
        if not new:
            break
        selected.append(best)
        gains[best["scenario_id"]] = new
        owners.update({feature: best["scenario_id"] for feature in new})
        covered.update(new)
        sources.add(best["physical_source_key"])
        remaining.remove(best)

    selected_ids = {row["scenario_id"] for row in selected}
    for sid in sorted(restored_ids - selected_ids):
        row = by_id[sid]
        selected.append(row)
        selected_ids.add(sid)
        remaining = [item for item in remaining if item["scenario_id"] != sid]
        sources.add(row["physical_source_key"])

    stratum_ids: set[str] = set()
    for field in ("backend_kind", "family", "difficulty_level", "horizon_bucket"):
        have = {_stratum_value(row, field) for row in selected}
        need = {_stratum_value(by_id[sid], field) for sid in eligible_ids}
        for missing in sorted(need - have):
            pool = [
                row
                for row in ordered
                if row["scenario_id"] not in selected_ids
                and row["scenario_id"] in eligible_ids
                and _stratum_value(row, field) == missing
            ]
            if not pool:
                continue
            best = max(
                pool,
                key=lambda row: (
                    not _is_hard_exclude(row, evidence),
                    _quality_rank(row, evidence),
                    -int(row["horizon_ticks"]),
                    row["scenario_id"],
                ),
            )
            selected.append(best)
            selected_ids.add(best["scenario_id"])
            stratum_ids.add(best["scenario_id"])
            remaining = [
                item for item in remaining if item["scenario_id"] != best["scenario_id"]
            ]
            sources.add(best["physical_source_key"])

    floor_ids: set[str] = set()
    for domain, floor in DOMAIN_FLOOR.items():
        have = sum(1 for row in selected if row["domain"] == domain)
        pool = [row for row in remaining if row["domain"] == domain]
        pool.sort(
            key=lambda row: (
                not _is_soft_exclude(row, evidence),
                _quality_rank(row, evidence),
                -int(row["horizon_ticks"]),
                row["scenario_id"],
            ),
            reverse=True,
        )
        while have < floor and pool:
            row = pool.pop(0)
            selected.append(row)
            selected_ids.add(row["scenario_id"])
            floor_ids.add(row["scenario_id"])
            remaining.remove(row)
            sources.add(row["physical_source_key"])
            have += 1

    must_keep_ids: set[str] = set()
    family_counts = Counter(row["family"] for row in selected)
    must_pool = [
        row
        for row in remaining
        if _quality_flags(evidence, row["scenario_id"]) & MUST_KEEP_FLAGS
    ]
    must_pool.sort(
        key=lambda row: (
            bool(_quality_flags(evidence, row["scenario_id"]) & MUST_KEEP_IGNORE_CAP),
            _quality_rank(row, evidence),
            -int(row["horizon_ticks"]),
            row["scenario_id"],
        ),
        reverse=True,
    )
    for row in must_pool:
        flags = _quality_flags(evidence, row["scenario_id"])
        ignore_cap = bool(flags & MUST_KEEP_IGNORE_CAP)
        cap = FAMILY_SOFT_CAP.get(row["family"], 10**9)
        if not ignore_cap and family_counts[row["family"]] >= cap:
            continue
        selected.append(row)
        selected_ids.add(row["scenario_id"])
        must_keep_ids.add(row["scenario_id"])
        family_counts[row["family"]] += 1
        remaining.remove(row)
        sources.add(row["physical_source_key"])

    enrichment_ids: set[str] = set()
    cells = {
        (
            row["family"],
            _horizon_bucket(int(row["horizon_ticks"])),
            row["difficulty_level"],
        )
        for row in selected
    }

    def enrich_key(row: dict[str, Any]) -> tuple:
        flags = _quality_flags(evidence, row["scenario_id"])
        rec = _quality_record(evidence, row["scenario_id"])
        hy3_hard = "hy3_hard_strict" in flags or "hy3_zero_unresolved" in flags
        core_not_193 = not rec.get("in_cpu_baseline", rec.get("in_parent_lite"))
        cell = (
            row["family"],
            _horizon_bucket(int(row["horizon_ticks"])),
            row["difficulty_level"],
        )
        under_cap = family_counts[row["family"]] < FAMILY_SOFT_CAP.get(
            row["family"], 10**9
        )
        return (
            not _is_soft_exclude(row, evidence),
            hy3_hard,
            cell not in cells,
            core_not_193 and hy3_hard,
            under_cap,
            _quality_rank(row, evidence),
            -int(row["horizon_ticks"]),
            row["scenario_id"],
        )

    while remaining and len(selected) < maximum:
        best = max(remaining, key=enrich_key)
        hy3_hard = enrich_key(best)[1]
        under_cap = enrich_key(best)[4]
        if len(selected) >= minimum:
            if not hy3_hard:
                break
            if not under_cap or _is_soft_exclude(best, evidence):
                remaining.remove(best)
                continue
        selected.append(best)
        selected_ids.add(best["scenario_id"])
        enrichment_ids.add(best["scenario_id"])
        family_counts[best["family"]] += 1
        cells.add(
            (
                best["family"],
                _horizon_bucket(int(best["horizon_ticks"])),
                best["difficulty_level"],
            )
        )
        remaining.remove(best)
        sources.add(best["physical_source_key"])

    while remaining and len(selected) < minimum:
        best = max(
            remaining,
            key=lambda row: (
                not _is_soft_exclude(row, evidence),
                _quality_rank(row, evidence),
                -int(row["horizon_ticks"]),
                row["scenario_id"],
            ),
        )
        selected.append(best)
        selected_ids.add(best["scenario_id"])
        enrichment_ids.add(best["scenario_id"])
        remaining.remove(best)
        sources.add(best["physical_source_key"])

    audit_rows = []
    for row in ordered:
        scenario_id = row["scenario_id"]
        included = scenario_id in selected_ids
        in_core = scenario_id in gains
        if in_core:
            stage, reason = "coverage_core", "adds_coverage"
        elif scenario_id in restored_ids:
            stage, reason = (
                "restored_excluded_representative",
                "restores_core_stratum_after_quality_gate",
            )
        elif scenario_id in stratum_ids:
            stage, reason = "stratum_completion", "completes_core_runtime_stratum"
        elif scenario_id in floor_ids:
            stage, reason = "domain_floor", "preserves_domain_floor"
        elif scenario_id in must_keep_ids:
            stage, reason = "must_keep_headroom", "preserves_open_headroom_or_cpu_gap"
        elif scenario_id in enrichment_ids:
            stage, reason = (
                "quality_enrichment",
                "adds_headroom_or_hy3_hard_diversity",
            )
        elif scenario_id in hard_ids:
            stage, reason = "excluded", "quality_gate_hard_exclude"
        elif _is_soft_exclude(row, evidence):
            stage, reason = "excluded", "quality_gate_soft_exclude"
        elif included:
            stage, reason = "quality_enrichment", "adds_headroom_or_hy3_hard_diversity"
        else:
            cap = FAMILY_SOFT_CAP.get(row["family"])
            family_n = family_counts[row["family"]]
            if cap is not None and family_n >= cap:
                stage, reason = "excluded", "family_soft_cap"
            elif len(selected) >= maximum:
                stage, reason = "excluded", "preferred_budget_reached"
            else:
                stage, reason = "excluded", "coverage_already_represented"
        token_set = tokens[scenario_id]
        covered_by = sorted(
            {owners[token] for token in token_set if token in owners}
        )
        audit_rows.append(
            {
                "scenario_id": scenario_id,
                "included": included,
                "selection_stage": stage,
                "reason": reason,
                "feature_ids": sorted(
                    feature_ids[token] for token in token_set if token in feature_ids
                ),
                "new_feature_ids": sorted(
                    feature_ids[token] for token in gains.get(scenario_id, set())
                ),
                "new_source_support_feature_ids": [],
                "covered_by": covered_by,
                "quality_rank": round(_quality_rank(row, evidence), 3),
                "quality_flags": sorted(_quality_flags(evidence, scenario_id)),
            }
        )
    return sorted(selected, key=lambda row: row["scenario_id"]), {
        "coverage_complete": covered == set(eligible_features),
        "coverage_core_scenario_ids": [
            row["scenario_id"] for row in selected if row["scenario_id"] in gains
        ],
        "n_coverage_core": len(gains),
        "n_restored_excluded_representative": len(restored_ids),
        "n_stratum_completion": len(stratum_ids),
        "n_domain_floor": len(floor_ids),
        "n_must_keep_headroom": len(must_keep_ids),
        "n_quality_enrichment": len(enrichment_ids),
        "n_hard_excluded": len(hard_ids),
        "preferred_size_range": list(preferred_range),
        "budget_satisfied": minimum <= len(selected) <= maximum,
        "feature_universe": "eligible_pool_after_quality_gate",
        "features": [json.loads(feature) for feature in eligible_features],
        "selection_order": [row["scenario_id"] for row in selected],
        "rows": audit_rows,
    }


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def build_payload(core_path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    core_bytes = core_path.read_bytes()
    core = json.loads(core_bytes)
    evidence_path = _quality_evidence_path(core_path)
    evidence_bytes = evidence_path.read_bytes()
    evidence = load_quality_evidence(core_path)
    selected, audit = _select_quality_gated_v6(
        list(core["scenarios"]),
        repo_root=repo_root,
        evidence=evidence,
    )
    audit = {
        **audit,
        "quality_evidence_sha256": _sha256_bytes(evidence_bytes),
        "quality_evidence_path": QUALITY_EVIDENCE_NAME,
    }
    payload = {
        "schema_version": "operate-lite-suite-v1",
        "suite_id": "operate_lite",
        "track": "efficiency_development",
        "formal_full_leaderboard_eligible": False,
        "parent_release_id": core["release_id"],
        "parent_core_suite_sha256": _sha256_bytes(core_bytes),
        "selection_algorithm": QUALITY_ALGORITHM,
        "selection_policy": {
            "eligibility": (
                "exact Core-locked rows with complete identities and verified YAML hashes; "
                "keep only hardness-eligible rows: open-headroom, all-models-zero with "
                "open oracle, Hy3-strict [5, 35), unresolved Hy3-zero, greedy-strong/"
                "LLM-lag, or discriminative rows with min=0 and mean<40; drop "
                "saturated/ceiling, unrankable dead cells, LLM-easy means>=50, "
                "Hy3>=50 unless models still lag, score plateaus, and horizon>=600 "
                "unless keep-headroom still lags. Families or domains with no hard "
                "row are omitted; unique-family hostages are not restored"
            ),
            "inclusion": (
                "coverage-close the hardness-eligible pool, complete eligible "
                "backend/family/difficulty/horizon strata, hit domain floors from "
                "the hard pool, keep open-headroom / greedy-peer-with-LLM-lag cells, "
                "then enrich Hy3-hard rows under family caps"
            ),
            "model_outcomes_used_for_selection": True,
            "scoring_version": evidence.get("scoring_version", "0.17.0"),
            "priority": [
                "most_uncovered_eligible_features",
                "quality_rank",
                "new_physical_source",
                "shorter_horizon",
                "scenario_id_ascending",
            ],
            "coverage_axes": [
                "domain_backend_family_mode_difficulty_horizon_cell",
                "source_dataset",
                "typed_event_hiddenness_profile",
                "native_control_and_task_contract_profile",
                "positive_native_size_shape",
                "driving_hazard_and_preventive_inspection_deadlines",
                "microgrid_site_forecast_bias_sigma_genset_availability",
                "building_dataset_native_event_kind_channel_hiddenness",
                "power_network_feeder_decision_axis",
                "traffic_controllable_topology",
            ],
            "native_size_fields": list(SCALE_FIELDS),
            "horizon_buckets": [label for _, _, label in HORIZON_BUCKETS],
            "preferred_size_range": list(QUALITY_PREFERRED_RANGE),
            "hard_exclude_flags": sorted(HARD_EXCLUDE_FLAGS),
            "soft_exclude_flags": sorted(SOFT_EXCLUDE_FLAGS),
            "domain_floor": dict(DOMAIN_FLOOR),
            "family_soft_cap": dict(FAMILY_SOFT_CAP),
            "budget_interpretation": (
                "upper bound limits quality enrichment and never removes required "
                "eligible coverage, stratum completion, or domain floors; lower bound "
                "is not a size target; Lite may omit a Core family or domain that has "
                "no hardness-eligible row"
            ),
            "quality_ranking": (
                "CPU wait/greedy/oracle 0.17, four-model Lite 0.17, and Hy3 Full "
                "ok scores; hardness-eligible rows rank above saturated, dead, "
                "plateau, LLM-easy, or Hy3-easy cells"
            ),
            "domain_or_stratum_quota": {
                "domain_floor": dict(DOMAIN_FLOOR),
                "stratum_complete": [
                    "backend_kind",
                    "family",
                    "difficulty_level",
                    "horizon_bucket",
                ],
            },
            "scope": (
                "hardness-filtered development coverage, not a statistical sample "
                "or Full leaderboard denominator"
            ),
        },
        "n_scenarios": len(selected),
        "n_physical_sources": len({row["physical_source_key"] for row in selected}),
        "n_semantic_fingerprints": len(
            {row["semantic_fingerprint"] for row in selected}
        ),
        "n_structural_fingerprints": len(
            {row["structural_fingerprint"] for row in selected}
        ),
        "total_horizon_ticks": sum(int(row["horizon_ticks"]) for row in selected),
        "by_domain": _counts(selected, "domain"),
        "by_backend": _counts(selected, "backend_kind"),
        "by_family": _counts(selected, "family"),
        "by_difficulty": _counts(selected, "difficulty_level"),
        "horizon_bucket_counts": dict(
            sorted(
                Counter(
                    _horizon_bucket(int(row["horizon_ticks"])) for row in selected
                ).items()
            )
        ),
        "selection_audit": audit,
        "scenarios": selected,
    }
    payload["suite_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--core-suite",
        type=Path,
        default=Path("release/operate_v0_62_0/core_suite.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("release/operate_v0_62_0/lite_suite.json")
    )
    args = parser.parse_args()
    payload = build_payload(args.core_suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.output}: {payload['n_scenarios']} rows, {payload['n_physical_sources']} physical sources"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
