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
    audit_rows = []
    for row in sorted(rows, key=lambda row: row["scenario_id"]):
        scenario_id = row["scenario_id"]
        in_core = scenario_id in gains
        included = in_core or scenario_id in enrichment
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
                else ("diversity_enrichment" if included else "excluded"),
                "reason": "adds_coverage"
                if in_core
                else (
                    "adds_independent_source_support"
                    if included
                    else (
                        "preferred_budget_reached"
                        if can_add_support
                        else "coverage_already_represented"
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


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def build_payload(core_path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    core_bytes = core_path.read_bytes()
    core = json.loads(core_bytes)
    selected, audit = _select_with_audit(list(core["scenarios"]), repo_root=repo_root)
    payload = {
        "schema_version": "operate-lite-suite-v1",
        "suite_id": "operate_lite",
        "track": "efficiency_development",
        "formal_full_leaderboard_eligible": False,
        "parent_release_id": core["release_id"],
        "parent_core_suite_sha256": _sha256_bytes(core_bytes),
        "selection_algorithm": "quality_gated_native_coverage_enrichment_v3",
        "selection_policy": {
            "eligibility": "exact Core-locked rows with complete identities and verified YAML hashes",
            "inclusion": (
                "preserve coverage closure, then add distinct physical-source support "
                "in complete support rounds until the preferred size range is reached"
            ),
            "priority": [
                "most_uncovered_features",
                "new_physical_source",
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
            "preferred_size_range": list(PREFERRED_RANGE),
            "budget_interpretation": (
                "user-selected development cost/diversity tradeoff, not a quality "
                "gate; the upper bound limits enrichment, never required coverage"
            ),
            "enrichment_priority": [
                "most_feature_source_support_deficits_filled",
                "new_physical_source",
                "scenario_id_ascending",
            ],
            "quality_ranking": "none; every eligible row has passed the same Core admission",
            "domain_or_stratum_quota": None,
            "scope": (
                "development coverage, not a statistical sample or a mathematical "
                "minimum; Full retains all admitted source/window variation"
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
        default=Path("release/operate_v0_61_0/core_suite.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("release/operate_v0_61_0/lite_suite.json")
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
