#!/usr/bin/env python3
"""Build the deterministic OPERATE-Lite development suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


OVERREPRESENTATION_MULTIPLIER = 2.0
STRATUM_REPLICATES = 3
STRATUM_FIELDS = ("backend_kind", "family", "difficulty_level")
REQUIRED_CORE_FIELDS = (
    "backend_kind",
    "difficulty_level",
    "domain",
    "family",
    "horizon_ticks",
    "physical_source_key",
    "scenario_id",
    "scenario_signature",
    "semantic_fingerprint",
    "source_denominator_key",
    "structural_fingerprint",
    "yaml_sha256",
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


def _coverage_tokens(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        f"backend:{row['backend_kind']}",
        f"family:{row['family']}",
        f"difficulty:{row['difficulty_level']}",
        f"horizon_bucket:{_horizon_bucket(int(row['horizon_ticks']))}",
    )


def _stratum_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in STRATUM_FIELDS) + (
        _horizon_bucket(int(row["horizon_ticks"])),
    )


def _overrepresented_domains(rows: list[dict[str, Any]]) -> set[str]:
    counts = Counter(str(row["domain"]) for row in rows)
    threshold = OVERREPRESENTATION_MULTIPLIER * median(counts.values())
    return {domain for domain, count in counts.items() if count > threshold}


def _validate_core_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != len({str(row.get("scenario_id")) for row in rows}):
        raise ValueError("Core contains duplicate scenario IDs")
    for row in rows:
        missing = [field for field in REQUIRED_CORE_FIELDS if not row.get(field)]
        if missing:
            raise ValueError(f"{row.get('scenario_id')}: missing {', '.join(missing)}")
        if row.get("status") != "core_locked" or row.get("core_disposition") != "core_locked":
            raise ValueError(f"{row['scenario_id']}: row is not Core-locked")


def _select_group(
    candidates: list[dict[str, Any]],
    quota: int,
    selected: list[dict[str, Any]],
) -> None:
    if quota > len(candidates):
        raise ValueError(f"quota {quota} exceeds candidate count {len(candidates)}")

    frequencies = Counter(
        token for row in candidates for token in _coverage_tokens(row)
    )
    remaining = sorted(candidates, key=lambda row: row["scenario_id"])
    candidate_ids = {row["scenario_id"] for row in candidates}

    while len([row for row in selected if row["scenario_id"] in candidate_ids]) < quota:
        selected_in_group = [
            row for row in selected if row["scenario_id"] in candidate_ids
        ]
        covered_tokens = {
            token for row in selected_in_group for token in _coverage_tokens(row)
        }
        covered_sources = {row["physical_source_key"] for row in selected}
        covered_semantics = {row["semantic_fingerprint"] for row in selected}
        covered_structures = {row["structural_fingerprint"] for row in selected}
        covered_horizons = {int(row["horizon_ticks"]) for row in selected}
        difficulty_counts = Counter(
            row["difficulty_level"] for row in selected_in_group
        )
        horizon_bucket_counts = Counter(
            _horizon_bucket(int(row["horizon_ticks"])) for row in selected_in_group
        )

        def score(row: dict[str, Any]) -> tuple[Any, ...]:
            new_tokens = [
                token for token in _coverage_tokens(row) if token not in covered_tokens
            ]
            return (
                len(new_tokens),
                sum(1.0 / frequencies[token] for token in new_tokens),
                int(row["physical_source_key"] not in covered_sources),
                int(row["semantic_fingerprint"] not in covered_semantics),
                int(row["structural_fingerprint"] not in covered_structures),
                int(int(row["horizon_ticks"]) not in covered_horizons),
                -difficulty_counts[row["difficulty_level"]],
                -horizon_bucket_counts[_horizon_bucket(int(row["horizon_ticks"]))],
                int(row["horizon_ticks"]),
            )

        best_score = max(score(row) for row in remaining)
        best = min(
            (row for row in remaining if score(row) == best_score),
            key=lambda row: row["scenario_id"],
        )
        selected.append(best)
        remaining.remove(best)


def select_lite(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    _validate_core_rows(rows)
    downsampled_domains = _overrepresented_domains(rows)
    selected: list[dict[str, Any]] = []
    for domain in sorted({str(row["domain"]) for row in rows}):
        domain_rows = [row for row in rows if row["domain"] == domain]
        if domain not in downsampled_domains:
            selected.extend(domain_rows)
            continue

        strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in domain_rows:
            strata[_stratum_key(row)].append(row)
        for key in sorted(strata):
            candidates = strata[key]
            _select_group(candidates, min(STRATUM_REPLICATES, len(candidates)), selected)

    return sorted(selected, key=lambda row: row["scenario_id"])


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def build_payload(core_path: Path) -> dict[str, Any]:
    core_bytes = core_path.read_bytes()
    core = json.loads(core_bytes)
    rows = list(core["scenarios"])
    selected = select_lite(rows)
    domain_counts = Counter(str(row["domain"]) for row in rows)
    median_domain_rows = median(domain_counts.values())
    downsampled_domains = sorted(_overrepresented_domains(rows))
    payload = {
        "schema_version": "operate-lite-suite-v1",
        "suite_id": "operate_lite",
        "track": "efficiency_development",
        "formal_full_leaderboard_eligible": False,
        "parent_release_id": core["release_id"],
        "parent_core_suite_sha256": _sha256_bytes(core_bytes),
        "selection_algorithm": "quality_gated_adaptive_stratified_selection_v1",
        "selection_policy": {
            "eligibility": "exact Core-locked rows with complete identity and hash fields",
            "domain_row_policy": "retain_all_rows_except_policy_overrepresented_domains",
            "parameter_interpretation": (
                "Core admission is the quality gate; the 2x threshold and three-row "
                "stratum limit are predeclared engineering compression heuristics, "
                "not quality thresholds or claims of statistical representativeness "
                "or optimal selection"
            ),
            "overrepresentation_rule": "domain_rows > 2 * median_domain_rows",
            "overrepresentation_multiplier": OVERREPRESENTATION_MULTIPLIER,
            "median_domain_rows": median_domain_rows,
            "downsampled_domains": downsampled_domains,
            "stratum_fields": [*STRATUM_FIELDS, "horizon_bucket"],
            "max_rows_per_nonempty_stratum": STRATUM_REPLICATES,
            "horizon_buckets": [label for _, _, label in HORIZON_BUCKETS],
            "uncapped_domains": sorted(
                {str(row["domain"]) for row in rows} - set(downsampled_domains)
            ),
            "within_stratum_priority": [
                "new_physical_source",
                "new_semantic_fingerprint",
                "new_structural_fingerprint",
                "new_exact_horizon",
                "longer_horizon",
            ],
            "tie_breaker": "scenario_id_ascending",
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
            sorted(Counter(_horizon_bucket(int(row["horizon_ticks"])) for row in selected).items())
        ),
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
        "--output",
        type=Path,
        default=Path("release/operate_v0_61_0/lite_suite.json"),
    )
    args = parser.parse_args()
    payload = build_payload(args.core_suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {args.output}: {payload['n_scenarios']} rows, "
        f"{payload['n_physical_sources']} physical sources"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
