#!/usr/bin/env python3
"""Fail closed on insufficient cross-domain release coverage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import (  # noqa: E402
    artifact_binding,
    extract_semantics,
    report_rows,
    required_semantics,
)
from core.protocol21_qualification_cohort import (  # noqa: E402
    REQUIRED_FORMAL_DOMAINS,
)
from core.source_asset_contract import (  # noqa: E402
    canonical_physical_source_asset_key,
)

RELEASE_COVERAGE_SCHEMA_VERSION = "1.0"
RELEASE_COVERAGE_POLICY_VERSION = "formal_release_coverage_v1"
FORMAL_RELEASE_COVERAGE_POLICY: dict[str, Any] = {
    "required_domains": list(REQUIRED_FORMAL_DOMAINS),
    "required_difficulty_levels": ["basic", "medium", "high", "extreme"],
    "min_rows_per_domain": 3,
    "min_effective_sources_per_domain": 3,
    "min_physical_sources_per_domain": 2,
    "min_difficulty_levels_per_domain": 2,
    "min_domains_with_high_or_extreme": 3,
}


def _present(value: Any) -> bool:
    return value not in (None, "", {}, [])


def _physical_source_lock(row: Mapping[str, Any]) -> str | None:
    ledger = row.get("case_ledger")
    ledger = ledger if isinstance(ledger, Mapping) else {}
    value = next(
        (
            candidate
            for candidate in (
                ledger.get("physical_source_lock"),
                row.get("physical_source_key"),
                ledger.get("physical_source_key"),
            )
            if _present(candidate)
        ),
        None,
    )
    if value is None:
        return None
    return canonical_physical_source_asset_key(value)


def _policy_value(
    policy: Mapping[str, Any],
    key: str,
) -> int:
    value = policy.get(key)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"release coverage policy {key} must be a positive integer")
    return value


def evaluate_formal_release_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] = FORMAL_RELEASE_COVERAGE_POLICY,
) -> dict[str, Any]:
    """Evaluate cross-domain coverage independently of score aggregation."""
    required_domains = tuple(str(value) for value in policy["required_domains"])
    required_levels = tuple(
        str(value) for value in policy["required_difficulty_levels"]
    )
    min_rows = _policy_value(policy, "min_rows_per_domain")
    min_effective = _policy_value(policy, "min_effective_sources_per_domain")
    min_physical = _policy_value(policy, "min_physical_sources_per_domain")
    min_levels = _policy_value(policy, "min_difficulty_levels_per_domain")
    min_adaptive_domains = _policy_value(
        policy, "min_domains_with_high_or_extreme"
    )
    if not required_domains or not required_levels:
        raise ValueError("release coverage policy requires domains and difficulty levels")

    by_domain_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        domain = str(row.get("domain") or "")
        if domain:
            by_domain_rows[domain].append(row)

    by_domain: dict[str, dict[str, Any]] = {}
    for domain in sorted(set(required_domains) | set(by_domain_rows)):
        domain_rows = by_domain_rows.get(domain, [])
        effective_keys = {
            str(row.get("source_denominator_key") or "")
            for row in domain_rows
            if _present(row.get("source_denominator_key"))
        }
        physical_locks = {
            lock
            for row in domain_rows
            if (lock := _physical_source_lock(row)) is not None
        }
        difficulties = {
            str(row.get("difficulty_level") or "")
            for row in domain_rows
            if _present(row.get("difficulty_level"))
        }
        families = {
            str(row.get("family") or "")
            for row in domain_rows
            if _present(row.get("family"))
        }
        backends = {
            str(row.get("backend_kind") or "")
            for row in domain_rows
            if _present(row.get("backend_kind"))
        }
        reasons: list[str] = []
        if domain in required_domains and not domain_rows:
            reasons.append("required_domain_missing")
        if len(domain_rows) < min_rows:
            reasons.append("min_rows_per_domain_not_met")
        if len(effective_keys) < min_effective:
            reasons.append("min_effective_sources_per_domain_not_met")
        if len(physical_locks) < min_physical:
            reasons.append("min_physical_sources_per_domain_not_met")
        if len(difficulties) < min_levels:
            reasons.append("min_difficulty_levels_per_domain_not_met")
        by_domain[domain] = {
            "n_rows": len(domain_rows),
            "n_effective_sources": len(effective_keys),
            "n_physical_sources": len(physical_locks),
            "n_difficulty_levels": len(difficulties),
            "difficulty_levels": sorted(difficulties),
            "n_families": len(families),
            "families": sorted(families),
            "n_backends": len(backends),
            "backends": sorted(backends),
            "has_high_or_extreme": bool(
                difficulties.intersection({"high", "extreme"})
            ),
            "reason_codes": reasons,
        }

    present_domains = set(by_domain_rows)
    observed_levels = {
        str(row.get("difficulty_level") or "")
        for row in rows
        if str(row.get("domain") or "") in required_domains
        and _present(row.get("difficulty_level"))
    }
    missing_levels = sorted(set(required_levels) - observed_levels)
    adaptive_domains = sorted(
        domain
        for domain, summary in by_domain.items()
        if domain in required_domains and summary["has_high_or_extreme"]
    )
    blockers = {
        f"missing_required_domain:{domain}"
        for domain in required_domains
        if domain not in present_domains
    }
    blockers.update(
        f"required_difficulty_level_missing:{level}" for level in missing_levels
    )
    if len(adaptive_domains) < min_adaptive_domains:
        blockers.add("min_domains_with_high_or_extreme_not_met")
    blockers.update(
        f"domain_coverage_failed:{domain}"
        for domain, summary in by_domain.items()
        if domain in required_domains and summary["reason_codes"]
    )
    return {
        "schema_version": RELEASE_COVERAGE_SCHEMA_VERSION,
        "status": "complete",
        "release_coverage_policy_version": RELEASE_COVERAGE_POLICY_VERSION,
        "policy": dict(policy),
        "n_rows": len(rows),
        "required_domains": list(required_domains),
        "observed_domains": sorted(present_domains),
        "required_difficulty_levels": list(required_levels),
        "observed_difficulty_levels": sorted(observed_levels),
        "missing_difficulty_levels": missing_levels,
        "domains_with_high_or_extreme": adaptive_domains,
        "n_domains_with_high_or_extreme": len(adaptive_domains),
        "by_domain": by_domain,
        "release_coverage_blockers": sorted(blockers),
        "release_coverage_passed": not blockers,
    }


def build_release_coverage(
    *,
    core: Mapping[str, Any],
    core_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Bind a coverage decision to the exact materialized Core artifact."""
    rows = report_rows(dict(core))
    report = evaluate_formal_release_coverage(rows)
    tree = implementation_identity(repo_root)["implementation_tree_sha256"]
    input_blockers: list[str] = []
    if core.get("status") != "protocol21_core_candidate":
        input_blockers.append("core_status_invalid")
    if extract_semantics(dict(core)) != required_semantics():
        input_blockers.append("core_semantics_stale")
    if core.get("implementation_tree_sha256") != tree:
        input_blockers.append("core_implementation_tree_mismatch")
    blockers = sorted(
        set(report["release_coverage_blockers"]) | set(input_blockers)
    )
    report.update(
        {
            "evaluation_semantics": required_semantics(),
            "implementation_tree_sha256": tree,
            "n_expected": len(rows),
            "n_completed": len(rows),
            "input_bindings": {"core": artifact_binding(core_path)},
            "input_blockers": input_blockers,
            "release_coverage_blockers": blockers,
            "release_coverage_passed": not blockers,
        }
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        core = json.loads(args.core.read_text(encoding="utf-8"))
        report = build_release_coverage(core=core, core_path=args.core)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "release_coverage_passed": report[
                    "release_coverage_passed"
                ],
                "release_coverage_blockers": report[
                    "release_coverage_blockers"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
