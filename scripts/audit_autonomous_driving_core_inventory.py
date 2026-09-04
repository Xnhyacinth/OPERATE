#!/usr/bin/env python3
"""Audit the autonomous-driving source pool without promoting a Core release.

The audit keeps three quantities separate:

* source-window denominator keys (one mined candidate is one key);
* difficulty slices (optional views of the same key, never denominator keys);
* admission-ready candidates (native legs, replay, and one primary scenario);
* provider and source-pool maturity as separate release diagnostics.

Bundle discovery is manifest-only by default so a large NGSIM cache can be
inspected quickly.  ``--full-verify`` opts into the expensive content and
SQLite evidence verification performed by ``verify_bundle``.  The report is
always held and never mutates the registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DIFFICULTIES = ("basic", "medium", "high", "extreme")

# These are explicit release-design targets, not an automatic promotion rule.
# A difficulty slice is deliberately absent from the denominator here.
CORE_TIERS: dict[str, dict[str, int]] = {
    "staging_pilot": {
        "minimum_source_windows": 3,
        "minimum_recordings": 2,
        "minimum_hazard_kinds": 1,
        "minimum_windows_per_hazard": 1,
    },
    "minimal_core": {
        "minimum_source_windows": 12,
        "minimum_recordings": 3,
        "minimum_hazard_kinds": 3,
        "minimum_windows_per_hazard": 3,
    },
    "robust_core": {
        "minimum_source_windows": 24,
        "minimum_recordings": 4,
        "minimum_hazard_kinds": 4,
        "minimum_windows_per_hazard": 4,
    },
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _quantiles(values: Iterable[float]) -> dict[str, float | None]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}

    def percentile(fraction: float) -> float:
        index = (len(ordered) - 1) * fraction
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "min": round(ordered[0], 3),
        "p25": round(percentile(0.25), 3),
        "median": round(percentile(0.5), 3),
        "p75": round(percentile(0.75), 3),
        "max": round(ordered[-1], 3),
    }


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _manifest_bundle_record(bundle: Path) -> dict[str, Any]:
    """Read the source identity without rescanning the raw SQLite payload.

    This is intentionally stricter than simply reading a YAML index, while
    remaining cheap enough to inspect every candidate in a multi-version
    cache.  Full content verification is available through ``--full-verify``.
    """
    manifest = _load(bundle / "bundle.json")
    if manifest.get("schema_version") != "autonomous_driving_source_bundle_v1":
        raise ValueError("autonomous_driving_inventory_bundle_schema_invalid")
    if not (bundle / "checksums.sha256").is_file():
        raise ValueError("autonomous_driving_inventory_checksums_missing")
    fixture = _load(bundle / "runtime/fixture.json")
    derivation = dict(fixture.get("derivation") or {})
    candidate_id = str(derivation.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("autonomous_driving_inventory_candidate_id_missing")
    if str(manifest.get("selected_candidate_id") or "") not in {"", candidate_id}:
        raise ValueError("autonomous_driving_inventory_candidate_binding_mismatch")
    mining = _load(bundle / "mining/candidates.json")
    candidate = next(
        (
            dict(value)
            for value in mining.get("candidates") or []
            if isinstance(value, dict) and str(value.get("candidate_id") or "") == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError("autonomous_driving_inventory_candidate_not_in_mining")
    context = dict(candidate.get("hazard_context") or {})
    if context.get("phase_window_complete") is not True:
        raise ValueError("autonomous_driving_inventory_requires_phase_complete_candidate")
    source_events = [
        value for value in fixture.get("source_events") or [] if isinstance(value, dict)
    ]
    if not source_events:
        raise ValueError("autonomous_driving_inventory_source_events_missing")
    source_lock = _load(bundle / "source/source.lock.json")
    source_plan = dict(source_lock.get("source_plan") or {})
    start = int(candidate.get("start_time_ms") or 0)
    end = int(candidate.get("end_time_ms_exclusive") or 0)
    if end <= start:
        raise ValueError("autonomous_driving_inventory_window_invalid")
    source_window = str(candidate.get("source_window_sha256") or "")
    evidence = dict(manifest.get("evidence") or {})
    source_event_chain = str(
        evidence.get("runtime_source_events_sha256")
        or manifest.get("source_event_chain_sha256")
        or ""
    )
    if not _valid_sha256(source_window) or not _valid_sha256(source_event_chain):
        raise ValueError("autonomous_driving_inventory_evidence_digest_invalid")
    return {
        "bundle_path": _relative(bundle),
        "candidate_id": candidate_id,
        "bundle_id": str(manifest.get("bundle_id") or ""),
        "source_window_sha256": source_window,
        "source_event_chain_sha256": source_event_chain,
        "source_event_count": len(source_events),
        "start_time_ms": start,
        "end_time_ms_exclusive": end,
        "recording_id": str(source_plan.get("recording_id") or source_plan.get("location") or ""),
        "hazard_kind": str(context.get("hazard_kind") or candidate.get("hazard_kind") or ""),
        "ego_actor_id": str(context.get("ego_actor_id") or ""),
        "conflict_actor_id": str(context.get("conflict_actor_id") or ""),
        "response_windows_ms": {
            "supervisory_prevention": int(context.get("supervisory_prevention_window_ms") or 0),
            "protective_response": int(context.get("protective_response_window_ms") or 0),
            "recovery": int(context.get("recovery_window_ms") or 0),
        },
    }


def discover_bundle_rows(
    roots: Iterable[Path], *, full_verify: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Discover one deterministic row per candidate ID.

    Repeated derived directories are common while recipes evolve.  Identical
    copies are collapsed; conflicting copies are reported as hard inventory
    errors instead of silently selecting one.
    """
    paths = sorted(
        {
            path.parent.resolve()
            for root in roots
            for path in root.rglob("bundle.json")
            if path.is_file()
        }
    )
    from domains.autonomous_driving.data.ngsim import verify_bundle

    by_candidate: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    for bundle in paths:
        try:
            row = _manifest_bundle_record(bundle)
            if full_verify:
                verify_bundle(bundle)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append({"path": _relative(bundle), "reason": str(error)})
            continue
        candidate_id = str(row["candidate_id"])
        existing = by_candidate.get(candidate_id)
        if existing is None:
            by_candidate[candidate_id] = row
            continue
        identity = (
            row["source_window_sha256"],
            row["recording_id"],
            row["start_time_ms"],
            row["end_time_ms_exclusive"],
            row["ego_actor_id"],
            row["conflict_actor_id"],
            row["hazard_kind"],
        )
        previous_identity = tuple(
            existing[key]
            for key in (
                "source_window_sha256",
                "recording_id",
                "start_time_ms",
                "end_time_ms_exclusive",
                "ego_actor_id",
                "conflict_actor_id",
                "hazard_kind",
            )
        )
        if identity != previous_identity:
            errors.append(
                {
                    "path": _relative(bundle),
                    "reason": f"candidate_identity_conflict:{candidate_id}",
                }
            )
    return sorted(by_candidate.values(), key=lambda row: str(row["candidate_id"])), errors


def _maximum_non_overlapping(rows: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    by_recording: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_recording[str(row.get("recording_id") or "")].append(row)
    for recording in sorted(by_recording):
        current_end: int | None = None
        ordered = sorted(
            by_recording[recording],
            key=lambda row: (
                int(row["end_time_ms_exclusive"]),
                int(row["start_time_ms"]),
                str(row["candidate_id"]),
            ),
        )
        for row in ordered:
            start = int(row["start_time_ms"])
            if current_end is None or start >= current_end:
                selected.append(str(row["candidate_id"]))
                current_end = int(row["end_time_ms_exclusive"])
    return sorted(selected)


def _row_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    recordings = Counter(str(row.get("recording_id") or "") for row in rows)
    hazards = Counter(str(row.get("hazard_kind") or "") for row in rows)
    durations = [
        (int(row["end_time_ms_exclusive"]) - int(row["start_time_ms"])) / 1000 for row in rows
    ]
    prevention = [int(row["response_windows_ms"]["supervisory_prevention"]) / 1000 for row in rows]
    protection = [int(row["response_windows_ms"]["protective_response"]) / 1000 for row in rows]
    recovery = [int(row["response_windows_ms"]["recovery"]) / 1000 for row in rows]
    windows = len({str(row["source_window_sha256"]) for row in rows})
    return {
        "source_window_count": windows,
        "candidate_count": len(rows),
        "recording_count": len({key for key in recordings if key}),
        "recording_distribution": dict(sorted(recordings.items())),
        "hazard_kind_count": len({key for key in hazards if key}),
        "hazard_distribution": dict(sorted(hazards.items())),
        "duration_seconds": _quantiles(durations),
        "supervisory_prevention_seconds": _quantiles(prevention),
        "protective_response_seconds": _quantiles(protection),
        "recovery_seconds": _quantiles(recovery),
    }


def _tier_assessment(summary: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, target in CORE_TIERS.items():
        source_count = int(summary["source_window_count"])
        recordings = int(summary["recording_count"])
        hazards = int(summary["hazard_kind_count"])
        hazard_counts = summary["hazard_distribution"]
        structural = {
            "source_windows": source_count >= target["minimum_source_windows"],
            "recordings": recordings >= target["minimum_recordings"],
            "hazard_kinds": hazards >= target["minimum_hazard_kinds"],
            "windows_per_hazard": all(
                int(count) >= target["minimum_windows_per_hazard"]
                for count in hazard_counts.values()
            )
            if hazard_counts
            else False,
        }
        result[name] = {
            "target": target,
            "structural_gates": structural,
            "structurally_sufficient": all(structural.values()),
            "disposition": "candidate_pool_sufficient"
            if all(structural.values())
            else "held_for_more_source_diversity",
        }
    return result


def _set_digest(report: dict[str, Any]) -> dict[str, Any]:
    unsigned = dict(report)
    unsigned.pop("inventory_digest_sha256", None)
    report["inventory_digest_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return report


def _suite_candidate_ids(suite_root: Path | None) -> dict[str, tuple[str, ...]] | None:
    """Return candidates and their materialized admission scenario levels.

    Inventory is intentionally independent from the readiness report.  When a
    suite root is supplied, it is the source of the structural difficulty
    gate; native replay and provider evidence remain readiness responsibilities.
    """
    if suite_root is None:
        return None
    eligible: dict[str, tuple[str, ...]] = {}
    for path in sorted(suite_root.rglob("suite_report.json")):
        try:
            report = _load(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if (
            report.get("schema_version")
            not in {
                "autonomous_driving_suite_report_v1",
                "autonomous_driving_suite_report_v2",
            }
            or report.get("status") != "held"
        ):
            continue
        candidate_id = str(report.get("candidate_id") or "")
        feasibility = report.get("difficulty_feasibility")
        slices = report.get("difficulty_slices")
        if not candidate_id or not isinstance(feasibility, dict) or not isinstance(slices, list):
            continue
        primary = report.get("primary_difficulty")
        expected_levels = (str(primary),) if primary is not None else DIFFICULTIES
        if any(level not in DIFFICULTIES for level in expected_levels):
            continue
        if len(slices) != len(expected_levels):
            continue
        if all((feasibility.get(level) or {}).get("status") == "included" for level in expected_levels):
            eligible[candidate_id] = tuple(expected_levels)
    return eligible


def build_inventory(
    rows: list[dict[str, Any]],
    *,
    catalog: dict[str, Any] | None = None,
    readiness: dict[str, Any] | None = None,
    suite_candidate_ids: dict[str, tuple[str, ...]] | set[str] | None = None,
    discovery_errors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic inventory from already verified/parsed rows."""
    rows = sorted(rows, key=lambda row: str(row.get("candidate_id") or ""))
    summary = _row_summary(rows)
    non_overlap_ids = _maximum_non_overlapping(rows)
    catalog_ids = {
        str(row.get("candidate_id") or "")
        for row in (catalog or {}).get("bundles") or []
        if isinstance(row, dict) and str(row.get("candidate_id") or "")
    }
    readiness_rows = {
        str(row.get("candidate_id") or ""): row
        for row in (readiness or {}).get("bundles") or []
        if isinstance(row, dict) and str(row.get("candidate_id") or "")
    }
    stages: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        ready = dict(readiness_rows.get(candidate_id) or {})
        gates = dict(ready.get("gates") or {})
        difficulty_contract_complete = bool(
            candidate_id in suite_candidate_ids
            if suite_candidate_ids is not None
            else gates.get("difficulty_contract")
        )
        difficulty_levels = (
            tuple(suite_candidate_ids.get(candidate_id, ()))
            if isinstance(suite_candidate_ids, dict)
            else (DIFFICULTIES if difficulty_contract_complete else ())
        )
        stages.append(
            {
                "candidate_id": candidate_id,
                "recording_id": row.get("recording_id"),
                "hazard_kind": row.get("hazard_kind"),
                "in_selected_catalog": candidate_id in catalog_ids,
                "native_validation_complete": all(
                    bool(gates.get(name))
                    for name in (
                        "source_event_materiality",
                        "three_leg_presence",
                        "collision_departure_safety",
                        "positive_oracle_headroom",
                        "reactive_closed_loop",
                        "deterministic_replay",
                    )
                ),
                "difficulty_contract_complete": difficulty_contract_complete,
                "difficulty_levels": list(difficulty_levels),
                "actual_llm_complete": bool(gates.get("actual_llm_evaluation")),
                "ready_for_full_admission": bool(ready.get("ready_for_full_admission")),
            }
        )
    selected_rows = [row for row in rows if str(row["candidate_id"]) in catalog_ids]
    selected_summary = _row_summary(selected_rows)
    difficulty_counts: Counter[str] = Counter()
    for row in stages:
        if row["in_selected_catalog"] and row["difficulty_contract_complete"]:
            for level in row["difficulty_levels"]:
                difficulty_counts[level] += 1
    native_ready_count = sum(1 for row in stages if row["ready_for_full_admission"])
    report: dict[str, Any] = {
        "schema_version": "autonomous_driving_core_inventory_v1",
        "status": "held",
        "formal_core_allowed": False,
        "registry_mutation_performed": False,
        "verification_mode": "manifest_only",
        "discovery_errors": sorted(discovery_errors or [], key=lambda row: row["path"]),
        "source_pool": {
            **summary,
            "maximum_non_overlapping_source_windows": len(non_overlap_ids),
            "maximum_non_overlapping_candidate_ids": non_overlap_ids,
        },
        "selected_catalog": {
            **selected_summary,
            "candidate_ids": sorted(catalog_ids),
            "difficulty_slice_count_by_level": dict(sorted(difficulty_counts.items())),
            "difficulty_slices_are_not_denominator_keys": True,
            "source_window_denominator_keys": selected_summary["source_window_count"],
            "difficulty_row_count_if_all_four_slices": selected_summary["source_window_count"]
            * len(DIFFICULTIES),
        },
        "candidate_stages": stages,
        "evidence_complete_core_candidate_count": native_ready_count,
        "tier_assessment": _tier_assessment(summary),
        "admission": {
            "source_pool_manifest_identity": "passed" if not discovery_errors else "failed",
            "selected_catalog_present": bool(catalog_ids),
            "selected_catalog_is_structurally_non_overlapping": bool(
                (catalog or {}).get("structural_dedup", {}).get("non_overlapping_windows")
            ),
            "all_selected_candidates_have_four_difficulties": all(
                int(difficulty_counts[level]) == len(catalog_ids) for level in DIFFICULTIES
            )
            if catalog_ids
            else False,
            "native_and_provider_evidence_complete": native_ready_count == len(catalog_ids)
            and bool(catalog_ids),
        },
        "recommendation": {
            "current_disposition": "staging_only_until_license_runtime_and_provider_llm_gates",
            "minimum_core_target": "minimal_core",
            "do_not_count_difficulty_slices_as_independent_sources": True,
            "add_source_diversity_before_promotion": summary["recording_count"] < 3
            or summary["hazard_kind_count"] < 3,
        },
    }
    return _set_digest(report)


def build_inventory_from_paths(
    bundle_roots: Iterable[Path],
    *,
    catalog_path: Path | None = None,
    readiness_path: Path | None = None,
    suite_path: Path | None = None,
    full_verify: bool = False,
) -> dict[str, Any]:
    rows, errors = discover_bundle_rows(bundle_roots, full_verify=full_verify)
    catalog = _load(catalog_path) if catalog_path is not None else None
    readiness = _load(readiness_path) if readiness_path is not None else None
    report = build_inventory(
        rows,
        catalog=catalog,
        readiness=readiness,
        suite_candidate_ids=_suite_candidate_ids(suite_path),
        discovery_errors=errors,
    )
    report["verification_mode"] = "full_bundle_verify" if full_verify else "manifest_only"
    report["inputs"] = {
        "bundle_roots": [_relative(path) for path in bundle_roots],
        "catalog": {
            "path": _relative(catalog_path),
            "sha256": _sha256(catalog_path),
        }
        if catalog_path is not None
        else None,
        "readiness": {
            "path": _relative(readiness_path),
            "sha256": _sha256(readiness_path),
        }
        if readiness_path is not None
        else None,
        "suite": {
            "path": _relative(suite_path),
            "sha256": _sha256(suite_path / "suite_coverage.json")
            if suite_path is not None and (suite_path / "suite_coverage.json").is_file()
            else None,
        }
        if suite_path is not None
        else None,
    }
    return _set_digest(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles-root", type=Path, action="append", required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--readiness", type=Path)
    parser.add_argument(
        "--suite-dir",
        type=Path,
        help="Read suite_report.json files directly for the structural four-level gate.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-verify", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    roots = [value if value.is_absolute() else REPO_ROOT / value for value in args.bundles_root]
    catalog = (
        (args.catalog if args.catalog.is_absolute() else REPO_ROOT / args.catalog)
        if args.catalog is not None
        else None
    )
    readiness = (
        (args.readiness if args.readiness.is_absolute() else REPO_ROOT / args.readiness)
        if args.readiness is not None
        else None
    )
    suite = (
        (args.suite_dir if args.suite_dir.is_absolute() else REPO_ROOT / args.suite_dir)
        if args.suite_dir is not None
        else None
    )
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    try:
        report = build_inventory_from_paths(
            roots,
            catalog_path=catalog,
            readiness_path=readiness,
            suite_path=suite,
            full_verify=args.full_verify,
        )
        if args.check:
            if not output.is_file():
                raise ValueError("autonomous_driving_inventory_report_missing")
            verified = _load(output) == report
        else:
            if output.exists():
                raise FileExistsError("autonomous_driving_inventory_output_exists")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            verified = True
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": "verified" if verified else "stale",
                "source_windows": report["source_pool"]["source_window_count"],
                "maximum_non_overlapping": report["source_pool"][
                    "maximum_non_overlapping_source_windows"
                ],
                "selected_catalog_windows": report["selected_catalog"][
                    "source_window_denominator_keys"
                ],
                "evidence_complete_core_candidates": report[
                    "evidence_complete_core_candidate_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
