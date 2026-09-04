#!/usr/bin/env python3
"""Build a deterministic catalog for independently mined driving bundles.

The catalog is intentionally a gate report, not a Core release switch.  It
proves that each bundle is source-bound and that selected windows do not
overlap; native replay, safety headroom, and licensing remain explicit
admission blockers until their reports are attached.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _bundle_record(bundle: Path) -> dict[str, Any]:
    from domains.autonomous_driving.data.contracts import (
        NGSIM_LICENSE_ID,
        NGSIM_LICENSE_URL,
        NGSIM_PORTAL_LICENSE_ID,
        NGSIM_PORTAL_LICENSE_URL,
    )
    from domains.autonomous_driving.data.ngsim import verify_bundle

    verify_bundle(bundle)
    manifest = _load(bundle / "bundle.json")
    fixture = _load(bundle / "runtime/fixture.json")
    derivation = dict(fixture.get("derivation") or {})
    candidate_id = str(derivation.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("autonomous_driving_catalog_candidate_id_missing")
    if str(manifest.get("selected_candidate_id") or "") not in {"", candidate_id}:
        raise ValueError("autonomous_driving_catalog_manifest_candidate_mismatch")
    mining = _load(bundle / "mining/candidates.json")
    source_lock = _load(bundle / "source/source.lock.json")
    source_plan = dict(source_lock.get("source_plan") or {})
    candidate = next(
        (
            dict(value)
            for value in mining.get("candidates") or []
            if str(value.get("candidate_id") or "") == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError("autonomous_driving_catalog_candidate_not_in_mining_report")
    context = dict(candidate.get("hazard_context") or {})
    if not bool(context.get("phase_window_complete")):
        raise ValueError("autonomous_driving_catalog_requires_phase_complete_candidate")
    source_events = [
        dict(value) for value in fixture.get("source_events") or [] if isinstance(value, dict)
    ]
    if not source_events:
        raise ValueError("autonomous_driving_catalog_source_events_missing")
    start_ms = int(candidate["start_time_ms"])
    end_ms = int(candidate["end_time_ms_exclusive"])
    if end_ms <= start_ms:
        raise ValueError("autonomous_driving_catalog_window_invalid")
    return {
        "bundle_path": _relative(bundle),
        "bundle_id": manifest.get("bundle_id"),
        "candidate_id": candidate_id,
        "source_window_sha256": candidate.get("source_window_sha256"),
        "start_time_ms": start_ms,
        "end_time_ms_exclusive": end_ms,
        "ego_actor_id": context.get("ego_actor_id"),
        "conflict_actor_id": context.get("conflict_actor_id"),
        "hazard_kind": context.get("hazard_kind"),
        "recording_id": source_plan.get("recording_id") or source_plan.get("location"),
        "license_review_status": manifest.get("license_review_status")
        or "pending_metadata_discrepancy",
        "license_declarations": manifest.get("license_declarations")
        or {
            "dataset_level": {
                "id": NGSIM_PORTAL_LICENSE_ID,
                "terms_url": NGSIM_PORTAL_LICENSE_URL,
            },
            "common_core_custom_field": {
                "id": NGSIM_LICENSE_ID,
                "terms_url": NGSIM_LICENSE_URL,
            },
        },
        "source_event_count": len(source_events),
        "source_event_chain_sha256": (manifest.get("evidence") or {}).get(
            "runtime_source_events_sha256"
        ),
        "response_windows_ms": {
            "supervisory_prevention": int(context.get("supervisory_prevention_window_ms") or 0),
            "protective_response": int(context.get("protective_response_window_ms") or 0),
            "recovery": int(context.get("recovery_window_ms") or 0),
        },
    }


def _manifest_bundle_record(bundle: Path) -> dict[str, Any]:
    """Read a bundle's locked identity without hashing its large SQLite file.

    This mode is only for structural staging audits.  It still requires the
    immutable manifest, checksum list, candidate binding, source events, and
    evidence digests; the Core readiness auditor keeps full content and native
    replay as separate gates.
    """
    from domains.autonomous_driving.data.contracts import (
        NGSIM_LICENSE_ID,
        NGSIM_LICENSE_URL,
        NGSIM_PORTAL_LICENSE_ID,
        NGSIM_PORTAL_LICENSE_URL,
    )
    from scripts.audit_autonomous_driving_core_inventory import (
        _manifest_bundle_record as read_manifest_record,
    )

    row = dict(read_manifest_record(bundle))
    manifest = _load(bundle / "bundle.json")
    row.update(
        {
            "bundle_id": manifest.get("bundle_id"),
            "license_review_status": manifest.get("license_review_status")
            or "pending_metadata_discrepancy",
            "license_declarations": manifest.get("license_declarations")
            or {
                "dataset_level": {
                    "id": NGSIM_PORTAL_LICENSE_ID,
                    "terms_url": NGSIM_PORTAL_LICENSE_URL,
                },
                "common_core_custom_field": {
                    "id": NGSIM_LICENSE_ID,
                    "terms_url": NGSIM_LICENSE_URL,
                },
            },
        }
    )
    return row


def build_catalog(
    bundles_root: Path | Iterable[Path],
    output: Path | None = None,
    *,
    candidate_ids: set[str] | None = None,
    manifest_only: bool = False,
) -> dict[str, Any]:
    roots = [bundles_root] if isinstance(bundles_root, Path) else list(bundles_root)
    all_bundles = sorted(
        {
            path.resolve()
            for root in roots
            for path in root.iterdir()
            if path.is_dir() and (path / "bundle.json").is_file()
        }
    )
    # Candidate filtering happens before verify_bundle.  Verification reads
    # and re-hashes the normalized SQLite window; doing that for every bundle
    # in a large staging root made a four-row Core slice needlessly scan
    # gigabytes of unrelated artifacts.
    if candidate_ids is None:
        bundles = all_bundles
    else:
        selected_paths: list[Path] = []
        fixture_seen = False
        for path in all_bundles:
            fixture_path = path / "runtime/fixture.json"
            if not fixture_path.is_file():
                continue
            fixture_seen = True
            try:
                fixture = _load(fixture_path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            candidate_id = str((fixture.get("derivation") or {}).get("candidate_id") or "")
            if candidate_id in candidate_ids:
                selected_paths.append(path)
        # Test doubles and legacy diagnostic roots may not have a fixture;
        # retain the old path in that case so the injected bundle reader can
        # still enforce its own candidate filter.
        bundles = selected_paths if fixture_seen else all_bundles
    if not bundles:
        if candidate_ids is not None:
            raise ValueError("autonomous_driving_catalog_candidate_filter_unknown")
        raise ValueError("autonomous_driving_catalog_no_bundles")
    record = _manifest_bundle_record if manifest_only else _bundle_record
    rows = [record(path) for path in bundles]
    if candidate_ids is not None:
        unknown = candidate_ids - {str(row.get("candidate_id") or "") for row in rows}
        if unknown:
            raise ValueError("autonomous_driving_catalog_candidate_filter_unknown")
        rows = [row for row in rows if str(row.get("candidate_id") or "") in candidate_ids]
        if not rows:
            raise ValueError("autonomous_driving_catalog_candidate_filter_empty")
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("recording_id") or ""),
            row["start_time_ms"],
            row["candidate_id"],
        ),
    )
    overlaps: list[dict[str, Any]] = []
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if (
            str(current.get("recording_id") or "")
            == str(previous.get("recording_id") or "")
            and int(current["start_time_ms"])
            < int(previous["end_time_ms_exclusive"])
        ):
            overlaps.append(
                {
                    "recording_id": str(current.get("recording_id") or ""),
                    "previous_candidate_id": previous["candidate_id"],
                    "candidate_id": current["candidate_id"],
                }
            )
    unique_windows = len({str(row["source_window_sha256"]) for row in rows})
    unique_hazard_identities = len(
        {
            (
                row["source_window_sha256"],
                row["ego_actor_id"],
                row["conflict_actor_id"],
                row["hazard_kind"],
            )
            for row in rows
        }
    )
    unique_recordings = len(
        {str(row.get("recording_id") or "") for row in rows if str(row.get("recording_id") or "")}
    )
    license_review = (
        "approved"
        if rows
        and all(str(row.get("license_review_status") or "") == "approved" for row in rows)
        else "pending_metadata_discrepancy"
    )
    denominator_reasons = [
        "native_reactive_runtime_validation_pending",
        "reactive_closed_loop_headroom_pending",
        "shield_only_vs_agent_headroom_pending",
    ]
    if license_review != "approved":
        denominator_reasons.append(
            "license_review_pending_cc_by_sa_3_vs_4_metadata_discrepancy"
        )
    report: dict[str, Any] = {
        "schema_version": "autonomous_driving_candidate_catalog_v1",
        "status": "held",
        "bundle_count": len(rows),
        "candidate_count": len(rows),
        "selection": {
            "candidate_ids": sorted(candidate_ids) if candidate_ids is not None else None,
            "selection_kind": "explicit_candidate_filter"
            if candidate_ids is not None
            else "all_bundles",
        },
        "verification_mode": "manifest_only" if manifest_only else "full_bundle",
        "bundles": rows,
        "structural_dedup": {
            "unique_source_windows": unique_windows,
            "unique_hazard_identities": unique_hazard_identities,
            "unique_recordings": unique_recordings,
            "overlap_count": len(overlaps),
            "overlaps": overlaps,
            "non_overlapping_windows": not overlaps,
        },
        "core_denominator_eligible": False,
        "core_denominator_reason": denominator_reasons,
        "admission": {
            "source_bundle_verification": "manifest_only" if manifest_only else "passed",
            "phase_complete_windows": "manifest_only" if manifest_only else "passed",
            "source_event_materiality": "manifest_only" if manifest_only else "passed",
            "window_overlap_gate": "passed" if not overlaps else "failed",
            "license_review": license_review,
            "formal_core_allowed": False,
        },
    }
    if manifest_only:
        report["core_denominator_reason"].insert(0, "full_bundle_verification_pending")
    if output is not None:
        if output.exists():
            raise FileExistsError("autonomous_driving_catalog_output_exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundles-root",
        type=Path,
        action="append",
        required=True,
        help="Bundle directory; repeat for independently locked recordings/sites.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Read locked manifests without hashing normalized SQLite payloads; staging only.",
    )
    parser.add_argument(
        "--candidate-id",
        action="append",
        dest="candidate_ids",
        help="Restrict the catalog to an exact candidate; repeat for a Core slice.",
    )
    args = parser.parse_args(argv)
    roots = [value if value.is_absolute() else REPO_ROOT / value for value in args.bundles_root]
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    try:
        if args.check:
            if not output.is_file():
                raise ValueError("autonomous_driving_catalog_report_missing")
            report = build_catalog(
                [root.resolve() for root in roots],
                None,
                candidate_ids=set(args.candidate_ids) if args.candidate_ids else None,
                manifest_only=args.manifest_only,
            )
            existing = _load(output)
            ok = existing == report
        else:
            report = build_catalog(
                [root.resolve() for root in roots],
                output.resolve(),
                candidate_ids=set(args.candidate_ids) if args.candidate_ids else None,
                manifest_only=args.manifest_only,
            )
            ok = True
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, indent=2))
        return 1
    print(json.dumps({"status": "verified" if ok else "stale", "bundles": report["bundle_count"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
