#!/usr/bin/env python3
"""Reconstruct US-101 OPERATE source windows from the extracted official txt.

Dry-run (default): print a JSON plan of the seven mature US-101 candidates plus the
next commands needed to proceed.  Writes ``<work-root>/dry_run_plan.json``
the first time it is run.

Execute (--execute, only when --raw exists):
  1. verify_ngsim_csv against the core plan
  2. create_source_lock  → <work-root>/source.lock.json
  3. normalize_csv       → <work-root>/trajectories.sqlite3  +
                           <work-root>/normalization.lock.json
  4. For every mature US-101 candidate, compute _window_semantic_sha256
     against the normalised database and compare to the historical hash.
  5. Write <work-root>/window_recovery_report.json

If --materialize-candidate ID is also supplied (requires a mining report
under <work-root>/mining/candidates.json), one portable bundle is built
via materialize_bundle and fails closed if the mining report lacks that ID.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.autonomous_driving.data.canonical_recovery import (  # noqa: E402
    CANONICAL_CANDIDATES,
    NGSIMCanonicalCandidate,
)
from domains.autonomous_driving.data.contracts import (  # noqa: E402
    build_ngsim_core_plan,
    write_json_exclusive,
)
from domains.autonomous_driving.data.ngsim import (  # noqa: E402
    MINING_RECIPE_VERSION,
    _window_semantic_sha256,
    create_source_lock,
    materialize_bundle,
    normalize_csv,
    verify_ngsim_csv,
)

REPORT_SCHEMA_VERSION = "ngsim_us101_window_recovery_v1"

MATURE_US101_CANDIDATE_IDS = (
    "ngsim:1118846999200:e08996e3fddf68d4",
    "ngsim:1118847070800:abc2ea840a747b78",
    "ngsim:1118847187100:3b6793cb928cf7fd",
    "ngsim:1118847260700:5aea66c4b9a5ba06",
    "ngsim:1118847482500:626fc1b70a91943d",
    "ngsim:1118847616700:5cf1d4d7a4c571a4",
    "ngsim:1118847677100:adc3ed02f831ff5e",
)
US101_EXCLUSIONS = {
    "ngsim:1118847062700:364428e7e12e2fee": "metadata_recipe_incomplete",
    "ngsim:1118847132300:fc9b160cb3ccb957": "metadata_recipe_incomplete",
    "ngsim:1118847360400:99e4d9e9718737e1": "positive_headroom_not_established",
    "ngsim:1118847551400:ccdc6d3703d5ad43": "positive_headroom_not_established",
}

_DEFAULT_RAW = (
    REPO_ROOT
    / "works/autonomous_driving/ngsim/recovery/extracted/us-101"
    / "trajectories-0750am-0805am.txt"
)
_DEFAULT_WORK_ROOT = REPO_ROOT / "works/autonomous_driving/ngsim/recovery/us101"


# ---------------------------------------------------------------------------
# Public helpers (used by tests)
# ---------------------------------------------------------------------------


def us101_candidates() -> tuple[NGSIMCanonicalCandidate, ...]:
    """Return the seven mature US-101 candidates selected for recovery."""
    by_id = {
        candidate.candidate_id: candidate
        for candidate in CANONICAL_CANDIDATES
        if candidate.recording_id == "us-101"
    }
    missing = set(MATURE_US101_CANDIDATE_IDS) - set(by_id)
    if missing:
        raise ValueError(f"mature_us101_candidates_missing:{','.join(sorted(missing))}")
    return tuple(by_id[candidate_id] for candidate_id in MATURE_US101_CANDIDATE_IDS)


def excluded_us101_candidates() -> list[dict[str, str]]:
    """Record why the other canonical US-101 windows are not recovered."""
    return [
        {"candidate_id": candidate_id, "disposition": reason}
        for candidate_id, reason in sorted(US101_EXCLUSIONS.items())
    ]


def window_start_ms(candidate: NGSIMCanonicalCandidate) -> int:
    """Return the mining-window start encoded in ``candidate_id``.

    ``event_time_ms`` is the hazard timestamp and can sit inside the window,
    not at its start.  Historical ``source_window_sha256`` values were
    computed over ``[start_ms, window_end)``.
    """
    parts = candidate.candidate_id.split(":")
    if len(parts) != 3 or parts[0] != "ngsim":
        raise ValueError(f"ngsim_candidate_id_invalid:{candidate.candidate_id}")
    return int(parts[1])


def candidate_plan_entry(candidate: NGSIMCanonicalCandidate) -> dict[str, Any]:
    """Serialise one candidate to the dry-run plan row schema."""
    return {
        "candidate_id": candidate.candidate_id,
        "recording_id": candidate.recording_id,
        "hazard_kind": candidate.hazard_kind,
        "ego_actor_id": candidate.ego_actor_id,
        "start_time_ms": window_start_ms(candidate),
        "event_time_ms": candidate.event_time_ms,
        "window_end_time_ms_exclusive": candidate.window_end_time_ms_exclusive,
        "recipe_complete": candidate.metadata_recipe_complete,
        "historical_recipe_version": candidate.recipe_version,
        "source_window_sha256": candidate.source_window_sha256,
    }


def build_dry_run_plan(
    candidates: tuple[NGSIMCanonicalCandidate, ...],
    *,
    work_root: Path,
    raw: Path,
) -> dict[str, Any]:
    """Build the dry-run plan dict (pure, no I/O)."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": "dry_run",
        "formal_core_allowed": False,
        "license_review_status": "approved",
        "license_review_basis": "operator_attestation_2026-09-02",
        "candidate_count": len(candidates),
        "recipe_complete_count": sum(c.metadata_recipe_complete for c in candidates),
        "candidates": [candidate_plan_entry(c) for c in candidates],
        "excluded_candidates": excluded_us101_candidates(),
        "next_commands": [
            (
                f"python scripts/recover_ngsim_us101_windows.py"
                f" --raw {raw}"
                f" --work-root {work_root}"
                f" --execute"
            )
        ],
    }


# ---------------------------------------------------------------------------
# Execute-path helpers
# ---------------------------------------------------------------------------


def _compute_window_entry(
    connection: sqlite3.Connection,
    candidate: NGSIMCanonicalCandidate,
) -> dict[str, Any]:
    """Compare reconstructed window hash against historical hash."""
    start_ms = window_start_ms(candidate)
    computed_sha256 = _window_semantic_sha256(
        connection,
        start_ms,
        candidate.window_end_time_ms_exclusive,
    )
    sha_match = computed_sha256 == candidate.source_window_sha256
    blockers: list[str] = []
    entry: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "start_time_ms": start_ms,
        "event_time_ms": candidate.event_time_ms,
        "window_end_time_ms_exclusive": candidate.window_end_time_ms_exclusive,
        "historical_source_window_sha256": candidate.source_window_sha256,
        "computed_source_window_sha256": computed_sha256,
        "source_window_sha256_match": sha_match,
        "metadata_recipe_complete": candidate.metadata_recipe_complete,
        "current_mining_recipe": MINING_RECIPE_VERSION,
        "historical_recipe_version": candidate.recipe_version,
        "formal_core_allowed": False,
        "license_review_status": "approved",
    }
    if not sha_match:
        entry["hash_rebind_required"] = True
        blockers.append("source_window_sha256_mismatch_rebind_required")
    if not candidate.metadata_recipe_complete:
        blockers.append("metadata_recipe_incomplete")
    entry["blockers"] = blockers
    return entry


def run_execute(
    raw: Path,
    work_root: Path,
    candidates: tuple[NGSIMCanonicalCandidate, ...],
) -> dict[str, Any]:
    """Run the full lock → normalise → window-hash pipeline."""
    plan = build_ngsim_core_plan()

    # Step 1: verify raw against the core plan (slow, ~1.2 M rows)
    verification = verify_ngsim_csv(raw, plan=plan)

    work_root.mkdir(parents=True, exist_ok=True)

    # Step 2: source lock (fails if source.lock.json already exists)
    source_lock_path = work_root / "source.lock.json"
    source_lock = create_source_lock(raw, plan, source_lock_path)

    # Step 3: normalise (fails if trajectories.sqlite3 or normalization.lock.json exist)
    database_path = work_root / "trajectories.sqlite3"
    normalization_lock_path = work_root / "normalization.lock.json"
    normalization_lock = normalize_csv(
        raw, source_lock, database_path, normalization_lock_path
    )

    # Step 4: window-hash reconstruction for every US-101 candidate
    window_entries: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as conn:
        for candidate in candidates:
            window_entries.append(_compute_window_entry(conn, candidate))

    hash_match_count = sum(1 for e in window_entries if e["source_window_sha256_match"])
    hash_rebind_count = sum(1 for e in window_entries if e.get("hash_rebind_required", False))

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": "execute",
        "formal_core_allowed": False,
        "license_review_status": "approved",
        "license_review_basis": "operator_attestation_2026-09-02",
        "source_verification": {
            "raw_sha256": verification["raw_sha256"],
            "row_count": verification["row_count"],
            "actor_count": verification["actor_count"],
            "min_timestamp_ms": verification["min_timestamp_ms"],
            "max_timestamp_ms": verification["max_timestamp_ms"],
            "source_evidence_sha256": source_lock["source_evidence_sha256"],
        },
        "normalization": {
            "database_file": database_path.name,
            "normalization_evidence_sha256": normalization_lock[
                "normalization_evidence_sha256"
            ],
        },
        "candidate_count": len(candidates),
        "recipe_complete_count": sum(c.metadata_recipe_complete for c in candidates),
        "hash_match_count": hash_match_count,
        "hash_rebind_count": hash_rebind_count,
        "windows": window_entries,
        "excluded_candidates": excluded_us101_candidates(),
    }

    # Step 5: write report (fail-exclusive)
    report_path = work_root / "window_recovery_report.json"
    write_json_exclusive(report_path, report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=_DEFAULT_RAW,
        metavar="PATH",
        help=(
            "Path to the extracted official trajectories txt "
            "(18-col space-separated, 1_180_598 rows)"
        ),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=_DEFAULT_WORK_ROOT,
        metavar="DIR",
        help="Working directory for lock files, database, and reports",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run lock/normalise pipeline (default: dry-run plan only)",
    )
    parser.add_argument(
        "--materialize-candidate",
        metavar="ID",
        help=(
            "After execute, materialise one bundle for this candidate ID.  "
            "Requires <work-root>/mining/candidates.json."
        ),
    )
    args = parser.parse_args(argv)

    raw: Path = args.raw
    work_root: Path = args.work_root
    candidates = us101_candidates()

    if not args.execute:
        plan = build_dry_run_plan(candidates, work_root=work_root, raw=raw)
        print(json.dumps(plan, indent=2, sort_keys=True))
        plan_path = work_root / "dry_run_plan.json"
        work_root.mkdir(parents=True, exist_ok=True)
        if not plan_path.exists():
            write_json_exclusive(plan_path, plan)
        return 0

    # Execute path — raw must exist
    if not raw.is_file():
        print(
            f"ERROR: raw txt not found: {raw}\n"
            "Re-run with --execute only after extracting the official US-101 txt.",
            file=sys.stderr,
        )
        return 1

    report = run_execute(raw, work_root, candidates)

    if args.materialize_candidate:
        mining_report_path = work_root / "mining" / "candidates.json"
        if not mining_report_path.is_file():
            print(
                f"ERROR: mining report not found at {mining_report_path}.\n"
                "Run the mining step before materialising a candidate bundle.",
                file=sys.stderr,
            )
            return 1
        cand_id = args.materialize_candidate
        output_dir = work_root / "bundles" / cand_id.replace(":", "_")
        materialize_bundle(
            raw_path=raw,
            source_lock_path=work_root / "source.lock.json",
            database_path=work_root / "trajectories.sqlite3",
            normalization_lock_path=work_root / "normalization.lock.json",
            mining_report_path=mining_report_path,
            output_dir=output_dir,
            candidate_id=cand_id,
            license_review_status="approved",
            license_review_basis="operator_attestation_2026-09-02",
        )
        print(f"Bundle written: {output_dir}")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
