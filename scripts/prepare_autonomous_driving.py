#!/usr/bin/env python3
"""Plan, fetch, verify, normalize, mine, and package public NGSIM data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.autonomous_driving.data.archives import extract_authoritative_member  # noqa: E402
from domains.autonomous_driving.data.contracts import (  # noqa: E402
    NGSIM_US101_AUTHORITATIVE_MEMBER,
    NGSIMSourcePlan,
    build_ngsim_core_plan,
    build_ngsim_plan,
    write_json_exclusive,
)

PROFILE_CHOICES = ("smoke", "core")
DEFAULT_CACHE_ROOT = REPO_ROOT / "works" / "autonomous_driving" / "ngsim"
DEFAULT_CORE_BUNDLE = (
    DEFAULT_CACHE_ROOT / "derived" / "ngsim_ego_pair_window_mining_v2" / "full" / "bundle"
)
DEFAULT_SMOKE_BUNDLE = DEFAULT_CACHE_ROOT / "derived" / "ngsim_smoke_refresh_v1" / "bundle"
SMOKE_START_MS = 1_118_846_979_700
SMOKE_END_MS = 1_118_846_989_700


def _add_profile(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=PROFILE_CHOICES, default="smoke")


def _profile_paths(profile: str) -> dict[str, Path]:
    root = DEFAULT_CACHE_ROOT / profile
    raw_name = "trajectories-0750am-0805am.txt" if profile == "core" else "ngsim.csv"
    return {
        "root": root,
        "plan": root / "plan.json",
        "raw": root / "raw" / raw_name,
        "download": (
            root / "raw" / "US-101-LosAngeles-CA.zip"
            if profile == "core"
            else root / "raw" / raw_name
        ),
        "source_lock": root / "source.lock.json",
        "database": root / "trajectories.sqlite3",
        "normalization_lock": root / "normalization.lock.json",
        "mining": root / "candidates.json",
        # The original ``ngsim/smoke/bundle`` predates the current seed and
        # source-event contracts.  Keep it immutable as a legacy artifact;
        # default verification points at the versioned, self-verifying
        # refresh instead of silently accepting stale evidence.
        "bundle": DEFAULT_SMOKE_BUNDLE if profile == "smoke" else root / "bundle",
    }


from domains.autonomous_driving.data.download import fetch_plan  # noqa: E402
from domains.autonomous_driving.data.ngsim import (  # noqa: E402
    create_source_lock,
    materialize_bundle,
    mine_lane_change_windows,
    mine_time_headway_windows,
    mine_windows,
    normalize_csv,
    verify_bundle,
    verify_ngsim_csv,
    verify_source_lock,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _emit(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    plan = subcommands.add_parser("plan")
    _add_profile(plan)
    plan.add_argument("--recording-id", default="us-101")
    plan.add_argument("--start-time-ms", type=int, default=SMOKE_START_MS)
    plan.add_argument("--end-time-ms", type=int, default=SMOKE_END_MS)
    plan.add_argument("--max-rows", type=int, default=50_000)
    plan.add_argument("--output", type=Path)

    fetch = subcommands.add_parser("fetch")
    _add_profile(fetch)
    fetch.add_argument("--plan", type=Path)
    fetch.add_argument("--output", type=Path)
    fetch.add_argument("--expected-sha256")

    verify = subcommands.add_parser("verify")
    _add_profile(verify)
    verify.add_argument("--raw", type=Path)
    verify.add_argument("--source-lock", type=Path)
    verify.add_argument("--bundle", type=Path)

    lock = subcommands.add_parser("lock")
    _add_profile(lock)
    lock.add_argument("--plan", type=Path)
    lock.add_argument("--raw", type=Path)
    lock.add_argument("--output", type=Path)

    normalize = subcommands.add_parser("normalize")
    _add_profile(normalize)
    normalize.add_argument("--raw", type=Path)
    normalize.add_argument("--source-lock", type=Path)
    normalize.add_argument("--output", type=Path)
    normalize.add_argument("--lock-output", type=Path)

    mine = subcommands.add_parser("mine")
    _add_profile(mine)
    mine.add_argument("--database", type=Path)
    mine.add_argument("--normalization-lock", type=Path)
    mine.add_argument("--output", type=Path)
    mine.add_argument("--window-ms", type=int, default=10_000)
    mine.add_argument("--stride-ms", type=int, default=5_000)
    mine.add_argument("--limit", type=int, default=20)
    mine.add_argument("--min-actors", type=int, default=2)
    mine.add_argument(
        "--phase-min-prevention-ms",
        type=int,
        help="Anchor on logged lead braking and require this supervisory window.",
    )
    mine.add_argument(
        "--lane-change",
        action="store_true",
        help="Mine source-native adjacent-lane/cut-in conflicts.",
    )
    mine.add_argument(
        "--headway",
        action="store_true",
        help="Mine source-native short time-headway conflicts.",
    )
    mine.add_argument(
        "--lane-change-min-prevention-ms",
        type=int,
        default=15_000,
    )
    mine.add_argument(
        "--lane-change-min-recovery-ms",
        type=int,
        default=20_000,
    )
    mine.add_argument("--headway-event-s", type=float, default=1.5)
    mine.add_argument("--headway-risk-s", type=float, default=0.8)

    materialize = subcommands.add_parser("materialize")
    _add_profile(materialize)
    materialize.add_argument("--raw", type=Path)
    materialize.add_argument("--source-lock", type=Path)
    materialize.add_argument("--database", type=Path)
    materialize.add_argument("--normalization-lock", type=Path)
    materialize.add_argument("--mining-report", type=Path)
    materialize.add_argument("--output-dir", type=Path)
    materialize.add_argument("--candidate-id")
    materialize.add_argument(
        "--license-review-approved",
        action="store_true",
        help="Record an already completed external license review in the new bundle.",
    )
    materialize.add_argument("--license-review-basis")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = _profile_paths(args.profile)
    if args.command == "plan":
        if args.profile == "core" and args.recording_id.strip().lower() != "us-101":
            raise ValueError(
                f"ngsim_core_recording_archive_not_pinned:{args.recording_id.strip().lower()}"
            )
        result = (
            build_ngsim_core_plan()
            if args.profile == "core"
            else build_ngsim_plan(
                recording_id=args.recording_id,
                start_time_ms=args.start_time_ms,
                end_time_ms=args.end_time_ms,
                max_rows=args.max_rows,
            )
        ).to_dict()
        if args.output:
            write_json_exclusive(args.output, result)
        return result
    if args.command == "fetch":
        plan_path = args.plan or paths["plan"]
        plan = (
            NGSIMSourcePlan.from_dict(_read_json(plan_path))
            if plan_path.is_file()
            else (
                build_ngsim_core_plan()
                if args.profile == "core"
                else build_ngsim_plan(
                    recording_id="us-101",
                    start_time_ms=SMOKE_START_MS,
                    end_time_ms=SMOKE_END_MS,
                    max_rows=5_000,
                )
            )
        )
        destination = args.output or paths["download"]
        result = fetch_plan(
            plan,
            destination,
            expected_sha256=args.expected_sha256 or plan.expected_download_sha256,
        )
        if args.profile != "core":
            return result
        authoritative = paths["raw"]
        # extract_authoritative_member walks nested zips, so the member inside
        # us-101-vehicle-trajectory-data.zip is found even though the outer
        # namelist does not contain the logical NGSIM_US101_AUTHORITATIVE_MEMBER path.
        member_info = extract_authoritative_member(
            destination,
            authoritative,
            suffixes=(NGSIM_US101_AUTHORITATIVE_MEMBER,),
            expected_sha256=plan.expected_authoritative_sha256,
        )
        return {
            **result,
            "authoritative_txt": str(authoritative),
            "authoritative_member": member_info["logical_nested_path"],
            "authoritative_sha256": member_info["sha256"],
            "authoritative_byte_size": member_info["byte_size"],
        }
    if args.command == "verify":
        if args.bundle and args.raw:
            raise ValueError("verify_accepts_only_one_of_raw_or_bundle")
        if args.bundle:
            return verify_bundle(args.bundle)
        default_bundle = paths["bundle"]
        if args.profile == "core" and not default_bundle.is_dir():
            default_bundle = DEFAULT_CORE_BUNDLE
        if not args.raw and default_bundle.is_dir():
            return {
                **verify_bundle(default_bundle),
                "verified_bundle": str(default_bundle),
            }
        raw = args.raw or paths["raw"]
        source_lock = args.source_lock or paths["source_lock"]
        if not raw.is_file():
            raise ValueError("verify_requires_existing_bundle_or_raw")
        if source_lock.is_file():
            return verify_source_lock(raw, _read_json(source_lock))
        return verify_ngsim_csv(raw)
    if args.command == "lock":
        plan_path = args.plan or paths["plan"]
        plan = (
            NGSIMSourcePlan.from_dict(_read_json(plan_path))
            if plan_path.is_file()
            else (
                build_ngsim_core_plan()
                if args.profile == "core"
                else build_ngsim_plan(
                    recording_id="us-101",
                    start_time_ms=SMOKE_START_MS,
                    end_time_ms=SMOKE_END_MS,
                    max_rows=5_000,
                )
            )
        )
        return create_source_lock(
            args.raw or paths["raw"], plan, args.output or paths["source_lock"]
        )
    if args.command == "normalize":
        return normalize_csv(
            args.raw or paths["raw"],
            _read_json(args.source_lock or paths["source_lock"]),
            args.output or paths["database"],
            args.lock_output or paths["normalization_lock"],
        )
    if args.command == "mine":
        if args.lane_change and args.headway:
            raise ValueError("mine_select_one_hazard_recipe")
        if (args.lane_change or args.headway) and args.phase_min_prevention_ms is not None:
            raise ValueError("mine_lane_change_and_phase_modes_are_mutually_exclusive")
        if args.lane_change:
            return mine_lane_change_windows(
                args.database or paths["database"],
                _read_json(args.normalization_lock or paths["normalization_lock"]),
                args.output or paths["mining"],
                window_ms=args.window_ms if args.window_ms != 10_000 else 60_000,
                stride_ms=args.stride_ms,
                limit=args.limit,
                min_prevention_ms=args.lane_change_min_prevention_ms,
                min_recovery_ms=args.lane_change_min_recovery_ms,
            )
        if args.headway:
            return mine_time_headway_windows(
                args.database or paths["database"],
                _read_json(args.normalization_lock or paths["normalization_lock"]),
                args.output or paths["mining"],
                window_ms=args.window_ms if args.window_ms != 10_000 else 60_000,
                limit=args.limit,
                min_prevention_ms=15_000,
                min_recovery_ms=20_000,
                event_headway_s=args.headway_event_s,
                risk_headway_s=args.headway_risk_s,
            )
        return mine_windows(
            args.database or paths["database"],
            _read_json(args.normalization_lock or paths["normalization_lock"]),
            args.output or paths["mining"],
            window_ms=args.window_ms,
            stride_ms=args.stride_ms,
            limit=args.limit,
            min_actors=args.min_actors,
            phase_min_prevention_ms=args.phase_min_prevention_ms,
        )
    if args.command == "materialize":
        return materialize_bundle(
            raw_path=args.raw or paths["raw"],
            source_lock_path=args.source_lock or paths["source_lock"],
            database_path=args.database or paths["database"],
            normalization_lock_path=args.normalization_lock or paths["normalization_lock"],
            mining_report_path=args.mining_report or paths["mining"],
            output_dir=args.output_dir or paths["bundle"],
            candidate_id=args.candidate_id,
            license_review_status=(
                "approved" if args.license_review_approved else "pending_metadata_discrepancy"
            ),
            license_review_basis=args.license_review_basis,
        )
    raise AssertionError(f"unhandled command {args.command}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _emit(run(args))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        _emit({"status": "failed", "reason": str(error)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
