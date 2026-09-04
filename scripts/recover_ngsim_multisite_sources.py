#!/usr/bin/env python3
"""Prepare hash-locked I-80 or Peachtree source bytes for v0.61 mining."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.autonomous_driving.data.archives import extract_authoritative_member  # noqa: E402
from domains.autonomous_driving.data.contracts import (  # noqa: E402
    NGSIM_COLUMNS,
    NGSIMSourcePlan,
    build_ngsim_archive_plan,
    object_sha256,
    write_json_exclusive,
)
from domains.autonomous_driving.data.ngsim import (  # noqa: E402
    mine_lane_change_windows,
    normalize_csv,
    verify_ngsim_csv,
)
from domains.autonomous_driving.data.recordings import (  # noqa: E402
    CONVERSION_RECIPE_VERSION,
    convert_official_archive_csv,
    create_archive_source_lock,
)


@dataclass(frozen=True)
class RecordingSpec:
    recording_id: str
    canonical_name: str
    conversion: str
    start_time_ms: int
    end_time_ms: int


@dataclass(frozen=True)
class SourceRequirement:
    archive_name: str
    archive_url: str
    archive_sha256: str
    authoritative_member: str
    authoritative_member_sha256: str


MULTISITE_REQUIREMENTS = {
    "i-80": SourceRequirement(
        archive_name="I-80-Emeryville-CA.zip",
        archive_url=(
            "https://data.transportation.gov/api/views/8ect-6jqj/files/"
            "ea269540-b86c-4b2d-a9c2-c8f4c0a3d0a0?download=true&"
            "filename=I-80-Emeryville-CA.zip"
        ),
        archive_sha256="b274bb96f20c971e37651e73ca0f86677fda8009e22885e71322716eca84ff73",
        authoritative_member="trajectories-0400-0415.txt",
        authoritative_member_sha256=(
            "a169ccbfd41bcb48e46da832857a0bac48bd96465423ceade2967bce2c8366df"
        ),
    ),
    "peachtree": SourceRequirement(
        archive_name="Peachtree-Street-Atlanta-GA.zip",
        archive_url=(
            "https://data.transportation.gov/api/views/8ect-6jqj/files/"
            "3dba3db1-dd9a-46b3-96d0-07d8c4461feb?download=true&"
            "filename=Peachtree-Street-Atlanta-GA.zip"
        ),
        archive_sha256="91b9b1d4ed1d261e8c525df536adc466b6844ff49dc6fb08b815c735fe072a2f",
        authoritative_member="NGSIM_Peachtree_Vehicle_Trajectories.csv",
        authoritative_member_sha256=(
            "1ec10349176788ca199132677154b9b7d207b7ec9605bfa16b600660085a76d9"
        ),
    ),
}


RECORDING_SPECS = {
    "i-80": RecordingSpec(
        recording_id="i-80",
        canonical_name="i80_1113433176100_1113433351500.csv",
        conversion="official_txt_18_to_csv",
        start_time_ms=1_113_433_176_100,
        end_time_ms=1_113_433_351_500,
    ),
    "peachtree": RecordingSpec(
        recording_id="peachtree",
        canonical_name="peachtree_1163040000_1163395200.csv",
        conversion="official_csv_25_to_18",
        start_time_ms=1_163_040_000,
        end_time_ms=1_163_395_200,
    ),
}


def _requirement(recording_id: str) -> SourceRequirement:
    return MULTISITE_REQUIREMENTS[recording_id]


def _build_plan(spec: RecordingSpec, requirement: SourceRequirement) -> NGSIMSourcePlan:
    return build_ngsim_archive_plan(
        recording_id=spec.recording_id,
        start_time_ms=spec.start_time_ms,
        end_time_ms=spec.end_time_ms,
        max_rows=1_000_000,
        archive_url=requirement.archive_url,
        archive_sha256=requirement.archive_sha256,
        authoritative_member=requirement.authoritative_member,
        member_sha256=requirement.authoritative_member_sha256,
        output_name=spec.canonical_name,
    )


def convert_authoritative_txt(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    spec: RecordingSpec,
    requirement: SourceRequirement,
) -> dict[str, Any]:
    """Select a bounded official 18-column TXT window into canonical CSV."""
    if output_path.exists() or report_path.exists():
        raise FileExistsError("ngsim_multisite_conversion_output_exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    row_count = 0
    source_rows = hashlib.sha256()
    try:
        with input_path.open("r", encoding="utf-8-sig") as source, temp_path.open(
            "w", encoding="utf-8", newline=""
        ) as target:
            writer = csv.writer(target)
            writer.writerow(NGSIM_COLUMNS)
            for row_number, line in enumerate(source, start=1):
                values = line.split()
                if len(values) != len(NGSIM_COLUMNS):
                    raise ValueError(f"ngsim_txt_schema_mismatch_at_row_{row_number}")
                timestamp_ms = int(values[3])
                if not spec.start_time_ms <= timestamp_ms < spec.end_time_ms:
                    continue
                writer.writerow(values)
                source_rows.update(line.encode("utf-8"))
                row_count += 1
            target.flush()
            os.fsync(target.fileno())
        if row_count == 0:
            raise ValueError("ngsim_multisite_conversion_empty")
        os.link(temp_path, output_path)
        temp_path.unlink()
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    report: dict[str, Any] = {
        "schema_version": "ngsim_archive_conversion_v1",
        "conversion_recipe_version": CONVERSION_RECIPE_VERSION,
        "recording_id": spec.recording_id,
        "authoritative_member": requirement.authoritative_member,
        "archive_sha256": requirement.archive_sha256,
        "member_sha256": requirement.authoritative_member_sha256,
        "source_row_digest_sha256": source_rows.hexdigest(),
        "row_count": row_count,
        "selected_time_window": {
            "start_time_ms": spec.start_time_ms,
            "end_time_ms_exclusive": spec.end_time_ms,
            "selection_rule": "official_member_rows_with_global_time_in_half_open_window",
        },
        "identity_mapping": "source_vehicle_id_preserved",
    }
    report["conversion_evidence_sha256"] = object_sha256(report)
    write_json_exclusive(report_path, report)
    return report


def prepare_recording(
    *, recording_id: str, archive_path: Path, work_root: Path
) -> dict[str, Any]:
    spec = RECORDING_SPECS[recording_id]
    requirement = _requirement(recording_id)
    if work_root.exists():
        raise FileExistsError("ngsim_multisite_work_root_exists")
    work_root.mkdir(parents=True)
    source_dir = work_root / "source"
    source_dir.mkdir()
    extracted = source_dir / requirement.authoritative_member
    member = extract_authoritative_member(
        archive_path,
        extracted,
        suffixes=(requirement.authoritative_member,),
        expected_sha256=requirement.authoritative_member_sha256,
    )
    plan = _build_plan(spec, requirement)
    plan_path = work_root / "source.plan.json"
    write_json_exclusive(plan_path, plan.to_dict())
    canonical = source_dir / spec.canonical_name
    conversion_path = work_root / "conversion.report.json"
    if spec.conversion == "official_txt_18_to_csv":
        conversion = convert_authoritative_txt(
            extracted,
            canonical,
            conversion_path,
            spec=spec,
            requirement=requirement,
        )
    else:
        conversion = convert_official_archive_csv(
            extracted,
            canonical,
            conversion_path,
            recording_id=recording_id,
            authoritative_member=requirement.authoritative_member,
            archive_sha256=requirement.archive_sha256,
            member_sha256=requirement.authoritative_member_sha256,
            start_time_ms=spec.start_time_ms,
            end_time_ms=spec.end_time_ms,
        )
    source_lock_path = work_root / "source.lock.json"
    source_lock = create_archive_source_lock(
        canonical,
        source_lock_path,
        plan=plan,
        conversion_report=conversion,
    )
    verification = verify_ngsim_csv(canonical, plan=plan)
    database = work_root / "trajectories.sqlite3"
    normalization_path = work_root / "normalization.lock.json"
    normalization = normalize_csv(
        canonical,
        source_lock,
        database,
        normalization_path,
    )
    mining_path = work_root / "mining" / "lane_change_candidates.json"
    mining = mine_lane_change_windows(
        database,
        normalization,
        mining_path,
        window_ms=60_000,
        stride_ms=5_000,
        limit=24,
        min_prevention_ms=15_000,
        min_recovery_ms=20_000,
    )
    report: dict[str, Any] = {
        "schema_version": "ngsim_multisite_source_recovery_v1",
        "status": "source_locked_and_mined",
        "recording_id": recording_id,
        "archive": {
            "path": str(archive_path.resolve()),
            "sha256": requirement.archive_sha256,
            "member": member,
        },
        "canonical_source": {
            "path": str(canonical.resolve()),
            "verification": verification,
            "source_evidence_sha256": source_lock["source_evidence_sha256"],
        },
        "conversion": conversion,
        "normalization": {
            "path": str(database.resolve()),
            "evidence_sha256": normalization["normalization_evidence_sha256"],
        },
        "mining": {
            "path": str(mining_path.resolve()),
            "candidate_count": len(mining.get("candidates") or []),
            "recipe_version": mining.get("mining_recipe_version"),
        },
        "formal_core_allowed": False,
    }
    write_json_exclusive(work_root / "source_recovery_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", choices=tuple(RECORDING_SPECS), required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = prepare_recording(
            recording_id=args.recording_id,
            archive_path=args.archive.resolve(),
            work_root=args.work_root.resolve(),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "recording_id": report["recording_id"],
                "candidate_count": report["mining"]["candidate_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
