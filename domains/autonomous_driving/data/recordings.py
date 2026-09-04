"""Deterministic conversion and provenance locks for official NGSIM archives."""

from __future__ import annotations

import csv
import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import (
    NGSIM_COLUMNS,
    NGSIM_METADATA_URL,
    NGSIMSourcePlan,
    build_ngsim_archive_plan,
    build_source_lock,
    file_sha256,
    object_sha256,
    write_json_exclusive,
)

OFFICIAL_NGSIM_COLUMNS: tuple[str, ...] = (
    "Vehicle_ID",
    "Frame_ID",
    "Total_Frames",
    "Global_Time",
    "Local_X",
    "Local_Y",
    "Global_X",
    "Global_Y",
    "v_length",
    "v_Width",
    "v_Class",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
    "O_Zone",
    "D_Zone",
    "Int_ID",
    "Section_ID",
    "Direction",
    "Movement",
    "Preceding",
    "Following",
    "Space_Headway",
    "Time_Headway",
    "Location",
)
CONVERSION_RECIPE_VERSION = "ngsim_official_archive_to_canonical_v1"
_SEGMENT_ID_STRIDE = 1_000_000


def _row_sha256(row: Mapping[str, str]) -> str:
    return hashlib.sha256(
        "\x1f".join(row[column] for column in OFFICIAL_NGSIM_COLUMNS).encode("utf-8")
    ).hexdigest()


def convert_official_archive_csv(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    recording_id: str,
    authoritative_member: str,
    archive_sha256: str,
    member_sha256: str,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
) -> dict[str, Any]:
    """Convert one official 25-column member into the canonical 18 columns.

    Some NGSIM multi-intersection archives reuse vehicle IDs after a segment
    reset.  A deterministic generation suffix keeps actor identity stable
    without pretending two source segments are one continuous vehicle.  The
    mapping and exact input/member hashes are retained in the report.
    """
    if output_path.exists() or report_path.exists():
        raise FileExistsError("ngsim_archive_conversion_output_exists")
    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise ValueError("ngsim_archive_member_missing_or_empty")
    if len(archive_sha256) != 64 or len(member_sha256) != 64:
        raise ValueError("ngsim_archive_hash_invalid")
    if (start_time_ms is None) != (end_time_ms is None) or (
        start_time_ms is not None
        and end_time_ms is not None
        and (start_time_ms <= 0 or end_time_ms <= start_time_ms)
    ):
        raise ValueError("ngsim_archive_selection_window_invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[int, tuple[int, int]] = {}
    generations: dict[int, int] = {}
    actor_keys: set[int] = set()
    row_count = 0
    segment_count = 0
    source_row_digest = hashlib.sha256()
    temp_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != OFFICIAL_NGSIM_COLUMNS:
                raise ValueError("ngsim_official_archive_schema_mismatch")
            with temp_path.open("w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=NGSIM_COLUMNS)
                writer.writeheader()
                for source_row_number, raw in enumerate(reader, start=2):
                    if None in raw or any(
                        raw.get(column, "") == "" for column in OFFICIAL_NGSIM_COLUMNS
                    ):
                        raise ValueError(
                            f"ngsim_official_archive_missing_value_at_row_{source_row_number}"
                        )
                    source_actor = int(raw["Vehicle_ID"])
                    timestamp_ms = int(raw["Global_Time"])
                    frame_id = int(raw["Frame_ID"])
                    prior = previous.get(source_actor)
                    if prior is not None and (
                        timestamp_ms <= prior[0]
                        or frame_id <= prior[1]
                        or (timestamp_ms - prior[0]) % 100
                    ):
                        generations[source_actor] = generations.get(source_actor, 0) + 1
                        segment_count += 1
                    previous[source_actor] = (timestamp_ms, frame_id)
                    generation = generations.get(source_actor, 0)
                    if start_time_ms is not None and not (
                        start_time_ms <= timestamp_ms < int(end_time_ms)
                    ):
                        continue
                    canonical_actor = source_actor + generation * _SEGMENT_ID_STRIDE
                    actor_keys.add(canonical_actor)
                    source_row_digest.update(_row_sha256(raw).encode("ascii"))
                    source_row_digest.update(b"\n")
                    writer.writerow(
                        {
                            "vehicle_id": canonical_actor,
                            "frame_id": frame_id,
                            "total_frames": raw["Total_Frames"],
                            "global_time": timestamp_ms,
                            "local_x": raw["Local_X"],
                            "local_y": raw["Local_Y"],
                            "global_x": raw["Global_X"],
                            "global_y": raw["Global_Y"],
                            "v_length": raw["v_length"],
                            "v_width": raw["v_Width"],
                            "v_class": raw["v_Class"],
                            "v_vel": raw["v_Vel"],
                            "v_acc": raw["v_Acc"],
                            "lane_id": raw["Lane_ID"],
                            "preceding": raw["Preceding"],
                            "following": raw["Following"],
                            "space_headway": raw["Space_Headway"],
                            "time_headway": raw["Time_Headway"],
                        }
                    )
                    row_count += 1
                target.flush()
                os.fsync(target.fileno())
        os.link(temp_path, output_path)
        temp_path.unlink()
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    report: dict[str, Any] = {
        "schema_version": "ngsim_archive_conversion_v1",
        "conversion_recipe_version": CONVERSION_RECIPE_VERSION,
        "source_dataset_id": "8ect-6jqj",
        "source_metadata_url": NGSIM_METADATA_URL,
        "recording_id": recording_id,
        "authoritative_member": authoritative_member,
        "archive_sha256": archive_sha256,
        "member_sha256": member_sha256,
        "input_member_file_sha256": file_sha256(input_path),
        "source_row_digest_sha256": source_row_digest.hexdigest(),
        "row_count": row_count,
        "actor_count": len(actor_keys),
        "segment_reset_count": segment_count,
        "selected_time_window": {
            "start_time_ms": start_time_ms,
            "end_time_ms_exclusive": end_time_ms,
            "selection_rule": "official_member_rows_with_global_time_in_half_open_window",
        },
        "identity_mapping": {
            "source_actor_id": "vehicle_id",
            "canonical_actor_id": "source_actor_id + generation * 1000000",
            "generation_increment_condition": "timestamp_or_frame_reset_or_non_100ms_gap",
        },
    }
    report["conversion_evidence_sha256"] = object_sha256(report)
    write_json_exclusive(report_path, report)
    return report


def create_archive_source_lock(
    raw_path: Path,
    output_path: Path,
    *,
    plan: NGSIMSourcePlan,
    conversion_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a source lock that binds canonical bytes to archive provenance."""
    if plan.source_kind != "official_archive_converted_csv":
        raise ValueError("ngsim_archive_lock_requires_archive_plan")
    from .ngsim import verify_ngsim_csv

    verification = verify_ngsim_csv(raw_path, plan=plan)
    lock = build_source_lock(
        plan,
        raw_path,
        row_count=int(verification["row_count"]),
        semantic_sha256=str(verification["raw_semantic_sha256"]),
    )
    lock["lock_strategy"] = "doi+archive_member+conversion+raw_sha256+row_semantic_sha256"
    lock["archive_provenance"] = {
        "archive_sha256": str(plan.expected_download_sha256 or ""),
        "member": str(plan.authoritative_member or ""),
        "member_sha256": str(plan.expected_authoritative_sha256 or ""),
        "conversion_recipe_version": CONVERSION_RECIPE_VERSION,
        "conversion_evidence_sha256": str(
            conversion_report.get("conversion_evidence_sha256") or ""
        ),
        "converted_file_sha256": file_sha256(raw_path),
    }
    lock.pop("source_evidence_sha256", None)
    lock["source_evidence_sha256"] = object_sha256(lock)
    write_json_exclusive(output_path, lock)
    return lock


def archive_plan_from_conversion(
    *,
    recording_id: str,
    start_time_ms: int,
    end_time_ms: int,
    max_rows: int,
    archive_url: str,
    archive_sha256: str,
    authoritative_member: str,
    member_sha256: str,
    output_name: str,
) -> NGSIMSourcePlan:
    return build_ngsim_archive_plan(
        recording_id=recording_id,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        max_rows=max_rows,
        archive_url=archive_url,
        archive_sha256=archive_sha256,
        authoritative_member=authoritative_member,
        member_sha256=member_sha256,
        output_name=output_name,
    )
