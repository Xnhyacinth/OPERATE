"""Verify, normalize, mine, and package USDOT NGSIM vehicle trajectories."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess  # nosec B404
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .contracts import (
    NGSIM_ATTRIBUTION,
    NGSIM_COLUMNS,
    NGSIM_DATASET_ID,
    NGSIM_DOI,
    NGSIM_LICENSE_ID,
    NGSIM_LICENSE_URL,
    NGSIM_PORTAL_LICENSE_ID,
    NGSIM_PORTAL_LICENSE_URL,
    NGSIM_US101_AUTHORITATIVE_ROW_COUNT,
    NGSIMSourcePlan,
    build_source_lock,
    canonical_json_bytes,
    file_sha256,
    object_sha256,
    write_json_exclusive,
)

FEET_TO_METRES = 0.3048
NORMALIZATION_RECIPE_VERSION = "ngsim_source_to_sqlite_si_v2"
MINING_RECIPE_VERSION = "ngsim_phase_event_window_mining_v3"
LANE_CHANGE_MINING_RECIPE_VERSION = "ngsim_lane_change_window_mining_v1"
HEADWAY_MINING_RECIPE_VERSION = "ngsim_time_headway_window_mining_v1"
SUPPORTED_MINING_RECIPE_VERSIONS = frozenset(
    {
        MINING_RECIPE_VERSION,
        LANE_CHANGE_MINING_RECIPE_VERSION,
        HEADWAY_MINING_RECIPE_VERSION,
    }
)
SQLITE_SCHEMA_VERSION = 1
HARD_BRAKE_THRESHOLD_MPS2 = -1.5
SOURCE_EVENT_TICK_MS = 5_000

INTEGER_COLUMNS = {
    "vehicle_id",
    "frame_id",
    "total_frames",
    "global_time",
    "v_class",
    "lane_id",
    "preceding",
    "following",
}
FLOAT_COLUMNS = set(NGSIM_COLUMNS) - INTEGER_COLUMNS


def _parse_row(raw: Mapping[str, str], row_number: int) -> dict[str, int | float]:
    parsed: dict[str, int | float] = {}
    try:
        for name in INTEGER_COLUMNS:
            parsed[name] = int(raw[name])
        for name in FLOAT_COLUMNS:
            value = float(raw[name])
            if not math.isfinite(value):
                raise ValueError
            parsed[name] = value
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"ngsim_invalid_value_at_row_{row_number}") from error
    if (
        int(parsed["vehicle_id"]) <= 0
        or int(parsed["frame_id"]) <= 0
        or int(parsed["total_frames"]) <= 0
        or int(parsed["global_time"]) <= 0
        or float(parsed["v_length"]) <= 0
        or float(parsed["v_width"]) <= 0
        or int(parsed["lane_id"]) < 0
    ):
        raise ValueError(f"ngsim_out_of_range_value_at_row_{row_number}")
    return parsed


def _iter_source_rows(path: Path) -> Iterable[tuple[int, dict[str, int | float]]]:
    if path.suffix.lower() == ".txt":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row_number, line in enumerate(stream, start=1):
                values = line.split()
                if len(values) != len(NGSIM_COLUMNS):
                    raise ValueError(f"ngsim_txt_schema_mismatch_at_row_{row_number}")
                yield (
                    row_number,
                    _parse_row(dict(zip(NGSIM_COLUMNS, values, strict=True)), row_number),
                )
        return
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        if fieldnames != NGSIM_COLUMNS:
            raise ValueError("ngsim_csv_schema_mismatch")
        for row_number, raw in enumerate(reader, start=2):
            if None in raw or any(raw.get(column, "") == "" for column in NGSIM_COLUMNS):
                raise ValueError(f"ngsim_missing_value_at_row_{row_number}")
            yield row_number, _parse_row(raw, row_number)


def verify_ngsim_csv(
    path: Path,
    *,
    plan: NGSIMSourcePlan | None = None,
) -> dict[str, Any]:
    """Validate schema, ordering, identities, cadence, and truncation guards."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("ngsim_csv_missing_or_empty")
    semantic = hashlib.sha256()
    previous_time_by_actor: dict[int, int] = {}
    previous_frame_by_actor: dict[int, int] = {}
    identities: set[tuple[int, int]] = set()
    actors: set[int] = set()
    timestamps: set[int] = set()
    row_count = 0
    if plan is not None and plan.profile == "core" and path.suffix.lower() != ".txt":
        raise ValueError("ngsim_core_requires_authoritative_txt_not_truncated_csv")
    for row_number, row in _iter_source_rows(path):
        vehicle_id = int(row["vehicle_id"])
        timestamp = int(row["global_time"])
        frame_id = int(row["frame_id"])
        identity = (vehicle_id, timestamp)
        if identity in identities:
            raise ValueError(f"ngsim_duplicate_actor_time_at_row_{row_number}")
        identities.add(identity)
        previous_actor_time = previous_time_by_actor.get(vehicle_id)
        previous_actor_frame = previous_frame_by_actor.get(vehicle_id)
        if previous_actor_time is not None:
            delta = timestamp - previous_actor_time
            if (
                delta <= 0
                or delta % 100
                or previous_actor_frame is None
                or frame_id <= previous_actor_frame
            ):
                raise ValueError(f"ngsim_cadence_violation_at_row_{row_number}")
        previous_time_by_actor[vehicle_id] = timestamp
        previous_frame_by_actor[vehicle_id] = frame_id
        actors.add(vehicle_id)
        timestamps.add(timestamp)
        semantic.update(canonical_json_bytes(row))
        semantic.update(b"\n")
        row_count += 1
    if not row_count:
        raise ValueError("ngsim_csv_has_no_data_rows")
    if plan is not None:
        if plan.profile == "core" and row_count != NGSIM_US101_AUTHORITATIVE_ROW_COUNT:
            raise ValueError("ngsim_core_authoritative_txt_row_count_mismatch")
        if (
            plan.source_kind == "official_archive_authoritative_txt"
            and plan.expected_authoritative_sha256
            and file_sha256(path) != plan.expected_authoritative_sha256
        ):
            raise ValueError("ngsim_core_authoritative_txt_sha256_mismatch")
        if plan.source_kind == "soda_bounded_csv" and row_count >= plan.max_rows:
            raise ValueError("ngsim_query_may_be_truncated_at_max_rows")
        if plan.source_kind in {"soda_bounded_csv", "official_archive_converted_csv"} and (
            min(timestamps) < plan.start_time_ms or max(timestamps) > plan.end_time_ms
        ):
            raise ValueError("ngsim_rows_outside_planned_time_window")
    return {
        "status": "verified",
        "raw_sha256": file_sha256(path),
        "raw_byte_size": path.stat().st_size,
        "raw_semantic_sha256": semantic.hexdigest(),
        "row_count": row_count,
        "actor_count": len(actors),
        "timestamp_count": len(timestamps),
        "min_timestamp_ms": min(timestamps),
        "max_timestamp_ms": max(timestamps),
        "sample_period_ms": 100,
    }


def create_source_lock(
    raw_path: Path,
    plan: NGSIMSourcePlan,
    output_path: Path,
) -> dict[str, Any]:
    verification = verify_ngsim_csv(raw_path, plan=plan)
    lock = build_source_lock(
        plan,
        raw_path,
        row_count=int(verification["row_count"]),
        semantic_sha256=str(verification["raw_semantic_sha256"]),
    )
    write_json_exclusive(output_path, lock)
    return lock


def verify_source_lock(raw_path: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the lock is canonical and matches the exact raw file."""
    if lock.get("schema_version") != "ngsim_source_lock_v1":
        raise ValueError("ngsim_source_lock_schema_mismatch")
    evidence = str(lock.get("source_evidence_sha256") or "")
    unsigned = dict(lock)
    unsigned.pop("source_evidence_sha256", None)
    if evidence != object_sha256(unsigned):
        raise ValueError("ngsim_source_lock_evidence_mismatch")
    plan = NGSIMSourcePlan.from_dict(lock.get("source_plan") or {})
    if plan.source_kind == "official_archive_converted_csv":
        provenance = lock.get("archive_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("ngsim_archive_source_provenance_missing")
        if (
            len(str(provenance.get("archive_sha256") or "")) != 64
            or len(str(provenance.get("member_sha256") or "")) != 64
            or str(provenance.get("member") or "") != str(plan.authoritative_member or "")
            or str(provenance.get("archive_sha256") or "")
            != str(plan.expected_download_sha256 or "")
            or str(provenance.get("member_sha256") or "")
            != str(plan.expected_authoritative_sha256 or "")
            or str(provenance.get("conversion_recipe_version") or "")
            != "ngsim_official_archive_to_canonical_v1"
        ):
            raise ValueError("ngsim_archive_source_provenance_mismatch")
    verification = verify_ngsim_csv(raw_path, plan=plan)
    comparisons = {
        "raw_file_name": raw_path.name,
        "raw_byte_size": verification["raw_byte_size"],
        "raw_sha256": verification["raw_sha256"],
        "row_count": verification["row_count"],
        "raw_semantic_sha256": verification["raw_semantic_sha256"],
    }
    for field, observed in comparisons.items():
        if lock.get(field) != observed:
            raise ValueError(f"ngsim_source_lock_{field}_mismatch")
    return {**verification, "source_evidence_sha256": evidence}


def _actor_id(recording_id: str, source_actor: int) -> str:
    return f"{recording_id}:{source_actor}"


def _optional_actor_id(recording_id: str, source_actor: int) -> str | None:
    return _actor_id(recording_id, source_actor) if source_actor > 0 else None


def _metres(value: int | float) -> float:
    return round(float(value) * FEET_TO_METRES, 9)


def _normalized_row(
    source: Mapping[str, int | float],
    *,
    recording_id: str,
    source_row_number: int,
) -> dict[str, Any]:
    source_actor = int(source["vehicle_id"])
    raw_row_sha256 = object_sha256(dict(source))
    return {
        "recording_id": recording_id,
        "actor_id": _actor_id(recording_id, source_actor),
        "source_actor_id": source_actor,
        "frame_id": int(source["frame_id"]),
        "total_frames": int(source["total_frames"]),
        "timestamp_ms": int(source["global_time"]),
        "local_x_m": _metres(source["local_x"]),
        "local_y_m": _metres(source["local_y"]),
        "global_x_m": _metres(source["global_x"]),
        "global_y_m": _metres(source["global_y"]),
        "length_m": _metres(source["v_length"]),
        "width_m": _metres(source["v_width"]),
        "class_id": int(source["v_class"]),
        "speed_mps": _metres(source["v_vel"]),
        "acceleration_mps2": _metres(source["v_acc"]),
        "lane_id": int(source["lane_id"]),
        "preceding_actor_id": _optional_actor_id(recording_id, int(source["preceding"])),
        "following_actor_id": _optional_actor_id(recording_id, int(source["following"])),
        "space_headway_m": _metres(source["space_headway"]),
        "time_headway_s": round(float(source["time_headway"]), 9),
        "source_row_number": source_row_number,
        "source_row_sha256": raw_row_sha256,
    }


_CREATE_SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE states (
    recording_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    source_actor_id INTEGER NOT NULL,
    frame_id INTEGER NOT NULL,
    total_frames INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    local_x_m REAL NOT NULL,
    local_y_m REAL NOT NULL,
    global_x_m REAL NOT NULL,
    global_y_m REAL NOT NULL,
    length_m REAL NOT NULL CHECK (length_m > 0),
    width_m REAL NOT NULL CHECK (width_m > 0),
    class_id INTEGER NOT NULL,
    speed_mps REAL NOT NULL,
    acceleration_mps2 REAL NOT NULL,
    lane_id INTEGER NOT NULL,
    preceding_actor_id TEXT,
    following_actor_id TEXT,
    space_headway_m REAL NOT NULL,
    time_headway_s REAL NOT NULL,
    source_row_number INTEGER NOT NULL UNIQUE,
    source_row_sha256 TEXT NOT NULL CHECK (length(source_row_sha256) = 64),
    PRIMARY KEY (actor_id, timestamp_ms)
) WITHOUT ROWID;

CREATE INDEX states_by_time ON states(timestamp_ms, actor_id);
CREATE INDEX states_by_lane_time ON states(lane_id, timestamp_ms, actor_id);
CREATE INDEX states_by_safety ON states(time_headway_s, acceleration_mps2);
"""

_INSERT_STATE = """
INSERT INTO states (
    recording_id, actor_id, source_actor_id, frame_id, total_frames,
    timestamp_ms, local_x_m, local_y_m, global_x_m, global_y_m,
    length_m, width_m, class_id, speed_mps, acceleration_mps2,
    lane_id, preceding_actor_id, following_actor_id, space_headway_m,
    time_headway_s, source_row_number, source_row_sha256
) VALUES (
    :recording_id, :actor_id, :source_actor_id, :frame_id, :total_frames,
    :timestamp_ms, :local_x_m, :local_y_m, :global_x_m, :global_y_m,
    :length_m, :width_m, :class_id, :speed_mps, :acceleration_mps2,
    :lane_id, :preceding_actor_id, :following_actor_id, :space_headway_m,
    :time_headway_s, :source_row_number, :source_row_sha256
)
"""


def _publish_file(temp_path: Path, destination: Path) -> None:
    try:
        os.link(temp_path, destination)
    except FileExistsError as error:
        raise FileExistsError("ngsim_materialization_target_exists") from error
    temp_path.unlink()


def normalize_csv(
    raw_path: Path,
    source_lock: Mapping[str, Any],
    database_path: Path,
    normalization_lock_path: Path,
) -> dict[str, Any]:
    """Normalize a locked CSV into an indexed SI-unit SQLite database."""
    if database_path.exists() or normalization_lock_path.exists():
        raise FileExistsError("ngsim_normalization_output_exists")
    verification = verify_source_lock(raw_path, source_lock)
    plan = NGSIMSourcePlan.from_dict(source_lock.get("source_plan") or {})
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = database_path.with_name(f".{database_path.name}.tmp-{os.getpid()}")
    if temp_path.exists():
        raise FileExistsError("ngsim_normalization_temp_exists")
    semantic = hashlib.sha256()
    row_count = 0
    actor_ids: set[str] = set()
    try:
        connection = sqlite3.connect(temp_path)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute(f"PRAGMA user_version={SQLITE_SCHEMA_VERSION}")
            connection.executescript(_CREATE_SCHEMA)
            metadata = {
                "schema_version": "ngsim_normalized_sqlite_v1",
                "normalization_recipe_version": NORMALIZATION_RECIPE_VERSION,
                "source_evidence_sha256": str(source_lock["source_evidence_sha256"]),
                "recording_id": plan.recording_id,
                "distance_unit": "metre",
                "speed_unit": "metre_per_second",
                "acceleration_unit": "metre_per_second_squared",
                "sample_period_ms": "100",
            }
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                sorted(metadata.items()),
            )
            for source_row_number, source in _iter_source_rows(raw_path):
                normalized = _normalized_row(
                    source,
                    recording_id=plan.recording_id,
                    source_row_number=source_row_number,
                )
                connection.execute(_INSERT_STATE, normalized)
                semantic.update(canonical_json_bytes(normalized))
                semantic.update(b"\n")
                actor_ids.add(str(normalized["actor_id"]))
                row_count += 1
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise ValueError("ngsim_sqlite_integrity_check_failed")
        finally:
            connection.close()
        _publish_file(temp_path, database_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    if row_count != verification["row_count"]:
        raise ValueError("ngsim_normalized_row_count_mismatch")
    report: dict[str, Any] = {
        "schema_version": "ngsim_normalization_lock_v1",
        "normalization_recipe_version": NORMALIZATION_RECIPE_VERSION,
        "source_evidence_sha256": source_lock["source_evidence_sha256"],
        "raw_sha256": source_lock["raw_sha256"],
        "database_file_name": database_path.name,
        "database_sha256": file_sha256(database_path),
        "normalized_semantic_sha256": semantic.hexdigest(),
        "row_count": row_count,
        "actor_count": len(actor_ids),
        "sqlite_schema_version": SQLITE_SCHEMA_VERSION,
        "si_conversion": {
            "feet_to_metres": FEET_TO_METRES,
            "time_headway_input_unit": "seconds",
        },
    }
    report["normalization_evidence_sha256"] = object_sha256(report)
    try:
        write_json_exclusive(normalization_lock_path, report)
    except Exception:
        database_path.unlink(missing_ok=True)
        raise
    return report


def verify_normalization_lock(
    database_path: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    if lock.get("schema_version") != "ngsim_normalization_lock_v1":
        raise ValueError("ngsim_normalization_lock_schema_mismatch")
    unsigned = dict(lock)
    evidence = str(unsigned.pop("normalization_evidence_sha256", ""))
    if evidence != object_sha256(unsigned):
        raise ValueError("ngsim_normalization_evidence_mismatch")
    if lock.get("database_file_name") != database_path.name:
        raise ValueError("ngsim_normalization_database_name_mismatch")
    if lock.get("database_sha256") != file_sha256(database_path):
        raise ValueError("ngsim_normalization_database_hash_mismatch")
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        row_count = connection.execute("SELECT COUNT(*) FROM states").fetchone()[0]
        actor_count = connection.execute("SELECT COUNT(DISTINCT actor_id) FROM states").fetchone()[
            0
        ]
    if integrity != ("ok",):
        raise ValueError("ngsim_normalization_database_integrity_failed")
    if row_count != lock.get("row_count") or actor_count != lock.get("actor_count"):
        raise ValueError("ngsim_normalization_database_count_mismatch")
    return {
        "status": "verified",
        "normalization_evidence_sha256": evidence,
        "row_count": row_count,
        "actor_count": actor_count,
    }


def _window_semantic_sha256(
    connection: sqlite3.Connection,
    start_ms: int,
    end_ms: int,
) -> str:
    digest = hashlib.sha256()
    cursor = connection.execute(
        """
        SELECT actor_id, timestamp_ms, local_x_m, local_y_m, speed_mps,
               acceleration_mps2, lane_id, space_headway_m, time_headway_s,
               source_row_sha256
          FROM states
         WHERE timestamp_ms >= ? AND timestamp_ms < ?
         ORDER BY timestamp_ms, actor_id
        """,
        (start_ms, end_ms),
    )
    for row in cursor:
        digest.update(canonical_json_bytes(list(row)))
        digest.update(b"\n")
    return digest.hexdigest()


def mine_windows(
    database_path: Path,
    normalization_lock: Mapping[str, Any],
    output_path: Path,
    *,
    window_ms: int = 10_000,
    stride_ms: int = 5_000,
    limit: int = 20,
    min_actors: int = 2,
    phase_min_prevention_ms: int | None = None,
) -> dict[str, Any]:
    """Rank fixed windows using deterministic, integer-valued risk features."""
    if phase_min_prevention_ms is not None and phase_min_prevention_ms > 0:
        return mine_phase_windows(
            database_path,
            normalization_lock,
            output_path,
            window_ms=window_ms,
            stride_ms=stride_ms,
            limit=limit,
            min_actors=min_actors,
            min_prevention_ms=phase_min_prevention_ms,
        )
    if output_path.exists():
        raise FileExistsError("ngsim_mining_output_exists")
    verify_normalization_lock(database_path, normalization_lock)
    if window_ms <= 0 or stride_ms <= 0 or window_ms % 100 or stride_ms % 100:
        raise ValueError("ngsim_mining_window_invalid")
    if limit <= 0 or min_actors <= 0:
        raise ValueError("ngsim_mining_limit_invalid")
    candidates: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        bounds = connection.execute(
            "SELECT MIN(timestamp_ms), MAX(timestamp_ms) FROM states"
        ).fetchone()
        if bounds is None or bounds[0] is None or bounds[1] is None:
            raise ValueError("ngsim_mining_database_empty")
        first_ms, last_ms = int(bounds[0]), int(bounds[1])
        start_ms = first_ms
        while start_ms <= last_ms:
            end_ms = start_ms + window_ms
            aggregate = connection.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT actor_id),
                       MIN(CASE WHEN time_headway_s > 0 THEN time_headway_s END),
                       MIN(acceleration_mps2), MIN(timestamp_ms), MAX(timestamp_ms)
                  FROM states
                 WHERE timestamp_ms >= ? AND timestamp_ms < ?
                """,
                (start_ms, end_ms),
            ).fetchone()
            if aggregate is None:
                raise RuntimeError("internal mining aggregate was not initialized")
            (
                row_count,
                actor_count,
                min_headway,
                min_acceleration,
                observed_start_ms,
                observed_end_ms,
            ) = aggregate
            if int(actor_count) >= min_actors:
                observed_start_ms = int(observed_start_ms)
                observed_end_ms = int(observed_end_ms)
                observed_coverage_ms = observed_end_ms - observed_start_ms + 100
                configured_window_complete = bool(
                    observed_start_ms == start_ms and observed_end_ms == end_ms - 100
                )
                window_semantics = {
                    "interval_convention": "[start_time_ms,end_time_ms_exclusive)",
                    "requested_window_ms": window_ms,
                    "sample_period_ms": 100,
                    "expected_timestamp_count": window_ms // 100,
                    "observed_start_time_ms": observed_start_ms,
                    "observed_end_time_ms_inclusive": observed_end_ms,
                    "observed_coverage_ms": observed_coverage_ms,
                    "configured_window_complete": configured_window_complete,
                }
                max_concurrent = connection.execute(
                    """
                    SELECT COALESCE(MAX(n), 0)
                      FROM (
                        SELECT COUNT(*) AS n
                          FROM states
                         WHERE timestamp_ms >= ? AND timestamp_ms < ?
                         GROUP BY timestamp_ms
                      )
                    """,
                    (start_ms, end_ms),
                ).fetchone()[0]
                lane_changes = connection.execute(
                    """
                    SELECT COALESCE(SUM(lanes - 1), 0)
                      FROM (
                        SELECT COUNT(DISTINCT lane_id) AS lanes
                          FROM states
                         WHERE timestamp_ms >= ? AND timestamp_ms < ?
                         GROUP BY actor_id
                      )
                    """,
                    (start_ms, end_ms),
                ).fetchone()[0]
                actors = [
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT DISTINCT actor_id
                          FROM states
                         WHERE timestamp_ms >= ? AND timestamp_ms < ?
                         ORDER BY actor_id
                        """,
                        (start_ms, end_ms),
                    )
                ]
                hazard_row = connection.execute(
                    """
                    SELECT actor_id, preceding_actor_id, timestamp_ms,
                           source_row_number, source_row_sha256, time_headway_s
                      FROM states
                     WHERE timestamp_ms >= ? AND timestamp_ms < ?
                       AND preceding_actor_id IS NOT NULL
                       AND time_headway_s > 0
                     ORDER BY time_headway_s, timestamp_ms, actor_id
                     LIMIT 1
                    """,
                    (start_ms, end_ms),
                ).fetchone()
                hazard_context: dict[str, Any] | None = None
                if hazard_row is not None:
                    ego_actor, conflict_actor, hazard_ms = (
                        str(hazard_row[0]),
                        str(hazard_row[1]),
                        int(hazard_row[2]),
                    )
                    first_observable_ms = int(
                        connection.execute(
                            """
                            SELECT MIN(timestamp_ms)
                              FROM states
                             WHERE timestamp_ms >= ? AND timestamp_ms < ?
                               AND actor_id = ? AND preceding_actor_id = ?
                            """,
                            (start_ms, end_ms, ego_actor, conflict_actor),
                        ).fetchone()[0]
                    )
                    braking_row = connection.execute(
                        """
                        SELECT actor_id, timestamp_ms, local_x_m, local_y_m,
                               speed_mps, acceleration_mps2, lane_id, length_m,
                               width_m, source_row_number, source_row_sha256
                          FROM states
                         WHERE actor_id = ?
                           AND timestamp_ms >= ? AND timestamp_ms <= ?
                           AND acceleration_mps2 <= ?
                         ORDER BY ABS(timestamp_ms - ?), timestamp_ms
                         LIMIT 1
                        """,
                        (
                            conflict_actor,
                            max(start_ms, hazard_ms - 6_000),
                            hazard_ms,
                            HARD_BRAKE_THRESHOLD_MPS2,
                            hazard_ms - 5_000,
                        ),
                    ).fetchone()
                    hazard_kind = (
                        "lead_vehicle_braking"
                        if braking_row is not None
                        else "minimum_time_headway_conflict"
                    )
                    hazard_event_ms = (
                        int(braking_row[1]) if braking_row is not None else hazard_ms - 5_000
                    )
                    preventive_window_ms = max(0, hazard_event_ms - first_observable_ms)
                    response_window_ms = max(0, hazard_ms - hazard_event_ms)
                    recovery_window_ms = max(0, end_ms - hazard_ms)
                    hazard_event = None
                    if braking_row is not None:
                        hazard_event = {
                            "actor_id": str(braking_row[0]),
                            "timestamp_ms": int(braking_row[1]),
                            "local_x_m": float(braking_row[2]),
                            "local_y_m": float(braking_row[3]),
                            "speed_mps": float(braking_row[4]),
                            "acceleration_mps2": float(braking_row[5]),
                            "lane_id": int(braking_row[6]),
                            "length_m": float(braking_row[7]),
                            "width_m": float(braking_row[8]),
                            "source_row_number": int(braking_row[9]),
                            "source_row_sha256": str(braking_row[10]),
                        }
                    hazard_context = {
                        "hazard_kind": hazard_kind,
                        "ego_actor_id": ego_actor,
                        "conflict_actor_id": conflict_actor,
                        "first_observable_time_ms": first_observable_ms,
                        "latest_preventive_command_time_ms": hazard_event_ms,
                        "hazard_event_time_ms": hazard_event_ms,
                        "hazard_event": hazard_event,
                        "risk_boundary_proxy_time_ms": hazard_ms,
                        "risk_boundary_proxy": "minimum_positive_logged_time_headway",
                        "recovery_window_end_time_ms": end_ms,
                        "supervisory_prevention_window_ms": preventive_window_ms,
                        "protective_response_window_ms": response_window_ms,
                        "recovery_window_ms": recovery_window_ms,
                        "source_row_number": int(hazard_row[3]),
                        "source_row_sha256": str(hazard_row[4]),
                        "minimum_time_headway_ms": int(round(float(hazard_row[5]) * 1000)),
                        "phase_window_complete": bool(
                            configured_window_complete
                            and braking_row is not None
                            and preventive_window_ms >= 5_000
                            and response_window_ms >= 100
                            and recovery_window_ms >= 20_000
                        ),
                    }
                min_accel = float(min_acceleration or 0.0)
                hard_brake_milli = max(0, int(round(-min_accel * 1000)))
                headway_ms = int(round(float(min_headway) * 1000)) if min_headway is not None else 0
                inverse_headway_milli = int(1_000_000 / max(1, headway_ms)) if headway_ms else 0
                risk_score_milli = (
                    hard_brake_milli * 4
                    + inverse_headway_milli * 3
                    + int(max_concurrent) * 100
                    + int(lane_changes) * 250
                )
                window_sha256 = _window_semantic_sha256(connection, start_ms, end_ms)
                identity = {
                    "normalization_evidence_sha256": normalization_lock[
                        "normalization_evidence_sha256"
                    ],
                    "start_time_ms": start_ms,
                    "end_time_ms_exclusive": end_ms,
                    "source_window_sha256": window_sha256,
                    "actor_ids": actors,
                }
                identity_sha256 = object_sha256(identity)
                candidates.append(
                    {
                        "candidate_id": f"ngsim:{start_ms}:{identity_sha256[:16]}",
                        **identity,
                        "row_count": int(row_count),
                        "actor_count": int(actor_count),
                        "risk_features": {
                            "minimum_positive_time_headway_ms": headway_ms,
                            "minimum_acceleration_milli_mps2": int(round(min_accel * 1000)),
                            "hard_brake_magnitude_milli_mps2": hard_brake_milli,
                            "max_concurrent_actors": int(max_concurrent),
                            "lane_change_count": int(lane_changes),
                        },
                        "risk_score_milli": risk_score_milli,
                        "hazard_kind": (
                            hazard_context.get("hazard_kind")
                            if hazard_context is not None
                            else "no_following_conflict_observed"
                        ),
                        "hazard_context": hazard_context,
                        "window_semantics": window_semantics,
                    }
                )
            start_ms += stride_ms
    candidates.sort(
        key=lambda item: (
            -int(item["risk_score_milli"]),
            int(item["start_time_ms"]),
            str(item["candidate_id"]),
        )
    )
    selected = candidates[:limit]
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
    report: dict[str, Any] = {
        "schema_version": "ngsim_mining_report_v1",
        "mining_recipe_version": MINING_RECIPE_VERSION,
        "hazard_semantics_version": "ngsim_lead_braking_hazard_v1",
        "window_semantics_version": "ngsim_half_open_phase_window_v1",
        "normalization_evidence_sha256": normalization_lock["normalization_evidence_sha256"],
        "window_ms": window_ms,
        "stride_ms": stride_ms,
        "min_actors": min_actors,
        "candidate_count_before_limit": len(candidates),
        "candidates": selected,
    }
    report["mining_evidence_sha256"] = object_sha256(report)
    write_json_exclusive(output_path, report)
    return report


def mine_phase_windows(
    database_path: Path,
    normalization_lock: Mapping[str, Any],
    output_path: Path,
    *,
    window_ms: int = 40_000,
    stride_ms: int = 5_000,
    limit: int = 20,
    min_actors: int = 2,
    min_prevention_ms: int = 15_000,
    min_recovery_ms: int = 20_000,
) -> dict[str, Any]:
    """Mine long, source-grounded braking phases rather than short buckets.

    The ordinary miner ranks the minimum headway in each fixed bucket.  That
    is useful for a smoke slice but can choose a conflict at the first tick,
    leaving no long supervisory window.  This path anchors a candidate on a
    logged lead-vehicle braking row, requires the ego/lead pair to have been
    observable for ``min_prevention_ms``, and keeps a post-boundary recovery
    tail in the same locked source window.
    """
    if output_path.exists():
        raise FileExistsError("ngsim_mining_output_exists")
    verify_normalization_lock(database_path, normalization_lock)
    if (
        window_ms <= 0
        or stride_ms <= 0
        or window_ms % 100
        or stride_ms % 100
        or min_prevention_ms < 5_000
        or min_recovery_ms < 20_000
        or window_ms < min_prevention_ms + min_recovery_ms + 100
    ):
        raise ValueError("ngsim_phase_window_parameters_invalid")
    if limit <= 0 or min_actors <= 0:
        raise ValueError("ngsim_mining_limit_invalid")

    candidates: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        bounds = connection.execute(
            "SELECT MIN(timestamp_ms), MAX(timestamp_ms) FROM states"
        ).fetchone()
        if bounds is None or bounds[0] is None or bounds[1] is None:
            raise ValueError("ngsim_mining_database_empty")
        first_ms, last_ms = int(bounds[0]), int(bounds[1])
        # Join the logged braking rows to a follower observed at the exact
        # same 100-ms sample.  Actor and pair first timestamps are computed
        # from the complete normalized table, not from a truncated bucket.
        phase_rows = connection.execute(
            """
            WITH actor_first AS (
                SELECT actor_id, MIN(timestamp_ms) AS first_ms
                  FROM states GROUP BY actor_id
            ), pair_first AS (
                SELECT actor_id, preceding_actor_id AS conflict_id,
                       MIN(timestamp_ms) AS first_ms
                  FROM states
                 WHERE preceding_actor_id IS NOT NULL
                 GROUP BY actor_id, preceding_actor_id
            ), brake_onsets AS (
                SELECT *
                  FROM (
                      SELECT states.*,
                             LAG(acceleration_mps2) OVER (
                                 PARTITION BY actor_id ORDER BY timestamp_ms
                             ) AS previous_acceleration_mps2
                        FROM states
                  )
                 WHERE acceleration_mps2 <= ?
                   AND (
                       previous_acceleration_mps2 IS NULL
                       OR previous_acceleration_mps2 > ?
                   )
            ), brakes AS (
                SELECT actor_id AS conflict_id, timestamp_ms,
                       local_x_m, local_y_m, speed_mps, acceleration_mps2,
                       lane_id, length_m, width_m, source_row_number,
                       source_row_sha256
                  FROM (
                      SELECT brake_onsets.*,
                             ROW_NUMBER() OVER (
                                 PARTITION BY actor_id, (timestamp_ms / 10000)
                                 ORDER BY acceleration_mps2, timestamp_ms,
                                          source_row_number
                             ) AS brake_rank
                        FROM brake_onsets
                  )
                 WHERE brake_rank = 1
            ), phase_candidates AS (
                SELECT follower.actor_id AS ego_id,
                   brakes.conflict_id AS conflict_id,
                   brakes.timestamp_ms AS hazard_event_ms,
                   brakes.local_x_m AS hazard_local_x_m,
                   brakes.local_y_m AS hazard_local_y_m,
                   brakes.speed_mps AS hazard_speed_mps,
                   brakes.acceleration_mps2 AS hazard_acceleration_mps2,
                   brakes.lane_id AS hazard_lane_id,
                   brakes.length_m AS hazard_length_m,
                   brakes.width_m AS hazard_width_m,
                   brakes.source_row_number AS hazard_source_row_number,
                   brakes.source_row_sha256 AS hazard_source_row_sha256,
                   follower.time_headway_s AS headway_at_brake,
                   pair_first.first_ms AS first_observable_ms,
                   MAX(ego_first.first_ms, conflict_first.first_ms)
                       AS actor_start_ms
              FROM brakes
              JOIN states AS follower
                ON follower.timestamp_ms = brakes.timestamp_ms
               AND follower.preceding_actor_id = brakes.conflict_id
              JOIN pair_first
                ON pair_first.actor_id = follower.actor_id
               AND pair_first.conflict_id = brakes.conflict_id
              JOIN actor_first AS ego_first
                ON ego_first.actor_id = follower.actor_id
              JOIN actor_first AS conflict_first
                ON conflict_first.actor_id = brakes.conflict_id
             WHERE brakes.timestamp_ms - pair_first.first_ms >= ?
            ), ranked AS (
                SELECT phase_candidates.*, ROW_NUMBER() OVER (
                    PARTITION BY (actor_start_ms / ?)
                    ORDER BY hazard_acceleration_mps2, hazard_event_ms,
                             ego_id, conflict_id
                ) AS phase_rank
                  FROM phase_candidates
            )
            SELECT * FROM ranked
             WHERE phase_rank <= 16
             ORDER BY hazard_event_ms, ego_id, hazard_source_row_number
            """,
            (
                HARD_BRAKE_THRESHOLD_MPS2,
                HARD_BRAKE_THRESHOLD_MPS2,
                min_prevention_ms,
                window_ms,
            ),
        ).fetchall()
        seen_phase: set[tuple[str, str, int]] = set()
        for phase in phase_rows:
            start_ms = int(phase["actor_start_ms"])
            end_ms = start_ms + window_ms
            if start_ms < first_ms or end_ms > last_ms + 100:
                continue
            phase_key = (
                str(phase["ego_id"]),
                str(phase["conflict_id"]),
                start_ms,
            )
            if phase_key in seen_phase:
                continue
            seen_phase.add(phase_key)
            risk = connection.execute(
                """
                SELECT timestamp_ms, local_x_m, local_y_m, speed_mps,
                       acceleration_mps2, lane_id, length_m, width_m,
                       source_row_number, source_row_sha256, time_headway_s
                  FROM states
                 WHERE actor_id = ? AND preceding_actor_id = ?
                   AND timestamp_ms >= ? AND timestamp_ms < ?
                   AND time_headway_s > 0
                 ORDER BY time_headway_s, timestamp_ms, source_row_number
                 LIMIT 1
                """,
                (
                    str(phase["ego_id"]),
                    str(phase["conflict_id"]),
                    int(phase["hazard_event_ms"]),
                    end_ms,
                ),
            ).fetchone()
            if risk is None:
                continue
            risk_ms = int(risk["timestamp_ms"])
            response_ms = risk_ms - int(phase["hazard_event_ms"])
            recovery_ms = end_ms - risk_ms
            if response_ms < 100 or recovery_ms < min_recovery_ms:
                continue
            aggregate = connection.execute(
                """
                SELECT COUNT(*) AS row_count,
                       COUNT(DISTINCT actor_id) AS actor_count,
                       MIN(CASE WHEN time_headway_s > 0 THEN time_headway_s END)
                           AS min_headway,
                       MIN(acceleration_mps2) AS min_acceleration,
                       MIN(timestamp_ms) AS observed_start_ms,
                       MAX(timestamp_ms) AS observed_end_ms
                  FROM states
                 WHERE timestamp_ms >= ? AND timestamp_ms < ?
                """,
                (start_ms, end_ms),
            ).fetchone()
            if aggregate is None or int(aggregate["actor_count"] or 0) < min_actors:
                continue
            observed_start = int(aggregate["observed_start_ms"])
            observed_end = int(aggregate["observed_end_ms"])
            configured_complete = bool(observed_start == start_ms and observed_end == end_ms - 100)
            if not configured_complete:
                continue
            actors = [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT DISTINCT actor_id FROM states
                     WHERE timestamp_ms >= ? AND timestamp_ms < ?
                     ORDER BY actor_id
                    """,
                    (start_ms, end_ms),
                )
            ]
            max_concurrent = int(
                connection.execute(
                    """
                    SELECT COALESCE(MAX(n), 0) FROM (
                        SELECT COUNT(*) AS n FROM states
                         WHERE timestamp_ms >= ? AND timestamp_ms < ?
                         GROUP BY timestamp_ms
                    )
                    """,
                    (start_ms, end_ms),
                ).fetchone()[0]
                or 0
            )
            lane_changes = int(
                connection.execute(
                    """
                    SELECT COALESCE(SUM(lanes - 1), 0) FROM (
                        SELECT COUNT(DISTINCT lane_id) AS lanes
                          FROM states
                         WHERE timestamp_ms >= ? AND timestamp_ms < ?
                         GROUP BY actor_id
                    )
                    """,
                    (start_ms, end_ms),
                ).fetchone()[0]
                or 0
            )
            headway_ms = int(round(float(aggregate["min_headway"] or 0.0) * 1000))
            min_accel = float(aggregate["min_acceleration"] or 0.0)
            hard_brake_milli = max(0, int(round(-min_accel * 1000)))
            inverse_headway_milli = int(1_000_000 / max(1, headway_ms)) if headway_ms else 0
            identity = {
                "normalization_evidence_sha256": normalization_lock[
                    "normalization_evidence_sha256"
                ],
                "start_time_ms": start_ms,
                "end_time_ms_exclusive": end_ms,
                "source_window_sha256": _window_semantic_sha256(connection, start_ms, end_ms),
                "actor_ids": actors,
            }
            hazard_context = {
                "hazard_kind": "lead_vehicle_braking",
                "ego_actor_id": str(phase["ego_id"]),
                "conflict_actor_id": str(phase["conflict_id"]),
                "first_observable_time_ms": int(phase["first_observable_ms"]),
                "latest_preventive_command_time_ms": int(phase["hazard_event_ms"]),
                "hazard_event_time_ms": int(phase["hazard_event_ms"]),
                "hazard_event": {
                    "actor_id": str(phase["conflict_id"]),
                    "timestamp_ms": int(phase["hazard_event_ms"]),
                    "local_x_m": float(phase["hazard_local_x_m"]),
                    "local_y_m": float(phase["hazard_local_y_m"]),
                    "speed_mps": float(phase["hazard_speed_mps"]),
                    "acceleration_mps2": float(phase["hazard_acceleration_mps2"]),
                    "lane_id": int(phase["hazard_lane_id"]),
                    "length_m": float(phase["hazard_length_m"]),
                    "width_m": float(phase["hazard_width_m"]),
                    "source_row_number": int(phase["hazard_source_row_number"]),
                    "source_row_sha256": str(phase["hazard_source_row_sha256"]),
                },
                "risk_boundary_proxy_time_ms": risk_ms,
                "risk_boundary_proxy": "minimum_positive_logged_time_headway_after_braking",
                "recovery_window_end_time_ms": end_ms,
                "supervisory_prevention_window_ms": int(phase["hazard_event_ms"])
                - int(phase["first_observable_ms"]),
                "protective_response_window_ms": response_ms,
                "recovery_window_ms": recovery_ms,
                "source_row_number": int(risk["source_row_number"]),
                "source_row_sha256": str(risk["source_row_sha256"]),
                "minimum_time_headway_ms": int(round(float(risk["time_headway_s"]) * 1000)),
                "phase_window_complete": True,
            }
            identity_sha = object_sha256(identity)
            candidates.append(
                {
                    "candidate_id": f"ngsim:{start_ms}:{identity_sha[:16]}",
                    **identity,
                    "row_count": int(aggregate["row_count"]),
                    "actor_count": int(aggregate["actor_count"]),
                    "risk_features": {
                        "minimum_positive_time_headway_ms": headway_ms,
                        "minimum_acceleration_milli_mps2": int(round(min_accel * 1000)),
                        "hard_brake_magnitude_milli_mps2": hard_brake_milli,
                        "max_concurrent_actors": max_concurrent,
                        "lane_change_count": lane_changes,
                    },
                    "risk_score_milli": hard_brake_milli * 4
                    + inverse_headway_milli * 3
                    + max_concurrent * 100
                    + lane_changes * 250,
                    "hazard_kind": "lead_vehicle_braking",
                    "hazard_context": hazard_context,
                    "window_semantics": {
                        "interval_convention": "[start_time_ms,end_time_ms_exclusive)",
                        "requested_window_ms": window_ms,
                        "sample_period_ms": 100,
                        "expected_timestamp_count": window_ms // 100,
                        "observed_start_time_ms": observed_start,
                        "observed_end_time_ms_inclusive": observed_end,
                        "observed_coverage_ms": observed_end - observed_start + 100,
                        "configured_window_complete": True,
                        "phase_anchor": "logged_lead_vehicle_braking_row",
                        "minimum_prevention_ms": min_prevention_ms,
                    },
                }
            )
    candidates.sort(
        key=lambda item: (
            -int(item["risk_score_milli"]),
            int(item["start_time_ms"]),
            str(item["candidate_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(
            int(candidate["start_time_ms"]) < int(row["end_time_ms_exclusive"])
            and int(row["start_time_ms"]) < int(candidate["end_time_ms_exclusive"])
            for row in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: (int(item["start_time_ms"]), str(item["candidate_id"])))
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
    report: dict[str, Any] = {
        "schema_version": "ngsim_mining_report_v1",
        "mining_recipe_version": MINING_RECIPE_VERSION,
        "hazard_semantics_version": "ngsim_lead_braking_hazard_v1",
        "window_semantics_version": "ngsim_half_open_phase_window_v1",
        "normalization_evidence_sha256": normalization_lock["normalization_evidence_sha256"],
        "window_ms": window_ms,
        "stride_ms": stride_ms,
        "min_actors": min_actors,
        "phase_window_mining": {
            "anchor": "logged_lead_vehicle_braking_row",
            "minimum_prevention_ms": min_prevention_ms,
            "minimum_recovery_ms": min_recovery_ms,
            "non_overlapping_selection": True,
        },
        "candidate_count_before_limit": len(candidates),
        "candidates": selected,
    }
    report["mining_evidence_sha256"] = object_sha256(report)
    write_json_exclusive(output_path, report)
    return report


def mine_lane_change_windows(
    database_path: Path,
    normalization_lock: Mapping[str, Any],
    output_path: Path,
    *,
    window_ms: int = 60_000,
    stride_ms: int = 5_000,
    limit: int = 32,
    min_prevention_ms: int = 15_000,
    min_recovery_ms: int = 20_000,
    max_conflict_gap_m: float = 35.0,
) -> dict[str, Any]:
    """Mine logged adjacent-lane transitions that create a cut-in conflict.

    This miner never invents an actor, trajectory, or lane change.  A
    transition is accepted only when the normalized source contains stable
    lanes immediately before and after the transition, a distinct actor in
    the target lane, and a later logged gap/time-headway boundary.  The
    resulting window keeps a source-derived prevention interval and a recovery
    tail so it can exercise the supervisory layer rather than only the shield.
    """
    if output_path.exists():
        raise FileExistsError("ngsim_mining_output_exists")
    verify_normalization_lock(database_path, normalization_lock)
    if (
        window_ms < min_prevention_ms + min_recovery_ms + 500
        or window_ms <= 0
        or stride_ms <= 0
        or window_ms % 100
        or stride_ms % 100
        or min_prevention_ms < 5_000
        or min_recovery_ms < 20_000
        or max_conflict_gap_m <= 0
    ):
        raise ValueError("ngsim_lane_change_window_parameters_invalid")
    if limit <= 0:
        raise ValueError("ngsim_mining_limit_invalid")

    def valid_lane(value: Any) -> bool:
        lane = int(value)
        # 9999 and large values are NGSIM missing/invalid lane sentinels.
        return 0 <= lane < 100

    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT actor_id, timestamp_ms, local_x_m, local_y_m, speed_mps,
                   acceleration_mps2, lane_id, length_m, width_m,
                   source_row_number, source_row_sha256, time_headway_s,
                   space_headway_m
              FROM states
             ORDER BY actor_id, timestamp_ms
            """
        ).fetchall()
        if not rows:
            raise ValueError("ngsim_mining_database_empty")
        by_actor: dict[str, list[sqlite3.Row]] = {}
        by_time_lane: dict[tuple[int, int], list[sqlite3.Row]] = {}
        timestamp_counts: dict[int, int] = {}
        for row in rows:
            actor_id = str(row["actor_id"])
            by_actor.setdefault(actor_id, []).append(row)
            timestamp = int(row["timestamp_ms"])
            timestamp_counts[timestamp] = timestamp_counts.get(timestamp, 0) + 1
            if valid_lane(row["lane_id"]):
                by_time_lane.setdefault((int(row["timestamp_ms"]), int(row["lane_id"])), []).append(
                    row
                )

        transitions: list[tuple[sqlite3.Row, int, int]] = []
        for actor_id in sorted(by_actor):
            actor_rows = by_actor[actor_id]
            for index in range(5, len(actor_rows) - 5):
                previous = actor_rows[index - 1]
                current = actor_rows[index]
                following = actor_rows[index + 1]
                previous_lane = int(previous["lane_id"])
                current_lane = int(current["lane_id"])
                if not (
                    valid_lane(previous_lane)
                    and valid_lane(current_lane)
                    and abs(current_lane - previous_lane) == 1
                    and int(following["lane_id"]) == current_lane
                ):
                    continue
                if any(
                    int(actor_rows[index - offset]["lane_id"]) != previous_lane
                    for offset in range(1, 6)
                ) or any(
                    int(actor_rows[index + offset]["lane_id"]) != current_lane
                    for offset in range(1, 6)
                ):
                    continue
                transitions.append((current, previous_lane, current_lane))

        candidates: list[dict[str, Any]] = []
        for transition, previous_lane, target_lane in transitions:
            transition_ms = int(transition["timestamp_ms"])
            start_ms = transition_ms - min_prevention_ms
            end_ms = start_ms + window_ms
            target_rows = [
                row
                for row in by_time_lane.get((transition_ms, target_lane), [])
                if str(row["actor_id"]) != str(transition["actor_id"])
            ]
            if not target_rows or start_ms < min(timestamp_counts):
                continue
            changer_id = str(transition["actor_id"])
            changer_rows = by_actor.get(changer_id, [])
            if not changer_rows:
                continue
            changer_window = [
                row for row in changer_rows if start_ms <= int(row["timestamp_ms"]) < end_ms
            ]
            if not changer_window or int(changer_window[0]["timestamp_ms"]) != start_ms:
                continue
            conflict: sqlite3.Row | None = None
            ego_window: list[sqlite3.Row] = []
            for target in sorted(
                target_rows,
                key=lambda row: (
                    abs(float(row["local_y_m"]) - float(transition["local_y_m"])),
                    str(row["actor_id"]),
                ),
            ):
                target_gap = abs(
                    float(target["local_y_m"]) - float(transition["local_y_m"])
                ) - 0.5 * (float(target["length_m"]) + float(transition["length_m"]))
                if target_gap <= 0 or target_gap > max_conflict_gap_m:
                    continue
                target_ego_rows = by_actor.get(str(target["actor_id"]), [])
                target_ego_window = [
                    row for row in target_ego_rows if start_ms <= int(row["timestamp_ms"]) < end_ms
                ]
                if not target_ego_window or int(target_ego_window[0]["timestamp_ms"]) != start_ms:
                    continue
                conflict = target
                ego_window = target_ego_window
                break
            if conflict is None:
                continue
            ego_id = str(conflict["actor_id"])
            changer_by_time = {int(row["timestamp_ms"]): row for row in changer_window}
            risk_row: sqlite3.Row | None = None
            min_gap_m = math.inf
            for ego_row in ego_window:
                timestamp_ms = int(ego_row["timestamp_ms"])
                if timestamp_ms <= transition_ms:
                    continue
                changer_at_time = changer_by_time.get(timestamp_ms)
                if changer_at_time is None:
                    continue
                gap_m = abs(
                    float(ego_row["local_y_m"]) - float(changer_at_time["local_y_m"])
                ) - 0.5 * (float(ego_row["length_m"]) + float(changer_at_time["length_m"]))
                min_gap_m = min(min_gap_m, gap_m)
                headway_s = float(ego_row["time_headway_s"])
                if gap_m <= 8.0 or (headway_s > 0.0 and headway_s <= 1.5):
                    risk_row = ego_row
                    break
            if risk_row is None:
                continue
            risk_ms = int(risk_row["timestamp_ms"])
            if risk_ms - transition_ms < 100 or end_ms - risk_ms < min_recovery_ms:
                continue
            aggregate = connection.execute(
                """
                SELECT COUNT(*), COUNT(DISTINCT actor_id), MIN(timestamp_ms),
                       MAX(timestamp_ms)
                  FROM states
                 WHERE timestamp_ms >= ? AND timestamp_ms < ?
                """,
                (start_ms, end_ms),
            ).fetchone()
            if aggregate is None or int(aggregate[0] or 0) == 0:
                continue
            observed_start = int(aggregate[2])
            observed_end = int(aggregate[3])
            if observed_start != start_ms or observed_end != end_ms - 100:
                continue
            actors = [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT DISTINCT actor_id FROM states
                     WHERE timestamp_ms >= ? AND timestamp_ms < ?
                     ORDER BY actor_id
                    """,
                    (start_ms, end_ms),
                )
            ]
            max_concurrent = max(
                timestamp_counts.get(timestamp, 0) for timestamp in range(start_ms, end_ms, 100)
            )
            identity = {
                "normalization_evidence_sha256": normalization_lock[
                    "normalization_evidence_sha256"
                ],
                "start_time_ms": start_ms,
                "end_time_ms_exclusive": end_ms,
                "source_window_sha256": _window_semantic_sha256(connection, start_ms, end_ms),
                "actor_ids": actors,
                "hazard_actor_ids": [changer_id, ego_id],
                "hazard_event_time_ms": transition_ms,
            }
            hazard_context = {
                "hazard_kind": "lane_change_conflict",
                "ego_actor_id": ego_id,
                "conflict_actor_id": changer_id,
                "first_observable_time_ms": start_ms,
                "latest_preventive_command_time_ms": transition_ms,
                "hazard_event_time_ms": transition_ms,
                "hazard_event": {
                    "actor_id": changer_id,
                    "timestamp_ms": transition_ms,
                    "local_x_m": float(transition["local_x_m"]),
                    "local_y_m": float(transition["local_y_m"]),
                    "speed_mps": float(transition["speed_mps"]),
                    "acceleration_mps2": float(transition["acceleration_mps2"]),
                    "lane_id": target_lane,
                    "previous_lane_id": previous_lane,
                    "target_lane_id": target_lane,
                    "length_m": float(transition["length_m"]),
                    "width_m": float(transition["width_m"]),
                    "source_row_number": int(transition["source_row_number"]),
                    "source_row_sha256": str(transition["source_row_sha256"]),
                },
                "risk_boundary": {
                    "actor_id": ego_id,
                    "timestamp_ms": risk_ms,
                    "gap_m": round(float(min_gap_m), 6),
                    "time_headway_s": float(risk_row["time_headway_s"]),
                    "source_row_number": int(risk_row["source_row_number"]),
                    "source_row_sha256": str(risk_row["source_row_sha256"]),
                },
                "risk_boundary_proxy_time_ms": risk_ms,
                "risk_boundary_proxy": "logged_cut_in_gap_or_time_headway",
                "recovery_window_end_time_ms": end_ms,
                "supervisory_prevention_window_ms": transition_ms - start_ms,
                "protective_response_window_ms": risk_ms - transition_ms,
                "recovery_window_ms": end_ms - risk_ms,
                "source_row_number": int(risk_row["source_row_number"]),
                "source_row_sha256": str(risk_row["source_row_sha256"]),
                "minimum_gap_m": round(float(min_gap_m), 6),
                "minimum_time_headway_ms": int(round(float(risk_row["time_headway_s"]) * 1000)),
                "phase_window_complete": True,
            }
            identity_sha = object_sha256(identity)
            candidates.append(
                {
                    "candidate_id": f"ngsim:{start_ms}:{identity_sha[:16]}",
                    **identity,
                    "row_count": int(aggregate[0]),
                    "actor_count": len(actors),
                    "risk_features": {
                        "minimum_positive_time_headway_ms": int(
                            round(float(risk_row["time_headway_s"]) * 1000)
                        ),
                        "minimum_gap_milli": int(round(float(min_gap_m) * 1000)),
                        "minimum_acceleration_milli_mps2": int(
                            round(float(transition["acceleration_mps2"]) * 1000)
                        ),
                        "hard_brake_magnitude_milli_mps2": max(
                            0, int(round(-float(transition["acceleration_mps2"]) * 1000))
                        ),
                        "lane_change_count": 1,
                        "max_concurrent_actors": max_concurrent,
                    },
                    "risk_score_milli": int(max(0.0, (max_conflict_gap_m - min_gap_m) * 1000))
                    + int(
                        round(
                            1_000_000 / max(1, int(round(float(risk_row["time_headway_s"]) * 1000)))
                        )
                    )
                    + 250,
                    "hazard_kind": "lane_change_conflict",
                    "hazard_context": hazard_context,
                    "window_semantics": {
                        "interval_convention": "[start_time_ms,end_time_ms_exclusive)",
                        "requested_window_ms": window_ms,
                        "sample_period_ms": 100,
                        "expected_timestamp_count": window_ms // 100,
                        "observed_start_time_ms": observed_start,
                        "observed_end_time_ms_inclusive": observed_end,
                        "observed_coverage_ms": observed_end - observed_start + 100,
                        "configured_window_complete": True,
                        "phase_anchor": "logged_adjacent_lane_transition",
                        "minimum_prevention_ms": min_prevention_ms,
                        "minimum_recovery_ms": min_recovery_ms,
                    },
                }
            )

    candidates.sort(
        key=lambda item: (
            -int(item["risk_score_milli"]),
            int(item["start_time_ms"]),
            str(item["candidate_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(
            int(candidate["start_time_ms"]) < int(row["end_time_ms_exclusive"])
            and int(row["start_time_ms"]) < int(candidate["end_time_ms_exclusive"])
            for row in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: (int(item["start_time_ms"]), str(item["candidate_id"])))
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
    report: dict[str, Any] = {
        "schema_version": "ngsim_mining_report_v1",
        "mining_recipe_version": LANE_CHANGE_MINING_RECIPE_VERSION,
        "hazard_semantics_version": "ngsim_lane_change_conflict_hazard_v1",
        "window_semantics_version": "ngsim_half_open_lane_change_window_v1",
        "normalization_evidence_sha256": normalization_lock["normalization_evidence_sha256"],
        "window_ms": window_ms,
        "stride_ms": stride_ms,
        "min_actors": 2,
        "lane_change_window_mining": {
            "anchor": "logged_adjacent_lane_transition",
            "minimum_prevention_ms": min_prevention_ms,
            "minimum_recovery_ms": min_recovery_ms,
            "max_conflict_gap_m": max_conflict_gap_m,
            "non_overlapping_selection": True,
        },
        "candidate_count_before_limit": len(candidates),
        "candidates": selected,
    }
    report["mining_evidence_sha256"] = object_sha256(report)
    write_json_exclusive(output_path, report)
    return report


def mine_time_headway_windows(
    database_path: Path,
    normalization_lock: Mapping[str, Any],
    output_path: Path,
    *,
    window_ms: int = 60_000,
    limit: int = 32,
    min_prevention_ms: int = 15_000,
    min_recovery_ms: int = 20_000,
    event_headway_s: float = 1.5,
    risk_headway_s: float = 0.8,
) -> dict[str, Any]:
    """Mine logged short-following gaps without inventing a brake event.

    This is deliberately separate from ``mine_phase_windows``: the latter
    anchors on a source-recorded leader brake, while this recipe anchors on a
    follower crossing a short time-headway boundary while the leader is not
    hard-braking.  Both the crossing and the later tighter boundary must be
    present in the normalized source rows, so the resulting scenario remains
    source-native and auditable.
    """
    if output_path.exists():
        raise FileExistsError("ngsim_mining_output_exists")
    verify_normalization_lock(database_path, normalization_lock)
    if (
        window_ms < min_prevention_ms + min_recovery_ms + 100
        or window_ms <= 0
        or window_ms % 100
        or min_prevention_ms < 5_000
        or min_recovery_ms < 20_000
        or limit <= 0
        or not (0.0 < risk_headway_s < event_headway_s)
    ):
        raise ValueError("ngsim_headway_window_parameters_invalid")

    def _event_payload(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "actor_id": str(row["actor_id"]),
            "timestamp_ms": int(row["timestamp_ms"]),
            "local_x_m": float(row["local_x_m"]),
            "local_y_m": float(row["local_y_m"]),
            "speed_mps": float(row["speed_mps"]),
            "acceleration_mps2": float(row["acceleration_mps2"]),
            "lane_id": int(row["lane_id"]),
            "length_m": float(row["length_m"]),
            "width_m": float(row["width_m"]),
            "source_row_number": int(row["source_row_number"]),
            "source_row_sha256": str(row["source_row_sha256"]),
        }

    candidates: list[dict[str, Any]] = []
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        bounds = connection.execute(
            "SELECT MIN(timestamp_ms), MAX(timestamp_ms) FROM states"
        ).fetchone()
        if bounds is None or bounds[0] is None or bounds[1] is None:
            raise ValueError("ngsim_mining_database_empty")
        first_ms, last_ms = int(bounds[0]), int(bounds[1])

        def _pair_groups() -> Iterable[tuple[str, str, list[sqlite3.Row]]]:
            current_key: tuple[str, str] | None = None
            current_rows: list[sqlite3.Row] = []
            cursor = connection.execute(
                """
                SELECT actor_id, preceding_actor_id, timestamp_ms,
                       local_x_m, local_y_m, speed_mps, acceleration_mps2,
                       lane_id, length_m, width_m, source_row_number,
                       source_row_sha256, time_headway_s
                  FROM states
                 WHERE preceding_actor_id IS NOT NULL
                 ORDER BY actor_id, preceding_actor_id, timestamp_ms
                """
            )
            for row in cursor:
                key = (str(row["actor_id"]), str(row["preceding_actor_id"]))
                if current_key is not None and key != current_key:
                    yield current_key[0], current_key[1], current_rows
                    current_rows = []
                current_key = key
                current_rows.append(row)
            if current_key is not None and current_rows:
                yield current_key[0], current_key[1], current_rows

        for ego_id, leader_id, rows in _pair_groups():
            if len(rows) < 2:
                continue
            pair_first_ms = int(rows[0]["timestamp_ms"])
            for index, row in enumerate(rows):
                current_headway = float(row["time_headway_s"])
                previous_headway = float(rows[index - 1]["time_headway_s"]) if index else math.inf
                if not (
                    0.0 < current_headway <= event_headway_s
                    and previous_headway > event_headway_s
                    and int(row["timestamp_ms"]) - pair_first_ms >= min_prevention_ms
                ):
                    continue
                event_ms = int(row["timestamp_ms"])
                start_ms = event_ms - min_prevention_ms
                end_ms = start_ms + window_ms
                if start_ms < first_ms or end_ms > last_ms + 100:
                    continue
                risk_row = next(
                    (
                        later
                        for later in rows[index + 1 :]
                        if int(later["timestamp_ms"]) >= event_ms + 100
                        and int(later["timestamp_ms"]) < end_ms - min_recovery_ms
                        and 0.0 < float(later["time_headway_s"]) <= risk_headway_s
                    ),
                    None,
                )
                if risk_row is None:
                    continue
                risk_ms = int(risk_row["timestamp_ms"])
                leader_row = connection.execute(
                    """
                    SELECT acceleration_mps2
                      FROM states
                     WHERE actor_id = ? AND timestamp_ms = ?
                    """,
                    (leader_id, event_ms),
                ).fetchone()
                if leader_row is None or float(leader_row[0]) <= HARD_BRAKE_THRESHOLD_MPS2:
                    continue
                initial = connection.execute(
                    """
                    SELECT actor_id
                      FROM states
                     WHERE timestamp_ms = ? AND actor_id IN (?, ?)
                    """,
                    (start_ms, ego_id, leader_id),
                ).fetchall()
                if {str(value[0]) for value in initial} != {ego_id, leader_id}:
                    continue
                aggregate = connection.execute(
                    """
                    SELECT COUNT(*) AS row_count,
                           COUNT(DISTINCT actor_id) AS actor_count,
                           MIN(timestamp_ms) AS observed_start_ms,
                           MAX(timestamp_ms) AS observed_end_ms,
                           MIN(CASE WHEN time_headway_s > 0 THEN time_headway_s END)
                             AS min_headway,
                           MIN(acceleration_mps2) AS min_acceleration
                      FROM states
                     WHERE timestamp_ms >= ? AND timestamp_ms < ?
                    """,
                    (start_ms, end_ms),
                ).fetchone()
                if aggregate is None or int(aggregate["row_count"] or 0) == 0:
                    continue
                if (
                    int(aggregate["observed_start_ms"]) != start_ms
                    or int(aggregate["observed_end_ms"]) != end_ms - 100
                ):
                    continue
                actors = [
                    str(value[0])
                    for value in connection.execute(
                        """
                        SELECT DISTINCT actor_id FROM states
                         WHERE timestamp_ms >= ? AND timestamp_ms < ?
                         ORDER BY actor_id
                        """,
                        (start_ms, end_ms),
                    )
                ]
                max_concurrent = int(
                    connection.execute(
                        """
                        SELECT COALESCE(MAX(n), 0) FROM (
                            SELECT COUNT(*) AS n FROM states
                             WHERE timestamp_ms >= ? AND timestamp_ms < ?
                             GROUP BY timestamp_ms
                        )
                        """,
                        (start_ms, end_ms),
                    ).fetchone()[0]
                    or 0
                )
                min_headway_ms = int(round(float(risk_row["time_headway_s"]) * 1000))
                identity = {
                    "normalization_evidence_sha256": normalization_lock[
                        "normalization_evidence_sha256"
                    ],
                    "start_time_ms": start_ms,
                    "end_time_ms_exclusive": end_ms,
                    "source_window_sha256": _window_semantic_sha256(connection, start_ms, end_ms),
                    "actor_ids": actors,
                    "hazard_actor_ids": [ego_id, leader_id],
                    "hazard_event_time_ms": event_ms,
                }
                hazard_context = {
                    "hazard_kind": "minimum_time_headway_conflict",
                    "ego_actor_id": ego_id,
                    "conflict_actor_id": leader_id,
                    "first_observable_time_ms": pair_first_ms,
                    "latest_preventive_command_time_ms": event_ms,
                    "hazard_event_time_ms": event_ms,
                    "hazard_event": _event_payload(row),
                    "risk_boundary": {
                        **_event_payload(risk_row),
                        "time_headway_s": float(risk_row["time_headway_s"]),
                    },
                    "risk_boundary_proxy_time_ms": risk_ms,
                    "risk_boundary_proxy": "logged_short_time_headway_boundary",
                    "recovery_window_end_time_ms": end_ms,
                    "supervisory_prevention_window_ms": event_ms - start_ms,
                    "protective_response_window_ms": risk_ms - event_ms,
                    "recovery_window_ms": end_ms - risk_ms,
                    "source_row_number": int(risk_row["source_row_number"]),
                    "source_row_sha256": str(risk_row["source_row_sha256"]),
                    "minimum_time_headway_ms": min_headway_ms,
                    "phase_window_complete": True,
                }
                identity_sha = object_sha256(identity)
                candidates.append(
                    {
                        "candidate_id": f"ngsim:{start_ms}:{identity_sha[:16]}",
                        **identity,
                        "row_count": int(aggregate["row_count"]),
                        "actor_count": int(aggregate["actor_count"]),
                        "risk_features": {
                            "minimum_positive_time_headway_ms": min_headway_ms,
                            "minimum_acceleration_milli_mps2": int(
                                round(float(aggregate["min_acceleration"] or 0.0) * 1000)
                            ),
                            "hard_brake_magnitude_milli_mps2": max(
                                0,
                                int(round(-float(aggregate["min_acceleration"] or 0.0) * 1000)),
                            ),
                            "max_concurrent_actors": max_concurrent,
                            "lane_change_count": 0,
                        },
                        "risk_score_milli": int(1_000_000 / max(1, min_headway_ms))
                        + max_concurrent * 100,
                        "hazard_kind": "minimum_time_headway_conflict",
                        "hazard_context": hazard_context,
                        "window_semantics": {
                            "interval_convention": "[start_time_ms,end_time_ms_exclusive)",
                            "requested_window_ms": window_ms,
                            "sample_period_ms": 100,
                            "expected_timestamp_count": window_ms // 100,
                            "observed_start_time_ms": start_ms,
                            "observed_end_time_ms_inclusive": end_ms - 100,
                            "observed_coverage_ms": window_ms,
                            "configured_window_complete": True,
                            "phase_anchor": "logged_short_time_headway_boundary",
                            "minimum_prevention_ms": min_prevention_ms,
                            "minimum_recovery_ms": min_recovery_ms,
                        },
                    }
                )

    candidates.sort(
        key=lambda item: (
            -int(item["risk_score_milli"]),
            int(item["start_time_ms"]),
            str(item["candidate_id"]),
        )
    )
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(
            int(candidate["start_time_ms"]) < int(row["end_time_ms_exclusive"])
            and int(row["start_time_ms"]) < int(candidate["end_time_ms_exclusive"])
            for row in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) >= limit:
            break
    selected.sort(key=lambda item: (int(item["start_time_ms"]), str(item["candidate_id"])))
    for rank, candidate in enumerate(selected, start=1):
        candidate["rank"] = rank
    report: dict[str, Any] = {
        "schema_version": "ngsim_mining_report_v1",
        "mining_recipe_version": HEADWAY_MINING_RECIPE_VERSION,
        "hazard_semantics_version": "ngsim_minimum_time_headway_conflict_hazard_v1",
        "window_semantics_version": "ngsim_half_open_headway_window_v1",
        "normalization_evidence_sha256": normalization_lock["normalization_evidence_sha256"],
        "window_ms": window_ms,
        "stride_ms": 0,
        "min_actors": 2,
        "time_headway_window_mining": {
            "anchor": "logged_short_time_headway_boundary",
            "event_headway_s": event_headway_s,
            "risk_headway_s": risk_headway_s,
            "minimum_prevention_ms": min_prevention_ms,
            "minimum_recovery_ms": min_recovery_ms,
            "leader_hard_brake_excluded": True,
            "non_overlapping_selection": True,
        },
        "candidate_count_before_limit": len(candidates),
        "candidates": selected,
    }
    report["mining_evidence_sha256"] = object_sha256(report)
    write_json_exclusive(output_path, report)
    return report


def verify_mining_report(report: Mapping[str, Any], normalization_lock: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "ngsim_mining_report_v1":
        raise ValueError("ngsim_mining_report_schema_mismatch")
    if report.get("normalization_evidence_sha256") != normalization_lock.get(
        "normalization_evidence_sha256"
    ):
        raise ValueError("ngsim_mining_normalization_mismatch")
    unsigned = dict(report)
    evidence = str(unsigned.pop("mining_evidence_sha256", ""))
    if evidence != object_sha256(unsigned):
        raise ValueError("ngsim_mining_evidence_mismatch")
    mining_recipe = str(report.get("mining_recipe_version") or "")
    if mining_recipe not in SUPPORTED_MINING_RECIPE_VERSIONS:
        raise ValueError("ngsim_mining_recipe_unsupported")
    if mining_recipe == MINING_RECIPE_VERSION:
        expected_hazard_semantics = "ngsim_lead_vehicle_braking_hazard_v1"
        # Preserve the original spelling as an accepted legacy alias in old
        # reports; newly generated reports use the canonical value above.
        if report.get("hazard_semantics_version") not in {
            expected_hazard_semantics,
            "ngsim_lead_braking_hazard_v1",
        }:
            raise ValueError("ngsim_mining_hazard_semantics_mismatch")
        if report.get("window_semantics_version") != "ngsim_half_open_phase_window_v1":
            raise ValueError("ngsim_mining_window_semantics_mismatch")
        phase_config = dict(report.get("phase_window_mining") or {})
        phase_min_prevention = int(phase_config.get("minimum_prevention_ms") or 0)
        phase_min_recovery = int(phase_config.get("minimum_recovery_ms") or 0)
    elif mining_recipe == LANE_CHANGE_MINING_RECIPE_VERSION:
        if report.get("hazard_semantics_version") != "ngsim_lane_change_conflict_hazard_v1":
            raise ValueError("ngsim_mining_hazard_semantics_mismatch")
        if report.get("window_semantics_version") != "ngsim_half_open_lane_change_window_v1":
            raise ValueError("ngsim_mining_window_semantics_mismatch")
        lane_config = dict(report.get("lane_change_window_mining") or {})
        phase_min_prevention = int(lane_config.get("minimum_prevention_ms") or 0)
        phase_min_recovery = int(lane_config.get("minimum_recovery_ms") or 0)
    else:
        if report.get("hazard_semantics_version") != (
            "ngsim_minimum_time_headway_conflict_hazard_v1"
        ):
            raise ValueError("ngsim_mining_hazard_semantics_mismatch")
        if report.get("window_semantics_version") != "ngsim_half_open_headway_window_v1":
            raise ValueError("ngsim_mining_window_semantics_mismatch")
        headway_config = dict(report.get("time_headway_window_mining") or {})
        phase_min_prevention = int(headway_config.get("minimum_prevention_ms") or 0)
        phase_min_recovery = int(headway_config.get("minimum_recovery_ms") or 0)
    for candidate in report.get("candidates") or []:
        semantics = dict(candidate.get("window_semantics") or {})
        requested_ms = int(semantics.get("requested_window_ms") or 0)
        if requested_ms != int(candidate["end_time_ms_exclusive"]) - int(
            candidate["start_time_ms"]
        ):
            raise ValueError("ngsim_candidate_window_duration_mismatch")
        if semantics.get("interval_convention") != ("[start_time_ms,end_time_ms_exclusive)"):
            raise ValueError("ngsim_candidate_window_interval_mismatch")
        hazard = dict(candidate.get("hazard_context") or {})
        if candidate.get("hazard_kind") != (
            hazard.get("hazard_kind") if hazard else "no_following_conflict_observed"
        ):
            raise ValueError("ngsim_candidate_hazard_kind_mismatch")
        if hazard.get("phase_window_complete"):
            event = dict(hazard.get("hazard_event") or {})
            allowed_kind = {
                MINING_RECIPE_VERSION: "lead_vehicle_braking",
                LANE_CHANGE_MINING_RECIPE_VERSION: "lane_change_conflict",
                HEADWAY_MINING_RECIPE_VERSION: "minimum_time_headway_conflict",
            }[mining_recipe]
            if (
                candidate.get("hazard_kind") != allowed_kind
                or not semantics.get("configured_window_complete")
                or int(hazard.get("supervisory_prevention_window_ms") or 0) < 5_000
                or int(hazard.get("protective_response_window_ms") or 0) < 100
                or int(hazard.get("recovery_window_ms") or 0) < 20_000
                or len(str(event.get("source_row_sha256") or "")) != 64
                or len(str(hazard.get("source_row_sha256") or "")) != 64
            ):
                raise ValueError("ngsim_candidate_phase_window_not_complete")
            if mining_recipe == LANE_CHANGE_MINING_RECIPE_VERSION and (
                int(event.get("target_lane_id", -1)) < 0
                or int(event.get("previous_lane_id", -1)) < 0
                or not isinstance(hazard.get("risk_boundary"), Mapping)
                or len(str(dict(hazard.get("risk_boundary") or {}).get("source_row_sha256") or ""))
                != 64
            ):
                raise ValueError("ngsim_lane_change_context_incomplete")
            if mining_recipe == HEADWAY_MINING_RECIPE_VERSION and (
                not isinstance(hazard.get("risk_boundary"), Mapping)
                or len(str(dict(hazard.get("risk_boundary") or {}).get("source_row_sha256") or ""))
                != 64
            ):
                raise ValueError("ngsim_headway_context_incomplete")
            if (
                phase_min_prevention
                and int(hazard.get("supervisory_prevention_window_ms") or 0) < phase_min_prevention
            ):
                raise ValueError("ngsim_phase_candidate_prevention_window_too_short")
            if (
                phase_min_recovery
                and int(hazard.get("recovery_window_ms") or 0) < phase_min_recovery
            ):
                raise ValueError("ngsim_phase_candidate_recovery_window_too_short")


def _copy(path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Source bundles are immutable and often contain a 0.5--1 GB SQLite
        # payload.  A same-filesystem hard link keeps each bundle path
        # independently auditable without multiplying the cache footprint;
        # cross-device exports fall back to a byte-for-byte copy.
        os.link(path, destination)
    except OSError:
        shutil.copyfile(path, destination)


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _source_event_from_row(
    *,
    candidate: Mapping[str, Any],
    row: tuple[Any, ...],
    lane_map: Mapping[int, int],
    kind: str,
    phase: str,
    control_duration_s: float | None = None,
) -> dict[str, Any]:
    start_ms = int(candidate["start_time_ms"])
    source_time_ms = int(row[1])
    relative_time_ms = source_time_ms - start_ms
    if relative_time_ms < 0:
        raise ValueError("ngsim_source_event_before_candidate_window")
    event: dict[str, Any] = {
        "event_id": (f"{candidate['candidate_id']}:{kind}:{row[0]}:{source_time_ms}"),
        "kind": kind,
        "hazard_kind": str(candidate["hazard_kind"]),
        "phase": phase,
        "actor_id": str(row[0]),
        "source_time_ms": source_time_ms,
        "relative_time_ms": relative_time_ms,
        "trigger_tick": relative_time_ms // SOURCE_EVENT_TICK_MS,
        "trigger_offset_ms_within_tick": relative_time_ms % SOURCE_EVENT_TICK_MS,
        "route_position_m": float(row[3]),
        "lateral_position_m": float(row[2]),
        "lane_index": int(lane_map[int(row[6])]),
        "speed_mps": max(0.0, float(row[4])),
        "acceleration_mps2": float(row[5]),
        "hidden": False,
        "source_provenance": {
            "candidate_id": str(candidate["candidate_id"]),
            "normalization_evidence_sha256": str(candidate["normalization_evidence_sha256"]),
            "source_window_sha256": str(candidate["source_window_sha256"]),
            "source_row_number": int(row[9]),
            "source_row_sha256": str(row[10]),
        },
    }
    if control_duration_s is not None:
        event["control_duration_s"] = max(0.1, float(control_duration_s))
    event_sha256 = object_sha256(event)
    event["source_event_sha256"] = event_sha256
    event["source_event_ids"] = [event_sha256]
    return event


def _derive_source_events(
    candidate: Mapping[str, Any],
    rows: list[tuple[Any, ...]],
    lane_map: Mapping[int, int],
) -> list[dict[str, Any]]:
    hazard = dict(candidate.get("hazard_context") or {})
    if not bool(hazard.get("phase_window_complete")):
        return []
    conflict_actor_id = str(hazard.get("conflict_actor_id") or "")
    event = dict(hazard.get("hazard_event") or {})
    event_sha256 = str(event.get("source_row_sha256") or "")
    event_actor_id = str(event.get("actor_id") or conflict_actor_id)
    event_row = next(
        (row for row in rows if str(row[0]) == event_actor_id and str(row[10]) == event_sha256),
        None,
    )
    if event_row is None:
        raise ValueError("ngsim_hazard_source_row_missing_from_window")
    risk_time_ms = int(hazard["risk_boundary_proxy_time_ms"])
    if candidate.get("hazard_kind") == "lead_vehicle_braking":
        conflict_rows = [row for row in rows if str(row[0]) == conflict_actor_id]
        if not conflict_rows:
            raise ValueError("ngsim_conflict_actor_rows_missing_from_window")
        risk_row = min(
            conflict_rows,
            key=lambda row: (abs(int(row[1]) - risk_time_ms), int(row[1])),
        )
        return [
            _source_event_from_row(
                candidate=candidate,
                row=event_row,
                lane_map=lane_map,
                kind="lead_vehicle_braking",
                phase="protective_response",
                control_duration_s=(
                    int(hazard.get("protective_response_window_ms") or 5000) / 1000.0
                ),
            ),
            _source_event_from_row(
                candidate=candidate,
                row=risk_row,
                lane_map=lane_map,
                kind="actor_state_update",
                phase="risk_boundary",
            ),
        ]
    if candidate.get("hazard_kind") == "lane_change_conflict":
        risk = dict(hazard.get("risk_boundary") or {})
        risk_sha256 = str(risk.get("source_row_sha256") or hazard.get("source_row_sha256") or "")
        ego_actor_id = str(hazard.get("ego_actor_id") or "")
        risk_rows = [
            row for row in rows if str(row[0]) == ego_actor_id and str(row[10]) == risk_sha256
        ]
        if not risk_rows:
            raise ValueError("ngsim_lane_change_risk_source_row_missing_from_window")
        risk_row = risk_rows[0]
        return [
            _source_event_from_row(
                candidate=candidate,
                row=event_row,
                lane_map=lane_map,
                kind="lane_change_conflict",
                phase="hazard_realized",
                control_duration_s=(
                    int(hazard.get("protective_response_window_ms") or 5000) / 1000.0
                ),
            ),
            _source_event_from_row(
                candidate=candidate,
                row=risk_row,
                lane_map=lane_map,
                kind="cut_in_gap_boundary",
                phase="risk_boundary",
            ),
        ]
    if candidate.get("hazard_kind") == "minimum_time_headway_conflict":
        leader_actor_id = str(hazard.get("conflict_actor_id") or "")
        event_time_ms = int(hazard.get("hazard_event_time_ms") or 0)
        leader_rows = [row for row in rows if str(row[0]) == leader_actor_id]
        if not leader_rows:
            raise ValueError("ngsim_headway_risk_source_row_missing_from_window")
        leader_event_row = min(
            leader_rows,
            key=lambda row: (abs(int(row[1]) - event_time_ms), int(row[1])),
        )
        return [
            _source_event_from_row(
                candidate=candidate,
                row=leader_event_row,
                lane_map=lane_map,
                kind="short_time_headway_boundary",
                phase="hazard_realized",
                control_duration_s=(
                    int(hazard.get("protective_response_window_ms") or 5000) / 1000.0
                ),
            ),
        ]
    raise ValueError("ngsim_phase_complete_candidate_hazard_unsupported")


def _derive_runtime_assets(
    database_path: Path,
    mining_report: Mapping[str, Any],
    output_dir: Path,
    *,
    candidate_id: str | None = None,
) -> dict[str, list[str]]:
    """Mechanically export one mined log window and its simulator siblings."""
    candidates = list(mining_report.get("candidates") or [])
    if not candidates:
        raise ValueError("ngsim_materialization_requires_mined_candidate")
    if candidate_id:
        candidate = next(
            (value for value in candidates if str(value.get("candidate_id")) == candidate_id),
            None,
        )
        if candidate is None:
            raise ValueError("ngsim_materialization_candidate_id_not_found")
    else:
        candidate = next(
            (
                value
                for value in candidates
                if bool((value.get("hazard_context") or {}).get("phase_window_complete"))
            ),
            candidates[0],
        )
    start_ms = int(candidate["start_time_ms"])
    end_ms = int(candidate["end_time_ms_exclusive"])
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        recording_row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'recording_id'"
        ).fetchone()
        rows = connection.execute(
            """
            SELECT actor_id, timestamp_ms, local_x_m, local_y_m, speed_mps,
                   acceleration_mps2, lane_id, length_m, width_m,
                   source_row_number, source_row_sha256
              FROM states
             WHERE timestamp_ms >= ? AND timestamp_ms < ?
             ORDER BY timestamp_ms, actor_id
            """,
            (start_ms, end_ms),
        ).fetchall()
    if recording_row is None or not str(recording_row[0]):
        raise ValueError("ngsim_materialization_recording_identity_missing")
    recording_id = str(recording_row[0])
    if not rows:
        raise ValueError("ngsim_materialization_window_empty")
    lane_ids = sorted({int(row[6]) for row in rows})
    lane_map = {lane_id: index for index, lane_id in enumerate(lane_ids)}
    initial_ms = min(int(row[1]) for row in rows)
    initial = [row for row in rows if int(row[1]) == initial_ms]
    initial_by_id = {str(row[0]): row for row in initial}
    hazard_context = dict(candidate.get("hazard_context") or {})
    phase_complete = bool(hazard_context.get("phase_window_complete"))
    if phase_complete:
        ego_id = str(hazard_context.get("ego_actor_id") or "")
        conflict_actor_id = str(hazard_context.get("conflict_actor_id") or "")
        if ego_id not in initial_by_id:
            raise ValueError("ngsim_materialization_candidate_ego_missing_at_window_start")
        if conflict_actor_id not in initial_by_id:
            raise ValueError("ngsim_materialization_conflict_actor_missing_at_window_start")
        ego_selection = "phase_complete_hazard_context"
    else:
        actor_row_counts: dict[str, int] = {}
        for row in rows:
            actor_id = str(row[0])
            actor_row_counts[actor_id] = actor_row_counts.get(actor_id, 0) + 1
        ego_id = min(
            initial_by_id,
            key=lambda actor_id: (-actor_row_counts.get(actor_id, 0), actor_id),
        )
        conflict_actor_id = ""
        ego_selection = "longest_observed_actor_at_window_start_diagnostic"

    # The full normalized window is retained for provenance and log replay,
    # but a SUMO reactive sibling must not try to insert hundreds of vehicles
    # at the same simulation instant.  That can delay the ego departure and
    # make a valid source event actor leave before its trigger tick.  Keep the
    # ego, the declared conflict actor, and nearby initial actors in the
    # executable route while recording the exact closed-loop subset.
    reactive_actor_ids = set(initial_by_id)
    if phase_complete:
        ego_position = float(initial_by_id[ego_id][3])
        nearby = sorted(
            initial_by_id,
            key=lambda actor_id: (
                abs(float(initial_by_id[actor_id][3]) - ego_position),
                actor_id,
            ),
        )
        reactive_actor_ids = {ego_id, conflict_actor_id}
        reactive_actor_ids.update(
            actor_id
            for actor_id in nearby
            if actor_id not in reactive_actor_ids
            and abs(float(initial_by_id[actor_id][3]) - ego_position) <= 200.0
        )
        reactive_actor_ids = set(sorted(reactive_actor_ids)[:16])
        reactive_actor_ids.update({ego_id, conflict_actor_id})

    def state(row: tuple[Any, ...], *, ego: bool) -> dict[str, Any]:
        return {
            "vehicle_id" if ego else "actor_id": str(row[0]),
            "route_position_m": float(row[3]),
            "lateral_position_m": float(row[2]),
            "lane_index": lane_map[int(row[6])],
            "speed_mps": max(0.0, float(row[4])),
            "length_m": float(row[7]),
            "width_m": float(row[8]),
        }

    max_position = max(float(row[3]) for row in rows)
    min_position = min(float(row[3]) for row in rows)
    speed_limit_mps = 30.0
    source_window_seconds = max(
        0.0,
        (int(candidate["end_time_ms_exclusive"]) - int(candidate["start_time_ms"])) / 1000.0,
    )
    runway_margin_m = 100.0
    reactive_runway_m = max(200.0, speed_limit_mps * source_window_seconds + runway_margin_m)
    route_length = max(100.0, max_position - min_position + reactive_runway_m)
    source_events = _derive_source_events(candidate, rows, lane_map)
    fixture = {
        "schema_version": "ngsim_runtime_fixture_v1",
        "derivation": {
            "candidate_id": candidate["candidate_id"],
            "source_window_sha256": candidate["source_window_sha256"],
            "ego_actor_id": ego_id,
            "conflict_actor_id": conflict_actor_id or None,
            "ego_selection": ego_selection,
            "candidate_hazard_context_bound": phase_complete,
            "hazard_kind": candidate["hazard_kind"],
            "source_event_tick_ms": SOURCE_EVENT_TICK_MS,
            "window_semantics": candidate["window_semantics"],
            "reactive_actor_ids": sorted(reactive_actor_ids),
            "reactive_runway_m": reactive_runway_m,
            "source_window_seconds": source_window_seconds,
            "mode": "logged_initial_state_with_backend_reactive_rollout",
        },
        "lane_count": len(lane_ids),
        "lane_width_m": 3.6,
        "route_length_m": route_length,
        "speed_limit_mps": speed_limit_mps,
        "ego": state(initial_by_id[ego_id], ego=True),
        "actors": [
            state(row, ego=False)
            for actor_id, row in sorted(initial_by_id.items())
            if actor_id != ego_id
        ],
        "source_events": source_events,
    }
    write_json_exclusive(output_dir / "runtime/fixture.json", fixture)
    log_lines = "".join(
        json.dumps(
            {
                "actor_id": row[0],
                "timestamp_ms": row[1],
                "local_x_m": row[2],
                "local_y_m": row[3],
                "speed_mps": row[4],
                "acceleration_mps2": row[5],
                "lane_id": row[6],
                "source_row_number": row[9],
                "source_row_sha256": row[10],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )
    _write_text_exclusive(output_dir / "runtime/log_replay.jsonl", log_lines)

    lane_xml = "".join(
        f'<lane id="ngsim_0_{index}" index="{index}" speed="30" length="{route_length:.3f}" '
        f'shape="0.000,{index * 3.6:.3f} {route_length:.3f},{index * 3.6:.3f}"/>'
        for index in range(len(lane_ids))
    )
    network = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<net version="1.20" junctionCornerDetail="5">\n'
        f'<location netOffset="0.00,0.00" convBoundary="0.00,0.00,{route_length:.3f},'
        f'{max(0.0, (len(lane_ids) - 1) * 3.6):.3f}" origBoundary="0.00,0.00,'
        f'{route_length:.3f},{max(0.0, (len(lane_ids) - 1) * 3.6):.3f}" projParameter="-"/>\n'
        f'<edge id="ngsim" from="start" to="end" priority="1">{lane_xml}</edge>\n'
        '<junction id="start" type="dead_end" x="0.00" y="0.00" incLanes="" intLanes=""/>\n'
        f'<junction id="end" type="dead_end" x="{route_length:.3f}" y="0.00" '
        f'incLanes="{" ".join(f"ngsim_0_{i}" for i in range(len(lane_ids)))}" intLanes=""/>\n'
        "</net>\n"
    )
    _write_text_exclusive(output_dir / "sumo/network.net.xml", network)
    source_event_actor_ids = {
        str(event.get("actor_id") or "")
        for event in source_events
        if str(event.get("actor_id") or "")
    }
    required_reactive_ids = {ego_id, *source_event_actor_ids}
    if phase_complete:
        required_reactive_ids.add(conflict_actor_id)
    ordered_reactive_ids = sorted(
        (actor_id for actor_id in reactive_actor_ids if actor_id in initial_by_id),
        key=lambda actor_id: (
            actor_id not in required_reactive_ids,
            lane_map[int(initial_by_id[actor_id][6])],
            -float(initial_by_id[actor_id][3]),
            actor_id,
        ),
    )
    vehicles = "".join(
        f'<vehicle id="{actor_id}" type="passenger" route="source_route" depart="0" '
        f'departLane="{lane_map[int(row[6])]}" departPos="{max(0.0, float(row[3]) - min_position):.3f}" '
        # SUMO may keep a zero-speed vehicle in the loaded queue instead of
        # inserting it. A tiny positive insertion speed preserves the source
        # row while allowing the backend to set the exact locked speed after
        # the vehicle is present.
        f'departSpeed="{max(0.1, float(row[4])):.3f}"/>\n'
        for actor_id in ordered_reactive_ids
        for row in (initial_by_id[actor_id],)
    )
    routes = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n'
        '<vType id="passenger" vClass="passenger" carFollowModel="Krauss"/>\n'
        '<route id="source_route" edges="ngsim"/>\n'
        f"{vehicles}</routes>\n"
    )
    _write_text_exclusive(output_dir / "sumo/routes.rou.xml", routes)
    sumocfg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n<configuration>\n<input>\n'
        '<net-file value="network.net.xml"/>\n<route-files value="routes.rou.xml"/>\n'
        '</input>\n<time><step-length value="0.1"/></time>\n</configuration>\n'
    )
    _write_text_exclusive(output_dir / "sumo/run.sumocfg", sumocfg)
    write_json_exclusive(
        output_dir / "runtime/reactive_sumo.json",
        {
            "schema_version": "ngsim_reactive_sumo_sibling_v1",
            "sumo_config": "sumo/run.sumocfg",
            "ego_vehicle_id": ego_id,
            "reactive_actor_ids": sorted(reactive_actor_ids),
            "source_window_sha256": candidate["source_window_sha256"],
            "admission_status": "held_pending_live_sumo_replay_validation",
        },
    )

    commonroad_path = output_dir / "commonroad/scenario.xml"
    commonroad_report = output_dir / "commonroad/export.json"
    commonroad_payload = output_dir / ".commonroad-export-input.json"
    write_json_exclusive(
        commonroad_payload,
        {
            "lane_count": len(lane_ids),
            "route_length_m": route_length,
            "recording_id": recording_id,
            "source": f"NGSIM {NGSIM_DATASET_ID}",
            "source_window_sha256": candidate["source_window_sha256"],
        },
    )
    child_env = dict(os.environ)
    child_env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    repo_root = Path(__file__).resolve().parents[3]
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), child_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    pilot_python = Path(
        child_env.get("OPERATE_AUTONOMOUS_DRIVING_PYTHON")
        or repo_root / ".venv-autonomous-driving/bin/python"
    )
    if not pilot_python.is_file():
        raise ValueError("ngsim_commonroad_isolated_runtime_missing")
    try:
        completed = subprocess.run(  # nosec B603
            (
                str(pilot_python),
                "-m",
                "domains.autonomous_driving.data.commonroad_export",
                "--payload",
                str(commonroad_payload),
                "--output",
                str(commonroad_path),
                "--report",
                str(commonroad_report),
            ),
            check=False,
            cwd=repo_root,
            env=child_env,
            capture_output=True,
            text=True,
        )
    finally:
        commonroad_payload.unlink(missing_ok=True)
    if completed.returncode != 0:
        raise ValueError("ngsim_commonroad_export_failed: " + completed.stderr.strip())
    return {
        "runtime": ["runtime/fixture.json", "runtime/log_replay.jsonl"],
        "reactive": [
            "runtime/reactive_sumo.json",
            "sumo/network.net.xml",
            "sumo/routes.rou.xml",
            "sumo/run.sumocfg",
        ],
        "commonroad": ["commonroad/scenario.xml", "commonroad/export.json"],
    }


def _license_notice(*, review_status: str, review_basis: str | None) -> str:
    if review_status not in {"pending_metadata_discrepancy", "approved"}:
        raise ValueError("ngsim_license_review_status_invalid")
    review = (
        "Review status: approved. The 3.0 versus 4.0 metadata discrepancy was "
        f"reviewed under {review_basis or 'an external release attestation'}.\n"
        if review_status == "approved"
        else (
            "Review status: pending human/legal disposition of the 3.0 versus 4.0 "
            "metadata discrepancy. This bundle contains a bounded, mechanically "
            "converted extract; do not treat this notice as redistribution approval.\n"
        )
    )
    return (
        "# NGSIM source attribution\n\n"
        f"{NGSIM_ATTRIBUTION}\n\n"
        f"Dataset identifier: {NGSIM_DATASET_ID}\n\n"
        "License declarations preserved from the locked official metadata:\n"
        f"- dataset-level API: {NGSIM_PORTAL_LICENSE_ID} ({NGSIM_PORTAL_LICENSE_URL})\n"
        f"- Common Core custom field: {NGSIM_LICENSE_ID} ({NGSIM_LICENSE_URL})\n\n"
        f"{review}"
    )


def materialize_bundle(
    *,
    raw_path: Path,
    source_lock_path: Path,
    database_path: Path,
    normalization_lock_path: Path,
    mining_report_path: Path,
    output_dir: Path,
    candidate_id: str | None = None,
    license_review_status: str = "pending_metadata_discrepancy",
    license_review_basis: str | None = None,
) -> dict[str, Any]:
    """Build a self-verifying source bundle without replacing any target."""
    if output_dir.exists():
        raise FileExistsError("ngsim_bundle_output_exists")
    if database_path.name != "trajectories.sqlite3":
        raise ValueError("ngsim_bundle_database_name_must_be_trajectories.sqlite3")
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    normalization_lock = json.loads(normalization_lock_path.read_text(encoding="utf-8"))
    mining_report = json.loads(mining_report_path.read_text(encoding="utf-8"))
    verify_source_lock(raw_path, source_lock)
    verify_normalization_lock(database_path, normalization_lock)
    verify_mining_report(mining_report, normalization_lock)
    if mining_report.get("mining_recipe_version") not in SUPPORTED_MINING_RECIPE_VERSIONS:
        raise ValueError("ngsim_materialization_requires_supported_mining_recipe")
    from domains.autonomous_driving.seeds.from_ngsim import build_seed_records

    seeds = build_seed_records(
        mining_report,
        source_evidence_sha256=str(source_lock["source_evidence_sha256"]),
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.with_name(f".{output_dir.name}.tmp-{os.getpid()}")
    if temp_dir.exists():
        raise FileExistsError("ngsim_bundle_temp_exists")
    temp_dir.mkdir()
    try:
        bundled_raw = f"source/{raw_path.name}"
        destinations = {
            bundled_raw: raw_path,
            "source/source.lock.json": source_lock_path,
            "normalized/trajectories.sqlite3": database_path,
            "normalized/normalization.lock.json": normalization_lock_path,
            "mining/candidates.json": mining_report_path,
        }
        for relative, source in destinations.items():
            _copy(source, temp_dir / relative)
        write_json_exclusive(temp_dir / "seeds/seeds.json", seeds)
        derived_assets = _derive_runtime_assets(
            database_path,
            mining_report,
            temp_dir,
            candidate_id=candidate_id,
        )
        runtime_fixture = json.loads(
            (temp_dir / "runtime/fixture.json").read_text(encoding="utf-8")
        )
        runtime_source_events_sha256 = object_sha256(runtime_fixture.get("source_events") or [])
        if license_review_status not in {"pending_metadata_discrepancy", "approved"}:
            raise ValueError("ngsim_license_review_status_invalid")
        notice_name = (
            "NGSIM_LICENSE_REVIEW_APPROVED.md"
            if license_review_status == "approved"
            else "NGSIM_LICENSE_REVIEW_PENDING.md"
        )
        notice_path = temp_dir / "LICENSES" / notice_name
        notice_path.parent.mkdir(parents=True, exist_ok=True)
        notice_path.write_text(
            _license_notice(
                review_status=license_review_status,
                review_basis=license_review_basis,
            ),
            encoding="utf-8",
        )
        bundle_identity = {
            "source_evidence_sha256": source_lock["source_evidence_sha256"],
            "normalization_evidence_sha256": normalization_lock["normalization_evidence_sha256"],
            "mining_evidence_sha256": mining_report["mining_evidence_sha256"],
            "seed_set_sha256": seeds["seed_set_sha256"],
            "runtime_source_events_sha256": runtime_source_events_sha256,
        }
        bundle: dict[str, Any] = {
            "schema_version": "autonomous_driving_source_bundle_v1",
            "bundle_id": f"ngsim-{object_sha256(bundle_identity)[:20]}",
            "selected_candidate_id": str(
                (runtime_fixture.get("derivation") or {}).get("candidate_id") or ""
            ),
            "domain": "autonomous_driving",
            "source_kind": "naturalistic_vehicle_trajectory_log",
            "naturalistic": True,
            "source_dataset_id": NGSIM_DATASET_ID,
            "source_release": f"doi:{NGSIM_DOI}",
            "license_id": NGSIM_LICENSE_ID,
            "license_url": NGSIM_LICENSE_URL,
            "license_declarations": {
                "dataset_level": {
                    "id": NGSIM_PORTAL_LICENSE_ID,
                    "terms_url": NGSIM_PORTAL_LICENSE_URL,
                },
                "common_core_custom_field": {
                    "id": NGSIM_LICENSE_ID,
                    "terms_url": NGSIM_LICENSE_URL,
                },
            },
            "license_review_status": license_review_status,
            "license_review_basis": license_review_basis,
            "redistribution_class": "full_with_attribution_and_share_alike",
            "actor_provenance": {
                "normalized": "logged",
                "log_replay": "materialized_from_locked_normalized_rows",
                "future_reactive_rollout": "materialized_sumo_sibling_held_pending_live_validation",
            },
            "source_event_contract": {
                "schema_version": "ngsim_source_events_v1",
                "trigger_tick_ms": SOURCE_EVENT_TICK_MS,
                "event_count": len(runtime_fixture.get("source_events") or []),
                "runtime_source_events_sha256": runtime_source_events_sha256,
                "source_grounding": "normalized_sqlite_rows_and_source_row_sha256",
            },
            "admission_status": "held_pending_live_sumo_reactive_validation",
            "derived_assets": derived_assets,
            "evidence": bundle_identity,
            "source_contract": {
                "runtime_input": [
                    "normalized/trajectories.sqlite3",
                    "runtime/fixture.json",
                    "runtime/log_replay.jsonl",
                    "runtime/reactive_sumo.json",
                    "sumo/network.net.xml",
                    "sumo/routes.rou.xml",
                    "sumo/run.sumocfg",
                ],
                "derivation_input": [bundled_raw],
                "implementation_asset": [
                    "commonroad/scenario.xml",
                    "commonroad/export.json",
                ],
                "metadata": [
                    "source/source.lock.json",
                    "normalized/normalization.lock.json",
                    "mining/candidates.json",
                    "seeds/seeds.json",
                    "bundle.json",
                ],
                "license": [f"LICENSES/{notice_name}"],
            },
        }
        write_json_exclusive(temp_dir / "bundle.json", bundle)
        checksums = {
            path.relative_to(temp_dir).as_posix(): file_sha256(path)
            for path in sorted(temp_dir.rglob("*"))
            if path.is_file()
        }
        checksum_lines = "".join(
            f"{digest}  {relative}\n" for relative, digest in sorted(checksums.items())
        )
        (temp_dir / "checksums.sha256").write_text(checksum_lines, encoding="ascii")
        verify_bundle(temp_dir)
        os.rename(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return bundle


def _verify_reactive_route_order(fixture: dict[str, Any], route_path: Path) -> None:
    """Require source-critical vehicles to be inserted before background traffic."""
    derivation = dict(fixture.get("derivation") or {})
    if derivation.get("candidate_hazard_context_bound") is not True:
        return
    try:
        # This XML is generated locally above and is then checksum-locked in
        # the immutable bundle; no external XML payload reaches this parser.
        route_root = ET.parse(route_path).getroot()  # nosec B314
    except ET.ParseError as exc:
        raise ValueError("ngsim_bundle_reactive_route_xml_invalid") from exc
    vehicle_ids = [str(row.attrib.get("id") or "") for row in route_root.findall("vehicle")]
    if not vehicle_ids or len(vehicle_ids) != len(set(vehicle_ids)):
        raise ValueError("ngsim_bundle_reactive_route_vehicle_ids_invalid")
    required = {
        str(derivation.get("ego_actor_id") or ""),
        str(derivation.get("conflict_actor_id") or ""),
        *{
            str(event.get("actor_id") or "")
            for event in fixture.get("source_events") or []
            if isinstance(event, dict)
        },
    }
    required.discard("")
    if not required.issubset(vehicle_ids):
        raise ValueError("ngsim_bundle_reactive_route_required_actor_missing")
    required_indices = {vehicle_ids.index(actor_id) for actor_id in required}
    optional_indices = {
        index for index, actor_id in enumerate(vehicle_ids) if actor_id not in required
    }
    if optional_indices and max(required_indices) >= min(optional_indices):
        raise ValueError("ngsim_bundle_reactive_route_required_actor_order")


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Verify exact bundle payloads and all nested evidence chains."""
    bundle_path = bundle_dir / "bundle.json"
    checksums_path = bundle_dir / "checksums.sha256"
    if not bundle_path.is_file() or not checksums_path.is_file():
        raise ValueError("ngsim_bundle_manifest_missing")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != "autonomous_driving_source_bundle_v1":
        raise ValueError("ngsim_bundle_schema_mismatch")
    expected: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64 or relative in expected:
            raise ValueError("ngsim_bundle_checksum_manifest_invalid")
        expected[relative] = digest
    observed_files = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file() and path != checksums_path
    }
    if observed_files != set(expected):
        raise ValueError("ngsim_bundle_file_set_mismatch")
    for relative, digest in expected.items():
        if file_sha256(bundle_dir / relative) != digest:
            raise ValueError("ngsim_bundle_file_hash_mismatch")
    source_lock = json.loads((bundle_dir / "source/source.lock.json").read_text(encoding="utf-8"))
    source_plan = NGSIMSourcePlan.from_dict(source_lock.get("source_plan") or {})
    expected_commonroad_map = {
        "us-101": "NGSIM_US101",
        "i-80": "NGSIM_I80",
        "lankershim": "NGSIM_LANKERSHIM",
        "peachtree": "NGSIM_PEACHTREE",
    }.get(source_plan.recording_id)
    commonroad_export = json.loads(
        (bundle_dir / "commonroad/export.json").read_text(encoding="utf-8")
    )
    if (
        expected_commonroad_map is None
        or commonroad_export.get("schema_version") != "ngsim_commonroad_export_v2"
        or commonroad_export.get("admission_status") != "format_roundtrip_only"
        or commonroad_export.get("map_name") != expected_commonroad_map
        or commonroad_export.get("geometry_mode") != "synthetic_straight_lane_proxy"
        or commonroad_export.get("core_validator_eligible") is not False
        or int(commonroad_export.get("lanelet_count") or 0) <= 0
    ):
        raise ValueError("ngsim_bundle_commonroad_export_contract_mismatch")
    normalization_lock = json.loads(
        (bundle_dir / "normalized/normalization.lock.json").read_text(encoding="utf-8")
    )
    mining_report = json.loads((bundle_dir / "mining/candidates.json").read_text(encoding="utf-8"))
    raw_name = str(source_lock.get("raw_file_name") or "")
    verify_source_lock(bundle_dir / "source" / raw_name, source_lock)
    verify_normalization_lock(bundle_dir / "normalized/trajectories.sqlite3", normalization_lock)
    verify_mining_report(mining_report, normalization_lock)
    fixture = json.loads((bundle_dir / "runtime/fixture.json").read_text(encoding="utf-8"))
    derivation = dict(fixture.get("derivation") or {})
    selected_candidate_id = str(bundle.get("selected_candidate_id") or "")
    if selected_candidate_id and selected_candidate_id != str(derivation.get("candidate_id") or ""):
        raise ValueError("ngsim_bundle_selected_candidate_mismatch")
    candidates = [dict(value) for value in mining_report.get("candidates") or []]
    selected = next(
        (
            value
            for value in candidates
            if value.get("candidate_id") == derivation.get("candidate_id")
        ),
        None,
    )
    if selected is None or derivation.get("source_window_sha256") != selected.get(
        "source_window_sha256"
    ):
        raise ValueError("ngsim_bundle_runtime_candidate_identity_mismatch")
    if commonroad_export.get("source_window_sha256") != derivation.get("source_window_sha256"):
        raise ValueError("ngsim_bundle_commonroad_export_contract_mismatch")
    fixture_ego_id = str((fixture.get("ego") or {}).get("vehicle_id") or "")
    if fixture_ego_id != str(derivation.get("ego_actor_id") or ""):
        raise ValueError("ngsim_bundle_runtime_ego_identity_mismatch")
    hazard_context = dict(selected.get("hazard_context") or {})
    phase_complete = bool(hazard_context.get("phase_window_complete"))
    if derivation.get("candidate_hazard_context_bound") is not phase_complete:
        raise ValueError("ngsim_bundle_runtime_phase_binding_mismatch")
    if phase_complete:
        if fixture_ego_id != str(hazard_context.get("ego_actor_id") or "") or str(
            derivation.get("conflict_actor_id") or ""
        ) != str(hazard_context.get("conflict_actor_id") or ""):
            raise ValueError("ngsim_bundle_runtime_hazard_identity_mismatch")
        if derivation.get("ego_selection") != "phase_complete_hazard_context":
            raise ValueError("ngsim_bundle_runtime_hazard_selection_mismatch")
        fixture_actor_ids = {
            str(value.get("actor_id") or "") for value in fixture.get("actors") or []
        }
        if str(derivation.get("conflict_actor_id") or "") not in fixture_actor_ids:
            raise ValueError("ngsim_bundle_runtime_conflict_actor_missing")
        reactive_actor_ids = {
            str(value) for value in derivation.get("reactive_actor_ids") or [] if str(value)
        }
        if reactive_actor_ids:
            required_reactive_ids = {
                fixture_ego_id,
                str(derivation.get("conflict_actor_id") or ""),
                *{
                    str(event.get("actor_id") or "")
                    for event in fixture.get("source_events") or []
                    if isinstance(event, dict)
                },
            }
            if not required_reactive_ids.issubset(reactive_actor_ids):
                raise ValueError("ngsim_bundle_runtime_reactive_actor_binding_mismatch")
    elif (
        derivation.get("ego_selection") != "longest_observed_actor_at_window_start_diagnostic"
        or derivation.get("conflict_actor_id") is not None
    ):
        raise ValueError("ngsim_bundle_runtime_diagnostic_identity_undeclared")
    _verify_reactive_route_order(fixture, bundle_dir / "sumo/routes.rou.xml")
    mining_recipe = str(mining_report.get("mining_recipe_version") or "")
    if mining_recipe in SUPPORTED_MINING_RECIPE_VERSIONS:
        if derivation.get("hazard_kind") != selected.get("hazard_kind"):
            raise ValueError("ngsim_bundle_runtime_hazard_kind_mismatch")
        if derivation.get("window_semantics") != selected.get("window_semantics"):
            raise ValueError("ngsim_bundle_runtime_window_semantics_mismatch")
        if derivation.get("source_event_tick_ms") != SOURCE_EVENT_TICK_MS:
            raise ValueError("ngsim_bundle_runtime_source_event_tick_mismatch")
        with sqlite3.connect(
            f"file:{bundle_dir / 'normalized/trajectories.sqlite3'}?mode=ro",
            uri=True,
        ) as connection:
            runtime_rows = connection.execute(
                """
                SELECT actor_id, timestamp_ms, local_x_m, local_y_m, speed_mps,
                       acceleration_mps2, lane_id, length_m, width_m,
                       source_row_number, source_row_sha256
                  FROM states
                 WHERE timestamp_ms >= ? AND timestamp_ms < ?
                 ORDER BY timestamp_ms, actor_id
                """,
                (
                    int(selected["start_time_ms"]),
                    int(selected["end_time_ms_exclusive"]),
                ),
            ).fetchall()
        runtime_lane_ids = sorted({int(row[6]) for row in runtime_rows})
        runtime_lane_map = {lane_id: index for index, lane_id in enumerate(runtime_lane_ids)}
        expected_source_events = _derive_source_events(
            selected,
            runtime_rows,
            runtime_lane_map,
        )
        if fixture.get("source_events") != expected_source_events:
            raise ValueError("ngsim_bundle_runtime_source_events_mismatch")
    from domains.autonomous_driving.seeds.from_ngsim import build_seed_records

    seed_set = json.loads((bundle_dir / "seeds/seeds.json").read_text(encoding="utf-8"))
    expected_seed_set = build_seed_records(
        mining_report,
        source_evidence_sha256=str(source_lock["source_evidence_sha256"]),
    )
    if seed_set != expected_seed_set:
        raise ValueError("ngsim_bundle_seed_set_mismatch")
    runtime_source_events = fixture.get("source_events") or []
    runtime_source_events_sha256 = object_sha256(runtime_source_events)
    if mining_recipe in SUPPORTED_MINING_RECIPE_VERSIONS:
        source_event_contract = dict(bundle.get("source_event_contract") or {})
        if source_event_contract != {
            "schema_version": "ngsim_source_events_v1",
            "trigger_tick_ms": SOURCE_EVENT_TICK_MS,
            "event_count": len(runtime_source_events),
            "runtime_source_events_sha256": runtime_source_events_sha256,
            "source_grounding": "normalized_sqlite_rows_and_source_row_sha256",
        }:
            raise ValueError("ngsim_bundle_source_event_contract_mismatch")
    evidence = bundle.get("evidence") or {}
    expected_evidence = {
        "source_evidence_sha256": source_lock["source_evidence_sha256"],
        "normalization_evidence_sha256": normalization_lock["normalization_evidence_sha256"],
        "mining_evidence_sha256": mining_report["mining_evidence_sha256"],
        "seed_set_sha256": seed_set["seed_set_sha256"],
    }
    if mining_recipe in SUPPORTED_MINING_RECIPE_VERSIONS:
        expected_evidence["runtime_source_events_sha256"] = runtime_source_events_sha256
    if evidence != expected_evidence:
        raise ValueError("ngsim_bundle_evidence_chain_mismatch")
    return {
        "status": "verified",
        "bundle_id": bundle["bundle_id"],
        "file_count": len(expected),
        "evidence": evidence,
    }
