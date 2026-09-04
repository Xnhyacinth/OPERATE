"""Immutable source plans and locks for the public USDOT NGSIM table."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

NGSIM_DATASET_ID = "8ect-6jqj"
NGSIM_DOI = "10.21949/1504477"
NGSIM_METADATA_URL = f"https://data.transportation.gov/api/views/{NGSIM_DATASET_ID}"
NGSIM_API_URL = f"https://data.transportation.gov/resource/{NGSIM_DATASET_ID}.csv"
NGSIM_LICENSE_ID = "CC-BY-SA-4.0"
NGSIM_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
# The authoritative API snapshot also exposes a dataset-level 3.0
# declaration.  Keep it separate from the Common Core 4.0 field; the
# discrepancy is an admission blocker until a human/legal review resolves it.
NGSIM_PORTAL_LICENSE_ID = "CC-BY-SA-3.0"
NGSIM_PORTAL_LICENSE_URL = "http://creativecommons.org/licenses/by-sa/3.0/legalcode"
NGSIM_ATTRIBUTION = (
    "U.S. Department of Transportation Federal Highway Administration. "
    "(2016). Next Generation Simulation (NGSIM) Vehicle Trajectories and "
    "Supporting Data. DOI: 10.21949/1504477."
)
NGSIM_SOURCE_SCHEMA_VERSION = "ngsim_source_plan_v1"
NGSIM_LOCATIONS = ("us-101", "i-80", "lankershim", "peachtree")
NGSIM_US101_ASSET_ID = "a4d1630d-2970-423d-96fe-035f8731f7be"
NGSIM_US101_ARCHIVE_URL = (
    f"https://data.transportation.gov/api/views/{NGSIM_DATASET_ID}/files/"
    f"{NGSIM_US101_ASSET_ID}?download=true&filename=US-101-LosAngeles-CA.zip"
)
NGSIM_US101_AUTHORITATIVE_MEMBER = "trajectories/us-101/trajectories-0750am-0805am.txt"
NGSIM_US101_AUTHORITATIVE_ROW_COUNT = 1_180_598
NGSIM_US101_ARCHIVE_SHA256 = "830442bb08f1f7a20a686b5879b4b82bfb7c47e526d48908db666fccc7e811dc"
NGSIM_US101_AUTHORITATIVE_SHA256 = (
    "6676066e5b3e7249b7b003a906b28585202623f5d274938b579b4b836f0668c8"
)

NGSIM_COLUMNS: tuple[str, ...] = (
    "vehicle_id",
    "frame_id",
    "total_frames",
    "global_time",
    "local_x",
    "local_y",
    "global_x",
    "global_y",
    "v_length",
    "v_width",
    "v_class",
    "v_vel",
    "v_acc",
    "lane_id",
    "preceding",
    "following",
    "space_headway",
    "time_headway",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical byte encoding used for all evidence digests."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Any) -> None:
    """Write an immutable JSON artifact, refusing every existing target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class NGSIMSourcePlan:
    """A bounded, canonical SODA query over one declared recording window."""

    recording_id: str
    start_time_ms: int
    end_time_ms: int
    max_rows: int
    query_url: str
    query_sha256: str
    output_name: str
    profile: str = "smoke"
    location: str = "us-101"
    source_kind: str = "soda_bounded_csv"
    authoritative_member: str | None = None
    expected_download_sha256: str | None = None
    expected_authoritative_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": NGSIM_SOURCE_SCHEMA_VERSION,
            "source_dataset_id": NGSIM_DATASET_ID,
            "source_release": f"doi:{NGSIM_DOI}",
            "source_metadata_url": NGSIM_METADATA_URL,
            "source_api_url": NGSIM_API_URL,
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
            "license_review_status": "pending_metadata_discrepancy",
            "attribution": NGSIM_ATTRIBUTION,
            "recording_id": self.recording_id,
            "profile": self.profile,
            "location": self.location,
            "source_kind": self.source_kind,
            "authoritative_member": self.authoritative_member,
            "expected_download_sha256": self.expected_download_sha256,
            "expected_authoritative_sha256": self.expected_authoritative_sha256,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "sample_period_ms": 100,
            "max_rows": self.max_rows,
            "columns": list(NGSIM_COLUMNS),
            "query_url": self.query_url,
            "query_sha256": self.query_sha256,
            "output_name": self.output_name,
            "public_packaging": "redistributable_with_attribution_and_share_alike",
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> NGSIMSourcePlan:
        if raw.get("schema_version") != NGSIM_SOURCE_SCHEMA_VERSION:
            raise ValueError("ngsim_plan_schema_mismatch")
        if raw.get("source_dataset_id") != NGSIM_DATASET_ID:
            raise ValueError("ngsim_plan_dataset_mismatch")
        if tuple(raw.get("columns") or ()) != NGSIM_COLUMNS:
            raise ValueError("ngsim_plan_columns_mismatch")
        profile = str(raw.get("profile") or "smoke")
        source_kind = str(raw.get("source_kind") or "soda_bounded_csv")
        if source_kind == "official_archive_converted_csv":
            plan = build_ngsim_archive_plan(
                recording_id=str(raw.get("recording_id") or ""),
                start_time_ms=int(raw.get("start_time_ms") or 0),
                end_time_ms=int(raw.get("end_time_ms") or 0),
                max_rows=int(raw.get("max_rows") or 0),
                archive_url=str(raw.get("query_url") or ""),
                archive_sha256=str(raw.get("expected_download_sha256") or ""),
                authoritative_member=str(raw.get("authoritative_member") or ""),
                member_sha256=str(raw.get("expected_authoritative_sha256") or ""),
                output_name=str(raw.get("output_name") or ""),
            )
        elif profile == "core":
            plan = build_ngsim_core_plan()
        else:
            plan = build_ngsim_plan(
                recording_id=str(raw.get("recording_id") or ""),
                start_time_ms=int(raw.get("start_time_ms") or 0),
                end_time_ms=int(raw.get("end_time_ms") or 0),
                max_rows=int(raw.get("max_rows") or 0),
            )
        if raw.get("query_url") != plan.query_url:
            raise ValueError("ngsim_plan_query_url_mismatch")
        if raw.get("query_sha256") != plan.query_sha256:
            raise ValueError("ngsim_plan_query_hash_mismatch")
        if raw.get("output_name") != plan.output_name:
            raise ValueError("ngsim_plan_output_name_mismatch")
        return plan


def build_ngsim_plan(
    *,
    recording_id: str,
    start_time_ms: int,
    end_time_ms: int,
    max_rows: int = 50_000,
) -> NGSIMSourcePlan:
    """Build a deterministic query plan without accessing the network."""
    recording = recording_id.strip().lower()
    if recording not in NGSIM_LOCATIONS:
        raise ValueError("ngsim_recording_id_invalid")
    if start_time_ms <= 0 or end_time_ms <= start_time_ms:
        raise ValueError("ngsim_time_window_invalid")
    if start_time_ms % 100 or end_time_ms % 100:
        raise ValueError("ngsim_time_window_not_100ms_aligned")
    if max_rows <= 0 or max_rows > 1_000_000:
        raise ValueError("ngsim_max_rows_invalid")
    params = (
        ("$limit", str(max_rows)),
        ("$order", "global_time,vehicle_id,frame_id"),
        ("$select", ",".join(NGSIM_COLUMNS)),
        (
            "$where",
            f"location='{recording}' AND global_time between {start_time_ms} and {end_time_ms}",
        ),
    )
    query = urlencode(params, quote_via=quote, safe=",")
    query_url = f"{NGSIM_API_URL}?{query}"
    query_sha256 = hashlib.sha256(query_url.encode("utf-8")).hexdigest()
    output_name = f"ngsim_{recording}_{start_time_ms}_{end_time_ms}_{query_sha256[:12]}.csv"
    return NGSIMSourcePlan(
        recording_id=recording,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        max_rows=max_rows,
        query_url=query_url,
        query_sha256=query_sha256,
        output_name=output_name,
        location=recording,
    )


def build_ngsim_core_plan() -> NGSIMSourcePlan:
    """Return the official US-101 archive plan used for Core preparation.

    The authoritative 07:50--08:05 block is the whitespace TXT member.  The
    companion CSV is deliberately not named because it is Excel-truncated.
    """
    query_sha256 = hashlib.sha256(NGSIM_US101_ARCHIVE_URL.encode("utf-8")).hexdigest()
    return NGSIMSourcePlan(
        recording_id="us-101",
        start_time_ms=0,
        end_time_ms=0,
        max_rows=0,
        query_url=NGSIM_US101_ARCHIVE_URL,
        query_sha256=query_sha256,
        output_name="US-101-LosAngeles-CA.zip",
        profile="core",
        location="us-101",
        source_kind="official_archive_authoritative_txt",
        authoritative_member=NGSIM_US101_AUTHORITATIVE_MEMBER,
        expected_download_sha256=NGSIM_US101_ARCHIVE_SHA256,
        expected_authoritative_sha256=NGSIM_US101_AUTHORITATIVE_SHA256,
    )


def build_ngsim_archive_plan(
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
    """Describe a canonical CSV converted from an official archive member.

    The converted file remains the runtime input, while the plan retains the
    immutable archive/member hashes needed to audit the transformation.  The
    downloader intentionally does not accept this plan as a mutable API
    query; callers must provision the official archive and then run the
    explicit converter.
    """
    base = build_ngsim_plan(
        recording_id=recording_id,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        max_rows=max_rows,
    )
    if not archive_url or len(archive_sha256) != 64 or len(member_sha256) != 64:
        raise ValueError("ngsim_archive_plan_provenance_invalid")
    if not authoritative_member or not output_name:
        raise ValueError("ngsim_archive_plan_names_missing")
    query_sha256 = hashlib.sha256(archive_url.encode("utf-8")).hexdigest()
    return replace(
        base,
        query_url=archive_url,
        query_sha256=query_sha256,
        output_name=output_name,
        profile="recording",
        source_kind="official_archive_converted_csv",
        authoritative_member=authoritative_member,
        expected_download_sha256=archive_sha256,
        expected_authoritative_sha256=member_sha256,
    )


def build_source_lock(
    plan: NGSIMSourcePlan,
    raw_path: Path,
    *,
    row_count: int,
    semantic_sha256: str,
) -> dict[str, Any]:
    """Bind the query plan to the exact downloaded bytes and parsed rows."""
    if row_count <= 0:
        raise ValueError("ngsim_source_has_no_rows")
    lock = {
        "schema_version": "ngsim_source_lock_v1",
        "source_plan": plan.to_dict(),
        "raw_file_name": raw_path.name,
        "raw_byte_size": raw_path.stat().st_size,
        "raw_sha256": file_sha256(raw_path),
        "row_count": row_count,
        "raw_semantic_sha256": semantic_sha256,
        "lock_strategy": "doi+canonical_query_or_archive+raw_sha256+row_semantic_sha256",
    }
    lock["source_evidence_sha256"] = object_sha256(lock)
    return lock
