"""Resumable, atomic, fail-closed downloads for autonomous-driving sources."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, BinaryIO
from urllib.request import Request, urlopen

from .contracts import NGSIMSourcePlan, file_sha256, write_json_exclusive

OpenUrl = Callable[[Request], BinaryIO]


def _exclusive_process_lock(path: Path) -> int:
    try:
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("ngsim_fetch_already_in_progress") from error


def _response_status(response: BinaryIO) -> int:
    status = getattr(response, "status", None)
    if status is not None:
        return int(status)
    getter = getattr(response, "getcode", None)
    value = getter() if callable(getter) else None
    return int(value) if value is not None else 200


def _content_range(response: BinaryIO) -> str:
    headers = getattr(response, "headers", {})
    value = headers.get("Content-Range") if hasattr(headers, "get") else None
    return str(value or "")


def _commit_file_no_overwrite(part_path: Path, destination: Path) -> None:
    """Atomically publish a same-filesystem file without replacement."""
    try:
        os.link(part_path, destination)
    except FileExistsError as error:
        raise FileExistsError("ngsim_download_target_exists") from error
    part_path.unlink()


def fetch_plan(
    plan: NGSIMSourcePlan,
    destination: Path,
    *,
    expected_sha256: str | None = None,
    opener: OpenUrl = urlopen,
) -> dict[str, Any]:
    """Fetch one plan, resuming a validated partial and publishing atomically."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_sha256 and file_sha256(destination) == expected_sha256:
            return {
                "status": "already_present_verified",
                "path": str(destination),
                "raw_sha256": expected_sha256,
                "raw_byte_size": destination.stat().st_size,
                "query_sha256": plan.query_sha256,
            }
        raise FileExistsError("ngsim_download_target_exists_unverified")

    part_path = destination.with_name(f".{destination.name}.part")
    part_meta = destination.with_name(f".{destination.name}.part.json")
    process_lock = destination.with_name(f".{destination.name}.fetch.lock")
    lock_fd = _exclusive_process_lock(process_lock)
    os.close(lock_fd)
    try:
        expected_part_meta = {
            "schema_version": "resumable_download_v1",
            "query_url": plan.query_url,
            "query_sha256": plan.query_sha256,
            "destination_name": destination.name,
        }
        if part_meta.exists():
            observed = json.loads(part_meta.read_text(encoding="utf-8"))
            if observed != expected_part_meta:
                raise ValueError("ngsim_partial_download_identity_mismatch")
        else:
            write_json_exclusive(part_meta, expected_part_meta)
        offset = part_path.stat().st_size if part_path.exists() else 0
        headers = {
            "Accept": "text/csv",
            "User-Agent": "OPERATE-NGSIM-Materializer/1",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(plan.query_url, headers=headers)
        with opener(request) as response:
            status = _response_status(response)
            append = offset > 0 and status == 206
            if append:
                if not _content_range(response).startswith(f"bytes {offset}-"):
                    raise ValueError("ngsim_resume_content_range_mismatch")
            elif status not in {200, 206}:
                raise OSError(f"ngsim_fetch_http_status_{status}")
            mode = "ab" if append else "wb"
            with part_path.open(mode) as stream:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        observed_sha256 = file_sha256(part_path)
        if expected_sha256 and observed_sha256 != expected_sha256:
            part_path.unlink(missing_ok=True)
            part_meta.unlink(missing_ok=True)
            raise ValueError("ngsim_download_sha256_mismatch")
        _commit_file_no_overwrite(part_path, destination)
        part_meta.unlink()
        return {
            "status": "fetched",
            "path": str(destination),
            "raw_sha256": observed_sha256,
            "raw_byte_size": destination.stat().st_size,
            "query_sha256": plan.query_sha256,
        }
    finally:
        process_lock.unlink(missing_ok=True)
