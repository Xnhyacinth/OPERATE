"""Fail-closed nested-zip archive helpers for NGSIM recovery.

Supports one level of zip-in-zip nesting (the USDOT US-101 official archive
pattern) without requiring the 312 MB real file in CI.  All public functions
are fail-closed: 0 or >1 suffix matches without a hash raises ValueError; a
wrong hash raises ValueError; a pre-existing destination raises FileExistsError.
"""

from __future__ import annotations

import hashlib
import io
import os
import posixpath
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple


class _Member(NamedTuple):
    """Lightweight descriptor for a located archive member."""

    display_name: str  # "outer.zip::inner/path.txt" or "flat/path.txt"
    inner_name: str  # path within the immediate containing zip
    read: Callable[[], bytes]  # deferred content loader


def _member_matches_any_suffix(name: str, suffixes: tuple[str, ...]) -> bool:
    """Return True if *name* matches any provided suffix (case-insensitive).

    Both the full ``name`` and its POSIX basename are compared against the full
    suffix and *its* POSIX basename.  This lets the logical path constant
    ``trajectories/us-101/trajectories-0750am-0805am.txt`` locate the actual
    nested member ``vehicle-trajectory-data/0750am-0805am/trajectories-0750am-
    0805am.txt`` by shared basename, without changing the constant.
    """
    name_lower = name.lower()
    name_base = posixpath.basename(name_lower)
    for suffix in suffixes:
        if not suffix:
            continue
        suffix_lower = suffix.lower()
        suffix_base = posixpath.basename(suffix_lower)
        if name_lower.endswith(suffix_lower):
            return True
        if suffix_base and name_base == suffix_base:
            return True
    return False


def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _walk_archive(archive_path: Path) -> list[_Member]:
    """Return all non-directory leaf members, recursing one level into nested zips.

    For members whose name ends with ``.zip`` the helper opens them from their
    extracted bytes (BytesIO) and lists their leaf contents.  A nested zip that
    fails to parse as a zip is treated as an opaque leaf rather than an error.
    """
    members: list[_Member] = []
    with zipfile.ZipFile(archive_path) as outer:
        for info in outer.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if name.lower().endswith(".zip"):
                nested_bytes = outer.read(name)
                try:
                    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as inner:
                        for inner_info in inner.infolist():
                            if inner_info.is_dir():
                                continue
                            display = f"{name}::{inner_info.filename}"
                            inner_name = inner_info.filename

                            def _make_nested_reader(
                                nb: bytes = nested_bytes, fn: str = inner_name
                            ) -> bytes:
                                with zipfile.ZipFile(io.BytesIO(nb)) as z:
                                    return z.read(fn)

                            members.append(_Member(display, inner_name, _make_nested_reader))
                except zipfile.BadZipFile:
                    # Treat as an opaque leaf (e.g. a zip-named non-zip blob).
                    content = nested_bytes

                    def _make_opaque_reader(c: bytes = content) -> bytes:
                        return c

                    members.append(_Member(name, name, _make_opaque_reader))
            else:

                def _make_flat_reader(ap: Path = archive_path, fn: str = name) -> bytes:
                    with zipfile.ZipFile(ap) as z:
                        return z.read(fn)

                members.append(_Member(name, name, _make_flat_reader))
    return members


def _resolve(
    archive_path: Path,
    suffixes: tuple[str, ...],
    expected_sha256: str | None,
) -> tuple[_Member, bytes, str]:
    """Locate the unique authoritative member and return ``(member, data, sha256)``.

    When *expected_sha256* is provided the member is found by hash (using
    suffix matches as the preferred search pool; falls back to all members
    when no suffix matches the outer or nested namelists).  When
    *expected_sha256* is ``None`` exactly one suffix match is required.

    Raises:
        ValueError: 0 or >1 suffix matches (hash not provided),
                    or no member whose hash equals *expected_sha256*.
        zipfile.BadZipFile: outer archive is not a valid zip.
    """
    all_members = _walk_archive(archive_path)
    suffix_matches = [
        m for m in all_members if _member_matches_any_suffix(m.inner_name, suffixes)
    ]

    if expected_sha256 is not None:
        pool = suffix_matches if suffix_matches else all_members
        for member in pool:
            data = member.read()
            sha = _sha256_of(data)
            if sha == expected_sha256:
                return member, data, sha
        raise ValueError("ngsim_archive_member_sha256_not_found")

    if len(suffix_matches) == 0:
        raise ValueError("ngsim_archive_member_not_found")
    if len(suffix_matches) > 1:
        names = ", ".join(m.display_name for m in suffix_matches)
        raise ValueError(f"ngsim_archive_member_ambiguous:{names}")

    member = suffix_matches[0]
    data = member.read()
    return member, data, _sha256_of(data)


def find_member_in_archive(
    archive_path: Path,
    *,
    suffixes: tuple[str, ...],
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Locate and hash the authoritative archive member without extracting it.

    Walks both the outer zip and any nested zips (one level).  Suffix matching
    is case-insensitive and also compares basenames, so the logical-path
    constant ``trajectories/us-101/trajectories-0750am-0805am.txt`` correctly
    locates ``vehicle-trajectory-data/0750am-0805am/trajectories-0750am-
    0805am.txt`` inside a nested zip.

    Args:
        archive_path: Path to the outer zip archive.
        suffixes: Candidate name suffixes (case-insensitive).
        expected_sha256: When provided, the member is found by hash (using
            suffix matches as the search pool when any match; otherwise all
            members).  Raises if no member's hash matches.

    Returns:
        Mapping with keys ``logical_nested_path``, ``sha256``, ``byte_size``.

    Raises:
        ValueError: 0 or >1 suffix matches (hash not provided), or
                    no member whose hash equals *expected_sha256*.
        zipfile.BadZipFile: outer archive is not a valid zip.
    """
    member, data, sha = _resolve(archive_path, suffixes, expected_sha256)
    return {
        "logical_nested_path": member.display_name,
        "sha256": sha,
        "byte_size": len(data),
    }


def extract_authoritative_member(
    archive_path: Path,
    destination: Path,
    *,
    suffixes: tuple[str, ...],
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Find, extract exclusively, fsync, and verify the authoritative member.

    Writes the resolved member to *destination* exclusively (fails if the path
    already exists), calls ``fsync``, and optionally verifies the SHA-256.

    Args:
        archive_path: Path to the outer zip archive.
        destination: Target path for the extracted file (must not exist).
        suffixes: Candidate name suffixes (see :func:`find_member_in_archive`).
        expected_sha256: When provided, member is located by hash; also acts
            as the post-extract verification digest.

    Returns:
        Mapping with keys ``logical_nested_path``, ``sha256``, ``byte_size``.

    Raises:
        FileExistsError: destination already exists.
        ValueError: member not found, ambiguous, or hash mismatch.
        zipfile.BadZipFile: outer archive is not a valid zip.
    """
    if destination.exists():
        raise FileExistsError(f"ngsim_extract_destination_exists:{destination}")

    member, data, sha = _resolve(archive_path, suffixes, expected_sha256)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())

    return {
        "logical_nested_path": member.display_name,
        "sha256": sha,
        "byte_size": len(data),
    }
