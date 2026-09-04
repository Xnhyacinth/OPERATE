#!/usr/bin/env python3
"""Safely audit Grid2Op source availability without triggering downloads.

This script is intentionally conservative: it never calls
``grid2op.make(env_name, test=False)`` on a remote environment name. Optional
live loading is allowed only for prechecked local directories, passed by path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from posixpath import normpath
from typing import Any, BinaryIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ENVS = [
    "l2rpn_case14_sandbox",
    "l2rpn_neurips_2020_track1_small",
    "l2rpn_icaps_2021_small",
    "l2rpn_neurips_2020_track2_small",
    "l2rpn_idf_2023",
    "l2rpn_wcci_2022",
    "l2rpn_wcci_2020",
]

DEFAULT_CHUNK_BYTES = 1024 * 1024
MAX_TAR_MEMBER_BYTES = 30 * 1024 * 1024 * 1024
MAX_TAR_COMPRESSION_RATIO = 100
DEFAULT_SOURCE_TREE_FIXTURES = [
    "l2rpn_idf_2023",
    "l2rpn_wcci_2022_dev",
]
GRID2OP_REQUIRED_MIRROR_FIELDS = [
    "mirror_url",
    "expected_sha256",
    "mirror_lock_strategy",
]


def _default_data_root(grid2op_module: Any | None = None) -> Path:
    if grid2op_module is not None:
        try:
            raw = grid2op_module.MakeEnv.PathUtils.DEFAULT_PATH_DATA
            return Path(raw).expanduser()
        except AttributeError:
            pass
    return Path.home() / "data_grid2op"


def _safe_call_list(grid2op_module: Any, attr: str) -> list[str]:
    fn = getattr(grid2op_module, attr, None)
    if not callable(fn):
        return []
    try:
        return sorted(str(x) for x in fn())
    except Exception:
        return []


def _is_complete_env_dir(path: Path) -> bool:
    return ((path / "config.py").exists() and (path / "grid.json").exists()) or (
        path / ".multimix"
    ).exists()


def _count_chronics(path: Path) -> int | None:
    chronics = path / "chronics"
    if not chronics.is_dir():
        return None
    return sum(1 for child in chronics.iterdir() if child.is_dir())


def _chronic_names(path: Path) -> list[str]:
    chronics = path / "chronics"
    if not chronics.is_dir():
        return []
    return sorted(child.name for child in chronics.iterdir() if child.is_dir())


def _dir_size_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(DEFAULT_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _git_dirty_paths(repo_root: Path) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    return [line.strip() for line in output.splitlines() if line.strip()]


def _source_tree_license_files(repo_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in ["LICENSE", "LICENSE.md", "LicensesInformation.md"]:
        path = repo_root / name
        if not path.is_file():
            continue
        out.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
        )
    return out


def _fingerprint_file(path: Path) -> dict[str, Any]:
    digest = _sha256_file(path) if path.is_file() else None
    return {
        "path": str(path),
        "path_base": "absolute",
        "absolute_path_at_build": str(path),
        "exists": path.exists(),
        "sha256": digest,
        "matches_current_file": (not path.exists() and digest is None)
        or (path.is_file() and _sha256_file(path) == digest),
    }


def _fixture_directory_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    for child in sorted(
        p for p in path.rglob("*") if p.is_file() and _fingerprint_fixture_file(p)
    ):
        rel = child.relative_to(path).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(_sha256_file(child).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()


def _fingerprint_fixture_file(path: Path) -> bool:
    if path.name in {".DS_Store"}:
        return False
    if path.suffix == ".pyc":
        return False
    return "__pycache__" not in path.parts


def _source_tree_fixture_input_fingerprints(
    *,
    data_root: Path,
    source_repo_root: Path,
    fixture_names: Iterable[str],
    commit: str | None,
    dirty_paths: list[str],
) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for item in _source_tree_license_files(source_repo_root):
        path = Path(str(item["path"]))
        files[f"license:{path.name}"] = _fingerprint_file(path)

    fixtures: dict[str, dict[str, Any]] = {}
    for fixture_name in fixture_names:
        fixture_dir = data_root / fixture_name
        fixture_files = sorted(
            p
            for p in fixture_dir.rglob("*")
            if p.is_file() and _fingerprint_fixture_file(p)
        )
        for path in fixture_files:
            rel = path.relative_to(fixture_dir).as_posix()
            files[f"fixture:{fixture_name}/{rel}"] = _fingerprint_file(path)
        fixtures[str(fixture_name)] = {
            "path": str(fixture_dir),
            "path_base": "absolute",
            "absolute_path_at_build": str(fixture_dir),
            "exists": fixture_dir.exists(),
            "file_count": len(fixture_files),
            "directory_sha256": _fixture_directory_sha256(fixture_dir),
        }

    return {
        "schema_version": "0.1",
        "source_tree": {
            "repo_root": str(source_repo_root),
            "commit": commit,
            "dirty_paths": dirty_paths,
        },
        "fixtures": fixtures,
        "files": files,
        "all_present": all(item["exists"] is True for item in files.values()),
        "all_sha256_match_current_files": all(
            item["matches_current_file"] is True for item in files.values()
        ),
    }


def _infer_source_tree_repo_root(data_root: Path) -> Path:
    if data_root.name == "data" and data_root.parent.name == "grid2op":
        return data_root.parents[1]
    return data_root


def _archive_names(env_name: str, remote_filename: str | None = None) -> list[str]:
    names = [f"{env_name}.tar.bz2"]
    if remote_filename:
        names.append(remote_filename)
        if not remote_filename.endswith(".tar.bz2"):
            names.append(f"{remote_filename}.tar.bz2")
    return list(dict.fromkeys(names))


def _partial_archives(
    data_root: Path,
    env_name: str,
    remote_filename: str | None = None,
) -> list[dict[str, Any]]:
    candidates = []
    for name in _archive_names(env_name, remote_filename):
        candidates.append(data_root / name)
        candidates.append(data_root / ".partial" / name)
    out: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        out.append(
            {
                "name": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
            }
        )
    return out


def _local_shape(data_root: Path, env_name: str) -> tuple[str, Path | None, str | None]:
    env_dir = data_root / env_name
    if not env_dir.exists():
        return "missing", None, None
    if _is_complete_env_dir(env_dir):
        return "complete", env_dir, None
    nested = sorted(
        child
        for child in env_dir.iterdir()
        if child.is_dir() and _is_complete_env_dir(child)
    )
    if nested:
        return (
            "nested_complete",
            nested[0],
            (
                f"{env_dir} is not directly loadable, but contains "
                f"nested complete env {nested[0].name!r}."
            ),
        )
    return "incomplete", env_dir, "missing config.py + grid.json or .multimix"


def _download_url(meta: dict[str, Any]) -> str | None:
    base_url = meta.get("base_url")
    filename = meta.get("filename")
    if not base_url or not filename:
        return None
    return f"{str(base_url).rstrip('/')}/{filename}"


def _download_target_name(env_name: str, remote_filename: str) -> str:
    if remote_filename.endswith(".tar.bz2"):
        return remote_filename
    return f"{remote_filename}.tar.bz2"


def _safe_remote_metadata(grid2op_module: Any) -> dict[str, dict[str, Any]]:
    hook = getattr(grid2op_module, "remote_env_metadata", None)
    if callable(hook):
        raw = hook()
        if isinstance(raw, dict):
            return {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}

    if getattr(grid2op_module, "__name__", None) != "grid2op":
        return {}

    try:
        from grid2op.MakeEnv.Make import _list_available_remote_env_aux  # type: ignore[import]
    except Exception:
        return {}
    try:
        raw = _list_available_remote_env_aux()
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): dict(v) for k, v in raw.items() if isinstance(v, dict)}


def grid2op_local_load_path(
    env_name: str,
    *,
    data_root: Path | str | None = None,
    grid2op_module: Any | None = None,
) -> Path:
    """Return a directly loadable local env path or raise.

    This is the guardrail for materializers: do not pass remote environment
    names to ``grid2op.make(..., test=False)``. Only a complete direct local
    directory is acceptable.
    """
    root = (
        Path(data_root).expanduser()
        if data_root is not None
        else _default_data_root(grid2op_module)
    )
    local_state, load_path, local_note = _local_shape(root, env_name)
    if local_state == "complete" and load_path is not None:
        return load_path
    detail = f" ({local_note})" if local_note else ""
    raise RuntimeError(
        f"{env_name!r} is not a complete local Grid2Op env under {root}{detail}. "
        "Download/extract it first; refusing to call grid2op.make(env_name, "
        "test=False) because that can trigger a large automatic download."
    )


def build_download_plan(
    env_name: str,
    *,
    data_root: Path | str | None = None,
    grid2op_module: Any | None = None,
) -> dict[str, Any]:
    """Build a controlled download plan without downloading bytes."""
    if grid2op_module is None:
        import grid2op as grid2op_module  # type: ignore[import]

    base: dict[str, Any] = {
        "schema_version": "0.1",
        "scope": "grid2op_controlled_download_plan",
        "non_release_artifact": True,
        "release_ready": False,
        "release_reentry_ready": False,
        "proceed_commands": [],
        "archive_count_basis": "controlled_archive_download_targets",
    }
    report = audit_sources(
        [env_name],
        data_root=data_root,
        grid2op_module=grid2op_module,
        load_local=False,
    )
    row = report["sources"][0]
    remote = row.get("remote_download")
    if row["status"] == "local_loadable":
        return base | {
            "env_name": env_name,
            "status": "already_local_loadable",
            "data_root": row["data_root"],
            "target_path": None,
            "url": None,
            "existing_bytes": 0,
            "n_archives": 0,
            "message": "No download needed; local env is already complete.",
        }
    if not remote or not remote.get("url") or not remote.get("filename"):
        raise RuntimeError(
            f"{env_name!r} has no remote download metadata; cannot build a "
            "controlled download plan."
        )

    root = Path(row["data_root"])
    target_name = _download_target_name(env_name, str(remote["filename"]))
    root_archive = root / target_name
    target = root / ".partial" / target_name
    existing_bytes = target.stat().st_size if target.exists() else 0
    root_archive_bytes = root_archive.stat().st_size if root_archive.exists() else 0
    status = "blocked_by_root_archive" if root_archive_bytes else "download_planned"
    return base | {
        "env_name": env_name,
        "status": status,
        "data_root": str(root),
        "url": remote["url"],
        "target_path": str(target),
        "target_name": target_name,
        "existing_bytes": existing_bytes,
        "n_archives": 1,
        "root_archive_path": str(root_archive) if root_archive.exists() else None,
        "root_archive_bytes": root_archive_bytes,
        "remote_download": remote,
        "note": (
            "This downloads only the archive into .partial; it does not "
            "extract, load, materialize, or modify release artifacts."
            if not root_archive_bytes
            else "A same-named archive already exists at the data root. Move "
            "or remove it deliberately before starting a controlled .partial "
            "download, to avoid splitting one dataset across two partial files."
        ),
    }


def download_archive(
    *,
    url: str,
    target_path: Path | str,
    restart: bool = False,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    timeout_s: float | None = None,
    opener: Callable[[urllib.request.Request], BinaryIO] | None = None,
) -> dict[str, Any]:
    """Download ``url`` into ``target_path`` with fail-fast resume semantics."""
    target = Path(target_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    if restart and target.exists():
        target.unlink()
    if timeout_s is not None and timeout_s <= 0:
        raise ValueError("timeout_s must be positive when provided")

    before = target.stat().st_size if target.exists() else 0
    headers: dict[str, str] = {}
    if before:
        headers["Range"] = f"bytes={before}-"
    request = urllib.request.Request(url, headers=headers)
    open_url = opener or urllib.request.urlopen
    response = open_url(request) if timeout_s is None else open_url(request, timeout=timeout_s)
    try:
        raw_status = getattr(response, "status", None)
        if raw_status is None:
            raw_status = response.getcode()
        status = int(raw_status)
        if before and status != 206:
            raise RuntimeError(
                f"Refusing to overwrite existing partial {target}: server "
                f"returned HTTP {status}, not 206 Partial Content. Re-run with "
                "--restart-download if a fresh download is intended."
            )
        mode = "ab" if before else "wb"
        with target.open(mode) as fh:
            shutil.copyfileobj(response, fh, length=chunk_bytes)
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    after = target.stat().st_size
    return {
        "url": url,
        "target_path": str(target),
        "bytes_before": before,
        "bytes_after": after,
        "bytes_downloaded": after - before,
        "http_status": status,
        "timeout_s": timeout_s,
    }


def verify_archive_checksum(
    archive_path: Path | str,
    expected_sha256: str | None,
) -> dict[str, Any]:
    """Verify a local archive against an expected sha256."""
    archive = Path(archive_path).expanduser()
    expected = expected_sha256 or ""
    base: dict[str, Any] = {
        "schema_version": "0.1",
        "scope": "grid2op_archive_checksum_verification",
        "non_release_artifact": True,
        "release_ready": False,
        "release_reentry_ready": False,
        "proceed_commands": [],
        "archive_count_basis": "local_archive_files_verified",
        "n_archives": 1,
    }
    if not _valid_sha256(expected):
        return base | {
            "archive_path": str(archive),
            "expected_sha256": expected_sha256,
            "actual_sha256": None,
            "matches": False,
            "error": "invalid_expected_sha256",
        }
    if not archive.exists():
        return base | {
            "archive_path": str(archive),
            "expected_sha256": expected,
            "actual_sha256": None,
            "matches": False,
            "error": "missing_archive",
        }
    h = hashlib.sha256()
    with archive.open("rb") as f:
        for chunk in iter(lambda: f.read(DEFAULT_CHUNK_BYTES), b""):
            h.update(chunk)
    actual = h.hexdigest()
    return base | {
        "archive_path": str(archive),
        "expected_sha256": expected,
        "actual_sha256": actual,
        "matches": actual.lower() == expected.lower(),
        "error": None,
    }


def preflight_download_url(
    url: str,
    *,
    opener: Callable[[urllib.request.Request], BinaryIO] | None = None,
) -> dict[str, Any]:
    """Check URL/TLS/HTTP reachability without downloading the archive body."""
    request = urllib.request.Request(url, method="HEAD")
    open_url = opener or urllib.request.urlopen
    try:
        response = open_url(request)
    except Exception as exc:
        return {
            "ok": False,
            "url": url,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    try:
        raw_status = getattr(response, "status", None)
        if raw_status is None:
            raw_status = response.getcode()
        headers = getattr(response, "headers", {})
        return {
            "ok": True,
            "url": url,
            "http_status": int(raw_status),
            "content_type": headers.get("content-type"),
            "content_length": headers.get("content-length"),
        }
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _issue(code: str, message: str, **context: Any) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "message": message}
    if context:
        item["context"] = context
    return item


def _valid_sha256(value: str | None) -> bool:
    if value is None:
        return False
    if len(value) != 64:
        return False
    return all(ch in "0123456789abcdefABCDEF" for ch in value)


def build_trusted_acquisition_gate(
    env_name: str,
    *,
    data_root: Path | str | None = None,
    grid2op_module: Any | None = None,
    mirror_url: str | None = None,
    expected_sha256: str | None = None,
    mirror_lock_strategy: str | None = None,
    opener: Callable[[urllib.request.Request], BinaryIO] | None = None,
) -> dict[str, Any]:
    """Dry-run trust gate for large Grid2Op archive acquisition.

    The gate never downloads archive bytes. It decides whether a future
    download is allowed through either the upstream verified-TLS URL or a
    checksum-locked mirror. A mirror without a sha256 is deliberately blocked:
    it would be provenance, not integrity.
    """
    plan = build_download_plan(
        env_name,
        data_root=data_root,
        grid2op_module=grid2op_module,
    )
    primary_url = plan.get("url")
    primary_preflight = (
        preflight_download_url(str(primary_url), opener=opener)
        if primary_url
        else {"ok": False, "url": None, "error_type": "missing_url"}
    )
    issues: list[dict[str, Any]] = []
    mirror: dict[str, Any] | None = None

    if plan.get("status") == "already_local_loadable":
        return {
            "schema_version": "0.1",
            "scope": "grid2op_trusted_acquisition_gate",
            "non_release_artifact": True,
            "env_name": env_name,
            "status": "already_local_loadable",
            "trusted_to_download": False,
            "release_ready": False,
            "release_reentry_ready": False,
            "proceed_commands": [],
            "required_mirror_fields": GRID2OP_REQUIRED_MIRROR_FIELDS,
            "archive_count_basis": "controlled_archive_download_targets",
            "n_archives": 0,
            "download_plan": plan,
            "primary_preflight": primary_preflight,
            "mirror": None,
            "issues": [],
            "policy": "local env already complete; do not download",
        }

    if plan.get("status") == "blocked_by_root_archive":
        issues.append(
            _issue(
                "root_archive_blocks_controlled_download",
                "same-named root archive exists; move or remove it deliberately",
                path=plan.get("root_archive_path"),
                bytes=plan.get("root_archive_bytes"),
            )
        )

    if not primary_preflight.get("ok"):
        issues.append(
            _issue(
                "primary_download_unreachable",
                "primary URL did not pass verified TLS/HTTP preflight",
                error_type=primary_preflight.get("error_type"),
                error=primary_preflight.get("error"),
            )
        )

    if mirror_url:
        if not _valid_sha256(expected_sha256):
            issues.append(
                _issue(
                    "missing_expected_sha256",
                    "trusted mirror downloads require a 64-hex expected sha256",
                )
            )
        if not mirror_lock_strategy:
            issues.append(
                _issue(
                    "missing_mirror_lock_strategy",
                    "trusted mirror downloads require an explicit lock strategy",
                )
            )
        if _valid_sha256(expected_sha256) and mirror_lock_strategy:
            mirror_preflight = preflight_download_url(mirror_url, opener=opener)
            mirror = {
                "url": mirror_url,
                "expected_sha256": expected_sha256,
                "lock_strategy": mirror_lock_strategy,
                "preflight": mirror_preflight,
            }
            if not mirror_preflight.get("ok"):
                issues.append(
                    _issue(
                        "mirror_download_unreachable",
                        "mirror URL did not pass verified TLS/HTTP preflight",
                        error_type=mirror_preflight.get("error_type"),
                        error=mirror_preflight.get("error"),
                    )
                )

    trusted_to_download = False
    if plan.get("status") == "download_planned":
        if primary_preflight.get("ok"):
            trusted_to_download = True
            status = "primary_url_ready"
        elif mirror and mirror.get("preflight", {}).get("ok"):
            trusted_to_download = True
            status = "trusted_mirror_ready"
        elif mirror_url and any(
            issue["code"] in {"missing_expected_sha256", "missing_mirror_lock_strategy"}
            for issue in issues
        ):
            status = "blocked_invalid_mirror_policy"
        else:
            status = "blocked_primary_tls_or_http"
    elif plan.get("status") == "blocked_by_root_archive":
        status = "blocked_by_root_archive"
    else:
        status = str(plan.get("status"))

    return {
        "schema_version": "0.1",
        "scope": "grid2op_trusted_acquisition_gate",
        "non_release_artifact": True,
        "env_name": env_name,
        "status": status,
        "trusted_to_download": trusted_to_download,
        "release_ready": False,
        "release_reentry_ready": False,
        "proceed_commands": [],
        "required_mirror_fields": GRID2OP_REQUIRED_MIRROR_FIELDS,
        "archive_count_basis": "controlled_archive_download_targets",
        "n_archives": 1 if plan.get("target_path") else 0,
        "download_plan": plan,
        "primary_preflight": primary_preflight,
        "mirror": mirror,
        "issues": issues if not trusted_to_download else [],
        "policy": (
            "Do not use TLS bypass or curl -k. Download only when the primary "
            "URL passes verified TLS/HTTP preflight, or when an explicit mirror "
            "URL passes verified preflight and is locked by expected_sha256 plus "
            "mirror_lock_strategy."
        ),
    }


def inspect_archive(archive_path: Path | str) -> dict[str, Any]:
    """Inspect a tar.bz2 archive without extracting it."""
    archive = Path(archive_path).expanduser()
    if not archive.exists():
        raise RuntimeError(f"archive not found: {archive}")
    compressed_bytes = archive.stat().st_size
    inspect_error = None
    try:
        with tarfile.open(archive, "r:bz2") as tar:
            members = tar.getmembers()
    except (tarfile.TarError, EOFError, OSError) as exc:
        members = []
        inspect_error = f"{type(exc).__name__}: {exc}"
    total_uncompressed = sum(member.size for member in members)
    names = [member.name for member in members]
    unsafe_members: list[dict[str, str]] = []
    for member in members:
        name = member.name
        normalized = normpath(name)
        parts = normalized.split("/")
        if name.startswith("/") or normalized == ".." or ".." in parts:
            unsafe_members.append({"name": name, "reason": "path_traversal"})
        elif member.issym() or member.islnk():
            unsafe_members.append({"name": name, "reason": "link_member"})
        elif not (member.isfile() or member.isdir()):
            unsafe_members.append({"name": name, "reason": "unsupported_member_type"})
    top_dirs = sorted({name.split("/", 1)[0] for name in names if name})
    has_config = any(
        name.endswith("/config.py") or name == "config.py" for name in names
    )
    has_grid = any(name.endswith("/grid.json") or name == "grid.json" for name in names)
    has_multimix = any(
        name.endswith("/.multimix") or name == ".multimix" for name in names
    )
    ratio = (
        total_uncompressed / compressed_bytes
        if compressed_bytes and total_uncompressed
        else 0.0
    )
    safe_to_extract = (
        total_uncompressed <= MAX_TAR_MEMBER_BYTES
        and ratio <= MAX_TAR_COMPRESSION_RATIO
        and (has_multimix or (has_config and has_grid))
        and not unsafe_members
        and inspect_error is None
    )
    return {
        "archive_path": str(archive),
        "compressed_bytes": compressed_bytes,
        "member_count": len(members),
        "total_uncompressed_bytes": total_uncompressed,
        "compression_ratio": round(ratio, 4),
        "top_dirs": top_dirs[:20],
        "has_config_py": has_config,
        "has_grid_json": has_grid,
        "has_multimix": has_multimix,
        "unsafe_members": unsafe_members,
        "inspect_error": inspect_error,
        "safe_to_extract": safe_to_extract,
        "safety_limits": {
            "max_uncompressed_bytes": MAX_TAR_MEMBER_BYTES,
            "max_compression_ratio": MAX_TAR_COMPRESSION_RATIO,
        },
    }


def _load_topology(grid2op_module: Any, env_path: Path) -> dict[str, Any]:
    env = grid2op_module.make(env_path, test=False)
    try:
        topo = {
            "n_sub": int(env.n_sub),
            "n_load": int(env.n_load),
            "n_gen": int(env.n_gen),
            "n_line": int(env.n_line),
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    n_chronics = _count_chronics(env_path)
    if n_chronics is not None:
        topo["n_chronics"] = n_chronics
    return topo


def audit_source_tree_fixtures(
    fixture_names: Iterable[str] = DEFAULT_SOURCE_TREE_FIXTURES,
    *,
    source_tree_data_root: Path | str,
    grid2op_module: Any | None = None,
    load_local: bool = False,
) -> dict[str, Any]:
    """Audit Grid2Op source-tree fixture datasets as dev-only evidence.

    Source-tree fixtures can be useful for local load/topology preflight, but
    they are partial samples bundled in a checkout, not release-acquired data.
    This report therefore keeps them explicitly non-release even when they load.
    """
    if load_local and grid2op_module is None:
        import grid2op as grid2op_module  # type: ignore[import]

    fixture_names = list(fixture_names)
    data_root = Path(source_tree_data_root).expanduser()
    source_repo_root = _infer_source_tree_repo_root(data_root)
    commit = _git_commit(source_repo_root)
    dirty_paths = _git_dirty_paths(source_repo_root)

    rows: list[dict[str, Any]] = []
    for fixture_name in fixture_names:
        fixture_dir = data_root / fixture_name
        local_state, load_path, local_note = _local_shape(data_root, fixture_name)
        blocker_codes = [
            "source_tree_fixture_subset",
            "not_archive_acquired_through_trusted_gate",
            "missing_release_manifest_source_lock",
        ]
        if local_state == "missing":
            status = "fixture_missing"
            blocker_codes = ["fixture_missing"]
        elif local_state == "complete":
            status = "dev_fixture_loadable_not_release_eligible"
        else:
            status = "dev_fixture_incomplete_not_release_eligible"

        row: dict[str, Any] = {
            "env_name": fixture_name,
            "status": status,
            "fixture_path": str(fixture_dir) if fixture_dir.exists() else None,
            "load_path": str(load_path) if load_path is not None else None,
            "local_note": local_note,
            "release_eligible": False,
            "release_blocker_codes": blocker_codes,
            "source_tree": {
                "data_root": str(data_root),
                "repo_root": str(source_repo_root),
                "commit": commit,
                "dirty_paths": dirty_paths,
                "license_files": _source_tree_license_files(source_repo_root),
            },
            "size_bytes": _dir_size_bytes(fixture_dir),
            "chronics": {
                "chronic_count_basis": "fixture_chronic_directories",
                "count": len(_chronic_names(fixture_dir)),
                "names": _chronic_names(fixture_dir),
            },
            "topology": None,
            "load_error": None,
        }
        if load_local and status == "dev_fixture_loadable_not_release_eligible":
            assert load_path is not None
            try:
                row["topology"] = _load_topology(grid2op_module, load_path)
            except Exception as exc:  # pragma: no cover - depends on local env
                row["status"] = "dev_fixture_load_failed_not_release_eligible"
                row["load_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    summary: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        summary[status] = summary.get(status, 0) + 1

    return {
        "schema_version": "0.1",
        "scope": "grid2op_source_tree_fixture_preflight",
        "non_release_artifact": True,
        "release_ready": False,
        "release_reentry_ready": False,
        "proceed_commands": [],
        "release_eligible": False,
        "fixture_count_basis": "grid2op_source_tree_fixture_rows",
        "n_fixtures": len(rows),
        "source_tree_data_root": str(data_root),
        "load_local": load_local,
        "summary": dict(sorted(summary.items())),
        "fixtures": rows,
        "input_fingerprints": _source_tree_fixture_input_fingerprints(
            data_root=data_root,
            source_repo_root=source_repo_root,
            fixture_names=fixture_names,
            commit=commit,
            dirty_paths=dirty_paths,
        ),
        "next_required_proof": (
            "Use a trusted acquisition gate with verified TLS or a "
            "sha256-locked mirror, inspect and load the acquired archive by "
            "local path, then run source-lock, structural-uniqueness, and "
            "behavioral gates before any release materialization."
        ),
        "safety_contract": (
            "source-tree fixtures are local preflight evidence only; they do "
            "not satisfy release acquisition or manifest source-lock gates"
        ),
    }


def audit_sources(
    env_names: Iterable[str] = DEFAULT_ENVS,
    *,
    data_root: Path | str | None = None,
    grid2op_module: Any | None = None,
    load_local: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable Grid2Op source availability report.

    ``load_local`` never loads remote names. It loads only local directories
    that pass ``_is_complete_env_dir`` and passes the directory path to
    ``grid2op.make``.
    """
    if grid2op_module is None:
        import grid2op as grid2op_module  # type: ignore[import]

    root = (
        Path(data_root).expanduser()
        if data_root is not None
        else _default_data_root(grid2op_module)
    )

    local_envs = set(_safe_call_list(grid2op_module, "list_available_local_env"))
    remote_envs = set(_safe_call_list(grid2op_module, "list_available_remote_env"))
    test_envs = set(_safe_call_list(grid2op_module, "list_available_test_env"))
    remote_metadata = _safe_remote_metadata(grid2op_module)

    rows: list[dict[str, Any]] = []
    for env_name in env_names:
        local_state, load_path, local_note = _local_shape(root, env_name)
        remote_meta = dict(remote_metadata.get(env_name, {}))
        remote_filename = (
            str(remote_meta["filename"]) if remote_meta.get("filename") else None
        )
        partials = _partial_archives(root, env_name, remote_filename)
        remote = env_name in remote_envs
        test_available = env_name in test_envs

        if local_state == "complete":
            status = "local_loadable"
        elif local_state in {"nested_complete", "incomplete"}:
            status = "local_incomplete"
        elif remote:
            status = "remote_only_not_downloaded"
        elif test_available:
            status = "test_env_available"
        else:
            status = "unknown_not_advertised"

        row: dict[str, Any] = {
            "env_name": env_name,
            "status": status,
            "data_root": str(root),
            "local_dir": str(root / env_name) if (root / env_name).exists() else None,
            "load_path": str(load_path) if load_path is not None else None,
            "listed_local": env_name in local_envs,
            "listed_remote": remote,
            "listed_test": test_available,
            "remote_download": (
                {
                    "base_url": remote_meta.get("base_url"),
                    "filename": remote_meta.get("filename"),
                    "url": _download_url(remote_meta),
                    "archive_names": _archive_names(env_name, remote_filename),
                }
                if remote_meta
                else None
            ),
            "partial_archives": partials,
            "local_note": local_note,
            "topology": None,
            "load_error": None,
        }

        if load_local and status == "local_loadable" and load_path is not None:
            try:
                row["topology"] = _load_topology(grid2op_module, load_path)
            except Exception as exc:  # pragma: no cover - depends on local env
                row["status"] = "local_load_failed"
                row["load_error"] = f"{type(exc).__name__}: {exc}"

        rows.append(row)

    summary: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        summary[status] = summary.get(status, 0) + 1

    return {
        "data_root": str(root),
        "load_local": load_local,
        "summary": dict(sorted(summary.items())),
        "sources": rows,
        "safety_contract": (
            "remote env names are never passed to grid2op.make(test=False); "
            "--load-local loads only complete local directories by path"
        ),
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"Grid2Op data root: {report['data_root']}")
    print(f"load_local: {report['load_local']}")
    print(f"summary: {report['summary']}")
    print(f"safety: {report['safety_contract']}")
    for row in report["sources"]:
        print(f"\n- {row['env_name']}: {row['status']}")
        print(
            "  listed: "
            f"local={row['listed_local']} "
            f"remote={row['listed_remote']} "
            f"test={row['listed_test']}"
        )
        if row["local_dir"]:
            print(f"  local_dir: {row['local_dir']}")
        if row["load_path"]:
            print(f"  load_path: {row['load_path']}")
        if row["local_note"]:
            print(f"  note: {row['local_note']}")
        if row["remote_download"]:
            print(f"  remote_download: {row['remote_download']}")
        if row["partial_archives"]:
            archives = ", ".join(
                f"{p['name']} ({p['bytes']} bytes)" for p in row["partial_archives"]
            )
            print(f"  partial_archives: {archives}")
        if row["topology"]:
            print(f"  topology: {row['topology']}")
        if row["load_error"]:
            print(f"  load_error: {row['load_error']}")


def _emit_json(result: dict[str, Any], output_path: Path | None = None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        dest="envs",
        action="append",
        help="Grid2Op env name to inspect. Repeatable; defaults to P1-B candidates.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Grid2Op data root. Defaults to grid2op's DEFAULT_PATH_DATA.",
    )
    parser.add_argument(
        "--load-local",
        action="store_true",
        help="Load only complete local env directories by path to report topology.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of text.",
    )
    parser.add_argument(
        "--download-env",
        help=(
            "Build a controlled archive download plan for one env. Use with "
            "--execute to actually download into .partial."
        ),
    )
    parser.add_argument(
        "--download-url",
        help=(
            "Download an explicitly trusted archive URL into --download-target. "
            "Use only after an acquisition gate has accepted the URL."
        ),
    )
    parser.add_argument(
        "--download-target",
        type=Path,
        help="Target archive path for --download-url.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute --download-env. Without this flag, download mode is dry-run.",
    )
    parser.add_argument(
        "--restart-download",
        action="store_true",
        help="Delete any existing .partial archive before --execute download.",
    )
    parser.add_argument(
        "--download-timeout-s",
        type=float,
        default=300.0,
        help=(
            "Per-socket-read timeout for controlled archive downloads. "
            "Use a positive value so stalled large downloads fail fast and "
            "leave the partial archive for deliberate restart/mirror handling."
        ),
    )
    parser.add_argument(
        "--inspect-archive",
        type=Path,
        help="Inspect a local tar.bz2 archive without extracting it.",
    )
    parser.add_argument(
        "--verify-archive-sha256",
        type=Path,
        help="Verify a local archive against --expected-sha256.",
    )
    parser.add_argument(
        "--preflight-url",
        help="Run a HEAD request to check URL/TLS/HTTP reachability.",
    )
    parser.add_argument(
        "--acquisition-gate-env",
        help=(
            "Dry-run trusted acquisition gate for one env. This combines the "
            "download plan, verified-TLS HEAD preflight, and optional mirror "
            "checksum policy without downloading bytes."
        ),
    )
    parser.add_argument(
        "--mirror-url",
        help="Trusted mirror URL to consider for --acquisition-gate-env.",
    )
    parser.add_argument(
        "--expected-sha256",
        help="Expected 64-hex archive sha256 for --mirror-url.",
    )
    parser.add_argument(
        "--mirror-lock-strategy",
        help="Mirror lock strategy, e.g. institutional_mirror+sha256.",
    )
    parser.add_argument(
        "--source-tree-fixtures-root",
        type=Path,
        help=(
            "Audit Grid2Op source-tree fixture data as a dev-only, non-release "
            "preflight. This never downloads data or modifies release artifacts."
        ),
    )
    parser.add_argument(
        "--source-tree-fixture",
        dest="source_tree_fixtures",
        action="append",
        help=(
            "Fixture env name under --source-tree-fixtures-root. Repeatable; "
            "defaults to known IDF/WCCI source-tree fixtures."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write JSON output to this path as well as stdout.",
    )
    args = parser.parse_args(argv)

    if args.source_tree_fixtures_root:
        result = audit_source_tree_fixtures(
            args.source_tree_fixtures or DEFAULT_SOURCE_TREE_FIXTURES,
            source_tree_data_root=args.source_tree_fixtures_root,
            load_local=args.load_local,
        )
        _emit_json(result, args.output)
        return 0

    if args.acquisition_gate_env:
        result = build_trusted_acquisition_gate(
            args.acquisition_gate_env,
            data_root=args.data_root,
            mirror_url=args.mirror_url,
            expected_sha256=args.expected_sha256,
            mirror_lock_strategy=args.mirror_lock_strategy,
        )
        _emit_json(result, args.output)
        return 0 if result.get("trusted_to_download") else 1

    if args.preflight_url:
        result = preflight_download_url(args.preflight_url)
        _emit_json(result, args.output)
        return 0 if result.get("ok") else 1

    if args.verify_archive_sha256:
        result = verify_archive_checksum(
            args.verify_archive_sha256,
            args.expected_sha256,
        )
        _emit_json(result, args.output)
        return 0 if result.get("matches") else 1

    if args.inspect_archive:
        result = inspect_archive(args.inspect_archive)
        _emit_json(result, args.output)
        return 0

    if args.download_env:
        plan = build_download_plan(args.download_env, data_root=args.data_root)
        if args.execute:
            if plan.get("status") == "blocked_by_root_archive":
                raise SystemExit(json.dumps(plan, indent=2, sort_keys=True))
            if not plan.get("url") or not plan.get("target_path"):
                raise SystemExit(json.dumps(plan, indent=2, sort_keys=True))
            plan["download_result"] = download_archive(
                url=str(plan["url"]),
                target_path=Path(str(plan["target_path"])),
                restart=args.restart_download,
                timeout_s=args.download_timeout_s,
            )
        if args.json or args.execute:
            _emit_json(plan, args.output)
        else:
            _emit_json(plan, args.output)
            print(
                "\nDry-run only. Re-run with --download-env "
                f"{args.download_env} --execute to download the archive."
            )
        return 0

    if args.download_url or args.download_target:
        if not args.download_url or args.download_target is None:
            raise SystemExit(
                "--download-url and --download-target must be used together"
            )
        result: dict[str, Any] = {
            "url": args.download_url,
            "target_path": str(args.download_target),
            "executed": bool(args.execute),
            "note": (
                "This mode trusts the caller's acquisition-gate evidence; it "
                "does not perform TLS/mirror policy validation itself."
            ),
        }
        if args.execute:
            result["download_result"] = download_archive(
                url=args.download_url,
                target_path=args.download_target,
                restart=args.restart_download,
                timeout_s=args.download_timeout_s,
            )
        _emit_json(result, args.output)
        return 0

    report = audit_sources(
        args.envs or DEFAULT_ENVS,
        data_root=args.data_root,
        load_local=args.load_local,
    )
    if args.json:
        _emit_json(report, args.output)
    else:
        if args.output is not None:
            _emit_json(report, args.output)
        _print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
