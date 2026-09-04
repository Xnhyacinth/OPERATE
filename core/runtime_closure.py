"""Fail-closed validation of the backend runtime used by a formal run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _portable_path(raw: object, *, label: str) -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\\" in raw:
        raise ValueError(f"formal backend runtime closure drift:{label}")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != raw:
        raise ValueError(f"formal backend runtime closure drift:{label}")
    return path


def _require_hash(raw: object, *, label: str) -> str:
    value = str(raw or "")
    if _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"formal backend runtime closure drift:{label}")
    return value


def _validate_archived_files(closure: Mapping[str, Any], *, repo_root: Path) -> None:
    links = closure.get("backend_links")
    files = closure.get("archived_files")
    if not isinstance(links, dict) or not isinstance(files, dict):
        raise ValueError("formal backend runtime closure drift:archive contract")

    targets: dict[str, tuple[str, Path]] = {}
    backend_parent: Path | None = None
    for works_name, backend_name in sorted(links.items()):
        if not all(
            isinstance(value, str)
            and value
            and len(PurePosixPath(value).parts) == 1
            and value not in {".", ".."}
            for value in (works_name, backend_name)
        ):
            raise ValueError("formal backend runtime closure drift:backend link")
        link = repo_root / "works" / works_name
        if not link.is_symlink():
            raise ValueError(
                f"formal backend runtime closure drift:backend link:{works_name}"
            )
        target = link.resolve(strict=True)
        try:
            target.relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(
                f"formal backend runtime closure drift:backend link:{works_name}"
            ) from exc
        if target.name != backend_name or target.parent.name != "backends":
            raise ValueError(
                f"formal backend runtime closure drift:backend link:{works_name}"
            )
        if backend_parent is None:
            backend_parent = target.parent
        elif target.parent != backend_parent:
            raise ValueError("formal backend runtime closure drift:backend link roots")
        targets[backend_name] = (works_name, target)

    for archive_name, raw in sorted(files.items()):
        archive_path = _portable_path(archive_name, label="archive path")
        if len(archive_path.parts) < 3 or archive_path.parts[0] != "backends":
            raise ValueError("formal backend runtime closure drift:archive path")
        target_binding = targets.get(archive_path.parts[1])
        if target_binding is None or not isinstance(raw, dict):
            raise ValueError("formal backend runtime closure drift:archive binding")
        works_name, target = target_binding
        source_path = _portable_path(raw.get("source_path"), label="source path")
        expected_source = PurePosixPath("works", works_name, *archive_path.parts[2:])
        allowed_root = target
        if archive_path.parts[:5] == (
            "backends",
            "resco",
            "resco_benchmark",
            "environments",
            "arterial4x4",
        ):
            expected_source = PurePosixPath(
                "sources", "resco", "arterial4x4", *archive_path.parts[5:]
            )
            allowed_root = repo_root / "sources/resco/arterial4x4"
        if source_path != expected_source:
            raise ValueError("formal backend runtime closure drift:source binding")
        actual = repo_root / Path(*source_path.parts)
        expected = _require_hash(raw.get("sha256"), label="archive sha256")
        try:
            resolved = actual.resolve(strict=True)
            resolved.relative_to(allowed_root.resolve(strict=True))
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"formal backend runtime closure drift:archive missing:{source_path}"
            ) from exc
        if not actual.is_file() or actual.is_symlink() or _sha256(actual) != expected:
            raise ValueError(
                f"formal backend runtime closure drift:archive hash:{source_path}"
            )


def _git_value(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("formal backend runtime closure drift:external checkout")
    return result.stdout.strip()


def _validate_external_sources(closure: Mapping[str, Any], *, repo_root: Path) -> None:
    sources = closure.get("external_sources")
    if not isinstance(sources, dict):
        raise ValueError("formal backend runtime closure drift:external sources")
    for source_id, raw in sorted(sources.items()):
        if not isinstance(source_id, str) or not source_id or not isinstance(raw, dict):
            raise ValueError("formal backend runtime closure drift:external source")
        delivery = raw.get("delivery")
        metadata = raw.get("metadata")
        required_files = raw.get("required_files")
        if (
            delivery not in {"git_checkout", "upstream_fetch", "user_provided"}
            or not isinstance(metadata, dict)
            or not isinstance(required_files, dict)
            or not required_files
        ):
            raise ValueError(
                f"formal backend runtime closure drift:external source:{source_id}"
            )
        relative_root = _portable_path(metadata.get("root"), label="external root")
        source_root = repo_root / Path(*relative_root.parts)
        try:
            resolved_root = source_root.resolve(strict=True)
            resolved_root.relative_to(repo_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"formal backend runtime closure drift:external root:{source_id}"
            ) from exc
        if not source_root.is_dir():
            raise ValueError(
                f"formal backend runtime closure drift:external root:{source_id}"
            )
        if delivery == "git_checkout":
            revision = str(raw.get("revision") or "")
            if _git_value(source_root, "rev-parse", "HEAD") != revision:
                raise ValueError(
                    f"formal backend runtime closure drift:external revision:{source_id}"
                )
            if _git_value(
                source_root, "status", "--porcelain", "--untracked-files=all"
            ):
                raise ValueError(
                    f"formal backend runtime closure drift:external dirty:{source_id}"
                )
        for raw_path, raw_digest in sorted(required_files.items()):
            relative = _portable_path(raw_path, label="external file")
            actual = repo_root / Path(*relative.parts)
            expected = _require_hash(raw_digest, label="external file sha256")
            try:
                resolved = actual.resolve(strict=True)
                resolved.relative_to(resolved_root)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(
                    f"formal backend runtime closure drift:external file:{source_id}"
                ) from exc
            if (
                not actual.is_file()
                or actual.is_symlink()
                or _sha256(actual) != expected
            ):
                raise ValueError(
                    f"formal backend runtime closure drift:external hash:{source_id}"
                )


def _validate_repo_files(
    closure: Mapping[str, Any],
    *,
    repo_root: Path,
    field: str,
    label: str,
) -> None:
    files = closure.get(field)
    if not isinstance(files, dict):
        raise ValueError(f"formal backend runtime closure drift:{label} files")
    for raw_path, raw in sorted(files.items()):
        relative = _portable_path(raw_path, label=f"{label} path")
        if not isinstance(raw, dict) or set(raw) != {
            "sha256",
            "roles",
            "backend_kinds",
        }:
            raise ValueError(f"formal backend runtime closure drift:{label} record")
        roles = raw.get("roles")
        backend_kinds = raw.get("backend_kinds")
        if (
            not isinstance(roles, list)
            or not roles
            or not all(isinstance(role, str) and role for role in roles)
            or roles != sorted(set(roles))
            or not isinstance(backend_kinds, list)
            or not backend_kinds
            or not all(isinstance(kind, str) and kind for kind in backend_kinds)
            or backend_kinds != sorted(set(backend_kinds))
        ):
            raise ValueError(f"formal backend runtime closure drift:{label} record")
        actual = repo_root / Path(*relative.parts)
        expected = _require_hash(raw.get("sha256"), label=f"{label} sha256")
        try:
            resolved = actual.resolve(strict=True)
            resolved.relative_to(repo_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                f"formal backend runtime closure drift:{label} missing:{relative}"
            ) from exc
        if not actual.is_file() or actual.is_symlink() or _sha256(actual) != expected:
            raise ValueError(
                f"formal backend runtime closure drift:{label} hash:{relative}"
            )


def _validate_runtime_packages(closure: Mapping[str, Any], *, repo_root: Path) -> None:
    packages = closure.get("runtime_packages")
    if not isinstance(packages, dict):
        raise ValueError("formal backend runtime closure drift:runtime packages")
    if not packages:
        return
    uv_lock = repo_root / "uv.lock"
    if not uv_lock.is_file() or uv_lock.is_symlink():
        raise ValueError("formal backend runtime closure drift:uv lock")
    uv_lock_sha256 = _sha256(uv_lock)
    for package, contract in sorted(packages.items()):
        if (
            not isinstance(package, str)
            or not package
            or not isinstance(contract, dict)
        ):
            raise ValueError("formal backend runtime closure drift:runtime package")
        backend_kinds = contract.get("backend_kinds")
        entries = contract.get("lock_entries")
        if (
            not isinstance(backend_kinds, list)
            or not backend_kinds
            or backend_kinds != sorted(set(backend_kinds))
            or not all(isinstance(kind, str) and kind for kind in backend_kinds)
            or not isinstance(entries, list)
            or not entries
            or contract.get("uv_lock_sha256") != uv_lock_sha256
            or contract.get("lock_entries_sha256") != _canonical_sha256(entries)
        ):
            raise ValueError(
                f"formal backend runtime closure drift:runtime package:{package}"
            )
        versions: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(
                    f"formal backend runtime closure drift:runtime package:{package}"
                )
            identity = entry.get("identity_sha256")
            unsigned = {
                key: value for key, value in entry.items() if key != "identity_sha256"
            }
            version = entry.get("version")
            if (
                identity != _canonical_sha256(unsigned)
                or not isinstance(version, str)
                or not version
            ):
                raise ValueError(
                    f"formal backend runtime closure drift:runtime package:{package}"
                )
            versions.add(version)
        try:
            installed_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ValueError(
                f"formal backend runtime closure drift:runtime package missing:{package}"
            ) from exc
        if installed_version not in versions:
            raise ValueError(
                f"formal backend runtime closure drift:runtime package version:{package}"
            )


def validate_live_backend_runtime_closure(
    *,
    repo_root: Path,
    release_root: Path,
    release_id: str,
    source_suite_sha256: str,
    binding: Mapping[str, Any],
) -> dict[str, str]:
    """Validate the release-bound backend bytes actually visible to adapters."""

    repo_root = repo_root.resolve()
    release_root = release_root.resolve()
    if not isinstance(binding, Mapping):
        raise ValueError("formal backend runtime closure drift:binding missing")
    relative = _portable_path(binding.get("path"), label="closure path")
    if relative.as_posix() != "backend_runtime_closure.json":
        raise ValueError("formal backend runtime closure drift:closure path")
    unresolved_closure_path = release_root / Path(*relative.parts)
    if unresolved_closure_path.is_symlink():
        raise ValueError("formal backend runtime closure drift:closure path")
    closure_path = unresolved_closure_path.resolve()
    try:
        closure_path.relative_to(release_root)
    except ValueError as exc:
        raise ValueError("formal backend runtime closure drift:closure path") from exc
    expected_hash = _require_hash(binding.get("sha256"), label="closure sha256")
    if (
        not closure_path.is_file()
        or closure_path.is_symlink()
        or _sha256(closure_path) != expected_hash
    ):
        raise ValueError("formal backend runtime closure drift:closure hash")
    try:
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("formal backend runtime closure drift:closure JSON") from exc
    if not isinstance(closure, dict):
        raise ValueError("formal backend runtime closure drift:closure object")
    identity = _require_hash(closure.get("identity_sha256"), label="identity")
    unsigned = {
        key: value for key, value in closure.items() if key != "identity_sha256"
    }
    if (
        closure.get("schema_version") != "operate-backend-runtime-closure-v1"
        or closure.get("release_id") != release_id
        or closure.get("status") != "backend_runtime_closure_complete"
        or closure.get("terminal") is not True
        or closure.get("portable") is not True
        or closure.get("source_suite_sha256") != source_suite_sha256
        or _canonical_sha256(unsigned) != identity
        or binding.get("identity_sha256") != identity
    ):
        raise ValueError("formal backend runtime closure drift:closure identity")
    summary = closure.get("summary")
    if (
        not isinstance(summary, dict)
        or any(
            summary.get(field) != len(closure.get(key) or {})
            for field, key in (
                ("n_archived_files", "archived_files"),
                ("n_external_sources", "external_sources"),
                ("n_backend_links", "backend_links"),
                ("n_repo_tracked_files", "repo_tracked_files"),
                ("n_runtime_packages", "runtime_packages"),
                ("n_separately_bundled_files", "separately_bundled_files"),
            )
        )
        or summary.get("n_unresolved") != 0
    ):
        raise ValueError("formal backend runtime closure drift:summary")
    expected_binding = {
        "path": relative.as_posix(),
        "sha256": expected_hash,
        "schema_version": closure["schema_version"],
        "n_archived_files": summary["n_archived_files"],
        "n_external_sources": summary["n_external_sources"],
        "n_backend_links": summary["n_backend_links"],
        "n_runtime_packages": summary["n_runtime_packages"],
        "identity_sha256": identity,
    }
    if dict(binding) != expected_binding:
        raise ValueError("formal backend runtime closure drift:binding")
    _validate_archived_files(closure, repo_root=repo_root)
    _validate_external_sources(closure, repo_root=repo_root)
    _validate_repo_files(
        closure,
        repo_root=repo_root,
        field="repo_tracked_files",
        label="repo tracked",
    )
    _validate_repo_files(
        closure,
        repo_root=repo_root,
        field="separately_bundled_files",
        label="separately bundled",
    )
    _validate_runtime_packages(closure, repo_root=repo_root)
    return {
        "path": str(closure_path),
        "sha256": expected_hash,
        "identity_sha256": identity,
    }
