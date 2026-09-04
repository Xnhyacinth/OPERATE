"""Shared fail-closed helpers for protocol-2.1 evidence artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from core.implementation_identity import implementation_identity
from evaluation.scorer import SCORING_VERSION
from runner.episode import (
    EVALUATION_IMPLEMENTATION_FINGERPRINT,
    EVALUATION_PROTOCOL_VERSION,
)

Identity = tuple[str, str]
REPO_ROOT = Path(__file__).resolve().parents[1]


def _portable_repo_string(value: str, *, repo_root: Path) -> str:
    """Lexically relativize repository-owned paths, including diagnostics."""
    path = Path(value)
    root = Path(os.path.abspath(os.path.normpath(repo_root)))
    if path.is_absolute():
        lexical = Path(os.path.normpath(value))
        try:
            return lexical.relative_to(root).as_posix()
        except ValueError:
            return value

    root_prefix = f"{root.as_posix()}/"
    return value.replace(root_prefix, "")


def canonicalize_repo_owned_paths(
    payload: Any,
    *,
    repo_root: Path | None = None,
) -> Any:
    """Recursively make repository-owned absolute paths clone-portable.

    Both mapping keys and values are covered.  The conversion is lexical and
    therefore never follows symlinks or requires the referenced path to exist.
    """
    root = repo_root or REPO_ROOT
    if isinstance(payload, str):
        return _portable_repo_string(payload, repo_root=root)
    if isinstance(payload, dict):
        canonical: dict[Any, Any] = {}
        for key, value in payload.items():
            canonical_key = canonicalize_repo_owned_paths(key, repo_root=root)
            canonical_value = canonicalize_repo_owned_paths(value, repo_root=root)
            if canonical_key in canonical and canonical[canonical_key] != canonical_value:
                raise ValueError(
                    f"canonical path key collision: {canonical_key}"
                )
            canonical[canonical_key] = canonical_value
        return canonical
    if isinstance(payload, list):
        return [
            canonicalize_repo_owned_paths(value, repo_root=root)
            for value in payload
        ]
    if isinstance(payload, tuple):
        return tuple(
            canonicalize_repo_owned_paths(value, repo_root=root)
            for value in payload
        )
    return payload


def portable_repo_path(path: Path, *, repo_root: Path | None = None) -> str:
    """Return a clone-portable path for repository-owned evidence."""
    resolved = path.resolve()
    root = (repo_root or REPO_ROOT).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def resolve_binding_path(
    raw_path: str | Path,
    *,
    repo_root: Path | None = None,
) -> Path:
    """Resolve a portable binding while rejecting repository traversal."""
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    root = (repo_root or REPO_ROOT).resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact binding escapes repository: {raw_path}") from exc
    return resolved


def required_semantics() -> dict[str, str]:
    return {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        "scoring_version": SCORING_VERSION,
    }


def extract_semantics(report: dict[str, Any]) -> dict[str, str]:
    raw = report.get("evaluation_semantics") or {}
    protocol = report.get("evaluation_protocol") or {}
    config = report.get("config") or {}
    return {
        "protocol_version": str(
            raw.get("protocol_version")
            or raw.get("evaluation_protocol_version")
            or raw.get("version")
            or protocol.get("version")
            or report.get("evaluation_protocol_version")
            or config.get("evaluation_protocol_version")
            or ""
        ),
        "implementation_fingerprint": str(
            raw.get("implementation_fingerprint")
            or raw.get("evaluation_implementation_fingerprint")
            or protocol.get("implementation_fingerprint")
            or report.get("evaluation_implementation_fingerprint")
            or config.get("evaluation_implementation_fingerprint")
            or ""
        ),
        "scoring_version": str(
            raw.get("scoring_version")
            or report.get("scoring_version")
            or config.get("scoring_version")
            or ""
        ),
    }


def report_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "samples", "scenarios", "items"):
        rows = report.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def row_identity(row: dict[str, Any]) -> Identity:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def group_rows_by_identity(
    report: dict[str, Any],
) -> dict[Identity, list[dict[str, Any]]]:
    grouped: dict[Identity, list[dict[str, Any]]] = defaultdict(list)
    for row in report_rows(report):
        grouped[row_identity(row)].append(row)
    return dict(grouped)


def artifact_binding(
    path: Path,
    *,
    implementation_tree_sha256: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    tree_hash = (
        implementation_tree_sha256
        or str(payload.get("implementation_tree_sha256") or "")
        or implementation_identity()["implementation_tree_sha256"]
    )
    return {
        "path": portable_repo_path(resolved, repo_root=repo_root),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "implementation_tree_sha256": tree_hash,
    }


def verify_artifact_binding(
    binding: dict[str, Any],
    path: Path,
    *,
    implementation_tree_sha256: str,
    repo_root: Path | None = None,
) -> list[str]:
    if not binding:
        return ["artifact_binding_missing"]
    actual = artifact_binding(
        path,
        implementation_tree_sha256=implementation_tree_sha256,
        repo_root=repo_root,
    )
    errors: list[str] = []
    try:
        declared_path = resolve_binding_path(
            str(binding.get("path") or ""), repo_root=repo_root
        )
    except ValueError:
        declared_path = None
    if declared_path != path.resolve():
        errors.append("artifact_path_mismatch")
    if binding.get("sha256") != actual["sha256"]:
        errors.append("artifact_hash_mismatch")
    if (
        binding.get("implementation_tree_sha256")
        != implementation_tree_sha256
    ):
        errors.append("implementation_tree_mismatch")
    return errors


def validate_report_scope(
    report: dict[str, Any],
    expected_identities: Iterable[Identity],
    *,
    implementation_tree_sha256: str,
    complexity_agents: tuple[str, ...] | None = None,
) -> list[str]:
    expected = list(expected_identities)
    expected_set = set(expected)
    grouped = group_rows_by_identity(report)
    errors: list[str] = []
    if extract_semantics(report) != required_semantics():
        errors.append("artifact_semantics_stale")
    if report.get("status") != "complete" and report.get("complete") is not True:
        errors.append("artifact_incomplete")
    if (
        str(report.get("implementation_tree_sha256") or "")
        != implementation_tree_sha256
    ):
        errors.append("implementation_tree_mismatch")
    expected_rows = len(expected) * (
        len(complexity_agents) if complexity_agents is not None else 1
    )
    if int(report.get("n_expected", -1)) != expected_rows:
        errors.append("artifact_expected_count_mismatch")
    if int(report.get("n_completed", -1)) != expected_rows:
        errors.append("artifact_completed_count_mismatch")
    if set(grouped) != expected_set:
        errors.append("artifact_identity_scope_mismatch")
    for identity, rows in grouped.items():
        if not all(identity):
            errors.append("artifact_identity_incomplete")
        if complexity_agents is None:
            if len(rows) != 1:
                errors.append("artifact_identity_duplicate")
        else:
            agents = [
                str(row.get("agent_name") or row.get("agent") or "")
                for row in rows
            ]
            if sorted(agents) != sorted(complexity_agents):
                errors.append("complexity_agent_scope_mismatch")
    skipped = report.get("skipped") or []
    if any(
        isinstance(row, dict) and not row.get("reason")
        for row in skipped
    ):
        errors.append("skipped_identity_reason_missing")
    return sorted(set(errors))
