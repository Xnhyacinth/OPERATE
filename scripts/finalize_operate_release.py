#!/usr/bin/env python3
"""Finalize an OPERATE release from immutable formal result trees.

This command is intentionally fail closed. Formal results are checked against
the pending release identity, copied into content-addressed directories inside
that release, and revalidated from the copied bytes before release state may
change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_release_integrity import (  # noqa: E402
    _logical_published_manifest_valid,
    _realtime_published_manifest_valid,
    build_release_integrity_report,
)

TREE_INDEX_SCHEMA_VERSION = "operate-formal-result-tree-index-v1"
TREE_INDEX_NAME = "FORMAL_RESULT_TREE_INDEX.json"
FORMAL_DISTRIBUTION_RECEIPT_NAME = "formal_distribution_receipt.json"
FORMAL_DISTRIBUTION_RECEIPT_SCHEMA = "operate-formal-distribution-receipt-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ReleaseFinalizationError(ValueError):
    """The supplied evidence cannot safely finalize the release."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _manifest_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    """Hash one unchanged regular file without following a final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseFinalizationError(f"cannot safely open regular file: {path}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReleaseFinalizationError(f"not a regular file: {path}")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ReleaseFinalizationError(f"file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseFinalizationError(f"{label} must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseFinalizationError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ReleaseFinalizationError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_distribution_receipt(
    candidate: dict[str, Any],
    *,
    bundle_manifest_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Validate the receipt emitted only after an exact private CAS upload."""

    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise ReleaseFinalizationError("formal distribution receipt file is missing")
    if bundle_manifest_path.is_symlink() or not bundle_manifest_path.is_file():
        raise ReleaseFinalizationError("formal distribution bundle manifest is missing")
    receipt = _load_json_object(receipt_path, label="formal distribution receipt")
    bundle = _load_json_object(
        bundle_manifest_path, label="formal distribution bundle manifest"
    )
    receipt_without_hash = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    if receipt.get("receipt_sha256") != _canonical_sha256(receipt_without_hash):
        raise ReleaseFinalizationError(
            "formal distribution receipt self-hash mismatch"
        )
    expected_keys = {
        "schema_version",
        "release_id",
        "hf_repo_id",
        "visibility",
        "revision",
        "verification",
        "bundle_manifest_sha256",
        "release_manifest_sha256",
        "formal_evidence_archive",
        "formal_evidence_archive_sha256",
        "formal_result_tree_roots",
        "receipt_sha256",
    }
    revision = str(receipt.get("revision") or "")
    archive = bundle.get("formal_evidence_archive")
    files = bundle.get("files")
    trees = bundle.get("formal_result_trees")
    evidence = candidate.get("formal_evidence")
    if not (
        set(receipt) == expected_keys
        and receipt.get("schema_version") == FORMAL_DISTRIBUTION_RECEIPT_SCHEMA
        and receipt.get("release_id") == candidate.get("release_id")
        and receipt.get("release_id") == bundle.get("release_id")
        and receipt.get("hf_repo_id") == bundle.get("hf_repo_id")
        and receipt.get("visibility") == bundle.get("visibility") == "private"
        and receipt.get("verification") == "private_cas_exact_snapshot_v1"
        and re.fullmatch(r"[0-9a-f]{40,64}", revision) is not None
        and set(revision) != {"0"}
        and receipt.get("bundle_manifest_sha256")
        == _file_sha256(bundle_manifest_path)
        and receipt.get("release_manifest_sha256")
        == bundle.get("release_manifest_sha256")
        == _manifest_payload_sha256(candidate)
        and isinstance(archive, str)
        and isinstance(files, dict)
        and receipt.get("formal_evidence_archive") == archive
        and receipt.get("formal_evidence_archive_sha256") == files.get(archive)
        and isinstance(trees, dict)
        and set(trees) == {"logical_persistent", "realtime_persistent"}
        and isinstance(evidence, dict)
    ):
        raise ReleaseFinalizationError("formal distribution receipt mismatch")
    roots: dict[str, str] = {}
    for mode, evidence_name in (
        ("logical_persistent", "logical_batch_manifest"),
        ("realtime_persistent", "realtime_batch_manifest"),
    ):
        contract = trees[mode]
        binding = contract.get("binding") if isinstance(contract, dict) else None
        tree_files = contract.get("files") if isinstance(contract, dict) else None
        if not (
            isinstance(binding, dict)
            and binding == evidence.get(evidence_name)
            and isinstance(tree_files, dict)
            and tree_files
        ):
            raise ReleaseFinalizationError(
                f"formal distribution result tree mismatch: {mode}"
            )
        roots[mode] = str(binding.get("tree_root_sha256") or "")
    if receipt.get("formal_result_tree_roots") != roots:
        raise ReleaseFinalizationError("formal distribution result roots mismatch")
    return receipt


def _contained_file(release_dir: Path, binding: object, *, label: str) -> Path:
    if not isinstance(binding, dict):
        raise ReleaseFinalizationError(f"{label} binding is missing")
    relative = Path(str(binding.get("path") or ""))
    if relative.is_absolute() or ".." in relative.parts:
        raise ReleaseFinalizationError(f"{label} path is invalid")
    path = (release_dir / relative).resolve()
    try:
        path.relative_to(release_dir.resolve())
    except ValueError as exc:
        raise ReleaseFinalizationError(f"{label} path escapes release") from exc
    if path.is_symlink() or not path.is_file():
        raise ReleaseFinalizationError(f"{label} file is missing")
    if binding.get("sha256") != _file_sha256(path):
        raise ReleaseFinalizationError(f"{label} hash binding is stale")
    return path


def _release_rows(
    manifest: dict[str, Any], *, release_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    core_path = _contained_file(
        release_dir, manifest.get("core_suite"), label="core_suite"
    )
    core = _load_json_object(core_path, label="core_suite")
    rows = core.get("scenarios")
    binding = manifest["core_suite"]
    if (
        not isinstance(rows, list)
        or not rows
        or not all(isinstance(row, dict) for row in rows)
        or binding.get("n_scenarios") != len(rows)
        or core.get("n_scenarios") != len(rows)
    ):
        raise ReleaseFinalizationError("release Core binding is invalid")
    return core, list(rows)


def _validate_pending_release(
    release_dir: Path, manifest: dict[str, Any], *, repo_root: Path
) -> None:
    if not (
        manifest.get("status") == "formal_evaluation_ready"
        and manifest.get("formal_evaluation_ready") is True
        and manifest.get("public_release_ready") is False
        and manifest.get("leaderboard_eligible") is False
    ):
        raise ReleaseFinalizationError("release is not in the pending formal state")
    evidence = manifest.get("formal_evidence")
    if not isinstance(evidence, dict) or any(
        name in evidence
        for name in ("logical_batch_manifest", "realtime_batch_manifest")
    ):
        raise ReleaseFinalizationError("release formal evidence is missing or already final")
    for portable, label in ((False, "full"), (True, "portable")):
        report = build_release_integrity_report(
            release_dir, portable=portable, artifact_root=repo_root
        )
        if report.get("ok") is not True or report.get("formal_run_ready") is not True:
            raise ReleaseFinalizationError(
                f"pending release {label} integrity is not green"
            )


def _runtime_identity(
    manifest: dict[str, Any], *, release_dir: Path, manifest_sha256: str
) -> dict[str, str]:
    """Read immutable qualification identity; execution code is bound separately."""

    replay = manifest.get("protocol21_replay")
    candidate = manifest.get("candidate_closure")
    backend = manifest.get("backend_runtime_closure")
    if not all(isinstance(value, dict) for value in (replay, candidate, backend)):
        raise ReleaseFinalizationError("release runtime closure bindings are incomplete")
    assert isinstance(replay, dict)
    assert isinstance(candidate, dict)
    assert isinstance(backend, dict)
    runtime_path = _contained_file(
        release_dir,
        manifest.get("formal_runtime_bundle"),
        label="formal_runtime_bundle",
    )
    core_path = _contained_file(
        release_dir, manifest.get("core_suite"), label="core_suite"
    )
    candidate_path = _contained_file(
        release_dir, candidate, label="candidate_closure"
    )
    backend_path = _contained_file(
        release_dir, backend, label="backend_runtime_closure"
    )
    source_path = release_dir / "protocol21_source_suite.json"
    evidence_path = release_dir / "protocol21_public_evidence_bundle.json"
    for path, expected, label in (
        (source_path, replay.get("source_suite_sha256"), "source_suite"),
        (evidence_path, replay.get("evidence_bundle_sha256"), "public_evidence"),
    ):
        if path.is_symlink() or not path.is_file() or _file_sha256(path) != expected:
            raise ReleaseFinalizationError(f"{label} hash binding is stale")
    runtime = _load_json_object(runtime_path, label="formal_runtime_bundle")
    evidence = _load_json_object(evidence_path, label="public_evidence")
    candidate_payload = _load_json_object(candidate_path, label="candidate_closure")
    backend_payload = _load_json_object(backend_path, label="backend_runtime_closure")
    expected = {
        "release_id": str(manifest.get("release_id") or ""),
        "formal_manifest_sha256": manifest_sha256,
        "formal_runtime_bundle_sha256": _file_sha256(runtime_path),
        "formal_core_suite_sha256": _file_sha256(core_path),
        "formal_source_suite_sha256": _file_sha256(source_path),
        "formal_public_evidence_sha256": _file_sha256(evidence_path),
        "formal_public_evidence_binding_root_sha256": str(
            evidence.get("binding_root_sha256") or ""
        ),
        "formal_candidate_closure_sha256": _file_sha256(candidate_path),
        "formal_candidate_closure_identity_sha256": _canonical_sha256(
            candidate_payload.get("identity_set_sha256")
        ),
        "formal_backend_runtime_closure_sha256": _file_sha256(backend_path),
        "formal_backend_runtime_closure_identity_sha256": str(
            backend_payload.get("identity_sha256") or ""
        ),
        "implementation_tree_sha256": str(
            manifest.get("implementation_tree_sha256") or ""
        ),
        "formal_core_release_pipeline_sha256": str(
            manifest.get("core_release_pipeline_sha256") or ""
        ),
        "formal_release_tooling_sha256": str(
            manifest.get("release_tooling_sha256") or ""
        ),
    }
    if not expected["release_id"] or any(
        _SHA256_RE.fullmatch(value) is None
        for name, value in expected.items()
        if name != "release_id"
    ):
        raise ReleaseFinalizationError("release runtime identity is incomplete")
    if not (
        runtime.get("release_id") == expected["release_id"]
        and runtime.get("implementation_tree_sha256")
        == expected["implementation_tree_sha256"]
        and runtime.get("core_release_pipeline_sha256")
        == expected["formal_core_release_pipeline_sha256"]
        and runtime.get("release_tooling_sha256")
        == expected["formal_release_tooling_sha256"]
    ):
        raise ReleaseFinalizationError("formal runtime bundle identity is stale")
    return expected


def _require_identity(
    actual: object, expected: dict[str, str], *, label: str
) -> None:
    if not isinstance(actual, dict):
        raise ReleaseFinalizationError(f"{label} formal runtime binding is missing")
    missing = sorted(
        key
        for key in expected
        if key not in actual or actual.get(key) is None or actual.get(key) == ""
    )
    if missing:
        raise ReleaseFinalizationError(
            f"{label} runner formal binding is incomplete: {', '.join(missing)}"
        )
    mismatch = sorted(
        key for key, value in expected.items() if str(actual.get(key)) != value
    )
    if mismatch:
        raise ReleaseFinalizationError(
            f"{label} formal runtime identity mismatch: {', '.join(mismatch)}"
        )


def _validate_batch_identities(
    logical: dict[str, Any], realtime: dict[str, Any], expected: dict[str, str]
) -> None:
    if not (
        logical.get("release_id")
        == logical.get("formal_release_id")
        == expected["release_id"]
    ):
        raise ReleaseFinalizationError(
            "logical duplicate release identity mismatch"
        )
    _require_identity(
        {key: logical.get(key) for key in expected}, expected, label="logical"
    )
    identity = realtime.get("batch_treatment_identity")
    if not isinstance(identity, dict):
        raise ReleaseFinalizationError("realtime treatment identity is missing")
    binding = dict(identity.get("formal_runtime_binding") or {})
    if not (
        identity.get("formal_release_id")
        == binding.get("release_id")
        == expected["release_id"]
    ):
        raise ReleaseFinalizationError(
            "realtime duplicate release identity mismatch"
        )
    actual = {
        **binding,
        "release_id": identity.get("formal_release_id"),
        "formal_manifest_sha256": identity.get("formal_manifest_sha256"),
        "implementation_tree_sha256": identity.get("implementation_tree_sha256"),
        "formal_core_release_pipeline_sha256": binding.get(
            "core_release_pipeline_sha256"
        ),
        "formal_backend_runtime_closure_identity_sha256": binding.get(
            "backend_runtime_closure_identity_sha256"
        ),
        "formal_release_tooling_sha256": binding.get("release_tooling_sha256"),
    }
    _require_identity(actual, expected, label="realtime")


def _tree_index(root: Path, *, ignore_index: bool = False) -> dict[str, Any]:
    """Return a canonical index; symlinks and special files are fatal."""

    if root.is_symlink() or not root.is_dir():
        raise ReleaseFinalizationError(f"formal result root is not a directory: {root}")
    root = root.resolve()
    files: list[dict[str, Any]] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise ReleaseFinalizationError(
                    f"formal result tree contains unsafe directory: {path}"
                )
        for name in names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if ignore_index and relative == TREE_INDEX_NAME:
                continue
            if path.is_symlink() or not path.is_file():
                raise ReleaseFinalizationError(
                    f"formal result tree contains unsafe file: {path}"
                )
            files.append(
                {
                    "path": relative,
                    "sha256": _file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    files.sort(key=lambda item: item["path"])
    payload = {"schema_version": TREE_INDEX_SCHEMA_VERSION, "files": files}
    payload["root_sha256"] = _canonical_sha256(payload)
    return payload


def _copy_indexed_tree(source: Path, destination: Path, index: dict[str, Any]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for item in index["files"]:
        relative = Path(item["path"])
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target, follow_symlinks=False)
    if _tree_index(source) != index or _tree_index(destination) != index:
        raise ReleaseFinalizationError(
            "formal result tree changed during materialization"
        )
    _atomic_write_json(destination / TREE_INDEX_NAME, index)


def _materialize_result_tree(
    source_manifest: Path,
    *,
    release_dir: Path,
    model: str,
    treatment_sha256: str,
    validator: Callable[[Path, dict[str, Any]], bool],
) -> tuple[Path, dict[str, Any], bool]:
    source_root = source_manifest.parent.resolve()
    index = _tree_index(source_root)
    destination = (
        release_dir
        / "formal_results"
        / quote(model, safe="._-")
        / treatment_sha256
        / index["root_sha256"]
    )
    if destination.exists():
        if destination.is_symlink() or _tree_index(
            destination, ignore_index=True
        ) != index:
            raise ReleaseFinalizationError(
                "existing formal result content address is corrupt"
            )
        if _load_json_object(
            destination / TREE_INDEX_NAME, label="formal result tree index"
        ) != index:
            raise ReleaseFinalizationError(
                "existing formal result content address has a stale index"
            )
        copied_manifest = destination / "RUN_MANIFEST.json"
        payload = _load_json_object(copied_manifest, label="materialized batch")
        if not validator(copied_manifest, payload):
            raise ReleaseFinalizationError(
                "existing materialized batch failed strict validation"
            )
        return copied_manifest, index, False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{index['root_sha256']}.{uuid.uuid4().hex}.tmp"
    try:
        _copy_indexed_tree(source_root, temporary, index)
        copied_manifest = temporary / "RUN_MANIFEST.json"
        payload = _load_json_object(copied_manifest, label="materialized batch")
        if not validator(copied_manifest, payload):
            raise ReleaseFinalizationError(
                "materialized batch failed validation; runner artifact paths must "
                "be batch-root-relative and verifier-portable"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "RUN_MANIFEST.json", index, True


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _contained_artifact_path(raw: object, *, batch_root: Path, label: str) -> Path:
    path = Path(str(raw or ""))
    if not str(raw or ""):
        raise ReleaseFinalizationError(f"{label} path is missing")
    resolved = path.resolve() if path.is_absolute() else (batch_root / path).resolve()
    try:
        resolved.relative_to(batch_root.resolve())
    except ValueError as exc:
        raise ReleaseFinalizationError(f"{label} escapes its batch root") from exc
    return resolved


def _assert_batch_artifacts_contained(
    manifest_path: Path, payload: dict[str, Any], *, interaction_mode: str
) -> None:
    """Check every published/nested path that validators subsequently reopen."""

    root = manifest_path.parent.resolve()
    if interaction_mode == "logical_persistent":
        published = payload.get("published_artifacts") or {}
        for name in ("episodes", "leaderboard"):
            binding = published.get(name) or {}
            path = _contained_artifact_path(
                binding.get("path"), batch_root=root, label=f"logical {name}"
            )
            if name != "episodes":
                continue
            try:
                rows = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReleaseFinalizationError("logical episodes are unreadable") from exc
            for row in rows:
                summary = row.get("trajectory_summary") if isinstance(row, dict) else {}
                if not isinstance(summary, dict):
                    raise ReleaseFinalizationError("logical trajectory summary is invalid")
                _contained_artifact_path(
                    summary.get("trajectory_path"),
                    batch_root=root,
                    label="logical trajectory prefix",
                )
                for key, value in summary.items():
                    if isinstance(value, dict) and "path" in value:
                        _contained_artifact_path(
                            value.get("path"),
                            batch_root=root,
                            label=f"logical {key}",
                        )
        return
    if interaction_mode != "realtime_persistent":
        raise ReleaseFinalizationError("unsupported formal interaction mode")
    artifacts = payload.get("artifacts") or {}
    jsonl_paths: list[Path] = []
    for name in ("episodes_journal", "episodes", "realtime_scorecard", "leaderboard"):
        binding = artifacts.get(name) or {}
        path = _contained_artifact_path(
            binding.get("path"), batch_root=root, label=f"realtime {name}"
        )
        if name in {"episodes_journal", "episodes"}:
            jsonl_paths.append(path)
    for binding in artifacts.get("episode_artifacts") or []:
        if not isinstance(binding, dict):
            raise ReleaseFinalizationError("realtime episode artifact binding is invalid")
        _contained_artifact_path(
            binding.get("artifact_path"),
            batch_root=root,
            label="realtime episode artifact",
        )
    for path in jsonl_paths:
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReleaseFinalizationError("realtime episode ledger is unreadable") from exc
        for row in rows:
            if not isinstance(row, dict):
                raise ReleaseFinalizationError("realtime episode ledger row is invalid")
            for key in ("artifact_path", "trajectory_dir"):
                if row.get(key):
                    _contained_artifact_path(
                        row[key], batch_root=root, label=f"realtime row {key}"
                    )


def _evidence_binding(
    path: Path,
    *,
    repo_root: Path,
    payload: dict[str, Any],
    model: str,
    interaction_mode: str,
    treatment_sha256: str,
    index: dict[str, Any],
) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
        index_relative = (
            (path.parent / TREE_INDEX_NAME)
            .resolve()
            .relative_to(repo_root.resolve())
            .as_posix()
        )
    except ValueError as exc:
        raise ReleaseFinalizationError("materialized evidence is outside repository") from exc
    return {
        "path": relative,
        "sha256": _file_sha256(path),
        "schema_version": payload.get("schema_version"),
        "model": model,
        "interaction_mode": interaction_mode,
        "treatment_sha256": treatment_sha256,
        "tree_index_path": index_relative,
        "tree_index_sha256": _file_sha256(path.parent / TREE_INDEX_NAME),
        "tree_root_sha256": index["root_sha256"],
    }


def _validate_candidate_manifest(
    candidate: dict[str, Any],
    *,
    release_dir: Path,
    repo_root: Path,
) -> None:
    """Validate candidate bytes against the complete live repository in memory."""

    failures = []
    for portable, label in ((False, "full"), (True, "portable")):
        report = build_release_integrity_report(
            release_dir,
            portable=portable,
            artifact_root=repo_root,
            manifest_override=candidate,
        )
        if report.get("ok") is not True:
            failures.append(label)
    if failures:
        raise ReleaseFinalizationError(
            "candidate integrity failed: " + ", ".join(failures)
        )


def finalize_release_manifest(
    *,
    release_manifest_path: Path,
    logical_batch_manifest_path: Path,
    realtime_batch_manifest_path: Path,
    output_manifest_path: Path,
    distribution_bundle_manifest_path: Path | None = None,
    distribution_receipt_path: Path | None = None,
    prepare_distribution_candidate: bool = False,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Prepare or publish a candidate backed by both formal treatments.

    Preparation materializes and validates the immutable result trees, then
    writes the exact manifest bytes that must be bundled and uploaded.  The
    publishing pass deterministically reconstructs those bytes and requires a
    receipt produced only after the private CAS snapshot was revalidated.
    """

    repo_root = repo_root.resolve()
    release_manifest_path = release_manifest_path.resolve()
    output_manifest_path = output_manifest_path.resolve()
    try:
        relative = release_manifest_path.relative_to(repo_root)
    except ValueError as exc:
        raise ReleaseFinalizationError("release manifest is outside repository") from exc
    if (
        relative.parts[:1] != ("release",)
        or len(relative.parts) != 3
        or relative.name != "manifest.json"
    ):
        raise ReleaseFinalizationError(
            "release manifest must be release/<id>/manifest.json"
        )
    release_dir = release_manifest_path.parent
    fixed_receipt_path = release_dir / FORMAL_DISTRIBUTION_RECEIPT_NAME
    if prepare_distribution_candidate:
        if output_manifest_path == release_manifest_path:
            raise ReleaseFinalizationError(
                "distribution candidate must not replace the pending release manifest"
            )
        try:
            output_manifest_path.relative_to(repo_root)
        except ValueError as exc:
            raise ReleaseFinalizationError(
                "distribution candidate must be staged inside the repository"
            ) from exc
        try:
            output_manifest_path.relative_to(release_dir)
        except ValueError:
            pass
        else:
            raise ReleaseFinalizationError(
                "distribution candidate must be staged outside the release directory"
            )
        if (
            distribution_bundle_manifest_path is not None
            or distribution_receipt_path is not None
        ):
            raise ReleaseFinalizationError(
                "distribution inputs are not accepted while preparing a candidate"
            )
    else:
        if output_manifest_path != release_manifest_path:
            raise ReleaseFinalizationError(
                "publishing must atomically replace the canonical release manifest"
            )
        if distribution_bundle_manifest_path is None or distribution_receipt_path is None:
            raise ReleaseFinalizationError(
                "verified formal distribution bundle and receipt are required"
            )
        if distribution_receipt_path.resolve() != fixed_receipt_path.resolve():
            raise ReleaseFinalizationError(
                "formal distribution receipt must use the fixed release path"
            )
    if output_manifest_path.exists() and output_manifest_path != release_manifest_path:
        raise ReleaseFinalizationError("output manifest already exists")
    manifest = _load_json_object(release_manifest_path, label="release manifest")
    manifest_sha = _file_sha256(release_manifest_path)
    if manifest.get("release_id") != release_dir.name:
        raise ReleaseFinalizationError("release identity does not match its directory")
    _validate_pending_release(release_dir, manifest, repo_root=repo_root)
    _core, rows = _release_rows(manifest, release_dir=release_dir)
    qualification_identity = _runtime_identity(
        manifest, release_dir=release_dir, manifest_sha256=manifest_sha
    )
    logical_path = logical_batch_manifest_path.resolve()
    realtime_path = realtime_batch_manifest_path.resolve()
    logical = _load_json_object(logical_path, label="logical batch manifest")
    realtime = _load_json_object(realtime_path, label="realtime batch manifest")
    if logical_path.name != "RUN_MANIFEST.json" or realtime_path.name != "RUN_MANIFEST.json":
        raise ReleaseFinalizationError(
            "formal batch inputs must be named RUN_MANIFEST.json"
        )
    if logical_path == realtime_path:
        raise ReleaseFinalizationError("logical and realtime evidence must be distinct")
    run_tree = str(logical.get("implementation_tree_sha256") or "")
    if _SHA256_RE.fullmatch(run_tree) is None:
        raise ReleaseFinalizationError("logical execution implementation tree is invalid")
    expected = {**qualification_identity, "implementation_tree_sha256": run_tree}
    _validate_batch_identities(logical, realtime, expected)

    tree = run_tree
    logical_contract = manifest.get("formal_batch_contract")
    run_contract = manifest.get("formal_run_contract")
    realtime_contract = manifest.get("formal_realtime_batch_contract")
    if not all(
        isinstance(value, dict)
        for value in (logical_contract, run_contract, realtime_contract)
    ):
        raise ReleaseFinalizationError(
            "release formal treatment contracts are incomplete"
        )
    assert isinstance(logical_contract, dict)
    assert isinstance(run_contract, dict)
    assert isinstance(realtime_contract, dict)
    suite_sha = realtime_contract.get("suite_manifest_sha256")

    def logical_valid(path: Path, payload: dict[str, Any]) -> bool:
        return _logical_published_manifest_valid(
            payload,
            manifest_path=path,
            tree=tree,
            suite_sha256=suite_sha,
            suite_rows=rows,
            logical_contract=logical_contract,
        )

    def realtime_valid(path: Path, payload: dict[str, Any]) -> bool:
        return _realtime_published_manifest_valid(
            payload,
            manifest_path=path,
            tree=tree,
            suite_sha256=suite_sha,
            suite_rows=rows,
            run_contract=run_contract,
            realtime_contract=realtime_contract,
        )

    if not logical_valid(logical_path, logical):
        raise ReleaseFinalizationError("logical batch failed strict formal validation")
    if not realtime_valid(realtime_path, realtime):
        raise ReleaseFinalizationError("realtime batch failed strict formal validation")
    models = logical.get("models")
    model = models[0] if isinstance(models, list) and len(models) == 1 else None
    if not isinstance(model, str) or not model or realtime.get("model") != model:
        raise ReleaseFinalizationError(
            "formal treatments must cover the same single model"
        )
    logical_treatment = (logical.get("agent_treatment_sha256_by_model") or {}).get(
        model
    )
    realtime_treatment = realtime.get("batch_treatment_sha256")
    if not all(
        isinstance(value, str) and _SHA256_RE.fullmatch(value)
        for value in (logical_treatment, realtime_treatment)
    ):
        raise ReleaseFinalizationError("formal treatment hashes are incomplete")

    created: list[Path] = []
    try:
        logical_copy, logical_index, was_created = _materialize_result_tree(
            logical_path,
            release_dir=release_dir,
            model=model,
            treatment_sha256=str(logical_treatment),
            validator=logical_valid,
        )
        if was_created:
            created.append(logical_copy.parent)
        realtime_copy, realtime_index, was_created = _materialize_result_tree(
            realtime_path,
            release_dir=release_dir,
            model=model,
            treatment_sha256=str(realtime_treatment),
            validator=realtime_valid,
        )
        if was_created:
            created.append(realtime_copy.parent)
        logical_payload = _load_json_object(
            logical_copy, label="materialized logical batch"
        )
        realtime_payload = _load_json_object(
            realtime_copy, label="materialized realtime batch"
        )
        _assert_batch_artifacts_contained(
            logical_copy,
            logical_payload,
            interaction_mode="logical_persistent",
        )
        _assert_batch_artifacts_contained(
            realtime_copy,
            realtime_payload,
            interaction_mode="realtime_persistent",
        )
        logical_binding = _evidence_binding(
            logical_copy,
            repo_root=repo_root,
            payload=logical_payload,
            model=model,
            interaction_mode="logical_persistent",
            treatment_sha256=str(logical_treatment),
            index=logical_index,
        )
        realtime_binding = _evidence_binding(
            realtime_copy,
            repo_root=repo_root,
            payload=realtime_payload,
            model=model,
            interaction_mode="realtime_persistent",
            treatment_sha256=str(realtime_treatment),
            index=realtime_index,
        )
        candidate = deepcopy(manifest)
        candidate.update(
            {
                "status": "formal_evaluation_complete",
                "formal_evaluation_ready": True,
                "public_release_ready": True,
                "leaderboard_eligible": True,
                "public_release_blockers": [],
            }
        )
        eligibility = deepcopy(candidate.get("leaderboard_eligibility") or {})
        eligibility.update({"eligible": True, "suite_exclusions": []})
        candidate["leaderboard_eligibility"] = eligibility
        formal_evidence = deepcopy(candidate["formal_evidence"])
        formal_evidence.update(
            {
                "logical_batch_manifest": logical_binding,
                "realtime_batch_manifest": realtime_binding,
            }
        )
        candidate["formal_evidence"] = formal_evidence
        candidate["formal_evaluation_completion"] = {
            "schema_version": "operate-formal-evaluation-completion-v2",
            "input_release_manifest_sha256": manifest_sha,
            "runtime_identity": expected,
            "model": model,
            "logical_batch_manifest": logical_binding,
            "realtime_batch_manifest": realtime_binding,
        }
        if run_tree != qualification_identity["implementation_tree_sha256"]:
            candidate["formal_evaluation_completion"]["qualification_implementation_tree_sha256"] = (
                qualification_identity["implementation_tree_sha256"]
            )
        if not prepare_distribution_candidate:
            assert distribution_bundle_manifest_path is not None
            assert distribution_receipt_path is not None
            _validate_distribution_receipt(
                candidate,
                bundle_manifest_path=distribution_bundle_manifest_path,
                receipt_path=distribution_receipt_path,
            )
            _validate_candidate_manifest(
                candidate,
                release_dir=release_dir,
                repo_root=repo_root,
            )
        _validate_pending_release(release_dir, manifest, repo_root=repo_root)
        if (
            _runtime_identity(
                manifest,
                release_dir=release_dir,
                manifest_sha256=manifest_sha,
            )
            != qualification_identity
        ):
            raise ReleaseFinalizationError(
                "release runtime identity changed during validation"
            )
        if (
            _file_sha256(release_manifest_path) != manifest_sha
            or _tree_index(logical_path.parent) != logical_index
            or _tree_index(realtime_path.parent) != realtime_index
            or _tree_index(logical_copy.parent, ignore_index=True) != logical_index
            or _tree_index(realtime_copy.parent, ignore_index=True) != realtime_index
        ):
            raise ReleaseFinalizationError("finalization inputs changed during validation")
        _atomic_write_json(output_manifest_path, candidate)
        return candidate
    except Exception:
        for path in reversed(created):
            if path.exists():
                shutil.rmtree(path)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", required=True, type=Path)
    parser.add_argument("--logical-batch-manifest", required=True, type=Path)
    parser.add_argument("--realtime-batch-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--distribution-bundle-manifest", type=Path)
    parser.add_argument("--distribution-receipt", type=Path)
    parser.add_argument("--prepare-distribution-candidate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        finalize_release_manifest(
            release_manifest_path=args.release_manifest,
            logical_batch_manifest_path=args.logical_batch_manifest,
            realtime_batch_manifest_path=args.realtime_batch_manifest,
            output_manifest_path=args.output_manifest,
            distribution_bundle_manifest_path=args.distribution_bundle_manifest,
            distribution_receipt_path=args.distribution_receipt,
            prepare_distribution_candidate=args.prepare_distribution_candidate,
        )
    except ReleaseFinalizationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
