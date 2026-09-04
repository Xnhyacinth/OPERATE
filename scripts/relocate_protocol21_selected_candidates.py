#!/usr/bin/env python3
"""Relocate freshly admitted Protocol-2.1 candidates into tracked paths.

The relocation boundary is intentionally narrow: only rows selected by one
hash-bound candidate replay are copied.  Candidate directories are never
scanned, and the only copied sidecar is an explicitly referenced top-level
``source_lock`` JSON file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.protocol21_evidence import (  # noqa: E402
    portable_repo_path,
    resolve_binding_path,
)
from core.suite_identity import (  # noqa: E402
    canonical_scenario_slug,
    canonical_suite_manifest_sha256,
    verify_scenario_row_against_yaml,
)
from runner.resume import recompute_signature_with_seed  # noqa: E402

LEGACY_RELEASE_ID = "operate_v0_58_0"
_RELEASE_ID_RE = re.compile(r"operate_v\d+_\d+_\d+")


@dataclass(frozen=True)
class PlannedWrite:
    path: Path
    content: bytes


@dataclass(frozen=True)
class RelocationPlan:
    report: dict[str, Any]
    targets: tuple[PlannedWrite, ...]
    writes: tuple[PlannedWrite, ...]
    repo_root: Path
    completion_marker: Path
    generated_artifacts: frozenset[Path]


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label}_missing_or_symlink:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label}_not_object:{path}")
    return payload


def _require_repo_path(path: Path, *, repo_root: Path, label: str) -> Path:
    if ".." in path.parts:
        raise ValueError(f"{label}_path_traversal:{path}")
    lexical = path if path.is_absolute() else repo_root / path
    absolute = Path(os.path.abspath(lexical))
    root = Path(os.path.abspath(repo_root))
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}_outside_repo:{path}") from exc
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"{label}_symlink:{cursor}")
    return absolute


def _resolve_declared_path(
    raw_path: object,
    *,
    repo_root: Path,
    label: str,
) -> Path:
    text = str(raw_path or "")
    if not text or ".." in Path(text).parts:
        raise ValueError(f"{label}_invalid:{text}")
    lexical = Path(text)
    if not lexical.is_absolute():
        lexical = repo_root / lexical
    _require_repo_path(lexical, repo_root=repo_root, label=label)
    resolved = resolve_binding_path(text, repo_root=repo_root)
    return _require_repo_path(resolved, repo_root=repo_root, label=label)


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label}_outside_allowed_root:{path}") from exc


def _gitignored(path: Path, *, repo_root: Path) -> bool:
    relative = path.relative_to(repo_root).as_posix()
    completed = subprocess.run(
        ["git", "check-ignore", "--no-index", "--quiet", "--", relative],
        cwd=repo_root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise ValueError(
            "git_check_ignore_failed:"
            + (completed.stderr.strip() or str(completed.returncode))
        )
    return completed.returncode == 0


def _validate_target(
    path: Path,
    *,
    repo_root: Path,
    label: str,
    allow_hl_generated: bool = False,
) -> None:
    _require_repo_path(path, repo_root=repo_root, label=label)
    allowed_hl_path = allow_hl_generated and path.is_relative_to(repo_root / ".hl")
    if _gitignored(path, repo_root=repo_root) and not allowed_hl_path:
        raise ValueError(f"{label}_gitignored:{path}")
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ValueError(f"{label}_not_regular_file:{path}")


def _check_destination(path: Path, content: bytes) -> bool:
    """Return whether a write is needed; reject a conflicting destination."""
    if not path.exists():
        return True
    if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
        raise ValueError(f"destination_conflict:{path}")
    return False


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _unique_index(
    rows: list[dict[str, Any]], *, label: str
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        identity = _identity(row)
        if not all(identity):
            raise ValueError(f"{label}_identity_incomplete:{identity}")
        if identity in indexed:
            raise ValueError(f"{label}_identity_duplicate:{identity}")
        indexed[identity] = row
    return indexed


def _rows(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    raw = payload.get(key)
    if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
        raise ValueError(f"{key}_rows_invalid")
    return raw


def _validate_bindings(
    *,
    source_suite_path: Path,
    selection_path: Path,
    pipeline_manifest_path: Path,
    source_suite: dict[str, Any],
    selection: dict[str, Any],
    manifest: dict[str, Any],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    del pipeline_manifest_path
    source_sha = _sha256(source_suite_path)
    selection_sha = _sha256(selection_path)
    if manifest.get("status") != "candidate_replay_complete":
        raise ValueError("pipeline_manifest_not_candidate_replay_complete")
    if manifest.get("completed_stage") != "materialize_core":
        raise ValueError("pipeline_manifest_not_materialize_core")
    if manifest.get("source_suite_sha256") != source_sha:
        raise ValueError("pipeline_source_suite_hash_mismatch")
    terminal = manifest.get("terminal_stage_artifact")
    if not isinstance(terminal, dict):
        raise ValueError("pipeline_terminal_stage_binding_missing")
    terminal_path = _resolve_declared_path(
        terminal.get("path"), repo_root=repo_root, label="terminal_stage"
    )
    if terminal_path != selection_path or terminal.get("sha256") != selection_sha:
        raise ValueError("pipeline_selection_binding_mismatch")
    tree_sha = str(manifest.get("implementation_tree_sha256") or "")
    if not tree_sha or selection.get("implementation_tree_sha256") != tree_sha:
        raise ValueError("selection_implementation_tree_mismatch")
    source_binding = (selection.get("input_bindings") or {}).get("source_suite")
    if not isinstance(source_binding, dict):
        raise ValueError("selection_source_suite_binding_missing")
    bound_path = _resolve_declared_path(
        source_binding.get("path"), repo_root=repo_root, label="source_binding"
    )
    if (
        bound_path != source_suite_path
        or source_binding.get("sha256") != source_sha
        or source_binding.get("implementation_tree_sha256") != tree_sha
    ):
        raise ValueError("selection_source_suite_binding_mismatch")

    source_rows = _rows(source_suite, "scenarios")
    source_index = _unique_index(source_rows, label="source_suite")
    selected = _rows(selection, "scenarios")
    rejected = _rows(selection, "rejected")
    secondary = _rows(selection, "secondary")
    if (
        source_suite.get("status") != "working_set"
        or selection.get("schema_version") != "2.1"
        or selection.get("status") != "protocol21_core_candidate"
        or selection.get("selection_policy") != "quality_maximal_v1"
        or selection.get("n_source") != len(source_rows)
        or selection.get("n_selected") != len(selected)
        or selection.get("n_rejected") != len(rejected)
        or selection.get("n_secondary") != len(secondary)
    ):
        raise ValueError("selection_accounting_mismatch")
    disposition_index = _unique_index(
        selected + rejected + secondary, label="selection_disposition"
    )
    if set(disposition_index) != set(source_index):
        raise ValueError("selection_disposition_identity_mismatch")
    selected_index = _unique_index(selected, label="selection_selected")
    for identity, row in selected_index.items():
        source_row = source_index.get(identity)
        if source_row is None or row.get("path") != source_row.get("path"):
            raise ValueError(f"selected_source_identity_mismatch:{identity}")
    return selected, source_index


def _yaml_bytes(body: dict[str, Any]) -> bytes:
    return yaml.safe_dump(
        body,
        sort_keys=False,
        allow_unicode=True,
    ).encode("utf-8")


def _release_bindings(
    *,
    release_id: str,
    release_root: Path | None,
    runtime_source_lock_path: Path | None,
    repo_root: Path,
) -> tuple[Path, Path]:
    if _RELEASE_ID_RE.fullmatch(release_id) is None:
        raise ValueError("release_id_invalid")
    if release_id != LEGACY_RELEASE_ID and (
        release_root is None or runtime_source_lock_path is None
    ):
        raise ValueError("new_release_bindings_required")

    expected_release_root = repo_root / "release" / release_id
    resolved_release_root = _require_repo_path(
        release_root or expected_release_root,
        repo_root=repo_root,
        label="release_root",
    )
    if resolved_release_root != expected_release_root:
        raise ValueError("release_root_mismatch")
    if release_id != LEGACY_RELEASE_ID and resolved_release_root.exists():
        raise ValueError(f"release_root_already_exists:{resolved_release_root}")

    canonical_lock_base = repo_root / "sources" / "locks" / release_id
    expected_runtime_lock = canonical_lock_base / "backend_runtime_sources.json"
    resolved_runtime_lock = _require_repo_path(
        runtime_source_lock_path or expected_runtime_lock,
        repo_root=repo_root,
        label="runtime_source_lock",
    )
    if resolved_runtime_lock != expected_runtime_lock:
        raise ValueError("runtime_source_lock_path_mismatch")
    runtime_lock = _load_object(
        resolved_runtime_lock,
        label="runtime_source_lock",
    )
    if (
        runtime_lock.get("schema_version")
        != "operate-backend-runtime-source-lock-v1"
        or runtime_lock.get("release_id") != release_id
    ):
        raise ValueError("runtime_source_lock_contract_invalid")
    return resolved_release_root, resolved_runtime_lock


def build_relocation_plan(
    *,
    selection_path: Path,
    source_suite_path: Path,
    pipeline_manifest_path: Path,
    output_source_suite_path: Path,
    output_selection_path: Path,
    identity_ledger_path: Path,
    repo_root: Path = REPO_ROOT,
    release_id: str = LEGACY_RELEASE_ID,
    release_root: Path | None = None,
    runtime_source_lock_path: Path | None = None,
    scenario_root: Path | None = None,
    lock_root: Path | None = None,
) -> RelocationPlan:
    """Validate all inputs and return a no-side-effect relocation plan."""
    repo_root = repo_root.absolute()
    selection_path = _require_repo_path(
        selection_path, repo_root=repo_root, label="selection"
    )
    source_suite_path = _require_repo_path(
        source_suite_path, repo_root=repo_root, label="source_suite"
    )
    pipeline_manifest_path = _require_repo_path(
        pipeline_manifest_path, repo_root=repo_root, label="pipeline_manifest"
    )
    output_source_suite_path = _require_repo_path(
        output_source_suite_path, repo_root=repo_root, label="output_source_suite"
    )
    output_selection_path = _require_repo_path(
        output_selection_path, repo_root=repo_root, label="output_selection"
    )
    identity_ledger_path = _require_repo_path(
        identity_ledger_path, repo_root=repo_root, label="identity_ledger"
    )
    release_root, runtime_source_lock_path = _release_bindings(
        release_id=release_id,
        release_root=release_root,
        runtime_source_lock_path=runtime_source_lock_path,
        repo_root=repo_root,
    )
    protected_inputs = {
        selection_path,
        source_suite_path,
        pipeline_manifest_path,
        runtime_source_lock_path,
    }
    outputs = {
        output_source_suite_path,
        output_selection_path,
        identity_ledger_path,
    }
    if outputs & protected_inputs:
        raise ValueError("relocation_output_overlaps_input")
    if len(outputs) != 3:
        raise ValueError("relocation_output_collision")
    canonical_scenario_base = repo_root / "scenarios" / release_id
    canonical_lock_base = repo_root / "sources" / "locks" / release_id
    scenario_root = _require_repo_path(
        scenario_root or canonical_scenario_base,
        repo_root=repo_root,
        label="scenario_root",
    )
    lock_root = _require_repo_path(
        lock_root or canonical_lock_base,
        repo_root=repo_root,
        label="lock_root",
    )
    _require_within(scenario_root, canonical_scenario_base, label="scenario_root")
    _require_within(lock_root, canonical_lock_base, label="lock_root")

    source_suite = _load_object(source_suite_path, label="source_suite")
    selection = _load_object(selection_path, label="selection")
    manifest = _load_object(pipeline_manifest_path, label="pipeline_manifest")
    selected, source_index = _validate_bindings(
        source_suite_path=source_suite_path,
        selection_path=selection_path,
        pipeline_manifest_path=pipeline_manifest_path,
        source_suite=source_suite,
        selection=selection,
        manifest=manifest,
        repo_root=repo_root,
    )

    source_artifact_root = source_suite_path.parent
    planned_contents: dict[Path, bytes] = {}
    destination_casefold: dict[str, Path] = {}
    relocated_rows: list[dict[str, Any]] = []
    relocated_bodies: dict[str, dict[str, Any]] = {}
    identities: list[dict[str, Any]] = []
    lock_identities: dict[Path, dict[str, str]] = {}

    for selected_row in selected:
        identity = _identity(selected_row)
        source_row = source_index[identity]
        source_yaml = _resolve_declared_path(
            source_row.get("path"), repo_root=repo_root, label="scenario_source"
        )
        _require_within(
            source_yaml, source_artifact_root, label="scenario_source"
        )
        if source_yaml.suffix != ".yaml" or not source_yaml.is_file():
            raise ValueError(f"scenario_source_not_yaml:{source_yaml}")
        errors = verify_scenario_row_against_yaml(source_row, path=source_yaml)
        if errors:
            raise ValueError(
                f"scenario_source_identity_mismatch:{identity}:{','.join(errors)}"
            )
        body = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
        if not isinstance(body, dict):
            raise ValueError(f"scenario_source_not_object:{source_yaml}")
        if (
            selected_row.get("status") != "core_locked"
            or selected_row.get("core_disposition") != "core_locked"
        ):
            raise ValueError(f"selected_row_not_core_locked:{identity}")
        domain = str(source_row.get("domain") or "")
        if not domain or Path(domain).name != domain:
            raise ValueError(f"scenario_domain_invalid:{domain}")
        destination = scenario_root / domain / "candidate_imports" / source_yaml.name
        destination = _require_repo_path(
            destination, repo_root=repo_root, label="scenario_destination"
        )
        folded = destination.relative_to(repo_root).as_posix().casefold()
        prior = destination_casefold.get(folded)
        if prior is not None and prior != destination:
            raise ValueError(f"destination_casefold_conflict:{prior}:{destination}")
        destination_casefold[folded] = destination
        if destination in planned_contents:
            raise ValueError(f"destination_duplicate:{destination}")

        copied_locks: list[dict[str, str]] = []
        source_lock = body.get("source_lock")
        if isinstance(source_lock, str):
            lock_source = _resolve_declared_path(
                source_lock, repo_root=repo_root, label="source_lock"
            )
            if not lock_source.is_file() or lock_source.suffix != ".json":
                raise ValueError(f"source_lock_not_json:{lock_source}")
            _load_object(lock_source, label="source_lock")
            lock_destination = lock_root / domain / lock_source.name
            lock_destination = _require_repo_path(
                lock_destination,
                repo_root=repo_root,
                label="source_lock_destination",
            )
            lock_content = lock_source.read_bytes()
            prior_content = planned_contents.get(lock_destination)
            if prior_content is not None and prior_content != lock_content:
                raise ValueError(f"destination_conflict:{lock_destination}")
            planned_contents[lock_destination] = lock_content
            body["source_lock"] = portable_repo_path(
                lock_destination, repo_root=repo_root
            )
            lock_record = {
                "old_path": portable_repo_path(lock_source, repo_root=repo_root),
                "new_path": portable_repo_path(
                    lock_destination, repo_root=repo_root
                ),
                "sha256": _sha256_bytes(lock_content),
            }
            lock_identities.setdefault(lock_destination, lock_record)
            copied_locks.append(lock_record)
        elif source_lock is not None and not isinstance(source_lock, dict):
            raise ValueError(f"source_lock_invalid:{source_yaml}")

        seed = body.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"scenario_seed_invalid:{source_yaml}")
        old_yaml_sha = _sha256(source_yaml)
        body.pop("scenario_signature", None)
        body["scenario_signature"] = recompute_signature_with_seed(body, seed)
        content = _yaml_bytes(body)
        planned_contents[destination] = content
        new_row = deepcopy(source_row)
        new_row["path"] = portable_repo_path(destination, repo_root=repo_root)
        new_row["scenario_signature"] = body["scenario_signature"]
        relocated_rows.append(new_row)
        slug = canonical_scenario_slug(new_row["path"])
        if slug in relocated_bodies:
            raise ValueError(f"canonical_scenario_slug_duplicate:{slug}")
        relocated_bodies[slug] = body
        identities.append(
            {
                "scenario_id": str(new_row["scenario_id"]),
                "old": {
                    "scenario_signature": identity[1],
                    "path": portable_repo_path(source_yaml, repo_root=repo_root),
                    "yaml_sha256": old_yaml_sha,
                },
                "new": {
                    "scenario_signature": body["scenario_signature"],
                    "path": new_row["path"],
                    "yaml_sha256": _sha256_bytes(content),
                },
                "locks": copied_locks,
            }
        )

    output_suite = deepcopy(source_suite)
    output_suite["scenarios"] = relocated_rows
    for count_key in ("n_scenarios", "n_expected", "n_completed"):
        if count_key in output_suite:
            output_suite[count_key] = len(relocated_rows)
    suite_content = _json_bytes(output_suite)
    new_suite_sha = _sha256_bytes(suite_content)
    ordered_slugs = [canonical_scenario_slug(row["path"]) for row in relocated_rows]
    suite_manifest_sha = canonical_suite_manifest_sha256(
        ordered_slugs, relocated_bodies
    )
    if output_source_suite_path in planned_contents:
        raise ValueError(f"destination_conflict:{output_source_suite_path}")
    planned_contents[output_source_suite_path] = suite_content
    original_selection_sha = _sha256(selection_path)
    remapped_selection = {
        "schema_version": "2.1",
        "status": "protocol21_core_candidate",
        "selection_policy": selection.get("selection_policy"),
        "selection_kind": "relocated_core_allowlist_v1",
        "implementation_tree_sha256": manifest["implementation_tree_sha256"],
        "n_selected": len(relocated_rows),
        "input_bindings": {
            "source_suite": {
                "path": portable_repo_path(
                    output_source_suite_path, repo_root=repo_root
                ),
                "sha256": new_suite_sha,
                "implementation_tree_sha256": manifest[
                    "implementation_tree_sha256"
                ],
            },
            "original_selection": {
                "path": portable_repo_path(selection_path, repo_root=repo_root),
                "sha256": original_selection_sha,
            },
        },
        "scenarios": [
            {
                "scenario_id": row["scenario_id"],
                "scenario_signature": row["scenario_signature"],
                "path": row["path"],
                "status": "core_locked",
                "core_disposition": "core_locked",
            }
            for row in relocated_rows
        ],
    }
    selection_content = _json_bytes(remapped_selection)
    if output_selection_path in planned_contents:
        raise ValueError(f"destination_conflict:{output_selection_path}")
    planned_contents[output_selection_path] = selection_content
    ledger = {
        "schema_version": "operate-canonical-relocation-v1",
        "status": "canonical_relocation_complete",
        "bindings": {
            "release": {
                "release_id": release_id,
                "root": portable_repo_path(release_root, repo_root=repo_root),
            },
            "runtime_source_lock": {
                "path": portable_repo_path(
                    runtime_source_lock_path, repo_root=repo_root
                ),
                "sha256": _sha256(runtime_source_lock_path),
            },
            "pipeline_manifest": {
                "path": portable_repo_path(
                    pipeline_manifest_path, repo_root=repo_root
                ),
                "sha256": _sha256(pipeline_manifest_path),
            },
            "selection": {
                "path": portable_repo_path(selection_path, repo_root=repo_root),
                "sha256": original_selection_sha,
            },
            "remapped_selection": {
                "path": portable_repo_path(
                    output_selection_path, repo_root=repo_root
                ),
                "sha256": _sha256_bytes(selection_content),
            },
            "old_source_suite": {
                "path": portable_repo_path(
                    source_suite_path, repo_root=repo_root
                ),
                "sha256": _sha256(source_suite_path),
            },
            "new_source_suite": {
                "path": portable_repo_path(
                    output_source_suite_path, repo_root=repo_root
                ),
                "sha256": new_suite_sha,
                "suite_manifest_sha256": suite_manifest_sha,
            },
        },
        "implementation_tree_sha256": manifest["implementation_tree_sha256"],
        "core_release_pipeline_sha256": manifest[
            "core_release_pipeline_sha256"
        ],
        "n_selected": len(relocated_rows),
        "n_locks": len(lock_identities),
        "identities": identities,
    }
    ledger_content = _json_bytes(ledger)
    if identity_ledger_path in planned_contents:
        raise ValueError(f"destination_conflict:{identity_ledger_path}")
    planned_contents[identity_ledger_path] = ledger_content

    ordered_contents = sorted(
        planned_contents.items(),
        key=lambda item: (
            item[0] == identity_ledger_path,
            item[0].as_posix(),
        ),
    )
    targets: list[PlannedWrite] = []
    writes: list[PlannedWrite] = []
    for path, content in ordered_contents:
        _validate_target(
            path,
            repo_root=repo_root,
            label="relocation_target",
            allow_hl_generated=path in outputs,
        )
        target = PlannedWrite(path=path, content=content)
        targets.append(target)
        if _check_destination(path, content):
            writes.append(target)
    if identity_ledger_path not in {write.path for write in writes} and writes:
        raise ValueError("completion_marker_precedes_incomplete_outputs")
    report = {
        "schema_version": "operate-canonical-relocation-plan-v1",
        "status": "planned",
        "release_id": release_id,
        "release_root": portable_repo_path(release_root, repo_root=repo_root),
        "runtime_source_lock_sha256": _sha256(runtime_source_lock_path),
        "n_selected": len(relocated_rows),
        "n_locks": len(lock_identities),
        "n_writes": len(writes),
        "old_source_suite_sha256": _sha256(source_suite_path),
        "new_source_suite_sha256": new_suite_sha,
        "new_suite_manifest_sha256": suite_manifest_sha,
        "remapped_selection_sha256": _sha256_bytes(selection_content),
        "identity_ledger_sha256": _sha256_bytes(ledger_content),
        "identity_ledger": ledger,
    }
    return RelocationPlan(
        report=report,
        targets=tuple(targets),
        writes=tuple(writes),
        repo_root=repo_root,
        completion_marker=identity_ledger_path,
        generated_artifacts=frozenset(outputs),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def apply_relocation_plan(plan: RelocationPlan) -> dict[str, Any]:
    """Stage then exclusively create targets, committing the ledger last."""
    staged: list[tuple[Path, Path]] = []
    committed: list[tuple[Path, int, int]] = []
    try:
        write_paths = {write.path for write in plan.writes}
        if (
            plan.writes
            and plan.completion_marker not in write_paths
        ):
            raise ValueError("completion_marker_missing_from_nonempty_plan")
        for target in plan.targets:
            _validate_target(
                target.path,
                repo_root=plan.repo_root,
                label="relocation_target",
                allow_hl_generated=target.path in plan.generated_artifacts,
            )
            if (
                target.path not in write_paths
                and _check_destination(target.path, target.content)
            ):
                raise ValueError(f"destination_missing_after_plan:{target.path}")
        ordered_writes = sorted(
            plan.writes,
            key=lambda write: (
                write.path == plan.completion_marker,
                write.path.as_posix(),
            ),
        )
        for write in ordered_writes:
            _validate_target(
                write.path,
                repo_root=plan.repo_root,
                label="relocation_target",
                allow_hl_generated=write.path in plan.generated_artifacts,
            )
            if not _check_destination(write.path, write.content):
                continue
            write.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{write.path.name}.", dir=write.path.parent
            )
            os.fchmod(descriptor, 0o644)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(write.content)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((Path(temporary_name), write.path))
        for temporary_path, destination in staged:
            if destination == plan.completion_marker:
                for target in plan.targets:
                    if target.path == plan.completion_marker:
                        continue
                    if _check_destination(target.path, target.content):
                        raise ValueError(
                            f"relocation_target_missing_before_completion:{target.path}"
                        )
            try:
                os.link(temporary_path, destination, follow_symlinks=False)
            except FileExistsError as exc:
                raise ValueError(f"destination_conflict:{destination}") from exc
            stat = os.lstat(destination)
            committed.append((destination, stat.st_dev, stat.st_ino))
            temporary_path.unlink()
            _fsync_directory(destination.parent)
    except BaseException:
        for temporary_path, _destination in staged:
            temporary_path.unlink(missing_ok=True)
        for destination, device, inode in reversed(committed):
            try:
                stat = os.lstat(destination)
            except FileNotFoundError:
                continue
            if (stat.st_dev, stat.st_ino) != (device, inode):
                continue
            destination.unlink()
            try:
                _fsync_directory(destination.parent)
            except OSError:
                pass
        raise
    return {**plan.report, "status": "complete"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--pipeline-manifest", type=Path, required=True)
    parser.add_argument("--output-source-suite", type=Path, required=True)
    parser.add_argument("--output-selection", type=Path, required=True)
    parser.add_argument("--identity-ledger", type=Path, required=True)
    parser.add_argument("--release-id", default=LEGACY_RELEASE_ID)
    parser.add_argument("--release-root", type=Path)
    parser.add_argument("--runtime-source-lock", type=Path)
    parser.add_argument("--scenario-root", type=Path)
    parser.add_argument("--lock-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_relocation_plan(
            selection_path=args.selection,
            source_suite_path=args.source_suite,
            pipeline_manifest_path=args.pipeline_manifest,
            output_source_suite_path=args.output_source_suite,
            output_selection_path=args.output_selection,
            identity_ledger_path=args.identity_ledger,
            release_id=args.release_id,
            release_root=args.release_root,
            runtime_source_lock_path=args.runtime_source_lock,
            scenario_root=args.scenario_root,
            lock_root=args.lock_root,
        )
        report = apply_relocation_plan(plan) if args.execute else plan.report
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if not args.execute:
        print("NO_FILES_WRITTEN=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
