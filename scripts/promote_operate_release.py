#!/usr/bin/env python3
"""Promote a green Protocol-2.1 replay into an OPERATE release manifest.

The promoter is deliberately fail-closed: it only materializes metadata after
the source suite, all replay stages, the live implementation tree, and every
selected scenario YAML agree.  It never edits scenario YAMLs.
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
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.source_asset_contract import (  # noqa: E402
    canonical_physical_source_asset_key,
)
from core.suite_identity import verify_scenario_row_against_yaml  # noqa: E402
from evaluation.dimension_applicability import (  # noqa: E402
    dimension_applicability_contract_issue,
)
from evaluation.leaderboard import (  # noqa: E402
    PRIMARY_LEADERBOARD_FORMULA_VERSION,
)
from evaluation.scorer import SCORING_VERSION  # noqa: E402
from scripts.build_protocol21_public_evidence_bundle import (  # noqa: E402
    build_public_evidence_bundle,
)
from scripts.build_protocol21_incremental_union import (  # noqa: E402
    validate_candidate_import_partition,
)
from scripts.build_protocol21_core_readiness import (  # noqa: E402
    FORMAL_WAKEUP_POLICY,
)
from scripts.finalize_operate_candidate_pool import (  # noqa: E402
    validate_compact_candidate_closure,
)

RELEASE_ID = "operate_v0_61_0"
FORMAL_RUNTIME_BUNDLE_NAME = "formal_runtime_bundle.json"
FORMAL_RUNTIME_BUNDLE_SCHEMA = "operate-formal-runtime-bundle-v1"
PRIMARY_INFERENCE_VERSION = "physical_cluster_hierarchical_bootstrap_randomization_v1"
EXPECTED_STAGES = (
    "preflight",
    "behavioral",
    "source_consumption",
    "task_contracts",
    "complexity",
    "observed_reference_depth",
    "strategy_depth",
    "source_grounded",
    "agentic_contract",
    "materialize_core",
    "release_coverage",
    "readiness",
)
STAGE_FILES = {
    "preflight": "protocol2_v21_working_set_preflight.json",
    "behavioral": "behavioral_calibration_protocol2_v21.json",
    "source_consumption": "source_consumption_protocol2_v21.json",
    "task_contracts": "task_contracts_protocol2_v21.json",
    "complexity": "complexity_protocol2_v21.json",
    "observed_reference_depth": "observed_reference_depth_protocol2_v21.json",
    "strategy_depth": "strategy_depth_protocol2_v21.json",
    "source_grounded": "source_grounded_protocol2_v21.json",
    "agentic_contract": "agentic_core_contract_protocol2_v21.json",
    "materialize_core": "refined_core_selection_protocol2_v21.json",
    "release_coverage": "release_coverage_protocol2_v21.json",
    "readiness": "protocol2_v21_core_readiness.json",
}
PIPELINE_HASH_FIELDS = {
    "preflight": "preflight_sha256",
    "behavioral": "behavioral_sha256",
    "source_consumption": "source_consumption_sha256",
    "task_contracts": "task_contracts_sha256",
    "complexity": "complexity_sha256",
    "observed_reference_depth": "observed_reference_depth_sha256",
    "strategy_depth": "strategy_depth_sha256",
    "source_grounded": "source_grounded_sha256",
    "agentic_contract": "agentic_contract_sha256",
    "materialize_core": "core_selection_sha256",
    "release_coverage": "release_coverage_sha256",
    "readiness": "readiness_sha256",
}


def _validate_formal_run_contract_for_release(
    contract: object,
    *,
    release_id: str,
) -> None:
    if not isinstance(contract, dict) or not contract:
        raise ValueError("formal_run_contract_missing")
    if contract.get("contract_version") != "agentic_persistent.v1":
        raise ValueError("formal_run_contract_version_unsupported")
    realtime = contract.get("realtime_formal_contract")
    if not isinstance(realtime, dict):
        raise ValueError("formal_realtime_contract_missing_or_invalid")
    release_version = tuple(
        int(part) for part in release_id.removeprefix("operate_v").split("_")
    )
    if release_version < (0, 61, 0):
        if realtime.get("contract_version") != "realtime_persistent.v1":
            raise ValueError("formal_realtime_contract_missing_or_invalid")
        return
    if (
        contract.get("wakeup_policy") != FORMAL_WAKEUP_POLICY
        or realtime.get("contract_version") != "realtime_persistent.v2"
        or realtime.get("realtime_coordinator") != "realtime_episode_v5"
        or realtime.get("wakeup_policy") != FORMAL_WAKEUP_POLICY
    ):
        raise ValueError("formal_wakeup_contract_missing_or_invalid")


def _public_release_blockers(release_version: str) -> list[str]:
    blockers = [
        "formal_logical_persistent_evaluation_pending",
        "formal_realtime_persistent_evaluation_pending",
    ]
    version = tuple(int(part) for part in release_version.split("."))
    if version >= (0, 58, 0):
        blockers.append("formal_runtime_evidence_distribution_pending")
    return blockers


UNRESOLVED_SOURCE_INVENTORIES = frozenset(
    {
        "held_candidates",
        "pending_candidates",
        "redesign_candidates",
    }
)
TERMINAL_SOURCE_INVENTORY = "abandoned_candidates"
MATERIALIZE_SECONDARY_DISPOSITIONS = frozenset({"secondary_duplicate"})
MATERIALIZE_REJECTED_DISPOSITIONS = frozenset({"retired_intrinsic"})
SAFE_SOURCE_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
BACKEND_RUNTIME_CLOSURE_SCHEMA = "operate-backend-runtime-closure-v1"
BACKEND_RUNTIME_CLOSURE_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "status",
        "terminal",
        "portable",
        "source_suite_sha256",
        "archived_files",
        "repo_tracked_files",
        "separately_bundled_files",
        "external_sources",
        "backend_links",
        "runtime_packages",
        "summary",
        "identity_sha256",
    }
)
BACKEND_RUNTIME_SUMMARY_FIELDS = frozenset(
    {
        "n_archived_files",
        "n_backend_links",
        "n_external_sources",
        "n_repo_tracked_files",
        "n_runtime_packages",
        "n_separately_bundled_files",
        "n_source_assets",
        "n_unresolved",
        "n_virtual_sources",
    }
)
EXTERNAL_SOURCE_FIELDS = frozenset(
    {
        "delivery",
        "url",
        "revision",
        "required_files",
        "metadata",
    }
)
EXTERNAL_SOURCE_METADATA_FIELDS = frozenset(
    {"backend_kinds", "license_status", "redistributed", "roles", "root"}
)
RUNTIME_PACKAGE_FIELDS = frozenset(
    {"backend_kinds", "lock_entries", "lock_entries_sha256", "uv_lock_sha256"}
)
RUNTIME_LOCK_ENTRY_FIELDS = frozenset(
    {"version", "source", "artifacts", "identity_sha256"}
)
RUNTIME_ARTIFACT_FIELDS = frozenset({"kind", "url", "hash", "size", "upload-time"})
RUNTIME_FILE_IDENTITY_FIELDS = frozenset({"sha256", "roles", "backend_kinds"})
PARENT_RETAINED_SCENARIO_FIELDS = (
    "scenario_id",
    "scenario_signature",
    "seed",
    "horizon_ticks",
    "domain",
    "family",
    "backend_kind",
    "difficulty_mode",
    "difficulty_level",
    "source_denominator_key",
    "structural_fingerprint",
    "semantic_fingerprint",
)

DEFAULT_PARENT = REPO_ROOT / "release/operate_v0_60_0/manifest.json"
DEFAULT_SOURCE = REPO_ROOT / ".hl/release_rebuild/operate_v0_61_0/protocol21_union_traffic.json"
DEFAULT_PIPELINE = REPO_ROOT / ".hl/release_rebuild/operate_v0_61_0/full_replay_release"
DEFAULT_OUTPUT = REPO_ROOT / "release/operate_v0_61_0"
OPENB_V2023_DATASET_DECLARATION = {
    "url": (
        "https://github.com/alibaba/clusterdata/tree/"
        "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71/cluster-trace-gpu-v2023"
    ),
    "license": "research trace terms; upstream repository license applies",
    "lock_strategy": "upstream_git_commit_raw_sha256_and_explicit_row_graph",
    "redistribution": (
        "not redistributed; fetched from the pinned upstream commit and "
        "verified by per-file SHA-256"
    ),
    "delivery": "upstream_fetch",
    "commit": "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71",
    "required_file_sha256s": {
        "works/clusterdata/cluster-trace-gpu-v2023/csv/openb_node_list_gpu_node.csv": (
            "2beca64b4d3dfa342036a34b56a495c6cef9225db836c81f541282cb1df320b5"
        ),
        "works/clusterdata/cluster-trace-gpu-v2023/csv/openb_pod_list_gpuspec33.csv": (
            "eca4f746db1e5b25864ad021b55ece3943e101a3ebd4574d09dcb95c46117652"
        ),
    },
}
NGSIM_US101_DATASET_DECLARATION = {
    "url": (
        "https://data.transportation.gov/Automobiles/"
        "Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj"
    ),
    "license": (
        "CC-BY-SA-3.0 dataset API metadata; CC-BY-SA-4.0 Common Core metadata "
        "(operator-reviewed)"
    ),
    "lock_strategy": (
        "doi+canonical_query_or_archive+raw_sha256+row_semantic_sha256"
    ),
    "redistribution": (
        "source-grounded US-101 runtime bundles with upstream attribution and "
        "operator-reviewed license evidence"
    ),
    "delivery": "bundle",
    "dataset_id": "8ect-6jqj",
    "source_release": "doi:10.21949/1504477",
    "recording_id": "us-101",
}
PGLIB_UC_REVISION = "39a7f38cf4703de92f0291f0c873c2e98c789301"
PGLIB_UC_URL = "https://github.com/power-grid-lib/pglib-uc.git"
PGLIB_UC_ROOT = "works/pglib-uc"
PGLIB_UC_LICENSE_PATH = f"{PGLIB_UC_ROOT}/LICENSE"
PGLIB_UC_LICENSE_SHA256 = (
    "b5ececfa64eb67fd5b0e5c135624f0f9004b938d399150e644f9641b659e628c"
)
PGLIB_UC_LICENSE_STATUS = "verified_cc_by_4_data_and_mit_software"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_uv_lock_sha256(repo_root: Path) -> str:
    uv_lock = repo_root / "uv.lock"
    if uv_lock.is_symlink() or not uv_lock.is_file():
        raise ValueError("backend_runtime_closure_uv_lock_missing")
    return _sha256(uv_lock)


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label}_missing:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label}_not_object:{path}")
    return payload


def _relative(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path_outside_repo:{resolved}") from exc


def _binding_matches_file(
    *,
    repo_root: Path,
    binding: object,
    expected_path: Path,
    live_tree: str,
) -> bool:
    if not isinstance(binding, Mapping):
        return False
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return False
    path = Path(raw_path)
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        _relative(resolved, repo_root)
    except ValueError:
        return False
    return bool(
        resolved == expected_path.resolve()
        and resolved.is_file()
        and binding.get("sha256") == _sha256(resolved)
        and binding.get("implementation_tree_sha256") == live_tree
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _ordered_identity_sha256(rows: list[Mapping[str, Any]]) -> str:
    identities = [
        {
            "scenario_id": str(row.get("scenario_id") or ""),
            "scenario_signature": str(row.get("scenario_signature") or ""),
            "path": str(row.get("path") or ""),
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(
            identities,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, _json_text(payload))


def _directory_snapshot(path: Path) -> dict[str, str] | None:
    """Return a content snapshot used to reject concurrent release changes."""
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_dir():
        raise ValueError("release_output_not_directory")
    snapshot: dict[str, str] = {}
    for member in sorted(path.rglob("*")):
        if member.is_symlink():
            raise ValueError(f"release_output_symlink:{member}")
        if member.is_file():
            snapshot[member.relative_to(path).as_posix()] = _sha256(member)
    return snapshot


@contextmanager
def _exclusive_promotion_lock(repo_root: Path) -> Iterator[None]:
    """Serialize release swaps with a persistent repository-local file lock."""
    if os.name != "posix":
        raise RuntimeError("release_promotion_lock_unsupported_platform")
    import fcntl  # noqa: PLC0415

    lock_dir = repo_root / ".hl/locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    if lock_dir.is_symlink() or not lock_dir.is_dir():
        raise ValueError("release_promotion_lock_directory_invalid")
    _relative(lock_dir, repo_root)
    lock_path = lock_dir / "release-promotion.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("release_promotion_lock_not_regular_file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _verify_staged_release(
    release: Path,
    *,
    repo_root: Path,
    require_public_evidence: bool,
) -> None:
    """Verify staged bytes without consulting or mutating the live release."""
    manifest_path = release / "manifest.json"
    manifest = _load_object(manifest_path, label="staged_manifest")
    closure_binding = manifest.get("candidate_closure")
    if not isinstance(closure_binding, dict):
        raise ValueError("staged_release_integrity_failed:candidate_closure_binding")
    closure_path = release / str(closure_binding.get("path") or "")
    if closure_path.parent != release or closure_path.name != "candidate_closure.json":
        raise ValueError("staged_release_integrity_failed:candidate_closure_path")
    closure = _load_object(closure_path, label="staged_candidate_closure")
    validate_compact_candidate_closure(closure)
    closure_summary = closure["summary"]
    expected_closure_binding = {
        "path": closure_path.name,
        "sha256": _sha256(closure_path),
        "schema_version": closure["schema_version"],
        "status": closure["status"],
        "n_independent_candidates": closure_summary["n_independent_candidates"],
        "n_terminal_candidates": closure_summary["n_terminal_candidates"],
        "n_unresolved_candidates": closure_summary["n_unresolved_candidates"],
        "identity_set_sha256": closure["identity_set_sha256"],
    }
    if closure_binding != expected_closure_binding:
        raise ValueError("staged_release_integrity_failed:candidate_closure_binding")
    runtime_binding = manifest.get("backend_runtime_closure")
    if not isinstance(runtime_binding, dict):
        raise ValueError(
            "staged_release_integrity_failed:backend_runtime_closure_binding"
        )
    runtime_path = release / str(runtime_binding.get("path") or "")
    if (
        runtime_path.parent != release
        or runtime_path.name != "backend_runtime_closure.json"
    ):
        raise ValueError("staged_release_integrity_failed:backend_runtime_closure_path")
    runtime_closure = _load_object(
        runtime_path,
        label="staged_backend_runtime_closure",
    )
    source_path = release / "protocol21_source_suite.json"
    _validate_backend_runtime_closure(
        runtime_closure,
        release_id=str(manifest.get("release_id") or ""),
        source_suite_sha256=_sha256(source_path),
        expected_uv_lock_sha256=_repo_uv_lock_sha256(repo_root),
    )
    runtime_summary = runtime_closure["summary"]
    expected_runtime_binding = {
        "path": runtime_path.name,
        "sha256": _sha256(runtime_path),
        "schema_version": runtime_closure["schema_version"],
        "n_archived_files": runtime_summary["n_archived_files"],
        "n_external_sources": runtime_summary["n_external_sources"],
        "n_backend_links": runtime_summary["n_backend_links"],
        "n_runtime_packages": runtime_summary["n_runtime_packages"],
        "identity_sha256": runtime_closure["identity_sha256"],
    }
    if runtime_binding != expected_runtime_binding:
        raise ValueError(
            "staged_release_integrity_failed:backend_runtime_closure_binding"
        )
    if require_public_evidence:
        from scripts.verify_release_integrity import (  # noqa: PLC0415
            build_protocol21_core_integrity_report,
        )

        report = build_protocol21_core_integrity_report(
            release,
            portable=True,
            artifact_root=repo_root,
        )
        if report.get("ok") is not True:
            codes = ",".join(
                str(issue.get("code") or "unknown")
                for issue in report.get("issues") or []
                if isinstance(issue, dict)
            )
            raise ValueError(f"staged_release_integrity_failed:{codes}")
        return

    core_path = release / "core_suite.json"
    core = _load_object(core_path, label="staged_core")
    source = _load_object(source_path, label="staged_source_suite")
    core_rows = core.get("scenarios")
    source_rows = source.get("scenarios")
    descriptors = manifest.get("backend_descriptors")
    if not (
        release.name == manifest.get("release_id") == core.get("release_id")
        and isinstance(core_rows, list)
        and isinstance(source_rows, list)
        and core.get("n_scenarios") == len(core_rows)
        and source.get("n_scenarios") == len(source_rows)
        and (manifest.get("core_suite") or {}).get("sha256") == _sha256(core_path)
        and (manifest.get("protocol21_replay") or {}).get("source_suite_sha256")
        == _sha256(source_path)
        and isinstance(descriptors, dict)
        and set(core.get("by_backend") or {}).issubset(descriptors)
    ):
        raise ValueError("staged_release_integrity_failed:metadata_closure")


def _transactional_publish_release(
    *,
    output_dir: Path,
    repo_root: Path,
    files: Mapping[str, str],
    expected_snapshot: dict[str, str] | None,
    require_public_evidence: bool,
    locked_input_validator: Callable[[], None],
) -> None:
    """Stage, verify, swap, and rollback a release on one filesystem."""
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.promotion.",
            dir=output_dir.parent,
        )
    )
    staged_release = staging_root / output_dir.name
    staged_release.mkdir()
    previous_release = staging_root / f"{output_dir.name}.previous"
    failed_release = staging_root / f"{output_dir.name}.failed"
    swapped = False
    try:
        for relative, content in sorted(files.items()):
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
                raise ValueError(f"release_output_path_invalid:{relative}")
            _atomic_write(staged_release / path, content)
        _verify_staged_release(
            staged_release,
            repo_root=repo_root,
            require_public_evidence=require_public_evidence,
        )
        with _exclusive_promotion_lock(repo_root):
            locked_input_validator()
            locked_snapshot = _directory_snapshot(output_dir)
            if locked_snapshot != expected_snapshot:
                raise ValueError("release_output_changed_during_promotion")
            if output_dir.exists():
                os.replace(output_dir, previous_release)
            try:
                os.replace(staged_release, output_dir)
                swapped = True
            except BaseException:
                if previous_release.exists() and not output_dir.exists():
                    os.replace(previous_release, output_dir)
                raise

            from scripts.verify_release_integrity import (  # noqa: PLC0415
                build_protocol21_core_integrity_report,
            )

            report = build_protocol21_core_integrity_report(
                output_dir,
                artifact_root=repo_root,
            )
            if report.get("ok") is not True:
                codes = ",".join(
                    str(issue.get("code") or "unknown")
                    for issue in report.get("issues") or []
                    if isinstance(issue, dict)
                )
                raise ValueError(f"published_release_integrity_failed:{codes}")
            if previous_release.exists():
                if _directory_snapshot(previous_release) != locked_snapshot:
                    raise ValueError("release_backup_changed_during_promotion")
                try:
                    shutil.rmtree(previous_release)
                except OSError:
                    # The new release is already fully verified.  Keep an
                    # undeleted backup remainder as evidence instead of rolling
                    # back to a directory that cleanup may have partially removed.
                    pass
    except BaseException:
        if swapped and output_dir.exists():
            os.replace(output_dir, failed_release)
        if previous_release.exists() and not output_dir.exists():
            os.replace(previous_release, output_dir)
        raise
    finally:
        if not previous_release.exists():
            shutil.rmtree(staging_root, ignore_errors=True)


def _scenario_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
        str(row.get("path") or ""),
    )


def _scenario_pair(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _validate_readiness_scenario_bindings(
    readiness: Mapping[str, Any],
    readiness_rows: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> None:
    bindings = readiness.get("scenario_yaml_bindings")
    scenario_paths: dict[str, Path] = {}
    for row in readiness_rows:
        scenario_id = str(row.get("scenario_id") or "")
        if not scenario_id or scenario_id in scenario_paths:
            raise ValueError("readiness_scenario_yaml_binding_identity_invalid")
        scenario_paths[scenario_id] = (repo_root / str(row.get("path") or "")).resolve()
    if not isinstance(bindings, Mapping) or set(map(str, bindings)) != set(
        scenario_paths
    ):
        raise ValueError("readiness_scenario_yaml_bindings_invalid")
    for scenario_id, expected_path in scenario_paths.items():
        binding = bindings.get(scenario_id)
        raw_path = binding.get("path") if isinstance(binding, Mapping) else None
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"readiness_scenario_yaml_binding_invalid:{scenario_id}")
        path = Path(raw_path)
        resolved = (
            path.resolve() if path.is_absolute() else (repo_root / path).resolve()
        )
        try:
            _relative(resolved, repo_root)
        except ValueError as exc:
            raise ValueError(
                f"readiness_scenario_yaml_binding_invalid:{scenario_id}"
            ) from exc
        if (
            resolved != expected_path
            or not resolved.is_file()
            or binding.get("sha256") != _sha256(resolved)
        ):
            raise ValueError(f"readiness_scenario_yaml_binding_invalid:{scenario_id}")


def _validate_readiness_source_bindings(
    readiness: Mapping[str, Any],
    readiness_rows: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> None:
    bindings = readiness.get("source_file_bindings")
    scenario_ids = [str(row.get("scenario_id") or "") for row in readiness_rows]
    if (
        not isinstance(bindings, Mapping)
        or len(scenario_ids) != len(set(scenario_ids))
        or set(map(str, bindings)) != set(scenario_ids)
    ):
        raise ValueError("readiness_source_file_bindings_invalid")
    for scenario_id in scenario_ids:
        sources = bindings.get(scenario_id)
        if not isinstance(sources, Mapping) or not sources:
            raise ValueError(f"readiness_source_file_binding_invalid:{scenario_id}")
        for raw_path, expected_sha256 in sources.items():
            raw = str(raw_path)
            digest = str(expected_sha256)
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError(f"readiness_source_file_binding_invalid:{scenario_id}")
            if SAFE_SOURCE_URI_RE.match(raw):
                parsed = urlsplit(raw)
                if (
                    parsed.scheme.lower() == "file"
                    or "\\" in raw
                    or any(part == ".." for part in PurePosixPath(parsed.path).parts)
                    or any(ord(character) < 32 for character in raw)
                ):
                    raise ValueError(
                        f"readiness_source_file_binding_invalid:{scenario_id}"
                    )
                continue
            path = Path(raw)
            resolved = (
                path.resolve() if path.is_absolute() else (repo_root / path).resolve()
            )
            try:
                _relative(resolved, repo_root)
            except ValueError as exc:
                raise ValueError(
                    f"readiness_source_file_binding_invalid:{scenario_id}"
                ) from exc
            if not resolved.is_file() or _sha256(resolved) != digest:
                raise ValueError(f"readiness_source_file_binding_invalid:{scenario_id}")


def _validate_source_candidate_inventories(
    source: Mapping[str, Any],
    source_rows: list[dict[str, Any]],
) -> None:
    candidate_fields = sorted(
        str(field) for field in source if str(field).endswith("_candidates")
    )
    for field in candidate_fields:
        if field in UNRESOLVED_SOURCE_INVENTORIES:
            raise ValueError(f"source_suite_unresolved_candidate_inventory:{field}")
        if field != TERMINAL_SOURCE_INVENTORY:
            raise ValueError(f"source_suite_candidate_inventory_unsupported:{field}")

    abandoned = source.get(TERMINAL_SOURCE_INVENTORY, [])
    if not isinstance(abandoned, list):
        raise ValueError("source_suite_abandoned_candidate_inventory_invalid")
    source_identities = {_scenario_pair(row) for row in source_rows}
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(abandoned):
        prefix = f"source_suite_abandoned_candidate_invalid:{index}"
        if not isinstance(row, Mapping):
            raise ValueError(f"{prefix}:object")
        identity = _scenario_pair(row)
        if not all(identity):
            raise ValueError(f"{prefix}:identity")
        if row.get("disposition") != "abandoned_terminal":
            raise ValueError(f"{prefix}:disposition")
        if row.get("included") is not False:
            raise ValueError(f"{prefix}:included")
        reason_codes = row.get("reason_codes")
        if not (
            isinstance(reason_codes, list)
            and reason_codes
            and all(isinstance(reason, str) and reason for reason in reason_codes)
        ):
            raise ValueError(f"{prefix}:reason_codes")
        if identity in seen:
            raise ValueError(f"{prefix}:duplicate")
        if identity in source_identities:
            raise ValueError(f"{prefix}:overlaps_scenarios")
        seen.add(identity)


def _stage_path(pipeline_dir: Path, stage: str) -> Path:
    canonical = pipeline_dir / STAGE_FILES[stage]
    if canonical.is_file():
        return canonical
    # A parameterized fixture may use the stage name directly.  Production
    # still resolves canonical filenames first.
    return pipeline_dir / f"{stage}.json"


def _physical_lock(row: dict[str, Any]) -> Any:
    ledger = row.get("case_ledger")
    if isinstance(ledger, dict):
        lock = ledger.get("physical_source_lock")
        if lock is not None:
            return lock
    lock = row.get("_physical_source_lock")
    if lock is None:
        raise ValueError(f"physical_source_lock_missing:{row.get('scenario_id', '')}")
    return lock


def build_pglib_uc_release_metadata(
    *,
    selected_rows: list[dict[str, Any]],
    backend_runtime_closure: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build PGLib-UC release claims only from selected, runtime-bound rows."""

    pglib_rows = [
        row
        for row in selected_rows
        if row.get("backend_kind") == "pglib_uc_synthetic"
    ]
    if not pglib_rows:
        raise ValueError("pglib_uc_release_metadata_invalid:no_selected_rows")
    external_sources = backend_runtime_closure.get("external_sources")
    source = (
        external_sources.get("pglib_uc")
        if isinstance(external_sources, Mapping)
        else None
    )
    metadata = source.get("metadata") if isinstance(source, Mapping) else None
    required_files = (
        source.get("required_files") if isinstance(source, Mapping) else None
    )
    roles = metadata.get("roles") if isinstance(metadata, Mapping) else None
    if not (
        isinstance(source, Mapping)
        and source.get("delivery") == "git_checkout"
        and source.get("url") == PGLIB_UC_URL
        and source.get("revision") == PGLIB_UC_REVISION
        and isinstance(required_files, Mapping)
        and isinstance(metadata, Mapping)
        and metadata.get("backend_kinds") == ["pglib_uc_synthetic"]
        and metadata.get("license_status") == PGLIB_UC_LICENSE_STATUS
        and metadata.get("redistributed") is False
        and metadata.get("root") == PGLIB_UC_ROOT
        and isinstance(roles, Mapping)
    ):
        raise ValueError("pglib_uc_release_metadata_invalid:runtime_source")

    expected_files = {PGLIB_UC_LICENSE_PATH: PGLIB_UC_LICENSE_SHA256}
    expected_roles: dict[str, list[str]] = {PGLIB_UC_LICENSE_PATH: ["license"]}
    for row in pglib_rows:
        lock = _physical_lock(row)
        required_assets = (
            lock.get("required_source_assets")
            if isinstance(lock, Mapping)
            else None
        )
        if (
            not isinstance(lock, Mapping)
            or lock.get("backend_kind") != "pglib_uc_synthetic"
            or not isinstance(required_assets, list)
            or not required_assets
        ):
            raise ValueError("pglib_uc_release_metadata_invalid:physical_lock")
        for asset in required_assets:
            path = asset.get("declared_path") if isinstance(asset, Mapping) else None
            digest = asset.get("sha256") if isinstance(asset, Mapping) else None
            if (
                not isinstance(path, str)
                or not path.startswith(f"{PGLIB_UC_ROOT}/")
                or path == PGLIB_UC_LICENSE_PATH
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                or path in expected_files
                and expected_files[path] != digest
            ):
                raise ValueError("pglib_uc_release_metadata_invalid:source_asset")
            expected_files[path] = digest
            expected_roles[path] = ["runtime_input"]
    if dict(required_files) != expected_files or dict(roles) != expected_roles:
        raise ValueError("pglib_uc_release_metadata_invalid:required_files")

    dataset = {
        "url": (
            "https://github.com/power-grid-lib/pglib-uc/tree/"
            f"{PGLIB_UC_REVISION}"
        ),
        "license": "CC-BY-4.0 data; MIT software",
        "lock_strategy": "git_commit+required_file_sha256+license_sha256",
        "redistribution": (
            "not redistributed; clone the pinned upstream checkout and verify "
            "every required file by SHA-256"
        ),
        "delivery": "git_checkout",
        "commit": PGLIB_UC_REVISION,
        "license_sha256": PGLIB_UC_LICENSE_SHA256,
        "required_file_sha256s": dict(sorted(expected_files.items())),
    }
    descriptor = {
        "category": "deterministic_unit_commitment_dispatch_simulator",
        "description": (
            "Deterministic multi-period unit-commitment and dispatch simulator "
            "driven by source-locked PGLib-UC demand, reserve, renewable, and "
            "generator constraints. It does not solve OPF or power flow."
        ),
        "formal_core_allowed": True,
        "released_scenarios": len(pglib_rows),
        "simulates_unit_commitment": "yes",
        "solves_opf": "no",
        "solves_power_flow": "no",
    }
    return dataset, descriptor


def _validate_scenario_yaml(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"scenario_yaml_missing:{path}")
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"scenario_yaml_not_object:{path}")
    errors = verify_scenario_row_against_yaml(row, path=path)
    if errors:
        raise ValueError(
            f"scenario_yaml_identity_invalid:{row.get('scenario_id', '')}:"
            + ",".join(sorted(errors))
        )
    provenance = body.get("provenance")
    if not isinstance(provenance, dict) or any(
        not str(provenance.get(field) or "")
        for field in ("data_source", "url", "license", "lock_strategy")
    ):
        raise ValueError(f"scenario_provenance_incomplete:{row.get('scenario_id', '')}")
    backend_config = body.get("backend_config")
    if not isinstance(backend_config, dict):
        raise ValueError(
            f"scenario_backend_config_missing:{row.get('scenario_id', '')}"
        )
    applicability = backend_config.get("dimension_applicability")
    applicability_issue = dimension_applicability_contract_issue(applicability)
    if applicability_issue is None:
        return body
    issue, dimension = applicability_issue
    if issue == "incomplete":
        raise ValueError(
            f"scenario_dimension_applicability_incomplete:{row.get('scenario_id', '')}"
        )
    if issue == "invalid":
        raise ValueError(
            f"scenario_dimension_applicability_invalid:"
            f"{row.get('scenario_id', '')}:{dimension}"
        )
    raise ValueError(
        f"scenario_dimension_reason_missing:{row.get('scenario_id', '')}:{dimension}"
    )


def _distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _canonical_runtime_path(
    value: object,
    *,
    label: str,
    archive_file: bool = False,
) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError(f"backend_runtime_closure_path_invalid:{label}")
    path = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or "\\" in value
        or any(ord(character) < 32 for character in value)
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
        or (archive_file and (len(path.parts) < 3 or path.parts[0] != "backends"))
    ):
        raise ValueError(f"backend_runtime_closure_path_invalid:{label}")
    return path


def _runtime_identity_sha256(payload: Mapping[str, Any]) -> str:
    identity = {
        key: value for key, value in payload.items() if key != "identity_sha256"
    }
    canonical = json.dumps(
        identity,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _canonical_payload_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _runtime_string_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and len(value) == len(set(value))
        and all(isinstance(item, str) and item for item in value)
    )


def _validate_backend_runtime_closure(
    payload: dict[str, Any],
    *,
    release_id: str,
    source_suite_sha256: str,
    expected_uv_lock_sha256: str | None = None,
) -> None:
    if set(payload) != BACKEND_RUNTIME_CLOSURE_FIELDS:
        raise ValueError("backend_runtime_closure_fields_invalid")
    if payload.get("schema_version") != BACKEND_RUNTIME_CLOSURE_SCHEMA:
        raise ValueError("backend_runtime_closure_schema_invalid")
    if payload.get("release_id") != release_id:
        raise ValueError("backend_runtime_closure_release_mismatch")
    if payload.get("status") != "backend_runtime_closure_complete":
        raise ValueError("backend_runtime_closure_status_invalid")
    if payload.get("terminal") is not True or payload.get("portable") is not True:
        raise ValueError("backend_runtime_closure_semantics_invalid")
    if payload.get("source_suite_sha256") != source_suite_sha256:
        raise ValueError("backend_runtime_closure_source_suite_mismatch")

    archived_files = payload.get("archived_files")
    repo_tracked_files = payload.get("repo_tracked_files")
    separately_bundled_files = payload.get("separately_bundled_files")
    external_sources = payload.get("external_sources")
    backend_links = payload.get("backend_links")
    runtime_packages = payload.get("runtime_packages")
    if not all(
        isinstance(value, dict)
        for value in (
            archived_files,
            repo_tracked_files,
            separately_bundled_files,
            external_sources,
            backend_links,
            runtime_packages,
        )
    ):
        raise ValueError("backend_runtime_closure_maps_invalid")

    archive_roots: set[str] = set()
    archived_sources: set[str] = set()
    for archive_name, row in archived_files.items():
        archive_path = _canonical_runtime_path(
            archive_name,
            label=f"archived_files:{archive_name}",
            archive_file=True,
        )
        if not isinstance(row, dict) or set(row) != {
            "source_path",
            "sha256",
            "roles",
            "backend_kinds",
        }:
            raise ValueError(
                f"backend_runtime_closure_archived_file_invalid:{archive_name}"
            )
        source_path = _canonical_runtime_path(
            row.get("source_path"),
            label=f"source_path:{archive_name}",
        ).as_posix()
        digest = str(row.get("sha256") or "")
        roles = row.get("roles")
        backend_kinds = row.get("backend_kinds")
        if (
            re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or source_path in archived_sources
            or not isinstance(roles, list)
            or not roles
            or len(roles) != len(set(roles))
            or not all(isinstance(role, str) and role for role in roles)
            or not isinstance(backend_kinds, list)
            or not backend_kinds
            or len(backend_kinds) != len(set(backend_kinds))
            or not all(isinstance(kind, str) and kind for kind in backend_kinds)
        ):
            raise ValueError(
                f"backend_runtime_closure_archived_file_invalid:{archive_name}"
            )
        archived_sources.add(source_path)
        archive_roots.add(archive_path.parts[1])

    classified_source_paths = set(archived_sources)
    for inventory_name, inventory in (
        ("repo_tracked_files", repo_tracked_files),
        ("separately_bundled_files", separately_bundled_files),
    ):
        for raw_path, row in inventory.items():
            path = _canonical_runtime_path(
                raw_path,
                label=f"{inventory_name}:{raw_path}",
            ).as_posix()
            roles = row.get("roles") if isinstance(row, dict) else None
            backend_kinds = (
                row.get("backend_kinds") if isinstance(row, dict) else None
            )
            if (
                not isinstance(row, dict)
                or set(row) != RUNTIME_FILE_IDENTITY_FIELDS
                or path in classified_source_paths
                or re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256") or ""))
                is None
                or not _runtime_string_list(roles)
                or roles != sorted(roles)
                or not _runtime_string_list(backend_kinds)
                or backend_kinds != sorted(backend_kinds)
            ):
                raise ValueError(
                    "backend_runtime_closure_file_identity_invalid:"
                    f"{inventory_name}:{raw_path}"
                )
            classified_source_paths.add(path)

    external_paths: set[str] = set()
    for source_id, row in external_sources.items():
        if (
            not isinstance(source_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", source_id) is None
            or not isinstance(row, dict)
            or set(row) != EXTERNAL_SOURCE_FIELDS
        ):
            raise ValueError(
                f"backend_runtime_closure_external_source_invalid:{source_id}"
            )
        parsed = urlsplit(str(row.get("url") or ""))
        required_files = row.get("required_files")
        metadata = row.get("metadata")
        if (
            row.get("delivery")
            not in {"git_checkout", "upstream_fetch", "user_provided"}
            or parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or not isinstance(row.get("revision"), str)
            or not row["revision"]
            or not isinstance(required_files, dict)
            or not required_files
            or not isinstance(metadata, dict)
            or set(metadata) != EXTERNAL_SOURCE_METADATA_FIELDS
            or not _runtime_string_list(metadata.get("backend_kinds"))
            or not isinstance(metadata.get("license_status"), str)
            or not metadata["license_status"]
            or metadata.get("redistributed") is not False
            or not isinstance(metadata.get("roles"), dict)
            or not isinstance(metadata.get("root"), str)
        ):
            raise ValueError(
                f"backend_runtime_closure_external_source_invalid:{source_id}"
            )
        root = _canonical_runtime_path(
            metadata["root"],
            label=f"external_source:{source_id}:root",
        )
        roles = metadata["roles"]
        if set(roles) != set(required_files):
            raise ValueError(
                f"backend_runtime_closure_external_source_invalid:{source_id}"
            )
        for required_path, digest in required_files.items():
            path = _canonical_runtime_path(
                required_path,
                label=f"external_source:{source_id}:{required_path}",
            )
            if (
                path != root
                and root not in path.parents
                or required_path in external_paths
                or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
                or not _runtime_string_list(roles.get(required_path))
            ):
                raise ValueError(
                    f"backend_runtime_closure_external_source_invalid:{source_id}"
                )
            external_paths.add(required_path)

    for works_name, target in backend_links.items():
        if (
            not isinstance(works_name, str)
            or not works_name
            or "/" in works_name
            or "\\" in works_name
            or any(ord(character) < 32 for character in works_name)
            or not isinstance(target, str)
            or PurePosixPath(target).parts != (target,)
            or (
                target not in archive_roots
                and (works_name, target) != ("DynaSchedBench", "dynasched")
            )
        ):
            raise ValueError(
                f"backend_runtime_closure_backend_link_invalid:{works_name}"
            )

    virtual_sources: dict[str, str] = {}
    runtime_uv_lock_sha256s: set[str] = set()
    for package_id, row in runtime_packages.items():
        package_fields = set(row) if isinstance(row, dict) else set()
        allowed_package_fields = {
            RUNTIME_PACKAGE_FIELDS,
            RUNTIME_PACKAGE_FIELDS | {"virtual_sources"},
        }
        if (
            not isinstance(package_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", package_id) is None
            or not isinstance(row, dict)
            or package_fields not in allowed_package_fields
            or ("virtual_sources" in package_fields and package_id != "simbench")
        ):
            raise ValueError(
                f"backend_runtime_closure_runtime_package_invalid:{package_id}"
            )
        lock_entries = row.get("lock_entries")
        if (
            not _runtime_string_list(row.get("backend_kinds"))
            or not isinstance(lock_entries, list)
            or not lock_entries
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("lock_entries_sha256") or ""))
            is None
            or row["lock_entries_sha256"] != _canonical_payload_sha256(lock_entries)
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("uv_lock_sha256") or ""))
            is None
        ):
            raise ValueError(
                f"backend_runtime_closure_runtime_package_invalid:{package_id}"
            )
        if (
            expected_uv_lock_sha256 is not None
            and row["uv_lock_sha256"] != expected_uv_lock_sha256
        ):
            raise ValueError(
                f"backend_runtime_closure_uv_lock_mismatch:{package_id}"
            )
        runtime_uv_lock_sha256s.add(row["uv_lock_sha256"])
        entry_identities: set[str] = set()
        for entry in lock_entries:
            fields = set(entry) if isinstance(entry, dict) else set()
            artifacts = entry.get("artifacts") if isinstance(entry, dict) else None
            source = entry.get("source") if isinstance(entry, dict) else None
            source_url = (
                next(iter(source.values()), None) if isinstance(source, dict) else None
            )
            parsed_source = urlsplit(source_url if isinstance(source_url, str) else "")
            entry_identity = (
                str(entry.get("identity_sha256") or "")
                if isinstance(entry, dict)
                else ""
            )
            if (
                fields
                not in {
                    RUNTIME_LOCK_ENTRY_FIELDS,
                    RUNTIME_LOCK_ENTRY_FIELDS | {"resolution_markers"},
                }
                or not isinstance(entry.get("version"), str)
                or not entry["version"]
                or not isinstance(source, dict)
                or len(source) != 1
                or set(source) not in ({"registry"}, {"git"})
                or parsed_source.scheme != "https"
                or not parsed_source.netloc
                or parsed_source.username is not None
                or parsed_source.password is not None
                or not isinstance(artifacts, list)
                or entry_identity in entry_identities
                or re.fullmatch(r"[0-9a-f]{64}", entry_identity) is None
                or entry_identity
                != _canonical_payload_sha256(
                    {
                        key: value
                        for key, value in entry.items()
                        if key != "identity_sha256"
                    }
                )
                or "resolution_markers" in entry
                and not _runtime_string_list(entry["resolution_markers"])
            ):
                raise ValueError(
                    f"backend_runtime_closure_runtime_package_invalid:{package_id}"
                )
            entry_identities.add(entry_identity)
            for artifact in artifacts:
                artifact_fields = set(artifact) if isinstance(artifact, dict) else set()
                parsed_artifact = urlsplit(
                    str(artifact.get("url") or "") if isinstance(artifact, dict) else ""
                )
                if (
                    not isinstance(artifact, dict)
                    or not {"kind", "url", "hash"}.issubset(artifact_fields)
                    or artifact_fields - RUNTIME_ARTIFACT_FIELDS
                    or artifact.get("kind") not in {"sdist", "wheel"}
                    or parsed_artifact.scheme != "https"
                    or not parsed_artifact.netloc
                    or parsed_artifact.username is not None
                    or parsed_artifact.password is not None
                    or re.fullmatch(
                        r"sha256:[0-9a-f]{64}", str(artifact.get("hash") or "")
                    )
                    is None
                    or "size" in artifact
                    and (type(artifact["size"]) is not int or artifact["size"] < 0)
                    or "upload-time" in artifact
                    and (
                        not isinstance(artifact["upload-time"], str)
                        or not artifact["upload-time"]
                    )
                ):
                    raise ValueError(
                        f"backend_runtime_closure_runtime_package_invalid:{package_id}"
                    )
        declared_virtual = row.get("virtual_sources", {})
        if not isinstance(declared_virtual, dict) or (
            "virtual_sources" in row and not declared_virtual
        ):
            raise ValueError(
                f"backend_runtime_closure_runtime_package_invalid:{package_id}"
            )
        for virtual_source, digest in declared_virtual.items():
            if (
                not isinstance(virtual_source, str)
                or not virtual_source.startswith("pandapower-simbench://")
                or not virtual_source.removeprefix("pandapower-simbench://")
                or virtual_source in virtual_sources
                or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
            ):
                raise ValueError(
                    f"backend_runtime_closure_runtime_package_invalid:{package_id}"
                )
            virtual_sources[virtual_source] = digest

    if len(runtime_uv_lock_sha256s) > 1:
        raise ValueError("backend_runtime_closure_uv_lock_inconsistent")

    summary = payload.get("summary")
    if not isinstance(summary, dict) or set(summary) != BACKEND_RUNTIME_SUMMARY_FIELDS:
        raise ValueError("backend_runtime_closure_summary_invalid")
    if not all(type(value) is int and value >= 0 for value in summary.values()):
        raise ValueError("backend_runtime_closure_summary_invalid")
    classified_assets = (
        len(archived_files)
        + len(repo_tracked_files)
        + len(separately_bundled_files)
        + len(virtual_sources)
    )
    if (
        summary["n_archived_files"] != len(archived_files)
        or summary["n_repo_tracked_files"] != len(repo_tracked_files)
        or summary["n_separately_bundled_files"]
        != len(separately_bundled_files)
        or summary["n_external_sources"] != len(external_sources)
        or summary["n_backend_links"] != len(backend_links)
        or summary["n_runtime_packages"] != len(runtime_packages)
        or summary["n_unresolved"] != 0
        or summary["n_virtual_sources"] != len(virtual_sources)
        or not (
            classified_assets
            <= summary["n_source_assets"]
            <= classified_assets + len(external_paths)
        )
    ):
        raise ValueError("backend_runtime_closure_summary_invalid")
    if payload.get("identity_sha256") != _runtime_identity_sha256(payload):
        raise ValueError("backend_runtime_closure_identity_invalid")


def _validate_backend_runtime_closure_input(
    path: Path,
    *,
    repo_root: Path,
    release_id: str,
    source_suite: Mapping[str, Any],
    source_suite_sha256: str,
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError("backend_runtime_closure_missing")
    initial_sha256 = _sha256(path)
    closure = _load_object(path, label="backend_runtime_closure")
    _validate_backend_runtime_closure(
        closure,
        release_id=release_id,
        source_suite_sha256=source_suite_sha256,
        expected_uv_lock_sha256=_repo_uv_lock_sha256(repo_root),
    )
    from scripts.build_operate_backend_runtime_closure import (  # noqa: PLC0415
        validate_opendss_runtime_asset_closure,
    )

    validate_opendss_runtime_asset_closure(
        repo_root=repo_root,
        source_suite=source_suite,
        closure=closure,
    )
    if _sha256(path) != initial_sha256:
        raise ValueError("backend_runtime_closure_changed_during_validation")
    return closure, initial_sha256


def _validate_parent_core_ancestry(
    *,
    parent_manifest_path: Path,
    parent: Mapping[str, Any],
    source_rows: list[dict[str, Any]],
    release_id: str,
) -> None:
    """Require every prior Core identity and stable field in the new source."""

    parent_release_id = parent.get("release_id")
    if release_id == RELEASE_ID and parent_release_id is None:
        return
    if (
        not isinstance(parent_release_id, str)
        or re.fullmatch(r"operate_v\d+_\d+_\d+", parent_release_id) is None
        or parent_release_id == release_id
    ):
        raise ValueError("parent_release_identity_invalid")
    binding = parent.get("core_suite")
    if not isinstance(binding, Mapping):
        raise ValueError("parent_core_binding_missing")
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("parent_core_binding_invalid")
    relative = Path(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("parent_core_binding_invalid")
    parent_root = parent_manifest_path.parent.resolve()
    parent_core_path = (parent_root / relative).resolve()
    try:
        parent_core_path.relative_to(parent_root)
    except ValueError as exc:
        raise ValueError("parent_core_binding_invalid") from exc
    if (
        not parent_core_path.is_file()
        or binding.get("sha256") != _sha256(parent_core_path)
    ):
        raise ValueError("parent_core_binding_invalid")
    parent_core = _load_object(parent_core_path, label="parent_core")
    parent_rows = parent_core.get("scenarios")
    if not (
        parent_core.get("release_id") == parent_release_id
        and parent_core.get("status") == "core_locked"
        and isinstance(parent_rows, list)
        and parent_rows
        and parent_core.get("n_scenarios") == len(parent_rows)
        and binding.get("n_scenarios") == len(parent_rows)
        and all(isinstance(row, dict) for row in parent_rows)
    ):
        raise ValueError("parent_core_binding_invalid")
    source_by_identity = {
        _scenario_pair(row): row
        for row in source_rows
    }
    parent_identities = [_scenario_pair(row) for row in parent_rows]
    if (
        any(not all(identity) for identity in parent_identities)
        or len(parent_identities) != len(set(parent_identities))
        or len(source_by_identity) != len(source_rows)
    ):
        raise ValueError("parent_core_ancestry_mismatch")
    for parent_row, identity in zip(parent_rows, parent_identities):
        source_row = source_by_identity.get(identity)
        if source_row is None or any(
            parent_row.get(field) != source_row.get(field)
            for field in PARENT_RETAINED_SCENARIO_FIELDS
        ):
            raise ValueError("parent_core_ancestry_mismatch")


def _relocation_binding_path(
    ledger: Mapping[str, Any],
    name: str,
    *,
    repo_root: Path,
) -> Path:
    binding = (ledger.get("bindings") or {}).get(name)
    if not isinstance(binding, Mapping):
        raise ValueError(f"candidate_closure_relocation_binding_missing:{name}")
    raw_path = str(binding.get("path") or "")
    path = (repo_root / raw_path).resolve()
    _relative(path, repo_root)
    if (
        not raw_path
        or Path(raw_path).is_absolute()
        or ".." in Path(raw_path).parts
        or not path.is_file()
        or binding.get("sha256") != _sha256(path)
    ):
        raise ValueError(f"candidate_closure_relocation_binding_invalid:{name}")
    return path


def _validated_relocation_identity_map(
    path: Path,
    *,
    repo_root: Path,
    release_id: str,
) -> dict[tuple[str, str], tuple[str, str]]:
    ledger = _load_object(path, label="candidate_relocation_ledger")
    identities = ledger.get("identities")
    if (
        ledger.get("schema_version") != "operate-canonical-relocation-v1"
        or ledger.get("status") != "canonical_relocation_complete"
        or not isinstance(identities, list)
        or ledger.get("n_selected") != len(identities)
    ):
        raise ValueError("candidate_closure_relocation_ledger_invalid")
    if not identities:
        if ledger.get("empty") is not True or ledger.get("bindings") != {}:
            raise ValueError("candidate_closure_empty_relocation_ledger_invalid")
        return {}

    required_bindings = {
        "pipeline_manifest",
        "selection",
        "remapped_selection",
        "old_source_suite",
        "new_source_suite",
    }
    release_bindings = {"release", "runtime_source_lock"}
    bindings = ledger.get("bindings")
    binding_names = set(bindings) if isinstance(bindings, Mapping) else set()
    if binding_names not in {
        frozenset(required_bindings),
        frozenset(required_bindings | release_bindings),
    }:
        raise ValueError("candidate_closure_relocation_bindings_invalid")
    bound_paths = {
        name: _relocation_binding_path(ledger, name, repo_root=repo_root)
        for name in sorted(required_bindings)
    }
    if release_bindings.issubset(binding_names):
        release_binding = bindings["release"]
        bound_release_id = str(
            release_binding.get("release_id")
            if isinstance(release_binding, Mapping)
            else ""
        )
        release_root = str(
            release_binding.get("root")
            if isinstance(release_binding, Mapping)
            else ""
        )
        if (
            re.fullmatch(r"operate_v\d+_\d+_\d+", bound_release_id) is None
            or bound_release_id != release_id
            or release_root != f"release/{bound_release_id}"
        ):
            raise ValueError("candidate_closure_relocation_release_binding_invalid")
        runtime_lock_path = _relocation_binding_path(
            ledger, "runtime_source_lock", repo_root=repo_root
        )
        runtime_lock = _load_object(
            runtime_lock_path, label="candidate_runtime_source_lock"
        )
        if (
            runtime_lock.get("schema_version")
            != "operate-backend-runtime-source-lock-v1"
            or runtime_lock.get("release_id") != bound_release_id
        ):
            raise ValueError(
                "candidate_closure_relocation_runtime_source_lock_invalid"
            )
    manifest = _load_object(
        bound_paths["pipeline_manifest"], label="candidate_replay_manifest"
    )
    old_source = _load_object(
        bound_paths["old_source_suite"], label="candidate_source_suite"
    )
    new_source = _load_object(
        bound_paths["new_source_suite"], label="relocated_source_suite"
    )
    selection = _load_object(bound_paths["selection"], label="candidate_selection")
    remapped = _load_object(
        bound_paths["remapped_selection"], label="relocated_selection"
    )
    terminal = manifest.get("terminal_stage_artifact")
    source_binding = (selection.get("input_bindings") or {}).get("source_suite")
    remapped_source_binding = (remapped.get("input_bindings") or {}).get("source_suite")
    if (
        manifest.get("status") != "candidate_replay_complete"
        or manifest.get("completed_stage") != "materialize_core"
        or manifest.get("source_suite_sha256")
        != _sha256(bound_paths["old_source_suite"])
        or manifest.get("implementation_tree_sha256")
        != ledger.get("implementation_tree_sha256")
        or manifest.get("core_release_pipeline_sha256")
        != ledger.get("core_release_pipeline_sha256")
        or not isinstance(terminal, Mapping)
        or terminal.get("sha256") != _sha256(bound_paths["selection"])
        or not isinstance(source_binding, Mapping)
        or source_binding.get("sha256") != _sha256(bound_paths["old_source_suite"])
        or not isinstance(remapped_source_binding, Mapping)
        or remapped_source_binding.get("sha256")
        != _sha256(bound_paths["new_source_suite"])
    ):
        raise ValueError("candidate_closure_relocation_replay_binding_invalid")

    def _index(rows: Any, *, label: str) -> dict[tuple[str, str], dict[str, Any]]:
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"candidate_closure_relocation_{label}_invalid")
        indexed = {_scenario_pair(row): row for row in rows}
        if len(indexed) != len(rows) or any(not all(pair) for pair in indexed):
            raise ValueError(f"candidate_closure_relocation_{label}_invalid")
        return indexed

    old_source_rows = _index(old_source.get("scenarios"), label="old_source")
    old_selected_rows = _index(selection.get("scenarios"), label="old_selection")
    new_source_rows = _index(new_source.get("scenarios"), label="new_source")
    new_selected_rows = _index(remapped.get("scenarios"), label="new_selection")
    mapping: dict[tuple[str, str], tuple[str, str]] = {}
    new_pairs: set[tuple[str, str]] = set()
    for identity in identities:
        old = identity.get("old") if isinstance(identity, dict) else None
        new = identity.get("new") if isinstance(identity, dict) else None
        scenario_id = str(
            identity.get("scenario_id") if isinstance(identity, dict) else ""
        )
        old_pair = (
            scenario_id,
            str(old.get("scenario_signature") if isinstance(old, dict) else ""),
        )
        new_pair = (
            scenario_id,
            str(new.get("scenario_signature") if isinstance(new, dict) else ""),
        )
        old_row = old_source_rows.get(old_pair)
        new_row = new_source_rows.get(new_pair)
        old_raw_path = str(old.get("path") or "") if isinstance(old, dict) else ""
        new_raw_path = str(new.get("path") or "") if isinstance(new, dict) else ""
        old_path = (repo_root / old_raw_path).resolve()
        new_path = (repo_root / new_raw_path).resolve()
        try:
            _relative(old_path, repo_root)
            _relative(new_path, repo_root)
        except ValueError as exc:
            raise ValueError(
                "candidate_closure_relocation_identity_path_invalid"
            ) from exc
        if (
            not all(old_pair)
            or not all(new_pair)
            or old_pair in mapping
            or new_pair in new_pairs
            or old_row is None
            or new_row is None
            or old_pair not in old_selected_rows
            or new_pair not in new_selected_rows
            or not old_raw_path
            or not new_raw_path
            or Path(old_raw_path).is_absolute()
            or Path(new_raw_path).is_absolute()
            or ".." in Path(old_raw_path).parts
            or ".." in Path(new_raw_path).parts
            or old_path.suffix != ".yaml"
            or new_path.suffix != ".yaml"
            or str(old_row.get("path") or "") != str(old.get("path") or "")
            or str(new_row.get("path") or "") != str(new.get("path") or "")
            or not old_path.is_file()
            or not new_path.is_file()
            or old.get("yaml_sha256") != _sha256(old_path)
            or new.get("yaml_sha256") != _sha256(new_path)
            or verify_scenario_row_against_yaml(old_row, path=old_path)
            or verify_scenario_row_against_yaml(new_row, path=new_path)
        ):
            raise ValueError("candidate_closure_relocation_identity_invalid")
        mapping[old_pair] = new_pair
        new_pairs.add(new_pair)
    if set(mapping) != set(old_selected_rows) or new_pairs != set(new_selected_rows):
        raise ValueError("candidate_closure_relocation_selection_mismatch")
    return mapping


def _validate_candidate_closure_input(
    path: Path, *, repo_root: Path, release_id: str
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError("candidate_closure_missing")
    initial_sha256 = _sha256(path)
    closure = _load_object(path, label="candidate_closure")
    validate_compact_candidate_closure(closure)
    if _sha256(path) != initial_sha256:
        raise ValueError("candidate_closure_changed_during_validation")
    for binding_or_bindings in closure["inputs"].values():
        bindings = (
            [binding_or_bindings]
            if isinstance(binding_or_bindings, dict)
            else binding_or_bindings
        )
        for binding in bindings:
            input_path = (repo_root / binding["path"]).resolve()
            _relative(input_path, repo_root)
            if not input_path.is_file() or _sha256(input_path) != binding["sha256"]:
                raise ValueError(
                    f"candidate_closure_input_hash_mismatch:{binding['path']}"
                )
    relocation_map: dict[tuple[str, str], tuple[str, str]] = {}
    canonical_pairs: set[tuple[str, str]] = set()
    for binding in closure["relocation_ledgers"]:
        relocation_path = (repo_root / binding["path"]).resolve()
        _relative(relocation_path, repo_root)
        if (
            not relocation_path.is_file()
            or _sha256(relocation_path) != binding["sha256"]
        ):
            raise ValueError(
                f"candidate_closure_relocation_hash_mismatch:{binding['path']}"
            )
        current_map = _validated_relocation_identity_map(
            relocation_path,
            repo_root=repo_root,
            release_id=release_id,
        )
        if set(current_map).intersection(relocation_map) or set(
            current_map.values()
        ).intersection(canonical_pairs):
            raise ValueError("candidate_closure_relocation_identity_duplicate")
        relocation_map.update(current_map)
        canonical_pairs.update(current_map.values())
    declared_mapping: dict[tuple[str, str], tuple[str, str]] = {}
    release_excluded_pairs: set[tuple[str, str]] = set()
    for row in closure["candidates"]:
        disposition = row.get("final_disposition")
        replay_pair = _scenario_pair(row.get("replay_identity") or {})
        if disposition == "selected_for_promotion":
            if replay_pair in declared_mapping:
                raise ValueError("candidate_closure_relocation_identity_mismatch")
            declared_mapping[replay_pair] = _scenario_pair(
                row.get("canonical_identity") or {}
            )
        elif (
            disposition == "rejected_terminal"
            and row.get("release_exclusion_reason_code")
        ):
            if row.get("canonical_identity") is not None:
                raise ValueError("candidate_closure_relocation_identity_mismatch")
            release_excluded_pairs.add(replay_pair)
    if (
        set(declared_mapping).intersection(release_excluded_pairs)
        or set(declared_mapping) | release_excluded_pairs != set(relocation_map)
        or any(
            relocation_map[replay_pair] != canonical_pair
            for replay_pair, canonical_pair in declared_mapping.items()
        )
    ):
        raise ValueError("candidate_closure_relocation_identity_mismatch")
    return closure, initial_sha256


def _validate_inputs(
    *,
    repo_root: Path,
    release_id: str,
    source_suite_path: Path,
    candidate_closure_path: Path,
    backend_runtime_closure_path: Path,
    pipeline_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Path],
    dict[str, Any],
    str,
    dict[str, Any],
    str,
]:
    readiness_path = pipeline_dir / STAGE_FILES["readiness"]
    readiness = _load_object(readiness_path, label="readiness")
    if not (
        readiness.get("status") == "formal_evaluation_ready"
        and readiness.get("formal_evaluation_ready") is True
        and readiness.get("formal_run_blockers") == []
        and readiness.get("scoring_version") == SCORING_VERSION
    ):
        raise ValueError("readiness_not_green")
    formal_run_contract = readiness.get("formal_run_contract")
    _validate_formal_run_contract_for_release(
        formal_run_contract,
        release_id=release_id,
    )

    live_identity = implementation_identity(repo_root)
    live_tree = live_identity["implementation_tree_sha256"]
    live_pipeline_sha256 = live_identity.get("core_release_pipeline_sha256")
    live_release_tooling_sha256 = live_identity.get("release_tooling_sha256")
    if (
        not isinstance(live_pipeline_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", live_pipeline_sha256) is None
    ):
        raise ValueError("core_release_pipeline_identity_missing")
    if (
        not isinstance(live_release_tooling_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", live_release_tooling_sha256) is None
    ):
        raise ValueError("release_tooling_identity_missing")
    if readiness.get("implementation_tree_sha256") != live_tree:
        raise ValueError("readiness_implementation_tree_drift")
    if readiness.get("core_release_pipeline_sha256") != live_pipeline_sha256:
        raise ValueError("readiness_core_release_pipeline_drift")

    source = _load_object(source_suite_path, label="source_suite")
    source_rows = source.get("scenarios")
    readiness_rows = readiness.get("scenarios")
    if not isinstance(source_rows, list) or not source_rows:
        raise ValueError("source_suite_scenarios_missing")
    if not isinstance(readiness_rows, list) or not readiness_rows:
        raise ValueError("readiness_scenarios_missing")
    _validate_source_candidate_inventories(source, source_rows)
    if source.get("n_scenarios") != len(source_rows):
        raise ValueError("source_suite_count_mismatch")
    if readiness.get("n_scenarios") != len(readiness_rows):
        raise ValueError("readiness_count_mismatch")
    source_identities = [_scenario_identity(row) for row in source_rows]
    readiness_identities = [_scenario_identity(row) for row in readiness_rows]
    if any(not all(identity) for identity in source_identities + readiness_identities):
        raise ValueError("scenario_identity_incomplete")
    if len(set(source_identities)) != len(source_identities):
        raise ValueError("source_scenario_identity_duplicate")
    if len(set(readiness_identities)) != len(readiness_identities):
        raise ValueError("scenario_identity_duplicate")
    _validate_readiness_scenario_bindings(
        readiness,
        readiness_rows,
        repo_root=repo_root,
    )
    _validate_readiness_source_bindings(
        readiness,
        readiness_rows,
        repo_root=repo_root,
    )

    pipeline_manifest_path = pipeline_dir / "protocol2_v21_pipeline_manifest.json"
    pipeline = _load_object(pipeline_manifest_path, label="pipeline_manifest")
    if pipeline.get("status") != "formal_evaluation_ready":
        raise ValueError("pipeline_not_green")
    if pipeline.get("source_suite_sha256") != _sha256(source_suite_path):
        raise ValueError("pipeline_source_suite_hash_mismatch")
    if pipeline.get("implementation_tree_sha256") != live_tree:
        raise ValueError("pipeline_implementation_tree_drift")
    if pipeline.get("core_release_pipeline_sha256") != live_pipeline_sha256:
        raise ValueError("pipeline_core_release_pipeline_drift")
    if pipeline.get("release_tooling_sha256") != live_release_tooling_sha256:
        raise ValueError("pipeline_release_tooling_drift")
    stages = pipeline.get("stages")
    if not isinstance(stages, list):
        raise ValueError("pipeline_stages_missing")
    stage_names = [str(stage.get("name") or "") for stage in stages]
    if tuple(stage_names) != EXPECTED_STAGES:
        raise ValueError("pipeline_stage_sequence_mismatch")
    stage_paths: dict[str, Path] = {}
    for stage in stages:
        name = str(stage["name"])
        path = _stage_path(pipeline_dir, name)
        stage_paths[name] = path
        artifact = _load_object(path, label=f"pipeline_stage_{name}")
        if stage.get("return_code") != 0:
            raise ValueError(f"pipeline_stage_failed:{name}")
        if stage.get("output_sha256") != _sha256(path):
            raise ValueError(f"pipeline_stage_hash_mismatch:{name}")
        if stage.get("implementation_tree_sha256") != live_tree:
            raise ValueError(f"pipeline_stage_tree_mismatch:{name}")
        if artifact.get("implementation_tree_sha256") != live_tree:
            raise ValueError(f"pipeline_artifact_tree_mismatch:{name}")
        if stage.get("core_release_pipeline_sha256") != live_pipeline_sha256:
            raise ValueError(f"pipeline_stage_toolchain_mismatch:{name}")
        if artifact.get("core_release_pipeline_sha256") != live_pipeline_sha256:
            raise ValueError(f"pipeline_artifact_toolchain_mismatch:{name}")

    selection_path = stage_paths["materialize_core"]
    selection = _load_object(selection_path, label="materialize_core")
    selected_rows = selection.get("scenarios")
    secondary_rows = selection.get("secondary")
    rejected_rows = selection.get("rejected")
    if not (
        selection.get("status") == "protocol21_core_candidate"
        and isinstance(selected_rows, list)
        and isinstance(secondary_rows, list)
        and isinstance(rejected_rows, list)
    ):
        raise ValueError("materialize_selection_invalid")
    expected_counts = {
        "n_source": len(source_rows),
        "n_selected": len(selected_rows),
        "n_secondary": len(secondary_rows),
        "n_rejected": len(rejected_rows),
    }
    if any(selection.get(name) != value for name, value in expected_counts.items()):
        raise ValueError("materialize_count_mismatch")
    if len(source_rows) != len(selected_rows) + len(secondary_rows) + len(
        rejected_rows
    ):
        raise ValueError("materialize_source_partition_invalid")

    selection_bindings = selection.get("input_bindings")
    if not (
        isinstance(selection_bindings, Mapping)
        and _binding_matches_file(
            repo_root=repo_root,
            binding=selection_bindings.get("source_suite"),
            expected_path=source_suite_path,
            live_tree=live_tree,
        )
    ):
        raise ValueError("materialize_source_binding_invalid")

    readiness_bindings = readiness.get("artifact_bindings")
    if not (
        isinstance(readiness_bindings, Mapping)
        and _binding_matches_file(
            repo_root=repo_root,
            binding=readiness_bindings.get("source_suite"),
            expected_path=source_suite_path,
            live_tree=live_tree,
        )
    ):
        raise ValueError("readiness_source_binding_invalid")
    if not _binding_matches_file(
        repo_root=repo_root,
        binding=readiness_bindings.get("core"),
        expected_path=selection_path,
        live_tree=live_tree,
    ):
        raise ValueError("readiness_core_binding_invalid")
    source_artifact = readiness.get("source_artifact")
    if not isinstance(source_artifact, str) or not source_artifact:
        raise ValueError("readiness_source_artifact_invalid")
    source_artifact_path = Path(source_artifact)
    source_artifact_path = (
        source_artifact_path.resolve()
        if source_artifact_path.is_absolute()
        else (repo_root / source_artifact_path).resolve()
    )
    if source_artifact_path != source_suite_path.resolve() or readiness.get(
        "source_artifact_sha256"
    ) != _sha256(source_suite_path):
        raise ValueError("readiness_source_artifact_invalid")

    selection_identities = [_scenario_identity(row) for row in selected_rows]
    if any(not all(identity) for identity in selection_identities):
        raise ValueError("materialize_selected_identity_incomplete")
    if selection_identities != readiness_identities:
        raise ValueError("readiness_core_selection_mismatch")

    source_pairs = [_scenario_pair(row) for row in source_rows]
    selected_pairs = [_scenario_pair(row) for row in selected_rows]
    secondary_pairs = [_scenario_pair(row) for row in secondary_rows]
    rejected_pairs = [_scenario_pair(row) for row in rejected_rows]
    partition_pairs = selected_pairs + secondary_pairs + rejected_pairs
    if (
        any(not all(pair) for pair in source_pairs + partition_pairs)
        or len(set(source_pairs)) != len(source_pairs)
        or len(set(partition_pairs)) != len(partition_pairs)
        or set(partition_pairs) != set(source_pairs)
    ):
        raise ValueError("materialize_source_partition_invalid")

    disposition_counts: Counter[str] = Counter()
    for row in selected_rows:
        disposition = str(row.get("core_disposition") or row.get("status") or "")
        if disposition != "core_locked":
            raise ValueError("materialize_selected_disposition_invalid")
        disposition_counts[disposition] += 1
    for row in secondary_rows:
        disposition = str(
            row.get("core_disposition")
            or row.get("disposition")
            or row.get("status")
            or ""
        )
        if disposition not in MATERIALIZE_SECONDARY_DISPOSITIONS:
            raise ValueError(
                f"materialize_secondary_disposition_invalid:{disposition or '<missing>'}"
            )
        disposition_counts[disposition] += 1
    for row in rejected_rows:
        disposition = str(
            row.get("disposition")
            or row.get("core_disposition")
            or row.get("status")
            or ""
        )
        if disposition not in MATERIALIZE_REJECTED_DISPOSITIONS:
            raise ValueError(
                f"materialize_rejected_disposition_invalid:{disposition or '<missing>'}"
            )
        disposition_counts[disposition] += 1
    declared_dispositions = selection.get("disposition_counts")
    if not isinstance(declared_dispositions, Mapping) or dict(
        disposition_counts
    ) != dict(declared_dispositions):
        raise ValueError("materialize_disposition_counts_mismatch")

    readiness_identity_set = set(readiness_identities)
    if readiness_identity_set != set(source_identities):
        raise ValueError("readiness_source_identity_mismatch")
    if source_identities != readiness_identities:
        raise ValueError("readiness_source_order_mismatch")

    candidate_closure, candidate_closure_sha256 = _validate_candidate_closure_input(
        candidate_closure_path,
        repo_root=repo_root,
        release_id=release_id,
    )
    closure_selected_pairs = [
        (
            str((row.get("canonical_identity") or {}).get("scenario_id") or ""),
            str((row.get("canonical_identity") or {}).get("scenario_signature") or ""),
        )
        for row in candidate_closure["candidates"]
        if row.get("final_disposition") == "selected_for_promotion"
    ]
    _base_pairs, imported_pairs = validate_candidate_import_partition(source)
    if Counter(closure_selected_pairs) != Counter(imported_pairs) or not set(
        imported_pairs
    ).issubset(selected_pairs):
        raise ValueError("candidate_closure_selection_identity_mismatch")
    backend_runtime_closure, backend_runtime_closure_sha256 = (
        _validate_backend_runtime_closure_input(
            backend_runtime_closure_path,
            repo_root=repo_root,
            release_id=release_id,
            source_suite=source,
            source_suite_sha256=_sha256(source_suite_path),
        )
    )
    return (
        source,
        readiness,
        selection,
        pipeline,
        live_identity,
        stage_paths,
        candidate_closure,
        candidate_closure_sha256,
        backend_runtime_closure,
        backend_runtime_closure_sha256,
    )


def promote_release(
    *,
    repo_root: Path = REPO_ROOT,
    parent_manifest_path: Path = DEFAULT_PARENT,
    source_suite_path: Path = DEFAULT_SOURCE,
    candidate_closure_path: Path,
    backend_runtime_closure_path: Path,
    pipeline_dir: Path = DEFAULT_PIPELINE,
    output_dir: Path = DEFAULT_OUTPUT,
    build_public_evidence: bool = True,
    release_id: str = RELEASE_ID,
    release_version: str = "0.61.0",
    selection_policy: str = "quality_core_v2_v060",
    core_settings_stamp: str = "v0.61.0-settings",
) -> dict[str, Any]:
    """Validate replay evidence and materialize release metadata."""
    repo_root = repo_root.resolve()
    parent_manifest_path = parent_manifest_path.resolve()
    source_suite_path = source_suite_path.resolve()
    candidate_closure_path = candidate_closure_path.resolve()
    if backend_runtime_closure_path.is_symlink():
        raise ValueError("backend_runtime_closure_symlink_forbidden")
    backend_runtime_closure_path = backend_runtime_closure_path.resolve()
    pipeline_dir = pipeline_dir.resolve()
    output_dir = output_dir.resolve()
    if re.fullmatch(r"operate_v\d+_\d+_\d+", release_id) is None:
        raise ValueError("release_id_invalid")
    expected_version = ".".join(release_id.removeprefix("operate_v").split("_"))
    if release_version != expected_version:
        raise ValueError("release_version_identity_mismatch")
    if output_dir.name != release_id:
        raise ValueError("release_output_identity_mismatch")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("release_output_not_directory")
    existing_output_snapshot = _directory_snapshot(output_dir)
    required_paths = (
        parent_manifest_path,
        source_suite_path,
        candidate_closure_path,
        backend_runtime_closure_path,
        pipeline_dir,
        output_dir,
    )
    for path in required_paths:
        _relative(path, repo_root)

    (
        source,
        readiness,
        selection,
        pipeline,
        live_identity,
        stage_paths,
        candidate_closure,
        candidate_closure_sha256,
        backend_runtime_closure,
        backend_runtime_closure_sha256,
    ) = _validate_inputs(
        repo_root=repo_root,
        release_id=release_id,
        source_suite_path=source_suite_path,
        candidate_closure_path=candidate_closure_path,
        backend_runtime_closure_path=backend_runtime_closure_path,
        pipeline_dir=pipeline_dir,
    )
    published_source_suite_path = output_dir / "protocol21_source_suite.json"
    published_source_suite_text = source_suite_path.read_text(encoding="utf-8")
    candidate_closure_text = candidate_closure_path.read_text(encoding="utf-8")
    if (
        hashlib.sha256(candidate_closure_text.encode("utf-8")).hexdigest()
        != candidate_closure_sha256
    ):
        raise ValueError("candidate_closure_changed_during_promotion")
    backend_runtime_closure_text = backend_runtime_closure_path.read_text(
        encoding="utf-8"
    )
    if (
        hashlib.sha256(backend_runtime_closure_text.encode("utf-8")).hexdigest()
        != backend_runtime_closure_sha256
    ):
        raise ValueError("backend_runtime_closure_changed_during_promotion")
    parent_sha256 = _sha256(parent_manifest_path)
    parent = _load_object(parent_manifest_path, label="parent_manifest")
    if _sha256(parent_manifest_path) != parent_sha256:
        raise ValueError("parent_manifest_changed_during_promotion")
    _validate_parent_core_ancestry(
        parent_manifest_path=parent_manifest_path,
        parent=parent,
        source_rows=source["scenarios"],
        release_id=release_id,
    )
    pipeline_manifest_path = pipeline_dir / "protocol2_v21_pipeline_manifest.json"

    rows: list[dict[str, Any]] = []
    scenario_sha256s: dict[Path, str] = {}
    for raw_row in readiness["scenarios"]:
        if not isinstance(raw_row, dict):
            raise ValueError("readiness_scenario_not_object")
        scenario_id = str(raw_row.get("scenario_id") or "")
        if raw_row.get("status") != "core_locked":
            raise ValueError(f"scenario_not_core_locked:{scenario_id}")
        disposition = raw_row.get("core_disposition")
        if disposition not in (None, "core_locked"):
            raise ValueError(f"scenario_disposition_not_core_locked:{scenario_id}")
        if raw_row.get("construct_contract") != "operational_agency.v1":
            raise ValueError(f"scenario_construct_contract_invalid:{scenario_id}")
        scenario_path = (repo_root / str(raw_row.get("path") or "")).resolve()
        _relative(scenario_path, repo_root)
        scenario_sha256 = _sha256(scenario_path)
        _validate_scenario_yaml(scenario_path, raw_row)
        if _sha256(scenario_path) != scenario_sha256:
            raise ValueError(f"scenario_yaml_changed_during_promotion:{scenario_id}")
        scenario_sha256s[scenario_path] = scenario_sha256
        physical_key = canonical_physical_source_asset_key(_physical_lock(raw_row))
        row = {
            key: raw_row[key]
            for key in (
                "scenario_id",
                "scenario_signature",
                "path",
                "seed",
                "horizon_ticks",
                "domain",
                "family",
                "backend_kind",
                "difficulty_mode",
                "difficulty_level",
                "source_denominator_key",
                "structural_fingerprint",
                "semantic_fingerprint",
                "construct_contract",
            )
            if key in raw_row
        }
        row.update(
            {
                "physical_source_key": physical_key,
                "status": "core_locked",
                "core_disposition": "core_locked",
                "yaml_sha256": _sha256(scenario_path),
            }
        )
        rows.append(row)

    for field in (
        "scenario_signature",
        "source_denominator_key",
        "structural_fingerprint",
    ):
        values = [str(row.get(field) or "") for row in rows]
        if any(not value for value in values):
            raise ValueError(f"core_{field}_missing")
        if len(values) != len(set(values)):
            raise ValueError(f"core_{field}_duplicate")

    by_domain = _distribution(rows, "domain")
    by_backend = _distribution(rows, "backend_kind")
    by_difficulty = _distribution(rows, "difficulty_level")
    physical_keys = {str(row["physical_source_key"]) for row in rows}
    core_suite = {
        "schema_version": "protocol21-core-v1",
        "release_id": release_id,
        "status": "core_locked",
        "selection_policy": selection_policy,
        "core_settings_stamp": core_settings_stamp,
        "leaderboard_eligible": False,
        "scoring_version": SCORING_VERSION,
        "implementation_tree_sha256": live_identity["implementation_tree_sha256"],
        "n_scenarios": len(rows),
        "n_effective_sources": len(rows),
        "n_physical_sources": len(physical_keys),
        "by_domain": by_domain,
        "by_backend": by_backend,
        "by_difficulty": by_difficulty,
        "scenarios": rows,
    }
    core_path = output_dir / "core_suite.json"
    core_text = _json_text(core_suite)

    evidence_path = output_dir / "protocol21_public_evidence_bundle.json"
    evidence_bundle: dict[str, Any] | None = None
    evidence_text: str | None = None
    if build_public_evidence:
        evidence_bundle = build_public_evidence_bundle(
            pipeline_manifest=pipeline_manifest_path,
            repo_root=repo_root,
        )
        evidence_text = _json_text(evidence_bundle)

    datasets = dict(parent.get("datasets") or {})
    datasets.pop("pglib_uc", None)
    datasets["dynaschedbench"] = {
        "url": "https://github.com/dsbx7/DynaSchedBench",
        "license": "Apache-2.0",
        "lock_strategy": "git_commit_plus_per_file_sha256",
        "redistribution": "source-locked DSBX benchmark assets",
    }
    datasets["alibaba_cluster_trace_gpu_v2023_openb"] = dict(
        OPENB_V2023_DATASET_DECLARATION
    )
    if by_backend.get("sumo_ego", 0):
        datasets["ngsim_us101"] = dict(NGSIM_US101_DATASET_DECLARATION)
    descriptors = {
        key: dict(value)
        for key, value in (parent.get("backend_descriptors") or {}).items()
        if isinstance(value, dict)
    }
    descriptors.pop("pglib_uc_synthetic", None)
    for backend, descriptor in descriptors.items():
        descriptor["released_scenarios"] = by_backend.get(backend, 0)
    if by_backend.get("pglib_uc_synthetic", 0):
        pglib_dataset, pglib_descriptor = build_pglib_uc_release_metadata(
            selected_rows=readiness["scenarios"],
            backend_runtime_closure=backend_runtime_closure,
        )
        if pglib_descriptor["released_scenarios"] != by_backend[
            "pglib_uc_synthetic"
        ]:
            raise ValueError("pglib_uc_release_metadata_invalid:scenario_count")
        datasets["pglib_uc"] = pglib_dataset
        descriptors["pglib_uc_synthetic"] = pglib_descriptor
    descriptors["dynasched_flexible_job_shop"] = {
        "category": "dynamic_flexible_job_shop_scheduling",
        "description": (
            "DynaSchedBench flexible job-shop scheduling over source-locked "
            "DSBX instances, events, and input models."
        ),
        "formal_core_allowed": True,
        "released_scenarios": by_backend.get("dynasched_flexible_job_shop", 0),
        "solves_job_shop_scheduling": "yes",
        "solves_power_flow": "no",
    }
    descriptors["alibaba_openb_gpu_placement"] = {
        "category": "datacenter_multi_resource_gpu_placement",
        "description": (
            "Alibaba OpenB multi-resource GPU pod placement over source-locked "
            "node inventory and pod-arrival traces."
        ),
        "formal_core_allowed": True,
        "released_scenarios": by_backend.get("alibaba_openb_gpu_placement", 0),
        "solves_job_shop_scheduling": "no",
        "solves_power_flow": "no",
    }
    if by_backend.get("sumo_ego", 0):
        descriptors["sumo_ego"] = {
            "category": "autonomous_driving_source_grounded_closed_loop",
            "description": (
                "Native live SUMO ego-control simulation over source-locked "
                "NGSIM US-101 trajectories, typed hazards, runtime assurance, "
                "and guarded recovery."
            ),
            "formal_core_allowed": True,
            "released_scenarios": by_backend["sumo_ego"],
            "runtime_fidelity": "native_live_sumo_reactive",
            "simulates_autonomous_driving": "yes",
            "solves_power_flow": "no",
        }

    pipeline_artifacts: dict[str, Any] = {
        "path": _relative(pipeline_dir, repo_root),
        "pipeline_manifest_sha256": _sha256(pipeline_manifest_path),
        "core_release_pipeline_sha256": live_identity["core_release_pipeline_sha256"],
        "release_tooling_sha256": live_identity["release_tooling_sha256"],
    }
    for stage, path in stage_paths.items():
        if path.relative_to(pipeline_dir) != Path(STAGE_FILES[stage]):
            raise ValueError(f"pipeline_stage_path_not_canonical:{stage}")
        pipeline_artifacts[PIPELINE_HASH_FIELDS[stage]] = _sha256(path)
    pipeline_artifacts["stage_artifacts"] = {
        stage: {
            "relative_path": STAGE_FILES[stage],
            "sha256": _sha256(path),
        }
        for stage, path in stage_paths.items()
    }

    formal_run_contract = dict(readiness.get("formal_run_contract") or {})
    model_count = int(formal_run_contract.get("required_model_count_per_shard") or 1)
    pass_k = int(formal_run_contract.get("minimum_pass_k") or 1)
    readiness_rel = _relative(stage_paths["readiness"], repo_root)
    source_rel = _relative(published_source_suite_path, repo_root)
    pipeline_rel = _relative(pipeline_dir, repo_root)
    release_rel = _relative(output_dir, repo_root)
    runtime_bundle_path = output_dir / FORMAL_RUNTIME_BUNDLE_NAME
    runtime_bundle_rel = _relative(runtime_bundle_path, repo_root)
    compact_runtime = tuple(int(part) for part in release_version.split(".")) >= (
        0,
        59,
        0,
    )
    if compact_runtime and evidence_text is None:
        raise ValueError("formal_runtime_bundle_requires_public_evidence")
    formal_runtime_root = release_rel if compact_runtime else pipeline_rel
    formal_selection_source = (
        f"{runtime_bundle_rel}#scenarios"
        if compact_runtime
        else f"{readiness_rel}#scenarios"
    )
    formal_batch_contract = {
        "contract_version": "agentic_persistent.v1",
        "runtime_evidence_root": formal_runtime_root,
        "selection_source": formal_selection_source,
        "interaction_mode": formal_run_contract.get(
            "required_interaction_mode", "logical_persistent"
        ),
        "prompt_mode": formal_run_contract.get("required_prompt_mode", "strict"),
        "seed_mode": formal_run_contract.get("required_seed_mode", "scenario"),
        "scheduler_mode": formal_run_contract.get("required_scheduler_mode", "global"),
        "temperature": formal_run_contract.get("required_temperature", 0.0),
        "max_workers": {
            "minimum": formal_run_contract.get("minimum_max_workers", 1),
            "maximum": formal_run_contract.get("maximum_max_workers", 32),
        },
        "requires_explicit_model_capabilities": formal_run_contract.get(
            "requires_explicit_model_capabilities", False
        ),
        "save_trajectories": formal_run_contract.get("save_trajectories", True),
        "primary_leaderboard_formula_version": (
            readiness.get("primary_leaderboard_formula_version")
            or PRIMARY_LEADERBOARD_FORMULA_VERSION
        ),
        "primary_inference_version": (
            readiness.get("primary_inference_version") or PRIMARY_INFERENCE_VERSION
        ),
    }
    formal_batch_contract.update(
        {
            "cardinality": {
                "models": {"per_shard": model_count},
                "pass_k": {"minimum": pass_k},
                "workers": formal_batch_contract.pop("max_workers"),
            },
            "expected_episode_formula": ("n_scenarios * models_per_shard * pass_k"),
            "agentic_profile": dict(formal_run_contract.get("agentic_profile") or {}),
        }
    )
    wakeup_policy = formal_run_contract.get("wakeup_policy")
    if isinstance(wakeup_policy, dict):
        formal_batch_contract["wakeup_policy"] = dict(wakeup_policy)
    realtime_contract = formal_run_contract.get("realtime_formal_contract")
    formal_realtime_batch_contract = None
    if not isinstance(realtime_contract, dict):
        raise ValueError("formal_realtime_contract_missing_or_invalid")
    formal_realtime_batch_contract = {
        **realtime_contract,
        "runtime_evidence_root": formal_runtime_root,
        "selection_source": formal_selection_source,
        "suite_manifest_sha256": readiness.get("suite_manifest_sha256"),
        "n_scenarios": len(rows),
    }
    public_release_blockers = _public_release_blockers(release_version)
    suite_exclusions = [
        {
            "scope": "logical_persistent_core",
            "reason": {
                "code": "formal_logical_persistent_evaluation_pending",
                "message": (
                    "No release-bound single-model persistent shard has "
                    "completed yet; pass_k is dynamic with a minimum of one."
                ),
            },
        },
        {
            "scope": "realtime_persistent_core",
            "reason": {
                "code": "formal_realtime_persistent_evaluation_pending",
                "message": (
                    "The independent realtime supervision scorecard has not "
                    "completed yet."
                ),
            },
        },
    ]
    if "formal_runtime_evidence_distribution_pending" in public_release_blockers:
        suite_exclusions.append(
            {
                "scope": "public_formal_runtime",
                "reason": {
                    "code": "formal_runtime_evidence_distribution_pending",
                    "message": (
                        "The hash-bound replay and diagnostic runtime evidence "
                        "is not yet distributed with the portable Core."
                    ),
                },
            }
        )
    readiness_by_identity = {
        _scenario_identity(row): row for row in readiness["scenarios"]
    }
    runtime_rows: list[dict[str, Any]] = []
    if compact_runtime:
        for row in rows:
            identity = _scenario_identity(row)
            readiness_row = readiness_by_identity.get(identity)
            case_ledger = (
                readiness_row.get("case_ledger")
                if isinstance(readiness_row, Mapping)
                else None
            )
            if not isinstance(case_ledger, dict) or not case_ledger:
                raise ValueError(
                    f"formal_runtime_case_ledger_missing:{row['scenario_id']}"
                )
            if str(case_ledger.get("source_denominator_key") or "") != str(
                row.get("source_denominator_key") or ""
            ):
                raise ValueError(
                    f"formal_runtime_case_ledger_source_mismatch:{row['scenario_id']}"
                )
            runtime_rows.append({**row, "case_ledger": case_ledger})

    stage_attestations: list[dict[str, Any]] = []
    for stage in pipeline["stages"]:
        stage_name = str(stage.get("name") or "")
        artifact = _load_object(
            stage_paths[stage_name], label=f"formal_runtime_stage_{stage_name}"
        )
        stage_attestations.append(
            {
                "name": stage_name,
                "status": artifact.get("status"),
                "counts": {
                    key: value
                    for key, value in sorted(artifact.items())
                    if key.startswith("n_")
                    and isinstance(value, int)
                    and not isinstance(value, bool)
                },
                "output_sha256": str(stage.get("output_sha256") or ""),
                "return_code": stage.get("return_code"),
                "implementation_tree_sha256": stage.get(
                    "implementation_tree_sha256"
                ),
                "core_release_pipeline_sha256": stage.get(
                    "core_release_pipeline_sha256"
                ),
            }
        )
    runtime_bundle = {
        "schema_version": FORMAL_RUNTIME_BUNDLE_SCHEMA,
        "release_id": release_id,
        "release_version": release_version,
        "status": "formal_evaluation_ready",
        "formal_evaluation_ready": True,
        "formal_run_blockers": [],
        "scoring_version": SCORING_VERSION,
        "core_selection_policy": selection_policy,
        "core_settings_stamp": core_settings_stamp,
        "implementation_tree_sha256": live_identity["implementation_tree_sha256"],
        "core_release_pipeline_sha256": live_identity[
            "core_release_pipeline_sha256"
        ],
        "release_tooling_sha256": live_identity["release_tooling_sha256"],
        "git_head": live_identity["git_head"],
        "suite_manifest_sha256": readiness.get("suite_manifest_sha256"),
        "primary_leaderboard_formula_version": (
            readiness.get("primary_leaderboard_formula_version")
            or PRIMARY_LEADERBOARD_FORMULA_VERSION
        ),
        "primary_inference_version": (
            readiness.get("primary_inference_version") or PRIMARY_INFERENCE_VERSION
        ),
        "task_completion_input_unit": readiness.get("task_completion_input_unit"),
        "task_completion_score_unit": readiness.get("task_completion_score_unit"),
        "weighted_equity_formula_version": readiness.get(
            "weighted_equity_formula_version"
        ),
        "formal_run_contract": formal_run_contract,
        "formal_batch_contract_sha256": hashlib.sha256(
            _json_text(formal_batch_contract).encode("utf-8")
        ).hexdigest(),
        "formal_realtime_batch_contract_sha256": hashlib.sha256(
            _json_text(formal_realtime_batch_contract).encode("utf-8")
        ).hexdigest(),
        "n_scenarios": len(runtime_rows),
        "ordered_scenario_identity_sha256": _ordered_identity_sha256(runtime_rows),
        "core_suite": {
            "path": "core_suite.json",
            "sha256": hashlib.sha256(core_text.encode("utf-8")).hexdigest(),
            "n_scenarios": len(rows),
            "ordered_scenario_identity_sha256": _ordered_identity_sha256(rows),
        },
        "source_suite": {
            "path": "protocol21_source_suite.json",
            "sha256": hashlib.sha256(
                published_source_suite_text.encode("utf-8")
            ).hexdigest(),
            "n_scenarios": len(source["scenarios"]),
            "ordered_scenario_identity_sha256": _ordered_identity_sha256(
                source["scenarios"]
            ),
        },
        "candidate_closure": {
            "path": "candidate_closure.json",
            "sha256": candidate_closure_sha256,
            "identity_set_sha256": candidate_closure["identity_set_sha256"],
        },
        "backend_runtime_closure": {
            "path": "backend_runtime_closure.json",
            "sha256": backend_runtime_closure_sha256,
            "identity_sha256": backend_runtime_closure["identity_sha256"],
        },
        "public_evidence": (
            {
                "path": evidence_path.name,
                "sha256": hashlib.sha256(evidence_text.encode("utf-8")).hexdigest(),
                "binding_root_sha256": evidence_bundle.get("binding_root_sha256"),
            }
            if evidence_text is not None and evidence_bundle is not None
            else None
        ),
        "internal_audit": {
            "pipeline_manifest_sha256": _sha256(pipeline_manifest_path),
            "readiness_sha256": _sha256(stage_paths["readiness"]),
            "stage_attestations": stage_attestations,
        },
        "scenarios": runtime_rows,
    }
    runtime_bundle_text = _json_text(runtime_bundle)
    manifest = {
        # The slim-Core schema binds formal treatment
        # metadata without changing the Core suite shape consumed by the v1
        # integrity verifier.
        "manifest_schema_version": "protocol21-core-v1",
        "release_id": release_id,
        "status": "formal_evaluation_ready",
        "protocol_version": "2.1",
        "scoring_version": SCORING_VERSION,
        "core_selection_policy": selection_policy,
        "core_settings_stamp": core_settings_stamp,
        "parent_published_core": parent.get("release_id"),
        "replaces_core": parent.get("release_id"),
        "cascade_bus_schema_version": parent.get("cascade_bus_schema_version", "1.0"),
        "git_head": live_identity["git_head"],
        "implementation_tree_sha256": live_identity["implementation_tree_sha256"],
        "core_release_pipeline_sha256": live_identity["core_release_pipeline_sha256"],
        "release_tooling_sha256": live_identity["release_tooling_sha256"],
        "core_suite": {
            "path": core_path.name,
            "sha256": hashlib.sha256(core_text.encode("utf-8")).hexdigest(),
            "n_scenarios": len(rows),
        },
        "candidate_closure": {
            "path": "candidate_closure.json",
            "sha256": candidate_closure_sha256,
            "schema_version": candidate_closure["schema_version"],
            "status": candidate_closure["status"],
            "n_independent_candidates": candidate_closure["summary"][
                "n_independent_candidates"
            ],
            "n_terminal_candidates": candidate_closure["summary"][
                "n_terminal_candidates"
            ],
            "n_unresolved_candidates": candidate_closure["summary"][
                "n_unresolved_candidates"
            ],
            "identity_set_sha256": candidate_closure["identity_set_sha256"],
        },
        "backend_runtime_closure": {
            "path": "backend_runtime_closure.json",
            "sha256": backend_runtime_closure_sha256,
            "schema_version": backend_runtime_closure["schema_version"],
            "n_archived_files": backend_runtime_closure["summary"]["n_archived_files"],
            "n_external_sources": backend_runtime_closure["summary"][
                "n_external_sources"
            ],
            "n_backend_links": backend_runtime_closure["summary"]["n_backend_links"],
            "n_runtime_packages": backend_runtime_closure["summary"][
                "n_runtime_packages"
            ],
            "identity_sha256": backend_runtime_closure["identity_sha256"],
        },
        "n_scenarios": len(rows),
        "n_effective_sources": len(rows),
        "n_physical_sources": len(physical_keys),
        "by_domain": by_domain,
        "by_backend": by_backend,
        "by_difficulty": by_difficulty,
        "datasets": datasets,
        "backend_descriptors": descriptors,
        "formal_evaluation_ready": True,
        "public_release_ready": False,
        "leaderboard_eligible": False,
        "public_release_blockers": public_release_blockers,
        "formal_run_contract": formal_run_contract,
        "formal_batch_contract": formal_batch_contract,
        **(
            {"formal_realtime_batch_contract": (formal_realtime_batch_contract)}
            if formal_realtime_batch_contract is not None
            else {}
        ),
        "pipeline_dir": pipeline_rel,
        "pipeline_artifacts": pipeline_artifacts,
        "formal_evidence": (
            {
                "runtime_root": release_rel,
                "readiness": runtime_bundle_rel,
                "internal_audit_root": pipeline_rel,
                "internal_readiness": readiness_rel,
            }
            if compact_runtime
            else {"runtime_root": pipeline_rel, "readiness": readiness_rel}
        ),
        **(
            {
                "formal_runtime_bundle": {
                    "path": FORMAL_RUNTIME_BUNDLE_NAME,
                    "sha256": hashlib.sha256(
                        runtime_bundle_text.encode("utf-8")
                    ).hexdigest(),
                    "size_bytes": len(runtime_bundle_text.encode("utf-8")),
                    "schema_version": FORMAL_RUNTIME_BUNDLE_SCHEMA,
                    "n_scenarios": len(runtime_rows),
                    "ordered_scenario_identity_sha256": _ordered_identity_sha256(
                        runtime_rows
                    ),
                }
            }
            if compact_runtime
            else {}
        ),
        "protocol21_replay": {
            "status": "complete",
            "pipeline_dir": pipeline_rel,
            "source_suite": source_rel,
            "source_suite_sha256": hashlib.sha256(
                published_source_suite_text.encode("utf-8")
            ).hexdigest(),
            "implementation_tree_sha256": live_identity["implementation_tree_sha256"],
            "core_release_pipeline_sha256": live_identity[
                "core_release_pipeline_sha256"
            ],
            "release_tooling_sha256": live_identity["release_tooling_sha256"],
            "n_source": selection["n_source"],
            "n_selected": len(rows),
            "n_secondary": selection["n_secondary"],
            "n_rejected": selection["n_rejected"],
            "n_held_repair": selection["disposition_counts"].get("held_repair", 0),
            "n_retired_intrinsic": selection["disposition_counts"].get(
                "retired_intrinsic", 0
            ),
            "selection_disposition_counts": dict(selection["disposition_counts"]),
            "release_coverage_passed": True,
            "formal_evaluation_ready": True,
            "leaderboard_eligible": False,
            **(
                {
                    "evidence_bundle": evidence_path.name,
                    "evidence_bundle_sha256": hashlib.sha256(
                        evidence_text.encode("utf-8")
                    ).hexdigest(),
                }
                if evidence_text is not None
                else {}
            ),
        },
        "leaderboard_eligibility": {
            "eligible": False,
            "diagnostic_cells": [],
            "uninformative_cells": [],
            "suite_exclusions": suite_exclusions,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_text = _json_text(manifest)
    remaining_blockers = ", ".join(
        f"`{blocker}`" for blocker in public_release_blockers
    )
    readme = f"""# OPERATE v{release_version} Core

This directory is generated by the parameterized Protocol-2.1 promoter from the
hash-bound Protocol-2.1 replay at `{pipeline_rel}`.

- Core scenarios: {len(rows)}
- Effective sources: {len(rows)}
- Canonical physical sources: {len(physical_keys)}
- Scoring version: {SCORING_VERSION}
- Implementation tree: `{live_identity["implementation_tree_sha256"]}`
- Replay status: formal evaluation ready
- Remaining release blockers: {remaining_blockers}

Run the formal batch from `manifest.json`; its `formal_batch_contract` binds
the exact Core pipeline and readiness artifacts. Diagnostic smoke and agency
positive controls are optional, independent reports and are not release gates.
"""
    release_files = {
        "README.md": readme,
        "backend_runtime_closure.json": backend_runtime_closure_text,
        "candidate_closure.json": candidate_closure_text,
        "core_suite.json": core_text,
        "manifest.json": manifest_text,
        "protocol21_source_suite.json": published_source_suite_text,
    }
    if evidence_text is not None:
        release_files[evidence_path.name] = evidence_text
    if compact_runtime:
        release_files[FORMAL_RUNTIME_BUNDLE_NAME] = runtime_bundle_text

    expected_validated_inputs = (
        source,
        readiness,
        selection,
        pipeline,
        live_identity,
        stage_paths,
        candidate_closure,
        candidate_closure_sha256,
        backend_runtime_closure,
        backend_runtime_closure_sha256,
    )

    def _validate_locked_inputs() -> None:
        current_validated_inputs = _validate_inputs(
            repo_root=repo_root,
            release_id=release_id,
            source_suite_path=source_suite_path,
            candidate_closure_path=candidate_closure_path,
            backend_runtime_closure_path=backend_runtime_closure_path,
            pipeline_dir=pipeline_dir,
        )
        if current_validated_inputs != expected_validated_inputs:
            raise ValueError("promotion_inputs_changed_during_promotion")
        if (
            _sha256(parent_manifest_path) != parent_sha256
            or _load_object(parent_manifest_path, label="parent_manifest") != parent
        ):
            raise ValueError("parent_manifest_changed_during_promotion")
        for scenario_path, expected_sha256 in scenario_sha256s.items():
            if not scenario_path.is_file() or _sha256(scenario_path) != expected_sha256:
                raise ValueError(
                    "scenario_yaml_changed_during_promotion:"
                    f"{_relative(scenario_path, repo_root)}"
                )

    _transactional_publish_release(
        output_dir=output_dir,
        repo_root=repo_root,
        files=release_files,
        expected_snapshot=existing_output_snapshot,
        require_public_evidence=build_public_evidence,
        locked_input_validator=_validate_locked_inputs,
    )
    return {
        "release_id": release_id,
        "n_scenarios": len(rows),
        "n_physical_sources": len(physical_keys),
        "implementation_tree_sha256": live_identity["implementation_tree_sha256"],
        "manifest": _relative(manifest_path, repo_root),
        "public_evidence_built": evidence_bundle is not None,
        "public_evidence_command": (
            f"python scripts/build_protocol21_public_evidence_bundle.py "
            f"--pipeline-manifest {_relative(pipeline_manifest_path, repo_root)} "
            f"--output {_relative(evidence_path, repo_root)}"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--source-suite", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--candidate-closure", type=Path, required=True)
    parser.add_argument("--backend-runtime-closure", type=Path, required=True)
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--release-id", default=RELEASE_ID)
    parser.add_argument("--release-version", default="0.61.0")
    parser.add_argument("--selection-policy", default="quality_core_v2_v060")
    parser.add_argument("--core-settings-stamp", default="v0.61.0-settings")
    parser.add_argument("--skip-public-evidence", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = promote_release(
        repo_root=args.repo_root,
        parent_manifest_path=args.parent_manifest,
        source_suite_path=args.source_suite,
        candidate_closure_path=args.candidate_closure,
        backend_runtime_closure_path=args.backend_runtime_closure,
        pipeline_dir=args.pipeline_dir,
        output_dir=args.output_dir,
        build_public_evidence=not args.skip_public_evidence,
        release_id=args.release_id,
        release_version=args.release_version,
        selection_policy=args.selection_policy,
        core_settings_stamp=args.core_settings_stamp,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
