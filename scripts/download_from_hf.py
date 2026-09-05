#!/usr/bin/env python3
"""Download and verify the public OPERATE runtime companion.

The default bundle complements an exact GitHub release tree with hash-bound
formal evidence and native runtime assets.  It is not a portable Core and will
fail closed unless the local release manifest and evaluation runtime match.

Authentication is optional for the public dataset. If present, ``HF_TOKEN`` or
the token returned by ``huggingface_hub.get_token()`` is passed through.

Usage:
    python scripts/download_from_hf.py
    python scripts/download_from_hf.py --revision <DATASET_COMMIT_SHA>
    python scripts/download_from_hf.py --download-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
# Stable compatibility install root for the runtime companion. The directory
# name is intentionally not the active release ID; MANIFEST.json supplies the
# exact release binding and prevents cross-release reuse.
DATA = REPO / "operate_data"
DEFAULT_REPO_ID = "Xnhyacinth/OPERATE"
CLUSTERDATA_URL = "https://github.com/alibaba/clusterdata.git"
CLUSTERDATA_EXPECTED_COMMIT = "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71"
CLUSTERDATA_SPARSE_PATHS = (
    "/README.md",
    "/cluster-trace-v2026-spot-gpu/job_info_df.csv",
    "/cluster-trace-v2026-spot-gpu/node_info_df.csv",
    "/cluster-trace-v2026-spot-gpu/README.md",
    "/cluster-trace-gpu-v2023/csv/openb_node_list_gpu_node.csv",
    "/cluster-trace-gpu-v2023/csv/openb_pod_list_gpuspec33.csv",
)
CLUSTERDATA_EXPECTED_ASSETS = {
    "README.md": "7fa18b5d52dd1640801b932702843da4aa39e62d9e27024a05526d9a59793b65",
    "cluster-trace-v2026-spot-gpu/job_info_df.csv": (
        "113ccee4c28f5c3bbaaca974cd164b9280b7d4c39e53b745443b28eea05e03dd"
    ),
    "cluster-trace-v2026-spot-gpu/node_info_df.csv": (
        "1aba161961a5a4a1a61aa581383c5e5abe3400b59f8597ba8c4eef7597bc9d18"
    ),
    "cluster-trace-v2026-spot-gpu/README.md": (
        "11eb33818d3b40ba430c1bc1e0fd42786af214db334152141e29bccb3d26db2b"
    ),
    "cluster-trace-gpu-v2023/csv/openb_node_list_gpu_node.csv": (
        "2beca64b4d3dfa342036a34b56a495c6cef9225db836c81f541282cb1df320b5"
    ),
    "cluster-trace-gpu-v2023/csv/openb_pod_list_gpuspec33.csv": (
        "eca4f746db1e5b25864ad021b55ece3943e101a3ebd4574d09dcb95c46117652"
    ),
}
_EXACT_SOURCE_ASSET_SPECS = {
    "m5": {
        "backend_kind": "orgym_invmgmt",
        "delivery": "bundle",
        "paths": {
            "works/M5/calendar.csv",
            "works/M5/sales_train_evaluation.csv",
            "works/M5/sell_prices.csv",
        },
        "metadata": {
            "works/M5/source_lock.json":
            "271c94965d27bf74b0d66ba89e71b5bc239ddc5192ce99305bbac256a848a9b3",
        },
        "roles": ["derivation_input", "runtime_input"],
        "redistribution": {
            "dataset": "m5_forecasting_accuracy",
            "license": "M5 competition terms; redistribution permission confirmed",
            "lock_strategy": "competition_release+raw_file_sha256",
            "notice": (
                "Exact release-referenced M5 inputs redistributed with dataset "
                "attribution under confirmed permission."
            ),
            "upstream_commit": "kaggle:m5-forecasting-accuracy:2020-06-01-files",
            "url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy",
        },
    },
    "alibaba_openb_v2023": {
        "backend_kind": "alibaba_openb_gpu_placement",
        "delivery": "upstream_fetch",
        "paths": {
            "works/clusterdata/cluster-trace-gpu-v2023/csv/"
            "openb_node_list_gpu_node.csv",
            "works/clusterdata/cluster-trace-gpu-v2023/csv/"
            "openb_pod_list_gpuspec33.csv",
        },
        "redistribution": {
            "dataset": "alibaba_cluster_trace_gpu_v2023_openb",
            "license": "research trace terms; upstream repository license applies",
            "lock_strategy": ("upstream_git_commit_raw_sha256_and_explicit_row_graph"),
            "notice": (
                "Trace bytes are not redistributed; the downloader fetches the "
                "exact files from the pinned upstream commit and verifies hashes."
            ),
            "upstream_commit": "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71",
            "url": (
                "https://github.com/alibaba/clusterdata/tree/master/"
                "cluster-trace-gpu-v2023"
            ),
        },
    },
    "realm_j2": {
        "backend_kind": "jsplib_job_shop",
        "delivery": "bundle",
        "paths": {
            "works/REALM-Bench-direct-pilot/datasets/clean/JSSP/J2.json",
        },
        "source_denominator_prefix": "realm_j2_ccby:",
        "redistribution": {
            "dataset": "realm_bench_j2_ccby",
            "license": ("CC-BY-4.0 (REALM-Bench README, selected J2 JSON instance)"),
            "lock_strategy": (
                "git_commit+file_sha256+selected_row_id+cc-by-runtime-source"
            ),
            "notice": (
                "Exact release-referenced CC-BY-4.0 runtime source with upstream "
                "attribution preserved in the bundle manifest."
            ),
            "upstream_commit": "9c3aa2ae97d65198f6ee29fe942d99f9b3a9c6eb",
            "url": (
                "https://github.com/genglongling/REALM-Bench/tree/"
                "9c3aa2ae97d65198f6ee29fe942d99f9b3a9c6eb"
            ),
        },
    },
    "nrel_microgrid": {
        "backend_kinds": {"pandapower_lv", "pymgrid_economic_dispatch"},
        "delivery": "bundle",
        "paths": {
            f"works/nrel-microgrid/{city}.npz"
            for city in (
                "albuquerque_nm",
                "atlanta_ga",
                "boston_ma",
                "chicago_il",
                "columbus_oh",
                "denver_co",
                "las_vegas_nv",
                "miami_fl",
                "minneapolis_mn",
                "nashville_tn",
                "phoenix_az",
                "portland_or",
                "sacramento_ca",
                "salt_lake_city_ut",
                "seattle_wa",
                "tucson_az",
            )
        },
        "redistribution": {
            "dataset": "nrel_oedi_derived_microgrid_profiles",
            "license": "mixed NREL/OEDI/NSRDB/OpenEI terms; attribution required",
            "lock_strategy": "release_runtime_closure_path_sha256",
            "notice": (
                "The bundle redistributes only the 16 deterministic derived NPZ "
                "profiles required by the Core; upstream attribution and source "
                "identifiers remain recorded in scenario provenance."
            ),
            "upstream_commit": "OEDI ComStock AMY2018 2021 release 1",
            "url": (
                "https://data.openei.org/submissions/4520 + "
                "https://nsrdb.nrel.gov + "
                "https://developer.nrel.gov/docs/solar/pvwatts/v8/ + "
                "https://apps.openei.org/IURDB/"
            ),
        },
        "roles": ["derivation_input"],
    },
}
_NGSIM_US101_SOURCE_SPEC = {
    "backend_kind": "sumo_ego",
    "delivery": "bundle",
    "root": (
        "works/autonomous_driving/ngsim/recovery/us101-v60-seven/bundles"
    ),
    "redistribution": {
        "dataset_id": "8ect-6jqj",
        "source_release": "doi:10.21949/1504477",
        "recording_id": "us-101",
        "license_id": "CC-BY-SA-4.0",
        "notice": (
            "Redistributed source-grounded NGSIM US-101 runtime bundles; "
            "shared bytes are stored once and restored to every hash-bound "
            "scenario install path."
        ),
    },
}

_PROTECTED_BUNDLE_TARGET_ROOTS = {
    ".git",
    ".hl",
    ".planning",
    "audit",
    "baselines",
    "core",
    "data",
    "docs",
    "domains",
    "evaluation",
    "release",
    "runner",
    "scenarios",
    "scripts",
    "sources",
    "tests",
    "works",
}
_BUNDLE_OWNER_FILE = ".operate-bundle-owner.json"
_BUNDLE_OWNER_SCHEMA = "operate-bundle-owner-v1"
_HF_REPO_ID = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bundle_path(value: object) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError(f"bundle_path_invalid:{value!r}")
    relative = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or "\0" in value
        or "\\" in value
        or any(character in value for character in "*?[]")
        or relative.is_absolute()
        or not relative.parts
        or any(part in {".", "..", ".cache"} for part in relative.parts)
        or relative.as_posix() != value
        or value in {"MANIFEST.json", ".gitattributes"}
    ):
        raise ValueError(f"bundle_path_invalid:{value!r}")
    return relative


def verify_manifest(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir / "MANIFEST.json"
    if data_dir.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"bundle_manifest_missing:{manifest_path}")
    root = data_dir.resolve()
    if manifest_path.is_symlink() or manifest_path.resolve().parent != root:
        raise ValueError("bundle_manifest_symlink_forbidden")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("bundle_manifest_root_invalid")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("bundle_manifest_files_invalid")
    for relative, expected in files.items():
        relative_path = _canonical_bundle_path(relative)
        if re.fullmatch(r"[0-9a-f]{64}", str(expected)) is None:
            raise ValueError(f"bundle_hash_invalid:{relative}")
        path = data_dir / relative_path
        current = data_dir
        if any(
            (current := current / part).is_symlink() for part in relative_path.parts
        ):
            raise ValueError(f"bundle_path_symlink_forbidden:{relative}")
        try:
            path.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"bundle_path_invalid:{relative}") from exc
        if not path.is_file():
            raise ValueError(f"bundle_file_missing:{relative}")
        if _sha256_file(path) != expected:
            raise ValueError(f"bundle_hash_mismatch:{relative}")
    return manifest


def _verify_exact_download_snapshot(
    data_dir: Path,
    manifest: dict[str, Any],
    *,
    allow_hf_metadata: bool,
) -> None:
    expected = {"MANIFEST.json", *manifest["files"]}
    observed: set[str] = set()
    for path in data_dir.rglob("*"):
        relative = path.relative_to(data_dir)
        relative_text = relative.as_posix()
        if allow_hf_metadata and (
            relative_text in {".DS_Store", ".gitattributes"}
            or relative.parts[:1] == (".cache",)
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"bundle_snapshot_symlink_forbidden:{relative_text}")
        if path.is_file():
            observed.add(relative_text)
    if observed != expected:
        raise ValueError(
            "bundle_snapshot_not_exact:"
            f"stale={sorted(observed - expected)}:"
            f"missing={sorted(expected - observed)}"
        )


def validate_bundle_download_target(
    data_dir: Path,
    *,
    repo_root: Path = REPO,
    require_within_repo: bool,
) -> None:
    """Reject aliases to source/release trees and non-directory targets."""
    _reject_existing_parent_symlinks(
        data_dir,
        boundary=repo_root if require_within_repo else None,
        error="bundle_download_target_parent_symlink",
    )
    if data_dir.is_symlink():
        raise ValueError("bundle_download_target_symlink")
    if data_dir.exists() and not data_dir.is_dir():
        raise ValueError("bundle_download_target_not_directory")
    repo = repo_root.resolve()
    target = data_dir.resolve()
    if target == repo or target == Path(target.anchor):
        raise ValueError("bundle_download_target_not_dedicated")
    try:
        relative = target.relative_to(repo)
    except ValueError:
        if require_within_repo:
            raise ValueError("bundle_download_target_outside_repo") from None
        return
    if not relative.parts or relative.parts[0] in _PROTECTED_BUNDLE_TARGET_ROOTS:
        raise ValueError("bundle_download_target_protected")


def _reject_existing_parent_symlinks(
    path: Path,
    *,
    boundary: Path | None,
    error: str,
) -> None:
    candidate = path.absolute()
    if boundary is None:
        current = Path(candidate.anchor)
        parts = candidate.parts[1:-1]
    else:
        current = boundary.absolute()
        try:
            relative = candidate.relative_to(current)
        except ValueError as exc:
            raise ValueError(f"{error}:outside_boundary:{path}") from exc
        parts = relative.parts[:-1]
        if current.is_symlink():
            raise ValueError(f"{error}:{current}")
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{error}:{current}")


def _validate_hf_binding(*, repo_id: str, revision: str | None) -> None:
    if _HF_REPO_ID.fullmatch(repo_id) is None:
        raise ValueError("bundle_download_repo_id_invalid")
    if revision is not None and (
        re.fullmatch(r"[0-9a-f]{40,64}", revision) is None
        or set(revision) == {"0"}
    ):
        raise ValueError("bundle_download_revision_invalid")


def _validate_bundle_distribution_binding(
    manifest: dict[str, Any], *, repo_id: str
) -> None:
    if manifest.get("hf_repo_id") != repo_id:
        raise ValueError("bundle_download_hf_repo_id_mismatch")
    if manifest.get("visibility") != "public":
        raise ValueError("bundle_download_visibility_invalid")
    if manifest.get("bundle_kind") != "public_runtime_companion":
        raise ValueError("bundle_download_kind_invalid")


def _bundle_owner_payload(
    *,
    repo_id: str,
    revision: str,
    manifest_path: Path,
) -> dict[str, str]:
    return {
        "schema_version": _BUNDLE_OWNER_SCHEMA,
        "repo_id": repo_id,
        "revision": revision,
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _validate_existing_bundle_owner(data_dir: Path, *, repo_id: str) -> None:
    if not data_dir.exists():
        return
    owner_path = data_dir / _BUNDLE_OWNER_FILE
    if not owner_path.is_file() or owner_path.is_symlink():
        raise ValueError("bundle_download_existing_target_not_owned")
    try:
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("bundle_download_existing_target_not_owned") from exc
    manifest_path = data_dir / "MANIFEST.json"
    if not (
        isinstance(owner, dict)
        and set(owner) == {"schema_version", "repo_id", "revision", "manifest_sha256"}
        and owner.get("schema_version") == _BUNDLE_OWNER_SCHEMA
        and owner.get("repo_id") == repo_id
        and re.fullmatch(r"[0-9a-f]{40,64}", str(owner.get("revision") or ""))
        is not None
        and manifest_path.is_file()
        and not manifest_path.is_symlink()
        and owner.get("manifest_sha256") == _sha256_file(manifest_path)
    ):
        raise ValueError("bundle_download_existing_target_not_owned")
    verify_manifest(data_dir)


@contextmanager
def _bundle_target_lock(data_dir: Path):
    lock_path = data_dir.parent / f".{data_dir.name}.download.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"bundle_download_target_locked:{lock_path}") from exc
    identity: tuple[int, int] | None = None
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("bundle_download_lock_not_regular")
        identity = (metadata.st_dev, metadata.st_ino)
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            current = lock_path.lstat()
        except FileNotFoundError:
            pass
        else:
            if identity == (current.st_dev, current.st_ino):
                lock_path.unlink()


def _download_verified_bundle(
    *,
    repo_id: str,
    revision: str,
    data_dir: Path,
    token: str | None,
    snapshot_download_fn: Callable[..., str],
    repo_root: Path = REPO,
    require_within_repo: bool,
    preflight_runtime_install: bool = True,
    validate_snapshot_fn: Callable[[Path, dict[str, Any]], None] | None = None,
    after_install_fn: Callable[[Path, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Lock, verify off-target, atomically replace, then finish installation."""
    _validate_hf_binding(repo_id=repo_id, revision=revision)
    validate_bundle_download_target(
        data_dir,
        repo_root=repo_root,
        require_within_repo=require_within_repo,
    )
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    with _bundle_target_lock(data_dir):
        return _download_verified_bundle_locked(
            repo_id=repo_id,
            revision=revision,
            data_dir=data_dir,
            token=token,
            snapshot_download_fn=snapshot_download_fn,
            repo_root=repo_root,
            require_within_repo=require_within_repo,
            preflight_runtime_install=preflight_runtime_install,
            validate_snapshot_fn=validate_snapshot_fn,
            after_install_fn=after_install_fn,
        )


def _download_verified_bundle_locked(
    *,
    repo_id: str,
    revision: str,
    data_dir: Path,
    token: str | None,
    snapshot_download_fn: Callable[..., str],
    repo_root: Path,
    require_within_repo: bool,
    preflight_runtime_install: bool,
    validate_snapshot_fn: Callable[[Path, dict[str, Any]], None] | None,
    after_install_fn: Callable[[Path, dict[str, Any]], None] | None,
) -> dict[str, Any]:
    validate_bundle_download_target(
        data_dir,
        repo_root=repo_root,
        require_within_repo=require_within_repo,
    )
    _validate_existing_bundle_owner(data_dir, repo_id=repo_id)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{data_dir.name}.download-",
            dir=data_dir.parent,
        )
    )
    backup: Path | None = None
    try:
        downloaded = Path(
            snapshot_download_fn(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                local_dir=str(staging),
                token=token,
            )
        ).resolve()
        if downloaded != staging.resolve():
            raise ValueError("bundle_download_snapshot_target_mismatch")
        manifest = verify_manifest(staging)
        _validate_bundle_distribution_binding(manifest, repo_id=repo_id)
        _verify_exact_download_snapshot(
            staging,
            manifest,
            allow_hf_metadata=True,
        )
        if validate_snapshot_fn is not None:
            validate_snapshot_fn(staging, manifest)
        from scripts.build_operate_bundle import (  # noqa: PLC0415
            validate_bundle_archives,
        )

        validate_bundle_archives(staging, manifest)
        shutil.rmtree(staging / ".cache", ignore_errors=True)
        (staging / ".DS_Store").unlink(missing_ok=True)
        (staging / ".gitattributes").unlink(missing_ok=True)
        _verify_exact_download_snapshot(
            staging,
            manifest,
            allow_hf_metadata=False,
        )
        (staging / _BUNDLE_OWNER_FILE).write_text(
            json.dumps(
                _bundle_owner_payload(
                    repo_id=repo_id,
                    revision=revision,
                    manifest_path=staging / "MANIFEST.json",
                ),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if preflight_runtime_install:
            _preflight_runtime_install_targets(
                staging,
                manifest,
                repo_root=repo_root,
            )
        else:
            validate_bundle_distribution_contract(staging, manifest)

        validate_bundle_download_target(
            data_dir,
            repo_root=repo_root,
            require_within_repo=require_within_repo,
        )
        if data_dir.exists():
            backup = data_dir.parent / f".{data_dir.name}.backup-{uuid.uuid4().hex}"
            data_dir.replace(backup)
        try:
            staging.replace(data_dir)
        except Exception:
            if backup is not None and backup.exists() and not data_dir.exists():
                backup.replace(data_dir)
                backup = None
            raise
        try:
            if after_install_fn is not None:
                with _formal_install_replay_lock(manifest, repo_root=repo_root):
                    targets = _repo_install_targets(
                        data_dir,
                        manifest,
                        repo_root=repo_root,
                    )
                    with _repo_file_transaction(targets, repo_root=repo_root):
                        after_install_fn(data_dir, manifest)
        except Exception:
            if data_dir.is_dir() and not data_dir.is_symlink():
                shutil.rmtree(data_dir)
            else:
                data_dir.unlink(missing_ok=True)
            if backup is not None:
                backup.replace(data_dir)
                backup = None
            raise
        if backup is not None:
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                print(
                    f"WARNING: verified bundle installed but backup cleanup failed: "
                    f"{backup}: {exc}",
                    file=sys.stderr,
                )
            backup = None
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and backup.exists() and not data_dir.exists():
            backup.replace(data_dir)


def validate_install_data_dir(
    data_dir: Path,
    *,
    repo_root: Path = REPO,
) -> None:
    """Keep full installs inside the repository's evidence trust boundary."""
    try:
        data_dir.resolve().relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError("full_install_data_dir_outside_repo") from exc


def _validate_bundle_formal_evidence_binding(
    data_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Bind bundled formal evidence to its bundled release manifest."""
    release_id = str(manifest.get("release_id") or "")
    release_relative = Path(release_id)
    if (
        not release_id
        or release_relative.is_absolute()
        or len(release_relative.parts) != 1
        or release_id in {".", ".."}
    ):
        raise ValueError("runtime_bundle_release_id_invalid")

    copied_manifest = data_dir / "release_manifest.json"
    archive_name = str(manifest.get("formal_evidence_archive") or "")
    if not archive_name:
        raise ValueError("formal_evidence_archive_missing")
    archive_relative = Path(archive_name)
    expected_manifest_sha256 = str(manifest.get("release_manifest_sha256") or "")
    if (
        archive_relative.is_absolute()
        or ".." in archive_relative.parts
        or archive_name not in (manifest.get("files") or {})
        or not copied_manifest.is_file()
        or copied_manifest.is_symlink()
        or expected_manifest_sha256
        != (manifest.get("files") or {}).get("release_manifest.json")
        or _sha256_file(copied_manifest) != expected_manifest_sha256
    ):
        raise ValueError("runtime_bundle_release_manifest_invalid")

    bundled_release = json.loads(copied_manifest.read_text(encoding="utf-8"))
    expected_tree = str(manifest.get("implementation_tree_sha256") or "")
    expected_pipeline = str(manifest.get("core_release_pipeline_sha256") or "")
    pipeline_artifacts = (
        bundled_release.get("pipeline_artifacts")
        if isinstance(bundled_release, dict)
        else None
    )
    release_replay = (
        bundled_release.get("protocol21_replay")
        if isinstance(bundled_release, dict)
        else None
    )
    if not (
        isinstance(bundled_release, dict)
        and bundled_release.get("release_id") == release_id
        and bundled_release.get("formal_evaluation_ready") is True
        and bundled_release.get("implementation_tree_sha256") == expected_tree
        and re.fullmatch(r"[0-9a-f]{64}", expected_pipeline) is not None
        and bundled_release.get("core_release_pipeline_sha256") == expected_pipeline
        and isinstance(pipeline_artifacts, dict)
        and pipeline_artifacts.get("core_release_pipeline_sha256")
        == expected_pipeline
        and isinstance(release_replay, dict)
        and release_replay.get("core_release_pipeline_sha256") == expected_pipeline
    ):
        raise ValueError("runtime_bundle_release_manifest_invalid")

    install_root = str(manifest.get("formal_evidence_install_root") or "")
    install_relative = Path(install_root)
    formal_contract = bundled_release.get("formal_batch_contract") or {}
    formal_evidence = bundled_release.get("formal_evidence") or {}
    evidence_files = manifest.get("formal_evidence_files")
    required = manifest.get("formal_evidence_required_files")
    compact_binding = bundled_release.get("formal_runtime_bundle")
    if compact_binding is not None:
        expected_root = f"release/{release_id}"
        if not (
            install_root == expected_root
            and install_relative.parts == ("release", release_id)
            and install_root
            == str(formal_contract.get("runtime_evidence_root") or "")
            == str(formal_evidence.get("runtime_root") or "")
        ):
            raise ValueError("formal_evidence_install_root_mismatch")
        replay = bundled_release.get("protocol21_replay")
        expected_source_path = f"{expected_root}/protocol21_source_suite.json"
        if not isinstance(replay, dict) or replay.get(
            "source_suite"
        ) != expected_source_path:
            raise ValueError("formal_runtime_bundle_source_suite_invalid")
        bindings = {
            "formal_runtime_bundle": compact_binding,
            "core_suite": bundled_release.get("core_suite"),
            "source_suite": {
                "path": "protocol21_source_suite.json",
                "sha256": replay.get("source_suite_sha256"),
            },
            "public_evidence": {
                "path": replay.get("evidence_bundle"),
                "sha256": replay.get("evidence_bundle_sha256"),
            },
            "backend_runtime_closure": bundled_release.get(
                "backend_runtime_closure"
            ),
        }
        if not isinstance(bundled_release.get("candidate_closure"), dict):
            raise ValueError("formal_runtime_bundle_candidate_closure_mismatch")
        bindings["candidate_closure"] = bundled_release["candidate_closure"]
        expected_evidence_files = {}
        for label, binding in bindings.items():
            if not isinstance(binding, dict):
                raise ValueError(
                    f"formal_runtime_bundle_artifact_invalid:{label}"
                )
            relative = binding.get("path")
            digest = binding.get("sha256")
            relative_path = Path(str(relative or ""))
            if not (
                isinstance(relative, str)
                and len(relative_path.parts) == 1
                and relative_path.as_posix() == relative
                and isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                and relative not in expected_evidence_files
            ):
                raise ValueError(
                    f"formal_runtime_bundle_artifact_invalid:{label}"
                )
            expected_evidence_files[relative] = digest
        formal_result_trees = manifest.get("formal_result_trees")
        release_result_evidence = {
            "logical_persistent": formal_evidence.get("logical_batch_manifest"),
            "realtime_persistent": formal_evidence.get("realtime_batch_manifest"),
        }
        declared_results = {
            mode for mode, binding in release_result_evidence.items() if binding is not None
        }
        if declared_results:
            if not (
                declared_results == set(release_result_evidence)
                and isinstance(formal_result_trees, dict)
                and set(formal_result_trees) == set(release_result_evidence)
            ):
                raise ValueError("formal_result_tree_contract_incomplete")
            models: set[str] = set()
            for mode, release_binding in release_result_evidence.items():
                contract = formal_result_trees[mode]
                if not isinstance(contract, dict) or set(contract) != {
                    "binding",
                    "files",
                }:
                    raise ValueError(f"formal_result_tree_contract_invalid:{mode}")
                binding = contract["binding"]
                tree_files = contract["files"]
                if not (
                    isinstance(release_binding, dict)
                    and binding == release_binding
                    and isinstance(tree_files, dict)
                    and tree_files
                    and binding.get("interaction_mode") == mode
                    and isinstance(binding.get("model"), str)
                    and binding.get("model")
                ):
                    raise ValueError(f"formal_result_tree_contract_invalid:{mode}")
                models.add(str(binding["model"]))
                release_prefix = Path(expected_root)
                manifest_path = Path(str(binding.get("path") or ""))
                index_path = Path(str(binding.get("tree_index_path") or ""))
                try:
                    manifest_relative = manifest_path.relative_to(release_prefix)
                    index_relative = index_path.relative_to(release_prefix)
                except ValueError as exc:
                    raise ValueError(
                        f"formal_result_tree_contract_invalid:{mode}"
                    ) from exc
                tree_root = manifest_relative.parent
                if not (
                    len(manifest_relative.parts) == 5
                    and manifest_relative.parts[0] == "formal_results"
                    and manifest_relative.name == "RUN_MANIFEST.json"
                    and index_relative == tree_root / "FORMAL_RESULT_TREE_INDEX.json"
                    and tree_root.name == binding.get("tree_root_sha256")
                    and tree_root.parts[-2] == binding.get("treatment_sha256")
                    and tree_files.get(manifest_relative.as_posix())
                    == binding.get("sha256")
                    and tree_files.get(index_relative.as_posix())
                    == binding.get("tree_index_sha256")
                    and all(
                        isinstance(relative, str)
                        and Path(relative).is_relative_to(tree_root)
                        and isinstance(digest, str)
                        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                        for relative, digest in tree_files.items()
                    )
                    and not set(expected_evidence_files) & set(tree_files)
                ):
                    raise ValueError(f"formal_result_tree_contract_invalid:{mode}")
                expected_evidence_files.update(tree_files)
            if len(models) != 1:
                raise ValueError("formal_result_tree_model_mismatch")
        elif formal_result_trees is not None:
            raise ValueError("formal_result_tree_contract_unbound")
        runtime_name = str(compact_binding.get("path") or "")
        runtime_sha256 = str(compact_binding.get("sha256") or "")
        expected_scenarios = bundled_release.get("n_scenarios")
        ordered_identity = compact_binding.get(
            "ordered_scenario_identity_sha256"
        )
        realtime_contract = bundled_release.get("formal_realtime_batch_contract")
        runtime_selection = f"{install_root}/{runtime_name}#scenarios"
        if not (
            set(compact_binding)
            == {
                "path",
                "sha256",
                "size_bytes",
                "schema_version",
                "n_scenarios",
                "ordered_scenario_identity_sha256",
            }
            and compact_binding.get("schema_version")
            == "operate-formal-runtime-bundle-v1"
            and compact_binding.get("n_scenarios") == expected_scenarios
            and isinstance(expected_scenarios, int)
            and expected_scenarios > 0
            and isinstance(compact_binding.get("size_bytes"), int)
            and compact_binding["size_bytes"] > 0
            and isinstance(ordered_identity, str)
            and re.fullmatch(r"[0-9a-f]{64}", ordered_identity) is not None
            and re.fullmatch(r"[0-9a-f]{64}", runtime_sha256) is not None
            and
            formal_evidence.get("readiness") == f"{install_root}/{runtime_name}"
            and formal_contract.get("selection_source") == runtime_selection
            and isinstance(realtime_contract, dict)
            and realtime_contract.get("selection_source") == runtime_selection
        ):
            raise ValueError("formal_evidence_install_root_mismatch")
    else:
        if not (
            not install_relative.is_absolute()
            and ".." not in install_relative.parts
            and len(install_relative.parts) >= 3
            and install_relative.parts[0] == "release"
            and install_root
            == str(formal_contract.get("runtime_evidence_root") or "")
            == str(formal_evidence.get("runtime_root") or "")
            == str(pipeline_artifacts.get("path") or "")
            == str(bundled_release.get("pipeline_dir") or "")
        ):
            raise ValueError("formal_evidence_install_root_mismatch")
        declared_stages = pipeline_artifacts.get("stage_artifacts")
        expected_evidence_files = {
            "protocol2_v21_pipeline_manifest.json": pipeline_artifacts.get(
                "pipeline_manifest_sha256"
            )
        }
        if not isinstance(declared_stages, dict) or not declared_stages:
            raise ValueError("formal_evidence_stage_bindings_missing")
        for stage, binding in declared_stages.items():
            if not isinstance(binding, dict):
                raise ValueError(f"formal_evidence_stage_binding_invalid:{stage}")
            relative = binding.get("relative_path")
            digest = binding.get("sha256")
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or Path(relative).as_posix() != relative
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise ValueError(f"formal_evidence_stage_binding_invalid:{stage}")
            previous = expected_evidence_files.get(relative)
            if previous is not None and previous != digest:
                raise ValueError(f"formal_evidence_stage_binding_conflict:{stage}")
            expected_evidence_files[relative] = digest
    if not (
        isinstance(evidence_files, dict)
        and evidence_files
        and isinstance(required, list)
        and len(required) == len(set(required))
        and set(required) == set(evidence_files)
        and all(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            and Path(relative).as_posix() == relative
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
            for relative, digest in evidence_files.items()
        )
        and evidence_files == expected_evidence_files
    ):
        raise ValueError("formal_evidence_file_contract_invalid")
    if "agency_input_bindings" in manifest:
        raise ValueError("deprecated_agency_input_bindings_forbidden")
    return bundled_release


def validate_runtime_bundle_compatibility(
    data_dir: Path,
    manifest: dict[str, Any],
    *,
    repo_root: Path = REPO,
    verify_implementation: bool = True,
    require_canonical_release_manifest: bool = True,
) -> None:
    """Require an exact local release/runtime before installation or upload.

    A final distribution candidate is intentionally staged outside the
    canonical release directory until its private CAS receipt exists. Upload
    preflight validates those bundled candidate bytes directly; installation
    continues to require the canonical local manifest by default.
    """
    if verify_manifest(data_dir) != manifest:
        raise ValueError("runtime_bundle_manifest_argument_mismatch")
    if not (
        manifest.get("schema_version") == "operate-runtime-bundle-v2"
        and manifest.get("bundle_kind")
        in {"public_runtime_companion", "private_runtime_companion"}
    ):
        raise ValueError("runtime_bundle_contract_invalid")
    release_id = str(manifest.get("release_id") or "")
    expected_manifest_sha256 = str(manifest.get("release_manifest_sha256") or "")
    bundled_release = _validate_bundle_formal_evidence_binding(data_dir, manifest)
    local_release = bundled_release
    if require_canonical_release_manifest:
        local_manifest = repo_root / "release" / release_id / "manifest.json"
        if (
            not local_manifest.is_file()
            or _sha256_file(local_manifest) != expected_manifest_sha256
        ):
            raise ValueError("local_release_manifest_mismatch")
        local_release = json.loads(local_manifest.read_text(encoding="utf-8"))
    expected_tree = str(manifest.get("implementation_tree_sha256") or "")
    expected_pipeline = str(manifest.get("core_release_pipeline_sha256") or "")
    release_pipeline_artifacts = local_release.get("pipeline_artifacts")
    release_replay = local_release.get("protocol21_replay")
    if not (
        (not require_canonical_release_manifest or bundled_release == local_release)
        and local_release.get("release_id") == release_id
        and local_release.get("formal_evaluation_ready") is True
        and local_release.get("implementation_tree_sha256") == expected_tree
        and re.fullmatch(r"[0-9a-f]{64}", expected_pipeline) is not None
        and local_release.get("core_release_pipeline_sha256") == expected_pipeline
        and isinstance(release_pipeline_artifacts, dict)
        and release_pipeline_artifacts.get("core_release_pipeline_sha256")
        == expected_pipeline
        and isinstance(release_replay, dict)
        and release_replay.get("core_release_pipeline_sha256") == expected_pipeline
    ):
        raise ValueError("local_release_manifest_mismatch")
    _validate_candidate_closure_binding(
        data_dir,
        manifest,
        release_manifest=bundled_release,
    )

    if verify_implementation:
        from core.implementation_identity import (  # noqa: PLC0415
            implementation_identity,
        )

        live_tree = implementation_identity(repo_root)["implementation_tree_sha256"]
        if live_tree != expected_tree:
            raise ValueError("local_implementation_tree_mismatch")

    _validate_bundle_source_asset_bindings(
        manifest,
        local_release=local_release,
        repo_root=repo_root,
    )


def _is_trusted_dynasched_asset_path(path: Path) -> bool:
    return (
        path.parts[:2] == ("sources", "dynasched")
        or path.parts[:3] == ("works", "DynaSchedBench", "data")
        or path.parts == ("works", "DynaSchedBench", "LICENSE")
    )


def _dynasched_archive_path(install_path: Path) -> Path:
    if install_path.parts[:2] == ("works", "DynaSchedBench"):
        return Path("backends") / "dynasched" / Path(*install_path.parts[2:])
    return Path("backends") / "dynasched_source_assets" / install_path


def _exact_source_archive_path(source_id: str, install_path: Path) -> Path:
    return Path("backends") / "release_source_assets" / source_id / install_path


def _ngsim_us101_archive_path(digest: str) -> Path:
    return Path("backends/release_source_assets/ngsim_us101/blobs") / digest


def _source_asset_contract_error(source_id: str) -> str:
    return f"{source_id}_source_asset_contract_invalid"


def _source_asset_file_rows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source_assets = manifest.get("source_assets") or {}
    if not isinstance(source_assets, dict):
        raise ValueError("bundle_source_assets_invalid")
    if not source_assets:
        return {}
    supported = {"dynasched", "ngsim_us101", *_EXACT_SOURCE_ASSET_SPECS}
    if not set(source_assets) <= supported:
        raise ValueError("bundle_source_assets_unsupported")
    validated: dict[str, dict[str, Any]] = {}
    archive_paths: set[str] = set()
    for source_id, contract in source_assets.items():
        error = _source_asset_contract_error(source_id)
        if not isinstance(contract, dict):
            raise ValueError(error)
        files = contract.get("files")
        scenario_ids = contract.get("scenario_ids")
        if not (
            isinstance(files, dict)
            and files
            and contract.get("n_files") == len(files)
            and isinstance(scenario_ids, list)
            and scenario_ids == sorted(set(scenario_ids))
            and contract.get("n_scenarios") == len(scenario_ids)
        ):
            raise ValueError(error)
        if source_id == "ngsim_us101":
            root = Path(str(_NGSIM_US101_SOURCE_SPEC["root"]))
            if not (
                contract.get("delivery") == "bundle"
                and contract.get("redistribution")
                == _NGSIM_US101_SOURCE_SPEC["redistribution"]
                and isinstance(contract.get("blobs"), dict)
                and contract.get("n_blobs") == len(contract["blobs"])
            ):
                raise ValueError(error)
            expected_blobs: dict[str, dict[str, Any]] = {}
            ngsim_rows: dict[str, dict[str, Any]] = {}
            for install_path, row in files.items():
                install = Path(str(install_path))
                if not isinstance(row, dict):
                    raise ValueError(error)
                archive_path = str(row.get("archive_path") or "")
                archive = Path(archive_path)
                digest = str(row.get("sha256") or "")
                roles = row.get("roles")
                bound_scenarios = row.get("scenario_ids")
                if not (
                    isinstance(install_path, str)
                    and root in install.parents
                    and not install.is_absolute()
                    and ".." not in install.parts
                    and install.as_posix() == install_path
                    and archive == _ngsim_us101_archive_path(digest)
                    and archive.as_posix() == archive_path
                    and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                    and row.get("delivery") == "bundle"
                    and isinstance(roles, list)
                    and roles == sorted(set(roles))
                    and roles
                    and set(roles)
                    <= {
                        "runtime_input",
                        "derivation_input",
                        "implementation_asset",
                        "metadata",
                        "license",
                    }
                    and isinstance(bound_scenarios, list)
                    and bound_scenarios == sorted(set(bound_scenarios))
                    and bound_scenarios
                    and set(bound_scenarios) <= set(scenario_ids)
                    and install_path not in validated
                ):
                    raise ValueError(f"{error}:{install_path}")
                blob = expected_blobs.setdefault(
                    archive_path,
                    {
                        "archive_path": archive_path,
                        "sha256": digest,
                        "source_path": install_path,
                        "install_paths": [],
                    },
                )
                if blob["sha256"] != digest:
                    raise ValueError(error)
                blob["install_paths"].append(install_path)
                ngsim_rows[install_path] = row
            for blob in expected_blobs.values():
                blob["install_paths"].sort()
                blob["source_path"] = blob["install_paths"][0]
            if (
                contract["blobs"] != dict(sorted(expected_blobs.items()))
                or set(expected_blobs) & archive_paths
            ):
                raise ValueError(error)
            archive_paths.update(expected_blobs)
            validated.update(ngsim_rows)
            continue
        if source_id == "dynasched":
            license_paths = contract.get("redistribution_license_paths")
            if not (
                isinstance(license_paths, list)
                and license_paths
                and license_paths == sorted(set(license_paths))
                and set(license_paths) <= set(files)
            ):
                raise ValueError(error)
        else:
            spec = _EXACT_SOURCE_ASSET_SPECS[source_id]
            if set(files) != set(spec["paths"]) | set(spec.get("metadata", {})):
                raise ValueError(error)
            if contract.get("delivery") != spec["delivery"]:
                raise ValueError(error)
            if contract.get("redistribution") != spec["redistribution"]:
                raise ValueError(f"{source_id}_redistribution_metadata_invalid")
        for install_path, row in files.items():
            install = Path(str(install_path))
            if not isinstance(row, dict):
                raise ValueError(error)
            archive_path = str(row.get("archive_path") or "")
            archive = Path(archive_path)
            digest = str(row.get("sha256") or "")
            roles = row.get("roles")
            delivery = row.get("delivery")
            bound_scenarios = row.get("scenario_ids")
            expected_archive = (
                _dynasched_archive_path(install)
                if source_id == "dynasched"
                else _exact_source_archive_path(source_id, install)
            )
            trusted_path = (
                _is_trusted_dynasched_asset_path(install)
                if source_id == "dynasched"
                else install_path in (
                    _EXACT_SOURCE_ASSET_SPECS[source_id]["paths"]
                    | set(_EXACT_SOURCE_ASSET_SPECS[source_id].get("metadata", {}))
                )
            )
            metadata_digest = (
                None if source_id == "dynasched"
                else _EXACT_SOURCE_ASSET_SPECS[source_id].get("metadata", {}).get(install_path)
            )
            if not (
                isinstance(install_path, str)
                and trusted_path
                and not install.is_absolute()
                and ".." not in install.parts
                and install.as_posix() == install_path
                and not archive.is_absolute()
                and ".." not in archive.parts
                and archive.as_posix() == archive_path
                and archive == expected_archive
                and archive_path not in archive_paths
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                and (metadata_digest is None or digest == metadata_digest)
                and isinstance(roles, list)
                and roles == sorted(set(roles))
                and roles
                and (
                    source_id == "dynasched"
                    or roles
                    == (
                        ["metadata"] if metadata_digest is not None
                        else _EXACT_SOURCE_ASSET_SPECS[source_id].get(
                            "roles", ["runtime_input"]
                        )
                    )
                )
                and (
                    source_id == "dynasched"
                    or delivery == _EXACT_SOURCE_ASSET_SPECS[source_id]["delivery"]
                )
                and isinstance(bound_scenarios, list)
                and bound_scenarios == sorted(set(bound_scenarios))
                and bound_scenarios
                and set(bound_scenarios) <= set(scenario_ids)
                and install_path not in validated
            ):
                raise ValueError(f"{error}:{install_path}")
            archive_paths.add(archive_path)
            validated[install_path] = row
        if source_id == "dynasched" and any(
            "license" not in files[path]["roles"] for path in license_paths
        ):
            raise ValueError("dynasched_redistribution_license_binding_invalid")
    return validated


def _suite_source_scenario_ids(
    source_id: str,
    rows: list[dict[str, Any]],
) -> list[str]:
    if source_id == "dynasched":
        return sorted(
            str(row.get("scenario_id") or "")
            for row in rows
            if row.get("backend_kind") == "dynasched_flexible_job_shop"
        )
    if source_id == "ngsim_us101":
        return sorted(
            str(row.get("scenario_id") or "")
            for row in rows
            if row.get("backend_kind") == "sumo_ego"
        )
    spec = _EXACT_SOURCE_ASSET_SPECS[source_id]
    backend_kinds = set(
        spec.get("backend_kinds") or {str(spec.get("backend_kind") or "")}
    )
    prefix = spec.get("source_denominator_prefix")
    return sorted(
        str(row.get("scenario_id") or "")
        for row in rows
        if row.get("backend_kind") in backend_kinds
        and (
            prefix is None
            or str(row.get("source_denominator_key") or "").startswith(str(prefix))
        )
    )


def _validate_bundle_source_asset_bindings(
    manifest: dict[str, Any],
    *,
    local_release: dict[str, Any],
    repo_root: Path,
) -> None:
    _source_asset_file_rows(manifest)
    if not manifest.get("backend_archive"):
        return
    replay = local_release.get("protocol21_replay")
    if not isinstance(replay, dict):
        raise ValueError("dynasched_source_suite_binding_invalid")
    relative = str(replay.get("source_suite") or "")
    digest = str(replay.get("source_suite_sha256") or "")
    path = Path(relative)
    suite_path = (repo_root / path).resolve()
    try:
        suite_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ValueError("dynasched_source_suite_binding_invalid") from exc
    if not (
        not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == relative
        and suite_path.is_file()
        and _sha256_file(suite_path) == digest
    ):
        raise ValueError("dynasched_source_suite_binding_invalid")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite_rows = suite.get("scenarios") if isinstance(suite, dict) else None
    if not isinstance(suite_rows, list):
        raise ValueError("dynasched_source_suite_binding_invalid")
    typed_suite_rows = [row for row in suite_rows if isinstance(row, dict)]
    contracts = manifest.get("source_assets") or {}
    for source_id in ("dynasched", "ngsim_us101", *_EXACT_SOURCE_ASSET_SPECS):
        expected_scenarios = _suite_source_scenario_ids(
            source_id,
            typed_suite_rows,
        )
        contract = contracts.get(source_id)
        if not expected_scenarios:
            if contract is not None:
                raise ValueError(f"unexpected_{source_id}_source_asset_contract")
            continue
        if contract is None:
            raise ValueError(f"{source_id}_source_asset_contract_missing")
        source_suite = contract.get("source_suite")
        if not isinstance(source_suite, dict) or not (
            source_suite.get("path") == relative
            and source_suite.get("sha256") == digest
            and contract.get("scenario_ids") == expected_scenarios
        ):
            raise ValueError(f"{source_id}_source_suite_binding_invalid")


def _validated_backend_links(manifest: dict[str, Any]) -> dict[str, str]:
    from scripts.build_operate_bundle import BACKEND_LINKS  # noqa: PLC0415

    links = manifest.get("backend_links") or {}
    expected = dict(sorted(BACKEND_LINKS.items()))
    if not isinstance(links, dict) or any(
        expected.get(name) != target for name, target in links.items()
    ):
        raise ValueError("bundle_backend_links_not_release_allowlist")
    return dict(links)


def _validate_backend_license_bindings(
    data_dir: Path,
    manifest: dict[str, Any],
) -> None:
    if not manifest.get("backend_archive"):
        return
    roots = set(_validated_backend_links(manifest).values())
    licenses = manifest.get("backend_licenses")
    archive_files = manifest.get("backend_archive_files") or {}
    closure_binding = manifest.get("backend_runtime_closure") or {}
    closure_path = data_dir / str(closure_binding.get("path") or "")
    if not closure_path.is_file() or closure_path.is_symlink():
        raise ValueError("bundle_backend_license_binding_invalid")
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    expected: dict[str, set[str]] = {}
    for member, row in (closure.get("archived_files") or {}).items():
        path = Path(str(member))
        if (
            isinstance(row, dict)
            and "redistribution_license" in (row.get("roles") or [])
            and path.parts[:1] == ("backends",)
            and len(path.parts) >= 3
        ):
            expected.setdefault(path.parts[1], set()).add(str(member))
    for contract in (manifest.get("source_assets") or {}).values():
        if not isinstance(contract, dict):
            continue
        for row in (contract.get("files") or {}).values():
            if not isinstance(row, dict) or not set(row.get("roles") or []) & {
                "license",
                "redistribution_license",
            }:
                continue
            member = str(row.get("archive_path") or "")
            path = Path(member)
            if path.parts[:1] == ("backends",) and len(path.parts) >= 3:
                expected.setdefault(path.parts[1], set()).add(member)
    expected_map = {root: sorted(expected.get(root, set())) for root in sorted(roots)}
    if not (
        isinstance(licenses, dict)
        and licenses == expected_map
        and all(expected_map.values())
        and isinstance(archive_files, dict)
    ):
        raise ValueError("bundle_backend_license_binding_invalid")
    for root, members in licenses.items():
        if not (
            isinstance(members, list)
            and members == sorted(set(members))
            and members
            and all(
                isinstance(member, str)
                and member in archive_files
                and Path(member).parts[:2] == ("backends", root)
                for member in members
            )
        ):
            raise ValueError(f"bundle_backend_license_binding_invalid:{root}")


def _validate_backend_archive_file_bindings(
    data_dir: Path,
    manifest: dict[str, Any],
) -> None:
    if not manifest.get("backend_archive"):
        return
    binding = manifest.get("backend_runtime_closure") or {}
    closure_path = data_dir / str(binding.get("path") or "")
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    expected = {
        str(member): str(row.get("sha256") or "")
        for member, row in (closure.get("archived_files") or {}).items()
        if isinstance(row, dict)
    }
    closure_members = set(expected)
    for row in _source_asset_file_rows(manifest).values():
        if row.get("delivery") == "upstream_fetch":
            continue
        member = str(row["archive_path"])
        digest = str(row["sha256"])
        if member in closure_members or (
            member in expected and expected[member] != digest
        ):
            raise ValueError(f"bundle_backend_archive_file_collision:{member}")
        expected[member] = digest
    if manifest.get("backend_archive_files") != dict(sorted(expected.items())):
        raise ValueError("bundle_backend_archive_files_binding_invalid")


def _validate_backend_runtime_closure_binding(
    data_dir: Path,
    manifest: dict[str, Any],
) -> None:
    links = _validated_backend_links(manifest)
    if not manifest.get("backend_archive"):
        if links:
            raise ValueError("bundle_backend_runtime_closure_binding_invalid")
        return
    binding = manifest.get("backend_runtime_closure")
    if not isinstance(binding, dict):
        raise ValueError("bundle_backend_runtime_closure_binding_missing")
    relative_text = str(binding.get("path") or "")
    expected = str(binding.get("sha256") or "")
    relative = Path(relative_text)
    closure_path = data_dir / relative
    if not (
        relative_text == "backend_runtime_closure.json"
        and not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == relative_text
        and re.fullmatch(r"[0-9a-f]{64}", expected) is not None
        and (manifest.get("files") or {}).get(relative_text) == expected
        and closure_path.is_file()
        and not closure_path.is_symlink()
        and _sha256_file(closure_path) == expected
    ):
        raise ValueError("bundle_backend_runtime_closure_binding_invalid")
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    top_level = {
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
    summary_fields = {
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
    archived_files = (
        closure.get("archived_files") if isinstance(closure, dict) else None
    )
    repo_tracked_files = (
        closure.get("repo_tracked_files") if isinstance(closure, dict) else None
    )
    separately_bundled_files = (
        closure.get("separately_bundled_files")
        if isinstance(closure, dict)
        else None
    )
    summary = closure.get("summary") if isinstance(closure, dict) else None
    identity = closure.get("identity_sha256") if isinstance(closure, dict) else None
    release_manifest_path = data_dir / "release_manifest.json"
    release_manifest = (
        json.loads(release_manifest_path.read_text(encoding="utf-8"))
        if release_manifest_path.is_file() and not release_manifest_path.is_symlink()
        else None
    )
    release_binding = (
        release_manifest.get("backend_runtime_closure")
        if isinstance(release_manifest, dict)
        else None
    )
    replay = (
        release_manifest.get("protocol21_replay")
        if isinstance(release_manifest, dict)
        else None
    )
    binding_fields = {
        "path",
        "sha256",
        "schema_version",
        "n_archived_files",
        "n_external_sources",
        "n_backend_links",
        "n_runtime_packages",
        "identity_sha256",
    }
    release_id = str(manifest.get("release_id") or "")
    file_inventories_valid = all(
        isinstance(inventory, dict)
        and all(
            isinstance(raw_path, str)
            and raw_path
            and "\\" not in raw_path
            and not PurePosixPath(raw_path).is_absolute()
            and ".." not in PurePosixPath(raw_path).parts
            and PurePosixPath(raw_path).as_posix() == raw_path
            and isinstance(row, dict)
            and set(row) == {"sha256", "roles", "backend_kinds"}
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256") or ""))
            is not None
            and isinstance(row.get("roles"), list)
            and bool(row["roles"])
            and all(isinstance(value, str) and value for value in row["roles"])
            and row["roles"] == sorted(set(row["roles"]))
            and isinstance(row.get("backend_kinds"), list)
            and bool(row["backend_kinds"])
            and all(
                isinstance(value, str) and value for value in row["backend_kinds"]
            )
            and row["backend_kinds"] == sorted(set(row["backend_kinds"]))
            for raw_path, row in inventory.items()
        )
        for inventory in (repo_tracked_files, separately_bundled_files)
    )
    archived_source_paths = {
        str(row.get("source_path") or "")
        for row in (archived_files or {}).values()
        if isinstance(row, dict)
    }
    classified_paths = (
        archived_source_paths
        | set(repo_tracked_files or {})
        | set(separately_bundled_files or {})
    )
    source_asset_rows = _source_asset_file_rows(manifest)
    separately_bundled_valid = all(
        isinstance(source_asset_rows.get(path), dict)
        and source_asset_rows[path].get("sha256") == row.get("sha256")
        and source_asset_rows[path].get("delivery") != "upstream_fetch"
        for path, row in (separately_bundled_files or {}).items()
    )
    raw_external_sources = manifest.get("external_sources") or {}
    external_sources = (
        raw_external_sources if isinstance(raw_external_sources, dict) else {}
    )
    n_external_files = sum(
        len(record.get("required_files") or {})
        for record in external_sources.values()
        if isinstance(record, dict)
    )
    raw_runtime_packages = manifest.get("runtime_packages") or {}
    runtime_packages = (
        raw_runtime_packages if isinstance(raw_runtime_packages, dict) else {}
    )
    n_virtual_sources = sum(
        len(record.get("virtual_sources") or {})
        for record in runtime_packages.values()
        if isinstance(record, dict)
    )
    classified_assets = (
        len(archived_files or {})
        + len(repo_tracked_files or {})
        + len(separately_bundled_files or {})
        + n_virtual_sources
    )
    if not (
        isinstance(closure, dict)
        and set(closure) == top_level
        and closure.get("schema_version") == "operate-backend-runtime-closure-v1"
        and closure.get("release_id") == release_id
        and closure.get("status") == "backend_runtime_closure_complete"
        and closure.get("terminal") is True
        and closure.get("portable") is True
        and re.fullmatch(r"[0-9a-f]{64}", str(closure.get("source_suite_sha256") or ""))
        is not None
        and isinstance(archived_files, dict)
        and file_inventories_valid
        and len(classified_paths)
        == len(archived_source_paths)
        + len(repo_tracked_files)
        + len(separately_bundled_files)
        and separately_bundled_valid
        and isinstance(summary, dict)
        and set(summary) == summary_fields
        and all(type(value) is int and value >= 0 for value in summary.values())
        and summary["n_archived_files"] == len(archived_files)
        and summary["n_repo_tracked_files"] == len(repo_tracked_files)
        and summary["n_separately_bundled_files"]
        == len(separately_bundled_files)
        and summary["n_backend_links"] == len(links)
        and summary["n_external_sources"] == len(manifest.get("external_sources") or {})
        and summary["n_runtime_packages"] == len(manifest.get("runtime_packages") or {})
        and summary["n_unresolved"] == 0
        and summary["n_virtual_sources"] == n_virtual_sources
        and classified_assets
        <= summary["n_source_assets"]
        <= classified_assets + n_external_files
        and identity
        == _canonical_payload_sha256(
            {key: value for key, value in closure.items() if key != "identity_sha256"}
        )
        and closure.get("backend_links") == links
        and closure.get("external_sources") == (manifest.get("external_sources") or {})
        and closure.get("runtime_packages") == (manifest.get("runtime_packages") or {})
        and isinstance(release_manifest, dict)
        and release_manifest.get("release_id") == release_id
        and isinstance(replay, dict)
        and closure.get("source_suite_sha256") == replay.get("source_suite_sha256")
        and isinstance(release_binding, dict)
        and set(binding) == binding_fields
        and set(release_binding) == binding_fields
        and release_binding.get("path") == relative_text
        and binding
        == {
            **release_binding,
            "path": relative_text,
        }
        and binding["sha256"] == expected
        and binding["schema_version"] == closure["schema_version"]
        and binding["identity_sha256"] == identity
        and all(
            binding[field] == summary[field]
            for field in summary_fields & binding_fields
        )
    ):
        raise ValueError("bundle_backend_runtime_closure_binding_invalid")
    archive_roots = {
        Path(str(relative_path)).parts[1]
        for relative_path in (manifest.get("backend_archive_files") or {})
        if Path(str(relative_path)).parts[:1] == ("backends",)
        and len(Path(str(relative_path)).parts) > 2
    }
    if not set(links.values()) <= archive_roots:
        raise ValueError("bundle_backend_runtime_closure_binding_invalid")


def _validate_candidate_closure_binding(
    data_dir: Path,
    manifest: dict[str, Any],
    *,
    release_manifest: dict[str, Any],
) -> None:
    binding = manifest.get("candidate_closure")
    release_binding = release_manifest.get("candidate_closure")
    if binding is None and release_binding is None:
        return
    binding_fields = {
        "path",
        "sha256",
        "schema_version",
        "status",
        "n_independent_candidates",
        "n_terminal_candidates",
        "n_unresolved_candidates",
        "identity_set_sha256",
    }
    if not (
        isinstance(binding, dict)
        and isinstance(release_binding, dict)
        and set(binding) == binding_fields
        and set(release_binding) == binding_fields
    ):
        raise ValueError("bundle_candidate_closure_binding_invalid")

    relative_text = str(binding.get("path") or "")
    relative = Path(relative_text)
    release_path = Path(str(release_binding.get("path") or ""))
    expected = str(binding.get("sha256") or "")
    closure_path = data_dir / relative
    if not (
        relative_text == "candidate_closure.json"
        and not relative.is_absolute()
        and len(relative.parts) == 1
        and relative.as_posix() == relative_text
        and not release_path.is_absolute()
        and ".." not in release_path.parts
        and release_path.as_posix() == relative_text
        and binding == {**release_binding, "path": relative_text}
        and re.fullmatch(r"[0-9a-f]{64}", expected) is not None
        and (manifest.get("files") or {}).get(relative_text) == expected
        and closure_path.is_file()
        and not closure_path.is_symlink()
        and _sha256_file(closure_path) == expected
    ):
        raise ValueError("bundle_candidate_closure_binding_invalid")
    payload = json.loads(closure_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("bundle_candidate_closure_binding_invalid")
    from scripts.finalize_operate_candidate_pool import (  # noqa: PLC0415
        validate_compact_candidate_closure,
    )

    try:
        validate_compact_candidate_closure(payload)
    except ValueError as exc:
        raise ValueError("bundle_candidate_closure_binding_invalid") from exc
    summary = payload["summary"]
    if not (
        binding["schema_version"] == payload["schema_version"]
        and binding["status"] == payload["status"]
        and binding["n_independent_candidates"] == summary["n_independent_candidates"]
        and binding["n_terminal_candidates"] == summary["n_terminal_candidates"]
        and binding["n_unresolved_candidates"] == summary["n_unresolved_candidates"]
        and binding["identity_set_sha256"] == payload["identity_set_sha256"]
    ):
        raise ValueError("bundle_candidate_closure_binding_invalid")


def _validate_candidate_evidence_binding(
    data_dir: Path,
    manifest: dict[str, Any],
) -> None:
    closure_binding = manifest.get("candidate_closure")
    evidence_fields = {
        "candidate_evidence_archive",
        "candidate_evidence_install_root",
        "candidate_evidence_required_files",
        "candidate_evidence_files",
    }
    present = {field for field in evidence_fields if manifest.get(field) is not None}
    if closure_binding is None:
        if present:
            raise ValueError("candidate_evidence_without_closure")
        return
    if not present:
        release_path = data_dir / "release_manifest.json"
        release = (
            json.loads(release_path.read_text(encoding="utf-8"))
            if release_path.is_file() and not release_path.is_symlink()
            else None
        )
        if isinstance(release, dict) and isinstance(
            release.get("formal_runtime_bundle"), dict
        ):
            return
    if present != evidence_fields:
        raise ValueError("candidate_evidence_contract_incomplete")
    closure_path = data_dir / "candidate_closure.json"
    payload = json.loads(closure_path.read_text(encoding="utf-8"))
    inputs = payload.get("inputs") if isinstance(payload, dict) else None
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("candidate_evidence_inputs_missing")
    expected_files: dict[str, str] = {}
    for raw_bindings in inputs.values():
        bindings = raw_bindings if isinstance(raw_bindings, list) else [raw_bindings]
        for binding in bindings:
            if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
                raise ValueError("candidate_evidence_binding_invalid")
            relative_text = str(binding.get("path") or "")
            relative = Path(relative_text)
            digest = str(binding.get("sha256") or "")
            if not (
                relative.parts[:2] == (".hl", "artifacts")
                and not relative.is_absolute()
                and ".." not in relative.parts
                and relative.as_posix() == relative_text
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
                and relative_text not in expected_files
            ):
                raise ValueError("candidate_evidence_binding_invalid")
            expected_files[relative_text] = digest
    archive_name = str(manifest.get("candidate_evidence_archive") or "")
    archive_path = Path(archive_name)
    release_id = str(manifest.get("release_id") or "")
    expected_archive_name = f"{release_id}_candidate_evidence.tar.zst"
    if not (
        re.fullmatch(r"operate_v\d+_\d+_\d+", release_id) is not None
        and archive_name == expected_archive_name
        and not archive_path.is_absolute()
        and manifest.get("candidate_evidence_install_root") == "candidate_evidence"
        and manifest.get("candidate_evidence_required_files")
        == sorted(expected_files)
        and manifest.get("candidate_evidence_files")
        == dict(sorted(expected_files.items()))
        and archive_name in (manifest.get("files") or {})
    ):
        raise ValueError("candidate_evidence_contract_invalid")


def validate_bundle_distribution_contract(
    data_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """Validate hash-bound bundle internals without inspecting local sources."""
    _validate_backend_runtime_closure_binding(data_dir, manifest)
    _validate_backend_archive_file_bindings(data_dir, manifest)
    _validate_backend_license_bindings(data_dir, manifest)
    formal_fields = {
        "formal_evidence_archive",
        "formal_evidence_files",
        "formal_evidence_install_root",
        "formal_evidence_required_files",
    }
    has_bundled_release = "release_manifest.json" in (manifest.get("files") or {})
    runtime_contract_declared = (
        "schema_version" in manifest or "bundle_kind" in manifest
    )
    if runtime_contract_declared and not (
        manifest.get("schema_version") == "operate-runtime-bundle-v2"
        and manifest.get("bundle_kind")
        in {"public_runtime_companion", "private_runtime_companion"}
    ):
        raise ValueError("runtime_bundle_contract_invalid")
    bundled_release = (
        _validate_bundle_formal_evidence_binding(data_dir, manifest)
        if runtime_contract_declared
        or "formal_evidence_archive" in manifest
        or (has_bundled_release and formal_fields & set(manifest))
        else None
    )
    release_manifest_path = data_dir / "release_manifest.json"
    if not release_manifest_path.is_file() or release_manifest_path.is_symlink():
        if manifest.get("candidate_closure") is not None:
            raise ValueError("bundle_release_manifest_missing")
        return
    release_manifest = bundled_release or json.loads(
        release_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(release_manifest, dict):
        raise ValueError("bundle_release_manifest_invalid")
    _validate_candidate_closure_binding(
        data_dir,
        manifest,
        release_manifest=release_manifest,
    )
    _validate_candidate_evidence_binding(data_dir, manifest)


def _validate_external_source_bindings(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    sources = manifest.get("external_sources") or {}
    if not isinstance(sources, dict):
        raise ValueError("bundle_external_sources_invalid")
    bundled_source_assets = _source_asset_file_rows(manifest)
    repo_root = repo_root.resolve()
    for source_id, contract in sources.items():
        if not isinstance(source_id, str) or not isinstance(contract, dict):
            raise ValueError(f"bundle_external_source_invalid:{source_id}")
        delivery = contract.get("delivery")
        required_files = contract.get("required_files")
        metadata = contract.get("metadata") or {}
        if not (
            delivery in {"git_checkout", "upstream_fetch", "user_provided"}
            and isinstance(required_files, dict)
            and required_files
            and isinstance(metadata, dict)
        ):
            raise ValueError(f"bundle_external_source_invalid:{source_id}")
        root_text = str(metadata.get("root") or "")
        root = Path(root_text)
        if root_text and (
            root.is_absolute() or ".." in root.parts or root.as_posix() != root_text
        ):
            raise ValueError(f"bundle_external_source_root_invalid:{source_id}")
        for relative_text, expected in required_files.items():
            relative = Path(str(relative_text))
            if not (
                isinstance(relative_text, str)
                and relative_text
                and not relative.is_absolute()
                and ".." not in relative.parts
                and relative.as_posix() == relative_text
                and re.fullmatch(r"[0-9a-f]{64}", str(expected)) is not None
                and (not root_text or relative == root or root in relative.parents)
            ):
                raise ValueError(f"bundle_external_source_file_invalid:{source_id}")
            path = repo_root / relative
            _reject_existing_parent_symlinks(
                path,
                boundary=repo_root,
                error=f"bundle_external_source_parent_symlink:{source_id}",
            )
            if not path.is_file() or path.is_symlink():
                bundled = bundled_source_assets.get(relative_text)
                if (
                    isinstance(bundled, dict)
                    and bundled.get("delivery") == "bundle"
                    and bundled.get("sha256") == expected
                ):
                    continue
                raise ValueError(
                    f"bundle_external_source_file_missing:{source_id}:{relative_text}"
                )
            if _sha256_file(path) != expected:
                raise ValueError(
                    f"bundle_external_source_hash_mismatch:{source_id}:{relative_text}"
                )
        if delivery != "git_checkout":
            continue
        revision = str(contract.get("revision") or "")
        checkout = repo_root / root
        if not root_text or not (checkout / ".git").exists():
            raise ValueError(f"bundle_external_source_git_missing:{source_id}")
        head = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if head.returncode != 0 or head.stdout.strip() != revision:
            raise ValueError(f"bundle_external_source_git_revision:{source_id}")
        status = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        if status.returncode != 0 or status.stdout:
            raise ValueError(f"bundle_external_source_git_dirty:{source_id}")


def _repo_install_targets(
    data_dir: Path,
    manifest: dict[str, Any],
    *,
    repo_root: Path,
) -> set[Path]:
    repo_root = repo_root.absolute()
    targets: set[Path] = set()
    links = _validated_backend_links(manifest)
    targets.update(repo_root / "works" / str(name) for name in links)
    install_root = str(manifest.get("formal_evidence_install_root") or "")
    if install_root:
        relative = Path(install_root)
        compact_release_root = relative.parts == (
            "release",
            str(manifest.get("release_id") or ""),
        )
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or (len(relative.parts) < 3 and not compact_release_root)
            or relative.parts[0] != "release"
            or relative.as_posix() != install_root
        ):
            raise ValueError("formal_evidence_install_root_invalid")
        targets.add(repo_root / relative)
    for install_path in _source_asset_file_rows(manifest):
        relative = Path(install_path)
        if relative.parts[:2] == ("works", "DynaSchedBench"):
            target = data_dir / "backends" / "dynasched" / Path(*relative.parts[2:])
            _reject_existing_parent_symlinks(
                target,
                boundary=data_dir,
                error="bundle_install_target_parent_symlink",
            )
            continue
        targets.add(repo_root / relative)
    return targets


def _preflight_runtime_install_targets(
    data_dir: Path,
    manifest: dict[str, Any],
    *,
    repo_root: Path = REPO,
) -> None:
    """Validate every repository mutation target before installation starts."""
    repo_root = repo_root.absolute()
    validate_bundle_distribution_contract(data_dir, manifest)
    _validate_external_source_bindings(manifest, repo_root=repo_root)
    _validate_runtime_package_bindings(manifest, repo_root=repo_root)
    _validate_repo_tracked_and_virtual_sources(
        data_dir,
        manifest,
        repo_root=repo_root,
    )
    targets = _repo_install_targets(data_dir, manifest, repo_root=repo_root)
    for target in sorted(targets):
        _reject_existing_parent_symlinks(
            target,
            boundary=repo_root,
            error="bundle_install_target_parent_symlink",
        )


def _remove_install_target(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _remove_owned_directory(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) != identity or not stat.S_ISDIR(
        current.st_mode
    ):
        raise ValueError(f"temporary_directory_identity_changed:{path}")
    shutil.rmtree(path)


@contextmanager
def _repo_file_transaction(targets: set[Path], *, repo_root: Path):
    """Rollback all declared repository file targets when installation fails."""
    ordered = sorted(targets, key=lambda path: (len(path.parts), str(path)))
    roots: list[Path] = []
    for target in ordered:
        if not any(target == root or root in target.parents for root in roots):
            roots.append(target)
    journal = Path(tempfile.mkdtemp(prefix=".operate-install-journal-", dir=repo_root))
    backups: dict[Path, Path | None] = {}
    missing_parents: set[Path] = set()
    try:
        for index, target in enumerate(roots):
            for parent in target.parents:
                if parent == repo_root:
                    break
                if not parent.exists() and not parent.is_symlink():
                    missing_parents.add(parent)
            backup = journal / str(index)
            if target.is_symlink():
                backup.symlink_to(os.readlink(target), target_is_directory=True)
            elif target.is_dir():
                shutil.copytree(
                    target,
                    backup,
                    symlinks=True,
                    copy_function=shutil.copy2,
                )
            elif target.is_file():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup)
            else:
                backups[target] = None
                continue
            backups[target] = backup
        try:
            yield
        except Exception:
            for target in reversed(roots):
                _remove_install_target(target)
                backup = backups.get(target)
                if backup is not None and (backup.exists() or backup.is_symlink()):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    backup.replace(target)
            for parent in sorted(
                missing_parents, key=lambda path: len(path.parts), reverse=True
            ):
                try:
                    parent.rmdir()
                except OSError:
                    pass
            raise
    finally:
        shutil.rmtree(journal, ignore_errors=True)


@contextmanager
def _formal_install_replay_lock(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
):
    """Share an existing formal output's ``.run.lock`` with the replay runner."""
    install_root = str(manifest.get("formal_evidence_install_root") or "")
    if not install_root:
        yield
        return
    target = repo_root / install_root
    if not target.is_dir() or target.is_symlink():
        yield
        return
    import fcntl  # noqa: PLC0415

    handle = (target / ".run.lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise ValueError("bundle_install_replay_locked") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _validate_m5_source_metadata(
    source: Path, rows: dict[str, dict[str, Any]],
) -> None:
    """Keep native M5 lock semantics separate from its three physical inputs."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "source_id": "m5_forecasting",
        "source_url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy",
        "license": "Kaggle competition rules",
        "inventory_environment_id": "InvManagement-v1",
        "package_version": "or-gym==0.5.0",
    }
    if not (
        isinstance(payload, dict)
        and all(payload.get(key) == value for key, value in required.items())
        and payload.get("license_verified") is True
        and payload.get("terms_accepted") is True
        and re.fullmatch(
            r"sha256:[0-9a-f]{64}", str(payload.get("license_or_terms_sha256") or "")
        ) is not None
        and payload.get("files") == {
            path: rows[path]["sha256"]
            for path in _EXACT_SOURCE_ASSET_SPECS["m5"]["paths"]
        }
        and payload.get("orgym_runtime_source") == {
            "commit": "0b18d16e569e2db70e83f09e867b53bdb4b87298",
            "license": "MIT",
        }
    ):
        raise ValueError("m5_source_lock_metadata_invalid")


def install_bundle_source_assets(
    data_dir: Path,
    manifest: dict[str, Any],
    *,
    repo_root: Path = REPO,
) -> int:
    """Restore hash-bound source files without overwriting a conflicting tree."""
    rows = _source_asset_file_rows(manifest)
    if not rows:
        return 0
    repo_root = repo_root.resolve()
    pending: list[tuple[Path, Path, str]] = []
    verified_blobs: dict[Path, str] = {}
    for install_path, row in rows.items():
        source = data_dir / str(row["archive_path"])
        destination = repo_root / install_path
        install = Path(install_path)
        if install.parts[:2] == ("works", "DynaSchedBench"):
            runtime_root = (data_dir / "backends" / "dynasched").resolve()
            works_link = repo_root / "works" / "DynaSchedBench"
            expected_destination = runtime_root / Path(*install.parts[2:])
            if (
                not works_link.is_symlink()
                or works_link.resolve() != runtime_root
                or destination.resolve() != expected_destination.resolve()
            ):
                raise ValueError(
                    f"bundle_source_asset_runtime_link_invalid:{install_path}"
                )
        else:
            try:
                destination.resolve().relative_to(repo_root)
            except ValueError as exc:
                raise ValueError(
                    f"bundle_source_asset_destination_invalid:{install_path}"
                ) from exc
        expected = str(row["sha256"])
        if row.get("delivery") == "upstream_fetch":
            if (
                not destination.is_file()
                or destination.is_symlink()
                or _sha256_file(destination) != expected
            ):
                raise ValueError(
                    f"bundle_source_asset_upstream_fetch_invalid:{install_path}"
                )
            continue
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"bundle_source_asset_missing:{install_path}")
        observed = verified_blobs.get(source)
        if observed is None:
            observed = _sha256_file(source)
            verified_blobs[source] = observed
        if observed != expected:
            raise ValueError(f"bundle_source_asset_hash_mismatch:{install_path}")
        if install_path in _EXACT_SOURCE_ASSET_SPECS["m5"]["metadata"]:
            _validate_m5_source_metadata(source, rows)
        if destination.exists() or destination.is_symlink():
            if not destination.is_file() or destination.is_symlink():
                raise ValueError(f"bundle_source_asset_target_invalid:{install_path}")
            if _sha256_file(destination) != expected:
                raise ValueError(f"bundle_source_asset_target_conflict:{install_path}")
            continue
        if install.parts[:2] == ("works", "DynaSchedBench"):
            raise ValueError(f"bundle_source_asset_runtime_missing:{install_path}")
        pending.append((source, destination, expected))
    if not pending:
        return 0
    staging = Path(tempfile.mkdtemp(prefix=".source-assets-", dir=repo_root))
    try:
        for source, destination, expected in pending:
            relative = destination.relative_to(repo_root)
            staged = staging / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            if _sha256_file(staged) != expected:
                raise ValueError(
                    f"bundle_source_asset_staging_hash_mismatch:{relative.as_posix()}"
                )
        for _source, destination, _expected in pending:
            relative = destination.relative_to(repo_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                destination.parent.resolve().relative_to(repo_root)
            except ValueError as exc:
                raise ValueError(
                    f"bundle_source_asset_destination_invalid:{relative.as_posix()}"
                ) from exc
            (staging / relative).replace(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return len(pending)


def _extract_backend_archive(
    data_dir: Path, manifest: dict[str, Any], *, force: bool = False
) -> bool:
    archive_name = str(manifest.get("backend_archive") or "")
    if not archive_name:
        return False
    archive_path = Path(archive_name)
    if archive_path.is_absolute() or ".." in archive_path.parts:
        raise ValueError(f"bundle_backend_archive_invalid:{archive_name}")
    archive = data_dir / archive_name
    archive_sha256 = str(manifest["files"][archive_name])
    marker = data_dir / ".backends_archive.sha256"
    backend_root = data_dir / "backends"
    if (
        backend_root.is_dir()
        and marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == archive_sha256
    ):
        return False
    if backend_root.exists() and not force:
        raise ValueError(
            f"backend_target_exists:{backend_root}; use --force-extract to replace it"
        )
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("zstd_not_found: install zstd, then rerun the command")

    staging = data_dir / ".backends.extracting"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    process = subprocess.Popen([zstd, "-dc", str(archive)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
            for member in stream:
                relative = Path(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0] != "backends"
                    or not (member.isdir() or member.isfile())
                ):
                    raise ValueError(f"unsafe_backend_archive_member:{member.name}")
                destination = staging / relative
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise ValueError(f"backend_archive_member_unreadable:{member.name}")
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
    except (OSError, tarfile.TarError, ValueError):
        process.terminate()
        process.wait()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        process.stdout.close()
    if process.wait() != 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("backend_archive_decompression_failed")
    extracted = staging / "backends"
    if not extracted.is_dir():
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError("backend_archive_root_missing")
    if backend_root.exists():
        shutil.rmtree(backend_root)
    extracted.replace(backend_root)
    staging.rmdir()
    marker.write_text(archive_sha256 + "\n", encoding="utf-8")
    return True


def _extract_formal_evidence_archive(
    data_dir: Path,
    manifest: dict[str, Any],
    *,
    repo_root: Path = REPO,
    force: bool = False,
) -> bool:
    repo_root = repo_root.resolve()
    validate_runtime_bundle_compatibility(
        data_dir,
        manifest,
        repo_root=repo_root,
    )
    archive_name = str(manifest["formal_evidence_archive"])
    archive_relative = Path(archive_name)
    install_relative = Path(str(manifest.get("formal_evidence_install_root") or ""))
    bundled_release = json.loads(
        (data_dir / "release_manifest.json").read_text(encoding="utf-8")
    )
    compact_formal_release = isinstance(
        bundled_release.get("formal_runtime_bundle"), dict
    )
    if (
        archive_relative.is_absolute()
        or ".." in archive_relative.parts
        or install_relative.is_absolute()
        or ".." in install_relative.parts
        or (
            len(install_relative.parts) != 2
            if compact_formal_release
            else len(install_relative.parts) < 3
        )
        or install_relative.parts[0] != "release"
    ):
        raise ValueError("formal_evidence_install_root_invalid")
    archive = data_dir / archive_relative
    archive_sha256 = str(manifest["files"][archive_name])
    target = repo_root / install_relative
    marker = target / ".archive.sha256"
    if compact_formal_release and target.is_symlink():
        raise ValueError("formal_evidence_install_root_symlink")
    formal_files = {
        Path(str(relative)): str(digest)
        for relative, digest in (manifest.get("formal_evidence_files") or {}).items()
    }
    required = [
        Path(str(value)) for value in manifest["formal_evidence_required_files"]
    ]
    if any(path.is_absolute() or ".." in path.parts for path in required):
        raise ValueError("formal_evidence_required_file_invalid")
    expected_files = {
        install_relative / relative: expected
        for relative, expected in formal_files.items()
    }
    for relative in (install_relative, *expected_files):
        destination = repo_root / relative
        try:
            destination.parent.resolve().relative_to(repo_root)
        except ValueError as exc:
            raise ValueError(
                f"formal_evidence_destination_escapes_repo:{relative.as_posix()}"
            ) from exc
    files_match = (
        target.is_dir()
        and all(
            (repo_root / path).is_file()
            and not (repo_root / path).is_symlink()
            and _sha256_file(repo_root / path) == expected
            for path, expected in expected_files.items()
        )
    )
    if files_match and (
        compact_formal_release
        or (
            marker.is_file()
            and marker.read_text(encoding="utf-8").strip() == archive_sha256
        )
    ):
        return False
    if compact_formal_release:
        conflicts = [
            relative.as_posix()
            for relative, expected in expected_files.items()
            if (repo_root / relative).exists()
            or (repo_root / relative).is_symlink()
            if not (
                (repo_root / relative).is_file()
                and not (repo_root / relative).is_symlink()
                and _sha256_file(repo_root / relative) == expected
            )
        ]
    else:
        conflicts = (
            [install_relative.as_posix()]
            if target.exists() or target.is_symlink()
            else []
        )
    if conflicts and not force:
        raise ValueError(
            "formal_evidence_target_exists:"
            f"{conflicts}; use --force-extract to replace it"
        )
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("zstd_not_found: install zstd, then rerun the command")

    _reject_existing_parent_symlinks(
        target,
        boundary=repo_root,
        error="formal_evidence_destination_parent_symlink",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.extracting-", dir=target.parent)
    )
    staging_stat = staging.lstat()
    staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
    process = subprocess.Popen([zstd, "-dc", str(archive)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    extracted_files: set[Path] = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
            for member in stream:
                member_path = Path(member.name)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or not member_path.parts
                    or not (member.isdir() or member.isfile())
                ):
                    raise ValueError(f"unsafe_formal_evidence_member:{member.name}")
                destination = staging / member_path
                if member.isdir():
                    if not any(
                        member_path in expected.parents for expected in expected_files
                    ):
                        raise ValueError(
                            f"unexpected_formal_evidence_member:{member.name}"
                        )
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if member_path not in expected_files or member_path in extracted_files:
                    raise ValueError(f"unexpected_formal_evidence_member:{member.name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise ValueError(f"formal_evidence_member_unreadable:{member.name}")
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted_files.add(member_path)
    except (OSError, tarfile.TarError, ValueError):
        process.terminate()
        process.wait()
        _remove_owned_directory(staging, staging_identity)
        raise
    finally:
        process.stdout.close()
    if process.wait() != 0:
        _remove_owned_directory(staging, staging_identity)
        raise RuntimeError("formal_evidence_archive_decompression_failed")
    missing = [
        path.as_posix()
        for path in expected_files
        if path not in extracted_files or not (staging / path).is_file()
    ]
    if missing:
        _remove_owned_directory(staging, staging_identity)
        raise ValueError(f"formal_evidence_required_files_missing:{missing}")
    mismatched = [
        path.as_posix()
        for path, expected in expected_files.items()
        if _sha256_file(staging / path) != expected
    ]
    if mismatched:
        _remove_owned_directory(staging, staging_identity)
        raise ValueError(f"formal_evidence_hash_mismatch:{mismatched}")
    if compact_formal_release:
        install_units = [
            relative
            for relative, expected in expected_files.items()
            if not (
                (repo_root / relative).is_file()
                and not (repo_root / relative).is_symlink()
                and _sha256_file(repo_root / relative) == expected
            )
        ]
    else:
        install_units = [install_relative]
    backup_root = staging / ".backup"
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for relative in install_units:
            destination = repo_root / relative
            if not destination.exists() and not destination.is_symlink():
                continue
            backup = backup_root / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            destination.replace(backup)
            backups[destination] = backup
        for relative in install_units:
            source = staging / relative
            destination = repo_root / relative
            if not source.exists():
                raise ValueError(
                    f"formal_evidence_staged_target_missing:{relative.as_posix()}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            installed.append(destination)

        if compact_formal_release:
            from scripts.verify_release_integrity import (  # noqa: PLC0415
                _formal_runtime_bundle_valid,
            )

            release_root = repo_root / "release" / manifest["release_id"]
            release_manifest = json.loads(
                (release_root / "manifest.json").read_text(encoding="utf-8")
            )
            core = json.loads(
                (release_root / "core_suite.json").read_text(encoding="utf-8")
            )
            core_rows = core.get("scenarios") if isinstance(core, dict) else None
            if not (
                isinstance(core_rows, list)
                and all(isinstance(row, dict) for row in core_rows)
                and _formal_runtime_bundle_valid(
                    release_root,
                    release_manifest,
                    core,
                    list(core_rows),
                )
            ):
                raise ValueError("formal_runtime_bundle_integrity_invalid")
            if manifest.get("formal_result_trees") is not None:
                from scripts.build_operate_bundle import (  # noqa: PLC0415
                    _formal_result_tree_files,
                )

                result_files, result_trees = _formal_result_tree_files(
                    repo_root=repo_root,
                    release_root=release_root,
                    release_manifest=release_manifest,
                )
                declared_trees = manifest.get("formal_result_trees")
                evidence_files = manifest.get("formal_evidence_files") or {}
                if not (
                    result_trees == declared_trees
                    and result_files
                    and all(
                        evidence_files.get(path) == digest
                        for path, digest in result_files.items()
                    )
                ):
                    raise ValueError("formal_result_tree_post_install_invalid")
        if manifest.get("backend_archive"):
            from scripts.batch_llm_eval import (  # noqa: PLC0415
                resolve_formal_manifest_slice,
            )

            resolve_formal_manifest_slice(
                repo_root / "release" / manifest["release_id"] / "manifest.json",
                repo_root=repo_root,
            )
        if not compact_formal_release:
            marker.write_text(archive_sha256 + "\n", encoding="utf-8")
    except Exception as exc:
        for path in reversed(installed):
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        for destination, backup in backups.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup.replace(destination)
        raise ValueError("formal_evidence_post_install_validation_failed") from exc
    finally:
        if staging.exists():
            _remove_owned_directory(staging, staging_identity)
    return True


def _extract_candidate_evidence_archive(
    data_dir: Path,
    manifest: dict[str, Any],
    *,
    force: bool = False,
) -> bool:
    """Restore the compact exhaustion inputs inside the installed bundle."""
    if manifest.get("candidate_closure") is None:
        return False
    _validate_candidate_evidence_binding(data_dir, manifest)
    if manifest.get("candidate_evidence_archive") is None:
        return False
    archive_name = str(manifest["candidate_evidence_archive"])
    archive = data_dir / archive_name
    install_root = Path(str(manifest["candidate_evidence_install_root"]))
    target = data_dir / install_root
    expected_files = {
        install_root / Path(relative): str(digest)
        for relative, digest in manifest["candidate_evidence_files"].items()
    }
    if target.is_dir() and all(
        (data_dir / relative).is_file()
        and not (data_dir / relative).is_symlink()
        and _sha256_file(data_dir / relative) == digest
        for relative, digest in expected_files.items()
    ):
        return False
    if (target.exists() or target.is_symlink()) and not force:
        raise ValueError("candidate_evidence_target_exists")
    zstd = shutil.which("zstd")
    if zstd is None:
        raise RuntimeError("zstd_not_found: install zstd, then rerun the command")
    staging = data_dir / ".candidate_evidence.extracting"
    if staging.exists() or staging.is_symlink():
        _remove_install_target(staging)
    staging.mkdir()
    process = subprocess.Popen([zstd, "-dc", str(archive)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    extracted: set[Path] = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as stream:
            for member in stream:
                relative = Path(member.name)
                if member.isdir():
                    continue
                if (
                    relative not in expected_files
                    or relative in extracted
                    or not member.isfile()
                ):
                    raise ValueError(
                        f"unexpected_candidate_evidence_member:{member.name}"
                    )
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = stream.extractfile(member)
                if source is None:
                    raise ValueError(
                        f"candidate_evidence_member_unreadable:{member.name}"
                    )
                with source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted.add(relative)
    except (OSError, tarfile.TarError, ValueError):
        process.terminate()
        process.wait()
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        process.stdout.close()
    if process.wait() != 0:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError("candidate_evidence_archive_decompression_failed")
    if extracted != set(expected_files) or any(
        _sha256_file(staging / relative) != digest
        for relative, digest in expected_files.items()
    ):
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError("candidate_evidence_archive_content_invalid")
    if target.exists() or target.is_symlink():
        _remove_install_target(target)
    (staging / install_root).replace(target)
    shutil.rmtree(staging, ignore_errors=True)
    return True


def link_bundle_backends(
    data_dir: Path, manifest: dict[str, Any], *, repo_root: Path = REPO
) -> int:
    links = _validated_backend_links(manifest)
    validated: list[tuple[str, str]] = []
    for works_name, data_name in links.items():
        if not all(
            isinstance(value, str)
            and value
            and value == value.strip()
            and value not in {".", ".."}
            and "/" not in value
            and "\\" not in value
            and "\0" not in value
            and not Path(value).is_absolute()
            and len(Path(value).parts) == 1
            for value in (works_name, data_name)
        ):
            raise ValueError(
                f"bundle_backend_link_invalid:{works_name!r}:{data_name!r}"
            )
        validated.append((works_name, data_name))
    works = repo_root / "works"
    works.mkdir(parents=True, exist_ok=True)
    linked = 0
    for works_name, data_name in validated:
        source = data_dir / "backends" / data_name
        destination = works / works_name
        if not source.is_dir():
            continue
        if destination.is_symlink():
            current_target = destination.resolve(strict=False)
            bundle_backends = (data_dir / "backends").resolve()
            try:
                current_target.relative_to(bundle_backends)
            except ValueError as exc:
                raise ValueError(
                    f"bundle_backend_link_not_operate_owned:{works_name}"
                ) from exc
            if current_target == source.resolve():
                continue
            destination.unlink()
        elif destination.exists():
            raise ValueError(f"bundle_backend_link_target_conflict:{works_name}")
        destination.symlink_to(source.resolve())
        linked += 1
    return linked


def _canonical_payload_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _validate_runtime_package_bindings(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    """Verify package locks already installed by the repository ``uv sync``."""
    packages = manifest.get("runtime_packages") or {}
    if not isinstance(packages, dict):
        raise ValueError("runtime_packages_invalid")
    if not packages:
        return
    uv_lock = repo_root / "uv.lock"
    if not uv_lock.is_file() or uv_lock.is_symlink():
        raise ValueError("runtime_packages_uv_lock_missing")
    uv_lock_sha256 = _sha256_file(uv_lock)
    for package, contract in packages.items():
        if not isinstance(package, str) or not isinstance(contract, dict):
            raise ValueError(f"runtime_package_contract_invalid:{package}")
        allowed_fields = {
            "backend_kinds",
            "lock_entries",
            "lock_entries_sha256",
            "uv_lock_sha256",
            "virtual_sources",
        }
        backend_kinds = contract.get("backend_kinds")
        entries = contract.get("lock_entries")
        if not (
            set(contract) <= allowed_fields
            and set(contract) >= allowed_fields - {"virtual_sources"}
            and isinstance(backend_kinds, list)
            and backend_kinds == sorted(set(backend_kinds))
            and backend_kinds
            and all(isinstance(kind, str) and kind for kind in backend_kinds)
            and isinstance(entries, list)
            and entries
            and contract.get("lock_entries_sha256")
            == _canonical_payload_sha256(entries)
        ):
            raise ValueError(f"runtime_package_contract_invalid:{package}")
        if contract.get("uv_lock_sha256") != uv_lock_sha256:
            raise ValueError(f"runtime_packages_uv_lock_mismatch:{package}")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"runtime_package_lock_entry_invalid:{package}")
            identity = entry.get("identity_sha256")
            identity_payload = {
                key: value for key, value in entry.items() if key != "identity_sha256"
            }
            if identity != _canonical_payload_sha256(identity_payload):
                raise ValueError(f"runtime_package_lock_entry_invalid:{package}")
        virtual_sources = contract.get("virtual_sources") or {}
        if not isinstance(virtual_sources, dict) or any(
            not isinstance(source, str)
            or "://" not in source
            or re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None
            for source, digest in virtual_sources.items()
        ):
            raise ValueError(f"runtime_package_virtual_sources_invalid:{package}")


def validate_runtime_package_contract(
    manifest: dict[str, Any],
    *,
    repo_root: Path = REPO,
) -> None:
    """Validate release package bindings against the selected repository lock."""
    _validate_runtime_package_bindings(manifest, repo_root=repo_root)


def _validate_repo_tracked_and_virtual_sources(
    data_dir: Path,
    manifest: dict[str, Any],
    *,
    repo_root: Path,
) -> None:
    if "release_manifest.json" not in (manifest.get("files") or {}):
        return
    release_manifest = json.loads(
        (data_dir / "release_manifest.json").read_text(encoding="utf-8")
    )
    replay = release_manifest.get("protocol21_replay")
    if not isinstance(replay, dict):
        raise ValueError("bundle_repo_tracked_source_suite_binding_missing")
    relative_text = str(replay.get("source_suite") or "")
    relative = Path(relative_text)
    suite_path = repo_root / relative
    if not (
        relative_text
        and not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == relative_text
        and suite_path.is_file()
        and not suite_path.is_symlink()
        and _sha256_file(suite_path) == replay.get("source_suite_sha256")
    ):
        raise ValueError("bundle_repo_tracked_source_suite_binding_invalid")
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    rows = suite.get("scenarios") if isinstance(suite, dict) else None
    if not isinstance(rows, list):
        raise ValueError("bundle_repo_tracked_source_suite_binding_invalid")
    closure_binding = manifest.get("backend_runtime_closure") or {}
    closure_path = data_dir / str(closure_binding.get("path") or "")
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    repo_tracked_files = closure.get("repo_tracked_files")
    if not isinstance(repo_tracked_files, dict):
        raise ValueError("bundle_repo_tracked_source_asset_invalid")
    observed_repo_tracked: set[str] = set()
    virtual_sources: dict[str, str] = {}
    for row in rows:
        ledger = row.get("case_ledger") if isinstance(row, dict) else None
        physical_lock = (
            ledger.get("physical_source_lock") if isinstance(ledger, dict) else None
        )
        assets = (
            physical_lock.get("required_source_assets")
            if isinstance(physical_lock, dict)
            else None
        )
        if not isinstance(assets, list):
            continue
        for asset in assets:
            if not isinstance(asset, dict):
                raise ValueError("bundle_repo_tracked_source_asset_invalid")
            source = str(asset.get("declared_path") or "")
            expected = str(asset.get("sha256") or "")
            if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                raise ValueError("bundle_repo_tracked_source_asset_invalid")
            if "://" in source:
                previous = virtual_sources.setdefault(source, expected)
                if previous != expected:
                    raise ValueError("bundle_virtual_source_binding_conflict")
                continue
            identity = repo_tracked_files.get(source)
            if identity is None:
                continue
            if not isinstance(identity, dict) or identity.get("sha256") != expected:
                raise ValueError("bundle_repo_tracked_source_asset_invalid")
            path = repo_root / source
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"bundle_repo_tracked_source_missing:{source}")
            if _sha256_file(path) != expected:
                raise ValueError(f"bundle_repo_tracked_source_hash_mismatch:{source}")
            observed_repo_tracked.add(source)
    if observed_repo_tracked != set(repo_tracked_files):
        raise ValueError("bundle_repo_tracked_source_asset_invalid")
    package_virtual_sources: dict[str, str] = {}
    for package in (manifest.get("runtime_packages") or {}).values():
        if not isinstance(package, dict):
            continue
        for source, expected in (package.get("virtual_sources") or {}).items():
            previous = package_virtual_sources.setdefault(source, expected)
            if previous != expected:
                raise ValueError("bundle_virtual_source_binding_conflict")
    if package_virtual_sources != virtual_sources:
        raise ValueError("bundle_virtual_source_binding_mismatch")


def _clusterdata_git_value(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verify_clusterdata_checkout(checkout: Path) -> dict[str, Any]:
    if not checkout.is_dir():
        raise ValueError("clusterdata_checkout_missing")
    commit = _clusterdata_git_value(checkout, "rev-parse", "HEAD")
    if commit != CLUSTERDATA_EXPECTED_COMMIT:
        raise ValueError("clusterdata_commit_mismatch")
    remote = _clusterdata_git_value(checkout, "remote", "get-url", "origin")
    if remote.rstrip("/") != CLUSTERDATA_URL.rstrip("/"):
        raise ValueError("clusterdata_remote_mismatch")
    if _clusterdata_git_value(
        checkout, "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("clusterdata_tracked_worktree_dirty")
    observed: dict[str, str] = {}
    for relative, expected in CLUSTERDATA_EXPECTED_ASSETS.items():
        path = checkout / relative
        if not path.is_file():
            raise ValueError(f"clusterdata_asset_missing:{relative}")
        digest = _sha256_file(path)
        if digest != expected:
            raise ValueError(f"clusterdata_asset_hash_mismatch:{relative}")
        observed[relative] = digest
    return {
        "status": "verified",
        "checkout": str(checkout),
        "repository": CLUSTERDATA_URL,
        "observed_commit": commit,
        "observed_remote": remote,
        "tracked_worktree_clean": True,
        "asset_sha256": observed,
    }


def _fetch_clusterdata_checkout(checkout: Path) -> dict[str, Any]:
    if checkout.exists():
        return _verify_clusterdata_checkout(checkout)
    checkout.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".clusterdata-fetch-", dir=checkout.parent)
    )
    staging = staging_root / "clusterdata"
    try:
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                CLUSTERDATA_URL,
                str(staging),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(staging),
                "sparse-checkout",
                "init",
                "--no-cone",
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(staging),
                "sparse-checkout",
                "set",
                "--no-cone",
                *CLUSTERDATA_SPARSE_PATHS,
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(staging),
                "fetch",
                "--depth=1",
                "origin",
                CLUSTERDATA_EXPECTED_COMMIT,
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(staging),
                "checkout",
                "--detach",
                CLUSTERDATA_EXPECTED_COMMIT,
            ],
            check=True,
        )
        _verify_clusterdata_checkout(staging)
        if checkout.exists():
            raise ValueError("clusterdata_checkout_appeared_during_fetch")
        staging.replace(checkout)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return _verify_clusterdata_checkout(checkout)


def ensure_clusterdata_sources(repo_root: Path = REPO) -> dict[str, Any]:
    """Fetch or verify the exact Alibaba source checkout used at runtime."""
    return _fetch_clusterdata_checkout(repo_root / "works" / "clusterdata")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"HF dataset repo id (default: {DEFAULT_REPO_ID})",
    )
    ap.add_argument(
        "--revision",
        default=None,
        help=(
            "optional exact 40-64 character dataset commit SHA; when omitted, "
            "the current public revision is resolved once and then pinned"
        ),
    )
    ap.add_argument(
        "--data-dir",
        default=str(DATA),
        help=(
            "local target dir (default: repo operate_data/ install root; "
            "release identity comes from the bundle manifest)"
        ),
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="check repo exists without downloading"
    )
    ap.add_argument(
        "--download-only",
        action="store_true",
        help="download and verify, but do not extract or link backends",
    )
    ap.add_argument(
        "--force-extract",
        action="store_true",
        help="replace an existing backend directory when its marker is stale",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    try:
        _validate_hf_binding(repo_id=args.repo_id, revision=args.revision)
        validate_bundle_download_target(
            data_dir,
            repo_root=REPO,
            require_within_repo=not args.download_only,
        )
    except ValueError as exc:
        print(f"FATAL: invalid bundle target: {exc}", file=sys.stderr)
        return 1
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token  # noqa: PLC0415
        except ImportError:
            get_token = None  # type: ignore[assignment]
        token = get_token() if get_token is not None else None

    from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415

    api = HfApi(token=token)

    try:
        info = api.dataset_info(args.repo_id, revision=args.revision)
        if info.private is not False:
            raise ValueError("bundle_dataset_visibility_not_public")
        revision = str(info.sha)
        _validate_hf_binding(repo_id=args.repo_id, revision=revision)
        if args.revision is not None and revision != args.revision:
            raise ValueError("bundle_dataset_revision_mismatch")
        print(f"repo: {args.repo_id} (last modified {info.last_modified})")
        print(f"resolved revision: {revision}")
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: could not reach dataset {args.repo_id}: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("DRY-RUN: repo reachable, would download ->", data_dir)
        print("DRY-RUN: runtime bindings would be validated from the bundle manifest")
        return 0

    def validate_snapshot(
        snapshot: Path,
        candidate_manifest: dict[str, Any],
    ) -> None:
        if args.download_only:
            validate_bundle_distribution_contract(snapshot, candidate_manifest)
            return
        validate_runtime_bundle_compatibility(
            snapshot,
            candidate_manifest,
            repo_root=REPO,
            verify_implementation=not bool(candidate_manifest.get("backend_archive")),
        )

    install_state: dict[str, Any] = {}

    def install_snapshot(
        installed_dir: Path,
        candidate_manifest: dict[str, Any],
    ) -> None:
        if args.download_only:
            return
        has_runtime = bool(candidate_manifest.get("backend_archive"))
        if has_runtime:
            extracted = _extract_backend_archive(
                installed_dir,
                candidate_manifest,
                force=args.force_extract,
            )
            validate_runtime_bundle_compatibility(
                installed_dir,
                candidate_manifest,
                repo_root=REPO,
            )
            linked = link_bundle_backends(installed_dir, candidate_manifest)
            source_assets_installed = install_bundle_source_assets(
                installed_dir,
                candidate_manifest,
                repo_root=REPO,
            )
        else:
            extracted = False
            source_assets_installed = 0
            linked = 0
        candidate_evidence_extracted = _extract_candidate_evidence_archive(
            installed_dir,
            candidate_manifest,
            force=args.force_extract,
        )
        evidence_extracted = _extract_formal_evidence_archive(
            installed_dir,
            candidate_manifest,
            repo_root=REPO,
            force=args.force_extract,
        )
        install_state.update(
            extracted=extracted,
            linked=linked,
            source_assets_installed=source_assets_installed,
            candidate_evidence_extracted=candidate_evidence_extracted,
            evidence_extracted=evidence_extracted,
        )

    print(f"downloading {args.repo_id}@{revision} -> staging ...")
    try:
        manifest = _download_verified_bundle(
            repo_id=args.repo_id,
            revision=revision,
            data_dir=data_dir,
            token=token,
            snapshot_download_fn=snapshot_download,
            repo_root=REPO,
            require_within_repo=not args.download_only,
            preflight_runtime_install=not args.download_only,
            validate_snapshot_fn=validate_snapshot,
            after_install_fn=None if args.download_only else install_snapshot,
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"FATAL: verified bundle download/install failed: {exc}", file=sys.stderr)
        return 1
    print(f"\nDONE: {data_dir} ({revision})")
    print(
        f"  verified {manifest.get('n_scenarios', '?')} scenarios, "
        f"{manifest.get('n_files', '?')} tracked files"
    )
    has_runtime_archive = bool(manifest.get("backend_archive"))
    if not args.download_only:
        has_formal_evidence_archive = bool(manifest.get("formal_evidence_archive"))
        extracted = bool(install_state["extracted"])
        source_assets_installed = int(install_state["source_assets_installed"])
        linked = int(install_state["linked"])
        candidate_evidence_extracted = bool(
            install_state["candidate_evidence_extracted"]
        )
        evidence_extracted = bool(install_state["evidence_extracted"])
        backend_status = (
            ("extracted" if extracted else "already current")
            if has_runtime_archive
            else "not included in runtime companion"
        )
        print(f"  backends: {backend_status}")
        if has_runtime_archive:
            print(f"  source assets installed: {source_assets_installed}")
        evidence_status = (
            ("installed" if evidence_extracted else "already current")
            if has_formal_evidence_archive
            else "not included in runtime companion"
        )
        print(f"  formal evidence: {evidence_status}")
        if manifest.get("candidate_evidence_archive"):
            candidate_status = (
                "installed" if candidate_evidence_extracted else "already current"
            )
            print(f"  candidate closure evidence: {candidate_status}")
        print(f"  works/ links created or refreshed: {linked}")
        if not has_runtime_archive:
            print(
                "  runtime companion: native backend assets are not included; "
                "no backend checkout was installed"
            )
    elif has_runtime_archive:
        print(
            "  runtime compatibility: deferred until backend installation "
            "(download-only)"
        )
    release_id = str(manifest.get("release_id") or "")
    if release_id:
        print(
            f"  ready: python scripts/verify_release_integrity.py release/{release_id}"
        )
    else:
        print("  ready: bundle installed; verify its declared release before testing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
