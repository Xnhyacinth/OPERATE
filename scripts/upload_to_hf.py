#!/usr/bin/env python3
"""Upload an explicitly selected manifest-backed eval bundle to Hugging Face.

Only files tracked by ``MANIFEST.json`` are uploaded, so locally extracted
backends are not duplicated. Bundle and destination are required arguments to
prevent a stale historical release default from receiving current artifacts.

Auth: reads ``HF_TOKEN`` from env (never hardcodes). Set it via
``export HF_TOKEN=hf_...`` or ``huggingface-cli login``.

Manual trigger only — this script is never auto-run by setup_eval_env.sh.

Usage:
    python scripts/upload_to_hf.py --private --repo-id Xnhyacinth/OPERATE --data-dir data_operate_v058
    python scripts/upload_to_hf.py --repo-id OWNER/DATASET --data-dir BUNDLE --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _distribution_receipt_payload(
    *,
    bundle_dir: Path,
    manifest: dict[str, object],
    repo_id: str,
    revision: str,
) -> dict[str, object]:
    """Bind a remotely verified private CAS revision without self-reference."""

    manifest_path = bundle_dir / "MANIFEST.json"
    observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    trees = manifest.get("formal_result_trees")
    archive = manifest.get("formal_evidence_archive")
    files = manifest.get("files")
    if not (
        observed == manifest
        and manifest.get("hf_repo_id") == repo_id
        and manifest.get("visibility") == "private"
        and re.fullmatch(r"[0-9a-f]{40,64}", revision) is not None
        and isinstance(manifest.get("release_id"), str)
        and re.fullmatch(r"operate_v\d+_\d+_\d+", str(manifest["release_id"]))
        is not None
        and isinstance(manifest.get("release_manifest_sha256"), str)
        and re.fullmatch(
            r"[0-9a-f]{64}", str(manifest["release_manifest_sha256"])
        )
        is not None
        and isinstance(archive, str)
        and isinstance(files, dict)
        and isinstance(files.get(archive), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(files[archive])) is not None
        and isinstance(trees, dict)
        and set(trees) == {"logical_persistent", "realtime_persistent"}
    ):
        raise ValueError("formal_distribution_receipt_input_invalid")
    roots: dict[str, str] = {}
    for mode, contract in trees.items():
        binding = contract.get("binding") if isinstance(contract, dict) else None
        tree_files = contract.get("files") if isinstance(contract, dict) else None
        root = binding.get("tree_root_sha256") if isinstance(binding, dict) else None
        if not (
            isinstance(root, str)
            and re.fullmatch(r"[0-9a-f]{64}", root) is not None
            and isinstance(tree_files, dict)
            and tree_files
        ):
            raise ValueError(f"formal_distribution_result_tree_invalid:{mode}")
        roots[mode] = root
    payload: dict[str, object] = {
        "schema_version": "operate-formal-distribution-receipt-v1",
        "release_id": manifest["release_id"],
        "hf_repo_id": repo_id,
        "visibility": "private",
        "revision": revision,
        "verification": "private_cas_exact_snapshot_v1",
        "bundle_manifest_sha256": _sha256(manifest_path),
        "release_manifest_sha256": manifest["release_manifest_sha256"],
        "formal_evidence_archive": archive,
        "formal_evidence_archive_sha256": files[archive],
        "formal_result_tree_roots": roots,
    }
    payload["receipt_sha256"] = _canonical_sha256(payload)
    return payload


def _write_distribution_receipt(path: Path, payload: dict[str, object]) -> None:
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
        temporary.unlink(missing_ok=True)


def _manifest_paths(manifest: dict[str, object]) -> list[str]:
    files = manifest.get("files") or {}
    if isinstance(files, dict):
        paths = list(files)
    elif isinstance(files, list):
        paths = files
    else:
        raise ValueError("manifest files must be an object or list")
    validated: list[str] = []
    for path in paths:
        if not isinstance(path, str):
            raise ValueError(f"manifest file path invalid:{path!r}")
        relative = PurePosixPath(path)
        if (
            not path
            or path != path.strip()
            or "\0" in path
            or "\\" in path
            or any(character in path for character in "*?[]")
            or relative.is_absolute()
            or not relative.parts
            or any(part in {".", "..", ".cache"} for part in relative.parts)
            or relative.as_posix() != path
            or path in {"MANIFEST.json", ".gitattributes"}
        ):
            raise ValueError(f"manifest file path invalid:{path!r}")
        validated.append(path)
    if len(validated) != len(set(validated)):
        raise ValueError("manifest file paths contain duplicates")
    return sorted(validated)


def _validate_publishable_bundle(
    data_dir: Path,
    *,
    repo_root: Path = REPO,
) -> dict[str, object]:
    """Run the complete immutable runtime-bundle preflight."""
    from scripts.build_operate_bundle import (  # noqa: PLC0415
        validate_bundle_archives,
    )
    from scripts.download_from_hf import (  # noqa: PLC0415
        validate_bundle_distribution_contract,
        validate_runtime_bundle_compatibility,
        validate_runtime_package_contract,
        verify_manifest,
    )

    manifest = verify_manifest(data_dir)
    _manifest_paths(manifest)
    required_runtime_fields = {
        "backend_archive": str,
        "backend_archive_files": dict,
        "backend_runtime_closure": dict,
        "external_sources": dict,
        "runtime_packages": dict,
    }
    if any(
        not isinstance(manifest.get(field), expected_type)
        or not manifest.get(field)
        for field, expected_type in required_runtime_fields.items()
    ):
        raise ValueError("publishable_runtime_companion_incomplete")
    validate_bundle_distribution_contract(data_dir, manifest)
    validate_runtime_package_contract(manifest, repo_root=repo_root)
    validate_runtime_bundle_compatibility(
        data_dir,
        manifest,
        repo_root=repo_root,
        require_canonical_release_manifest=False,
    )
    validate_bundle_archives(data_dir, manifest)
    return manifest


def _ensure_private_repository(*, api: Any, repo_id: str, observed: Any) -> Any:
    if observed.private is not True:
        api.update_repo_settings(
            repo_id=repo_id,
            repo_type="dataset",
            private=True,
        )
        observed = api.dataset_info(repo_id, revision="main")
    if observed.private is not True:
        raise RuntimeError("repository_visibility_not_private")
    return observed


def _download_command(repo_id: str, revision: str) -> str:
    return (
        f"python scripts/download_from_hf.py --repo-id {repo_id} "
        f"--revision {revision} --data-dir data_operate_v058"
    )


def _verify_remote_replacement(*, local_bundle: Path, remote_snapshot: Path) -> None:
    manifest = json.loads((local_bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object")
    expected = {"MANIFEST.json", *_manifest_paths(manifest)}
    observed = {
        path.relative_to(remote_snapshot).as_posix()
        for path in remote_snapshot.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(remote_snapshot).parts
        and path.relative_to(remote_snapshot).as_posix() != ".gitattributes"
    }
    if observed != expected:
        stale = sorted(observed - expected)
        missing = sorted(expected - observed)
        raise ValueError(f"remote_snapshot_not_exact:stale={stale}:missing={missing}")
    if (remote_snapshot / "MANIFEST.json").read_bytes() != (
        local_bundle / "MANIFEST.json"
    ).read_bytes():
        raise ValueError("remote_manifest_bytes_mismatch")


def _verify_revision_snapshot(
    *,
    snapshot_download_fn: Callable[..., str],
    repo_id: str,
    revision: str,
    token: str,
    local_bundle: Path,
    validate_snapshot_fn: Callable[[Path], dict[str, object]],
) -> None:
    with tempfile.TemporaryDirectory(prefix="operate-hf-verify-") as temp_dir:
        snapshot = Path(
            snapshot_download_fn(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                token=token,
                local_dir=temp_dir,
            )
        )
        validate_snapshot_fn(snapshot)
        _verify_remote_replacement(
            local_bundle=local_bundle,
            remote_snapshot=snapshot,
        )


@contextmanager
def _validated_bundle_snapshot(
    *,
    local_bundle: Path,
    captured_manifest: dict[str, object],
    captured_manifest_bytes: bytes,
    validate_snapshot_fn: Callable[[Path], dict[str, object]],
):
    snapshot = Path(tempfile.mkdtemp(prefix="operate-hf-upload-"))
    try:
        bundle_root = local_bundle.resolve()
        for relative in ["MANIFEST.json", *_manifest_paths(captured_manifest)]:
            source = local_bundle / relative
            current = local_bundle
            if any(
                (current := current / part).is_symlink()
                for part in Path(relative).parts
            ):
                raise ValueError(f"local_bundle_file_symlink_forbidden:{relative}")
            try:
                source.resolve(strict=True).relative_to(bundle_root)
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(f"local_bundle_file_invalid:{relative}") from exc
            if not source.is_file():
                raise ValueError(f"local_bundle_file_invalid:{relative}")
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        observed_manifest = validate_snapshot_fn(snapshot)
        if not (
            observed_manifest == captured_manifest
            and (snapshot / "MANIFEST.json").read_bytes() == captured_manifest_bytes
        ):
            raise ValueError("upload_snapshot_manifest_mismatch")
        for path in sorted(snapshot.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        snapshot.chmod(0o555)
        yield snapshot
    finally:
        try:
            if snapshot.exists():
                snapshot.chmod(0o700)
                for path in snapshot.rglob("*"):
                    path.chmod(0o700 if path.is_dir() else 0o600)
                shutil.rmtree(snapshot)
        except OSError as exc:
            raise RuntimeError(f"upload_snapshot_cleanup_failed:{snapshot}") from exc


def _publish_verified_bundle(
    *,
    api: Any,
    snapshot_download_fn: Callable[..., str],
    repo_id: str,
    local_bundle: Path,
    manifest: dict[str, object],
    main_parent: str,
    token: str,
    commit_message: str,
    validate_snapshot_fn: Callable[[Path], dict[str, object]],
) -> str:
    captured_manifest_bytes = (local_bundle / "MANIFEST.json").read_bytes()
    if validate_snapshot_fn(local_bundle) != manifest:
        raise ValueError("local_bundle_manifest_argument_mismatch")
    if (local_bundle / "MANIFEST.json").read_bytes() != captured_manifest_bytes:
        raise ValueError("local_bundle_manifest_changed_during_preflight")
    with _validated_bundle_snapshot(
        local_bundle=local_bundle,
        captured_manifest=manifest,
        captured_manifest_bytes=captured_manifest_bytes,
        validate_snapshot_fn=validate_snapshot_fn,
    ) as snapshot:
        return _publish_verified_snapshot(
            api=api,
            snapshot_download_fn=snapshot_download_fn,
            repo_id=repo_id,
            local_bundle=snapshot,
            manifest=manifest,
            main_parent=main_parent,
            token=token,
            commit_message=commit_message,
            validate_snapshot_fn=validate_snapshot_fn,
        )


def _publish_verified_snapshot(
    *,
    api: Any,
    snapshot_download_fn: Callable[..., str],
    repo_id: str,
    local_bundle: Path,
    manifest: dict[str, object],
    main_parent: str,
    token: str,
    commit_message: str,
    validate_snapshot_fn: Callable[[Path], dict[str, object]],
) -> str:
    """Verify an isolated revision before one CAS-protected main commit."""
    from huggingface_hub import (  # noqa: PLC0415
        CommitOperationAdd,
        CommitOperationCopy,
        CommitOperationDelete,
    )

    if re.fullmatch(r"[0-9a-f]{40,64}", main_parent) is None:
        raise ValueError("main_parent_commit_invalid")
    expected = ["MANIFEST.json", *_manifest_paths(manifest)]
    bundle_root = local_bundle.resolve()
    for relative in expected:
        local = local_bundle / relative
        current = local_bundle
        if any(
            (current := current / part).is_symlink() for part in Path(relative).parts
        ):
            raise ValueError(f"local_bundle_file_symlink_forbidden:{relative}")
        try:
            local.resolve(strict=True).relative_to(bundle_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"local_bundle_file_invalid:{relative}") from exc
        if not local.is_file():
            raise ValueError(f"local_bundle_file_invalid:{relative}")

    observed_before = api.dataset_info(repo_id, revision="main")
    if observed_before.private is not True:
        raise RuntimeError("repository_not_private_before_publish")
    if str(observed_before.sha) != main_parent:
        raise RuntimeError(
            "main_superseded_before_publish:"
            f"expected={main_parent}:observed={observed_before.sha}"
        )
    remote_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
        revision=main_parent,
    )
    if not isinstance(remote_files, list) or any(
        not isinstance(path, str)
        or not path
        or PurePosixPath(path).is_absolute()
        or ".." in PurePosixPath(path).parts
        or PurePosixPath(path).as_posix() != path
        for path in remote_files
    ):
        raise ValueError("remote_repository_file_path_invalid")
    stale = sorted(set(remote_files) - set(expected) - {".gitattributes"})
    staging_branch = f"operate-staging-{uuid.uuid4().hex}"
    api.create_branch(
        repo_id=repo_id,
        repo_type="dataset",
        branch=staging_branch,
        revision=main_parent,
    )
    pending_error: BaseException | None = None
    try:
        staging_operations = [
            CommitOperationDelete(path_in_repo=path, is_folder=False) for path in stale
        ] + [
            CommitOperationAdd(
                path_in_repo=relative,
                path_or_fileobj=local_bundle / relative,
            )
            for relative in expected
        ]
        staging_commit = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            revision=staging_branch,
            parent_commit=main_parent,
            operations=staging_operations,
            commit_message=f"Stage and verify: {commit_message}",
        )
        staging_oid = str(staging_commit.oid)
        if re.fullmatch(r"[0-9a-f]{40,64}", staging_oid) is None:
            raise ValueError("staging_commit_oid_invalid")
        _verify_revision_snapshot(
            snapshot_download_fn=snapshot_download_fn,
            repo_id=repo_id,
            revision=staging_oid,
            token=token,
            local_bundle=local_bundle,
            validate_snapshot_fn=validate_snapshot_fn,
        )

        main_operations = [
            CommitOperationDelete(path_in_repo=path, is_folder=False) for path in stale
        ] + [
            CommitOperationCopy(
                src_path_in_repo=relative,
                path_in_repo=relative,
                src_revision=staging_oid,
            )
            for relative in expected
        ]
        main_commit = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            revision="main",
            parent_commit=main_parent,
            operations=main_operations,
            commit_message=commit_message,
        )
        main_oid = str(main_commit.oid)
        if re.fullmatch(r"[0-9a-f]{40,64}", main_oid) is None:
            raise ValueError("main_commit_oid_invalid")
        _verify_revision_snapshot(
            snapshot_download_fn=snapshot_download_fn,
            repo_id=repo_id,
            revision=main_oid,
            token=token,
            local_bundle=local_bundle,
            validate_snapshot_fn=validate_snapshot_fn,
        )
        observed_after = api.dataset_info(repo_id, revision="main")
        if observed_after.private is not True:
            raise RuntimeError("repository_not_private_after_publish")
        observed_main = str(observed_after.sha)
        if observed_main != main_oid:
            raise RuntimeError(
                "main_superseded_after_publish:"
                f"published={main_oid}:observed={observed_main}"
            )
        return main_oid
    except BaseException as exc:
        pending_error = exc
        raise
    finally:
        try:
            api.delete_branch(
                repo_id=repo_id,
                repo_type="dataset",
                branch=staging_branch,
            )
        except Exception as exc:
            state = "after successful publication" if pending_error is None else ""
            print(
                f"WARNING: staging branch cleanup failed {state}: "
                f"{staging_branch}: {exc}",
                file=sys.stderr,
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True, help="HF dataset repo id")
    ap.add_argument(
        "--data-dir",
        required=True,
        help="local manifest-backed bundle to upload",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="validate inputs without uploading"
    )
    ap.add_argument(
        "--private",
        action="store_true",
        help="create or update the dataset repo as private",
    )
    ap.add_argument(
        "--receipt-output",
        type=Path,
        help=(
            "required for bundles containing formal result trees; writes the "
            "verified immutable distribution receipt"
        ),
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"FATAL: {data_dir} not found", file=sys.stderr)
        return 1
    manifest = data_dir / "MANIFEST.json"
    if not manifest.exists():
        print(f"FATAL: {manifest} not found", file=sys.stderr)
        return 1

    try:
        m = _validate_publishable_bundle(data_dir)
    except (OSError, ValueError) as exc:
        print(f"FATAL: bundle verification failed: {exc}", file=sys.stderr)
        return 1
    release_id = m.get("release_id", "unspecified")
    has_formal_results = isinstance(m.get("formal_result_trees"), dict)
    receipt_output = args.receipt_output.resolve() if args.receipt_output else None
    expected_receipt = REPO / "release" / str(release_id) / (
        "formal_distribution_receipt.json"
    )
    if has_formal_results and receipt_output != expected_receipt:
        print(
            "FATAL: formal result publication requires --receipt-output "
            f"{expected_receipt}",
            file=sys.stderr,
        )
        return 1
    if not has_formal_results and receipt_output is not None:
        print(
            "FATAL: --receipt-output requires a bundle with formal result trees",
            file=sys.stderr,
        )
        return 1
    declared_repo_id = m.get("hf_repo_id")
    if declared_repo_id is not None and declared_repo_id != args.repo_id:
        print(
            "FATAL: bundle_repo_id_mismatch:"
            f"manifest={declared_repo_id!r}:requested={args.repo_id!r}",
            file=sys.stderr,
        )
        return 1
    declared_visibility = m.get("visibility")
    if declared_visibility != "private":
        print(
            f"FATAL: bundle_visibility_invalid:{declared_visibility!r}",
            file=sys.stderr,
        )
        return 1
    private = True
    n_scenarios = m.get("n_scenarios", "?")
    n_files = m.get("n_files", len(m.get("files") or []))
    n_release = m.get("n_release_artifacts", n_files)
    n_backend = m.get("n_backend_files", 0)
    backend_summary = (
        f"compressed backend snapshot {m['backend_archive']}"
        if m.get("backend_archive")
        else f"{n_backend} backend files"
    )
    print(
        f"data/ : {release_id}; {n_scenarios} scenarios, "
        f"{n_release} release artifacts, "
        f"{backend_summary} ({n_files} total tracked)"
    )
    print(f"repo  : {args.repo_id}")

    if args.dry_run:
        print("DRY-RUN: would upload", data_dir, "->", args.repo_id)
        print("  (set HF_TOKEN and re-run without --dry-run to upload)")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token  # noqa: PLC0415
        except ImportError:
            get_token = None  # type: ignore[assignment]
        token = get_token() if get_token is not None else None
    if not token:
        print(
            "FATAL: no Hugging Face token. Set HF_TOKEN or run "
            "`huggingface-cli login`. Get a token from "
            "https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        return 1

    from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415
    from huggingface_hub.utils import RepositoryNotFoundError  # noqa: PLC0415

    api = HfApi(token=token)

    # Create the dataset repo if it doesn't exist.
    try:
        info = api.dataset_info(args.repo_id, revision="main")
        print("  repo exists")
    except RepositoryNotFoundError:
        print(f"  creating dataset repo {args.repo_id} (private={private})...")
        try:
            api.create_repo(
                args.repo_id,
                repo_type="dataset",
                private=private,
                exist_ok=False,
            )
            info = api.dataset_info(args.repo_id, revision="main")
        except Exception as exc:  # noqa: BLE001
            print(f"FATAL: dataset creation failed: {exc}", file=sys.stderr)
            return 1
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: dataset lookup failed: {exc}", file=sys.stderr)
        return 1

    try:
        info = _ensure_private_repository(
            api=api,
            repo_id=args.repo_id,
            observed=info,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: visibility update failed: {exc}", file=sys.stderr)
        return 1
    print("  visibility: private")

    print(
        f"  uploading {data_dir} -> {args.repo_id} (this may take a while for the first upload)..."
    )
    try:
        main_oid = _publish_verified_bundle(
            api=api,
            snapshot_download_fn=snapshot_download,
            repo_id=args.repo_id,
            local_bundle=data_dir,
            manifest=m,
            main_parent=str(info.sha),
            token=token,
            commit_message=(
                f"Upload OPERATE {release_id} eval data ({m['n_scenarios']} scenarios)"
            ),
            validate_snapshot_fn=_validate_publishable_bundle,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: verified publication failed: {exc}", file=sys.stderr)
        return 1
    print(f"\nDONE: {args.repo_id}")
    print(f"  commit: {main_oid}")
    if receipt_output is not None:
        try:
            receipt = _distribution_receipt_payload(
                bundle_dir=data_dir,
                manifest=m,
                repo_id=args.repo_id,
                revision=main_oid,
            )
            _write_distribution_receipt(receipt_output, receipt)
        except (OSError, ValueError) as exc:
            print(
                f"FATAL: verified publication receipt write failed: {exc}",
                file=sys.stderr,
            )
            return 1
        print(f"  receipt: {receipt_output}")
    print(f"  view at https://huggingface.co/datasets/{args.repo_id}")
    print(f"  others can fetch with: {_download_command(args.repo_id, main_oid)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
