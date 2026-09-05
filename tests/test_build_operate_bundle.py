from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from core.implementation_identity import implementation_identity
from evaluation.leaderboard import PRIMARY_LEADERBOARD_FORMULA_VERSION
from evaluation.scorer import SCORING_VERSION
from scripts import batch_llm_eval
from scripts import build_operate_bundle as bundle_builder
from scripts import download_from_hf
from scripts import finalize_operate_release as finalizer
from scripts import verify_release_integrity
from scripts.build_operate_bundle import build_operate_bundle
from scripts.build_operate_bundle import _collect_dynasched_source_assets
from scripts.build_operate_bundle import _collect_exact_source_assets
from scripts.build_operate_bundle import _collect_nrel_microgrid_source_assets
from scripts.build_operate_bundle import _collect_ngsim_us101_source_assets
from scripts.build_operate_bundle import validate_bundle_archives
from scripts.download_from_hf import (
    _extract_formal_evidence_archive,
    _validate_bundle_source_asset_bindings,
    install_bundle_source_assets,
    link_bundle_backends,
    validate_install_data_dir,
    validate_runtime_bundle_compatibility,
    verify_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _formal_result_binding(
    *,
    repo: Path,
    release: Path,
    model: str,
    treatment: str,
    mode: str,
) -> dict[str, str]:
    source = repo / f"{mode}-batch"
    _write_json(source / "RUN_MANIFEST.json", {"interaction_mode": mode})
    (source / "episodes.jsonl").write_text('{"status":"ok"}\n', encoding="utf-8")
    copied, index, _created = finalizer._materialize_result_tree(
        source / "RUN_MANIFEST.json",
        release_dir=release,
        model=model,
        treatment_sha256=treatment,
        validator=lambda _path, _payload: True,
    )
    return {
        "path": copied.relative_to(repo).as_posix(),
        "sha256": _sha256(copied),
        "schema_version": "fixture",
        "model": model,
        "interaction_mode": mode,
        "treatment_sha256": treatment,
        "tree_index_path": (copied.parent / "FORMAL_RESULT_TREE_INDEX.json")
        .relative_to(repo)
        .as_posix(),
        "tree_index_sha256": _sha256(
            copied.parent / "FORMAL_RESULT_TREE_INDEX.json"
        ),
        "tree_root_sha256": index["root_sha256"],
    }


def test_nrel_microgrid_profiles_are_bound_as_bundle_source_assets(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    spec = bundle_builder._NREL_MICROGRID_SOURCE_SPEC
    required_files: dict[str, str] = {}
    locked_assets: list[dict[str, str]] = []
    for index, relative in enumerate(sorted(spec["paths"])):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"profile-{index}\n".encode())
        digest = _sha256(path)
        required_files[relative] = digest
        locked_assets.append({"declared_path": relative, "sha256": digest})
    scenario_id = "microgrid/fixture"
    suite = {
        "scenarios": [
            {
                "scenario_id": scenario_id,
                "backend_kind": "pymgrid_economic_dispatch",
                "case_ledger": {
                    "physical_source_lock": {
                        "schema_version": "source_asset_graph_v1",
                        "backend_kind": "pymgrid_economic_dispatch",
                        "required_source_assets": locked_assets,
                    }
                },
            }
        ]
    }
    external_contract = {
        "delivery": "user_provided",
        "url": spec["redistribution"]["url"],
        "revision": spec["redistribution"]["upstream_commit"],
        "required_files": required_files,
        "metadata": {
            "backend_kinds": sorted(spec["backend_kinds"]),
            "root": "works/nrel-microgrid",
            "roles": {
                relative: list(spec["roles"]) for relative in sorted(spec["paths"])
            },
        },
    }

    contract = _collect_nrel_microgrid_source_assets(
        repo, suite, external_contract
    )

    assert contract["n_scenarios"] == 1
    assert contract["n_files"] == 16
    assert contract["delivery"] == "bundle"
    assert set(contract["files"]) == spec["paths"]
    assert all(
        row["archive_path"].startswith(
            "backends/release_source_assets/nrel_microgrid/works/nrel-microgrid/"
        )
        and row["roles"] == ["derivation_input"]
        and row["scenario_ids"] == [scenario_id]
        for row in contract["files"].values()
    )


def _runtime_closure(
    *,
    source_suite_sha256: str,
    archived_files: dict[str, dict[str, Any]],
    external_sources: dict[str, dict[str, Any]] | None = None,
    backend_links: dict[str, str] | None = None,
    repo_tracked_files: dict[str, dict[str, Any]] | None = None,
    runtime_packages: dict[str, dict[str, Any]] | None = None,
    separately_bundled_files: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    external_sources = external_sources or {}
    backend_links = backend_links or {}
    repo_tracked_files = repo_tracked_files or {}
    runtime_packages = runtime_packages or {}
    separately_bundled_files = separately_bundled_files or {}
    closure = {
        "schema_version": "operate-backend-runtime-closure-v1",
        "release_id": "operate_v0_58_0",
        "status": "backend_runtime_closure_complete",
        "terminal": True,
        "portable": True,
        "source_suite_sha256": source_suite_sha256,
        "archived_files": archived_files,
        "repo_tracked_files": repo_tracked_files,
        "separately_bundled_files": separately_bundled_files,
        "external_sources": external_sources,
        "backend_links": backend_links,
        "runtime_packages": runtime_packages,
        "summary": {
            "n_archived_files": len(archived_files),
            "n_backend_links": len(backend_links),
            "n_external_sources": len(external_sources),
            "n_repo_tracked_files": len(repo_tracked_files),
            "n_runtime_packages": len(runtime_packages),
            "n_separately_bundled_files": len(separately_bundled_files),
            "n_source_assets": (
                len(archived_files)
                + sum(
                    len(record["required_files"])
                    for record in external_sources.values()
                )
                + len(repo_tracked_files)
                + len(separately_bundled_files)
            ),
            "n_unresolved": 0,
            "n_virtual_sources": sum(
                len(package.get("virtual_sources", {}))
                for package in runtime_packages.values()
            ),
        },
    }
    closure["identity_sha256"] = hashlib.sha256(
        json.dumps(closure, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return closure


def _bind_runtime_closure(
    paths: dict[str, Any],
    *,
    source_suite_sha256: str,
    external_sources: dict[str, dict[str, Any]] | None = None,
) -> None:
    runtime_source = paths["repo"] / "works" / "CityLearn" / "runtime.py"
    license_source = paths["repo"] / "works" / "CityLearn" / "LICENSE"
    license_source.write_text("MIT License fixture\n", encoding="utf-8")
    closure = _runtime_closure(
        source_suite_sha256=source_suite_sha256,
        archived_files={
            "backends/citylearn/LICENSE": {
                "source_path": "works/CityLearn/LICENSE",
                "sha256": _sha256(license_source),
                "roles": ["redistribution_license"],
                "backend_kinds": ["citylearn"],
            },
            "backends/citylearn/runtime.py": {
                "source_path": "works/CityLearn/runtime.py",
                "sha256": _sha256(runtime_source),
                "roles": ["runtime"],
                "backend_kinds": ["citylearn"],
            },
        },
        external_sources=external_sources,
        backend_links={"CityLearn": "citylearn"},
    )
    closure_path = paths["release"] / "backend_runtime_closure.json"
    _write_json(closure_path, closure)
    release = json.loads(paths["release_manifest"].read_text(encoding="utf-8"))
    release["backend_runtime_closure"] = {
        "path": closure_path.name,
        "sha256": _sha256(closure_path),
        "schema_version": closure["schema_version"],
        "n_archived_files": closure["summary"]["n_archived_files"],
        "n_external_sources": closure["summary"]["n_external_sources"],
        "n_backend_links": closure["summary"]["n_backend_links"],
        "n_runtime_packages": closure["summary"]["n_runtime_packages"],
        "identity_sha256": closure["identity_sha256"],
    }
    _write_json(paths["release_manifest"], release)


def _write_tar_zst(
    path: Path,
    members: dict[str, bytes],
    *,
    mtime: int = 0,
) -> None:
    process = subprocess.Popen(
        ["zstd", "-q", "-f", "-o", str(path)],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = mtime
            archive.addfile(info, io.BytesIO(payload))
    process.stdin.close()
    assert process.wait() == 0


def test_archive_validation_rejects_unlisted_backend_member(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    archive_name = "backends.tar.zst"
    expected_name = "backends/fixture/runtime.py"
    expected_payload = b"VALUE = 1\n"
    _write_tar_zst(
        bundle / archive_name,
        {
            expected_name: expected_payload,
            "backends/fixture/secret.txt": b"must not be published\n",
        },
    )

    with pytest.raises(ValueError, match="archive_member_unexpected"):
        validate_bundle_archives(
            bundle,
            {
                "backend_archive": archive_name,
                "backend_archive_files": {
                    expected_name: hashlib.sha256(expected_payload).hexdigest()
                },
            },
        )


def test_archive_validation_rejects_nondeterministic_metadata(
    tmp_path: Path,
) -> None:
    archive_name = "backends.tar.zst"
    member_name = "backends/fixture/runtime.py"
    payload = b"VALUE = 1\n"
    _write_tar_zst(
        tmp_path / archive_name,
        {member_name: payload},
        mtime=123456789,
    )

    with pytest.raises(ValueError, match="archive_member_metadata_invalid"):
        validate_bundle_archives(
            tmp_path,
            {
                "backend_archive": archive_name,
                "backend_archive_files": {
                    member_name: hashlib.sha256(payload).hexdigest()
                },
            },
        )


def test_archive_validation_rejects_archive_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.tar.zst"
    _write_tar_zst(outside, {"backends/fixture/runtime.py": b"VALUE = 1\n"})

    with pytest.raises(ValueError, match="backend_archive_invalid"):
        validate_bundle_archives(
            tmp_path,
            {
                "backend_archive": "../outside.tar.zst",
                "backend_archive_files": {
                    "backends/fixture/runtime.py": hashlib.sha256(
                        b"VALUE = 1\n"
                    ).hexdigest()
                },
            },
        )


def test_formal_evidence_archive_is_reproducible_across_source_metadata(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    first_output = paths["repo"] / "bundle-first"
    first = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=first_output,
        include_backends=False,
    )

    os.chmod(paths["stage"], 0o600)
    os.utime(paths["stage"], (123456789, 123456789))
    second_output = paths["repo"] / "bundle-second"
    second = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=second_output,
        include_backends=False,
    )

    archive_name = str(first["formal_evidence_archive"])
    assert first["files"][archive_name] == second["files"][archive_name]
    assert (first_output / archive_name).read_bytes() == (
        second_output / archive_name
    ).read_bytes()


def test_archive_validation_rejects_unlisted_formal_evidence_member(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    output = paths["repo"] / "bundle"
    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=output,
        include_backends=False,
    )
    install_root = str(manifest["formal_evidence_install_root"])
    members = {
        f"{install_root}/{relative}": (paths["evidence"] / relative).read_bytes()
        for relative in manifest["formal_evidence_files"]
    }
    members[f"{install_root}/unlisted.json"] = b"{}\n"
    _write_tar_zst(output / str(manifest["formal_evidence_archive"]), members)

    with pytest.raises(ValueError, match="formal_evidence_archive_member_unexpected"):
        validate_bundle_archives(output, manifest)


def test_backend_archive_returns_exact_member_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    backend = repo / "works" / "CityLearn"
    backend.mkdir(parents=True)
    (backend / "runtime.py").write_bytes(b"VALUE = 1\n")
    (backend / "input.csv").write_bytes(b"value\n1\n")
    dynasched = repo / "works" / "DynaSchedBench"
    for name, payload in {
        "pyproject.toml": b"[project]\nname = 'fixture'\n",
        "README.md": b"fixture\n",
        "LICENSE": b"fixture\n",
        "src/dsbx/__init__.py": b"__version__ = '1'\n",
    }.items():
        path = dynasched / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    archive = tmp_path / "backends.tar.zst"
    source_by_archive = {
        "backends/citylearn/input.csv": backend / "input.csv",
        "backends/citylearn/runtime.py": backend / "runtime.py",
        "backends/dynasched/LICENSE": dynasched / "LICENSE",
        "backends/dynasched/README.md": dynasched / "README.md",
        "backends/dynasched/pyproject.toml": dynasched / "pyproject.toml",
        "backends/dynasched/src/dsbx/__init__.py": (
            dynasched / "src" / "dsbx" / "__init__.py"
        ),
    }
    archived_files = {
        name: {
            "source_path": path.relative_to(repo).as_posix(),
            "sha256": _sha256(path),
            "roles": ["runtime"],
            "backend_kinds": ["citylearn"] if "citylearn" in name else ["dynasched"],
        }
        for name, path in source_by_archive.items()
    }
    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files=archived_files,
        backend_links={"CityLearn": "citylearn", "DynaSchedBench": "dynasched"},
    )

    files = bundle_builder._build_backend_archive(
        repo,
        archive,
        {},
        backend_source_closure=closure,
    )

    expected_names = {
        "backends/citylearn/input.csv",
        "backends/citylearn/runtime.py",
        "backends/dynasched/LICENSE",
        "backends/dynasched/README.md",
        "backends/dynasched/pyproject.toml",
        "backends/dynasched/src/dsbx/__init__.py",
    }
    assert set(files) == expected_names
    assert all(len(digest) == 64 for digest in files.values())
    validate_bundle_archives(
        tmp_path,
        {
            "backend_archive": archive.name,
            "backend_archive_files": files,
        },
    )

    (backend / "unlisted-secret.txt").write_text("must not ship\n", encoding="utf-8")
    files_with_extra = bundle_builder._build_backend_archive(
        repo,
        tmp_path / "backends-with-extra.tar.zst",
        {},
        backend_source_closure=closure,
    )
    assert files_with_extra == files


def test_backend_archive_accepts_in_repo_parent_symlink_as_regular_member(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    backend = repo / "data" / "backends" / "citylearn"
    backend.mkdir(parents=True)
    source = backend / "runtime.py"
    source.write_bytes(b"VALUE = 1\n")
    works = repo / "works"
    works.mkdir()
    (works / "CityLearn").symlink_to(backend, target_is_directory=True)
    declared = works / "CityLearn" / "runtime.py"
    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files={
            "backends/citylearn/runtime.py": {
                "source_path": "works/CityLearn/runtime.py",
                "sha256": _sha256(source),
                "roles": ["runtime"],
                "backend_kinds": ["citylearn"],
            }
        },
        backend_links={"CityLearn": "citylearn"},
    )
    archive = tmp_path / "backends.tar.zst"

    files = bundle_builder._build_backend_archive(
        repo,
        archive,
        {},
        backend_source_closure=closure,
    )

    assert declared.is_file()
    assert files == {"backends/citylearn/runtime.py": _sha256(source)}
    validate_bundle_archives(
        tmp_path,
        {
            "backend_archive": archive.name,
            "backend_archive_files": files,
        },
    )


@pytest.mark.parametrize(
    ("link_kind", "error"),
    (
        ("terminal", "backend_runtime_closure_source_missing"),
        ("dangling", "backend_runtime_closure_source_missing"),
        ("outside_parent", "backend_runtime_closure_source_outside_repo"),
    ),
)
def test_backend_archive_rejects_unsafe_source_symlinks(
    tmp_path: Path,
    link_kind: str,
    error: str,
) -> None:
    repo = tmp_path / "repo"
    works = repo / "works"
    works.mkdir(parents=True)
    declared = works / "CityLearn" / "runtime.py"
    if link_kind == "outside_parent":
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "runtime.py"
        target.write_bytes(b"VALUE = 1\n")
        declared.parent.symlink_to(outside, target_is_directory=True)
        digest = _sha256(target)
    else:
        declared.parent.mkdir()
        target = repo / "data" / "runtime.py"
        if link_kind == "terminal":
            target.parent.mkdir()
            target.write_bytes(b"VALUE = 1\n")
            digest = _sha256(target)
        else:
            digest = "a" * 64
        declared.symlink_to(target)
    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files={
            "backends/citylearn/runtime.py": {
                "source_path": "works/CityLearn/runtime.py",
                "sha256": digest,
                "roles": ["runtime"],
                "backend_kinds": ["citylearn"],
            }
        },
        backend_links={"CityLearn": "citylearn"},
    )

    with pytest.raises((FileNotFoundError, ValueError), match=error):
        bundle_builder._build_backend_archive(
            repo,
            tmp_path / "backends.tar.zst",
            {},
            backend_source_closure=closure,
        )


@pytest.mark.parametrize("mutation", ("missing", "tampered"))
def test_backend_archive_rejects_missing_or_tampered_closure_member(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "works" / "CityLearn" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 1\n")
    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files={
            "backends/citylearn/runtime.py": {
                "source_path": "works/CityLearn/runtime.py",
                "sha256": _sha256(source),
                "roles": ["runtime"],
                "backend_kinds": ["citylearn"],
            }
        },
        backend_links={"CityLearn": "citylearn"},
    )
    if mutation == "missing":
        source.unlink()
    else:
        source.write_bytes(b"VALUE = 2\n")

    with pytest.raises(
        (FileNotFoundError, ValueError),
        match="backend_runtime_closure_(source_missing|hash_mismatch)",
    ):
        bundle_builder._build_backend_archive(
            repo,
            tmp_path / "backends.tar.zst",
            {},
            backend_source_closure=closure,
        )


@pytest.mark.parametrize("mutation", ("missing", "tampered"))
def test_runtime_closure_rejects_repo_tracked_file_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "sources" / "alibaba" / "demo.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"value\n1\n")
    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files={},
        repo_tracked_files={
            "sources/alibaba/demo.csv": {
                "sha256": _sha256(source),
                "roles": ["runtime_input"],
                "backend_kinds": ["alibaba_trace_sim"],
            }
        },
    )
    if mutation == "missing":
        source.unlink()
    else:
        source.write_bytes(b"value\n2\n")

    with pytest.raises(
        (FileNotFoundError, ValueError),
        match="backend_runtime_closure_bound_file_(missing|hash_mismatch)",
    ):
        bundle_builder._validated_backend_runtime_closure(repo, closure)


def test_backend_archive_requires_every_separately_bundled_identity(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "works" / "REALM-Bench-direct-pilot" / "demo.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"{}\n")
    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files={},
        separately_bundled_files={
            "works/REALM-Bench-direct-pilot/demo.json": {
                "sha256": _sha256(source),
                "roles": ["runtime_input"],
                "backend_kinds": ["jsplib_job_shop"],
            }
        },
    )

    with pytest.raises(
        ValueError,
        match="backend_runtime_closure_separately_bundled_file_unbound",
    ):
        bundle_builder._build_backend_archive(
            repo,
            tmp_path / "backends.tar.zst",
            {},
            backend_source_closure=closure,
        )


def test_backend_archive_rejects_nonportable_or_unmapped_closure_source(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "data" / "backends" / "fixture" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 1\n")
    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files={
            "backends/fixture/runtime.py": {
                "source_path": "data/backends/fixture/runtime.py",
                "sha256": _sha256(source),
                "roles": ["runtime"],
                "backend_kinds": ["fixture"],
            }
        },
    )

    with pytest.raises(ValueError, match="backend_runtime_closure_source_path_invalid"):
        bundle_builder._build_backend_archive(
            repo,
            tmp_path / "backends.tar.zst",
            {},
            backend_source_closure=closure,
        )


def test_backend_archive_accepts_repo_owned_resco_environment_license(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    source = repo / "sources" / "resco" / "arterial4x4" / "LICENSE"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"CC BY-NC-SA 4.0\n")
    archive_name = (
        "backends/resco/resco_benchmark/environments/arterial4x4/LICENSE"
    )
    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files={
            archive_name: {
                "source_path": "sources/resco/arterial4x4/LICENSE",
                "sha256": _sha256(source),
                "roles": ["redistribution_license"],
                "backend_kinds": ["sumo"],
            }
        },
    )

    files = bundle_builder._build_backend_archive(
        repo,
        tmp_path / "backends.tar.zst",
        {"resco": {"files": {}}},
        backend_source_closure=closure,
    )

    assert files[archive_name] == _sha256(source)


def test_runtime_closure_validates_uv_lock_package_contract(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source = repo / "works" / "CityLearn" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 1\n")
    uv_lock = repo / "uv.lock"
    uv_lock.write_text("version = 1\n", encoding="utf-8")
    entry: dict[str, Any] = {
        "version": "1.0.0",
        "source": {"registry": "https://pypi.org/simple"},
        "artifacts": [],
    }
    entry["identity_sha256"] = hashlib.sha256(
        json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    entries = [entry]
    runtime_packages = {
        "citylearn": {
            "backend_kinds": ["citylearn"],
            "lock_entries": entries,
            "lock_entries_sha256": hashlib.sha256(
                json.dumps(entries, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "uv_lock_sha256": _sha256(uv_lock),
        }
    }
    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files={
            "backends/citylearn/runtime.py": {
                "source_path": "works/CityLearn/runtime.py",
                "sha256": _sha256(source),
                "roles": ["runtime"],
                "backend_kinds": ["citylearn"],
            }
        },
        backend_links={"CityLearn": "citylearn"},
        runtime_packages=runtime_packages,
    )

    validated = bundle_builder._validated_backend_runtime_closure(repo, closure)

    assert len(validated) == 1
    closure["runtime_packages"]["citylearn"]["uv_lock_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime_package_invalid:citylearn"):
        bundle_builder._validated_backend_runtime_closure(repo, closure)


def _dynasched_source_fixture(
    tmp_path: Path,
    *,
    works_layout: bool = False,
    include_physical_lock: bool = True,
) -> tuple[Path, dict[str, Any]]:
    repo = tmp_path / "repo"
    source_root = (
        repo / "works" / "DynaSchedBench" / "data" / "fixture"
        if works_layout
        else repo / "sources" / "dynasched" / "fixture"
    )
    assets = {
        "input_model.json": b'{"jobs": []}\n',
        "events.jsonl": b'{"event_type": "ARRIVAL"}\n',
        "LICENSE": b"Apache-2.0 fixture\n",
    }
    source_assets: dict[str, dict[str, str]] = {}
    for name, payload in assets.items():
        path = (
            repo / "works" / "DynaSchedBench" / "LICENSE"
            if works_layout and name == "LICENSE"
            else source_root / name
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        role = "license" if name == "LICENSE" else "runtime_input"
        source_assets[name] = {
            "path": path.relative_to(repo).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "role": role,
        }
    scenario = repo / "scenarios" / "operate_v0_58_0" / "dynasched.yaml"
    scenario.parent.mkdir(parents=True)
    scenario_body = {
        "scenario_id": "logistics/dynasched-fixture",
        "backend_kind": "dynasched_flexible_job_shop",
        "backend_config": {"source_assets": source_assets},
        "source_contract": {
            "runtime_input": [
                source_assets["input_model.json"]["path"],
                source_assets["events.jsonl"]["path"],
            ],
            "derivation_input": [],
            "implementation_asset": [],
            "metadata": [],
            "license": [source_assets["LICENSE"]["path"]],
        },
    }
    if include_physical_lock:
        scenario_body["physical_source_lock"] = {
            "schema_version": "source_asset_graph_v1",
            "backend_kind": "dynasched_flexible_job_shop",
            "required_source_assets": [
                {
                    "declared_path": source_assets[name]["path"],
                    "sha256": source_assets[name]["sha256"],
                }
                for name in ("input_model.json", "events.jsonl")
            ],
        }
    scenario.write_text(
        json.dumps(scenario_body) + "\n",
        encoding="utf-8",
    )
    suite = {
        "n_scenarios": 1,
        "scenarios": [
            {
                "scenario_id": "logistics/dynasched-fixture",
                "backend_kind": "dynasched_flexible_job_shop",
                "path": scenario.relative_to(repo).as_posix(),
            }
        ],
    }
    return repo, suite


def _exact_source_fixture(
    tmp_path: Path,
    source_id: str,
) -> tuple[Path, dict[str, Any]]:
    repo = tmp_path / "repo"
    spec = bundle_builder._EXACT_SOURCE_SPECS[source_id]
    payloads = (
        {
            next(path for path in spec["paths"] if "node_list" in path): (
                b"sn,cpu_milli,memory_mib,gpu,model\nopenb-node-1,1000,2048,1,V100M16\n"
            ),
            next(path for path in spec["paths"] if "pod_list" in path): (
                b"name,cpu_milli,memory_mib,num_gpu,gpu_milli,gpu_spec,qos,"
                b"pod_phase,creation_time,deletion_time,scheduled_time\n"
                b"openb-pod-1,100,256,1,500,V100M16,LS,Running,0,120,0\n"
            ),
        }
        if source_id == "alibaba_openb_v2023"
        else {next(iter(spec["paths"])): b'{"instances": []}\n'}
    )
    digests: dict[str, str] = {}
    for relative, payload in payloads.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        digests[relative] = hashlib.sha256(payload).hexdigest()
    scenario_id = f"fixture/{source_id}"
    scenario_path = repo / "scenarios" / "operate_v0_58_0" / f"{source_id}.yaml"
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    source_contract: dict[str, Any] = {
        "runtime_input": sorted(payloads),
        "derivation_input": [],
        "implementation_asset": [],
        "metadata": [],
        "license": [],
    }
    if source_id == "alibaba_openb_v2023":
        source_contract["file_sha256s"] = dict(sorted(digests.items()))
    scenario_path.write_text(
        yaml.safe_dump(
            {
                "scenario_id": scenario_id,
                "backend_kind": spec["backend_kind"],
                "source_contract": source_contract,
                "provenance": {
                    "data_source": spec["dataset"],
                    "files": sorted(payloads),
                    "commit": spec["upstream_commit"],
                    "url": spec["url"],
                    "lock_strategy": spec["lock_strategy"],
                    "license": spec["license"],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    row = {
        "scenario_id": scenario_id,
        "backend_kind": spec["backend_kind"],
        "source_denominator_key": (
            f"{spec.get('source_denominator_prefix', source_id)}fixture"
        ),
        "path": scenario_path.relative_to(repo).as_posix(),
        "case_ledger": {
            "physical_source_lock": {
                "schema_version": "source_asset_graph_v1",
                "backend_kind": spec["backend_kind"],
                "required_source_assets": [
                    {"declared_path": path, "sha256": digest}
                    for path, digest in sorted(digests.items())
                ],
            }
        },
    }
    return repo, {"n_scenarios": 1, "scenarios": [row]}


def _ngsim_shared_source_fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any]]:
    repo = tmp_path / "repo"
    rows: list[dict[str, Any]] = []
    first_runtime: Path | None = None
    first_license: Path | None = None
    for candidate in ("candidate-a", "candidate-b"):
        root = (
            repo
            / "works/autonomous_driving/ngsim/recovery/us101-v60-seven/bundles"
            / candidate
        )
        runtime = root / "normalized/trajectories.sqlite3"
        license_path = root / "LICENSES/NGSIM_LICENSE_REVIEW_APPROVED.md"
        runtime.parent.mkdir(parents=True, exist_ok=True)
        license_path.parent.mkdir(parents=True, exist_ok=True)
        if first_runtime is None:
            runtime.write_bytes(b"shared normalized trajectories\n")
            license_path.write_bytes(b"CC-BY-SA-4.0 approved\n")
            first_runtime = runtime
            first_license = license_path
        else:
            assert first_license is not None
            os.link(first_runtime, runtime)
            os.link(first_license, license_path)
        runtime_relative = runtime.relative_to(repo).as_posix()
        license_relative = license_path.relative_to(repo).as_posix()
        (root / "checksums.sha256").write_text(
            "".join(
                f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
                for path in sorted((runtime, license_path))
            ),
            encoding="ascii",
        )
        scenario_id = f"autonomous_driving/{candidate}"
        scenario = repo / "scenarios/operate_v0_60_0" / f"{candidate}.yaml"
        scenario.parent.mkdir(parents=True, exist_ok=True)
        file_sha256s = {runtime_relative: _sha256(runtime)}
        scenario.write_text(
            yaml.safe_dump(
                {
                    "scenario_id": scenario_id,
                    "backend_kind": "sumo_ego",
                    "source_contract": {
                        "runtime_input": [runtime_relative],
                        "derivation_input": [],
                        "implementation_asset": [],
                        "metadata": [],
                        "license": [license_relative],
                        "file_sha256s": file_sha256s,
                    },
                    "provenance": {
                        "dataset_id": "8ect-6jqj",
                        "source_release": "doi:10.21949/1504477",
                        "recording_id": "us-101",
                        "license_id": "CC-BY-SA-4.0",
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "scenario_id": scenario_id,
                "backend_kind": "sumo_ego",
                "path": scenario.relative_to(repo).as_posix(),
                "case_ledger": {
                    "physical_source_lock": {
                        "schema_version": "source_asset_graph_v1",
                        "backend_kind": "sumo_ego",
                        "required_source_assets": [
                            {"declared_path": path, "sha256": digest}
                            for path, digest in sorted(file_sha256s.items())
                        ],
                    }
                },
            }
        )
    return repo, {"n_scenarios": len(rows), "scenarios": rows}


def test_ngsim_shared_bundle_deduplicates_bytes_and_preserves_install_paths(
    tmp_path: Path,
) -> None:
    repo, suite = _ngsim_shared_source_fixture(tmp_path)

    contract = _collect_ngsim_us101_source_assets(repo, suite)

    assert contract["n_scenarios"] == 2
    assert contract["n_files"] == 6
    assert contract["n_blobs"] == 3
    assert len({row["archive_path"] for row in contract["files"].values()}) == 3
    checksums = {
        path: row for path, row in contract["files"].items()
        if path.endswith("/checksums.sha256")
    }
    assert len(checksums) == 2
    assert all(row["roles"] == ["metadata"] for row in checksums.values())
    assert all(
        row["archive_path"].startswith(
            "backends/release_source_assets/ngsim_us101/blobs/"
        )
        for row in contract["files"].values()
    )

    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files={},
        separately_bundled_files={
            install_path: {
                "sha256": row["sha256"],
                "roles": row["roles"],
                "backend_kinds": ["sumo_ego"],
            }
            for install_path, row in contract["files"].items()
        },
    )
    archive = tmp_path / "ngsim-backends.tar.zst"

    files = bundle_builder._build_backend_archive(
        repo,
        archive,
        {"ngsim_us101": contract},
        backend_source_closure=closure,
    )

    assert files == {
        row["archive_path"]: row["sha256"]
        for row in contract["blobs"].values()
    }
    validate_bundle_archives(
        tmp_path,
        {
            "backend_archive": archive.name,
            "backend_archive_files": files,
        },
    )


def test_ngsim_shared_bundle_rejects_missing_license_path(
    tmp_path: Path,
) -> None:
    repo, suite = _ngsim_shared_source_fixture(tmp_path)
    scenario_path = repo / suite["scenarios"][0]["path"]
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    license_path = scenario["source_contract"]["license"][0]
    (repo / license_path).unlink()

    with pytest.raises(FileNotFoundError):
        _collect_ngsim_us101_source_assets(repo, suite)


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_ngsim_bundle_rejects_missing_or_stale_checksum_metadata(
    tmp_path: Path, mutation: str,
) -> None:
    repo, suite = _ngsim_shared_source_fixture(tmp_path)
    checksum = next(repo.glob("works/**/candidate-a/checksums.sha256"))
    if mutation == "missing":
        checksum.unlink()
    else:
        checksum.write_bytes(b"0" * 64 + b"  normalized/trajectories.sqlite3\n")
    with pytest.raises((FileNotFoundError, ValueError), match="checksums"):
        _collect_ngsim_us101_source_assets(repo, suite)


def _m5_metadata_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any]]:
    repo = tmp_path / "repo"
    spec = bundle_builder._EXACT_SOURCE_SPECS["m5"]
    locked = {}
    for relative in sorted(spec["paths"]):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture CSV\n")
        locked[relative] = _sha256(path)
    metadata_path = repo / "works/M5/source_lock.json"
    _write_json(metadata_path, {"source_id": "m5_forecasting", "files": locked})
    monkeypatch.setitem(spec, "metadata", {
        "works/M5/source_lock.json": _sha256(metadata_path),
    })
    provenance = spec["provenance_variants"][0]
    scenario = repo / "scenarios/m5.yaml"
    _write_json(scenario, {
        "scenario_id": "fixture/m5", "backend_kind": "orgym_invmgmt",
        "source_contract": {
            "runtime_input": sorted(locked), "metadata": ["works/M5/source_lock.json"],
        },
        "provenance": {
            "license": provenance["license"], "url": provenance["url"],
            "lock_strategy": provenance["lock_strategy"],
            "commit": provenance["upstream_commit"],
        },
    })
    return repo, {"scenarios": [{
        "scenario_id": "fixture/m5", "backend_kind": "orgym_invmgmt",
        "path": "scenarios/m5.yaml", "case_ledger": {"physical_source_lock": {
            "schema_version": "source_asset_graph_v1", "backend_kind": "orgym_invmgmt",
            "required_source_assets": [
                {"declared_path": p, "sha256": digest} for p, digest in locked.items()
            ],
        }},
    }]}


def test_m5_bundle_includes_locked_metadata_without_changing_physical_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, suite = _m5_metadata_fixture(tmp_path, monkeypatch)
    contract = _collect_exact_source_assets(repo, suite, source_id="m5")
    assert contract["n_files"] == 4
    metadata = contract["files"]["works/M5/source_lock.json"]
    assert metadata["roles"] == ["metadata"]
    assert metadata["sha256"] == _sha256(repo / "works/M5/source_lock.json")
    assert metadata["scenario_ids"] == ["fixture/m5"]
    assert len(suite["scenarios"][0]["case_ledger"]["physical_source_lock"]["required_source_assets"]) == 3


@pytest.mark.parametrize("mutation", ["missing", "tampered", "undeclared"])
def test_m5_bundle_rejects_unverified_source_lock_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    repo, suite = _m5_metadata_fixture(tmp_path, monkeypatch)
    path = repo / "works/M5/source_lock.json"
    if mutation == "missing":
        path.unlink()
    elif mutation == "tampered":
        path.write_bytes(b"{}\n")
    else:
        scenario = repo / suite["scenarios"][0]["path"]
        body = json.loads(scenario.read_text())
        body["source_contract"]["metadata"] = []
        _write_json(scenario, body)
    with pytest.raises((FileNotFoundError, ValueError), match="source_lock|metadata"):
        _collect_exact_source_assets(repo, suite, source_id="m5")


def test_ngsim_shared_bundle_rejects_blob_manifest_drift(
    tmp_path: Path,
) -> None:
    repo, suite = _ngsim_shared_source_fixture(tmp_path)
    contract = _collect_ngsim_us101_source_assets(repo, suite)
    next(iter(contract["blobs"].values()))["install_paths"].pop()

    with pytest.raises(ValueError, match="ngsim_us101_blob_contract_invalid"):
        bundle_builder._build_backend_archive(
            repo,
            tmp_path / "ngsim-backends.tar.zst",
            {"ngsim_us101": contract},
            backend_source_closure=_runtime_closure(
                source_suite_sha256="a" * 64,
                archived_files={},
                separately_bundled_files={
                    install_path: {
                        "sha256": row["sha256"],
                        "roles": row["roles"],
                        "backend_kinds": ["sumo_ego"],
                    }
                    for install_path, row in contract["files"].items()
                },
            ),
        )


@pytest.mark.parametrize(
    ("source_id", "expected_files"),
    [
        ("alibaba_openb_v2023", 2),
        ("realm_j2", 1),
    ],
)
def test_exact_source_bundle_uses_only_suite_locked_runtime_files(
    tmp_path: Path,
    source_id: str,
    expected_files: int,
) -> None:
    repo, suite = _exact_source_fixture(tmp_path, source_id)
    unrelated = repo / "works" / "unrelated-large-tree" / "unused.bin"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"must not be packaged")

    contract = _collect_exact_source_assets(repo, suite, source_id=source_id)

    assert contract["n_scenarios"] == 1
    assert contract["n_files"] == expected_files
    assert (
        contract["delivery"]
        == (bundle_builder._EXACT_SOURCE_SPECS[source_id]["delivery"])
    )
    assert set(contract["files"]) == set(
        bundle_builder._EXACT_SOURCE_SPECS[source_id]["paths"]
    )
    assert all(
        row["archive_path"].startswith(
            f"backends/release_source_assets/{source_id}/works/"
        )
        for row in contract["files"].values()
    )
    assert (
        contract["redistribution"]["dataset"]
        == (bundle_builder._EXACT_SOURCE_SPECS[source_id]["dataset"])
    )
    assert len({row["archive_path"] for row in contract["files"].values()}) == (
        expected_files
    )


@pytest.mark.parametrize("source_id", ["alibaba_openb_v2023", "realm_j2"])
def test_exact_source_bundle_rejects_locked_byte_drift(
    tmp_path: Path,
    source_id: str,
) -> None:
    repo, suite = _exact_source_fixture(tmp_path, source_id)
    drifted = repo / next(iter(bundle_builder._EXACT_SOURCE_SPECS[source_id]["paths"]))
    drifted.write_bytes(drifted.read_bytes() + b"drift")

    with pytest.raises(ValueError, match=f"{source_id}_source_asset_hash_mismatch"):
        _collect_exact_source_assets(repo, suite, source_id=source_id)


def test_realm_source_asset_is_archived_once_and_restores_cleanly(
    tmp_path: Path,
) -> None:
    repo, suite = _exact_source_fixture(tmp_path, "realm_j2")
    runtime = repo / "works" / "CityLearn" / "runtime.py"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_bytes(b"VALUE = 1\n")
    contract = _collect_exact_source_assets(repo, suite, source_id="realm_j2")
    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files={
            "backends/citylearn/runtime.py": {
                "source_path": "works/CityLearn/runtime.py",
                "sha256": _sha256(runtime),
                "roles": ["runtime"],
                "backend_kinds": ["citylearn"],
            }
        },
        backend_links={"CityLearn": "citylearn"},
    )
    data_dir = tmp_path / "bundle"
    data_dir.mkdir()
    archive = data_dir / "backends.tar.zst"

    files = bundle_builder._build_backend_archive(
        repo,
        archive,
        {"realm_j2": contract},
        backend_source_closure=closure,
    )

    realm_member = next(iter(contract["files"].values()))["archive_path"]
    assert list(files).count(realm_member) == 1
    validate_bundle_archives(
        data_dir,
        {
            "backend_archive": archive.name,
            "backend_archive_files": files,
        },
    )
    process = subprocess.Popen(
        ["zstd", "-dc", str(archive)],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    observed: list[str] = []
    with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
        for member in tar:
            observed.append(member.name)
            if member.name == realm_member:
                stream = tar.extractfile(member)
                assert stream is not None
                destination = data_dir / realm_member
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(stream.read())
    process.stdout.close()
    assert process.wait() == 0
    assert observed.count(realm_member) == 1

    clone = tmp_path / "clean-clone"
    clone.mkdir()
    assert (
        install_bundle_source_assets(
            data_dir,
            {"source_assets": {"realm_j2": contract}},
            repo_root=clone,
        )
        == 1
    )
    install_path, row = next(iter(contract["files"].items()))
    assert _sha256(clone / install_path) == row["sha256"]


def test_backend_archive_rejects_source_asset_runtime_member_collision(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    runtime = repo / "works" / "CityLearn" / "runtime.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_bytes(b"VALUE = 1\n")
    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files={
            "backends/citylearn/runtime.py": {
                "source_path": "works/CityLearn/runtime.py",
                "sha256": _sha256(runtime),
                "roles": ["runtime"],
                "backend_kinds": ["citylearn"],
            }
        },
        backend_links={"CityLearn": "citylearn"},
    )
    source_assets = {
        "realm_j2": {
            "files": {
                "works/CityLearn/runtime.py": {
                    "archive_path": "backends/citylearn/runtime.py",
                    "delivery": "bundle",
                    "sha256": _sha256(runtime),
                }
            }
        }
    }

    with pytest.raises(ValueError, match="source_asset_collision"):
        bundle_builder._build_backend_archive(
            repo,
            tmp_path / "backends.tar.zst",
            source_assets,
            backend_source_closure=closure,
        )


def test_dynasched_source_assets_are_separate_exact_archive_members(
    tmp_path: Path,
) -> None:
    repo, suite = _dynasched_source_fixture(tmp_path, works_layout=True)
    runtime = repo / "works" / "CityLearn" / "runtime.py"
    runtime.parent.mkdir(parents=True, exist_ok=True)
    runtime.write_bytes(b"VALUE = 1\n")
    contract = _collect_dynasched_source_assets(repo, suite)
    closure = _runtime_closure(
        source_suite_sha256="a" * 64,
        archived_files={
            "backends/citylearn/runtime.py": {
                "source_path": "works/CityLearn/runtime.py",
                "sha256": _sha256(runtime),
                "roles": ["runtime"],
                "backend_kinds": ["citylearn"],
            }
        },
        backend_links={"CityLearn": "citylearn", "DynaSchedBench": "dynasched"},
        separately_bundled_files={
            install_path: {
                "sha256": row["sha256"],
                "roles": sorted(row["roles"]),
                "backend_kinds": ["dynasched_flexible_job_shop"],
            }
            for install_path, row in contract["files"].items()
        },
    )

    files = bundle_builder._build_backend_archive(
        repo,
        tmp_path / "backends.tar.zst",
        {"dynasched": contract},
        backend_source_closure=closure,
    )

    assert {row["archive_path"] for row in contract["files"].values()} < set(files)


@pytest.mark.parametrize("source_id", ["alibaba_openb_v2023", "realm_j2"])
def test_clean_clone_restores_exact_source_contract(
    tmp_path: Path,
    source_id: str,
) -> None:
    repo, suite = _exact_source_fixture(tmp_path, source_id)
    contract = _collect_exact_source_assets(repo, suite, source_id=source_id)
    data_dir = tmp_path / "bundle"
    for install_path, row in contract["files"].items():
        if row["delivery"] == "bundle":
            archive_path = data_dir / row["archive_path"]
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / install_path, archive_path)
    clone = tmp_path / "clean-clone"
    clone.mkdir()
    if contract["delivery"] == "upstream_fetch":
        for install_path in contract["files"]:
            destination = clone / install_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / install_path, destination)
    manifest = {"source_assets": {source_id: contract}}

    assert install_bundle_source_assets(
        data_dir,
        manifest,
        repo_root=clone,
    ) == (contract["n_files"] if contract["delivery"] == "bundle" else 0)
    assert (
        install_bundle_source_assets(
            data_dir,
            manifest,
            repo_root=clone,
        )
        == 0
    )
    for install_path, row in contract["files"].items():
        restored = clone / install_path
        assert restored.is_file()
        assert not restored.is_symlink()
        assert _sha256(restored) == row["sha256"]


def test_dynasched_bundle_collects_only_suite_declared_source_assets(
    tmp_path: Path,
) -> None:
    repo, suite = _dynasched_source_fixture(tmp_path)
    unrelated = repo / "works" / "DynaSchedBench" / "data" / "large.bin"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_bytes(b"must not be packaged")

    contract = _collect_dynasched_source_assets(repo, suite)

    assert set(contract["files"]) == {
        "sources/dynasched/fixture/input_model.json",
        "sources/dynasched/fixture/events.jsonl",
        "sources/dynasched/fixture/LICENSE",
    }
    assert all(
        row["archive_path"].startswith("backends/dynasched_source_assets/sources/")
        for row in contract["files"].values()
    )
    assert contract["scenario_ids"] == ["logistics/dynasched-fixture"]
    assert contract["redistribution_license_paths"] == [
        "sources/dynasched/fixture/LICENSE"
    ]


def test_dynasched_bundle_rejects_declared_source_hash_drift(tmp_path: Path) -> None:
    repo, suite = _dynasched_source_fixture(tmp_path)
    scenario_path = repo / suite["scenarios"][0]["path"]
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario["backend_config"]["source_assets"]["events.jsonl"]["sha256"] = "0" * 64
    scenario_path.write_text(json.dumps(scenario) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dynasched_source_asset_hash_mismatch"):
        _collect_dynasched_source_assets(repo, suite)


def test_dynasched_bundle_accepts_pinned_works_dataset_paths(
    tmp_path: Path,
) -> None:
    repo, suite = _dynasched_source_fixture(
        tmp_path,
        works_layout=True,
        include_physical_lock=False,
    )

    contract = _collect_dynasched_source_assets(repo, suite)

    assert set(contract["files"]) == {
        "works/DynaSchedBench/data/fixture/input_model.json",
        "works/DynaSchedBench/data/fixture/events.jsonl",
        "works/DynaSchedBench/LICENSE",
    }
    assert len({row["archive_path"] for row in contract["files"].values()}) == len(
        contract["files"]
    )
    assert (
        contract["files"]["works/DynaSchedBench/LICENSE"]["archive_path"]
        == "backends/dynasched/LICENSE"
    )
    assert (
        contract["files"]["works/DynaSchedBench/data/fixture/input_model.json"][
            "archive_path"
        ]
        == "backends/dynasched/data/fixture/input_model.json"
    )
    assert contract["redistribution_license_paths"] == ["works/DynaSchedBench/LICENSE"]


def test_clean_clone_installs_every_manifest_bound_dynasched_source(
    tmp_path: Path,
) -> None:
    repo, suite = _dynasched_source_fixture(tmp_path)
    contract = _collect_dynasched_source_assets(repo, suite)
    data_dir = tmp_path / "bundle"
    for install_path, row in contract["files"].items():
        archive_path = data_dir / row["archive_path"]
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / install_path, archive_path)
    clone = tmp_path / "clean-clone"
    clone.mkdir()

    assert install_bundle_source_assets(
        data_dir,
        {"source_assets": {"dynasched": contract}},
        repo_root=clone,
    ) == len(contract["files"])
    for install_path, row in contract["files"].items():
        restored = clone / install_path
        assert restored.is_file()
        assert _sha256(restored) == row["sha256"]
    assert (
        install_bundle_source_assets(
            data_dir,
            {"source_assets": {"dynasched": contract}},
            repo_root=clone,
        )
        == 0
    )


def test_clean_clone_restores_dynasched_works_dataset_paths(
    tmp_path: Path,
) -> None:
    repo, suite = _dynasched_source_fixture(tmp_path, works_layout=True)
    contract = _collect_dynasched_source_assets(repo, suite)
    data_dir = tmp_path / "bundle"
    for install_path, row in contract["files"].items():
        archive_path = data_dir / row["archive_path"]
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / install_path, archive_path)
    clone = tmp_path / "clean-clone"
    clone.mkdir()
    manifest = {
        "backend_links": {"DynaSchedBench": "dynasched"},
        "source_assets": {"dynasched": contract},
    }

    assert link_bundle_backends(data_dir, manifest, repo_root=clone) == 1
    assert (
        install_bundle_source_assets(
            data_dir,
            manifest,
            repo_root=clone,
        )
        == 0
    )
    assert (clone / "works" / "DynaSchedBench").is_symlink()
    for install_path, row in contract["files"].items():
        assert _sha256(clone / install_path) == row["sha256"]


def test_runtime_bundle_manifest_binds_final_suite_dynasched_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    source_repo, suite = _dynasched_source_fixture(tmp_path / "source-fixture")
    shutil.copytree(source_repo / "sources", paths["repo"] / "sources")
    shutil.copytree(source_repo / "scenarios", paths["repo"] / "scenarios")
    source_suite = paths["release"] / "protocol21_source_suite.json"
    _write_json(source_suite, suite)
    suite_sha256 = _sha256(source_suite)
    pipeline = json.loads(paths["pipeline"].read_text(encoding="utf-8"))
    pipeline["source_suite_sha256"] = suite_sha256
    _write_json(paths["pipeline"], pipeline)
    release = json.loads(paths["release_manifest"].read_text(encoding="utf-8"))
    release["protocol21_replay"]["source_suite_sha256"] = suite_sha256
    release["pipeline_artifacts"]["pipeline_manifest_sha256"] = _sha256(
        paths["pipeline"]
    )
    _write_json(paths["release_manifest"], release)
    _bind_runtime_closure(paths, source_suite_sha256=suite_sha256)

    captured: dict[str, Any] = {}
    archive_files = {
        "backends/citylearn/LICENSE": _sha256(
            paths["repo"] / "works" / "CityLearn" / "LICENSE"
        ),
        "backends/fixture/runtime.py": "a" * 64,
    }

    def fake_archive(
        _repo_root: Path,
        archive_path: Path,
        source_assets: dict[str, dict[str, Any]],
        *,
        backend_source_closure: dict[str, Any],
    ) -> dict[str, str]:
        assert backend_source_closure
        captured.update(source_assets["dynasched"])
        archive_path.write_bytes(b"fixture archive")
        return archive_files

    monkeypatch.setattr(bundle_builder, "_build_backend_archive", fake_archive)
    output = paths["repo"] / "data_operate_v058"
    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=output,
        include_backends=True,
    )

    assert manifest["source_assets"]["dynasched"] == captured
    assert captured["source_suite"] == {
        "path": "release/operate_v0_58_0/protocol21_source_suite.json",
        "sha256": suite_sha256,
    }
    assert captured["n_files"] == 3
    assert manifest["backend_archive_files"] == archive_files
    assert manifest["n_backend_files"] == len(archive_files)
    closure_binding = manifest["backend_runtime_closure"]
    assert closure_binding["path"] in manifest["files"]
    assert manifest["files"][closure_binding["path"]] == closure_binding["sha256"]
    assert closure_binding["n_archived_files"] == 2
    assert closure_binding["n_backend_links"] == 1
    assert len(closure_binding["identity_sha256"]) == 64


def test_runtime_bundle_manifest_binds_openb_and_realm_suite_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    source_repo, openb_suite = _exact_source_fixture(
        tmp_path / "source-fixture",
        "alibaba_openb_v2023",
    )
    _, realm_suite = _exact_source_fixture(
        tmp_path / "source-fixture",
        "realm_j2",
    )
    shutil.copytree(
        source_repo / "works",
        paths["repo"] / "works",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        source_repo / "scenarios",
        paths["repo"] / "scenarios",
        dirs_exist_ok=True,
    )
    suite = {
        "n_scenarios": 2,
        "scenarios": openb_suite["scenarios"] + realm_suite["scenarios"],
    }
    source_suite = paths["release"] / "protocol21_source_suite.json"
    _write_json(source_suite, suite)
    suite_sha256 = _sha256(source_suite)
    pipeline = json.loads(paths["pipeline"].read_text(encoding="utf-8"))
    pipeline["source_suite_sha256"] = suite_sha256
    _write_json(paths["pipeline"], pipeline)
    release = json.loads(paths["release_manifest"].read_text(encoding="utf-8"))
    release["protocol21_replay"]["source_suite_sha256"] = suite_sha256
    release["pipeline_artifacts"]["pipeline_manifest_sha256"] = _sha256(
        paths["pipeline"]
    )
    _write_json(paths["release_manifest"], release)
    openb_spec = bundle_builder._EXACT_SOURCE_SPECS["alibaba_openb_v2023"]
    _bind_runtime_closure(
        paths,
        source_suite_sha256=suite_sha256,
        external_sources={
            "alibaba_openb_v2023": {
                "delivery": "upstream_fetch",
                "url": openb_spec["url"],
                "revision": openb_spec["upstream_commit"],
                "required_files": {
                    relative: _sha256(source_repo / relative)
                    for relative in sorted(openb_spec["paths"])
                },
                "metadata": {
                    "backend_kinds": ["alibaba_openb_gpu_placement"],
                    "license_status": "upstream_research_terms",
                    "redistributed": False,
                    "roles": {
                        relative: ["runtime_input"]
                        for relative in sorted(openb_spec["paths"])
                    },
                    "root": "works/clusterdata",
                },
            }
        },
    )

    captured: dict[str, dict[str, Any]] = {}

    def fake_archive(
        _repo_root: Path,
        archive_path: Path,
        source_assets: dict[str, dict[str, Any]],
        *,
        backend_source_closure: dict[str, Any],
    ) -> dict[str, str]:
        assert backend_source_closure
        captured.update(source_assets)
        archive_path.write_bytes(b"fixture archive")
        return {
            "backends/citylearn/LICENSE": _sha256(
                paths["repo"] / "works" / "CityLearn" / "LICENSE"
            ),
            "backends/fixture/runtime.py": "a" * 64,
        }

    monkeypatch.setattr(bundle_builder, "_build_backend_archive", fake_archive)
    output = paths["repo"] / "data_operate_v058"
    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=output,
        include_backends=True,
    )

    assert set(manifest["source_assets"]) == {
        "alibaba_openb_v2023",
        "realm_j2",
    }
    assert manifest["source_assets"] == captured
    assert manifest["external_sources"] == {
        "alibaba_openb_v2023": {
            "delivery": "upstream_fetch",
            "url": openb_spec["url"],
            "revision": openb_spec["upstream_commit"],
            "required_files": {
                relative: _sha256(source_repo / relative)
                for relative in sorted(openb_spec["paths"])
            },
            "metadata": {
                "backend_kinds": ["alibaba_openb_gpu_placement"],
                "license_status": "upstream_research_terms",
                "redistributed": False,
                "roles": {
                    relative: ["runtime_input"]
                    for relative in sorted(openb_spec["paths"])
                },
                "root": "works/clusterdata",
            },
        }
    }
    assert manifest["backend_links"] == {"CityLearn": "citylearn"}
    assert "clusterdata" not in manifest["backend_links"]
    assert sum(contract["n_files"] for contract in captured.values()) == 3
    assert all(
        contract["source_suite"]
        == {
            "path": "release/operate_v0_58_0/protocol21_source_suite.json",
            "sha256": suite_sha256,
        }
        for contract in captured.values()
    )
    _validate_bundle_source_asset_bindings(
        manifest,
        local_release=release,
        repo_root=paths["repo"],
    )


def test_downloader_rejects_backend_archive_missing_dynasched_source_contract(
    tmp_path: Path,
) -> None:
    repo, suite = _dynasched_source_fixture(tmp_path)
    suite_path = repo / "release" / "operate_v0_58_0" / "protocol21_source_suite.json"
    _write_json(suite_path, suite)
    local_release = {
        "protocol21_replay": {
            "source_suite": suite_path.relative_to(repo).as_posix(),
            "source_suite_sha256": _sha256(suite_path),
        }
    }

    with pytest.raises(ValueError, match="dynasched_source_asset_contract_missing"):
        _validate_bundle_source_asset_bindings(
            {"backend_archive": "backends.tar.zst", "source_assets": {}},
            local_release=local_release,
            repo_root=repo,
        )


def _fixture(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    card = repo / "docs" / "hf" / "OPERATE_DATASET_CARD.md"
    card.parent.mkdir(parents=True)
    card.write_text(
        "---\npretty_name: OPERATE\n---\n\n"
        "# OPERATE\n\nCurrent release: `operate_v0_58_0`.\n",
        encoding="utf-8",
    )
    runtime_code = repo / "core" / "example.py"
    runtime_code.parent.mkdir(parents=True)
    runtime_code.write_text("VALUE = 1\n", encoding="utf-8")
    backend_runtime = repo / "works" / "CityLearn" / "runtime.py"
    backend_runtime.parent.mkdir(parents=True)
    backend_runtime.write_text("VALUE = 1\n", encoding="utf-8")
    identity = implementation_identity(repo)
    tree = identity["implementation_tree_sha256"]
    pipeline_tree = identity["core_release_pipeline_sha256"]

    release = repo / "release" / "operate_v0_58_0"
    evidence = repo / "release" / "operate_v0_58_0_candidate" / "operate_v058_formal"
    readiness = evidence / "protocol2_v21_core_readiness.json"
    agency = evidence / "agency_readiness_bundle.json"
    diagnostic = evidence / "diagnostic_test_readiness.json"
    pipeline = evidence / "protocol2_v21_pipeline_manifest.json"
    stage_names = dict(batch_llm_eval.FORMAL_CORE_STAGE_FILES)
    stage = evidence / stage_names["behavioral"]
    row = {
        "scenario_id": "fixture/scenario",
        "scenario_signature": "fixture-signature",
        "path": "scenarios/operate_v0_58_0/fixture.yaml",
        "construct_contract": "operational_agency.v1",
    }
    source_suite = release / "protocol21_source_suite.json"
    _write_json(source_suite, {"n_scenarios": 1})
    _write_json(
        readiness,
        {
            "schema_version": "1.0",
            "status": "formal_evaluation_ready",
            "formal_evaluation_ready": True,
            "formal_run_blockers": [],
            "scoring_version": SCORING_VERSION,
            "primary_leaderboard_formula_version": (
                PRIMARY_LEADERBOARD_FORMULA_VERSION
            ),
            "primary_inference_version": (
                "physical_cluster_hierarchical_bootstrap_randomization_v1"
            ),
            "implementation_tree_sha256": tree,
            "core_release_pipeline_sha256": pipeline_tree,
            "suite_manifest_sha256": "a" * 64,
            "n_scenarios": 1,
            "scenarios": [row],
            "formal_run_contract": {
                "contract_version": "agentic_persistent.v1",
                "required_construct_contract": "operational_agency.v1",
            },
        },
    )
    _write_json(agency, {"diagnostic_only": True})
    _write_json(diagnostic, {"diagnostic_only": True})
    stage_artifacts = {}
    pipeline_stages = []
    for stage_name, filename in stage_names.items():
        stage_path = evidence / filename
        if stage_name != "readiness":
            _write_json(
                stage_path,
                {
                    "implementation_tree_sha256": tree,
                    "core_release_pipeline_sha256": pipeline_tree,
                },
            )
        digest = _sha256(stage_path)
        stage_artifacts[stage_name] = {
            "relative_path": filename,
            "sha256": digest,
        }
        pipeline_stages.append(
            {
                "name": stage_name,
                "output_sha256": digest,
                "return_code": 0,
                "implementation_tree_sha256": tree,
                "core_release_pipeline_sha256": pipeline_tree,
            }
        )
    _write_json(
        pipeline,
        {
            "status": "formal_evaluation_ready",
            "implementation_tree_sha256": tree,
            "core_release_pipeline_sha256": pipeline_tree,
            "source_suite_sha256": _sha256(source_suite),
            "stages": pipeline_stages,
        },
    )
    runtime_root = evidence.relative_to(repo).as_posix()
    release_manifest = release / "manifest.json"
    _write_json(
        release_manifest,
        {
            "release_id": "operate_v0_58_0",
            "release_version": "0.58.0",
            "n_scenarios": 1,
            "formal_evaluation_ready": True,
            "implementation_tree_sha256": tree,
            "core_release_pipeline_sha256": pipeline_tree,
            "protocol21_replay": {
                "source_suite": source_suite.relative_to(repo).as_posix(),
                "source_suite_sha256": _sha256(source_suite),
                "core_release_pipeline_sha256": pipeline_tree,
            },
            "pipeline_dir": runtime_root,
            "formal_evidence": {
                "runtime_root": runtime_root,
                "readiness": readiness.relative_to(repo).as_posix(),
            },
            "formal_batch_contract": {
                "runtime_evidence_root": runtime_root,
                "selection_source": (
                    readiness.relative_to(repo).as_posix() + "#scenarios"
                ),
            },
            "pipeline_artifacts": {
                "path": runtime_root,
                "core_release_pipeline_sha256": pipeline_tree,
                "pipeline_manifest_sha256": _sha256(pipeline),
                "readiness_sha256": _sha256(readiness),
                "behavioral_sha256": _sha256(stage),
                "stage_artifacts": stage_artifacts,
            },
        },
    )
    paths = {
        "repo": repo,
        "release": release,
        "release_manifest": release_manifest,
        "evidence": evidence,
        "pipeline": pipeline,
        "readiness": readiness,
        "stage": stage,
    }
    _bind_runtime_closure(paths, source_suite_sha256=_sha256(source_suite))
    return paths


def test_runtime_companion_archives_formal_evidence_for_clean_clone(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    output = paths["repo"] / "data_operate_v058"

    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=output,
        repo_id="Xnhyacinth/OPERATE",
        include_backends=False,
    )
    validate_bundle_archives(output, manifest)

    assert manifest["schema_version"] == "operate-runtime-bundle-v2"
    assert manifest["bundle_kind"] == "private_runtime_companion"
    assert (
        manifest["core_release_pipeline_sha256"]
        == json.loads(paths["release_manifest"].read_text(encoding="utf-8"))[
            "core_release_pipeline_sha256"
        ]
    )
    assert manifest["formal_evidence_archive"] in manifest["files"]
    assert (
        manifest["formal_evidence_install_root"]
        == paths["evidence"].relative_to(paths["repo"]).as_posix()
    )
    assert set(manifest["formal_evidence_required_files"]) == set(
        manifest["formal_evidence_files"]
    )
    assert manifest["formal_evidence_files"] == {
        paths["pipeline"].name: _sha256(paths["pipeline"]),
        **{
            binding["relative_path"]: binding["sha256"]
            for binding in json.loads(
                paths["release_manifest"].read_text(encoding="utf-8")
            )["pipeline_artifacts"]["stage_artifacts"].values()
        },
    }
    assert "agency_input_bindings" not in manifest
    assert verify_manifest(output) == manifest
    card = (output / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\n")
    assert "pretty_name: OPERATE" in card
    assert "`operate_v0_58_0`" in card

    clone = tmp_path / "clean-clone"
    shutil.copytree(paths["repo"] / "core", clone / "core")
    clone_release = clone / "release" / "operate_v0_58_0"
    clone_release.mkdir(parents=True)
    shutil.copy2(paths["release_manifest"], clone_release / "manifest.json")
    shutil.copy2(
        paths["release"] / "protocol21_source_suite.json",
        clone_release / "protocol21_source_suite.json",
    )
    validate_runtime_bundle_compatibility(output, manifest, repo_root=clone)
    assert _extract_formal_evidence_archive(output, manifest, repo_root=clone) is True

    with pytest.raises(ValueError, match="formal backend runtime closure drift"):
        batch_llm_eval.resolve_formal_manifest_slice(
            clone_release / "manifest.json", repo_root=clone
        )
    installed_evidence = clone / manifest["formal_evidence_install_root"]
    assert not (installed_evidence / "diagnostic_test_readiness.json").exists()
    assert not (installed_evidence / "agency_readiness_bundle.json").exists()


def test_runtime_companion_archives_compact_formal_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        verify_release_integrity,
        "_formal_runtime_bundle_valid",
        lambda *_args, **_kwargs: True,
    )
    paths = _fixture(tmp_path)
    release = json.loads(paths["release_manifest"].read_text(encoding="utf-8"))
    install_root = paths["release"].relative_to(paths["repo"]).as_posix()

    core_suite = paths["release"] / "core_suite.json"
    public_evidence = paths["release"] / "protocol21_public_evidence_bundle.json"
    runtime_bundle = paths["release"] / "formal_runtime_bundle.json"
    ordered_identity = "f" * 64
    _write_json(core_suite, {"n_scenarios": 1, "scenarios": [{"scenario_id": "x"}]})
    _write_json(public_evidence, {"schema_version": "fixture-public-evidence-v1"})

    backend_binding = release["backend_runtime_closure"]
    candidate_closure = paths["release"] / "candidate_closure.json"
    shutil.copyfile(
        Path(__file__).parents[1] / "release/operate_v0_61_0/candidate_closure.json",
        candidate_closure,
    )
    candidate_payload = json.loads(candidate_closure.read_text(encoding="utf-8"))
    for raw_bindings in candidate_payload["inputs"].values():
        bindings = raw_bindings if isinstance(raw_bindings, list) else [raw_bindings]
        for index, candidate_input in enumerate(bindings):
            input_path = paths["repo"] / candidate_input["path"]
            input_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json(input_path, {"name": input_path.name, "index": index})
            candidate_input["sha256"] = _sha256(input_path)
    _write_json(candidate_closure, candidate_payload)
    candidate_binding = {
        "schema_version": candidate_payload["schema_version"],
        "path": candidate_closure.name,
        "sha256": _sha256(candidate_closure),
        "status": candidate_payload["status"],
        "n_independent_candidates": candidate_payload["summary"][
            "n_independent_candidates"
        ],
        "n_terminal_candidates": candidate_payload["summary"][
            "n_terminal_candidates"
        ],
        "n_unresolved_candidates": candidate_payload["summary"][
            "n_unresolved_candidates"
        ],
        "identity_set_sha256": candidate_payload["identity_set_sha256"],
    }
    compact = {
        "schema_version": "operate-formal-runtime-bundle-v1",
        "release_id": release["release_id"],
        "status": "formal_evaluation_ready",
        "formal_evaluation_ready": True,
        "formal_run_blockers": [],
        "implementation_tree_sha256": release["implementation_tree_sha256"],
        "core_release_pipeline_sha256": release[
            "core_release_pipeline_sha256"
        ],
        "n_scenarios": 1,
        "ordered_scenario_identity_sha256": ordered_identity,
        "core_suite": {
            "path": core_suite.name,
            "sha256": _sha256(core_suite),
            "n_scenarios": 1,
            "ordered_scenario_identity_sha256": ordered_identity,
        },
        "source_suite": {
            "path": paths["release"].joinpath("protocol21_source_suite.json").name,
            "sha256": _sha256(
                paths["release"] / "protocol21_source_suite.json"
            ),
            "n_scenarios": 1,
            "ordered_scenario_identity_sha256": ordered_identity,
        },
        "public_evidence": {
            "path": public_evidence.name,
            "sha256": _sha256(public_evidence),
        },
        "backend_runtime_closure": {
            "path": backend_binding["path"],
            "sha256": backend_binding["sha256"],
            "identity_sha256": backend_binding["identity_sha256"],
        },
        "candidate_closure": {
            "path": candidate_binding["path"],
            "sha256": candidate_binding["sha256"],
            "identity_set_sha256": candidate_binding["identity_set_sha256"],
        },
    }
    _write_json(runtime_bundle, compact)

    release["core_suite"] = compact["core_suite"]
    release["candidate_closure"] = candidate_binding
    release["formal_runtime_bundle"] = {
        "path": runtime_bundle.name,
        "sha256": _sha256(runtime_bundle),
        "schema_version": compact["schema_version"],
        "n_scenarios": 1,
        "ordered_scenario_identity_sha256": ordered_identity,
        "size_bytes": runtime_bundle.stat().st_size,
    }
    release["formal_batch_contract"].update(
        runtime_evidence_root=install_root,
        selection_source=f"{install_root}/{runtime_bundle.name}#scenarios",
    )
    release["formal_realtime_batch_contract"] = {
        "selection_source": f"{install_root}/{runtime_bundle.name}#scenarios"
    }
    release["formal_evidence"] = {
        "runtime_root": install_root,
        "readiness": f"{install_root}/{runtime_bundle.name}",
    }
    model = "provider/model"
    logical_binding = _formal_result_binding(
        repo=paths["repo"],
        release=paths["release"],
        model=model,
        treatment="1" * 64,
        mode="logical_persistent",
    )
    realtime_binding = _formal_result_binding(
        repo=paths["repo"],
        release=paths["release"],
        model=model,
        treatment="2" * 64,
        mode="realtime_persistent",
    )
    release["formal_evidence"].update(
        logical_batch_manifest=logical_binding,
        realtime_batch_manifest=realtime_binding,
    )
    release["protocol21_replay"].update(
        evidence_bundle=public_evidence.name,
        evidence_bundle_sha256=_sha256(public_evidence),
    )
    _write_json(paths["release_manifest"], release)

    output = paths["repo"] / "compact-bundle"
    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=output,
        include_backends=False,
    )

    assert manifest["formal_evidence_install_root"] == install_root
    base_files = {
        "backend_runtime_closure.json": backend_binding["sha256"],
        "candidate_closure.json": candidate_binding["sha256"],
        "core_suite.json": _sha256(core_suite),
        "formal_runtime_bundle.json": _sha256(runtime_bundle),
        "protocol21_public_evidence_bundle.json": _sha256(public_evidence),
        "protocol21_source_suite.json": _sha256(
            paths["release"] / "protocol21_source_suite.json"
        ),
    }
    formal_result_files = {
        path.relative_to(paths["release"]).as_posix(): _sha256(path)
        for path in (paths["release"] / "formal_results").rglob("*")
        if path.is_file()
    }
    assert manifest["formal_evidence_files"] == {
        **base_files,
        **formal_result_files,
    }
    assert "candidate_evidence_archive" not in manifest
    validate_bundle_archives(output, manifest)
    download_from_hf.validate_bundle_distribution_contract(output, manifest)

    clone = tmp_path / "compact-clone"
    shutil.copytree(paths["repo"] / "core", clone / "core")
    shutil.copytree(paths["release"], clone / install_root)
    shutil.rmtree(clone / install_root / "formal_results")
    assert _extract_formal_evidence_archive(
        output,
        manifest,
        repo_root=clone,
    ) is True
    assert all(
        _sha256(clone / install_root / relative) == digest
        for relative, digest in formal_result_files.items()
    )
    assert _extract_formal_evidence_archive(
        output,
        manifest,
        repo_root=clone,
    ) is False
    assert (clone / install_root / "manifest.json").is_file()
    assert not (clone / install_root / ".archive.sha256").exists()

    installed_core = clone / install_root / "core_suite.json"
    installed_core.unlink()
    assert _extract_formal_evidence_archive(
        output,
        manifest,
        repo_root=clone,
    ) is True
    assert _sha256(installed_core) == manifest["formal_evidence_files"][
        "core_suite.json"
    ]
    assert (clone / install_root / "manifest.json").is_file()

    unlisted = clone / install_root / "README.md"
    unlisted.write_text("keep me\n", encoding="utf-8")
    installed_core.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="formal_evidence_target_exists"):
        _extract_formal_evidence_archive(output, manifest, repo_root=clone)
    assert unlisted.read_text(encoding="utf-8") == "keep me\n"
    assert _extract_formal_evidence_archive(
        output,
        manifest,
        repo_root=clone,
        force=True,
    ) is True
    assert _sha256(installed_core) == manifest["formal_evidence_files"][
        "core_suite.json"
    ]
    assert unlisted.read_text(encoding="utf-8") == "keep me\n"
    assert (clone / install_root / "manifest.json").is_file()
    assert not (clone / install_root / ".archive.sha256").exists()

    matching_core = clone / "matching-core-suite.json"
    shutil.copy2(installed_core, matching_core)
    installed_core.unlink()
    installed_core.symlink_to(matching_core)
    with pytest.raises(ValueError, match="formal_evidence_target_exists"):
        _extract_formal_evidence_archive(output, manifest, repo_root=clone)

    installed_core.unlink()
    dangling_target = clone / "missing-core-suite.json"
    installed_core.symlink_to(dangling_target)
    monkeypatch.setattr(
        verify_release_integrity,
        "_formal_runtime_bundle_valid",
        lambda *_args, **_kwargs: False,
    )
    with pytest.raises(
        ValueError,
        match="formal_evidence_post_install_validation_failed",
    ):
        _extract_formal_evidence_archive(
            output,
            manifest,
            repo_root=clone,
            force=True,
        )
    assert installed_core.is_symlink()
    assert installed_core.readlink() == dangling_target


def test_runtime_companion_parameterizes_release_identity(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    release = paths["repo"] / "release/operate_v0_59_0"
    shutil.copytree(paths["release"], release)
    release_manifest_path = release / "manifest.json"
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    release_manifest["release_id"] = "operate_v0_59_0"
    release_manifest["release_version"] = "0.59.0"
    release_manifest["protocol21_replay"]["source_suite"] = (
        "release/operate_v0_59_0/protocol21_source_suite.json"
    )
    _write_json(release_manifest_path, release_manifest)
    output = paths["repo"] / "data_operate_v059"

    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=release,
        output_dir=output,
        include_backends=False,
    )

    assert manifest["release_id"] == "operate_v0_59_0"
    assert manifest["release_version"] == "0.59.0"
    assert manifest["formal_evidence_archive"] == (
        "operate_v0_59_0_formal_evidence.tar.zst"
    )


def test_runtime_companion_parameterizes_backend_closure_identity(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    release = paths["repo"] / "release/operate_v0_59_0"
    shutil.copytree(paths["release"], release)
    release_manifest_path = release / "manifest.json"
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    release_manifest["release_id"] = "operate_v0_59_0"
    release_manifest["release_version"] = "0.59.0"
    release_manifest["protocol21_replay"]["source_suite"] = (
        "release/operate_v0_59_0/protocol21_source_suite.json"
    )
    source_suite_path = release / "protocol21_source_suite.json"
    _write_json(source_suite_path, {"n_scenarios": 0, "scenarios": []})
    release_manifest["protocol21_replay"]["source_suite_sha256"] = _sha256(
        source_suite_path
    )
    runtime_path = release / "backend_runtime_closure.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["release_id"] = "operate_v0_59_0"
    runtime["source_suite_sha256"] = _sha256(source_suite_path)
    runtime["identity_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in runtime.items() if key != "identity_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_json(runtime_path, runtime)
    release_manifest["backend_runtime_closure"].update(
        {
            "sha256": _sha256(runtime_path),
            "identity_sha256": runtime["identity_sha256"],
        }
    )
    _write_json(release_manifest_path, release_manifest)
    output = paths["repo"] / "data_operate_v059"

    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=release,
        output_dir=output,
        include_backends=True,
    )

    assert manifest["backend_archive"] == "operate_v0_59_0_backends.tar.zst"
    assert manifest["backend_runtime_closure"]["identity_sha256"] == runtime[
        "identity_sha256"
    ]


def test_runtime_companion_rejects_incomplete_core_evidence_closure(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    output = paths["repo"] / "data_operate_v058"
    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=output,
        include_backends=False,
    )
    stage_name = paths["stage"].name
    del manifest["formal_evidence_files"][stage_name]
    manifest["formal_evidence_required_files"].remove(stage_name)
    _write_json(output / "MANIFEST.json", manifest)

    with pytest.raises(ValueError, match="formal_evidence_file_contract_invalid"):
        validate_runtime_bundle_compatibility(
            output,
            manifest,
            repo_root=paths["repo"],
        )


def test_runtime_companion_rejects_deprecated_agency_bindings(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    output = paths["repo"] / "data_operate_v058"
    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=output,
        include_backends=False,
    )
    manifest["agency_input_bindings"] = {}
    _write_json(output / "MANIFEST.json", manifest)

    with pytest.raises(ValueError, match="deprecated_agency_input_bindings_forbidden"):
        validate_runtime_bundle_compatibility(
            output,
            manifest,
            repo_root=paths["repo"],
        )


@pytest.mark.parametrize(
    "location",
    ("bundle", "pipeline_artifacts", "protocol21_replay", "non_hex"),
)
def test_runtime_companion_rejects_core_pipeline_bundle_drift(
    tmp_path: Path,
    location: str,
) -> None:
    paths = _fixture(tmp_path)
    output = paths["repo"] / "data_operate_v058"
    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=output,
        include_backends=False,
    )
    if location == "bundle":
        manifest["core_release_pipeline_sha256"] = "0" * 64
    else:
        release = json.loads(paths["release_manifest"].read_text(encoding="utf-8"))
        if location == "non_hex":
            invalid_hash = "z" * 64
            manifest["core_release_pipeline_sha256"] = invalid_hash
            release["core_release_pipeline_sha256"] = invalid_hash
            release["pipeline_artifacts"]["core_release_pipeline_sha256"] = invalid_hash
            release["protocol21_replay"]["core_release_pipeline_sha256"] = invalid_hash
        else:
            release[location]["core_release_pipeline_sha256"] = "0" * 64
        _write_json(paths["release_manifest"], release)
        _write_json(output / "release_manifest.json", release)
        release_digest = _sha256(output / "release_manifest.json")
        manifest["release_manifest_sha256"] = release_digest
        manifest["files"]["release_manifest.json"] = release_digest
    _write_json(output / "MANIFEST.json", manifest)

    with pytest.raises(ValueError, match="runtime_bundle_release_manifest_invalid"):
        validate_runtime_bundle_compatibility(
            output,
            manifest,
            repo_root=paths["repo"],
        )


def test_runtime_companion_rejects_missing_formal_evidence_archive(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    output = paths["repo"] / "data_operate_v058"
    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=output,
        include_backends=False,
    )
    del manifest["formal_evidence_archive"]
    _write_json(output / "MANIFEST.json", manifest)

    with pytest.raises(ValueError, match="formal_evidence_archive_missing"):
        validate_runtime_bundle_compatibility(
            output,
            manifest,
            repo_root=paths["repo"],
        )
    with pytest.raises(ValueError, match="formal_evidence_archive_missing"):
        _extract_formal_evidence_archive(
            output,
            manifest,
            repo_root=paths["repo"],
        )


def test_runtime_companion_rejects_local_release_manifest_mismatch(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    output = paths["repo"] / "data_operate_v058"
    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=output,
        include_backends=False,
    )

    clone = tmp_path / "clean-clone"
    shutil.copytree(paths["repo"] / "core", clone / "core")
    clone_release = clone / "release" / "operate_v0_58_0"
    clone_release.mkdir(parents=True)
    shutil.copy2(paths["release_manifest"], clone_release / "manifest.json")
    (clone_release / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local_release_manifest_mismatch"):
        validate_runtime_bundle_compatibility(output, manifest, repo_root=clone)


def test_runtime_companion_upload_accepts_valid_staged_candidate_manifest(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    candidate = json.loads(paths["release_manifest"].read_text(encoding="utf-8"))
    candidate["public_release_ready"] = True
    candidate_path = paths["repo"] / ".hl" / "candidate_manifest.json"
    _write_json(candidate_path, candidate)
    output = paths["repo"] / "candidate-bundle"
    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        release_manifest_path=candidate_path,
        output_dir=output,
        include_backends=False,
    )

    validate_runtime_bundle_compatibility(
        output,
        manifest,
        repo_root=paths["repo"],
        require_canonical_release_manifest=False,
    )


def test_runtime_companion_rejects_live_runtime_drift(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = paths["repo"] / "data_operate_v058"
    manifest = build_operate_bundle(
        repo_root=paths["repo"],
        release_dir=paths["release"],
        output_dir=output,
        include_backends=False,
    )
    clone = tmp_path / "clean-clone"
    shutil.copytree(paths["repo"] / "core", clone / "core")
    clone_release = clone / "release" / "operate_v0_58_0"
    clone_release.mkdir(parents=True)
    shutil.copy2(paths["release_manifest"], clone_release / "manifest.json")
    (clone / "core" / "example.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local_implementation_tree_mismatch"):
        validate_runtime_bundle_compatibility(output, manifest, repo_root=clone)


def test_runtime_companion_builder_rejects_release_tree_drift(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    (paths["repo"] / "core" / "example.py").write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="release_implementation_tree_mismatch"):
        build_operate_bundle(
            repo_root=paths["repo"],
            release_dir=paths["release"],
            output_dir=paths["repo"] / "data_operate_v058",
            include_backends=False,
        )


def test_runtime_companion_builder_rejects_admission_tooling_drift(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    script = paths["repo"] / "scripts" / "calibrate_core_candidate.py"
    script.parent.mkdir(parents=True)
    script.write_text("SUMO_WORKERS = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="release_core_pipeline_mismatch"):
        build_operate_bundle(
            repo_root=paths["repo"],
            release_dir=paths["release"],
            output_dir=paths["repo"] / "data_operate_v058",
            include_backends=False,
        )


def test_full_install_requires_repo_owned_bundle_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    validate_install_data_dir(repo / "data_operate_v058", repo_root=repo)
    with pytest.raises(ValueError, match="full_install_data_dir_outside_repo"):
        validate_install_data_dir(tmp_path / "external", repo_root=repo)


def test_runtime_companion_rejects_nonportable_formal_evidence(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    stage = paths["stage"]
    _write_json(
        stage,
        {
            "implementation_tree_sha256": implementation_identity(paths["repo"])[
                "implementation_tree_sha256"
            ],
            "core_release_pipeline_sha256": implementation_identity(paths["repo"])[
                "core_release_pipeline_sha256"
            ],
            "nested": {str(paths["repo"] / "sources" / "input.csv"): "leak"},
        },
    )
    release_manifest = json.loads(paths["release_manifest"].read_text())
    release_manifest["pipeline_artifacts"]["behavioral_sha256"] = _sha256(stage)
    release_manifest["pipeline_artifacts"]["stage_artifacts"]["behavioral"][
        "sha256"
    ] = _sha256(stage)
    pipeline = json.loads(paths["pipeline"].read_text(encoding="utf-8"))
    next(row for row in pipeline["stages"] if row["name"] == "behavioral")[
        "output_sha256"
    ] = _sha256(stage)
    _write_json(paths["pipeline"], pipeline)
    release_manifest["pipeline_artifacts"]["pipeline_manifest_sha256"] = _sha256(
        paths["pipeline"]
    )
    _write_json(paths["release_manifest"], release_manifest)
    with pytest.raises(ValueError, match="formal_evidence_nonportable_path"):
        build_operate_bundle(
            repo_root=paths["repo"],
            release_dir=paths["release"],
            output_dir=paths["repo"] / "data_operate_v058",
            include_backends=False,
        )


def test_fresh_clone_fetches_and_verifies_pinned_clusterdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "clean-clone"
    assets = {
        "README.md": b"root readme\n",
        "cluster-trace-v2026-spot-gpu/job_info_df.csv": b"job\n",
        "cluster-trace-v2026-spot-gpu/node_info_df.csv": b"node\n",
        "cluster-trace-v2026-spot-gpu/README.md": b"trace readme\n",
    }
    monkeypatch.setattr(
        download_from_hf,
        "CLUSTERDATA_EXPECTED_ASSETS",
        {
            relative: hashlib.sha256(payload).hexdigest()
            for relative, payload in assets.items()
        },
    )
    calls: list[tuple[str, ...]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        calls.append(tuple(command))
        if command[1] == "clone":
            checkout = Path(command[-1])
            assert checkout != repo / "works" / "clusterdata"
            assert checkout.parent.name.startswith(".clusterdata-fetch-")
            for relative, payload in assets.items():
                path = checkout / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            return SimpleNamespace(returncode=0, stdout="")
        if "rev-parse" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=download_from_hf.CLUSTERDATA_EXPECTED_COMMIT + "\n",
            )
        if "get-url" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=download_from_hf.CLUSTERDATA_URL + "\n",
            )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(download_from_hf.subprocess, "run", fake_run)

    report = download_from_hf.ensure_clusterdata_sources(repo)

    assert report["status"] == "verified"
    assert report["observed_commit"] == download_from_hf.CLUSTERDATA_EXPECTED_COMMIT
    assert any(command[1] == "clone" for command in calls)
    assert (repo / "works" / "clusterdata").is_dir()
    assert list((repo / "works").glob(".clusterdata-fetch-*")) == []


def test_interrupted_clusterdata_fetch_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "clean-clone"

    def fake_run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        if command[1] == "clone":
            Path(command[-1]).mkdir(parents=True)
            return SimpleNamespace(returncode=0, stdout="")
        raise download_from_hf.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(download_from_hf.subprocess, "run", fake_run)

    with pytest.raises(download_from_hf.subprocess.CalledProcessError):
        download_from_hf.ensure_clusterdata_sources(repo)

    assert not (repo / "works" / "clusterdata").exists()
    assert list((repo / "works").glob(".clusterdata-fetch-*")) == []
