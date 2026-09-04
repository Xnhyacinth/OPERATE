from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path

import __init__ as package_metadata

from scripts.verify_release_integrity import DEFAULT_RELEASE
from scripts.promote_operate_release import DEFAULT_PARENT
from scripts.summarize_leaderboard_results import (
    DEFAULT_RELEASE as SUMMARY_DEFAULT_RELEASE,
    DEFAULT_OUTPUT_JSON,
    DEFAULT_OUTPUT_MARKDOWN,
    DEFAULT_SUMMARY_CSV,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_clis_start_from_external_cwd_without_install(
    tmp_path: Path,
) -> None:
    dependency_path = sysconfig.get_paths()["purelib"]
    for script_name in (
        "build_operate_candidate_source_metadata.py",
        "build_operate_bundle.py",
        "build_works_candidate_inventory.py",
        "finalize_operate_candidate_pool.py",
        "promote_operate_release.py",
        "refine_datacenter_archive_candidate_pool.py",
        "refine_infrastructure_candidate_pool.py",
        "refine_logistics_manufacturing_candidate_pool.py",
        "summarize_leaderboard_results.py",
        "verify_release_integrity.py",
    ):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(REPO_ROOT / "scripts" / script_name),
                "--help",
            ],
            cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": dependency_path},
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_public_bundle_cli_starts_from_external_cwd_without_install(
    tmp_path: Path,
) -> None:
    dependency_path = sysconfig.get_paths()["purelib"]
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            str(REPO_ROOT / "tools" / "download_public_bundle.py"),
            "--help",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": dependency_path},
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_default_release_is_anchored_to_repository() -> None:
    assert DEFAULT_RELEASE == REPO_ROOT / "release" / "operate_v0_61_0"
    assert DEFAULT_RELEASE.is_absolute()
    assert SUMMARY_DEFAULT_RELEASE == DEFAULT_RELEASE
    assert "operate_v0_61_0" in DEFAULT_SUMMARY_CSV.parts
    assert DEFAULT_OUTPUT_JSON.name == "operate_v061_leaderboard_results.json"
    assert DEFAULT_OUTPUT_MARKDOWN.name == "operate_v061_leaderboard_results.md"
    assert DEFAULT_PARENT == REPO_ROOT / "release" / "operate_v0_60_0" / "manifest.json"


def test_package_version_matches_project_metadata() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    assert package_metadata.__version__ == project["project"]["version"]


def test_operate_data_bundles_are_gitignored() -> None:
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "--quiet",
            "operate_data/runtime.bin",
        ],
        cwd=REPO_ROOT,
        check=False,
    )

    assert result.returncode == 0


def test_fresh_clone_entrypoints_target_current_release() -> None:
    setup = (REPO_ROOT / "scripts" / "setup_eval_env.sh").read_text(
        encoding="utf-8"
    )
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    downloader = (REPO_ROOT / "scripts" / "download_from_hf.py").read_text(
        encoding="utf-8"
    )

    assert "release/operate_v0_61_0/manifest.json" in setup
    assert "scripts/prepare_local_source_locks.py" in setup
    assert "release/operate_v0_61_0" in ci
    assert "operate_v0_61_0/" in dockerfile
    assert 'DATA = REPO / "operate_data"' in downloader
    assert "data_operate_v058" not in downloader
    data_readme = (REPO_ROOT / "data" / "README.md").read_text(encoding="utf-8")
    assert "scripts/download_from_hf.py --download-only" in data_readme
    assert "operate_data/" in data_readme
