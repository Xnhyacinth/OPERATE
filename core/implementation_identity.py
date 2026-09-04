"""Stable identity for the code that implements protocol-2.1 evaluation."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_CODE_ROOTS = (
    "core",
    "runner",
    "evaluation",
    "domains",
    "baselines",
    "data",
)
_RUNTIME_SCRIPT_PATHS = (
    "scripts/analyze_batch_results.py",
    "scripts/analyze_decision_impact.py",
    "scripts/analyze_single_model_core_run.py",
    "scripts/audit_agent_failure_recipes.py",
    "scripts/audit_eval_logs.py",
    "scripts/audit_evidence_applicability.py",
    "scripts/audit_protocol21_diagnostic_test_readiness.py",
    "scripts/audit_staleness_consumption.py",
    "scripts/audit_tool_effects.py",
    "scripts/batch_llm_eval.py",
    "scripts/batch_realtime_llm_eval.py",
    "scripts/build_operational_agency_readiness_bundle.py",
    "scripts/build_protocol21_core_readiness.py",
    "scripts/merge_formal_llm_shards.py",
    "scripts/run_operational_agency_known_groups_calibration.py",
)
_CORE_RELEASE_PIPELINE_SCRIPT_PATHS = (
    "scripts/run_protocol21_core_pipeline.py",
    "scripts/preflight_protocol21_working_set.py",
    "scripts/calibrate_core_candidate.py",
    "scripts/audit_protocol21_source_consumption.py",
    "scripts/calibrate_task_contracts.py",
    "scripts/calibrate_core_complexity.py",
    "scripts/audit_observed_reference_depth.py",
    "scripts/audit_strategy_depth_calibration.py",
    "scripts/audit_source_grounded_pipeline.py",
    "scripts/audit_protocol21_core_contract.py",
    "scripts/materialize_protocol2_core.py",
    "scripts/audit_protocol21_release_coverage.py",
    "scripts/build_protocol21_core_readiness.py",
    "scripts/build_primary_suite.py",
)
_EXCLUDED_PARTS = {
    "__pycache__",
    ".audit-cache",
    "release",
    "batch_results",
    "reports",
    "tests",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _runtime_code_files(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for root_name in _RUNTIME_CODE_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        paths.extend(
            path
            for path in root.rglob("*.py")
            if not _EXCLUDED_PARTS.intersection(path.relative_to(repo_root).parts)
            and path.relative_to(repo_root).parts[:2] != ("data", "backends")
        )
    run_py = repo_root / "run.py"
    if run_py.is_file():
        paths.append(run_py)
    paths.extend(
        path
        for relative in _RUNTIME_SCRIPT_PATHS
        if (path := repo_root / relative).is_file()
    )
    return sorted(set(paths), key=lambda path: path.relative_to(repo_root).as_posix())


def _release_tooling_files(repo_root: Path) -> list[Path]:
    runtime_scripts = set(_RUNTIME_SCRIPT_PATHS)
    scripts_root = repo_root / "scripts"
    if not scripts_root.is_dir():
        return []
    return sorted(
        (
            path
            for path in scripts_root.rglob("*.py")
            if path.relative_to(repo_root).as_posix() not in runtime_scripts
            and not _EXCLUDED_PARTS.intersection(path.relative_to(repo_root).parts)
        ),
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )


def _core_release_pipeline_files(
    repo_root: Path,
    runtime_files: list[Path],
) -> list[Path]:
    paths = list(runtime_files)
    audit_root = repo_root / "audit"
    if audit_root.is_dir():
        paths.extend(
            path
            for path in audit_root.rglob("*.py")
            if not _EXCLUDED_PARTS.intersection(path.relative_to(repo_root).parts)
        )
    paths.extend(
        path
        for relative in _CORE_RELEASE_PIPELINE_SCRIPT_PATHS
        if (path := repo_root / relative).is_file()
    )
    return sorted(
        set(paths),
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )


def _files_sha256(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_output(repo_root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return completed.stdout if completed.returncode == 0 else b""


def implementation_identity(
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Return separate runtime-semantics and release-tooling identities."""
    root = repo_root.resolve()
    files = _runtime_code_files(root)
    tooling_files = _release_tooling_files(root)
    core_pipeline_files = _core_release_pipeline_files(root, files)
    runtime_paths = (*_RUNTIME_CODE_ROOTS, *_RUNTIME_SCRIPT_PATHS, "run.py")
    runtime_sha256 = _files_sha256(root, files)
    return {
        # Compatibility name retained while manifests migrate to the explicit
        # runtime-semantics field.
        "implementation_tree_sha256": runtime_sha256,
        "evaluation_runtime_sha256": runtime_sha256,
        "core_release_pipeline_sha256": _files_sha256(root, core_pipeline_files),
        "release_tooling_sha256": _files_sha256(root, tooling_files),
        "git_head": _git_output(root, "rev-parse", "HEAD").decode().strip(),
        "tracked_diff_sha256": _sha256(
            _git_output(root, "diff", "--binary", "--", *runtime_paths)
        ),
        "n_code_files": len(files),
        "n_core_release_pipeline_files": len(core_pipeline_files),
        "n_release_tooling_files": len(tooling_files),
    }
