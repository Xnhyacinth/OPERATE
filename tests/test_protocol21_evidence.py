import json
import shutil
from pathlib import Path

import pytest

from core.protocol21_evidence import (
    artifact_binding,
    canonicalize_repo_owned_paths,
    required_semantics,
    validate_report_scope,
    verify_artifact_binding,
)


def test_recursive_repo_path_canonicalization_is_lexical_and_covers_keys(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "clone"
    repo_root.mkdir()
    absolute = repo_root / "release" / "evidence.json"
    outside = tmp_path / "external" / "source.json"
    payload = {
        str(absolute): {
            "path": str(repo_root / "scenarios" / "case.yaml"),
            "outside": str(outside),
        },
        "rows": [str(repo_root / "sources" / "input.csv")],
    }

    canonical = canonicalize_repo_owned_paths(payload, repo_root=repo_root)

    assert canonical == {
        "release/evidence.json": {
            "path": "scenarios/case.yaml",
            "outside": str(outside),
        },
        "rows": ["sources/input.csv"],
    }


def test_recursive_repo_path_canonicalization_covers_embedded_diagnostics(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "clone"
    repo_root.mkdir()
    outside = tmp_path / "external" / "source.json"
    payload = {
        "detail": f"Missing {repo_root / 'works' / 'source'}",
        "traceback": f'File "{repo_root / ".venv" / "module.py"}", line 4',
        "outside": f"External source remains at {outside}",
    }

    canonical = canonicalize_repo_owned_paths(payload, repo_root=repo_root)

    assert canonical == {
        "detail": "Missing works/source",
        "traceback": 'File ".venv/module.py", line 4',
        "outside": f"External source remains at {outside}",
    }


def test_recursive_repo_path_canonicalization_rejects_conflicting_keys(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "clone"
    repo_root.mkdir()

    with pytest.raises(ValueError, match="canonical path key collision"):
        canonicalize_repo_owned_paths(
            {
                str(repo_root / "release" / "evidence.json"): "first",
                "release/evidence.json": "second",
            },
            repo_root=repo_root,
        )


def _report(tree: str) -> dict:
    return {
        "status": "complete",
        "evaluation_semantics": required_semantics(),
        "implementation_tree_sha256": tree,
        "n_expected": 1,
        "n_completed": 1,
        "results": [
            {
                "scenario_id": "scenario",
                "scenario_signature": "signature",
            }
        ],
    }


def test_artifact_binding_detects_byte_and_tree_drift(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report("tree")), encoding="utf-8")
    binding = artifact_binding(path, implementation_tree_sha256="tree")

    assert verify_artifact_binding(
        binding,
        path,
        implementation_tree_sha256="tree",
    ) == []
    path.write_text(json.dumps({**_report("tree"), "changed": True}), encoding="utf-8")
    errors = verify_artifact_binding(
        binding,
        path,
        implementation_tree_sha256="tree",
    )
    assert "artifact_hash_mismatch" in errors


def test_repo_relative_artifact_binding_survives_clone_root_change(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old-clone"
    new_root = tmp_path / "new-clone"
    old_path = old_root / "release" / "report.json"
    old_path.parent.mkdir(parents=True)
    old_path.write_text(json.dumps(_report("tree")), encoding="utf-8")

    binding = artifact_binding(
        old_path,
        implementation_tree_sha256="tree",
        repo_root=old_root,
    )
    assert binding["path"] == "release/report.json"

    new_path = new_root / "release" / "report.json"
    new_path.parent.mkdir(parents=True)
    shutil.copyfile(old_path, new_path)
    assert verify_artifact_binding(
        binding,
        new_path,
        implementation_tree_sha256="tree",
        repo_root=new_root,
    ) == []


def test_scope_requires_exact_three_complexity_agents() -> None:
    tree = "tree"
    identity = ("scenario", "signature")
    report = {
        "status": "complete",
        "evaluation_semantics": required_semantics(),
        "implementation_tree_sha256": tree,
        "n_expected": 3,
        "n_completed": 3,
        "results": [
            {
                "scenario_id": identity[0],
                "scenario_signature": identity[1],
                "agent_name": agent,
            }
            for agent in ("oracle_offline", "greedy_heuristic", "wait_only")
        ],
    }

    assert validate_report_scope(
        report,
        [identity],
        implementation_tree_sha256=tree,
        complexity_agents=(
            "oracle_offline",
            "greedy_heuristic",
            "wait_only",
        ),
    ) == []
    report["results"][2]["agent_name"] = "oracle_offline"
    assert "complexity_agent_scope_mismatch" in validate_report_scope(
        report,
        [identity],
        implementation_tree_sha256=tree,
        complexity_agents=(
            "oracle_offline",
            "greedy_heuristic",
            "wait_only",
        ),
    )
