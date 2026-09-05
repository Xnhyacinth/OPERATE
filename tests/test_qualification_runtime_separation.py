"""Historical qualification stays immutable while fresh runs bind current code."""

import json
import shutil

import pytest

from core.implementation_identity import implementation_identity
from scripts import batch_llm_eval as logical
from scripts import finalize_operate_release as finalizer
from scripts import verify_release_integrity as verifier
from tests.test_batch_llm_eval import _write_green_formal_gate
from tests.test_verify_release_integrity import _promoted_closure_release, _published_formal_fixture


def _change_runtime(repo):
    path = repo / "core" / "maintenance_fix.py"
    path.parent.mkdir(exist_ok=True)
    path.write_text("MAINTENANCE_FIX = True\n")


def test_new_run_uses_current_tree_without_rewriting_qualification(tmp_path):
    repo = tmp_path / "repo"
    paths = _write_green_formal_gate(repo)
    original = {path: path.read_bytes() for path in paths.values() if path.is_file()}
    old_tree = implementation_identity(repo)["implementation_tree_sha256"]
    _change_runtime(repo)
    current_tree = implementation_identity(repo)["implementation_tree_sha256"]
    assert current_tree != old_tree

    binding = logical.resolve_formal_manifest_slice(paths["manifest"], repo_root=repo)
    old_treatment = logical._formal_agent_treatment_hashes(
        {"model": "a" * 64}, formal_manifest_binding=binding,
        implementation_tree_sha256=old_tree,
    )
    new_treatment = logical._formal_agent_treatment_hashes(
        {"model": "a" * 64}, formal_manifest_binding=binding,
        implementation_tree_sha256=current_tree,
    )
    assert old_treatment != new_treatment
    assert all(path.read_bytes() == body for path, body in original.items())
    paths["backend_file"].write_bytes(b"unlocked source")
    with pytest.raises(ValueError, match="backend runtime closure drift"):
        logical.resolve_formal_manifest_slice(paths["manifest"], repo_root=repo)


def test_verifier_retains_qualification_but_reports_new_runtime(tmp_path, monkeypatch):
    release, repo = _promoted_closure_release(tmp_path, monkeypatch)
    original = (release / "manifest.json").read_bytes()
    qualification = json.loads(original)["implementation_tree_sha256"]
    _change_runtime(repo)
    report = verifier.build_release_integrity_report(release, artifact_root=repo)
    assert report["checks"]["agentic_implementation_tree_binding_valid"] is True
    assert report["checks"]["agentic_release_pipeline_binding_valid"] is True
    assert report["diagnostics"]["qualification_implementation_tree_sha256"] == qualification
    assert report["diagnostics"]["live_implementation_matches_qualification"] is False
    assert report["runtime_evidence_implementation_tree_sha256"] == qualification
    assert (release / "manifest.json").read_bytes() == original


def test_compact_release_accepts_new_code_but_not_changed_qualification(tmp_path, monkeypatch):
    from tests.test_promote_operate_release import (
        _fixture, _bind_parent_release, _refresh_backend_runtime_closure, promote_release,
    )

    paths = _fixture(tmp_path)
    _bind_parent_release(paths)
    release = paths["repo"] / "release/operate_v0_59_0"
    release.mkdir(parents=True)
    _refresh_backend_runtime_closure(paths, release_id=release.name)
    monkeypatch.setattr("scripts.promote_operate_release.verify_scenario_row_against_yaml",
                        lambda row, path: [])
    promote_release(
        repo_root=paths["repo"], parent_manifest_path=paths["parent"],
        source_suite_path=paths["source_suite"], candidate_closure_path=paths["candidate_closure"],
        backend_runtime_closure_path=paths["backend_runtime_closure"],
        pipeline_dir=paths["pipeline"], output_dir=release, build_public_evidence=True,
        release_id=release.name, release_version="0.59.0",
        selection_policy="quality_core_v2_v059", core_settings_stamp="v0.59.0-settings",
    )
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    monkeypatch.setattr(logical, "validate_live_backend_runtime_closure", lambda **k: {
        "identity_sha256": manifest["backend_runtime_closure"]["identity_sha256"],
    })
    _change_runtime(paths["repo"])
    binding = logical.resolve_formal_manifest_slice(manifest_path, repo_root=paths["repo"])
    assert binding["formal_runtime_bundle_sha256"] == manifest["formal_runtime_bundle"]["sha256"]
    manifest["implementation_tree_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="formal runtime bundle integrity mismatch"):
        logical.resolve_formal_manifest_slice(manifest_path, repo_root=paths["repo"])


def test_finalizer_keeps_old_proof_and_binds_both_new_result_trees(tmp_path, monkeypatch):
    release, manifest = _published_formal_fixture(tmp_path)
    repo = release.parents[1]
    proof_tree = manifest["implementation_tree_sha256"]
    new_tree = "9" * 64
    batches = []
    for name in ("logical_batch_manifest", "realtime_batch_manifest"):
        source = repo / manifest["formal_evidence"][name]["path"]
        destination = repo / "batches" / name
        shutil.copytree(source.parent, destination)
        (destination / finalizer.TREE_INDEX_NAME).unlink()
        batches.append(destination / "RUN_MANIFEST.json")
    manifest["status"] = "formal_evaluation_ready"
    manifest["formal_evidence"] = {}
    manifest.pop("formal_evaluation_completion")
    for name in ("formal_batch_contract", "formal_run_contract", "formal_realtime_batch_contract"):
        manifest[name] = {}
    path = release / "manifest.json"
    path.write_text(json.dumps(manifest))
    original = path.read_bytes()
    manifest_sha = finalizer._file_sha256(path)
    logical_payload = json.loads(batches[0].read_text())
    logical_payload.update(implementation_tree_sha256=new_tree,
                           formal_manifest_sha256=manifest_sha,
                           formal_release_id=release.name)
    batches[0].write_text(json.dumps(logical_payload))
    realtime = json.loads(batches[1].read_text())
    realtime["batch_treatment_identity"].update(
        implementation_tree_sha256=new_tree, formal_manifest_sha256=manifest_sha,
    )
    realtime["batch_treatment_identity"]["formal_runtime_binding"]["release_id"] = release.name
    batches[1].write_text(json.dumps(realtime))
    # Scope this fixture to finalization identity/materialization; scientific
    # episode validation has its own full regressions and remains unchanged.
    monkeypatch.setattr(finalizer, "_validate_pending_release", lambda *a, **k: None)
    monkeypatch.setattr(finalizer, "_release_rows", lambda *a, **k: ({}, [{}]))
    monkeypatch.setattr(finalizer, "_logical_published_manifest_valid", lambda payload, **k:
                        payload["implementation_tree_sha256"] == k["tree"] == new_tree)
    monkeypatch.setattr(finalizer, "_realtime_published_manifest_valid", lambda payload, **k:
                        payload["batch_treatment_identity"]["implementation_tree_sha256"] == k["tree"] == new_tree)
    candidate = finalizer.finalize_release_manifest(
        release_manifest_path=path, logical_batch_manifest_path=batches[0],
        realtime_batch_manifest_path=batches[1], output_manifest_path=repo / "candidate.json",
        prepare_distribution_candidate=True, repo_root=repo,
    )
    completion = candidate["formal_evaluation_completion"]
    assert candidate["implementation_tree_sha256"] == proof_tree
    assert completion["qualification_implementation_tree_sha256"] == proof_tree
    assert completion["runtime_identity"]["implementation_tree_sha256"] == new_tree
    identity, valid = verifier._formal_runtime_identity(release, candidate)
    assert valid and identity == completion["runtime_identity"]
    assert verifier._formal_publication_checks(release, candidate, artifact_root=repo)[
        "agentic_formal_completion_identity_valid"
    ]
    completion["qualification_implementation_tree_sha256"] = "0" * 64
    assert not verifier._formal_runtime_identity(release, candidate)[1]
    completion["qualification_implementation_tree_sha256"] = proof_tree
    assert path.read_bytes() == original

    realtime["batch_treatment_identity"]["implementation_tree_sha256"] = "8" * 64
    batches[1].write_text(json.dumps(realtime))
    with pytest.raises(finalizer.ReleaseFinalizationError, match="realtime formal runtime identity mismatch"):
        finalizer.finalize_release_manifest(
            release_manifest_path=path, logical_batch_manifest_path=batches[0],
            realtime_batch_manifest_path=batches[1], output_manifest_path=repo / "mixed.json",
            prepare_distribution_candidate=True, repo_root=repo,
        )
