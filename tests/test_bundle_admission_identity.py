import json

from scripts import build_operate_bundle as builder
from scripts import download_from_hf as reader


def test_compact_proof_packaging_keeps_admission_identity(monkeypatch, tmp_path):
    release = {
        "implementation_tree_sha256": "a" * 64,
        "core_release_pipeline_sha256": "b" * 64,
        "formal_runtime_bundle": {"path": "formal_runtime_bundle.json"},
        "formal_batch_contract": {}, "formal_evidence": {}, "pipeline_artifacts": {},
    }
    monkeypatch.setattr(builder, "implementation_identity", lambda _: {
        "implementation_tree_sha256": "c" * 64,
        "core_release_pipeline_sha256": "d" * 64,
    })
    expected = (tmp_path, "release/current", {"proof.json": "e" * 64}, {})
    monkeypatch.setattr(builder, "_resolve_compact_formal_evidence", lambda **_: expected)
    assert builder._resolve_formal_evidence(
        repo_root=tmp_path, release_root=tmp_path, release_manifest=release,
    ) == expected
    assert release["implementation_tree_sha256"] == "a" * 64


def test_compact_download_allows_new_code_not_changed_release(monkeypatch, tmp_path):
    release = {
        "release_id": "operate", "formal_evaluation_ready": True,
        "implementation_tree_sha256": "a" * 64,
        "core_release_pipeline_sha256": "b" * 64,
        "pipeline_artifacts": {"core_release_pipeline_sha256": "b" * 64},
        "protocol21_replay": {"core_release_pipeline_sha256": "b" * 64},
        "formal_runtime_bundle": {"path": "formal_runtime_bundle.json"},
    }
    local = tmp_path / "benchmark/manifest.json"
    local.parent.mkdir(parents=True)
    local.write_text(json.dumps(release))
    manifest = {
        "schema_version": "operate-runtime-bundle-v2",
        "bundle_kind": "public_runtime_companion",
        "release_id": release["release_id"],
        "release_manifest_sha256": reader._sha256_file(local),
        "implementation_tree_sha256": "a" * 64,
        "core_release_pipeline_sha256": "b" * 64,
    }
    monkeypatch.setattr(reader, "verify_manifest", lambda _: manifest)
    reader.validate_runtime_bundle_compatibility(tmp_path, manifest, repo_root=tmp_path)
    local.write_text("{}")
    reader.validate_runtime_bundle_compatibility(tmp_path, manifest, repo_root=tmp_path)
