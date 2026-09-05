import json

import pytest

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
    from core import implementation_identity

    release = {
        "release_id": "operate_v0_61_0", "formal_evaluation_ready": True,
        "implementation_tree_sha256": "a" * 64,
        "core_release_pipeline_sha256": "b" * 64,
        "pipeline_artifacts": {"core_release_pipeline_sha256": "b" * 64},
        "protocol21_replay": {"core_release_pipeline_sha256": "b" * 64},
        "formal_runtime_bundle": {"path": "formal_runtime_bundle.json"},
    }
    local = tmp_path / "release/operate_v0_61_0/manifest.json"
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
    monkeypatch.setattr(reader, "_validate_bundle_formal_evidence_binding", lambda *_: release)
    monkeypatch.setattr(reader, "_validate_candidate_closure_binding", lambda *_, **__: None)
    checked = []
    monkeypatch.setattr(reader, "_validate_bundle_source_asset_bindings", lambda *_, **__: checked.append(True))
    monkeypatch.setattr(implementation_identity, "implementation_identity", lambda _: {"implementation_tree_sha256": "c" * 64})
    reader.validate_runtime_bundle_compatibility(tmp_path, manifest, repo_root=tmp_path)
    assert checked == [True]
    local.write_text("{}")
    with pytest.raises(ValueError, match="local_release_manifest_mismatch"):
        reader.validate_runtime_bundle_compatibility(tmp_path, manifest, repo_root=tmp_path)


def test_public_bundle_omits_redundant_release_version(tmp_path):
    from tests.test_build_operate_bundle import _fixture

    paths = _fixture(tmp_path)
    manifest = builder.build_operate_bundle(
        repo_root=paths["repo"], release_dir=paths["release"],
        output_dir=paths["repo"] / "public-bundle", include_backends=False,
        versionless_filenames=True,
    )
    assert "release_version" not in manifest
    assert manifest["release_id"] == paths["release"].name
