import json
from pathlib import Path

import pytest

parquet = pytest.importorskip("pyarrow.parquet")

from tools import build_hf_parquet as builder  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_public_metadata_is_minimal_and_reversible(tmp_path):
    manifest = builder.build_exports(ROOT, tmp_path / "public", public_metadata=True)
    assert "release_id" not in manifest
    for subset, suite_name in (("full", "core_suite.json"), ("lite", "lite_suite.json")):
        artifact = manifest["artifacts"][subset]
        assert "suite_id" not in artifact
        source = tmp_path / "public" / artifact["path"]
        table = parquet.read_table(source)
        assert not {
            "release_id", "suite_id", "track", "is_official_full_denominator",
            "declared_leaderboard_eligible", "selection_algorithm", "status",
            "core_disposition", "construct_contract", "suite_template_json",
        }.intersection(table.column_names)
        assert len(table.column_names) == 22
        assert b"operate.suite_template_json" in table.schema.metadata
        assert {"subset", "scenario_id", "scenario_yaml", "yaml_sha256"} <= set(table.column_names)
        rebuilt = builder.rebuild_parquet(source, tmp_path / subset)
        assert rebuilt.read_bytes() == (ROOT / "release/operate_v0_61_0" / suite_name).read_bytes()


def test_private_export_keeps_existing_metadata(tmp_path):
    manifest = builder.build_exports(ROOT, tmp_path)
    assert "release_id" in manifest
    artifact = manifest["artifacts"]["full"]
    assert "suite_id" in artifact
    table = parquet.read_table(tmp_path / artifact["path"])
    assert {"release_id", "suite_id", "status", "core_disposition", "construct_contract"} <= set(table.column_names)
    assert "suite_template_json" in table.column_names
    rebuilt = builder.rebuild_parquet(tmp_path / artifact["path"], tmp_path / "rebuilt")
    assert rebuilt.read_bytes() == (ROOT / "release/operate_v0_61_0/core_suite.json").read_bytes()


@pytest.mark.parametrize("tamper", [False, True])
def test_public_template_metadata_is_required_and_hash_bound(tmp_path, tamper):
    manifest = builder.build_exports(ROOT, tmp_path / "public", public_metadata=True)
    source = tmp_path / "public" / manifest["artifacts"]["lite"]["path"]
    table = parquet.read_table(source)
    metadata = dict(table.schema.metadata)
    if tamper:
        template = json.loads(metadata[b"operate.suite_template_json"])
        template.append(["unexpected_field", True])
        metadata[b"operate.suite_template_json"] = json.dumps(template).encode()
    else:
        metadata.pop(b"operate.suite_template_json", None)
    changed = tmp_path / "changed.parquet"
    parquet.write_table(table.replace_schema_metadata(metadata), changed)
    with pytest.raises(ValueError, match="suite template metadata missing|rebuilt suite JSON"):
        builder.rebuild_parquet(changed, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
