from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


pyarrow = pytest.importorskip("pyarrow")
parquet = pytest.importorskip("pyarrow.parquet")

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = REPO_ROOT / "release/operate_v0_61_0"


def _builder_module():
    path = REPO_ROOT / "tools/build_hf_parquet.py"
    spec = importlib.util.spec_from_file_location("operate_build_hf_parquet", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_and_lite_parquet_are_deterministic_and_reversible(
    tmp_path: Path,
) -> None:
    builder = _builder_module()
    full_suite = json.loads((RELEASE_DIR / "core_suite.json").read_text())
    lite_suite = json.loads((RELEASE_DIR / "lite_suite.json").read_text())
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = builder.build_exports(REPO_ROOT, first)
    second_manifest = builder.build_exports(REPO_ROOT, second)

    assert first_manifest == second_manifest
    assert first_manifest["artifacts"]["full"]["n_rows"] == full_suite["n_scenarios"]
    assert first_manifest["artifacts"]["lite"]["n_rows"] == lite_suite["n_scenarios"]
    assert first_manifest["artifacts"]["full"]["n_physical_sources"] == full_suite[
        "n_physical_sources"
    ]
    assert first_manifest["artifacts"]["lite"]["n_physical_sources"] == lite_suite[
        "n_physical_sources"
    ]

    for subset, suite_name in (("full", "core_suite.json"), ("lite", "lite_suite.json")):
        relative = first_manifest["artifacts"][subset]["path"]
        first_parquet = first / relative
        second_parquet = second / relative
        assert first_parquet.read_bytes() == second_parquet.read_bytes()

        table = parquet.read_table(first_parquet)
        assert table.num_rows == first_manifest["artifacts"][subset]["n_rows"]
        assert table.schema.metadata[b"operate.subset"].decode() == subset
        assert "scenario_yaml" in table.column_names
        assert "scenario_metadata_json" in table.column_names
        assert "suite_template_json" in table.column_names

        rebuilt = tmp_path / f"rebuilt-{subset}"
        suite_output = builder.rebuild_parquet(first_parquet, rebuilt)
        assert suite_output.read_bytes() == (RELEASE_DIR / suite_name).read_bytes()
        rebuilt_suite = json.loads(suite_output.read_text(encoding="utf-8"))
        for row in rebuilt_suite["scenarios"]:
            assert (rebuilt / row["path"]).read_bytes() == (
                REPO_ROOT / row["path"]
            ).read_bytes()
