from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = REPO_ROOT / "release/operate_v0_61_0/core_suite.json"
LITE_PATH = REPO_ROOT / "release/operate_v0_61_0/lite_suite.json"


def _builder_module():
    path = REPO_ROOT / "tools/build_lite_suite.py"
    spec = importlib.util.spec_from_file_location("operate_build_lite_suite", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runner_module():
    path = REPO_ROOT / "run_lite.py"
    spec = importlib.util.spec_from_file_location("operate_run_lite", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_lite_suite_is_deterministic_and_covers_runtime_strata() -> None:
    builder = _builder_module()
    expected = json.loads(LITE_PATH.read_text(encoding="utf-8"))
    rebuilt = builder.build_payload(CORE_PATH)
    assert rebuilt == expected

    core_rows = json.loads(CORE_PATH.read_text(encoding="utf-8"))["scenarios"]
    lite_rows = rebuilt["scenarios"]
    assert len(lite_rows) == 159
    assert rebuilt["n_physical_sources"] == 88
    assert rebuilt["selection_policy"]["downsampled_domains"] == [
        "datacenter",
        "logistics",
    ]
    for field in ("backend_kind", "family", "difficulty_level"):
        assert {row[field] for row in lite_rows} == {row[field] for row in core_rows}
    assert {
        builder._horizon_bucket(int(row["horizon_ticks"])) for row in lite_rows
    } == {label for _, _, label in builder.HORIZON_BUCKETS}

    for domain in rebuilt["selection_policy"]["uncapped_domains"]:
        core_domain = [row for row in core_rows if row["domain"] == domain]
        lite_domain = [row for row in lite_rows if row["domain"] == domain]
        assert lite_domain == core_domain

    for domain in rebuilt["selection_policy"]["downsampled_domains"]:
        core_strata = Counter(
            builder._stratum_key(row)
            for row in core_rows
            if row["domain"] == domain
        )
        lite_strata = Counter(
            builder._stratum_key(row)
            for row in lite_rows
            if row["domain"] == domain
        )
        assert lite_strata == {
            key: min(builder.STRATUM_REPLICATES, count)
            for key, count in core_strata.items()
        }


def test_lite_rows_are_exact_members_of_parent_core() -> None:
    core_rows = json.loads(CORE_PATH.read_text(encoding="utf-8"))["scenarios"]
    lite_rows = json.loads(LITE_PATH.read_text(encoding="utf-8"))["scenarios"]
    core_by_id = {row["scenario_id"]: row for row in core_rows}
    assert len(lite_rows) == len({row["scenario_id"] for row in lite_rows})
    assert all(core_by_id[row["scenario_id"]] == row for row in lite_rows)


def test_lite_runner_forwards_all_rows_as_scenario_slugs(monkeypatch) -> None:
    runner = _runner_module()
    captured: list[str] = []

    def fake_main() -> int:
        captured.extend(sys.argv)
        return 0

    monkeypatch.setattr(runner.batch_llm_eval, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["run_lite.py", "--dry-run"])

    assert runner.main() == 0
    start = captured.index("--scenarios") + 1
    slugs = captured[start : captured.index("--dry-run")]
    assert len(slugs) == 159
    assert all(not slug.startswith("scenarios/") for slug in slugs)
    assert all(not slug.endswith(".yaml") for slug in slugs)

    from runner.batch import expand_scenarios

    assert set(expand_scenarios(slugs)) == set(slugs)
