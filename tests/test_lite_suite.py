from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import pytest

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
    assert 0 < len(lite_rows) < len(core_rows)
    assert rebuilt["selection_audit"]["coverage_complete"] is True
    for field in ("backend_kind", "family", "difficulty_level"):
        assert {row[field] for row in lite_rows} == {row[field] for row in core_rows}
    assert {
        builder._horizon_bucket(int(row["horizon_ticks"])) for row in lite_rows
    } == {label for _, _, label in builder.HORIZON_BUCKETS}

    audit = rebuilt["selection_audit"]
    selected_ids = {row["scenario_id"] for row in lite_rows}
    assert len(audit["rows"]) == len(core_rows)
    selected_features = set()
    for row in audit["rows"]:
        assert row["included"] == (row["scenario_id"] in selected_ids)
        if row["included"]:
            if row["selection_stage"] == "coverage_core":
                assert row["new_feature_ids"]
            elif row["selection_stage"] == "small_domain_retention":
                assert row["reason"] == "preserves_admitted_small_domain_variation"
            else:
                assert row["selection_stage"] == "diversity_enrichment"
                assert row["new_source_support_feature_ids"]
                assert row["reason"] == "adds_independent_source_support"
            selected_features.update(row["feature_ids"])
        else:
            assert row["reason"] in {
                "coverage_already_represented",
                "preferred_budget_reached",
            }
            assert set(row["covered_by"]) <= selected_ids
    assert selected_features == set(range(len(audit["features"])))


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
    assert captured.index("--dry-run") < start
    slugs = captured[start:]
    assert len(slugs) == json.loads(LITE_PATH.read_text())["n_scenarios"]
    assert all(not slug.startswith("scenarios/") for slug in slugs)
    assert all(not slug.endswith(".yaml") for slug in slugs)

    from runner.batch import expand_scenarios

    assert set(expand_scenarios(slugs)) == set(slugs)


def _row(
    tmp_path,
    name,
    *,
    event="breakdown",
    domain="logistics",
    config=None,
    provenance=None,
):
    body = {
        "scenario_id": name,
        "domain": domain,
        "backend_kind": "fixture",
        "family": "fixture",
        "difficulty_level": "high",
        "difficulty_mode": "time_pressure",
        "horizon_ticks": 48,
        "seed": 42,
        "scenario_signature": name,
        "backend_config": config or {},
        "provenance": provenance or {"data_source": "fixture"},
        "perturbations": [{"kind": event, "hidden": False}],
    }
    path = tmp_path / "scenarios" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body) + "\n")
    return {
        **{
            key: body[key]
            for key in (
                "scenario_id",
                "domain",
                "backend_kind",
                "family",
                "difficulty_level",
                "difficulty_mode",
                "horizon_ticks",
                "seed",
                "scenario_signature",
            )
        },
        "path": path.relative_to(tmp_path).as_posix(),
        "yaml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "physical_source_key": name,
        "semantic_fingerprint": name,
        "structural_fingerprint": name,
        "source_denominator_key": name,
        "status": "core_locked",
        "core_disposition": "core_locked",
    }


def test_new_coverage_is_included_without_a_stratum_quantity_cap(tmp_path):
    builder = _builder_module()
    rows = [_row(tmp_path, f"row-{n}", event=f"event-{n}") for n in range(6)]
    assert builder.select_lite(rows, repo_root=tmp_path) == rows
    assert builder.select_lite(list(reversed(rows)), repo_root=tmp_path) == rows


def test_coverage_redundant_rows_are_not_forced_into_compressed_domains(tmp_path):
    builder = _builder_module()
    rows = [_row(tmp_path, name) for name in ("a", "b", "c")]
    for row in rows:
        row["physical_source_key"] = "same-source"
    assert builder.select_lite(rows, repo_root=tmp_path) == [rows[0]]


@pytest.mark.parametrize("mutation", ["not_admitted", "yaml_drift", "duplicate_id"])
def test_lite_rejects_ineligible_or_unbound_rows(tmp_path, mutation):
    builder = _builder_module()
    row = _row(tmp_path, "a")
    rows = [row]
    if mutation == "not_admitted":
        row["status"] = "candidate"
    elif mutation == "yaml_drift":
        (tmp_path / row["path"]).write_text("{}")
    else:
        rows.append(dict(row))
    with pytest.raises(ValueError):
        builder.select_lite(rows, repo_root=tmp_path)


def test_enrichment_adds_independent_source_support_in_complete_rounds(tmp_path):
    builder = _builder_module()
    rows = [_row(tmp_path, name) for name in ("a", "b", "c", "d", "e", "f")]
    duplicate = _row(tmp_path, "a-copy")
    duplicate["physical_source_key"] = "a"
    rows.append(duplicate)
    core, _ = builder._select_with_audit(
        rows, repo_root=tmp_path, preferred_range=(0, 0)
    )
    selected, audit = builder._select_with_audit(
        rows, repo_root=tmp_path, preferred_range=(4, 6)
    )
    assert core == [rows[0]]
    assert all(row in selected for row in core)
    assert [row["scenario_id"] for row in selected] == ["a", "b", "c", "d"]
    assert audit["budget_satisfied"] is True
    assert [r["target_distinct_sources"] for r in audit["enrichment_rounds"]] == [
        2,
        3,
        4,
    ]
    assert all(r["completed"] for r in audit["enrichment_rounds"])
    assert builder._select_with_audit(
        list(reversed(rows)), repo_root=tmp_path, preferred_range=(4, 6)
    ) == (selected, audit)


def test_enrichment_budget_never_removes_required_coverage(tmp_path):
    builder = _builder_module()
    rows = [_row(tmp_path, f"row-{n}", event=f"event-{n}") for n in range(6)]
    selected, audit = builder._select_with_audit(
        rows, repo_root=tmp_path, preferred_range=(3, 5)
    )
    assert selected == rows
    assert audit["coverage_complete"] is True
    assert audit["budget_satisfied"] is False
    assert audit["enrichment_rounds"] == []


def test_enrichment_cap_stops_additions_without_filling_with_duplicates(tmp_path):
    builder = _builder_module()
    rows = [_row(tmp_path, f"a-{n}") for n in range(4)]
    rows += [_row(tmp_path, f"b-{n}", event="other") for n in range(4)]
    selected, audit = builder._select_with_audit(
        rows, repo_root=tmp_path, preferred_range=(3, 3)
    )
    assert len(selected) == 3
    assert audit["enrichment_rounds"][-1]["completed"] is False
    assert audit["coverage_complete"] is True


def test_declared_hazard_and_site_regimes_are_coverage_not_opaque_ids(tmp_path):
    builder = _builder_module()
    rows = [
        _row(
            tmp_path,
            "driving-a",
            domain="autonomous_driving",
            config={
                "task_requirements": {
                    "latest_preventive_command_tick": 3,
                    "paid_safety_inspection_deadline_tick": 3,
                }
            },
            provenance={"data_source": "ngsim", "hazard_kind": "lead_vehicle_braking"},
        ),
        _row(
            tmp_path,
            "driving-b",
            domain="autonomous_driving",
            config={
                "task_requirements": {
                    "latest_preventive_command_tick": 3,
                    "paid_safety_inspection_deadline_tick": 3,
                }
            },
            provenance={
                "data_source": "ngsim",
                "hazard_kind": "minimum_time_headway_conflict",
            },
        ),
        _row(
            tmp_path,
            "microgrid-a",
            domain="microgrid",
            config={
                "site": "seattle_wa",
                "forecast_bias": 0.1,
                "forecast_error_sigma": 0.12,
                "genset_available": False,
            },
        ),
        _row(
            tmp_path,
            "microgrid-b",
            domain="microgrid",
            config={
                "site": "phoenix_az",
                "forecast_bias": 0.1,
                "forecast_error_sigma": 0.12,
                "genset_available": False,
            },
        ),
    ]
    assert builder.select_lite(rows, repo_root=tmp_path) == rows
