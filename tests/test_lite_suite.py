from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _catalog_dir() -> Path:
    return REPO_ROOT / "benchmark"


CORE_PATH = _catalog_dir() / "core_suite.json"
LITE_PATH = _catalog_dir() / "lite_suite.json"


def _runner_module():
    path = REPO_ROOT / "run_lite.py"
    spec = importlib.util.spec_from_file_location("operate_run_lite", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_lite_suite_discloses_parent_coverage_scope() -> None:
    rebuilt = json.loads(LITE_PATH.read_text(encoding="utf-8"))
    core_rows = json.loads(CORE_PATH.read_text(encoding="utf-8"))["scenarios"]
    audit = rebuilt["selection_audit"]
    selected_ids = {row["scenario_id"] for row in rebuilt["scenarios"]}
    assert rebuilt["selection_policy"]["model_outcomes_used_for_selection"] is True
    assert rebuilt["selection_policy"]["new_model_score_thresholds_used"] is False
    assert audit["coverage_complete"] is True
    assert audit["budget_satisfied"] is True
    assert len(audit["rows"]) == len(core_rows)
    selected_features = set()
    for row in audit["rows"]:
        assert row["included"] == (row["scenario_id"] in selected_ids)
        assert set(row["covered_by"]) <= selected_ids
        if row["included"]:
            selected_features.update(row["feature_ids"])
    assert len(selected_features) == audit["n_retained_parent_features"] == 204
    assert set(audit["uncovered_parent_feature_ids"]) == set(range(204)) - selected_features


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


def _forwarded_argv(monkeypatch, runner, argv: list[str]) -> list[str]:
    captured: list[str] = []

    def fake_main() -> int:
        captured.extend(sys.argv)
        return 0

    monkeypatch.setattr(runner.batch_llm_eval, "main", fake_main)
    monkeypatch.setattr(sys, "argv", ["run_lite.py", *argv])
    assert runner.main() == 0
    return captured


def _forwarded_option_value(argv: list[str], option: str) -> str | None:
    for index, argument in enumerate(argv):
        name, separator, value = argument.partition("=")
        if name == option:
            return value if separator else (argv[index + 1] if index + 1 < len(argv) else "")
    return None


def test_lite_runner_injects_documented_profile_when_caller_is_silent(monkeypatch) -> None:
    runner = _runner_module()
    captured = _forwarded_argv(monkeypatch, runner, ["--dry-run"])

    assert _forwarded_option_value(captured, "--persistent-history-max-messages") == "64"
    assert _forwarded_option_value(captured, "--persistent-context-max-chars") == "512000"
    assert _forwarded_option_value(captured, "--persistent-memory-max-items") == "128"
    assert _forwarded_option_value(captured, "--provider-failure-policy") == "abort"
    assert _forwarded_option_value(captured, "--max-consecutive-provider-failures") == "1"
    # Injected bounds must precede the fixed scope tail that redefines the
    # scenario set, and must never appear after --scenarios.
    assert captured.index("--persistent-history-max-messages") < captured.index("--scenarios")


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--persistent-history-max-messages", "40"),
        ("--persistent-context-max-chars", "64000"),
        ("--persistent-memory-max-items", "80"),
    ],
)
def test_lite_runner_keeps_explicit_profile_choice(
    monkeypatch, option: str, value: str,
) -> None:
    runner = _runner_module()
    captured = _forwarded_argv(monkeypatch, runner, ["--dry-run", option, value])

    assert _forwarded_option_value(captured, option) == value
    assert captured.count(option) == 1
    # The two bounds the caller did not name are still filled in.
    expected_missing = {
        flag for flag, _ in runner.LITE_PROFILE_FLAGS if flag != option
    }
    for flag in expected_missing:
        assert _forwarded_option_value(captured, flag) is not None


def test_lite_runner_honors_flag_equals_form_and_leaves_other_bounds_defaulted(
    monkeypatch,
) -> None:
    runner = _runner_module()
    captured = _forwarded_argv(
        monkeypatch, runner, ["--dry-run", "--persistent-context-max-chars=256000"]
    )

    assert _forwarded_option_value(captured, "--persistent-context-max-chars") == "256000"
    assert captured.count("--persistent-context-max-chars=256000") == 1
    assert not any(
        argument == "--persistent-context-max-chars" for argument in captured
    )
    assert _forwarded_option_value(captured, "--persistent-history-max-messages") == "64"
    assert _forwarded_option_value(captured, "--persistent-memory-max-items") == "128"


def test_lite_runner_honors_caller_abbreviation_of_documented_bound(monkeypatch) -> None:
    """Downstream argparse accepts unambiguous prefixes, so `--persistent-context`
    is a legal caller declaration of `--persistent-context-max-chars`. Injecting
    the default would be appended last and silently replace the caller's value."""
    runner = _runner_module()
    captured = _forwarded_argv(
        monkeypatch, runner, ["--dry-run", "--persistent-context", "256000"]
    )

    assert captured.count("--persistent-context-max-chars") == 0
    assert "--persistent-context" in captured
    assert captured[captured.index("--persistent-context") + 1] == "256000"
    # The two bounds the caller did not name are still filled in.
    assert _forwarded_option_value(captured, "--persistent-history-max-messages") == "64"
    assert _forwarded_option_value(captured, "--persistent-memory-max-items") == "128"


def test_lite_runner_logs_profile_provenance_for_each_bound(
    monkeypatch, capsys,
) -> None:
    """Roadmap P0-4 records where each bound came from; run_lite.py has no
    run-metadata channel, so the provenance goes to the run log."""
    runner = _runner_module()
    _forwarded_argv(
        monkeypatch, runner,
        ["--dry-run", "--persistent-context-max-chars", "256000"],
    )

    stderr = capsys.readouterr().err
    assert "--persistent-context-max-chars=caller" in stderr
    assert "--persistent-history-max-messages=run_lite_default" in stderr
    assert "--persistent-memory-max-items=run_lite_default" in stderr
    assert "--provider-failure-policy=run_lite_default" in stderr
    assert "efficiency/development track" in stderr


def test_reviewed_lite_budget_does_not_prune_suspect_old_zero_scores():
    rebuilt = json.loads(LITE_PATH.read_text(encoding="utf-8"))
    assert rebuilt['selection_algorithm'] == 'reviewed_horizon_balanced_v11'
    assert rebuilt['n_scenarios'] == 141
    assert rebuilt['n_physical_sources'] == 82
    assert sorted(row['horizon_ticks'] for row in rebuilt['scenarios'] if row['horizon_ticks'] > 192) == [298, 400]
    assert rebuilt['total_horizon_ticks'] == 3709
    audit = rebuilt['selection_audit']
    assert audit['n_parent_features'] == 204
    assert audit['n_retained_parent_features'] == 204
    assert audit['coverage_complete'] is True
    assert len(audit['review_excluded_scenario_ids']) == 3
    assert all(row['backend_kind'] == 'dynasched_flexible_job_shop'
               for row in audit['rows'] if row.get('review_disposition') == 'long_horizon_diagnostic')
    assert any('routing_cvrp_X-n313-k71' in row['path'] for row in rebuilt['scenarios'])
