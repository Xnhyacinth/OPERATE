import json

import pytest

from scripts import audit_agent_failure_recipes, audit_staleness_consumption, audit_tool_effects
from scripts import batch_llm_eval as batch
from scripts.analyze_batch_results import analyze_output_dir


@pytest.mark.parametrize("audit,key", [
    (audit_agent_failure_recipes, "agent_failure_recipes"),
    (audit_staleness_consumption, "staleness_consumption_audit"),
    (audit_tool_effects, "tool_effect_audit"),
])
def test_audits_resolve_portable_trajectory_from_batch_root(tmp_path, monkeypatch, audit, key):
    run = tmp_path / "run"
    trajectory = run / "trajectories" / "episode.trajectory.jsonl"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(json.dumps({"tick": 0, "tool_results": []}) + "\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    row = {"status": "ok", "trajectory_summary": {"trajectory_path": "trajectories/episode"}}
    report = audit.build_report([row], batch_root=run)
    assert report[key]["totals"]["episodes_with_trajectory"] == 1


def test_resume_uses_batch_root_for_artifact_verification(tmp_path, monkeypatch):
    def eligible(row, **kwargs):
        return (True, []) if kwargs.get("batch_root") == tmp_path else (False, ["missing"])

    monkeypatch.setattr(batch, "_formal_row_eligibility", eligible)
    row = {"status": "ok", "agent_treatment_sha256": "bound"}
    assert batch._row_is_clean_for_resume(row, batch_root=tmp_path)


def test_quota_rows_are_unavailable_execution_not_task_failures():
    rows = [
        {"model": "m", "scenario_slug": "a", "seed": 42, "status": "ok"},
        {"model": "m", "scenario_slug": "b", "seed": 42, "status": "error",
         "quota_parked": True, "error": "ProviderQuotaExhaustedError"},
    ]
    report = batch._pass_k_success_summary(
        rows, configured_models=["m"], configured_seeds=[42], n_scenarios=2,
    )
    assert report["metric_kind"] == "execution_completion"
    assert report["expected_total_cells"] == 2
    assert report["per_model"]["m"]["error_pass_units"] == 0
    assert report["per_model"]["m"]["unavailable_pass_units"] == 1
    assert report["per_model"]["m"]["failed_cells"] == 0
    assert report["per_model"]["m"]["unavailable_cells"] == 1


def test_deep_report_distinguishes_missing_task_outcome_and_quota(tmp_path):
    rows = [
        {"model": "m", "status": "ok", "score": {"total_score": 1},
         "task_completion": {"completed": True}},
        {"model": "m", "status": "ok", "score": {"total_score": 0}},
        {"model": "m", "status": "error", "quota_parked": True},
    ]
    report = analyze_output_dir(tmp_path, rows)
    assert report["n_error"] == 0
    assert report["n_episodes_quota_unavailable"] == 1
    assert report["event_adaptive_autonomy_by_model"]["m"]["task_outcome_observed"] == 1
    assert report["event_adaptive_autonomy_by_model"]["m"]["task_completion_rate"] == 1.0


def test_trajectory_resolution_cannot_escape_batch(tmp_path):
    from evaluation.trajectory_paths import trajectory_file

    run = tmp_path / "run"
    run.mkdir()
    outside = tmp_path / "outside.trajectory.jsonl"
    outside.write_text("{}\n")
    assert trajectory_file({"trajectory_summary": {"trajectory_path": "../outside"}}, batch_root=run) is None


def test_parked_rows_preserve_seed_identity():
    rows = [batch._quota_parked_result({"model": "m", "scenario_slug": "s", "seed": seed})
            for seed in (42, 43)]
    assert [row["seed"] for row in rows] == [42, 43]
    assert all(row["execution_started"] is False for row in rows)
    summary = batch._pass_k_success_summary(
        rows, configured_models=["m"], configured_seeds=[42, 43], n_scenarios=1,
    )["per_model"]["m"]
    assert summary["unavailable_pass_units"] == 2
    assert summary["missing_pass_units"] == 0


def test_quota_counts_distinguish_attempted_and_legacy_unstarted():
    from evaluation.batch_status import execution_status_counts

    attempted = {"status": "error", "quota_parked": True,
                 "error": "ProviderQuotaExhaustedError: account quota exhausted"}
    unstarted = {"status": "error", "quota_parked": True,
                 "error": "ProviderQuotaExhaustedError: parked after provider quota exhausted; reset_at=2026-09-05"}
    counts = execution_status_counts([attempted] * 32 + [unstarted] * 713)
    assert counts["n_episodes_quota_unavailable"] == 745
    assert counts["n_episodes_quota_parked"] == 713


def test_legacy_nonformal_trajectory_prefix_resolves_inside_batch(tmp_path, monkeypatch):
    from evaluation.trajectory_paths import trajectory_file

    monkeypatch.chdir(tmp_path)
    run = tmp_path / "reports" / "run"
    path = run / "trajectories" / "episode.trajectory.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n")
    row = {"trajectory_summary": {"trajectory_path": "reports/run/trajectories/episode"}}
    assert trajectory_file(row, batch_root=run) == path


def test_legacy_nonformal_sidecar_prefix_keeps_hash_verification(tmp_path, monkeypatch):
    from tests.test_batch_llm_eval import _bind_formal_artifacts, _formally_eligible_protocol21_row

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "reports" / "run"
    root.mkdir(parents=True)
    row = _bind_formal_artifacts(_formally_eligible_protocol21_row(), root, key="legacy")
    row = batch._portable_formal_result_paths([row], batch_root=tmp_path)[0]
    assert batch._row_is_clean_for_resume(row, batch_root=root)
    path = tmp_path / row["trajectory_summary"]["provider_audit_artifact"]["path"]
    path.write_bytes(path.read_bytes() + b'{"tampered":true}\n')
    assert not batch._row_is_clean_for_resume(row, batch_root=root)
