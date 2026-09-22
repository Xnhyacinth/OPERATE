from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from runner.episode import _public_agent_config
from scripts import batch_llm_eval as batch


def _saved_job(tmp_path):
    directory = tmp_path / "trajectory"
    directory.mkdir()
    cfg = batch._llm_config_from_dict({"model": "test-model", "provider": "openai"})
    identity = {
        "scenario_signature": "sig", "seed": 42, "agent_name": "llm_agent",
        "agent_config": _public_agent_config({"config": cfg}),
        "implementation": {"implementation_tree_sha256": "tree"},
        "checkpoint_identity": None,
    }
    source = directory / "case.completed_runtime.json"
    source.write_text(json.dumps({"payload": {
        "identity": identity,
        "counterfactual_settings": {
            "masking_policy": "wait_only", "per_action": True, "per_action_cap": None,
            "per_action_groups": True, "per_action_group_cap": None,
        },
        "postprocessing_context": {"within_tick_interaction": True},
    }}))
    descriptor = {
        "path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "byte_count": source.stat().st_size,
    }
    (directory / "case.summary.json").write_text(json.dumps({
        "trajectory_summary": {"completed_runtime_artifact": descriptor},
    }))
    job = {
        "scenario_slug": "case", "scenario_signature": "sig", "seed": 42,
        "model": "test-model", "llm_config": batch._llm_config_to_dict(cfg),
        "batch_output_dir": str(tmp_path), "trajectory_dir": str(directory),
        "resume_postprocessing": True, "implementation_tree_sha256": "tree",
        "implementation_policy": "provenance",
    }
    return job, source, identity


def _recover(source, expected_sha256, *, expected_identity, output):
    result = {
        "schema_version": "offline_completed_episode_recovery_v1",
        "result": {"score": {"total_score": 42}}, "source_identity": expected_identity,
        "recompute_identity": {"implementation_tree_sha256": "tree"},
        "legacy_context_reconstructed": False, "same_contract_recovery": True,
    }
    output.write_text(json.dumps(result))
    return result


def _no_provider(monkeypatch):
    monkeypatch.setattr(batch, "implementation_identity", lambda _: {"implementation_tree_sha256": "tree"})
    monkeypatch.setattr(batch, "_run_one_safe", lambda *a: pytest.fail("paid interaction rerun"))
    monkeypatch.setattr(batch, "_quarantine_retry_trajectory", lambda *a: pytest.fail("snapshot quarantined"))
    monkeypatch.setattr("runner.postprocessing.recover_completed_episode", _recover)


def test_batch_resume_uses_bound_snapshot_before_quarantine(tmp_path, monkeypatch):
    job, source, _ = _saved_job(tmp_path)
    before = source.read_bytes()
    _no_provider(monkeypatch)
    row = batch._run_llm_episode_job(job)
    assert row["status"] == "ok"
    assert row["score"]["total_score"] == 42
    assert row["postprocessing_recovery"]["diagnostic_only"] is False
    assert Path(row["postprocessing_recovery_artifact"]["path"]).is_file()
    assert source.read_bytes() == before


@pytest.mark.parametrize("change", ["bytes", "identity", "unbound", "strict-tree", "size"])
def test_invalid_saved_runtime_never_falls_back_to_provider(tmp_path, monkeypatch, change):
    job, source, _ = _saved_job(tmp_path)
    summary_path = source.with_name("case.summary.json")
    if change == "bytes":
        source.write_text(source.read_text() + " ")
    elif change == "identity":
        job["seed"] = 99
    elif change == "unbound":
        summary_path.unlink()
    elif change == "strict-tree":
        job["implementation_policy"] = "strict"
        payload = json.loads(source.read_text())
        payload["payload"]["identity"]["implementation"]["implementation_tree_sha256"] = "old"
        source.write_text(json.dumps(payload))
        summary = json.loads(summary_path.read_text())
        summary["trajectory_summary"]["completed_runtime_artifact"].update(
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(), byte_count=source.stat().st_size,
        )
        summary_path.write_text(json.dumps(summary))
    else:
        summary = json.loads(summary_path.read_text())
        summary["trajectory_summary"]["completed_runtime_artifact"]["byte_count"] += 1
        summary_path.write_text(json.dumps(summary))
    _no_provider(monkeypatch)
    row = batch._run_llm_episode_job(job)
    assert row["status"] == "error"
    assert row["error_type"] == "CompletedRuntimeRecoveryError"
    assert source.exists()


def test_timeout_row_is_infrastructure_and_next_resume_retryable(tmp_path, monkeypatch):
    from runner.worker_deadline import WorkerDeadlineError

    job, _, _ = _saved_job(tmp_path)
    job.update(episode_timeout_s=10, postprocessing_timeout_s=2)

    def timeout(*args, **kwargs):
        raise WorkerDeadlineError("postprocessing", 2)

    monkeypatch.setattr("runner.worker_deadline.run_with_deadline", timeout)
    row = batch._run_llm_episode_job_with_deadline(job)
    assert row["status"] == "error"
    assert row["termination_category"] == "harness_error"
    assert row["timeout_stage"] == "postprocessing"
    assert row["timeout_budget_s"] == 2
    assert "score" not in row
    assert batch._retryable_infrastructure_row(row)


def test_diagnostic_recovery_cannot_enter_formal_leaderboard():
    row = {"status": "ok", "postprocessing_recovery": {"diagnostic_only": True}}
    eligible, reasons = batch._formal_row_eligibility(row)
    assert not eligible
    assert "postprocessing_recovery_diagnostic_only" in reasons


def test_resume_log_counts_terminal_errors_separately(monkeypatch):
    monkeypatch.setattr(batch, "_terminal_attempt_key", lambda row: row["id"])
    jobs = [{"id": value} for value in range(3)]
    rows = [{"id": 0, "status": "ok"}, {"id": 1, "status": "error"}]
    counts = batch._resume_skip_counts(jobs, [jobs[2]], rows, "retry-infrastructure")
    assert counts["ok"] == counts["error"] == 1
    assert counts["repair"] == 0


def test_saved_postprocessing_bypasses_provider_quota(tmp_path, monkeypatch):
    job, _, _ = _saved_job(tmp_path)
    _no_provider(monkeypatch)
    monkeypatch.setattr(batch, "_active_quota_sentinel", lambda job: {"reset_at": "tomorrow"})
    row = batch._run_llm_episode_job(job)
    assert row["status"] == "ok"
    assert row["provider_request_accounting_scope"] == "retained_completed_runtime"


@pytest.mark.parametrize("field,value", [
    ("masking_policy", "no_action"), ("per_action", False), ("per_action_cap", 20),
    ("per_action_groups", False), ("per_action_group_cap", 20),
    ("within_tick_interaction", False),
])
def test_saved_runtime_execution_settings_must_match_batch(tmp_path, monkeypatch, field, value):
    job, source, _ = _saved_job(tmp_path)
    envelope = json.loads(source.read_text())
    section = "postprocessing_context" if field == "within_tick_interaction" else "counterfactual_settings"
    envelope["payload"][section][field] = value
    source.write_text(json.dumps(envelope))
    summary_path = source.with_name("case.summary.json")
    summary = json.loads(summary_path.read_text())
    summary["trajectory_summary"]["completed_runtime_artifact"].update(
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(), byte_count=source.stat().st_size,
    )
    summary_path.write_text(json.dumps(summary))
    _no_provider(monkeypatch)
    monkeypatch.setattr("runner.postprocessing.recover_completed_episode",
                        lambda *a, **k: pytest.fail("incompatible scoring recovery"))
    row = batch._run_llm_episode_job(job)
    assert row["status"] == "error"
    assert row["error_type"] == "CompletedRuntimeRecoveryError"
