from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import batch_llm_eval as batch
from tests.test_resume_artifact_integrity import _bound_attempt


def _interrupt_retry(tmp_path, monkeypatch, *, relative=False):
    job, row = _bound_attempt(tmp_path)
    job["trajectory_dir"] = str(
        Path(row["trajectory_summary"]["trajectory_artifact"]["path"]).parent
    )
    job["batch_output_dir"] = str(tmp_path)
    job["invocation_started_at_utc"] = "2026-09-06T00:00:00+00:00"
    row.update(
        status="error", error_type="APITimeoutError", error_stage="interaction_loop"
    )
    if relative:
        monkeypatch.chdir(tmp_path.parent)
        job["trajectory_dir"] = str(Path(job["trajectory_dir"]).relative_to(tmp_path.parent))
        summary = row["trajectory_summary"]
        summary["trajectory_path"] = str(Path(summary["trajectory_path"]).relative_to(tmp_path.parent))
        for key, value in summary.items():
            if key.endswith("_artifact"):
                value["path"] = str(Path(value["path"]).relative_to(tmp_path.parent))
    monkeypatch.setattr(
        batch, "_llm_config_from_dict", lambda _: SimpleNamespace(temperature=0)
    )
    monkeypatch.setattr(
        batch,
        "implementation_identity",
        lambda _: {"implementation_tree_sha256": "tree"},
    )

    def interrupt(_):
        raise KeyboardInterrupt("simulated process interruption")

    monkeypatch.setattr(batch, "_run_one_safe", interrupt)
    pending = batch._filter_pending_jobs(
        [job], [row], batch_root=tmp_path, resume_policy="retry-infrastructure"
    )
    with pytest.raises(KeyboardInterrupt):
        batch._run_llm_episode_job(pending[0])
    return job, row


def test_interrupted_retry_resumes_from_exact_archived_attempt_binding(
    tmp_path, monkeypatch
):
    job, row = _interrupt_retry(tmp_path, monkeypatch)
    Path(job["trajectory_dir"]).mkdir()
    assert batch._filter_pending_jobs(
        [job], [row], batch_root=tmp_path, resume_policy="retry-infrastructure"
    )
    manifests = list((tmp_path / ".attempt_archive_bindings").glob("*.json"))
    assert len(manifests) == 1
    binding = json.loads(manifests[0].read_text())
    assert binding["prior_row_sha256"] == batch._canonical_json_sha256(row)


def test_worker_start_proof_is_written_only_inside_execution(tmp_path, monkeypatch):
    job, _ = _interrupt_retry(tmp_path, monkeypatch)
    records = [
        json.loads(line)
        for line in (tmp_path / "worker_starts.jsonl").read_text().splitlines()
    ]
    assert len(records) == 1
    proof = records[0]
    assert proof["schema_version"] == "worker_execution_start_v1"
    assert proof["invocation_started_at_utc"] == job["invocation_started_at_utc"]
    assert proof["execution_attempt_id"]
    assert proof["scenario_slug"] == job["scenario_slug"]
    assert proof["agent_treatment_sha256"] == job["agent_treatment_sha256"]


def test_repeated_interruption_keeps_original_archive_and_detects_tampering(
    tmp_path, monkeypatch
):
    job, row = _interrupt_retry(tmp_path, monkeypatch)
    (manifest,) = (tmp_path / ".attempt_archive_bindings").glob("*.json")
    original_binding = manifest.read_bytes()
    replacement = Path(job["trajectory_dir"])
    replacement.mkdir()
    (replacement / "partial.jsonl").write_text("partial\n")
    retry = batch._filter_pending_jobs(
        [job], [row], batch_root=tmp_path, resume_policy="retry-infrastructure"
    )[0]
    with pytest.raises(KeyboardInterrupt):
        batch._run_llm_episode_job(retry)
    assert manifest.read_bytes() == original_binding
    binding = json.loads(original_binding)
    archive = Path(binding["archived_trajectory_dir"])
    next(archive.glob("*.provider_audit.jsonl")).write_text("tampered\n")
    with pytest.raises(batch.ResumeArtifactIntegrityError):
        batch._filter_pending_jobs(
            [job], [row], batch_root=tmp_path, resume_policy="retry-infrastructure"
        )


def test_real_integrity_summary_producer_is_accepted_by_campaign_reader(
    tmp_path, monkeypatch
):
    from scripts import run_eval_campaign as campaign
    from tests.test_run_eval_campaign import setup_worker

    def execute(cmd, **kwargs):
        out = Path(cmd[cmd.index("--output-dir") + 1])

        def fail():
            raise batch.ResumeArtifactIntegrityError(
                [{"reasons": ["fixture_missing"]}], out
            )

        monkeypatch.setattr(batch, "_run_batch_main", fail)
        return SimpleNamespace(returncode=batch.main())

    config, state_path = setup_worker(tmp_path, monkeypatch, execute)
    campaign.worker(config, "tencent", once=True)
    campaign.worker(config, "tencent", once=True)
    saved = json.loads(state_path.read_text())["jobs"]["hy3"]
    assert saved["status"] == "needs_attention"
    assert saved["reason"] == "resume_artifact_integrity_failed"
    assert Path(saved["integrity_summary"]).is_file()
    assert not saved.get("attempt_failures")
    assert not saved.get("interrupted_invocations")


@pytest.mark.parametrize(
    "error_type",
    [
        "RemoteProtocolError",
        "ReadError",
        "ReadTimeout",
        "WriteError",
        "WriteTimeout",
        "ConnectError",
    ],
)
def test_stream_transport_error_rows_remain_retryable(error_type):
    assert batch._retryable_infrastructure_row(
        {
            "status": "error",
            "error_type": "ProviderCircuitOpenError",
            "error_cause_type": error_type,
        }
    )


def test_returned_worker_result_links_exact_start_proof(tmp_path, monkeypatch):
    job, _ = _interrupt_retry(tmp_path, monkeypatch)
    monkeypatch.setattr(batch, "_run_one_safe", lambda _: {"status": "ok"})
    row = batch._run_llm_episode_job(job)
    starts = [
        json.loads(line)
        for line in (tmp_path / "worker_starts.jsonl").read_text().splitlines()
    ]
    assert len(starts) == 2
    assert row["execution_started"] is True
    assert row["execution_attempt_id"] == starts[-1]["execution_attempt_id"]
    assert row["invocation_started_at_utc"] == starts[-1]["invocation_started_at_utc"]
    assert starts[0]["execution_attempt_id"] != starts[1]["execution_attempt_id"]


def test_archived_legacy_relative_paths_do_not_depend_on_original_files_existing(tmp_path, monkeypatch):
    job, row = _interrupt_retry(tmp_path, monkeypatch, relative=True)
    assert batch._filter_pending_jobs([job], [row], batch_root=tmp_path,
                                     resume_policy="retry-infrastructure")
