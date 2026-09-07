import json
from pathlib import Path

import pytest

from scripts import batch_realtime_llm_eval as batch
from tests.test_batch_realtime_llm_eval import (
    _identity,
    _job,
    _artifact,
    _episode_identity,
)


def config(tmp_path):
    return batch.initialize_run_directory(tmp_path, _identity())[1]


def test_observed_ineligible_model_outcome_is_not_automatically_retried(tmp_path):
    cfg = config(tmp_path)
    job = _job(cfg["batch_treatment_sha256"])
    path = (
        Path(cfg["output_dir"])
        / "trajectories"
        / f"treatment-{cfg['batch_treatment_sha256']}"
        / "episode.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_artifact(_episode_identity())))
    row = {
        **job,
        "status": "ineligible",
        "artifact_path": path.relative_to(Path(cfg["output_dir"])).as_posix(),
        "artifact_sha256": batch.file_sha256(path),
    }
    assert batch._campaign_pending_jobs([job], [row], cfg) == []


def test_damaged_failed_artifact_stops_resume_instead_of_regeneration(tmp_path):
    cfg = config(tmp_path)
    job = _job(cfg["batch_treatment_sha256"])
    path = (
        Path(cfg["output_dir"])
        / "trajectories"
        / f"treatment-{cfg['batch_treatment_sha256']}"
        / "episode.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(_artifact(_episode_identity())))
    row = {
        **job,
        "status": "ineligible",
        "artifact_path": path.relative_to(Path(cfg["output_dir"])).as_posix(),
        "artifact_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="artifact integrity"):
        batch._campaign_pending_jobs([job], [row], cfg)


def test_infrastructure_without_artifact_stays_pending(tmp_path):
    cfg = config(tmp_path)
    job = _job(cfg["batch_treatment_sha256"])
    assert batch._campaign_pending_jobs(
        [job], [{**job, "status": "infrastructure_error"}], cfg
    ) == [job]


def test_realtime_cli_parses_chunk_scope_without_provider(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(
        batch,
        "load_formal_contract",
        lambda path: (
            called.append(path)
            or (_ for _ in ()).throw(ValueError("stop before provider"))
        ),
    )
    rc = batch.main(
        [
            "--suite",
            "suite.json",
            "--formal-manifest",
            "manifest.json",
            "--output-root",
            str(tmp_path),
            "--model",
            "hy3-ioa",
            "--model-context-window-tokens",
            "192000",
            "--model-max-output-tokens",
            "64000",
            "--suite-kind",
            "lite",
            "--max-jobs",
            "12",
            "--no-finalize",
            "--resume-policy",
            "retry-infrastructure",
            "--dry-run",
        ]
    )
    assert rc == 1 and called


def test_lite_yaml_path_is_canonicalized_for_child_runner(tmp_path):
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "path": "scenarios/new/example.yaml",
                        "scenario_id": "example",
                        "scenario_signature": "a" * 16,
                        "horizon_ticks": 2,
                        "seed": 42,
                    }
                ]
            }
        )
    )
    assert batch._load_suite(suite)[0]["scenario_slug"] == "new/example"


def test_damaged_collection_is_attention_not_normal_closed(tmp_path):
    cfg = config(tmp_path)
    job = _job(cfg["batch_treatment_sha256"])
    path = (
        Path(cfg["output_dir"])
        / "trajectories"
        / f"treatment-{cfg['batch_treatment_sha256']}"
        / "episode.json"
    )
    path.parent.mkdir(parents=True)
    artifact = _artifact(_episode_identity())
    artifact["evidence_closure"]["closure_complete"] = False
    path.write_text(json.dumps(artifact))
    row = batch.terminal_row_from_artifact(job, path, cfg)
    with pytest.raises(ValueError, match="collection or contract"):
        batch._campaign_pending_jobs([job], [row], cfg)


def test_recovered_wire_failure_does_not_become_unrecovered_episode_failure():
    artifact = {
        "turns": [{"turn_id": "t1", "status": "completed"}],
        "provider_audit": [
            {
                "turn_id": "t1",
                "provider_responses": [
                    {
                        "response": {
                            "status": "failed",
                            "error_reason": "provider_transport_error",
                        }
                    },
                    {"response": {"status": "success"}},
                ],
            }
        ],
    }
    assert batch._unrecovered_transport_failure(artifact) is False
    artifact["turns"][0]["status"] = "failed"
    artifact["provider_audit"][0]["provider_responses"].pop()
    assert batch._unrecovered_transport_failure(artifact) is True


def test_exact_released_lite_selection_binds_source_contracts():
    root = batch.REPO_ROOT
    manifest = root / "release/operate_v0_61_0/manifest.json"
    rows, selection = batch._select_suite(
        manifest.parent / "lite_suite.json",
        "lite",
        {"formal_runtime_binding": {"release_id": "operate_v0_61_0"}},
        manifest,
    )
    assert len(rows) == 193
    assert selection["formal_full_leaderboard_eligible"] is False
    assert all(row["scenario_slug"].startswith("operate_") for row in rows)
    assert all(
        row["case_ledger"]["source_denominator_key"] == row["source_denominator_key"]
        for row in rows
    )


def test_worker_start_uuid_is_joined_to_terminal_and_unique_log(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from uuid import UUID

    cfg = config(tmp_path)
    out = Path(cfg["output_dir"])
    job = batch._build_jobs(
        [
            {
                "scenario_slug": "datacenter/example",
                "scenario_id": "dc_example_s42",
                "scenario_signature": "d" * 64,
                "seed": 42,
                "horizon_ticks": 4,
            }
        ],
        out,
        cfg,
    )[0]
    job["invocation_started_at_utc"] = "2026-09-06T00:00:00Z"

    def watchdog(command, *, log_path, **kwargs):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("local stub only")
        return {
            "log_path": str(log_path),
            "returncode": 1,
            "orphaned": False,
            "timed_out": False,
        }

    monkeypatch.setattr(batch, "run_subprocess_with_watchdog", watchdog)
    monkeypatch.setattr(batch, "_find_artifact", lambda _: None)
    row = batch._execute_job(
        job,
        cfg,
        SimpleNamespace(api_key_env="FAKE", base_url=None, responses_base_url=None),
    )
    [marker] = batch._load_jsonl(out / "worker_starts.jsonl")
    assert UUID(marker["execution_attempt_id"])
    assert marker["execution_attempt_id"] == row["execution_attempt_id"]
    assert marker["invocation_started_at_utc"] == row["invocation_started_at_utc"]
    assert marker["implementation_tree_sha256"] == row["implementation_tree_sha256"]
    assert marker["execution_attempt_id"] in row["subprocess"]["log_path"]
    assert row["subprocess"]["log_sha256"]


def test_lite_child_command_carries_revalidated_source_binding(tmp_path):
    from types import SimpleNamespace

    cfg = config(tmp_path)
    job = _job(cfg["batch_treatment_sha256"])
    job.update(
        trajectory_dir=str(tmp_path / "trajectory"),
        construct_contract="operational_agency.v1",
        source_denominator_key="source-1",
        case_ledger={"source_denominator_key": "source-1"},
        lite_core_lineage={"join": "exact_path_signature_seed"},
    )
    command = batch._command_for_job(
        job,
        cfg,
        SimpleNamespace(api_key_env="FAKE", base_url=None, responses_base_url=None),
    )
    binding = json.loads(command[command.index("--scenario-contract-binding") + 1])
    assert binding["lite_core_lineage"] == job["lite_core_lineage"]


def test_bounded_main_resume_never_repeats_completed_scope(tmp_path, monkeypatch):
    from copy import deepcopy

    release = tmp_path / "release" / "operate"
    release.mkdir(parents=True)
    suite = release / "readiness.json"
    suite.write_text(
        json.dumps(
            {
                "suite_manifest_sha256": "a" * 64,
                "scenarios": [
                    {
                        "scenario_slug": f"datacenter/example-{i}",
                        "scenario_id": f"dc_example_{i}",
                        "scenario_signature": "d" * 64,
                        "seed": 42,
                        "horizon_ticks": 4,
                    }
                    for i in range(2)
                ],
            }
        )
    )
    manifest = release / "manifest.json"
    manifest.write_text("{}")
    binding = {
        "release_id": "operate",
        "release_tooling_sha256": "1" * 64,
        "manifest_path": str(manifest),
        "manifest_sha256": "b" * 64,
        "readiness_path": str(suite),
        "readiness_sha256": batch.file_sha256(suite),
        "core_release_pipeline_sha256": "e" * 64,
        "backend_runtime_closure_identity_sha256": "f" * 64,
    }
    formal = {
        "selection_path": str(suite),
        "selection_sha256": batch.file_sha256(suite),
        "formal_runtime_binding": binding,
        "manifest_sha256": "b" * 64,
        "agentic_profile": deepcopy(batch.CANONICAL_AGENTIC_PROFILE),
        "realtime_contract": {
            "suite_manifest_sha256": "a" * 64,
            "clock_profile": {
                "tick_interval_s": 0.25,
                "episode_timeout_policy": batch.EPISODE_TIMEOUT_POLICY,
                "process_hard_timeout_overhead_s": 30.0,
                "termination_grace_s": 5.0,
            },
        },
    }
    monkeypatch.setattr(batch, "load_formal_contract", lambda _: formal)
    monkeypatch.setattr(
        batch,
        "implementation_identity",
        lambda *_: {"implementation_tree_sha256": "c" * 64},
    )
    monkeypatch.setattr(batch, "_require_clean_git_tree", lambda: None)
    monkeypatch.setenv("LOCAL_STUB_KEY", "not-a-real-key")
    executed = []

    def execute(job, cfg, args):
        executed.append(job["job_key"])
        artifact = _artifact(_episode_identity())
        artifact.update(scenario_id=job["scenario_id"])
        path = Path(job["trajectory_dir"]) / "realtime_stub.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact))
        row = batch.terminal_row_from_artifact(job, path, cfg)
        assert row["status"] == "ok", row["eligibility_reasons"]
        return row

    monkeypatch.setattr(batch, "_execute_job", execute)
    argv = [
        "--suite",
        str(suite),
        "--formal-manifest",
        str(manifest),
        "--output-root",
        str(tmp_path / "out"),
        "--model",
        "hy3-ioa",
        "--base-url",
        "https://copilot.tencent.com/v2?token=secret",
        "--api-key-env",
        "LOCAL_STUB_KEY",
        "--model-context-window-tokens",
        "192000",
        "--model-max-output-tokens",
        "65536",
        "--max-tokens",
        "65536",
        "--max-workers",
        "4",
        "--max-jobs",
        "1",
        "--no-finalize",
        "--resume",
    ]
    assert batch.main(argv) == 0
    summary_path = next((tmp_path / "out").glob("treatment-*/invocation_summary.json"))
    first = json.loads(summary_path.read_text())
    assert (first["dispatched"], first["pending_after"], first["resume_terminal"]) == (
        1,
        1,
        1,
    )
    assert batch.main(argv) == 0
    second = json.loads(summary_path.read_text())
    assert second["scope_attempts_closed"] is True
    assert batch.main(argv) == 0
    assert len(executed) == len(set(executed)) == 2
    assert json.loads(summary_path.read_text())["dispatched"] == 0


def test_batch_provider_validator_uses_proven_retry_chain():
    from tests.test_realtime_recovered_wire_retry import retry_row

    row = retry_row()
    artifact = {
        "provider_audit": [row],
        "llm_interaction_stats": {
            "provider_model_identity_records": row["provider_model_identities"],
            "provider_model_identity_request_count": 2,
            "provider_model_identity_closed_count": 2,
            "provider_model_identity_exact_count": 1,
            "provider_model_identity_failed_request_count": 1,
            "provider_model_identity_missing_count": 0,
            "provider_model_identity_mismatch_count": 0,
        },
    }
    assert batch._provider_evidence_reasons(artifact, requested_model="model") == []
    row["provider_requests"][1]["sha256"] = "0" * 64
    assert "provider_response_failed" in batch._provider_evidence_reasons(
        artifact, requested_model="model"
    )
