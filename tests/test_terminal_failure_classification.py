from scripts import batch_llm_eval as batch


def test_closed_or_harness_failure_is_not_retried_for_old_transport_log():
    history = {
        "llm": {
            "failed_tick_log": [
                {
                    "reason": "provider_transport_error",
                    "exc_type": "RemoteProtocolError",
                }
            ]
        }
    }
    for terminal in ({"status": "ok"}, {"status": "error", "error_type": "ValueError"}):
        assert not batch._retryable_infrastructure_row(
            {**terminal, "trajectory_summary": history}
        )


def test_native_contract_error_is_reported_as_needing_repair(monkeypatch):
    import run
    from runner import batch as runner_batch

    def fail(**kwargs):
        raise ValueError(
            "native outcome_tick must be the current completed step boundary"
        )

    monkeypatch.setattr(run, "load_scenario_yaml", lambda _: {})
    monkeypatch.setattr(runner_batch, "run_one", fail)
    row = runner_batch.run_one_safe(("route", "llm_agent", 42, {}))
    assert row["termination_category"] == "harness_error"
    assert row["needs_repair"] is True
    assert not batch._retryable_infrastructure_row(row)


def test_held_and_repair_cells_keep_scope_incomplete_without_blocking_other_work(
    tmp_path,
):
    from tests.test_batch_lite_execution import _job

    held, repair, ready = [_job(name) for name in ("held", "repair", "ready")]
    held["campaign_hold"] = True
    row = {
        **repair,
        "status": "error",
        "termination_category": "harness_error",
        "needs_repair": True,
    }
    summary = batch._invocation_summary(
        [held, repair, ready],
        [],
        [row],
        pending_before=3,
        resume_policy="retry-infrastructure",
        batch_root=tmp_path,
        started_at_utc="2026-09-08T00:00:00Z",
        status="completed",
    )
    assert summary["total_scope_jobs"] == summary["pending_after"] == 3
    assert summary["held_count"] == 2
    assert summary["runnable_pending"] == 1
    assert summary["scope_attempts_closed"] is False


def test_retry_budget_survives_worker_roundtrip_and_changes_treatment():
    import argparse

    args = argparse.Namespace(
        api_key_env="TEST",
        api_mode="chat_completions",
        formal_run=False,
        provider_retry_max_attempts=12,
        provider_retry_max_elapsed_s=900.0,
    )
    cfg = batch._batch_llm_config(
        model="fixture",
        temperature=0,
        args=args,
        base_url="https://example.test/v1",
        api_version=None,
        responses_base_url=None,
    )
    restored = batch._llm_config_from_dict(batch._llm_config_to_dict(cfg))
    assert restored.provider_retry_max_attempts == 12
    assert restored.provider_retry_max_elapsed_s == 900.0
    before = batch._agent_treatment_sha256(restored)
    restored.provider_retry_max_attempts = 13
    assert batch._agent_treatment_sha256(restored) != before


def test_harness_fault_keeps_batch_partial_even_with_all_cells_accounted_for():
    state = batch._batch_state(
        coverage={"configured_models": ["fixture"], "is_partial_batch": False},
        results=[
            {
                "status": "error",
                "termination_category": "harness_error",
                "needs_repair": True,
            }
        ],
        log_audit_report=None,
    )
    assert state["batch_state"] == "partial"
    assert state["n_episodes_harness_error"] == 1
    assert any("repair" in reason for reason in state["reasons"])


def test_batch_worker_forwards_checkpoint_identity_without_changing_it(
    monkeypatch, tmp_path
):
    import run
    from runner import batch as runner_batch

    seen = {}

    def execute(**kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(run, "load_scenario_yaml", lambda _: {})
    monkeypatch.setattr(runner_batch, "run_one", execute)
    identity = {"implementation_tree_sha256": "fixture-tree", "seed": 42}
    checkpoint = tmp_path / "checkpoint.jsonl"
    row = runner_batch.run_one_safe(
        (
            "fixture",
            "llm_agent",
            42,
            {},
            {
                "checkpoint_path": str(checkpoint),
                "checkpoint_identity": identity,
            },
        )
    )
    assert row["status"] == "ok"
    assert seen["checkpoint_path"] == checkpoint
    assert seen["checkpoint_identity"] == identity


def test_artifact_repair_isolated_from_independent_pending_cell(tmp_path):
    from tests.test_resume_artifact_integrity import _bound_attempt
    from tests.test_batch_lite_execution import _job

    job, row = _bound_attempt(tmp_path)
    from pathlib import Path

    Path(row["trajectory_summary"]["trajectory_artifact"]["path"]).write_text(
        "tampered\n"
    )
    ready = _job("ready")
    repairs = []
    pending = batch._filter_pending_jobs(
        [job, ready],
        [row],
        batch_root=tmp_path,
        resume_policy="retry-infrastructure",
        repair_failures=repairs,
    )
    assert pending == [ready]
    assert len(repairs) == 1 and repairs[0]["needs_repair"]
    assert repairs[0]["scenario_slug"] == job["scenario_slug"]


def test_legacy_native_value_error_remains_visible_as_repair_cell(tmp_path):
    from tests.test_batch_lite_execution import _job

    job = _job("legacy-citylearn")
    row = {
        **job,
        "status": "error",
        "error_type": "ValueError",
        "error": "ValueError: actual objective must be finite and non-negative",
    }
    summary = batch._invocation_summary(
        [job],
        [],
        [row],
        pending_before=0,
        resume_policy="retry-infrastructure",
        batch_root=tmp_path,
        started_at_utc="2026-09-08T00:00:00Z",
        status="completed",
    )
    assert summary["pending_after"] == summary["needs_repair_count"] == 1
    assert summary["runnable_pending"] == 0
    assert summary["repair_cells"][0]["scenario_slug"] == job["scenario_slug"]


def test_campaign_forwards_recovery_contract_into_new_batch(tmp_path):
    from scripts import run_eval_campaign as campaign
    from tests.test_run_eval_campaign import job

    spec = {
        **job(),
        "provider_retry_max_attempts": 12,
        "provider_retry_max_elapsed_s": 900,
        "episode_checkpoint": True,
    }
    command = campaign.build_command(tmp_path, tmp_path / "out", spec)
    assert command[command.index("--provider-retry-max-attempts") + 1] == "12"
    assert command[command.index("--provider-retry-max-elapsed-s") + 1] == "900"
    assert "--episode-checkpoint" in command


def test_interrupted_configuration_error_does_not_reopen_provider_lane(tmp_path):
    import json
    from datetime import UTC, datetime
    from scripts import run_eval_campaign as campaign
    from tests.test_run_eval_campaign import result, summary, record_started

    payload = summary([result()])
    payload.update(status="running", started_at_utc=datetime.now(UTC).isoformat())
    marker = record_started(tmp_path, payload, payload["dispatched_results"])[0]
    terminal = {
        **marker,
        "status": "error",
        "error_type": "ProviderModelIdentityError",
        "termination_category": "provider_configuration_error",
    }
    (tmp_path / "episodes.jsonl").write_text(json.dumps(terminal) + "\n")
    charged, blocked = campaign.interrupted_attempts(
        tmp_path, {"started_at": datetime.now(UTC).timestamp() - 1}, payload
    )
    assert charged == []
    assert blocked


def test_legacy_quota_counter_is_not_relabelled_as_transport_charge(tmp_path):
    import json
    from scripts import run_eval_campaign as campaign
    from tests.test_run_eval_campaign import result, summary

    quota = result(
        status="error",
        retryable_infrastructure=True,
        quota_parked=True,
        execution_attempt_id="quota-only",
    )
    key = campaign.attempt_key(quota)
    state = {"jobs": {"hy3": {"attempt_failures": {key: 1}, "pending": 1}}, "pools": {}}
    (tmp_path / "hy3-1.summary.json").write_text(json.dumps(summary([quota])))
    ledger = campaign.initialize_attempt_ledger(tmp_path, state)
    row = state["jobs"]["hy3"]
    assert row["attempt_failures"] == {}
    assert row["legacy_unattributed_budget"][key] == 1
    assert all(
        json.loads(line)["event"] != "attempt_charged"
        for line in ledger.read_text().splitlines()
    )
    campaign.update_cell_holds(row, 3)
    assert row["held_cells"][key] == "legacy_attempts_unattributed"
    campaign.extend_attempt_budget(
        ledger, state, "hy3", key, 1, "authorize retry while preserving old ambiguity"
    )
    campaign.restore_attempt_ledger(ledger, state)
    campaign.update_cell_holds(row, 3)
    assert key not in row["held_cells"]
    assert row["attempt_failures"] == {}
    assert row["legacy_unattributed_budget"][key] == 1


def test_portable_sidecars_do_not_rewrite_hash_bound_scoring_inputs(tmp_path):
    import hashlib
    from pathlib import Path
    from data import TrajectoryLogger

    root = tmp_path / "trajectories" / "cell"
    logger = TrajectoryLogger("fixture", root)
    binding = logger.write_snapshot(
        "scoring_inputs", {"historical_path": str(root / "source")}
    )
    original = Path(binding["path"]).read_bytes()
    batch._portabilize_formal_trajectory_json_sidecars(
        {
            "formal_run": True,
            "batch_output_dir": str(tmp_path),
            "trajectory_dir": str(root),
        }
    )
    assert Path(binding["path"]).read_bytes() == original
    assert hashlib.sha256(original).hexdigest() == binding["sha256"]


def test_checkpoint_mode_is_bound_to_run_semantics():
    ordinary = batch._run_semantics_fingerprint("strict", 64000, "logical_persistent")
    recovery = batch._run_semantics_fingerprint(
        "strict", 64000, "logical_persistent", episode_checkpoint=True
    )
    assert recovery != ordinary
    assert "logical_episode_checkpoint_v1" in recovery
