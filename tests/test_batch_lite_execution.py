from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import batch_llm_eval as batch


def _bind_saved_artifacts(job: dict, row: dict) -> dict:
    """Mock execution still fulfills the requested saved-artifact contract."""
    root = Path(job["trajectory_dir"])
    root.mkdir(parents=True, exist_ok=True)
    prefix = root / "fixture"
    summary = row.setdefault("trajectory_summary", {})
    summary["trajectory_path"] = str(prefix)
    for name, suffix, schema in (
        ("trajectory_artifact", "trajectory", "episode_trajectory_jsonl_v1"),
        ("evidence_ledger_artifact", "evidence", "evidence_ledger_jsonl_v1"),
        ("provider_audit_artifact", "provider_audit", "provider_interaction_audit_v1"),
        ("semantic_ledger_artifact", "semantic_ledger", "semantic_session_ledger_v1"),
    ):
        payload = b'{}\n'
        path = Path(f"{prefix}.{suffix}.jsonl")
        path.write_bytes(payload)
        summary[name] = {"path": str(path), "schema_version": schema,
                         "sha256": hashlib.sha256(payload).hexdigest(),
                         "event_count": 1, "byte_count": len(payload)}
    return row


def test_diagnostic_abort_profile_reaches_worker_config_and_identity() -> None:
    args = argparse.Namespace(
        api_key_env="TEST_PROVIDER_KEY",
        api_mode="chat_completions",
        formal_run=False,
        provider_failure_policy="abort",
        max_consecutive_provider_failures=1,
    )
    config = batch._batch_llm_config(
        model="test-model",
        temperature=0.0,
        args=args,
        base_url="https://example.test/v1",
        api_version=None,
        responses_base_url=None,
    )
    assert config.provider_failure_policy == "abort"
    assert config.max_consecutive_provider_failures == 1
    identity = batch._agent_treatment_identity(config)
    assert identity["provider_failure_policy"] == "abort"
    assert identity["max_consecutive_provider_failures"] == 1


def _job(slug: str) -> dict:
    return {
        "scenario_slug": slug, "model": "test-model", "seed": 42,
        "scenario_signature": slug, "temperature": 0.0, "pass_id": "pass-0",
        "run_semantics_fingerprint": "profile-a", "suite_manifest_sha256": "suite-a",
        "suite_eligibility_sha256": "eligibility-a", "implementation_tree_sha256": "tree-a",
        "agent_treatment_sha256": "treatment-a",
    }


def test_resume_retains_model_failures_but_retries_transient_infrastructure() -> None:
    jobs = [_job(str(i)) for i in range(7)]
    rows = [
        {**jobs[0], "status": "ok", "task_completion": {"completed": False}},
        {**jobs[1], "status": "error", "error_type": "ValueError",
         "error": "action_required: invalid model protocol"},
        {**jobs[2], "status": "error", "error_type": "ProviderCircuitOpenError",
         "trajectory_summary": {"llm": {"failed_tick_log": [
             {"reason": "provider_rate_limit", "exc_type": "RateLimitError"}
         ]}}},
        {**jobs[3], "status": "error", "error_type": "APITimeoutError"},
        {**jobs[4], "status": "in_flight"},
        {**jobs[5], "status": "error", "error_type": "AuthenticationError"},
    ]
    pending = batch._filter_pending_jobs(jobs, rows, resume_policy="retry-infrastructure")
    assert pending == [jobs[i] for i in (2, 3, 4, 6)]


@pytest.mark.parametrize("changed", [
    {"implementation_tree_sha256": "tree-b"},
    {"agent_treatment_sha256": "treatment-b"},
    {"scenario_signature": "sig-b"},
    {"suite_manifest_sha256": "suite-b"},
    {"pass_id": "pass-1"},
])
def test_terminal_resume_never_reuses_other_execution_identity(changed: dict) -> None:
    job = _job("scenario")
    row = {**job, "status": "ok", **changed}
    assert batch._filter_pending_jobs(
        [job], [row], resume_policy="retry-infrastructure"
    ) == [job]


@pytest.mark.parametrize("order", ["longest-first", "shortest-first"])
def test_one_job_invocations_keep_full_scope_and_do_not_retry_model_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    order: str,
) -> None:
    scenarios = ["test/short", "test/long"]
    monkeypatch.setattr(batch, "_resolve_patterns", lambda *_: scenarios)
    monkeypatch.setattr(batch, "_expand_scenarios", lambda _: scenarios)
    monkeypatch.setattr(batch, "load_scenario_yaml", lambda slug: {
        "seed": 42, "scenario_signature": slug, "backend_kind": "mock",
        "horizon_ticks": 2 if slug.endswith("long") else 1,
    })
    monkeypatch.setattr(batch, "_suite_eligibility_binding", lambda _: {"suite_blocked": False})
    monkeypatch.setattr(batch, "_load_zhsrc_exports", lambda: {})
    monkeypatch.setattr(batch, "_load_named_zshrc_export", lambda _: None)
    monkeypatch.setattr(batch, "implementation_identity", lambda _: {
        "implementation_tree_sha256": "stable-tree",
    })
    monkeypatch.setattr(batch, "_git_metadata", lambda: {
        "git_metadata_available": True, "git_commit": "fixed", "git_dirty": False,
    })
    monkeypatch.setenv("TEST_LITE_KEY", "local-test-only")
    dispatched = []

    def execute(jobs, path, mode, workers):
        assert workers == 1
        assert len(jobs) == 1
        for job in jobs:
            dispatched.append(job["scenario_slug"])
            row = batch._apply_llm_job_metadata(job, {
                "status": "ok" if len(dispatched) == 1 else "error",
                "task_completion": {"completed": False},
                "error_type": None if len(dispatched) == 1 else "ValueError",
                "termination_category": None if len(dispatched) == 1 else "model_failure",
                "implementation_tree_sha256": job["implementation_tree_sha256"],
            })
            batch._append_jsonl_atomic(path, _bind_saved_artifacts(job, row))

    monkeypatch.setattr(batch, "_run_global_jobs", execute)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", [
        "run_lite.py", "--output-dir", str(out), "--models", "test-model",
        "--interaction-mode", "logical_stateless", "--max-workers", "1",
        "--api-key-env", "TEST_LITE_KEY", "--base-url-env", "TEST_LITE_BASE",
        "--api-version-env", "TEST_LITE_VERSION", "--responses-base-url-env", "TEST_LITE_RESPONSES",
        "--provider-failure-policy", "abort", "--max-consecutive-provider-failures", "1",
        "--max-jobs", "1", "--no-finalize", "--resume-policy", "retry-infrastructure",
        "--reasoning-effort", "high", "--reasoning-effort-format", "native",
        "--thinking-type", "enabled",
        "--job-order", order,
    ])
    assert batch.main() == 0
    first_config = json.loads((out / "run_config.json").read_text())
    first = json.loads((out / "invocation_summary.json").read_text())
    assert first["total_scope_jobs"] == 2
    assert first["pending_before"] == 2
    assert first["dispatched"] == 1
    assert first["pending_after"] == 1
    assert first["resume_terminal"] == 1
    assert first_config["reasoning_effort_format"] == "native"
    assert first_config["thinking_type"] == "enabled"
    assert batch.main() == 2
    assert batch.main() == 2
    last_config = json.loads((out / "run_config.json").read_text())
    last = json.loads((out / "invocation_summary.json").read_text())
    assert dispatched == (scenarios if order == "shortest-first" else scenarios[::-1])
    assert last["status"] == "completed"
    assert last["pending_before"] == last["pending_after"] == last["dispatched"] == 0
    assert last["resume_terminal"] == 2
    assert last["terminal_errors"] == 1
    assert last_config["n_scenarios"] == first_config["n_scenarios"] == 2
    assert last_config["suite_manifest_sha256"] == first_config["suite_manifest_sha256"]
    assert last_config["agent_treatment_sha256_by_model"] == first_config["agent_treatment_sha256_by_model"]
    assert last_config.get("job_order", "longest-first") == order
    original = (out / "run_config.json").read_bytes()
    sys.argv[-1] = "shortest-first" if order == "longest-first" else "longest-first"
    assert batch.main() == 1
    assert (out / "run_config.json").read_bytes() == original


@pytest.mark.parametrize("http_status,retryable", [(429, True), (503, True), (401, False), (400, False)])
def test_circuit_error_keeps_machine_readable_cause_for_retry(
    monkeypatch: pytest.MonkeyPatch, http_status: int, retryable: bool,
) -> None:
    import run
    from baselines.llm_agent import ProviderCircuitOpenError
    from runner import batch as runner_batch

    class UpstreamError(Exception):
        status_code = http_status

    def fail(**_kwargs):
        try:
            raise UpstreamError("upstream request failed")
        except UpstreamError as exc:
            raise ProviderCircuitOpenError("provider circuit opened") from exc

    monkeypatch.setattr(run, "load_scenario_yaml", lambda _: {"seed": 42})
    monkeypatch.setattr(runner_batch, "run_one", fail)
    row = runner_batch.run_one_safe(("fixture", "llm_agent", 42, {}))
    assert row["error_type"] == "ProviderCircuitOpenError"
    assert row["error_cause_type"] == "UpstreamError"
    assert row["error_http_status"] == http_status
    assert batch._retryable_infrastructure_row(row) is retryable


def test_quota_parked_result_is_visible_without_consuming_execution_budget(tmp_path: Path) -> None:
    job = _job("quota-cell")
    row = batch._quota_parked_result(job, reset_at="2026-09-06T00:00:00Z")
    summary = batch._invocation_summary(
        [job], [job], [row], pending_before=1, resume_policy="retry-infrastructure",
        batch_root=tmp_path, started_at_utc="2026-09-05T00:00:00Z", status="completed",
    )
    assert summary["pending_after"] == 1
    result = summary["dispatched_results"][0]
    assert result["status"] == "error"
    assert result["quota_parked"] is True
    assert result["execution_started"] is False
    assert result["quota_reset_at"] == "2026-09-06T00:00:00Z"


def test_provider_thinking_configuration_survives_workers_and_changes_identity() -> None:
    args = argparse.Namespace(
        api_key_env="TEST_PROVIDER_KEY", api_mode="chat_completions", formal_run=False,
        reasoning_effort="high", reasoning_effort_format="native", thinking_type="enabled",
    )
    config = batch._batch_llm_config(
        model="test-model", temperature=0.0, args=args,
        base_url="https://example.test/v1", api_version=None, responses_base_url=None,
    )
    assert getattr(config, "reasoning_effort_format", None) == "native"
    assert getattr(config, "thinking_type", None) == "enabled"
    restored = batch._llm_config_from_dict(batch._llm_config_to_dict(config))
    assert restored.reasoning_effort_format == "native"
    assert restored.thinking_type == "enabled"
    original_hash = batch._agent_treatment_sha256(restored)
    restored.thinking_type = "disabled"
    assert batch._agent_treatment_sha256(restored) != original_hash
    restored.thinking_type = "enabled"
    restored.reasoning_effort_format = "openrouter"
    assert batch._agent_treatment_sha256(restored) != original_hash


@pytest.mark.parametrize("flags", [
    ["--max-jobs", "0"],
    ["--max-consecutive-provider-failures", "0"],
    ["--formal-run", "--provider-failure-policy", "compat_fallback"],
    ["--formal-run", "--max-consecutive-provider-failures", "2"],
])
def test_invalid_controls_fail_before_initializing_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, flags: list[str],
) -> None:
    output = tmp_path / "blocked"
    monkeypatch.setattr(sys, "argv", ["batch", "--output-dir", str(output), *flags])
    assert batch.main() == 1
    assert not output.exists()


def test_running_invocation_does_not_relabel_previous_attempt_as_new_result(tmp_path: Path) -> None:
    job = _job("retry")
    previous = {**job, "status": "error", "error_type": "APITimeoutError", "execution_started": True}
    summary = batch._invocation_summary(
        [job], [job], [previous], pending_before=1, resume_policy="retry-infrastructure",
        batch_root=tmp_path, started_at_utc="2026-09-05T00:00:00Z", status="running",
    )
    assert summary["dispatched_results"][0]["status"] == "pending"
    assert summary["dispatched_results"][0]["execution_started"] is None


def test_formal_bounded_chunks_keep_manifest_scope_and_require_publication_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    scenarios = ["test/full-short", "test/full-long"]
    readiness = {
        "formal_evaluation_ready": True, "suite_blocked": False,
        "suite_manifest_sha256": "fixture-suite", "scoring_version": batch.SCORING_VERSION,
        "primary_leaderboard_formula_version": batch.PRIMARY_LEADERBOARD_FORMULA_VERSION,
        "primary_inference_version": batch.PRIMARY_INFERENCE_VERSION,
        "task_completion_input_unit": batch.TASK_COMPLETION_INPUT_UNIT,
        "task_completion_score_unit": batch.TASK_COMPLETION_SCORE_UNIT,
        "weighted_equity_formula_version": batch.WEIGHTED_EQUITY_FORMULA_VERSION,
        "formal_run_contract": {
            "contract_version": "agentic_persistent.v1",
            "wakeup_policy": dict(batch.CANONICAL_WAKEUP_POLICY),
            "requires_explicit_model_capabilities": True,
            "agentic_profile": {
                "provider_failure_policy": "abort", "max_consecutive_provider_failures": 1,
            },
        },
    }
    binding = {
        "slice_name": "fixture-full", "dynamic_slice_spec": ("fixture", "unused.json", {}),
        "release_id": "fixture-full", "manifest_path": str(tmp_path / "manifest.json"),
        **{name: "a" * 64 for name in (
            "manifest_sha256", "release_tooling_sha256", "readiness_sha256",
            "core_release_pipeline_sha256", "backend_runtime_closure_identity_sha256",
        )},
    }
    monkeypatch.setattr(batch, "DYNAMIC_SCENARIO_SLICES", {})
    monkeypatch.setattr(batch, "resolve_formal_manifest_slice", lambda _: binding)
    monkeypatch.setattr(batch, "_resolve_patterns", lambda *_: scenarios)
    monkeypatch.setattr(batch, "_expand_scenarios", lambda _: scenarios)
    monkeypatch.setattr(batch, "load_scenario_yaml", lambda slug: {
        "seed": 42, "scenario_signature": slug, "backend_kind": "mock",
        "construct_contract": "operational_agency.v1",
        "horizon_ticks": 2 if slug.endswith("long") else 1,
    })
    monkeypatch.setattr(batch, "_bind_scenario_contracts_for_slice", lambda *_: None)
    monkeypatch.setattr(batch, "_suite_manifest_sha256_for_slice", lambda *_: "fixture-suite")
    monkeypatch.setattr(batch, "_suite_eligibility_binding", lambda _: readiness)
    monkeypatch.setattr(batch, "_formal_runtime_binding_reasons", lambda _: [])
    monkeypatch.setattr(batch, "_load_zhsrc_exports", lambda: {})
    monkeypatch.setattr(batch, "_load_named_zshrc_export", lambda _: None)
    monkeypatch.setattr(batch, "implementation_identity", lambda _: {"implementation_tree_sha256": "b" * 64})
    monkeypatch.setattr(batch, "_git_metadata", lambda: {
        "git_metadata_available": True, "git_commit": "fixed", "git_dirty": False,
    })
    monkeypatch.setenv("TEST_FULL_KEY", "local-test-only")
    dispatched = []

    def execute(jobs, path, mode, workers):
        assert len(jobs) == 1
        for job in jobs:
            dispatched.append(job["scenario_slug"])
            batch._append_jsonl_atomic(path, _bind_saved_artifacts(job, batch._apply_llm_job_metadata(job, {
                "status": "ok", "task_completion": {"completed": False},
                "implementation_tree_sha256": job["implementation_tree_sha256"],
            })))

    monkeypatch.setattr(batch, "_run_global_jobs", execute)
    finalized = []

    def finalize(out_dir, rows, meta):
        assert len(rows) == meta["n_scenarios"] == 2
        gate = batch._formal_leaderboard_eligibility(meta, {}, {}, {}, {})
        assert not gate["eligible"]
        assert "formal_coverage_incomplete" in gate["blockers"]
        assert "formal_agentic_profile_provider_failure_policy_mismatch" not in gate["blockers"]
        assert "formal_agentic_profile_max_consecutive_provider_failures_mismatch" not in gate["blockers"]
        finalized.append(gate)
        return {"leaderboard_eligible": False, "n_episodes_ok": 2,
                "n_episodes_error": 0, "artifacts": {"plots": []}}

    monkeypatch.setattr(batch, "_finalize_outputs", finalize)
    root = tmp_path / "formal-output"
    monkeypatch.setattr(sys, "argv", [
        "batch", "--output-dir", str(root), "--models", "test-model",
        "--formal-run", "--formal-manifest", str(tmp_path / "manifest.json"),
        "--max-workers", "1", "--max-jobs", "1", "--job-order", "shortest-first",
        "--resume-policy", "retry-infrastructure", "--no-finalize", "--seed-mode", "scenario",
        "--api-key-env", "TEST_FULL_KEY", "--base-url-env", "TEST_FULL_BASE",
        "--api-version-env", "TEST_FULL_VERSION", "--responses-base-url-env", "TEST_FULL_RESPONSES",
        "--model-context-window-tokens", "192000", "--model-max-output-tokens", "64000",
        "--max-tokens", "64000",
    ])
    assert batch.main() == 2
    leaf, = root.glob("treatment-*")
    config = json.loads((leaf / "run_config.json").read_text())
    first = json.loads((leaf / "invocation_summary.json").read_text())
    assert config["formal_run"] is True
    assert config["n_scenarios"] == first["total_scope_jobs"] == 2
    assert first["pending_after"] == 1 and not first["formal_completion_claimed"]
    assert batch.main() == 2
    second = json.loads((leaf / "invocation_summary.json").read_text())
    assert second["scope_attempts_closed"] and not second["formal_completion_claimed"]
    assert dispatched == scenarios
    assert not finalized
    sys.argv[sys.argv.index("--no-finalize")] = "--finalize"
    assert batch.main() == 2
    assert len(finalized) == 1
    final = json.loads((leaf / "invocation_summary.json").read_text())
    assert final["dispatched"] == 0 and not final["formal_completion_claimed"]
    assert json.loads((leaf / "run_config.json").read_text())["n_scenarios"] == 2
