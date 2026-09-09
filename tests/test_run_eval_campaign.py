from __future__ import annotations

import pytest
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from scripts import run_eval_campaign as campaign

from scripts.run_eval_campaign import (
    apply_invocation,
    build_command,
    choose_job,
    lane_lock,
    load_credentials,
)


def job(name="hy3"):
    return {
        "id": name, "suite": "full", "model": "hy3-ioa",
        "api_key_env": "API_KEY", "base_url": "https://copilot.tencent.com/v2",
        "context_window": 192000, "max_output": 64000,
        "reasoning_effort": "high", "reasoning_effort_format": "native",
        "thinking_type": "enabled", "rpm": 20, "rpd": 0,
        "quota_scope": "tencent-shared", "workers": 2, "chunk_jobs": 2,
    }


def state():
    return {"jobs": {}, "pools": {}}


def result(**changes):
    row = {"scenario_slug": "case/a", "model": "hy3-ioa", "seed": 42,
           "pass_id": "p1", "status": "ok", "retryable_infrastructure": False,
           "execution_started": True, "scenario_signature": "fixture-signature",
           **{field: "a" * 64 for field in (
               "agent_treatment_sha256", "implementation_tree_sha256",
               "run_semantics_fingerprint", "suite_manifest_sha256", "suite_eligibility_sha256",
           )}}
    row.update(changes)
    return row


def summary(rows, pending=3):
    return {"schema_version": "batch_invocation_v1", "status": "completed",
            "worker_start_contract": "worker_execution_start_v1",
            "pending_after": pending, "total_scope_jobs": 769,
            "resume_terminal": 769-pending, "terminal_errors": 0,
            "scope_attempts_closed": pending == 0, "dispatched_results": rows}


def record_started(out, payload, rows):
    markers = []
    with (out / "worker_starts.jsonl").open("a") as handle:
        for row in rows:
            marker = {
                **row, "schema_version": "worker_execution_start_v1",
                "execution_attempt_id": str(uuid4()),
                "invocation_started_at_utc": payload["started_at_utc"],
                "worker_started_at_utc": datetime.now(UTC).isoformat(),
            }
            markers.append(marker)
            handle.write(json.dumps(marker) + "\n")
    return markers


def test_scope_and_high_ceiling(tmp_path):
    cmd = build_command(tmp_path, tmp_path / "out", job())
    assert "--formal-run" in cmd
    assert cmd[cmd.index("--max-jobs") + 1] == "2"
    assert cmd[cmd.index("--resume-policy") + 1] == "retry-infrastructure"
    assert cmd[cmd.index("--provider-failure-policy") + 1] == "abort"
    lite = job() | {"suite": "lite", "workers": 1, "chunk_jobs": 1}
    cmd = build_command(tmp_path, tmp_path / "out", lite)
    assert "run_lite.py" in cmd[1] and "--formal-run" not in cmd
    with pytest.raises(ValueError, match="high"):
        build_command(tmp_path, tmp_path / "out", job() | {"reasoning_effort": "xhigh"})


def test_round_robin_and_shared_pool_cooldown():
    s = state()
    s["jobs"] = {"hy3": {"last_started": 20}, "luna": {"last_started": 10}}
    jobs = [job(), job("luna")]
    assert choose_job(jobs, s, 30)["id"] == "luna"
    s["pools"]["tencent-shared"] = 100
    assert choose_job(jobs, s, 99) is None
    assert choose_job(jobs, s, 101)["id"] == "luna"
    s["not_before"] = 200
    assert choose_job(jobs, s, 101) is None


def test_transport_backoff_and_bounded_episode_retries():
    s = state()
    data = summary([result(status="error", retryable_infrastructure=True)], pending=1)
    for attempt in range(3):
        apply_invocation(s, job(), data, now=100, cooldown=60, max_attempts=3)
        j = s["jobs"]["hy3"]
        if attempt < 2:
            assert j["not_before"] >= 160
    assert j["status"] == "needs_attention"
    assert choose_job([job()], s, 10000) is None


def test_started_quota_is_suspended_without_transport_charge():
    s = state()
    for _ in range(4):
        apply_invocation(s, job(), summary([result(status="error", quota_parked=True,
                         retryable_infrastructure=True, execution_started=True)]),
                         now=100, cooldown=60, max_attempts=3)
    assert s["jobs"]["hy3"]["attempt_failures"] == {}
    assert s["jobs"]["hy3"]["status"] == "ready"


def test_retry_at_holds_only_one_cell_until_server_deadline():
    s = state()
    apply_invocation(s, job(), summary([result(status="error", retryable_infrastructure=True,
                     retry_at="1970-01-01T00:16:40+00:00")]),
                     now=100, cooldown=60, success_cooldown=0, max_attempts=3)
    row = s["jobs"]["hy3"]
    assert choose_job([job()], s, 101) == job()
    assert list(row["attempt_failures"].values()) == [1]
    assert list(row["held_cells"].values()) == ["provider_cooldown"]
    choose_job([job()], s, 1000)
    assert row["held_cells"] == {}


def test_exhausted_cell_and_repair_cell_leave_other_cells_runnable():
    s = state()
    for _ in range(3):
        apply_invocation(s, job(), summary([result(status="error", retryable_infrastructure=True)]),
                         now=100, cooldown=60, max_attempts=3)
    apply_invocation(s, job(), summary([result(scenario_slug="case/b", status="error",
                     needs_repair=True, termination_category="harness_error")]),
                     now=200, cooldown=60, max_attempts=3)
    row = s["jobs"]["hy3"]
    assert row["status"] == "ready"
    assert len(row["held_cells"]) == 2
    assert choose_job([job()], s, 10000) == job()


def test_attempt_ledger_restores_erased_state_and_extension_is_additive(tmp_path):
    s = state()
    ledger = tmp_path / "attempts.jsonl"
    payload = summary([result(status="error", retryable_infrastructure=True,
                             execution_attempt_id="attempt-one")])
    for _ in range(2):
        apply_invocation(s, job(), payload, now=100, cooldown=60, max_attempts=3,
                         ledger_path=ledger)
    key = campaign.attempt_key(payload["dispatched_results"][0])
    assert s["jobs"]["hy3"]["attempt_failures"][key] == 1
    s["jobs"]["hy3"]["attempt_failures"] = {}
    campaign.restore_attempt_ledger(ledger, s)
    assert s["jobs"]["hy3"]["attempt_failures"][key] == 1
    campaign.extend_attempt_budget(ledger, s, "hy3", key, 2, "transport repaired")
    campaign.restore_attempt_ledger(ledger, s)
    assert s["jobs"]["hy3"]["attempt_failures"][key] == 1
    assert s["jobs"]["hy3"]["attempt_extensions"][key] == 2
    assert json.loads(ledger.read_text().splitlines()[-1])["event"] == "attempt_budget_extended"


def test_attempt_ledger_imports_archives_even_after_counter_reset(tmp_path):
    s = state()
    s["jobs"]["hy3"] = {"attempt_failures": {}}
    for index in range(4):
        payload = summary([result(status="error", retryable_infrastructure=True,
                                  execution_attempt_id=f"attempt-{index}")])
        (tmp_path / f"hy3-{index}.summary.json").write_text(json.dumps(payload))
    ledger = campaign.initialize_attempt_ledger(tmp_path, s)
    assert list(s["jobs"]["hy3"]["attempt_failures"].values()) == [4]
    s["jobs"]["hy3"]["attempt_failures"] = {}
    campaign.initialize_attempt_ledger(tmp_path, s)
    assert len(ledger.read_text().splitlines()) == 4
    assert list(s["jobs"]["hy3"]["attempt_failures"].values()) == [4]


def test_worker_passes_cell_holds_without_hiding_remaining_scope(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out = campaign.Path(cmd[cmd.index("--output-dir") + 1])
        holds = json.loads(campaign.Path(cmd[cmd.index("--held-cells") + 1]).read_text())
        if len(calls) == 1:
            assert holds == {"cells": []}
            payload = summary([result(status="error", needs_repair=True)], pending=2)
        else:
            assert holds["cells"][0]["scenario_slug"] == "case/a"
            payload = summary([result(scenario_slug="case/b")], pending=1)
        payload["started_at_utc"] = datetime.now(UTC).isoformat()
        (out / "invocation_summary.json").write_text(json.dumps(payload))
        return SimpleNamespace(returncode=2)

    config, state_path = setup_worker(tmp_path, monkeypatch, fake_run)
    for _ in range(3):
        campaign.worker(config, "tencent", once=True)
    assert len(calls) == 2
    saved = json.loads(state_path.read_text())["jobs"]["hy3"]
    assert saved["status"] == "needs_attention"
    assert saved["pending"] == 1


def test_success_spacing_can_be_removed_without_removing_error_backoff():
    s = state()
    campaign.apply_invocation(s, job(), summary([result()]), now=100,
                              cooldown=600, success_cooldown=0, max_attempts=3)
    assert s["jobs"]["hy3"]["not_before"] == 100
    campaign.apply_invocation(s, job(), summary([result(status="error", retryable_infrastructure=True)]),
                              now=200, cooldown=600, success_cooldown=0, max_attempts=3)
    assert s["jobs"]["hy3"]["not_before"] == 800


def test_quota_parking_does_not_spend_attempt_budget():
    s = state()
    row = result(status="error", retryable_infrastructure=True, quota_parked=True,
                 execution_started=False, quota_reset_at="2030-01-01T00:00:00Z")
    apply_invocation(s, job(), summary([row]), now=100, cooldown=60, max_attempts=3)
    assert not s["jobs"]["hy3"]["attempt_failures"]
    assert s["pools"]["tencent-shared"] > 1_000_000_000


def test_model_failure_is_terminal_without_retry_until_success():
    s = state()
    data = summary([result(status="error", retryable_infrastructure=False)], pending=0)
    data["terminal_errors"] = 1
    apply_invocation(s, job(), data, now=100, cooldown=60, max_attempts=3)
    assert s["jobs"]["hy3"]["status"] == "attempts_closed"
    assert s["jobs"]["hy3"]["terminal_errors"] == 1
    assert not s["jobs"]["hy3"]["attempt_failures"]


def test_worker_lock_rejects_duplicate(tmp_path):
    with lane_lock(tmp_path / "worker.lock"):
        with pytest.raises(BlockingIOError):
            with lane_lock(tmp_path / "worker.lock"):
                pass


def test_batch_lock_probe_does_not_initialize_a_new_namespace(tmp_path):
    out = tmp_path / "batch"
    out.mkdir()
    assert not campaign.batch_locked(out)
    assert list(out.iterdir()) == []


def test_credentials_are_literal_and_never_execute(tmp_path):
    rc = tmp_path / "zshrc"
    rc.write_text('export API_KEY="literal-value"\nexport BAD="$(touch leaked)"\n')
    assert load_credentials(["API_KEY"], rc, {}) == {"API_KEY": "literal-value"}
    with pytest.raises(ValueError, match="literal"):
        load_credentials(["BAD"], rc, {})
    assert not (tmp_path / "leaked").exists()


def setup_worker(tmp_path, monkeypatch, fake_run):
    config = {"runtime_root": str(tmp_path/"runtime"), "max_episode_attempts": 3,
              "lanes": {"tencent": {"cooldown_s": 0, "jobs": [job()]}}}
    path = tmp_path/"campaign.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(campaign, "verify_bindings", lambda *args: None)
    monkeypatch.setattr(campaign, "load_credentials", lambda *args: {"API_KEY": "test"})
    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    return path, tmp_path/"queue/tencent/state.json"


def test_worker_formal_subdir_closure_then_failed_finalizer_does_not_loop(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out = campaign.Path(cmd[cmd.index("--output-dir")+1])/"treatment-test"
        out.mkdir(parents=True, exist_ok=True)
        payload = summary([result()], pending=0)
        payload["started_at_utc"] = datetime.now(UTC).isoformat()
        if "--finalize" in cmd:
            payload["status"] = "running"
        (out/"invocation_summary.json").write_text(json.dumps(payload))
        return SimpleNamespace(returncode=1 if "--finalize" in cmd else 0)

    config, state_path = setup_worker(tmp_path, monkeypatch, fake_run)
    campaign.worker(config, "tencent", once=True)
    assert json.loads(state_path.read_text())["jobs"]["hy3"]["needs_finalize"]
    campaign.worker(config, "tencent", once=True)
    campaign.worker(config, "tencent", once=True)
    saved = json.loads(state_path.read_text())
    assert saved["jobs"]["hy3"]["status"] == "reports_need_attention"
    assert not saved["jobs"]["hy3"]["needs_finalize"]
    assert len(calls) == 2


def test_interruption_and_transport_share_case_attempt_budget(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out = campaign.Path(cmd[cmd.index("--output-dir")+1])
        payload = summary([result(status="error", retryable_infrastructure=True)], pending=1)
        payload["started_at_utc"] = datetime.now(UTC).isoformat()
        if len(calls) in {1, 3}:
            payload["status"] = "running"
            record_started(out, payload, payload["dispatched_results"])
        (out/"invocation_summary.json").write_text(json.dumps(payload))
        return SimpleNamespace(returncode=-15 if len(calls) in {1, 3} else 2)

    config, state_path = setup_worker(tmp_path, monkeypatch, fake_run)
    for _ in range(4):
        campaign.worker(config, "tencent", once=True)
    saved = json.loads(state_path.read_text())["jobs"]["hy3"]
    assert len(calls) == 3
    assert saved["status"] == "needs_attention"
    assert list(saved["attempt_failures"].values()) == [3]


def test_stale_summary_is_not_accepted_after_preflight_failure(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        return SimpleNamespace(returncode=1)

    config, state_path = setup_worker(tmp_path, monkeypatch, fake_run)
    out = tmp_path/"results/hy3"
    out.mkdir(parents=True)
    payload = summary([result()], pending=0)
    payload["started_at_utc"] = datetime.now(UTC).isoformat()
    (out/"invocation_summary.json").write_text(json.dumps(payload))
    campaign.worker(config, "tencent", once=True)
    saved = json.loads(state_path.read_text())["jobs"]["hy3"]
    assert saved["status"] == "needs_attention"
    assert not saved.get("needs_finalize")


def test_metadata_repair_hook_preserves_scope(tmp_path, monkeypatch):
    original = summary([result(status="error")], pending=0)
    original["terminal_errors"] = 1
    original["started_at_utc"] = datetime.now(UTC).isoformat()
    proposed = json.loads(json.dumps(original))
    proposed.update(pending_after=1, resume_terminal=768, scope_attempts_closed=False)
    proposed["dispatched_results"][0]["retryable_infrastructure"] = True
    config = {"metadata_repair_script": str(tmp_path/"repair.py")}
    monkeypatch.setattr(campaign.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=json.dumps({"proposed_invocation_summary": proposed}), stderr=""))
    repaired = campaign.repair_invocation_summary(config, tmp_path, original, tmp_path/"step.log")
    assert repaired["pending_after"] == 1
    assert repaired["dispatched_results"][0]["retryable_infrastructure"]
    assert original["pending_after"] == 0
    proposed["total_scope_jobs"] = 99
    with pytest.raises(ValueError, match="scope"):
        campaign.repair_invocation_summary(config, tmp_path, original, tmp_path/"step.log")


def test_stop_after_current_invocation_still_closes_its_summary(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out = campaign.Path(cmd[cmd.index("--output-dir") + 1])
        payload = summary([result()], pending=3)
        payload["started_at_utc"] = datetime.now(UTC).isoformat()
        (out / "invocation_summary.json").write_text(json.dumps(payload))
        (tmp_path / "queue/tencent/STOP").touch()
        return SimpleNamespace(returncode=0)

    config, state_path = setup_worker(tmp_path, monkeypatch, fake_run)
    campaign.worker(config, "tencent")
    saved = json.loads(state_path.read_text())
    assert saved["worker_status"] == "stopped"
    assert "active" not in saved
    assert saved["jobs"]["hy3"]["terminal"] == 766
    assert len(calls) == 1


def test_unverifiable_resume_evidence_parks_lane_without_spending_model_attempt(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        out = campaign.Path(cmd[cmd.index("--output-dir") + 1])
        payload = {
            "schema_version": "batch_invocation_v1", "status": "needs_attention",
            "reason": "resume_artifact_integrity_failed",
            "started_at_utc": datetime.now(UTC).isoformat(),
            "integrity_failures": [{"scenario_slug": "case/a", "reason": "provider_audit_missing"}],
        }
        (out / "invocation_summary.json").write_text(json.dumps(payload))
        return SimpleNamespace(returncode=1)

    config, state_path = setup_worker(tmp_path, monkeypatch, fake_run)
    campaign.worker(config, "tencent", once=True)
    campaign.worker(config, "tencent", once=True)
    saved = json.loads(state_path.read_text())["jobs"]["hy3"]
    assert saved["status"] == "needs_attention"
    assert saved["reason"] == "resume_artifact_integrity_failed"
    assert not saved.get("attempt_failures")
    assert not saved.get("interrupted_invocations")
    assert len(calls) == 1


def test_interrupted_chunk_charges_only_actual_worker_starts(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        out = campaign.Path(cmd[cmd.index("--output-dir") + 1])
        rows = [result(scenario_slug=name, execution_started=None) for name in (
            "started", "never-submitted", "also-unstarted",
        )]
        payload = summary(rows)
        payload.update(status="running", started_at_utc=datetime.now(UTC).isoformat())
        record_started(out, payload, rows[:1])
        (out / "invocation_summary.json").write_text(json.dumps(payload))
        return SimpleNamespace(returncode=-15)

    config, state_path = setup_worker(tmp_path, monkeypatch, fake_run)
    campaign.worker(config, "tencent", once=True)
    failures = json.loads(state_path.read_text())["jobs"]["hy3"]["attempt_failures"]
    assert len(failures) == 1
    assert json.loads(next(iter(failures)))[0] == "started"


def test_interrupted_chunk_without_start_proof_requires_attention(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        out = campaign.Path(cmd[cmd.index("--output-dir") + 1])
        payload = summary([result(execution_started=None)])
        payload.update(status="running", started_at_utc=datetime.now(UTC).isoformat())
        (out / "invocation_summary.json").write_text(json.dumps(payload))
        return SimpleNamespace(returncode=-15)

    config, state_path = setup_worker(tmp_path, monkeypatch, fake_run)
    campaign.worker(config, "tencent", once=True)
    saved = json.loads(state_path.read_text())["jobs"]["hy3"]
    assert saved["status"] == "needs_attention"
    assert saved["reason"] == "interrupted_start_evidence_invalid"
    assert not saved.get("attempt_failures")


@pytest.mark.parametrize("terminal_status", ["ok", "error"])
def test_interruption_does_not_charge_settled_success_or_model_failure(tmp_path, monkeypatch, terminal_status):
    def fake_run(cmd, **kwargs):
        out = campaign.Path(cmd[cmd.index("--output-dir") + 1])
        rows = [result(scenario_slug=name, execution_started=None) for name in ("settled", "unfinished")]
        payload = summary(rows)
        payload.update(status="running", started_at_utc=datetime.now(UTC).isoformat())
        markers = record_started(out, payload, rows)
        terminal = {**markers[0], "status": terminal_status, "error_type": "ValueError"}
        (out / "episodes.jsonl").write_text(json.dumps(terminal) + "\n")
        (out / "invocation_summary.json").write_text(json.dumps(payload))
        return SimpleNamespace(returncode=-15)

    config, state_path = setup_worker(tmp_path, monkeypatch, fake_run)
    campaign.worker(config, "tencent", once=True)
    failures = json.loads(state_path.read_text())["jobs"]["hy3"]["attempt_failures"]
    assert len(failures) == 1 and json.loads(next(iter(failures)))[0] == "unfinished"


def test_verified_empty_start_journal_charges_no_episode(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        out = campaign.Path(cmd[cmd.index("--output-dir") + 1])
        payload = summary([result(execution_started=None)])
        payload.update(status="running", started_at_utc=datetime.now(UTC).isoformat())
        (out / "worker_starts.jsonl").touch()
        (out / "invocation_summary.json").write_text(json.dumps(payload))
        return SimpleNamespace(returncode=-15)

    config, state_path = setup_worker(tmp_path, monkeypatch, fake_run)
    campaign.worker(config, "tencent", once=True)
    saved = json.loads(state_path.read_text())["jobs"]["hy3"]
    assert saved["status"] == "ready" and not saved["attempt_failures"]
