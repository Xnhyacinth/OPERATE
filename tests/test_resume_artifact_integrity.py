from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import batch_llm_eval as batch
from tests.test_batch_llm_eval import (
    _bind_formal_artifacts,
    _formally_eligible_protocol21_row,
)


def _bound_attempt(tmp_path):
    row = _bind_formal_artifacts(
        _formally_eligible_protocol21_row(), tmp_path, key="integrity"
    )
    row.update(
        scenario_slug="fixture",
        model="model",
        seed=42,
        scenario_signature="signature",
        temperature=0.0,
        pass_id="pass-0",
        implementation_tree_sha256="tree",
    )
    job = {
        key: row[key]
        for key in (
            "scenario_slug",
            "model",
            "seed",
            "scenario_signature",
            "temperature",
            "pass_id",
            "implementation_tree_sha256",
            "agent_treatment_sha256",
            "suite_manifest_sha256",
            "suite_eligibility_sha256",
        )
    }
    job["evaluation_implementation_fingerprint"] = (
        batch.EVALUATION_IMPLEMENTATION_FINGERPRINT
    )
    job["trajectory_dir"] = str(tmp_path)
    job["llm_config"] = {"interaction_mode": "logical_persistent"}
    return job, row


@pytest.mark.parametrize("damage", ["delete", "corrupt", "remove_binding"])
def test_retry_infrastructure_stops_for_repair_of_bound_success(tmp_path, damage):
    job, row = _bound_attempt(tmp_path)
    assert (
        batch._filter_pending_jobs(
            [job], [row], batch_root=tmp_path, resume_policy="retry-infrastructure"
        )
        == []
    )
    artifact = row["trajectory_summary"]["provider_audit_artifact"]
    if damage == "delete":
        Path(artifact["path"]).unlink()
    elif damage == "corrupt":
        Path(artifact["path"]).write_text("{}\n")
    else:
        del row["trajectory_summary"]["provider_audit_artifact"]
    with pytest.raises(ValueError, match="resume_artifact_integrity_failed"):
        batch._filter_pending_jobs(
            [job], [row], batch_root=tmp_path, resume_policy="retry-infrastructure"
        )


def test_valid_model_failure_is_retained_despite_formal_ineligibility(tmp_path):
    job, row = _bound_attempt(tmp_path)
    row["trajectory_summary"]["terminal_integrity"]["release_ready"] = False
    row["trajectory_summary"]["llm"]["native_tool_protocol_invalid_responses"] = 1
    assert (
        batch._filter_pending_jobs(
            [job], [row], batch_root=tmp_path, resume_policy="retry-infrastructure"
        )
        == []
    )


def test_partial_error_bound_artifacts_are_not_skipped(tmp_path):
    job, row = _bound_attempt(tmp_path)
    row.update(status="error", error_stage="interaction_loop", error_type="APIError")
    row["provider_audit_artifact"] = row["trajectory_summary"][
        "provider_audit_artifact"
    ]
    del row["trajectory_summary"]
    Path(row["provider_audit_artifact"]["path"]).unlink()
    with pytest.raises(ValueError, match="resume_artifact_integrity_failed"):
        batch._filter_pending_jobs(
            [job], [row], batch_root=tmp_path, resume_policy="retry-infrastructure"
        )


def test_main_reports_repair_attention_instead_of_closed_scope(tmp_path, monkeypatch):
    def fail():
        raise batch.ResumeArtifactIntegrityError(
            [
                {
                    "scenario_slug": "fixture",
                    "reasons": ["provider_audit_artifact:unreadable"],
                }
            ],
            tmp_path,
        )

    monkeypatch.setattr(batch, "_run_batch_main", fail)
    assert batch.main() == 1
    summary = json.loads((tmp_path / "invocation_summary.json").read_text())
    assert summary["status"] == "needs_attention"
    assert summary["reason"] == "resume_artifact_integrity_failed"
    assert summary["scope_attempts_closed"] is False
    assert summary["formal_completion_claimed"] is False


def test_conflicting_top_level_binding_cannot_hide_corrupt_artifact(tmp_path):
    job, row = _bound_attempt(tmp_path)
    row["provider_audit_artifact"] = {
        **row["trajectory_summary"]["provider_audit_artifact"],
        "sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="resume_artifact_integrity_failed"):
        batch._filter_pending_jobs(
            [job], [row], batch_root=tmp_path, resume_policy="retry-infrastructure"
        )


def test_dry_run_integrity_failure_does_not_rewrite_existing_summary(
    tmp_path, monkeypatch
):
    import sys

    path = tmp_path / "invocation_summary.json"
    path.write_text('{"status":"prior"}\n')

    def fail():
        error = batch.ResumeArtifactIntegrityError([{"reasons": ["missing"]}], tmp_path)
        error.dry_run = True
        raise error

    monkeypatch.setattr(batch, "_run_batch_main", fail)
    monkeypatch.setattr(sys, "argv", ["batch_llm_eval.py", "--dry-run"])
    assert batch.main() == 1
    assert path.read_text() == '{"status":"prior"}\n'


@pytest.mark.parametrize("flag", ["--dry", "--dry-r"])
def test_parsed_dry_run_abbreviation_preserves_summary(tmp_path, monkeypatch, flag):
    import sys

    output = tmp_path / "dry-run"
    monkeypatch.setattr(batch, "_resolve_patterns", lambda *_: ["fixture"])
    monkeypatch.setattr(batch, "_expand_scenarios", lambda _: ["fixture"])
    monkeypatch.setattr(
        batch,
        "load_scenario_yaml",
        lambda _: {
            "seed": 42,
            "scenario_signature": "sig",
            "backend_kind": "mock",
            "horizon_ticks": 1,
        },
    )
    monkeypatch.setattr(
        batch, "_suite_eligibility_binding", lambda _: {"suite_blocked": False}
    )
    monkeypatch.setattr(batch, "_load_zhsrc_exports", lambda: {})
    monkeypatch.setattr(batch, "_load_named_zshrc_export", lambda _: None)
    monkeypatch.setattr(
        batch,
        "implementation_identity",
        lambda _: {"implementation_tree_sha256": "fixed"},
    )
    monkeypatch.setattr(
        batch,
        "_git_metadata",
        lambda: {
            "git_metadata_available": True,
            "git_commit": "fixed",
            "git_dirty": False,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch",
            "--output-dir",
            str(output),
            "--models",
            "fixture",
            "--interaction-mode",
            "logical_stateless",
            "--resume-policy",
            "retry-infrastructure",
            "--base-url-env",
            "TEST_DRY_BASE",
            "--api-version-env",
            "TEST_DRY_VERSION",
            "--responses-base-url-env",
            "TEST_DRY_RESPONSES",
            flag,
        ],
    )
    assert batch.main() == 0
    summary = output / "invocation_summary.json"
    summary.write_text('{"status":"prior"}\n')

    def fail(*args, **kwargs):
        raise batch.ResumeArtifactIntegrityError([{"reasons": ["missing"]}], output)

    monkeypatch.setattr(batch, "_filter_pending_jobs", fail)
    assert batch.main() == 1
    assert summary.read_text() == '{"status":"prior"}\n'
