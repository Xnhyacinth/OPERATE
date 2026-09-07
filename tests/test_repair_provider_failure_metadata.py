"""Evidence-backed amendments to derived provider-failure metadata."""

from __future__ import annotations

import pytest

from baselines.llm_agent import classify_provider_error


@pytest.mark.parametrize(
    "message",
    [
        "Upstream error from Nvidia: Service temporarily overloaded",
        "Service temporarily overloaded",
    ],
)
def test_explicit_provider_overload_is_transient(message):
    assert classify_provider_error(message) == "provider_server_error"


def test_unrelated_overload_word_is_not_a_provider_server_failure():
    assert (
        classify_provider_error("Invalid operation: overloaded circuit")
        == "provider_other_error"
    )


def _batch_fixture(
    tmp_path, message="Upstream error from Nvidia: Service temporarily overloaded"
):
    import hashlib
    import json

    root = tmp_path / "batch"
    root.mkdir()
    (root / ".run.lock").touch()
    audit = root / "trajectory.provider_audit.jsonl"
    records = [
        {
            "record_kind": "provider_request",
            "sequence": 1,
            "envelope": {"request_kind": "decision"},
        },
        {
            "record_kind": "provider_response",
            "request_sequence": 1,
            "response": {
                "status": "failed",
                "error_reason": "provider_other_error",
                "error_summary": message,
            },
        },
    ]
    audit.write_text("".join(json.dumps(row) + "\n" for row in records))
    row = {
        "status": "error",
        "scenario_slug": "case/a",
        "model": "nvidia/model:free",
        "seed": 42,
        "pass_id": "pass-0",
        "scenario_signature": "scenario-sha",
        "temperature": 0,
        "run_semantics_fingerprint": "semantics",
        "implementation_tree_sha256": "tree",
        "agent_treatment_sha256": "treatment",
        "suite_manifest_sha256": "suite",
        "error_type": "ProviderCircuitOpenError",
        "error_cause_type": "APIError",
        "error": "provider circuit opened",
        "execution_started": True,
        "provider_audit_artifact": {
            "path": str(audit),
            "schema_version": "provider_interaction_audit_v1",
            "sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
            "byte_count": audit.stat().st_size,
            "event_count": 2,
        },
    }
    (root / "episodes.jsonl").write_text(json.dumps(row, indent=None) + "\n")
    config = {
        "models": [row["model"]],
        "implementation_tree_sha256": "tree",
        "agent_treatment_sha256_by_model": {row["model"]: "treatment"},
        "scenario_seed_pairs": [["case/a", 42], ["case/b", 43]],
        "pass_k": 1,
        "suite_manifest_sha256": "suite",
    }
    (root / "run_config.json").write_text(json.dumps(config))
    summary = {
        "schema_version": "batch_invocation_v1",
        "status": "completed",
        "started_at_utc": "2026-09-05T00:00:00+00:00",
        "updated_at_utc": "2026-09-05T00:01:00+00:00",
        "resume_policy": "retry-infrastructure",
        "total_scope_jobs": 2,
        "pending_before": 2,
        "pending_after": 1,
        "resume_terminal": 1,
        "infrastructure_failures": 0,
        "terminal_errors": 1,
        "dispatched": 1,
        "dispatched_results": [
            {k: row[k] for k in ["scenario_slug", "model", "seed", "pass_id", "status"]}
        ],
    }
    (root / "invocation_summary.json").write_text(json.dumps(summary))
    return root, row, audit


def test_amendment_is_previewable_append_only_and_idempotent(tmp_path):
    import json
    from scripts import repair_provider_failure_metadata as repair
    from scripts.batch_llm_eval import _retryable_infrastructure_row

    root, original, audit = _batch_fixture(tmp_path)
    original_journal = (root / "episodes.jsonl").read_bytes()
    original_summary = (root / "invocation_summary.json").read_bytes()
    original_audit = audit.read_bytes()
    before_files = sorted(str(path.relative_to(root)) for path in root.rglob("*"))

    preview = repair.repair_metadata(root)
    assert preview["mode"] == "dry_run"
    assert preview["repairs"][0]["case_key"] == [
        "case/a",
        "nvidia/model:free",
        42,
        "pass-0",
    ]
    assert preview["repairs"][0]["restored_http_status"] is None
    assert (root / "episodes.jsonl").read_bytes() == original_journal
    assert (
        sorted(str(path.relative_to(root)) for path in root.rglob("*")) == before_files
    )

    result = repair.repair_metadata(root, apply=True)
    written = (root / "episodes.jsonl").read_bytes()
    assert written.startswith(original_journal)
    amended = json.loads(written.splitlines()[-1])
    assert amended["status"] == "error"
    assert "error_http_status" not in amended
    assert _retryable_infrastructure_row(amended)
    assert amended["metadata_amendment"]["schema_version"] == repair.AMENDMENT_SCHEMA
    for field in [
        "implementation_tree_sha256",
        "agent_treatment_sha256",
        "scenario_signature",
        "execution_started",
    ]:
        assert amended[field] == original[field]
    assert audit.read_bytes() == original_audit
    assert (root / "invocation_summary.json").read_bytes() == original_summary
    backup = root / ".metadata_amendments" / result["amendment_batch_id"]
    assert (backup / "episodes.before.jsonl").read_bytes() == original_journal
    assert (backup / "invocation_summary.before.json").read_bytes() == original_summary
    proposed = result["proposed_invocation_summary"]
    assert proposed["pending_after"] == 2
    assert proposed["resume_terminal"] == 0
    assert proposed["total_scope_jobs"] == 2
    assert proposed["started_at_utc"] == json.loads(original_summary)["started_at_utc"]
    assert proposed["metadata_amendment"]["derived"] is True

    repeated = repair.repair_metadata(root, apply=True)
    assert repeated["repairs"] == []
    assert repeated["proposed_invocation_summary"]["pending_after"] == 2
    assert (root / "episodes.jsonl").read_bytes() == written


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "Streaming response failed: [502] Upstream error from Nvidia: Internal server error",
            502,
        ),
        ("Error code: 503 - Endpoint is unavailable", 503),
        ("HTTP 429 rate limit exceeded", 429),
    ],
)
def test_explicit_transport_status_is_recovered(tmp_path, message, expected):
    import json
    from scripts import repair_provider_failure_metadata as repair

    root, _, _ = _batch_fixture(tmp_path, message)
    result = repair.repair_metadata(root, apply=True)
    assert result["repairs"][0]["restored_http_status"] == expected
    assert result["repairs"][0]["executed_attempt_count_before"] is None
    assert (
        result["repairs"][0]["attempt_budget_action"]
        == "preserve_from_controller_history"
    )
    assert (
        json.loads((root / "episodes.jsonl").read_text().splitlines()[-1])[
            "error_http_status"
        ]
        == expected
    )


@pytest.mark.parametrize("field", ["sha256", "byte_count", "event_count"])
def test_bad_provider_binding_blocks_all_writes(tmp_path, field):
    import json
    from scripts import repair_provider_failure_metadata as repair

    root, row, _ = _batch_fixture(tmp_path)
    row["provider_audit_artifact"][field] = "0" * 64 if field == "sha256" else 999
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    before = (root / "episodes.jsonl").read_bytes()
    with pytest.raises(repair.MetadataRepairError, match="mismatch"):
        repair.repair_metadata(root, apply=True)
    assert (root / "episodes.jsonl").read_bytes() == before
    assert not (root / ".metadata_amendments").exists()


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 400 - invalid function parameters",
        "Invalid model decision with max_tokens=50000",
        "Unknown provider error",
    ],
)
def test_nontransport_errors_are_not_repaired(tmp_path, message):
    from scripts import repair_provider_failure_metadata as repair

    root, _, _ = _batch_fixture(tmp_path, message)
    before = (root / "episodes.jsonl").read_bytes()
    result = repair.repair_metadata(root, apply=True)
    assert result["repairs"] == []
    assert (root / "episodes.jsonl").read_bytes() == before
    assert not (root / ".metadata_amendments").exists()


def test_known_nonretryable_status_cannot_be_overridden_by_overload_text(tmp_path):
    import json
    from scripts import repair_provider_failure_metadata as repair

    root, row, _ = _batch_fixture(tmp_path)
    row["error_http_status"] = 400
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    before = (root / "episodes.jsonl").read_bytes()
    with pytest.raises(repair.MetadataRepairError, match="contradict"):
        repair.repair_metadata(root, apply=True)
    assert (root / "episodes.jsonl").read_bytes() == before


def test_busy_or_missing_lock_never_creates_output_files(tmp_path):
    import fcntl
    from scripts import repair_provider_failure_metadata as repair

    root, _, _ = _batch_fixture(tmp_path)
    before = (root / "episodes.jsonl").read_bytes()
    with (root / ".run.lock").open("r+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(repair.MetadataRepairError, match="held"):
            repair.repair_metadata(root, apply=True)
    assert (root / "episodes.jsonl").read_bytes() == before
    (root / ".run.lock").unlink()
    with pytest.raises(repair.MetadataRepairError, match="existing"):
        repair.repair_metadata(root, apply=True)
    assert not (root / ".run.lock").exists()
    assert not (root / ".metadata_amendments").exists()


def test_latest_success_and_model_protocol_failure_are_not_retried(tmp_path):
    import json
    from scripts import repair_provider_failure_metadata as repair

    root, row, _ = _batch_fixture(tmp_path)
    success = {**row, "status": "ok", "score": {"total_score": 50}}
    with (root / "episodes.jsonl").open("a") as handle:
        handle.write(json.dumps(success) + "\n")
    assert repair.repair_metadata(root, apply=True)["repairs"] == []
    row["error_cause_type"] = "ModelProtocolError"
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    assert repair.repair_metadata(root, apply=True)["repairs"] == []


def test_last_successful_native_response_is_not_relabelled_transport(tmp_path):
    import hashlib
    import json
    from scripts import repair_provider_failure_metadata as repair

    root, row, audit = _batch_fixture(tmp_path, "Error code: 502")
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    records += [
        {
            "record_kind": "provider_request",
            "sequence": 2,
            "envelope": {"request_kind": "decision"},
        },
        {
            "record_kind": "provider_response",
            "request_sequence": 2,
            "response": {"status": "success", "decision_valid": False},
        },
    ]
    audit.write_text("".join(json.dumps(r) + "\n" for r in records))
    row["provider_audit_artifact"].update(
        sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
        byte_count=audit.stat().st_size,
        event_count=4,
    )
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    assert repair.repair_metadata(root, apply=True)["repairs"] == []


def test_prepared_backup_can_be_reused_after_atomic_write_failure(
    tmp_path, monkeypatch
):
    from scripts import repair_provider_failure_metadata as repair

    root, _, _ = _batch_fixture(tmp_path)
    before = (root / "episodes.jsonl").read_bytes()
    with monkeypatch.context() as patch:

        def fail_replace(*_args):
            raise OSError("test disk write failure")

        patch.setattr(repair.os, "replace", fail_replace)
        with pytest.raises(OSError, match="disk write"):
            repair.repair_metadata(root, apply=True)
    assert (root / "episodes.jsonl").read_bytes() == before
    assert not list(root.glob(".metadata-repair-*"))
    assert repair.repair_metadata(root, apply=True)["mode"] == "applied"


def test_batch_agent_bound_semantics_matches_its_run_config(tmp_path):
    import json
    from scripts import repair_provider_failure_metadata as repair

    root, row, _ = _batch_fixture(tmp_path)
    config = json.loads((root / "run_config.json").read_text())
    config["run_semantics_fingerprint"] = "semantics"
    row["run_semantics_fingerprint"] = "semantics:agent-treatment"
    (root / "run_config.json").write_text(json.dumps(config))
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    assert len(repair.repair_metadata(root)["repairs"]) == 1


def test_backup_symlink_cannot_write_outside_batch(tmp_path):
    from scripts import repair_provider_failure_metadata as repair

    root, _, _ = _batch_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / ".metadata_amendments").symlink_to(outside, target_is_directory=True)
    before = (root / "episodes.jsonl").read_bytes()
    with pytest.raises(repair.MetadataRepairError, match="symlink|escapes"):
        repair.repair_metadata(root, apply=True)
    assert list(outside.iterdir()) == []
    assert (root / "episodes.jsonl").read_bytes() == before


def test_unmatched_run_identity_does_not_propose_all_jobs_pending(tmp_path):
    import json
    from scripts import repair_provider_failure_metadata as repair

    root, _, _ = _batch_fixture(tmp_path)
    config = json.loads((root / "run_config.json").read_text())
    config["implementation_tree_sha256"] = "different-tree"
    (root / "run_config.json").write_text(json.dumps(config))
    with pytest.raises(repair.MetadataRepairError, match="scope|identity"):
        repair.repair_metadata(root)


def test_inflight_and_external_provider_audit_are_rejected(tmp_path):
    import json
    from scripts import repair_provider_failure_metadata as repair

    root, row, audit = _batch_fixture(tmp_path)
    with (root / "episodes.jsonl").open("a") as handle:
        handle.write(json.dumps({**row, "status": "in_flight"}) + "\n")
    with pytest.raises(repair.MetadataRepairError, match="in_flight"):
        repair.repair_metadata(root, apply=True)
    outside = tmp_path / "outside-audit.jsonl"
    outside.write_bytes(audit.read_bytes())
    row["provider_audit_artifact"]["path"] = str(outside)
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(repair.MetadataRepairError, match="escapes"):
        repair.repair_metadata(root, apply=True)


def test_legacy_artifact_without_byte_count_retains_measured_bytes(tmp_path):
    import json
    from scripts import repair_provider_failure_metadata as repair

    root, row, audit = _batch_fixture(tmp_path)
    row["provider_audit_artifact"].pop("byte_count")
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    result = repair.repair_metadata(root)
    assert result["repairs"][0]["verified_audit"]["byte_count"] == audit.stat().st_size


def test_verified_audit_mirror_survives_later_attempt_replacement(tmp_path):
    import hashlib
    import json
    from pathlib import Path
    from scripts import repair_provider_failure_metadata as repair

    root, _, audit = _batch_fixture(tmp_path)
    before = audit.read_bytes()
    result = repair.repair_metadata(root, apply=True)
    amended = json.loads((root / "episodes.jsonl").read_text().splitlines()[-1])
    proof = amended["metadata_amendment"]["verified_provider_audit"]
    mirror = Path(proof["mirror"]["path"])
    assert proof["path"] == str(audit)
    assert proof["mirror"]["authoritative"] is False
    assert mirror.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == proof["mirror"]["sha256"]
    assert result["repairs"][0]["verified_audit"]["mirror"]["path"] == str(mirror)

    audit.write_text('{"new_attempt": true}\n')
    repeated = repair.repair_metadata(root, apply=True)
    assert repeated["repairs"] == []
    assert mirror.read_bytes() == before
    assert repeated["proposed_invocation_summary"]["pending_after"] == 2
