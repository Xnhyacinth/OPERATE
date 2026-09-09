from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest


def _encoded(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _hash(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _records(statuses):
    requests, responses = [], []
    for index, status in enumerate(statuses, 1):
        envelope = {"model": "hy3-ioa", "messages": [{"content": f"request-{index}"}]}
        response = {
            "status": status,
            "model_identity_closure": {
                "schema_version": "provider_model_identity_closure_v1",
                "request_sequence": index,
                "requested_model": "hy3-ioa",
                "observed_models": ["hy3-ioa"],
                "closure": "request_failed" if status == "failed" else "exact",
            },
        }
        if status == "failed":
            response["error_reason"] = "provider_transport_error"
        requests.append(
            {
                "record_kind": "provider_request",
                "sequence": index,
                "tick": index,
                "envelope": envelope,
                "sha256": _hash(envelope),
            }
        )
        responses.append(
            {
                "record_kind": "provider_response",
                "sequence": index,
                "request_sequence": index,
                "tick": index,
                "response": response,
                "sha256": _hash(response),
            }
        )
    return requests + responses


def _row(directory, attempt, statuses, reused):
    directory.mkdir(parents=True, exist_ok=True)
    records = _records(statuses)
    payload = b"".join(_encoded(record) + b"\n" for record in records)
    path = directory / "cell.provider_audit.jsonl"
    path.write_bytes(payload)
    identity = {
        key: "bound"
        for key in (
            "scenario_slug",
            "scenario_signature",
            "agent_treatment_sha256",
            "implementation_tree_sha256",
            "run_semantics_fingerprint",
            "suite_manifest_sha256",
            "suite_eligibility_sha256",
        )
    }
    return {
        **identity,
        "model": "hy3-ioa",
        "seed": 7,
        "pass_id": "pass-0",
        "execution_attempt_id": attempt,
        "status": "error" if statuses[-1] == "failed" else "ok",
        "error_type": "ProviderCircuitOpenError" if statuses[-1] == "failed" else None,
        "error_cause_type": "RemoteProtocolError" if statuses[-1] == "failed" else None,
        "checkpoint_progress": {"reused_provider_request_count": reused},
        "provider_audit_artifact": {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "event_count": len(records),
            "schema_version": "provider_interaction_audit_v1",
        },
    }


def test_recovery_counts_new_calls_without_counting_replayed_prefix(tmp_path):
    from runner.recovery_audit import build_recovery_audit, validate_recovery_audit

    old = _row(
        tmp_path / "cell.stale-20260908T100000Z",
        "old",
        ["success", "success", "failed"],
        0,
    )
    current = _row(tmp_path / "cell", "new", ["success"] * 4, 2)
    result = build_recovery_audit([], old, current, tmp_path, "hy3-ioa")
    assert result["closed"] is True
    assert result["eligible"] is True
    assert result["logical_provider_request_count"] == 4
    assert result["totals"]["request_attempts"] == 5
    assert result["totals"]["failed_requests"] == 1
    assert [item["new_request_count"] for item in result["attempts"]] == [3, 2]
    current["recovery_audit"] = result
    assert validate_recovery_audit(current, tmp_path, tmp_path / "cell") == []


@pytest.mark.parametrize(
    "fault",
    [
        "parent_missing",
        "prefix_altered",
        "unsettled",
        "artifact_tamper",
        "wrong_cell",
        "wrong_identity",
    ],
)
def test_recovery_audit_fails_closed_for_unknown_or_divergent_history(tmp_path, fault):
    from runner.recovery_audit import build_recovery_audit

    old = _row(tmp_path / "cell.stale-t1", "old", ["success", "success", "failed"], 0)
    current = _row(tmp_path / "cell", "new", ["success"] * 4, 2)
    if fault == "parent_missing":
        old = None
    elif fault == "wrong_cell":
        old = _row(
            tmp_path / "different-cell", "old", ["success", "success", "failed"], 0
        )
    elif fault == "wrong_identity":
        old["seed"] = 8
    elif fault == "artifact_tamper":
        Path(old["provider_audit_artifact"]["path"]).write_bytes(b"modified\n")
    else:
        target = current if fault == "prefix_altered" else old
        binding = target["provider_audit_artifact"]
        path = Path(binding["path"])
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if fault == "prefix_altered":
            records[0]["envelope"]["messages"][0]["content"] = "different request"
            records[0]["sha256"] = _hash(records[0]["envelope"])
        else:
            records.pop()
        payload = b"".join(_encoded(record) + b"\n" for record in records)
        path.write_bytes(payload)
        binding.update(
            sha256=hashlib.sha256(payload).hexdigest(), event_count=len(records)
        )
    audit = build_recovery_audit([], old, current, tmp_path, "hy3-ioa")
    assert audit["closed"] is False
    assert audit["eligible"] is False
    if fault == "parent_missing":
        assert audit["totals"]["request_attempts"] is None


def test_three_attempt_history_survives_only_same_cell_archive_relocation(tmp_path):
    from runner.recovery_audit import build_recovery_audit, validate_recovery_audit

    first = _row(
        tmp_path / "cell.stale-t1", "first", ["success", "success", "failed"], 0
    )
    second = _row(tmp_path / "cell", "second", ["success"] * 3 + ["failed"], 2)
    second["recovery_audit"] = build_recovery_audit(
        [], first, second, tmp_path, "hy3-ioa"
    )
    (tmp_path / "cell").rename(tmp_path / "cell.stale-t2")
    relocated = str(tmp_path / "cell.stale-t2" / "cell.provider_audit.jsonl")
    second["provider_audit_artifact"]["path"] = relocated
    second["recovery_audit"]["attempts"][-1]["provider_audit_artifact"]["path"] = (
        relocated
    )
    assert validate_recovery_audit(second, tmp_path, tmp_path / "cell") == []
    third = _row(tmp_path / "cell", "third", ["success"] * 5, 3)
    audit = build_recovery_audit(
        second["recovery_audit"]["attempts"], second, third, tmp_path, "hy3-ioa"
    )
    third["recovery_audit"] = audit
    assert audit["closed"] is True
    assert audit["totals"]["request_attempts"] == 7
    assert audit["totals"]["failed_requests"] == 2
    assert validate_recovery_audit(third, tmp_path, tmp_path / "cell") == []
    tampered = copy.deepcopy(third)
    tampered["recovery_audit"]["totals"]["request_attempts"] = 5
    assert validate_recovery_audit(tampered, tmp_path, tmp_path / "cell")


def test_known_harness_failure_is_closed_but_not_eligible(tmp_path):
    from runner.recovery_audit import build_recovery_audit

    old = _row(tmp_path / "cell.stale-t1", "old", ["success", "success", "failed"], 0)
    old.update(
        error_type="ValueError",
        error_cause_type=None,
        termination_category="harness_error",
    )
    current = _row(tmp_path / "cell", "new", ["success"] * 4, 2)
    audit = build_recovery_audit([], old, current, tmp_path, "hy3-ioa")
    assert audit["closed"] is True
    assert audit["eligible"] is False


def test_hard_kill_marker_cannot_claim_complete_request_accounting(tmp_path):
    from runner.recovery_audit import build_recovery_audit

    old = _row(tmp_path / "cell.stale-t1", "old", ["success", "success", "failed"], 0)
    old["status"] = "in_flight"
    current = _row(tmp_path / "cell", "new", ["success"] * 4, 2)
    assert (
        build_recovery_audit([], old, current, tmp_path, "hy3-ioa")["closed"] is False
    )


def test_bound_prior_history_cannot_drop_an_attempt_with_no_progress(tmp_path):
    from runner.recovery_audit import build_recovery_audit

    first = _row(
        tmp_path / "cell.stale-t1", "first", ["success", "success", "failed"], 0
    )
    second = _row(
        tmp_path / "cell.stale-t2", "second", ["success", "success", "failed"], 2
    )
    second["recovery_audit"] = build_recovery_audit(
        [],
        first,
        second,
        tmp_path,
        "hy3-ioa",
        expected_trajectory_dir=tmp_path / "cell",
    )
    current = _row(tmp_path / "cell", "third", ["success"] * 4, 2)
    audit = build_recovery_audit(
        second["recovery_audit"]["attempts"][:1], second, current, tmp_path, "hy3-ioa"
    )
    assert audit["closed"] is False
    assert "prior_bound_history_mismatch" in audit["reasons"]


@pytest.mark.parametrize(
    "local_audit",
    [
        {"request_budget": {"status": "preflight_rejected"}},
        {"provider_rate_limit": {"status": "daily_quota_exhausted"}},
        {"provider_rate_limit": {"status": "state_error"}},
        {"provider_rate_limit": {"status": "wait_interrupted"}},
        {"provider_rate_limit": {"status": "canceled_before_reservation"}},
        {"provider_retry_budget": {"remaining_s": 0}},
    ],
)
def test_local_rejections_are_not_presented_as_network_calls(tmp_path, local_audit):
    from runner.recovery_audit import build_recovery_audit

    row = _row(tmp_path / "cell", "first", ["failed"], 0)
    records = _records(["failed"])
    records[0]["envelope"].update(local_audit)
    records[0]["sha256"] = _hash(records[0]["envelope"])
    payload = b"".join(_encoded(record) + b"\n" for record in records)
    binding = row["provider_audit_artifact"]
    Path(binding["path"]).write_bytes(payload)
    binding["sha256"] = hashlib.sha256(payload).hexdigest()
    audit = build_recovery_audit([], None, row, tmp_path, "hy3-ioa")
    assert audit["closed"] is True
    assert audit["totals"]["request_attempts"] == 1
    assert audit["totals"]["known_local_rejections"] == 1


def test_known_model_mismatch_history_is_never_eligible(tmp_path):
    from runner.recovery_audit import build_recovery_audit

    row = _row(tmp_path / "cell", "first", ["success"], 0)
    records = _records(["success"])
    records[-1]["response"]["model_identity_closure"]["observed_models"] = [
        "other-model"
    ]
    records[-1]["sha256"] = _hash(records[-1]["response"])
    payload = b"".join(_encoded(record) + b"\n" for record in records)
    binding = row["provider_audit_artifact"]
    Path(binding["path"]).write_bytes(payload)
    binding["sha256"] = hashlib.sha256(payload).hexdigest()
    audit = build_recovery_audit([], None, row, tmp_path, "hy3-ioa")
    assert audit["closed"] is True
    assert audit["eligible"] is False
    assert "provider_model_mismatch" in audit["reasons"]


@pytest.mark.parametrize(
    "malformation", ["record_list", "identity_list", "observed_string", "byte_count"]
)
def test_malformed_bound_audit_is_reported_without_raising(tmp_path, malformation):
    from runner.recovery_audit import build_recovery_audit

    row = _row(tmp_path / "cell", "first", ["success"], 0)
    records = _records(["success"])
    if malformation == "record_list":
        records[0] = []
    elif malformation == "identity_list":
        records[-1]["response"]["model_identity_closure"] = []
    elif malformation == "observed_string":
        records[-1]["response"]["model_identity_closure"]["observed_models"] = "hy3-ioa"
    records[-1]["sha256"] = _hash(records[-1]["response"])
    payload = b"".join(_encoded(record) + b"\n" for record in records)
    binding = row["provider_audit_artifact"]
    Path(binding["path"]).write_bytes(payload)
    binding["sha256"] = hashlib.sha256(payload).hexdigest()
    if malformation == "byte_count":
        binding["byte_count"] = len(payload) + 1
    audit = build_recovery_audit([], None, row, tmp_path, "hy3-ioa")
    assert audit["closed"] is False
    assert audit["totals"]["request_attempts"] is None
