import hashlib
import json

from evaluation.offline_behavior import extract_behavior


def artifact(tmp_path, name, rows):
    data = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path = tmp_path / name
    path.write_bytes(data)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "event_count": len(rows),
    }


def extract(tmp_path, summary):
    return extract_behavior(
        {"trajectory_summary": summary}, root=tmp_path, journal_dir=tmp_path
    )


def test_missing_is_not_zero(tmp_path):
    result = extract(tmp_path, {"n_tool_calls": 0, "llm": {"llm_calls_failed": 0}})
    assert result["summary"]["n_tool_calls"] == 0
    assert result["summary"]["llm_calls_failed"] == 0
    assert result["summary"]["session_compactions"] is None
    assert result["summary"]["total_tokens"] is None
    assert result["summary"]["unique_calls"] is None


def test_hash_mismatch_does_not_use_trajectory_path(tmp_path):
    desc = artifact(tmp_path, "trajectory.jsonl", [{"tick": 1}])
    desc["sha256"] = "bad"
    result = extract(
        tmp_path, {"trajectory_artifact": desc, "trajectory_path": desc["path"]}
    )
    assert not result["summary"]["trajectory_verified"]
    assert "trajectory:sha256_mismatch" in result["issues"]


def test_duplicate_receipts_do_not_imply_engine_effect(tmp_path):
    call = {"call_id": "a", "name": "control"}
    receipt = {"call_id": "a", "name": "control", "ok": True, "state_changing": True}
    step = {
        "tick": 1,
        "action": {"actions": [call]},
        "tool_results": [receipt, receipt],
    }
    result = extract(
        tmp_path, {"trajectory_artifact": artifact(tmp_path, "t.jsonl", [step, step])}
    )
    assert result["summary"]["unique_calls"] == 1
    assert result["summary"]["successful_receipts"] == 1
    assert result["summary"]["engine_effect_calls"] is None
    assert result["summary"]["proven_engine_effect_calls"] == 0
    assert result["summary"]["unresolved_call_count"] == 1
    assert result["lifecycle_rows"][0]["receipt_count"] == 1


def test_native_timeline_excludes_public_observations(tmp_path):
    rows = [
        {
            "kind": "backend_tick",
            "source": "engine",
            "tick": tick,
            "evidence_id": f"e{tick}",
            "payload": {"tick": tick, "loss": value, "done": False},
        }
        for tick, value in [(2, 0), (1, 7)]
    ]
    rows.append(
        {"kind": "backend_tick", "source": "agent", "tick": 3, "payload": {"loss": 999}}
    )
    result = extract(
        tmp_path, {"evidence_ledger_artifact": artifact(tmp_path, "e.jsonl", rows)}
    )
    assert [(r["tick"], r["value"]) for r in result["time_rows"]] == [(1, 7), (2, 0)]
    assert all(r["unit"] is None for r in result["time_rows"])


def test_provider_usage_deduplicated_and_missing_not_zero(tmp_path):
    row = {
        "record_kind": "provider_response",
        "sequence": 1,
        "response": {
            "provider_metadata": {"usage": {"prompt_tokens": 5, "completion_tokens": 0}}
        },
    }
    desc = artifact(tmp_path, "p.jsonl", [row, row])
    result = extract(tmp_path, {"provider_audit_artifact": desc})
    assert result["summary"]["total_tokens"] == 5
    assert result["summary"]["output_tokens"] == 0
    desc = artifact(
        tmp_path, "q.jsonl", [row, {"record_kind": "provider_response", "sequence": 2}]
    )
    assert (
        extract(tmp_path, {"provider_audit_artifact": desc})["summary"]["total_tokens"]
        is None
    )


def test_canceled_call_cannot_receive_engine_credit(tmp_path):
    effect = {
        "origin": "agent_caused",
        "call_id": "a",
        "event_id": "x",
        "effect_tick": 1,
        "changed_state_fields": ["power"],
        "before_state_digest": "old",
        "after_state_digest": "new",
        "evidence_ids": ["e"],
        "action_to_outcome_edge": {
            "kind": "action_to_outcome",
            "source": "call:a",
            "target": "outcome:x",
        },
    }
    step = {
        "tick": 1,
        "action": {"actions": [{"call_id": "a", "name": "control"}]},
        "tool_results": [{"call_id": "a", "ok": True, "evidence_id": "e"}],
        "world_evolution_records": [effect],
    }
    desc = artifact(tmp_path, "good.jsonl", [step])
    assert (
        extract(tmp_path, {"trajectory_artifact": desc})["summary"][
            "engine_effect_calls"
        ]
        == 1
    )
    step["tool_results"].append({"call_id": "a", "error_code": "expired", "ok": False})
    desc = artifact(tmp_path, "expired.jsonl", [step])
    assert (
        extract(tmp_path, {"trajectory_artifact": desc})["lifecycle_rows"][0][
            "engine_effect"
        ]
        is False
    )


def test_empty_verified_trajectory_has_zero_calls(tmp_path):
    result = extract(
        tmp_path, {"trajectory_artifact": artifact(tmp_path, "empty.jsonl", [])}
    )
    assert result["summary"]["proven_engine_effect_calls"] == 0
    assert result["summary"]["engine_effect_calls"] == 0
    assert result["summary"]["effect_measurement_complete"]


def test_unidentified_calls_are_reported_as_unmeasured(tmp_path):
    step = {
        "tick": 1,
        "action": {"actions": [{"name": "control"}]},
        "tool_results": [{"name": "control", "ok": True}],
    }
    result = extract(
        tmp_path, {"trajectory_artifact": artifact(tmp_path, "anonymous.jsonl", [step])}
    )
    assert result["summary"]["request_records_without_call_id"] == 1
    assert result["summary"]["receipt_records_without_call_id"] == 1
    assert not result["summary"]["effect_measurement_complete"]
    assert result["summary"]["engine_effect_calls"] is None
