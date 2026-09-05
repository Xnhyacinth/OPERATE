import json

import pytest

from scripts.audit_agent_failure_recipes import build_report as recipe_report
from scripts.audit_tool_effects import build_report as tool_report
from scripts.audit_tool_effects import classify_state_changing_score_evidence


@pytest.mark.parametrize("proven", [True, False])
def test_both_audits_require_same_call_proven_native_effect(tmp_path, proven):
    entry = _entry()
    entry["info"]["extra"]["tool_trace_edges"][0]["effect_proven"] = proven
    entry["evidence_ids"] = ["receipt", "effect"]
    entry["tool_results"] = [
        {
            "name": "dispatch_ready_operations",
            "call_id": "c",
            "ok": True,
            "state_changing": True,
            "idempotency_key": "c",
            "evidence_id": "receipt",
            "produces_evidence_ids": ["receipt", "effect"],
        }
    ]
    path = tmp_path / "episode.trajectory.jsonl"
    path.write_text(json.dumps(entry) + "\n")
    row = {
        "status": "ok",
        "domain": "logistics",
        "backend_kind": "jsplib_job_shop",
        "trajectory_summary": {"trajectory_path": str(path)},
        "score": {
            "dimensions": [
                {
                    "name": "tool_use_efficiency",
                    "applicable": True,
                    "evidence_ids": ["effect"],
                }
            ]
        },
    }
    for report, key in [
        (tool_report, "tool_effect_audit"),
        (recipe_report, "agent_failure_recipes"),
    ]:
        totals = report([row], batch_root=tmp_path)[key]["totals"]
        assert totals["state_changing_evidence_not_in_score"] == int(not proven)


def _entry():
    return {
        "info": {
            "extra": {
                "tool_trace_edges": [
                    {
                        "call_id": "c",
                        "state_changing": True,
                        "effect_proven": True,
                        "backend_effect_evidence_ids": ["effect"],
                    }
                ],
                "world_evolution_records": [
                    {
                        "call_id": "c",
                        "origin": "agent_caused",
                        "evidence_ids": ["receipt", "effect"],
                    }
                ],
            }
        }
    }


def test_native_effect_scored_without_requiring_duplicate_receipt_citation():
    dimensions = [
        {"name": "tool_use_efficiency", "applicable": True, "evidence_ids": ["effect"]}
    ]
    result = {
        "call_id": "c",
        "ok": True,
        "state_changing": True,
        "evidence_id": "receipt",
        "produces_evidence_ids": ["receipt", "effect"],
    }
    assert (
        classify_state_changing_score_evidence(
            "dispatch_ready_operations", dimensions, result, entry=_entry()
        )
        == "score_evidence_present"
    )


def test_unrelated_produced_id_or_other_call_does_not_close_score_gap():
    dimensions = [
        {
            "name": "tool_use_efficiency",
            "applicable": True,
            "evidence_ids": ["unrelated"],
        }
    ]
    result = {
        "call_id": "c",
        "ok": True,
        "state_changing": True,
        "evidence_id": "receipt",
        "produces_evidence_ids": ["unrelated"],
    }
    assert (
        classify_state_changing_score_evidence(
            "dispatch_ready_operations", dimensions, result, entry=_entry()
        )
        == "expected_score_evidence_missing"
    )
    dimensions[0]["evidence_ids"] = ["effect"]
    result["produces_evidence_ids"] = ["effect"]
    result["call_id"] = "other-call"
    assert (
        classify_state_changing_score_evidence(
            "dispatch_ready_operations", dimensions, result, entry=_entry()
        )
        == "expected_score_evidence_missing"
    )
