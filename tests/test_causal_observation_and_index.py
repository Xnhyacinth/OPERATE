"""Regressions for observed responses versus backend-only effects."""

from types import SimpleNamespace

import pytest

from core import EvidenceLogger
from data.trajectory_logger import TrajectoryLogger
from evaluation.operational_agency import evaluate_operational_agency
from runner.episode import _build_event_response_records, _operational_agency_artifacts


def _steps(consumed):
    return [
        {
            "tick": 21,
            "info": {
                "extra": {
                    "world_evolution_records": [
                        {
                            "event_id": "ready",
                            "origin": "source_schedule",
                            "applied_tick": 21,
                            "evidence_ids": ["trigger"],
                        },
                        {
                            "event_id": "effect",
                            "origin": "agent_caused",
                            "call_id": "dispatch",
                            "causal_parent_event_id": "ready",
                            "outcome_tick": 23,
                            "evidence_ids": ["effect"],
                        },
                    ]
                }
            },
            "tool_trace_edges": [
                {
                    "call_id": "dispatch",
                    "state_changing": True,
                    "effect_proven": True,
                    "request_tick": 21,
                    "effect_tick": 23,
                    "consumes_evidence_ids": consumed,
                    "produces_evidence_ids": ["effect"],
                }
            ],
        }
    ]


def test_backend_effect_without_observation_is_preserved_but_not_a_response():
    record = _build_event_response_records(_steps(["unrelated"]))[0]
    assert record["response_status"] == "backend_effect_only"
    assert record["first_observed_tick"] is None
    assert record["backend_effect_evidence_ids"] == ["effect"]
    assert record["first_effect_tick"] == 23
    assert record["observation_evidence_ids"] == []


@pytest.mark.parametrize(
    "observed, consumed", [(None, ["trigger"]), (24, ["trigger"]), (21, ["unrelated"])]
)
def test_external_causal_claim_requires_observation_and_consumption(observed, consumed):
    record = _build_event_response_records(_steps(["trigger"]))[0]
    record.update(
        response_status="causal",
        first_observed_tick=observed,
        action_consumes_evidence_ids=consumed,
        masked_action_group_delta=5,
        native_burden_before=10,
        native_burden_basis="episode_replay_no_action_cost",
    )
    profile = evaluate_operational_agency(
        [record],
        valid_evidence_ids={"trigger", "effect", "unrelated"},
        masked_replay_by_call_id={"dispatch": 5},
    )
    assert profile["causal_record_count"] == 0


def test_logger_indexes_tool_result_evidence_without_changing_observation():
    logger = TrajectoryLogger()
    observation = {"tick": 22}
    logger.log_step(
        22,
        observation,
        None,
        0,
        evidence_ids=["world"],
        tool_results=[
            {"evidence_id": "receipt", "produces_evidence_ids": ["receipt", "effect"]},
            SimpleNamespace(evidence_id="query", produces_evidence_ids=["query"]),
        ],
    )
    assert logger.entries[0].evidence_ids == ["world", "receipt", "effect", "query"]
    assert observation == {"tick": 22}


def test_consumed_linked_reveal_can_close_response_chain():
    evidence = EvidenceLogger("linked")
    trigger = evidence.log("realized_event", 1, {}, source="engine")
    reveal = evidence.log("investigation", 2, {}, source="tool")
    query_receipt = evidence.log(
        "tool_call",
        2,
        {
            "call_id": "query",
            "ok": True,
            "linked_result_evidence_id": reveal,
        },
        source="tool",
    )
    evidence.log(
        "tool_call",
        3,
        {
            "call_id": "dispatch",
            "ok": True,
            "state_changing": True,
            "consumes_evidence_ids": [query_receipt],
        },
        source="tool",
    )
    effect = evidence.log("realized_event", 4, {}, source="engine")
    steps = _steps([query_receipt])
    parent, outcome = steps[0]["info"]["extra"]["world_evolution_records"]
    parent.update(applied_tick=1, evidence_ids=[trigger], reveal_evidence_ids=[reveal])
    outcome.update(outcome_tick=4, evidence_ids=[effect])
    steps[0]["tool_trace_edges"][0].update(
        request_tick=3, effect_tick=4, produces_evidence_ids=[effect]
    )
    records, _, profile = _operational_agency_artifacts(
        env=SimpleNamespace(evidence=evidence),
        ground_truth={},
        analysis_steps=steps,
        counterfactual=SimpleNamespace(
            per_action=[{"call_id": "dispatch", "marginal_prevented_loss": 5}],
            per_action_groups=[],
            applicable=True,
            counterfactual_cost=10,
        ),
    )
    assert records[0]["first_observed_tick"] == 2
    assert records[0]["response_status"] == "causal"
    assert records[0]["action_consumes_evidence_ids"] == [query_receipt]
    assert reveal in records[0]["resolved_action_consumes_evidence_ids"]
    assert profile["causal_record_count"] == 1


@pytest.mark.parametrize("in_ledger", [True, False])
def test_audit_distinguishes_missing_index_from_missing_authoritative_evidence(tmp_path, in_ledger):
    import json
    from scripts.audit_tool_effects import build_report

    path = tmp_path / "episode.trajectory.jsonl"
    path.write_text(json.dumps({"tick": 22, "evidence_ids": [], "tool_results": [{
        "name": "dispatch", "call_id": "call", "ok": True, "state_changing": True,
        "idempotency_key": "call", "evidence_id": "receipt",
    }]}) + "\n")
    path.with_name("episode.evidence.jsonl").write_text(
        json.dumps({"evidence_id": "receipt" if in_ledger else "other"}) + "\n"
    )
    report = build_report([{
        "status": "ok", "trajectory_summary": {"trajectory_path": str(path)},
    }], batch_root=tmp_path)["tool_effect_audit"]
    totals = report["totals"]
    assert totals["state_changing_evidence_not_in_step"] == 1
    assert totals["state_changing_index_omission_in_ledger"] == int(in_ledger)
    assert totals["state_changing_evidence_not_in_ledger"] == int(not in_ledger)
    assert report["status"] == "blocking"
