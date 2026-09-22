"""Capability reporting must not turn missing behavioral evidence into scores."""
from copy import deepcopy

from evaluation.capability_report import build_capability_report
from evaluation.operational_agency import evaluate_operational_agency


def causal_episode():
    record = {
        "event_id": "event", "call_id": "call", "event_tick": 2,
        "first_observed_tick": 3, "first_control_call_tick": 4,
        "first_effect_tick": 8, "mandatory_response_tick": 6,
        "response_status": "causal", "masked_action_group_delta": 20.0,
        "native_burden_before": 80.0,
        "native_burden_basis": "episode_replay_no_action_cost",
        "trigger_evidence_ids": ["obs"],
        "action_consumes_evidence_ids": ["obs"],
        "action_evidence_ids": ["action"],
        "backend_effect_evidence_ids": ["effect"],
        "plan_evidence_ids": ["plan"], "dependency_depth": 2,
    }
    ids = {"obs", "action", "effect", "plan"}
    return {
        "interaction_mode": "logical_persistent", "n_ticks_ran": 400,
        "trajectory_summary": {
            "n_ticks": 400,
            "event_response_records": [record],
            "operational_agency_valid_evidence_ids": sorted(ids),
            "operational_agency_profile": evaluate_operational_agency(
                [record], valid_evidence_ids=ids,
                masked_replay_by_call_id={"call": 20.0}),
        },
        "counterfactual": {"per_action": [
            {"call_id": "call", "marginal_prevented_loss": 20.0}]},
    }


def test_verified_chain_survives_missing_native_references():
    episode = causal_episode()
    before = deepcopy(episode)
    report = build_capability_report(episode, scenario_spec={"horizon_ticks": 400})
    assert report["operational_agency"]["verified"] is True
    assert report["operational_agency"]["dimensions"]["initiative"]["score"] == 50
    assert report["operational_agency"]["opportunity_count"] is None
    chain = report["temporal_evidence"]["chains"][0]
    assert chain["action_to_effect_ticks"] == 4
    assert chain["plan_evidence_ids"] == ["plan"]
    assert report["horizon"]["configured_stratum"] == "above_192_ticks"
    assert report["horizon"]["memory_retention_score"] is None
    assert report["realtime_supervision"]["applicable"] is False
    assert episode == before


def test_unbound_replay_withholds_behavior_credit():
    episode = causal_episode()
    episode["counterfactual"]["per_action"][0]["marginal_prevented_loss"] = 99
    report = build_capability_report(episode)
    assert report["operational_agency"]["verified"] is False
    assert report["operational_agency"]["dimensions"] is None
    assert report["temporal_evidence"]["chains"] is None


def test_missing_horizon_and_profile_are_not_zeros():
    report = build_capability_report({"interaction_mode": "logical_persistent"})
    assert report["horizon"]["configured_ticks"] is None
    assert report["horizon"]["recorded_ticks"] is None
    assert report["operational_agency"]["dimensions"] is None
    assert report["native_outcome"]["actual_cost"] is None


def test_tick_disagreement_is_visible_not_cherry_picked():
    episode = causal_episode()
    episode["trajectory_summary"]["n_ticks"] = 399
    report = build_capability_report(episode)
    assert report["horizon"]["recorded_ticks"] is None
    assert report["horizon"]["reason"] == "recorded_tick_counts_disagree"


def test_realtime_is_not_inferred_from_logical_or_summary_labels():
    episode = causal_episode()
    episode["interaction_mode"] = "realtime_persistent"
    report = build_capability_report(episode)
    assert report["realtime_supervision"]["score"] is None
    assert report["realtime_supervision"]["reason"] == "use_independent_realtime_ledger_scorecard"
    assert report["formal_run_certified"] is False


def test_valid_profile_does_not_promote_noncausal_record_to_temporal_chain():
    episode = causal_episode()
    summary = episode["trajectory_summary"]
    fake = deepcopy(summary["event_response_records"][0])
    fake.update(call_id="other", response_status="backend_effect_only")
    summary["event_response_records"].append(fake)
    summary["operational_agency_profile"] = evaluate_operational_agency(
        summary["event_response_records"],
        valid_evidence_ids=set(summary["operational_agency_valid_evidence_ids"]),
        masked_replay_by_call_id={"call": 20.0, "other": 20.0},
    )
    episode["counterfactual"]["per_action"].append(
        {"call_id": "other", "marginal_prevented_loss": 20.0})
    report = build_capability_report(episode)
    assert report["operational_agency"]["verified"] is True
    assert [r["call_id"] for r in report["temporal_evidence"]["chains"]] == ["call"]


def long_fixture():
    suite = [
        {"scenario_signature": key, "seed": 42, "horizon_ticks": ticks,
         "domain": "logistics", "backend_kind": "dynasched_flexible_job_shop",
         "source_denominator_key": key}
        for key, ticks in [("long1", 298), ("long2", 400), ("short", 72)]
    ]
    rows = [
        {"model": "m", "scenario_signature": key, "seed": 42,
         "artifact_binding": {"verified": True, "trace_ticks": 235},
         "capability_report": {"native_outcome": {
             "applicable": True, "actual_cost": 580, "feasible": True,
             "evidence_ids": ["cost"]}},
         "native_quality": {"score": score, "constraint_failed": False,
                            "task_success": True}}
        for key, score in [("long1", 10), ("long2", 30), ("short", 99)]
    ]
    return suite, rows


def test_long_task_report_uses_fixed_two_case_denominator():
    from evaluation.capability_report import build_long_task_report
    suite, rows = long_fixture()
    report = build_long_task_report(rows, suite, "m")
    assert report["expected_cases"] == 2
    assert report["index"] == 20
    assert report["complete"] is True
    assert len(report["cases"]) == 2
    assert report["n_authenticated_native_outcomes"] == 2


def test_long_task_report_does_not_shrink_when_one_case_missing():
    from evaluation.capability_report import build_long_task_report
    suite, rows = long_fixture()
    report = build_long_task_report(rows[:1], suite, "m")
    assert report["expected_cases"] == 2
    assert report["index"] is None
    assert report["complete"] is False
    assert report["n_reference_normalized_cases"] == 1
    assert report["cases"][1]["reason"] == "missing_or_duplicate_attempt"


def test_long_task_report_keeps_bound_native_cost_without_reference():
    from evaluation.capability_report import build_long_task_report
    suite, rows = long_fixture()
    rows[1]["native_quality"] = {"score": None, "reason": "reference_missing"}
    report = build_long_task_report(rows, suite, "m")
    assert report["index"] is None
    assert report["n_authenticated_native_outcomes"] == 2
    assert report["cases"][1]["native_outcome"]["actual_cost"] == 580
    rows[1]["artifact_binding"]["verified"] = False
    report = build_long_task_report(rows, suite, "m")
    assert report["n_authenticated_native_outcomes"] == 1
    assert report["cases"][1]["native_outcome"] is None


def test_no_declared_long_task_scope_is_explicitly_unavailable():
    from evaluation.capability_report import build_long_task_report
    report = build_long_task_report([], [], "m")
    assert report["expected_cases"] == 0
    assert report["index"] is None
    assert report["reason"] == "no_declared_long_task_cases"
