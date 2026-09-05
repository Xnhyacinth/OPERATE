from copy import deepcopy

import pytest

from scripts import run_protocol21_diagnostic_smoke as smoke


def result():
    cf = {"per_action_capped": False}
    for prefix in ("per_action", "per_action_group"):
        cf.update({f"{prefix}_{key}": value for key, value in {
            "status": "complete", "expected": 0, "attempted": 0,
            "completed": 0, "failures": [],
        }.items()})
    return {
        "status": "ok", "agent_name": "wait_only", "scenario_id": "case",
        "scenario_signature": "signature", "score": {}, "task_completion": {},
        "counterfactual": cf, "diagnostic_input_binding": {"verified": True},
        "implementation_tree_sha256": "tree",
        "diagnostic_runtime_integrity": {
            "implementation_tree_stable": True,
            "implementation_tree_sha256_start": "tree",
            "implementation_tree_sha256_end": "tree",
            "process_check_available": True, "orphan_pids": [],
        },
        "trajectory_summary": {
            "tool_semantic_coverage": {"covered": True},
            "event_response_records": [],
            "operational_agency_profile": {
                "schema_version": "operational_agency_profile_v1",
                "runtime_evidence_binding_verified": True,
                "masked_replay_binding_verified": True,
                "event_response_record_count": 0,
            },
            "terminal_integrity": {
                "release_ready": False, "response_window_extensions": 0,
                "terminal_feedback_reasons": [],
                "unanswered_interrupt_reasons": ["safety_warning"],
                "unresolved_pending_actions": {},
            },
        },
    }


def test_installation_keeps_strict_failure_and_original_evidence():
    row = result()
    original = deepcopy(row)
    assert smoke.validate_result(row)["errors"] == ["terminal_integrity_not_ready"]
    check = smoke.validate_result(row, check_profile="runtime_installation")
    assert check["passed"]
    assert check["strict_errors"] == ["terminal_integrity_not_ready"]
    assert check["warnings"] == ["wait_only_unanswered_interrupts"]
    assert row == original


@pytest.mark.parametrize("terminal", [
    None, {}, {"release_ready": "true"}, {"release_ready": 1},
    {"release_ready": False},
    {**result()["trajectory_summary"]["terminal_integrity"], "unresolved_pending_actions": {"x": 1}},
    {**result()["trajectory_summary"]["terminal_integrity"], "terminal_feedback_reasons": ["receipt"]},
    {**result()["trajectory_summary"]["terminal_integrity"], "unanswered_interrupt_reasons": []},
])
def test_installation_does_not_excuse_missing_or_other_terminal_failures(terminal):
    row = result()
    row["trajectory_summary"]["terminal_integrity"] = terminal
    assert not smoke.validate_result(row, check_profile="runtime_installation")["passed"]


@pytest.mark.parametrize("field,value", [
    ("orphan_pids", [123]), ("process_check_available", False),
    ("implementation_tree_stable", False),
])
def test_installation_preserves_runtime_checks(field, value):
    row = result()
    row["diagnostic_runtime_integrity"][field] = value
    assert not smoke.validate_result(row, check_profile="runtime_installation")["passed"]


@pytest.mark.parametrize("fault", ["tools", "counterfactual", "binding", "fatal"])
def test_installation_preserves_evidence_and_policy_checks(fault):
    row = result()
    if fault == "tools":
        row["trajectory_summary"]["tool_semantic_coverage"]["unknown_tool_names"] = ["unknown"]
    elif fault == "counterfactual":
        row["counterfactual"]["per_action_status"] = "failed"
    elif fault == "binding":
        row["diagnostic_input_binding"]["verified"] = False
    else:
        row["ground_truth_summary"] = {"chose_fatal_option": True}
    assert not smoke.validate_result(row, check_profile="runtime_installation")["passed"]


def test_installation_rejects_non_wait_agents():
    row = result()
    row["agent_name"] = "oracle_offline"
    with pytest.raises(ValueError, match="wait_only"):
        smoke.validate_result(row, check_profile="runtime_installation")
    with pytest.raises(SystemExit):
        smoke.main(["--check-profile", "runtime_installation", "--agents", "oracle_offline"])


def test_installation_report_is_not_strict_success():
    report = smoke.build_report(
        slice_payload={"scenarios": [{}]}, results=[result()],
        requested_agents=["wait_only"], repeats=1,
        check_profile="runtime_installation",
    )
    assert report["status"] == "passed"
    assert report["n_strict_failures"] == 1
    assert report["n_check_failures"] == 0
    assert report["model_success_claimed"] is False
    assert report["release_admission"] is False
