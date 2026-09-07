from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from core.agentic_core_contract import (
    _persistent_policy_evidence,
    artifact_binding,
    build_agentic_contract_report,
)
from core.difficulty_contract import DIFFICULTY_CONTRACT_VERSION
from evaluation.scorer import SCORING_VERSION
from runner.episode import EVALUATION_IMPLEMENTATION_FINGERPRINT

PROTOCOL = "2.1"
FINGERPRINT = EVALUATION_IMPLEMENTATION_FINGERPRINT
SCORER = SCORING_VERSION


def _semantics() -> dict:
    return {
        "protocol_version": PROTOCOL,
        "implementation_fingerprint": FINGERPRINT,
        "scoring_version": SCORER,
    }


def _artifacts() -> dict[str, dict]:
    scenario_id = "traffic/incident/high/example"
    signature = "sig-example"
    identity = {
        "scenario_id": scenario_id,
        "scenario_signature": signature,
    }
    source = {
        "status": "complete",
        "evaluation_semantics": _semantics(),
        "n_expected": 1,
        "n_completed": 1,
        "scenarios": [
            {
                **identity,
                "path": "scenario.yaml",
                "domain": "traffic",
                "backend_kind": "mock_sumo",
                "difficulty_level": "high",
                "horizon_ticks": 12,
                "seed": 42,
                "policy_contract": {"strict_prompt": True},
                "capability": {
                    "control_tools": ["set_signal_phase"],
                    "clock_semantics": "simulator_owned",
                },
            }
        ],
    }
    behavioral = {
        "status": "complete",
        "evaluation_semantics": _semantics(),
        "n_expected": 1,
        "n_completed": 1,
        "results": [
            {
                **identity,
                "status": "passed",
                "checks": {"native_backend_executable": True},
                "agentic_evidence": {
                    "simulator_ticks": 12,
                    "model_decision_ticks": 3,
                    "provider_calls": 3,
                    "autonomous_hold_ticks": 4,
                    "pending_action_hold_ticks": 0,
                    "scheduled_review_ticks": [6],
                    "periodic_scan_ticks": [],
                    "actual_supervisory_review_observed": True,
                    "standing_plan_committed": True,
                    "standing_plan_commit_ticks": [0],
                    "wake_reason_counts": {"visible_interrupt": 1},
                    "visible_interrupt_count": 1,
                    "declared_predesigned_event_ids": ["incident-1"],
                    "declared_predesigned_event_ticks": [4],
                    "realized_predesigned_event_ids": ["incident-1"],
                    "realized_predesigned_event_ticks": [4],
                    "exogenous_state_change_ticks": [4],
                    "agent_caused_state_change_ticks": [5],
                    "source_consumption_ticks": [1, 2, 3, 4],
                    "state_change_ticks": [1, 4, 5],
                    "state_changing_tool_calls": 2,
                    "available_native_control_tool_names": ["set_signal_phase"],
                    "successful_native_control_tool_names": ["set_signal_phase"],
                    "decision_graph_nodes": 4,
                    "decision_graph_edges": 3,
                    "decision_graph_acyclic": True,
                    "event_adaptive_cadence_declared": True,
                    "world_change_contract_declared": True,
                    "material_exogenous_event_records": [
                        {
                            "event_id": "incident-1",
                            "applied_tick": 4,
                            "material_exogenous": True,
                        }
                    ],
                    "post_change_decision_ticks": [5],
                    "event_to_decision_action_edges": [
                        {
                            "source_event_id": "incident-1",
                            "target_tick": 5,
                            "kind": "event_to_post_change_decision",
                        }
                    ],
                    "adaptive_replanning_observed": True,
                    "valid_plan_delegation_observed": False,
                    "agent_action_backend_effect_observed": True,
                },
            }
        ],
    }
    tasks = {
        "status": "complete",
        "evaluation_semantics": _semantics(),
        "n_expected": 1,
        "n_completed": 1,
        "results": [
            {
                **identity,
                "status": "passed",
                "completed": True,
                "terminal_integrity": {"release_ready": True},
                "material_headroom": {"status": "passed"},
            }
        ],
    }
    complexity = {
        "status": "complete",
        "evaluation_semantics": _semantics(),
        "n_expected": 3,
        "n_completed": 3,
        "results": [
            {
                **identity,
                "agent_name": name,
                "status": "complete",
                "task_loss": task_loss,
                "replay_minimization": {
                    "decision_graph_present": True,
                    "decision_graph_acyclic": True,
                    "bounded_or_exact_depth_available": True,
                },
            }
            for name, task_loss in (
                ("oracle_offline", 1.0),
                ("greedy_heuristic", 3.0),
                ("wait_only", 10.0),
            )
        ],
    }
    observed_depth = {
        "status": "complete",
        "evaluation_semantics": _semantics(),
        "n_expected": 1,
        "n_completed": 1,
        "samples": [
            {
                **identity,
                "disposition": "bounded_replay_required",
            }
        ],
    }
    strategy_depth = {
        "status": "complete",
        "evaluation_semantics": _semantics(),
        "n_expected": 1,
        "n_completed": 1,
        "samples": [
            {
                **identity,
                "core_action": "keep",
                "exact_task_dependency_depth": 2,
                "difficulty_calibration": {
                    "version": DIFFICULTY_CONTRACT_VERSION,
                    "status": "passed",
                    "declared_difficulty_level": "high",
                    "calibrated_difficulty_level": "high",
                    "declared_level_matches_evidence": True,
                },
            }
        ],
    }
    source_grounded = {
        "status": "complete",
        "evaluation_semantics": _semantics(),
        "n_expected": 1,
        "n_completed": 1,
        "results": [
            {
                **identity,
                "status": "admitted",
                "passed_gates": [
                    "domain_boundary",
                    "source_lock",
                    "source_consumption",
                    "capability_contract",
                    "deterministic_replay",
                    "task_headroom",
                    "decision_graph",
                    "difficulty_proof",
                    "counterfactual_replay",
                    "source_independence",
                ],
                "failed_gates": [],
            }
        ],
    }
    source_consumption = {
        "status": "complete",
        "evaluation_semantics": _semantics(),
        "n_expected": 1,
        "n_completed": 1,
        "results": [
            {
                **identity,
                "backend_kind": "mock_sumo",
                "status": "passed",
            }
        ],
    }
    return {
        "source_suite": source,
        "behavioral": behavioral,
        "task_contracts": tasks,
        "complexity": complexity,
        "observed_depth": observed_depth,
        "strategy_depth": strategy_depth,
        "source_grounded": source_grounded,
        "source_consumption": source_consumption,
    }


def _write_artifacts(tmp_path: Path, artifacts: dict[str, dict]) -> dict:
    bindings = {}
    for name, payload in artifacts.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        bindings[name] = artifact_binding(path)
    return bindings


def test_complete_current_evidence_passes_agentic_contract(tmp_path: Path) -> None:
    artifacts = _artifacts()
    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    assert report["status"] == "complete"
    assert report["n_expected"] == 1
    assert report["n_passed"] == 1
    assert report["n_held"] == 0
    row = report["results"][0]
    assert row["status"] == "passed"
    assert all(row["checks"].values())
    assert set(row["agentic_contract"]) == {
        "policy_contract",
        "tool_contract",
        "task_contract",
        "world_evolution_contract",
        "decision_process_contract",
        "source_contract",
        "source_consumption_contract",
    }


def test_simulator_owned_clock_without_actual_supervision_is_held(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    artifacts["source_suite"]["scenarios"][0]["difficulty_level"] = "medium"
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["autonomous_hold_ticks"] = 0
    evidence["simulator_advance_without_model_ticks"] = []
    evidence["simulator_owned_clock_observed"] = True
    evidence.pop("scheduled_review_ticks")
    evidence.pop("periodic_scan_ticks")
    evidence["actual_supervisory_review_observed"] = False
    evidence["standing_plan_committed"] = False
    evidence["standing_plan_commit_ticks"] = []

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["checks"]["parallel_simulator_agent_clock_observed"] is True
    assert row["checks"]["actual_supervisory_review_observed"] is False
    assert row["status"] == "held"
    assert "actual_supervisory_review_unobserved" in row["blockers"]


def test_stale_semantics_and_missing_runtime_evidence_are_held(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    artifacts["behavioral"]["evaluation_semantics"]["protocol_version"] = "2.0"
    artifacts["behavioral"]["results"][0].pop("agentic_evidence")

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert "artifact_semantics_stale" in row["blockers"]
    assert "evidence_missing_runtime_evolution" in row["blockers"]


def test_explicit_task_failure_is_retired(tmp_path: Path) -> None:
    artifacts = _artifacts()
    task = artifacts["task_contracts"]["results"][0]
    task["status"] = "failed"
    task["completed"] = False

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    assert report["results"][0]["status"] == "retired"
    assert "task_contract_failed" in report["results"][0]["blockers"]


def test_missing_replay_evidence_is_held_but_explicit_failure_is_retired(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    source_row = artifacts["source_grounded"]["results"][0]
    source_row["status"] = "held"
    source_row["passed_gates"].remove("deterministic_replay")
    source_row["failed_gates"] = ["deterministic_replay"]

    missing = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    assert missing["results"][0]["status"] == "held"
    assert "deterministic_replay_evidence_missing" in missing["results"][0]["blockers"]

    source_row["gate_evidence"] = {"deterministic_replay": {"status": "failed"}}
    failed = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    assert failed["results"][0]["status"] == "retired"
    assert "deterministic_replay_failed" in failed["results"][0]["blockers"]


def test_signature_mismatch_fails_closed(tmp_path: Path) -> None:
    artifacts = _artifacts()
    artifacts = deepcopy(artifacts)
    artifacts["strategy_depth"]["samples"][0]["scenario_signature"] = "wrong"

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert "artifact_identity_mismatch" in row["blockers"]


def test_non_reference_control_evidence_cannot_promote_reference(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    artifacts["behavioral"]["results"][0].pop("agentic_evidence")
    task = artifacts["task_contracts"]["results"][0]
    task["agent_name"] = "oracle_offline"
    for result in artifacts["complexity"]["results"]:
        result["agentic_evidence"] = {
            "simulator_ticks": 12,
            "realized_event_ticks": [4],
            "state_change_ticks": [4],
            "state_changing_tool_calls": (
                2 if result["agent_name"] == "greedy_heuristic" else 0
            ),
            "native_control_tool_names": ["set_signal_phase"],
            "periodic_cadence_observed": True,
            "decision_graph_nodes": 4,
            "decision_graph_acyclic": True,
        }

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert "evidence_missing_native_control_use" in row["blockers"]


def test_wake_reason_counts_cannot_replace_explicit_cadence_contract(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["event_adaptive_cadence_declared"] = False
    evidence["wake_reason_counts"] = {"visible_interrupt": 2}

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert row["checks"]["event_adaptive_cadence_declared"] is False
    assert "event_adaptive_cadence_undeclared" in row["blockers"]


def test_medium_standing_plan_is_valid_but_high_requires_active_replanning(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    source_row = artifacts["source_suite"]["scenarios"][0]
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["adaptive_replanning_observed"] = False
    evidence["valid_plan_delegation_observed"] = False
    source_row["difficulty_level"] = "medium"

    medium = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    medium_row = medium["results"][0]
    assert medium_row["status"] == "passed"
    assert (
        medium_row["checks"]["difficulty_appropriate_control_response_observed"] is True
    )
    assert (
        medium_row["agentic_contract"]["decision_process_contract"][
            "required_control_response"
        ]
        == "standing_plan_with_post_change_monitoring"
    )

    source_row["difficulty_level"] = "high"
    high = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    high_row = high["results"][0]
    assert high_row["status"] == "held"
    assert "adaptive_replanning_or_delegation_unproven" in high_row["blockers"]
    assert (
        high_row["agentic_contract"]["decision_process_contract"][
            "required_control_response"
        ]
        == "active_replanning_or_valid_delegation"
    )


def test_medium_reactive_action_without_standing_plan_is_held(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    artifacts["source_suite"]["scenarios"][0]["difficulty_level"] = "medium"
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["standing_plan_committed"] = False
    evidence["standing_plan_commit_ticks"] = []
    evidence["actual_supervisory_review_observed"] = False
    evidence["scheduled_review_ticks"] = []
    evidence["adaptive_replanning_observed"] = False
    evidence["valid_plan_delegation_observed"] = False

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert row["checks"]["difficulty_appropriate_control_response_observed"] is False
    assert "standing_plan_response_unproven" in row["blockers"]


def test_medium_standing_plan_requires_causal_runtime_order(
    tmp_path: Path,
) -> None:
    for commits, reviews, effect_ticks in (
        ([6], [1], [5]),
        ([5], [6], [1]),
    ):
        artifacts = _artifacts()
        artifacts["source_suite"]["scenarios"][0]["difficulty_level"] = "medium"
        evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
        evidence["adaptive_replanning_observed"] = False
        evidence["valid_plan_delegation_observed"] = False
        evidence["standing_plan_commit_ticks"] = commits
        evidence["scheduled_review_ticks"] = reviews
        evidence["agent_caused_state_change_ticks"] = effect_ticks

        report = build_agentic_contract_report(
            **artifacts,
            input_bindings=_write_artifacts(tmp_path, artifacts),
        )

        row = report["results"][0]
        assert row["status"] == "held"
        assert (
            row["checks"]["difficulty_appropriate_control_response_observed"] is False
        )
        assert (
            row["agentic_contract"]["decision_process_contract"][
                "standing_plan_timeline_observed"
            ]
            is False
        )
        assert "standing_plan_response_unproven" in row["blockers"]


def test_high_pre_event_delegation_cannot_satisfy_active_replanning(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    artifacts["source_suite"]["scenarios"][0]["difficulty_level"] = "high"
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["adaptive_replanning_observed"] = False
    evidence["valid_plan_delegation_observed"] = True
    evidence["delegated_plan_opportunity_ticks"] = [1]

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert row["checks"]["difficulty_appropriate_control_response_observed"] is False
    assert "adaptive_replanning_or_delegation_unproven" in row["blockers"]


def test_basic_or_medium_adaptive_action_cannot_replace_required_standing_plan(
    tmp_path: Path,
) -> None:
    for difficulty in ("basic", "medium"):
        artifacts = _artifacts()
        artifacts["source_suite"]["scenarios"][0]["difficulty_level"] = difficulty
        evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
        evidence["standing_plan_committed"] = False
        evidence["standing_plan_commit_ticks"] = []
        evidence["standing_control_commit_ticks"] = []
        evidence["scheduled_review_ticks"] = []
        evidence["periodic_scan_ticks"] = []
        evidence["continuous_supervisory_review_ticks"] = []
        evidence["adaptive_replanning_observed"] = True
        evidence["valid_plan_delegation_observed"] = False

        report = build_agentic_contract_report(
            **artifacts,
            input_bindings=_write_artifacts(
                tmp_path,
                artifacts,
            ),
        )

        row = report["results"][0]
        assert row["status"] == "held"
        assert (
            row["checks"]["difficulty_appropriate_control_response_observed"] is False
        )
        assert "standing_plan_response_unproven" in row["blockers"]


def test_supervision_boolean_cannot_spoof_missing_runtime_ticks(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["scheduled_review_ticks"] = []
    evidence["periodic_scan_ticks"] = []
    evidence["actual_supervisory_review_observed"] = True

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert row["checks"]["actual_supervisory_review_observed"] is False
    assert "actual_supervisory_review_unobserved" in row["blockers"]


def test_continuous_native_supervision_and_control_commit_are_runtime_evidence(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    artifacts["source_suite"]["scenarios"][0]["difficulty_level"] = "medium"
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["scheduled_review_ticks"] = []
    evidence["periodic_scan_ticks"] = []
    evidence["continuous_supervisory_review_ticks"] = [1, 2, 3, 5]
    evidence["actual_supervisory_review_observed"] = False
    evidence["standing_plan_committed"] = False
    evidence["standing_plan_commit_ticks"] = []
    evidence["standing_control_commit_ticks"] = [0]

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "passed"
    assert row["checks"]["actual_supervisory_review_observed"] is True
    assert (
        row["agentic_contract"]["decision_process_contract"]["standing_plan_observed"]
        is True
    )


def test_out_of_horizon_supervision_ticks_are_not_runtime_evidence(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["scheduled_review_ticks"] = [999]
    evidence["periodic_scan_ticks"] = []
    evidence["standing_plan_commit_ticks"] = [999]
    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert row["checks"]["actual_supervisory_review_observed"] is False
    assert "actual_supervisory_review_unobserved" in row["blockers"]


def test_non_finite_runtime_tick_evidence_is_ignored_fail_closed(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["realized_predesigned_event_ticks"] = [float("nan"), float("inf")]
    evidence["exogenous_state_change_ticks"] = [float("-inf")]
    evidence["source_consumption_ticks"] = [float("nan")]
    evidence["declared_predesigned_event_ticks"] = [float("inf")]
    evidence["simulator_ticks"] = float("nan")

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert "evidence_missing_runtime_evolution" in row["blockers"]
    assert "predesigned_event_not_reached" in row["blockers"]


def test_huge_integer_runtime_tick_evidence_is_ignored_fail_closed(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    # Keep the JSON fixture within Python's default integer-string limit while
    # still forcing float() to raise OverflowError in the runtime parser.
    huge_tick = 10**4000
    evidence["realized_predesigned_event_ticks"] = [huge_tick]
    evidence["exogenous_state_change_ticks"] = [huge_tick]
    evidence["source_consumption_ticks"] = [huge_tick]
    evidence["declared_predesigned_event_ticks"] = [huge_tick]
    evidence["simulator_ticks"] = huge_tick

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert "evidence_missing_runtime_evolution" in row["blockers"]
    assert "predesigned_event_not_reached" in row["blockers"]


def test_medium_persistent_policy_review_requires_runtime_attribution(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    artifacts["source_suite"]["scenarios"][0]["difficulty_level"] = "medium"
    artifacts["source_suite"]["scenarios"][0]["domain"] = "datacenter"
    artifacts["source_suite"]["scenarios"][0]["backend_kind"] = "alibaba_trace_sim"
    artifacts["source_suite"]["scenarios"][0]["capability"] = {
        "control_tools": ["set_queue_policy"],
        "persistent_policy_review": {
            "backend_kind": "alibaba_trace_sim",
            "review_tool_name": "review_persistent_policy",
            "policy_tool_names": ["set_queue_policy"],
        },
    }
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["adaptive_replanning_observed"] = False
    evidence["valid_plan_delegation_observed"] = False
    evidence["standing_plan_committed"] = False
    evidence["standing_plan_commit_ticks"] = []
    evidence["standing_control_commit_ticks"] = []
    evidence["available_native_control_tool_names"] = ["set_queue_policy"]
    evidence["successful_native_control_tool_names"] = ["set_queue_policy"]
    evidence["persistent_control_tool_names"] = ["set_queue_policy"]
    evidence["persistent_policy_review_records"] = [
        {
            "review_id": "review-incident-1",
            "review_tick": 6,
            "event_ids": ["incident-1"],
            "event_ticks": [4],
            "policy_generation": 1,
            "policy_digest": "policy-v1",
            "queue_order_digest": "queue-after-review",
            "policy_tool_name": "set_queue_policy",
            "policy_effect_evidence_id": "policy-effect-1",
            "review_tool_name": "review_persistent_policy",
            "decision": "keep",
            "outcome_effect_ticks": [7],
            "evidence_ids": ["review-evidence-1"],
        }
    ]
    evidence["persistent_policy_review_bindings"] = [
        {
            "review_id": "review-incident-1",
            "review_tool_name": "review_persistent_policy",
            "call_id": "review-call-1",
            "accepted": True,
            "evidence_ids": ["review-evidence-1"],
            "action_graph_evidence_ids": ["review-evidence-1"],
        }
    ]
    evidence["persistent_policy_effect_bindings"] = [
        {
            "policy_generation": 1,
            "policy_tool_name": "set_queue_policy",
            "accepted": True,
            "effect_tick": 5,
            "call_id": "policy-call-1",
            "evidence_ids": ["policy-effect-1"],
        }
    ]
    evidence["persistent_policy_attribution"] = {
        "status": "passed",
        "review_id": "review-incident-1",
        "review_tool_name": "review_persistent_policy",
        "policy_tool_name": "set_queue_policy",
        "policy_generation": 1,
        "policy_effect_evidence_id": "policy-effect-1",
        "event_ids": ["incident-1"],
        "deterministic_replay": True,
        "actual_state_digest": "state-after",
        "masked_state_digest": "state-without-policy",
        "actual_queue_order_digest": "queue-after-review",
        "masked_queue_order_digest": "queue-without-policy",
        "material_delta": 2.0,
        "materiality_threshold": 1.0,
        "effect_ticks": [7],
    }
    evidence["agent_caused_state_change_ticks"] = [5, 7]
    evidence["scheduled_review_ticks"] = []
    evidence["periodic_scan_ticks"] = []
    evidence["continuous_supervisory_review_ticks"] = []

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "passed"
    assert row["checks"]["difficulty_appropriate_control_response_observed"] is True
    decision_process = row["agentic_contract"]["decision_process_contract"]
    assert decision_process["persistent_policy_review_observed"] is True
    assert decision_process["persistent_policy_timeline_observed"] is True


def test_persistent_policy_review_on_traffic_fixture_is_held_fail_closed(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    artifacts["source_suite"]["scenarios"][0]["difficulty_level"] = "medium"
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["standing_plan_committed"] = False
    evidence["standing_plan_commit_ticks"] = []
    evidence["standing_control_commit_ticks"] = []
    evidence["persistent_control_tool_names"] = ["set_signal_phase"]
    evidence["persistent_policy_review_records"] = [
        {
            "review_id": "review-incident-1",
            "review_tick": 6,
            "event_ids": ["incident-1"],
            "event_ticks": [4],
            "policy_generation": 1,
            "policy_digest": "policy-v1",
            "queue_order_digest": "queue-after-review",
            "policy_tool_name": "set_queue_policy",
            "policy_effect_evidence_id": "policy-effect-1",
            "review_tool_name": "review_persistent_policy",
            "decision": "keep",
            "outcome_effect_ticks": [7],
            "evidence_ids": ["review-evidence-1"],
        }
    ]
    evidence["persistent_policy_review_bindings"] = [
        {
            "review_id": "review-incident-1",
            "review_tool_name": "review_persistent_policy",
            "call_id": "review-call-1",
            "accepted": True,
            "evidence_ids": ["review-evidence-1"],
            "action_graph_evidence_ids": ["review-evidence-1"],
        }
    ]
    evidence["persistent_policy_attribution"] = {
        "status": "passed",
        "review_id": "review-incident-1",
        "review_tool_name": "review_persistent_policy",
        "policy_tool_name": "set_queue_policy",
        "policy_generation": 1,
        "policy_effect_evidence_id": "policy-effect-1",
        "deterministic_replay": True,
        "actual_state_digest": "state-after",
        "masked_state_digest": "state-without-policy",
        "actual_queue_order_digest": "queue-after-review",
        "masked_queue_order_digest": "queue-without-policy",
        "material_delta": 2.0,
        "materiality_threshold": 1.0,
        "effect_ticks": [7],
    }
    evidence["agent_caused_state_change_ticks"] = [5, 7]
    evidence["scheduled_review_ticks"] = []
    evidence["periodic_scan_ticks"] = []
    evidence["continuous_supervisory_review_ticks"] = []

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert row["checks"]["difficulty_appropriate_control_response_observed"] is False
    assert "standing_plan_response_unproven" in row["blockers"]


def _direct_persistent_policy_evidence() -> dict:
    return {
        "persistent_control_tool_names": ["set_queue_policy"],
        "persistent_policy_review_records": [
            {
                "review_id": "review-1",
                "review_tick": 6,
                "event_ids": ["arrival-1"],
                "event_ticks": [4],
                "policy_generation": 1,
                "policy_digest": "policy-v1",
                "queue_order_digest": "queue-after-review",
                "policy_tool_name": "set_queue_policy",
                "policy_effect_evidence_id": "policy-effect-1",
                "review_tool_name": "review_persistent_policy",
                "decision": "keep",
                "outcome_effect_ticks": [7],
                "evidence_ids": ["review-evidence-1"],
            }
        ],
        "persistent_policy_review_bindings": [
            {
                "review_id": "review-1",
                "review_tool_name": "review_persistent_policy",
                "call_id": "review-call-1",
                "accepted": True,
                "evidence_ids": ["review-evidence-1"],
                "action_graph_evidence_ids": ["review-evidence-1"],
            }
        ],
        "persistent_policy_effect_bindings": [
            {
                "policy_generation": 1,
                "policy_tool_name": "set_queue_policy",
                "accepted": True,
                "effect_tick": 1,
                "call_id": "policy-call-1",
                "evidence_ids": ["policy-effect-1"],
            }
        ],
        "persistent_policy_attribution": {
            "status": "passed",
            "review_id": "review-1",
            "review_tool_name": "review_persistent_policy",
            "policy_tool_name": "set_queue_policy",
            "policy_generation": 1,
            "policy_effect_evidence_id": "policy-effect-1",
            "deterministic_replay": True,
            "actual_state_digest": "state-after",
            "masked_state_digest": "state-without-policy",
            "actual_queue_order_digest": "queue-after-review",
            "masked_queue_order_digest": "queue-without-policy",
            "material_delta": 2.0,
            "materiality_threshold": 1.0,
            "effect_ticks": [7],
        },
    }


def _direct_persistent_policy_source_row() -> dict:
    return {
        "backend_kind": "alibaba_trace_sim",
        "capability": {
            "control_tools": ["set_queue_policy"],
            "persistent_policy_review": {
                "backend_kind": "alibaba_trace_sim",
                "review_tool_name": "review_persistent_policy",
                "policy_tool_names": ["set_queue_policy"],
            },
        },
    }


def test_persistent_policy_effect_binding_requires_tool_and_finite_tick() -> None:
    for mutation in (
        {"policy_tool_name": "reserve_gpu_capacity"},
        {"effect_tick": float("nan")},
    ):
        evidence = _direct_persistent_policy_evidence()
        evidence["persistent_policy_effect_bindings"][0].update(mutation)
        result = _persistent_policy_evidence(
            evidence=evidence,
            material_exogenous_records=[{"event_id": "arrival-1", "applied_tick": 4}],
            post_change_decision_ticks=[5, 6],
            horizon=12,
            control_tools=["set_queue_policy"],
            successful_control_tools={"set_queue_policy"},
            source_row=_direct_persistent_policy_source_row(),
        )

        assert result["review_observed"] is False
        assert result["attribution_passed"] is False


def test_persistent_policy_review_without_evidence_binding_is_held() -> None:
    evidence = {
        "persistent_policy_review_records": [
            {
                "review_id": "review-1",
                "review_tick": 6,
                "event_ids": ["arrival-1"],
                "event_ticks": [4],
                "policy_generation": 1,
                "policy_digest": "policy-v1",
                "queue_order_digest": "queue-after-review",
                "policy_tool_name": "set_queue_policy",
                "policy_effect_evidence_id": "policy-effect-1",
                "review_tool_name": "review_persistent_policy",
                "decision": "keep",
                "outcome_effect_ticks": [7],
            }
        ],
        "persistent_policy_attribution": {
            "status": "passed",
            "review_id": "review-1",
            "review_tool_name": "review_persistent_policy",
            "policy_tool_name": "set_queue_policy",
            "policy_generation": 1,
            "policy_effect_evidence_id": "policy-effect-1",
            "deterministic_replay": True,
            "actual_state_digest": "state-after",
            "masked_state_digest": "state-without-policy",
            "actual_queue_order_digest": "queue-after-review",
            "masked_queue_order_digest": "queue-without-policy",
            "material_delta": 2.0,
            "materiality_threshold": 1.0,
            "effect_ticks": [7],
        },
    }
    result = _persistent_policy_evidence(
        evidence=evidence,
        material_exogenous_records=[{"event_id": "arrival-1", "applied_tick": 4}],
        post_change_decision_ticks=[5, 6],
        horizon=12,
        control_tools=["set_queue_policy"],
        successful_control_tools={"set_queue_policy"},
        source_row={
            "backend_kind": "alibaba_trace_sim",
            "capability": {
                "control_tools": ["set_queue_policy"],
                "persistent_policy_review": {
                    "backend_kind": "alibaba_trace_sim",
                    "review_tool_name": "review_persistent_policy",
                    "policy_tool_names": ["set_queue_policy"],
                },
            },
        },
    )

    assert result["review_observed"] is False
    assert result["attribution_passed"] is False


def test_persistent_policy_review_accepts_multiple_events_at_one_tick() -> None:
    evidence = {
        "persistent_control_tool_names": ["set_queue_policy"],
        "persistent_policy_review_records": [
            {
                "review_id": "review-batch-1",
                "review_tick": 6,
                "event_ids": ["arrival-a", "arrival-b"],
                "event_ticks": [4, 4],
                "policy_generation": 1,
                "policy_digest": "policy-v1",
                "queue_order_digest": "queue-after-review",
                "policy_tool_name": "set_queue_policy",
                "policy_effect_evidence_id": "policy-effect-1",
                "review_tool_name": "review_persistent_policy",
                "decision": "keep",
                "outcome_effect_ticks": [7],
                "evidence_ids": ["review-evidence-1"],
            }
        ],
        "persistent_policy_attribution": {
            "status": "passed",
            "review_id": "review-batch-1",
            "review_tool_name": "review_persistent_policy",
            "policy_tool_name": "set_queue_policy",
            "policy_generation": 1,
            "policy_effect_evidence_id": "policy-effect-1",
            "deterministic_replay": True,
            "actual_state_digest": "state-after",
            "masked_state_digest": "state-without-policy",
            "actual_queue_order_digest": "queue-after-review",
            "masked_queue_order_digest": "queue-without-policy",
            "material_delta": 2.0,
            "materiality_threshold": 1.0,
            "effect_ticks": [7],
        },
        "persistent_policy_review_bindings": [
            {
                "review_id": "review-batch-1",
                "review_tool_name": "review_persistent_policy",
                "call_id": "review-call-1",
                "accepted": True,
                "evidence_ids": ["review-evidence-1"],
                "action_graph_evidence_ids": ["review-evidence-1"],
            }
        ],
        "persistent_policy_effect_bindings": [
            {
                "policy_generation": 1,
                "policy_tool_name": "set_queue_policy",
                "accepted": True,
                "evidence_ids": ["policy-effect-1"],
                "effect_tick": 1,
                "call_id": "policy-call-1",
            }
        ],
    }
    result = _persistent_policy_evidence(
        evidence=evidence,
        material_exogenous_records=[
            {"event_id": "arrival-a", "applied_tick": 4},
            {"event_id": "arrival-b", "applied_tick": 4},
        ],
        post_change_decision_ticks=[5, 6],
        horizon=12,
        control_tools=["set_queue_policy"],
        successful_control_tools={"set_queue_policy"},
        source_row={
            "backend_kind": "alibaba_trace_sim",
            "capability": {
                "control_tools": ["set_queue_policy"],
                "persistent_policy_review": {
                    "backend_kind": "alibaba_trace_sim",
                    "review_tool_name": "review_persistent_policy",
                    "policy_tool_names": ["set_queue_policy"],
                },
            },
        },
    )

    assert result["review_observed"] is True
    assert result["attribution_passed"] is True
    assert result["timeline_observed"] is True


def test_persistent_policy_review_without_masked_attribution_is_held(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    artifacts["source_suite"]["scenarios"][0]["difficulty_level"] = "medium"
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["standing_plan_committed"] = False
    evidence["standing_plan_commit_ticks"] = []
    evidence["standing_control_commit_ticks"] = []
    evidence["scheduled_review_ticks"] = []
    evidence["periodic_scan_ticks"] = []
    evidence["continuous_supervisory_review_ticks"] = []
    evidence["persistent_policy_review_records"] = [
        {
            "review_id": "review-incident-1",
            "review_tick": 6,
            "event_ids": ["incident-1"],
            "event_ticks": [4],
            "policy_generation": 1,
            "policy_digest": "policy-v1",
            "queue_order_digest": "queue-after-review",
            "decision": "keep",
            "outcome_effect_ticks": [7],
            "evidence_ids": ["review-evidence-1"],
        }
    ]
    evidence["persistent_policy_attribution"] = {
        "status": "held",
        "review_id": "review-incident-1",
        "event_ids": ["incident-1"],
        "deterministic_replay": False,
    }

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert row["checks"]["difficulty_appropriate_control_response_observed"] is False
    assert "standing_plan_response_unproven" in row["blockers"]


def test_quality_core_v2_requires_persistent_runtime_contract(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    artifacts["source_suite"]["constraints"] = {
        "core_admission_profile": "quality_core_v2"
    }
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["realized_predesigned_event_ids"] = []
    evidence["realized_predesigned_event_ticks"] = []
    evidence["world_change_contract_declared"] = False
    evidence["event_adaptive_cadence_declared"] = False
    evidence["scheduled_review_ticks"] = []
    evidence["autonomous_hold_ticks"] = 0

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "held"
    assert row["admission_profile"] == "quality_core_v2"
    assert set(row["admission_blockers"]) == {
        "check_failed:event_adaptive_cadence_declared",
        "check_failed:world_change_contract_declared",
    }
    assert "actual_supervisory_review_unobserved" in row["diagnostic_blockers"]
    assert "parallel_clock_not_observed" in row["diagnostic_blockers"]
    assert "predesigned_event_not_reached" in row["diagnostic_blockers"]
    assert "event_adaptive_cadence_undeclared" in row["diagnostic_blockers"]


def test_quality_core_v2_keeps_difficulty_specific_response_diagnostic(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    artifacts["source_suite"]["constraints"] = {
        "core_admission_profile": "quality_core_v2"
    }
    evidence = artifacts["behavioral"]["results"][0]["agentic_evidence"]
    evidence["adaptive_replanning_observed"] = False
    evidence["valid_plan_delegation_observed"] = False

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "passed"
    assert row["admission_blockers"] == []
    assert row["checks"]["post_change_decision_observed"] is True
    assert row["checks"]["agent_action_backend_effect_observed"] is True
    assert "adaptive_replanning_or_delegation_unproven" in row["diagnostic_blockers"]


def test_quality_core_v2_keeps_reference_agent_behavior_diagnostic(
    tmp_path: Path,
) -> None:
    mutations = (
        (
            "post_change_decision",
            lambda evidence: evidence.update(
                post_change_decision_ticks=[],
                event_to_decision_action_edges=[],
            ),
            "post_change_decision_unproven",
        ),
        (
            "backend_effect",
            lambda evidence: evidence.update(
                agent_action_backend_effect_observed=False,
            ),
            "agent_action_backend_effect_unproven",
        ),
        (
            "native_control_use",
            lambda evidence: evidence.update(
                successful_native_control_tool_names=[],
            ),
            "evidence_missing_native_control_use",
        ),
    )

    for case_name, mutate, expected_diagnostic in mutations:
        artifacts = _artifacts()
        artifacts["source_suite"]["constraints"] = {
            "core_admission_profile": "quality_core_v2"
        }
        mutate(artifacts["behavioral"]["results"][0]["agentic_evidence"])
        case_dir = tmp_path / case_name
        case_dir.mkdir()

        report = build_agentic_contract_report(
            **artifacts,
            input_bindings=_write_artifacts(case_dir, artifacts),
        )

        row = report["results"][0]
        assert row["status"] == "passed"
        assert row["admission_blockers"] == []
        assert expected_diagnostic in row["diagnostic_blockers"]


def test_quality_core_v2_keeps_task_outcome_diagnostic_without_masking_native_use(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts()
    artifacts["source_suite"]["constraints"] = {
        "core_admission_profile": "quality_core_v2"
    }
    artifacts["task_contracts"]["results"][0].update(
        {
            "status": "failed",
            "completed": False,
            "terminal_integrity": {"release_ready": False},
        }
    )

    report = build_agentic_contract_report(
        **artifacts,
        input_bindings=_write_artifacts(tmp_path, artifacts),
    )

    row = report["results"][0]
    assert row["status"] == "passed"
    assert row["checks"]["successful_reference_used_native_control"] is True
    assert row["checks"]["task_contract_passed"] is False
    assert row["checks"]["terminal_integrity_passed"] is False
    assert row["admission_blockers"] == []
    assert "task_contract_failed" in row["diagnostic_blockers"]
    assert "terminal_integrity_failed" in row["diagnostic_blockers"]


def test_quality_core_v2_rejects_stale_or_cross_identity_evidence(
    tmp_path: Path,
) -> None:
    for mutation in ("stale_semantics", "signature_mismatch"):
        case_dir = tmp_path / mutation
        case_dir.mkdir()
        artifacts = _artifacts()
        artifacts["source_suite"]["constraints"] = {
            "core_admission_profile": "quality_core_v2"
        }
        if mutation == "stale_semantics":
            artifacts["behavioral"]["evaluation_semantics"] = {
                **_semantics(),
                "protocol_version": "2.0",
            }
        else:
            artifacts["behavioral"]["results"][0]["scenario_signature"] = (
                "wrong-signature"
            )

        report = build_agentic_contract_report(
            **artifacts,
            input_bindings=_write_artifacts(case_dir, artifacts),
        )

        row = report["results"][0]
        assert row["status"] == "held"
        assert any(
            blocker
            in {
                "check_failed:current_protocol_semantics",
                "check_failed:identity_bound_across_artifacts",
                "check_failed:scenario_signature_current",
            }
            for blocker in row["admission_blockers"]
        )
