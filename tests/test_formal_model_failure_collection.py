from __future__ import annotations

import hashlib
import json

import pytest

from scripts import batch_llm_eval as batch
from tests.test_batch_llm_eval import _formally_eligible_protocol21_row


def _write_artifact(tmp_path, row, key, suffix, schema, records):
    path = tmp_path / f"episode.{suffix}.jsonl"
    data = "".join(json.dumps(record) + "\n" for record in records).encode()
    path.write_bytes(data)
    row["trajectory_summary"][key] = {
        "path": str(path),
        "schema_version": schema,
        "sha256": hashlib.sha256(data).hexdigest(),
        "event_count": len(records),
        "byte_count": len(data),
    }


def _fixture(tmp_path):
    row = _formally_eligible_protocol21_row()
    row["trajectory_summary"]["trajectory_path"] = str(tmp_path / "episode")
    observation = {"tick": 1, "__last_early_stop_warnings__": ["MRM"]}
    request = {"sequence": 1, "envelope": {"messages": []}}
    response = {
        "sequence": 1,
        "request_sequence": 1,
        "response": {"status": "success", "decision_valid": False},
    }
    envelope = {
        "simulator_tick": 1,
        "model_decision_index": 2,
        "provider_status": "success",
        "pre_action_observation": observation,
        "pre_action_observation_sha256": batch._canonical_json_sha256(observation),
        "presented_early_stop_warnings": ["MRM"],
        "provider_requests": [request],
        "provider_responses": [response],
    }
    steps = [
        {"tick": 0, "info": {"early_stop_warnings": ["MRM"]}},
        {
            "tick": 1,
            "observation": {"tick": 2},
            "action": {
                "dominant_action": "protocol_repair_no_tool_call",
                "actions": [],
            },
            "info": {
                "early_stop_warnings": ["MRM"],
                "realized_events": [],
                "forecast_updates": {},
                "decision_envelope": envelope,
            },
        },
    ]
    failure = {
        "simulator_tick": 1,
        "model_decision_index": 2,
        "dominant": "protocol_repair_no_tool_call",
        "provider_status": "success",
        "presented_warnings": ["MRM"],
        "terminal_warnings": ["MRM"],
        "pre_action_observation_sha256": envelope["pre_action_observation_sha256"],
        "provider_request_sequences": [1],
    }
    row["trajectory_summary"]["terminal_integrity"] = {
        "release_ready": False,
        "collection_complete": True,
        "terminal_disposition": "observed_invalid_terminal_model_response",
        "unresolved_pending_actions": {},
        "unanswered_interrupt_reasons": ["safety_warning"],
        "model_response_failure": failure,
    }
    _write_artifact(
        tmp_path,
        row,
        "trajectory_artifact",
        "trajectory",
        "episode_trajectory_jsonl_v1",
        steps,
    )
    _write_artifact(
        tmp_path,
        row,
        "provider_audit_artifact",
        "provider_audit",
        "provider_interaction_audit_v1",
        [
            {"record_kind": "provider_request", **request},
            {"record_kind": "provider_response", **response},
        ],
    )
    return row, steps


def test_completed_invalid_response_remains_formally_measured(tmp_path):
    row, _ = _fixture(tmp_path)
    eligible, reasons = batch._formal_row_eligibility(row)
    assert eligible, reasons
    assert row["trajectory_summary"]["terminal_integrity"]["release_ready"] is False


@pytest.mark.parametrize(
    "change",
    [
        "pending",
        "metadata_only",
        "new_warning",
        "new_event",
        "transport",
        "forged_hash",
    ],
)
def test_collection_exception_never_excuses_unobserved_or_unverified_failure(
    tmp_path, change
):
    row, steps = _fixture(tmp_path)
    terminal = row["trajectory_summary"]["terminal_integrity"]
    if change == "pending":
        terminal["unresolved_pending_actions"] = {"call-x": 3}
    elif change == "metadata_only":
        del row["trajectory_summary"]["trajectory_artifact"]
    elif change == "new_warning":
        steps[-1]["info"]["early_stop_warnings"].append("new-risk")
        terminal["model_response_failure"]["terminal_warnings"].append("new-risk")
    elif change == "new_event":
        steps[-1]["info"]["realized_events"] = [
            {"type": "new-risk", "event_class": "safety"}
        ]
    elif change == "transport":
        steps[-1]["info"]["decision_envelope"]["provider_status"] = "failed"
    elif change == "forged_hash":
        terminal["model_response_failure"]["pre_action_observation_sha256"] = "0" * 64
    if change in {"new_warning", "new_event", "transport"}:
        _write_artifact(
            tmp_path,
            row,
            "trajectory_artifact",
            "trajectory",
            "episode_trajectory_jsonl_v1",
            steps,
        )
    assert batch._formal_row_eligibility(row)[0] is False


def test_changed_bound_trajectory_bytes_cannot_certify_collection(tmp_path):
    row, steps = _fixture(tmp_path)
    steps[-1]["info"]["decision_envelope"]["provider_status"] = "failed"
    path = tmp_path / "episode.trajectory.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in steps))
    assert batch._formal_row_eligibility(row)[0] is False


@pytest.mark.parametrize("defect", ["unresolved_native_call", "prior_missing_window"])
def test_header_cannot_hide_native_collection_defect(tmp_path, defect):
    row, steps = _fixture(tmp_path)
    if defect == "unresolved_native_call":
        steps[0]["tool_results"] = [
            {
                "call_id": "pending-native",
                "payload": {"_status": "pending", "due_tick": 3},
            }
        ]
    else:
        steps[0]["info"]["realized_events"] = [
            {
                "type": "hazard",
                "response_window_required": True,
                "terminal_response_window_missing": True,
            }
        ]
    _write_artifact(
        tmp_path,
        row,
        "trajectory_artifact",
        "trajectory",
        "episode_trajectory_jsonl_v1",
        steps,
    )
    assert batch._formal_row_eligibility(row)[0] is False


def test_real_loop_collection_metadata_is_admitted_from_bound_artifacts(tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace

    from core import Action
    from runner.episode import _run_episode_loop
    from tests.test_runner import FakeEnv, PersistentCaptureAgent

    class Invalid(PersistentCaptureAgent):
        def __init__(self):
            super().__init__()
            self.requests = []
            self.responses = []

        def act(self, observation, tool_specs):
            super().act(observation, tool_specs)
            sequence = len(self.requests) + 1
            self.requests.append({"sequence": sequence, "envelope": {"messages": []}})
            self.responses.append(
                {
                    "sequence": sequence,
                    "request_sequence": sequence,
                    "response": {"status": "success", "decision_valid": False},
                }
            )
            return Action(dominant="protocol_repair_no_tool_call")

        def get_last_provider_outcome(self):
            return {"status": "success"}

        def get_interaction_stats(self):
            return {
                "provider_request_records": self.requests,
                "provider_response_records": self.responses,
            }

    agent = Invalid()
    steps = []
    logger = SimpleNamespace(log_step=lambda **record: steps.append(deepcopy(record)))
    loop = _run_episode_loop(
        env=FakeEnv(rewards=[0.0, 0.0], warnings_per_tick=[["MRM"], ["MRM"]]),
        agent=agent,
        logger=logger,
    )
    row, _ = _fixture(tmp_path)
    row["trajectory_summary"]["terminal_integrity"] = loop["terminal_integrity"]
    _write_artifact(
        tmp_path,
        row,
        "trajectory_artifact",
        "trajectory",
        "episode_trajectory_jsonl_v1",
        steps,
    )
    _write_artifact(
        tmp_path,
        row,
        "provider_audit_artifact",
        "provider_audit",
        "provider_interaction_audit_v1",
        [
            *[{"record_kind": "provider_request", **r} for r in agent.requests],
            *[{"record_kind": "provider_response", **r} for r in agent.responses],
        ],
    )
    assert batch._formal_row_eligibility(row)[0] is True
    assert loop["terminal_integrity"]["release_ready"] is False
    assert loop["terminal_integrity"]["unanswered_interrupt_reasons"] == [
        "safety_warning"
    ]


def test_portable_model_failure_counts_in_coverage_and_diagnostic_scores(tmp_path):
    row, _ = _fixture(tmp_path)
    row.update(model="model-a", scenario_slug="fixture", seed=42)
    row.setdefault("score", {})["total_score"] = 12.5
    row["trajectory_summary"]["llm"]["native_tool_protocol_invalid_responses"] = 1
    row["trajectory_summary"]["trajectory_path"] = "episode"
    for key in ("trajectory_artifact", "provider_audit_artifact"):
        row["trajectory_summary"][key]["path"] = row["trajectory_summary"][key][
            "path"
        ].split("/")[-1]
    coverage = batch._coverage_summary(
        [row],
        configured_models=["model-a"],
        configured_seeds=[42],
        n_scenarios=1,
        batch_root=tmp_path,
    )
    assert coverage["is_partial_batch"] is False
    state = batch._batch_state(
        coverage=coverage, results=[row], log_audit_report={}, batch_root=tmp_path
    )
    assert state["batch_state"] == "final"
    assert state["n_model_output_failure_episodes"] == 1
    board = batch._leaderboard_from_rows(
        [row], "fixed_all_dimensions", batch_root=tmp_path
    )
    assert board[0]["n_episodes"] == 1
    assert board[0]["mean"] == 12.5
