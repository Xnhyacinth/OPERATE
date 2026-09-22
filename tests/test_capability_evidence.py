"""Offline evidence binding rejects changed summaries and detached artifacts."""
from copy import deepcopy
import hashlib
import json

from evaluation.capability_evidence import bind_capability_evidence


def fixture(tmp_path):
    ledger = [{"evidence_id": "e1", "tick": 1, "kind": "realized_event"}]
    trace = [{"tick": 1, "evidence_ids": ["e1"]}]
    cf = {"actual_cost": 12, "per_action": []}
    snapshot = {"payload": {
        "identity": {"scenario_signature": "sig", "seed": 42,
                     "implementation": {"implementation_tree_sha256": "tree"}},
        "inputs": {"cost_components": {"cost": 12}, "counterfactual_report": cf,
                   "evidence_logger": {"items": ledger}},
    }}
    summary = {"n_ticks": 1, "operational_agency_valid_evidence_ids": ["e1"]}
    for key, name, value in [
        ("trajectory_artifact", "sample.trajectory.jsonl", trace),
        ("evidence_ledger_artifact", "sample.evidence.jsonl", ledger),
        ("scoring_inputs_artifact", "sample.scoring_inputs.json", snapshot),
    ]:
        p = tmp_path / name
        raw = ("\n".join(json.dumps(r) for r in value) if isinstance(value, list)
               else json.dumps(value)).encode()
        p.write_bytes(raw)
        summary[key] = {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(),
                        "byte_count": len(raw)}
        if isinstance(value, list):
            summary[key]["event_count"] = len(value)
    episode = {"scenario_signature": "sig", "seed": 42,
               "implementation_tree_sha256": "tree", "n_ticks_ran": 1,
               "ground_truth_summary": {"cost_components": {"cost": 12}},
               "counterfactual": cf, "trajectory_summary": summary}
    return episode, {"scenario_signature": "sig", "seed": 42, "horizon_ticks": 2}


def test_binds_real_artifacts_and_live_cost(tmp_path):
    episode, spec = fixture(tmp_path)
    before = deepcopy(episode)
    bound = bind_capability_evidence(episode, spec)
    assert bound["verified"] is True
    assert bound["native_cost_bound"] is True
    assert bound["trace_ticks"] == 1
    assert episode == before


def test_changed_cost_fails_even_with_authentic_attachment(tmp_path):
    episode, spec = fixture(tmp_path)
    episode["ground_truth_summary"]["cost_components"]["cost"] = -999
    assert bind_capability_evidence(episode, spec)["reason"] == "native_cost_snapshot_mismatch"


def test_unknown_evidence_is_not_authoritative(tmp_path):
    episode, spec = fixture(tmp_path)
    episode["trajectory_summary"]["operational_agency_valid_evidence_ids"].append("fake")
    assert bind_capability_evidence(episode, spec)["reason"] == "summary_evidence_not_in_ledger"


def test_missing_artifact_is_unavailable():
    assert bind_capability_evidence({}, {})["verified"] is False


def test_changed_trace_and_inflated_horizon_are_rejected(tmp_path):
    episode, spec = fixture(tmp_path)
    episode["n_ticks_ran"] = 400
    assert bind_capability_evidence(episode, spec)["reason"] == "trajectory_tick_count_mismatch"
    episode["n_ticks_ran"] = 1
    path = episode["trajectory_summary"]["trajectory_artifact"]["path"]
    from pathlib import Path
    Path(path).write_text('{"tick": 99}\n')
    assert bind_capability_evidence(episode, spec)["reason"] == "artifact_hash_mismatch"


def test_changed_replay_summary_is_rejected(tmp_path):
    episode, spec = fixture(tmp_path)
    episode["counterfactual"]["actual_cost"] = 13
    assert bind_capability_evidence(episode, spec)["reason"] == "counterfactual_snapshot_mismatch"


def test_detached_header_identity_is_rejected(tmp_path):
    episode, spec = fixture(tmp_path)
    (tmp_path / "sample.header.json").write_text(json.dumps({
        "scenario_signature": "wrong", "seed": 42, "total_ticks": 1,
        "horizon_ticks": 2,
    }))
    assert bind_capability_evidence(episode, spec)["reason"] == "trajectory_header_identity_mismatch"


def test_archive_resolution_uses_only_exact_declared_suffix(tmp_path):
    from pathlib import Path
    episode, spec = fixture(tmp_path)
    root = tmp_path / "archive"
    target = root / "trajectories" / "model"
    target.mkdir(parents=True)
    for descriptor in episode["trajectory_summary"].values():
        if isinstance(descriptor, dict) and "path" in descriptor:
            old = Path(descriptor["path"])
            (target / old.name).write_bytes(old.read_bytes())
            descriptor["path"] = str(tmp_path / "missing" / "trajectories" / "model" / old.name)
    assert bind_capability_evidence(episode, spec, root / "episodes.jsonl")["verified"]
    assert not bind_capability_evidence(episode, spec)["verified"]


def test_user_assumption_separates_code_metadata_without_changing_artifacts(tmp_path):
    episode, spec = fixture(tmp_path)
    episode["implementation_tree_sha256"] = "different-tree"
    before = deepcopy(episode)
    assert bind_capability_evidence(episode, spec)["reason"] == "scoring_snapshot_runtime_mismatch"
    bound = bind_capability_evidence(episode, spec, comparison_policy="latest_framework_user_assumed")
    assert bound["verified"] and bound["native_cost_bound"] and bound["counterfactual_bound"]
    assert bound["execution_metadata"]["snapshot_runtime_identity"] == "tree"
    assert bound["execution_metadata"]["episode_runtime_identity"] == "different-tree"
    assert bound["execution_metadata"]["compatibility_user_assumed"] is True
    assert episode == before
    episode["ground_truth_summary"]["cost_components"]["cost"] = -1
    assert bind_capability_evidence(episode, spec, comparison_policy="latest_framework_user_assumed")["reason"] == "native_cost_snapshot_mismatch"


def test_user_assumption_preserves_hash_and_case_checks(tmp_path):
    from pathlib import Path
    episode, spec = fixture(tmp_path)
    episode["implementation_tree_sha256"] = "different-tree"
    assert bind_capability_evidence(episode, {**spec, "seed": 9}, comparison_policy="latest_framework_user_assumed")["reason"] == "episode_suite_identity_mismatch"
    Path(episode["trajectory_summary"]["trajectory_artifact"]["path"]).write_text('{}\n')
    assert bind_capability_evidence(episode, spec, comparison_policy="latest_framework_user_assumed")["reason"] == "artifact_hash_mismatch"
