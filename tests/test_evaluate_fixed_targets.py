import hashlib
import json

import pytest

from scripts import evaluate_fixed_targets as evaluator


def sign_target(target):
    target.pop("contract_sha256", None)
    target["contract_sha256"] = hashlib.sha256(
        json.dumps(
            target, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def fixture(tmp_path, monkeypatch, backend="citylearn"):
    spec = dict(
        scenario_signature="a",
        seed=42,
        domain="d",
        backend_kind=backend,
        source_denominator_key="s",
        physical_source_key="s",
        yaml_sha256="yaml",
        horizon_ticks=72,
    )
    row = dict(
        **spec,
        model="m",
        implementation_tree_sha256="tree",
        implementation_tree_sha256_start="tree",
        implementation_tree_sha256_end="tree",
        agent_profile_sha256="p",
        agent_treatment_sha256="t",
        interaction_mode="logical_persistent",
        run_semantics_fingerprint="f",
        suite_manifest_sha256="suite",
        score={"scoring_version": "0.21.0"},
    )
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"scenarios": [spec]}))
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps(row) + "\n")
    target = dict(
        **spec,
        runtime_identity="tree",
        status="ready",
        objective_id="cost",
        unit="usd",
        target_cost=10,
        target_evidence_ids=["reference"],
        absolute_tolerance=1e-7,
        relative_tolerance=1e-10,
        schema_version="fixed_expert_target.v1",
        candidate_policies=["greedy_heuristic", "oracle_offline"],
        source_yaml_sha256="yaml",
    )
    from evaluation.native_quality import ABS_TOLERANCE, REL_TOLERANCE

    target.update(absolute_tolerance=ABS_TOLERANCE, relative_tolerance=REL_TOLERANCE)
    sign_target(target)
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            dict(
                schema_version="fixed_target_suite.v1",
                suite_sha256=hashlib.sha256(suite.read_bytes()).hexdigest(),
                targets=[target],
                target_provenance=[],
            )
        )
    )
    config = dict(
        suite=str(suite),
        runs={"m": [str(episodes)]},
        input_sha256={str(episodes): hashlib.sha256(episodes.read_bytes()).hexdigest()},
        fixed_targets=dict(
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        ),
    )
    old = dict(
        artifact_binding=dict(verified=True, native_cost_bound=True),
        scenario_signature="a",
        seed=42,
        execution_provenance={"model": "m"},
    )
    group = dict(graded_episodes=[old])

    def legacy(config):
        assert config["references"] == []
        assert config["require_input_hashes"] is True
        return dict(
            by_model={"m": group},
            comparison_policy="strict",
            compatibility_user_assumed=False,
            episode_inputs=[],
        )

    monkeypatch.setattr(evaluator, "evaluate_legacy", legacy)
    native = dict(
        applicable=True,
        actual_cost=9.0,
        feasible=True,
        hard_failure=False,
        task_success=True,
        evidence_ids=["e"],
        objective_id="cost",
        unit="usd",
    )
    monkeypatch.setattr(evaluator, "extract_native_objective", lambda row: native)
    return config, group, native


def test_fixed_scoring_needs_no_weak_reference(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch)
    assert evaluator.evaluate(config)["leaderboard"][0]["score"] == 100


@pytest.mark.parametrize("field", ["input_sha256", "fixed_targets"])
def test_hash_tampering_refused(tmp_path, monkeypatch, field):
    config, _, _ = fixture(tmp_path, monkeypatch)
    if field == "fixed_targets":
        config[field]["sha256"] = "wrong"
    else:
        config[field] = {}
    with pytest.raises(ValueError, match="hash"):
        evaluator.evaluate(config)


def test_fjsp_incomplete_is_zero(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch, "dynasched_flexible_job_shop")
    monkeypatch.setattr(
        evaluator,
        "evaluate_task_quality",
        lambda *a, **k: dict(
            mandatory_complete=False, evidence_ids=["terminal"], reason="incomplete"
        ),
    )
    assert evaluator.evaluate(config)["leaderboard"][0]["score"] == 0


@pytest.mark.parametrize("block", ["identity", "binding"])
def test_identity_and_binding_cannot_be_bypassed_by_hard_failure(
    tmp_path, monkeypatch, block
):
    config, group, native = fixture(tmp_path, monkeypatch)
    native["hard_failure"] = True
    if block == "identity":
        group["identity_blocker"] = "mixed_execution"
    else:
        group["graded_episodes"][0]["artifact_binding"]["verified"] = False
    report = evaluator.evaluate(config)
    assert report["models"][0]["score"] is None
    assert not report["leaderboard"]


def test_missing_target_not_hidden_by_hard_failure(tmp_path, monkeypatch):
    config, _, native = fixture(tmp_path, monkeypatch)
    native["hard_failure"] = True
    target_path = config["fixed_targets"]["path"]
    from pathlib import Path

    payload = json.loads(Path(target_path).read_bytes())
    payload["targets"] = []
    Path(target_path).write_text(json.dumps(payload))
    config["fixed_targets"]["sha256"] = hashlib.sha256(
        Path(target_path).read_bytes()
    ).hexdigest()
    assert evaluator.evaluate(config)["models"][0]["score"] is None


def test_complete_fjsp_without_old_quality_reference_can_score(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch, "dynasched_flexible_job_shop")
    monkeypatch.setattr(
        evaluator,
        "evaluate_task_quality",
        lambda *a, **k: dict(
            mandatory_complete=True,
            evidence_ids=["terminal"],
            score=None,
            reason="quality_reference_unavailable",
        ),
    )
    assert evaluator.evaluate(config)["leaderboard"][0]["score"] == 100


def test_fjsp_missing_completion_evidence_is_na(tmp_path, monkeypatch):
    config, _, _ = fixture(tmp_path, monkeypatch, "dynasched_flexible_job_shop")
    monkeypatch.setattr(
        evaluator,
        "evaluate_task_quality",
        lambda *a, **k: dict(
            mandatory_complete=None,
            evidence_ids=[],
            reason="terminal_native_evidence_missing",
        ),
    )
    assert evaluator.evaluate(config)["models"][0]["score"] is None


def test_strict_target_runtime_identity_is_not_implicitly_relaxed(
    tmp_path, monkeypatch
):
    config, _, _ = fixture(tmp_path, monkeypatch)
    from pathlib import Path

    target_path = Path(config["fixed_targets"]["path"])
    payload = json.loads(target_path.read_bytes())
    payload["targets"][0]["runtime_identity"] = "another_tree"
    sign_target(payload["targets"][0])
    target_path.write_text(json.dumps(payload))
    config["fixed_targets"]["sha256"] = hashlib.sha256(
        target_path.read_bytes()
    ).hexdigest()
    report = evaluator.evaluate(config)
    assert report["models"][0]["score"] is None
    assert (
        report["graded_episodes"][0]["measurement"]["reason"]
        == "fixed_target_runtime_identity_mismatch"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_yaml_sha256", "wrong"),
        ("horizon_ticks", 999),
        ("domain", "other"),
        ("candidate_policies", ["wait_only"]),
        ("contract_sha256", "wrong"),
    ],
)
def test_target_source_protocol_and_contract_hash_checked(
    tmp_path, monkeypatch, field, value
):
    config, _, _ = fixture(tmp_path, monkeypatch)
    from pathlib import Path

    target_path = Path(config["fixed_targets"]["path"])
    payload = json.loads(target_path.read_bytes())
    payload["targets"][0][field] = value
    target_path.write_text(json.dumps(payload))
    config["fixed_targets"]["sha256"] = hashlib.sha256(
        target_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="fixed target"):
        evaluator.evaluate(config)
