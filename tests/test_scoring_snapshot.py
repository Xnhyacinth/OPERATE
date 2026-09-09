import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import EvidenceLogger
from core.ethical_dilemma import Dilemma, EthicalDilemmaManager, MoralChoice, MoralOption
from core.stakeholder_trust import StakeholderGroup, StakeholderTrustManager
from evaluation.scorer import ScoringInputs, score_episode, score_optimality_gap


def test_signed_native_cost_is_not_clipped_and_nonfinite_still_fails():
    score = score_optimality_gap(None, -4.2, objective_component="energy_cost",
                                evidence_ids=["native"], objective_value_domain="signed")
    assert not score.applicable
    assert score.evidence_ids == ["native"]
    positive_ref = score_optimality_gap(10.0, -4.2, objective_component="energy_cost",
                                      evidence_ids=["native"], objective_value_domain="signed")
    assert "-4.2" in positive_ref.reason
    for invalid in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            score_optimality_gap(None, invalid, objective_component="energy_cost",
                                evidence_ids=[], objective_value_domain="signed")
    with pytest.raises(ValueError, match="non-negative"):
        score_optimality_gap(10.0, -4.2, objective_component="production_cost", evidence_ids=[])


def test_scoring_snapshot_roundtrip_preserves_every_dimension_and_evidence(tmp_path):
    from evaluation.scoring_snapshot import snapshot_inputs, restore_inputs, write_rescore
    from data import TrajectoryLogger

    evidence = EvidenceLogger("saved")
    evidence.log("trust_event", 1)
    evidence.log("moral_choice", 1)
    trust = StakeholderTrustManager()
    trust.register(StakeholderGroup("a", "A", baseline_trust=.42))
    trust.register(StakeholderGroup("b", "B", baseline_trust=.79))
    trust.record_event("a", "promise_kept", 1)
    dilemmas = EthicalDilemmaManager()
    dilemmas.register_dilemma(Dilemma("d", 0, "test", [MoralOption("o", "O")]))
    dilemmas.maybe_trigger(0, {})
    dilemmas.record_choice(MoralChoice("d", "o", "reason", 0, reversibility_assessment="reversible"))
    dilemmas.record_consequence("d", {"casualties": 0, "expected": 0})
    inputs = ScoringInputs([], [], {"energy_cost": -4.2}, {}, {}, evidence, trust, dilemmas,
                           optimality_objective_component="energy_cost",
                           optimality_objective_value_domain="signed", scenario_signature="sig")
    payload = snapshot_inputs(inputs)
    restored = restore_inputs(json.loads(json.dumps(payload)))
    assert snapshot_inputs(restored) == payload
    assert restored.evidence_logger.to_jsonable() == evidence.to_jsonable()
    assert score_episode(restored).to_dict() == score_episode(inputs).to_dict()
    logger = TrajectoryLogger("saved", tmp_path)
    binding = logger.write_snapshot("scoring_inputs", {"identity": {"scenario_signature": "sig"},
                                                       "inputs": payload})
    source = Path(binding["path"])
    before = source.read_bytes()
    target = tmp_path / "rescore.json"
    write_rescore(source, binding["sha256"], target)
    result = json.loads(target.read_text())
    assert result["source_snapshot_sha256"] == binding["sha256"]
    assert result["score"] == score_episode(inputs).to_dict()
    assert result["formal_completion_claimed"] is False
    assert source.read_bytes() == before
    with pytest.raises(FileExistsError):
        write_rescore(source, binding["sha256"], target)
    with pytest.raises(ValueError, match="hash"):
        write_rescore(source, "0" * 64, tmp_path / "bad.json")


def test_nonfinite_ground_truth_survives_strict_json_snapshot(tmp_path):
    from data import TrajectoryLogger
    from evaluation.scoring_snapshot import decode_json

    binding = TrajectoryLogger("bad-native", tmp_path).write_snapshot(
        "completed_runtime", {"ground_truth": {"energy_cost": math.nan}})
    payload = json.loads(Path(binding["path"]).read_text())
    assert math.isnan(decode_json(payload["payload"])["ground_truth"]["energy_cost"])


@pytest.mark.parametrize("stage", ["counterfactual", "scoring"])
def test_completed_runtime_is_persisted_before_postprocessing_failure(tmp_path, monkeypatch, stage):
    from runner import episode
    from core import Action

    evidence = EvidenceLogger("completed")
    evidence.log("backend_tick", 1, {"cost": -4.2})
    agent = SimpleNamespace(reset=lambda *a, **k: None,
        get_interaction_stats=lambda: {"provider_request_records": [{"sequence": 1}],
                                      "provider_response_records": [{"sequence": 2, "request_sequence": 1}]})
    env = SimpleNamespace(reset=lambda *a, **k: None, evidence=evidence, stakeholders=None, dilemmas=None, tick=1)
    monkeypatch.setattr(episode, "make_agent", lambda *a, **k: agent)
    monkeypatch.setattr(episode, "_run_episode_loop", lambda **k: {
        "actions": [Action()], "tool_results_ok": 0, "tool_results_failed": 0,
        "stale_observation_records": [], "analysis_steps": []})
    monkeypatch.setattr(episode, "_snapshot_and_close_completed_environment",
                        lambda env: ({"cost_components": {"energy_cost": -4.2}}, None, [], []))

    def fail_cf(**kwargs):
        snapshots = list(tmp_path.glob("*.completed_runtime.json"))
        assert len(snapshots) == 1
        assert json.loads(snapshots[0].read_text())["payload"]["ground_truth"]["cost_components"]["energy_cost"] == -4.2
        assert list(tmp_path.glob("*.provider_audit.jsonl"))
        assert list(tmp_path.glob("*.evidence.jsonl"))
        if stage == "counterfactual":
            raise RuntimeError("counterfactual failed")
        return SimpleNamespace(actual_cost=-4.2, counterfactual_cost=10.0, prevented_loss=14.2,
            applicable=True, reason_code="", masking_policy="wait_only", per_action_status="disabled",
            per_action=[], per_action_group_status="disabled", per_action_groups=[],
            to_dict=lambda: {"actual_cost": -4.2, "counterfactual_cost": 10.0, "applicable": True})

    monkeypatch.setattr(episode, "domain_counterfactual_report", fail_cf)
    spec = SimpleNamespace(scenario_signature=lambda s, seed: "sig", env_factory=lambda: object,
        uses_runner_lp_oracle=False, objective_cost_component="energy_cost", equity_shed_key="unserved",
        adaptive_recovery_signal_key=None, adaptive_recovery_signal_name=None)
    monkeypatch.setattr(episode, "reference_optimum_from_backend_config", lambda env: None)
    monkeypatch.setattr(episode, "reference_optimum_objective_component", lambda env, **kwargs: "energy_cost")
    monkeypatch.setattr(episode, "_operational_agency_artifacts", lambda **kwargs: ([], set(), {}))
    monkeypatch.setattr(episode, "score_episode", lambda inputs: (_ for _ in ()).throw(RuntimeError("scoring failed")))
    with pytest.raises(RuntimeError, match=f"{stage} failed") as caught:
        episode._run_one_with_environment({"seed_id": "case", "backend_kind": "citylearn", "horizon_ticks": 1},
            "test", env=env, spec=spec, seed=42, agent_kwargs={}, trajectory_dir=tmp_path,
            counterfactual_masking="wait_only", multi_turn=False, multi_turn_rounds=1,
            per_action_attribution=False, per_action_cap=None, per_action_group_attribution=False,
            per_action_group_cap=None, within_tick_interaction=False)
    assert caught.value.episode_error_details["error_stage"] == stage
    assert caught.value.episode_error_details["completed_runtime_artifact"]["sha256"]
    if stage == "scoring":
        from evaluation.scoring_snapshot import write_rescore
        binding = caught.value.episode_error_details["scoring_inputs_artifact"]
        repaired = write_rescore(Path(binding["path"]), binding["sha256"], tmp_path / "rescored.json")
        assert repaired["score"]["scenario_signature"] == "sig"
        summary = json.loads(next(tmp_path.glob("*.summary.json")).read_text())
        assert summary["trajectory_summary"]["scoring_inputs_artifact"] == binding
