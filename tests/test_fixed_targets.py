from copy import deepcopy

import pytest

from evaluation.fixed_targets import compile_target, score_target


def fixture():
    measurement = dict(
        applicable=True,
        actual_cost=10.0,
        feasible=True,
        hard_failure=False,
        objective_id="native",
        unit="cost",
        evidence_ids=["native-proof"],
    )
    reference = dict(
        scenario_signature="s",
        seed=1,
        backend_kind="citylearn",
        policy_measurements={
            p: [deepcopy(measurement), deepcopy(measurement)]
            for p in ("greedy_heuristic", "oracle_offline")
        },
        policy_determinism={p: True for p in ("greedy_heuristic", "oracle_offline")},
    )
    return measurement, reference


def test_target_does_not_need_weak_policy_gap_or_positive_cost():
    measurement, reference = fixture()
    for cost in (10.0, 0.0, -10.0):
        for pair in reference["policy_measurements"].values():
            for m in pair:
                m["actual_cost"] = cost
        target = compile_target(reference)
        measurement["actual_cost"] = cost
        assert score_target(measurement, target)["score"] == 100
        measurement["actual_cost"] = cost + 1
        assert score_target(measurement, target)["score"] == 0
        measurement["actual_cost"] = cost - 1
        assert score_target(measurement, target)["score"] == 100


@pytest.mark.parametrize("fault", ["missing", "unstable", "no_proof"])
def test_missing_candidate_does_not_weaken_target(fault):
    _, ref = fixture()
    pair = ref["policy_measurements"]["oracle_offline"]
    if fault == "missing":
        pair.pop()
    elif fault == "unstable":
        pair[1]["actual_cost"] += 1
    else:
        pair[1]["evidence_ids"] = []
    assert compile_target(ref)["status"] == "unavailable"


def test_proven_infeasible_candidate_excluded_not_failed_execution():
    m, ref = fixture()
    for candidate in ref["policy_measurements"]["oracle_offline"]:
        candidate.update(feasible=False, hard_failure=True, actual_cost=-100)
    t = compile_target(ref)
    assert t["target_cost"] == 10
    assert t["target_policies"] == ["greedy_heuristic"]
    m["hard_failure"] = True
    assert score_target(m, t)["score"] == 0


def test_objective_mismatch_or_missing_measurement_is_unavailable():
    m, ref = fixture()
    t = compile_target(ref)
    m["objective_id"] = "different"
    assert score_target(m, t)["score"] is None
    m["objective_id"] = "native"
    m["actual_cost"] = float("nan")
    assert score_target(m, t)["score"] is None


def test_reference_addition_cannot_affect_frozen_target():
    m, ref = fixture()
    t = compile_target(ref)
    ref["policy_measurements"]["random"] = [{**m, "actual_cost": -10000}] * 2
    assert compile_target(ref) == t
    m["actual_cost"] = 10 + 1e-9
    assert score_target(m, t)["attained"]
    m["actual_cost"] = 11
    assert not score_target(m, t)["attained"]


@pytest.mark.parametrize("proof", ["not-a-list", [None], [""], []])
def test_invalid_target_proof_cannot_create_score(proof):
    m, ref = fixture()
    t = compile_target(ref)
    t["target_evidence_ids"] = proof
    assert score_target(m, t)["score"] is None
