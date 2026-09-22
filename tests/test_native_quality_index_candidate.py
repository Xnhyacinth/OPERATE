"""Behavioral tests for the offline, unpromoted index candidate."""

from copy import deepcopy

import pytest

from tools.native_quality_index_candidate import score_quality, summarize


def contract():
    return {
        "scenario_signature": "case",
        "seed": 1,
        "objective_id": "makespan.v1",
        "runtime_identity": "tree",
        "suite_sha256": "suite",
        "interaction_mode": "logical_persistent",
        "workload_contract_hash": "workload",
        "unit": "native_time",
        "weak_cost": 1000.0,
        "strong_cost": 500.0,
        "minimum_anchor_gap": 10.0,
        "anchor_evidence_ids": ["weak-replay", "strong-replay"],
        "anchor_kind": "executed_policies",
        "domain": "logistics",
        "backend_kind": "dynasched",
        "source_denominator_key": "source",
    }


def sample(cost=750.0):
    return {
        "model": "model",
        "backend_kind": "dynasched",
        "source_denominator_key": "source",
        "scenario_signature": "case",
        "seed": 1,
        "objective_id": "makespan.v1",
        "runtime_identity": "tree",
        "suite_sha256": "suite",
        "interaction_mode": "logical_persistent",
        "workload_contract_hash": "workload",
        "unit": "native_time",
        "actual_cost": cost,
        "feasible": True,
        "hard_failure": False,
        "evidence_ids": ["cost", "constraints"],
    }


def test_quality_resolves_completed_schedules_and_below_reference_harm():
    assert score_quality(sample(600), contract())["score"] == 80
    assert score_quality(sample(900), contract())["score"] == 20
    assert score_quality(sample(1250), contract())["score"] == -50
    assert score_quality(sample(1000), contract())["score"] == 0


def test_signed_costs_and_unit_or_constant_offset_do_not_change_score():
    row, anchor = sample(), contract()
    expected = score_quality(row, anchor)["score"]
    for offset, scale in [(-2000, 1), (0, 1000), (99, 0.1)]:
        r, c = deepcopy(row), deepcopy(anchor)
        r["actual_cost"] = scale * (r["actual_cost"] + offset)
        for key in ("weak_cost", "strong_cost"):
            c[key] = scale * (c[key] + offset)
        c["minimum_anchor_gap"] *= scale
        assert score_quality(r, c)["score"] == pytest.approx(expected)


def test_gates_and_tail_values_are_explicit():
    row = sample(0)
    assert score_quality(row, contract())["raw_score"] == 200
    assert score_quality(row, contract())["score"] == 100
    row["feasible"] = False
    assert score_quality(row, contract())["score"] == -100
    row.pop("feasible")
    assert score_quality(row, contract())["score"] is None


@pytest.mark.parametrize("field", ["runtime_identity", "objective_id", "unit", "seed"])
def test_mismatched_identity_is_not_scored(field):
    row = sample()
    row[field] = "wrong"
    assert score_quality(row, contract())["score"] is None


@pytest.mark.parametrize("mutation", ["tiny_gap", "reversed", "no_evidence", "nan"])
def test_unusable_anchors_fail_closed(mutation):
    c = contract()
    if mutation == "tiny_gap":
        c["strong_cost"] = 999
    if mutation == "reversed":
        c["strong_cost"] = 1100
    if mutation == "no_evidence":
        c["anchor_evidence_ids"] = []
    if mutation == "nan":
        c["weak_cost"] = float("nan")
    assert score_quality(sample(), c)["score"] is None


def test_missing_case_or_duplicate_never_reweights_index():
    c = contract()
    c2 = {**c, "scenario_signature": "other", "source_denominator_key": "other"}
    report = summarize([sample()], [c, c2], ["model"])
    assert report["models"]["model"]["index"] is None
    report = summarize([sample(), sample()], [c], ["model"])
    assert report["models"]["model"]["index"] is None


def test_macro_balances_domains_and_does_not_rank_missing_models():
    c1 = contract()
    c2 = {**c1, "scenario_signature": "c2", "source_denominator_key": "s2"}
    c3 = {**c1, "scenario_signature": "c3", "domain": "grid"}
    rows = [
        sample(500),
        {**sample(500), "scenario_signature": "c2", "source_denominator_key": "s2"},
        {**sample(1000), "scenario_signature": "c3"},
    ]
    before = deepcopy(rows)
    result = summarize(rows, [c1, c2, c3], ["model", "missing"])
    assert result["models"]["model"]["index"] == 50
    assert result["models"]["missing"]["index"] is None
    assert result["formal_eligible"] is False
    assert rows == before


@pytest.mark.parametrize(
    "field",
    [
        "suite_sha256",
        "interaction_mode",
        "backend_kind",
        "source_denominator_key",
        "workload_contract_hash",
    ],
)
def test_conflicting_comparison_contract_is_rejected(field):
    row, c = sample(), contract()
    row[field] = "different"
    assert score_quality(row, c)["score"] is None


def test_aggregate_rejects_mixed_execution_scopes():
    c1 = contract()
    c2 = {**c1, "scenario_signature": "other", "runtime_identity": "other-tree"}
    rows = [
        sample(),
        {**sample(), "scenario_signature": "other", "runtime_identity": "other-tree"},
    ]
    with pytest.raises(ValueError, match="mixed comparison scope"):
        summarize(rows, [c1, c2], ["model"])


def test_saturation_and_pairs_reveal_hidden_quality_differences():
    rows = [{**sample(0), "model": "a"}, {**sample(-100), "model": "b"}]
    report = summarize(rows, [contract()], ["a", "b"])
    assert report["models"]["a"]["upper_bound_fraction"] == 1
    assert report["models"]["a"]["domain_scores"] == {"logistics": 100}
    pair = report["pairwise"][0]
    assert pair["index_difference"] == 0
    assert pair["n_clipping_hidden_differences"] == 1


def test_large_finite_costs_do_not_overflow_intermediate_subtraction():
    c = {**contract(), "weak_cost": 1e308, "strong_cost": 5e307}
    assert score_quality(sample(-1e308), c)["raw_score"] == pytest.approx(400)


def test_adding_model_does_not_rescale_existing_index():
    a = {**sample(700), "model": "a"}
    alone = summarize([a], [contract()], ["a"])
    both = summarize([a, {**sample(-100), "model": "b"}], [contract()], ["a", "b"])
    assert alone["models"]["a"] == both["models"]["a"]


def test_lower_native_cost_is_monotone_and_activity_is_not_rewarded():
    costs = [-100, 0, 500, 600, 900, 1000, 1100, 1500, 2000]
    scores = [score_quality(sample(cost), contract())["score"] for cost in costs]
    assert scores == sorted(scores, reverse=True)
    noisy = {**sample(), "tool_calls": 999, "plan_length": 9999}
    assert score_quality(noisy, contract()) == score_quality(sample(), contract())


def test_unrepresentable_json_integer_is_missing_evidence_not_crash():
    row = sample()
    row["actual_cost"] = 10**400
    assert score_quality(row, contract())["reason"] == "native_objective_missing"
