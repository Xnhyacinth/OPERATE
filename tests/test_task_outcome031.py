"""0.31 anchored case scores, aggregation, substitution and uncertainty."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from evaluation.task_outcome031 import (
    COMBINATION_V2,
    COMBINATION_V3,
    SCORE_BOUNDS,
    aggregate031,
    headline_weights,
    load_policy031,
    score_case031,
    substitute_domain_rows,
)
from evaluation.uncertainty031 import (
    bootstrap_uncertainty,
    compact_letter_display,
    domain_variance_share,
    influence_diagnostics,
    retest_agreement,
)

ROOT = Path(__file__).resolve().parents[1]
RULES = dict(h_min=0.02, delta=1.0)


def anchor(**kw):
    base = dict(
        C_wait=100.0,
        C_ref=60.0,
        cost_scale=100.0,
        cost_axis_informative=True,
        F_wait=50.0,
        service_applicable=True,
        service_axis_informative=True,
        headline_eligible=True,
        domain="d",
        task_family="t",
    )
    return {**base, **kw}


def score(cost, fulfillment, hard=False, **kw):
    return score_case031(cost, fulfillment, hard, anchor(**kw), **RULES)


def test_do_nothing_scores_zero_and_reference_closes_full_cost_gap():
    assert score(100.0, 50.0)["S"] == 0.0
    reference = score(60.0, 100.0)
    assert reference["g"] == pytest.approx(100.0)
    assert reference["S"] == pytest.approx(100.0)
    assert reference["binding_axis"] == "cost"


def test_cheap_trajectory_that_abandons_service_scores_below_do_nothing():
    result = score(10.0, 20.0)
    assert result["g"] > 100
    assert result["f"] == pytest.approx(-60.0)
    assert result["S"] == pytest.approx(-60.0)
    assert result["binding_axis"] == "service"


def test_full_service_never_caps_superior_cost():
    assert score(40.0, 100.0)["S"] == pytest.approx(150.0)
    assert score(52.0, 99.5)["S"] == pytest.approx(120.0)


def test_scores_are_clipped_to_declared_bounds():
    assert score(-1000.0, 100.0)["S"] == SCORE_BOUNDS[1]
    assert score(10_000.0, 100.0)["S"] == SCORE_BOUNDS[0]


def test_verified_hard_failure_is_floor_and_unknown_state_stays_unknown():
    hard = score(10.0, 100.0, hard=True)
    assert (hard["S"], hard["binding_axis"]) == (-100.0, "hard_failure")
    for unknown in (
        score(10.0, 100.0, hard=None),
        score(None, 100.0),
        score(10.0, None),
    ):
        assert unknown["determined"] is False
        assert unknown["S"] is None
        assert unknown["missing_bounds"] == list(SCORE_BOUNDS)


def test_uninformative_cost_axis_only_guards_against_regression():
    flat = dict(cost_axis_informative=False, C_ref=100.0)
    assert score(101.0, 75.0, **flat)["binding_axis"] == "service"
    regression = score(130.0, 100.0, **flat)
    assert regression["g"] == pytest.approx(-30.0)
    assert regression["S"] == pytest.approx(-30.0)


def test_uninformative_service_axis_only_guards_against_regression():
    full = dict(F_wait=100.0, service_axis_informative=False)
    assert score(80.0, 99.5, **full)["binding_axis"] == "cost"
    assert score(80.0, 90.0, **full)["f"] == pytest.approx(-10.0)


def test_economic_only_case_uses_cost_axis():
    result = score(
        80.0,
        None,
        service_applicable=False,
        F_wait=None,
        service_axis_informative=False,
    )
    assert result["S"] == pytest.approx(50.0)
    assert result["reason"] == "reference_anchored_cost_only"


def score_v2(cost, fulfillment, hard=False, **kw):
    return score_case031(
        cost, fulfillment, hard, anchor(**kw), combination=COMBINATION_V2, **RULES
    )


def test_v1_jumps_at_service_threshold_and_v2_is_continuous():
    below, at = score(30.0, 98.999), score(30.0, 99.0)
    assert at["S"] - below["S"] > 50
    v2_below, v2_at = score_v2(30.0, 98.999), score_v2(30.0, 99.0)
    assert abs(v2_at["S"] - v2_below["S"]) < 0.05
    assert score_v2(30.0, 100.0)["S"] == pytest.approx(150.0)


def test_v2_cost_savings_cannot_buy_back_unserved_obligations():
    assert score_v2(10.0, 20.0)["S"] == pytest.approx(-60.0)
    abandoning = score_v2(40.0, 90.0)
    assert abandoning["S"] == pytest.approx(80.0)
    assert abandoning["binding_axis"] == "service"
    flat = dict(F_wait=100.0, service_axis_informative=False)
    assert score_v2(10.0, 80.0, **flat)["S"] == pytest.approx(-20.0)


@pytest.mark.parametrize(
    "flags",
    [
        {},
        dict(cost_axis_informative=False, C_ref=100.0),
        dict(F_wait=100.0, service_axis_informative=False),
        dict(service_applicable=False, F_wait=None, service_axis_informative=False),
    ],
)
def test_v2_is_continuous_and_monotone_in_cost_and_service(flags):
    service = (
        (lambda F: None) if flags.get("service_applicable") is False else (lambda F: F)
    )
    costs = [20.0 + 0.01 * i for i in range(12001)]
    for F in (20.0, 50.0, 97.0, 98.5, 99.0, 99.5, 100.0):
        values = [score_v2(c, service(F), **flags)["S_unclipped"] for c in costs]
        assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))
        assert max(abs(b - a) for a, b in zip(values, values[1:])) < 0.5
    if flags.get("service_applicable") is False:
        return
    for cost in (120.0, 101.0, 80.0, 60.0, 40.0):
        values = [
            score_v2(cost, 0.01 * i, **flags)["S_unclipped"] for i in range(10001)
        ]
        assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))
        assert max(abs(b - a) for a, b in zip(values, values[1:])) < 0.5


def test_v2_keeps_known_group_endpoints():
    assert score_v2(100.0, 50.0)["S"] == 0.0
    assert score_v2(60.0, 100.0)["S"] == pytest.approx(100.0)
    assert score_v2(10.0, 100.0, hard=True)["S"] == -100.0


def score_v3(cost, fulfillment, hard=False, **kw):
    return score_case031(
        cost, fulfillment, hard, anchor(**kw), combination=COMBINATION_V3, **RULES
    )


def test_v3_incomplete_service_cannot_reach_reference_attainment():
    # Reproduces the HY4 case with 96.774% fulfillment and g=123.433.
    flags = dict(F_wait=0.0)
    assert score_v2(50.62682652802351, 96.7741935483871, **flags)["S"] > 100
    result = score_v3(50.62682652802351, 96.7741935483871, **flags)
    assert result["S"] == pytest.approx(96.7741935483871)
    assert result["g"] > 100  # Excess savings stay available as a diagnostic.
    for F in (96, 99, 99.5, 99.99999):
        assert score_v3(0.0, F)["S"] < 100
        assert score_v3(0.0, F, F_wait=100.0, service_axis_informative=False)["S"] < 100
    assert score_v3(0.0, 100.0)["S"] == 100


@pytest.mark.parametrize(
    "flags",
    [
        {},
        dict(cost_axis_informative=False, C_ref=100.0),
        dict(F_wait=100.0, service_axis_informative=False),
        dict(F_wait=99.5, service_axis_informative=False),
        dict(service_applicable=False, F_wait=None, service_axis_informative=False),
    ],
)
def test_v3_is_monotone_and_continuous_including_full_service(flags):
    def evaluate(C, F):
        return score_v3(
            C, None if flags.get("service_applicable") is False else F, **flags
        )["S"]

    for F in (20.0, 50.0, 97.0, 99.0, 99.99999, 100.0):
        values = [evaluate(20.0 + 0.1 * i, F) for i in range(1201)]
        assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))
    for C in (20.0, 60.0, 101.0, 110.0, 140.0):
        values = [evaluate(C, i * 0.01) for i in range(10001)]
        assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))
        assert max(abs(b - a) for a, b in zip(values, values[1:])) < 0.5
        assert abs(evaluate(C, 100) - evaluate(C, 99.99999)) < 0.001


def test_v3_bounds_zero_anchor_and_hard_failure():
    assert score_v3(100.0, 50.0)["S"] == 0
    assert score_v3(60.0, 100.0)["S"] == 100
    assert score_v3(10.0, 100.0, hard=True)["S"] == -100
    assert score_v3(None, 100.0)["missing_bounds"] == [-100, 100]
    assert score_v3(-1000.0, None, service_applicable=False)["S"] == 100


def anchors_for(domains):
    return {
        (f"s{i}", 1): dict(domain=d, headline_eligible=eligible)
        for i, (d, eligible) in enumerate(domains)
    }


def test_ineligible_cases_are_dropped_and_mass_renormalized_globally():
    anchors = anchors_for([("a", True), ("a", True), ("b", True), ("b", False)])
    nominal = {("s0", 1): 0.25, ("s1", 1): 0.25, ("s2", 1): 0.25, ("s3", 1): 0.25}
    weights = headline_weights(nominal, anchors)
    assert weights == {
        ("s0", 1): pytest.approx(1 / 3),
        ("s1", 1): pytest.approx(1 / 3),
        ("s2", 1): pytest.approx(1 / 3),
    }
    excluded = headline_weights(nominal, anchors, exclude_domains=("a",))
    assert excluded == {("s2", 1): 1.0}


def case(S, model, key):
    return dict(
        S=S,
        determined=S is not None,
        missing_bounds=[S, S] if S is not None else list(SCORE_BOUNDS),
        hard_failure=False,
        binding_axis="cost",
        reason="x",
        model=model,
    )


def test_incomplete_models_keep_bounds_and_no_rank_and_ties_share_rank():
    anchors = anchors_for([("a", True), ("b", True)])
    weights = {("s0", 1): 0.5, ("s1", 1): 0.5}
    cases = {
        ("m1", "s0", 1): case(40.0, "m1", ("s0", 1)),
        ("m1", "s1", 1): case(60.0, "m1", ("s1", 1)),
        ("m2", "s0", 1): case(60.0, "m2", ("s0", 1)),
        ("m2", "s1", 1): case(40.0, "m2", ("s1", 1)),
        ("m3", "s0", 1): case(90.0, "m3", ("s0", 1)),
    }
    result = {
        r["model"]: r
        for r in aggregate031(cases, weights, ["m1", "m2", "m3"], anchors)["models"]
    }
    assert result["m1"]["primary_rank"] == result["m2"]["primary_rank"] == 1
    assert result["m3"]["primary_rank"] is None
    assert result["m3"]["missing_measurement_bounds"] == pytest.approx([-5.0, 120.0])
    assert result["m3"]["domains"]["a"]["score"] == 90.0
    assert result["m3"]["domains"]["b"]["score"] is None


def base_report():
    contracts = [
        dict(scenario_signature="drive", seed=1, domain="autonomous_driving"),
        dict(scenario_signature="grid", seed=1, domain="power_grid"),
    ]
    rows = [
        dict(
            model=m,
            scenario_signature=k,
            seed=1,
            C=1.0,
            F=50.0,
            Q=40.0,
            outcome030=dict(evidence_qualified=True, hard_failure=False),
        )
        for m in ("a", "b")
        for k in ("drive", "grid")
    ]
    return dict(
        evaluation_version="0.30.0",
        source_contracts=dict(contracts=contracts),
        graded_episodes=rows,
    )


def supplement(models, value=9.0):
    report = base_report()
    report["graded_episodes"] = [
        dict(r, model=models[r["model"]], C=value)
        for r in report["graded_episodes"]
        if r["model"] in models
    ]
    return report


def test_mapped_retest_rows_replace_domain_and_unmapped_models_become_unknown():
    result = substitute_domain_rows(
        base_report(),
        [
            dict(
                report=supplement({"a": "a-retest"}),
                sha256="x",
                label="retest",
                models={"a": "a-retest"},
            )
        ],
        domain="autonomous_driving",
    )
    rows = {(r["model"], r["scenario_signature"]): r for r in result["graded_episodes"]}
    assert rows[("a", "drive")]["C"] == 9.0
    assert rows[("a", "drive")]["substitution"]["supplement_model"] == "a-retest"
    assert rows[("a", "grid")]["C"] == 1.0
    assert rows[("b", "drive")]["outcome030"]["evidence_qualified"] is False
    assert rows[("b", "drive")]["substitution"]["superseded_Q030"] == 40.0


@pytest.mark.parametrize("damage", ["twice", "undeclared", "contracts"])
def test_substitution_rejects_ambiguous_or_foreign_supplements(damage):
    good = dict(report=supplement({"a": "a"}), sha256="x", label="r", models={"a": "a"})
    supplements = [good, deepcopy(good)] if damage == "twice" else [good]
    if damage == "undeclared":
        good["models"] = {"zzz": "a"}
    if damage == "contracts":
        good["report"]["source_contracts"]["contracts"][0]["domain"] = "other"
    with pytest.raises(ValueError):
        substitute_domain_rows(base_report(), supplements, domain="autonomous_driving")


def synthetic_report(scores):
    keys = [("c%d" % i, 1) for i in range(len(next(iter(scores.values()))))]
    weights = [
        dict(scenario_signature=k[0], seed=k[1], weight=1 / len(keys)) for k in keys
    ]
    cases = [
        dict(
            model=m,
            scenario_signature=k[0],
            seed=k[1],
            domain="a" if i % 2 else "b",
            S=s,
            determined=True,
        )
        for m, values in scores.items()
        for i, (k, s) in enumerate(zip(keys, values))
    ]
    models = [
        dict(model=m, complete=True, primary_score=sum(v) / len(v))
        for m, v in scores.items()
    ]
    return dict(headline_weights=weights, cases=cases, models=models)


def test_compact_letters_never_separate_an_unresolved_pair():
    letters = compact_letter_display(["A", "B", "C"], {frozenset(("A", "C"))})
    assert letters == {"A": "a", "B": "ab", "C": "b"}
    assert compact_letter_display(["A", "B"], set()) == {"A": "a", "B": "a"}


def test_cluster_bootstrap_moves_same_source_cases_together():
    report = synthetic_report({"x": [80, 80, 20, 20], "y": [60, 60, 40, 40]})
    clusters = {("c0", 1): "s1", ("c2", 1): "s1", ("c1", 1): "s2", ("c3", 1): "s2"}
    result = bootstrap_uncertainty(report, replicates=200, seed=1, clusters=clusters)
    assert result["method"].startswith("domain_stratified_source_cluster")
    assert result["resampling_units_per_domain"] == {"a": 1, "b": 1}
    assert result["coarse_domains"] == ["a", "b"]


@pytest.mark.parametrize("complete", [False, True])
def test_report_renders_when_fewer_than_two_models_are_complete(complete):
    from scripts.evaluate_anchored_outcomes import render

    report = synthetic_report({"only": [70.0, 80.0]})
    report["models"][0].update(
        complete=complete,
        primary_score=75.0 if complete else None,
        primary_rank=1 if complete else None,
        n_scored=2 if complete else 1,
        n_expected=2,
        missing_measurement_bounds=[75.0, 75.0] if complete else [-15.0, 85.0],
        hard_failure_cases=0,
        Q030=None,
        domains={},
    )
    report.update(
        evaluation_version="0.31.2",
        anchor_semantics=dict(
            reference_joint_score_100_cases=1,
            eligible_cases=2,
            privileged_upper_anchor_cases=1,
        ),
        uncertainty=bootstrap_uncertainty(report),
        sensitivity={"exclude_domain:autonomous_driving": {"only": None}},
        domain_variance_share={"shares": {}},
    )
    text = render(report)
    assert "0/0 pairs resolved" in text
    assert "only" in text
    assert report["uncertainty"]["replicates"] == 0
    assert report["uncertainty"]["models"] == []


def test_bootstrap_is_deterministic_and_separates_clear_gaps_into_groups():
    report = synthetic_report(
        {"top": [90, 95, 88, 92], "mid": [50, 55, 52, 48], "near": [56, 49, 47, 52]}
    )
    first = bootstrap_uncertainty(report, replicates=300, seed=3)
    assert first["letters"] == {"top": "a", "mid": "b", "near": "b"}
    assert first["resolved_pairs"] == 2
    assert first == bootstrap_uncertainty(report, replicates=300, seed=3)
    assert first["heuristic_groups"] == [["top"], ["mid", "near"]]
    assert first["pairwise"]["top"]["mid"] == 1.0


def test_influence_diagnostics_flag_the_case_that_drives_the_order():
    report = synthetic_report(
        {"x": [10, 50, 50, 50], "y": [90, 48, 48, 48], "z": [50, 49, 49, 49]}
    )
    for case in report["cases"]:
        case.update(task_family="fam", binding_axis="cost", S_unclipped=case["S"])
    report["rules"] = dict(score_bounds=[-100.0, 150.0])
    result = influence_diagnostics(report, top=2)
    worst = result["leave_one_case_out"]["most_influential"][0]
    assert worst["scenario_signature"] == "c0"
    assert worst["max_rank_shift"] >= 1
    assert result["case_variance_share"]["top"][0]["scenario_signature"] == "c0"
    assert sum(result["family_variance_share"].values()) == pytest.approx(1.0)


def test_domain_variance_shares_sum_to_one():
    report = synthetic_report(
        {"x": [10, 90, 20, 80], "y": [30, 60, 35, 65], "z": [50, 20, 55, 25]}
    )
    shares = domain_variance_share(report)["shares"]
    assert sum(shares.values()) == pytest.approx(1.0)


def test_test_retest_pairs_only_cases_measured_in_both_executions():
    old = [
        dict(model="m", scenario_signature=s, seed=1, domain="d", S=v, determined=True)
        for s, v in (("a", 0.0), ("b", 50.0), ("c", 10.0))
    ]
    new = [dict(o, S=o["S"] + 5) for o in old[:2]] + [
        dict(old[2], determined=False, S=None)
    ]
    result = retest_agreement(old, new, domain="d")
    assert result["paired_cases"] == 2


def test_committed_policy_and_anchors_verify_and_tamper_fails(tmp_path):
    policy = load_policy031(ROOT / "evaluation/policies/lite141_031.json", root=ROOT)
    assert policy["reference_anchors"]["counts"]["cases"] == 141
    assert policy["weight_manifest"]["weight_policy"] == "equal_domain_then_case"
    raw = json.loads((ROOT / "evaluation/policies/lite141_031.json").read_bytes())
    raw["rules"]["h_min"] = 0.5
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="hash"):
        load_policy031(path, root=ROOT)


def test_v3_policy_version_cannot_silently_select_historical_rules(tmp_path):
    from evaluation.trajectory_outcome027 import canonical_digest

    original = ROOT / "evaluation/policies/lite141_0312.json"
    policy = load_policy031(original, root=ROOT)
    assert policy["rules"]["score_bounds"] == [-100, 100]
    raw = json.loads(original.read_bytes())
    raw["rules"]["combination"] = COMBINATION_V2
    raw.pop("policy_sha256")
    raw["policy_sha256"] = canonical_digest(raw)
    path = tmp_path / "wrong_version.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="version_rules"):
        load_policy031(path, root=ROOT)


def test_raw_authentication_does_not_certify_cross_execution_formal_run():
    from scripts.evaluate_anchored_outcomes import certification

    inputs = dict(
        base=dict(raw_reauthenticated=True),
        domain_substitution=dict(
            supplements=[dict(binding=dict(raw_reauthenticated=True))]
        ),
        case_substitution=None,
    )
    result = certification(
        dict(formal_run_certified=True, same_run_141_merge_certified=True), inputs
    )
    assert result["raw_reauthenticated"] is True
    assert result["formal_run_certified"] is False
    assert result["same_run_141_merge_certified"] is False
    inputs["domain_substitution"] = None
    result = certification({}, inputs)
    assert result["formal_run_certified"] is False
    assert "source_report_not_formal_run_certified" in result["reasons"]


@pytest.mark.parametrize("field,value", [("hard_failure_score", 0), ("ramp_width", 0)])
def test_v3_policy_rejects_unsupported_floor_or_zero_ramp(tmp_path, field, value):
    from evaluation.trajectory_outcome027 import canonical_digest

    raw = json.loads((ROOT / "evaluation/policies/lite141_0312.json").read_bytes())
    raw["rules"][field] = value
    raw.pop("policy_sha256")
    raw["policy_sha256"] = canonical_digest(raw)
    path = tmp_path / "invalid_rules.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="invalid_failure_or_ramp"):
        load_policy031(path, root=ROOT)


def test_specification_sensitivity_reports_large_rank_shift():
    from scripts.evaluate_anchored_outcomes import rank_sensitivity

    report = dict(
        models=[
            dict(model=m, primary_rank=i + 1, complete=True)
            for i, m in enumerate("abc")
        ],
        sensitivity={"weights:uniform_case": dict(a=10, b=30, c=20)},
    )
    rows = {r["model"]: r for r in rank_sensitivity(report)["models"]}
    assert rows["a"]["specification_rank_range"] == [1, 3]
    assert rows["a"]["max_rank_shift"] == 2
