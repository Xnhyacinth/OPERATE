from copy import deepcopy

import pytest

from evaluation.operate_quality import aggregate_operate_rows


def fixtures():
    specs = [
        dict(
            scenario_signature=k,
            seed=1,
            domain=d,
            backend_kind="b",
            source_denominator_key=k,
            physical_source_key=k,
            primary_dimension=dim,
        )
        for k, d, dim in [
            ("a", "d1", "R"),
            ("b", "d2", "R"),
            ("c", "d1", "A"),
            ("d", "d1", "L"),
        ]
    ]
    rows = [
        dict(
            model="m",
            scenario_signature=s["scenario_signature"],
            seed=1,
            measurement={"score": v, "evidence_ids": ["ev_" + s["scenario_signature"]]},
            safety={
                "verified": True,
                "hard_failure": False,
                "evidence_ids": ["safety"],
            },
        )
        for s, v in zip(specs, [80, 20, 40, 100])
    ]
    return specs, rows


def test_aggregation_uses_fixed_dimensions_then_source_backend_domain():
    specs, rows = fixtures()
    before = deepcopy(rows)
    result = aggregate_operate_rows(rows, specs, ["m"])
    model = result["models"]["m"]
    assert model["dimensions"]["R"]["score"] == 50
    assert model["index"] == 60
    assert model["complete"]
    assert rows == before


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "nan", "no_evidence", "missing_safety"]
)
def test_invalid_or_missing_measurement_cannot_shrink_denominator(fault):
    specs, rows = fixtures()
    if fault == "missing":
        rows = rows[:-1]
    elif fault == "duplicate":
        rows.append(deepcopy(rows[-1]))
    elif fault == "nan":
        rows[-1]["measurement"]["score"] = float("nan")
    elif fault == "no_evidence":
        rows[-1]["measurement"]["evidence_ids"] = []
    else:
        rows[-1]["safety"]["verified"] = False
    report = aggregate_operate_rows(rows, specs, ["m"])
    assert report["models"]["m"]["index"] is None
    assert report["models"]["m"]["dimensions"]["L"]["n_expected"] == 1
    assert report["ranking"] == []


def test_unmeasured_whole_dimension_is_not_redistributed():
    specs, rows = fixtures()
    result = aggregate_operate_rows(rows[:2], specs[:2], ["m"])
    assert result["models"]["m"]["index"] is None
    assert (
        result["models"]["m"]["dimensions"]["A"]["reason"]
        == "dimension_not_in_declared_suite"
    )


def test_hard_failure_gates_only_its_case_and_is_reported_separately():
    specs, rows = fixtures()
    rows[0]["safety"]["hard_failure"] = True
    result = aggregate_operate_rows(rows, specs, ["m"])["models"]["m"]
    assert result["dimensions"]["R"]["score"] == 10
    assert result["index"] == 40
    assert result["safety"]["hard_failures"] == 1
    assert result["safety"]["n_measured"] == 4


def test_same_source_duplicate_variants_do_not_increase_domain_weight():
    specs, rows = fixtures()
    extra = {**specs[0], "scenario_signature": "a2"}
    row = {**rows[0], "scenario_signature": "a2"}
    assert (
        aggregate_operate_rows(rows + [row], specs + [extra], ["m"])["models"]["m"][
            "index"
        ]
        == 60
    )


def test_outside_model_case_and_ambiguous_suite_rejected():
    specs, rows = fixtures()
    for changed_specs, changed_rows in [
        (specs + specs[:1], rows),
        (specs, [{**rows[0], "model": "other"}]),
        (specs, [{**rows[0], "seed": 2}]),
    ]:
        with pytest.raises(ValueError):
            aggregate_operate_rows(changed_rows, changed_specs, ["m"])


def test_dimension_weights_are_validated_not_silently_renormalized():
    specs, rows = fixtures()
    for weights in [
        {"R": 1},
        {"R": 0.5, "A": 0.4, "L": 0.4},
        {"R": 1, "A": -1, "L": 1},
        {"R": True, "A": 0, "L": 0},
    ]:
        with pytest.raises(ValueError):
            aggregate_operate_rows(rows, specs, ["m"], weights=weights)


def test_joint_cluster_bootstrap_pairs_models_and_retains_estimand():
    specs, rows = fixtures()
    other = [{**r, "model": "n"} for r in rows]
    result = aggregate_operate_rows(rows + other, specs, ["m", "n"], bootstrap=10)
    assert result["models"]["m"]["ci"]["lo"] == 60
    assert result["models"]["m"]["ci"]["hi"] == 60
    assert result["pairwise"][0]["mean_difference"] == 0
    assert result["pairwise"][0]["ci_lo"] == 0
    assert result["pairwise"][0]["ci_hi"] == 0
    assert result["models"]["m"]["ci"]["limited_cluster_support"]


def test_huge_numeric_input_is_unavailable_not_an_overflow():
    specs, rows = fixtures()
    rows[0]["measurement"]["score"] = 10**1000
    assert aggregate_operate_rows(rows, specs, ["m"])["models"]["m"]["index"] is None


def test_ci_does_not_split_one_effective_source_into_different_physical_units():
    specs, rows = fixtures()
    specs.append(
        {**specs[0], "scenario_signature": "a2", "physical_source_key": "another"}
    )
    rows.append({**rows[0], "scenario_signature": "a2"})
    result = aggregate_operate_rows(rows, specs, ["m"], bootstrap=10)
    assert result["models"]["m"]["index"] == 60
    assert result["inference_reason"] == "effective_source_spans_physical_clusters"


def test_same_task_has_distinct_fixed_measurements_without_tripling_safety():
    specs, rows = fixtures()
    specs = [{**specs[0], "primary_dimension": d} for d in ("R", "A", "L")]
    rows = [
        {
            **rows[0],
            "primary_dimension": d,
            "measurement": {"score": v, "evidence_ids": [d]},
        }
        for d, v in zip(("R", "A", "L"), (80, 40, 100))
    ]
    result = aggregate_operate_rows(rows, specs, ["m"], bootstrap=10)
    assert result["models"]["m"]["index"] == 75
    assert result["models"]["m"]["n_expected"] == 3
    assert result["models"]["m"]["safety"]["n_expected"] == 1
    assert result["models"]["m"]["safety"]["n_measured"] == 1
    assert result["models"]["m"]["ci"]["n_physical_clusters"] == 1
    with pytest.raises(ValueError, match="dimension"):
        aggregate_operate_rows(
            [{k: v for k, v in r.items() if k != "primary_dimension"} for r in rows],
            specs,
            ["m"],
        )


def test_multi_axis_case_cannot_disagree_on_safety_or_lineage():
    specs, rows = fixtures()
    specs = [{**specs[0], "primary_dimension": d} for d in "RAL"]
    rows = [{**rows[0], "primary_dimension": d} for d in "RAL"]
    rows[1] = {**rows[1], "safety": {**rows[1]["safety"], "hard_failure": True}}
    with pytest.raises(ValueError, match="safety"):
        aggregate_operate_rows(rows, specs, ["m"])
    rows[1]["safety"]["hard_failure"] = False
    specs[1] = {**specs[1], "domain": "different"}
    with pytest.raises(ValueError, match="lineage"):
        aggregate_operate_rows(rows, specs, ["m"])


def test_domain_balanced_index_uses_frozen_structural_axes_not_available_scores():
    specs, rows = fixtures()
    report = aggregate_operate_rows(
        rows, specs, ["m"], aggregation_mode="domain_balanced", bootstrap=10
    )
    # d1=(.5*80+.25*40+.25*100)=75; d2 only R=20; equal domains.
    assert report["models"]["m"]["index"] == 47.5
    assert report["models"]["m"]["axis_first_diagnostic_index"] == 60
    assert report["domain_dimension_weights"] == {
        "d1": {"R": 0.5, "A": 0.25, "L": 0.25},
        "d2": {"R": 1.0},
    }
    assert report["effective_dimension_weights"] == {"R": 0.75, "A": 0.125, "L": 0.125}
    assert report["models"]["m"]["ci"]["lo"] == 47.5
    missing = aggregate_operate_rows(
        rows[:-1], specs, ["m"], aggregation_mode="domain_balanced"
    )
    assert missing["models"]["m"]["index"] is None
    assert missing["domain_dimension_weights"] == report["domain_dimension_weights"]
