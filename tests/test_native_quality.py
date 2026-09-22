from copy import deepcopy

import pytest

from evaluation.native_quality import calibrate_row, score_native_quality


def measurement(cost):
    return {
        "applicable": True,
        "objective_id": "job.cost.v1",
        "unit": "native",
        "actual_cost": cost,
        "feasible": True,
        "hard_failure": False,
        "task_success": True,
        "evidence_ids": ["cost", "constraint"],
    }


def references():
    return {
        p: [measurement(c), measurement(c)]
        for p, c in zip(
            ("wait_only", "greedy_heuristic", "oracle_offline", "random"),
            (1000, 700, 500, 900),
        )
    }


def test_fixed_reference_set_calibration_is_model_independent():
    result = calibrate_row(references())
    assert result["status"] == "ready"
    assert (result["weak_cost"], result["strong_cost"]) == (1000, 500)
    assert score_native_quality(measurement(600), result)["score"] == pytest.approx(
        100 * 0.8 / 1.8
    )
    assert score_native_quality(measurement(1250), result)["score"] == pytest.approx(
        -100 / 3
    )


def test_dyna_incomplete_references_cannot_define_quality_scale():
    refs = references()
    for value in refs["wait_only"]:
        value["feasible"] = False
    result = calibrate_row(refs)
    assert result["weak_cost"] == 900
    m = measurement(10)
    m["feasible"] = False
    assert score_native_quality(m, result)["score"] == -100


def test_degenerate_unstable_or_incomplete_calibration_is_unavailable():
    refs = references()
    refs["greedy_heuristic"][1]["actual_cost"] += 10
    assert calibrate_row(refs)["status"] != "ready"
    refs = references()
    refs.pop("random")
    assert calibrate_row(refs)["status"] != "ready"
    refs = {p: [measurement(7), measurement(7)] for p in references()}
    assert calibrate_row(refs)["status"] != "ready"


def test_measurement_contract_and_evidence_required_without_zero_fill():
    anchor = calibrate_row(references())
    for change in (
        {"applicable": False},
        {"evidence_ids": []},
        {"objective_id": "other"},
        {"feasible": None},
        {"actual_cost": float("nan")},
    ):
        assert (
            score_native_quality({**measurement(600), **change}, anchor)["score"]
            is None
        )


def test_scoring_preserves_raw_tail_and_never_mutates_inputs():
    m = measurement(-100)
    before = deepcopy(m)
    result = score_native_quality(m, calibrate_row(references()))
    assert result["score"] == pytest.approx(68.75)
    assert result["raw_score"] == pytest.approx(220)
    assert m == before


def test_native_index_has_fixed_denominator_and_affine_correct_intervals():
    from evaluation.native_quality import aggregate_native_rows

    suite = [
        {
            "scenario_signature": "a",
            "seed": 1,
            "domain": "d",
            "backend_kind": "b",
            "source_denominator_key": "s",
            "physical_source_key": "p",
        }
    ]
    rows = [
        {
            "model": "m",
            "scenario_signature": "a",
            "seed": 1,
            "native_quality": {
                "score": -50,
                "task_success": False,
                "constraint_failed": False,
                "raw_score": -50,
            },
        }
    ]
    result = aggregate_native_rows(rows, suite, ["m"], bootstrap=4)
    assert result["models"]["m"]["index"] == -50
    assert result["models"]["m"]["ci"] == {"lo": -50, "hi": -50}
    assert aggregate_native_rows([], suite, ["m"])["models"]["m"]["index"] is None
    assert aggregate_native_rows(rows * 2, suite, ["m"])["models"]["m"]["index"] is None


def test_main_quality_does_not_flatten_policies_better_than_reference():
    anchor = calibrate_row(references())
    assert (
        score_native_quality(measurement(400), anchor)["score"]
        < score_native_quality(measurement(200), anchor)["score"]
    )


def test_forged_nonfinite_anchor_gap_is_rejected():
    anchor = {**calibrate_row(references()), "weak_cost": 1e308, "strong_cost": -1e308}
    assert score_native_quality(measurement(1), anchor)["score"] is None


def test_native_quality_monotonic_and_strong_reference_is_fifty():
    anchor = calibrate_row(references())
    assert score_native_quality(measurement(500), anchor)["score"] == 50
    costs = [-10000, -1000, 0, 250, 500, 750, 1000, 1500, 10000]
    scores = [score_native_quality(measurement(c), anchor)["score"] for c in costs]
    assert all(a > b for a, b in zip(scores, scores[1:]))


def test_makespan_scale_does_not_amplify_nearly_equal_dispatch_references():
    refs = references()
    for policy, pair in refs.items():
        for item in pair:
            item.update(
                objective_id="dynasched_flexible_job_shop.makespan_unfinished_penalty.v1",
                unit="native_time_plus_unfinished_penalty",
                actual_cost=594.2489266666666
                if policy == "greedy_heuristic"
                else 594.6324730952381,
            )
    anchor = calibrate_row(refs)
    m = {**refs["greedy_heuristic"][0], "actual_cost": 594.2489266666666 * 0.99}
    result = score_native_quality(m, anchor)
    assert 0 < result["score"] < 2
    assert anchor["normalization_scale"] == pytest.approx(594.2489266666666)
    assert result["raw_score"] > 100


def test_identical_feasible_makespans_still_have_a_natural_reference_scale():
    refs = references()
    for pair in refs.values():
        for item in pair:
            item.update(
                objective_id="dynasched_flexible_job_shop.makespan_unfinished_penalty.v1",
                actual_cost=500,
            )
    anchor = calibrate_row(refs)
    assert anchor["status"] == "ready"
    m = {**refs["greedy_heuristic"][0], "actual_cost": 550}
    result = score_native_quality(m, anchor)
    assert result["score"] == pytest.approx(-100 / 11)
    assert result["raw_score"] is None
    assert result["above_strong_reference"] is False
