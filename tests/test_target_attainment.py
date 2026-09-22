from copy import deepcopy

import pytest

from evaluation.target_attainment import aggregate_target_attainment


def suite(signature="a", source="s", domain="d", backend="b"):
    return dict(
        scenario_signature=signature,
        seed=42,
        domain=domain,
        backend_kind=backend,
        source_denominator_key=source,
        physical_source_key=source,
    )


def row(signature="a", model="m", attained=True):
    return dict(
        model=model,
        scenario_signature=signature,
        seed=42,
        measurement=dict(
            score=100.0 if attained else 0.0,
            attained=attained,
            evidence_ids=["e"],
            reason="target",
        ),
        safety=dict(verified=True, hard_failure=False, evidence_ids=["s"]),
    )


def test_fixed_denominator_missing_and_absent_models_are_na():
    report = aggregate_target_attainment(
        [row()], [suite(), suite("b")], models=["m", "absent"]
    )
    assert not report["leaderboard"]
    assert {r["model"] for r in report["incomplete_models"]} == {"m", "absent"}
    for result in report["models"]:
        assert result["score"] is None
        assert result["micro_pass_rate"] is None
        assert result["n_expected"] == 2


def test_duplicate_is_not_averaged_or_selected():
    report = aggregate_target_attainment([row(), row(attained=False)], [suite()])
    assert report["models"][0]["score"] is None
    assert report["models"][0]["issues"][0]["reason"] == "duplicate_case"


@pytest.mark.parametrize(
    "mutation",
    [
        {
            "measurement": {
                "score": 100.0,
                "attained": True,
                "evidence_ids": [],
                "reason": "target",
            }
        },
        {
            "measurement": {
                "score": 50.0,
                "attained": True,
                "evidence_ids": ["e"],
                "reason": "target",
            }
        },
        {"safety": {"verified": False, "hard_failure": False, "evidence_ids": ["s"]}},
    ],
)
def test_invalid_measurement_or_unverified_safety_is_na(mutation):
    value = row()
    value.update(mutation)
    assert aggregate_target_attainment([value], [suite()])["models"][0]["score"] is None


def test_verified_hard_failure_cannot_hide_missing_target_measurement():
    value = row()
    value["measurement"] = None
    value["safety"]["hard_failure"] = True
    assert aggregate_target_attainment([value], [suite()])["models"][0]["score"] is None


def test_verified_hard_failure_gates_valid_measurement():
    value = row()
    value["safety"]["hard_failure"] = True
    result = aggregate_target_attainment([value], [suite()])["leaderboard"][0]
    assert result["score"] == 0
    assert result["n_passed"] == 0
    assert result["n_hard_failures"] == 1


def test_hierarchy_source_balance_and_model_independence():
    cases = [suite(), suite("b", "t"), suite("c", "u", "other")]
    values = [row(), row("b", attained=False), row("c", attained=False)]
    result = aggregate_target_attainment(values, cases)["leaderboard"][0]
    assert result["score"] == 25
    assert result["micro_pass_rate"] == pytest.approx(100 / 3)
    assert result["domain_scores"] == {"d": 50, "other": 0}
    cases.append(suite("variant"))
    values.append(row("variant"))
    assert aggregate_target_attainment(values, cases)["leaderboard"][0]["score"] == 25
    other = deepcopy(values)
    for value in other:
        value["model"] = "other"
    report = aggregate_target_attainment(values + other, cases)
    assert [r["rank"] for r in report["leaderboard"]] == [1, 1]
    assert all(r["score"] == 25 for r in report["leaderboard"])


def test_suite_duplicate_rejected_and_unexpected_case_invalidates_model():
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_target_attainment([row()], [suite(), suite()])
    report = aggregate_target_attainment([row(), row("unknown")], [suite()])
    assert report["models"][0]["score"] is None


@pytest.mark.parametrize("models", [[], ["m", "m"], ["other"]])
def test_declared_model_scope_cannot_expand_silently(models):
    with pytest.raises(ValueError):
        aggregate_target_attainment([row()], [suite()], models=models)
