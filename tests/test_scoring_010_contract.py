from __future__ import annotations

import math

import pytest

from evaluation.scorer import (
    DISCRIMINATIVE_CORE_DIMENSIONS,
    HEADLINE_SCORE_GROUPS,
    SCORING_VERSION,
    discriminative_core_total,
    score_weighted_equity_score,
)


def _full_discriminative_dimensions() -> list[dict]:
    return [
        {
            "name": name,
            "applicable": True,
            "calibrated_score": 50.0,
            "evidence_ids": [f"evidence:{name}"],
        }
        for name in DISCRIMINATIVE_CORE_DIMENSIONS
        if name != "task_completion"
    ]


def test_scoring_version_is_0170() -> None:
    assert SCORING_VERSION == "0.17.0"


def test_headline_groups_do_not_count_a_dimension_twice() -> None:
    for group in HEADLINE_SCORE_GROUPS.values():
        dimensions = group["dimensions"]
        assert len(dimensions) == len(set(dimensions))


def test_task_completion_fraction_is_scaled_to_points_before_weighting() -> None:
    dimensions = _full_discriminative_dimensions()

    incomplete = discriminative_core_total(
        dimensions,
        task_completion=0.0,
    )
    complete = discriminative_core_total(
        dimensions,
        task_completion=1.0,
    )

    assert incomplete["task_completion_raw"] == 0.0
    assert incomplete["task_completion_score"] == 0.0
    assert complete["task_completion_raw"] == 1.0
    assert complete["task_completion_score"] == 100.0
    assert complete["task_completion_input_unit"] == "fraction_0_1"
    assert complete["task_completion_score_unit"] == "points_0_100"
    assert incomplete["legacy_five_group_weight_denominator"] == 100.0
    assert incomplete["legacy_five_group_total"] == 35.0
    assert complete["legacy_five_group_total"] == 65.0
    assert complete["legacy_five_group_total"] - incomplete["legacy_five_group_total"] == 30.0
    assert incomplete["total_score"] == 50.0
    assert complete["total_score"] == 50.0
    assert incomplete["aggregation"] == "wait_relative_outcome_v1"
    assert incomplete["group_scores"]["task_completion"] == 0.0
    assert complete["group_scores"]["task_completion"] == 100.0
    assert DISCRIMINATIVE_CORE_DIMENSIONS["task_completion"] == 30.0


def test_partial_task_completion_is_scaled_once_to_points() -> None:
    result = discriminative_core_total(
        _full_discriminative_dimensions(),
        task_completion=0.5,
    )

    assert result["task_completion_raw"] == 0.5
    assert result["task_completion_score"] == 50.0
    assert result["total_score"] == 50.0
    assert result["legacy_five_group_total"] == 50.0


def test_missing_evidence_for_an_entire_group_fails_closed() -> None:
    dimensions = _full_discriminative_dimensions()
    for dimension in dimensions:
        if dimension["name"] in {
            "adaptive_replanning",
            "foresight_score",
        }:
            dimension["applicable"] = False
            dimension["evidence_ids"] = []

    result = discriminative_core_total(dimensions, task_completion=1.0)

    assert result["formal_score_eligible"] is True
    assert result["legacy_formal_score_eligible"] is False
    assert result["missing_groups"] == ["adaptation_and_foresight"]
    assert result["group_scores"]["adaptation_and_foresight"] == 0.0
    assert result["total_score"] == 50.0


def test_within_group_score_is_mean_of_supported_members_only() -> None:
    """Pin v0.11.0 within-group mean (not a silent formula change).

    Group *weights* stay fixed. A wholly missing group is ineligible.
    Inside a supported group, dropping a non-applicable dimension raises
    that group's mean.
    """
    keep = {
        "system_survival": 100.0,
        "safety_violation": 95.0,
        "economic_cost": 0.0,
        "counterfactual_prevention": 50.0,
        "adaptive_replanning": 20.0,
        "tool_use_efficiency": 40.0,
    }
    dimensions = []
    for name in DISCRIMINATIVE_CORE_DIMENSIONS:
        if name == "task_completion":
            continue
        if name in keep:
            dimensions.append(
                {
                    "name": name,
                    "applicable": True,
                    "calibrated_score": keep[name],
                    "evidence_ids": [f"evidence:{name}"],
                }
            )
        else:
            dimensions.append(
                {
                    "name": name,
                    "applicable": False,
                    "calibrated_score": 0.0,
                    "evidence_ids": [],
                }
            )

    result = discriminative_core_total(dimensions, task_completion=0.0)

    assert result["formal_score_eligible"] is True
    assert result["missing_groups"] == []
    assert result["group_support"]["safety_and_responsibility"] == [
        "system_survival",
        "safety_violation",
    ]
    assert result["group_scores"]["safety_and_responsibility"] == 97.5
    assert result["group_scores"]["adaptation_and_foresight"] == 20.0
    assert result["group_scores"]["system_outcome"] == 25.0
    assert result["group_scores"]["action_efficiency"] == 40.0
    # Primary is CF only (50), not the (econ 0 + CF 50) / 2 mix.
    assert result["wait_relative_source"] == "counterfactual_prevention"
    assert result["total_score"] == pytest.approx(50.0)
    assert result["legacy_five_group_total"] == pytest.approx(32.75)


def test_declared_zero_efficiency_dimensions_keep_action_denominator_fixed() -> None:
    dimensions = _full_discriminative_dimensions()
    for dimension in dimensions:
        if dimension["name"] == "information_efficiency":
            dimension["calibrated_score"] = 0.0
            dimension["evidence_ids"] = ["evidence:applicability-contract"]
        elif dimension["name"] == "tool_use_efficiency":
            dimension["calibrated_score"] = 100.0

    result = discriminative_core_total(dimensions, task_completion=1.0)

    assert result["group_support"]["action_efficiency"] == [
        "information_efficiency",
        "tool_use_efficiency",
    ]
    assert result["group_scores"]["action_efficiency"] == 50.0


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_five_group_headline_rejects_non_finite_dimension_scores(
    invalid: float,
) -> None:
    dimensions = _full_discriminative_dimensions()
    dimensions[0]["calibrated_score"] = invalid

    with pytest.raises(ValueError, match="finite"):
        discriminative_core_total(dimensions, task_completion=1.0)


@pytest.mark.parametrize("invalid", ["evidence", [None], [""], ["ok", 1]])
def test_five_group_headline_rejects_malformed_evidence_ids(
    invalid: object,
) -> None:
    dimensions = _full_discriminative_dimensions()
    dimensions[0]["evidence_ids"] = invalid

    with pytest.raises(ValueError, match="evidence_ids"):
        discriminative_core_total(dimensions, task_completion=1.0)


@pytest.mark.parametrize(
    "invalid",
    [-0.1, 1.1, 50, 100, math.nan, math.inf, "1"],
)
def test_task_completion_rejects_unknown_or_out_of_range_units(
    invalid: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        discriminative_core_total(
            _full_discriminative_dimensions(),
            task_completion=invalid,  # type: ignore[arg-type]
        )


def test_task_completion_missing_value_fails_closed() -> None:
    with pytest.raises((TypeError, ValueError)):
        discriminative_core_total(
            _full_discriminative_dimensions(),
            task_completion=None,  # type: ignore[arg-type]
        )


def test_weighted_equity_uses_unit_interval_criticality_scale() -> None:
    low = score_weighted_equity_score(
        {"residential-load": 10.0},
        {"residential-load": "residential"},
        evidence_ids=["equity-low"],
    ).to_dict()
    high = score_weighted_equity_score(
        {"hospital-load": 10.0},
        {"hospital-load": "hospital"},
        evidence_ids=["equity-high"],
    ).to_dict()
    mixed = score_weighted_equity_score(
        {"residential-load": 5.0, "hospital-load": 5.0},
        {
            "residential-load": "residential",
            "hospital-load": "hospital",
        },
        evidence_ids=["equity-mixed"],
    ).to_dict()
    zero = score_weighted_equity_score(
        {"residential-load": 0.0},
        {"residential-load": "residential"},
        evidence_ids=[],
    ).to_dict()

    assert low["raw_score"] == 75.0
    assert high["raw_score"] == 5.0
    assert mixed["raw_score"] == 40.0
    assert low["calibrated_score"] == 75.0
    assert high["calibrated_score"] == 5.0
    assert mixed["calibrated_score"] == 40.0
    assert zero["applicable"] is False
    assert low["raw_attainable_min"] == 0.0
    assert low["raw_attainable_max"] == 100.0
    assert low["criticality_min"] == 0.0
    assert low["criticality_max"] == 1.0
    assert low["formula_version"] == "entity_criticality_unit_interval_v2"
    assert low["evidence_ids"] == ["equity-low"]


def test_weighted_equity_unknown_class_uses_declared_default_without_clamp() -> None:
    score = score_weighted_equity_score(
        {"unknown-load": 10.0},
        {"unknown-load": "unknown"},
        evidence_ids=["equity-unknown"],
    ).to_dict()

    assert score["raw_score"] == 50.0
    assert score["calibrated_score"] == 50.0


def test_weighted_equity_uses_entity_criticality_when_declared() -> None:
    score = score_weighted_equity_score(
        {"priority-load": 5.0, "flexible-load": 5.0},
        {"priority-load": "commercial", "flexible-load": "commercial"},
        load_criticalities={"priority-load": 0.9, "flexible-load": 0.2},
        evidence_ids=["equity-native"],
    ).to_dict()

    assert score["raw_score"] == pytest.approx(45.0)
    assert score["calibrated_score"] == pytest.approx(45.0)
    assert score["criticality_min"] == 0.0
    assert score["criticality_max"] == 1.0
    assert score["formula_version"] == "entity_criticality_unit_interval_v2"


def test_fixed_headline_uses_emitted_weighted_dimensions_only() -> None:
    from core import DimensionScore
    from evaluation.scorer import _score_views

    views = _score_views(
        [
            DimensionScore(
                name="system_survival",
                calibrated_score=100.0,
                applicable=True,
            ),
            DimensionScore(
                name="ethical_quality",
                calibrated_score=100.0,
                applicable=False,
            ),
        ],
        difficulty_level="basic",
        adaptive_raw_total=100.0,
    )

    fixed = views["fixed_all_dimensions"]
    assert fixed["weight_denominator"] == 3.5
    assert fixed["raw_total"] == pytest.approx(42.8571428571)
    assert fixed["total_score"] == pytest.approx(fixed["raw_total"])
    assert views["adaptive_applicable"]["total_score"] == 100.0
