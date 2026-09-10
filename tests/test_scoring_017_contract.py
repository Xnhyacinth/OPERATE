from __future__ import annotations

import pytest

from evaluation.scorer import (
    DISCRIMINATIVE_CORE_DIMENSIONS,
    PRIMARY_HEADLINE_AGGREGATION,
    QUALIFICATION_SCORING_VERSION,
    SCORING_VERSION,
    discriminative_core_total,
    reanchor_economic_cost,
)


def _dimensions(**overrides: float) -> list[dict]:
    rows = []
    for name in DISCRIMINATIVE_CORE_DIMENSIONS:
        if name == "task_completion":
            continue
        rows.append(
            {
                "name": name,
                "applicable": True,
                "calibrated_score": float(overrides.get(name, 40.0)),
                "evidence_ids": [f"evidence:{name}"],
            }
        )
    return rows


def _drop(name: str, dimensions: list[dict]) -> list[dict]:
    for dimension in dimensions:
        if dimension["name"] == name:
            dimension["applicable"] = False
            dimension["evidence_ids"] = []
            dimension["calibrated_score"] = 0.0
    return dimensions


def test_scoring_version_and_aggregation() -> None:
    assert QUALIFICATION_SCORING_VERSION == "0.15.0"
    assert SCORING_VERSION == "0.17.0"
    assert PRIMARY_HEADLINE_AGGREGATION == "wait_relative_outcome_v1"
    result = discriminative_core_total(_dimensions(), task_completion=1.0)
    assert result["aggregation"] == "wait_relative_outcome_v1"
    assert result["wait_relative_source"] == "counterfactual_prevention"


def test_economic_cost_reanchor_maps_wait_parity_to_zero() -> None:
    assert reanchor_economic_cost(50.0) == 0.0
    assert reanchor_economic_cost(100.0) == 100.0
    assert reanchor_economic_cost(0.0) == 0.0
    assert reanchor_economic_cost(75.0) == 50.0


def test_primary_prefers_counterfactual_and_ignores_economic_anchor() -> None:
    result = discriminative_core_total(
        _dimensions(counterfactual_prevention=20.0, economic_cost=100.0, optimality_gap=90.0),
        task_completion=1.0,
        completion_contract_kind="mitigation",
    )
    assert result["total_score"] == 20.0
    assert result["wait_relative_source"] == "counterfactual_prevention"
    assert result["group_scores"]["system_outcome"] == pytest.approx(70.0)


def test_wait_parity_is_zero_when_only_economic_cost_applies() -> None:
    result = discriminative_core_total(
        _drop("counterfactual_prevention", _dimensions(economic_cost=50.0)),
        task_completion=1.0,
        completion_contract_kind="mitigation",
    )
    assert result["wait_relative_source"] == "economic_cost_reanchored"
    assert result["total_score"] == 0.0
    assert result["survivor_outcome"] == 0.0


def test_reanchored_economic_cost_beats_wait() -> None:
    result = discriminative_core_total(
        _drop("counterfactual_prevention", _dimensions(economic_cost=75.0)),
        task_completion=1.0,
    )
    assert result["wait_relative_source"] == "economic_cost_reanchored"
    assert result["total_score"] == 50.0


def test_optimality_gap_does_not_pad_wait_relative_primary() -> None:
    result = discriminative_core_total(
        _dimensions(counterfactual_prevention=0.0, optimality_gap=80.0, economic_cost=50.0),
        task_completion=1.0,
    )
    assert result["total_score"] == 0.0
    assert result["group_scores"]["system_outcome"] == pytest.approx(130.0 / 3.0)


def test_floor_zeros_primary_but_keeps_survivor_outcome() -> None:
    dimensions = _dimensions(counterfactual_prevention=55.0)
    for dimension in dimensions:
        if dimension["name"] == "system_survival":
            dimension["floor_violation"] = True
    result = discriminative_core_total(
        dimensions,
        task_completion=1.0,
        completion_contract_kind="mitigation",
    )
    assert result["catastrophe_zeroed"] is True
    assert result["total_score"] == 0.0
    assert result["survivor_outcome"] == pytest.approx(55.0)
    assert result["formal_score_eligible"] is True


def test_feasibility_gate_zeros_primary_not_coverage_diagnostic() -> None:
    incomplete = discriminative_core_total(
        _dimensions(counterfactual_prevention=80.0),
        task_completion=0.0,
        completion_contract_kind="feasibility",
        schedule_coverage=0.9,
    )
    complete = discriminative_core_total(
        _dimensions(counterfactual_prevention=80.0),
        task_completion=1.0,
        completion_contract_kind="feasibility",
        schedule_coverage=1.0,
    )
    assert incomplete["feasibility_zeroed"] is True
    assert incomplete["total_score"] == 0.0
    assert incomplete["schedule_coverage"] == 0.9
    assert incomplete["survivor_outcome"] == 80.0
    assert complete["total_score"] == 80.0
    assert complete["schedule_coverage"] == 1.0


def test_missing_wait_relative_is_ineligible() -> None:
    dimensions = _drop(
        "economic_cost",
        _drop("counterfactual_prevention", _dimensions(optimality_gap=90.0)),
    )
    result = discriminative_core_total(
        dimensions,
        task_completion=1.0,
        completion_contract_kind="mitigation",
    )
    assert result["formal_score_eligible"] is False
    assert result["total_score"] == 0.0
    assert result["wait_relative_source"] is None
