from __future__ import annotations

import pytest

from evaluation.scorer import (
    DISCRIMINATIVE_CORE_DIMENSIONS,
    discriminative_core_total,
)
from evaluation.task_completion import (
    contract_kind_from_name,
    task_completion_contract_kind,
)


def _dimensions() -> list[dict]:
    return [
        {
            "name": name,
            "applicable": True,
            "calibrated_score": 40.0,
            "evidence_ids": [f"evidence:{name}"],
        }
        for name in DISCRIMINATIVE_CORE_DIMENSIONS
        if name != "task_completion"
    ]


def test_mitigation_completion_does_not_enter_primary() -> None:
    incomplete = discriminative_core_total(
        _dimensions(),
        task_completion=0.0,
        completion_contract_kind="mitigation",
    )
    complete = discriminative_core_total(
        _dimensions(),
        task_completion=1.0,
        completion_contract_kind="mitigation",
    )
    assert incomplete["total_score"] == 40.0
    assert complete["total_score"] == 40.0
    assert incomplete["group_scores"]["task_completion"] == 0.0
    assert complete["group_scores"]["task_completion"] == 100.0
    assert complete["legacy_five_group_total"] - incomplete[
        "legacy_five_group_total"
    ] == pytest.approx(30.0)


def test_feasibility_gates_primary_on_full_completion() -> None:
    incomplete = discriminative_core_total(
        _dimensions(),
        task_completion=0.0,
        completion_contract_kind="feasibility",
    )
    complete = discriminative_core_total(
        _dimensions(),
        task_completion=1.0,
        completion_contract_kind="feasibility",
    )
    assert incomplete["formal_score_eligible"] is True
    assert incomplete["total_score"] == 0.0
    assert complete["total_score"] == 40.0


def test_survival_floor_zeros_primary() -> None:
    dimensions = _dimensions()
    for dimension in dimensions:
        if dimension["name"] == "system_survival":
            dimension["floor_violation"] = True
    result = discriminative_core_total(
        dimensions,
        task_completion=1.0,
        completion_contract_kind="mitigation",
    )
    assert result["survival_floor_zeroed"] is True
    assert result["catastrophe_zeroed"] is True
    assert result["formal_score_eligible"] is True
    assert result["total_score"] == 0.0
    assert result["survivor_outcome"] == 40.0
    assert result["group_scores"]["system_outcome"] == 40.0


def test_unsupported_contract_is_ineligible() -> None:
    result = discriminative_core_total(
        _dimensions(),
        task_completion=1.0,
        completion_contract_kind="unsupported",
    )
    assert result["formal_score_eligible"] is False
    assert result["total_score"] == 0.0


def test_job_shop_contract_kind_is_feasibility() -> None:
    assert (
        task_completion_contract_kind("logistics", "job_shop_dispatch")
        == "feasibility"
    )
    assert (
        contract_kind_from_name("datacenter.queue_sla_mitigation.v2")
        == "mitigation"
    )
    assert contract_kind_from_name("unsupported") == "unsupported"


def test_unknown_contract_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="completion_contract_kind"):
        discriminative_core_total(
            _dimensions(),
            task_completion=1.0,
            completion_contract_kind="quality",
        )
