"""Behavioral contracts for evidence-backed native outcome extraction."""

from copy import deepcopy

import pytest

from evaluation.native_objectives import extract_native_objective


def episode(backend="citylearn", domain="building_energy"):
    return {
        "status": "ok",
        "backend_kind": backend,
        "domain": domain,
        "ground_truth_summary": {
            "cost_components": {"energy_cost": 20.0},
            "chose_fatal_option": False,
        },
        "counterfactual": {
            "actual_cost": 20.0,
            "actual_components": {"energy_cost": 20.0},
        },
        "task_completion": {
            "contract": "building_energy.citylearn.storage_dispatch.v1",
            "applicable": True,
            "completed": False,
            "evidence": {"actual_cost": 20.0},
        },
        "trajectory_summary": {
            "operational_agency_valid_evidence_ids": ["cost", "survival"]
        },
        "score": {
            "dimensions": [
                {"name": "economic_cost", "applicable": True, "evidence_ids": ["cost"]},
                {
                    "name": "system_survival",
                    "applicable": True,
                    "floor_violation": False,
                    "evidence_ids": ["survival"],
                },
            ]
        },
    }


def test_mitigation_failure_is_not_infeasibility():
    row = episode()
    before = deepcopy(row)
    result = extract_native_objective(row)
    assert result["applicable"]
    assert result["actual_cost"] == 20.0
    assert result["feasible"] is True
    assert result["hard_failure"] is False
    assert result["task_success"] is False
    assert result["evidence_ids"] == ["cost", "survival"]
    assert row == before


@pytest.mark.parametrize(
    "field", ["ground_truth_summary", "score", "trajectory_summary"]
)
def test_missing_evidence_is_unavailable(field):
    row = episode()
    del row[field]
    assert not extract_native_objective(row)["applicable"]


@pytest.mark.parametrize("cost", [True, float("nan"), float("inf"), "20", 10**400])
def test_non_numeric_or_nonfinite_cost_is_unavailable(cost):
    row = episode()
    row["ground_truth_summary"]["cost_components"]["energy_cost"] = cost
    assert not extract_native_objective(row)["applicable"]


def test_actual_native_cost_is_not_replaced_by_counterfactual_replay():
    row = episode()
    row["counterfactual"]["actual_components"]["energy_cost"] = 21
    row["counterfactual"]["actual_cost"] = 21
    result = extract_native_objective(row)
    assert result["applicable"] and result["actual_cost"] == 20
    assert result["counterfactual_consistent"] is False
    del row["counterfactual"]
    result = extract_native_objective(row)
    assert result["applicable"] and result["actual_cost"] == 20
    assert result["counterfactual_consistent"] is None


def test_counterfactual_evidence_only_certifies_exactly_matching_actual_run():
    row = episode()
    row["score"]["dimensions"][0].update(applicable=False, evidence_ids=["declaration"])
    row["score"]["dimensions"].append(
        {
            "name": "counterfactual_prevention",
            "applicable": True,
            "evidence_ids": ["cost"],
        }
    )
    result = extract_native_objective(row)
    assert result["applicable"] and result["actual_cost"] == 20
    assert result["cost_evidence_source"] == "matched_actual_replay"
    row["counterfactual"]["actual_components"]["energy_cost"] = 21
    assert not extract_native_objective(row)["applicable"]
    row["counterfactual"]["actual_components"]["energy_cost"] = 20
    row["score"]["dimensions"][-1]["applicable"] = False
    assert not extract_native_objective(row)["applicable"]


def test_signed_settlement_requires_backend_declaration():
    row = episode()
    for key in ("ground_truth_summary", "counterfactual"):
        row[key][
            "cost_components" if key == "ground_truth_summary" else "actual_components"
        ] = {"energy_cost": -20.0}
    row["counterfactual"]["actual_cost"] = -20
    row["task_completion"]["evidence"]["actual_cost"] = -20
    assert not extract_native_objective(row)["applicable"]
    row["ground_truth_summary"]["cost_component_value_domains"] = {
        "energy_cost": "signed"
    }
    assert extract_native_objective(row)["actual_cost"] == -20


def test_pymgrid_uses_declared_state_loss_not_economic_total():
    row = episode("pymgrid_economic_dispatch", "microgrid")
    row["task_completion"]["contract"] = "microgrid.native_state_loss.v1"
    row["task_completion"]["evidence"].update(
        actual_task_loss=75.0,
        task_loss_component_keys=["balance_error_mw", "shed_penalty"],
    )
    assert extract_native_objective(row)["actual_cost"] == 75
    del row["task_completion"]["evidence"]["actual_task_loss"]
    assert not extract_native_objective(row)["applicable"]


def test_pymgrid_rejects_task_loss_conflicting_with_available_native_records():
    row = episode("pymgrid_economic_dispatch", "microgrid")
    row["task_completion"]["contract"] = "microgrid.native_state_loss.v1"
    row["task_completion"]["evidence"].update(
        actual_task_loss=75.0,
        task_loss_component_keys=["balance_error_mw", "shed_penalty"],
    )
    row["ground_truth_summary"]["_task_tick_records"] = [
        {"balance_error_mw": -0.25, "shed_penalty": 25.0}
    ]
    assert extract_native_objective(row)["actual_cost"] == 75
    row["task_completion"]["evidence"]["actual_task_loss"] = 76
    assert not extract_native_objective(row)["applicable"]


def test_hard_failure_requires_evidenced_survival_gate():
    row = episode()
    row["score"]["dimensions"][1]["floor_violation"] = True
    result = extract_native_objective(row)
    assert result["applicable"] and result["hard_failure"] is True
    assert result["feasible"] is False
    row["score"]["dimensions"][1]["evidence_ids"] = ["unknown"]
    assert not extract_native_objective(row)["applicable"]


def test_dynasched_unfinished_is_infeasible_not_missing():
    row = episode("dynasched_flexible_job_shop", "logistics")
    row["ground_truth_summary"]["cost_components"] = {"production_cost": 2020.0}
    row["counterfactual"].update(
        actual_cost=2020.0, actual_components={"production_cost": 2020.0}
    )
    row["task_completion"].update(
        contract="logistics.job_shop.all_operations_scheduled.v1"
    )
    row["task_completion"]["evidence"] = {
        "operations_required": 3,
        "operations_completed": 1,
        "operations_scheduled": 1,
        "operations_total": 3,
        "operations_cancelled": 0,
    }
    result = extract_native_objective(row)
    assert result["applicable"] and result["feasible"] is False
    assert result["actual_cost"] == 2020
    assert result["hard_failure"] is False
    del row["task_completion"]["evidence"]["operations_completed"]
    assert not extract_native_objective(row)["applicable"]


def test_unknown_contract_or_backend_is_not_guessed():
    row = episode()
    row["task_completion"]["contract"] = "future.v9"
    assert not extract_native_objective(row)["applicable"]


def test_fatal_state_uses_explicit_completion_evidence_when_native_not_modeled():
    row = episode()
    row["ground_truth_summary"]["chose_fatal_option"] = None
    assert not extract_native_objective(row)["applicable"]
    row["task_completion"]["evidence"]["chose_fatal_option"] = False
    assert extract_native_objective(row)["hard_failure"] is False
    row["task_completion"]["evidence"]["chose_fatal_option"] = True
    assert extract_native_objective(row)["hard_failure"] is True
    row = episode("future_backend")
    assert not extract_native_objective(row)["applicable"]
