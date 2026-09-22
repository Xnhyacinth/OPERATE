"""Absolute fulfillment must dominate selective cheap service."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.task_quality import evaluate_task_quality


def episode(tmp_path, *, completed=10, cost=100, hard_failure=False, arrived=2):
    counts = {
        "operations_total": 10,
        "operations_cancelled": 0,
        "operations_completed": completed,
        "operations_scheduled": completed,
        "jobs_total": 2,
        "jobs_arrived": arrived,
    }
    evidence = {**counts, "operations_required": 10}
    row = {
        "status": "ok",
        "scenario_signature": "case",
        "seed": 42,
        "backend_kind": "dynasched_flexible_job_shop",
        "domain": "logistics",
        "ground_truth_summary": {
            "cost_components": {"production_cost": cost},
            "chose_fatal_option": hard_failure,
        },
        "task_completion": {
            "contract": "logistics.job_shop.all_operations_scheduled.v1",
            "applicable": True,
            "completed": completed == 10,
            "evidence": evidence,
        },
        "trajectory_summary": {
            "operational_agency_valid_evidence_ids": ["cost", "safety"]
        },
        "score": {
            "dimensions": [
                {"name": "economic_cost", "applicable": True, "evidence_ids": ["cost"]},
                {
                    "name": "system_survival",
                    "applicable": True,
                    "floor_violation": hard_failure,
                    "evidence_ids": ["safety"],
                },
            ]
        },
    }
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "tick": 1,
                "observation": {**counts, "makespan": cost},
                "evidence_ids": ["cost", "safety"],
            }
        )
        + "\n"
    )
    binding = {
        "verified": True,
        "native_cost_bound": True,
        "artifacts": {
            "trajectory_artifact": {
                "path": str(trace),
                "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            }
        },
    }
    reference = {
        "schema_version": "native_quality.v1",
        "status": "ready",
        "scenario_signature": "case",
        "seed": 42,
        "backend_kind": row["backend_kind"],
        "objective_id": "dynasched_flexible_job_shop.makespan_unfinished_penalty.v1",
        "unit": "native_time_plus_unfinished_penalty",
        "weak_cost": 200,
        "strong_cost": 100,
        "normalization_scale": 100,
        "minimum_anchor_gap": 1e-7,
        "anchor_evidence_ids": ["reference"],
        "contract_sha256": "reference-hash",
    }
    return row, binding, reference


def test_full_bad_schedule_beats_eighty_percent_easy_jobs(tmp_path):
    row, binding, ref = episode(tmp_path, cost=100000)
    full = evaluate_task_quality(row, artifact_binding=binding, reference_contract=ref)
    row, binding, ref = episode(tmp_path, completed=8, cost=1)
    partial = evaluate_task_quality(
        row, artifact_binding=binding, reference_contract=ref
    )
    assert full["score"] > partial["score"] == 0
    assert partial["completion_fraction"] == 0.8
    assert partial["quality_fraction"] is None


def test_missing_quality_does_not_imply_perfect_delivery(tmp_path):
    row, binding, _ = episode(tmp_path)
    result = evaluate_task_quality(row, artifact_binding=binding)
    assert result["score"] is None
    assert result["completion_fraction"] == 1
    assert result["mandatory_complete"] is True
    assert result["reason"] == "quality_reference_unavailable"


def test_unarrived_jobs_cannot_shrink_denominator(tmp_path):
    row, binding, ref = episode(tmp_path, arrived=1)
    result = evaluate_task_quality(
        row, artifact_binding=binding, reference_contract=ref
    )
    assert result["score"] == 0
    assert result["mandatory_complete"] is False
    assert result["completion_fraction"] is None
    assert result["reason"] == "incomplete_source_obligations"


def test_hard_failure_has_zero_reward(tmp_path):
    row, binding, ref = episode(tmp_path, hard_failure=True)
    result = evaluate_task_quality(
        row, artifact_binding=binding, reference_contract=ref
    )
    assert result["score"] == 0
    assert result["hard_failure"] is True


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), True])
def test_invalid_native_cost_is_unavailable(tmp_path, cost):
    row, binding, ref = episode(tmp_path, cost=cost)
    assert (
        evaluate_task_quality(row, artifact_binding=binding, reference_contract=ref)[
            "score"
        ]
        is None
    )


def test_changed_summary_counts_are_rejected(tmp_path):
    row, binding, ref = episode(tmp_path, completed=8)
    row["task_completion"]["evidence"]["operations_completed"] = 10
    row["task_completion"]["evidence"]["operations_scheduled"] = 10
    assert (
        evaluate_task_quality(row, artifact_binding=binding, reference_contract=ref)[
            "reason"
        ]
        == "completion_trace_mismatch"
    )


def test_trace_hash_and_binding_are_required(tmp_path):
    row, binding, ref = episode(tmp_path)
    binding["verified"] = False
    assert (
        evaluate_task_quality(row, artifact_binding=binding, reference_contract=ref)[
            "score"
        ]
        is None
    )
    binding["verified"] = True
    (tmp_path / "trace.jsonl").write_text("{}\n")
    assert (
        evaluate_task_quality(row, artifact_binding=binding, reference_contract=ref)[
            "reason"
        ]
        == "terminal_trace_hash_mismatch"
    )


def test_reference_identity_cannot_be_substituted(tmp_path):
    row, binding, ref = episode(tmp_path)
    ref["scenario_signature"] = "another-case"
    assert (
        evaluate_task_quality(row, artifact_binding=binding, reference_contract=ref)[
            "reason"
        ]
        == "quality_reference_identity_mismatch"
    )


def test_optimization_does_not_invent_absolute_completion(tmp_path):
    row, binding, ref = episode(tmp_path)
    row.update(backend_kind="citylearn", domain="building_energy")
    row["task_completion"].update(
        contract="building_energy.citylearn.storage_dispatch.v1", completed=False
    )
    ref.update(
        backend_kind="citylearn",
        objective_id="citylearn.storage_dispatch_total.v1",
        unit="native_cost_units",
    )
    first = evaluate_task_quality(row, artifact_binding=binding, reference_contract=ref)
    row["task_completion"]["completed"] = True
    second = evaluate_task_quality(
        row, artifact_binding=binding, reference_contract=ref
    )
    assert first["score"] == second["score"] == 75
    assert first["mission_type"] == "optimization"
    assert first["completion_fraction"] is None
    assert first["mandatory_complete"] is None


def test_no_mutation_and_better_complete_schedule_improves_quality(tmp_path):
    row, binding, ref = episode(tmp_path, cost=100)
    before = deepcopy(row)
    good = evaluate_task_quality(row, artifact_binding=binding, reference_contract=ref)
    assert row == before
    row, binding, ref = episode(tmp_path, cost=300)
    bad = evaluate_task_quality(row, artifact_binding=binding, reference_contract=ref)
    assert 0 < bad["score"] < good["score"] <= 100


def contract():
    return {
        "schema_version": "task_acceptance.v1",
        "scenario_signature": "case",
        "seed": 42,
        "backend_kind": "dynasched_flexible_job_shop",
        "source": "frozen_source_acceptance_manifest",
        "completion_fraction": {
            "numerator_path": ["operations_completed"],
            "denominator": 10,
        },
        "mandatory": [
            {"path": ["operations_completed"], "op": "eq", "target": 10},
            {"path": ["operations_cancelled"], "op": "eq", "target": 0},
        ],
        "quality": {
            "cost_path": ["makespan"],
            "good_cost": 50,
            "bad_cost": 500,
            "excludes_unfulfilled_penalties": True,
        },
    }


def test_predeclared_absolute_contract_uses_only_bound_native_fields(tmp_path):
    row, binding, _ = episode(tmp_path, cost=500)
    row["task_completion"]["completed"] = False
    result = evaluate_task_quality(
        row, artifact_binding=binding, acceptance_contract=contract()
    )
    assert result["score"] == 0
    assert result["mandatory_complete"]
    assert result["acceptance_contract_sha256"]


def test_predicate_cannot_access_python_or_unknown_fields(tmp_path):
    row, binding, _ = episode(tmp_path)
    spec = contract()
    spec["mandatory"][0]["path"] = "__class__.__dict__"
    result = evaluate_task_quality(
        row, artifact_binding=binding, acceptance_contract=spec
    )
    assert result["score"] is None
    assert result["reason"] == "absolute_mandatory_predicate_invalid"


def test_predeclared_denominator_is_not_replaced_by_completed_work(tmp_path):
    row, binding, _ = episode(tmp_path, completed=8)
    spec = contract()
    spec["mandatory"][0]["target"] = 8
    result = evaluate_task_quality(
        row, artifact_binding=binding, acceptance_contract=spec
    )
    assert result["completion_fraction"] == 0.8
    assert result["score"] == 0


def test_quality_without_penalty_separation_is_unavailable(tmp_path):
    row, binding, _ = episode(tmp_path)
    spec = contract()
    del spec["quality"]["excludes_unfulfilled_penalties"]
    result = evaluate_task_quality(
        row, artifact_binding=binding, acceptance_contract=spec
    )
    assert result["score"] is None
    assert result["completion_fraction"] == 1


def test_declared_quality_overflow_is_not_silently_clipped(tmp_path):
    row, binding, _ = episode(tmp_path)
    spec = contract()
    spec["quality"].update(good_cost=-1e308, bad_cost=1e308)
    result = evaluate_task_quality(
        row, artifact_binding=binding, acceptance_contract=spec
    )
    assert result["score"] is None


@pytest.mark.parametrize(
    "backend",
    [
        "citylearn",
        "alibaba_trace_sim",
        "pyvrp_cvrp",
        "pyvrp_vrptw",
        "orgym_invmgmt",
        "pandapower_lv",
        "pymgrid_economic_dispatch",
        "dynasched_flexible_job_shop",
        "sumo_ego",
        "cigre_distribution",
        "pglib_uc_synthetic",
        "opendss_fresh_feeders",
        "opendss_ieee13",
    ],
)
def test_each_native_backend_keeps_its_mission_semantics(tmp_path, backend):
    from evaluation.native_objectives import NATIVE_OBJECTIVES, _CONTRACTS

    row, binding, ref = episode(tmp_path)
    domain, objective, _ = NATIVE_OBJECTIVES[backend]
    row.update(backend_kind=backend, domain=domain)
    row["task_completion"]["contract"] = sorted(_CONTRACTS[backend])[0]
    if backend == "pymgrid_economic_dispatch":
        row["task_completion"]["evidence"].update(
            actual_task_loss=100,
            task_loss_component_keys=["balance_error_mw", "shed_penalty"],
        )
    ref.update(
        backend_kind=backend,
        objective_id=f"{backend}.{objective}.v1",
        unit="native_time_plus_unfinished_penalty"
        if backend == "dynasched_flexible_job_shop"
        else "native_cost_units",
    )
    result = evaluate_task_quality(
        row, artifact_binding=binding, reference_contract=ref
    )
    assert result["applicable"] is True
    assert 0 <= result["score"] <= 100
    assert result["schema_version"] == "task_quality.v2"
    assert result["protocol_revision"] == 2
    assert result["mission_type"] == (
        "feasibility" if backend == "dynasched_flexible_job_shop" else "optimization"
    )


def test_authenticated_optimization_does_not_reparse_unused_terminal_trace(
    tmp_path, monkeypatch
):
    row, binding, ref = episode(tmp_path)
    row.update(backend_kind="citylearn", domain="building_energy")
    row["task_completion"].update(
        contract="building_energy.citylearn.storage_dispatch.v1", completed=False
    )
    ref.update(
        backend_kind="citylearn",
        objective_id="citylearn.storage_dispatch_total.v1",
        unit="native_cost_units",
    )

    def unexpected_read(self):
        raise AssertionError("optimization must use already bound native objective")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    assert (
        evaluate_task_quality(row, artifact_binding=binding, reference_contract=ref)[
            "score"
        ]
        is not None
    )
