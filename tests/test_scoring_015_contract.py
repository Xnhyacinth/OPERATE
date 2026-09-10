from __future__ import annotations

import math

import pytest

from core import EvidenceLogger
from evaluation.scorer import (
    HEADLINE_SCORE_GROUPS,
    ScoringInputs,
    discriminative_core_total,
    score_episode,
    score_optimality_gap,
)


def _dimensions():
    return [
        {
            "name": name,
            "applicable": True,
            "calibrated_score": 50.0,
            "evidence_ids": [f"evidence:{name}"],
        }
        for group in HEADLINE_SCORE_GROUPS.values()
        for name in group["dimensions"]
    ]


def _inputs(**kwargs):
    return ScoringInputs(
        backend_tick_records=[],
        realized_events=[],
        cost_components={},
        per_load_shed_mwh={},
        load_classes={},
        stakeholder_mgr=None,
        dilemma_mgr=None,
        evidence_logger=kwargs.pop("evidence_logger", None),
        **kwargs,
    )


def test_structural_na_group_renormalizes_only_predeclared_contract():
    dimensions = _dimensions()
    contract = {
        "adaptive_replanning": {"applicable": False, "reason": "static"},
        "foresight_score": False,
    }
    for dimension in dimensions:
        if dimension["name"] in contract:
            dimension.update(applicable=False, evidence_ids=[])
    result = discriminative_core_total(
        dimensions, task_completion=1.0, dimension_applicability=contract
    )
    assert result["formal_score_eligible"] is True
    assert result["excluded_groups"] == ["adaptation_and_foresight"]
    assert result["legacy_five_group_weight_denominator"] == 85.0
    assert result["legacy_five_group_total"] == pytest.approx(57.5 / 0.85)
    assert result["total_score"] == 50.0
    assert result["fixed_five_group_total"] == 57.5
    assert sum(result["effective_group_weights"].values()) == pytest.approx(1.0)
    assert result["effective_group_weights"]["adaptation_and_foresight"] == 0.0
    without_contract = discriminative_core_total(dimensions, task_completion=1.0)
    assert without_contract["formal_score_eligible"] is True
    assert without_contract["legacy_formal_score_eligible"] is False
    assert without_contract["legacy_five_group_weight_denominator"] == 100.0


@pytest.mark.parametrize(
    "contract",
    [
        {"adaptive_replanning": False},
        {"adaptive_replanning": False, "foresight_score": True},
        {"adaptive_replanning": False, "foresight_score": "false"},
    ],
)
def test_partial_or_malformed_na_never_removes_group(contract):
    result = discriminative_core_total(
        [], task_completion=1.0, dimension_applicability=contract
    )
    assert "adaptation_and_foresight" in result["missing_groups"]
    assert "adaptation_and_foresight" not in result["excluded_groups"]


@pytest.mark.parametrize("source", ["engine", "agent"])
def test_serialized_applicability_requires_exact_engine_contract(source):
    logger = EvidenceLogger("contract")
    contract = {"adaptive_replanning": False, "foresight_score": False}
    eid = logger.log(
        "dimension_applicability_contract",
        tick=0,
        payload={"dimensions": contract},
        source=source,
    )
    score = score_episode(
        _inputs(
            evidence_logger=logger,
            dimension_applicability=contract,
            dimension_applicability_evidence_ids=[eid],
        )
    )
    assert score.to_dict()["dimension_applicability"] == (
        contract if source == "engine" else {}
    )


def test_unusable_counterfactual_cannot_credit_economic_cost():
    score = score_episode(
        _inputs(
            counterfactual_report={
                "applicable": False,
                "counterfactual_cost": 100.0,
                "reason_code": "replay_schedule_unproven",
            }
        )
    )
    economic = next(d for d in score.dimensions if d.name == "economic_cost")
    assert economic.applicable is False


def test_infeasible_solution_cannot_beat_reference_by_serving_less():
    score = score_optimality_gap(
        429.0,
        410.935,
        objective_component="routing_operating_cost",
        evidence_ids=["oracle"],
        feasibility={"feasible": False, "reason": "unserved_customers"},
    )
    assert score.applicable is True
    assert score.raw_score == 0.0
    assert "unserved_customers" in score.reason


@pytest.mark.parametrize(
    "cost,reference",
    [(math.nan, 10), (10, math.nan), (math.inf, 10), (10, math.inf), (-1, 10)],
)
def test_optimality_rejects_nonfinite_or_negative_costs(cost, reference):
    with pytest.raises(ValueError):
        score_optimality_gap(
            reference,
            cost,
            objective_component="production_cost",
            evidence_ids=["oracle"],
        )


@pytest.mark.parametrize("feasible", [True, False])
def test_structurally_incomparable_reference_is_uniform_na(feasible):
    score = score_optimality_gap(
        429.0,
        410.935,
        objective_component="routing_operating_cost",
        evidence_ids=["oracle"],
        feasibility={
            "feasible": feasible,
            "comparable": False,
            "reason": "closed_integer_reference_vs_open_dynamic_cost",
        },
    )
    assert score.applicable is False
    assert score.raw_score == 0.0
    assert "closed_integer_reference_vs_open_dynamic_cost" in score.reason


@pytest.mark.parametrize("run_to_end", [True, False])
def test_native_routing_records_reference_mismatch_and_actual_feasibility(run_to_end):
    from core import Action
    from domains.logistics.adapter import LogisticsEnvironment

    env = LogisticsEnvironment()
    env.reset(
        {
            "seed_id": "reference-contract",
            "backend_kind": "pyvrp_cvrp",
            "horizon_ticks": 4,
            "backend_config": {
                "network": {
                    "depot": {"x": 0, "y": 0},
                    "n_vehicles": 1,
                    "capacity": 10,
                    "customers": [{"id": "c1", "x": 1.2, "y": 0, "demand": 2}],
                }
            },
        },
        seed=1,
    )
    if run_to_end:
        for _ in range(4):
            env.step(Action())
    contract = env.ground_truth()["optimality_feasibility"]
    assert contract["comparable"] is False
    assert contract["feasible"] is run_to_end
    assert contract["unserved_customers"] == (0 if run_to_end else 1)


@pytest.mark.parametrize("pending", [True, False])
def test_information_consumption_requires_prior_evidence_and_terminal_consumer(pending):
    from evaluation.scorer import score_information_efficiency

    logger = EvidenceLogger("information-order")
    if pending:
        information = logger.log("investigation", 0, {}, source="tool")
        logger.log(
            "tool_call",
            1,
            {
                "name": "dispatch",
                "ok": True,
                "state_changing": True,
                "consumes_evidence_ids": [information],
                "payload": {"_status": "pending"},
            },
            source="tool",
        )
    else:
        consumer = logger.log(
            "tool_call",
            0,
            {"name": "dispatch", "ok": True, "state_changing": True},
            source="tool",
        )
        information = logger.log("investigation", 1, {}, source="tool")
        # Simulate a forged future reference in recorded agent dependency metadata.
        next(item for item in logger.items() if item.evidence_id == consumer).payload[
            "consumes_evidence_ids"
        ] = [information]
    score = score_information_efficiency(logger, evidence_ids=[information])
    assert score.raw_score == 0.0


def test_pending_consumer_cannot_make_read_tool_effective():
    from evaluation.scorer import score_tool_use_efficiency

    logger = EvidenceLogger("tool-order")
    information = logger.log(
        "tool_call",
        0,
        {"name": "query", "call_id": "q", "ok": True, "state_changing": False},
        source="tool",
    )
    logger.log(
        "tool_call",
        1,
        {
            "name": "dispatch",
            "call_id": "d",
            "ok": True,
            "state_changing": True,
            "consumes_evidence_ids": [information],
            "payload": {"_status": "pending"},
        },
        source="tool",
    )
    score = score_tool_use_efficiency(logger, evidence_ids=[information])
    assert score.raw_score == 0.0


def test_declared_na_member_cannot_contribute_stale_positive_score():
    result = discriminative_core_total(
        _dimensions(),
        task_completion=1.0,
        dimension_applicability={"adaptive_replanning": False},
    )
    assert result["group_support"]["adaptation_and_foresight"] == ["foresight_score"]


def test_declared_applicable_missing_member_cannot_raise_group_mean():
    dimensions = _dimensions()
    for dimension in dimensions:
        if dimension["name"] == "information_efficiency":
            dimension.update(applicable=False, evidence_ids=[])
        if dimension["name"] == "tool_use_efficiency":
            dimension["calibrated_score"] = 100.0
    result = discriminative_core_total(
        dimensions,
        task_completion=1.0,
        dimension_applicability={"information_efficiency": True},
    )
    assert result["group_scores"]["action_efficiency"] == 50.0
    assert result["formal_score_eligible"] is True
    assert result["legacy_formal_score_eligible"] is False
    assert result["missing_declared_dimensions"] == ["information_efficiency"]
    assert result["primary_missing_declared_dimensions"] == []


def test_native_runner_binds_routing_comparability_to_score_and_evidence(tmp_path):
    import json

    from runner.episode import run_one

    scenario = {
        "seed_id": "reference-runner-contract",
        "domain": "logistics",
        "family": "cvrp_dispatch",
        "backend_kind": "pyvrp_cvrp",
        "horizon_ticks": 4,
        "backend_config": {
            "network": {
                "depot": {"x": 0, "y": 0},
                "n_vehicles": 1,
                "capacity": 10,
                "customers": [{"id": "c1", "x": 1.2, "y": 0, "demand": 2}],
            }
        },
    }
    row = run_one(scenario, "wait_only", seed_override=1, trajectory_dir=tmp_path)
    dimension = next(
        d for d in row["score"]["dimensions"] if d["name"] == "optimality_gap"
    )
    assert dimension["applicable"] is False
    assert "reference_not_comparable" in dimension["reason"]
    evidence = [
        json.loads(line)
        for file in tmp_path.glob("*.evidence.jsonl")
        for line in file.read_text().splitlines()
    ]
    oracle = next(item for item in evidence if item["kind"] == "lp_oracle")
    assert oracle["payload"]["optimality_feasibility"]["comparable"] is False
    assert oracle["evidence_id"] in dimension["evidence_ids"]
