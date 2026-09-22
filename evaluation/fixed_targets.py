"""Fixed expert-target attainment, without a weakest-policy normalization.

Targets are compiled from authenticated, model-independent controller runs.
They are feasible performance targets, not certified optima. Oracle information
privileges are explicit; attainment is not original-task completion probability.
"""

from __future__ import annotations

import math
from statistics import mean

from evaluation.native_quality import ABS_TOLERANCE, REL_TOLERANCE, _valid

TARGET_POLICIES = ("greedy_heuristic", "oracle_offline")


def compile_target(reference):
    """Audit both fixed candidates; only proven infeasibility excludes a policy."""
    result = dict(
        schema_version="fixed_expert_target.v1",
        status="unavailable",
        reference_is_optimum=False,
        candidate_policies=list(TARGET_POLICIES),
        information_scope="oracle_may_use_future_information",
        same_information_attainability_certified=False,
    )
    for key in (
        "scenario_signature",
        "seed",
        "domain",
        "backend_kind",
        "runtime_identity",
    ):
        if key in reference:
            result[key] = reference[key]
    candidates, identities, evidence = {}, set(), []
    for policy in TARGET_POLICIES:
        pair = reference.get("policy_measurements", {}).get(policy, [])
        if (
            len(pair) != 2
            or not all(_valid(m) for m in pair)
            or reference.get("policy_determinism", {}).get(policy) is not True
        ):
            return {**result, "reason": "fixed_candidate_evidence_missing_or_unstable"}
        a, b = pair
        if any(
            a.get(k) != b.get(k) for k in ("feasible", "hard_failure", "task_success")
        ) or not math.isclose(
            a["actual_cost"],
            b["actual_cost"],
            rel_tol=REL_TOLERANCE,
            abs_tol=ABS_TOLERANCE,
        ):
            return {**result, "reason": "fixed_candidate_repeat_disagreement"}
        for m in pair:
            identities.add((m["objective_id"], m["unit"]))
            evidence.extend(m["evidence_ids"])
        if a["feasible"] and not a["hard_failure"]:
            candidates[policy] = mean(m["actual_cost"] for m in pair)
    if len(identities) != 1 or not candidates:
        return {**result, "reason": "no_feasible_fixed_target_or_objective_mismatch"}
    objective, unit = next(iter(identities))
    target = min(candidates.values())
    return {
        **result,
        "status": "ready",
        "objective_id": objective,
        "unit": unit,
        "target_cost": target,
        "candidate_costs": candidates,
        "target_policies": [p for p, c in candidates.items() if c == target],
        "target_evidence_ids": sorted(set(evidence)),
        "absolute_tolerance": ABS_TOLERANCE,
        "relative_tolerance": REL_TOLERANCE,
        "reason": "audited_fixed_expert_target",
    }


def score_target(measurement, target):
    result = dict(
        schema_version="expert_target_attainment.v1",
        score=None,
        attained=None,
        evidence_ids=measurement.get("evidence_ids", []),
    )
    if target.get("status") != "ready":
        return {**result, "reason": "fixed_target_unavailable"}
    if not _valid(measurement):
        return {**result, "reason": "native_measurement_unavailable"}
    if any(measurement[k] != target.get(k) for k in ("objective_id", "unit")):
        return {**result, "reason": "fixed_target_objective_mismatch"}
    goal = target.get("target_cost")
    proof = target.get("target_evidence_ids")
    if (
        type(goal) not in (int, float)
        or not math.isfinite(goal)
        or not isinstance(proof, list)
        or not proof
        or not all(isinstance(value, str) and value for value in proof)
        or target.get("absolute_tolerance") != ABS_TOLERANCE
        or target.get("relative_tolerance") != REL_TOLERANCE
    ):
        return {**result, "reason": "invalid_fixed_target"}
    actual = measurement["actual_cost"]
    feasible = measurement["feasible"] and not measurement["hard_failure"]
    attained = feasible and (
        actual <= goal
        or math.isclose(actual, goal, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE)
    )
    gap = actual - goal
    return {
        **result,
        "score": 100.0 if attained else 0.0,
        "attained": attained,
        "native_cost": actual,
        "target_cost": goal,
        "native_target_gap": gap if math.isfinite(gap) else None,
        "target_evidence_ids": target["target_evidence_ids"],
        "reason": "target_attained"
        if attained
        else ("native_constraint_failed" if not feasible else "target_not_attained"),
    }
