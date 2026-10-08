"""Frozen, evidence-linked operational acceptance; no simulation or calibration.

Callers authenticate source contracts and measurement artifacts before invoking
this module, including horizon_ticks from the frozen source contract. Predicate
paths require native outcome review: the denylist catches obvious process paths
but does not certify arbitrary field names as business outcomes. A contract is a requirement, not evidence of its own reachability.
"""

from __future__ import annotations

import hashlib
import json
import math

IDENTITY = (
    "scenario_signature",
    "seed",
    "domain",
    "backend_kind",
    "scenario_sha256",
    "objective_id",
    "unit",
)
GROUPS = ("service", "trajectory_constraints", "resource_budgets", "settlement")
FORBIDDEN = {
    "action",
    "actions",
    "tool_results",
    "tool_calls",
    "plan",
    "plans",
    "messages",
    "reasoning",
    "fog_of_war",
    "tool_call_count",
    "plan_count",
    "action_count",
}


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _validate(contract, source):
    _require(
        contract.get("schema_version") == "operational_acceptance.v1",
        "invalid_acceptance_schema",
    )
    identity = contract.get("identity", {})
    _require(
        all(k in identity and identity[k] == source.get(k) for k in IDENTITY),
        "acceptance_identity_mismatch",
    )
    _require(
        type(identity["seed"]) is int
        and all(
            isinstance(identity[k], str) and identity[k]
            for k in IDENTITY
            if k != "seed"
        ),
        "invalid_acceptance_identity",
    )
    provenance = contract.get("provenance", {})
    _require(
        provenance.get("kind") in {"source_requirement", "same_information_reference"},
        "unsupported_target_provenance",
    )
    _require(
        isinstance(provenance.get("source"), str) and bool(provenance["source"]),
        "target_provenance_missing",
    )
    _require(
        provenance.get("per_seed_best_selection", False) is False
        and provenance.get("uses_future_information", False) is False,
        "unfair_target_information",
    )
    artifacts = provenance.get("artifacts")
    _require(
        isinstance(artifacts, list)
        and bool(artifacts)
        and all(
            isinstance(a, dict)
            and isinstance(a.get("path"), str)
            and bool(a["path"])
            and isinstance(a.get("sha256"), str)
            and len(a["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in a["sha256"])
            for a in artifacts
        ),
        "target_artifacts_missing_or_invalid",
    )
    if provenance["kind"] == "same_information_reference":
        _require(
            provenance.get("information_parity_verified") is True
            and provenance.get("policy_frozen_before_test") is True
            and bool(provenance.get("information_parity_evidence")),
            "same_information_proof_missing",
        )
        policy = provenance.get("reference_policy_id")
        _require(
            isinstance(policy, str) and bool(policy) and "oracle" not in policy.lower(),
            "oracle_not_same_information_reference",
        )
        _require(
            provenance.get("selection_rule") == "fixed_before_test"
            and isinstance(provenance.get("calibration_split"), str)
            and provenance["calibration_split"].strip()
            and provenance["calibration_split"].strip().lower() != "test",
            "reference_selection_not_development_frozen",
        )
        parity = provenance.get("information_parity_evidence")
        _require(isinstance(parity, dict), "information_parity_bindings_missing")
        for key in (
            "observation_profile",
            "action_profile",
            "tool_latency",
            "resource_budget",
            "horizon",
        ):
            pair = parity.get(key, {})
            reference_hash = (
                pair.get("reference_sha256") if isinstance(pair, dict) else None
            )
            _require(
                isinstance(reference_hash, str)
                and len(reference_hash) == 64
                and all(c in "0123456789abcdef" for c in reference_hash)
                and reference_hash == pair.get("task_sha256"),
                "information_parity_binding_mismatch",
            )
    _require(
        type(contract.get("requirements_communicated")) is bool,
        "communication_status_missing",
    )
    q = contract.get("quality", {})
    _require(
        _number(q.get("absolute_tolerance")) and q["absolute_tolerance"] >= 0,
        "invalid_numerical_tolerance",
    )
    _require(
        isinstance(q.get("tolerance_rationale"), str)
        and bool(q["tolerance_rationale"]),
        "tolerance_rationale_missing",
    )
    if q.get("mode") == "fixed_threshold":
        _require(_number(q.get("threshold")), "invalid_acceptance_threshold")
    elif q.get("mode") == "uniform_threshold_interval":
        _require(
            all(_number(q.get(k)) for k in ("good", "bad", "success_threshold")),
            "invalid_acceptance_interval",
        )
        _require(
            q["good"] < q["bad"]
            and _number(q["bad"] - q["good"])
            and q["good"] <= q["success_threshold"] <= q["bad"],
            "invalid_acceptance_interval",
        )
    else:
        raise ValueError("unsupported_acceptance_mode")
    _require(
        contract.get("horizon_policy", "fixed_horizon")
        in {"fixed_horizon", "native_mission_terminal"},
        "invalid_horizon_policy",
    )
    for group in GROUPS:
        spec = contract.get("requirements", {}).get(group, {})
        predicates = spec.get("predicates")
        if spec.get("not_applicable_reason"):
            _require(
                isinstance(spec["not_applicable_reason"], str) and not predicates,
                "ambiguous_requirement_applicability",
            )
            continue
        _require(
            isinstance(predicates, list) and bool(predicates),
            "requirement_declaration_missing",
        )
        for p in predicates:
            _require(isinstance(p, dict), "invalid_predicate")
            path = p.get("path")
            _require(
                isinstance(path, list)
                and bool(path)
                and all(isinstance(k, str) and k and k not in FORBIDDEN for k in path),
                "invalid_measurement_path",
            )
            _require(
                p.get("source") in {"measurement", "snapshot", "trace"},
                "invalid_predicate_source",
            )
            _require(p.get("op") in {"eq", "le", "ge"}, "invalid_predicate_operator")
            _require(
                _number(p.get("target"))
                or (type(p.get("target")) is bool and p["op"] == "eq"),
                "invalid_predicate_target",
            )
            _require(
                p.get("scope") in {"terminal", "all_ticks", "window"},
                "invalid_predicate_scope",
            )
            _require(
                p.get("aggregation") in {"all", "sum", "min", "max"},
                "invalid_predicate_aggregation",
            )
            _require(
                "dt" not in p and "multiply_dt" not in p,
                "time_integration_not_supported",
            )
            if p["source"] != "trace":
                _require(p["scope"] == "terminal", "invalid_nontrace_scope")
            elif p["scope"] != "terminal":
                start, end = p.get("start_tick"), p.get("end_tick")
                _require(
                    type(start) is int
                    and (
                        (type(end) is int and 0 <= start <= end)
                        or (
                            end == "mission_end"
                            and start == 1
                            and p["scope"] == "all_ticks"
                            and contract.get("horizon_policy")
                            == "native_mission_terminal"
                        )
                    ),
                    "invalid_tick_window",
                )
            if p["aggregation"] == "sum":
                _require(
                    p.get("value_semantics") == "per_interval_amount",
                    "cumulative_or_rate_sum_forbidden",
                )


def validate_acceptance_contract(contract, source_contract):
    """Validate against an independently authenticated source identity."""
    try:
        _validate(contract, source_contract)
        encoded = json.dumps(contract, sort_keys=True, allow_nan=False).encode()
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return {"status": "unavailable", "reason": str(exc)}
    return {
        "status": "ready",
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
        "retrospective_only": True,
        "requirements_communicated_declared": contract["requirements_communicated"],
        "formal_eligibility": False,
        "provenance_verification": "caller_must_authenticate_artifact_hashes_and_requirement_communication",
    }


def _ids(record, valid):
    ids = record.get("evidence_ids")
    _require(
        isinstance(ids, list)
        and bool(ids)
        and all(isinstance(i, str) and i in valid for i in ids),
        "measurement_evidence_missing",
    )
    return ids


def _value(record, path):
    current = record
    for key in path:
        if isinstance(current, dict):
            _require(
                not any(
                    key in (current.get(marker) or [])
                    for marker in ("_hidden_attrs", "_noisy_attrs", "_stale_attrs")
                ),
                "predicate_measurement_hidden_noisy_or_stale",
            )
        _require(
            isinstance(current, dict)
            and not current.get("is_fogged", False)
            and not current.get("fogged", False)
            and key in current,
            "predicate_measurement_missing_or_fogged",
        )
        current = current[key]
    _require(_number(current) or type(current) is bool, "predicate_measurement_invalid")
    return current


def _conjunction(values):
    return False if False in values else None if None in values else True


def _predicate(p, measurement, snapshot, trace, valid):
    p = dict(p)
    if p.get("end_tick") == "mission_end":
        p["end_tick"] = measurement.get("horizon_ticks")
    if p["source"] == "trace":
        _require(isinstance(trace, list) and bool(trace), "trace_missing")
        _require(
            all(
                isinstance(row, dict) and type(row.get("tick")) is int for row in trace
            ),
            "trace_ticks_invalid",
        )
        ticks = [row["tick"] for row in trace]
        _require(ticks == sorted(set(ticks)), "trace_ticks_duplicate_or_unordered")
        if p["scope"] == "all_ticks":
            horizon = measurement.get("horizon_ticks")
            _require(
                type(horizon) is int
                and horizon > 0
                and p["start_tick"] == 1
                and p["end_tick"] == horizon,
                "all_ticks_horizon_mismatch",
            )
        if p["scope"] == "terminal":
            horizon = measurement.get("horizon_ticks")
            _require(
                type(horizon) is int and trace[-1]["tick"] == horizon,
                "terminal_tick_horizon_mismatch",
            )
            records = trace[-1:]
        else:
            records = [
                row for row in trace if p["start_tick"] <= row["tick"] <= p["end_tick"]
            ]
            _require(
                [row["tick"] for row in records]
                == list(range(p["start_tick"], p["end_tick"] + 1)),
                "predicate_tick_window_incomplete",
            )
    else:
        records = [measurement if p["source"] == "measurement" else snapshot]
    evidence, values = [], []
    for row in records:
        evidence.extend(_ids(row, valid))
        values.append(_value(row, p["path"]))
    if p["aggregation"] != "all":
        _require(
            all(_number(v) for v in values), "numeric_aggregation_requires_numbers"
        )
        values = [{"sum": sum, "min": min, "max": max}[p["aggregation"]](values)]
        _require(_number(values[0]), "nonfinite_predicate_aggregate")
    target = p["target"]
    _require(
        all(
            (type(v) is bool and type(target) is bool)
            or (_number(v) and _number(target))
            for v in values
        ),
        "predicate_value_type_mismatch",
    )
    uncertainty = 0.0
    if p["source"] == "measurement":
        if p["path"] == ["cost"]:
            uncertainty = measurement.get("cost_precision_bound", 0.0)
        elif len(p["path"]) == 2 and p["path"][0] == "components":
            uncertainty = measurement.get("component_precision_bounds", {}).get(
                p["path"][1], 0.0
            )
    _require(_number(uncertainty) and uncertainty >= 0, "invalid_predicate_precision")
    results = []
    for value in values:
        if type(value) is bool:
            results.append(value == target)
            continue
        low, high = value - uncertainty, value + uncertainty
        _require(_number(low) and _number(high), "nonfinite_predicate_interval")
        if p["op"] == "le":
            results.append(True if high <= target else False if low > target else None)
        elif p["op"] == "ge":
            results.append(True if low >= target else False if high < target else None)
        else:
            results.append(
                True
                if low == high == target
                else False
                if target < low or target > high
                else None
            )
    passed = _conjunction(results)
    return passed, evidence


def score_acceptance(
    measurement, contract, *, snapshot_inputs, trace, valid_evidence_ids
):
    """Score authenticated measurements; missing evidence never means success.

    Measurement must carry the same full source identity as the contract.
    Numerical tolerance affects strict attainment only, never interval quality.
    """
    result = {
        "schema_version": "operational_acceptance_score.v1",
        "score": None,
        "attained": None,
        "applicable": False,
        "evidence_ids": [],
    }
    validation = validate_acceptance_contract(contract, measurement)
    if validation["status"] != "ready":
        return {**result, "reason": validation["reason"]}
    result.update({k: v for k, v in validation.items() if k != "status"})
    try:
        valid = set(valid_evidence_ids)
        evidence = list(_ids(measurement, valid))
        _require(
            measurement.get("applicable") is True
            and type(measurement.get("hard_failure")) is bool,
            "native_measurement_unavailable",
        )
        if measurement["hard_failure"]:
            return {
                **result,
                "applicable": True,
                "score": 0.0,
                "attained": False,
                "evidence_ids": evidence,
                "reason": "native_hard_failure",
            }
        _require(
            measurement.get("complete_observation_window") is True,
            "incomplete_observation_window",
        )
        configured = measurement.get("configured_horizon_ticks")
        observed = measurement.get("horizon_ticks")
        if configured is not None:
            _require(
                type(configured) is int
                and type(observed) is int
                and 0 < observed <= configured,
                "invalid_observation_horizon",
            )
            if observed < configured:
                _require(
                    contract.get("horizon_policy") == "native_mission_terminal"
                    and measurement.get("early_completion_verified") is True,
                    "unverified_early_mission_completion",
                )
        actual = measurement.get("cost")
        _require(_number(actual), "native_cost_invalid")
        outcomes = {}
        for group in GROUPS:
            results = []
            for p in contract["requirements"][group].get("predicates", []):
                passed, ids = _predicate(p, measurement, snapshot_inputs, trace, valid)
                results.append(passed)
                evidence.extend(ids)
            outcomes[group] = _conjunction(results)
        q = contract["quality"]
        target = (
            q["threshold"] if q["mode"] == "fixed_threshold" else q["success_threshold"]
        )
        precision = measurement.get("cost_precision_bound", 0.0)
        _require(_number(precision) and precision >= 0, "invalid_cost_precision_bound")
        lower_cost, upper_cost = actual - precision, actual + precision
        _require(_number(lower_cost) and _number(upper_cost), "nonfinite_cost_interval")
        cutoff = target + q["absolute_tolerance"]
        _require(_number(cutoff), "nonfinite_acceptance_cutoff")
        attained = (
            upper_cost <= cutoff
            if upper_cost <= cutoff or lower_cost > cutoff
            else None
        )
        gate = _conjunction(list(outcomes.values()))
        if gate is False:
            attained = False
        elif gate is None and attained is not False:
            attained = None
        score = None if attained is None else 100.0 * attained
        score_interval = [0.0, 100.0] if attained is None else [score, score]
        if q["mode"] == "uniform_threshold_interval":
            score = (
                100.0
                if actual <= q["good"]
                else 0.0
                if actual >= q["bad"]
                else 100.0 * ((q["bad"] - actual) / (q["bad"] - q["good"]))
            )

            def quality_at(cost):
                if cost <= q["good"]:
                    return 100.0
                if cost >= q["bad"]:
                    return 0.0
                return 100.0 * ((q["bad"] - cost) / (q["bad"] - q["good"]))

            score_interval = [quality_at(upper_cost), quality_at(lower_cost)]
            if gate is False:
                score = 0.0
                score_interval = [0.0, 0.0]
            elif gate is None:
                score = None
                score_interval[0] = 0.0
        return {
            **result,
            "applicable": True,
            "score": score,
            "attained": attained,
            "native_cost": actual,
            "score_interval": score_interval,
            "approximate_cost": precision > 0,
            "cost_interval": [lower_cost, upper_cost],
            "requirement_results": outcomes,
            "evidence_ids": sorted(set(evidence)),
            "reason": "frozen_acceptance_evaluated",
        }
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return {**result, "reason": str(exc)}
