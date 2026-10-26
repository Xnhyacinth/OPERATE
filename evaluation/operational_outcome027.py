"""Evidence-qualified 0.27 outcomes on an unchanged, fixed task distribution.

Artifact/source authentication belongs to the caller. Both evaluator and audit
use these case rules; numeric hard zero and fulfillment support are separate.
Missing-measurement bounds condition on recorded point costs, not repeat draws.
"""

from __future__ import annotations

from collections import Counter
import math

VERSION = "0.27.0"
REVISION = "source_evidenced_operational_outcome.v1"


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _score(value: object) -> bool:
    return _number(value) and 0 <= value <= 100


def native_quality027(cost: float, scale: float, *, signed: bool) -> float:
    """The frozen cost mapping, with division arranged to avoid overflow."""
    if not (_number(cost) and _number(scale) and scale > 0 and type(signed) is bool):
        raise ValueError("invalid_native_cost_or_scale")
    if cost < 0:
        if not signed:
            raise ValueError("unsigned_negative_native_cost")
        return 100.0
    if cost <= scale:
        return 100.0 / (1.0 + cost / scale)
    inverse = scale / cost
    return 100.0 * inverse / (1.0 + inverse)


def score_case027(
    mission: dict,
    completion: dict,
    source_contract: dict,
    *,
    evidence_verified: bool,
) -> dict:
    """Return point availability, publish eligibility and F support together.

    ``evidence_verified`` means authenticated source, selected identity, native
    evidence and a source-authorized time/terminal window. It cannot be inferred
    from a model-supplied flag or the candidate report being audited.
    """
    if type(evidence_verified) is not bool:
        raise ValueError("invalid_evidence_qualification")
    source_service = source_contract.get("service") or {}
    economic_only = (
        source_contract.get("backend_kind") == "citylearn"
        and source_service.get("kind") == "automatic_exogenous_load_service"
        and source_service.get("applicable") is False
    )
    expected_applicable = not economic_only
    if completion.get("applicable") is not expected_applicable:
        raise ValueError("completion_applicability_not_source_authorized")
    fraction = completion.get("score")
    if fraction is not None and not _score(fraction):
        raise ValueError("invalid_completion_score")
    if economic_only and fraction is not None:
        raise ValueError("structural_completion_has_score")
    source_scale = (
        (source_contract.get("scales") or {}).get("native_cost", {}).get("value")
    )
    if not (_number(source_scale) and source_scale > 0):
        raise ValueError("source_scale_missing")
    domain = source_contract.get("native_cost_value_domain")
    if domain not in {"signed", "nonnegative"}:
        raise ValueError("source_cost_value_domain_invalid")
    result = {
        "evaluation_version": VERSION,
        "protocol_revision": REVISION,
        "numeric_determined": False,
        "publish_eligible": False,
        "evidence_qualified": evidence_verified,
        "native_quality": None,
        "native_quality_if_no_hard_failure": None,
        "completion_applicable": expected_applicable,
        "completion_score": None,
        "completion_measured": False,
        "hard_failure": None,
        "native_catastrophe_applicable": None,
        "hard_gate_signal_source": None,
        "score": None,
        "score_interval": None,
        "missing_measurement_bounds": [0.0, 100.0],
        "bounds_interpretation": "fixed_contract_point_cost_missing_measurement_range_not_confidence_interval",
        "reason": "source_identity_or_native_evidence_unqualified",
        "evidence_ids": [],
    }
    native_ids = mission.get("native_evidence_ids") or []
    if not evidence_verified or not native_ids:
        result["evidence_qualified"] = False
        return result
    result["evidence_ids"] = list(native_ids)
    if expected_applicable and _score(fraction) and completion.get("evidence_ids"):
        result.update(completion_score=float(fraction), completion_measured=True)
        result["evidence_ids"] += list(completion["evidence_ids"])
    safety = mission.get("safety") or {}
    signal = safety.get("signal_source")
    result["hard_gate_signal_source"] = signal
    result["native_catastrophe_applicable"] = (
        False
        if signal == "native_catastrophe_unmodeled"
        else True
        if signal
        in {
            "native_snapshot_marker",
            "native_ledger_marker",
            "native_convergence_recovered",
            "native_collapse_recovered",
        }
        else None
    )
    hard = safety.get("hard_failure")
    if hard is not None and type(hard) is not bool:
        raise ValueError("invalid_hard_failure_state")
    if safety.get("verified") is True and safety.get("evidence_ids"):
        result["hard_failure"] = hard
    hard = result["hard_failure"]
    if hard is True:
        result.update(
            native_quality=0.0,
            score=0.0,
            score_interval=[0.0, 0.0],
            missing_measurement_bounds=[0.0, 0.0],
            numeric_determined=True,
            publish_eligible=True,
            reason="verified_native_hard_failure",
        )
        result["evidence_ids"] += list(safety["evidence_ids"])
        return result
    cost, scale = mission.get("native_objective"), mission.get("source_scale")
    if scale is not None and scale != source_scale:
        raise ValueError("native_scale_not_source_authorized")
    signed = domain == "signed"
    quality = None
    if cost is not None and scale is not None:
        quality = native_quality027(cost, scale, signed=signed)
    interval = mission.get("native_quality_interval", [quality, quality])
    if quality is not None and not (
        isinstance(interval, list)
        and len(interval) == 2
        and all(_score(v) for v in interval)
        and interval[0] <= interval[1]
        and interval[0] - 1e-8 <= quality <= interval[1] + 1e-8
    ):
        raise ValueError("invalid_native_precision_interval")
    result["native_quality_if_no_hard_failure"] = quality
    upper = 100.0
    if quality is not None:
        upper = min(upper, quality)
    if result["completion_measured"]:
        upper = min(upper, result["completion_score"])
    result["missing_measurement_bounds"] = [0.0, upper]
    if hard is None:
        result["reason"] = "native_hard_failure_state_unknown"
        return result
    result["native_quality"] = quality
    if quality is None:
        result["reason"] = "native_cost_measurement_missing"
        return result
    if expected_applicable and not result["completion_measured"]:
        result["reason"] = completion.get("reason") or "completion_evidence_missing"
        return result
    low, high = min(interval[0], quality), max(interval[1], quality)
    point = min(quality, result["completion_score"]) if expected_applicable else quality
    bounds = (
        [min(low, fraction), min(high, fraction)]
        if expected_applicable
        else [low, high]
    )
    result.update(
        score=point,
        score_interval=bounds,
        numeric_determined=True,
        publish_eligible=True,
        missing_measurement_bounds=[point, point],
        reason="source_evidenced_joint_outcome"
        if expected_applicable
        else "source_authorized_economic_only",
    )
    return result


def aggregate_outcome027(
    rows: list[dict],
    contracts: dict[tuple[str, int], dict],
    weights: dict[tuple[str, int], float],
    *,
    models: list[str],
) -> dict:
    """Aggregate exactly the source panel; missing rows never change weights."""
    if not contracts or contracts.keys() != weights.keys():
        raise ValueError("invalid_fixed_case_population")
    if not models or len(set(models)) != len(models):
        raise ValueError("invalid_model_population")
    if any(not _number(w) or w <= 0 for w in weights.values()) or not math.isclose(
        math.fsum(weights.values()), 1.0, rel_tol=0, abs_tol=1e-12
    ):
        raise ValueError("invalid_frozen_weights")
    indexed = {}
    for row in rows:
        identity = row["model"], row["scenario_signature"], row["seed"]
        if (
            identity in indexed
            or identity[0] not in models
            or identity[1:] not in contracts
        ):
            raise ValueError("duplicate_or_foreign_candidate_case")
        indexed[identity] = row["outcome027"]
        outcome = row["outcome027"]
        bounds = outcome.get("missing_measurement_bounds")
        if not (
            type(outcome.get("numeric_determined")) is bool
            and type(outcome.get("publish_eligible")) is bool
            and type(outcome.get("evidence_qualified")) is bool
            and isinstance(bounds, list)
            and len(bounds) == 2
            and all(_score(v) for v in bounds)
            and bounds[0] <= bounds[1]
            and (outcome["numeric_determined"] is False or _score(outcome.get("score")))
            and (outcome["numeric_determined"] is True or outcome.get("score") is None)
            and (
                outcome["publish_eligible"] is False
                or (outcome["numeric_determined"] and outcome["evidence_qualified"])
            )
        ):
            raise ValueError("invalid_candidate_case_status")
        if outcome["numeric_determined"] and bounds != [
            outcome["score"],
            outcome["score"],
        ]:
            raise ValueError("determined_case_bounds_mismatch")
        if not outcome["evidence_qualified"] and bounds != [0.0, 100.0]:
            raise ValueError("unqualified_case_cannot_tighten_bounds")
    applicable = {
        key
        for key, contract in contracts.items()
        if not (
            contract.get("backend_kind") == "citylearn"
            and (contract.get("service") or {}).get("kind")
            == "automatic_exogenous_load_service"
            and (contract.get("service") or {}).get("applicable") is False
        )
    }
    fulfillment_mass = math.fsum(weights[key] for key in applicable)
    table = []
    for model in models:
        values = {key: indexed.get((model, *key)) for key in contracts}
        known = {
            key: value["score"]
            for key, value in values.items()
            if value and value["publish_eligible"] is True and _score(value["score"])
        }
        valid_f = {
            key: value["completion_score"]
            for key, value in values.items()
            if key in applicable and value and value["completion_measured"] is True
        }
        native = {
            key: value["native_quality"]
            for key, value in values.items()
            if value and _score(value["native_quality"])
        }
        hard = {
            key: value["hard_failure"]
            for key, value in values.items()
            if value and type(value["hard_failure"]) is bool
        }
        complete = len(known) == len(contracts)
        invalid = any(
            value and value["evidence_qualified"] is not True
            for value in values.values()
        )
        limits = {
            key: value["missing_measurement_bounds"] if value else [0.0, 100.0]
            for key, value in values.items()
        }
        point = (
            math.fsum(weights[key] * known[key] for key in contracts)
            if complete
            else None
        )
        table.append(
            {
                "model": model,
                "primary_score": point,
                "primary_rank": None,
                "complete": complete,
                "n_expected": len(contracts),
                "n_scored": len(known),
                "n_observed": sum(value is not None for value in values.values()),
                "valid_for_comparison": not invalid,
                "missing_measurement_bounds": [
                    math.fsum(weights[key] * limits[key][j] for key in contracts)
                    for j in (0, 1)
                ],
                "score_interval": [
                    math.fsum(
                        weights[key] * values[key]["score_interval"][j]
                        for key in contracts
                    )
                    for j in (0, 1)
                ]
                if complete
                else None,
                "native_quality": math.fsum(
                    weights[key] * native[key] for key in contracts
                )
                if len(native) == len(contracts)
                else None,
                "completion_only": {
                    "score": math.fsum(
                        weights[key] * valid_f[key] for key in applicable
                    )
                    / fulfillment_mass
                    if applicable and len(valid_f) == len(applicable)
                    else None,
                    "n_expected": len(applicable),
                    "n_measured": len(valid_f),
                    "weight_mass": fulfillment_mass,
                },
                "recorded_hard_failure_rate": 100
                * math.fsum(weights[key] * hard[key] for key in contracts)
                if len(hard) == len(contracts)
                else None,
                "safety_coverage": {
                    "declared_hard_gate_known_cases": len(hard),
                    "declared_hard_gate_known_weight_mass": math.fsum(
                        weights[key] for key in hard
                    ),
                    "native_catastrophe": {
                        label: {
                            "cases": sum(
                                (
                                    value.get("native_catastrophe_applicable")
                                    if value
                                    else None
                                )
                                is state
                                for value in values.values()
                            ),
                            "weight_mass": math.fsum(
                                weights[key]
                                for key, value in values.items()
                                if (
                                    value.get("native_catastrophe_applicable")
                                    if value
                                    else None
                                )
                                is state
                            ),
                        }
                        for label, state in (
                            ("applicable", True),
                            ("unmodeled", False),
                            ("unknown", None),
                        )
                    },
                    "interpretation": "recorded_declared_gates_not_comprehensive_safety",
                },
                "reason_counts": dict(
                    Counter(
                        value["reason"] if value else "missing_selected_case"
                        for value in values.values()
                    )
                ),
            }
        )
    ordered = sorted(
        (row for row in table if row["complete"] and row["valid_for_comparison"]),
        key=lambda row: (-row["primary_score"], row["model"]),
    )
    previous = rank = None
    for position, row in enumerate(ordered, 1):
        if row["primary_score"] != previous:
            rank = position
        row["primary_rank"] = rank
        previous = row["primary_score"]
    return {
        "evaluation_version": VERSION,
        "protocol_revision": REVISION,
        "n_expected": len(contracts),
        "models": table,
        "leaderboard": ordered,
        "bounds_interpretation": "fixed_panel_missing_measurement_identification_range_conditioned_on_recorded_point_costs_not_confidence_or_prediction",
        "strict_attainment_status": "not_calibrated",
        "primary_is_binary_task_attainment": False,
    }
