"""Domain-native material headroom contract."""

from __future__ import annotations

import math
from typing import Any, Literal

HeadroomDirection = Literal["lower_is_better", "higher_is_better"]
TRAFFIC_NATIVE_SIGNAL_HEADROOM_V2 = (
    "traffic.native_signal_supervision.v2"
)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def evaluate_material_headroom(
    *,
    metric_name: str,
    direction: HeadroomDirection,
    reference_value: Any,
    wait_value: Any,
    threshold: Any,
    native_units: str,
    evidence_kind: str,
) -> dict[str, Any]:
    reference = _finite(reference_value)
    wait = _finite(wait_value)
    floor = _finite(threshold)
    base = {
        "status": "held",
        "metric_name": str(metric_name),
        "direction": direction,
        "reference_value": reference,
        "wait_value": wait,
        "absolute_delta": None,
        "relative_delta": None,
        "threshold": floor,
        "native_units": str(native_units),
        "evidence_kind": str(evidence_kind),
        "reason_code": "native_headroom_value_missing",
    }
    if reference is None or wait is None or floor is None:
        return base
    if floor < 0:
        return {**base, "reason_code": "material_headroom_threshold_invalid"}
    delta = wait - reference if direction == "lower_is_better" else reference - wait
    relative = delta / abs(wait) if wait != 0 else (1.0 if delta > 0 else 0.0)
    passed = delta >= floor
    return {
        **base,
        "status": "passed" if passed else "failed",
        "absolute_delta": delta,
        "relative_delta": relative,
        "reason_code": (
            "material_headroom_passed"
            if passed
            else "material_headroom_below_threshold"
        ),
    }


def material_headroom_from_task_completion(
    completion: dict[str, Any],
) -> dict[str, Any]:
    """Translate task-native completion evidence into the common envelope."""
    evidence = completion.get("evidence") or {}
    if {
        "operations_scheduled",
        "operations_total",
    }.issubset(evidence):
        total = _finite(evidence.get("operations_total"))
        scheduled = _finite(evidence.get("operations_scheduled"))
        reference_remaining = (
            total - scheduled if total is not None and scheduled is not None else None
        )
        return evaluate_material_headroom(
            metric_name="unscheduled_operations",
            direction="lower_is_better",
            reference_value=reference_remaining,
            wait_value=total,
            threshold=1.0,
            native_units="operations",
            evidence_kind="completion_gap",
        )
    if {
        "actual_task_loss",
        "counterfactual_task_loss",
    }.issubset(evidence):
        return evaluate_material_headroom(
            metric_name="native_task_loss",
            direction="lower_is_better",
            reference_value=evidence.get("actual_task_loss"),
            wait_value=evidence.get("counterfactual_task_loss"),
            threshold=evidence.get("task_loss_reduction_threshold"),
            native_units="backend_native_loss",
            evidence_kind="counterfactual_replay",
        )
    if {
        "actual_voltage_violation_cost",
        "counterfactual_voltage_violation_cost",
    }.issubset(evidence):
        return evaluate_material_headroom(
            metric_name="voltage_violation_cost",
            direction="lower_is_better",
            reference_value=evidence.get("actual_voltage_violation_cost"),
            wait_value=evidence.get("counterfactual_voltage_violation_cost"),
            threshold=evidence.get("voltage_reduction_threshold"),
            native_units="violation_cost",
            evidence_kind="counterfactual_replay",
        )
    if {"actual_cost", "counterfactual_cost"}.issubset(evidence):
        return evaluate_material_headroom(
            metric_name="native_operational_cost",
            direction="lower_is_better",
            reference_value=evidence.get("actual_cost"),
            wait_value=evidence.get("counterfactual_cost"),
            threshold=evidence.get("materiality_threshold"),
            native_units="backend_native_cost",
            evidence_kind="counterfactual_replay",
        )
    return evaluate_material_headroom(
        metric_name="unknown",
        direction="lower_is_better",
        reference_value=None,
        wait_value=None,
        threshold=None,
        native_units="unknown",
        evidence_kind="missing",
    )


def build_traffic_native_signal_headroom_v2(
    *,
    baseline_metrics: dict[str, Any],
    baseline_repeat_metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
    reference_repeat_metrics: dict[str, Any],
    native_control_effect: Any,
    safety: Any,
) -> dict[str, Any]:
    """Pareto-safe materiality contract for native signal supervision."""
    fixed = {
        "network_vehicle_time_auc_s": 30.0,
        "controlled_lane_waiting_time_auc_s": 10.0,
        "controlled_lane_halting_auc": 5.0,
        "arrived_vehicles": 1.0,
    }
    units = {
        "network_vehicle_time_auc_s": "vehicle_seconds",
        "controlled_lane_waiting_time_auc_s": "vehicle_seconds",
        "controlled_lane_halting_auc": "vehicle_seconds",
        "arrived_vehicles": "vehicles",
    }
    congestion = tuple(fixed)[:3]
    required = (*congestion, "arrived_vehicles")
    missing = [
        name
        for name in required
        if _finite(baseline_metrics.get(name)) is None
        or _finite(baseline_repeat_metrics.get(name)) is None
        or _finite(reference_metrics.get(name)) is None
        or _finite(reference_repeat_metrics.get(name)) is None
    ]
    base = {
        "contract_id": TRAFFIC_NATIVE_SIGNAL_HEADROOM_V2,
        "status": "held",
        "reason_code": "traffic_headroom_metric_missing",
        "objective_passed": False,
        "throughput_only": False,
        "adverse_regression_passed": False,
        "objective_components": {},
        "guardrails": {},
        "baseline_repeat_drift": {},
        "fixed_thresholds": fixed,
        "effective_thresholds": {},
        "native_units": units,
        "missing_metrics": missing,
        "missing_safety_metrics": [],
    }
    if missing:
        return base

    required_safety = (
        "collision_count",
        "emergency_braking_count",
        "teleport_count",
    )
    safety_evidence = safety if isinstance(safety, dict) else {}
    safety_evidence = {
        **safety_evidence,
        "emergency_braking_count": safety_evidence.get(
            "emergency_braking_count",
            safety_evidence.get("emergency_braking"),
        ),
    }
    missing_safety = [
        name
        for name in required_safety
        if _finite(safety_evidence.get(name)) is None
    ]
    if missing_safety:
        return {
            **base,
            "reason_code": "safety_evidence_missing",
            "missing_safety_metrics": missing_safety,
        }

    drift = {
        name: abs(
            float(baseline_metrics[name])
            - float(baseline_repeat_metrics[name])
        )
        for name in required
    }
    effective = {
        name: max(fixed[name], 2.0 * drift[name])
        for name in required
    }
    improvements = {
        name: float(baseline_metrics[name])
        - float(reference_metrics[name])
        for name in congestion
    }
    improvements["arrived_vehicles"] = (
        float(reference_metrics["arrived_vehicles"])
        - float(baseline_metrics["arrived_vehicles"])
    )
    unfinished = max(
        _finite(baseline_metrics.get("minimum_expected_vehicles_end"))
        or 0.0,
        _finite(reference_metrics.get("minimum_expected_vehicles_end"))
        or 0.0,
    )
    if unfinished > 0 and improvements["arrived_vehicles"] != 0:
        return {
            **base,
            "reason_code": "traffic_headroom_right_censored",
            "missing_metrics": [],
            "missing_safety_metrics": [],
            "right_censoring": {
                "minimum_expected_vehicles_end": unfinished,
                "arrived_vehicles_difference": improvements[
                    "arrived_vehicles"
                ],
            },
        }
    objective_components = {
        name: {
            "improvement": improvements[name],
            "effective_threshold": effective[name],
            "passed": improvements[name] >= effective[name],
        }
        for name in congestion
    }
    objective_passed = any(
        row["passed"] for row in objective_components.values()
    )
    throughput_only = (
        not objective_passed
        and improvements["arrived_vehicles"]
        >= effective["arrived_vehicles"]
    )
    congestion_guardrails = {
        name: improvements[name] >= -effective[name]
        for name in congestion
    }
    throughput_guardrail = improvements["arrived_vehicles"] > -1.0
    safety_violation_count = max(
        int(baseline_metrics.get("safety_violation_count") or 0),
        int(reference_metrics.get("safety_violation_count") or 0),
    )
    clearance_violation_count = max(
        int(baseline_metrics.get("clearance_violation_count") or 0),
        int(reference_metrics.get("clearance_violation_count") or 0),
    )
    safety_ok = (
        all(float(safety_evidence[name]) == 0 for name in required_safety)
        and safety_violation_count == 0
        and clearance_violation_count == 0
    )
    adverse_ok = (
        all(congestion_guardrails.values())
        and throughput_guardrail
    )
    control_ok = bool(native_control_effect)
    reason = "traffic_headroom_passed"
    status = "passed"
    if not safety_ok:
        status = "failed"
        reason = "traffic_safety_violation_observed"
    elif not adverse_ok:
        status = "failed"
        reason = "traffic_headroom_adverse_regression"
    elif throughput_only:
        status = "failed"
        reason = "traffic_headroom_throughput_only"
    elif not objective_passed or not control_ok:
        status = "failed"
        reason = "traffic_headroom_below_threshold"
    return {
        **base,
        "status": status,
        "reason_code": reason,
        "objective_passed": objective_passed,
        "throughput_only": throughput_only,
        "adverse_regression_passed": adverse_ok,
        "objective_components": objective_components,
        "guardrails": {
            "congestion": congestion_guardrails,
            "throughput": throughput_guardrail,
            "safety": safety_ok,
            "native_control_effect": control_ok,
        },
        "baseline_repeat_drift": drift,
        "effective_thresholds": effective,
        "missing_metrics": [],
        "missing_safety_metrics": [],
        "improvements": improvements,
    }
