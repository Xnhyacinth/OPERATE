"""Optional existing-wait diagnostics; no score, replay or harm attribution.

The caller authenticates the snapshot and binds its counterfactual. Economic
components describe recorded losses, not complete service or safety equivalence.
"""

from __future__ import annotations

import math

from core.counterfactual import _cost_components_are_usable

_SERVICE_KEYS = {
    "alibaba_trace_sim": (
        "queue_wait_cost",
        "sla_violation_cost",
        "unfinished_work_penalty",
    ),
    "pyvrp_cvrp": ("unmet_demand_cost", "drop_order_penalty"),
    "pyvrp_vrptw": ("unmet_demand_cost", "drop_order_penalty"),
    "orgym_invmgmt": ("inventory_lost_sales_penalty",),
    "pymgrid_economic_dispatch": ("balance_error_cost", "shed_penalty"),
    "pandapower_lv": ("voltage_violation_cost", "shed_penalty"),
    "sumo_ego": (
        "collision_cost",
        "road_departure_cost",
        "risk_exposure_cost",
        "route_delay_cost",
        "comfort_cost",
        "mrm_cost",
        "mrm_failure_cost",
        "shield_intervention_cost",
    ),
}


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _close(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-7)


def evaluate_wait_diagnostics(
    snapshot_inputs: dict,
    *,
    counterfactual_bound: bool,
    native_outcome: dict,
    backend_kind: str,
) -> dict:
    """Compare bound cost components without treating cost increases as harm."""
    result = dict(
        applicable=False,
        main_weight=0,
        score=None,
        reason="existing_bound_wait_unavailable",
        components={},
        economic_cost_difference=None,
        measured_service_components=dict(
            applicable=False, relation=None, missing_keys=[]
        ),
        overall_no_harm=None,
        no_harm_reason="baseline_safety_and_complete_service_obligations_unproven",
    )
    cf = snapshot_inputs.get("counterfactual_report")
    if not (
        counterfactual_bound is True
        and isinstance(cf, dict)
        and cf.get("applicable") is True
        and cf.get("masking_policy") == "wait_only"
        and native_outcome.get("applicable") is True
        and native_outcome.get("counterfactual_consistent") is True
    ):
        return result
    actual, wait = cf.get("actual_components"), cf.get("counterfactual_components")
    ground = snapshot_inputs.get("cost_components")
    domains = cf.get("cost_component_value_domains", {})
    if not (
        isinstance(actual, dict)
        and actual
        and isinstance(wait, dict)
        and isinstance(ground, dict)
        and set(actual) == set(wait) == set(ground)
        and all(isinstance(key, str) for key in actual)
        and all(
            _finite(value)
            for values in (actual, wait, ground)
            for value in values.values()
        )
        and all(_close(actual[key], ground[key]) for key in actual)
        and all(
            _cost_components_are_usable(values, domains)
            for values in (actual, wait, ground)
        )
        and _finite(cf.get("actual_cost"))
        and _finite(cf.get("counterfactual_cost"))
        and _close(sum(actual.values()), cf["actual_cost"])
        and _close(sum(wait.values()), cf["counterfactual_cost"])
    ):
        return {**result, "reason": "wait_components_or_totals_invalid"}
    delta = cf["counterfactual_cost"] - cf["actual_cost"]
    differences = {key: wait[key] - actual[key] for key in actual}
    if not _finite(delta) or not all(_finite(value) for value in differences.values()):
        return {**result, "reason": "wait_difference_nonfinite"}
    result.update(
        applicable=True,
        reason="existing_bound_component_comparison",
        components={
            key: dict(actual=actual[key], wait=wait[key], reduction=differences[key])
            for key in sorted(actual)
        },
        economic_cost_difference=dict(
            actual=cf["actual_cost"],
            wait=cf["counterfactual_cost"],
            reduction=delta,
            unit="recorded_native_cost_component_units",
            is_native_objective=backend_kind != "pymgrid_economic_dispatch",
        ),
    )
    keys = _SERVICE_KEYS.get(backend_kind, ())
    missing = sorted(set(keys) - set(actual))
    service = result["measured_service_components"]
    service.update(
        keys=list(keys),
        missing_keys=missing,
        reason="missing_service_components"
        if missing
        else "no_service_component_contract",
    )
    if keys and not missing:
        signs = {
            0 if _close(actual[key], wait[key]) else (1 if differences[key] > 0 else -1)
            for key in keys
        }
        relation = (
            "tradeoff"
            if 1 in signs and -1 in signs
            else "improved"
            if 1 in signs
            else "worsened"
            if -1 in signs
            else "equal"
        )
        service.update(
            applicable=True,
            relation=relation,
            reason="measured_components_only_not_complete_service_or_safety",
        )
    return result
