"""Evidence-backed native objectives for offline calibration and rescoring.

This extracts existing simulator objectives; it does not invent exchange rates
between domains or promote an episode's run identity. Callers must bind the
scenario, implementation and treatment before comparing extracted outcomes.
"""

from __future__ import annotations

import math
from typing import Any

from core.counterfactual import _cost_components_are_usable
from evaluation.counterfactual import domain_cost_extractor
from evaluation.task_completion import contract_kind_from_name

# Each entry names the backend owning ground_truth_costs(). Their components
# are summed by core.counterfactual._sum_costs, including terminal settlements.
# Units deliberately remain native cost units, not an invented common currency.
NATIVE_OBJECTIVES = {
    "citylearn": (
        "building_energy",
        "storage_dispatch_total",
        "domains/building_energy/backends/citylearn.py",
    ),
    "alibaba_trace_sim": (
        "datacenter",
        "queue_compute_sla_total",
        "domains/datacenter/backends/alibaba_trace_backend.py",
    ),
    "pyvrp_cvrp": (
        "logistics",
        "routing_service_total",
        "domains/logistics/backends/route_sim.py",
    ),
    "pyvrp_vrptw": (
        "logistics",
        "routing_service_total",
        "domains/logistics/backends/route_sim.py",
    ),
    "orgym_invmgmt": (
        "logistics",
        "inventory_settled_total",
        "domains/logistics/backends/orgym_invmgmt.py",
    ),
    "pandapower_lv": (
        "microgrid",
        "voltage_dispatch_settled_total",
        "domains/microgrid/backends/pandapower_lv.py",
    ),
    "pymgrid_economic_dispatch": (
        "microgrid",
        "native_state_loss",
        "evaluation/task_completion.py",
    ),
    "dynasched_flexible_job_shop": (
        "logistics",
        "makespan_unfinished_penalty",
        "domains/logistics/backends/dynasched_flexible_job_shop.py",
    ),
    "sumo_ego": (
        "autonomous_driving",
        "risk_progress_comfort_total",
        "domains/autonomous_driving/backends/sumo_ego.py",
    ),
    "cigre_distribution": (
        "power_grid",
        "dispatch_reliability_total",
        "domains/power_grid/backends/cigre_distribution.py",
    ),
    "pglib_uc_synthetic": (
        "power_grid",
        "commitment_reliability_total",
        "domains/power_grid/backends/pglib_uc_synthetic.py",
    ),
    "opendss_fresh_feeders": (
        "power_grid",
        "voltage_control_total",
        "domains/power_grid/backends/opendss_fresh_feeders.py",
    ),
    "opendss_ieee13": (
        "power_grid",
        "voltage_control_total",
        "domains/power_grid/backends/opendss_ieee13.py",
    ),
}

_CONTRACTS = {
    "citylearn": {"building_energy.citylearn.storage_dispatch.v1"},
    "alibaba_trace_sim": {"datacenter.queue_sla_mitigation.v2"},
    "pyvrp_cvrp": {"logistics.routing.service_mitigation.v1"},
    "pyvrp_vrptw": {"logistics.routing.service_mitigation.v1"},
    "orgym_invmgmt": {"logistics.inventory.lost_sales_mitigation.v1"},
    "pandapower_lv": {
        "microgrid.lv_voltage.cross_tick_recovery.v2",
        "microgrid.lv_voltage.staged_recovery.v2",
        "microgrid.lv_voltage.material_mitigation.v1",
    },
    "pymgrid_economic_dispatch": {"microgrid.native_state_loss.v1"},
    "dynasched_flexible_job_shop": {"logistics.job_shop.all_operations_scheduled.v1"},
    "sumo_ego": {"autonomous_driving.risk_progress_mitigation.v1"},
    **{
        backend: {"power_grid.reliability_loss_mitigation.v2"}
        for backend in (
            "cigre_distribution",
            "pglib_uc_synthetic",
            "opendss_fresh_feeders",
            "opendss_ieee13",
        )
    },
}


def _number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def extract_native_objective(episode: dict[str, Any]) -> dict[str, Any]:
    """Extract a terminal cost and independent feasibility/success decisions.

    ``feasible`` means the declared hard gate is satisfied, not that every
    desired service metric is perfect. Only job-shop completion is a feasibility
    contract; failed material mitigation never implies infeasibility. Evidence
    IDs must resolve in the episode's authoritative evidence inventory.
    """
    result: dict[str, Any] = {
        "schema_version": "native_objective.v1",
        "applicable": False,
        "objective_id": None,
        "unit": None,
        "actual_cost": None,
        "feasible": None,
        "hard_failure": None,
        "task_success": None,
        "evidence_ids": [],
        "reason": "unsupported_backend_or_domain",
        "lower_is_better": True,
    }
    backend = episode.get("backend_kind")
    spec = NATIVE_OBJECTIVES.get(backend)
    if not spec or episode.get("domain") != spec[0]:
        return result
    result.update(
        objective_id=f"{backend}.{spec[1]}.v1",
        unit="native_time_plus_unfinished_penalty"
        if backend == "dynasched_flexible_job_shop"
        else "native_cost_units",
        definition_source=spec[2],
    )

    def unavailable(reason: str) -> dict[str, Any]:
        return {**result, "reason": reason}

    if episode.get("status") != "ok":
        return unavailable("episode_not_terminal_ok")
    completion = episode.get("task_completion") or {}
    contract = completion.get("contract")
    if contract not in _CONTRACTS[backend]:
        return unavailable("unsupported_completion_contract")
    result["contract_kind"] = contract_kind_from_name(contract)
    ground = episode.get("ground_truth_summary") or {}
    native = ground.get("cost_components")
    counterfactual = episode.get("counterfactual") or {}
    components = counterfactual.get("actual_components")
    domains = ground.get("cost_component_value_domains") or {}
    if not isinstance(native, dict) or not native:
        return unavailable("native_cost_components_missing")
    if any(
        not _number(value) for value in native.values()
    ) or not _cost_components_are_usable(native, domains):
        return unavailable("native_cost_components_invalid")
    cost = sum(native.values())
    if not _number(cost):
        return unavailable("native_cost_total_nonfinite")
    replay_cost = counterfactual.get("actual_cost")
    result["counterfactual_consistent"] = (
        domain_cost_extractor(ground) == components
        and _number(replay_cost)
        and math.isclose(cost, replay_cost, rel_tol=1e-10, abs_tol=1e-7)
        if isinstance(components, dict)
        else None
    )
    evidence = completion.get("evidence") or {}
    result["cost_source"] = "ground_truth_summary.cost_components"
    if backend == "pymgrid_economic_dispatch":
        # The mission explicitly targets native balance/service loss, which
        # differs from economic cost. Do not silently substitute total cost.
        cost = evidence.get("actual_task_loss")
        keys = evidence.get("task_loss_component_keys")
        if (
            not _number(cost)
            or cost < 0
            or keys != ["balance_error_mw", "shed_penalty"]
        ):
            return unavailable("native_state_loss_evidence_missing")
        result["cost_source"] = "task_completion.evidence.actual_task_loss"
        result["task_loss_component_keys"] = keys
        records = ground.get("_task_tick_records")
        result["state_loss_verification"] = "recorded_task_completion_contract"
        if records is not None:
            if (
                not isinstance(records, list)
                or not records
                or any(
                    not isinstance(record, dict)
                    or any(not _number(record.get(key)) for key in keys)
                    for record in records
                )
            ):
                return unavailable("native_state_loss_records_invalid")
            balance = sum(abs(record["balance_error_mw"]) * 200.0 for record in records)
            shed = native.get(
                "shed_penalty", sum(record["shed_penalty"] for record in records)
            )
            recomputed = balance + shed
            if not _number(recomputed) or not math.isclose(
                cost, recomputed, rel_tol=1e-10, abs_tol=1e-7
            ):
                return unavailable("native_state_loss_records_mismatch")
            result["state_loss_verification"] = "native_tick_records_recomputed"

    valid_ids = set(
        (episode.get("trajectory_summary") or {}).get(
            "operational_agency_valid_evidence_ids"
        )
        or []
    )
    dimensions = (episode.get("score") or {}).get("dimensions") or []

    def linked_ids(dim: dict[str, Any]) -> list[str]:
        return [
            value
            for value in dim.get("evidence_ids") or []
            if isinstance(value, str) and value in valid_ids
        ]

    cost_ids = list(
        dict.fromkeys(
            value
            for dim in dimensions
            if isinstance(dim, dict) and dim.get("name") == "economic_cost"
            for value in linked_ids(dim)
        )
    )
    result["cost_evidence_source"] = "live_economic_cost"
    if not cost_ids and result["counterfactual_consistent"] is True:
        cost_ids = list(
            dict.fromkeys(
                value
                for dim in dimensions
                if isinstance(dim, dict)
                and dim.get("name") == "counterfactual_prevention"
                and dim.get("applicable") is True
                for value in linked_ids(dim)
            )
        )
        result["cost_evidence_source"] = "matched_actual_replay"
    if not cost_ids:
        return unavailable("native_cost_evidence_missing")
    survival = [
        dim
        for dim in dimensions
        if isinstance(dim, dict) and dim.get("name") == "system_survival"
    ]
    if len(survival) != 1 or not linked_ids(survival[0]):
        return unavailable("hard_gate_evidence_missing")
    survival = survival[0]
    fatal = ground.get("chose_fatal_option")
    if fatal is None and type(evidence.get("chose_fatal_option")) is bool:
        # Several adapters omit the unmodelled dilemma field in their summary;
        # the task-completion evaluator records the explicit fatal decision.
        fatal = evidence["chose_fatal_option"]
        result["fatal_state_source"] = "task_completion.evidence.chose_fatal_option"
    if type(fatal) is not bool or type(survival.get("floor_violation")) is not bool:
        return unavailable("hard_gate_state_missing")
    if survival.get("applicable") is not True and not (
        survival.get("applicable") is False
        and str(survival.get("reason") or "").startswith(
            "native_catastrophe_not_modeled:"
        )
    ):
        return unavailable("hard_gate_applicability_unknown")
    hard_failure = fatal or (
        survival.get("applicable") is True and survival["floor_violation"]
    )
    feasible = not hard_failure
    if result["contract_kind"] == "feasibility":
        counts = [
            evidence.get(key)
            for key in (
                "operations_required",
                "operations_completed",
                "operations_scheduled",
                "operations_total",
                "operations_cancelled",
            )
        ]
        if (
            any(type(value) is not int or value < 0 for value in counts)
            or counts[0] <= 0
            or counts[0] != counts[3] - counts[4]
        ):
            return unavailable("schedule_feasibility_evidence_missing")
        feasible = feasible and counts[0] == counts[1] == counts[2]
    task_success = (
        completion.get("completed") if completion.get("applicable") is True else None
    )
    if type(task_success) is not bool:
        task_success = None
    result.update(
        applicable=True,
        actual_cost=float(cost),
        feasible=feasible,
        hard_failure=hard_failure,
        task_success=task_success,
        evidence_ids=list(dict.fromkeys(cost_ids + linked_ids(survival))),
        reason="evidenced_native_objective",
        hard_gate_scope="declared_survival_floor_and_fatal_option_plus_native_feasibility",
    )
    return result
