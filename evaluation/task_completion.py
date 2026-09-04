"""Domain-backed task completion contracts for leaderboard-eligible episodes."""

from __future__ import annotations

import math
from typing import Any

_MICROGRID_NATIVE_TASK_LOSS_KEYS = (
    "balance_error_mw",
    "shed_penalty",
)

_POWER_GRID_TASK_FAMILIES = frozenset(
    {
        "acopf_dispatch_24h",
        "critical_winter_peak",
        "daily_ops_24h",
        "daily_ops_real_forecast_24h",
        "distribution_volt_var",
        "distribution_volt_var_oberrhein",
        "opendss_fresh_feeders_solar_ramp",
        "opendss_fresh_feeders_volt_var",
        "opendss_infrastructure_feeder_control",
        "opendss_ieee13_volt_var",
        "reserve_stress_24h",
        "simbench_lv_timeseries_control",
        "simbench_mv_rural_timeseries_control",
        "simbench_mv_rural1_timeseries_control",
        "simbench_mv_semiurban_timeseries_control",
        "simbench_mv_semiurban1_timeseries_control",
        "simbench_mv_urban_timeseries_control",
        "simbench_mv_commercial_timeseries_control",
        "storm_emergency_6h",
        "storm_emergency_6h_idf2023",
        "storm_l2rpn_icaps2021",
        "storm_l2rpn_neurips2020_track1",
        "storm_l2rpn_neurips2020_track2",
        "storm_l2rpn_sandbox",
        "storm_wcci_2022",
        "wind_uncertainty_24h",
    }
)

_TRAFFIC_TASK_FAMILIES = frozenset(
    {
        "construction_lane_reallocation",
        "coordinated_overflow_relief",
        "demand_surge_metering",
        "detector_dropout_recovery",
        "emergency_corridor_preemption",
        "freight_corridor_pressure",
        "incident_response",
        "peak_spillback_recovery",
        "route_assignment",
        "school_zone_activation",
        "signal_coordination",
        "signal_failure_recovery",
        "transit_signal_priority",
        "vip_priority_dilemma",
        "work_zone_detour_recovery",
    }
)

_DATACENTER_TASK_FAMILIES = frozenset(
    {
        "gpu_cluster_queue_control",
        "gpu_cluster_sla_control",
        "gpu_cluster_spot_sla_control",
        "gpu_sharing_placement_and_rescheduling",
    }
)

_DISTINCT_TASK_REQUIREMENT_KEYS = frozenset(
    {
        "min_distinct_control_ticks",
        "min_distinct_physical_tools",
    }
)

_PROCESS_CAPABILITY_CHECK_KEYS = (
    "native_control_requirements_met",
    "ordered_tool_milestones_met",
    "strategy_reversals_met",
    "response_windows_met",
    "effective_control_tick_floor_met",
    "cross_tick_control_milestones_met",
    "plan_reversal_observed",
    "declared_control_axis_observed",
    "recovery_completed",
    "recovery_sequence_met",
    "paid_inspection_met",
    "preventive_action_met",
    "decision_epoch_floor_met",
    "observation_tools_met",
)


def separate_task_outcome_and_process(
    completion: dict[str, Any],
    *,
    scenario: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose outcome success separately from native process capability.

    Counterfactual outcome improvement is the task-completion contract.  It
    must not also imply that a scenario's ordered tools, response windows, or
    strategy-reversal requirements were satisfied.  This helper preserves the
    existing completion payload and adds explicit, independently aggregatable
    fields.
    """
    result = dict(completion)
    evidence = result.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    candidate_checks = {
        key: evidence[key]
        for key in _PROCESS_CAPABILITY_CHECK_KEYS
        if isinstance(evidence.get(key), bool)
    }
    backend_config = (
        scenario.get("backend_config")
        if isinstance(scenario, dict)
        and isinstance(scenario.get("backend_config"), dict)
        else {}
    )
    declared_requirements = (
        backend_config.get("task_requirements")
        or backend_config.get("task_contract")
        or evidence.get("requirements")
        or {}
    )
    non_native_checks = {
        key: value
        for key, value in candidate_checks.items()
        if key != "native_control_requirements_met"
    }
    process_declared = bool(declared_requirements) or bool(non_native_checks)
    checks = candidate_checks if process_declared else {}
    result["outcome_task_completed"] = bool(result.get("completed", False))
    result["process_capability_applicable"] = bool(checks)
    result["process_capability_satisfied"] = all(checks.values()) if checks else None
    result["process_capability_checks"] = checks
    return result


def _finite_float(value: Any) -> float:
    """Return a finite numeric value for native task-loss accounting."""
    try:
        parsed = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _validate_distinct_task_requirement_keys(scenario: dict[str, Any]) -> None:
    backend_config = scenario.get("backend_config") or {}
    requirements = backend_config.get("task_requirements") or {}
    if not isinstance(requirements, dict):
        return
    unknown = sorted(
        str(key)
        for key in requirements
        if str(key) not in _DISTINCT_TASK_REQUIREMENT_KEYS
        and str(key).startswith(("min_distinct_", "minimum_distinct_"))
    )
    if unknown:
        raise ValueError(f"unknown task requirement key: {', '.join(unknown)}")


def _milestone_action_predicate_passes(
    milestone: dict[str, Any],
    *,
    record: dict[str, Any] | None = None,
    tool_name: str = "",
    tick: int = 0,
    records: Any = (),
) -> bool:
    """Check exact arguments and an optional predicate for one realized call."""
    if record is None:
        matching = [
            row
            for row in records
            if isinstance(row, dict)
            and str(row.get("tool_name") or row.get("name") or "") == tool_name
            and int(_finite_float(row.get("tick"))) == tick
            and row.get("ok") is True
            and row.get("state_changing") is True
        ]
        return any(
            _milestone_action_predicate_passes(
                milestone,
                record={
                    "request_args": row.get("args") or {},
                    "request_result": row.get("result") or row.get("payload") or {},
                },
            )
            for row in matching
        )
    expected_args = milestone.get("args")
    if expected_args is not None:
        request_args = record.get("request_args")
        if not isinstance(expected_args, dict) or not isinstance(request_args, dict):
            return False
        if any(request_args.get(key) != value for key, value in expected_args.items()):
            return False
    predicate = milestone.get("action_predicate")
    if not predicate:
        return True
    if not isinstance(predicate, dict):
        return False
    kind = str(predicate.get("kind") or "")
    result = record.get("effect_result") or record.get("request_result") or {}
    if not isinstance(result, dict):
        return False
    if kind != "power_grid_shed_load":
        # Unknown predicates are deliberately not treated as satisfied.
        return False
    excluded = {
        str(value) for value in predicate.get("excluded_stakeholder_classes") or []
    }
    stakeholder = str(result.get("stakeholder_class") or "")
    criticality = _finite_float(result.get("criticality"))
    shed_mw = _finite_float(result.get("shed_mw") or result.get("mw"))
    max_criticality = predicate.get("max_criticality")
    min_shed_mw = predicate.get("min_shed_mw")
    if stakeholder in excluded:
        return False
    if max_criticality is not None and criticality > _finite_float(max_criticality):
        return False
    return not (min_shed_mw is not None and shed_mw < _finite_float(min_shed_mw))


def _physical_control_identities(
    scenario: dict[str, Any], controls: dict[str, Any]
) -> tuple[list[str], str, int]:
    """Return runtime-proven physical controls under the signed scenario contract."""
    tools = sorted(
        {str(tool) for tool in controls.get("distinct_physical_tools") or []}
    )
    backend_config = scenario.get("backend_config") or {}
    contract = backend_config.get("actuator_endpoint_contract") or {}
    if not contract:
        return tools, "tool_name", 0
    if not isinstance(contract, dict):
        return [], "invalid_endpoint_contract", 0
    mode = str(contract.get("endpoint_identity") or "")
    tool_name = str(contract.get("tool_name") or "")
    if mode != "tool_name|tls_id" or not tool_name:
        return [], "invalid_endpoint_contract", 0
    allowed = {
        str(endpoint)
        for endpoint in contract.get("candidate_endpoint_ids") or []
        if str(endpoint).startswith(f"{tool_name}|")
    }
    observed = {
        str(endpoint)
        for endpoint in controls.get("distinct_physical_actuator_endpoints") or []
    }
    minimum = int(contract.get("minimum_distinct_endpoints") or 0)
    return sorted(observed & allowed), mode, minimum


def _microgrid_native_task_loss(
    components: Any,
    records: Any,
    keys: tuple[str, ...],
) -> float:
    """Extract the declared pymgrid loss from costs or native tick records."""
    component_map = components if isinstance(components, dict) else {}
    record_rows = (
        [row for row in records if isinstance(row, dict)]
        if isinstance(records, list | tuple)
        else []
    )
    total = 0.0
    for key in keys:
        if key == "balance_error_mw":
            if record_rows:
                total += sum(
                    abs(_finite_float(row.get(key))) * 200.0 for row in record_rows
                )
            elif key in component_map:
                total += abs(_finite_float(component_map[key])) * 200.0
            else:
                # Older pymgrid cost summaries expose only this cost alias.
                total += _finite_float(component_map.get("balance_error_cost"))
            continue
        if key in component_map:
            total += _finite_float(component_map[key])
        else:
            total += sum(_finite_float(row.get(key)) for row in record_rows)
    return total


def task_completion_contract(domain: str, family: str) -> str:
    """Return the declared native completion contract for a scenario cell."""
    if domain == "building_energy" and family in {
        "citylearn_der_storage_control",
        "citylearn_native_load_solar_storage_control",
    }:
        return "building_energy.citylearn.storage_dispatch.v1"
    if domain == "microgrid" and family == "microgrid_lv_voltage_recovery_10h":
        return "microgrid.lv_voltage.cross_tick_recovery.v2"
    if domain == "microgrid" and family == "microgrid_lv_voltage_staged_6h":
        return "microgrid.lv_voltage.staged_recovery.v2"
    if domain == "microgrid" and family == "microgrid_lv_voltage_6h":
        return "microgrid.lv_voltage.material_mitigation.v1"
    if domain == "microgrid" and family == "microgrid_economic_dispatch_24h":
        return "microgrid.native_state_loss.v1"
    if domain == "logistics" and family == "job_shop_dispatch":
        return "logistics.job_shop.all_operations_scheduled.v1"
    if domain == "logistics" and family in {"cvrp_dispatch", "vrptw_dispatch"}:
        return "logistics.routing.service_mitigation.v1"
    if domain == "logistics" and family == "inventory_replenishment":
        return "logistics.inventory.lost_sales_mitigation.v1"
    if domain == "power_grid" and family in _POWER_GRID_TASK_FAMILIES:
        return "power_grid.reliability_loss_mitigation.v2"
    if domain == "traffic" and family in _TRAFFIC_TASK_FAMILIES:
        return "traffic.travel_delay_mitigation.v1"
    if domain == "autonomous_driving" and family in {
        "sustained_highway_risk_supervision",
        "cut_in_prevention_and_emergency",
        "odd_degradation_mrm_recovery",
    }:
        return "autonomous_driving.risk_progress_mitigation.v1"
    if domain == "datacenter" and family in _DATACENTER_TASK_FAMILIES:
        return "datacenter.queue_sla_mitigation.v2"
    return "unsupported"


def evaluate_task_completion(
    *,
    scenario: dict[str, Any],
    ground_truth: dict[str, Any],
    counterfactual: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any]:
    """Return a fail-closed, machine-readable completion decision.

    Static job-shop tasks have an intrinsic feasibility condition: every
    operation must be scheduled. Dynamic control tasks must demonstrate a
    material improvement over deterministic no-action replay. This common
    floor is intentionally stricter than process completion; backend-specific
    contracts can replace it as richer native objectives are added.
    """
    _validate_distinct_task_requirement_keys(scenario)
    family = str(scenario.get("family") or "unknown")
    domain = str(scenario.get("domain") or "unknown")
    fatal = bool(ground_truth.get("chose_fatal_option", False))
    survival_floor = any(
        bool(dim.get("floor_violation"))
        for dim in score.get("dimensions", [])
        if (
            isinstance(dim, dict)
            and dim.get("name") == "system_survival"
            # A floor is meaningful only for an applicable dimension.  Keep
            # the historical fail-closed behavior for legacy score payloads
            # that omit the applicability field entirely.
            and dim.get("applicable", True)
        )
    )
    safe = not fatal and not survival_floor

    contract_name = task_completion_contract(domain, family)
    if contract_name == "unsupported":
        actual_cost = _finite_float(counterfactual.get("actual_cost"))
        return {
            "schema_version": "1.0",
            "contract": contract_name,
            "applicable": False,
            "completed": False,
            "reason_code": "unsupported_task_contract",
            "evidence": {
                "actual_cost": actual_cost,
                "counterfactual_cost": _finite_float(
                    counterfactual.get("counterfactual_cost")
                ),
                "prevented_loss": _finite_float(counterfactual.get("prevented_loss")),
                "materiality_threshold": max(1.0, abs(actual_cost) * 0.001),
                "survival_floor_violation": survival_floor,
                "chose_fatal_option": fatal,
            },
        }

    if domain == "building_energy" and family in {
        "citylearn_der_storage_control",
        "citylearn_native_load_solar_storage_control",
    }:
        applicable = bool(counterfactual.get("applicable", False))
        actual_cost = _finite_float(counterfactual.get("actual_cost"))
        counterfactual_cost = _finite_float(
            counterfactual.get("counterfactual_cost")
        )
        prevented_loss = _finite_float(counterfactual.get("prevented_loss"))
        materiality_threshold = max(1e-9, abs(counterfactual_cost) * 0.001)
        backend_config = scenario.get("backend_config") or {}
        requirements = backend_config.get("task_requirements") or {}
        controls = ground_truth.get("control_summary") or {}
        effective_ticks = sorted(
            {int(tick) for tick in controls.get("effective_control_ticks") or []}
        )
        physical_endpoints = sorted(
            {
                str(endpoint)
                for endpoint in controls.get("distinct_physical_endpoints") or []
            }
        )
        response_windows = [
            window
            for window in controls.get("response_windows") or []
            if isinstance(window, dict)
        ]
        valid_response_windows = [
            window
            for window in response_windows
            if str(window.get("event_id") or "")
            and window.get("event_observed") is True
            and bool(window.get("event_evidence_id"))
            and bool(window.get("expected_control_policy"))
            and window.get("direction_met") is True
            and bool(window.get("control_ticks"))
        ]
        responded_event_ids = sorted(
            {str(window["event_id"]) for window in valid_response_windows}
        )
        responded_windows = len(responded_event_ids)
        strategy_reversals = int(controls.get("strategy_reversal_count") or 0)
        tool_ticks = {
            str(tool): sorted({int(tick) for tick in ticks or []})
            for tool, ticks in (controls.get("tool_ticks") or {}).items()
        }
        selected_milestone_ticks: list[int] = []
        selected_milestone_tools: list[str] = []
        selected_milestone_event_ids: list[str | None] = []
        selected_milestone_control_policies: list[str | None] = []
        previous_tick = -1
        ordered_met = True
        for milestone in requirements.get("ordered_tool_milestones") or []:
            tool = str(milestone.get("tool") or "")
            allowed_tools = {
                str(value) for value in milestone.get("tools") or [] if str(value)
            }
            if tool:
                allowed_tools.add(tool)
            if milestone.get("any_state_changing") is True:
                allowed_tools.update(tool_ticks)
            earliest = int(milestone.get("not_before_tick", 0) or 0)
            latest = int(milestone.get("not_after_tick", 10**9) or 10**9)
            expected_event_id = str(milestone.get("event_id") or "")
            expected_control_policy = str(
                milestone.get("expected_control_policy") or ""
            )
            bound_control_pairs: set[tuple[str, int]] | None = None
            if expected_event_id or expected_control_policy:
                bound_control_pairs = set()
                for window in valid_response_windows:
                    if expected_event_id and (
                        str(window.get("event_id") or "") != expected_event_id
                    ):
                        continue
                    if expected_control_policy and (
                        str(window.get("expected_control_policy") or "")
                        != expected_control_policy
                    ):
                        continue
                    native_control = str(window.get("native_control") or "")
                    for candidate_tool in allowed_tools:
                        if native_control and native_control != candidate_tool:
                            continue
                        bound_control_pairs.update(
                            (candidate_tool, int(tick))
                            for tick in window.get("control_ticks") or []
                        )

            candidates = sorted(
                (tick, candidate_tool)
                for candidate_tool in allowed_tools
                for tick in tool_ticks.get(candidate_tool, [])
                if tick > previous_tick
                and earliest <= tick <= latest
                and (
                    bound_control_pairs is None
                    or (candidate_tool, tick) in bound_control_pairs
                )
            )
            if not candidates:
                ordered_met = False
                break
            previous_tick, selected_tool = candidates[0]
            selected_milestone_ticks.append(previous_tick)
            selected_milestone_tools.append(selected_tool)
            selected_milestone_event_ids.append(expected_event_id or None)
            selected_milestone_control_policies.append(expected_control_policy or None)
        native_requirements_met = (
            len(effective_ticks)
            >= int(requirements.get("min_distinct_control_ticks") or 0)
            and len(physical_endpoints)
            >= int(requirements.get("min_distinct_physical_tools") or 0)
            and strategy_reversals
            >= int(requirements.get("min_strategy_reversals") or 0)
            and responded_windows >= int(requirements.get("min_response_windows") or 0)
        )
        completed = applicable and safe and prevented_loss > materiality_threshold
        if completed:
            reason_code = "citylearn_storage_response_completed"
        elif not applicable:
            reason_code = "counterfactual_unavailable"
        elif not safe:
            reason_code = "unsafe_outcome"
        elif prevented_loss <= materiality_threshold:
            reason_code = "no_material_improvement_vs_no_action"
        else:
            reason_code = "insufficient_native_outcome_improvement"
        return {
            "schema_version": "1.0",
            "contract": task_completion_contract(domain, family),
            "applicable": applicable,
            "completed": completed,
            "reason_code": reason_code,
            "evidence": {
                "actual_cost": actual_cost,
                "counterfactual_cost": counterfactual_cost,
                "prevented_loss": prevented_loss,
                "materiality_threshold": materiality_threshold,
                "materiality_threshold_basis": "0.1_percent_of_no_action_cost",
                "native_control_requirements_met": native_requirements_met,
                "effective_control_ticks": effective_ticks,
                "distinct_physical_endpoints": physical_endpoints,
                "strategy_reversal_count": strategy_reversals,
                "tool_ticks": tool_ticks,
                "ordered_tool_milestones_met": ordered_met,
                "selected_milestone_ticks": selected_milestone_ticks,
                "selected_milestone_tools": selected_milestone_tools,
                "selected_milestone_event_ids": selected_milestone_event_ids,
                "selected_milestone_control_policies": (
                    selected_milestone_control_policies
                ),
                "responded_response_windows": responded_windows,
                "responded_event_ids": responded_event_ids,
                "response_windows": response_windows,
                "survival_floor_violation": survival_floor,
                "chose_fatal_option": fatal,
            },
        }

    if domain == "microgrid" and family in {
        "microgrid_lv_voltage_6h",
        "microgrid_lv_voltage_staged_6h",
        "microgrid_lv_voltage_recovery_10h",
    }:
        applicable = bool(counterfactual.get("applicable", False))
        actual_components = counterfactual.get("actual_components") or {}
        counterfactual_components = (
            counterfactual.get("counterfactual_components") or {}
        )
        actual_voltage_cost = float(
            actual_components.get("voltage_violation_cost") or 0.0
        )
        counterfactual_voltage_cost = float(
            counterfactual_components.get("voltage_violation_cost") or 0.0
        )
        voltage_reduction = counterfactual_voltage_cost - actual_voltage_cost
        voltage_threshold = max(1.0, counterfactual_voltage_cost * 0.05)
        prevented_loss = float(counterfactual.get("prevented_loss") or 0.0)
        actual_cost = float(counterfactual.get("actual_cost") or 0.0)
        materiality_threshold = max(1.0, abs(actual_cost) * 0.001)
        survival_scores = [
            float(dim.get("raw_score") or 0.0)
            for dim in score.get("dimensions", [])
            if isinstance(dim, dict)
            and dim.get("name") == "system_survival"
            and dim.get("applicable")
        ]
        catastrophic = (
            fatal
            or survival_floor
            or (bool(survival_scores) and max(survival_scores) <= 0.0)
        )
        material_mitigation = (
            applicable
            and not catastrophic
            and prevented_loss > materiality_threshold
            and voltage_reduction > voltage_threshold
        )
        contract_config = dict(
            (scenario.get("backend_config") or {}).get("task_contract") or {}
        )
        cross_tick_contract = family in {
            "microgrid_lv_voltage_staged_6h",
            "microgrid_lv_voltage_recovery_10h",
        } or contract_config.get("contract") in {
            "microgrid.lv_voltage.staged_recovery.v2",
            "microgrid.lv_voltage.cross_tick_recovery.v2",
        }
        phase_reductions: dict[str, int] = {}
        phases_completed = True
        distinct_control_ticks: list[int] = []
        plan_reversal_observed = False
        if cross_tick_contract:
            actual_records = {
                int(record.get("tick")): record
                for record in ground_truth.get("_task_tick_records") or []
                if isinstance(record, dict) and record.get("tick") is not None
            }
            counterfactual_records = {
                int(record.get("tick")): record
                for record in counterfactual.get("_counterfactual_task_tick_records")
                or []
                if isinstance(record, dict) and record.get("tick") is not None
            }
            phase_ticks = [
                int(tick) for tick in contract_config.get("phase_ticks") or []
            ]
            minimum_reduction = int(
                contract_config.get("minimum_reduction_each_phase") or 1
            )
            for tick in phase_ticks:
                actual_record = actual_records.get(tick)
                counterfactual_record = counterfactual_records.get(tick)
                if actual_record is None or counterfactual_record is None:
                    phases_completed = False
                    continue
                reduction = int(
                    counterfactual_record.get("n_voltage_violations") or 0
                ) - int(actual_record.get("n_voltage_violations") or 0)
                phase_reductions[str(tick)] = reduction
                phases_completed = phases_completed and reduction >= minimum_reduction

            reversal = dict(contract_config.get("reversal") or {})
            reversal_tool = str(reversal.get("tool") or "set_battery_dispatch")
            reversal_argument = str(reversal.get("argument") or "p_mw")
            later_not_before = int(reversal.get("later_not_before_tick") or 0)
            positive_ticks: list[int] = []
            negative_ticks: list[int] = []
            for record in ground_truth.get("_task_control_records") or []:
                if not isinstance(record, dict):
                    continue
                tick = int(record.get("tick") or 0)
                tool_name = str(record.get("tool_name") or "")
                if tool_name not in {"wait", "noop"}:
                    distinct_control_ticks.append(tick)
                if tool_name != reversal_tool:
                    continue
                args = record.get("args") or {}
                value = float(args.get(reversal_argument) or 0.0)
                if value > 0:
                    positive_ticks.append(tick)
                elif value < 0:
                    negative_ticks.append(tick)
            plan_reversal_observed = any(
                positive_tick < negative_tick and negative_tick >= later_not_before
                for positive_tick in positive_ticks
                for negative_tick in negative_ticks
            )
            minimum_control_ticks = int(
                contract_config.get("minimum_distinct_control_ticks") or 3
            )
            distinct_control_ticks = sorted(set(distinct_control_ticks))
            control_milestones_met = (
                len(distinct_control_ticks) >= minimum_control_ticks
            )
            completed = material_mitigation and phases_completed
        else:
            control_milestones_met = True
            completed = material_mitigation

        if (
            completed
            and contract_config.get("contract")
            == "microgrid.lv_voltage.staged_recovery.v2"
        ):
            reason_code = "staged_voltage_recovery_completed"
        elif completed and cross_tick_contract:
            reason_code = "cross_tick_recovery_completed"
        elif completed:
            reason_code = "material_voltage_violation_mitigation"
        elif not applicable:
            reason_code = "counterfactual_unavailable"
        elif catastrophic:
            reason_code = "catastrophic_outcome"
        elif prevented_loss <= materiality_threshold:
            reason_code = "no_material_improvement_vs_no_action"
        elif cross_tick_contract and not phases_completed:
            reason_code = "phase_recovery_incomplete"
        else:
            reason_code = "insufficient_voltage_violation_mitigation"
        return {
            "schema_version": "1.0",
            "contract": task_completion_contract(domain, family),
            "applicable": applicable,
            "completed": completed,
            "reason_code": reason_code,
            "evidence": {
                "actual_cost": actual_cost,
                "counterfactual_cost": float(
                    counterfactual.get("counterfactual_cost") or 0.0
                ),
                "prevented_loss": prevented_loss,
                "materiality_threshold": materiality_threshold,
                "actual_voltage_violation_cost": actual_voltage_cost,
                "counterfactual_voltage_violation_cost": counterfactual_voltage_cost,
                "voltage_violation_cost_reduction": voltage_reduction,
                "voltage_reduction_threshold": voltage_threshold,
                "system_survival_score": max(survival_scores, default=None),
                "chose_fatal_option": fatal,
                **(
                    {
                        "phase_reductions": phase_reductions,
                        "phase_recovery_completed": phases_completed,
                        "distinct_control_ticks": distinct_control_ticks,
                        "cross_tick_control_milestones_met": control_milestones_met,
                        "plan_reversal_observed": plan_reversal_observed,
                    }
                    if cross_tick_contract
                    else {}
                ),
            },
        }

    if domain == "logistics" and family == "job_shop_dispatch":
        scheduled = int(ground_truth.get("operations_scheduled") or 0)
        completed_operations = int(
            ground_truth.get("operations_completed", scheduled) or 0
        )
        cancelled_operations = int(ground_truth.get("operations_cancelled") or 0)
        total = int(ground_truth.get("operations_total") or 0)
        required_operations = max(0, total - cancelled_operations)
        applicable = total > 0
        native_requirements_met = True
        native_requirement_evidence: dict[str, Any] = {}
        requirements = dict(
            (scenario.get("backend_config") or {}).get("task_requirements") or {}
        )
        if requirements:
            controls = dict(ground_truth.get("control_summary") or {})
            actual_control_ticks = sorted(
                {int(tick) for tick in controls.get("distinct_control_ticks") or []}
            )
            physical_tools = sorted(
                {str(tool) for tool in controls.get("distinct_physical_tools") or []}
            )
            tool_ticks = {
                str(tool): sorted({int(tick) for tick in ticks or []})
                for tool, ticks in (controls.get("tool_ticks") or {}).items()
            }
            selected_milestone_ticks: list[int] = []
            selected_milestone_tools: list[str] = []
            previous_tick = -1
            ordered_met = True
            for milestone in requirements.get("ordered_tool_milestones") or []:
                tool = str(milestone.get("tool") or "")
                allowed_tools = {
                    str(value) for value in milestone.get("tools") or [] if str(value)
                }
                if tool:
                    allowed_tools.add(tool)
                if milestone.get("any_state_changing") is True:
                    allowed_tools.update(tool_ticks)
                earliest = int(milestone.get("not_before_tick", 0) or 0)
                latest = int(milestone.get("not_after_tick", 10**9) or 10**9)
                candidates = sorted(
                    (tick, candidate_tool)
                    for candidate_tool in allowed_tools
                    for tick in tool_ticks.get(candidate_tool, [])
                    if tick > previous_tick and earliest <= tick <= latest
                )
                if not candidates:
                    ordered_met = False
                    break
                previous_tick, selected_tool = candidates[0]
                selected_milestone_ticks.append(previous_tick)
                selected_milestone_tools.append(selected_tool)
            native_requirement_evidence = {
                "requirements": requirements,
                "distinct_control_ticks": actual_control_ticks,
                "distinct_physical_tools": physical_tools,
                "tool_ticks": tool_ticks,
                "ordered_tool_milestones_met": ordered_met,
                "selected_milestone_ticks": selected_milestone_ticks,
                "selected_milestone_tools": selected_milestone_tools,
            }
            native_requirements_met = len(actual_control_ticks) >= int(
                requirements.get("min_distinct_control_ticks") or 0
            ) and len(physical_tools) >= int(
                requirements.get("min_distinct_physical_tools") or 0
            )
            native_requirement_evidence["native_control_requirements_met"] = (
                native_requirements_met and ordered_met
            )
        completed = bool(
            applicable
            and scheduled == required_operations
            and completed_operations == required_operations
            and not fatal
            and not survival_floor
        )
        return {
            "schema_version": "1.0",
            "contract": task_completion_contract(domain, family),
            "applicable": applicable,
            "completed": completed,
            "reason_code": (
                "completed"
                if completed
                else "unsafe_outcome"
                if fatal or survival_floor
                else "cancellation_credited_as_scheduled"
                if cancelled_operations > 0 and scheduled > required_operations
                else "incomplete_schedule"
                if scheduled != required_operations
                or completed_operations != required_operations
                else "incomplete_schedule"
            ),
            "evidence": {
                "operations_scheduled": scheduled,
                "operations_completed": completed_operations,
                "operations_cancelled": cancelled_operations,
                "operations_total": total,
                "operations_required": required_operations,
                "cancellations_credited_as_completed": False,
                "survival_floor_violation": survival_floor,
                "chose_fatal_option": fatal,
                **native_requirement_evidence,
            },
        }

    logistics_task_loss_keys = {
        "cvrp_dispatch": ("unmet_demand_cost", "drop_order_penalty"),
        "vrptw_dispatch": (
            "unmet_demand_cost",
            "drop_order_penalty",
            "late_delivery_penalty",
        ),
        "inventory_replenishment": ("inventory_lost_sales_penalty",),
    }
    task_loss_keys: tuple[str, ...] | None = None
    contract_name = ""
    native_task_loss_from_records = False
    if domain == "power_grid":
        task_loss_keys = (
            "shed_penalty",
            "voltage_violation_cost",
            "overload_cost",
            "disconnection_cost",
            "reserve_shortfall_cost",
            "balance_error_cost",
            "safety_violation_cost",
        )
        contract_name = task_completion_contract(domain, family)
    elif domain == "traffic":
        task_loss_keys = (
            "travel_time_cost",
            "shed_delay_cost",
            "unserved_demand_cost",
            "collision_cost",
        )
        contract_name = task_completion_contract(domain, family)
    elif domain == "autonomous_driving":
        task_loss_keys = (
            "collision_cost",
            "road_departure_cost",
            "risk_exposure_cost",
            "shield_intervention_cost",
            "mrm_failure_cost",
            "route_delay_cost",
        )
        contract_name = task_completion_contract(domain, family)
    elif domain == "datacenter":
        task_loss_keys = (
            "queue_wait_cost",
            "sla_violation_cost",
            "unfinished_work_penalty",
            "preemption_waste_cost",
        )
        contract_name = task_completion_contract(domain, family)
    elif domain == "microgrid" and family == "microgrid_economic_dispatch_24h":
        native_task = dict(
            (scenario.get("backend_config") or {}).get("native_state_loss_task") or {}
        )
        declared_keys = native_task.get("required_task_loss_keys")
        if not isinstance(declared_keys, list | tuple):
            declared_keys = _MICROGRID_NATIVE_TASK_LOSS_KEYS
        task_loss_keys = (
            tuple(
                str(key)
                for key in declared_keys or _MICROGRID_NATIVE_TASK_LOSS_KEYS
                if str(key)
            )
            or _MICROGRID_NATIVE_TASK_LOSS_KEYS
        )
        contract_name = task_completion_contract(domain, family)
        native_task_loss_from_records = True
    elif domain == "logistics" and family in logistics_task_loss_keys:
        task_loss_keys = logistics_task_loss_keys[family]
        contract_name = task_completion_contract(domain, family)
    if task_loss_keys is not None:
        applicable = bool(counterfactual.get("applicable", False))
        actual_components = counterfactual.get("actual_components") or {}
        counterfactual_components = (
            counterfactual.get("counterfactual_components") or {}
        )
        keys = task_loss_keys
        if native_task_loss_from_records:
            actual_task_loss = _microgrid_native_task_loss(
                actual_components,
                ground_truth.get("_task_tick_records"),
                keys,
            )
            counterfactual_task_loss = _microgrid_native_task_loss(
                counterfactual_components,
                counterfactual.get("_counterfactual_task_tick_records"),
                keys,
            )
        else:
            actual_task_loss = sum(
                float(actual_components.get(key) or 0.0) for key in keys
            )
            counterfactual_task_loss = sum(
                float(counterfactual_components.get(key) or 0.0) for key in keys
            )
        task_loss_reduction = counterfactual_task_loss - actual_task_loss
        task_loss_threshold = max(1.0, counterfactual_task_loss * 0.001)
        prevented_loss = float(counterfactual.get("prevented_loss") or 0.0)
        actual_cost = float(counterfactual.get("actual_cost") or 0.0)
        materiality_threshold = max(1.0, abs(actual_cost) * 0.001)
        completion_prevented_loss = prevented_loss
        if native_task_loss_from_records:
            # Native task contracts may deliberately trade aggregate economic
            # cost for a materially safer/feasible operating state.  Keep the
            # cost delta in evidence, but gate completion on the declared
            # native loss units rather than mixing the two objectives.
            materiality_threshold = max(1.0, abs(counterfactual_task_loss) * 0.001)
            completion_prevented_loss = max(prevented_loss, task_loss_reduction)
        survival_scores = [
            float(dim.get("raw_score") or 0.0)
            for dim in score.get("dimensions", [])
            if isinstance(dim, dict)
            and dim.get("name") == "system_survival"
            and dim.get("applicable")
        ]
        catastrophic = (
            fatal
            or survival_floor
            or (
                domain == "power_grid"
                and bool(survival_scores)
                and max(survival_scores) <= 0.0
            )
        )
        native_requirements_met = True
        native_requirement_evidence: dict[str, Any] = {}
        if domain == "autonomous_driving":
            requirements = dict(
                (scenario.get("backend_config") or {}).get("task_requirements") or {}
            )
            assurance = dict(ground_truth.get("runtime_assurance") or {})
            mrm_ticks = [
                int(value)
                for value in assurance.get("mrm_ticks") or []
                if isinstance(value, int | float) and not isinstance(value, bool)
            ]
            mode_trace = [
                dict(value)
                for value in assurance.get("mode_trace") or []
                if isinstance(value, dict)
            ]
            required_dwell = max(
                0, int(requirements.get("required_stable_dwell_ticks") or 0)
            )
            tail_nominal = 0
            for row in reversed(mode_trace):
                if str(row.get("mode") or "") != "nominal":
                    break
                tail_nominal += 1
            recovery_required = bool(
                mrm_ticks
                and requirements.get("guarded_recovery_required_if_mrm") is True
            )
            recovery_completed = not recovery_required or bool(
                assurance.get("recovery_completed") is True
                and tail_nominal >= required_dwell
            )
            required_sequence = [
                str(value)
                for value in requirements.get("recovery_sequence") or []
                if str(value)
            ]
            observed_sequence = [
                str(value)
                for value in assurance.get("recovery_action_trace") or []
                if str(value)
            ]
            investigation_trace = [
                dict(value)
                for value in ground_truth.get("investigation_trace") or []
                if isinstance(value, dict)
            ]
            tactical_action_trace = [
                dict(value)
                for value in ground_truth.get("tactical_action_trace") or []
                if isinstance(value, dict)
            ]
            paid_inspection_required = (
                requirements.get("requires_paid_safety_inspection") is True
            )
            paid_inspection_deadline = int(
                requirements.get("paid_safety_inspection_deadline_tick") or 0
            )
            paid_inspection_ticks = [
                int(value["tick"])
                for value in investigation_trace
                if str(value.get("tool_name") or "")
                in {"inspect_local_scene", "inspect_safety_state"}
                and isinstance(value.get("tick"), int | float)
                and not isinstance(value.get("tick"), bool)
            ]
            paid_inspection_met = not paid_inspection_required or any(
                tick <= paid_inspection_deadline for tick in paid_inspection_ticks
            )
            observed_observation_tools = {
                str(value.get("tool_name") or "")
                for value in investigation_trace
                if str(value.get("tool_name") or "")
            }
            required_observation_tools = {
                str(value)
                for value in requirements.get("required_observation_tools") or []
                if str(value)
            }
            observation_tools_met = required_observation_tools.issubset(
                observed_observation_tools
            )
            preventive_action_required = (
                requirements.get("requires_preventive_action") is True
            )
            preventive_action_deadline = int(
                requirements.get("latest_preventive_command_tick") or 0
            )
            preventive_action_tools = {
                str(value)
                for value in requirements.get("preventive_action_tools") or []
                if str(value)
            }
            preventive_action_ticks = [
                int(value["tick"])
                for value in tactical_action_trace
                if str(value.get("tool_name") or "") in preventive_action_tools
                and isinstance(value.get("tick"), int | float)
                and not isinstance(value.get("tick"), bool)
                and int(value["tick"]) <= preventive_action_deadline
            ]
            preventive_action_met = not preventive_action_required or bool(
                preventive_action_ticks
            )
            decision_ticks = sorted(
                {
                    int(value["tick"])
                    for value in [*investigation_trace, *tactical_action_trace]
                    if isinstance(value.get("tick"), int | float)
                    and not isinstance(value.get("tick"), bool)
                }
            )
            minimum_decision_epochs = max(
                0, int(requirements.get("minimum_decision_epochs") or 0)
            )
            decision_epoch_floor_met = len(decision_ticks) >= minimum_decision_epochs

            def _is_subsequence(expected: list[str], observed: list[str]) -> bool:
                position = 0
                for value in observed:
                    if position < len(expected) and value == expected[position]:
                        position += 1
                return position == len(expected)

            sequence_met = (
                not recovery_required
                or not required_sequence
                or _is_subsequence(required_sequence, observed_sequence)
                or (
                    len(required_sequence) > 1
                    and _is_subsequence(required_sequence[1:], observed_sequence)
                )
            )
            native_requirements_met = (
                recovery_completed
                and sequence_met
                and paid_inspection_met
                and preventive_action_met
                and decision_epoch_floor_met
                and observation_tools_met
            )
            native_requirement_evidence = {
                "requirements": requirements,
                "mrm_ticks": mrm_ticks,
                "mode_trace": mode_trace,
                "required_stable_dwell_ticks": required_dwell,
                "observed_terminal_nominal_dwell_ticks": tail_nominal,
                "guarded_recovery_required": recovery_required,
                "recovery_completed": recovery_completed,
                "required_recovery_sequence": required_sequence,
                "observed_recovery_sequence": observed_sequence,
                "recovery_sequence_met": sequence_met,
                "paid_inspection_required": paid_inspection_required,
                "paid_inspection_deadline_tick": paid_inspection_deadline,
                "paid_inspection_ticks": paid_inspection_ticks,
                "paid_inspection_met": paid_inspection_met,
                "tactical_action_trace": tactical_action_trace,
                "requires_preventive_action": preventive_action_required,
                "preventive_action_deadline_tick": preventive_action_deadline,
                "preventive_action_tools": sorted(preventive_action_tools),
                "preventive_action_ticks": preventive_action_ticks,
                "preventive_action_met": preventive_action_met,
                "decision_ticks": decision_ticks,
                "minimum_decision_epochs": minimum_decision_epochs,
                "decision_epoch_floor_met": decision_epoch_floor_met,
                "required_observation_tools": sorted(required_observation_tools),
                "observed_observation_tools": sorted(observed_observation_tools),
                "observation_tools_met": observation_tools_met,
            }
        if domain in {"power_grid", "traffic", "logistics", "microgrid"}:
            requirements = dict(
                (scenario.get("backend_config") or {}).get("task_requirements") or {}
            )
            controls = dict(ground_truth.get("control_summary") or {})
            declared_control_axis_tool = ""
            declared_control_axis_observed = True
            if (
                domain == "power_grid"
                and family == "opendss_ieee13_volt_var"
                and str(
                    (scenario.get("backend_config") or {}).get("decision_axis") or ""
                )
            ):
                declared_control_axis_tool = str(
                    (
                        (scenario.get("backend_config") or {}).get(
                            "control_action_probe"
                        )
                        or {}
                    ).get("tool")
                    or ""
                )
                declared_control_axis_observed = bool(
                    declared_control_axis_tool
                    and declared_control_axis_tool
                    in {
                        str(tool)
                        for tool in controls.get("distinct_physical_tools") or []
                    }
                )
                native_requirements_met = declared_control_axis_observed
                native_requirement_evidence = {
                    "declared_control_axis_tool": declared_control_axis_tool,
                    "declared_control_axis_observed": (declared_control_axis_observed),
                }
            if requirements:
                actual_control_ticks = sorted(
                    {int(tick) for tick in controls.get("distinct_control_ticks") or []}
                )
                physical_tools = sorted(
                    {
                        str(tool)
                        for tool in controls.get("distinct_physical_tools") or []
                    }
                )
                physical_controls, identity_mode, endpoint_minimum = (
                    _physical_control_identities(scenario, controls)
                )
                request_tool_ticks = {
                    str(tool): sorted({int(tick) for tick in ticks or []})
                    for tool, ticks in (controls.get("tool_ticks") or {}).items()
                }
                raw_effect_tool_ticks = (
                    controls.get("effect_tool_ticks")
                    if "effect_tool_ticks" in controls
                    else controls.get("tool_ticks") or {}
                )
                effect_tool_ticks = {
                    str(tool): sorted({int(tick) for tick in ticks or []})
                    for tool, ticks in (raw_effect_tool_ticks or {}).items()
                }
                lifecycle_records = [
                    row
                    for row in controls.get("control_lifecycle_records") or []
                    if isinstance(row, dict)
                    and bool(row.get("call_id"))
                    and bool(row.get("effect_event_id"))
                    and isinstance(row.get("request_tick"), int)
                    and not isinstance(row.get("request_tick"), bool)
                    and isinstance(row.get("effect_tick"), int)
                    and not isinstance(row.get("effect_tick"), bool)
                    and int(row["effect_tick"]) >= int(row["request_tick"])
                ]
                selected_milestone_ticks: list[int] = []
                selected_milestone_tools: list[str] = []
                previous_tick = -1
                ordered_met = True
                for milestone in requirements.get("ordered_tool_milestones") or []:
                    tool = str(milestone.get("tool") or "")
                    allowed_tools = {
                        str(value)
                        for value in milestone.get("tools") or []
                        if str(value)
                    }
                    if tool:
                        allowed_tools.add(tool)
                    if milestone.get("any_state_changing") is True:
                        allowed_tools.update(effect_tool_ticks)
                    earliest = int(milestone.get("not_before_tick", 0) or 0)
                    latest = int(milestone.get("not_after_tick", 10**9) or 10**9)
                    requires_call_binding = bool(
                        milestone.get("args") is not None
                        or milestone.get("action_predicate") is not None
                    )
                    if lifecycle_records:
                        candidates = sorted(
                            (int(row["effect_tick"]), str(row.get("tool_name") or ""))
                            for row in lifecycle_records
                            if str(row.get("tool_name") or "") in allowed_tools
                            and int(row["effect_tick"]) > previous_tick
                            and earliest <= int(row["effect_tick"]) <= latest
                            and _milestone_action_predicate_passes(
                                milestone,
                                record=row,
                            )
                        )
                    elif requires_call_binding:
                        candidates = []
                    else:
                        candidates = sorted(
                            (tick, candidate_tool)
                            for candidate_tool in allowed_tools
                            for tick in effect_tool_ticks.get(candidate_tool, [])
                            if tick > previous_tick and earliest <= tick <= latest
                        )
                    if not candidates:
                        ordered_met = False
                        break
                    previous_tick, selected_tool = candidates[0]
                    selected_milestone_ticks.append(previous_tick)
                    selected_milestone_tools.append(selected_tool)
                native_requirement_evidence = {
                    "requirements": requirements,
                    "distinct_control_ticks": actual_control_ticks,
                    "distinct_physical_tools": physical_tools,
                    "distinct_physical_control_identities": physical_controls,
                    "physical_control_identity_mode": identity_mode,
                    "tool_ticks": request_tool_ticks,
                    "effect_tool_ticks": effect_tool_ticks,
                    "control_lifecycle_records": lifecycle_records,
                    "ordered_tool_milestones_met": ordered_met,
                    "selected_milestone_ticks": selected_milestone_ticks,
                    "selected_milestone_tools": selected_milestone_tools,
                    "declared_control_axis_tool": declared_control_axis_tool,
                    "declared_control_axis_observed": (declared_control_axis_observed),
                }
                native_requirements_met = (
                    len(actual_control_ticks)
                    >= int(requirements.get("min_distinct_control_ticks") or 0)
                    and len(physical_controls)
                    >= max(
                        int(requirements.get("min_distinct_physical_tools") or 0),
                        endpoint_minimum,
                    )
                    and declared_control_axis_observed
                )
        elif domain == "datacenter":
            requirements = dict(
                (scenario.get("backend_config") or {}).get("task_requirements") or {}
            )
            controls = dict(ground_truth.get("control_summary") or {})
            actual_control_ticks = sorted(
                {int(tick) for tick in controls.get("distinct_control_ticks") or []}
            )
            tool_ticks = {
                str(tool): sorted({int(tick) for tick in ticks or []})
                for tool, ticks in (controls.get("tool_ticks") or {}).items()
            }
            selected_milestone_ticks: list[int] = []
            selected_milestone_tools: list[str] = []
            previous_tick = -1
            ordered_met = True
            for milestone in requirements.get("ordered_tool_milestones") or []:
                tool = str(milestone.get("tool") or "")
                allowed_tools = {
                    str(value) for value in milestone.get("tools") or [] if str(value)
                }
                if tool:
                    allowed_tools.add(tool)
                if milestone.get("any_state_changing") is True:
                    allowed_tools.update(tool_ticks)
                earliest = int(milestone.get("not_before_tick", 0) or 0)
                latest = int(milestone.get("not_after_tick", 10**9) or 10**9)
                candidates = sorted(
                    (tick, candidate_tool)
                    for candidate_tool in allowed_tools
                    for tick in tool_ticks.get(candidate_tool, [])
                    if tick > previous_tick and earliest <= tick <= latest
                )
                if not candidates:
                    ordered_met = False
                    break
                previous_tick, selected_tool = candidates[0]
                selected_milestone_ticks.append(previous_tick)
                selected_milestone_tools.append(selected_tool)
            physical_tools = sorted(
                {str(tool) for tool in controls.get("distinct_physical_tools") or []}
            )
            physical_controls, identity_mode, endpoint_minimum = (
                _physical_control_identities(scenario, controls)
            )
            minimum_physical_tools = max(
                int(requirements.get("min_native_physical_tools") or 0),
                int(requirements.get("min_distinct_physical_tools") or 0),
                endpoint_minimum,
            )
            observed_physical_count = len(
                physical_controls if identity_mode != "tool_name" else physical_tools
            )
            response_windows = [
                window
                for window in controls.get("response_windows") or []
                if isinstance(window, dict)
            ]
            required_response_windows = int(
                requirements.get("response_windows_required") or 0
            )
            responded_windows = sum(
                bool(window.get("control_ticks")) for window in response_windows
            )
            response_windows_met = (
                required_response_windows <= 0
                or responded_windows >= required_response_windows
            )
            strategy_reversal_count = int(controls.get("strategy_reversal_count") or 0)
            required_strategy_reversals = int(
                requirements.get("min_strategy_reversals") or 0
            )
            strategy_reversals_met = (
                strategy_reversal_count >= required_strategy_reversals
            )
            required_effective_control_ticks = int(
                requirements.get("min_effective_control_ticks") or 0
            )
            effective_control_tick_floor_met = (
                len(actual_control_ticks) >= required_effective_control_ticks
            )
            native_requirement_evidence = {
                "requirements": requirements,
                "queue_policy_changes": int(controls.get("queue_policy_changes") or 0),
                "reservation_arrivals": int(controls.get("reservation_arrivals") or 0),
                "distinct_control_ticks": actual_control_ticks,
                "distinct_physical_tools": physical_tools,
                "distinct_physical_control_identities": physical_controls,
                "physical_control_identity_mode": identity_mode,
                "tool_ticks": tool_ticks,
                "ordered_tool_milestones_met": ordered_met,
                "selected_milestone_ticks": selected_milestone_ticks,
                "selected_milestone_tools": selected_milestone_tools,
                "strategy_reversal_count": strategy_reversal_count,
                "strategy_reversals_met": strategy_reversals_met,
                "effective_control_tick_floor_met": (effective_control_tick_floor_met),
                "response_windows": response_windows,
                "responded_response_windows": responded_windows,
                "response_windows_met": response_windows_met,
            }
            native_requirements_met = (
                native_requirement_evidence["queue_policy_changes"]
                >= int(requirements.get("min_queue_policy_changes") or 0)
                and native_requirement_evidence["reservation_arrivals"]
                >= int(requirements.get("min_reservation_arrivals") or 0)
                and len(actual_control_ticks)
                >= int(requirements.get("min_distinct_control_ticks") or 0)
                and effective_control_tick_floor_met
                and observed_physical_count >= minimum_physical_tools
                and strategy_reversals_met
                and response_windows_met
            )
        completed = (
            applicable
            and not catastrophic
            and (native_requirements_met if domain == "autonomous_driving" else True)
            and completion_prevented_loss > materiality_threshold
            and task_loss_reduction > task_loss_threshold
        )
        if completed:
            reason_code = "material_task_loss_mitigation"
        elif not applicable:
            reason_code = "counterfactual_unavailable"
        elif catastrophic:
            reason_code = "unsafe_outcome"
        elif completion_prevented_loss <= materiality_threshold:
            reason_code = "no_material_improvement_vs_no_action"
        else:
            reason_code = "insufficient_task_loss_mitigation"
        return {
            "schema_version": "1.0",
            "contract": contract_name,
            "applicable": applicable,
            "completed": completed,
            "reason_code": reason_code,
            "evidence": {
                "actual_cost": actual_cost,
                "counterfactual_cost": float(
                    counterfactual.get("counterfactual_cost") or 0.0
                ),
                "prevented_loss": prevented_loss,
                "completion_prevented_loss": completion_prevented_loss,
                "materiality_threshold": materiality_threshold,
                "task_loss_component_keys": list(keys),
                "actual_task_loss": actual_task_loss,
                "counterfactual_task_loss": counterfactual_task_loss,
                "task_loss_reduction": task_loss_reduction,
                "task_loss_reduction_threshold": task_loss_threshold,
                "system_survival_score": max(survival_scores, default=None),
                "chose_fatal_option": fatal,
                "native_control_requirements_met": (native_requirements_met),
                **native_requirement_evidence,
            },
        }

    actual_cost = float(counterfactual.get("actual_cost") or 0.0)
    prevented_loss = float(counterfactual.get("prevented_loss") or 0.0)
    materiality_threshold = max(1.0, abs(actual_cost) * 0.001)
    return {
        "schema_version": "1.0",
        "contract": task_completion_contract(domain, family),
        "applicable": False,
        "completed": False,
        "reason_code": "unsupported_task_contract",
        "evidence": {
            "actual_cost": actual_cost,
            "counterfactual_cost": float(
                counterfactual.get("counterfactual_cost") or 0.0
            ),
            "prevented_loss": prevented_loss,
            "materiality_threshold": materiality_threshold,
            "survival_floor_violation": survival_floor,
            "chose_fatal_option": fatal,
        },
    }
