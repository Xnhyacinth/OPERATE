"""Offline fulfillment of source obligations over a complete native episode.

This module measures delivered work, demand, or compliant operating exposure.
It never counts model actions as completed work and never derives a denominator
from the work an agent chose to accept. Economic-only CityLearn tasks have no
separately modeled service shortfall and are explicitly inapplicable here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from evaluation.mission_contracts import _digest, _read_locked
from evaluation.operational_service import SERVICE_BACKENDS


REVISION = "source_obligation_fulfillment.v2"
_VOLTAGE_BACKENDS = frozenset(
    {"pandapower_lv", "cigre_distribution", "opendss_ieee13", "opendss_fresh_feeders"}
)

# These are measurement types, not interchangeable percentages of finished
# missions. The source contract fixes their population before any model runs.
_COMPLETION_TYPES = {
    "pyvrp_cvrp": (
        "source_delivery_demand",
        "demand_weighted_customer_delivery",
        "all_source_customers_and_seeded_urgent_orders",
        "validly_served_original_obligations",
    ),
    "pyvrp_vrptw": (
        "source_delivery_demand",
        "demand_weighted_customer_delivery",
        "all_source_customers_and_seeded_urgent_orders",
        "validly_served_original_obligations",
    ),
    "dynasched_flexible_job_shop": (
        "complete_source_operations",
        "operation_progress",
        "all_source_operations_including_future_arrivals",
        "completed_operations_not_scheduled_or_cancelled",
    ),
    "alibaba_trace_sim": (
        "source_job_service",
        "job_completion",
        "all_jobs_in_locked_source_window",
        "jobs_with_native_done_status",
    ),
    "orgym_invmgmt": (
        "source_customer_demand",
        "demand_fulfillment",
        "full_locked_demand_stream",
        "fulfilled_demand_units",
    ),
    "sumo_ego": (
        "safe_mobility_and_guarded_recovery",
        "conditional_guarded_recovery_attainment",
        "source_required_stable_nominal_ticks_if_mrm_occurs",
        "full_credit_without_mrm_else_terminal_nominal_ticks_capped_at_required_dwell",
    ),
    "pymgrid_economic_dispatch": (
        "critical_supply_and_balance",
        "energy_demand_fulfillment",
        "source_load_plus_seeded_spikes",
        "demand_energy_less_metered_shed",
    ),
    "pglib_uc_synthetic": (
        "load_and_reserve_service",
        "energy_demand_fulfillment",
        "locked_source_load_plus_seeded_surges",
        "demand_energy_less_shed_or_shortfall_once",
    ),
    "pandapower_lv": (
        "voltage_reliability",
        "service_compliance_bottleneck",
        "source_load_exposure_and_fixed_bus_time",
        "minimum_of_load_fulfillment_and_voltage_compliant_bus_time",
    ),
    "cigre_distribution": (
        "voltage_reliability",
        "service_compliance_bottleneck",
        "source_load_exposure_and_fixed_bus_time",
        "minimum_of_load_fulfillment_and_voltage_compliant_bus_time",
    ),
    "opendss_ieee13": (
        "voltage_reliability",
        "node_time_compliance",
        "fixed_feeder_nodes_over_episode_ticks",
        "voltage_compliant_node_time",
    ),
    "opendss_fresh_feeders": (
        "voltage_reliability",
        "node_time_compliance",
        "fixed_feeder_nodes_over_episode_ticks",
        "voltage_compliant_node_time",
    ),
    "citylearn": (
        "automatic_exogenous_load_service",
        "economic_only_no_completion",
        "not_applicable",
        "no_independent_controllable_unserved_load_meter",
    ),
}


def compile_completion_contract(scenario: dict, source_contract: dict) -> dict:
    """Declare exactly what the fixed-source completion fraction measures."""
    backend = scenario["backend_kind"]
    expected_service, metric, population, numerator = _COMPLETION_TYPES[backend]
    if (source_contract.get("service") or {}).get("kind") != expected_service:
        raise ValueError("completion_source_service_kind_mismatch")
    if source_contract["scenario_signature"] != scenario["scenario_signature"] or (
        source_contract["seed"] != scenario["seed"]
    ):
        raise ValueError("completion_source_identity_mismatch")
    stable_dwell = None
    if backend == "sumo_ego":
        requirements = (scenario.get("backend_config") or {}).get(
            "task_requirements"
        ) or {}
        stable_dwell = requirements.get("required_stable_dwell_ticks")
        if (
            type(stable_dwell) is not int
            or stable_dwell <= 0
            or requirements.get("guarded_recovery_required_if_mrm") is not True
            or (scenario.get("task_contract") or {}).get("contract")
            != "autonomous_driving.risk_progress_mitigation.v1"
        ):
            raise ValueError("completion_source_stable_recovery_requirement_missing")
    work_progress = backend in {"dynasched_flexible_job_shop", "alibaba_trace_sim"}
    late_arrival_rule = "not_a_source_work_progress_metric"
    if work_progress:
        late_arrival_rule = (
            "uncompleted_late_arrivals_remain_in_source_work_denominator;"
            "not_all_unfinished_work_is_attributable_to_agent"
        )
    elif backend in {"pyvrp_cvrp", "pyvrp_vrptw"}:
        late_arrival_rule = (
            "seeded_urgent_orders_remain_in_denominator;"
            "not_all_undelivered_late_orders_are_attributable_to_agent"
        )
    contract = {
        "schema_version": "source_completion_contract.v1",
        "scenario_signature": source_contract["scenario_signature"],
        "seed": source_contract["seed"],
        "scenario_sha256": source_contract["scenario_sha256"],
        "backend_kind": backend,
        "completion_kind": metric,
        "denominator_scope": population,
        "numerator_definition": numerator,
        "required_stable_dwell_ticks": stable_dwell,
        "window_rule": (
            "conditional_terminal_nominal_dwell_after_native_mrm"
            if backend == "sumo_ego"
            else "source_work_progress_at_episode_end_includes_future_arrivals"
            if work_progress
            else "source_obligations_or_exposure_within_horizon"
            if backend != "citylearn"
            else "not_applicable"
        ),
        "late_arrival_rule": late_arrival_rule,
        "timeliness_in_completion": False,
        "completion_is_binary_mission_success": False,
        "joint_qualified_supply_claim": False,
        "structural_not_applicable_reason": (
            "automatic_service_without_controllable_unserved_load_meter"
            if backend == "citylearn"
            else None
        ),
        "evidence_requirements": (
            ["hash_bound_native_assurance_tick_records", "native_terminal_mode"]
            if backend == "sumo_ego"
            else ["authenticated_025_service_result", "source_denominator_recheck"]
            if backend in SERVICE_BACKENDS
            else []
            if backend == "citylearn"
            else ["hash_bound_scoring_snapshot", "hash_bound_native_trajectory"]
        ),
    }
    contract["contract_sha256"] = _digest(contract)
    return contract


def _number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _result(
    kind: str, *, reason: str, score: float | None = None, **details: object
) -> dict:
    return {
        "protocol_revision": REVISION,
        "kind": kind,
        "applicable": kind != "economic_optimization_only",
        "score": score,
        "reason": reason,
        **details,
    }


def _score(numerator: float, denominator: float) -> float | None:
    if not (_number(numerator) and _number(denominator) and denominator > 0):
        return None
    if numerator > denominator + max(1e-6, denominator * 1e-6):
        return None
    return min(100.0, 100.0 * min(numerator, denominator) / denominator)


def _fjsp_job_diagnostic(trace: list[dict], jobs_total: object) -> dict:
    """Separate finished-job delivery from the operation-progress main meter."""
    obs = (trace[-1].get("observation") or {}) if trace else {}
    arrived, completed, cancelled = (
        obs.get("jobs_arrived"),
        obs.get("jobs_completed"),
        obs.get("jobs_cancelled"),
    )
    if not (
        type(jobs_total) is int
        and jobs_total > 0
        and obs.get("jobs_total") == jobs_total
        and all(type(v) is int and v >= 0 for v in (arrived, completed, cancelled))
        and completed + cancelled <= arrived <= jobs_total
    ):
        return {"score": None, "reason": "native_job_completion_evidence_missing"}
    return {
        "score": 100.0 * completed / jobs_total,
        "numerator": completed,
        "denominator": jobs_total,
        "jobs_arrived": arrived,
        "jobs_cancelled": cancelled,
        "reason": "native_completed_jobs_over_all_source_jobs",
    }


def score_joint_outcome(
    completion: dict,
    native_quality: object,
    native_interval: object,
    *,
    hard_failure: bool,
) -> dict:
    """Require both source service and native quality without double penalties.

    The lower of two bounded result measures is the noncompensatory bottleneck.
    CityLearn has no independent completion meter, so its native quality alone
    applies. A verified hard failure is zero regardless of partial delivery.
    """
    if not (
        _number(native_quality)
        and native_quality <= 100
        and isinstance(native_interval, list)
        and len(native_interval) == 2
        and all(_number(v) and v <= 100 for v in native_interval)
        and native_interval[0] <= native_quality <= native_interval[1]
    ):
        return {
            "score": None,
            "score_interval": None,
            "reason": "native_quality_evidence_missing",
        }
    if hard_failure:
        return {
            "score": 0.0,
            "score_interval": [0.0, 0.0],
            "reason": "native_hard_failure",
        }
    if completion["applicable"] is False:
        return {
            "score": native_quality,
            "score_interval": native_interval,
            "reason": "native_economic_quality_only",
        }
    fraction = completion.get("score")
    if not _number(fraction) or fraction > 100:
        return {
            "score": None,
            "score_interval": None,
            "reason": completion.get("reason", "source_completion_missing"),
        }
    return {
        "score": min(fraction, native_quality),
        "score_interval": [
            min(fraction, native_interval[0]),
            min(fraction, native_interval[1]),
        ],
        "reason": "completion_capped_native_quality",
    }


def _cumulative_shed(
    trace: list[dict], expected: dict[str, float]
) -> list[float] | None:
    """Recover per-tick physical shed from monotone load meters."""
    if (
        not isinstance(expected, dict)
        or not expected
        or any(not isinstance(k, str) or not _number(v) for k, v in expected.items())
    ):
        return None
    previous = {key: 0.0 for key in expected}
    increments = []
    for row in trace:
        entities = (row.get("observation") or {}).get("entities") or {}
        current = {}
        for key in expected:
            entity = entities.get(key)
            value = (
                entity.get("cumulative_shed_mwh") if isinstance(entity, dict) else None
            )
            if not _number(value) or value + 1e-6 < previous[key]:
                return None
            current[key] = float(value)
        increments.append(sum(max(0.0, current[k] - previous[k]) for k in expected))
        previous = current
    if any(
        not math.isclose(previous[k], expected[k], rel_tol=1e-5, abs_tol=0.02)
        for k in expected
    ):
        return None
    return increments


def _source_pymgrid_demand(scenario: dict) -> float | None:
    config = scenario.get("backend_config") or {}
    loads = (config.get("profiles") or {}).get("load_mw") or []
    horizon = scenario["horizon_ticks"]
    hours = float(scenario.get("tick_minutes", 60)) / 60.0
    if len(loads) < horizon or not _number(hours) or hours <= 0:
        return None
    demand = 0.0
    for tick in range(horizon):
        factor = 0.0
        for event in scenario.get("perturbations") or []:
            if (
                event.get("kind") == "load_spike"
                and event["trigger_tick"]
                <= tick
                < event["trigger_tick"] + event["duration_ticks"]
            ):
                factor = max(0.0, float(event["intensity"]))
        if not _number(loads[tick]):
            return None
        demand += float(loads[tick]) * (1.0 + factor) * hours
    return demand


def _source_pglib_demand(scenario: dict, root: Path) -> list[float] | None:
    config = scenario.get("backend_config") or {}
    path = config.get("case_file")
    hashes = (scenario.get("source_contract") or {}).get("file_sha256s") or {}
    if not isinstance(path, str) or path not in hashes:
        return None
    case = json.loads(
        _read_locked(root, {"path": path, "sha256": hashes[path]}, label="source_asset")
    )
    first = int(config.get("first_period", 1)) - 1
    demand = case["demand"][first : first + scenario["horizon_ticks"]]
    if len(demand) != scenario["horizon_ticks"]:
        return None
    result = []
    for tick, value in enumerate(demand):
        if not _number(value):
            return None
        factor = 0.0
        for event in scenario.get("perturbations") or []:
            if (
                event.get("kind") == "load_surge"
                and event["trigger_tick"]
                <= tick
                < event["trigger_tick"] + event["duration_ticks"]
            ):
                factor = float(event["intensity"])
        result.append(float(value) * (1.0 + factor))
    return result


def _voltage_measurement(backend: str, records: list[dict], trace: list[dict]) -> dict:
    first = (trace[0].get("observation") or {}) if trace else {}
    if backend.startswith("opendss_"):
        population = first.get("n_nodes")
        same_population = all(
            (row.get("observation") or {}).get("n_nodes") == population for row in trace
        )
        population_basis = "fixed_source_feeder_nodes"
    else:
        bus_ids = {
            key
            for key in (first.get("entities") or {})
            if isinstance(key, str) and key.startswith("bus_")
        }
        population = len(bus_ids)
        same_population = all(
            {
                key
                for key in ((row.get("observation") or {}).get("entities") or {})
                if isinstance(key, str) and key.startswith("bus_")
            }
            == bus_ids
            for row in trace
        )
        population_basis = "fixed_source_network_buses"
    if type(population) is not int or population <= 0 or not same_population:
        return _result("voltage_reliability", reason="voltage_population_unverified")
    compliant = 0.0
    for record, row in zip(records, trace, strict=True):
        violations = record.get("n_voltage_violations")
        converged = record.get("converged")
        if converged is None and backend == "pandapower_lv":
            # Older LV snapshots omitted the solver flag. A positive native
            # delivered demand and finite bus voltages certify a solved tick.
            entities = (row.get("observation") or {}).get("entities") or {}
            bus_voltages = [entities[key].get("vm_pu") for key in bus_ids]
            if (
                _number(record.get("aggregate_demand_mw"))
                and record["aggregate_demand_mw"] > 0
                and all(_number(v) for v in bus_voltages)
            ):
                converged = True
        if (
            type(violations) is not int
            or violations < 0
            or violations > population
            or type(converged) is not bool
        ):
            return _result("voltage_reliability", reason="voltage_tick_invalid")
        compliant += population - violations if converged else 0
    denominator = population * len(records)
    return _result(
        "voltage_reliability",
        reason="native_voltage_compliant_node_time",
        score=_score(compliant, denominator),
        numerator=compliant,
        denominator=denominator,
        population_basis=population_basis,
    )


def measure_completion(
    scenario: dict,
    *,
    source_contract: dict,
    snapshot_inputs: dict,
    trace: list[dict],
    existing_service: dict | None = None,
    root: Path = Path("."),
) -> dict:
    """Measure fulfilled external obligations using authenticated native inputs.

    The caller must validate the scenario, native artifacts or previously
    authenticated service result, and evidence IDs.
    A missing or inconsistent measurement returns N/A, never a success guess.
    """
    backend = scenario["backend_kind"]
    service_kind = (source_contract.get("service") or {}).get("kind")
    if backend == "citylearn" and service_kind == "automatic_exogenous_load_service":
        return _result(
            "economic_optimization_only",
            reason="no_agent_controllable_unserved_building_load_measure",
        )
    if (
        backend != "sumo_ego"
        and existing_service
        and existing_service.get("applicable") is True
    ):
        value = existing_service.get("score")
        if (
            not _number(value)
            or value > 100
            or not existing_service.get("evidence_ids")
        ):
            return _result(
                service_kind or backend, reason="source_service_evidence_missing"
            )
        details = {}
        if backend == "dynasched_flexible_job_shop":
            details["job_completion_diagnostic"] = _fjsp_job_diagnostic(
                trace, (source_contract.get("service") or {}).get("jobs_total")
            )
        return _result(
            service_kind or backend,
            reason="fixed_source_obligation_delivered",
            score=float(value),
            numerator=existing_service.get("numerator"),
            denominator=existing_service.get("denominator"),
            timeliness=existing_service.get("timeliness"),
            **details,
        )
    if backend in SERVICE_BACKENDS and backend != "sumo_ego":
        return _result(
            service_kind or backend, reason="source_service_evidence_missing"
        )
    if not trace or not isinstance(snapshot_inputs.get("backend_tick_records"), list):
        return _result(
            service_kind or backend, reason="native_completion_evidence_missing"
        )
    records = snapshot_inputs["backend_tick_records"]
    if len(records) != len(trace) or len(records) > scenario["horizon_ticks"]:
        return _result(
            service_kind or backend, reason="native_completion_window_invalid"
        )

    if backend == "sumo_ego":
        requirements = (scenario.get("backend_config") or {}).get(
            "task_requirements"
        ) or {}
        required = requirements.get("required_stable_dwell_ticks")
        if (
            type(required) is not int
            or required <= 0
            or requirements.get("guarded_recovery_required_if_mrm") is not True
            or scenario.get("task_contract", {}).get("contract")
            != "autonomous_driving.risk_progress_mitigation.v1"
        ):
            return _result(
                service_kind, reason="source_stable_recovery_requirement_missing"
            )
        allowed = {
            "nominal",
            "degraded",
            "mrm_active",
            "minimal_risk_condition",
            "recovery_pending",
        }
        mrm_modes = {"mrm_active", "minimal_risk_condition", "recovery_pending"}
        if any(
            r.get("assurance_mode") not in allowed
            or type(r.get("mrm_active")) is not bool
            or type(r.get("catastrophic_failure")) is not bool
            or r["mrm_active"] != (r["assurance_mode"] in mrm_modes)
            for r in records
        ):
            return _result(service_kind, reason="native_recovery_state_missing")
        progress = records[-1].get("route_progress")
        if (
            not _number(progress)
            or progress > 1
            or not existing_service
            or existing_service.get("applicable") is not True
            or not _number(existing_service.get("score"))
            or not math.isclose(
                existing_service["score"], 100 * progress, rel_tol=0, abs_tol=1e-4
            )
        ):
            return _result(
                service_kind, reason="native_route_progress_evidence_mismatch"
            )
        tail = 0
        for record in reversed(records):
            if record["assurance_mode"] != "nominal":
                break
            tail += 1
        mrm_observed = any(record["mrm_active"] for record in records)
        credited = min(tail, required) if mrm_observed else required
        return _result(
            service_kind,
            reason=(
                "source_required_terminal_stable_recovery_dwell"
                if mrm_observed
                else "no_mrm_guarded_recovery_not_required"
            ),
            score=100.0 * credited / required,
            numerator=credited,
            denominator=required,
            terminal_nominal_ticks=tail,
            mrm_observed=mrm_observed,
            recovery_required=mrm_observed,
            route_progress_diagnostic=100 * progress,
            interpretation="terminal_native_stability_not_full_route_completion",
        )

    if backend == "pymgrid_economic_dispatch":
        demand = _source_pymgrid_demand(scenario)
        shed_map = snapshot_inputs.get("per_load_shed_mwh")
        if (
            demand is None
            or not isinstance(shed_map, dict)
            or any(not _number(v) for v in shed_map.values())
        ):
            return _result(service_kind, reason="source_demand_or_shed_missing")
        shed = math.fsum(shed_map.values())
        score = (
            _score(max(0.0, demand - shed), demand) if shed <= demand + 1e-6 else None
        )
        return _result(
            service_kind,
            reason="source_demand_minus_native_unserved_energy"
            if score is not None
            else "shed_exceeds_source_demand",
            score=score,
            numerator=demand - shed,
            denominator=demand,
            denominator_basis="source_profile_plus_seeded_load_spikes",
        )

    if backend == "pglib_uc_synthetic":
        expected = _source_pglib_demand(scenario, root)
        shed = _cumulative_shed(trace, snapshot_inputs.get("per_load_shed_mwh"))
        hours = float(scenario.get("tick_minutes", 60)) / 60.0
        if expected is None or shed is None or not _number(hours) or hours <= 0:
            return _result(service_kind, reason="source_demand_or_shed_missing")
        if any(
            not _number(r.get("aggregate_demand_mw"))
            or not math.isclose(
                r["aggregate_demand_mw"], value, rel_tol=0, abs_tol=0.02
            )
            or type(r.get("balance_error_mw")) not in (int, float)
            or not math.isfinite(r["balance_error_mw"])
            for r, value in zip(records, expected, strict=True)
        ):
            return _result(service_kind, reason="source_demand_tick_mismatch")
        demand = math.fsum(value * hours for value in expected)
        unserved = math.fsum(
            min(
                value * hours,
                max(shed[tick], max(0.0, -record["balance_error_mw"]) * hours),
            )
            for tick, (record, value) in enumerate(zip(records, expected, strict=True))
        )
        score = _score(demand - unserved, demand)
        return _result(
            service_kind,
            reason="source_load_satisfied_after_shed_and_shortfall"
            if score is not None
            else "source_supply_invalid",
            score=score,
            numerator=demand - unserved,
            denominator=demand,
            denominator_basis="locked_case_demand_plus_seeded_surge",
        )

    if backend in _VOLTAGE_BACKENDS:
        voltage = _voltage_measurement(backend, records, trace)
        if voltage["score"] is None:
            return voltage
        if backend.startswith("opendss_"):
            return voltage
        hours = float(scenario.get("tick_minutes", 60)) / 60.0
        if not _number(hours) or hours <= 0:
            return _result(service_kind, reason="invalid_native_tick_duration")
        if backend == "pandapower_lv":
            shed = _cumulative_shed(trace, snapshot_inputs.get("per_load_shed_mwh"))
            if shed is None or any(
                not _number(r.get("aggregate_demand_mw")) for r in records
            ):
                return _result(service_kind, reason="native_load_service_missing")
            if any(
                r.get("unserved_energy_mwh") is not None
                and (
                    not _number(r["unserved_energy_mwh"])
                    or not math.isclose(
                        r["unserved_energy_mwh"],
                        shed[tick],
                        rel_tol=1e-5,
                        abs_tol=0.02,
                    )
                )
                for tick, r in enumerate(records)
            ):
                return _result(service_kind, reason="native_shed_meter_conflict")
            demand = math.fsum(
                r["aggregate_demand_mw"] * hours + shed[tick]
                for tick, r in enumerate(records)
            )
            unserved = math.fsum(shed)
        else:
            shed = _cumulative_shed(trace, snapshot_inputs.get("per_load_shed_mwh"))
            if shed is None or any(
                not _number(r.get("aggregate_demand_mw")) for r in records
            ):
                return _result(service_kind, reason="native_load_service_missing")
            demand = math.fsum(
                r["aggregate_demand_mw"] * hours + shed[tick]
                for tick, r in enumerate(records)
            )
            unserved = math.fsum(
                shed[tick]
                + (
                    r["aggregate_demand_mw"] * hours
                    if r.get("converged") is False
                    else 0.0
                )
                for tick, r in enumerate(records)
            )
        supplied = (
            _score(max(0.0, demand - unserved), demand)
            if unserved <= demand + 1e-6
            else None
        )
        if supplied is None:
            return _result(service_kind, reason="native_supply_fraction_invalid")
        # Both reliable voltage and delivered load are required; one cannot
        # compensate for the other. The bottleneck is reported with its parts.
        score = min(voltage["score"], supplied)
        return _result(
            service_kind,
            reason="bottleneck_voltage_and_load_service",
            score=score,
            numerator=None,
            denominator=None,
            voltage_compliance=voltage["score"],
            load_fulfillment=supplied,
            aggregation="minimum_noncompensatory_required_services",
        )

    return _result(service_kind or backend, reason="completion_backend_unsupported")
