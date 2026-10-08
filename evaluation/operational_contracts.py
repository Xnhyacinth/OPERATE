"""Source-only utility scales for the complete released Lite suite.

Scales are declared unit-of-loss normalizers, NOT acceptance budgets, optima,
CPU policy outcomes, or bounds on achievable quality. The caller chooses and
labels a utility mapping separately. Compilation never instantiates a backend.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path

import yaml

from evaluation.mission_contracts import _digest, _read_locked
from evaluation.leaderboard import _physical_source_identity

PROTOCOL_REVISION = "operational_utility.source_scales.v1"


# Explicit component names prevent a new backend loss being silently ignored.
# Historical settlement absence remains an evidence-quality concern for callers.
_COMPONENTS = {
    "sumo_ego": [
        "collision_cost",
        "road_departure_cost",
        "risk_exposure_cost",
        "route_delay_cost",
        "comfort_cost",
        "mrm_cost",
        "mrm_failure_cost",
        "shield_intervention_cost",
    ],
    "citylearn": ["energy_cost", "terminal_storage_settlement"],
    "dynasched_flexible_job_shop": ["production_cost"],
    "orgym_invmgmt": [
        "inventory_procurement_cost",
        "inventory_holding_cost",
        "inventory_lost_sales_penalty",
        "inventory_actuation_cost",
        "inventory_asset_settlement",
    ],
    "pyvrp_cvrp": [
        "routing_operating_cost",
        "vehicle_dispatch_fixed_cost",
        "drop_order_penalty",
        "unmet_demand_cost",
    ],
    "alibaba_trace_sim": [
        "compute_cost",
        "queue_wait_cost",
        "sla_violation_cost",
        "unfinished_work_penalty",
        "preemption_waste_cost",
        "reserve_capacity_cost",
    ],
    "pymgrid_economic_dispatch": [
        "production_cost",
        "startup_cost",
        "shed_penalty",
        "balance_error_cost",
    ],
    "pandapower_lv": [
        "production_cost",
        "startup_cost",
        "shed_penalty",
        "voltage_violation_cost",
        "overload_cost",
        "disconnection_cost",
        "terminal_storage_settlement",
    ],
    "cigre_distribution": [
        "production_cost",
        "startup_cost",
        "shed_penalty",
        "voltage_violation_cost",
        "overload_cost",
        "disconnection_cost",
        "balance_error_cost",
        "reserve_violation_cost",
    ],
    "pglib_uc_synthetic": [
        "production_cost",
        "startup_cost",
        "shed_penalty",
        "balance_error_cost",
        "reserve_violation_cost",
    ],
    "opendss_ieee13": [
        "production_cost",
        "voltage_violation_cost",
        "voltage_band_deviation_cost",
    ],
}
_COMPONENTS["pyvrp_vrptw"] = _COMPONENTS["pyvrp_cvrp"]
_COMPONENTS["opendss_fresh_feeders"] = _COMPONENTS["opendss_ieee13"]


def _scale(
    value: float, formula: str, evidence: list[str], unit: str = "native_cost"
) -> dict:
    if not math.isfinite(value) or value <= 0:
        raise ValueError("invalid_source_utility_scale")
    return dict(value=float(value), formula=formula, evidence=evidence, unit=unit)


def _compile(scenario: dict, row: dict, root: Path, cache: dict) -> dict:
    for key in (
        "scenario_signature",
        "seed",
        "domain",
        "backend_kind",
        "horizon_ticks",
    ):
        if scenario.get(key) != row.get(key):
            raise ValueError("scenario_suite_identity_mismatch")
    if scenario.get("family") != row.get("family"):
        raise ValueError("scenario_suite_family_mismatch")
    h = scenario["horizon_ticks"]
    if type(h) is not int or h <= 0:
        raise ValueError("invalid_source_horizon")
    b = scenario["backend_kind"]
    c = scenario.get("backend_config") or {}
    artifacts = [dict(path=row["path"], sha256=row["yaml_sha256"], role="scenario")]
    file_hashes = (scenario.get("source_contract") or {}).get("file_sha256s") or {}

    def read(path: str, expected: str | None = None) -> bytes:
        expected = expected or file_hashes.get(path)
        if not expected:
            raise ValueError("consumed_source_hash_missing")
        key = (str(root.resolve()), path, expected)
        if key not in cache:
            cache[key] = _read_locked(
                root, dict(path=path, sha256=expected), label="source_asset"
            )
        artifact = dict(path=path, sha256=expected, role="scale_source")
        if artifact not in artifacts:
            artifacts.append(artifact)
        return cache[key]

    def table(path: str) -> list[dict]:
        key = ("csv", str(root.resolve()), path, file_hashes.get(path))
        raw = read(path)
        if key not in cache:
            cache[key] = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        return cache[key]

    service = dict(applicable=True, kind="source_service_obligations")
    signed = False
    scales = {}
    if b == "sumo_ego":
        seconds = float(scenario.get("tick_seconds", 1))
        scales["native_cost"] = _scale(
            h * seconds,
            "horizon_times_tick_seconds",
            ["horizon_ticks", "tick_seconds"],
            "route_delay_seconds_cost",
        )
        scales["risk"] = _scale(
            h * 20,
            "one_unit_residual_risk_per_tick_times_20",
            [
                "horizon_ticks",
                "domains/autonomous_driving/backends/sumo_ego.py:ground_truth_costs",
            ],
        )
        service["kind"] = "safe_mobility_and_guarded_recovery"
    elif b == "citylearn":
        paths = [p for p in file_hashes if p.endswith("/schema.json")]
        if len(paths) != 1:
            raise ValueError("citylearn_schema_source_ambiguous")
        schema_path = paths[0]
        schema = json.loads(read(schema_path))
        base = Path(schema_path).parent
        first = int(c.get("simulation_start_time_step", 0))
        # Runtime records use the pre-step source index (first through first+h-1).
        indices = range(first, first + h)
        value = 0.0
        source_load = [0.0] * h
        buildings_count = 0
        zero_priced = False
        for building in schema["buildings"].values():
            if building.get("include", True) is not True:
                continue
            energy = table(str(base / building["energy_simulation"]))
            prices = (
                table(str(base / building["pricing"]))
                if building.get("pricing")
                else None
            )
            zero_priced |= prices is None
            buildings_count += 1
            for i in indices:
                if i >= len(energy) or (prices is not None and i >= len(prices)):
                    raise ValueError("citylearn_source_window_out_of_bounds")
                load = max(0.0, float(energy[i]["non_shiftable_load"]))
                source_load[i - first] += load
                if prices is not None:
                    value += load * abs(float(prices[i]["electricity_pricing"]))
        if c.get("native_peak_response_objective"):
            windows = c["task_contract"]["response_windows"]
            peak = sum(
                load * load / buildings_count
                for tick, load in enumerate(source_load)
                if any(
                    w.get("expected_control_policy") == "discharge"
                    and w["first_tick"] <= tick <= w["last_tick"]
                    for w in windows
                )
            )
            value += peak
        scales["native_cost"] = _scale(
            value,
            "source_tariff_cost_plus_declared_peak_response_load_squared"
            if c.get("native_peak_response_objective")
            else "sum_source_nonshiftable_load_times_absolute_tariff",
            [
                "schema.buildings.include",
                "non_shiftable_load",
                "electricity_pricing",
                "simulation_start_time_step",
                "horizon_ticks",
                "backend_config.native_peak_response_objective",
                "backend_config.task_contract.response_windows",
            ],
        )
        service = dict(
            applicable=False,
            kind="automatic_exogenous_load_service",
            reason="automatic_service_is_not_agent_completion",
        )
        signed = not zero_priced
    elif b == "dynasched_flexible_job_shop":
        asset = c["source_assets"]["static_jobs_json"]
        jobs = json.loads(read(asset["path"], asset["sha256"]))["jobs"]
        service.update(
            kind="complete_source_operations",
            jobs_total=len(jobs),
            operations_total=sum(len(j["routing"]) for j in jobs.values()),
        )
        model_asset = c["source_assets"]["input_model_json"]
        model = json.loads(read(model_asset["path"], model_asset["sha256"]))
        # Decision ticks jump between native event boundaries; they are not time.
        scales["native_cost"] = _scale(
            float(model["scale"]["horizon"]),
            "locked_source_native_time_horizon",
            ["backend_config.source_assets.input_model_json:scale.horizon"],
            "native_time",
        )
        scales["unfinished_operation"] = _scale(
            1000,
            "native_unfinished_operation_penalty",
            [
                "domains/logistics/backends/dynasched_flexible_job_shop.py:ground_truth_costs"
            ],
        )
    elif b == "orgym_invmgmt":
        cfg = c["orgym_env_config"]
        demand = (
            sum(float(x) for x in cfg["user_D"][:h])
            if "user_D" in cfg
            else float(c["m5_demand_sum_units"])
        )
        price = sum(float(x) for x in cfg["r"][:-1])
        scales["native_cost"] = _scale(
            demand * price,
            "source_demand_times_procurement_unit_cost",
            [
                "backend_config.orgym_env_config.user_D",
                "backend_config.orgym_env_config.r",
            ],
        )
        scales["loss"] = _scale(
            demand * float(cfg["k"][0]),
            "source_demand_times_lost_sale_penalty",
            [
                "backend_config.orgym_env_config.user_D",
                "backend_config.orgym_env_config.k",
            ],
        )
        service.update(kind="source_customer_demand", demand_total=demand)
        signed = True
    elif b in ("pyvrp_cvrp", "pyvrp_vrptw"):
        network = c["network"]
        depot = network["depot"]
        customers = network["customers"]
        distance = sum(
            2
            * math.hypot(
                float(x["x"]) - float(depot["x"]), float(x["y"]) - float(depot["y"])
            )
            for x in customers
        )
        scales["native_cost"] = _scale(
            distance + 100 * float(network["n_vehicles"]),
            "sum_depot_customer_roundtrip_distance_plus_fleet_dispatch_cost",
            [
                "backend_config.network",
                "domains/logistics/backends/route_sim.py:_COST_PER_DISTANCE",
                "domains/logistics/backends/route_sim.py:_DISPATCH_FIXED_COST",
            ],
        )
        scales["loss"] = _scale(
            sum(float(x["demand"]) for x in customers) * h * 200,
            "source_demand_times_horizon_times_native_unmet_penalty",
            [
                "backend_config.network.customers",
                "horizon_ticks",
                "domains/logistics/backends/route_sim.py:_DROP_PENALTY_BASE",
            ],
        )
        service.update(
            kind="source_delivery_demand",
            customers_total=len(customers),
            demand_total=sum(float(x["demand"]) for x in customers),
        )
    elif b == "alibaba_trace_sim":
        transform = c["source_transform"]
        count = int(transform["window_size"])
        scales["native_cost"] = _scale(
            count * h,
            "source_jobs_times_horizon_queue_unit_cost",
            [
                "backend_config.source_transform.window_size",
                "horizon_ticks",
                "domains/datacenter/backends/alibaba_trace_backend.py:queue_wait_cost",
            ],
        )
        service.update(kind="source_job_service", jobs_total=count)
    elif b == "pymgrid_economic_dispatch":
        profiles = c["profiles"]
        load = [float(x) for x in profiles["load_mw"][:h]]
        price = [abs(float(x)) for x in profiles["price"][:h]]
        if len(load) != h or len(price) != h:
            raise ValueError("source_profile_horizon_mismatch")
        hours = float(scenario.get("tick_minutes", 60)) / 60
        scales["native_cost"] = _scale(
            sum(x * p * hours for x, p in zip(load, price, strict=True)),
            "source_load_energy_times_absolute_tariff",
            [
                "backend_config.profiles.load_mw",
                "backend_config.profiles.price",
                "tick_minutes",
            ],
        )
        scales["loss"] = _scale(
            sum(load) * 200,
            "source_load_times_native_balance_loss_coefficient",
            [
                "backend_config.profiles.load_mw",
                "backend_config.native_state_loss_task.task_loss_formula",
            ],
        )
        service.update(
            kind="critical_supply_and_balance", demand_total=sum(load) * hours
        )
        scales["resource"] = scales["native_cost"]
        scales["native_cost"] = scales["loss"]
        signed = False
    elif b in ("pandapower_lv", "cigre_distribution"):
        scales["native_cost"] = _scale(
            h * 1200,
            "one_voltage_violation_per_tick_times_native_penalty",
            [
                "horizon_ticks",
                f"domains/{'microgrid' if b == 'pandapower_lv' else 'power_grid'}/backends/{b}.py:VOLTAGE_VIOLATION_COST_PER_TICK",
            ],
        )
        service.update(
            kind="voltage_reliability", denominator_kind="monitored_node_time"
        )
        signed = b == "pandapower_lv"
    elif b in ("opendss_ieee13", "opendss_fresh_feeders"):
        scales["native_cost"] = _scale(
            h * 100,
            "one_voltage_violation_per_tick_times_native_penalty",
            [
                "horizon_ticks",
                "domains/power_grid/backends/opendss_ieee13.py:ground_truth_costs",
            ],
        )
        service.update(
            kind="voltage_reliability", denominator_kind="monitored_node_time"
        )
    elif b == "pglib_uc_synthetic":
        case = json.loads(read(c["case_file"]))
        first = int(c.get("first_period", 1)) - 1
        demand = [float(x) for x in case["demand"][first : first + h]]
        if len(demand) != h:
            raise ValueError("source_profile_horizon_mismatch")
        generators = case["thermal_generators"].values()
        capacity = sum(float(g["power_output_maximum"]) for g in generators)
        weighted_price = (
            sum(
                float(g["power_output_maximum"])
                * (
                    float(g["piecewise_production"][0]["cost"])
                    / max(float(g["piecewise_production"][0]["mw"]), 1)
                    if g.get("piecewise_production")
                    else 30.0
                )
                for g in generators
            )
            / capacity
        )
        scales["native_cost"] = _scale(
            sum(demand) * weighted_price,
            "source_demand_times_capacity_weighted_native_unit_cost",
            [
                "backend_config.case_file:demand",
                "thermal_generators.piecewise_production",
                "domains/power_grid/backends/pglib_uc_synthetic.py:_production_cost",
            ],
        )
        scales["loss"] = _scale(
            sum(demand) * 200,
            "source_demand_times_native_balance_penalty",
            [
                "backend_config.case_file:demand",
                "domains/power_grid/backends/pglib_uc_synthetic.py:BALANCE_ERROR_PENALTY_PER_MW",
            ],
        )
        service.update(kind="load_and_reserve_service", demand_total=sum(demand))
    else:
        raise ValueError(f"unsupported_operational_backend:{b}")
    components = list(_COMPONENTS[b])
    if b == "citylearn" and c.get("native_peak_response_objective"):
        components += [
            "source_event_peak_response_burden",
            "native_storage_charging_burden",
        ]
    result = {
        "schema_version": PROTOCOL_REVISION,
        "evaluation_version": "0.25.0",
        **{
            k: row[k]
            for k in (
                "scenario_signature",
                "seed",
                "domain",
                "backend_kind",
                "horizon_ticks",
                "scenario_id",
                "source_denominator_key",
            )
        },
        "scenario_path": row["path"],
        "scenario_sha256": row["yaml_sha256"],
        "task_family": row.get("family", scenario.get("family")),
        "physical_source_cluster": _physical_source_identity(row),
        "eligible": True,
        "scales": scales,
        "service": service,
        "allowed_cost_components": components,
        "required_cost_components": components,
        "cost_component_vocabulary": "backend_ground_truth_costs_before_counterfactual_aliases",
        "objective_reduction": (
            "native_state_loss_from_records"
            if b == "pymgrid_economic_dispatch"
            else "sum_settled_native_components"
        ),
        "normalization_interpretation": "declared_unit_loss_scale_not_success_threshold_or_optimality",
        "native_cost_value_domain": "signed" if signed else "nonnegative",
        "scale_is_acceptance_threshold": False,
        "scale_is_optimum": False,
        "model_outcomes_used": False,
        "baseline_execution_required": False,
        "importance_policy": "retain_native_penalties_and_source_priority_no_invented_customer_classes",
        "source_artifacts": artifacts,
    }
    if b == "citylearn":
        result["zero_priced_source"] = zero_priced
        result["native_peak_objective"] = c.get("native_peak_response_objective")
    result["contract_sha256"] = _digest(result)
    return result


def compile_operational_suite(
    suite_path: str | Path, *, root: str | Path = "."
) -> dict:
    """Compile authenticated source scales; no outcome or calibration inputs."""
    raw = Path(suite_path).read_bytes()
    suite = json.loads(raw)
    rows = suite["scenarios"]
    identities = [(r["scenario_signature"], r["seed"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate_suite_case")
    root = Path(root)
    contracts = []
    cache = {}
    for row in rows:
        scenario_raw = _read_locked(
            root, dict(path=row["path"], sha256=row["yaml_sha256"]), label="scenario"
        )
        contracts.append(_compile(yaml.safe_load(scenario_raw), row, root, cache))
    return dict(
        schema_version=PROTOCOL_REVISION,
        evaluation_version="0.25.0",
        suite_sha256=hashlib.sha256(raw).hexdigest(),
        contracts=contracts,
        new_execution_required=False,
        coverage=dict(
            suite_cases=len(contracts),
            utility_contracts=len(contracts),
            by_backend=dict(Counter(c["backend_kind"] for c in contracts)),
        ),
    )
