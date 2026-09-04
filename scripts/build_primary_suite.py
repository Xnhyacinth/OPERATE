"""Shared source-identity and decision-axis helpers for Core materialization.

The historical primary-suite builder was retired during the OPERATE namespace
migration.  Current materialization and candidate tooling still share these
small, deterministic helpers, so they live here without the retired CLI and
selection implementation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


SEED_ONLY_FIELDS = {"seed", "seed_id"}


def _without_seed_only(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _without_seed_only(item)
            for key, item in sorted(value.items())
            if key not in SEED_ONLY_FIELDS
        }
    if isinstance(value, list):
        return [_without_seed_only(item) for item in value]
    return value


def structural_fingerprint(body: dict[str, Any]) -> str:
    """Return the legacy-compatible, seed-independent structure hash."""
    normalized = json.dumps(
        _without_seed_only(body),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _physical_source_key(row: dict[str, Any]) -> str:
    """Return the legacy-compatible physical-source grouping key."""
    backend = str(row.get("backend_kind"))
    axes = row.get("source_axes") or {}
    if backend == "pandapower_acopf":
        return f"opf_case:{axes.get('case_name')}"
    if backend == "grid2op":
        window = "|".join(
            str(axes.get(key)) for key in ("env_name", "chronics_id", "start_step")
        )
        return f"chronic_window:{window}"
    if backend == "cigre_distribution":
        return f"feeder:{axes.get('network') or row.get('provenance_source')}"
    if backend in {
        "pandapower_lv",
        "pymgrid_islanding",
        "pymgrid_economic_dispatch",
        "pymgrid_solar_ramp",
    }:
        return (
            "microgrid_site:"
            f"{axes.get('backend')}|{axes.get('site')}|{axes.get('forecast_regime_idx')}"
        )
    if backend in {"pyvrp_cvrp", "pyvrp_vrptw", "pyvrp_lastmile"}:
        return (
            "routing_instance:"
            f"{axes.get('backend')}|{axes.get('instance_name')}|"
            f"{axes.get('has_time_windows')}"
        )
    if backend == "jsplib_job_shop":
        return f"job_shop_instance:{axes.get('instance_name')}"
    if backend == "orgym_invmgmt":
        return (
            "inventory_environment:"
            f"{axes.get('inventory_environment_id')}:{axes.get('demand_profile_id')}"
        )
    if backend == "opendss_ieee13":
        return f"feeder:opendss_ieee13:{axes.get('feeder')}"
    if backend == "opendss_fresh_feeders":
        return f"feeder:opendss_fresh_feeders:{axes.get('feeder')}"
    if backend in {"mock_sumo", "sumo"}:
        denominator = (row.get("case_ledger") or {}).get("source_denominator_key")
        return f"traffic_corridor_source:{denominator}"
    denominator = (row.get("case_ledger") or {}).get("source_denominator_key")
    return f"uc_source_window:{denominator}"


def _independence_axis(row: dict[str, Any]) -> str:
    """Describe the backend-native unit of source independence."""
    backend = str(row.get("backend_kind"))
    if backend == "grid2op":
        return "chronic_window"
    if backend == "pandapower_acopf":
        return "opf_case_x_mode_x_level"
    if backend == "cigre_distribution":
        return "distribution_topology"
    if backend in {
        "pandapower_lv",
        "pymgrid_islanding",
        "pymgrid_economic_dispatch",
        "pymgrid_solar_ramp",
    }:
        return "microgrid_site_profile"
    if backend in {"pyvrp_cvrp", "pyvrp_vrptw", "pyvrp_lastmile"}:
        return "routing_instance"
    if backend == "jsplib_job_shop":
        return "job_shop_instance"
    if backend == "dynasched_flexible_job_shop":
        return "dynamic_flexible_job_shop_instance_event_bundle"
    if backend == "orgym_invmgmt":
        return "orgym_inventory_environment_config"
    if backend == "citylearn":
        return "citylearn_dataset_building_window"
    if backend in {"mock_sumo", "sumo"}:
        return "traffic_network_route_and_demand_window"
    if backend == "alibaba_openb_gpu_placement":
        return "gpu_node_pod_trace_graph"
    if backend == "opendss_ieee13":
        return "opendss_feeder_x_control_object"
    if backend == "opendss_fresh_feeders":
        return "opendss_fresh_feeder_topology"
    if backend == "pglib_uc_synthetic":
        return "uc_source_window"
    return "source_key"


def _decision_pressure_axis(row: dict[str, Any], body: dict[str, Any]) -> str:
    """Describe the backend-native operational decision pressure."""
    backend = str(row.get("backend_kind"))
    family = str(row.get("family"))
    perturbation_kinds = {
        str(item.get("kind"))
        for item in body.get("perturbations") or []
        if item.get("kind")
    }
    if backend == "grid2op":
        if "opponent_attack" in perturbation_kinds:
            return "opponent_attack_and_overload_recovery"
        return "line_outage_and_storm_overload_recovery"
    if backend == "pandapower_acopf":
        backend_config = body.get("backend_config") or {}
        if (
            backend_config.get("v0_30_probe_id")
            == "acopf_cross_tick_commitment_or_reserve_mechanism"
        ):
            return "acopf_cross_tick_commitment_or_reserve_mechanism"
        return "acopf_feasibility_cost_voltage_and_line_loading"
    if backend == "cigre_distribution":
        return "voltage_band_der_outage_and_load_shed_tradeoff"
    if backend == "pandapower_lv":
        return "lv_voltage_band_pv_ramp_and_reactive_power_control"
    if backend.startswith("pymgrid_"):
        return "microgrid_energy_balance_storage_genset_and_forecast_control"
    if backend in {"pyvrp_cvrp", "pyvrp_vrptw", "pyvrp_lastmile"}:
        return "route_capacity_time_window_and_priority_replanning"
    if backend == "jsplib_job_shop":
        return "operation_precedence_machine_capacity_and_makespan_optimization"
    if backend == "dynasched_flexible_job_shop":
        return "machine_breakdown_dynamic_arrival_priority_and_rescheduling"
    if backend == "orgym_invmgmt":
        return "lost_sales_replenishment_lead_time_and_capacity_tradeoff"
    if backend == "citylearn":
        return "building_storage_load_solar_price_and_carbon_dispatch"
    if backend in {"mock_sumo", "sumo"}:
        return "traffic_signal_timing_demand_spillback_and_incident_control"
    if backend == "alibaba_openb_gpu_placement":
        return "gpu_placement_fragmentation_queue_wait_and_sla_tradeoff"
    if backend == "opendss_ieee13":
        return "distribution_voltage_band_control_under_unbalanced_load"
    if backend == "opendss_fresh_feeders":
        return "distribution_voltage_band_control_under_unbalanced_three_phase_load"
    if family == "reserve_stress_24h":
        return "reserve_scarcity_and_generator_outage"
    if family == "wind_uncertainty_24h":
        return "wind_dropout_and_forecast_uncertainty"
    if family == "daily_ops_real_forecast_24h":
        return "day_ahead_to_real_time_forecast_error"
    if family == "critical_winter_peak":
        return "winter_peak_load_and_capacity_margin"
    return "unit_commitment_cost_reserve_and_stakeholder_tradeoff"
