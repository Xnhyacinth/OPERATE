"""
domains.registry — domain dispatch for the shared runner / audit (T0).

Maps ``scenario['domain']`` → a :class:`DomainSpec` that the
domain-agnostic ``run.py`` / ``audit.py`` use to:

    * construct the right ``<Domain>Environment`` (``env_factory``),
    * rebuild a seed from its dict form for signature recomputation
      (``rebuild_seed_from_dict``),
    * read the per-entity equity map out of ``ground_truth()`` under the
      domain's canonical key (``equity_shed_key``),
    * locate the oracle ``reference_optimum`` that feeds
      ``ScoringInputs.lp_optimum`` (``reference_optimum``).

Design constraints (see AGENTS.md red lines):

* **Lazy imports.** Resolving one domain must never import another
  domain's heavy backend deps (grid2op, pandapower, pymgrid, PyVRP,
  SUMO, RCRS). The adapter module is imported only when its factory is
  actually called, so a Traffic run never pulls in Grid2Op.
* **No new cross-domain edges.** This aggregator lives at the
  ``domains/`` package root, not inside any ``domains/<x>/``; individual
  domains still import only ``core`` + their own backend libraries.
* **power_grid parity.** The power-grid path is intentionally identical
  to the pre-T0 hard-wired runner (same env, same backend-records
  helper, same LP/AC-OPF oracle wiring lives in ``run.py``), so the
  published power-grid releases score byte-for-byte unchanged.

The four v0.7 domains (traffic / microgrid / logistics / disaster) all
cache their oracle into ``seed.backend_config['reference_optimum']`` with
a uniform envelope ``{"reference_optimum": <float>, "method": ..., ...}``;
power_grid instead computes its LP / AC-OPF optimum inside ``run.py`` and
sets ``uses_runner_lp_oracle=True`` so the registry does not double-read.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

DEFAULT_DOMAIN = "power_grid"


@dataclass(frozen=True)
class DomainSpec:
    """Per-domain dispatch descriptor consumed by the shared runner/audit."""

    domain: str
    adapter_module: str
    env_class: str
    #: ``ground_truth()`` key holding the per-entity unmet/shed/delay map
    #: that feeds the equity/fairness dimension.
    equity_shed_key: str
    #: When True the LP / AC-OPF optimum is computed inside ``run.py``
    #: (power_grid only) and the registry must NOT also read
    #: ``backend_config['reference_optimum']``.
    uses_runner_lp_oracle: bool = False
    #: Realized ``ground_truth.cost_components`` key minimized by the
    #: reference optimum. ``None`` means the domain has no common objective
    #: contract and optimality scoring must remain inapplicable.
    objective_cost_component: str | None = None
    #: Legacy scorer-row key carrying a lower-is-better native recovery burden.
    adaptive_recovery_signal_key: str = "balance_error_mw"
    #: Reader-facing native meaning; never describe non-grid values as MW.
    adaptive_recovery_signal_name: str = "native_operational_burden"

    def env_factory(self) -> Callable[[], Any]:
        """Return the ``<Domain>Environment`` class (lazy import)."""
        mod = importlib.import_module(self.adapter_module)
        return getattr(mod, self.env_class)

    def rebuild_seed_from_dict(
        self, scenario: dict[str, Any], override_seed: int
    ) -> Any:
        """Rebuild a typed seed from its dict form for signature recompute."""
        mod = importlib.import_module(self.adapter_module)
        fn = mod._rebuild_seed_from_dict
        return fn(scenario, override_seed)

    def scenario_signature(self, scenario: dict[str, Any], seed: int) -> str:
        """Recompute the scenario signature under the realized run seed."""
        return self.rebuild_seed_from_dict(scenario, int(seed)).signature()


@dataclass(frozen=True)
class BackendCapabilitySpec:
    """Static, domain-owned contract for one executable backend surface."""

    backend_kind: str
    fidelity_contract: str
    native_state_fields: tuple[str, ...]
    observation_tools: tuple[str, ...]
    control_tools: tuple[str, ...]
    clock_semantics: str
    source_contract_builder: str
    source_evidence_adapter: str
    source_consumption_mode: Literal[
        "direct_runtime_files",
        "derived_source_window",
        "native_include_graph",
        "declared_source_unused",
    ]
    decision_cadence_mode: Literal[
        "event_driven",
        "periodic",
        "pending_action",
        "hybrid",
    ]
    source_scheduled_event_types: tuple[str, ...]
    runtime_fidelity: str
    formal_core_allowed: bool
    periodic_scan_every_ticks: int = 0
    max_review_after_ticks: int = 0
    # Controls whose native state remains in force until a later control
    # replaces it.  This is deliberately explicit: an arbitrary successful
    # state-changing call is not evidence of a standing plan.
    persistent_control_tools: tuple[str, ...] = ()

    @property
    def source_consumption_adapter(self) -> str | None:
        """Compatibility alias for protocol-2.0 report readers."""
        return self.source_evidence_adapter


def resolve_backend_source_evidence_adapter(
    capability: BackendCapabilitySpec,
) -> Callable[..., dict[str, Any]]:
    """Resolve the registry-bound source extractor, failing closed."""
    binding = str(capability.source_evidence_adapter or "").strip()
    if not binding or ":" not in binding:
        raise ImportError(
            f"source evidence adapter is unimplemented for "
            f"{capability.backend_kind!r}"
        )
    module_name, attribute = binding.split(":", 1)
    if not module_name or not attribute:
        raise ImportError(f"invalid source evidence adapter binding: {binding!r}")
    module = importlib.import_module(module_name)
    extractor = getattr(module, attribute)
    if not callable(extractor):
        raise TypeError(f"source evidence adapter is not callable: {binding!r}")
    return extractor


def resolve_backend_source_contract_builder(
    capability: BackendCapabilitySpec,
) -> Callable[..., dict[str, Any]]:
    """Resolve the backend-owned source-contract builder, failing closed."""
    binding = str(capability.source_contract_builder or "").strip()
    if not binding or ":" not in binding:
        raise ImportError(
            f"source contract builder is unimplemented for "
            f"{capability.backend_kind!r}"
        )
    module_name, attribute = binding.split(":", 1)
    if not module_name or not attribute:
        raise ImportError(f"invalid source contract builder binding: {binding!r}")
    module = importlib.import_module(module_name)
    builder = getattr(module, attribute)
    if not callable(builder):
        raise TypeError(f"source contract builder is not callable: {binding!r}")
    return builder


# ── Registry ────────────────────────────────────────────────────────────────
# Keyed by the lower-cased ``scenario['domain']`` value. Adapter modules are
# import strings (resolved lazily) so importing this registry stays cheap.
_REGISTRY: dict[str, DomainSpec] = {
    "power_grid": DomainSpec(
        domain="power_grid",
        adapter_module="domains.power_grid.adapter",
        env_class="PowerGridEnvironment",
        equity_shed_key="per_load_shed_mwh",
        uses_runner_lp_oracle=True,
        objective_cost_component="production_cost",
        adaptive_recovery_signal_name="power_balance_violation",
    ),
    "traffic": DomainSpec(
        domain="traffic",
        adapter_module="domains.traffic.adapter",
        env_class="TrafficEnvironment",
        equity_shed_key="per_corridor_delay_minutes",
        objective_cost_component="travel_time_cost",
        adaptive_recovery_signal_name="traffic_queue_pressure",
    ),
    "microgrid": DomainSpec(
        domain="microgrid",
        adapter_module="domains.microgrid.adapter",
        env_class="MicrogridEnvironment",
        equity_shed_key="per_load_shed_mwh",
        objective_cost_component="production_cost",
        adaptive_recovery_signal_name="microgrid_energy_balance_violation",
    ),
    "logistics": DomainSpec(
        domain="logistics",
        adapter_module="domains.logistics.adapter",
        env_class="LogisticsEnvironment",
        equity_shed_key="per_customer_unmet_units",
        objective_cost_component="routing_operating_cost",
        adaptive_recovery_signal_name="unmet_work_or_demand",
    ),
    "datacenter": DomainSpec(
        domain="datacenter",
        adapter_module="domains.datacenter.adapter",
        env_class="DatacenterEnvironment",
        equity_shed_key="per_job_sla_violation_minutes",
        adaptive_recovery_signal_name="queued_sla_exposure",
    ),
    "disaster": DomainSpec(
        domain="disaster",
        adapter_module="domains.disaster.adapter",
        env_class="DisasterEnvironment",
        equity_shed_key="per_zone_unserved_minutes",
        adaptive_recovery_signal_name="unresolved_rescue_burden",
    ),
    # Building Energy remains a domain-native surface. CityLearn candidates
    # may enter formal Protocol-2.1; row admission and release coverage still
    # fail closed on their replay-bound evidence rather than this registry bit.
    "building_energy": DomainSpec(
        domain="building_energy",
        adapter_module="domains.building_energy.adapter",
        env_class="BuildingEnergyEnvironment",
        equity_shed_key="per_building_unserved_units",
        objective_cost_component="energy_cost",
        adaptive_recovery_signal_key="native_operational_burden",
        adaptive_recovery_signal_name="building_energy_net_load",
    ),
    "autonomous_driving": DomainSpec(
        domain="autonomous_driving",
        adapter_module="domains.autonomous_driving.adapter",
        env_class="AutonomousDrivingEnvironment",
        equity_shed_key="road_user_harm",
        objective_cost_component=None,
        adaptive_recovery_signal_key="residual_risk_burden",
        adaptive_recovery_signal_name="driving_risk_exposure",
    ),
}

_BACKEND_CAPABILITIES: dict[str, BackendCapabilitySpec] = {
    "alibaba_trace_sim": BackendCapabilitySpec(
        "alibaba_trace_sim",
        "deterministic_trace_driven_gpu_cluster_simulator",
        ("queued_jobs", "running_jobs", "available_gpu_units"),
        ("query_job_queue", "query_cluster_capacity", "forecast_trace_arrivals"),
        ("set_queue_policy", "preempt_job", "reserve_gpu_capacity"),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:alibaba_trace_sim",
        source_evidence_adapter="domains.source_evidence:alibaba_trace_sim",
        source_consumption_mode="derived_source_window",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=("job_arrival", "capacity_change"),
        runtime_fidelity="source_driven_simulator",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=4,
        persistent_control_tools=("set_queue_policy", "reserve_gpu_capacity"),
    ),
    "alibaba_openb_gpu_placement": BackendCapabilitySpec(
        "alibaba_openb_gpu_placement",
        "deterministic_openb_multi_resource_gpu_placement_simulator",
        (
            "pod_assignments",
            "queued_pods",
            "node_resource_allocation",
            "placement_fragmentation",
            "qos_delay_risk",
        ),
        ("query_node_placements", "forecast_pod_arrivals"),
        ("set_placement_policy", "place_pod", "migrate_pod"),
        clock_semantics="simulator_owned",
        source_contract_builder=(
            "domains.source_contracts:alibaba_openb_gpu_placement"
        ),
        source_evidence_adapter=(
            "domains.source_evidence:alibaba_openb_gpu_placement"
        ),
        source_consumption_mode="direct_runtime_files",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=("pod_arrival",),
        runtime_fidelity="source_driven_simulator",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=4,
        persistent_control_tools=("set_placement_policy",),
    ),
    "jsplib_job_shop": BackendCapabilitySpec(
        "jsplib_job_shop",
        "deterministic_job_shop_instance_simulator",
        (
            "machine_available_at",
            "job_next_operation",
            "makespan",
            "active_machine_disruptions",
        ),
        ("query_job_queue",),
        (
            "dispatch_ready_operations",
            "dispatch_job_operation",
            "repair_machine",
        ),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:jsplib_job_shop",
        source_evidence_adapter="domains.source_evidence:jsplib_job_shop",
        source_consumption_mode="derived_source_window",
        decision_cadence_mode="pending_action",
        source_scheduled_event_types=("machine_breakdown",),
        runtime_fidelity="source_driven_simulator",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=4,
    ),
    "dynasched_flexible_job_shop": BackendCapabilitySpec(
        "dynasched_flexible_job_shop",
        "official_dynasched_dynamic_flexible_job_shop_simulator",
        (
            "jobs",
            "machines",
            "ready_operations",
            "event_counters",
        ),
        ("query_flexible_job_shop",),
        ("dispatch_flexible_operations",),
        clock_semantics="simulator_owned",
        source_contract_builder=(
            "domains.logistics.backends.dynasched_flexible_job_shop:"
            "build_dynasched_source_contract"
        ),
        source_evidence_adapter=(
            "domains.logistics.backends.dynasched_flexible_job_shop:"
            "extract_dynasched_source_evidence"
        ),
        source_consumption_mode="direct_runtime_files",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=(
            "job_arrival",
            "machine_breakdown",
            "priority_change",
            "process_time_change",
            "route_change",
            "order_cancellation",
            "preventive_maintenance",
        ),
        runtime_fidelity="native_library",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=4,
    ),
    "mock_sumo": BackendCapabilitySpec(
        "mock_sumo",
        "deterministic_corridor_traffic_simulator",
        ("corridor_queues", "signal_programs", "travel_time_cost"),
        ("query_network_state", "query_detector", "inspect_intersection"),
        ("change_signal_plan", "reroute_flow", "meter_inflow"),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:mock_sumo",
        source_evidence_adapter="domains.source_evidence:mock_sumo",
        source_consumption_mode="declared_source_unused",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=("traffic_demand_change", "incident"),
        runtime_fidelity="mock",
        formal_core_allowed=False,
    ),
    "opendss_fresh_feeders": BackendCapabilitySpec(
        "opendss_fresh_feeders",
        "opendss_unbalanced_distribution_power_flow",
        ("bus_voltage_pu", "line_loading_percent", "tap_positions"),
        ("query_grid_state", "investigate_substation"),
        ("set_transformer_tap", "switch_capacitor", "switch_branch"),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:opendss_fresh_feeders",
        source_evidence_adapter="domains.source_evidence:opendss_fresh_feeders",
        source_consumption_mode="native_include_graph",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=(
            "load_change",
            "generation_ramp",
            "asset_outage",
        ),
        runtime_fidelity="native_library",
        formal_core_allowed=True,
        periodic_scan_every_ticks=1,
        max_review_after_ticks=2,
        persistent_control_tools=(
            "set_transformer_tap",
            "switch_capacitor",
            "switch_branch",
        ),
    ),
    "opendss_ieee13": BackendCapabilitySpec(
        "opendss_ieee13",
        "opendss_ieee13_unbalanced_distribution_power_flow",
        ("bus_voltage_pu", "line_loading_percent", "tap_positions"),
        ("query_grid_state", "investigate_substation"),
        ("set_transformer_tap", "switch_capacitor"),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:opendss_ieee13",
        source_evidence_adapter="domains.source_evidence:opendss_ieee13",
        source_consumption_mode="native_include_graph",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=("load_change", "asset_outage"),
        runtime_fidelity="native_library",
        formal_core_allowed=True,
        periodic_scan_every_ticks=1,
        max_review_after_ticks=2,
        persistent_control_tools=(
            "set_transformer_tap",
            "switch_capacitor",
        ),
    ),
    "orgym_invmgmt": BackendCapabilitySpec(
        "orgym_invmgmt",
        "deterministic_orgym_inventory_transition_model",
        ("inventory", "pipeline_orders", "realized_demand"),
        ("forecast_demand",),
        ("place_replenishment_order",),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:orgym_invmgmt",
        source_evidence_adapter="domains.source_evidence:orgym_invmgmt",
        source_consumption_mode="derived_source_window",
        decision_cadence_mode="periodic",
        source_scheduled_event_types=("demand_realization", "delivery_arrival"),
        runtime_fidelity="source_driven_simulator",
        formal_core_allowed=True,
        periodic_scan_every_ticks=1,
        max_review_after_ticks=1,
        persistent_control_tools=("place_replenishment_order",),
    ),
    "pandapower_acopf": BackendCapabilitySpec(
        "pandapower_acopf",
        "pandapower_nonlinear_ac_optimal_power_flow",
        ("bus_voltage_pu", "line_loading_percent", "generator_dispatch_mw"),
        ("query_grid_state", "forecast_query", "investigate_substation"),
        (
            "redispatch_generation",
            "commit_reserve",
            "shed_load",
        ),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:pandapower_acopf",
        source_evidence_adapter="domains.source_evidence:pandapower_acopf",
        source_consumption_mode="direct_runtime_files",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=("demand_change", "generator_outage"),
        runtime_fidelity="native_library",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=4,
        persistent_control_tools=(
            "redispatch_generation",
            "commit_reserve",
        ),
    ),
    "pandapower_lv": BackendCapabilitySpec(
        "pandapower_lv",
        "pandapower_lv_ac_power_flow",
        ("bus_voltage_pu", "line_loading_percent", "der_dispatch"),
        ("forecast_query", "investigate_asset"),
        ("set_battery_dispatch", "curtail_der", "set_der_reactive_power"),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:pandapower_lv",
        source_evidence_adapter="domains.source_evidence:pandapower_lv",
        source_consumption_mode="derived_source_window",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=("load_change", "generation_change", "asset_outage"),
        runtime_fidelity="native_library",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=2,
        persistent_control_tools=(
            "set_battery_dispatch",
            "curtail_der",
            "set_der_reactive_power",
        ),
    ),
    "cigre_distribution": BackendCapabilitySpec(
        "cigre_distribution",
        "pandapower_cigre_distribution_power_flow",
        (
            "bus_voltage_pu",
            "line_loading_percent",
            "der_dispatch",
            "capacitor_states",
        ),
        ("query_grid_state", "forecast_query", "investigate_substation"),
        (
            "switch_capacitor",
            "set_der_reactive_power",
            "set_transformer_tap",
            "shed_load",
        ),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:cigre_distribution",
        source_evidence_adapter="domains.source_evidence:cigre_distribution",
        source_consumption_mode="derived_source_window",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=(
            "load_surge",
            "generator_forced_outage",
            "storm_window",
        ),
        runtime_fidelity="native_library",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=4,
        persistent_control_tools=(
            "switch_capacitor",
            "set_der_reactive_power",
            "set_transformer_tap",
        ),
    ),
    "pglib_uc_synthetic": BackendCapabilitySpec(
        "pglib_uc_synthetic",
        "deterministic_aggregate_unit_commitment_without_power_flow",
        ("generator_dispatch_mw", "demand_mw", "reserve_shortfall_mw"),
        ("query_grid_state", "query_chronics_window", "forecast_query"),
        (
            "redispatch_generation",
            "dispatch_generation_portfolio",
            "commit_reserve",
            "shed_load",
        ),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:pglib_uc_synthetic",
        source_evidence_adapter="domains.source_evidence:pglib_uc_synthetic",
        source_consumption_mode="direct_runtime_files",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=("demand_change", "maintenance_window"),
        runtime_fidelity="source_driven_simulator",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=4,
        persistent_control_tools=(
            "redispatch_generation",
        ),
    ),
    "pymgrid_economic_dispatch": BackendCapabilitySpec(
        "pymgrid_economic_dispatch",
        "deterministic_microgrid_economic_dispatch_simulator",
        ("battery_soc", "grid_exchange_mw", "unmet_load_mw"),
        ("forecast_query", "investigate_asset"),
        (
            "set_battery_dispatch",
            "dispatch_genset",
            "set_grid_exchange",
            "connect_pcc",
            "curtail_der",
            "shed_load",
        ),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:pymgrid_economic_dispatch",
        source_evidence_adapter="domains.source_evidence:pymgrid_economic_dispatch",
        source_consumption_mode="derived_source_window",
        decision_cadence_mode="periodic",
        source_scheduled_event_types=("load_change", "generation_change", "tariff_change"),
        runtime_fidelity="source_driven_simulator",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=4,
        persistent_control_tools=(
            "set_battery_dispatch",
            "dispatch_genset",
            "set_grid_exchange",
        ),
    ),
    "pyvrp_cvrp": BackendCapabilitySpec(
        "pyvrp_cvrp",
        "deterministic_capacitated_vehicle_routing_simulator",
        ("vehicle_routes", "unserved_orders", "route_cost"),
        ("query_eta", "forecast_demand"),
        ("assign_stop", "reroute_vehicle", "dispatch_vehicle"),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:pyvrp_cvrp",
        source_evidence_adapter="domains.source_evidence:pyvrp_cvrp",
        source_consumption_mode="derived_source_window",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=("order_arrival", "travel_time_change"),
        runtime_fidelity="source_driven_simulator",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=4,
        persistent_control_tools=("assign_stop", "reroute_vehicle"),
    ),
    "pyvrp_vrptw": BackendCapabilitySpec(
        "pyvrp_vrptw",
        "deterministic_vehicle_routing_with_time_windows_simulator",
        ("vehicle_routes", "unserved_orders", "time_window_violations"),
        ("query_eta", "forecast_demand"),
        ("assign_stop", "reroute_vehicle", "dispatch_vehicle"),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:pyvrp_vrptw",
        source_evidence_adapter="domains.source_evidence:pyvrp_vrptw",
        source_consumption_mode="derived_source_window",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=("order_arrival", "travel_time_change"),
        runtime_fidelity="source_driven_simulator",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=4,
        persistent_control_tools=("assign_stop", "reroute_vehicle"),
    ),
    "sumo": BackendCapabilitySpec(
        "sumo",
        "live_sumo_native_traffic_simulator",
        (
            "controlled_lane_queues",
            "controlled_lane_waiting_time",
            "signal_programs",
            "signal_phases",
            "native_throughput",
            "travel_time_cost",
        ),
        ("query_signal_control",),
        ("set_signal_program", "set_signal_phase_duration"),
        clock_semantics="simulator_owned",
        source_contract_builder="domains.source_contracts:sumo",
        source_evidence_adapter="domains.source_evidence:sumo",
        source_consumption_mode="direct_runtime_files",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=("traffic_demand_change", "incident"),
        runtime_fidelity="native_library",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=4,
        persistent_control_tools=(
            "set_signal_program",
        ),
    ),
    "citylearn": BackendCapabilitySpec(
        "citylearn",
        "citylearn_native_building_energy",
        (
            "net_electricity_consumption",
            "storage_soc",
            "storage_energy_balance",
            "carbon_intensity",
        ),
        ("inspect_building_state",),
        ("set_storage_dispatch",),
        clock_semantics="simulator_owned",
        source_contract_builder=(
            "domains.building_energy.source_contracts:citylearn"
        ),
        source_evidence_adapter=(
            "domains.building_energy.source_evidence:citylearn"
        ),
        source_consumption_mode="direct_runtime_files",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=(
            "load_change",
            "generation_change",
            "tariff_change",
        ),
        runtime_fidelity="native_library",
        formal_core_allowed=True,
        periodic_scan_every_ticks=2,
        max_review_after_ticks=4,
        persistent_control_tools=("set_storage_dispatch",),
    ),
    "sumo_ego": BackendCapabilitySpec(
        "sumo_ego",
        "ngsim_source_derived_native_sumo_ego_closed_loop",
        (
            "ego_vehicle_state",
            "nearby_road_actors",
            "route_progress",
            "runtime_assurance_mode",
        ),
        (
            "inspect_ego_state",
            "inspect_local_scene",
            "inspect_odd_status",
            "inspect_safety_state",
        ),
        (
            "set_driving_envelope",
            "request_tactical_maneuver",
            "request_minimal_risk_maneuver",
            "request_recovery_check",
            "authorize_recovery",
        ),
        clock_semantics="simulator_owned_substeps",
        source_contract_builder=(
            "domains.autonomous_driving.source_contracts:ngsim"
        ),
        source_evidence_adapter=(
            "domains.autonomous_driving.source_evidence:ngsim"
        ),
        source_consumption_mode="direct_runtime_files",
        decision_cadence_mode="hybrid",
        source_scheduled_event_types=(
            "cut_in",
            "leader_change",
            "hard_braking",
        ),
        runtime_fidelity="native_live_sumo_reactive",
        formal_core_allowed=True,
        periodic_scan_every_ticks=1,
        max_review_after_ticks=2,
        persistent_control_tools=("set_driving_envelope",),
    ),
}


def known_domains() -> tuple[str, ...]:
    """Return the registered domain keys (stable order)."""
    return tuple(_REGISTRY.keys())


def get_domain_spec(domain: str | None) -> DomainSpec:
    """Resolve a :class:`DomainSpec` for ``scenario['domain']``.

    A missing/empty domain defaults to ``power_grid`` (the v0.1–v0.6
    scenarios predate the ``domain`` field). Unknown domains raise so a
    typo never silently scores against the wrong physics.
    """
    key = (domain or DEFAULT_DOMAIN).strip().lower()
    spec = _REGISTRY.get(key)
    if spec is None:
        raise KeyError(
            f"unknown scenario domain {domain!r}; "
            f"registered domains: {', '.join(known_domains())}"
        )
    return spec


def get_backend_capability(
    backend_kind: str | None,
) -> BackendCapabilitySpec:
    key = str(backend_kind or "").strip()
    spec = _BACKEND_CAPABILITIES.get(key)
    if spec is None:
        raise KeyError(f"backend capability contract is not registered: {key!r}")
    return spec


def known_backend_capabilities() -> tuple[str, ...]:
    return tuple(_BACKEND_CAPABILITIES)


def apply_supervisory_cadence(
    backend_kind: str,
    cadence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish the agent-owned review and typed-wakeup contract."""
    get_backend_capability(backend_kind)
    merged = dict(cadence or {})
    merged.pop("periodic_scan_every_ticks", None)
    merged.pop("max_review_after_ticks", None)
    merged.update(
        {
            "cadence_contract": "agent_scheduled_v1",
            "review_owner": "agent",
            "harness_periodic_supervisory_scan": False,
            "typed_actionable_events": True,
        }
    )
    return merged


def build_backend_records(env: Any) -> list[dict[str, Any]]:
    """Return the per-tick 14-key scorer rows for ``env``'s backend.

    Domain-neutral version of the power-grid helper: every shipped
    backend implements the public ``scoring_records()`` contract, so this
    simply forwards to it. The power-grid module keeps its own richer
    helper (with a defensive private-record fallback) for backward
    compatibility; this generic one is used for the v0.7 domains.
    """
    backend = getattr(env, "_backend", None)
    if backend is None:
        return []
    records_fn = getattr(backend, "scoring_records", None)
    if callable(records_fn):
        return list(records_fn())
    return []


def reference_optimum_from_backend_config(env: Any) -> float | None:
    """Read the cached oracle optimum that feeds ``ScoringInputs.lp_optimum``.

    The v0.7 domains cache ``seed.backend_config['reference_optimum'] =
    {"reference_optimum": <float>, ...}`` during ``env.reset``. Returns
    the positive scalar, or ``None`` when no oracle ran (e.g. the oracle
    dependency was unavailable, or the domain has no optimum — disaster).
    """
    seed_obj = getattr(env, "seed_obj", None)
    if seed_obj is None:
        return None
    backend_config = getattr(seed_obj, "backend_config", None) or {}
    envelope = backend_config.get("reference_optimum")
    if not isinstance(envelope, dict):
        return None
    value = envelope.get("reference_optimum")
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def reference_optimum_objective_component(
    env: Any,
    *,
    default: str | None = None,
) -> str | None:
    """Read the realized cost key minimized by a cached oracle envelope."""
    seed_obj = getattr(env, "seed_obj", None)
    backend_config = getattr(seed_obj, "backend_config", None) or {}
    envelope = backend_config.get("reference_optimum")
    if not isinstance(envelope, dict):
        return default
    component = str(envelope.get("objective_component") or "").strip()
    if component:
        return component
    # Existing JSPLIB/CO-Bench envelopes predate ``objective_component`` but
    # already carry an explicit makespan objective.  Preserve that contract
    # while newly cached envelopes write the component directly.
    if str(envelope.get("objective") or "") == "minimize_makespan":
        return "production_cost"
    return default
