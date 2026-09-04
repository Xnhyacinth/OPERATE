"""Microgrid/Building-Energy boundary for external candidate conversion."""

from __future__ import annotations

from typing import Any

MICROGRID_ELECTRICAL_CONTROLS = frozenset(
    {
        "battery",
        "battery_storage",
        "connect_pcc",
        "curtail_der",
        "demand_response",
        "der_dispatch",
        "dispatch_genset",
        "electrical_storage",
        "ev_storage",
        "grid_export",
        "grid_import",
        "load_shedding",
        "outage_response",
        "renewable_curtailment",
        "set_battery_dispatch",
        "set_der_reactive_power",
        "set_grid_exchange",
        "shed_load",
    }
)
BUILDING_ENERGY_CONTROLS = frozenset(
    {
        "cooling_device",
        "dhw_storage",
        "heating_device",
        "hvac",
        "occupant_comfort",
        "temperature_setpoint",
    }
)


def classify_citylearn_task(config: dict[str, Any]) -> str:
    controls = {str(value) for value in config.get("controls") or []}
    has_microgrid = bool(controls & MICROGRID_ELECTRICAL_CONTROLS)
    has_building = bool(controls & BUILDING_ENERGY_CONTROLS)
    unknown = controls - MICROGRID_ELECTRICAL_CONTROLS - BUILDING_ENERGY_CONTROLS
    if unknown or (has_microgrid and has_building) or not controls:
        return "mixed_domain_held"
    if has_microgrid:
        return "microgrid"
    return "building_energy"


def microgrid_capability_contract(
    backend_kind: str,
    *,
    control_tools: list[str],
) -> dict[str, object]:
    controls = sorted({str(name) for name in control_tools})
    if (
        classify_citylearn_task({"controls": controls}) != "microgrid"
        and str(backend_kind) == "citylearn"
    ) or not set(controls).issubset(MICROGRID_ELECTRICAL_CONTROLS):
        controls = []
    return {
        "backend_kind": str(backend_kind),
        "native_state_fields": [
            "load_mw",
            "renewable_generation_mw",
            "storage_soc",
            "grid_connection_state",
        ],
        "observation_tools": [
            "query_microgrid_state",
            "forecast_query",
        ],
        "control_tools": controls,
        "clock_semantics": "simulator_owned",
        "deterministic_seed": True,
        "counterfactual_reset": True,
        "adaptive_recovery_signal": "unserved_energy_burden",
    }
