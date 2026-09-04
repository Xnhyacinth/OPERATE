"""Power-grid boundary for the source-grounded candidate pipeline."""

from __future__ import annotations

from collections.abc import Iterable

POWER_GRID_CONTROLS = frozenset(
    {
        "commit_reserve",
        "dispatch_generation_portfolio",
        "redispatch_generation",
        "request_mutual_aid",
        "set_battery_dispatch",
        "set_der_reactive_power",
        "set_transformer_tap",
        "shed_load",
        "switch_branch",
        "switch_capacitor",
        "topology_action",
    }
)


def classify_power_grid_controls(controls: Iterable[str]) -> str:
    names = {str(name) for name in controls}
    return (
        "power_grid"
        if names and names.issubset(POWER_GRID_CONTROLS)
        else "mixed_domain_held"
    )


def power_grid_capability_contract(
    backend_kind: str,
    *,
    control_tools: Iterable[str],
) -> dict[str, object]:
    """Declare only the shared, audited power-grid capability surface."""
    controls = sorted({str(name) for name in control_tools})
    if classify_power_grid_controls(controls) != "power_grid":
        controls = []
    return {
        "backend_kind": str(backend_kind),
        "native_state_fields": [
            "balance_error_mw",
            "bus_voltage_pu",
            "line_loading_percent",
        ],
        "observation_tools": [
            "query_grid_state",
            "query_chronics_window",
            "forecast_query",
        ],
        "control_tools": controls,
        "clock_semantics": "simulator_owned",
        "deterministic_seed": True,
        "counterfactual_reset": True,
        "adaptive_recovery_signal": "power_balance_violation",
    }
