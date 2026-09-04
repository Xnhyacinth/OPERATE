"""Build source-locked OR-Gym InvManagement release seeds."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from domains.logistics.backends.orgym_invmgmt import (
    ORGYM_ENV_ID,
    ORGYM_LICENSE,
    ORGYM_PACKAGE_VERSION,
    ORGYM_SOURCE_COMMIT,
    ORGYM_SOURCE_URL,
)

from .schema import CustomerPriority, LogisticsScenarioSeed, Provenance

_ORGYM_ENV_CONFIG: dict[str, Any] = {
    "periods": 6,
    "I0": [0],
    "p": 10,
    "r": [2, 1],
    "k": [8, 0],
    "h": [0.1],
    "c": [15],
    "L": [1],
    "dist": 5,
    "user_D": [0, 8, 0, 8, 0, 8],
    "alpha": 1.0,
    "seed_int": 123,
}

_DEMAND_PROFILE_ID = "alternating_8_unit_lost_sales_leadtime1_capacity15"
_SOURCE_DENOMINATOR_KEY = (
    f"orgym_invmgmt:{ORGYM_ENV_ID}:lost_sales:{_DEMAND_PROFILE_ID}"
)

_HONEST_ZERO_KEYS = ["n_voltage_violations", "n_disconnected_lines"]
_KEY_ALIASES = {
    "aggregate_demand_mw": "customer_demand_units",
    "aggregate_generation_mw": "served_customer_units",
    "balance_error_mw": "lost_sales_units",
    "reserves_required_mw": "same_tick_customer_demand_units",
    "reserves_procured_mw": "available_inventory_plus_pipeline_units",
    "production_cost": "procurement_plus_holding_cost",
    "startup_cost": "order_actuation_cost",
    "shed_penalty": "lost_sales_penalty",
    "rho_max": "demand_or_order_capacity_pressure",
    "n_overloads": "lost_sales_or_capacity_shortfall_indicator",
}

_DIMENSION_APPLICABILITY = {
    "system_survival": {
        "applicable": True,
        "reason": "lost_sales_and_capacity_pressure_tick_records_available",
    },
    "economic_cost": {
        "applicable": True,
        "reason": "positive_inventory_cost_components_and_wait_counterfactual_available",
    },
    "safety_violation": {
        "applicable": True,
        "reason": "lost_sales_capacity_shortfall_records_available",
    },
    "weighted_equity_score": {
        "applicable": True,
        "reason": "retailer_lost_sales_units_recorded_as_customer_unmet_demand",
    },
    "ethical_quality": {
        "applicable": False,
        "reason": "orgym_inventory_release_has_no_moral_dilemma_payload",
    },
    "stakeholder_management": {
        "applicable": False,
        "reason": "orgym_inventory_release_has_no_trust_manager_delta",
    },
    "adaptive_replanning": {
        "applicable": True,
        "reason": "state_changing_replenishment_orders_can_prevent_future_lost_sales_events",
    },
    "information_efficiency": {
        "applicable": False,
        "reason": "demand_stream_is_source_locked_configuration_not_paid_forecast_tool",
    },
    "foresight_score": {
        "applicable": False,
        "reason": "baseline_oracle_does_not_emit_commit_to_plan_predictions",
    },
    "optimality_gap": {
        "applicable": False,
        "reason": "no_milp_reference_optimum_for_first_orgym_release_cell",
    },
    "counterfactual_prevention": {
        "applicable": True,
        "reason": "deterministic_masked_action_replay_over_same_orgym_seed",
    },
}


def build_orgym_inventory_seed(
    *,
    seed_id: str = "inventory_replenishment/time_pressure/basic/orgym_invmgmt_native_lost_sales",
    seed: int = 123,
    difficulty_mode: str = "time_pressure",
    difficulty_level: str = "basic",
) -> LogisticsScenarioSeed:
    """Build the v0.9 single-cell OR-Gym native inventory seed."""
    env_config = dict(_ORGYM_ENV_CONFIG)
    env_config["seed_int"] = int(seed)
    config_hash = "sha256:" + _sha256_json(env_config)
    backend_config = {
        "inventory_environment_id": ORGYM_ENV_ID,
        "orgym_env_config": env_config,
        "orgym_env_config_hash": config_hash,
        "demand_profile_id": _DEMAND_PROFILE_ID,
        "demand_stream_hash": "sha256:" + _sha256_json(env_config["user_D"]),
        "source_denominator_key": _SOURCE_DENOMINATOR_KEY,
        "stages": 2,
        "lead_times": list(env_config["L"]),
        "capacities": list(env_config["c"]),
        "cost_profile_hash": "sha256:"
        + _sha256_json(
            {
                "p": env_config["p"],
                "r": env_config["r"],
                "k": env_config["k"],
                "h": env_config["h"],
            }
        ),
        "honest_zero_keys": list(_HONEST_ZERO_KEYS),
        "logistics_key_aliases": dict(_KEY_ALIASES),
        "dimension_applicability": json.loads(json.dumps(_DIMENSION_APPLICABILITY)),
        "release_ready": True,
        "release_reentry_ready": True,
        "source_integration_rung": "executed_with_live_backend",
    }
    return LogisticsScenarioSeed(
        seed_id=seed_id,
        family="inventory_replenishment",
        domain="logistics",
        backend_kind="orgym_invmgmt",
        backend_config=backend_config,
        horizon_ticks=int(env_config["periods"]),
        tick_minutes=60,
        seed=int(seed),
        load_assignments=[
            CustomerPriority(
                load_id="retailer",
                stakeholder_class="commercial",
                criticality=0.5,
                demand=float(sum(env_config["user_D"])),
            )
        ],
        perturbations=[],
        dilemmas=[],
        difficulty_mode=difficulty_mode,  # type: ignore[arg-type]
        difficulty_level=difficulty_level,  # type: ignore[arg-type]
        provenance=Provenance(
            data_source="orgym",
            files=[
                "works/OR-Gym/or_gym/envs/supply_chain/inventory_management.py",
                "works/OR-Gym/or_gym/envs/__init__.py",
                "works/OR-Gym/or_gym/version.py",
                "works/OR-Gym/LICENSE",
            ],
            commit=ORGYM_SOURCE_COMMIT,
            url=ORGYM_SOURCE_URL,
            lock_strategy="git_commit+package_version+license",
            time_window={
                "inventory_environment_id": ORGYM_ENV_ID,
                "package_version": ORGYM_PACKAGE_VERSION,
                "env_config_hash": config_hash,
                "demand_profile_id": _DEMAND_PROFILE_ID,
            },
            license=ORGYM_LICENSE,
            notes=(
                "Source-locked OR-Gym native InvManagement-v1 lost-sales benchmark "
                "configuration. Demand is simulator-defined/user_D in the locked "
                "environment config, not empirical retail history."
            ),
        ),
    )


def orgym_source_denominator_key() -> str:
    return _SOURCE_DENOMINATOR_KEY


def _sha256_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
