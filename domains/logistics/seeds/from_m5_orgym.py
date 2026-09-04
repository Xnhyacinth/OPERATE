"""Build empirical M5-demand OR-Gym inventory release seeds.

M5 supplies the real SKU-store demand stream. OR-Gym supplies the native
``InvManagement-v1`` lost-sales simulator mechanics (orders, lead times,
capacity, holding/procurement/lost-sales costs). Raw M5 files are not
redistributed in current release artifacts; release materializers must write the
redacted backend-config form produced by ``redact_m5_orgym_backend_config`` and
the runtime reconstructs the private demand stream from the local SHA-locked
``works/M5`` source contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from domains.logistics.backends.orgym_invmgmt import (
    ORGYM_ENV_ID,
    ORGYM_LICENSE,
    ORGYM_PACKAGE_VERSION,
    ORGYM_SOURCE_COMMIT,
    ORGYM_SOURCE_URL,
)

from .schema import CustomerPriority, LogisticsScenarioSeed, Provenance

REPO_ROOT = Path(__file__).resolve().parents[3]
M5_ROOT = REPO_ROOT / "works" / "M5"
M5_SOURCE_LOCK = M5_ROOT / "source_lock.json"

M5_SOURCE_ID = "m5_forecasting"
M5_SOURCE_URL = "https://www.kaggle.com/competitions/m5-forecasting-accuracy"
M5_LICENSE = "Kaggle competition rules"
M5_REQUIRED_FILES = (
    "works/M5/calendar.csv",
    "works/M5/sales_train_evaluation.csv",
    "works/M5/sell_prices.csv",
)
M5_RELEASE_LABEL = "m5-forecasting-accuracy 2020-06-01 files"


# Keep v0.9 tiny: two distinct empirical SKU-store demand streams, both with
# nonzero baseline gap and oracle-vs-wait headroom under OR-Gym's native
# lost-sales dynamics. Day numbers are 1-indexed M5 ``d_N`` columns.
@dataclass(frozen=True)
class M5OrgymWindow:
    sku_store_key: str
    item_id: str
    store_id: str
    start_day: int
    seed: int
    category_id: str = ""
    lead_time_days: int = 1
    capacity_scale: float = 1.0
    profile_id: str = ""
    window_length_days: int = 30
    difficulty_level: Literal["basic", "medium", "high", "extreme", "cascading"] = (
        "basic"
    )


RELEASE_WINDOWS: tuple[M5OrgymWindow, ...] = (
    M5OrgymWindow(
        sku_store_key="HOBBIES_1_001_CA_1_evaluation",
        item_id="HOBBIES_1_001",
        store_id="CA_1",
        start_day=1904,
        seed=4201,
    ),
    M5OrgymWindow(
        sku_store_key="HOBBIES_1_003_CA_1_evaluation",
        item_id="HOBBIES_1_003",
        store_id="CA_1",
        start_day=1758,
        seed=4203,
    ),
)
WINDOW_LENGTH_DAYS = 30
OBSERVATION_FORECAST_HORIZON = 7

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
        "reason": "inventory_procurement_holding_lost_sales_costs_and_wait_counterfactual_available",
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
        "reason": "orgym_inventory_m5_release_has_no_moral_dilemma_payload",
    },
    "stakeholder_management": {
        "applicable": False,
        "reason": "orgym_inventory_m5_release_has_no_trust_manager_delta",
    },
    "adaptive_replanning": {
        "applicable": True,
        "reason": "state_changing_replenishment_orders_can_prevent_future_lost_sales_events",
    },
    "information_efficiency": {
        "applicable": False,
        "reason": "no_paid_investigation_or_forecast_tool_for_first_m5_release_cell",
    },
    "foresight_score": {
        "applicable": False,
        "reason": "baseline_oracle_does_not_emit_commit_to_plan_predictions",
    },
    "optimality_gap": {
        "applicable": False,
        "reason": "no_milp_reference_optimum_for_first_orgym_m5_release_cell",
    },
    "counterfactual_prevention": {
        "applicable": True,
        "reason": "deterministic_masked_action_replay_over_same_m5_orgym_seed",
    },
}


def build_m5_orgym_inventory_seeds(
    *,
    source_root: Path = M5_ROOT,
    windows: tuple[M5OrgymWindow, ...] | None = None,
) -> list[LogisticsScenarioSeed]:
    """Build empirical M5+OR-Gym inventory seeds for the requested windows."""
    source_lock = verify_m5_orgym_source_lock(source_root=source_root)
    sales_rows = _read_sales_rows(source_root / "sales_train_evaluation.csv")
    prices = _read_first_prices(source_root / "sell_prices.csv")
    selected_windows = RELEASE_WINDOWS if windows is None else windows
    seeds: list[LogisticsScenarioSeed] = []
    for window in selected_windows:
        seeds.append(
            build_m5_orgym_inventory_seed(
                window=window,
                sales_rows=sales_rows,
                prices=prices,
                source_lock=source_lock,
            )
        )
    return seeds


def build_m5_orgym_inventory_seed(
    *,
    window: M5OrgymWindow,
    sales_rows: dict[str, dict[str, str]] | None = None,
    prices: dict[tuple[str, str], float] | None = None,
    source_lock: dict[str, Any] | None = None,
    source_root: Path = M5_ROOT,
) -> LogisticsScenarioSeed:
    """Build one empirical M5-demand OR-Gym release seed."""
    if source_lock is None:
        source_lock = verify_m5_orgym_source_lock(source_root=source_root)
    if sales_rows is None:
        sales_rows = _read_sales_rows(source_root / "sales_train_evaluation.csv")
    if prices is None:
        prices = _read_first_prices(source_root / "sell_prices.csv")
    row = sales_rows.get(window.sku_store_key)
    if row is None:
        raise KeyError(f"missing M5 SKU-store row: {window.sku_store_key}")
    window_length = int(window.window_length_days)
    if window_length < 2:
        raise ValueError("M5 OR-Gym window_length_days must be >= 2")
    demand = _window_demand(row, start_day=window.start_day, length=window_length)
    category_id = str(window.category_id or row.get("cat_id") or "unknown")
    department_id = str(row.get("dept_id") or "unknown")
    state_id = str(row.get("state_id") or "unknown")
    price = float(prices.get((window.store_id, window.item_id), 1.0))
    profile_id = _inventory_profile_id(window)
    profile_suffix = f"_{profile_id}" if profile_id else ""
    denominator_profile_suffix = f":{profile_id}" if profile_id else ""
    env_config = _orgym_env_config(
        demand=demand,
        price=price,
        seed=window.seed,
        lead_time_days=window.lead_time_days,
        capacity_scale=window.capacity_scale,
    )
    env_config_hash = "sha256:" + _sha256_json(env_config)
    demand_hash = "sha256:" + _sha256_json(demand)
    cost_profile = {
        "p": env_config["p"],
        "r": env_config["r"],
        "k": env_config["k"],
        "h": env_config["h"],
        "c": env_config["c"],
        "L": env_config["L"],
    }
    cost_profile_hash = "sha256:" + _sha256_json(cost_profile)
    day_end = window.start_day + window_length - 1
    demand_profile_id = (
        f"m5_{window.item_id}_{window.store_id}_evaluation_"
        f"d{window.start_day}_d{day_end}{profile_suffix}"
    )
    source_denominator_key = (
        "orgym_invmgmt:m5_forecasting:InvManagement-v1:"
        f"{window.sku_store_key}:d{window.start_day}_d{day_end}"
        f"{denominator_profile_suffix}"
    )
    seed_id = (
        f"inventory_replenishment/time_pressure/{window.difficulty_level}/"
        f"m5_{window.item_id.lower()}_{window.store_id.lower()}_"
        f"d{window.start_day}_{window_length}d{profile_suffix}"
    )
    m5_lock_summary = _m5_source_lock_summary(source_lock)
    backend_config = {
        "inventory_environment_id": ORGYM_ENV_ID,
        "orgym_env_config": env_config,
        "orgym_env_config_hash": env_config_hash,
        "demand_profile_id": demand_profile_id,
        "demand_stream_hash": demand_hash,
        "source_denominator_key": source_denominator_key,
        "stages": 2,
        "lead_times": list(env_config["L"]),
        "capacities": list(env_config["c"]),
        "cost_profile_hash": cost_profile_hash,
        "hide_full_demand_stream": True,
        "observation_forecast_horizon": OBSERVATION_FORECAST_HORIZON,
        "honest_zero_keys": list(_HONEST_ZERO_KEYS),
        "logistics_key_aliases": dict(_KEY_ALIASES),
        "dimension_applicability": json.loads(json.dumps(_DIMENSION_APPLICABILITY)),
        "release_ready": True,
        "release_reentry_ready": True,
        "source_integration_rung": "executed_with_live_backend",
        "m5_source_lock": m5_lock_summary,
        "m5_sku_store_key": window.sku_store_key,
        "m5_item_id": window.item_id,
        "m5_store_id": window.store_id,
        "m5_category_id": category_id,
        "m5_department_id": department_id,
        "m5_state_id": state_id,
        "m5_start_day": f"d_{window.start_day}",
        "m5_end_day": f"d_{day_end}",
        "m5_window_length_days": window_length,
        "m5_demand_sum_units": int(sum(demand)),
        "m5_nonzero_demand_days": int(sum(1 for value in demand if value > 0)),
    }
    if profile_id:
        backend_config.update(
            {
                "inventory_profile_id": profile_id,
                "m5_lead_time_days": int(window.lead_time_days),
                "m5_capacity_scale": float(window.capacity_scale),
            }
        )
    return LogisticsScenarioSeed(
        seed_id=seed_id,
        family="inventory_replenishment",
        domain="logistics",
        backend_kind="orgym_invmgmt",
        backend_config=backend_config,
        horizon_ticks=window_length,
        tick_minutes=1440,
        seed=int(window.seed),
        load_assignments=[
            CustomerPriority(
                load_id="retailer",
                stakeholder_class="commercial",
                criticality=0.5,
                demand=float(sum(demand)),
            )
        ],
        perturbations=[],
        dilemmas=[],
        difficulty_mode="time_pressure",
        difficulty_level=window.difficulty_level,
        provenance=Provenance(
            data_source="m5_forecasting+orgym",
            files=[
                "works/M5/source_lock.json",
                *M5_REQUIRED_FILES,
                "works/OR-Gym/or_gym/envs/supply_chain/inventory_management.py",
                "works/OR-Gym/or_gym/version.py",
                "works/OR-Gym/LICENSE",
            ],
            commit=(
                f"{source_lock.get('git_commit_or_release_tag')};"
                f"orgym:{ORGYM_SOURCE_COMMIT}"
            ),
            url=M5_SOURCE_URL,
            lock_strategy=(
                "kaggle_competition_terms+file_sha256+orgym_git_commit+env_config_hash"
            ),
            time_window={
                "sku_store_key": window.sku_store_key,
                "item_id": window.item_id,
                "store_id": window.store_id,
                "category_id": category_id,
                "department_id": department_id,
                "state_id": state_id,
                "start_day": f"d_{window.start_day}",
                "end_day": f"d_{day_end}",
                "window_length_days": window_length,
                "demand_sum_units": int(sum(demand)),
                "nonzero_demand_days": int(sum(1 for value in demand if value > 0)),
                "demand_stream_hash": demand_hash,
                "env_config_hash": env_config_hash,
                "cost_profile_hash": cost_profile_hash,
                "m5_source_lock_sha256": _sha256_path(M5_SOURCE_LOCK),
                "m5_license_or_terms_sha256": source_lock.get(
                    "license_or_terms_sha256"
                ),
                "orgym_commit": ORGYM_SOURCE_COMMIT,
                "orgym_package_version": ORGYM_PACKAGE_VERSION,
                **(
                    {
                        "inventory_profile_id": profile_id,
                        "lead_time_days": int(window.lead_time_days),
                        "capacity_scale": float(window.capacity_scale),
                    }
                    if profile_id
                    else {}
                ),
            },
            license=f"{M5_LICENSE} + OR-Gym {ORGYM_LICENSE}",
            notes=(
                "Raw M5 files are not redistributed; users fetch through Kaggle "
                "after accepting competition rules. M5 supplies empirical "
                "SKU-store demand, while OR-Gym supplies native inventory "
                "transition dynamics and economic knobs. Procurement/cost "
                "parameters are simulator economic parameters, not empirical "
                "Walmart procurement costs."
            ),
        ),
    )


def verify_m5_orgym_source_lock(*, source_root: Path = M5_ROOT) -> dict[str, Any]:
    """Verify local M5 sidecar/file hashes plus repo-local OR-Gym lock."""
    sidecar_path = source_root / "source_lock.json"
    if not sidecar_path.exists():
        raise FileNotFoundError(f"missing M5 source lock: {sidecar_path}")
    source_lock = json.loads(sidecar_path.read_text(encoding="utf-8"))
    tracked = [sidecar_path]
    tracked.extend(REPO_ROOT / rel for rel in (source_lock.get("files") or {}))
    signature = tuple(
        (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
        for path in tracked
    )
    return deepcopy(_verify_m5_orgym_source_lock_cached(str(source_root.resolve()), signature))


@lru_cache(maxsize=8)
def _verify_m5_orgym_source_lock_cached(
    source_root_text: str,
    _signature: tuple[tuple[str, int, int], ...],
) -> dict[str, Any]:
    source_root = Path(source_root_text)
    sidecar_path = source_root / "source_lock.json"
    source_lock = json.loads(sidecar_path.read_text(encoding="utf-8"))
    required_pairs = {
        "source_id": M5_SOURCE_ID,
        "source_url": M5_SOURCE_URL,
        "license": M5_LICENSE,
        "inventory_environment_id": ORGYM_ENV_ID,
        "package_version": "or-gym==0.5.0",
    }
    for key, expected in required_pairs.items():
        actual = source_lock.get(key)
        if actual != expected:
            raise ValueError(f"M5 source_lock.{key}={actual!r}, expected {expected!r}")
    if source_lock.get("license_verified") is not True:
        raise ValueError("M5 source_lock.license_verified must be true")
    if source_lock.get("terms_accepted") is not True:
        raise ValueError("M5 source_lock.terms_accepted must be true")
    terms_hash = str(source_lock.get("license_or_terms_sha256") or "")
    if not _valid_sha256_field(terms_hash):
        raise ValueError(
            "M5 source_lock.license_or_terms_sha256 must be sha256:<64hex>"
        )
    files = source_lock.get("files") or {}
    for rel in M5_REQUIRED_FILES:
        expected = files.get(rel)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"M5 source_lock.files missing SHA-256 for {rel}")
        actual = _sha256_path(REPO_ROOT / rel)
        if actual != expected:
            raise ValueError(f"M5 file hash mismatch for {rel}: {actual} != {expected}")
    runtime = source_lock.get("orgym_runtime_source") or {}
    if runtime.get("commit") != ORGYM_SOURCE_COMMIT:
        raise ValueError("M5 source_lock OR-Gym commit does not match release constant")
    if runtime.get("license") != ORGYM_LICENSE:
        raise ValueError(
            "M5 source_lock OR-Gym license does not match release constant"
        )
    return source_lock


def m5_dataset_manifest_entry(
    source_lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if source_lock is None:
        source_lock = verify_m5_orgym_source_lock()
    return {
        "license": M5_LICENSE,
        "url": M5_SOURCE_URL,
        "data_release": M5_RELEASE_LABEL,
        "lock_strategy": "kaggle_competition_terms+file_sha256",
        "license_or_terms_sha256": source_lock.get("license_or_terms_sha256"),
        "git_commit_or_release_tag": source_lock.get("git_commit_or_release_tag"),
        "files": dict(source_lock.get("files") or {}),
        "redistribution_limit": source_lock.get("redistribution_limit"),
    }


def redact_m5_orgym_backend_config(backend_config: dict[str, Any]) -> dict[str, Any]:
    """Return a release-artifact-safe M5 OR-Gym backend config.

    The transient runtime config contains OR-Gym's ``user_D`` demand vector. M5
    raw demand is covered by Kaggle access terms, so redistributable scenario
    artifacts carry only the source handle and hashes. ``resolve_m5_orgym_env_config``
    rehydrates the full config on hosts that have accepted and downloaded M5.
    """
    cfg = json.loads(json.dumps(backend_config))
    env_config = cfg.get("orgym_env_config")
    if not isinstance(env_config, dict) or "user_D" not in env_config:
        return cfg

    public_env_config = {
        key: value for key, value in env_config.items() if key != "user_D"
    }
    cfg["orgym_env_config"] = public_env_config
    cfg["orgym_env_config_redacted"] = True
    cfg["orgym_env_config_redaction_reason"] = (
        "m5_kaggle_raw_demand_stream_not_redistributed"
    )
    cfg["m5_demand_handle"] = {
        "source_id": M5_SOURCE_ID,
        "source_url": M5_SOURCE_URL,
        "source_lock_file": "works/M5/source_lock.json",
        "calendar_file": "works/M5/calendar.csv",
        "sales_file": "works/M5/sales_train_evaluation.csv",
        "sell_prices_file": "works/M5/sell_prices.csv",
        "sku_store_key": cfg.get("m5_sku_store_key"),
        "item_id": cfg.get("m5_item_id"),
        "store_id": cfg.get("m5_store_id"),
        "start_day": cfg.get("m5_start_day"),
        "end_day": cfg.get("m5_end_day"),
        "window_length_days": cfg.get("m5_window_length_days"),
        "demand_stream_hash": cfg.get("demand_stream_hash"),
        "env_config_hash": cfg.get("orgym_env_config_hash"),
    }
    return cfg


def resolve_m5_orgym_env_config(
    backend_config: dict[str, Any],
    *,
    source_root: Path = M5_ROOT,
) -> dict[str, Any]:
    """Rehydrate a full OR-Gym config from a redacted M5 release config."""
    env_config = dict(backend_config.get("orgym_env_config") or {})
    user_d = env_config.get("user_D")
    if isinstance(user_d, list) and user_d:
        return env_config
    if str(backend_config.get("inventory_environment_id", "")) != ORGYM_ENV_ID:
        raise ValueError("M5 OR-Gym config requires InvManagement-v1")

    verify_m5_orgym_source_lock(source_root=source_root)
    sku_store_key = str(backend_config.get("m5_sku_store_key") or "")
    item_id = str(backend_config.get("m5_item_id") or "")
    store_id = str(backend_config.get("m5_store_id") or "")
    if not sku_store_key or not item_id or not store_id:
        handle = backend_config.get("m5_demand_handle") or {}
        sku_store_key = str(handle.get("sku_store_key") or sku_store_key)
        item_id = str(handle.get("item_id") or item_id)
        store_id = str(handle.get("store_id") or store_id)
    start_day = _parse_m5_day(backend_config.get("m5_start_day"))
    if start_day <= 0:
        handle = backend_config.get("m5_demand_handle") or {}
        start_day = _parse_m5_day(handle.get("start_day"))
    length = int(backend_config.get("m5_window_length_days") or WINDOW_LENGTH_DAYS)
    if not sku_store_key or not item_id or not store_id or start_day <= 0:
        raise ValueError("redacted M5 OR-Gym config missing demand handle fields")

    row = _read_sales_row(
        str((source_root / "sales_train_evaluation.csv").resolve()), sku_store_key
    )
    demand = _window_demand(row, start_day=start_day, length=length)
    demand_hash = "sha256:" + _sha256_json(demand)
    expected_demand_hash = str(backend_config.get("demand_stream_hash") or "")
    if expected_demand_hash and demand_hash != expected_demand_hash:
        raise ValueError(
            "M5 demand hash mismatch for redacted OR-Gym config: "
            f"{demand_hash} != {expected_demand_hash}"
        )

    price = _read_first_price(
        str((source_root / "sell_prices.csv").resolve()), store_id, item_id
    )
    lead_time_days = int(
        backend_config.get("m5_lead_time_days")
        or (backend_config.get("lead_times") or [1])[0]
    )
    capacity_scale = float(backend_config.get("m5_capacity_scale") or 1.0)
    seed_int = int(env_config.get("seed_int", backend_config.get("seed_int", 0)) or 0)
    hydrated = _orgym_env_config(
        demand=demand,
        price=price,
        seed=seed_int,
        lead_time_days=lead_time_days,
        capacity_scale=capacity_scale,
    )
    expected_env_hash = str(backend_config.get("orgym_env_config_hash") or "")
    env_hash = "sha256:" + _sha256_json(hydrated)
    if expected_env_hash and env_hash != expected_env_hash:
        raise ValueError(
            "M5 OR-Gym env config hash mismatch after redaction rehydrate: "
            f"{env_hash} != {expected_env_hash}"
        )
    return hydrated


def orgym_dataset_manifest_entry() -> dict[str, Any]:
    return {
        "license": ORGYM_LICENSE,
        "url": ORGYM_SOURCE_URL,
        "commit": ORGYM_SOURCE_COMMIT,
        "package_version": ORGYM_PACKAGE_VERSION,
        "lock_strategy": "git_commit+package_version+license",
    }


def _inventory_profile_id(window: M5OrgymWindow) -> str:
    if window.profile_id:
        return window.profile_id
    if int(window.lead_time_days) == 1 and float(window.capacity_scale) == 1.0:
        return ""
    capacity_pct = int(round(float(window.capacity_scale) * 100))
    return f"lt{int(window.lead_time_days)}_cap{capacity_pct}"


def _orgym_env_config(
    *,
    demand: list[int],
    price: float,
    seed: int,
    lead_time_days: int = 1,
    capacity_scale: float = 1.0,
) -> dict[str, Any]:
    # A deliberately small two-stage lost-sales instance: the agent controls
    # retailer replenishment from an infinite upstream supplier with one-day
    # lead time. Prices/costs scale with the M5 sell price so different SKU
    # streams have distinct economic profiles without claiming empirical M5
    # procurement costs.
    capacity = max(1, int(sum(demand) * float(capacity_scale)))
    unit_price = round(price * 2.0, 4)
    procurement_cost = round(price, 4)
    lost_sales_penalty = round(price * 3.0, 4)
    return {
        "periods": len(demand),
        "I0": [0],
        "p": unit_price,
        "r": [procurement_cost, round(procurement_cost * 0.5, 4)],
        "k": [lost_sales_penalty, 0.0],
        "h": [round(max(price * 0.1, 0.01), 4)],
        "c": [capacity],
        "L": [max(1, int(lead_time_days))],
        "backlog": False,
        "dist": 5,
        "user_D": [int(value) for value in demand],
        "alpha": 1.0,
        "seed_int": int(seed),
    }


def _m5_source_lock_summary(source_lock: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": source_lock.get("source_id"),
        "source_url": source_lock.get("source_url"),
        "license": source_lock.get("license"),
        "license_or_terms_sha256": source_lock.get("license_or_terms_sha256"),
        "lock_strategy": source_lock.get("lock_strategy"),
        "git_commit_or_release_tag": source_lock.get("git_commit_or_release_tag"),
        "file_sha256s": dict(source_lock.get("files") or {}),
        "redistribution_limit": source_lock.get("redistribution_limit"),
        "orgym_runtime_source": dict(source_lock.get("orgym_runtime_source") or {}),
    }


def _read_sales_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {str(row["id"]): dict(row) for row in csv.DictReader(f) if row.get("id")}


def _read_first_prices(path: Path) -> dict[tuple[str, str], float]:
    prices: dict[tuple[str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (str(row.get("store_id") or ""), str(row.get("item_id") or ""))
            if not all(key) or key in prices:
                continue
            try:
                prices[key] = float(row.get("sell_price") or 1.0)
            except ValueError:
                prices[key] = 1.0
    return prices


@lru_cache(maxsize=256)
def _read_sales_row(path: str, sku_store_key: str) -> dict[str, str]:
    """Read one M5 row for runtime rehydration without materializing the corpus."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("id") or "") == sku_store_key:
                return dict(row)
    raise KeyError(f"missing M5 SKU-store row: {sku_store_key}")


@lru_cache(maxsize=256)
def _read_first_price(path: str, store_id: str, item_id: str) -> float:
    """Read the first matching M5 price for runtime rehydration."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("store_id") != store_id or row.get("item_id") != item_id:
                continue
            try:
                return float(row.get("sell_price") or 1.0)
            except ValueError:
                return 1.0
    return 1.0


def _window_demand(row: dict[str, str], *, start_day: int, length: int) -> list[int]:
    values: list[int] = []
    for day in range(start_day, start_day + length):
        raw = row.get(f"d_{day}")
        if raw is None:
            raise KeyError(f"missing M5 demand column d_{day}")
        values.append(max(0, int(float(raw))))
    return values


def _parse_m5_day(value: Any) -> int:
    text = str(value or "").strip()
    if text.startswith("d_"):
        text = text[2:]
    if text.startswith("d"):
        text = text[1:]
    try:
        return int(text)
    except ValueError:
        return 0


def _sha256_json(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_sha256_field(value: str) -> bool:
    if not value.startswith("sha256:"):
        return False
    digest = value.split(":", 1)[1]
    return len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
