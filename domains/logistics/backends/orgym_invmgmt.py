"""OR-Gym inventory-management backend and M5 source-lock probe.

``OrgymInvmgmtBackend`` is the v0.9 release path for the source-locked
repo-local OR-Gym ``InvManagement-v1`` environment under ``works/OR-Gym``.  It
uses OR-Gym's native multi-period lost-sales inventory dynamics and locks the
runtime source by upstream URL, commit SHA, package version, and MIT license.
The v0.9 release rows overlay source-locked M5 SKU-store demand streams from
``works/M5``; OR-Gym-native simulator-defined demand remains available only as
a test/development carrier unless a release materializer explicitly cites it.

``M5InventoryProbeBackend`` below is preserved as the older non-release M5
source-lock ladder probe.  It parses real M5 CSV rows and proves native
inventory tools/evidence through ``core.ToolRegistry`` before any future M5
materializer is allowed to create scenarios.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from contextlib import suppress
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from core import EvidenceLogger, ToolContext, ToolRegistry, ToolSpec


class InventoryDatasetLockPending(RuntimeError):
    """Raised when the inventory family is invoked without a real demand lock."""


# Machine-readable descriptor block consumed by the (later T6) manifest.
ORGYM_SOURCE_URL = "https://github.com/hubbs5/or-gym"
ORGYM_SOURCE_COMMIT = "0b18d16e569e2db70e83f09e867b53bdb4b87298"
ORGYM_PACKAGE_VERSION = "0.5.0"
ORGYM_LICENSE = "MIT"
ORGYM_ENV_ID = "InvManagement-v1"
M5_RECIPE_VERSION = "m5-sku-store-window-v1"
M5_RUNTIME_FILES = (
    "works/M5/calendar.csv",
    "works/M5/sales_train_evaluation.csv",
    "works/M5/sell_prices.csv",
)
_INVENTORY_EVENT_REGISTRY = MappingProxyType(
    {"inventory_demand_realized": "task"}
)

# Machine-readable descriptor block consumed by the v0.9 manifest.
DESCRIPTOR: dict[str, Any] = {
    "backend_kind": "orgym_invmgmt",
    "category": "operations_research_inventory_management",
    "domain": "logistics",
    "source_integration_rung": "executed_with_live_backend",
    "solves_power_flow": "no",
    "solves_vehicle_routing": "no",
    "solves_inventory_management": "yes (OR-Gym InvManagement-v1 lost-sales env)",
    "requires_external_solver": "none",
    "released_scenarios": 2,
    "status": "release_capable_v0_9",
    "publishable": True,
    "source_lock": {
        "url": ORGYM_SOURCE_URL,
        "commit": ORGYM_SOURCE_COMMIT,
        "package_version": ORGYM_PACKAGE_VERSION,
        "license": ORGYM_LICENSE,
        "lock_strategy": "git_commit+package_version+license",
    },
    "description": (
        "Source-locked OR-Gym InvManagement-v1 multi-period lost-sales "
        "inventory environment. v0.9 overlays M5 SKU-store demand streams on "
        "OR-Gym's native replenishment/lead-time/capacity mechanics. It models "
        "supply-chain replenishment orders, lead times, production capacity, "
        "holding costs, procurement costs, sales revenue, and lost-sales "
        "penalties. It does not solve power flow or vehicle routing."
    ),
    "honest_caveats": (
        "The released M5 rows use empirical Walmart SKU-store demand streams "
        "locked by Kaggle terms and file SHA-256s; OR-Gym supplies the native "
        "inventory-transition simulator, not empirical procurement costs. "
        "Electrical voltage/disconnection keys are honest-zero; capacity "
        "pressure and lost sales are mapped onto the canonical safety/cost "
        "record shape."
    ),
}


@dataclass
class InventoryTickRecord:
    tick: int
    aggregate_demand: float = 0.0
    served_demand: float = 0.0
    unmet_demand: float = 0.0
    required_standby: float = 0.0
    procured_standby: float = 0.0
    routing_cost: float = 0.0
    dispatch_fixed_cost: float = 0.0
    drop_penalty: float = 0.0
    max_utilization: float = 0.0
    n_capacity_violations: int = 0
    n_time_window_violations: int = 0
    n_failed_routes: int = 0
    done: bool = False
    realized_events: list[dict[str, Any]] = field(default_factory=list)


def resolve_orgym_m5_source_window(
    *,
    backend_config: dict[str, Any],
    provenance: dict[str, Any],
    repo_root: Path,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve one explicit, SHA-locked M5 SKU-store demand window."""
    handle = dict(backend_config.get("m5_demand_handle") or {})
    time_window = dict(provenance.get("time_window") or {})
    explicit = {
        "sku_store_key": (
            handle.get("sku_store_key")
            or backend_config.get("m5_sku_store_key")
            or time_window.get("sku_store_key")
        ),
        "item_id": (
            handle.get("item_id")
            or backend_config.get("m5_item_id")
            or time_window.get("item_id")
        ),
        "store_id": (
            handle.get("store_id")
            or backend_config.get("m5_store_id")
            or time_window.get("store_id")
        ),
        "start_day": (
            handle.get("start_day")
            or backend_config.get("m5_start_day")
            or time_window.get("start_day")
        ),
        "end_day": (
            handle.get("end_day")
            or backend_config.get("m5_end_day")
            or time_window.get("end_day")
        ),
    }
    if not all(explicit.values()):
        raise ValueError("source_window_metadata_missing")
    for key, values in {
        "sku_store_key": (
            handle.get("sku_store_key"),
            backend_config.get("m5_sku_store_key"),
            time_window.get("sku_store_key"),
        ),
        "item_id": (
            handle.get("item_id"),
            backend_config.get("m5_item_id"),
            time_window.get("item_id"),
        ),
        "store_id": (
            handle.get("store_id"),
            backend_config.get("m5_store_id"),
            time_window.get("store_id"),
        ),
        "start_day": (
            handle.get("start_day"),
            backend_config.get("m5_start_day"),
            time_window.get("start_day"),
        ),
        "end_day": (
            handle.get("end_day"),
            backend_config.get("m5_end_day"),
            time_window.get("end_day"),
        ),
    }.items():
        present = {str(value) for value in values if value not in (None, "")}
        if len(present) > 1:
            raise ValueError(f"source_window_metadata_mismatch:{key}")

    m5_root = source_root or repo_root / "works" / "M5"
    source_lock_path = m5_root / "source_lock.json"
    if not source_lock_path.is_file():
        raise ValueError("source_window_metadata_missing:source_lock")
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if (
        source_lock.get("source_id") != "m5_forecasting"
        or source_lock.get("terms_accepted") is not True
        or source_lock.get("license_verified") is not True
        or source_lock.get("inventory_environment_id") != ORGYM_ENV_ID
        or source_lock.get("package_version") != f"or-gym=={ORGYM_PACKAGE_VERSION}"
    ):
        raise ValueError("source_lock_identity_mismatch")
    runtime_lock = dict(source_lock.get("orgym_runtime_source") or {})
    if (
        runtime_lock.get("commit") != ORGYM_SOURCE_COMMIT
        or runtime_lock.get("license") != ORGYM_LICENSE
    ):
        raise ValueError("source_lock_identity_mismatch:orgym_runtime")

    opened_sha256 = {"works/M5/source_lock.json": _sha256_path(source_lock_path)}
    locked_hashes: dict[str, str] = {}
    for relative in M5_RUNTIME_FILES:
        path = m5_root / Path(relative).name
        expected = str((source_lock.get("files") or {}).get(relative) or "")
        actual = _sha256_path(path) if path.is_file() else ""
        if not expected or actual != expected:
            raise ValueError(f"source_hash_mismatch:{relative}:{actual}!={expected}")
        opened_sha256[relative] = actual
        locked_hashes[relative] = actual

    sku_store_key = str(explicit["sku_store_key"])
    item_id = str(explicit["item_id"])
    store_id = str(explicit["store_id"])
    start_day_number = _parse_m5_day(explicit["start_day"])
    end_day_number = _parse_m5_day(explicit["end_day"])
    if start_day_number <= 0 or end_day_number < start_day_number:
        raise ValueError("source_window_metadata_mismatch:day_range")
    window_length = end_day_number - start_day_number + 1
    declared_length = int(
        handle.get("window_length_days")
        or backend_config.get("m5_window_length_days")
        or time_window.get("window_length_days")
        or window_length
    )
    if declared_length != window_length:
        raise ValueError("source_window_metadata_mismatch:window_length")

    sales_path = m5_root / "sales_train_evaluation.csv"
    calendar_path = m5_root / "calendar.csv"
    prices_path = m5_root / "sell_prices.csv"
    sales_row = _read_m5_sales_row(
        sales_path,
        sku_store_key,
        source_sha256=locked_hashes["works/M5/sales_train_evaluation.csv"],
    )
    if (
        str(sales_row.get("item_id") or "") != item_id
        or str(sales_row.get("store_id") or "") != store_id
    ):
        raise ValueError("source_window_metadata_mismatch:item_or_store")
    demand = [
        int(float(sales_row[f"d_{day}"] or 0))
        for day in range(start_day_number, end_day_number + 1)
    ]
    calendar_by_day = _read_m5_calendar(
        calendar_path,
        source_sha256=locked_hashes["works/M5/calendar.csv"],
    )
    try:
        calendar_alignment = [
            {
                "day": f"d_{day}",
                "date": calendar_by_day[f"d_{day}"]["date"],
                "wm_yr_wk": calendar_by_day[f"d_{day}"]["wm_yr_wk"],
            }
            for day in range(start_day_number, end_day_number + 1)
        ]
    except KeyError as exc:
        raise ValueError(f"source_window_metadata_mismatch:calendar:{exc}") from exc
    sell_price = _read_m5_first_price(
        prices_path,
        store_id,
        item_id,
        source_sha256=locked_hashes["works/M5/sell_prices.csv"],
    )
    digest = _semantic_digest(demand)
    expected_demand_digest = str(
        handle.get("demand_stream_hash")
        or backend_config.get("demand_stream_hash")
        or time_window.get("demand_stream_hash")
        or ""
    ).removeprefix("sha256:")
    if expected_demand_digest and digest != expected_demand_digest:
        raise ValueError(
            f"source_window_digest_mismatch:{digest}!={expected_demand_digest}"
        )
    split = (
        "evaluation"
        if sku_store_key.endswith("_evaluation")
        else "validation"
        if sku_store_key.endswith("_validation")
        else "unknown"
    )
    trace_window = {
        "recipe_version": M5_RECIPE_VERSION,
        "item_or_sku": item_id,
        "sku_store_key": sku_store_key,
        "store": store_id,
        "start_day": f"d_{start_day_number}",
        "end_day": f"d_{end_day_number}",
        "split_or_version": split,
        "source_window_sha256": digest,
        "runtime_window_digest": digest,
    }
    return {
        **trace_window,
        "normalized_demand": demand,
        "calendar_alignment": calendar_alignment,
        "sell_price": sell_price,
        "source_lock_identity": {
            key: source_lock.get(key)
            for key in (
                "source_id",
                "source_url",
                "license",
                "git_commit_or_release_tag",
                "license_or_terms_sha256",
            )
        },
        "runtime_opened_assets": [
            {
                "path": path,
                "sha256": digest_value,
                "role": (
                    "source_lock_metadata"
                    if path.endswith("source_lock.json")
                    else "runtime_derivation_input"
                ),
            }
            for path, digest_value in sorted(opened_sha256.items())
        ],
        "opened_source_sha256": opened_sha256,
        "locked_derivation_source_hashes": locked_hashes,
        "consumed_channels": [
            "calendar_day",
            "demand_units",
            "sell_price",
        ],
        "trace_source_window": trace_window,
    }


def _build_orgym_env_config_from_source(
    *,
    public_config: dict[str, Any],
    backend_config: dict[str, Any],
    demand: list[int],
    sell_price: float,
    seed: int,
) -> dict[str, Any]:
    lead_times = list(public_config.get("L") or backend_config.get("lead_times") or [1])
    lead_time = max(1, int(lead_times[0]))
    capacity_scale = float(backend_config.get("m5_capacity_scale") or 1.0)
    price = float(sell_price)
    return {
        "periods": len(demand),
        "I0": [0],
        "p": round(price * 2.0, 4),
        "r": [round(price, 4), round(price * 0.5, 4)],
        "k": [round(price * 3.0, 4), 0.0],
        "h": [round(max(price * 0.1, 0.01), 4)],
        "c": [max(1, int(sum(demand) * capacity_scale))],
        "L": [lead_time],
        "backlog": False,
        "dist": 5,
        "user_D": [int(value) for value in demand],
        "alpha": 1.0,
        "seed_int": int(seed),
    }


def _parse_m5_day(value: Any) -> int:
    text = str(value or "").strip().lower()
    if text.startswith("d_"):
        text = text[2:]
    elif text.startswith("d"):
        text = text[1:]
    try:
        return int(text)
    except ValueError:
        return 0


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.resolve()), stat.st_size, stat.st_mtime_ns


def _read_m5_sales_row(
    path: Path,
    sku_store_key: str,
    *,
    source_sha256: str | None = None,
) -> dict[str, str]:
    digest = source_sha256 or _sha256_path(path)
    header, offsets = _m5_sales_index(*_path_identity(path), digest)
    offset = offsets.get(sku_store_key)
    if offset is None:
        raise ValueError(f"source_window_metadata_mismatch:sku:{sku_store_key}")
    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.readline()
    values = next(csv.reader([raw.decode("utf-8")]))
    return dict(zip(header, values, strict=False))


def _path_identity(path: Path) -> tuple[str, int, int]:
    """Return the stable path portion used by verified source indexes."""
    path_text, size, mtime_ns = _path_signature(path)
    return path_text, size, mtime_ns


@lru_cache(maxsize=16)
def _m5_sales_index(
    path_text: str,
    _size: int,
    _mtime_ns: int,
    source_sha256: str,
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Index M5 sales rows without materialising the wide demand columns.

    The verified file digest is part of the cache key.  Callers still hash the
    file before using this index, so a same-size/same-mtime mutation cannot
    bypass the source lock.
    """
    path = Path(path_text)
    with path.open("rb") as handle:
        header_raw = handle.readline()
        header = tuple(next(csv.reader([header_raw.decode("utf-8")])))
        offsets: dict[str, int] = {}
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            first_field = raw.split(b",", 1)[0].decode("utf-8")
            offsets.setdefault(first_field, offset)
    return header, offsets


def _read_m5_calendar(
    path: Path,
    *,
    source_sha256: str | None = None,
) -> dict[str, dict[str, str]]:
    digest = source_sha256 or _sha256_path(path)
    return {
        key: dict(value)
        for key, value in _m5_calendar_index(*_path_identity(path), digest).items()
    }


@lru_cache(maxsize=8)
def _m5_calendar_index(
    path_text: str,
    _size: int,
    _mtime_ns: int,
    source_sha256: str,
) -> dict[str, dict[str, str]]:
    with Path(path_text).open(newline="", encoding="utf-8") as handle:
        return {
            str(row["d"]): dict(row) for row in csv.DictReader(handle) if row.get("d")
        }


def _read_m5_first_price(
    path: Path,
    store_id: str,
    item_id: str,
    *,
    source_sha256: str | None = None,
) -> float:
    digest = source_sha256 or _sha256_path(path)
    prices = _m5_price_index(*_path_identity(path), digest)
    try:
        return prices[(store_id, item_id)]
    except KeyError as exc:
        raise ValueError(
            f"source_window_metadata_mismatch:price:{store_id}:{item_id}"
        ) from exc


@lru_cache(maxsize=8)
def _m5_price_index(
    path_text: str,
    _size: int,
    _mtime_ns: int,
    source_sha256: str,
) -> dict[tuple[str, str], float]:
    """Build the small first-price lookup once per verified price file."""
    prices: dict[tuple[str, str], float] = {}
    with Path(path_text).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("store_id") or ""), str(row.get("item_id") or ""))
            if key not in prices:
                prices[key] = float(row.get("sell_price") or 1.0)
    return prices


class OrgymInvmgmtBackend:
    """Release backend for OR-Gym ``InvManagement-v1`` lost-sales dynamics."""

    backend_kind = "orgym_invmgmt"

    def __init__(
        self,
        *,
        source_root: Path | None = None,
        m5_source_root: Path | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self._source_root = source_root or repo_root / "works" / "OR-Gym"
        self._m5_source_root = m5_source_root or repo_root / "works" / "M5"
        self._env: Any | None = None
        self._seed_obj: Any | None = None
        self._tick_records: list[InventoryTickRecord] = []
        self._current_tick = 0
        self._pending_action: np.ndarray | None = None
        self._last_observation: Any = None
        self._costs = {
            "inventory_procurement_cost": 0.0,
            "inventory_holding_cost": 0.0,
            "inventory_lost_sales_penalty": 0.0,
            "inventory_actuation_cost": 0.0,
        }
        self._cumulative_lost_sales: dict[str, float] = {"retailer": 0.0}
        self._last_tool_effects: list[dict[str, Any]] = []
        self._state_changing_tool_calls = 0
        self._config: dict[str, Any] = {}
        self._backend_config: dict[str, Any] = {}
        self._demand_profile_id = ""
        self._source_denominator_key = ""
        self._source_window: dict[str, Any] | None = None
        self._source_consumption_ticks: list[int] = []
        self._post_source_state_digests: list[dict[str, Any]] = []
        self._runtime_demand_events: list[dict[str, Any]] = []
        self._action_effects: list[dict[str, Any]] = []
        self._pending_order_intent: dict[str, Any] | None = None
        self._active_orders: list[dict[str, Any]] = []
        self._initial_state_digest = ""

    def reset(self, scenario_seed: Any) -> None:
        self._seed_obj = scenario_seed
        cfg = dict(getattr(scenario_seed, "backend_config", {}) or {})
        self._backend_config = cfg
        orgym_cfg = dict(cfg.get("orgym_env_config") or {})
        provenance_obj = getattr(scenario_seed, "provenance", None)
        provenance = dict(vars(provenance_obj)) if provenance_obj is not None else {}
        is_m5 = bool(
            cfg.get("m5_demand_handle")
            or cfg.get("m5_sku_store_key")
            or "m5" in str(provenance.get("data_source") or "").lower()
        )
        self._source_window = None
        if is_m5:
            window = resolve_orgym_m5_source_window(
                backend_config=cfg,
                provenance=provenance,
                repo_root=Path(__file__).resolve().parents[3],
                source_root=self._m5_source_root,
            )
            baked_demand = orgym_cfg.get("user_D")
            if (
                baked_demand is not None
                and [int(value) for value in baked_demand]
                != window["normalized_demand"]
            ):
                raise ValueError("source_baked_demand_mismatch")
            orgym_cfg = _build_orgym_env_config_from_source(
                public_config=orgym_cfg,
                backend_config=cfg,
                demand=window["normalized_demand"],
                sell_price=float(window["sell_price"]),
                seed=int(
                    orgym_cfg.get("seed_int", getattr(scenario_seed, "seed", 0)) or 0
                ),
            )
            expected_env_hash = str(
                cfg.get("orgym_env_config_hash") or ""
            ).removeprefix("sha256:")
            actual_env_hash = _semantic_digest(orgym_cfg)
            if expected_env_hash and actual_env_hash != expected_env_hash:
                raise ValueError(
                    "source_baked_env_config_mismatch: "
                    f"{actual_env_hash} != {expected_env_hash}"
                )
            self._source_window = window
        if not orgym_cfg:
            raise ValueError(
                "orgym_invmgmt seed missing backend_config.orgym_env_config"
            )
        configured_lead_times = _as_float_list(orgym_cfg.get("L", []))
        if not configured_lead_times or any(
            lead_time < 1 for lead_time in configured_lead_times
        ):
            raise ValueError("orgym_invmgmt_requires_positive_lead_time")
        if str(cfg.get("inventory_environment_id", ORGYM_ENV_ID)) != ORGYM_ENV_ID:
            raise ValueError("orgym_invmgmt release backend requires InvManagement-v1")
        self._config = orgym_cfg
        self._demand_profile_id = str(cfg.get("demand_profile_id") or "orgym_native")
        self._source_denominator_key = str(
            cfg.get("source_denominator_key")
            or f"orgym_invmgmt:{ORGYM_ENV_ID}:{self._demand_profile_id}"
        )
        env_cls = _load_orgym_lost_sales_env(self._source_root)
        self._env = env_cls(**orgym_cfg)
        seed_int = int(
            orgym_cfg.get("seed_int", getattr(scenario_seed, "seed", 0)) or 0
        )
        if hasattr(self._env, "seed"):
            self._env.seed(seed_int)
        self._last_observation = self._env.reset()
        self._current_tick = 0
        stages = max(1, int(getattr(self._env, "num_stages", 2)) - 1)
        self._pending_action = np.zeros(stages, dtype=np.int16)
        self._tick_records = []
        for key in self._costs:
            self._costs[key] = 0.0
        self._cumulative_lost_sales = {"retailer": 0.0}
        self._last_tool_effects = []
        self._state_changing_tool_calls = 0
        self._source_consumption_ticks = [0] if self._source_window else []
        self._post_source_state_digests = []
        self._runtime_demand_events = []
        self._action_effects = []
        self._pending_order_intent = None
        self._active_orders = []
        self._initial_state_digest = self._inventory_state_digest()

    def snapshot(self) -> dict[str, Any]:
        env = self._require_env()
        demand = _as_float_list(getattr(env, "user_D", []))
        period = int(getattr(env, "period", self._current_tick) or 0)
        forecast_horizon = int(
            self._backend_config.get("observation_forecast_horizon", len(demand))
            or len(demand)
        )
        if bool(self._backend_config.get("hide_full_demand_stream", False)):
            visible_demand = demand[
                period : min(len(demand), period + forecast_horizon)
            ]
        else:
            visible_demand = demand
        inventory = _as_float_list(getattr(env, "I", [[0.0]])[period])
        pipeline = _as_float_list(getattr(env, "T", [[0.0]])[period])
        capacity = _as_float_list(getattr(env, "supply_capacity", []))
        lead_times = _as_float_list(getattr(env, "lead_time", []))
        return {
            "domain": "logistics",
            "backend_kind": self.backend_kind,
            "decision_opportunity": True,
            "decision_cadence": {
                "mode": "native_periodic",
                "native_opportunity": True,
            },
            "inventory_environment_id": ORGYM_ENV_ID,
            "source_denominator_key": self._source_denominator_key,
            "period": period,
            "tick": self._current_tick,
            "inventory_on_hand": inventory,
            "pipeline_inventory": pipeline,
            "supply_capacity": capacity,
            "lead_times": lead_times,
            "pending_action": _as_float_list(self._pending_action),
            "demand_forecast_units": visible_demand,
            "demand_forecast_start_period": period,
            "demand_forecast_horizon": len(visible_demand),
            "demand_forecast_is_partial": bool(
                self._backend_config.get("hide_full_demand_stream", False)
            ),
            "next_demand_units": float(demand[period]) if period < len(demand) else 0.0,
            "cumulative_lost_sales_units": float(
                self._cumulative_lost_sales.get("retailer", 0.0)
            ),
            "last_tool_effects": list(self._last_tool_effects[-4:]),
            "totals": {
                "aggregate_demand_mw": float(
                    sum(r.aggregate_demand for r in self._tick_records)
                ),
                "served_demand_mw": float(
                    sum(r.served_demand for r in self._tick_records)
                ),
                "balance_error_mw": float(
                    self._cumulative_lost_sales.get("retailer", 0.0)
                ),
                "reserves_required_mw": float(demand[period])
                if period < len(demand)
                else 0.0,
                "reserves_procured_mw": float(inventory[0] + pipeline[0])
                if inventory and pipeline
                else 0.0,
            },
            "entities": {
                "retailer": {
                    "kind": "inventory_stage",
                    "stage": 0,
                    "on_hand_units": float(inventory[0]) if inventory else 0.0,
                    "pipeline_units": float(pipeline[0]) if pipeline else 0.0,
                    "next_demand_units": float(demand[period])
                    if period < len(demand)
                    else 0.0,
                    "criticality": 0.5,
                }
            },
        }

    def place_replenishment_order(
        self,
        *,
        quantity: int,
        stage: int = 0,
        execution_tick: int | None = None,
    ) -> dict[str, Any]:
        env = self._require_env()
        capacity = _as_float_list(getattr(env, "supply_capacity", []))
        n_stages = max(1, len(capacity))
        stage = int(stage)
        quantity = int(quantity)
        if stage < 0 or stage >= n_stages:
            return {
                "_status": "error",
                "error": "unknown_inventory_stage",
                "stage": stage,
            }
        if quantity <= 0:
            return {
                "_status": "no_effect",
                "reason": "quantity_must_be_positive",
                "stage": stage,
            }
        stage_capacity = int(capacity[stage] if stage < len(capacity) else quantity)
        if quantity > stage_capacity:
            return {
                "_status": "error",
                "error": "order_exceeds_stage_capacity",
                "stage": stage,
                "quantity_requested": quantity,
                "capacity_units": float(stage_capacity),
            }
        if self._pending_order_intent is not None:
            return {
                "_status": "error",
                "error": "replenishment_order_already_pending_this_tick",
                "stage": stage,
                "quantity_requested": quantity,
            }
        clipped = quantity
        action = np.zeros(n_stages, dtype=np.int16)
        action[stage] = int(clipped)
        self._pending_action = action
        physical_tick = int(
            self._current_tick if execution_tick is None else execution_tick
        )
        lead_time = int(
            getattr(env, "lead_time", [0])[stage]
            if stage < len(getattr(env, "lead_time", []))
            else 0
        )
        result = {
            "_status": "order_placed",
            "state_changed": True,
            "stage": stage,
            "quantity_requested": quantity,
            "quantity_accepted": int(clipped),
            "capacity_units": float(
                capacity[stage] if stage < len(capacity) else clipped
            ),
            "effect_due_tick": physical_tick,
            "pipeline_effect_tick": physical_tick,
            "arrival_due_tick": physical_tick + lead_time,
        }
        self._pending_order_intent = {
            "stage": stage,
            "quantity_requested": int(clipped),
            "quantity": int(clipped),
            "placed_tick": physical_tick,
            "due_tick": physical_tick + lead_time,
            "lead_time": lead_time,
        }
        self._last_tool_effects.append(result)
        return result

    def bind_tool_result(
        self,
        *,
        name: str,
        call_id: str,
        evidence_id: str | None,
        payload: dict[str, Any],
        causal_parent_event_id: str | None = None,
    ) -> None:
        if (
            name != "place_replenishment_order"
            or self._pending_order_intent is None
            or payload.get("_status") != "order_placed"
        ):
            return
        self._pending_order_intent.update(
            {
                "call_id": str(call_id),
                "evidence_ids": [str(evidence_id)] if evidence_id else [],
            }
        )
        if causal_parent_event_id:
            self._pending_order_intent["causal_parent_event_id"] = str(
                causal_parent_event_id
            )

    def tick(self, current_tick: int) -> InventoryTickRecord:
        env = self._require_env()
        tick = int(current_tick)
        self._current_tick = tick
        action = self._pending_action
        if action is None:
            action = np.zeros(
                max(1, int(getattr(env, "num_stages", 2)) - 1), dtype=np.int16
            )
        before_period = int(getattr(env, "period", tick) or tick)
        pending_order = (
            dict(self._pending_order_intent)
            if self._pending_order_intent is not None
            else None
        )
        before_state_digest = self._native_period_state_digest(
            period=before_period,
            state_period=before_period,
        )
        observation, reward, done, _info = env.step(action)
        self._last_observation = observation
        self._pending_action = np.zeros_like(action)
        record = self._record_from_native_step(
            tick=tick,
            before_period=before_period,
            action=action,
            reward=float(reward),
            done=bool(done),
        )
        after_state_digest = self._native_period_state_digest(
            period=before_period,
            state_period=before_period + 1,
        )
        if pending_order is not None and np.count_nonzero(action):
            stage = int(pending_order["stage"])
            native_quantity = (
                int(env.R[before_period, stage])
                if before_period < env.R.shape[0] and stage < env.R.shape[1]
                else 0
            )
            self._pending_order_intent = None
            if native_quantity > 0:
                pending_order["quantity"] = native_quantity
                pending_order["native_action_period"] = before_period
                self._active_orders.append(pending_order)
                self._state_changing_tool_calls += 1
                entry_fields = ["replenishment_orders"]
                if int(pending_order["lead_time"]) > 0:
                    entry_fields.append("pipeline_inventory")
                event_id = (
                    "replenishment_order_entered_pipeline:"
                    f"{pending_order.get('call_id', 'unbound')}:{tick}"
                )
                entry = {
                    "type": "replenishment_order_entered_pipeline",
                    "event_id": event_id,
                    "origin": "agent_caused",
                    "agent_caused": True,
                    "tick": tick,
                    "decision_required": False,
                    "actionable": False,
                    "call_id": str(pending_order.get("call_id") or ""),
                    "tool_name": "place_replenishment_order",
                    "requested_action": {
                        "stage": stage,
                        "quantity": int(pending_order["quantity_requested"]),
                    },
                    "applied_action": {
                        "stage": stage,
                        "quantity": native_quantity,
                        "arrival_due_tick": int(pending_order["due_tick"]),
                    },
                    "before_state_digest": before_state_digest,
                    "after_state_digest": after_state_digest,
                    "changed_state_fields": entry_fields,
                    "outcome_tick": tick,
                    "evidence_ids": list(pending_order.get("evidence_ids") or []),
                    "action_to_outcome_edge": {
                        "source": f"call:{pending_order.get('call_id', '')}",
                        "target": f"outcome:{event_id}",
                        "kind": "action_to_outcome",
                    },
                }
                if pending_order.get("causal_parent_event_id"):
                    entry["causal_parent_event_id"] = str(
                        pending_order["causal_parent_event_id"]
                    )
                record.realized_events.append(entry)
                self._action_effects.append(entry)
        due_orders = [
            order for order in self._active_orders if int(order["due_tick"]) == tick
        ]
        for order in due_orders:
            stage = int(order["stage"])
            placed_period = int(order.get("native_action_period", order["placed_tick"]))
            delivered_quantity = (
                int(env.R[placed_period, stage])
                if placed_period < env.R.shape[0] and stage < env.R.shape[1]
                else 0
            )
            self._active_orders.remove(order)
            if delivered_quantity <= 0:
                continue
            event_id = f"replenishment_arrived:{order.get('call_id', 'unbound')}:{tick}"
            effect = {
                "type": "replenishment_arrived",
                "event_id": event_id,
                "origin": "agent_caused",
                "agent_caused": True,
                "tick": tick,
                "decision_required": False,
                "actionable": False,
                "call_id": str(order.get("call_id") or ""),
                "tool_name": "place_replenishment_order",
                "requested_action": {
                    "stage": int(order["stage"]),
                    "quantity": int(order["quantity"]),
                },
                "applied_action": {
                    "stage": stage,
                    "quantity": delivered_quantity,
                    "placed_tick": int(order["placed_tick"]),
                },
                "before_state_digest": before_state_digest,
                "after_state_digest": after_state_digest,
                "changed_state_fields": self._arrival_changed_state_fields(
                    stage=stage,
                    before_period=before_period,
                    after_period=before_period + 1,
                ),
                "outcome_tick": tick,
                "evidence_ids": list(order.get("evidence_ids") or []),
                "action_to_outcome_edge": {
                    "source": f"call:{order.get('call_id', '')}",
                    "target": f"outcome:{event_id}",
                    "kind": "action_to_outcome",
                },
            }
            if order.get("causal_parent_event_id"):
                effect["causal_parent_event_id"] = str(order["causal_parent_event_id"])
            record.realized_events.append(effect)
            self._action_effects.append(effect)
        if self._source_window is not None:
            if tick not in self._source_consumption_ticks:
                self._source_consumption_ticks.append(tick)
            self._post_source_state_digests.append(
                {"tick": tick, "sha256": after_state_digest}
            )
            self._runtime_demand_events.extend(
                event
                for event in record.realized_events
                if event.get("type") == "inventory_demand_realized"
            )
        self._tick_records.append(record)
        return record

    def protocol21_source_trace(self) -> dict[str, Any]:
        if self._source_window is None:
            return {
                "status": "held",
                "proof_kind": "derived_source_window",
                "runtime_trace_observed": False,
                "evidence_from_scenario_config_only": True,
                "blockers": ["source_window_metadata_missing"],
            }
        window = self._source_window
        semantic_payload = {
            "source_window": window["trace_source_window"],
            "consumption_ticks": sorted(set(self._source_consumption_ticks)),
            "initial_state_digest": self._initial_state_digest,
            "post_source_state_digests": self._post_source_state_digests,
            "runtime_demand_events": self._runtime_demand_events,
            "order_action_effects": self._action_effects,
        }
        source_effect = any(
            int(event.get("tick") or 0) > 0
            and float(event.get("demand_units") or 0.0) >= 1.0
            for event in self._runtime_demand_events
        )
        return {
            "status": "passed" if source_effect else "held",
            "proof_kind": "derived_source_window",
            "runtime_opened_assets": list(window["runtime_opened_assets"]),
            "opened_source_paths": sorted(window["opened_source_sha256"]),
            "opened_source_sha256": dict(window["opened_source_sha256"]),
            "locked_derivation_source_hashes": dict(
                window["locked_derivation_source_hashes"]
            ),
            "consumed_source_hashes": dict(window["locked_derivation_source_hashes"]),
            "lineage_source_hashes": dict(window["locked_derivation_source_hashes"]),
            "consumed_window_sha256": window["source_window_sha256"],
            "recipe_version": M5_RECIPE_VERSION,
            "source_window": dict(window["trace_source_window"]),
            "consumed_channels": list(window["consumed_channels"]),
            "derived_backend_state_fields": [
                "on_hand_inventory",
                "inventory_position",
                "pipeline_orders",
                "realized_demand",
                "lost_sales",
            ],
            "consumption_ticks": sorted(set(self._source_consumption_ticks)),
            "initial_state_digest": self._initial_state_digest,
            "post_source_state_digests": list(self._post_source_state_digests),
            "runtime_demand_events": list(self._runtime_demand_events),
            "order_action_effects": list(self._action_effects),
            "source_state_effect_observed": source_effect,
            "state_effect_observed": source_effect,
            "deterministic_source_trace": True,
            "trace_semantic_digest": _semantic_digest(semantic_payload),
            "runtime_trace_observed": True,
            "evidence_from_scenario_config_only": False,
            "blockers": [] if source_effect else ["source_state_effect_unproven"],
        }

    def _inventory_state_digest(self) -> str:
        env = self._require_env()
        period = int(getattr(env, "period", 0) or 0)
        inventory = (
            _as_float_list(env.I[period]) if period < len(getattr(env, "I", [])) else []
        )
        pipeline = (
            _as_float_list(env.T[period]) if period < len(getattr(env, "T", [])) else []
        )
        lost_sales = (
            float(np.sum(env.LS[:period])) if period and hasattr(env, "LS") else 0.0
        )
        return _semantic_digest(
            {
                "period": period,
                "next_demand_units": (
                    float(env.user_D[period])
                    if period < len(getattr(env, "user_D", []))
                    else 0.0
                ),
                "on_hand_inventory": inventory,
                "pipeline_orders": pipeline,
                "inventory_position": [
                    inventory[index] + pipeline[index]
                    for index in range(min(len(inventory), len(pipeline)))
                ],
                "lost_sales": lost_sales,
            }
        )

    def _native_period_state_digest(self, *, period: int, state_period: int) -> str:
        """Digest native OR-Gym R/I/T/LS arrays at one transition boundary."""
        env = self._require_env()
        state_index = min(max(0, int(state_period)), env.I.shape[0] - 1)
        action_index = min(max(0, int(period)), env.R.shape[0] - 1)
        return _semantic_digest(
            {
                "native_action_period": int(period),
                "native_state_period": state_index,
                "replenishment_orders": _as_float_list(env.R[action_index]),
                "on_hand_inventory": _as_float_list(env.I[state_index]),
                "pipeline_inventory": _as_float_list(env.T[state_index]),
                "cumulative_lost_sales": float(np.sum(env.LS[:state_index])),
            }
        )

    def _arrival_changed_state_fields(
        self, *, stage: int, before_period: int, after_period: int
    ) -> list[str]:
        env = self._require_env()
        before_index = min(max(0, before_period), env.I.shape[0] - 1)
        after_index = min(max(0, after_period), env.I.shape[0] - 1)
        changed = ["inventory_receipts"]
        if float(env.T[before_index, stage]) != float(env.T[after_index, stage]):
            changed.append("pipeline_inventory")
        if float(env.I[before_index, stage]) != float(env.I[after_index, stage]):
            changed.append("on_hand_inventory")
        return changed

    def scoring_records(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in self._tick_records:
            lost_sales_fraction = rec.unmet_demand / max(1.0, rec.aggregate_demand)
            out.append(
                {
                    "tick": rec.tick,
                    "aggregate_demand_mw": rec.aggregate_demand,
                    "aggregate_generation_mw": rec.served_demand,
                    "balance_error_mw": rec.unmet_demand,
                    "reserves_required_mw": rec.required_standby,
                    "reserves_procured_mw": rec.procured_standby,
                    "production_cost": rec.routing_cost,
                    "startup_cost": rec.dispatch_fixed_cost,
                    "shed_penalty": rec.drop_penalty,
                    "rho_max": rec.max_utilization,
                    "n_overloads": rec.n_capacity_violations,
                    "n_voltage_violations": 0,
                    "n_disconnected_lines": 0,
                    "done": False,
                    "catastrophic_failure": False,
                    "safety_violation_severity": min(
                        1.0, max(0.0, lost_sales_fraction)
                    ),
                }
            )
        return out

    def ground_truth_costs(self) -> dict[str, float]:
        return {key: round(float(value), 6) for key, value in self._costs.items()}

    def per_customer_unmet_units(self) -> dict[str, float]:
        return dict(self._cumulative_lost_sales)

    def _record_from_native_step(
        self,
        *,
        tick: int,
        before_period: int,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> InventoryTickRecord:
        env = self._require_env()
        demand = float(env.D[before_period]) if before_period < len(env.D) else 0.0
        served = (
            float(env.S[before_period, 0]) if before_period < env.S.shape[0] else 0.0
        )
        lost = (
            float(env.LS[before_period, 0]) if before_period < env.LS.shape[0] else 0.0
        )
        orders = (
            float(np.sum(env.R[before_period]))
            if before_period < env.R.shape[0]
            else 0.0
        )
        ending_inventory = (
            float(np.sum(env.I[before_period + 1]))
            if before_period + 1 < env.I.shape[0]
            else 0.0
        )
        unit_cost = _as_float_list(getattr(env, "unit_cost", []))
        demand_cost = _as_float_list(getattr(env, "demand_cost", []))
        holding_cost = _as_float_list(getattr(env, "holding_cost", []))
        procurement_cost = sum(
            float(env.R[before_period, i]) * float(unit_cost[i])
            for i in range(min(env.R.shape[1], len(unit_cost)))
        )
        lost_penalty = float(lost) * (float(demand_cost[0]) if demand_cost else 0.0)
        holding = sum(
            float(env.I[before_period + 1, i]) * float(holding_cost[i])
            for i in range(min(env.I.shape[1], len(holding_cost)))
            if before_period + 1 < env.I.shape[0]
        )
        actuation_cost = 0.25 * float(np.count_nonzero(action))
        self._costs["inventory_procurement_cost"] += procurement_cost
        self._costs["inventory_holding_cost"] += holding
        self._costs["inventory_lost_sales_penalty"] += lost_penalty
        self._costs["inventory_actuation_cost"] += actuation_cost
        self._cumulative_lost_sales["retailer"] = (
            self._cumulative_lost_sales.get("retailer", 0.0) + lost
        )
        capacity = _as_float_list(getattr(env, "supply_capacity", []))
        max_capacity = max(capacity) if capacity else max(1.0, orders)
        pressure = max(
            float(demand) / max(max_capacity, 1.0),
            float(orders) / max(max_capacity, 1.0),
        )
        events: list[dict[str, Any]] = []
        if tick > 0 and (demand > 0 or lost > 0):
            has_later_response = tick + 1 < int(getattr(env, "num_periods", 0) or 0)
            event_class = _INVENTORY_EVENT_REGISTRY.get(
                "inventory_demand_realized"
            )
            actionable = bool(event_class is not None and has_later_response)
            events.append(
                {
                    "type": "inventory_demand_realized",
                    "event_id": (
                        f"inventory_demand_realized:{self._demand_profile_id}:{tick}"
                    ),
                    "origin": "source_schedule",
                    "event_class": event_class or "telemetry",
                    "tick": tick,
                    "stage": "retailer",
                    "demand_units": demand,
                    "served_units": served,
                    "lost_sales_units": lost,
                    "intensity": min(1.0, demand / max(max_capacity, 1.0)),
                    "changed_state_fields": (
                        [
                            "on_hand_inventory",
                            "inventory_position",
                            "realized_demand",
                            "lost_sales",
                        ]
                        if demand > 0
                        else []
                    ),
                    "materiality_metric": "demand_units",
                    "materiality_value": demand,
                    "materiality_threshold": 1.0,
                    "materiality_passed": demand >= 1.0,
                    "decision_required": actionable,
                    "actionable": actionable,
                    "response_window_required": actionable,
                    "response_opportunity_tick": (
                        tick + 1 if actionable else None
                    ),
                    "terminal_response_window_missing": False,
                }
            )
        return InventoryTickRecord(
            tick=tick,
            aggregate_demand=demand,
            served_demand=served,
            unmet_demand=lost,
            required_standby=demand,
            procured_standby=max(0.0, ending_inventory + orders),
            routing_cost=procurement_cost + holding,
            dispatch_fixed_cost=actuation_cost,
            drop_penalty=lost_penalty,
            max_utilization=round(float(pressure), 6),
            n_capacity_violations=1 if lost > 0 else 0,
            n_time_window_violations=0,
            n_failed_routes=1 if lost > 0 else 0,
            done=False,
            realized_events=events,
        )

    def _require_env(self) -> Any:
        if self._env is None:
            raise RuntimeError("OrgymInvmgmtBackend has not been reset")
        return self._env


def _load_orgym_lost_sales_env(source_root: Path) -> Any:
    if not source_root.exists():
        raise InventoryDatasetLockPending(f"OR-Gym source tree missing: {source_root}")
    source_str = str(source_root)
    inserted = False
    if source_str not in sys.path:
        sys.path.insert(0, source_str)
        inserted = True
    try:
        if not hasattr(np, "Inf"):
            np.Inf = np.inf  # type: ignore[attr-defined]
        from or_gym.envs.supply_chain.inventory_management import (
            InvManagementLostSalesEnv,
        )

        return InvManagementLostSalesEnv
    finally:
        if inserted:
            with suppress(ValueError):
                sys.path.remove(source_str)


def _as_float_list(value: Any) -> list[float]:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        return [float(v) for v in arr.tolist()]
    except Exception:
        if isinstance(value, list):
            out: list[float] = []
            for item in value:
                with suppress(TypeError, ValueError):
                    out.append(float(item))
            return out
        return []


def register_orgym_inventory_tools(
    reg: ToolRegistry, backend: OrgymInvmgmtBackend
) -> None:
    """Register OR-Gym inventory controls through the shared tool protocol."""

    reg.register(
        ToolSpec(
            name="forecast_demand",
            description=(
                "Read the source-locked demand values currently visible to "
                "the planner. Hidden future periods are never returned."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_h_orgym_forecast_demand(backend),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="inventory_demand_forecast",
            cost_units=0.1,
        )
    )
    reg.register(
        ToolSpec(
            name="place_replenishment_order",
            description=(
                "Place a replenishment order in the OR-Gym InvManagement-v1 "
                "lost-sales environment. The order mutates the native OR-Gym "
                "action vector consumed on the next backend tick."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "quantity": {"type": "integer", "minimum": 1},
                    "stage": {"type": "integer", "minimum": 0},
                },
                "required": ["quantity"],
            },
            handler=_h_orgym_place_replenishment_order(backend),
            state_changing=True,
            semantic_role="control",
            native_target_kind="inventory_replenishment_pipeline",
            actuator_family="replenishment_ordering",
            cost_units=0.1,
        )
    )
    reg.register(
        ToolSpec(
            name="wait",
            description="Advance one inventory period without placing a replenishment order.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda args, ctx: {"_status": "waited"},
            semantic_role="meta",
            native_target_kind="simulation_clock",
        )
    )
    reg.register(
        ToolSpec(
            name="noop",
            description="Alias for wait.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda args, ctx: {"_status": "noop"},
            semantic_role="meta",
            native_target_kind="simulation_clock",
        )
    )


def _h_orgym_forecast_demand(backend: OrgymInvmgmtBackend):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        snapshot = backend.snapshot()
        result = {
            "_status": "observed",
            "demand_forecast_units": list(snapshot["demand_forecast_units"]),
            "forecast_start_period": int(snapshot["demand_forecast_start_period"]),
            "forecast_horizon": int(snapshot["demand_forecast_horizon"]),
            "is_partial": bool(snapshot["demand_forecast_is_partial"]),
            "source_denominator_key": snapshot["source_denominator_key"],
        }
        evidence = ctx.extra.get("evidence")
        if isinstance(evidence, EvidenceLogger):
            result["evidence_id"] = evidence.log(
                "inventory_demand_forecast",
                ctx.tick,
                payload={
                    "tool": "forecast_demand",
                    "ok": True,
                    "backend_kind": "orgym_invmgmt",
                    **result,
                },
                source="tool",
            )
        return result

    return handler


def _h_orgym_place_replenishment_order(backend: OrgymInvmgmtBackend):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        result = backend.place_replenishment_order(
            quantity=int(args.get("quantity") or 0),
            stage=int(args.get("stage") or 0),
            execution_tick=ctx.tick,
        )
        evidence = ctx.extra.get("evidence")
        if isinstance(evidence, EvidenceLogger):
            result["evidence_id"] = evidence.log(
                "inventory_tool_effect",
                ctx.tick,
                payload={
                    "tool": "place_replenishment_order",
                    "ok": result.get("_status") == "order_placed",
                    "backend_kind": "orgym_invmgmt",
                    **result,
                },
                source="tool",
            )
        return result

    return handler


@dataclass
class M5InventoryProbeBackend:
    """Non-release M5 adapter probe with deterministic inventory dynamics.

    This is not an OR-Gym release backend. It is a source-locked tool/evidence
    probe that proves OPERATE can convert real M5 SKU-store demand rows
    into native inventory state and mutate that state through ``ToolRegistry``.
    """

    demand_rows: dict[str, dict[str, str]]
    sell_prices: dict[tuple[str, str, str], float]
    calendar_days: list[str]
    pipeline_orders: list[dict[str, Any]] = field(default_factory=list)
    on_hand_units: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_source_root(cls, source_root: Path) -> M5InventoryProbeBackend:
        sales_rows = _csv_rows(source_root / "sales_train_evaluation.csv")
        calendar_rows = _csv_rows(source_root / "calendar.csv")
        price_rows = _csv_rows(source_root / "sell_prices.csv")
        demand_rows = {
            str(row.get("id") or ""): row for row in sales_rows if row.get("id")
        }
        sell_prices = {
            (
                str(row.get("store_id") or ""),
                str(row.get("item_id") or ""),
                str(row.get("wm_yr_wk") or ""),
            ): float(row.get("sell_price") or 0.0)
            for row in price_rows
            if row.get("store_id") and row.get("item_id") and row.get("wm_yr_wk")
        }
        calendar_days = [
            str(row.get("d") or "") for row in calendar_rows if row.get("d")
        ]
        return cls(
            demand_rows=demand_rows,
            sell_prices=sell_prices,
            calendar_days=calendar_days,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "sku_store_series": len(self.demand_rows),
            "calendar_days": len(self.calendar_days),
            "pipeline_orders": list(self.pipeline_orders),
            "pipeline_units_total": sum(
                int(order.get("quantity", 0)) for order in self.pipeline_orders
            ),
            "on_hand_units": dict(self.on_hand_units),
        }

    def demand_for_tick(self, *, sku_store_key: str, tick: int) -> int:
        row = self.demand_rows.get(sku_store_key) or {}
        day = f"d_{tick + 1}"
        try:
            return max(0, int(float(row.get(day) or 0)))
        except ValueError:
            return 0

    def receive_due_orders(self, *, sku_store_key: str, tick: int) -> int:
        received = 0
        remaining = []
        for order in self.pipeline_orders:
            if order.get("sku_store_key") != sku_store_key:
                remaining.append(order)
                continue
            due_tick = int(order.get("placed_tick", 0)) + int(
                order.get("lead_time_ticks", 0)
            )
            if due_tick <= tick:
                received += int(order.get("quantity", 0))
            else:
                remaining.append(order)
        self.pipeline_orders = remaining
        if received:
            self.on_hand_units[sku_store_key] = (
                self.on_hand_units.get(sku_store_key, 0) + received
            )
        return received

    def consume_demand(
        self, *, sku_store_key: str, tick: int, evidence: EvidenceLogger | None = None
    ) -> dict[str, Any]:
        demand = self.demand_for_tick(sku_store_key=sku_store_key, tick=tick)
        on_hand_before = self.on_hand_units.get(sku_store_key, 0)
        served = min(on_hand_before, demand)
        stockout = max(0, demand - served)
        on_hand_after = on_hand_before - served
        self.on_hand_units[sku_store_key] = on_hand_after
        payload = {
            "sku_store_key": sku_store_key,
            "tick": tick,
            "demand_units": demand,
            "served_units": served,
            "stockout_units": stockout,
            "on_hand_before": on_hand_before,
            "on_hand_after": on_hand_after,
        }
        if evidence is not None:
            payload["evidence_id"] = evidence.log(
                "inventory_stockout_event" if stockout else "inventory_service_level",
                tick,
                payload=payload,
                source="engine",
            )
        return payload

    def place_replenishment_order(
        self, *, sku_store_key: str, quantity: int, lead_time_ticks: int
    ) -> dict[str, Any]:
        if sku_store_key not in self.demand_rows:
            return {
                "_status": "error",
                "error": "unknown_sku_store_key",
                "sku_store_key": sku_store_key,
            }
        if quantity <= 0:
            return {
                "_status": "no_effect",
                "reason": "quantity_must_be_positive",
                "sku_store_key": sku_store_key,
            }
        if lead_time_ticks < 1:
            return {
                "_status": "out_of_range",
                "reason": "lead_time_ticks_must_be_at_least_1",
                "sku_store_key": sku_store_key,
            }

        row = self.demand_rows[sku_store_key]
        price = self._sell_price_for_row(row)
        order = {
            "sku_store_key": sku_store_key,
            "quantity": int(quantity),
            "lead_time_ticks": int(lead_time_ticks),
            "placed_tick": int(getattr(self, "_current_tick", 0)),
            "unit_cost": price,
            "order_cost": float(price * quantity),
        }
        self.pipeline_orders.append(order)
        return {
            "_status": "order_placed",
            "state_changed": True,
            **order,
            "pipeline_units_total": self.snapshot()["pipeline_units_total"],
        }

    def _sell_price_for_row(self, row: dict[str, str]) -> float:
        item_id = str(row.get("item_id") or "")
        store_id = str(row.get("store_id") or "")
        wm_yr_wk = "11101"
        key = (store_id, item_id, wm_yr_wk)
        if key in self.sell_prices:
            return self.sell_prices[key]
        for (price_store, price_item, _week), price in self.sell_prices.items():
            if price_store == store_id and price_item == item_id:
                return price
        return 0.0


def register_inventory_probe_tools(
    reg: ToolRegistry, backend: M5InventoryProbeBackend
) -> None:
    """Register the non-release M5 inventory probe tools through core protocol."""

    reg.register(
        ToolSpec(
            name="place_replenishment_order",
            description=(
                "Place a replenishment order for a source-locked M5 SKU-store "
                "series, adding units to deterministic pipeline inventory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "sku_store_key": {"type": "string"},
                    "quantity": {"type": "integer", "minimum": 1},
                    "lead_time_ticks": {"type": "integer", "minimum": 1},
                },
                "required": ["sku_store_key", "quantity", "lead_time_ticks"],
            },
            handler=_h_place_replenishment_order(backend),
            state_changing=True,
            semantic_role="control",
            native_target_kind="inventory_replenishment_pipeline",
            actuator_family="replenishment_ordering",
            cost_units=0.1,
        )
    )


def _h_place_replenishment_order(backend: M5InventoryProbeBackend):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        backend._current_tick = int(ctx.tick)
        result = backend.place_replenishment_order(
            sku_store_key=str(args.get("sku_store_key") or ""),
            quantity=int(args.get("quantity") or 0),
            lead_time_ticks=int(args.get("lead_time_ticks") or 0),
        )
        evidence = ctx.extra.get("evidence")
        if isinstance(evidence, EvidenceLogger):
            result["evidence_id"] = evidence.log(
                "inventory_tool_effect",
                ctx.tick,
                payload={
                    "tool": "place_replenishment_order",
                    "ok": result.get("_status") == "order_placed",
                    **result,
                },
                source="tool",
            )
        return result

    return handler


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def refuse_without_demand_dataset(*, has_real_demand_lock: bool = False) -> None:
    """Generator guard (mirrors ``generate_egret_scenarios.py``).

    Exits with code 3 when invoked without a real demand dataset lock so the
    live registry never carries unauditable inventory rows.
    """
    if has_real_demand_lock:
        return
    sys.stderr.write(
        "[orgym_invmgmt] REFUSED: inventory_replenishment requires a real "
        "demand dataset lock (M5 / Corporacion-Favorita) before empirical "
        "inventory rows are registered. Exiting 3.\n"
    )
    raise SystemExit(3)
