"""
domains.microgrid.seeds.from_pymgrid — structural family seed builders.

Builds ``MicrogridScenarioSeed`` objects for the four v0.7 microgrid
families from the **baked** NSRDB/OEDI overlays (``from_nrel_microgrid``).
The released pymgrid-family provenance is the baked NSRDB/OEDI series, NOT
pymgrid's bundled/synthetic series (spec §3). The optional pymgrid
cross-check (``EmsSimulator.evaluate_with_pymgrid``) is the only place the
bundled pymgrid runtime is touched.

The integer ``seed`` is STRUCTURAL, not fog-only (spec §5):

- ``islanding_trigger_tick = 2 + seed % 4``
- ``controllable_der_count = base(level) + seed % 3``
- ``genset_available       = bool(seed % 2)``
- ``forecast_regime_idx    = seed % len(FORECAST_REGIMES)``
- ``critical_load_fraction`` tier = one of three tiers by ``seed % 3``
- ``hidden_parity          = seed % 2`` (DER failure visible vs discovered)

so a ``(mode, level)`` bucket across 6–8 seeds has non-zero std on ≥3
complexity metrics (``n_islanding_events``, ``controllable_asset_count``,
``forecast_error_sigma``, ``decision_depth``, ``critical_load_fraction``) —
pinned by a unit test. Signatures stay stable per fixed kwargs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .from_nrel_microgrid import (
    BAKED_SITES,
    FORECAST_REGIMES,
    _pymgrid_version,
    _runtime_lock_signature,
    baked_overlay_provenance_files,
    load_overlay,
    site_is_anchored,
)
from .schema import (
    MicrogridLoad,
    MicrogridScenarioSeed,
    Perturbation,
    Provenance,
    criticality_default,
)
from .source_locks import provenance_lock_kwargs

_BASE_DER = {"basic": 2, "medium": 3, "high": 4, "extreme": 5, "cascading": 5}
_BASE_STRESSORS = {"basic": 1, "medium": 2, "high": 3, "extreme": 4, "cascading": 4}
_HORIZON_BY_FAMILY = {
    "microgrid_islanding_24h": 24,
    "microgrid_economic_dispatch_24h": 24,
    "microgrid_solar_ramp_24h": 24,
    "microgrid_lv_voltage_6h": 6,
    "microgrid_lv_voltage_staged_6h": 6,
    "microgrid_lv_voltage_recovery_10h": 10,
}
_CRITICAL_TIERS = [0.20, 0.35, 0.50]
_SITES = ["phoenix_az", "denver_co", "boston_ma", "seattle_wa"]
_PYMGRID_HONEST_ZERO = [
    "rho_max",
    "n_overloads",
    "n_voltage_violations",
    "n_disconnected_lines",
]


def source_window_sha256(*, load_mw: list[float], pv_mw: list[float]) -> str:
    """Hash the exact source window consumed by the LV backend."""
    payload = {
        "load_mw": [round(float(value), 9) for value in load_mw],
        "pv_mw": [round(float(value), 9) for value in pv_mw],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ems_source_window_sha256(
    *,
    load_mw: list[float],
    pv_mw: list[float],
    wind_mw: list[float],
    price: list[float],
) -> str:
    """Hash the exact four-channel window consumed by the EMS backend."""
    payload = {
        key: [round(float(value), 9) for value in values]
        for key, values in {
            "load_mw": load_mw,
            "pv_mw": pv_mw,
            "wind_mw": wind_mw,
            "price": price,
        }.items()
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Load criticality classes (load_assignments analog)
# ─────────────────────────────────────────────────────────────────────────────


def _build_load_classes(critical_fraction: float) -> list[MicrogridLoad]:
    crit = max(0.0, min(0.8, critical_fraction))
    rest = 1.0 - crit
    spec = [
        ("hospital", crit * 0.6),
        ("water", crit * 0.4),
        ("data_center", rest * 0.2),
        ("commercial", rest * 0.3),
        ("residential", rest * 0.5),
    ]
    return [
        MicrogridLoad(
            load_id=f"load_{cls}",
            stakeholder_class=cls,  # type: ignore[arg-type]
            criticality=criticality_default(cls),  # type: ignore[arg-type]
            demand_fraction=round(frac, 6),
        )
        for cls, frac in spec
        if frac > 0
    ]


def _build_ders(der_count: int, *, with_wind: bool) -> list[dict[str, Any]]:
    ders: list[dict[str, Any]] = []
    for i in range(max(1, der_count)):
        kind = "wind" if (with_wind and i == der_count - 1) else "pv"
        ders.append({"der_id": f"der{i}", "kind": kind})
    return ders


# ─────────────────────────────────────────────────────────────────────────────
# pymgrid-family builders
# ─────────────────────────────────────────────────────────────────────────────


def _ems_common(
    *,
    family: str,
    backend_kind: str,
    seed_id: str | None,
    seed: int,
    difficulty_level: str,
    difficulty_mode: str,
    site: str | None,
    with_wind: bool,
    source_profile_start_index: int | None = None,
) -> tuple[MicrogridScenarioSeed, dict[str, Any]]:
    horizon = _HORIZON_BY_FAMILY[family]
    site = site or _SITES[seed % len(_SITES)]
    peak = float(BAKED_SITES.get(site, BAKED_SITES["denver_co"])["peak_load_mw"])
    start_index = int(source_profile_start_index or 0)

    # Structural seed parameters (§5).
    der_count = _BASE_DER.get(difficulty_level, 2) + (seed % 3)
    genset_available = bool(seed % 2)
    forecast_regime_idx = seed % len(FORECAST_REGIMES)
    critical_fraction = _CRITICAL_TIERS[seed % len(_CRITICAL_TIERS)]
    islanding_trigger = 2 + (seed % 4)

    overlay = load_overlay(
        site,
        horizon_ticks=horizon,
        seed=seed,
        forecast_regime_idx=forecast_regime_idx,
        pv_scale=1.0,
        wind_scale=0.4 if with_wind else 0.0,
        start_index=start_index,
    )

    load_assignments = _build_load_classes(critical_fraction)
    ders = _build_ders(der_count, with_wind=with_wind)

    backend_config: dict[str, Any] = {
        "site": site,
        "profiles": {
            "load_mw": overlay["load_mw"],
            "pv_mw": overlay["pv_mw"],
            "wind_mw": overlay["wind_mw"],
            "price": overlay["price"],
        },
        "battery": {
            "capacity_mwh": round(peak * 0.4, 3),
            "init_soc": 0.5,
            "max_charge_mw": round(peak * 0.25, 3),
            "max_discharge_mw": round(peak * 0.25, 3),
            "efficiency": 0.95,
        },
        "genset": {
            "min_mw": round(peak * 0.05, 3),
            "max_mw": round(peak * 0.5, 3),
            "fuel_cost_per_mwh": 120.0,
            "startup_cost": round(peak * 2.0, 3),
            "available": genset_available,
        },
        "grid": {
            "max_import_mw": round(peak * 1.2, 3),
            "max_export_mw": round(peak * 0.6, 3),
        },
        "ders": ders,
        "controllable_der_count": der_count,
        "genset_available": genset_available,
        "forecast_regime_idx": forecast_regime_idx,
        "forecast_error_sigma": overlay["forecast_sigma"],
        "forecast_bias": overlay["forecast_bias"],
        "islanding_trigger_tick": islanding_trigger,
        "hidden_parity": seed % 2,
        "cascade_permissive": difficulty_level == "cascading",
        # pymgrid is an aggregate EMS — the four power-flow keys are honest-0.
        "honest_zero_keys": list(_PYMGRID_HONEST_ZERO),
        "derivation_recipe": {
            "pipeline_version": "nrel-ems-window-v1",
            "profile_start_index": int(overlay.get("start_index", 0)),
            "source_window_sha256": ems_source_window_sha256(
                load_mw=overlay["load_mw"],
                pv_mw=overlay["pv_mw"],
                wind_mw=overlay["wind_mw"],
                price=overlay["price"],
            ),
        },
    }

    pm_ver = _pymgrid_version()
    lock = provenance_lock_kwargs("nsrdb", "oedi")
    anchored = site_is_anchored(site)
    provenance = Provenance(
        data_source="nsrdb+oedi",
        files=list(overlay["files"]),
        commit=lock["commit"],
        url=lock["url"],
        lock_strategy=_runtime_lock_signature(pymgrid_version=pm_ver),
        time_window={"horizon_ticks": horizon, "tick_minutes": 60},
        license=lock.get("license", "see spec §3"),
        notes=(
            f"Baked NSRDB solar + OEDI load overlay for site {site!r} "
            f"({'anchored' if anchored else 'offline-synthesized (network-blocked)'}); "
            "released pymgrid-family provenance is the baked NSRDB/OEDI series, "
            "NOT pymgrid's bundled/synthetic series (spec §3). pymgrid is used "
            "only on the optional dynamic-link cross-check path."
        ),
    )

    seed_obj = MicrogridScenarioSeed(
        seed_id=seed_id or f"{family}_{difficulty_mode}_{difficulty_level}_s{seed}",
        family=family,
        domain="microgrid",
        backend_kind=backend_kind,
        backend_config=backend_config,
        horizon_ticks=horizon,
        tick_minutes=60,
        seed=seed,
        load_assignments=load_assignments,
        perturbations=[],  # filled by caller
        dilemmas=[],
        difficulty_mode=difficulty_mode,  # type: ignore[arg-type]
        difficulty_level=difficulty_level,  # type: ignore[arg-type]
        provenance=provenance,
    )
    return seed_obj, backend_config


def build_microgrid_islanding_24h_seed(
    *,
    seed: int = 42,
    seed_id: str | None = None,
    difficulty_level: str = "basic",
    difficulty_mode: str = "time_pressure",
    site: str | None = None,
) -> MicrogridScenarioSeed:
    """mid-horizon grid outage → island; ride through on battery+genset,
    shed low-criticality to protect critical (spec §4)."""
    seed_obj, bc = _ems_common(
        family="microgrid_islanding_24h",
        backend_kind="pymgrid_islanding",
        seed_id=seed_id,
        seed=seed,
        difficulty_level=difficulty_level,
        difficulty_mode=difficulty_mode,
        site=site,
        with_wind=False,
    )
    trigger = int(bc["islanding_trigger_tick"])
    hidden = bool(seed % 2)
    n_stressors = _BASE_STRESSORS.get(difficulty_level, 1) + (seed % 2)
    island_dur = max(3, seed_obj.horizon_ticks // 3)
    pool = [
        Perturbation(
            kind="grid_outage",
            trigger_tick=trigger,
            duration_ticks=island_dur,
            hidden=False,
            target={},
            intensity=1.0,
            notes="PCC trips → island; ride-through on battery+genset.",
        ),
        Perturbation(
            kind="der_failure",
            trigger_tick=trigger + 1,
            duration_ticks=max(2, island_dur - 1),
            hidden=hidden,
            target={"der_index": seed % max(1, bc["controllable_der_count"])},
            intensity=1.0,
            notes="A controllable DER drops out mid-island.",
        ),
        Perturbation(
            kind="load_spike",
            trigger_tick=trigger + 2,
            duration_ticks=2,
            hidden=False,
            target={},
            intensity=round(0.2 + 0.1 * (seed % 3), 3),
            notes="Demand surge during the island window.",
        ),
    ]
    seed_obj.perturbations = pool[: max(1, n_stressors)]
    return seed_obj


def build_microgrid_economic_dispatch_24h_seed(
    *,
    seed: int = 42,
    seed_id: str | None = None,
    difficulty_level: str = "basic",
    difficulty_mode: str = "time_pressure",
    site: str | None = None,
    source_profile_start_index: int | None = None,
) -> MicrogridScenarioSeed:
    """time-varying import price + PV intermittency; battery arbitrage to
    minimize fuel+import (spec §4)."""
    seed_obj, bc = _ems_common(
        family="microgrid_economic_dispatch_24h",
        backend_kind="pymgrid_economic_dispatch",
        seed_id=seed_id,
        seed=seed,
        difficulty_level=difficulty_level,
        difficulty_mode=difficulty_mode,
        site=site,
        with_wind=False,
        source_profile_start_index=source_profile_start_index,
    )
    first = 2 + (seed % 4)
    n_stressors = _BASE_STRESSORS.get(difficulty_level, 1) + (seed % 2)
    pool = [
        Perturbation(
            kind="price_spike",
            trigger_tick=first,
            duration_ticks=max(2, seed_obj.horizon_ticks // 4),
            hidden=False,
            target={},
            intensity=round(0.5 + 0.25 * (seed % 4), 3),
            notes="Grid-import price spike (arbitrage pressure).",
        ),
        Perturbation(
            kind="pv_ramp",
            trigger_tick=first + 1,
            duration_ticks=2,
            hidden=False,
            target={},
            intensity=round(0.4 + 0.1 * (seed % 3), 3),
            notes="PV intermittency (cloud edge).",
        ),
        Perturbation(
            kind="forecast_bias",
            trigger_tick=0,
            duration_ticks=seed_obj.horizon_ticks,
            hidden=True,
            target={"bias_direction": "under-forecast"},
            intensity=round(0.05 + 0.05 * (seed % 3), 3),
            notes="Silent forecast bias (information_efficiency pressure).",
        ),
    ]
    seed_obj.perturbations = pool[: max(1, n_stressors)]
    return seed_obj


def build_microgrid_solar_ramp_24h_seed(
    *,
    seed: int = 42,
    seed_id: str | None = None,
    difficulty_level: str = "basic",
    difficulty_mode: str = "time_pressure",
    site: str | None = None,
) -> MicrogridScenarioSeed:
    """large PV ramp + forecast bias; pre-position SoC via noised forecast
    (spec §4)."""
    seed_obj, bc = _ems_common(
        family="microgrid_solar_ramp_24h",
        backend_kind="pymgrid_solar_ramp",
        seed_id=seed_id,
        seed=seed,
        difficulty_level=difficulty_level,
        difficulty_mode=difficulty_mode,
        site=site,
        with_wind=True,
    )
    first = 2 + (seed % 4)
    hidden = bool(seed % 2)
    n_stressors = _BASE_STRESSORS.get(difficulty_level, 1) + (seed % 2)
    pool = [
        Perturbation(
            kind="pv_ramp",
            trigger_tick=first,
            duration_ticks=max(2, seed_obj.horizon_ticks // 5),
            hidden=False,
            target={},
            intensity=round(0.2 + 0.1 * (seed % 4), 3),
            notes="Large PV ramp (forecast pre-positioning).",
        ),
        Perturbation(
            kind="forecast_bias",
            trigger_tick=0,
            duration_ticks=seed_obj.horizon_ticks,
            hidden=True,
            target={"bias_direction": "over-forecast"},
            intensity=round(0.08 + 0.04 * (seed % 3), 3),
            notes="Forecast bias on the renewable forecast.",
        ),
        Perturbation(
            kind="der_failure",
            trigger_tick=first + 2,
            duration_ticks=3,
            hidden=hidden,
            target={"der_index": seed % max(1, bc["controllable_der_count"])},
            intensity=1.0,
            notes="A DER trips during the ramp.",
        ),
    ]
    seed_obj.perturbations = pool[: max(1, n_stressors)]
    return seed_obj


# ─────────────────────────────────────────────────────────────────────────────
# LV power-flow family builder (pandapower; runs on a bare host today)
# ─────────────────────────────────────────────────────────────────────────────


def build_microgrid_lv_voltage_6h_seed(
    *,
    seed: int = 42,
    seed_id: str | None = None,
    difficulty_level: str = "basic",
    difficulty_mode: str = "time_pressure",
    site: str | None = None,
    source_profile_start_index: int | None = None,
    _staged_contract: bool = False,
    _recovery_contract: bool = False,
) -> MicrogridScenarioSeed:
    """rooftop-PV reverse power flow → over-voltage; hold the [0.95,1.05] pu
    band with battery + DER-Q + curtail (spec §4). Real AC power-flow keys
    via pandapower (in tree); no pymgrid required.
    """
    family = (
        "microgrid_lv_voltage_recovery_10h"
        if _recovery_contract
        else "microgrid_lv_voltage_staged_6h"
        if _staged_contract
        else "microgrid_lv_voltage_6h"
    )
    horizon = _HORIZON_BY_FAMILY[family]
    if _recovery_contract and difficulty_level != "extreme":
        raise ValueError("cross-tick recovery contract is extreme-only")
    if _recovery_contract and source_profile_start_index is None:
        raise ValueError("cross-tick recovery contract requires a locked source window")
    if _staged_contract and difficulty_level != "high":
        raise ValueError("staged recovery contract is high-only")
    if _staged_contract and source_profile_start_index is None:
        raise ValueError("staged recovery contract requires a locked source window")
    site = site or _SITES[seed % len(_SITES)]
    forecast_regime_idx = seed % len(FORECAST_REGIMES)
    critical_fraction = _CRITICAL_TIERS[seed % len(_CRITICAL_TIERS)]
    der_count = _BASE_DER.get(difficulty_level, 2) + (seed % 3)
    hidden = bool(seed % 2)
    first = 2 + (seed % 3)
    n_stressors = _BASE_STRESSORS.get(difficulty_level, 1) + (seed % 2)
    # Higher levels push more PV → harder over-voltage to hold.
    pv_scale = (
        5.0
        if _recovery_contract or _staged_contract
        else {"basic": 5.0, "medium": 6.5, "high": 8.0, "extreme": 10.0}.get(
            difficulty_level, 6.0
        )
        + 0.5 * (seed % 3)
    )

    load_assignments = _build_load_classes(critical_fraction)
    # Bind loads to nominal LV bus ids (adapter/backend re-map by index).
    for i, la in enumerate(load_assignments):
        la.bus_id = f"lv_bus_{i}"

    battery_capacity_mwh = (
        0.2
        if _recovery_contract
        else 0.1
        if (
            _staged_contract
            or (
                source_profile_start_index is not None
                and difficulty_level == "medium"
            )
        )
        else 0.05
    )
    battery_rate_mw = battery_capacity_mwh / 2.0
    backend_config: dict[str, Any] = {
        "site": site,
        "pv_scale": round(pv_scale, 3),
        "battery_e_mwh": battery_capacity_mwh,
        "controllable_der_count": der_count,
        "genset_available": False,
        "forecast_regime_idx": forecast_regime_idx,
        "forecast_error_sigma": FORECAST_REGIMES[forecast_regime_idx]["sigma"],
        "forecast_bias": FORECAST_REGIMES[forecast_regime_idx]["bias"],
        "cascade_permissive": False,
        # LV family fills the four power-flow keys honestly → no honest-0.
        "honest_zero_keys": ["startup_cost"],
    }
    if _recovery_contract:
        backend_config["task_contract"] = {
            "contract": "microgrid.lv_voltage.cross_tick_recovery.v2",
            "phase_ticks": [2, 7, 9],
            "minimum_reduction_each_phase": 1,
            "minimum_distinct_control_ticks": 3,
            "reversal": {
                "tool": "set_battery_dispatch",
                "argument": "p_mw",
                "first_sign": "positive",
                "later_sign": "negative",
                "later_not_before_tick": 7,
            },
        }
    elif _staged_contract:
        backend_config["task_contract"] = {
            "contract": "microgrid.lv_voltage.staged_recovery.v2",
            "phase_ticks": [2, 4],
            "minimum_reduction_each_phase": 1,
            "minimum_distinct_control_ticks": 2,
            "reversal": {
                "tool": "set_battery_dispatch",
                "argument": "p_mw",
                "first_sign": "positive",
                "later_sign": "negative",
                "later_not_before_tick": 3,
            },
        }
    if source_profile_start_index is not None:
        if not site_is_anchored(site, strict=True, min_horizon_ticks=horizon):
            raise ValueError(
                f"source-grounded LV candidate requires an anchored overlay: {site}"
            )
        overlay = load_overlay(
            site,
            horizon_ticks=horizon,
            seed=seed,
            forecast_regime_idx=forecast_regime_idx,
            start_index=source_profile_start_index,
        )
        load_profile = [float(value) for value in overlay["load_mw"]]
        pv_profile = [float(value) for value in overlay["pv_mw"]]
        load_reference = sum(load_profile) / len(load_profile)
        pv_reference = max(pv_profile)
        if load_reference <= 0 or pv_reference <= 0:
            raise ValueError(
                "source-grounded LV window requires positive load and PV references"
            )
        window_sha256 = source_window_sha256(
            load_mw=load_profile,
            pv_mw=pv_profile,
        )
        backend_config.update(
            {
                "source_profile_applied": True,
                "profile_start_index": int(source_profile_start_index),
                "source_profiles": {
                    "load_mw": load_profile,
                    "pv_mw": pv_profile,
                },
                "source_profile_reference": {
                    "load_mw": round(load_reference, 8),
                    "pv_mw": round(pv_reference, 8),
                },
                "derivation_recipe": {
                    "pipeline_version": "microgrid_source_consumed_v2",
                    "source_window_sha256": window_sha256,
                    "profile_start_index": int(source_profile_start_index),
                    "selection_rule": (
                        "rank_full_window_pv_variation_plus_load_variation"
                    ),
                    "network": (
                        "pandapower.create_synthetic_voltage_control_lv_network"
                    ),
                    "load_mapping": (
                        "native_load_p_mw * source_load_mw[t] / "
                        "mean(source_window_load_mw)"
                    ),
                    "pv_mapping": (
                        "native_sgen_p_mw * pv_scale * source_pv_mw[t] / "
                        "max(source_window_pv_mw)"
                    ),
                    "magnitude_scope": (
                        "source temporal shape mapped onto the native synthetic "
                        "LV feeder; source site MW magnitude is not claimed as "
                        "the feeder magnitude"
                    ),
                },
                "battery": {
                    "capacity_mwh": battery_capacity_mwh,
                    "init_soc": 0.5,
                    "max_charge_mw": battery_rate_mw,
                    "max_discharge_mw": battery_rate_mw,
                    "efficiency": 0.95,
                },
            }
        )

    pool = [
        Perturbation(
            kind="pv_ramp",
            trigger_tick=first,
            duration_ticks=max(2, horizon // 2),
            hidden=False,
            target={},
            intensity=round(1.0 + 0.2 * (seed % 3), 3),
            notes="Rooftop-PV ramp → reverse flow / over-voltage.",
        ),
        Perturbation(
            kind="der_failure",
            trigger_tick=first + 1,
            duration_ticks=2,
            hidden=hidden,
            target={"der_index": seed % max(1, der_count)},
            intensity=1.0,
            notes="A rooftop-PV string drops out.",
        ),
        Perturbation(
            kind="load_spike",
            trigger_tick=min(horizon - 1, first + 2),
            duration_ticks=1,
            hidden=False,
            target={},
            intensity=round(0.15 + 0.1 * (seed % 3), 3),
            notes="Brief demand swing.",
        ),
        Perturbation(
            kind="der_failure",
            trigger_tick=min(horizon - 1, first + 3),
            duration_ticks=1,
            hidden=True,
            target={"der_index": (seed + 1) % max(1, der_count)},
            intensity=1.0,
            notes="A second DER trips late, invalidating the initial recovery plan.",
        ),
    ]
    if source_profile_start_index is not None:
        # Leave one full decision tick after every source-grounded shock.
        # The legacy schedule placed the fourth extreme event on the terminal
        # tick, where no agent could observe and respond to it.
        source_trigger_ticks = (1, 2, 3, 4)
        for perturbation, trigger_tick in zip(pool, source_trigger_ticks, strict=False):
            perturbation.trigger_tick = trigger_tick
        pool[0].duration_ticks = 3
        pool[0].intensity = max(1.15, pool[0].intensity)
    selected_perturbations = pool[: max(1, n_stressors)]
    if source_profile_start_index is not None:
        recipe = backend_config["derivation_recipe"]
        recipe["stress_overlay_scope"] = (
            "deterministic simulator stressors declared by kind, tick, duration, "
            "target, and intensity; not claimed as source-window observations"
        )
        recipe["stress_overlays"] = [
            {
                "kind": perturbation.kind,
                "trigger_tick": perturbation.trigger_tick,
                "duration_ticks": perturbation.duration_ticks,
                "hidden": perturbation.hidden,
                "target": perturbation.target,
                "intensity": perturbation.intensity,
            }
            for perturbation in selected_perturbations
        ]
    if _recovery_contract:
        selected_perturbations = [
            Perturbation(
                kind="pv_ramp",
                trigger_tick=2,
                duration_ticks=2,
                hidden=False,
                target={},
                intensity=1.5,
                notes="Visible PV surge requires storage pre-positioning.",
            ),
            Perturbation(
                kind="load_spike",
                trigger_tick=4,
                duration_ticks=6,
                hidden=False,
                target={},
                intensity=5.0,
                notes=(
                    "Sustained demand reversal invalidates the over-voltage plan "
                    "and requires reactive support."
                ),
            ),
            Perturbation(
                kind="der_failure",
                trigger_tick=6,
                duration_ticks=4,
                hidden=True,
                target={"der_index": 2},
                intensity=1.0,
                notes=(
                    "A hidden DER trip requires late storage direction reversal."
                ),
            ),
        ]
        recipe = backend_config["derivation_recipe"]
        recipe["selection_rule"] = "behavioral_cross_tick_recovery_scan_v1"
        recipe["stress_overlays"] = [
            {
                "kind": perturbation.kind,
                "trigger_tick": perturbation.trigger_tick,
                "duration_ticks": perturbation.duration_ticks,
                "hidden": perturbation.hidden,
                "target": perturbation.target,
                "intensity": perturbation.intensity,
            }
            for perturbation in selected_perturbations
        ]
    elif _staged_contract:
        selected_perturbations = [
            Perturbation(
                kind="pv_ramp",
                trigger_tick=1,
                duration_ticks=2,
                hidden=False,
                target={},
                intensity=1.5,
                notes="Visible PV surge requires absorptive controls.",
            ),
            Perturbation(
                kind="load_spike",
                trigger_tick=3,
                duration_ticks=3,
                hidden=True,
                target={},
                intensity=4.0,
                notes=(
                    "A latent sustained demand reversal invalidates the "
                    "initial absorptive plan and must be inferred from native "
                    "voltage observations."
                ),
            ),
        ]
        recipe = backend_config["derivation_recipe"]
        recipe["selection_rule"] = "behavioral_two_stage_recovery_scan_v1"
        recipe["stress_overlays"] = [
            {
                "kind": perturbation.kind,
                "trigger_tick": perturbation.trigger_tick,
                "duration_ticks": perturbation.duration_ticks,
                "hidden": perturbation.hidden,
                "target": perturbation.target,
                "intensity": perturbation.intensity,
            }
            for perturbation in selected_perturbations
        ]

    lock = provenance_lock_kwargs("nsrdb")
    provenance = Provenance(
        data_source="nsrdb_rooftop",
        files=baked_overlay_provenance_files(site),
        commit=lock["commit"],
        url=lock["url"],
        lock_strategy=_runtime_lock_signature(pymgrid_version=_pymgrid_version()),
        time_window={
            "horizon_ticks": horizon,
            "tick_minutes": 60,
            **(
                {"start_index": int(source_profile_start_index)}
                if source_profile_start_index is not None
                else {}
            ),
            **(
                {"source_window_sha256": window_sha256}
                if source_profile_start_index is not None
                else {}
            ),
        },
        license=lock.get("license", "see spec §3"),
        notes=(
            "LV power-flow tier on pandapower's synthetic voltage-control LV "
            "network (BSD-3-Clause, in tree); rooftop PV scaled from baked "
            "NSRDB irradiance. For source-grounded candidates, the locked "
            "load/PV temporal shape is normalized onto the native feeder; "
            "backend_config.derivation_recipe records the exact mapping and "
            "does not claim the source site's MW magnitude as feeder magnitude. "
            "Fills the four AC power-flow keys honestly; no pymgrid required."
        ),
    )

    return MicrogridScenarioSeed(
        seed_id=seed_id or f"{family}_{difficulty_mode}_{difficulty_level}_s{seed}",
        family=family,
        domain="microgrid",
        backend_kind="pandapower_lv",
        backend_config=backend_config,
        horizon_ticks=horizon,
        tick_minutes=60,
        seed=seed,
        load_assignments=load_assignments,
        perturbations=selected_perturbations,
        dilemmas=[],
        difficulty_mode=difficulty_mode,  # type: ignore[arg-type]
        difficulty_level=difficulty_level,  # type: ignore[arg-type]
        provenance=provenance,
    )


def build_microgrid_lv_voltage_recovery_10h_seed(
    *,
    seed: int = 42,
    seed_id: str | None = None,
    difficulty_mode: str = "time_pressure",
    site: str,
    source_profile_start_index: int,
) -> MicrogridScenarioSeed:
    """Build the source-grounded Extreme LV cross-tick recovery task."""
    return build_microgrid_lv_voltage_6h_seed(
        seed=seed,
        seed_id=seed_id,
        difficulty_level="extreme",
        difficulty_mode=difficulty_mode,
        site=site,
        source_profile_start_index=source_profile_start_index,
        _recovery_contract=True,
    )


def build_microgrid_lv_voltage_staged_6h_seed(
    *,
    seed: int = 42,
    seed_id: str | None = None,
    difficulty_mode: str = "time_pressure",
    site: str,
    source_profile_start_index: int,
) -> MicrogridScenarioSeed:
    """Build the source-grounded High two-stage LV recovery task."""
    return build_microgrid_lv_voltage_6h_seed(
        seed=seed,
        seed_id=seed_id,
        difficulty_level="high",
        difficulty_mode=difficulty_mode,
        site=site,
        source_profile_start_index=source_profile_start_index,
        _staged_contract=True,
    )
