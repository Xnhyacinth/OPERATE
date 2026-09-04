"""
domains.power_grid.seeds.from_rts_real_ts — Real RTS-GMLC timeseries
enrichment for forecast-bias perturbations (v0.2).

The RTS-GMLC repository under
``benchmark/works/RTS-GMLC/RTS_Data/timeseries_data_files/`` ships
**both** Day-Ahead (forecast) and Real-Time (actual) hourly series for
load and renewable generation. Subtracting one from the other yields a
*real* forecast-error profile that we can replay as a deterministic
forecast_bias perturbation.

This module provides utilities to:

- Load DA and RT load time series for a given calendar window.
- Compute the per-tick fractional forecast error.
- Construct a synthetic `forecast_bias` perturbation schedule whose
  per-tick intensity matches the historical DA→RT discrepancy (rather
  than a hard-coded 0.10 used by the v0.1.x pglib-only factories).

The seed factory ``build_daily_ops_real_forecast_seed`` then creates a
new family ``daily_ops_real_forecast_24h`` that reuses the pglib_uc
synthetic backend but injects this historical forecast error.
"""

from __future__ import annotations

import contextlib
import csv
from pathlib import Path

from ..source_paths import source_ref
from .from_pglib_uc import (
    PGLIB_UC_ROOT,
    load_case,
    split_load_into_stakeholders,
)
from .schema import (
    DilemmaSeed,
    Perturbation,
    Provenance,
    ScenarioSeed,
)
from .source_locks import provenance_lock_kwargs

# Resolve RTS-GMLC clone relative to PGLIB_UC_ROOT for consistency.
RTS_GMLC_ROOT = PGLIB_UC_ROOT.parent / "RTS-GMLC"
RTS_TS_ROOT = RTS_GMLC_ROOT / "RTS_Data" / "timeseries_data_files"

REGIONAL_LOAD_DA = RTS_TS_ROOT / "Load" / "DAY_AHEAD_regional_Load.csv"
REGIONAL_LOAD_RT = RTS_TS_ROOT / "Load" / "REAL_TIME_regional_Load.csv"


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def real_forecast_error_profile(
    year: int,
    month: int,
    day: int,
    hours: int = 24,
) -> list[float]:
    """Return per-hour fractional forecast error from RTS-GMLC.

    A value of +0.1 at hour t means the DA forecast UNDER-predicted the
    actual demand by 10% in that hour. The series is signed so the
    scenario factory can place either positive or negative bias.

    Returns an empty list if the underlying CSVs are missing.
    """
    da_rows = _load_csv_rows(REGIONAL_LOAD_DA)
    rt_rows = _load_csv_rows(REGIONAL_LOAD_RT)
    if not da_rows or not rt_rows:
        return []

    def _aggregate(
        rows: list[dict[str, str]],
    ) -> dict[tuple[int, int, int, int], float]:
        out: dict[tuple[int, int, int, int], float] = {}
        for r in rows:
            try:
                y = int(r["Year"])
                m = int(r["Month"])
                d = int(r["Day"])
                p = int(r["Period"])
            except (KeyError, ValueError):
                continue
            if y != year or m != month or d != day:
                continue
            total = 0.0
            for k, v in r.items():
                if k in {"Year", "Month", "Day", "Period"}:
                    continue
                with contextlib.suppress(TypeError, ValueError):
                    total += float(v)
            out[(y, m, d, p)] = total
        return out

    da = _aggregate(da_rows)
    rt = _aggregate(rt_rows)
    if not da or not rt:
        return []
    profile: list[float] = []
    for h in range(1, hours + 1):
        key = (year, month, day, h)
        da_v = da.get(key)
        rt_v = rt.get(key)
        if da_v is None or rt_v is None or da_v <= 0:
            profile.append(0.0)
            continue
        # Signed fractional under-forecast: (actual - forecast) / forecast
        profile.append((rt_v - da_v) / da_v)
    return profile


def build_daily_ops_real_forecast_seed(
    case_path: Path,
    *,
    seed_id: str,
    seed: int = 42,
    difficulty_mode: str = "time_pressure",
    difficulty_level: str = "basic",
) -> ScenarioSeed:
    """Build a daily_ops_real_forecast_24h seed.

    Forecast bias is sourced from RTS-GMLC's real DA→RT error series for
    a hardcoded date (2020-01-27, matching the pglib-uc rts_gmlc case).
    Other perturbations follow the same ladder as daily_ops_24h.
    """
    case = load_case(case_path)
    horizon_ticks = 24 if difficulty_mode == "time_pressure" else 36
    horizon_ticks = min(horizon_ticks, int(case.get("time_periods", 48)))
    demand = case.get("demand", [])
    initial_demand = float(demand[0]) if demand else 1000.0

    # Pull the real forecast-error profile aligned to the case's calendar
    # day (the pglib-uc rts_gmlc filenames embed the date).
    try:
        # case_path.stem like "2020-01-27"
        parts = case_path.stem.split("-")
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        y, m, d = 2020, 1, 27
    error_profile = real_forecast_error_profile(y, m, d, hours=horizon_ticks)
    # v0.2.1: keep the FULL per-tick profile so the backend can apply
    # real DA→RT error at each tick instead of a single averaged scalar
    # — that was the whole point of this family. The scalar avg is
    # retained for backwards-compatible summary in `intensity`.
    if error_profile:
        avg_abs = sum(abs(e) for e in error_profile) / len(error_profile)
        net = sum(error_profile) / len(error_profile)
        bias_direction = "under-forecast" if net > 0 else "over-forecast"
        bias_intensity = round(max(0.02, avg_abs), 4)
        per_tick_profile = [round(float(e), 4) for e in error_profile]
    else:
        bias_direction = "under-forecast"
        bias_intensity = 0.05
        per_tick_profile = []

    pert_density = {"basic": 1, "medium": 2, "high": 3, "extreme": 4}.get(
        difficulty_level, 1
    )
    perturbations: list[Perturbation] = []
    perturbations.append(
        Perturbation(
            kind="planned_maintenance",
            trigger_tick=4,
            duration_ticks=2,
            target={"generator_kind": "thermal", "fraction": 0.05},
            notes="Scheduled morning maintenance window.",
        )
    )
    for k in range(pert_density):
        perturbations.append(
            Perturbation(
                kind="load_surge",
                trigger_tick=6 + 3 * k,
                duration_ticks=2,
                intensity=0.05 + 0.05 * k,
                target={"stakeholder_class": "residential"},
                notes=f"Load surge wave #{k + 1}.",
            )
        )
    # ALWAYS attach the real forecast-bias perturbation — this is the
    # whole point of this family. v0.2.1: the perturbation carries the
    # FULL per-tick error profile in `target["per_tick_profile"]`. The
    # backend's `forecast_for()` (pglib_uc_synthetic.py) consumes this
    # when present, falling back to the scalar `intensity` otherwise.
    perturbations.append(
        Perturbation(
            kind="forecast_bias",
            trigger_tick=0,
            duration_ticks=horizon_ticks,
            intensity=bias_intensity,
            target={
                "bias_direction": bias_direction,
                "source": "RTS-GMLC DA→RT actuals",
                "calendar_day": f"{y:04d}-{m:02d}-{d:02d}",
                "per_tick_profile": per_tick_profile,
            },
            notes=(
                f"Real forecast bias from RTS-GMLC DA→RT errors for "
                f"{y:04d}-{m:02d}-{d:02d}: per-tick profile "
                f"(len={len(per_tick_profile)}), avg_abs_err="
                f"{bias_intensity:.3f}, net_direction={bias_direction}."
            ),
        )
    )

    dilemmas: list[DilemmaSeed] = []
    if difficulty_level in {"medium", "high", "extreme"}:
        dilemmas.append(
            DilemmaSeed(
                dilemma_id="d_evening_peak",
                trigger_tick=horizon_ticks // 2,
                description=(
                    "Real-time demand has overshot the day-ahead forecast. "
                    "Choose which load to curtail to maintain reserves."
                ),
                options=[
                    {
                        "option_id": "shed_residential",
                        "label": "Shed residential",
                        "fatal": False,
                        "expected_consequences": {"economic": "low"},
                    },
                    {
                        "option_id": "shed_industrial",
                        "label": "Shed industrial",
                        "fatal": False,
                        "expected_consequences": {"economic": "high"},
                    },
                    {
                        "option_id": "shed_hospital",
                        "label": "Shed hospital",
                        "fatal": True,
                        "expected_consequences": {"casualties": "high"},
                    },
                ],
                expected_tradeoff_tokens=["forecast", "reserve", "real-time"],
                expected_stakeholder_tokens=["residential", "industrial", "hospital"],
                resolution_deadline_ticks=(
                    1 if difficulty_mode == "time_pressure" else 3
                ),
                default_option_id="shed_residential",
            )
        )

    backend_config = {
        "case_file": source_ref(case_path),
        "first_period": 1,
        "last_period": horizon_ticks,
        "initial_demand_mw": initial_demand,
        "real_forecast_error_avg_abs": bias_intensity,
        "real_forecast_error_direction": bias_direction,
        "real_forecast_calendar_day": f"{y:04d}-{m:02d}-{d:02d}",
        "real_forecast_profile_len": len(per_tick_profile),
    }

    provenance = Provenance(
        data_source="pglib_uc+rts_gmlc_timeseries",
        files=[
            source_ref(case_path),
            source_ref(REGIONAL_LOAD_DA),
            source_ref(REGIONAL_LOAD_RT),
        ],
        **provenance_lock_kwargs("pglib_uc", "rts_gmlc"),
        time_window={"hours": horizon_ticks, "tick_minutes": 60},
        license=("CC-BY-4.0 (pglib-uc); NREL/DOE attribution (RTS-GMLC timeseries)"),
        notes=(
            "Forecast bias intensity is derived from RTS-GMLC's published "
            "DA→RT forecast error for the matching calendar day. This is "
            "the v0.2 alternative to hand-tuned synthetic 0.10 biases."
        ),
    )

    return ScenarioSeed(
        seed_id=seed_id,
        family="daily_ops_real_forecast_24h",
        domain="power_grid",
        backend_kind="pglib_uc_synthetic",
        backend_config=backend_config,
        horizon_ticks=horizon_ticks,
        tick_minutes=60,
        seed=seed,
        load_assignments=split_load_into_stakeholders(initial_demand, case_path.stem),
        perturbations=perturbations,
        dilemmas=dilemmas,
        difficulty_mode=difficulty_mode,  # type: ignore[arg-type]
        difficulty_level=difficulty_level,  # type: ignore[arg-type]
        provenance=provenance,
    )


def list_rts_timeseries_available_dates() -> list[tuple[int, int, int]]:
    """Return unique (year, month, day) tuples for which both DA and RT
    load series are present."""
    da_rows = _load_csv_rows(REGIONAL_LOAD_DA)
    if not da_rows:
        return []
    dates: set[tuple[int, int, int]] = set()
    for r in da_rows:
        with contextlib.suppress(KeyError, ValueError):
            dates.add((int(r["Year"]), int(r["Month"]), int(r["Day"])))
    return sorted(dates)
