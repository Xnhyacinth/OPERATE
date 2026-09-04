"""RTS-GMLC regional-load helpers for PGLib-OPF pairing.

Case73 keeps an exact bus/branch identity check. Smaller PGLib cases may
consume only the regional load trajectory as a synthetic topology-profile
composition: hourly mean of 12 five-minute values, scaled by the RTS
declared base MW onto the PGLib native load vector. That is not a
historical pairing and does not claim a shared generator fleet.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

LOAD_CSV = (
    "works/RTS-GMLC/RTS_Data/timeseries_data_files/Load/"
    "REAL_TIME_regional_Load.csv"
)
BUS_CSV = "works/RTS-GMLC/RTS_Data/SourceData/bus.csv"
BRANCH_CSV = "works/RTS-GMLC/RTS_Data/SourceData/branch.csv"
RTS_GMLC_COMMIT = "3ece0d3725c844056132393ee252b3083dd4eab4"
EXCLUDED_DATES = frozenset({"2020-07-20"})
WINTER_MONTHS = frozenset({12, 1, 2})
CASE73_CONTRACT = "rts_gmlc_case73_regional_load_v1"
SYNTHETIC_CONTRACT = "rts_gmlc_pglib_opf_synthetic_load_profile_v1"
SYNTHETIC_CASES = frozenset({"pglib_opf_case14_ieee", "pglib_opf_case30_ieee"})
CASE73_CASE = "pglib_opf_case73_ieee_rts"
WINDOW_RECIPE = "rts_gmlc_hourly_mean_of_12_five_minute_periods_v1"
REGION_COLUMNS = ("1", "2", "3")


def parse_calendar_date(raw: str) -> date:
    try:
        year, month, day = (int(part) for part in str(raw).split("-"))
        return date(year, month, day)
    except (TypeError, ValueError) as exc:
        raise ValueError("paired_timeseries calendar_date must be YYYY-MM-DD") from exc


def region_base_mw(bus_rows: list[dict[str, str]]) -> dict[int, float]:
    totals: dict[int, float] = {}
    for row in bus_rows:
        region = int(row["Area"])
        totals[region] = totals.get(region, 0.0) + float(row["MW Load"])
    return totals


def load_day_rows(path: Path, calendar_day: date) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if (int(row["Year"]), int(row["Month"]), int(row["Day"]))
            == (calendar_day.year, calendar_day.month, calendar_day.day)
        ]


def select_horizon_rows(
    rows: list[dict[str, str]],
    *,
    horizon_ticks: int,
    periods_per_tick: int,
) -> list[dict[str, str]]:
    if periods_per_tick <= 0:
        raise ValueError("paired_timeseries periods_per_tick must be positive")
    expected_periods = horizon_ticks * periods_per_tick
    if len(rows) < expected_periods:
        raise ValueError(
            "paired RTS-GMLC load window is shorter than the scenario horizon"
        )
    selected = rows[:expected_periods]
    expected_ids = list(range(1, expected_periods + 1))
    if [int(row["Period"]) for row in selected] != expected_ids:
        raise ValueError("paired RTS-GMLC load periods are not contiguous")
    return selected


def window_sha256(
    rows: list[dict[str, str]],
    *,
    calendar_date: str,
    periods_per_tick: int,
) -> str:
    payload = {
        "calendar_date": calendar_date,
        "periods_per_tick": periods_per_tick,
        "recipe_version": WINDOW_RECIPE,
        "periods": [
            {
                "Period": int(row["Period"]),
                "1": round(float(row["1"]), 9),
                "2": round(float(row["2"]), 9),
                "3": round(float(row["3"]), 9),
            }
            for row in rows
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def hourly_region_means(
    rows: list[dict[str, str]],
    *,
    periods_per_tick: int,
    regions: dict[int, float],
) -> list[dict[int, float]]:
    hours: list[dict[int, float]] = []
    for offset in range(0, len(rows), periods_per_tick):
        window = rows[offset : offset + periods_per_tick]
        hours.append(
            {
                region: sum(float(row[str(region)]) for row in window)
                / periods_per_tick
                for region in regions
            }
        )
    return hours


def synthetic_scaled_profile(
    *,
    base_load_p_mw: dict[int, float],
    hourly_region_mw: list[dict[int, float]],
    region_base: dict[int, float],
) -> list[dict[int, float]]:
    total_base = sum(region_base.values())
    if total_base <= 0:
        raise ValueError("RTS-GMLC declared base load must be positive")
    profile: list[dict[int, float]] = []
    for hour in hourly_region_mw:
        scale = sum(hour.values()) / total_base
        profile.append(
            {idx: base_mw * scale for idx, base_mw in base_load_p_mw.items()}
        )
    return profile


def mine_stress_days(
    load_csv: Path,
    *,
    exclude: frozenset[str] = EXCLUDED_DATES,
) -> dict[str, Any]:
    """Rank 2020 days by peak hourly total and hour-to-hour ramp."""

    days: dict[date, list[tuple[int, float]]] = defaultdict(list)
    with load_csv.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = date(int(row["Year"]), int(row["Month"]), int(row["Day"]))
            total = sum(float(row[column]) for column in REGION_COLUMNS)
            days[key].append((int(row["Period"]), total))

    ranked: list[dict[str, Any]] = []
    for day, rows in sorted(days.items()):
        rows = sorted(rows, key=lambda item: item[0])[:288]
        if len(rows) < 288:
            continue
        hourly = [
            sum(value for _, value in rows[offset : offset + 12]) / 12.0
            for offset in range(0, 288, 12)
        ]
        ramps = [abs(hourly[idx] - hourly[idx - 1]) for idx in range(1, len(hourly))]
        iso = day.isoformat()
        ranked.append(
            {
                "calendar_date": iso,
                "month": day.month,
                "winter": day.month in WINTER_MONTHS,
                "excluded": iso in exclude,
                "peak_mw": round(max(hourly), 6),
                "mean_mw": round(sum(hourly) / len(hourly), 6),
                "max_hour_to_hour_ramp_mw": round(max(ramps), 6),
            }
        )

    eligible = [row for row in ranked if not row["excluded"]]
    winter = [row for row in eligible if row["winter"]]
    by_peak = sorted(winter, key=lambda row: -float(row["peak_mw"]))
    by_ramp = sorted(winter, key=lambda row: -float(row["max_hour_to_hour_ramp_mw"]))
    selected: list[dict[str, Any]] = []
    if by_ramp:
        selected.append({**by_ramp[0], "selection_reason": "highest_winter_hour_to_hour_ramp"})
    if by_peak:
        peak_row = by_peak[0]
        if all(row["calendar_date"] != peak_row["calendar_date"] for row in selected):
            selected.append({**peak_row, "selection_reason": "highest_winter_peak_total_load"})
        elif len(by_peak) > 1:
            selected.append(
                {
                    **by_peak[1],
                    "selection_reason": "second_highest_winter_peak_total_load",
                }
            )
    return {
        "n_days": len(ranked),
        "excluded": sorted(exclude),
        "selected": selected,
        "top_winter_peak": by_peak[:5],
        "top_winter_ramp": by_ramp[:5],
        "top_overall_peak": sorted(
            eligible, key=lambda row: -float(row["peak_mw"])
        )[:5],
    }
