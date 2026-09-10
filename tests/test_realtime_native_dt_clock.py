from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from core.realtime_clock import (
    CORE_SPEED_SCORECARD_N,
    MAX_EPISODE_PLANT_WALL_S,
    NATIVE_DT_CLOCK_PROFILE,
    POLICY_VERSION,
    SPEED_CRITICAL_MAX_NATIVE_S,
    classify_realtime_clock,
    native_seconds_per_tick,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _catalog_dir() -> Path:
    private = REPO_ROOT / "release" / "operate_v0_62_0"
    if (private / "core_suite.json").is_file():
        return private
    return REPO_ROOT / "benchmark"


def test_native_seconds_prefers_tick_seconds() -> None:
    assert native_seconds_per_tick({"tick_seconds": 5.0, "horizon_ticks": 18}) == 5.0


def test_native_seconds_from_tick_minutes() -> None:
    assert native_seconds_per_tick({"tick_minutes": 1, "horizon_ticks": 12}) == 60.0


def test_native_seconds_rejects_both_fields() -> None:
    with pytest.raises(ValueError, match="both"):
        native_seconds_per_tick({"tick_seconds": 5, "tick_minutes": 1})


def test_driving_is_scorecard_eligible() -> None:
    clock = classify_realtime_clock(
        {"tick_seconds": 5.0, "horizon_ticks": 18},
        horizon_ticks=18,
    )
    assert clock["scorecard_eligible"] is True
    assert clock["wall_tick_interval_s"] == 5.0
    assert clock["fidelity"] == "native_1to1"


def test_hour_scale_citylearn_is_not_speed_critical() -> None:
    clock = classify_realtime_clock(
        {"tick_minutes": 60, "horizon_ticks": 72},
        horizon_ticks=72,
    )
    assert clock["speed_critical"] is False
    assert clock["scorecard_eligible"] is False
    assert clock["reason"] == "native_tick_exceeds_operator_response_window"


def test_long_jobshop_minute_ticks_exceed_wall_budget() -> None:
    clock = classify_realtime_clock(
        {"tick_minutes": 1, "horizon_ticks": 592},
        horizon_ticks=592,
    )
    assert clock["speed_critical"] is True
    assert clock["scorecard_eligible"] is False
    assert clock["reason"] == "episode_plant_wall_exceeds_scorecard_budget"
    assert clock["plant_wall_s"] > MAX_EPISODE_PLANT_WALL_S


def test_frozen_core_speed_suite_matches_live_filter() -> None:
    catalog = _catalog_dir()
    core = json.loads((catalog / "core_suite.json").read_text(encoding="utf-8"))
    frozen = json.loads(
        (catalog / "realtime_speed_suite.json").read_text(encoding="utf-8")
    )
    selected = []
    for row in core["scenarios"]:
        scenario = yaml.safe_load((REPO_ROOT / row["path"]).read_text(encoding="utf-8"))
        clock = classify_realtime_clock(
            scenario, horizon_ticks=int(row["horizon_ticks"])
        )
        if clock["scorecard_eligible"]:
            selected.append(row["scenario_id"])
    assert frozen["n_scenarios"] == CORE_SPEED_SCORECARD_N
    assert len(selected) == CORE_SPEED_SCORECARD_N
    assert selected == [row["scenario_id"] for row in frozen["scenarios"]]
    assert frozen["clock_policy"] == POLICY_VERSION
    assert frozen["speed_critical_max_native_s"] == SPEED_CRITICAL_MAX_NATIVE_S
    assert NATIVE_DT_CLOCK_PROFILE["tick_interval_policy"] == POLICY_VERSION
