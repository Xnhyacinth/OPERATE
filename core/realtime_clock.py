"""Per-scenario native wall clock for ``realtime_persistent``.

Formal v0.62 used a uniform 5s stress overlay. Live policy ``native_dt_v1``
binds the wall tick to the scenario's source-converted plant quantum
(``tick_seconds`` or ``tick_minutes``) so thinking time is scored against
the same decision cadence the data source actually has.

Hour-scale and multi-hour plant walls stay out of the default speed
scorecard: they are not 1:1 operator latency tests and must not be
compressed into 5s ticks.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

POLICY_VERSION = "native_dt_v1"
SPEED_CRITICAL_MAX_NATIVE_S = 60.0
MAX_EPISODE_PLANT_WALL_S = 1800.0
SELECTION_BINDING = "native_dt_speed_critical_v1"
CORE_SPEED_SCORECARD_N = 37

NATIVE_DT_CLOCK_PROFILE: dict[str, Any] = {
    "kind": "soft_realtime_monotonic_single_writer",
    "tick_interval_policy": POLICY_VERSION,
    "speed_critical_max_native_s": SPEED_CRITICAL_MAX_NATIVE_S,
    "max_episode_plant_wall_s": MAX_EPISODE_PLANT_WALL_S,
    "episode_timeout_policy": (
        "horizon_ticks_x_tick_plus_provider_timeout_plus_tick"
    ),
    "process_hard_timeout_overhead_s": 30.0,
    "termination_grace_s": 5.0,
}


def native_seconds_per_tick(scenario: Mapping[str, Any]) -> float:
    """Return the source-converted plant seconds represented by one tick."""

    tick_seconds = scenario.get("tick_seconds")
    tick_minutes = scenario.get("tick_minutes")
    if tick_seconds is not None and tick_minutes is not None:
        raise ValueError("scenario must not declare both tick_seconds and tick_minutes")
    if tick_seconds is not None:
        if isinstance(tick_seconds, bool) or not isinstance(tick_seconds, (int, float)):
            raise ValueError("tick_seconds must be a finite positive number")
        native = float(tick_seconds)
    elif tick_minutes is not None:
        if isinstance(tick_minutes, bool) or not isinstance(tick_minutes, (int, float)):
            raise ValueError("tick_minutes must be a finite positive number")
        native = float(tick_minutes) * 60.0
    else:
        raise ValueError("scenario is missing tick_seconds or tick_minutes")
    if not math.isfinite(native) or native <= 0:
        raise ValueError("native seconds per tick must be finite and positive")
    return native


def classify_realtime_clock(
    scenario: Mapping[str, Any],
    *,
    horizon_ticks: int | None = None,
) -> dict[str, Any]:
    """Classify whether a row is a 1:1 speed-critical realtime cell."""

    native = native_seconds_per_tick(scenario)
    raw_horizon = (
        horizon_ticks if horizon_ticks is not None else scenario.get("horizon_ticks")
    )
    if (
        isinstance(raw_horizon, bool)
        or not isinstance(raw_horizon, int)
        or raw_horizon < 1
    ):
        raise ValueError("horizon_ticks must be a positive integer")
    plant_wall_s = float(raw_horizon) * native
    speed_critical = native <= SPEED_CRITICAL_MAX_NATIVE_S
    wall_tractable = speed_critical and plant_wall_s <= MAX_EPISODE_PLANT_WALL_S
    if wall_tractable:
        reason = "native_one_to_one_speed_critical"
        fidelity = "native_1to1"
    elif speed_critical:
        reason = "episode_plant_wall_exceeds_scorecard_budget"
        fidelity = "native_1to1_unbounded"
    else:
        reason = "native_tick_exceeds_operator_response_window"
        fidelity = "not_speed_critical"
    return {
        "policy": POLICY_VERSION,
        "native_seconds_per_tick": native,
        "wall_tick_interval_s": native,
        "horizon_ticks": raw_horizon,
        "plant_wall_s": plant_wall_s,
        "speed_critical": speed_critical,
        "wall_tractable": wall_tractable,
        "scorecard_eligible": wall_tractable,
        "fidelity": fidelity,
        "reason": reason,
    }


def default_formal_clock_profile() -> dict[str, Any]:
    return dict(NATIVE_DT_CLOCK_PROFILE)
