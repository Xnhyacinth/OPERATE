"""Pure tactical helpers for autonomous-driving greedy and oracle baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import Action, ToolCall


def _available(tool_specs: list[dict[str, Any]]) -> set[str]:
    return {
        str((row.get("function") or {}).get("name") or "")
        for row in tool_specs
        if isinstance(row, dict)
    }


def _call(name: str, args: dict[str, Any], *, tick: int, label: str) -> Action:
    return Action(
        tool_calls=[
            ToolCall(
                name=name,
                args=args,
                idempotency_key=f"{label}-{name}-t{tick}",
            )
        ],
        dominant=name,
    )


def _with_initial_safety_investigation(
    action: Action,
    available: set[str],
    *,
    tick: int,
    label: str,
) -> Action:
    """Make reference policies satisfy the scenario's observable safety contract."""
    if tick != 0:
        return action
    inspection_calls = [
        ToolCall(
            name=name,
            args={},
            idempotency_key=f"{label}-{name}-t{tick}",
        )
        for name in (
            "inspect_local_scene",
            "inspect_safety_state",
            "inspect_odd_status",
        )
        if name in available
    ]
    if not inspection_calls:
        return action
    return Action(
        tool_calls=[*inspection_calls, *action.tool_calls],
        dominant=action.dominant,
    )


def greedy_action(
    observation: dict[str, Any],
    tool_specs: list[dict[str, Any]],
) -> Action:
    """Choose a conservative tactical response from visible safety state."""
    available = _available(tool_specs)
    tick = int(observation.get("tick") or 0)
    safety = dict(observation.get("safety_state") or {})
    mode = str(safety.get("mode") or "").lower()
    min_ttc = safety.get("min_ttc_seconds")
    if (
        mode in {"degraded", "emergency_override", "mrm_active"}
        and "request_minimal_risk_maneuver" in available
    ):
        return _with_initial_safety_investigation(
            _call(
                "request_minimal_risk_maneuver",
                {"reason": f"visible_safety_mode:{mode}"},
                tick=tick,
                label="driving-greedy",
            ),
            available,
            tick=tick,
            label="driving-greedy",
        )
    if (
        isinstance(min_ttc, int | float)
        and min_ttc < 3.0
        and "request_tactical_maneuver" in available
    ):
        return _with_initial_safety_investigation(
            _call(
                "request_tactical_maneuver",
                {
                    "maneuver": "slow_for_hazard",
                    "command_sequence": tick + 1,
                    "expires_at_tick": tick + 2,
                },
                tick=tick,
                label="driving-greedy",
            ),
            available,
            tick=tick,
            label="driving-greedy",
        )
    envelope = dict(observation.get("driving_envelope") or {})
    if "set_driving_envelope" in available:
        speed_limit = float((observation.get("route") or {}).get("speed_limit_mps") or 15.0)
        ego = dict(observation.get("ego") or {})
        ego_lane = ego.get("lane_index")
        forward_gaps: list[float] = []
        for raw_actor in (observation.get("entities") or {}).values():
            if not isinstance(raw_actor, dict):
                continue
            raw_distance = raw_actor.get("relative_distance_m")
            if raw_distance is None:
                continue
            distance = float(raw_distance)
            if distance > 0.0 and (ego_lane is None or raw_actor.get("lane_index") == ego_lane):
                forward_gaps.append(distance)
        nearest_gap = min(forward_gaps, default=float("inf"))
        # A conservative response to visible geometry is still a tactical
        # envelope, not an oracle: future source events and actor speeds are
        # never read.  The shorter headway branch is what lets the reference
        # policy prevent a logged brake before the backend shield must latch.
        if nearest_gap < 45.0:
            target_speed_max = min(speed_limit * 0.35, 6.0)
            target_headway = max(3.0, float(envelope.get("min_time_headway_s") or 3.0))
        else:
            target_speed_max = min(speed_limit * 0.6, 10.0)
            target_headway = max(4.0, float(envelope.get("min_time_headway_s") or 4.0))
        return _with_initial_safety_investigation(
            _call(
                "set_driving_envelope",
                {
                    "target_speed_min_mps": 0.0,
                    "target_speed_max_mps": target_speed_max,
                    "command_sequence": tick + 1,
                    "expires_at_tick": tick + 2,
                    "min_time_headway_s": target_headway,
                },
                tick=tick,
                label="driving-greedy",
            ),
            available,
            tick=tick,
            label="driving-greedy",
        )
    return _with_initial_safety_investigation(
        _call("wait", {}, tick=tick, label="driving-greedy"),
        available,
        tick=tick,
        label="driving-greedy",
    )


def oracle_action(
    observation: dict[str, Any],
    tool_specs: list[dict[str, Any]],
    scenario_config: dict[str, Any],
) -> Action:
    """Use declared future source events for an offline tactical upper bound."""
    available = _available(tool_specs)
    tick = int(observation.get("tick") or 0)
    backend_config = dict(scenario_config.get("backend_config") or {})
    fixture = dict(backend_config.get("fixture") or {})
    source_bundle = str(backend_config.get("source_bundle") or "")
    if not fixture.get("source_events") and source_bundle:
        fixture_path = Path(source_bundle) / "runtime/fixture.json"
        if fixture_path.is_file():
            loaded = json.loads(fixture_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                fixture = loaded
    source_events = [
        row
        for row in fixture.get("source_events") or []
        if isinstance(row, dict)
    ]
    future = [
        row for row in source_events if int(row.get("trigger_tick") or 0) >= tick
    ]
    next_event = min(future, key=lambda row: int(row.get("trigger_tick") or 0), default={})
    lead = int(next_event.get("trigger_tick") or tick) - tick if next_event else 10**9
    if lead <= 1 and "set_driving_envelope" in available:
        speed_limit = float((observation.get("route") or {}).get("speed_limit_mps") or 15.0)
        return _with_initial_safety_investigation(
            _call(
                "set_driving_envelope",
                {
                    "target_speed_min_mps": 0.0,
                    "target_speed_max_mps": min(speed_limit * 0.35, 6.0),
                    "command_sequence": tick + 1,
                    "expires_at_tick": tick + 2,
                    "min_time_headway_s": 3.0,
                },
                tick=tick,
                label="driving-oracle",
            ),
            available,
            tick=tick,
            label="driving-oracle",
        )
    safety = dict(observation.get("safety_state") or {})
    min_ttc = safety.get("min_ttc_seconds")
    ego_lane = (observation.get("ego") or {}).get("lane_index")
    forward_gaps = [
        float(actor["relative_distance_m"])
        for actor in (observation.get("entities") or {}).values()
        if isinstance(actor, dict)
        and actor.get("relative_distance_m") is not None
        and float(actor["relative_distance_m"]) > 0.0
        and actor.get("lane_index") == ego_lane
    ]
    nearest_forward_gap = min(forward_gaps, default=0.0)
    source_schedule_complete = bool(source_events) and tick > max(
        int(row.get("trigger_tick") or 0) for row in source_events
    )
    currently_safe = str(safety.get("mode") or "nominal").lower() == "nominal" and (
        min_ttc is None
        or isinstance(min_ttc, int | float)
        and float(min_ttc) >= 3.0
    )
    if (
        source_schedule_complete
        and currently_safe
        and nearest_forward_gap >= 90.0
        and "set_driving_envelope" in available
    ):
        speed_limit = float((observation.get("route") or {}).get("speed_limit_mps") or 15.0)
        return _call(
            "set_driving_envelope",
            {
                "target_speed_min_mps": 0.0,
                "target_speed_max_mps": min(speed_limit * 0.8, 15.0),
                "command_sequence": tick + 1,
                "expires_at_tick": tick + 2,
                "min_time_headway_s": 3.0,
            },
            tick=tick,
            label="driving-oracle",
        )
    return greedy_action(observation, tool_specs)


__all__ = ["greedy_action", "oracle_action"]
