from __future__ import annotations

import hashlib
import json

from core import Action, ToolCall
from domains.autonomous_driving.adapter import AutonomousDrivingEnvironment


def _scenario(*, difficulty_level: str = "basic") -> dict[str, object]:
    return {
        "domain": "autonomous_driving",
        "family": "highway_cut_in_braking",
        "backend_kind": "sumo_ego",
        "seed_id": "adapter-test",
        "horizon_ticks": 4,
        "tick_seconds": 5.0,
        "difficulty_level": difficulty_level,
        "clock_contract": {
            "schema_version": "driving_clock_v1",
            "physics_step_seconds": 0.1,
            "shield_step_seconds": 0.1,
            "substeps_per_supervisory_tick": 50,
            "provider_wall_clock_advances_simulation": False,
        },
        "backend_config": {
            "physics_step_seconds": 0.1,
            "fixture": {
                "lane_count": 2,
                "lane_width_m": 3.6,
                "route_length_m": 500.0,
                "speed_limit_mps": 20.0,
                "ego": {
                    "vehicle_id": "ego",
                    "route_position_m": 0.0,
                    "lane_index": 0,
                    "speed_mps": 15.0,
                },
                "actors": [
                    {
                        "actor_id": "lead",
                        "route_position_m": 30.0,
                        "lane_index": 1,
                        "speed_mps": 5.0,
                    }
                ],
                "source_events": [
                    {
                        "event_id": "cut-in-0",
                        "kind": "cut_in",
                        "actor_id": "lead",
                        "trigger_tick": 0,
                        "gap_m": 12.0,
                    }
                ],
            },
        },
    }


def _tool_names(env: AutonomousDrivingEnvironment) -> set[str]:
    return {row["function"]["name"] for row in env.get_tool_specs()}


def test_tool_surface_is_tactical_and_routes_through_protocol() -> None:
    env = AutonomousDrivingEnvironment()
    observation = env.reset(_scenario(), seed=17)
    names = _tool_names(env)

    assert observation["formal_core_allowed"] is False
    assert observation["physics_substeps_per_tick"] == 50
    assert observation["safety_state"]["evidence_ids"]
    assert {"brake", "throttle", "steer", "set_acceleration", "set_steering"}.isdisjoint(names)
    assert {
        "set_driving_envelope",
        "request_tactical_maneuver",
        "request_minimal_risk_maneuver",
        "request_recovery_check",
        "authorize_recovery",
    } <= names

    result = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="set_driving_envelope",
                    args={
                        "target_speed_min_mps": 0.0,
                        "target_speed_max_mps": 18.0,
                        "command_sequence": 1,
                        "expires_at_tick": 2,
                    },
                    call_id="envelope-0",
                    idempotency_key="envelope-0",
                )
            ]
        )
    )

    assert result.tool_results[0].ok is True
    assert result.tool_results[0].state_changing is True
    [effect] = [row for row in result.info.realized_events if row["origin"] == "agent_caused"]
    assert effect["call_id"] == "envelope-0"
    assurance = result.info.extra["runtime_assurance"]
    assert assurance["low_level_control_owner"] == "backend_runtime_assurance"
    assert assurance["intervention_records"]
    assert assurance["evidence_ids"]
    assert result.observation["safety_state"]["evidence_ids"] == (
        assurance["evidence_ids"]
    )
    trace = env.ground_truth()["tactical_action_trace"]
    assert trace[0]["tool_name"] == "set_driving_envelope"
    assert trace[0]["tick"] == 0


def test_high_fog_hides_relative_speed_until_paid_inspection() -> None:
    env = AutonomousDrivingEnvironment()
    observation = env.reset(_scenario(difficulty_level="high"), seed=17)

    assert observation["entities"]["lead"]["relative_speed_mps"] is None
    assert observation["safety_state"]["min_ttc_seconds"] is None
    assert observation["safety_state"]["detail_available_via"] == ("inspect_safety_state")
    scheduled = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="inspect_local_scene",
                    call_id="scene-0",
                    idempotency_key="scene-0",
                )
            ]
        )
    )
    assert scheduled.tool_results[0].latency_ticks == 1
    result = env.step(
        Action(tool_calls=[ToolCall(name="wait", call_id="wait-1", idempotency_key="wait-1")])
    )

    [scene] = [row for row in result.tool_results if row.name == "inspect_local_scene"]
    assert scene.ok is True
    assert scene.payload["actors"][0]["speed_mps"] == 5.0
    assert result.observation["entities"]["lead"]["relative_speed_mps"] is not None
    trace = env.ground_truth()["investigation_trace"]
    assert {row["tool_name"] for row in trace} >= {"inspect_local_scene"}


def test_shadow_shield_is_explicitly_diagnostic_in_step_evidence() -> None:
    scenario = _scenario()
    backend_config = dict(scenario["backend_config"])
    backend_config.update(
        {
            "diagnostic_shield_mode": "shadow",
            "unsafe_diagnostic_acknowledged": True,
        }
    )
    scenario["backend_config"] = backend_config
    env = AutonomousDrivingEnvironment()
    env.reset(scenario, seed=17)

    result = env.step(
        Action(tool_calls=[ToolCall(name="wait", call_id="wait", idempotency_key="wait")])
    )

    assurance = result.info.extra["runtime_assurance"]
    assert assurance["shield_mode"] == "shadow"
    assert assurance["shield_enforcing"] is False
    assert assurance["diagnostic_only"] is True
    assert assurance["low_level_control_owner"] == ("nominal_controller_unshielded_diagnostic")


def _semantic_trace() -> str:
    env = AutonomousDrivingEnvironment()
    env.reset(_scenario(), seed=23)
    actions = [
        Action(
            tool_calls=[
                ToolCall(
                    name="set_driving_envelope",
                    args={
                        "target_speed_min_mps": 0.0,
                        "target_speed_max_mps": 18.0,
                        "command_sequence": 1,
                        "expires_at_tick": 2,
                    },
                    call_id="envelope-0",
                    idempotency_key="envelope-0",
                )
            ]
        ),
        Action(tool_calls=[ToolCall(name="wait", call_id="wait-1", idempotency_key="wait-1")]),
    ]
    rows: list[dict[str, object]] = []
    for action in actions:
        result = env.step(action)
        rows.append(
            {
                "observation": result.observation,
                "tool_results": [item.to_dict() for item in result.tool_results],
                "reward": result.reward,
                "events": result.info.realized_events,
                "runtime_assurance": result.info.extra["runtime_assurance"],
            }
        )
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_fixture_replay_semantic_digest_is_stable_across_three_runs() -> None:
    digests = [_semantic_trace() for _ in range(3)]

    assert len(set(digests)) == 1


def test_invalid_envelope_is_rejected_without_exposing_low_level_control() -> None:
    env = AutonomousDrivingEnvironment()
    env.reset(_scenario(), seed=17)

    result = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="set_driving_envelope",
                    args={
                        "target_speed_min_mps": 0.0,
                        "target_speed_max_mps": 200.0,
                        "command_sequence": 1,
                        "expires_at_tick": 2,
                    },
                    call_id="invalid-envelope",
                )
            ]
        )
    )

    assert result.tool_results[0].ok is False
    assert result.tool_results[0].error_code == "DOMAIN_REJECTED"
    assert env.tick == 1


def test_investigation_binds_delayed_driving_envelope_effect() -> None:
    env = AutonomousDrivingEnvironment()
    env.reset(_scenario(), seed=17)
    try:
        env._tools.get("set_driving_envelope").delay_ticks = 1
        env._tools.get("set_driving_envelope").fail_rate = 0.0
        env.step(Action(tool_calls=[ToolCall(
            name="set_driving_envelope",
            args={
                "target_speed_min_mps": 0.0,
                "target_speed_max_mps": 18.0,
                "command_sequence": 1,
                "expires_at_tick": 3,
            },
            call_id="delayed-envelope",
        )]))
        _, [receipt] = env.execute_investigation(Action(tool_calls=[]))
        assert receipt.ok
        assert env.tick == 1
        result = env.step(Action(tool_calls=[]))
        [effect] = [event for event in result.info.realized_events
                    if event.get("origin") == "agent_caused"]
        assert effect["call_id"] == "delayed-envelope"
        assert receipt.evidence_id in effect["evidence_ids"]
    finally:
        env.close()
