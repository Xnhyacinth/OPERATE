from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("citylearn")

from core import Action, ToolCall
from core.world_evolution_contract import canonicalize_runtime_events
from domains.building_energy.adapter import BuildingEnergyEnvironment
from domains.building_energy.backends.citylearn import (
    CityLearnBackend,
    CityLearnSourceLockError,
    _capture_runtime_opens,
    _clear_reset_priming,
    _derivation_source_files,
    _runtime_source_files,
    _replay_source_files,
    _verify_source_lock,
    electrical_storage_action_indices,
)
from domains.building_energy.seeds.schema import rebuild_seed_from_dict

SOURCE_ROOT = Path(
    "works/CityLearn/data/datasets/citylearn_challenge_2022_phase_3"
)
SOURCE_LOCK = Path(
    "sources/locks/citylearn_challenge_2022_phase_3.json"
)


class _EffectProbeEnv:
    def __init__(
        self,
        source_values: list[float],
        *,
        energy_costs: list[float] | None = None,
    ) -> None:
        self.time_step = 0
        n_ticks = len(source_values)
        self.buildings = [
            SimpleNamespace(
                energy_simulation=SimpleNamespace(
                    non_shiftable_load=source_values,
                    solar_generation=[0.0] * n_ticks,
                ),
                weather=SimpleNamespace(
                    outdoor_dry_bulb_temperature=[20.0] * n_ticks,
                ),
                pricing=SimpleNamespace(electricity_pricing=[0.2] * n_ticks),
                carbon_intensity=SimpleNamespace(
                    carbon_intensity=[0.1] * n_ticks
                ),
                electrical_storage=SimpleNamespace(
                    energy_balance=[0.0] * n_ticks
                ),
                net_electricity_consumption=list(source_values),
                net_electricity_consumption_cost=(
                    energy_costs
                    if energy_costs is not None
                    else [1.0] * n_ticks
                ),
                net_electricity_consumption_emission=[0.1] * n_ticks,
            )
        ]

    def step(
        self, actions: list[np.ndarray]
    ) -> tuple[list[float], list[float], list[bool], list[bool], dict[str, object]]:
        storage_rate = float(actions[0][0])
        building = self.buildings[0]
        building.electrical_storage.energy_balance[self.time_step] = storage_rate
        building.net_electricity_consumption[self.time_step] = (
            float(building.energy_simulation.non_shiftable_load[self.time_step])
            + storage_rate
        )
        self.time_step += 1
        return [0.0], [0.0], [False], [False], {}


def _effect_probe_backend(
    *,
    source_values: list[float],
    storage_rate: float,
    energy_costs: list[float] | None = None,
) -> CityLearnBackend:
    backend = CityLearnBackend()
    backend._env = _EffectProbeEnv(
        source_values,
        energy_costs=energy_costs,
    )
    backend._buildings = ["Building_1"]
    backend._action_vector = np.asarray([storage_rate], dtype=np.float32)
    return backend


def _scenario() -> dict[str, object]:
    return {
        "domain": "building_energy",
        "family": "citylearn_der_storage_control",
        "backend_kind": "citylearn",
        "seed_id": "citylearn_pilot_24h",
        "horizon_ticks": 4,
        "tick_minutes": 60,
        "difficulty_level": "medium",
        "source_root": str(SOURCE_ROOT),
        "source_lock": str(SOURCE_LOCK),
        "backend_config": {
            "simulation_start_time_step": 0,
            "simulation_end_time_step": 47,
            "episode_time_steps": 48,
        },
    }


def _solar_native_event(*, building_id: str = "Building_11") -> dict[str, object]:
    source_asset = SOURCE_ROOT / f"{building_id}.csv"
    source_ref = source_asset.as_posix()
    source_files = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))["files"]
    return {
        "event_id": "citylearn_solar_generation_onset_t6",
        "kind": "generation_change",
        "trigger_tick": 6,
        "channel": "building_timeseries.solar_generation",
        "source_asset": source_ref,
        "source_asset_sha256": source_files[source_ref],
        "source_row_before": 5,
        "source_row_after": 6,
        "source_value_before": 0.0,
        "source_value_after": 8.806666,
        "materiality_metric": "source_value_absolute_delta",
        "materiality_threshold": 1.0,
        "source_observed": True,
        "procedural_overlay": False,
    }


def _scenario_with_solar_event() -> dict[str, object]:
    scenario = _scenario()
    scenario["backend_config"] = {
        **scenario["backend_config"],
        "native_source_events": [_solar_native_event()],
        "task_contract": {
            "response_windows": [
                {
                    "event_id": "citylearn_solar_generation_onset_t6",
                    "first_tick": 6,
                    "last_tick": 15,
                    "native_control": "set_storage_dispatch",
                    "expected_control_policy": "charge",
                }
            ]
        },
    }
    return scenario


def _trajectory(env: BuildingEnergyEnvironment) -> str:
    actions = [
        Action(
            tool_calls=[
                ToolCall(
                    name="set_storage_dispatch",
                    args={"building_id": "Building_11", "rate": 1.0},
                    call_id="c0",
                )
            ]
        ),
        Action(tool_calls=[ToolCall(name="wait", call_id="c1")]),
        Action(
            tool_calls=[
                ToolCall(
                    name="set_storage_dispatch",
                    args={"building_id": "Building_11", "rate": -1.0},
                    call_id="c2",
                )
            ]
        ),
        Action(tool_calls=[ToolCall(name="wait", call_id="c3")]),
    ]
    rows: list[dict[str, object]] = []
    env.reset(_scenario(), seed=2022)
    for action in actions:
        result = env.step(action)
        rows.append(
            {
                "tick": env.tick,
                "observation": result.observation,
                "results": [item.to_dict() for item in result.tool_results],
                "reward": result.reward,
                "info": result.info.to_dict(),
            }
        )
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, default=str).encode()
    ).hexdigest()


def test_citylearn_native_control_has_state_effect_and_source_evidence() -> None:
    env = BuildingEnergyEnvironment()
    observation = env.reset(_scenario(), seed=2022)

    assert observation["domain"] == "building_energy"
    assert observation["clock_semantics"] == "simulator_owned"
    assert "Building_11" in observation["buildings"]
    assert "set_storage_dispatch" in {
        schema["function"]["name"] for schema in env.get_tool_specs()
    }

    result = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="set_storage_dispatch",
                    args={"building_id": "Building_11", "rate": 1.0},
                    call_id="storage-0",
                )
            ]
        )
    )

    assert env.tick == 1
    assert result.tool_results[0].ok is True
    assert result.tool_results[0].state_changing is True
    assert result.info.extra["native_state_effect_observed"] is True
    evidence = env.source_consumption_evidence(scenario=_scenario())
    assert evidence["status"] == "passed"
    assert evidence["state_effect_observed"] is True
    assert evidence["consumption_ticks"] == [0]
    assert evidence["runtime_open_evidence_kind"] == "instrumented_runtime_open_v1"
    assert evidence["runtime_opened_assets_complete"] is True
    assert evidence["missing_runtime_assets"] == []
    assert evidence["runtime_channel_proofs"]["schema"][
        "state_effect_observed"
    ] is True
    assert evidence["deterministic_replay"] is False
    assert all(
        asset["opened_by_citylearn"] is True and asset["open_phases"]
        for asset in evidence["runtime_opened_assets"]
    )
    assert not any(
        "/misc/" in asset["path"] for asset in evidence["runtime_opened_assets"]
    )
    assert {Path(asset["path"]).name for asset in evidence["derivation_assets"]} == {
        "battery_choices.yaml",
        "lbl-tracking_the_sun-res-pv.csv",
    }
    assert "pv_sizing" not in evidence["consumed_channels"]
    assert "battery_sizing" not in evidence["consumed_channels"]


def test_citylearn_reset_priming_is_cleared_without_changing_source_or_capacity() -> None:
    class Device:
        def __init__(self, consumption: float) -> None:
            self.time_step = 0
            self.nominal_power = 3.0
            self.electricity_consumption = np.asarray([consumption], dtype=float)

        def update_electricity_consumption(
            self, value: float, *, enforce_polarity: bool
        ) -> None:
            assert enforce_polarity is False
            self.electricity_consumption[self.time_step] += value

    source_demand = np.asarray([4.0], dtype=float)
    devices = [Device(1.0), Device(2.0), Device(0.0), Device(0.5)]
    building = SimpleNamespace(
        energy_simulation=SimpleNamespace(heating_demand=source_demand.copy()),
        cooling_device=devices[0],
        heating_device=devices[1],
        dhw_device=devices[2],
        non_shiftable_load_device=devices[3],
    )
    env = SimpleNamespace(time_step=0, buildings=[building])

    evidence = _clear_reset_priming(env)

    assert evidence == {
        "applied": True,
        "cleared_device_count": 3,
        "cleared_consumption": pytest.approx(3.5),
    }
    assert [device.electricity_consumption.tolist() for device in devices] == [
        [0.0],
        [0.0],
        [0.0],
        [0.0],
    ]
    assert [device.nominal_power for device in devices] == [3.0] * 4
    assert building.energy_simulation.heating_demand.tolist() == [4.0]


def test_citylearn_unopened_required_runtime_asset_fails_closed() -> None:
    seed = rebuild_seed_from_dict(_scenario(), override_seed=2022)
    backend = CityLearnBackend()
    backend.reset(seed)
    backend._source_lock["runtime_files"]["unused.csv"] = {
        "path": "works/CityLearn/data/datasets/unused.csv",
        "sha256": "0" * 64,
    }
    backend.tick(0)

    evidence = backend.source_consumption_evidence()

    assert evidence["status"] == "held"
    assert evidence["runtime_opened_assets_complete"] is False
    assert evidence["missing_runtime_assets"] == ["unused.csv"]
    assert "runtime_source_assets_not_opened" in evidence["blockers"]


def test_citylearn_source_gate_requires_native_state_effect() -> None:
    backend = CityLearnBackend()
    backend._source_consumption_ticks = [0]
    backend._source_state_effect_observed = False

    evidence = backend.source_consumption_evidence()

    assert evidence["status"] == "held"
    assert evidence["blockers"] == [
        "runtime_source_open_graph_missing",
        "runtime_source_state_effect_unobserved",
    ]


def test_citylearn_failed_replay_is_not_reported_deterministic() -> None:
    backend = CityLearnBackend()
    backend._deterministic_replay_evidence = {"deterministic_replay": False}

    evidence = backend.source_consumption_evidence()

    assert evidence["deterministic_replay"] is False


def test_citylearn_changing_source_is_not_reported_as_control_effect() -> None:
    backend = _effect_probe_backend(source_values=[1.0, 2.0], storage_rate=0.0)

    record = backend.tick(0)

    assert record.source_state_effect_observed is True
    assert record.control_state_effect_observed is False
    assert record.realized_events[0]["source_state_effect_observed"] is True
    assert record.realized_events[0]["control_state_effect_observed"] is False
    assert record.realized_events[0]["state_effect_observed"] is True
    evidence = backend.source_consumption_evidence()
    assert evidence["status"] == "held"
    assert evidence["source_state_effect_observed"] is False
    assert evidence["source_channel_input_transition_ticks"][
        "building_timeseries"
    ] == [0]
    assert evidence["source_channel_effect_ticks"] == {}
    assert "runtime_source_open_graph_missing" in evidence["blockers"]
    assert backend.control_summary()["native_state_changing_leverage"] is False


def test_citylearn_control_effect_does_not_prove_flat_source_consumption() -> None:
    backend = _effect_probe_backend(source_values=[1.0, 1.0], storage_rate=1.0)

    record = backend.tick(0)

    assert record.source_state_effect_observed is False
    assert record.control_state_effect_observed is True
    assert record.realized_events[0]["source_state_effect_observed"] is False
    assert record.realized_events[0]["control_state_effect_observed"] is True
    assert record.realized_events[0]["state_effect_observed"] is False
    evidence = backend.source_consumption_evidence()
    assert evidence["status"] == "held"
    assert evidence["state_effect_observed"] is False
    assert backend.control_summary()["native_state_changing_leverage"] is True


def test_citylearn_preserves_signed_native_export_credit() -> None:
    backend = _effect_probe_backend(
        source_values=[1.0, 1.0],
        storage_rate=0.0,
        energy_costs=[-2.5, 1.0],
    )

    record = backend.tick(0)

    assert record.energy_cost == pytest.approx(-2.5)
    assert backend.ground_truth_costs()["energy_cost"] == pytest.approx(-2.5)


def test_citylearn_source_event_peak_burden_is_opt_in_and_action_sensitive() -> None:
    wait_backend = _effect_probe_backend(
        source_values=[4.0, 4.0], storage_rate=0.0, energy_costs=[0.0, 0.0]
    )
    wait_backend._seed = SimpleNamespace(
        backend_config={
            "native_peak_response_objective": (
                "source_event_peak_response_burden_v1"
            ),
            "task_contract": {
                "response_windows": [
                    {
                        "first_tick": 0,
                        "last_tick": 0,
                        "expected_control_policy": "discharge",
                    }
                ]
            },
        }
    )
    wait_backend._task_response_windows = wait_backend._seed.backend_config[
        "task_contract"
    ]["response_windows"]
    discharge_backend = _effect_probe_backend(
        source_values=[4.0, 4.0], storage_rate=-1.0, energy_costs=[0.0, 0.0]
    )
    discharge_backend._seed = wait_backend._seed
    discharge_backend._task_response_windows = wait_backend._task_response_windows

    wait_record = wait_backend.tick(0)
    discharge_record = discharge_backend.tick(0)

    assert wait_record.district_net_electricity_consumption == pytest.approx(4.0)
    assert discharge_record.district_net_electricity_consumption == pytest.approx(3.0)
    assert wait_record.source_event_peak_response_burden == pytest.approx(16.0)
    assert discharge_record.source_event_peak_response_burden == pytest.approx(9.0)
    assert wait_backend.ground_truth_costs() == {
        "energy_cost": 0.0,
        "source_event_peak_response_burden": pytest.approx(16.0),
        "native_storage_charging_burden": 0.0,
    }

    legacy_backend = _effect_probe_backend(
        source_values=[4.0, 4.0], storage_rate=0.0, energy_costs=[0.0, 0.0]
    )
    legacy_backend.tick(0)
    assert legacy_backend.ground_truth_costs() == {"energy_cost": 0.0}


def test_citylearn_non_window_charging_has_native_energy_burden() -> None:
    backend = _effect_probe_backend(
        source_values=[4.0, 4.0], storage_rate=1.0, energy_costs=[0.0, 0.0]
    )
    backend._seed = SimpleNamespace(
        backend_config={
            "native_peak_response_objective": (
                "source_event_peak_response_burden_v1"
            )
        }
    )

    record = backend.tick(0)

    assert record.source_event_peak_response_burden == 0.0
    assert record.native_storage_charging_burden == pytest.approx(1.0)
    assert backend.ground_truth_costs()[
        "native_storage_charging_burden"
    ] == pytest.approx(1.0)


def test_citylearn_batch_storage_dispatch_is_atomic() -> None:
    backend = CityLearnBackend()
    backend._buildings = ["Building_1", "Building_2"]
    backend._storage_indices = [0, 1]
    backend._action_vector = np.zeros(2, dtype=np.float32)

    rejected = backend.queue_storage_rates(
        [
            {"building_id": "Building_1", "rate": 0.4},
            {"building_id": "unknown", "rate": -0.4},
        ]
    )

    assert rejected["_status"] == "error"
    assert backend._action_vector.tolist() == [0.0, 0.0]

    accepted = backend.queue_storage_rates(
        [
            {"building_id": "Building_1", "rate": 0.4},
            {"building_id": "Building_2", "rate": -0.4},
        ]
    )

    assert accepted["_status"] == "accepted"
    assert accepted["native_control_policy"] == "mixed"
    assert [row["building_id"] for row in accepted["dispatches"]] == [
        "Building_1",
        "Building_2",
    ]
    assert backend._action_vector.tolist() == pytest.approx([0.4, -0.4])


def test_citylearn_runtime_open_capture_isolated_across_threads(
    tmp_path: Path,
) -> None:
    source_root = tmp_path.resolve()
    paths = [source_root / f"source-{index}.csv" for index in range(2)]
    for path in paths:
        path.write_text("value\n1\n", encoding="utf-8")

    def capture(index: int) -> set[str]:
        with _capture_runtime_opens(source_root, f"worker-{index}") as trace:
            paths[index].read_text(encoding="utf-8")
            time.sleep(0.05)
            paths[index].read_text(encoding="utf-8")
        return {path.name for path in trace}

    with ThreadPoolExecutor(max_workers=2) as executor:
        traces = list(executor.map(capture, range(2)))

    assert traces == [{"source-0.csv"}, {"source-1.csv"}]


def test_citylearn_strategy_switch_requires_a_real_direction_reversal() -> None:
    backend = _effect_probe_backend(
        source_values=[1.0, 1.0, 1.0, 1.0],
        storage_rate=0.5,
    )
    backend.tick(0)
    backend._action_vector[:] = 0.5
    backend.tick(1)

    assert backend.control_summary()["strategy_reversal_count"] == 0

    backend._action_vector[:] = -0.5
    backend.tick(2)

    summary = backend.control_summary()
    assert summary["strategy_reversal_count"] == 1
    assert summary["strategy_switch_count"] == 1


def test_citylearn_named_source_event_is_material_and_response_linked() -> None:
    backend = _effect_probe_backend(
        source_values=[1.0, 2.0, 2.0],
        storage_rate=0.0,
    )
    backend._native_source_events = [
        {
            "event_id": "load-step-1",
            "kind": "load_change",
            "trigger_tick": 1,
            "channel": "building_timeseries.non_shiftable_load",
            "source_asset": "locked-load.csv",
            "source_asset_sha256": "1" * 64,
            "source_row_before": 0,
            "source_row_after": 1,
            "source_value_before": 1.0,
            "source_value_after": 2.0,
            "materiality_metric": "source_value_absolute_delta",
            "materiality_threshold": 0.5,
        }
    ]
    backend._task_response_windows = [
        {
            "event_id": "load-step-1",
            "first_tick": 1,
            "last_tick": 2,
            "native_control": "set_storage_dispatch",
            "expected_control_policy": "discharge",
        }
    ]

    record = backend.tick(0)
    event = next(
        item for item in record.realized_events if item["event_id"] == "load-step-1"
    )
    canonical = canonicalize_runtime_events([event], applied_tick=1)[0]

    assert event["changed_state_fields"] == [
        "building_timeseries.non_shiftable_load"
    ]
    assert event["before_state_digest"] != event["after_state_digest"]
    assert event["materiality_value"] == pytest.approx(1.0)
    assert event["materiality_threshold"] == pytest.approx(0.5)
    assert event["materiality_passed"] is True
    assert event["event_class"] == "alarm"
    assert event["actionable"] is True
    assert event["response_window_required"] is True
    assert event["response_opportunity_tick"] == 1
    assert event["terminal_response_window_missing"] is False
    assert canonical["material_exogenous"] is True
    assert canonical["decision_required"] is True
    assert canonical["event_contract_violations"] == []


def test_citylearn_building_event_reads_its_named_source_owner() -> None:
    backend = CityLearnBackend()
    backend._env = SimpleNamespace(
        buildings=[
            SimpleNamespace(
                name="Building_11",
                energy_simulation=SimpleNamespace(solar_generation=[0.0, 1.0]),
            ),
            SimpleNamespace(
                name="Building_12",
                energy_simulation=SimpleNamespace(solar_generation=[0.0, 2.0]),
            ),
        ]
    )
    source_asset = "locked/runtime/Building_12.csv"
    source_digest = "2" * 64
    backend._source_lock = {
        "files": {
            "building-12": {
                "path": source_asset,
                "sha256": source_digest,
            }
        }
    }
    backend._native_source_events = [
        {
            "event_id": "building-12-solar-step",
            "kind": "generation_change",
            "trigger_tick": 1,
            "channel": "building_timeseries.solar_generation",
            "source_asset": source_asset,
            "source_asset_sha256": source_digest,
            "source_value_before": 0.0,
            "source_value_after": 2.0,
            "materiality_metric": "source_value_absolute_delta",
            "materiality_threshold": 1.0,
            "source_observed": True,
            "procedural_overlay": False,
        }
    ]
    backend._task_response_windows = [
        {
            "event_id": "building-12-solar-step",
            "first_tick": 1,
            "last_tick": 2,
            "native_control": "set_storage_dispatch",
            "expected_control_policy": "charge",
        }
    ]

    backend._verify_native_event_contracts()


def test_citylearn_reset_rejects_unknown_native_event_kind() -> None:
    scenario = _scenario_with_solar_event()
    scenario["backend_config"]["native_source_events"][0]["kind"] = (
        "unregistered_source_change"
    )
    env = BuildingEnergyEnvironment()

    try:
        with pytest.raises(
            CityLearnSourceLockError,
            match="unsupported CityLearn native event kind",
        ):
            env.reset(scenario, seed=2022)
    finally:
        env.close()


def test_citylearn_reset_rejects_mismatched_native_event_class() -> None:
    scenario = _scenario_with_solar_event()
    scenario["backend_config"]["native_source_events"][0]["event_class"] = (
        "routine"
    )
    env = BuildingEnergyEnvironment()

    try:
        with pytest.raises(
            CityLearnSourceLockError,
            match="CityLearn native event class does not match registry",
        ):
            env.reset(scenario, seed=2022)
    finally:
        env.close()


def test_citylearn_named_events_require_causal_ablation_for_source_pass() -> None:
    env = BuildingEnergyEnvironment()
    scenario = _scenario_with_solar_event()
    try:
        env.reset(scenario, seed=2022)
        env.step(Action(tool_calls=[ToolCall(name="wait", call_id="wait-0")]))

        evidence = env._backend.source_consumption_evidence()

        assert evidence["runtime_opened_assets_complete"] is True
        assert evidence["state_effect_observed"] is True
        assert evidence["named_events_causally_proven"] is False
        assert evidence["status"] == "held"
        assert "named_source_events_causal_proof_missing" in evidence["blockers"]
    finally:
        env.close()


def test_citylearn_completed_episode_adapter_runs_bounded_locked_source_probe() -> None:
    env = BuildingEnergyEnvironment()
    scenario = _scenario_with_solar_event()
    scenario["horizon_ticks"] = 8
    try:
        env.reset(scenario, seed=2022)
        env.step(Action(tool_calls=[ToolCall(name="wait", call_id="wait-0")]))
        before_tick = env.tick
        before_time_step = env.snapshot()["time_step"]

        evidence = env.source_consumption_evidence(scenario=scenario)

        assert env.tick == before_tick
        assert env.snapshot()["time_step"] == before_time_step
        assert evidence["status"] == "passed"
        assert evidence["named_events_causally_proven"] is True
        assert evidence["bounded_source_probe"]["executed"] is True
        assert evidence["bounded_source_probe"]["live_clock_unchanged"] is True
    finally:
        env.close()


def test_citylearn_reset_only_adapter_runs_bounded_locked_source_probe() -> None:
    env = BuildingEnergyEnvironment()
    scenario = _scenario_with_solar_event()
    scenario["horizon_ticks"] = 8
    try:
        observation = env.reset(scenario, seed=2022)
        before_time_step = observation["time_step"]

        evidence = env.source_consumption_evidence(scenario=scenario)

        assert env.tick == 0
        assert env.snapshot()["time_step"] == before_time_step
        assert evidence["status"] == "passed"
        assert evidence["state_effect_observed"] is True
        assert evidence["named_events_causally_proven"] is True
        assert evidence["deterministic_source_trace"] is True
        assert evidence["bounded_source_probe"]["executed"] is True
        assert evidence["bounded_source_probe"]["n_ticks"] == 7
        assert evidence["bounded_source_probe"]["live_clock_unchanged"] is True
        assert evidence["source_ablation_proofs"]
        assert all(
            proof["causal_native_output_change"] is True
            for proof in evidence["source_ablation_proofs"]
        )
        assert evidence["blockers"] == []
    finally:
        env.close()


def test_citylearn_control_effect_is_call_and_evidence_linked() -> None:
    backend = _effect_probe_backend(
        source_values=[1.0, 1.0],
        storage_rate=0.0,
    )
    payload = backend.queue_storage_rate("Building_1", -0.5)
    backend.bind_control_evidence(
        call_id="dispatch-1",
        evidence_id="evidence-dispatch-1",
        payload=payload,
    )

    record = backend.tick(0)
    effect = next(
        item for item in record.realized_events if item.get("origin") == "agent_caused"
    )

    assert payload["physical_actuator_id"] == "Building_1"
    assert payload["native_control_policy"] == "discharge"
    assert payload["signed_control_value"] == pytest.approx(-0.5)
    assert effect["call_id"] == "dispatch-1"
    assert effect["evidence_ids"] == ["evidence-dispatch-1"]
    assert effect["before_state_digest"] != effect["after_state_digest"]
    assert effect["action_to_outcome_edge"]["source_call_id"] == "dispatch-1"


def test_citylearn_tick_advances_native_clock_once() -> None:
    env = BuildingEnergyEnvironment()
    env.reset(_scenario(), seed=2022)

    result = env.step(Action(tool_calls=[ToolCall(name="wait", call_id="wait-0")]))

    assert env.tick == 1
    assert result.observation["time_step"] == 1
    assert result.info.extra["trajectory_state_digest"]


def test_citylearn_ground_truth_costs_cover_the_full_episode() -> None:
    env = BuildingEnergyEnvironment()
    env.reset(_scenario(), seed=2022)
    env.step(Action(tool_calls=[ToolCall(name="wait", call_id="wait-0")]))
    env.step(Action(tool_calls=[ToolCall(name="wait", call_id="wait-1")]))

    records = env._backend.scoring_records()
    ground_truth = env.ground_truth()
    costs = ground_truth["cost_components"]

    assert len(records) == 2
    assert costs["energy_cost"] == pytest.approx(
        sum(row["energy_cost"] for row in records)
    )
    assert ground_truth["emissions_components"]["carbon_emissions"] == pytest.approx(
        sum(row["carbon_emissions"] for row in records)
    )
    assert "carbon_emissions" not in costs
    assert costs["energy_cost"] > records[-1]["energy_cost"]


def test_citylearn_source_evidence_satisfies_direct_runtime_contract() -> None:
    env = BuildingEnergyEnvironment()
    try:
        env.reset(_scenario(), seed=2022)
        env.step(Action(tool_calls=[ToolCall(name="wait", call_id="wait-0")]))
        env.step(Action(tool_calls=[ToolCall(name="wait", call_id="wait-1")]))

        evidence = env.source_consumption_evidence(scenario=_scenario())

        assert evidence["proof_kind"] == "direct_runtime_files"
        assert evidence["runtime_trace_observed"] is True
        assert evidence["evidence_from_scenario_config_only"] is False
        assert evidence["deterministic_source_trace"] is True
        assert evidence["trace_semantic_digest"]
        assert evidence["initial_state_digest"]
        assert len(evidence["post_source_state_digests"]) == 2
        assert evidence["derived_backend_state_fields"] == [
            "carbon_emissions",
            "energy_cost",
            "net_electricity_consumption",
            "storage_energy_balance",
        ]
        assert evidence["consumed_source_hashes"]
        assert set(evidence["opened_source_paths"]) == set(
            evidence["consumed_source_hashes"]
        )
        assert evidence["opened_source_sha256"] == evidence[
            "consumed_source_hashes"
        ]
    finally:
        env.close()


def test_citylearn_replay_is_deterministic() -> None:
    assert _trajectory(BuildingEnergyEnvironment()) == _trajectory(
        BuildingEnergyEnvironment()
    )


def test_citylearn_rejects_unknown_building_without_advancing_clock() -> None:
    env = BuildingEnergyEnvironment()
    env.reset(_scenario(), seed=2022)
    result = env.step(
        Action(
            tool_calls=[
                ToolCall(
                    name="set_storage_dispatch",
                    args={"building_id": "Building_999", "rate": 1.0},
                    call_id="bad-0",
                )
            ]
        )
    )

    assert result.tool_results[0].ok is False
    assert result.tool_results[0].error_code == "VALIDATION_ERROR"
    assert env.tick == 1


def test_citylearn_rejects_scalar_source_lock_hash_drift(tmp_path: Path) -> None:
    sidecar = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    sidecar["schema_sha256"] = "sha256:" + ("0" * 64)
    tampered = tmp_path / "citylearn_source_lock.json"
    tampered.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(CityLearnSourceLockError, match="schema_sha256"):
        _verify_source_lock(SOURCE_ROOT.resolve(), tampered)


def test_citylearn_accepts_locked_dataset_with_declared_optional_asset_absent(
    tmp_path: Path,
) -> None:
    from scripts.lock_citylearn_source import build as build_source_lock

    source_root = Path("works/CityLearn/data/datasets/baeda_3dem")
    if not (source_root / "schema.json").is_file():
        pytest.skip("CityLearn baeda_3dem dataset is not cloned")
    assert not (source_root / "carbon_intensity.csv").exists()
    sidecar = build_source_lock(source_root)
    lock_path = tmp_path / "baeda_source_lock.json"
    lock_path.write_text(json.dumps(sidecar), encoding="utf-8")

    verified = _verify_source_lock(source_root.resolve(), lock_path)

    assert "pricing.csv" in verified["runtime_files"]
    assert "carbon_intensity.csv" not in verified["runtime_files"]


def test_citylearn_epw_sizing_asset_is_locked_as_derivation_not_tick_runtime() -> None:
    source_root = Path(
        "works/CityLearn/data/datasets/ca_alameda_county_neighborhood"
    )
    if not (source_root / "weather.epw").is_file():
        pytest.skip("CityLearn neighborhood dataset is not cloned")

    runtime_files = _runtime_source_files(source_root)
    derivation_files = _derivation_source_files(source_root)

    assert "weather.epw" not in runtime_files
    assert derivation_files["weather.epw"] == source_root.resolve() / "weather.epw"
    assert _replay_source_files(source_root)["weather.epw"] == (
        source_root.resolve() / "weather.epw"
    )


def test_citylearn_rejects_schema_referenced_asset_marked_optional_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from domains.building_energy.backends import citylearn as backend_module

    source_root = tmp_path / "dataset"
    source_root.mkdir()
    schema_path = source_root / "schema.json"
    weather_path = source_root / "weather.csv"
    schema_path.write_text(
        json.dumps(
            {
                "buildings": {
                    "Building_1": {
                        "weather": "weather.csv",
                        "pricing": None,
                        "carbon_intensity": "carbon_intensity.csv",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    weather_path.write_text("temperature\n20\n", encoding="utf-8")
    pv_path = tmp_path / "pv.csv"
    battery_path = tmp_path / "battery.yaml"
    pv_path.write_text("size\n1\n", encoding="utf-8")
    battery_path.write_text("battery: 1\n", encoding="utf-8")
    monkeypatch.setattr(
        backend_module,
        "_locked_derivation_source_files",
        lambda _declared, _root: {
            "lbl-tracking_the_sun-res-pv.csv": pv_path,
            "battery_choices.yaml": battery_path,
        },
    )
    monkeypatch.setattr(
        backend_module,
        "_verify_runtime_identity",
        lambda **_kwargs: {},
    )

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    sidecar = {
        "package_version": "citylearn==2.5.0",
        "torch_version": "torch==2.13.0",
        "optional_runtime_assets_absent": [
            "carbon_intensity.csv",
            "pricing.csv",
        ],
        "files": {
            str(path.resolve()): digest(path)
            for path in (schema_path, weather_path, pv_path, battery_path)
        },
        "schema_sha256": f"sha256:{digest(schema_path)}",
        "weather_file_or_timeseries_lock": f"sha256:{digest(weather_path)}",
        "pricing_file_sha256": None,
        "carbon_intensity_file_sha256": None,
        "pv_sizing_file_sha256": f"sha256:{digest(pv_path)}",
        "battery_sizing_file_sha256": f"sha256:{digest(battery_path)}",
    }
    lock_path = tmp_path / "source_lock.json"
    lock_path.write_text(json.dumps(sidecar), encoding="utf-8")

    runtime_files = _runtime_source_files(source_root)
    assert "carbon_intensity.csv" in runtime_files
    assert not runtime_files["carbon_intensity.csv"].exists()
    with pytest.raises(CityLearnSourceLockError, match="missing:carbon_intensity.csv"):
        _verify_source_lock(source_root, lock_path)


def test_citylearn_rejects_runtime_package_version_drift(monkeypatch) -> None:
    import citylearn

    monkeypatch.setattr(citylearn, "__version__", "9.9.9")

    with pytest.raises(CityLearnSourceLockError, match="runtime_citylearn_version"):
        _verify_source_lock(SOURCE_ROOT.resolve(), SOURCE_LOCK.resolve())


def test_citylearn_rejects_unbound_runtime_implementation_path(
    monkeypatch, tmp_path: Path
) -> None:
    import citylearn

    fake_package = tmp_path / "citylearn"
    fake_package.mkdir()
    fake_init = fake_package / "__init__.py"
    fake_init.write_text('__version__ = "2.5.0"\n', encoding="utf-8")
    monkeypatch.setattr(citylearn, "__file__", str(fake_init))

    with pytest.raises(
        CityLearnSourceLockError,
        match="runtime_citylearn_implementation",
    ):
        _verify_source_lock(SOURCE_ROOT.resolve(), SOURCE_LOCK.resolve())


def test_citylearn_rejects_dirty_runtime_checkout(monkeypatch) -> None:
    from domains.building_energy.backends import citylearn as backend_module

    original = backend_module._git_output

    def dirty_status(root: Path, *args: str) -> str:
        if args and args[0] == "status":
            return "?? untracked-runtime.py"
        return original(root, *args)

    monkeypatch.setattr(backend_module, "_git_output", dirty_status)

    with pytest.raises(CityLearnSourceLockError, match="runtime_citylearn_checkout_dirty"):
        _verify_source_lock(SOURCE_ROOT.resolve(), SOURCE_LOCK.resolve())


def test_citylearn_rejects_unpinned_implementation_tree(tmp_path: Path) -> None:
    sidecar = json.loads(SOURCE_LOCK.read_text(encoding="utf-8"))
    sidecar["implementation_tree_sha256"] = "0" * 64
    tampered = tmp_path / "citylearn_source_lock.json"
    tampered.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(
        CityLearnSourceLockError,
        match="runtime_citylearn_implementation_lock",
    ):
        _verify_source_lock(SOURCE_ROOT.resolve(), tampered)


def test_citylearn_masked_action_replay_is_deterministic() -> None:
    seed = rebuild_seed_from_dict(_scenario(), override_seed=2022)
    backend = CityLearnBackend()
    backend.reset(seed)

    actions = [[1.0] * backend.action_width, [-1.0] * backend.action_width]
    report = backend.masked_action_replay(actions)

    assert report["status"] == "passed"
    assert report["deterministic_replay"] is True
    assert report["masked_counterfactual"]["trajectory_hash"] != report["action_run"]["trajectory_hash"]
    assert report["masked_counterfactual"]["action_count"] == 0
    assert report["action_run"]["state_effect_observed"] is True
    assert report["runtime_opened_assets_complete"] is True
    evidence = backend.source_consumption_evidence()
    assert evidence["deterministic_replay"] is True
    assert evidence["deterministic_replay_evidence"]["action_trajectory_hash"]
    assert evidence["deterministic_replay_evidence"][
        "masked_counterfactual_trajectory_hash"
    ]
    backend.close()


def test_electrical_storage_indices_follow_named_central_actions() -> None:
    env = SimpleNamespace(
        buildings=[
            SimpleNamespace(name="Building_1"),
            SimpleNamespace(name="Building_2"),
            SimpleNamespace(name="Building_3"),
        ],
        action_names=[
            [
                "dhw_storage",
                "electrical_storage",
                "cooling_device",
                "dhw_storage",
                "electrical_storage",
                "cooling_device",
                "dhw_storage",
                "electrical_storage",
                "cooling_device",
            ]
        ],
        action_space=[SimpleNamespace(shape=(9,))],
    )

    assert electrical_storage_action_indices(env) == [1, 4, 7]


def test_citylearn_2023_reset_maps_electrical_storage_not_full_width() -> None:
    source_root = Path(
        "works/CityLearn/data/datasets/citylearn_challenge_2023_phase_1"
    )
    if not (source_root / "schema.json").is_file():
        pytest.skip("CityLearn 2023 phase_1 dataset is not cloned")
    from citylearn.citylearn import CityLearnEnv

    env = CityLearnEnv(
        schema=source_root / "schema.json",
        root_directory=source_root,
        central_agent=True,
        simulation_start_time_step=144,
        simulation_end_time_step=144 + 95,
        episode_time_steps=72,
        random_seed=2022,
        render_mode="none",
    )
    env.reset(seed=2022)
    try:
        assert int(env.action_space[0].shape[0]) == 9
        assert electrical_storage_action_indices(env) == [1, 4, 7]
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def test_runtime_source_files_include_schema_lstm_weights() -> None:
    from domains.building_energy.backends.citylearn import _runtime_source_files

    source_root = Path(
        "works/CityLearn/data/datasets/citylearn_challenge_2023_phase_1"
    )
    if not (source_root / "schema.json").is_file():
        pytest.skip("CityLearn 2023 phase_1 dataset is not cloned")
    files = _runtime_source_files(source_root.resolve())
    assert files["Building_1.pth"].is_file()
    assert files["Building_2.pth"].is_file()
    assert files["Building_3.pth"].is_file()
    phase3 = Path("works/CityLearn/data/datasets/citylearn_challenge_2022_phase_3")
    if (phase3 / "schema.json").is_file():
        phase3_files = _runtime_source_files(phase3.resolve())
        assert not any(name.endswith(".pth") for name in phase3_files)
    charging = Path(
        "works/CityLearn/data/datasets/citylearn_charging_constraints_demo"
    )
    if (charging / "schema.json").is_file():
        charging_files = _runtime_source_files(charging.resolve())
        assert charging_files["charger_1_1.csv"].is_file()
        assert charging_files["Washing_Machine_1.csv"].is_file()


class _WideActionProbeEnv:
    def __init__(self) -> None:
        n_ticks = 4
        zeros = [0.0] * n_ticks
        self.time_step = 0
        self.buildings = [
            SimpleNamespace(
                name=f"Building_{index}",
                energy_simulation=SimpleNamespace(
                    non_shiftable_load=[1.0] * n_ticks,
                    solar_generation=zeros,
                ),
                weather=SimpleNamespace(
                    outdoor_dry_bulb_temperature=[20.0] * n_ticks
                ),
                pricing=SimpleNamespace(electricity_pricing=[0.2] * n_ticks),
                carbon_intensity=SimpleNamespace(carbon_intensity=[0.1] * n_ticks),
                electrical_storage=SimpleNamespace(energy_balance=list(zeros)),
                net_electricity_consumption=[1.0] * n_ticks,
                net_electricity_consumption_cost=[1.0] * n_ticks,
                net_electricity_consumption_emission=[0.1] * n_ticks,
            )
            for index in (1, 2, 3)
        ]

    def step(
        self, actions: list[np.ndarray]
    ) -> tuple[list[float], list[float], list[bool], list[bool], dict[str, object]]:
        action = np.asarray(actions[0], dtype=float).reshape(-1)
        for building, index in zip(self.buildings, (1, 4, 7), strict=True):
            building.electrical_storage.energy_balance[self.time_step] = float(
                action[index]
            )
        self.time_step += 1
        return [0.0], [0.0], [False], [False], {}


def test_citylearn_tick_records_storage_endpoints_on_wide_action_vector() -> None:
    backend = CityLearnBackend()
    backend._env = _WideActionProbeEnv()
    backend._buildings = ["Building_1", "Building_2", "Building_3"]
    backend._storage_indices = [1, 4, 7]
    backend._action_vector = np.zeros(9, dtype=np.float32)
    backend._action_vector[1] = 0.5
    backend._action_vector[7] = -0.4

    record = backend.tick(0)
    summary = backend.control_summary()

    assert len(record.action_vector) == 9
    assert summary["distinct_physical_endpoints"] == [
        "electrical_storage|Building_1",
        "electrical_storage|Building_3",
    ]
    assert backend._applied_controls[0]["control_policy"] == "mixed"
    assert summary["state_changing_control_count"] == 1


def test_citylearn_runner_binds_counterfactual_and_five_group_scoring() -> None:
    from evaluation.scorer import SCORING_VERSION, discriminative_core_total
    from runner.episode import run_one

    result = run_one(_scenario(), "wait_only", seed_override=2022)

    assert result["counterfactual"]["applicable"] is True
    assert result["counterfactual"]["masking_policy"] == "wait_only"
    assert result["task_completion"]["contract"] == (
        "building_energy.citylearn.storage_dispatch.v1"
    )
    assert result["task_completion"]["completed"] is False
    assert result["task_completion"]["reason_code"] == (
        "no_material_improvement_vs_no_action"
    )
    assert result["score"]["scoring_version"] == SCORING_VERSION
    headline = discriminative_core_total(
        result["score"]["dimensions"],
        task_completion=float(result["task_completion"]["completed"]),
        difficulty_level=result["difficulty_level"],
    )
    assert headline["aggregation"] == "scenario_applicable_five_group_v2"
    assert set(headline["group_scores"]) == {
        "task_completion",
        "system_outcome",
        "safety_and_responsibility",
        "adaptation_and_foresight",
        "action_efficiency",
    }


def test_citylearn_oracle_derives_material_policy_from_locked_future_data() -> None:
    from runner.episode import run_one

    scenario = _scenario()
    scenario["horizon_ticks"] = 24
    scenario["difficulty_level"] = "medium"
    scenario["backend_config"] = {
        **scenario["backend_config"],
        "oracle_policy_contract": (
            "citylearn.locked_future_tariff_storage_arbitrage.v1"
        ),
    }

    result = run_one(scenario, "oracle_offline", seed_override=2022)

    assert result["counterfactual"]["applicable"] is True
    assert result["counterfactual"]["prevented_loss"] > 2.5
    assert result["task_completion"]["completed"] is True
    assert result["trajectory_summary"]["tool_histogram"][
        "set_storage_dispatch"
    ] == 10


def test_citylearn_named_response_and_reversal_are_depth_evidence() -> None:
    from evaluation.task_completion import evaluate_task_completion

    scenario = _scenario()
    scenario["backend_config"] = {
        **scenario["backend_config"],
        "task_requirements": {
            "min_distinct_control_ticks": 2,
            "min_distinct_physical_tools": 1,
            "min_strategy_reversals": 1,
            "min_response_windows": 2,
            "ordered_tool_milestones": [
                {
                    "tool": "set_storage_dispatch",
                    "event_id": "solar",
                    "expected_control_policy": "charge",
                    "not_before_tick": 6,
                    "not_after_tick": 15,
                },
                {
                    "tool": "set_storage_dispatch",
                    "event_id": "tariff",
                    "expected_control_policy": "discharge",
                    "not_before_tick": 16,
                    "not_after_tick": 20,
                },
            ],
        },
    }
    counterfactual = {
        "applicable": True,
        "actual_cost": 30.0,
        "counterfactual_cost": 35.0,
        "prevented_loss": 5.0,
    }
    ground_truth = {
        "control_summary": {
            "effective_control_ticks": [10, 17],
            "distinct_physical_endpoints": [
                "electrical_storage|Building_11"
            ],
            "strategy_reversal_count": 0,
            "tool_ticks": {"set_storage_dispatch": [10, 17]},
            "response_windows": [
                {
                    "event_id": "solar",
                    "event_observed": True,
                    "event_evidence_id": "native-event:solar",
                    "expected_control_policy": "charge",
                    "observed_control_policies": ["charge"],
                    "direction_met": True,
                    "control_ticks": [10],
                },
                {
                    "event_id": "tariff",
                    "event_observed": True,
                    "event_evidence_id": "native-event:tariff",
                    "expected_control_policy": "discharge",
                    "observed_control_policies": ["charge"],
                    "direction_met": False,
                    "control_ticks": [17],
                },
            ],
        }
    }

    failed = evaluate_task_completion(
        scenario=scenario,
        ground_truth=ground_truth,
        counterfactual=counterfactual,
        score={"dimensions": []},
    )
    ground_truth["control_summary"]["strategy_reversal_count"] = 1
    ground_truth["control_summary"]["response_windows"][1].update(
        {
            "observed_control_policies": ["discharge"],
            "direction_met": True,
        }
    )
    passed = evaluate_task_completion(
        scenario=scenario,
        ground_truth=ground_truth,
        counterfactual=counterfactual,
        score={"dimensions": []},
    )

    assert failed["completed"] is True
    assert failed["evidence"]["native_control_requirements_met"] is False
    assert passed["completed"] is True
    assert passed["reason_code"] == "citylearn_storage_response_completed"
    assert passed["evidence"]["responded_response_windows"] == 2
    assert passed["evidence"]["ordered_tool_milestones_met"] is True
    assert passed["evidence"]["selected_milestone_ticks"] == [10, 17]
    assert passed["evidence"]["selected_milestone_event_ids"] == [
        "solar",
        "tariff",
    ]
    assert passed["evidence"]["selected_milestone_control_policies"] == [
        "charge",
        "discharge",
    ]

    scenario["backend_config"]["task_requirements"][
        "ordered_tool_milestones"
    ][0]["expected_control_policy"] = "discharge"
    wrongly_bound = evaluate_task_completion(
        scenario=scenario,
        ground_truth=ground_truth,
        counterfactual=counterfactual,
        score={"dimensions": []},
    )
    assert wrongly_bound["completed"] is True
    assert wrongly_bound["evidence"]["ordered_tool_milestones_met"] is False


def test_citylearn_response_windows_count_distinct_source_events() -> None:
    from evaluation.task_completion import evaluate_task_completion

    scenario = _scenario()
    scenario["backend_config"] = {
        **scenario["backend_config"],
        "task_requirements": {
            "min_distinct_control_ticks": 2,
            "min_distinct_physical_tools": 1,
            "min_strategy_reversals": 0,
            "min_response_windows": 2,
        },
    }
    duplicated_window = {
        "event_id": "solar",
        "event_observed": True,
        "event_evidence_id": "native-event:solar",
        "expected_control_policy": "charge",
        "observed_control_policies": ["charge"],
        "direction_met": True,
        "control_ticks": [10],
    }
    ground_truth = {
        "control_summary": {
            "effective_control_ticks": [10, 11],
            "distinct_physical_endpoints": [
                "electrical_storage|Building_11"
            ],
            "strategy_reversal_count": 0,
            "tool_ticks": {"set_storage_dispatch": [10, 11]},
            "response_windows": [
                duplicated_window,
                {**duplicated_window, "control_ticks": [11]},
            ],
        }
    }

    result = evaluate_task_completion(
        scenario=scenario,
        ground_truth=ground_truth,
        counterfactual={
            "applicable": True,
            "actual_cost": 30.0,
            "counterfactual_cost": 35.0,
            "prevented_loss": 5.0,
        },
        score={"dimensions": []},
    )

    assert result["completed"] is True
    assert result["evidence"]["native_control_requirements_met"] is False
    assert result["evidence"]["responded_response_windows"] == 1
    assert result["evidence"]["responded_event_ids"] == ["solar"]


def test_citylearn_source_timestep_has_valid_non_actionable_telemetry_contract():
    from core.event_protocol import EventDecisionClass, resolve_event_decision

    backend = _effect_probe_backend(source_values=[1.0, 2.0], storage_rate=0.0)
    event = backend.tick(0).realized_events[0]
    decision = resolve_event_decision(event)
    assert decision.violation_codes == ()
    assert decision.decision_class is EventDecisionClass.TELEMETRY
    assert decision.requires_decision is False
