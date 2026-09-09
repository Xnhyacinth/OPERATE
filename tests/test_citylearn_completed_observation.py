from types import SimpleNamespace

from domains.building_energy.backends.citylearn import CityLearnBackend


def test_inspection_reads_completed_outputs_without_changing_decision_clock():
    backend = CityLearnBackend()
    building = SimpleNamespace(
        electrical_storage=SimpleNamespace(
            soc=[0.36, 0.21, 0.0],
            energy_balance=[2.5, -1.0, 0.0],
            capacity=6.4,
            nominal_power=5.0,
        ),
        net_electricity_consumption=[5.77, -0.87, 0.0],
    )
    backend._env = SimpleNamespace(time_step=2, buildings=[building])
    backend._buildings = ["Building_1"]
    backend._last_completed_source_tick = 1
    for view in (backend.snapshot(), backend.inspect_building_state()):
        assert view["time_step"] == 2
        assert view["buildings"]["Building_1"]["soc"] == 0.21
        assert view["buildings"]["Building_1"]["net_electricity_consumption"] == -0.87
        assert view["buildings"]["Building_1"]["storage_energy_balance"] == -1.0


def test_reset_observation_and_final_step_have_explicit_native_boundary():
    backend = CityLearnBackend()
    building = SimpleNamespace(
        electrical_storage=SimpleNamespace(
            soc=[0.0, 0.4], energy_balance=[0.0, 2.0], capacity=6.4, nominal_power=5.0
        ),
        net_electricity_consumption=[1.0, 3.0],
    )
    backend._env = SimpleNamespace(time_step=0, buildings=[building])
    backend._buildings = ["Building_1"]
    assert backend.inspect_building_state()["buildings"]["Building_1"]["soc"] == 0
    # A backend may stop on its final native index rather than increment it.
    backend._env.time_step = 1
    backend._last_completed_source_tick = 1
    assert backend.inspect_building_state()["buildings"]["Building_1"]["soc"] == 0.4


def test_native_tick_publishes_its_completed_output_index():
    from tests.test_citylearn_building_energy_adapter import _effect_probe_backend

    backend = _effect_probe_backend(source_values=[3.0, 9.0, 8.0], storage_rate=0.5)
    storage = backend._env.buildings[0].electrical_storage
    storage.soc = [0.3, 0.0, 0.0]
    storage.capacity = 6.4
    storage.nominal_power = 5.0
    record = backend.tick(1)
    state = backend.inspect_building_state()["buildings"]["Building_1"]
    assert backend.current_time_step == 1
    assert (
        state["net_electricity_consumption"]
        == record.district_net_electricity_consumption
        == 3.5
    )
    assert state["soc"] == 0.3
