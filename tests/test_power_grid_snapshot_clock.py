"""The power-grid adapter owns the causal clock, including the initial view."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from core import Action
from domains.power_grid.adapter import PowerGridEnvironment


@pytest.mark.parametrize("native_tick", [None, -1, 99])
def test_snapshot_uses_adapter_clock_for_missing_or_stale_native_tick(native_tick):
    env = PowerGridEnvironment()
    env._backend = SimpleNamespace(
        snapshot=lambda: {"tick": native_tick, "native_time": 19}
    )
    env._tick = 3
    observation = env.snapshot()
    assert observation["tick"] == 3
    assert observation["native_time"] == 19


def test_native_acopf_reset_and_followup_snapshot_use_causal_tick():
    pytest.importorskip("pandapower")
    path = Path(__file__).resolve().parents[1] / (
        "scenarios/power_grid/acopf_dispatch_24h/deep_planning/extreme/"
        "pglib_opf_case30_ieee_ordered_recovery_no_line_outage_v2_extreme_s42__physical_reserve_ordered_recovery_v1.yaml"
    )
    scenario = yaml.safe_load(path.read_text())
    env = PowerGridEnvironment()
    try:
        initial = env.reset(scenario, seed=scenario["seed"])
        assert initial["tick"] == env.tick == 0
        assert env.snapshot()["tick"] == 0
        step = env.step(Action())
        assert step.observation["tick"] == env.snapshot()["tick"] == env.tick == 1
    finally:
        env.close()
