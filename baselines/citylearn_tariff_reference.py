"""Privileged CityLearn storage reference for 0.31.1 score anchors.

Reads the locked native load, PV and price windows, including future values,
like ``oracle_offline``; it never reads model outcomes. Each configuration is
a fixed per-building rule: store the building's own PV surplus, and discharge
only to cover its own demand when the price is at or above a window quantile.
Residual stored energy is left to the fixed_reset_deliverable_energy_v1
terminal settlement. The reference calibration runs the whole ``GRID`` and the
anchor builder keeps the lowest-cost configuration per case, so the grid is
fixed in code before any run.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from core import Action, ToolCall
from core.pomdp_env import POMDPEnvironment

from .base import BaselineAgent

GRID = tuple(
    dict(rate_cap=rate_cap, discharge_quantile=quantile)
    for rate_cap in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
    for quantile in (0.0, 0.5, 0.8)
)


def config_id(config: dict[str, Any]) -> str:
    return f"cap{config['rate_cap']:g}_dq{config['discharge_quantile']:g}"


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[int(math.floor(q * (len(ordered) - 1)))]


class CityLearnTariffReferenceAgent(BaselineAgent):
    name = "citylearn_tariff_reference"

    def __init__(self, rate_cap: float = 0.2, discharge_quantile: float = 0.5):
        self.rate_cap, self.discharge_quantile = rate_cap, discharge_quantile
        self._env = None
        self._scenario: dict[str, Any] = {}
        self._tick = 0
        self._schedule: list[dict[str, float]] | None = None
        self._last: dict[str, float] = {}

    def reset(
        self, env: POMDPEnvironment, scenario_config: dict[str, Any], seed: int
    ) -> None:
        self._env, self._scenario = env, scenario_config
        self._tick, self._schedule, self._last = 0, None, {}
        self._reset_idem_seq()

    def _plan(self) -> None:
        native = getattr(getattr(self._env, "_backend", None), "_env", None)
        buildings = list(getattr(native, "buildings", []) or [])
        horizon = int(self._scenario.get("horizon_ticks") or 0)
        self._schedule = []
        if not buildings or horizon <= 0:
            return
        prices = [float(v) for v in buildings[0].pricing.electricity_pricing[:horizon]]
        if len(prices) != horizon:
            return
        threshold = _quantile(prices, self.discharge_quantile)
        rows = [dict() for _ in range(horizon)]
        for building in buildings:
            capacity = float(building.electrical_storage.capacity)
            load = np.asarray(
                building.energy_simulation.non_shiftable_load, dtype=float
            )[:horizon]
            pv = (
                np.asarray(building.energy_simulation.solar_generation, dtype=float)[
                    :horizon
                ]
                * float(building.pv.nominal_power)
                / 1000.0
            )
            if capacity <= 0 or len(load) != horizon or len(pv) != horizon:
                continue
            for tick, (demand, generation, price) in enumerate(zip(load, pv, prices)):
                net = float(demand - generation)
                if net < 0:
                    rate = min(self.rate_cap, -net / capacity)
                elif price >= threshold:
                    rate = -min(self.rate_cap, net / capacity)
                else:
                    rate = 0.0
                rows[tick][str(building.name)] = round(rate, 6)
        self._schedule = rows

    def _wait(self) -> Action:
        return Action(
            tool_calls=[
                ToolCall(name="wait", idempotency_key=self._next_idem_key("ctr_wait"))
            ],
            dominant="wait",
        )

    def act(
        self, observation: dict[str, Any], tool_specs: list[dict[str, Any]]
    ) -> Action:
        self._tick += 1
        if self._scenario.get("backend_kind") != "citylearn":
            return self._wait()
        if self._schedule is None:
            self._plan()
        delay = int(
            (self._scenario.get("backend_config") or {}).get(
                "storage_control_delay_ticks"
            )
            or 0
        )
        effect_tick = self._tick - 1 + delay
        rates = self._schedule[effect_tick] if effect_tick < len(self._schedule) else {}
        tools = {
            str(spec.get("name") or (spec.get("function") or {}).get("name") or "")
            for spec in tool_specs or []
            if isinstance(spec, dict)
        }
        idle = all(r == 0.0 for r in rates.values()) and all(
            r == 0.0 for r in self._last.values()
        )
        if "set_storage_dispatch" not in tools or not rates or idle:
            return self._wait()
        self._last = dict(rates)
        return Action(
            tool_calls=[
                ToolCall(
                    name="set_storage_dispatch",
                    args={
                        "dispatches": [
                            {"building_id": b, "rate": r}
                            for b, r in sorted(rates.items())
                        ]
                    },
                    idempotency_key=self._next_idem_key("ctr_storage"),
                )
            ],
            dominant="set_storage_dispatch",
        )
