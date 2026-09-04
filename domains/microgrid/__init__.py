"""
domains.microgrid — v0.7 Microgrid Energy-Management-System (EMS) vertical.

A domain-native second-domain slice (ships before logistics per the v0.7
arc). It evaluates an LLM as a real-time microgrid EMS: per supervisory
tick it balances net load (load − PV − wind) with battery dispatch, a
controllable genset, grid import/export at the point of common coupling
(PCC), DER curtailment and load shed — under islanding events, price
signals and noised solar/wind forecasts.

Backends:

- ``ems_sim`` — a deterministic, seeded, **pure-Python** EMS simulator that
  IS the environment (mirrors ``domains.logistics.backends.route_sim``). It
  never requires pymgrid to step. pymgrid (LGPL-3.0, dynamic-link) is used
  ONLY on the optional cross-check path (``evaluate_with_pymgrid``), which
  raises the typed ``MicrogridBackendUnavailable`` when pymgrid is absent.
- ``pandapower_lv`` — the LV power-flow tier (``microgrid_lv_voltage_6h``)
  that fills the AC power-flow keys honestly via ``pp.runpp`` and runs on a
  bare host today (pandapower already in tree).

``core/`` imports nothing from here (Red Line: clean core boundary).
"""

from __future__ import annotations
