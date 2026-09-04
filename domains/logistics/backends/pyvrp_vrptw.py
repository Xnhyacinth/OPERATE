"""
domains.logistics.backends.pyvrp_vrptw — VRPTW dispatch backend.

Routing with delivery time windows (Solomon / Gehring-Homberger). Primary
stressor: traffic delay + tight windows → time-window violations
(``n_voltage_violations`` carries the time-window-breach count). All 14
keys are real (standby + reserves modeled).
"""

from __future__ import annotations

from .route_sim import RouteDemandSimulator


class PyvrpVrptwBackend(RouteDemandSimulator):
    backend_kind = "pyvrp_vrptw"
