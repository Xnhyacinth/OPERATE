"""
domains.logistics.backends.pyvrp_lastmile — Last-mile priority backend.

Time-windowed last-mile delivery (Amazon LMRRC, CC-BY-NC-4.0). Primary
stressor: urgent priority-order injection → equity/ethics dilemma + carrier
trust. All 14 keys real. The equity / ethics / stakeholder wiring lives in
the adapter (dilemmas + carrier/customer trust); the backend physics is the
shared ``RouteDemandSimulator``.
"""

from __future__ import annotations

from .route_sim import RouteDemandSimulator


class PyvrpLastmileBackend(RouteDemandSimulator):
    backend_kind = "pyvrp_lastmile"
