"""
domains.logistics.backends.pyvrp_cvrp — CVRP dispatch backend.

Capacity-only routing (Augerat / Uchoa CVRP); no time windows. Primary
stressor: vehicle breakdown + demand surge → capacity over-utilization.
Honest-0 keys (§7): ``n_voltage_violations`` (no time windows); ``reserves_*``
on ``cvrp_dispatch/basic`` (no standby modeled there).

This is a thin subclass of the pure-Python ``RouteDemandSimulator``; the
family flags come from the seed's ``backend_config`` at ``reset``. PyVRP
(MIT) is used only on the optional fixed-plan cost-eval path.
"""

from __future__ import annotations

from .route_sim import RouteDemandSimulator


class PyvrpCvrpBackend(RouteDemandSimulator):
    backend_kind = "pyvrp_cvrp"
