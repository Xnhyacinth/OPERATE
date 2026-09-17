"""
domains.logistics.backends.pyvrp_vrptw — VRPTW dispatch backend.

Source-backed dispatch-wave model using Solomon / Gehring-Homberger demand
and coordinates. Native travel duration and time-window feasibility are not
implemented: runtime execution_contract marks these capabilities inapplicable.
The historical backend identifier is retained for artifact identity only.

"""

from __future__ import annotations

from .route_sim import RouteDemandSimulator


class PyvrpVrptwBackend(RouteDemandSimulator):
    backend_kind = "pyvrp_vrptw"
