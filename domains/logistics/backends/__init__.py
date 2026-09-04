"""Logistics-domain backends.

- ``route_sim.RouteDemandSimulator`` — the pure-Python deterministic seeded
  route/demand simulator that IS the environment (no PyVRP required to step).
- ``pyvrp_cvrp`` / ``pyvrp_vrptw`` / ``pyvrp_lastmile`` — thin family-specific
  subclasses that set the honest-0 gating and family flags. PyVRP (MIT) is
  used only for fixed-plan route-cost *evaluation*; when absent the typed
  ``LogisticsBackendUnavailable`` is raised on that path only.
- ``orgym_invmgmt`` — descriptor-only inventory backend (0 released scenarios
  until a real demand dataset lock clears).
"""

from .route_sim import LogisticsBackendUnavailable

__all__ = ["LogisticsBackendUnavailable"]
