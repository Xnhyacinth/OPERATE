"""
domains.microgrid.backends.pymgrid_backend — pymgrid-family EMS backends.

The three pymgrid families (``microgrid_islanding_24h``,
``microgrid_economic_dispatch_24h``, ``microgrid_solar_ramp_24h``) all run
on the pure-Python :class:`~domains.microgrid.backends.ems_sim.EmsSimulator`
(the simulator IS the environment). These are thin subclasses that only set
their canonical ``backend_kind`` label (mirrors
``domains.logistics.backends.pyvrp_cvrp.PyvrpCvrpBackend``).

pymgrid (LGPL-3.0) is required ONLY on the optional cross-check path
(``EmsSimulator.evaluate_with_pymgrid``); the families step without it and
the typed ``MicrogridBackendUnavailable`` is raised there only when pymgrid
is absent (spec §"Runtime gate" graceful-skip). ``MicrogridBackendUnavailable``
is re-exported here so callers can ``from ...pymgrid_backend import
MicrogridBackendUnavailable``.
"""

from __future__ import annotations

from .ems_sim import (
    PYMGRID_AVAILABLE,
    EmsSimulator,
    MicrogridBackendUnavailable,
)

__all__ = [
    "PYMGRID_AVAILABLE",
    "MicrogridBackendUnavailable",
    "PymgridIslandingBackend",
    "PymgridEconomicDispatchBackend",
    "PymgridSolarRampBackend",
]


class PymgridIslandingBackend(EmsSimulator):
    backend_kind = "pymgrid_islanding"


class PymgridEconomicDispatchBackend(EmsSimulator):
    backend_kind = "pymgrid_economic_dispatch"


class PymgridSolarRampBackend(EmsSimulator):
    backend_kind = "pymgrid_solar_ramp"
