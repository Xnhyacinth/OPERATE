"""Pilot-only native building-energy integration.

CityLearn is intentionally not registered as a release backend yet.  The
adapter in this package is used to prove that the public CityLearn runtime can
be driven through the benchmark tool protocol with simulator-owned time,
source-consumption evidence, and deterministic replay.  Formal Core admission
still requires a domain contract, native task/headroom baselines, and the full
Protocol-2.1 gates.
"""

from .adapter import BuildingEnergyEnvironment

__all__ = ["BuildingEnergyEnvironment"]
