"""Logistics / Industrial-OR domain — vehicle-routing dispatch (v0.7).

This package is the second *real-backend* OPERATE domain (see
``docs/v0.7_logistics_spec.md``). The decision center is a real-time
**fleet dispatcher**: assign / route / hold / drop orders across a horizon
while disruptions (vehicle breakdown, urgent order, demand surge, blocked
arc, traffic delay) arrive.

Per ``.hl/policy.md`` Red Line #3 the logistics domain is INTENTIONALLY
independent of ``domains.power_grid`` and ``domains.disaster``: native
entity types (depot / vehicle / customer / order / arc), native tools,
native ``Perturbation`` enum. Cross-domain coupling goes through
``core.cascade_bus.CascadeBus`` only.

Runtime posture (all pure-Python / offline / no sidecar, no Docker):

- ``backends.pyvrp_cvrp`` / ``pyvrp_vrptw`` / ``pyvrp_lastmile`` — a
  deterministic seeded route/demand **simulator IS the environment**
  (no PyVRP needed to step). PyVRP (MIT) is used only for fixed-plan
  route-cost *evaluation*; when absent, the cost-eval path raises the
  typed ``LogisticsBackendUnavailable`` and callers skip it.
- ``oracle.ortools_reference`` — historical module name for the bounded,
  dependency-invariant capacity-partition + route-ordering reference. It is
  content-cached across resets and never invoked live per tick.
- ``backends.orgym_invmgmt`` — descriptor-only inventory backend that
  registers **0 scenarios** until a real demand dataset lock clears
  (the ``egret_acopf`` precedent).
"""
