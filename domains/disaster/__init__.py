"""Disaster-response domain — RCRS sidecar backend (Phase 3.3 spike).

This package implements the disaster vertical slice spec'd in
``docs/v0.3_disaster_design.md``. v0.3 ships a SKELETON only:

- ``backends.mock_rcrs.MockRcrsBackend`` — pure-Python deterministic
  simulator that mimics the RCRS protocol response shape and lets the
  adapter, native tools, fog of war, and tests run end-to-end without
  Java or Docker.
- ``backends.rcrs_backend.RcrsBackend`` — real-impl stub. All methods
  raise ``NotImplementedError`` with pointers to the design doc; the
  next engineer fills in the Docker lifecycle + TCP protocol against
  the same method surface.

The disaster domain is INTENTIONALLY independent of ``domains.power_grid``
(see ``.hl/policy.md`` Red Line #3): native entity types, native tools,
native stakeholders, native ``Perturbation`` enum. Cross-domain coupling
goes through ``core.cascade_bus.CascadeBus`` only.
"""
