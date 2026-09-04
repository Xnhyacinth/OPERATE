"""
core.sidecar — external-process transport boundary.

A *sidecar* owns the lifecycle and transport of a single external
simulator process so that ``core/`` itself never imports a backend SDK
(``.hl/policy.md`` red-line #10 + the ``core/`` ↔ ``domains/`` boundary).

The only sidecar shipped today is :class:`~core.sidecar.sumo_sidecar.SumoSidecar`,
which selects ``libsumo`` → ``traci`` → Docker ``eclipse/sumo`` transport and
exposes a one-pull-per-tick cached snapshot. A domain backend
(``domains/traffic/backends/sumo_backend.py``) is the *only* module that talks
to the sidecar; the sidecar talks to SUMO; ``core`` talks to neither.

Design invariants (mirrors the RCRS Docker-sidecar stance in
``docs/v0.7_traffic_spec.md`` §11):

- ``core.sidecar`` imports **no** backend and **no** ``domains.*`` module.
- ``SumoSidecar.close()`` is ``finally``-guarded and force-kills any orphan
  TraCI/Docker process (draft risk #7).
- Transport is *probed*, never assumed: on a host without SUMO the sidecar is
  constructible and ``available()`` is ``False`` — only ``start()`` raises.
"""

from __future__ import annotations

from .sumo_sidecar import (
    SumoSidecar,
    SumoSidecarUnavailable,
    probe_sumo_transport,
    sumo_available,
)

__all__ = [
    "SumoSidecar",
    "SumoSidecarUnavailable",
    "probe_sumo_transport",
    "sumo_available",
]
