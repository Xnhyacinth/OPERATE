"""
domains.traffic — SUMO traffic-control vertical slice (v0.7).

Evaluates an LLM as a city traffic-management-center operator making 5-minute
supervisory decisions (signal-plan switching, corridor rerouting, lane closure,
emergency pre-emption, inbound metering) on a real microscopic/mesoscopic SUMO
network under partial observability, sudden incidents, and ethical conflicts.

Boundary discipline (``.hl/policy.md`` red-lines #3/#10, ``docs/v0.7_traffic_spec.md``):

- This package imports only from ``core`` (+ its optional SUMO transport through
  ``core.sidecar``); it never imports another ``domains.*`` package.
- Native vocabulary only — ``meter_inflow``, not ``shed_load``; junction
  gridlock, not bus voltage. The 14-key canonical scorer contract is reused via
  honest aliases (see ``backends/mock_sumo.py``), with NO new scoring dimension
  and NO ``SCORING_VERSION`` bump.
- SUMO is reached as a Python library / single external process via the
  sidecar; the mock backend keeps stages 1–3 runnable on a clean checkout with
  no SUMO installed.
"""

from __future__ import annotations

__all__ = ["TrafficEnvironment"]


def __getattr__(name: str):  # pragma: no cover - thin lazy import shim
    # Lazy so ``import domains.traffic`` never forces the adapter (and its
    # transitive imports) to load until actually needed.
    if name == "TrafficEnvironment":
        from .adapter import TrafficEnvironment

        return TrafficEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
