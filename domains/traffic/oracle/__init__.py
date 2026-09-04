"""domains.traffic.oracle — offline reference-optimum computation.

Exposes :func:`compute_reference_optimum`, the traffic analogue of the
microgrid economic-dispatch oracle. It computes the deterministic
system-optimal (Wardrop) minimum total travel-time cost for a scenario and
caches it into ``seed.backend_config['reference_optimum']`` so the per-tick
scorer can read a replay-stable scalar for ``optimality_gap``.
"""

from __future__ import annotations

from .wardrop_ue import compute_reference_optimum

__all__ = ["compute_reference_optimum"]
