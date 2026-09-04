"""Backends behind the power-grid adapter.

Two are available in v0.1:

- ``pglib_uc_synthetic``: pure-Python UC simulator. No external simulator
  needed; uses the real pglib-uc / RTS-GMLC time series. This is the
  default backend so OPERATE can run end-to-end without Grid2Op.
- ``grid2op``: optional wrapper around the Grid2Op env. Gives us a real
  AC/DC power-flow backend, native opponent attacks, real chronics, and
  topology actions. Used by the storm_emergency_6h family.
"""

from __future__ import annotations
