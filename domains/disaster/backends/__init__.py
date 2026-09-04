"""Backends behind the disaster adapter.

Phase 3.3 spike ships:

- ``mock_rcrs.MockRcrsBackend``: pure-Python deterministic simulator. No
  Docker, no Java. Default backend so OPERATE can run disaster
  end-to-end without external runtime. The mock implements the SAME
  method surface as the real backend so swapping is a one-line change.
- ``rcrs_backend.RcrsBackend``: real-impl stub. Constructor sets the
  Docker image / TCP host / TCP port; every other method raises
  ``NotImplementedError`` pointing to ``docs/v0.3_disaster_design.md``.
"""

from __future__ import annotations
