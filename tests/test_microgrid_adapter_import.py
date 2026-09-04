"""Importing the microgrid adapter must not require pandapower."""
from __future__ import annotations

import domains.microgrid.adapter as adapter


def test_adapter_import_does_not_bind_pandapower_lv() -> None:
    assert "PandapowerLvBackend" not in vars(adapter)
    backend = adapter._build_backend("pymgrid_economic_dispatch")
    assert backend.backend_kind == "pymgrid_economic_dispatch"
