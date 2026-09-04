#!/usr/bin/env python3
"""Compatibility entrypoint for :mod:`scripts.download_from_hf`."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import download_from_hf  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(download_from_hf.main())
