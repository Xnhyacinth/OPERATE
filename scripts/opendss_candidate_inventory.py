"""Shared OpenDSS candidate-entrypoint discovery."""

from __future__ import annotations

import hashlib
from pathlib import Path


def discover_opendss_entrypoints(root: Path) -> list[Path]:
    """Return the canonical, case-insensitive DSS entrypoint universe."""
    root = root.resolve()
    candidates = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".dss"
        and (
            path.stem.lower().startswith("master")
            or path.stem.lower().startswith("run_")
            or path.stem.lower().startswith("rundss")
        )
    }
    return sorted(candidates)


def opendss_entrypoint_units(root: Path) -> list[str]:
    root = root.resolve()
    return [path.relative_to(root).as_posix() for path in discover_opendss_entrypoints(root)]


def opendss_entrypoint_manifest_sha256(root: Path) -> str:
    payload = "\n".join(opendss_entrypoint_units(root)).encode()
    return hashlib.sha256(payload).hexdigest()
