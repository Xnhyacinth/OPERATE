"""Shared constants and utilities for the audit package."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# REPO_ROOT is now one level deeper (audit/_common.py -> parent.parent = repo root)
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # type: ignore[import]  # noqa: E402

from domains.registry import get_domain_spec  # noqa: E402

SCENARIOS_ROOT = REPO_ROOT / "scenarios" / "power_grid"
DEFAULT_REGISTRY_PATH = SCENARIOS_ROOT / "_registry.json"


def _rebuild_seed_from_dict(body: dict[str, Any], override_seed: int) -> Any:
    """T0: domain-dispatched seed rebuild.

    Resolves the domain from ``body['domain']`` (defaulting to power_grid, so
    auditing a v0.1--v0.6 power-grid release is byte-identical) and delegates to
    that domain adapter's ``_rebuild_seed_from_dict``. Lets the same auditor
    integrity-check any v0.7 domain registry when pointed at it.
    """
    return get_domain_spec(body.get("domain")).rebuild_seed_from_dict(
        body, int(override_seed)
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_scenario_path(path: str | Path) -> Path:
    """Resolve a registry path, including the sanctioned P3 archive move."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if candidate.exists():
        return candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(
            (REPO_ROOT / "scenarios" / "releases").resolve()
        )
    except ValueError:
        return candidate
    archived = REPO_ROOT / "scenarios" / "releases" / "archive" / relative
    return archived if archived.exists() else candidate


def _load_registry(registry_path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    if not registry_path.exists():
        raise FileNotFoundError(f"registry missing at {registry_path}")
    return json.loads(registry_path.read_text(encoding="utf-8"))


def _registry_metadata(registry_path: Path) -> dict[str, Any]:
    blob = registry_path.read_bytes()
    try:
        rel_path = str(registry_path.relative_to(REPO_ROOT))
    except ValueError:
        rel_path = str(registry_path)
    return {
        "path": rel_path,
        "sha256": hashlib.sha256(blob).hexdigest(),
    }
