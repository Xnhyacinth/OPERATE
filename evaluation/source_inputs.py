"""Locate frozen scoring inputs without duplicating the runtime catalog."""

from __future__ import annotations

import json
from pathlib import Path


def source_input_path(root: Path, path: str, sha256: str) -> Path:
    """Resolve a declared input inside root; callers must verify its bytes.

    Existing declared paths always win, so a damaged input cannot silently
    fall back to a second copy. Relocations must bind the requested digest.
    """
    root = Path(root).resolve()
    declared = (root / path).resolve()
    if not declared.is_relative_to(root):
        raise ValueError("source_input_outside_root")
    if declared.exists():
        return declared
    inventory = root / "evaluation/policies/source_inputs.json"
    if not inventory.is_file():
        return declared
    if not inventory.resolve().is_relative_to(root):
        raise ValueError("source_input_outside_root")
    index = json.loads(inventory.read_bytes())
    if index.get("schema_version") != "operate-source-input-locations-v1":
        raise ValueError("source_input_locations_schema_invalid")
    binding = index["files"].get(declared.relative_to(root).as_posix())
    if binding is None:
        return declared
    if binding["sha256"] != sha256:
        raise ValueError("source_input_binding_mismatch")
    resolved = (root / binding["path"]).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("source_input_outside_root")
    return resolved
