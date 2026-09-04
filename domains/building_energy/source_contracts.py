"""CityLearn source-contract builder for pilot/preflight diagnostics.

The builder only describes locked runtime assets.  It never turns the
sidecar's declarations into consumption evidence; that proof comes from the
native backend trace exposed by :mod:`source_evidence`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def citylearn(
    scenario: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Return the complete CityLearn runtime asset contract.

    The sidecar is deliberately read here rather than importing CityLearn, so
    static preflight remains cheap and works even when the optional runtime is
    unavailable.  Paths remain exactly as locked by the sidecar; hash and
    runtime-effect verification is delegated to ``CityLearnBackend.reset``.
    """

    config = scenario.get("backend_config") or {}
    source_lock = str(
        scenario.get("source_lock")
        or config.get("source_lock")
        or "sources/locks/citylearn_challenge_2022_phase_3.json"
    )
    source_root = str(
        scenario.get("source_root")
        or config.get("source_root")
        or "works/CityLearn/data/datasets/citylearn_challenge_2022_phase_3"
    )
    result: dict[str, Any] = {
        "runtime_input": [],
        "derivation_input": [],
        "implementation_asset": [],
        "metadata": [source_lock],
        "license": [],
        "file_sha256s": {},
    }
    lock_path = Path(source_lock)
    if not lock_path.is_absolute():
        lock_path = repo_root / lock_path
    try:
        sidecar = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # The ordinary source-asset contract resolver will report a missing or
        # malformed sidecar.  Keep this builder deterministic and fail closed.
        return result
    locked_files = sidecar.get("files")
    if isinstance(locked_files, dict):
        runtime_prefix = f"{source_root.rstrip('/')}/"
        result["runtime_input"] = sorted(
            str(path) for path in locked_files if str(path).startswith(runtime_prefix)
        )
        result["derivation_input"] = sorted(
            str(path)
            for path in locked_files
            if str(path) and not str(path).startswith(runtime_prefix)
        )
        required = [*result["runtime_input"], *result["derivation_input"]]
        result["file_sha256s"] = {path: str(locked_files[path]).lower() for path in required}
    if source_root and not result["runtime_input"]:
        result["runtime_input"] = [f"{source_root}/schema.json"]
    return result
