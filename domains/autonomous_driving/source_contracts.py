"""Role-aware source contracts for materialized autonomous-driving bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data.contracts import file_sha256
from .data.ngsim import verify_bundle


def _declared_path(bundle_value: str, relative: str) -> str:
    return (Path(bundle_value) / relative).as_posix()


def build_ngsim_source_contract(bundle_dir: Path) -> dict[str, Any]:
    """Return roles and exact hashes from a verified NGSIM source bundle."""
    verify_bundle(bundle_dir)
    bundle = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    fixture = json.loads(
        (bundle_dir / "runtime/fixture.json").read_text(encoding="utf-8")
    )
    derivation = dict(fixture.get("derivation") or {})
    raw_contract = bundle.get("source_contract") or {}
    required = [
        *raw_contract.get("runtime_input", []),
        *raw_contract.get("derivation_input", []),
        *raw_contract.get("implementation_asset", []),
        *raw_contract.get("metadata", []),
        *raw_contract.get("license", []),
    ]
    return {
        "runtime_input": [
            (bundle_dir / path).as_posix() for path in raw_contract.get("runtime_input", [])
        ],
        "derivation_input": [
            (bundle_dir / path).as_posix() for path in raw_contract.get("derivation_input", [])
        ],
        "implementation_asset": [
            (bundle_dir / path).as_posix() for path in raw_contract.get("implementation_asset", [])
        ],
        "metadata": [(bundle_dir / path).as_posix() for path in raw_contract.get("metadata", [])],
        "license": [(bundle_dir / path).as_posix() for path in raw_contract.get("license", [])],
        "file_sha256s": {
            (bundle_dir / path).as_posix(): file_sha256(bundle_dir / path) for path in required
        },
        "derived_window": {
            "sha256": str(derivation.get("source_window_sha256") or ""),
            "recipe_version": "ngsim_phase_complete_window_v1",
        },
    }


def ngsim(scenario: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Registry-compatible builder using ``backend_config.source_bundle``."""
    config = scenario.get("backend_config") or {}
    bundle_value = str(config.get("source_bundle") or scenario.get("source_bundle") or "").strip()
    if not bundle_value:
        raise ValueError("autonomous_driving_source_bundle_missing")
    bundle_dir = Path(bundle_value)
    if not bundle_dir.is_absolute():
        bundle_dir = repo_root / bundle_dir
    contract = build_ngsim_source_contract(bundle_dir.resolve())
    execution_mode = str(config.get("execution_mode") or "emulated_source_initialized")
    shared_runtime = {
        (bundle_dir / "bundle.json").as_posix(),
        (bundle_dir / "mining/candidates.json").as_posix(),
        (bundle_dir / "normalized/trajectories.sqlite3").as_posix(),
        (bundle_dir / "runtime/fixture.json").as_posix(),
    }
    live_runtime = {
        *shared_runtime,
        (bundle_dir / "sumo/network.net.xml").as_posix(),
        (bundle_dir / "sumo/routes.rou.xml").as_posix(),
        (bundle_dir / "sumo/run.sumocfg").as_posix(),
    }
    live_intent = execution_mode == "live" or (
        execution_mode == "auto"
        and bool(config.get("sumo_config_path"))
        and bool(config.get("ego_vehicle_id"))
    )
    exact_runtime = live_runtime if live_intent else shared_runtime
    all_declared = {
        *contract["runtime_input"],
        *contract["metadata"],
    }
    missing_runtime = sorted(exact_runtime - all_declared)
    if missing_runtime:
        raise ValueError(f"ngsim_runtime_asset_missing:{','.join(missing_runtime)}")
    displaced_runtime = set(contract["runtime_input"]) - exact_runtime
    contract["runtime_input"] = sorted(exact_runtime)
    contract["metadata"] = sorted(set(contract["metadata"]) - exact_runtime)
    contract["implementation_asset"] = sorted(
        {*contract["implementation_asset"], *displaced_runtime}
    )
    required_hashes = {*contract["runtime_input"], *contract["derivation_input"]}
    contract["file_sha256s"] = {
        path: digest
        for path, digest in contract["file_sha256s"].items()
        if path in required_hashes
    }
    if Path(bundle_value).is_absolute():
        return contract
    relative_contract: dict[str, Any] = {}
    for role in (
        "runtime_input",
        "derivation_input",
        "implementation_asset",
        "metadata",
        "license",
    ):
        relative_contract[role] = [
            _declared_path(bundle_value, Path(path).relative_to(bundle_dir).as_posix())
            for path in contract[role]
        ]
    relative_contract["file_sha256s"] = {
        _declared_path(bundle_value, Path(path).relative_to(bundle_dir).as_posix()): digest
        for path, digest in contract["file_sha256s"].items()
    }
    relative_contract["derived_window"] = contract["derived_window"]
    return relative_contract


def ngsim_source_evidence(scenario: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Registry-compatible evidence extractor for consumed locked artifacts."""
    contract = ngsim(scenario, repo_root)
    return {
        "status": "held",
        "blockers": ["reactive_closed_loop_not_validated"],
        "runtime_input": contract["runtime_input"],
        "file_sha256s": contract["file_sha256s"],
        "derived_window": contract["derived_window"],
    }
