"""Portable evidence bindings for autonomous-driving native runs."""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from core.sidecar.sumo_sidecar import probe_sumo_transport
from domains.autonomous_driving.data.contracts import file_sha256, object_sha256
from domains.autonomous_driving.data.ngsim import verify_bundle

SUMO_BUNDLE_ASSETS = (
    "sumo/network.net.xml",
    "sumo/routes.rou.xml",
    "sumo/run.sumocfg",
)
RUNTIME_DISTRIBUTIONS = (
    "eclipse-sumo",
    "numpy",
    "scipy",
    "shapely",
    "lxml",
    "pandas",
    "PyYAML",
)
TRANSPORT_RUNTIME_SUFFIXES = {".py", ".so", ".dylib", ".dll", ".pyd"}
SUMO_VERSION_PROBE_TIMEOUT_SECONDS = 30


def runtime_implementation_binding(repo_root: Path) -> dict[str, Any]:
    runtime_paths = (
        "domains/autonomous_driving/adapter.py",
        "domains/autonomous_driving/backends/live_sumo_ego.py",
        "domains/autonomous_driving/backends/sumo_ego.py",
        "domains/autonomous_driving/runtime_assurance.py",
        "domains/autonomous_driving/lane_geometry.py",
        "domains/autonomous_driving/native_tools.py",
        "baselines/autonomous_driving_policy.py",
        "core/sidecar/sumo_sidecar.py",
        "domains/autonomous_driving/data/contracts.py",
        "domains/autonomous_driving/data/ngsim.py",
        "domains/autonomous_driving/data/commonroad_export.py",
        "domains/autonomous_driving/evidence_binding.py",
        "scripts/run_autonomous_driving_legs.py",
        "scripts/replay_autonomous_driving_calibration.py",
    )
    files = {relative: file_sha256(repo_root / relative) for relative in runtime_paths}
    semantics = {relative: files[relative] for relative in runtime_paths[:8]}
    return {
        "autonomous_driving_slice_sha256": object_sha256(files),
        "semantics_sha256": object_sha256(semantics),
        "files": files,
    }


def _distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in RUNTIME_DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _sumo_binary() -> Path | None:
    local = Path(sys.executable).parent / "sumo"
    if local.is_file():
        return local.resolve()
    discovered = shutil.which("sumo")
    return Path(discovered).resolve() if discovered else None


def _transport_package_file_hashes(module_path: Path) -> dict[str, str]:
    module_path = module_path.resolve()
    if module_path.name == "__init__.py":
        root = module_path.parent
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() in TRANSPORT_RUNTIME_SUFFIXES
        )
        return {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}
    prefixes = {module_path.stem, f"_{module_path.stem.lstrip('_')}"}
    paths = sorted(
        path
        for path in module_path.parent.iterdir()
        if path.is_file()
        and any(path.name.startswith(prefix) for prefix in prefixes)
        and path.suffix.lower() in TRANSPORT_RUNTIME_SUFFIXES
    )
    return {path.name: file_sha256(path) for path in paths}


def sumo_runtime_binding(repo_root: Path) -> dict[str, Any]:
    """Fingerprint the exact native runtime without imposing one host OS."""
    transport = probe_sumo_transport()
    binary = _sumo_binary()
    if transport is None or binary is None:
        raise ValueError("autonomous_driving_native_sumo_runtime_unavailable")
    try:
        completed = subprocess.run(
            [str(binary), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=SUMO_VERSION_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("autonomous_driving_sumo_version_probe_failed") from error
    version_output = "\n".join(
        line.rstrip() for line in completed.stdout.replace("\r\n", "\n").splitlines()
    )
    if not version_output:
        raise ValueError("autonomous_driving_sumo_version_output_missing")
    lock_files = {}
    for relative in (
        "requirements-autonomous-driving-pilot.txt",
        "requirements-autonomous-driving-full-regression.txt",
    ):
        path = repo_root / relative
        if path.is_file():
            lock_files[relative] = file_sha256(path)
    module_name = "libsumo" if transport == "libsumo" else "traci"
    module_spec = importlib.util.find_spec(module_name)
    module_path = Path(module_spec.origin).resolve() if module_spec and module_spec.origin else None
    if module_path is None or not module_path.is_file():
        raise ValueError("autonomous_driving_sumo_transport_module_missing")
    transport_files = _transport_package_file_hashes(module_path)
    if not transport_files:
        raise ValueError("autonomous_driving_sumo_transport_package_empty")
    payload = {
        "schema_version": "autonomous_driving_sumo_runtime_v2",
        "transport": transport,
        "sumo_binary_sha256": file_sha256(binary),
        "sumo_version_output_sha256": object_sha256(version_output),
        "transport_module": module_name,
        "transport_module_sha256": file_sha256(module_path),
        "transport_package_files_sha256": transport_files,
        "distributions": _distribution_versions(),
        "dependency_lock_sha256": lock_files,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
    }
    return {**payload, "runtime_sha256": object_sha256(payload)}


def verified_bundle_input_binding(bundle: Path, legs: list[dict[str, Any]]) -> dict[str, Any]:
    bundle = bundle.resolve()
    verification = verify_bundle(bundle)
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    fixture = json.loads((bundle / "runtime/fixture.json").read_text(encoding="utf-8"))
    checksum_path = bundle / "checksums.sha256"
    checksum_rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator:
            raise ValueError("driving_evidence_checksum_manifest_invalid")
        checksum_rows[relative] = digest
    sumo_assets = {relative: checksum_rows.get(relative) for relative in SUMO_BUNDLE_ASSETS}
    if any(not value for value in sumo_assets.values()):
        raise ValueError("driving_evidence_sumo_asset_missing")
    opened_sets: list[dict[str, str]] = []
    for leg in legs:
        opened = dict((leg.get("source_consumption") or {}).get("opened_source_sha256") or {})
        verified: dict[str, str] = {}
        for path_text, expected_sha256 in opened.items():
            path = Path(str(path_text))
            if not path.is_absolute():
                path = bundle / path
            try:
                relative = path.resolve().relative_to(bundle).as_posix()
            except ValueError as error:
                raise ValueError("driving_evidence_opened_source_outside_bundle") from error
            observed_sha256 = file_sha256(path)
            if observed_sha256 != str(expected_sha256):
                raise ValueError("driving_evidence_opened_source_hash_mismatch")
            verified[relative] = observed_sha256
        opened_sets.append(verified)
    if (
        len(opened_sets) != 3
        or not opened_sets[0]
        or any(value != opened_sets[0] for value in opened_sets[1:])
    ):
        raise ValueError("driving_evidence_three_leg_opened_sources_mismatch")
    source_window_sha256 = str((fixture.get("derivation") or {}).get("source_window_sha256") or "")
    source_event_chain_sha256 = str(
        (manifest.get("evidence") or {}).get("runtime_source_events_sha256") or ""
    )
    input_sha256 = object_sha256(
        {
            "bundle_verification": verification,
            "bundle_checksum_manifest_sha256": file_sha256(checksum_path),
            "bundle_file_set_sha256": object_sha256(checksum_rows),
            "sumo_assets": sumo_assets,
            "opened_files": opened_sets[0],
            "source_window_sha256": source_window_sha256,
            "source_event_chain_sha256": source_event_chain_sha256,
        }
    )
    return {
        "verification": verification,
        "opened_source_hashes_verified": True,
        "bundle_checksum_manifest_sha256": file_sha256(checksum_path),
        "bundle_file_set_sha256": object_sha256(checksum_rows),
        "sumo_asset_sha256": sumo_assets,
        "source_window_sha256": source_window_sha256,
        "source_event_chain_sha256": source_event_chain_sha256,
        "input_sha256": input_sha256,
    }


def calibration_evidence_binding(
    *, repo_root: Path, bundle: Path, candidate_id: str, legs: list[dict[str, Any]]
) -> dict[str, str]:
    implementation = runtime_implementation_binding(repo_root)
    source = verified_bundle_input_binding(bundle, legs)
    runtime = sumo_runtime_binding(repo_root)
    return {
        "candidate_id": candidate_id,
        "implementation_sha256": str(implementation["autonomous_driving_slice_sha256"]),
        "semantics_sha256": str(implementation["semantics_sha256"]),
        "input_sha256": str(source["input_sha256"]),
        "runtime_sha256": str(runtime["runtime_sha256"]),
        "source_window_sha256": str(source["source_window_sha256"]),
        "source_event_chain_sha256": str(source["source_event_chain_sha256"]),
    }
