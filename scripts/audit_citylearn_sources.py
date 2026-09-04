#!/usr/bin/env python3
"""Preflight CityLearn building-energy sources before any release work.

This report is deliberately read-only and non-release. It checks whether a
future CityLearn demand-response carrier has a source-locked schema snapshot,
timeseries files, package/runtime lock, and deterministic replay contract. It
does not install packages, download datasets, write scenario YAMLs, or modify
release artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REPORT_SCOPE = "citylearn_source_preflight"
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "works"
    / "CityLearn"
    / "data"
    / "datasets"
    / "citylearn_challenge_2022_phase_3"
)
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "citylearn_source_preflight.json"
SOURCE_LOCK_SIDECAR_NAME = "source_lock.json"
DEFAULT_SOURCE_LOCK_SIDECAR = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "citylearn_source_lock.json"
)
SAFE_COMMANDS_NOW = [
    (
        ".venv/bin/python scripts/audit_citylearn_sources.py "
        "--output reports/citylearn_source_preflight.json"
    ),
    (
        ".venv/bin/python scripts/data_expansion_readiness.py "
        "--output reports/data_expansion_readiness.json"
    ),
]
CITYLEARN_REQUIRED_MODULES = ("citylearn",)
CITYLEARN_RUNTIME_MODULES = ("citylearn.citylearn",)
CITYLEARN_RUNTIME_DEPENDENCY_PACKAGES = ("torch",)
SELECTED_SOURCE_ID = "citylearn_challenge_2022_phase_3"
SELECTED_SOURCE_SELECTION_REASON_CODE = (
    "first_building_energy_carrier_storage_comfort_demand_response"
)
CANONICAL_REPO_URL = "https://github.com/intelligent-environments-lab/CityLearn"
DATASET_DEFAULT_REPO_URL = "https://github.com/intelligent-environments-lab/CityLearn"
SOURCE_URL = CANONICAL_REPO_URL
DOCS_URL = "https://www.citylearn.net/"
PYPI_URL = "https://pypi.org/project/citylearn/"
PACKAGE_VERSION_POLICY = "stable_only_for_release; beta_allowed_for_dev_probe_only"
SOURCE_DELIVERY_REQUIRED_BEFORE_ADAPTER_PROBE = [
    "citylearn_package_available_and_version_locked",
    "citylearn_torch_version_locked",
    "citylearn_schema_present",
    "citylearn_license_verified",
    "citylearn_schema_sha256_recorded",
    "weather_timeseries_lock_recorded",
    "simulation_timeseries_lock_recorded",
    "pricing_and_carbon_timeseries_lock_recorded",
    "citylearn_pv_and_battery_sizing_lock_recorded",
    "building_cluster_and_episode_window_recorded",
    "citylearn_offline_mode_recorded",
    "episode_split_determinism_recorded",
    "simulator_seed_recorded",
]
SOURCE_LOCK_REQUIRED_FIELDS = [
    "source_id",
    "source_url",
    "dataset_source_url",
    "license",
    "lock_strategy",
    "package_version",
    "package_version_policy",
    "torch_version",
    "torch_version_policy",
    "git_commit_or_release_tag",
    "dataset_release_or_challenge_version",
    "schema_or_dataset_name",
    "schema_sha256",
    "weather_file_or_timeseries_lock",
    "simulation_file_sha256",
    "pricing_file_sha256",
    "carbon_intensity_file_sha256",
    "building_cluster",
    "episode_window",
    "citylearn_offline",
    "random_episode_split",
    "rolling_episode_split",
    "simulator_seed",
]
CITYLEARN_REQUIRED_FILES = [
    "schema.json",
    "weather.csv",
    "pricing.csv",
    "carbon_intensity.csv",
]
CITYLEARN_AUXILIARY_FILES = [
    "lbl-tracking_the_sun-res-pv.csv",
    "battery_choices.yaml",
]
CITYLEARN_TOOL_CANDIDATES = [
    {
        "tool_name": "set_battery_charge_rate",
        "state_effect": "change_electrical_storage_soc_target",
        "decision_axis": "peak_load_and_price_arbitrage",
    },
    {
        "tool_name": "set_dhw_storage_charge_rate",
        "state_effect": "change_domestic_hot_water_storage_soc_target",
        "decision_axis": "thermal_storage_shift_under_demand_forecast",
    },
    {
        "tool_name": "shift_flexible_building_load",
        "state_effect": "move_deferrable_load_across_timesteps",
        "decision_axis": "comfort_preserving_demand_response",
    },
    {
        "tool_name": "set_cooling_or_heating_setpoint",
        "state_effect": "change_comfort_setpoint_with_violation_risk",
        "decision_axis": "occupant_comfort_vs_grid_peak_or_carbon",
    },
]
CITYLEARN_EVIDENCE_CANDIDATES = [
    {
        "evidence_kind": "building_energy_balance_snapshot",
        "score_dimensions": ["information_efficiency", "foresight_score"],
    },
    {
        "evidence_kind": "storage_state_of_charge_trace",
        "score_dimensions": ["adaptive_replanning", "economic_cost"],
    },
    {
        "evidence_kind": "occupant_comfort_violation_event",
        "score_dimensions": ["safety_violation", "weighted_equity_score"],
    },
    {
        "evidence_kind": "peak_demand_or_emissions_reduction",
        "score_dimensions": ["economic_cost", "optimality_gap"],
    },
    {
        "evidence_kind": "tool_effect_on_building_load",
        "score_dimensions": ["counterfactual_prevention", "adaptive_replanning"],
    },
]
FORBIDDEN_SOURCE_SPECS = [
    {
        "source_id": "synthetic_building_timeseries_only",
        "reason": (
            "Synthetic load/weather traces may be useful for dev tests, but must "
            "not enter a released CityLearn source path without a real "
            "CityLearn schema/timeseries denominator."
        ),
    }
]


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _bundle_sha256(paths: list[Path]) -> str | None:
    if not paths or any(not path.is_file() for path in paths):
        return None
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(_sha256(path)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _path_fields(path: Path) -> dict[str, str]:
    try:
        rel = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        return {
            "path": rel,
            "path_base": "repo_root",
            "absolute_path_at_build": str(path),
        }
    except ValueError:
        return {
            "path": str(path),
            "path_base": "absolute",
            "absolute_path_at_build": str(path),
        }


def _file_fingerprint(path: Path) -> dict[str, Any]:
    digest = _sha256(path)
    exists = path.exists()
    return {
        **_path_fields(path),
        "exists": exists,
        "sha256": digest,
        "matches_current_file": (
            _sha256(path) == digest if exists and path.is_file() else not exists
        ),
    }


def build_input_fingerprints() -> dict[str, Any]:
    files = {
        "script": _file_fingerprint(Path(__file__).resolve()),
        "frontier_inventory": _file_fingerprint(
            REPO_ROOT / "scripts" / "audit_frontier_domain_candidates.py"
        ),
    }
    return {
        "schema_version": "0.1",
        "files": files,
        "all_present": all(item["exists"] is True for item in files.values()),
        "all_sha256_match_current_files": all(
            item["matches_current_file"] is True for item in files.values()
        ),
    }


def _module_report(module: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    if spec is None:
        return {
            "module": module,
            "importable": False,
            "origin": None,
            "import_error": "module_spec_not_found",
        }
    try:
        importlib.import_module(module)
    except Exception as exc:  # pragma: no cover - exact dependency errors vary by host
        return {
            "module": module,
            "importable": False,
            "origin": getattr(spec, "origin", None),
            "import_error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "module": module,
        "importable": True,
        "origin": getattr(spec, "origin", None),
        "import_error": None,
    }


def _distribution_report(package: str) -> dict[str, Any]:
    try:
        metadata = importlib.metadata.metadata(package)
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return {"package": package, "installed": False}
    return {
        "package": package,
        "installed": True,
        "version": version,
        "name": metadata.get("Name"),
        "license": metadata.get("License") or metadata.get("Classifier"),
        "summary": metadata.get("Summary"),
        "home_page": metadata.get("Home-page") or metadata.get("Project-URL"),
    }


def _json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _present(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _valid_sha256_lock(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(c in "0123456789abcdef" for c in digest.lower())


def _parse_package_version_spec(value: Any, *, package: str) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    prefix = f"{package}=="
    if not value.startswith(prefix):
        return None
    version = value.removeprefix(prefix).strip()
    return version or None


def _package_version_is_stable(version: str | None) -> bool:
    if not version:
        return False
    return version.replace(".", "").isdigit()


def _source_ref(source_root: Path, name: str) -> str:
    try:
        rel_root = source_root.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        rel_root = str(source_root)
    return f"{rel_root}/{name}"


def _source_lock_path(source_root: Path) -> Path:
    if source_root.resolve() == DEFAULT_SOURCE_ROOT.resolve():
        return DEFAULT_SOURCE_LOCK_SIDECAR
    return source_root / SOURCE_LOCK_SIDECAR_NAME


def _source_file_paths(source_root: Path) -> dict[str, Path]:
    paths = {name: source_root / name for name in CITYLEARN_REQUIRED_FILES}
    schema = _json_file(source_root / "schema.json") or {}
    buildings = schema.get("buildings") or {}
    if isinstance(buildings, dict):
        for row in buildings.values():
            if isinstance(row, dict) and _present(row.get("energy_simulation")):
                name = str(row["energy_simulation"])
                paths[name] = source_root / name
    return paths


def _auxiliary_file_paths(source_root: Path) -> dict[str, Path]:
    """Return local CityLearn runtime assets used outside the challenge schema."""
    data_root = source_root.resolve().parent.parent
    misc_root = data_root / "misc"
    return {name: misc_root / name for name in CITYLEARN_AUXILIARY_FILES}


def _required_file_reports(source_root: Path) -> list[dict[str, Any]]:
    reports = []
    paths = {**_source_file_paths(source_root), **_auxiliary_file_paths(source_root)}
    for name, path in paths.items():
        ref = _path_fields(path)["path"]
        reports.append(
            {
                **_file_fingerprint(path),
                "ref": ref,
                "name": name,
                "size_bytes": (
                    path.stat().st_size if path.exists() and path.is_file() else None
                ),
            }
        )
    return reports


def _citylearn_package_preflight(source_root: Path) -> dict[str, Any]:
    modules = {module: _module_report(module) for module in CITYLEARN_REQUIRED_MODULES}
    runtime_modules = {
        module: _module_report(module) for module in CITYLEARN_RUNTIME_MODULES
    }
    distribution = _distribution_report("citylearn")
    runtime_dependencies = {
        package: _distribution_report(package)
        for package in CITYLEARN_RUNTIME_DEPENDENCY_PACKAGES
    }
    sidecar = _json_file(_source_lock_path(source_root)) or {}
    expected_package_spec = sidecar.get("package_version")
    expected_package_version = _parse_package_version_spec(
        expected_package_spec, package="citylearn"
    )
    installed_version = distribution.get("version")
    missing_modules = [
        module
        for module, report in modules.items()
        if report.get("importable") is not True
    ]
    missing_runtime_modules = [
        module
        for module, report in runtime_modules.items()
        if report.get("importable") is not True
    ]
    version_verified = (
        distribution.get("installed") is True
        and isinstance(installed_version, str)
        and expected_package_version is not None
        and installed_version == expected_package_version
        and _package_version_is_stable(expected_package_version)
    )
    dependency_versions_verified: dict[str, bool] = {}
    expected_dependency_versions: dict[str, str | None] = {}
    for package, dependency in runtime_dependencies.items():
        expected_spec = sidecar.get(f"{package}_version")
        expected_version = _parse_package_version_spec(expected_spec, package=package)
        installed_dependency_version = dependency.get("version")
        expected_dependency_versions[package] = expected_version
        dependency_versions_verified[package] = (
            dependency.get("installed") is True
            and isinstance(installed_dependency_version, str)
            and expected_version is not None
            and installed_dependency_version == expected_version
            and _package_version_is_stable(expected_version)
        )
    runtime_versions_locked = version_verified and all(
        dependency_versions_verified.values()
    )
    return {
        "package": "citylearn",
        "modules": modules,
        "runtime_modules": runtime_modules,
        "packages": {"citylearn": distribution, **runtime_dependencies},
        "all_required_modules_importable": not missing_modules,
        "missing_modules": missing_modules,
        "all_runtime_modules_importable": not missing_runtime_modules,
        "missing_runtime_modules": missing_runtime_modules,
        "package_metadata_available": distribution.get("installed") is True,
        "package_version_locked": version_verified,
        "runtime_versions_locked": runtime_versions_locked,
        "expected_package_version": expected_package_spec,
        "expected_runtime_dependency_versions": expected_dependency_versions,
        "docs_url": DOCS_URL,
        "pypi_url": PYPI_URL,
        "runtime_lock_status": {
            "citylearn_package_version_verified": version_verified,
            "torch_version_verified": dependency_versions_verified.get(
                "torch", False
            ),
            "citylearn_runtime_importable": not missing_runtime_modules,
            "citylearn_package_version": installed_version,
            "expected_citylearn_package_version": expected_package_version,
            "expected_torch_version": expected_dependency_versions.get("torch"),
            "torch_version": runtime_dependencies.get("torch", {}).get("version"),
            "package_version_policy": PACKAGE_VERSION_POLICY,
            "expected_package_version_is_stable": _package_version_is_stable(
                expected_package_version
            ),
            "citylearn_upstream_commit_or_tag_verified": False,
            "citylearn_schema_selected": _present(
                sidecar.get("schema_or_dataset_name")
            ),
            "episode_window_recorded": _present(sidecar.get("episode_window")),
            "simulator_seed_recorded": sidecar.get("simulator_seed") is not None,
        },
    }


def _source_lock_sidecar_report(
    source_root: Path, required_file_reports: list[dict[str, Any]]
) -> dict[str, Any]:
    sidecar_path = _source_lock_path(source_root)
    sidecar = _json_file(sidecar_path)
    if sidecar is None:
        exists = sidecar_path.exists()
        return {
            **_path_fields(sidecar_path),
            "exists": exists,
            "status": (
                "blocked_source_lock_sidecar_invalid"
                if exists
                else "missing_source_lock_sidecar"
            ),
            "blocker_codes": [
                (
                    "citylearn_source_lock_sidecar_invalid"
                    if exists
                    else "citylearn_source_lock_sidecar_missing"
                )
            ],
            "verification_basis": (
                "local sidecar declaration plus schema/timeseries SHA-256 match"
            ),
            "closed": False,
            "source_lock_verified": False,
            "license_verified": False,
            "schema_sha256_recorded": False,
            "weather_timeseries_lock_recorded": False,
            "simulation_timeseries_lock_recorded": False,
            "pricing_and_carbon_timeseries_lock_recorded": False,
            "pv_and_battery_sizing_lock_recorded": False,
            "required_file_hashes": [],
        }

    blockers: list[str] = []
    required_scalar_fields = {
        "source_id": sidecar.get("source_id"),
        "source_url": sidecar.get("source_url") or sidecar.get("url"),
        "dataset_source_url": sidecar.get("dataset_source_url"),
        "license": sidecar.get("license"),
        "lock_strategy": sidecar.get("lock_strategy"),
        "package_version": sidecar.get("package_version"),
        "package_version_policy": sidecar.get("package_version_policy"),
        "torch_version": sidecar.get("torch_version"),
        "torch_version_policy": sidecar.get("torch_version_policy"),
        "git_commit_or_release_tag": sidecar.get("git_commit_or_release_tag"),
        "dataset_release_or_challenge_version": sidecar.get(
            "dataset_release_or_challenge_version"
        ),
        "schema_or_dataset_name": sidecar.get("schema_or_dataset_name"),
        "building_cluster": sidecar.get("building_cluster"),
        "episode_window": sidecar.get("episode_window"),
    }
    for field, value in required_scalar_fields.items():
        if not _present(value):
            blockers.append(f"citylearn_{field}_missing")

    declared_url = required_scalar_fields["source_url"]
    declared_dataset_url = required_scalar_fields["dataset_source_url"]
    if sidecar.get("source_id") != SELECTED_SOURCE_ID:
        blockers.append("citylearn_source_id_mismatch")
    if declared_url != SOURCE_URL:
        blockers.append("citylearn_source_url_mismatch")
    if declared_dataset_url != DATASET_DEFAULT_REPO_URL:
        blockers.append("citylearn_dataset_source_url_mismatch")
    if sidecar.get("package_version_policy") != PACKAGE_VERSION_POLICY:
        blockers.append("citylearn_package_version_policy_mismatch")
    if sidecar.get("torch_version_policy") != PACKAGE_VERSION_POLICY:
        blockers.append("citylearn_torch_version_policy_mismatch")
    expected_package_version = _parse_package_version_spec(
        sidecar.get("package_version"), package="citylearn"
    )
    if not _package_version_is_stable(expected_package_version):
        blockers.append("citylearn_package_prerelease_forbidden_for_release")
    expected_torch_version = _parse_package_version_spec(
        sidecar.get("torch_version"), package="torch"
    )
    if not _package_version_is_stable(expected_torch_version):
        blockers.append("citylearn_torch_version_missing_or_unstable")
    if sidecar.get("license_verified") is not True:
        blockers.append("citylearn_license_not_verified")
    if sidecar.get("terms_verified") is not True:
        blockers.append("citylearn_terms_not_verified")
    if sidecar.get("citylearn_offline") is not True:
        blockers.append("citylearn_offline_mode_not_declared")
    if sidecar.get("random_episode_split") is not False:
        blockers.append("citylearn_random_episode_split_not_disabled")
    if sidecar.get("rolling_episode_split") is not False:
        blockers.append("citylearn_rolling_episode_split_not_disabled")

    hash_fields = {
        "schema_sha256": sidecar.get("schema_sha256"),
        "weather_file_or_timeseries_lock": sidecar.get(
            "weather_file_or_timeseries_lock"
        ),
        "simulation_file_sha256": sidecar.get("simulation_file_sha256"),
        "pricing_file_sha256": sidecar.get("pricing_file_sha256"),
        "carbon_intensity_file_sha256": sidecar.get("carbon_intensity_file_sha256"),
        "pv_sizing_file_sha256": sidecar.get("pv_sizing_file_sha256"),
        "battery_sizing_file_sha256": sidecar.get("battery_sizing_file_sha256"),
    }
    for field, value in hash_fields.items():
        if not _valid_sha256_lock(value):
            blockers.append(f"citylearn_{field}_invalid")

    if sidecar.get("simulator_seed") is None:
        blockers.append("citylearn_simulator_seed_missing")

    declared_files = sidecar.get("files")
    if not isinstance(declared_files, dict):
        declared_files = {}
        blockers.append("citylearn_files_mapping_missing")

    required_file_hashes = []
    file_hashes_recorded = True
    file_hashes_match = True
    for report in required_file_reports:
        ref = str(report.get("ref"))
        actual_sha = report.get("sha256")
        declared_sha = declared_files.get(ref)
        matches = bool(actual_sha and declared_sha == actual_sha)
        if not declared_sha:
            file_hashes_recorded = False
            blockers.append("citylearn_file_sha256_missing")
        elif not matches:
            file_hashes_match = False
            blockers.append("citylearn_file_sha256_mismatch")
        required_file_hashes.append(
            {
                "ref": ref,
                "declared_sha256": declared_sha,
                "actual_sha256": actual_sha,
                "matches_current_file": matches,
            }
        )

    per_file_field = {
        "schema.json": "schema_sha256",
        "weather.csv": "weather_file_or_timeseries_lock",
        "pricing.csv": "pricing_file_sha256",
        "carbon_intensity.csv": "carbon_intensity_file_sha256",
        "lbl-tracking_the_sun-res-pv.csv": "pv_sizing_file_sha256",
        "battery_choices.yaml": "battery_sizing_file_sha256",
    }
    all_paths = {**_source_file_paths(source_root), **_auxiliary_file_paths(source_root)}
    actual_by_ref = {
        row["ref"]: row.get("actual_sha256") for row in required_file_hashes
    }
    for filename, field in per_file_field.items():
        lock_value = sidecar.get(field)
        actual_sha = actual_by_ref.get(_path_fields(all_paths[filename])["path"])
        if (
            _valid_sha256_lock(lock_value)
            and actual_sha
            and lock_value != (f"sha256:{actual_sha}")
        ):
            blockers.append(f"citylearn_{field}_mismatch")

    simulation_paths = [
        path
        for name, path in _source_file_paths(source_root).items()
        if name.startswith("Building_")
    ]
    simulation_bundle_sha = _bundle_sha256(simulation_paths)
    if (
        simulation_bundle_sha
        and sidecar.get("simulation_file_sha256")
        != f"sha256:{simulation_bundle_sha}"
    ):
        blockers.append("citylearn_simulation_file_sha256_mismatch")

    blockers = sorted(set(blockers))
    closed = not blockers
    return {
        **_path_fields(sidecar_path),
        "exists": True,
        "status": (
            "source_lock_sidecar_verified"
            if closed
            else "blocked_source_lock_sidecar_invalid"
        ),
        "blocker_codes": blockers,
        "verification_basis": (
            "local sidecar declaration plus schema/weather/simulation/pricing/"
            "carbon SHA-256 match; dataset terms remain an external packaging "
            "responsibility"
        ),
        "closed": closed,
        "source_id": sidecar.get("source_id"),
        "source_url": declared_url,
        "dataset_source_url": declared_dataset_url,
        "license": sidecar.get("license"),
        "lock_strategy": sidecar.get("lock_strategy"),
        "package_version": sidecar.get("package_version"),
        "package_version_policy": sidecar.get("package_version_policy"),
        "torch_version": sidecar.get("torch_version"),
        "torch_version_policy": sidecar.get("torch_version_policy"),
        "git_commit_or_release_tag": sidecar.get("git_commit_or_release_tag"),
        "dataset_release_or_challenge_version": sidecar.get(
            "dataset_release_or_challenge_version"
        ),
        "schema_or_dataset_name": sidecar.get("schema_or_dataset_name"),
        "building_cluster": sidecar.get("building_cluster"),
        "episode_window": sidecar.get("episode_window"),
        "citylearn_offline": sidecar.get("citylearn_offline") is True,
        "random_episode_split": sidecar.get("random_episode_split"),
        "rolling_episode_split": sidecar.get("rolling_episode_split"),
        "schema_sha256": sidecar.get("schema_sha256"),
        "weather_file_or_timeseries_lock": sidecar.get(
            "weather_file_or_timeseries_lock"
        ),
        "simulation_file_sha256": sidecar.get("simulation_file_sha256"),
        "pricing_file_sha256": sidecar.get("pricing_file_sha256"),
        "carbon_intensity_file_sha256": sidecar.get("carbon_intensity_file_sha256"),
        "simulator_seed": sidecar.get("simulator_seed"),
        "source_lock_verified": closed,
        "license_verified": (
            sidecar.get("license_verified") is True
            and _present(sidecar.get("license"))
            and sidecar.get("terms_verified") is True
        ),
        "citylearn_offline_mode_recorded": sidecar.get("citylearn_offline") is True,
        "episode_split_determinism_recorded": (
            sidecar.get("random_episode_split") is False
            and sidecar.get("rolling_episode_split") is False
        ),
        "schema_sha256_recorded": (
            _valid_sha256_lock(sidecar.get("schema_sha256"))
            and file_hashes_recorded
            and file_hashes_match
        ),
        "weather_timeseries_lock_recorded": (
            _valid_sha256_lock(sidecar.get("weather_file_or_timeseries_lock"))
            and file_hashes_recorded
            and file_hashes_match
        ),
        "simulation_timeseries_lock_recorded": (
            _valid_sha256_lock(sidecar.get("simulation_file_sha256"))
            and file_hashes_recorded
            and file_hashes_match
        ),
        "pricing_and_carbon_timeseries_lock_recorded": (
            _valid_sha256_lock(sidecar.get("pricing_file_sha256"))
            and _valid_sha256_lock(sidecar.get("carbon_intensity_file_sha256"))
            and file_hashes_recorded
            and file_hashes_match
        ),
        "pv_and_battery_sizing_lock_recorded": (
            _valid_sha256_lock(sidecar.get("pv_sizing_file_sha256"))
            and _valid_sha256_lock(sidecar.get("battery_sizing_file_sha256"))
            and file_hashes_recorded
            and file_hashes_match
        ),
        "pv_sizing_file_sha256": sidecar.get("pv_sizing_file_sha256"),
        "battery_sizing_file_sha256": sidecar.get("battery_sizing_file_sha256"),
        "required_file_hashes": required_file_hashes,
    }


def _schema_snapshot_probe(source_root: Path) -> dict[str, Any]:
    schema_path = source_root / "schema.json"
    blockers: list[str] = []
    schema = _json_file(schema_path)
    if not schema_path.exists():
        blockers.append("citylearn_schema_missing")
    if schema_path.exists() and schema is None:
        blockers.append("citylearn_schema_parse_failed")

    buildings = []
    if schema is not None:
        raw_buildings = schema.get("buildings")
        if isinstance(raw_buildings, dict) and raw_buildings:
            buildings = [
                {"name": name, **row}
                for name, row in raw_buildings.items()
                if isinstance(row, dict)
            ]
        elif isinstance(raw_buildings, list) and raw_buildings:
            buildings = [row for row in raw_buildings if isinstance(row, dict)]
        else:
            blockers.append("citylearn_schema_buildings_missing")

    active_buildings = [
        row for row in buildings if row.get("include", True) is not False
    ]
    if schema is not None and not active_buildings:
        blockers.append("citylearn_schema_active_buildings_missing")

    schema_actions = schema.get("actions") if schema else None
    schema_observations = schema.get("observations") if schema else None
    action_names: set[str] = {
        str(name)
        for name, config in (schema_actions or {}).items()
        if isinstance(config, dict) and config.get("active") is True
    }
    observation_names: set[str] = {
        str(name)
        for name, config in (schema_observations or {}).items()
        if isinstance(config, dict) and config.get("active") is True
    }
    simulation_files: set[str] = set()
    weather_files: set[str] = set()
    for row in active_buildings:
        simulation_name = row.get("energy_simulation") or row.get(
            "simulation_file_name"
        )
        weather_name = row.get("weather") or row.get("weather_file_name")
        if _present(simulation_name):
            simulation_files.add(str(simulation_name))
        if _present(weather_name):
            weather_files.add(str(weather_name))
    if not action_names:
        blockers.append("citylearn_schema_action_names_missing")
    if not observation_names:
        blockers.append("citylearn_schema_observation_names_missing")

    for filename in (
        "weather.csv",
        "pricing.csv",
        "carbon_intensity.csv",
    ):
        if not (source_root / filename).exists():
            blockers.append(f"citylearn_{filename.removesuffix('.csv')}_missing")

    return {
        "status": "source_snapshot_parse_ready" if not blockers else "blocked",
        "passed": not blockers,
        "blocker_codes": sorted(set(blockers)),
        "schema_file": _file_fingerprint(schema_path),
        "n_buildings": len(buildings),
        "n_active_buildings": len(active_buildings),
        "action_names_sampled": sorted(action_names),
        "observation_names_sampled": sorted(observation_names),
        "simulation_files_declared": sorted(simulation_files),
        "weather_files_declared": sorted(weather_files),
        "schema_required_fields": [
            "buildings[].name",
            "buildings[].include",
            "buildings[].observation_names",
            "actions.*.active",
            "observations.*.active",
            "buildings.*.energy_simulation",
            "buildings.*.weather",
        ],
    }


def _source_delivery_contract(
    source_root: Path,
    package_preflight: dict[str, Any],
    required_file_reports: list[dict[str, Any]],
    selected_lock: dict[str, Any],
) -> dict[str, Any]:
    files_present = all(row.get("exists") is True for row in required_file_reports)
    requirement_status = {
        "citylearn_package_available_and_version_locked": (
            package_preflight.get("package_version_locked") is True
        ),
        "citylearn_torch_version_locked": (
            package_preflight.get("runtime_lock_status", {}).get(
                "torch_version_verified"
            )
            is True
        ),
        "citylearn_schema_present": files_present,
        "citylearn_license_verified": selected_lock.get("license_verified") is True,
        "citylearn_schema_sha256_recorded": (
            selected_lock.get("schema_sha256_recorded") is True
        ),
        "weather_timeseries_lock_recorded": (
            selected_lock.get("weather_timeseries_lock_recorded") is True
        ),
        "simulation_timeseries_lock_recorded": (
            selected_lock.get("simulation_timeseries_lock_recorded") is True
        ),
        "pricing_and_carbon_timeseries_lock_recorded": (
            selected_lock.get("pricing_and_carbon_timeseries_lock_recorded") is True
        ),
        "citylearn_pv_and_battery_sizing_lock_recorded": (
            selected_lock.get("pv_and_battery_sizing_lock_recorded") is True
        ),
        "building_cluster_and_episode_window_recorded": _present(
            selected_lock.get("building_cluster")
        )
        and _present(selected_lock.get("episode_window")),
        "citylearn_offline_mode_recorded": (
            selected_lock.get("citylearn_offline_mode_recorded") is True
        ),
        "episode_split_determinism_recorded": (
            selected_lock.get("episode_split_determinism_recorded") is True
        ),
        "simulator_seed_recorded": selected_lock.get("simulator_seed") is not None,
    }
    current_blockers = [
        key for key, passed in requirement_status.items() if passed is not True
    ]
    current_blockers.extend(selected_lock.get("blocker_codes") or [])
    current_blockers = sorted(set(current_blockers))
    source = {
        "source_id": SELECTED_SOURCE_ID,
        "root": _path_fields(source_root)["path"],
        "url": SOURCE_URL,
        "dataset_source_url": DATASET_DEFAULT_REPO_URL,
        "docs_url": DOCS_URL,
        "pypi_url": PYPI_URL,
        "license": "MIT package claim; bundled dataset/timeseries terms must be verified",
        "license_posture": "package_oss_but_dataset_timeseries_terms_need_verification",
        "selection_reason_code": SELECTED_SOURCE_SELECTION_REASON_CODE,
        "families": ["building_energy_demand_response"],
        "required_files": [
            _path_fields(path)["path"]
            for path in {
                **_source_file_paths(source_root),
                **_auxiliary_file_paths(source_root),
            }.values()
        ],
        "current_blockers": current_blockers,
        "blocker_groups": {
            "runtime": ["citylearn_package_available_and_version_locked"]
            if not requirement_status["citylearn_package_available_and_version_locked"]
            else [],
            "source_lock": list(selected_lock.get("blocker_codes") or []),
            "files": ["citylearn_schema_present"] if not files_present else [],
        },
    }
    return {
        "status": (
            "selected_citylearn_source_locked"
            if selected_lock.get("closed") is True
            else "blocked_waiting_for_citylearn_source_lock"
        ),
        "root_count_basis": "citylearn_source_delivery_specs",
        "n_source_roots": 1,
        "selected_source_id": SELECTED_SOURCE_ID,
        "selected_source_candidate": source,
        "selected_source_file_lock": {
            "source_id": SELECTED_SOURCE_ID,
            "closed": selected_lock.get("closed") is True,
            "status": selected_lock.get("status"),
            "source_url": selected_lock.get("source_url") or SOURCE_URL,
            "dataset_source_url": selected_lock.get("dataset_source_url")
            or DATASET_DEFAULT_REPO_URL,
            "sidecar_path": selected_lock.get("path"),
            "blocker_codes": list(selected_lock.get("blocker_codes") or []),
            "required_file_hashes": list(
                selected_lock.get("required_file_hashes") or []
            ),
        },
        "requirement_status": requirement_status,
        "missing_source_delivery_requirements": current_blockers,
        "lock_closure_required_fields": list(SOURCE_LOCK_REQUIRED_FIELDS),
        "required_before_adapter_probe": list(
            SOURCE_DELIVERY_REQUIRED_BEFORE_ADAPTER_PROBE
        ),
        "sources": [source],
        "forbidden_sources": list(FORBIDDEN_SOURCE_SPECS),
        "next_source_lock_action": (
            "Use the official CityLearn dataset directory with schema.json, "
            "Building_*.csv, weather.csv, pricing.csv, and carbon_intensity.csv; "
            "then record package version, "
            "commit/tag, dataset/challenge version, file hashes, building cluster, "
            "episode window, CITYLEARN_OFFLINE=true, deterministic episode-split "
            "flags, license/terms verification, and simulator seed in the "
            "candidate release's citylearn_source_lock.json."
        ),
    }


def _case_ledger_preview(
    selected_lock: dict[str, Any], schema_probe: dict[str, Any]
) -> dict[str, Any]:
    if (
        selected_lock.get("closed") is not True
        or schema_probe.get("passed") is not True
    ):
        return {
            "status": "blocked_waiting_for_locked_citylearn_schema",
            "candidate_rows": [],
            "blocker_codes": [
                "citylearn_source_lock_not_closed",
                *list(selected_lock.get("blocker_codes") or []),
                *list(schema_probe.get("blocker_codes") or []),
            ],
        }
    dataset = str(selected_lock.get("dataset_release_or_challenge_version"))
    schema_name = str(selected_lock.get("schema_or_dataset_name"))
    cluster = str(selected_lock.get("building_cluster"))
    schema_sha = str(selected_lock.get("schema_sha256"))
    weather_sha = str(selected_lock.get("weather_file_or_timeseries_lock"))
    row = {
        "family": "building_energy_demand_response",
        "backend_kind": "citylearn_gymnasium",
        "source_denominator_key": (
            f"citylearn:{dataset}:{schema_name}:{cluster}:{schema_sha}"
        ),
        "source_variant_key": f"citylearn_weather:{dataset}:{weather_sha}",
        "independence_axis": (
            "CityLearn dataset/challenge schema + building cluster + schema hash"
        ),
        "decision_pressure_axis": (
            "storage and flexible-load dispatch under peak, price, carbon, and "
            "comfort pressure"
        ),
        "source_axes": {
            "dataset_release_or_challenge_version": dataset,
            "schema_or_dataset_name": schema_name,
            "building_cluster": cluster,
            "episode_window": selected_lock.get("episode_window"),
            "schema_sha256": schema_sha,
            "weather_file_or_timeseries_lock": weather_sha,
            "simulator_seed": selected_lock.get("simulator_seed"),
        },
        "complexity_tags": [
            "forecast_dependent_load_shifting",
            "storage_soc_persistence",
            "comfort_constraint_tradeoff",
            "price_carbon_multi_objective",
        ],
        "dimension_applicability": {
            "economic_cost": {"applicable": True, "reason": "electricity_cost_trace"},
            "safety_violation": {
                "applicable": True,
                "reason": "comfort_or_equipment_limit_violation",
            },
            "weighted_equity_score": {
                "applicable": True,
                "reason": "per_building_comfort_or_service_imbalance",
            },
            "counterfactual_prevention": {
                "applicable": True,
                "reason": "masked_action_replay_over_fixed_citylearn_trace_required",
            },
        },
        "honest_zero_keys": ["n_voltage_violations"],
        "expected_native_tools": [
            tool["tool_name"] for tool in CITYLEARN_TOOL_CANDIDATES
        ],
        "expected_evidence_kinds": [
            evidence["evidence_kind"] for evidence in CITYLEARN_EVIDENCE_CANDIDATES
        ],
        "keep_rationale": (
            "Building energy demand response adds native storage, comfort, "
            "price, and carbon scheduling pressure not covered by traffic, "
            "routing, inventory, or power-flow rows."
        ),
        "release_blockers": [
            "citylearn_release_backend_formalization_pending",
            "citylearn_scorer_evidence_not_wired",
            "citylearn_behavioral_gate_not_run",
            "citylearn_scorer_masked_replay_wiring_not_implemented",
            "citylearn_release_materializer_not_implemented",
        ],
    }
    return {
        "status": "case_ledger_preview_ready",
        "candidate_rows": [row],
        "blocker_codes": list(row["release_blockers"]),
    }


def _replay_contract_preview(
    selected_lock: dict[str, Any], schema_probe: dict[str, Any]
) -> dict[str, Any]:
    ready = selected_lock.get("closed") is True and schema_probe.get("passed") is True
    return {
        "status": (
            "source_locked_replay_contract_preview_ready"
            if ready
            else "blocked_waiting_for_locked_citylearn_trace"
        ),
        "deterministic_replay_contract_declared": ready,
        "counterfactual_prevention_applicable": ready,
        "masked_action_replay_required": True,
        "release_blocker_codes": []
        if ready
        else [
            "citylearn_source_lock_not_closed",
            *list(selected_lock.get("blocker_codes") or []),
            *list(schema_probe.get("blocker_codes") or []),
        ],
        "future_action_stream_shape": {
            "tick": "integer CityLearn timestep",
            "actions": [
                "set_battery_charge_rate",
                "set_dhw_storage_charge_rate",
                "shift_flexible_building_load",
                "set_cooling_or_heating_setpoint",
            ],
            "masked_counterfactual": (
                "same schema, episode window, seed, and weather/load traces "
                "with CITYLEARN_OFFLINE=true and baseline no-op storage actions"
            ),
        },
    }


def _runtime_execution_proof(
    package_preflight: dict[str, Any], selected_lock: dict[str, Any]
) -> dict[str, Any]:
    source_locked = selected_lock.get("closed") is True
    runtime_locked = package_preflight.get("runtime_versions_locked") is True
    blockers = []
    if not source_locked:
        blockers.append("citylearn_source_lock_not_closed")
    if not runtime_locked:
        blockers.append("citylearn_runtime_dependency_versions_not_locked")
    if package_preflight.get("all_runtime_modules_importable") is not True:
        blockers.append("citylearn_runtime_import_failed")
    pilot_adapter_importable = False
    pilot_adapter_formal_core_allowed = None
    try:
        from domains.building_energy.adapter import BuildingEnergyEnvironment

        pilot_adapter_importable = True
        pilot_adapter_formal_core_allowed = bool(
            BuildingEnergyEnvironment.formal_core_allowed
        )
    except (ImportError, AttributeError, ModuleNotFoundError):
        blockers.append("citylearn_pilot_adapter_import_failed")
    blockers.extend(
        [
            "missing_citylearn_release_runtime_execution",
            "citylearn_release_backend_scorer_evidence_not_wired",
            "citylearn_scorer_masked_replay_wiring_not_implemented",
        ]
    )
    return {
        "status": (
            "source_locked_runtime_probe_blocked"
            if source_locked
            else "blocked_waiting_for_citylearn_source_lock"
        ),
        "executed_with_release_backend": False,
        "pilot_adapter_importable": pilot_adapter_importable,
        "pilot_adapter_formal_core_allowed": pilot_adapter_formal_core_allowed,
        "scorer_consumed_release_backend_evidence": False,
        "release_backend_kind": "citylearn_gymnasium",
        "package_version_locked": runtime_locked,
        "blocker_codes": sorted(set(blockers)),
    }


def build_citylearn_source_preflight_report(
    *, source_root: Path = DEFAULT_SOURCE_ROOT
) -> dict[str, Any]:
    required_file_reports = _required_file_reports(source_root)
    package_preflight = _citylearn_package_preflight(source_root)
    selected_lock = _source_lock_sidecar_report(source_root, required_file_reports)
    schema_probe = _schema_snapshot_probe(source_root)
    source_contract = _source_delivery_contract(
        source_root, package_preflight, required_file_reports, selected_lock
    )
    case_ledger = _case_ledger_preview(selected_lock, schema_probe)
    replay_contract = _replay_contract_preview(selected_lock, schema_probe)
    runtime_execution = _runtime_execution_proof(package_preflight, selected_lock)
    release_blockers = sorted(
        set(
            source_contract["missing_source_delivery_requirements"]
            + runtime_execution["blocker_codes"]
            + case_ledger["blocker_codes"]
            + [
                "missing_citylearn_evidence_wiring",
                "missing_citylearn_behavioral_gates",
                "missing_citylearn_release_materializer",
            ]
        )
    )
    status = (
        "blocked_waiting_for_citylearn_source_lock"
        if selected_lock.get("closed") is not True
        else (
            "blocked_waiting_for_citylearn_runtime_lock"
            if package_preflight.get("package_version_locked") is not True
            else "source_locked_runtime_probe_blocked"
        )
    )
    return {
        "schema_version": "0.1",
        "scope": REPORT_SCOPE,
        "status": status,
        "non_release_artifact": True,
        "release_ready": False,
        "release_reentry_ready": False,
        "proceed_commands": [],
        "safe_commands_now": list(SAFE_COMMANDS_NOW),
        "source_root": str(source_root),
        "package_preflight": package_preflight,
        "source_delivery_contract": source_contract,
        "schema_snapshot_probe": schema_probe,
        "tool_candidates": list(CITYLEARN_TOOL_CANDIDATES),
        "evidence_candidates": list(CITYLEARN_EVIDENCE_CANDIDATES),
        "case_ledger_preview": case_ledger,
        "replay_contract_preview": replay_contract,
        "runtime_execution_proof": runtime_execution,
        "release_blocker_codes": release_blockers,
        "input_fingerprints": build_input_fingerprints(),
        "policy": {
            "writes_release_artifacts": False,
            "writes_scenario_yaml": False,
            "installs_dependencies": False,
            "downloads_data": False,
            "uses_synthetic_buildings_for_release": False,
            "requires_authorized_release_unit_for_promotion": True,
        },
        "next_required_proof": (
            "The non-release CityLearn adapter now proves native storage effects "
            "and deterministic simulator-owned time. Next formalize a release "
            "backend contract with scorer-consumed evidence, visible response "
            "windows, and scorer-consumed masked-action replay before any "
            "materialization."
        ),
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    args = parser.parse_args(argv)

    report = build_citylearn_source_preflight_report(source_root=args.source_root)
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
