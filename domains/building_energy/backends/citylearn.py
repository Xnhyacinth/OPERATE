"""Native CityLearn backend for the non-release Building Energy pilot."""

from __future__ import annotations

import builtins
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..seeds.schema import BuildingEnergyScenarioSeed

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT
    / "works"
    / "CityLearn"
    / "data"
    / "datasets"
    / "citylearn_challenge_2022_phase_3"
)
DEFAULT_SOURCE_LOCK = (
    REPO_ROOT
    / "sources"
    / "locks"
    / "citylearn_challenge_2022_phase_3.json"
)
DEFAULT_IMPLEMENTATION_ROOT = REPO_ROOT / "works" / "CityLearn"
SOURCE_CHANNELS = (
    "building_timeseries",
    "weather",
    "pricing",
    "carbon_intensity",
)
SOURCE_EVENT_PEAK_RESPONSE_BURDEN_V1 = "source_event_peak_response_burden_v1"
CITYLEARN_NATIVE_EVENT_REGISTRY = MappingProxyType(
    {
        "load_change": (
            frozenset({"building_timeseries.non_shiftable_load"}),
            "alarm",
            True,
        ),
        "generation_change": (
            frozenset({"building_timeseries.solar_generation"}),
            "alarm",
            True,
        ),
        "tariff_change": (
            frozenset({"pricing.electricity_pricing"}),
            "alarm",
            True,
        ),
        "carbon_intensity_change": (
            frozenset({"carbon_intensity.carbon_intensity"}),
            "alarm",
            True,
        ),
    }
)
_RUNTIME_OPEN_LOCK = threading.RLock()


class CityLearnSourceLockError(RuntimeError):
    """Raised when a CityLearn runtime asset differs from its lock."""


@dataclass
class CityLearnTickRecord:
    tick: int
    simulator_time_step: int
    reward: float
    energy_cost: float
    carbon_emissions: float
    district_net_electricity_consumption: float
    source_event_peak_response_burden: float
    native_storage_charging_burden: float
    done: bool
    action_vector: list[float]
    source_state_effect_observed: bool
    control_state_effect_observed: bool
    source_consumed: bool
    realized_events: list[dict[str, Any]]

    @property
    def state_effect_observed(self) -> bool:
        """Keep the existing adapter field explicitly control-scoped."""

        return self.control_state_effect_observed


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def electrical_storage_action_indices(env: Any) -> list[int]:
    """Map each building to its native central-agent electrical_storage index.

    CityLearn 2022 exposes one storage action per building. Later challenges
    concatenate DHW/cooling/device actions, so action width is not the
    building count. The Building Energy tools only dispatch electrical
    storage; other native actions stay at zero.
    """

    names = getattr(env, "action_names", None)
    if isinstance(names, list) and names and isinstance(names[0], list):
        names = names[0]
    buildings = list(getattr(env, "buildings", []) or [])
    if isinstance(names, list) and names:
        indices = [
            index
            for index, name in enumerate(names)
            if str(name) == "electrical_storage"
        ]
        if len(indices) != len(buildings):
            raise CityLearnSourceLockError(
                "CityLearn electrical_storage action indices do not match building inventory"
            )
        return indices
    width = int(env.action_space[0].shape[0])
    if width != len(buildings):
        raise CityLearnSourceLockError(
            "CityLearn central action width does not match building inventory"
        )
    return list(range(width))


def _resolve_repo_path(value: str | Path, *, default: Path) -> Path:
    path = Path(value) if str(value).strip() else default
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


_SCHEMA_RUNTIME_SUFFIXES = {".csv", ".epw", ".json", ".pkl", ".pth"}


def _collect_local_schema_files(
    value: Any,
    source_root: Path,
    names: dict[str, Path],
    *,
    document: Path,
    visited_json: set[Path],
) -> None:
    """Collect the recursive schema graph using dataset-relative paths."""

    if isinstance(value, dict):
        for item in value.values():
            _collect_local_schema_files(
                item,
                source_root,
                names,
                document=document,
                visited_json=visited_json,
            )
        return
    if isinstance(value, list):
        for item in value:
            _collect_local_schema_files(
                item,
                source_root,
                names,
                document=document,
                visited_json=visited_json,
            )
        return
    if not isinstance(value, str):
        return
    if Path(value).suffix.lower() not in _SCHEMA_RUNTIME_SUFFIXES:
        return
    path = (document.parent / value).resolve()
    try:
        relative = path.relative_to(source_root).as_posix()
    except ValueError as exc:
        raise CityLearnSourceLockError(
            f"CityLearn schema asset escapes dataset root: {value}"
        ) from exc
    names[relative] = path
    if not path.is_file():
        return
    if path.suffix.lower() != ".json" or path in visited_json:
        return
    visited_json.add(path)
    try:
        nested = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CityLearnSourceLockError(
            f"invalid CityLearn schema asset {relative}: {exc}"
        ) from exc
    _collect_local_schema_files(
        nested,
        source_root,
        names,
        document=path,
        visited_json=visited_json,
    )


def _runtime_source_files(source_root: Path) -> dict[str, Path]:
    source_root = source_root.resolve()
    schema_path = source_root / "schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CityLearnSourceLockError(f"invalid CityLearn schema: {exc}") from exc
    names: dict[str, Path] = {"schema.json": schema_path}
    _collect_local_schema_files(
        schema,
        source_root,
        names,
        document=schema_path,
        visited_json={schema_path},
    )
    # EPW files initialize PV sizing through PySAM's native reader. They are
    # locked inputs, but not per-tick runtime streams opened by CityLearn's
    # Python I/O path.
    names = {
        name: path for name, path in names.items() if path.suffix.lower() != ".epw"
    }
    return names


def _derivation_source_files(source_root: Path) -> dict[str, Path]:
    misc_root = source_root.parent.parent / "misc"
    files = {
        name: misc_root / name
        for name in ("lbl-tracking_the_sun-res-pv.csv", "battery_choices.yaml")
    }
    epw = source_root.resolve() / "weather.epw"
    if epw.is_file():
        files[epw.name] = epw
    return files


def _replay_source_files(source_root: Path) -> dict[str, Path]:
    """Return locked files that must accompany a replay-root copy."""

    source_root = source_root.resolve()
    return {
        **_runtime_source_files(source_root),
        **{
            name: path
            for name, path in _derivation_source_files(source_root).items()
            if path.is_relative_to(source_root)
        },
    }


def _locked_derivation_source_files(
    declared: dict[str, Any], source_root: Path
) -> dict[str, Path]:
    """Resolve non-runtime derivation assets from their pinned lock paths."""

    resolved: dict[str, Path] = {}
    defaults = _derivation_source_files(source_root)
    for name, default in defaults.items():
        matches = [
            _locked_path(str(raw_path))
            for raw_path in declared
            if Path(str(raw_path)).name == name
            and not _locked_path(str(raw_path)).is_relative_to(source_root)
        ]
        if len(matches) != 1:
            resolved[name] = default
        else:
            resolved[name] = matches[0]
    return resolved


def _git_output(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CityLearnSourceLockError(
            f"CityLearn implementation checkout inspection failed: {exc}"
        ) from exc
    return result.stdout.strip()


def _locked_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _read_mode(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    mode = kwargs.get("mode")
    if mode is None and args:
        mode = args[0]
    mode = str(mode or "r")
    return not any(flag in mode for flag in ("w", "a", "x", "+"))


@contextmanager
def _capture_runtime_opens(
    source_root: Path, phase: str
) -> Iterator[dict[Path, set[str]]]:
    """Capture source files CityLearn actually opens during native execution."""

    opened: dict[Path, set[str]] = {}
    owner_thread = threading.get_ident()

    def record(file: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        if threading.get_ident() != owner_thread or not _read_mode(args, kwargs):
            return
        try:
            path = Path(os.fspath(file)).resolve()
        except (TypeError, ValueError, OSError):
            return
        if path.is_relative_to(source_root):
            opened.setdefault(path, set()).add(phase)

    def builtin_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        record(file, args, kwargs)
        return original_builtin_open(file, *args, **kwargs)

    def io_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        record(file, args, kwargs)
        return original_io_open(file, *args, **kwargs)

    # ``open`` is process-global while baseline workers are threads.  Serialize
    # the small reset/replay sections and attribute reads only to their owner
    # thread so one episode cannot claim another episode's source graph.
    with _RUNTIME_OPEN_LOCK:
        original_builtin_open = builtins.open
        original_io_open = io.open
        builtins.open = builtin_open
        io.open = io_open
        try:
            yield opened
        finally:
            builtins.open = original_builtin_open
            io.open = original_io_open


def _merge_open_trace(
    target: dict[Path, set[str]], source: dict[Path, set[str]]
) -> None:
    for path, phases in source.items():
        target.setdefault(path, set()).update(phases)


def _runtime_channel(asset_name: str) -> str:
    basename = Path(asset_name).name
    if basename == "schema.json":
        return "schema"
    if basename == "weather.csv":
        return "weather"
    if basename == "pricing.csv":
        return "pricing"
    if basename == "carbon_intensity.csv":
        return "carbon_intensity"
    if Path(asset_name).suffix.lower() == ".pth":
        return "building_dynamics"
    if basename.startswith("charger_"):
        return "charger_simulation"
    return "building_timeseries"


def _runtime_open_evidence(
    source_lock: dict[str, Any], open_trace: dict[Path, set[str]]
) -> dict[str, Any]:
    expected = source_lock.get("runtime_files", {})
    expected = expected if isinstance(expected, dict) else {}
    required_assets: list[dict[str, Any]] = []
    opened_assets: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    missing_assets: list[str] = []
    for name, raw in sorted(expected.items()):
        if not isinstance(raw, dict) or not raw.get("path"):
            missing_assets.append(str(name))
            continue
        path = _locked_path(str(raw["path"]))
        expected_paths.add(path)
        required = {
            "asset_name": str(name),
            "path": str(raw["path"]),
            "sha256": raw.get("sha256"),
            "channel": _runtime_channel(str(name)),
        }
        required_assets.append(required)
        phases = sorted(open_trace.get(path, set()))
        if phases:
            opened_assets.append(
                {
                    **required,
                    "opened_by_citylearn": True,
                    "open_phases": phases,
                }
            )
        else:
            missing_assets.append(str(name))
    unexpected = sorted(
        str(path.relative_to(REPO_ROOT))
        if path.is_relative_to(REPO_ROOT)
        else str(path)
        for path in open_trace
        if path not in expected_paths
    )
    complete = bool(required_assets) and not missing_assets and not unexpected
    return {
        "required_runtime_assets": required_assets,
        "runtime_opened_assets": opened_assets,
        "missing_runtime_assets": sorted(missing_assets),
        "unexpected_runtime_opened_assets": unexpected,
        "runtime_opened_assets_complete": complete,
        "runtime_open_graph_hash": _stable_hash(opened_assets),
    }


def _verify_source_lock(
    source_root: Path,
    source_lock: Path,
    implementation_root: Path = DEFAULT_IMPLEMENTATION_ROOT,
) -> dict[str, Any]:
    try:
        sidecar = json.loads(source_lock.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CityLearnSourceLockError(f"invalid CityLearn source lock: {exc}") from exc
    declared = sidecar.get("files") if isinstance(sidecar.get("files"), dict) else {}
    runtime_files = _runtime_source_files(source_root)
    optional_names = {"pricing.csv", "carbon_intensity.csv"}
    declared_optional_absent = set(sidecar.get("optional_runtime_assets_absent") or [])
    derivation_files = _locked_derivation_source_files(declared, source_root)
    files = {**runtime_files, **derivation_files}
    mismatches: list[str] = []
    fingerprints: dict[str, dict[str, Any]] = {}
    for name, path in files.items():
        digest = _sha256(path)
        try:
            ref = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            ref = str(path)
        fingerprints[name] = {"path": ref, "sha256": digest}
        if digest is None:
            mismatches.append(f"missing:{name}")
        if declared.get(ref) != digest:
            mismatches.append(f"hash:{name}")

    def fingerprint_for(asset_name: str) -> dict[str, Any] | None:
        exact = fingerprints.get(asset_name)
        if exact is not None:
            return exact
        matches = [
            row
            for relative, row in fingerprints.items()
            if Path(relative).name == asset_name
        ]
        return matches[0] if len(matches) == 1 else None

    package_spec = str(sidecar.get("package_version") or "")
    torch_spec = str(sidecar.get("torch_version") or "")
    if package_spec != "citylearn==2.5.0":
        mismatches.append("citylearn_version_lock")
    if torch_spec != "torch==2.13.0":
        mismatches.append("torch_version_lock")
    runtime_identity = _verify_runtime_identity(
        implementation_root=implementation_root,
        package_spec=package_spec,
        torch_spec=torch_spec,
        revision=str(sidecar.get("git_commit_or_release_tag") or ""),
        pinned_tree_sha256=str(sidecar.get("implementation_tree_sha256") or ""),
    )
    for field, name in (
        ("schema_sha256", "schema.json"),
        ("weather_file_or_timeseries_lock", "weather.csv"),
        ("pricing_file_sha256", "pricing.csv"),
        ("carbon_intensity_file_sha256", "carbon_intensity.csv"),
        ("pv_sizing_file_sha256", "lbl-tracking_the_sun-res-pv.csv"),
        ("battery_sizing_file_sha256", "battery_choices.yaml"),
    ):
        fingerprint = fingerprint_for(name)
        if fingerprint is None:
            if (
                name in optional_names
                and name in declared_optional_absent
                and sidecar.get(field) is None
            ):
                continue
            mismatches.append(f"lock_field:{field}")
            continue
        if (
            name in optional_names
            and fingerprint["sha256"] is None
            and name in declared_optional_absent
            and sidecar.get(field) is None
        ):
            continue
        if sidecar.get(field) != f"sha256:{fingerprint['sha256']}":
            mismatches.append(f"lock_field:{field}")
    if mismatches:
        raise CityLearnSourceLockError(
            "CityLearn source lock mismatch: " + ", ".join(sorted(set(mismatches)))
        )
    return {
        "path": source_lock.relative_to(REPO_ROOT).as_posix()
        if source_lock.is_relative_to(REPO_ROOT)
        else str(source_lock),
        "sha256": _sha256(source_lock),
        "source_id": sidecar.get("source_id"),
        "package_version": package_spec,
        "torch_version": torch_spec,
        "runtime_identity": runtime_identity,
        "files": fingerprints,
        "runtime_files": {
            name: fingerprints[name]
            for name in runtime_files
            if fingerprints[name]["sha256"] is not None
        },
        "derivation_files": {
            name: fingerprints[name] for name in derivation_files
        },
    }


def _verify_runtime_identity(
    *,
    implementation_root: Path,
    package_spec: str,
    torch_spec: str,
    revision: str,
    pinned_tree_sha256: str,
) -> dict[str, Any]:
    """Bind the imported simulator implementation to the locked checkout."""

    import citylearn
    import torch

    expected_citylearn = package_spec.partition("==")[2]
    expected_torch = torch_spec.partition("==")[2]
    runtime_citylearn = str(getattr(citylearn, "__version__", ""))
    runtime_torch = str(getattr(torch, "__version__", "")).split("+", 1)[0]
    mismatches: list[str] = []
    if not expected_citylearn or runtime_citylearn != expected_citylearn:
        mismatches.append("runtime_citylearn_version")
    if not expected_torch or runtime_torch != expected_torch:
        mismatches.append("runtime_torch_version")

    checkout_root = implementation_root.resolve()
    checkout_package = checkout_root / "citylearn"
    runtime_init = Path(str(getattr(citylearn, "__file__", ""))).resolve()
    runtime_package = runtime_init.parent
    expected_revision = revision.rpartition("@")[2]
    checkout_revision = _git_output(checkout_root, "rev-parse", "HEAD")
    checkout_dirty = _git_output(
        checkout_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if (
        not expected_revision
        or checkout_revision != expected_revision
        or not (checkout_root / "LICENSE").is_file()
    ):
        mismatches.append("runtime_citylearn_checkout")
    if checkout_dirty:
        mismatches.append("runtime_citylearn_checkout_dirty")

    checkout_graph = {
        path.relative_to(checkout_package).as_posix(): _sha256(path)
        for path in sorted(checkout_package.rglob("*.py"))
    }
    runtime_graph = {
        relative: _sha256(runtime_package / relative)
        for relative in checkout_graph
        if (runtime_package / relative).is_file()
    }
    implementation_graph_sha256 = _stable_hash(checkout_graph)
    if (
        len(pinned_tree_sha256) != 64
        or pinned_tree_sha256 != implementation_graph_sha256
    ):
        mismatches.append("runtime_citylearn_implementation_lock")
    if (
        not runtime_init.is_file()
        or not checkout_graph
        or runtime_graph != checkout_graph
    ):
        mismatches.append("runtime_citylearn_implementation")
    if mismatches:
        raise CityLearnSourceLockError(
            "CityLearn runtime identity mismatch: "
            + ", ".join(sorted(set(mismatches)))
        )
    return {
        "citylearn_version": runtime_citylearn,
        "torch_version": runtime_torch,
        "checkout_revision": checkout_revision,
        "implementation_graph_sha256": implementation_graph_sha256,
        "pinned_implementation_graph_sha256": pinned_tree_sha256,
        "checkout_clean": not checkout_dirty,
        "implementation_file_count": len(checkout_graph),
    }


def _current_value(building: Any, attribute: str, index: int) -> float:
    values = np.asarray(getattr(building, attribute), dtype=float).reshape(-1)
    if not len(values):
        return 0.0
    return float(values[min(max(index, 0), len(values) - 1)])


def _clear_reset_priming(env: Any) -> dict[str, Any]:
    """Remove CityLearn's timestep-zero baseline device priming before step.

    CityLearn initializes native device electricity consumption from the
    locked source loads during ``reset``.  Its first ``step`` then applies the
    same timestep-zero loads again, so ``available_nominal_power`` can be
    reduced twice before the native demand-capacity check.  Clear only that
    derived reset-time accumulator through CityLearn's public device API.  The
    source demand arrays and device capacities are intentionally untouched and
    the first native step recomputes the same loads exactly once.
    """

    if int(getattr(env, "time_step", -1)) != 0:
        return {
            "applied": False,
            "cleared_device_count": 0,
            "cleared_consumption": 0.0,
        }
    cleared_device_count = 0
    cleared_consumption = 0.0
    for building in list(getattr(env, "buildings", []) or []):
        for name in (
            "cooling_device",
            "heating_device",
            "dhw_device",
            "non_shiftable_load_device",
        ):
            device = getattr(building, name, None)
            if device is None:
                continue
            values = np.asarray(
                getattr(device, "electricity_consumption", []), dtype=float
            ).reshape(-1)
            index = int(getattr(device, "time_step", 0))
            if not 0 <= index < len(values):
                continue
            consumption = float(values[index])
            if not np.isfinite(consumption) or abs(consumption) <= 1e-12:
                continue
            device.update_electricity_consumption(
                -consumption,
                enforce_polarity=False,
            )
            cleared_device_count += 1
            cleared_consumption += consumption
    return {
        "applied": bool(cleared_device_count),
        "cleared_device_count": cleared_device_count,
        "cleared_consumption": cleared_consumption,
    }


def _native_tick_metrics(
    buildings: list[Any],
    index: int,
    *,
    peak_response_objective_enabled: bool = False,
    peak_response_active: bool = False,
) -> dict[str, float]:
    district_net_load = float(
        sum(
            _current_value(building, "net_electricity_consumption", index)
            for building in buildings
        )
    )
    return {
        "energy_cost": float(
            sum(
                _current_value(
                    building, "net_electricity_consumption_cost", index
                )
                for building in buildings
            )
        ),
        "carbon_emissions": float(
            sum(
                _current_value(
                    building, "net_electricity_consumption_emission", index
                )
                for building in buildings
            )
        ),
        "district_net_electricity_consumption": district_net_load,
        "source_event_peak_response_burden": (
            max(0.0, district_net_load) ** 2 / max(1, len(buildings))
            if peak_response_objective_enabled and peak_response_active
            else 0.0
        ),
        "native_storage_charging_burden": (
            float(
                sum(
                    max(
                        0.0,
                        _current_value(
                            building.electrical_storage, "energy_balance", index
                        ),
                    )
                    for building in buildings
                )
            )
            if peak_response_objective_enabled
            else 0.0
        ),
    }


def _source_channel_changes(
    building: Any, current: int, following: int
) -> dict[str, bool]:
    source_channels = {
        "building_timeseries": (
            (building.energy_simulation, "non_shiftable_load"),
            (building.energy_simulation, "solar_generation"),
        ),
        "weather": ((building.weather, "outdoor_dry_bulb_temperature"),),
        "pricing": ((building.pricing, "electricity_pricing"),),
        "carbon_intensity": (
            (building.carbon_intensity, "carbon_intensity"),
        ),
    }
    return {
        channel: any(
            abs(
                _current_value(owner, field, current)
                - _current_value(owner, field, following)
            )
            > 1e-8
            for owner, field in fields
        )
        for channel, fields in source_channels.items()
    }


class CityLearnBackend:
    """Wrap CityLearnEnv without letting it own benchmark time or tools."""

    backend_kind = "citylearn"
    supported_tool_names = frozenset(
        {
            "inspect_building_state",
            "set_storage_dispatch",
        }
    )

    def __init__(self) -> None:
        self._env: Any = None
        self._seed: BuildingEnergyScenarioSeed | None = None
        self._source_lock: dict[str, Any] = {}
        self._runtime_open_trace: dict[Path, set[str]] = {}
        self._source_channel_effect_ticks: dict[str, list[int]] = {
            channel: [] for channel in SOURCE_CHANNELS
        }
        self._native_source_events: list[dict[str, Any]] = []
        self._task_response_windows: list[dict[str, Any]] = []
        self._realized_named_source_events: dict[str, dict[str, Any]] = {}
        self._deterministic_replay_evidence: dict[str, Any] | None = None
        self._bounded_source_probe_evidence: dict[str, Any] | None = None
        self._source_ablation_proofs: list[dict[str, Any]] = []
        self._buildings: list[str] = []
        self._storage_indices: list[int] = []
        self._action_vector = np.zeros(0, dtype=np.float32)
        self._source_consumption_ticks: list[int] = []
        self._source_state_effect_observed = False
        self._initial_source_state_digest = ""
        self._post_source_state_digests: list[str] = []
        self._state_effect_ticks: list[int] = []
        self._applied_controls: list[dict[str, Any]] = []
        self._pending_control_evidence: dict[str, dict[str, Any]] = {}
        self._records: list[CityLearnTickRecord] = []
        self._last_reward = 0.0

    def reset(self, seed_obj: BuildingEnergyScenarioSeed) -> None:
        source_root = _resolve_repo_path(seed_obj.source_root, default=DEFAULT_SOURCE_ROOT)
        source_lock = _resolve_repo_path(seed_obj.source_lock, default=DEFAULT_SOURCE_LOCK)
        config = dict(seed_obj.backend_config)
        objective = config.get("native_peak_response_objective")
        if objective not in {None, "", SOURCE_EVENT_PEAK_RESPONSE_BURDEN_V1}:
            raise CityLearnSourceLockError(
                f"unsupported CityLearn native peak-response objective: {objective}"
            )
        implementation_root = _resolve_repo_path(
            config.get("implementation_checkout_root", ""),
            default=DEFAULT_IMPLEMENTATION_ROOT,
        )
        self._source_lock = _verify_source_lock(
            source_root,
            source_lock,
            implementation_root,
        )
        self._runtime_open_trace = {}
        with _capture_runtime_opens(source_root, "live_reset") as opened:
            self._env = self._new_env(seed_obj)
            self._env.reset(seed=seed_obj.seed)
            _clear_reset_priming(self._env)
        _merge_open_trace(self._runtime_open_trace, opened)
        self._seed = seed_obj
        self._buildings = [str(building.name) for building in self._env.buildings]
        width = int(self._env.action_space[0].shape[0])
        self._storage_indices = electrical_storage_action_indices(self._env)
        if max(self._storage_indices, default=-1) >= width:
            raise CityLearnSourceLockError(
                "CityLearn electrical_storage action index exceeds native action width"
            )
        self._action_vector = np.zeros(width, dtype=np.float32)
        self._native_source_events = [
            dict(event)
            for event in config.get("native_source_events") or []
            if isinstance(event, dict)
        ]
        task_contract = config.get("task_contract") or {}
        self._task_response_windows = [
            dict(window)
            for window in task_contract.get("response_windows") or []
            if isinstance(window, dict)
        ]
        self._verify_native_event_contracts()
        self._source_consumption_ticks = []
        self._source_state_effect_observed = False
        self._initial_source_state_digest = self._native_state_digest(
            self.current_time_step
        )
        self._post_source_state_digests = []
        self._source_channel_effect_ticks = {
            channel: [] for channel in SOURCE_CHANNELS
        }
        self._realized_named_source_events = {}
        self._deterministic_replay_evidence = None
        self._bounded_source_probe_evidence = None
        self._source_ablation_proofs = []
        self._state_effect_ticks = []
        self._applied_controls = []
        self._pending_control_evidence = {}
        self._records = []
        self._last_reward = 0.0

    def _verify_native_event_contracts(self) -> None:
        """Reject source events that are not exact locked native transitions."""

        locked_by_path = {
            str(row.get("path")): str(row.get("sha256"))
            for row in self._source_lock.get("files", {}).values()
            if isinstance(row, dict)
        }
        seen: set[str] = set()
        for event in self._native_source_events:
            event_id = str(event.get("event_id") or "")
            kind = str(event.get("kind") or "")
            channel = str(event.get("channel") or "")
            trigger = int(event.get("trigger_tick") or -1)
            source_asset = str(event.get("source_asset") or "")
            event_contract = CITYLEARN_NATIVE_EVENT_REGISTRY.get(kind)
            if event_contract is None:
                raise CityLearnSourceLockError(
                    f"unsupported CityLearn native event kind: {kind or '<missing>'}"
                )
            allowed_channels, event_class, _ = event_contract
            if channel not in allowed_channels:
                raise CityLearnSourceLockError(
                    "CityLearn native event kind/channel mismatch: "
                    f"{kind}/{channel or '<missing>'}"
                )
            declared_class = event.get("event_class", event.get("class"))
            if declared_class not in {None, "", event_class}:
                raise CityLearnSourceLockError(
                    "CityLearn native event class does not match registry: "
                    f"{kind}/{declared_class}"
                )
            if (
                not event_id
                or event_id in seen
                or trigger <= 0
                or bool(event.get("procedural_overlay", False))
                or event.get("source_observed") is not True
                or locked_by_path.get(source_asset)
                != str(event.get("source_asset_sha256") or "")
            ):
                raise CityLearnSourceLockError(
                    f"CityLearn native event is not source-bound: {event_id or '<missing>'}"
                )
            seen.add(event_id)
            owner, field = self._native_source_owner(channel, source_asset)
            local_trigger = self._event_local_tick(trigger)
            before = _current_value(owner, field, local_trigger - 1)
            after = _current_value(owner, field, local_trigger)
            if (
                not np.isclose(
                    before,
                    float(event.get("source_value_before")),
                    rtol=1e-6,
                    atol=1e-6,
                )
                or not np.isclose(
                    after,
                    float(event.get("source_value_after")),
                    rtol=1e-6,
                    atol=1e-6,
                )
                or abs(after - before) <= 1e-8
            ):
                raise CityLearnSourceLockError(
                    f"CityLearn native event value mismatch: {event_id}"
                )
            materiality_metric = str(event.get("materiality_metric") or "")
            materiality_threshold = float(
                event.get("materiality_threshold") or 0.0
            )
            if (
                not materiality_metric
                or not np.isfinite(materiality_threshold)
                or materiality_threshold <= 0.0
                or abs(after - before) < materiality_threshold
            ):
                raise CityLearnSourceLockError(
                    f"CityLearn native event is not materially bounded: {event_id}"
                )
        window_ids = {
            str(window.get("event_id") or "")
            for window in self._task_response_windows
        }
        if self._native_source_events and not seen.issubset(window_ids):
            raise CityLearnSourceLockError(
                "CityLearn native event lacks a response-window contract"
            )

    def _event_local_tick(self, source_tick: int) -> int:
        start = int(
            (self._seed.backend_config if self._seed is not None else {}).get(
                "simulation_start_time_step"
            )
            or 0
        )
        return int(source_tick) - start

    def _native_source_owner(
        self, channel: str, source_asset: str
    ) -> tuple[Any, str]:
        if not self._env.buildings:
            raise CityLearnSourceLockError("CityLearn native building inventory is empty")
        owner_name, separator, field = channel.partition(".")
        if not separator or owner_name not in {
            "building_timeseries",
            "weather",
            "pricing",
            "carbon_intensity",
        }:
            raise CityLearnSourceLockError(
                f"unsupported CityLearn native event channel: {channel}"
            )
        source_filename = Path(source_asset).name
        if owner_name == "building_timeseries":
            building = next(
                (
                    candidate
                    for candidate in self._env.buildings
                    if f"{candidate.name}.csv" == source_filename
                ),
                None,
            )
            if building is None:
                raise CityLearnSourceLockError(
                    "CityLearn native event source asset owner mismatch: "
                    f"{source_asset}"
                )
        else:
            expected_filename = {
                "weather": "weather.csv",
                "pricing": "pricing.csv",
                "carbon_intensity": "carbon_intensity.csv",
            }[owner_name]
            if source_filename != expected_filename:
                raise CityLearnSourceLockError(
                    "CityLearn native event source asset owner mismatch: "
                    f"{source_asset}"
                )
            building = self._env.buildings[0]
        owner = (
            building.energy_simulation
            if owner_name == "building_timeseries"
            else getattr(building, owner_name)
        )
        if not hasattr(owner, field):
            raise CityLearnSourceLockError(
                f"missing CityLearn native event field: {channel}"
            )
        return owner, field

    def _named_events_at(self, source_tick: int, tick: int) -> list[dict[str, Any]]:
        realized: list[dict[str, Any]] = []
        for event in self._native_source_events:
            if self._event_local_tick(int(event["trigger_tick"])) != source_tick + 1:
                continue
            event_id = str(event["event_id"])
            source_value_before = float(event["source_value_before"])
            source_value_after = float(event["source_value_after"])
            source_channel = str(event["channel"])
            _, event_class, actionable = CITYLEARN_NATIVE_EVENT_REGISTRY[
                str(event["kind"])
            ]
            materiality_value = abs(source_value_after - source_value_before)
            materiality_threshold = float(event["materiality_threshold"])
            response_window = next(
                window
                for window in self._task_response_windows
                if str(window.get("event_id") or "") == event_id
            )
            event_evidence_id = (
                "citylearn_source_event:"
                + _stable_hash(
                    {
                        "event_id": event_id,
                        "source_asset_sha256": event["source_asset_sha256"],
                        "trigger_tick": event["trigger_tick"],
                    }
                )[:20]
            )
            payload = {
                "event_id": event_id,
                "type": str(event["kind"]),
                "event_class": event_class,
                "origin": "source_trace",
                "tick": int(tick),
                "source_time_step": int(event["trigger_tick"]),
                "observed_after_control_tick": int(tick),
                "hidden": bool(event.get("hidden", False)),
                "decision_required": actionable,
                "actionable": actionable,
                "source_consumed": True,
                "state_effect_observed": True,
                "source_state_effect_observed": True,
                "control_state_effect_observed": False,
                "source_channel": source_channel,
                "source_asset": str(event["source_asset"]),
                "source_asset_sha256": str(event["source_asset_sha256"]),
                "source_evidence_key": f"native-event:{event_id}",
                "event_evidence_id": event_evidence_id,
                "evidence_ids": [event_evidence_id],
                "changed_state_fields": [source_channel],
                "before_state_digest": _stable_hash(
                    {
                        "channel": source_channel,
                        "row": event.get("source_row_before"),
                        "value": source_value_before,
                    }
                ),
                "after_state_digest": _stable_hash(
                    {
                        "channel": source_channel,
                        "row": event.get("source_row_after"),
                        "value": source_value_after,
                    }
                ),
                "materiality_metric": str(event["materiality_metric"]),
                "materiality_value": materiality_value,
                "materiality_threshold": materiality_threshold,
                "materiality_passed": materiality_value >= materiality_threshold,
                "response_window_required": True,
                "response_opportunity_tick": int(response_window["first_tick"]),
                "terminal_response_window_missing": False,
            }
            self._realized_named_source_events[event_id] = payload
            realized.append(payload)
        return realized

    @staticmethod
    def _new_env(
        seed_obj: BuildingEnergyScenarioSeed,
        *,
        source_root_override: Path | None = None,
    ) -> Any:
        """Construct a fresh native environment for deterministic replay.

        CityLearn owns the physical state inside each environment instance. A
        counterfactual therefore gets a new instance with the same locked
        source graph, configuration and seed instead of mutating the live
        episode that the agent is controlling.
        """

        from citylearn.citylearn import CityLearnEnv

        source_root = source_root_override or _resolve_repo_path(
            seed_obj.source_root,
            default=DEFAULT_SOURCE_ROOT,
        )
        config = dict(seed_obj.backend_config)
        start = int(config.get("simulation_start_time_step") or 0)
        end = max(int(config.get("simulation_end_time_step") or start + 47), start)
        simulation_span = end - start + 1
        episode_steps = int(config.get("episode_time_steps") or simulation_span)
        if not 1 <= episode_steps <= simulation_span:
            raise CityLearnSourceLockError(
                "CityLearn episode_time_steps must fit the configured simulation span"
            )
        return CityLearnEnv(
            schema=source_root / "schema.json",
            root_directory=source_root,
            central_agent=True,
            simulation_start_time_step=start,
            simulation_end_time_step=end,
            episode_time_steps=episode_steps,
            random_seed=seed_obj.seed,
            render_mode="none",
        )

    def _replay_stream(
        self,
        actions: list[list[float]],
        action_masks: list[list[bool]],
        source_ablation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._seed is None:
            raise RuntimeError("CityLearn backend is not reset")
        source_root = _resolve_repo_path(
            self._seed.source_root, default=DEFAULT_SOURCE_ROOT
        )
        source_lock = _resolve_repo_path(
            self._seed.source_lock, default=DEFAULT_SOURCE_LOCK
        )
        implementation_root = _resolve_repo_path(
            self._seed.backend_config.get("implementation_checkout_root", ""),
            default=DEFAULT_IMPLEMENTATION_ROOT,
        )
        # Re-verify the complete source graph for every independent replay so
        # a file mutation between the live episode and counterfactual cannot
        # silently invalidate deterministic evidence.
        _verify_source_lock(source_root, source_lock, implementation_root)
        with ExitStack() as stack:
            execution_source_root = source_root
            ablation: dict[str, Any] | None = None
            if source_ablation is not None:
                execution_source_root = Path(
                    stack.enter_context(tempfile.TemporaryDirectory())
                ).resolve()
                for name, path in _replay_source_files(source_root).items():
                    target = execution_source_root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                channel = str(source_ablation["channel"])
                _, _, field = channel.partition(".")
                trigger = int(source_ablation["trigger_tick"])
                source_asset = _locked_path(
                    str(source_ablation["source_asset"])
                )
                try:
                    relative_source_asset = source_asset.relative_to(source_root)
                except ValueError as exc:
                    raise CityLearnSourceLockError(
                        "CityLearn source ablation asset escapes dataset root"
                    ) from exc
                target = execution_source_root / relative_source_asset
                with target.open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    fieldnames = list(reader.fieldnames or [])
                    rows = list(reader)
                if field not in fieldnames or not 0 <= trigger < len(rows):
                    raise CityLearnSourceLockError(
                        f"CityLearn source ablation target is invalid: {channel}"
                    )
                original = float(rows[trigger][field])
                ablated = float(source_ablation["source_value_before"])
                rows[trigger][field] = str(ablated)
                with target.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                ablation = {
                    "event_id": str(source_ablation["event_id"]),
                    "channel": channel,
                    "trigger_tick": trigger,
                    "original_value": original,
                    "ablated_value": ablated,
                }
            opened = stack.enter_context(
                _capture_runtime_opens(execution_source_root, "masked_replay")
            )
            replay_env = self._new_env(
                self._seed,
                source_root_override=execution_source_root,
            )
            try:
                replay_env.reset(seed=self._seed.seed)
                _clear_reset_priming(replay_env)
                width = int(replay_env.action_space[0].shape[0])
                if len(actions) != len(action_masks):
                    raise ValueError(
                        "action stream and masks must have the same length"
                    )
                rows: list[dict[str, Any]] = []
                rewards: list[float] = []
                state_effect_observed = False
                action_count = 0
                for tick, (values, mask) in enumerate(
                    zip(actions, action_masks, strict=True)
                ):
                    if len(values) != width or len(mask) != width:
                        raise ValueError(
                            "CityLearn action stream width does not match runtime"
                        )
                    action = np.asarray(
                        [
                            float(value) if bool(enabled) else 0.0
                            for value, enabled in zip(values, mask, strict=True)
                        ],
                        dtype=np.float32,
                    )
                    source_tick = int(getattr(replay_env, "time_step", tick))
                    source_channel_changes = {
                        channel: False for channel in SOURCE_CHANNELS
                    }
                    for building in replay_env.buildings:
                        for channel, changed in _source_channel_changes(
                            building, source_tick, source_tick + 1
                        ).items():
                            source_channel_changes[channel] |= changed
                    before_balance = [
                        _current_value(
                            building.electrical_storage,
                            "energy_balance",
                            source_tick,
                        )
                        for building in replay_env.buildings
                    ]
                    next_observation, reward, terminated, truncated, _ = (
                        replay_env.step([action])
                    )
                    after_balance = [
                        _current_value(
                            building.electrical_storage,
                            "energy_balance",
                            source_tick,
                        )
                        for building in replay_env.buildings
                    ]
                    state_effect = any(
                        abs(after - before) > 1e-8
                        for before, after in zip(
                            before_balance, after_balance, strict=True
                        )
                    )
                    state_effect_observed |= state_effect
                    if bool(np.any(np.abs(action) > 1e-8)):
                        action_count += 1
                    reward_values = (
                        np.asarray(reward, dtype=float).reshape(-1).tolist()
                    )
                    rewards.extend(float(value) for value in reward_values)
                    tick_metrics = _native_tick_metrics(
                        list(replay_env.buildings),
                        source_tick,
                        peak_response_objective_enabled=(
                            self._peak_response_objective is not None
                        ),
                        peak_response_active=self._peak_response_active(tick),
                    )
                    native_outputs = {
                        "net_electricity_consumption": float(
                            sum(
                                _current_value(
                                    building,
                                    "net_electricity_consumption",
                                    source_tick,
                                )
                                for building in replay_env.buildings
                            )
                        ),
                        "storage_energy_balance": float(
                            sum(after_balance)
                        ),
                    }
                    native_state_digest = _stable_hash(
                        {
                            **native_outputs,
                            **tick_metrics,
                            "reward": reward_values,
                        }
                    )
                    rows.append(
                        {
                            "tick": tick,
                            "simulator_time_step": int(
                                getattr(replay_env, "time_step", tick + 1)
                            ),
                            "action_vector": action.tolist(),
                            "action_mask": [bool(value) for value in mask],
                            "observation_hash": _stable_hash(
                                np.asarray(next_observation).tolist()
                            ),
                            "state_effect_observed": state_effect,
                            "source_state_effect_observed": any(
                                source_channel_changes.values()
                            ),
                            "source_channel_changes": source_channel_changes,
                            "native_state_digest": native_state_digest,
                            "reward": reward_values,
                            **tick_metrics,
                            **native_outputs,
                            "terminated": bool(np.asarray(terminated).all()),
                            "truncated": bool(np.asarray(truncated).all()),
                        }
                    )
                replay = {
                    "n_ticks": len(rows),
                    "rows": rows,
                    "reward_sum": float(sum(rewards)),
                    "trajectory_hash": _stable_hash(rows),
                    "state_effect_observed": state_effect_observed,
                    "action_count": action_count,
                    "cost_components": {
                        "energy_cost": float(
                            sum(row["energy_cost"] for row in rows)
                        ),
                        **(
                            {
                                "source_event_peak_response_burden": float(
                                    sum(
                                        row["source_event_peak_response_burden"]
                                        for row in rows
                                    )
                                ),
                                "native_storage_charging_burden": float(
                                    sum(
                                        row["native_storage_charging_burden"]
                                        for row in rows
                                    )
                                ),
                            }
                            if self._peak_response_objective is not None
                            else {}
                        ),
                    },
                    "emissions_components": {
                        "carbon_emissions": float(
                            sum(row["carbon_emissions"] for row in rows)
                        )
                    },
                    "source_ablation": ablation,
                }
            finally:
                close = getattr(replay_env, "close", None)
                if callable(close):
                    close()
        replay.update(_runtime_open_evidence(self._source_lock, opened))
        return replay

    def masked_action_replay(
        self,
        actions: list[list[float]],
        action_masks: list[list[bool]] | None = None,
    ) -> dict[str, Any]:
        """Replay an action stream and a masked no-action counterfactual.

        Both streams use a fresh CityLearn instance with the same locked source
        graph and seed. The mask is explicit per building and timestep, so a
        future scorer can attribute a result to the actions that were actually
        retained rather than silently replacing the stream with a different
        baseline. This is diagnostic until the generic scorer consumes the
        returned evidence.
        """

        if self._seed is None:
            raise RuntimeError("CityLearn backend is not reset")
        width = self.action_width
        candidate_masks = action_masks or [[True] * width for _ in actions]
        if len(candidate_masks) != len(actions):
            raise ValueError("action stream and masks must have the same length")
        no_action_masks = [[False] * width for _ in actions]
        action_run = self._replay_stream(actions, candidate_masks)
        action_repeat = self._replay_stream(actions, candidate_masks)
        masked_run = self._replay_stream(actions, no_action_masks)
        masked_repeat = self._replay_stream(actions, no_action_masks)
        source_ablation_proofs: list[dict[str, Any]] = []
        source_window_start = int(
            self._seed.backend_config.get("simulation_start_time_step") or 0
        )
        for event in self._native_source_events:
            trigger = int(event.get("trigger_tick") or -1)
            replay_tick = trigger - source_window_start
            if not 0 <= replay_tick < len(action_run["rows"]):
                continue
            ablated_run = self._replay_stream(
                actions,
                candidate_masks,
                source_ablation=event,
            )
            actual_row = action_run["rows"][replay_tick]
            ablated_row = ablated_run["rows"][replay_tick]
            native_fields = (
                "energy_cost",
                "carbon_emissions",
                "net_electricity_consumption",
                "storage_energy_balance",
            )
            changed_fields = [
                field
                for field in native_fields
                if abs(float(actual_row[field]) - float(ablated_row[field])) > 1e-8
            ]
            if _stable_hash(actual_row["reward"]) != _stable_hash(
                ablated_row["reward"]
            ):
                changed_fields.append("reward")
            source_ablation_proofs.append(
                {
                    "event_id": str(event["event_id"]),
                    "channel": str(event["channel"]),
                    "trigger_tick": trigger,
                    "replay_tick": replay_tick,
                    "event_evidence_id": self._realized_named_source_events.get(
                        str(event["event_id"]), {}
                    ).get("event_evidence_id"),
                    "changed_native_output_fields": changed_fields,
                    "causal_native_output_change": bool(changed_fields),
                    "actual_trajectory_hash": action_run["trajectory_hash"],
                    "ablated_trajectory_hash": ablated_run["trajectory_hash"],
                    "ablation": ablated_run["source_ablation"],
                }
            )
        self._source_ablation_proofs = source_ablation_proofs
        deterministic = (
            action_run["trajectory_hash"] == action_repeat["trajectory_hash"]
            and masked_run["trajectory_hash"] == masked_repeat["trajectory_hash"]
            and all(
                run["runtime_opened_assets_complete"]
                for run in (action_run, action_repeat, masked_run, masked_repeat)
            )
            and len(
                {
                    run["runtime_open_graph_hash"]
                    for run in (action_run, action_repeat, masked_run, masked_repeat)
                }
            )
            == 1
        )
        self._deterministic_replay_evidence = {
            "evidence_kind": "citylearn_masked_action_replay_v1",
            "deterministic_replay": deterministic,
            "action_trajectory_hash": action_run["trajectory_hash"],
            "action_repeat_trajectory_hash": action_repeat["trajectory_hash"],
            "masked_counterfactual_trajectory_hash": masked_run["trajectory_hash"],
            "masked_counterfactual_repeat_trajectory_hash": masked_repeat[
                "trajectory_hash"
            ],
            "runtime_open_graph_hash": action_run["runtime_open_graph_hash"],
            "runtime_opened_assets_complete": all(
                run["runtime_opened_assets_complete"]
                for run in (action_run, action_repeat, masked_run, masked_repeat)
            ),
            "source_ablation_proofs": list(source_ablation_proofs),
        }
        return {
            "status": "passed" if deterministic else "held",
            "seed": self._seed.seed,
            "action_stream_hash": _stable_hash(actions),
            "source_ablation_proofs": list(source_ablation_proofs),
            "deterministic_replay": deterministic,
            "action_run": action_run,
            "masked_counterfactual": masked_run,
            "masked_counterfactual_repeat_hash": masked_repeat["trajectory_hash"],
            "reward_delta_vs_masked": action_run["reward_sum"] - masked_run["reward_sum"],
            "native_cost_prevented_vs_masked": float(
                sum(masked_run["cost_components"].values())
                - sum(action_run["cost_components"].values())
            ),
            "runtime_opened_assets_complete": self._deterministic_replay_evidence[
                "runtime_opened_assets_complete"
            ],
            "deterministic_replay_evidence": dict(
                self._deterministic_replay_evidence
            ),
            "evidence_kind": "citylearn_masked_action_replay",
            "release_admission": "pilot_only",
        }

    def run_bounded_source_probe(self) -> dict[str, Any]:
        """Exercise a locked source trace without advancing the live episode.

        Formal adapter preflight calls source evidence immediately after reset.
        A fresh native replay is therefore required to observe source-driven
        transitions and named-event ablations.  The probe is bounded to the
        latest declared event (or two ticks when no event is declared), capped
        by the scenario horizon, and uses an all-zero action stream.
        """

        if self._seed is None or self._env is None:
            raise RuntimeError("CityLearn backend is not reset")
        live_time_before = self.current_time_step
        source_window_start = int(
            self._seed.backend_config.get("simulation_start_time_step") or 0
        )
        latest_event_tick = max(
            (
                int(event.get("trigger_tick") or source_window_start)
                - source_window_start
                for event in self._native_source_events
            ),
            default=1,
        )
        n_ticks = min(
            int(self._seed.horizon_ticks),
            max(2, latest_event_tick + 1),
        )
        actions = [[0.0] * self.action_width for _ in range(n_ticks)]
        replay = self.masked_action_replay(actions)
        action_run = replay["action_run"]
        rows = list(action_run.get("rows") or [])
        runtime_complete = action_run.get("runtime_opened_assets_complete") is True
        self._source_consumption_ticks = (
            [int(row["tick"]) for row in rows] if runtime_complete else []
        )
        self._source_state_effect_observed = any(
            row.get("source_state_effect_observed") is True for row in rows
        )
        self._post_source_state_digests = [
            str(row["native_state_digest"])
            for row in rows
            if row.get("native_state_digest")
        ]
        self._source_channel_effect_ticks = {
            channel: [
                int(row["tick"])
                for row in rows
                if (row.get("source_channel_changes") or {}).get(channel) is True
            ]
            for channel in SOURCE_CHANNELS
        }
        live_time_after = self.current_time_step
        self._bounded_source_probe_evidence = {
            "executed": True,
            "evidence_kind": "citylearn_locked_source_bounded_probe_v1",
            "n_ticks": len(rows),
            "latest_declared_event_tick": latest_event_tick,
            "live_time_step_before": live_time_before,
            "live_time_step_after": live_time_after,
            "live_clock_unchanged": live_time_before == live_time_after,
            "runtime_opened_assets_complete": runtime_complete,
            "deterministic_replay": replay.get("deterministic_replay") is True,
            "source_transition_observed": self._source_state_effect_observed,
            "named_event_proof_count": len(self._source_ablation_proofs),
        }
        return dict(self._bounded_source_probe_evidence)

    @property
    def buildings(self) -> list[str]:
        return list(self._buildings)

    @property
    def current_time_step(self) -> int:
        return int(getattr(self._env, "time_step", 0))

    @property
    def action_width(self) -> int:
        return int(self._action_vector.shape[0])

    @property
    def _peak_response_objective(self) -> str | None:
        if self._seed is None:
            return None
        value = self._seed.backend_config.get("native_peak_response_objective")
        return str(value) if value else None

    def _peak_response_active(self, tick: int) -> bool:
        """Return whether a source-bound peak-response interval is active."""

        return bool(
            self._peak_response_objective
            and any(
                str(window.get("expected_control_policy") or "") == "discharge"
                and int(window.get("first_tick") or 0)
                <= int(tick)
                <= int(window.get("last_tick", -1))
                for window in self._task_response_windows
            )
        )

    def queue_storage_rate(self, building_id: str, rate: float) -> dict[str, Any]:
        if building_id not in self._buildings:
            return {"_status": "error", "error_code": "DOMAIN_REJECTED", "reason": "unknown_building"}
        bounded = float(rate)
        if not np.isfinite(bounded) or bounded < -1.0 or bounded > 1.0:
            return {"_status": "error", "error_code": "DOMAIN_REJECTED", "reason": "rate_out_of_bounds"}
        building_pos = self._buildings.index(building_id)
        index = (
            self._storage_indices[building_pos]
            if self._storage_indices
            else building_pos
        )
        self._action_vector[index] = bounded
        policy = "charge" if bounded > 0.0 else "discharge" if bounded < 0.0 else "hold"
        return {
            "_status": "accepted",
            "building_id": building_id,
            "rate": bounded,
            "action_index": index,
            "control_endpoint": f"electrical_storage|{building_id}",
            "physical_actuator_id": building_id,
            "native_control_policy": policy,
            "signed_control_value": bounded,
        }

    def queue_storage_rates(
        self, dispatches: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Atomically queue a sparse native storage vector in one receipt."""

        if not isinstance(dispatches, list) or not dispatches:
            return {
                "_status": "error",
                "error_code": "DOMAIN_REJECTED",
                "reason": "empty_dispatches",
            }
        normalized: list[tuple[str, float]] = []
        seen: set[str] = set()
        for dispatch in dispatches:
            if not isinstance(dispatch, dict):
                return {
                    "_status": "error",
                    "error_code": "DOMAIN_REJECTED",
                    "reason": "invalid_dispatch",
                }
            building_id = str(dispatch.get("building_id") or "")
            try:
                rate = float(dispatch.get("rate"))
            except (TypeError, ValueError):
                rate = float("nan")
            if building_id not in self._buildings:
                return {
                    "_status": "error",
                    "error_code": "DOMAIN_REJECTED",
                    "reason": "unknown_building",
                }
            if building_id in seen:
                return {
                    "_status": "error",
                    "error_code": "DOMAIN_REJECTED",
                    "reason": "duplicate_building",
                }
            if not np.isfinite(rate) or rate < -1.0 or rate > 1.0:
                return {
                    "_status": "error",
                    "error_code": "DOMAIN_REJECTED",
                    "reason": "rate_out_of_bounds",
                }
            seen.add(building_id)
            normalized.append((building_id, rate))
        accepted = [
            self.queue_storage_rate(building_id, rate)
            for building_id, rate in normalized
        ]
        policies = {str(row["native_control_policy"]) for row in accepted}
        policy = next(iter(policies)) if len(policies) == 1 else "mixed"
        return {
            "_status": "accepted",
            "dispatches": accepted,
            "physical_actuator_ids": [
                str(row["physical_actuator_id"]) for row in accepted
            ],
            "native_control_policy": policy,
            "signed_control_values": [float(row["rate"]) for row in accepted],
        }

    def bind_control_evidence(
        self,
        *,
        call_id: str | None,
        evidence_id: str | None,
        payload: dict[str, Any],
        causal_parent_event_id: str | None = None,
    ) -> None:
        """Bind an accepted control call to a later native storage effect."""

        batch = payload.get("dispatches")
        if isinstance(batch, list):
            for dispatch in batch:
                if isinstance(dispatch, dict):
                    self.bind_control_evidence(
                        call_id=call_id,
                        evidence_id=evidence_id,
                        payload=dispatch,
                        causal_parent_event_id=causal_parent_event_id,
                    )
            return
        building_id = str(payload.get("building_id") or "")
        if (
            payload.get("_status") != "accepted"
            or building_id not in self._buildings
            or not call_id
            or not evidence_id
        ):
            return
        self._pending_control_evidence[building_id] = {
            "call_id": str(call_id),
            "evidence_id": str(evidence_id),
            "rate": float(payload["rate"]),
            "native_control_policy": str(payload["native_control_policy"]),
            "physical_actuator_id": str(payload["physical_actuator_id"]),
            **(
                {"causal_parent_event_id": causal_parent_event_id}
                if causal_parent_event_id
                else {}
            ),
        }

    def _electrical_storage_endpoints(self, action: np.ndarray) -> list[str]:
        """Name storage endpoints from native indices, not full action width.

        CityLearn 2023 concatenates DHW/cooling/device actions, so the live
        vector is wider than the building inventory. Endpoint evidence must
        stay on electrical_storage slots only.
        """

        if self._storage_indices:
            return [
                f"electrical_storage|{name}"
                for name, index in zip(
                    self._buildings, self._storage_indices, strict=True
                )
                if 0 <= index < len(action) and abs(float(action[index])) > 1e-8
            ]
        return [
            f"electrical_storage|{name}"
            for name, value in zip(self._buildings, action, strict=True)
            if abs(float(value)) > 1e-8
        ]

    def inspect_building_state(self, building_id: str | None = None) -> dict[str, Any]:
        if building_id is not None and building_id not in self._buildings:
            return {"_status": "error", "error_code": "DOMAIN_REJECTED", "reason": "unknown_building"}
        index = max(0, self.current_time_step)
        selected = [building_id] if building_id else self._buildings
        rows = {
            name: self._building_snapshot(self._env.buildings[self._buildings.index(name)], index)
            for name in selected
        }
        return {"_status": "ok", "time_step": self.current_time_step, "buildings": rows}

    def tick(self, tick: int) -> CityLearnTickRecord:
        if self._env is None:
            raise RuntimeError("CityLearn backend is not reset")
        action = self._action_vector.copy()
        source_tick = int(getattr(self._env, "time_step", tick))
        before_balance = [
            _current_value(building.electrical_storage, "energy_balance", source_tick)
            for building in self._env.buildings
        ]
        observation, reward, terminated, truncated, _ = self._env.step([action])
        del observation
        after_balance = [
            _current_value(building.electrical_storage, "energy_balance", source_tick)
            for building in self._env.buildings
        ]
        next_source_tick = source_tick + 1
        source_channel_changes = {
            channel: False for channel in self._source_channel_effect_ticks
        }
        for building in self._env.buildings:
            for channel, changed in _source_channel_changes(
                building, source_tick, next_source_tick
            ).items():
                source_channel_changes[channel] |= changed
        source_fields_changed = any(source_channel_changes.values())
        self._source_state_effect_observed |= source_fields_changed
        for channel, changed in source_channel_changes.items():
            if changed:
                self._source_channel_effect_ticks[channel].append(int(tick))
        control_state_effect_observed = any(
            abs(after - before) > 1e-8
            for before, after in zip(before_balance, after_balance, strict=True)
        )
        if control_state_effect_observed:
            self._state_effect_ticks.append(int(tick))
        agent_caused_events: list[dict[str, Any]] = []
        for name, before, after in zip(
            self._buildings,
            before_balance,
            after_balance,
            strict=True,
        ):
            linked = self._pending_control_evidence.get(name)
            if linked is None or abs(after - before) <= 1e-8:
                continue
            event_id = f"citylearn-storage-effect:{linked['call_id']}:{tick}"
            agent_caused_events.append(
                {
                    "event_id": event_id,
                    "type": "storage_dispatch_effect",
                    "origin": "agent_caused",
                    "agent_caused": True,
                    "tick": int(tick),
                    "outcome_tick": int(tick),
                    "decision_required": False,
                    "actionable": False,
                    "call_id": linked["call_id"],
                    "tool_name": "set_storage_dispatch",
                    "requested_action": {
                        "building_id": name,
                        "rate": linked["rate"],
                    },
                    "applied_action": {
                        "physical_actuator_id": linked[
                            "physical_actuator_id"
                        ],
                        "native_control_policy": linked[
                            "native_control_policy"
                        ],
                        "signed_control_value": linked["rate"],
                    },
                    "changed_state_fields": [
                        f"buildings.{name}.storage_energy_balance"
                    ],
                    "before_state_digest": _stable_hash(
                        {"building_id": name, "storage_energy_balance": before}
                    ),
                    "after_state_digest": _stable_hash(
                        {"building_id": name, "storage_energy_balance": after}
                    ),
                    "evidence_ids": [linked["evidence_id"]],
                    **(
                        {
                            "causal_parent_event_id": linked[
                                "causal_parent_event_id"
                            ]
                        }
                        if linked.get("causal_parent_event_id")
                        else {}
                    ),
                    "action_to_outcome_edge": {
                        "kind": "native_control_to_state_effect",
                        "source_call_id": linked["call_id"],
                        "target_event_id": event_id,
                    },
                }
            )
        if np.any(np.abs(action) > 1e-8):
            nonzero = action[np.abs(action) > 1e-8]
            policy = (
                "charge"
                if bool(np.all(nonzero > 0.0))
                else "discharge"
                if bool(np.all(nonzero < 0.0))
                else "mixed"
            )
            self._applied_controls.append(
                {
                    "tick": int(tick),
                    "action_vector": [float(value) for value in action],
                    "endpoints": self._electrical_storage_endpoints(action),
                    "state_effect_observed": control_state_effect_observed,
                    "control_state_effect_observed": control_state_effect_observed,
                    "control_policy": policy,
                }
            )
        runtime_opened = _runtime_open_evidence(
            self._source_lock, self._runtime_open_trace
        )["runtime_opened_assets_complete"]
        if runtime_opened:
            self._source_consumption_ticks.append(int(tick))
        reward_value = float(np.asarray(reward, dtype=float).sum())
        tick_metrics = _native_tick_metrics(
            list(self._env.buildings),
            source_tick,
            peak_response_objective_enabled=(
                self._peak_response_objective is not None
            ),
            peak_response_active=self._peak_response_active(tick),
        )
        self._post_source_state_digests.append(
            self._native_state_digest(source_tick)
        )
        self._last_reward = reward_value
        done = bool(np.asarray(terminated).all() or np.asarray(truncated).all())
        named_events = self._named_events_at(source_tick, tick)
        record = CityLearnTickRecord(
            tick=int(tick),
            simulator_time_step=int(getattr(self._env, "time_step", tick + 1)),
            reward=reward_value,
            energy_cost=tick_metrics["energy_cost"],
            carbon_emissions=tick_metrics["carbon_emissions"],
            district_net_electricity_consumption=tick_metrics[
                "district_net_electricity_consumption"
            ],
            source_event_peak_response_burden=tick_metrics[
                "source_event_peak_response_burden"
            ],
            native_storage_charging_burden=tick_metrics[
                "native_storage_charging_burden"
            ],
            done=done,
            action_vector=[float(value) for value in action],
            source_state_effect_observed=source_fields_changed,
            control_state_effect_observed=control_state_effect_observed,
            source_consumed=runtime_opened,
            realized_events=[
                {
                    "event_id": f"citylearn-source-timestep:{tick}",
                    "type": "building_energy_source_timestep",
                    "origin": "backend",
                    "tick": int(tick),
                    "source_consumed": runtime_opened,
                    "state_effect_observed": source_fields_changed,
                    "source_state_effect_observed": source_fields_changed,
                    "control_state_effect_observed": control_state_effect_observed,
                    "source_channel_changes": dict(source_channel_changes),
                    "decision_required": False,
                    "actionable": False,
                }
            ]
            + named_events
            + agent_caused_events,
        )
        self._records.append(record)
        self._action_vector.fill(0.0)
        self._pending_control_evidence = {}
        return record

    def _building_snapshot(self, building: Any, index: int) -> dict[str, Any]:
        return {
            "kind": "building",
            "soc": _current_value(building.electrical_storage, "soc", index),
            "net_electricity_consumption": _current_value(
                building, "net_electricity_consumption", index
            ),
            "storage_energy_balance": _current_value(
                building.electrical_storage, "energy_balance", index
            ),
            "storage_capacity": float(building.electrical_storage.capacity),
            "nominal_power": float(building.electrical_storage.nominal_power),
        }

    def _native_state_digest(self, index: int) -> str:
        """Hash source-driven native outputs without benchmark metadata."""

        return _stable_hash(
            {
                str(name): {
                    "net_electricity_consumption": _current_value(
                        building, "net_electricity_consumption", index
                    ),
                    "storage_energy_balance": _current_value(
                        building.electrical_storage, "energy_balance", index
                    ),
                    "energy_cost": _current_value(
                        building,
                        "net_electricity_consumption_cost",
                        index,
                    ),
                    "carbon_emissions": _current_value(
                        building,
                        "net_electricity_consumption_emission",
                        index,
                    ),
                }
                for name, building in zip(
                    self._buildings,
                    self._env.buildings,
                    strict=True,
                )
            }
        )

    def snapshot(self) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("CityLearn backend is not reset")
        index = max(0, self.current_time_step)
        return {
            "domain": "building_energy",
            "backend_kind": self.backend_kind,
            "clock_semantics": "simulator_owned",
            "time_step": self.current_time_step,
            "buildings": {
                name: self._building_snapshot(building, index)
                for name, building in zip(self._buildings, self._env.buildings, strict=True)
            },
            "action_bounds": {"rate_min": -1.0, "rate_max": 1.0},
            "source_lock": dict(self._source_lock),
            "source_consumption_ticks": list(self._source_consumption_ticks),
            "native_state_effect_ticks": list(self._state_effect_ticks),
            "decision_cadence": {
                "mode": "periodic_supervision",
                "native_opportunity": True,
                "periodic_scan_every_ticks": 2,
                "max_review_after_ticks": 4,
            },
        }

    def ground_truth(self) -> dict[str, Any]:
        state = self.snapshot()
        state["control_summary"] = self.control_summary()
        state["cost_components"] = self.ground_truth_costs()
        state["emissions_components"] = {
            "carbon_emissions": float(
                sum(record.carbon_emissions for record in self._records)
            )
        }
        state["per_building_unserved_units"] = {
            name: 0.0 for name in self._buildings
        }
        return state

    def ground_truth_costs(self) -> dict[str, float]:
        if self._env is None:
            return {"energy_cost": 0.0}
        return {
            "energy_cost": float(sum(record.energy_cost for record in self._records)),
            **(
                {
                    "source_event_peak_response_burden": float(
                        sum(
                            record.source_event_peak_response_burden
                            for record in self._records
                        )
                    ),
                    "native_storage_charging_burden": float(
                        sum(
                            record.native_storage_charging_burden
                            for record in self._records
                        )
                    ),
                }
                if self._peak_response_objective is not None
                else {}
            ),
        }

    def scoring_records(self) -> list[dict[str, Any]]:
        return [
            {
                "tick": record.tick,
                "native_operational_burden": max(0.0, -record.reward),
                "reward": record.reward,
                "energy_cost": record.energy_cost,
                "carbon_emissions": record.carbon_emissions,
                "district_net_electricity_consumption": (
                    record.district_net_electricity_consumption
                ),
                "source_event_peak_response_burden": (
                    record.source_event_peak_response_burden
                ),
                "native_storage_charging_burden": (
                    record.native_storage_charging_burden
                ),
                "source_consumed": record.source_consumed,
                "state_effect_observed": record.state_effect_observed,
                "source_state_effect_observed": record.source_state_effect_observed,
                "control_state_effect_observed": record.control_state_effect_observed,
            }
            for record in self._records
        ]

    def executed_action_stream(self) -> list[list[float]]:
        """Return the exact action vectors accepted by native ``step`` calls."""

        return [list(record.action_vector) for record in self._records]

    def control_summary(self) -> dict[str, Any]:
        endpoints = sorted(
            {
                endpoint
                for control in self._applied_controls
                for endpoint in control["endpoints"]
            }
        )
        effective_controls = [
            control
            for control in self._applied_controls
            if control["state_effect_observed"]
        ]
        policies = [
            str(control["control_policy"])
            for control in effective_controls
            if control.get("control_policy") in {"charge", "discharge"}
        ]
        reversals = sum(
            before != after
            for before, after in zip(policies, policies[1:], strict=False)
        )
        response_windows = []
        for window in self._task_response_windows:
            event_id = str(window.get("event_id") or "")
            event = self._realized_named_source_events.get(event_id)
            first_tick = int(window.get("first_tick") or 0)
            last_tick = int(window.get("last_tick") or -1)
            expected_policy = str(window.get("expected_control_policy") or "")
            window_controls = [
                control
                for control in effective_controls
                if first_tick <= int(control["tick"]) <= last_tick
            ]
            matching_controls = [
                control
                for control in window_controls
                if str(control.get("control_policy") or "") == expected_policy
            ]
            response_windows.append(
                {
                    **window,
                    "event_observed": event is not None,
                    "event_evidence_id": (
                        str(event["event_evidence_id"]) if event else None
                    ),
                    "observed_control_policies": sorted(
                        {
                            str(control["control_policy"])
                            for control in window_controls
                        }
                    ),
                    "direction_met": bool(event and matching_controls),
                    "control_ticks": sorted(
                        {
                            int(control["tick"])
                            for control in matching_controls
                        }
                    ),
                }
            )
        return {
            "state_changing_control_count": len(self._applied_controls),
            "distinct_physical_endpoints": endpoints,
            "effective_control_ticks": list(self._state_effect_ticks),
            "native_state_changing_leverage": bool(self._state_effect_ticks),
            "strategy_reversal_count": reversals,
            "strategy_switch_count": reversals,
            "tool_ticks": {
                "set_storage_dispatch": sorted(
                    {int(control["tick"]) for control in effective_controls}
                )
            },
            "response_windows": response_windows,
            "named_source_events": list(
                self._realized_named_source_events.values()
            ),
        }

    def source_consumption_evidence(self) -> dict[str, Any]:
        runtime = _runtime_open_evidence(
            self._source_lock, self._runtime_open_trace
        )
        causal_effect_ticks: dict[str, list[int]] = {
            channel: [] for channel in SOURCE_CHANNELS
        }
        for proof in self._source_ablation_proofs:
            if proof.get("causal_native_output_change") is not True:
                continue
            channel = str(proof.get("channel") or "").partition(".")[0]
            if channel in causal_effect_ticks:
                causal_effect_ticks[channel].append(int(proof["trigger_tick"]))
        named_events_causally_proven = bool(self._native_source_events) and (
            len(self._source_ablation_proofs) == len(self._native_source_events)
            and all(
                proof.get("causal_native_output_change") is True
                for proof in self._source_ablation_proofs
            )
        )
        source_effect_proven = bool(
            self._source_consumption_ticks
            and self._source_state_effect_observed
            and self._post_source_state_digests
        )
        consumed_channels = sorted(
            {
                asset["channel"] for asset in runtime["runtime_opened_assets"]
            }
        )
        channel_proofs = {
            channel: {
                "opened_assets": [
                    asset["path"]
                    for asset in runtime["runtime_opened_assets"]
                    if asset["channel"] == channel
                ],
                "effect_ticks": (
                    [0]
                    if channel == "schema"
                    and runtime["runtime_opened_assets_complete"]
                    and self._buildings
                    else list(causal_effect_ticks.get(channel, []))
                ),
                "state_effect_observed": (
                    bool(self._buildings)
                    and runtime["runtime_opened_assets_complete"]
                    if channel == "schema"
                    else bool(causal_effect_ticks.get(channel, []))
                ),
            }
            for channel in (
                "schema",
                "weather",
                "building_timeseries",
                "pricing",
                "carbon_intensity",
            )
        }
        blockers: list[str] = []
        if not runtime["runtime_opened_assets_complete"]:
            blockers.append(
                "runtime_source_open_graph_missing"
                if not runtime["runtime_opened_assets"]
                else "runtime_source_assets_not_opened"
            )
        if not source_effect_proven:
            blockers.append("runtime_source_state_effect_unobserved")
        if self._native_source_events and not named_events_causally_proven:
            blockers.append("named_source_events_causal_proof_missing")
        consumed_source_hashes = {
            str(asset["path"]): str(asset["sha256"])
            for asset in runtime["runtime_opened_assets"]
            if asset.get("path") and asset.get("sha256")
        }
        trace_payload = {
            "initial_state_digest": self._initial_source_state_digest,
            "post_source_state_digests": self._post_source_state_digests,
            "consumption_ticks": self._source_consumption_ticks,
            "consumed_source_hashes": consumed_source_hashes,
            "source_channel_input_transition_ticks": (
                self._source_channel_effect_ticks
            ),
        }
        runtime_trace_observed = bool(
            runtime["runtime_opened_assets_complete"]
            and self._source_consumption_ticks
            and self._initial_source_state_digest
            and self._post_source_state_digests
        )
        return {
            "status": (
                "passed"
                if source_effect_proven
                and runtime["runtime_opened_assets_complete"]
                and (
                    not self._native_source_events
                    or named_events_causally_proven
                )
                else "held"
            ),
            "proof_kind": "direct_runtime_files",
            "runtime_open_evidence_kind": "instrumented_runtime_open_v1",
            **runtime,
            "derivation_assets": list(
                self._source_lock.get("derivation_files", {}).values()
            ),
            "opened_source_paths": sorted(consumed_source_hashes),
            "opened_source_sha256": dict(sorted(consumed_source_hashes.items())),
            "consumed_source_hashes": dict(sorted(consumed_source_hashes.items())),
            "lineage_source_hashes": {},
            "consumed_channels": consumed_channels,
            "derived_backend_state_fields": [
                "carbon_emissions",
                "energy_cost",
                "net_electricity_consumption",
                "storage_energy_balance",
                *(
                    [
                        "source_event_peak_response_burden",
                        "native_storage_charging_burden",
                    ]
                    if self._peak_response_objective is not None
                    else []
                ),
            ],
            "runtime_channel_proofs": channel_proofs,
            "consumption_ticks": list(self._source_consumption_ticks),
            "initial_state_digest": self._initial_source_state_digest,
            "post_source_state_digests": list(
                self._post_source_state_digests
            ),
            "source_channel_effect_ticks": {
                channel: list(ticks)
                for channel, ticks in causal_effect_ticks.items()
                if ticks
            },
            "source_channel_input_transition_ticks": {
                channel: list(ticks)
                for channel, ticks in self._source_channel_effect_ticks.items()
            },
            "source_ablation_proofs": list(self._source_ablation_proofs),
            "named_events_causally_proven": named_events_causally_proven,
            "derived_state_fields": [
                "non_shiftable_load",
                "solar_generation",
                "outdoor_dry_bulb_temperature",
                "electricity_pricing",
                "carbon_intensity",
            ],
            "state_effect_observed": source_effect_proven,
            "source_state_effect_observed": source_effect_proven,
            "deterministic_source_trace": runtime_trace_observed,
            "trace_semantic_digest": (
                _stable_hash(trace_payload) if runtime_trace_observed else None
            ),
            "runtime_trace_observed": runtime_trace_observed,
            "evidence_from_scenario_config_only": False,
            "deterministic_replay": bool(
                (self._deterministic_replay_evidence or {}).get(
                    "deterministic_replay", False
                )
            ),
            "deterministic_replay_evidence": (
                dict(self._deterministic_replay_evidence)
                if self._deterministic_replay_evidence is not None
                else None
            ),
            "bounded_source_probe": (
                dict(self._bounded_source_probe_evidence)
                if self._bounded_source_probe_evidence is not None
                else {"executed": False}
            ),
            "blockers": blockers,
        }

    def close(self) -> None:
        self._env = None
