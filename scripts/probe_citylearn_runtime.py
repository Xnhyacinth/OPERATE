#!/usr/bin/env python3
"""Run a bounded, non-release CityLearn runtime and replay probe.

The probe intentionally stops short of Core admission.  It verifies that the
locked CityLearn schema is opened by the native runtime, the simulator owns
time progression, a native storage action changes state, and a fixed action
stream replays deterministically.  The no-op stream below is an explicit
masked-action replay diagnostic, not a declaration-only baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
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
DEFAULT_OUTPUT = Path("/tmp/dt_sched_citylearn/native_runtime_probe.json")
CANDIDATE_REPAIR_DATASETS = (
    (
        "infrastructure/citylearn/ca-alameda-county-neighborhood",
        "ca_alameda_county_neighborhood",
    ),
    (
        "infrastructure/citylearn/quebec-neighborhood-with-demand-response-set-points",
        "quebec_neighborhood_with_demand_response_set_points",
    ),
    (
        "infrastructure/citylearn/quebec-neighborhood-without-demand-response-set-points",
        "quebec_neighborhood_without_demand_response_set_points",
    ),
    (
        "infrastructure/citylearn/tx-travis-county-neighborhood",
        "tx_travis_county_neighborhood",
    ),
    (
        "infrastructure/citylearn/vt-chittenden-county-neighborhood",
        "vt_chittenden_county_neighborhood",
    ),
)


def _classify_runtime_error(exc: BaseException) -> str:
    """Return stable repair codes for known CityLearn compatibility failures."""

    message = str(exc)
    if (
        "node array from the pickle has an incompatible dtype" in message
        and "missing_go_to_left" in message
    ):
        return "sklearn_legacy_tree_pickle_incompatible"
    if "demand is greater than" in message and "device max output" in message:
        return "citylearn_reset_device_capacity_double_count"
    return f"unclassified_runtime_error:{type(exc).__name__}"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _source_files(source_root: Path) -> dict[str, Path]:
    schema = json.loads((source_root / "schema.json").read_text(encoding="utf-8"))
    names = {
        "schema.json": source_root / "schema.json",
        "weather.csv": source_root / "weather.csv",
        "pricing.csv": source_root / "pricing.csv",
        "carbon_intensity.csv": source_root / "carbon_intensity.csv",
    }
    for row in (schema.get("buildings") or {}).values():
        if isinstance(row, dict) and row.get("energy_simulation"):
            name = str(row["energy_simulation"])
            names[name] = source_root / name
    data_root = source_root.resolve().parent.parent
    for name in ("lbl-tracking_the_sun-res-pv.csv", "battery_choices.yaml"):
        names[name] = data_root / "misc" / name
    return names


def _verify_source_lock(
    source_root: Path, source_lock_path: Path
) -> tuple[dict[str, Any], list[str]]:
    sidecar = json.loads(source_lock_path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    files = _source_files(source_root)
    actual = {name: _sha256(path) for name, path in files.items()}
    declared = sidecar.get("files") if isinstance(sidecar.get("files"), dict) else {}
    for name, path in files.items():
        ref = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        if actual[name] is None:
            blockers.append(f"missing_source_file:{name}")
        if declared.get(ref) != actual[name]:
            blockers.append(f"source_hash_mismatch:{name}")
    for field, name in (
        ("schema_sha256", "schema.json"),
        ("weather_file_or_timeseries_lock", "weather.csv"),
        ("pricing_file_sha256", "pricing.csv"),
        ("carbon_intensity_file_sha256", "carbon_intensity.csv"),
        ("pv_sizing_file_sha256", "lbl-tracking_the_sun-res-pv.csv"),
        ("battery_sizing_file_sha256", "battery_choices.yaml"),
    ):
        if sidecar.get(field) != f"sha256:{actual[name]}":
            blockers.append(f"lock_field_mismatch:{field}")
    return {
        "path": source_lock_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "sha256": _sha256(source_lock_path),
        "files": {
            name: {"path": str(path), "sha256": actual[name]}
            for name, path in files.items()
        },
    }, sorted(set(blockers))


def _action_schedule(n_ticks: int, width: int) -> list[list[float]]:
    """Use a fixed native storage schedule with a visible charge then release."""
    values = [1.0, 0.0, -1.0, 0.0]
    return [
        (values[tick % len(values)] * np.ones(width, dtype=np.float32)).tolist()
        for tick in range(n_ticks)
    ]


def _new_citylearn_env(source_root: Path, *, seed: int, n_ticks: int) -> Any:
    from citylearn.citylearn import CityLearnEnv

    return CityLearnEnv(
        schema=source_root / "schema.json",
        root_directory=source_root,
        central_agent=True,
        simulation_start_time_step=0,
        # CityLearn derives multi-step prediction windows from the configured
        # simulation span, so keep a small but valid 48-hour source window even
        # when this probe only executes a few native ticks.
        simulation_end_time_step=max(n_ticks + 1, 47),
        episode_time_steps=max(n_ticks + 1, 48),
        random_seed=seed,
        render_mode="none",
    )


def _native_action_width(source_root: Path, seed: int) -> int:
    """Read the action width from the locked native runtime, not a YAML guess."""

    env = _new_citylearn_env(source_root, seed=seed, n_ticks=0)
    try:
        env.reset(seed=seed)
        from domains.building_energy.backends.citylearn import (
            _clear_reset_priming,
        )

        _clear_reset_priming(env)
        return int(env.action_space[0].shape[0])
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _run_episode(
    source_root: Path,
    *,
    seed: int,
    actions: list[list[float]],
    noop: bool,
    action_masks: list[list[bool]] | None = None,
) -> dict[str, Any]:
    env = _new_citylearn_env(source_root, seed=seed, n_ticks=len(actions))
    observation, _ = env.reset(seed=seed)
    from domains.building_energy.backends.citylearn import _clear_reset_priming

    _clear_reset_priming(env)
    width = int(env.action_space[0].shape[0])
    if any(len(action) != width for action in actions):
        raise ValueError("action schedule width does not match CityLearn action space")
    masks = action_masks or [[True] * width for _ in actions]
    if len(masks) != len(actions) or any(len(mask) != width for mask in masks):
        raise ValueError("action mask shape does not match CityLearn action stream")
    rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    action_count = 0
    for tick, (values, mask) in enumerate(zip(actions, masks, strict=True)):
        before_soc = [float(building.electrical_storage.soc[0]) for building in env.buildings]
        action_values = [
            float(value) if bool(enabled) else 0.0
            for value, enabled in zip(values, mask, strict=True)
        ]
        if noop:
            action_values = [0.0] * width
        if any(abs(value) > 1e-8 for value in action_values):
            action_count += 1
        action = [np.asarray(action_values, dtype=np.float32)]
        next_observation, reward, terminated, truncated, _ = env.step(action)
        after_soc = [float(building.electrical_storage.soc[0]) for building in env.buildings]
        rows.append(
            {
                "tick": tick,
                "simulator_time_step": int(env.time_step),
                "action_mask": [bool(value) for value in mask],
                "observation_hash": _stable_hash(np.asarray(next_observation).tolist()),
                "before_soc": before_soc,
                "after_soc": after_soc,
                "soc_delta": [
                    after - before
                    for before, after in zip(before_soc, after_soc, strict=True)
                ],
                "reward": np.asarray(reward, dtype=float).tolist(),
                "terminated": bool(np.asarray(terminated).all()),
                "truncated": bool(np.asarray(truncated).all()),
            }
        )
        rewards.extend(np.asarray(reward, dtype=float).reshape(-1).tolist())
    return {
        "noop": noop,
        "n_ticks": len(rows),
        "rows": rows,
        "reward_sum": float(sum(rewards)),
        "trajectory_hash": _stable_hash(rows),
        "action_count": action_count,
    }


def _candidate_action_surface(source_root: Path) -> dict[str, bool]:
    schema = json.loads((source_root / "schema.json").read_text(encoding="utf-8"))
    actions = schema.get("actions") or {}
    return {
        str(name): bool(row.get("active"))
        for name, row in actions.items()
        if isinstance(row, dict)
    }


def _native_action_names(env: Any) -> list[str]:
    names = getattr(env, "action_names", [])
    if isinstance(names, list) and names and isinstance(names[0], list):
        names = names[0]
    return [str(name) for name in names] if isinstance(names, list) else []


def _native_noop(env: Any) -> np.ndarray:
    """Return a legal zero-like action using the runtime's actual bounds."""

    space = env.action_space[0]
    width = int(space.shape[0])
    low = np.asarray(space.low, dtype=np.float32).reshape(width)
    high = np.asarray(space.high, dtype=np.float32).reshape(width)
    return np.clip(np.zeros(width, dtype=np.float32), low, high)


def _run_candidate_repair_episode(
    source_root: Path,
    *,
    seed: int,
    n_ticks: int,
) -> dict[str, Any]:
    from domains.building_energy.backends.citylearn import _clear_reset_priming

    env = _new_citylearn_env(source_root, seed=seed, n_ticks=n_ticks)
    try:
        observation, _ = env.reset(seed=seed)
        reset_repair = _clear_reset_priming(env)
        action_names = _native_action_names(env)
        storage_indices = [
            index
            for index, name in enumerate(action_names)
            if name == "electrical_storage"
        ]
        width = int(env.action_space[0].shape[0])
        if action_names and len(action_names) != width:
            raise ValueError("CityLearn action names do not match native action width")
        if storage_indices and len(storage_indices) != len(env.buildings):
            raise ValueError(
                "CityLearn electrical_storage actions do not match building inventory"
            )
        rows: list[dict[str, Any]] = []
        state_effect_observed = False
        for tick in range(n_ticks):
            source_tick = int(env.time_step)
            before = (
                [
                    float(building.electrical_storage.energy_balance[source_tick])
                    for building in env.buildings
                ]
                if storage_indices
                else []
            )
            action = _native_noop(env)
            rate = 0.18 if tick % 2 == 0 else -0.18
            if storage_indices:
                low = np.asarray(env.action_space[0].low, dtype=np.float32)
                high = np.asarray(env.action_space[0].high, dtype=np.float32)
                action[storage_indices] = np.clip(
                    rate,
                    low[storage_indices],
                    high[storage_indices],
                )
            next_observation, reward, terminated, truncated, _ = env.step([action])
            after = (
                [
                    float(building.electrical_storage.energy_balance[source_tick])
                    for building in env.buildings
                ]
                if storage_indices
                else []
            )
            state_effect = any(
                abs(right - left) > 1e-8
                for left, right in zip(before, after, strict=True)
            )
            state_effect_observed |= state_effect
            rows.append(
                {
                    "tick": tick,
                    "simulator_time_step": int(env.time_step),
                    "observation_hash": _stable_hash(
                        np.asarray(next_observation).tolist()
                    ),
                    "reward": np.asarray(reward, dtype=float).tolist(),
                    "storage_rate": rate if storage_indices else None,
                    "storage_state_effect_observed": state_effect,
                    "terminated": bool(np.asarray(terminated).all()),
                    "truncated": bool(np.asarray(truncated).all()),
                }
            )
        return {
            "native_step_executed": len(rows) == n_ticks,
            "n_ticks": len(rows),
            "action_width": width,
            "native_action_names": action_names,
            "electrical_storage_action_indices": storage_indices,
            "reset_priming_repair": reset_repair,
            "initial_observation_hash": _stable_hash(
                np.asarray(observation).tolist()
            ),
            "native_storage_state_effect_observed": state_effect_observed,
            "rows": rows,
            "trajectory_hash": _stable_hash(rows),
        }
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def run_candidate_repair_probe(
    source_root: Path,
    *,
    seed: int = 2022,
    n_ticks: int = 2,
) -> dict[str, Any]:
    """Probe one held CityLearn source without altering its scientific inputs."""

    source_root = source_root.resolve()
    action_surface = _candidate_action_surface(source_root)
    report: dict[str, Any] = {
        "source_root": (
            source_root.relative_to(REPO_ROOT.resolve()).as_posix()
            if source_root.is_relative_to(REPO_ROOT.resolve())
            else str(source_root)
        ),
        "source_schema_sha256": _sha256(source_root / "schema.json"),
        "electrical_storage_control_axis": action_surface.get(
            "electrical_storage", False
        ),
        "active_action_surface": sorted(
            name for name, active in action_surface.items() if active
        ),
        "probe_runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "citylearn_version": _distribution_version("citylearn"),
            "scikit_learn_version": _distribution_version("scikit-learn"),
        },
        "scientific_constraints": {
            "source_values_changed": False,
            "source_window_changed": False,
            "headroom_threshold_changed": False,
        },
    }
    task_axis_blockers: list[str] = []
    if report["electrical_storage_control_axis"] is not True:
        task_axis_blockers.append("electrical_storage_control_axis_absent")
        if action_surface.get("heating_device") is True:
            task_axis_blockers.append("heating_control_task_not_designed")
    try:
        first = _run_candidate_repair_episode(
            source_root,
            seed=seed,
            n_ticks=n_ticks,
        )
        repeat = _run_candidate_repair_episode(
            source_root,
            seed=seed,
            n_ticks=n_ticks,
        )
    except Exception as exc:  # pragma: no cover - native dependency-specific
        code = _classify_runtime_error(exc)
        report.update(
            {
                "status": "blocked_runtime_compatibility",
                "runtime_failure_code": code,
                "runtime_error": f"{type(exc).__name__}: {exc}",
                "environment_repair_passed": False,
                "environment_blockers": [code],
                "task_axis_blockers": task_axis_blockers,
                "task_axis_ready_for_storage_refinement": False,
                "native_step_executed": False,
                "deterministic_replay": False,
            }
        )
        return report
    deterministic = first["trajectory_hash"] == repeat["trajectory_hash"]
    environment_passed = bool(first["native_step_executed"] and deterministic)
    if (
        report["electrical_storage_control_axis"] is True
        and first["native_storage_state_effect_observed"] is not True
    ):
        task_axis_blockers.append("native_storage_state_effect_unproven")
    task_axis_ready = bool(environment_passed and not task_axis_blockers)
    report.update(
        {
            "status": (
                "environment_repair_passed"
                if environment_passed
                else "blocked_runtime_contract"
            ),
            "runtime_failure_code": None,
            "environment_repair_passed": environment_passed,
            "environment_blockers": (
                [] if environment_passed else ["focused_native_replay_not_passed"]
            ),
            "task_axis_blockers": task_axis_blockers,
            "task_axis_ready_for_storage_refinement": task_axis_ready,
            "native_step_executed": first["native_step_executed"],
            "native_storage_state_effect_observed": first[
                "native_storage_state_effect_observed"
            ],
            "deterministic_replay": deterministic,
            "trajectory_hash": first["trajectory_hash"],
            "repeat_trajectory_hash": repeat["trajectory_hash"],
            "reset_priming_repair": first["reset_priming_repair"],
            "action_width": first["action_width"],
            "native_action_names": first["native_action_names"],
            "electrical_storage_action_indices": first[
                "electrical_storage_action_indices"
            ],
        }
    )
    return report


def run_probe(
    *, source_root: Path = DEFAULT_SOURCE_ROOT,
    source_lock_path: Path = DEFAULT_SOURCE_LOCK,
    seed: int = 2022,
    n_ticks: int = 4,
) -> dict[str, Any]:
    lock, blockers = _verify_source_lock(source_root, source_lock_path)
    report: dict[str, Any] = {
        "schema_version": "citylearn_native_runtime_probe_v1",
        "release_admission": "pilot_only",
        "source_root": source_root.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "source_lock": lock,
        "seed": seed,
        "requested_ticks": n_ticks,
        "source_lock_blockers": blockers,
    }
    if blockers:
        report.update({"status": "blocked_source_lock", "release_ready": False})
        return report

    try:
        import citylearn  # noqa: F401
        from citylearn.citylearn import CityLearnEnv  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency-specific
        report.update(
            {
                "status": "blocked_runtime_import",
                "release_ready": False,
                "runtime_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return report

    action_width = _native_action_width(source_root, seed)
    probe_actions = _action_schedule(n_ticks, action_width)
    try:
        action_run = _run_episode(source_root, seed=seed, actions=probe_actions, noop=False)
        replay_run = _run_episode(source_root, seed=seed, actions=probe_actions, noop=False)
        noop_run = _run_episode(
            source_root,
            seed=seed,
            actions=probe_actions,
            noop=False,
            action_masks=[[False] * action_width for _ in probe_actions],
        )
        from domains.building_energy.backends.citylearn import CityLearnBackend
        from domains.building_energy.seeds.schema import rebuild_seed_from_dict

        native_backend = CityLearnBackend()
        native_backend.reset(
            rebuild_seed_from_dict(
                {
                    "domain": "building_energy",
                    "family": "citylearn_der_storage_control",
                    "backend_kind": "citylearn",
                    "seed_id": "citylearn_native_runtime_probe",
                    "horizon_ticks": max(1, n_ticks),
                    "tick_minutes": 60,
                    "difficulty_level": "medium",
                    "source_root": str(source_root),
                    "source_lock": str(source_lock_path),
                    "backend_config": {
                        "simulation_start_time_step": 0,
                        "simulation_end_time_step": max(n_ticks + 1, 47),
                        "episode_time_steps": max(n_ticks + 1, 48),
                    },
                },
                override_seed=seed,
            )
        )
        for building_id, rate in zip(
            native_backend.buildings, probe_actions[0], strict=True
        ):
            native_backend.queue_storage_rate(building_id, rate)
        native_backend.tick(0)
        native_masked_replay = native_backend.masked_action_replay(probe_actions)
        source_consumption = native_backend.source_consumption_evidence()
        native_backend.close()
    except Exception as exc:  # pragma: no cover - native dependency/runtime-specific
        report.update(
            {
                "status": "blocked_runtime_execution",
                "release_ready": False,
                "runtime_error": f"{type(exc).__name__}: {exc}",
            }
        )
        return report

    state_effect = any(
        abs(delta) > 1e-8
        for row in action_run["rows"]
        for delta in row["soc_delta"]
    )
    deterministic = action_run["trajectory_hash"] == replay_run["trajectory_hash"]
    runtime_proof_passed = bool(
        deterministic
        and source_consumption.get("status") == "passed"
        and native_masked_replay.get("deterministic_replay") is True
        and native_masked_replay.get("runtime_opened_assets_complete") is True
    )
    report.update(
        {
            "status": (
                "runtime_probe_passed"
                if runtime_proof_passed
                else "runtime_probe_held"
            ),
            "release_ready": False,
            "source_consumption": source_consumption,
            "runtime": {
                "backend": "citylearn_gymnasium",
                "package_version": citylearn.__version__,
                "simulator_owns_clock": all(
                    row["simulator_time_step"] == row["tick"] + 1
                    for row in action_run["rows"]
                ),
                "native_action": "electrical_storage",
                "action_width": action_width,
                "native_state_effect_observed": state_effect,
                "deterministic_replay": deterministic,
                "action_run": action_run,
                "replay_run": replay_run,
                "noop_counterfactual": noop_run,
                "diagnostic_reward_delta_vs_noop": action_run["reward_sum"]
                - noop_run["reward_sum"],
            },
            "masked_action_replay": {
                "deterministic_replay": native_masked_replay[
                    "deterministic_replay"
                ],
                "candidate_trajectory_hash": native_masked_replay["action_run"][
                    "trajectory_hash"
                ],
                "masked_counterfactual_trajectory_hash": native_masked_replay[
                    "masked_counterfactual"
                ]["trajectory_hash"],
                "masked_counterfactual_action_count": native_masked_replay[
                    "masked_counterfactual"
                ]["action_count"],
                "runtime_opened_assets_complete": native_masked_replay[
                    "runtime_opened_assets_complete"
                ],
                "evidence_kind": "citylearn_masked_action_replay_v1",
            },
            "release_blockers": [
                "citylearn_positive_headroom_task_not_proven",
                "citylearn_protocol21_full_gates_not_run",
            ],
        }
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--ticks", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-repair", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.candidate_repair:
        report = run_candidate_repair_probe(
            args.source_root,
            seed=args.seed,
            n_ticks=args.ticks,
        )
        passed = report.get("environment_repair_passed") is True
    else:
        report = run_probe(
            source_root=args.source_root,
            source_lock_path=args.source_lock,
            seed=args.seed,
            n_ticks=args.ticks,
        )
        passed = report.get("status") == "runtime_probe_passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
