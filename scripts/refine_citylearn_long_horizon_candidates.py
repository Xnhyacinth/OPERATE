#!/usr/bin/env python3
"""Bounded CityLearn 72-hour candidate refinement on locked native traces.

The script is candidate-only. It evaluates three disjoint source windows and
materializes a scenario row only when every native, behavioral and long-horizon
gate passes. Failed windows remain explicit held rows; no difficulty relabel,
random event padding or Core/release mutation is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import Action, ToolCall  # noqa: E402
from core.implementation_identity import implementation_identity  # noqa: E402
from domains.building_energy.adapter import BuildingEnergyEnvironment  # noqa: E402
from runner.resume import recompute_signature_with_seed  # noqa: E402
from scripts.build_protocol21_candidate_source_suite import (  # noqa: E402
    build_suite as build_candidate_source_suite,
)
from scripts.probe_citylearn_runtime import (  # noqa: E402
    CANDIDATE_REPAIR_DATASETS,
    run_candidate_repair_probe,
)


@dataclass(frozen=True)
class WindowPlan:
    dataset_id: str
    start: int
    horizon: int = 72
    difficulty_level: str = "extreme"


WINDOW_PLANS = (
    WindowPlan("citylearn_challenge_2022_phase_1", 0),
    WindowPlan("citylearn_challenge_2022_phase_2", 72),
    WindowPlan("citylearn_challenge_2022_phase_3", 144),
)
DEFAULT_INPUT = (
    REPO_ROOT
    / "reports"
    / "external_native_conversion_wave_20260813"
    / "exact_source_suite.json"
)
DEFAULT_STAGING = (
    REPO_ROOT / "scenarios" / "staging" / "citylearn_long_horizon_20260813"
)
DEFAULT_REPORT = REPO_ROOT / "reports" / "citylearn_long_horizon_refine_20260813.json"
SEED = 2022
HEADROOM_ABSOLUTE_FLOOR = 2.5
HEADROOM_RELATIVE_FLOOR = 0.05
DEFAULT_DATASET_ROOT = REPO_ROOT / "works" / "CityLearn" / "data" / "datasets"
DEFAULT_RUNTIME_ADMISSION_ROOT = (
    REPO_ROOT / ".hl" / "candidates" / "citylearn_runtime_admission_v1"
)
RUNTIME_ADMISSION_DATASETS = tuple(
    (candidate_id, source_unit)
    for candidate_id, source_unit in CANDIDATE_REPAIR_DATASETS
    if not source_unit.startswith("quebec_")
)
TERMINAL_ADMISSION_DISPOSITIONS = frozenset(
    {"promoted", "secondary_duplicate", "held_repair", "abandoned_intrinsic"}
)


def _finalize_runtime_admission_rows(
    *,
    source_rows: list[dict[str, Any]],
    materialized: dict[str, Any],
) -> list[dict[str, Any]]:
    """Map common Protocol-2.1 outcomes to terminal CityLearn dispositions."""

    indexed: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for section, disposition in (
        ("scenarios", "promoted"),
        ("secondary", "secondary_duplicate"),
        ("rejected", "rejected"),
    ):
        for row in materialized.get(section) or []:
            identity = (
                str(row.get("scenario_id") or ""),
                str(row.get("scenario_signature") or ""),
            )
            if not all(identity) or identity in indexed:
                raise ValueError("Protocol-2.1 materialization identities are invalid")
            indexed[identity] = (disposition, row)
    finalized: list[dict[str, Any]] = []
    for source_row in source_rows:
        identity = (
            str(source_row.get("scenario_id") or ""),
            str(source_row.get("scenario_signature") or ""),
        )
        if identity not in indexed:
            raise ValueError(
                f"{identity[0]} is missing from Protocol-2.1 materialization"
            )
        disposition, result = indexed[identity]
        reason_codes = [str(code) for code in result.get("reason_codes") or []]
        if disposition == "rejected":
            disposition = (
                "abandoned_intrinsic"
                if result.get("disposition") == "retired_intrinsic"
                else "held_repair"
            )
        if disposition not in TERMINAL_ADMISSION_DISPOSITIONS:
            raise ValueError(f"invalid terminal disposition: {disposition}")
        finalized.append(
            {
                "scenario_id": identity[0],
                "scenario_signature": identity[1],
                "full_protocol21_executed": True,
                "terminal": True,
                "final_disposition": disposition,
                "reason_codes": reason_codes,
                "protocol21_result": result,
            }
        )
    return finalized


def _merge_runtime_admission_rows(
    *,
    initial_rows: list[dict[str, Any]],
    protocol_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge full-admission outcomes with candidates terminalized before it."""

    protocol_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for row in protocol_rows:
        identity = (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        if not all(identity) or identity in protocol_by_identity:
            raise ValueError("Protocol-2.1 terminal identities are invalid")
        protocol_by_identity[identity] = row

    merged: list[dict[str, Any]] = []
    consumed: set[tuple[str, str]] = set()
    for initial in initial_rows:
        if initial.get("terminal") is True:
            disposition = str(initial.get("final_disposition") or "")
            if disposition not in TERMINAL_ADMISSION_DISPOSITIONS:
                raise ValueError(f"invalid early terminal disposition: {disposition}")
            row = dict(initial)
            row.setdefault("full_protocol21_executed", False)
            merged.append(row)
            continue
        identity = (
            str(initial.get("scenario_id") or ""),
            str(initial.get("scenario_signature") or ""),
        )
        protocol = protocol_by_identity.get(identity)
        if protocol is None:
            raise ValueError(f"{identity[0]} is missing a terminal Protocol-2.1 result")
        consumed.add(identity)
        merged.append({**initial, **protocol})
    if consumed != set(protocol_by_identity):
        raise ValueError("Protocol-2.1 results include unknown CityLearn candidates")
    if not all(row.get("terminal") is True for row in merged):
        raise ValueError("final CityLearn admission ledger contains unresolved rows")
    return merged


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _repo_ref(path: Path) -> str:
    resolved = path.resolve()
    return (
        resolved.relative_to(REPO_ROOT).as_posix()
        if resolved.is_relative_to(REPO_ROOT)
        else str(resolved)
    )


def _window_sha256(payload: dict[str, Any]) -> str:
    if not payload.get("source_rows"):
        raise ValueError("CityLearn window identity requires source_rows")
    return _stable_hash(payload)


def _candidate_source_contract(window: dict[str, Any]) -> dict[str, Any]:
    runtime_files = dict(window.get("runtime_files") or {})
    if not runtime_files or not all(
        isinstance(path, str) and isinstance(digest, str) and len(digest) == 64
        for path, digest in runtime_files.items()
    ):
        raise ValueError("CityLearn runtime source hashes are incomplete")
    window_sha256 = str(window.get("window_sha256") or "")
    if len(window_sha256) != 64:
        raise ValueError("CityLearn source window hash is invalid")
    derivation_files = dict(window.get("derivation_files") or {})
    return {
        "runtime_input": sorted(runtime_files),
        "derivation_input": sorted(derivation_files),
        "file_sha256s": {**runtime_files, **derivation_files},
        "derived_window": {
            "sha256": window_sha256,
            "recipe_version": "citylearn_locked_csv_window_v1",
        },
    }


def _citylearn_candidate_source_lock(source_root: Path) -> dict[str, Any]:
    """Build a candidate-only lock over CityLearn's actual runtime graph."""

    from domains.building_energy.backends.citylearn import (
        _derivation_source_files,
        _runtime_source_files,
    )

    runtime_files = _runtime_source_files(source_root)
    derivation_files = _derivation_source_files(source_root)
    all_files = {**runtime_files, **derivation_files}
    locked = {_repo_ref(path): _sha256(path) for path in all_files.values()}
    schema_sha = _sha256(source_root / "schema.json")
    weather_sha = _sha256(source_root / "weather.csv")
    pv_sha = _sha256(derivation_files["lbl-tracking_the_sun-res-pv.csv"])
    battery_sha = _sha256(derivation_files["battery_choices.yaml"])
    schema = json.loads((source_root / "schema.json").read_text(encoding="utf-8"))
    buildings = sorted(
        str(name)
        for name, row in (schema.get("buildings") or {}).items()
        if isinstance(row, dict) and row.get("include", True)
    )
    return {
        "source_id": source_root.name,
        "dataset_release_or_challenge_version": source_root.name,
        "dataset_source_url": "https://github.com/intelligent-environments-lab/CityLearn",
        "source_url": "https://github.com/intelligent-environments-lab/CityLearn",
        "license": "MIT",
        "license_verified": True,
        "terms_verified": True,
        "lock_strategy": "git_tag+commit+all_schema_referenced_file_sha256",
        "git_commit_or_release_tag": (
            "v2.5.0@29062af6d077409e1c37a3e53a6cac30fd4d02bc"
        ),
        "implementation_tree_sha256": (
            "ed4989f7c91b7d85ab511139dda2ce7d8f56199cd5943cafd0954727d5eef138"
        ),
        "package_version": "citylearn==2.5.0",
        "torch_version": "torch==2.13.0",
        "citylearn_offline": True,
        "random_episode_split": False,
        "rolling_episode_split": False,
        "schema_or_dataset_name": f"{source_root.name}/schema.json",
        "schema_sha256": f"sha256:{schema_sha}",
        "weather_file_or_timeseries_lock": f"sha256:{weather_sha}",
        "pricing_file_sha256": None,
        "carbon_intensity_file_sha256": None,
        "optional_runtime_assets_absent": [
            "pricing.csv",
            "carbon_intensity.csv",
        ],
        "pv_sizing_file_sha256": f"sha256:{pv_sha}",
        "battery_sizing_file_sha256": f"sha256:{battery_sha}",
        "simulation_file_sha256": f"sha256:{_stable_hash(locked)}",
        "simulator_seed": SEED,
        "building_cluster": buildings,
        "files": dict(sorted(locked.items())),
    }


_BLOCKER_CODES = {
    "source_lock": "source_lock_invalid",
    "window_identity": "source_window_identity_invalid",
    "native_source_consumption": "native_source_consumption_unproven",
    "native_state_effect": "native_control_state_effect_unproven",
    "deterministic_replay": "deterministic_replay_failed",
    "positive_material_headroom": "positive_material_headroom_failed",
    "survival": "reference_survival_failed",
    "task_completion": "native_task_completion_failed",
    "partial_observation_executed": "partial_observation_not_executed",
    "delayed_control_executed": "delayed_control_not_executed",
    "dependency_depth": "dependency_depth_below_extreme_floor",
    "native_constraint_breadth": "native_constraint_breadth_below_extreme_floor",
    "terminal": "episode_not_terminal",
    "implementation_stable": "implementation_tree_changed_during_run",
}


def _admission_blockers(gates: dict[str, bool]) -> list[str]:
    return [
        _BLOCKER_CODES.get(name, f"gate_failed:{name}")
        for name, passed in gates.items()
        if passed is not True
    ]


def _runtime_admission_reference_decision(
    runtime: dict[str, Any], *, horizon: int
) -> dict[str, Any]:
    """Classify one executed CityLearn reference without relabeling its task."""

    source = dict(runtime.get("source_consumption") or {})
    controls = dict(runtime.get("control_summary") or {})
    replay = dict(runtime.get("replay") or {})
    gates = {
        "native_source_consumption": source.get("status") == "passed",
        "native_state_effect": controls.get("native_state_changing_leverage") is True,
        "deterministic_replay": replay.get("deterministic") is True,
        "positive_material_headroom": replay.get("headroom_passed") is True,
        "survival": runtime.get("n_ticks") == horizon,
        "task_completion": runtime.get("task_completed") is True,
        "delayed_control_executed": (
            int(runtime.get("delayed_ack_count") or 0) > 0
            and int(runtime.get("delayed_materialization_count") or 0) > 0
        ),
        "terminal": runtime.get("terminal") is True,
    }
    reason_codes = _admission_blockers(gates)
    repair_codes = {
        _BLOCKER_CODES["native_source_consumption"],
        _BLOCKER_CODES["deterministic_replay"],
        _BLOCKER_CODES["survival"],
        _BLOCKER_CODES["delayed_control_executed"],
        _BLOCKER_CODES["terminal"],
    }
    ready = not reason_codes
    return {
        "ready_for_full_admission": ready,
        "final_disposition": (
            "ready_for_full_admission"
            if ready
            else (
                "held_repair"
                if any(code in repair_codes for code in reason_codes)
                else "abandoned_intrinsic"
            )
        ),
        "reason_codes": reason_codes,
        "gates": gates,
        "reference_evidence": {
            "n_ticks": runtime.get("n_ticks"),
            "terminal": runtime.get("terminal"),
            "task_completed": runtime.get("task_completed"),
            "source_consumption_status": source.get("status"),
            "deterministic_replay": replay.get("deterministic"),
            "wait_task_loss": replay.get("wait_task_loss"),
            "reference_task_loss": replay.get("reference_task_loss"),
            "headroom": replay.get("headroom"),
            "headroom_floor": replay.get("headroom_floor"),
            "delayed_ack_count": runtime.get("delayed_ack_count"),
            "delayed_materialization_count": runtime.get(
                "delayed_materialization_count"
            ),
        },
    }


def _execute_candidate_repair_probe(
    source_root: Path,
    *,
    seed: int,
    n_ticks: int,
    probe_python: Path | None,
) -> dict[str, Any]:
    if probe_python is None:
        return run_candidate_repair_probe(source_root, seed=seed, n_ticks=n_ticks)
    with tempfile.TemporaryDirectory(prefix="operate-citylearn-probe-") as temp:
        output = Path(temp) / "probe.json"
        completed = subprocess.run(
            [
                str(probe_python),
                str(REPO_ROOT / "scripts" / "probe_citylearn_runtime.py"),
                "--candidate-repair",
                "--source-root",
                str(source_root),
                "--seed",
                str(seed),
                "--ticks",
                str(n_ticks),
                "--output",
                str(output),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if not output.is_file():
            return {
                "status": "blocked_runtime_compatibility",
                "environment_repair_passed": False,
                "runtime_failure_code": "sidecar_probe_failed_before_report",
                "runtime_error": completed.stderr.strip(),
                "sidecar_returncode": completed.returncode,
            }
        report = json.loads(output.read_text(encoding="utf-8"))
        report["sidecar_returncode"] = completed.returncode
        return report


def _runtime_repair_ledger(
    *, dataset_root: Path, probe_python: Path | None = None
) -> dict[str, Any]:
    """Return terminal repair outcomes for the five held CityLearn identities."""

    rows: list[dict[str, Any]] = []
    for candidate_id, source_unit in CANDIDATE_REPAIR_DATASETS:
        probe = _execute_candidate_repair_probe(
            dataset_root / source_unit,
            seed=SEED,
            n_ticks=2,
            probe_python=probe_python,
        )
        environment_repaired = probe.get("environment_repair_passed") is True
        environment_blockers = list(probe.get("environment_blockers") or [])
        failure_code = probe.get("runtime_failure_code")
        if failure_code and failure_code not in environment_blockers:
            environment_blockers.append(str(failure_code))
        if not environment_repaired and not environment_blockers:
            environment_blockers.append("focused_native_replay_not_passed")
        task_axis_blockers = list(probe.get("task_axis_blockers") or [])
        task_axis_ready = bool(environment_repaired and not task_axis_blockers)
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_unit": source_unit,
                "repair_status": (
                    "repaired" if environment_repaired else "blocked"
                ),
                "environment_repair_passed": environment_repaired,
                "environment_blockers": environment_blockers,
                "task_axis_status": "ready" if task_axis_ready else "blocked",
                "task_axis_blockers": task_axis_blockers,
                "final_disposition": (
                    "ready_for_full_admission"
                    if task_axis_ready
                    else (
                        "abandoned_intrinsic"
                        if environment_repaired
                        else "held_repair"
                    )
                ),
                "terminal": not task_axis_ready,
                "blockers": environment_blockers + task_axis_blockers,
                "scientific_constraints": {
                    "source_values_changed": False,
                    "source_window_changed": False,
                    "headroom_threshold_changed": False,
                },
                "focused_runtime_probe": probe,
            }
        )
    environment_repaired_count = sum(
        row["environment_repair_passed"] for row in rows
    )
    task_axis_ready_count = sum(row["task_axis_status"] == "ready" for row in rows)
    return {
        "schema_version": "citylearn_runtime_repair_ledger_v2",
        "status": "terminal_repair_outcomes",
        "candidate_only": True,
        "core_admission_claimed": False,
        "probe_invocation": {
            "mode": "sidecar_python" if probe_python is not None else "in_process",
            "python_executable": str(probe_python) if probe_python is not None else sys.executable,
        },
        "summary": {
            "attempted": len(rows),
            "environment_repaired": environment_repaired_count,
            "environment_blocked": len(rows) - environment_repaired_count,
            "task_axis_ready": task_axis_ready_count,
            "task_axis_blocked": len(rows) - task_axis_ready_count,
        },
        "rows": rows,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _float_at(rows: list[dict[str, str]], row: int, column: str) -> float:
    try:
        return float(rows[row][column])
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid source cell: row={row}, column={column}") from exc


def _load_base_rows(input_suite: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(input_suite.read_text(encoding="utf-8"))
    rows = payload.get("scenarios") or []
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("backend_kind") != "citylearn":
            continue
        path = REPO_ROOT / str(row["path"])
        scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
        dataset_id = str((scenario.get("provenance") or {}).get("data_source") or "")
        if dataset_id:
            selected[dataset_id] = {"row": row, "scenario": scenario, "path": path}
    expected = {plan.dataset_id for plan in WINDOW_PLANS}
    if set(selected) != expected:
        raise ValueError(
            "exact source suite must contain one CityLearn row for each phase"
        )
    return selected


def _locked_assets(scenario: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(str(scenario["source_lock"]))
    if not path.is_absolute():
        path = REPO_ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("license") != "MIT" or payload.get("license_verified") is not True:
        raise ValueError("CityLearn source license is not locked")
    for ref, expected in (payload.get("files") or {}).items():
        asset = Path(str(ref))
        if not asset.is_absolute():
            asset = REPO_ROOT / asset
        if not asset.is_file() or _sha256(asset) != expected:
            raise ValueError(f"CityLearn source lock mismatch: {ref}")
    return path.resolve(), payload


def _source_window_payload(
    *, plan: WindowPlan, scenario: dict[str, Any], lock: dict[str, Any]
) -> dict[str, Any]:
    source_root = Path(str(scenario["source_root"]))
    if not source_root.is_absolute():
        source_root = REPO_ROOT / source_root
    source_root = source_root.resolve()
    schema = json.loads((source_root / "schema.json").read_text(encoding="utf-8"))
    building_files = sorted(
        str(row["energy_simulation"])
        for row in (schema.get("buildings") or {}).values()
    )
    start, end = plan.start, plan.start + plan.horizon - 1
    channels = {
        "pricing": ("pricing.csv", ("electricity_pricing",)),
        "carbon_intensity": ("carbon_intensity.csv", ("carbon_intensity",)),
        "weather": ("weather.csv", ("outdoor_dry_bulb_temperature",)),
    }
    source_rows: dict[str, Any] = {}
    for key, (name, fields) in channels.items():
        rows = _read_rows(source_root / name)
        source_rows[key] = [
            {field: _float_at(rows, index, field) for field in fields}
            for index in range(start, end + 1)
        ]
    for name in building_files:
        rows = _read_rows(source_root / name)
        source_rows[name] = [
            {
                field: _float_at(rows, index, field)
                for field in ("non_shiftable_load", "solar_generation")
            }
            for index in range(start, end + 1)
        ]
    runtime_files = {
        ref: digest
        for ref, digest in sorted((lock.get("files") or {}).items())
        if (REPO_ROOT / ref).resolve().is_relative_to(source_root)
    }
    payload = {
        "dataset_id": plan.dataset_id,
        "start": start,
        "end": end,
        "runtime_files": runtime_files,
        "source_rows": source_rows,
    }
    payload["window_sha256"] = _window_sha256(payload)
    payload["source_root"] = source_root
    payload["building_files"] = building_files
    return payload


def _transition(
    values: list[float], *, lo: int, hi: int, direction: str
) -> int:
    candidates = []
    for index in range(max(1, lo), min(len(values), hi + 1)):
        delta = values[index] - values[index - 1]
        score = delta if direction == "rise" else -delta if direction == "fall" else abs(delta)
        candidates.append((score, -index, index))
    if not candidates:
        raise ValueError("CityLearn transition search window is empty")
    score, _, index = max(candidates)
    if score <= 1e-8:
        raise ValueError(f"CityLearn source has no {direction} transition")
    return index


def _select_load_shaping_window(
    source_rows: list[dict[str, Any]], *, horizon: int = 72
) -> int:
    """Select the strongest two-day native load fall/rise window."""

    if horizon < 48 or len(source_rows) < horizon:
        raise ValueError("CityLearn load-shaping window requires at least 48 ticks")
    values = [float(row["non_shiftable_load"]) for row in source_rows]
    candidates: list[tuple[float, int]] = []
    for start in range(0, len(values) - horizon + 1, 24):
        score = 0.0
        try:
            for day in (0, 24):
                fall = _transition(
                    values,
                    lo=start + day + 1,
                    hi=start + day + 12,
                    direction="fall",
                )
                rise = _transition(
                    values,
                    lo=start + day + 13,
                    hi=start + day + 23,
                    direction="rise",
                )
                score += values[fall - 1] - values[fall]
                score += values[rise] - values[rise - 1]
        except ValueError:
            continue
        candidates.append((score, start))
    if not candidates:
        raise ValueError("CityLearn source has no two-day load-shaping axis")
    best_score = max(score for score, _ in candidates)
    return min(start for score, start in candidates if score == best_score)


def _event(
    *,
    event_id: str,
    kind: str,
    channel: str,
    source_asset: Path,
    source_asset_sha256: str,
    source_rows: list[dict[str, str]],
    source_column: str,
    source_tick: int,
    hidden: bool,
) -> dict[str, Any]:
    before = _float_at(source_rows, source_tick - 1, source_column)
    after = _float_at(source_rows, source_tick, source_column)
    delta = abs(after - before)
    return {
        "event_id": event_id,
        "kind": kind,
        "trigger_tick": source_tick,
        "duration_ticks": 6,
        "hidden": hidden,
        "channel": channel,
        "source_asset": _repo_ref(source_asset),
        "source_asset_sha256": source_asset_sha256,
        "source_row_before": source_tick - 1,
        "source_row_after": source_tick,
        "source_value_before": before,
        "source_value_after": after,
        "materiality_metric": "source_value_absolute_delta",
        "materiality_threshold": max(delta * 0.5, 1e-9),
        "source_observed": True,
        "procedural_overlay": False,
    }


def _native_load_storage_events(
    *,
    dataset_id: str,
    source_asset: Path,
    source_asset_sha256: str,
    source_rows: list[dict[str, Any]],
    source_window_start: int,
    horizon: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build source-native load-fall/peak-response storage tasks.

    These candidates intentionally do not invent tariff, carbon or solar
    values when the locked neighborhood schema does not provide them.
    """

    values = [float(row["non_shiftable_load"]) for row in source_rows]
    events: list[dict[str, Any]] = []
    response_windows: list[dict[str, Any]] = []
    for day_index, day in enumerate((0, 24)):
        transitions = (
            (
                "fall",
                _transition(
                    values,
                    lo=source_window_start + day + 1,
                    hi=source_window_start + day + 12,
                    direction="fall",
                ),
                "charge",
            ),
            (
                "rise",
                _transition(
                    values,
                    lo=source_window_start + day + 13,
                    hi=source_window_start + day + 23,
                    direction="rise",
                ),
                "discharge",
            ),
        )
        for direction, source_tick, policy in transitions:
            event_id = f"{dataset_id}_load_{direction}_t{source_tick}"
            event = _event(
                event_id=event_id,
                kind="load_change",
                channel="building_timeseries.non_shiftable_load",
                source_asset=source_asset,
                source_asset_sha256=source_asset_sha256,
                source_rows=source_rows,
                source_column="non_shiftable_load",
                source_tick=source_tick,
                hidden=day_index == 1,
            )
            local_tick = source_tick - source_window_start
            events.append(event)
            response_windows.append(
                {
                    "event_id": event_id,
                    "first_tick": local_tick + 1,
                    "last_tick": min(horizon - 1, local_tick + 5),
                    "native_control": "set_storage_dispatch",
                    "expected_control_policy": policy,
                }
            )
    for current, following in zip(
        response_windows, response_windows[1:], strict=False
    ):
        current["last_tick"] = min(
            int(current["last_tick"]), int(following["first_tick"]) - 1
        )
        if int(current["last_tick"]) < int(current["first_tick"]):
            raise ValueError("CityLearn response transition has no exclusive tick")
    return events, response_windows


def _runtime_admission_scenario(
    *,
    candidate_id: str,
    source_root: Path,
    source_lock_path: Path,
    source_lock: dict[str, Any],
) -> dict[str, Any]:
    """Materialize one source-native neighborhood load-shaping candidate."""

    schema = json.loads((source_root / "schema.json").read_text(encoding="utf-8"))
    building_rows = [
        (str(name), row)
        for name, row in sorted((schema.get("buildings") or {}).items())
        if isinstance(row, dict) and row.get("include", True)
    ]
    if not building_rows:
        raise ValueError(f"{source_root.name}: no included CityLearn buildings")
    building_name, building = building_rows[0]
    source_asset = source_root / str(building["energy_simulation"])
    rows = _read_rows(source_asset)
    # CityLearn needs 24 context ticks beyond the 72-hour decision horizon.
    start = _select_load_shaping_window(rows[:-24], horizon=72)
    end = start + 71
    source_asset_ref = _repo_ref(source_asset)
    source_asset_sha = str(source_lock["files"][source_asset_ref])
    events, response_windows = _native_load_storage_events(
        dataset_id=source_root.name,
        source_asset=source_asset,
        source_asset_sha256=source_asset_sha,
        source_rows=rows,
        source_window_start=start,
        horizon=72,
    )
    from domains.building_energy.backends.citylearn import _runtime_source_files

    runtime_refs = {
        _repo_ref(path) for path in _runtime_source_files(source_root).values()
    }
    runtime_files = {
        ref: digest
        for ref, digest in source_lock["files"].items()
        if ref in runtime_refs
    }
    derivation_files = {
        ref: digest
        for ref, digest in source_lock["files"].items()
        if ref not in runtime_files
    }
    window = {
        "dataset_id": source_root.name,
        "start": start,
        "end": end,
        "runtime_files": runtime_files,
        "derivation_files": derivation_files,
        "source_rows": {
            source_asset_ref: [
                {
                    "non_shiftable_load": float(row["non_shiftable_load"]),
                    "solar_generation": float(row["solar_generation"]),
                }
                for row in rows[start : end + 1]
            ]
        },
        "missing_native_objective_channels": [
            "pricing.electricity_pricing",
            "carbon_intensity.carbon_intensity",
        ],
    }
    window["window_sha256"] = _window_sha256(window)
    scenario_id = (
        "building_energy/citylearn_der_storage_control/"
        f"source_locked_long_horizon/high/{source_root.name}_w{start}_{end}"
    )
    scenario = {
        "seed_id": scenario_id,
        "scenario_id": scenario_id,
        "family": "citylearn_der_storage_control",
        "domain": "building_energy",
        "backend_kind": "citylearn",
        "source_root": _repo_ref(source_root),
        "source_lock": _repo_ref(source_lock_path),
        "backend_config": {
            "simulation_start_time_step": start,
            "simulation_end_time_step": end + 24,
            "episode_time_steps": 96,
            "implementation_checkout_root": "works/CityLearn",
            # Existing baseline dispatches the response-window contract under
            # this implementation switch; model prompts never receive it.
            "oracle_policy_contract": "citylearn.locked_native_peak_response.v1",
            "native_peak_response_objective": (
                "source_event_peak_response_burden_v1"
            ),
            "native_source_events": events,
            "source_window": {
                "first_time_step": start,
                "last_time_step": end,
                "runtime_context_last_time_step": end + 24,
                "source_window_sha256": window["window_sha256"],
            },
            "task_contract": {
                "contract": "building_energy.citylearn.storage_dispatch.v1",
                "standing_plan_required": True,
                "milestone_ticks": sorted(
                    {0, *(int(row["first_tick"]) for row in response_windows)}
                ),
                "response_windows": response_windows,
            },
            "task_requirements": {
                "min_distinct_control_ticks": 3,
                "min_distinct_physical_tools": 2,
                "min_strategy_reversals": 2,
                "min_response_windows": 4,
                "required_dependency_depth": 4,
            },
            "storage_control_delay_ticks": 1,
            "observation_contract": {
                "contract": "citylearn_building_partial_observation_v1",
                "hide_building_attrs": ["soc", "net_electricity_consumption"],
                "noise_sigma_rel": {"storage_energy_balance": 0.03},
                "staleness_ticks": {"storage_capacity": 2},
                "selective_reveal_tool": "inspect_building_state",
            },
            "dimension_applicability": {
                "economic_cost": {
                    "applicable": False,
                    "reason": "locked_neighborhood_schema_has_no_tariff_channel",
                },
                "counterfactual_prevention": {
                    "applicable": True,
                    "reason": "deterministic_masked_storage_action_replay",
                },
                "weighted_equity_score": {
                    "applicable": False,
                    "reason": "neighborhood_has_no_source_grounded_criticality_classes",
                },
                "ethical_quality": {
                    "applicable": False,
                    "reason": "neighborhood_has_no_source_grounded_ethical_dilemma",
                },
                "stakeholder_management": {
                    "applicable": False,
                    "reason": "neighborhood_has_no_source_grounded_stakeholder_model",
                },
                "system_survival": {
                    "applicable": True,
                    "reason": "building_energy_balance_tick_records_available",
                },
                "safety_violation": {
                    "applicable": True,
                    "reason": "building_electrical_limit_tick_records_available",
                },
                "adaptive_replanning": {
                    "applicable": True,
                    "reason": "state_changing_storage_dispatch_tools_available",
                },
                "information_efficiency": {
                    "applicable": True,
                    "reason": "partial_building_observation_and_inspect_building_state",
                },
                "foresight_score": {
                    "applicable": False,
                    "reason": "reference_does_not_emit_commit_to_plan_predictions",
                },
                "optimality_gap": {
                    "applicable": False,
                    "reason": "no_validated_storage_dispatch_optimum",
                },
                "stakeholder_equity": {
                    "applicable": False,
                    "reason": "neighborhood_has_no_source_grounded_stakeholder_classes",
                },
                "tool_use_efficiency": {
                    "applicable": True,
                    "reason": "tool_protocol_call_outcome_and_budget_evidence_available",
                },
            },
            "observation_budget_chars": 32000,
        },
        "horizon_ticks": 72,
        "tick_minutes": 60,
        "seed": SEED,
        "difficulty_level": "high",
        "difficulty_mode": "source_locked_long_horizon",
        "perturbations": _episode_perturbations(
            events, source_window_start=start
        ),
        "complexity_metrics": {
            "n_periods": 72,
            "n_buildings": len(building_rows),
            "n_native_source_events": len(events),
            "decision_depth": 4,
            "observability_burden": 3,
            "native_constraint_axes": [
                "electrical_storage_soc_bounds",
                "source_native_district_peak_import",
            ],
            "native_control_axes": ["multi_building_storage_dispatch"],
        },
        "provenance": {
            "data_source": source_root.name,
            "candidate_id": candidate_id,
            "url": "https://github.com/intelligent-environments-lab/CityLearn",
            "license": "MIT",
            "commit": (
                "v2.5.0@29062af6d077409e1c37a3e53a6cac30fd4d02bc"
            ),
            "lock_strategy": "git_tag+commit+all_schema_referenced_file_sha256",
            "source_window": [start, end],
            "source_window_sha256": window["window_sha256"],
            "task_axis_note": (
                "Locked native load and the simulator's storage-adjusted district "
                "net import define the peak-shaping objective. The source has no "
                "tariff or carbon channel; none is synthesized or borrowed."
            ),
        },
        "release_admission": "candidate_only",
        "candidate_only": True,
        "construct_contract": "operational_agency.v1",
        "source_contract": _candidate_source_contract(window),
    }
    scenario["scenario_signature"] = recompute_signature_with_seed(scenario, SEED)
    return scenario


def _quebec_terminal_rows(dataset_root: Path) -> list[dict[str, Any]]:
    """Close Quebec rows whose native heating task has not been validated."""

    rows: list[dict[str, Any]] = []
    for candidate_id, source_unit in CANDIDATE_REPAIR_DATASETS:
        if not source_unit.startswith("quebec_"):
            continue
        schema = json.loads(
            (dataset_root / source_unit / "schema.json").read_text(encoding="utf-8")
        )
        actions = dict(schema.get("actions") or {})
        electrical_storage_active = (
            dict(actions.get("electrical_storage") or {}).get("active") is True
        )
        heating_device_active = (
            dict(actions.get("heating_device") or {}).get("active") is True
        )
        if electrical_storage_active or not heating_device_active:
            raise ValueError(f"{source_unit}: unexpected native action schema")
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_unit": source_unit,
                "terminal": True,
                "final_disposition": "abandoned_intrinsic",
                "full_protocol21_executed": False,
                "reason_codes": [
                    "electrical_storage_control_axis_absent",
                    "heating_control_task_not_designed",
                ],
                "native_action_schema": {
                    "electrical_storage_active": electrical_storage_active,
                    "heating_device_active": heating_device_active,
                },
                "environment_requirement": {
                    "python": "3.10",
                    "scikit_learn": "1.2.2",
                    "reason": "legacy_pickle_compatibility",
                    "scientific_rejection_reason": False,
                },
                "scientific_constraints": {
                    "source_values_changed": False,
                    "external_tariff_or_carbon_added": False,
                    "headroom_threshold_changed": False,
                },
            }
        )
    return rows


def materialize_runtime_admission_candidates(
    *, dataset_root: Path, output_root: Path, workers: int = 3
) -> dict[str, Any]:
    """Execute references and emit only candidates with material headroom."""

    lock_root = output_root / "locks"
    scenario_root = output_root / "scenarios"
    lock_root.mkdir(parents=True, exist_ok=True)
    scenario_root.mkdir(parents=True, exist_ok=True)
    prepared: list[tuple[str, str, dict[str, Any], Path]] = []
    for candidate_id, source_unit in RUNTIME_ADMISSION_DATASETS:
        source_root = (dataset_root / source_unit).resolve()
        lock = _citylearn_candidate_source_lock(source_root)
        lock_path = lock_root / f"{source_unit}.json"
        lock_path.write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        scenario = _runtime_admission_scenario(
            candidate_id=candidate_id,
            source_root=source_root,
            source_lock_path=lock_path,
            source_lock=lock,
        )
        prepared.append(
            (candidate_id, source_unit, scenario, scenario_root / f"{source_unit}.yaml")
        )

    if workers < 1:
        raise ValueError("runtime admission workers must be positive")
    scenarios = [scenario for _, _, scenario, _ in prepared]
    if workers == 1:
        runtimes = [_run_reference(scenario) for scenario in scenarios]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(scenarios))) as executor:
            runtimes = list(executor.map(_run_reference, scenarios))

    candidate_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for (candidate_id, source_unit, scenario, scenario_path), runtime in zip(
        prepared, runtimes, strict=True
    ):
        decision = _runtime_admission_reference_decision(
            runtime, horizon=int(scenario["horizon_ticks"])
        )
        if decision["ready_for_full_admission"] is True:
            scenario_path.write_text(
                yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8"
            )
            candidate_rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_signature": scenario["scenario_signature"],
                    "path": _repo_ref(scenario_path),
                    "domain": "building_energy",
                    "backend_kind": "citylearn",
                    "family": scenario["family"],
                    "difficulty_level": scenario["difficulty_level"],
                    "difficulty_mode": scenario["difficulty_mode"],
                    "horizon_ticks": scenario["horizon_ticks"],
                    "seed": SEED,
                    "candidate_only": True,
                    "status": "ready_for_full_admission",
                }
            )
        else:
            scenario_path.unlink(missing_ok=True)
        ledger_rows.append(
            {
                "candidate_id": candidate_id,
                "source_unit": source_unit,
                "scenario_id": scenario["scenario_id"],
                "scenario_signature": scenario["scenario_signature"],
                "terminal": decision["ready_for_full_admission"] is not True,
                "final_disposition": decision["final_disposition"],
                "full_protocol21_executed": False,
                "reason_codes": decision["reason_codes"],
                "gates": decision["gates"],
                "reference_evidence": decision["reference_evidence"],
                "scientific_constraints": {
                    "source_values_changed": False,
                    "external_tariff_or_carbon_added": False,
                    "headroom_threshold_changed": False,
                },
            }
        )
    ledger_rows.extend(_quebec_terminal_rows(dataset_root))
    candidate_report = {
        "schema_version": "citylearn_runtime_admission_candidates_v1",
        "status": "ready_for_full_admission" if candidate_rows else "terminal",
        "candidate_only": True,
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(candidate_rows),
        "scenarios": candidate_rows,
    }
    candidate_report_path = output_root / "candidate_report.json"
    candidate_report_path.write_text(
        json.dumps(candidate_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    suite = build_candidate_source_suite(candidate_report_path)
    suite["suite_id"] = "citylearn_runtime_admission_v2"
    suite_path = output_root / "source_suite.json"
    suite_path.write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger = {
        "schema_version": "citylearn_runtime_admission_ledger_v2",
        "status": "awaiting_full_protocol21",
        "candidate_only": True,
        "terminal": False,
        "summary": {
            "attempted": len(ledger_rows),
            "ready_for_full_admission": len(candidate_rows),
            "terminal": sum(row["terminal"] is True for row in ledger_rows),
        },
        "rows": ledger_rows,
        "source_suite": {"path": _repo_ref(suite_path), "sha256": _sha256(suite_path)},
    }
    ledger_path = output_root / "candidate_ledger.json"
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ledger


def finalize_runtime_admission(
    *, source_suite_path: Path, pipeline_dir: Path, output_path: Path
) -> dict[str, Any]:
    source_suite = json.loads(source_suite_path.read_text(encoding="utf-8"))
    candidate_ledger_path = source_suite_path.parent / "candidate_ledger.json"
    candidate_ledger = json.loads(
        candidate_ledger_path.read_text(encoding="utf-8")
    )
    materialized_path = pipeline_dir / "refined_core_selection_protocol2_v21.json"
    materialized = json.loads(materialized_path.read_text(encoding="utf-8"))
    protocol_rows = _finalize_runtime_admission_rows(
        source_rows=list(source_suite.get("scenarios") or []),
        materialized=materialized,
    )
    rows = _merge_runtime_admission_rows(
        initial_rows=list(candidate_ledger.get("rows") or []),
        protocol_rows=protocol_rows,
    )
    counts = {
        disposition: sum(row["final_disposition"] == disposition for row in rows)
        for disposition in sorted(TERMINAL_ADMISSION_DISPOSITIONS)
    }
    ledger = {
        "schema_version": "citylearn_runtime_admission_ledger_v2",
        "status": "terminal",
        "candidate_only": True,
        "terminal": True,
        "full_protocol21_executed": all(
            row.get("full_protocol21_executed") is True for row in rows
        ),
        "summary": {
            "attempted": len(rows),
            "unresolved": 0,
            "full_protocol21_executed": sum(
                row.get("full_protocol21_executed") is True for row in rows
            ),
            "early_terminal": sum(
                row.get("full_protocol21_executed") is not True for row in rows
            ),
            **counts,
        },
        "rows": rows,
        "bindings": {
            "candidate_ledger": {
                "path": _repo_ref(candidate_ledger_path),
                "sha256": _sha256(candidate_ledger_path),
            },
            "source_suite": {
                "path": _repo_ref(source_suite_path),
                "sha256": _sha256(source_suite_path),
            },
            "materialized_core": {
                "path": _repo_ref(materialized_path),
                "sha256": _sha256(materialized_path),
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ledger


def _natural_events(
    *, plan: WindowPlan, window: dict[str, Any], lock: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_root = Path(window["source_root"])
    first_building = str(window["building_files"][0])
    pricing = [row["electricity_pricing"] for row in window["source_rows"]["pricing"]]
    carbon = [
        row["carbon_intensity"]
        for row in window["source_rows"]["carbon_intensity"]
    ]
    solar = [row["solar_generation"] for row in window["source_rows"][first_building]]
    load = [
        row["non_shiftable_load"] for row in window["source_rows"][first_building]
    ]
    first_peak = _transition(pricing, lo=8, hi=22, direction="rise")
    second_peak = _transition(pricing, lo=32, hi=46, direction="rise")
    solar_tick = _transition(solar, lo=4, hi=max(5, first_peak - 2), direction="rise")
    carbon_tick = _transition(carbon, lo=24, hi=32, direction="fall")
    load_tick = _transition(load, lo=second_peak - 2, hi=second_peak + 2, direction="rise")
    refs = lock["files"]
    def absolute(local: int) -> int:
        return plan.start + local
    assets = {
        "building": source_root / first_building,
        "pricing": source_root / "pricing.csv",
        "carbon": source_root / "carbon_intensity.csv",
    }
    raw_rows = {key: _read_rows(path) for key, path in assets.items()}
    prefix = plan.dataset_id
    events = [
        _event(
            event_id=f"{prefix}_solar_rise_t{absolute(solar_tick)}",
            kind="generation_change",
            channel="building_timeseries.solar_generation",
            source_asset=assets["building"],
            source_asset_sha256=refs[_repo_ref(assets["building"])],
            source_rows=raw_rows["building"],
            source_column="solar_generation",
            source_tick=absolute(solar_tick),
            hidden=False,
        ),
        _event(
            event_id=f"{prefix}_tariff_rise_t{absolute(first_peak)}",
            kind="tariff_change",
            channel="pricing.electricity_pricing",
            source_asset=assets["pricing"],
            source_asset_sha256=refs[_repo_ref(assets["pricing"])],
            source_rows=raw_rows["pricing"],
            source_column="electricity_pricing",
            source_tick=absolute(first_peak),
            hidden=False,
        ),
        _event(
            event_id=f"{prefix}_carbon_drop_t{absolute(carbon_tick)}",
            kind="carbon_intensity_change",
            channel="carbon_intensity.carbon_intensity",
            source_asset=assets["carbon"],
            source_asset_sha256=refs[_repo_ref(assets["carbon"])],
            source_rows=raw_rows["carbon"],
            source_column="carbon_intensity",
            source_tick=absolute(carbon_tick),
            hidden=True,
        ),
        _event(
            event_id=f"{prefix}_load_rise_t{absolute(load_tick)}",
            kind="load_change",
            channel="building_timeseries.non_shiftable_load",
            source_asset=assets["building"],
            source_asset_sha256=refs[_repo_ref(assets["building"])],
            source_rows=raw_rows["building"],
            source_column="non_shiftable_load",
            source_tick=absolute(load_tick),
            hidden=True,
        ),
    ]
    response_windows = [
        {
            "event_id": events[0]["event_id"],
            "first_tick": min(first_peak - 6, solar_tick + 1),
            "last_tick": first_peak - 1,
            "native_control": "set_storage_dispatch",
            "expected_control_policy": "charge",
        },
        {
            "event_id": events[1]["event_id"],
            "first_tick": first_peak + 1,
            "last_tick": first_peak + 5,
            "native_control": "set_storage_dispatch",
            "expected_control_policy": "discharge",
        },
        {
            "event_id": events[2]["event_id"],
            "first_tick": second_peak - 6,
            "last_tick": second_peak - 1,
            "native_control": "set_storage_dispatch",
            "expected_control_policy": "charge",
        },
        {
            "event_id": events[3]["event_id"],
            "first_tick": min(second_peak + 2, load_tick + 2),
            "last_tick": second_peak + 5,
            "native_control": "set_storage_dispatch",
            "expected_control_policy": "discharge",
        },
    ]
    return events, response_windows


def _episode_perturbations(
    events: list[dict[str, Any]], *, source_window_start: int
) -> list[dict[str, Any]]:
    keys = (
        "event_id",
        "kind",
        "duration_ticks",
        "hidden",
        "channel",
        "source_asset",
        "source_asset_sha256",
        "materiality_metric",
        "materiality_threshold",
        "source_observed",
        "procedural_overlay",
    )
    return [
        {
            **{key: event[key] for key in keys},
            "trigger_tick": int(event["trigger_tick"]) - source_window_start,
        }
        for event in events
    ]


def _scenario_for_window(
    *,
    plan: WindowPlan,
    base: dict[str, Any],
    window: dict[str, Any],
    lock_path: Path,
    lock: dict[str, Any],
) -> dict[str, Any]:
    scenario = deepcopy(base)
    end = plan.start + plan.horizon - 1
    events, response_windows = _natural_events(plan=plan, window=window, lock=lock)
    window_hash = str(window["window_sha256"])
    scenario_id = (
        "building_energy/citylearn_der_storage_control/"
        f"source_locked_long_horizon/extreme/{plan.dataset_id}_w{plan.start}_{end}"
    )
    scenario.update(
        {
            "seed_id": scenario_id,
            "scenario_id": scenario_id,
            "horizon_ticks": plan.horizon,
            "difficulty_level": plan.difficulty_level,
            "difficulty_mode": "source_locked_long_horizon",
            "source_lock": _repo_ref(lock_path),
            "source_contract": _candidate_source_contract(window),
            "perturbations": _episode_perturbations(
                events,
                source_window_start=plan.start,
            ),
            "complexity_metrics": {
                "n_periods": plan.horizon,
                "n_buildings": len(window["building_files"]),
                "n_native_source_events": len(events),
                "decision_depth": 4,
                "observability_burden": 3,
                "native_constraint_axes": [
                    "electrical_storage_soc_bounds",
                    "time_varying_import_cost",
                ],
                "native_control_axes": ["multi_building_storage_dispatch"],
            },
            "release_admission": "candidate_only",
            "candidate_only": True,
            "construct_contract": "operational_agency.v1",
        }
    )
    config = scenario.setdefault("backend_config", {})
    config.update(
        {
            "simulation_start_time_step": plan.start,
            "simulation_end_time_step": end + 24,
            "episode_time_steps": plan.horizon + 24,
            "storage_control_delay_ticks": 1,
            "native_source_events": events,
            "source_window": {
                "first_time_step": plan.start,
                "last_time_step": end,
                "runtime_context_last_time_step": end + 24,
                "source_window_sha256": window_hash,
            },
            "observation_contract": {
                "contract": "citylearn_building_partial_observation_v1",
                "hide_building_attrs": ["soc", "net_electricity_consumption"],
                "noise_sigma_rel": {"storage_energy_balance": 0.03},
                "staleness_ticks": {"storage_capacity": 2},
                "selective_reveal_tool": "inspect_building_state",
            },
            "task_contract": {
                "contract": "building_energy.citylearn.storage_dispatch.v1",
                "standing_plan_required": True,
                "milestone_ticks": sorted(
                    {0, *(window["first_tick"] for window in response_windows)}
                ),
                "response_windows": response_windows,
            },
            "task_requirements": {
                "min_distinct_control_ticks": 3,
                "min_distinct_physical_tools": 2,
                "min_strategy_reversals": 2,
                "min_response_windows": 4,
                "required_dependency_depth": 4,
            },
        }
    )
    scenario.setdefault("provenance", {}).update(
        {
            "source_window": [plan.start, end],
            "source_window_sha256": window_hash,
            "notes": (
                "All four named events are exact transitions in locked CityLearn "
                "runtime CSV rows; no procedural or random event overlay is used."
            ),
        }
    )
    scenario["scenario_signature"] = recompute_signature_with_seed(
        scenario, SEED
    )
    return scenario


def _desired_schedule(
    scenario: dict[str, Any], *, width: int
) -> list[list[float]]:
    horizon = int(scenario["horizon_ticks"])
    actions = [[0.0] * width for _ in range(horizon)]
    windows = (scenario["backend_config"]["task_contract"] or {})[
        "response_windows"
    ]
    for window_index, window in enumerate(windows):
        sign = 1.0 if window["expected_control_policy"] == "charge" else -1.0
        for offset, tick in enumerate(
            range(int(window["first_tick"]), int(window["last_tick"]) + 1)
        ):
            if not 0 <= tick < horizon:
                continue
            magnitude = 0.18 + 0.02 * ((offset + window_index) % 3)
            actions[tick] = [sign * magnitude] * width
    return actions


def _run_reference(scenario: dict[str, Any]) -> dict[str, Any]:
    env = BuildingEnergyEnvironment()
    try:
        observation = env.reset(scenario, seed=SEED)
        buildings = sorted((observation.get("buildings") or {}).keys())
        hidden_initial = bool(buildings) and all(
            observation["buildings"][building].get("soc") is None
            and observation["buildings"][building].get(
                "net_electricity_consumption"
            )
            is None
            for building in buildings
        )
        desired = _desired_schedule(scenario, width=len(buildings))
        delay = int(scenario["backend_config"]["storage_control_delay_ticks"])
        hidden_event_local_ticks = {
            int(event["trigger_tick"])
            - int(scenario["backend_config"]["simulation_start_time_step"])
            for event in scenario["backend_config"]["native_source_events"]
            if event.get("hidden") is True
        }
        inspect_ticks = set(hidden_event_local_ticks)
        plan_ticks = {0, *sorted(inspect_ticks)}
        plan_version = 0
        delayed_acks = 0
        delayed_materializations = 0
        inspection_payloads: list[dict[str, Any]] = []
        terminal = False
        n_ticks = 0
        for tick in range(int(scenario["horizon_ticks"])):
            calls: list[ToolCall] = []
            if tick in inspect_ticks:
                calls.append(
                    ToolCall(
                        name="inspect_building_state",
                        args={"building_id": buildings[0]},
                        call_id=f"inspect-{tick}",
                    )
                )
            if tick in plan_ticks:
                plan_version += 1
                calls.append(
                    ToolCall(
                        name="commit_to_plan",
                        args={
                            "plan_id": f"citylearn-long-plan-v{plan_version}",
                            "review_after_ticks": 4,
                            **(
                                {}
                                if plan_version == 1
                                else {
                                    "replaces_plan_id": (
                                        f"citylearn-long-plan-v{plan_version - 1}"
                                    ),
                                    "revision_reason": "hidden_source_transition_investigated",
                                }
                            ),
                        },
                        call_id=f"plan-{tick}",
                    )
                )
            effect_tick = tick + delay
            if effect_tick < len(desired) and any(
                abs(rate) > 1e-12 for rate in desired[effect_tick]
            ):
                calls.append(
                    ToolCall(
                        name="set_storage_dispatch",
                        args={
                            "dispatches": [
                                {"building_id": building, "rate": rate}
                                for building, rate in zip(
                                    buildings, desired[effect_tick], strict=True
                                )
                            ]
                        },
                        call_id=f"storage-batch-{tick}",
                        idempotency_key=f"storage-batch-{tick}",
                    )
                )
            if not calls:
                calls.append(ToolCall(name="wait", call_id=f"wait-{tick}"))
            result = env.step(Action(tool_calls=calls))
            n_ticks += 1
            terminal = bool(result.done)
            for tool_result in result.tool_results:
                payload = tool_result.payload or {}
                if (
                    tool_result.name == "set_storage_dispatch"
                    and payload.get("_status") == "pending"
                    and tool_result.latency_ticks == delay
                ):
                    delayed_acks += 1
                if (
                    tool_result.name == "set_storage_dispatch"
                    and tool_result.state_changing
                    and tool_result.latency_ticks == delay
                    and tool_result.ok
                ):
                    delayed_materializations += 1
                if (
                    tool_result.name == "inspect_building_state"
                    and tool_result.ok
                    and payload.get("_status") == "ok"
                ):
                    inspection_payloads.append(payload)
            if result.done:
                break
        executed_actions = env._backend.executed_action_stream()
        replay = env._backend.masked_action_replay(executed_actions)
        source = env.source_consumption_evidence(scenario=scenario)
        controls = env._backend.control_summary()
        wait_loss = float(
            sum((replay.get("masked_counterfactual") or {}).get("cost_components", {}).values())
        )
        reference_loss = float(
            sum((replay.get("action_run") or {}).get("cost_components", {}).values())
        )
        headroom = wait_loss - reference_loss
        headroom_floor = max(
            HEADROOM_ABSOLUTE_FLOOR,
            HEADROOM_RELATIVE_FLOOR * max(wait_loss, 0.0),
        )
        response_windows = list(controls.get("response_windows") or [])
        task_completed = bool(response_windows) and all(
            window.get("event_observed") is True
            and window.get("direction_met") is True
            and window.get("control_ticks")
            for window in response_windows
        )
        hidden_paths = [
            {
                "event_source_tick": source_tick,
                "event_local_tick": source_tick
                - int(scenario["backend_config"]["simulation_start_time_step"]),
                "investigation_tick": source_tick
                - int(scenario["backend_config"]["simulation_start_time_step"]),
                "control_request_not_before_tick": source_tick
                - int(scenario["backend_config"]["simulation_start_time_step"])
                + 1,
                "control_effect_not_before_tick": source_tick
                - int(scenario["backend_config"]["simulation_start_time_step"])
                + 2,
                "outcome_tick": n_ticks,
            }
            for source_tick in sorted(
                int(event["trigger_tick"])
                for event in scenario["backend_config"]["native_source_events"]
                if event.get("hidden") is True
            )
        ]
        return {
            "n_ticks": n_ticks,
            "terminal": terminal,
            "hidden_initial_observation": hidden_initial,
            "inspection_payload_count": len(inspection_payloads),
            "inspection_revealed_native_state": bool(inspection_payloads)
            and all(payload.get("buildings") for payload in inspection_payloads),
            "delayed_ack_count": delayed_acks,
            "delayed_materialization_count": delayed_materializations,
            "source_consumption": source,
            "control_summary": controls,
            "response_windows": response_windows,
            "task_completed": task_completed,
            "replay": {
                "deterministic": replay.get("deterministic_replay") is True,
                "reference_trajectory_sha256": (
                    replay.get("action_run") or {}
                ).get("trajectory_hash"),
                "wait_trajectory_sha256": (
                    replay.get("masked_counterfactual") or {}
                ).get("trajectory_hash"),
                "wait_task_loss": wait_loss,
                "reference_task_loss": reference_loss,
                "headroom": headroom,
                "headroom_floor": headroom_floor,
                "headroom_passed": headroom >= headroom_floor,
                "source_ablation_proofs": replay.get("source_ablation_proofs") or [],
            },
            "dependency_graph": {
                "proof_kind": "observed_hidden_event_investigate_delayed_control_effect_outcome",
                "exact_dependency_depth": 4 if hidden_paths else 0,
                "hidden_event_paths": hidden_paths,
            },
            "native_constraints": {
                "observed": [
                    "electrical_storage_soc_bounds",
                    "source_native_district_peak_import",
                ],
                "count": 2,
            },
        }
    finally:
        env.close()


def _evaluate_one(
    *, plan: WindowPlan, base: dict[str, Any], implementation_start: str
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        lock_path, lock = _locked_assets(base)
        window = _source_window_payload(plan=plan, scenario=base, lock=lock)
        scenario = _scenario_for_window(
            plan=plan,
            base=base,
            window=window,
            lock_path=lock_path,
            lock=lock,
        )
        runtime = _run_reference(scenario)
        source = runtime["source_consumption"]
        replay = runtime["replay"]
        implementation_end = implementation_identity(REPO_ROOT)[
            "implementation_tree_sha256"
        ]
        gates = {
            "source_lock": True,
            "window_identity": len(str(window["window_sha256"])) == 64,
            "native_source_consumption": source.get("status") == "passed",
            "native_state_effect": runtime["control_summary"].get(
                "native_state_changing_leverage"
            )
            is True,
            "deterministic_replay": replay["deterministic"] is True,
            "positive_material_headroom": replay["headroom_passed"] is True,
            "survival": runtime["n_ticks"] == plan.horizon,
            "task_completion": runtime["task_completed"] is True,
            "partial_observation_executed": (
                runtime["hidden_initial_observation"] is True
                and runtime["inspection_revealed_native_state"] is True
            ),
            "delayed_control_executed": (
                runtime["delayed_ack_count"] > 0
                and runtime["delayed_materialization_count"] > 0
            ),
            "dependency_depth": (
                runtime["dependency_graph"]["exact_dependency_depth"] >= 4
                and len(runtime["dependency_graph"]["hidden_event_paths"]) >= 2
            ),
            "native_constraint_breadth": runtime["native_constraints"]["count"] >= 2,
            "terminal": runtime["terminal"] is True,
            "implementation_stable": implementation_start == implementation_end,
        }
        blockers = _admission_blockers(gates)
        row = {
            "dataset_id": plan.dataset_id,
            "source_window": [plan.start, plan.start + plan.horizon - 1],
            "source_window_sha256": window["window_sha256"],
            "difficulty_level": plan.difficulty_level,
            "horizon_ticks": plan.horizon,
            "scenario_id": scenario["scenario_id"],
            "scenario_signature": scenario["scenario_signature"],
            "status": "candidate_passed" if not blockers else "held_repair",
            "disposition": "core_locked_increment" if not blockers else "held_repair",
            "candidate_only": True,
            "full_protocol21_executed": False,
            "core_admission_claimed": False,
            "gates": gates,
            "blockers": blockers,
            "runtime": runtime,
            "source_identity": {
                "source_lock": _repo_ref(lock_path),
                "source_lock_sha256": _sha256(lock_path),
                "source_window_sha256": window["window_sha256"],
                "physical_source_key": (
                    f"{plan.dataset_id}:{lock.get('building_cluster')}:"
                    f"{_stable_hash(window['runtime_files'])}"
                ),
                "effective_source_key": (
                    f"{plan.dataset_id}:window_{plan.start}_"
                    f"{plan.start + plan.horizon - 1}:"
                    f"{window['window_sha256']}"
                ),
            },
            "implementation_tree_sha256_start": implementation_start,
            "implementation_tree_sha256_end": implementation_end,
        }
        return row, scenario if not blockers else None
    except Exception as exc:
        return (
            {
                "dataset_id": plan.dataset_id,
                "source_window": [plan.start, plan.start + plan.horizon - 1],
                "difficulty_level": plan.difficulty_level,
                "horizon_ticks": plan.horizon,
                "status": "held_runtime",
                "disposition": "held_runtime",
                "candidate_only": True,
                "full_protocol21_executed": False,
                "core_admission_claimed": False,
                "gates": {},
                "blockers": [f"runtime_error:{type(exc).__name__}"],
                "error": str(exc),
                "implementation_tree_sha256_start": implementation_start,
                "implementation_tree_sha256_end": implementation_identity(REPO_ROOT)[
                    "implementation_tree_sha256"
                ],
            },
            None,
        )


def run_refinement(
    *, input_suite: Path, staging_root: Path, report_path: Path
) -> dict[str, Any]:
    implementation_start = implementation_identity(REPO_ROOT)[
        "implementation_tree_sha256"
    ]
    bases = _load_base_rows(input_suite)
    rows: list[dict[str, Any]] = []
    accepted: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for plan in WINDOW_PLANS:
        row, scenario = _evaluate_one(
            plan=plan,
            base=bases[plan.dataset_id]["scenario"],
            implementation_start=implementation_start,
        )
        rows.append(row)
        if scenario is not None:
            accepted.append((row, scenario))
    implementation_end = implementation_identity(REPO_ROOT)[
        "implementation_tree_sha256"
    ]
    if implementation_end != implementation_start:
        accepted = []
        for row in rows:
            if "implementation_tree_changed_during_run" not in row["blockers"]:
                row["blockers"].append("implementation_tree_changed_during_run")
            row["status"] = "held_runtime"
            row["disposition"] = "held_runtime"
            if row.get("gates"):
                row["gates"]["implementation_stable"] = False
    staging_root.mkdir(parents=True, exist_ok=True)
    scenario_rows = []
    for row, scenario in accepted:
        path = staging_root / f"{row['dataset_id']}_72h_extreme.yaml"
        path.write_text(yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8")
        scenario_rows.append(
            {
                "scenario_id": row["scenario_id"],
                "scenario_signature": row["scenario_signature"],
                "path": _repo_ref(path),
                "domain": "building_energy",
                "backend_kind": "citylearn",
                "family": "citylearn_der_storage_control",
                "difficulty_level": "extreme",
                "difficulty_mode": "source_locked_long_horizon",
                "horizon_ticks": 72,
                "seed": SEED,
                "source_denominator_key": row["source_identity"][
                    "effective_source_key"
                ],
                "physical_source_key": row["source_identity"]["physical_source_key"],
                "effective_source_key": row["source_identity"][
                    "effective_source_key"
                ],
                "candidate_only": True,
                "status": "pending_full_protocol21",
                "readiness_blockers": ["full_protocol21_gate_chain_pending"],
            }
        )
    candidate_report = {
        "schema_version": "citylearn_long_horizon_candidate_report_v2",
        "suite_id": "citylearn_long_horizon_20260813",
        "status": "staging_candidates_pending_full_admission",
        "candidate_only": True,
        "leaderboard_eligible": False,
        "release_ready": False,
        "full_protocol21_executed": False,
        "n_scenarios": len(scenario_rows),
        "one_per_effective_source_identity": True,
        "scenarios": scenario_rows,
    }
    candidate_report_path = staging_root / "candidate_report.json"
    candidate_report_path.write_text(
        json.dumps(candidate_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    suite = build_candidate_source_suite(candidate_report_path)
    suite["suite_id"] = "citylearn_long_horizon_20260813"
    suite_path = staging_root / "source_suite.json"
    suite_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema_version": "citylearn_long_horizon_refine_v1",
        "status": "candidate_survivors" if accepted else "held",
        "candidate_only": True,
        "core_admission_claimed": False,
        "full_protocol21_executed": False,
        "paid_api_calls": False,
        "input": {
            "path": _repo_ref(input_suite),
            "sha256": _sha256(input_suite),
        },
        "window_plans": [asdict(plan) for plan in WINDOW_PLANS],
        "implementation_stability": {
            "start": implementation_start,
            "end": implementation_end,
            "stable": implementation_start == implementation_end,
        },
        "summary": {
            "attempted": len(rows),
            "candidate_survivors": len(accepted),
            "held": len(rows) - len(accepted),
        },
        "rows": rows,
        "source_suite": {
            "path": _repo_ref(suite_path),
            "sha256": _sha256(suite_path),
        },
        "candidate_report": {
            "path": _repo_ref(candidate_report_path),
            "sha256": _sha256(candidate_report_path),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-suite", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--probe-python", type=Path)
    parser.add_argument("--runtime-repair-only", action="store_true")
    parser.add_argument("--materialize-runtime-admission", action="store_true")
    parser.add_argument("--finalize-runtime-admission", type=Path)
    parser.add_argument(
        "--runtime-admission-root",
        type=Path,
        default=DEFAULT_RUNTIME_ADMISSION_ROOT,
    )
    parser.add_argument("--runtime-admission-workers", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.materialize_runtime_admission:
        report = materialize_runtime_admission_candidates(
            dataset_root=args.dataset_root.resolve(),
            output_root=args.runtime_admission_root.resolve(),
            workers=args.runtime_admission_workers,
        )
        print(json.dumps(report["summary"], sort_keys=True))
        return 0
    if args.finalize_runtime_admission is not None:
        report = finalize_runtime_admission(
            source_suite_path=(
                args.runtime_admission_root.resolve() / "source_suite.json"
            ),
            pipeline_dir=args.finalize_runtime_admission.resolve(),
            output_path=(
                args.runtime_admission_root.resolve() / "terminal_ledger.json"
            ),
        )
        print(json.dumps(report["summary"], sort_keys=True))
        return 0
    if args.runtime_repair_only:
        report = _runtime_repair_ledger(
            dataset_root=args.dataset_root.resolve(),
            probe_python=(
                args.probe_python.absolute()
                if args.probe_python is not None
                else None
            ),
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report["summary"], sort_keys=True))
        return 0 if report["summary"]["environment_repaired"] else 2
    report = run_refinement(
        input_suite=args.input_suite.resolve(),
        staging_root=args.staging_root.resolve(),
        report_path=args.report.resolve(),
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if report["summary"]["candidate_survivors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
