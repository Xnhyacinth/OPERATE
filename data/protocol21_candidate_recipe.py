"""Fail-closed validation for source-first Protocol-2.1 candidate recipes.

This module validates staging metadata only.  It does not execute a backend and
never treats a declared source path as proof that a runtime consumed it; the
runtime section must already carry observed native evidence from a real audit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

_ALLOWED_LEVELS = {"basic", "medium", "high", "extreme"}
_DIFFICULTY_FLOORS: dict[str, tuple[int, int, int]] = {
    "basic": (1, 1, 0),
    "medium": (1, 1, 0),
    "high": (2, 2, 1),
    "extreme": (3, 2, 2),
}


def _resolve_asset(repo_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else repo_root / path


def _has_source_lock(source_lock: Mapping[str, Any]) -> bool:
    return bool(
        str(source_lock.get("url") or "").strip()
        and str(source_lock.get("license") or "").strip()
        and str(
            source_lock.get("version")
            or source_lock.get("commit")
            or source_lock.get("release")
            or ""
        ).strip()
        and isinstance(source_lock.get("files"), list)
        and bool(source_lock.get("files"))
    )


def _response_window_valid(replay: Mapping[str, Any]) -> bool:
    event_tick = replay.get("event_tick")
    if event_tick is None:
        return True
    try:
        event = int(event_tick)
        terminal = int(replay["terminal_tick"])
        response_ticks = [int(tick) for tick in replay.get("response_ticks") or []]
    except (KeyError, TypeError, ValueError):
        return False
    return event < terminal and any(event < tick <= terminal for tick in response_ticks)


def _difficulty_floor_valid(
    level: str,
    replay: Mapping[str, Any],
) -> bool:
    floor = _DIFFICULTY_FLOORS.get(level)
    if floor is None:
        return False
    ticks = {
        int(tick)
        for tick in replay.get("effective_decision_ticks") or []
        if isinstance(tick, (int, str)) and str(tick).lstrip("-").isdigit()
    }
    control_types = {
        str(value) for value in replay.get("native_control_types") or [] if str(value)
    }
    try:
        switches = int(replay.get("strategy_switch_count", 0))
    except (TypeError, ValueError):
        return False
    min_ticks, min_controls, min_switches = floor
    return (
        len(ticks) >= min_ticks
        and len(control_types) >= min_controls
        and switches >= min_switches
    )


def validate_recipe(
    recipe: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Validate one candidate recipe and return a non-mutating audit record."""
    source_lock = recipe.get("source_lock")
    runtime = recipe.get("runtime")
    task_contract = recipe.get("task_contract")
    replay = recipe.get("replay")
    independence = recipe.get("independence")
    source_lock = source_lock if isinstance(source_lock, Mapping) else {}
    runtime = runtime if isinstance(runtime, Mapping) else {}
    task_contract = task_contract if isinstance(task_contract, Mapping) else {}
    replay = replay if isinstance(replay, Mapping) else {}
    independence = independence if isinstance(independence, Mapping) else {}

    reason_codes: list[str] = []
    source_files = list(source_lock.get("files") or [])
    runtime_assets = list(runtime.get("runtime_assets") or [])
    source_lock_valid = _has_source_lock(source_lock)
    runtime_native = runtime.get("native") is True
    runtime_passed = str(runtime.get("status") or "") == "passed"
    source_consumed = runtime.get("consumes_source") is True
    state_effect = runtime.get("state_effect_observed") is True
    source_assets_exist = bool(source_files) and all(
        _resolve_asset(repo_root, value).is_file() for value in source_files
    )
    runtime_assets_exist = bool(runtime_assets) and all(
        _resolve_asset(repo_root, value).is_file() for value in runtime_assets
    )
    task_passed = (
        str(task_contract.get("status") or "") == "passed"
        and task_contract.get("completed") is True
        and task_contract.get("applicable") is True
    )
    replay_deterministic = replay.get("deterministic") is True
    replay_counterfactual = replay.get("counterfactual") is True
    response_window = _response_window_valid(replay)
    level = str(recipe.get("difficulty_level") or "").strip().lower()
    difficulty_valid = level in _ALLOWED_LEVELS and _difficulty_floor_valid(level, replay)
    source_key = str(
        recipe.get("source_denominator_key")
        or independence.get("source_denominator_key")
        or ""
    ).strip()
    physical_key = str(independence.get("physical_source_key") or "").strip()
    structural_fp = str(independence.get("structural_fingerprint") or "").strip()

    if not source_lock_valid:
        reason_codes.append("source_lock_incomplete")
    if not source_assets_exist:
        reason_codes.append("source_asset_missing")
    if not runtime_native or not runtime_passed or not source_consumed:
        reason_codes.append("external_data_without_native_runtime")
    if not runtime_assets_exist:
        reason_codes.append("runtime_asset_missing")
    if not state_effect:
        reason_codes.append("native_state_effect_unproven")
    if not task_passed:
        reason_codes.append("task_contract_not_passed")
    if not replay_deterministic:
        reason_codes.append("deterministic_replay_missing")
    if not replay_counterfactual:
        reason_codes.append("counterfactual_replay_missing")
    if not response_window:
        reason_codes.append("no_executable_response_window")
    if not difficulty_valid:
        reason_codes.append("difficulty_floor_not_met")
    if not source_key:
        reason_codes.append("source_denominator_key_missing")
    if not physical_key:
        reason_codes.append("physical_source_key_missing")
    if not structural_fp:
        reason_codes.append("structural_fingerprint_missing")

    checks = {
        "source_lock_complete": source_lock_valid,
        "source_assets_exist": source_assets_exist,
        "native_runtime_passed": runtime_native and runtime_passed,
        "source_consumption_observed": source_consumed,
        "runtime_assets_exist": runtime_assets_exist,
        "native_state_effect_observed": state_effect,
        "task_contract_passed": task_passed,
        "deterministic_replay": replay_deterministic,
        "counterfactual_replay": replay_counterfactual,
        "response_window": response_window,
        "difficulty_floor": difficulty_valid,
        "source_denominator_key": bool(source_key),
        "physical_source_key": bool(physical_key),
        "structural_fingerprint": bool(structural_fp),
    }
    return {
        "scenario_id": str(recipe.get("scenario_id") or ""),
        "status": "ready" if not reason_codes else "held",
        "reason_codes": sorted(set(reason_codes)),
        "checks": checks,
        "source_denominator_key": source_key or None,
        "physical_source_key": physical_key or None,
        "structural_fingerprint": structural_fp or None,
        "difficulty_level": level or None,
    }


def validate_recipes(
    recipes: Iterable[Mapping[str, Any]],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Validate recipes and apply batch-level source/fingerprint uniqueness."""
    reports = [validate_recipe(recipe, repo_root=repo_root) for recipe in recipes]
    source_keys = [
        str(report["source_denominator_key"])
        for report in reports
        if report.get("source_denominator_key")
    ]
    fingerprints = [
        str(report["structural_fingerprint"])
        for report in reports
        if report.get("structural_fingerprint")
    ]
    physical_keys = [
        str(report["physical_source_key"])
        for report in reports
        if report.get("physical_source_key")
    ]
    duplicate_source_keys = sorted(
        key for key, count in Counter(source_keys).items() if count > 1
    )
    duplicate_fingerprints = sorted(
        key for key, count in Counter(fingerprints).items() if count > 1
    )
    duplicate_physical_keys = sorted(
        key for key, count in Counter(physical_keys).items() if count > 1
    )
    if duplicate_source_keys or duplicate_fingerprints or duplicate_physical_keys:
        for report in reports:
            if report["source_denominator_key"] in duplicate_source_keys:
                report["reason_codes"].append("duplicate_source_denominator_key")
            if report["structural_fingerprint"] in duplicate_fingerprints:
                report["reason_codes"].append("duplicate_structural_fingerprint")
            if report["physical_source_key"] in duplicate_physical_keys:
                report["reason_codes"].append("duplicate_physical_source_key")
            report["reason_codes"] = sorted(set(report["reason_codes"]))
            report["status"] = "held"
    n_ready = sum(report["status"] == "ready" for report in reports)
    return {
        "schema_version": "protocol21-candidate-recipe-1",
        "status": "ready" if n_ready == len(reports) else "held",
        "n_expected": len(reports),
        "n_ready": n_ready,
        "n_held": len(reports) - n_ready,
        "duplicate_source_denominator_keys": duplicate_source_keys,
        "duplicate_structural_fingerprints": duplicate_fingerprints,
        "duplicate_physical_source_keys": duplicate_physical_keys,
        "results": reports,
    }
