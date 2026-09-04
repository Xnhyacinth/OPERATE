#!/usr/bin/env python3
"""Mine and calibrate source-consumed LV microgrid candidates.

This is a fail-closed staging pipeline. It selects high-variation windows from
the locked NSRDB/OEDI overlays, runs deterministic wait/reference replays, and
admits no row directly to Core. Rows that pass still require one-minimal replay,
semantic duplicate review, and model calibration before release selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.difficulty_contract import (  # noqa: E402
    DIFFICULTY_CONTRACT_VERSION,
    DIFFICULTY_REQUIREMENTS,
    evaluate_difficulty_contract,
)
from core.scenario_validator import validate_scenario_yaml  # noqa: E402
from domains.microgrid.backends.pandapower_lv import PandapowerLvBackend  # noqa: E402
from domains.microgrid.seeds.from_nrel_microgrid import (  # noqa: E402
    baked_overlay_provenance_report,
    load_overlay,
    site_is_anchored,
)
from domains.microgrid.seeds.from_pymgrid import (  # noqa: E402
    build_microgrid_lv_voltage_6h_seed,
    build_microgrid_lv_voltage_recovery_10h_seed,
    build_microgrid_lv_voltage_staged_6h_seed,
    source_window_sha256,
)
from run import run_one  # noqa: E402
from scripts.audit_core_difficulty import _semantic_fingerprint  # noqa: E402
from scripts.build_primary_suite import structural_fingerprint  # noqa: E402
from scripts.grounded_candidate_pipeline import (  # noqa: E402
    DIFFICULTY_FLOORS,
    _episode_replay_fingerprint,
)

PIPELINE_VERSION = "microgrid_source_consumed_v2"
CANDIDATE_DIR = REPO_ROOT / "release" / "dt_sched_bench_v0_52_0_candidate"
DEFAULT_OUTPUT = CANDIDATE_DIR / "grounded_microgrid_candidate_admission.json"
DEFAULT_FINAL_OUTPUT = CANDIDATE_DIR / "grounded_microgrid_post_minimality.json"
DEFAULT_MODEL_OUTPUT = CANDIDATE_DIR / "grounded_microgrid_model_diagnostic.json"
DEFAULT_SELECTION = CANDIDATE_DIR / "refined_core_selection_v4_source_grounded.json"
DEFAULT_SCENARIO_DIR = (
    REPO_ROOT / "scenarios" / "staging" / "v0_52_grounded_microgrid"
)
NREL_DIR = REPO_ROOT / "works" / "nrel-microgrid"
PHYSICAL_TOOLS = {
    "curtail_der",
    "set_battery_dispatch",
    "set_der_reactive_power",
    "shed_load",
}
DEFAULT_CASES = (
    ("denver_co", "medium", 42, 6, None),
    ("phoenix_az", "medium", 42, 6, 3269),
    ("boston_ma", "medium", 42, 6, 4035),
    ("nashville_tn", "medium", 42, 6, 4588),
    ("seattle_wa", "high", 42, 6, 4877),
    ("minneapolis_mn", "high", 42, 6, 4623),
    ("atlanta_ga", "high", 42, 6, 4263),
    ("albuquerque_nm", "extreme", 42, 10, 1590),
    ("phoenix_az", "extreme", 42, 10, None),
)


def _window_score(load: np.ndarray, pv: np.ndarray) -> float:
    pv_peak = float(np.max(pv))
    load_mean = float(np.mean(load))
    if pv_peak <= 0 or load_mean <= 0:
        return float("-inf")
    pv_variation = float(np.max(pv) - np.min(pv)) / pv_peak
    load_variation = float(np.max(load) - np.min(load)) / load_mean
    return pv_variation + load_variation


def rank_windows(site: str, *, horizon: int = 6, limit: int = 3) -> list[dict[str, Any]]:
    """Return deterministic, non-overlapping high-variation source windows."""
    path = NREL_DIR / f"{site}.npz"
    if not site_is_anchored(site, strict=True, min_horizon_ticks=horizon):
        raise ValueError(f"{site}: anchored source validation failed")
    data = np.load(path, allow_pickle=False)
    load = np.asarray(data["load_mw"], dtype=float).ravel()
    pv = np.asarray(data["pv_mw"], dtype=float).ravel()
    ranked = sorted(
        (
            (_window_score(load[start : start + horizon], pv[start : start + horizon]), start)
            for start in range(0, len(load) - horizon + 1)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    selected: list[dict[str, Any]] = []
    for score, start in ranked:
        if not np.isfinite(score):
            continue
        if any(abs(start - int(row["start_index"])) < horizon for row in selected):
            continue
        selected.append(
            {
                "site": site,
                "start_index": start,
                "window_score": round(float(score), 8),
                "load_mw": [round(float(value), 5) for value in load[start : start + horizon]],
                "pv_mw": [round(float(value), 5) for value in pv[start : start + horizon]],
            }
        )
        if len(selected) >= limit:
            break
    return selected


def source_window(site: str, *, horizon: int, start: int) -> dict[str, Any]:
    """Describe one deterministic window chosen by a behavioral scan."""
    path = NREL_DIR / f"{site}.npz"
    if not site_is_anchored(site, strict=True, min_horizon_ticks=horizon):
        raise ValueError(f"{site}: anchored source validation failed")
    data = np.load(path, allow_pickle=False)
    load = np.asarray(data["load_mw"], dtype=float).ravel()
    pv = np.asarray(data["pv_mw"], dtype=float).ravel()
    if start < 0 or start + horizon > len(load):
        raise ValueError(f"{site}: source window {start}:{start + horizon} is invalid")
    return {
        "site": site,
        "start_index": start,
        "window_score": round(
            _window_score(load[start : start + horizon], pv[start : start + horizon]),
            8,
        ),
        "load_mw": [
            round(float(value), 5) for value in load[start : start + horizon]
        ],
        "pv_mw": [
            round(float(value), 5) for value in pv[start : start + horizon]
        ],
        "selection_rule": "behavioral_cross_tick_recovery_scan_v1",
    }


def _source_proof(seed: dict[str, Any]) -> dict[str, Any]:
    backend = PandapowerLvBackend()
    from domains.microgrid.adapter import _rebuild_seed_from_dict

    seed_obj = _rebuild_seed_from_dict(seed, int(seed["seed"]))
    backend.reset(seed_obj)
    backend.tick(0)
    snapshot = backend.snapshot()
    proof = snapshot.get("source_profile") or {}
    config = seed.get("backend_config") or {}
    profiles = config.get("source_profiles") or {}
    expected_index = int(config.get("profile_start_index") or 0)
    site = str(config.get("site") or "")
    source_overlay = load_overlay(
        site,
        horizon_ticks=int(seed["horizon_ticks"]),
        seed=int(seed["seed"]),
        forecast_regime_idx=int(config.get("forecast_regime_idx") or 0),
        start_index=expected_index,
    )
    embedded_load = [float(value) for value in profiles.get("load_mw") or []]
    embedded_pv = [float(value) for value in profiles.get("pv_mw") or []]
    source_load = [float(value) for value in source_overlay["load_mw"]]
    source_pv = [float(value) for value in source_overlay["pv_mw"]]
    references = config.get("source_profile_reference") or {}
    source_reference_values_match = bool(embedded_load and embedded_pv) and (
        np.isclose(
            float(references.get("load_mw") or 0.0),
            float(np.mean(embedded_load)),
            rtol=0.0,
            atol=1e-8,
        )
        and np.isclose(
            float(references.get("pv_mw") or 0.0),
            float(np.max(embedded_pv)),
            rtol=0.0,
            atol=1e-8,
        )
    )
    recipe = config.get("derivation_recipe") or {}
    actual_window_sha256 = source_window_sha256(
        load_mw=embedded_load,
        pv_mw=embedded_pv,
    )
    sidecar = baked_overlay_provenance_report(NREL_DIR / f"{site}.npz")
    return {
        "applied": proof.get("applied") is True,
        "source_index_matches": proof.get("source_index") == expected_index,
        "load_value_matches": proof.get("load_mw") == profiles.get("load_mw", [None])[0],
        "pv_value_matches": proof.get("pv_mw") == profiles.get("pv_mw", [None])[0],
        "source_file_values_match": (
            embedded_load == source_load and embedded_pv == source_pv
        ),
        "source_reference_values_match": bool(source_reference_values_match),
        "source_window_sha256_matches": (
            recipe.get("source_window_sha256") == actual_window_sha256
            and (seed.get("provenance", {}).get("time_window") or {}).get(
                "source_window_sha256"
            )
            == actual_window_sha256
        ),
        "derivation_recipe_complete": (
            isinstance(recipe.get("profile_start_index"), int)
            and int(recipe["profile_start_index"]) >= 0
            and all(
                recipe.get(key)
                for key in (
                    "source_window_sha256",
                    "selection_rule",
                    "network",
                    "load_mapping",
                    "pv_mapping",
                    "magnitude_scope",
                    "stress_overlay_scope",
                    "stress_overlays",
                )
            )
        ),
        "source_sidecar_valid": bool(sidecar.get("valid")),
        "source_sidecar_sha256": (sidecar.get("metadata") or {}).get("sha256"),
        "source_window_sha256": actual_window_sha256,
        "observed": proof,
    }


def _event_reachability(seed: dict[str, Any]) -> list[dict[str, Any]]:
    """Prove each event changes native state when isolated from other events."""
    from domains.microgrid.adapter import _rebuild_seed_from_dict

    results: list[dict[str, Any]] = []
    for event_index, event in enumerate(seed.get("perturbations") or []):
        stressed_body = deepcopy(seed)
        stressed_body["perturbations"] = [deepcopy(event)]
        control_body = deepcopy(seed)
        control_body["perturbations"] = []
        stressed_seed = _rebuild_seed_from_dict(stressed_body, int(seed["seed"]))
        control_seed = _rebuild_seed_from_dict(control_body, int(seed["seed"]))
        stressed = PandapowerLvBackend()
        control = PandapowerLvBackend()
        stressed.reset(stressed_seed)
        control.reset(control_seed)
        trigger = int(event.get("trigger_tick") or 0)
        for tick in range(trigger + 1):
            stressed_record = stressed.tick(tick)
            control_record = control.tick(tick)
        stressed_totals = stressed.snapshot().get("totals") or {}
        control_totals = control.snapshot().get("totals") or {}
        deltas = {
            "demand_mw": abs(
                stressed_record.aggregate_demand_mw
                - control_record.aggregate_demand_mw
            ),
            "der_generation_mw": abs(
                float(stressed_totals.get("der_generation_mw") or 0.0)
                - float(control_totals.get("der_generation_mw") or 0.0)
            ),
            "rho_max": abs(stressed_record.rho_max - control_record.rho_max),
            "n_voltage_violations": abs(
                stressed_record.n_voltage_violations
                - control_record.n_voltage_violations
            ),
        }
        results.append(
            {
                "event_index": event_index,
                "kind": event.get("kind"),
                "trigger_tick": trigger,
                "hidden": bool(event.get("hidden")),
                "changes_native_state": any(
                    float(value) > 1e-8 for value in deltas.values()
                ),
                "native_state_deltas": deltas,
            }
        )
    return results


def _physical_source_key(seed: dict[str, Any]) -> str:
    config = seed.get("backend_config") or {}
    recipe = config.get("derivation_recipe") or {}
    return json.dumps(
        {
            "backend": seed.get("backend_kind"),
            "network": recipe.get("network"),
            "site": config.get("site"),
            "source_window_sha256": recipe.get("source_window_sha256"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _admission(
    seed: dict[str, Any],
    *,
    wait_first: dict[str, Any],
    wait_second: dict[str, Any],
    reference: dict[str, Any],
    existing_fingerprints: set[str],
    existing_physical_sources: set[str],
) -> dict[str, Any]:
    level = str(seed["difficulty_level"])
    floor = DIFFICULTY_FLOORS[level]
    source_proof = _source_proof(seed)
    task = reference.get("task_completion") or {}
    complexity = (reference.get("trajectory_summary") or {}).get("complexity") or {}
    effective_ticks = sorted(
        {int(value) for value in complexity.get("effective_control_ticks") or []}
    )
    physical_tools = sorted(
        set(complexity.get("observed_state_changing_tool_set") or []) & PHYSICAL_TOOLS
    )
    strategy_switches = int(complexity.get("control_strategy_switch_count") or 0)
    perturbations = list(seed.get("perturbations") or [])
    horizon = int(seed["horizon_ticks"])
    environment_ticks = {
        int(value)
        for value in (
            (wait_first.get("trajectory_summary") or {})
            .get("complexity", {})
            .get("environment_change_ticks")
            or []
        )
    }
    structural = structural_fingerprint(seed)
    physical_source = _physical_source_key(seed)
    counterfactual = reference.get("counterfactual") or {}
    event_reachability = _event_reachability(seed)
    difficulty_contract = evaluate_difficulty_contract(seed)
    visible_events = [
        event for event in perturbations if not bool(event.get("hidden"))
    ]
    checks = {
        "source_lock_complete": bool(
            seed.get("provenance", {}).get("url")
            and seed.get("provenance", {}).get("commit")
            and seed.get("provenance", {}).get("lock_strategy")
            and not any(
                str(path).startswith("<offline-synthesized:")
                for path in seed.get("provenance", {}).get("files") or []
            )
        ),
        "source_series_consumed_by_backend": all(
            bool(source_proof.get(key))
            for key in (
                "applied",
                "source_index_matches",
                "load_value_matches",
                "pv_value_matches",
                "source_file_values_match",
                "source_reference_values_match",
                "source_window_sha256_matches",
                "derivation_recipe_complete",
                "source_sidecar_valid",
            )
        ),
        "deterministic_wait_replay": (
            _episode_replay_fingerprint(wait_first)
            == _episode_replay_fingerprint(wait_second)
        ),
        "reference_task_completed": bool(task.get("applicable") and task.get("completed")),
        "counterfactual_material": float(counterfactual.get("prevented_loss") or 0.0) > 0,
        "events_inside_horizon": all(
            0 <= int(event.get("trigger_tick") or 0) < horizon
            and int(event.get("trigger_tick") or 0)
            + max(1, int(event.get("duration_ticks") or 1))
            <= horizon
            for event in perturbations
        ),
        "visible_events_observed": bool(visible_events)
        and all(
            int(event.get("trigger_tick") or 0) + 1 in environment_ticks
            for event in visible_events
        ),
        "all_events_change_native_state": bool(event_reachability)
        and all(row["changes_native_state"] for row in event_reachability),
        "difficulty_effective_tick_floor": (
            len(effective_ticks) >= int(floor["effective_ticks"])
        ),
        "difficulty_physical_tool_floor": (
            len(physical_tools) >= int(floor["physical_tools"])
        ),
        "difficulty_strategy_switch_floor": (
            strategy_switches >= int(floor["strategy_switches"])
        ),
        "not_structural_duplicate": structural not in existing_fingerprints,
        "independent_physical_source": (
            physical_source not in existing_physical_sources
        ),
        "static_difficulty_contract": (
            difficulty_contract["status"] == "pending"
            and not difficulty_contract["failures"]
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "preadmitted_pending_one_minimal" if not failures else "held",
        "checks": checks,
        "failures": failures,
        "source_consumption_proof": source_proof,
        "observed_strategy": {
            "physical_tool_set": physical_tools,
            "effective_control_ticks": effective_ticks,
            "control_strategy_switch_count": strategy_switches,
            "shortest_successful_tool_set": complexity.get("shortest_successful_tool_set"),
            "required_distinct_tool_count": complexity.get("required_distinct_tool_count"),
            "exact_dependency_depth": complexity.get("exact_dependency_depth"),
            "minimality_status": complexity.get("minimality_status"),
            "actual_interaction_turns": complexity.get(
                "actual_interaction_turns"
            ),
            "dependency_depth_status": complexity.get(
                "dependency_depth_status"
            ),
        },
        "difficulty_contract": difficulty_contract,
        "task_contract": task,
        "counterfactual": counterfactual,
        "event_reachability": event_reachability,
        "structural_fingerprint": structural,
        "physical_source_key": physical_source,
        "semantic_fingerprint": _semantic_fingerprint(seed),
    }


def build_report(
    *,
    selection_path: Path,
    scenario_dir: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    existing_fingerprints = {
        str(row["structural_fingerprint"])
        for row in selection.get("scenarios") or []
        if row.get("structural_fingerprint")
    }
    existing_physical_sources = {
        json.dumps(
            json.loads(str(row["source_key"])),
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in selection.get("scenarios") or []
        if row.get("domain") == "microgrid"
        and isinstance(row.get("source_key"), str)
        and str(row["source_key"]).startswith("{")
    }
    results: list[dict[str, Any]] = []
    for site, level, seed_value, horizon, preferred_start in DEFAULT_CASES:
        windows = (
            [source_window(site, horizon=horizon, start=preferred_start)]
            if preferred_start is not None
            else rank_windows(site, horizon=horizon, limit=8)
        )
        for window in windows:
            start = int(window["start_index"])
            if level == "extreme":
                seed_obj = build_microgrid_lv_voltage_recovery_10h_seed(
                    seed=seed_value,
                    seed_id=(
                        f"microgrid_lv_recovery_{site}_{level}_p{start}_s{seed_value}"
                    ),
                    difficulty_mode="time_pressure",
                    site=site,
                    source_profile_start_index=start,
                )
            elif level == "high":
                seed_obj = build_microgrid_lv_voltage_staged_6h_seed(
                    seed=seed_value,
                    seed_id=(
                        f"microgrid_lv_staged_{site}_{level}_p{start}_s{seed_value}"
                    ),
                    difficulty_mode="time_pressure",
                    site=site,
                    source_profile_start_index=start,
                )
            else:
                seed_obj = build_microgrid_lv_voltage_6h_seed(
                    seed=seed_value,
                    seed_id=(
                        f"microgrid_lv_source_{site}_{level}_p{start}_s{seed_value}"
                    ),
                    difficulty_level=level,
                    difficulty_mode="time_pressure",
                    site=site,
                    source_profile_start_index=start,
                )
            seed = seed_obj.to_dict()
            seed["scenario_id"] = (
                f"microgrid/{seed_obj.family}/time_pressure/{level}/"
                f"{seed_obj.seed_id}"
            )
            seed["scenario_signature"] = seed_obj.signature()
            seed["complexity_metrics"] = seed_obj.complexity_metrics()
            validation_errors = validate_scenario_yaml(seed)
            wait_first = run_one(seed, agent_name="wait_only")
            wait_second = run_one(seed, agent_name="wait_only")
            reference = run_one(seed, agent_name="oracle_offline")
            admission = _admission(
                seed,
                wait_first=wait_first,
                wait_second=wait_second,
                reference=reference,
                existing_fingerprints=existing_fingerprints,
                existing_physical_sources=existing_physical_sources,
            )
            scenario_path = (
                scenario_dir
                / "microgrid"
                / seed_obj.family
                / "time_pressure"
                / level
                / f"{seed_obj.seed_id}.yaml"
            )
            if (
                not validation_errors
                and admission["status"] == "preadmitted_pending_one_minimal"
            ):
                scenario_path.parent.mkdir(parents=True, exist_ok=True)
                scenario_path.write_text(
                    yaml.safe_dump(seed, sort_keys=False),
                    encoding="utf-8",
                )
            results.append(
                {
                    "scenario_id": seed["scenario_id"],
                    "path": str(scenario_path.relative_to(REPO_ROOT)),
                    "scenario_signature": seed["scenario_signature"],
                    "domain": "microgrid",
                    "backend_kind": "pandapower_lv",
                    "family": seed_obj.family,
                    "site": site,
                    "difficulty_level": level,
                    "source_window": window,
                    "source_sidecar": baked_overlay_provenance_report(
                        NREL_DIR / f"{site}.npz"
                    ),
                    "schema_validation_errors": validation_errors,
                    "admission": admission,
                }
            )
            if admission["status"] == "preadmitted_pending_one_minimal":
                break

    admitted_paths = {
        REPO_ROOT / str(row["path"])
        for row in results
        if row["admission"]["status"] == "preadmitted_pending_one_minimal"
    }
    stale_scenarios_removed = _reconcile_staging(
        scenario_dir,
        admitted_paths,
    )
    statuses = Counter(row["admission"]["status"] for row in results)
    scenarios = [
        {
            key: row[key]
            for key in (
                "scenario_id",
                "path",
                "scenario_signature",
                "domain",
                "backend_kind",
                "family",
                "difficulty_level",
            )
        }
        for row in results
        if row["admission"]["status"] == "preadmitted_pending_one_minimal"
    ]
    return {
        "schema_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "status": "staging_only_not_core",
        "input_selection": str(selection_path.relative_to(REPO_ROOT)),
        "n_attempted": len(results),
        "n_preadmitted_pending_one_minimal": statuses.get(
            "preadmitted_pending_one_minimal", 0
        ),
        "n_held": statuses.get("held", 0),
        "stale_scenarios_removed": [
            str(path.relative_to(REPO_ROOT)) for path in stale_scenarios_removed
        ],
        "policy": {
            "no_llm_as_environment": True,
            "source_series_must_drive_backend_state": True,
            "quality_over_quantity": True,
            "core_promotion_automatic": False,
            "hy3_allowed_only_after_static_behavioral_gates": True,
            "single_model_failure_can_reject_data": False,
            "model_success_required_for_data_quality": False,
            "model_evidence_role": (
                "difficulty_and_discrimination_calibration_only"
            ),
        },
        "difficulty_floors": DIFFICULTY_FLOORS,
        "difficulty_contract_version": DIFFICULTY_CONTRACT_VERSION,
        "scenarios": scenarios,
        "results": results,
    }


def _reconcile_staging(
    scenario_dir: Path,
    admitted_paths: set[Path],
) -> list[Path]:
    """Remove stale YAMLs only from this pipeline's dedicated staging root."""
    root = scenario_dir.resolve()
    if not root.exists():
        return []
    protected_roots = {
        Path("/").resolve(),
        REPO_ROOT.resolve(),
        REPO_ROOT.parent.resolve(),
    }
    if root in protected_roots:
        raise ValueError(f"refusing broad staging reconciliation root: {root}")
    admitted = {path.resolve() for path in admitted_paths}
    removed: list[Path] = []
    for path in sorted(root.rglob("microgrid_lv_source_*.yaml")):
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"staging path escapes dedicated root: {path}")
        if resolved in admitted:
            continue
        path.unlink()
        removed.append(path)
    return removed


def finalize_with_minimality(
    admission_report: dict[str, Any],
    minimality_report: dict[str, Any],
) -> dict[str, Any]:
    replay_by_id = {
        str(row["scenario_id"]): row
        for row in minimality_report.get("results") or []
    }
    scenario_by_id = {
        str(row["scenario_id"]): row
        for row in admission_report.get("scenarios") or []
    }
    results: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for candidate in admission_report.get("results") or []:
        scenario_id = str(candidate["scenario_id"])
        replay = replay_by_id.get(scenario_id) or {}
        minimization = replay.get("replay_minimization") or {}
        level = str(candidate["difficulty_level"])
        floor = DIFFICULTY_FLOORS[level]
        requirements = DIFFICULTY_REQUIREMENTS[level]
        ticks = list(minimization.get("one_minimal_decision_ticks") or [])
        tools = sorted(
            set(minimization.get("one_minimal_successful_tool_set") or [])
            & PHYSICAL_TOOLS
        )
        switches = int(
            (candidate.get("admission") or {})
            .get("observed_strategy", {})
            .get("control_strategy_switch_count")
            or 0
        )
        checks = {
            "pre_admission_passed": (
                candidate.get("admission", {}).get("status")
                == "preadmitted_pending_one_minimal"
            ),
            "minimality_report_complete": (
                minimality_report.get("status") == "complete"
                and replay.get("status") == "complete"
            ),
            "scenario_signature_matches": (
                replay.get("scenario_signature")
                == candidate.get("scenario_signature")
            ),
            "one_minimal_replay_proven": minimization.get("status") == "one_minimal",
            "one_minimal_tick_floor": len(ticks) >= int(floor["effective_ticks"]),
            "one_minimal_physical_tool_floor": (
                len(tools) >= int(floor["physical_tools"])
            ),
            "strategy_switch_floor": switches >= int(floor["strategy_switches"]),
            "static_difficulty_contract": (
                (candidate.get("admission") or {})
                .get("difficulty_contract", {})
                .get("status")
                == "pending"
                and not (
                    (candidate.get("admission") or {})
                    .get("difficulty_contract", {})
                    .get("failures")
                    or []
                )
            ),
            "exact_dependency_depth_floor": (
                minimization.get("exact_dependency_depth") is not None
                and int(minimization["exact_dependency_depth"])
                >= requirements.min_dependency_depth
            ),
            "scenario_staged": scenario_id in scenario_by_id,
        }
        failures = [name for name, passed in checks.items() if not passed]
        status = "pending_model_discrimination" if not failures else "held"
        results.append(
            {
                "scenario_id": scenario_id,
                "difficulty_level": level,
                "status": status,
                "checks": checks,
                "failures": failures,
                "one_minimal": {
                    "decision_ticks": ticks,
                    "physical_tool_set": tools,
                    "distinct_physical_tool_count": len(tools),
                    "non_meta_call_count": minimization.get(
                        "one_minimal_non_meta_call_count"
                    ),
                    "non_meta_call_count_lower_bound": minimization.get(
                        "non_meta_call_count_lower_bound"
                    ),
                    "non_meta_call_count_upper_bound": minimization.get(
                        "non_meta_call_count_upper_bound"
                    ),
                    "claim": minimization.get("claim"),
                    "global_shortest_successful_tool_set": minimization.get(
                        "global_shortest_successful_tool_set"
                    ),
                    "exact_dependency_depth": minimization.get(
                        "exact_dependency_depth"
                    ),
                    "dependency_depth_status": minimization.get(
                        "dependency_depth_status"
                    ),
                },
            }
        )
        if status == "pending_model_discrimination":
            pending.append(scenario_by_id[scenario_id])
    return {
        "schema_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete_staging_only_not_core",
        "release_membership_changed": False,
        "quality_admission_basis": [
            "source_and_provenance",
            "deterministic_native_backend",
            "reference_solvability",
            "material_improvement_over_wait",
            "tool_event_and_interaction_contracts",
            "independence_and_duplicate_gates",
        ],
        "model_evidence_role": "difficulty_and_discrimination_calibration_only",
        "single_model_failure_can_reject_data": False,
        "n_candidates": len(results),
        "n_pending_model_discrimination": len(pending),
        "status_counts": dict(
            sorted(Counter(row["status"] for row in results).items())
        ),
        "scenarios": pending,
        "results": results,
    }


def diagnose_model_runs(
    post_report: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Flag successful model strategies that undercut a declared tier floor."""
    scenario_by_id = {
        str(row["scenario_id"]): row
        for row in post_report.get("scenarios") or []
    }
    results: list[dict[str, Any]] = []
    for candidate in post_report.get("results") or []:
        scenario_id = str(candidate["scenario_id"])
        scenario = scenario_by_id.get(scenario_id) or {}
        signature = scenario.get("scenario_signature")
        level = str(candidate["difficulty_level"])
        floor = DIFFICULTY_FLOORS[level]
        matched = [
            row
            for row in episodes
            if row.get("scenario_signature") == signature
            and row.get("status") == "ok"
        ]
        successful = [
            row for row in matched if (row.get("task_completion") or {}).get("completed")
        ]
        failed_task_trials = len(matched) - len(successful)
        contradictions: list[dict[str, Any]] = []
        for row in successful:
            complexity = (row.get("trajectory_summary") or {}).get("complexity") or {}
            effective_ticks = sorted(
                {
                    int(value)
                    for value in complexity.get("effective_control_ticks") or []
                }
            )
            physical_tools = sorted(
                set(complexity.get("observed_state_changing_tool_set") or [])
                & PHYSICAL_TOOLS
            )
            strategy_switches = int(
                complexity.get("control_strategy_switch_count") or 0
            )
            if (
                len(effective_ticks) < int(floor["effective_ticks"])
                or len(physical_tools) < int(floor["physical_tools"])
                or strategy_switches < int(floor["strategy_switches"])
            ):
                contradictions.append(
                    {
                        "model": row.get("model"),
                        "evaluation_implementation_fingerprint": row.get(
                            "evaluation_implementation_fingerprint"
                        ),
                        "effective_control_ticks": effective_ticks,
                        "effective_control_tick_count": len(effective_ticks),
                        "physical_tool_set": physical_tools,
                        "physical_tool_count": len(physical_tools),
                        "strategy_switch_count": strategy_switches,
                    }
                )
        pre_model_quality_passed = (
            candidate.get("status") == "pending_model_discrimination"
        )
        if not pre_model_quality_passed:
            status = "pre_model_quality_failure"
            recommended_action = "repair_or_retire_from_pre_model_evidence"
        elif contradictions:
            status = "difficulty_label_review"
            recommended_action = "review_or_relabel_difficulty"
        else:
            status = "model_calibration_pending"
            recommended_action = "collect_independent_model_repeats"
        results.append(
            {
                "scenario_id": scenario_id,
                "scenario_signature": signature,
                "difficulty_level": level,
                "status": status,
                "data_quality_status": (
                    "pre_model_quality_gates_passed"
                    if pre_model_quality_passed
                    else "pre_model_quality_gates_failed"
                ),
                "data_quality_rejected": not pre_model_quality_passed,
                "recommended_action": recommended_action,
                "matched_trials": len(matched),
                "successful_trials": len(successful),
                "failed_task_trials": failed_task_trials,
                "models": sorted(
                    {
                        str(row.get("model"))
                        for row in matched
                        if row.get("model")
                    }
                ),
                "difficulty_floor": floor,
                "difficulty_contradictions": contradictions,
            }
        )
    return {
        "schema_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "status": "diagnostic_only_not_core",
        "release_membership_changed": False,
        "required_formal_model_gate": {
            "independent_model_families": 3,
            "repeats_per_scenario_model": 3,
            "universal_model_success_required": False,
            "purpose": (
                "estimate score spread, item discrimination, saturation, and "
                "difficulty; not require every model to solve every sample"
            ),
        },
        "policy": {
            "single_model_failure_can_reject_data": False,
            "single_model_success_can_admit_data": False,
            "model_outcomes_override_pre_model_quality_gates": False,
            "successful_shorter_strategy_action": (
                "review_or_relabel_difficulty_not_reject_quality"
            ),
        },
        "status_counts": dict(
            sorted(Counter(row["status"] for row in results).items())
        ),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--finalize-minimality", type=Path)
    parser.add_argument("--final-output", type=Path, default=DEFAULT_FINAL_OUTPUT)
    parser.add_argument("--model-episodes", type=Path)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_OUTPUT)
    args = parser.parse_args()
    if args.model_episodes:
        episodes = [
            json.loads(line)
            for line in args.model_episodes.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        diagnostic = diagnose_model_runs(
            json.loads(args.final_output.read_text(encoding="utf-8")),
            episodes,
        )
        temporary = args.model_output.with_suffix(args.model_output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(diagnostic, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(args.model_output)
        print(
            json.dumps(
                {
                    "status": diagnostic["status"],
                    "status_counts": diagnostic["status_counts"],
                },
                indent=2,
            )
        )
        return
    if args.finalize_minimality:
        finalized = finalize_with_minimality(
            json.loads(args.output.read_text(encoding="utf-8")),
            json.loads(args.finalize_minimality.read_text(encoding="utf-8")),
        )
        temporary = args.final_output.with_suffix(args.final_output.suffix + ".tmp")
        temporary.write_text(json.dumps(finalized, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.final_output)
        print(
            json.dumps(
                {
                    key: finalized[key]
                    for key in (
                        "status",
                        "n_candidates",
                        "n_pending_model_discrimination",
                        "status_counts",
                    )
                },
                indent=2,
            )
        )
        return
    report = build_report(
        selection_path=args.selection.resolve(),
        scenario_dir=args.scenario_dir.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "n_attempted",
                    "n_preadmitted_pending_one_minimal",
                    "n_held",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
