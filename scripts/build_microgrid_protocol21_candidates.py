#!/usr/bin/env python3
"""Materialize source-grounded Microgrid staging candidates.

The builders in :mod:`domains.microgrid.seeds.from_pymgrid` consume the
anchored NREL/OEDI profile window and expose native pandapower LV controls.
This helper only creates a small, deterministic staging cohort; it never
promotes rows into a release or weakens any Protocol-2.1 gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.source_asset_contract import (  # noqa: E402
    physical_source_lock_from_contract,
    resolve_source_asset_contract,
)
from domains.microgrid.seeds.from_pymgrid import (  # noqa: E402
    build_microgrid_lv_voltage_6h_seed,
    build_microgrid_lv_voltage_recovery_10h_seed,
    build_microgrid_lv_voltage_staged_6h_seed,
)
from domains.microgrid.seeds.schema import Perturbation  # noqa: E402
from runner.resume import recompute_signature_with_seed  # noqa: E402
from scripts.audit_core_difficulty import _semantic_fingerprint  # noqa: E402
from scripts.build_primary_suite import structural_fingerprint  # noqa: E402
from scripts.prepare_protocol21_working_set import _source_contract  # noqa: E402

DEFAULT_STAGING = (
    REPO_ROOT / "scenarios" / "staging" / "v0_52_protocol21_microgrid_native"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "microgrid_protocol21_candidates_v1.json"
)


# The profile offsets are deliberately explicit and are part of each source
# denominator key.  They are daylight windows with non-zero anchored PV, not
# copied replicas of the existing v8 cohort.
DEFAULT_CASES: tuple[dict[str, Any], ...] = (
    {
        "site": "miami_fl",
        "level": "high",
        "seed": 57,
        "start_index": 5217,
        "slug": "miami_fl_high_p5217",
    },
    {
        "site": "las_vegas_nv",
        "level": "high",
        "seed": 61,
        "start_index": 1281,
        "slug": "las_vegas_nv_high_p1281",
    },
    {
        "site": "salt_lake_city_ut",
        "level": "extreme",
        "seed": 58,
        "start_index": 6123,
        "slug": "salt_lake_city_ut_extreme_p6123",
    },
    {
        "site": "portland_or",
        "level": "extreme",
        "seed": 62,
        "start_index": 1543,
        "slug": "portland_or_extreme_p1543",
    },
)

# Restore the source-grounded p4035 task under a clean Basic identity.  The
# historical Medium/fixed-point identity remains terminal; this candidate
# preserves its native task mechanics without carrying difficulty lineage or
# reusing historical admission evidence.
BOSTON_BASIC_RECOVERY_CASE: dict[str, Any] = {
    "site": "boston_ma",
    "level": "basic",
    "seed": 42,
    "start_index": 4035,
    "slug": "boston_ma_basic_p4035",
    "repair_profile": "boston_basic_recovery",
}

# The former Boston row was a label-only relabel of a one-stage task and is
# therefore held by the current replay gates.  Its replacement is a genuine
# High task: replay evidence showed two native response windows, two physical
# tools, three strategy switches, and exact dependency depth two.  Keep it
# separate from the older held row and bind it to a new locked profile window.
BOSTON_HIGH_REPAIR_CASE: dict[str, Any] = {
    "site": "boston_ma",
    "level": "high",
    "seed": 42,
    "start_index": 4040,
    "slug": "boston_ma_high_p4040",
    "repair_profile": "boston_staged_recovery",
}

# Sacramento uses a distinct NREL profile window and the same native two-stage
# recovery contract. It is a separate effective source and is admitted only
# after its own full Protocol-2.1 replay.
SACRAMENTO_HIGH_REPAIR_CASE: dict[str, Any] = {
    "site": "sacramento_ca",
    "level": "high",
    "seed": 42,
    "start_index": 4040,
    "slug": "sacramento_ca_high_p4040",
    "repair_profile": "staged_recovery",
}


def _build_body(case: dict[str, Any]) -> dict[str, Any]:
    level = str(case["level"])
    seed = int(case["seed"])
    slug = str(case["slug"])
    if case.get("repair_profile") == "boston_basic_recovery":
        return _build_boston_basic_recovery_body(case)
    if case.get("repair_profile") in {
        "boston_staged_recovery",
        "staged_recovery",
    }:
        return _build_boston_staged_repair_body(case)
    seed_id = f"microgrid/microgrid_lv_voltage_{'recovery_10h' if level == 'extreme' else 'staged_6h'}/time_pressure/{level}/microgrid_lv_{slug}_s{seed}"
    kwargs = {
        "seed": seed,
        "seed_id": seed_id,
        "difficulty_mode": "time_pressure",
        "site": str(case["site"]),
        "source_profile_start_index": int(case["start_index"]),
    }
    seed_obj = (
        build_microgrid_lv_voltage_recovery_10h_seed(**kwargs)
        if level == "extreme"
        else build_microgrid_lv_voltage_staged_6h_seed(**kwargs)
    )
    body = seed_obj.to_dict()
    body["scenario_id"] = body["seed_id"]
    config = dict(body.get("backend_config") or {})

    # Native replay refinement is explicit and source-preserving.  The NREL
    # profile window remains unchanged; only the derived PV scale, procedural
    # stress magnitude, and response windows are selected from bounded
    # pandapower replays so the task is solvable without masking the native
    # control trade-off.  These are candidate design fields, never source
    # observations.
    replay_refinement: dict[str, Any] = {
        "method": "bounded_native_replay_refinement_v1",
        "source_profile_unchanged": True,
        "source_observation_claim": False,
    }
    site = str(case["site"])
    if level == "high" and site in {"miami_fl", "las_vegas_nv"}:
        config["pv_scale"] = 3.0
        config["task_contract"] = dict(config.get("task_contract") or {})
        config["task_contract"]["phase_ticks"] = [1, 2]
        replay_refinement.update(
            {
                "pv_scale": 3.0,
                "load_spike_intensity": 3.0,
                "phase_ticks": [1, 2],
                "reason": "positive_native_headroom_and_safe_two_stage_recovery",
            }
        )
        for event in config.get("derivation_recipe", {}).get("stress_overlays", []):
            if event.get("kind") == "load_spike":
                event["intensity"] = 3.0
        for event in body.get("perturbations", []):
            if event.get("kind") == "load_spike":
                event["intensity"] = 3.0
    elif level == "extreme" and site == "salt_lake_city_ut":
        config["pv_scale"] = 3.0
        config["task_contract"] = dict(config.get("task_contract") or {})
        config["task_contract"]["phase_ticks"] = [7, 8, 9]
        replay_refinement.update(
            {
                "pv_scale": 3.0,
                "load_spike_intensity": 5.0,
                "phase_ticks": [7, 8, 9],
                "reason": "positive_native_headroom_and_safe_three_stage_recovery",
            }
        )
    config["protocol21_replay_refinement"] = replay_refinement
    config["release_ready"] = False
    config["release_reentry_ready"] = False
    config["source_integration_rung"] = "staging_source_consumed_native_lv_v1"
    config["source_denominator_key"] = json.dumps(
        {
            "backend": "pandapower_lv",
            "network": "pandapower.create_synthetic_voltage_control_lv_network",
            "site": str(case["site"]),
            "source_window_sha256": str(
                (config.get("derivation_recipe") or {}).get("source_window_sha256", "")
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    body["backend_config"] = config
    body["complexity_metrics"] = seed_obj.complexity_metrics()
    body["source_contract"] = _source_contract(body)
    body["scenario_signature"] = recompute_signature_with_seed(body, seed)
    return body


def _build_boston_basic_recovery_body(case: dict[str, Any]) -> dict[str, Any]:
    """Build the p4035 task under its empirically calibrated Basic label."""
    seed = int(case["seed"])
    site = str(case["site"])
    start_index = int(case["start_index"])
    seed_id = (
        "microgrid/microgrid_lv_voltage_6h/time_pressure/basic/"
        f"microgrid_lv_{case['slug']}_s{seed}_v1"
    )
    seed_obj = build_microgrid_lv_voltage_6h_seed(
        seed=seed,
        seed_id=seed_id,
        difficulty_level="basic",
        difficulty_mode="time_pressure",
        site=site,
        source_profile_start_index=start_index,
    )
    config = seed_obj.backend_config
    config.update(
        {
            "pv_scale": 6.5,
            "battery_e_mwh": 0.1,
            "controllable_der_count": 3,
            "battery": {
                "capacity_mwh": 0.1,
                "init_soc": 0.5,
                "max_charge_mw": 0.05,
                "max_discharge_mw": 0.05,
                "efficiency": 0.95,
            },
            "protocol21_replay_refinement": {
                "method": "canonical_basic_identity_recovery_v1",
                "source_profile_unchanged": True,
                "source_observation_claim": False,
                "source_profile_start_index": start_index,
                "difficulty_is_diagnostic": True,
            },
            "release_ready": False,
            "release_reentry_ready": False,
            "source_integration_rung": "staging_source_consumed_native_lv_v1",
        }
    )
    perturbations = [
        Perturbation(
            kind="pv_ramp",
            trigger_tick=1,
            duration_ticks=3,
            hidden=False,
            target={},
            intensity=1.15,
            notes="Rooftop-PV ramp → reverse flow / over-voltage.",
        ),
        Perturbation(
            kind="der_failure",
            trigger_tick=2,
            duration_ticks=2,
            hidden=False,
            target={"der_index": 0},
            intensity=1.0,
            notes="A rooftop-PV string drops out.",
        ),
    ]
    seed_obj.perturbations = perturbations
    recipe = config["derivation_recipe"]
    recipe["stress_overlays"] = [
        {
            "kind": event.kind,
            "trigger_tick": event.trigger_tick,
            "duration_ticks": event.duration_ticks,
            "hidden": event.hidden,
            "target": event.target,
            "intensity": event.intensity,
        }
        for event in perturbations
    ]
    config["source_denominator_key"] = json.dumps(
        {
            "backend": "pandapower_lv",
            "network": "pandapower.create_synthetic_voltage_control_lv_network",
            "site": site,
            "source_window_sha256": str(recipe["source_window_sha256"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    body = seed_obj.to_dict()
    body["scenario_id"] = body["seed_id"]
    body["complexity_metrics"] = seed_obj.complexity_metrics()
    body["source_contract"] = _source_contract(body)
    body["scenario_signature"] = recompute_signature_with_seed(body, seed)
    return body


def _build_boston_staged_repair_body(case: dict[str, Any]) -> dict[str, Any]:
    """Build a genuine two-window High replacement for a locked site profile.

    This is intentionally not a difficulty relabel.  The source profile is
    consumed from a new locked window (4040), while the native LV backend is
    given two deterministic, non-terminal response windows:

    * tick 1: visible PV surge; curtailment materially reduces voltage
      violations;
    * tick 3: hidden DER failure; battery discharge is required after the
      initial charging plan.

    The positive battery command at tick 0 and negative command after the
    hidden event make the reversal evidence explicit.  The candidate is
    declared High because replay proves the High contract (two physical
    tools, two effective ticks, one hidden event, one strategy reversal, and
    dependency depth two); admission still remains fail-closed on all gates.
    """
    seed = int(case["seed"])
    slug = str(case["slug"])
    site = str(case["site"])
    start_index = int(case["start_index"])
    seed_id = (
        "microgrid/microgrid_lv_voltage_staged_6h/time_pressure/high/"
        f"microgrid_lv_{slug}_s{seed}"
    )
    seed_obj = build_microgrid_lv_voltage_6h_seed(
        seed=seed,
        seed_id=seed_id,
        difficulty_level="high",
        difficulty_mode="time_pressure",
        site=site,
        source_profile_start_index=start_index,
    )
    # Preserve native seed provenance and source consumption.  Only the task
    # contract and deterministic procedural stressors are refined here.
    seed_obj.family = "microgrid_lv_voltage_staged_6h"
    body = seed_obj.to_dict()
    config = dict(body.get("backend_config") or {})
    pv_scale = 2.5 if site == "sacramento_ca" else 3.0
    config["pv_scale"] = pv_scale
    config["battery"] = {
        "capacity_mwh": 0.1,
        "init_soc": 0.25,
        "max_charge_mw": 0.05,
        "max_discharge_mw": 0.05,
        "efficiency": 0.95,
    }
    config["battery_e_mwh"] = 0.1
    config["task_contract"] = {
        "contract": "microgrid.lv_voltage.staged_recovery.v2",
        "phase_ticks": [1, 3],
        "minimum_reduction_each_phase": 1,
        "minimum_distinct_control_ticks": 2,
        "reversal": {
            "tool": "set_battery_dispatch",
            "argument": "p_mw",
            "first_sign": "positive",
            "later_sign": "negative",
            "later_not_before_tick": 3,
        },
    }
    body["perturbations"] = [
        {
            "kind": "pv_ramp",
            "trigger_tick": 1,
            "duration_ticks": 2,
            "hidden": False,
            "target": {},
            "intensity": 1.5,
            "notes": (
                "Visible PV surge requires native DER curtailment before "
                "the next supervisory interval."
            ),
        },
        {
            "kind": "der_failure",
            "trigger_tick": 3,
            "duration_ticks": 2,
            "hidden": True,
            "target": {"der_index": 0},
            "intensity": 1.0,
            "notes": (
                "A hidden DER failure invalidates the initial absorptive "
                "plan and requires battery discharge."
            ),
        },
    ]
    seed_obj.perturbations = [
        Perturbation(
            kind="pv_ramp",
            trigger_tick=1,
            duration_ticks=2,
            hidden=False,
            target={},
            intensity=1.5,
            notes=(
                "Visible PV surge requires native DER curtailment before "
                "the next supervisory interval."
            ),
        ),
        Perturbation(
            kind="der_failure",
            trigger_tick=3,
            duration_ticks=2,
            hidden=True,
            target={"der_index": 0},
            intensity=1.0,
            notes=(
                "A hidden DER failure invalidates the initial absorptive "
                "plan and requires battery discharge."
            ),
        ),
    ]
    config["protocol21_replay_refinement"] = {
        "method": "bounded_native_replay_refinement_high_v1",
        "source_profile_unchanged": True,
        "source_observation_claim": False,
        "source_profile_start_index": start_index,
        "pv_scale": pv_scale,
        "phase_ticks": [1, 3],
        "response_windows": {
            "visible_pv_surge": {"trigger_tick": 1, "last_response_tick": 2},
            "hidden_der_failure": {"trigger_tick": 3, "last_response_tick": 4},
        },
        "reason": "native_two_stage_recovery_with_response_window",
    }
    config["release_ready"] = False
    config["release_reentry_ready"] = False
    config["source_integration_rung"] = "staging_source_consumed_native_lv_v1"
    config["source_denominator_key"] = json.dumps(
        {
            "backend": "pandapower_lv",
            "network": "pandapower.create_synthetic_voltage_control_lv_network",
            "site": site,
            "source_window_sha256": str(
                (config.get("derivation_recipe") or {}).get("source_window_sha256", "")
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    seed_obj.backend_config = config
    body["backend_config"] = config
    body["scenario_id"] = body["seed_id"]
    body["complexity_metrics"] = seed_obj.complexity_metrics()
    body["source_contract"] = _source_contract(body)
    body["scenario_signature"] = recompute_signature_with_seed(body, seed)
    return body


def _row(body: dict[str, Any], relative_path: str) -> dict[str, Any]:
    candidate_path = Path(relative_path)
    if candidate_path.is_absolute():
        try:
            relative_path = candidate_path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            relative_path = candidate_path.as_posix()
    config = dict(body.get("backend_config") or {})
    source_key = str(config.get("source_denominator_key") or "")
    source_contract = dict(body.get("source_contract") or {})
    contract = resolve_source_asset_contract(
        {**body, "source_contract": source_contract},
        repo_root=REPO_ROOT,
    )
    physical_source_lock = physical_source_lock_from_contract(
        contract,
        backend_kind=str(body.get("backend_kind") or ""),
    )
    if physical_source_lock is None:
        raise ValueError(
            f"{body['scenario_id']}: verified physical source lock missing"
        )
    semantic = _semantic_fingerprint(body)
    case_ledger = {
        "schema_version": "0.1",
        "source_denominator_key": source_key,
        "physical_source_lock": physical_source_lock,
        "independence_axis": "microgrid_site_profile_window",
        "decision_pressure_axis": (
            "native_voltage_recovery_with_source_profile_stress"
        ),
        "additional_decision_axis": (
            f"difficulty={body.get('difficulty_mode')}/{body.get('difficulty_level')}"
        ),
        "decision_variant_key": semantic,
        "complexity_tags": [
            f"n_perturbations={len(body.get('perturbations') or [])}",
            "procedural_events_source_locked_targets",
            "response_window_required",
            "pending_full_protocol21_gates",
        ],
        "event_repairs": [],
        "source_refinement": {
            "pipeline": str(
                (config.get("protocol21_replay_refinement") or {}).get(
                    "method", "bounded_native_replay_refinement_v1"
                )
            ),
            "candidate_path": relative_path,
            "source_profile_unchanged": True,
        },
    }
    return {
        "scenario_id": body["scenario_id"],
        "path": relative_path,
        "domain": body["domain"],
        "backend_kind": body["backend_kind"],
        "family": body["family"],
        "difficulty_mode": body["difficulty_mode"],
        "difficulty_level": body["difficulty_level"],
        "horizon_ticks": body["horizon_ticks"],
        "seed": body["seed"],
        "scenario_signature": body["scenario_signature"],
        "source_key": source_key,
        "source_denominator_key": source_key,
        "structural_fingerprint": structural_fingerprint(body),
        "semantic_fingerprint": semantic,
        "case_ledger": case_ledger,
        "protocol21_lineage": {
            "physical_identity_origin": "verified_source_asset_graph",
            "ready": True,
            "status": "ready",
            "reason_codes": [],
        },
        "status": "pending_protocol21_full_admission",
        "reason_codes": [
            "staging_source_grounded_candidate",
            "procedural_stressors_not_source_observed",
            "requires_behavior_task_depth_agentic_gates",
        ],
    }


def build(
    *,
    repo_root: Path = REPO_ROOT,
    staging_root: Path = DEFAULT_STAGING,
    cases: Iterable[dict[str, Any]] = DEFAULT_CASES,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    del repo_root  # builders resolve repository assets from their module path
    files: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for case in cases:
        body = _build_body(dict(case))
        relative = staging_root / f"{str(body['scenario_id']).replace('/', '__')}.yaml"
        files[relative] = body
        rows.append(_row(body, str(relative)))
    source_keys = [str(row["source_denominator_key"]) for row in rows]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("microgrid candidate source keys must be independent")
    return (
        {
            "schema_version": "protocol21-microgrid-native-candidates-v1",
            "status": "staging_candidates_pending_full_admission",
            "candidate_only": True,
            "release_admission": False,
            "leaderboard_eligible": False,
            "release_ready": False,
            "n_candidates": len(rows),
            "difficulty_counts": dict(
                sorted(Counter(str(row["difficulty_level"]) for row in rows).items())
            ),
            "source_key_count": len(source_keys),
            "scenarios": rows,
        },
        files,
    )


def build_source_suite(
    *,
    staging_root: Path,
    cases: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Wrap candidates in a reproducible Protocol-2.1 working-set envelope."""
    report, _files = build(staging_root=staging_root, cases=cases)
    return {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "working_set",
        "selection_policy": "quality_maximal_v1",
        "candidate_only": True,
        "release_admission": False,
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": int(report["n_candidates"]),
        "constraints": {
            "core_admission_profile": "quality_core_v2",
            "candidate_evidence_merge_only": True,
            "candidate_replacements_staging_only": True,
            "formal_evaluation_ready": False,
            "model_outcomes_used_for_filtering": False,
            "one_per_effective_source_identity": True,
            "preserve_each_eligible_family_difficulty_cell": True,
            "quality_maximal_selection": True,
        },
        "scenarios": report["scenarios"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    repair_group = parser.add_mutually_exclusive_group()
    repair_group.add_argument(
        "--boston-basic-recovery",
        action="store_true",
        help="materialize the p4035 task under a clean Basic candidate identity",
    )
    repair_group.add_argument(
        "--boston-high-repair",
        action="store_true",
        help="materialize only the held Boston replacement as a High candidate",
    )
    repair_group.add_argument(
        "--sacramento-high-repair",
        action="store_true",
        help="materialize an independent Sacramento High candidate",
    )
    parser.add_argument("--source-suite-output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    selected_cases = (
        [SACRAMENTO_HIGH_REPAIR_CASE]
        if args.sacramento_high_repair
        else [BOSTON_HIGH_REPAIR_CASE]
        if args.boston_high_repair
        else [BOSTON_BASIC_RECOVERY_CASE]
        if args.boston_basic_recovery
        else DEFAULT_CASES
    )
    report, files = build(
        staging_root=args.staging_root.resolve(),
        cases=selected_cases,
    )
    if args.execute:
        for path, body in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if args.source_suite_output is not None:
            suite = build_source_suite(
                staging_root=args.staging_root.resolve(),
                cases=selected_cases,
            )
            args.source_suite_output.parent.mkdir(parents=True, exist_ok=True)
            args.source_suite_output.write_text(
                json.dumps(suite, indent=2) + "\n", encoding="utf-8"
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
