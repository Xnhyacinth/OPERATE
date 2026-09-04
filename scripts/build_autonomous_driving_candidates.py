#!/usr/bin/env python3
"""Materialize the honest, held NGSIM driving pilot scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = Path("works/autonomous_driving/ngsim/smoke/bundle")
DEFAULT_OUTPUT = Path("scenarios/staging/autonomous_driving_pilot")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def build(bundle: Path) -> tuple[Path, str, dict[str, Any]]:
    manifest = _load_json(bundle / "bundle.json")
    seed_set = _load_json(bundle / "seeds/seeds.json")
    runtime_fixture = _load_json(bundle / "runtime/fixture.json")
    seeds = seed_set.get("seeds") or []
    if (
        manifest.get("admission_status") != ("held_pending_live_sumo_reactive_validation")
        or not seeds
    ):
        raise ValueError("autonomous_driving_bundle_not_a_held_pilot")
    derivation = dict(runtime_fixture.get("derivation") or {})
    materialized_candidate_id = str(derivation.get("candidate_id") or "")
    seed = next(
        (dict(value) for value in seeds if value.get("candidate_id") == materialized_candidate_id),
        None,
    )
    if seed is None:
        raise ValueError("autonomous_driving_runtime_candidate_missing_from_seed_set")
    if seed.get("source_window_sha256") != derivation.get("source_window_sha256"):
        raise ValueError("autonomous_driving_runtime_seed_window_mismatch")
    ego = dict(runtime_fixture.get("ego") or {})
    ego_actor_id = str(ego.get("vehicle_id") or "")
    if not ego_actor_id or ego_actor_id != str(derivation.get("ego_actor_id") or ""):
        raise ValueError("autonomous_driving_runtime_fixture_missing_ego_identity")
    scenario_id = (
        "autonomous_driving/sustained_highway_risk_supervision/"
        "time_pressure/basic/ngsim_us101_smoke_s42"
    )
    try:
        relative_bundle = bundle.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        relative_bundle = bundle.as_posix()
    scenario = {
        "seed_id": scenario_id,
        "scenario_id": scenario_id,
        "family": "sustained_highway_risk_supervision",
        "domain": "autonomous_driving",
        "backend_kind": "sumo_ego",
        "backend_config": {
            "source_bundle": relative_bundle,
            "candidate_id": seed["candidate_id"],
            "ego_actor_id": ego_actor_id,
            "execution_mode": "emulated_source_initialized",
            "task_contract": {
                "contract": "autonomous_driving.risk_progress_mitigation.v1",
                "standing_plan_required": True,
            },
        },
        "horizon_ticks": 4,
        "tick_seconds": 5.0,
        "clock_contract": {
            "schema_version": "driving_clock_v1",
            "physics_step_seconds": 0.1,
            "shield_step_seconds": 0.1,
            "substeps_per_supervisory_tick": 50,
            "provider_wall_clock_advances_simulation": False,
        },
        "seed": 42,
        "difficulty_level": "basic",
        "difficulty_mode": "time_pressure",
        "formal_core_allowed": False,
        "release_admission": "held_diagnostic_pilot",
        "held_reasons": [
            "live_sumo_reactive_rollout_not_validated",
            "preventive_and_recovery_windows_not_yet_source_grounded",
            "shield_only_material_headroom_not_yet_demonstrated",
        ],
        "source_window_sha256": seed["source_window_sha256"],
        "provenance": {
            "dataset_id": manifest["source_dataset_id"],
            "doi": "10.21949/1504477",
            "license_id": manifest["license_id"],
            "source_evidence_sha256": manifest["evidence"]["source_evidence_sha256"],
            "bundle_id": manifest["bundle_id"],
        },
        "dimension_applicability": {
            "system_survival": {"applicable": True},
            "safety_violation": {"applicable": True},
            "economic_cost": {"applicable": True},
            "adaptive_replanning": {
                "applicable": False,
                "reason": "no_source_grounded_material_hazard_or_recovery_window",
            },
            "information_efficiency": {"applicable": True},
            "foresight_score": {
                "applicable": False,
                "reason": "no_source_grounded_future_hazard_in_smoke_sibling",
            },
            "counterfactual_prevention": {
                "applicable": False,
                "reason": "shield_only_material_headroom_not_demonstrated",
            },
            "tool_use_efficiency": {"applicable": True},
            "optimality_gap": {
                "applicable": False,
                "reason": "no_validated_native_trajectory_optimum",
            },
            "weighted_equity_score": {
                "applicable": False,
                "reason": "no_source_grounded_road_user_criticality_classes",
            },
            "ethical_quality": {
                "applicable": False,
                "reason": "no_source_grounded_ethical_dilemma",
            },
            "stakeholder_management": {
                "applicable": False,
                "reason": "no_source_grounded_stakeholder_model",
            },
            "stakeholder_equity": {
                "applicable": False,
                "reason": "no_source_grounded_stakeholder_classes",
            },
        },
    }
    relative = Path(
        "autonomous_driving/sustained_highway_risk_supervision/"
        "time_pressure/basic/ngsim_us101_smoke_s42.yaml"
    )
    payload = yaml.safe_dump(scenario, sort_keys=False, allow_unicode=True)
    report = {
        "schema_version": "autonomous_driving_candidate_report_v1",
        "status": "held",
        "materialized_scenarios": [relative.as_posix()],
        "missing_families": {
            "cut_in_prevention_and_emergency": ("smoke source window has no mined lane change"),
            "odd_degradation_mrm_recovery": (
                "NGSIM contains no source-grounded ODD or component fault channel"
            ),
        },
        "bundle_id": manifest["bundle_id"],
    }
    return relative, payload, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    bundle = args.bundle if args.bundle.is_absolute() else REPO_ROOT / args.bundle
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    relative, payload, report = build(bundle.resolve())
    expected = {
        output / relative: payload,
        output / "candidate_report.json": json.dumps(report, sort_keys=True, indent=2) + "\n",
    }
    if args.check:
        stale = [
            str(path)
            for path, text in expected.items()
            if not path.is_file() or path.read_text() != text
        ]
        if stale:
            print(json.dumps({"status": "stale", "paths": stale}, indent=2))
            return 1
        print(json.dumps({"status": "verified", "paths": len(expected)}))
        return 0
    for path, text in expected.items():
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "written", "paths": len(expected)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
