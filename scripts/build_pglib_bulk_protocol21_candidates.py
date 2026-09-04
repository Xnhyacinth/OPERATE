#!/usr/bin/env python3
"""Materialize source-locked PGLib-UC candidates and terminal PGLib-OPF records.

PGLib-UC instances contain native multi-period demand, reserve, and renewable
series, so one long-horizon candidate is materialized per source file.  PGLib-
OPF cases are static network snapshots; they are source-locked and assigned a
terminal disposition, but are not padded with an unrelated or synthetic time
series.  This script is staging-only and never admits rows to Core.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.suite_identity import recompute_signature_with_seed  # noqa: E402
from domains.power_grid.seeds.from_pglib_uc import (  # noqa: E402
    build_critical_winter_peak_seed,
    build_reserve_stress_seed,
    build_wind_uncertainty_seed,
    ordered_uc_task_requirements,
)
from scripts.build_protocol21_candidate_source_suite import build_suite  # noqa: E402

DEFAULT_UC_ROOT = ROOT / "works/pglib-uc"
DEFAULT_OPF_ROOT = ROOT / "works/PGLib-OPF"
DEFAULT_BASE_CORE = (
    ROOT
    / "reports/protocol21_pending_union_fresh_current_20260812_wave2_realtraffic_stable"
    / "refined_core_selection_protocol2_v21.json"
)

_PGLIB_UC_DIMENSION_APPLICABILITY = {
    "system_survival": {
        "applicable": True,
        "reason": "aggregate_balance_and_reserve_survival_records_available",
    },
    "economic_cost": {
        "applicable": True,
        "reason": "production_startup_shed_and_reserve_cost_components_available",
    },
    "safety_violation": {
        "applicable": True,
        "reason": "balance_excursion_and_reserve_shortfall_records_available",
    },
    "weighted_equity_score": {
        "applicable": False,
        "reason": "pglib_uc_has_no_source_grounded_customer_criticality_ledger",
    },
    "ethical_quality": {
        "applicable": False,
        "reason": "source_native_uc_schedule_has_no_ethical_dilemma_payload",
    },
    "stakeholder_management": {
        "applicable": False,
        "reason": "source_native_uc_schedule_has_no_stakeholder_trust_event",
    },
    "adaptive_replanning": {
        "applicable": False,
        "reason": "source_native_schedule_has_no_exogenous_disruption_or_recovery_window",
    },
    "information_efficiency": {
        "applicable": True,
        "reason": "partial_grid_observation_and_source_forecast_tools_available",
    },
    "foresight_score": {
        "applicable": False,
        "reason": "source_interval_telemetry_is_not_a_registered_forecastable_hazard",
    },
    "optimality_gap": {
        "applicable": True,
        "reason": "locked_case_lp_dispatch_reference_available",
    },
    "counterfactual_prevention": {
        "applicable": True,
        "reason": "deterministic_no_action_replay_over_same_uc_schedule",
    },
    "stakeholder_equity": {
        "applicable": False,
        "reason": "pglib_uc_has_no_source_grounded_cross_stakeholder_outcome_ledger",
    },
    "tool_use_efficiency": {
        "applicable": True,
        "reason": "tool_protocol_call_outcome_and_budget_evidence_available",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _locked_source_text(base_core: Path) -> str:
    if not base_core.is_file():
        return ""
    payload = json.loads(base_core.read_text(encoding="utf-8"))
    return json.dumps(payload.get("scenarios", []), sort_keys=True)


def _uc_seed(case_path: Path, *, seed: int) -> Any:
    subset = case_path.parent.name
    seed_id = f"pglib_uc_{subset}_{_slug(case_path.stem)}_long_extreme_s{seed}"
    kwargs = {
        "seed_id": seed_id,
        "seed": seed,
        "difficulty_mode": "deep_planning",
        "difficulty_level": "extreme",
    }
    if subset == "ca":
        return build_reserve_stress_seed(case_path, **kwargs)
    if subset == "ferc":
        return build_wind_uncertainty_seed(case_path, **kwargs)
    if subset == "rts_gmlc":
        return build_critical_winter_peak_seed(case_path, **kwargs)
    raise ValueError(f"unsupported PGLib-UC subset: {subset}")


def _uc_physical_task_design(
    case: dict[str, Any], *, horizon_ticks: int
) -> dict[str, Any]:
    """Describe only control axes that exist in the locked UC instance."""
    generators = [
        spec
        for spec in (case.get("thermal_generators") or {}).values()
        if isinstance(spec, dict)
    ]
    horizon = max(1, int(horizon_ticks))
    demand = [float(value) for value in (case.get("demand") or [])][:horizon]
    reserves = [float(value) for value in (case.get("reserves") or [])][:horizon]
    ramp_dispatch = any(
        float(spec.get("power_output_maximum", 0.0) or 0.0)
        > float(spec.get("power_output_minimum", 0.0) or 0.0)
        and (
            float(spec.get("ramp_up_limit", 0.0) or 0.0) > 0.0
            or float(spec.get("ramp_down_limit", 0.0) or 0.0) > 0.0
        )
        for spec in generators
    )
    startup_candidates = sum(
        not bool(spec.get("unit_on_t0", False) or spec.get("must_run", False))
        and int(spec.get("time_down_t0", 0) or 0)
        >= int(spec.get("time_down_minimum", 1) or 1)
        and float(spec.get("power_output_maximum", 0.0) or 0.0) > 0.0
        for spec in generators
    )
    shutdown_candidates = sum(
        bool(spec.get("unit_on_t0", False))
        and not bool(spec.get("must_run", False))
        and int(spec.get("time_up_t0", 0) or 0)
        >= int(spec.get("time_up_minimum", 1) or 1)
        for spec in generators
    )
    axes: list[str] = []
    if startup_candidates or shutdown_candidates:
        axes.append("commitment_startup_shutdown")
    if ramp_dispatch:
        axes.append("generation_ramp_dispatch")
    if max(reserves, default=0.0) > 0.0:
        axes.append("reserve_procurement")
    demand_ramps = [abs(current - previous) for previous, current in zip(demand, demand[1:])]
    maximum_ramp = max(demand_ramps, default=0.0)
    maximum_ramp_tick = demand_ramps.index(maximum_ramp) + 1 if demand_ramps else None
    return {
        "schema_version": "pglib_uc_physical_task_v1",
        "task_axis_valid": ramp_dispatch or bool(startup_candidates or shutdown_candidates),
        "source_native_axes": axes,
        "decision_pressure": {
            "time_periods": len(demand),
            "peak_demand_mw": round(max(demand, default=0.0), 6),
            "peak_reserve_requirement_mw": round(max(reserves, default=0.0), 6),
            "maximum_absolute_demand_ramp_mw": round(maximum_ramp, 6),
            "maximum_absolute_demand_ramp_tick": maximum_ramp_tick,
            "startup_candidate_units": startup_candidates,
            "shutdown_candidate_units": shutdown_candidates,
        },
        "allowed_native_controls": [
            "dispatch_generation_portfolio",
            *(["commit_reserve"] if "reserve_procurement" in axes else []),
        ],
        "procedural_stress_used": False,
    }


def build(
    *,
    uc_root: Path,
    opf_root: Path,
    base_core: Path,
    staging_root: Path,
    report_path: Path,
    suite_path: Path,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any], dict[Path, dict[str, Any]]]:
    """Build reports and YAML bodies without writing them."""
    locked_text = _locked_source_text(base_core)
    source_units: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    files: dict[Path, dict[str, Any]] = {}

    for case_path in sorted(uc_root.glob("*/*.json")):
        relative_source = _relative(case_path)
        source_hash = _sha256(case_path)
        if relative_source in locked_text or case_path.name in locked_text:
            source_units.append(
                {
                    "source_family": "pglib_uc",
                    "source_unit": relative_source,
                    "source_sha256": source_hash,
                    "work_state": "terminal",
                    "disposition": "secondary_duplicate",
                    "reason_codes": ["physical_source_already_in_locked_core"],
                }
            )
            continue
        case = json.loads(case_path.read_text(encoding="utf-8"))
        seed_obj = _uc_seed(case_path, seed=seed)
        body = seed_obj.to_dict()
        task_design = _uc_physical_task_design(
            case,
            horizon_ticks=int(body["horizon_ticks"]),
        )
        if not task_design["task_axis_valid"]:
            source_units.append(
                {
                    "source_family": "pglib_uc",
                    "source_unit": relative_source,
                    "source_sha256": source_hash,
                    "work_state": "terminal",
                    "disposition": "abandoned_intrinsic",
                    "reason_codes": ["source_native_control_axis_absent"],
                    "physical_task_design": task_design,
                }
            )
            continue
        # PGLib-UC already supplies the multi-period demand, reserve and
        # renewable envelopes that define the scheduling pressure.  Candidate
        # admission must measure that source-native problem before considering
        # procedural stress variants; injected outages/surges and synthetic
        # dilemmas previously made otherwise useful cases intrinsically unsafe.
        body["perturbations"] = []
        body["dilemmas"] = []
        body["provenance"]["notes"] = (
            "Source-native multi-period UC schedule; no procedural perturbations."
        )
        scenario_stem = str(seed_obj.seed_id)
        scenario_id = f"power_grid/{body['family']}/deep_planning/extreme/{scenario_stem}"
        body["scenario_id"] = scenario_id
        config = body.setdefault("backend_config", {})
        source_key = f"pglib_uc:{case_path.parent.name}:{case_path.name}"
        config["source_denominator_key"] = source_key
        config["dimension_applicability"] = json.loads(
            json.dumps(_PGLIB_UC_DIMENSION_APPLICABILITY)
        )
        config["task_requirements"] = ordered_uc_task_requirements(
            difficulty_level="extreme",
            horizon_ticks=int(body["horizon_ticks"]),
            case=case,
        )
        config["physical_task_design"] = task_design
        config["candidate_refine_contract"] = {
            "schema_version": "pglib_uc_long_horizon_v2",
            "candidate_only": True,
            "source_native_operations_only": True,
            "source_series_drives_backend": [
                "demand",
                "reserves",
                "renewable_generators",
            ],
            "declared_perturbations_do_not_create_source_independence": True,
            "required_replay_gates": [
                "deterministic_replay",
                "native_action_effect",
                "no_action_counterfactual",
                "positive_headroom",
                "reference_task_completion",
                "observed_dependency_depth",
            ],
        }
        body["source_contract"] = {
            "runtime_input": [relative_source],
            "derivation_input": [],
            "implementation_asset": [],
            "metadata": [],
            "license": ["works/pglib-uc/LICENSE"],
            "file_sha256s": {relative_source: source_hash},
        }
        body["scenario_signature"] = recompute_signature_with_seed(body, seed)
        output = (
            staging_root
            / "power_grid"
            / str(body["family"])
            / "deep_planning"
            / "extreme"
            / f"{scenario_stem}.yaml"
        )
        files[output] = body
        candidate_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_signature": body["scenario_signature"],
                "path": _relative(output),
                "source_unit": relative_source,
                "source_denominator_key": source_key,
                "status": "pending_native_prefilter",
            }
        )
        source_units.append(
            {
                "source_family": "pglib_uc",
                "source_unit": relative_source,
                "source_sha256": source_hash,
                "work_state": "pending",
                "disposition": None,
                "reason_codes": ["materialized_source_driven_long_horizon_candidate"],
            }
        )

    for case_path in sorted(opf_root.glob("pglib_opf_case*.m")):
        relative_source = _relative(case_path)
        case_name = case_path.stem
        duplicate = case_name in locked_text
        source_units.append(
            {
                "source_family": "pglib_opf",
                "source_unit": relative_source,
                "source_sha256": _sha256(case_path),
                "work_state": "terminal",
                "disposition": "secondary_duplicate" if duplicate else "held_repair",
                "reason_codes": [
                    "physical_source_already_in_locked_core"
                    if duplicate
                    else "static_snapshot_missing_paired_source_timeseries",
                    "synthetic_or_unrelated_timeseries_forbidden",
                ],
                "conversion_recipe": {
                    "backend_kind": "pandapower_acopf",
                    "native_loader": "pandapower.converter.from_mpc",
                    "native_controls": [
                        "redispatch_generation",
                        "commit_reserve",
                        "shed_load",
                    ],
                    "temporal_materialization_allowed": False,
                    "required_next_input": "source-locked compatible load/generation profile",
                },
            }
        )

    report = {
        "schema_version": "pglib-bulk-candidate-report-v1",
        "status": "candidate_materialization_complete_prefilter_pending",
        "candidate_only": True,
        "release_admission": False,
        "base_core": {"path": _relative(base_core), "sha256": _sha256(base_core)},
        "scenarios": candidate_rows,
        "source_units": source_units,
        "summary": {
            "n_source_units": len(source_units),
            "n_uc_materialized": len(candidate_rows),
            "n_opf_terminal": sum(row["source_family"] == "pglib_opf" for row in source_units),
            "terminal_dispositions": dict(
                sorted(
                    Counter(
                        row["disposition"]
                        for row in source_units
                        if row["work_state"] == "terminal"
                    ).items()
                )
            ),
            "all_source_units_have_one_state": len(source_units)
            == len({(row["source_family"], row["source_unit"]) for row in source_units}),
        },
    }
    # build_suite reads the report and YAMLs after execution; return a marker
    # here so the pure build path remains free of filesystem writes.
    suite = {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "pending_report_write",
        "n_scenarios": len(candidate_rows),
        "output_path": _relative(suite_path),
        "report_path": _relative(report_path),
    }
    return report, suite, files


def execute(
    *,
    report: dict[str, Any],
    files: dict[Path, dict[str, Any]],
    report_path: Path,
    suite_path: Path,
) -> dict[str, Any]:
    for path, body in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    suite = build_suite(report_path)
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uc-root", type=Path, default=DEFAULT_UC_ROOT)
    parser.add_argument("--opf-root", type=Path, default=DEFAULT_OPF_ROOT)
    parser.add_argument("--base-core", type=Path, default=DEFAULT_BASE_CORE)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report, suite, files = build(
        uc_root=args.uc_root.resolve(),
        opf_root=args.opf_root.resolve(),
        base_core=args.base_core.resolve(),
        staging_root=args.staging_root.resolve(),
        report_path=args.report.resolve(),
        suite_path=args.suite.resolve(),
        seed=args.seed,
    )
    if args.execute:
        suite = execute(
            report=report,
            files=files,
            report_path=args.report.resolve(),
            suite_path=args.suite.resolve(),
        )
    print(json.dumps({**report["summary"], "n_suite_rows": suite["n_scenarios"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
