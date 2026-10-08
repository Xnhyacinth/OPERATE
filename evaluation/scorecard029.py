"""Source-locked outcome and capability scorecards for authenticated trajectories.

The public command reconstructs the 0.28 base from raw evidence first. This
module never promotes cached numeric rows to raw authentication. Capability
contracts come only from the policy-authenticated scenario bytes, not from a
model, report, or caller-provided self-hashed measurement overlay.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

import yaml

from evaluation.capability_measurement029 import AXES, score_capabilities
from evaluation.outcome_analysis029 import build_outcome_analysis
from evaluation.operational_outcome028 import aggregate_outcome028
from evaluation.operational_weights import validate_task_weights
from evaluation.retrospective_ranking029 import build_retrospective_ranking
from evaluation.trajectory_outcome027 import canonical_digest, read_bound

VERSION = "0.29.1"
CONTRACT_FIELD = "operational_capability_contract029"

RETROSPECTIVE_PROTOCOL = {
    "protocol_id": "lite_fixed_panel_retrospective_outcome029.v1",
    "target": "recorded_operational_performance_on_the_fixed_panel",
    "primary_index": "outcome028_Q",
    "aggregation": "frozen_domain_family_physical_source_case_weights",
    "point_ranking": "descending_unrounded_Q_competition_ties_complete_cohort_only",
    "missingness": "retain_fixed_weight_bounds_no_imputation_no_partial_point_rank",
    "measurement_ordering": "strict_separation_of_recorded_precision_bounds",
    "sensitivity": "delete_whole_source_or_domain_retaining_original_relative_weights",
    "capability_contracts_required": False,
    "new_model_runs_required": False,
    "required_companion_results": [
        "N",
        "F",
        "coverage",
        "native_costs_and_components",
        "safety_coverage",
        "domain_and_family",
        "component_pareto",
        "measurement_bounds",
        "rank_cohort",
        "source_and_domain_deletion",
        "task_semantics",
    ],
    "claim_scope": "retrospective_model_informed_panel_not_held_out_or_isolated_six_axis_ability",
}

# Interpretations are measurement boundaries, not new scoring coefficients.
TASK_SEMANTICS = {
    "alibaba_trace_sim": (
        "completed_jobs_and_queue_lateness",
        "queue_due_tick_is_not_source_job_completion_deadline",
    ),
    "pyvrp_cvrp": (
        "source_demand_dispatch_wave_delivery",
        "native_travel_duration_not_executed",
    ),
    "pyvrp_vrptw": (
        "source_demand_dispatch_wave_delivery",
        "original_travel_service_times_and_time_windows_not_executed",
    ),
    "dynasched_flexible_job_shop": (
        "source_operation_progress_and_native_schedule_cost",
        "operation_progress_is_not_complete_job_delivery",
    ),
    "sumo_ego": (
        "conditional_terminal_supervisory_recovery",
        "not_route_arrival_or_continuous_physics_stability",
    ),
    "pandapower_lv": (
        "voltage_compliance_and_delivered_energy_bottleneck",
        "not_joint_load_node_time_compliant_delivery",
    ),
    "cigre_distribution": (
        "voltage_compliance_and_delivered_energy_bottleneck",
        "not_joint_load_node_time_compliant_delivery",
    ),
    "opendss_ieee13": (
        "population_normalized_voltage_quality",
        "no_independent_delivered_energy_meter",
    ),
    "opendss_fresh_feeders": (
        "population_normalized_voltage_quality",
        "no_independent_delivered_energy_meter",
    ),
    "citylearn": ("economic_quality", "no_independent_controllable_service_F"),
    "orgym_invmgmt": ("fixed_source_demand_service", "not_a_universal_success_rate"),
    "pymgrid_economic_dispatch": (
        "fixed_demand_service_and_native_state_loss",
        "not_comprehensive_safety_or_causal_agency",
    ),
    "pglib_uc_synthetic": (
        "fixed_demand_service_and_native_cost",
        "native_catastrophe_unmodeled",
    ),
}


def portable_report(value):
    """Normalize authenticated storage locations, retaining selection paths."""
    if isinstance(value, dict):
        result = {k: portable_report(v) for k, v in value.items()}
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            result["path"] = "sha256:" + value["sha256"]
        return result
    if isinstance(value, list):
        return [portable_report(v) for v in value]
    return value


def _sources(report: dict, root: Path) -> tuple[dict, list]:
    sources, refs = {}, []
    for contract in report["source_contracts"]["contracts"]:
        key = contract["scenario_signature"], contract["seed"]
        raw, ref = read_bound(
            {
                "path": str(root / contract["scenario_path"]),
                "sha256": contract["scenario_sha256"],
            }
        )
        scenario = yaml.safe_load(raw)
        if any(
            scenario.get(field) != contract[field]
            for field in ("scenario_signature", "seed", "backend_kind", "horizon_ticks")
        ):
            raise ValueError("scorecard_source_identity_mismatch")
        capability = scenario.get(CONTRACT_FIELD)
        if capability is not None and (
            not isinstance(capability, dict)
            or any(
                capability.get(field) != contract[field]
                for field in ("scenario_signature", "seed", "horizon_ticks")
            )
        ):
            raise ValueError("scorecard_capability_source_identity_mismatch")
        sources[key] = capability
        refs.append(ref)
    return sources, refs


def _capability_models(cases: list, weights: dict, models: list[str]) -> list[dict]:
    result = []
    for model in models:
        selected = {
            (r["scenario_signature"], r["seed"]): r
            for r in cases
            if r["model"] == model
        }
        axes = {}
        for axis in AXES:
            readings = {key: r["axes"][axis] for key, r in selected.items()}
            # An absent source contract is unknown applicability, not permission
            # to renormalize a partial model into a full capability result.
            applicable = {
                k: v for k, v in readings.items() if v.get("applicable") is not False
            }
            mass = math.fsum(weights[k] for k in applicable)
            measured = {
                k: v for k, v in applicable.items() if v.get("score") is not None
            }
            complete = bool(applicable) and len(measured) == len(applicable)
            axes[axis] = {
                "score": math.fsum(weights[k] * v["score"] for k, v in measured.items())
                / mass
                if complete
                else None,
                "expected_cases": len(applicable),
                "measured_cases": len(measured),
                "structural_nonapplicable_cases": len(readings) - len(applicable),
                "expected_weight_mass": mass,
                "measured_weight_mass": math.fsum(weights[k] for k in measured),
                "complete": complete,
                "reason_counts": dict(
                    Counter(v.get("reason") for v in readings.values())
                ),
                "evidence_ids_by_case": [
                    {
                        "scenario_signature": k[0],
                        "seed": k[1],
                        "evidence_ids": v["evidence_ids"],
                    }
                    for k, v in measured.items()
                ],
                "capability_ranking_ready": False,
            }
        result.append({"model": model, "axes": axes})
    return result


def build_scorecard(report: dict, *, root: Path) -> dict:
    """Analyze a caller-authenticated 0.28 report; never authenticate cached scores.

    The only public ingestion path calls the raw trajectory reader. Library callers
    own this trust boundary, as with existing native/agency analysis modules.
    """
    analysis = build_outcome_analysis(report)
    ranking = build_retrospective_ranking(analysis)
    protocol = {
        **deepcopy(RETROSPECTIVE_PROTOCOL),
        "suite_sha256": analysis["suite_sha256"],
        "weight_manifest_sha256": analysis["weight_manifest_sha256"],
        "source_contracts_sha256": canonical_digest(report["source_contracts"]),
        "base_policy_sha256": (report.get("policy") or {}).get("sha256"),
        "outcome_index_version": "0.28.0",
    }
    protocol["protocol_sha256"] = canonical_digest(protocol)
    weights = validate_task_weights(
        report["task_weight_manifest"], report["source_contracts"]
    )
    contracts = {
        (c["scenario_signature"], c["seed"]): c
        for c in report["source_contracts"]["contracts"]
    }
    sources, source_refs = _sources(report, Path(root))
    capability_cases = []
    indexed = {
        (r["model"], r["scenario_signature"], r["seed"]): r
        for r in report["graded_episodes"]
    }
    population = [
        (model["model"], key) for model in report["models"] for key in contracts
    ]
    for model, key in population:
        row = indexed.get((model, *key), {})
        source = sources[key]
        binding = row.get("artifact_binding") or {}
        verified = (
            row.get("native_measurement_qualified") is True
            and binding.get("verified") is True
        )
        ledger = None
        if source is not None and verified:
            raw, _ = read_bound(binding["artifacts"]["evidence_ledger_artifact"])
            ledger = [json.loads(line) for line in raw.splitlines() if line.strip()]
        identity = {"scenario_signature": key[0], "seed": key[1]}
        capability = score_capabilities(
            source,
            evidence_ledger=ledger,
            identity=identity,
            expected_contract_sha256=canonical_digest(source)
            if source is not None
            else None,
            evidence_verified=verified,
        )
        capability_cases.append({"model": model, **identity, **capability})
    # Inputs must still match their locked bytes after all dependent reads.
    for ref in source_refs:
        read_bound(ref)
    model_names = [m["model"] for m in report["models"]]
    # Do not let an old cached model summary override freshly checked rows.
    outcome_models = aggregate_outcome028(
        report["graded_episodes"], contracts, weights, models=model_names
    )["models"]
    for model in outcome_models:
        selected = [
            r for r in report["graded_episodes"] if r["model"] == model["model"]
        ]
        model["n_observed"] = sum(
            bool((r.get("origin") or {}).get("canonical_episode_sha256"))
            for r in selected
        )
        model["selected_actual_model_ids"] = sorted(
            {
                r["origin"]["actual_model"]
                for r in selected
                if (r.get("origin") or {}).get("actual_model")
            }
        )
    semantics = []
    for key, contract in contracts.items():
        construct, limitation = TASK_SEMANTICS.get(
            contract["backend_kind"],
            ("source_contract_outcome", "backend_semantics_not_reviewed_for029"),
        )
        semantics.append(
            {
                "scenario_signature": key[0],
                "seed": key[1],
                "backend_kind": contract["backend_kind"],
                "construct": construct,
                "limitation": limitation,
                "source_contract_sha256": contract["contract_sha256"],
                "capability_contract_present": sources[key] is not None,
                "capability_contract_sha256": canonical_digest(sources[key])
                if sources[key] is not None
                else None,
            }
        )
    return {
        "schema_version": "operate_operational_scorecard029.v1",
        "evaluation_version": VERSION,
        "outcome_index_version": "0.28.0",
        "base_report_sha256": canonical_digest(
            portable_report(
                {k: v for k, v in report.items() if k != "current_implementation"}
            )
        ),
        "base_report_digest_policy": "storage_independent_excluding_scoring_git_metadata_runtime_bound_separately",
        "input_manifest_sha256": report.get("input_manifest_sha256"),
        "current_implementation": deepcopy(report.get("current_implementation")),
        "comparison_policy": report.get("comparison_policy"),
        "formal_run_certified": False,
        "same_run_141_merge_certified": False,
        "capability_ranking_ready": False,
        "capability_calibration_status": "source_construct_validation_required",
        "outcome_models": outcome_models,
        "retrospective_protocol": protocol,
        "retrospective_ranking": ranking,
        "outcome_analysis": analysis,
        "capability_models": _capability_models(capability_cases, weights, model_names),
        "capability_cases": capability_cases,
        "task_semantics": semantics,
        "source_bindings": source_refs,
        "uncertainty": {
            "measurement_bounds": "recorded_precision_and_missing_measurement",
            "rank_ranges": "deterministic_source_or_domain_deletion_sensitivity",
            "repeat_run_confidence_intervals": None,
            "reason": "one_selected_episode_per_case_does_not_estimate_run_variance",
        },
        "execution": deepcopy(report.get("execution")),
    }
