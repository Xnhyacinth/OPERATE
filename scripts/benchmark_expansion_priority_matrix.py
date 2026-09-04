#!/usr/bin/env python3
"""Build a non-release priority matrix for benchmark expansion tracks.

The matrix is a decision artifact, not a materializer. It reads the current
release manifest and existing preflight/readiness reports, then ranks the next
expansion tracks by release value, source-lock readiness, behavioral evidence,
and implementation risk. It never installs dependencies, downloads data, writes
scenario YAML, or changes release suite membership.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = REPO_ROOT / "release" / "dt_sched_bench_v0_50_0"
DEFAULT_REPORTS_DIR = REPO_ROOT / "reports"
DEFAULT_OUTPUT = DEFAULT_REPORTS_DIR / "benchmark_expansion_priority_matrix.json"
DEFAULT_MARKDOWN_OUTPUT = REPO_ROOT / "docs" / "VNEXT_EXPANSION_PRIORITY.md"

NONE_DELTA = {
    "registry": None,
    "primary": None,
    "core": None,
    "effective_sources": None,
    "physical_sources": None,
    "diagnostic_cells": None,
}
POLICY = {
    "writes_release_artifacts": False,
    "writes_scenario_yaml": False,
    "installs_dependencies": False,
    "downloads_data": False,
    "changes_scoring_or_tool_protocol": False,
    "changes_suite_membership": False,
}


def build_method_transfer_catalog() -> list[dict[str, Any]]:
    """Return external benchmark methods that are safe to transfer as recipes.

    The catalog is intentionally explicit that none of these sources directly
    contributes a Core scenario.  Their assets and licenses must be audited by
    a native adapter before any future promotion.
    """
    rows = [
        {
            "source_id": "dynaschedbench",
            "title": "DynaSchedBench",
            "url": "https://arxiv.org/abs/2605.27566",
            "license_status": "upstream_code_and_instance_terms_require_lock",
            "method_transfer": [
                "calibrated_event_streams",
                "schedule_stress_index_difficulty_axes",
                "event_driven_snapshot_evaluation",
                "observability_ablation",
            ],
            "native_data_status": "generated_dfjsp_instances_not_source_locked_for_core",
            "direct_data_admission": "no",
            "disposition": "diagnostic_only",
            "integration_notes": (
                "Use SSI-like stress axes and event calibration for JSPLIB or a "
                "future source-locked dynamic job-shop adapter; do not import "
                "generated instances as real source data."
            ),
        },
        {
            "source_id": "oragentbench",
            "title": "ORAgentBench",
            "url": "https://oragentbench.github.io/",
            "license_status": "code_and_task_assets_require_per_task_review",
            "method_transfer": [
                "executable_artifact_schema",
                "hidden_feasibility_validator",
                "normalized_objective_quality_gate",
                "solver_workflow_audit",
            ],
            "native_data_status": "offline_or_workflow_tasks_not_native_dt_runtime",
            "direct_data_admission": "no",
            "disposition": "pilot_only",
            "integration_notes": (
                "Reuse hidden-validator separation between schema validity and "
                "objective quality for task contracts and diagnostic harnesses."
            ),
        },
        {
            "source_id": "elecbench",
            "title": "ElecBench",
            "url": "https://arxiv.org/abs/2407.05365",
            "license_status": "dataset_and_question_terms_require_review",
            "method_transfer": [
                "power_safety_rubric",
                "security_and_stability_dimensions",
                "fairness_dimension_vocabulary",
                "professional_business_scenario_taxonomy",
            ],
            "native_data_status": "knowledge_and_dispatch_qa_not_replayable_runtime",
            "direct_data_admission": "no",
            "disposition": "diagnostic_only",
            "integration_notes": (
                "Map only rubric concepts to evidence-linked power metrics; "
                "never turn QA prompts into simulator episodes."
            ),
        },
        {
            "source_id": "realm_bench",
            "title": "REALM-Bench",
            "url": "https://github.com/genglongling/REALM-Bench",
            "license_status": "MIT_code_cc_by_dataset_with_upstream_terms",
            "method_transfer": [
                "multi_step_planning_decomposition",
                "dynamic_planning_task_families",
                "multi_agent_coordination_diagnostics",
            ],
            "native_data_status": "hand_authored_or_static_planning_tasks",
            "direct_data_admission": "no",
            "disposition": "diagnostic_only",
            "integration_notes": (
                "Use task decomposition and long-horizon reporting patterns; "
                "require a native backend and locked public source for rows."
            ),
        },
        {
            "source_id": "frontier_eng",
            "title": "Frontier-Eng",
            "url": "https://arxiv.org/abs/2604.12290",
            "license_status": "paper_and_task_asset_terms_require_review",
            "method_transfer": [
                "propose_execute_evaluate_revise_loop",
                "continuous_executable_verifier_feedback",
                "fixed_interaction_budget",
                "width_depth_search_diagnostic",
            ],
            "native_data_status": "engineering_tasks_not_protocol21_source_rows",
            "direct_data_admission": "no",
            "disposition": "diagnostic_only",
            "integration_notes": (
                "Adapt the interaction pattern to standing plans and simulator "
                "feedback without letting the agent author environment state."
            ),
        },
        {
            "source_id": "edgebench",
            "title": "EdgeBench",
            "url": "https://arxiv.org/abs/2607.05155",
            "license_status": "paper_and_task_release_terms_require_review",
            "method_transfer": [
                "ultra_long_horizon_episode_structure",
                "multilevel_feedback_logging",
                "learning_curve_aggregation",
                "continuous_operation_diagnostics",
            ],
            "native_data_status": "heterogeneous_real_world_tasks_not_domain_native_here",
            "direct_data_admission": "no",
            "disposition": "diagnostic_only",
            "integration_notes": (
                "Use long-horizon trajectory views and feedback cadence reports; "
                "do not import task narratives into Core."
            ),
        },
    ]
    return [dict(row) for row in rows]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("scenarios") or payload.get("rows") or []
    return list(raw) if isinstance(raw, list) else []


def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _current_release(release_dir: Path) -> dict[str, Any]:
    manifest = _load_json(release_dir / "manifest.json")
    registry = _load_json(release_dir / "registry.json")
    primary = _load_json(release_dir / "primary_suite.json")
    core = _load_json(release_dir / "core_suite.json")
    registry_rows = _rows(registry)
    primary_rows = _rows(primary)
    core_rows = _rows(core)
    diagnostics = (manifest.get("leaderboard_eligibility") or {}).get(
        "diagnostic_cells", []
    )
    registry_by_domain = _count(registry_rows, "domain")
    released_new_domains = sorted(
        domain for domain in registry_by_domain if domain not in {"power_grid", "None"}
    )
    return {
        "release_id": manifest.get("release_id") or release_dir.name,
        "counts": {
            "registry": manifest.get("n_scenarios", len(registry_rows)),
            "primary": (manifest.get("primary_suite") or {}).get(
                "n_scenarios", len(primary_rows)
            ),
            "core": (manifest.get("core_suite") or {}).get(
                "n_scenarios", len(core_rows)
            ),
            "diagnostic_cells": len(diagnostics),
            "effective_sources": (manifest.get("core_suite") or {}).get(
                "n_effective_sources"
            ),
            "physical_sources": (manifest.get("core_suite") or {}).get(
                "n_physical_sources"
            ),
        },
        "released_new_domains": released_new_domains,
        "dev_only_domains": ["disaster", "microgrid", "traffic"],
        "registry_by_domain": registry_by_domain,
        "registry_by_backend": _count(registry_rows, "backend_kind"),
        "primary_by_backend": _count(primary_rows, "backend_kind"),
        "core_by_backend": _count(core_rows, "backend_kind"),
        "source_lock_status": (manifest.get("primary_suite") or {}).get(
            "source_lock_status"
        ),
        "full_publish_ready": (
            (
                (manifest.get("behavioral_readiness") or {}).get(
                    "full_behavioral_audit"
                )
                or {}
            ).get("status")
            == "complete"
        ),
    }


def _frontier_candidate(frontier: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    candidates = frontier.get("candidates") or []
    if isinstance(candidates, dict):
        return dict(candidates.get(candidate_id) or {})
    for row in candidates:
        if row.get("candidate_id") == candidate_id:
            return dict(row)
    return {}


def _readiness_track(readiness: dict[str, Any], track_id: str) -> dict[str, Any]:
    return dict((readiness.get("tracks") or {}).get(track_id) or {})


def _candidate(
    *,
    candidate_id: str,
    name: str,
    domain: str,
    classification: str,
    priority_score: int,
    rationale: str,
    expected_delta: dict[str, Any],
    source_lock_state: str,
    license_runtime_risk: str,
    is_new_domain: bool,
    backend_native_tools: list[str],
    expected_evidence_kinds: list[str],
    behavioral_gate_state: dict[str, str],
    counterfactual_replay_feasibility: str,
    agent_environment_effect: str,
    eval_dimension_notes: dict[str, str],
    blockers: list[str],
    next_step: str,
    implementation_cost: str,
    release_boundary_risk: str,
    authorization_required: bool = False,
    authorization_prompt: str | None = None,
    supporting_reports: list[str] | None = None,
    fail_fast_assertions: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "name": name,
        "domain": domain,
        "classification": classification,
        "priority_score": priority_score,
        "rationale": rationale,
        "expected_delta": dict(expected_delta),
        "source_lock_state": source_lock_state,
        "license_runtime_risk": license_runtime_risk,
        "is_new_domain": is_new_domain,
        "backend_native_tools": backend_native_tools,
        "expected_evidence_kinds": expected_evidence_kinds,
        "behavioral_gate_state": behavioral_gate_state,
        "counterfactual_replay_feasibility": counterfactual_replay_feasibility,
        "agent_environment_effect": agent_environment_effect,
        "eval_dimension_notes": eval_dimension_notes,
        "blockers": blockers,
        "next_step": next_step,
        "implementation_cost": implementation_cost,
        "release_boundary_risk": release_boundary_risk,
        "authorization_required": authorization_required,
        "authorization_prompt": authorization_prompt,
        "supporting_reports": supporting_reports or [],
        "fail_fast_assertions": fail_fast_assertions or [],
        "release_ready": False,
        "release_reentry_ready": False,
        "proceed_commands": [],
    }


def _opendss_candidate(readiness: dict[str, Any], reports_dir: Path) -> dict[str, Any]:
    track = _readiness_track(readiness, "opendss_fresh_feeders_source_preflight")
    review = _load_json(reports_dir / "opendss_fresh_feeders_promotion_review.json")
    delta = review.get("proposed_release_delta") or {}
    expected_delta = {
        "registry": delta.get("registry"),
        "primary": delta.get("primary"),
        "core": delta.get("core"),
        "effective_sources": delta.get("core_effective_sources"),
        "physical_sources": delta.get("core_physical_sources"),
        "diagnostic_cells": delta.get("diagnostic_cells", 0),
    }
    return _candidate(
        candidate_id="opendss_ieee34_ieee123_fresh_feeders",
        name="OpenDSS IEEE34/IEEE123 fresh feeders",
        domain="power_grid_distribution",
        classification="publishable_release_candidate_authorization_required",
        priority_score=94,
        rationale=(
            "Adds two fresh physical power-flow feeder sources with verified "
            "source locks, live OpenDSS behavior, release semantics, and draft "
            "materializer artifacts."
        ),
        expected_delta=expected_delta,
        source_lock_state="source_locked_non_release_proof_chain_complete",
        license_runtime_risk="low: BSD-3 source, optional in-process dss-python",
        is_new_domain=False,
        backend_native_tools=["switch_capacitor", "set_transformer_tap"],
        expected_evidence_kinds=[
            "voltage_state_snapshot",
            "opendss_tool_effect",
            "masked_counterfactual_replay",
        ],
        behavioral_gate_state={
            "baseline_gap": "passed_non_release_behavioral_gate",
            "score_headroom": "passed_non_release_behavioral_gate",
            "state_changing_tool_effect": "passed_non_release_behavioral_gate",
            "deterministic_replay": "passed_release_semantics_probe",
            "evidence_complete": "passed_release_semantics_probe",
        },
        counterfactual_replay_feasibility=(
            "Already proven in non-release release-semantics probe."
        ),
        agent_environment_effect=(
            "Capacitor and regulator actions change OpenDSS feeder voltage state."
        ),
        eval_dimension_notes={
            "system_survival": "voltage feasibility records",
            "safety_violation": "voltage violation records",
            "optimality_gap": "not applicable until a reference OPF is added",
            "counterfactual_prevention": "masked replay proven",
        },
        blockers=list(track.get("release_blocker_codes") or [])
        + ["explicit_authorization_required"],
        next_step=(
            "Run a dedicated authorized release-boundary materialization unit "
            "that moves descriptor, dataset, scenario YAML, registry, suites, "
            "docs, audits, and readiness together."
        ),
        implementation_cost="medium",
        release_boundary_risk="medium: writes official release artifacts if authorized",
        authorization_required=True,
        authorization_prompt=track.get("authorization_prompt")
        or review.get("authorization_prompt"),
        supporting_reports=[
            "reports/opendss_fresh_feeders_source_preflight.json",
            "reports/opendss_fresh_feeders_behavioral_gate.json",
            "reports/opendss_fresh_feeders_release_semantics_probe.json",
            "reports/opendss_fresh_feeders_promotion_review.json",
        ],
        fail_fast_assertions=[
            "source file sha256 manifest must match",
            "DSS compile/solve must pass",
            "official release artifacts must remain absent until authorized",
        ],
    )


def _logistics_candidate(reports_dir: Path) -> dict[str, Any]:
    scan = _load_json(reports_dir / "logistics_standard_instance_candidates.json")
    scan_status = str(scan.get("status") or "missing_logistics_standard_instance_scan")
    summary = scan.get("summary") or {}
    n_instances = int(summary.get("n_instances", 0) or 0)
    n_consumed = int(summary.get("n_consumed_by_current_release", 0) or 0)
    n_unconsumed = int(summary.get("n_unconsumed_source_locked_instances", 0) or 0)
    n_blocked = int(summary.get("n_release_blocked_instances", 0) or 0)
    local_anchors_exhausted = (
        scan_status == "blocked_no_unconsumed_source_locked_instances"
        and n_instances > 0
        and n_unconsumed == 0
    )
    classification = (
        "blocked_no_unconsumed_local_source_locked_instances"
        if local_anchors_exhausted
        else "non_release_source_locked_candidate_scan"
    )
    priority_score = 66 if local_anchors_exhausted else 84
    source_lock_state = (
        "blocked: local source-locked VRPLIB/JSPLIB anchors are already consumed"
        if local_anchors_exhausted
        else "partial: VRPLIB/JSPLIB anchors exist; fresh instances need locks"
    )
    blockers = [
        "fresh_instance_selection_required",
        "per_instance_license_and_bound_lock_required",
        "behavioral_gate_required",
    ]
    if local_anchors_exhausted:
        blockers.insert(0, "all_local_source_locked_instances_consumed")
    next_step = (
        scan.get("next_required_proof")
        if local_anchors_exhausted and scan.get("next_required_proof")
        else (
            "Run a source-locked candidate scan over additional VRPLIB/Solomon/"
            "JSPLIB instances, then gate only fresh effective-source keys."
        )
    )
    return _candidate(
        candidate_id="logistics_standard_instance_expansion",
        name="Logistics standard-instance expansion",
        domain="logistics",
        classification=classification,
        priority_score=priority_score,
        rationale=(
            "The released Logistics domain is small but has strong oracle and "
            "known-bound structure; expanding standard VRP/JSP instances can "
            "raise non-power-grid diversity with limited runtime risk. The "
            "current local source-locked anchors are already consumed by v0.7."
        ),
        expected_delta=dict(NONE_DELTA),
        source_lock_state=source_lock_state,
        license_runtime_risk="low-medium: academic instance terms require per-file notes",
        is_new_domain=False,
        backend_native_tools=[
            "dispatch_route_stop",
            "dispatch_job_operation",
            "resequence_vehicle_route",
        ],
        expected_evidence_kinds=[
            "route_cost_delta",
            "capacity_or_time_window_violation",
            "job_shop_operation_completion",
            "makespan_delta",
        ],
        behavioral_gate_state={
            "baseline_gap": "required_for_fresh_instances",
            "score_headroom": "required_for_fresh_instances",
            "state_changing_tool_effect": "released backends already have state-changing tools",
            "deterministic_replay": "expected via deterministic simulator state",
            "evidence_complete": "must be checked per fresh row",
        },
        counterfactual_replay_feasibility=(
            "High for deterministic routing/job-shop simulators."
        ),
        agent_environment_effect=(
            "Route and operation dispatch actions directly change vehicle or "
            "machine schedules."
        ),
        eval_dimension_notes={
            "optimality_gap": "known optima/bounds are available for many instances",
            "counterfactual_prevention": "routing disruptions yes; static JSP may be N/A",
            "weighted_equity_score": "needs customer/job priority ledger if used",
        },
        blockers=blockers,
        next_step=str(next_step),
        implementation_cost="low-medium",
        release_boundary_risk="low until materialization",
        supporting_reports=[
            "reports/logistics_standard_instance_candidates.json",
            "reports/jsplib_job_shop_candidate_audit.json",
            "reports/jsplib_job_shop_candidate_gate.json",
            "release/dt_sched_bench_v0_7_0/manifest.json",
        ],
        fail_fast_assertions=[
            "known optimum or bound must be present for optimality_gap rows",
            "source_denominator_key must be fresh",
            "capacity/precedence infeasibility must raise visible gate failures",
        ],
    ) | {
        "candidate_scan_status": scan_status,
        "candidate_scan_fresh": (
            (scan.get("input_fingerprints") or {}).get("all_present") is True
            and (scan.get("input_fingerprints") or {}).get(
                "all_sha256_match_current_files"
            )
            is True
        ),
        "n_instances": n_instances,
        "n_consumed_by_current_release": n_consumed,
        "n_unconsumed_source_locked_instances": n_unconsumed,
        "n_release_blocked_instances": n_blocked,
        "consumed_source_denominator_keys": list(
            scan.get("consumed_source_denominator_keys") or []
        ),
    }


def _remaining_triage_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    buckets = {row.get("bucket_id"): row for row in report.get("triage_buckets") or []}
    fresh = buckets.get("fresh_physical_systemic_opf_inertness") or {}
    recommended = fresh.get("recommended_next_gate") or {}
    return {
        "scope": report.get("scope"),
        "n_remaining_acopf_diagnostic_cells": report.get(
            "n_remaining_acopf_diagnostic_cells"
        ),
        "n_already_consumed_acopf_lever_cells": report.get(
            "n_already_consumed_acopf_lever_cells"
        ),
        "highest_value_next_probe": report.get("highest_value_next_probe"),
        "fresh_physical_systemic_opf_inertness": {
            "n_cells": fresh.get("n_cells"),
            "cases": list(fresh.get("cases") or []),
            "recommended_next_gate_max_cells": recommended.get("max_cells"),
        },
    }


def _recommended_fresh_delta(report: dict[str, Any]) -> dict[str, int] | None:
    buckets = {row.get("bucket_id"): row for row in report.get("triage_buckets") or []}
    fresh = buckets.get("fresh_physical_systemic_opf_inertness") or {}
    recommended = fresh.get("recommended_next_gate") or {}
    cells = list(recommended.get("cells") or [])
    if not cells:
        return None
    cases = {str(cell).split("|", 1)[0] for cell in cells}
    return {
        "registry": 0,
        "primary": len(cells) * 2,
        "core": len(cells),
        "effective_sources": len(cells),
        "physical_sources": len(cases),
        "diagnostic_cells": -len(cells),
    }


def _case73_gate_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    return {
        "scope": report.get("scope"),
        "status": report.get("status"),
        "n_target_cells": report.get("n_target_cells"),
        "n_cells_evaluated": report.get("n_cells_evaluated"),
        "n_behavioral_signal_passed": report.get("n_behavioral_signal_passed"),
        "n_gate_passed": report.get("n_gate_passed"),
        "action_required": (report.get("gate_summary") or {}).get("action_required"),
        "expected_release_delta_if_all_pass": report.get(
            "expected_release_delta_if_all_pass"
        )
        or {},
        "case_ledger_delta_estimate": report.get("case_ledger_delta_estimate") or {},
    }


def _case60_probe_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    consumption = report.get("official_release_consumption") or {}
    return {
        "scope": report.get("scope"),
        "status": report.get("status"),
        "n_target_cells": report.get("n_target_cells"),
        "n_cells_evaluated": report.get("n_cells_evaluated"),
        "n_behavioral_signal_passed": report.get("n_behavioral_signal_passed"),
        "n_gate_passed": report.get("n_gate_passed"),
        "action_required": (report.get("gate_summary") or {}).get("action_required"),
        "expected_release_delta_if_all_pass": report.get(
            "expected_release_delta_if_all_pass"
        )
        or {},
        "case_ledger_delta_estimate": report.get("case_ledger_delta_estimate") or {},
        "official_release_consumption": {
            "status": consumption.get("status"),
            "all_target_release_keys_absent_from_primary_core": consumption.get(
                "all_target_release_keys_absent_from_primary_core"
            ),
            "fresh_physical_source_absent_from_current_core": consumption.get(
                "fresh_physical_source_absent_from_current_core"
            ),
            "fresh_physical_source_key": consumption.get("fresh_physical_source_key"),
        },
    }


def _case73_replay_ledger_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    summary = dict(report.get("summary") or {})
    ledger = report.get("case_ledger_delta_proof") or {}
    summary.update(
        {
            "case_ledger_delta_proof_status": ledger.get("status"),
            "all_candidate_source_keys_absent_from_current_primary_core": ledger.get(
                "all_candidate_source_keys_absent_from_current_primary_core"
            ),
            "all_release_source_keys_present_in_current_primary_core": ledger.get(
                "all_release_source_keys_present_in_current_primary_core"
            ),
            "new_primary_rows_if_promoted": ledger.get("new_primary_rows_if_promoted"),
            "new_core_rows_if_promoted": ledger.get("new_core_rows_if_promoted"),
            "new_effective_source_keys_if_promoted": ledger.get(
                "new_effective_source_keys_if_promoted"
            ),
            "new_physical_source_keys_if_promoted": ledger.get(
                "new_physical_source_keys_if_promoted"
            ),
            "physical_source_key": ledger.get("physical_source_key"),
        }
    )
    return summary


def _case60_replay_ledger_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    summary = dict(report.get("summary") or {})
    ledger = report.get("case_ledger_delta_proof") or {}
    summary.update(
        {
            "case_ledger_delta_proof_status": ledger.get("status"),
            "all_candidate_source_keys_absent_from_current_primary_core": ledger.get(
                "all_candidate_source_keys_absent_from_current_primary_core"
            ),
            "all_release_source_keys_absent_from_current_primary_core": ledger.get(
                "all_release_source_keys_absent_from_current_primary_core"
            ),
            "new_primary_rows_if_promoted": ledger.get("new_primary_rows_if_promoted"),
            "new_core_rows_if_promoted": ledger.get("new_core_rows_if_promoted"),
            "new_effective_source_keys_if_promoted": ledger.get(
                "new_effective_source_keys_if_promoted"
            ),
            "new_physical_source_keys_if_promoted": ledger.get(
                "new_physical_source_keys_if_promoted"
            ),
            "physical_source_key": ledger.get("physical_source_key"),
        }
    )
    return summary


def _case60_release_boundary_plan_summary(report: dict[str, Any]) -> dict[str, Any]:
    if not report:
        return {}
    preconditions = report.get("preconditions") or {}
    policy = report.get("release_reentry_policy") or {}
    return {
        "scope": report.get("scope"),
        "status": report.get("status"),
        "authorization_prompt": report.get("authorization_prompt"),
        "candidate_cells_accounted": preconditions.get("candidate_cells_accounted"),
        "signature_changes_accounted": preconditions.get("signature_changes_accounted"),
        "full_v0_6_behavioral_audit_required": policy.get(
            "full_v0_6_behavioral_audit_required"
        ),
        "reuse_behavioral_audit_is_insufficient": policy.get(
            "reuse_behavioral_audit_is_insufficient"
        ),
        "v0_7_must_be_rematerialized_from_promoted_v0_6": policy.get(
            "v0_7_must_be_rematerialized_from_promoted_v0_6"
        ),
    }


def _acopf_grid2op_candidate(
    readiness: dict[str, Any], reports_dir: Path
) -> dict[str, Any]:
    acopf = _readiness_track(readiness, "acopf_reserve_lever")
    emergency = _readiness_track(readiness, "acopf_emergency_reserve_lever")
    grid2op = _readiness_track(readiness, "grid2op_local_chronic_candidates")
    cross_tick = _load_json(reports_dir / "acopf_cross_tick_commitment_probe.json")
    cross_tick_gate = _load_json(reports_dir / "acopf_cross_tick_behavioral_gate.json")
    cross_tick_review = _load_json(
        reports_dir / "acopf_cross_tick_promotion_review.json"
    )
    cross_tick_dry_run = _load_json(
        reports_dir
        / "acopf_cross_tick_release_candidate_dry_run"
        / "release_candidate_dry_run.json"
    )
    cross_tick_replay = _load_json(
        reports_dir / "acopf_cross_tick_deterministic_replay_proof.json"
    )
    remaining_triage = _load_json(
        reports_dir / "acopf_remaining_diagnostic_triage.json"
    )
    case73_gate = _load_json(
        reports_dir / "acopf_case73_remaining_behavioral_gate.json"
    )
    case73_replay = _load_json(reports_dir / "acopf_case73_replay_ledger_proof.json")
    case73_review = _load_json(reports_dir / "acopf_case73_promotion_review.json")
    case60_probe = _load_json(reports_dir / "acopf_case60_fresh_physical_probe.json")
    case60_replay = _load_json(reports_dir / "acopf_case60_replay_ledger_proof.json")
    case60_review = _load_json(reports_dir / "acopf_case60_promotion_review.json")
    case60_boundary = _load_json(
        reports_dir / "acopf_case60_release_boundary_plan.json"
    )
    n_cross_tick_candidates = int(cross_tick.get("n_candidates", 0) or 0)
    n_cross_tick_gate_passed = int(cross_tick_gate.get("n_gate_passed", 0) or 0)
    n_remaining_diagnostic = int(
        remaining_triage.get("n_remaining_acopf_diagnostic_cells", 0) or 0
    )
    review_status = str(cross_tick_review.get("review_status") or "")
    review_ready = review_status == "promotion_review_ready_not_release_ready"
    already_materialized = review_status == "promotion_already_materialized_in_release"
    remaining_triage_ready = (
        remaining_triage.get("status") == "remaining_diagnostic_triage_ready"
        and n_remaining_diagnostic > 0
    )
    case73_gate_status = str(case73_gate.get("status") or "")
    case73_behavioral_signal_passed = (
        case73_gate_status == "behavioral_signal_passed_replay_and_ledger_required"
        and case73_gate.get("n_behavioral_signal_passed")
        == case73_gate.get("n_target_cells")
        == 4
    )
    case73_replay_status = str(case73_replay.get("status") or "")
    case73_replay_summary = case73_replay.get("summary") or {}
    case73_ledger = case73_replay.get("case_ledger_delta_proof") or {}
    case73_replay_ledger_ready = (
        case73_replay_status == "replay_ledger_proof_ready"
        and case73_replay_summary.get("deterministic_replay_status")
        == "passed_non_release_replay_proof"
        and case73_ledger.get("status")
        in {
            "case_ledger_delta_proof_ready",
            "case_ledger_delta_already_materialized",
        }
    )
    case73_review_status = str(case73_review.get("review_status") or "")
    case73_review_ready = case73_review_status in {
        "promotion_review_ready_not_release_ready",
        "promotion_already_materialized_in_release",
    }
    case73_already_materialized = (
        case73_review_status == "promotion_already_materialized_in_release"
    )
    case60_probe_status = str(case60_probe.get("status") or "")
    case60_consumption = case60_probe.get("official_release_consumption") or {}
    case60_behavioral_signal_passed = (
        case60_probe_status == "behavioral_signal_passed_replay_and_ledger_required"
        and case60_probe.get("n_behavioral_signal_passed")
        == case60_probe.get("n_target_cells")
        == 4
        and case60_consumption.get("all_target_release_keys_absent_from_primary_core")
        is True
        and case60_consumption.get("fresh_physical_source_absent_from_current_core")
        is True
    )
    case60_replay_status = str(case60_replay.get("status") or "")
    case60_replay_summary = case60_replay.get("summary") or {}
    case60_ledger = case60_replay.get("case_ledger_delta_proof") or {}
    case60_replay_ledger_ready = (
        case60_replay_status == "replay_ledger_proof_ready"
        and case60_replay_summary.get("deterministic_replay_status")
        == "passed_non_release_replay_proof"
        and case60_ledger.get("status") == "case_ledger_delta_proof_ready"
    )
    case60_review_status = str(case60_review.get("review_status") or "")
    case60_review_ready = (
        case60_review_status == "promotion_review_ready_not_release_ready"
    )
    case60_boundary_status = str(case60_boundary.get("status") or "")
    case60_boundary_ready = (
        case60_boundary_status == "release_boundary_plan_ready_not_authorized"
        and case60_boundary.get("release_ready") is False
        and case60_boundary.get("release_reentry_ready") is False
        and case60_boundary.get("proceed_commands") == []
    )
    dry_run_status = str(cross_tick_dry_run.get("status") or "")
    dry_run_ready = dry_run_status == "release_candidate_dry_run_ready"
    replay_status = str(cross_tick_replay.get("status") or "")
    replay_summary = cross_tick_replay.get("summary") or {}
    replay_ready = replay_status == "deterministic_replay_proof_ready"
    replay_passed = (
        replay_summary.get("deterministic_replay_status")
        == "passed_non_release_replay_proof"
    )
    fresh_delta = _recommended_fresh_delta(remaining_triage)
    expected_delta = (
        fresh_delta
        if already_materialized and remaining_triage_ready and fresh_delta is not None
        else (
            dict(
                (cross_tick_review.get("expected_release_delta_by_release") or {}).get(
                    "v0_7"
                )
                or NONE_DELTA
            )
            if review_ready
            else dict(NONE_DELTA)
        )
    )
    next_step = (
        "AC-OPF case60 release-boundary plan is ready but not authorized. "
        "Next prompt: Authorize a dedicated v0.6/v0.7 release-boundary "
        "materialization/audit unit for AC-OPF case60 high/medium cells."
        if already_materialized
        and case73_already_materialized
        and remaining_triage_ready
        and case60_behavioral_signal_passed
        and case60_replay_ledger_ready
        and case60_review_ready
        and case60_boundary_ready
        else (
            "Build a report-only case60 release-boundary materialization/audit "
            "plan that enumerates official YAML/registry/suite/manifest writes, "
            "full audit commands, blockers, and forbidden actions; do not write "
            "release artifacts."
            if already_materialized
            and case73_already_materialized
            and remaining_triage_ready
            and case60_behavioral_signal_passed
            and case60_replay_ledger_ready
            and case60_review_ready
            else (
                "Run a report-only case60 promotion review accounting for candidate "
                "signatures, primary/core deltas, diagnostic removal, and full audit "
                "requirements; do not write release artifacts until an authorized "
                "release-boundary unit."
                if already_materialized
                and case73_already_materialized
                and remaining_triage_ready
                and case60_behavioral_signal_passed
                and case60_replay_ledger_ready
                else (
                    "Run deterministic counterfactual replay and a case-ledger delta proof "
                    "for the four case60 high/medium cells; keep release artifacts untouched "
                    "until replay, ledger, promotion review, and full audits pass."
                    if already_materialized
                    and case73_already_materialized
                    and remaining_triage_ready
                    and case60_behavioral_signal_passed
                    else (
                        str(remaining_triage.get("next_required_proof"))
                        if already_materialized
                        and case73_already_materialized
                        and remaining_triage_ready
                        else (
                            "Run an authorized release-boundary materialization/audit unit for the "
                            "four case73 high/medium cells only after confirming scenario YAML, "
                            "registry, primary/core, full v0.6 audit, v0.7 wrapper checks, and v0.5 "
                            "frozen guard will move together."
                            if already_materialized
                            and case73_behavioral_signal_passed
                            and case73_replay_ledger_ready
                            and case73_review_ready
                            else (
                                "Run a report-only case73 promotion review accounting for signatures, "
                                "primary/core deltas, diagnostic removal, and full audit requirements; "
                                "do not write release artifacts until an authorized release-boundary unit."
                                if already_materialized
                                and case73_behavioral_signal_passed
                                and case73_replay_ledger_ready
                                else (
                                    "Run deterministic counterfactual replay and a case-ledger delta proof "
                                    "for the four case73 high/medium cells; keep release artifacts untouched "
                                    "until replay, ledger, promotion review, and full audits pass."
                                    if already_materialized
                                    and case73_behavioral_signal_passed
                                    else (
                                        str(remaining_triage.get("next_required_proof"))
                                        if already_materialized
                                        and remaining_triage_ready
                                        else (
                                            "Do not re-promote the already materialized case118 cross-tick cells; "
                                            "start a fresh non-release gate over the remaining unconsumed AC-OPF "
                                            "diagnostic cells or switch to a new physical OPF source."
                                        )
                                        if already_materialized
                                        else (
                                            "Run a full v0.6 behavioral audit on officially materialized case118 "
                                            "cross-tick artifacts, then rematerialize/check the v0.7 wrapper before "
                                            "any official release artifact write."
                                            if replay_ready and replay_passed
                                            else "Run deterministic counterfactual replay over the dry-run case118 "
                                            "cross-tick artifacts, then run full v0.6/v0.7 release audits before "
                                            "any official release artifact write."
                                            if dry_run_ready
                                            else (
                                                "Run a release-candidate replay/materializer dry-run for the four "
                                                "case118 cross-tick cells: materialize into a scratch path, replace "
                                                "explicit_opt_out_non_release_probe with deterministic replay evidence, "
                                                "and rerun full v0.6/v0.7 release audits before any official release "
                                                "artifact write."
                                            )
                                            if review_ready
                                            else (
                                                "Read reports/acopf_cross_tick_behavioral_gate.json, then build a "
                                                "report-only promotion review for the gate-passed case118 cells; "
                                                "do not materialize release YAML/registry/suites until that review "
                                                "accounts for new signatures, case-ledger keys, diagnostics, and "
                                                "full v0.6/v0.7 release audit impact."
                                                if n_cross_tick_gate_passed
                                                else (
                                                    "Run the bounded behavioral gate proposed by "
                                                    "reports/acopf_cross_tick_commitment_probe.json over the top "
                                                    "case118 high/medium AC-OPF cells; require baseline-gap, "
                                                    "score-headroom, state-changing commit_reserve evidence, and replay "
                                                    "proof before any release materialization."
                                                    if n_cross_tick_candidates
                                                    else (
                                                        "Read reports/acopf_grid2op_decision_lever_repair_plan.json, then "
                                                        "prototype the highest-value fresh AC-OPF/Grid2Op decision-pressure "
                                                        "probe without release writes."
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    if (
        already_materialized
        and case73_already_materialized
        and remaining_triage_ready
        and case60_behavioral_signal_passed
        and case60_replay_ledger_ready
        and case60_review_ready
        and case60_boundary_ready
    ):
        behavioral_state = {
            "baseline_gap": "passed_non_release_case60_behavioral_probe",
            "score_headroom": "passed_non_release_case60_behavioral_probe",
            "state_changing_tool_effect": "passed_non_release_case60_behavioral_probe",
            "deterministic_replay": "passed_non_release_case60_replay_proof",
            "evidence_complete": "case60_release_boundary_plan_ready_not_authorized",
        }
    elif (
        already_materialized
        and case73_already_materialized
        and remaining_triage_ready
        and case60_behavioral_signal_passed
        and case60_replay_ledger_ready
        and case60_review_ready
    ):
        behavioral_state = {
            "baseline_gap": "passed_non_release_case60_behavioral_probe",
            "score_headroom": "passed_non_release_case60_behavioral_probe",
            "state_changing_tool_effect": "passed_non_release_case60_behavioral_probe",
            "deterministic_replay": "passed_non_release_case60_replay_proof",
            "evidence_complete": "case60_promotion_review_ready_not_release_ready",
        }
    elif (
        already_materialized
        and case73_already_materialized
        and remaining_triage_ready
        and case60_behavioral_signal_passed
        and case60_replay_ledger_ready
    ):
        behavioral_state = {
            "baseline_gap": "passed_non_release_case60_behavioral_probe",
            "score_headroom": "passed_non_release_case60_behavioral_probe",
            "state_changing_tool_effect": "passed_non_release_case60_behavioral_probe",
            "deterministic_replay": "passed_non_release_case60_replay_proof",
            "evidence_complete": "passed_non_release_case60_replay_ledger_proof",
        }
    elif (
        already_materialized
        and case73_already_materialized
        and remaining_triage_ready
        and case60_behavioral_signal_passed
    ):
        behavioral_state = {
            "baseline_gap": "passed_non_release_case60_behavioral_probe",
            "score_headroom": "passed_non_release_case60_behavioral_probe",
            "state_changing_tool_effect": "passed_non_release_case60_behavioral_probe",
            "deterministic_replay": "required_for_fresh_case60_cells",
            "evidence_complete": "case60_behavioral_signal_passed_replay_required",
        }
    elif (
        already_materialized and case73_already_materialized and remaining_triage_ready
    ):
        behavioral_state = {
            "baseline_gap": "required_for_fresh_case60_cells",
            "score_headroom": "required_for_fresh_case60_cells",
            "state_changing_tool_effect": "required_for_fresh_case60_cells",
            "deterministic_replay": "required_for_fresh_case60_cells",
            "evidence_complete": "case73_consumed_start_fresh_remaining_gate",
        }
    elif (
        already_materialized
        and case73_behavioral_signal_passed
        and case73_replay_ledger_ready
        and case73_review_ready
    ):
        behavioral_state = {
            "baseline_gap": "passed_non_release_case73_behavioral_gate",
            "score_headroom": "passed_non_release_case73_behavioral_gate",
            "state_changing_tool_effect": "passed_non_release_case73_behavioral_gate",
            "deterministic_replay": "passed_non_release_case73_replay_proof",
            "evidence_complete": "case73_promotion_review_ready_not_release_ready",
        }
    elif (
        already_materialized
        and case73_behavioral_signal_passed
        and case73_replay_ledger_ready
    ):
        behavioral_state = {
            "baseline_gap": "passed_non_release_case73_behavioral_gate",
            "score_headroom": "passed_non_release_case73_behavioral_gate",
            "state_changing_tool_effect": "passed_non_release_case73_behavioral_gate",
            "deterministic_replay": "passed_non_release_case73_replay_proof",
            "evidence_complete": "passed_non_release_case73_replay_ledger_proof",
        }
    elif already_materialized and case73_behavioral_signal_passed:
        behavioral_state = {
            "baseline_gap": "passed_non_release_case73_behavioral_gate",
            "score_headroom": "passed_non_release_case73_behavioral_gate",
            "state_changing_tool_effect": "passed_non_release_case73_behavioral_gate",
            "deterministic_replay": "required_for_case73_remaining_cells",
            "evidence_complete": "case73_behavioral_signal_passed_replay_required",
        }
    elif already_materialized and remaining_triage_ready:
        behavioral_state = {
            "baseline_gap": "required_for_remaining_diagnostic_cells",
            "score_headroom": "required_for_remaining_diagnostic_cells",
            "state_changing_tool_effect": "required_for_remaining_diagnostic_cells",
            "deterministic_replay": "required_for_remaining_diagnostic_cells",
            "evidence_complete": "required_for_remaining_diagnostic_cells",
        }
    else:
        behavioral_state = {
            "baseline_gap": (
                "consumed_by_current_release"
                if already_materialized
                else "passed_non_release_behavioral_gate"
            ),
            "score_headroom": (
                "consumed_by_current_release"
                if already_materialized
                else "passed_non_release_behavioral_gate"
            ),
            "state_changing_tool_effect": (
                "consumed_by_current_release"
                if already_materialized
                else "passed_non_release_behavioral_gate"
            ),
            "deterministic_replay": (
                "consumed_by_current_release"
                if already_materialized
                else "passed_non_release_replay_proof"
                if replay_ready and replay_passed
                else "explicit_opt_out_non_release_probe"
            ),
            "evidence_complete": (
                "consumed_by_current_release"
                if already_materialized
                else "passed_non_release_replay_proof"
                if replay_ready and replay_passed
                else "release_candidate_dry_run_ready"
                if dry_run_ready
                else "promotion_review_ready_not_release_ready"
                if review_ready
                else "promotion_review_required"
            ),
        }
    if not n_cross_tick_gate_passed:
        behavioral_state = {
            "baseline_gap": "fresh_cells_required",
            "score_headroom": "fresh_cells_required",
            "state_changing_tool_effect": "required",
            "deterministic_replay": "must remain deterministic",
            "evidence_complete": "required",
        }
    return _candidate(
        candidate_id="acopf_grid2op_decision_lever_repair",
        name="AC-OPF/Grid2Op decision-lever repair",
        domain="power_grid",
        classification=(
            "non_release_mechanism_discovery_followup"
            if already_materialized and remaining_triage_ready
            else "consumed_current_release_followup_only"
            if already_materialized
            else "non_release_mechanism_discovery"
        ),
        priority_score=78,
        rationale=(
            "Repairs the highest-value existing power-grid debt: inert AC-OPF "
            "cells and sparse Grid2Op chronic diversity."
        ),
        expected_delta=expected_delta,
        source_lock_state="uses existing source-locked PGLib-OPF/Grid2Op inputs",
        license_runtime_risk="low for existing sources; optional backend availability required",
        is_new_domain=False,
        backend_native_tools=["commit_reserve", "shed_load", "redispatch_generator"],
        expected_evidence_kinds=[
            "reserve_commitment_event",
            "emergency_reserve_protection_failed",
            "grid2op_topology_or_redispatch_effect",
            "baseline_gap_trace",
        ],
        behavioral_gate_state=behavioral_state,
        counterfactual_replay_feasibility=(
            "Medium-high; existing release already supports replay, but new "
            "mechanisms must prove deterministic action masking."
        ),
        agent_environment_effect=(
            "Reserve/topology/dispatch choices must change future feasibility or cost, "
            "not merely mirror per-tick OPF."
        ),
        eval_dimension_notes={
            "optimality_gap": "AC-OPF has reference optimum; Grid2Op topology does not",
            "adaptive_replanning": "requires state-changing evidence",
            "foresight_score": "should stress cross-tick commitment",
        },
        blockers=[
            acopf.get("status", "acopf_status_unknown"),
            emergency.get("status", "emergency_status_unknown"),
            grid2op.get("status", "grid2op_status_unknown"),
            "fresh_gate_clean_cells_required",
        ],
        next_step=next_step,
        implementation_cost="medium-high",
        release_boundary_risk="medium: likely changes scenario signatures if promoted",
        supporting_reports=[
            "reports/acopf_grid2op_decision_lever_repair_plan.json",
            "reports/acopf_cross_tick_commitment_probe.json",
            "reports/acopf_cross_tick_behavioral_gate.json",
            "reports/acopf_cross_tick_promotion_review.json",
            "reports/acopf_cross_tick_release_candidate_dry_run/release_candidate_dry_run.json",
            "reports/acopf_cross_tick_deterministic_replay_proof.json",
            "reports/acopf_remaining_diagnostic_triage.json",
            "reports/acopf_case73_remaining_behavioral_gate.json",
            "reports/acopf_case73_replay_ledger_proof.json",
            "reports/acopf_case73_promotion_review.json",
            "reports/acopf_case60_fresh_physical_probe.json",
            "reports/acopf_case60_replay_ledger_proof.json",
            "reports/acopf_case60_promotion_review.json",
            "reports/acopf_case60_release_boundary_plan.json",
            "reports/acopf_reserve_lever_candidates.json",
            "reports/acopf_emergency_reserve_candidates.json",
            "reports/grid2op_chronic_candidate_gate.json",
        ],
        fail_fast_assertions=[
            "oracle_opf_inert cells must stay diagnostic",
            "new scenario signatures must be listed before promotion",
            "same-chronic variants must not inflate physical source count",
        ],
    ) | {
        "n_cross_tick_commitment_candidates": n_cross_tick_candidates,
        "cross_tick_candidate_count_basis": cross_tick.get("candidate_count_basis"),
        "cross_tick_gate_status": cross_tick.get("gate_status") or {},
        "n_cross_tick_gate_passed": n_cross_tick_gate_passed,
        "cross_tick_behavioral_gate_summary": cross_tick_gate.get("gate_summary") or {},
        "cross_tick_promotion_review_status": review_status or None,
        "cross_tick_promotion_review_summary": (
            cross_tick_review.get("affected_summary") or {}
        ),
        "cross_tick_release_candidate_dry_run_status": dry_run_status or None,
        "cross_tick_release_candidate_dry_run_summary": (
            cross_tick_dry_run.get("summary") or {}
        ),
        "cross_tick_deterministic_replay_proof_status": replay_status or None,
        "cross_tick_deterministic_replay_proof_summary": replay_summary,
        "cross_tick_expected_delta_by_release": (
            cross_tick_review.get("expected_release_delta_by_release") or {}
        ),
        "cross_tick_expected_post_promotion_counts_by_release": (
            cross_tick_review.get("expected_post_promotion_counts_by_release") or {}
        ),
        "cross_tick_release_reentry_blockers": (
            []
            if already_materialized
            else cross_tick_replay.get("blocking_release_gates")
            or cross_tick_dry_run.get("blocking_release_gates")
            or cross_tick_review.get("release_reentry_blockers")
            or []
        ),
        "remaining_diagnostic_triage_status": remaining_triage.get("status"),
        "remaining_diagnostic_triage_summary": _remaining_triage_summary(
            remaining_triage
        ),
        "case73_remaining_behavioral_gate_status": case73_gate_status or None,
        "case73_remaining_behavioral_gate_summary": _case73_gate_summary(case73_gate),
        "case73_replay_ledger_proof_status": case73_replay_status or None,
        "case73_replay_ledger_proof_summary": _case73_replay_ledger_summary(
            case73_replay
        ),
        "case73_expected_post_promotion_counts_by_release": {
            "v0_6": case73_ledger.get("v0_6_expected_post_promotion_counts") or {},
            "v0_7": case73_ledger.get("v0_7_expected_post_promotion_counts") or {},
        },
        "case73_promotion_review_status": case73_review_status or None,
        "case73_promotion_review_summary": (
            case73_review.get("affected_summary") or {}
        ),
        "case73_release_reentry_blockers": (
            case73_review.get("release_reentry_blockers") or []
        ),
        "case60_fresh_physical_probe_status": case60_probe_status or None,
        "case60_fresh_physical_probe_summary": _case60_probe_summary(case60_probe),
        "case60_replay_ledger_proof_status": case60_replay_status or None,
        "case60_replay_ledger_proof_summary": _case60_replay_ledger_summary(
            case60_replay
        ),
        "case60_expected_post_promotion_counts_by_release": {
            "v0_6": case60_ledger.get("v0_6_expected_post_promotion_counts") or {},
            "v0_7": case60_ledger.get("v0_7_expected_post_promotion_counts") or {},
        },
        "case60_promotion_review_status": case60_review_status or None,
        "case60_promotion_review_summary": (
            case60_review.get("affected_summary") or {}
        ),
        "case60_release_reentry_blockers": (
            case60_review.get("release_reentry_blockers")
            if case60_review_ready
            else case60_replay.get("blocking_release_gates") or []
        ),
        "case60_release_boundary_plan_status": case60_boundary_status or None,
        "case60_release_boundary_plan_summary": (
            _case60_release_boundary_plan_summary(case60_boundary)
        ),
        "case60_release_boundary_blockers": (
            case60_boundary.get("release_blockers") or []
        ),
    }


def _microgrid_candidate(readiness: dict[str, Any]) -> dict[str, Any]:
    track = _readiness_track(readiness, "microgrid_overlays")
    return _candidate(
        candidate_id="real_microgrid_overlays",
        name="Real microgrid overlays",
        domain="microgrid",
        classification="blocked_waiting_for_source_locked_data",
        priority_score=76,
        rationale=(
            "Would add a true energy-management domain with storage, DER, price, "
            "and reliability tradeoffs, but release provenance is not ready."
        ),
        expected_delta=dict(NONE_DELTA),
        source_lock_state="blocked: real .npz overlays and sidecars missing/invalid",
        license_runtime_risk="medium: NREL/OEDI/NSRDB terms and sidecars required",
        is_new_domain=True,
        backend_native_tools=[
            "set_battery_dispatch",
            "set_generator_dispatch",
            "shed_load",
            "set_der_reactive_power",
        ],
        expected_evidence_kinds=[
            "microgrid_energy_balance",
            "storage_state_of_charge_trace",
            "unserved_load_event",
            "dispatch_cost_delta",
        ],
        behavioral_gate_state={
            "baseline_gap": "blocked_until_real_overlays",
            "score_headroom": "blocked_until_real_overlays",
            "state_changing_tool_effect": "dev-only backend exists",
            "deterministic_replay": "must be proven after source lock",
            "evidence_complete": "must be proven after source lock",
        },
        counterfactual_replay_feasibility=(
            "Likely high once deterministic overlays are source-locked."
        ),
        agent_environment_effect=(
            "Battery/DER/load actions change power balance, cost, and reliability state."
        ),
        eval_dimension_notes={
            "economic_cost": "price trace required",
            "safety_violation": "unserved load / voltage if LV power flow active",
            "weighted_equity_score": "needs customer/critical-load ledger",
        },
        blockers=[
            "real_npz_overlay_required",
            "public_http_url_required",
            "source_ids.nsrdb_site_required",
            track.get("status", "microgrid_status_unknown"),
        ],
        next_step=(
            "Provide real source-locked Phoenix/Denver overlays with sidecars, "
            "then rerun microgrid preflight before any release materialization."
        ),
        implementation_cost="medium",
        release_boundary_risk="medium-high until provenance is clean",
        supporting_reports=["reports/microgrid_overlay_preflight.json"],
        fail_fast_assertions=[
            "invalid_provenance_field:url",
            "provenance_site_mismatch",
            "provenance_source_id_mismatch",
            "non-finite or negative physical arrays must fail preflight",
        ],
    )


def _frontier_track_candidate(
    *,
    row: dict[str, Any],
    candidate_id: str,
    name: str,
    priority_score: int,
    rationale: str,
) -> dict[str, Any]:
    ladder = row.get("adapter_ladder_sketch") or {}
    source_lock = row.get("source_lock_preflight") or {}
    return _candidate(
        candidate_id=candidate_id,
        name=name,
        domain=str(row.get("domain") or "unknown_domain"),
        classification="non_release_frontier_preflight",
        priority_score=priority_score,
        rationale=rationale,
        expected_delta=dict(NONE_DELTA),
        source_lock_state="blocked: " + str(row.get("preflight_status")),
        license_runtime_risk="unverified package/license/runtime boundary",
        is_new_domain=True,
        backend_native_tools=list(ladder.get("expected_native_tools") or []),
        expected_evidence_kinds=list(ladder.get("expected_evidence_kinds") or []),
        behavioral_gate_state={
            "baseline_gap": "not_run_frontier_only",
            "score_headroom": "not_run_frontier_only",
            "state_changing_tool_effect": "adapter_missing",
            "deterministic_replay": "not_proven",
            "evidence_complete": "not_proven",
        },
        counterfactual_replay_feasibility=(
            "Unknown until package/runtime determinism and adapter state snapshots are proven."
        ),
        agent_environment_effect=(
            "Expected native controls should change simulator state, but no adapter "
            "proof exists yet."
        ),
        eval_dimension_notes={
            "optimality_gap": "requires oracle or known bound",
            "counterfactual_prevention": "requires deterministic replay or explicit opt-out",
            "adaptive_replanning": "requires state-changing tool evidence",
        },
        blockers=list(row.get("release_blocker_codes") or [])
        + list(source_lock.get("release_blockers") or []),
        next_step=(
            "Run package/license/source-lock preflight in a throwaway dev "
            "environment; keep adapters and release artifacts out until the "
            "full ladder passes."
        ),
        implementation_cost="medium",
        release_boundary_risk="low now, high if promoted before license/source gates",
        supporting_reports=["reports/frontier_domain_candidates.json"],
        fail_fast_assertions=[
            "package metadata and upstream license must match",
            "source lock fields must be verified before adapter work",
            "repo-local dependency install remains forbidden",
        ],
    )


def _grid2op_ieee118_candidate(readiness: dict[str, Any]) -> dict[str, Any]:
    track = _readiness_track(readiness, "grid2op_ieee118_acquisition")
    return _candidate(
        candidate_id="trusted_grid2op_ieee118_acquisition",
        name="Trusted Grid2Op IEEE-118 acquisition",
        domain="power_grid_transmission",
        classification="blocked_waiting_for_verified_source_acquisition",
        priority_score=70,
        rationale=(
            "True IEEE-118 L2RPN chronics would raise physical-grid diversity, "
            "but current acquisition evidence is blocked by TLS/mirror trust."
        ),
        expected_delta=dict(NONE_DELTA),
        source_lock_state="blocked: verified TLS or sha256-locked mirror required",
        license_runtime_risk="medium: large external data acquisition",
        is_new_domain=False,
        backend_native_tools=["redispatch_generator", "set_line_status", "shed_load"],
        expected_evidence_kinds=[
            "rho_overload_trace",
            "line_disconnection_event",
            "topology_action_effect",
        ],
        behavioral_gate_state={
            "baseline_gap": "blocked_until_source_acquired",
            "score_headroom": "blocked_until_source_acquired",
            "state_changing_tool_effect": "required",
            "deterministic_replay": "required",
            "evidence_complete": "required",
        },
        counterfactual_replay_feasibility=(
            "Expected through Grid2Op once a trusted local env is acquired."
        ),
        agent_environment_effect=(
            "Topology and redispatch actions should alter overload and cascade outcomes."
        ),
        eval_dimension_notes={
            "optimality_gap": "not applicable unless a topology-control reference is added",
            "counterfactual_prevention": "masked replay required",
        },
        blockers=[
            "verified_tls_or_sha256_locked_mirror_required",
            track.get("status", "grid2op_acquisition_status_unknown"),
        ],
        next_step=(
            "Provide a trusted mirror with expected_sha256 and lock strategy, then "
            "run acquisition/source-lock/behavioral gates."
        ),
        implementation_cost="medium-high",
        release_boundary_risk="medium",
        supporting_reports=[
            "reports/grid2op_idf_2023_acquisition_gate.json",
            "reports/grid2op_wcci_2022_acquisition_gate.json",
        ],
        fail_fast_assertions=[
            "no TLS bypass",
            "mirror sha256 must match",
            "same-chronic variants must not count as independent physical sources",
        ],
    )


def build_expansion_priority_matrix(
    *,
    release_dir: Path = DEFAULT_RELEASE,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
) -> dict[str, Any]:
    release = _current_release(release_dir)
    readiness = _load_json(reports_dir / "data_expansion_readiness.json")
    frontier = _load_json(reports_dir / "frontier_domain_candidates.json")

    flatland = _frontier_candidate(frontier, "flatland_rail")
    citylearn = _frontier_candidate(frontier, "citylearn_building_energy")
    or_gym = _frontier_candidate(frontier, "or_gym_inventory")
    sumo = _frontier_candidate(frontier, "sumo_rl_traffic_signal")

    candidates = [
        _opendss_candidate(readiness, reports_dir),
        _logistics_candidate(reports_dir),
        _acopf_grid2op_candidate(readiness, reports_dir),
        _microgrid_candidate(readiness),
        _frontier_track_candidate(
            row=sumo,
            candidate_id="sumo_rl_or_resco_traffic_signal_control",
            name="SUMO-RL/RESCO traffic-signal control",
            priority_score=74,
            rationale=(
                "Would upgrade Traffic from mock/dev-only to real microscopic "
                "traffic feedback with native signal-control decisions."
            ),
        ),
        _frontier_track_candidate(
            row=flatland,
            candidate_id="flatland_railway_scheduling",
            name="Flatland railway scheduling",
            priority_score=72,
            rationale=(
                "Adds multi-agent path conflict and disruption rescheduling, "
                "a qualitatively new scheduling mechanic."
            ),
        ),
        _grid2op_ieee118_candidate(readiness),
        _frontier_track_candidate(
            row=citylearn,
            candidate_id="citylearn_building_energy",
            name="CityLearn building energy",
            priority_score=68,
            rationale=(
                "Adds demand-response, comfort, emissions, and equity/stakeholder "
                "pressure beyond the current power-grid/logistics mix."
            ),
        ),
        _frontier_track_candidate(
            row=or_gym,
            candidate_id="or_gym_inventory_supply_chain",
            name="OR-Gym inventory/supply-chain",
            priority_score=64,
            rationale=(
                "Adds replenishment and service-level tradeoffs, but needs package "
                "and demand-trace source locks before adapter work."
            ),
        ),
    ]
    candidates.sort(key=lambda row: (-int(row["priority_score"]), row["candidate_id"]))
    for idx, row in enumerate(candidates, 1):
        row["priority_rank"] = idx

    highest_release = next(
        (
            row["candidate_id"]
            for row in candidates
            if row["classification"]
            == "publishable_release_candidate_authorization_required"
        ),
        None,
    )
    highest_unblocked = next(
        (
            row["candidate_id"]
            for row in candidates
            if row["classification"]
            in {
                "non_release_source_locked_candidate_scan",
                "non_release_mechanism_discovery",
                "non_release_mechanism_discovery_followup",
            }
        ),
        None,
    )
    return {
        "schema_version": "0.1",
        "scope": "benchmark_expansion_priority_matrix",
        "non_release_artifact": True,
        "release_ready": False,
        "release_reentry_ready": False,
        "proceed_commands": [],
        "policy": dict(POLICY),
        "current_release": release,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "method_transfer_catalog": build_method_transfer_catalog(),
        "top_recommendations": {
            "highest_release_candidate_if_authorized": highest_release,
            "highest_unblocked_non_release_work": highest_unblocked,
            "highest_blocked_data_track": "real_microgrid_overlays",
            "highest_frontier_preflight": "sumo_rl_or_resco_traffic_signal_control",
        },
        "input_reports": [
            "release/dt_sched_bench_v0_7_0/manifest.json",
            "release/dt_sched_bench_v0_7_0/registry.json",
            "release/dt_sched_bench_v0_7_0/primary_suite.json",
            "release/dt_sched_bench_v0_7_0/core_suite.json",
            "reports/data_expansion_readiness.json",
            "reports/frontier_domain_candidates.json",
            "reports/opendss_fresh_feeders_promotion_review.json",
            "reports/microgrid_overlay_preflight.json",
        ],
        "next_required_proof": (
            "Pick exactly one candidate and run its next non-release proof or "
            "authorized release-boundary unit; do not promote any row from this "
            "matrix alone."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    release = report["current_release"]
    counts = release["counts"]
    candidates_by_id = {row["candidate_id"]: row for row in report["candidates"]}
    top = report["top_recommendations"]
    highest_release_id = top.get("highest_release_candidate_if_authorized")
    highest_release = candidates_by_id.get(str(highest_release_id), {})
    highest_unblocked_id = top.get("highest_unblocked_non_release_work")
    highest_unblocked = candidates_by_id.get(str(highest_unblocked_id), {})
    lines = [
        "# vNext Expansion Priority",
        "",
        "> Machine-readable source: "
        f"`{DEFAULT_OUTPUT.relative_to(REPO_ROOT)}`. This document is generated "
        "from the current release manifest and non-release preflight reports.",
        "",
        "## Current Truth",
        "",
        (
            f"- v0.7 release: {counts['registry']} -> {counts['primary']} -> "
            f"{counts['core']}; effective/physical "
            f"{counts['effective_sources']}/{counts['physical_sources']}; "
            f"diagnostic cells {counts['diagnostic_cells']}."
        ),
        "- Released new domain: Logistics. OpenDSS IEEE13 is a power-grid extension.",
        "- Traffic/Disaster/Microgrid/frontier candidates remain non-release.",
        "",
        "## Ranked Tracks",
        "",
        "| Rank | Track | Classification | Expected Delta | Next Step |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in report["candidates"]:
        delta = row["expected_delta"]
        delta_text = (
            "TBD"
            if delta["registry"] is None
            else (
                f"registry {delta['registry']}, primary {delta['primary']}, "
                f"core {delta['core']}, effective {delta['effective_sources']}, "
                f"physical {delta['physical_sources']}"
            )
        )
        lines.append(
            f"| {row['priority_rank']} | {row['name']} | "
            f"{row['classification']} | {delta_text} | {row['next_step']} |"
        )
    lines.extend(
        [
            "",
            "## Immediate Recommendation",
            "",
            (
                "- Highest publishable candidate, if explicitly authorized: "
                f"`{highest_release_id}` ({highest_release.get('name')})."
            ),
            (
                "- Highest unblocked non-release work: "
                f"`{highest_unblocked_id}` ({highest_unblocked.get('name')})."
            ),
            "",
            "## Artifact Paths",
            "",
            f"- JSON: `{DEFAULT_OUTPUT.relative_to(REPO_ROOT)}`",
            f"- Markdown: `{DEFAULT_MARKDOWN_OUTPUT.relative_to(REPO_ROOT)}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_markdown(markdown: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    args = parser.parse_args(argv)

    report = build_expansion_priority_matrix(
        release_dir=args.release_dir, reports_dir=args.reports_dir
    )
    write_report(report, args.output)
    write_markdown(render_markdown(report), args.markdown_output)
    print(
        json.dumps(
            {
                "status": "benchmark_expansion_priority_matrix_written",
                "output": str(args.output),
                "markdown_output": str(args.markdown_output),
                "n_candidates": report["n_candidates"],
                "top_recommendations": report["top_recommendations"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
