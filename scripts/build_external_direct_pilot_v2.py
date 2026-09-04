#!/usr/bin/env python3
"""Build a source-audited external benchmark catalog and one REALM JSSP pilot.

This is deliberately an isolated staging builder.  It does not modify the
Protocol-2.1 working set and it never claims that an external task is Core.
The REALM pilot proves the narrower statement that a locked public JSSP graph
can execute on the existing native Job-Shop backend.  Admission stays held
until the J2 disruption sidecar itself is consumed and evidenced at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tool_protocol import DifficultyImperfectionProfile  # noqa: E402
from domains.logistics.seeds.from_jsplib import parse_jsplib_instance  # noqa: E402
from domains.logistics.seeds.schema import (  # noqa: E402
    LogisticsScenarioSeed,
    Perturbation,
    Provenance,
)
from domains.source_contracts import jsplib_job_shop  # noqa: E402
from runner.resume import recompute_signature_with_seed  # noqa: E402
from scripts.audit_core_difficulty import _semantic_fingerprint  # noqa: E402
from scripts.build_primary_suite import structural_fingerprint  # noqa: E402

REALM_COMMIT = "9c3aa2ae97d65198f6ee29fe942d99f9b3a9c6eb"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "works" / "REALM-Bench-direct-pilot"
DEFAULT_STAGING_PATH = (
    REPO_ROOT
    / "scenarios"
    / "staging"
    / "v0_52_external_realm_j2_native_pilot"
    / "realm_rcmax_20_15_1_dynamic_high_s71.yaml"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "protocol21_expansion_trials"
    / "external_realm_j2_direct_pilot_v1"
)

_REALM_REQUIRED_GATE_STAGES = (
    "preflight",
    "behavioral",
    "source_consumption",
    "task_contracts",
    "complexity",
    "observed_reference_depth",
    "strategy_depth",
    "source_grounded",
    "agentic_contract",
    "materialize_core",
)
_REALM_GATE_ARTIFACTS = {
    "source_consumption": "source_consumption_protocol2_v21.json",
    "task_contracts": "task_contracts_protocol2_v21.json",
    "source_grounded": "source_grounded_protocol2_v21.json",
    "agentic_contract": "agentic_core_contract_protocol2_v21.json",
    "materialize_core": "refined_core_selection_protocol2_v21.json",
}


@dataclass(frozen=True)
class RealmSourceLock:
    """Exact source identities needed by the standalone REALM pilot."""

    commit: str
    raw_relative_path: str
    raw_sha256: str
    j2_relative_path: str
    j2_sha256: str
    disrupted_instance_id: str


DEFAULT_REALM_LOCK = RealmSourceLock(
    commit=REALM_COMMIT,
    raw_relative_path="datasets/J1/DMU/rcmax_20_15_1.txt",
    raw_sha256="4ca12bc77023f4bd10ea8c7461a1442b61528fb96a0c9cafbfb57e3b2be1046f",
    j2_relative_path="datasets/clean/JSSP/J2.json",
    j2_sha256="6202b6019857d7dc2939e34f8259b719ffa1bf48f0690a627c1a5a7446adef65",
    disrupted_instance_id="rcmax_20_15_1_disrupted",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _locked_file(root: Path, relative_path: str, expected: str, label: str) -> Path:
    path = root / relative_path
    if not path.is_file():
        raise ValueError(f"REALM {label} asset missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"REALM {label} sha256 mismatch: expected={expected} actual={actual}"
        )
    return path


def _realm_sidecar_instance(path: Path, instance_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("REALM J2 sidecar must contain an instances list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("instance_id") == instance_id
    ]
    if len(matches) != 1:
        raise ValueError(f"REALM J2 instance selection is not unique: {instance_id}")
    return matches[0]


def _normalized_sidecar_graph(row: dict[str, Any]) -> list[list[dict[str, int]]]:
    jobs = row.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("REALM J2 instance is missing jobs")
    normalized: list[list[dict[str, int]]] = []
    for job_idx, job in enumerate(jobs):
        if not isinstance(job, list):
            raise ValueError(f"REALM J2 job {job_idx} is not a list")
        normalized.append(
            [
                {
                    # REALM JSON uses 1-based machine identifiers.  The raw
                    # DMU/JSPLIB file and native backend use 0-based IDs.
                    "machine": int(operation["machine"]) - 1,
                    "duration": int(operation["processing_time"]),
                }
                for operation in job
            ]
        )
    return normalized


def _source_breakdown(row: dict[str, Any]) -> dict[str, int]:
    disruptions = row.get("disruptions")
    if not isinstance(disruptions, list):
        raise ValueError("REALM J2 instance is missing disruptions")
    matches = [
        item
        for item in disruptions
        if isinstance(item, dict) and item.get("type") == "machine_breakdown"
    ]
    if len(matches) != 1:
        raise ValueError("REALM pilot requires exactly one source machine breakdown")
    item = matches[0]
    machine_id = int(item["machine_id"]) - 1
    trigger_tick = int(item["start_time"])
    duration_ticks = int(item["duration"])
    if machine_id < 0 or trigger_tick <= 0 or duration_ticks <= 0:
        raise ValueError("REALM J2 breakdown has invalid native bounds")
    return {
        "machine_id": machine_id,
        "trigger_tick": trigger_tick,
        "duration_ticks": duration_ticks,
    }


def _realm_case_ledger(
    *,
    source_identity: str,
    physical_source_key: str,
    source_raw: str,
    raw_sha256: str,
    source_j2: str,
    j2_sha256: str,
    instance_id: str,
    breakdown: dict[str, int],
    parsed: dict[str, Any],
    difficulty_mode: str,
    difficulty_level: str,
) -> dict[str, Any]:
    """Return the deterministic case ledger shared by pilot and suite row."""
    return {
        "schema_version": "0.1",
        "source_denominator_key": source_identity,
        "independence_axis": "realm_j2_selected_instance",
        "decision_variant_key": (
            f"{instance_id}:machine_breakdown:{breakdown['machine_id']}"
            f"@{breakdown['trigger_tick']}+{breakdown['duration_ticks']}"
        ),
        "decision_pressure_axis": (
            "source_observed_machine_breakdown_recovery_and_long_horizon_scheduling"
        ),
        "additional_decision_axis": (
            f"J2 selected_instance_id={instance_id}; "
            f"difficulty={difficulty_mode}/{difficulty_level}"
        ),
        "complexity_tags": [
            "external_source_pilot",
            "source_observed_machine_breakdown",
            f"n_jobs={int(parsed['jobs'])}",
            f"n_machines={int(parsed['machines'])}",
            f"n_operations={int(parsed['operations'])}",
            f"breakdown_tick={breakdown['trigger_tick']}",
            f"breakdown_duration_ticks={breakdown['duration_ticks']}",
        ],
        "physical_source_key": physical_source_key,
        "physical_source_lock": {
            "schema_version": "source_asset_graph_v1",
            "backend_kind": "jsplib_job_shop",
            "required_source_assets": [
                {"declared_path": source_raw, "sha256": raw_sha256},
                {"declared_path": source_j2, "sha256": j2_sha256},
            ],
        },
        "keep_rationale": (
            "Independent REALM J2 selected instance with a locked raw operation "
            "graph, runtime-consumed disruption sidecar, and native recovery "
            "controls; held outside Core pending isolated gates."
        ),
        "diagnostic_risk": [
            "external_pilot_not_core",
            "upstream_dmu_license_chain_review_pending",
        ],
    }


def _scenario_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _serialized_sha256(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _inspect_realm_protocol21_evidence(
    *,
    evidence_dir: Path | None,
    source_suite: dict[str, Any],
) -> dict[str, Any]:
    """Verify row-level Protocol-2.1 evidence without trusting a stale report.

    A singleton pilot is expected to fail release coverage/readiness.  The
    evidence we need here is the row-level pipeline through materialization;
    release coverage and readiness are recorded separately and may return 4
    solely because the pilot is not a five-domain suite.
    """
    result: dict[str, Any] = {
        "status": "not_verified",
        "evidence_dir": str(evidence_dir) if evidence_dir else None,
        "source_suite_sha256": _serialized_sha256(source_suite),
        "manifest_source_suite_sha256": None,
        "source_suite_hash_matches": False,
        "metadata_only_drift": False,
        "required_stage_return_codes": {},
        "missing_artifacts": [],
        "selected_scenario_ids": [],
        "reason": "protocol21_full_gates_not_run",
    }
    if evidence_dir is None:
        return result
    manifest_path = evidence_dir / "protocol2_v21_pipeline_manifest.json"
    if not manifest_path.is_file():
        result["reason"] = "protocol21_pipeline_manifest_missing"
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result["reason"] = "protocol21_pipeline_manifest_invalid"
        return result
    manifest_suite_sha = manifest.get("source_suite_sha256")
    result["manifest_source_suite_sha256"] = manifest_suite_sha
    result["source_suite_hash_matches"] = (
        manifest_suite_sha == result["source_suite_sha256"]
    )
    stages = {
        stage.get("name"): stage.get("return_code")
        for stage in manifest.get("stages", [])
        if isinstance(stage, dict) and stage.get("name")
    }
    result["required_stage_return_codes"] = {
        name: stages.get(name) for name in _REALM_REQUIRED_GATE_STAGES
    }
    failed_stages = [
        name
        for name in _REALM_REQUIRED_GATE_STAGES
        if stages.get(name) != 0
    ]
    if failed_stages:
        result["reason"] = "protocol21_row_gate_failed"
        result["failed_stages"] = failed_stages
        return result

    for filename in _REALM_GATE_ARTIFACTS.values():
        if not (evidence_dir / filename).is_file():
            result["missing_artifacts"].append(filename)
    if result["missing_artifacts"]:
        result["reason"] = "protocol21_gate_artifact_missing"
        return result
    try:
        refined = json.loads(
            (evidence_dir / _REALM_GATE_ARTIFACTS["materialize_core"]).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError):
        result["reason"] = "protocol21_materialization_invalid"
        return result
    selected_ids = [
        row.get("scenario_id")
        for row in refined.get("scenarios", [])
        if isinstance(row, dict) and row.get("scenario_id")
    ]
    result["selected_scenario_ids"] = selected_ids
    expected_ids = [
        row.get("scenario_id")
        for row in source_suite.get("scenarios", [])
        if isinstance(row, dict) and row.get("scenario_id")
    ]
    if refined.get("n_selected") != len(expected_ids) or refined.get("n_rejected") != 0:
        result["reason"] = "protocol21_materialization_identity_mismatch"
        return result
    expected_rows = {
        row.get("scenario_id"): row
        for row in source_suite.get("scenarios", [])
        if isinstance(row, dict) and row.get("scenario_id")
    }
    selected_rows = {
        row.get("scenario_id"): row
        for row in refined.get("scenarios", [])
        if isinstance(row, dict) and row.get("scenario_id")
    }
    immutable_keys = (
        "scenario_signature",
        "path",
        "source_denominator_key",
        "physical_source_key",
        "source_key",
        "case_ledger",
        "structural_fingerprint",
        "semantic_fingerprint",
    )
    if sorted(selected_ids) != sorted(expected_ids) or any(
        expected_rows[scenario_id].get(key)
        != selected_rows.get(scenario_id, {}).get(key)
        for scenario_id in expected_rows
        for key in immutable_keys
    ):
        result["reason"] = "protocol21_materialization_identity_mismatch"
        return result
    if not result["source_suite_hash_matches"]:
        result["metadata_only_drift"] = True
        result["reason"] = "protocol21_evidence_metadata_only_drift"
    else:
        result["reason"] = None
    result["status"] = "verified_row_gates"
    return result


def build_realm_pilot(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    source_lock: RealmSourceLock = DEFAULT_REALM_LOCK,
    provenance_root: str = "works/REALM-Bench-direct-pilot",
    scenario_path: Path = DEFAULT_STAGING_PATH,
    seed: int = 71,
    protocol21_evidence_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build one fail-closed REALM J2 pilot without writing any files."""
    raw_path = _locked_file(
        source_root,
        source_lock.raw_relative_path,
        source_lock.raw_sha256,
        "raw JSSP",
    )
    j2_path = _locked_file(
        source_root,
        source_lock.j2_relative_path,
        source_lock.j2_sha256,
        "J2 sidecar",
    )
    parsed = parse_jsplib_instance(raw_path)
    sidecar_row = _realm_sidecar_instance(
        j2_path, source_lock.disrupted_instance_id
    )
    if _normalized_sidecar_graph(sidecar_row) != parsed["jobs_detail"]:
        raise ValueError("REALM J2 operation graph mismatch with locked raw JSSP")
    if int(sidecar_row.get("num_jobs", 0)) != int(parsed["jobs"]):
        raise ValueError("REALM J2 job count mismatch with locked raw JSSP")
    if int(sidecar_row.get("num_machines", 0)) != int(parsed["machines"]):
        raise ValueError("REALM J2 machine count mismatch with locked raw JSSP")

    breakdown = _source_breakdown(sidecar_row)
    if breakdown["machine_id"] not in set(parsed["machine_ids"]):
        raise ValueError("REALM J2 breakdown machine is not in the source graph")
    operations = int(parsed["operations"])
    imperfection = DifficultyImperfectionProfile().for_level("high")
    fail_rate = float(imperfection["fail_rate"])
    delay_ticks = int(imperfection["delay_ticks"])
    failed_calls = math.ceil(operations * fail_rate / max(0.01, 1.0 - fail_rate))
    retry_allowance = failed_calls * 2
    horizon = max(
        10,
        operations + retry_allowance + delay_ticks + 2,
        breakdown["trigger_tick"] + breakdown["duration_ticks"] + 3,
    )
    target_job = f"j{max(0, int(parsed['jobs']) - 1)}"
    source_raw = f"{provenance_root}/{source_lock.raw_relative_path}"
    source_j2 = f"{provenance_root}/{source_lock.j2_relative_path}"
    instance_name = Path(source_lock.raw_relative_path).name
    source_identity = (
        f"realm_bench:{source_lock.commit}:{source_lock.raw_relative_path}:"
        f"{source_lock.raw_sha256}"
    )
    physical_source_key = f"realm_dmu:{source_lock.raw_sha256}"
    case_ledger = _realm_case_ledger(
        source_identity=source_identity,
        physical_source_key=physical_source_key,
        source_raw=source_raw,
        raw_sha256=source_lock.raw_sha256,
        source_j2=source_j2,
        j2_sha256=source_lock.j2_sha256,
        instance_id=source_lock.disrupted_instance_id,
        breakdown=breakdown,
        parsed=parsed,
        difficulty_mode="time_pressure",
        difficulty_level="high",
    )
    backend_config: dict[str, Any] = {
        "instance_name": instance_name,
        "source_integration_rung": "parsed_external_source_pilot",
        "release_ready": False,
        "release_reentry_ready": False,
        "source_denominator_key": source_identity,
        "physical_source_key": physical_source_key,
        "job_shop": parsed,
        "reference": {"type": "unlocked_reference"},
        "expected_sha256": source_lock.raw_sha256,
        "actual_sha256": source_lock.raw_sha256,
        "external_source_assets": {
            "raw_jssp": {
                "path": source_raw,
                "sha256": source_lock.raw_sha256,
                "runtime_consumed": True,
            },
            "j2_event_sidecar": {
                "path": source_j2,
                "sha256": source_lock.j2_sha256,
                "git_commit": source_lock.commit,
                "selected_instance_id": source_lock.disrupted_instance_id,
                "runtime_consumed": True,
                "converter_consumed": True,
            },
        },
        "source_axes": {
            "benchmark_tier": "J2",
            "benchmark_instance_id": source_lock.disrupted_instance_id,
            "jobs": parsed["jobs"],
            "machines": parsed["machines"],
            "operations": parsed["operations"],
            "source_event_type": "machine_breakdown",
        },
        "logistics_key_aliases": {
            "aggregate_demand_mw": "operations_total",
            "aggregate_generation_mw": "operations_scheduled",
            "balance_error_mw": "operations_unscheduled",
            "production_cost": "makespan",
            "shed_penalty": "unscheduled_operation_penalty",
            "rho_max": "machine_queue_pressure",
            "n_overloads": "machine_conflict_count",
        },
        "honest_zero_keys": [
            "reserves_required_mw",
            "reserves_procured_mw",
            "n_voltage_violations",
            "n_disconnected_lines",
        ],
        "dimension_applicability": {
            "optimality_gap": {
                "applicable": False,
                "reason": "no_commit_locked_makespan_optimum_for_selected_REALM_row",
            },
            "safety_violation": {
                "applicable": True,
                "reason": "precedence_machine_capacity_and_outage_constraints",
            },
            "counterfactual_prevention": {
                "applicable": True,
                "reason": "deterministic_machine_breakdown_recovery_replay",
            },
            "weighted_equity_score": {
                "applicable": False,
                "reason": "source_instance_has_no_stakeholder_priority_classes",
            },
            "ethical_quality": {
                "applicable": False,
                "reason": "source_instance_has_no_ethical_dilemma_payload",
            },
            "stakeholder_management": {
                "applicable": False,
                "reason": "source_instance_has_no_stakeholder_trust_model",
            },
        },
        "interaction_budget_basis": {
            "operations": operations,
            "tool_fail_rate": fail_rate,
            "tool_delay_ticks": delay_ticks,
            "retry_allowance_ticks": retry_allowance,
            "cooldown_after_failure_ticks": 1,
        },
        "dynamic_job_shop": {
            "enabled": True,
            "event_source": "realm_j2_plus_explicit_procedural_overlay_v1",
            "max_dispatch_batch_size": 4,
            "recovery_clearance_ticks": 1,
            "source_observed_events": True,
        },
        "task_contract": {
            "contract": "logistics.job_shop.dynamic_recovery.v1",
            "event_response_window": {
                "first_tick": breakdown["trigger_tick"] + 1,
                "last_tick": horizon - 1,
            },
            "native_controls": [
                "dispatch_ready_operations",
                "dispatch_job_operation",
                "repair_machine",
            ],
        },
        "task_requirements": {
            "min_distinct_control_ticks": 3,
            "min_distinct_physical_tools": 2,
            "min_plan_reversals": 1,
            "ordered_tool_milestones": [
                {
                    "tool": "dispatch_ready_operations",
                    "not_after_tick": breakdown["trigger_tick"],
                },
                {
                    "tool": "repair_machine",
                    "not_before_tick": breakdown["trigger_tick"] + 1,
                    "not_after_tick": horizon - 1,
                },
                {
                    "tool": "dispatch_ready_operations",
                    "not_before_tick": breakdown["trigger_tick"] + 3,
                    "not_after_tick": horizon - 1,
                },
            ],
        },
        "source_event_contract": {
            "source_observed": True,
            "source_sidecar_runtime_consumed": True,
            "source_sidecar_converter_consumed": True,
            "sidecar_path": source_j2,
            "sidecar_sha256": source_lock.j2_sha256,
            "selected_instance_id": source_lock.disrupted_instance_id,
            "runtime_effect_required": True,
            "procedural_overlay": True,
            "procedural_event_source_observed": False,
        },
    }
    perturbations = [
        Perturbation(
            kind="machine_breakdown",
            trigger_tick=breakdown["trigger_tick"],
            duration_ticks=breakdown["duration_ticks"],
            hidden=False,
            target={
                "machine_id": breakdown["machine_id"],
                "source_observed": True,
                "source_instance_id": source_lock.disrupted_instance_id,
                "source_sidecar_sha256": source_lock.j2_sha256,
            },
            intensity=1.0,
            notes=(
                "Source-observed REALM J2 machine breakdown, converted from "
                "the locked 1-based sidecar ID to the native 0-based ID."
            ),
        ),
        Perturbation(
            kind="demand_surge",
            trigger_tick=min(horizon - 2, breakdown["trigger_tick"] + 2),
            duration_ticks=4,
            hidden=True,
            target={
                "job_id": target_job,
                "source_observed": False,
                "source": "locked_REALM_job_set",
            },
            intensity=1.5,
            notes=(
                "Deterministic procedural urgency overlay; explicitly not a "
                "REALM source-observed event or provenance evidence."
            ),
        ),
    ]
    provenance = Provenance(
        data_source="realm_bench_j2_dmu",
        files=[source_raw, source_j2],
        commit=source_lock.commit,
        url=(
            "https://github.com/genglongling/REALM-Bench/tree/"
            f"{source_lock.commit}"
        ),
        lock_strategy="git_commit+file_sha256+selected_row_id",
        time_window={
            "objective": "minimize_makespan_with_adaptation",
            "source_event_tick": breakdown["trigger_tick"],
            "source_event_duration": breakdown["duration_ticks"],
        },
        license=(
            "REALM README declares JSSP dataset CC-BY-4.0; upstream DMU/"
            "OR-Library instance terms remain subject to independent review"
        ),
        notes=(
            "The raw DMU graph and selected J2 sidecar row are opened by the "
            "native backend. Runtime admission is proven for the locked graph "
            "and source-observed breakdown; Core admission remains held pending "
            "the isolated protocol gates and upstream DMU license review."
        ),
    )
    seed_id = (
        "logistics/job_shop_dispatch/time_pressure/high/"
        f"realm_{Path(instance_name).stem}_dynamic_high_s{seed}"
    )
    seed_obj = LogisticsScenarioSeed(
        seed_id=seed_id,
        family="job_shop_dispatch",
        backend_kind="jsplib_job_shop",
        backend_config=backend_config,
        horizon_ticks=horizon,
        tick_minutes=1,
        seed=seed,
        perturbations=perturbations,
        difficulty_mode="time_pressure",
        difficulty_level="high",
        provenance=provenance,
    )
    scenario = seed_obj.to_dict()
    scenario["scenario_id"] = seed_id
    scenario["complexity_metrics"] = {
        **seed_obj.complexity_metrics(),
        "n_jobs": parsed["jobs"],
        "n_machines": parsed["machines"],
        "n_operations": operations,
        "source_observed_event_types": ["machine_breakdown"],
        "procedural_event_types": ["demand_surge"],
    }
    source_contract = jsplib_job_shop(scenario, REPO_ROOT)
    source_contract["runtime_input"] = [source_raw, source_j2]
    source_contract["metadata"] = []
    scenario["source_contract"] = source_contract
    scenario["case_ledger"] = case_ledger
    scenario["scenario_signature"] = recompute_signature_with_seed(scenario, seed)
    row = {
        "scenario_id": seed_id,
        "scenario_signature": scenario["scenario_signature"],
        "path": _scenario_reference(scenario_path),
        "seed": seed,
        "horizon_ticks": horizon,
        "family": scenario["family"],
        "domain": scenario["domain"],
        "backend_kind": scenario["backend_kind"],
        "difficulty_level": scenario["difficulty_level"],
        "difficulty_mode": scenario["difficulty_mode"],
        "source_denominator_key": source_identity,
        "physical_source_key": physical_source_key,
        "source_key": json.dumps(
            {
                "backend": "jsplib_job_shop",
                "commit": source_lock.commit,
                "raw_sha256": source_lock.raw_sha256,
                "sidecar_sha256": source_lock.j2_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "structural_fingerprint": structural_fingerprint(scenario),
        "semantic_fingerprint": _semantic_fingerprint(scenario),
        "case_ledger": case_ledger,
        "status": "held_repair",
        "reason_codes": [
            "external_source_pilot",
            "upstream_dmu_license_chain_review_pending",
            "protocol21_full_gates_not_run",
        ],
    }
    suite = {
        "schema_version": "protocol21-external-realm-direct-pilot-v1",
        "status": "working_set",
        "selection_policy": "quality_maximal_v1",
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": 1,
        "constraints": {
            "candidate_evidence_merge_only": True,
            "candidate_replacements_staging_only": True,
            "formal_evaluation_ready": False,
            "model_outcomes_used_for_filtering": False,
            "one_per_effective_source_identity": True,
            "preserve_each_eligible_family_difficulty_cell": True,
            "quality_maximal_selection": True,
        },
        "scenarios": [row],
    }
    if protocol21_evidence_dir is None and source_root.resolve() == DEFAULT_SOURCE_ROOT.resolve():
        protocol21_evidence_dir = DEFAULT_OUTPUT_DIR / "full_pipeline_v2"
    gate_evidence = _inspect_realm_protocol21_evidence(
        evidence_dir=protocol21_evidence_dir,
        source_suite=suite,
    )
    if gate_evidence["status"] == "verified_row_gates":
        row["reason_codes"].remove("protocol21_full_gates_not_run")
        # Rebind the report to the emitted suite.  The isolated run predates
        # this metadata-only reason-code cleanup, so the helper records that
        # drift explicitly while rechecking immutable row identity.
        gate_evidence = _inspect_realm_protocol21_evidence(
            evidence_dir=protocol21_evidence_dir,
            source_suite=suite,
        )
    report = {
        "schema_version": "external-realm-j2-direct-pilot-report-v1",
        "status": "held_repair",
        "core_admission": False,
        "direct_asset_status": "runtime_consumption_proven",
        "source_lock": {
            "commit": source_lock.commit,
            "raw_path": source_raw,
            "raw_sha256": source_lock.raw_sha256,
            "j2_path": source_j2,
            "j2_sha256": source_lock.j2_sha256,
            "selected_instance_id": source_lock.disrupted_instance_id,
        },
        "verified_conversion": {
            "raw_graph_matches_j2": True,
            "machine_id_transform": "REALM_1_based_to_backend_0_based",
            "jobs": parsed["jobs"],
            "machines": parsed["machines"],
            "operations": parsed["operations"],
            "source_breakdown": breakdown,
        },
        "protocol21_gate_evidence": gate_evidence,
        "blockers": [
            "upstream_dmu_license_chain_review_pending",
        ]
        + (
            []
            if gate_evidence["status"] == "verified_row_gates"
            else ["protocol21_full_gates_not_run"]
        ),
        "next_required_action": (
            "Resolve the upstream DMU license chain before any release "
            "materialization."
            if gate_evidence["status"] == "verified_row_gates"
            else "Run the isolated protocol-2.1 replay and behavioral gates "
            "against the runtime-consumed J2 sidecar, then resolve the "
            "upstream DMU license chain before any release materialization."
        ),
    }
    return report, scenario, suite


def build_external_feasibility_catalog(
    *, realm_gate_evidence: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return primary-source-locked feasibility findings for six benchmarks."""
    realm_gate_verified = bool(
        realm_gate_evidence
        and realm_gate_evidence.get("status") == "verified_row_gates"
    )
    realm_blockers = ["upstream_dmu_license_chain_review_pending"]
    if not realm_gate_verified:
        realm_blockers.append("protocol21_full_gates_not_run")
    return {
        "schema_version": "external-direct-feasibility-v2",
        "review_date": "2026-08-08",
        "admission_rule": (
            "A paper/task corpus is not a simulator episode. Direct admission "
            "requires a locked instance, native runtime consumption, native "
            "state-changing control, time evolution, replay, and evidence."
        ),
        "sources": [
            {
                "source_id": "dynaschedbench",
                "title": "DynaSchedBench",
                "paper": "https://arxiv.org/abs/2605.27566",
                "repository": "https://github.com/dsbx7/DynaSchedBench",
                "commit": "08975bf4a0473c5dff9177393bc6743db9ddc946",
                "license": "Apache-2.0",
                "has_released_instance_assets": True,
                "published_scenario_bundles_observed": 2899,
                "selected_asset_hashes": {
                    "events.jsonl": "86133e09d45913a34d2f4eddfd3479fd2567f9d795644035294db349e6b95253",
                    "input_model.json": "1368dca31791bfdc5ea2ed3a74a8a18d79a54e7f77fe620a7f0b5441850090f9",
                    "static_jobs.json": "dde3f57c092735220d18de833b347e550f2581065188423beb93abfc80fcceed",
                    "static_machines.json": "f53c074032a49366bbe7d0189b584640dc7564a1dd9278d4298109c611b5fe21",
                },
                "direct_conversion_allowed": False,
                "disposition": "implement_next",
                "exact_blockers": [
                    "existing_jsplib_backend_lacks_machine_group_eligibility",
                    "existing_jsplib_backend_lacks_continuous_arrival_semantics",
                    "native_dynasched_runtime_evidence_adapter_missing",
                ],
                "recommendation": "Implement a native flexible/dynamic job-shop adapter; do not collapse alternatives to one machine.",
            },
            {
                "source_id": "oragentbench",
                "title": "ORAgentBench",
                "paper": "https://arxiv.org/abs/2606.19787",
                "repository": "https://github.com/ORAgentBench/ORAgentBench",
                "commit": "c9eb952435a4352f33daa2a35efe0f8c76d31b28",
                "license": "MIT code; per-task data lineage is mixed",
                "has_released_instance_assets": True,
                "observed_task_files": 1196,
                "direct_conversion_allowed": False,
                "disposition": "method_only",
                "exact_blockers": [
                    "offline_modeling_workflow_not_closed_loop_runtime",
                    "tasks_mix_public_and_synthetic_inputs",
                    "per_task_native_source_identity_not_locked",
                ],
                "recommendation": "Reuse task-decomposition and verifier patterns only after binding a separately locked native DT-Sched backend.",
            },
            {
                "source_id": "elecbench",
                "title": "ElecBench",
                "paper": "https://arxiv.org/abs/2407.05365",
                "repository": "https://github.com/xiyuan-zhou/ElecBench-a-Power-Dispatch-Evaluation-Benchmark-for-Large-Language-Models",
                "commit": "a6fc8f65c75388b79df8efd5f2f51c09b06cafa8",
                "license": "no repository LICENSE observed at locked commit",
                "has_released_instance_assets": True,
                "asset_kind": "dispatch/monitoring/blackstart/general QA JSONL",
                "direct_conversion_allowed": False,
                "disposition": "method_only",
                "exact_blockers": [
                    "qa_corpus_not_simulator_state",
                    "no_native_state_changing_control_surface",
                    "repository_license_missing",
                ],
                "recommendation": "Use only as a reviewed vocabulary/constraint-template source; never relabel QA as grid simulation.",
            },
            {
                "source_id": "realm_bench",
                "title": "REALM-Bench",
                "paper": "https://arxiv.org/abs/2502.18836",
                "repository": "https://github.com/genglongling/REALM-Bench",
                "commit": REALM_COMMIT,
                "license": "MIT code; README declares JSSP data CC-BY-4.0; upstream terms apply",
                "has_released_instance_assets": True,
                "direct_conversion_allowed": True,
                "disposition": "pilot_current_backend",
                "pilot_source": {
                    "raw_sha256": DEFAULT_REALM_LOCK.raw_sha256,
                    "j2_sha256": DEFAULT_REALM_LOCK.j2_sha256,
                    "selected_instance_id": DEFAULT_REALM_LOCK.disrupted_instance_id,
                },
                "exact_blockers": realm_blockers,
                "pilot_gate_status": (
                    "isolated_row_gates_verified"
                    if realm_gate_verified
                    else "isolated_row_gates_not_verified"
                ),
                "recommendation": (
                    "Keep the pilot isolated until the upstream DMU license chain "
                    "is resolved; the isolated row-level Protocol-2.1 gates are "
                    "already verified."
                    if realm_gate_verified
                    else "Run the isolated Protocol-2.1 row gates and resolve "
                    "the upstream DMU license chain before any release "
                    "materialization."
                ),
            },
            {
                "source_id": "frontier_engineering",
                "title": "Frontier-Engineering",
                "paper": "https://arxiv.org/abs/2604.12290",
                "repository": "https://github.com/EinsiaLab/Frontier-Engineering",
                "commit": "e3fa29c193356af2ce1ec8b3d23ab1a2e2410071",
                "license": "no root repository LICENSE observed at locked commit",
                "has_released_instance_assets": True,
                "jobshop_manifest_sha256": "17d12fcb317057188a00deab413b2b321b0ab2d9b2526ae8c9676e33b6b4d0a4",
                "jobshop_rows_observed": 162,
                "jobshop_rows_matching_local_jsplib": 161,
                "independent_jobshop_sources": 0,
                "direct_conversion_allowed": False,
                "disposition": "pilot_only",
                "exact_blockers": [
                    "jobshop_assets_duplicate_existing_jsplib_sources",
                    "root_license_missing",
                    "ev2gym_requires_new_native_adapter_and_domain_contract",
                ],
                "recommendation": "Treat JobShop as duplicate; consider EV2Gym only as a separately licensed EV-charging pilot.",
            },
            {
                "source_id": "edgebench",
                "title": "EdgeBench",
                "paper": "https://arxiv.org/abs/2607.05155",
                "repository": "https://github.com/ByteDance-Seed/EdgeBench",
                "commit": "a87350ab80eeb320b13cb71d1b0c3ffcc20a670f",
                "license": "Apache-2.0 code; README declares tasks CC-BY-4.0",
                "has_released_instance_assets": True,
                "open_tasks_observed": 51,
                "direct_conversion_allowed": False,
                "disposition": "diagnostic_only",
                "exact_blockers": [
                    "no_native_scheduling_instance_for_current_backends",
                    "no_simulator_owned_clock_or_counterfactual_replay_contract",
                ],
                "recommendation": "Reuse long-horizon harness diagnostics only; do not admit tasks as scheduling episodes.",
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--staging-path", type=Path, default=DEFAULT_STAGING_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--protocol21-evidence-dir",
        type=Path,
        default=None,
        help="Optional isolated Protocol-2.1 evidence directory to bind to the report",
    )
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        report, scenario, suite = build_realm_pilot(
            source_root=args.source_root.resolve(),
            scenario_path=args.staging_path.resolve(),
            seed=args.seed,
            protocol21_evidence_dir=(
                args.protocol21_evidence_dir.resolve()
                if args.protocol21_evidence_dir is not None
                else None
            ),
        )
        catalog = build_external_feasibility_catalog(
            realm_gate_evidence=report["protocol21_gate_evidence"]
        )
        if args.execute:
            args.staging_path.parent.mkdir(parents=True, exist_ok=True)
            args.staging_path.write_text(
                yaml.safe_dump(scenario, sort_keys=False), encoding="utf-8"
            )
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for name, payload in (
                ("source_suite.json", suite),
                ("conversion_report.json", report),
                ("feasibility_catalog.json", catalog),
            ):
                (args.output_dir / name).write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "scenario_id": scenario["scenario_id"],
                "status": report["status"],
                "core_admission": report["core_admission"],
                "blockers": report["blockers"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
