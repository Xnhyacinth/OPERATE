from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.implementation_identity import implementation_identity
from evaluation.operational_agency import evaluate_operational_agency
from scripts import batch_llm_eval as mod

REPR_SMOKE_NON_G2O = {
    "power_grid/daily_ops_24h/time_pressure/basic/do_20200127_time_pressure_basic_s42",
    "power_grid/daily_ops_real_forecast_24h/time_pressure/medium/drf_20200127_time_pressure_medium_s42",
    "power_grid/distribution_volt_var/deep_planning/basic/dvv_mv_with_der_all_deep_planning_basic_s42",
}

REPR_FULL_NON_G2O = {
    "power_grid/daily_ops_24h/time_pressure/basic/do_20200127_time_pressure_basic_s42",
    "power_grid/daily_ops_real_forecast_24h/time_pressure/medium/drf_20200127_time_pressure_medium_s42",
    "power_grid/critical_winter_peak/time_pressure/medium/wp_20201125_time_pressure_medium_s42",
    "power_grid/reserve_stress_24h/time_pressure/medium/rs_20140901_r0_time_pressure_medium_s42",
    "power_grid/wind_uncertainty_24h/time_pressure/medium/wu_20150101_hw_time_pressure_medium_s42",
    "power_grid/distribution_volt_var/deep_planning/basic/dvv_mv_with_der_all_deep_planning_basic_s42",
}

REPR_FULL_STORM = {
    "power_grid/storm_emergency_6h/time_pressure/medium/st_chron0_time_pressure_medium_s42",
    "power_grid/storm_emergency_6h/time_pressure/medium/st_chron1_time_pressure_medium_s42",
    "power_grid/storm_emergency_6h/time_pressure/medium/st_chron2_time_pressure_medium_s42",
    "power_grid/storm_emergency_6h/time_pressure/medium/st_chron3_time_pressure_medium_s42",
    "power_grid/storm_emergency_6h/time_pressure/medium/st_chron4_time_pressure_medium_s42",
}


def _provider_audit_artifact() -> dict[str, Any]:
    return {
        "schema_version": "provider_interaction_audit_v1",
        "sha256": "a" * 64,
        "event_count": 2,
    }


def _bind_formal_artifacts(
    row: dict[str, Any], tmp_path: Path, *, key: str
) -> dict[str, Any]:
    source_key = str(row.setdefault("source_denominator_key", f"fixture:{key}"))
    row.setdefault(
        "case_ledger",
        {"schema_version": "0.1", "source_denominator_key": source_key},
    )
    row.setdefault("evaluation_protocol", {}).setdefault(
        "construct_contract", "operational_agency.v1"
    )
    summary = row.setdefault("trajectory_summary", {})
    summary.setdefault("event_response_records", [])
    summary.setdefault("operational_agency_valid_evidence_ids", [])
    summary.setdefault(
        "operational_agency_profile",
        evaluate_operational_agency(
            [], valid_evidence_ids=set(), masked_replay_by_call_id={}
        ),
    )
    row.setdefault(
        "counterfactual",
        {
            "per_action": [],
            "per_action_capped": False,
            "per_action_status": "complete",
            "per_action_expected": 0,
            "per_action_attempted": 0,
            "per_action_completed": 0,
            "per_action_failures": [],
            "per_action_groups": [],
            "per_action_group_status": "complete",
            "per_action_group_expected": 0,
            "per_action_group_attempted": 0,
            "per_action_group_completed": 0,
            "per_action_group_failures": [],
        },
    )
    treatment_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    artifact_dir = tmp_path / f"treatment-{treatment_hash}"
    artifact_dir.mkdir(exist_ok=True)
    prefix = artifact_dir / key.replace("/", "_")
    provider_payload = (
        b'{"record_kind":"provider_request"}\n{"record_kind":"provider_response"}\n'
    )
    semantic_payload = b'{"role":"system","content":"contract"}\n'
    trajectory_payload = b'{"tick":0,"evidence_ids":["ev-1"]}\n'
    evidence_payload = b'{"evidence_id":"ev-1","tick":0}\n'
    provider_path = Path(f"{prefix}.provider_audit.jsonl")
    semantic_path = Path(f"{prefix}.semantic_ledger.jsonl")
    trajectory_path = Path(f"{prefix}.trajectory.jsonl")
    evidence_path = Path(f"{prefix}.evidence.jsonl")
    provider_path.write_bytes(provider_payload)
    semantic_path.write_bytes(semantic_payload)
    trajectory_path.write_bytes(trajectory_payload)
    evidence_path.write_bytes(evidence_payload)
    row["agent_treatment_sha256"] = treatment_hash
    llm = summary.setdefault("llm", {})
    if "provider_model_identity_records" not in llm:
        model = str(row.get("model") or "")
        llm.update(
            {
                "provider_models": [model],
                "provider_model_identity_records": [
                    {
                        "schema_version": "provider_model_identity_closure_v1",
                        "request_sequence": 1,
                        "requested_model": model,
                        "observed_models": [model],
                        "closure": "exact",
                    }
                ],
                "provider_model_identity_request_count": 1,
                "provider_model_identity_closed_count": 1,
                "provider_model_identity_exact_count": 1,
                "provider_model_identity_missing_count": 0,
                "provider_model_identity_mismatch_count": 0,
                "provider_model_identity_failed_request_count": 0,
                "provider_request_count": 1,
                "provider_response_count": 1,
            }
        )
    summary["trajectory_path"] = str(prefix)
    summary["provider_audit_artifact"] = {
        "schema_version": "provider_interaction_audit_v1",
        "path": str(provider_path),
        "sha256": hashlib.sha256(provider_payload).hexdigest(),
        "event_count": 2,
    }
    summary["semantic_ledger_artifact"] = {
        "schema_version": "semantic_session_ledger_v1",
        "path": str(semantic_path),
        "sha256": hashlib.sha256(semantic_payload).hexdigest(),
        "event_count": 1,
    }
    summary["trajectory_artifact"] = {
        "schema_version": "episode_trajectory_jsonl_v1",
        "path": str(trajectory_path),
        "sha256": hashlib.sha256(trajectory_payload).hexdigest(),
        "event_count": 1,
        "byte_count": len(trajectory_payload),
    }
    summary["evidence_ledger_artifact"] = {
        "schema_version": "evidence_ledger_jsonl_v1",
        "path": str(evidence_path),
        "sha256": hashlib.sha256(evidence_payload).hexdigest(),
        "event_count": 1,
        "byte_count": len(evidence_payload),
    }
    return row


def _tool_surface_contract() -> dict[str, Any]:
    return {
        "schema_version": "tool-surface-contract-v1",
        "backend_kind": "fixture",
        "complete": True,
        "exposed_tool_names": ["wait"],
        "declared_observation_tool_names": [],
        "declared_control_tool_names": [],
        "effective_commit_tool_names": [],
        "missing_observation_tool_names": [],
        "missing_control_tool_names": [],
        "missing_commit_control_tool_names": [],
        "exposed_undeclared_tool_names": [],
        "exposed_schema_sha256": "c" * 64,
    }


def _formally_eligible_protocol21_row() -> dict[str, Any]:
    suite_eligibility = {
        "suite_blocked": False,
        "diagnostic_cells": [],
        "uninformative_cells": [],
        "wait_dominant_cells": [],
    }
    structured_memory = {
        "schema_version": "persistent_working_memory_v2",
        "unresolved_alarms": [],
        "open_obligations": [],
        "confirmed_facts": [],
        "active_commitments": [],
        "forecast_ledger": [],
        "state_trends": [],
        "last_updated_tick": 0,
    }
    structured_memory_bytes = len(
        json.dumps(
            structured_memory,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "status": "ok",
        "model": "requested-model",
        "interaction_mode": "logical_stateless",
        "agent_config": {"config": {"persistent_memory_max_items": 8}},
        "structured_memory": structured_memory,
        "evaluation_protocol": {
            "version": mod.EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": (mod.EVALUATION_IMPLEMENTATION_FINGERPRINT),
            "construct_contract": "operational_agency.v1",
        },
        "score": {"scoring_version": mod.SCORING_VERSION},
        "source_denominator_key": "fixture:source-a",
        "case_ledger": {
            "schema_version": "0.1",
            "source_denominator_key": "fixture:source-a",
        },
        "suite_manifest_sha256": "suite-a",
        "suite_eligibility": suite_eligibility,
        "suite_eligibility_sha256": mod._canonical_json_sha256(suite_eligibility),
        "trajectory_summary": {
            "llm": {
                "provider_models": ["requested-model"],
                "provider_model_identity_records": [
                    {
                        "schema_version": "provider_model_identity_closure_v1",
                        "request_sequence": 1,
                        "requested_model": "requested-model",
                        "observed_models": ["requested-model"],
                        "closure": "exact",
                    }
                ],
                "provider_model_identity_request_count": 1,
                "provider_model_identity_closed_count": 1,
                "provider_model_identity_exact_count": 1,
                "provider_model_identity_missing_count": 0,
                "provider_model_identity_mismatch_count": 0,
                "provider_model_identity_failed_request_count": 0,
                "provider_request_count": 1,
                "provider_response_count": 1,
                "session_ledger_events": 1,
                "structured_memory_bytes": structured_memory_bytes,
            },
            "structured_memory": structured_memory,
            "semantic_ledger_artifact": {
                "schema_version": "semantic_session_ledger_v1",
                "sha256": "b" * 64,
                "event_count": 1,
            },
            "trajectory_artifact": {
                "schema_version": "episode_trajectory_jsonl_v1",
                "path": "/tmp/fixture.trajectory.jsonl",
                "sha256": "d" * 64,
                "event_count": 1,
                "byte_count": 1,
            },
            "evidence_ledger_artifact": {
                "schema_version": "evidence_ledger_jsonl_v1",
                "path": "/tmp/fixture.evidence.jsonl",
                "sha256": "e" * 64,
                "event_count": 1,
                "byte_count": 1,
            },
            "terminal_integrity": {"release_ready": True},
            "event_contract": {"schema_version": "1.0", "violation_count": 0},
            "provider_audit_artifact": _provider_audit_artifact(),
            "tool_semantic_coverage": {
                "covered": True,
                "registered_tool_names": ["wait"],
                "unknown_tool_names": [],
                "unclassified_tool_names": [],
                "explicit_semantic_roles_complete": True,
                "missing_explicit_semantic_role_names": [],
                "native_targets_complete": True,
                "missing_native_target_kind_names": [],
                "state_changing_actuators_complete": True,
                "missing_actuator_family_names": [],
            },
            "tool_surface_contract": {
                "schema_version": "tool-surface-contract-v1",
                "backend_kind": "fixture",
                "complete": True,
                "exposed_tool_names": ["wait"],
                "declared_observation_tool_names": [],
                "declared_control_tool_names": [],
                "effective_commit_tool_names": [],
                "missing_observation_tool_names": [],
                "missing_control_tool_names": [],
                "missing_commit_control_tool_names": [],
                "exposed_undeclared_tool_names": [],
                "exposed_schema_sha256": "c" * 64,
            },
            "event_response_records": [],
            "operational_agency_valid_evidence_ids": [],
            "operational_agency_profile": evaluate_operational_agency(
                [],
                valid_evidence_ids=set(),
                masked_replay_by_call_id={},
            ),
        },
        "counterfactual": {
            "per_action": [],
            "per_action_capped": False,
            "per_action_status": "complete",
            "per_action_expected": 0,
            "per_action_attempted": 0,
            "per_action_completed": 0,
            "per_action_failures": [],
            "per_action_groups": [],
            "per_action_group_status": "complete",
            "per_action_group_expected": 0,
            "per_action_group_attempted": 0,
            "per_action_group_completed": 0,
            "per_action_group_failures": [],
        },
    }


def _green_formal_readiness(root: Path) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "status": "formal_evaluation_ready",
        "formal_evaluation_ready": True,
        "formal_run_blockers": [],
        "scoring_version": mod.SCORING_VERSION,
        "primary_leaderboard_formula_version": (
            mod.PRIMARY_LEADERBOARD_FORMULA_VERSION
        ),
        "primary_inference_version": mod.PRIMARY_INFERENCE_VERSION,
        "implementation_tree_sha256": implementation_identity(root)[
            "implementation_tree_sha256"
        ],
        "suite_manifest_sha256": "a" * 64,
        "formal_run_contract": {
            "contract_version": "agentic_persistent.v1",
            "required_construct_contract": "operational_agency.v1",
            "wakeup_policy": dict(mod.CANONICAL_WAKEUP_POLICY),
        },
        "n_scenarios": 1,
        "scenarios": [
            {
                "scenario_id": "fixture",
                "construct_contract": "operational_agency.v1",
            }
        ],
    }


def _agentic_formal_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "scenario_slice": "manifest_operate_v0_58_0",
        "formal_manifest_bound": True,
        "models": ["a"],
        "pass_k": 1,
        "max_workers": 4,
        "temperature": 0.0,
        "prompt_mode": "strict",
        "interaction_mode": "logical_persistent",
        "seed_mode": "scenario",
        "scheduler_mode": "global",
        "save_trajectories": True,
        "finalize": True,
        "allow_blocked_suite": False,
        "diagnostic_only": False,
        "git_metadata_available": True,
        "git_dirty": False,
        "provider_rpm_limit": 20,
        "provider_rpd_limit": 1_000,
        "provider_rate_limit_scope": "formal-provider-quota",
    }
    config.update(overrides)
    return config


def _agentic_formal_readiness(**overrides: Any) -> dict[str, Any]:
    readiness = {
        "formal_evaluation_ready": True,
        "suite_manifest_sha256": "suite",
        "scoring_version": mod.SCORING_VERSION,
        "primary_leaderboard_formula_version": (
            mod.PRIMARY_LEADERBOARD_FORMULA_VERSION
        ),
        "primary_inference_version": mod.PRIMARY_INFERENCE_VERSION,
        "task_completion_input_unit": mod.TASK_COMPLETION_INPUT_UNIT,
        "task_completion_score_unit": mod.TASK_COMPLETION_SCORE_UNIT,
        "weighted_equity_formula_version": (mod.WEIGHTED_EQUITY_FORMULA_VERSION),
        "formal_run_contract": {
            "contract_version": "agentic_persistent.v1",
            "required_model_count_per_shard": 1,
            "minimum_pass_k": 1,
            "minimum_max_workers": 1,
            "maximum_max_workers": 32,
            "required_temperature": 0.0,
            "required_interaction_mode": "logical_persistent",
            "wakeup_policy": dict(mod.CANONICAL_WAKEUP_POLICY),
        },
    }
    readiness.update(overrides)
    return readiness


def _write_green_formal_gate(root: Path) -> dict[str, Path]:
    manifest_dir = root / "release" / "vnext"
    readiness_dir = manifest_dir / "run"
    readiness_dir.mkdir(parents=True)
    backend_root = root / "data_operate_v058" / "backends" / "fixture"
    backend_root.mkdir(parents=True)
    backend_file = backend_root / "runtime.bin"
    backend_file.write_bytes(b"locked runtime\n")
    works_root = root / "works"
    works_root.mkdir()
    (works_root / "Fixture").symlink_to(backend_root)
    source_suite = manifest_dir / "protocol21_source_suite.json"
    source_suite.write_text(json.dumps({"n_scenarios": 1}), encoding="utf-8")
    source_digest = hashlib.sha256(source_suite.read_bytes()).hexdigest()
    readiness = readiness_dir / "protocol2_v21_core_readiness.json"
    readiness.write_text(
        json.dumps(_green_formal_readiness(root)),
        encoding="utf-8",
    )
    identity = implementation_identity(root)
    live_tree = identity["implementation_tree_sha256"]
    pipeline_tree = identity["core_release_pipeline_sha256"]
    stage_artifacts: dict[str, dict[str, str]] = {}
    pipeline_stages = []
    for stage_name in mod.FORMAL_CORE_PIPELINE_STAGES:
        stage_path = (
            readiness
            if stage_name == "readiness"
            else readiness_dir / mod.FORMAL_CORE_STAGE_FILES[stage_name]
        )
        if stage_name != "readiness":
            stage_path.write_text(
                json.dumps(
                    {
                        "stage": stage_name,
                        "core_release_pipeline_sha256": pipeline_tree,
                    }
                ),
                encoding="utf-8",
            )
        else:
            payload = json.loads(stage_path.read_text(encoding="utf-8"))
            payload["core_release_pipeline_sha256"] = pipeline_tree
            stage_path.write_text(json.dumps(payload), encoding="utf-8")
        stage_digest = hashlib.sha256(stage_path.read_bytes()).hexdigest()
        stage_artifacts[stage_name] = {
            "relative_path": stage_path.name,
            "sha256": stage_digest,
        }
        pipeline_stages.append(
            {
                "name": stage_name,
                "output_sha256": stage_digest,
                "return_code": 0,
                "implementation_tree_sha256": live_tree,
                "core_release_pipeline_sha256": pipeline_tree,
            }
        )
    readiness_digest = hashlib.sha256(readiness.read_bytes()).hexdigest()
    pipeline_manifest = readiness_dir / "protocol2_v21_pipeline_manifest.json"
    pipeline_manifest.write_text(
        json.dumps(
            {
                "status": "formal_evaluation_ready",
                "implementation_tree_sha256": live_tree,
                "core_release_pipeline_sha256": pipeline_tree,
                "source_suite_sha256": source_digest,
                "stages": pipeline_stages,
            }
        ),
        encoding="utf-8",
    )
    runtime_closure = manifest_dir / "backend_runtime_closure.json"
    closure_payload = {
        "schema_version": "operate-backend-runtime-closure-v1",
        "release_id": "vnext",
        "status": "backend_runtime_closure_complete",
        "terminal": True,
        "portable": True,
        "source_suite_sha256": source_digest,
        "archived_files": {
            "backends/fixture/runtime.bin": {
                "source_path": "works/Fixture/runtime.bin",
                "sha256": hashlib.sha256(backend_file.read_bytes()).hexdigest(),
                "roles": ["runtime_input"],
                "backend_kinds": ["fixture"],
            }
        },
        "repo_tracked_files": {},
        "separately_bundled_files": {},
        "external_sources": {},
        "backend_links": {"Fixture": "fixture"},
        "runtime_packages": {},
        "summary": {
            "n_archived_files": 1,
            "n_backend_links": 1,
            "n_external_sources": 0,
            "n_repo_tracked_files": 0,
            "n_runtime_packages": 0,
            "n_separately_bundled_files": 0,
            "n_source_assets": 1,
            "n_unresolved": 0,
            "n_virtual_sources": 0,
        },
    }
    closure_payload["identity_sha256"] = hashlib.sha256(
        json.dumps(
            closure_payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    runtime_closure.write_text(json.dumps(closure_payload), encoding="utf-8")
    runtime_closure_digest = hashlib.sha256(runtime_closure.read_bytes()).hexdigest()
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "release_id": "vnext",
                "implementation_tree_sha256": live_tree,
                "core_release_pipeline_sha256": pipeline_tree,
                "release_tooling_sha256": identity["release_tooling_sha256"],
                "protocol21_replay": {
                    "source_suite": "release/vnext/protocol21_source_suite.json",
                    "source_suite_sha256": source_digest,
                    "core_release_pipeline_sha256": pipeline_tree,
                },
                "backend_runtime_closure": {
                    "path": "backend_runtime_closure.json",
                    "sha256": runtime_closure_digest,
                    "schema_version": closure_payload["schema_version"],
                    "n_archived_files": 1,
                    "n_external_sources": 0,
                    "n_backend_links": 1,
                    "n_runtime_packages": 0,
                    "identity_sha256": closure_payload["identity_sha256"],
                },
                "formal_batch_contract": {
                    "runtime_evidence_root": "release/vnext/run",
                    "selection_source": (
                        "release/vnext/run/protocol2_v21_core_readiness.json#scenarios"
                    ),
                },
                "pipeline_artifacts": {
                    "path": "release/vnext/run",
                    "core_release_pipeline_sha256": pipeline_tree,
                    "pipeline_manifest_sha256": hashlib.sha256(
                        pipeline_manifest.read_bytes()
                    ).hexdigest(),
                    "readiness_sha256": readiness_digest,
                    "stage_artifacts": stage_artifacts,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "readiness": readiness,
        "pipeline": pipeline_manifest,
        "stage": readiness_dir / mod.FORMAL_CORE_STAGE_FILES["behavioral"],
        "source_suite": source_suite,
        "runtime_closure": runtime_closure,
        "backend_file": backend_file,
        "backend_link": works_root / "Fixture",
    }


def _rebind_runtime_closure(paths: dict[str, Path]) -> dict[str, Any]:
    closure = json.loads(paths["runtime_closure"].read_text(encoding="utf-8"))
    closure.pop("identity_sha256", None)
    closure["identity_sha256"] = hashlib.sha256(
        json.dumps(closure, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    paths["runtime_closure"].write_text(json.dumps(closure), encoding="utf-8")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    binding = manifest["backend_runtime_closure"]
    binding.update(
        {
            "sha256": hashlib.sha256(paths["runtime_closure"].read_bytes()).hexdigest(),
            "n_archived_files": len(closure["archived_files"]),
            "n_external_sources": len(closure["external_sources"]),
            "n_backend_links": len(closure["backend_links"]),
            "n_runtime_packages": len(closure["runtime_packages"]),
            "identity_sha256": closure["identity_sha256"],
        }
    )
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    return closure


def _rebind_formal_readiness(paths: dict[str, Path]) -> None:
    readiness_digest = hashlib.sha256(paths["readiness"].read_bytes()).hexdigest()
    pipeline = json.loads(paths["pipeline"].read_text(encoding="utf-8"))
    readiness_stage = next(
        row for row in pipeline["stages"] if row["name"] == "readiness"
    )
    readiness_stage["output_sha256"] = readiness_digest
    paths["pipeline"].write_text(json.dumps(pipeline), encoding="utf-8")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    artifacts = manifest["pipeline_artifacts"]
    artifacts["readiness_sha256"] = readiness_digest
    artifacts["stage_artifacts"]["readiness"]["sha256"] = readiness_digest
    artifacts["pipeline_manifest_sha256"] = hashlib.sha256(
        paths["pipeline"].read_bytes()
    ).hexdigest()
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")


def _bind_external_checkout(paths: dict[str, Path], *, root: Path) -> tuple[Path, Path]:
    checkout = root / "works" / "External"
    checkout.mkdir()
    source = checkout / "input.json"
    source.write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "add", "input.json"], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "initial"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    closure = json.loads(paths["runtime_closure"].read_text(encoding="utf-8"))
    closure["external_sources"] = {
        "external": {
            "delivery": "git_checkout",
            "url": "https://example.test/external.git",
            "revision": revision,
            "required_files": {
                "works/External/input.json": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
            },
            "metadata": {
                "backend_kinds": ["fixture"],
                "license_status": "verified_mit",
                "redistributed": False,
                "roles": {"works/External/input.json": ["runtime_input"]},
                "root": "works/External",
            },
        }
    }
    closure["summary"]["n_external_sources"] = 1
    paths["runtime_closure"].write_text(json.dumps(closure), encoding="utf-8")
    _rebind_runtime_closure(paths)
    return checkout, source


def test_legacy_slice_registry_is_removed() -> None:
    assert mod.SCENARIO_SLICES == {}
    assert mod.DYNAMIC_SCENARIO_SLICES == {}
    assert mod.PROTOCOL21_FORMAL_SLICES == frozenset()
    with pytest.raises(ValueError, match="unknown scenario slice"):
        mod._resolve_patterns("manifest_fixture", None)


def test_formal_manifest_resolves_hash_bound_readiness(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)

    binding = mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)

    assert binding["slice_name"].startswith("manifest_")
    assert binding["dynamic_slice_spec"] == (
        "vnext",
        "run/protocol2_v21_core_readiness.json",
        {},
    )
    assert (
        binding["readiness_sha256"]
        == hashlib.sha256(paths["readiness"].read_bytes()).hexdigest()
    )
    assert (
        binding["core_release_pipeline_sha256"]
        == json.loads(paths["manifest"].read_text(encoding="utf-8"))[
            "core_release_pipeline_sha256"
        ]
    )
    assert (
        binding["backend_runtime_closure_identity_sha256"]
        == json.loads(paths["runtime_closure"].read_text(encoding="utf-8"))[
            "identity_sha256"
        ]
    )


def test_formal_manifest_rejects_changed_archived_backend_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    paths["backend_file"].write_bytes(b"mutated runtime\n")

    with pytest.raises(ValueError, match="formal backend runtime closure drift"):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


def test_formal_manifest_rejects_symlinked_runtime_closure(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    target = paths["runtime_closure"].with_name("runtime-closure-target.json")
    paths["runtime_closure"].replace(target)
    paths["runtime_closure"].symlink_to(target.name)

    with pytest.raises(ValueError, match="formal backend runtime closure drift"):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


def test_formal_manifest_rejects_external_checkout_commit_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    checkout, _source = _bind_external_checkout(paths, root=root)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "--allow-empty", "-qm", "drift"],
        check=True,
    )

    with pytest.raises(ValueError, match="formal backend runtime closure drift"):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


def test_formal_manifest_rejects_dirty_external_checkout(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    checkout, _source = _bind_external_checkout(paths, root=root)
    (checkout / "untracked.tmp").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(ValueError, match="formal backend runtime closure drift"):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


def test_formal_manifest_rejects_changed_external_required_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    source_root = root / "works" / "UserSource"
    source_root.mkdir()
    source = source_root / "input.json"
    source.write_text("{}\n", encoding="utf-8")
    closure = json.loads(paths["runtime_closure"].read_text(encoding="utf-8"))
    closure["external_sources"] = {
        "external": {
            "delivery": "user_provided",
            "url": "https://example.test/source",
            "revision": "fixture-v1",
            "required_files": {
                "works/UserSource/input.json": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest()
            },
            "metadata": {
                "backend_kinds": ["fixture"],
                "license_status": "verified_terms",
                "redistributed": False,
                "roles": {"works/UserSource/input.json": ["runtime_input"]},
                "root": "works/UserSource",
            },
        }
    }
    closure["summary"]["n_external_sources"] = 1
    paths["runtime_closure"].write_text(json.dumps(closure), encoding="utf-8")
    _rebind_runtime_closure(paths)
    source.write_text('{"drift": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="formal backend runtime closure drift"):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


def test_formal_manifest_rejects_changed_backend_link_target(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    alternate = root / "data_operate_v058" / "backends" / "alternate"
    alternate.mkdir()
    paths["backend_link"].unlink()
    paths["backend_link"].symlink_to(alternate)

    with pytest.raises(ValueError, match="formal backend runtime closure drift"):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


def test_formal_manifest_rejects_installed_runtime_package_version_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    uv_lock = root / "uv.lock"
    uv_lock.write_text("version = 1\n", encoding="utf-8")
    paths = _write_green_formal_gate(root)
    entry = {
        "version": importlib.metadata.version("pytest") + ".drift",
        "source": {"registry": "https://pypi.org/simple"},
        "artifacts": [],
    }
    entry["identity_sha256"] = hashlib.sha256(
        json.dumps(entry, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    entries = [entry]
    closure = json.loads(paths["runtime_closure"].read_text(encoding="utf-8"))
    closure["runtime_packages"] = {
        "pytest": {
            "backend_kinds": ["fixture"],
            "lock_entries": entries,
            "lock_entries_sha256": hashlib.sha256(
                json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest(),
            "uv_lock_sha256": hashlib.sha256(uv_lock.read_bytes()).hexdigest(),
        }
    }
    closure["summary"]["n_runtime_packages"] = 1
    paths["runtime_closure"].write_text(json.dumps(closure), encoding="utf-8")
    _rebind_runtime_closure(paths)

    with pytest.raises(ValueError, match="formal backend runtime closure drift"):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


def test_formal_manifest_rejects_cross_release_runtime_evidence_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    evidence_root = root / "release" / "vnext_runtime" / "protocol21"
    evidence_root.parent.mkdir(parents=True)
    paths["readiness"].parent.replace(evidence_root)
    moved = {"readiness": evidence_root / paths["readiness"].name}

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    contract = manifest["formal_batch_contract"]
    contract["runtime_evidence_root"] = evidence_root.relative_to(root).as_posix()
    contract["selection_source"] = (
        moved["readiness"].relative_to(root).as_posix() + "#scenarios"
    )
    manifest["pipeline_artifacts"]["path"] = evidence_root.relative_to(root).as_posix()
    manifest["pipeline_artifacts"].update(
        {
            "readiness_sha256": hashlib.sha256(
                moved["readiness"].read_bytes()
            ).hexdigest(),
        }
    )
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="formal release identity binding mismatch"):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


def test_formal_runtime_binding_rejects_changed_readiness(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    binding = mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)
    meta = mod._formal_runtime_binding_metadata(binding)
    paths["readiness"].write_text("{}\n", encoding="utf-8")

    reasons = mod._formal_runtime_binding_reasons(meta, repo_root=root)

    assert reasons == ["formal_runtime_evidence_revalidation_failed:ValueError"]


def test_formal_runtime_binding_includes_release_and_tooling_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)

    binding = mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)
    meta = mod._formal_runtime_binding_metadata(binding)

    assert binding["release_id"] == "vnext"
    assert binding["release_tooling_sha256"] == implementation_identity(root)[
        "release_tooling_sha256"
    ]
    assert meta["formal_release_id"] == "vnext"
    assert meta["formal_release_tooling_sha256"] == binding[
        "release_tooling_sha256"
    ]


def test_formal_sidecar_paths_are_batch_relative_and_contained(
    tmp_path: Path,
) -> None:
    batch_root = tmp_path / "batch"
    trajectory = batch_root / "trajectories" / "episode"
    trajectory.parent.mkdir(parents=True)
    sidecar = trajectory.with_suffix(".provider_audit.jsonl")
    sidecar.write_text('{"kind":"request"}\n', encoding="utf-8")
    row = {
        "agent_treatment_sha256": "a" * 64,
        "trajectory_summary": {
            "trajectory_path": "trajectories/episode",
            "provider_audit_artifact": {
                "schema_version": "provider_interaction_audit_v1",
                "path": "trajectories/episode.provider_audit.jsonl",
                "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                "event_count": 1,
            },
        },
    }

    assert mod._trajectory_sidecar_eligibility_reasons(
        row,
        summary_key="provider_audit_artifact",
        stem="provider_audit",
        schema_version="provider_interaction_audit_v1",
        require_nonempty=True,
        batch_root=batch_root,
    ) == ["provider_audit_artifact_treatment_path_mismatch"]

    row["trajectory_summary"]["provider_audit_artifact"]["path"] = str(
        tmp_path / "outside.jsonl"
    )
    assert mod._trajectory_sidecar_eligibility_reasons(
        row,
        summary_key="provider_audit_artifact",
        stem="provider_audit",
        schema_version="provider_interaction_audit_v1",
        require_nonempty=True,
        batch_root=batch_root,
    ) == ["provider_audit_artifact_path_outside_batch"]


def test_portable_formal_result_paths_reject_escape(tmp_path: Path) -> None:
    batch_root = tmp_path / "batch"
    prefix = batch_root / "treatment-" / "episode"
    row = {
        "episode_log_path": str(batch_root / "logs" / "episode.json"),
        "trajectory_summary": {
            "trajectory_path": str(prefix),
            "provider_audit_artifact": {
                "path": str(prefix) + ".provider_audit.jsonl"
            },
        },
    }

    portable = mod._portable_formal_result_paths([row], batch_root=batch_root)[0]
    assert portable["episode_log_path"] == "logs/episode.json"
    assert portable["trajectory_summary"]["trajectory_path"] == (
        "treatment-/episode"
    )
    assert portable["trajectory_summary"]["provider_audit_artifact"]["path"] == (
        "treatment-/episode.provider_audit.jsonl"
    )

    row["episode_log_path"] = str(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="escapes batch root"):
        mod._portable_formal_result_paths([row], batch_root=batch_root)


def test_formal_runtime_binding_detects_valid_runtime_closure_replacement(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    binding = mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)
    meta = mod._formal_runtime_binding_metadata(binding)
    paths["backend_file"].write_bytes(b"replacement runtime\n")
    closure = json.loads(paths["runtime_closure"].read_text(encoding="utf-8"))
    closure["archived_files"]["backends/fixture/runtime.bin"]["sha256"] = (
        hashlib.sha256(paths["backend_file"].read_bytes()).hexdigest()
    )
    paths["runtime_closure"].write_text(json.dumps(closure), encoding="utf-8")
    _rebind_runtime_closure(paths)

    reasons = mod._formal_runtime_binding_reasons(meta, repo_root=root)

    assert (
        "formal_runtime_binding_changed:formal_backend_runtime_closure_identity_sha256"
    ) in reasons


def test_formal_manifest_rejects_changed_core_stage(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    paths["stage"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="formal pipeline stage invalid:behavioral"):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


def test_formal_manifest_rejects_changed_source_suite(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    paths["source_suite"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="formal source suite binding mismatch"):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


def test_formal_manifest_does_not_require_live_admission_tooling_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    script = root / "scripts" / "calibrate_core_candidate.py"
    script.parent.mkdir(parents=True)
    script.write_text("SUMO_WORKERS = 1\n", encoding="utf-8")

    binding = mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)

    assert binding["manifest_path"] == str(paths["manifest"].resolve())


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("release", "formal core release pipeline binding mismatch"),
        ("pipeline_artifacts", "formal core release pipeline binding mismatch"),
        ("protocol21_replay", "formal core release pipeline binding mismatch"),
        ("pipeline", "formal core release pipeline binding mismatch"),
        ("stage_row", "formal pipeline stage invalid"),
        ("stage_artifact", "formal pipeline stage invalid"),
        ("non_hex", "formal core release pipeline hash invalid"),
    ),
)
def test_formal_manifest_rejects_core_pipeline_closure_drift(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    pipeline = json.loads(paths["pipeline"].read_text(encoding="utf-8"))
    if mutation == "release":
        manifest["core_release_pipeline_sha256"] = "0" * 64
    elif mutation in {"pipeline_artifacts", "protocol21_replay"}:
        manifest[mutation]["core_release_pipeline_sha256"] = "0" * 64
    elif mutation == "pipeline":
        pipeline["core_release_pipeline_sha256"] = "0" * 64
    elif mutation == "stage_row":
        pipeline["stages"][0]["core_release_pipeline_sha256"] = "0" * 64
    elif mutation == "stage_artifact":
        stage = paths["stage"]
        artifact = json.loads(stage.read_text(encoding="utf-8"))
        artifact["core_release_pipeline_sha256"] = "0" * 64
        stage.write_text(json.dumps(artifact), encoding="utf-8")
        digest = hashlib.sha256(stage.read_bytes()).hexdigest()
        manifest["pipeline_artifacts"]["stage_artifacts"]["behavioral"]["sha256"] = (
            digest
        )
        row = next(item for item in pipeline["stages"] if item["name"] == "behavioral")
        row["output_sha256"] = digest
    else:
        manifest["core_release_pipeline_sha256"] = "z" * 64
    if mutation not in {"release", "pipeline_artifacts", "protocol21_replay"}:
        paths["pipeline"].write_text(json.dumps(pipeline), encoding="utf-8")
        manifest["pipeline_artifacts"]["pipeline_manifest_sha256"] = hashlib.sha256(
            paths["pipeline"].read_bytes()
        ).hexdigest()
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    (
        ("pipeline_status", "formal pipeline stage set mismatch"),
        ("pipeline_source", "formal pipeline stage set mismatch"),
        ("stage_order", "formal pipeline stage set mismatch"),
        ("duplicate_stage", "formal pipeline stage set mismatch"),
        ("noncanonical_path", "formal pipeline stage path invalid"),
    ),
)
def test_formal_manifest_rejects_malformed_pipeline_contract(
    tmp_path: Path,
    mutation: str,
    expected_error: str,
) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    pipeline = json.loads(paths["pipeline"].read_text(encoding="utf-8"))
    if mutation == "pipeline_status":
        pipeline["status"] = "incomplete"
    elif mutation == "pipeline_source":
        pipeline["source_suite_sha256"] = "0" * 64
    elif mutation == "stage_order":
        pipeline["stages"][0], pipeline["stages"][1] = (
            pipeline["stages"][1],
            pipeline["stages"][0],
        )
    elif mutation == "duplicate_stage":
        pipeline["stages"][1] = dict(pipeline["stages"][0])
    else:
        manifest["pipeline_artifacts"]["stage_artifacts"]["behavioral"][
            "relative_path"
        ] = "behavioral.json"
    paths["pipeline"].write_text(json.dumps(pipeline), encoding="utf-8")
    manifest["pipeline_artifacts"]["pipeline_manifest_sha256"] = hashlib.sha256(
        paths["pipeline"].read_bytes()
    ).hexdigest()
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


def test_formal_manifest_does_not_require_diagnostic_smoke(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)

    binding = mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)

    assert "diagnostic_readiness_path" not in binding
    assert "agency_readiness_bundle_path" not in binding


def test_formal_manifest_rejects_empty_source_readiness(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    manifest_dir = root / "release" / "vnext"
    manifest_dir.mkdir(parents=True)
    readiness = manifest_dir / "readiness.json"
    readiness.write_text("{}\n", encoding="utf-8")
    diagnostic = manifest_dir / "diagnostic.json"
    diagnostic.write_text("{}\n", encoding="utf-8")
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "formal_batch_contract": {
                    "selection_source": "release/vnext/readiness.json",
                    "diagnostic_readiness": "release/vnext/diagnostic.json",
                },
                "pipeline_artifacts": {
                    "readiness_sha256": hashlib.sha256(
                        readiness.read_bytes()
                    ).hexdigest(),
                    "diagnostic_readiness_sha256": hashlib.sha256(
                        diagnostic.read_bytes()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="formal_run_contract_missing"):
        mod.resolve_formal_manifest_slice(manifest, repo_root=root)


def test_formal_manifest_rejects_changed_wakeup_policy(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    paths = _write_green_formal_gate(root)
    readiness = json.loads(paths["readiness"].read_text(encoding="utf-8"))
    readiness["formal_run_contract"]["wakeup_policy"][
        "harness_periodic_supervisory_scan"
    ] = True
    paths["readiness"].write_text(json.dumps(readiness), encoding="utf-8")
    _rebind_formal_readiness(paths)

    with pytest.raises(ValueError, match="formal wakeup policy mismatch"):
        mod.resolve_formal_manifest_slice(paths["manifest"], repo_root=root)


def test_formal_manifest_rejects_stale_readiness_hash(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    manifest_dir = root / "release" / "vnext"
    readiness_dir = manifest_dir
    manifest_dir.mkdir(parents=True)
    readiness = readiness_dir / "protocol2_v21_core_readiness.json"
    readiness.write_text("{}\n", encoding="utf-8")
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "formal_batch_contract": {
                    "selection_source": (
                        "release/vnext/protocol2_v21_core_readiness.json"
                    )
                },
                "pipeline_artifacts": {"readiness_sha256": "0" * 64},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="readiness hash mismatch"):
        mod.resolve_formal_manifest_slice(manifest, repo_root=root)


def test_formal_manifest_rejects_cross_release_readiness(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    manifest_dir = root / "release" / "vnext"
    readiness_dir = root / "release" / "other"
    manifest_dir.mkdir(parents=True)
    readiness_dir.mkdir(parents=True)
    readiness = readiness_dir / "protocol2_v21_core_readiness.json"
    readiness.write_text("{}\n", encoding="utf-8")
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "formal_batch_contract": {
                    "selection_source": (
                        "release/other/protocol2_v21_core_readiness.json"
                    )
                },
                "pipeline_artifacts": {
                    "readiness_sha256": hashlib.sha256(
                        readiness.read_bytes()
                    ).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same release directory"):
        mod.resolve_formal_manifest_slice(manifest, repo_root=root)


def test_protocol21_v52_slice_uses_formal_run_contract() -> None:
    reasons = mod._validate_protocol21_formal_run(
        _agentic_formal_config(),
        _agentic_formal_readiness(),
        suite_manifest_sha256="suite",
    )

    assert reasons == []


def test_formal_run_contract_rejects_dirty_git_tree() -> None:
    reasons = mod._validate_protocol21_formal_run(
        _agentic_formal_config(git_dirty=True),
        _agentic_formal_readiness(),
        suite_manifest_sha256="suite",
    )

    assert "formal_git_tree_must_be_clean" in reasons


def test_formal_run_contract_rejects_stateless_interaction() -> None:
    reasons = mod._validate_protocol21_formal_run(
        _agentic_formal_config(interaction_mode="logical_stateless"),
        _agentic_formal_readiness(),
        suite_manifest_sha256="suite",
    )

    assert "formal_interaction_mode_must_be_logical_persistent" in reasons


def test_formal_run_contract_rejects_missing_interaction_mode() -> None:
    reasons = mod._validate_protocol21_formal_run(
        _agentic_formal_config(interaction_mode=None),
        _agentic_formal_readiness(),
        suite_manifest_sha256="suite",
    )

    assert "formal_interaction_mode_must_be_logical_persistent" in reasons


def test_formal_run_contract_missing_fails_closed() -> None:
    reasons = mod._validate_protocol21_formal_run(
        _agentic_formal_config(),
        _agentic_formal_readiness(formal_run_contract=None),
        suite_manifest_sha256="suite",
    )

    assert "formal_run_contract_missing" in reasons


def test_formal_run_contract_rejects_changed_wakeup_policy() -> None:
    contract = copy.deepcopy(_agentic_formal_readiness()["formal_run_contract"])
    contract["wakeup_policy"]["harness_periodic_supervisory_scan"] = True

    reasons = mod._validate_protocol21_formal_run(
        _agentic_formal_config(),
        _agentic_formal_readiness(formal_run_contract=contract),
        suite_manifest_sha256="suite",
    )

    assert "formal_wakeup_policy_mismatch" in reasons


def test_formal_run_contract_allows_unverified_provider_quota() -> None:
    reasons = mod._validate_protocol21_formal_run(
        _agentic_formal_config(
            provider_rpm_limit=None,
            provider_rpd_limit=None,
            provider_rate_limit_scope=None,
        ),
        _agentic_formal_readiness(),
        suite_manifest_sha256="suite",
    )

    assert reasons == []


def test_legacy_formal_run_contract_is_unsupported() -> None:
    reasons = mod._validate_protocol21_formal_run(
        _agentic_formal_config(),
        _agentic_formal_readiness(
            formal_run_contract={"contract_version": "legacy_stateless.v1"}
        ),
        suite_manifest_sha256="suite",
    )

    assert "formal_run_contract_version_unsupported" in reasons


def test_formal_run_contract_requires_finalization() -> None:
    reasons = mod._validate_protocol21_formal_run(
        _agentic_formal_config(finalize=False),
        _agentic_formal_readiness(),
        suite_manifest_sha256="suite",
    )

    assert "formal_finalization_required" in reasons


def test_formal_run_contract_rejects_missing_scenario_construct_before_jobs() -> None:
    reasons = mod._validate_protocol21_formal_run(
        _agentic_formal_config(),
        _agentic_formal_readiness(),
        suite_manifest_sha256="suite",
        scenario_bodies={
            "valid": {"construct_contract": "operational_agency.v1"},
            "missing": {},
            "wrong": {"construct_contract": "legacy"},
        },
    )

    assert reasons == [
        "formal_scenario_construct_contract_mismatch:wrong",
        "formal_scenario_construct_contract_missing:missing",
    ]


def test_dynamic_slice_manifest_uses_bound_row_horizon_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scenario_path = tmp_path / "scenarios" / "fixture.yaml"
    scenario_path.parent.mkdir()
    scenario_path.write_text(
        "scenario_signature: sig\nseed: 42\n",
        encoding="utf-8",
    )
    release_dir = tmp_path / "release" / "fixture"
    release_dir.mkdir(parents=True)
    artifact = {
        "n_scenarios": 1,
        "scenarios": [
            {
                "path": "scenarios/fixture.yaml",
                "scenario_signature": "sig",
                "seed": 42,
                "horizon_ticks": 7,
            }
        ],
    }
    (release_dir / "readiness.json").write_text(
        json.dumps(artifact),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        mod.DYNAMIC_SCENARIO_SLICES,
        "fixture",
        ("fixture", "readiness.json", {}),
    )

    actual = mod._suite_manifest_sha256_for_slice(
        "fixture",
        ["fixture"],
        {"fixture": {"scenario_signature": "sig", "seed": 42}},
    )
    expected = mod.canonical_suite_manifest_sha256(
        ["fixture"],
        {
            "fixture": {
                "scenario_signature": "sig",
                "seed": 42,
                "horizon_ticks": 7,
            }
        },
    )

    assert actual == expected


def test_dynamic_slice_injects_hash_bound_construct_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release_dir = tmp_path / "release" / "fixture"
    release_dir.mkdir(parents=True)
    artifact = {
        "n_scenarios": 1,
        "scenarios": [
            {
                "path": "scenarios/fixture.yaml",
                "scenario_signature": "sig",
                "seed": 42,
                "horizon_ticks": 7,
                "construct_contract": "operational_agency.v1",
                "source_denominator_key": "fixture:source-a",
                "case_ledger": {
                    "schema_version": "0.1",
                    "source_denominator_key": "fixture:source-a",
                },
            }
        ],
    }
    (release_dir / "readiness.json").write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        mod.DYNAMIC_SCENARIO_SLICES,
        "fixture",
        ("fixture", "readiness.json", {}),
    )
    bodies = {
        "fixture": {
            "scenario_signature": "sig",
            "seed": 42,
            "horizon_ticks": 7,
        }
    }

    mod._bind_scenario_contracts_for_slice("fixture", ["fixture"], bodies)

    assert bodies["fixture"]["construct_contract"] == "operational_agency.v1"
    assert bodies["fixture"]["source_denominator_key"] == "fixture:source-a"
    assert bodies["fixture"]["case_ledger"] == {
        "schema_version": "0.1",
        "source_denominator_key": "fixture:source-a",
    }


@pytest.mark.parametrize(
    ("row_update", "error"),
    [
        ({"source_denominator_key": None}, "source denominator key"),
        ({"case_ledger": None}, "case ledger"),
        (
            {
                "source_denominator_key": "fixture:source-a",
                "case_ledger": {
                    "schema_version": "0.1",
                    "source_denominator_key": "fixture:source-b",
                },
            },
            "case ledger source denominator mismatch",
        ),
    ],
)
def test_dynamic_slice_formal_source_contract_fails_closed(
    tmp_path: Path,
    monkeypatch,
    row_update: dict[str, Any],
    error: str,
) -> None:
    release_dir = tmp_path / "release" / "fixture"
    release_dir.mkdir(parents=True)
    row = {
        "path": "scenarios/fixture.yaml",
        "construct_contract": "operational_agency.v1",
        "source_denominator_key": "fixture:source-a",
        "case_ledger": {
            "schema_version": "0.1",
            "source_denominator_key": "fixture:source-a",
        },
    }
    row.update(row_update)
    (release_dir / "readiness.json").write_text(
        json.dumps({"scenarios": [row]}), encoding="utf-8"
    )
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        mod.DYNAMIC_SCENARIO_SLICES,
        "fixture",
        ("fixture", "readiness.json", {}),
    )

    with pytest.raises(ValueError, match=error):
        mod._bind_scenario_contracts_for_slice("fixture", ["fixture"], {"fixture": {}})


def test_dynamic_slice_manifest_keeps_artifact_order_when_run_order_is_sorted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release_dir = tmp_path / "release" / "fixture"
    release_dir.mkdir(parents=True)
    artifact_rows = [
        {
            "path": "scenarios/z.yaml",
            "scenario_signature": "sig-z",
            "seed": 42,
            "horizon_ticks": 7,
        },
        {
            "path": "scenarios/a.yaml",
            "scenario_signature": "sig-a",
            "seed": 43,
            "horizon_ticks": 8,
        },
    ]
    (release_dir / "readiness.json").write_text(
        json.dumps({"n_scenarios": 2, "scenarios": artifact_rows}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        mod.DYNAMIC_SCENARIO_SLICES,
        "fixture",
        ("fixture", "readiness.json", {}),
    )
    bodies = {
        "a": {"scenario_signature": "sig-a", "seed": 43},
        "z": {"scenario_signature": "sig-z", "seed": 42},
    }

    actual = mod._suite_manifest_sha256_for_slice(
        "fixture",
        ["a", "z"],
        bodies,
    )
    expected = mod.canonical_suite_manifest_sha256(
        ["z", "a"],
        {
            "z": {**bodies["z"], "horizon_ticks": 7},
            "a": {**bodies["a"], "horizon_ticks": 8},
        },
    )

    assert actual == expected


def test_protocol21_binding_reverifies_all_upstream_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts.build_protocol21_core_readiness import (
        build_readiness_from_paths,
    )
    from tests.test_build_protocol21_core_readiness import _write_fixture

    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    paths = _write_fixture(evidence_dir)
    report = build_readiness_from_paths(**paths)
    assert report["formal_evaluation_ready"] is True
    release_dir = tmp_path / "release" / "fixture"
    release_dir.mkdir(parents=True)
    readiness_path = release_dir / "readiness.json"
    readiness_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        mod.DYNAMIC_SCENARIO_SLICES,
        "manifest_fixture",
        ("fixture", "readiness.json", {}),
    )

    green = mod._suite_eligibility_binding("manifest_fixture")
    assert green["suite_blocked"] is False

    paths["behavioral"].write_text(
        json.dumps({"tampered": True}),
        encoding="utf-8",
    )
    blocked = mod._suite_eligibility_binding("manifest_fixture")
    assert blocked["suite_blocked"] is True
    assert "artifact_hash_mismatch" in blocked["reason"]["blockers"]


@pytest.mark.parametrize(
    "binding_field",
    [
        "artifact_bindings",
        "scenario_yaml_bindings",
        "source_file_bindings",
    ],
)
def test_protocol21_binding_rejects_frozen_binding_map_drift(
    tmp_path: Path,
    monkeypatch,
    binding_field: str,
) -> None:
    from scripts.build_protocol21_core_readiness import (
        build_readiness_from_paths,
    )
    from tests.test_build_protocol21_core_readiness import _write_fixture

    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    paths = _write_fixture(evidence_dir)
    report = build_readiness_from_paths(**paths)
    assert report["formal_evaluation_ready"] is True

    frozen = copy.deepcopy(report)
    if binding_field == "artifact_bindings":
        frozen[binding_field]["behavioral"]["sha256"] = "0" * 64
    elif binding_field == "scenario_yaml_bindings":
        scenario_id = next(iter(frozen[binding_field]))
        frozen[binding_field][scenario_id]["sha256"] = "0" * 64
    else:
        scenario_id = next(iter(frozen[binding_field]))
        source_path = next(iter(frozen[binding_field][scenario_id]))
        frozen[binding_field][scenario_id][source_path] = "0" * 64

    release_dir = tmp_path / "release" / "fixture"
    release_dir.mkdir(parents=True)
    readiness_path = release_dir / "readiness.json"
    readiness_path.write_text(json.dumps(frozen), encoding="utf-8")
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        mod.DYNAMIC_SCENARIO_SLICES,
        "manifest_fixture",
        ("fixture", "readiness.json", {}),
    )

    blocked = mod._suite_eligibility_binding("manifest_fixture")

    assert blocked["suite_blocked"] is True
    assert (
        f"readiness_reverification_mismatch:{binding_field}"
        in blocked["reason"]["blockers"]
    )


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"models": ["a", "b"]}, "formal_model_count_per_shard_must_equal_one"),
        ({"pass_k": 0}, "formal_pass_k_below_minimum"),
        ({"max_workers": 33}, "formal_max_workers_out_of_range"),
        ({"temperature": 1.0}, "formal_temperature_must_equal_zero"),
        ({"prompt_mode": "debug"}, "formal_prompt_mode_must_be_strict"),
        ({"seed_mode": "fixed"}, "formal_seed_mode_must_be_scenario"),
        ({"scheduler_mode": "per_model"}, "formal_scheduler_must_be_global"),
        ({"save_trajectories": False}, "formal_trajectories_required"),
        ({"allow_blocked_suite": True}, "formal_cannot_allow_blocked_suite"),
        (
            {"provider_rpm_limit": -1},
            "formal_provider_rpm_limit_invalid",
        ),
        (
            {"provider_rpm_limit": 0},
            "formal_provider_rpm_limit_invalid",
        ),
        (
            {"provider_rpd_limit": -1},
            "formal_provider_rpd_limit_invalid",
        ),
        (
            {"provider_rpd_limit": 0},
            "formal_provider_rpd_limit_invalid",
        ),
        (
            {"provider_rate_limit_scope": "   "},
            "formal_provider_rate_limit_scope_required",
        ),
    ],
)
def test_formal_run_contract_rejects_invalid_configuration(
    override: dict,
    reason: str,
) -> None:
    config = _agentic_formal_config(**override)

    assert reason in mod._validate_protocol21_formal_run(
        config,
        _agentic_formal_readiness(),
        suite_manifest_sha256="suite",
    )


def test_formal_run_contract_accepts_agentic_persistent_config() -> None:
    reasons = mod._validate_protocol21_formal_run(
        _agentic_formal_config(),
        _agentic_formal_readiness(),
        suite_manifest_sha256="suite",
    )

    assert reasons == []


def test_formal_run_contract_rejects_readiness_and_suite_hash_mismatch() -> None:
    config = {
        "scenario_slice": "manifest_fixture",
        "models": ["a", "b", "c"],
        "pass_k": 3,
        "max_workers": 4,
        "prompt_mode": "strict",
        "seed_mode": "scenario",
        "scheduler_mode": "global",
        "save_trajectories": True,
        "allow_blocked_suite": False,
    }

    reasons = mod._validate_protocol21_formal_run(
        config,
        {
            "formal_evaluation_ready": False,
            "suite_manifest_sha256": "expected",
            "readiness_source_binding_valid": False,
        },
        suite_manifest_sha256="actual",
    )

    assert "formal_readiness_not_green" in reasons
    assert "formal_readiness_source_hash_mismatch" in reasons
    assert "formal_suite_manifest_mismatch" in reasons


@pytest.mark.parametrize(
    ("readiness_override", "reason"),
    [
        ({"scoring_version": "0.9.0"}, "formal_scoring_version_mismatch"),
        (
            {"primary_leaderboard_formula_version": None},
            "formal_primary_leaderboard_formula_missing",
        ),
        (
            {"primary_leaderboard_formula_version": "legacy"},
            "formal_primary_leaderboard_formula_mismatch",
        ),
        (
            {"primary_inference_version": "legacy"},
            "formal_primary_inference_version_mismatch",
        ),
        (
            {"task_completion_input_unit": "points"},
            "formal_task_completion_input_unit_mismatch",
        ),
        (
            {"task_completion_score_unit": "fraction"},
            "formal_task_completion_score_unit_mismatch",
        ),
        (
            {"weighted_equity_formula_version": "legacy"},
            "formal_weighted_equity_formula_mismatch",
        ),
    ],
)
def test_formal_run_contract_rejects_stale_scoring_contract(
    readiness_override: dict,
    reason: str,
) -> None:
    config = _agentic_formal_config()
    readiness = _agentic_formal_readiness(**readiness_override)

    assert reason in mod._validate_protocol21_formal_run(
        config,
        readiness,
        suite_manifest_sha256="suite",
    )


def test_blocked_protocol21_slice_fails_before_jobs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    scenario = "traffic/example"
    monkeypatch.setattr(mod, "_resolve_patterns", lambda *_: [scenario])
    monkeypatch.setattr(mod, "_expand_scenarios", lambda _: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda _: {
            "seed": 42,
            "scenario_signature": "sig",
            "backend_kind": "mock_sumo",
        },
    )
    monkeypatch.setattr(
        mod,
        "_suite_eligibility_binding",
        lambda _: {
            "suite_blocked": True,
            "formal_evaluation_ready": False,
            "suite_manifest_sha256": "unused",
        },
    )
    monkeypatch.setattr(
        mod,
        "_build_jobs",
        lambda **_: pytest.fail("blocked suite must not build jobs"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(tmp_path / "blocked"),
            "--scenario-slice",
            "manifest_fixture",
            "--models",
            "a,b,c",
            "--dry-run",
        ],
    )

    assert mod.main() == 1
    assert not (tmp_path / "blocked" / "run_config.json").exists()


def _fake_episode_result(
    scenario_slug: str,
    model: str,
    seed: int,
    scenario_signature: str,
    family: str,
    backend_kind: str,
    *,
    total_score: float,
    n_tool_calls: int,
    llm_failed: int = 0,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario_slug": scenario_slug,
        "scenario_id": scenario_slug.split("/")[-1],
        "scenario_signature": scenario_signature,
        "family": family,
        "difficulty_mode": "time_pressure",
        "difficulty_level": "basic",
        "backend_kind": backend_kind,
        "model": model,
        "agent_name": f"llm_agent/{model}",
        "seed": seed,
        "temperature": 1.0,
        "status": status,
        "score": {
            "total_score": total_score,
            "raw_total": total_score * 1.2,
            "dimensions": [
                {
                    "name": "system_survival",
                    "raw_score": 100.0,
                    "calibrated_score": 100.0,
                    "applicable": True,
                    "support_count": 1,
                    "evidence_ids": ["ev_state"],
                    "reason": "fixture state evidence",
                    "weight": 1.5,
                }
            ],
        },
        "counterfactual": {
            "prevented_loss": total_score / 10.0,
            "actual_cost": 100.0,
            "counterfactual_cost": 120.0,
        },
        "decision_impact": {
            "n_control_calls": max(1, n_tool_calls // 2),
            "outcome_changed": True,
        },
        "foresight": {"foresight_score": total_score / 2.0},
        "trajectory_summary": {
            "n_tool_calls": n_tool_calls,
            "n_wait_actions": 1,
            "trajectory_path": f"/tmp/{scenario_slug.replace('/', '_')}.jsonl",
            "tool_histogram": {
                "query_grid_state": 1,
                "redispatch_generation": max(0, n_tool_calls - 1),
            },
            "llm": {"llm_calls_ok": 2, "llm_calls_failed": llm_failed},
        },
        "ground_truth_summary": {"chose_fatal_option": False},
        "episode_log_path": f"/tmp/{model}/{scenario_slug.replace('/', '_')}.log",
    }
    if error:
        row["error"] = error
    return row


def test_recover_execution_hints_supports_scenario_seed_log_format(
    tmp_path: Path,
) -> None:
    (tmp_path / "batch_run.log").write_text(
        "2026-07-25 11:04:39,446 [INFO] Scheduled 223 episodes "
        "(223 scenarios × 1 models × scenario seed mode × pass_k=1); "
        "max_workers=16\n"
        "2026-07-25 22:39:42,749 [INFO] Scheduled 0 episodes "
        "(223 scenarios × 4 models × fixed seed mode × pass_k=1); "
        "max_workers=6\n",
        encoding="utf-8",
    )

    hints = mod._recover_execution_hints_from_batch_log(tmp_path)

    assert hints["scheduled_episodes"] == 223
    assert hints["n_models"] == 1
    assert hints["seed_mode"] == "scenario"
    assert hints["pass_k"] == 1
    assert hints["max_workers_effective"] == 16


def test_recover_models_prefers_observed_when_original_schedule_is_complete() -> None:
    rows = [
        {"model": "hy3-ioa", "status": "ok"},
        {"model": "hy3-ioa", "status": "ok"},
    ]

    recovered = mod._recover_models_for_finalize(
        configured_models=["gpt-5", "gemini-pro", "o3", "gpt-4.1"],
        rows=rows,
        scheduled_model_count=1,
    )

    assert recovered == ["hy3-ioa"]


def test_recover_models_preserves_configured_models_for_partial_batch() -> None:
    rows = [{"model": "gpt-5", "status": "ok"}]

    recovered = mod._recover_models_for_finalize(
        configured_models=["gpt-5", "gemini-pro"],
        rows=rows,
        scheduled_model_count=2,
    )

    assert recovered == ["gpt-5", "gemini-pro"]


def test_resume_key_uses_scenario_signature_and_temperature_when_present() -> None:
    jobs = [
        {
            "scenario_slug": "power_grid/foo",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
        {
            "scenario_slug": "power_grid/foo",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-b",
            "temperature": 1.0,
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
        {
            "scenario_slug": "power_grid/foo",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 0.3,
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
    ]
    prior_rows = [
        {
            "status": "ok",
            "scenario_slug": "power_grid/foo",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        }
    ]
    pending = mod._filter_pending_jobs(jobs, prior_rows)
    assert pending == jobs[1:]


def test_resume_key_distinguishes_explicit_pass_ids() -> None:
    jobs = [
        {
            "scenario_slug": "power_grid/foo",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "pass_id": "pass-0",
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
        {
            "scenario_slug": "power_grid/foo",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "pass_id": "pass-1",
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
    ]
    prior_rows = [
        {
            "status": "ok",
            "scenario_slug": "power_grid/foo",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "pass_id": "pass-0",
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        }
    ]

    pending = mod._filter_pending_jobs(jobs, prior_rows)

    assert pending == [jobs[1]]


def test_resume_key_rejects_stale_suite_and_eligibility_bindings() -> None:
    base = {
        "scenario_slug": "power_grid/foo",
        "model": "gpt-5",
        "seed": 42,
        "scenario_signature": "sig-a",
        "temperature": 1.0,
        "evaluation_implementation_fingerprint": (
            mod.EVALUATION_IMPLEMENTATION_FINGERPRINT
        ),
    }
    jobs = [
        {
            **base,
            "suite_manifest_sha256": "suite-new",
            "suite_eligibility_sha256": "eligibility-new",
        }
    ]
    prior_rows = [
        {
            **base,
            "status": "ok",
            "suite_manifest_sha256": "suite-old",
            "suite_eligibility_sha256": "eligibility-old",
        }
    ]

    assert mod._filter_pending_jobs(jobs, prior_rows) == jobs


def test_resume_key_falls_back_to_legacy_slug_model_seed_when_signature_or_temperature_missing() -> (
    None
):
    jobs = [
        {
            "scenario_slug": "power_grid/foo",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
        },
        {
            "scenario_slug": "power_grid/bar",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-b",
            "temperature": 1.0,
        },
    ]
    prior_rows = [
        {
            "status": "ok",
            "scenario_slug": "power_grid/foo",
            "model": "gpt-5",
            "seed": 42,
        }
    ]
    pending = mod._filter_pending_jobs(jobs, prior_rows)
    assert pending == jobs


def test_resume_ignores_in_flight_placeholder_rows() -> None:
    jobs = [
        {
            "scenario_slug": "power_grid/foo",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        }
    ]
    prior_rows = [
        {
            "status": "in_flight",
            "scenario_slug": "power_grid/foo",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        }
    ]
    pending = mod._filter_pending_jobs(jobs, prior_rows)
    assert pending == jobs


def test_resume_retries_unclean_ok_rows() -> None:
    jobs = [
        {
            "scenario_slug": "power_grid/foo",
            "model": "gemini",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
        },
        {
            "scenario_slug": "power_grid/bar",
            "model": "gemini",
            "seed": 42,
            "scenario_signature": "sig-b",
            "temperature": 1.0,
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
        {
            "scenario_slug": "power_grid/baz",
            "model": "gemini",
            "seed": 42,
            "scenario_signature": "sig-c",
            "temperature": 1.0,
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
    ]
    prior_rows = [
        {
            "status": "ok",
            "scenario_slug": "power_grid/foo",
            "model": "gemini",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
            "trajectory_summary": {"llm": {"llm_calls_failed": 2}},
        },
        {
            "status": "ok",
            "scenario_slug": "power_grid/bar",
            "model": "gemini",
            "seed": 42,
            "scenario_signature": "sig-b",
            "temperature": 1.0,
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
            "trajectory_summary": {"llm": {"fallback_wait_ratio": 0.25}},
        },
        {
            "status": "ok",
            "scenario_slug": "power_grid/baz",
            "model": "gemini",
            "seed": 42,
            "scenario_signature": "sig-c",
            "temperature": 1.0,
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
            "trajectory_summary": {"llm": {"llm_calls_failed": 0}},
        },
    ]

    pending = mod._filter_pending_jobs(jobs, prior_rows)

    assert pending == jobs[:1]


def test_resume_retries_any_provider_contaminated_ok_row() -> None:
    job = {
        "scenario_slug": "power_grid/foo",
        "model": "gpt-5",
        "seed": 42,
        "scenario_signature": "sig-a",
        "temperature": 0.0,
        "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
    }
    prior = {
        **job,
        "status": "ok",
        "trajectory_summary": {"llm": {"llm_calls_ok": 99, "llm_calls_failed": 1}},
    }

    assert mod._filter_pending_jobs([job], [prior]) == [job]


def test_resume_retries_tool_argument_parse_contaminated_ok_row() -> None:
    job = {
        "scenario_slug": "power_grid/foo",
        "model": "gpt-5",
        "seed": 42,
        "scenario_signature": "sig-a",
        "temperature": 0.0,
        "evaluation_implementation_fingerprint": (
            mod.EVALUATION_IMPLEMENTATION_FINGERPRINT
        ),
    }
    prior = {
        **job,
        "status": "ok",
        "trajectory_summary": {
            "llm": {
                "llm_calls_failed": 0,
                "tool_argument_parse_failures": 1,
                "fallback_wait_ratio": 0.0,
            }
        },
    }

    assert mod._filter_pending_jobs([job], [prior]) == [job]


def test_resume_keeps_classified_model_malformed_argument_row() -> None:
    job = {
        "scenario_slug": "power_grid/foo",
        "model": "gpt-5",
        "seed": 42,
        "scenario_signature": "sig-a",
        "temperature": 0.0,
        "evaluation_implementation_fingerprint": (
            mod.EVALUATION_IMPLEMENTATION_FINGERPRINT
        ),
    }
    prior = {
        **job,
        "status": "ok",
        "trajectory_summary": {
            "llm": {
                "llm_calls_failed": 0,
                "tool_argument_parse_failures": 1,
                "tool_argument_truncation_failures": 0,
                "tool_argument_parse_classification_version": 1,
                "fallback_wait_ratio": 0.0,
            }
        },
    }

    assert mod._filter_pending_jobs([job], [prior]) == []


def test_formal_row_keeps_model_output_and_argument_failures_as_scores() -> None:
    row = _formally_eligible_protocol21_row()
    row["trajectory_summary"]["llm"] = {
        "provider_models": ["requested-model"],
        "llm_calls_failed": 0,
        "tool_argument_parse_failures": 1,
        "tool_argument_truncation_failures": 0,
        "tool_argument_parse_classification_version": 1,
        "fallback_wait_ratio": 0.0,
    }

    eligible, reasons = mod._formal_row_eligibility(
        row,
        required_suite_hash="suite-a",
    )
    assert eligible is True
    assert reasons == []

    row["trajectory_summary"]["llm"]["tool_argument_truncation_failures"] = 1
    eligible, reasons = mod._formal_row_eligibility(
        row,
        required_suite_hash="suite-a",
    )
    assert eligible is True
    assert reasons == []

    row["trajectory_summary"]["llm"]["provider_output_truncation_count"] = 1
    eligible, reasons = mod._formal_row_eligibility(
        row,
        required_suite_hash="suite-a",
    )
    assert eligible is True
    assert reasons == []


def test_formal_row_eligibility_accepts_persistent_interaction_mode() -> None:
    row = _formally_eligible_protocol21_row()
    row["interaction_mode"] = "logical_persistent"

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is True
    assert reasons == []


def test_formal_row_eligibility_rejects_missing_interaction_mode() -> None:
    row = _formally_eligible_protocol21_row()
    row.pop("interaction_mode", None)

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is False
    assert "formal_row_interaction_mode_unsupported" in reasons


def test_formal_row_eligibility_rejects_event_contract_violations() -> None:
    row = _formally_eligible_protocol21_row()
    row["trajectory_summary"]["event_contract"] = {
        "schema_version": "1.0",
        "violation_count": 1,
    }

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is False
    assert "event_decision_contract_violation" in reasons

    row = _formally_eligible_protocol21_row()
    row["trajectory_summary"]["event_contract"]["schema_version"] = "0.9"
    eligible, reasons = mod._formal_row_eligibility(row)
    assert eligible is False
    assert "event_decision_contract_version_mismatch" in reasons


def test_formal_row_eligibility_requires_bound_provider_audit() -> None:
    missing = _formally_eligible_protocol21_row()
    missing["trajectory_summary"].pop("provider_audit_artifact")
    invalid = _formally_eligible_protocol21_row()
    invalid["trajectory_summary"]["provider_audit_artifact"]["event_count"] = 0

    assert "provider_audit_artifact_missing" in mod._formal_row_eligibility(missing)[1]
    assert "provider_audit_artifact_invalid" in mod._formal_row_eligibility(invalid)[1]


@pytest.mark.parametrize(
    ("provider_models", "reason"),
    [
        ([], "provider_model_identity_missing"),
        (["replacement-model"], "provider_model_identity_mismatch"),
        (
            ["requested-model", "replacement-model"],
            "provider_model_identity_mismatch",
        ),
    ],
)
def test_formal_row_requires_exact_provider_model_identity(
    provider_models: list[str],
    reason: str,
) -> None:
    row = _formally_eligible_protocol21_row()
    row["agent_treatment_sha256"] = "a" * 64
    row["trajectory_summary"]["llm"]["provider_models"] = provider_models

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is False
    assert reason in reasons


def test_formal_row_allows_same_model_openrouter_provider_failover() -> None:
    row = _formally_eligible_protocol21_row()
    row["agent_treatment_sha256"] = "a" * 64
    row["trajectory_summary"]["llm"].update(
        {
            "provider_models": ["requested-model"],
            "provider_response_ids": ["generation-a", "generation-b"],
        }
    )

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is True
    assert reasons == []


@pytest.mark.parametrize(
    ("second_closure", "second_models", "reason"),
    [
        ("missing", [], "provider_model_identity_missing"),
        (
            "mismatch",
            ["replacement-model"],
            "provider_model_identity_mismatch",
        ),
    ],
)
def test_formal_row_checks_every_logical_provider_response_identity(
    second_closure: str,
    second_models: list[str],
    reason: str,
) -> None:
    row = _formally_eligible_protocol21_row()
    row["agent_treatment_sha256"] = "a" * 64
    llm = row["trajectory_summary"]["llm"]
    llm["provider_models"] = ["requested-model"]
    llm["provider_model_identity_records"] = [
        {
            "schema_version": "provider_model_identity_closure_v1",
            "request_sequence": 1,
            "requested_model": "requested-model",
            "observed_models": ["requested-model"],
            "closure": "exact",
        },
        {
            "schema_version": "provider_model_identity_closure_v1",
            "request_sequence": 2,
            "requested_model": "requested-model",
            "observed_models": second_models,
            "closure": second_closure,
        },
    ]
    llm.update(
        {
            "provider_model_identity_request_count": 2,
            "provider_model_identity_closed_count": 2,
            "provider_model_identity_exact_count": 1,
            "provider_model_identity_missing_count": int(second_closure == "missing"),
            "provider_model_identity_mismatch_count": int(second_closure == "mismatch"),
            "provider_model_identity_failed_request_count": 0,
        }
    )

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is False
    assert reason in reasons


def test_formal_row_byte_verification_rejects_deleted_or_tampered_sidecar(
    tmp_path: Path,
) -> None:
    row = _formally_eligible_protocol21_row()
    treatment_hash = "b" * 64
    trajectory_dir = tmp_path / f"treatment-{treatment_hash}"
    trajectory_dir.mkdir()
    prefix = trajectory_dir / "episode"
    provider_payload = (
        b'{"record_kind":"provider_request"}\n{"record_kind":"provider_response"}\n'
    )
    semantic_payload = b'{"role":"system","content":"contract"}\n'
    trajectory_payload = b'{"tick":0,"evidence_ids":["ev-1"]}\n'
    evidence_payload = b'{"evidence_id":"ev-1","tick":0}\n'
    provider_path = Path(f"{prefix}.provider_audit.jsonl")
    semantic_path = Path(f"{prefix}.semantic_ledger.jsonl")
    trajectory_path = Path(f"{prefix}.trajectory.jsonl")
    evidence_path = Path(f"{prefix}.evidence.jsonl")
    provider_path.write_bytes(provider_payload)
    semantic_path.write_bytes(semantic_payload)
    trajectory_path.write_bytes(trajectory_payload)
    evidence_path.write_bytes(evidence_payload)
    row["agent_treatment_sha256"] = treatment_hash
    row["trajectory_summary"]["trajectory_path"] = str(prefix)
    row["trajectory_summary"]["provider_audit_artifact"] = {
        "schema_version": "provider_interaction_audit_v1",
        "path": str(provider_path),
        "sha256": hashlib.sha256(provider_payload).hexdigest(),
        "event_count": 2,
    }
    row["trajectory_summary"]["semantic_ledger_artifact"] = {
        "schema_version": "semantic_session_ledger_v1",
        "path": str(semantic_path),
        "sha256": hashlib.sha256(semantic_payload).hexdigest(),
        "event_count": 1,
    }
    row["trajectory_summary"]["trajectory_artifact"] = {
        "schema_version": "episode_trajectory_jsonl_v1",
        "path": str(trajectory_path),
        "sha256": hashlib.sha256(trajectory_payload).hexdigest(),
        "event_count": 1,
        "byte_count": len(trajectory_payload),
    }
    row["trajectory_summary"]["evidence_ledger_artifact"] = {
        "schema_version": "evidence_ledger_jsonl_v1",
        "path": str(evidence_path),
        "sha256": hashlib.sha256(evidence_payload).hexdigest(),
        "event_count": 1,
        "byte_count": len(evidence_payload),
    }

    assert mod._formal_row_eligibility(row, verify_artifact_bytes=True)[0] is True

    provider_path.write_bytes(provider_payload + b'{"tampered":true}\n')
    assert (
        "provider_audit_artifact_sha256_mismatch"
        in mod._formal_row_eligibility(row, verify_artifact_bytes=True)[1]
    )
    provider_path.unlink()
    assert (
        "provider_audit_artifact_unreadable"
        in mod._formal_row_eligibility(row, verify_artifact_bytes=True)[1]
    )

    provider_path.write_bytes(provider_payload)
    trajectory_path.write_bytes(trajectory_payload + b'{"tampered":true}\n')
    assert (
        "trajectory_artifact_sha256_mismatch"
        in mod._formal_row_eligibility(row, verify_artifact_bytes=True)[1]
    )

    trajectory_path.write_bytes(trajectory_payload)
    evidence_path.unlink()
    assert (
        "evidence_ledger_artifact_unreadable"
        in mod._formal_row_eligibility(row, verify_artifact_bytes=True)[1]
    )


def test_persistent_formal_row_requires_bounded_memory_and_nonempty_ledger(
    tmp_path: Path,
) -> None:
    row = _bind_formal_artifacts(
        _formally_eligible_protocol21_row(), tmp_path, key="persistent"
    )
    row["interaction_mode"] = "logical_persistent"
    row["agent_config"] = {
        "config": {
            "interaction_mode": "logical_persistent",
            "persistent_memory_max_items": 8,
        }
    }
    memory = {
        "schema_version": "persistent_working_memory_v2",
        "unresolved_alarms": [],
        "open_obligations": [],
        "confirmed_facts": [],
        "active_commitments": [],
        "forecast_ledger": [],
        "state_trends": [],
        "last_updated_tick": 0,
    }
    row["structured_memory"] = memory
    summary = row["trajectory_summary"]
    summary["structured_memory"] = memory
    summary["llm"].update(
        {
            "session_ledger_events": 1,
            "structured_memory_bytes": len(
                json.dumps(
                    memory,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
        }
    )

    assert mod._formal_row_eligibility(row, verify_artifact_bytes=True)[0] is True

    summary["semantic_ledger_artifact"]["event_count"] = 0
    eligible, reasons = mod._formal_row_eligibility(row)
    assert eligible is False
    assert "semantic_ledger_artifact_missing_or_invalid" in reasons

    summary["semantic_ledger_artifact"]["event_count"] = 1
    summary["structured_memory"] = {"schema_version": "wrong"}
    eligible, reasons = mod._formal_row_eligibility(row)
    assert eligible is False
    assert "structured_memory_missing_or_invalid" in reasons


def test_formal_row_eligibility_rejects_v17_and_suite_hash_mismatch() -> None:
    base = _formally_eligible_protocol21_row()
    base["evaluation_implementation_fingerprint"] = (
        "protocol-2.0-v18-fail-closed:prompt-strict"
    )
    base["trajectory_summary"]["llm"] = {
        "llm_calls_failed": 0,
        "tool_argument_parse_failures": 0,
        "fallback_wait_ratio": 0.0,
    }

    eligible, reasons = mod._formal_row_eligibility(base, required_suite_hash="suite-a")
    assert eligible is True
    assert reasons == []

    missing_semantics = {
        **base,
        "trajectory_summary": dict(base["trajectory_summary"]),
    }
    missing_semantics["trajectory_summary"].pop("tool_semantic_coverage")
    eligible, reasons = mod._formal_row_eligibility(missing_semantics)
    assert eligible is False
    assert "tool_semantic_coverage_missing" in reasons

    missing_surface = {
        **base,
        "trajectory_summary": dict(base["trajectory_summary"]),
    }
    missing_surface["trajectory_summary"].pop("tool_surface_contract")
    eligible, reasons = mod._formal_row_eligibility(missing_surface)
    assert eligible is False
    assert "tool_surface_contract_missing" in reasons

    stale = {
        **base,
        "evaluation_protocol": {
            "version": mod.EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": "stale-v5",
        },
        "score": {"scoring_version": "0.10.0"},
    }
    eligible, reasons = mod._formal_row_eligibility(stale)
    assert eligible is False
    assert "evaluation_implementation_fingerprint_mismatch" in reasons
    assert "scoring_version_mismatch" in reasons

    eligible, reasons = mod._formal_row_eligibility(base, required_suite_hash="suite-b")
    assert eligible is False
    assert "suite_manifest_mismatch" in reasons

    eligible, reasons = mod._formal_row_eligibility(
        base,
        required_suite_hash="suite-a",
        required_implementation_tree_sha256="tree-a",
    )
    assert eligible is False
    assert "implementation_tree_mismatch" in reasons

    current_tree = {**base, "implementation_tree_sha256": "tree-a"}
    eligible, reasons = mod._formal_row_eligibility(
        current_tree,
        required_suite_hash="suite-a",
        required_implementation_tree_sha256="tree-a",
    )
    assert eligible is True
    assert reasons == []

    v17 = {
        **base,
        "evaluation_protocol": {
            "version": "1.4",
            "implementation_fingerprint": (
                "protocol-1.4-meta-tools-v17-bounded-decision-waves"
            ),
        },
        "evaluation_implementation_fingerprint": (
            "protocol-1.4-meta-tools-v17-bounded-decision-waves:prompt-strict"
        ),
    }
    eligible, reasons = mod._formal_row_eligibility(v17)
    assert eligible is False
    assert "diagnostic_protocol_v17" in reasons


@pytest.mark.parametrize(
    ("construct_contract", "expected_reason"),
    [
        (None, "construct_contract_missing"),
        ("legacy_construct.v0", "construct_contract_mismatch"),
    ],
)
def test_required_suite_formal_row_requires_operational_agency_contract(
    construct_contract: str | None,
    expected_reason: str,
) -> None:
    row = _formally_eligible_protocol21_row()
    if construct_contract is None:
        row["evaluation_protocol"].pop("construct_contract")
    else:
        row["evaluation_protocol"]["construct_contract"] = construct_contract

    eligible, reasons = mod._formal_row_eligibility(
        row,
        required_suite_hash="suite-a",
    )

    assert eligible is False
    assert expected_reason in reasons


def test_treatment_bound_resume_rejects_incomplete_formal_row_without_suite_arg(
    tmp_path: Path,
) -> None:
    row = _bind_formal_artifacts(
        _formally_eligible_protocol21_row(), tmp_path, key="old-r10"
    )
    row["evaluation_protocol"].pop("construct_contract")
    row["source_denominator_key"] = None
    row["case_ledger"] = None

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is False
    assert "construct_contract_missing" in reasons
    assert "source_denominator_key_missing" in reasons
    assert "case_ledger_missing_or_invalid" in reasons
    assert mod._row_is_clean_for_resume(row) is False


def test_formal_row_rejects_causal_response_without_observation_dependency() -> None:
    row = _formally_eligible_protocol21_row()
    row["trajectory_summary"]["event_response_records"] = [
        {
            "event_id": "surprise-1",
            "response_status": "causal",
            "first_observed_tick": None,
            "action_consumes_evidence_ids": [],
        }
    ]

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is False
    assert "causal_response_first_observed_missing" in reasons
    assert "causal_response_evidence_dependency_missing" in reasons


def test_protocol2_eligibility_is_row_bound_and_fails_closed() -> None:
    eligibility = {
        "suite_blocked": False,
        "diagnostic_cells": [
            {
                "family": "fam",
                "difficulty_mode": "mode",
                "difficulty_level": "high",
            }
        ],
        "uninformative_cells": [],
        "wait_dominant_cells": [],
    }
    row = {
        "family": "fam",
        "difficulty_mode": "mode",
        "difficulty_level": "high",
        "evaluation_protocol": {"version": "2.0"},
        "suite_eligibility": eligibility,
        "suite_eligibility_sha256": mod._canonical_json_sha256(eligibility),
    }

    assert mod._is_discriminative(row) is False

    tampered = {
        **row,
        "suite_eligibility": {**eligibility, "diagnostic_cells": []},
    }
    assert mod._is_discriminative(tampered) is False
    eligible, reasons = mod._formal_row_eligibility(
        {
            **tampered,
            "status": "ok",
            "suite_manifest_sha256": "suite-a",
            "trajectory_summary": {"llm": {}},
        }
    )
    assert eligible is False
    assert "suite_eligibility_hash_mismatch" in reasons


def test_protocol2_formal_row_rejects_terminal_integrity_failure() -> None:
    suite_eligibility = {
        "suite_blocked": False,
        "diagnostic_cells": [],
        "uninformative_cells": [],
        "wait_dominant_cells": [],
    }
    row = {
        "status": "ok",
        "interaction_mode": "logical_stateless",
        "evaluation_protocol": {
            "version": mod.EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": (mod.EVALUATION_IMPLEMENTATION_FINGERPRINT),
        },
        "score": {"scoring_version": mod.SCORING_VERSION},
        "source_denominator_key": "fixture:source-a",
        "case_ledger": {
            "schema_version": "0.1",
            "source_denominator_key": "fixture:source-a",
        },
        "suite_manifest_sha256": "suite-a",
        "suite_eligibility": suite_eligibility,
        "suite_eligibility_sha256": mod._canonical_json_sha256(suite_eligibility),
        "trajectory_summary": {
            "llm": {},
            "terminal_integrity": {
                "release_ready": False,
                "unanswered_interrupt_reasons": ["visible_event"],
            },
        },
    }

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is False
    assert "terminal_integrity_failure" in reasons


def test_protocol2_formal_row_rejects_unknown_tool_semantics() -> None:
    suite_eligibility = {
        "suite_blocked": False,
        "diagnostic_cells": [],
        "uninformative_cells": [],
        "wait_dominant_cells": [],
    }
    row = {
        "status": "ok",
        "evaluation_protocol": {
            "version": mod.EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": (mod.EVALUATION_IMPLEMENTATION_FINGERPRINT),
        },
        "score": {"scoring_version": mod.SCORING_VERSION},
        "suite_manifest_sha256": "suite-a",
        "suite_eligibility": suite_eligibility,
        "suite_eligibility_sha256": mod._canonical_json_sha256(suite_eligibility),
        "trajectory_summary": {
            "llm": {},
            "terminal_integrity": {"release_ready": True},
            "event_contract": {"schema_version": "1.0", "violation_count": 0},
            "provider_audit_artifact": _provider_audit_artifact(),
            "tool_semantic_coverage": {
                "covered": False,
                "unknown_tool_names": ["unregistered_control"],
                "unclassified_tool_names": [],
                "explicit_semantic_roles_complete": True,
                "missing_explicit_semantic_role_names": [],
                "native_targets_complete": True,
                "missing_native_target_kind_names": [],
                "state_changing_actuators_complete": True,
                "missing_actuator_family_names": [],
            },
        },
    }

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is False
    assert "unknown_tool_semantics" in reasons

    inconsistent = copy.deepcopy(row)
    inconsistent["trajectory_summary"]["tool_semantic_coverage"] = {
        "covered": 1,
        "unknown_tool_names": [],
        "unclassified_tool_names": [],
        "explicit_semantic_roles_complete": True,
        "missing_explicit_semantic_role_names": [],
        "native_targets_complete": True,
        "missing_native_target_kind_names": [],
        "state_changing_actuators_complete": True,
        "missing_actuator_family_names": [],
    }

    eligible, reasons = mod._formal_row_eligibility(inconsistent)

    assert eligible is False
    assert "tool_semantic_coverage_inconsistent" in reasons

    missing_actuator = copy.deepcopy(row)
    missing_actuator["trajectory_summary"]["tool_semantic_coverage"] = {
        "covered": True,
        "unknown_tool_names": [],
        "unclassified_tool_names": [],
        "explicit_semantic_roles_complete": True,
        "missing_explicit_semantic_role_names": [],
        "native_targets_complete": True,
        "missing_native_target_kind_names": [],
        "state_changing_actuators_complete": False,
        "missing_actuator_family_names": ["control"],
    }

    eligible, reasons = mod._formal_row_eligibility(missing_actuator)

    assert eligible is False
    assert "tool_semantic_coverage_inconsistent" in reasons


def test_operational_agency_contract_accepts_zero_credit_profile() -> None:
    suite_eligibility = {
        "suite_blocked": False,
        "diagnostic_cells": [],
        "uninformative_cells": [],
        "wait_dominant_cells": [],
    }
    row = {
        "status": "ok",
        "interaction_mode": "logical_stateless",
        "evaluation_protocol": {
            "version": mod.EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": (mod.EVALUATION_IMPLEMENTATION_FINGERPRINT),
            "construct_contract": "operational_agency.v1",
        },
        "score": {"scoring_version": mod.SCORING_VERSION},
        "source_denominator_key": "fixture:source-a",
        "case_ledger": {
            "schema_version": "0.1",
            "source_denominator_key": "fixture:source-a",
        },
        "suite_manifest_sha256": "suite-a",
        "suite_eligibility": suite_eligibility,
        "suite_eligibility_sha256": mod._canonical_json_sha256(suite_eligibility),
        "trajectory_summary": {
            "llm": {},
            "terminal_integrity": {"release_ready": True},
            "event_contract": {"schema_version": "1.0", "violation_count": 0},
            "provider_audit_artifact": _provider_audit_artifact(),
            "trajectory_artifact": {
                "schema_version": "episode_trajectory_jsonl_v1",
                "path": "/tmp/zero-credit.trajectory.jsonl",
                "sha256": "d" * 64,
                "event_count": 1,
                "byte_count": 1,
            },
            "evidence_ledger_artifact": {
                "schema_version": "evidence_ledger_jsonl_v1",
                "path": "/tmp/zero-credit.evidence.jsonl",
                "sha256": "e" * 64,
                "event_count": 1,
                "byte_count": 1,
            },
            "tool_semantic_coverage": {
                "covered": True,
                "unknown_tool_names": [],
                "unclassified_tool_names": [],
                "explicit_semantic_roles_complete": True,
                "missing_explicit_semantic_role_names": [],
                "native_targets_complete": True,
                "missing_native_target_kind_names": [],
                "state_changing_actuators_complete": True,
                "missing_actuator_family_names": [],
            },
            "tool_surface_contract": _tool_surface_contract(),
            "event_response_records": [],
            "operational_agency_valid_evidence_ids": [],
            "operational_agency_profile": evaluate_operational_agency(
                [],
                valid_evidence_ids=set(),
                masked_replay_by_call_id={},
            ),
        },
        "counterfactual": {
            "per_action": [],
            "per_action_capped": False,
            "per_action_status": "complete",
            "per_action_expected": 0,
            "per_action_attempted": 0,
            "per_action_completed": 0,
            "per_action_failures": [],
            "per_action_groups": [],
            "per_action_group_status": "complete",
            "per_action_group_expected": 0,
            "per_action_group_attempted": 0,
            "per_action_group_completed": 0,
            "per_action_group_failures": [],
        },
    }

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is True
    assert reasons == []


def test_operational_agency_contract_rejects_capped_attribution() -> None:
    row = _formally_eligible_protocol21_row()
    row["counterfactual"]["per_action_capped"] = True

    eligible, reasons = mod._formal_row_eligibility(
        row,
        required_suite_hash="suite-a",
    )

    assert eligible is False
    assert "construct_attribution_incomplete" in reasons


def test_operational_agency_contract_rejects_inconsistent_profile_fields() -> None:
    suite_eligibility = {
        "suite_blocked": False,
        "diagnostic_cells": [],
        "uninformative_cells": [],
        "wait_dominant_cells": [],
    }
    dimensions = {
        name: {
            "applicable": False,
            "score": None,
            "support_count": 0,
            "evidence_ids": [],
            "reason": "not_applicable",
        }
        for name in mod.OPERATIONAL_AGENCY_DIMENSIONS
    }
    dimensions["outcome_influence"] = {
        "applicable": True,
        "score": 50.0,
        "support_count": 1,
        "evidence_ids": ["effect-1"],
        "reason": None,
    }
    row = {
        "status": "ok",
        "evaluation_protocol": {
            "version": mod.EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": (mod.EVALUATION_IMPLEMENTATION_FINGERPRINT),
            "construct_contract": "operational_agency.v1",
        },
        "score": {"scoring_version": mod.SCORING_VERSION},
        "suite_manifest_sha256": "suite-a",
        "suite_eligibility": suite_eligibility,
        "suite_eligibility_sha256": mod._canonical_json_sha256(suite_eligibility),
        "trajectory_summary": {
            "llm": {},
            "terminal_integrity": {"release_ready": True},
            "tool_semantic_coverage": {
                "covered": True,
                "unknown_tool_names": [],
                "unclassified_tool_names": [],
                "explicit_semantic_roles_complete": True,
                "missing_explicit_semantic_role_names": [],
                "native_targets_complete": True,
                "missing_native_target_kind_names": [],
                "state_changing_actuators_complete": True,
                "missing_actuator_family_names": [],
            },
            "event_response_records": [{"backend_effect_evidence_ids": ["effect-1"]}],
            "operational_agency_valid_evidence_ids": ["effect-1"],
            "operational_agency_profile": {
                "schema_version": "operational_agency_profile_v1",
                "diagnostic_only": True,
                "headline_score_included": False,
                "runtime_binding_verified": True,
                "runtime_evidence_binding_verified": True,
                "masked_replay_binding_verified": True,
                "event_response_record_count": 1,
                "causal_record_count": 1,
                "dimensions": dimensions,
            },
        },
        "counterfactual": {
            "per_action": [],
            "per_action_capped": False,
            "per_action_status": "complete",
            "per_action_expected": 0,
            "per_action_attempted": 0,
            "per_action_completed": 0,
            "per_action_failures": [],
            "per_action_groups": [],
            "per_action_group_status": "complete",
            "per_action_group_expected": 0,
            "per_action_group_attempted": 0,
            "per_action_group_completed": 0,
            "per_action_group_failures": [],
        },
    }

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is False
    assert "construct_evidence_inconsistent" in reasons

    inconsistent = copy.deepcopy(row)
    inconsistent["trajectory_summary"]["operational_agency_profile"][
        "causal_record_count"
    ] = 2
    inconsistent["trajectory_summary"]["operational_agency_profile"]["dimensions"] = {}

    eligible, reasons = mod._formal_row_eligibility(inconsistent)

    assert eligible is False
    assert "construct_evidence_inconsistent" in reasons

    unbound = copy.deepcopy(row)
    unbound["trajectory_summary"]["operational_agency_profile"][
        "runtime_binding_verified"
    ] = False

    eligible, reasons = mod._formal_row_eligibility(unbound)

    assert eligible is False
    assert "construct_evidence_inconsistent" in reasons

    malformed = copy.deepcopy(row)
    malformed["trajectory_summary"]["operational_agency_profile"][
        "causal_record_count"
    ] = "not-an-int"

    eligible, reasons = mod._formal_row_eligibility(malformed)

    assert eligible is False
    assert "construct_evidence_inconsistent" in reasons


def test_blocked_candidate_row_is_resumable_but_not_formal() -> None:
    suite_eligibility = {
        "suite_blocked": True,
        "reason": {"code": "candidate_not_leaderboard_eligible"},
        "diagnostic_cells": [],
        "uninformative_cells": [],
        "wait_dominant_cells": [],
    }
    row = {
        "status": "ok",
        "evaluation_protocol": {
            "version": mod.EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": (mod.EVALUATION_IMPLEMENTATION_FINGERPRINT),
        },
        "score": {"scoring_version": mod.SCORING_VERSION},
        "suite_manifest_sha256": "suite-a",
        "suite_eligibility": suite_eligibility,
        "suite_eligibility_sha256": mod._canonical_json_sha256(suite_eligibility),
        "trajectory_summary": {
            "llm": {},
            "terminal_integrity": {"release_ready": True},
            "tool_semantic_coverage": {
                "covered": True,
                "unknown_tool_names": [],
                "unclassified_tool_names": [],
                "explicit_semantic_roles_complete": True,
                "missing_explicit_semantic_role_names": [],
                "native_targets_complete": True,
                "missing_native_target_kind_names": [],
                "state_changing_actuators_complete": True,
                "missing_actuator_family_names": [],
            },
        },
    }

    assert mod._formal_row_eligibility(row)[0] is False
    assert mod._row_is_clean_for_resume(row) is True


def test_formal_row_eligibility_labels_prompt_budget_separately_from_provider() -> None:
    row = _formally_eligible_protocol21_row()
    row["trajectory_summary"]["llm"] = {
        "llm_calls_failed": 2,
        "failed_tick_log": [
            {
                "tick": 2,
                "exc_type": "ValueError",
                "exc_msg_head": (
                    "mandatory prompt state exceeds max_chars; refusing to omit "
                    "action-critical fields (8401 > 8000)"
                ),
            },
            {
                "tick": 3,
                "reason": "prompt_budget_exceeded",
                "exc_msg_head": "mandatory prompt state exceeds max_chars",
            },
        ],
    }
    eligible, reasons = mod._formal_row_eligibility(row)
    assert eligible is False
    assert "prompt_budget_exceeded" in reasons
    assert "provider_call_failure" not in reasons
    assert mod._row_is_clean_for_resume(row) is False


def test_formal_row_rejects_provider_tool_call_fallback() -> None:
    row = _formally_eligible_protocol21_row()
    llm = row["trajectory_summary"]["llm"]
    llm.update(
        {
            "llm_calls_failed": 0,
            "provider_tool_call_failures": 1,
            "fallback_without_tools_count": 1,
        }
    )

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is False
    assert "provider_call_failure" in reasons
    assert mod._row_is_clean_for_resume(row) is False


def test_formal_row_rejects_failed_provider_identity_request() -> None:
    row = _formally_eligible_protocol21_row()
    llm = row["trajectory_summary"]["llm"]
    llm["provider_model_identity_records"][0].update(
        {"observed_models": [], "closure": "request_failed"}
    )
    llm.update(
        {
            "llm_calls_failed": 0,
            "provider_model_identity_exact_count": 0,
            "provider_model_identity_failed_request_count": 1,
        }
    )

    eligible, reasons = mod._formal_row_eligibility(row)

    assert eligible is False
    assert "provider_call_failure" in reasons
    assert mod._row_is_clean_for_resume(row) is False


def test_coverage_summary_reports_execution_completed_for_blocked_ok_rows() -> None:
    row = _formally_eligible_protocol21_row()
    row["model"] = "hy3-ioa"
    row["scenario_slug"] = "fam/a"
    row["seed"] = 42
    row["suite_eligibility"]["suite_blocked"] = True
    row["suite_eligibility_sha256"] = mod._canonical_json_sha256(
        row["suite_eligibility"]
    )
    cov = mod._coverage_summary(
        [row],
        configured_models=["hy3-ioa"],
        configured_seeds=[42],
        n_scenarios=1,
    )
    assert cov["per_model_realized"]["hy3-ioa"] == 0
    assert cov["per_model_execution_realized"]["hy3-ioa"] == 1
    assert cov["per_model_execution_coverage"]["hy3-ioa"] == 1.0


def test_leaderboard_excludes_contaminated_status_ok_rows() -> None:
    clean = {
        **_row("fam/clean", "model-a", 42, 10.0),
        "trajectory_summary": {
            "llm": {
                "llm_calls_failed": 0,
                "tool_argument_parse_failures": 0,
                "fallback_wait_ratio": 0.0,
            }
        },
    }
    contaminated = {
        **_row("fam/bad", "model-a", 42, 1000.0),
        "trajectory_summary": {
            "llm": {
                "llm_calls_failed": 0,
                "tool_argument_parse_failures": 1,
                "fallback_wait_ratio": 0.0,
            }
        },
    }

    board = mod._leaderboard_from_rows([clean, contaminated], "fixed_all_dimensions")

    assert board[0]["n_episodes"] == 1
    assert board[0]["mean"] == 10.0


def test_batch_state_keeps_malformed_model_output_as_capability_result() -> None:
    results = [
        {
            **_row("fam/a", "model-a", 42, 10.0),
            "trajectory_summary": {
                "llm": {
                    "llm_calls_failed": 0,
                    "tool_argument_parse_failures": 1,
                    "tool_argument_parse_classification_version": 1,
                    "fallback_wait_ratio": 0.0,
                }
            },
        }
    ]
    coverage = mod._coverage_summary(
        results,
        configured_models=["model-a"],
        configured_seeds=[42],
        n_scenarios=1,
    )

    state = mod._batch_state(
        coverage=coverage,
        results=results,
        log_audit_report=None,
    )

    assert state["batch_state"] == mod.BATCH_STATE_FINAL
    assert state["n_model_output_failure_episodes"] == 1


def test_resume_retries_rows_from_an_old_evaluation_implementation() -> None:
    job = {
        "scenario_slug": "power_grid/foo",
        "model": "hy3-ioa",
        "seed": 42,
        "scenario_signature": "sig-a",
        "temperature": 0.0,
        "evaluation_implementation_fingerprint": (
            mod.EVALUATION_IMPLEMENTATION_FINGERPRINT
        ),
    }
    prior = {
        **job,
        "status": "ok",
        "evaluation_implementation_fingerprint": "protocol-1.4-old-semantics",
    }

    assert mod._filter_pending_jobs([job], [prior]) == [job]


def test_run_semantics_fingerprint_separates_strict_and_debug_prompts() -> None:
    assert mod._run_semantics_fingerprint("strict") != mod._run_semantics_fingerprint(
        "debug"
    )


def test_agent_treatment_hash_covers_session_and_provider_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = mod.LLMConfig(
        provider="openai_compatible",
        model="model",
        base_url="https://user:secret@example.com/v1?token=secret&deployment=blue",
        interaction_mode="logical_stateless",
    )
    persistent = mod.LLMConfig(
        provider="openai_compatible",
        model="model",
        base_url="https://example.com/v1?token=different&deployment=blue",
        interaction_mode="logical_persistent",
    )
    same_public_route = mod.LLMConfig(
        provider="openai_compatible",
        model="model",
        base_url="https://other:rotated@example.com/v1?token=rotated&deployment=blue",
        interaction_mode="logical_stateless",
    )
    different_behavior_route = mod.LLMConfig(
        provider="openai_compatible",
        model="model",
        base_url="https://example.com/v1?token=secret&deployment=green",
        interaction_mode="logical_stateless",
    )
    different_memory_bound = mod.LLMConfig(
        provider="openai_compatible",
        model="model",
        base_url="https://example.com/v1",
        interaction_mode="logical_stateless",
        persistent_memory_max_items=7,
    )

    assert mod._agent_treatment_sha256(base) == mod._agent_treatment_sha256(
        same_public_route
    )
    assert mod._agent_treatment_sha256(base) != mod._agent_treatment_sha256(
        different_behavior_route
    )
    assert mod._agent_treatment_sha256(base) != mod._agent_treatment_sha256(persistent)
    assert mod._agent_treatment_sha256(base) != mod._agent_treatment_sha256(
        different_memory_bound
    )
    bounded = mod.LLMConfig(
        provider="openai_compatible",
        model="model",
        interaction_mode="logical_stateless",
        model_context_window_tokens=128_000,
        model_max_output_tokens=16_384,
    )
    differently_bounded = mod.LLMConfig(
        provider="openai_compatible",
        model="model",
        interaction_mode="logical_stateless",
        model_context_window_tokens=64_000,
        model_max_output_tokens=8_192,
    )
    assert mod._agent_treatment_sha256(bounded) != mod._agent_treatment_sha256(
        differently_bounded
    )
    different_header_value = mod.LLMConfig(
        provider="openai_compatible",
        model="model",
        base_url="https://user:secret@example.com/v1?token=secret&deployment=blue",
        interaction_mode="logical_stateless",
        extra_headers={"X-Route": "canary"},
    )
    assert mod._agent_treatment_sha256(base) != mod._agent_treatment_sha256(
        different_header_value
    )
    rotated_secret_header = mod.LLMConfig(
        provider="openai_compatible",
        model="model",
        base_url="https://example.com/v1?token=rotated&deployment=blue",
        interaction_mode="logical_stateless",
        extra_headers={"Authorization": "Bearer rotated-secret"},
    )
    original_secret_header = mod.LLMConfig(
        provider="openai_compatible",
        model="model",
        base_url="https://example.com/v1?token=secret&deployment=blue",
        interaction_mode="logical_stateless",
        extra_headers={"Authorization": "Bearer original-secret"},
    )
    assert mod._agent_treatment_sha256(
        original_secret_header
    ) == mod._agent_treatment_sha256(rotated_secret_header)
    rotated_cookie = mod.LLMConfig(
        provider="openai_compatible",
        model="model",
        base_url="https://example.com/v1?token=rotated&deployment=blue",
        interaction_mode="logical_stateless",
        extra_headers={"Cookie": "session=rotated"},
    )
    original_cookie = mod.LLMConfig(
        provider="openai_compatible",
        model="model",
        base_url="https://example.com/v1?token=secret&deployment=blue",
        interaction_mode="logical_stateless",
        extra_headers={"Cookie": "session=original"},
    )
    assert mod._agent_treatment_sha256(original_cookie) == mod._agent_treatment_sha256(
        rotated_cookie
    )
    wakeup_policy = {
        "session_start": True,
        "typed_actionable_events": True,
        "agent_scheduled_reviews": True,
        "harness_periodic_supervisory_scan": False,
        "unknown_events_actionable": False,
    }
    monkeypatch.setattr(
        mod, "CANONICAL_WAKEUP_POLICY", wakeup_policy, raising=False
    )
    canonical_wakeup_hash = mod._agent_treatment_sha256(base)
    monkeypatch.setattr(
        mod,
        "CANONICAL_WAKEUP_POLICY",
        {**wakeup_policy, "harness_periodic_supervisory_scan": True},
    )
    assert mod._agent_treatment_sha256(base) != canonical_wakeup_hash


def test_formal_logical_treatment_hash_binds_release_runtime_and_implementation() -> None:
    profile_hashes = {"hy3-ioa": "a" * 64}
    binding = {
        "release_id": "operate_v0_61_0",
        "manifest_sha256": "b" * 64,
        "release_tooling_sha256": "c" * 64,
        "readiness_sha256": "d" * 64,
        "core_release_pipeline_sha256": "e" * 64,
        "backend_runtime_closure_identity_sha256": "f" * 64,
    }

    baseline = mod._formal_agent_treatment_hashes(
        profile_hashes,
        formal_manifest_binding=binding,
        implementation_tree_sha256="1" * 64,
    )

    for field, replacement in (
        ("release_id", "operate_v0_62_0"),
        ("manifest_sha256", "2" * 64),
        ("readiness_sha256", "3" * 64),
        ("core_release_pipeline_sha256", "4" * 64),
        ("backend_runtime_closure_identity_sha256", "5" * 64),
    ):
        changed = mod._formal_agent_treatment_hashes(
            profile_hashes,
            formal_manifest_binding={**binding, field: replacement},
            implementation_tree_sha256="1" * 64,
        )
        assert changed != baseline

    assert mod._formal_agent_treatment_hashes(
        profile_hashes,
        formal_manifest_binding=binding,
        implementation_tree_sha256="6" * 64,
    ) != baseline
    assert profile_hashes == {"hy3-ioa": "a" * 64}


def test_effective_episode_rows_are_sorted_by_stable_identity() -> None:
    rows = [
        {
            "status": "ok",
            "scenario_slug": "scenario-b",
            "model": "model-z",
            "seed": 42,
            "scenario_signature": "sig-b",
            "temperature": 1.0,
        },
        {
            "status": "ok",
            "scenario_slug": "scenario-a",
            "model": "model-a",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
        },
    ]

    effective = mod._effective_episode_rows(rows)

    assert [row["scenario_slug"] for row in effective] == ["scenario-a", "scenario-b"]


def test_run_global_jobs_appends_results_as_each_future_completes(
    monkeypatch, tmp_path: Path
) -> None:
    jobs = [
        {"scenario_slug": "slow", "model": "gpt", "seed": 42},
        {"scenario_slug": "fast", "model": "gpt", "seed": 42},
    ]
    completions = [
        {"scenario_slug": "fast", "status": "ok"},
        {"scenario_slug": "slow", "status": "ok"},
    ]
    writes_seen_by_submit: list[list[str]] = []

    class FakeFile:
        def __init__(self) -> None:
            self.rows: list[dict[str, Any]] = []

        def write(self, line: str) -> None:
            self.rows.append(json.loads(line))

        def flush(self) -> None:
            return None

        def __enter__(self) -> FakeFile:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    class FakeFuture:
        def __init__(self, row: dict[str, Any]) -> None:
            self.row = row

        def result(self) -> dict[str, Any]:
            return self.row

    class FakePool:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers
            self.submitted: list[dict[str, Any]] = []

        def submit(self, fn: object, job: dict[str, Any]) -> FakeFuture:
            writes_seen_by_submit.append(
                [row["scenario_slug"] for row in fake_file.rows]
            )
            self.submitted.append(job)
            return FakeFuture(completions[len(self.submitted) - 1])

        def __enter__(self) -> FakePool:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    fake_file = FakeFile()
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: fake_file)
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", FakePool)
    monkeypatch.setattr("concurrent.futures.as_completed", lambda fs: fs)

    mod._run_global_jobs(jobs, tmp_path / "episodes.jsonl", "w", 4)

    terminal_rows = [row for row in fake_file.rows if row.get("status") != "in_flight"]
    assert [row["scenario_slug"] for row in terminal_rows] == ["fast", "slow"]
    assert sum(row.get("status") == "in_flight" for row in fake_file.rows) == 2


def test_run_global_jobs_bounds_submitted_futures(monkeypatch, tmp_path: Path) -> None:
    jobs = [
        {"scenario_slug": f"case-{index}", "model": "gpt", "seed": 42}
        for index in range(10)
    ]
    rows_written_before_submit: list[int] = []

    class FakeFile:
        def __init__(self) -> None:
            self.rows: list[dict[str, Any]] = []

        def write(self, line: str) -> None:
            self.rows.append(json.loads(line))

        def flush(self) -> None:
            return None

        def __enter__(self) -> FakeFile:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    class FakeFuture:
        def __init__(self, job: dict[str, Any]) -> None:
            self.job = job

        def result(self) -> dict[str, Any]:
            return {
                "scenario_slug": self.job["scenario_slug"],
                "status": "ok",
            }

    class FakePool:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def submit(self, fn: object, job: dict[str, Any]) -> FakeFuture:
            rows_written_before_submit.append(len(fake_file.rows))
            return FakeFuture(job)

        def __enter__(self) -> FakePool:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    fake_file = FakeFile()
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: fake_file)
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", FakePool)
    monkeypatch.setattr("concurrent.futures.as_completed", lambda fs: fs)

    mod._run_global_jobs(jobs, tmp_path / "episodes.jsonl", "w", 2)

    # The queue starts with four jobs, then refills one slot per completion;
    # it no longer waits for the whole submission window to drain. Each
    # submitted job also gets an in-flight checkpoint before worker dispatch.
    assert rows_written_before_submit[:4] == [1, 2, 3, 4]
    assert rows_written_before_submit[4:] == [6, 8, 10, 12, 14, 16]
    assert sum(row.get("status") == "ok" for row in fake_file.rows) == 10
    assert sum(row.get("status") == "in_flight" for row in fake_file.rows) == 10


def test_run_global_jobs_terminates_workers_on_interrupt(
    monkeypatch, tmp_path: Path
) -> None:
    terminated = False

    class FakeFile:
        def write(self, line: str) -> None:
            return None

        def flush(self) -> None:
            return None

        def __enter__(self) -> FakeFile:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    class InterruptedFuture:
        def result(self) -> dict[str, Any]:
            raise KeyboardInterrupt

        def cancel(self) -> bool:
            return True

    class FakePool:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def submit(self, fn: object, job: dict[str, Any]) -> InterruptedFuture:
            return InterruptedFuture()

        def terminate_workers(self) -> None:
            nonlocal terminated
            terminated = True

        def __enter__(self) -> FakePool:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: FakeFile())
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", FakePool)
    monkeypatch.setattr("concurrent.futures.as_completed", lambda fs: fs)

    with pytest.raises(KeyboardInterrupt):
        mod._run_global_jobs(
            [{"scenario_slug": "case", "model": "hy3-ioa", "seed": 42}],
            tmp_path / "episodes.jsonl",
            "w",
            1,
        )

    assert terminated is True


def test_scenario_identity_report_flags_signature_aliases() -> None:
    results = [
        {
            "status": "ok",
            "scenario_slug": "traffic/a",
            "scenario_id": "shared_id",
            "scenario_signature": "sig-shared",
        },
        {
            "status": "ok",
            "scenario_slug": "traffic/b",
            "scenario_id": "shared_id",
            "scenario_signature": "sig-shared",
        },
        {
            "status": "ok",
            "scenario_slug": "traffic/c",
            "scenario_id": "unique_id",
            "scenario_signature": "sig-c",
        },
    ]

    report = mod._scenario_identity_report(results)

    assert report["scenario_rows"] == 3
    assert report["unique_scenario_slugs"] == 3
    assert report["unique_scenario_signatures"] == 2
    assert report["duplicate_signature_excess_rows"] == 1
    assert report["duplicate_signature_groups"] == [
        {
            "scenario_signature": "sig-shared",
            "count": 2,
            "scenario_ids": ["shared_id"],
            "scenario_slugs": ["traffic/a", "traffic/b"],
        }
    ]


def test_build_jobs_uses_slug_in_trajectory_dir_to_avoid_alias_collisions(
    tmp_path: Path,
) -> None:
    slug_a = "traffic/incident_response/deep_planning/basic/traffic_live_incident_response_deep_planning_basic"
    slug_b = "traffic/vip_priority_dilemma/deep_planning/basic/traffic_vip_priority_dilemma_deep_planning_basic"
    shared_seed_id = "traffic_sumo365_20230620_deep_planning_basic_s42"
    args = argparse.Namespace(
        api_key_env="OPENAI_API_KEY",
        api_mode="auto",
        save_trajectories=True,
        pass_k=1,
    )

    jobs = mod._build_jobs(
        scenarios=[slug_a, slug_b],
        scenario_bodies={
            slug_a: {"seed_id": shared_seed_id, "scenario_signature": "sig-a"},
            slug_b: {"seed_id": shared_seed_id, "scenario_signature": "sig-b"},
        },
        models=["gpt-5"],
        seeds=[42],
        temperature=1.0,
        args=args,
        out_dir=tmp_path,
        base_url=None,
        api_version=None,
        responses_base_url=None,
    )

    trajectory_dirs = [job["trajectory_dir"] for job in jobs]
    assert len(set(trajectory_dirs)) == 2
    assert "traffic_incident_response_deep_planning_basic" in trajectory_dirs[0]
    assert "traffic_vip_priority_dilemma_deep_planning_basic" in trajectory_dirs[1]


def test_build_jobs_hashes_overlong_path_components(tmp_path: Path) -> None:
    long_leaf = "m5_" + ("inventory_replenishment_extreme_" * 8) + "final"
    slug = f"logistics/inventory/{long_leaf}"
    args = argparse.Namespace(
        api_key_env="OPENAI_API_KEY",
        api_mode="auto",
        save_trajectories=True,
        pass_k=1,
    )
    jobs = mod._build_jobs(
        scenarios=[slug],
        scenario_bodies={slug: {"seed": 42, "horizon_ticks": 10}},
        models=["gpt-5.6-luna"],
        seeds=[42],
        temperature=1.0,
        args=args,
        out_dir=tmp_path,
        base_url=None,
        api_version=None,
        responses_base_url=None,
    )
    log_name = Path(jobs[0]["episode_log_path"]).name
    traj_name = Path(jobs[0]["trajectory_dir"]).name
    assert len(log_name.encode("utf-8")) <= mod._FS_NAME_KEEP
    assert len(traj_name.encode("utf-8")) <= mod._FS_NAME_KEEP
    assert log_name.startswith("h")
    mapping_path = tmp_path / "path_shorten_map.json"
    assert mapping_path.is_file()
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mapping["n_shortened"] >= 2
    assert jobs[0]["batch_output_dir"] == str(tmp_path)


def test_formal_path_shorten_map_is_clone_portable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    long_leaf = "m5_" + ("inventory_replenishment_extreme_" * 8) + "final"
    slug = f"logistics/inventory/{long_leaf}"
    args = argparse.Namespace(
        api_key_env="OPENAI_API_KEY",
        api_mode="auto",
        save_trajectories=True,
        pass_k=1,
        formal_run=True,
    )
    out_dir = tmp_path / "batch_results" / "treatment-a"
    out_dir.mkdir(parents=True)

    mod._build_jobs(
        scenarios=[slug],
        scenario_bodies={slug: {"seed": 42, "horizon_ticks": 10}},
        models=["hy3-ioa"],
        seeds=[42],
        temperature=0.0,
        args=args,
        out_dir=out_dir,
        base_url=None,
        api_version=None,
        responses_base_url=None,
    )

    mapping_path = out_dir / "path_shorten_map.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    assert mod.canonicalize_repo_owned_paths(
        mapping, repo_root=tmp_path
    ) == mapping


def test_formal_trajectory_json_sidecars_are_batch_relative(tmp_path: Path) -> None:
    batch_root = tmp_path / "batch"
    trajectory_dir = batch_root / "trajectories" / "episode"
    trajectory_dir.mkdir(parents=True)
    summary_path = trajectory_dir / "episode.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "trajectory_summary": {
                    "trajectory_path": str(trajectory_dir / "episode"),
                    "semantic_ledger_artifact": {
                        "path": str(trajectory_dir / "episode.semantic_ledger.jsonl")
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    mod._portabilize_formal_trajectory_json_sidecars(
        {
            "formal_run": True,
            "batch_output_dir": str(batch_root),
            "trajectory_dir": str(trajectory_dir),
        }
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["trajectory_summary"]["trajectory_path"] == (
        "trajectories/episode/episode"
    )
    assert payload["trajectory_summary"]["semantic_ledger_artifact"]["path"] == (
        "trajectories/episode/episode.semantic_ledger.jsonl"
    )


def test_relative_formal_output_keeps_sidecar_byte_validation_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    treatment = "a" * 64
    out_dir = mod._resolve_logical_output_namespace(
        Path("batch_results/formal"),
        {"hy3-ioa": treatment},
        formal_run=True,
    )
    out_dir.mkdir(parents=True)
    args = argparse.Namespace(
        api_key_env="OPENAI_API_KEY",
        api_mode="auto",
        save_trajectories=True,
        pass_k=1,
        formal_run=True,
    )
    jobs = mod._build_jobs(
        scenarios=["logistics/example"],
        scenario_bodies={"logistics/example": {"seed": 42, "horizon_ticks": 10}},
        models=["hy3-ioa"],
        seeds=[42],
        temperature=0.0,
        args=args,
        out_dir=out_dir,
        base_url=None,
        api_version=None,
        responses_base_url=None,
        agent_profile_sha256_by_model={"hy3-ioa": "b" * 64},
        agent_treatment_sha256_by_model={"hy3-ioa": treatment},
    )
    assert Path(jobs[0]["trajectory_dir"]).is_absolute()
    prefix = Path(jobs[0]["trajectory_dir"]) / "episode"
    prefix.parent.mkdir(parents=True)
    sidecar = Path(f"{prefix}.semantic_ledger.jsonl")
    sidecar.write_text('{"kind":"mission"}\n', encoding="utf-8")
    row = {
        "agent_treatment_sha256": treatment,
        "trajectory_summary": {
            "trajectory_path": str(prefix),
            "semantic_ledger_artifact": {
                "schema_version": "semantic_session_ledger_v1",
                "path": str(sidecar),
                "sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
                "event_count": 1,
            },
        },
    }

    portable = mod._portable_formal_result_paths([row], batch_root=out_dir)[0]

    assert portable["trajectory_summary"]["trajectory_path"].startswith(
        "trajectories/"
    )
    assert mod._trajectory_sidecar_eligibility_reasons(
        portable,
        summary_key="semantic_ledger_artifact",
        stem="semantic_ledger",
        schema_version="semantic_session_ledger_v1",
        require_nonempty=True,
        batch_root=out_dir,
    ) == []


def test_build_jobs_schedules_long_horizons_first(tmp_path: Path) -> None:
    args = argparse.Namespace(
        api_key_env="OPENAI_API_KEY",
        api_mode="auto",
        save_trajectories=False,
        pass_k=1,
    )
    jobs = mod._build_jobs(
        scenarios=["short", "long", "medium"],
        scenario_bodies={
            "short": {"seed": 1, "horizon_ticks": 10},
            "long": {"seed": 1, "horizon_ticks": 1000},
            "medium": {"seed": 1, "horizon_ticks": 100},
        },
        models=["hy3-ioa"],
        seeds=[42],
        temperature=0.0,
        args=args,
        out_dir=tmp_path,
        base_url=None,
        api_version=None,
        responses_base_url=None,
    )

    assert [job["scenario_slug"] for job in jobs] == [
        "long",
        "medium",
        "short",
    ]
    assert [job["estimated_horizon_ticks"] for job in jobs] == [1000, 100, 10]


def test_build_jobs_expands_pass_k_replicates(tmp_path: Path) -> None:
    scenario_slug = "power_grid/foo/time_pressure/basic/foo_s42"
    args = argparse.Namespace(
        api_key_env="OPENAI_API_KEY",
        api_mode="auto",
        save_trajectories=True,
        pass_k=2,
    )

    jobs = mod._build_jobs(
        scenarios=[scenario_slug],
        scenario_bodies={
            scenario_slug: {
                "seed_id": "foo_s42",
                "scenario_signature": "sig-a",
                "domain": "power_grid",
                "backend_kind": "pandapower_acopf",
                "construct_contract": "operational_agency.v1",
                "source_denominator_key": "pglib:case-a",
                "case_ledger": {
                    "schema_version": "0.1",
                    "source_denominator_key": "pglib:case-a",
                },
            }
        },
        models=["gpt-5"],
        seeds=[42],
        temperature=1.0,
        args=args,
        out_dir=tmp_path,
        base_url=None,
        api_version=None,
        responses_base_url=None,
    )

    assert [job["pass_id"] for job in jobs] == ["pass-0", "pass-1"]
    assert [job["pass_index"] for job in jobs] == [0, 1]
    assert {job["pass_k"] for job in jobs} == {2}
    assert jobs[0]["episode_log_path"].endswith("_s42_pass-0.log")
    assert jobs[1]["episode_log_path"].endswith("_s42_pass-1.log")
    assert jobs[0]["trajectory_dir"].endswith("foo_s42_s42_pass-0")
    assert jobs[1]["trajectory_dir"].endswith("foo_s42_s42_pass-1")
    assert {job["domain"] for job in jobs} == {"power_grid"}
    assert {job["backend_kind"] for job in jobs} == {"pandapower_acopf"}
    assert {job["construct_contract"] for job in jobs} == {"operational_agency.v1"}
    assert {job["source_denominator_key"] for job in jobs} == {"pglib:case-a"}
    assert jobs[0]["case_ledger"] == {
        "schema_version": "0.1",
        "source_denominator_key": "pglib:case-a",
    }

    terminal = mod._apply_llm_job_metadata(
        jobs[0],
        {
            "status": "ok",
            "evaluation_protocol": {
                "version": mod.EVALUATION_PROTOCOL_VERSION,
                "implementation_fingerprint": (
                    mod.EVALUATION_IMPLEMENTATION_FINGERPRINT
                ),
            },
        },
    )
    assert terminal["evaluation_protocol"]["construct_contract"] == (
        "operational_agency.v1"
    )
    assert terminal["source_denominator_key"] == "pglib:case-a"
    assert terminal["case_ledger"] == jobs[0]["case_ledger"]
    assert terminal["agent_profile_sha256"] == jobs[0]["agent_profile_sha256"]


def test_build_jobs_can_use_each_scenarios_locked_seed(tmp_path: Path) -> None:
    scenario_slug = "logistics/family/time_pressure/basic/sample"
    args = argparse.Namespace(
        api_key_env="OPENAI_API_KEY",
        api_mode="auto",
        save_trajectories=False,
        pass_k=1,
        seed_mode="scenario",
    )

    jobs = mod._build_jobs(
        scenarios=[scenario_slug],
        scenario_bodies={
            scenario_slug: {
                "seed": 7703,
                "seed_id": "sample",
                "scenario_signature": "suite-signature",
            }
        },
        models=["model"],
        seeds=[42],
        temperature=1.0,
        args=args,
        out_dir=tmp_path,
        base_url=None,
        api_version=None,
        responses_base_url=None,
    )

    assert len(jobs) == 1
    assert jobs[0]["seed"] == 7703
    assert jobs[0]["seed_mode"] == "scenario"
    assert jobs[0]["suite_scenario_signature"] == "suite-signature"


def test_episode_worker_preserves_runtime_identity_and_quarantines_stale_trajectory(
    tmp_path: Path, monkeypatch
) -> None:
    captured: list[tuple] = []

    def fake_run_one_safe(args: tuple) -> dict:
        captured.append(args)
        return {
            "status": "ok",
            "scenario_id": "foo_s42",
            "scenario_signature": "runtime-sig",
        }

    monkeypatch.setattr(mod, "_run_one_safe", fake_run_one_safe)
    trajectory_dir = tmp_path / "trajectory"
    trajectory_dir.mkdir()
    (trajectory_dir / "partial.trajectory.jsonl").write_text("{}\n", encoding="utf-8")
    cfg = mod.LLMConfig(provider="openai", model="deepseek-v4-pro")
    result = mod._run_llm_episode_job(
        {
            "scenario_slug": "power_grid/foo/time_pressure/basic/foo_s42",
            "seed": 42,
            "model": "deepseek-v4-pro",
            "temperature": 1.0,
            "scenario_signature": "sig-a",
            "domain": "power_grid",
            "backend_kind": "pandapower_acopf",
            "trajectory_dir": str(trajectory_dir),
            "llm_config": mod._llm_config_to_dict(cfg),
        }
    )

    assert result["domain"] == "power_grid"
    assert result["backend_kind"] == "pandapower_acopf"
    assert result["scenario_signature"] == "runtime-sig"
    assert result["suite_scenario_signature"] == "sig-a"
    assert not trajectory_dir.exists()
    stale = list(tmp_path.glob("trajectory.stale-*"))
    assert len(stale) == 1
    assert (stale[0] / "partial.trajectory.jsonl").is_file()
    run_options = captured[0][4]
    assert run_options["per_action_attribution"] is True
    assert run_options["per_action_group_attribution"] is True
    assert run_options["per_action_cap"] is None
    assert run_options["per_action_group_cap"] is None


def _minimal_llm_job(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    cfg = mod.LLMConfig(provider="openai", model="hy3-ioa")
    job: dict[str, Any] = {
        "scenario_slug": "power_grid/foo/time_pressure/basic/foo_s42",
        "seed": 42,
        "model": "hy3-ioa",
        "temperature": 1.0,
        "scenario_signature": "sig-a",
        "batch_output_dir": str(tmp_path),
        "llm_config": mod._llm_config_to_dict(cfg),
        "implementation_tree_sha256": "tree-a",
    }
    job.update(overrides)
    return job


def test_episode_worker_skips_when_quota_sentinel_is_active(
    tmp_path: Path, monkeypatch
) -> None:
    called: list[int] = []
    monkeypatch.setattr(
        mod,
        "_run_one_safe",
        lambda *_args, **_kwargs: called.append(1) or {"status": "ok"},
    )
    job = _minimal_llm_job(tmp_path)
    mod._write_quota_sentinel(
        job,
        {
            "quota_reset_at": "2099-01-01 00:00:00 UTC+8",
            "error": "ProviderQuotaExhaustedError",
        },
    )
    result = mod._run_llm_episode_job(job)
    assert result["quota_parked"] is True
    assert result["status"] == "error"
    assert called == []


def test_episode_worker_writes_quota_sentinel(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        mod,
        "implementation_identity",
        lambda *_args: {"implementation_tree_sha256": "tree-a"},
    )
    monkeypatch.setattr(
        mod,
        "_run_one_safe",
        lambda *_args, **_kwargs: {
            "status": "error",
            "error": (
                "ProviderQuotaExhaustedError: 超出频率限制，将在 "
                "2026-08-19 16:33:40 UTC+8 后恢复"
            ),
        },
    )
    result = mod._run_llm_episode_job(
        _minimal_llm_job(tmp_path, scenario_slug="cell-a")
    )
    assert "ProviderQuotaExhaustedError" in str(result["error"])
    sentinel = tmp_path / ".quota_exhausted_hy3-ioa"
    assert sentinel.is_file()
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    assert payload["reset_at"] == "2026-08-19 16:33:40 UTC+8"


def test_run_llm_model_lane_parks_remaining_jobs_after_quota(
    tmp_path: Path, monkeypatch
) -> None:
    def fake_run(job: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "error",
            "error": "ProviderQuotaExhaustedError: 超出频率限制",
            "model": job["model"],
            "scenario_slug": job["scenario_slug"],
            "quota_reset_at": "2026-08-19 16:33:40 UTC+8",
        }

    calls: list[str] = []

    def tracking_run(job: dict[str, Any]) -> dict[str, Any]:
        calls.append(job["scenario_slug"])
        return fake_run(job)

    monkeypatch.setattr(mod, "_run_llm_episode_job", tracking_run)
    lane_jobs = [
        _minimal_llm_job(tmp_path, scenario_slug="cell-a"),
        _minimal_llm_job(tmp_path, scenario_slug="cell-b"),
    ]
    summary = mod._run_llm_model_lane(
        {
            "model": "hy3-ioa",
            "jobs": lane_jobs,
            "episodes_path": str(tmp_path / "episodes.jsonl"),
        }
    )
    assert summary["n_completed"] == 2
    assert calls == ["cell-a"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    parked = [row for row in rows if row.get("quota_parked")]
    assert len(parked) == 1
    assert parked[0]["scenario_slug"] == "cell-b"


def test_run_global_jobs_does_not_submit_remaining_jobs_after_quota(
    monkeypatch, tmp_path: Path
) -> None:
    jobs = [
        {
            "scenario_slug": f"case-{index}",
            "model": "hy3-ioa",
            "seed": 42,
            "batch_output_dir": str(tmp_path),
            "temperature": 1.0,
        }
        for index in range(6)
    ]
    submitted: list[str] = []

    class FakeFile:
        def __init__(self) -> None:
            self.rows: list[dict[str, Any]] = []

        def write(self, line: str) -> None:
            self.rows.append(json.loads(line))

        def flush(self) -> None:
            return None

        def __enter__(self) -> FakeFile:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    class FakeFuture:
        def __init__(self, job: dict[str, Any]) -> None:
            self.job = job

        def result(self) -> dict[str, Any]:
            if self.job["scenario_slug"] == "case-0":
                return {
                    "status": "error",
                    "error": "ProviderQuotaExhaustedError: 6004",
                    "model": "hy3-ioa",
                    "quota_reset_at": "2026-08-19 16:33:40 UTC+8",
                }
            return {
                "status": "ok",
                "model": "hy3-ioa",
                "scenario_slug": self.job["scenario_slug"],
            }

    class FakePool:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def submit(self, fn: object, job: dict[str, Any]) -> FakeFuture:
            submitted.append(job["scenario_slug"])
            return FakeFuture(job)

        def __enter__(self) -> FakePool:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    fake_file = FakeFile()
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: fake_file)
    monkeypatch.setattr("concurrent.futures.ProcessPoolExecutor", FakePool)
    monkeypatch.setattr("concurrent.futures.as_completed", lambda fs: fs)

    mod._run_global_jobs(jobs, tmp_path / "episodes.jsonl", "w", 1)

    assert submitted == ["case-0", "case-1"]
    parked = [row for row in fake_file.rows if row.get("quota_parked")]
    assert [row["scenario_slug"] for row in parked] == [
        "case-2",
        "case-3",
        "case-4",
        "case-5",
    ]


def test_episode_worker_fails_closed_on_implementation_tree_drift(
    monkeypatch,
) -> None:
    identities = iter(
        [
            {"implementation_tree_sha256": "tree-a"},
            {"implementation_tree_sha256": "tree-b"},
        ]
    )
    monkeypatch.setattr(mod, "implementation_identity", lambda *_args: next(identities))
    monkeypatch.setattr(
        mod,
        "_run_one_safe",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "scenario_id": "foo_s42",
            "scenario_signature": "runtime-sig",
        },
    )
    cfg = mod.LLMConfig(provider="openai", model="model")

    result = mod._run_llm_episode_job(
        {
            "scenario_slug": "power_grid/foo/time_pressure/basic/foo_s42",
            "seed": 42,
            "model": "model",
            "temperature": 1.0,
            "scenario_signature": "runtime-sig",
            "implementation_tree_sha256": "tree-a",
            "llm_config": mod._llm_config_to_dict(cfg),
        }
    )

    assert result["status"] == "error"
    assert result["error"] == "implementation_tree_drift"
    assert result["implementation_tree_sha256"] == "tree-a"
    assert result["implementation_tree_sha256_end"] == "tree-b"


def test_grid2op_local_cache_requirements_ignore_test_envs() -> None:
    bodies = {
        "power_grid/storm/release_case": {
            "backend_kind": "grid2op",
            "backend_config": {
                "env_name": "l2rpn_case14_sandbox",
                "test": False,
            },
        },
        "power_grid/storm/test_case": {
            "backend_kind": "grid2op",
            "backend_config": {
                "env_name": "l2rpn_case14_sandbox",
                "test": True,
            },
        },
        "power_grid/uc/case": {
            "backend_kind": "pglib_uc_synthetic",
        },
    }

    required = mod._grid2op_env_names_requiring_local_cache(
        bodies,
        [
            "power_grid/storm/release_case",
            "power_grid/storm/test_case",
            "power_grid/uc/case",
        ],
    )

    assert required == ["l2rpn_case14_sandbox"]


def test_grid2op_local_cache_requirements_fail_without_env_name() -> None:
    bodies = {
        "power_grid/storm/bad": {
            "backend_kind": "grid2op",
            "backend_config": {"test": False},
        },
    }

    with pytest.raises(ValueError, match="lacks backend_config.env_name"):
        mod._grid2op_env_names_requiring_local_cache(
            bodies,
            ["power_grid/storm/bad"],
        )


def test_grid2op_cache_preflight_blocks_before_job_build(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    output_dir = tmp_path / "blocked_grid2op"
    scenario = (
        "power_grid/storm_l2rpn_icaps2021/time_pressure/high/"
        "storm_icaps2021_c0_time_pressure_high_s42"
    )

    monkeypatch.setattr(
        mod,
        "_load_zhsrc_exports",
        lambda: {
            "OPENAI_API_KEY": "test-key",
            "OPERATE_MODELS": "gpt-5-2025-08-07",
        },
    )
    monkeypatch.setattr(mod, "_expand_scenarios", lambda patterns: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda slug: {
            "seed_id": slug.split("/")[-1],
            "family": "storm_l2rpn_icaps2021",
            "backend_kind": "grid2op",
            "backend_config": {
                "env_name": "l2rpn_icaps_2021_small",
                "test": False,
            },
            "scenario_signature": "sig-grid2op",
        },
    )
    monkeypatch.setattr(
        mod,
        "_grid2op_local_cache_preflight",
        lambda envs: {
            "ok": False,
            "required_envs": list(envs),
            "summary": {"remote_only_not_downloaded": 1},
            "blockers": [
                {
                    "env_name": "l2rpn_icaps_2021_small",
                    "status": "remote_only_not_downloaded",
                    "data_root": "/tmp/grid2op",
                }
            ],
        },
    )
    monkeypatch.setattr(
        mod,
        "_build_jobs",
        lambda **kwargs: pytest.fail("jobs must not be built after failed preflight"),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(output_dir),
            "--scenario-slice",
            "custom",
            "--scenarios",
            scenario,
            "--models",
            "gpt-5-2025-08-07",
            "--interaction-mode",
            "logical_stateless",
            "--dry-run",
        ],
    )

    rc = mod.main()

    captured = capsys.readouterr()
    assert rc == 1
    assert "Grid2Op local cache preflight failed" in captured.err
    assert "l2rpn_icaps_2021_small=remote_only_not_downloaded" in captured.err
    assert not (output_dir / "run_config.json").exists()


def test_non_grid2op_dry_run_skips_grid2op_cache_preflight(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "non_grid2op"
    scenario = "power_grid/daily_ops_24h/time_pressure/basic/do_20200127_time_pressure_basic_s42"

    monkeypatch.setattr(
        mod,
        "_load_zhsrc_exports",
        lambda: {
            "OPENAI_API_KEY": "test-key",
            "OPERATE_MODELS": "gpt-5-2025-08-07",
        },
    )
    monkeypatch.setattr(mod, "_expand_scenarios", lambda patterns: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda slug: {
            "seed_id": slug.split("/")[-1],
            "family": "daily_ops_24h",
            "backend_kind": "pglib_uc_synthetic",
            "scenario_signature": "sig-uc",
        },
    )
    monkeypatch.setattr(
        mod,
        "_grid2op_local_cache_preflight",
        lambda envs: pytest.fail("non-grid2op slices must not preflight Grid2Op"),
    )
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {"git_commit": "abc", "git_dirty": False, "git_status_short": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(output_dir),
            "--scenario-slice",
            "custom",
            "--scenarios",
            scenario,
            "--models",
            "gpt-5-2025-08-07",
            "--interaction-mode",
            "logical_stateless",
            "--dry-run",
        ],
    )

    rc = mod.main()

    assert rc == 0
    run_cfg = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_cfg["has_grid2op"] is False
    assert run_cfg["grid2op_required_local_envs"] == []
    assert run_cfg["grid2op_local_cache_preflight"] is None


def test_dry_run_does_not_require_api_key(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "credential_free_plan"
    scenario = "power_grid/daily_ops_24h/time_pressure/basic/do_20200127_time_pressure_basic_s42"
    monkeypatch.setattr(mod, "_load_zhsrc_exports", lambda: {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(mod, "_expand_scenarios", lambda patterns: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda slug: {
            "seed_id": slug.split("/")[-1],
            "family": "daily_ops_24h",
            "backend_kind": "pglib_uc_synthetic",
            "scenario_signature": "sig-uc",
        },
    )
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {"git_commit": "abc", "git_dirty": False, "git_status_short": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(output_dir),
            "--scenario-slice",
            "custom",
            "--scenarios",
            scenario,
            "--models",
            "deepseek-v4-pro",
            "--interaction-mode",
            "logical_stateless",
            "--dry-run",
        ],
    )

    assert mod.main() == 0
    assert not (output_dir / "episodes.jsonl").exists()


def test_grid2op_dry_run_records_local_cache_preflight(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "cached_grid2op"
    scenario = (
        "power_grid/storm_l2rpn_sandbox/time_pressure/high/"
        "storm_sandbox_c0_time_pressure_high_s42"
    )

    monkeypatch.setattr(
        mod,
        "_load_zhsrc_exports",
        lambda: {
            "OPENAI_API_KEY": "test-key",
            "OPERATE_MODELS": "gpt-5-2025-08-07",
        },
    )
    monkeypatch.setattr(mod, "_expand_scenarios", lambda patterns: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda slug: {
            "seed_id": slug.split("/")[-1],
            "family": "storm_l2rpn_sandbox",
            "backend_kind": "grid2op",
            "backend_config": {
                "env_name": "l2rpn_case14_sandbox",
                "test": False,
            },
            "scenario_signature": "sig-sandbox",
        },
    )

    def fake_cache_preflight(envs: list[str]) -> dict[str, Any]:
        assert envs == ["l2rpn_case14_sandbox"]
        return {
            "ok": True,
            "required_envs": list(envs),
            "data_root": "/tmp/grid2op",
            "summary": {"local_loadable": 1},
            "sources": [
                {
                    "env_name": "l2rpn_case14_sandbox",
                    "status": "local_loadable",
                    "load_path": "/tmp/grid2op/l2rpn_case14_sandbox",
                }
            ],
            "blockers": [],
        }

    monkeypatch.setattr(mod, "_grid2op_local_cache_preflight", fake_cache_preflight)
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {"git_commit": "abc", "git_dirty": False, "git_status_short": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(output_dir),
            "--scenario-slice",
            "custom",
            "--scenarios",
            scenario,
            "--models",
            "gpt-5-2025-08-07",
            "--interaction-mode",
            "logical_stateless",
            "--dry-run",
        ],
    )

    rc = mod.main()

    assert rc == 0
    run_cfg = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_cfg["has_grid2op"] is True
    assert run_cfg["grid2op_required_local_envs"] == ["l2rpn_case14_sandbox"]
    assert run_cfg["grid2op_local_cache_preflight"]["ok"] is True


def test_finalize_pipeline_writes_expected_artifacts_and_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "repr"
    scenarios = [
        "power_grid/daily_ops_24h/time_pressure/basic/do_20200127_time_pressure_basic_s42",
        "power_grid/distribution_volt_var/deep_planning/basic/dvv_mv_with_der_all_deep_planning_basic_s42",
    ]
    models = ["gpt-5-2025-08-07", "o3-2025-04-16"]

    scenario_map = {
        scenarios[0]: {
            "seed_id": "do_20200127_time_pressure_basic_s42",
            "family": "daily_ops_24h",
            "backend_kind": "pglib_uc_synthetic",
            "scenario_signature": "sig-daily",
        },
        scenarios[1]: {
            "seed_id": "dvv_mv_with_der_all_deep_planning_basic_s42",
            "family": "distribution_volt_var",
            "backend_kind": "cigre_distribution",
            "scenario_signature": "sig-dvv",
        },
    }

    def fake_load_zhsrc_exports() -> dict[str, str]:
        return {
            "OPENAI_API_KEY": "test-key",
            "OPERATE_MODELS": ",".join(models),
            "OPERATE_API_BASE_URL": "http://example.test",
            "OPERATE_API_VERSION": "2024-03-01-preview",
        }

    def fake_expand(patterns: list[str]) -> list[str]:
        assert patterns == scenarios
        return list(scenarios)

    def fake_load_scenario_yaml(slug: str) -> dict[str, Any]:
        body = dict(scenario_map[slug])
        body.update(
            {
                "domain": "power_grid",
                "difficulty_mode": "time_pressure",
                "difficulty_level": "basic",
                "horizon_ticks": 24,
                "tick_minutes": 60,
            }
        )
        return body

    def fake_signature(scenario: dict[str, Any], seed: int) -> str:
        return f"{scenario['scenario_signature']}-seed{seed}"

    def fake_run_job(job: dict[str, Any]) -> dict[str, Any]:
        scenario = scenario_map[job["scenario_slug"]]
        total_score = 81.0 if scenario["family"] == "daily_ops_24h" else 67.5
        total_score += 4.0 if job["model"].startswith("gpt-5") else -2.0
        row = _fake_episode_result(
            job["scenario_slug"],
            job["model"],
            job["seed"],
            job["scenario_signature"],
            scenario["family"],
            scenario["backend_kind"],
            total_score=total_score,
            n_tool_calls=5 if scenario["family"] == "daily_ops_24h" else 3,
        )
        log_path = Path(job["episode_log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("falling back to wait\n", encoding="utf-8")
        return mod._apply_llm_job_metadata(job, row)

    class FakeFuture:
        def __init__(self, row: dict[str, Any]) -> None:
            self.row = row

        def result(self) -> dict[str, Any]:
            return self.row

    class FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def submit(self, fn, job):
            del fn
            return FakeFuture(fake_run_job(job))

    monkeypatch.setattr(mod, "_load_zhsrc_exports", fake_load_zhsrc_exports)
    monkeypatch.setattr(mod, "_expand_scenarios", fake_expand)
    monkeypatch.setattr(mod, "load_scenario_yaml", fake_load_scenario_yaml)
    monkeypatch.setattr(mod, "_scenario_signature_for_run", fake_signature)
    monkeypatch.setattr(mod, "_run_llm_episode_job", fake_run_job)
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {"git_commit": "abc", "git_dirty": False, "git_status_short": []},
    )

    import concurrent.futures as futures

    monkeypatch.setattr(futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(futures, "as_completed", lambda fs: fs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(output_dir),
            "--scenario-slice",
            "custom",
            "--scenarios",
            *scenarios,
            "--models",
            ",".join(models),
            "--interaction-mode",
            "logical_stateless",
            "--seeds",
            "42",
        ],
    )

    rc = mod.main()
    assert rc == 0

    expected = [
        "episodes.jsonl",
        "summary.csv",
        "leaderboard.json",
        "ANALYSIS.md",
        "analysis_deep.json",
        "decision_impact_report.json",
        "DECISION_IMPACT.md",
        "evidence_applicability_report.json",
        "EVIDENCE_APPLICABILITY.md",
        "tool_effect_audit_report.json",
        "TOOL_EFFECT_AUDIT.md",
        "staleness_consumption_report.json",
        "STALENESS_CONSUMPTION.md",
        "agent_failure_recipes_report.json",
        "AGENT_FAILURE_RECIPES.md",
        "LOG_AUDIT.json",
        "LOG_AUDIT.md",
        "RUN_MANIFEST.json",
        "run_config.json",
        "batch_run.log",
    ]
    for name in expected:
        path = output_dir / name
        assert path.exists(), name
        assert path.read_text(encoding="utf-8").strip(), name

    for name in mod.PLOT_FILES:
        path = output_dir / "plots" / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name

    manifest = json.loads(
        (output_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["scenario_slice"] == "custom"
    assert manifest["n_scenarios"] == 2
    assert manifest["n_episodes_total"] == 4
    assert manifest["n_episodes_ok"] == 4
    assert manifest["temperature"] == 1.0
    assert manifest["pass_k"] == 1
    assert sorted(manifest["models"]) == sorted(models)
    assert manifest["artifacts"]["evidence_applicability_json"].endswith(
        "evidence_applicability_report.json"
    )
    assert manifest["artifacts"]["evidence_applicability_markdown"].endswith(
        "EVIDENCE_APPLICABILITY.md"
    )
    assert manifest["artifacts"]["tool_effect_audit_json"].endswith(
        "tool_effect_audit_report.json"
    )
    assert manifest["artifacts"]["tool_effect_audit_markdown"].endswith(
        "TOOL_EFFECT_AUDIT.md"
    )
    assert manifest["artifacts"]["staleness_consumption_json"].endswith(
        "staleness_consumption_report.json"
    )
    assert manifest["artifacts"]["staleness_consumption_markdown"].endswith(
        "STALENESS_CONSUMPTION.md"
    )
    assert manifest["artifacts"]["agent_failure_recipes_json"].endswith(
        "agent_failure_recipes_report.json"
    )
    assert manifest["artifacts"]["agent_failure_recipes_markdown"].endswith(
        "AGENT_FAILURE_RECIPES.md"
    )
    assert manifest["artifacts"]["plots"]
    assert {row["scenario_slug"] for row in manifest["scenarios"]} == set(scenarios)

    # Partial-batch / comparability fields (introduced 2026-06-02): when every
    # configured (model, seed, scenario) cell has a status=ok row the batch is
    # NOT partial and the intersection equals expected per-model totals.
    assert manifest["expected_total"] == 4
    assert manifest["is_partial_batch"] is False
    assert manifest["comparable_intersection_size"] == 2
    assert manifest["comparability_warning"] is None
    realized = manifest["realized_coverage"]
    assert set(realized) == set(models)
    assert all(abs(v - 1.0) < 1e-9 for v in realized.values())
    assert manifest["coverage"]["expected_episodes_per_model"] == 2
    assert "_comparable_pairs" not in manifest["coverage"]
    assert isinstance(manifest["intersection_leaderboard"], list)
    assert len(manifest["intersection_leaderboard"]) == len(models)
    inter_models = {row["agent_id"] for row in manifest["intersection_leaderboard"]}
    assert inter_models == set(models)

    # Formal partial/final state machine (introduced 2026-06-02): all cells ok,
    # no orphan interrupted logs → state must be "final".
    assert manifest["batch_state"] == mod.BATCH_STATE_FINAL
    assert manifest["batch_state_reasons"]
    assert manifest["n_orphan_interrupted_logs"] == 0
    leaderboard_payload = json.loads(
        (output_dir / "leaderboard.json").read_text(encoding="utf-8")
    )
    assert leaderboard_payload["batch_state"] == mod.BATCH_STATE_FINAL
    assert leaderboard_payload["n_orphan_interrupted_logs"] == 0
    analysis_text = (output_dir / "ANALYSIS.md").read_text(encoding="utf-8")
    assert "Batch state: `final`" in analysis_text

    leaderboard_payload = json.loads(
        (output_dir / "leaderboard.json").read_text(encoding="utf-8")
    )
    assert leaderboard_payload["expected_total"] == 4
    assert leaderboard_payload["is_partial_batch"] is False
    assert leaderboard_payload["comparable_intersection_size"] == 2
    assert "_comparable_pairs" not in leaderboard_payload["coverage"]
    assert leaderboard_payload["intersection_leaderboard"]

    analysis_md = (output_dir / "ANALYSIS.md").read_text(encoding="utf-8")
    assert "Coverage & comparability" in analysis_md
    assert "Replicate execution completion" in analysis_md
    assert "Fraction of planned cells with all replicates completed" in analysis_md
    assert "Intersection leaderboard" in analysis_md


def test_write_leaderboard_json_includes_fixed_and_adaptive_views(
    tmp_path: Path,
) -> None:
    # ``dimensions`` carries only DISCRIMINATIVE_CORE_DIMENSIONS-member scores
    # so discriminative_core's recompute (task_completion folded in, curated
    # weights) tracks fixed_all_dimensions's ranking on this synthetic data.
    rows = [
        {
            "status": "ok",
            "interaction_mode": "logical_stateless",
            "model": "model-a",
            "score": {
                "total_score": 10.0,
                "score_views": {
                    "adaptive_applicable": {"total_score": 40.0},
                    "fixed_all_dimensions": {"total_score": 10.0},
                },
                "dimensions": [
                    {
                        "name": "system_survival",
                        "applicable": True,
                        "calibrated_score": 10.0,
                        "evidence_ids": ["survival:a"],
                    },
                ],
            },
            "ground_truth_summary": {"chose_fatal_option": False},
        },
        {
            "status": "ok",
            "interaction_mode": "logical_stateless",
            "model": "model-b",
            "score": {
                "total_score": 12.0,
                "score_views": {
                    "adaptive_applicable": {"total_score": 20.0},
                    "fixed_all_dimensions": {"total_score": 12.0},
                },
                "dimensions": [
                    {
                        "name": "system_survival",
                        "applicable": True,
                        "calibrated_score": 12.0,
                        "evidence_ids": ["survival:b"],
                    },
                ],
            },
            "ground_truth_summary": {"chose_fatal_option": False},
        },
    ]

    mod._write_leaderboard_json(tmp_path, rows)

    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    views = payload["diagnostic_leaderboard_views"]
    assert views["fixed_all_dimensions"][0]["agent_id"] == "model-b"
    assert views["adaptive_applicable"][0]["agent_id"] == "model-a"
    assert "discriminative_core" not in views
    assert payload["diagnostic_flat_leaderboard"] == views["fixed_all_dimensions"]
    assert "diagnostic_flat_holm_pairwise" in payload
    assert "diagnostic_flat_holm_pairwise_repeat_diagnostics" in payload
    assert "holm_pairwise" not in payload
    assert "primary_leaderboard" not in payload


def test_formal_leaderboard_excludes_provider_contaminated_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite_eligibility = {
        "suite_blocked": False,
        "diagnostic_cells": [],
        "uninformative_cells": [],
        "wait_dominant_cells": [],
    }

    def row(scenario_id: str, failed: int) -> dict:
        return {
            "status": "ok",
            "interaction_mode": "logical_stateless",
            "scenario_id": scenario_id,
            "scenario_slug": scenario_id,
            "scenario_signature": scenario_id,
            "seed": 42,
            "model": "model-a",
            "score": {
                "total_score": 10.0,
                "scoring_version": mod.SCORING_VERSION,
            },
            "evaluation_protocol": {
                "version": mod.EVALUATION_PROTOCOL_VERSION,
                "implementation_fingerprint": (
                    mod.EVALUATION_IMPLEMENTATION_FINGERPRINT
                ),
            },
            "suite_manifest_sha256": "suite",
            "suite_eligibility": suite_eligibility,
            "suite_eligibility_sha256": mod._canonical_json_sha256(suite_eligibility),
            "trajectory_summary": {
                "llm": {
                    "llm_calls_failed": failed,
                    "provider_models": ["model-a"],
                },
                "terminal_integrity": {"release_ready": True},
                "event_contract": {"schema_version": "1.0", "violation_count": 0},
                "provider_audit_artifact": _provider_audit_artifact(),
                "tool_surface_contract": _tool_surface_contract(),
                "tool_semantic_coverage": {
                    "covered": True,
                    "unknown_tool_names": [],
                    "unclassified_tool_names": [],
                    "explicit_semantic_roles_complete": True,
                    "missing_explicit_semantic_role_names": [],
                    "native_targets_complete": True,
                    "missing_native_target_kind_names": [],
                    "state_changing_actuators_complete": True,
                    "missing_actuator_family_names": [],
                },
            },
        }

    captured: list[dict] = []

    def fake_primary(rows: list[dict]) -> dict:
        captured.extend(rows)
        return {
            "leaderboard": [],
            "scoring_version": mod.SCORING_VERSION,
            "primary_leaderboard_formula_version": (
                "effective_source_backend_domain_macro_v1"
            ),
        }

    monkeypatch.setattr(mod, "_primary_leaderboard_payload", fake_primary)
    clean_row = _bind_formal_artifacts(row("clean", 0), tmp_path, key="clean")
    contaminated_row = _bind_formal_artifacts(
        row("contaminated", 1), tmp_path, key="contaminated"
    )
    mod._write_leaderboard_json(
        tmp_path,
        [clean_row, contaminated_row],
        coverage={
            "configured_models": ["model-a"],
            "configured_seeds": [42],
            "pass_k": 1,
            "expected_total": 1,
            "comparable_intersection_size": 1,
            "is_partial_batch": False,
            "_comparable_pairs": [["clean", 42]],
        },
        state={
            "batch_state": mod.BATCH_STATE_FINAL,
            "reasons": [],
            "n_orphan_interrupted_logs": 0,
        },
        formal=True,
    )

    assert [item["scenario_id"] for item in captured] == ["clean"]
    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    assert payload["formal_primary_exclusions"] == [
        {
            "scenario_id": "contaminated",
            "model": "model-a",
            "reasons": ["provider_call_failure"],
        }
    ]


def test_formal_leaderboard_is_suppressed_for_partial_batch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mod,
        "_primary_leaderboard_payload",
        lambda _rows: pytest.fail("partial formal batch must not be ranked"),
    )

    returned = mod._write_leaderboard_json(
        tmp_path,
        [],
        coverage={
            "configured_models": ["model-a", "model-b"],
            "configured_seeds": [42],
            "pass_k": 1,
            "expected_total": 2,
            "comparable_intersection_size": 0,
            "is_partial_batch": True,
            "_comparable_pairs": [],
        },
        state={
            "batch_state": mod.BATCH_STATE_PARTIAL,
            "reasons": ["coverage incomplete"],
            "n_orphan_interrupted_logs": 0,
        },
        formal=True,
    )

    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    assert returned == []
    assert "primary_leaderboard" not in payload
    assert payload["formal_primary_blockers"] == [
        "formal_batch_not_final",
        "formal_coverage_incomplete",
    ]


def test_formal_leaderboard_records_incomplete_five_group_as_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        mod, "_formal_row_eligibility", lambda *_args, **_kwargs: (True, [])
    )
    row = {
        "status": "ok",
        "interaction_mode": "logical_stateless",
        "scenario_id": "sparse",
        "scenario_slug": "sparse",
        "scenario_signature": "sparse",
        "seed": 42,
        "model": "model-a",
        "difficulty_level": "basic",
        "task_completion": {
            "applicable": True,
            "completed": True,
            "contract": "native.task.v1",
            "evidence": {"objective_met": True},
        },
        "score": {
            "total_score": 10.0,
            "dimensions": [
                {
                    "name": "system_survival",
                    "applicable": True,
                    "calibrated_score": 100.0,
                    "evidence_ids": ["survival:e1"],
                }
            ],
        },
    }

    returned = mod._write_leaderboard_json(
        tmp_path,
        [row],
        coverage={
            "configured_models": ["model-a"],
            "configured_seeds": [42],
            "pass_k": 1,
            "expected_total": 1,
            "comparable_intersection_size": 1,
            "is_partial_batch": False,
            "_comparable_pairs": [["sparse", 42]],
        },
        state={
            "batch_state": mod.BATCH_STATE_FINAL,
            "reasons": [],
            "n_orphan_interrupted_logs": 0,
        },
        formal=True,
    )

    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    assert returned == []
    assert "primary_leaderboard" not in payload
    assert payload["formal_primary_blockers"] == ["formal_primary_contract_error"]
    assert "five-group evidence" in payload["formal_primary_contract_error"]


def test_formal_leaderboard_blocks_ineligible_configured_episode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite_eligibility = {"suite_blocked": False}
    row = {
        "status": "ok",
        "interaction_mode": "logical_stateless",
        "scenario_id": "scenario-a",
        "scenario_slug": "scenario-a",
        "scenario_signature": "sig-a",
        "seed": 42,
        "model": "model-a",
        "score": {
            "total_score": 10.0,
            "scoring_version": mod.SCORING_VERSION,
        },
        "evaluation_protocol": {
            "version": mod.EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
        "suite_manifest_sha256": "suite",
        "suite_eligibility": suite_eligibility,
        "suite_eligibility_sha256": mod._canonical_json_sha256(suite_eligibility),
        "trajectory_summary": {
            "llm": {"provider_models": ["model-a"]},
            "terminal_integrity": {"release_ready": False},
            "event_contract": {"schema_version": "1.0", "violation_count": 0},
            "provider_audit_artifact": _provider_audit_artifact(),
            "tool_surface_contract": _tool_surface_contract(),
            "tool_semantic_coverage": {
                "covered": True,
                "unknown_tool_names": [],
                "unclassified_tool_names": [],
                "explicit_semantic_roles_complete": True,
                "missing_explicit_semantic_role_names": [],
                "native_targets_complete": True,
                "missing_native_target_kind_names": [],
                "state_changing_actuators_complete": True,
                "missing_actuator_family_names": [],
            },
        },
    }
    monkeypatch.setattr(
        mod,
        "_primary_leaderboard_payload",
        lambda _rows: pytest.fail("an ineligible configured cell must block primary"),
    )
    _bind_formal_artifacts(row, tmp_path, key="ineligible")

    mod._write_leaderboard_json(
        tmp_path,
        [row],
        coverage={
            "configured_models": ["model-a"],
            "configured_seeds": [42],
            "pass_k": 1,
            "expected_total": 1,
            "comparable_intersection_size": 1,
            "is_partial_batch": False,
            "_comparable_pairs": [["scenario-a", 42]],
        },
        state={
            "batch_state": mod.BATCH_STATE_FINAL,
            "reasons": [],
            "n_orphan_interrupted_logs": 0,
        },
        formal=True,
    )

    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    assert "primary_leaderboard" not in payload
    assert payload["formal_primary_blockers"] == [
        "formal_configured_episode_ineligible"
    ]
    assert payload["formal_configured_episode_failures"][0]["reasons"] == [
        "terminal_integrity_failure"
    ]


def test_formal_leaderboard_blocks_stale_suite_bound_episode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    row = _formally_eligible_protocol21_row()
    row.update(
        {
            "scenario_id": "scenario-a",
            "scenario_slug": "scenario-a",
            "scenario_signature": "sig-a",
            "seed": 42,
            "model": "model-a",
            "score": {
                "total_score": 10.0,
                "scoring_version": mod.SCORING_VERSION,
            },
            "suite_manifest_sha256": "suite-old",
        }
    )
    row["trajectory_summary"]["llm"]["provider_models"] = ["model-a"]
    row["trajectory_summary"]["llm"]["provider_model_identity_records"][0][
        "requested_model"
    ] = "model-a"
    row["trajectory_summary"]["llm"]["provider_model_identity_records"][0][
        "observed_models"
    ] = ["model-a"]
    monkeypatch.setattr(
        mod,
        "_primary_leaderboard_payload",
        lambda _rows: pytest.fail("stale suite evidence must block primary"),
    )
    coverage = {
        "configured_models": ["model-a"],
        "configured_seeds": [42],
        "pass_k": 1,
        "expected_total": 1,
        "comparable_intersection_size": 1,
        "is_partial_batch": False,
        "_comparable_pairs": [["scenario-a", 42]],
    }
    _bind_formal_artifacts(row, tmp_path, key="stale-suite")

    mod._write_leaderboard_json(
        tmp_path,
        [row],
        coverage=coverage,
        state={
            "batch_state": mod.BATCH_STATE_FINAL,
            "reasons": [],
            "n_orphan_interrupted_logs": 0,
        },
        formal=True,
        required_suite_hash="suite-new",
    )

    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    assert payload["formal_primary_blockers"] == [
        "formal_configured_episode_ineligible"
    ]
    assert payload["formal_configured_episode_failures"][0]["reasons"] == [
        "suite_manifest_mismatch"
    ]


def test_formal_leaderboard_excludes_models_outside_configured_scope(
    tmp_path: Path,
    monkeypatch,
) -> None:
    suite_eligibility = {"suite_blocked": False}

    def row(model: str) -> dict:
        return {
            "status": "ok",
            "interaction_mode": "logical_stateless",
            "scenario_id": "scenario-a",
            "scenario_slug": "scenario-a",
            "scenario_signature": "sig-a",
            "seed": 42,
            "model": model,
            "scoring_version": mod.SCORING_VERSION,
            "score": {"scoring_version": mod.SCORING_VERSION},
            "evaluation_protocol": {
                "version": mod.EVALUATION_PROTOCOL_VERSION,
                "implementation_fingerprint": (
                    mod.EVALUATION_IMPLEMENTATION_FINGERPRINT
                ),
            },
            "suite_manifest_sha256": "suite",
            "suite_eligibility": suite_eligibility,
            "suite_eligibility_sha256": mod._canonical_json_sha256(suite_eligibility),
            "trajectory_summary": {
                "llm": {"provider_models": [model]},
                "terminal_integrity": {"release_ready": True},
                "event_contract": {"schema_version": "1.0", "violation_count": 0},
                "provider_audit_artifact": _provider_audit_artifact(),
                "tool_surface_contract": _tool_surface_contract(),
                "tool_semantic_coverage": {
                    "covered": True,
                    "unknown_tool_names": [],
                    "unclassified_tool_names": [],
                    "explicit_semantic_roles_complete": True,
                    "missing_explicit_semantic_role_names": [],
                    "native_targets_complete": True,
                    "missing_native_target_kind_names": [],
                    "state_changing_actuators_complete": True,
                    "missing_actuator_family_names": [],
                },
            },
        }

    captured: list[dict] = []

    def fake_primary(rows: list[dict]) -> dict:
        captured.extend(rows)
        return {
            "leaderboard": [],
            "scoring_version": mod.SCORING_VERSION,
            "primary_leaderboard_formula_version": (
                "effective_source_backend_domain_macro_v1"
            ),
        }

    monkeypatch.setattr(mod, "_primary_leaderboard_payload", fake_primary)
    configured_row = _bind_formal_artifacts(
        row("model-a"), tmp_path, key="configured-model"
    )
    extra_row = _bind_formal_artifacts(row("model-extra"), tmp_path, key="extra-model")
    mod._write_leaderboard_json(
        tmp_path,
        [configured_row, extra_row],
        coverage={
            "configured_models": ["model-a"],
            "configured_seeds": [42],
            "pass_k": 1,
            "expected_total": 1,
            "comparable_intersection_size": 1,
            "is_partial_batch": False,
            "_comparable_pairs": [["scenario-a", 42]],
        },
        state={
            "batch_state": mod.BATCH_STATE_FINAL,
            "reasons": [],
            "n_orphan_interrupted_logs": 0,
        },
        formal=True,
    )

    assert [item["model"] for item in captured] == ["model-a"]
    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    assert payload["formal_primary_exclusions"][-1]["reasons"] == [
        "model_outside_configured_scope"
    ]


def test_write_leaderboard_json_includes_per_domain_view(tmp_path: Path) -> None:
    # Power-grid rows omit ``domain`` and must resolve via backend_kind; the
    # logistics rows carry an explicit ``domain``. Each domain gets its own
    # leaderboard so a heavily-populated domain does not dilute the headline.
    rows = [
        {
            "status": "ok",
            "interaction_mode": "logical_stateless",
            "model": "model-a",
            "backend_kind": "pandapower_acopf",
            "score": {"total_score": 30.0},
            "ground_truth_summary": {"chose_fatal_option": False},
        },
        {
            "status": "ok",
            "interaction_mode": "logical_stateless",
            "model": "model-b",
            "backend_kind": "pandapower_acopf",
            "score": {"total_score": 10.0},
            "ground_truth_summary": {"chose_fatal_option": False},
        },
        {
            "status": "ok",
            "interaction_mode": "logical_stateless",
            "model": "model-a",
            "domain": "logistics",
            "backend_kind": "jsplib_job_shop",
            "score": {"total_score": 5.0},
            "ground_truth_summary": {"chose_fatal_option": False},
        },
        {
            "status": "ok",
            "interaction_mode": "logical_stateless",
            "model": "model-b",
            "domain": "logistics",
            "backend_kind": "jsplib_job_shop",
            "score": {"total_score": 25.0},
            "ground_truth_summary": {"chose_fatal_option": False},
        },
    ]

    mod._write_leaderboard_json(tmp_path, rows)

    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    by_domain = payload["diagnostic_leaderboard_by_domain"]
    assert set(by_domain) == {"power_grid", "logistics"}
    # model-a leads power_grid; model-b leads logistics — opposite orderings,
    # which the diluted aggregate would have masked.
    assert by_domain["power_grid"][0]["agent_id"] == "model-a"
    assert by_domain["logistics"][0]["agent_id"] == "model-b"


def test_write_leaderboard_json_omits_adaptive_view_for_legacy_rows(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "status": "ok",
            "model": "legacy-model",
            "score": {
                "total_score": 10.0,
                "dimensions": [
                    {
                        "name": "system_survival",
                        "applicable": True,
                        "calibrated_score": 100.0,
                        "weight": 1.5,
                    }
                ],
            },
            "ground_truth_summary": {"chose_fatal_option": False},
        }
    ]

    mod._write_leaderboard_json(tmp_path, rows)

    payload = json.loads((tmp_path / "leaderboard.json").read_text())
    assert "fixed_all_dimensions" in payload["diagnostic_leaderboard_views"]
    assert "adaptive_applicable" not in payload["diagnostic_leaderboard_views"]


def test_load_episodes_jsonl_tolerates_malformed_trailing_line(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        '{"status":"ok","scenario_slug":"a","model":"m","seed":42}\n{"status":"ok"',
        encoding="utf-8",
    )
    rows = mod._load_episodes_jsonl(path)
    assert len(rows) == 1
    assert rows[0]["scenario_slug"] == "a"


def test_resume_repairs_malformed_trailing_line_before_append(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text(
        '{"status":"ok","scenario_slug":"a"}\n{"status":"in_f',
        encoding="utf-8",
    )

    rows = mod._load_episodes_jsonl(path, repair_trailing=True)
    assert [row["scenario_slug"] for row in rows] == ["a"]
    mod._append_jsonl_atomic(
        path,
        {"status": "ok", "scenario_slug": "b"},
    )

    assert [row["scenario_slug"] for row in mod._load_episodes_jsonl(path)] == [
        "a",
        "b",
    ]


def test_output_dir_lock_rejects_concurrent_runner(tmp_path: Path) -> None:
    first = mod._acquire_output_dir_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already has an active runner"):
            mod._acquire_output_dir_lock(tmp_path)
    finally:
        first.close()


def test_llm_config_round_trip_preserves_api_mode_and_responses_base_url() -> None:
    cfg = mod.LLMConfig(
        provider="azure",
        model="gpt-5.2-2025-12-11",
        base_url="http://chat-endpoint",
        responses_base_url="https://responses-endpoint",
        api_mode="responses",
        max_consecutive_provider_failures=7,
        provider_failure_policy="abort",
        stream_chat_completions=True,
        persistent_history_max_messages=18,
        persistent_context_max_chars=12_345,
        persistent_memory_max_items=11,
        tool_choice="required",
        tool_choice_supported=False,
        reasoning_effort="low",
        protocol_repair_max_tokens=333,
        provider_rpm_limit=20,
        provider_rpd_limit=1_000,
        provider_rate_limit_scope="openrouter-free-shared",
        model_context_window_tokens=128_000,
        model_max_output_tokens=16_384,
    )
    restored = mod._llm_config_from_dict(mod._llm_config_to_dict(cfg))
    assert restored.api_mode == "responses"
    assert restored.max_consecutive_provider_failures == 7
    assert restored.provider_failure_policy == "abort"
    assert restored.responses_base_url == "https://responses-endpoint"
    assert restored.stream_chat_completions is True
    assert restored.persistent_history_max_messages == 18
    assert restored.persistent_context_max_chars == 12_345
    assert restored.persistent_memory_max_items == 11
    assert restored.tool_choice == "required"
    assert restored.tool_choice_supported is False
    assert restored.reasoning_effort == "low"
    assert restored.protocol_repair_max_tokens == 333
    assert restored.provider_rpm_limit == 20
    assert restored.provider_rpd_limit == 1_000
    assert restored.provider_rate_limit_scope == "openrouter-free-shared"
    assert restored.model_context_window_tokens == 128_000
    assert restored.model_max_output_tokens == 16_384
    assert restored.token_count_method == "utf8_bytes_upper_bound"
    assert restored.token_count_version == "1"


def test_load_zhsrc_exports_reads_zshrc_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPERATE_MODELS", raising=False)
    monkeypatch.delenv("OPERATE_RESPONSES_API_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".zshrc").write_text(
        'export OPERATE_MODELS="gpt-5.2-2025-12-11,gemini-3.1-fl"\n'
        'export OPERATE_RESPONSES_API_BASE_URL="https://responses-endpoint"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(mod.Path, "home", staticmethod(lambda: home))
    env = mod._load_zhsrc_exports()
    assert env["OPERATE_MODELS"] == "gpt-5.2-2025-12-11,gemini-3.1-fl"
    assert env["OPERATE_RESPONSES_API_BASE_URL"] == "https://responses-endpoint"


def test_load_named_zshrc_export_supports_custom_api_key_env(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / ".zshrc").write_text(
        'export APYI_API_KEY="secret-test-value"\n', encoding="utf-8"
    )
    monkeypatch.setattr(mod.Path, "home", staticmethod(lambda: home))

    assert mod._load_named_zshrc_export("APYI_API_KEY") == "secret-test-value"
    assert mod._load_named_zshrc_export("INVALID-NAME") is None


def test_build_model_lanes_groups_jobs_by_model_preserving_job_order() -> None:
    jobs = [
        {"model": "m1", "scenario_slug": "s1", "seed": 42},
        {"model": "m2", "scenario_slug": "s1", "seed": 42},
        {"model": "m1", "scenario_slug": "s2", "seed": 42},
        {"model": "m2", "scenario_slug": "s2", "seed": 42},
    ]
    lanes = mod._build_model_lanes(jobs, episodes_path=Path("/tmp/episodes.jsonl"))
    assert [lane["model"] for lane in lanes] == ["m1", "m2"]
    assert [job["scenario_slug"] for job in lanes[0]["jobs"]] == ["s1", "s2"]
    assert [job["scenario_slug"] for job in lanes[1]["jobs"]] == ["s1", "s2"]
    assert lanes[0]["episodes_path"] == "/tmp/episodes.jsonl"


def test_per_model_scheduler_writes_scheduler_mode_and_uses_model_lanes(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "per_model"
    scenarios = [
        "power_grid/daily_ops_24h/time_pressure/basic/do_20200127_time_pressure_basic_s42",
        "power_grid/daily_ops_real_forecast_24h/time_pressure/medium/drf_20200127_time_pressure_medium_s42",
    ]
    models = ["gpt-5-2025-08-07", "gemini-3.1-fl"]

    scenario_map = {
        scenarios[0]: {
            "seed_id": "do_20200127_time_pressure_basic_s42",
            "family": "daily_ops_24h",
            "backend_kind": "pglib_uc_synthetic",
            "scenario_signature": "sig-daily",
        },
        scenarios[1]: {
            "seed_id": "drf_20200127_time_pressure_medium_s42",
            "family": "daily_ops_real_forecast_24h",
            "backend_kind": "pglib_uc_synthetic",
            "scenario_signature": "sig-drf",
        },
    }

    def fake_load_zhsrc_exports() -> dict[str, str]:
        return {
            "OPENAI_API_KEY": "test-key",
            "OPERATE_MODELS": ",".join(models),
            "OPERATE_API_BASE_URL": "http://example.test",
            "OPERATE_API_VERSION": "2024-03-01-preview",
        }

    def fake_expand(patterns: list[str]) -> list[str]:
        assert patterns == scenarios
        return list(scenarios)

    def fake_load_scenario_yaml(slug: str) -> dict[str, Any]:
        body = dict(scenario_map[slug])
        body.update(
            {
                "domain": "power_grid",
                "difficulty_mode": "time_pressure",
                "difficulty_level": "basic",
                "horizon_ticks": 24,
                "tick_minutes": 60,
            }
        )
        return body

    def fake_signature(scenario: dict[str, Any], seed: int) -> str:
        return f"{scenario['scenario_signature']}-seed{seed}"

    lane_events: list[tuple[str, list[str]]] = []

    def fake_run_lane(lane: dict[str, Any]) -> dict[str, Any]:
        episode_path = Path(lane["episodes_path"])
        seen = []
        for job in lane["jobs"]:
            scenario = scenario_map[job["scenario_slug"]]
            seen.append(job["scenario_slug"])
            row = _fake_episode_result(
                job["scenario_slug"],
                lane["model"],
                job["seed"],
                job["scenario_signature"],
                scenario["family"],
                scenario["backend_kind"],
                total_score=80.0 if lane["model"].startswith("gpt") else 70.0,
                n_tool_calls=4,
            )
            mod._append_jsonl_atomic(episode_path, row)
        lane_events.append((lane["model"], seen))
        return {"model": lane["model"], "n_completed": len(lane["jobs"])}

    class FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def map(self, fn, jobs, chunksize: int = 1):
            del chunksize
            for job in jobs:
                yield fn(job)

    monkeypatch.setattr(mod, "_load_zhsrc_exports", fake_load_zhsrc_exports)
    monkeypatch.setattr(mod, "_expand_scenarios", fake_expand)
    monkeypatch.setattr(mod, "load_scenario_yaml", fake_load_scenario_yaml)
    monkeypatch.setattr(mod, "_scenario_signature_for_run", fake_signature)
    monkeypatch.setattr(mod, "_run_llm_model_lane", fake_run_lane)
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {"git_commit": "abc", "git_dirty": False, "git_status_short": []},
    )

    import concurrent.futures as futures

    monkeypatch.setattr(futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(output_dir),
            "--scenario-slice",
            "custom",
            "--scenarios",
            *scenarios,
            "--models",
            ",".join(models),
            "--interaction-mode",
            "logical_stateless",
            "--seeds",
            "42",
            "--scheduler-mode",
            "per_model",
            "--max-workers",
            "6",
        ],
    )

    rc = mod.main()
    assert rc == 0

    run_cfg = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_cfg["scheduler_mode"] == "per_model"


def test_main_pass_k_writes_pass_ids_and_manifest_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "pass_k"
    scenario = "power_grid/daily_ops_24h/time_pressure/basic/do_20200127_time_pressure_basic_s42"
    model = "gpt-5-2025-08-07"

    def fake_load_zhsrc_exports() -> dict[str, str]:
        return {
            "OPENAI_API_KEY": "test-key",
            "OPERATE_MODELS": model,
        }

    def fake_expand(patterns: list[str]) -> list[str]:
        assert patterns == [scenario]
        return [scenario]

    def fake_load_scenario_yaml(slug: str) -> dict[str, Any]:
        assert slug == scenario
        return {
            "domain": "power_grid",
            "seed_id": "do_20200127_time_pressure_basic_s42",
            "family": "daily_ops_24h",
            "backend_kind": "pglib_uc_synthetic",
            "scenario_signature": "sig-daily",
            "difficulty_mode": "time_pressure",
            "difficulty_level": "basic",
            "horizon_ticks": 24,
            "tick_minutes": 60,
        }

    def fake_signature(scenario_body: dict[str, Any], seed: int) -> str:
        return f"{scenario_body['scenario_signature']}-seed{seed}"

    def fake_run_job(job: dict[str, Any]) -> dict[str, Any]:
        row = _fake_episode_result(
            job["scenario_slug"],
            job["model"],
            job["seed"],
            job["scenario_signature"],
            "daily_ops_24h",
            "pglib_uc_synthetic",
            total_score=70.0 + float(job["pass_index"]),
            n_tool_calls=4,
        )
        row["pass_id"] = job["pass_id"]
        row["pass_index"] = job["pass_index"]
        row["pass_k"] = job["pass_k"]
        Path(job["episode_log_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(job["episode_log_path"]).write_text("ok\n", encoding="utf-8")
        return mod._apply_llm_job_metadata(job, row)

    class FakeFuture:
        def __init__(self, row: dict[str, Any]) -> None:
            self.row = row

        def result(self) -> dict[str, Any]:
            return self.row

    class FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self) -> FakeExecutor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def submit(self, fn, job):
            del fn
            return FakeFuture(fake_run_job(job))

    monkeypatch.setattr(mod, "_load_zhsrc_exports", fake_load_zhsrc_exports)
    monkeypatch.setattr(mod, "_expand_scenarios", fake_expand)
    monkeypatch.setattr(mod, "load_scenario_yaml", fake_load_scenario_yaml)
    monkeypatch.setattr(mod, "_scenario_signature_for_run", fake_signature)
    monkeypatch.setattr(mod, "_run_llm_episode_job", fake_run_job)
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {"git_commit": "abc", "git_dirty": False, "git_status_short": []},
    )

    import concurrent.futures as futures

    monkeypatch.setattr(futures, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(futures, "as_completed", lambda fs: fs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(output_dir),
            "--scenario-slice",
            "custom",
            "--scenarios",
            scenario,
            "--models",
            model,
            "--interaction-mode",
            "logical_stateless",
            "--seeds",
            "42",
            "--pass-k",
            "2",
            "--max-workers",
            "1",
        ],
    )

    rc = mod.main()

    assert rc == 0
    rows = [
        json.loads(line)
        for line in (output_dir / "episodes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert [row["pass_id"] for row in rows] == ["pass-0", "pass-1"]
    assert [row["pass_index"] for row in rows] == [0, 1]
    assert {row["pass_k"] for row in rows} == {2}
    manifest = json.loads(
        (output_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["pass_k"] == 2
    assert manifest["n_episodes_total"] == 2
    assert manifest["expected_total"] == 2
    assert manifest["coverage"]["pass_k"] == 2
    assert manifest["coverage"]["expected_episodes_per_model"] == 2
    assert manifest["coverage"]["per_model_realized"] == {model: 2}
    assert manifest["pass_k_success"]["pass_k"] == 2
    assert manifest["pass_k_success"]["overall_success_probability"] == 1.0
    assert manifest["pass_k_success"]["per_model"][model]["successful_cells"] == 1
    assert manifest["comparable_intersection_size"] == 1
    assert "_comparable_pass_ids" not in manifest["coverage"]
    leaderboard_payload = json.loads(
        (output_dir / "leaderboard.json").read_text(encoding="utf-8")
    )
    assert leaderboard_payload["pass_k_success"] == manifest["pass_k_success"]


def test_finalize_only_recovers_real_scope_from_existing_batch_metadata_and_rows(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "storm_finalize"
    output_dir.mkdir(parents=True)
    (output_dir / "logs" / "gpt-5-2025-08-07").mkdir(parents=True)
    (output_dir / "trajectories").mkdir(parents=True)

    scenario_slug = (
        "power_grid/storm_emergency_6h/deep_planning/basic/"
        "st_chron0_deep_planning_basic_s42"
    )
    scenario_slug_2 = (
        "power_grid/storm_emergency_6h/deep_planning/cascading/"
        "st_chron1_deep_planning_cascading_s42"
    )
    model = "gpt-5-2025-08-07"
    row = _fake_episode_result(
        scenario_slug,
        model,
        42,
        "sig-storm",
        "storm_emergency_6h",
        "grid2op",
        total_score=33.0,
        n_tool_calls=4,
    )
    row["difficulty_mode"] = "deep_planning"
    row["difficulty_level"] = "basic"
    row["episode_log_path"] = str(output_dir / "logs" / model / "storm_episode.log")
    row2 = _fake_episode_result(
        scenario_slug_2,
        model,
        42,
        "sig-storm-2",
        "storm_emergency_6h",
        "grid2op",
        total_score=29.0,
        n_tool_calls=3,
    )
    row2["difficulty_mode"] = "deep_planning"
    row2["difficulty_level"] = "cascading"
    row2["episode_log_path"] = str(output_dir / "logs" / model / "storm_episode_2.log")
    Path(row["episode_log_path"]).write_text(
        '2026-05-28 18:47:28,401 [INFO] HTTP Request: "HTTP/1.1 200 OK"\n',
        encoding="utf-8",
    )
    Path(row2["episode_log_path"]).write_text(
        '2026-05-28 18:47:29,111 [INFO] HTTP Request: "HTTP/1.1 200 OK"\n',
        encoding="utf-8",
    )
    (output_dir / "episodes.jsonl").write_text(
        json.dumps(row, ensure_ascii=False)
        + "\n"
        + json.dumps(row2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "runner_version": "0.2.3-formalized",
                "scenario_slice": "custom",
                "patterns": [
                    "power_grid/daily_ops_24h/time_pressure/basic/do_20200127_time_pressure_basic_s42"
                ],
                "n_scenarios": 1,
                "models": [model],
                "seeds": [42],
                "temperature": 1.0,
                "scheduler_mode": "global",
                "base_url": "http://example.test",
                "api_version": "2024-03-01-preview",
                "responses_base_url": "https://responses.example.test",
                "api_mode": "auto",
                "max_workers_requested": 6,
                "max_workers_effective": 6,
                "save_trajectories": True,
                "resume_enabled": True,
                "finalize_enabled": True,
                "finalize_only": True,
                "started_at_utc": "2026-05-29T00:00:00+00:00",
                "has_grid2op": False,
                "git_commit": "abc",
                "git_dirty": False,
                "git_status_short": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_load_zhsrc_exports() -> dict[str, str]:
        return {"OPENAI_API_KEY": "test-key", "OPERATE_MODELS": model}

    def fake_expand(patterns: list[str]) -> list[str]:
        assert patterns == [scenario_slug, scenario_slug_2]
        return [scenario_slug, scenario_slug_2]

    def fake_load_scenario_yaml(slug: str) -> dict[str, Any]:
        return {
            "domain": "power_grid",
            "family": "storm_emergency_6h",
            "backend_kind": "grid2op",
            "difficulty_mode": "deep_planning",
            "difficulty_level": "basic",
            "seed_id": slug.rsplit("/", 1)[-1],
            "scenario_signature": (
                "sig-storm" if slug == scenario_slug else "sig-storm-2"
            ),
        }

    monkeypatch.setattr(mod, "_load_zhsrc_exports", fake_load_zhsrc_exports)
    monkeypatch.setattr(mod, "_expand_scenarios", fake_expand)
    monkeypatch.setattr(mod, "load_scenario_yaml", fake_load_scenario_yaml)
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {"git_commit": "abc", "git_dirty": False, "git_status_short": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_llm_eval.py", "--output-dir", str(output_dir), "--finalize-only"],
    )

    original_config = (output_dir / "run_config.json").read_bytes()
    rc = mod.main()

    assert rc == 1
    assert (output_dir / "run_config.json").read_bytes() == original_config
    assert not (output_dir / "RUN_MANIFEST.json").exists()


# ─────────────────────────────────────────────────────────────────────────────
def test_formal_finalize_only_uses_recovered_meta_for_exit_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "formal_finalize_only"
    output_dir.mkdir()
    scenario = "traffic/example"
    models = ["model-a", "model-b", "model-c"]
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "formal_run": True,
                "scenario_slice": "custom",
                "patterns": [scenario],
                "n_scenarios": 1,
                "models": models,
                "seeds": [42],
                "seed_mode": "scenario",
                "scenario_seed_pairs": [[scenario, 42]],
                "pass_k": 3,
                "temperature": 1.0,
                "prompt_mode": "strict",
                "scheduler_mode": "global",
                "save_trajectories": True,
                "finalize_enabled": True,
                "max_workers_requested": 4,
                "max_workers_effective": 4,
                "implementation_tree_sha256": "tree",
                "git_commit": "commit",
                "git_metadata_available": True,
                "git_dirty": False,
            }
        ),
        encoding="utf-8",
    )
    (output_dir / "episodes.jsonl").write_text(
        json.dumps(
            {
                "status": "ok",
                "scenario_slug": scenario,
                "model": models[0],
                "seed": 42,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "_load_zhsrc_exports", lambda: {})
    monkeypatch.setattr(mod, "_expand_scenarios", lambda _patterns: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda _slug: {"backend_kind": "mock", "seed": 42},
    )
    monkeypatch.setattr(mod, "_suite_manifest_sha256", lambda *_args: "suite")
    monkeypatch.setattr(mod, "_suite_manifest_sha256_for_slice", lambda *_args: "suite")
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {
            "git_commit": "commit",
            "git_metadata_available": True,
            "git_dirty": False,
            "git_status_short": [],
        },
    )
    monkeypatch.setattr(
        mod,
        "implementation_identity",
        lambda *_args: {"implementation_tree_sha256": "tree"},
    )
    monkeypatch.setattr(mod, "_formal_runtime_binding_reasons", lambda *_args: [])
    monkeypatch.setattr(
        mod,
        "_finalize_outputs",
        lambda _out, _rows, meta: {
            "leaderboard_eligible": False,
            "n_episodes_ok": 1,
            "n_episodes_error": 0,
            "artifacts": {"plots": []},
            "formal_run": meta["formal_run"],
        },
    )
    monkeypatch.setattr(mod, "_print_batch_leaderboard", lambda _out: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["batch_llm_eval.py", "--output-dir", str(output_dir), "--finalize-only"],
    )

    assert mod.main() == 1


# v0.2.4 P1-A1 regression: LLMConfig round-trip must preserve prompt_mode
# and extra_headers across the worker-process boundary.
# ─────────────────────────────────────────────────────────────────────────────


def test_llm_config_roundtrip_preserves_prompt_mode_strict() -> None:
    from baselines.llm_agent import LLMConfig

    cfg = LLMConfig(provider="openai", model="gpt-test", prompt_mode="strict")
    d = mod._llm_config_to_dict(cfg)
    assert d["prompt_mode"] == "strict"
    cfg2 = mod._llm_config_from_dict(d)
    assert cfg2.prompt_mode == "strict"


def test_llm_config_roundtrip_preserves_prompt_mode_debug() -> None:
    from baselines.llm_agent import LLMConfig

    cfg = LLMConfig(provider="openai", model="gpt-test", prompt_mode="debug")
    d = mod._llm_config_to_dict(cfg)
    assert d["prompt_mode"] == "debug"
    cfg2 = mod._llm_config_from_dict(d)
    # Without v0.2.4 fix, this resurfaced as "strict" silently.
    assert cfg2.prompt_mode == "debug"


def test_llm_config_roundtrip_preserves_extra_headers() -> None:
    from baselines.llm_agent import LLMConfig

    headers = {"Ocp-Apim-Subscription-Key": "secret", "X-Custom": "1"}
    cfg = LLMConfig(provider="azure", model="gpt-test", extra_headers=headers)
    d = mod._llm_config_to_dict(cfg)
    assert d["extra_headers"] == headers
    cfg2 = mod._llm_config_from_dict(d)
    assert cfg2.extra_headers == headers


def test_llm_config_roundtrip_preserves_insecure_transport_opt_in() -> None:
    cfg = mod.LLMConfig(
        provider="openai_compatible",
        model="diagnostic",
        allow_insecure_http=True,
    )

    cfg2 = mod._llm_config_from_dict(mod._llm_config_to_dict(cfg))

    assert cfg2.allow_insecure_http is True


def test_llm_config_legacy_payload_defaults_to_strict() -> None:
    """Worker payloads serialized before v0.2.4 will not carry
    ``prompt_mode`` or ``extra_headers``; the loader must default
    them to ``strict`` / ``{}`` rather than crash or pick up host-
    process defaults."""
    cfg = mod._llm_config_from_dict({"provider": "openai", "model": "gpt-legacy"})
    assert cfg.prompt_mode == "strict"
    assert cfg.extra_headers == {}


# ─────────────────────────────────────────────────────────────────────────────
# 2026-06-02 partial-batch / comparable intersection regression tests.
# These exercise ``_coverage_summary`` and ``_intersection_leaderboard``
# directly (no subprocess / no main()) so the contract is locked in even if
# higher-level pipeline tests change.
# ─────────────────────────────────────────────────────────────────────────────


def _row(slug: str, model: str, seed: int, score: float, status: str = "ok") -> dict:
    return {
        "scenario_slug": slug,
        "model": model,
        "agent_name": f"llm_agent/{model}",
        "seed": seed,
        "status": status,
        "interaction_mode": "logical_stateless",
        "score": {"total_score": score},
        "ground_truth_summary": {"chose_fatal_option": False},
    }


def _job(slug: str, model: str, seed: int = 42) -> dict[str, Any]:
    return {
        "scenario_slug": slug,
        "model": model,
        "seed": seed,
        "scenario_signature": f"sig::{slug}::{seed}",
        "temperature": 1.0,
        "episode_log_path": f"/tmp/logs/{model}/{slug.replace('/', '_')}_s{seed}.log",
    }


def test_write_analysis_excludes_dirty_ok_from_score_means(tmp_path: Path) -> None:
    clean = _row("logistics/inv/basic", "hy3-ioa", 42, 10.0)
    dirty = _row("traffic/ingolstadt21", "hy3-ioa", 42, 100.0)
    dirty["trajectory_summary"] = {
        "n_tool_calls": 20,
        "n_wait_actions": 18,
        "llm": {
            "llm_calls_ok": 2,
            "llm_calls_failed": 18,
            "fallback_wait_ratio": 0.90,
        },
    }
    mod._write_analysis(tmp_path, [clean, dirty])
    analysis = (tmp_path / "ANALYSIS.md").read_text(encoding="utf-8")
    assert "| hy3-ioa | 10.00 | 1 |" in analysis
    assert "Dirty OK excluded from score means: 1" in analysis
    assert "Clean OK used for score means: 1" in analysis
    stats = json.loads((tmp_path / "stats_by_model.json").read_text(encoding="utf-8"))
    assert stats["by_model"]["hy3-ioa"] == {"mean": 10.0, "n": 1}
    assert stats["n_ok"] == 2
    assert stats["n_clean_ok"] == 1
    assert stats["n_dirty_ok"] == 1
    assert stats["tool_stats"]["hy3-ioa"]["n_episodes"] == 2.0


def test_model_label_strips_llm_agent_prefix() -> None:
    assert mod._model_label({"model": "llm_agent/gpt-5"}) == "gpt-5"
    assert mod._model_label({"model": "gpt-5"}) == "gpt-5"
    assert mod._model_label({"agent_name": "llm_agent/o3"}) == "o3"
    assert mod._model_label({}) == ""


def test_coverage_summary_full_batch_no_warning_intersection_equals_expected() -> None:
    models = ["gpt-5", "o3"]
    seeds = [42]
    slugs = ["fam/scen_a", "fam/scen_b"]
    results = [_row(s, m, 42, 80.0) for s in slugs for m in models]

    cov = mod._coverage_summary(
        results,
        configured_models=models,
        configured_seeds=seeds,
        n_scenarios=len(slugs),
    )
    assert cov["expected_episodes_per_model"] == 2
    assert cov["expected_total"] == 4
    assert cov["per_model_realized"] == {"gpt-5": 2, "o3": 2}
    assert cov["per_model_coverage"] == {"gpt-5": 1.0, "o3": 1.0}
    assert cov["comparable_intersection_size"] == 2
    assert cov["is_partial_batch"] is False
    assert cov["comparability_warning"] is None
    assert sorted(cov["_comparable_pairs"]) == [
        ["fam/scen_a", 42],
        ["fam/scen_b", 42],
    ]


def test_coverage_summary_counts_pass_k_units_when_configured() -> None:
    models = ["gpt-5", "o3"]
    results = [
        {**_row("fam/a", model, 42, 80.0), "pass_id": pass_id}
        for model in models
        for pass_id in ("pass-0", "pass-1")
    ]

    cov = mod._coverage_summary(
        results,
        configured_models=models,
        configured_seeds=[42],
        n_scenarios=1,
        pass_k=2,
    )

    assert cov["pass_k"] == 2
    assert cov["expected_episodes_per_model"] == 2
    assert cov["expected_total"] == 4
    assert cov["per_model_realized"] == {"gpt-5": 2, "o3": 2}
    assert cov["per_model_coverage"] == {"gpt-5": 1.0, "o3": 1.0}
    assert cov["per_model_execution_realized"] == {"gpt-5": 2, "o3": 2}
    assert cov["per_model_execution_coverage"] == {"gpt-5": 1.0, "o3": 1.0}
    assert cov["comparable_intersection_size"] == 1
    assert cov["is_partial_batch"] is False
    assert cov["_comparable_pairs"] == [["fam/a", 42]]


def test_resume_retries_treatment_bound_rows_with_missing_or_tampered_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _bind_formal_artifacts(
        _formally_eligible_protocol21_row(), tmp_path, key="resume-cell"
    )
    row.update(
        {
            "scenario_slug": "fam/resume",
            "model": "hy3-ioa",
            "seed": 42,
            "scenario_signature": "sig-resume",
            "temperature": 0.0,
            "pass_id": "pass-0",
        }
    )
    row["trajectory_summary"]["llm"]["provider_models"] = ["hy3-ioa"]
    row["trajectory_summary"]["llm"]["provider_model_identity_records"][0][
        "requested_model"
    ] = "hy3-ioa"
    row["trajectory_summary"]["llm"]["provider_model_identity_records"][0][
        "observed_models"
    ] = ["hy3-ioa"]
    job = {
        "scenario_slug": row["scenario_slug"],
        "model": row["model"],
        "seed": row["seed"],
        "scenario_signature": row["scenario_signature"],
        "temperature": row["temperature"],
        "pass_id": row["pass_id"],
        "evaluation_implementation_fingerprint": (
            mod.EVALUATION_IMPLEMENTATION_FINGERPRINT
        ),
        "suite_manifest_sha256": row["suite_manifest_sha256"],
        "suite_eligibility_sha256": row["suite_eligibility_sha256"],
    }

    assert mod._filter_pending_jobs([job], [row]) == []

    portable = mod._portable_formal_result_paths([row], batch_root=tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert mod._filter_pending_jobs([job], portable, batch_root=tmp_path) == []

    provider_path = Path(row["trajectory_summary"]["provider_audit_artifact"]["path"])
    provider_payload = provider_path.read_bytes()
    provider_path.unlink()
    assert mod._filter_pending_jobs([job], [row]) == [job]
    assert mod._filter_pending_jobs([job], portable, batch_root=tmp_path) == [job]

    provider_path.write_bytes(provider_payload)
    semantic_path = Path(row["trajectory_summary"]["semantic_ledger_artifact"]["path"])
    semantic_path.write_bytes(semantic_path.read_bytes() + b'{"tampered":true}\n')
    assert mod._filter_pending_jobs([job], [row]) == [job]


def test_coverage_summary_dedupes_duplicate_pass_ids() -> None:
    results = [
        {**_row("fam/a", "gpt-5", 42, 80.0), "pass_id": "pass-0"},
        {**_row("fam/a", "gpt-5", 42, 81.0), "pass_id": "pass-0"},
    ]

    cov = mod._coverage_summary(
        results,
        configured_models=["gpt-5"],
        configured_seeds=[42],
        n_scenarios=1,
        pass_k=2,
    )

    assert cov["expected_episodes_per_model"] == 2
    assert cov["per_model_realized"] == {"gpt-5": 1}
    assert cov["per_model_coverage"] == {"gpt-5": 0.5}
    assert cov["comparable_intersection_size"] == 0
    assert cov["is_partial_batch"] is True


def test_pass_k_success_summary_requires_all_configured_passes_ok() -> None:
    results = [
        {**_row("fam/a", "gpt-5", 42, 80.0), "pass_id": "pass-0"},
        {**_row("fam/a", "gpt-5", 42, 81.0), "pass_id": "pass-1"},
        {**_row("fam/b", "gpt-5", 42, 82.0), "pass_id": "pass-0"},
        {**_row("fam/b", "gpt-5", 42, 0.0, status="error"), "pass_id": "pass-1"},
        {**_row("fam/a", "o3", 42, 70.0), "pass_id": "pass-0"},
        {**_row("fam/a", "o3", 42, 71.0), "pass_id": "pass-1"},
        {**_row("fam/b", "o3", 42, 72.0), "pass_id": "pass-0"},
        # duplicate pass-0 must not satisfy pass-1
        {**_row("fam/b", "o3", 42, 73.0), "pass_id": "pass-0"},
    ]

    summary = mod._pass_k_success_summary(
        results,
        configured_models=["gpt-5", "o3"],
        configured_seeds=[42],
        n_scenarios=2,
        pass_k=2,
    )

    assert summary["pass_k"] == 2
    assert summary["required_pass_ids"] == ["pass-0", "pass-1"]
    assert summary["expected_cells_per_model"] == 2
    assert summary["overall_success_probability"] == 0.5
    assert summary["per_model"]["gpt-5"]["successful_cells"] == 1
    assert summary["per_model"]["gpt-5"]["failed_cells"] == 1
    assert summary["per_model"]["gpt-5"]["missing_pass_units"] == 0
    assert summary["per_model"]["gpt-5"]["error_pass_units"] == 1
    assert summary["per_model"]["o3"]["successful_cells"] == 1
    assert summary["per_model"]["o3"]["failed_cells"] == 1
    assert summary["per_model"]["o3"]["missing_pass_units"] == 1
    assert summary["per_model"]["o3"]["duplicate_pass_units_ignored"] == 1


def test_pass_k_success_summary_is_seed_aware() -> None:
    results = [
        {**_row("fam/a", "gpt-5", 42, 80.0), "pass_id": "pass-0"},
        {**_row("fam/a", "gpt-5", 42, 81.0), "pass_id": "pass-1"},
        {**_row("fam/a", "gpt-5", 43, 82.0), "pass_id": "pass-0"},
        {**_row("fam/a", "gpt-5", 43, 83.0), "pass_id": "pass-1"},
    ]

    summary = mod._pass_k_success_summary(
        results,
        configured_models=["gpt-5"],
        configured_seeds=[42, 43],
        n_scenarios=1,
        pass_k=2,
    )

    assert summary["expected_cells_per_model"] == 2
    assert summary["per_model"]["gpt-5"]["successful_cells"] == 2
    assert summary["per_model"]["gpt-5"]["success_probability"] == 1.0


def _formal_eligibility_inputs() -> tuple[dict, dict, dict, dict, dict]:
    models = ["model-a"]
    meta = {
        "formal_run": True,
        "scenario_slice": "v0_54_protocol2_v21_core_candidate",
        "formal_manifest": "release/operate_v0_58_0/manifest.json",
        "formal_evaluation_ready": True,
        "suite_manifest_sha256": "a" * 64,
        "suite_eligibility": {
            "suite_blocked": False,
            "formal_evaluation_ready": True,
            "scoring_version": mod.SCORING_VERSION,
            "primary_leaderboard_formula_version": (
                mod.PRIMARY_LEADERBOARD_FORMULA_VERSION
            ),
            "primary_inference_version": mod.PRIMARY_INFERENCE_VERSION,
            "task_completion_input_unit": mod.TASK_COMPLETION_INPUT_UNIT,
            "task_completion_score_unit": mod.TASK_COMPLETION_SCORE_UNIT,
            "weighted_equity_formula_version": (mod.WEIGHTED_EQUITY_FORMULA_VERSION),
            "readiness_source_binding_valid": True,
            "suite_manifest_sha256": "a" * 64,
            "formal_run_contract": {
                "contract_version": "agentic_persistent.v1",
                "wakeup_policy": dict(mod.CANONICAL_WAKEUP_POLICY),
                "required_model_count_per_shard": 1,
                "minimum_pass_k": 1,
                "minimum_max_workers": 1,
                "maximum_max_workers": 32,
                "required_temperature": 0.0,
                "required_interaction_mode": "logical_persistent",
            },
        },
        "git_dirty": False,
        "git_metadata_available": True,
        "git_commit": "b" * 40,
        "git_commit_end": "b" * 40,
        "implementation_tree_stable": True,
        "models": models,
        "n_scenarios": 1,
        "scenario_seed_pairs": [["scenario-a", 42]],
        "pass_k": 1,
        "max_workers_requested": 4,
        "max_workers_effective": 4,
        "temperature": 0.0,
        "prompt_mode": "strict",
        "interaction_mode": "logical_persistent",
        "seed_mode": "scenario",
        "scheduler_mode": "global",
        "save_trajectories": True,
        "finalize_enabled": True,
        "diagnostic_only": False,
        "formal_runtime_binding_stable": True,
        "provider_rpm_limit": 20,
        "provider_rpd_limit": 1_000,
        "provider_rate_limit_scope": "formal-provider-fixture",
    }
    coverage = {
        "configured_models": models,
        "seed_mode": "scenario",
        "expected_total": 1,
        "expected_episodes_per_model": 1,
        "pass_k": 1,
        "n_scenarios": 1,
        "per_model_realized": {model: 1 for model in models},
        "per_model_coverage": {model: 1.0 for model in models},
        "comparable_intersection_size": 1,
        "is_partial_batch": False,
        "_comparable_pairs": [["scenario-a", 42]],
    }
    pass_k_success = {
        "pass_k": 1,
        "configured_models": models,
        "seed_mode": "scenario",
        "n_scenarios": 1,
        "expected_cells_per_model": 1,
        "expected_total_cells": 1,
        "successful_total_cells": 1,
        "overall_success_probability": 1.0,
        "per_model": {
            model: {
                "expected_cells": 1,
                "successful_cells": 1,
                "failed_cells": 0,
                "success_probability": 1.0,
                "missing_pass_units": 0,
                "error_pass_units": 0,
            }
            for model in models
        },
    }
    state = {
        "batch_state": mod.BATCH_STATE_FINAL,
        "reasons": [],
        "n_orphan_interrupted_logs": 0,
    }
    leaderboard_payload = {
        "primary_leaderboard": [{"model": model} for model in models],
    }
    return meta, coverage, pass_k_success, state, leaderboard_payload


def test_formal_leaderboard_eligibility_certifies_complete_run() -> None:
    eligibility = mod._formal_leaderboard_eligibility(*_formal_eligibility_inputs())

    assert eligibility == {"eligible": True, "blockers": []}


def test_formal_leaderboard_eligibility_fails_closed() -> None:
    meta, coverage, pass_k_success, state, leaderboard_payload = (
        _formal_eligibility_inputs()
    )
    meta["git_dirty"] = True
    coverage["is_partial_batch"] = True
    coverage["comparable_intersection_size"] = 0
    pass_k_success["successful_total_cells"] = 0
    pass_k_success["overall_success_probability"] = 0.0
    state["batch_state"] = mod.BATCH_STATE_DEGRADED
    leaderboard_payload = {"formal_primary_blockers": ["bad_episode"]}

    eligibility = mod._formal_leaderboard_eligibility(
        meta, coverage, pass_k_success, state, leaderboard_payload
    )

    assert eligibility["eligible"] is False
    assert eligibility["blockers"] == [
        "formal_batch_not_final",
        "formal_comparable_intersection_incomplete",
        "formal_coverage_incomplete",
        "formal_git_tree_must_be_clean",
        "formal_pass_k_incomplete",
        "formal_primary_blocker:bad_episode",
        "formal_primary_leaderboard_missing",
    ]


@pytest.mark.parametrize(
    ("field", "value", "blocker"),
    [
        ("prompt_mode", "debug", "formal_prompt_mode_must_be_strict"),
        ("seed_mode", "fixed", "formal_seed_mode_must_be_scenario"),
        ("scheduler_mode", "per_model", "formal_scheduler_must_be_global"),
        ("temperature", 1.0, "formal_temperature_must_equal_zero"),
        ("max_workers_requested", 36, "formal_max_workers_out_of_range"),
        ("max_workers_effective", 36, "formal_effective_workers_mismatch"),
        ("save_trajectories", False, "formal_trajectories_required"),
        ("finalize_enabled", False, "formal_finalization_required"),
        ("diagnostic_only", True, "formal_diagnostic_only_forbidden"),
        ("git_metadata_available", False, "formal_git_metadata_unavailable"),
    ],
)
def test_formal_leaderboard_eligibility_rechecks_stored_contract(
    field: str, value: object, blocker: str
) -> None:
    meta, coverage, pass_k_success, state, leaderboard_payload = (
        _formal_eligibility_inputs()
    )
    meta[field] = value

    eligibility = mod._formal_leaderboard_eligibility(
        meta, coverage, pass_k_success, state, leaderboard_payload
    )

    assert blocker in eligibility["blockers"]


@pytest.mark.parametrize(
    "mutation,blocker",
    [
        ("coverage_models", "formal_coverage_model_scope_mismatch"),
        ("coverage_seed_mode", "formal_coverage_seed_mode_mismatch"),
        ("coverage_pairs", "formal_coverage_scenario_seed_scope_mismatch"),
        ("coverage_total", "formal_coverage_expected_total_mismatch"),
        ("coverage_realized", "formal_coverage_realized_mismatch"),
        ("coverage_pass_k", "formal_coverage_pass_k_mismatch"),
        ("pass_models", "formal_pass_k_model_scope_mismatch"),
        ("pass_seed_mode", "formal_pass_k_seed_mode_mismatch"),
        ("pass_per_model", "formal_pass_k_per_model_incomplete"),
        ("pass_total", "formal_pass_k_expected_total_mismatch"),
        ("scenario_seed_pairs", "formal_scenario_seed_scope_invalid"),
        ("leaderboard_models", "formal_leaderboard_model_scope_mismatch"),
    ],
)
def test_formal_leaderboard_eligibility_cross_binds_scope(
    mutation: str, blocker: str
) -> None:
    meta, coverage, pass_k_success, state, leaderboard_payload = (
        _formal_eligibility_inputs()
    )
    if mutation == "coverage_models":
        coverage["configured_models"] = ["model-a", "model-b", "other"]
    elif mutation == "coverage_seed_mode":
        coverage["seed_mode"] = "fixed"
    elif mutation == "coverage_pairs":
        coverage["_comparable_pairs"] = [["other-scenario", 42]]
    elif mutation == "coverage_total":
        coverage["expected_total"] = 3
    elif mutation == "coverage_realized":
        coverage["per_model_realized"]["model-a"] = 2
    elif mutation == "coverage_pass_k":
        coverage["pass_k"] = 2
    elif mutation == "pass_models":
        pass_k_success["configured_models"] = ["model-a", "model-b", "other"]
    elif mutation == "pass_seed_mode":
        pass_k_success["seed_mode"] = "fixed"
    elif mutation == "pass_per_model":
        pass_k_success["per_model"]["model-a"]["successful_cells"] = 0
        pass_k_success["per_model"]["model-a"]["failed_cells"] = 1
    elif mutation == "pass_total":
        pass_k_success["expected_total_cells"] = 2
        pass_k_success["successful_total_cells"] = 1
    elif mutation == "scenario_seed_pairs":
        meta["scenario_seed_pairs"] = None
    else:
        leaderboard_payload["primary_leaderboard"][-1]["model"] = "other"

    eligibility = mod._formal_leaderboard_eligibility(
        meta, coverage, pass_k_success, state, leaderboard_payload
    )

    assert blocker in eligibility["blockers"]


def test_intersection_leaderboard_limits_rows_to_configured_pass_k() -> None:
    results = [
        {**_row("fam/a", "gpt-5", 42, 10.0), "pass_id": "pass-0"},
        {**_row("fam/a", "gpt-5", 42, 12.0), "pass_id": "pass-1"},
        {**_row("fam/a", "gpt-5", 42, 1000.0), "pass_id": "pass-2"},
        {**_row("fam/a", "gpt-5", 42, 999.0), "pass_id": "pass-1"},
        {**_row("fam/a", "o3", 42, 8.0), "pass_id": "pass-0"},
        {**_row("fam/a", "o3", 42, 9.0), "pass_id": "pass-1"},
    ]
    coverage = mod._coverage_summary(
        results,
        configured_models=["gpt-5", "o3"],
        configured_seeds=[42],
        n_scenarios=1,
        pass_k=2,
    )

    rows = mod._intersection_leaderboard(results, coverage)

    by_id = {row["agent_id"]: row for row in rows}
    assert by_id["gpt-5"]["n_episodes"] == 2
    assert by_id["gpt-5"]["mean"] == 11.0
    assert by_id["o3"]["n_episodes"] == 2
    assert by_id["o3"]["mean"] == 8.5


def test_coverage_summary_partial_batch_emits_warning_and_shrinks_intersection() -> (
    None
):
    models = ["gpt-5", "o3", "gemini"]
    seeds = [42]
    slugs = ["fam/x", "fam/y", "fam/z"]
    # gpt-5 covers all 3; o3 covers 2; gemini covers only the first.
    results = [
        _row("fam/x", "gpt-5", 42, 70.0),
        _row("fam/y", "gpt-5", 42, 71.0),
        _row("fam/z", "gpt-5", 42, 72.0),
        _row("fam/x", "o3", 42, 60.0),
        _row("fam/y", "o3", 42, 61.0),
        _row("fam/x", "gemini", 42, 55.0),
        # An error row must NOT count toward coverage even though slug/seed match.
        _row("fam/y", "gemini", 42, 0.0, status="error"),
    ]
    cov = mod._coverage_summary(
        results,
        configured_models=models,
        configured_seeds=seeds,
        n_scenarios=len(slugs),
    )
    assert cov["expected_episodes_per_model"] == 3
    assert cov["expected_total"] == 9
    assert cov["per_model_realized"] == {"gpt-5": 3, "o3": 2, "gemini": 1}
    assert cov["per_model_coverage"]["gemini"] < 1.0
    # Intersection is just the single (slug, seed) every model completed.
    assert cov["comparable_intersection_size"] == 1
    assert cov["_comparable_pairs"] == [["fam/x", 42]]
    assert cov["is_partial_batch"] is True
    assert cov["comparability_warning"]
    assert "Partial batch" in cov["comparability_warning"]
    assert "intersection_leaderboard" in cov["comparability_warning"]


def test_coverage_summary_handles_zero_realized_for_a_configured_model() -> None:
    """A model with NO ok rows must still appear with realized=0 / coverage=0
    and force ``is_partial_batch=True``."""
    models = ["gpt-5", "ghost"]
    cov = mod._coverage_summary(
        [_row("fam/a", "gpt-5", 42, 80.0)],
        configured_models=models,
        configured_seeds=[42],
        n_scenarios=1,
    )
    assert cov["per_model_realized"]["ghost"] == 0
    assert cov["per_model_coverage"]["ghost"] == 0.0
    assert cov["is_partial_batch"] is True
    # No model overlap → empty intersection.
    assert cov["comparable_intersection_size"] == 0


def test_intersection_leaderboard_restricts_to_common_pairs_only() -> None:
    models = ["gpt-5", "o3"]
    # Both models cover (x, 42); only gpt-5 covers (y, 42).
    results = [
        _row("fam/x", "gpt-5", 42, 90.0),
        _row("fam/y", "gpt-5", 42, 30.0),  # this MUST be excluded
        _row("fam/x", "o3", 42, 50.0),
    ]
    cov = mod._coverage_summary(
        results,
        configured_models=models,
        configured_seeds=[42],
        n_scenarios=2,
    )
    rows = mod._intersection_leaderboard(results, cov)
    assert {r["agent_id"] for r in rows} == set(models)
    by_id = {r["agent_id"]: r for r in rows}
    # Each model has exactly 1 episode in the intersection.
    assert by_id["gpt-5"]["n_episodes"] == 1
    assert by_id["o3"]["n_episodes"] == 1
    # Mean must come from only the (x, 42) cell, NOT include (y, 42)=30.0.
    assert abs(by_id["gpt-5"]["mean"] - 90.0) < 1e-9
    assert abs(by_id["o3"]["mean"] - 50.0) < 1e-9


def test_intersection_leaderboard_empty_when_no_common_pairs() -> None:
    cov = mod._coverage_summary(
        [
            _row("fam/x", "gpt-5", 42, 80.0),
            _row("fam/y", "o3", 42, 60.0),
        ],
        configured_models=["gpt-5", "o3"],
        configured_seeds=[42],
        n_scenarios=2,
    )
    assert cov["comparable_intersection_size"] == 0
    rows = mod._intersection_leaderboard(
        [
            _row("fam/x", "gpt-5", 42, 80.0),
            _row("fam/y", "o3", 42, 60.0),
        ],
        cov,
    )
    assert rows == []


def test_coverage_summary_dedupes_repeated_seeds_in_meta() -> None:
    """A duplicated ``--seeds 42 42`` must NOT inflate ``expected_per_model``
    or falsely flag a complete batch as partial."""
    results = [
        _row("fam/a", "gpt-5", 42, 80.0),
        _row("fam/a", "o3", 42, 70.0),
    ]
    cov = mod._coverage_summary(
        results,
        configured_models=["gpt-5", "o3"],
        configured_seeds=[42, 42, 42],
        n_scenarios=1,
    )
    assert cov["expected_episodes_per_model"] == 1
    assert cov["expected_total"] == 2
    assert cov["configured_seeds"] == [42]
    assert cov["is_partial_batch"] is False
    assert cov["comparability_warning"] is None
    assert cov["comparable_intersection_size"] == 1


def test_coverage_summary_isolates_non_configured_models_to_extras() -> None:
    """Baseline / legacy rows for non-configured models must NOT pollute
    ``per_model_realized`` and must surface in ``extra_models_seen``."""
    results = [
        _row("fam/a", "gpt-5", 42, 80.0),
        _row("fam/a", "o3", 42, 70.0),
        # Non-configured baseline rows that historically leaked into the
        # coverage table — they belong in ``extra_models_seen`` only.
        _row("fam/a", "wait_only", 42, 10.0),
        _row("fam/a", "greedy_heuristic", 42, 12.0),
    ]
    cov = mod._coverage_summary(
        results,
        configured_models=["gpt-5", "o3"],
        configured_seeds=[42],
        n_scenarios=1,
    )
    assert set(cov["per_model_realized"]) == {"gpt-5", "o3"}
    assert set(cov["per_model_coverage"]) == {"gpt-5", "o3"}
    assert "wait_only" not in cov["per_model_coverage"]
    assert "greedy_heuristic" not in cov["per_model_coverage"]
    assert cov["extra_models_seen"] == ["greedy_heuristic", "wait_only"]
    # Two configured models, both fully covered → not partial.
    assert cov["is_partial_batch"] is False


def test_coverage_summary_filters_results_by_configured_seeds() -> None:
    """An ``ok`` row whose seed isn't in ``configured_seeds`` must not
    contribute to coverage (e.g. resume after the seed schema changed)."""
    results = [
        _row("fam/a", "gpt-5", 42, 80.0),
        _row("fam/a", "gpt-5", 99, 80.0),  # out-of-scope seed
    ]
    cov = mod._coverage_summary(
        results,
        configured_models=["gpt-5"],
        configured_seeds=[42],
        n_scenarios=1,
    )
    assert cov["per_model_realized"]["gpt-5"] == 1


def test_coverage_summary_empty_configured_models_emits_explicit_warning() -> None:
    """When ``meta["models"]`` is empty (corrupt run_config.json), the
    contract must mark the batch as not-comparable rather than reporting
    a misleading 'complete and uncomparable' state."""
    cov = mod._coverage_summary(
        [_row("fam/a", "gpt-5", 42, 80.0)],
        configured_models=[],
        configured_seeds=[42],
        n_scenarios=1,
    )
    assert cov["is_partial_batch"] is True
    assert cov["comparable_intersection_size"] == 0
    assert cov["comparability_warning"]
    assert "cannot be assessed" in cov["comparability_warning"]


def test_coverage_summary_no_common_pairs_warning_does_not_advertise_intersection() -> (
    None
):
    """When the intersection is empty, the warning must NOT tell consumers
    to "use intersection_leaderboard (n_pairs=0)" — that's misleading."""
    cov = mod._coverage_summary(
        [
            _row("fam/x", "gpt-5", 42, 80.0),
            _row("fam/y", "o3", 42, 60.0),
        ],
        configured_models=["gpt-5", "o3"],
        configured_seeds=[42],
        n_scenarios=2,
    )
    assert cov["comparable_intersection_size"] == 0
    assert cov["is_partial_batch"] is True
    msg = cov["comparability_warning"] or ""
    assert "Rerun the missing cells" in msg
    assert "intersection_leaderboard (n_pairs=0)" not in msg


def test_write_leaderboard_json_handles_legacy_callers_without_coverage(
    tmp_path: Path,
) -> None:
    """``coverage`` and ``intersection_leaderboard`` are optional kwargs;
    legacy callers must not see crashes or new fields."""
    out = tmp_path / "legacy"
    out.mkdir()
    rows = [
        _row("fam/a", "gpt-5", 42, 80.0),
        _row("fam/a", "o3", 42, 70.0),
    ]
    leaderboard = mod._write_leaderboard_json(out, rows)
    assert leaderboard
    payload = json.loads((out / "leaderboard.json").read_text(encoding="utf-8"))
    assert "coverage" not in payload
    assert "intersection_leaderboard" not in payload
    assert payload["n_episodes_total"] == 2


def test_manifest_top_level_and_nested_coverage_fields_agree(tmp_path: Path) -> None:
    """Guard against drift between top-level manifest fields and the
    nested ``coverage`` block."""
    # We re-use the helper directly so this test stays focused & fast
    # (no main() pipeline dependency).
    cov = mod._coverage_summary(
        [_row("fam/a", "gpt-5", 42, 80.0), _row("fam/a", "o3", 42, 70.0)],
        configured_models=["gpt-5", "o3"],
        configured_seeds=[42],
        n_scenarios=1,
    )
    public = {k: v for k, v in cov.items() if not k.startswith("_")}
    # Mirror what _finalize_outputs writes at manifest top-level.
    assert public["expected_total"] == cov["expected_total"]
    assert public["comparable_intersection_size"] == cov["comparable_intersection_size"]
    assert public["is_partial_batch"] == cov["is_partial_batch"]
    assert public["per_model_coverage"] == cov["per_model_coverage"]
    assert "_comparable_pairs" not in public


def test_effective_episode_rows_drop_in_flight_and_keep_latest_terminal() -> None:
    rows = [
        {
            "status": "in_flight",
            "scenario_slug": "fam/a",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
        },
        _row("fam/a", "gpt-5", 42, 10.0, status="error"),
        {
            "status": "in_flight",
            "scenario_slug": "fam/a",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
        },
        _row("fam/a", "gpt-5", 42, 80.0),
        _row("fam/b", "o3", 42, 60.0),
    ]
    effective = mod._effective_episode_rows(rows)
    assert len(effective) == 2
    by_key = {(r["scenario_slug"], r["model"], r["seed"]): r for r in effective}
    assert by_key[("fam/a", "gpt-5", 42)]["status"] == "ok"
    assert by_key[("fam/a", "gpt-5", 42)]["score"]["total_score"] == 80.0
    assert by_key[("fam/b", "o3", 42)]["status"] == "ok"


def test_compact_episode_rows_keeps_latest_in_flight_when_no_terminal_exists() -> None:
    rows = [
        {
            "status": "in_flight",
            "scenario_slug": "fam/a",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
        },
        {
            "status": "in_flight",
            "scenario_slug": "fam/a",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "started_at": "later",
        },
    ]
    compact = mod._compact_episode_rows(rows)
    assert len(compact) == 1
    assert compact[0]["status"] == "in_flight"
    assert compact[0]["started_at"] == "later"


def test_compact_episode_rows_keeps_distinct_pass_ids() -> None:
    rows = [
        {
            "status": "in_flight",
            "scenario_slug": "fam/a",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "pass_id": "pass-0",
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
        {
            **_row("fam/a", "gpt-5", 42, 80.0),
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "pass_id": "pass-0",
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
        {
            **_row("fam/a", "gpt-5", 42, 81.0),
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "pass_id": "pass-1",
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
    ]

    compact = mod._compact_episode_rows(rows)

    assert len(compact) == 2
    assert {row["pass_id"] for row in compact} == {"pass-0", "pass-1"}
    by_pass = {row["pass_id"]: row for row in compact}
    assert by_pass["pass-0"]["score"]["total_score"] == 80.0
    assert by_pass["pass-1"]["score"]["total_score"] == 81.0


def test_prefer_current_implementation_rows_drops_stale_fingerprint() -> None:
    stale = {
        **_row("fam/a", "gpt-5", 42, 99.0),
        "pass_id": "pass-0",
        "evaluation_implementation_fingerprint": "protocol-old",
    }
    fresh = {
        **_row("fam/a", "gpt-5", 42, 10.0),
        "pass_id": "pass-0",
        "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
    }
    chosen = mod._prefer_current_implementation_rows([stale, fresh])
    assert len(chosen) == 1
    assert chosen[0]["score"]["total_score"] == 10.0


def test_effective_episode_rows_for_analysis_does_not_double_count_fingerprint_bump() -> (
    None
):
    rows = [
        {
            **_row("fam/a", "gpt-5", 42, 40.0),
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "pass_id": "pass-0",
            "evaluation_implementation_fingerprint": "protocol-old",
        },
        {
            **_row("fam/a", "gpt-5", 42, 12.0),
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "pass_id": "pass-0",
            "evaluation_implementation_fingerprint": mod.EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
    ]
    compact = mod._compact_episode_rows(rows)
    assert len(compact) == 2
    analysis = mod.effective_episode_rows_for_analysis(rows)
    assert len(analysis) == 1
    assert analysis[0]["score"]["total_score"] == 12.0


def test_retry_cells_allowlist_parses_log_audit_paths() -> None:
    jobs = [
        _job(
            "power_grid/storm_emergency_6h/deep_planning/high/st_chron0_deep_planning_high_s45",
            "o3-2025-04-16",
        ),
        _job(
            "power_grid/storm_emergency_6h/time_pressure/basic/st_chron4_time_pressure_basic_s45",
            "gpt-5.2-2025-12-11",
        ),
        _job(
            "power_grid/storm_emergency_6h/time_pressure/medium/st_chron0_time_pressure_medium_s42",
            "gpt-5-2025-08-07",
        ),
    ]
    audit_payload = {
        "log_files_orphan_interrupted": 2,
        "sample_orphan_interrupted_logs": [
            {
                "path": "logs/o3-2025-04-16/power_grid_storm_emergency_6h_deep_planning_high_st_chron0_deep_planning_high_s45_s42.log"
            },
            {
                "path": "logs/gpt-5.2-2025-12-11/power_grid_storm_emergency_6h_time_pressure_basic_st_chron4_time_pressure_basic_s45_s42.log"
            },
        ],
    }
    selected = mod._apply_retry_cells_allowlist(jobs, audit_payload)
    assert [(j["model"], j["scenario_slug"], j["seed"]) for j in selected] == [
        (
            "o3-2025-04-16",
            "power_grid/storm_emergency_6h/deep_planning/high/st_chron0_deep_planning_high_s45",
            42,
        ),
        (
            "gpt-5.2-2025-12-11",
            "power_grid/storm_emergency_6h/time_pressure/basic/st_chron4_time_pressure_basic_s45",
            42,
        ),
    ]


def test_retry_cells_allowlist_refuses_truncated_samples() -> None:
    jobs = [_job("power_grid/foo/bar", "gpt-5")]
    audit_payload = {
        "log_files_orphan_interrupted": 9,
        "sample_orphan_interrupted_logs": [
            {"path": "logs/gpt-5/power_grid_foo_bar_s42.log"}
        ],
    }
    with pytest.raises(ValueError, match="sample_orphan_interrupted_logs"):
        mod._apply_retry_cells_allowlist(jobs, audit_payload)


# ---------------------------------------------------------------------------
# _batch_state state machine — direct unit tests for all four legal states.
# ---------------------------------------------------------------------------


def _coverage(*, configured_models, is_partial=False, per_model=None):
    """Build the minimal coverage dict that ``_batch_state`` looks at."""
    return {
        "configured_models": list(configured_models),
        "is_partial_batch": bool(is_partial),
        "per_model_coverage": dict(per_model or {}),
    }


def test_batch_state_unknown_when_no_configured_models() -> None:
    state = mod._batch_state(
        coverage=_coverage(configured_models=[]),
        results=[_row("fam/a", "gpt-5", 42, 80.0)],
        log_audit_report={"log_files_orphan_interrupted": 0},
    )
    assert state["batch_state"] == mod.BATCH_STATE_UNKNOWN
    assert state["reasons"]
    assert "configured" in state["reasons"][0].lower()


def test_batch_state_partial_takes_precedence_over_degradation() -> None:
    """A partial batch must report ``partial`` even if it also has errors —
    a partial grid is structurally not yet comparable, so the degradation
    discussion is moot until the grid is full."""
    state = mod._batch_state(
        coverage=_coverage(
            configured_models=["gpt-5", "o3"],
            is_partial=True,
            per_model={"gpt-5": 0.5, "o3": 1.0},
        ),
        results=[_row("fam/a", "gpt-5", 42, 80.0, status="error")],
        log_audit_report={"log_files_orphan_interrupted": 3},
    )
    assert state["batch_state"] == mod.BATCH_STATE_PARTIAL
    assert any("partial" in r.lower() for r in state["reasons"])


def test_batch_state_degraded_when_full_coverage_but_errors_or_interrupted() -> None:
    state_err = mod._batch_state(
        coverage=_coverage(
            configured_models=["gpt-5"], is_partial=False, per_model={"gpt-5": 1.0}
        ),
        results=[_row("fam/a", "gpt-5", 42, 80.0, status="error")],
        log_audit_report={"log_files_orphan_interrupted": 0},
    )
    assert state_err["batch_state"] == mod.BATCH_STATE_DEGRADED
    assert state_err["n_episodes_error"] == 1

    state_orphan = mod._batch_state(
        coverage=_coverage(
            configured_models=["gpt-5"], is_partial=False, per_model={"gpt-5": 1.0}
        ),
        results=[_row("fam/a", "gpt-5", 42, 80.0)],
        log_audit_report={"log_files_orphan_interrupted": 2},
    )
    assert state_orphan["batch_state"] == mod.BATCH_STATE_DEGRADED
    assert state_orphan["n_orphan_interrupted_logs"] == 2


def test_batch_state_degraded_when_provider_failure_is_backed_by_episode() -> None:
    state = mod._batch_state(
        coverage=_coverage(
            configured_models=["hy3-ioa"],
            is_partial=False,
            per_model={"hy3-ioa": 1.0},
        ),
        results=[
            {
                **_row("fam/a", "hy3-ioa", 42, 80.0),
                "trajectory_summary": {
                    "llm": {"llm_calls_ok": 99, "llm_calls_failed": 1}
                },
            }
        ],
        log_audit_report={
            "log_files_orphan_interrupted": 0,
            "episodes_with_llm_failures": {"hy3-ioa": 1},
        },
    )

    assert state["batch_state"] == mod.BATCH_STATE_DEGRADED
    assert state["n_provider_contaminated_episodes"] == 1
    assert any("provider" in reason for reason in state["reasons"])


def test_batch_state_reports_but_does_not_filter_high_fallback_wait_ratio() -> None:
    state = mod._batch_state(
        coverage=_coverage(
            configured_models=["gpt-5", "o3"],
            is_partial=False,
            per_model={"gpt-5": 1.0, "o3": 1.0},
        ),
        results=[
            {
                **_row("fam/a", "gpt-5", 42, 80.0),
                "trajectory_summary": {"llm": {"fallback_wait_ratio": 0.9}},
            },
            {
                **_row("fam/a", "o3", 42, 70.0),
                "trajectory_summary": {"llm": {"fallback_wait_ratio": 0.0}},
            },
        ],
        log_audit_report={"log_files_orphan_interrupted": 0},
    )
    assert state["batch_state"] == mod.BATCH_STATE_FINAL
    assert state["n_high_fallback_wait_episodes"] == 1


def test_batch_state_does_not_create_survivorship_bias_for_one_fallback_row() -> None:
    results = [_row("fam/a", "gpt-5", seed, 80.0) for seed in range(100)]
    results[0]["trajectory_summary"] = {"llm": {"fallback_wait_ratio": 0.9}}
    state = mod._batch_state(
        coverage=_coverage(
            configured_models=["gpt-5"],
            is_partial=False,
            per_model={"gpt-5": 1.0},
        ),
        results=results,
        log_audit_report={"log_files_orphan_interrupted": 0},
    )
    assert state["batch_state"] == mod.BATCH_STATE_FINAL
    assert state["n_high_fallback_wait_episodes"] == 1


def test_batch_state_partial_reports_fallback_without_retrying_model_failure() -> None:
    state = mod._batch_state(
        coverage=_coverage(
            configured_models=["gpt-5"],
            is_partial=True,
            per_model={"gpt-5": 0.4},
        ),
        results=[
            {
                **_row("fam/a", "gpt-5", 42, 80.0),
                "trajectory_summary": {"llm": {"fallback_wait_ratio": 0.75}},
            }
        ],
        log_audit_report={"log_files_orphan_interrupted": 0},
    )
    assert state["batch_state"] == mod.BATCH_STATE_PARTIAL
    assert state["n_high_fallback_wait_episodes"] == 1
    assert not any("high_fallback_wait_ratio" in r for r in state["reasons"])


def test_batch_state_final_for_clean_full_batch() -> None:
    state = mod._batch_state(
        coverage=_coverage(
            configured_models=["gpt-5", "o3"],
            is_partial=False,
            per_model={"gpt-5": 1.0, "o3": 1.0},
        ),
        results=[
            _row("fam/a", "gpt-5", 42, 80.0),
            _row("fam/a", "o3", 42, 70.0),
        ],
        log_audit_report={"log_files_orphan_interrupted": 0},
    )
    assert state["batch_state"] == mod.BATCH_STATE_FINAL
    assert state["n_episodes_error"] == 0
    assert state["n_orphan_interrupted_logs"] == 0


def test_batch_state_legal_states_constant_is_complete() -> None:
    assert set(mod.LEGAL_BATCH_STATES) == {
        mod.BATCH_STATE_FINAL,
        mod.BATCH_STATE_PARTIAL,
        mod.BATCH_STATE_DEGRADED,
        mod.BATCH_STATE_UNKNOWN,
    }


def test_batch_state_partial_lists_collateral_error_and_orphan_reasons() -> None:
    """Reviewer follow-up: partial dominates the state, but operators still
    need to see that the partial grid ALSO contains errors / interrupted
    orphans so they can fix them in the same rerun."""
    state = mod._batch_state(
        coverage=_coverage(
            configured_models=["gpt-5"],
            is_partial=True,
            per_model={"gpt-5": 0.4},
        ),
        results=[_row("fam/a", "gpt-5", 42, 80.0, status="error")],
        log_audit_report={"log_files_orphan_interrupted": 7},
    )
    assert state["batch_state"] == mod.BATCH_STATE_PARTIAL
    joined = " | ".join(state["reasons"])
    assert "partial batch" in joined
    assert "1 episode rows" in joined
    assert "7 orphan log file(s)" in joined
    assert state["n_episodes_error"] == 1
    assert state["n_orphan_interrupted_logs"] == 7


def test_batch_state_handles_none_coverage_gracefully() -> None:
    """``coverage`` is now declared Optional; passing None must not crash."""
    state = mod._batch_state(
        coverage=None,
        results=[],
        log_audit_report=None,
    )
    assert state["batch_state"] == mod.BATCH_STATE_UNKNOWN
    assert state["n_episodes_error"] == 0
    assert state["n_orphan_interrupted_logs"] == 0


def test_finalize_demotes_to_degraded_when_audit_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If `audit_logs` blows up (e.g. permission error mid-run), finalize
    must still emit leaderboard.json + RUN_MANIFEST.json with at least
    ``batch_state == degraded`` and the audit error captured in reasons."""
    out_dir = tmp_path / "audit_fail"
    out_dir.mkdir()

    rows = [
        _row("fam/a", "gpt-5", 42, 80.0),
        _row("fam/a", "o3", 42, 70.0),
    ]
    treatment_hashes = {"gpt-5": "1" * 64, "o3": "2" * 64}
    for row in rows:
        row["interaction_mode"] = "logical_stateless"
        row["agent_treatment_sha256"] = treatment_hashes[row["model"]]
    # `analyze_output_dir` (downstream of finalize) requires episodes.jsonl
    # to exist. Mirror what main() would have written before finalize.
    (out_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    meta = {
        "models": ["gpt-5", "o3"],
        "seeds": [42],
        "n_scenarios": 1,
        "interaction_mode": "logical_stateless",
        "agent_treatment_sha256_by_model": treatment_hashes,
    }

    def boom(_path):
        raise OSError("synthetic audit failure")

    monkeypatch.setattr(mod, "audit_logs", boom)

    mod._finalize_outputs(out_dir, rows, meta)

    manifest = json.loads((out_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    # Was a clean grid; audit failure must demote to degraded.
    assert manifest["batch_state"] == mod.BATCH_STATE_DEGRADED
    assert any("log audit failed" in r for r in manifest["batch_state_reasons"])
    # Leaderboard / analysis must still be produced.
    assert (out_dir / "leaderboard.json").exists()
    assert (out_dir / "ANALYSIS.md").exists()


def test_formal_refinalize_fail_closes_old_leaderboard_before_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out_dir = tmp_path / "formal_refinalize_failure"
    out_dir.mkdir()
    (out_dir / "RUN_MANIFEST.json").write_text(
        json.dumps({"formal_run": True, "leaderboard_eligible": True}),
        encoding="utf-8",
    )
    (out_dir / "leaderboard.json").write_text(
        json.dumps({"leaderboard_eligible": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mod,
        "_coverage_summary",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(RuntimeError, match="boom"):
        mod._finalize_outputs(out_dir, [], {"formal_run": True})

    manifest = json.loads((out_dir / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    leaderboard = json.loads((out_dir / "leaderboard.json").read_text(encoding="utf-8"))
    assert manifest["leaderboard_eligible"] is False
    assert leaderboard["leaderboard_eligible"] is False


def test_finalize_outputs_ignore_in_flight_placeholders_in_manifest_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "inflight_finalize"
    out_dir.mkdir()
    raw_rows = [
        {
            "status": "in_flight",
            "scenario_slug": "fam/a",
            "model": "gpt-5",
            "seed": 42,
            "scenario_signature": "sig-a",
            "temperature": 1.0,
            "episode_log_path": str(out_dir / "logs" / "gpt-5" / "fam_a_s42.log"),
        },
        _row("fam/a", "gpt-5", 42, 80.0),
        _row("fam/a", "o3", 42, 70.0),
    ]
    treatment_hashes = {"gpt-5": "1" * 64, "o3": "2" * 64}
    for row in raw_rows:
        row["interaction_mode"] = "logical_stateless"
        row["agent_treatment_sha256"] = treatment_hashes[row["model"]]
    (out_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(r) for r in raw_rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        mod,
        "audit_logs",
        lambda _path: {
            "log_files_orphan_interrupted": 0,
            "log_files_orphan_empty": 0,
            "orphan_in_flight_rows": 0,
        },
    )
    monkeypatch.setattr(mod, "write_log_audit_markdown", lambda *_a, **_k: None)
    manifest = mod._finalize_outputs(
        out_dir,
        mod._effective_episode_rows(raw_rows),
        {
            "models": ["gpt-5", "o3"],
            "seeds": [42],
            "n_scenarios": 1,
            "interaction_mode": "logical_stateless",
            "agent_treatment_sha256_by_model": treatment_hashes,
        },
    )
    assert manifest["n_episodes_total"] == 2
    assert manifest["n_episodes_ok"] == 2


# ─────────────────────────────────────────────────────────────────────────────
# --prompt-mode CLI flag: LLMConfig previously only ever got prompt_mode via
# its dataclass default ("strict"). A batch run had no way to request
# "debug" mode (or to record which mode a formal run actually used) without
# hand-editing LLMConfig in code. This wires argparse -> LLMConfig and
# records the choice in run_config.json for audit traceability.
# ─────────────────────────────────────────────────────────────────────────────


def test_current_defaults_use_strict_persistent_treatment(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "default_mode"
    scenario = "power_grid/foo/time_pressure/basic/foo_s42"

    monkeypatch.setattr(
        mod,
        "_load_zhsrc_exports",
        lambda: {"OPENAI_API_KEY": "test-key"},
    )
    monkeypatch.setattr(mod, "_expand_scenarios", lambda patterns: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda slug: {
            "seed_id": "foo_s42",
            "family": "foo",
            "backend_kind": "pglib_uc_synthetic",
            "scenario_signature": "sig-foo",
        },
    )
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {"git_commit": "abc", "git_dirty": False, "git_status_short": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(output_dir),
            "--scenario-slice",
            "custom",
            "--scenarios",
            scenario,
            "--models",
            "stealth/ox-alpha",
            "--dry-run",
        ],
    )

    rc = mod.main()
    assert rc == 0
    run_cfg = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_cfg["prompt_mode"] == "strict"
    assert run_cfg["interaction_mode"] == "logical_persistent"
    assert run_cfg["max_tokens"] == 8192
    assert run_cfg["model_context_window_tokens_by_model"] == {
        "stealth/ox-alpha": 1_048_576
    }
    assert run_cfg["tool_choice_supported_by_model"] == {"stealth/ox-alpha": True}
    assert run_cfg["wakeup_policy"] == {
        "session_start": True,
        "typed_actionable_events": True,
        "agent_scheduled_reviews": True,
        "harness_periodic_supervisory_scan": False,
        "unknown_events_actionable": False,
    }


def test_no_resume_refuses_existing_run_config_or_episode_journal(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "formal"
    out_dir.mkdir()
    episodes_path = out_dir / "episodes.jsonl"

    assert not mod._no_resume_output_conflict(
        resume=False,
        finalize_only=False,
        existing_run_config=None,
        episodes_path=episodes_path,
    )
    assert mod._no_resume_output_conflict(
        resume=False,
        finalize_only=False,
        existing_run_config={"schema_version": "fixture"},
        episodes_path=episodes_path,
    )
    episodes_path.write_text("{}\n", encoding="utf-8")
    assert mod._no_resume_output_conflict(
        resume=False,
        finalize_only=False,
        existing_run_config=None,
        episodes_path=episodes_path,
    )
    assert not mod._no_resume_output_conflict(
        resume=True,
        finalize_only=False,
        existing_run_config={"schema_version": "fixture"},
        episodes_path=episodes_path,
    )
    assert not mod._no_resume_output_conflict(
        resume=False,
        finalize_only=True,
        existing_run_config={"schema_version": "fixture"},
        episodes_path=episodes_path,
    )


def test_batch_llm_config_aligns_persistent_agentic_defaults() -> None:
    args = argparse.Namespace(
        api_key_env="OPENROUTER_API_KEY",
        api_mode="chat_completions",
        interaction_mode="logical_persistent",
        stream_chat_completions=False,
        prompt_mode="strict",
    )

    cfg = mod._batch_llm_config(
        model="stealth/ox-alpha",
        temperature=0.0,
        args=args,
        base_url="https://openrouter.ai/api/v1",
        api_version=None,
        responses_base_url=None,
    )

    assert cfg.max_tokens == 8192
    assert cfg.timeout_s == 150.0
    assert cfg.persistent_history_max_messages == 32
    assert cfg.persistent_context_max_chars == 48_000
    assert cfg.persistent_memory_max_items == 64
    assert cfg.tool_choice == "auto"
    assert cfg.reasoning_effort is None
    assert cfg.protocol_repair_max_tokens == 4096
    assert cfg.stream_chat_completions is False
    assert cfg.model_context_window_tokens == 1_048_576
    assert cfg.model_max_output_tokens == 131_072
    assert cfg.max_consecutive_provider_failures == 5


def test_formal_batch_llm_config_fails_closed_on_first_provider_error() -> None:
    args = argparse.Namespace(
        api_key_env="OPENROUTER_API_KEY",
        api_mode="chat_completions",
        interaction_mode="logical_persistent",
        stream_chat_completions=True,
        prompt_mode="strict",
        formal_run=True,
    )

    cfg = mod._batch_llm_config(
        model="z-ai/glm-5.2:free",
        temperature=0.0,
        args=args,
        base_url="https://openrouter.ai/api/v1",
        api_version=None,
        responses_base_url=None,
    )

    assert cfg.max_consecutive_provider_failures == 1
    assert cfg.provider_failure_policy == "abort"


def test_formal_provider_failure_profile_is_shared_by_startup_validation() -> None:
    profile = mod._provider_failure_profile(formal_run=True)
    readiness = _agentic_formal_readiness()
    readiness["formal_run_contract"]["agentic_profile"] = dict(profile)

    reasons = mod._validate_protocol21_formal_run(
        _agentic_formal_config(**profile),
        readiness,
        suite_manifest_sha256="suite",
    )

    assert profile == {
        "max_consecutive_provider_failures": 1,
        "provider_failure_policy": "abort",
    }
    assert reasons == []


def test_formal_main_startup_validation_receives_fail_closed_provider_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def capture_formal_config(
        config: dict[str, Any],
        _readiness: dict[str, Any],
        **_kwargs: Any,
    ) -> list[str]:
        captured.update(config)
        return ["stop_after_startup_validation"]

    monkeypatch.setattr(mod, "DYNAMIC_SCENARIO_SLICES", {})
    monkeypatch.setattr(
        mod,
        "resolve_formal_manifest_slice",
        lambda _path: {
            "slice_name": "manifest_fixture",
            "dynamic_slice_spec": ("fixture", "readiness.json", {}),
        },
    )
    monkeypatch.setattr(mod, "_resolve_patterns", lambda *_args: ["fixture"])
    monkeypatch.setattr(mod, "_expand_scenarios", lambda _patterns: ["fixture"])
    monkeypatch.setattr(mod, "load_scenario_yaml", lambda _slug: {"seed": 42})
    monkeypatch.setattr(mod, "_bind_scenario_contracts_for_slice", lambda *_args: None)
    monkeypatch.setattr(mod, "_suite_manifest_sha256_for_slice", lambda *_args: "suite")
    monkeypatch.setattr(mod, "_suite_eligibility_binding", lambda _slice: {})
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {
            "git_commit": "commit",
            "git_metadata_available": True,
            "git_dirty": False,
            "git_status_short": [],
        },
    )
    monkeypatch.setattr(mod, "_validate_protocol21_formal_run", capture_formal_config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(tmp_path / "formal"),
            "--formal-run",
            "--formal-manifest",
            str(tmp_path / "manifest.json"),
            "--models",
            "model-a",
            "--dry-run",
        ],
    )

    assert mod.main() == 1
    assert captured["max_consecutive_provider_failures"] == 1
    assert captured["provider_failure_policy"] == "abort"


@pytest.mark.parametrize(
    ("ambient_name", "ambient_value", "message"),
    [
        (
            "OPERATE_API_VERSION",
            "2026-08-01",
            "API version is unsupported for non-Azure providers",
        ),
        (
            "OPERATE_RESPONSES_API_BASE_URL",
            "https://azure.example.test/responses",
            "responses base URL is unsupported for non-Azure providers",
        ),
    ],
)
def test_formal_logical_chat_rejects_ambient_nonazure_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    ambient_name: str,
    ambient_value: str,
    message: str,
) -> None:
    scenario = "logistics/formal-transport-fixture"
    monkeypatch.delenv("OPERATE_API_VERSION", raising=False)
    monkeypatch.delenv("OPERATE_RESPONSES_API_BASE_URL", raising=False)
    monkeypatch.setenv(ambient_name, ambient_value)
    monkeypatch.setenv("FORMAL_TEST_BASE_URL", "https://provider.example.test/v1")
    monkeypatch.setattr(mod, "DYNAMIC_SCENARIO_SLICES", {})
    monkeypatch.setattr(mod, "_load_zhsrc_exports", lambda: {})
    monkeypatch.setattr(mod, "_load_named_zshrc_export", lambda _name: None)
    monkeypatch.setattr(
        mod,
        "resolve_formal_manifest_slice",
        lambda _path: {
            "slice_name": "manifest_fixture",
            "dynamic_slice_spec": ("fixture", "readiness.json", {}),
        },
    )
    monkeypatch.setattr(mod, "_resolve_patterns", lambda *_args: [scenario])
    monkeypatch.setattr(mod, "_expand_scenarios", lambda _patterns: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda _slug: {
            "seed": 42,
            "scenario_signature": "fixture-signature",
            "backend_kind": "jsplib_job_shop",
        },
    )
    monkeypatch.setattr(mod, "_bind_scenario_contracts_for_slice", lambda *_: None)
    monkeypatch.setattr(mod, "_suite_manifest_sha256_for_slice", lambda *_: "suite")
    monkeypatch.setattr(mod, "_suite_eligibility_binding", lambda _slice: {})
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {
            "git_commit": "commit",
            "git_metadata_available": True,
            "git_dirty": False,
            "git_status_short": [],
        },
    )
    monkeypatch.setattr(mod, "_validate_protocol21_formal_run", lambda *_a, **_k: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(tmp_path / ambient_name.lower()),
            "--formal-run",
            "--formal-manifest",
            str(tmp_path / "manifest.json"),
            "--models",
            "z-ai/glm-5.2:free",
            "--api-mode",
            "chat_completions",
            "--base-url-env",
            "FORMAL_TEST_BASE_URL",
            "--model-context-window-tokens",
            "256000",
            "--model-max-output-tokens",
            "230400",
            "--dry-run",
        ],
    )

    assert mod.main() == 1
    assert message in capsys.readouterr().err


def test_batch_llm_config_snapshots_inkling_tool_choice_capability() -> None:
    args = argparse.Namespace(
        api_key_env="OPENROUTER_API_KEY",
        api_mode="chat_completions",
        interaction_mode="logical_persistent",
        stream_chat_completions=True,
        prompt_mode="strict",
        model_context_window_tokens=1_048_576,
        model_max_output_tokens=262_144,
    )

    cfg = mod._batch_llm_config(
        model="thinkingmachines/inkling:free",
        temperature=0.0,
        args=args,
        base_url="https://openrouter.ai/api/v1",
        api_version=None,
        responses_base_url=None,
    )

    assert cfg.tool_choice == "auto"
    assert cfg.tool_choice_supported is False


def test_tool_choice_capability_snapshot_is_treatment_bound() -> None:
    supported = mod.LLMConfig(
        provider="openai_compatible",
        model="provider-test-model",
        tool_choice="auto",
        tool_choice_supported=True,
    )
    unsupported = mod.LLMConfig(
        provider="openai_compatible",
        model="provider-test-model",
        tool_choice="auto",
        tool_choice_supported=False,
    )

    assert mod._agent_treatment_sha256(supported) != mod._agent_treatment_sha256(
        unsupported
    )


def test_batch_llm_config_binds_persistent_working_context_overrides() -> None:
    base_args = {
        "api_key_env": "OPENROUTER_API_KEY",
        "api_mode": "chat_completions",
        "interaction_mode": "logical_persistent",
        "stream_chat_completions": False,
        "prompt_mode": "strict",
    }
    default = mod._batch_llm_config(
        model="stealth/ox-alpha",
        temperature=0.0,
        args=argparse.Namespace(**base_args),
        base_url="https://openrouter.ai/api/v1",
        api_version=None,
        responses_base_url=None,
    )
    overridden = mod._batch_llm_config(
        model="stealth/ox-alpha",
        temperature=0.0,
        args=argparse.Namespace(
            **base_args,
            persistent_history_max_messages=40,
            persistent_context_max_chars=64_000,
            persistent_memory_max_items=80,
        ),
        base_url="https://openrouter.ai/api/v1",
        api_version=None,
        responses_base_url=None,
    )

    assert overridden.persistent_history_max_messages == 40
    assert overridden.persistent_context_max_chars == 64_000
    assert overridden.persistent_memory_max_items == 80
    assert mod._agent_treatment_sha256(overridden) != mod._agent_treatment_sha256(
        default
    )


def test_batch_llm_config_preserves_formal_stateless_defaults() -> None:
    args = argparse.Namespace(
        api_key_env="OPENAI_API_KEY",
        api_mode="chat_completions",
        interaction_mode="logical_stateless",
        stream_chat_completions=False,
        prompt_mode="strict",
    )

    cfg = mod._batch_llm_config(
        model="gpt-test",
        temperature=1.0,
        args=args,
        base_url=None,
        api_version=None,
        responses_base_url=None,
    )

    assert cfg.max_tokens == 4096
    assert cfg.timeout_s == 60.0
    assert cfg.persistent_history_max_messages == 24
    assert cfg.persistent_context_max_chars == 16_000
    assert cfg.persistent_memory_max_items == 32
    assert cfg.tool_choice == "auto"
    assert cfg.reasoning_effort is None
    assert cfg.protocol_repair_max_tokens == 512


def test_stateless_ox_does_not_silently_bind_frozen_model_capabilities() -> None:
    args = argparse.Namespace(
        api_key_env="OPENROUTER_API_KEY",
        api_mode="chat_completions",
        interaction_mode="logical_stateless",
        stream_chat_completions=False,
        prompt_mode="strict",
    )

    cfg = mod._batch_llm_config(
        model="stealth/ox-alpha",
        temperature=1.0,
        args=args,
        base_url="https://openrouter.ai/api/v1",
        api_version=None,
        responses_base_url=None,
    )

    assert cfg.model_context_window_tokens is None
    assert cfg.model_max_output_tokens is None


def test_explicit_stateless_model_capabilities_form_a_diagnostic_treatment() -> None:
    base_args = {
        "api_key_env": "OPENROUTER_API_KEY",
        "api_mode": "chat_completions",
        "interaction_mode": "logical_stateless",
        "stream_chat_completions": False,
        "prompt_mode": "strict",
    }
    unbound = mod._batch_llm_config(
        model="stealth/ox-alpha",
        temperature=1.0,
        args=argparse.Namespace(**base_args),
        base_url="https://openrouter.ai/api/v1",
        api_version=None,
        responses_base_url=None,
    )
    bound = mod._batch_llm_config(
        model="stealth/ox-alpha",
        temperature=1.0,
        args=argparse.Namespace(
            **base_args,
            model_context_window_tokens=128_000,
            model_max_output_tokens=16_384,
        ),
        base_url="https://openrouter.ai/api/v1",
        api_version=None,
        responses_base_url=None,
    )

    assert bound.model_context_window_tokens == 128_000
    assert bound.model_max_output_tokens == 16_384
    assert mod._agent_treatment_sha256(bound) != mod._agent_treatment_sha256(unbound)


@pytest.mark.parametrize(
    ("context_window", "max_output", "decision_reserve", "repair_reserve", "message"),
    [
        (128, 256, 64, 32, "cannot exceed"),
        (1_024, 128, 256, 32, "decision output reserve"),
        (1_024, 128, 64, 256, "repair output reserve"),
    ],
)
def test_batch_model_capability_preflight_fails_closed(
    context_window: int,
    max_output: int,
    decision_reserve: int,
    repair_reserve: int,
    message: str,
) -> None:
    assert message in mod._model_capability_preflight_error(
        model="custom/model",
        context_window=context_window,
        max_output=max_output,
        decision_reserve=decision_reserve,
        repair_reserve=repair_reserve,
    )


def test_native_runtime_binding_fails_closed_without_real_sumo_gate(
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPERATE_TRAFFIC_BACKEND_REAL", raising=False)
    monkeypatch.delenv("OPERATE_TRAFFIC_FORCE_TRANSPORT", raising=False)

    binding = mod._native_runtime_binding(
        {"traffic/case": {"backend_kind": "sumo"}},
        ["traffic/case"],
        max_workers=8,
        scheduler_mode="global",
    )

    assert binding["ok"] is False
    assert binding["blockers"] == ["real_sumo_gate_missing"]


def test_native_runtime_binding_rejects_parallel_libsumo(monkeypatch) -> None:
    monkeypatch.setenv("OPERATE_TRAFFIC_BACKEND_REAL", "1")
    monkeypatch.delenv("OPERATE_TRAFFIC_FORCE_TRANSPORT", raising=False)
    monkeypatch.setattr(mod, "probe_sumo_transport", lambda: "libsumo")

    binding = mod._native_runtime_binding(
        {"traffic/case": {"backend_kind": "sumo"}},
        ["traffic/case"],
        max_workers=8,
        scheduler_mode="global",
    )

    assert binding["ok"] is False
    assert binding["blockers"] == ["parallel_libsumo_unsupported"]


def test_native_runtime_binding_accepts_parallel_traci(monkeypatch) -> None:
    monkeypatch.setenv("OPERATE_TRAFFIC_BACKEND_REAL", "1")
    monkeypatch.setenv("OPERATE_TRAFFIC_FORCE_TRANSPORT", "traci")
    monkeypatch.setattr(mod, "probe_sumo_transport", lambda: "traci")

    binding = mod._native_runtime_binding(
        {"traffic/case": {"backend_kind": "sumo"}},
        ["traffic/case"],
        max_workers=8,
        scheduler_mode="global",
    )

    assert binding == {
        "ok": True,
        "requires_real_sumo": True,
        "traffic_real_enabled": True,
        "forced_transport": "traci",
        "resolved_transport": "traci",
        "blockers": [],
    }


def test_native_runtime_binding_ignores_sumo_env_for_unrelated_suites(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPERATE_TRAFFIC_BACKEND_REAL", "1")
    monkeypatch.setenv("OPERATE_TRAFFIC_FORCE_TRANSPORT", "traci")

    binding = mod._native_runtime_binding(
        {"logistics/case": {"backend_kind": "jsplib_job_shop"}},
        ["logistics/case"],
        max_workers=8,
        scheduler_mode="global",
    )

    assert binding == {
        "ok": True,
        "requires_real_sumo": False,
        "traffic_real_enabled": False,
        "forced_transport": None,
        "resolved_transport": None,
        "blockers": [],
    }


def test_persistent_batch_cli_uses_agentic_defaults(
    monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "persistent_defaults"
    scenario = "power_grid/foo/time_pressure/basic/foo_s42"

    monkeypatch.setattr(
        mod,
        "_load_zhsrc_exports",
        lambda: {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
        },
    )
    monkeypatch.setattr(mod, "_expand_scenarios", lambda patterns: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda slug: {
            "seed_id": "foo_s42",
            "family": "foo",
            "backend_kind": "pglib_uc_synthetic",
            "scenario_signature": "sig-foo",
        },
    )
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {"git_commit": "abc", "git_dirty": False, "git_status_short": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(output_dir),
            "--scenario-slice",
            "custom",
            "--scenarios",
            scenario,
            "--models",
            "stealth/ox-alpha",
            "--interaction-mode",
            "logical_persistent",
            "--dry-run",
        ],
    )

    assert mod.main() == 0
    run_cfg = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_cfg["temperature"] == 0.0
    assert run_cfg["max_tokens"] == 8192
    assert run_cfg["provider_timeout_s"] == 150.0
    assert run_cfg["persistent_history_max_messages"] == 32
    assert run_cfg["persistent_context_max_chars"] == 48_000
    assert run_cfg["persistent_memory_max_items"] == 64

    override_output_dir = tmp_path / "persistent_overrides"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(override_output_dir),
            "--scenario-slice",
            "custom",
            "--scenarios",
            scenario,
            "--models",
            "stealth/ox-alpha",
            "--interaction-mode",
            "logical_persistent",
            "--persistent-history-max-messages",
            "40",
            "--persistent-context-max-chars",
            "64000",
            "--persistent-memory-max-items",
            "80",
            "--dry-run",
        ],
    )

    assert mod.main() == 0
    override_cfg = json.loads(
        (override_output_dir / "run_config.json").read_text(encoding="utf-8")
    )
    assert override_cfg["persistent_history_max_messages"] == 40
    assert override_cfg["persistent_context_max_chars"] == 64_000
    assert override_cfg["persistent_memory_max_items"] == 80


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--persistent-history-max-messages", "3"),
        ("--persistent-context-max-chars", "499"),
        ("--persistent-memory-max-items", "3"),
    ],
)
def test_persistent_batch_cli_rejects_invalid_working_context_bounds(
    monkeypatch, tmp_path: Path, flag: str, value: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(tmp_path / "invalid-persistent-bound"),
            "--interaction-mode",
            "logical_persistent",
            flag,
            value,
            "--dry-run",
        ],
    )

    assert mod.main() == 1


def test_multi_model_batch_rejects_scalar_model_capabilities(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    scenario = "power_grid/foo/time_pressure/basic/foo_s42"
    monkeypatch.setattr(mod, "_expand_scenarios", lambda patterns: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda slug: {
            "seed_id": "foo_s42",
            "family": "foo",
            "backend_kind": "pglib_uc_synthetic",
            "scenario_signature": "sig-foo",
        },
    )
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {"git_commit": "abc", "git_dirty": False, "git_status_short": []},
    )
    monkeypatch.setattr(mod, "_load_zhsrc_exports", lambda: {})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(tmp_path / "scalar-multi-model"),
            "--scenario-slice",
            "custom",
            "--scenarios",
            scenario,
            "--models",
            "model-a,model-b",
            "--interaction-mode",
            "logical_persistent",
            "--model-context-window-tokens",
            "128000",
            "--model-max-output-tokens",
            "16384",
            "--dry-run",
        ],
    )

    assert mod.main() == 1
    assert "run each model separately" in capsys.readouterr().err


def test_prompt_mode_debug_flows_into_llm_config_and_run_config(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    output_dir = tmp_path / "debug_mode"
    scenario = "power_grid/foo/time_pressure/basic/foo_s42"

    monkeypatch.setattr(
        mod,
        "_load_zhsrc_exports",
        lambda: {"OPENAI_API_KEY": "test-key"},
    )
    monkeypatch.setattr(mod, "_expand_scenarios", lambda patterns: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda slug: {
            "seed_id": "foo_s42",
            "family": "foo",
            "backend_kind": "pglib_uc_synthetic",
            "scenario_signature": "sig-foo",
        },
    )
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {"git_commit": "abc", "git_dirty": False, "git_status_short": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(output_dir),
            "--scenario-slice",
            "custom",
            "--scenarios",
            scenario,
            "--models",
            "gpt-5-2025-08-07",
            "--interaction-mode",
            "logical_stateless",
            "--prompt-mode",
            "debug",
            "--dry-run",
        ],
    )

    rc = mod.main()
    assert rc == 0
    captured = capsys.readouterr()
    # The pre-scan warning (before run_config.json exists) must fire so an
    # operator sees it even on --dry-run.
    assert "leaderboard-eligible" in captured.err
    run_cfg = json.loads((output_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_cfg["prompt_mode"] == "debug"


def test_build_jobs_wires_prompt_mode_into_llm_config(tmp_path: Path) -> None:
    scenario_slug = "power_grid/foo/time_pressure/basic/foo_s42"
    args = argparse.Namespace(
        api_key_env="OPENAI_API_KEY",
        api_mode="auto",
        save_trajectories=False,
        pass_k=1,
        prompt_mode="debug",
    )

    jobs = mod._build_jobs(
        scenarios=[scenario_slug],
        scenario_bodies={
            scenario_slug: {"seed_id": "foo_s42", "scenario_signature": "sig-foo"},
        },
        models=["gpt-5"],
        seeds=[42],
        temperature=1.0,
        args=args,
        out_dir=tmp_path,
        base_url=None,
        api_version=None,
        responses_base_url=None,
    )

    assert len(jobs) == 1
    assert jobs[0]["llm_config"]["prompt_mode"] == "debug"


def test_build_jobs_defaults_prompt_mode_to_strict_when_args_lack_attribute(
    tmp_path: Path,
) -> None:
    """Callers (and older tests) that build an argparse.Namespace without a
    prompt_mode attribute must still get the benchmark-correct default,
    not an AttributeError."""
    scenario_slug = "power_grid/foo/time_pressure/basic/foo_s42"
    args = argparse.Namespace(
        api_key_env="OPENAI_API_KEY",
        api_mode="auto",
        save_trajectories=False,
        pass_k=1,
    )

    jobs = mod._build_jobs(
        scenarios=[scenario_slug],
        scenario_bodies={
            scenario_slug: {"seed_id": "foo_s42", "scenario_signature": "sig-foo"},
        },
        models=["gpt-5"],
        seeds=[42],
        temperature=1.0,
        args=args,
        out_dir=tmp_path,
        base_url=None,
        api_version=None,
        responses_base_url=None,
    )

    assert jobs[0]["llm_config"]["prompt_mode"] == "strict"


def test_build_jobs_uses_explicit_batch_output_budget(tmp_path: Path) -> None:
    scenario_slug = "power_grid/foo/time_pressure/basic/foo_s42"
    args = argparse.Namespace(
        api_key_env="OPENAI_API_KEY",
        api_mode="auto",
        save_trajectories=True,
        pass_k=1,
        prompt_mode="strict",
        max_tokens=4096,
    )

    jobs = mod._build_jobs(
        scenarios=[scenario_slug],
        scenario_bodies={
            scenario_slug: {"seed_id": "foo_s42", "scenario_signature": "sig-foo"},
        },
        models=["gpt-5"],
        seeds=[42],
        temperature=1.0,
        args=args,
        out_dir=tmp_path,
        base_url=None,
        api_version=None,
        responses_base_url=None,
    )

    assert jobs[0]["llm_config"]["max_tokens"] == 4096
    assert jobs[0]["evaluation_implementation_fingerprint"] == (
        mod.EVALUATION_IMPLEMENTATION_FINGERPRINT
    )
    assert (
        ":prompt-strict:interaction-logical_stateless:max-tokens-4096"
        in (jobs[0]["run_semantics_fingerprint"])
    )
    assert jobs[0]["run_semantics_fingerprint"].endswith(
        jobs[0]["agent_treatment_sha256"]
    )
    assert len(jobs[0]["agent_treatment_sha256"]) == 64
    treatment_component = f"treatment-{jobs[0]['agent_treatment_sha256']}"
    assert treatment_component in Path(jobs[0]["episode_log_path"]).parts
    assert treatment_component in Path(jobs[0]["trajectory_dir"]).parts


def test_resume_key_distinguishes_output_budget_semantics() -> None:
    base = {
        "scenario_slug": "power_grid/foo",
        "model": "gpt-5",
        "seed": 42,
        "scenario_signature": "sig-a",
        "temperature": 1.0,
        "evaluation_implementation_fingerprint": (
            mod.EVALUATION_IMPLEMENTATION_FINGERPRINT
        ),
    }
    job = {
        **base,
        "run_semantics_fingerprint": mod._run_semantics_fingerprint("strict", 4096),
    }
    prior = {
        **base,
        "status": "ok",
        "run_semantics_fingerprint": mod._run_semantics_fingerprint("strict", 1200),
    }

    assert mod._filter_pending_jobs([job], [prior]) == [job]


def test_analysis_cell_key_is_bound_to_agent_treatment() -> None:
    base = {
        "scenario_slug": "power_grid/foo",
        "model": "hy3-ioa",
        "seed": 42,
        "pass_id": "pass-0",
    }

    assert mod._analysis_cell_key(
        {**base, "agent_treatment_sha256": "stateless"}
    ) != mod._analysis_cell_key({**base, "agent_treatment_sha256": "persistent"})


def test_run_config_treatment_guard_blocks_a_b_but_allows_resume_a() -> None:
    treatment_a = {
        "models": ["hy3-ioa"],
        "interaction_mode": "logical_stateless",
        "agent_treatment_sha256_by_model": {"hy3-ioa": "a" * 64},
    }
    treatment_b = {
        "models": ["hy3-ioa"],
        "interaction_mode": "logical_persistent",
        "agent_treatment_sha256_by_model": {"hy3-ioa": "b" * 64},
    }

    assert mod._run_config_treatment_compatibility_reasons(
        treatment_a, treatment_b
    ) == ["output_dir_agent_treatment_mismatch"]
    assert (
        mod._run_config_treatment_compatibility_reasons(treatment_a, treatment_a) == []
    )


def test_run_config_treatment_guard_requires_wakeup_policy_metadata() -> None:
    config = {
        "models": ["hy3-ioa"],
        "interaction_mode": "logical_persistent",
        "agent_treatment_sha256_by_model": {"hy3-ioa": "a" * 64},
        "wakeup_policy": dict(mod.CANONICAL_WAKEUP_POLICY),
    }
    missing_policy = dict(config)
    missing_policy.pop("wakeup_policy")

    assert mod._run_config_treatment_compatibility_reasons(
        missing_policy, config
    ) == ["output_dir_immutable_run_scope_mismatch"]


def test_run_config_guard_rejects_changed_agent_profile_binding() -> None:
    config = {
        "models": ["hy3-ioa"],
        "interaction_mode": "logical_persistent",
        "agent_profile_sha256_by_model": {"hy3-ioa": "a" * 64},
        "agent_treatment_sha256_by_model": {"hy3-ioa": "b" * 64},
    }

    assert mod._run_config_treatment_compatibility_reasons(
        config,
        {
            **config,
            "agent_profile_sha256_by_model": {"hy3-ioa": "c" * 64},
        },
    ) == ["output_dir_immutable_run_scope_mismatch"]


def test_run_config_guard_rejects_changed_model_capability_binding() -> None:
    base = {
        "models": ["stealth/ox-alpha"],
        "interaction_mode": "logical_persistent",
        "agent_treatment_sha256_by_model": {"stealth/ox-alpha": "a" * 64},
        "model_context_window_tokens_by_model": {"stealth/ox-alpha": 1_048_576},
        "model_max_output_tokens_by_model": {"stealth/ox-alpha": 131_072},
        "tool_choice_supported_by_model": {"stealth/ox-alpha": True},
        "token_count_method": "utf8_bytes_upper_bound",
        "token_count_version": "1",
    }
    changed = {
        **base,
        "model_context_window_tokens_by_model": {"stealth/ox-alpha": 128_000},
    }

    assert mod._run_config_treatment_compatibility_reasons(base, changed) == [
        "output_dir_immutable_run_scope_mismatch"
    ]

    changed_tool_choice = {
        **base,
        "tool_choice_supported_by_model": {"stealth/ox-alpha": False},
    }
    assert mod._run_config_treatment_compatibility_reasons(
        base, changed_tool_choice
    ) == ["output_dir_immutable_run_scope_mismatch"]


def test_run_config_guard_rejects_changed_native_runtime_binding() -> None:
    base = {
        "models": ["stealth/ox-alpha"],
        "interaction_mode": "logical_persistent",
        "agent_treatment_sha256_by_model": {"stealth/ox-alpha": "a" * 64},
        "native_runtime_binding": {
            "requires_real_sumo": True,
            "resolved_transport": "traci",
        },
    }
    changed = {
        **base,
        "native_runtime_binding": {
            "requires_real_sumo": True,
            "resolved_transport": "libsumo",
        },
    }

    assert mod._run_config_treatment_compatibility_reasons(base, changed) == [
        "output_dir_immutable_run_scope_mismatch"
    ]


def test_run_config_guard_rejects_changed_persistent_working_context() -> None:
    base = {
        "models": ["stealth/ox-alpha"],
        "interaction_mode": "logical_persistent",
        "agent_treatment_sha256_by_model": {"stealth/ox-alpha": "a" * 64},
        "persistent_history_max_messages": 32,
        "persistent_context_max_chars": 48_000,
        "persistent_memory_max_items": 64,
    }

    for field, value in (
        ("persistent_history_max_messages", 40),
        ("persistent_context_max_chars", 64_000),
        ("persistent_memory_max_items", 80),
    ):
        assert mod._run_config_treatment_compatibility_reasons(
            base, {**base, field: value}
        ) == ["output_dir_immutable_run_scope_mismatch"]


def test_provider_rate_limits_are_treatment_and_resume_bound() -> None:
    base_config = mod.LLMConfig(
        model="free-model",
        interaction_mode="logical_stateless",
        provider_rpm_limit=20,
        provider_rpd_limit=1_000,
        provider_rate_limit_scope="openrouter-free-shared",
    )
    changed_scope = mod.LLMConfig(
        model="free-model",
        interaction_mode="logical_stateless",
        provider_rpm_limit=20,
        provider_rpd_limit=1_000,
        provider_rate_limit_scope="openrouter-ox",
    )
    assert mod._agent_treatment_sha256(base_config) != mod._agent_treatment_sha256(
        changed_scope
    )

    base_run = {
        "models": ["free-model"],
        "interaction_mode": "logical_persistent",
        "agent_treatment_sha256_by_model": {"free-model": "a" * 64},
        "provider_rpm_limit": 20,
        "provider_rpd_limit": 1_000,
        "provider_rate_limit_scope": "openrouter-free-shared",
    }
    assert mod._run_config_treatment_compatibility_reasons(
        base_run,
        {**base_run, "provider_rpd_limit": 50},
    ) == ["output_dir_immutable_run_scope_mismatch"]


def test_quota_sentinel_accepts_utc_rpd_reset(tmp_path: Path) -> None:
    sentinel = tmp_path / ".quota_exhausted_model"
    sentinel.write_text(
        json.dumps({"model": "model", "reset_at": "2099-09-01T00:00:00Z"}),
        encoding="utf-8",
    )
    assert mod._quota_sentinel_is_active(mod._quota_sentinel_payload(sentinel) or {})
    assert (
        mod._quota_reset_text(
            "ProviderQuotaExhaustedError: reset_at=2099-09-01T00:00:00Z"
        )
        == "2099-09-01T00:00:00Z"
    )


def test_quota_sentinel_without_provider_reset_reprobes_after_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
    monkeypatch.setattr(mod, "_quota_now_utc", lambda: now)
    job = _minimal_llm_job(tmp_path)

    sentinel = mod._write_quota_sentinel(
        job,
        {"error": "ProviderQuotaExhaustedError: quota exhausted"},
    )

    assert sentinel is not None
    payload = mod._quota_sentinel_payload(sentinel)
    assert payload is not None
    assert payload["reset_source"] == "bounded_reprobe"
    assert payload["reset_at"] == "2026-08-31T12:05:00Z"
    assert mod._quota_sentinel_is_active(payload)

    monkeypatch.setattr(
        mod,
        "_quota_now_utc",
        lambda: now + timedelta(minutes=5, microseconds=1),
    )
    assert not mod._quota_sentinel_is_active(payload)


def test_quota_sentinel_uses_utc_midnight_for_configured_daily_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = datetime(2026, 8, 31, 23, 0, tzinfo=UTC)
    monkeypatch.setattr(mod, "_quota_now_utc", lambda: now)
    cfg = mod.LLMConfig(
        provider="openai_compatible",
        model="free-model",
        provider_rpd_limit=50,
        provider_rate_limit_scope="free-daily",
    )
    job = _minimal_llm_job(
        tmp_path,
        model="free-model",
        llm_config=mod._llm_config_to_dict(cfg),
    )

    sentinel = mod._write_quota_sentinel(
        job,
        {"error": "ProviderQuotaExhaustedError: daily quota exhausted"},
    )

    payload = mod._quota_sentinel_payload(sentinel or Path())
    assert payload is not None
    assert payload["reset_source"] == "configured_rpd_utc_midnight"
    assert payload["reset_at"] == "2026-09-01T00:00:00Z"


def test_run_config_guard_rejects_same_treatment_with_different_suite() -> None:
    base = {
        "models": ["hy3-ioa"],
        "interaction_mode": "logical_stateless",
        "agent_treatment_sha256_by_model": {"hy3-ioa": "a" * 64},
        "suite_manifest_sha256": "suite-a",
    }

    assert mod._run_config_treatment_compatibility_reasons(
        base, {**base, "suite_manifest_sha256": "suite-b"}
    ) == ["output_dir_immutable_run_scope_mismatch"]


@pytest.mark.parametrize(
    "field",
    (
        "formal_manifest_sha256",
        "formal_readiness_sha256",
        "formal_core_release_pipeline_sha256",
    ),
)
def test_run_config_guard_rejects_changed_formal_runtime_evidence_hash(
    field: str,
) -> None:
    base = {
        "models": ["hy3-ioa"],
        "interaction_mode": "logical_persistent",
        "agent_treatment_sha256_by_model": {"hy3-ioa": "a" * 64},
        "formal_run": True,
        field: "b" * 64,
    }

    assert mod._run_config_treatment_compatibility_reasons(
        base, {**base, field: "c" * 64}
    ) == ["output_dir_immutable_run_scope_mismatch"]


@pytest.mark.parametrize(
    "binding",
    [
        {},
        {"hy3-ioa": "not-a-sha256"},
        {"other-model": "a" * 64},
    ],
)
def test_run_config_guard_rejects_invalid_treatment_hash_map(
    binding: dict[str, str],
) -> None:
    config = {
        "models": ["hy3-ioa"],
        "interaction_mode": "logical_stateless",
        "agent_treatment_sha256_by_model": binding,
    }

    assert mod._run_config_treatment_compatibility_reasons(config, config)


def test_formal_output_namespace_is_automatic_and_resume_stable(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    root = tmp_path / "formal"
    leaf = root / f"treatment-{digest}"
    binding = {"hy3-ioa": digest}

    assert mod._resolve_logical_output_namespace(
        root,
        binding,
        formal_run=True,
    ) == leaf
    assert mod._resolve_logical_output_namespace(
        leaf,
        binding,
        formal_run=True,
    ) == leaf

    with pytest.raises(ValueError, match="treatment namespace mismatch"):
        mod._resolve_logical_output_namespace(
            root / f"treatment-{'b' * 64}",
            binding,
            formal_run=True,
        )

    assert mod._resolve_logical_output_namespace(
        root,
        binding,
        formal_run=False,
    ) == root


def test_formal_output_namespace_resolves_relative_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    digest = "a" * 64

    resolved = mod._resolve_logical_output_namespace(
        Path("batch_results/formal"),
        {"hy3-ioa": digest},
        formal_run=True,
    )

    assert resolved == (
        tmp_path / "batch_results" / "formal" / f"treatment-{digest}"
    ).resolve()


def test_formal_output_namespace_binding_fails_closed_on_tampered_config(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    out_dir = tmp_path / f"treatment-{digest}"
    meta = {
        "formal_run": True,
        "models": ["hy3-ioa"],
        "agent_treatment_sha256_by_model": {"hy3-ioa": digest},
        "output_dir": str(out_dir.resolve()),
        "output_namespace_treatment_sha256": digest,
    }

    assert mod._formal_output_namespace_binding_error(meta, out_dir) is None
    assert (
        mod._formal_output_namespace_binding_error(
            {**meta, "output_namespace_treatment_sha256": "b" * 64},
            out_dir,
        )
        == "formal_output_treatment_hash_mismatch"
    )
    assert (
        mod._formal_output_namespace_binding_error(
            {**meta, "output_dir": str(tmp_path / "other")},
            out_dir,
        )
        == "formal_output_directory_binding_mismatch"
    )


def test_formal_manifest_portabilizer_rewrites_repo_owned_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    payload = {
        "formal_manifest": str(tmp_path / "release" / "manifest.json"),
        "formal_readiness_path": str(tmp_path / "release" / "readiness.json"),
        "output_dir": str(tmp_path / "batch_results" / "treatment-a"),
    }

    assert mod._portable_formal_manifest(payload) == {
        "formal_manifest": "release/manifest.json",
        "formal_readiness_path": "release/readiness.json",
        "output_dir": "batch_results/treatment-a",
    }


def test_formal_run_config_is_portable_before_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    payload = {
        "formal_run": True,
        "formal_manifest": str(tmp_path / "release" / "manifest.json"),
        "formal_readiness_path": str(tmp_path / "release" / "readiness.json"),
        "output_dir": str(tmp_path / "batch_results" / "treatment-a"),
    }

    assert mod._portable_formal_run_config(payload) == {
        "formal_run": True,
        "formal_manifest": "release/manifest.json",
        "formal_readiness_path": "release/readiness.json",
        "output_dir": "batch_results/treatment-a",
    }
    assert mod._portable_formal_run_config({**payload, "formal_run": False}) == {
        **payload,
        "formal_run": False,
    }


def test_formal_finalize_json_tree_is_clone_portable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    out_dir = tmp_path / "batch_results" / "treatment-a"
    out_dir.mkdir(parents=True)
    treatment = "a" * 64
    row = _row("fam/a", "hy3-ioa", 42, 80.0)
    row.update(
        {
            "interaction_mode": "logical_persistent",
            "agent_treatment_sha256": treatment,
        }
    )
    (out_dir / "episodes.jsonl").write_text(
        json.dumps(row) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        mod,
        "audit_logs",
        lambda _path: {
            "output_dir": str(out_dir.resolve()),
            "log_files_orphan_interrupted": 0,
            "log_files_orphan_empty": 0,
            "orphan_in_flight_rows": 0,
        },
    )
    monkeypatch.setattr(mod, "write_log_audit_markdown", lambda *_a, **_k: None)
    monkeypatch.setattr(mod, "_write_plots", lambda *_a, **_k: [])

    mod._finalize_outputs(
        out_dir,
        [row],
        {
            "formal_run": True,
            "models": ["hy3-ioa"],
            "seeds": [42],
            "n_scenarios": 1,
            "scenario_seed_pairs": [["fam/a", 42]],
            "pass_k": 1,
            "interaction_mode": "logical_persistent",
            "agent_treatment_sha256_by_model": {"hy3-ioa": treatment},
            "output_dir": str(out_dir.resolve()),
        },
    )

    for path in out_dir.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert mod.canonicalize_repo_owned_paths(
            payload, repo_root=tmp_path
        ) == payload, path


def test_formal_main_dry_run_writes_only_treatment_leaf(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario = "logistics/formal-namespace-fixture-" + "x" * 240
    output_root = tmp_path / "formal-root"
    formal_binding = {
        "slice_name": "manifest_fixture",
        "dynamic_slice_spec": ("operate_v0_61_0", "readiness.json", {}),
        "release_id": "operate_v0_61_0",
        "manifest_sha256": "a" * 64,
        "release_tooling_sha256": "b" * 64,
        "readiness_sha256": "c" * 64,
        "core_release_pipeline_sha256": "d" * 64,
        "backend_runtime_closure_identity_sha256": "e" * 64,
    }
    monkeypatch.setattr(mod, "DYNAMIC_SCENARIO_SLICES", {})
    monkeypatch.setattr(mod, "_load_zhsrc_exports", lambda: {})
    monkeypatch.setattr(mod, "_load_named_zshrc_export", lambda _name: None)
    monkeypatch.setattr(
        mod,
        "resolve_formal_manifest_slice",
        lambda _path: formal_binding,
    )
    monkeypatch.setattr(mod, "_resolve_patterns", lambda *_args: [scenario])
    monkeypatch.setattr(mod, "_expand_scenarios", lambda _patterns: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda _slug: {
            "seed": 42,
            "scenario_signature": "fixture-signature",
            "backend_kind": "job_shop",
            "horizon_ticks": 1,
        },
    )
    monkeypatch.setattr(mod, "_bind_scenario_contracts_for_slice", lambda *_: None)
    monkeypatch.setattr(mod, "_suite_manifest_sha256_for_slice", lambda *_: "suite")
    monkeypatch.setattr(
        mod,
        "_suite_eligibility_binding",
        lambda _slice: {"suite_blocked": False, "formal_evaluation_ready": True},
    )
    monkeypatch.setattr(
        mod, "_validate_protocol21_formal_run",
        lambda *_a, **_k: ["formal_git_tree_must_be_clean"],
    )
    monkeypatch.setattr(
        mod,
        "_native_runtime_binding",
        lambda *_a, **_k: {"ok": True, "blockers": []},
    )
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {
            "git_commit": "commit",
            "git_metadata_available": True,
            "git_dirty": True,
            "git_status_short": [],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(output_root),
            "--formal-run",
            "--formal-manifest",
            str(tmp_path / "manifest.json"),
            "--models",
            "hy3-ioa",
            "--model-context-window-tokens",
            "192000",
            "--model-max-output-tokens",
            "65536",
            "--dry-run",
        ],
    )

    assert mod.main() == 0
    leaves = list(output_root.glob("treatment-*"))
    assert len(leaves) == 1
    config = json.loads((leaves[0] / "run_config.json").read_text(encoding="utf-8"))
    digest = config["agent_treatment_sha256_by_model"]["hy3-ioa"]
    profile_digest = config["agent_profile_sha256_by_model"]["hy3-ioa"]
    profile_identity = config["agent_profile_identity_by_model"]["hy3-ioa"]
    assert leaves[0].name == f"treatment-{digest}"
    assert mod._canonical_json_sha256(profile_identity) == profile_digest
    assert digest != profile_digest
    assert config["formal_release_id"] == "operate_v0_61_0"
    assert config["provider_rpm_limit"] is None
    assert config["provider_rpd_limit"] is None
    assert config["provider_rate_limit_scope"] is None
    assert profile_identity["provider_rpm_limit"] is None
    assert profile_identity["provider_rpd_limit"] is None
    assert config["output_dir"] == mod._portable_formal_manifest(
        {"path": str(leaves[0].resolve())}
    )["path"]
    assert config["output_namespace_treatment_sha256"] == digest
    assert not (output_root / "run_config.json").exists()
    assert config["git_dirty"] is True
    assert config["implementation_tree_sha256"] == mod.implementation_identity(
        mod.REPO_ROOT
    )["implementation_tree_sha256"]
    config_path = leaves[0] / "run_config.json"
    config["git_dirty"] = False  # A prior compatible clean execution record.
    config_path.write_text(json.dumps(config))
    (leaves[0] / "episodes.jsonl").write_text('{"interrupted":')
    (leaves[0] / "path_shorten_map.json").write_text('{"prior": true}')
    def snapshot():
        return {
            path.relative_to(leaves[0]).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in leaves[0].rglob("*") if path.is_file()
        }
    with mod._acquire_output_dir_lock(leaves[0]):
        original = snapshot()
        assert mod.main() == 0  # Even a locked existing namespace is read-only.
        assert snapshot() == original
    original_argv = list(sys.argv)
    real_acquire = mod._acquire_output_dir_lock
    for mismatch in (False, True):
        raced = {}
        def acquire_with_new_config(out_dir):
            handle = real_acquire(out_dir)
            payload = {
                **config, "max_tokens": config["max_tokens"] + int(mismatch),
                "output_dir": mod._portable_formal_manifest({"path": str(out_dir.resolve())})["path"],
            }
            target = out_dir / "run_config.json"
            target.write_text(json.dumps(payload))
            raced["path"], raced["bytes"] = target, target.read_bytes()
            return handle
        monkeypatch.setattr(mod, "_acquire_output_dir_lock", acquire_with_new_config)
        argv = list(original_argv)
        argv[argv.index("--output-dir") + 1] = str(tmp_path / f"race-{mismatch}")
        monkeypatch.setattr(sys, "argv", argv)
        assert mod.main() == int(mismatch)
        assert raced["path"].read_bytes() == raced["bytes"]
    monkeypatch.setattr(mod, "_acquire_output_dir_lock", real_acquire)
    monkeypatch.setattr(sys, "argv", original_argv)
    config["max_tokens"] += 1
    config_path.write_text(json.dumps(config))
    incompatible = snapshot()
    assert mod.main() == 1
    assert snapshot() == incompatible
    monkeypatch.setattr(sys, "argv", [arg for arg in sys.argv if arg != "--dry-run"])
    assert mod.main() == 1  # Actual execution still rejects this dirty checkout.


def test_run_config_loader_rejects_corrupt_existing_json(tmp_path: Path) -> None:
    path = tmp_path / "run_config.json"
    path.write_text('{"interaction_mode":', encoding="utf-8")
    original = path.read_bytes()

    with pytest.raises(ValueError, match="invalid existing run config"):
        mod._load_run_config_fail_closed(path)

    assert path.read_bytes() == original


def test_retry_quarantine_rejects_path_outside_output_logs(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    outside = tmp_path / "outside.log"
    outside.write_text("do not mutate", encoding="utf-8")

    with pytest.raises(ValueError, match="escapes output logs directory"):
        mod._quarantine_retry_cell_logs(
            output_dir,
            {"sample_orphan_interrupted_logs": [{"path": str(outside)}]},
        )

    assert outside.read_text(encoding="utf-8") == "do not mutate"


def test_main_output_dir_rejects_a_b_and_then_resumes_a(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scenario = "traffic/treatment_guard"
    output_dir = tmp_path / "treatment_guard"
    monkeypatch.setattr(mod, "_resolve_patterns", lambda *_: [scenario])
    monkeypatch.setattr(mod, "_expand_scenarios", lambda _: [scenario])
    monkeypatch.setattr(
        mod,
        "load_scenario_yaml",
        lambda _: {
            "seed": 42,
            "scenario_signature": "sig-treatment",
            "backend_kind": "mock_sumo",
            "horizon_ticks": 1,
        },
    )
    monkeypatch.setattr(
        mod,
        "_suite_eligibility_binding",
        lambda _: {
            "suite_blocked": False,
            "formal_evaluation_ready": False,
        },
    )
    monkeypatch.setattr(
        mod, "_suite_manifest_sha256_for_slice", lambda *_: "suite-treatment"
    )
    monkeypatch.setattr(mod, "_load_zhsrc_exports", lambda: {})
    monkeypatch.setattr(
        mod,
        "_git_metadata",
        lambda: {
            "git_metadata_available": True,
            "git_commit": "abc",
            "git_dirty": False,
            "git_status_short": [],
        },
    )

    def run(interaction_mode: str) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "batch_llm_eval.py",
                "--output-dir",
                str(output_dir),
                "--models",
                "hy3-ioa",
                "--interaction-mode",
                interaction_mode,
                "--dry-run",
            ],
        )
        return mod.main()

    assert run("logical_stateless") == 0
    original = (output_dir / "run_config.json").read_bytes()
    assert run("logical_persistent") == 1
    assert (output_dir / "run_config.json").read_bytes() == original
    assert run("logical_stateless") == 0


def test_formal_row_treatment_binding_rejects_persistent_and_mixed_rows() -> None:
    expected = "a" * 64
    other = "b" * 64
    meta = {
        "models": ["hy3-ioa"],
        "interaction_mode": "logical_stateless",
        "agent_treatment_sha256_by_model": {"hy3-ioa": expected},
    }
    stateless = {
        "status": "ok",
        "model": "hy3-ioa",
        "interaction_mode": "logical_stateless",
        "agent_treatment_sha256": expected,
    }
    persistent = {
        **stateless,
        "interaction_mode": "logical_persistent",
        "agent_treatment_sha256": other,
    }

    assert mod._formal_treatment_binding_reasons(meta, [stateless]) == []
    assert set(
        mod._formal_treatment_binding_reasons(meta, [stateless, persistent])
    ) == {
        "formal_row_interaction_mode_mismatch",
        "formal_row_agent_treatment_mismatch",
        "formal_agent_treatment_not_homogeneous",
    }


def test_select_rows_for_treatment_excludes_other_treatment() -> None:
    expected = "a" * 64
    rows = [
        {
            "model": "hy3-ioa",
            "interaction_mode": "logical_stateless",
            "agent_treatment_sha256": expected,
        },
        {
            "model": "hy3-ioa",
            "interaction_mode": "logical_persistent",
            "agent_treatment_sha256": "b" * 64,
        },
    ]
    meta = {
        "models": ["hy3-ioa"],
        "interaction_mode": "logical_stateless",
        "agent_treatment_sha256_by_model": {"hy3-ioa": expected},
    }

    assert mod._select_rows_for_treatment(rows, meta) == [rows[0]]


@pytest.mark.parametrize(
    "binding",
    [
        {},
        {"hy3-ioa": "corrupt"},
        {"other-model": "a" * 64},
    ],
)
def test_invalid_treatment_binding_selects_no_analysis_rows(
    binding: dict[str, str],
) -> None:
    rows = [
        {
            "model": "hy3-ioa",
            "interaction_mode": "logical_stateless",
            "agent_treatment_sha256": "a" * 64,
        }
    ]
    meta = {
        "models": ["hy3-ioa"],
        "interaction_mode": "logical_stateless",
        "agent_treatment_sha256_by_model": binding,
    }

    assert mod._select_rows_for_treatment(rows, meta) == []


def test_main_rejects_nonpositive_output_budget(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_llm_eval.py",
            "--output-dir",
            str(tmp_path / "invalid-budget"),
            "--max-tokens",
            "0",
            "--dry-run",
        ],
    )

    assert mod.main() == 1
