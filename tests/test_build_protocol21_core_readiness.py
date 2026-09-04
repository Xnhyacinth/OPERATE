from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

from core.implementation_identity import implementation_identity
from core.source_consumption_contract import resolve_declared_sources
from runner.resume import recompute_signature_with_seed
from scripts.build_protocol21_core_readiness import build_readiness_from_paths
from tests.test_agentic_core_contract import (
    FINGERPRINT,
    PROTOCOL,
    SCORER,
    _artifacts,
)


def _write_fixture(
    tmp_path: Path,
    *,
    mutate: Callable[[dict], None] | None = None,
) -> dict[str, Path]:
    artifacts = _artifacts()
    row = artifacts["source_suite"]["scenarios"][0]
    scenario = tmp_path / "scenario.yaml"
    provenance = tmp_path / "source.csv"
    provenance.write_text("value\n1\n", encoding="utf-8")
    scenario_body = {
        "seed_id": row["scenario_id"],
        "family": "incident_response",
        "domain": row["domain"],
        # This fixture tests readiness wiring. A formal-capable descriptor is
        # used so backend-fidelity rejection does not mask the binding checks.
        "backend_kind": "pandapower_lv",
        "horizon_ticks": 12,
        "seed": 42,
        "difficulty_level": "high",
        "difficulty_mode": "time_pressure",
        "provenance": {
            "data_source": "fixture",
            "files": [str(provenance)],
            "source_locked": True,
        },
        "source_contract": {
            "runtime_input": [str(provenance)],
            "derivation_input": [],
            "implementation_asset": [],
            "metadata": [],
            "license": [],
        },
    }
    signature = recompute_signature_with_seed(scenario_body, 42)
    scenario_body["scenario_signature"] = signature
    import yaml

    scenario.write_text(
        yaml.safe_dump(scenario_body, sort_keys=False),
        encoding="utf-8",
    )
    old_signature = row["scenario_signature"]
    for payload in artifacts.values():
        for key in ("scenarios", "results", "samples"):
            for item in payload.get(key) or []:
                if item.get("scenario_signature") == old_signature:
                    item["scenario_signature"] = signature
                    item["backend_kind"] = scenario_body["backend_kind"]
    row["path"] = str(scenario)
    row["scenario_signature"] = signature
    row["backend_kind"] = scenario_body["backend_kind"]
    row["family"] = scenario_body["family"]
    row["difficulty_level"] = scenario_body["difficulty_level"]
    row["difficulty_mode"] = scenario_body["difficulty_mode"]
    row["seed"] = 42
    row["horizon_ticks"] = 12
    row["provenance_files"] = [str(provenance)]
    tree = implementation_identity()["implementation_tree_sha256"]
    for payload in artifacts.values():
        payload["implementation_tree_sha256"] = tree

    agentic_row = {
        "scenario_id": row["scenario_id"],
        "scenario_signature": row["scenario_signature"],
        "domain": row["domain"],
        "backend_kind": row["backend_kind"],
        "status": "passed",
        "blockers": [],
        "checks": {},
        "agentic_contract": {},
    }
    source_gate_row = artifacts["source_grounded"]["results"][0]
    _, source_hashes, _ = resolve_declared_sources(
        scenario_body,
        repo_root=tmp_path,
    )
    source_gate_row["scenario_file_sha256"] = hashlib.sha256(
        scenario.read_bytes()
    ).hexdigest()
    source_gate_row["source_file_hashes"] = source_hashes
    core_row = deepcopy(row)
    core_row.update(
        {
            "status": "core_locked",
            "core_disposition": "core_locked",
            "protocol21_admission_status": "passed",
            "admission_fingerprint": "a" * 64,
        }
    )
    core = {
        "schema_version": "2.1",
        "status": "protocol21_core_candidate",
        "formal_evaluation_ready": False,
        "leaderboard_eligible": False,
        "evaluation_protocol": {
            "version": PROTOCOL,
            "implementation_fingerprint": FINGERPRINT,
        },
        "scoring_version": SCORER,
        "implementation_tree_sha256": tree,
        "selection_policy": "quality_maximal_v1",
        "constraint_validation": {
            "preserve_each_eligible_family_difficulty_cell": True,
            "max_domain_share_passed": True,
            "max_backend_share_passed": True,
            "effective_source_identity_unique": True,
            "quality_maximal_admission_passed": True,
        },
        "incremental_freeze_ledger": [
            {
                "scenario_id": core_row["scenario_id"],
                "scenario_signature": core_row["scenario_signature"],
                "source_denominator_key": (
                    core_row.get("source_denominator_key")
                    or (core_row.get("case_ledger") or {}).get(
                        "source_denominator_key"
                    )
                ),
                "disposition": "core_locked",
                "admission_fingerprint": core_row["admission_fingerprint"],
            }
        ],
        "scenarios": [core_row],
    }
    artifacts["agentic_contract"] = {
        "schema_version": "1.0",
        "status": "complete",
        "evaluation_semantics": {
            "protocol_version": PROTOCOL,
            "implementation_fingerprint": FINGERPRINT,
            "scoring_version": SCORER,
        },
        "n_expected": 1,
        "n_completed": 1,
        "n_passed": 1,
        "implementation_tree_sha256": tree,
        "results": [agentic_row],
    }
    artifacts["preflight"] = {
        "status": "passed",
        "n_expected": 1,
        "n_completed": 1,
        "n_fatal": 0,
        "implementation_tree_sha256": tree,
    }
    artifacts["source_consumption"] = {
        "status": "complete",
        "evaluation_semantics": {
            "protocol_version": PROTOCOL,
            "implementation_fingerprint": FINGERPRINT,
            "scoring_version": SCORER,
        },
        "implementation_tree_sha256": tree,
        "n_expected": 1,
        "n_completed": 1,
        "results": [
            {
                "scenario_id": row["scenario_id"],
                "scenario_signature": signature,
                "status": "passed",
                "required_runtime_source_files": [str(provenance)],
                "required_derivation_source_files": [],
                "locked_source_hashes": source_hashes,
            }
        ],
    }
    artifacts["source_grounded"]["results"] = [source_gate_row]
    artifacts["core"] = core
    if mutate:
        mutate(artifacts)

    paths: dict[str, Path] = {}
    for name, payload in artifacts.items():
        if name in {"source_grounded", "agentic_contract", "core"}:
            continue
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path
    artifacts["source_grounded"]["input_bindings"] = {
        name: {
            "path": str(paths[name].resolve()),
            "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            "implementation_tree_sha256": tree,
        }
        for name in (
            "source_suite",
            "behavioral",
            "source_consumption",
            "task_contracts",
            "complexity",
            "strategy_depth",
        )
    }
    source_grounded_path = tmp_path / "source_grounded.json"
    source_grounded_path.write_text(
        json.dumps(artifacts["source_grounded"]), encoding="utf-8"
    )
    paths["source_grounded"] = source_grounded_path
    artifacts["agentic_contract"]["input_bindings"] = {
        name: {
            "path": str(paths[name].resolve()),
            "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            "implementation_tree_sha256": tree,
        }
        for name in (
            "source_suite",
            "behavioral",
            "source_consumption",
            "task_contracts",
            "complexity",
            "observed_depth",
            "strategy_depth",
            "source_grounded",
        )
    }
    agentic_path = tmp_path / "agentic_contract.json"
    agentic_path.write_text(
        json.dumps(artifacts["agentic_contract"]), encoding="utf-8"
    )
    paths["agentic_contract"] = agentic_path
    artifacts["core"]["input_bindings"] = {
        name: {
            "path": str(paths[name].resolve()),
            "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
            "implementation_tree_sha256": tree,
        }
        for name in (
            "source_suite",
            "behavioral",
            "task_contracts",
            "observed_depth",
            "strategy_depth",
            "source_grounded",
            "agentic_contract",
        )
    }
    core_path = tmp_path / "core.json"
    core_path.write_text(json.dumps(artifacts["core"]), encoding="utf-8")
    paths["core"] = core_path
    coverage = {
        "schema_version": "1.0",
        "status": "complete",
        "evaluation_semantics": {
            "protocol_version": PROTOCOL,
            "implementation_fingerprint": FINGERPRINT,
            "scoring_version": SCORER,
        },
        "implementation_tree_sha256": tree,
        "n_expected": 1,
        "n_completed": 1,
        "release_coverage_passed": True,
        "release_coverage_blockers": [],
        "input_bindings": {
            "core": {
                "path": str(core_path.resolve()),
                "sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
                "implementation_tree_sha256": tree,
            }
        },
    }
    coverage_path = tmp_path / "release_coverage.json"
    coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
    paths["release_coverage"] = coverage_path
    return paths


def test_complete_v21_evidence_allows_formal_sampling_not_leaderboard(
    tmp_path: Path,
) -> None:
    report = build_readiness_from_paths(**_write_fixture(tmp_path))

    assert report["status"] == "formal_evaluation_ready"
    assert report["formal_evaluation_ready"] is True
    assert report["leaderboard_eligible"] is False
    assert report["leaderboard_blockers"] == [
        "formal_logical_persistent_evaluation_pending",
        "formal_realtime_persistent_evaluation_pending",
    ]
    assert report["formal_run_contract"]["maximum_max_workers"] == 32
    assert report["formal_run_contract"]["minimum_pass_k"] == 1
    assert (
        report["formal_run_contract"]["required_interaction_mode"]
        == "logical_persistent"
    )
    assert report["formal_run_contract"]["required_construct_contract"] == (
        "operational_agency.v1"
    )
    wakeup_policy = {
        "session_start": True,
        "typed_actionable_events": True,
        "agent_scheduled_reviews": True,
        "harness_periodic_supervisory_scan": False,
        "unknown_events_actionable": False,
    }
    assert report["formal_run_contract"]["wakeup_policy"] == wakeup_policy
    realtime_contract = report["formal_run_contract"][
        "realtime_formal_contract"
    ]
    assert realtime_contract["contract_version"] == "realtime_persistent.v2"
    assert realtime_contract["scorecard_version"] == (
        "realtime-diagnostics/1.6"
    )
    assert realtime_contract["diagnostic_schema_version"] == (
        "realtime-diagnostics/1.6"
    )
    assert realtime_contract["batch_schema_version"] == (
        "realtime-formal-batch/1.1"
    )
    assert realtime_contract["scorecard_schema_version"] == (
        "realtime-formal-scorecard/1.1"
    )
    assert realtime_contract["episode_schema_version"] == (
        "realtime-episode/1.1"
    )
    assert realtime_contract["treatment_schema_version"] == (
        "realtime-treatment/1.1"
    )
    assert realtime_contract["realtime_coordinator"] == "realtime_episode_v5"
    assert realtime_contract["wakeup_policy"] == wakeup_policy
    assert all(
        row["construct_contract"] == "operational_agency.v1"
        for row in report["scenarios"]
    )
    assert report["scoring_version"] == SCORER
    assert report["primary_leaderboard_formula_version"] == (
        "effective_source_backend_domain_macro_v1"
    )
    assert report["primary_inference_version"] == (
        "physical_cluster_hierarchical_bootstrap_randomization_v1"
    )
    assert report["task_completion_input_unit"] == "fraction_0_1"
    assert report["task_completion_score_unit"] == "points_0_100"
    assert report["weighted_equity_formula_version"] == (
        "entity_criticality_unit_interval_v2"
    )


def test_distribution_concentration_is_diagnostic_not_an_admission_gate(
    tmp_path: Path,
) -> None:
    def mutate(artifacts: dict) -> None:
        constraints = artifacts["core"]["constraint_validation"]
        constraints["max_domain_share_passed"] = False
        constraints["max_backend_share_passed"] = False
        constraints["quality_maximal_admission_passed"] = True

    report = build_readiness_from_paths(**_write_fixture(tmp_path, mutate=mutate))

    assert report["formal_evaluation_ready"] is True
    assert "core_selection_constraints_failed" not in report["formal_run_blockers"]


def test_legacy_family_difficulty_cell_preservation_is_diagnostic(
    tmp_path: Path,
) -> None:
    def mutate(artifacts: dict) -> None:
        constraints = artifacts["core"]["constraint_validation"]
        constraints["preserve_each_eligible_family_difficulty_cell"] = False

    report = build_readiness_from_paths(**_write_fixture(tmp_path, mutate=mutate))

    assert report["formal_evaluation_ready"] is True
    assert "core_selection_constraints_failed" not in report["formal_run_blockers"]


def test_quality_core_v2_does_not_repeat_exact_strategy_minimality_gate(
    tmp_path: Path,
) -> None:
    def mutate(artifacts: dict) -> None:
        artifacts["source_suite"]["constraints"] = {
            "core_admission_profile": "quality_core_v2"
        }
        for name in ("source_grounded", "agentic_contract"):
            artifacts[name]["admission_profile"] = "quality_core_v2"
            artifacts[name]["results"][0]["admission_profile"] = "quality_core_v2"
        artifacts["core"]["constraint_validation"][
            "core_admission_profile"
        ] = "quality_core_v2"
        depth = artifacts["strategy_depth"]["samples"][0]
        depth["core_action"] = "hold_relabel_or_redesign"
        depth["difficulty_calibration"] = {
            "version": "source_grounded_behavioral_v3",
            "status": "held",
            "declared_difficulty_level": "high",
            "calibrated_difficulty_level": None,
            "declared_level_matches_evidence": False,
        }
        artifacts["agentic_contract"]["results"][0][
            "diagnostic_blockers"
        ] = ["standing_plan_response_unproven"]

    report = build_readiness_from_paths(**_write_fixture(tmp_path, mutate=mutate))

    assert report["formal_evaluation_ready"] is True
    assert "strategy_depth_not_kept" not in report["formal_run_blockers"]
    assert (
        "difficulty_calibration_not_current_or_matching"
        not in report["formal_run_blockers"]
    )
    assert set(
        report["diagnostic_row_labels"]["traffic/incident/high/example"]
    ) == {
        "difficulty_label_unproven",
        "standing_plan_response_unproven",
        "strategy_depth_unproven",
    }


def test_diagnostic_replay_artifacts_do_not_reject_a_locked_core_row(
    tmp_path: Path,
) -> None:
    def mutate(artifacts: dict) -> None:
        for name in (
            "task_contracts",
            "complexity",
            "observed_depth",
            "strategy_depth",
        ):
            artifact = artifacts[name]
            artifact["status"] = "diagnostic_incomplete"
            artifact["evaluation_semantics"]["protocol_version"] = "diagnostic-v0"
            artifact["n_completed"] = 0
            artifact["results"] = []
            artifact["samples"] = []

    report = build_readiness_from_paths(**_write_fixture(tmp_path, mutate=mutate))

    assert report["formal_evaluation_ready"] is True
    assert report["formal_run_blockers"] == []
    assert set(report["diagnostic_artifact_issues"]) == {
        "complexity",
        "observed_depth",
        "strategy_depth",
        "task_contracts",
    }
    for issues in report["diagnostic_artifact_issues"].values():
        assert "artifact_incomplete" in issues
        assert "artifact_semantics_stale" in issues


def test_readiness_rejects_unlocked_or_fingerprint_mismatched_core_row(
    tmp_path: Path,
) -> None:
    def mutate(artifacts: dict) -> None:
        artifacts["core"]["scenarios"][0]["status"] = "working_set"
        artifacts["core"]["incremental_freeze_ledger"][0][
            "admission_fingerprint"
        ] = "b" * 64

    report = build_readiness_from_paths(**_write_fixture(tmp_path, mutate=mutate))

    assert report["formal_evaluation_ready"] is False
    assert "core_row_not_locked" in report["formal_run_blockers"]
    assert "admission_fingerprint_mismatch" in report["formal_run_blockers"]


def test_stale_task_replay_semantics_are_diagnostic(tmp_path: Path) -> None:
    def mutate(artifacts: dict) -> None:
        artifacts["task_contracts"]["evaluation_semantics"][
            "protocol_version"
        ] = "2.0"

    report = build_readiness_from_paths(**_write_fixture(tmp_path, mutate=mutate))

    assert report["formal_evaluation_ready"] is True
    assert "artifact_semantics_stale" not in report["formal_run_blockers"]
    assert report["diagnostic_artifact_issues"]["task_contracts"] == [
        "artifact_semantics_stale"
    ]


def test_readiness_rejects_cross_profile_gate_artifacts(tmp_path: Path) -> None:
    def mutate(artifacts: dict) -> None:
        artifacts["agentic_contract"]["admission_profile"] = "quality_core_v2"
        artifacts["agentic_contract"]["results"][0][
            "admission_profile"
        ] = "quality_core_v2"

    report = build_readiness_from_paths(**_write_fixture(tmp_path, mutate=mutate))

    assert report["formal_evaluation_ready"] is False
    assert "admission_profile_mismatch" in report["formal_run_blockers"]


def test_readiness_rejects_explicit_cross_profile_source_row(tmp_path: Path) -> None:
    def mutate(artifacts: dict) -> None:
        artifacts["source_suite"]["scenarios"][0][
            "admission_profile"
        ] = "quality_core_v2"

    report = build_readiness_from_paths(**_write_fixture(tmp_path, mutate=mutate))

    assert report["formal_evaluation_ready"] is False
    assert "admission_profile_mismatch" in report["formal_run_blockers"]


def test_signature_mismatch_and_missing_gate_rows_are_rejected(
    tmp_path: Path,
) -> None:
    def mutate(artifacts: dict) -> None:
        artifacts["task_contracts"]["results"][0][
            "scenario_signature"
        ] = "different"
        artifacts["source_grounded"]["results"] = []

    report = build_readiness_from_paths(**_write_fixture(tmp_path, mutate=mutate))

    assert report["formal_evaluation_ready"] is False
    assert "source_gate_row_missing" in report["formal_run_blockers"]
    assert "canonical_identity_mismatch" in report[
        "diagnostic_artifact_issues"
    ]["task_contracts"]


def test_diagnostic_outcomes_do_not_repeat_admission_but_domain_loss_does(
    tmp_path: Path,
) -> None:
    def mutate(artifacts: dict) -> None:
        artifacts["agentic_contract"]["results"][0]["status"] = "held"
        artifacts["agentic_contract"]["n_passed"] = 0
        task = artifacts["task_contracts"]["results"][0]
        task["status"] = "failed"
        task["completed"] = False
        task["terminal_integrity"]["release_ready"] = False
        artifacts["strategy_depth"]["samples"][0]["core_action"] = (
            "hold_relabel_or_redesign"
        )
        artifacts["observed_depth"]["samples"][0]["disposition"] = (
            "contradicted_by_oracle_tick_floor"
        )
        artifacts["source_suite"]["scenarios"].append(
            {
                **deepcopy(artifacts["source_suite"]["scenarios"][0]),
                "scenario_id": "power_grid/example",
                "scenario_signature": "sig-power",
                "domain": "power_grid",
            }
        )
        artifacts["source_suite"]["n_expected"] = 2
        artifacts["source_suite"]["n_completed"] = 2

    report = build_readiness_from_paths(**_write_fixture(tmp_path, mutate=mutate))

    assert report["formal_evaluation_ready"] is False
    assert "task_contract_not_passed" not in report["formal_run_blockers"]
    assert "agentic_contract_not_passed" not in report["formal_run_blockers"]
    assert "strategy_depth_not_kept" not in report["formal_run_blockers"]
    assert "depth_contradiction" not in report["formal_run_blockers"]
    assert "source_domain_missing_from_core" in report["formal_run_blockers"]


def test_changed_input_content_breaks_bound_hash(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    behavioral = json.loads(paths["behavioral"].read_text(encoding="utf-8"))
    behavioral["tampered_after_core_binding"] = True
    paths["behavioral"].write_text(json.dumps(behavioral), encoding="utf-8")

    report = build_readiness_from_paths(**paths)

    assert report["formal_evaluation_ready"] is False
    assert "artifact_hash_mismatch" in report["formal_run_blockers"]


def test_failed_release_coverage_is_composition_diagnostic(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    coverage = json.loads(paths["release_coverage"].read_text(encoding="utf-8"))
    coverage["release_coverage_passed"] = False
    coverage["release_coverage_blockers"] = [
        "min_effective_sources_per_domain_not_met"
    ]
    paths["release_coverage"].write_text(json.dumps(coverage), encoding="utf-8")

    report = build_readiness_from_paths(**paths)

    assert report["formal_evaluation_ready"] is True
    assert "release_coverage_failed" not in report["formal_run_blockers"]
    assert report["release_composition_ready"] is False
    assert report["release_composition_blockers"] == [
        "min_effective_sources_per_domain_not_met"
    ]


def test_quality_core_v2_treats_depth_contradiction_as_diagnostic(
    tmp_path: Path,
) -> None:
    def mutate(artifacts: dict) -> None:
        artifacts["source_suite"]["constraints"] = {
            "core_admission_profile": "quality_core_v2"
        }
        for name in ("source_grounded", "agentic_contract"):
            artifacts[name]["admission_profile"] = "quality_core_v2"
            artifacts[name]["results"][0]["admission_profile"] = "quality_core_v2"
        artifacts["core"]["constraint_validation"][
            "core_admission_profile"
        ] = "quality_core_v2"
        artifacts["observed_depth"]["samples"][0]["disposition"] = (
            "contradicted_by_oracle_tick_floor"
        )

    report = build_readiness_from_paths(**_write_fixture(tmp_path, mutate=mutate))

    assert report["formal_evaluation_ready"] is True
    assert "depth_contradiction" not in report["formal_run_blockers"]
    assert report["diagnostic_depth_contradictions"] == [
        "traffic/incident/high/example"
    ]
