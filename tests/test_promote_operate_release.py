from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from threading import Event, Thread
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from core.implementation_identity import implementation_identity
from evaluation.scorer import SCORING_VERSION
from runner.resume import recompute_signature_with_seed
from scripts.build_protocol21_core_readiness import FORMAL_RUN_CONTRACT
from scripts.finalize_operate_candidate_pool import build_compact_candidate_closure
from scripts.promote_operate_release import (
    STAGE_FILES,
    _public_release_blockers,
    _validate_backend_runtime_closure,
    _validate_candidate_closure_input,
    _validate_formal_run_contract_for_release,
    _validate_parent_core_ancestry,
    _validate_scenario_yaml,
    _validated_relocation_identity_map,
    promote_release as _promote_release,
)
from scripts.verify_release_integrity import build_protocol21_core_integrity_report

STAGES = (
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
    "release_coverage",
    "readiness",
)

LEGACY_FORMAL_RUN_CONTRACT = deepcopy(FORMAL_RUN_CONTRACT)
LEGACY_FORMAL_RUN_CONTRACT.pop("wakeup_policy", None)
LEGACY_FORMAL_RUN_CONTRACT["realtime_formal_contract"] = {
    **LEGACY_FORMAL_RUN_CONTRACT["realtime_formal_contract"],
    "contract_version": "realtime_persistent.v1",
    "realtime_coordinator": "realtime_episode_v4",
}
LEGACY_FORMAL_RUN_CONTRACT["realtime_formal_contract"].pop(
    "wakeup_policy", None
)

# Most fixtures below intentionally model the historical v0.58 release. Keep
# their identity explicit instead of inheriting the active production defaults.
def promote_release(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("release_id", "operate_v0_58_0")
    kwargs.setdefault("release_version", "0.58.0")
    kwargs.setdefault("selection_policy", "quality_core_v2_v058")
    kwargs.setdefault("core_settings_stamp", "v0.58.0-settings")
    with patch("scripts.promote_operate_release.RELEASE_ID", "operate_v0_58_0"):
        return _promote_release(**kwargs)


def test_v058_adds_runtime_evidence_distribution_blocker() -> None:
    assert _public_release_blockers("0.58.0") == [
        "formal_logical_persistent_evaluation_pending",
        "formal_realtime_persistent_evaluation_pending",
        "formal_runtime_evidence_distribution_pending",
    ]
    assert _public_release_blockers("0.57.1") == [
        "formal_logical_persistent_evaluation_pending",
        "formal_realtime_persistent_evaluation_pending",
    ]


def test_formal_contract_validation_preserves_history_and_rejects_v061_drift() -> None:
    _validate_formal_run_contract_for_release(
        LEGACY_FORMAL_RUN_CONTRACT,
        release_id="operate_v0_60_0",
    )
    drifted = deepcopy(FORMAL_RUN_CONTRACT)
    drifted["realtime_formal_contract"]["wakeup_policy"] = {
        **drifted["realtime_formal_contract"]["wakeup_policy"],
        "harness_periodic_supervisory_scan": True,
    }

    with pytest.raises(ValueError, match="formal_wakeup_contract_missing_or_invalid"):
        _validate_formal_run_contract_for_release(
            drifted,
            release_id="operate_v0_61_0",
        )


def test_promotion_lock_serializes_competing_publishers(tmp_path: Path) -> None:
    from scripts import promote_operate_release as promoter

    repo = tmp_path / "repo"
    repo.mkdir()
    attempted = Event()
    entered = Event()

    def _contender() -> None:
        attempted.set()
        with promoter._exclusive_promotion_lock(repo):
            entered.set()

    contender = Thread(target=_contender)
    with promoter._exclusive_promotion_lock(repo):
        contender.start()
        assert attempted.wait(timeout=1)
        assert not entered.wait(timeout=0.05)
    contender.join(timeout=1)
    assert not contender.is_alive()
    assert entered.is_set()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _binding(repo: Path, path: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(repo).as_posix(),
        "sha256": _sha256(path),
    }


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _runtime_identity_sha256(payload: dict) -> str:
    return _canonical_sha256(
        {key: value for key, value in payload.items() if key != "identity_sha256"}
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    (repo / "core").mkdir(parents=True)
    (repo / "core/example.py").write_text("VALUE = 1\n", encoding="utf-8")
    uv_lock = repo / "uv.lock"
    uv_lock.write_text("version = 1\n", encoding="utf-8")
    identity = implementation_identity(repo)
    tree = identity["implementation_tree_sha256"]
    release_pipeline_sha256 = identity["core_release_pipeline_sha256"]
    release_tooling_sha256 = identity["release_tooling_sha256"]
    scenario_rel = "scenarios/staging/v0_55/demo.yaml"
    scenario = repo / scenario_rel
    scenario.parent.mkdir(parents=True)
    dimensions = {
        name: {"applicable": False, "reason": "fixture_not_applicable"}
        for name in (
            "system_survival",
            "economic_cost",
            "safety_violation",
            "weighted_equity_score",
            "ethical_quality",
            "stakeholder_management",
            "adaptive_replanning",
            "information_efficiency",
            "foresight_score",
            "optimality_gap",
            "counterfactual_prevention",
            "tool_use_efficiency",
            "stakeholder_equity",
        )
    }
    scenario.write_text(
        yaml.safe_dump(
            {
                "scenario_id": "logistics/job_shop_dispatch/time_pressure/high/demo",
                "seed_id": "logistics/job_shop_dispatch/time_pressure/high/demo",
                "domain": "logistics",
                "family": "job_shop_dispatch",
                "backend_kind": "dynasched_flexible_job_shop",
                "difficulty_mode": "time_pressure",
                "difficulty_level": "high",
                "seed": 42,
                "horizon_ticks": 4,
                "tick_minutes": 1,
                "load_assignments": [],
                "perturbations": [],
                "dilemmas": [],
                "backend_config": {
                    "observation_budget_chars": 16000,
                    "dimension_applicability": dimensions,
                },
                "provenance": {
                    "data_source": "DynaSchedBench",
                    "url": "https://github.com/dsbx7/DynaSchedBench",
                    "license": "Apache-2.0",
                    "lock_strategy": "git_commit_plus_per_file_sha256",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    scenario_signature = recompute_signature_with_seed(
        yaml.safe_load(scenario.read_text(encoding="utf-8")), 42
    )
    source_asset = repo / "sources/demo.json"
    _write_json(source_asset, {"jobs": ["source-job"]})
    physical_lock = {
        "schema_version": "source_asset_graph_v1",
        "backend_kind": "dynasched_flexible_job_shop",
        "upstream_commit": "08975bf4a0473c5dff9177393bc6743db9ddc946",
        "bundle_path": "rep02",
        "required_source_assets": [
            {"declared_path": "events.jsonl", "sha256": "a" * 64},
            {"declared_path": "input_model.json", "sha256": "b" * 64},
        ],
    }
    row = {
        "scenario_id": "logistics/job_shop_dispatch/time_pressure/high/demo",
        "scenario_signature": scenario_signature,
        "path": scenario_rel,
        "domain": "logistics",
        "family": "job_shop_dispatch",
        "backend_kind": "dynasched_flexible_job_shop",
        "difficulty_mode": "time_pressure",
        "difficulty_level": "high",
        "source_denominator_key": "dynasched:rep02",
        "structural_fingerprint": "fixture-structure",
        "physical_source_key": "noncanonical-legacy-hash",
        "case_ledger": {
            "source_denominator_key": "dynasched:rep02",
            "physical_source_lock": physical_lock,
        },
        "status": "core_locked",
        "core_disposition": "core_locked",
        "construct_contract": "operational_agency.v1",
        "historical_admission": {
            "status": "previously_core_locked_requires_current_replay"
        },
    }
    source_suite = repo / "release/operate_v0_58_0/protocol21_source_suite.json"
    _write_json(
        source_suite,
        {
            "schema_version": "protocol2.1-working-set-v1",
            "status": "working_set",
            "leaderboard_eligible": False,
            "n_scenarios": 1,
            "scenarios": [row],
            "candidate_import_partition": {
                "schema_version": "operate-candidate-import-partition-v1",
                "status": "complete",
                "n_base": 0,
                "n_imported": 1,
                "base_identities": [],
                "imported_identities": [
                    {
                        "scenario_id": row["scenario_id"],
                        "scenario_signature": row["scenario_signature"],
                    }
                ],
            },
        },
    )
    pipeline = repo / "release/operate_v0_58_0_candidate/operate_v058_formal"
    artifacts: dict[str, Path] = {}
    for stage in STAGES[:-1]:
        artifact = pipeline / STAGE_FILES[stage]
        payload = {
            "schema_version": "1.0",
            "status": "complete",
            "implementation_tree_sha256": tree,
            "core_release_pipeline_sha256": release_pipeline_sha256,
        }
        if stage == "materialize_core":
            payload = {
                "schema_version": "2.1",
                "status": "protocol21_core_candidate",
                "implementation_tree_sha256": tree,
                "core_release_pipeline_sha256": release_pipeline_sha256,
                "n_source": 1,
                "n_selected": 1,
                "n_secondary": 0,
                "n_rejected": 0,
                "scenarios": [row],
                "secondary": [],
                "rejected": [],
                "disposition_counts": {"core_locked": 1},
                "input_bindings": {
                    "source_suite": {
                        **_binding(repo, source_suite),
                        "implementation_tree_sha256": tree,
                    }
                },
            }
        _write_json(
            artifact,
            payload,
        )
        artifacts[stage] = artifact
    readiness = pipeline / "protocol2_v21_core_readiness.json"
    _write_json(
        readiness,
        {
            "schema_version": "1.0",
            "status": "formal_evaluation_ready",
            "formal_evaluation_ready": True,
            "formal_run_blockers": [],
            "leaderboard_eligible": False,
            "scoring_version": SCORING_VERSION,
            "primary_leaderboard_formula_version": (
                "effective_source_backend_domain_macro_v1"
            ),
            "primary_inference_version": (
                "physical_cluster_hierarchical_bootstrap_randomization_v1"
            ),
            "implementation_tree_sha256": tree,
            "core_release_pipeline_sha256": release_pipeline_sha256,
            "suite_manifest_sha256": "a" * 64,
            "n_scenarios": 1,
            "scenarios": [row],
            "source_artifact": str(source_suite.resolve()),
            "source_artifact_sha256": _sha256(source_suite),
            "artifact_bindings": {
                "source_suite": {
                    **_binding(repo, source_suite),
                    "implementation_tree_sha256": tree,
                },
                "core": {
                    **_binding(repo, artifacts["materialize_core"]),
                    "implementation_tree_sha256": tree,
                },
            },
            "scenario_yaml_bindings": {
                row["scenario_id"]: _binding(repo, scenario),
            },
            "source_file_bindings": {
                row["scenario_id"]: {
                    source_asset.relative_to(repo).as_posix(): _sha256(source_asset),
                }
            },
            "formal_run_contract": LEGACY_FORMAL_RUN_CONTRACT,
        },
    )
    artifacts["readiness"] = readiness
    pipeline_manifest = pipeline / "protocol2_v21_pipeline_manifest.json"
    _write_json(
        pipeline_manifest,
        {
            "schema_version": "1.0",
            "status": "formal_evaluation_ready",
            "source_suite_sha256": _sha256(source_suite),
            "implementation_tree_sha256": tree,
            "core_release_pipeline_sha256": release_pipeline_sha256,
            "release_tooling_sha256": release_tooling_sha256,
            "stages": [
                {
                    "name": name,
                    "return_code": 0,
                    "output_sha256": _sha256(artifacts[name]),
                    "implementation_tree_sha256": tree,
                    "core_release_pipeline_sha256": release_pipeline_sha256,
                }
                for name in STAGES
            ],
        },
    )
    parent = repo / "release/operate_lineage_base.json"
    _write_json(
        parent,
        {
            "release_id": None,
            "cascade_bus_schema_version": "1.0",
            "datasets": {
                "jsplib": {
                    "url": "https://example.test",
                    "license": "terms",
                    "lock_strategy": "sha256",
                }
            },
            "backend_descriptors": {},
        },
    )
    replay_row = deepcopy(row)
    candidate_source = repo / ".hl/artifacts/fixture_candidate/source_suite.json"
    _write_json(
        candidate_source,
        {
            "schema_version": "protocol2.1-working-set-v1",
            "status": "working_set",
            "n_scenarios": 1,
            "scenarios": [replay_row],
        },
    )
    candidate_selection = (
        repo
        / ".hl/artifacts/fixture_candidate/refined_core_selection_protocol2_v21.json"
    )
    _write_json(
        candidate_selection,
        {
            "schema_version": "2.1",
            "status": "protocol21_core_candidate",
            "implementation_tree_sha256": tree,
            "core_release_pipeline_sha256": release_pipeline_sha256,
            "input_bindings": {
                "source_suite": {
                    **_binding(repo, candidate_source),
                    "implementation_tree_sha256": tree,
                }
            },
            "n_selected": 1,
            "scenarios": [replay_row],
        },
    )
    candidate_manifest = (
        repo / ".hl/artifacts/fixture_candidate/protocol2_v21_pipeline_manifest.json"
    )
    _write_json(
        candidate_manifest,
        {
            "schema_version": "1.0",
            "status": "candidate_replay_complete",
            "completed_stage": "materialize_core",
            "source_suite_sha256": _sha256(candidate_source),
            "implementation_tree_sha256": tree,
            "core_release_pipeline_sha256": release_pipeline_sha256,
            "terminal_stage_artifact": _binding(repo, candidate_selection),
        },
    )
    relocation_ledgers = []
    for domain in ("datacenter", "infrastructure", "logistics"):
        relocation = (
            repo / f"release/operate_v0_58_0_imports/{domain}/relocation_ledger.json"
        )
        identities = []
        if domain == "logistics":
            identities = [
                {
                    "scenario_id": row["scenario_id"],
                    "old": {
                        "scenario_signature": row["scenario_signature"],
                        "path": scenario_rel,
                        "yaml_sha256": _sha256(scenario),
                    },
                    "new": {
                        "scenario_signature": row["scenario_signature"],
                        "path": scenario_rel,
                        "yaml_sha256": _sha256(scenario),
                    },
                }
            ]
        bindings = {}
        if identities:
            bindings = {
                "pipeline_manifest": _binding(repo, candidate_manifest),
                "selection": _binding(repo, candidate_selection),
                "remapped_selection": _binding(repo, artifacts["materialize_core"]),
                "old_source_suite": _binding(repo, candidate_source),
                "new_source_suite": _binding(repo, source_suite),
            }
        _write_json(
            relocation,
            {
                "schema_version": "operate-canonical-relocation-v1",
                "status": "canonical_relocation_complete",
                "n_selected": len(identities),
                "identities": identities,
                "bindings": bindings,
                "empty": not identities,
                "implementation_tree_sha256": tree,
                "core_release_pipeline_sha256": release_pipeline_sha256,
            },
        )
        relocation_ledgers.append(relocation)
    candidate_terminal_ledger = repo / ".hl/artifacts/candidate_terminal_ledger.json"
    _write_json(candidate_terminal_ledger, {"status": "terminal"})
    candidate_closure = repo / "release/operate_v0_58_0_imports/candidate_closure.json"
    compact = build_compact_candidate_closure(
        {
            "schema_version": "operate-candidate-terminal-ledger-v1",
            "status": "candidate_pool_exhausted_non_admitting",
            "candidate_only": True,
            "release_admission": False,
            "summary": {
                "n_independent_candidates": 1,
                "n_terminal_candidates": 1,
                "n_unresolved_candidates": 0,
                "candidate_dispositions": {"selected_for_promotion": 1},
            },
            "inputs": {
                "candidate_terminal_ledger": _binding(repo, candidate_terminal_ledger),
                "source_suite": _binding(repo, source_suite),
            },
            "rows": [
                {
                    "candidate_id": "fixture-candidate",
                    "domain": row["domain"],
                    "source_id": "fixture-source",
                    "source_unit": "fixture-unit",
                    "classification_scope": "candidate",
                    "final_disposition": "selected_for_promotion",
                    "closure_status": "selected_for_promotion",
                    "reason_codes": ["replay:selected_for_promotion"],
                    "replay_outcome": {
                        "scenario_id": row["scenario_id"],
                        "scenario_signature": row["scenario_signature"],
                    },
                }
            ],
        },
        repo_root=repo,
        relocation_ledger_paths=relocation_ledgers,
    )
    _write_json(candidate_closure, compact)
    backend_runtime_closure = (
        repo / "release/operate_v0_58_0_imports/backend_runtime_closure.json"
    )
    archived_files = {
        "backends/fixture/demo.json": {
            "source_path": source_asset.relative_to(repo).as_posix(),
            "sha256": _sha256(source_asset),
            "roles": ["runtime_input"],
            "backend_kinds": ["dynasched_flexible_job_shop"],
        }
    }
    runtime_lock_entry = {
        "version": "1.6.2",
        "source": {"registry": "https://pypi.org/simple"},
        "artifacts": [
            {
                "kind": "wheel",
                "url": "https://example.test/simbench-1.6.2.whl",
                "hash": f"sha256:{'b' * 64}",
                "size": 123,
                "upload-time": "2026-04-02T15:04:43.984Z",
            }
        ],
    }
    runtime_lock_entry["identity_sha256"] = _canonical_sha256(runtime_lock_entry)
    runtime_lock_entries = [runtime_lock_entry]
    virtual_sources = {"pandapower-simbench://fixture@1.6.2": "c" * 64}
    runtime_closure = {
        "schema_version": "operate-backend-runtime-closure-v1",
        "release_id": "operate_v0_58_0",
        "status": "backend_runtime_closure_complete",
        "terminal": True,
        "portable": True,
        "source_suite_sha256": _sha256(source_suite),
        "archived_files": archived_files,
        "repo_tracked_files": {},
        "separately_bundled_files": {},
        "external_sources": {
            "fixture_external": {
                "delivery": "git_checkout",
                "url": "https://example.test/fixture.git",
                "revision": "0" * 40,
                "required_files": {"works/Fixture/input.json": "a" * 64},
                "metadata": {
                    "backend_kinds": ["dynasched_flexible_job_shop"],
                    "license_status": "verified_mit",
                    "redistributed": False,
                    "roles": {"works/Fixture/input.json": ["runtime_input"]},
                    "root": "works/Fixture",
                },
            }
        },
        "backend_links": {"Fixture": "fixture"},
        "runtime_packages": {
            "simbench": {
                "backend_kinds": ["cigre_distribution"],
                "lock_entries": runtime_lock_entries,
                "lock_entries_sha256": _canonical_sha256(runtime_lock_entries),
                "uv_lock_sha256": _sha256(uv_lock),
                "virtual_sources": virtual_sources,
            }
        },
        "summary": {
            "n_archived_files": 1,
            "n_external_sources": 1,
            "n_backend_links": 1,
            "n_repo_tracked_files": 0,
            "n_runtime_packages": 1,
            "n_separately_bundled_files": 0,
            "n_source_assets": 3,
            "n_unresolved": 0,
            "n_virtual_sources": 1,
        },
    }
    runtime_closure["identity_sha256"] = _runtime_identity_sha256(runtime_closure)
    _write_json(backend_runtime_closure, runtime_closure)
    return {
        "repo": repo,
        "parent": parent,
        "source_suite": source_suite,
        "pipeline": pipeline,
        "output": repo / "release/operate_v0_58_0",
        "scenario": scenario,
        "source_asset": source_asset,
        "candidate_closure": candidate_closure,
        "candidate_terminal_ledger": candidate_terminal_ledger,
        "backend_runtime_closure": backend_runtime_closure,
        "uv_lock": uv_lock,
    }


def _refresh_materialize_fixture_bindings(paths: dict[str, Path]) -> None:
    selection_path = paths["pipeline"] / STAGE_FILES["materialize_core"]
    readiness_path = paths["pipeline"] / STAGE_FILES["readiness"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["artifact_bindings"]["core"] = {
        **_binding(paths["repo"], selection_path),
        "implementation_tree_sha256": readiness["implementation_tree_sha256"],
    }
    _write_json(readiness_path, readiness)

    pipeline_manifest = paths["pipeline"] / "protocol2_v21_pipeline_manifest.json"
    pipeline = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
    for stage in pipeline["stages"]:
        if stage["name"] in {"materialize_core", "readiness"}:
            stage["output_sha256"] = _sha256(
                paths["pipeline"] / STAGE_FILES[stage["name"]]
            )
    _write_json(pipeline_manifest, pipeline)


def _bind_parent_release(
    paths: dict[str, Path], *, mutate_row: Callable[[dict], None] | None = None
) -> None:
    source = json.loads(paths["source_suite"].read_text(encoding="utf-8"))
    parent_row = deepcopy(source["scenarios"][0])
    if mutate_row is not None:
        mutate_row(parent_row)
    parent_dir = paths["repo"] / "release/operate_v0_58_0"
    parent_core = parent_dir / "core_suite.json"
    _write_json(
        parent_core,
        {
            "schema_version": "protocol21-core-v1",
            "release_id": "operate_v0_58_0",
            "status": "core_locked",
            "n_scenarios": 1,
            "scenarios": [parent_row],
        },
    )
    parent_manifest = parent_dir / "manifest.json"
    _write_json(
        parent_manifest,
        {
            "manifest_schema_version": "protocol21-core-v1",
            "release_id": "operate_v0_58_0",
            "status": "formal_evaluation_ready",
            "core_suite": {
                "path": parent_core.name,
                "sha256": _sha256(parent_core),
                "n_scenarios": 1,
            },
            "cascade_bus_schema_version": "1.0",
            "datasets": {},
            "backend_descriptors": {},
        },
    )
    paths["parent"] = parent_manifest


def _refresh_backend_runtime_closure(
    paths: dict[str, Path],
    *,
    release_id: str = "operate_v0_58_0",
) -> None:
    closure = json.loads(paths["backend_runtime_closure"].read_text(encoding="utf-8"))
    closure["release_id"] = release_id
    closure["source_suite_sha256"] = _sha256(paths["source_suite"])
    closure["identity_sha256"] = _runtime_identity_sha256(closure)
    _write_json(paths["backend_runtime_closure"], closure)


def _refresh_candidate_closure_source_binding(paths: dict[str, Path]) -> None:
    closure = json.loads(paths["candidate_closure"].read_text(encoding="utf-8"))
    closure["inputs"]["source_suite"] = _binding(paths["repo"], paths["source_suite"])
    remapped_selection = paths["pipeline"] / STAGE_FILES["materialize_core"]
    for binding in closure["relocation_ledgers"]:
        relocation_path = paths["repo"] / binding["path"]
        relocation = json.loads(relocation_path.read_text(encoding="utf-8"))
        if relocation.get("n_selected"):
            relocation["bindings"]["new_source_suite"] = _binding(
                paths["repo"], paths["source_suite"]
            )
            relocation["bindings"]["remapped_selection"] = _binding(
                paths["repo"], remapped_selection
            )
            _write_json(relocation_path, relocation)
            binding["sha256"] = _sha256(relocation_path)
    _write_json(paths["candidate_closure"], closure)


def _refresh_source_fixture_bindings(paths: dict[str, Path]) -> None:
    source_suite = paths["source_suite"]
    selection_path = paths["pipeline"] / STAGE_FILES["materialize_core"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["input_bindings"]["source_suite"] = {
        **_binding(paths["repo"], source_suite),
        "implementation_tree_sha256": selection["implementation_tree_sha256"],
    }
    _write_json(selection_path, selection)

    readiness_path = paths["pipeline"] / STAGE_FILES["readiness"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["source_artifact_sha256"] = _sha256(source_suite)
    readiness["artifact_bindings"]["source_suite"] = {
        **_binding(paths["repo"], source_suite),
        "implementation_tree_sha256": readiness["implementation_tree_sha256"],
    }
    _write_json(readiness_path, readiness)
    _refresh_materialize_fixture_bindings(paths)

    pipeline_manifest = paths["pipeline"] / "protocol2_v21_pipeline_manifest.json"
    pipeline = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
    pipeline["source_suite_sha256"] = _sha256(source_suite)
    _write_json(pipeline_manifest, pipeline)
    _refresh_candidate_closure_source_binding(paths)
    _refresh_backend_runtime_closure(paths)


def test_promoter_accepts_harness_default_observation_budget(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    scenario = yaml.safe_load(paths["scenario"].read_text(encoding="utf-8"))
    scenario["backend_config"].pop("observation_budget_chars")
    paths["scenario"].write_text(
        yaml.safe_dump(scenario, sort_keys=False),
        encoding="utf-8",
    )
    source = json.loads(paths["source_suite"].read_text(encoding="utf-8"))
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    _validate_scenario_yaml(paths["scenario"], source["scenarios"][0])


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("incomplete", "scenario_dimension_applicability_incomplete"),
        ("invalid", "scenario_dimension_applicability_invalid:.*system_survival"),
        ("reason", "scenario_dimension_reason_missing:.*system_survival"),
    ],
)
def test_promoter_rejects_invalid_dimension_applicability_contract(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    reason: str,
) -> None:
    paths = _fixture(tmp_path)
    scenario = yaml.safe_load(paths["scenario"].read_text(encoding="utf-8"))
    applicability = scenario["backend_config"]["dimension_applicability"]
    if mutation == "incomplete":
        applicability.pop("system_survival")
    elif mutation == "invalid":
        applicability["system_survival"]["applicable"] = 1
    else:
        applicability["system_survival"]["reason"] = 1
    paths["scenario"].write_text(
        yaml.safe_dump(scenario, sort_keys=False),
        encoding="utf-8",
    )
    source = json.loads(paths["source_suite"].read_text(encoding="utf-8"))
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    with pytest.raises(ValueError, match=reason):
        _validate_scenario_yaml(paths["scenario"], source["scenarios"][0])


def test_promoter_builds_hash_bound_release_and_canonicalizes_physical_key(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    summary = promote_release(
        repo_root=paths["repo"],
        parent_manifest_path=paths["parent"],
        source_suite_path=paths["source_suite"],
        candidate_closure_path=paths["candidate_closure"],
        backend_runtime_closure_path=paths["backend_runtime_closure"],
        pipeline_dir=paths["pipeline"],
        output_dir=paths["output"],
        build_public_evidence=False,
    )

    core = json.loads((paths["output"] / "core_suite.json").read_text())
    manifest = json.loads((paths["output"] / "manifest.json").read_text())
    assert summary["n_scenarios"] == 1
    assert core["n_physical_sources"] == 1
    assert core["core_settings_stamp"] == "v0.58.0-settings"
    assert core["scenarios"][0]["physical_source_key"].startswith("{")
    assert (
        manifest["backend_descriptors"]["dynasched_flexible_job_shop"][
            "released_scenarios"
        ]
        == 1
    )
    assert manifest["datasets"]["dynaschedbench"]["license"] == "Apache-2.0"
    assert manifest["core_settings_stamp"] == "v0.58.0-settings"
    assert manifest["public_release_blockers"] == [
        "formal_logical_persistent_evaluation_pending",
        "formal_realtime_persistent_evaluation_pending",
        "formal_runtime_evidence_distribution_pending",
    ]
    assert manifest["core_suite"] == {
        "n_scenarios": 1,
        "path": "core_suite.json",
        "sha256": _sha256(paths["output"] / "core_suite.json"),
    }
    assert (paths["output"] / "candidate_closure.json").read_bytes() == paths[
        "candidate_closure"
    ].read_bytes()
    compact = json.loads(paths["candidate_closure"].read_text(encoding="utf-8"))
    assert manifest["candidate_closure"] == {
        "path": "candidate_closure.json",
        "sha256": _sha256(paths["output"] / "candidate_closure.json"),
        "schema_version": "operate-candidate-closure-compact-v1",
        "status": "candidate_pool_exhausted_non_admitting",
        "n_independent_candidates": 1,
        "n_terminal_candidates": 1,
        "n_unresolved_candidates": 0,
        "identity_set_sha256": compact["identity_set_sha256"],
    }
    assert (paths["output"] / "backend_runtime_closure.json").read_bytes() == paths[
        "backend_runtime_closure"
    ].read_bytes()
    runtime_closure = json.loads(
        paths["backend_runtime_closure"].read_text(encoding="utf-8")
    )
    assert manifest["backend_runtime_closure"] == {
        "path": "backend_runtime_closure.json",
        "sha256": _sha256(paths["output"] / "backend_runtime_closure.json"),
        "schema_version": "operate-backend-runtime-closure-v1",
        "n_archived_files": 1,
        "n_external_sources": 1,
        "n_backend_links": 1,
        "n_runtime_packages": 1,
        "identity_sha256": runtime_closure["identity_sha256"],
    }
    assert manifest["formal_batch_contract"]["cardinality"] == {
        "models": {"per_shard": 1},
        "pass_k": {"minimum": 1},
        "workers": {"minimum": 1, "maximum": 32},
    }
    assert manifest["formal_batch_contract"]["selection_source"].endswith(
        "protocol2_v21_core_readiness.json#scenarios"
    )
    assert "wakeup_policy" not in manifest["formal_batch_contract"]
    assert manifest["formal_realtime_batch_contract"]["contract_version"] == (
        "realtime_persistent.v1"
    )
    assert manifest["formal_realtime_batch_contract"]["realtime_coordinator"] == (
        "realtime_episode_v4"
    )
    assert "diagnostic_readiness" not in manifest["formal_batch_contract"]
    assert "agency_readiness_bundle" not in manifest["formal_batch_contract"]

    assert set(manifest["formal_evidence"]) == {"runtime_root", "readiness"}
    assert manifest["pipeline_artifacts"]["readiness_sha256"] == _sha256(
        paths["pipeline"] / "protocol2_v21_core_readiness.json"
    )
    release_pipeline_sha256 = implementation_identity(paths["repo"])[
        "core_release_pipeline_sha256"
    ]
    release_tooling_sha256 = implementation_identity(paths["repo"])[
        "release_tooling_sha256"
    ]
    assert manifest["core_release_pipeline_sha256"] == release_pipeline_sha256
    assert manifest["release_tooling_sha256"] == release_tooling_sha256
    assert (
        manifest["pipeline_artifacts"]["core_release_pipeline_sha256"]
        == release_pipeline_sha256
    )
    assert (
        manifest["pipeline_artifacts"]["release_tooling_sha256"]
        == release_tooling_sha256
    )
    assert (
        manifest["protocol21_replay"]["core_release_pipeline_sha256"]
        == release_pipeline_sha256
    )
    assert (
        manifest["protocol21_replay"]["release_tooling_sha256"]
        == release_tooling_sha256
    )
    assert manifest["pipeline_artifacts"]["stage_artifacts"] == {
        name: {
            "relative_path": STAGE_FILES[name],
            "sha256": manifest["pipeline_artifacts"][
                {
                    "materialize_core": "core_selection",
                    "observed_reference_depth": "observed_reference_depth",
                }.get(name, name)
                + "_sha256"
            ],
        }
        for name in STAGES
    }
    monkeypatch.setattr(
        "scripts.verify_release_integrity._repo_root", lambda: paths["repo"]
    )
    integrity = build_protocol21_core_integrity_report(paths["output"])
    assert integrity["ok"] is True, integrity["issues"]


def test_v061_promotion_binds_agent_owned_wakeup_contract(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    paths["output"] = paths["repo"] / "release/operate_v0_61_0"
    readiness_path = paths["pipeline"] / STAGE_FILES["readiness"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["formal_run_contract"] = FORMAL_RUN_CONTRACT
    _write_json(readiness_path, readiness)
    _refresh_materialize_fixture_bindings(paths)
    _refresh_backend_runtime_closure(paths, release_id="operate_v0_61_0")
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    monkeypatch.setattr(
        "scripts.promote_operate_release.build_public_evidence_bundle",
        lambda **kwargs: {"binding_root_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        "scripts.verify_release_integrity.build_protocol21_core_integrity_report",
        lambda *args, **kwargs: {"ok": True, "issues": []},
    )

    with patch("scripts.promote_operate_release.RELEASE_ID", "operate_v0_61_0"):
        _promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=True,
            release_id="operate_v0_61_0",
            release_version="0.61.0",
            selection_policy="quality_core_v2_v061",
            core_settings_stamp="v0.61.0-settings",
        )

    manifest = json.loads((paths["output"] / "manifest.json").read_text())
    wakeup_policy = {
        "session_start": True,
        "typed_actionable_events": True,
        "agent_scheduled_reviews": True,
        "harness_periodic_supervisory_scan": False,
        "unknown_events_actionable": False,
    }
    assert manifest["formal_batch_contract"]["wakeup_policy"] == wakeup_policy
    realtime = manifest["formal_realtime_batch_contract"]
    assert realtime["contract_version"] == "realtime_persistent.v2"
    assert realtime["realtime_coordinator"] == "realtime_episode_v5"
    assert realtime["wakeup_policy"] == wakeup_policy
    runtime = json.loads(
        (paths["output"] / "formal_runtime_bundle.json").read_text()
    )
    assert runtime["formal_run_contract"]["wakeup_policy"] == wakeup_policy
    assert runtime["formal_run_contract"]["realtime_formal_contract"] == (
        manifest["formal_run_contract"]["realtime_formal_contract"]
    )


@pytest.mark.parametrize(
    "delivery",
    ["git_checkout", "upstream_fetch", "user_provided"],
)
def test_backend_runtime_closure_accepts_producer_schema_deliveries(
    tmp_path: Path,
    delivery: str,
) -> None:
    paths = _fixture(tmp_path)
    closure = json.loads(paths["backend_runtime_closure"].read_text(encoding="utf-8"))
    closure["external_sources"]["fixture_external"]["delivery"] = delivery
    closure["identity_sha256"] = _runtime_identity_sha256(closure)

    _validate_backend_runtime_closure(
        closure,
        release_id="operate_v0_58_0",
        source_suite_sha256=_sha256(paths["source_suite"]),
    )


@pytest.mark.parametrize("variant", ["resolution_markers", "git_source"])
def test_backend_runtime_closure_accepts_producer_lock_entry_variants(
    tmp_path: Path,
    variant: str,
) -> None:
    paths = _fixture(tmp_path)
    closure = json.loads(paths["backend_runtime_closure"].read_text(encoding="utf-8"))
    package = closure["runtime_packages"]["simbench"]
    entry = package["lock_entries"][0]
    if variant == "resolution_markers":
        entry["resolution_markers"] = ["python_full_version >= '3.14'"]
    else:
        entry["source"] = {
            "git": "https://example.test/runtime.git?rev=fixture#fixture"
        }
        entry["artifacts"] = []
    entry["identity_sha256"] = _canonical_sha256(
        {key: value for key, value in entry.items() if key != "identity_sha256"}
    )
    package["lock_entries_sha256"] = _canonical_sha256(package["lock_entries"])
    closure["identity_sha256"] = _runtime_identity_sha256(closure)

    _validate_backend_runtime_closure(
        closure,
        release_id="operate_v0_58_0",
        source_suite_sha256=_sha256(paths["source_suite"]),
    )


def test_backend_runtime_closure_accepts_file_identity_maps(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    closure = json.loads(paths["backend_runtime_closure"].read_text(encoding="utf-8"))
    identity = {
        "sha256": "e" * 64,
        "roles": ["runtime_input"],
        "backend_kinds": ["dynasched_flexible_job_shop"],
    }
    closure["repo_tracked_files"] = {"sources/tracked.json": identity}
    closure["separately_bundled_files"] = {"sources/bundled.json": identity}
    closure["summary"]["n_repo_tracked_files"] = 1
    closure["summary"]["n_separately_bundled_files"] = 1
    closure["summary"]["n_source_assets"] = 5
    closure["identity_sha256"] = _runtime_identity_sha256(closure)

    _validate_backend_runtime_closure(
        closure,
        release_id="operate_v0_58_0",
        source_suite_sha256=_sha256(paths["source_suite"]),
    )


def test_promoter_rejects_runtime_closure_bound_to_different_uv_lock(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    closure = json.loads(paths["backend_runtime_closure"].read_text(encoding="utf-8"))
    closure["runtime_packages"]["simbench"]["uv_lock_sha256"] = "d" * 64
    closure["identity_sha256"] = _runtime_identity_sha256(closure)
    _write_json(paths["backend_runtime_closure"], closure)

    with pytest.raises(ValueError, match="backend_runtime_closure_uv_lock_mismatch"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )


def test_backend_runtime_closure_requires_one_carried_uv_lock_digest(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    closure = json.loads(paths["backend_runtime_closure"].read_text(encoding="utf-8"))
    package = deepcopy(closure["runtime_packages"]["simbench"])
    package.pop("virtual_sources")
    package["uv_lock_sha256"] = "d" * 64
    closure["runtime_packages"]["pandapower"] = package
    closure["summary"]["n_runtime_packages"] = 2
    closure["identity_sha256"] = _runtime_identity_sha256(closure)

    with pytest.raises(
        ValueError,
        match="backend_runtime_closure_uv_lock_inconsistent",
    ):
        _validate_backend_runtime_closure(
            closure,
            release_id="operate_v0_58_0",
            source_suite_sha256=_sha256(paths["source_suite"]),
        )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "candidate_closure_missing"),
        ("tampered", "candidate closure identity-set hashes are invalid"),
        ("relocation_tampered", "candidate_closure_relocation_hash_mismatch"),
    ],
)
def test_promoter_rejects_missing_or_tampered_candidate_closure(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    paths = _fixture(tmp_path)
    if mutation == "missing":
        paths["candidate_closure"].unlink()
    elif mutation == "tampered":
        closure = json.loads(paths["candidate_closure"].read_text(encoding="utf-8"))
        closure["identity_set_sha256"]["all_candidates"] = "0" * 64
        _write_json(paths["candidate_closure"], closure)
    else:
        closure = json.loads(paths["candidate_closure"].read_text(encoding="utf-8"))
        relocation = paths["repo"] / closure["relocation_ledgers"][0]["path"]
        _write_json(relocation, {"tampered": True})

    with pytest.raises(ValueError, match=reason):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )


def test_promoter_rejects_tampered_candidate_closure_input(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    _write_json(paths["candidate_terminal_ledger"], {"status": "tampered"})
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    with pytest.raises(ValueError, match="candidate_closure_input_hash_mismatch"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )


def test_relocation_identity_rejects_path_outside_repo(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    closure = json.loads(paths["candidate_closure"].read_text(encoding="utf-8"))
    relocation_path = next(
        paths["repo"] / binding["path"]
        for binding in closure["relocation_ledgers"]
        if json.loads((paths["repo"] / binding["path"]).read_text())["n_selected"]
    )
    relocation = json.loads(relocation_path.read_text(encoding="utf-8"))
    relocation["identities"][0]["old"]["path"] = "../outside.yaml"
    _write_json(relocation_path, relocation)

    with pytest.raises(
        ValueError, match="candidate_closure_relocation_identity_path_invalid"
    ):
        _validated_relocation_identity_map(
            relocation_path,
            repo_root=paths["repo"],
            release_id="operate_v0_58_0",
        )


def test_relocation_identity_accepts_release_and_runtime_lock_bindings(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    closure = json.loads(paths["candidate_closure"].read_text(encoding="utf-8"))
    relocation_path = next(
        paths["repo"] / binding["path"]
        for binding in closure["relocation_ledgers"]
        if json.loads((paths["repo"] / binding["path"]).read_text())["n_selected"]
    )
    runtime_lock = (
        paths["repo"]
        / "sources/locks/operate_v0_60_0/backend_runtime_sources.json"
    )
    _write_json(
        runtime_lock,
        {
            "schema_version": "operate-backend-runtime-source-lock-v1",
            "release_id": "operate_v0_60_0",
        },
    )
    relocation = json.loads(relocation_path.read_text(encoding="utf-8"))
    relocation["bindings"].update(
        {
            "release": {
                "release_id": "operate_v0_60_0",
                "root": "release/operate_v0_60_0",
            },
            "runtime_source_lock": _binding(paths["repo"], runtime_lock),
        }
    )
    _write_json(relocation_path, relocation)

    identity_map = _validated_relocation_identity_map(
        relocation_path,
        repo_root=paths["repo"],
        release_id="operate_v0_60_0",
    )
    assert len(identity_map) == 1


def test_relocation_identity_rejects_different_promotion_release(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    closure = json.loads(paths["candidate_closure"].read_text(encoding="utf-8"))
    relocation_path = next(
        paths["repo"] / binding["path"]
        for binding in closure["relocation_ledgers"]
        if json.loads((paths["repo"] / binding["path"]).read_text())["n_selected"]
    )
    runtime_lock = (
        paths["repo"]
        / "sources/locks/operate_v0_60_0/backend_runtime_sources.json"
    )
    _write_json(
        runtime_lock,
        {
            "schema_version": "operate-backend-runtime-source-lock-v1",
            "release_id": "operate_v0_60_0",
        },
    )
    relocation = json.loads(relocation_path.read_text(encoding="utf-8"))
    relocation["bindings"].update(
        {
            "release": {
                "release_id": "operate_v0_60_0",
                "root": "release/operate_v0_60_0",
            },
            "runtime_source_lock": _binding(paths["repo"], runtime_lock),
        }
    )
    _write_json(relocation_path, relocation)

    with pytest.raises(
        ValueError,
        match="candidate_closure_relocation_release_binding_invalid",
    ):
        _validated_relocation_identity_map(
            relocation_path,
            repo_root=paths["repo"],
            release_id="operate_v0_59_0",
        )


def test_relocation_identity_rejects_self_consistent_yaml_mismatch(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    closure = json.loads(paths["candidate_closure"].read_text(encoding="utf-8"))
    relocation_path = next(
        paths["repo"] / binding["path"]
        for binding in closure["relocation_ledgers"]
        if json.loads((paths["repo"] / binding["path"]).read_text())["n_selected"]
    )
    relocation = json.loads(relocation_path.read_text(encoding="utf-8"))
    old_source_path = paths["repo"] / relocation["bindings"]["old_source_suite"]["path"]
    old_selection_path = paths["repo"] / relocation["bindings"]["selection"]["path"]
    replay_manifest_path = (
        paths["repo"] / relocation["bindings"]["pipeline_manifest"]["path"]
    )
    stale_signature = "self-consistent-but-not-yaml-bound"

    old_source = json.loads(old_source_path.read_text(encoding="utf-8"))
    old_source["scenarios"][0]["scenario_signature"] = stale_signature
    _write_json(old_source_path, old_source)
    old_selection = json.loads(old_selection_path.read_text(encoding="utf-8"))
    old_selection["scenarios"][0]["scenario_signature"] = stale_signature
    old_selection["input_bindings"]["source_suite"]["sha256"] = _sha256(old_source_path)
    _write_json(old_selection_path, old_selection)
    replay_manifest = json.loads(replay_manifest_path.read_text(encoding="utf-8"))
    replay_manifest["source_suite_sha256"] = _sha256(old_source_path)
    replay_manifest["terminal_stage_artifact"]["sha256"] = _sha256(old_selection_path)
    _write_json(replay_manifest_path, replay_manifest)
    relocation["identities"][0]["old"]["scenario_signature"] = stale_signature
    for name, path in (
        ("old_source_suite", old_source_path),
        ("selection", old_selection_path),
        ("pipeline_manifest", replay_manifest_path),
    ):
        relocation["bindings"][name]["sha256"] = _sha256(path)
    _write_json(relocation_path, relocation)

    with pytest.raises(
        ValueError, match="candidate_closure_relocation_identity_invalid"
    ):
        _validated_relocation_identity_map(
            relocation_path,
            repo_root=paths["repo"],
            release_id="operate_v0_58_0",
        )


def test_promoter_rejects_stale_candidate_closure_selection_identity(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    closure = json.loads(paths["candidate_closure"].read_text(encoding="utf-8"))
    closure["candidates"][0]["replay_identity"]["scenario_signature"] = (
        "stale-signature"
    )
    _write_json(paths["candidate_closure"], closure)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    with pytest.raises(
        ValueError, match="candidate_closure_relocation_identity_mismatch"
    ):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )


def test_promoter_rejects_relocation_canonical_value_mismatch(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    closure = json.loads(paths["candidate_closure"].read_text(encoding="utf-8"))
    closure["candidates"][0]["canonical_identity"]["scenario_signature"] = (
        "wrong-canonical-signature"
    )
    _write_json(paths["candidate_closure"], closure)

    with pytest.raises(
        ValueError,
        match="candidate_closure_relocation_identity_mismatch",
    ):
        _validate_candidate_closure_input(
            paths["candidate_closure"],
            repo_root=paths["repo"],
            release_id="operate_v0_58_0",
        )


def test_promoter_accepts_release_excluded_relocation_partition(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    closure = json.loads(paths["candidate_closure"].read_text(encoding="utf-8"))
    candidate = closure["candidates"][0]
    reason_code = "unsupported_edge_weight_semantics"
    report = {
        "schema_version": "operate-candidate-terminal-ledger-v1",
        "status": "candidate_pool_exhausted_non_admitting",
        "candidate_only": True,
        "release_admission": False,
        "summary": {
            "n_independent_candidates": 1,
            "n_terminal_candidates": 1,
            "n_unresolved_candidates": 0,
            "candidate_dispositions": {"rejected_terminal": 1},
        },
        "inputs": closure["inputs"],
        "rows": [
            {
                **{
                    key: candidate[key]
                    for key in ("candidate_id", "domain", "source_id", "source_unit")
                },
                "classification_scope": "candidate",
                "final_disposition": "rejected_terminal",
                "closure_status": "rejected_terminal",
                "disposition": "held_repair",
                "reason_codes": [f"release_import:{reason_code}"],
                "release_reconciliation": {"reason_code": reason_code},
                "replay_outcome": {
                    **candidate["replay_identity"],
                    "closure_status": "rejected_terminal",
                },
            }
        ],
    }
    relocation_paths = [
        paths["repo"] / binding["path"] for binding in closure["relocation_ledgers"]
    ]
    compact = build_compact_candidate_closure(
        report,
        repo_root=paths["repo"],
        relocation_ledger_paths=relocation_paths,
    )
    _write_json(paths["candidate_closure"], compact)

    validated, digest = _validate_candidate_closure_input(
        paths["candidate_closure"],
        repo_root=paths["repo"],
        release_id="operate_v0_58_0",
    )

    assert validated == compact
    assert digest == _sha256(paths["candidate_closure"])
    assert "canonical_identity" not in validated["candidates"][0]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "backend_runtime_closure_missing"),
        ("source_suite", "backend_runtime_closure_source_suite_mismatch"),
        ("non_terminal", "backend_runtime_closure_semantics_invalid"),
        ("unknown_field", "backend_runtime_closure_fields_invalid"),
        ("archive_path", "backend_runtime_closure_path_invalid"),
        ("archive_hash", "backend_runtime_closure_archived_file_invalid"),
        ("external_metadata", "backend_runtime_closure_external_source_invalid"),
        ("backend_link", "backend_runtime_closure_backend_link_invalid"),
        ("runtime_package", "backend_runtime_closure_runtime_package_invalid"),
        ("lock_entries_digest", "backend_runtime_closure_runtime_package_invalid"),
        ("file_identity", "backend_runtime_closure_file_identity_invalid"),
        ("file_count", "backend_runtime_closure_summary_invalid"),
        ("identity", "backend_runtime_closure_identity_invalid"),
        ("unresolved", "backend_runtime_closure_summary_invalid"),
        ("source_asset_count", "backend_runtime_closure_summary_invalid"),
    ],
)
def test_promoter_rejects_invalid_backend_runtime_closure(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    paths = _fixture(tmp_path)
    closure_path = paths["backend_runtime_closure"]
    if mutation == "missing":
        closure_path.unlink()
    else:
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        if mutation == "source_suite":
            closure["source_suite_sha256"] = "0" * 64
        elif mutation == "non_terminal":
            closure["terminal"] = False
        elif mutation == "unknown_field":
            closure["pending"] = []
        elif mutation == "archive_path":
            row = closure["archived_files"].pop("backends/fixture/demo.json")
            closure["archived_files"]["../demo.json"] = row
        elif mutation == "archive_hash":
            closure["archived_files"]["backends/fixture/demo.json"]["sha256"] = "z" * 64
        elif mutation == "external_metadata":
            closure["external_sources"]["fixture_external"]["metadata"]["roles"] = {}
        elif mutation == "backend_link":
            closure["backend_links"]["Fixture"] = "missing-root"
        elif mutation == "runtime_package":
            closure["runtime_packages"]["simbench"]["lock_entries"][0][
                "identity_sha256"
            ] = "z" * 64
        elif mutation == "lock_entries_digest":
            closure["runtime_packages"]["simbench"]["lock_entries_sha256"] = "0" * 64
        elif mutation == "file_identity":
            closure["repo_tracked_files"]["sources/invalid.json"] = {
                "sha256": "e" * 64,
                "roles": [],
                "backend_kinds": ["dynasched_flexible_job_shop"],
            }
        elif mutation == "file_count":
            closure["summary"]["n_repo_tracked_files"] = 1
        elif mutation == "identity":
            closure["identity_sha256"] = "0" * 64
        elif mutation == "unresolved":
            closure["summary"]["n_unresolved"] = 1
        else:
            closure["summary"]["n_source_assets"] = 100
        if mutation in {
            "archive_path",
            "archive_hash",
            "external_metadata",
            "backend_link",
            "runtime_package",
            "lock_entries_digest",
            "file_identity",
            "file_count",
            "source_asset_count",
        }:
            closure["identity_sha256"] = _runtime_identity_sha256(closure)
        _write_json(closure_path, closure)

    with pytest.raises(ValueError, match=reason):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=closure_path,
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )


def test_promoter_runs_live_backend_source_graph_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    def reject_incomplete_graph(**_kwargs: object) -> None:
        raise ValueError("opendss_runtime_input_mismatch:fixture")

    monkeypatch.setattr(
        "scripts.build_operate_backend_runtime_closure."
        "validate_opendss_runtime_asset_closure",
        reject_incomplete_graph,
    )

    with pytest.raises(ValueError, match="opendss_runtime_input_mismatch"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )


def test_promoter_rejects_scenario_yaml_tamper_when_public_evidence_skipped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    scenario = yaml.safe_load(paths["scenario"].read_text(encoding="utf-8"))
    scenario["candidate_only"] = True
    paths["scenario"].write_text(
        yaml.safe_dump(scenario, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    with pytest.raises(ValueError, match="readiness_scenario_yaml_binding_invalid"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )


def test_promoter_rejects_source_tamper_when_public_evidence_skipped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    _write_json(paths["source_asset"], {"jobs": ["tampered-job"]})
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    with pytest.raises(ValueError, match="readiness_source_file_binding_invalid"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )


@pytest.mark.parametrize(
    "backend_kind",
    ("alibaba_openb_gpu_placement", "sumo_ego"),
)
def test_promoter_declares_new_backend_without_parent_descriptor(
    tmp_path: Path, monkeypatch, backend_kind: str
) -> None:
    paths = _fixture(tmp_path)

    source = json.loads(paths["source_suite"].read_text(encoding="utf-8"))
    source["scenarios"][0]["backend_kind"] = backend_kind
    _write_json(paths["source_suite"], source)

    selection_path = paths["pipeline"] / STAGE_FILES["materialize_core"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["scenarios"][0]["backend_kind"] = backend_kind
    selection["input_bindings"]["source_suite"] = {
        **_binding(paths["repo"], paths["source_suite"]),
        "implementation_tree_sha256": selection["implementation_tree_sha256"],
    }
    _write_json(selection_path, selection)

    readiness_path = paths["pipeline"] / STAGE_FILES["readiness"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["scenarios"][0]["backend_kind"] = backend_kind
    readiness["source_artifact_sha256"] = _sha256(paths["source_suite"])
    readiness["artifact_bindings"]["source_suite"] = {
        **_binding(paths["repo"], paths["source_suite"]),
        "implementation_tree_sha256": readiness["implementation_tree_sha256"],
    }
    _write_json(readiness_path, readiness)
    _refresh_materialize_fixture_bindings(paths)

    pipeline_path = paths["pipeline"] / "protocol2_v21_pipeline_manifest.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline["source_suite_sha256"] = _sha256(paths["source_suite"])
    _write_json(pipeline_path, pipeline)
    _refresh_candidate_closure_source_binding(paths)
    _refresh_backend_runtime_closure(paths)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    promote_release(
        repo_root=paths["repo"],
        parent_manifest_path=paths["parent"],
        source_suite_path=paths["source_suite"],
        candidate_closure_path=paths["candidate_closure"],
        backend_runtime_closure_path=paths["backend_runtime_closure"],
        pipeline_dir=paths["pipeline"],
        output_dir=paths["output"],
        build_public_evidence=False,
    )

    manifest = json.loads((paths["output"] / "manifest.json").read_text())
    descriptor = manifest["backend_descriptors"][backend_kind]
    assert descriptor["released_scenarios"] == 1
    assert descriptor["formal_core_allowed"] is True
    if backend_kind == "sumo_ego":
        assert descriptor["runtime_fidelity"] == "native_live_sumo_reactive"
        dataset = manifest["datasets"]["ngsim_us101"]
        assert dataset["source_release"] == "doi:10.21949/1504477"
        assert dataset["recording_id"] == "us-101"
        assert dataset["license"] == (
            "CC-BY-SA-3.0 dataset API metadata; CC-BY-SA-4.0 Common Core "
            "metadata (operator-reviewed)"
        )
        assert dataset["lock_strategy"] == (
            "doi+canonical_query_or_archive+raw_sha256+row_semantic_sha256"
        )
        return
    dataset = manifest["datasets"]["alibaba_cluster_trace_gpu_v2023_openb"]
    assert dataset["commit"] == "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71"
    assert dataset["license"] == (
        "research trace terms; upstream repository license applies"
    )
    assert dataset["lock_strategy"] == (
        "upstream_git_commit_raw_sha256_and_explicit_row_graph"
    )
    assert dataset["required_file_sha256s"] == {
        "works/clusterdata/cluster-trace-gpu-v2023/csv/openb_node_list_gpu_node.csv": (
            "2beca64b4d3dfa342036a34b56a495c6cef9225db836c81f541282cb1df320b5"
        ),
        "works/clusterdata/cluster-trace-gpu-v2023/csv/openb_pod_list_gpuspec33.csv": (
            "eca4f746db1e5b25864ad021b55ece3943e101a3ebd4574d09dcb95c46117652"
        ),
    }
    monkeypatch.setattr(
        "scripts.verify_release_integrity._repo_root", lambda: paths["repo"]
    )
    integrity = build_protocol21_core_integrity_report(paths["output"])
    assert integrity["ok"] is True, integrity["issues"]


@pytest.mark.parametrize(
    "disposition",
    ("held_repair", "candidate_pending_full_protocol21"),
)
def test_promoter_rejects_unresolved_or_unknown_readiness_subset(
    tmp_path: Path, monkeypatch, disposition: str
) -> None:
    paths = _fixture(tmp_path)
    output = paths["repo"] / "release/operate_v0_58_0"
    source = json.loads(paths["source_suite"].read_text(encoding="utf-8"))
    rejected = dict(source["scenarios"][0])
    rejected.update(
        {
            "scenario_id": "logistics/job_shop_dispatch/time_pressure/high/rejected",
            "scenario_signature": "rejected-signature",
            "path": "scenarios/staging/v0_55/rejected.yaml",
            "structural_fingerprint": "rejected-structure",
            "status": disposition,
            "core_disposition": disposition,
        }
    )
    source["n_scenarios"] = 2
    source["scenarios"].append(rejected)
    _write_json(paths["source_suite"], source)

    selection_path = paths["pipeline"] / STAGE_FILES["materialize_core"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection.update(
        {
            "n_source": 2,
            "n_rejected": 1,
            "rejected": [
                {
                    "scenario_id": rejected["scenario_id"],
                    "scenario_signature": rejected["scenario_signature"],
                    "disposition": disposition,
                }
            ],
            "disposition_counts": {"core_locked": 1, disposition: 1},
        }
    )
    selection["input_bindings"]["source_suite"] = {
        **_binding(paths["repo"], paths["source_suite"]),
        "implementation_tree_sha256": selection["implementation_tree_sha256"],
    }
    _write_json(selection_path, selection)

    readiness_path = paths["pipeline"] / STAGE_FILES["readiness"]
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    readiness["source_artifact_sha256"] = _sha256(paths["source_suite"])
    readiness["artifact_bindings"]["source_suite"] = {
        **_binding(paths["repo"], paths["source_suite"]),
        "implementation_tree_sha256": readiness["implementation_tree_sha256"],
    }
    readiness["artifact_bindings"]["core"] = {
        **_binding(paths["repo"], selection_path),
        "implementation_tree_sha256": readiness["implementation_tree_sha256"],
    }
    _write_json(readiness_path, readiness)

    pipeline_manifest = paths["pipeline"] / "protocol2_v21_pipeline_manifest.json"
    pipeline = json.loads(pipeline_manifest.read_text(encoding="utf-8"))
    pipeline["source_suite_sha256"] = _sha256(paths["source_suite"])
    for stage in pipeline["stages"]:
        if stage["name"] in {"materialize_core", "readiness"}:
            stage["output_sha256"] = _sha256(
                paths["pipeline"] / STAGE_FILES[stage["name"]]
            )
    _write_json(pipeline_manifest, pipeline)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    with pytest.raises(
        ValueError,
        match=f"materialize_rejected_disposition_invalid:{disposition}",
    ):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=output,
            build_public_evidence=False,
            release_id="operate_v0_58_0",
            release_version="0.58.0",
            selection_policy="quality_core_v2_v058",
            core_settings_stamp="v0.58.0-settings",
        )

    assert not (output / "manifest.json").exists()


def test_promoter_rejects_formal_source_suite_identity_subset(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    source = json.loads(paths["source_suite"].read_text(encoding="utf-8"))
    omitted = deepcopy(source["scenarios"][0])
    omitted.update(
        {
            "scenario_id": "logistics/job_shop_dispatch/time_pressure/high/omitted",
            "scenario_signature": "omitted-signature",
            "path": "scenarios/staging/v0_55/omitted.yaml",
            "structural_fingerprint": "omitted-structure",
        }
    )
    omitted.pop("historical_admission")
    source["scenarios"].append(omitted)
    source["n_scenarios"] = 2
    source["candidate_import_partition"]["n_base"] = 1
    source["candidate_import_partition"]["base_identities"] = [
        {
            "scenario_id": omitted["scenario_id"],
            "scenario_signature": omitted["scenario_signature"],
        }
    ]
    _write_json(paths["source_suite"], source)

    selection_path = paths["pipeline"] / STAGE_FILES["materialize_core"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection.update(
        {
            "n_source": 2,
            "n_rejected": 1,
            "rejected": [
                {
                    "scenario_id": omitted["scenario_id"],
                    "scenario_signature": omitted["scenario_signature"],
                    "disposition": "retired_intrinsic",
                }
            ],
            "disposition_counts": {"core_locked": 1, "retired_intrinsic": 1},
        }
    )
    _write_json(selection_path, selection)
    _refresh_source_fixture_bindings(paths)

    with pytest.raises(ValueError, match="readiness_source_identity_mismatch"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )

    assert not (paths["output"] / "manifest.json").exists()


def test_promoter_rejects_unresolved_top_level_candidate_inventory(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    source = json.loads(paths["source_suite"].read_text(encoding="utf-8"))
    source["held_candidates"] = [
        {
            "scenario_id": "microgrid/legacy-boston-relabel",
            "scenario_signature": "legacy-signature",
            "disposition": "held_repair",
            "included": False,
        }
    ]
    _write_json(paths["source_suite"], source)

    with pytest.raises(
        ValueError,
        match="source_suite_unresolved_candidate_inventory:held_candidates",
    ):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )

    assert not (paths["output"] / "manifest.json").exists()


@pytest.mark.parametrize(
    ("candidate_inventory", "reason"),
    [
        (
            {
                "abandoned_candidates": [
                    {
                        "scenario_id": "microgrid/legacy-boston-relabel",
                        "scenario_signature": "legacy-signature",
                        "disposition": "held_repair",
                        "included": False,
                        "reason_codes": ["still_unresolved"],
                    }
                ]
            },
            "source_suite_abandoned_candidate_invalid:0:disposition",
        ),
        (
            {"abandoned_candidates": [{"disposition": "abandoned_terminal"}]},
            "source_suite_abandoned_candidate_invalid:0:identity",
        ),
        (
            {"shadow_candidates": []},
            "source_suite_candidate_inventory_unsupported:shadow_candidates",
        ),
    ],
)
def test_promoter_rejects_invalid_terminal_candidate_inventory(
    tmp_path: Path,
    candidate_inventory: dict,
    reason: str,
) -> None:
    paths = _fixture(tmp_path)
    source = json.loads(paths["source_suite"].read_text(encoding="utf-8"))
    source.update(candidate_inventory)
    _write_json(paths["source_suite"], source)

    with pytest.raises(ValueError, match=reason):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )


def test_promoter_accepts_typed_terminal_candidate_inventory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    source = json.loads(paths["source_suite"].read_text(encoding="utf-8"))
    source["abandoned_candidates"] = [
        {
            "scenario_id": "microgrid/legacy-boston-relabel",
            "scenario_signature": "legacy-signature",
            "disposition": "abandoned_terminal",
            "included": False,
            "reason_codes": ["label_only_difficulty_relabel"],
        }
    ]
    _write_json(paths["source_suite"], source)
    _refresh_source_fixture_bindings(paths)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    promote_release(
        repo_root=paths["repo"],
        parent_manifest_path=paths["parent"],
        source_suite_path=paths["source_suite"],
        candidate_closure_path=paths["candidate_closure"],
        backend_runtime_closure_path=paths["backend_runtime_closure"],
        pipeline_dir=paths["pipeline"],
        output_dir=paths["output"],
        build_public_evidence=False,
    )

    published = json.loads(
        (paths["output"] / "protocol21_source_suite.json").read_text()
    )
    assert published["abandoned_candidates"] == source["abandoned_candidates"]


def test_promoter_rejects_stale_materialize_source_binding(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    selection_path = paths["pipeline"] / STAGE_FILES["materialize_core"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["input_bindings"]["source_suite"]["sha256"] = "0" * 64
    _write_json(selection_path, selection)
    _refresh_materialize_fixture_bindings(paths)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    with pytest.raises(ValueError, match="materialize_source_binding_invalid"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )


def test_promoter_rejects_readiness_not_equal_to_materialize_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    selection_path = paths["pipeline"] / STAGE_FILES["materialize_core"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["scenarios"].pop()
    selection.update(
        {
            "n_selected": 0,
            "n_rejected": 1,
            "rejected": [
                {
                    "scenario_id": selected["scenario_id"],
                    "scenario_signature": selected["scenario_signature"],
                    "disposition": "held_repair",
                }
            ],
            "disposition_counts": {"held_repair": 1},
        }
    )
    _write_json(selection_path, selection)
    _refresh_materialize_fixture_bindings(paths)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    with pytest.raises(ValueError, match="readiness_core_selection_mismatch"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )


def test_promoter_parameterizes_new_release_identity(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    _bind_parent_release(paths)
    output = paths["repo"] / "release/operate_v0_59_0"
    output.mkdir(parents=True)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    _refresh_backend_runtime_closure(paths, release_id="operate_v0_59_0")

    promote_release(
        repo_root=paths["repo"],
        parent_manifest_path=paths["parent"],
        source_suite_path=paths["source_suite"],
        candidate_closure_path=paths["candidate_closure"],
        backend_runtime_closure_path=paths["backend_runtime_closure"],
        pipeline_dir=paths["pipeline"],
        output_dir=output,
        build_public_evidence=True,
        release_id="operate_v0_59_0",
        release_version="0.59.0",
        selection_policy="quality_core_v2_v059",
        core_settings_stamp="v0.59.0-settings",
    )

    manifest = json.loads((output / "manifest.json").read_text())
    core = json.loads((output / "core_suite.json").read_text())
    assert manifest["release_id"] == "operate_v0_59_0"
    assert manifest["core_selection_policy"] == "quality_core_v2_v059"
    assert core["release_id"] == "operate_v0_59_0"
    assert core["core_settings_stamp"] == "v0.59.0-settings"
    runtime_bundle = json.loads((output / "formal_runtime_bundle.json").read_text())
    assert runtime_bundle["release_id"] == "operate_v0_59_0"
    assert runtime_bundle["n_scenarios"] == 1
    assert runtime_bundle["scenarios"][0]["case_ledger"]
    assert manifest["formal_batch_contract"]["runtime_evidence_root"] == (
        "release/operate_v0_59_0"
    )
    assert manifest["formal_batch_contract"]["selection_source"] == (
        "release/operate_v0_59_0/formal_runtime_bundle.json#scenarios"
    )
    assert manifest["formal_runtime_bundle"]["sha256"] == _sha256(
        output / "formal_runtime_bundle.json"
    )
    from scripts.batch_llm_eval import resolve_formal_manifest_slice

    monkeypatch.setattr(
        "scripts.batch_llm_eval.validate_live_backend_runtime_closure",
        lambda **_kwargs: {
            "identity_sha256": manifest["backend_runtime_closure"][
                "identity_sha256"
            ]
        },
    )
    binding = resolve_formal_manifest_slice(
        output / "manifest.json", repo_root=paths["repo"]
    )
    assert binding["dynamic_slice_spec"] == (
        "operate_v0_59_0",
        "formal_runtime_bundle.json",
        {},
    )
    assert binding["formal_runtime_bundle_sha256"] == manifest[
        "formal_runtime_bundle"
    ]["sha256"]
    assert (output / "protocol21_source_suite.json").read_bytes() == paths[
        "source_suite"
    ].read_bytes()

    runtime_bundle_path = output / "formal_runtime_bundle.json"
    runtime_bundle_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="formal manifest readiness hash mismatch"):
        resolve_formal_manifest_slice(output / "manifest.json", repo_root=paths["repo"])


def test_promoter_rejects_new_release_without_parent_release(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    output = paths["repo"] / "release/operate_v0_59_0"
    output.mkdir(parents=True)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    _refresh_backend_runtime_closure(paths, release_id="operate_v0_59_0")

    with pytest.raises(ValueError, match="parent_release_identity_invalid"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=output,
            build_public_evidence=False,
            release_id="operate_v0_59_0",
            release_version="0.59.0",
            selection_policy="quality_core_v2_v059",
            core_settings_stamp="v0.59.0-settings",
        )


@pytest.mark.parametrize("case", [
    "valid", "missing", "cross_source", "cross_physical_source", "duplicate",
    "unknown_parent", "missing_reason", "missing_replacement", "retained_parent",
])
def test_parent_core_explicit_refinement(tmp_path: Path, case: str) -> None:
    paths = _fixture(tmp_path)
    _bind_parent_release(paths)
    parent = json.loads(paths["parent"].read_text())
    row = json.loads(paths["source_suite"].read_text())["scenarios"][0]
    replacement = {**row, "scenario_id": "refined", "scenario_signature": "new-signature"}
    entry = {
        "parent": {key: row[key] for key in ("scenario_id", "scenario_signature")},
        "replacement": {key: replacement[key] for key in ("scenario_id", "scenario_signature")},
        "reason": "Expose the seeded disruption before the reference completes.",
    }
    if case == "cross_source":
        replacement["source_denominator_key"] = "different-source"
    if case == "cross_physical_source":
        replacement["physical_source_key"] = "different-source"
    if case == "unknown_parent":
        entry["parent"]["scenario_id"] = "unknown"
    if case == "missing_reason":
        entry.pop("reason")
    if case == "missing_replacement":
        entry["replacement"]["scenario_id"] = "missing"
    refinements = [] if case == "missing" else [entry]
    if case == "duplicate":
        refinements.append(deepcopy(entry))
    kwargs = dict(
        parent_manifest_path=paths["parent"], parent=parent,
        source_rows=[replacement, row] if case == "retained_parent" else [replacement],
        release_id="operate_v0_59_0",
        core_refinements=refinements,
    )
    if case == "valid":
        _validate_parent_core_ancestry(**kwargs)
    else:
        with pytest.raises(ValueError, match="parent_core_"):
            _validate_parent_core_ancestry(**kwargs)


def test_promoter_rejects_parent_core_ancestry_drift(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    _bind_parent_release(
        paths,
        mutate_row=lambda row: row.update({"backend_kind": "different_backend"}),
    )
    output = paths["repo"] / "release/operate_v0_59_0"
    output.mkdir(parents=True)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    _refresh_backend_runtime_closure(paths, release_id="operate_v0_59_0")

    with pytest.raises(ValueError, match="parent_core_ancestry_mismatch"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=output,
            build_public_evidence=False,
            release_id="operate_v0_59_0",
            release_version="0.59.0",
            selection_policy="quality_core_v2_v059",
            core_settings_stamp="v0.59.0-settings",
        )


def test_promoter_allows_parent_core_path_relocation(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    _bind_parent_release(
        paths,
        mutate_row=lambda row: row.update(
            {"path": "scenarios/operate_v0_58_0/logistics/demo.yaml"}
        ),
    )
    output = paths["repo"] / "release/operate_v0_59_0"
    output.mkdir(parents=True)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    _refresh_backend_runtime_closure(paths, release_id="operate_v0_59_0")

    promote_release(
        repo_root=paths["repo"],
        parent_manifest_path=paths["parent"],
        source_suite_path=paths["source_suite"],
        candidate_closure_path=paths["candidate_closure"],
        backend_runtime_closure_path=paths["backend_runtime_closure"],
        pipeline_dir=paths["pipeline"],
        output_dir=output,
        build_public_evidence=True,
        release_id="operate_v0_59_0",
        release_version="0.59.0",
        selection_policy="quality_core_v2_v059",
        core_settings_stamp="v0.59.0-settings",
    )

    core = json.loads((output / "core_suite.json").read_text())
    assert core["scenarios"][0]["path"] == (
        "scenarios/staging/v0_55/demo.yaml"
    )


@pytest.mark.parametrize("existing_name", ("manifest.json", "unknown.txt"))
def test_promoter_transactionally_replaces_existing_release(
    tmp_path: Path, monkeypatch, existing_name: str
) -> None:
    paths = _fixture(tmp_path)
    existing_path = paths["output"] / existing_name
    existing_bytes = b"existing release bytes\n"
    existing_path.write_bytes(existing_bytes)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    promote_release(
        repo_root=paths["repo"],
        parent_manifest_path=paths["parent"],
        source_suite_path=paths["source_suite"],
        candidate_closure_path=paths["candidate_closure"],
        backend_runtime_closure_path=paths["backend_runtime_closure"],
        pipeline_dir=paths["pipeline"],
        output_dir=paths["output"],
        build_public_evidence=False,
    )

    assert (paths["output"] / "manifest.json").is_file()
    assert (paths["output"] / "core_suite.json").is_file()
    assert (paths["output"] / "README.md").is_file()
    if existing_name == "unknown.txt":
        assert not existing_path.exists()
    else:
        assert existing_path.read_bytes() != existing_bytes


def test_promoter_late_staging_failure_preserves_existing_release(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    sentinel = paths["output"] / "existing.txt"
    sentinel.write_bytes(b"published release\n")
    before = {
        path.relative_to(paths["output"]).as_posix(): path.read_bytes()
        for path in paths["output"].rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    def _fail_staged_verification(*args, **kwargs):
        raise ValueError("late_stage_failure")

    monkeypatch.setattr(
        "scripts.promote_operate_release._verify_staged_release",
        _fail_staged_verification,
    )

    with pytest.raises(ValueError, match="late_stage_failure"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )

    after = {
        path.relative_to(paths["output"]).as_posix(): path.read_bytes()
        for path in paths["output"].rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not list(paths["output"].parent.glob(".operate_v0_58_0.promotion.*"))


def test_promoter_rejects_staged_backend_runtime_closure_tamper(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    from scripts import promote_operate_release as promoter

    real_verify = promoter._verify_staged_release

    def _tamper_then_verify(release: Path, **kwargs) -> None:
        closure_path = release / "backend_runtime_closure.json"
        closure = json.loads(closure_path.read_text(encoding="utf-8"))
        closure["portable"] = False
        _write_json(closure_path, closure)
        real_verify(release, **kwargs)

    monkeypatch.setattr(promoter, "_verify_staged_release", _tamper_then_verify)

    with pytest.raises(ValueError, match="backend_runtime_closure_semantics_invalid"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )

    assert not (paths["output"] / "manifest.json").exists()


def test_promoter_revalidates_pipeline_bindings_inside_lock(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    sentinel = paths["output"] / "existing.txt"
    sentinel.write_bytes(b"published release\n")
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    from scripts import promote_operate_release as promoter

    real_verify = promoter._verify_staged_release

    def _verify_then_change_pipeline(*args, **kwargs):
        real_verify(*args, **kwargs)
        artifact = paths["pipeline"] / STAGE_FILES["behavioral"]
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["concurrent_change"] = True
        _write_json(artifact, payload)

    monkeypatch.setattr(
        promoter,
        "_verify_staged_release",
        _verify_then_change_pipeline,
    )

    with pytest.raises(ValueError, match="pipeline_stage_hash_mismatch:behavioral"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )

    assert sentinel.read_bytes() == b"published release\n"
    assert not (paths["output"] / "manifest.json").exists()
    assert not list(paths["output"].parent.glob(".operate_v0_58_0.promotion.*"))


def test_promoter_revalidates_backend_runtime_closure_inside_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = _fixture(tmp_path)
    sentinel = paths["output"] / "existing.txt"
    sentinel.write_bytes(b"published release\n")
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    from scripts import promote_operate_release as promoter

    real_verify = promoter._verify_staged_release

    def _verify_then_change_runtime_closure(*args, **kwargs) -> None:
        real_verify(*args, **kwargs)
        closure_path = paths["backend_runtime_closure"]
        closure_path.write_text(
            closure_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        promoter,
        "_verify_staged_release",
        _verify_then_change_runtime_closure,
    )

    with pytest.raises(ValueError, match="promotion_inputs_changed_during_promotion"):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )

    assert sentinel.read_bytes() == b"published release\n"
    assert not (paths["output"] / "manifest.json").exists()


def test_promoter_post_swap_verification_failure_restores_existing_release(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    sentinel = paths["output"] / "existing.txt"
    sentinel.write_bytes(b"published release\n")
    before = {
        path.relative_to(paths["output"]).as_posix(): path.read_bytes()
        for path in paths["output"].rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    monkeypatch.setattr(
        "scripts.verify_release_integrity.build_protocol21_core_integrity_report",
        lambda *args, **kwargs: {
            "ok": False,
            "issues": [{"code": "forced_post_swap_failure"}],
        },
    )

    with pytest.raises(
        ValueError,
        match="published_release_integrity_failed:forced_post_swap_failure",
    ):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )

    after = {
        path.relative_to(paths["output"]).as_posix(): path.read_bytes()
        for path in paths["output"].rglob("*")
        if path.is_file()
    }
    assert after == before
    assert not list(paths["output"].parent.glob(".operate_v0_58_0.promotion.*"))


def test_promoter_rolls_back_when_backup_differs_from_locked_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    sentinel = paths["output"] / "existing.txt"
    sentinel.write_bytes(b"published release\n")
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    from scripts import promote_operate_release as promoter

    real_snapshot = promoter._directory_snapshot
    output_snapshots = 0

    def _snapshot_then_inject_unknown_file(path: Path):
        nonlocal output_snapshots
        snapshot = real_snapshot(path)
        if Path(path).resolve() == paths["output"].resolve():
            output_snapshots += 1
            if output_snapshots == 2:
                (paths["output"] / "concurrent.txt").write_bytes(
                    b"concurrent writer bytes\n"
                )
        return snapshot

    monkeypatch.setattr(
        promoter, "_directory_snapshot", _snapshot_then_inject_unknown_file
    )

    with pytest.raises(
        ValueError,
        match="release_backup_changed_during_promotion",
    ):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )

    assert sentinel.read_bytes() == b"published release\n"
    assert (paths["output"] / "concurrent.txt").read_bytes() == (
        b"concurrent writer bytes\n"
    )
    assert not (paths["output"] / "manifest.json").exists()
    assert not list(paths["output"].parent.glob(".operate_v0_58_0.promotion.*"))


def test_promoter_keeps_verified_release_when_backup_cleanup_fails(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    (paths["output"] / "existing.txt").write_bytes(b"published release\n")
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    real_rmtree = __import__("shutil").rmtree

    def _fail_backup_cleanup(path, *args, **kwargs):
        if Path(path).name.endswith(".previous"):
            raise OSError("forced_backup_cleanup_failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(
        "scripts.promote_operate_release.shutil.rmtree",
        _fail_backup_cleanup,
    )

    promote_release(
        repo_root=paths["repo"],
        parent_manifest_path=paths["parent"],
        source_suite_path=paths["source_suite"],
        candidate_closure_path=paths["candidate_closure"],
        backend_runtime_closure_path=paths["backend_runtime_closure"],
        pipeline_dir=paths["pipeline"],
        output_dir=paths["output"],
        build_public_evidence=False,
    )

    assert (paths["output"] / "manifest.json").is_file()
    assert not (paths["output"] / "existing.txt").exists()
    assert list(paths["output"].parent.glob(".operate_v0_58_0.promotion.*"))


def test_promoter_rejects_non_green_or_tree_drift(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    readiness = paths["pipeline"] / "protocol2_v21_core_readiness.json"
    payload = json.loads(readiness.read_text())
    payload["formal_evaluation_ready"] = False
    _write_json(readiness, payload)

    try:
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )
        raise AssertionError("non-green readiness must fail closed")
    except ValueError as exc:
        assert "readiness_not_green" in str(exc)


@pytest.mark.parametrize(
    ("location", "reason"),
    [
        ("readiness", "readiness_core_release_pipeline_drift"),
        ("pipeline", "pipeline_core_release_pipeline_drift"),
        ("release_tooling", "pipeline_release_tooling_drift"),
        ("stage", "pipeline_stage_toolchain_mismatch:behavioral"),
        ("artifact", "pipeline_artifact_toolchain_mismatch:behavioral"),
    ],
)
def test_promoter_rejects_missing_or_stale_release_pipeline_identity(
    tmp_path: Path,
    location: str,
    reason: str,
) -> None:
    paths = _fixture(tmp_path)
    pipeline_manifest_path = paths["pipeline"] / "protocol2_v21_pipeline_manifest.json"
    if location == "readiness":
        target = paths["pipeline"] / STAGE_FILES["readiness"]
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload.pop("core_release_pipeline_sha256")
        _write_json(target, payload)
    elif location == "pipeline":
        payload = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
        payload["core_release_pipeline_sha256"] = "0" * 64
        _write_json(pipeline_manifest_path, payload)
    elif location == "release_tooling":
        payload = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
        payload["release_tooling_sha256"] = "0" * 64
        _write_json(pipeline_manifest_path, payload)
    elif location == "stage":
        payload = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
        next(stage for stage in payload["stages"] if stage["name"] == "behavioral")[
            "core_release_pipeline_sha256"
        ] = "0" * 64
        _write_json(pipeline_manifest_path, payload)
    else:
        target = paths["pipeline"] / STAGE_FILES["behavioral"]
        artifact = json.loads(target.read_text(encoding="utf-8"))
        artifact["core_release_pipeline_sha256"] = "0" * 64
        _write_json(target, artifact)
        payload = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
        next(stage for stage in payload["stages"] if stage["name"] == "behavioral")[
            "output_sha256"
        ] = _sha256(target)
        _write_json(pipeline_manifest_path, payload)

    with pytest.raises(ValueError, match=reason):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )


@pytest.mark.parametrize(
    "location",
    ("manifest", "pipeline_artifacts", "protocol21_replay"),
)
def test_full_integrity_rejects_release_pipeline_identity_drift(
    tmp_path: Path,
    monkeypatch,
    location: str,
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )
    promote_release(
        repo_root=paths["repo"],
        parent_manifest_path=paths["parent"],
        source_suite_path=paths["source_suite"],
        candidate_closure_path=paths["candidate_closure"],
        backend_runtime_closure_path=paths["backend_runtime_closure"],
        pipeline_dir=paths["pipeline"],
        output_dir=paths["output"],
        build_public_evidence=False,
    )
    manifest_path = paths["output"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = manifest if location == "manifest" else manifest[location]
    target["core_release_pipeline_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    monkeypatch.setattr(
        "scripts.verify_release_integrity._repo_root", lambda: paths["repo"]
    )

    integrity = build_protocol21_core_integrity_report(paths["output"])

    assert integrity["ok"] is False
    assert integrity["checks"]["agentic_release_pipeline_binding_valid"] is False


@pytest.mark.parametrize(
    ("formal_run_contract", "reason"),
    [
        (None, "formal_run_contract_missing"),
        (
            {"contract_version": "legacy_stateless.v1"},
            "formal_run_contract_version_unsupported",
        ),
    ],
)
def test_promoter_rejects_missing_or_legacy_formal_contract(
    tmp_path: Path,
    formal_run_contract: dict | None,
    reason: str,
) -> None:
    paths = _fixture(tmp_path)
    readiness = paths["pipeline"] / "protocol2_v21_core_readiness.json"
    payload = json.loads(readiness.read_text(encoding="utf-8"))
    payload["formal_run_contract"] = formal_run_contract
    _write_json(readiness, payload)

    with pytest.raises(ValueError, match=reason):
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=False,
        )

    assert not (paths["output"] / "manifest.json").exists()


def test_promoter_does_not_materialize_release_when_public_evidence_fails(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path)
    monkeypatch.setattr(
        "scripts.promote_operate_release.verify_scenario_row_against_yaml",
        lambda row, path: [],
    )

    def _fail_public_evidence(**kwargs):
        raise ValueError("public_evidence_invalid")

    monkeypatch.setattr(
        "scripts.promote_operate_release.build_public_evidence_bundle",
        _fail_public_evidence,
    )
    try:
        promote_release(
            repo_root=paths["repo"],
            parent_manifest_path=paths["parent"],
            source_suite_path=paths["source_suite"],
            candidate_closure_path=paths["candidate_closure"],
            backend_runtime_closure_path=paths["backend_runtime_closure"],
            pipeline_dir=paths["pipeline"],
            output_dir=paths["output"],
            build_public_evidence=True,
        )
        raise AssertionError("invalid public evidence must fail closed")
    except ValueError as exc:
        assert "public_evidence_invalid" in str(exc)
    assert not (paths["output"] / "core_suite.json").exists()
    assert not (paths["output"] / "manifest.json").exists()
