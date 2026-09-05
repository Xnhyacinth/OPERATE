from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import yaml

from baselines.llm_agent import prompt_contract_sha256
from core.implementation_identity import implementation_identity
from evaluation.dimension_applicability import (
    DIMENSION_APPLICABILITY_DIMENSIONS,
    dimension_applicability_contract_issue,
)
from evaluation.scorer import DISCRIMINATIVE_CORE_DIMENSIONS
from evaluation.realtime_diagnostics import SCHEMA_VERSION as REALTIME_DIAGNOSTIC_SCHEMA
from scripts.verify_release_integrity import (
    AGENTIC_PROFILE_V1,
    PIPELINE_STAGE_FILES,
    PIPELINE_STAGE_HASH_FIELDS,
    REALTIME_FORMAL_CONTRACT_V1,
    REALTIME_FORMAL_CONTRACT_V2,
    _agentic_formal_checks,
    _canonical_payload_sha256,
    _canonical_sha256,
    _formal_tool_choice_matches,
    _formal_publication_checks,
    _formal_contracts_for_release,
    _formal_wakeup_policy_valid,
    _logical_rows_for_validation,
    _portable_evidence_closure_valid,
    build_release_integrity_report,
)
from scripts import verify_release_integrity as verify_mod
from runner import EVALUATION_IMPLEMENTATION_FINGERPRINT


REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = verify_mod.DEFAULT_RELEASE


def test_dimension_applicability_contract_is_canonical_and_fail_closed() -> None:
    expected = frozenset(DISCRIMINATIVE_CORE_DIMENSIONS) - {"task_completion"}
    valid = {
        name: {"applicable": True, "reason": "measured_by_native_evidence"}
        for name in expected
    }

    assert len(DIMENSION_APPLICABILITY_DIMENSIONS) == 13
    assert DIMENSION_APPLICABILITY_DIMENSIONS == expected
    assert dimension_applicability_contract_issue(valid) is None
    assert dimension_applicability_contract_issue({}) == ("incomplete", None)

    invalid = deepcopy(valid)
    invalid["system_survival"]["applicable"] = 1
    assert dimension_applicability_contract_issue(invalid) == (
        "invalid",
        "system_survival",
    )

    missing_reason = deepcopy(valid)
    missing_reason["system_survival"]["reason"] = 1
    assert dimension_applicability_contract_issue(missing_reason) == (
        "reason_missing",
        "system_survival",
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_result_tree(
    release: Path,
    *,
    model: str,
    treatment: str,
    payload: dict,
    files: dict[str, str],
) -> tuple[dict, Path]:
    staging = release / "tree-staging"
    staging.mkdir(parents=True)
    _write_json(staging / "RUN_MANIFEST.json", payload)
    for relative, contents in files.items():
        path = staging / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    indexed = []
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            indexed.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    index = {
        "schema_version": "operate-formal-result-tree-index-v1",
        "files": indexed,
    }
    index["root_sha256"] = _canonical_sha256(index)
    root = release / "formal_results" / model / treatment / index["root_sha256"]
    root.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(root)
    index_path = root / "FORMAL_RESULT_TREE_INDEX.json"
    _write_json(index_path, index)
    return index, root / "RUN_MANIFEST.json"


def _refresh_formal_result_binding(repo: Path, binding: dict) -> None:
    root = (repo / binding["path"]).parent
    index_path = root / "FORMAL_RESULT_TREE_INDEX.json"
    indexed = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != index_path:
            indexed.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    index = {
        "schema_version": "operate-formal-result-tree-index-v1",
        "files": indexed,
    }
    index["root_sha256"] = _canonical_sha256(index)
    destination = root.with_name(index["root_sha256"])
    root.rename(destination)
    index_path = destination / "FORMAL_RESULT_TREE_INDEX.json"
    _write_json(index_path, index)
    manifest_path = destination / "RUN_MANIFEST.json"
    binding.update(
        {
            "path": manifest_path.relative_to(repo).as_posix(),
            "sha256": _sha256(manifest_path),
            "tree_index_path": index_path.relative_to(repo).as_posix(),
            "tree_index_sha256": _sha256(index_path),
            "tree_root_sha256": index["root_sha256"],
        }
    )


def _attach_formal_distribution_receipt(release: Path, manifest: dict) -> None:
    evidence = manifest["formal_evidence"]
    receipt = {
        "schema_version": "operate-formal-distribution-receipt-v1",
        "release_id": manifest["release_id"],
        "hf_repo_id": "Xnhyacinth/OPERATE",
        "visibility": "private",
        "revision": "a" * 40,
        "verification": "private_cas_exact_snapshot_v1",
        "bundle_manifest_sha256": "b" * 64,
        "release_manifest_sha256": hashlib.sha256(
            (
                json.dumps(
                    manifest, indent=2, ensure_ascii=False, sort_keys=True
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest(),
        "formal_evidence_archive": "formal_runtime_evidence.tar.zst",
        "formal_evidence_archive_sha256": "c" * 64,
        "formal_result_tree_roots": {
            "logical_persistent": evidence["logical_batch_manifest"][
                "tree_root_sha256"
            ],
            "realtime_persistent": evidence["realtime_batch_manifest"][
                "tree_root_sha256"
            ],
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    _write_json(release / "formal_distribution_receipt.json", receipt)


def _published_formal_fixture(tmp_path: Path) -> tuple[Path, dict]:
    repo = tmp_path / "repo"
    release = repo / "release" / "operate_v0_61_0"
    release.mkdir(parents=True)
    hashes = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in (
            "manifest",
            "runtime",
            "core",
            "source",
            "evidence",
            "evidence-root",
            "candidate",
            "candidate-identity",
            "backend",
            "backend-identity",
            "implementation",
            "pipeline",
            "tooling",
        )
    }
    _write_json(release / "core_suite.json", {"scenarios": []})
    _write_json(release / "protocol21_source_suite.json", {"scenarios": []})
    public_evidence = {"binding_root_sha256": hashes["evidence-root"]}
    _write_json(release / "protocol21_public_evidence_bundle.json", public_evidence)
    candidate_identity = {"all_candidates": hashes["candidate-identity"]}
    _write_json(
        release / "candidate_closure.json",
        {"identity_set_sha256": candidate_identity},
    )
    _write_json(
        release / "backend_runtime_closure.json",
        {"identity_sha256": hashes["backend-identity"]},
    )
    hashes.update(
        {
            "core": _sha256(release / "core_suite.json"),
            "source": _sha256(release / "protocol21_source_suite.json"),
            "evidence": _sha256(release / "protocol21_public_evidence_bundle.json"),
            "candidate": _sha256(release / "candidate_closure.json"),
            "candidate-identity": _canonical_sha256(candidate_identity),
            "backend": _sha256(release / "backend_runtime_closure.json"),
        }
    )
    _write_json(
        release / "formal_runtime_bundle.json",
        {
            "release_id": release.name,
            "implementation_tree_sha256": hashes["implementation"],
            "core_release_pipeline_sha256": hashes["pipeline"],
            "release_tooling_sha256": hashes["tooling"],
            "public_evidence": {
                "binding_root_sha256": hashes["evidence-root"],
                "sha256": hashes["evidence"],
            },
        },
    )
    hashes["runtime"] = _sha256(release / "formal_runtime_bundle.json")
    identity = {
        "release_id": release.name,
        "formal_manifest_sha256": hashes["manifest"],
        "formal_runtime_bundle_sha256": hashes["runtime"],
        "formal_core_suite_sha256": hashes["core"],
        "formal_source_suite_sha256": hashes["source"],
        "formal_public_evidence_sha256": hashes["evidence"],
        "formal_public_evidence_binding_root_sha256": hashes["evidence-root"],
        "formal_candidate_closure_sha256": hashes["candidate"],
        "formal_candidate_closure_identity_sha256": hashes["candidate-identity"],
        "formal_backend_runtime_closure_sha256": hashes["backend"],
        "formal_backend_runtime_closure_identity_sha256": hashes["backend-identity"],
        "implementation_tree_sha256": hashes["implementation"],
        "formal_core_release_pipeline_sha256": hashes["pipeline"],
        "formal_release_tooling_sha256": hashes["tooling"],
    }
    model = "provider-model"
    logical_treatment = "1" * 64
    realtime_treatment = "2" * 64
    logical_payload = {
        **identity,
        "models": [model],
        "agent_treatment_sha256_by_model": {model: logical_treatment},
        "published_artifacts": {
            "episodes": {"path": "formal_episodes.jsonl"},
            "leaderboard": {"path": "leaderboard.json"},
        },
    }
    logical_index, logical_manifest = _formal_result_tree(
        release,
        model=model,
        treatment=logical_treatment,
        payload=logical_payload,
        files={
            "formal_episodes.jsonl": json.dumps(
                {
                    "trajectory_summary": {
                        "trajectory_path": "trajectories/episode",
                        "provider_audit_artifact": {
                            "path": "trajectories/episode.provider_audit.jsonl"
                        },
                    }
                }
            )
            + "\n",
            "leaderboard.json": "{}\n",
            "trajectories/episode.provider_audit.jsonl": "{}\n",
        },
    )
    realtime_binding = {
        key: value
        for key, value in identity.items()
        if key
        not in {
            "release_id",
            "formal_manifest_sha256",
            "implementation_tree_sha256",
            "formal_core_release_pipeline_sha256",
            "formal_backend_runtime_closure_identity_sha256",
            "formal_release_tooling_sha256",
        }
    }
    realtime_binding.update(
        {
            "core_release_pipeline_sha256": identity[
                "formal_core_release_pipeline_sha256"
            ],
            "backend_runtime_closure_identity_sha256": identity[
                "formal_backend_runtime_closure_identity_sha256"
            ],
            "release_tooling_sha256": identity["formal_release_tooling_sha256"],
        }
    )
    realtime_payload = {
        "model": model,
        "batch_treatment_sha256": realtime_treatment,
        "batch_treatment_identity": {
            "formal_release_id": release.name,
            "formal_manifest_sha256": identity["formal_manifest_sha256"],
            "implementation_tree_sha256": identity["implementation_tree_sha256"],
            "formal_runtime_binding": realtime_binding,
        },
        "artifacts": {
            "episodes_journal": {"path": "episodes.jsonl"},
            "episodes": {"path": "formal_episodes.jsonl"},
            "realtime_scorecard": {"path": "scorecard.json"},
            "leaderboard": {"path": "leaderboard.json"},
            "episode_artifacts": [
                {"job_key": "job", "artifact_path": "trajectories/episode.json"}
            ],
        },
    }
    realtime_index, realtime_manifest = _formal_result_tree(
        release,
        model=model,
        treatment=realtime_treatment,
        payload=realtime_payload,
        files={
            "episodes.jsonl": json.dumps(
                {"job_key": "job", "artifact_path": "trajectories/episode.json"}
            )
            + "\n",
            "formal_episodes.jsonl": json.dumps(
                {"job_key": "job", "artifact_path": "trajectories/episode.json"}
            )
            + "\n",
            "scorecard.json": "{}\n",
            "leaderboard.json": "{}\n",
            "trajectories/episode.json": "{}\n",
        },
    )

    def binding(path: Path, index: dict, mode: str, treatment: str) -> dict:
        index_path = path.parent / "FORMAL_RESULT_TREE_INDEX.json"
        return {
            "path": path.relative_to(repo).as_posix(),
            "sha256": _sha256(path),
            "model": model,
            "interaction_mode": mode,
            "treatment_sha256": treatment,
            "tree_index_path": index_path.relative_to(repo).as_posix(),
            "tree_index_sha256": _sha256(index_path),
            "tree_root_sha256": index["root_sha256"],
        }

    logical_binding = binding(
        logical_manifest,
        logical_index,
        "logical_persistent",
        logical_treatment,
    )
    realtime_evidence_binding = binding(
        realtime_manifest,
        realtime_index,
        "realtime_persistent",
        realtime_treatment,
    )
    manifest = {
        "release_id": release.name,
        "status": "formal_evaluation_complete",
        "implementation_tree_sha256": identity["implementation_tree_sha256"],
        "core_release_pipeline_sha256": identity["formal_core_release_pipeline_sha256"],
        "release_tooling_sha256": identity["formal_release_tooling_sha256"],
        "formal_runtime_bundle": {
            "path": "formal_runtime_bundle.json",
            "sha256": identity["formal_runtime_bundle_sha256"],
        },
        "core_suite": {
            "path": "core_suite.json",
            "sha256": identity["formal_core_suite_sha256"],
        },
        "candidate_closure": {
            "path": "candidate_closure.json",
            "sha256": identity["formal_candidate_closure_sha256"],
            "identity_set_sha256": candidate_identity,
        },
        "backend_runtime_closure": {
            "path": "backend_runtime_closure.json",
            "sha256": identity["formal_backend_runtime_closure_sha256"],
            "identity_sha256": identity[
                "formal_backend_runtime_closure_identity_sha256"
            ],
        },
        "protocol21_replay": {
            "source_suite_sha256": identity["formal_source_suite_sha256"],
            "evidence_bundle_sha256": identity["formal_public_evidence_sha256"],
        },
        "formal_evidence": {
            "logical_batch_manifest": logical_binding,
            "realtime_batch_manifest": realtime_evidence_binding,
        },
        "formal_evaluation_completion": {
            "schema_version": "operate-formal-evaluation-completion-v2",
            "input_release_manifest_sha256": identity["formal_manifest_sha256"],
            "runtime_identity": identity,
            "model": model,
            "logical_batch_manifest": dict(logical_binding),
            "realtime_batch_manifest": dict(realtime_evidence_binding),
        },
    }
    _attach_formal_distribution_receipt(release, manifest)
    return release, manifest


def test_formal_publication_checks_accept_relocated_release_tree(
    tmp_path: Path,
) -> None:
    release, manifest = _published_formal_fixture(tmp_path)
    original_repo = tmp_path / "repo"
    relocated_repo = tmp_path / "relocated-repo"
    shutil.copytree(original_repo, relocated_repo)
    shutil.rmtree(original_repo)
    release = relocated_repo / "release" / release.name

    checks = _formal_publication_checks(
        release,
        manifest,
        artifact_root=relocated_repo,
    )

    assert all(checks.values()), checks


@pytest.mark.parametrize("mutation", ("missing", "tampered"))
def test_v061_formal_publication_requires_distribution_receipt(
    tmp_path: Path, mutation: str
) -> None:
    release, manifest = _published_formal_fixture(tmp_path)
    receipt_path = release / "formal_distribution_receipt.json"
    if mutation == "missing":
        receipt_path.unlink()
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["formal_evidence_archive_sha256"] = "d" * 64
        _write_json(receipt_path, receipt)

    checks = _formal_publication_checks(
        release,
        manifest,
        artifact_root=tmp_path / "repo",
    )

    assert checks["agentic_formal_distribution_receipt_valid"] is False
    assert checks["agentic_formal_result_tree_valid"] is True


def test_historical_formal_publication_does_not_require_distribution_receipt(
    tmp_path: Path,
) -> None:
    release, manifest = _published_formal_fixture(tmp_path)
    manifest["release_id"] = "operate_v0_60_0"
    (release / "formal_distribution_receipt.json").unlink()

    checks = _formal_publication_checks(
        release,
        manifest,
        artifact_root=tmp_path / "repo",
    )

    assert checks["agentic_formal_distribution_receipt_valid"] is True


def test_distribution_receipt_does_not_substitute_for_installed_result_trees(
    tmp_path: Path,
) -> None:
    release, manifest = _published_formal_fixture(tmp_path)
    repo = tmp_path / "repo"
    logical = repo / manifest["formal_evidence"]["logical_batch_manifest"]["path"]
    shutil.rmtree(logical.parent)

    checks = _formal_publication_checks(release, manifest, artifact_root=repo)

    assert checks["agentic_formal_distribution_receipt_valid"] is True
    assert checks["agentic_formal_result_tree_valid"] is False
    assert checks["agentic_formal_result_paths_portable"] is False


def test_logical_sidecars_resolve_from_batch_manifest_root(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    sidecar = root / "trajectories" / "episode.provider_audit.jsonl"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{}\n", encoding="utf-8")
    rows = [
        {
            "trajectory_summary": {
                "trajectory_path": "trajectories/episode",
                "provider_audit_artifact": {
                    "path": "trajectories/episode.provider_audit.jsonl"
                },
            }
        }
    ]

    resolved = _logical_rows_for_validation(rows, batch_root=root)

    assert resolved is not None
    assert resolved[0]["trajectory_summary"]["trajectory_path"] == str(
        root / "trajectories" / "episode"
    )
    assert resolved[0]["trajectory_summary"]["provider_audit_artifact"]["path"] == str(
        sidecar
    )
    assert rows[0]["trajectory_summary"]["trajectory_path"] == "trajectories/episode"


@pytest.mark.parametrize(
    "target",
    ("completion", "logical", "realtime"),
)
def test_formal_publication_rejects_runtime_identity_drift(
    tmp_path: Path, target: str
) -> None:
    release, manifest = _published_formal_fixture(tmp_path)
    evidence = manifest["formal_evidence"]
    if target == "completion":
        manifest["formal_evaluation_completion"]["runtime_identity"][
            "formal_release_tooling_sha256"
        ] = "f" * 64
    else:
        batch_path = tmp_path / "repo" / evidence[f"{target}_batch_manifest"]["path"]
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        if target == "logical":
            batch["formal_release_tooling_sha256"] = "f" * 64
        else:
            batch["batch_treatment_identity"]["formal_runtime_binding"][
                "release_tooling_sha256"
            ] = "f" * 64
        _write_json(batch_path, batch)
        _refresh_formal_result_binding(
            tmp_path / "repo", evidence[f"{target}_batch_manifest"]
        )
        manifest["formal_evaluation_completion"][f"{target}_batch_manifest"] = dict(
            evidence[f"{target}_batch_manifest"]
        )

    checks = _formal_publication_checks(
        release,
        manifest,
        artifact_root=tmp_path / "repo",
    )

    assert checks["agentic_formal_completion_identity_valid"] is False
    assert checks["agentic_formal_result_tree_valid"] is True


def test_formal_publication_separates_portability_from_tree_integrity(
    tmp_path: Path,
) -> None:
    release, manifest = _published_formal_fixture(tmp_path)
    repo = tmp_path / "repo"
    binding = manifest["formal_evidence"]["logical_batch_manifest"]
    batch_path = repo / binding["path"]
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    episodes_path = batch_path.parent / batch["published_artifacts"]["episodes"]["path"]
    row = json.loads(episodes_path.read_text(encoding="utf-8"))
    row["trajectory_summary"]["provider_audit_artifact"]["path"] = str(
        episodes_path.parent / "trajectories" / "episode.provider_audit.jsonl"
    )
    episodes_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _refresh_formal_result_binding(repo, binding)
    manifest["formal_evaluation_completion"]["logical_batch_manifest"] = dict(binding)

    checks = _formal_publication_checks(release, manifest, artifact_root=repo)

    assert checks["agentic_formal_result_tree_valid"] is True
    assert checks["agentic_formal_result_paths_portable"] is False


@pytest.mark.parametrize("mutation", ("tamper", "extra", "symlink"))
def test_formal_publication_rejects_unindexed_or_unsafe_result_bytes(
    tmp_path: Path, mutation: str
) -> None:
    release, manifest = _published_formal_fixture(tmp_path)
    binding = manifest["formal_evidence"]["logical_batch_manifest"]
    manifest_path = tmp_path / "repo" / binding["path"]
    root = manifest_path.parent
    if mutation == "tamper":
        (root / "leaderboard.json").write_text('{"tampered":true}\n', encoding="utf-8")
    elif mutation == "extra":
        (root / "unindexed.log").write_text("unexpected\n", encoding="utf-8")
    else:
        target = tmp_path / "outside.jsonl"
        target.write_text("{}\n", encoding="utf-8")
        (root / "unsafe.jsonl").symlink_to(target)

    checks = _formal_publication_checks(
        release,
        manifest,
        artifact_root=tmp_path / "repo",
    )

    assert checks["agentic_formal_result_tree_valid"] is False


def test_logical_sidecar_cannot_escape_batch_manifest_root(tmp_path: Path) -> None:
    root = tmp_path / "batch"
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")

    resolved = _logical_rows_for_validation(
        [
            {
                "trajectory_summary": {
                    "trajectory_path": "trajectories/episode",
                    "provider_audit_artifact": {"path": str(outside)},
                }
            }
        ],
        batch_root=root,
    )

    assert resolved is None


def test_formal_result_manifest_must_live_under_its_release(tmp_path: Path) -> None:
    release, manifest = _published_formal_fixture(tmp_path)
    repo = tmp_path / "repo"
    binding = manifest["formal_evidence"]["logical_batch_manifest"]
    source_manifest = repo / binding["path"]
    outside_root = repo / "detached-formal-result"
    source_manifest.parent.rename(outside_root)
    detached_manifest = outside_root / "RUN_MANIFEST.json"
    detached_index = outside_root / "FORMAL_RESULT_TREE_INDEX.json"
    binding["path"] = detached_manifest.relative_to(repo).as_posix()
    binding["tree_index_path"] = detached_index.relative_to(repo).as_posix()

    checks = _formal_publication_checks(release, manifest, artifact_root=repo)

    assert checks["agentic_formal_result_tree_valid"] is False


def test_formal_result_tree_index_cannot_be_a_symlink(tmp_path: Path) -> None:
    release, manifest = _published_formal_fixture(tmp_path)
    repo = tmp_path / "repo"
    binding = manifest["formal_evidence"]["logical_batch_manifest"]
    index_path = repo / binding["tree_index_path"]
    outside = tmp_path / "copied-index.json"
    shutil.copyfile(index_path, outside)
    index_path.unlink()
    index_path.symlink_to(outside)

    checks = _formal_publication_checks(release, manifest, artifact_root=repo)

    assert checks["agentic_formal_result_tree_valid"] is False


def _authoritative_release_scenario_count() -> int:
    manifest = json.loads((RELEASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    core = json.loads((RELEASE_DIR / "core_suite.json").read_text(encoding="utf-8"))
    scenario_count = len(core["scenarios"])

    assert core["n_scenarios"] == scenario_count
    assert manifest["n_scenarios"] == scenario_count
    assert manifest["core_suite"]["n_scenarios"] == scenario_count
    return scenario_count


def test_published_evidence_uses_the_declared_formal_tool_choice() -> None:
    assert _formal_tool_choice_matches("auto", AGENTIC_PROFILE_V1) is True
    assert _formal_tool_choice_matches("required", AGENTIC_PROFILE_V1) is False


def test_formal_realtime_contract_tracks_diagnostic_schema() -> None:
    assert REALTIME_FORMAL_CONTRACT_V1["scorecard_version"] == (
        REALTIME_DIAGNOSTIC_SCHEMA
    )
    assert REALTIME_FORMAL_CONTRACT_V1["diagnostic_schema_version"] == (
        REALTIME_DIAGNOSTIC_SCHEMA
    )


def test_v061_requires_event_driven_realtime_v2_contract() -> None:
    run_contract, realtime_contract = _formal_contracts_for_release(
        {"release_id": "operate_v0_61_0"}
    )

    assert run_contract["wakeup_policy"] == realtime_contract["wakeup_policy"]
    assert run_contract["realtime_formal_contract"] == REALTIME_FORMAL_CONTRACT_V2
    assert realtime_contract["contract_version"] == "realtime_persistent.v2"
    assert realtime_contract["realtime_coordinator"] == "realtime_episode_v5"
    assert realtime_contract["wakeup_policy"] == {
        "session_start": True,
        "typed_actionable_events": True,
        "agent_scheduled_reviews": True,
        "harness_periodic_supervisory_scan": False,
        "unknown_events_actionable": False,
    }


def _logical_treatment_identity_payload() -> dict:
    model = "provider-model"
    base_url = "https://copilot.tencent.com/v2"
    provider_route_sha256 = _canonical_sha256(
        {
            "base_url": {
                "scheme": "https",
                "host": "copilot.tencent.com",
                "port": None,
                "path": "/v2",
                "query": [],
            },
            "responses_base_url": None,
            "extra_headers": [],
        }
    )
    profile_identity = {
        "schema_version": "agent_treatment_v1",
        "provider": "openai_compatible",
        "model": model,
        "base_url": base_url,
        "api_version": None,
        "api_version_env": "OPERATE_API_VERSION",
        "responses_base_url": None,
        "responses_base_url_env": "OPERATE_RESPONSES_API_BASE_URL",
        "private_provider_route_sha256": provider_route_sha256,
        "api_mode": "chat_completions",
        "stream_chat_completions": True,
        "temperature": 0.0,
        "max_tokens": 32_768,
        "model_context_window_tokens": 192_000,
        "model_max_output_tokens": 65_536,
        "token_count_method": "utf8_bytes_upper_bound",
        "token_count_version": "1",
        "timeout_s": 300.0,
        "max_consecutive_provider_failures": 1,
        "provider_failure_policy": "abort",
        "provider_rpm_limit": None,
        "provider_rpd_limit": None,
        "provider_rate_limit_scope": None,
        "prompt_mode": "strict",
        "prompt_contract_sha256": prompt_contract_sha256(
            "logical_persistent", "strict"
        ),
        "interaction_mode": "logical_persistent",
        "persistent_history_max_messages": 64,
        "persistent_context_max_chars": 512_000,
        "persistent_memory_max_items": 128,
        "tool_choice": "auto",
        "tool_choice_supported": True,
        "reasoning_effort": None,
        "protocol_repair_max_tokens": 8_192,
        "allow_insecure_http": False,
        "extra_header_names": [],
        "harness": "direct_api",
        "prompt_context_compiler_binding": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        "tool_schema_binding": "decision_envelope.available_tool_schema_sha256",
        "wakeup_policy": deepcopy(REALTIME_FORMAL_CONTRACT_V2["wakeup_policy"]),
    }
    profile_sha256 = _canonical_sha256(profile_identity)
    runtime_identity = {
        "formal_release_id": "operate_v0_61_0",
        "formal_manifest_sha256": "1" * 64,
        "formal_release_tooling_sha256": "2" * 64,
        "formal_readiness_sha256": "3" * 64,
        "formal_core_release_pipeline_sha256": "4" * 64,
        "formal_backend_runtime_closure_identity_sha256": "5" * 64,
    }

    treatment_sha256 = _canonical_sha256(
        {
            "schema_version": "formal_logical_treatment_v1",
            "interaction_mode": "logical_persistent",
            "agent_profile_sha256": profile_sha256,
            "formal_runtime_binding": runtime_identity,
            "implementation_tree_sha256": "6" * 64,
        }
    )
    return {
        "formal_run": True,
        "interaction_mode": "logical_persistent",
        "models": [model],
        "temperature": 0.0,
        "max_tokens": 32_768,
        "model_context_window_tokens_by_model": {model: 192_000},
        "model_max_output_tokens_by_model": {model: 65_536},
        "tool_choice_supported_by_model": {model: True},
        "token_count_method": "utf8_bytes_upper_bound",
        "token_count_version": "1",
        "prompt_mode": "strict",
        "wakeup_policy": deepcopy(REALTIME_FORMAL_CONTRACT_V2["wakeup_policy"]),
        "persistent_history_max_messages": 64,
        "persistent_context_max_chars": 512_000,
        "persistent_memory_max_items": 128,
        "harness": "direct_api",
        "provider_timeout_s": 300.0,
        "provider_rpm_limit": None,
        "provider_rpd_limit": None,
        "provider_rate_limit_scope": None,
        "protocol_repair_max_tokens": 8_192,
        "tool_choice": "auto",
        "reasoning_effort": None,
        "max_consecutive_provider_failures": 1,
        "provider_failure_policy": "abort",
        "base_url": base_url,
        "api_version": None,
        "responses_base_url": None,
        "api_mode": "chat_completions",
        "stream_chat_completions": True,
        "evaluation_implementation_fingerprint": (
            EVALUATION_IMPLEMENTATION_FINGERPRINT
        ),
        "agent_profile_schema_version": "agent_treatment_v1",
        "agent_profile_identity_by_model": {model: profile_identity},
        "agent_profile_sha256_by_model": {model: profile_sha256},
        "agent_treatment_schema_version": "formal_logical_treatment_v1",
        "agent_treatment_sha256_by_model": {model: treatment_sha256},
        **runtime_identity,
        "implementation_tree_sha256": "6" * 64,
    }


def _logical_treatment_sha256(payload: dict, profile_sha256: str) -> str:
    runtime_identity = {
        field: payload[field]
        for field in verify_mod._LOGICAL_FORMAL_RUNTIME_IDENTITY_FIELDS
        if payload.get(field) is not None
    }
    return _canonical_sha256(
        {
            "schema_version": "formal_logical_treatment_v1",
            "interaction_mode": "logical_persistent",
            "agent_profile_sha256": profile_sha256,
            "formal_runtime_binding": runtime_identity,
            "implementation_tree_sha256": payload["implementation_tree_sha256"],
        }
    )


def test_logical_treatment_identity_rejects_self_consistent_hash_tampering() -> None:
    payload = _logical_treatment_identity_payload()
    model = payload["models"][0]
    assert verify_mod._logical_treatment_identity_valid(payload) is True

    tampered_profile_sha256 = "a" * 64
    tampered = deepcopy(payload)
    tampered["agent_profile_sha256_by_model"][model] = tampered_profile_sha256
    tampered["agent_treatment_sha256_by_model"][model] = _logical_treatment_sha256(
        tampered,
        tampered_profile_sha256
    )

    assert verify_mod._logical_treatment_identity_valid(tampered) is False


def test_logical_treatment_identity_rejects_self_consistent_profile_tampering() -> None:
    payload = _logical_treatment_identity_payload()
    model = payload["models"][0]
    assert verify_mod._logical_treatment_identity_valid(payload) is True

    tampered = deepcopy(payload)
    tampered_profile = tampered["agent_profile_identity_by_model"][model]
    tampered_profile["max_tokens"] = 65_536
    tampered_profile_sha256 = _canonical_sha256(tampered_profile)
    tampered["agent_profile_sha256_by_model"][model] = tampered_profile_sha256
    tampered["agent_treatment_sha256_by_model"][model] = _logical_treatment_sha256(
        tampered,
        tampered_profile_sha256
    )

    assert verify_mod._logical_treatment_identity_valid(tampered) is False


@pytest.mark.parametrize("release_id", ("operate_v0_59_0", "operate_v0_60_0"))
def test_pre_v061_releases_keep_historical_realtime_v1_contract(
    release_id: str,
) -> None:
    run_contract, realtime_contract = _formal_contracts_for_release(
        {"release_id": release_id}
    )

    assert run_contract["realtime_formal_contract"] == REALTIME_FORMAL_CONTRACT_V1
    assert realtime_contract["contract_version"] == "realtime_persistent.v1"
    assert realtime_contract["realtime_coordinator"] == "realtime_episode_v4"
    assert "wakeup_policy" not in run_contract
    assert "wakeup_policy" not in realtime_contract


def test_v061_wakeup_policy_is_exact_without_requiring_periodic_activity() -> None:
    policy = deepcopy(REALTIME_FORMAL_CONTRACT_V2["wakeup_policy"])
    zero_periodic_activity = {"wakeup_policy": policy}

    assert _formal_wakeup_policy_valid(
        REALTIME_FORMAL_CONTRACT_V2,
        zero_periodic_activity,
    )
    assert not _formal_wakeup_policy_valid(REALTIME_FORMAL_CONTRACT_V2, {})

    tampered = deepcopy(zero_periodic_activity)
    tampered["wakeup_policy"]["harness_periodic_supervisory_scan"] = True
    assert not _formal_wakeup_policy_valid(REALTIME_FORMAL_CONTRACT_V2, tampered)
    assert _formal_wakeup_policy_valid(REALTIME_FORMAL_CONTRACT_V1, {})


@pytest.mark.parametrize("mutation", ("missing", "tampered"))
def test_v061_formal_integrity_rejects_top_level_wakeup_policy_drift(
    mutation: str,
) -> None:
    manifest = json.loads((RELEASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    core = json.loads((RELEASE_DIR / "core_suite.json").read_text(encoding="utf-8"))
    manifest["release_id"] = "operate_v0_61_0"
    run_contract, _realtime_contract = _formal_contracts_for_release(manifest)
    manifest["formal_run_contract"] = deepcopy(run_contract)

    valid = _agentic_formal_checks(
        RELEASE_DIR,
        manifest,
        core,
        core["scenarios"],
        portable=True,
        artifact_root=REPO_ROOT,
    )
    assert valid["agentic_formal_run_contract_valid"] is True

    if mutation == "missing":
        manifest["formal_run_contract"].pop("wakeup_policy")
    else:
        manifest["formal_run_contract"]["wakeup_policy"][
            "harness_periodic_supervisory_scan"
        ] = True
    drifted = _agentic_formal_checks(
        RELEASE_DIR,
        manifest,
        core,
        core["scenarios"],
        portable=True,
        artifact_root=REPO_ROOT,
    )

    assert drifted["agentic_formal_run_contract_valid"] is False


def _portable_pipeline_fixture(tmp_path: Path) -> tuple[Path, dict, list[dict]]:
    release = tmp_path / "release" / "operate_v0_58_0"
    source = release / "protocol21_source_suite.json"
    _write_json(source, {"scenarios": []})
    tree = "a" * 64
    release_pipeline = "b" * 64
    row = {
        "scenario_id": "traffic/corridor/time_pressure/high/demo",
        "scenario_signature": "demo-signature",
        "path": "scenarios/demo.yaml",
        "yaml_sha256": "c" * 64,
    }
    stage_hashes = {
        name: hashlib.sha256(name.encode()).hexdigest()
        for name in PIPELINE_STAGE_HASH_FIELDS
    }
    pipeline_artifacts = {
        "pipeline_manifest_sha256": "d" * 64,
        "core_release_pipeline_sha256": release_pipeline,
        "stage_artifacts": {
            name: {
                "relative_path": PIPELINE_STAGE_FILES[name],
                "sha256": stage_hashes[name],
            }
            for name in PIPELINE_STAGE_HASH_FIELDS
        },
        **{
            hash_field: stage_hashes[name]
            for name, hash_field in PIPELINE_STAGE_HASH_FIELDS.items()
        },
    }
    evidence = {
        "schema_version": "protocol21-public-evidence-bundle-v1",
        "status": "complete",
        "scope": "portable_summary_of_immutable_internal_evidence",
        "pipeline": {
            "manifest": {"sha256": "d" * 64},
            "status": "formal_evaluation_ready",
            "implementation_tree_sha256": tree,
            "core_release_pipeline_sha256": release_pipeline,
            "source_suite_sha256": _sha256(source),
        },
        "artifacts": {
            **{
                name: {
                    "sha256": stage_hashes[name],
                    "implementation_tree_sha256": tree,
                    "core_release_pipeline_sha256": release_pipeline,
                }
                for name in PIPELINE_STAGE_HASH_FIELDS
            },
            "source_suite": {"sha256": _sha256(source)},
        },
        "artifact_dependency_edges": [],
        "scenario_bindings": [
            {
                "scenario_id": row["scenario_id"],
                "scenario_signature": row["scenario_signature"],
                "scenario_uri": f"repo://{row['path']}",
                "scenario_yaml_sha256": row["yaml_sha256"],
            }
        ],
        "source_asset_bindings": [],
        "counts": {
            "pipeline_stages": len(PIPELINE_STAGE_HASH_FIELDS),
            "artifact_nodes": len(PIPELINE_STAGE_HASH_FIELDS) + 1,
            "artifact_dependency_edges": 0,
            "core_scenarios": 1,
            "unique_source_assets": 0,
        },
    }
    evidence["binding_root_sha256"] = _canonical_payload_sha256(evidence)
    evidence_path = release / "protocol21_public_evidence_bundle.json"
    _write_json(evidence_path, evidence)
    manifest = {
        "implementation_tree_sha256": tree,
        "core_release_pipeline_sha256": release_pipeline,
        "pipeline_artifacts": pipeline_artifacts,
        "protocol21_replay": {
            "source_suite": ("release/operate_v0_58_0/protocol21_source_suite.json"),
            "source_suite_sha256": _sha256(source),
            "evidence_bundle": evidence_path.name,
            "evidence_bundle_sha256": _sha256(evidence_path),
            "core_release_pipeline_sha256": release_pipeline,
        },
    }
    return release, manifest, [row]


def test_portable_pipeline_closure_does_not_require_live_tooling(
    tmp_path: Path,
) -> None:
    release, manifest, rows = _portable_pipeline_fixture(tmp_path)

    assert _portable_evidence_closure_valid(release, manifest, rows) is True


def test_portable_pipeline_closure_rejects_internal_identity_drift(
    tmp_path: Path,
) -> None:
    release, manifest, rows = _portable_pipeline_fixture(tmp_path)
    evidence_path = release / "protocol21_public_evidence_bundle.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["artifacts"]["behavioral"]["core_release_pipeline_sha256"] = "0" * 64
    evidence["binding_root_sha256"] = _canonical_payload_sha256(evidence)
    _write_json(evidence_path, evidence)
    manifest["protocol21_replay"]["evidence_bundle_sha256"] = _sha256(evidence_path)

    assert _portable_evidence_closure_valid(release, manifest, rows) is False


def _promoted_closure_release(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    from tests.test_promote_operate_release import _fixture, promote_release

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
    return paths["output"], paths["repo"]


def test_integrity_rejects_candidate_closure_hash_tamper(
    tmp_path: Path, monkeypatch
) -> None:
    release, repo = _promoted_closure_release(tmp_path, monkeypatch)
    closure_path = release / "candidate_closure.json"
    closure_path.write_text(
        closure_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    report = build_release_integrity_report(release, artifact_root=repo)

    assert report["checks"]["candidate_closure_manifest_binding_valid"] is False
    assert report["ok"] is False


def test_integrity_validates_candidate_override_without_writing_pending_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, repo = _promoted_closure_release(tmp_path, monkeypatch)
    manifest_path = release / "manifest.json"
    pending = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = deepcopy(pending)
    candidate["public_release_ready"] = True
    candidate["leaderboard_eligible"] = True

    report = build_release_integrity_report(
        release,
        portable=True,
        artifact_root=repo,
        manifest_override=candidate,
    )

    assert report["manifest"]["public_release_ready"] is True
    assert report["manifest"]["leaderboard_eligible"] is True
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == pending


def test_integrity_rejects_tampered_candidate_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, repo = _promoted_closure_release(tmp_path, monkeypatch)
    manifest_path = release / "manifest.json"
    pending_bytes = manifest_path.read_bytes()
    candidate = json.loads(pending_bytes)
    candidate["core_suite"]["sha256"] = "0" * 64

    report = build_release_integrity_report(
        release,
        portable=True,
        artifact_root=repo,
        manifest_override=candidate,
    )

    assert report["checks"]["core_sha256_matches_manifest"] is False
    assert report["ok"] is False
    assert manifest_path.read_bytes() == pending_bytes


def test_integrity_rejects_invalid_dimension_applicability_contract(
    tmp_path: Path, monkeypatch
) -> None:
    release, repo = _promoted_closure_release(tmp_path, monkeypatch)
    core_path = release / "core_suite.json"
    core = json.loads(core_path.read_text(encoding="utf-8"))
    scenario_path = repo / core["scenarios"][0]["path"]
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario["backend_config"]["dimension_applicability"]["system_survival"][
        "reason"
    ] = 1
    scenario_path.write_text(
        yaml.safe_dump(scenario, sort_keys=False),
        encoding="utf-8",
    )
    core["scenarios"][0]["yaml_sha256"] = _sha256(scenario_path)
    _write_json(core_path, core)
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["core_suite"]["sha256"] = _sha256(core_path)
    _write_json(manifest_path, manifest)

    report = build_release_integrity_report(release, artifact_root=repo)

    assert report["checks"]["core_dimension_applicability_complete"] is False
    assert report["ok"] is False


def test_distribution_only_tooling_drift_is_diagnostic_for_formal_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, repo = _promoted_closure_release(tmp_path, monkeypatch)
    before = implementation_identity(repo)
    tooling = repo / "scripts/release_only_tool.py"
    tooling.parent.mkdir(parents=True, exist_ok=True)
    tooling.write_text("VALUE = 'drifted'\n", encoding="utf-8")
    after = implementation_identity(repo)

    report = build_release_integrity_report(release, artifact_root=repo)

    assert after["evaluation_runtime_sha256"] == before["evaluation_runtime_sha256"]
    assert (
        after["core_release_pipeline_sha256"] == before["core_release_pipeline_sha256"]
    )
    assert after["release_tooling_sha256"] != before["release_tooling_sha256"]
    assert report["checks"]["agentic_release_tooling_binding_valid"] is True
    assert (
        report["diagnostics"]["live_release_tooling_matches_promoted_snapshot"] is False
    )
    assert {"code": "live_release_tooling_matches_promoted_snapshot"} not in report[
        "issues"
    ]
    assert report["ok"] is True
    assert report["formal_run_ready"] is True


def test_integrity_rejects_internal_release_tooling_hash_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, repo = _promoted_closure_release(tmp_path, monkeypatch)
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pipeline_path = (
        repo / manifest["pipeline_dir"] / "protocol2_v21_pipeline_manifest.json"
    )
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline["release_tooling_sha256"] = "0" * 64
    _write_json(pipeline_path, pipeline)
    manifest["pipeline_artifacts"]["pipeline_manifest_sha256"] = _sha256(pipeline_path)
    _write_json(manifest_path, manifest)

    report = build_release_integrity_report(release, artifact_root=repo)

    assert report["checks"]["agentic_pipeline_artifact_binding_valid"] is True
    assert report["checks"]["agentic_release_tooling_binding_valid"] is False
    assert report["ok"] is False
    assert report["formal_run_ready"] is False


def _refresh_candidate_closure_binding(release: Path) -> None:
    closure_path = release / "candidate_closure.json"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    summary = closure["summary"]
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["candidate_closure"] = {
        "path": closure_path.name,
        "sha256": _sha256(closure_path),
        "schema_version": closure["schema_version"],
        "status": closure["status"],
        "n_independent_candidates": summary["n_independent_candidates"],
        "n_terminal_candidates": summary["n_terminal_candidates"],
        "n_unresolved_candidates": summary["n_unresolved_candidates"],
        "identity_set_sha256": closure["identity_set_sha256"],
    }
    _write_json(manifest_path, manifest)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("schema", "candidate_closure_semantics_valid"),
        ("status", "candidate_closure_terminal_accounting_valid"),
        ("accounting", "candidate_closure_terminal_accounting_valid"),
        ("selection", "candidate_closure_selection_identity_valid"),
    ],
)
def test_integrity_rejects_candidate_closure_semantic_tamper(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    failed_check: str,
) -> None:
    release, repo = _promoted_closure_release(tmp_path, monkeypatch)
    closure_path = release / "candidate_closure.json"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    if mutation == "schema":
        closure["schema_version"] = "stale-schema"
    elif mutation == "status":
        closure["status"] = "stale-status"
    elif mutation == "accounting":
        closure["summary"]["n_unresolved_candidates"] = 1
    else:
        closure["candidates"][0]["canonical_identity"]["scenario_signature"] = (
            "stale-signature"
        )
    _write_json(closure_path, closure)
    _refresh_candidate_closure_binding(release)

    report = build_release_integrity_report(release, artifact_root=repo)

    assert report["checks"][failed_check] is False
    assert report["ok"] is False


def _refresh_backend_runtime_closure_binding(release: Path) -> None:
    closure_path = release / "backend_runtime_closure.json"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    summary = closure["summary"]
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["backend_runtime_closure"] = {
        "path": closure_path.name,
        "sha256": _sha256(closure_path),
        "schema_version": closure["schema_version"],
        "n_archived_files": summary["n_archived_files"],
        "n_external_sources": summary["n_external_sources"],
        "n_backend_links": summary["n_backend_links"],
        "n_runtime_packages": summary["n_runtime_packages"],
        "identity_sha256": closure["identity_sha256"],
    }
    _write_json(manifest_path, manifest)


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        ("hash", "backend_runtime_closure_manifest_binding_valid"),
        ("schema", "backend_runtime_closure_semantics_valid"),
        ("status", "backend_runtime_closure_terminal_accounting_valid"),
        ("terminal", "backend_runtime_closure_terminal_accounting_valid"),
        ("accounting", "backend_runtime_closure_terminal_accounting_valid"),
    ],
)
def test_integrity_rejects_backend_runtime_closure_tamper(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    failed_check: str,
) -> None:
    release, repo = _promoted_closure_release(tmp_path, monkeypatch)
    closure_path = release / "backend_runtime_closure.json"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    if mutation == "hash":
        closure_path.write_text(
            closure_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    else:
        if mutation == "schema":
            closure["schema_version"] = "stale-schema"
        elif mutation == "status":
            closure["status"] = "stale-status"
        elif mutation == "terminal":
            closure["terminal"] = False
        else:
            closure["summary"]["n_unresolved"] = 1
        closure["identity_sha256"] = _canonical_sha256(
            {key: value for key, value in closure.items() if key != "identity_sha256"}
        )
        _write_json(closure_path, closure)
        _refresh_backend_runtime_closure_binding(release)

    report = build_release_integrity_report(release, artifact_root=repo)

    assert report["checks"][failed_check] is False
    assert report["ok"] is False


def test_full_integrity_uses_live_graph_and_portable_uses_carried_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release, repo = _promoted_closure_release(tmp_path, monkeypatch)

    validation_modes: list[bool] = []

    def reject_incomplete_live_graph(**kwargs: object) -> None:
        require_live = bool(kwargs["require_live_contract"])
        validation_modes.append(require_live)
        if require_live:
            raise ValueError("opendss_runtime_input_mismatch:fixture")

    monkeypatch.setattr(
        "scripts.build_operate_backend_runtime_closure."
        "validate_opendss_runtime_asset_closure",
        reject_incomplete_live_graph,
    )

    full = build_release_integrity_report(release, artifact_root=repo)
    portable = build_release_integrity_report(
        release,
        artifact_root=repo,
        portable=True,
    )

    assert full["checks"]["backend_runtime_closure_semantics_valid"] is False
    assert full["ok"] is False
    assert portable["checks"]["backend_runtime_closure_semantics_valid"] is True
    assert validation_modes == [True, False]


@pytest.mark.parametrize("mutation", ["drift", "missing"])
def test_full_integrity_binds_runtime_packages_to_live_uv_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    release, repo = _promoted_closure_release(tmp_path, monkeypatch)
    uv_lock = repo / "uv.lock"
    if mutation == "drift":
        uv_lock.write_text("version = 2\n", encoding="utf-8")
    else:
        uv_lock.unlink()

    full = build_release_integrity_report(release, artifact_root=repo)
    portable = build_release_integrity_report(
        release,
        artifact_root=repo,
        portable=True,
    )

    assert full["checks"]["backend_runtime_closure_semantics_valid"] is False
    assert full["ok"] is False
    assert portable["checks"]["backend_runtime_closure_semantics_valid"] is True


def test_active_release_portable_integrity() -> None:
    report = build_release_integrity_report(RELEASE_DIR, portable=True)
    scenario_count = _authoritative_release_scenario_count()

    assert report["ok"], report["issues"]
    assert report["release_id"] == RELEASE_DIR.name
    assert report["core"]["len_scenarios"] == scenario_count
    assert report["verification_mode"] == "portable"


def test_public_runtime_uses_carried_evidence_when_private_pipeline_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = json.loads((RELEASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    core = json.loads((RELEASE_DIR / "core_suite.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(
        "scripts.verify_release_integrity.implementation_identity",
        lambda _root: {
            "implementation_tree_sha256": manifest["implementation_tree_sha256"],
            "core_release_pipeline_sha256": manifest[
                "core_release_pipeline_sha256"
            ],
            "release_tooling_sha256": "0" * 64,
        },
    )

    checks = _agentic_formal_checks(
        RELEASE_DIR,
        manifest,
        core,
        core["scenarios"],
        portable=False,
        artifact_root=tmp_path,
    )

    assert checks["agentic_pipeline_artifact_binding_valid"] is True
    assert checks["agentic_release_pipeline_binding_valid"] is True
    assert checks["agentic_release_tooling_binding_valid"] is True
    assert checks["agentic_implementation_tree_binding_valid"] is True


def test_core_only_formal_integrity_ignores_optional_diagnostics() -> None:
    manifest = json.loads((RELEASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    core = json.loads((RELEASE_DIR / "core_suite.json").read_text(encoding="utf-8"))
    rows = core["scenarios"]

    logical_contract = manifest["formal_batch_contract"]
    logical_contract.pop("diagnostic_readiness", None)
    logical_contract.pop("agency_readiness_bundle", None)
    logical_contract["agentic_profile"] = deepcopy(AGENTIC_PROFILE_V1)
    manifest["formal_evidence"]["diagnostic_readiness"] = "missing/diagnostic.json"
    manifest["formal_evidence"]["agency_readiness_bundle"] = "missing/agency.json"
    manifest["pipeline_artifacts"]["diagnostic_readiness_sha256"] = "0" * 64
    manifest["pipeline_artifacts"]["agency_readiness_bundle_sha256"] = "1" * 64

    checks = _agentic_formal_checks(
        RELEASE_DIR,
        manifest,
        core,
        rows,
        portable=True,
        artifact_root=REPO_ROOT,
    )

    assert all(checks.values()), checks


def test_pending_release_exposes_publication_checks_without_requiring_results() -> None:
    manifest = json.loads((RELEASE_DIR / "manifest.json").read_text(encoding="utf-8"))
    core = json.loads((RELEASE_DIR / "core_suite.json").read_text(encoding="utf-8"))

    checks = _agentic_formal_checks(
        RELEASE_DIR,
        manifest,
        core,
        core["scenarios"],
        portable=True,
        artifact_root=REPO_ROOT,
    )

    assert checks["agentic_formal_completion_identity_valid"] is True
    assert checks["agentic_formal_result_tree_valid"] is True
    assert checks["agentic_formal_result_paths_portable"] is True
