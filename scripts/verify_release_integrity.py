#!/usr/bin/env python3
"""Verify release suite integrity and provenance completeness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.llm_agent import (  # noqa: E402
    TOKEN_COUNT_METHOD_UTF8_BYTES,
    TOKEN_COUNT_VERSION_V1,
    prompt_contract_sha256,
)
from core.implementation_identity import implementation_identity  # noqa: E402
from evaluation.dimension_applicability import (  # noqa: E402
    dimension_applicability_contract_is_valid,
)
from runner import EVALUATION_IMPLEMENTATION_FINGERPRINT  # noqa: E402

DEFAULT_RELEASE = REPO_ROOT / "release" / "operate_v0_61_0"

JsonDict = dict[str, Any]

FORMAL_RESULT_TREE_INDEX_NAME = "FORMAL_RESULT_TREE_INDEX.json"
FORMAL_RESULT_TREE_INDEX_SCHEMA = "operate-formal-result-tree-index-v1"
FORMAL_DISTRIBUTION_RECEIPT_NAME = "formal_distribution_receipt.json"
FORMAL_DISTRIBUTION_RECEIPT_SCHEMA = "operate-formal-distribution-receipt-v1"

PRIMARY_LEADERBOARD_FORMULA_VERSION = "effective_source_backend_domain_macro_v1"
PRIMARY_INFERENCE_VERSION = "physical_cluster_hierarchical_bootstrap_randomization_v1"

AGENTIC_PROFILE_V1: JsonDict = {
    "max_tokens": 32_768,
    "protocol_repair_max_tokens": 8_192,
    "persistent_history_max_messages": 64,
    "persistent_context_max_chars": 512_000,
    "persistent_memory_max_items": 128,
    "provider_timeout_s": 300.0,
    "max_consecutive_provider_failures": 1,
    "provider_failure_policy": "abort",
    "tool_choice": "auto",
    "stream_chat_completions": True,
}

REALTIME_FORMAL_CONTRACT_BASE_V1: JsonDict = {
    "contract_version": "realtime_persistent.v1",
    "interaction_mode": "realtime_persistent",
    "leaderboard": "realtime_supervision",
    "scorecard_version": "realtime-diagnostics/1.4",
    "aggregation_version": "realtime-scorecard-micro-v1",
    "merge_with_primary_leaderboard": False,
    "selection_binding": "same_release_core",
    "clock_profile": {
        "kind": "soft_realtime_monotonic_single_writer",
        "tick_interval_s": 5.0,
        "episode_timeout_policy": (
            "horizon_ticks_x_tick_plus_provider_timeout_plus_tick"
        ),
        "process_hard_timeout_overhead_s": 30.0,
        "termination_grace_s": 5.0,
    },
    "safety_profile": {
        "supervisor": "domain_neutral_hold",
        "native_takeover_applicable": False,
    },
    "required_artifacts": [
        "provider_audit",
        "event_contract",
        "evidence_closure",
        "action_lifecycle",
        "semantic_ledger",
        "structured_memory",
    ],
}

REALTIME_FORMAL_CONTRACT_V1 = {
    **REALTIME_FORMAL_CONTRACT_BASE_V1,
    "scorecard_version": "realtime-diagnostics/1.6",
    "diagnostic_schema_version": "realtime-diagnostics/1.6",
    "batch_schema_version": "realtime-formal-batch/1.1",
    "scorecard_schema_version": "realtime-formal-scorecard/1.1",
    "episode_schema_version": "realtime-episode/1.1",
    "treatment_schema_version": "realtime-treatment/1.1",
    "realtime_coordinator": "realtime_episode_v4",
}

FORMAL_WAKEUP_POLICY_V2: JsonDict = {
    "session_start": True,
    "typed_actionable_events": True,
    "agent_scheduled_reviews": True,
    "harness_periodic_supervisory_scan": False,
    "unknown_events_actionable": False,
}

REALTIME_FORMAL_CONTRACT_V2: JsonDict = {
    **REALTIME_FORMAL_CONTRACT_V1,
    "contract_version": "realtime_persistent.v2",
    "realtime_coordinator": "realtime_episode_v5",
    "wakeup_policy": FORMAL_WAKEUP_POLICY_V2,
}

AGENTIC_FORMAL_RUN_CONTRACT_V1: JsonDict = {
    "contract_version": "agentic_persistent.v1",
    "required_interaction_mode": "logical_persistent",
    "required_model_count_per_shard": 1,
    "minimum_pass_k": 1,
    "minimum_max_workers": 1,
    "maximum_max_workers": 32,
    "required_prompt_mode": "strict",
    "required_seed_mode": "scenario",
    "required_scheduler_mode": "global",
    "required_temperature": 0.0,
    "requires_explicit_model_capabilities": True,
    "agentic_profile": AGENTIC_PROFILE_V1,
    "save_trajectories": True,
    "required_construct_contract": "operational_agency.v1",
    "shard_merge_key": "formal_treatment_family_sha256",
    "realtime_formal_contract": REALTIME_FORMAL_CONTRACT_V1,
}
AGENTIC_FORMAL_RUN_CONTRACT_V2: JsonDict = {
    **AGENTIC_FORMAL_RUN_CONTRACT_V1,
    "wakeup_policy": FORMAL_WAKEUP_POLICY_V2,
    "realtime_formal_contract": REALTIME_FORMAL_CONTRACT_V2,
}
_V058_REALTIME_FORMAL_CONTRACT_V1: JsonDict = {
    **REALTIME_FORMAL_CONTRACT_V1,
    "scorecard_version": "realtime-diagnostics/1.5",
    "diagnostic_schema_version": "realtime-diagnostics/1.5",
}
_V058_AGENTIC_FORMAL_RUN_CONTRACT_V1: JsonDict = {
    **AGENTIC_FORMAL_RUN_CONTRACT_V1,
    "realtime_formal_contract": _V058_REALTIME_FORMAL_CONTRACT_V1,
}
_V058_LEGACY_RUNTIME_IDENTITY = (
    "3b24e19d3d52e0e38e386188b325e48a99f838b0c4ec14b78bc2dbaa793601bd",
    "25dbc04fcf0d847e31445882b06a8613a66bf6fc136a18018e76915a659fe8fb",
)

PIPELINE_STAGE_HASH_FIELDS = {
    "preflight": "preflight_sha256",
    "behavioral": "behavioral_sha256",
    "source_consumption": "source_consumption_sha256",
    "task_contracts": "task_contracts_sha256",
    "complexity": "complexity_sha256",
    "observed_reference_depth": "observed_reference_depth_sha256",
    "strategy_depth": "strategy_depth_sha256",
    "source_grounded": "source_grounded_sha256",
    "agentic_contract": "agentic_contract_sha256",
    "materialize_core": "core_selection_sha256",
    "release_coverage": "release_coverage_sha256",
    "readiness": "readiness_sha256",
}

PIPELINE_STAGE_FILES = {
    "preflight": "protocol2_v21_working_set_preflight.json",
    "behavioral": "behavioral_calibration_protocol2_v21.json",
    "source_consumption": "source_consumption_protocol2_v21.json",
    "task_contracts": "task_contracts_protocol2_v21.json",
    "complexity": "complexity_protocol2_v21.json",
    "observed_reference_depth": "observed_reference_depth_protocol2_v21.json",
    "strategy_depth": "strategy_depth_protocol2_v21.json",
    "source_grounded": "source_grounded_protocol2_v21.json",
    "agentic_contract": "agentic_core_contract_protocol2_v21.json",
    "materialize_core": "refined_core_selection_protocol2_v21.json",
    "release_coverage": "release_coverage_protocol2_v21.json",
    "readiness": "protocol2_v21_core_readiness.json",
}
FORMAL_RUNTIME_BUNDLE_NAME = "formal_runtime_bundle.json"
FORMAL_RUNTIME_BUNDLE_SCHEMA = "operate-formal-runtime-bundle-v1"


def _load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scenario_id(row: JsonDict) -> str:
    return str(row.get("scenario_id") or row.get("id") or "")


def _duplicates(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicated: set[str] = set()
    for scenario_id in ids:
        if scenario_id in seen:
            duplicated.add(scenario_id)
        seen.add(scenario_id)
    return sorted(duplicated)


def _explicit_provenance_gap(source_lock: JsonDict) -> bool:
    return source_lock.get("provenance_complete") is False


def _has_modern_provenance(row: JsonDict) -> bool:
    provenance = (
        row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    )
    source_lock = (
        row.get("source_lock") if isinstance(row.get("source_lock"), dict) else {}
    )
    if _explicit_provenance_gap(source_lock):
        return False

    required = ("data_source", "license", "lock_strategy")
    if not all(provenance.get(key) for key in required):
        return False

    lock_strategy = str(
        provenance.get("lock_strategy") or source_lock.get("lock_strategy") or ""
    )
    inherited = "inherited" in lock_strategy or bool(
        source_lock.get("inherited_release")
    )
    file_hash = bool(
        source_lock.get("file_sha256")
        or source_lock.get("sha256")
        or source_lock.get("source_file_sha256")
        or source_lock.get("file_sha256s")
    )
    return inherited or file_hash


def _has_legacy_source_lock(row: JsonDict) -> bool:
    source_lock = (
        row.get("source_lock") if isinstance(row.get("source_lock"), dict) else {}
    )
    issues = set(source_lock.get("provenance_issues") or [])
    non_fatal_legacy_issues = {"unmapped_provenance_source"}
    if _explicit_provenance_gap(source_lock) and not issues:
        return False
    if not issues.issubset(non_fatal_legacy_issues):
        return False

    required = ("data_source", "license", "lock_strategy", "files")
    if not all(source_lock.get(key) for key in required):
        return False

    lock_strategy = str(source_lock.get("lock_strategy") or "")
    has_file_hash_evidence = bool(
        source_lock.get("file_sha256")
        or source_lock.get("sha256")
        or source_lock.get("source_file_sha256")
        or source_lock.get("file_sha256s")
        or source_lock.get("expected_file_sha256s")
    )
    hash_locked_strategy = (
        ("sha256" in lock_strategy or "hash" in lock_strategy)
        and source_lock.get("data_source") == "sumo_ingolstadt_365"
        and "unmapped_provenance_source" not in issues
    )
    if hash_locked_strategy and not has_file_hash_evidence:
        return False

    has_lock_evidence = bool(
        source_lock.get("commit")
        or has_file_hash_evidence
        or "inherited" in lock_strategy
    )
    return has_lock_evidence


def _has_provenance(row: JsonDict, *, allow_legacy_source_lock: bool) -> bool:
    if _has_modern_provenance(row):
        return True
    return allow_legacy_source_lock and _has_legacy_source_lock(row)


def _requires_suite_sha256(release_id: str) -> bool:
    return True


def _sha256_check(
    entry: JsonDict, path: Path, *, sha256_required: bool
) -> tuple[bool, bool]:
    manifest_sha = entry.get("sha256")
    if not manifest_sha:
        return not sha256_required, sha256_required
    return manifest_sha == _sha256(path), True


def _resolve_suite_path(
    release: Path, entry: JsonDict, default_name: str
) -> tuple[Path, bool]:
    release_root = release.resolve()
    suite_path = (release / str(entry.get("path") or default_name)).resolve()
    try:
        suite_path.relative_to(release_root)
    except ValueError:
        return suite_path, False
    return suite_path, True


def _provenance_summary(
    rows: list[JsonDict], *, allow_legacy_source_lock: bool
) -> JsonDict:
    incomplete = [
        _scenario_id(row)
        for row in rows
        if not _has_provenance(row, allow_legacy_source_lock=allow_legacy_source_lock)
    ]
    return {
        "complete": len(rows) - len(incomplete),
        "incomplete": len(incomplete),
        "incomplete_ids": incomplete[:25],
    }


def _manifest_suite_entry(manifest: JsonDict, key: str) -> JsonDict:
    entry = manifest.get(key)
    return entry if isinstance(entry, dict) else {}


def _release_version_from_id(release_id: str) -> tuple[int, int, int]:
    """Parse the active OPERATE release namespace; reject unrelated ids."""
    if not release_id.startswith("operate_v"):
        return (0, 0, 0)
    parts = release_id.removeprefix("operate_v").split("_")
    if len(parts) != 3:
        return (0, 0, 0)
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError:
        return (0, 0, 0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _agentic_contract_required(manifest: JsonDict) -> bool:
    release_requires_agentic = _release_version(manifest) >= (0, 57, 0)
    run_contract = manifest.get("formal_run_contract")
    batch_contract = manifest.get("formal_batch_contract")
    return bool(
        release_requires_agentic
        or "formal_realtime_batch_contract" in manifest
        or (
            isinstance(run_contract, dict)
            and run_contract.get("contract_version") == "agentic_persistent.v1"
        )
        or (
            isinstance(batch_contract, dict)
            and batch_contract.get("contract_version") == "agentic_persistent.v1"
        )
    )


def _release_version(manifest: JsonDict) -> tuple[int, int, int]:
    return _release_version_from_id(str(manifest.get("release_id") or ""))


def _formal_contracts_for_release(manifest: JsonDict) -> tuple[JsonDict, JsonDict]:
    """Select the immutable formal contract family for one release line."""

    manifest_runtime_identity = (
        manifest.get("implementation_tree_sha256"),
        manifest.get("core_release_pipeline_sha256"),
    )
    if (
        manifest.get("release_id") == "operate_v0_58_0"
        and manifest_runtime_identity == _V058_LEGACY_RUNTIME_IDENTITY
    ):
        return (
            _V058_AGENTIC_FORMAL_RUN_CONTRACT_V1,
            _V058_REALTIME_FORMAL_CONTRACT_V1,
        )
    if _release_version(manifest) >= (0, 61, 0):
        return AGENTIC_FORMAL_RUN_CONTRACT_V2, REALTIME_FORMAL_CONTRACT_V2
    return AGENTIC_FORMAL_RUN_CONTRACT_V1, REALTIME_FORMAL_CONTRACT_V1


def _release_identity_closure_required(manifest: JsonDict, core: JsonDict) -> bool:
    if core.get("release_id") is not None or core.get("scoring_version") is not None:
        return True
    return _release_version(manifest) >= (0, 55, 0)


def _resolve_repo_artifact(
    raw_path: object, *, artifact_root: Path | None = None
) -> tuple[Path, bool]:
    repo = (artifact_root or _repo_root()).resolve()
    text = str(raw_path or "")
    candidate = Path(text)
    if not text or candidate.is_absolute():
        return candidate, False
    resolved = (repo / candidate).resolve()
    try:
        resolved.relative_to(repo)
    except ValueError:
        return resolved, False
    return resolved, True


def _safe_load_json(path: Path) -> JsonDict:
    try:
        payload = _load_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _scenario_identity_for_binding(row: JsonDict) -> tuple[str, str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
        str(row.get("path") or ""),
    )


def _release_file_binding(
    release: Path, binding: object, *, expected_name: str
) -> tuple[Path, bool]:
    if not isinstance(binding, dict):
        return Path(), False
    relative = Path(str(binding.get("path") or ""))
    if (
        relative.as_posix() != expected_name
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        return Path(), False
    path = (release / relative).resolve()
    return path, bool(
        path.parent == release.resolve()
        and not path.is_symlink()
        and path.is_file()
        and _sha256(path) == binding.get("sha256")
    )


def _formal_runtime_identity(
    release: Path, manifest: JsonDict
) -> tuple[JsonDict, bool]:
    completion = manifest.get("formal_evaluation_completion")
    replay = manifest.get("protocol21_replay")
    candidate = manifest.get("candidate_closure")
    backend = manifest.get("backend_runtime_closure")
    if not all(
        isinstance(value, dict) for value in (completion, replay, candidate, backend)
    ):
        return {}, False
    assert isinstance(completion, dict)
    assert isinstance(replay, dict)
    assert isinstance(candidate, dict)
    assert isinstance(backend, dict)
    runtime_path, runtime_ok = _release_file_binding(
        release,
        manifest.get("formal_runtime_bundle"),
        expected_name=FORMAL_RUNTIME_BUNDLE_NAME,
    )
    core_path, core_ok = _release_file_binding(
        release, manifest.get("core_suite"), expected_name="core_suite.json"
    )
    candidate_path, candidate_ok = _release_file_binding(
        release, candidate, expected_name="candidate_closure.json"
    )
    backend_path, backend_ok = _release_file_binding(
        release, backend, expected_name="backend_runtime_closure.json"
    )
    source_path = release / "protocol21_source_suite.json"
    evidence_path = release / "protocol21_public_evidence_bundle.json"
    source_ok = bool(
        source_path.is_file()
        and not source_path.is_symlink()
        and _sha256(source_path) == replay.get("source_suite_sha256")
    )
    evidence_ok = bool(
        evidence_path.is_file()
        and not evidence_path.is_symlink()
        and _sha256(evidence_path) == replay.get("evidence_bundle_sha256")
    )
    if not all((runtime_ok, core_ok, candidate_ok, backend_ok, source_ok, evidence_ok)):
        return {}, False
    runtime = _safe_load_json(runtime_path)
    evidence = _safe_load_json(evidence_path)
    candidate_payload = _safe_load_json(candidate_path)
    backend_payload = _safe_load_json(backend_path)
    identity: JsonDict = {
        "release_id": str(manifest.get("release_id") or ""),
        "formal_manifest_sha256": str(
            completion.get("input_release_manifest_sha256") or ""
        ),
        "formal_runtime_bundle_sha256": _sha256(runtime_path),
        "formal_core_suite_sha256": _sha256(core_path),
        "formal_source_suite_sha256": _sha256(source_path),
        "formal_public_evidence_sha256": _sha256(evidence_path),
        "formal_public_evidence_binding_root_sha256": str(
            evidence.get("binding_root_sha256") or ""
        ),
        "formal_candidate_closure_sha256": _sha256(candidate_path),
        "formal_candidate_closure_identity_sha256": _canonical_sha256(
            candidate_payload.get("identity_set_sha256")
        ),
        "formal_backend_runtime_closure_sha256": _sha256(backend_path),
        "formal_backend_runtime_closure_identity_sha256": str(
            backend_payload.get("identity_sha256") or ""
        ),
        "implementation_tree_sha256": str(
            manifest.get("implementation_tree_sha256") or ""
        ),
        "formal_core_release_pipeline_sha256": str(
            manifest.get("core_release_pipeline_sha256") or ""
        ),
        "formal_release_tooling_sha256": str(
            manifest.get("release_tooling_sha256") or ""
        ),
    }
    runtime_evidence = runtime.get("public_evidence") or {}
    internally_bound = bool(
        runtime.get("release_id") == identity["release_id"]
        and runtime.get("implementation_tree_sha256")
        == identity["implementation_tree_sha256"]
        and runtime.get("core_release_pipeline_sha256")
        == identity["formal_core_release_pipeline_sha256"]
        and runtime.get("release_tooling_sha256")
        == identity["formal_release_tooling_sha256"]
        and isinstance(runtime_evidence, dict)
        and runtime_evidence.get("sha256") == identity["formal_public_evidence_sha256"]
        and runtime_evidence.get("binding_root_sha256")
        == identity["formal_public_evidence_binding_root_sha256"]
        and candidate.get("identity_set_sha256")
        == candidate_payload.get("identity_set_sha256")
        and backend.get("identity_sha256")
        == identity["formal_backend_runtime_closure_identity_sha256"]
    )
    return identity, bool(
        internally_bound
        and identity["release_id"] == release.name
        and all(
            _valid_sha256(value)
            for name, value in identity.items()
            if name != "release_id"
        )
    )


def _indexed_tree_payload(root: Path) -> tuple[JsonDict, bool]:
    if root.is_symlink() or not root.is_dir():
        return {}, False
    files: list[JsonDict] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                return {}, False
        for name in names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if relative == FORMAL_RESULT_TREE_INDEX_NAME:
                continue
            if path.is_symlink() or not path.is_file():
                return {}, False
            files.append(
                {
                    "path": relative,
                    "sha256": _sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    files.sort(key=lambda item: str(item["path"]))
    payload: JsonDict = {
        "schema_version": FORMAL_RESULT_TREE_INDEX_SCHEMA,
        "files": files,
    }
    payload["root_sha256"] = _canonical_sha256(payload)
    return payload, True


def _strict_batch_path(raw: object, *, root: Path) -> tuple[Path, bool]:
    text = str(raw or "")
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        return relative, False
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return path, False
    return path, True


def _strict_repo_artifact(raw: object, *, artifact_root: Path) -> tuple[Path, bool]:
    text = str(raw or "")
    relative = Path(text)
    root = artifact_root.resolve()
    if not text or relative.is_absolute() or ".." in relative.parts:
        return relative, False
    lexical = root / relative
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return lexical, False
    try:
        lexical.resolve().relative_to(root)
    except ValueError:
        return lexical, False
    return lexical, True


def _logical_rows_for_validation(
    rows: list[JsonDict], *, batch_root: Path
) -> list[JsonDict] | None:
    resolved_rows = deepcopy(rows)
    for row in resolved_rows:
        summary = row.get("trajectory_summary")
        if not isinstance(summary, dict):
            return None
        prefix, prefix_ok = _strict_batch_path(
            summary.get("trajectory_path"), root=batch_root
        )
        if not prefix_ok:
            return None
        summary["trajectory_path"] = str(prefix)
        for value in summary.values():
            if not isinstance(value, dict) or "path" not in value:
                continue
            artifact_path, artifact_ok = _strict_batch_path(
                value.get("path"), root=batch_root
            )
            if not artifact_ok or not artifact_path.is_file():
                return None
            value["path"] = str(artifact_path)
    return resolved_rows


def _batch_paths_portable(path: Path, payload: JsonDict, *, mode: str) -> bool:
    root = path.parent.resolve()
    if mode == "logical_persistent":
        artifacts = payload.get("published_artifacts") or {}
        episodes_path, episodes_ok = _strict_batch_path(
            (artifacts.get("episodes") or {}).get("path"), root=root
        )
        _leaderboard_path, leaderboard_ok = _strict_batch_path(
            (artifacts.get("leaderboard") or {}).get("path"), root=root
        )
        if not episodes_ok or not leaderboard_ok or not episodes_path.is_file():
            return False
        try:
            rows = [
                json.loads(line)
                for line in episodes_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        return _logical_rows_for_validation(rows, batch_root=root) is not None
    if mode != "realtime_persistent":
        return False
    artifacts = payload.get("artifacts") or {}
    for name in ("episodes_journal", "episodes", "realtime_scorecard", "leaderboard"):
        artifact_path, valid = _strict_batch_path(
            (artifacts.get(name) or {}).get("path"), root=root
        )
        if not valid or not artifact_path.is_file():
            return False
    for binding in artifacts.get("episode_artifacts") or []:
        if not isinstance(binding, dict):
            return False
        artifact_path, valid = _strict_batch_path(
            binding.get("artifact_path"), root=root
        )
        if not valid or not artifact_path.is_file():
            return False
    episodes_path, _ = _strict_batch_path(
        (artifacts.get("episodes") or {}).get("path"), root=root
    )
    try:
        rows = [
            json.loads(line)
            for line in episodes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    for row in rows:
        if not isinstance(row, dict):
            return False
        artifact_path, artifact_valid = _strict_batch_path(
            row.get("artifact_path"), root=root
        )
        if not artifact_valid or not artifact_path.is_file():
            return False
        if row.get("trajectory_dir"):
            trajectory_dir, trajectory_valid = _strict_batch_path(
                row.get("trajectory_dir"), root=root
            )
            if not trajectory_valid or not trajectory_dir.is_dir():
                return False
    return True


def _formal_result_binding_valid(
    release: Path,
    binding: object,
    *,
    artifact_root: Path,
    expected_model: str,
    expected_mode: str,
) -> tuple[JsonDict, bool, bool]:
    if not isinstance(binding, dict):
        return {}, False, False
    path, contained = _strict_repo_artifact(
        binding.get("path"), artifact_root=artifact_root
    )
    result_root = (release / "formal_results").resolve()
    try:
        relative = path.resolve().relative_to(result_root)
    except ValueError:
        return {}, False, False
    if (
        not contained
        or len(relative.parts) != 4
        or relative.name != "RUN_MANIFEST.json"
        or path.is_symlink()
        or not path.is_file()
        or _sha256(path) != binding.get("sha256")
    ):
        return {}, False, False
    root = path.parent.resolve()
    index_path, index_contained = _strict_repo_artifact(
        binding.get("tree_index_path"), artifact_root=artifact_root
    )
    if not (
        index_contained
        and index_path == root / FORMAL_RESULT_TREE_INDEX_NAME
        and index_path.is_file()
        and not index_path.is_symlink()
        and _sha256(index_path) == binding.get("tree_index_sha256")
    ):
        return {}, False, False
    index = _safe_load_json(index_path)
    actual_index, tree_safe = _indexed_tree_payload(root)
    payload = _safe_load_json(path)
    treatment = str(binding.get("treatment_sha256") or "")
    if expected_mode == "logical_persistent":
        payload_treatment_valid = bool(
            payload.get("models") == [expected_model]
            and (payload.get("agent_treatment_sha256_by_model") or {}).get(
                expected_model
            )
            == treatment
        )
    else:
        payload_treatment_valid = bool(
            payload.get("model") == expected_model
            and payload.get("batch_treatment_sha256") == treatment
        )
    tree_valid = bool(
        tree_safe
        and index == actual_index
        and index.get("root_sha256") == binding.get("tree_root_sha256")
        and root.name == binding.get("tree_root_sha256")
        and root.parent.name == treatment
        and _valid_sha256(treatment)
        and binding.get("model") == expected_model
        and binding.get("interaction_mode") == expected_mode
        and payload_treatment_valid
    )
    paths_valid = bool(
        tree_valid and _batch_paths_portable(path, payload, mode=expected_mode)
    )
    return payload, tree_valid, paths_valid


def _formal_publication_checks(
    release: Path,
    manifest: JsonDict,
    *,
    artifact_root: Path,
) -> dict[str, bool]:
    published = manifest.get("status") in {"formal_evaluation_complete", "released"}
    names = {
        "agentic_formal_completion_identity_valid": True,
        "agentic_formal_result_tree_valid": True,
        "agentic_formal_result_paths_portable": True,
        "agentic_formal_distribution_receipt_valid": True,
    }
    if not published:
        return names
    completion = manifest.get("formal_evaluation_completion")
    evidence = manifest.get("formal_evidence")
    if not isinstance(completion, dict) or not isinstance(evidence, dict):
        return dict.fromkeys(names, False)
    identity, identity_valid = _formal_runtime_identity(release, manifest)
    model = str(completion.get("model") or "")
    logical, logical_tree_valid, logical_paths_valid = _formal_result_binding_valid(
        release,
        evidence.get("logical_batch_manifest"),
        artifact_root=artifact_root,
        expected_model=model,
        expected_mode="logical_persistent",
    )
    realtime, realtime_tree_valid, realtime_paths_valid = _formal_result_binding_valid(
        release,
        evidence.get("realtime_batch_manifest"),
        artifact_root=artifact_root,
        expected_model=model,
        expected_mode="realtime_persistent",
    )
    logical_actual = {key: logical.get(key) for key in identity}
    realtime_identity = realtime.get("batch_treatment_identity") or {}
    runtime_binding = (
        realtime_identity.get("formal_runtime_binding")
        if isinstance(realtime_identity, dict)
        else {}
    )
    runtime_binding = runtime_binding if isinstance(runtime_binding, dict) else {}
    realtime_actual = {
        **runtime_binding,
        "release_id": realtime_identity.get("formal_release_id"),
        "formal_manifest_sha256": realtime_identity.get("formal_manifest_sha256"),
        "implementation_tree_sha256": realtime_identity.get(
            "implementation_tree_sha256"
        ),
        "formal_core_release_pipeline_sha256": runtime_binding.get(
            "core_release_pipeline_sha256"
        ),
        "formal_backend_runtime_closure_identity_sha256": runtime_binding.get(
            "backend_runtime_closure_identity_sha256"
        ),
        "formal_release_tooling_sha256": runtime_binding.get("release_tooling_sha256"),
    }
    completion_valid = bool(
        identity_valid
        and completion.get("schema_version")
        == "operate-formal-evaluation-completion-v2"
        and completion.get("runtime_identity") == identity
        and completion.get("logical_batch_manifest")
        == evidence.get("logical_batch_manifest")
        and completion.get("realtime_batch_manifest")
        == evidence.get("realtime_batch_manifest")
        and model
        and logical_actual == identity
        and {key: realtime_actual.get(key) for key in identity} == identity
    )
    distribution_receipt_valid = True
    if _release_version(manifest) >= (0, 61, 0):
        receipt_path = release / FORMAL_DISTRIBUTION_RECEIPT_NAME
        receipt = _safe_load_json(receipt_path)
        receipt_without_hash = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        expected_receipt_keys = {
            "schema_version",
            "release_id",
            "hf_repo_id",
            "visibility",
            "revision",
            "verification",
            "bundle_manifest_sha256",
            "release_manifest_sha256",
            "formal_evidence_archive",
            "formal_evidence_archive_sha256",
            "formal_result_tree_roots",
            "receipt_sha256",
        }
        revision = str(receipt.get("revision") or "")
        logical_binding = evidence.get("logical_batch_manifest")
        realtime_binding = evidence.get("realtime_batch_manifest")
        expected_roots = {
            "logical_persistent": str(
                logical_binding.get("tree_root_sha256")
                if isinstance(logical_binding, dict)
                else ""
            ),
            "realtime_persistent": str(
                realtime_binding.get("tree_root_sha256")
                if isinstance(realtime_binding, dict)
                else ""
            ),
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        distribution_receipt_valid = bool(
            receipt_path.is_file()
            and not receipt_path.is_symlink()
            and set(receipt) == expected_receipt_keys
            and receipt.get("schema_version") == FORMAL_DISTRIBUTION_RECEIPT_SCHEMA
            and receipt.get("release_id") == manifest.get("release_id")
            and isinstance(receipt.get("hf_repo_id"), str)
            and bool(receipt.get("hf_repo_id"))
            and receipt.get("visibility") == "private"
            and receipt.get("verification") == "private_cas_exact_snapshot_v1"
            and 40 <= len(revision) <= 64
            and all(char in "0123456789abcdef" for char in revision)
            and set(revision) != {"0"}
            and _valid_sha256(receipt.get("bundle_manifest_sha256"))
            and receipt.get("release_manifest_sha256")
            == hashlib.sha256(manifest_bytes).hexdigest()
            and isinstance(receipt.get("formal_evidence_archive"), str)
            and bool(receipt.get("formal_evidence_archive"))
            and _valid_sha256(receipt.get("formal_evidence_archive_sha256"))
            and receipt.get("formal_result_tree_roots") == expected_roots
            and receipt.get("receipt_sha256")
            == _canonical_sha256(receipt_without_hash)
        )
    return {
        "agentic_formal_completion_identity_valid": completion_valid,
        "agentic_formal_result_tree_valid": bool(
            logical_tree_valid and realtime_tree_valid
        ),
        "agentic_formal_result_paths_portable": bool(
            logical_paths_valid and realtime_paths_valid
        ),
        "agentic_formal_distribution_receipt_valid": distribution_receipt_valid,
    }


def _valid_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _canonical_payload_sha256(payload: JsonDict) -> str:
    canonical = json.dumps(
        {key: value for key, value in payload.items() if key != "binding_root_sha256"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _canonical_sha256(payload: object) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _formal_tool_choice_matches(value: object, profile: JsonDict) -> bool:
    expected = profile.get("tool_choice")
    return isinstance(expected, str) and value == expected


def _formal_wakeup_policy_valid(
    contract: JsonDict, *identities: object
) -> bool:
    expected = contract.get("wakeup_policy")
    if expected is None:
        return True
    return bool(
        isinstance(expected, dict)
        and identities
        and all(
            isinstance(identity, dict)
            and identity.get("wakeup_policy") == expected
            for identity in identities
        )
    )


def _contained_batch_artifact(
    raw_path: object, *, batch_root: Path
) -> tuple[Path, bool]:
    text = str(raw_path or "")
    if not text:
        return Path(), False
    candidate = Path(text)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (batch_root / candidate).resolve()
    )
    try:
        resolved.relative_to(batch_root.resolve())
    except ValueError:
        return resolved, False
    return resolved, True


_LOGICAL_FORMAL_RUNTIME_IDENTITY_FIELDS = (
    "formal_release_id",
    "formal_manifest_sha256",
    "formal_release_tooling_sha256",
    "formal_readiness_sha256",
    "formal_core_release_pipeline_sha256",
    "formal_backend_runtime_closure_identity_sha256",
    "formal_runtime_bundle_sha256",
    "formal_core_suite_sha256",
    "formal_source_suite_sha256",
    "formal_public_evidence_sha256",
    "formal_public_evidence_binding_root_sha256",
    "formal_candidate_closure_sha256",
    "formal_candidate_closure_identity_sha256",
    "formal_backend_runtime_closure_sha256",
)


def _logical_provider_route_sha256(
    base_url: str | None, responses_base_url: str | None
) -> str:
    """Rebuild the generator's secret-free route identity from public URLs."""

    def projection(value: str | None) -> JsonDict | None:
        if not value:
            return None
        parsed = urlsplit(value)
        behavior_fields = {
            "api-version",
            "deployment",
            "model",
            "region",
            "route",
            "variant",
            "version",
        }
        return {
            "scheme": parsed.scheme,
            "host": parsed.hostname,
            "port": parsed.port,
            "path": parsed.path,
            "query": sorted(
                (
                    str(name),
                    str(raw_value)
                    if str(name).lower() in behavior_fields
                    else "[redacted]",
                )
                for name, raw_value in parse_qsl(
                    parsed.query, keep_blank_values=True
                )
            ),
        }

    return _canonical_sha256(
        {
            "base_url": projection(base_url),
            "responses_base_url": projection(responses_base_url),
            "extra_headers": [],
        }
    )


def _logical_profile_identity_from_manifest(
    payload: JsonDict, *, model: str
) -> JsonDict | None:
    """Reconstruct the canonical logical profile from independent run fields."""

    context_windows = payload.get("model_context_window_tokens_by_model")
    output_limits = payload.get("model_max_output_tokens_by_model")
    tool_choice_support = payload.get("tool_choice_supported_by_model")
    base_url = payload.get("base_url")
    responses_base_url = payload.get("responses_base_url")
    if not bool(
        isinstance(context_windows, dict)
        and isinstance(output_limits, dict)
        and isinstance(tool_choice_support, dict)
        and isinstance(base_url, (str, type(None)))
        and isinstance(responses_base_url, (str, type(None)))
        and payload.get("interaction_mode") == "logical_persistent"
        and payload.get("prompt_mode") == "strict"
        and payload.get("harness") == "direct_api"
        and payload.get("token_count_method") == TOKEN_COUNT_METHOD_UTF8_BYTES
        and payload.get("token_count_version") == TOKEN_COUNT_VERSION_V1
        and payload.get("evaluation_implementation_fingerprint")
        == EVALUATION_IMPLEMENTATION_FINGERPRINT
        and payload.get("wakeup_policy") == FORMAL_WAKEUP_POLICY_V2
    ):
        return None
    api_mode = payload.get("api_mode")
    provider = (
        "azure"
        if base_url and api_mode == "azure"
        else ("openai_compatible" if base_url else "openai")
    )
    return {
        "schema_version": "agent_treatment_v1",
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_version": payload.get("api_version"),
        "api_version_env": "OPERATE_API_VERSION",
        "responses_base_url": responses_base_url,
        "responses_base_url_env": "OPERATE_RESPONSES_API_BASE_URL",
        "private_provider_route_sha256": _logical_provider_route_sha256(
            base_url, responses_base_url
        ),
        "api_mode": api_mode,
        "stream_chat_completions": payload.get("stream_chat_completions"),
        "temperature": payload.get("temperature"),
        "max_tokens": payload.get("max_tokens"),
        "model_context_window_tokens": context_windows.get(model),
        "model_max_output_tokens": output_limits.get(model),
        "token_count_method": TOKEN_COUNT_METHOD_UTF8_BYTES,
        "token_count_version": TOKEN_COUNT_VERSION_V1,
        "timeout_s": payload.get("provider_timeout_s"),
        "max_consecutive_provider_failures": payload.get(
            "max_consecutive_provider_failures"
        ),
        "provider_failure_policy": payload.get("provider_failure_policy"),
        "provider_rpm_limit": payload.get("provider_rpm_limit"),
        "provider_rpd_limit": payload.get("provider_rpd_limit"),
        "provider_rate_limit_scope": (
            str(payload.get("provider_rate_limit_scope") or "").strip() or None
        ),
        "prompt_mode": "strict",
        "prompt_contract_sha256": prompt_contract_sha256(
            "logical_persistent", "strict"
        ),
        "interaction_mode": "logical_persistent",
        "persistent_history_max_messages": payload.get(
            "persistent_history_max_messages"
        ),
        "persistent_context_max_chars": payload.get("persistent_context_max_chars"),
        "persistent_memory_max_items": payload.get("persistent_memory_max_items"),
        "tool_choice": payload.get("tool_choice"),
        "tool_choice_supported": tool_choice_support.get(model),
        "reasoning_effort": payload.get("reasoning_effort"),
        "protocol_repair_max_tokens": payload.get("protocol_repair_max_tokens"),
        "allow_insecure_http": False,
        "extra_header_names": [],
        "harness": "direct_api",
        "prompt_context_compiler_binding": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        "tool_schema_binding": "decision_envelope.available_tool_schema_sha256",
        "wakeup_policy": deepcopy(FORMAL_WAKEUP_POLICY_V2),
    }


def _logical_treatment_identity_valid(payload: JsonDict) -> bool:
    """Recompute the formal logical treatment from its canonical inputs."""
    models = payload.get("models")
    profile_identities = payload.get("agent_profile_identity_by_model")
    profile_hashes = payload.get("agent_profile_sha256_by_model")
    treatment_hashes = payload.get("agent_treatment_sha256_by_model")
    if not bool(
        payload.get("formal_run") is True
        and payload.get("interaction_mode") == "logical_persistent"
        and payload.get("agent_profile_schema_version") == "agent_treatment_v1"
        and payload.get("agent_treatment_schema_version")
        == "formal_logical_treatment_v1"
        and isinstance(models, list)
        and len(models) == 1
        and isinstance(profile_identities, dict)
        and isinstance(profile_hashes, dict)
        and isinstance(treatment_hashes, dict)
        and set(profile_identities) == set(models)
        and set(profile_hashes) == set(models)
        and set(treatment_hashes) == set(models)
    ):
        return False
    assert isinstance(models, list)
    assert isinstance(profile_identities, dict)
    assert isinstance(profile_hashes, dict)
    assert isinstance(treatment_hashes, dict)
    model = models[0]
    profile_identity = profile_identities.get(model)
    if not isinstance(profile_identity, dict) or not bool(
        profile_identity.get("schema_version") == "agent_treatment_v1"
        and profile_identity.get("model") == model
        and profile_identity.get("interaction_mode") == "logical_persistent"
    ):
        return False
    expected_profile_identity = _logical_profile_identity_from_manifest(
        payload, model=model
    )
    if expected_profile_identity is None or profile_identity != expected_profile_identity:
        return False
    profile_sha256 = _canonical_sha256(profile_identity)
    if profile_hashes.get(model) != profile_sha256:
        return False
    runtime_identity = {
        field: payload.get(field)
        for field in _LOGICAL_FORMAL_RUNTIME_IDENTITY_FIELDS
        if payload.get(field) is not None
    }
    required_runtime_fields = {
        "formal_release_id",
        "formal_manifest_sha256",
        "formal_release_tooling_sha256",
        "formal_readiness_sha256",
        "formal_core_release_pipeline_sha256",
        "formal_backend_runtime_closure_identity_sha256",
    }
    release_id = str(runtime_identity.get("formal_release_id") or "")
    if (
        not required_runtime_fields <= set(runtime_identity)
        or not release_id
        or Path(release_id).name != release_id
        or any(
            not _valid_sha256(runtime_identity.get(field))
            for field in required_runtime_fields - {"formal_release_id"}
        )
    ):
        return False
    expected = _canonical_sha256(
        {
            "schema_version": "formal_logical_treatment_v1",
            "interaction_mode": "logical_persistent",
            "agent_profile_sha256": profile_sha256,
            "formal_runtime_binding": runtime_identity,
            "implementation_tree_sha256": payload.get(
                "implementation_tree_sha256"
            ),
        }
    )
    return bool(
        _valid_sha256(payload.get("implementation_tree_sha256"))
        and treatment_hashes.get(model) == expected
    )


def _logical_published_manifest_valid(
    payload: JsonDict,
    *,
    manifest_path: Path,
    tree: object,
    suite_sha256: object,
    suite_rows: list[JsonDict],
    logical_contract: JsonDict,
) -> bool:
    n_scenarios = len(suite_rows)
    coverage = payload.get("coverage") or {}
    models = payload.get("models") or []
    treatment_hashes = payload.get("agent_treatment_sha256_by_model") or {}
    context_windows = payload.get("model_context_window_tokens_by_model") or {}
    output_limits = payload.get("model_max_output_tokens_by_model") or {}
    pass_k = payload.get("pass_k")
    expected = n_scenarios * int(pass_k or 0)
    if not bool(
        payload.get("formal_run") is True
        and payload.get("batch_state") == "final"
        and payload.get("interaction_mode") == "logical_persistent"
        and payload.get("prompt_mode") == "strict"
        and payload.get("temperature") == 0.0
        and payload.get("seed_mode") == "scenario"
        and payload.get("scheduler_mode") == "global"
        and payload.get("save_trajectories") is True
        and _formal_tool_choice_matches(
            payload.get("tool_choice"),
            logical_contract.get("agentic_profile") or {},
        )
        and payload.get("stream_chat_completions") is True
        and _formal_wakeup_policy_valid(logical_contract, payload)
        and all(
            payload.get(key) == value
            for key, value in (logical_contract.get("agentic_profile") or {}).items()
        )
        and isinstance(payload.get("max_workers_requested"), int)
        and logical_contract.get("cardinality", {}).get("workers", {}).get("minimum", 1)
        <= payload.get("max_workers_requested")
        <= logical_contract.get("cardinality", {}).get("workers", {}).get("maximum", 32)
        and payload.get("max_workers_effective") == payload.get("max_workers_requested")
        and payload.get("implementation_tree_sha256") == tree
        and payload.get("implementation_tree_stable") is True
        and payload.get("formal_runtime_binding_stable") is True
        and payload.get("suite_manifest_sha256") == suite_sha256
        and payload.get("n_scenarios") == n_scenarios
        and isinstance(pass_k, int)
        and not isinstance(pass_k, bool)
        and pass_k >= 1
        and isinstance(models, list)
        and len(models) == 1
        and _logical_treatment_identity_valid(payload)
        and isinstance(treatment_hashes, dict)
        and set(treatment_hashes) == set(models)
        and all(_valid_sha256(value) for value in treatment_hashes.values())
        and set(context_windows) == set(models)
        and set(output_limits) == set(models)
        and all(
            isinstance(context_windows.get(model), int)
            and context_windows.get(model) > 0
            and isinstance(output_limits.get(model), int)
            and output_limits.get(model) >= payload.get("max_tokens")
            for model in models
        )
        and payload.get("n_episodes_total") == expected
        and payload.get("n_episodes_ok") == expected
        and payload.get("n_episodes_error") == 0
        and coverage.get("expected_total") == expected
        and coverage.get("n_scenarios") == n_scenarios
        and coverage.get("pass_k") == pass_k
        and coverage.get("is_partial_batch") is False
        and (payload.get("leaderboard_eligibility") or {}).get("eligible") is True
        and payload.get("leaderboard_eligible") is True
    ):
        return False
    published_artifacts = payload.get("published_artifacts") or {}
    loaded: dict[str, object] = {}
    for name in ("episodes", "leaderboard"):
        binding = published_artifacts.get(name)
        if not isinstance(binding, dict):
            return False
        path, contained = _contained_batch_artifact(
            binding.get("path"), batch_root=manifest_path.parent.resolve()
        )
        if (
            not contained
            or not path.is_file()
            or _sha256(path) != binding.get("sha256")
        ):
            return False
        try:
            if name == "episodes":
                loaded[name] = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            else:
                loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
    episode_rows = loaded["episodes"]
    leaderboard = loaded["leaderboard"]
    if not isinstance(episode_rows, list) or not isinstance(leaderboard, dict):
        return False
    if not all(isinstance(row, dict) for row in episode_rows):
        return False
    episode_rows = _logical_rows_for_validation(
        list(episode_rows), batch_root=manifest_path.parent.resolve()
    )
    if episode_rows is None:
        return False
    expected_scope = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
            int(row.get("seed", 42)),
            int(row.get("horizon_ticks", 4)),
            f"pass-{pass_index}",
        )
        for row in suite_rows
        for pass_index in range(pass_k)
    }
    actual_scope = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
            int(row.get("seed", 42)),
            int(row.get("horizon_ticks", 4)),
            str(row.get("pass_id") or ""),
        )
        for row in episode_rows
        if isinstance(row, dict)
        and not isinstance(row.get("seed"), bool)
        and isinstance(row.get("seed"), int)
    }
    if actual_scope != expected_scope:
        return False
    try:
        from scripts.batch_llm_eval import (
            _formal_row_eligibility,
            _primary_leaderboard_payload,
        )

        if any(
            not _formal_row_eligibility(
                row,
                required_suite_hash=str(suite_sha256),
                required_implementation_tree_sha256=str(tree),
                required_interaction_mode="logical_persistent",
                verify_artifact_bytes=True,
            )[0]
            for row in episode_rows
        ):
            return False
        derived = _primary_leaderboard_payload(episode_rows)
    except (ImportError, TypeError, ValueError, RuntimeError):
        return False
    return bool(
        len(episode_rows) == expected
        and all(
            isinstance(row, dict)
            and row.get("status") == "ok"
            and row.get("suite_manifest_sha256") == suite_sha256
            and row.get("implementation_tree_sha256") == tree
            and row.get("agent_treatment_sha256")
            == treatment_hashes.get(row.get("model"))
            for row in episode_rows
        )
        and leaderboard.get("batch_state") == "final"
        and leaderboard.get("primary_leaderboard") == derived.get("leaderboard")
        and leaderboard.get("scoring_version") == derived.get("scoring_version")
        and leaderboard.get("primary_leaderboard_formula_version")
        == derived.get("primary_leaderboard_formula_version")
        and leaderboard.get("primary_pairwise") == derived.get("primary_pairwise", [])
        and leaderboard.get("leaderboard_eligible") is True
        and (leaderboard.get("leaderboard_eligibility") or {}).get("eligible") is True
    )


def _realtime_published_manifest_valid(
    payload: JsonDict,
    *,
    manifest_path: Path,
    tree: object,
    suite_sha256: object,
    suite_rows: list[JsonDict],
    run_contract: JsonDict,
    realtime_contract: JsonDict,
) -> bool:
    n_scenarios = len(suite_rows)
    batch_schema_version = str(
        realtime_contract.get("batch_schema_version") or "realtime-formal-batch/1.0"
    )
    scorecard_schema_version = str(
        realtime_contract.get("scorecard_schema_version")
        or "realtime-formal-scorecard/1.0"
    )
    diagnostic_schema_version = str(
        realtime_contract.get("diagnostic_schema_version")
        or realtime_contract.get("scorecard_version")
        or ""
    )
    episode_schema_version = str(
        realtime_contract.get("episode_schema_version") or "realtime-episode/1.0"
    )
    treatment_schema_version = str(
        realtime_contract.get("treatment_schema_version") or "realtime-treatment/1.0"
    )
    realtime_coordinator = str(
        realtime_contract.get("realtime_coordinator") or "realtime_episode_v3"
    )
    identity = payload.get("batch_treatment_identity")
    treatment_sha256 = payload.get("batch_treatment_sha256")
    coverage = payload.get("coverage") or {}
    artifacts = payload.get("artifacts") or {}
    clock = identity.get("clock") if isinstance(identity, dict) else {}
    safety = identity.get("safety") if isinstance(identity, dict) else {}
    model_shard = identity.get("model_shard") if isinstance(identity, dict) else {}
    scheduler = identity.get("scheduler") if isinstance(identity, dict) else {}
    sampling = identity.get("sampling") if isinstance(identity, dict) else {}
    pass_k = (
        (identity.get("sampling") or {}).get("pass_k")
        if isinstance(identity, dict)
        else None
    )
    expected = n_scenarios * int(pass_k or 0)
    agentic_profile = run_contract.get("agentic_profile") or {}
    if not bool(
        payload.get("schema_version") == batch_schema_version
        and payload.get("track") == "realtime_supervision"
        and payload.get("scorecard_schema_version") == scorecard_schema_version
        and payload.get("leaderboard_eligible") is True
        and payload.get("blockers") == []
        and payload.get("safety_profile") == "domain_neutral_hold"
        and payload.get("native_takeover_applicable") is False
        and payload.get("merge_with_logical_primary") is False
        and isinstance(identity, dict)
        and identity.get("schema_version") == batch_schema_version
        and _canonical_sha256(identity) == treatment_sha256
        and payload.get("implementation_tree_sha256") == tree
        and identity.get("implementation_tree_sha256") == tree
        and payload.get("suite_manifest_sha256") == suite_sha256
        and identity.get("suite_sha256") == suite_sha256
        and identity.get("track") == "realtime_supervision"
        and identity.get("interaction_mode") == "realtime_persistent"
        and identity.get("agent_session_mode") == "logical_persistent"
        and _formal_wakeup_policy_valid(realtime_contract, identity)
        and isinstance(model_shard, dict)
        and model_shard.get("model") == payload.get("model")
        and model_shard.get("model_count") == 1
        and model_shard.get("temperature") == 0.0
        and model_shard.get("prompt_mode") == "strict"
        and _formal_tool_choice_matches(model_shard.get("tool_choice"), agentic_profile)
        and model_shard.get("stream_chat_completions") is True
        and all(
            model_shard.get(key) == value
            for key, value in agentic_profile.items()
            if key
            in {
                "max_tokens",
                "protocol_repair_max_tokens",
                "persistent_history_max_messages",
                "persistent_context_max_chars",
                "persistent_memory_max_items",
                "provider_timeout_s",
                "tool_choice",
                "stream_chat_completions",
            }
        )
        and isinstance(model_shard.get("model_context_window_tokens"), int)
        and model_shard.get("model_context_window_tokens") > 0
        and isinstance(model_shard.get("model_max_output_tokens"), int)
        and model_shard.get("model_max_output_tokens") >= model_shard.get("max_tokens")
        and {
            key: clock.get(key)
            for key in (realtime_contract.get("clock_profile") or {})
        }
        == realtime_contract.get("clock_profile")
        and clock.get("process_exit_hard_deadline") is True
        and safety.get("profile") == "domain_neutral_hold"
        and safety.get("native_takeover_applicable") is False
        and safety.get("implementation") == "runner.realtime_actor.HoldSafetySupervisor"
        and scheduler.get("kind") == "bounded_subprocess_pool"
        and scheduler.get("process_group_watchdog") is True
        and isinstance(scheduler.get("max_workers"), int)
        and run_contract.get("minimum_max_workers", 1)
        <= scheduler.get("max_workers")
        <= run_contract.get("maximum_max_workers", 32)
        and sampling.get("seed_mode") == "scenario"
        and identity.get("scorecard_schema_version") == scorecard_schema_version
        and identity.get("diagnostic_schema_version") == diagnostic_schema_version
        and isinstance(pass_k, int)
        and not isinstance(pass_k, bool)
        and pass_k >= 1
        and coverage
        == {"expected": expected, "terminal": expected, "eligible": expected}
    ):
        return False

    batch_root = manifest_path.parent.resolve()
    loaded: dict[str, object] = {}
    journal_binding = artifacts.get("episodes_journal")
    if not isinstance(journal_binding, dict):
        return False
    journal_path, journal_contained = _contained_batch_artifact(
        journal_binding.get("path"), batch_root=batch_root
    )
    if (
        not journal_contained
        or not journal_path.is_file()
        or _sha256(journal_path) != journal_binding.get("sha256")
    ):
        return False
    for name in ("episodes", "realtime_scorecard", "leaderboard"):
        binding = artifacts.get(name)
        if not isinstance(binding, dict):
            return False
        path, contained = _contained_batch_artifact(
            binding.get("path"), batch_root=batch_root
        )
        if (
            not contained
            or not path.is_file()
            or _sha256(path) != binding.get("sha256")
        ):
            return False
        if name == "episodes":
            try:
                loaded[name] = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return False
        else:
            loaded[name] = _safe_load_json(path)

    episode_artifacts = artifacts.get("episode_artifacts")
    if not isinstance(episode_artifacts, list) or len(episode_artifacts) != expected:
        return False
    job_keys: set[str] = set()
    episode_payloads: dict[str, JsonDict] = {}
    episode_paths: dict[str, Path] = {}
    for binding in episode_artifacts:
        if not isinstance(binding, dict):
            return False
        job_key = str(binding.get("job_key") or "")
        path, contained = _contained_batch_artifact(
            binding.get("artifact_path"), batch_root=batch_root
        )
        if (
            not job_key
            or job_key in job_keys
            or not contained
            or not path.is_file()
            or _sha256(path) != binding.get("artifact_sha256")
        ):
            return False
        episode = _safe_load_json(path)
        episode_identity = episode.get("treatment_identity") or {}
        implementation_contract = episode_identity.get("implementation_contract") or {}
        provider = episode_identity.get("provider_public_config") or {}
        episode_clock = episode_identity.get("clock") or {}
        if not bool(
            episode.get("schema_version") == episode_schema_version
            and episode.get("interaction_mode") == "realtime_persistent"
            and episode.get("episode_status") == "complete"
            and episode.get("evaluation_ready") is True
            and episode.get("behavioral_state_artifact_status") == "complete"
            and isinstance(episode_identity, dict)
            and episode_identity.get("schema_version") == treatment_schema_version
            and _canonical_sha256(episode_identity) == episode.get("treatment_sha256")
            and _formal_wakeup_policy_valid(
                realtime_contract, episode_identity
            )
            and implementation_contract.get("implementation_tree_sha256") == tree
            and implementation_contract.get("realtime_coordinator")
            == realtime_coordinator
            and implementation_contract.get("prompt_context_compiler")
            == "persistent_event_compiler_v3"
            and _valid_sha256(implementation_contract.get("prompt_contract_sha256"))
            and _valid_sha256(implementation_contract.get("tool_schema_sha256"))
            and provider.get("model") == model_shard.get("model")
            and provider.get("provider") == model_shard.get("provider")
            and provider.get("base_url") == model_shard.get("base_url")
            and provider.get("api_mode") == model_shard.get("api_mode")
            and provider.get("prompt_mode") == "strict"
            and provider.get("interaction_mode") == "logical_persistent"
            and provider.get("temperature") == 0.0
            and _formal_tool_choice_matches(
                provider.get("tool_choice"), agentic_profile
            )
            and provider.get("stream_chat_completions") is True
            and episode_clock.get("tick_interval_s") == clock.get("tick_interval_s")
            and (episode.get("artifact_validation") or {}).get("valid") is True
            and not (
                (episode.get("artifact_validation") or {}).get("blocker_codes") or []
            )
            and (episode.get("provider_audit_contract") or {}).get("schema_version")
            == "realtime-provider-audit-contract/1.0"
            and (episode.get("provider_audit_contract") or {}).get("complete") is True
            and (episode.get("event_contract") or {}).get("violation_count") == 0
            and (episode.get("evidence_closure") or {}).get("closure_complete") is True
            and isinstance(episode.get("action_lifecycle"), list)
            and isinstance(episode.get("semantic_ledger"), dict)
            and isinstance(episode.get("structured_memory"), dict)
        ):
            return False
        job_keys.add(job_key)
        episode_payloads[job_key] = episode
        episode_paths[job_key] = path

    episode_rows = loaded["episodes"]
    if not isinstance(episode_rows, list) or len(episode_rows) != expected:
        return False
    rows_by_job = {
        str(row.get("job_key") or ""): row
        for row in episode_rows
        if isinstance(row, dict)
    }
    expected_scope = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
            int(row.get("seed", 42)),
            int(row.get("horizon_ticks", 4)),
            f"pass-{pass_index}",
        )
        for row in suite_rows
        for pass_index in range(pass_k)
    }
    actual_scope = {
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
            int(row.get("seed", 42)),
            int(row.get("horizon_ticks", 4)),
            str(row.get("pass_id") or ""),
        )
        for row in rows_by_job.values()
        if isinstance(row.get("seed"), int) and not isinstance(row.get("seed"), bool)
    }
    if (
        set(rows_by_job) != job_keys
        or actual_scope != expected_scope
        or any(
            row.get("status") != "ok"
            or row.get("batch_treatment_sha256") != treatment_sha256
            or row.get("artifact_sha256")
            != next(
                binding.get("artifact_sha256")
                for binding in episode_artifacts
                if binding.get("job_key") == job_key
            )
            or row.get("episode_treatment_sha256")
            != episode_payloads[job_key].get("treatment_sha256")
            or row.get("diagnostics") != episode_payloads[job_key].get("diagnostics")
            for job_key, row in rows_by_job.items()
        )
    ):
        return False

    run_config = {
        "batch_treatment_identity": identity,
        "batch_treatment_sha256": treatment_sha256,
        "output_dir": str(batch_root),
        "model": payload.get("model"),
        "safety_profile": payload.get("safety_profile"),
        "native_takeover_applicable": payload.get("native_takeover_applicable"),
    }
    try:
        from scripts.batch_realtime_llm_eval import (
            aggregate_realtime_scorecard,
            realtime_artifact_eligibility,
            terminal_row_from_artifact,
        )

        if any(
            realtime_artifact_eligibility(episode_payloads[job_key], row, run_config)
            for job_key, row in rows_by_job.items()
        ):
            return False
        if any(
            (
                revalidated := terminal_row_from_artifact(
                    row, episode_paths[job_key], run_config
                )
            ).get("status")
            != "ok"
            or revalidated.get("diagnostics") != row.get("diagnostics")
            or revalidated.get("turn_deadlines") != row.get("turn_deadlines")
            for job_key, row in rows_by_job.items()
        ):
            return False
        derived_scorecard = aggregate_realtime_scorecard(
            list(rows_by_job.values()),
            list(rows_by_job.values()),
            run_config,
        )
    except (ImportError, TypeError, ValueError, RuntimeError):
        return False

    scorecard = loaded["realtime_scorecard"]
    leaderboard = loaded["leaderboard"]
    if not isinstance(scorecard, dict) or not isinstance(leaderboard, dict):
        return False
    return bool(
        scorecard == derived_scorecard
        and scorecard.get("schema_version") == scorecard_schema_version
        and scorecard.get("track") == "realtime_supervision"
        and scorecard.get("batch_treatment_sha256") == treatment_sha256
        and scorecard.get("model") == payload.get("model")
        and scorecard.get("coverage") == coverage
        and leaderboard.get("schema_version") == scorecard_schema_version
        and leaderboard.get("track") == "realtime_supervision"
        and leaderboard.get("merge_with_logical_primary") is False
        and leaderboard.get("rows") == [scorecard]
    )


def _scenario_bindings_match(actual: object, expected: list[dict[str, str]]) -> bool:
    """Compare scenario evidence as a strict identity-keyed collection."""
    if not isinstance(actual, list) or len(actual) != len(expected):
        return False
    if not all(isinstance(row, dict) for row in actual):
        return False

    fields = (
        "scenario_id",
        "scenario_signature",
        "scenario_uri",
        "scenario_yaml_sha256",
    )
    actual_keys = [
        tuple(str(row.get(field) or "") for field in fields) for row in actual
    ]
    expected_keys = [
        tuple(str(row.get(field) or "") for field in fields) for row in expected
    ]
    return len(set(actual_keys)) == len(actual_keys) and sorted(actual_keys) == sorted(
        expected_keys
    )


def _portable_evidence_closure_valid(
    release: Path, manifest: JsonDict, rows: list[JsonDict]
) -> bool:
    replay = manifest.get("protocol21_replay")
    pipeline_artifacts = manifest.get("pipeline_artifacts")
    if not isinstance(replay, dict) or not isinstance(pipeline_artifacts, dict):
        return False
    evidence_name = str(replay.get("evidence_bundle") or "")
    source_name = "protocol21_source_suite.json"
    source_path = (release / source_name).resolve()
    expected_source_reference = f"release/{release.name}/{source_name}"
    evidence_path = (release / evidence_name).resolve()
    try:
        evidence_path.relative_to(release.resolve())
    except ValueError:
        return False
    if (
        evidence_name != "protocol21_public_evidence_bundle.json"
        or Path(evidence_name).is_absolute()
        or len(Path(evidence_name).parts) != 1
        or not evidence_path.is_file()
        or replay.get("evidence_bundle_sha256") != _sha256(evidence_path)
        or replay.get("source_suite") != expected_source_reference
        or not source_path.is_file()
        or replay.get("source_suite_sha256") != _sha256(source_path)
    ):
        return False
    evidence = _safe_load_json(evidence_path)
    if (
        evidence.get("schema_version") != "protocol21-public-evidence-bundle-v1"
        or evidence.get("status") != "complete"
        or evidence.get("scope") != "portable_summary_of_immutable_internal_evidence"
        or evidence.get("binding_root_sha256") != _canonical_payload_sha256(evidence)
    ):
        return False

    tree = manifest.get("implementation_tree_sha256")
    release_pipeline_sha256 = manifest.get("core_release_pipeline_sha256")
    pipeline = evidence.get("pipeline")
    artifacts = evidence.get("artifacts")
    counts = evidence.get("counts")
    stage_map = pipeline_artifacts.get("stage_artifacts")
    if not all(
        isinstance(value, dict) for value in (pipeline, artifacts, counts, stage_map)
    ):
        return False
    assert isinstance(pipeline, dict)
    assert isinstance(artifacts, dict)
    assert isinstance(counts, dict)
    assert isinstance(stage_map, dict)
    expected_artifact_names = {*PIPELINE_STAGE_HASH_FIELDS, "source_suite"}
    if set(artifacts) != expected_artifact_names or set(stage_map) != set(
        PIPELINE_STAGE_HASH_FIELDS
    ):
        return False
    if not (
        pipeline.get("status") == "formal_evaluation_ready"
        and pipeline.get("implementation_tree_sha256") == tree
        and _valid_sha256(release_pipeline_sha256)
        and pipeline.get("core_release_pipeline_sha256") == release_pipeline_sha256
        and pipeline_artifacts.get("core_release_pipeline_sha256")
        == release_pipeline_sha256
        and replay.get("core_release_pipeline_sha256") == release_pipeline_sha256
        and (pipeline.get("manifest") or {}).get("sha256")
        == pipeline_artifacts.get("pipeline_manifest_sha256")
        and pipeline.get("source_suite_sha256") == replay.get("source_suite_sha256")
    ):
        return False
    for name, hash_field in PIPELINE_STAGE_HASH_FIELDS.items():
        binding = stage_map.get(name)
        artifact = artifacts.get(name)
        if not isinstance(binding, dict) or not isinstance(artifact, dict):
            return False
        if binding != {
            "relative_path": PIPELINE_STAGE_FILES[name],
            "sha256": pipeline_artifacts.get(hash_field),
        }:
            return False
        if not (
            artifact.get("sha256") == pipeline_artifacts.get(hash_field)
            and artifact.get("implementation_tree_sha256") == tree
            and artifact.get("core_release_pipeline_sha256") == release_pipeline_sha256
        ):
            return False
    source_artifact = artifacts.get("source_suite")
    if not isinstance(source_artifact, dict) or source_artifact.get(
        "sha256"
    ) != replay.get("source_suite_sha256"):
        return False

    scenario_bindings = evidence.get("scenario_bindings")
    dependency_edges = evidence.get("artifact_dependency_edges")
    source_bindings = evidence.get("source_asset_bindings")
    expected_scenarios = [
        {
            "scenario_id": str(row.get("scenario_id") or ""),
            "scenario_signature": str(row.get("scenario_signature") or ""),
            "scenario_uri": f"repo://{row.get('path') or ''}",
            "scenario_yaml_sha256": str(row.get("yaml_sha256") or ""),
        }
        for row in rows
    ]
    if not _scenario_bindings_match(scenario_bindings, expected_scenarios):
        return False
    return bool(
        isinstance(dependency_edges, list)
        and isinstance(source_bindings, list)
        and counts.get("pipeline_stages") == len(PIPELINE_STAGE_HASH_FIELDS)
        and counts.get("artifact_nodes") == len(artifacts)
        and counts.get("artifact_dependency_edges") == len(dependency_edges)
        and counts.get("core_scenarios") == len(rows)
        and counts.get("unique_source_assets") == len(source_bindings)
    )


def _ordered_scenario_identity_sha256(rows: list[JsonDict]) -> str:
    identities = [
        {
            "scenario_id": str(row.get("scenario_id") or ""),
            "scenario_signature": str(row.get("scenario_signature") or ""),
            "path": str(row.get("path") or ""),
        }
        for row in rows
    ]
    return hashlib.sha256(
        json.dumps(
            identities,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _formal_runtime_bundle_valid(
    release: Path,
    manifest: JsonDict,
    core: JsonDict,
    rows: list[JsonDict],
) -> bool:
    binding = manifest.get("formal_runtime_bundle")
    if not isinstance(binding, dict):
        return False
    raw_path = str(binding.get("path") or "")
    relative = Path(raw_path)
    if (
        raw_path != FORMAL_RUNTIME_BUNDLE_NAME
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        return False
    path = release / relative
    if path.is_symlink() or not path.is_file():
        return False
    payload = _safe_load_json(path)
    runtime_rows = payload.get("scenarios")
    if not isinstance(runtime_rows, list) or not all(
        isinstance(row, dict) for row in runtime_rows
    ):
        return False
    runtime_rows = list(runtime_rows)
    expected_identity_sha256 = _ordered_scenario_identity_sha256(rows)
    if binding != {
        "path": FORMAL_RUNTIME_BUNDLE_NAME,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "schema_version": FORMAL_RUNTIME_BUNDLE_SCHEMA,
        "n_scenarios": len(rows),
        "ordered_scenario_identity_sha256": expected_identity_sha256,
    }:
        return False
    if not (
        payload.get("schema_version") == FORMAL_RUNTIME_BUNDLE_SCHEMA
        and payload.get("release_id") == manifest.get("release_id") == release.name
        and payload.get("status") == "formal_evaluation_ready"
        and payload.get("formal_evaluation_ready") is True
        and payload.get("formal_run_blockers") == []
        and payload.get("scoring_version") == manifest.get("scoring_version")
        and payload.get("core_selection_policy")
        == manifest.get("core_selection_policy")
        and payload.get("core_settings_stamp") == manifest.get("core_settings_stamp")
        and payload.get("implementation_tree_sha256")
        == manifest.get("implementation_tree_sha256")
        and payload.get("core_release_pipeline_sha256")
        == manifest.get("core_release_pipeline_sha256")
        and payload.get("release_tooling_sha256")
        == manifest.get("release_tooling_sha256")
        and payload.get("formal_run_contract") == manifest.get("formal_run_contract")
        and payload.get("n_scenarios") == len(rows) == core.get("n_scenarios")
        and payload.get("ordered_scenario_identity_sha256") == expected_identity_sha256
    ):
        return False
    logical = manifest.get("formal_batch_contract")
    realtime = manifest.get("formal_realtime_batch_contract")
    if not isinstance(logical, dict) or not isinstance(realtime, dict):
        return False
    if (
        payload.get("formal_batch_contract_sha256")
        != hashlib.sha256(
            (
                json.dumps(logical, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
        ).hexdigest()
    ):
        return False
    if (
        payload.get("formal_realtime_batch_contract_sha256")
        != hashlib.sha256(
            (
                json.dumps(realtime, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
    ):
        return False
    for runtime_row, core_row in zip(runtime_rows, rows, strict=True):
        case_ledger = runtime_row.get("case_ledger")
        compact_row = {
            key: value for key, value in runtime_row.items() if key != "case_ledger"
        }
        if (
            compact_row != core_row
            or not isinstance(case_ledger, dict)
            or not case_ledger
        ):
            return False
        if str(case_ledger.get("source_denominator_key") or "") != str(
            core_row.get("source_denominator_key") or ""
        ):
            return False
    exact_bindings = {
        "core_suite": (
            "core_suite.json",
            (manifest.get("core_suite") or {}).get("sha256"),
        ),
        "source_suite": (
            "protocol21_source_suite.json",
            (manifest.get("protocol21_replay") or {}).get("source_suite_sha256"),
        ),
        "candidate_closure": (
            "candidate_closure.json",
            (manifest.get("candidate_closure") or {}).get("sha256"),
        ),
        "backend_runtime_closure": (
            "backend_runtime_closure.json",
            (manifest.get("backend_runtime_closure") or {}).get("sha256"),
        ),
    }
    for name, (filename, digest) in exact_bindings.items():
        item = payload.get(name)
        artifact = release / filename
        if not (
            isinstance(item, dict)
            and item.get("path") == filename
            and item.get("sha256") == digest == _sha256(artifact)
        ):
            return False
    core_binding = payload["core_suite"]
    source_binding = payload["source_suite"]
    source = _safe_load_json(release / "protocol21_source_suite.json")
    source_rows = source.get("scenarios")
    if not isinstance(source_rows, list) or not all(
        isinstance(row, dict) for row in source_rows
    ):
        return False
    if not (
        core_binding.get("n_scenarios") == len(rows)
        and core_binding.get("ordered_scenario_identity_sha256")
        == expected_identity_sha256
        and source_binding.get("n_scenarios") == len(source_rows)
        and source_binding.get("ordered_scenario_identity_sha256")
        == _ordered_scenario_identity_sha256(source_rows)
        and payload["candidate_closure"].get("identity_set_sha256")
        == (manifest.get("candidate_closure") or {}).get("identity_set_sha256")
        and payload["backend_runtime_closure"].get("identity_sha256")
        == (manifest.get("backend_runtime_closure") or {}).get("identity_sha256")
    ):
        return False
    public = payload.get("public_evidence")
    replay = manifest.get("protocol21_replay") or {}
    evidence_name = str(replay.get("evidence_bundle") or "")
    evidence_path = release / evidence_name
    evidence = _safe_load_json(evidence_path) if evidence_path.is_file() else {}
    if not (
        isinstance(public, dict)
        and public.get("path")
        == evidence_name
        == "protocol21_public_evidence_bundle.json"
        and public.get("sha256")
        == replay.get("evidence_bundle_sha256")
        == _sha256(evidence_path)
        and public.get("binding_root_sha256") == evidence.get("binding_root_sha256")
    ):
        return False
    audit = payload.get("internal_audit")
    attestations = audit.get("stage_attestations") if isinstance(audit, dict) else None
    pipeline = manifest.get("pipeline_artifacts") or {}
    stage_bindings = pipeline.get("stage_artifacts")
    if not isinstance(attestations, list) or not isinstance(stage_bindings, dict):
        return False
    if [row.get("name") for row in attestations] != list(PIPELINE_STAGE_FILES):
        return False
    if audit.get("pipeline_manifest_sha256") != pipeline.get(
        "pipeline_manifest_sha256"
    ) or audit.get("readiness_sha256") != pipeline.get("readiness_sha256"):
        return False
    release_pipeline_sha256 = manifest.get("core_release_pipeline_sha256")
    tree = manifest.get("implementation_tree_sha256")
    return all(
        isinstance(row, dict)
        and row.get("return_code") == 0
        and isinstance(row.get("counts"), dict)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in row["counts"].values()
        )
        and (
            row["counts"].get("n_completed") is None
            or row["counts"].get("n_expected") is None
            or row["counts"]["n_completed"] == row["counts"]["n_expected"]
        )
        and (row.get("status") is not None or row.get("name") == "strategy_depth")
        and row.get("output_sha256")
        == (stage_bindings.get(str(row.get("name") or "")) or {}).get("sha256")
        and row.get("implementation_tree_sha256") == tree
        and row.get("core_release_pipeline_sha256") == release_pipeline_sha256
        for row in attestations
    )


def _agentic_formal_checks(
    release: Path,
    manifest: JsonDict,
    core: JsonDict,
    rows: list[JsonDict],
    *,
    portable: bool,
    artifact_root: Path,
) -> dict[str, bool]:
    if not _agentic_contract_required(manifest):
        return {
            "agentic_formal_metadata_complete": True,
            "agentic_formal_run_contract_valid": True,
            "agentic_logical_batch_contract_valid": True,
            "agentic_realtime_batch_contract_valid": True,
            "agentic_selection_readiness_binding_valid": True,
            "agentic_pipeline_artifact_binding_valid": True,
            "agentic_release_pipeline_binding_valid": True,
            "agentic_release_tooling_binding_valid": True,
            "agentic_implementation_tree_binding_valid": True,
            "agentic_release_state_valid": True,
            "agentic_published_evidence_valid": True,
            "agentic_portable_evidence_closure_valid": True,
            "agentic_formal_runtime_bundle_valid": True,
            "agentic_formal_completion_identity_valid": True,
            "agentic_formal_result_tree_valid": True,
            "agentic_formal_result_paths_portable": True,
            "agentic_formal_distribution_receipt_valid": True,
        }

    run_contract = manifest.get("formal_run_contract")
    logical_contract = manifest.get("formal_batch_contract")
    realtime_contract = manifest.get("formal_realtime_batch_contract")
    formal_evidence = manifest.get("formal_evidence")
    pipeline_artifacts = manifest.get("pipeline_artifacts")
    replay = manifest.get("protocol21_replay")
    runtime_bundle_required = _release_version(manifest) >= (0, 59, 0)
    runtime_bundle_ok = _formal_runtime_bundle_valid(release, manifest, core, rows)
    metadata_complete = all(
        isinstance(value, dict)
        for value in (
            run_contract,
            logical_contract,
            realtime_contract,
            formal_evidence,
            pipeline_artifacts,
            replay,
        )
    )
    if not metadata_complete:
        return {
            "agentic_formal_metadata_complete": False,
            "agentic_formal_run_contract_valid": False,
            "agentic_logical_batch_contract_valid": False,
            "agentic_realtime_batch_contract_valid": False,
            "agentic_selection_readiness_binding_valid": False,
            "agentic_pipeline_artifact_binding_valid": False,
            "agentic_release_pipeline_binding_valid": False,
            "agentic_release_tooling_binding_valid": False,
            "agentic_implementation_tree_binding_valid": False,
            "agentic_release_state_valid": False,
            "agentic_published_evidence_valid": False,
            "agentic_portable_evidence_closure_valid": False,
            "agentic_formal_runtime_bundle_valid": False,
            "agentic_formal_completion_identity_valid": False,
            "agentic_formal_result_tree_valid": False,
            "agentic_formal_result_paths_portable": False,
            "agentic_formal_distribution_receipt_valid": False,
        }

    assert isinstance(run_contract, dict)
    assert isinstance(logical_contract, dict)
    assert isinstance(realtime_contract, dict)
    assert isinstance(formal_evidence, dict)
    assert isinstance(pipeline_artifacts, dict)
    assert isinstance(replay, dict)
    publication_checks = _formal_publication_checks(
        release,
        manifest,
        artifact_root=artifact_root,
    )

    pipeline_text = str(manifest.get("pipeline_dir") or "")
    runtime_text = str(formal_evidence.get("runtime_root") or "")
    readiness_text = str(formal_evidence.get("readiness") or "")
    pipeline_path, pipeline_path_ok = _resolve_repo_artifact(
        pipeline_text, artifact_root=artifact_root
    )
    release_locator = f"release/{release.name}"
    if readiness_text.startswith(f"{release_locator}/"):
        readiness_relative = readiness_text.removeprefix(f"{release_locator}/")
        readiness_path = (release / readiness_relative).resolve()
        readiness_path_ok = bool(
            ".." not in Path(readiness_relative).parts
            and not Path(readiness_relative).is_absolute()
        )
    else:
        readiness_path, readiness_path_ok = _resolve_repo_artifact(
            readiness_text, artifact_root=artifact_root
        )
    if runtime_text == release_locator:
        runtime_path = release.resolve()
        runtime_path_ok = True
    else:
        runtime_path, runtime_path_ok = _resolve_repo_artifact(
            runtime_text, artifact_root=artifact_root
        )
    pipeline_manifest_path = pipeline_path / "protocol2_v21_pipeline_manifest.json"

    expected_artifact_paths = bool(
        (
            runtime_bundle_required
            and runtime_path == release.resolve()
            and readiness_path == release.resolve() / FORMAL_RUNTIME_BUNDLE_NAME
        )
        or (
            not runtime_bundle_required
            and readiness_path == pipeline_path / "protocol2_v21_core_readiness.json"
        )
    )
    paths_ok = bool(
        pipeline_path_ok
        and runtime_path_ok
        and readiness_path_ok
        and expected_artifact_paths
        and pipeline_artifacts.get("path") == pipeline_text
        and replay.get("pipeline_dir") == pipeline_text
    )

    readiness = _safe_load_json(readiness_path) if readiness_path_ok else {}
    pipeline = _safe_load_json(pipeline_manifest_path) if pipeline_path_ok else {}
    leaderboard_formula = (
        PRIMARY_LEADERBOARD_FORMULA_VERSION
        if portable
        else readiness.get("primary_leaderboard_formula_version")
    )
    inference_version = (
        PRIMARY_INFERENCE_VERSION
        if portable
        else readiness.get("primary_inference_version")
    )
    suite_manifest_sha256 = (
        realtime_contract.get("suite_manifest_sha256")
        if portable
        else readiness.get("suite_manifest_sha256")
    )

    expected_logical_contract: JsonDict = {
        "contract_version": "agentic_persistent.v1",
        "runtime_evidence_root": runtime_text,
        "selection_source": f"{readiness_text}#scenarios",
        "interaction_mode": "logical_persistent",
        "prompt_mode": "strict",
        "seed_mode": "scenario",
        "scheduler_mode": "global",
        "temperature": 0.0,
        "requires_explicit_model_capabilities": True,
        "save_trajectories": True,
        "primary_leaderboard_formula_version": leaderboard_formula,
        "primary_inference_version": inference_version,
        "cardinality": {
            "models": {"per_shard": 1},
            "pass_k": {"minimum": 1},
            "workers": {"minimum": 1, "maximum": 32},
        },
        "expected_episode_formula": "n_scenarios * models_per_shard * pass_k",
        "agentic_profile": AGENTIC_PROFILE_V1,
    }
    expected_run_contract, expected_realtime_contract_base = (
        _formal_contracts_for_release(manifest)
    )
    if expected_realtime_contract_base.get("wakeup_policy") is not None:
        expected_logical_contract["wakeup_policy"] = deepcopy(
            expected_realtime_contract_base["wakeup_policy"]
        )
    expected_realtime_contract: JsonDict = {
        **expected_realtime_contract_base,
        "runtime_evidence_root": runtime_text,
        "selection_source": f"{readiness_text}#scenarios",
        "suite_manifest_sha256": suite_manifest_sha256,
        "n_scenarios": len(rows),
    }

    run_contract_ok = run_contract == expected_run_contract and (
        portable or readiness.get("formal_run_contract") == expected_run_contract
    )
    logical_contract_ok = (
        logical_contract == expected_logical_contract
        and logical_contract.get("agentic_profile")
        == run_contract.get("agentic_profile")
    )
    realtime_contract_ok = (
        realtime_contract == expected_realtime_contract
        and realtime_contract.get("clock_profile")
        == (run_contract.get("realtime_formal_contract") or {}).get("clock_profile")
    )

    readiness_rows = readiness.get("scenarios")
    selection_binding_ok = bool(
        paths_ok
        and logical_contract.get("selection_source") == f"{readiness_text}#scenarios"
        and realtime_contract.get("selection_source") == f"{readiness_text}#scenarios"
        and _valid_sha256(suite_manifest_sha256)
        and realtime_contract.get("suite_manifest_sha256") == suite_manifest_sha256
    )
    selection_ok = selection_binding_ok and (
        portable
        or bool(
            readiness.get("status") == "formal_evaluation_ready"
            and readiness.get("formal_evaluation_ready") is True
            and readiness.get("formal_run_blockers") == []
            and readiness.get("scoring_version") == manifest.get("scoring_version")
            and readiness.get("n_scenarios") == len(rows)
            and isinstance(readiness_rows, list)
            and [
                _scenario_identity_for_binding(row)
                for row in readiness_rows
                if isinstance(row, dict)
            ]
            == [_scenario_identity_for_binding(row) for row in rows]
        )
    )

    source_text = str(replay.get("source_suite") or "")
    source_path, source_path_ok = _resolve_repo_artifact(
        source_text, artifact_root=artifact_root
    )
    expected_source_path = (release / "protocol21_source_suite.json").resolve()
    required_files = (
        readiness_path,
        pipeline_manifest_path,
        source_path,
    )
    artifacts_exist = all(path.is_file() for path in required_files)
    hash_bindings_ok = bool(
        artifacts_exist
        and (
            (
                runtime_bundle_required
                and (manifest.get("formal_runtime_bundle") or {}).get("sha256")
                == _sha256(readiness_path)
            )
            or (
                not runtime_bundle_required
                and pipeline_artifacts.get("readiness_sha256")
                == _sha256(readiness_path)
            )
        )
        and pipeline_artifacts.get("pipeline_manifest_sha256")
        == _sha256(pipeline_manifest_path)
        and source_path_ok
        and source_path == expected_source_path
        and replay.get("source_suite_sha256") == _sha256(source_path)
        and pipeline.get("source_suite_sha256") == _sha256(source_path)
    )
    stages = pipeline.get("stages")
    stage_artifacts = pipeline_artifacts.get("stage_artifacts")
    release_pipeline_sha256 = manifest.get("core_release_pipeline_sha256")
    stage_bindings_ok = (
        isinstance(stages, list)
        and isinstance(stage_artifacts, dict)
        and len(stages) == len(PIPELINE_STAGE_HASH_FIELDS)
        and set(stage_artifacts) == set(PIPELINE_STAGE_HASH_FIELDS)
    )
    if stage_bindings_ok:
        assert isinstance(stage_artifacts, dict)
        seen_stages: set[str] = set()
        seen_output_paths: set[Path] = set()
        for stage in stages:
            if not isinstance(stage, dict):
                stage_bindings_ok = False
                break
            name = str(stage.get("name") or "")
            hash_field = PIPELINE_STAGE_HASH_FIELDS.get(name)
            artifact_binding = stage_artifacts.get(name)
            if not isinstance(artifact_binding, dict):
                stage_bindings_ok = False
                break
            relative_path = str(artifact_binding.get("relative_path") or "")
            relative = Path(relative_path)
            output_path = (pipeline_path / relative).resolve()
            try:
                output_path.relative_to(pipeline_path.resolve())
                output_path_ok = bool(
                    relative_path == PIPELINE_STAGE_FILES.get(name)
                    and not relative.is_absolute()
                    and ".." not in relative.parts
                    and output_path.is_file()
                )
            except ValueError:
                output_path_ok = False
            if (
                hash_field is None
                or name in seen_stages
                or not output_path_ok
                or output_path in seen_output_paths
                or stage.get("return_code") != 0
                or _sha256(output_path) != stage.get("output_sha256")
                or stage.get("output_sha256") != pipeline_artifacts.get(hash_field)
                or artifact_binding.get("sha256") != stage.get("output_sha256")
                or set(artifact_binding) != {"relative_path", "sha256"}
                or stage.get("core_release_pipeline_sha256") != release_pipeline_sha256
                or _safe_load_json(output_path).get("core_release_pipeline_sha256")
                != release_pipeline_sha256
            ):
                stage_bindings_ok = False
                break
            seen_stages.add(name)
            seen_output_paths.add(output_path)
        stage_bindings_ok = stage_bindings_ok and seen_stages == set(
            PIPELINE_STAGE_HASH_FIELDS
        )
    pipeline_ok = bool(
        paths_ok
        and hash_bindings_ok
        and stage_bindings_ok
        and pipeline.get("status") == "formal_evaluation_ready"
    )
    public_evidence_closure_ok = _portable_evidence_closure_valid(
        release, manifest, rows
    )
    public_evidence_fallback = bool(
        runtime_bundle_required
        and runtime_bundle_ok
        and public_evidence_closure_ok
        and not pipeline_manifest_path.is_file()
    )
    portable_closure_ok = (
        public_evidence_closure_ok
        if portable or public_evidence_fallback
        else True
    )
    if portable:
        pipeline_ok = public_evidence_closure_ok
    elif public_evidence_fallback:
        pipeline_ok = True
    if runtime_bundle_required:
        pipeline_ok = bool(pipeline_ok and runtime_bundle_ok)

    live_release_pipeline_sha256 = (
        None
        if portable
        else implementation_identity(artifact_root).get("core_release_pipeline_sha256")
    )
    release_pipeline_ok = bool(
        _valid_sha256(release_pipeline_sha256)
        and pipeline_artifacts.get("core_release_pipeline_sha256")
        == release_pipeline_sha256
        and replay.get("core_release_pipeline_sha256") == release_pipeline_sha256
        and pipeline.get("core_release_pipeline_sha256") == release_pipeline_sha256
        and readiness.get("core_release_pipeline_sha256") == release_pipeline_sha256
        and live_release_pipeline_sha256 == release_pipeline_sha256
    )
    if portable:
        release_pipeline_ok = bool(
            public_evidence_closure_ok
            and _valid_sha256(release_pipeline_sha256)
            and pipeline_artifacts.get("core_release_pipeline_sha256")
            == release_pipeline_sha256
            and replay.get("core_release_pipeline_sha256") == release_pipeline_sha256
        )
    elif public_evidence_fallback:
        release_pipeline_ok = bool(
            live_release_pipeline_sha256 == release_pipeline_sha256
            and pipeline_artifacts.get("core_release_pipeline_sha256")
            == release_pipeline_sha256
            and replay.get("core_release_pipeline_sha256") == release_pipeline_sha256
        )

    release_tooling_sha256 = manifest.get("release_tooling_sha256")
    release_tooling_ok = bool(
        _valid_sha256(release_tooling_sha256)
        and pipeline_artifacts.get("release_tooling_sha256") == release_tooling_sha256
        and replay.get("release_tooling_sha256") == release_tooling_sha256
        and pipeline.get("release_tooling_sha256") == release_tooling_sha256
    )
    if portable:
        release_tooling_ok = bool(
            _valid_sha256(release_tooling_sha256)
            and pipeline_artifacts.get("release_tooling_sha256")
            == release_tooling_sha256
            and replay.get("release_tooling_sha256") == release_tooling_sha256
        )
    elif public_evidence_fallback:
        release_tooling_ok = bool(
            _valid_sha256(release_tooling_sha256)
            and pipeline_artifacts.get("release_tooling_sha256")
            == release_tooling_sha256
            and replay.get("release_tooling_sha256") == release_tooling_sha256
        )

    tree = manifest.get("implementation_tree_sha256")
    live_tree = (
        None
        if portable
        else implementation_identity(artifact_root)["implementation_tree_sha256"]
    )
    tree_ok = bool(
        _valid_sha256(tree)
        and live_tree == tree
        and core.get("implementation_tree_sha256") == tree
        and readiness.get("implementation_tree_sha256") == tree
        and pipeline.get("implementation_tree_sha256") == tree
        and replay.get("implementation_tree_sha256") == tree
        and isinstance(stages, list)
        and all(
            isinstance(stage, dict) and stage.get("implementation_tree_sha256") == tree
            for stage in stages
        )
    )
    if portable:
        tree_ok = bool(
            public_evidence_closure_ok
            and _valid_sha256(tree)
            and core.get("implementation_tree_sha256") == tree
            and replay.get("implementation_tree_sha256") == tree
        )
    elif public_evidence_fallback:
        tree_ok = bool(
            live_tree == tree
            and _valid_sha256(tree)
            and core.get("implementation_tree_sha256") == tree
            and replay.get("implementation_tree_sha256") == tree
        )

    exclusion_codes = {
        str((item.get("reason") or {}).get("code") or "")
        for item in (
            (manifest.get("leaderboard_eligibility") or {}).get("suite_exclusions")
            or []
        )
        if isinstance(item, dict)
    }
    expected_blockers = {
        "formal_logical_persistent_evaluation_pending",
        "formal_realtime_persistent_evaluation_pending",
    }
    release_version = _release_version(manifest)
    if release_version >= (0, 58, 0):
        expected_blockers.add("formal_runtime_evidence_distribution_pending")
    replay_base_complete = bool(
        replay.get("status") == "complete"
        and replay.get("formal_evaluation_ready") is True
        and replay.get("release_coverage_passed") is True
        and replay.get("n_selected") == len(rows)
    )
    if release_version >= (0, 58, 0):
        disposition_counts = replay.get("selection_disposition_counts")
        valid_dispositions = bool(
            isinstance(disposition_counts, dict)
            and disposition_counts
            and all(
                isinstance(count, int) and not isinstance(count, bool) and count >= 0
                for count in disposition_counts.values()
            )
        )
        n_source = replay.get("n_source")
        n_selected = replay.get("n_selected")
        n_secondary = replay.get("n_secondary")
        n_rejected = replay.get("n_rejected")
        replay_partition_complete = bool(
            valid_dispositions
            and all(
                isinstance(count, int) and not isinstance(count, bool) and count >= 0
                for count in (n_source, n_selected, n_secondary, n_rejected)
            )
            and n_source == n_selected + n_secondary + n_rejected
            and sum(disposition_counts.values()) == n_source
            and disposition_counts.get("core_locked", 0) == n_selected
            and disposition_counts.get("secondary_duplicate", 0) == n_secondary
            and n_rejected
            == sum(
                count
                for disposition, count in disposition_counts.items()
                if disposition not in {"core_locked", "secondary_duplicate"}
            )
            and replay.get("n_held_repair") == disposition_counts.get("held_repair", 0)
            and replay.get("n_retired_intrinsic")
            == disposition_counts.get("retired_intrinsic", 0)
        )
        replay_complete = replay_base_complete and replay_partition_complete
    else:
        replay_complete = bool(
            replay_base_complete and replay.get("n_held_repair") == 0
        )
    pending_state_ok = bool(
        manifest.get("status") == "formal_evaluation_ready"
        and manifest.get("formal_evaluation_ready") is True
        and manifest.get("public_release_ready") is False
        and manifest.get("leaderboard_eligible") is False
        and set(manifest.get("public_release_blockers") or []) == expected_blockers
        and expected_blockers.issubset(exclusion_codes)
        and replay_complete
    )
    published_evidence_ok = False
    if isinstance(formal_evidence, dict):
        logical_binding = formal_evidence.get("logical_batch_manifest")
        realtime_binding = formal_evidence.get("realtime_batch_manifest")
        if isinstance(logical_binding, dict) and isinstance(realtime_binding, dict):
            logical_path, logical_path_ok = _resolve_repo_artifact(
                logical_binding.get("path"), artifact_root=artifact_root
            )
            realtime_path, realtime_path_ok = _resolve_repo_artifact(
                realtime_binding.get("path"), artifact_root=artifact_root
            )
            logical_payload = _safe_load_json(logical_path) if logical_path_ok else {}
            realtime_payload = (
                _safe_load_json(realtime_path) if realtime_path_ok else {}
            )
            expected_suite_sha256 = realtime_contract.get("suite_manifest_sha256")
            published_evidence_ok = bool(
                logical_path_ok
                and realtime_path_ok
                and logical_path.is_file()
                and realtime_path.is_file()
                and _sha256(logical_path) == logical_binding.get("sha256")
                and _sha256(realtime_path) == realtime_binding.get("sha256")
                and _logical_published_manifest_valid(
                    logical_payload,
                    manifest_path=logical_path,
                    tree=tree,
                    suite_sha256=expected_suite_sha256,
                    suite_rows=rows,
                    logical_contract=logical_contract,
                )
                and _realtime_published_manifest_valid(
                    realtime_payload,
                    manifest_path=realtime_path,
                    tree=tree,
                    suite_sha256=expected_suite_sha256,
                    suite_rows=rows,
                    run_contract=run_contract,
                    realtime_contract=realtime_contract,
                )
            )
    published_state_ok = bool(
        manifest.get("status") in {"formal_evaluation_complete", "released"}
        and manifest.get("formal_evaluation_ready") is True
        and manifest.get("public_release_ready") is True
        and manifest.get("leaderboard_eligible") is True
        and not (manifest.get("public_release_blockers") or [])
        and (manifest.get("leaderboard_eligibility") or {}).get("eligible") is True
        and not exclusion_codes
        and published_evidence_ok
        and replay_complete
    )
    release_state_ok = pending_state_ok or published_state_ok

    return {
        "agentic_formal_metadata_complete": metadata_complete,
        "agentic_formal_run_contract_valid": run_contract_ok,
        "agentic_logical_batch_contract_valid": logical_contract_ok,
        "agentic_realtime_batch_contract_valid": realtime_contract_ok,
        "agentic_selection_readiness_binding_valid": selection_ok,
        "agentic_pipeline_artifact_binding_valid": pipeline_ok,
        "agentic_release_pipeline_binding_valid": release_pipeline_ok,
        "agentic_release_tooling_binding_valid": release_tooling_ok,
        "agentic_implementation_tree_binding_valid": tree_ok,
        "agentic_release_state_valid": release_state_ok,
        "agentic_published_evidence_valid": (
            published_evidence_ok if published_state_ok else pending_state_ok
        ),
        "agentic_portable_evidence_closure_valid": portable_closure_ok,
        "agentic_formal_runtime_bundle_valid": (
            runtime_bundle_ok if runtime_bundle_required else True
        ),
        **publication_checks,
    }


def _yaml_provenance_complete(path: Path) -> bool:
    try:
        import yaml
    except ImportError:
        return False
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    if not isinstance(provenance, dict):
        return False
    return all(
        provenance.get(key) for key in ("data_source", "license", "lock_strategy")
    )


def _yaml_dimension_applicability_complete(path: Path) -> bool:
    try:
        import yaml
    except ImportError:
        return False
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = payload.get("backend_config") if isinstance(payload, dict) else None
    applicability = (
        config.get("dimension_applicability") if isinstance(config, dict) else None
    )
    return dimension_applicability_contract_is_valid(applicability)


def _release_closure_checks(
    release: Path,
    manifest: JsonDict,
    rows: list[JsonDict],
    *,
    repo_root: Path,
    portable: bool,
) -> dict[str, bool]:
    check_names = (
        "candidate_closure_manifest_binding_valid",
        "candidate_closure_semantics_valid",
        "candidate_closure_terminal_accounting_valid",
        "candidate_closure_selection_identity_valid",
        "backend_runtime_closure_manifest_binding_valid",
        "backend_runtime_closure_semantics_valid",
        "backend_runtime_closure_terminal_accounting_valid",
    )
    candidate_binding = manifest.get("candidate_closure")
    runtime_binding = manifest.get("backend_runtime_closure")
    if candidate_binding is None and runtime_binding is None:
        return {name: True for name in check_names}
    if not isinstance(candidate_binding, dict):
        candidate_binding = {}
    if not isinstance(runtime_binding, dict):
        runtime_binding = {}

    candidate_path, candidate_contained = _resolve_suite_path(
        release, candidate_binding, "candidate_closure.json"
    )
    candidate = _safe_load_json(candidate_path) if candidate_contained else {}
    candidate_semantics_valid = False
    if candidate:
        try:
            from scripts.finalize_operate_candidate_pool import (  # noqa: PLC0415
                validate_compact_candidate_closure,
            )

            validate_compact_candidate_closure(candidate)
            candidate_semantics_valid = True
        except (KeyError, TypeError, ValueError):
            pass
    candidate_summary = candidate.get("summary")
    candidate_terminal_valid = bool(
        candidate_semantics_valid
        and isinstance(candidate_summary, dict)
        and candidate.get("status") == "candidate_pool_exhausted_non_admitting"
        and candidate_summary.get("n_independent_candidates")
        == candidate_summary.get("n_terminal_candidates")
        and candidate_summary.get("n_unresolved_candidates") == 0
    )
    expected_candidate_binding = {
        "path": "candidate_closure.json",
        "sha256": _sha256(candidate_path) if candidate_path.is_file() else None,
        "schema_version": candidate.get("schema_version"),
        "status": candidate.get("status"),
        "n_independent_candidates": (candidate_summary or {}).get(
            "n_independent_candidates"
        ),
        "n_terminal_candidates": (candidate_summary or {}).get("n_terminal_candidates"),
        "n_unresolved_candidates": (candidate_summary or {}).get(
            "n_unresolved_candidates"
        ),
        "identity_set_sha256": candidate.get("identity_set_sha256"),
    }
    candidate_binding_valid = bool(
        candidate_contained
        and candidate_path == (release / "candidate_closure.json").resolve()
        and candidate_path.is_file()
        and candidate_binding == expected_candidate_binding
    )
    selected_pairs = sorted(
        (
            str((row.get("canonical_identity") or {}).get("scenario_id") or ""),
            str((row.get("canonical_identity") or {}).get("scenario_signature") or ""),
        )
        for row in candidate.get("candidates") or []
        if isinstance(row, dict)
        and row.get("final_disposition") == "selected_for_promotion"
    )
    core_pairs = sorted(
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in rows
    )
    source_path = release / "protocol21_source_suite.json"
    source = _safe_load_json(source_path) if source_path.is_file() else {}
    imported_pairs: list[tuple[str, str]] = []
    partition_valid = False
    try:
        from scripts.build_protocol21_incremental_union import (  # noqa: PLC0415
            validate_candidate_import_partition,
        )

        _base_pairs, imported_pairs = validate_candidate_import_partition(source)
        partition_valid = True
    except (KeyError, TypeError, ValueError):
        pass
    candidate_selection_valid = bool(
        candidate_semantics_valid
        and partition_valid
        and selected_pairs == sorted(imported_pairs)
        and set(imported_pairs).issubset(core_pairs)
    )

    source_sha256 = _sha256(source_path) if source_path.is_file() else ""
    runtime_path, runtime_contained = _resolve_suite_path(
        release, runtime_binding, "backend_runtime_closure.json"
    )
    runtime = _safe_load_json(runtime_path) if runtime_contained else {}
    runtime_semantics_valid = False
    if runtime and source_sha256:
        try:
            from scripts.promote_operate_release import (  # noqa: PLC0415
                _repo_uv_lock_sha256,
                _validate_backend_runtime_closure,
            )

            _validate_backend_runtime_closure(
                runtime,
                release_id=str(manifest.get("release_id") or ""),
                source_suite_sha256=source_sha256,
                expected_uv_lock_sha256=(
                    None if portable else _repo_uv_lock_sha256(repo_root)
                ),
            )
            from scripts.build_operate_backend_runtime_closure import (  # noqa: PLC0415
                validate_opendss_runtime_asset_closure,
            )

            validate_opendss_runtime_asset_closure(
                repo_root=repo_root,
                source_suite=source,
                closure=runtime,
                require_live_contract=not portable,
            )
            runtime_semantics_valid = True
        except (ImportError, KeyError, TypeError, ValueError):
            pass
    runtime_summary = runtime.get("summary")
    runtime_terminal_valid = bool(
        runtime_semantics_valid
        and isinstance(runtime_summary, dict)
        and runtime.get("status") == "backend_runtime_closure_complete"
        and runtime.get("terminal") is True
        and runtime_summary.get("n_unresolved") == 0
    )
    expected_runtime_binding = {
        "path": "backend_runtime_closure.json",
        "sha256": _sha256(runtime_path) if runtime_path.is_file() else None,
        "schema_version": runtime.get("schema_version"),
        "n_archived_files": (runtime_summary or {}).get("n_archived_files"),
        "n_external_sources": (runtime_summary or {}).get("n_external_sources"),
        "n_backend_links": (runtime_summary or {}).get("n_backend_links"),
        "n_runtime_packages": (runtime_summary or {}).get("n_runtime_packages"),
        "identity_sha256": runtime.get("identity_sha256"),
    }
    runtime_binding_valid = bool(
        runtime_contained
        and runtime_path == (release / "backend_runtime_closure.json").resolve()
        and runtime_path.is_file()
        and runtime_binding == expected_runtime_binding
    )
    return {
        "candidate_closure_manifest_binding_valid": candidate_binding_valid,
        "candidate_closure_semantics_valid": candidate_semantics_valid,
        "candidate_closure_terminal_accounting_valid": candidate_terminal_valid,
        "candidate_closure_selection_identity_valid": candidate_selection_valid,
        "backend_runtime_closure_manifest_binding_valid": runtime_binding_valid,
        "backend_runtime_closure_semantics_valid": runtime_semantics_valid,
        "backend_runtime_closure_terminal_accounting_valid": runtime_terminal_valid,
    }


def build_protocol21_core_integrity_report(
    release: Path,
    *,
    portable: bool = False,
    artifact_root: Path | None = None,
    manifest_override: JsonDict | None = None,
) -> JsonDict:
    """Integrity checks for slim ``protocol21-core-v1`` Core cuts (v0.52+)."""
    manifest_path = release / "manifest.json"
    manifest = (
        deepcopy(manifest_override)
        if manifest_override is not None
        else _load_json(manifest_path)
    )
    core_entry = _manifest_suite_entry(manifest, "core_suite")
    core_path, core_path_contained = _resolve_suite_path(
        release, core_entry, "core_suite.json"
    )
    core = _load_json(core_path)
    rows = list(core.get("scenarios") or [])
    ids = [_scenario_id(row) for row in rows]
    signatures = [str(row.get("scenario_signature") or "") for row in rows]
    source_keys = [str(row.get("source_denominator_key") or "") for row in rows]
    duplicate_ids = _duplicates(ids)
    duplicate_signatures = _duplicates([s for s in signatures if s])
    duplicate_keys = _duplicates([k for k in source_keys if k])
    fingerprints = [str(row.get("structural_fingerprint") or "") for row in rows]
    duplicate_fingerprints = _duplicates([fp for fp in fingerprints if fp])
    missing_fingerprints = [
        _scenario_id(row)
        for row in rows
        if not str(row.get("structural_fingerprint") or "")
    ]
    missing_physical = [
        _scenario_id(row)
        for row in rows
        if row.get("physical_source_key") in (None, "", {}, [])
    ]
    physical_identities: list[str] = []
    for row in rows:
        identity = row.get("physical_source_key")
        if identity in (None, "", {}, []):
            continue
        try:
            physical_identities.append(
                json.dumps(
                    identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
            )
        except (TypeError, ValueError):
            missing_physical.append(_scenario_id(row))
    n_physical = len(set(physical_identities))
    expected_physical = manifest.get("n_physical_sources")
    repo = (artifact_root or _repo_root()).resolve()
    missing_yaml: list[str] = []
    reports_paths: list[str] = []
    incomplete_ids: list[str] = []
    applicability_incomplete_ids: list[str] = []
    yaml_sha_mismatch: list[str] = []
    requires_applicability = bool(manifest.get("core_settings_stamp"))
    for row in rows:
        rel = str(row.get("path") or "")
        if rel.startswith("reports/"):
            reports_paths.append(rel)
        yaml_path, yaml_path_contained = _resolve_repo_artifact(rel, artifact_root=repo)
        if not yaml_path_contained or not yaml_path.is_file():
            missing_yaml.append(rel)
            continue
        expected_yaml_sha = str(row.get("yaml_sha256") or "")
        if not expected_yaml_sha or _sha256(yaml_path) != expected_yaml_sha:
            yaml_sha_mismatch.append(_scenario_id(row))
        if not _yaml_provenance_complete(yaml_path):
            incomplete_ids.append(_scenario_id(row))
        if requires_applicability and not _yaml_dimension_applicability_complete(
            yaml_path
        ):
            applicability_incomplete_ids.append(_scenario_id(row))
    sha256_required = _requires_suite_sha256(str(manifest.get("release_id") or ""))
    core_sha_ok, core_sha_required = _sha256_check(
        core_entry, core_path, sha256_required=sha256_required
    )
    expected_n = core.get("n_scenarios")
    backend_descriptors = manifest.get("backend_descriptors")
    descriptor_keys = (
        set(backend_descriptors) if isinstance(backend_descriptors, dict) else set()
    )
    by_backend = (
        core.get("by_backend") if isinstance(core.get("by_backend"), dict) else {}
    )
    missing_descriptors = sorted(set(by_backend) - descriptor_keys)
    eligibility = manifest.get("leaderboard_eligibility")
    eligibility_codes: list[str] = []
    eligibility_ok = False
    if isinstance(eligibility, dict) and isinstance(eligibility.get("eligible"), bool):
        exclusions = eligibility.get("suite_exclusions")
        if isinstance(exclusions, list):
            ok_exclusions = True
            for item in exclusions:
                reason = item.get("reason") if isinstance(item, dict) else None
                code = str((reason or {}).get("code") or "")
                if not code:
                    ok_exclusions = False
                    break
                eligibility_codes.append(code)
            blockers = {
                str(code)
                for code in (manifest.get("public_release_blockers") or [])
                if code
            }
            if eligibility.get("eligible") is True:
                eligibility_ok = ok_exclusions and not exclusions and not blockers
            else:
                eligibility_ok = (
                    bool(exclusions)
                    and ok_exclusions
                    and blockers.issubset(set(eligibility_codes))
                )
    datasets = manifest.get("datasets")
    datasets_ok = isinstance(datasets, dict) and bool(datasets)
    agentic_contract = _agentic_contract_required(manifest)
    agentic_checks = _agentic_formal_checks(
        release,
        manifest,
        core,
        rows,
        portable=portable,
        artifact_root=repo,
    )
    diagnostics = {
        "live_release_tooling_matches_promoted_snapshot": (
            None
            if portable
            else implementation_identity(repo).get("release_tooling_sha256")
            == manifest.get("release_tooling_sha256")
        )
    }
    closure_checks = _release_closure_checks(
        release,
        manifest,
        rows,
        repo_root=repo,
        portable=portable,
    )
    identity_closure_required = _release_identity_closure_required(manifest, core)
    release_id = str(manifest.get("release_id") or "")
    release_identity_closed = bool(
        not identity_closure_required
        or (
            release_id
            and release.name == release_id
            and core.get("release_id") == release_id
        )
    )
    selection_policy_matches = bool(
        not identity_closure_required
        or (
            manifest.get("core_selection_policy")
            and core.get("selection_policy") == manifest.get("core_selection_policy")
        )
    )
    settings_stamp_matches = bool(
        not identity_closure_required
        or (
            manifest.get("core_settings_stamp")
            and core.get("core_settings_stamp") == manifest.get("core_settings_stamp")
        )
    )
    scoring_version_matches = bool(
        not identity_closure_required
        or (
            manifest.get("scoring_version")
            and core.get("scoring_version") == manifest.get("scoring_version")
        )
    )
    checks = {
        "release_identity_closed": release_identity_closed,
        "core_selection_policy_matches_manifest": selection_policy_matches,
        "core_settings_stamp_matches_manifest": settings_stamp_matches,
        "core_scoring_version_matches_manifest": scoring_version_matches,
        "core_count_matches_manifest": core_entry.get("n_scenarios") == expected_n,
        "core_len_matches_declared": len(rows) == expected_n,
        "core_sha256_matches_manifest": core_sha_ok,
        "unique_core_ids": not duplicate_ids and all(ids),
        "unique_core_signatures": not duplicate_signatures and all(signatures),
        "unique_source_denominator_keys": not duplicate_keys and all(source_keys),
        "unique_structural_fingerprints": (
            not duplicate_fingerprints and not missing_fingerprints
        ),
        "core_physical_source_keys_present": not missing_physical,
        "core_physical_source_count_matches_manifest": (
            expected_physical is not None and n_physical == expected_physical
        ),
        "core_yaml_present": not missing_yaml,
        "core_yaml_sha256_matches": not yaml_sha_mismatch,
        "core_yaml_not_under_reports": not reports_paths,
        "core_provenance_complete": not incomplete_ids,
        "core_dimension_applicability_complete": not applicability_incomplete_ids,
        "suite_paths_contained": core_path_contained,
        "backend_descriptors_cover_core": not missing_descriptors,
        "cascade_bus_schema_version_present": bool(
            manifest.get("cascade_bus_schema_version")
        ),
        "leaderboard_eligibility_machine_readable": eligibility_ok,
        "datasets_block_present": datasets_ok,
        "no_legacy_primary_required": True,
        **closure_checks,
        **agentic_checks,
    }
    issues: list[JsonDict] = []
    for check, passed in checks.items():
        if not passed:
            issues.append({"code": check})
    if duplicate_ids:
        issues.append({"code": "duplicate_core_ids", "ids": duplicate_ids[:25]})
    if missing_yaml:
        issues.append({"code": "missing_core_yaml", "paths": missing_yaml[:25]})
    if reports_paths:
        issues.append({"code": "core_yaml_under_reports", "paths": reports_paths[:25]})
    if incomplete_ids:
        issues.append(
            {
                "code": "core_provenance_incomplete",
                "count": len(incomplete_ids),
                "ids": incomplete_ids[:25],
            }
        )
    if applicability_incomplete_ids:
        issues.append(
            {
                "code": "core_dimension_applicability_incomplete",
                "count": len(applicability_incomplete_ids),
                "ids": applicability_incomplete_ids[:25],
            }
        )
    if duplicate_fingerprints or missing_fingerprints:
        issues.append(
            {
                "code": "structural_fingerprint_incomplete",
                "duplicate": duplicate_fingerprints[:25],
                "missing": missing_fingerprints[:25],
            }
        )
    if yaml_sha_mismatch:
        issues.append(
            {
                "code": "yaml_sha256_mismatch",
                "count": len(yaml_sha_mismatch),
                "ids": yaml_sha_mismatch[:25],
            }
        )
    if missing_physical:
        issues.append(
            {
                "code": "missing_physical_source_key",
                "count": len(missing_physical),
                "ids": missing_physical[:25],
            }
        )
    if missing_descriptors:
        issues.append(
            {
                "code": "missing_backend_descriptors",
                "backends": missing_descriptors,
            }
        )
    ok = all(checks.values())
    runtime_evidence_verified = bool(
        agentic_contract
        and not portable
        and agentic_checks["agentic_pipeline_artifact_binding_valid"]
    )
    portable_formal_input_ready = bool(
        agentic_contract
        and agentic_checks["agentic_formal_runtime_bundle_valid"]
        and agentic_checks["agentic_portable_evidence_closure_valid"]
    )
    formal_run_ready = bool(
        ok
        and (
            portable_formal_input_ready
            if _release_version(manifest) >= (0, 59, 0)
            else runtime_evidence_verified
        )
    )
    return {
        "release": str(release),
        "release_id": manifest.get("release_id"),
        "manifest_schema_version": manifest.get("manifest_schema_version"),
        "manifest": {
            "path": str(manifest_path),
            "scoring_version": manifest.get("scoring_version"),
            "public_release_ready": manifest.get("public_release_ready"),
            "leaderboard_eligible": manifest.get("leaderboard_eligible"),
        },
        "core": {
            "path": str(core_path),
            "n_scenarios": expected_n,
            "len_scenarios": len(rows),
            "sha256": _sha256(core_path),
            "manifest_sha256": core_entry.get("sha256"),
            "sha256_required": core_sha_required,
            "duplicate_ids": duplicate_ids,
            "missing_yaml": missing_yaml[:25],
            "n_physical_sources": n_physical,
            "expected_physical_sources": expected_physical,
            "n_structural_fingerprints": len(set(fingerprints)),
        },
        "checks": checks,
        "diagnostics": diagnostics,
        "issues": issues,
        "verification_mode": "portable" if portable else "full",
        "runtime_evidence_bytes_verified": runtime_evidence_verified,
        "portable_formal_input_ready": portable_formal_input_ready,
        "formal_run_ready": formal_run_ready,
        "ok": ok,
    }


def build_release_integrity_report(
    release: Path = DEFAULT_RELEASE,
    *,
    portable: bool = False,
    artifact_root: Path | None = None,
    manifest_override: JsonDict | None = None,
) -> JsonDict:
    manifest_path = release / "manifest.json"
    manifest = (
        deepcopy(manifest_override)
        if manifest_override is not None
        else _load_json(manifest_path)
    )
    if manifest.get("manifest_schema_version") == "protocol21-core-v1":
        return build_protocol21_core_integrity_report(
            release,
            portable=portable,
            artifact_root=artifact_root,
            manifest_override=manifest,
        )
    primary_entry = _manifest_suite_entry(manifest, "primary_suite")
    core_entry = _manifest_suite_entry(manifest, "core_suite")
    primary_path, primary_path_contained = _resolve_suite_path(
        release, primary_entry, "primary_suite.json"
    )
    core_path, core_path_contained = _resolve_suite_path(
        release, core_entry, "core_suite.json"
    )
    primary = _load_json(primary_path)
    core = _load_json(core_path)
    primary_rows = list(primary.get("scenarios") or [])
    core_rows = list(core.get("scenarios") or [])
    primary_ids = [_scenario_id(row) for row in primary_rows]
    core_ids = [_scenario_id(row) for row in core_rows]
    primary_duplicate_ids = _duplicates(primary_ids)
    core_duplicate_ids = _duplicates(core_ids)
    release_id = str(manifest.get("release_id") or "")
    allow_legacy_source_lock = False
    primary_prov = _provenance_summary(
        primary_rows, allow_legacy_source_lock=allow_legacy_source_lock
    )
    core_prov = _provenance_summary(
        core_rows, allow_legacy_source_lock=allow_legacy_source_lock
    )
    sha256_required = _requires_suite_sha256(release_id)
    primary_sha_ok, primary_sha_required = _sha256_check(
        primary_entry, primary_path, sha256_required=sha256_required
    )
    core_sha_ok, core_sha_required = _sha256_check(
        core_entry, core_path, sha256_required=sha256_required
    )

    primary_manifest_n = primary_entry.get("n_scenarios")
    core_manifest_n = core_entry.get("n_scenarios")
    checks = {
        "primary_count_matches_manifest": primary_manifest_n
        == primary.get("n_scenarios"),
        "core_count_matches_manifest": core_manifest_n == core.get("n_scenarios"),
        "primary_len_matches_declared": len(primary_rows) == primary.get("n_scenarios"),
        "core_len_matches_declared": len(core_rows) == core.get("n_scenarios"),
        "primary_sha256_matches_manifest": primary_sha_ok,
        "core_sha256_matches_manifest": core_sha_ok,
        "unique_primary_ids": not primary_duplicate_ids and all(primary_ids),
        "unique_core_ids": not core_duplicate_ids and all(core_ids),
        "core_subset_primary": set(core_ids).issubset(set(primary_ids)),
        "primary_provenance_complete": primary_prov["incomplete"] == 0,
        "core_provenance_complete": core_prov["incomplete"] == 0,
        "suite_paths_contained": primary_path_contained and core_path_contained,
    }
    issues: list[JsonDict] = []
    for check, passed in checks.items():
        if not passed:
            issues.append({"code": check})
    if primary_duplicate_ids:
        issues.append(
            {"code": "duplicate_primary_ids", "ids": primary_duplicate_ids[:25]}
        )
    if core_duplicate_ids:
        issues.append({"code": "duplicate_core_ids", "ids": core_duplicate_ids[:25]})
    if primary_prov["incomplete"]:
        issues.append(
            {
                "code": "primary_provenance_incomplete",
                "count": primary_prov["incomplete"],
                "ids": primary_prov["incomplete_ids"],
            }
        )
    if core_prov["incomplete"]:
        issues.append(
            {
                "code": "core_provenance_incomplete",
                "count": core_prov["incomplete"],
                "ids": core_prov["incomplete_ids"],
            }
        )
    if not primary_path_contained:
        issues.append(
            {
                "code": "suite_path_escapes_release",
                "suite": "primary",
                "path": str(primary_path),
            }
        )
    if not core_path_contained:
        issues.append(
            {
                "code": "suite_path_escapes_release",
                "suite": "core",
                "path": str(core_path),
            }
        )

    return {
        "release": str(release),
        "release_id": manifest.get("release_id"),
        "manifest": {
            "path": str(manifest_path),
            "publishable": manifest.get("publishable"),
            "scoring_version": manifest.get("scoring_version"),
        },
        "release_metadata": {
            "publishable": manifest.get("publishable"),
            "full_row_audit_passed": (manifest.get("quality_gates") or {}).get(
                "full_row_audit_passed"
            ),
            "issues": list(manifest.get("issues") or []),
        },
        "primary": {
            "path": str(primary_path),
            "n_scenarios": primary.get("n_scenarios"),
            "len_scenarios": len(primary_rows),
            "manifest_n_scenarios": primary_manifest_n,
            "sha256": _sha256(primary_path),
            "manifest_sha256": primary_entry.get("sha256"),
            "sha256_required": primary_sha_required,
            "duplicate_ids": primary_duplicate_ids,
        },
        "core": {
            "path": str(core_path),
            "n_scenarios": core.get("n_scenarios"),
            "len_scenarios": len(core_rows),
            "manifest_n_scenarios": core_manifest_n,
            "sha256": _sha256(core_path),
            "manifest_sha256": core_entry.get("sha256"),
            "sha256_required": core_sha_required,
            "duplicate_ids": core_duplicate_ids,
        },
        "provenance": {
            "primary_complete": primary_prov["complete"],
            "primary_incomplete": primary_prov["incomplete"],
            "primary_incomplete_ids": primary_prov["incomplete_ids"],
            "core_complete": core_prov["complete"],
            "core_incomplete": core_prov["incomplete"],
            "core_incomplete_ids": core_prov["incomplete_ids"],
            "legacy_source_lock_allowed": allow_legacy_source_lock,
        },
        "checks": checks,
        "issues": issues,
        "ok": all(checks.values()),
    }


def _resolve_bundle_member(bundle_root: Path, raw_path: object) -> tuple[Path, bool]:
    value = str(raw_path or "")
    relative = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return bundle_root / value, False
    resolved = (bundle_root / Path(*relative.parts)).resolve()
    try:
        resolved.relative_to(bundle_root)
    except ValueError:
        return resolved, False
    return resolved, True


def build_portable_bundle_integrity_report(bundle_root: Path) -> JsonDict:
    """Verify a relocated public bundle without consulting the repository."""
    bundle_root = bundle_root.resolve()
    bundle_manifest_path = bundle_root / "MANIFEST.json"
    bundle_manifest = _safe_load_json(bundle_manifest_path)
    release_id = str(bundle_manifest.get("release_id") or "")
    _, release_id_safe = _resolve_bundle_member(bundle_root, release_id)
    release_id_safe = release_id_safe and len(PurePosixPath(release_id).parts) == 1
    raw_files = bundle_manifest.get("files")
    files = raw_files if isinstance(raw_files, dict) else {}

    paths_safe = bool(files)
    hashes_valid = bool(files)
    files_match = bool(files)
    for raw_path, expected_sha256 in files.items():
        path, path_safe = _resolve_bundle_member(bundle_root, raw_path)
        digest_valid = _valid_sha256(expected_sha256)
        paths_safe = paths_safe and path_safe
        hashes_valid = hashes_valid and digest_valid
        files_match = bool(
            files_match
            and path_safe
            and digest_valid
            and path.is_file()
            and _sha256(path) == expected_sha256
        )

    release_prefix = f"release/{release_id}"
    release_path = bundle_root / release_prefix
    release_manifest_rel = f"{release_prefix}/manifest.json"
    core_rel = f"{release_prefix}/core_suite.json"
    release_manifest = _safe_load_json(bundle_root / release_manifest_rel)
    core = _safe_load_json(bundle_root / core_rel)
    rows = core.get("scenarios")
    scenario_rows = rows if isinstance(rows, list) else []
    agentic_release = _agentic_contract_required(release_manifest)
    required_release_files = {
        "README.md",
        release_manifest_rel,
        core_rel,
        f"{release_prefix}/README.md",
    }
    if agentic_release:
        required_release_files.update(
            {
                f"{release_prefix}/protocol21_source_suite.json",
                f"{release_prefix}/protocol21_public_evidence_bundle.json",
            }
        )
    scenario_paths = {
        str(row.get("path") or "") for row in scenario_rows if isinstance(row, dict)
    }
    scenario_paths_valid = bool(scenario_rows) and all(scenario_paths)
    scenario_files_declared = bool(
        scenario_paths_valid and scenario_paths.issubset(files)
    )
    required_files_declared = required_release_files.issubset(files)
    counts_match = bool(
        isinstance(bundle_manifest.get("n_files"), int)
        and not isinstance(bundle_manifest.get("n_files"), bool)
        and bundle_manifest.get("n_files") == len(files)
        and isinstance(bundle_manifest.get("n_scenarios"), int)
        and not isinstance(bundle_manifest.get("n_scenarios"), bool)
        and bundle_manifest.get("n_scenarios") == len(scenario_rows)
        and core.get("n_scenarios") == len(scenario_rows)
    )
    manifest_valid = bool(
        bundle_manifest_path.is_file()
        and bundle_manifest
        and release_id_safe
        and bundle_manifest.get("bundle_kind") == "portable_core_source_manifest"
        and isinstance(raw_files, dict)
        and release_manifest.get("release_id") == release_id
    )

    try:
        release_report = build_release_integrity_report(
            release_path,
            portable=True,
            artifact_root=bundle_root,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        release_report = {
            "release": str(release_path),
            "release_id": release_id or None,
            "checks": {},
            "issues": [{"code": "bundle_release_integrity_unreadable"}],
            "verification_mode": "portable",
            "runtime_evidence_bytes_verified": False,
            "formal_run_ready": False,
            "ok": False,
        }

    checks = {
        "bundle_manifest_valid": manifest_valid,
        "bundle_file_paths_safe": paths_safe,
        "bundle_file_hashes_valid": hashes_valid,
        "bundle_files_match_manifest": files_match,
        "bundle_required_release_files_declared": required_files_declared,
        "bundle_scenario_files_declared": scenario_files_declared,
        "bundle_counts_match": counts_match,
        "release_portable_integrity": release_report.get("ok") is True,
    }
    issues = [{"code": name} for name, passed in checks.items() if not passed]
    return {
        "bundle_root": str(bundle_root),
        "release_id": release_id or None,
        "bundle_manifest": str(bundle_manifest_path),
        "bundle_manifest_sha256": (
            _sha256(bundle_manifest_path) if bundle_manifest_path.is_file() else None
        ),
        "checks": checks,
        "issues": issues,
        "release_report": release_report,
        "verification_mode": "portable_bundle",
        "runtime_evidence_bytes_verified": False,
        "formal_run_ready": False,
        "ok": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", nargs="?", type=Path)
    parser.add_argument(
        "--portable",
        action="store_true",
        help=(
            "verify only carried Core and compact evidence closure; this does "
            "not verify runtime evidence bytes or authorize a formal run"
        ),
    )
    parser.add_argument(
        "--bundle-root",
        type=Path,
        help=(
            "verify a relocated portable bundle rooted at MANIFEST.json; "
            "requires --portable and cannot be combined with release"
        ),
    )
    args = parser.parse_args()
    if args.bundle_root is not None:
        if not args.portable:
            parser.error("--bundle-root requires --portable")
        if args.release is not None:
            parser.error("--bundle-root cannot be combined with release")
        report = build_portable_bundle_integrity_report(args.bundle_root)
    else:
        report = build_release_integrity_report(
            args.release or DEFAULT_RELEASE, portable=args.portable
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
