#!/usr/bin/env python3
"""
Multi-model LLM batch evaluation for OPERATE.

Loads API settings from environment (or ~/.zhsrc / ~/.zshrc exports), runs each model
as ``llm_agent`` on a configurable scenario slice, and writes leaderboard,
analysis, audit, and plotting artifacts.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fnmatch
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# P3-2: batch helpers now live in ``runner/``. Import them from the public
# source but bind to the legacy ``_``-private names so call sites and test
# monkeypatches (``monkeypatch.setattr(mod, "_expand_scenarios", ...)``)
# keep working without modification. ``F401`` is intentional on the
# re-exports bound under ``_``-private aliases.
from baselines import LLMConfig  # noqa: E402
from baselines.llm_agent import (  # noqa: E402
    TOKEN_COUNT_METHOD_UTF8_BYTES,
    TOKEN_COUNT_VERSION_V1,
    frozen_model_capabilities,
    frozen_model_tool_choice_support,
    parse_tencent_quota_reset,
    prompt_contract_sha256,
    public_provider_url,
)
from core.event_protocol import EVENT_DECISION_CONTRACT_VERSION  # noqa: E402
from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import (  # noqa: E402
    canonicalize_repo_owned_paths,
    resolve_binding_path,
)
from core.runtime_closure import (  # noqa: E402
    validate_live_backend_runtime_closure,
)
from core.sidecar.sumo_sidecar import probe_sumo_transport  # noqa: E402
from core.suite_identity import (  # noqa: E402
    canonical_scenario_slug,
    canonical_suite_manifest_sha256,
)
from evaluation import (  # noqa: E402
    OPERATIONAL_AGENCY_DIMENSIONS,  # noqa: F401
    OPERATIONAL_AGENCY_PROFILE_VERSION,  # noqa: F401
    SCORING_VERSION,
    build_leaderboard,
    operational_agency_profile_is_consistent,
)
from evaluation.discrimination import build_discrimination_report  # noqa: E402
from evaluation.trajectory_paths import resolve_batch_path  # noqa: E402
from evaluation.leaderboard import (  # noqa: E402
    PRIMARY_LEADERBOARD_FORMULA_VERSION,
    PrimaryLeaderboardContractError,
    infer_primary_leaderboard,
)
from evaluation.scorer import (  # noqa: E402
    TASK_COMPLETION_INPUT_UNIT,
    TASK_COMPLETION_SCORE_UNIT,
    WEIGHTED_EQUITY_FORMULA_VERSION,
)
from run import load_scenario_yaml  # noqa: E402
from runner import (  # noqa: E402
    EVALUATION_IMPLEMENTATION_FINGERPRINT,
    EVALUATION_PROTOCOL_VERSION,
)
from runner import expand_scenarios as _expand_scenarios  # noqa: E402, F401
from runner import (  # noqa: E402
    recompute_signature_with_seed as _recompute_signature_with_seed,  # noqa: F401
)
from runner import run_one_safe as _run_one_safe  # noqa: E402, F401
from scripts.analyze_batch_results import analyze_output_dir  # noqa: E402
from evaluation.batch_status import (  # noqa: E402
    execution_status_counts as _execution_status_counts,
    row_is_quota_exhausted as _row_is_quota_exhausted,
)
from scripts.analyze_decision_impact import (  # noqa: E402
    build_report as build_decision_impact_report,
)
from scripts.analyze_decision_impact import (  # noqa: E402
    write_markdown as write_decision_impact_markdown,
)
from scripts.audit_agent_failure_recipes import (  # noqa: E402
    build_report as build_agent_failure_recipes_report,
)
from scripts.audit_agent_failure_recipes import (  # noqa: E402
    write_markdown as write_agent_failure_recipes_markdown,
)
from scripts.audit_eval_logs import audit as audit_logs  # noqa: E402
from scripts.audit_eval_logs import write_markdown as write_log_audit_markdown  # noqa: E402
from scripts.audit_evidence_applicability import (  # noqa: E402
    build_report as build_evidence_applicability_report,
)
from scripts.audit_evidence_applicability import (  # noqa: E402
    write_markdown as write_evidence_applicability_markdown,
)
from scripts.audit_staleness_consumption import (  # noqa: E402
    build_report as build_staleness_consumption_report,
)
from scripts.audit_staleness_consumption import (  # noqa: E402
    write_markdown as write_staleness_consumption_markdown,
)
from scripts.audit_tool_effects import (  # noqa: E402
    build_report as build_tool_effect_report,
)
from scripts.audit_tool_effects import (  # noqa: E402
    write_markdown as write_tool_effect_markdown,
)
from scripts.build_protocol21_core_readiness import (  # noqa: E402
    build_readiness_from_paths,
)

LOGGER = logging.getLogger("batch_llm_eval")
PRIMARY_INFERENCE_VERSION = "physical_cluster_hierarchical_bootstrap_randomization_v1"
CANONICAL_WAKEUP_POLICY = {
    "session_start": True,
    "typed_actionable_events": True,
    "agent_scheduled_reviews": True,
    "harness_periodic_supervisory_scan": False,
    "unknown_events_actionable": False,
}

SCENARIO_SLICES: dict[str, list[str]] = {}

DynamicSliceSpec = tuple[str, str, dict[str, set[str]]]
DYNAMIC_SCENARIO_SLICES: dict[str, DynamicSliceSpec] = {}
PROTOCOL21_FORMAL_SLICES: frozenset[str] = frozenset()
FORMAL_CORE_STAGE_FILES = {
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
FORMAL_CORE_PIPELINE_STAGES = tuple(FORMAL_CORE_STAGE_FILES)


def resolve_formal_manifest_slice(
    manifest_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Resolve a versioned release manifest to its exact readiness artifact."""
    if manifest_path.is_symlink():
        raise ValueError("formal manifest must be the canonical manifest.json")
    manifest_path = manifest_path.resolve()
    if (
        manifest_path.name != "manifest.json"
        or not manifest_path.is_file()
    ):
        raise ValueError("formal manifest must be the canonical manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release_root = (repo_root / "release").resolve()
    try:
        manifest_path.relative_to(release_root)
    except ValueError as exc:
        raise ValueError("formal manifest must live under release/") from exc

    formal_contract = manifest.get("formal_batch_contract") or {}
    raw_runtime_root = str(formal_contract.get("runtime_evidence_root") or "")
    if raw_runtime_root:
        runtime_root = Path(raw_runtime_root)
        if not runtime_root.is_absolute():
            runtime_root = repo_root / runtime_root
        runtime_root = runtime_root.resolve()
        try:
            runtime_root.relative_to(release_root)
        except ValueError as exc:
            raise ValueError(
                "formal runtime evidence root must live under release/"
            ) from exc
        if not runtime_root.is_dir():
            raise ValueError("formal runtime evidence root missing")
    else:
        runtime_root = manifest_path.parent
    selection_source = str(formal_contract.get("selection_source") or "")
    raw_readiness_path = selection_source.split("#", 1)[0]
    if not raw_readiness_path:
        pipeline = manifest.get("pipeline_artifacts") or {}
        pipeline_path = str(pipeline.get("path") or "")
        if pipeline_path:
            raw_readiness_path = (
                f"{pipeline_path.rstrip('/')}/protocol2_v21_core_readiness.json"
            )
    if not raw_readiness_path:
        readiness_binding = manifest.get("readiness") or {}
        raw_readiness_path = str(readiness_binding.get("path") or "")
    if not raw_readiness_path:
        raise ValueError("formal manifest readiness path missing")

    readiness_path = Path(raw_readiness_path)
    if not readiness_path.is_absolute():
        readiness_path = repo_root / readiness_path
    readiness_path = readiness_path.resolve()
    try:
        readiness_relative = readiness_path.relative_to(release_root)
    except ValueError as exc:
        raise ValueError("formal readiness must live under release/") from exc
    if len(readiness_relative.parts) < 2 or not readiness_path.is_file():
        raise ValueError("formal manifest readiness artifact missing")
    try:
        readiness_path.relative_to(runtime_root)
    except ValueError as exc:
        raise ValueError(
            "formal manifest and readiness must live in the same release directory"
        ) from exc

    actual_hash = hashlib.sha256(readiness_path.read_bytes()).hexdigest()
    runtime_bundle_binding = manifest.get("formal_runtime_bundle")
    if isinstance(runtime_bundle_binding, dict):
        bundle_name = str(runtime_bundle_binding.get("path") or "")
        if (
            bundle_name != "formal_runtime_bundle.json"
            or readiness_path != manifest_path.parent / bundle_name
        ):
            raise ValueError("formal runtime bundle path mismatch")
        declared_hash = str(runtime_bundle_binding.get("sha256") or "")
    else:
        declared_hash = str(
            (manifest.get("pipeline_artifacts") or {}).get("readiness_sha256")
            or (manifest.get("readiness") or {}).get("sha256")
            or ""
        )
    if not declared_hash:
        raise ValueError("formal manifest readiness hash missing")
    if declared_hash != actual_hash:
        raise ValueError("formal manifest readiness hash mismatch")
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    formal_run_contract = (
        readiness.get("formal_run_contract") if isinstance(readiness, dict) else None
    )
    if not isinstance(formal_run_contract, dict) or not formal_run_contract:
        raise ValueError("formal_run_contract_missing")
    release_id = str(manifest.get("release_id") or "")
    historical_wakeup_contract = release_id in {
        "operate_v0_58_0",
        "operate_v0_59_0",
        "operate_v0_60_0",
    }
    if (
        not historical_wakeup_contract
        and formal_run_contract.get("wakeup_policy") != CANONICAL_WAKEUP_POLICY
    ):
        raise ValueError("formal wakeup policy mismatch")
    if (
        isinstance(readiness, dict)
        and readiness.get("schema_version") == "operate-formal-runtime-bundle-v1"
    ):
        from scripts.verify_release_integrity import (  # noqa: PLC0415
            build_release_integrity_report,
        )

        release_dir = manifest_path.parent
        if runtime_root != release_dir:
            raise ValueError("formal runtime bundle root mismatch")
        report = build_release_integrity_report(
            release_dir,
            portable=True,
            artifact_root=repo_root,
        )
        if report.get("ok") is not True or not (
            report.get("checks") or {}
        ).get("agentic_formal_runtime_bundle_valid"):
            raise ValueError("formal runtime bundle integrity mismatch")
        live_tree = implementation_identity(repo_root)["implementation_tree_sha256"]
        if not (
            manifest.get("implementation_tree_sha256") == live_tree
            and readiness.get("implementation_tree_sha256") == live_tree
        ):
            raise ValueError("formal implementation tree mismatch")
        replay_binding = manifest.get("protocol21_replay")
        if not isinstance(replay_binding, dict):
            raise ValueError("formal source suite binding missing")
        source_path = release_dir / "protocol21_source_suite.json"
        declared_source_hash = str(replay_binding.get("source_suite_sha256") or "")
        if not (
            replay_binding.get("source_suite")
            == source_path.relative_to(repo_root).as_posix()
            and source_path.is_file()
            and declared_source_hash
            == hashlib.sha256(source_path.read_bytes()).hexdigest()
        ):
            raise ValueError("formal source suite binding mismatch")
        runtime_closure = validate_live_backend_runtime_closure(
            repo_root=repo_root,
            release_root=release_dir,
            release_id=str(manifest.get("release_id") or ""),
            source_suite_sha256=declared_source_hash,
            binding=manifest.get("backend_runtime_closure") or {},
        )
        release_pipeline_hash = str(
            manifest.get("core_release_pipeline_sha256") or ""
        )
        release_id = str(manifest.get("release_id") or "")
        release_tooling_hash = str(manifest.get("release_tooling_sha256") or "")
        if not (
            release_id == release_dir.name
            and re.fullmatch(r"[0-9a-f]{64}", release_pipeline_hash)
            and re.fullmatch(r"[0-9a-f]{64}", release_tooling_hash)
            and readiness.get("core_release_pipeline_sha256")
            == release_pipeline_hash
            and readiness.get("release_tooling_sha256") == release_tooling_hash
            and replay_binding.get("core_release_pipeline_sha256")
            == release_pipeline_hash
        ):
            raise ValueError("formal release runtime binding mismatch")
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        bundle_binding = manifest.get("formal_runtime_bundle") or {}
        core_binding = manifest.get("core_suite") or {}
        evidence_binding = readiness.get("public_evidence") or {}
        candidate_binding = manifest.get("candidate_closure") or {}
        backend_binding = manifest.get("backend_runtime_closure") or {}
        candidate_identity = candidate_binding.get("identity_set_sha256")
        if not isinstance(candidate_identity, dict) or not candidate_identity:
            raise ValueError("formal candidate closure identity missing")
        return {
            "slice_name": f"manifest_{manifest_hash[:16]}",
            "dynamic_slice_spec": (
                release_dir.name,
                readiness_path.name,
                {},
            ),
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "release_id": release_id,
            "release_tooling_sha256": release_tooling_hash,
            "readiness_path": str(readiness_path),
            "readiness_sha256": actual_hash,
            "formal_runtime_bundle_sha256": str(bundle_binding.get("sha256") or ""),
            "formal_core_suite_sha256": str(core_binding.get("sha256") or ""),
            "formal_source_suite_sha256": declared_source_hash,
            "formal_public_evidence_sha256": str(
                evidence_binding.get("sha256") or ""
            ),
            "formal_public_evidence_binding_root_sha256": str(
                evidence_binding.get("binding_root_sha256") or ""
            ),
            "formal_candidate_closure_sha256": str(
                candidate_binding.get("sha256") or ""
            ),
            "formal_candidate_closure_identity_sha256": _canonical_json_sha256(
                candidate_identity
            ),
            "formal_backend_runtime_closure_sha256": str(
                backend_binding.get("sha256") or ""
            ),
            "core_release_pipeline_sha256": release_pipeline_hash,
            "backend_runtime_closure_identity_sha256": runtime_closure[
                "identity_sha256"
            ],
        }
    scenarios = readiness.get("scenarios") if isinstance(readiness, dict) else None
    if formal_run_contract.get("contract_version") != "agentic_persistent.v1":
        raise ValueError("formal_run_contract_version_unsupported")
    if not (
        isinstance(readiness, dict)
        and readiness.get("schema_version") == "1.0"
        and readiness.get("status") == "formal_evaluation_ready"
        and readiness.get("formal_evaluation_ready") is True
        and readiness.get("formal_run_blockers") == []
        and readiness.get("scoring_version") == SCORING_VERSION
        and readiness.get("primary_leaderboard_formula_version")
        == PRIMARY_LEADERBOARD_FORMULA_VERSION
        and readiness.get("primary_inference_version") == PRIMARY_INFERENCE_VERSION
        and isinstance(scenarios, list)
        and bool(scenarios)
        and readiness.get("n_scenarios") == len(scenarios)
        and isinstance(readiness.get("suite_manifest_sha256"), str)
        and len(readiness["suite_manifest_sha256"]) == 64
        and isinstance(formal_run_contract, dict)
        and formal_run_contract.get("required_construct_contract")
        == "operational_agency.v1"
        and all(
            isinstance(row, dict)
            and row.get("construct_contract") == "operational_agency.v1"
            for row in scenarios
        )
    ):
        raise ValueError("formal readiness is not current and green")

    live_tree = implementation_identity(repo_root)["implementation_tree_sha256"]
    if not (
        manifest.get("implementation_tree_sha256") == live_tree
        and readiness.get("implementation_tree_sha256") == live_tree
    ):
        raise ValueError("formal implementation tree mismatch")

    replay_binding = manifest.get("protocol21_replay")
    if not isinstance(replay_binding, dict):
        raise ValueError("formal source suite binding missing")
    raw_source_path = str(replay_binding.get("source_suite") or "")
    source_path = Path(raw_source_path)
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    source_path = source_path.resolve()
    declared_source_hash = str(replay_binding.get("source_suite_sha256") or "")
    if not (
        raw_source_path
        and source_path == manifest_path.parent / "protocol21_source_suite.json"
        and source_path.is_file()
        and declared_source_hash == hashlib.sha256(source_path.read_bytes()).hexdigest()
    ):
        raise ValueError("formal source suite binding mismatch")

    runtime_closure = validate_live_backend_runtime_closure(
        repo_root=repo_root,
        release_root=manifest_path.parent,
        release_id=str(manifest.get("release_id") or ""),
        source_suite_sha256=declared_source_hash,
        binding=manifest.get("backend_runtime_closure") or {},
    )

    pipeline_binding = manifest.get("pipeline_artifacts")
    if not isinstance(pipeline_binding, dict):
        raise ValueError("formal pipeline binding missing")
    raw_pipeline_path = str(pipeline_binding.get("path") or "")
    pipeline_path = Path(raw_pipeline_path)
    if not pipeline_path.is_absolute():
        pipeline_path = repo_root / pipeline_path
    if not raw_pipeline_path or pipeline_path.resolve() != runtime_root:
        raise ValueError("formal pipeline root mismatch")
    pipeline_manifest_path = runtime_root / "protocol2_v21_pipeline_manifest.json"
    declared_pipeline_hash = str(pipeline_binding.get("pipeline_manifest_sha256") or "")
    if not (
        pipeline_manifest_path.is_file()
        and declared_pipeline_hash
        == hashlib.sha256(pipeline_manifest_path.read_bytes()).hexdigest()
    ):
        raise ValueError("formal pipeline manifest hash mismatch")
    pipeline_manifest = json.loads(pipeline_manifest_path.read_text(encoding="utf-8"))
    release_pipeline_hash = str(manifest.get("core_release_pipeline_sha256") or "")
    if re.fullmatch(r"[0-9a-f]{64}", release_pipeline_hash) is None:
        raise ValueError("formal core release pipeline hash invalid")
    if not (
        pipeline_binding.get("core_release_pipeline_sha256") == release_pipeline_hash
        and replay_binding.get("core_release_pipeline_sha256") == release_pipeline_hash
        and pipeline_manifest.get("core_release_pipeline_sha256")
        == release_pipeline_hash
    ):
        raise ValueError("formal core release pipeline binding mismatch")
    pipeline_stages = pipeline_manifest.get("stages")
    stage_bindings = pipeline_binding.get("stage_artifacts")
    if not isinstance(pipeline_stages, list) or not isinstance(stage_bindings, dict):
        raise ValueError("formal pipeline stage bindings missing")
    stage_names = [
        str(row.get("name") or "") for row in pipeline_stages if isinstance(row, dict)
    ]
    stage_rows = {
        stage_name: row
        for stage_name, row in zip(stage_names, pipeline_stages, strict=True)
    }
    if (
        pipeline_manifest.get("status") != "formal_evaluation_ready"
        or pipeline_manifest.get("implementation_tree_sha256") != live_tree
        or pipeline_manifest.get("source_suite_sha256") != declared_source_hash
        or tuple(stage_names) != FORMAL_CORE_PIPELINE_STAGES
        or set(stage_bindings) != set(FORMAL_CORE_PIPELINE_STAGES)
    ):
        raise ValueError("formal pipeline stage set mismatch")
    bound_stage_paths: set[Path] = set()
    for stage_name in FORMAL_CORE_PIPELINE_STAGES:
        binding = stage_bindings[stage_name]
        row = stage_rows[stage_name]
        if not isinstance(binding, dict):
            raise ValueError("formal pipeline stage binding invalid")
        relative_path = Path(str(binding.get("relative_path") or ""))
        if (
            relative_path.as_posix() != FORMAL_CORE_STAGE_FILES[stage_name]
            or relative_path.is_absolute()
            or not relative_path.parts
        ):
            raise ValueError("formal pipeline stage path invalid")
        stage_path = (runtime_root / relative_path).resolve()
        try:
            stage_path.relative_to(runtime_root)
        except ValueError as exc:
            raise ValueError("formal pipeline stage path invalid") from exc
        if stage_path in bound_stage_paths:
            raise ValueError("formal pipeline stage path invalid")
        bound_stage_paths.add(stage_path)
        stage_hash = str(binding.get("sha256") or "")
        try:
            stage_artifact = json.loads(stage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"formal pipeline stage invalid:{stage_name}") from exc
        if not (
            stage_path.is_file()
            and stage_hash == hashlib.sha256(stage_path.read_bytes()).hexdigest()
            and row.get("output_sha256") == stage_hash
            and row.get("return_code") == 0
            and row.get("implementation_tree_sha256") == live_tree
            and row.get("core_release_pipeline_sha256") == release_pipeline_hash
            and isinstance(stage_artifact, dict)
            and stage_artifact.get("core_release_pipeline_sha256")
            == release_pipeline_hash
        ):
            raise ValueError(f"formal pipeline stage invalid:{stage_name}")
        if stage_name == "readiness" and (
            stage_path != readiness_path or stage_hash != actual_hash
        ):
            raise ValueError("formal pipeline readiness binding mismatch")

    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    release_id = readiness_relative.parts[0]
    release_tooling_hash = str(manifest.get("release_tooling_sha256") or "")
    if (
        manifest.get("release_id") != release_id
        or re.fullmatch(r"[0-9a-f]{64}", release_tooling_hash) is None
    ):
        raise ValueError("formal release identity binding mismatch")
    artifact_name = Path(*readiness_relative.parts[1:]).as_posix()
    return {
        "slice_name": f"manifest_{manifest_hash[:16]}",
        "dynamic_slice_spec": (release_id, artifact_name, {}),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "release_id": release_id,
        "release_tooling_sha256": release_tooling_hash,
        "readiness_path": str(readiness_path),
        "readiness_sha256": actual_hash,
        "core_release_pipeline_sha256": release_pipeline_hash,
        "backend_runtime_closure_identity_sha256": runtime_closure["identity_sha256"],
    }


_FORMAL_RUNTIME_BINDING_FIELDS = (
    ("formal_release_id", "release_id"),
    ("formal_manifest", "manifest_path"),
    ("formal_manifest_sha256", "manifest_sha256"),
    ("formal_release_tooling_sha256", "release_tooling_sha256"),
    ("formal_readiness_path", "readiness_path"),
    ("formal_readiness_sha256", "readiness_sha256"),
    (
        "formal_core_release_pipeline_sha256",
        "core_release_pipeline_sha256",
    ),
    (
        "formal_backend_runtime_closure_identity_sha256",
        "backend_runtime_closure_identity_sha256",
    ),
    ("formal_runtime_bundle_sha256", "formal_runtime_bundle_sha256"),
    ("formal_core_suite_sha256", "formal_core_suite_sha256"),
    ("formal_source_suite_sha256", "formal_source_suite_sha256"),
    ("formal_public_evidence_sha256", "formal_public_evidence_sha256"),
    (
        "formal_public_evidence_binding_root_sha256",
        "formal_public_evidence_binding_root_sha256",
    ),
    ("formal_candidate_closure_sha256", "formal_candidate_closure_sha256"),
    (
        "formal_candidate_closure_identity_sha256",
        "formal_candidate_closure_identity_sha256",
    ),
    (
        "formal_backend_runtime_closure_sha256",
        "formal_backend_runtime_closure_sha256",
    ),
)


def _formal_runtime_binding_metadata(
    binding: dict[str, Any] | None,
) -> dict[str, Any]:
    binding = binding or {}
    return {
        meta_field: binding.get(binding_field)
        for meta_field, binding_field in _FORMAL_RUNTIME_BINDING_FIELDS
    }


def _formal_runtime_binding_reasons(
    meta: dict[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    manifest_path = str(meta.get("formal_manifest") or "")
    if not manifest_path:
        return ["formal_runtime_manifest_missing"]
    try:
        live_binding = resolve_formal_manifest_slice(
            resolve_binding_path(manifest_path, repo_root=repo_root),
            repo_root=repo_root,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"formal_runtime_evidence_revalidation_failed:{type(exc).__name__}"]
    portable_live_binding = canonicalize_repo_owned_paths(
        live_binding, repo_root=repo_root
    )
    return sorted(
        f"formal_runtime_binding_changed:{meta_field}"
        for meta_field, binding_field in _FORMAL_RUNTIME_BINDING_FIELDS
        if str(meta.get(meta_field) or "")
        != str(portable_live_binding.get(binding_field) or "")
    )


PLOT_FILES = [
    "score_by_model.svg",
    "score_by_family_model.svg",
    "failures_by_model.svg",
    "tool_calls_vs_score.svg",
]

FALLBACK_WAIT_RATIO_THRESHOLD = 0.5


def _provider_failure_profile(*, formal_run: bool) -> dict[str, Any]:
    return {
        "max_consecutive_provider_failures": 1 if formal_run else 5,
        "provider_failure_policy": "abort" if formal_run else "compat_fallback",
    }


def _load_zhsrc_exports() -> dict[str, str]:
    """Parse export lines from ~/.zshrc without executing shell files."""
    out: dict[str, str] = {}
    pat = re.compile(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
    for rc in (Path.home() / ".zshrc",):
        if not rc.exists():
            continue
        for line in rc.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            m = pat.match(line)
            if not m:
                continue
            key, val = m.group(1), m.group(2).strip().strip('"').strip("'")
            if key.startswith("OPERATE_") or key == "OPENAI_API_KEY":
                out[key] = val
    for k, v in os.environ.items():
        if k.startswith("OPERATE_") or k == "OPENAI_API_KEY":
            out[k] = v
    return out


def _load_named_zshrc_export(name: str) -> str | None:
    """Read one explicitly requested export without executing shell startup files."""
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        return None
    pattern = re.compile(rf"^export\s+{re.escape(name)}=(.*)$")
    rc = Path.home() / ".zshrc"
    if not rc.exists():
        return None
    for line in rc.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1).strip().strip('"').strip("'")
    return None


def _llm_config_to_dict(cfg: LLMConfig) -> dict[str, Any]:
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "api_key_env": cfg.api_key_env,
        "base_url": cfg.base_url,
        "api_version": cfg.api_version,
        "api_version_env": cfg.api_version_env,
        "responses_base_url": cfg.responses_base_url,
        "responses_base_url_env": cfg.responses_base_url_env,
        "api_mode": cfg.api_mode,
        "stream_chat_completions": cfg.stream_chat_completions,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "model_context_window_tokens": cfg.model_context_window_tokens,
        "model_max_output_tokens": cfg.model_max_output_tokens,
        "token_count_method": cfg.token_count_method,
        "token_count_version": cfg.token_count_version,
        "timeout_s": cfg.timeout_s,
        "max_consecutive_provider_failures": cfg.max_consecutive_provider_failures,
        "provider_failure_policy": cfg.provider_failure_policy,
        "provider_rpm_limit": cfg.provider_rpm_limit or None,
        "provider_rpd_limit": cfg.provider_rpd_limit or None,
        "provider_rate_limit_scope": cfg.provider_rate_limit_scope,
        # v0.2.4: previously lost across worker-process round-trip; the loss
        # silently resurfaced ``prompt_mode="strict"`` regardless of the
        # configured value, and discarded operator-provided ``extra_headers``
        # (e.g. Azure ``Ocp-Apim-Subscription-Key``). Audit-trail and gateway
        # routing both broke without surface markers.
        "prompt_mode": cfg.prompt_mode,
        "interaction_mode": cfg.interaction_mode,
        "persistent_history_max_messages": cfg.persistent_history_max_messages,
        "persistent_context_max_chars": cfg.persistent_context_max_chars,
        "persistent_memory_max_items": cfg.persistent_memory_max_items,
        "tool_choice": cfg.tool_choice,
        "tool_choice_supported": cfg.tool_choice_supported,
        "reasoning_effort": cfg.reasoning_effort,
        "protocol_repair_max_tokens": cfg.protocol_repair_max_tokens,
        "allow_insecure_http": cfg.allow_insecure_http,
        "extra_headers": dict(cfg.extra_headers or {}),
    }


def _llm_config_from_dict(d: dict[str, Any]) -> LLMConfig:
    return LLMConfig(
        provider=d["provider"],
        model=d["model"],
        api_key_env=d.get("api_key_env", "OPENAI_API_KEY"),
        base_url=d.get("base_url"),
        api_version=d.get("api_version"),
        api_version_env=d.get("api_version_env", "OPERATE_API_VERSION"),
        responses_base_url=d.get("responses_base_url"),
        responses_base_url_env=d.get(
            "responses_base_url_env", "OPERATE_RESPONSES_API_BASE_URL"
        ),
        api_mode=d.get("api_mode", "auto"),
        stream_chat_completions=bool(d.get("stream_chat_completions", False)),
        temperature=float(d.get("temperature", 1.0)),
        max_tokens=int(d.get("max_tokens", 1200)),
        model_context_window_tokens=(
            int(d["model_context_window_tokens"])
            if d.get("model_context_window_tokens") is not None
            else None
        ),
        model_max_output_tokens=(
            int(d["model_max_output_tokens"])
            if d.get("model_max_output_tokens") is not None
            else None
        ),
        token_count_method=str(
            d.get("token_count_method", TOKEN_COUNT_METHOD_UTF8_BYTES)
        ),
        token_count_version=str(d.get("token_count_version", TOKEN_COUNT_VERSION_V1)),
        timeout_s=float(d.get("timeout_s", 60.0)),
        max_consecutive_provider_failures=int(
            d.get("max_consecutive_provider_failures", 5)
        ),
        provider_failure_policy=str(
            d.get("provider_failure_policy", "compat_fallback")
        ),
        provider_rpm_limit=int(d.get("provider_rpm_limit", 0) or 0),
        provider_rpd_limit=int(d.get("provider_rpd_limit", 0) or 0),
        provider_rate_limit_scope=(
            str(d["provider_rate_limit_scope"])
            if d.get("provider_rate_limit_scope") is not None
            else None
        ),
        prompt_mode=str(d.get("prompt_mode", "strict")),
        interaction_mode=str(d.get("interaction_mode", "logical_stateless")),
        persistent_history_max_messages=int(
            d.get("persistent_history_max_messages", 24)
        ),
        persistent_context_max_chars=int(d.get("persistent_context_max_chars", 16_000)),
        persistent_memory_max_items=int(d.get("persistent_memory_max_items", 32)),
        tool_choice=str(d.get("tool_choice", "auto")),
        tool_choice_supported=(
            bool(d["tool_choice_supported"])
            if d.get("tool_choice_supported") is not None
            else None
        ),
        reasoning_effort=(
            str(d["reasoning_effort"])
            if d.get("reasoning_effort") is not None
            else None
        ),
        protocol_repair_max_tokens=int(d.get("protocol_repair_max_tokens", 512)),
        allow_insecure_http=bool(d.get("allow_insecure_http", False)),
        extra_headers=dict(d.get("extra_headers", {}) or {}),
    )


def _safe_model_dir(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


# APFS/HFS NAME_MAX is 255. Quarantine appends ``.stale-YYYYMMDDTHHMMSSffffffZ``
# (~28 chars), so keep the live component well under the limit.
_FS_NAME_KEEP = 200
_CST8 = timezone(timedelta(hours=8))
_UNKNOWN_QUOTA_REPROBE_SECONDS = 300


class QuotaSentinelStateError(RuntimeError):
    """Fail closed when a persisted provider-quota sentinel is unreadable."""


def _quota_now_utc() -> datetime:
    return datetime.now(UTC)


def _quota_utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _fit_fs_component(path: Path, mapping: dict[str, str]) -> Path:
    """Hash a path's final component when it would exceed ``_FS_NAME_KEEP``."""
    name = path.name
    if len(name.encode("utf-8")) <= _FS_NAME_KEEP:
        return path
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:20]
    new_name = f"h{digest}{path.suffix}" if path.suffix else f"h{digest}"
    fitted = path.with_name(new_name)
    mapping[str(path)] = str(fitted)
    return fitted


def _quota_sentinel_path(out_dir: str | Path, model: str) -> Path:
    return Path(out_dir) / f".quota_exhausted_{_safe_model_dir(model)}"


def _parse_quota_reset_at(text: object) -> datetime | None:
    stamp = parse_tencent_quota_reset(text)
    raw = str(text or "").strip()
    if stamp is None and raw.endswith("UTC+8"):
        stamp = raw
    if stamp is None:
        match = re.search(
            r"(?:reset_at=)?(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))",
            raw,
        )
        stamp = match.group(1) if match else None
    if not stamp:
        return None
    if re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})",
        stamp,
    ):
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    try:
        naive = datetime.strptime(
            stamp.replace(" UTC+8", "").strip(), "%Y-%m-%d %H:%M:%S"
        )
    except ValueError:
        return None
    return naive.replace(tzinfo=_CST8)


def _quota_reset_text(text: object) -> str | None:
    tencent_stamp = parse_tencent_quota_reset(text)
    if tencent_stamp is not None:
        return tencent_stamp
    parsed = _parse_quota_reset_at(text)
    if parsed is None:
        return None
    return parsed.isoformat().replace("+00:00", "Z")


def _quota_sentinel_payload(path: Path) -> dict[str, Any] | None:
    import fcntl

    lock_path = path.with_name(f"{path.name}.lock")
    try:
        lock_handle = lock_path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise QuotaSentinelStateError(
            f"quota sentinel lock unavailable: {lock_path}"
        ) from exc
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            try:
                raw_payload = path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return None
            payload = json.loads(raw_payload)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise QuotaSentinelStateError(f"invalid quota sentinel: {path}") from exc
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            lock_handle.close()
            raise QuotaSentinelStateError(
                f"quota sentinel lock cleanup failed: {lock_path}"
            ) from exc
        lock_handle.close()
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") not in {None, "provider-quota-sentinel-v1"}
        or not isinstance(payload.get("model"), str)
        or not str(payload.get("model") or "").strip()
        or "reset_at" not in payload
    ):
        raise QuotaSentinelStateError(f"invalid quota sentinel: {path}")
    reset_at = payload.get("reset_at")
    if reset_at not in (None, "") and _parse_quota_reset_at(reset_at) is None:
        raise QuotaSentinelStateError(f"invalid quota sentinel reset_at: {path}")
    return payload


def _quota_sentinel_is_active(payload: dict[str, Any]) -> bool:
    reset_at = _parse_quota_reset_at(payload.get("reset_at"))
    if reset_at is None:
        return False
    return _quota_now_utc() < reset_at


def _active_quota_sentinel(job: dict[str, Any]) -> dict[str, Any] | None:
    out_dir = job.get("batch_output_dir")
    if not out_dir:
        return None
    payload = _quota_sentinel_payload(_quota_sentinel_path(out_dir, str(job["model"])))
    if payload is None:
        return None
    if not _quota_sentinel_is_active(payload):
        return None
    return payload


def _write_quota_sentinel(job: dict[str, Any], row: dict[str, Any]) -> Path | None:
    out_dir = job.get("batch_output_dir")
    if not out_dir:
        return None
    path = _quota_sentinel_path(out_dir, str(job["model"]))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise QuotaSentinelStateError(
            f"quota sentinel directory unavailable: {path.parent}"
        ) from exc
    now = _quota_now_utc()
    reset_at = row.get("quota_reset_at") or _quota_reset_text(row.get("error"))
    reset_source = "provider_signal"
    if reset_at is None:
        rpd_limit = int((job.get("llm_config") or {}).get("provider_rpd_limit") or 0)
        if rpd_limit > 0:
            reset_at = _quota_utc_text(
                (now + timedelta(days=1)).replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
            )
            reset_source = "configured_rpd_utc_midnight"
        else:
            reset_at = _quota_utc_text(
                now + timedelta(seconds=_UNKNOWN_QUOTA_REPROBE_SECONDS)
            )
            reset_source = "bounded_reprobe"
    payload = {
        "schema_version": "provider-quota-sentinel-v1",
        "model": job["model"],
        "reset_at": reset_at,
        "reset_source": reset_source,
        "written_at": _quota_utc_text(now),
        "scenario_slug": job.get("scenario_slug"),
    }
    import fcntl

    lock_path = path.with_name(f"{path.name}.lock")
    try:
        lock_handle = lock_path.open("a+", encoding="utf-8")
    except OSError as exc:
        raise QuotaSentinelStateError(
            f"quota sentinel lock unavailable: {lock_path}"
        ) from exc
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            _atomic_write_text(
                path,
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            )
        except OSError as exc:
            raise QuotaSentinelStateError(
                f"failed to persist quota sentinel: {path}"
            ) from exc
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            lock_handle.close()
            raise QuotaSentinelStateError(
                f"quota sentinel lock cleanup failed: {lock_path}"
            ) from exc
        lock_handle.close()
    return path


def _apply_llm_job_metadata(job: dict[str, Any], r: dict[str, Any]) -> dict[str, Any]:
    model = job["model"]
    r["model"] = model
    r["scenario_slug"] = job["scenario_slug"]
    if r.get("seed") is None and job.get("seed") is not None:
        r["seed"] = int(job["seed"])
    r["temperature"] = float(job.get("temperature", 0.0))
    r["evaluation_implementation_fingerprint"] = str(
        job.get("evaluation_implementation_fingerprint")
        or EVALUATION_IMPLEMENTATION_FINGERPRINT
    )
    r["run_semantics_fingerprint"] = str(
        job.get("run_semantics_fingerprint")
        or job.get("evaluation_implementation_fingerprint")
        or EVALUATION_IMPLEMENTATION_FINGERPRINT
    )
    r["agent_treatment_sha256"] = job.get("agent_treatment_sha256")
    r["agent_profile_sha256"] = job.get("agent_profile_sha256")
    r["interaction_mode"] = str(
        (job.get("llm_config") or {}).get("interaction_mode", "logical_stateless")
    )
    r["suite_scenario_signature"] = job.get(
        "suite_scenario_signature", job.get("scenario_signature")
    )
    r["suite_manifest_sha256"] = job.get("suite_manifest_sha256")
    r["suite_eligibility"] = job.get("suite_eligibility")
    r["suite_eligibility_sha256"] = job.get("suite_eligibility_sha256")
    r["seed_mode"] = str(job.get("seed_mode") or "fixed")
    if not r.get("scenario_signature"):
        r["scenario_signature"] = job.get("scenario_signature")
    r["domain"] = job.get("domain")
    r["backend_kind"] = job.get("backend_kind")
    r["source_denominator_key"] = job.get("source_denominator_key")
    r["case_ledger"] = job.get("case_ledger")
    evaluation_protocol = r.get("evaluation_protocol")
    if not isinstance(evaluation_protocol, dict):
        evaluation_protocol = {}
        r["evaluation_protocol"] = evaluation_protocol
    evaluation_protocol["construct_contract"] = job.get("construct_contract")
    r["pass_id"] = str(job.get("pass_id") or "pass-0")
    r["pass_index"] = int(job.get("pass_index", 0) or 0)
    r["pass_k"] = int(job.get("pass_k", 1) or 1)
    if r.get("status") == "ok":
        r["agent_name"] = f"llm_agent/{model}"
    return r


def _quota_parked_result(
    job: dict[str, Any], *, reset_at: str | None = None
) -> dict[str, Any]:
    message = "ProviderQuotaExhaustedError: parked after provider quota exhausted"
    if reset_at:
        message += f"; reset_at={reset_at}"
    r: dict[str, Any] = {
        "status": "error",
        "error": message,
        "quota_parked": True,
        "execution_started": False,
    }
    if reset_at:
        r["quota_reset_at"] = reset_at
    return _apply_llm_job_metadata(job, r)


def _scenario_signature_for_run(scenario: dict[str, Any], seed: int) -> str:
    sig = scenario.get("scenario_signature")
    if sig and int(scenario.get("seed", 42)) == int(seed):
        return str(sig)
    return _recompute_signature_with_seed(scenario, seed)


def _suite_manifest_sha256(
    scenarios: list[str],
    scenario_bodies: dict[str, dict[str, Any]],
) -> str:
    """Compatibility wrapper around the canonical suite identity helper."""
    return canonical_suite_manifest_sha256(scenarios, scenario_bodies)


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_BEHAVIOR_QUERY_VALUE_FIELDS = frozenset(
    {"api-version", "deployment", "model", "region", "route", "variant", "version"}
)
_BEHAVIOR_HEADER_VALUE_FIELDS = frozenset(
    {"x-deployment", "x-model", "x-region", "x-route", "x-variant"}
)


def _behavior_url_projection(value: str | None) -> dict[str, Any] | None:
    if not value:
        return None
    parsed = urlsplit(str(value))
    query = [
        (
            str(name),
            str(raw_value)
            if str(name).lower() in _BEHAVIOR_QUERY_VALUE_FIELDS
            else "[redacted]",
        )
        for name, raw_value in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    return {
        "scheme": parsed.scheme,
        "host": parsed.hostname,
        "port": parsed.port,
        "path": parsed.path,
        "query": sorted(query),
    }


def _private_provider_route_sha256(cfg: LLMConfig) -> str:
    """Bind behavioral routing without persisting or hashing secret values."""
    return _canonical_json_sha256(
        {
            "base_url": _behavior_url_projection(cfg.base_url),
            "responses_base_url": _behavior_url_projection(cfg.responses_base_url),
            "extra_headers": sorted(
                (
                    str(name),
                    str(value)
                    if str(name).lower() in _BEHAVIOR_HEADER_VALUE_FIELDS
                    else "[redacted]",
                )
                for name, value in (cfg.extra_headers or {}).items()
            ),
        }
    )


def _agent_treatment_identity(cfg: LLMConfig) -> dict[str, Any]:
    """Return the secret-free canonical profile whose hash identifies behavior."""
    return {
        "schema_version": "agent_treatment_v1",
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": public_provider_url(cfg.base_url),
        "api_version": cfg.api_version,
        "api_version_env": cfg.api_version_env,
        "responses_base_url": public_provider_url(cfg.responses_base_url),
        "responses_base_url_env": cfg.responses_base_url_env,
        "private_provider_route_sha256": (_private_provider_route_sha256(cfg)),
        "api_mode": cfg.api_mode,
        "stream_chat_completions": cfg.stream_chat_completions,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        **(
            {
                "model_context_window_tokens": (cfg.model_context_window_tokens),
                "model_max_output_tokens": cfg.model_max_output_tokens,
                "token_count_method": cfg.token_count_method,
                "token_count_version": cfg.token_count_version,
            }
            if cfg.model_context_window_tokens is not None
            else {}
        ),
        "timeout_s": cfg.timeout_s,
        "max_consecutive_provider_failures": cfg.max_consecutive_provider_failures,
        "provider_failure_policy": cfg.provider_failure_policy,
        "provider_rpm_limit": cfg.provider_rpm_limit or None,
        "provider_rpd_limit": cfg.provider_rpd_limit or None,
        "provider_rate_limit_scope": cfg.provider_rate_limit_scope,
        "prompt_mode": cfg.prompt_mode,
        "prompt_contract_sha256": prompt_contract_sha256(
            cfg.interaction_mode,
            cfg.prompt_mode,
        ),
        "interaction_mode": cfg.interaction_mode,
        "persistent_history_max_messages": cfg.persistent_history_max_messages,
        "persistent_context_max_chars": cfg.persistent_context_max_chars,
        "persistent_memory_max_items": cfg.persistent_memory_max_items,
        "tool_choice": cfg.tool_choice,
        "tool_choice_supported": cfg.tool_choice_supported,
        "reasoning_effort": cfg.reasoning_effort,
        "protocol_repair_max_tokens": cfg.protocol_repair_max_tokens,
        "allow_insecure_http": cfg.allow_insecure_http,
        "extra_header_names": sorted(str(key) for key in (cfg.extra_headers or {})),
        "harness": "direct_api",
        "prompt_context_compiler_binding": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        "tool_schema_binding": "decision_envelope.available_tool_schema_sha256",
        "wakeup_policy": CANONICAL_WAKEUP_POLICY,
    }


def _agent_treatment_sha256(cfg: LLMConfig) -> str:
    """Hash every public execution choice that can change agent behavior."""
    return _canonical_json_sha256(_agent_treatment_identity(cfg))


def _batch_llm_config(
    *,
    model: str,
    temperature: float,
    args: argparse.Namespace,
    base_url: str | None,
    api_version: str | None,
    responses_base_url: str | None,
) -> LLMConfig:
    """Build the single canonical provider/session treatment for a batch job."""
    interaction_mode = (
        getattr(args, "interaction_mode", "logical_stateless") or "logical_stateless"
    )
    persistent = interaction_mode == "logical_persistent"
    configured_max_tokens = getattr(args, "max_tokens", None)
    max_tokens = int(
        configured_max_tokens
        if configured_max_tokens is not None
        else (8192 if persistent else 4096)
    )
    frozen_capabilities = frozen_model_capabilities(model)
    model_context_window_tokens = getattr(args, "model_context_window_tokens", None)
    model_max_output_tokens = getattr(args, "model_max_output_tokens", None)
    persistent_history_max_messages = getattr(
        args, "persistent_history_max_messages", None
    )
    persistent_context_max_chars = getattr(args, "persistent_context_max_chars", None)
    persistent_memory_max_items = getattr(args, "persistent_memory_max_items", None)
    if (
        persistent
        and frozen_capabilities is not None
        and model_context_window_tokens is None
    ):
        model_context_window_tokens, model_max_output_tokens = frozen_capabilities
    return LLMConfig(
        provider=(
            "azure"
            if base_url and args.api_mode == "azure"
            else ("openai_compatible" if base_url else "openai")
        ),
        model=model,
        api_key_env=args.api_key_env,
        base_url=base_url,
        api_version=api_version,
        responses_base_url=responses_base_url,
        api_mode=args.api_mode,
        stream_chat_completions=bool(getattr(args, "stream_chat_completions", False)),
        temperature=temperature,
        max_tokens=max_tokens,
        model_context_window_tokens=model_context_window_tokens,
        model_max_output_tokens=model_max_output_tokens,
        timeout_s=float(
            getattr(args, "provider_timeout_s", None) or (150.0 if persistent else 60.0)
        ),
        **_provider_failure_profile(
            formal_run=bool(getattr(args, "formal_run", False))
        ),
        prompt_mode=getattr(args, "prompt_mode", "strict") or "strict",
        interaction_mode=interaction_mode,
        persistent_history_max_messages=int(
            persistent_history_max_messages
            if persistent_history_max_messages is not None
            else (32 if persistent else 24)
        ),
        persistent_context_max_chars=int(
            persistent_context_max_chars
            if persistent_context_max_chars is not None
            else (48_000 if persistent else 16_000)
        ),
        persistent_memory_max_items=int(
            persistent_memory_max_items
            if persistent_memory_max_items is not None
            else (64 if persistent else 32)
        ),
        # Use the cross-provider common denominator. Some formally evaluated
        # routes support tools but do not expose a `tool_choice` parameter.
        # The protocol validator still rejects non-tool decisions at epochs
        # that require an executable action.
        tool_choice="auto",
        tool_choice_supported=frozen_model_tool_choice_support(model),
        reasoning_effort=getattr(args, "reasoning_effort", None),
        protocol_repair_max_tokens=int(
            getattr(args, "protocol_repair_max_tokens", None)
            or (4096 if persistent else 512)
        ),
        provider_rpm_limit=int(getattr(args, "provider_rpm_limit", 0) or 0),
        provider_rpd_limit=int(getattr(args, "provider_rpd_limit", 0) or 0),
        provider_rate_limit_scope=(
            str(getattr(args, "provider_rate_limit_scope", "") or "").strip() or None
        ),
    )


def _model_capability_preflight_error(
    *,
    model: str,
    context_window: int,
    max_output: int,
    decision_reserve: int,
    repair_reserve: int,
) -> str | None:
    """Return a deterministic configuration error before scheduling work."""

    if max_output > context_window:
        return f"{model}: model maximum output cannot exceed the model context window"
    if decision_reserve > max_output:
        return f"{model}: decision output reserve exceeds model maximum output"
    if repair_reserve > max_output:
        return f"{model}: repair output reserve exceeds model maximum output"
    return None


def _batch_agent_treatment_hashes(
    *,
    models: list[str],
    temperature: float,
    args: argparse.Namespace,
    base_url: str | None,
    api_version: str | None,
    responses_base_url: str | None,
) -> dict[str, str]:
    return {
        model: _canonical_json_sha256(identity)
        for model, identity in _batch_agent_treatment_identities(
            models=models,
            temperature=temperature,
            args=args,
            base_url=base_url,
            api_version=api_version,
            responses_base_url=responses_base_url,
        ).items()
    }


def _batch_agent_treatment_identities(
    *,
    models: list[str],
    temperature: float,
    args: argparse.Namespace,
    base_url: str | None,
    api_version: str | None,
    responses_base_url: str | None,
) -> dict[str, dict[str, Any]]:
    return {
        model: _agent_treatment_identity(
            _batch_llm_config(
                model=model,
                temperature=temperature,
                args=args,
                base_url=base_url,
                api_version=api_version,
                responses_base_url=responses_base_url,
            )
        )
        for model in models
    }


def _formal_agent_treatment_hashes(
    agent_profile_sha256_by_model: dict[str, str],
    *,
    formal_manifest_binding: dict[str, Any],
    implementation_tree_sha256: str,
) -> dict[str, str]:
    """Bind a behavior-only agent profile to one formal release runtime."""

    if re.fullmatch(r"[0-9a-f]{64}", implementation_tree_sha256) is None:
        raise ValueError("formal implementation tree hash is invalid")
    required_hash_fields = {
        "manifest_sha256",
        "release_tooling_sha256",
        "readiness_sha256",
        "core_release_pipeline_sha256",
        "backend_runtime_closure_identity_sha256",
    }
    release_id = str(formal_manifest_binding.get("release_id") or "")
    if not release_id or Path(release_id).name != release_id:
        raise ValueError("formal release identity is invalid")
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(formal_manifest_binding.get(field) or ""))
        is None
        for field in required_hash_fields
    ):
        raise ValueError("formal runtime binding is incomplete")
    runtime_identity = {
        meta_field: formal_manifest_binding.get(binding_field)
        for meta_field, binding_field in _FORMAL_RUNTIME_BINDING_FIELDS
        if meta_field not in {"formal_manifest", "formal_readiness_path"}
        and formal_manifest_binding.get(binding_field) is not None
    }
    treatments: dict[str, str] = {}
    for model, profile_sha256 in agent_profile_sha256_by_model.items():
        if not model or re.fullmatch(r"[0-9a-f]{64}", profile_sha256) is None:
            raise ValueError("formal agent profile binding is invalid")
        treatments[model] = _canonical_json_sha256(
            {
                "schema_version": "formal_logical_treatment_v1",
                "interaction_mode": "logical_persistent",
                "agent_profile_sha256": profile_sha256,
                "formal_runtime_binding": runtime_identity,
                "implementation_tree_sha256": implementation_tree_sha256,
            }
        )
    if not treatments:
        raise ValueError("formal agent profile binding is missing")
    return treatments


def _resolve_logical_output_namespace(
    output_dir: Path,
    treatment_hashes: dict[str, str],
    *,
    formal_run: bool,
) -> Path:
    """Bind a formal single-model run to exactly one treatment leaf."""

    if not formal_run:
        return output_dir
    if len(treatment_hashes) != 1:
        raise ValueError("formal output namespace requires exactly one model treatment")
    digest = next(iter(treatment_hashes.values()))
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("formal output namespace treatment hash is invalid")
    component = f"treatment-{digest}"
    if output_dir.name.startswith("treatment-"):
        if output_dir.name != component:
            raise ValueError("formal output treatment namespace mismatch")
        return output_dir.resolve()
    return (output_dir / component).resolve()


def _formal_output_namespace_binding_error(
    meta: dict[str, Any], out_dir: Path
) -> str | None:
    if meta.get("formal_run") is not True:
        return None
    treatment_hashes = meta.get("agent_treatment_sha256_by_model")
    if not isinstance(treatment_hashes, dict) or len(treatment_hashes) != 1:
        return "formal_output_treatment_binding_invalid"
    digest = str(next(iter(treatment_hashes.values())))
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        return "formal_output_treatment_binding_invalid"
    if meta.get("output_namespace_treatment_sha256") != digest:
        return "formal_output_treatment_hash_mismatch"
    configured = str(meta.get("output_dir") or "")
    if (
        not configured
        or resolve_binding_path(configured, repo_root=REPO_ROOT) != out_dir.resolve()
        or out_dir.name != f"treatment-{digest}"
    ):
        return "formal_output_directory_binding_mismatch"
    return None


def _legacy_job_key(scenario_slug: str, model: str, seed: int) -> tuple[str, str, int]:
    return (scenario_slug, model, int(seed))


def _run_semantics_fingerprint(
    prompt_mode: str,
    max_tokens: int | None = None,
    interaction_mode: str = "logical_stateless",
) -> str:
    output_budget = "" if max_tokens is None else f":max-tokens-{int(max_tokens)}"
    return (
        f"{EVALUATION_IMPLEMENTATION_FINGERPRINT}:"
        f"prompt-{str(prompt_mode or 'strict').lower()}"
        f":interaction-{str(interaction_mode or 'logical_stateless').lower()}"
        f"{output_budget}"
    )


def _strong_job_key(
    scenario_slug: str,
    model: str,
    seed: int,
    scenario_signature: str,
    temperature: float,
    pass_id: str | None = None,
    implementation_fingerprint: str = EVALUATION_IMPLEMENTATION_FINGERPRINT,
    suite_manifest_sha256: str = "",
    suite_eligibility_sha256: str = "",
) -> tuple[str, str, int, str, str, str, str, str, str]:
    return (
        scenario_slug,
        model,
        int(seed),
        str(scenario_signature),
        f"{float(temperature):.6f}",
        str(pass_id or "pass-0"),
        str(implementation_fingerprint),
        str(suite_manifest_sha256),
        str(suite_eligibility_sha256),
    )


def _job_resume_keys(
    job: dict[str, Any],
) -> tuple[
    tuple[str, str, int],
    tuple[str, str, int, str, str, str, str, str, str],
]:
    return (
        _legacy_job_key(job["scenario_slug"], job["model"], int(job["seed"])),
        _strong_job_key(
            job["scenario_slug"],
            job["model"],
            int(job["seed"]),
            str(job.get("scenario_signature", "")),
            float(job.get("temperature", 1.0)),
            str(job.get("pass_id") or "pass-0"),
            str(
                job.get("run_semantics_fingerprint")
                or job.get("evaluation_implementation_fingerprint")
                or EVALUATION_IMPLEMENTATION_FINGERPRINT
            ),
            str(job.get("suite_manifest_sha256") or ""),
            str(job.get("suite_eligibility_sha256") or ""),
        ),
    )


def _row_resume_keys(
    row: dict[str, Any],
) -> tuple[
    tuple[str, str, int],
    tuple[str, str, int, str, str, str, str, str, str] | None,
]:
    slug = str(row.get("scenario_slug") or "")
    model = str(row.get("model") or row.get("agent_name", "")).replace("llm_agent/", "")
    seed = int(row.get("seed", -1))
    legacy = _legacy_job_key(slug, model, seed)
    sig = row.get("scenario_signature")
    temp = row.get("temperature")
    pass_id = str(row.get("pass_id") or "pass-0")
    protocol = row.get("evaluation_protocol") or {}
    implementation_fingerprint = (
        row.get("run_semantics_fingerprint")
        or row.get("evaluation_implementation_fingerprint")
        or protocol.get("implementation_fingerprint")
    )
    strong = None
    if (
        slug
        and model
        and seed >= 0
        and sig not in (None, "")
        and temp is not None
        and implementation_fingerprint
    ):
        strong = _strong_job_key(
            slug,
            model,
            seed,
            str(sig),
            float(temp),
            pass_id,
            str(implementation_fingerprint),
            str(row.get("suite_manifest_sha256") or ""),
            str(row.get("suite_eligibility_sha256") or ""),
        )
    return legacy, strong


def _row_implementation_fingerprint(row: dict[str, Any]) -> str:
    protocol = row.get("evaluation_protocol") or {}
    return str(
        row.get("evaluation_implementation_fingerprint")
        or protocol.get("implementation_fingerprint")
        or ""
    )


def _analysis_cell_key(row: dict[str, Any]) -> tuple[str, str, int, str, str]:
    slug = str(row.get("scenario_slug") or "")
    model = str(row.get("model") or row.get("agent_name", "")).replace("llm_agent/", "")
    seed = int(row.get("seed", -1))
    pass_id = str(row.get("pass_id") or "pass-0")
    treatment = str(row.get("agent_treatment_sha256") or "")
    return (slug, model, seed, pass_id, treatment)


def _prefer_current_implementation_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse fingerprint-bumped retries onto one analysis cell.

    Resume identity still includes the implementation fingerprint so an old-tree
    ok row does not skip a new-tree job. Analysis/finalize should not double-count
    both trees for the same (slug, model, seed, pass_id).
    """
    current = EVALUATION_IMPLEMENTATION_FINGERPRINT
    grouped: dict[tuple[str, str, int, str, str], list[dict[str, Any]]] = {}
    order: list[tuple[str, str, int, str, str]] = []
    for row in rows:
        key = _analysis_cell_key(row)
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(row)
    chosen: list[dict[str, Any]] = []
    for key in order:
        cell = grouped[key]
        matching = [
            row for row in cell if _row_implementation_fingerprint(row) == current
        ]
        chosen.append(matching[-1] if matching else cell[-1])
    return chosen


def _run_config_treatment_compatibility_reasons(
    existing: dict[str, Any], requested: dict[str, Any]
) -> list[str]:
    """Fail closed before an output directory can mix agent treatments."""

    for config in (existing, requested):
        binding_error = _treatment_hash_binding_error(config)
        if binding_error is not None:
            return [f"output_dir_agent_treatment_{binding_error}"]
    existing_hashes = existing.get("agent_treatment_sha256_by_model")
    requested_hashes = requested.get("agent_treatment_sha256_by_model")
    assert isinstance(existing_hashes, dict)
    assert isinstance(requested_hashes, dict)
    normalized_existing = {
        str(model): str(digest) for model, digest in existing_hashes.items()
    }
    normalized_requested = {
        str(model): str(digest) for model, digest in requested_hashes.items()
    }
    if normalized_existing != normalized_requested or str(
        existing.get("interaction_mode") or ""
    ) != str(requested.get("interaction_mode") or ""):
        return ["output_dir_agent_treatment_mismatch"]
    immutable_fields = (
        "scenario_slice",
        "patterns",
        "suite_manifest_sha256",
        "suite_eligibility_sha256",
        "formal_run",
        "implementation_tree_sha256",
        "models",
        "seeds",
        "seed_mode",
        "scenario_seed_pairs",
        "pass_k",
        "temperature",
        "max_tokens",
        "model_context_window_tokens_by_model",
        "model_max_output_tokens_by_model",
        "tool_choice_supported_by_model",
        "token_count_method",
        "token_count_version",
        "persistent_history_max_messages",
        "persistent_context_max_chars",
        "persistent_memory_max_items",
        "provider_rpm_limit",
        "provider_rpd_limit",
        "provider_rate_limit_scope",
        "max_consecutive_provider_failures",
        "provider_failure_policy",
        "output_dir",
        "output_namespace_treatment_sha256",
        "prompt_mode",
        "wakeup_policy",
        "evaluation_protocol_version",
        "evaluation_implementation_fingerprint",
        "run_semantics_fingerprint",
        "scoring_version",
        "scheduler_mode",
        "max_workers_requested",
        "max_workers_effective",
        "base_url",
        "api_version",
        "responses_base_url",
        "api_mode",
        "stream_chat_completions",
        "save_trajectories",
        "native_runtime_binding",
        "agent_profile_schema_version",
        "agent_profile_identity_by_model",
        "agent_profile_sha256_by_model",
        "agent_treatment_schema_version",
        *(meta_field for meta_field, _ in _FORMAL_RUNTIME_BINDING_FIELDS),
    )
    if any(
        existing.get(field) != requested.get(field)
        for field in immutable_fields
        if field in existing or field in requested
    ):
        return ["output_dir_immutable_run_scope_mismatch"]
    return []


def _treatment_hash_binding_error(meta: dict[str, Any]) -> str | None:
    expected = meta.get("agent_treatment_sha256_by_model")
    if not isinstance(expected, dict) or not expected:
        return "binding_missing"
    if any(
        not isinstance(model, str)
        or not model.strip()
        or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for model, digest in expected.items()
    ):
        return "binding_invalid"
    configured_models = meta.get("models")
    if not isinstance(configured_models, list) or not configured_models:
        return "model_scope_missing"
    normalized_models = [str(model).strip() for model in configured_models]
    if any(not model for model in normalized_models) or len(
        set(normalized_models)
    ) != len(normalized_models):
        return "model_scope_invalid"
    if set(expected) != set(normalized_models):
        return "model_scope_mismatch"
    return None


def _select_rows_for_treatment(
    rows: list[dict[str, Any]], meta: dict[str, Any]
) -> list[dict[str, Any]]:
    """Select only rows explicitly bound to the output directory treatment."""
    expected = meta.get("agent_treatment_sha256_by_model")
    if _treatment_hash_binding_error(meta) is not None:
        return []
    assert isinstance(expected, dict)
    expected_mode = str(meta.get("interaction_mode") or "")
    selected: list[dict[str, Any]] = []
    for row in rows:
        model = str(
            row.get("model")
            or str(row.get("agent_name") or "").replace("llm_agent/", "")
        )
        if (
            str(row.get("agent_treatment_sha256") or "")
            == str(expected.get(model) or "")
            and str(row.get("interaction_mode") or "") == expected_mode
        ):
            selected.append(row)
    return selected


def _formal_treatment_binding_reasons(
    meta: dict[str, Any], rows: list[dict[str, Any]]
) -> list[str]:
    """Validate formal row-level session treatment and batch homogeneity."""
    reasons: list[str] = []
    expected = meta.get("agent_treatment_sha256_by_model")
    binding_error = _treatment_hash_binding_error(meta)
    if binding_error is not None:
        return [f"formal_agent_treatment_{binding_error}"]
    assert isinstance(expected, dict)
    expected_mode = str(meta.get("interaction_mode") or "")
    if expected_mode not in {"logical_stateless", "logical_persistent"}:
        reasons.append("formal_interaction_mode_unsupported")
    seen_by_model: dict[str, set[str]] = {}
    for row in rows:
        model = str(
            row.get("model")
            or str(row.get("agent_name") or "").replace("llm_agent/", "")
        )
        row_mode = str(row.get("interaction_mode") or "")
        digest = str(row.get("agent_treatment_sha256") or "")
        seen_by_model.setdefault(model, set()).add(digest)
        if row_mode != expected_mode:
            reasons.append("formal_row_interaction_mode_mismatch")
        if not digest or digest != str(expected.get(model) or ""):
            reasons.append("formal_row_agent_treatment_mismatch")
    if any(len(digests) != 1 for digests in seen_by_model.values()):
        reasons.append("formal_agent_treatment_not_homogeneous")
    return list(dict.fromkeys(reasons))


def _episode_identity_key(row: dict[str, Any]) -> tuple[Any, ...]:
    legacy, strong = _row_resume_keys(row)
    if strong is not None:
        return ("strong",) + strong
    return ("legacy",) + legacy


def _episode_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    legacy, strong = _row_resume_keys(row)
    if strong is not None:
        return ("strong",) + strong
    return ("legacy",) + legacy


def _effective_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in _compact_episode_rows(rows)
            if str(row.get("status", "ok")) != "in_flight"
        ],
        key=_episode_sort_key,
    )


def effective_episode_rows_for_analysis(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compact append-only batch rows without collapsing legacy loose rows."""
    batch_rows: list[dict[str, Any]] = []
    loose_rows: list[dict[str, Any]] = []
    for row in rows:
        legacy, _ = _row_resume_keys(row)
        scenario_slug, model, seed = legacy
        if scenario_slug and model and seed >= 0:
            batch_rows.append(row)
        elif str(row.get("status", "ok")) != "in_flight":
            loose_rows.append(row)
    return (
        _prefer_current_implementation_rows(_effective_episode_rows(batch_rows))
        + loose_rows
    )


def _compact_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop transient ``in_flight`` placeholders and keep the latest terminal row.

    Episodes are append-only. After the two-phase runner hardening lands, a
    single logical cell may appear as:

      in_flight -> ok
      in_flight -> error
      error -> in_flight -> ok

    Finalization and derived reports should reason over the latest
    *terminal* row per logical cell, never over raw placeholders.
    """
    latest_any: dict[tuple[Any, ...], dict[str, Any]] = {}
    latest_terminal: dict[tuple[Any, ...], dict[str, Any]] = {}
    encounter_order: list[tuple[Any, ...]] = []
    for row in rows:
        key = _episode_identity_key(row)
        if key not in latest_any:
            encounter_order.append(key)
        latest_any[key] = row
        if str(row.get("status", "ok")) != "in_flight":
            latest_terminal[key] = row
    return [latest_terminal.get(key, latest_any[key]) for key in encounter_order]


def _strong_key_text_for_job(job: dict[str, Any]) -> str:
    _, strong = _job_resume_keys(job)
    if strong is not None:
        return "|".join(str(part) for part in strong)
    legacy, _ = _job_resume_keys(job)
    return "|".join(str(part) for part in legacy)


def _in_flight_placeholder_row(job: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": "in_flight",
        "scenario_slug": job["scenario_slug"],
        "model": job["model"],
        "agent_name": f"llm_agent/{job['model']}",
        "seed": int(job["seed"]),
        "pass_id": str(job.get("pass_id") or "pass-0"),
        "pass_index": int(job.get("pass_index", 0) or 0),
        "pass_k": int(job.get("pass_k", 1) or 1),
        "scenario_signature": job.get("scenario_signature"),
        "temperature": float(job.get("temperature", 1.0)),
        "evaluation_implementation_fingerprint": str(
            job.get("evaluation_implementation_fingerprint")
            or EVALUATION_IMPLEMENTATION_FINGERPRINT
        ),
        "run_semantics_fingerprint": str(
            job.get("run_semantics_fingerprint")
            or job.get("evaluation_implementation_fingerprint")
            or EVALUATION_IMPLEMENTATION_FINGERPRINT
        ),
        "strong_key": _strong_key_text_for_job(job),
        "started_at": datetime.now(UTC).isoformat(),
    }
    for key in ("suite_manifest_sha256", "suite_eligibility_sha256"):
        value = job.get(key)
        if value not in (None, ""):
            row[key] = value
    if job.get("episode_log_path"):
        row["episode_log_path"] = job["episode_log_path"]
    return row


def _quarantine_log_file(log_path: str | Path, *, tag: str) -> Path | None:
    path = Path(log_path)
    if not path.exists():
        return None
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.stem}.{tag}-{ts}{path.suffix}")
    suffix_idx = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.{tag}-{ts}.{suffix_idx}{path.suffix}")
        suffix_idx += 1
    path.rename(candidate)
    return candidate


def _retry_item_path(item: Any) -> str:
    raw = item.get("path") if isinstance(item, dict) else item
    if not raw:
        raise ValueError("retry-cells entry missing path")
    return str(raw)


def _retry_log_match_key(raw_path: str | Path) -> tuple[str, str]:
    path = Path(str(raw_path))
    model_dir = path.parent.name
    if not model_dir or not path.name:
        raise ValueError(f"unparseable retry log path: {raw_path}")
    return model_dir, path.name


def _apply_retry_cells_allowlist(
    jobs: list[dict[str, Any]],
    audit_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    sample = list(audit_payload.get("sample_orphan_interrupted_logs") or [])
    n_orphans = int(audit_payload.get("log_files_orphan_interrupted", len(sample)) or 0)
    if n_orphans > len(sample):
        raise ValueError(
            "sample_orphan_interrupted_logs is truncated; cannot safely retry all "
            "orphan cells from a partial sample"
        )
    wanted = {_retry_log_match_key(_retry_item_path(item)) for item in sample}
    selected = [
        job for job in jobs if _retry_log_match_key(job["episode_log_path"]) in wanted
    ]
    if len(selected) != len(wanted):
        missing = sorted(
            wanted - {_retry_log_match_key(j["episode_log_path"]) for j in selected}
        )
        raise ValueError(f"retry-cells did not resolve to jobs: {missing}")
    return selected


def _quarantine_retry_cell_logs(
    out_dir: Path,
    audit_payload: dict[str, Any],
    *,
    selected_jobs: list[dict[str, Any]] | None = None,
) -> list[str]:
    logs_root = (out_dir / "logs").resolve()
    canonical_by_key = {
        _retry_log_match_key(job["episode_log_path"]): Path(
            job["episode_log_path"]
        ).resolve()
        for job in selected_jobs or []
    }
    renamed: list[str] = []
    for item in audit_payload.get("sample_orphan_interrupted_logs") or []:
        raw = _retry_item_path(item)
        key = _retry_log_match_key(raw)
        if selected_jobs is not None:
            path = canonical_by_key.get(key)
            if path is None:
                raise ValueError(f"retry log is not a selected job: {raw}")
        else:
            candidate = Path(raw)
            path = (
                candidate.resolve()
                if candidate.is_absolute()
                else (out_dir / candidate).resolve()
            )
        try:
            path.relative_to(logs_root)
        except ValueError as exc:
            raise ValueError(f"retry log escapes output logs directory: {raw}") from exc
        dst = _quarantine_log_file(path, tag="orphan")
        if dst is not None:
            renamed.append(str(dst))
    return renamed


def _atomic_write_text(path: Path, payload: str) -> None:
    """Durably replace one metadata artifact without exposing partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _rewrite_episodes_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def _load_episodes_jsonl(
    path: Path, *, repair_trailing: bool = False
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    malformed_trailing = False
    for idx, line in enumerate(raw_lines):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Allow interrupted append-only runs to recover by ignoring
                # a truncated final line; interior corruption remains fatal.
                if idx == len(raw_lines) - 1:
                    LOGGER.warning("ignoring malformed trailing JSONL line in %s", path)
                    malformed_trailing = True
                    break
                raise
    if malformed_trailing and repair_trailing:
        _rewrite_episodes_jsonl(path, rows)
    return rows


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("failed to parse JSON object from %s", path)
        return None
    if not isinstance(payload, dict):
        LOGGER.warning(
            "expected JSON object in %s, got %s", path, type(payload).__name__
        )
        return None
    return payload


def _load_run_config_fail_closed(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"run config is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid existing run config: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"existing run config must be a JSON object: {path}")
    return payload


def _no_resume_output_conflict(
    *,
    resume: bool,
    finalize_only: bool,
    existing_run_config: dict[str, Any] | None,
    episodes_path: Path,
) -> bool:
    """Refuse a destructive restart over any initialized run namespace."""
    return not resume and not finalize_only and (
        existing_run_config is not None or episodes_path.exists()
    )


def _recover_patterns_from_existing_batch(out_dir: Path) -> list[str]:
    episodes_path = out_dir / "episodes.jsonl"
    rows = _load_episodes_jsonl(episodes_path)
    scenario_slugs = sorted(
        {
            str(r.get("scenario_slug")).strip()
            for r in rows
            if str(r.get("scenario_slug", "")).strip()
        }
    )
    if not scenario_slugs:
        return []
    # Exact slugs are intentional. Compressing rows into family globs can
    # silently expand a smoke or partial batch into unrelated scenarios when
    # finalize-only reconstructs coverage metadata.
    return scenario_slugs


def _infer_has_grid2op_from_patterns(patterns: list[str]) -> bool:
    for pattern in patterns:
        if pattern.startswith("power_grid/storm_emergency_6h/") or pattern.startswith(
            "power_grid/storm_emergency_6h_idf2023/"
        ):
            return True
        if "/storm_l2rpn_" in pattern or pattern.startswith("storm_l2rpn_"):
            return True
        if fnmatch.fnmatch(pattern, "power_grid/storm_emergency_6h*/*"):
            return True
    return False


def _config_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _native_runtime_binding(
    scenario_bodies: dict[str, dict[str, Any]],
    scenarios: list[str],
    *,
    max_workers: int,
    scheduler_mode: str,
) -> dict[str, Any]:
    """Resolve native runtime prerequisites before any provider request."""

    requires_real_sumo = any(
        str(scenario_bodies[slug].get("backend_kind") or "").strip().lower() == "sumo"
        for slug in scenarios
    )
    traffic_real_enabled = requires_real_sumo and (
        os.environ.get("OPERATE_TRAFFIC_BACKEND_REAL") == "1"
    )
    forced_transport = (
        str(os.environ.get("OPERATE_TRAFFIC_FORCE_TRANSPORT") or "").strip()
        if requires_real_sumo
        else ""
    )
    resolved_transport = (
        probe_sumo_transport() if requires_real_sumo and traffic_real_enabled else None
    )
    blockers: list[str] = []
    if requires_real_sumo and not traffic_real_enabled:
        blockers.append("real_sumo_gate_missing")
    elif requires_real_sumo and resolved_transport is None:
        blockers.append("sumo_transport_unavailable")
    if (
        requires_real_sumo
        and scheduler_mode == "global"
        and int(max_workers) > 1
        and resolved_transport == "libsumo"
    ):
        blockers.append("parallel_libsumo_unsupported")
    return {
        "ok": not blockers,
        "requires_real_sumo": requires_real_sumo,
        "traffic_real_enabled": traffic_real_enabled,
        "forced_transport": forced_transport or None,
        "resolved_transport": resolved_transport,
        "blockers": blockers,
    }


def _grid2op_env_names_requiring_local_cache(
    scenario_bodies: dict[str, dict[str, Any]], scenarios: list[str]
) -> list[str]:
    required: set[str] = set()
    for slug in scenarios:
        scenario = scenario_bodies[slug]
        if str(scenario.get("backend_kind", "")) != "grid2op":
            continue
        backend_config = scenario.get("backend_config") or {}
        if not isinstance(backend_config, dict):
            raise ValueError(f"{slug} grid2op scenario has invalid backend_config")
        if _config_truthy(backend_config.get("test", False)):
            continue
        env_name = str(backend_config.get("env_name") or "").strip()
        if not env_name:
            raise ValueError(f"{slug} grid2op scenario lacks backend_config.env_name")
        required.add(env_name)
    return sorted(required)


def _grid2op_local_cache_preflight(env_names: list[str]) -> dict[str, Any]:
    required_envs = sorted(
        dict.fromkeys(str(e).strip() for e in env_names if str(e).strip())
    )
    if not required_envs:
        return {
            "ok": True,
            "required_envs": [],
            "summary": {},
            "sources": [],
            "blockers": [],
        }

    try:
        from scripts.audit_grid2op_sources import audit_sources

        report = audit_sources(required_envs, load_local=True)
    except Exception as exc:
        return {
            "ok": False,
            "required_envs": required_envs,
            "summary": {"grid2op_cache_audit_failed": len(required_envs)},
            "sources": [],
            "blockers": [
                {
                    "env_name": env_name,
                    "status": "grid2op_cache_audit_failed",
                    "load_error": f"{type(exc).__name__}: {exc}",
                }
                for env_name in required_envs
            ],
        }

    sources = list(report.get("sources") or [])
    blockers: list[dict[str, Any]] = []
    seen_envs: set[str] = set()
    for row in sources:
        env_name = str(row.get("env_name") or "")
        if env_name:
            seen_envs.add(env_name)
        if row.get("status") == "local_loadable":
            continue
        blockers.append(
            {
                "env_name": env_name,
                "status": row.get("status"),
                "data_root": row.get("data_root"),
                "local_dir": row.get("local_dir"),
                "load_path": row.get("load_path"),
                "local_note": row.get("local_note"),
                "load_error": row.get("load_error"),
                "remote_download": row.get("remote_download"),
                "partial_archives": row.get("partial_archives"),
            }
        )
    for env_name in required_envs:
        if env_name not in seen_envs:
            blockers.append({"env_name": env_name, "status": "missing_audit_row"})

    return {
        **report,
        "ok": not blockers,
        "required_envs": required_envs,
        "blockers": blockers,
    }


def _format_grid2op_cache_blockers(preflight: dict[str, Any]) -> str:
    blockers = preflight.get("blockers") or []
    parts: list[str] = []
    for blocker in blockers:
        env_name = str(blocker.get("env_name") or "<unknown>")
        status = str(blocker.get("status") or "unknown")
        data_root = blocker.get("data_root")
        suffix = f" under {data_root}" if data_root else ""
        parts.append(f"{env_name}={status}{suffix}")
    return "; ".join(parts) if parts else "unknown Grid2Op cache blocker"


def _infer_scenario_slice_name(patterns: list[str]) -> str:
    for name in DYNAMIC_SCENARIO_SLICES:
        try:
            if list(patterns) == _release_suite_scenarios(name):
                return name
        except (OSError, ValueError, json.JSONDecodeError):
            # A not-yet-materialized dynamic release cannot match an
            # existing finalize-only batch.
            continue
    for name, builtins in SCENARIO_SLICES.items():
        if list(patterns) == list(builtins):
            return name
    return "custom"


def _release_suite_scenarios(slice_name: str) -> list[str]:
    try:
        release_id, artifact_name, filters = DYNAMIC_SCENARIO_SLICES[slice_name]
    except KeyError as exc:
        raise ValueError(f"unknown dynamic scenario slice: {slice_name}") from exc
    suite_path = REPO_ROOT / "release" / release_id / artifact_name
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    rows = suite.get("scenarios")
    if not isinstance(rows, list):
        raise ValueError(f"{suite_path} has no scenarios list")
    expected = int(suite.get("n_scenarios") or len(rows))
    if len(rows) != expected:
        raise ValueError(
            f"{suite_path} n_scenarios={expected} but contains {len(rows)} rows"
        )
    slugs: list[str] = []
    for row in rows:
        if not _dynamic_slice_row_matches(row, filters):
            continue
        path = str(row.get("path") or "")
        if not path:
            raise ValueError(f"{suite_path} row lacks path: {row}")
        if path.startswith("scenarios/"):
            path = path[len("scenarios/") :]
        if path.endswith(".yaml"):
            path = path[:-5]
        slugs.append(path)
    return slugs


def _suite_manifest_sha256_for_slice(
    slice_name: str,
    scenarios: list[str],
    scenario_bodies: dict[str, dict[str, Any]],
) -> str:
    """Bind dynamic runs to the row metadata used by readiness.

    Some release YAMLs omit optional ``seed`` or ``horizon_ticks`` fields and
    rely on their selected-suite row to supply them. Readiness hashes that
    canonical projection, so a formal batch must reproduce the same projection
    instead of silently hashing ``None`` from the YAML-only view.
    """
    if slice_name not in DYNAMIC_SCENARIO_SLICES:
        return _suite_manifest_sha256(scenarios, scenario_bodies)
    release_id, artifact_name, filters = DYNAMIC_SCENARIO_SLICES[slice_name]
    artifact_path = REPO_ROOT / "release" / release_id / artifact_name
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    rows = artifact.get("scenarios")
    if not isinstance(rows, list):
        raise ValueError(f"{artifact_path} has no scenarios list")
    bound_rows: dict[str, dict[str, Any]] = {}
    ordered_slugs: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or not _dynamic_slice_row_matches(row, filters):
            continue
        raw_path = str(row.get("path") or "")
        if not raw_path:
            raise ValueError(f"{artifact_path} row lacks path: {row}")
        slug = canonical_scenario_slug(raw_path)
        if slug in bound_rows:
            raise ValueError(f"{artifact_path} has duplicate scenario slug: {slug}")
        ordered_slugs.append(slug)
        bound_rows[slug] = row
    if len(scenarios) != len(set(scenarios)) or set(ordered_slugs) != set(scenarios):
        raise ValueError(
            f"{artifact_path} scenario identity does not match expanded run"
        )
    projected: dict[str, dict[str, Any]] = {}
    for slug in ordered_slugs:
        body = dict(scenario_bodies[slug])
        row = bound_rows[slug]
        for field in ("scenario_signature", "seed", "horizon_ticks"):
            if field not in body:
                body[field] = row.get(field)
        projected[slug] = body
    return canonical_suite_manifest_sha256(ordered_slugs, projected)


def _bind_scenario_contracts_for_slice(
    slice_name: str,
    scenarios: list[str],
    scenario_bodies: dict[str, dict[str, Any]],
) -> None:
    """Apply only contracts explicitly frozen in a dynamic readiness row."""
    if slice_name not in DYNAMIC_SCENARIO_SLICES:
        return
    release_id, artifact_name, filters = DYNAMIC_SCENARIO_SLICES[slice_name]
    artifact_path = REPO_ROOT / "release" / release_id / artifact_name
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    rows = artifact.get("scenarios")
    if not isinstance(rows, list):
        raise ValueError(f"{artifact_path} has no scenarios list")
    contracts: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not _dynamic_slice_row_matches(row, filters):
            continue
        slug = canonical_scenario_slug(str(row.get("path") or ""))
        contract = str(row.get("construct_contract") or "")
        source_key = str(row.get("source_denominator_key") or "")
        case_ledger = row.get("case_ledger")
        if not contract:
            continue
        if not source_key:
            raise ValueError(f"{artifact_path} source denominator key missing: {slug}")
        if not isinstance(case_ledger, dict) or not case_ledger:
            raise ValueError(f"{artifact_path} case ledger missing: {slug}")
        if str(case_ledger.get("source_denominator_key") or "") != source_key:
            raise ValueError(
                f"{artifact_path} case ledger source denominator mismatch: {slug}"
            )
        contracts[slug] = {
            "construct_contract": contract,
            "source_denominator_key": source_key,
            "case_ledger": case_ledger,
        }
    if set(contracts) != set(scenarios):
        missing = sorted(set(scenarios) - set(contracts))
        raise ValueError(
            f"{artifact_path} construct contract coverage mismatch: {missing}"
        )
    for slug in scenarios:
        binding = contracts[slug]
        contract = str(binding["construct_contract"])
        if contract != "operational_agency.v1":
            raise ValueError(f"{artifact_path} unsupported construct contract: {slug}")
        scenario_bodies[slug]["construct_contract"] = contract
        scenario_bodies[slug]["source_denominator_key"] = binding[
            "source_denominator_key"
        ]
        scenario_bodies[slug]["case_ledger"] = binding["case_ledger"]


def _suite_eligibility_binding(slice_name: str) -> dict[str, Any]:
    """Snapshot release eligibility into every protocol-2.x episode row."""
    if slice_name not in DYNAMIC_SCENARIO_SLICES:
        return {
            "suite_blocked": True,
            "reason": {"code": "unbound_custom_or_builtin_suite"},
            "diagnostic_cells": [],
            "uninformative_cells": [],
            "wait_dominant_cells": [],
        }
    release_id, artifact_name, _ = DYNAMIC_SCENARIO_SLICES[slice_name]
    artifact_path = REPO_ROOT / "release" / release_id / artifact_name
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("schema_version") == "operate-formal-runtime-bundle-v1":
        source_binding = artifact.get("source_suite") or {}
        source_name = str(source_binding.get("path") or "")
        source_path = artifact_path.parent / source_name
        source_hash = str(source_binding.get("sha256") or "")
        source_binding_valid = bool(
            source_name == "protocol21_source_suite.json"
            and source_path.is_file()
            and hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
        )
        formal_ready = bool(
            artifact.get("status") == "formal_evaluation_ready"
            and artifact.get("formal_evaluation_ready") is True
            and artifact.get("formal_run_blockers") == []
            and source_binding_valid
        )
        return {
            "suite_blocked": not formal_ready,
            "formal_evaluation_ready": formal_ready,
            "reason": (
                None
                if formal_ready
                else {
                    "code": "formal_runtime_bundle_invalid",
                    "blockers": (
                        [] if source_binding_valid else ["source_suite_hash_mismatch"]
                    ),
                }
            ),
            "diagnostic_cells": [],
            "uninformative_cells": [],
            "wait_dominant_cells": [],
            "source_artifact": f"release/{release_id}/{artifact_name}",
            "source_artifact_sha256": hashlib.sha256(
                artifact_path.read_bytes()
            ).hexdigest(),
            "readiness_source_artifact": (
                f"release/{release_id}/{source_name}"
            ),
            "readiness_source_artifact_sha256": source_hash,
            "readiness_source_binding_valid": source_binding_valid,
            "readiness_artifact_sha256": hashlib.sha256(
                artifact_path.read_bytes()
            ).hexdigest(),
            "suite_manifest_sha256": artifact.get("suite_manifest_sha256"),
            "scoring_version": artifact.get("scoring_version"),
            "primary_leaderboard_formula_version": artifact.get(
                "primary_leaderboard_formula_version"
            ),
            "primary_inference_version": artifact.get("primary_inference_version"),
            "task_completion_input_unit": artifact.get(
                "task_completion_input_unit"
            ),
            "task_completion_score_unit": artifact.get(
                "task_completion_score_unit"
            ),
            "weighted_equity_formula_version": artifact.get(
                "weighted_equity_formula_version"
            ),
            "formal_run_contract": artifact.get("formal_run_contract") or {},
        }
    if slice_name in PROTOCOL21_FORMAL_SLICES or slice_name.startswith("manifest_"):
        verification_reasons: list[str] = []
        regenerated: dict[str, Any] | None = None
        bindings = artifact.get("artifact_bindings") or {}
        required_artifacts = (
            "core",
            "source_suite",
            "preflight",
            "behavioral",
            "source_consumption",
            "task_contracts",
            "complexity",
            "observed_depth",
            "strategy_depth",
            "source_grounded",
            "agentic_contract",
            "release_coverage",
        )
        bound_paths: dict[str, Path] = {}
        for name in required_artifacts:
            raw_path = str((bindings.get(name) or {}).get("path") or "")
            path = Path(raw_path)
            if raw_path and not path.is_absolute():
                path = REPO_ROOT / path
            if not raw_path or not path.is_file():
                verification_reasons.append(f"readiness_bound_artifact_missing:{name}")
            else:
                bound_paths[name] = path
        if not verification_reasons:
            try:
                regenerated = build_readiness_from_paths(
                    core=bound_paths["core"],
                    source_suite=bound_paths["source_suite"],
                    preflight=bound_paths["preflight"],
                    behavioral=bound_paths["behavioral"],
                    source_consumption=bound_paths["source_consumption"],
                    task_contracts=bound_paths["task_contracts"],
                    complexity=bound_paths["complexity"],
                    observed_depth=bound_paths["observed_depth"],
                    strategy_depth=bound_paths["strategy_depth"],
                    source_grounded=bound_paths["source_grounded"],
                    agentic_contract=bound_paths["agentic_contract"],
                    release_coverage=bound_paths["release_coverage"],
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                verification_reasons.append(
                    f"readiness_reverification_failed:{type(exc).__name__}"
                )
        if regenerated is not None:
            if regenerated.get("formal_evaluation_ready") is not True:
                verification_reasons.extend(
                    regenerated.get("formal_run_blockers")
                    or ["readiness_reverification_not_green"]
                )
            for field in (
                "suite_manifest_sha256",
                "implementation_tree_sha256",
                "n_scenarios",
            ):
                if regenerated.get(field) != artifact.get(field):
                    verification_reasons.append(
                        f"readiness_reverification_mismatch:{field}"
                    )
            for field in (
                "artifact_bindings",
                "scenario_yaml_bindings",
                "source_file_bindings",
            ):
                if regenerated.get(field) != artifact.get(field):
                    verification_reasons.append(
                        f"readiness_reverification_mismatch:{field}"
                    )
        source_artifact = str(artifact.get("source_artifact") or "")
        source_path = Path(source_artifact)
        if source_artifact and not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        source_hash = str(artifact.get("source_artifact_sha256") or "")
        source_binding_valid = bool(
            source_hash
            and source_path.is_file()
            and hashlib.sha256(source_path.read_bytes()).hexdigest() == source_hash
        )
        formal_ready = bool(
            artifact.get("formal_evaluation_ready") and not verification_reasons
        )
        blockers = list(artifact.get("formal_run_blockers") or [])
        blockers.extend(verification_reasons)
        if not source_binding_valid:
            blockers.append("readiness_source_artifact_hash_mismatch")
        return {
            "suite_blocked": not (formal_ready and source_binding_valid),
            "formal_evaluation_ready": formal_ready,
            "reason": (
                None
                if formal_ready and source_binding_valid
                else {
                    "code": "protocol21_formal_evaluation_blocked",
                    "blockers": sorted(set(blockers)),
                }
            ),
            "diagnostic_cells": [],
            "uninformative_cells": [],
            "wait_dominant_cells": [],
            "source_artifact": f"release/{release_id}/{artifact_name}",
            "source_artifact_sha256": hashlib.sha256(
                artifact_path.read_bytes()
            ).hexdigest(),
            "readiness_source_artifact": source_artifact,
            "readiness_source_artifact_sha256": source_hash,
            "readiness_source_binding_valid": source_binding_valid,
            "readiness_artifact_sha256": hashlib.sha256(
                artifact_path.read_bytes()
            ).hexdigest(),
            "suite_manifest_sha256": artifact.get("suite_manifest_sha256"),
            "scoring_version": artifact.get("scoring_version"),
            "primary_leaderboard_formula_version": artifact.get(
                "primary_leaderboard_formula_version"
            ),
            "primary_inference_version": artifact.get("primary_inference_version"),
            "task_completion_input_unit": artifact.get("task_completion_input_unit"),
            "task_completion_score_unit": artifact.get("task_completion_score_unit"),
            "weighted_equity_formula_version": artifact.get(
                "weighted_equity_formula_version"
            ),
            "formal_run_contract": artifact.get("formal_run_contract") or {},
        }
    release_manifest: dict[str, Any] = {}
    manifest_path = artifact_path.parent / "manifest.json"
    if manifest_path != artifact_path and manifest_path.is_file():
        release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = (
        artifact.get("leaderboard_eligibility")
        or release_manifest.get("leaderboard_eligibility")
        or {}
    )
    blockers = list(
        artifact.get("leaderboard_blockers")
        or artifact.get("release_blockers")
        or artifact.get("public_release_blockers")
        or release_manifest.get("leaderboard_blockers")
        or release_manifest.get("release_blockers")
        or release_manifest.get("public_release_blockers")
        or []
    )
    return {
        "suite_blocked": not bool(artifact.get("leaderboard_eligible", False)),
        "reason": (
            None
            if bool(artifact.get("leaderboard_eligible", False))
            else {
                "code": "candidate_not_leaderboard_eligible",
                "blockers": blockers,
            }
        ),
        "diagnostic_cells": list(declared.get("diagnostic_cells") or []),
        "uninformative_cells": list(declared.get("uninformative_cells") or []),
        "wait_dominant_cells": list(declared.get("wait_dominant_cells") or []),
        "source_artifact": f"release/{release_id}/{artifact_name}",
        "source_artifact_sha256": hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest(),
    }


def _validate_protocol21_formal_run(
    config: dict[str, Any],
    readiness: dict[str, Any],
    *,
    suite_manifest_sha256: str | None,
    scenario_bodies: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Return stable fail-closed reasons for a formal Protocol-2.1 run."""
    reasons: list[str] = []
    if (
        config.get("scenario_slice") not in PROTOCOL21_FORMAL_SLICES
        and config.get("formal_manifest_bound") is not True
    ):
        reasons.append("formal_slice_must_be_protocol21_core")
    models = [
        str(model).strip() for model in config.get("models") or [] if str(model).strip()
    ]
    formal_contract = readiness.get("formal_run_contract") or {}
    persistent_contract = (
        formal_contract.get("contract_version") == "agentic_persistent.v1"
    )
    if not formal_contract:
        reasons.append("formal_run_contract_missing")
    elif not persistent_contract:
        reasons.append("formal_run_contract_version_unsupported")
    if persistent_contract:
        if formal_contract.get("wakeup_policy") != CANONICAL_WAKEUP_POLICY:
            reasons.append("formal_wakeup_policy_mismatch")
        required_model_count = int(
            formal_contract.get("required_model_count_per_shard") or 1
        )
        if len(models) != required_model_count or len(set(models)) != len(models):
            reasons.append("formal_model_count_per_shard_must_equal_one")
        raw_pass_k = config.get("pass_k")
        if (
            isinstance(raw_pass_k, bool)
            or not isinstance(raw_pass_k, int)
            or raw_pass_k < int(formal_contract.get("minimum_pass_k") or 1)
        ):
            reasons.append("formal_pass_k_below_minimum")
        raw_max_workers = config.get("max_workers")
        if (
            isinstance(raw_max_workers, bool)
            or not isinstance(raw_max_workers, int)
            or not (
                int(formal_contract.get("minimum_max_workers") or 1)
                <= raw_max_workers
                <= int(formal_contract.get("maximum_max_workers") or 32)
            )
        ):
            reasons.append("formal_max_workers_out_of_range")
        required_temperature = float(formal_contract.get("required_temperature", 0.0))
        temperature = config.get("temperature")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) != required_temperature
        ):
            reasons.append("formal_temperature_must_equal_zero")
        required_mode = str(
            formal_contract.get("required_interaction_mode", "logical_persistent")
        )
        if config.get("interaction_mode") != required_mode:
            reasons.append("formal_interaction_mode_must_be_logical_persistent")
        if formal_contract.get("requires_explicit_model_capabilities") is True:
            context_caps = config.get("model_context_window_tokens_by_model")
            output_caps = config.get("model_max_output_tokens_by_model")
            scalar_context = config.get("model_context_window_tokens")
            scalar_output = config.get("model_max_output_tokens")
            explicit_scalar = (
                isinstance(scalar_context, int)
                and not isinstance(scalar_context, bool)
                and scalar_context > 0
                and isinstance(scalar_output, int)
                and not isinstance(scalar_output, bool)
                and scalar_output > 0
            )
            explicit_maps = (
                isinstance(context_caps, dict)
                and isinstance(output_caps, dict)
                and set(context_caps) == set(models)
                and set(output_caps) == set(models)
                and all(
                    isinstance(context_caps[model], int)
                    and not isinstance(context_caps[model], bool)
                    and context_caps[model] > 0
                    and isinstance(output_caps[model], int)
                    and not isinstance(output_caps[model], bool)
                    and output_caps[model] > 0
                    for model in models
                )
            )
            if not (explicit_scalar or explicit_maps):
                reasons.append("formal_model_capabilities_must_be_explicit")
        agentic_profile = formal_contract.get("agentic_profile") or {}
        for field, expected in agentic_profile.items():
            if config.get(field) != expected:
                reasons.append(f"formal_agentic_profile_{field}_mismatch")
    if config.get("prompt_mode") != "strict":
        reasons.append("formal_prompt_mode_must_be_strict")
    if config.get("seed_mode") != "scenario":
        reasons.append("formal_seed_mode_must_be_scenario")
    if config.get("scheduler_mode") != "global":
        reasons.append("formal_scheduler_must_be_global")
    if config.get("save_trajectories") is not True:
        reasons.append("formal_trajectories_required")
    if config.get("finalize") is not True:
        reasons.append("formal_finalization_required")
    if config.get("allow_blocked_suite") is not False:
        reasons.append("formal_cannot_allow_blocked_suite")
    if config.get("diagnostic_only") is not False:
        reasons.append("formal_diagnostic_only_forbidden")
    if config.get("git_metadata_available") is not True:
        reasons.append("formal_git_metadata_unavailable")
    if config.get("git_dirty") is not False:
        reasons.append("formal_git_tree_must_be_clean")
    rpm_limit = config.get("provider_rpm_limit")
    if (
        rpm_limit is not None
        and (isinstance(rpm_limit, bool) or not isinstance(rpm_limit, int) or rpm_limit <= 0)
    ):
        reasons.append("formal_provider_rpm_limit_invalid")
    rpd_limit = config.get("provider_rpd_limit")
    if (
        rpd_limit is not None
        and (isinstance(rpd_limit, bool) or not isinstance(rpd_limit, int) or rpd_limit <= 0)
    ):
        reasons.append("formal_provider_rpd_limit_invalid")
    quota_limit_enabled = any(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (rpm_limit, rpd_limit)
    )
    if quota_limit_enabled and not str(
        config.get("provider_rate_limit_scope") or ""
    ).strip():
        reasons.append("formal_provider_rate_limit_scope_required")
    if readiness.get("formal_evaluation_ready") is not True:
        reasons.append("formal_readiness_not_green")
    scoring_version = readiness.get("scoring_version")
    if scoring_version != SCORING_VERSION:
        reasons.append("formal_scoring_version_mismatch")
    formula_version = readiness.get("primary_leaderboard_formula_version")
    if formula_version in (None, ""):
        reasons.append("formal_primary_leaderboard_formula_missing")
    elif formula_version != PRIMARY_LEADERBOARD_FORMULA_VERSION:
        reasons.append("formal_primary_leaderboard_formula_mismatch")
    if readiness.get("primary_inference_version") != PRIMARY_INFERENCE_VERSION:
        reasons.append("formal_primary_inference_version_mismatch")
    unit_contracts = (
        (
            "task_completion_input_unit",
            TASK_COMPLETION_INPUT_UNIT,
            "formal_task_completion_input_unit_mismatch",
        ),
        (
            "task_completion_score_unit",
            TASK_COMPLETION_SCORE_UNIT,
            "formal_task_completion_score_unit_mismatch",
        ),
        (
            "weighted_equity_formula_version",
            WEIGHTED_EQUITY_FORMULA_VERSION,
            "formal_weighted_equity_formula_mismatch",
        ),
    )
    for field, expected, reason in unit_contracts:
        if readiness.get(field) != expected:
            reasons.append(reason)
    if readiness.get("readiness_source_binding_valid") is False:
        reasons.append("formal_readiness_source_hash_mismatch")
    expected_manifest = readiness.get("suite_manifest_sha256")
    if suite_manifest_sha256 is not None and expected_manifest != suite_manifest_sha256:
        reasons.append("formal_suite_manifest_mismatch")
    if scenario_bodies is not None:
        for slug, body in sorted(scenario_bodies.items()):
            construct_contract = body.get("construct_contract")
            if construct_contract in (None, ""):
                reasons.append(f"formal_scenario_construct_contract_missing:{slug}")
            elif construct_contract != "operational_agency.v1":
                reasons.append(f"formal_scenario_construct_contract_mismatch:{slug}")
    return sorted(set(reasons))


def _dynamic_slice_row_matches(
    row: dict[str, Any],
    filters: dict[str, set[str]],
) -> bool:
    for key, allowed_values in filters.items():
        value: Any = row
        for part in key.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if str(value or "") not in allowed_values:
            return False
    return True


def _recover_execution_hints_from_batch_log(out_dir: Path) -> dict[str, Any]:
    log_path = out_dir / "batch_run.log"
    if not log_path.is_file():
        return {}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}

    scheduled_legacy_re = re.compile(
        r"Scheduled\s+(?P<episodes>\d+)\s+episodes\s+\("
        r"(?P<scenarios>\d+)\s+scenarios\s+×\s+"
        r"(?P<models>\d+)\s+models\s+×\s+"
        r"(?P<seeds>\d+)\s+seeds\);\s+max_workers=(?P<workers>\d+)"
    )
    scheduled_seed_mode_re = re.compile(
        r"Scheduled\s+(?P<episodes>\d+)\s+episodes\s+\("
        r"(?P<scenarios>\d+)\s+scenarios\s+×\s+"
        r"(?P<models>\d+)\s+models\s+×\s+"
        r"(?P<seed_mode>fixed|scenario)\s+seed mode\s+×\s+"
        r"pass_k=(?P<pass_k>\d+)\);\s+max_workers=(?P<workers>\d+)"
    )
    per_model_re = re.compile(
        r"per-model scheduler:\s+(?P<lanes>\d+)\s+lanes across\s+"
        r"(?P<models>\d+)\s+models;\s+lane_workers=(?P<workers>\d+)"
    )

    hints: dict[str, Any] = {}
    for line in lines:
        m = scheduled_seed_mode_re.search(line)
        seed_mode_format = m is not None
        if m is None:
            m = scheduled_legacy_re.search(line)
        if not m:
            continue
        episodes = int(m.group("episodes"))
        if episodes <= 0:
            continue
        hints["scheduled_episodes"] = episodes
        hints["n_scenarios"] = int(m.group("scenarios"))
        hints["n_models"] = int(m.group("models"))
        if seed_mode_format:
            hints["seed_mode"] = str(m.group("seed_mode"))
            hints["pass_k"] = int(m.group("pass_k"))
        else:
            hints["n_seeds"] = int(m.group("seeds"))
        hints["max_workers_effective"] = int(m.group("workers"))
        break

    for line in lines:
        m = per_model_re.search(line)
        if not m:
            continue
        hints["scheduler_mode"] = "per_model"
        hints["max_workers_effective"] = int(m.group("workers"))
        break

    if "scheduler_mode" not in hints:
        if any("allowing concurrent model lanes" in line for line in lines):
            hints["scheduler_mode"] = "per_model"
        elif any("forcing --max-workers=1" in line for line in lines):
            hints["scheduler_mode"] = "global"

    if any("grid2op-backed scenario detected" in line for line in lines):
        hints["has_grid2op"] = True

    return hints


def _recover_models_for_finalize(
    *,
    configured_models: list[str],
    rows: list[dict[str, Any]],
    scheduled_model_count: int | None,
) -> list[str]:
    observed_models = sorted(
        {
            str(row.get("model")).strip()
            for row in rows
            if str(row.get("model", "")).strip()
        }
    )
    if not observed_models:
        return configured_models
    if scheduled_model_count is not None:
        if scheduled_model_count == len(observed_models):
            return observed_models
        return configured_models or observed_models
    return configured_models or observed_models


def _completed_ok_resume_index(
    rows: list[dict[str, Any]],
    *, batch_root: Path | None = None,
) -> tuple[
    set[tuple[str, str, int]],
    set[tuple[str, str, int, str, str, str, str, str, str]],
]:
    legacy_done: set[tuple[str, str, int]] = set()
    strong_done: set[tuple[str, str, int, str, str, str, str, str, str]] = set()
    for r in rows:
        if r.get("status") != "ok":
            continue
        if not _row_is_clean_for_resume(r, batch_root=batch_root):
            continue
        legacy, strong = _row_resume_keys(r)
        if not legacy[0] or not legacy[1] or legacy[2] < 0:
            continue
        if strong is None:
            legacy_done.add(legacy)
        else:
            strong_done.add(strong)
    return legacy_done, strong_done


def _semantic_coverage_is_formally_valid(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("covered") is not True:
        return False
    unknown = value.get("unknown_tool_names")
    unclassified = value.get("unclassified_tool_names")
    missing_explicit = value.get("missing_explicit_semantic_role_names")
    missing_targets = value.get("missing_native_target_kind_names")
    missing_actuators = value.get("missing_actuator_family_names")
    return (
        isinstance(unknown, list)
        and isinstance(unclassified, list)
        and isinstance(missing_explicit, list)
        and isinstance(missing_targets, list)
        and isinstance(missing_actuators, list)
        and not unknown
        and not unclassified
        and not missing_explicit
        and not missing_targets
        and not missing_actuators
        and value.get("explicit_semantic_roles_complete") is True
        and value.get("native_targets_complete") is True
        and value.get("state_changing_actuators_complete") is True
    )


def _operational_agency_profile_is_consistent(
    trajectory_summary: dict[str, Any],
    *,
    counterfactual: dict[str, Any] | None = None,
) -> bool:
    return operational_agency_profile_is_consistent(
        trajectory_summary,
        counterfactual=counterfactual,
    )


def _agency_attribution_is_formally_complete(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("per_action_capped") is not False:
        return False
    for prefix in ("per_action", "per_action_group"):
        status = value.get(f"{prefix}_status")
        expected = value.get(f"{prefix}_expected")
        attempted = value.get(f"{prefix}_attempted")
        completed = value.get(f"{prefix}_completed")
        failures = value.get(f"{prefix}_failures")
        if (
            status != "complete"
            or isinstance(expected, bool)
            or not isinstance(expected, int)
            or isinstance(attempted, bool)
            or not isinstance(attempted, int)
            or isinstance(completed, bool)
            or not isinstance(completed, int)
            or not 0 <= completed <= attempted <= expected
            or completed != expected
            or not isinstance(failures, list)
            or failures
        ):
            return False
    return True


def _failed_tick_reason(item: Any) -> str:
    if not isinstance(item, dict):
        return "provider_other_error"
    reason = str(item.get("reason") or "").strip()
    if reason:
        return reason
    message = str(item.get("exc_msg_head") or "")
    if "max_chars" in message and "action-critical" in message:
        return "prompt_budget_exceeded"
    return "provider_other_error"


def _llm_call_failure_eligibility_reasons(llm: dict[str, Any]) -> list[str]:
    """Classify LLM call failures without treating local prompt overflow as an API outage."""
    failure_counters = (
        "llm_calls_failed",
        "provider_tool_call_failures",
        "fallback_without_tools_count",
        "provider_model_identity_failed_request_count",
    )
    if not any(int(llm.get(field, 0) or 0) > 0 for field in failure_counters):
        return []
    kinds = {_failed_tick_reason(item) for item in (llm.get("failed_tick_log") or [])}
    if not kinds:
        return ["provider_call_failure"]
    reasons: list[str] = []
    if any(kind.startswith("provider_") for kind in kinds):
        reasons.append("provider_call_failure")
    if "prompt_budget_exceeded" in kinds:
        reasons.append("prompt_budget_exceeded")
    return reasons or ["provider_call_failure"]


def _formal_source_contract_reasons(row: dict[str, Any]) -> list[str]:
    source_key = str(row.get("source_denominator_key") or "")
    case_ledger = row.get("case_ledger")
    reasons: list[str] = []
    if not source_key:
        reasons.append("source_denominator_key_missing")
    if not isinstance(case_ledger, dict) or not case_ledger:
        reasons.append("case_ledger_missing_or_invalid")
    elif str(case_ledger.get("source_denominator_key") or "") != source_key:
        reasons.append("case_ledger_source_denominator_mismatch")
    return reasons


def _causal_response_contract_reasons(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    records = summary.get("event_response_records")
    if not isinstance(records, list):
        return ["event_response_records_missing_or_invalid"]
    for record in records:
        if not isinstance(record, dict) or record.get("response_status") != "causal":
            continue
        observed_tick = record.get("first_observed_tick")
        if (
            isinstance(observed_tick, bool)
            or not isinstance(observed_tick, int)
            or observed_tick < 0
        ):
            reasons.append("causal_response_first_observed_missing")
        dependencies = record.get("action_consumes_evidence_ids")
        if (
            not isinstance(dependencies, list)
            or not dependencies
            or any(not isinstance(item, str) or not item for item in dependencies)
        ):
            reasons.append("causal_response_evidence_dependency_missing")
    return reasons


def _trajectory_sidecar_eligibility_reasons(
    row: dict[str, Any],
    *,
    summary_key: str,
    stem: str,
    schema_version: str,
    require_nonempty: bool,
    path_stem: str | None = None,
    require_byte_count: bool = False,
    batch_root: Path | None = None,
) -> list[str]:
    """Verify a formal row against the exact treatment-bound sidecar bytes."""

    summary = row.get("trajectory_summary") or {}
    artifact = summary.get(summary_key)
    reason_prefix = f"{stem}_artifact"
    if not isinstance(artifact, dict):
        return [f"{reason_prefix}_missing"]
    if artifact.get("schema_version") != schema_version:
        return [f"{reason_prefix}_invalid"]
    expected_hash = str(artifact.get("sha256") or "")
    expected_count = artifact.get("event_count")
    expected_bytes = artifact.get("byte_count")
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        or isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < (1 if require_nonempty else 0)
        or (
            require_byte_count
            and (
                isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, int)
                or expected_bytes < 0
            )
        )
    ):
        return [f"{reason_prefix}_invalid"]

    trajectory_prefix = summary.get("trajectory_path")
    artifact_path = artifact.get("path")
    if not trajectory_prefix or not artifact_path:
        return [f"{reason_prefix}_path_missing"]
    root = batch_root.resolve() if batch_root is not None else None

    expected_path = resolve_batch_path(
        f"{trajectory_prefix}.{path_stem or stem}.jsonl", batch_root=root
    )
    declared_path = resolve_batch_path(artifact_path, batch_root=root)
    if root is not None:
        try:
            expected_path.relative_to(root)
            declared_path.relative_to(root)
        except ValueError:
            return [f"{reason_prefix}_path_outside_batch"]
    if declared_path != expected_path:
        return [f"{reason_prefix}_path_mismatch"]
    treatment_hash = str(row.get("agent_treatment_sha256") or "")
    if treatment_hash and f"treatment-{treatment_hash}" not in expected_path.parts:
        return [f"{reason_prefix}_treatment_path_mismatch"]
    try:
        payload = expected_path.read_bytes()
    except OSError:
        return [f"{reason_prefix}_unreadable"]
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        return [f"{reason_prefix}_sha256_mismatch"]
    if require_byte_count and len(payload) != expected_bytes:
        return [f"{reason_prefix}_byte_count_mismatch"]
    if len(payload.splitlines()) != expected_count:
        return [f"{reason_prefix}_event_count_mismatch"]
    return []


def _formal_primary_artifact_reasons(
    row: dict[str, Any], *, verify_artifact_bytes: bool, batch_root: Path | None = None
) -> list[str]:
    summary = row.get("trajectory_summary") or {}
    reasons: list[str] = []
    artifacts = (
        (
            "trajectory_artifact",
            "trajectory",
            "trajectory",
            "episode_trajectory_jsonl_v1",
        ),
        (
            "evidence_ledger_artifact",
            "evidence_ledger",
            "evidence",
            "evidence_ledger_jsonl_v1",
        ),
    )
    for summary_key, reason_stem, path_stem, schema_version in artifacts:
        artifact = summary.get(summary_key)
        if (
            not isinstance(artifact, dict)
            or artifact.get("schema_version") != schema_version
            or re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256") or "")) is None
            or not str(artifact.get("path") or "")
            or isinstance(artifact.get("event_count"), bool)
            or not isinstance(artifact.get("event_count"), int)
            or int(artifact.get("event_count") or 0) <= 0
            or isinstance(artifact.get("byte_count"), bool)
            or not isinstance(artifact.get("byte_count"), int)
            or int(artifact.get("byte_count") or 0) <= 0
        ):
            reasons.append(f"{reason_stem}_artifact_missing_or_invalid")
            continue
        if verify_artifact_bytes:
            reasons.extend(
                _trajectory_sidecar_eligibility_reasons(
                    row,
                    summary_key=summary_key,
                    stem=reason_stem,
                    path_stem=path_stem,
                    schema_version=schema_version,
                    require_nonempty=True,
                    require_byte_count=True,
                    batch_root=batch_root,
                )
            )
    return reasons


_PERSISTENT_MEMORY_LIST_FIELDS = (
    "unresolved_alarms",
    "open_obligations",
    "confirmed_facts",
    "active_commitments",
    "forecast_ledger",
    "state_trends",
)


def _persistent_session_eligibility_reasons(
    row: dict[str, Any], *, verify_artifact_bytes: bool, batch_root: Path | None = None
) -> list[str]:
    """Validate model-visible memory and the authoritative semantic ledger."""

    if row.get("interaction_mode") != "logical_persistent":
        return []
    reasons: list[str] = []
    summary = row.get("trajectory_summary") or {}
    memory = summary.get("structured_memory")
    if not isinstance(memory, dict) or memory.get("schema_version") != (
        "persistent_working_memory_v2"
    ):
        reasons.append("structured_memory_missing_or_invalid")
        memory = {}
    if row.get("structured_memory") != memory:
        reasons.append("structured_memory_summary_mismatch")
    config = (row.get("agent_config") or {}).get("config") or {}
    limit = config.get("persistent_memory_max_items")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 4:
        reasons.append("structured_memory_limit_missing_or_invalid")
        limit = 0
    for field in _PERSISTENT_MEMORY_LIST_FIELDS:
        records = memory.get(field)
        if (
            not isinstance(records, list)
            or any(not isinstance(record, dict) for record in records)
            or (limit > 0 and len(records) > limit)
        ):
            reasons.append("structured_memory_schema_or_bound_invalid")
            break
    last_tick = memory.get("last_updated_tick")
    if last_tick is not None and (
        isinstance(last_tick, bool) or not isinstance(last_tick, int) or last_tick < 0
    ):
        reasons.append("structured_memory_schema_or_bound_invalid")
    llm = summary.get("llm") or {}
    expected_memory_bytes = len(
        json.dumps(
            memory,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    if llm.get("structured_memory_bytes") != expected_memory_bytes:
        reasons.append("structured_memory_stats_mismatch")
    artifact = summary.get("semantic_ledger_artifact")
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != "semantic_session_ledger_v1"
        or re.fullmatch(r"[0-9a-f]{64}", str(artifact.get("sha256") or "")) is None
        or isinstance(artifact.get("event_count"), bool)
        or not isinstance(artifact.get("event_count"), int)
        or int(artifact.get("event_count") or 0) <= 0
    ):
        reasons.append("semantic_ledger_artifact_missing_or_invalid")
    elif llm.get("session_ledger_events") != artifact.get("event_count"):
        reasons.append("semantic_ledger_stats_mismatch")
    if verify_artifact_bytes:
        reasons.extend(
            _trajectory_sidecar_eligibility_reasons(
                row,
                summary_key="semantic_ledger_artifact",
                stem="semantic_ledger",
                schema_version="semantic_session_ledger_v1",
                require_nonempty=True,
                batch_root=batch_root,
            )
        )
    return reasons


def _formal_row_eligibility(
    row: dict[str, Any],
    *,
    required_suite_hash: str | None = None,
    required_implementation_tree_sha256: str | None = None,
    required_interaction_mode: str | None = None,
    verify_artifact_bytes: bool = False,
    batch_root: Path | None = None,
) -> tuple[bool, list[str]]:
    """Return whether an episode may contribute to a formal evaluation.

    ``status == "ok"`` only means the simulator episode returned normally.
    Provider fallbacks, runner-truncated tool arguments, stale suite bindings,
    and the superseded v17 decision-wave protocol are diagnostic evidence
    rather than clean model measurements. Complete but malformed model output
    remains a model capability signal and is scored as a failed tool call.
    """
    reasons: list[str] = []
    if row.get("status") != "ok":
        reasons.append("episode_status_not_ok")
    if row.get("interaction_mode") not in {
        "logical_stateless",
        "logical_persistent",
    }:
        reasons.append("formal_row_interaction_mode_unsupported")
    elif (
        required_interaction_mode is not None
        and row.get("interaction_mode") != required_interaction_mode
    ):
        reasons.append("formal_row_interaction_mode_mismatch")
    reasons.extend(
        _persistent_session_eligibility_reasons(
            row,
            verify_artifact_bytes=False,
            batch_root=batch_root,
        )
    )

    llm = (row.get("trajectory_summary") or {}).get("llm") or {}
    reasons.extend(_llm_call_failure_eligibility_reasons(llm))
    if (
        int(llm.get("tool_argument_parse_failures", 0) or 0) > 0
        and llm.get("tool_argument_parse_classification_version") != 1
    ):
        reasons.append("tool_argument_parse_classification_missing")
    # Model-side output-cap exhaustion, malformed arguments, no-tool replies,
    # and resulting waits are capability outcomes, not transport contamination.
    # Excluding classified failures would create survivorship bias. Historical
    # rows without the parser-classification contract remain ambiguous and
    # fail closed; provider/API and benchmark prompt-budget failures above are
    # likewise retryable infrastructure errors.

    protocol = row.get("evaluation_protocol") or {}
    protocol_implementation = str(protocol.get("implementation_fingerprint") or "")
    if "v17-bounded-decision-waves" in protocol_implementation:
        reasons.append("diagnostic_protocol_v17")

    protocol_version = str(protocol.get("version") or "")
    suite_hash = str(row.get("suite_manifest_sha256") or "")
    if protocol_version.startswith("2."):
        reasons.extend(_formal_source_contract_reasons(row))
        trajectory_summary = row.get("trajectory_summary") or {}
        reasons.extend(_causal_response_contract_reasons(trajectory_summary))
        reasons.extend(
            _formal_primary_artifact_reasons(
                row, verify_artifact_bytes=verify_artifact_bytes
                , batch_root=batch_root
            )
        )
        if row.get("agent_treatment_sha256"):
            requested_model = str(row.get("model") or "")
            provider_models = llm.get("provider_models")
            if (
                not requested_model
                or not isinstance(provider_models, list)
                or not provider_models
                or any(model in (None, "") for model in provider_models)
            ):
                reasons.append("provider_model_identity_missing")
            elif any(str(model) != requested_model for model in provider_models):
                reasons.append("provider_model_identity_mismatch")
            identity_records = llm.get("provider_model_identity_records")
            if not isinstance(identity_records, list) or not identity_records:
                reasons.append("provider_model_identity_closure_missing")
            else:
                request_sequences: list[int] = []
                closure_counts = {
                    "exact": 0,
                    "missing": 0,
                    "mismatch": 0,
                    "request_failed": 0,
                }
                for record in identity_records:
                    if not isinstance(record, dict):
                        reasons.append("provider_model_identity_closure_inconsistent")
                        continue
                    if (
                        record.get("schema_version")
                        != "provider_model_identity_closure_v1"
                    ):
                        reasons.append("provider_model_identity_closure_inconsistent")
                    sequence = record.get("request_sequence")
                    if (
                        isinstance(sequence, bool)
                        or not isinstance(sequence, int)
                        or sequence < 1
                    ):
                        reasons.append("provider_model_identity_closure_inconsistent")
                    else:
                        request_sequences.append(sequence)
                    closure = str(record.get("closure") or "")
                    observed_models = record.get("observed_models")
                    if record.get("requested_model") != requested_model:
                        reasons.append("provider_model_identity_mismatch")
                    if not isinstance(observed_models, list):
                        reasons.append("provider_model_identity_closure_inconsistent")
                        observed_models = []
                    if closure == "exact":
                        closure_counts["exact"] += 1
                        if not observed_models:
                            reasons.append("provider_model_identity_missing")
                        elif any(
                            str(model) != requested_model for model in observed_models
                        ):
                            reasons.append("provider_model_identity_mismatch")
                    elif closure == "missing":
                        closure_counts["missing"] += 1
                        reasons.append("provider_model_identity_missing")
                    elif closure == "mismatch":
                        closure_counts["mismatch"] += 1
                        reasons.append("provider_model_identity_mismatch")
                    elif closure == "request_failed":
                        closure_counts["request_failed"] += 1
                    else:
                        reasons.append("provider_model_identity_closure_inconsistent")
                expected_sequences = list(range(1, len(identity_records) + 1))
                if sorted(request_sequences) != expected_sequences:
                    reasons.append("provider_model_identity_closure_inconsistent")
                if (
                    closure_counts["request_failed"] > 0
                    and "provider_call_failure" not in reasons
                ):
                    reasons.append("provider_call_failure")
                expected_counts = {
                    "provider_model_identity_request_count": len(identity_records),
                    "provider_model_identity_closed_count": len(identity_records),
                    "provider_model_identity_exact_count": closure_counts["exact"],
                    "provider_model_identity_missing_count": closure_counts["missing"],
                    "provider_model_identity_mismatch_count": closure_counts[
                        "mismatch"
                    ],
                    "provider_model_identity_failed_request_count": (
                        closure_counts["request_failed"]
                    ),
                    "provider_request_count": len(identity_records),
                    "provider_response_count": len(identity_records),
                }
                if any(
                    isinstance(llm.get(field), bool) or llm.get(field) != expected
                    for field, expected in expected_counts.items()
                ):
                    reasons.append("provider_model_identity_closure_inconsistent")
        if protocol_version != EVALUATION_PROTOCOL_VERSION:
            reasons.append("evaluation_protocol_version_mismatch")
        if protocol_implementation != EVALUATION_IMPLEMENTATION_FINGERPRINT:
            reasons.append("evaluation_implementation_fingerprint_mismatch")
        row_scoring_version = str(
            row.get("scoring_version")
            or (row.get("score") or {}).get("scoring_version")
            or ""
        )
        if row_scoring_version != SCORING_VERSION:
            reasons.append("scoring_version_mismatch")
        if not suite_hash:
            reasons.append("suite_manifest_missing")
        suite_eligibility = row.get("suite_eligibility")
        eligibility_hash = str(row.get("suite_eligibility_sha256") or "")
        if not isinstance(suite_eligibility, dict) or not eligibility_hash:
            reasons.append("suite_eligibility_missing")
        elif _canonical_json_sha256(suite_eligibility) != eligibility_hash:
            reasons.append("suite_eligibility_hash_mismatch")
        elif bool(suite_eligibility.get("suite_blocked")):
            reasons.append("suite_release_blocked")
        terminal_integrity = (row.get("trajectory_summary") or {}).get(
            "terminal_integrity"
        )
        if not isinstance(terminal_integrity, dict):
            reasons.append("terminal_integrity_missing")
        elif not bool(terminal_integrity.get("release_ready")):
            reasons.append("terminal_integrity_failure")
        event_contract = (row.get("trajectory_summary") or {}).get("event_contract")
        if not isinstance(event_contract, dict):
            reasons.append("event_decision_contract_missing")
        elif event_contract.get("schema_version") != EVENT_DECISION_CONTRACT_VERSION:
            reasons.append("event_decision_contract_version_mismatch")
        elif (
            isinstance(event_contract.get("violation_count"), bool)
            or not isinstance(event_contract.get("violation_count"), int)
            or event_contract.get("violation_count") != 0
        ):
            reasons.append("event_decision_contract_violation")
        provider_audit = (row.get("trajectory_summary") or {}).get(
            "provider_audit_artifact"
        )
        if not isinstance(provider_audit, dict):
            reasons.append("provider_audit_artifact_missing")
        elif (
            provider_audit.get("schema_version") != "provider_interaction_audit_v1"
            or re.fullmatch(r"[0-9a-f]{64}", str(provider_audit.get("sha256") or ""))
            is None
            or isinstance(provider_audit.get("event_count"), bool)
            or not isinstance(provider_audit.get("event_count"), int)
            or int(provider_audit.get("event_count") or 0) <= 0
        ):
            reasons.append("provider_audit_artifact_invalid")
        if verify_artifact_bytes:
            reasons.extend(
                _trajectory_sidecar_eligibility_reasons(
                    row,
                    summary_key="provider_audit_artifact",
                    stem="provider_audit",
                    schema_version="provider_interaction_audit_v1",
                    require_nonempty=True,
                    batch_root=batch_root,
                )
            )
            semantic_ledger = (row.get("trajectory_summary") or {}).get(
                "semantic_ledger_artifact"
            )
            if isinstance(semantic_ledger, dict):
                reasons.extend(
                    _trajectory_sidecar_eligibility_reasons(
                        row,
                        summary_key="semantic_ledger_artifact",
                        stem="semantic_ledger",
                        schema_version="semantic_session_ledger_v1",
                        require_nonempty=True,
                        batch_root=batch_root,
                    )
                )
        semantic_coverage = (row.get("trajectory_summary") or {}).get(
            "tool_semantic_coverage"
        )
        if not isinstance(semantic_coverage, dict):
            reasons.append("tool_semantic_coverage_missing")
        elif semantic_coverage.get("covered") is False:
            reasons.append("unknown_tool_semantics")
        elif not _semantic_coverage_is_formally_valid(semantic_coverage):
            reasons.append("tool_semantic_coverage_inconsistent")
        tool_surface = (row.get("trajectory_summary") or {}).get(
            "tool_surface_contract"
        )
        if not isinstance(tool_surface, dict):
            reasons.append("tool_surface_contract_missing")
        elif (
            tool_surface.get("schema_version") != "tool-surface-contract-v1"
            or tool_surface.get("complete") is not True
            or re.fullmatch(
                r"[0-9a-f]{64}",
                str(tool_surface.get("exposed_schema_sha256") or ""),
            )
            is None
            or any(
                tool_surface.get(field)
                for field in (
                    "missing_observation_tool_names",
                    "missing_control_tool_names",
                    "missing_commit_control_tool_names",
                )
            )
        ):
            reasons.append("tool_surface_contract_incomplete")
        construct_contract = protocol.get("construct_contract")
        if not construct_contract:
            reasons.append("construct_contract_missing")
        elif construct_contract != "operational_agency.v1":
            reasons.append("construct_contract_mismatch")
        if construct_contract == "operational_agency.v1":
            if not _agency_attribution_is_formally_complete(row.get("counterfactual")):
                reasons.append("construct_attribution_incomplete")
            if not _operational_agency_profile_is_consistent(
                trajectory_summary,
                counterfactual=row.get("counterfactual"),
            ):
                reasons.append("construct_evidence_inconsistent")
    if required_suite_hash is not None and suite_hash != required_suite_hash:
        reasons.append("suite_manifest_mismatch")
    if (
        required_implementation_tree_sha256 is not None
        and row.get("implementation_tree_sha256") != required_implementation_tree_sha256
    ):
        reasons.append("implementation_tree_mismatch")
    return not reasons, reasons


def _row_is_clean_for_resume(
    row: dict[str, Any], *, batch_root: Path | None = None
) -> bool:
    # Historical diagnostic rows predate treatment identity. They may remain
    # readable for legacy analysis, but formal eligibility above still fails
    # closed and every newly treatment-bound row must carry an exact mode.
    historical_unbound = not row.get("agent_treatment_sha256")
    _, reasons = _formal_row_eligibility(
        row, verify_artifact_bytes=not historical_unbound, batch_root=batch_root
    )
    tolerated = {"suite_release_blocked"}
    if "interaction_mode" not in row and historical_unbound:
        tolerated.add("formal_row_interaction_mode_unsupported")
    if historical_unbound:
        trajectory_summary = row.get("trajectory_summary") or {}
        tolerated.update(
            {
                "construct_contract_missing",
                "source_denominator_key_missing",
                "case_ledger_missing_or_invalid",
                "event_response_records_missing_or_invalid",
                "trajectory_artifact_missing_or_invalid",
                "evidence_ledger_artifact_missing_or_invalid",
            }
        )
        if "event_contract" not in trajectory_summary:
            tolerated.add("event_decision_contract_missing")
        if "provider_audit_artifact" not in trajectory_summary:
            tolerated.add("provider_audit_artifact_missing")
        if "tool_surface_contract" not in trajectory_summary:
            tolerated.add("tool_surface_contract_missing")
    # Candidate-wide release blockers do not contaminate an otherwise clean
    # episode checkpoint. Reuse it for diagnostic/remediation runs while
    # continuing to exclude it from every formal leaderboard view.
    return not (set(reasons) - tolerated)


def _filter_pending_jobs(
    jobs: list[dict[str, Any]], rows: list[dict[str, Any]],
    *, batch_root: Path | None = None,
) -> list[dict[str, Any]]:
    required_trees = {
        str(job.get("implementation_tree_sha256") or "") for job in jobs
    } - {""}
    eligible_rows = (
        [
            row
            for row in rows
            if str(row.get("implementation_tree_sha256") or "") in required_trees
        ]
        if required_trees
        else rows
    )
    legacy_done, strong_done = _completed_ok_resume_index(
        eligible_rows, batch_root=batch_root
    )
    pending: list[dict[str, Any]] = []
    for job in jobs:
        legacy, strong = _job_resume_keys(job)
        if strong in strong_done or (strong is None and legacy in legacy_done):
            continue
        pending.append(job)
    return pending


def _run_llm_episode_job(job: dict[str, Any]) -> dict[str, Any]:
    """Process-pool worker: one episode with optional trajectory + log file."""
    sentinel = _active_quota_sentinel(job)
    if sentinel is not None:
        return _quota_parked_result(job, reset_at=sentinel.get("reset_at"))

    slug = job["scenario_slug"]
    seed = int(job["seed"])
    cfg = _llm_config_from_dict(job["llm_config"])
    start_tree = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    expected_tree = str(job.get("implementation_tree_sha256") or start_tree)
    kwargs = {"config": cfg}
    run_options: dict[str, Any] = {}
    run_options.update(
        {
            "per_action_attribution": True,
            "per_action_cap": None,
            "per_action_group_attribution": True,
            "per_action_group_cap": None,
        }
    )
    if job.get("trajectory_dir"):
        _quarantine_trajectory_dir(job["trajectory_dir"])
        run_options["trajectory_dir"] = job["trajectory_dir"]
    if job.get("episode_log_path"):
        run_options["episode_log_path"] = job["episode_log_path"]
    r: dict[str, Any]
    if start_tree != expected_tree:
        r = {
            "status": "error",
            "error": "implementation_tree_changed_before_episode",
        }
    else:
        r = _run_one_safe((slug, "llm_agent", seed, kwargs, run_options))
    r["execution_started"] = start_tree == expected_tree
    _portabilize_formal_trajectory_json_sidecars(job)
    end_tree = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    r["implementation_tree_sha256"] = expected_tree
    r["implementation_tree_sha256_start"] = start_tree
    r["implementation_tree_sha256_end"] = end_tree
    if start_tree != expected_tree or end_tree != expected_tree:
        r["status"] = "error"
        r["error"] = "implementation_tree_drift"
    r = _apply_llm_job_metadata(job, r)
    r["temperature"] = float(job.get("temperature", cfg.temperature))
    if _row_is_quota_exhausted(r):
        r["quota_parked"] = True
        reset_at = _quota_reset_text(r.get("error"))
        if reset_at:
            r["quota_reset_at"] = reset_at
        _write_quota_sentinel(job, r)
    return r


def _portabilize_formal_trajectory_json_sidecars(job: dict[str, Any]) -> None:
    """Make formal trajectory JSON sidecars relocatable with their batch tree."""

    if job.get("formal_run") is not True or not job.get("trajectory_dir"):
        return
    batch_root = Path(str(job.get("batch_output_dir") or "")).resolve()
    trajectory_dir = Path(str(job["trajectory_dir"])).resolve()
    try:
        trajectory_dir.relative_to(batch_root)
    except ValueError as exc:
        raise ValueError("formal trajectory directory escapes batch root") from exc
    for path in sorted(trajectory_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        portable = canonicalize_repo_owned_paths(payload, repo_root=batch_root)
        _atomic_write_text(
            path,
            json.dumps(portable, indent=2, ensure_ascii=False) + "\n",
        )


def _quarantine_trajectory_dir(raw_path: str | Path) -> Path | None:
    """Move an orphaned/partial episode directory aside before a clean rerun."""
    path = Path(raw_path)
    if not path.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    stale = path.with_name(f"{path.name}.stale-{stamp}")
    suffix = 1
    while stale.exists():
        stale = path.with_name(f"{path.name}.stale-{stamp}.{suffix}")
        suffix += 1
    path.rename(stale)
    return stale


def _build_model_lanes(
    jobs: list[dict[str, Any]], episodes_path: Path | None = None
) -> list[dict[str, Any]]:
    """Group jobs into per-model sequential lanes, preserving encounter order."""
    lanes_by_model: dict[str, list[dict[str, Any]]] = {}
    model_order: list[str] = []
    for job in jobs:
        model = str(job["model"])
        if model not in lanes_by_model:
            lanes_by_model[model] = []
            model_order.append(model)
        lanes_by_model[model].append(job)
    out: list[dict[str, Any]] = []
    episodes_path_str = str(episodes_path) if episodes_path is not None else None
    for model in model_order:
        out.append(
            {
                "model": model,
                "jobs": lanes_by_model[model],
                "episodes_path": episodes_path_str,
            }
        )
    return out


def _append_jsonl_atomic(path: str | Path, row: dict[str, Any]) -> None:
    """Atomically append one JSON line from a worker process."""
    import fcntl

    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("short write while appending episodes JSONL")
            remaining = remaining[written:]
        os.fsync(fd)
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _acquire_output_dir_lock(out_dir: Path) -> Any:
    """Hold an exclusive process lock for one mutable batch output directory."""
    import fcntl

    out_dir.mkdir(parents=True, exist_ok=True)
    handle = (out_dir / ".run.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        handle.close()
        raise RuntimeError(
            f"output directory already has an active runner: {out_dir}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {"pid": os.getpid(), "acquired_at_utc": datetime.now(UTC).isoformat()}
        )
        + "\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def _run_llm_model_lane(lane: dict[str, Any]) -> dict[str, Any]:
    """Process-pool worker: run one model sequentially over its assigned jobs."""
    model = str(lane["model"])
    jobs = list(lane.get("jobs") or [])
    episodes_path = lane.get("episodes_path")
    n_completed = 0
    parked_reset_at: str | None = None
    for job in jobs:
        if parked_reset_at is not None or _active_quota_sentinel(job):
            if parked_reset_at is None:
                sentinel = _active_quota_sentinel(job) or {}
                parked_reset_at = sentinel.get("reset_at")
            row = _quota_parked_result(job, reset_at=parked_reset_at)
            if episodes_path:
                _append_jsonl_atomic(episodes_path, row)
            n_completed += 1
            continue
        if job.get("episode_log_path"):
            _quarantine_log_file(job["episode_log_path"], tag="stale")
        if episodes_path:
            _append_jsonl_atomic(episodes_path, _in_flight_placeholder_row(job))
        row = _run_llm_episode_job(job)
        if _row_is_quota_exhausted(row):
            parked_reset_at = row.get("quota_reset_at") or _quota_reset_text(
                row.get("error")
            )
        if episodes_path:
            _append_jsonl_atomic(episodes_path, row)
        n_completed += 1
    return {"model": model, "n_completed": n_completed}


def _run_global_jobs(
    jobs: list[dict[str, Any]], episodes_path: Path, write_mode: str, max_workers: int
) -> None:
    import concurrent.futures as futures

    worker_count = max(1, min(max_workers, len(jobs)))
    submission_window = max(1, worker_count * 2)
    with (
        open(episodes_path, write_mode, encoding="utf-8") as ep_f,
        futures.ProcessPoolExecutor(max_workers=worker_count) as pool,
    ):
        completed = 0
        job_offset = 0
        parked_models: dict[str, str | None] = {}
        pending: dict[Any, dict[str, Any]] = {}

        def submit_available() -> None:
            """Keep the bounded queue full as soon as a worker frees up."""
            nonlocal completed, job_offset
            while job_offset < len(jobs) and len(pending) < submission_window:
                job = jobs[job_offset]
                job_offset += 1
                model = str(job["model"])
                if model in parked_models:
                    row = _quota_parked_result(job, reset_at=parked_models[model])
                    ep_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    ep_f.flush()
                    completed += 1
                    continue
                if job.get("episode_log_path"):
                    _quarantine_log_file(job["episode_log_path"], tag="stale")
                # Write a durable checkpoint before dispatching the worker. If
                # a long-horizon episode or the runner is interrupted before a
                # terminal row exists, resume/finalize can distinguish that
                # cell from a never-submitted job and rerun it cleanly.
                ep_f.write(
                    json.dumps(_in_flight_placeholder_row(job), ensure_ascii=False)
                    + "\n"
                )
                ep_f.flush()
                pending[pool.submit(_run_llm_episode_job, job)] = job

        def record_completed(future: Any) -> None:
            nonlocal completed
            job = pending.pop(future)
            row = future.result()
            # Keep terminal rows keyed with the same cell identity as their
            # pre-dispatch checkpoint, even if a provider/fixture returns a
            # sparse result before the normal metadata decorator runs.
            for key in (
                "scenario_slug",
                "model",
                "seed",
                "scenario_signature",
                "temperature",
                "pass_id",
                "pass_index",
                "pass_k",
                "evaluation_implementation_fingerprint",
            ):
                if key in job:
                    row.setdefault(key, job[key])
            row.setdefault("suite_manifest_sha256", job.get("suite_manifest_sha256"))
            row.setdefault("suite_eligibility", job.get("suite_eligibility"))
            row.setdefault(
                "suite_eligibility_sha256", job.get("suite_eligibility_sha256")
            )
            ep_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            ep_f.flush()
            completed += 1
            if _row_is_quota_exhausted(row):
                parked_models[str(row.get("model") or job["model"])] = row.get(
                    "quota_reset_at"
                ) or _quota_reset_text(row.get("error"))
            if completed % 10 == 0 or completed == len(jobs):
                LOGGER.info("completed %d / %d new episodes", completed, len(jobs))

        try:
            submit_available()
            while pending or job_offset < len(jobs):
                if not pending:
                    submit_available()
                    continue
                # ``as_completed`` is deliberately recreated after each
                # completion. This gives us a real bounded work queue: a
                # finished long-horizon episode immediately makes room for
                # the next job instead of waiting for the whole window.
                future = next(iter(futures.as_completed(tuple(pending))))
                record_completed(future)
                submit_available()
        except BaseException:
            for future in pending:
                future.cancel()
            terminate_workers = getattr(pool, "terminate_workers", None)
            if callable(terminate_workers):
                terminate_workers()
            else:
                for process in getattr(pool, "_processes", {}).values():
                    process.terminate()
            raise


def _model_list(env: dict[str, str], override: str | None) -> list[str]:
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    raw = env.get("OPERATE_MODELS", "")
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return [
        "gpt-5-2025-08-07",
        "gemini-2.5-pro-preview-06-05",
        "o3-2025-04-16",
        "gpt-4.1-2025-04-14",
    ]


def _model_label(row: dict[str, Any]) -> str:
    """Normalized model id from an episode row (drops the ``llm_agent/`` prefix)."""
    return str(row.get("model") or row.get("agent_name", "")).replace("llm_agent/", "")


def _row_is_in_configured_scope(
    row: dict[str, Any], coverage: dict[str, Any] | None
) -> bool:
    """Return whether *row* is one of the cells the formal run promised."""
    coverage = coverage or {}
    configured_models = set(coverage.get("configured_models") or [])
    comparable_pairs = {
        (str(pair[0]), int(pair[1]))
        for pair in coverage.get("_comparable_pairs") or []
        if isinstance(pair, list) and len(pair) == 2
    }
    model = _model_label(row)
    slug = str(row.get("scenario_slug") or "")
    seed = row.get("seed")
    if model not in configured_models or not slug or seed is None:
        return False
    pair = (slug, int(seed))
    if pair not in comparable_pairs:
        return False
    allowed = set(
        ((coverage.get("_comparable_pass_ids") or {}).get(model) or {}).get(
            f"{slug}||{int(seed)}",
            [f"pass-{index}" for index in range(int(coverage.get("pass_k") or 1))],
        )
    )
    return str(row.get("pass_id") or "pass-0") in allowed


def _coverage_summary(
    results: list[dict[str, Any]],
    *,
    configured_models: list[str],
    configured_seeds: list[int],
    n_scenarios: int,
    pass_k: int = 1,
    configured_pairs: list[list[Any]] | None = None,
) -> dict[str, Any]:
    """Compute expected/realized coverage and the cross-model comparable pair set.

    A pair is ``(scenario_slug, seed)``. ``comparable_pairs`` is the set of pairs
    for which **every configured model** has a ``status=="ok"`` row, i.e. the
    only slice on which a fair, like-for-like leaderboard can be computed when a
    batch is partial.

    Notes:
    - ``configured_seeds`` is deduplicated before counting so an accidental
      ``--seeds 42 42`` does not inflate ``expected_total`` and falsely flag a
      complete batch as partial.
    - Non-configured models that appear in ``results`` (e.g. baseline rows or
      legacy data) are tracked separately under ``extra_models_seen`` and do
      NOT pollute ``per_model_realized`` / ``per_model_coverage``.
    """
    n_models = max(1, len(configured_models))
    unique_seeds = sorted({int(s) for s in configured_seeds})
    pair_filter = (
        {(str(pair[0]), int(pair[1])) for pair in configured_pairs}
        if configured_pairs
        else None
    )
    n_seeds = max(1, len(unique_seeds))
    pass_k = max(1, int(pass_k))
    required_pass_ids = [f"pass-{index}" for index in range(pass_k)]
    required_pass_id_set = set(required_pass_ids)
    expected_base_pairs = (
        len(pair_filter) if pair_filter is not None else int(n_scenarios) * n_seeds
    )
    expected_per_model = expected_base_pairs * pass_k
    expected_total = expected_per_model * n_models
    seed_filter = set(unique_seeds)

    expected_pairs = {
        (str(r.get("scenario_slug") or ""), int(r.get("seed", -1)))
        for r in results
        if r.get("status") == "ok"
        and str(r.get("scenario_slug") or "")
        and r.get("seed") is not None
        and (pair_filter is not None or int(r.get("seed", -1)) in seed_filter)
    }
    if pair_filter is not None:
        expected_pairs = set(pair_filter)
    per_model_pairs: dict[str, set[tuple[str, int]]] = {
        m: set() for m in configured_models
    }
    per_model_pass_units: dict[str, dict[tuple[str, int], set[str]]] = {
        m: defaultdict(set) for m in configured_models
    }
    extra_models_seen: set[str] = set()
    per_model_execution_pass_units: dict[str, dict[tuple[str, int], set[str]]] = {
        m: defaultdict(set) for m in configured_models
    }
    for r in results:
        m = _model_label(r)
        slug = str(r.get("scenario_slug") or "")
        seed = r.get("seed")
        if not slug or seed is None:
            continue
        seed_int = int(seed)
        base_pair = (slug, seed_int)
        in_scope = m in per_model_pairs and (
            (pair_filter is not None and base_pair in pair_filter)
            or (pair_filter is None and (not seed_filter or seed_int in seed_filter))
        )
        pass_id = str(r.get("pass_id") or "pass-0").strip()
        if r.get("status") == "ok" and in_scope and pass_id in required_pass_id_set:
            per_model_execution_pass_units[m][base_pair].add(pass_id)
        if not _formal_row_eligibility(r)[0]:
            continue
        m = _model_label(r)
        slug = str(r.get("scenario_slug") or "")
        seed = r.get("seed")
        if not slug or seed is None:
            continue
        seed_int = int(seed)
        # Only configured (model, seed) cells count toward coverage; episodes
        # for unrelated agents or out-of-scope seeds are tracked as "extra".
        base_pair = (slug, seed_int)
        if (
            m not in per_model_pairs
            or (pair_filter is not None and base_pair not in pair_filter)
            or (pair_filter is None and seed_filter and seed_int not in seed_filter)
        ):
            if m and m not in per_model_pairs:
                extra_models_seen.add(m)
            continue
        pass_id = str(r.get("pass_id") or "pass-0").strip()
        if pass_id not in required_pass_id_set:
            continue
        per_model_pass_units[m][base_pair].add(pass_id)

    for model in configured_models:
        per_model_pairs[model] = {
            pair
            for pair, pass_ids in per_model_pass_units[model].items()
            if required_pass_id_set.issubset(pass_ids)
        }

    per_model_realized = {
        m: (
            sum(
                len(per_model_pass_units[m].get(pair, set()) & required_pass_id_set)
                for pair in expected_pairs
            )
        )
        for m in configured_models
    }
    per_model_coverage = {
        m: (
            round(per_model_realized[m] / expected_per_model, 4)
            if expected_per_model
            else 0.0
        )
        for m in configured_models
    }
    per_model_execution_realized = {
        m: sum(
            len(
                per_model_execution_pass_units[m].get(pair, set())
                & required_pass_id_set
            )
            for pair in expected_pairs
        )
        for m in configured_models
    }
    per_model_execution_coverage = {
        m: (
            round(per_model_execution_realized[m] / expected_per_model, 4)
            if expected_per_model
            else 0.0
        )
        for m in configured_models
    }

    if configured_models:
        comparable_pairs_set: set[tuple[str, int]] = set.intersection(
            *(per_model_pairs[m] for m in configured_models)
        )
    else:
        comparable_pairs_set = set()
    comparable_pass_ids: dict[str, dict[str, list[str]]] = {}
    for model in configured_models:
        comparable_pass_ids[model] = {
            f"{slug}||{seed}": list(required_pass_ids)
            for pair in sorted(comparable_pairs_set)
            for slug, seed in [pair]
        }

    is_partial_batch = (not configured_models) or any(
        per_model_realized.get(m, 0) < expected_per_model for m in configured_models
    )
    warning: str | None = None
    if not configured_models:
        warning = (
            "No configured models recorded; comparability cannot be assessed. "
            "Treat the raw `leaderboard` as informational only."
        )
    elif is_partial_batch:
        worst = min(
            (per_model_coverage.get(m, 0.0) for m in configured_models), default=0.0
        )
        if comparable_pairs_set:
            warning = (
                f"Partial batch: at least one configured model has coverage < 1.0 "
                f"(min={worst:.2f}). The raw `leaderboard` is NOT comparable across "
                f"models because per-model n_episodes differ. Use "
                f"`intersection_leaderboard` (n_pairs={len(comparable_pairs_set)}) "
                f"for fair model comparison."
            )
        else:
            warning = (
                f"Partial batch: at least one configured model has coverage < 1.0 "
                f"(min={worst:.2f}) and there are NO scenario/seed pairs every "
                f"configured model completed. Rerun the missing cells before "
                f"comparing models — neither the raw `leaderboard` nor "
                f"`intersection_leaderboard` is fair."
            )

    return {
        "expected_episodes_per_model": expected_per_model,
        "expected_total": expected_total,
        "configured_models": list(configured_models),
        "configured_seeds": unique_seeds,
        "seed_mode": "scenario" if pair_filter is not None else "fixed",
        "pass_k": pass_k,
        "n_scenarios": int(n_scenarios),
        "per_model_realized": per_model_realized,
        "per_model_coverage": per_model_coverage,
        "per_model_execution_realized": per_model_execution_realized,
        "per_model_execution_coverage": per_model_execution_coverage,
        "extra_models_seen": sorted(extra_models_seen),
        "comparable_intersection_size": len(comparable_pairs_set),
        "is_partial_batch": bool(is_partial_batch),
        "comparability_warning": warning,
        # Internal: kept under a leading underscore so consumers know not to
        # treat it as a stable public field; serialized as a sorted list of
        # ``[slug, seed]`` pairs so it round-trips through JSON.
        "_comparable_pairs": [list(p) for p in sorted(comparable_pairs_set)],
        "_comparable_pass_ids": comparable_pass_ids,
    }


def _pass_k_success_summary(
    results: list[dict[str, Any]],
    *,
    configured_models: list[str],
    configured_seeds: list[int],
    n_scenarios: int,
    pass_k: int = 1,
    configured_pairs: list[list[Any]] | None = None,
) -> dict[str, Any]:
    """Estimate pass^k reliability over configured scenario/model/seed cells.

    Coverage answers "did we run enough rows?"; this answers "for each
    scenario/seed cell, did all k independent attempts finish successfully?".
    Replicates are keyed by explicit pass_id values so duplicate retries cannot
    inflate reliability.
    """
    pass_k = max(1, int(pass_k))
    unique_seeds = sorted({int(s) for s in configured_seeds})
    seed_filter = set(unique_seeds)
    pair_filter = (
        {(str(pair[0]), int(pair[1])) for pair in configured_pairs}
        if configured_pairs
        else None
    )
    configured_model_set = set(configured_models)
    required_pass_ids = [f"pass-{i}" for i in range(pass_k)]
    required_pass_id_set = set(required_pass_ids)
    scenario_slugs = sorted(
        {
            str(row.get("scenario_slug") or "")
            for row in results
            if str(row.get("scenario_slug") or "")
            and row.get("seed") is not None
            and (pair_filter is not None or int(row.get("seed", -1)) in seed_filter)
            and _model_label(row) in configured_model_set
        }
    )
    expected_cells_per_model = (
        len(pair_filter)
        if pair_filter is not None
        else int(n_scenarios) * len(unique_seeds)
    )
    if len(scenario_slugs) < int(n_scenarios):
        for idx in range(int(n_scenarios) - len(scenario_slugs)):
            scenario_slugs.append(f"__missing_scenario_{idx}")

    per_model_pass: dict[str, dict[tuple[str, int], dict[str, str]]] = {
        model: defaultdict(dict) for model in configured_models
    }
    duplicate_counts: dict[str, int] = {model: 0 for model in configured_models}
    ignored_extra_pass_counts: dict[str, int] = {
        model: 0 for model in configured_models
    }
    for row in results:
        model = _model_label(row)
        if model not in configured_model_set:
            continue
        slug = str(row.get("scenario_slug") or "")
        seed = row.get("seed")
        if not slug or seed is None:
            continue
        seed_int = int(seed)
        cell = (slug, seed_int)
        if pair_filter is not None and cell not in pair_filter:
            continue
        if pair_filter is None and seed_filter and seed_int not in seed_filter:
            continue
        pass_id = str(row.get("pass_id") or "pass-0")
        if pass_id not in required_pass_id_set:
            ignored_extra_pass_counts[model] += 1
            continue
        if pass_id in per_model_pass[model][cell]:
            duplicate_counts[model] += 1
            continue
        per_model_pass[model][cell][pass_id] = (
            "unavailable" if _row_is_quota_exhausted(row)
            else str(row.get("status", "ok"))
        )

    per_model: dict[str, dict[str, Any]] = {}
    total_successful = 0
    total_cells = expected_cells_per_model * len(configured_models)
    for model in configured_models:
        successful_cells = 0
        failed_cells = 0
        missing_pass_units = 0
        error_pass_units = 0
        unavailable_pass_units = 0
        unavailable_cells = 0
        expected_cells = (
            sorted(pair_filter)
            if pair_filter is not None
            else [
                (slug, seed)
                for slug in scenario_slugs[: int(n_scenarios)]
                for seed in unique_seeds
            ]
        )
        for slug, seed in expected_cells:
            pass_status = per_model_pass[model].get((slug, seed), {})
            missing = [
                pass_id for pass_id in required_pass_ids if pass_id not in pass_status
            ]
            errors = [
                pass_id for pass_id, status in pass_status.items()
                if status not in {"ok", "unavailable"}
            ]
            unavailable = [
                pass_id for pass_id, status in pass_status.items()
                if status == "unavailable"
            ]
            missing_pass_units += len(missing)
            error_pass_units += len(errors)
            unavailable_pass_units += len(unavailable)
            if unavailable:
                unavailable_cells += 1
            elif not missing and not errors:
                successful_cells += 1
            else:
                failed_cells += 1
        total_successful += successful_cells
        probability = (
            successful_cells / expected_cells_per_model
            if expected_cells_per_model
            else 0.0
        )
        per_model[model] = {
            "expected_cells": expected_cells_per_model,
            "successful_cells": successful_cells,
            "failed_cells": failed_cells,
            "success_probability": round(probability, 4),
            "missing_pass_units": missing_pass_units,
            "error_pass_units": error_pass_units,
            "unavailable_pass_units": unavailable_pass_units,
            "unavailable_cells": unavailable_cells,
            "duplicate_pass_units_ignored": duplicate_counts[model],
            "extra_pass_units_ignored": ignored_extra_pass_counts[model],
        }

    overall_probability = total_successful / total_cells if total_cells else 0.0
    return {
        "metric_kind": "execution_completion",
        "pass_k": pass_k,
        "definition": (
            "A scenario/model/seed cell succeeds only when every required "
            "explicit pass_id has status=ok. This measures execution completion, "
            "not task success or formal eligibility. Provider quota-unavailable "
            "cells retain the planned denominator but are not execution failures."
        ),
        "required_pass_ids": required_pass_ids,
        "configured_models": list(configured_models),
        "configured_seeds": unique_seeds,
        "seed_mode": "scenario" if pair_filter is not None else "fixed",
        "n_scenarios": int(n_scenarios),
        "expected_cells_per_model": expected_cells_per_model,
        "expected_total_cells": total_cells,
        "successful_total_cells": total_successful,
        "overall_success_probability": round(overall_probability, 4),
        "per_model": per_model,
    }


def _intersection_leaderboard(
    results: list[dict[str, Any]],
    coverage: dict[str, Any],
) -> list[dict[str, Any]]:
    """Leaderboard restricted to (scenario_slug, seed) pairs all models completed."""
    pairs = {(str(s), int(seed)) for s, seed in coverage.get("_comparable_pairs", [])}
    configured_models = list(coverage.get("configured_models") or [])
    pass_k = max(1, int(coverage.get("pass_k", 1) or 1))
    comparable_pass_ids = coverage.get("_comparable_pass_ids") or {}
    if not pairs or not configured_models:
        return []
    per_model: dict[str, list[float]] = {m: [] for m in configured_models}
    per_fatal: dict[str, list[bool]] = {m: [] for m in configured_models}
    seen_pass_units: set[tuple[str, str, int, str]] = set()
    for r in results:
        if r.get("status") != "ok":
            continue
        m = _model_label(r)
        if m not in per_model:
            continue
        slug = str(r.get("scenario_slug") or "")
        seed = r.get("seed")
        if seed is None or (slug, int(seed)) not in pairs:
            continue
        if pass_k > 1:
            allowed = comparable_pass_ids.get(m, {}).get(f"{slug}||{int(seed)}") or []
            pass_id = str(r.get("pass_id") or "")
            if pass_id not in set(allowed):
                continue
            pass_unit = (m, slug, int(seed), pass_id)
            if pass_unit in seen_pass_units:
                continue
            seen_pass_units.add(pass_unit)
        per_model[m].append(
            float((r.get("score") or {}).get("total_score", 0.0) or 0.0)
        )
        per_fatal[m].append(
            bool((r.get("ground_truth_summary") or {}).get("chose_fatal_option", False))
        )
    rows = build_leaderboard(per_model, fatal_flags=per_fatal)
    return [row.to_dict() for row in rows]


# Legal `batch_state` values. Treat this list as the canonical contract:
# downstream consumers (papers, dashboards, audit scripts) should switch on
# these strings and refuse anything else.
BATCH_STATE_FINAL = "final"
BATCH_STATE_PARTIAL = "partial"
BATCH_STATE_DEGRADED = "degraded"
BATCH_STATE_UNKNOWN = "unknown"
LEGAL_BATCH_STATES = (
    BATCH_STATE_FINAL,
    BATCH_STATE_PARTIAL,
    BATCH_STATE_DEGRADED,
    BATCH_STATE_UNKNOWN,
)


def _batch_state(
    *,
    coverage: dict[str, Any] | None,
    results: list[dict[str, Any]],
    log_audit_report: dict[str, Any] | None,
    required_suite_hash: str | None = None,
    required_interaction_mode: str | None = None,
) -> dict[str, Any]:
    """Compute the formal partial/final state of a batch.

    Returns a dict with at least ``batch_state`` and a list of human-readable
    ``reasons`` so consumers don't have to re-derive ``why`` it landed in a
    particular state.

    Rules:

    - ``unknown``  : ``coverage.configured_models`` is empty (we cannot tell
      what the run was supposed to cover).
    - ``degraded`` : full coverage but at least one of:
        * any episode row has ``status != "ok"``
        * ``log_files_orphan_interrupted`` > 0
        * too many otherwise-ok rows are dominated by fallback waits
      i.e. the cell grid is filled but with caveats.
    - ``partial``  : ``coverage.is_partial_batch`` is True (some configured
      cells are missing).
    - ``final``    : full coverage, all rows ``status=="ok"``, no interrupted
      orphan logs.

    Note: ``partial`` takes precedence over ``degraded`` when both apply,
    because a partial batch is structurally not yet comparable; the
    degradation only matters once the grid is full.
    """
    reasons: list[str] = []
    coverage = coverage or {}
    execution_counts = _execution_status_counts(results)
    configured_models = list(coverage.get("configured_models") or [])
    if not configured_models:
        return {
            "batch_state": BATCH_STATE_UNKNOWN,
            "reasons": [
                "no configured models recorded; batch state cannot be determined",
            ],
            **execution_counts,
            "n_orphan_interrupted_logs": int(
                (log_audit_report or {}).get("log_files_orphan_interrupted", 0) or 0
            ),
        }

    is_partial = bool(coverage.get("is_partial_batch", False)) or bool(
        execution_counts["n_episodes_quota_unavailable"]
        or execution_counts["n_episodes_in_flight"]
    )
    n_errors = execution_counts["n_episodes_error"]
    if execution_counts["n_episodes_quota_unavailable"]:
        reasons.append(
            f"{execution_counts['n_episodes_quota_unavailable']} provider-quota-unavailable "
            "episodes require retry; no task outcome measured"
        )
    n_interrupted = int(
        (log_audit_report or {}).get("log_files_orphan_interrupted", 0) or 0
    )
    ok_rows = [r for r in results if r.get("status") == "ok"]
    provider_contaminated_rows = 0
    prompt_budget_rows = 0
    for r in ok_rows:
        _, row_reasons = _formal_row_eligibility(
            r,
            required_suite_hash=required_suite_hash,
            required_interaction_mode=required_interaction_mode,
        )
        if "provider_call_failure" in row_reasons:
            provider_contaminated_rows += 1
        if "prompt_budget_exceeded" in row_reasons:
            prompt_budget_rows += 1
    n_provider_contaminated = provider_contaminated_rows
    n_prompt_budget_contaminated = prompt_budget_rows
    n_model_output_failures = sum(
        int(
            int(llm.get("provider_output_truncation_count", 0) or 0) > 0
            or int(llm.get("tool_argument_parse_failures", 0) or 0) > 0
        )
        for row in ok_rows
        for llm in [((row.get("trajectory_summary") or {}).get("llm") or {})]
    )
    n_diagnostic_protocol = sum(
        "diagnostic_protocol_v17"
        in _formal_row_eligibility(
            r,
            required_suite_hash=required_suite_hash,
            required_interaction_mode=required_interaction_mode,
        )[1]
        for r in ok_rows
    )
    configured_ineligible_rows = [
        r
        for r in results
        if _row_is_in_configured_scope(r, coverage)
        and not _formal_row_eligibility(
            r,
            required_suite_hash=required_suite_hash,
            required_interaction_mode=required_interaction_mode,
        )[0]
    ]
    n_configured_ineligible = len(configured_ineligible_rows)
    n_high_fallback = sum(
        1
        for r in ok_rows
        if float(
            ((r.get("trajectory_summary") or {}).get("llm") or {}).get(
                "fallback_wait_ratio", 0.0
            )
            or 0.0
        )
        > FALLBACK_WAIT_RATIO_THRESHOLD
    )
    if is_partial:
        worst = min(
            (
                coverage.get("per_model_coverage", {}).get(m, 0.0)
                for m in configured_models
            ),
            default=0.0,
        )
        reasons.append(
            f"partial batch: at least one model has coverage < 1.0 (min={worst:.2f})"
        )
        # Surface collateral degradation even when partial dominates the
        # state: operators should know the partial grid ALSO contains
        # error rows / interrupted orphans so they can clean those up
        # in the same rerun.
        if n_errors > 0:
            reasons.append(f"{n_errors} episode rows have status != 'ok' (collateral)")
        if n_interrupted > 0:
            reasons.append(
                f"{n_interrupted} orphan log file(s) look like interrupted attempts (collateral)"
            )
        if n_provider_contaminated > 0:
            reasons.append(
                f"{n_provider_contaminated} provider-contaminated episode(s) require retry (collateral)"
            )
        if n_prompt_budget_contaminated > 0:
            reasons.append(
                f"{n_prompt_budget_contaminated} prompt-budget-contaminated episode(s) require retry (collateral)"
            )
        if n_diagnostic_protocol > 0:
            reasons.append(
                f"{n_diagnostic_protocol} diagnostic protocol-v17 episode(s) are not formal measurements (collateral)"
            )
        if n_configured_ineligible > 0:
            reasons.append(
                f"{n_configured_ineligible} configured episode(s) fail formal eligibility (collateral)"
            )
        state = BATCH_STATE_PARTIAL
    elif (
        n_errors > 0
        or n_interrupted > 0
        or n_provider_contaminated > 0
        or n_prompt_budget_contaminated > 0
        or n_diagnostic_protocol > 0
        or n_configured_ineligible > 0
    ):
        if n_errors > 0:
            reasons.append(f"{n_errors} episode rows have status != 'ok'")
        if n_interrupted > 0:
            reasons.append(
                f"{n_interrupted} orphan log file(s) look like interrupted attempts"
            )
        if n_provider_contaminated > 0:
            reasons.append(
                f"{n_provider_contaminated} provider-contaminated episode(s) require retry"
            )
        if n_prompt_budget_contaminated > 0:
            reasons.append(
                f"{n_prompt_budget_contaminated} prompt-budget-contaminated episode(s) require retry"
            )
        if n_diagnostic_protocol > 0:
            reasons.append(
                f"{n_diagnostic_protocol} diagnostic protocol-v17 episode(s) are not formal measurements"
            )
        if n_configured_ineligible > 0:
            reasons.append(
                f"{n_configured_ineligible} configured episode(s) fail formal eligibility"
            )
        state = BATCH_STATE_DEGRADED
    else:
        reasons.append(
            "all configured cells filled with status=ok and no orphan residue"
        )
        state = BATCH_STATE_FINAL

    return {
        "batch_state": state,
        "reasons": reasons,
        **execution_counts,
        "n_orphan_interrupted_logs": n_interrupted,
        "n_provider_contaminated_episodes": n_provider_contaminated,
        "n_prompt_budget_contaminated_episodes": n_prompt_budget_contaminated,
        "n_model_output_failure_episodes": n_model_output_failures,
        "n_high_fallback_wait_episodes": n_high_fallback,
        "n_diagnostic_protocol_episodes": n_diagnostic_protocol,
        "n_configured_formal_ineligible_episodes": n_configured_ineligible,
    }


def _write_analysis(
    out_dir: Path,
    results: list[dict[str, Any]],
    coverage: dict[str, Any] | None = None,
    intersection_leaderboard: list[dict[str, Any]] | None = None,
    state: dict[str, Any] | None = None,
    pass_k_success: dict[str, Any] | None = None,
) -> None:
    """Emit ANALYSIS.md + per-model stats JSON from episode rows."""
    ok = [r for r in results if r.get("status") == "ok"]
    err = [r for r in results if r.get("status") != "ok" and not _row_is_quota_exhausted(r)]
    execution_counts = _execution_status_counts(results)
    clean_ok = [r for r in ok if _row_is_clean_for_resume(r, batch_root=out_dir)]
    dirty_ok = [r for r in ok if not _row_is_clean_for_resume(r, batch_root=out_dir)]
    by_model: dict[str, list[float]] = defaultdict(list)
    by_model_family: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    tool_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "n_episodes": 0,
            "n_tool_calls": 0.0,
            "n_wait": 0.0,
            "llm_ok": 0.0,
            "llm_fail": 0.0,
        }
    )
    for r in clean_ok:
        model = str(r.get("model") or r.get("agent_name", "")).replace("llm_agent/", "")
        by_model[model].append(float(r["score"]["total_score"]))
        fam = str(r.get("family", ""))
        by_model_family[model][fam].append(float(r["score"]["total_score"]))
    for r in ok:
        model = str(r.get("model") or r.get("agent_name", "")).replace("llm_agent/", "")
        traj = r.get("trajectory_summary") or {}
        st = tool_stats[model]
        st["n_episodes"] += 1
        st["n_tool_calls"] += float(traj.get("n_tool_calls", 0) or 0)
        st["n_wait"] += float(traj.get("n_wait_actions", 0) or 0)
        llm = traj.get("llm") or {}
        st["llm_ok"] += float(llm.get("llm_calls_ok", 0) or 0)
        st["llm_fail"] += float(llm.get("llm_calls_failed", 0) or 0)

    lines = [
        "# OPERATE LLM Eval Analysis",
        "",
        f"- Output directory: `{out_dir}`",
        f"- Episodes OK: {len(ok)} / {len(results)}",
        f"- Clean OK used for score means: {len(clean_ok)}",
        f"- Dirty OK excluded from score means: {len(dirty_ok)}",
        f"- Execution errors: {execution_counts['n_episodes_error']}",
        f"- Provider quota unavailable: {execution_counts['n_episodes_quota_unavailable']} (not task failures)",
        f"- Parked before execution: {execution_counts['n_episodes_quota_parked']}",
        f"- Trajectories: `{out_dir / 'trajectories'}/<model>/`",
        f"- Per-episode logs: `{out_dir / 'logs'}/<model>/`",
        f"- Plot directory: `{out_dir / 'plots'}`",
    ]
    if dirty_ok:
        lines.append(
            "- Score means exclude provider/parse/prompt-budget/fallback/"
            "protocol-contaminated `status=ok` rows (same filter as resume)."
        )
    if state is not None:
        lines.extend(
            [
                "",
                f"- **Batch state: `{state['batch_state']}`**",
            ]
        )
        for reason in state.get("reasons") or []:
            lines.append(f"  - {reason}")
    lines.extend(
        [
            "",
            "## Mean total_score by model",
            "",
            "| model | mean | n |",
            "|-------|------|---|",
        ]
    )
    for model, scores in sorted(by_model.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        mean = sum(scores) / len(scores)
        lines.append(f"| {model} | {mean:.2f} | {len(scores)} |")

    lines.extend(["", "## Mean score by model × family", ""])
    for model in sorted(by_model_family):
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| family | mean | n |")
        lines.append("|--------|------|---|")
        for fam, scores in sorted(by_model_family[model].items()):
            lines.append(f"| {fam} | {sum(scores) / len(scores):.2f} | {len(scores)} |")
        lines.append("")

    if tool_stats:
        lines.extend(["", "## Interaction stats (trajectory_summary)", ""])
        lines.append("")
        lines.append("| model | ep | tools/ep | wait/ep | llm_ok/ep | llm_fail/ep |")
        lines.append("|-------|-----|----------|---------|-----------|------------|")
        for model, st in sorted(tool_stats.items()):
            ep = max(int(st["n_episodes"]), 1)
            lines.append(
                f"| {model} | {ep} | {st['n_tool_calls'] / ep:.1f} | "
                f"{st['n_wait'] / ep:.1f} | {st['llm_ok'] / ep:.1f} | "
                f"{st['llm_fail'] / ep:.1f} |"
            )

    if err:
        lines.extend(["", "## Failures", ""])
        for r in err[:30]:
            lines.append(
                f"- `{r.get('scenario_id', r.get('scenario_slug'))}` "
                f"model={r.get('model')} seed={r.get('seed')}: {r.get('error')}"
            )
        if len(err) > 30:
            lines.append(f"- ... and {len(err) - 30} more")

    if coverage is not None:
        lines.extend(["", "## Coverage & comparability", ""])
        lines.append(
            f"- Expected episodes per model: **{coverage['expected_episodes_per_model']}**"
        )
        lines.append(f"- Expected total: **{coverage['expected_total']}**")
        lines.append(
            f"- Comparable intersection size (all configured models, status=ok): "
            f"**{coverage['comparable_intersection_size']}**"
        )
        lines.append(f"- Partial batch: **{coverage['is_partial_batch']}**")
        if state is not None:
            lines.append(f"- Batch state: **{state['batch_state']}**")
            if state.get("n_orphan_interrupted_logs"):
                lines.append(
                    f"- Orphan interrupted logs: "
                    f"**{state['n_orphan_interrupted_logs']}** "
                    "(see LOG_AUDIT.md → 'Sample interrupted-attempt orphan logs')"
                )
        if coverage.get("comparability_warning"):
            lines.append("")
            lines.append(f"> ⚠️ {coverage['comparability_warning']}")
        lines.extend(
            [
                "",
                "Formal-clean coverage (leaderboard admission; excludes suite blockers and contaminated ok rows):",
                "",
                "| model | realized | expected | coverage |",
                "|-------|----------|----------|----------|",
            ]
        )
        for model in coverage.get("configured_models", []):
            realized = coverage["per_model_realized"].get(model, 0)
            expected = coverage["expected_episodes_per_model"]
            cov = coverage["per_model_coverage"].get(model, 0.0)
            lines.append(f"| {model} | {realized} | {expected} | {cov:.2%} |")
        if coverage.get("per_model_execution_realized"):
            lines.extend(
                [
                    "",
                    "Execution-completed coverage (`status=ok`, may still be contaminated):",
                    "",
                    "| model | ok | expected | coverage |",
                    "|-------|----|----------|----------|",
                ]
            )
            for model in coverage.get("configured_models", []):
                realized = coverage["per_model_execution_realized"].get(model, 0)
                expected = coverage["expected_episodes_per_model"]
                cov = coverage.get("per_model_execution_coverage", {}).get(model, 0.0)
                lines.append(f"| {model} | {realized} | {expected} | {cov:.2%} |")

        if pass_k_success is not None:
            lines.extend(
                [
                    "",
                    "### Replicate execution completion",
                    "",
                    (
                        "- Cell success requires every configured `pass_id` for "
                        "the same `(model, scenario_slug, seed)` to finish with "
                        "`status=ok`. This is execution-completed, not formal-clean."
                    ),
                    (
                        "- Fraction of planned cells with all replicates completed: "
                        f"**{pass_k_success['overall_success_probability']:.2%}**"
                    ),
                    "",
                    "| model | completed cells | planned cells | completion fraction | missing attempts | execution errors | quota unavailable attempts |",
                    "|-------|-----------------|---------------|---------------------|------------------|------------------|----------------------------|",
                ]
            )
            for model in pass_k_success.get("configured_models", []):
                row = pass_k_success["per_model"].get(model, {})
                lines.append(
                    f"| {model} | {row.get('successful_cells', 0)} | "
                    f"{row.get('expected_cells', 0)} | "
                    f"{float(row.get('success_probability', 0.0)):.2%} | "
                    f"{row.get('missing_pass_units', 0)} | "
                    f"{row.get('error_pass_units', 0)} | "
                    f"{row.get('unavailable_pass_units', 0)} |"
                )

        if intersection_leaderboard:
            lines.extend(
                [
                    "",
                    "### Intersection leaderboard (fair, like-for-like)",
                    "",
                    "Restricted to `(scenario_slug, seed)` pairs every configured model completed.",
                    "",
                    "| model | mean | ci_lo | ci_hi | n |",
                    "|-------|------|-------|-------|---|",
                ]
            )
            for row in intersection_leaderboard:
                lines.append(
                    f"| {row['agent_id']} | {row['mean']:.2f} | "
                    f"{row['ci_lo']:.2f} | {row['ci_hi']:.2f} | {row['n_episodes']} |"
                )

    (out_dir / "ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats_export = {
        **execution_counts,
        "by_model": {
            m: {"mean": sum(s) / len(s), "n": len(s)} for m, s in by_model.items()
        },
        "n_ok": len(ok),
        "n_clean_ok": len(clean_ok),
        "n_dirty_ok": len(dirty_ok),
        "tool_stats": dict(tool_stats),
    }
    (out_dir / "stats_by_model.json").write_text(
        json.dumps(stats_export, indent=2), encoding="utf-8"
    )

    # P0-3: non-blocking discrimination report. Surfaces degenerate dimensions
    # (e.g. stakeholder_equity, safety_violation) so reviewers see which dims
    # carry no model signal. Never fails the batch on the descriptive pass.
    try:
        disc = build_discrimination_report(results)
        (out_dir / "discrimination_report.json").write_text(
            json.dumps(disc, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary = disc.get("summary", {}) if isinstance(disc, dict) else {}
        if int(summary.get("n_degenerate_dimensions", 0)) > 0:
            names = summary.get("degenerate_dimensions", [])
            print(
                f"WARNING: {summary['n_degenerate_dimensions']} degenerate "
                f"dimension(s): {names}",
                file=sys.stderr,
            )
    except Exception as exc:  # non-blocking
        (out_dir / "discrimination_report.json").write_text(
            json.dumps({"error": str(exc)}, indent=2), encoding="utf-8"
        )


def _resolve_patterns(slice_name: str, custom_scenarios: list[str] | None) -> list[str]:
    if slice_name == "custom":
        if not custom_scenarios:
            raise ValueError("--scenarios required for custom slice")
        return list(custom_scenarios)
    if slice_name in DYNAMIC_SCENARIO_SLICES:
        return _release_suite_scenarios(slice_name)
    if slice_name not in SCENARIO_SLICES:
        raise ValueError(f"unknown scenario slice: {slice_name}")
    return list(SCENARIO_SLICES[slice_name])


def _git_metadata() -> dict[str, Any]:
    def _run(cmd: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    sha = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--short"])
    available = sha is not None and status is not None
    return {
        "git_commit": sha,
        "git_metadata_available": available,
        "git_dirty": bool(status) if status is not None else None,
        "git_status_short": status.splitlines()[:200] if status else [],
    }


def _write_summary_csv(out_dir: Path, results: list[dict[str, Any]]) -> None:
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario_id",
                "family",
                "difficulty_mode",
                "difficulty_level",
                "model",
                "seed",
                "status",
                "scenario_signature",
                "temperature",
                "total_score",
                "raw_total",
                "prevented_loss",
                "n_control_calls",
                "outcome_changed",
                "foresight_score",
                "n_tool_calls",
                "llm_calls_failed",
                "trajectory_path",
                "episode_log_path",
                "error",
            ]
        )
        for r in results:
            score = r.get("score") or {}
            cf = r.get("counterfactual") or {}
            impact = r.get("decision_impact") or {}
            fs = r.get("foresight") or {}
            traj = r.get("trajectory_summary") or {}
            w.writerow(
                [
                    r.get("scenario_id"),
                    r.get("family"),
                    r.get("difficulty_mode"),
                    r.get("difficulty_level"),
                    r.get("model", r.get("agent_name")),
                    r.get("seed"),
                    r.get("status", "ok"),
                    r.get("scenario_signature"),
                    r.get("temperature"),
                    score.get("total_score"),
                    score.get("raw_total"),
                    cf.get("prevented_loss"),
                    impact.get("n_control_calls"),
                    impact.get("outcome_changed"),
                    fs.get("foresight_score"),
                    traj.get("n_tool_calls"),
                    (traj.get("llm") or {}).get("llm_calls_failed"),
                    traj.get("trajectory_path"),
                    r.get("episode_log_path"),
                    r.get("error"),
                ]
            )


_DOMAIN_BY_BACKEND: dict[str, str] = {
    "pglib_uc_synthetic": "power_grid",
    "grid2op": "power_grid",
    "cigre_distribution": "power_grid",
    "pandapower_acopf": "power_grid",
    "opendss_ieee13": "power_grid",
    "opendss_fresh_feeders": "power_grid",
    "egret_acopf": "power_grid",
    "pyvrp_cvrp": "logistics",
    "pyvrp_vrptw": "logistics",
    "jsplib_job_shop": "logistics",
    "orgym_invmgmt": "logistics",
    "mock_sumo": "traffic",
    "sumo": "traffic",
    "pandapower_lv": "microgrid",
    "pymgrid_economic_dispatch": "microgrid",
}


def _domain_for_result(row: dict[str, Any]) -> str:
    """Resolve the released domain for an episode result row.

    Prefers an explicit ``domain`` field (logistics/traffic/microgrid rows
    carry it); power-grid rows omit it, so fall back to the canonical
    backend->domain map and finally to ``power_grid``.
    """
    explicit = row.get("domain")
    if explicit:
        return str(explicit)
    backend = str(row.get("backend_kind") or "")
    return _DOMAIN_BY_BACKEND.get(backend, "power_grid")


def _eligibility(row: dict[str, Any] | None = None) -> dict[str, Any]:
    """Use only hash-bound per-episode eligibility in current evaluations."""
    protocol = (row or {}).get("evaluation_protocol") or {}
    if str(protocol.get("version") or "").startswith("2."):
        bound = (row or {}).get("suite_eligibility")
        bound_hash = str((row or {}).get("suite_eligibility_sha256") or "")
        if (
            isinstance(bound, dict)
            and bound_hash
            and _canonical_json_sha256(bound) == bound_hash
        ):
            return bound
        return {
            "suite_blocked": True,
            "reason": {"code": "suite_eligibility_unbound"},
        }
    return {
        "suite_blocked": True,
        "reason": {"code": "legacy_unbound_suite_unsupported"},
    }


def _cell_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("family", "")),
        str(row.get("difficulty_mode", "")),
        str(row.get("difficulty_level", "")),
    )


def _is_discriminative(row: dict[str, Any]) -> bool:
    """True iff the cell is NOT in any exclusion set (cell-level filter).

    Excludes individual cells in diagnostic_cells / uninformative_cells /
    wait_dominant_cells. Does NOT exclude by wait_dominant_families — that
    block is a diagnostic flag only, so a family with a mix of good and
    wait-dominated cells keeps its good cells.
    """
    le = _eligibility(row)
    if bool(le.get("suite_blocked")):
        return False
    cell = _cell_key(row)
    for key in ("diagnostic_cells", "uninformative_cells", "wait_dominant_cells"):
        if cell in {
            (_c["family"], _c["difficulty_mode"], _c["difficulty_level"])
            for _c in le.get(key, [])
        }:
            return False
    return True


def _score_for_leaderboard_view(row: dict[str, Any], view_name: str) -> float | None:
    score = row.get("score") or {}
    if view_name == "fixed_all_dimensions":
        return float(score.get("total_score", 0.0) or 0.0)
    if view_name == "discriminative_core":
        from evaluation.scorer import discriminative_core_total

        dimensions = score.get("dimensions") or []
        result = discriminative_core_total(
            dimensions,
            task_completion=task_completion_for_row(row),
            difficulty_level=str(row.get("difficulty_level", "basic")),
        )
        return float(result["total_score"])
    score_views = score.get("score_views") or {}
    if view_name not in score_views:
        return None
    view = score_views.get(view_name) or {}
    return float(view.get("total_score", 0.0) or 0.0)


def task_completion_for_row(row: dict[str, Any]) -> float:
    """Read the explicit domain task contract; missing contracts fail closed."""
    if row.get("status") != "ok":
        return 0.0
    completion = row.get("task_completion")
    if not isinstance(completion, dict):
        return 0.0
    if completion.get("applicable") is not True:
        return 0.0
    if not str(completion.get("contract") or ""):
        return 0.0
    evidence = completion.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        return 0.0
    for dimension in (row.get("score") or {}).get("dimensions") or []:
        if (
            dimension.get("name") == "system_survival"
            and dimension.get("floor_violation") is True
        ):
            return 0.0
    return 1.0 if completion.get("completed") is True else 0.0


def _strict_task_completion_for_row(row: dict[str, Any]) -> float:
    completion = row.get("task_completion")
    if not isinstance(completion, dict):
        raise PrimaryLeaderboardContractError(
            "formal row is missing task_completion contract"
        )
    if completion.get("applicable") is not True:
        raise PrimaryLeaderboardContractError(
            "formal task_completion must be applicable"
        )
    if not str(completion.get("contract") or ""):
        raise PrimaryLeaderboardContractError(
            "formal task_completion contract is missing"
        )
    evidence = completion.get("evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise PrimaryLeaderboardContractError(
            "formal task_completion evidence is missing"
        )
    return task_completion_for_row(row)


def _primary_leaderboard_payload(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Adapt episode rows once, then use the canonical primary helper."""

    prepared: list[dict[str, Any]] = []
    for row in rows:
        if "discriminative_core_score" in row and "task_completion_raw" in row:
            protocol = row.get("evaluation_protocol") or {}
            if (
                row.get("formal_score_eligible") is not True
                or str(row.get("scoring_version") or "") != SCORING_VERSION
                or str(protocol.get("version") or "") != EVALUATION_PROTOCOL_VERSION
                or str(protocol.get("implementation_fingerprint") or "")
                != EVALUATION_IMPLEMENTATION_FINGERPRINT
            ):
                raise PrimaryLeaderboardContractError(
                    "precomputed formal score is stale or not evidence-eligible"
                )
            from evaluation.scorer import discriminative_core_total

            task_completion = _strict_task_completion_for_row(row)
            score_contract = discriminative_core_total(
                (row.get("score") or {}).get("dimensions") or [],
                task_completion=task_completion,
                difficulty_level=str(row.get("difficulty_level", "basic")),
            )
            if score_contract["formal_score_eligible"] is not True:
                raise PrimaryLeaderboardContractError(
                    "precomputed formal score is missing five-group evidence"
                )
            try:
                precomputed_score = float(row["discriminative_core_score"])
                precomputed_completion = float(row["task_completion_raw"])
            except (TypeError, ValueError) as exc:
                raise PrimaryLeaderboardContractError(
                    "precomputed formal score has invalid units"
                ) from exc
            if not math.isclose(
                precomputed_score,
                float(score_contract["total_score"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ) or not math.isclose(
                precomputed_completion,
                task_completion,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise PrimaryLeaderboardContractError(
                    "precomputed formal score mismatch"
                )
            prepared.append(
                {
                    **row,
                    "discriminative_core_score": score_contract["total_score"],
                    "task_completion_raw": task_completion,
                }
            )
            continue
        from evaluation.scorer import discriminative_core_total

        task_completion = _strict_task_completion_for_row(row)
        score_contract = discriminative_core_total(
            (row.get("score") or {}).get("dimensions") or [],
            task_completion=task_completion,
            difficulty_level=str(row.get("difficulty_level", "basic")),
        )
        if score_contract["formal_score_eligible"] is not True:
            raise PrimaryLeaderboardContractError(
                "formal five-group evidence is incomplete: "
                + ", ".join(score_contract["missing_groups"])
            )
        prepared.append(
            {
                "model": row.get("model", row.get("agent_name")),
                "domain": row.get("domain"),
                "backend_kind": row.get("backend_kind"),
                "source_denominator_key": row.get("source_denominator_key"),
                "case_ledger": row.get("case_ledger"),
                "discriminative_core_score": score_contract["total_score"],
                "task_completion_raw": task_completion,
                "formal_score_eligible": True,
                "scoring_version": SCORING_VERSION,
                "evaluation_protocol": {
                    "version": EVALUATION_PROTOCOL_VERSION,
                    "implementation_fingerprint": (
                        EVALUATION_IMPLEMENTATION_FINGERPRINT
                    ),
                },
            }
        )
    return infer_primary_leaderboard(prepared)


def _leaderboard_from_rows(
    rows: list[dict[str, Any]], view_name: str
) -> list[dict[str, Any]]:
    per_model: dict[str, list[float]] = {}
    per_fatal: dict[str, list[bool]] = {}
    for row in rows:
        if not _formal_row_eligibility(row)[0]:
            continue
        model = str(row.get("model", row.get("agent_name")))
        value = _score_for_leaderboard_view(row, view_name)
        if value is None:
            continue
        per_model.setdefault(model, []).append(value)
        per_fatal.setdefault(model, []).append(
            bool(
                (row.get("ground_truth_summary") or {}).get("chose_fatal_option", False)
            )
        )
    return [
        row.to_dict() for row in build_leaderboard(per_model, fatal_flags=per_fatal)
    ]


def _leaderboard_for_view(
    results: list[dict[str, Any]], view_name: str
) -> list[dict[str, Any]]:
    return _leaderboard_from_rows(results, view_name)


def _leaderboard_by_domain(
    results: list[dict[str, Any]], view_name: str = "fixed_all_dimensions"
) -> dict[str, list[dict[str, Any]]]:
    """Per-domain leaderboards so a heavily-populated domain (e.g. logistics)
    does not dilute cross-agent gaps in the aggregate headline view."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault(_domain_for_result(row), []).append(row)
    out: dict[str, list[dict[str, Any]]] = {}
    for domain in sorted(grouped):
        board = _leaderboard_from_rows(grouped[domain], view_name)
        if board:
            out[domain] = board
    return out


def _scenario_aligned_scores_by_model(
    rows: list[dict[str, Any]], *, view_name: str
) -> dict[str, list[float]]:
    """Per-model score lists, aligned by ``scenario_signature`` so position i
    means "the same scenario run" for every model.

    ``holm_pairwise_ci`` computes a PAIRED bootstrap (``ta[i] - tb[i]``).
    ``episodes.jsonl`` is written in concurrent-completion order, not
    scenario order, so building per-model lists by simple row order (as a
    naive groupby would) silently pairs unrelated episodes whenever two
    models complete scenarios in different sequences — collapsing real
    performance gaps into noise. Keying by ``scenario_signature`` (which
    already encodes scenario + seed, see ``_scenario_signature_for_run``)
    and restricting to the intersection every model shares fixes this.
    """
    by_model_signature = _scenario_score_repeats_by_model(rows, view_name=view_name)
    means_by_model_signature = {
        model: {
            signature: sum(values) / len(values)
            for signature, values in signatures.items()
        }
        for model, signatures in by_model_signature.items()
    }
    if len(means_by_model_signature) < 2:
        return {
            model: [signatures[sig] for sig in sorted(signatures)]
            for model, signatures in means_by_model_signature.items()
        }
    common_sigs = sorted(
        set.intersection(
            *(set(signatures) for signatures in means_by_model_signature.values())
        )
    )
    if not common_sigs:
        return {}
    return {
        model: [signatures[sig] for sig in common_sigs]
        for model, signatures in means_by_model_signature.items()
    }


def _scenario_score_repeats_by_model(
    rows: list[dict[str, Any]], *, view_name: str
) -> dict[str, dict[str, list[float]]]:
    """Collect one score per repeat identity for each model/scenario cell."""
    grouped, _ = _scenario_score_repeats_and_duplicates_by_model(
        rows, view_name=view_name
    )
    return grouped


def _scenario_score_repeats_and_duplicates_by_model(
    rows: list[dict[str, Any]], *, view_name: str
) -> tuple[dict[str, dict[str, list[float]]], dict[str, int]]:
    by_model_signature_pass: dict[str, dict[str, dict[str, float]]] = {}
    duplicates_by_model: dict[str, int] = {}
    for r in rows:
        if not _formal_row_eligibility(r)[0]:
            continue
        m = str(r.get("model", r.get("agent_name")))
        sig = str(r.get("scenario_signature") or "")
        v = _score_for_leaderboard_view(r, view_name)
        if v is not None and sig:
            raw_pass_id = str(r.get("pass_id") or "").strip()
            if raw_pass_id:
                pass_identity = f"pass_id:{raw_pass_id}"
            elif r.get("pass_index") is not None:
                pass_identity = f"pass_index:{r['pass_index']}"
            else:
                pass_identity = "implicit_single_pass"
            by_pass = by_model_signature_pass.setdefault(m, {}).setdefault(sig, {})
            if pass_identity in by_pass:
                duplicates_by_model[m] = duplicates_by_model.get(m, 0) + 1
                if by_pass[pass_identity] != v:
                    raise ValueError(
                        f"conflicting_duplicate_pairwise_pass:{m}:{sig}:{pass_identity}"
                    )
                continue
            by_pass[pass_identity] = v
    grouped = {
        model: {
            signature: [by_pass[key] for key in sorted(by_pass)]
            for signature, by_pass in signatures.items()
        }
        for model, signatures in by_model_signature_pass.items()
    }
    return grouped, duplicates_by_model


def _scenario_repeat_diagnostics_by_model(
    rows: list[dict[str, Any]], *, view_name: str
) -> dict[str, dict[str, int | float]]:
    """Describe repeat coverage/noise without treating repeats as cells."""
    grouped, duplicates_by_model = _scenario_score_repeats_and_duplicates_by_model(
        rows, view_name=view_name
    )
    diagnostics: dict[str, dict[str, int | float]] = {}
    for model, signatures in sorted(grouped.items()):
        repeated = [values for values in signatures.values() if len(values) > 1]
        variances = []
        for values in repeated:
            mean = sum(values) / len(values)
            variances.append(sum((value - mean) ** 2 for value in values) / len(values))
        diagnostics[model] = {
            "n_episode_scores": sum(len(values) for values in signatures.values()),
            "n_duplicate_episode_rows_ignored": duplicates_by_model.get(model, 0),
            "n_scenario_signatures": len(signatures),
            "n_repeated_scenario_signatures": len(repeated),
            "max_repeats_per_scenario": max(
                (len(values) for values in signatures.values()), default=0
            ),
            "mean_within_repeated_scenario_variance": (
                sum(variances) / len(variances) if variances else 0.0
            ),
        }
    return diagnostics


def _write_leaderboard_json(
    out_dir: Path,
    results: list[dict[str, Any]],
    coverage: dict[str, Any] | None = None,
    intersection_leaderboard: list[dict[str, Any]] | None = None,
    state: dict[str, Any] | None = None,
    pass_k_success: dict[str, Any] | None = None,
    formal: bool = False,
    required_suite_hash: str | None = None,
    required_implementation_tree_sha256: str | None = None,
    required_interaction_mode: str | None = None,
    batch_root: Path | None = None,
) -> list[dict[str, Any]]:
    leaderboard_views = {
        "fixed_all_dimensions": _leaderboard_for_view(results, "fixed_all_dimensions"),
    }
    adaptive_leaderboard = _leaderboard_for_view(results, "adaptive_applicable")
    if adaptive_leaderboard:
        leaderboard_views["adaptive_applicable"] = adaptive_leaderboard
    # discriminative_core: exclude diagnostic + wait-dominated cells, and
    # score under the curated DISCRIMINATIVE_CORE_DIMENSIONS weighting
    # (drops redundant trust dims, folds in objective task_completion).
    disc_results = [r for r in results if _is_discriminative(r)]
    disc_board = _leaderboard_for_view(disc_results, "discriminative_core")
    if disc_board:
        leaderboard_views["discriminative_core"] = disc_board
    # Holm pairwise over the discriminative set.
    from evaluation.statistical import holm_pairwise_ci

    disc_per_model = _scenario_aligned_scores_by_model(
        disc_results, view_name="discriminative_core"
    )
    holm = holm_pairwise_ci(disc_per_model, seed=0) if len(disc_per_model) >= 2 else []
    # Provenance: how many episodes excluded, by reason. Each excluded row is
    # counted once, bucketed by priority diagnostic > uninformative >
    # wait_dominated, so the buckets are disjoint and sum to the total excluded.
    le = _eligibility(results[0] if results else None)
    diag_set = {
        (_c["family"], _c["difficulty_mode"], _c["difficulty_level"])
        for _c in le.get("diagnostic_cells", [])
    }
    uninf_set = {
        (_c["family"], _c["difficulty_mode"], _c["difficulty_level"])
        for _c in le.get("uninformative_cells", [])
    }
    wd_set = {
        (_c["family"], _c["difficulty_mode"], _c["difficulty_level"])
        for _c in le.get("wait_dominant_cells", [])
    }
    excluded = {"diagnostic": 0, "uninformative": 0, "wait_dominated": 0}
    for r in results:
        if _is_discriminative(r):
            continue
        cell = _cell_key(r)
        if cell in diag_set:
            excluded["diagnostic"] += 1
        elif cell in uninf_set:
            excluded["uninformative"] += 1
        elif cell in wd_set:
            excluded["wait_dominated"] += 1
    diagnostic_flat = (
        leaderboard_views.get("discriminative_core")
        or leaderboard_views["fixed_all_dimensions"]
    )
    payload: dict[str, Any] = {
        "diagnostic_flat_leaderboard": diagnostic_flat,
        "diagnostic_leaderboard_views": leaderboard_views,
        "n_episodes_total": len(results),
        "n_episodes_ok": sum(1 for r in results if r.get("status") == "ok"),
        "n_discriminative_episodes": len(disc_results),
        "excluded_cells": excluded,
        "diagnostic_flat_holm_pairwise": holm,
        "diagnostic_flat_holm_pairwise_repeat_diagnostics": (
            _scenario_repeat_diagnostics_by_model(
                disc_results, view_name="discriminative_core"
            )
        ),
    }
    leaderboard_by_domain = _leaderboard_by_domain(results, "fixed_all_dimensions")
    if leaderboard_by_domain:
        payload["diagnostic_leaderboard_by_domain"] = leaderboard_by_domain
    primary_leaderboard: list[dict[str, Any]] = []
    if formal:
        formal_blockers: list[str] = []
        if coverage is None:
            formal_blockers.append("formal_coverage_missing")
        elif coverage.get("is_partial_batch") is not False:
            formal_blockers.append("formal_coverage_incomplete")
        if state is None or state.get("batch_state") != BATCH_STATE_FINAL:
            formal_blockers.append("formal_batch_not_final")

        configured_models = set((coverage or {}).get("configured_models") or [])
        comparable_pairs = {
            (str(pair[0]), int(pair[1]))
            for pair in (coverage or {}).get("_comparable_pairs") or []
            if isinstance(pair, list) and len(pair) == 2
        }
        if not formal_blockers and (not configured_models or not comparable_pairs):
            formal_blockers.append("formal_configured_scope_missing")

        if formal_blockers:
            payload["formal_primary_blockers"] = sorted(set(formal_blockers))
        else:
            formal_rows = []
            formal_exclusions = []
            configured_failures = []
            pass_k = int((coverage or {}).get("pass_k") or 1)
            pass_ids = (coverage or {}).get("_comparable_pass_ids") or {}
            for row in results:
                eligible, eligibility_reasons = _formal_row_eligibility(
                    row,
                    required_suite_hash=required_suite_hash,
                    required_implementation_tree_sha256=(
                        required_implementation_tree_sha256
                    ),
                    required_interaction_mode=required_interaction_mode,
                    verify_artifact_bytes=True,
                    batch_root=batch_root,
                )
                reasons = list(eligibility_reasons)
                in_configured_scope = _row_is_in_configured_scope(row, coverage)
                if eligible:
                    model = _model_label(row)
                    slug = str(row.get("scenario_slug") or "")
                    seed = row.get("seed")
                    pair = (slug, int(seed)) if slug and seed is not None else None
                    if model not in configured_models:
                        reasons.append("model_outside_configured_scope")
                    elif pair not in comparable_pairs:
                        reasons.append("scenario_seed_outside_configured_scope")
                    elif pass_k > 1:
                        allowed = set(
                            (pass_ids.get(model) or {}).get(f"{pair[0]}||{pair[1]}", [])
                        )
                        if str(row.get("pass_id") or "") not in allowed:
                            reasons.append("pass_id_outside_configured_scope")
                if eligibility_reasons and in_configured_scope:
                    configured_failures.append(
                        {
                            "scenario_id": row.get("scenario_id"),
                            "scenario_slug": row.get("scenario_slug"),
                            "seed": row.get("seed"),
                            "pass_id": row.get("pass_id"),
                            "model": row.get("model", row.get("agent_name")),
                            "reasons": eligibility_reasons,
                        }
                    )
                if reasons:
                    formal_exclusions.append(
                        {
                            "scenario_id": row.get("scenario_id"),
                            "model": row.get("model", row.get("agent_name")),
                            "reasons": reasons,
                        }
                    )
                else:
                    formal_rows.append(row)
            payload["formal_primary_exclusions"] = formal_exclusions
            if configured_failures:
                payload["formal_primary_blockers"] = [
                    "formal_configured_episode_ineligible"
                ]
                payload["formal_configured_episode_failures"] = configured_failures
            else:
                try:
                    primary = _primary_leaderboard_payload(formal_rows)
                except PrimaryLeaderboardContractError as exc:
                    # Incomplete five-group evidence (and other primary
                    # contract failures) must not abort leaderboard.json
                    # after a paid run. Record a blocker the same way
                    # coverage / configured-episode failures do.
                    payload["formal_primary_blockers"] = [
                        "formal_primary_contract_error"
                    ]
                    payload["formal_primary_contract_error"] = str(exc)
                else:
                    primary_leaderboard = list(primary["leaderboard"])
                    payload.update(
                        {
                            "scoring_version": primary["scoring_version"],
                            "primary_leaderboard_formula_version": primary[
                                "primary_leaderboard_formula_version"
                            ],
                            "primary_leaderboard": primary_leaderboard,
                            "primary_inference_version": primary.get(
                                "primary_inference_version"
                            ),
                            "primary_inference_n_physical_clusters": primary.get(
                                "primary_inference_n_physical_clusters"
                            ),
                            "primary_pairwise": primary.get("primary_pairwise", []),
                        }
                    )
    if coverage is not None:
        # Drop the underscored internal pair list before serialising.
        coverage_public = {k: v for k, v in coverage.items() if not k.startswith("_")}
        payload["coverage"] = coverage_public
        payload["expected_total"] = coverage["expected_total"]
        payload["comparable_intersection_size"] = coverage[
            "comparable_intersection_size"
        ]
        payload["is_partial_batch"] = coverage["is_partial_batch"]
        if coverage.get("comparability_warning"):
            payload["comparability_warning"] = coverage["comparability_warning"]
    if intersection_leaderboard is not None:
        payload["intersection_leaderboard"] = intersection_leaderboard
    if state is not None:
        payload["batch_state"] = state["batch_state"]
        payload["batch_state_reasons"] = state["reasons"]
        payload["n_orphan_interrupted_logs"] = state["n_orphan_interrupted_logs"]
    if pass_k_success is not None:
        payload["pass_k_success"] = pass_k_success
    with open(out_dir / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return primary_leaderboard if formal else diagnostic_flat


def _normalized_scenario_seed_pairs(value: Any) -> list[tuple[str, int]] | None:
    if not isinstance(value, list):
        return None
    normalized: list[tuple[str, int]] = []
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return None
        slug = str(pair[0] or "").strip()
        if not slug or isinstance(pair[1], bool):
            return None
        try:
            seed = int(pair[1])
        except (TypeError, ValueError):
            return None
        normalized.append((slug, seed))
    return normalized


def _formal_leaderboard_eligibility(
    meta: dict[str, Any],
    coverage: dict[str, Any],
    pass_k_success: dict[str, Any],
    state: dict[str, Any],
    leaderboard_payload: dict[str, Any],
) -> dict[str, Any]:
    """Certify that a finalized batch satisfies the formal publication contract."""
    blockers: list[str] = []
    blockers.extend(
        str(reason) for reason in meta.get("agent_treatment_binding_reasons") or []
    )
    readiness = meta.get("suite_eligibility") or {}
    formal_config = {
        "scenario_slice": meta.get("scenario_slice"),
        "formal_manifest_bound": bool(meta.get("formal_manifest")),
        "models": meta.get("models"),
        "pass_k": meta.get("pass_k"),
        "max_workers": meta.get("max_workers_requested"),
        "temperature": meta.get("temperature"),
        "prompt_mode": meta.get("prompt_mode"),
        "interaction_mode": meta.get("interaction_mode"),
        "seed_mode": meta.get("seed_mode"),
        "scheduler_mode": meta.get("scheduler_mode"),
        "save_trajectories": meta.get("save_trajectories"),
        "finalize": meta.get("finalize_enabled"),
        "allow_blocked_suite": False,
        "diagnostic_only": meta.get("diagnostic_only"),
        "git_metadata_available": meta.get("git_metadata_available"),
        "git_dirty": meta.get("git_dirty"),
        "model_context_window_tokens_by_model": meta.get(
            "model_context_window_tokens_by_model"
        ),
        "model_max_output_tokens_by_model": meta.get(
            "model_max_output_tokens_by_model"
        ),
        "max_tokens": meta.get("max_tokens"),
        "protocol_repair_max_tokens": meta.get("protocol_repair_max_tokens"),
        "persistent_history_max_messages": meta.get("persistent_history_max_messages"),
        "persistent_context_max_chars": meta.get("persistent_context_max_chars"),
        "persistent_memory_max_items": meta.get("persistent_memory_max_items"),
        "provider_timeout_s": meta.get("provider_timeout_s"),
        "provider_rpm_limit": meta.get("provider_rpm_limit"),
        "provider_rpd_limit": meta.get("provider_rpd_limit"),
        "provider_rate_limit_scope": meta.get("provider_rate_limit_scope"),
        "tool_choice": meta.get("tool_choice"),
        "stream_chat_completions": meta.get("stream_chat_completions"),
    }
    if meta.get("formal_run") is True:
        blockers.extend(
            _validate_protocol21_formal_run(
                formal_config,
                readiness,
                suite_manifest_sha256=meta.get("suite_manifest_sha256"),
            )
        )
    else:
        blockers.append("formal_run_required")
    if meta.get("formal_evaluation_ready") is not True:
        blockers.append("formal_readiness_not_green")
    if readiness.get("suite_blocked") is not False:
        blockers.append("formal_suite_blocked")
    if not meta.get("git_commit") or meta.get("git_commit_end") != meta.get(
        "git_commit"
    ):
        blockers.append("formal_git_commit_changed")
    if meta.get("implementation_tree_stable") is not True:
        blockers.append("formal_implementation_tree_unstable")
    if meta.get("formal_runtime_binding_stable") is not True:
        blockers.append("formal_runtime_binding_unstable")
    models = [str(model) for model in meta.get("models") or []]
    pass_k = int(meta.get("pass_k") or 0)
    requested_workers = int(meta.get("max_workers_requested") or 0)
    effective_workers = int(meta.get("max_workers_effective") or 0)
    if effective_workers != requested_workers:
        blockers.append("formal_effective_workers_mismatch")

    n_scenarios = int(meta.get("n_scenarios") or 0)
    scenario_seed_pairs = _normalized_scenario_seed_pairs(
        meta.get("scenario_seed_pairs")
    )
    if (
        n_scenarios <= 0
        or scenario_seed_pairs is None
        or len(scenario_seed_pairs) != n_scenarios
        or len(set(scenario_seed_pairs)) != n_scenarios
        or len({slug for slug, _seed in scenario_seed_pairs}) != n_scenarios
    ):
        blockers.append("formal_scenario_seed_scope_invalid")
        expected_pairs: set[tuple[str, int]] = set()
    else:
        expected_pairs = set(scenario_seed_pairs)
    expected_total = n_scenarios * len(models) * pass_k
    expected_per_model = n_scenarios * pass_k
    per_model_coverage = coverage.get("per_model_coverage") or {}
    coverage_models = [str(model) for model in coverage.get("configured_models") or []]
    if coverage_models != models:
        blockers.append("formal_coverage_model_scope_mismatch")
    if coverage.get("seed_mode") != "scenario":
        blockers.append("formal_coverage_seed_mode_mismatch")
    coverage_pairs = _normalized_scenario_seed_pairs(coverage.get("_comparable_pairs"))
    if coverage_pairs is None or set(coverage_pairs) != expected_pairs:
        blockers.append("formal_coverage_scenario_seed_scope_mismatch")
    if int(coverage.get("pass_k") or 0) != pass_k:
        blockers.append("formal_coverage_pass_k_mismatch")
    if int(coverage.get("n_scenarios") or 0) != n_scenarios:
        blockers.append("formal_coverage_scenario_scope_mismatch")
    if int(coverage.get("expected_total") or 0) != expected_total:
        blockers.append("formal_coverage_expected_total_mismatch")
    if int(coverage.get("expected_episodes_per_model") or 0) != expected_per_model:
        blockers.append("formal_coverage_expected_per_model_mismatch")
    per_model_realized = coverage.get("per_model_realized") or {}
    if any(
        int(per_model_realized.get(model) or 0) != expected_per_model
        for model in models
    ):
        blockers.append("formal_coverage_realized_mismatch")
    if coverage.get("is_partial_batch") is not False or any(
        float(per_model_coverage.get(model, 0.0)) != 1.0 for model in models
    ):
        blockers.append("formal_coverage_incomplete")
    if int(coverage.get("comparable_intersection_size") or 0) != n_scenarios:
        blockers.append("formal_comparable_intersection_incomplete")
    if state.get("batch_state") != BATCH_STATE_FINAL:
        blockers.append("formal_batch_not_final")
    if int(state.get("n_orphan_interrupted_logs") or 0) != 0:
        blockers.append("formal_orphan_interrupted_logs")

    pass_models = [
        str(model) for model in pass_k_success.get("configured_models") or []
    ]
    expected_cells = n_scenarios * len(models)
    if pass_models != models:
        blockers.append("formal_pass_k_model_scope_mismatch")
    if pass_k_success.get("seed_mode") != "scenario":
        blockers.append("formal_pass_k_seed_mode_mismatch")
    if int(pass_k_success.get("n_scenarios") or 0) != n_scenarios:
        blockers.append("formal_pass_k_scenario_scope_mismatch")
    if int(pass_k_success.get("expected_cells_per_model") or 0) != n_scenarios:
        blockers.append("formal_pass_k_expected_per_model_mismatch")
    if int(pass_k_success.get("expected_total_cells") or 0) != expected_cells:
        blockers.append("formal_pass_k_expected_total_mismatch")
    if (
        int(pass_k_success.get("pass_k") or 0) != pass_k
        or int(pass_k_success.get("successful_total_cells") or 0) != expected_cells
        or float(pass_k_success.get("overall_success_probability") or 0.0) != 1.0
    ):
        blockers.append("formal_pass_k_incomplete")
    per_model_pass = pass_k_success.get("per_model") or {}
    if set(per_model_pass) != set(models) or any(
        int((per_model_pass.get(model) or {}).get("expected_cells") or 0) != n_scenarios
        or int((per_model_pass.get(model) or {}).get("successful_cells") or 0)
        != n_scenarios
        or int((per_model_pass.get(model) or {}).get("failed_cells") or 0) != 0
        or float((per_model_pass.get(model) or {}).get("success_probability") or 0.0)
        != 1.0
        or int((per_model_pass.get(model) or {}).get("missing_pass_units") or 0) != 0
        or int((per_model_pass.get(model) or {}).get("error_pass_units") or 0) != 0
        for model in models
    ):
        blockers.append("formal_pass_k_per_model_incomplete")

    for blocker in leaderboard_payload.get("formal_primary_blockers") or []:
        blockers.append(f"formal_primary_blocker:{blocker}")
    primary_leaderboard = leaderboard_payload.get("primary_leaderboard") or []
    if not primary_leaderboard:
        blockers.append("formal_primary_leaderboard_missing")
    elif sorted(str(row.get("model")) for row in primary_leaderboard) != sorted(models):
        blockers.append("formal_leaderboard_model_scope_mismatch")
    blockers = sorted(set(blockers))
    return {"eligible": not blockers, "blockers": blockers}


def _svg_escape(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_svg(path: Path, body: str, width: int = 1100, height: int = 700) -> None:
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        "<style>text{font-family:Arial,sans-serif;fill:#1f2937}.small{font-size:12px}.label{font-size:14px}.title{font-size:20px;font-weight:bold}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#e5e7eb;stroke-width:1}.bar{fill:#2563eb}.bar3{fill:#dc2626}.cell{stroke:#ffffff;stroke-width:1}</style>"
        f"{body}</svg>"
    )
    path.write_text(svg, encoding="utf-8")


def _plot_score_by_model(rows: list[dict[str, Any]], path: Path) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        grouped[str(r.get("model") or r.get("agent_name", "?"))].append(
            float((r.get("score") or {}).get("total_score", 0.0) or 0.0)
        )
    items = sorted(
        ((m, sum(v) / len(v), len(v)) for m, v in grouped.items()),
        key=lambda x: -x[1],
    )
    width, height = 1100, 700
    left, right, top, bottom = 100, 40, 80, 180
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max([v for _, v, _ in items], default=1.0)
    scale = plot_h / max(max_val, 1.0)
    body = [f'<text x="{left}" y="40" class="title">Mean total score by model</text>']
    for i in range(6):
        y = top + plot_h - (plot_h * i / 5)
        val = max_val * i / 5
        body.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>'
        )
        body.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="small">{val:.1f}</text>'
        )
    n = max(len(items), 1)
    bar_w = min(120, plot_w / n * 0.7)
    gap = plot_w / n
    for idx, (model, mean, count) in enumerate(items):
        x = left + idx * gap + (gap - bar_w) / 2
        h = mean * scale
        y = top + plot_h - h
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" class="bar"/>'
        )
        body.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="small">{mean:.1f}</text>'
        )
        body.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 20}" text-anchor="middle" class="small">n={count}</text>'
        )
        body.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 42}" text-anchor="middle" class="small" '
            f'transform="rotate(25 {x + bar_w / 2:.1f} {top + plot_h + 42})">{_svg_escape(model)}</text>'
        )
    body.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>'
    )
    body.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>'
    )
    _write_svg(path, "".join(body), width=width, height=height)


def _heat_color(value: float, min_v: float, max_v: float) -> str:
    ratio = 0.5 if max_v <= min_v else (value - min_v) / (max_v - min_v)
    r = int(245 - ratio * 160)
    g = int(245 - ratio * 80)
    b = int(255 - ratio * 210)
    return f"rgb({r},{g},{b})"


def _plot_score_by_family_model(rows: list[dict[str, Any]], path: Path) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    models = sorted({str(r.get("model") or r.get("agent_name", "?")) for r in ok})
    families = sorted({str(r.get("family") or "unknown") for r in ok})
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in ok:
        scores[
            (
                str(r.get("family") or "unknown"),
                str(r.get("model") or r.get("agent_name", "?")),
            )
        ].append(float((r.get("score") or {}).get("total_score", 0.0) or 0.0))
    means = {k: sum(v) / len(v) for k, v in scores.items()}
    vals = list(means.values()) or [0.0]
    min_v, max_v = min(vals), max(vals)
    width = 220 + max(1, len(models)) * 180
    height = 140 + max(1, len(families)) * 70
    left, top = 180, 80
    cell_w, cell_h = 160, 50
    body = [
        '<text x="40" y="40" class="title">Mean total score by family × model</text>'
    ]
    for j, model in enumerate(models):
        x = left + j * cell_w + cell_w / 2
        body.append(
            f'<text x="{x:.1f}" y="70" text-anchor="middle" class="small">{_svg_escape(model)}</text>'
        )
    for i, family in enumerate(families):
        y = top + i * cell_h + cell_h / 2 + 5
        body.append(
            f'<text x="{left - 10}" y="{y:.1f}" text-anchor="end" class="label">{_svg_escape(family)}</text>'
        )
        for j, model in enumerate(models):
            x = left + j * cell_w
            y0 = top + i * cell_h
            val = means.get((family, model))
            fill = "#f3f4f6" if val is None else _heat_color(val, min_v, max_v)
            label = "—" if val is None else f"{val:.1f}"
            body.append(
                f'<rect x="{x}" y="{y0}" width="{cell_w}" height="{cell_h}" fill="{fill}" class="cell"/>'
            )
            body.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{y0 + cell_h / 2 + 5:.1f}" text-anchor="middle" class="label">{label}</text>'
            )
    _write_svg(path, "".join(body), width=width, height=height)


def _plot_failures_by_model(rows: list[dict[str, Any]], path: Path) -> None:
    counts: Counter[str] = Counter()
    for r in rows:
        if r.get("status") != "ok":
            counts[str(r.get("model") or r.get("agent_name", "?"))] += 1
    items = sorted(counts.items())
    width, height = 1000, 600
    left, right, top, bottom = 100, 40, 80, 140
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_val = max([v for _, v in items], default=1)
    body = [f'<text x="{left}" y="40" class="title">Failure count by model</text>']
    if not items:
        body.append(
            f'<text x="{left}" y="{top + 40}" class="label">No failed episodes in this run.</text>'
        )
        _write_svg(path, "".join(body), width=width, height=height)
        return
    gap = plot_w / max(len(items), 1)
    bar_w = min(120, gap * 0.7)
    for i, (model, count) in enumerate(items):
        x = left + i * gap + (gap - bar_w) / 2
        h = plot_h * (count / max(max_val, 1))
        y = top + plot_h - h
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#dc2626"/>'
        )
        body.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="small">{count}</text>'
        )
        body.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 36}" text-anchor="middle" class="small" '
            f'transform="rotate(25 {x + bar_w / 2:.1f} {top + plot_h + 36})">{_svg_escape(model)}</text>'
        )
    body.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>'
    )
    _write_svg(path, "".join(body), width=width, height=height)


def _plot_tool_calls_vs_score(rows: list[dict[str, Any]], path: Path) -> None:
    ok = [r for r in rows if r.get("status") == "ok"]
    width, height = 1000, 650
    left, right, top, bottom = 100, 40, 80, 90
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [
        float((r.get("trajectory_summary") or {}).get("n_tool_calls", 0) or 0)
        for r in ok
    ]
    ys = [float((r.get("score") or {}).get("total_score", 0.0) or 0.0) for r in ok]
    max_x = max(xs, default=1.0)
    min_y = min(ys, default=0.0)
    max_y = max(ys, default=1.0)
    body = [f'<text x="{left}" y="40" class="title">Tool calls vs total score</text>']
    body.append(
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>'
    )
    body.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>'
    )
    for i in range(6):
        xv = max_x * i / 5
        x = left + (plot_w * i / 5)
        body.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" class="grid"/>'
        )
        body.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 20}" text-anchor="middle" class="small">{xv:.0f}</text>'
        )
    for i in range(6):
        yv = min_y + (max_y - min_y) * i / 5 if max_y > min_y else max_y
        y = top + plot_h - (plot_h * i / 5)
        body.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>'
        )
        body.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="small">{yv:.1f}</text>'
        )
    palette = ["#2563eb", "#059669", "#7c3aed", "#ea580c", "#dc2626", "#0891b2"]
    models = sorted({str(r.get("model") or r.get("agent_name", "?")) for r in ok})
    color_for = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    legend_y = top - 20
    for i, model in enumerate(models):
        lx = left + i * 180
        body.append(
            f'<circle cx="{lx}" cy="{legend_y}" r="5" fill="{color_for[model]}" />'
        )
        body.append(
            f'<text x="{lx + 12}" y="{legend_y + 4}" class="small">{_svg_escape(model)}</text>'
        )
    for r in ok:
        x_val = float((r.get("trajectory_summary") or {}).get("n_tool_calls", 0) or 0)
        y_val = float((r.get("score") or {}).get("total_score", 0.0) or 0.0)
        model = str(r.get("model") or r.get("agent_name", "?"))
        x = left + (0 if max_x <= 0 else plot_w * x_val / max_x)
        y = (
            top
            + plot_h
            - (0 if max_y == min_y else plot_h * (y_val - min_y) / (max_y - min_y))
        )
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color_for[model]}" opacity="0.75"/>'
        )
    _write_svg(path, "".join(body), width=width, height=height)


def _write_plots(out_dir: Path, results: list[dict[str, Any]]) -> list[str]:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_score_by_model(results, plots_dir / "score_by_model.svg")
    _plot_score_by_family_model(results, plots_dir / "score_by_family_model.svg")
    _plot_failures_by_model(results, plots_dir / "failures_by_model.svg")
    _plot_tool_calls_vs_score(results, plots_dir / "tool_calls_vs_score.svg")
    return [str(plots_dir / name) for name in PLOT_FILES]


def _scenario_identity_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize scenario identity uniqueness in finalized episode rows."""
    scenario_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in results:
        if row.get("status") != "ok":
            continue
        key = (str(row.get("scenario_slug")), str(row.get("scenario_signature")))
        scenario_rows.setdefault(key, row)

    slugs = sorted({str(row.get("scenario_slug")) for row in scenario_rows.values()})
    ids = sorted({str(row.get("scenario_id")) for row in scenario_rows.values()})
    signatures = sorted(
        {str(row.get("scenario_signature")) for row in scenario_rows.values()}
    )
    by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scenario_rows.values():
        by_signature[str(row.get("scenario_signature"))].append(row)

    duplicate_signature_groups: list[dict[str, Any]] = []
    for signature, rows_for_signature in sorted(by_signature.items()):
        if len(rows_for_signature) <= 1:
            continue
        duplicate_signature_groups.append(
            {
                "scenario_signature": signature,
                "count": len(rows_for_signature),
                "scenario_ids": sorted(
                    {str(row.get("scenario_id")) for row in rows_for_signature}
                ),
                "scenario_slugs": sorted(
                    {str(row.get("scenario_slug")) for row in rows_for_signature}
                ),
            }
        )

    return {
        "scenario_rows": len(scenario_rows),
        "unique_scenario_slugs": len(slugs),
        "unique_scenario_ids": len(ids),
        "unique_scenario_signatures": len(signatures),
        "duplicate_signature_excess_rows": sum(
            group["count"] - 1 for group in duplicate_signature_groups
        ),
        "duplicate_signature_groups": duplicate_signature_groups,
    }


def _portable_formal_result_paths(
    rows: list[dict[str, Any]], *, batch_root: Path
) -> list[dict[str, Any]]:
    """Serialize formal row artifact locators relative to their batch root."""

    root = batch_root.resolve()

    def relative(value: object, *, label: str) -> str:
        path = Path(str(value))
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"formal {label} path escapes batch root") from exc

    portable = deepcopy(rows)
    for row in portable:
        if row.get("episode_log_path"):
            row["episode_log_path"] = relative(
                row["episode_log_path"], label="episode log"
            )
        summary = row.get("trajectory_summary")
        if not isinstance(summary, dict):
            continue
        for field in ("trajectory_path", "evidence_path"):
            if summary.get(field):
                summary[field] = relative(summary[field], label=field)
        for name, binding in summary.items():
            if isinstance(binding, dict) and binding.get("path"):
                binding["path"] = relative(binding["path"], label=name)
    return portable


def _portable_formal_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove clone-specific repository prefixes from a formal manifest."""

    portable = canonicalize_repo_owned_paths(manifest, repo_root=REPO_ROOT)
    if not isinstance(portable, dict):  # pragma: no cover - defensive typing
        raise TypeError("formal manifest must remain an object")
    return portable


def _portable_formal_run_config(config: dict[str, Any]) -> dict[str, Any]:
    """Persist clone-portable paths for formal run resume and publication."""

    if config.get("formal_run") is not True:
        return config
    return _portable_formal_manifest(config)


def _finalize_outputs(
    out_dir: Path, results: list[dict[str, Any]], meta: dict[str, Any]
) -> dict[str, Any]:
    if bool(meta.get("formal_run")):
        # Fail closed across interrupted re-finalization: never leave a prior
        # eligible manifest visible while required reports are being rebuilt.
        (out_dir / "RUN_MANIFEST.json").write_text(
            json.dumps(
                {
                    "status": "finalization_in_progress",
                    "formal_run": True,
                    "leaderboard_eligible": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        leaderboard_path = out_dir / "leaderboard.json"
        if leaderboard_path.is_file():
            try:
                prior_leaderboard = json.loads(
                    leaderboard_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                prior_leaderboard = {}
            prior_leaderboard["leaderboard_eligible"] = False
            prior_leaderboard["leaderboard_eligibility"] = {
                "eligible": False,
                "blockers": ["formal_finalization_in_progress"],
            }
            leaderboard_path.write_text(
                json.dumps(prior_leaderboard, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    effective_results = _prefer_current_implementation_rows(
        _effective_episode_rows(results)
    )
    treatment_reasons = _formal_treatment_binding_reasons(meta, effective_results)
    selected_results = _select_rows_for_treatment(effective_results, meta)
    if len(selected_results) != len(effective_results):
        LOGGER.warning(
            "excluded %d rows outside the bound agent treatment",
            len(effective_results) - len(selected_results),
        )
    if bool(meta.get("formal_run")):
        meta["agent_treatment_binding_reasons"] = treatment_reasons
    results = selected_results
    if bool(meta.get("formal_run")):
        results = _portable_formal_result_paths(results, batch_root=out_dir)
    configured_models = list(meta.get("models") or [])
    configured_seeds = list(meta.get("seeds") or [])
    configured_pairs = meta.get("scenario_seed_pairs")
    n_scenarios = int(meta.get("n_scenarios") or 0)
    coverage = _coverage_summary(
        results,
        configured_models=configured_models,
        configured_seeds=configured_seeds,
        n_scenarios=n_scenarios,
        pass_k=int(meta.get("pass_k", 1) or 1),
        configured_pairs=configured_pairs,
    )
    intersection_leaderboard = _intersection_leaderboard(results, coverage)
    pass_k_success = _pass_k_success_summary(
        results,
        configured_models=configured_models,
        configured_seeds=configured_seeds,
        n_scenarios=n_scenarios,
        pass_k=int(meta.get("pass_k", 1) or 1),
        configured_pairs=configured_pairs,
    )
    if coverage.get("comparability_warning"):
        LOGGER.warning("comparability: %s", coverage["comparability_warning"])

    # Run the log audit BEFORE the leaderboard / analysis writers so the
    # batch_state computation can incorporate orphan-interrupted residue
    # alongside coverage and episode errors. Audit failures must NOT break
    # finalize: degrade gracefully, record the error in batch_state, and
    # let the rest of the pipeline still produce leaderboard / manifest.
    audit_error: str | None = None
    try:
        log_audit_report = audit_logs(out_dir)
        if bool(meta.get("formal_run")):
            log_audit_report = _portable_formal_manifest(log_audit_report)
        (out_dir / "LOG_AUDIT.json").write_text(
            json.dumps(log_audit_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        write_log_audit_markdown(log_audit_report, out_dir / "LOG_AUDIT.md")
    except Exception as exc:  # pragma: no cover - exercised by audit-fail test
        LOGGER.exception("log audit failed; continuing with degraded state")
        audit_error = f"{type(exc).__name__}: {exc}"
        log_audit_report = {
            "log_files_orphan_interrupted": 0,
            "log_files_orphan_empty": 0,
            "_audit_error": audit_error,
        }

    state = _batch_state(
        coverage=coverage,
        results=results,
        log_audit_report=log_audit_report,
        required_suite_hash=str(meta.get("suite_manifest_sha256") or "") or None,
        required_interaction_mode=(
            str(meta.get("interaction_mode") or "")
            if bool(meta.get("formal_run"))
            else None
        ),
    )
    if audit_error is not None:
        # Audit failure forces at least `degraded` so consumers don't
        # mistake it for a clean batch.
        if state["batch_state"] == BATCH_STATE_FINAL:
            state["batch_state"] = BATCH_STATE_DEGRADED
        state["reasons"].append(f"log audit failed: {audit_error}")
    if state["batch_state"] != BATCH_STATE_FINAL:
        LOGGER.warning(
            "batch_state=%s reasons=%s",
            state["batch_state"],
            state["reasons"],
        )

    _write_summary_csv(out_dir, results)
    leaderboard = _write_leaderboard_json(
        out_dir,
        results,
        coverage=coverage,
        intersection_leaderboard=intersection_leaderboard,
        state=state,
        pass_k_success=pass_k_success,
        formal=bool(meta.get("formal_run")),
        required_suite_hash=str(meta.get("suite_manifest_sha256") or "") or None,
        required_implementation_tree_sha256=str(
            meta.get("implementation_tree_sha256") or ""
        )
        or None,
        required_interaction_mode=(
            str(meta.get("interaction_mode") or "")
            if bool(meta.get("formal_run"))
            else None
        ),
        batch_root=out_dir if bool(meta.get("formal_run")) else None,
    )
    leaderboard_path = out_dir / "leaderboard.json"
    leaderboard_payload = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    leaderboard_eligibility = _formal_leaderboard_eligibility(
        meta,
        coverage,
        pass_k_success,
        state,
        leaderboard_payload,
    )
    leaderboard_payload["leaderboard_eligible"] = leaderboard_eligibility["eligible"]
    leaderboard_payload["leaderboard_eligibility"] = leaderboard_eligibility
    leaderboard_path.write_text(
        json.dumps(leaderboard_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_analysis(
        out_dir,
        results,
        coverage=coverage,
        intersection_leaderboard=intersection_leaderboard,
        state=state,
        pass_k_success=pass_k_success,
    )
    analysis_report = analyze_output_dir(out_dir, rows=results)
    (out_dir / "analysis_deep.json").write_text(
        json.dumps(analysis_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    decision_report = build_decision_impact_report(results)
    (out_dir / "decision_impact_report.json").write_text(
        json.dumps(decision_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_decision_impact_markdown(decision_report, out_dir / "DECISION_IMPACT.md")
    evidence_report = build_evidence_applicability_report(results)
    (out_dir / "evidence_applicability_report.json").write_text(
        json.dumps(evidence_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_evidence_applicability_markdown(
        evidence_report, out_dir / "EVIDENCE_APPLICABILITY.md"
    )
    tool_effect_report = build_tool_effect_report(results, batch_root=out_dir)
    (out_dir / "tool_effect_audit_report.json").write_text(
        json.dumps(tool_effect_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_tool_effect_markdown(tool_effect_report, out_dir / "TOOL_EFFECT_AUDIT.md")
    staleness_report = build_staleness_consumption_report(results, batch_root=out_dir)
    (out_dir / "staleness_consumption_report.json").write_text(
        json.dumps(staleness_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_staleness_consumption_markdown(
        staleness_report, out_dir / "STALENESS_CONSUMPTION.md"
    )
    failure_recipes_report = build_agent_failure_recipes_report(results, batch_root=out_dir)
    (out_dir / "agent_failure_recipes_report.json").write_text(
        json.dumps(failure_recipes_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_agent_failure_recipes_markdown(
        failure_recipes_report, out_dir / "AGENT_FAILURE_RECIPES.md"
    )
    plot_files = _write_plots(out_dir, results)

    scenario_manifest: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for r in results:
        if r.get("status") != "ok":
            continue
        key = (str(r.get("scenario_slug")), str(r.get("scenario_signature")))
        if key in seen:
            continue
        seen.add(key)
        scenario_manifest.append(
            {
                "scenario_slug": r.get("scenario_slug"),
                "scenario_id": r.get("scenario_id"),
                "scenario_signature": r.get("scenario_signature"),
                "family": r.get("family"),
                "difficulty_mode": r.get("difficulty_mode"),
                "difficulty_level": r.get("difficulty_level"),
                "backend_kind": r.get("backend_kind"),
            }
        )

    manifest_has_grid2op = bool(meta.get("has_grid2op", False)) or any(
        str(r.get("backend_kind", "")) == "grid2op" for r in results
    )
    published_episodes_path = out_dir / "formal_episodes.jsonl"
    if bool(meta.get("formal_run")):
        portable_rows = "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in results
        )
        _atomic_write_text(out_dir / "episodes.jsonl", portable_rows)
        _atomic_write_text(
            published_episodes_path,
            portable_rows,
        )
    else:
        published_episodes_path = out_dir / "episodes.jsonl"

    coverage_public = {k: v for k, v in coverage.items() if not k.startswith("_")}
    scenario_identity = _scenario_identity_report(results)
    manifest = {
        **meta,
        "has_grid2op": manifest_has_grid2op,
        "finalized_at_utc": datetime.now(UTC).isoformat(),
        "n_episodes_total": len(results),
        **_execution_status_counts(results),
        "expected_total": coverage["expected_total"],
        "realized_coverage": coverage["per_model_coverage"],
        "comparable_intersection_size": coverage["comparable_intersection_size"],
        "is_partial_batch": coverage["is_partial_batch"],
        "comparability_warning": coverage.get("comparability_warning"),
        "batch_state": state["batch_state"],
        "batch_state_reasons": state["reasons"],
        "n_orphan_interrupted_logs": state["n_orphan_interrupted_logs"],
        "coverage": coverage_public,
        "pass_k_success": pass_k_success,
        "leaderboard_eligible": leaderboard_eligibility["eligible"],
        "leaderboard_eligibility": leaderboard_eligibility,
        "scenarios": scenario_manifest,
        "scenario_identity": scenario_identity,
        "artifacts": {
            "episodes_jsonl": str(out_dir / "episodes.jsonl"),
            "summary_csv": str(out_dir / "summary.csv"),
            "leaderboard_json": str(out_dir / "leaderboard.json"),
            "analysis_markdown": str(out_dir / "ANALYSIS.md"),
            "analysis_deep_json": str(out_dir / "analysis_deep.json"),
            "decision_impact_json": str(out_dir / "decision_impact_report.json"),
            "decision_impact_markdown": str(out_dir / "DECISION_IMPACT.md"),
            "evidence_applicability_json": str(
                out_dir / "evidence_applicability_report.json"
            ),
            "evidence_applicability_markdown": str(
                out_dir / "EVIDENCE_APPLICABILITY.md"
            ),
            "tool_effect_audit_json": str(out_dir / "tool_effect_audit_report.json"),
            "tool_effect_audit_markdown": str(out_dir / "TOOL_EFFECT_AUDIT.md"),
            "staleness_consumption_json": str(
                out_dir / "staleness_consumption_report.json"
            ),
            "staleness_consumption_markdown": str(out_dir / "STALENESS_CONSUMPTION.md"),
            "agent_failure_recipes_json": str(
                out_dir / "agent_failure_recipes_report.json"
            ),
            "agent_failure_recipes_markdown": str(out_dir / "AGENT_FAILURE_RECIPES.md"),
            "discrimination_json": str(out_dir / "discrimination_report.json"),
            "log_audit_json": str(out_dir / "LOG_AUDIT.json"),
            "log_audit_markdown": str(out_dir / "LOG_AUDIT.md"),
            "plots": plot_files,
        },
        "published_artifacts": {
            "episodes": {
                "path": published_episodes_path.relative_to(out_dir).as_posix(),
                "sha256": (
                    hashlib.sha256(published_episodes_path.read_bytes()).hexdigest()
                    if published_episodes_path.is_file()
                    else None
                ),
            },
            "leaderboard": {
                "path": leaderboard_path.relative_to(out_dir).as_posix(),
                "sha256": hashlib.sha256(leaderboard_path.read_bytes()).hexdigest(),
            },
        },
        (
            "primary_leaderboard"
            if bool(meta.get("formal_run"))
            else "diagnostic_flat_leaderboard"
        ): leaderboard,
        "intersection_leaderboard": intersection_leaderboard,
    }
    if bool(meta.get("formal_run")):
        artifact_root = out_dir.resolve()
        for name, value in list(manifest["artifacts"].items()):
            values = value if isinstance(value, list) else [value]
            portable_values: list[str] = []
            for raw_path in values:
                path = Path(str(raw_path))
                resolved = (
                    path.resolve()
                    if path.is_absolute()
                    else (artifact_root / path).resolve()
                )
                try:
                    portable_values.append(
                        resolved.relative_to(artifact_root).as_posix()
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"formal artifact path escapes batch root: {name}"
                    ) from exc
            manifest["artifacts"][name] = (
                portable_values if isinstance(value, list) else portable_values[0]
            )
        manifest = _portable_formal_manifest(manifest)
    (out_dir / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def _build_jobs(
    *,
    scenarios: list[str],
    scenario_bodies: dict[str, dict[str, Any]],
    models: list[str],
    seeds: list[int],
    temperature: float,
    args: argparse.Namespace,
    out_dir: Path,
    base_url: str | None,
    api_version: str | None,
    responses_base_url: str | None,
    suite_manifest_sha256: str | None = None,
    suite_eligibility: dict[str, Any] | None = None,
    agent_profile_sha256_by_model: dict[str, str] | None = None,
    agent_treatment_sha256_by_model: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    path_mapping: dict[str, str] = {}
    if suite_manifest_sha256 is None:
        suite_manifest_sha256 = _suite_manifest_sha256(scenarios, scenario_bodies)
    if suite_eligibility is None:
        suite_eligibility = {
            "suite_blocked": True,
            "reason": {"code": "suite_eligibility_not_supplied"},
            "diagnostic_cells": [],
            "uninformative_cells": [],
            "wait_dominant_cells": [],
        }
    suite_eligibility_sha256 = _canonical_json_sha256(suite_eligibility)
    pass_k = max(1, int(getattr(args, "pass_k", 1) or 1))
    seed_mode = str(getattr(args, "seed_mode", "fixed") or "fixed")
    for slug in scenarios:
        scenario = scenario_bodies[slug]
        try:
            estimated_horizon = max(0, int(scenario.get("horizon_ticks", 0) or 0))
        except (TypeError, ValueError):
            estimated_horizon = 0
        scenario_seeds = (
            [int(scenario.get("seed", 42))] if seed_mode == "scenario" else seeds
        )
        for model in models:
            cfg = _batch_llm_config(
                model=model,
                temperature=temperature,
                args=args,
                base_url=base_url,
                api_version=api_version,
                responses_base_url=responses_base_url,
            )
            agent_profile_sha256 = (
                (agent_profile_sha256_by_model or {}).get(model)
                or _agent_treatment_sha256(cfg)
            )
            agent_treatment_sha256 = (
                (agent_treatment_sha256_by_model or {}).get(model)
                or agent_profile_sha256
            )
            for seed in scenario_seeds:
                scenario_signature = _scenario_signature_for_run(scenario, seed)
                mdir = _safe_model_dir(model)
                for pass_index in range(pass_k):
                    pass_id = f"pass-{pass_index}"
                    job: dict[str, Any] = {
                        "scenario_slug": slug,
                        "seed": seed,
                        "pass_id": pass_id,
                        "pass_index": pass_index,
                        "pass_k": pass_k,
                        "model": model,
                        "temperature": temperature,
                        "scenario_signature": scenario_signature,
                        "evaluation_implementation_fingerprint": (
                            EVALUATION_IMPLEMENTATION_FINGERPRINT
                        ),
                        "run_semantics_fingerprint": (
                            _run_semantics_fingerprint(
                                cfg.prompt_mode,
                                cfg.max_tokens,
                                cfg.interaction_mode,
                            )
                            + f":agent-{agent_treatment_sha256}"
                        ),
                        "agent_treatment_sha256": agent_treatment_sha256,
                        "agent_profile_sha256": agent_profile_sha256,
                        "suite_scenario_signature": scenario.get("scenario_signature"),
                        "suite_manifest_sha256": suite_manifest_sha256,
                        "suite_eligibility": suite_eligibility,
                        "suite_eligibility_sha256": (suite_eligibility_sha256),
                        "seed_mode": seed_mode,
                        "domain": scenario.get("domain"),
                        "backend_kind": scenario.get("backend_kind"),
                        "construct_contract": scenario.get("construct_contract"),
                        "source_denominator_key": scenario.get("source_denominator_key")
                        or (
                            (scenario.get("case_ledger") or {}).get(
                                "source_denominator_key"
                            )
                            if isinstance(
                                scenario.get("case_ledger"),
                                dict,
                            )
                            else None
                        ),
                        "case_ledger": scenario.get("case_ledger"),
                        "estimated_horizon_ticks": estimated_horizon,
                        "llm_config": _llm_config_to_dict(cfg),
                        "operational_agency_attribution": {
                            "per_action": True,
                            "per_action_cap": None,
                            "per_action_groups": True,
                            "per_action_group_cap": None,
                        },
                    }
                    slug_safe = slug.replace("/", "_")
                    job["batch_output_dir"] = str(out_dir)
                    job["formal_run"] = bool(getattr(args, "formal_run", False))
                    if args.save_trajectories:
                        job["trajectory_dir"] = str(
                            _fit_fs_component(
                                out_dir
                                / "trajectories"
                                / mdir
                                / f"treatment-{agent_treatment_sha256}"
                                / f"{slug_safe}_s{seed}_{pass_id}",
                                path_mapping,
                            )
                        )
                    job["episode_log_path"] = str(
                        _fit_fs_component(
                            out_dir
                            / "logs"
                            / mdir
                            / f"treatment-{agent_treatment_sha256}"
                            / f"{slug_safe}_s{seed}_{pass_id}.log",
                            path_mapping,
                        )
                    )
                    jobs.append(job)
    jobs.sort(key=lambda job: -int(job.get("estimated_horizon_ticks", 0) or 0))
    if path_mapping:
        if bool(getattr(args, "formal_run", False)):
            path_mapping = canonicalize_repo_owned_paths(
                path_mapping, repo_root=REPO_ROOT
            )
        (out_dir / "path_shorten_map.json").write_text(
            json.dumps(
                {
                    "n_shortened": len(path_mapping),
                    "name_max_keep": _FS_NAME_KEEP,
                    "shortened": path_mapping,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return jobs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--scenario-slice",
        default="custom",
        help=(
            "Use 'custom' with --scenarios. Formal runs derive a private, "
            "hash-bound slice from --formal-manifest; legacy named presets "
            "are unsupported."
        ),
    )
    p.add_argument("--scenarios", nargs="*", help="Used when --scenario-slice=custom")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument(
        "--seed-mode",
        choices=["fixed", "scenario"],
        default="fixed",
        help=(
            "fixed = cross every scenario with --seeds; scenario = run each "
            "scenario once with its manifest-locked seed."
        ),
    )
    p.add_argument(
        "--pass-k",
        type=int,
        default=1,
        help=(
            "Independent evaluation replicates per scenario/model/seed cell. "
            "Replicates share the same simulator seed and are labelled with "
            "explicit pass_id values for reliability analysis; they do not "
            "increase structural scenario diversity."
        ),
    )
    p.add_argument("--models", default=None, help="Comma-separated model names")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum generated tokens per LLM turn. The batch default is 4096 "
            "so a legal multi-tool response is not truncated by the runner."
        ),
    )
    p.add_argument("--model-context-window-tokens", type=int, default=None)
    p.add_argument("--model-max-output-tokens", type=int, default=None)
    p.add_argument(
        "--persistent-history-max-messages",
        type=int,
        default=None,
        help="Persistent-session message bound (default: 32).",
    )
    p.add_argument(
        "--persistent-context-max-chars",
        type=int,
        default=None,
        help="Fixed persistent working-context bound (default: 48000 characters).",
    )
    p.add_argument(
        "--persistent-memory-max-items",
        type=int,
        default=None,
        help="Structured persistent-memory item bound (default: 64).",
    )
    p.add_argument("--provider-timeout-s", type=float, default=None)
    p.add_argument(
        "--provider-rpm-limit",
        type=int,
        default=None,
        help="Optional verified shared RPM limit; omitted means unknown/unbounded.",
    )
    p.add_argument(
        "--provider-rpd-limit",
        type=int,
        default=None,
        help="Optional verified shared RPD limit; omitted means unknown/unbounded.",
    )
    p.add_argument(
        "--provider-rate-limit-scope",
        default=None,
        help=(
            "Non-secret shared quota scope. Processes and shards with the same "
            "scope coordinate through one cross-process limiter."
        ),
    )
    p.add_argument("--protocol-repair-max-tokens", type=int, default=None)
    p.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default=None,
    )
    p.add_argument(
        "--prompt-mode",
        choices=["strict", "debug"],
        default="strict",
        help=(
            "LLM scenario briefing leak-control. 'strict' (default, "
            "benchmark-correct) hides difficulty_mode / difficulty_level "
            "/ complexity metrics from the agent. 'debug' restores the "
            "legacy v0.2.1 briefing for local development. Leaderboard-"
            "eligible batch runs MUST use 'strict' (Hard Red Line #6); "
            "'debug' triggers a warning below."
        ),
    )
    p.add_argument(
        "--interaction-mode",
        choices=["logical_stateless", "logical_persistent"],
        default="logical_persistent",
        help=(
            "LLM session treatment. logical_persistent is the current default; "
            "logical_stateless remains available only as an explicit historical "
            "ablation. Formal manifests still validate the selected treatment."
        ),
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help=(
            "Concurrency cap. In global mode this is total episode workers; "
            "in per_model mode this is max concurrent model lanes."
        ),
    )
    p.add_argument(
        "--scheduler-mode",
        default="global",
        choices=["global", "per_model"],
        help=(
            "global = one shared worker pool over all episodes; "
            "per_model = one sequential lane per model, lanes run in parallel."
        ),
    )
    p.add_argument(
        "--save-trajectories",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write per-episode trajectory JSON under trajectories/<model>/",
    )
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--base-url-env", default="OPERATE_API_BASE_URL")
    p.add_argument(
        "--responses-base-url-env",
        default="OPERATE_RESPONSES_API_BASE_URL",
    )
    p.add_argument("--api-version-env", default="OPERATE_API_VERSION")
    p.add_argument(
        "--api-mode",
        default="auto",
        choices=["auto", "chat_completions", "responses"],
    )
    p.add_argument(
        "--stream-chat-completions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Aggregate streamed Chat Completions chunks (required by stream-only gateways).",
    )
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip episodes already present as status=ok in episodes.jsonl",
    )
    p.add_argument(
        "--finalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh all derived reports, audits, and plots at the end of the run.",
    )
    p.add_argument(
        "--finalize-only",
        action="store_true",
        help="Do not schedule new episodes; rebuild reports from existing episodes.jsonl.",
    )
    p.add_argument(
        "--retry-cells",
        default=None,
        help=(
            "Path to LOG_AUDIT.json whose sample_orphan_interrupted_logs should be "
            "used as an allowlist for targeted reruns."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve scope and print what would run without scheduling episodes.",
    )
    p.add_argument(
        "--allow-blocked-suite",
        action="store_true",
        help=(
            "Diagnostic-only override for a blocked dynamic suite. It cannot "
            "be combined with --formal-run."
        ),
    )
    p.add_argument(
        "--formal-run",
        action="store_true",
        help=(
            "Require the release-bound Protocol-2.1 formal sampling contract. "
            "The current contract uses one logical_persistent model per shard."
        ),
    )
    p.add_argument(
        "--formal-manifest",
        type=Path,
        default=None,
        help=(
            "Versioned release/.../manifest.json binding the exact readiness "
            "artifact used by a formal run. When supplied it replaces the "
            "hard-coded --scenario-slice registry lookup."
        ),
    )
    args = p.parse_args()
    persistent_treatment = args.interaction_mode == "logical_persistent"
    if args.temperature is None:
        args.temperature = 0.0 if persistent_treatment else 1.0
    if args.max_tokens is None:
        args.max_tokens = (
            32_768
            if args.formal_run and persistent_treatment
            else (8192 if persistent_treatment else 4096)
        )
    if args.provider_timeout_s is None:
        args.provider_timeout_s = (
            300.0
            if args.formal_run and persistent_treatment
            else (150.0 if persistent_treatment else 60.0)
        )
    if args.protocol_repair_max_tokens is None:
        args.protocol_repair_max_tokens = (
            8_192
            if args.formal_run and persistent_treatment
            else (4096 if persistent_treatment else 512)
        )
    if args.formal_run and persistent_treatment:
        if args.persistent_history_max_messages is None:
            args.persistent_history_max_messages = 64
        if args.persistent_context_max_chars is None:
            args.persistent_context_max_chars = 512_000
        if args.persistent_memory_max_items is None:
            args.persistent_memory_max_items = 128
        if args.stream_chat_completions is None:
            args.stream_chat_completions = True
    elif args.stream_chat_completions is None:
        args.stream_chat_completions = False
    if args.pass_k < 1:
        print("[FATAL] --pass-k must be >= 1", file=sys.stderr)
        return 1
    if args.max_tokens < 1:
        print("[FATAL] --max-tokens must be >= 1", file=sys.stderr)
        return 1
    if args.provider_timeout_s <= 0:
        print("[FATAL] --provider-timeout-s must be positive", file=sys.stderr)
        return 1
    if any(
        limit is not None and limit <= 0
        for limit in (args.provider_rpm_limit, args.provider_rpd_limit)
    ):
        print("[FATAL] provider request limits must be positive", file=sys.stderr)
        return 1
    if any(
        limit is not None and limit > 0
        for limit in (args.provider_rpm_limit, args.provider_rpd_limit)
    ) and not str(args.provider_rate_limit_scope or "").strip():
        print(
            "[FATAL] --provider-rate-limit-scope is required when a provider "
            "request limit is enabled",
            file=sys.stderr,
        )
        return 1
    if args.protocol_repair_max_tokens < 1:
        print(
            "[FATAL] --protocol-repair-max-tokens must be positive",
            file=sys.stderr,
        )
        return 1
    if (args.model_context_window_tokens is None) != (
        args.model_max_output_tokens is None
    ):
        print(
            "[FATAL] --model-context-window-tokens and "
            "--model-max-output-tokens must be supplied together",
            file=sys.stderr,
        )
        return 1
    if (
        args.model_context_window_tokens is not None
        and args.model_context_window_tokens < 1
    ) or (
        args.model_max_output_tokens is not None and args.model_max_output_tokens < 1
    ):
        print("[FATAL] model capability token limits must be positive", file=sys.stderr)
        return 1
    if (
        args.persistent_history_max_messages is not None
        and args.persistent_history_max_messages < 4
    ):
        print(
            "[FATAL] --persistent-history-max-messages must be >= 4",
            file=sys.stderr,
        )
        return 1
    if (
        args.persistent_context_max_chars is not None
        and args.persistent_context_max_chars < 500
    ):
        print(
            "[FATAL] --persistent-context-max-chars must be >= 500",
            file=sys.stderr,
        )
        return 1
    if (
        args.persistent_memory_max_items is not None
        and args.persistent_memory_max_items < 4
    ):
        print(
            "[FATAL] --persistent-memory-max-items must be >= 4",
            file=sys.stderr,
        )
        return 1
    if args.prompt_mode == "debug":
        print(
            "[WARN] --prompt-mode=debug leaks difficulty_mode / difficulty_level / "
            "complexity metrics to the agent. Results from this run are NOT "
            "leaderboard-eligible (Hard Red Line #6); use only for local development.",
            file=sys.stderr,
        )
    if args.interaction_mode == "logical_persistent":
        print(
            "[INFO] --interaction-mode=logical_persistent is the current "
            "formal agentic treatment; logical_stateless is historical.",
            file=sys.stderr,
        )

    formal_manifest_binding: dict[str, Any] | None = None
    try:
        if args.formal_manifest is not None:
            formal_manifest_binding = resolve_formal_manifest_slice(
                args.formal_manifest
            )
            args.scenario_slice = str(formal_manifest_binding["slice_name"])
            DYNAMIC_SCENARIO_SLICES[args.scenario_slice] = formal_manifest_binding[
                "dynamic_slice_spec"
            ]
        if args.finalize_only:
            # Finalize-only scope is recovered from the existing run metadata
            # and episode rows after the output directory is opened below.
            patterns = []
            scenarios = []
            scenario_bodies = {}
            suite_manifest_sha256 = ""
        else:
            patterns = _resolve_patterns(args.scenario_slice, args.scenarios)
            scenarios = _expand_scenarios(patterns)
            scenario_bodies = {slug: load_scenario_yaml(slug) for slug in scenarios}
            if args.formal_run:
                _bind_scenario_contracts_for_slice(
                    args.scenario_slice, scenarios, scenario_bodies
                )
            suite_manifest_sha256 = _suite_manifest_sha256_for_slice(
                args.scenario_slice, scenarios, scenario_bodies
            )
        suite_eligibility = _suite_eligibility_binding(args.scenario_slice)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1

    is_protocol21_slice = (
        args.scenario_slice in PROTOCOL21_FORMAL_SLICES
        or formal_manifest_binding is not None
    )
    requested_models = [
        value.strip() for value in str(args.models or "").split(",") if value.strip()
    ]
    git_metadata = _git_metadata()
    if args.formal_run and not args.finalize_only:
        formal_reasons = _validate_protocol21_formal_run(
            {
                "scenario_slice": args.scenario_slice,
                "models": requested_models,
                "pass_k": args.pass_k,
                "max_workers": args.max_workers,
                "temperature": args.temperature,
                "prompt_mode": args.prompt_mode,
                "interaction_mode": args.interaction_mode,
                "seed_mode": args.seed_mode,
                "scheduler_mode": args.scheduler_mode,
                "save_trajectories": args.save_trajectories,
                "finalize": args.finalize,
                "allow_blocked_suite": args.allow_blocked_suite,
                "diagnostic_only": False,
                "formal_manifest_bound": formal_manifest_binding is not None,
                "git_metadata_available": git_metadata["git_metadata_available"],
                "git_dirty": git_metadata["git_dirty"],
                "model_context_window_tokens": (args.model_context_window_tokens),
                "model_max_output_tokens": args.model_max_output_tokens,
                "max_tokens": args.max_tokens,
                "protocol_repair_max_tokens": (args.protocol_repair_max_tokens),
                "persistent_history_max_messages": (
                    args.persistent_history_max_messages
                ),
                "persistent_context_max_chars": (args.persistent_context_max_chars),
                "persistent_memory_max_items": (args.persistent_memory_max_items),
                "provider_timeout_s": args.provider_timeout_s,
                "provider_rpm_limit": args.provider_rpm_limit,
                "provider_rpd_limit": args.provider_rpd_limit,
                "provider_rate_limit_scope": args.provider_rate_limit_scope,
                "tool_choice": "auto",
                "stream_chat_completions": args.stream_chat_completions,
                **_provider_failure_profile(formal_run=True),
            },
            suite_eligibility,
            suite_manifest_sha256=suite_manifest_sha256,
            scenario_bodies=scenario_bodies,
        )
        if formal_reasons:
            print(
                "[FATAL] formal Protocol-2.1 contract failed: "
                + ", ".join(formal_reasons),
                file=sys.stderr,
            )
            return 1
    if (
        not args.finalize_only
        and is_protocol21_slice
        and suite_eligibility.get("suite_blocked")
        and not args.allow_blocked_suite
    ):
        print(
            "[FATAL] Protocol-2.1 suite is blocked; use "
            "--allow-blocked-suite only for diagnostic runs",
            file=sys.stderr,
        )
        return 1
    env = _load_zhsrc_exports()
    base_url = (
        env.get(args.base_url_env)
        or os.getenv(args.base_url_env)
        or _load_named_zshrc_export(args.base_url_env)
    )
    api_version = (
        env.get(args.api_version_env)
        or os.getenv(args.api_version_env)
        or _load_named_zshrc_export(args.api_version_env)
    )
    responses_base_url = (
        env.get(args.responses_base_url_env)
        or os.getenv(args.responses_base_url_env)
        or _load_named_zshrc_export(args.responses_base_url_env)
    )
    if args.formal_run and not args.finalize_only and args.api_mode != "azure":
        if api_version:
            print(
                "[FATAL] formal API version is unsupported for non-Azure providers",
                file=sys.stderr,
            )
            return 1
        if responses_base_url:
            print(
                "[FATAL] formal responses base URL is unsupported for non-Azure providers",
                file=sys.stderr,
            )
            return 1
    api_key = (
        env.get(args.api_key_env)
        or os.getenv(args.api_key_env)
        or _load_named_zshrc_export(args.api_key_env)
    )
    if not api_key and not args.finalize_only and not args.dry_run:
        print(
            f"[FATAL] No API key in env ({args.api_key_env}), ~/.zhsrc, or ~/.zshrc",
            file=sys.stderr,
        )
        return 1
    if api_key:
        os.environ.setdefault(args.api_key_env, api_key)

    requested_output_dir = Path(args.output_dir)
    defer_formal_namespace = bool(args.formal_run and not args.finalize_only)
    out_dir = requested_output_dir
    existing_run_config_path = out_dir / "run_config.json"
    try:
        existing_run_config = (
            None
            if defer_formal_namespace
            else _load_run_config_fail_closed(existing_run_config_path)
        )
    except ValueError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1
    if not defer_formal_namespace and _no_resume_output_conflict(
        resume=args.resume,
        finalize_only=args.finalize_only,
        existing_run_config=existing_run_config,
        episodes_path=out_dir / "episodes.jsonl",
    ):
        print(
            "[FATAL] --no-resume refuses an initialized output namespace; "
            "use --resume or a new --output-dir",
            file=sys.stderr,
        )
        return 1
    if args.finalize_only and existing_run_config is None:
        print(
            "[FATAL] finalize-only requires a valid treatment-bound run_config.json",
            file=sys.stderr,
        )
        return 1
    if (
        not defer_formal_namespace
        and existing_run_config is None
        and out_dir.exists()
        and any(out_dir.iterdir())
    ):
        print(
            "[FATAL] non-empty output directory has no valid run_config.json",
            file=sys.stderr,
        )
        return 1
    existing_log_hints = (
        _recover_execution_hints_from_batch_log(out_dir) if args.finalize_only else {}
    )
    if args.finalize_only and existing_run_config is not None:
        existing_patterns = list(existing_run_config.get("patterns") or [])
        recovered_patterns = _recover_patterns_from_existing_batch(out_dir)
        if recovered_patterns and list(recovered_patterns) != list(existing_patterns):
            patterns = recovered_patterns
        elif existing_patterns:
            patterns = existing_patterns
        elif recovered_patterns:
            patterns = recovered_patterns
        else:
            print(
                f"[FATAL] Could not recover scenario patterns for finalize-only from "
                f"{existing_run_config_path} or episodes.jsonl",
                file=sys.stderr,
            )
            return 1
        if list(existing_patterns) != list(patterns):
            recovered_slice = _infer_scenario_slice_name(patterns)
        else:
            recovered_slice = str(existing_run_config.get("scenario_slice", "custom"))
        scenarios = _expand_scenarios(patterns)
        scenario_bodies = {slug: load_scenario_yaml(slug) for slug in scenarios}
        if bool(existing_run_config.get("formal_run")):
            _bind_scenario_contracts_for_slice(
                recovered_slice, scenarios, scenario_bodies
            )
        suite_manifest_sha256 = _suite_manifest_sha256(scenarios, scenario_bodies)

    active_slice_name = recovered_slice if args.finalize_only else args.scenario_slice
    suite_manifest_sha256 = _suite_manifest_sha256_for_slice(
        active_slice_name, scenarios, scenario_bodies
    )
    if active_slice_name != args.scenario_slice:
        suite_eligibility = _suite_eligibility_binding(active_slice_name)
    suite_eligibility_sha256 = _canonical_json_sha256(suite_eligibility)
    if suite_eligibility.get("suite_blocked"):
        LOGGER.warning(
            "scenario suite %s is release-blocked; clean checkpoints remain "
            "resumable, but all resulting rows are diagnostic-only until a "
            "new eligibility-bound suite artifact clears the blockers",
            active_slice_name,
        )
    models = _model_list(env, args.models)
    if args.finalize_only and existing_run_config is not None:
        configured_models = [
            str(model)
            for model in (existing_run_config.get("models") or [])
            if str(model).strip()
        ]
        models = _recover_models_for_finalize(
            configured_models=configured_models,
            rows=_load_episodes_jsonl(out_dir / "episodes.jsonl"),
            scheduled_model_count=(
                int(existing_log_hints["n_models"])
                if "n_models" in existing_log_hints
                else None
            ),
        )
    if (
        not args.finalize_only
        and len(models) > 1
        and args.model_context_window_tokens is not None
    ):
        print(
            "[FATAL] scalar model context/output capabilities cannot be "
            "shared across a multi-model batch; run each model separately",
            file=sys.stderr,
        )
        return 1
    if persistent_treatment and not args.finalize_only:
        missing_capabilities = [
            model
            for model in models
            if args.model_context_window_tokens is None
            and frozen_model_capabilities(model) is None
        ]
        if missing_capabilities:
            print(
                "[FATAL] persistent treatments require explicit model "
                "context/output capabilities for: "
                + ", ".join(sorted(missing_capabilities)),
                file=sys.stderr,
            )
            return 1
    if not args.finalize_only:
        repair_reserve = int(args.protocol_repair_max_tokens)
        for model in models:
            capabilities = (
                (
                    int(args.model_context_window_tokens),
                    int(args.model_max_output_tokens),
                )
                if args.model_context_window_tokens is not None
                else (
                    frozen_model_capabilities(model) if persistent_treatment else None
                )
            )
            if capabilities is None:
                continue
            capability_error = _model_capability_preflight_error(
                model=model,
                context_window=capabilities[0],
                max_output=capabilities[1],
                decision_reserve=int(args.max_tokens),
                repair_reserve=repair_reserve,
            )
            if capability_error is not None:
                print(f"[FATAL] {capability_error}", file=sys.stderr)
                return 1

    grid2op_required_envs = list(
        existing_run_config.get("grid2op_required_local_envs", [])
        if args.finalize_only and existing_run_config is not None
        else []
    )
    if not args.finalize_only:
        try:
            grid2op_required_envs = _grid2op_env_names_requiring_local_cache(
                scenario_bodies, scenarios
            )
        except ValueError as exc:
            print(f"[FATAL] {exc}", file=sys.stderr)
            return 1
    has_grid2op = bool(grid2op_required_envs) or any(
        str(scenario_bodies[slug].get("backend_kind", "")) == "grid2op"
        for slug in scenarios
    )
    grid2op_cache_preflight = None
    if grid2op_required_envs and not args.finalize_only:
        grid2op_cache_preflight = _grid2op_local_cache_preflight(grid2op_required_envs)
        if not grid2op_cache_preflight.get("ok", False):
            print(
                "[FATAL] Grid2Op local cache preflight failed: "
                f"{_format_grid2op_cache_blockers(grid2op_cache_preflight)}. "
                "Download/extract/load-verify these envs before formal LLM runs.",
                file=sys.stderr,
            )
            return 1
    max_workers = args.max_workers
    if (
        args.formal_run
        and has_grid2op
        and args.scheduler_mode == "global"
        and args.max_workers > 1
    ):
        print(
            "[FATAL] formal_concurrency_contract_unsatisfied",
            file=sys.stderr,
        )
        return 1
    if has_grid2op and max_workers > 1 and args.scheduler_mode == "global":
        LOGGER.warning(
            "grid2op-backed scenario detected; forcing --max-workers=1 for runner stability"
        )
        max_workers = 1
    elif has_grid2op and max_workers > 1 and args.scheduler_mode == "per_model":
        LOGGER.warning(
            "grid2op-backed scenario detected with --scheduler-mode=per_model; "
            "allowing concurrent model lanes because each model runs in its own process"
        )

    native_runtime_binding = (
        dict(existing_run_config.get("native_runtime_binding") or {})
        if args.finalize_only and existing_run_config is not None
        else _native_runtime_binding(
            scenario_bodies,
            scenarios,
            max_workers=max_workers,
            scheduler_mode=args.scheduler_mode,
        )
    )
    if not args.finalize_only and not native_runtime_binding.get("ok", False):
        print(
            "[FATAL] native runtime preflight failed: "
            + ", ".join(native_runtime_binding.get("blockers") or ["unknown"]),
            file=sys.stderr,
        )
        return 1

    current_tree_at_start = implementation_identity(REPO_ROOT)[
        "implementation_tree_sha256"
    ]
    expected_run_tree = str(
        (existing_run_config or {}).get("implementation_tree_sha256")
        or current_tree_at_start
    )
    if current_tree_at_start != expected_run_tree:
        print(
            "[FATAL] implementation_tree_changed_since_run_start",
            file=sys.stderr,
        )
        return 1

    agent_profile_identity_by_model = _batch_agent_treatment_identities(
        models=models,
        temperature=args.temperature,
        args=args,
        base_url=base_url,
        api_version=api_version,
        responses_base_url=responses_base_url,
    )
    agent_profile_sha256_by_model = {
        model: _canonical_json_sha256(identity)
        for model, identity in agent_profile_identity_by_model.items()
    }
    agent_treatment_sha256_by_model = (
        _formal_agent_treatment_hashes(
            agent_profile_sha256_by_model,
            formal_manifest_binding=formal_manifest_binding or {},
            implementation_tree_sha256=current_tree_at_start,
        )
        if args.formal_run
        else agent_profile_sha256_by_model
    )
    if defer_formal_namespace:
        try:
            out_dir = _resolve_logical_output_namespace(
                requested_output_dir,
                agent_treatment_sha256_by_model,
                formal_run=True,
            )
            existing_run_config_path = out_dir / "run_config.json"
            existing_run_config = _load_run_config_fail_closed(
                existing_run_config_path
            )
        except ValueError as exc:
            print(f"[FATAL] {exc}", file=sys.stderr)
            return 1
        if _no_resume_output_conflict(
            resume=args.resume,
            finalize_only=False,
            existing_run_config=existing_run_config,
            episodes_path=out_dir / "episodes.jsonl",
        ):
            print(
                "[FATAL] --no-resume refuses an initialized output namespace; "
                "use --resume or a new --output-dir",
                file=sys.stderr,
            )
            return 1
        if existing_run_config is None and out_dir.exists() and any(out_dir.iterdir()):
            print(
                "[FATAL] non-empty output directory has no valid run_config.json",
                file=sys.stderr,
            )
            return 1
    output_namespace_treatment_sha256 = (
        next(iter(agent_treatment_sha256_by_model.values()))
        if args.formal_run
        else None
    )
    meta = {
        "runner_version": "0.2.3-formalized",
        "scenario_slice": args.scenario_slice,
        "patterns": patterns,
        "n_scenarios": len(scenarios),
        "suite_manifest_sha256": suite_manifest_sha256,
        "suite_eligibility": suite_eligibility,
        "suite_eligibility_sha256": suite_eligibility_sha256,
        "formal_run": bool(args.formal_run),
        "implementation_tree_sha256": expected_run_tree,
        "implementation_tree_sha256_start": current_tree_at_start,
        "release_id": (formal_manifest_binding or {}).get("release_id"),
        **_formal_runtime_binding_metadata(formal_manifest_binding),
        "diagnostic_only": bool(is_protocol21_slice and args.allow_blocked_suite),
        "leaderboard_eligible": False,
        "formal_run_contract": (suite_eligibility.get("formal_run_contract") or {}),
        "formal_evaluation_ready": bool(
            suite_eligibility.get("formal_evaluation_ready")
        ),
        "readiness_source_artifact": suite_eligibility.get("readiness_source_artifact"),
        "readiness_source_artifact_sha256": suite_eligibility.get(
            "readiness_source_artifact_sha256"
        ),
        "readiness_artifact_sha256": suite_eligibility.get("readiness_artifact_sha256"),
        "requested_concurrency": args.max_workers,
        "effective_concurrency": max_workers,
        "models": models,
        "seeds": args.seeds,
        "seed_mode": args.seed_mode,
        "scenario_seed_pairs": (
            [[slug, int(scenario_bodies[slug].get("seed", 42))] for slug in scenarios]
            if args.seed_mode == "scenario"
            else None
        ),
        "pass_k": args.pass_k,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "model_context_window_tokens_by_model": {
            model: _batch_llm_config(
                model=model,
                temperature=args.temperature,
                args=args,
                base_url=base_url,
                api_version=api_version,
                responses_base_url=responses_base_url,
            ).model_context_window_tokens
            for model in models
        },
        "model_max_output_tokens_by_model": {
            model: _batch_llm_config(
                model=model,
                temperature=args.temperature,
                args=args,
                base_url=base_url,
                api_version=api_version,
                responses_base_url=responses_base_url,
            ).model_max_output_tokens
            for model in models
        },
        "tool_choice_supported_by_model": {
            model: _batch_llm_config(
                model=model,
                temperature=args.temperature,
                args=args,
                base_url=base_url,
                api_version=api_version,
                responses_base_url=responses_base_url,
            ).tool_choice_supported
            for model in models
        },
        "token_count_method": TOKEN_COUNT_METHOD_UTF8_BYTES,
        "token_count_version": TOKEN_COUNT_VERSION_V1,
        "prompt_mode": args.prompt_mode,
        "interaction_mode": args.interaction_mode,
        "wakeup_policy": deepcopy(CANONICAL_WAKEUP_POLICY),
        "persistent_history_max_messages": (
            args.persistent_history_max_messages
            if args.persistent_history_max_messages is not None
            else (32 if args.interaction_mode == "logical_persistent" else 24)
        ),
        "persistent_context_max_chars": (
            args.persistent_context_max_chars
            if args.persistent_context_max_chars is not None
            else (48_000 if args.interaction_mode == "logical_persistent" else 16_000)
        ),
        "persistent_memory_max_items": (
            args.persistent_memory_max_items
            if args.persistent_memory_max_items is not None
            else (64 if args.interaction_mode == "logical_persistent" else 32)
        ),
        "harness": "direct_api",
        "provider_timeout_s": (args.provider_timeout_s),
        "provider_rpm_limit": args.provider_rpm_limit,
        "provider_rpd_limit": args.provider_rpd_limit,
        "provider_rate_limit_scope": args.provider_rate_limit_scope,
        "protocol_repair_max_tokens": args.protocol_repair_max_tokens,
        "tool_choice": "auto",
        "reasoning_effort": args.reasoning_effort,
        **_provider_failure_profile(formal_run=args.formal_run),
        "agent_profile_schema_version": "agent_treatment_v1",
        "agent_profile_identity_by_model": agent_profile_identity_by_model,
        "agent_profile_sha256_by_model": agent_profile_sha256_by_model,
        "agent_treatment_schema_version": (
            "formal_logical_treatment_v1" if args.formal_run else "agent_treatment_v1"
        ),
        "agent_treatment_sha256_by_model": (agent_treatment_sha256_by_model),
        **(
            {
                "output_dir": str(out_dir.resolve()),
                "output_namespace_treatment_sha256": (
                    output_namespace_treatment_sha256
                ),
            }
            if args.formal_run
            else {}
        ),
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "evaluation_implementation_fingerprint": (
            EVALUATION_IMPLEMENTATION_FINGERPRINT
        ),
        "run_semantics_fingerprint": _run_semantics_fingerprint(
            args.prompt_mode,
            args.max_tokens,
            args.interaction_mode,
        ),
        "within_tick_interaction": True,
        "scoring_version": SCORING_VERSION,
        "scheduler_mode": args.scheduler_mode,
        "base_url": public_provider_url(base_url),
        "api_version": api_version,
        "responses_base_url": public_provider_url(responses_base_url),
        "api_mode": args.api_mode,
        "stream_chat_completions": args.stream_chat_completions,
        "max_workers_requested": args.max_workers,
        "max_workers_effective": max_workers,
        "save_trajectories": args.save_trajectories,
        "resume_enabled": args.resume,
        "finalize_enabled": args.finalize,
        "finalize_only": args.finalize_only,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "has_grid2op": has_grid2op,
        "grid2op_required_local_envs": grid2op_required_envs,
        "grid2op_local_cache_preflight": grid2op_cache_preflight,
        "native_runtime_binding": native_runtime_binding,
        **git_metadata,
    }
    if args.finalize_only and existing_run_config is not None:
        meta = dict(existing_run_config)
        meta.update(
            {
                "scenario_slice": recovered_slice,
                "patterns": patterns,
                "n_scenarios": len(scenarios),
                "models": models,
                "seeds": list(existing_run_config.get("seeds") or args.seeds),
                "seed_mode": str(
                    existing_log_hints.get(
                        "seed_mode",
                        existing_run_config.get("seed_mode", args.seed_mode),
                    )
                ),
                "scenario_seed_pairs": existing_run_config.get("scenario_seed_pairs"),
                "pass_k": int(
                    existing_log_hints.get(
                        "pass_k",
                        existing_run_config.get("pass_k", args.pass_k),
                    )
                    or 1
                ),
                "temperature": float(
                    existing_run_config.get("temperature", args.temperature)
                ),
                "max_tokens": int(
                    existing_run_config.get("max_tokens", args.max_tokens)
                ),
                "prompt_mode": str(
                    existing_run_config.get("prompt_mode", args.prompt_mode)
                ),
                "interaction_mode": str(
                    existing_run_config.get("interaction_mode", args.interaction_mode)
                ),
                "scheduler_mode": str(
                    existing_log_hints.get(
                        "scheduler_mode",
                        existing_run_config.get("scheduler_mode", args.scheduler_mode),
                    )
                ),
                "save_trajectories": bool(
                    existing_run_config.get("save_trajectories", args.save_trajectories)
                ),
                "max_consecutive_provider_failures": int(
                    existing_run_config.get("max_consecutive_provider_failures", 5)
                ),
                "provider_failure_policy": str(
                    existing_run_config.get(
                        "provider_failure_policy", "compat_fallback"
                    )
                ),
                "max_workers_requested": int(
                    existing_run_config.get("max_workers_requested", args.max_workers)
                ),
                "max_workers_effective": int(
                    existing_log_hints.get(
                        "max_workers_effective",
                        existing_run_config.get(
                            "max_workers_effective",
                            existing_run_config.get(
                                "max_workers_requested", args.max_workers
                            ),
                        ),
                    )
                ),
                "base_url": existing_run_config.get("base_url", base_url),
                "api_version": existing_run_config.get("api_version", api_version),
                "responses_base_url": existing_run_config.get(
                    "responses_base_url", responses_base_url
                ),
                "api_mode": existing_run_config.get("api_mode", args.api_mode),
                "resume_enabled": args.resume,
                "finalize_enabled": args.finalize,
                "finalize_only": True,
                "started_at_utc": datetime.now(UTC).isoformat(),
                "has_grid2op": bool(existing_log_hints.get("has_grid2op", False))
                or bool(existing_run_config.get("has_grid2op", False))
                or has_grid2op
                or _infer_has_grid2op_from_patterns(patterns),
                "grid2op_required_local_envs": existing_run_config.get(
                    "grid2op_required_local_envs",
                    grid2op_required_envs,
                ),
                "grid2op_local_cache_preflight": existing_run_config.get(
                    "grid2op_local_cache_preflight",
                    grid2op_cache_preflight,
                ),
                "native_runtime_binding": existing_run_config.get(
                    "native_runtime_binding",
                    native_runtime_binding,
                ),
            }
        )
    meta = _portable_formal_run_config(meta)
    namespace_error = _formal_output_namespace_binding_error(meta, out_dir)
    if namespace_error is not None:
        print(f"[FATAL] {namespace_error}", file=sys.stderr)
        return 1
    if str(meta.get("prompt_mode", "strict")) == "debug":
        LOGGER.warning(
            "run_config.json records prompt_mode=debug for this run directory; "
            "any leaderboard/report artifacts built from it are NOT "
            "leaderboard-eligible (Hard Red Line #6)."
        )
    if existing_run_config is not None:
        treatment_reasons = _run_config_treatment_compatibility_reasons(
            existing_run_config, meta
        )
        if treatment_reasons:
            print(
                "[FATAL] incompatible existing run_config.json: "
                + ", ".join(treatment_reasons),
                file=sys.stderr,
            )
            return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    _run_lock_handle = None
    if not args.dry_run:
        try:
            _run_lock_handle = _acquire_output_dir_lock(out_dir)
        except RuntimeError as exc:
            print(f"[FATAL] {exc}", file=sys.stderr)
            return 1
    (out_dir / "logs").mkdir(exist_ok=True)
    if args.save_trajectories:
        (out_dir / "trajectories").mkdir(exist_ok=True)
    log_file = out_dir / "batch_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    _atomic_write_text(
        out_dir / "run_config.json",
        json.dumps(meta, indent=2, ensure_ascii=False),
    )

    jobs: list[dict[str, Any]] = []
    retry_payload: dict[str, Any] | None = None
    if not args.finalize_only:
        jobs = _build_jobs(
            scenarios=scenarios,
            scenario_bodies=scenario_bodies,
            models=models,
            seeds=args.seeds,
            temperature=args.temperature,
            args=args,
            out_dir=out_dir,
            base_url=base_url,
            api_version=api_version,
            responses_base_url=responses_base_url,
            suite_manifest_sha256=suite_manifest_sha256,
            suite_eligibility=suite_eligibility,
            agent_profile_sha256_by_model=agent_profile_sha256_by_model,
            agent_treatment_sha256_by_model=agent_treatment_sha256_by_model,
        )
        for job in jobs:
            job["implementation_tree_sha256"] = expected_run_tree
        if args.retry_cells:
            retry_path = Path(args.retry_cells)
            if not retry_path.is_absolute():
                retry_path = (REPO_ROOT / retry_path).resolve()
            retry_payload = json.loads(retry_path.read_text(encoding="utf-8"))
            jobs = _apply_retry_cells_allowlist(jobs, retry_payload)
            if args.dry_run:
                LOGGER.info(
                    "retry-cells: selected %d jobs from %s; dry-run so no log quarantine",
                    len(jobs),
                    retry_path,
                )
            else:
                quarantined = _quarantine_retry_cell_logs(
                    out_dir,
                    retry_payload,
                    selected_jobs=jobs,
                )
                LOGGER.info(
                    "retry-cells: selected %d jobs from %s; quarantined=%d",
                    len(jobs),
                    retry_path,
                    len(quarantined),
                )

    episodes_path = out_dir / "episodes.jsonl"
    prior_rows = (
        _load_episodes_jsonl(episodes_path, repair_trailing=True) if args.resume else []
    )
    if args.resume and jobs:
        before = len(jobs)
        jobs = _filter_pending_jobs(jobs, prior_rows, batch_root=out_dir)
        skipped = before - len(jobs)
        if skipped:
            LOGGER.info(
                "resume: skipping %d completed episodes (%d remaining)",
                skipped,
                len(jobs),
            )

    LOGGER.info(
        "Scheduled %d episodes (%d scenarios × %d models × %s seed mode × pass_k=%d); max_workers=%d",
        len(jobs),
        len(scenarios),
        len(models),
        args.seed_mode,
        int(meta.get("pass_k", args.pass_k) or 1),
        max_workers,
    )
    if not args.finalize_only:
        print(
            f"Scheduled {len(jobs)} episodes — workers={max_workers} "
            f"(logs under {out_dir / 'logs'})"
        )
    if args.dry_run:
        preview = [
            {
                "model": j["model"],
                "scenario_slug": j["scenario_slug"],
                "seed": j["seed"],
                "pass_id": j.get("pass_id"),
                "episode_log_path": j.get("episode_log_path"),
            }
            for j in jobs[:20]
        ]
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "n_jobs": len(jobs),
                    "scheduler_mode": args.scheduler_mode,
                    "preview": preview,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    import concurrent.futures as futures

    write_mode = "a" if args.resume and episodes_path.exists() else "w"
    if jobs:
        if args.scheduler_mode == "per_model":
            if write_mode == "w":
                episodes_path.write_text("", encoding="utf-8")
            lanes = _build_model_lanes(jobs, episodes_path=episodes_path)
            lane_workers = max(1, min(max_workers, len(lanes)))
            LOGGER.info(
                "per-model scheduler: %d lanes across %d models; lane_workers=%d",
                len(lanes),
                len(lanes),
                lane_workers,
            )
            completed = 0
            with futures.ProcessPoolExecutor(max_workers=lane_workers) as pool:
                for i, summary in enumerate(
                    pool.map(_run_llm_model_lane, lanes, chunksize=1), start=1
                ):
                    completed += int(summary.get("n_completed", 0) or 0)
                    LOGGER.info(
                        "completed %d / %d model lanes (%d / %d episodes appended)",
                        i,
                        len(lanes),
                        completed,
                        len(jobs),
                    )
        else:
            _run_global_jobs(jobs, episodes_path, write_mode, max_workers)
    elif args.finalize_only:
        LOGGER.info("finalize-only mode: no new episodes scheduled")
    else:
        LOGGER.info("no pending episodes — using existing episodes.jsonl only")

    results = _load_episodes_jsonl(episodes_path)
    if not results:
        print(f"[FATAL] No episode rows found at {episodes_path}", file=sys.stderr)
        return 1
    compact_rows = _compact_episode_rows(results)
    if len(compact_rows) != len(results):
        _rewrite_episodes_jsonl(episodes_path, compact_rows)
        results = compact_rows

    current_tree_at_end = implementation_identity(REPO_ROOT)[
        "implementation_tree_sha256"
    ]
    if current_tree_at_end != expected_run_tree:
        meta["implementation_tree_sha256_end"] = current_tree_at_end
        meta["implementation_tree_stable"] = False
        _atomic_write_text(
            out_dir / "run_config.json",
            json.dumps(meta, indent=2, ensure_ascii=False),
        )
        print(
            "[FATAL] implementation_tree_drift; refusing finalization",
            file=sys.stderr,
        )
        return 1
    meta["implementation_tree_sha256_end"] = current_tree_at_end
    meta["implementation_tree_stable"] = True
    git_metadata_end = _git_metadata()
    meta["git_metadata_available_end"] = git_metadata_end.get("git_metadata_available")
    meta["git_commit_end"] = git_metadata_end.get("git_commit")
    meta["git_dirty_end"] = git_metadata_end.get("git_dirty")
    meta["git_status_short_end"] = git_metadata_end.get("git_status_short") or []
    if bool(meta.get("formal_run")) and (
        git_metadata_end.get("git_metadata_available") is not True
        or git_metadata_end.get("git_dirty") is not False
        or not meta.get("git_commit")
        or git_metadata_end.get("git_commit") != meta.get("git_commit")
    ):
        _atomic_write_text(
            out_dir / "run_config.json",
            json.dumps(meta, indent=2, ensure_ascii=False),
        )
        print(
            "[FATAL] formal_git_state_changed_or_unavailable; refusing finalization",
            file=sys.stderr,
        )
        return 1
    if bool(meta.get("formal_run")):
        runtime_binding_reasons = _formal_runtime_binding_reasons(meta)
        meta["formal_runtime_binding_reasons"] = runtime_binding_reasons
        meta["formal_runtime_binding_stable"] = not runtime_binding_reasons
        if runtime_binding_reasons:
            _atomic_write_text(
                out_dir / "run_config.json",
                json.dumps(meta, indent=2, ensure_ascii=False),
            )
            print(
                "[FATAL] formal_runtime_evidence_changed_or_unavailable; "
                "refusing finalization",
                file=sys.stderr,
            )
            return 1
    _atomic_write_text(
        out_dir / "run_config.json",
        json.dumps(meta, indent=2, ensure_ascii=False),
    )

    manifest: dict[str, Any] | None = None
    if args.finalize:
        manifest = _finalize_outputs(out_dir, results, meta)
        LOGGER.info(
            "finalized artifacts: ok=%d error=%d plots=%d",
            manifest["n_episodes_ok"],
            manifest["n_episodes_error"],
            len(manifest["artifacts"]["plots"]),
        )

    _print_batch_leaderboard(out_dir)
    if bool(meta.get("formal_run")) and not bool(
        (manifest or {}).get("leaderboard_eligible")
    ):
        return 2
    return 0 if all(r.get("status") == "ok" for r in results) else 2


def _print_batch_leaderboard(out_dir: Path) -> None:
    """Print a short leaderboard summary; no-op when finalize was skipped."""
    path = out_dir / "leaderboard.json"
    if not path.is_file():
        print(f"Results written to {out_dir} (no leaderboard.json yet)")
        return
    leaderboard_payload = json.loads(path.read_text(encoding="utf-8"))
    leaderboard_rows = (
        leaderboard_payload.get("primary_leaderboard")
        or leaderboard_payload.get("diagnostic_flat_leaderboard")
        or []
    )
    print(f"Results written to {out_dir}")
    for row in leaderboard_rows:
        if "primary_leaderboard_score" in row:
            print(
                f"  {row['model']}: "
                f"primary={row['primary_leaderboard_score']:.2f} "
                f"n={row['n_samples']}"
            )
        else:
            print(f"  {row['agent_id']}: mean={row['mean']:.2f} n={row['n_episodes']}")


if __name__ == "__main__":
    sys.exit(main())
