#!/usr/bin/env python3
"""Build a fail-closed readiness binding for a Protocol-2.1 Core."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.agentic_core_contract import (  # noqa: E402
    REQUIRED_SEMANTICS,
)
from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_admission import (  # noqa: E402
    STRICT_ADMISSION_PROFILE,
    declared_protocol21_admission_profile,
    resolve_protocol21_admission_profile,
)
from core.protocol21_evidence import (  # noqa: E402
    artifact_binding,
    canonicalize_repo_owned_paths,
    resolve_binding_path,
)
from core.source_consumption_contract import resolve_declared_sources  # noqa: E402
from core.suite_identity import (  # noqa: E402
    canonical_scenario_slug,
    canonical_suite_manifest_sha256,
    scenario_yaml_binding,
    verify_scenario_row_against_yaml,
)
from domains.registry import get_backend_capability  # noqa: E402
from evaluation.leaderboard import (  # noqa: E402
    PRIMARY_LEADERBOARD_FORMULA_VERSION,
)
from evaluation.scorer import (  # noqa: E402
    SCORING_VERSION,
    TASK_COMPLETION_INPUT_UNIT,
    TASK_COMPLETION_SCORE_UNIT,
    WEIGHTED_EQUITY_FORMULA_VERSION,
)

FORMAL_WAKEUP_POLICY = {
    "session_start": True,
    "typed_actionable_events": True,
    "agent_scheduled_reviews": True,
    "harness_periodic_supervisory_scan": False,
    "unknown_events_actionable": False,
}
FORMAL_RUN_CONTRACT = {
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
    "agentic_profile": {
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
    },
    "save_trajectories": True,
    "required_construct_contract": "operational_agency.v1",
    "shard_merge_key": "formal_treatment_family_sha256",
    "wakeup_policy": dict(FORMAL_WAKEUP_POLICY),
    "realtime_formal_contract": {
        "contract_version": "realtime_persistent.v2",
        "interaction_mode": "realtime_persistent",
        "leaderboard": "realtime_supervision",
        "scorecard_version": "realtime-diagnostics/1.6",
        "diagnostic_schema_version": "realtime-diagnostics/1.6",
        "batch_schema_version": "realtime-formal-batch/1.1",
        "scorecard_schema_version": "realtime-formal-scorecard/1.1",
        "episode_schema_version": "realtime-episode/1.1",
        "treatment_schema_version": "realtime-treatment/1.1",
        "realtime_coordinator": "realtime_episode_v5",
        "wakeup_policy": dict(FORMAL_WAKEUP_POLICY),
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
    },
}

SCIENTIFIC_EVIDENCE_NAMES = (
    "behavioral",
    "source_consumption",
    "source_grounded",
    "agentic_contract",
)
DIAGNOSTIC_ARTIFACT_NAMES = (
    "task_contracts",
    "complexity",
    "observed_depth",
    "strategy_depth",
)


def _record_evidence_issue(
    name: str,
    code: str,
    *,
    blockers: set[str],
    diagnostics: dict[str, set[str]],
) -> None:
    if name in DIAGNOSTIC_ARTIFACT_NAMES:
        diagnostics[name].add(code)
    else:
        blockers.add(code)


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "samples", "scenarios"):
        value = report.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _semantics(report: dict[str, Any]) -> dict[str, str]:
    raw = report.get("evaluation_semantics") or {}
    protocol = report.get("evaluation_protocol") or {}
    config = report.get("config") or {}
    return {
        "protocol_version": str(
            raw.get("protocol_version")
            or raw.get("evaluation_protocol_version")
            or raw.get("version")
            or protocol.get("version")
            or report.get("evaluation_protocol_version")
            or config.get("evaluation_protocol_version")
            or ""
        ),
        "implementation_fingerprint": str(
            raw.get("implementation_fingerprint")
            or raw.get("evaluation_implementation_fingerprint")
            or protocol.get("implementation_fingerprint")
            or report.get("evaluation_implementation_fingerprint")
            or config.get("evaluation_implementation_fingerprint")
            or ""
        ),
        "scoring_version": str(
            raw.get("scoring_version")
            or report.get("scoring_version")
            or config.get("scoring_version")
            or ""
        ),
    }


def _complete(report: dict[str, Any]) -> bool:
    return report.get("status") == "complete" or report.get("complete") is True


def _index(
    report: dict[str, Any],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _rows(report):
        grouped[_identity(row)].append(row)
    return dict(grouped)


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _resolve(raw: str, *, repo_root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _suite_manifest_sha256(
    rows: list[dict[str, Any]],
    *,
    repo_root: Path,
) -> str:
    scenarios: list[str] = []
    bodies: dict[str, dict[str, Any]] = {}
    for row in rows:
        raw_path = str(row.get("path") or "")
        path = _resolve(raw_path, repo_root=repo_root)
        body: dict[str, Any] = {}
        if path.is_file():
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                body = loaded
        slug = canonical_scenario_slug(raw_path)
        scenarios.append(slug)
        bodies[slug] = {
            **body,
            "scenario_signature": body.get(
                "scenario_signature", row.get("scenario_signature")
            ),
            "seed": body.get("seed", row.get("seed")),
            "horizon_ticks": body.get(
                "horizon_ticks", row.get("horizon_ticks")
            ),
            "construct_contract": row.get(
                "construct_contract", "operational_agency.v1"
            ),
        }
    return canonical_suite_manifest_sha256(scenarios, bodies)


def _distribution(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    distribution = {
        field: dict(
            sorted(
                Counter(str(row.get(field) or "") for row in rows).items()
            )
        )
        for field in (
            "domain",
            "backend_kind",
            "family",
            "difficulty_level",
            "difficulty_mode",
        )
    }
    return {
        **distribution,
        **{f"by_{field}": counts for field, counts in distribution.items()},
    }


def build_readiness(
    *,
    payloads: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    source = payloads["source_suite"]
    core = payloads["core"]
    source_rows = _rows(source)
    core_rows = _rows(core)
    blockers: set[str] = set()
    diagnostic_artifact_issues: dict[str, set[str]] = defaultdict(set)
    diagnostic_row_labels: dict[str, set[str]] = defaultdict(set)
    diagnostic_depth_contradictions: list[str] = []
    source_admission_profile = resolve_protocol21_admission_profile(source)
    if any(
        declared_protocol21_admission_profile(row)
        not in (None, source_admission_profile)
        for row in source_rows
    ):
        blockers.add("admission_profile_mismatch")
    core_admission_profile = str(
        (core.get("constraint_validation") or {}).get(
            "core_admission_profile", STRICT_ADMISSION_PROFILE
        )
    )
    if core_admission_profile != source_admission_profile:
        blockers.add("admission_profile_mismatch")
    for name in ("source_grounded", "agentic_contract"):
        report = payloads[name]
        if resolve_protocol21_admission_profile(report) != source_admission_profile:
            blockers.add("admission_profile_mismatch")
        if any(
            resolve_protocol21_admission_profile(row)
            != source_admission_profile
            for row in _rows(report)
        ):
            blockers.add("admission_profile_mismatch")
    current_tree = implementation_identity(repo_root)[
        "implementation_tree_sha256"
    ]
    preflight = payloads["preflight"]
    if (
        preflight.get("status") != "passed"
        or int(preflight.get("n_expected", -1)) != len(source_rows)
        or int(preflight.get("n_completed", -1)) != len(source_rows)
        or int(preflight.get("n_fatal", -1)) != 0
    ):
        blockers.add("production_preflight_failed")
    if preflight.get("implementation_tree_sha256") != current_tree:
        blockers.add("implementation_tree_mismatch")

    evidence_names = SCIENTIFIC_EVIDENCE_NAMES + DIAGNOSTIC_ARTIFACT_NAMES
    for name in evidence_names:
        report = payloads[name]
        if _semantics(report) != REQUIRED_SEMANTICS:
            _record_evidence_issue(
                name,
                "artifact_semantics_stale",
                blockers=blockers,
                diagnostics=diagnostic_artifact_issues,
            )
        if not _complete(report):
            _record_evidence_issue(
                name,
                "artifact_incomplete",
                blockers=blockers,
                diagnostics=diagnostic_artifact_issues,
            )
        if report.get("implementation_tree_sha256") != current_tree:
            _record_evidence_issue(
                name,
                "implementation_tree_mismatch",
                blockers=blockers,
                diagnostics=diagnostic_artifact_issues,
            )
        expected = report.get("n_expected")
        completed = report.get("n_completed")
        if (
            expected is not None
            and completed is not None
            and int(expected) != int(completed)
        ):
            _record_evidence_issue(
                name,
                "artifact_incomplete",
                blockers=blockers,
                diagnostics=diagnostic_artifact_issues,
            )
    if core.get("status") != "protocol21_core_candidate":
        blockers.add("core_status_invalid")
    if _semantics(core) != REQUIRED_SEMANTICS:
        blockers.add("artifact_semantics_stale")
    if core.get("implementation_tree_sha256") != current_tree:
        blockers.add("implementation_tree_mismatch")

    actual_bindings = {
        name: artifact_binding(path, repo_root=repo_root)
        for name, path in paths.items()
    }
    release_coverage = payloads.get("release_coverage") or {}
    release_composition_blockers = sorted(
        str(code)
        for code in release_coverage.get("release_coverage_blockers") or []
        if str(code)
    )
    release_composition_ready = bool(
        release_coverage.get("release_coverage_passed") is True
        and not release_composition_blockers
    )
    if not release_coverage:
        blockers.add("release_coverage_artifact_missing")
    else:
        if _semantics(release_coverage) != REQUIRED_SEMANTICS:
            blockers.add("release_coverage_semantics_stale")
        if not _complete(release_coverage):
            blockers.add("release_coverage_artifact_incomplete")
        if release_coverage.get("implementation_tree_sha256") != current_tree:
            blockers.add("implementation_tree_mismatch")
    binding_contracts = {
        "core": (
            "source_suite",
            "behavioral",
            "task_contracts",
            "observed_depth",
            "strategy_depth",
            "source_grounded",
            "agentic_contract",
        ),
        "agentic_contract": (
            "source_suite",
            "behavioral",
            "source_consumption",
            "task_contracts",
            "complexity",
            "observed_depth",
            "strategy_depth",
            "source_grounded",
        ),
        "source_grounded": (
            "source_suite",
            "behavioral",
            "source_consumption",
            "task_contracts",
            "complexity",
            "strategy_depth",
        ),
        "release_coverage": ("core",),
    }
    for owner, names in binding_contracts.items():
        declared_bindings = payloads[owner].get("input_bindings") or {}
        for name in names:
            declared = declared_bindings.get(name) or {}
            if not declared:
                _record_evidence_issue(
                    name,
                    "artifact_binding_missing",
                    blockers=blockers,
                    diagnostics=diagnostic_artifact_issues,
                )
                continue
            actual = actual_bindings[name]
            if declared.get("sha256") != actual.get("sha256"):
                _record_evidence_issue(
                    name,
                    "artifact_hash_mismatch",
                    blockers=blockers,
                    diagnostics=diagnostic_artifact_issues,
                )
            if declared.get("implementation_tree_sha256") != current_tree:
                _record_evidence_issue(
                    name,
                    "implementation_tree_mismatch",
                    blockers=blockers,
                    diagnostics=diagnostic_artifact_issues,
                )
            declared_path = str(declared.get("path") or "")
            try:
                resolved_declared_path = resolve_binding_path(
                    declared_path, repo_root=repo_root
                )
            except ValueError:
                resolved_declared_path = None
            if not declared_path or resolved_declared_path != paths[name].resolve():
                _record_evidence_issue(
                    name,
                    "artifact_path_mismatch",
                    blockers=blockers,
                    diagnostics=diagnostic_artifact_issues,
                )

    source_by_identity = {
        _identity(row): row
        for row in source_rows
        if all(_identity(row))
    }
    source_ids = [str(row.get("scenario_id") or "") for row in source_rows]
    source_identities = [_identity(row) for row in source_rows]
    ids = [str(row.get("scenario_id") or "") for row in core_rows]
    signatures = [
        str(row.get("scenario_signature") or "") for row in core_rows
    ]
    seeds = [row.get("seed") for row in core_rows]
    raw_paths = [str(row.get("path") or "") for row in core_rows]
    identities = list(zip(ids, signatures, strict=True))
    canonical_checks = {
        "source_scenario_id_unique": bool(source_ids)
        and all(source_ids)
        and len(source_ids) == len(set(source_ids)),
        "source_scenario_identity_unique": bool(source_identities)
        and all(
            scenario_id and signature
            for scenario_id, signature in source_identities
        )
        and len(source_identities) == len(set(source_identities)),
        "scenario_id_unique": bool(ids)
        and all(ids)
        and len(ids) == len(set(ids)),
        "scenario_identity_unique": bool(identities)
        and all(scenario_id and signature for scenario_id, signature in identities)
        and len(identities) == len(set(identities)),
        "scenario_signature_present": bool(signatures)
        and all(signatures)
        and len(signatures) == len(ids),
        "path_unique": bool(raw_paths)
        and all(raw_paths)
        and len(raw_paths) == len(set(raw_paths)),
        "seed_present": all(seed is not None for seed in seeds),
        "all_core_rows_in_source": set(identities).issubset(
            source_by_identity
        ),
    }
    if not all(canonical_checks.values()):
        blockers.add("canonical_identity_not_unique")

    report_indexes = {
        name: _index(payloads[name]) for name in evidence_names
    }
    freeze_ledger: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in core.get("incremental_freeze_ledger") or []:
        if isinstance(entry, dict):
            freeze_ledger[_identity(entry)].append(entry)
    if len(core.get("incremental_freeze_ledger") or []) != len(core_rows):
        blockers.add("incremental_freeze_ledger_scope_mismatch")
    source_identity_set = set(source_identities)
    for name in ("source_consumption", "source_grounded", "agentic_contract"):
        if set(report_indexes[name]) != source_identity_set:
            blockers.add(f"{name}_source_scope_mismatch")
    for row in core_rows:
        scenario_id, signature = _identity(row)
        identity = (scenario_id, signature)
        if not (
            row.get("status") == "core_locked"
            and row.get("core_disposition") == "core_locked"
            and row.get("protocol21_admission_status") == "passed"
        ):
            blockers.add("core_row_not_locked")
        fingerprint = str(row.get("admission_fingerprint") or "")
        ledger_entries = freeze_ledger.get(identity, [])
        if (
            len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
            or len(ledger_entries) != 1
            or ledger_entries[0].get("disposition") != "core_locked"
            or ledger_entries[0].get("admission_fingerprint") != fingerprint
        ):
            blockers.add("admission_fingerprint_mismatch")
        source_row = source_by_identity.get(identity)
        if source_row is None:
            blockers.add("core_unreviewed_row")
            continue
        for field in ("seed", "path"):
            if (
                source_row.get(field) is not None
                and str(source_row.get(field)) != str(row.get(field))
            ):
                blockers.add("canonical_identity_mismatch")
        for name, index in report_indexes.items():
            matches = index.get(identity, [])
            if not matches:
                identity_mismatch = any(
                    str(item.get("scenario_id") or "") == scenario_id
                    for item in _rows(payloads[name])
                )
                missing_code = {
                    "task_contracts": "task_contract_row_missing",
                    "source_grounded": "source_gate_row_missing",
                    "agentic_contract": "agentic_contract_row_missing",
                }.get(name, f"{name}_row_missing")
                if name in DIAGNOSTIC_ARTIFACT_NAMES:
                    if identity_mismatch:
                        diagnostic_artifact_issues[name].add(
                            "canonical_identity_mismatch"
                        )
                    diagnostic_artifact_issues[name].add(missing_code)
                else:
                    if identity_mismatch:
                        blockers.add("canonical_identity_mismatch")
                    blockers.add(missing_code)
                continue
            if name != "complexity" and len(matches) != 1:
                if name in DIAGNOSTIC_ARTIFACT_NAMES:
                    diagnostic_artifact_issues[name].add(
                        "canonical_identity_mismatch"
                    )
                else:
                    blockers.add("canonical_identity_mismatch")
        complexity_agents = {
            str(item.get("agent_name") or item.get("agent") or "")
            for item in report_indexes["complexity"].get(identity, [])
        }
        if not {
            "oracle_offline",
            "greedy_heuristic",
            "wait_only",
        }.issubset(complexity_agents):
            diagnostic_artifact_issues["complexity"].add(
                "complexity_reference_rows_missing"
            )
            diagnostic_row_labels[scenario_id].add(
                "complexity_reference_rows_missing"
            )
        task_rows = report_indexes["task_contracts"].get(identity, [])
        if task_rows and not any(
            item.get("status") == "passed" or item.get("completed") is True
            for item in task_rows
        ):
            diagnostic_row_labels[scenario_id].add(
                "task_contract_replay_not_passed"
            )
        strategy_rows = report_indexes["strategy_depth"].get(identity, [])
        if strategy_rows:
            if not any(item.get("core_action") == "keep" for item in strategy_rows):
                diagnostic_row_labels[scenario_id].add(
                    "strategy_depth_unproven"
                )
            for item in strategy_rows:
                calibration = item.get("difficulty_calibration")
                if not isinstance(calibration, dict):
                    diagnostic_row_labels[scenario_id].add(
                        "difficulty_label_evidence_missing"
                    )
                elif (
                    calibration.get("status") != "passed"
                    or calibration.get("declared_level_matches_evidence") is not True
                ):
                    diagnostic_row_labels[scenario_id].add(
                        "difficulty_label_unproven"
                    )
        for item in report_indexes["agentic_contract"].get(identity, []):
            diagnostic_row_labels[scenario_id].update(
                str(code)
                for code in item.get("diagnostic_blockers") or []
                if str(code)
            )
        source_gate_rows = report_indexes["source_grounded"].get(
            identity, []
        )
        if not any(
            item.get("status") in {"admitted", "admitted_for_core_review"}
            for item in source_gate_rows
        ):
            blockers.add("source_gate_not_admitted")
        observed_rows = report_indexes["observed_depth"].get(identity, [])
        if any(
            "contradicted" in str(item.get("disposition") or "")
            for item in observed_rows
        ):
            diagnostic_depth_contradictions.append(scenario_id)
            diagnostic_row_labels[scenario_id].add(
                "observed_tick_floor_contradiction"
            )
        consumption_rows = report_indexes["source_consumption"].get(
            identity, []
        )
        if not any(item.get("status") == "passed" for item in consumption_rows):
            blockers.add("source_consumption_not_passed")
        try:
            capability = get_backend_capability(row.get("backend_kind"))
        except KeyError:
            blockers.add("backend_fidelity_missing")
        else:
            if not capability.formal_core_allowed or capability.runtime_fidelity in {
                "mock",
                "synthetic_stub",
            }:
                blockers.add("backend_formal_fidelity_not_allowed")

    missing_yaml = []
    missing_provenance = []
    scenario_yaml_bindings: dict[str, dict[str, Any]] = {}
    source_file_bindings: dict[str, dict[str, str]] = {}
    for row in core_rows:
        scenario_id = str(row.get("scenario_id") or "")
        identity = _identity(row)
        path = _resolve(str(row.get("path") or ""), repo_root=repo_root)
        if not path.is_file():
            missing_yaml.append(scenario_id)
            continue
        try:
            yaml_binding = scenario_yaml_binding(path)
            scenario_yaml_bindings[scenario_id] = {
                key: value
                for key, value in yaml_binding.items()
                if key != "body"
            }
            identity_errors = verify_scenario_row_against_yaml(
                row,
                path=path,
            )
            if identity_errors:
                blockers.add("scenario_yaml_identity_mismatch")
            scenario = yaml_binding["body"]
        except Exception:
            blockers.add("scenario_yaml_identity_mismatch")
            continue
        declared, hashes, missing = resolve_declared_sources(
            scenario,
            repo_root=repo_root,
        )
        source_file_bindings[scenario_id] = hashes
        for raw in missing:
            missing_provenance.append(
                {"scenario_id": scenario_id, "path": raw}
            )
        source_gate_rows = report_indexes["source_grounded"].get(identity, [])
        if len(source_gate_rows) != 1:
            blockers.add("source_gate_row_missing")
        else:
            source_gate_row = source_gate_rows[0]
            if source_gate_row.get("scenario_file_sha256") != yaml_binding["sha256"]:
                blockers.add("scenario_yaml_hash_mismatch")
            if dict(source_gate_row.get("source_file_hashes") or {}) != hashes:
                blockers.add("source_file_hash_mismatch")
        source_consumption_rows = report_indexes["source_consumption"].get(
            identity, []
        )
        if len(source_consumption_rows) != 1:
            blockers.add("source_consumption_row_missing")
        else:
            consumption = source_consumption_rows[0]
            required = [
                *(consumption.get("required_runtime_source_files") or []),
                *(consumption.get("required_derivation_source_files") or []),
            ]
            if required != declared:
                blockers.add("source_consumption_declaration_mismatch")
            if dict(consumption.get("locked_source_hashes") or {}) != hashes:
                blockers.add("source_file_hash_mismatch")
    if missing_yaml:
        blockers.add("scenario_yaml_missing")
    if missing_provenance:
        blockers.add("provenance_missing")
    if not core_rows:
        blockers.add("core_empty")
    source_domains = {
        str(row.get("domain") or "") for row in source_rows if row.get("domain")
    }
    core_domains = {
        str(row.get("domain") or "") for row in core_rows if row.get("domain")
    }
    if not source_domains.issubset(core_domains):
        blockers.add("source_domain_missing_from_core")
    constraint_validation = core.get("constraint_validation") or {}
    required_constraints = (
        "effective_source_identity_unique",
        "quality_maximal_admission_passed",
    )
    if not all(
        constraint_validation.get(name) is True
        for name in required_constraints
    ):
        blockers.add("core_selection_constraints_failed")
    if core.get("selection_policy") != "quality_maximal_v1":
        blockers.add("core_selection_policy_stale")

    per_domain: dict[str, Counter[str]] = defaultdict(Counter)
    per_backend: dict[str, Counter[str]] = defaultdict(Counter)
    for row in core_rows:
        identity = _identity(row)
        agentic_passed = any(
            item.get("status") == "passed"
            for item in report_indexes["agentic_contract"].get(
                identity, []
            )
        )
        source_admitted = any(
            item.get("status") in {"admitted", "admitted_for_core_review"}
            for item in report_indexes["source_grounded"].get(
                identity, []
            )
        )
        source_consumption_passed = any(
            item.get("status") == "passed"
            for item in report_indexes["source_consumption"].get(identity, [])
        )
        for counter in (
            per_domain[str(row.get("domain") or "")],
            per_backend[str(row.get("backend_kind") or "")],
        ):
            counter["n_rows"] += 1
            counter["agentic_passed"] += int(agentic_passed)
            counter["source_admitted"] += int(source_admitted)
            counter["source_consumption_passed"] += int(
                source_consumption_passed
            )

    for row in core_rows:
        row["construct_contract"] = "operational_agency.v1"

    ordered_blockers = sorted(blockers)
    ready = not ordered_blockers
    report = {
        "schema_version": "1.0",
        "status": "formal_evaluation_ready" if ready else "blocked",
        "formal_evaluation_ready": ready,
        "formal_run_blockers": ordered_blockers,
        "diagnostic_artifact_issues": {
            name: sorted(issues)
            for name, issues in sorted(diagnostic_artifact_issues.items())
            if issues
        },
        "diagnostic_row_labels": {
            scenario_id: sorted(labels)
            for scenario_id, labels in sorted(diagnostic_row_labels.items())
            if labels
        },
        "diagnostic_depth_contradictions": diagnostic_depth_contradictions,
        "leaderboard_eligible": False,
        "leaderboard_blockers": [
            "formal_logical_persistent_evaluation_pending",
            "formal_realtime_persistent_evaluation_pending",
        ],
        "scoring_version": SCORING_VERSION,
        "primary_leaderboard_formula_version": (
            PRIMARY_LEADERBOARD_FORMULA_VERSION
        ),
        "primary_inference_version": (
            "physical_cluster_hierarchical_bootstrap_randomization_v1"
        ),
        "task_completion_input_unit": TASK_COMPLETION_INPUT_UNIT,
        "task_completion_score_unit": TASK_COMPLETION_SCORE_UNIT,
        "weighted_equity_formula_version": (
            WEIGHTED_EQUITY_FORMULA_VERSION
        ),
        "evaluation_semantics": dict(REQUIRED_SEMANTICS),
        "implementation_tree_sha256": current_tree,
        "artifact_bindings": actual_bindings,
        "release_coverage": release_coverage,
        "release_composition_ready": release_composition_ready,
        "release_composition_blockers": release_composition_blockers,
        "source_artifact": str(paths["source_suite"].absolute()),
        "source_artifact_sha256": actual_bindings["source_suite"]["sha256"],
        "suite_manifest_sha256": _suite_manifest_sha256(
            core_rows, repo_root=repo_root
        ),
        "scenario_yaml_bindings": scenario_yaml_bindings,
        "source_file_bindings": source_file_bindings,
        "constraint_validation": constraint_validation,
        "core_admission_profile": source_admission_profile,
        "canonical_identity_checks": canonical_checks,
        "distribution": _distribution(core_rows),
        "per_domain_gate_summary": {
            key: dict(value) for key, value in sorted(per_domain.items())
        },
        "per_backend_gate_summary": {
            key: dict(value) for key, value in sorted(per_backend.items())
        },
        "formal_run_contract": copy.deepcopy(FORMAL_RUN_CONTRACT),
        "missing_scenario_files": missing_yaml,
        "missing_provenance": missing_provenance,
        "n_scenarios": len(core_rows),
        "scenarios": core_rows,
    }
    return canonicalize_repo_owned_paths(report, repo_root=repo_root)


def build_readiness_from_paths(
    *,
    core: Path,
    source_suite: Path,
    preflight: Path,
    behavioral: Path,
    source_consumption: Path,
    task_contracts: Path,
    complexity: Path,
    observed_depth: Path,
    strategy_depth: Path,
    source_grounded: Path,
    agentic_contract: Path,
    release_coverage: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    paths = {
        "core": core,
        "source_suite": source_suite,
        "preflight": preflight,
        "behavioral": behavioral,
        "source_consumption": source_consumption,
        "task_contracts": task_contracts,
        "complexity": complexity,
        "observed_depth": observed_depth,
        "strategy_depth": strategy_depth,
        "source_grounded": source_grounded,
        "agentic_contract": agentic_contract,
        "release_coverage": release_coverage,
    }
    payloads = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    return build_readiness(
        payloads=payloads,
        paths=paths,
        repo_root=repo_root,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--behavioral", type=Path, required=True)
    parser.add_argument("--source-consumption", type=Path, required=True)
    parser.add_argument("--task-contracts", type=Path, required=True)
    parser.add_argument("--complexity", type=Path, required=True)
    parser.add_argument("--observed-depth", type=Path, required=True)
    parser.add_argument("--strategy-depth", type=Path, required=True)
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--agentic-contract", type=Path, required=True)
    parser.add_argument("--release-coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_readiness_from_paths(
        core=args.core,
        source_suite=args.source_suite,
        preflight=args.preflight,
        behavioral=args.behavioral,
        source_consumption=args.source_consumption,
        task_contracts=args.task_contracts,
        complexity=args.complexity,
        observed_depth=args.observed_depth,
        strategy_depth=args.strategy_depth,
        source_grounded=args.source_gate,
        agentic_contract=args.agentic_contract,
        release_coverage=args.release_coverage,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "formal_evaluation_ready": report[
                    "formal_evaluation_ready"
                ],
                "formal_run_blockers": report["formal_run_blockers"],
            },
            indent=2,
        )
    )
    return 0 if report["formal_evaluation_ready"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
