#!/usr/bin/env python3
"""Fail-closed readiness audit for the Protocol-2.1 long-horizon diagnostic.

This is a diagnostic preflight only.  It does not score episodes, mutate Core,
or make any release/leaderboard admission decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_admission import (  # noqa: E402
    requires_exact_strategy_minimality,
    resolve_protocol21_admission_profile,
)
from core.sidecar.sumo_sidecar import probe_sumo_transport  # noqa: E402
from core.source_asset_contract import virtual_source_identity_sha256  # noqa: E402
from evaluation import operational_agency_profile_is_consistent  # noqa: E402
from scripts.build_operational_agency_readiness_bundle import (  # noqa: E402
    validate_readiness_bundle_payload,
)
from scripts.run_operational_agency_known_groups_calibration import (  # noqa: E402
    audit_known_groups_artifact,
)

DEFAULT_PIPELINE = (
    REPO_ROOT / "release/operate_v0_58_0_candidate/operate_v058_formal"
)
DEFAULT_SLICE = DEFAULT_PIPELINE / "diagnostic/diagnostic_slice.json"
DEFAULT_READINESS = (
    DEFAULT_PIPELINE / "protocol2_v21_core_readiness.json"
)
DEFAULT_MANIFEST = DEFAULT_READINESS.with_name("protocol2_v21_pipeline_manifest.json")
DEFAULT_EPISODES = (
    DEFAULT_PIPELINE / "diagnostic/smoke/episodes.jsonl",
)
DEFAULT_OUTPUT = DEFAULT_PIPELINE / "diagnostic_test_readiness.json"
DEFAULT_AGENCY_POSITIVE_CONTROL = (
    REPO_ROOT / "reports/operate_v0_58_0/agency/generic_runtime.json"
)
DEFAULT_AGENCY_KNOWN_GROUPS = (
    DEFAULT_PIPELINE / "agency/known_groups.json"
)
DEFAULT_AGENCY_READINESS_BUNDLE = (
    DEFAULT_PIPELINE / "agency_readiness_bundle.json"
)
DEFAULT_AGENTS = ("wait_only", "random", "greedy_heuristic", "oracle_offline")
PRIMARY_FORMULA = "effective_source_backend_domain_macro_v1"
PRIMARY_INFERENCE = "physical_cluster_hierarchical_bootstrap_randomization_v1"
DIFFICULTY_LEVELS = {"basic", "medium", "high", "extreme"}
REQUIRED_ENVIRONMENT_EVENT_CHECKS = (
    "current_protocol_semantics",
    "material_exogenous_change_observed",
    "native_state_changing_control_available",
    "post_change_decision_observed",
    "source_consumption_passed",
    "source_independence_passed",
    "task_contract_passed",
    "world_change_contract_declared",
)
KNOWN_GROUPS_CALIBRATION_DOMAINS = {"logistics"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_binding(repo_root: Path, path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {"path": display_path, "sha256": _sha256(resolved)}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSON object required: {path}:{line_number}")
            rows.append(value)
    return rows


def _resolve(repo_root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _source_binding_matches(
    *, repo_root: Path, raw_source: object, expected_sha: object
) -> bool:
    if not isinstance(raw_source, str) or not isinstance(expected_sha, str):
        return False
    virtual_sha = virtual_source_identity_sha256(raw_source)
    if virtual_sha is not None:
        return expected_sha == virtual_sha
    resolved = _resolve(repo_root, raw_source)
    return (
        resolved is not None
        and resolved.is_file()
        and _sha256(resolved) == expected_sha
    )


def _int_or_zero(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _physical_identity(row: Mapping[str, Any]) -> str | None:
    ledger = row.get("case_ledger")
    if not isinstance(ledger, Mapping):
        ledger = {}
    value = ledger.get("physical_source_lock") or row.get("physical_source_key")
    if value in (None, "", {}, []):
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _block(
    blockers: list[dict[str, Any]],
    code: str,
    detail: str,
    *,
    gate: str,
    scenario_id: str | None = None,
) -> None:
    row: dict[str, Any] = {"code": code, "detail": detail, "gate": gate}
    if scenario_id is not None:
        row["scenario_id"] = scenario_id
    blockers.append(row)


def _audit_bound_files(
    *,
    repo_root: Path,
    rows: list[dict[str, Any]],
    readiness: Mapping[str, Any],
    blockers: list[dict[str, Any]],
) -> None:
    yaml_bindings = readiness.get("scenario_yaml_bindings")
    source_bindings = readiness.get("source_file_bindings")
    if not isinstance(yaml_bindings, Mapping):
        _block(
            blockers,
            "scenario_yaml_bindings_missing",
            "source readiness has no scenario YAML binding map",
            gate="baseline_preflight",
        )
        yaml_bindings = {}
    if not isinstance(source_bindings, Mapping):
        _block(
            blockers,
            "source_file_bindings_missing",
            "source readiness has no source-file binding map",
            gate="baseline_preflight",
        )
        source_bindings = {}
    for row in rows:
        scenario_id = str(row.get("scenario_id") or "")
        binding = yaml_bindings.get(scenario_id)
        bound_path = (
            _resolve(repo_root, binding.get("path")) if isinstance(binding, Mapping) else None
        )
        row_path = _resolve(repo_root, row.get("path"))
        expected_sha = binding.get("sha256") if isinstance(binding, Mapping) else None
        if (
            bound_path is None
            or row_path is None
            or bound_path.resolve() != row_path.resolve()
            or not bound_path.is_file()
            or not isinstance(expected_sha, str)
            or _sha256(bound_path) != expected_sha
        ):
            _block(
                blockers,
                "scenario_yaml_binding_mismatch",
                "scenario path or SHA-256 differs from frozen readiness",
                gate="baseline_preflight",
                scenario_id=scenario_id,
            )
        sources = source_bindings.get(scenario_id)
        if not isinstance(sources, Mapping) or not sources:
            _block(
                blockers,
                "source_file_binding_missing",
                "scenario has no frozen source-file binding",
                gate="baseline_preflight",
                scenario_id=scenario_id,
            )
            continue
        for source_path, source_sha in sources.items():
            if not _source_binding_matches(
                repo_root=repo_root,
                raw_source=source_path,
                expected_sha=source_sha,
            ):
                _block(
                    blockers,
                    "source_file_binding_mismatch",
                    f"source path or SHA-256 differs from readiness: {source_path}",
                    gate="baseline_preflight",
                    scenario_id=scenario_id,
                )


def _audit_artifact_bindings(
    *,
    repo_root: Path,
    readiness: Mapping[str, Any],
    blockers: list[dict[str, Any]],
) -> None:
    tree = readiness.get("implementation_tree_sha256")
    bindings = readiness.get("artifact_bindings")
    if bindings is None:
        return
    if not isinstance(bindings, Mapping):
        _block(
            blockers,
            "artifact_bindings_invalid",
            "readiness artifact_bindings must be an object",
            gate="baseline_preflight",
        )
        return
    for name, value in bindings.items():
        path = _resolve(repo_root, value.get("path")) if isinstance(value, Mapping) else None
        expected = value.get("sha256") if isinstance(value, Mapping) else None
        binding_tree = (
            value.get("implementation_tree_sha256") if isinstance(value, Mapping) else None
        )
        if (
            path is None
            or not path.is_file()
            or not isinstance(expected, str)
            or _sha256(path) != expected
            or binding_tree != tree
        ):
            _block(
                blockers,
                "readiness_artifact_binding_mismatch",
                f"artifact {name} is missing, stale, or has an inconsistent tree binding",
                gate="baseline_preflight",
            )


def _audit_slice_rows(
    *,
    rows: list[dict[str, Any]],
    readiness: Mapping[str, Any],
    blockers: list[dict[str, Any]],
) -> tuple[int, int]:
    admission_profile = resolve_protocol21_admission_profile(dict(readiness))
    source_rows = readiness.get("scenarios")
    if not isinstance(source_rows, list):
        _block(
            blockers,
            "source_readiness_scenarios_missing",
            "source readiness has no scenario list",
            gate="baseline_preflight",
        )
        source_rows = []
    source_by_id = {
        str(row.get("scenario_id") or ""): row for row in source_rows if isinstance(row, dict)
    }
    identities: Counter[tuple[str, str]] = Counter()
    extreme_count = 0
    physical: set[str] = set()
    for row in rows:
        scenario_id = str(row.get("scenario_id") or "")
        signature = str(row.get("scenario_signature") or "")
        identities[(scenario_id, signature)] += 1
        if not scenario_id or not signature or identities[(scenario_id, signature)] > 1:
            _block(
                blockers,
                "slice_identity_incomplete_or_duplicate",
                "scenario_id/signature must be present and unique",
                gate="baseline_preflight",
                scenario_id=scenario_id,
            )
        source_row = source_by_id.get(scenario_id)
        if source_row is None or source_row != row:
            _block(
                blockers,
                "slice_row_not_exact_readiness_member",
                "slice row is missing from or differs from frozen readiness",
                gate="baseline_preflight",
                scenario_id=scenario_id,
            )
        physical_id = _physical_identity(row)
        if physical_id is None or not row.get("source_denominator_key"):
            _block(
                blockers,
                "hierarchical_source_identity_missing",
                "physical and effective-source identities are both required",
                gate="baseline_preflight",
                scenario_id=scenario_id,
            )
        else:
            physical.add(physical_id)
        level = str(row.get("difficulty_level") or "")
        if level == "extreme":
            extreme_count += 1
        calibration = row.get("strategy_depth_validation")
        evaluation: object = None
        if isinstance(calibration, Mapping):
            difficulty = calibration.get("difficulty_calibration")
            if isinstance(difficulty, Mapping):
                evaluations = difficulty.get("evaluations")
                if isinstance(evaluations, Mapping):
                    evaluation = evaluations.get(level)
        observed = row.get("observed_depth_validation")
        observed_base_ok = (
            isinstance(observed, Mapping)
            and observed.get("task_contract_completed") is True
            and isinstance(observed.get("observed_effective_decision_ticks"), int)
            and isinstance(observed.get("tier_floor"), int)
        )
        observed_diagnostic_ok = (
            observed_base_ok
            and observed["observed_effective_decision_ticks"] >= 0
            and observed["tier_floor"] > 0
            and observed.get("disposition")
            in {"bounded_replay_required", "replace_or_retire_depth_contradicted"}
        )
        strict_depth_ok = level in DIFFICULTY_LEVELS and (
            isinstance(evaluation, Mapping)
            and evaluation.get("status") == "passed"
            and isinstance(calibration, Mapping)
            and calibration.get("disposition") == "required_depth_lower_bound_met"
            and observed_base_ok
            and observed["observed_effective_decision_ticks"] >= observed["tier_floor"]
        )
        diagnostic_depth_ok = (
            level in DIFFICULTY_LEVELS
            and isinstance(evaluation, Mapping)
            and evaluation.get("status") in {"passed", "held"}
            and isinstance(calibration, Mapping)
            and calibration.get("disposition")
            in {"required_depth_lower_bound_met", "difficulty_evidence_missing"}
            and observed_diagnostic_ok
        )
        depth_ok = (
            strict_depth_ok
            if requires_exact_strategy_minimality(profile=admission_profile)
            else diagnostic_depth_ok
        )
        if not depth_ok:
            _block(
                blockers,
                "difficulty_depth_gate_failed",
                f"selected {level!r} depth/observation evidence is not passed",
                gate="baseline_preflight",
                scenario_id=scenario_id,
            )
        agentic = row.get("agentic_contract")
        if not isinstance(agentic, Mapping) or agentic.get("status") != "passed":
            _block(
                blockers,
                "agentic_static_contract_failed",
                "static agentic contract is not passed",
                gate="baseline_preflight",
                scenario_id=scenario_id,
            )
        environment_checks = agentic.get("checks") if isinstance(agentic, Mapping) else None
        recipe_statuses_ok = (
            row.get("status") == "core_locked"
            and row.get("core_disposition") == "core_locked"
            and row.get("protocol21_admission_status") == "passed"
            and isinstance(row.get("source_grounded_validation"), Mapping)
            and row["source_grounded_validation"].get("status") == "admitted"
            and isinstance(row.get("task_contract_validation"), Mapping)
            and row["task_contract_validation"].get("status") == "passed"
            and isinstance(row.get("native_behavioral_validation"), Mapping)
            and row["native_behavioral_validation"].get("status") == "passed"
        )
        if (
            not recipe_statuses_ok
            or not isinstance(environment_checks, Mapping)
            or any(
                environment_checks.get(check) is not True
                for check in REQUIRED_ENVIRONMENT_EVENT_CHECKS
            )
        ):
            _block(
                blockers,
                "environment_event_contract_failed",
                "source-driven world change, native control, and post-change decision evidence are required",
                gate="baseline_preflight",
                scenario_id=scenario_id,
            )
    if extreme_count == 0:
        _block(
            blockers,
            "extreme_candidate_missing",
            "diagnostic slice contains no Extreme scenario",
            gate="baseline_preflight",
        )
    return extreme_count, len(physical)


def _result_errors(
    result: Mapping[str, Any],
    *,
    semantics: Mapping[str, Any],
    live_implementation_tree_sha256: str,
) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    if result.get("status") != "ok":
        errors.append(("episode_not_ok", "episode status is not ok"))
    if result.get("implementation_tree_sha256") != live_implementation_tree_sha256:
        errors.append(
            (
                "episode_implementation_tree_mismatch",
                "episode does not bind the live implementation tree",
            )
        )
    input_binding = result.get("diagnostic_input_binding")
    if (
        not isinstance(input_binding, Mapping)
        or input_binding.get("verified") is not True
        or input_binding.get("expected_scenario_id") != result.get("scenario_id")
        or input_binding.get("expected_scenario_signature")
        != result.get("scenario_signature")
    ):
        errors.append(
            (
                "diagnostic_input_binding_unverified",
                "episode is not bound to the exact diagnostic slice identity",
            )
        )
    runtime = result.get("diagnostic_runtime_integrity")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("implementation_tree_stable") is not True
        or runtime.get("implementation_tree_sha256_start")
        != live_implementation_tree_sha256
        or runtime.get("implementation_tree_sha256_end")
        != live_implementation_tree_sha256
    ):
        errors.append(
            (
                "episode_implementation_tree_drift",
                "implementation tree changed during the diagnostic episode",
            )
        )
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("process_check_available") is not True
    ):
        errors.append(
            (
                "episode_orphan_check_unavailable",
                "episode has no process-lifecycle verification",
            )
        )
    elif runtime.get("orphan_pids") != []:
        errors.append(
            (
                "episode_orphan_processes",
                "episode left native runtime processes behind",
            )
        )
    ground_truth = result.get("ground_truth_summary")
    if (
        not isinstance(ground_truth, Mapping)
        or "chose_fatal_option" not in ground_truth
        or (
            ground_truth.get("chose_fatal_option") is not None
            and not isinstance(ground_truth.get("chose_fatal_option"), bool)
        )
    ):
        errors.append(
            (
                "episode_fatal_state_evidence_missing",
                "episode lacks an explicit simulator-owned fatal-state result",
            )
        )
    protocol = result.get("evaluation_protocol")
    score = result.get("score")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("version") != semantics.get("protocol_version")
        or protocol.get("implementation_fingerprint") != semantics.get("implementation_fingerprint")
        or not isinstance(score, Mapping)
        or score.get("scoring_version") != semantics.get("scoring_version")
    ):
        errors.append(("episode_evaluation_semantics_mismatch", "episode semantics are stale"))
    trajectory = result.get("trajectory_summary")
    if not isinstance(trajectory, Mapping):
        return errors + [("trajectory_summary_missing", "trajectory summary missing")]
    terminal = trajectory.get("terminal_integrity")
    if not isinstance(terminal, Mapping) or terminal.get("release_ready") is not True:
        errors.append(
            (
                "episode_terminal_integrity_not_ready",
                "episode terminal integrity is incomplete",
            )
        )
    coverage = trajectory.get("tool_semantic_coverage")
    semantic = trajectory.get("tool_semantic_histogram")
    histogram = trajectory.get("tool_histogram")
    if not isinstance(coverage, Mapping):
        errors.append(("tool_semantic_coverage_missing", "ToolSpec semantic coverage missing"))
    else:
        unknown = coverage.get("unknown_tool_names")
        unclassified = coverage.get("unclassified_tool_names")
        missing_explicit = coverage.get("missing_explicit_semantic_role_names")
        missing_targets = coverage.get("missing_native_target_kind_names")
        missing_actuators = coverage.get("missing_actuator_family_names")
        registered = coverage.get("registered_tool_names")
        invoked = set(histogram) if isinstance(histogram, Mapping) else set()
        registered_set = set(registered) if isinstance(registered, list) else set()
        unknown_calls = semantic.get("n_unknown_calls") if isinstance(semantic, Mapping) else 0
        if (
            coverage.get("covered") is not True
            or bool(unknown)
            or bool(unclassified)
            or bool(missing_explicit)
            or bool(missing_targets)
            or bool(missing_actuators)
            or coverage.get("explicit_semantic_roles_complete") is not True
            or coverage.get("native_targets_complete") is not True
            or coverage.get("state_changing_actuators_complete") is not True
            or unknown_calls not in (None, 0)
            or not invoked.issubset(registered_set)
        ):
            errors.append(
                (
                    "tool_semantic_coverage_inconsistent",
                    "covered must agree with zero unknown/unclassified calls and registered tools",
                )
            )
    records = trajectory.get("event_response_records")
    profile = trajectory.get("operational_agency_profile")
    if not isinstance(records, list) or not isinstance(profile, Mapping):
        errors.append(
            ("agency_runtime_evidence_missing", "agency profile or runtime records missing")
        )
        return errors
    counterfactual = result.get("counterfactual")
    if not isinstance(counterfactual, Mapping):
        errors.append(
            (
                "agency_attribution_incomplete",
                "per-action and action-group attribution evidence is missing",
            )
        )
    else:
        for prefix in ("per_action", "per_action_group"):
            expected = counterfactual.get(f"{prefix}_expected")
            if (
                not isinstance(expected, int)
                or isinstance(expected, bool)
                or expected < 0
                or counterfactual.get(f"{prefix}_status") != "complete"
                or counterfactual.get(f"{prefix}_attempted") != expected
                or counterfactual.get(f"{prefix}_completed") != expected
                or counterfactual.get(f"{prefix}_failures") != []
            ):
                errors.append(
                    (
                        "agency_attribution_incomplete",
                        "per-action and action-group attribution must be uncapped and complete",
                    )
                )
                break
        if counterfactual.get("per_action_capped") is not False:
            errors.append(
                (
                    "agency_attribution_incomplete",
                    "per-action and action-group attribution must be uncapped and complete",
                )
            )
    if not operational_agency_profile_is_consistent(
        trajectory,
        counterfactual=(
            counterfactual if isinstance(counterfactual, Mapping) else None
        ),
    ):
        errors.append(
            (
                "agency_profile_inconsistent",
                "agency profile schema, counts, dimensions, or episode evidence are inconsistent",
            )
        )
    return errors


def _audit_agency_positive_control(
    *,
    repo_root: Path,
    payload: Mapping[str, Any] | None,
    live_implementation_tree_sha256: str,
    blockers: list[dict[str, Any]],
) -> bool:
    if payload is None:
        return False
    errors: list[str] = []
    if (
        payload.get("schema_version")
        != "operational-agency-runtime-positive-control-v1"
        or payload.get("status") != "passed"
        or payload.get("diagnostic_only") is not True
        or payload.get("release_admission") is not False
        or payload.get("core_admission_claimed") is not False
        or payload.get("source_independence_credit") is not False
    ):
        errors.append("top_level_contract")
    identity = payload.get("implementation_identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("implementation_tree_sha256")
        != live_implementation_tree_sha256
    ):
        errors.append("implementation_binding")
    base = payload.get("base_scenario_binding")
    base_path = _resolve(repo_root, base.get("path")) if isinstance(base, Mapping) else None
    if (
        base_path is None
        or not base_path.is_file()
        or not isinstance(base.get("sha256"), str)
        or _sha256(base_path) != base.get("sha256")
    ):
        errors.append("base_scenario_binding")
    sources = payload.get("source_file_bindings")
    if not isinstance(sources, Mapping) or not sources:
        errors.append("source_file_bindings")
    else:
        for value, expected_sha in sources.items():
            if not _source_binding_matches(
                repo_root=repo_root,
                raw_source=value,
                expected_sha=expected_sha,
            ):
                errors.append("source_file_binding_mismatch")
                break
    overlay = payload.get("overlay_contract")
    if (
        not isinstance(overlay, Mapping)
        or overlay.get("origin") != "declared_perturbation"
        or overlay.get("source_independence_credit") is not False
    ):
        errors.append("overlay_contract")
    determinism = payload.get("determinism")
    if (
        not isinstance(determinism, Mapping)
        or determinism.get("passed") is not True
        or _int_or_zero(determinism.get("repeats")) < 2
    ):
        errors.append("determinism")

    result = payload.get("result")
    counterfactual = result.get("counterfactual") if isinstance(result, Mapping) else None
    group = (
        counterfactual.get("repair_group")
        if isinstance(counterfactual, Mapping)
        else None
    )
    record = result.get("event_response_record") if isinstance(result, Mapping) else None
    profile = (
        result.get("operational_agency_profile")
        if isinstance(result, Mapping)
        else None
    )
    call_ids = group.get("call_ids") if isinstance(group, Mapping) else None
    try:
        group_delta = float(group.get("masked_action_group_delta"))
        record_delta = float(record.get("masked_action_group_delta"))
    except (AttributeError, TypeError, ValueError):
        group_delta = 0.0
        record_delta = -1.0
    if (
        not isinstance(result, Mapping)
        or not isinstance(counterfactual, Mapping)
        or counterfactual.get("per_action_group_status") != "complete"
        or _int_or_zero(counterfactual.get("per_action_group_expected")) < 1
        or _int_or_zero(counterfactual.get("per_action_group_attempted"))
        != _int_or_zero(counterfactual.get("per_action_group_expected"))
        or _int_or_zero(counterfactual.get("per_action_group_completed"))
        != _int_or_zero(counterfactual.get("per_action_group_expected"))
        or counterfactual.get("per_action_group_failures") != []
        or _int_or_zero(counterfactual.get("per_action_group_completed")) < 1
        or not isinstance(group, Mapping)
        or not isinstance(call_ids, list)
        or len(call_ids) < 2
        or len(set(str(value) for value in call_ids)) != len(call_ids)
        or group_delta <= 0.0
        or group_delta != group_delta
        or abs(group_delta) == float("inf")
        or not isinstance(record, Mapping)
        or record.get("masked_action_group_id") != group.get("group_id")
        or record.get("masked_action_group_call_ids") != call_ids
        or record_delta != group_delta
        or not record.get("backend_effect_evidence_ids")
    ):
        errors.append("action_group_replay_binding")
    profile_summary = {
        "event_response_records": result.get("event_response_records")
        if isinstance(result, Mapping)
        else None,
        "operational_agency_valid_evidence_ids": result.get(
            "operational_agency_valid_evidence_ids"
        )
        if isinstance(result, Mapping)
        else None,
        "operational_agency_profile": profile,
    }
    canonical_counterfactual = {
        "per_action": [],
        "per_action_groups": counterfactual.get("per_action_groups")
        if isinstance(counterfactual, Mapping)
        else [],
    }
    outcome = (
        (profile.get("dimensions") or {}).get("outcome_influence")
        if isinstance(profile, Mapping)
        else None
    )
    task = result.get("task_completion") if isinstance(result, Mapping) else None
    if (
        not isinstance(profile, Mapping)
        or profile.get("schema_version") != "operational_agency_profile_v1"
        or profile.get("diagnostic_only") is not True
        or profile.get("headline_score_included") is not False
        or profile.get("runtime_binding_verified") is not True
        or profile.get("runtime_evidence_binding_verified") is not True
        or profile.get("masked_replay_binding_verified") is not True
        or _int_or_zero(profile.get("causal_record_count")) < 1
        or not isinstance(outcome, Mapping)
        or outcome.get("applicable") is not True
        or not outcome.get("evidence_ids")
        or not isinstance(task, Mapping)
        or task.get("completed") is not True
        or not operational_agency_profile_is_consistent(
            profile_summary,
            counterfactual=canonical_counterfactual,
        )
    ):
        errors.append("runtime_agency_profile")
    if errors:
        _block(
            blockers,
            "agency_runtime_positive_control_invalid",
            ",".join(sorted(set(errors))),
            gate="llm_test",
        )
        return False
    return True


def _audit_agency_known_groups_calibration(
    *,
    repo_root: Path,
    payload: Mapping[str, Any] | None,
    live_implementation_tree_sha256: str,
    blockers: list[dict[str, Any]],
) -> bool:
    errors = audit_known_groups_artifact(
        repo_root=repo_root,
        payload=payload,
        live_implementation_tree_sha256=live_implementation_tree_sha256,
        required_domains=KNOWN_GROUPS_CALIBRATION_DOMAINS,
    )
    if errors:
        _block(
            blockers,
            "agency_known_groups_calibration_missing",
            "known-groups runtime artifact is not current and green: "
            + ",".join(errors),
            gate="llm_test",
        )
        return False
    return True


def _audit_results(
    *,
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    required_agents: list[str],
    repeats: int,
    semantics: Mapping[str, Any],
    blockers: list[dict[str, Any]],
    positive_control_verified: bool,
    live_implementation_tree_sha256: str,
) -> dict[str, int]:
    expected: set[tuple[str, str, str, int]] = set()
    for row in rows:
        for repeat in range(repeats):
            for agent in required_agents:
                expected.add(
                    (
                        str(row.get("scenario_id") or ""),
                        str(row.get("scenario_signature") or ""),
                        agent,
                        int(row.get("seed", 42)) + repeat,
                    )
                )
    actual: Counter[tuple[str, str, str, int]] = Counter()
    valid = 0
    credited_profiles = 0
    semantic_coverage_verified = 0
    agency_binding_verified = 0
    expected_identity_rows = 0
    for result in results:
        try:
            key = (
                str(result.get("scenario_id") or ""),
                str(result.get("scenario_signature") or ""),
                str(result.get("agent_name") or ""),
                int(result.get("seed")),
            )
        except (TypeError, ValueError):
            key = (str(result.get("scenario_id") or ""), "", "", -1)
        actual[key] += 1
        errors = _result_errors(
            result,
            semantics=semantics,
            live_implementation_tree_sha256=live_implementation_tree_sha256,
        )
        if key not in expected:
            errors.append(("episode_identity_not_expected", "episode is not in frozen matrix"))
        else:
            expected_identity_rows += 1
        error_codes = {code for code, _detail in errors}
        trajectory = result.get("trajectory_summary")
        coverage = (
            trajectory.get("tool_semantic_coverage") if isinstance(trajectory, Mapping) else None
        )
        if isinstance(coverage, Mapping) and not any(
            code.startswith("tool_semantic_") for code in error_codes
        ):
            semantic_coverage_verified += 1
        profile = (
            trajectory.get("operational_agency_profile")
            if isinstance(trajectory, Mapping)
            else None
        )
        if isinstance(profile, Mapping) and not any(
            code.startswith("agency_") for code in error_codes
        ):
            agency_binding_verified += 1
        if not errors:
            valid += 1
        for code, detail in errors:
            _block(
                blockers,
                code,
                detail,
                gate="llm_test",
                scenario_id=key[0],
            )
        if isinstance(profile, Mapping) and _int_or_zero(
            profile.get("causal_record_count")
        ) > 0:
            credited_profiles += 1
    missing = sorted(expected - set(actual))
    duplicates = sorted(key for key, count in actual.items() if count > 1)
    if missing or duplicates:
        _block(
            blockers,
            "episode_matrix_incomplete",
            f"missing={len(missing)}, duplicates={len(duplicates)}",
            gate="llm_test",
        )
    return {
        "expected_episode_bindings": len(expected),
        "observed_episode_rows": len(results),
        "valid_episode_bindings": valid,
        "missing_episode_bindings": len(missing),
        "duplicate_episode_bindings": len(duplicates),
        "expected_identity_episode_rows": expected_identity_rows,
        "tool_semantic_coverage_verified": semantic_coverage_verified,
        "agency_binding_verified": agency_binding_verified,
        "agency_profiles_with_causal_credit": credited_profiles,
        "runtime_positive_control_verified": int(positive_control_verified),
    }


def _audit_agency_readiness_bundle(
    *,
    repo_root: Path,
    payload: Mapping[str, Any] | None,
    live_implementation_tree_sha256: str,
    blockers: list[dict[str, Any]],
) -> bool:
    errors = validate_readiness_bundle_payload(
        payload,
        repo_root=repo_root,
        live_tree=live_implementation_tree_sha256,
    )
    if errors:
        _block(
            blockers,
            "agency_readiness_bundle_missing_or_invalid",
            "five-domain natural-source agency controls are invalid: "
            + ",".join(errors),
            gate="llm_test",
        )
    return not errors


def audit_test_readiness(
    *,
    repo_root: Path,
    slice_payload: dict[str, Any],
    source_readiness: dict[str, Any],
    source_readiness_sha256: str,
    pipeline_manifest: dict[str, Any],
    results: list[dict[str, Any]],
    required_agents: list[str],
    expected_scenarios: int,
    live_implementation_tree_sha256: str,
    runtime_environment: Mapping[str, str],
    runtime_available: bool,
    repeats: int = 1,
    agency_positive_control: Mapping[str, Any] | None = None,
    agency_known_groups_calibration: Mapping[str, Any] | None = None,
    agency_readiness_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit frozen inputs and diagnostic evidence without scoring them."""
    blockers: list[dict[str, Any]] = []
    rows_value = slice_payload.get("scenarios")
    rows = (
        [row for row in rows_value if isinstance(row, dict)] if isinstance(rows_value, list) else []
    )
    if (
        slice_payload.get("schema_version") != "protocol21-diagnostic-slice-v1"
        or slice_payload.get("status") != "working_set"
        or slice_payload.get("diagnostic_only") is not True
        or slice_payload.get("release_ready") is not False
        or slice_payload.get("leaderboard_eligible") is not False
        or slice_payload.get("n_scenarios") != len(rows)
        or len(rows) != expected_scenarios
    ):
        _block(
            blockers,
            "diagnostic_slice_contract_invalid",
            f"expected exactly {expected_scenarios} diagnostic-only rows",
            gate="baseline_preflight",
        )
    if slice_payload.get("source_readiness_sha256") != source_readiness_sha256:
        _block(
            blockers,
            "source_readiness_hash_mismatch",
            "slice does not bind the supplied readiness bytes",
            gate="baseline_preflight",
        )
    if (
        source_readiness.get("formal_evaluation_ready") is not True
        or source_readiness.get("status") != "formal_evaluation_ready"
    ):
        _block(
            blockers,
            "source_readiness_not_green",
            "source readiness is not formal-evaluation green",
            gate="baseline_preflight",
        )
    readiness_tree = source_readiness.get("implementation_tree_sha256")
    if readiness_tree != live_implementation_tree_sha256:
        _block(
            blockers,
            "source_readiness_implementation_stale",
            f"frozen={readiness_tree}, live={live_implementation_tree_sha256}",
            gate="baseline_preflight",
        )
    manifest_status = pipeline_manifest.get("status")
    if manifest_status != "formal_evaluation_ready":
        _block(
            blockers,
            "pipeline_manifest_not_complete",
            "required='formal_evaluation_ready', "
            f"observed={manifest_status!r}",
            gate="baseline_preflight",
        )
    manifest_tree = pipeline_manifest.get("implementation_tree_sha256")
    if manifest_tree != live_implementation_tree_sha256:
        _block(
            blockers,
            "pipeline_manifest_implementation_stale",
            f"frozen={manifest_tree}, live={live_implementation_tree_sha256}",
            gate="baseline_preflight",
        )
    if manifest_tree != readiness_tree:
        _block(
            blockers,
            "manifest_readiness_tree_mismatch",
            "pipeline manifest and readiness bind different implementation trees",
            gate="baseline_preflight",
        )
    if source_readiness.get("primary_leaderboard_formula_version") != PRIMARY_FORMULA:
        _block(
            blockers,
            "primary_formula_contract_missing_or_stale",
            f"required={PRIMARY_FORMULA}",
            gate="llm_test",
        )
    if source_readiness.get("primary_inference_version") != PRIMARY_INFERENCE:
        _block(
            blockers,
            "hierarchical_inference_contract_missing_or_stale",
            f"required={PRIMARY_INFERENCE}",
            gate="llm_test",
        )
    runtime_binding = pipeline_manifest.get("runtime_binding")
    requires_sumo = (
        isinstance(runtime_binding, Mapping) and runtime_binding.get("requires_real_sumo") is True
    )
    if requires_sumo and runtime_environment.get("OPERATE_TRAFFIC_BACKEND_REAL") != "1":
        _block(
            blockers,
            "real_sumo_env_gate_missing",
            "OPERATE_TRAFFIC_BACKEND_REAL=1 is required by the frozen manifest",
            gate="baseline_preflight",
        )
    if requires_sumo and not runtime_available:
        _block(
            blockers,
            "real_sumo_runtime_unavailable",
            "no reachable SUMO transport is available",
            gate="baseline_preflight",
        )
    extreme_count, physical_count = _audit_slice_rows(
        rows=rows,
        readiness=source_readiness,
        blockers=blockers,
    )
    _audit_bound_files(
        repo_root=repo_root,
        rows=rows,
        readiness=source_readiness,
        blockers=blockers,
    )
    _audit_artifact_bindings(
        repo_root=repo_root,
        readiness=source_readiness,
        blockers=blockers,
    )
    semantics = source_readiness.get("evaluation_semantics")
    if not isinstance(semantics, Mapping):
        semantics = {}
        _block(
            blockers,
            "evaluation_semantics_missing",
            "readiness has no frozen evaluation semantics",
            gate="baseline_preflight",
        )
    positive_control_verified = _audit_agency_positive_control(
        repo_root=repo_root,
        payload=agency_positive_control,
        live_implementation_tree_sha256=live_implementation_tree_sha256,
        blockers=blockers,
    )
    known_groups_verified = _audit_agency_known_groups_calibration(
        repo_root=repo_root,
        payload=agency_known_groups_calibration,
        live_implementation_tree_sha256=live_implementation_tree_sha256,
        blockers=blockers,
    )
    agency_bundle_verified = _audit_agency_readiness_bundle(
        repo_root=repo_root,
        payload=agency_readiness_bundle,
        live_implementation_tree_sha256=live_implementation_tree_sha256,
        blockers=blockers,
    )
    result_counts = _audit_results(
        rows=rows,
        results=results,
        required_agents=required_agents,
        repeats=repeats,
        semantics=semantics,
        blockers=blockers,
        positive_control_verified=positive_control_verified,
        live_implementation_tree_sha256=live_implementation_tree_sha256,
    )
    baseline_blocked = any(row["gate"] == "baseline_preflight" for row in blockers)
    test_blocked = bool(blockers)
    can_run_baselines = not baseline_blocked
    can_run_tests = not test_blocked
    agency_score_calibrated = bool(
        positive_control_verified
        and known_groups_verified
        and agency_bundle_verified
        and not any(
            str(row.get("code") or "").startswith("agency_")
            for row in blockers
        )
    )
    if can_run_tests:
        status = "ready_for_diagnostic_testing"
    elif can_run_baselines:
        status = "ready_for_baseline_repair_only"
    else:
        status = "held_fail_closed"
    blocker_counts = Counter(str(row["code"]) for row in blockers)
    return {
        "schema_version": "protocol21-long-horizon-test-readiness-v1",
        "status": status,
        "diagnostic_only": True,
        "release_admission": False,
        "leaderboard_eligible": False,
        "can_collect_episodes": can_run_baselines,
        "agency_score_calibrated": agency_score_calibrated,
        "leaderboard_ready": can_run_tests,
        "public_package_gate_evaluated": False,
        "public_package_ready": None,
        "can_start_diagnostic_baselines": can_run_baselines,
        "can_start_llm_testing": can_run_tests,
        "counts": {
            "slice_scenarios": len(rows),
            "extreme_scenarios": extreme_count,
            "physical_source_clusters": physical_count,
            **result_counts,
            "agency_known_groups_calibration_verified": int(
                known_groups_verified
            ),
            "agency_domain_positive_controls_verified": int(
                agency_bundle_verified
            ),
        },
        "contracts": {
            "primary_formula": source_readiness.get("primary_leaderboard_formula_version"),
            "primary_inference": source_readiness.get("primary_inference_version"),
            "required_primary_formula": PRIMARY_FORMULA,
            "required_primary_inference": PRIMARY_INFERENCE,
        },
        "implementation_binding": {
            "live": live_implementation_tree_sha256,
            "readiness": readiness_tree,
            "pipeline_manifest": manifest_tree,
        },
        "runtime": {
            "requires_real_sumo": requires_sumo,
            "OPERATE_TRAFFIC_BACKEND_REAL": runtime_environment.get("OPERATE_TRAFFIC_BACKEND_REAL"),
            "runtime_available": runtime_available,
        },
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, default=DEFAULT_SLICE)
    parser.add_argument("--source-readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--pipeline-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--episodes",
        type=Path,
        action="append",
        default=None,
        help="JSONL episode ledger; repeat to combine disjoint smoke batches",
    )
    parser.add_argument("--agents", nargs="+", default=list(DEFAULT_AGENTS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--expected-scenarios", type=int, default=40)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--agency-positive-control",
        type=Path,
        default=DEFAULT_AGENCY_POSITIVE_CONTROL,
    )
    parser.add_argument(
        "--agency-known-groups-calibration",
        type=Path,
        default=DEFAULT_AGENCY_KNOWN_GROUPS,
    )
    parser.add_argument(
        "--agency-readiness-bundle",
        type=Path,
        default=DEFAULT_AGENCY_READINESS_BUNDLE,
    )
    args = parser.parse_args(argv)
    episode_paths = args.episodes or list(DEFAULT_EPISODES)
    readiness_sha = _sha256(args.source_readiness)
    manifest = _load_object(args.pipeline_manifest)
    runtime_binding = manifest.get("runtime_binding")
    requires_sumo = (
        isinstance(runtime_binding, Mapping) and runtime_binding.get("requires_real_sumo") is True
    )
    runtime_available = not requires_sumo or probe_sumo_transport() is not None
    report = audit_test_readiness(
        repo_root=REPO_ROOT,
        slice_payload=_load_object(args.slice),
        source_readiness=_load_object(args.source_readiness),
        source_readiness_sha256=readiness_sha,
        pipeline_manifest=manifest,
        results=_load_jsonl(episode_paths),
        required_agents=[str(agent) for agent in args.agents],
        expected_scenarios=args.expected_scenarios,
        live_implementation_tree_sha256=implementation_identity()["implementation_tree_sha256"],
        runtime_environment=os.environ,
        runtime_available=runtime_available,
        repeats=args.repeats,
        agency_positive_control=_load_object(args.agency_positive_control),
        agency_known_groups_calibration=(
            _load_object(args.agency_known_groups_calibration)
            if args.agency_known_groups_calibration.is_file()
            else None
        ),
        agency_readiness_bundle=(
            _load_object(args.agency_readiness_bundle)
            if args.agency_readiness_bundle.is_file()
            else None
        ),
    )
    report["input_bindings"] = {
        "source_readiness": _file_binding(REPO_ROOT, args.source_readiness),
        "pipeline_manifest": _file_binding(REPO_ROOT, args.pipeline_manifest),
        "diagnostic_slice": _file_binding(REPO_ROOT, args.slice),
        "episode_ledgers": [
            _file_binding(REPO_ROOT, path) for path in episode_paths
        ],
        "runtime_sensitivity_control": _file_binding(
            REPO_ROOT, args.agency_positive_control
        ),
        "known_groups": (
            _file_binding(REPO_ROOT, args.agency_known_groups_calibration)
            if args.agency_known_groups_calibration.is_file()
            else None
        ),
        "agency_readiness_bundle": (
            _file_binding(REPO_ROOT, args.agency_readiness_bundle)
            if args.agency_readiness_bundle.is_file()
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "can_start_diagnostic_baselines": report["can_start_diagnostic_baselines"],
                "can_start_llm_testing": report["can_start_llm_testing"],
                "counts": report["counts"],
                "blocker_counts": report["blocker_counts"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["can_start_llm_testing"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
