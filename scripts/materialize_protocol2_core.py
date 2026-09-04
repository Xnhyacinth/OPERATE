#!/usr/bin/env python3
"""Materialize a fail-closed Protocol-2.0 Core from replay evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit._common import _resolve_scenario_path  # noqa: E402
from core.agentic_core_contract import (  # noqa: E402
    REQUIRED_SEMANTICS,
    artifact_binding,
)
from core.difficulty_contract import (  # noqa: E402
    DIFFICULTY_CONTRACT_VERSION,
    difficulty_calibration_matches_level,
)
from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_admission import (  # noqa: E402
    QUALITY_CORE_V2_ADMISSION_PROFILE,
    declared_protocol21_admission_profile,
    requires_exact_strategy_minimality,
    resolve_protocol21_admission_profile,
)
from core.protocol21_evidence import row_identity  # noqa: E402
from core.source_asset_contract import (  # noqa: E402
    canonical_physical_source_asset_key,
)
from core.working_set_contract import (  # noqa: E402
    extract_protocol21_selection_constraints,
    validate_protocol21_row_lineage,
)
from scripts.build_primary_suite import (  # noqa: E402
    _decision_pressure_axis,
    _independence_axis,
)

_ARTIFACT_IDENTITY_FAILURES = {
    "artifact_identity_missing",
    "artifact_identity_mismatch",
    "artifact_identity_multiplicity",
}


def _distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        f"by_{field}": dict(
            sorted(Counter(str(row.get(field) or "") for row in rows).items())
        )
        for field in (
            "domain",
            "backend_kind",
            "family",
            "difficulty_level",
            "difficulty_mode",
        )
    }


def _body(row: dict[str, Any]) -> dict[str, Any]:
    loaded = yaml.safe_load(
        _resolve_scenario_path(str(row["path"])).read_text(encoding="utf-8")
    )
    return loaded if isinstance(loaded, dict) else {}


def _effective_source_key(row: dict[str, Any]) -> str:
    ledger = row.get("case_ledger") or {}
    return str(
        row.get("source_denominator_key") or ledger.get("source_denominator_key") or ""
    )


def _physical_source_key(row: dict[str, Any]) -> str:
    ledger = row.get("case_ledger") or {}
    value = (
        ledger.get("physical_source_lock")
        or row.get("physical_source_key_or_lock")
        or row.get("physical_source_key")
        or ledger.get("physical_source_key")
        or ""
    )
    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    return str(value)


def _physical_asset_key(row: dict[str, Any]) -> str:
    ledger = row.get("case_ledger") or {}
    value = (
        ledger.get("physical_source_lock")
        or row.get("physical_source_key_or_lock")
        or row.get("physical_source_key")
        or ledger.get("physical_source_key")
        or ""
    )
    return canonical_physical_source_asset_key(value)


def _is_pending_admission_marker(value: Any) -> bool:
    normalized = str(value or "").lower()
    return any(
        marker in normalized
        for marker in (
            "protocol21_full_gates_not_run",
            "requires_behavior_task_depth_agentic_gates",
            "pending_full_protocol21_gates",
            "pending isolated protocol-2.1 admission",
        )
    )


def _finalize_admission_metadata(row: dict[str, Any]) -> None:
    """Archive source-stage diagnostics after current gates pass."""
    archived = deepcopy(row.get("pre_admission_metadata") or {})
    reason_codes = row.pop("reason_codes", None)
    if reason_codes:
        archived["reason_codes"] = deepcopy(reason_codes)

    ledger = deepcopy(row.get("case_ledger") or {})
    diagnostic_risk = list(ledger.get("diagnostic_risk") or [])
    pending_risks = [
        value for value in diagnostic_risk if _is_pending_admission_marker(value)
    ]
    active_risks = [
        value for value in diagnostic_risk if not _is_pending_admission_marker(value)
    ]
    if pending_risks:
        archived["case_ledger_diagnostic_risk"] = pending_risks
    if active_risks:
        ledger["diagnostic_risk"] = active_risks
    else:
        ledger.pop("diagnostic_risk", None)

    complexity_tags = list(ledger.get("complexity_tags") or [])
    pending_tags = [
        value for value in complexity_tags if _is_pending_admission_marker(value)
    ]
    if pending_tags:
        archived["case_ledger_pending_complexity_tags"] = pending_tags
        ledger["complexity_tags"] = [
            value
            for value in complexity_tags
            if not _is_pending_admission_marker(value)
        ]

    keep_rationale = str(ledger.get("keep_rationale") or "")
    if _is_pending_admission_marker(keep_rationale):
        archived["case_ledger_keep_rationale"] = keep_rationale
        ledger["keep_rationale"] = (
            "Retained after all current identity-bound Protocol-2.1 behavior, "
            "task, headroom, depth, source-grounded, agentic, replay, and "
            "independence gates passed."
        )

    row["case_ledger"] = ledger
    if archived:
        row["pre_admission_metadata"] = archived


def _partition_effective_source_duplicates(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep a stable representative per effective source and secondary the rest."""

    primary: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []
    representative_by_source: dict[str, str] = {}
    for row in rows:
        source_key = _effective_source_key(row)
        if not source_key:
            continue
        scenario_id = str(row.get("scenario_id") or "")
        prior = representative_by_source.get(source_key)
        if prior is None or scenario_id < prior:
            representative_by_source[source_key] = scenario_id
    for row in rows:
        source_key = _effective_source_key(row)
        if (
            source_key
            and str(row.get("scenario_id") or "")
            != representative_by_source[source_key]
        ):
            duplicate = deepcopy(row)
            duplicate["status"] = "secondary_duplicate"
            duplicate["core_disposition"] = "secondary_duplicate"
            duplicate["duplicate_effective_source_key"] = source_key
            secondary.append(duplicate)
            continue
        primary.append(row)
    return primary, secondary


_ENVIRONMENT_REPAIR_SOURCE_BLOCKERS = frozenset(
    {
        "backend_formal_fidelity_not_allowed",
        "constructor_version_mismatch",
        "dependency_version_mismatch",
        "environment_pin_mismatch",
        "python_extra_missing",
        "required_source_file_missing",
    }
)
_ENVIRONMENT_RUNTIME_SOURCE_BLOCKERS = frozenset(
    {
        "native_runtime_missing",
        "optional_runtime",
        "runtime_unavailable",
        "sidecar_unavailable",
    }
)
_INTRINSIC_SOURCE_BLOCKERS = frozenset(
    {
        "backend_did_not_consume_declared_source",
        "controlled_source_intervention_no_effect",
        "source_hash_or_lineage_mismatch",
    }
)


def _source_consumption_contract_evidence(
    *,
    source_gate: dict[str, Any],
    agentic_contract: dict[str, Any],
) -> dict[str, Any]:
    """Return the most direct preserved source-consumption evidence."""
    direct = source_gate.get("source_consumption_contract")
    if isinstance(direct, dict) and direct:
        return deepcopy(direct)
    nested = (agentic_contract.get("agentic_contract") or {}).get(
        "source_consumption_contract"
    )
    if isinstance(nested, dict) and nested:
        return deepcopy(nested)
    compact = (source_gate.get("gate_evidence") or {}).get("source_consumption")
    return deepcopy(compact) if isinstance(compact, dict) else {}


def _source_consumption_blocker_sets(
    evidence: dict[str, Any],
) -> tuple[set[str], set[str]]:
    blockers = {str(code) for code in evidence.get("blockers") or []}
    raw_taxonomy = evidence.get("blocker_taxonomy")
    taxonomy = raw_taxonomy if isinstance(raw_taxonomy, dict) else {}
    environment = {
        blocker
        for blocker in blockers
        if blocker in _ENVIRONMENT_REPAIR_SOURCE_BLOCKERS
        or blocker in _ENVIRONMENT_RUNTIME_SOURCE_BLOCKERS
        or str(taxonomy.get(blocker) or "")
        in {
            "environment_repair",
            "environment_runtime",
            "implementation_repair",
        }
    }
    intrinsic = {
        blocker
        for blocker in blockers
        if blocker in _INTRINSIC_SOURCE_BLOCKERS
        or str(taxonomy.get(blocker) or "") == "intrinsic_content_failure"
    }
    return environment, intrinsic


def _rejection_disposition(
    *,
    agentic_blockers: set[str],
    environment_blockers: set[str],
    reason_codes: list[str],
    intrinsic_source_blockers: set[str] | None = None,
    behavioral_status: str | None = None,
) -> str:
    if environment_blockers.intersection(_ENVIRONMENT_REPAIR_SOURCE_BLOCKERS):
        return "held_repair"
    if environment_blockers.intersection(_ENVIRONMENT_RUNTIME_SOURCE_BLOCKERS):
        return "held_runtime"
    if environment_blockers:
        return "held_repair"
    if behavioral_status == "error":
        return "held_repair"
    if any(
        reason.rsplit(":", 1)[-1] in _ARTIFACT_IDENTITY_FAILURES
        for reason in reason_codes
    ):
        return "held_repair"
    intrinsic = {
        "deterministic_replay_failed",
        "task_contract_failed",
        "reference_headroom_nonpositive",
        "predesigned_event_unreachable",
    }
    if intrinsic_source_blockers or agentic_blockers.intersection(intrinsic):
        return "retired_intrinsic"
    joined = " ".join(reason_codes + sorted(agentic_blockers)).lower()
    if any(
        marker in joined
        for marker in (
            "runtime_unavailable",
            "sidecar_unavailable",
            "native_runtime_missing",
            "optional_runtime",
        )
    ):
        return "held_runtime"
    return "held_repair"


def _admission_fingerprint(
    row: dict[str, Any],
    *,
    implementation_tree_sha256: str,
) -> str:
    payload = {
        "scenario_id": row.get("scenario_id"),
        "scenario_signature": row.get("scenario_signature"),
        "source_denominator_key": _effective_source_key(row),
        "physical_source_key_or_lock": _physical_source_key(row),
        "implementation_tree_sha256": implementation_tree_sha256,
        "evaluation_semantics": REQUIRED_SEMANTICS,
        "difficulty_contract_version": DIFFICULTY_CONTRACT_VERSION,
        "behavioral": row.get("native_behavioral_validation"),
        "task": row.get("task_contract_validation"),
        "observed_depth": row.get("observed_depth_validation"),
        "strategy_depth": row.get("strategy_depth_validation"),
        "source_grounded": row.get("source_grounded_validation"),
        "agentic": row.get("agentic_contract"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _enrich_ledger(row: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    ledger = deepcopy(row.get("case_ledger") or {})
    backend = str(row.get("backend_kind") or "")
    if not ledger.get("independence_axis"):
        ledger["independence_axis"] = (
            "alibaba_trace_nonoverlapping_job_window"
            if backend == "alibaba_trace_sim"
            else _independence_axis(row)
        )
    if not ledger.get("decision_pressure_axis"):
        ledger["decision_pressure_axis"] = (
            "gpu_queue_sla_capacity_and_preemption_tradeoff"
            if backend == "alibaba_trace_sim"
            else _decision_pressure_axis(row, body)
        )
    perturbations = list(body.get("perturbations") or [])
    if not ledger.get("complexity_tags"):
        phase_ticks = (
            (body.get("backend_config") or {}).get("task_contract") or {}
        ).get("phase_ticks") or []
        ledger["complexity_tags"] = [
            f"difficulty={row.get('difficulty_level')}",
            f"horizon_ticks={int(body.get('horizon_ticks') or 0)}",
            f"perturbations={len(perturbations)}",
            (
                "hidden_perturbation"
                if any(bool(item.get("hidden")) for item in perturbations)
                else "fully_observed_or_static"
            ),
            f"task_phase_count={len(set(phase_ticks))}",
        ]
    if not ledger.get("keep_rationale"):
        denominator = (
            ledger.get("source_denominator_key")
            or row.get("source_key")
            or row.get("scenario_id")
        )
        ledger["keep_rationale"] = (
            f"Independent source denominator {denominator} retained after "
            "Protocol-2.0 task completion, terminal integrity, provenance, "
            "duplicate, and necessary-depth gates."
        )
    return ledger


def materialize(
    *,
    source: dict[str, Any],
    tasks: dict[str, Any],
    depth: dict[str, Any],
    behavioral: dict[str, Any] | None = None,
    observed_depth: dict[str, Any] | None = None,
    source_gate: dict[str, Any] | None = None,
    agentic_contract: dict[str, Any] | None = None,
    require_protocol21_gates: bool = False,
    input_bindings: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if require_protocol21_gates:
        return _materialize_protocol21(
            source=source,
            behavioral=behavioral,
            tasks=tasks,
            observed_depth=observed_depth,
            strategy_depth=depth,
            source_gate=source_gate,
            agentic_contract=agentic_contract,
            input_bindings=input_bindings or {},
        )
    if tasks.get("status") != "complete":
        raise ValueError("task-contract replay must be complete")
    task_by_id = {str(row["scenario_id"]): row for row in tasks.get("results") or []}
    depth_by_id = {str(row["scenario_id"]): row for row in depth.get("samples") or []}
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in source.get("scenarios") or []:
        scenario_id = str(raw["scenario_id"])
        task = task_by_id.get(scenario_id)
        if not (
            task
            and task.get("status") == "passed"
            and bool((task.get("terminal_integrity") or {}).get("release_ready"))
        ):
            rejected.append(
                {
                    "scenario_id": scenario_id,
                    "reason_code": "protocol2_reference_task_contract_failed",
                    "task_result": task,
                }
            )
            continue
        prior_depth = raw.get("strategy_depth_validation") or {}
        current_depth = depth_by_id.get(scenario_id)
        if str(prior_depth.get("status")) != "passed" and not (
            current_depth and current_depth.get("core_action") == "keep"
        ):
            rejected.append(
                {
                    "scenario_id": scenario_id,
                    "reason_code": "necessary_strategy_depth_not_proven",
                    "depth_result": current_depth or prior_depth,
                }
            )
            continue
        row = deepcopy(raw)
        row["case_ledger"] = _enrich_ledger(row, _body(row))
        row["task_contract_validation"] = {
            key: task.get(key)
            for key in (
                "status",
                "completed",
                "contract",
                "reason_code",
                "task_reason_code",
                "evidence",
                "terminal_integrity",
                "evaluation_protocol",
                "scoring_version",
                "agent_name",
                "reference_agents_attempted",
            )
        }
        if current_depth and current_depth.get("core_action") == "keep":
            row["strategy_depth_validation"] = {
                **current_depth,
                "status": "passed",
            }
        kept.append(row)
    return {
        "schema_version": "2.0",
        "status": "protocol2_core_candidate",
        "leaderboard_eligible": False,
        "n_source": len(source.get("scenarios") or []),
        "n_selected": len(kept),
        "n_rejected": len(rejected),
        "evaluation_implementation_fingerprint": tasks.get(
            "evaluation_implementation_fingerprint"
        ),
        "selection_policy": (
            "protocol2_task_terminal_provenance_duplicate_and_depth_fail_closed"
        ),
        "distribution": _distribution(kept),
        "rejected": sorted(rejected, key=lambda row: row["scenario_id"]),
        "scenarios": kept,
    }


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "samples"):
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
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in _rows(report):
        grouped.setdefault(row_identity(row), []).append(row)
    return grouped


def _identity_matches(
    source_row: dict[str, Any],
    result: dict[str, Any] | None,
) -> bool:
    return bool(
        result
        and str(result.get("scenario_id") or "")
        == str(source_row.get("scenario_id") or "")
        and str(result.get("scenario_signature") or "")
        == str(source_row.get("scenario_signature") or "")
    )


def _materialize_protocol21(
    *,
    source: dict[str, Any],
    behavioral: dict[str, Any] | None,
    tasks: dict[str, Any],
    observed_depth: dict[str, Any] | None,
    strategy_depth: dict[str, Any],
    source_gate: dict[str, Any] | None,
    agentic_contract: dict[str, Any] | None,
    input_bindings: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    constraint_key, constraints = extract_protocol21_selection_constraints(source)
    admission_profile = resolve_protocol21_admission_profile(source)
    task_outcome_is_diagnostic = admission_profile == QUALITY_CORE_V2_ADMISSION_PROFILE
    source_row_profile_mismatches = [
        str(row.get("scenario_id") or "")
        for row in source.get("scenarios") or []
        if isinstance(row, dict)
        if declared_protocol21_admission_profile(row) not in (None, admission_profile)
    ]
    if source_row_profile_mismatches:
        raise ValueError(
            "protocol-2.1 source row admission profile mismatch: "
            + ", ".join(sorted(source_row_profile_mismatches))
        )
    reports = {
        "behavioral": behavioral,
        "tasks": tasks,
        "observed_depth": observed_depth,
        "strategy_depth": strategy_depth,
        "source_gate": source_gate,
        "agentic_contract": agentic_contract,
    }
    missing = [name for name, report in reports.items() if report is None]
    if missing:
        raise ValueError("protocol-2.1 gates require: " + ", ".join(sorted(missing)))
    typed_reports = {
        name: report for name, report in reports.items() if isinstance(report, dict)
    }
    stale = [
        name
        for name, report in typed_reports.items()
        if _semantics(report) != REQUIRED_SEMANTICS
    ]
    if stale:
        raise ValueError(
            "protocol-2.1 gate semantics stale: " + ", ".join(sorted(stale))
        )
    incomplete = [
        name for name, report in typed_reports.items() if not _complete(report)
    ]
    if incomplete:
        raise ValueError(
            "protocol-2.1 gate reports incomplete: " + ", ".join(sorted(incomplete))
        )
    current_tree = implementation_identity()["implementation_tree_sha256"]
    tree_mismatches = [
        name
        for name, report in typed_reports.items()
        if report.get("implementation_tree_sha256") != current_tree
    ]
    if tree_mismatches:
        raise ValueError(
            "protocol-2.1 implementation tree mismatch: "
            + ", ".join(sorted(tree_mismatches))
        )
    profile_bound_reports = {
        name: typed_reports[name] for name in ("source_gate", "agentic_contract")
    }
    profile_mismatches = [
        name
        for name, report in profile_bound_reports.items()
        if resolve_protocol21_admission_profile(report) != admission_profile
    ]
    if profile_mismatches:
        raise ValueError(
            "protocol-2.1 admission profile mismatch: "
            + ", ".join(sorted(profile_mismatches))
        )
    row_profile_mismatches = [
        name
        for name, report in profile_bound_reports.items()
        if any(
            resolve_protocol21_admission_profile(row) != admission_profile
            for row in _rows(report)
        )
    ]
    if row_profile_mismatches:
        raise ValueError(
            "protocol-2.1 row admission profile mismatch: "
            + ", ".join(sorted(row_profile_mismatches))
        )

    indexes = {name: _index(report) for name, report in typed_reports.items()}
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in source.get("scenarios") or []:
        scenario_id = str(raw.get("scenario_id") or "")
        scenario_body = _body(raw)
        identity = row_identity(raw)
        gate_rows = {name: index.get(identity, []) for name, index in indexes.items()}
        failures: dict[str, list[str]] = {}
        lineage_blockers = validate_protocol21_row_lineage(raw)
        if not constraints:
            lineage_blockers.append("working_set_selection_constraints_missing")
        if lineage_blockers:
            failures["lineage"] = sorted(set(lineage_blockers))
        normalized: dict[str, dict[str, Any]] = {}
        for name, matches in gate_rows.items():
            if len(matches) != 1:
                report_rows = [row for rows in indexes[name].values() for row in rows]
                same_scenario_id = any(
                    str(row.get("scenario_id") or "") == scenario_id
                    for row in report_rows
                )
                if len(matches) > 1:
                    identity_failure = "artifact_identity_multiplicity"
                elif same_scenario_id:
                    identity_failure = "artifact_identity_mismatch"
                else:
                    identity_failure = "artifact_identity_missing"
                failures.setdefault(name, []).append(identity_failure)
                normalized[name] = matches[0] if matches else {}
            else:
                normalized[name] = matches[0]
        if normalized["behavioral"].get("status") != "passed":
            failures.setdefault("behavioral", []).append("behavioral_not_passed")
        task = normalized["tasks"]
        if not task_outcome_is_diagnostic and not (
            task.get("status") == "passed"
            and task.get("completed") is True
            and (task.get("terminal_integrity") or {}).get("release_ready") is True
        ):
            failures.setdefault("tasks", []).append("task_contract_not_passed")
        strategy = normalized["strategy_depth"]
        exact_strategy_required = requires_exact_strategy_minimality(
            profile=admission_profile
        )
        observed = normalized["observed_depth"]
        # High/Extreme observed-tick floors are the same paperwork family as
        # exact strategy minimality. quality_core_v2 keeps behavioral and
        # persistent native-control gates while treating task-outcome
        # duplication and tick-floor contradiction as diagnostics.
        if exact_strategy_required and (
            "contradicted" in str(observed.get("disposition") or "")
            or str(observed.get("decision") or "") == "retire"
        ):
            failures.setdefault("observed_depth", []).append("depth_contradicted")
        if exact_strategy_required and strategy.get("core_action") != "keep":
            failures.setdefault("strategy_depth", []).append("core_action_not_keep")
        difficulty = strategy.get("difficulty_calibration") or {}
        row_level = str(raw.get("difficulty_level") or "")
        body_level = str(scenario_body.get("difficulty_level") or "")
        if row_level != body_level:
            failures.setdefault("strategy_depth", []).append(
                "declared_difficulty_level_mismatch"
            )
        elif exact_strategy_required and not difficulty_calibration_matches_level(
            difficulty, row_level
        ):
            failures.setdefault("strategy_depth", []).append(
                "difficulty_calibration_not_current_or_matching"
            )
        grounded = normalized["source_gate"]
        if grounded.get("status") not in {
            "admitted",
            "admitted_for_core_review",
        }:
            failures.setdefault("source_gate", []).append(
                "source_ten_gate_not_admitted"
            )
        agentic = normalized["agentic_contract"]
        if agentic.get("status") != "passed":
            failures.setdefault("agentic_contract", []).append(
                "agentic_contract_not_passed"
            )
        if failures:
            reason_codes = sorted(
                f"{gate}:{reason}"
                for gate, reasons in failures.items()
                for reason in reasons
            )
            retired_codes = {str(code) for code in agentic.get("blockers") or []}
            source_consumption = _source_consumption_contract_evidence(
                source_gate=grounded,
                agentic_contract=agentic,
            )
            environment_blockers, intrinsic_source_blockers = (
                _source_consumption_blocker_sets(source_consumption)
            )
            if intrinsic_source_blockers:
                retired_codes.add("source_consumption_failed")
            elif source_consumption:
                retired_codes.discard("source_consumption_failed")
            disposition = _rejection_disposition(
                agentic_blockers=retired_codes,
                environment_blockers=environment_blockers,
                reason_codes=reason_codes,
                intrinsic_source_blockers=intrinsic_source_blockers,
                behavioral_status=str(normalized["behavioral"].get("status") or ""),
            )
            rejection = {
                "scenario_id": scenario_id,
                "scenario_signature": raw.get("scenario_signature"),
                "reason_code": reason_codes[0],
                "reason_codes": reason_codes,
                "failed_gates": failures,
                "disposition": disposition,
            }
            if source_consumption:
                rejection["source_consumption_contract"] = source_consumption
            rejected.append(rejection)
            continue

        row = deepcopy(raw)
        source_status = row.get("status")
        if source_status is not None:
            row["source_status_before_protocol21_admission"] = source_status
        row["status"] = "core_locked"
        row["core_disposition"] = "core_locked"
        row["protocol21_admission_status"] = "passed"
        for stale_field in (
            "native_behavioral_validation",
            "task_contract_validation",
            "strategy_depth_validation",
            "source_grounded_validation",
            "agentic_contract",
        ):
            row.pop(stale_field, None)
        row["native_behavioral_validation"] = deepcopy(normalized["behavioral"])
        row["task_contract_validation"] = deepcopy(normalized["tasks"])
        row["observed_depth_validation"] = deepcopy(normalized["observed_depth"])
        row["strategy_depth_validation"] = deepcopy(normalized["strategy_depth"])
        row["source_grounded_validation"] = deepcopy(normalized["source_gate"])
        row["agentic_contract"] = deepcopy(normalized["agentic_contract"])
        if row.get("seed") is None:
            row["seed"] = scenario_body.get("seed")
        if row.get("horizon_ticks") is None:
            row["horizon_ticks"] = scenario_body.get("horizon_ticks")
        row["case_ledger"] = _enrich_ledger(row, scenario_body)
        _finalize_admission_metadata(row)
        row["admission_fingerprint"] = _admission_fingerprint(
            row,
            implementation_tree_sha256=current_tree,
        )
        kept.append(row)

    kept, secondary = _partition_effective_source_duplicates(kept)

    source_rows = list(source.get("scenarios") or [])
    eligible_cells = sorted(
        {
            (
                str(row.get("family") or ""),
                str(row.get("difficulty_level") or ""),
            )
            for row in source_rows
        }
    )
    selected_cells = sorted(
        {
            (
                str(row.get("family") or ""),
                str(row.get("difficulty_level") or ""),
            )
            for row in kept
        }
    )
    domain_counts = Counter(str(row.get("domain") or "") for row in kept)
    backend_counts = Counter(str(row.get("backend_kind") or "") for row in kept)
    denominator = max(1, len(kept))
    max_domain_share = max(domain_counts.values(), default=0) / denominator
    max_backend_share = max(backend_counts.values(), default=0) / denominator
    source_keys = [_effective_source_key(row) for row in kept]
    physical_source_groups: dict[str, list[str]] = {}
    for row in kept:
        physical_key = _physical_asset_key(row)
        if physical_key:
            physical_source_groups.setdefault(physical_key, []).append(
                str(row.get("scenario_id") or "")
            )
    physical_duplicates = {
        key: sorted(values)
        for key, values in physical_source_groups.items()
        if len(values) > 1
    }
    effective_unique = (
        bool(source_keys)
        and all(source_keys)
        and len(source_keys) == len(set(source_keys))
    )
    constraint_validation = {
        "core_admission_profile": admission_profile,
        "eligible_family_difficulty_cells": eligible_cells,
        "selected_family_difficulty_cells": selected_cells,
        "preserve_each_eligible_family_difficulty_cell": (
            set(eligible_cells).issubset(selected_cells)
        ),
        "max_domain_share_actual": round(max_domain_share, 9),
        "max_backend_share_actual": round(max_backend_share, 9),
        "selection_constraints_present": bool(constraints),
        "selection_constraints_key": constraint_key,
        "max_domain_share_passed": bool(
            constraints.get("max_domain_share") is not None
            and max_domain_share <= float(constraints["max_domain_share"])
        ),
        "max_backend_share_passed": bool(
            constraints.get("max_backend_share") is not None
            and max_backend_share <= float(constraints["max_backend_share"])
        ),
        "effective_source_identity_unique": effective_unique,
        "quality_maximal_admission_passed": effective_unique,
        "distribution_constraints_are_diagnostic": True,
        "physical_source_duplicate_groups": physical_duplicates,
    }
    disposition_counts = Counter(
        ["core_locked"] * len(kept)
        + ["secondary_duplicate"] * len(secondary)
        + [str(row["disposition"]) for row in rejected]
    )
    incremental_freeze_ledger = [
        {
            "scenario_id": row["scenario_id"],
            "scenario_signature": row["scenario_signature"],
            "source_denominator_key": _effective_source_key(row),
            "disposition": "core_locked",
            "admission_fingerprint": row["admission_fingerprint"],
        }
        for row in kept
    ]
    output = {
        "schema_version": "2.1",
        "status": "protocol21_core_candidate",
        "formal_evaluation_ready": False,
        "leaderboard_eligible": False,
        "evaluation_protocol": {
            "version": REQUIRED_SEMANTICS["protocol_version"],
            "implementation_fingerprint": REQUIRED_SEMANTICS[
                "implementation_fingerprint"
            ],
        },
        "scoring_version": REQUIRED_SEMANTICS["scoring_version"],
        "input_bindings": input_bindings,
        "implementation_tree_sha256": current_tree,
        "selection_policy": "quality_maximal_v1",
        "n_source": len(source.get("scenarios") or []),
        "n_selected": len(kept),
        "n_rejected": len(rejected),
        "n_secondary": len(secondary),
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "incremental_freeze_ledger": incremental_freeze_ledger,
        "distribution": _distribution(kept),
        "constraint_validation": constraint_validation,
        "eligible_family_difficulty_cells": eligible_cells,
        "selected_family_difficulty_cells": selected_cells,
        "max_domain_share_actual": round(max_domain_share, 9),
        "max_backend_share_actual": round(max_backend_share, 9),
        "effective_source_identity_unique": effective_unique,
        "rejected": sorted(rejected, key=lambda row: row["scenario_id"]),
        "secondary": sorted(secondary, key=lambda row: row["scenario_id"]),
        "scenarios": kept,
    }
    if constraint_key is not None:
        output[constraint_key] = deepcopy(constraints)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--behavioral", type=Path)
    parser.add_argument("--observed-depth", type=Path)
    parser.add_argument("--source-gate", type=Path)
    parser.add_argument("--agentic-contract", type=Path)
    parser.add_argument("--require-protocol21-gates", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.require_protocol21_gates:
        missing = [
            flag
            for flag, path in (
                ("--behavioral", args.behavioral),
                ("--observed-depth", args.observed_depth),
                ("--source-gate", args.source_gate),
                ("--agentic-contract", args.agentic_contract),
            )
            if path is None
        ]
        if missing:
            parser.error("--require-protocol21-gates requires " + ", ".join(missing))
    paths = {
        "source_suite": args.source,
        "task_contracts": args.tasks,
        "strategy_depth": args.depth,
    }
    for name, path in (
        ("behavioral", args.behavioral),
        ("observed_depth", args.observed_depth),
        ("source_grounded", args.source_gate),
        ("agentic_contract", args.agentic_contract),
    ):
        if path is not None:
            paths[name] = path
    payload = materialize(
        source=json.loads(args.source.read_text(encoding="utf-8")),
        tasks=json.loads(args.tasks.read_text(encoding="utf-8")),
        depth=json.loads(args.depth.read_text(encoding="utf-8")),
        behavioral=(
            json.loads(args.behavioral.read_text(encoding="utf-8"))
            if args.behavioral
            else None
        ),
        observed_depth=(
            json.loads(args.observed_depth.read_text(encoding="utf-8"))
            if args.observed_depth
            else None
        ),
        source_gate=(
            json.loads(args.source_gate.read_text(encoding="utf-8"))
            if args.source_gate
            else None
        ),
        agentic_contract=(
            json.loads(args.agentic_contract.read_text(encoding="utf-8"))
            if args.agentic_contract
            else None
        ),
        require_protocol21_gates=args.require_protocol21_gates,
        input_bindings={name: artifact_binding(path) for name, path in paths.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in ("status", "n_source", "n_selected", "n_rejected")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
