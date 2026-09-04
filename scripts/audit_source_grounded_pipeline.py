#!/usr/bin/env python3
"""Audit source-grounded candidate envelopes without promoting held rows."""

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

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_admission import (  # noqa: E402
    declared_protocol21_admission_profile,
    resolve_protocol21_admission_profile,
)
from core.protocol21_evidence import (  # noqa: E402
    artifact_binding,
    canonicalize_repo_owned_paths,
    extract_semantics,
    required_semantics,
    row_identity,
)
from core.source_asset_contract import (  # noqa: E402
    resolve_source_asset_contract,
    virtual_source_identity_sha256,
)
from core.source_consumption_contract import (  # noqa: E402
    resolve_declared_sources,
)
from core.source_grounded_pipeline import (  # noqa: E402
    PIPELINE_VERSION,
    evaluate_source_grounded_candidate,
)
from domains.registry import get_backend_capability  # noqa: E402
from evaluation.scorer import SCORING_VERSION  # noqa: E402
from runner.episode import (  # noqa: E402
    EVALUATION_IMPLEMENTATION_FINGERPRINT,
    EVALUATION_PROTOCOL_VERSION,
)


_ENVIRONMENT_REPAIR_SOURCE_BLOCKERS = frozenset(
    {
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
_IMPLEMENTATION_REPAIR_SOURCE_BLOCKERS = frozenset(
    {"backend_formal_fidelity_not_allowed"}
)
_INTRINSIC_SOURCE_BLOCKERS = frozenset(
    {
        "backend_did_not_consume_declared_source",
        "controlled_source_intervention_no_effect",
        "source_hash_or_lineage_mismatch",
    }
)


def _source_consumption_contract_evidence(
    row: dict[str, Any],
) -> dict[str, Any]:
    """Preserve the runtime contract and add a blocker-level taxonomy."""
    if not row:
        return {}
    evidence = deepcopy(row)
    blockers = [str(code) for code in row.get("blockers") or []]
    raw_taxonomy = row.get("blocker_taxonomy")
    taxonomy = deepcopy(raw_taxonomy) if isinstance(raw_taxonomy, dict) else {}
    for blocker in blockers:
        if blocker in _ENVIRONMENT_REPAIR_SOURCE_BLOCKERS:
            category = "environment_repair"
        elif blocker in _ENVIRONMENT_RUNTIME_SOURCE_BLOCKERS:
            category = "environment_runtime"
        elif blocker in _IMPLEMENTATION_REPAIR_SOURCE_BLOCKERS:
            category = "implementation_repair"
        elif blocker in _INTRINSIC_SOURCE_BLOCKERS:
            category = "intrinsic_content_failure"
        else:
            category = "evidence_gap"
        taxonomy.setdefault(blocker, category)
    evidence["blockers"] = blockers
    evidence["blocker_taxonomy"] = taxonomy
    return evidence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_file(raw: str, *, repo_root: Path) -> Path | None:
    path = Path(raw)
    candidate = path if path.is_absolute() else repo_root / path
    return candidate.resolve() if candidate.is_file() else None


def _canonical_gate(name: str) -> str:
    return {
        "counterfactual": "counterfactual_replay",
        "independence": "source_independence",
    }.get(name, name)


def _semantics_current(report: dict[str, Any]) -> bool:
    raw = report.get("evaluation_semantics") or {}
    config = report.get("config") or {}
    return (
        str(
            raw.get("protocol_version")
            or report.get("evaluation_protocol_version")
            or config.get("evaluation_protocol_version")
            or ""
        )
        == EVALUATION_PROTOCOL_VERSION
        and str(
            raw.get("implementation_fingerprint")
            or report.get("evaluation_implementation_fingerprint")
            or config.get("evaluation_implementation_fingerprint")
            or ""
        )
        == EVALUATION_IMPLEMENTATION_FINGERPRINT
        and str(
            raw.get("scoring_version")
            or report.get("scoring_version")
            or config.get("scoring_version")
            or ""
        )
        == SCORING_VERSION
    )


def _report_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "samples"):
        rows = report.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _physical_control_identities(
    minimization: dict[str, Any],
    observed: dict[str, Any],
    control_tools: list[str],
) -> list[str]:
    """Select runtime-backed physical controls accepted by the backend."""
    candidates = list(
        minimization.get("one_minimal_physical_actuator_endpoint_set")
        or minimization.get(
            "successful_physical_actuator_endpoint_set_upper_bound"
        )
        or minimization.get("one_minimal_successful_tool_set")
        or minimization.get("successful_tool_set_upper_bound")
        or observed.get("observed_physical_actuator_endpoint_set")
        or observed.get("observed_state_changing_tool_set")
        or []
    )
    allowed = set(control_tools)
    return sorted(
        {
            str(identity)
            for identity in candidates
            if str(identity).split("|", 1)[0] in allowed
        }
    )


def _runtime_envelope(
    candidate: dict[str, Any],
    *,
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Attach only current replay evidence needed by the ten-gate audit."""
    required_reports = {
        "behavioral",
        "source_consumption",
        "task_contracts",
        "complexity",
        "strategy_depth",
    }
    if set(reports) != required_reports or not all(
        _semantics_current(value) for value in reports.values()
    ):
        return candidate
    scenario_id = str(candidate.get("scenario_id") or "")
    grouped = {
        name: [
            row
            for row in _report_rows(report)
            if str(row.get("scenario_id") or "") == scenario_id
            and str(row.get("scenario_signature") or "")
            == str(candidate.get("scenario_signature") or "")
        ]
        for name, report in reports.items()
    }
    behavior = (grouped.get("behavioral") or [{}])[0]
    source_consumption = (grouped.get("source_consumption") or [{}])[0]
    task = (grouped.get("task_contracts") or [{}])[0]
    strategy = (grouped.get("strategy_depth") or [{}])[0]
    task_agent = str(task.get("agent_name") or "oracle_offline")
    references = [
        row
        for row in grouped.get("complexity") or []
        if str(row.get("agent_name") or "") == task_agent
    ]
    reference = references[0] if references else {}
    observed = reference.get("observed") or {}
    minimization = reference.get("replay_minimization") or {}
    task_evidence = task.get("evidence") or {}
    graph = dict(observed.get("evidence_action_graph") or {})
    try:
        capability_contract = get_backend_capability(
            candidate.get("backend_kind")
        )
    except KeyError:
        capability_contract = None
    control_tools = list(
        capability_contract.control_tools if capability_contract else ()
    )
    required_tools = _physical_control_identities(
        minimization,
        observed,
        control_tools,
    )
    successful_tools = set(observed.get("observed_successful_tool_set") or [])
    observation_tools = sorted(successful_tools - set(control_tools))
    if graph:
        depth_proof_kinds = list(strategy.get("depth_proof_kinds") or [])
        exact_dependency_depth = (
            strategy.get("exact_task_dependency_depth")
            or minimization.get("exact_dependency_depth")
        )
        required_depth_lower_bound = (
            strategy.get("required_depth_lower_bound")
            or minimization.get("required_depth_lower_bound")
        )
        one_minimal_single_stage = bool(
            candidate.get("difficulty_level") in {"basic", "medium"}
            and minimization.get("status") == "one_minimal"
            and exact_dependency_depth is not None
            and (
                minimization.get("one_minimal_successful_tool_set")
                or minimization.get("one_minimal_physical_actuator_endpoint_set")
            )
        )
        dependency_depth_status = (
            "task_contract_ordered_milestone_lower_bound"
            if "task_contract_ordered_milestone_lower_bound" in depth_proof_kinds
            else (
                "one_minimal_single_stage_action_dag"
                if one_minimal_single_stage
                else "declared_evidence_action_dag"
            )
        )
        graph.update(
            {
                "successful_reference": task.get("completed") is True,
                "required_tools": required_tools,
                "exact_dependency_depth": exact_dependency_depth,
                "required_depth_lower_bound": required_depth_lower_bound,
                "depth_proof_kinds": depth_proof_kinds,
                "dependency_depth_status": dependency_depth_status,
                # Difficulty contracts require observed physical strategy
                # changes. A harness-level commit_to_plan revision remains
                # useful diagnostic evidence, but is not a substitute for a
                # native control-policy switch.
                "plan_reversal_count": int(
                    observed.get("control_strategy_switch_count") or 0
                ),
                "explicit_plan_revision_count": int(
                    observed.get("explicit_plan_revision_count") or 0
                ),
            }
        )
    replay_evidence = behavior.get("replay_evidence") or {}
    wait_loss = task_evidence.get("counterfactual_task_loss")
    reference_loss = task_evidence.get("actual_task_loss")
    counterfactual = reference.get("counterfactual") or {}
    enriched = deepcopy(candidate)
    identity_binding_allowed = (
        (candidate.get("domain_boundary") or {}).get("allowed") is not False
    )
    enriched["domain_boundary"] = {
        "classification": candidate.get("domain"),
        "allowed": bool(control_tools) and identity_binding_allowed,
    }
    enriched["capability"] = {
        "native_state_fields": list(
            capability_contract.native_state_fields
            if capability_contract
            else ()
        ),
        "observation_tools": list(
            capability_contract.observation_tools
            if capability_contract
            else observation_tools
        ),
        "control_tools": control_tools,
        "clock_semantics": (
            capability_contract.clock_semantics
            if capability_contract
            else None
        ),
        "deterministic_seed": True,
        "counterfactual_reset": bool(counterfactual),
        "adaptive_recovery_signal": (
            "runtime_native_operational_burden"
            if counterfactual
            else None
        ),
    }
    enriched["replay"] = {
        **replay_evidence,
        "reference_task_completed": task.get("completed") is True,
        "wait_task_loss": wait_loss,
        "reference_task_loss": reference_loss,
        "material_headroom": dict(task.get("material_headroom") or {}),
        "counterfactual_supported": bool(counterfactual),
    }
    source = dict(enriched.get("source") or {})
    source["consumed_by_backend"] = (
        True
        if source_consumption.get("status") == "passed"
        else (
            False
            if source_consumption.get("status") == "failed"
            else None
        )
    )
    source["consumed_fields"] = list(
        source_consumption.get("derived_backend_state_fields") or []
    )
    enriched["source"] = source
    enriched["_source_consumption_contract"] = (
        _source_consumption_contract_evidence(source_consumption)
    )
    enriched["decision_graph"] = graph
    enriched["difficulty_proof"] = {
        "contract_passed": strategy.get("core_action") == "keep",
        "minimality_status": minimization.get("status"),
        "required_depth_lower_bound": strategy.get("required_depth_lower_bound"),
        "depth_proof_kinds": list(strategy.get("depth_proof_kinds") or []),
    }
    return enriched


def _rebind_candidate(
    candidate: dict[str, Any],
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], Path | None, dict[str, str]]:
    """Bind source-lock fields to current YAML and source file bytes.

    Runtime gates remain fail-closed: this helper intentionally does not turn
    legacy ``candidate_gate`` or ``core_admission_review`` status into current
    evidence.
    """
    rebound = deepcopy(candidate)
    require_consumption_proof = False
    scenario_raw = str(candidate.get("path") or "")
    scenario_path = (
        _resolved_file(scenario_raw, repo_root=repo_root)
        if scenario_raw
        else None
    )
    if scenario_path is None:
        raw_source_files = list(
            (candidate.get("source") or {}).get("files")
            or candidate.get("provenance_files")
            or []
        )
    else:
        body = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        if not isinstance(body, dict):
            body = {}
        provenance = body.get("provenance") or {}
        declared_source_files, _declared_hashes, _missing = (
            resolve_declared_sources(body, repo_root=repo_root)
        )
        source_asset_contract = resolve_source_asset_contract(
            body,
            repo_root=repo_root,
        )
        raw_source_files = list(
            declared_source_files
            or provenance.get("files")
            or candidate.get("provenance_files")
            or (candidate.get("source") or {}).get("files")
            or []
        )
        if provenance:
            require_consumption_proof = True
            consumption_proof = candidate.get("source_consumption_proof") or {}
            current_consumption_proof = (
                _semantics_current(consumption_proof)
                and consumption_proof.get("status") == "passed"
            )
            rebound["source"] = {
                "dataset_id": (
                    provenance.get("dataset_id") or provenance.get("data_source")
                ),
                "files": raw_source_files,
                "url": provenance.get("url"),
                "version_lock": (
                    provenance.get("commit") or provenance.get("source_release")
                ),
                "license": (
                    provenance.get("license") or provenance.get("license_id")
                ),
                # Constructor-backed networks have no local file bytes. Keep
                # the strict, whitelisted constructor/version digest as the
                # locked derivation window when rebinding the candidate.
                "window_sha256": source_asset_contract.derived_window_sha256,
                # Source consumption is runtime evidence, not provenance
                # metadata. A legacy nested boolean cannot promote a row.
                "consumed_by_backend": (
                    consumption_proof.get("consumed_by_backend")
                    if current_consumption_proof
                    else None
                ),
                "consumed_fields": list(
                    (
                        consumption_proof.get("consumed_fields")
                        if current_consumption_proof
                        else []
                    )
                    or []
                ),
            }
        identity_matches = all(
            not body.get(field)
            or str(body.get(field)) == str(candidate.get(field))
            for field in (
                "scenario_id",
                "scenario_signature",
                "domain",
                "backend_kind",
                "difficulty_level",
            )
        )
        boundary = dict(rebound.get("domain_boundary") or {})
        if not identity_matches:
            boundary["allowed"] = False
        rebound["domain_boundary"] = boundary

    source_file_hashes: dict[str, str] = {}
    for raw in raw_source_files:
        resolved = _resolved_file(str(raw), repo_root=repo_root)
        if resolved is not None:
            source_file_hashes[str(raw)] = _sha256(resolved)
            continue
        # Constructor-backed networks have no local file to hash, but they do
        # have a strict, versioned identity digest. Keep that digest in the
        # same binding map as file-backed inputs so the source-grounded report
        # and the readiness audit compare identical locks.
        virtual_hash = virtual_source_identity_sha256(str(raw))
        if virtual_hash is not None:
            source_file_hashes[str(raw)] = virtual_hash
    consumption_proof = candidate.get("source_consumption_proof") or {}
    if (
        require_consumption_proof
        and
        (rebound.get("source") or {}).get("consumed_by_backend") is True
        and consumption_proof.get("source_file_hashes")
        != source_file_hashes
    ):
        source = dict(rebound.get("source") or {})
        source["consumed_by_backend"] = None
        source["consumed_fields"] = []
        rebound["source"] = source
    if raw_source_files and len(source_file_hashes) == len(raw_source_files):
        digest = hashlib.sha256()
        for raw, sha256 in sorted(source_file_hashes.items()):
            digest.update(raw.encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256.encode("ascii"))
            digest.update(b"\0")
        source = dict(rebound.get("source") or {})
        source["window_sha256"] = digest.hexdigest()
        rebound["source"] = source
    return rebound, scenario_path, source_file_hashes


def summarize_candidates(
    candidates: list[dict[str, Any]],
    *,
    source_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
    reports: dict[str, dict[str, Any]] | None = None,
    input_bindings: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    admission_profiles = {
        resolve_protocol21_admission_profile(row) for row in candidates
    }
    if len(admission_profiles) != 1:
        raise ValueError("candidate admission profiles must be uniform")
    admission_profile = next(iter(admission_profiles))
    rebound = [
        _rebind_candidate(row, repo_root=repo_root)
        for row in candidates
    ]
    structural_counts = Counter(
        str(row.get("structural_fingerprint") or "")
        for row, _, _ in rebound
        if row.get("structural_fingerprint")
    )
    prepared = []
    for row, scenario_path, source_hashes in rebound:
        if (
            row.get("structural_fingerprint")
            and row.get("semantic_fingerprint")
        ):
            row["independence"] = {
                "structural_fingerprint": row["structural_fingerprint"],
                "semantic_fingerprint": row["semantic_fingerprint"],
                "is_duplicate": (
                    structural_counts[str(row["structural_fingerprint"])] > 1
                ),
            }
        prepared.append(
            (
                _runtime_envelope(row, reports=reports or {}),
                scenario_path,
                source_hashes,
            )
        )
    evaluated = sorted(
        (
            (
                evaluate_source_grounded_candidate(row),
                row,
                scenario_path,
                source_hashes,
            )
            for row, scenario_path, source_hashes in prepared
        ),
        key=lambda item: row_identity(item[1]),
    )
    results: list[dict[str, Any]] = []
    for result, candidate, scenario_path, source_file_hashes in evaluated:
        source_consumption_contract = deepcopy(
            candidate.get("_source_consumption_contract") or {}
        )
        if source_consumption_contract:
            result = deepcopy(result)
            result["source_consumption_contract"] = source_consumption_contract
            gate_evidence = deepcopy(result.get("gate_evidence") or {})
            gate_evidence["source_consumption"] = {
                "status": source_consumption_contract.get("status"),
                "blockers": list(
                    source_consumption_contract.get("blockers") or []
                ),
                "blocker_taxonomy": deepcopy(
                    source_consumption_contract.get("blocker_taxonomy") or {}
                ),
            }
            result["gate_evidence"] = gate_evidence
        gates = result.get("gates") or {}
        passed_gates = sorted(
            _canonical_gate(str(name))
            for name, passed in gates.items()
            if passed is True
        )
        failed_gates = sorted(
            _canonical_gate(str(name))
            for name, passed in gates.items()
            if passed is not True
        )
        results.append(
            {
                **result,
                "scenario_signature": candidate.get("scenario_signature"),
                "path": candidate.get("path"),
                "status": (
                    "admitted"
                    if result["status"] == "admitted_for_core_review"
                    else "held"
                ),
                "passed_gates": passed_gates,
                "failed_gates": failed_gates,
                "source_file_hashes": source_file_hashes,
                "scenario_file_sha256": (
                    _sha256(scenario_path) if scenario_path is not None else None
                ),
            }
        )
    counts = Counter(str(row["status"]) for row, _, _, _ in evaluated)
    source_binding = (
        str(source_path.absolute()) if source_path is not None else None
    )
    tree_hash = implementation_identity()["implementation_tree_sha256"]
    reports_complete = bool(reports) and all(
        extract_semantics(report) == required_semantics()
        and (
            report.get("status") == "complete"
            or report.get("complete") is True
        )
        and report.get("implementation_tree_sha256") == tree_hash
        for report in reports.values()
    )
    identities = [row_identity(row) for row, _, _ in prepared]
    unique_identities = (
        len(identities) == len(set(identities))
        and len({identity[0] for identity in identities}) == len(identities)
    )
    report = {
        "schema_version": "2.1",
        "admission_profile": admission_profile,
        "pipeline_version": PIPELINE_VERSION,
        "status": (
            "complete"
            if reports_complete and unique_identities and len(results) == len(prepared)
            else "partial"
        ),
        "evaluation_semantics": {
            "protocol_version": EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": (
                EVALUATION_IMPLEMENTATION_FINGERPRINT
            ),
            "scoring_version": SCORING_VERSION,
        },
        "source_artifact": source_binding,
        "source_artifact_sha256": (
            _sha256(source_path)
            if source_path is not None and source_path.is_file()
            else None
        ),
        "input_bindings": input_bindings or {},
        "implementation_tree_sha256": tree_hash,
        "n_expected": len(results),
        "n_completed": len(results),
        "n_admitted": counts["admitted_for_core_review"],
        "n_held": counts["held"],
        "results": results,
        "counts": {
            "admitted_for_core_review": counts["admitted_for_core_review"],
            "held": counts["held"],
            "total": len(results),
        },
        "admitted_for_core_review": [
            row
            for row in results
            if row["status"] == "admitted"
        ],
        "held": [row for row in results if row["status"] == "held"],
    }
    return canonicalize_repo_owned_paths(report, repo_root=repo_root)


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    admission_profile: str | None = None
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        admission_profile = resolve_protocol21_admission_profile(payload)
        rows = payload.get("candidates")
        if rows is None:
            rows = payload.get("scenarios")
    else:
        rows = None
    if not isinstance(rows, list) or not all(
        isinstance(row, dict) for row in rows
    ):
        raise ValueError(
            "input must be a JSON list, {'candidates': [...]}, or "
            "{'scenarios': [...]} object"
        )
    if admission_profile is None or admission_profile == "strict_v1":
        return rows
    prepared = []
    for row in rows:
        declared_profile = declared_protocol21_admission_profile(row)
        if declared_profile not in (None, admission_profile):
            raise ValueError(
                "candidate row admission profile conflicts with source suite: "
                f"{row.get('scenario_id')}"
            )
        candidate = deepcopy(row)
        candidate["core_admission_profile"] = admission_profile
        prepared.append(candidate)
    return prepared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--behavioral", type=Path)
    parser.add_argument("--source-consumption", type=Path)
    parser.add_argument("--task-contracts", type=Path)
    parser.add_argument("--complexity", type=Path)
    parser.add_argument("--strategy-depth", type=Path)
    parser.add_argument("--require-protocol21-evidence", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report_paths = {
        name: path
        for name, path in (
            ("behavioral", args.behavioral),
            ("source_consumption", args.source_consumption),
            ("task_contracts", args.task_contracts),
            ("complexity", args.complexity),
            ("strategy_depth", args.strategy_depth),
        )
        if path is not None
    }
    required_names = {
        "behavioral",
        "source_consumption",
        "task_contracts",
        "complexity",
        "strategy_depth",
    }
    if args.require_protocol21_evidence and set(report_paths) != required_names:
        parser.error(
            "--require-protocol21-evidence requires behavioral, "
            "source-consumption, task-contracts, complexity, and strategy-depth"
        )
    tree_hash = implementation_identity()["implementation_tree_sha256"]
    report = summarize_candidates(
        _load_candidates(args.input),
        source_path=args.input,
        reports={
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in report_paths.items()
        },
        input_bindings={
            "source_suite": artifact_binding(
                args.input,
                implementation_tree_sha256=tree_hash,
            ),
            **{
                name: artifact_binding(
                    path,
                    implementation_tree_sha256=tree_hash,
                )
                for name, path in report_paths.items()
            },
        },
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    else:
        print(encoded, end="")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
