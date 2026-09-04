"""Row-level Protocol-2.1 admission evidence and selective invalidation."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from core.protocol21_evidence import (
    artifact_binding,
    extract_semantics,
    report_rows,
    required_semantics,
)
from core.protocol21_qualification import (
    validate_row_qualification_evidence,
)
from domains.registry import (
    get_backend_capability,
    get_domain_spec,
    resolve_backend_source_contract_builder,
    resolve_backend_source_evidence_adapter,
)
from evaluation.leaderboard import PRIMARY_LEADERBOARD_FORMULA_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
ADMISSION_LEDGER_SCHEMA_VERSION = "1.0"
ADMISSION_DEPENDENCY_SCHEMA_VERSION = "1.0"

GATE_ORDER = (
    "lineage",
    "behavioral",
    "source_consumption",
    "task_contract",
    "complexity_headroom",
    "observed_depth",
    "strategy_depth",
    "source_grounded",
    "agentic_contract",
    "qualification",
)
DOMAIN_FREEZE_GATES = GATE_ORDER[:-1]
RUNTIME_GATES = GATE_ORDER[1:-1]
SCORING_GATES = GATE_ORDER[3:]
FORMULA_GATES = ("qualification",)

_GATE_IMPLEMENTATION_PATHS = {
    "lineage": ("core/suite_identity.py",),
    "behavioral": (
        "scripts/calibrate_core_candidate.py",
        "runner/episode.py",
    ),
    "source_consumption": (
        "scripts/audit_protocol21_source_consumption.py",
        "core/source_consumption_contract.py",
    ),
    "task_contract": (
        "scripts/calibrate_task_contracts.py",
        "evaluation/task_completion.py",
        "evaluation/scorer.py",
    ),
    "complexity_headroom": (
        "scripts/calibrate_core_complexity.py",
        "evaluation/scorer.py",
    ),
    "observed_depth": ("scripts/audit_observed_reference_depth.py",),
    "strategy_depth": ("scripts/audit_strategy_depth_calibration.py",),
    "source_grounded": ("scripts/audit_source_grounded_pipeline.py",),
    "agentic_contract": (
        "scripts/audit_protocol21_core_contract.py",
        "core/agentic_core_contract.py",
    ),
    "qualification": (
        "core/protocol21_qualification.py",
        "scripts/run_protocol21_core_pilot.py",
        "scripts/build_protocol21_core_readiness.py",
        "evaluation/leaderboard.py",
    ),
}

_POSITIVE_DISPOSITIONS = {
    "admitted",
    "bounded_replay_required",
    "keep",
    "passed",
    "required_depth_lower_bound_met",
}
_DETERMINISTIC_RETIREMENT_REASONS = {
    "backend_formal_fidelity_not_allowed",
}


def row_prequalification_evidence_complete(
    row: Mapping[str, Any],
) -> bool:
    """Return whether one ledger row has every current prequalification gate."""
    statuses = row.get("gate_statuses")
    return bool(
        row.get("prequalification_ready") is True
        and isinstance(statuses, Mapping)
        and all(statuses.get(gate) == "passed" for gate in DOMAIN_FREEZE_GATES)
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _resolve(path: str | Path, *, repo_root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repo_root / value).resolve()


def _relative_or_absolute(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _file_bindings(
    paths: Iterable[Path],
    *,
    repo_root: Path,
) -> list[dict[str, str]]:
    bindings = []
    for path in sorted(set(paths), key=lambda item: str(item)):
        resolved = path.resolve()
        if not resolved.is_file():
            continue
        bindings.append(
            {
                "path": _relative_or_absolute(resolved, repo_root=repo_root),
                "sha256": _sha256_file(resolved),
            }
        )
    return bindings


def gate_dependency_fingerprint(
    gate: str,
    *,
    row: Mapping[str, Any],
    dependency_paths: Iterable[Path],
    input_artifact_sha256: str,
    row_evidence_digest: str,
    protocol_version: str,
    scoring_version: str,
    primary_formula_version: str,
    repo_root: Path = REPO_ROOT,
    upstream_fingerprints: Mapping[str, str] | None = None,
    dependency_file_bindings: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Fingerprint only the concrete dependencies of one admission gate."""
    if gate not in GATE_ORDER:
        raise ValueError(f"unknown admission gate: {gate}")
    files = (
        dependency_file_bindings
        if dependency_file_bindings is not None
        else _file_bindings(dependency_paths, repo_root=repo_root)
    )
    payload: dict[str, Any] = {
        "dependency_schema_version": ADMISSION_DEPENDENCY_SCHEMA_VERSION,
        "gate": gate,
        "scenario_id": str(row.get("scenario_id") or ""),
        "scenario_signature": str(row.get("scenario_signature") or ""),
        "structural_fingerprint": str(
            row.get("structural_fingerprint") or ""
        ),
        "semantic_fingerprint": str(row.get("semantic_fingerprint") or ""),
        "dependency_files": files,
        "input_artifact_sha256": input_artifact_sha256,
        "row_evidence_digest": row_evidence_digest,
        "protocol_version": protocol_version,
        "upstream_fingerprints": dict(sorted((upstream_fingerprints or {}).items())),
    }
    if gate in SCORING_GATES:
        payload["scoring_version"] = scoring_version
    if gate in FORMULA_GATES:
        payload["primary_leaderboard_formula_version"] = primary_formula_version
    return {
        "sha256": _sha256_bytes(_canonical_json(payload)),
        "payload": payload,
    }


def _module_path(module_name: str, *, repo_root: Path) -> Path | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.origin:
        return None
    path = Path(spec.origin).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _callable_path(value: Any, *, repo_root: Path) -> Path | None:
    raw = inspect.getsourcefile(value)
    if not raw:
        return None
    path = Path(raw).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def _backend_dependency_paths(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    errors: list[str] = []
    try:
        domain = get_domain_spec(str(row.get("domain") or ""))
        adapter = _module_path(domain.adapter_module, repo_root=repo_root)
        if adapter is None:
            errors.append("backend_dependency_paths_unresolved")
        else:
            paths.append(adapter)
        capability = get_backend_capability(str(row.get("backend_kind") or ""))
        for resolver in (
            resolve_backend_source_contract_builder,
            resolve_backend_source_evidence_adapter,
        ):
            resolved = _callable_path(resolver(capability), repo_root=repo_root)
            if resolved is None:
                errors.append("backend_dependency_paths_unresolved")
            else:
                paths.append(resolved)
    except (ImportError, KeyError, TypeError):
        errors.append("backend_dependency_paths_unresolved")
    return sorted(set(paths)), sorted(set(errors))


def _scenario_document(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Path | None, dict[str, Any]]:
    raw = str(row.get("path") or "")
    if not raw:
        return None, {}
    path = _resolve(raw, repo_root=repo_root)
    if not path.is_file():
        return path, {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return path, {}
    return path, payload if isinstance(payload, dict) else {}


def _source_asset_paths(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[list[Path], list[str]]:
    scenario_path, document = _scenario_document(row, repo_root=repo_root)
    paths: list[Path] = []
    errors: list[str] = []
    if scenario_path is not None:
        if scenario_path.is_file():
            paths.append(scenario_path)
        else:
            errors.append("scenario_dependency_path_unresolved")
    explicit: list[Any] = []
    contract = document.get("source_contract") or {}
    if isinstance(contract, dict):
        for key in (
            "runtime_input",
            "derivation_input",
            "implementation_asset",
            "metadata",
            "license",
        ):
            values = contract.get(key) or []
            if isinstance(values, list):
                explicit.extend(values)
    provenance = document.get("provenance") or {}
    if isinstance(provenance, dict):
        values = provenance.get("files") or []
        if isinstance(values, list):
            explicit.extend(values)
    row_paths = row.get("source_asset_paths") or []
    if isinstance(row_paths, list):
        explicit.extend(row_paths)
    for value in explicit:
        if not isinstance(value, str) or not value.strip():
            continue
        path = _resolve(value, repo_root=repo_root)
        if path.is_file():
            paths.append(path)
        else:
            errors.append("source_asset_dependency_path_unresolved")
    return sorted(set(paths)), sorted(set(errors))


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _index_report(report: Mapping[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in report_rows(dict(report)):
        grouped[_identity(row)].append(row)
    return dict(grouped)


def _index_qualification_report(
    report: Mapping[str, Any],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in report.get("rows") or []:
        if isinstance(row, dict):
            grouped[_identity(row)].append(row)
    return dict(grouped)


def _row_digest(rows: list[dict[str, Any]]) -> str:
    return _sha256_bytes(_canonical_json(rows)) if rows else ""


def _row_reason_codes(rows: list[dict[str, Any]]) -> list[str]:
    reasons: set[str] = set()
    for row in rows:
        for key in ("reason_code", "disposition", "core_action"):
            value = row.get(key)
            if isinstance(value, str) and value and value not in _POSITIVE_DISPOSITIONS:
                reasons.add(value)
        for key in ("reason_codes", "blockers", "failures", "violations"):
            values = row.get(key) or []
            if isinstance(values, list):
                reasons.update(str(value) for value in values if value)
    return sorted(reasons)


def _gate_passed(gate: str, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    if gate == "qualification":
        return all(
            row.get("qualification_status") == "passed"
            for row in rows
        )
    if gate == "complexity_headroom":
        statuses = [
            str(row.get("status") or "")
            for row in rows
        ]
        return all(status in {"complete", "passed"} for status in statuses)
    for row in rows:
        status = str(row.get("status") or "")
        disposition = str(row.get("disposition") or "")
        core_action = str(row.get("core_action") or "")
        if status in {"failed", "error", "partial", "blocked"}:
            return False
        if status in {"admitted", "passed"}:
            continue
        if disposition in _POSITIVE_DISPOSITIONS:
            continue
        if core_action == "keep":
            continue
        if row.get("passed") is True or row.get("release_eligible") is True:
            continue
        if gate == "observed_depth" and disposition == "bounded_replay_required":
            continue
        return False
    return True


def _physical_identity(row: Mapping[str, Any]) -> Any:
    direct = row.get("physical_source_key_or_lock") or row.get(
        "physical_source_key"
    )
    ledger = row.get("case_ledger") or {}
    nested = ledger.get("physical_source_lock") if isinstance(ledger, dict) else None
    return direct or nested


def _denominator(row: Mapping[str, Any]) -> str:
    direct = row.get("source_denominator_key")
    ledger = row.get("case_ledger") or {}
    nested = ledger.get("source_denominator_key") if isinstance(ledger, dict) else None
    return str(direct or nested or "")


def _load_optional(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _retirement_index(
    report: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {_identity(row): row for row in report_rows(dict(report))}


def _previous_index(
    report: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        _identity(row): row
        for row in report.get("rows") or []
        if isinstance(row, dict)
    }


def _dependency_paths_for_gate(
    gate: str,
    *,
    row: Mapping[str, Any],
    repo_root: Path,
    backend_dependencies: tuple[list[Path], list[str]] | None = None,
    source_dependencies: tuple[list[Path], list[str]] | None = None,
) -> tuple[list[Path], list[str]]:
    paths = [
        repo_root / relative
        for relative in _GATE_IMPLEMENTATION_PATHS.get(gate, ())
    ]
    missing = [
        "gate_dependency_paths_unresolved"
        for path in paths
        if not path.is_file()
    ]
    paths = [path for path in paths if path.is_file()]
    backend_paths, backend_errors = (
        backend_dependencies
        if backend_dependencies is not None
        else _backend_dependency_paths(row, repo_root=repo_root)
    )
    source_paths, source_errors = (
        source_dependencies
        if source_dependencies is not None
        else _source_asset_paths(row, repo_root=repo_root)
    )
    if gate in RUNTIME_GATES:
        paths.extend(backend_paths)
        missing.extend(backend_errors)
    if gate == "lineage" or gate in GATE_ORDER[2:]:
        paths.extend(source_paths)
        missing.extend(source_errors)
    return sorted(set(paths)), sorted(set(missing))


def _artifact_is_current(report: Mapping[str, Any]) -> bool:
    if not report:
        return False
    return extract_semantics(dict(report)) == required_semantics() and (
        report.get("status") == "complete" or report.get("complete") is True
    )


def _gate_report_name(gate: str) -> str:
    return {
        "complexity_headroom": "complexity_headroom",
        "task_contract": "task_contract",
    }.get(gate, gate)


def build_admission_ledger(
    *,
    source_suite_path: Path,
    evidence_paths: Mapping[str, Path],
    qualification_path: Path | None = None,
    retirement_ledger_path: Path | None = None,
    previous_ledger_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build a fail-closed admission ledger over the entire source suite."""
    repo_root = repo_root.resolve()
    source_suite = json.loads(source_suite_path.read_text(encoding="utf-8"))
    source_rows = [
        row
        for row in source_suite.get("scenarios") or []
        if isinstance(row, dict)
    ]
    reports: dict[str, dict[str, Any]] = {
        gate: _load_optional(path)
        for gate, path in evidence_paths.items()
    }
    if qualification_path is not None:
        reports["qualification"] = _load_optional(qualification_path)
    report_indexes = {
        gate: (
            _index_qualification_report(report)
            if gate == "qualification"
            else _index_report(report)
        )
        for gate, report in reports.items()
    }
    retirements = _load_optional(retirement_ledger_path)
    retirement_index = _retirement_index(retirements)
    previous = _load_optional(previous_ledger_path)
    previous_index = _previous_index(previous)
    semantics = {
        **required_semantics(),
        "primary_leaderboard_formula_version": (
            PRIMARY_LEADERBOARD_FORMULA_VERSION
        ),
    }

    external_summary = Counter(
        {
            "backend_formal_fidelity_not_allowed_current": 0,
            "stale_runtime_retirements_needing_replay": 0,
            "blindly_reused_unfingerprinted_retirements": 0,
        }
    )
    for retired in report_rows(retirements):
        reason = str(retired.get("reason_code") or "")
        if reason == "backend_formal_fidelity_not_allowed":
            try:
                current = not get_backend_capability(
                    str(retired.get("backend_kind") or "")
                ).formal_core_allowed
            except KeyError:
                current = False
            if current:
                external_summary[
                    "backend_formal_fidelity_not_allowed_current"
                ] += 1
        elif reason:
            external_summary["stale_runtime_retirements_needing_replay"] += 1

    rows: list[dict[str, Any]] = []
    file_binding_cache: dict[Path, dict[str, str]] = {}
    for source_row in source_rows:
        identity = _identity(source_row)
        previous_row = previous_index.get(identity) or {}
        prior_retirement = retirement_index.get(identity) or {}
        gate_evidence: dict[str, Any] = {}
        gate_fingerprints: dict[str, str] = {}
        gate_fingerprint_details: dict[str, Any] = {}
        gate_statuses: dict[str, str] = {}
        reused_gates: list[str] = []
        invalidated_gates: list[str] = []
        unverifiable_gates: list[str] = []
        blockers: set[str] = set()
        upstream: dict[str, str] = {}
        backend_dependencies = _backend_dependency_paths(
            source_row,
            repo_root=repo_root,
        )
        source_dependencies = _source_asset_paths(
            source_row,
            repo_root=repo_root,
        )

        for gate in GATE_ORDER:
            report = reports.get(_gate_report_name(gate), {})
            indexed = report_indexes.get(_gate_report_name(gate), {})
            evidence_rows = indexed.get(identity, [])
            qualification_errors = (
                validate_row_qualification_evidence(report)
                if gate == "qualification" and report
                else []
            )
            current = (
                not qualification_errors
                if gate == "qualification"
                else _artifact_is_current(report)
            )
            dependency_paths, dependency_errors = _dependency_paths_for_gate(
                gate,
                row=source_row,
                repo_root=repo_root,
                backend_dependencies=backend_dependencies,
                source_dependencies=source_dependencies,
            )
            dependency_file_bindings = []
            for dependency_path in dependency_paths:
                resolved = dependency_path.resolve()
                binding = file_binding_cache.get(resolved)
                if binding is None:
                    binding = {
                        "path": _relative_or_absolute(
                            resolved,
                            repo_root=repo_root,
                        ),
                        "sha256": _sha256_file(resolved),
                    }
                    file_binding_cache[resolved] = binding
                dependency_file_bindings.append(binding)
            blockers.update(dependency_errors)
            artifact_sha = ""
            binding: dict[str, Any] | None = None
            path = (
                qualification_path
                if gate == "qualification"
                else evidence_paths.get(_gate_report_name(gate))
            )
            if path is not None and path.is_file():
                artifact_sha = _sha256_file(path)
                binding = artifact_binding(path)
            row_digest = _row_digest(evidence_rows)
            fingerprint = gate_dependency_fingerprint(
                gate,
                row=source_row,
                dependency_paths=dependency_paths,
                input_artifact_sha256=artifact_sha,
                row_evidence_digest=row_digest,
                protocol_version=semantics["protocol_version"],
                scoring_version=(
                    extract_semantics(report).get("scoring_version")
                    or semantics["scoring_version"]
                ),
                primary_formula_version=(
                    semantics["primary_leaderboard_formula_version"]
                ),
                repo_root=repo_root,
                upstream_fingerprints=upstream,
                dependency_file_bindings=dependency_file_bindings,
            )
            gate_fingerprints[gate] = fingerprint["sha256"]
            gate_fingerprint_details[gate] = fingerprint
            upstream[gate] = fingerprint["sha256"]

            if not report or not evidence_rows:
                state = "missing"
            elif not current:
                state = "stale"
            elif _gate_passed(gate, evidence_rows):
                state = "passed"
            else:
                state = "failed"
            gate_statuses[gate] = state
            reasons = sorted(
                set(_row_reason_codes(evidence_rows))
                | set(qualification_errors)
            )
            if state == "failed":
                blockers.update(reasons or {f"{gate}_failed"})
            elif state == "stale":
                blockers.update(
                    reasons or {f"{gate}_evidence_stale"}
                )
            elif state == "missing" and gate != "lineage":
                blockers.add(f"{gate}_evidence_missing")
            gate_evidence[gate] = {
                "state": state,
                "current": current,
                "artifact": binding,
                "row_evidence_digest": row_digest,
                "reason_codes": reasons,
            }

            previous_fp = (previous_row.get("gate_fingerprints") or {}).get(gate)
            previous_evidence = (previous_row.get("gate_evidence") or {}).get(gate)
            if previous_fp is None and previous_row:
                unverifiable_gates.append(gate)
            elif previous_fp == fingerprint["sha256"] and (
                isinstance(previous_evidence, dict)
                and previous_evidence.get("current") is True
            ):
                reused_gates.append(gate)
            elif previous_fp == fingerprint["sha256"]:
                unverifiable_gates.append(gate)
            elif previous_fp is not None:
                invalidated_gates.append(gate)

        # Lineage is carried by the source suite, not by a separate report.
        lineage_ok = all(identity) and bool(
            source_row.get("structural_fingerprint")
        ) and bool(source_row.get("semantic_fingerprint"))
        if lineage_ok:
            gate_statuses["lineage"] = "passed"
            gate_evidence["lineage"]["state"] = "passed"
            gate_evidence["lineage"]["current"] = True
        else:
            gate_statuses["lineage"] = "failed"
            gate_evidence["lineage"]["state"] = "failed"
            blockers.add("lineage_incomplete")

        denominator = _denominator(source_row)
        physical = _physical_identity(source_row)
        if not denominator:
            blockers.add("source_denominator_key_missing")
        if not physical:
            blockers.add("physical_source_identity_missing")
        if invalidated_gates:
            blockers.add("gate_fingerprint_invalidated")
        if unverifiable_gates:
            blockers.add("previous_gate_fingerprint_unverifiable")

        deterministic_retirement = False
        prior_reason = str(prior_retirement.get("reason_code") or "")
        if prior_reason in _DETERMINISTIC_RETIREMENT_REASONS:
            try:
                deterministic_retirement = not get_backend_capability(
                    str(source_row.get("backend_kind") or "")
                ).formal_core_allowed
            except KeyError:
                deterministic_retirement = False
        stale_retirement = bool(prior_retirement) and not deterministic_retirement
        if stale_retirement:
            blockers.add("prior_retirement_evidence_stale")

        prequalification_gates = GATE_ORDER[:-1]
        prequalification_ready = (
            bool(denominator)
            and bool(physical)
            and all(gate_statuses[gate] == "passed" for gate in prequalification_gates)
            and not any(
                gate in prequalification_gates
                for gate in invalidated_gates
            )
            and not dependency_errors
        )
        formal_qualification_passed = gate_statuses["qualification"] == "passed"
        invalidated_prequalification_gates = [
            gate
            for gate in invalidated_gates
            if gate in prequalification_gates
        ]
        if deterministic_retirement:
            status = "retired"
            blockers.add(prior_reason)
        elif (
            stale_retirement
            or any(gate_statuses[gate] in {"missing", "stale"} for gate in RUNTIME_GATES)
            or bool(invalidated_prequalification_gates)
            or "backend_dependency_paths_unresolved" in blockers
        ):
            status = "needs_runtime"
        elif prequalification_ready and not formal_qualification_passed:
            status = "held"
            blockers.add("formal_qualification_not_run")
        elif prequalification_ready and formal_qualification_passed:
            status = "admitted"
        else:
            status = "held"

        recommended_rerun_order = [
            gate
            for gate in GATE_ORDER[1:]
            if gate_statuses[gate] in {"missing", "stale"}
            or gate in invalidated_gates
        ]
        release_eligible = status == "admitted"
        rows.append(
            {
                "scenario_id": identity[0],
                "scenario_signature": identity[1],
                "domain": str(source_row.get("domain") or ""),
                "backend_kind": str(source_row.get("backend_kind") or ""),
                "family": str(source_row.get("family") or ""),
                "difficulty_level": str(
                    source_row.get("difficulty_level") or ""
                ),
                "source_key": str(source_row.get("source_key") or ""),
                "source_denominator_key": denominator,
                "physical_source_key_or_lock": physical,
                "structural_fingerprint": str(
                    source_row.get("structural_fingerprint") or ""
                ),
                "semantic_fingerprint": str(
                    source_row.get("semantic_fingerprint") or ""
                ),
                "status": status,
                "prior_disposition": str(
                    prior_retirement.get("disposition")
                    or previous_row.get("status")
                    or ""
                ),
                "prequalification_ready": prequalification_ready,
                "formal_qualification_passed": formal_qualification_passed,
                "release_eligible": release_eligible,
                "diagnostic_materialization_disposition": (
                    "eligible" if prequalification_ready else "excluded"
                ),
                "blockers": sorted(blockers),
                "reason_codes": sorted(blockers),
                "gate_statuses": gate_statuses,
                "gate_evidence": gate_evidence,
                "gate_fingerprints": gate_fingerprints,
                "gate_fingerprint_details": gate_fingerprint_details,
                "invalidated_gates": invalidated_gates,
                "reused_gates": reused_gates,
                "unverifiable_gates": unverifiable_gates,
                "recommended_rerun_order": recommended_rerun_order,
            }
        )

    status_counts = Counter(row["status"] for row in rows)
    reused_gate_count = sum(len(row["reused_gates"]) for row in rows)
    invalidated_gate_count = sum(len(row["invalidated_gates"]) for row in rows)
    summary: dict[str, Any] = {
        "n_source_rows": len(source_rows),
        "n_external_retirement_rows": len(report_rows(retirements)),
        "status_counts": dict(sorted(status_counts.items())),
        "prequalification_ready_count": sum(
            bool(row["prequalification_ready"]) for row in rows
        ),
        "formal_admitted_count": status_counts["admitted"],
        "reused_gate_count": reused_gate_count,
        "invalidated_gate_count": invalidated_gate_count,
        "missing_denominator_count": sum(
            not bool(row["source_denominator_key"]) for row in rows
        ),
        "missing_physical_identity_count": sum(
            not bool(row["physical_source_key_or_lock"]) for row in rows
        ),
    }
    for field in ("domain", "backend_kind", "difficulty_level"):
        summary[f"{field}_status_counts"] = {
            key: dict(sorted(counts.items()))
            for key, counts in sorted(
                (
                    value,
                    Counter(
                        row["status"]
                        for row in rows
                        if row[field] == value
                    ),
                )
                for value in {row[field] for row in rows}
            )
        }
    summary["effective_source_status_counts"] = {
        value: dict(
            sorted(
                Counter(
                    row["status"]
                    for row in rows
                    if row["source_denominator_key"] == value
                ).items()
            )
        )
        for value in sorted(
            {
                row["source_denominator_key"]
                for row in rows
                if row["source_denominator_key"]
            }
        )
    }
    summary["physical_source_counts"] = dict(
        sorted(
            Counter(
                json.dumps(
                    row["physical_source_key_or_lock"],
                    sort_keys=True,
                    default=str,
                )
                for row in rows
                if row["physical_source_key_or_lock"]
            ).items()
        )
    )

    return {
        "schema_version": ADMISSION_LEDGER_SCHEMA_VERSION,
        "ledger_kind": "protocol21_candidate_admission",
        "official_release_claim": False,
        "dependency_schema_version": ADMISSION_DEPENDENCY_SCHEMA_VERSION,
        "evaluation_semantics": semantics,
        "source_suite": artifact_binding(source_suite_path),
        "previous_ledger": (
            artifact_binding(previous_ledger_path)
            if previous_ledger_path is not None
            else None
        ),
        "rows": rows,
        "summary": summary,
        "external_retirement_summary": dict(sorted(external_summary.items())),
    }
