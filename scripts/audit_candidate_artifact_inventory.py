#!/usr/bin/env python3
"""Audit candidate ledgers and wave artifacts without running a backend.

This is a read-only accounting check for candidate work.  It deliberately
does not promote rows or alter queues: it verifies identity uniqueness,
terminal coverage, source/hash bindings, and whether an artifact was produced
under the live implementation tree.  Historical ledgers are reported as
stale rather than silently treated as current evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402

VALID_DISPOSITIONS = {
    "core_locked_increment",
    "held_repair",
    "held_runtime",
    "held_license_or_terms",
    "held_access_terms",
    "transfer_only",
    "secondary_duplicate",
    "retired_intrinsic",
    "candidate_prefilter",
    "held_missing_assets",
    "method_transfer_only",
    "held_stale_evidence",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _identity(row: dict[str, Any]) -> tuple[str, str] | None:
    scenario_id = str(row.get("scenario_id") or "")
    signature = str(row.get("scenario_signature") or "")
    if not scenario_id or not signature:
        return None
    return scenario_id, signature


def _finding(
    findings: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    *,
    path: str | None = None,
) -> None:
    item: dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if path is not None:
        item["path"] = path
    findings.append(item)


def _check_file_binding(
    findings: list[dict[str, Any]],
    *,
    label: str,
    binding: Any,
    base_dir: Path = ROOT,
) -> Path | None:
    if not isinstance(binding, dict):
        _finding(findings, "error", f"{label}_binding_missing", "artifact binding must be an object")
        return None
    raw_path = binding.get("path")
    expected = binding.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        _finding(findings, "error", f"{label}_path_missing", "artifact binding path is missing")
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    if not path.is_file():
        _finding(findings, "error", f"{label}_path_missing", f"bound artifact does not exist: {path}")
        return path
    actual = _sha256(path)
    if expected != actual:
        _finding(
            findings,
            "error",
            f"{label}_hash_mismatch",
            f"bound artifact hash differs from the recorded hash: {path}",
        )
    return path


def _audit_exact_rows(
    rows: Iterable[Any],
    *,
    label: str,
    findings: list[dict[str, Any]],
    require_identity: bool = True,
) -> dict[str, Any]:
    identities: list[tuple[str, str]] = []
    missing_identity = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            _finding(findings, "error", f"{label}_row_type", f"row {index} is not an object")
            continue
        identity = _identity(row)
        if identity is None:
            missing_identity += 1
            if require_identity:
                _finding(findings, "error", f"{label}_identity_missing", f"row {index} lacks exact identity")
            continue
        identities.append(identity)
    duplicates = sorted(identity for identity, count in Counter(identities).items() if count > 1)
    if duplicates:
        _finding(
            findings,
            "error",
            f"{label}_duplicate_identity",
            f"{len(duplicates)} exact identity/identities occur more than once",
        )
    return {
        "n_rows": len(list(rows)) if not isinstance(rows, list) else len(rows),
        "n_exact_identities": len(set(identities)),
        "n_missing_identity": missing_identity,
        "n_duplicate_identities": len(duplicates),
        "identities": [list(identity) for identity in sorted(set(identities))],
    }


def _audit_source_units(rows: Any, findings: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        _finding(findings, "error", f"{label}_rows_missing", "source units must be a list")
        return {"n_rows": 0, "n_terminal": 0, "dispositions": {}}
    terminal = 0
    dispositions: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            _finding(findings, "error", f"{label}_row_type", f"source unit {index} is not an object")
            continue
        if row.get("work_state") == "terminal":
            terminal += 1
        else:
            _finding(findings, "error", f"{label}_nonterminal", f"source unit {index} is not terminal")
        disposition = row.get("disposition")
        if disposition not in VALID_DISPOSITIONS:
            _finding(findings, "error", f"{label}_disposition_invalid", f"source unit {index} has invalid disposition")
        else:
            dispositions[str(disposition)] += 1
    return {"n_rows": len(rows), "n_terminal": terminal, "dispositions": dict(sorted(dispositions.items()))}


def _suite_identities(suite: dict[str, Any], findings: list[dict[str, Any]], label: str) -> set[tuple[str, str]]:
    rows = suite.get("scenarios")
    if not isinstance(rows, list):
        _finding(findings, "error", f"{label}_rows_missing", "scenario suite rows must be a list")
        return set()
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            _finding(findings, "error", f"{label}_row_type", f"scenario row {index} is not an object")
            continue
        identity = _identity(row)
        if identity is None:
            _finding(findings, "error", f"{label}_identity_missing", f"scenario row {index} lacks exact identity")
            continue
        if identity in identities:
            _finding(findings, "error", f"{label}_duplicate_identity", f"duplicate suite identity {identity[0]}")
        identities.add(identity)
        raw_path = row.get("path")
        if isinstance(raw_path, str) and not _resolve(raw_path).is_file():
            _finding(findings, "error", f"{label}_scenario_path_missing", f"scenario artifact does not exist: {raw_path}")
    declared = suite.get("n_scenarios")
    if isinstance(declared, int) and declared != len(rows):
        _finding(findings, "error", f"{label}_count_mismatch", f"declared {declared} rows, observed {len(rows)}")
    return identities


def _audit_power_queue(
    queue_path: Path,
    suite_path: Path,
    base_core_path: Path,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    queue = _load(queue_path)
    if queue.get("schema_version") != "candidate-batch-queue-v1":
        _finding(findings, "error", "power_queue_schema", "Power queue schema is not candidate-batch-queue-v1", path=str(queue_path))
    bindings = queue.get("input_bindings")
    if not isinstance(bindings, dict):
        _finding(findings, "error", "power_queue_input_bindings_missing", "Power queue input bindings are missing")
    else:
        if bindings.get("candidate_suite", {}).get("path") != str(suite_path):
            _finding(findings, "error", "power_queue_suite_path_mismatch", "queue candidate suite path differs from audit input")
        if bindings.get("candidate_suite", {}).get("sha256") != _sha256(suite_path):
            _finding(findings, "error", "power_queue_suite_hash_mismatch", "queue candidate suite hash is stale")
        if bindings.get("base_core", {}).get("path") != str(base_core_path):
            _finding(findings, "error", "power_queue_core_path_mismatch", "queue base Core path differs from audit input")
        if bindings.get("base_core", {}).get("sha256") != _sha256(base_core_path):
            _finding(findings, "error", "power_queue_core_hash_mismatch", "queue base Core hash is stale")
    items = queue.get("items")
    if not isinstance(items, list):
        _finding(findings, "error", "power_queue_items_missing", "Power queue items are missing")
        items = []
    work_ids = [str(item.get("work_id") or "") for item in items if isinstance(item, dict)]
    for work_id, count in Counter(work_ids).items():
        if not work_id or count > 1:
            _finding(findings, "error", "power_queue_work_id_duplicate", f"duplicate or missing queue work_id: {work_id!r}")
    n_scenarios = 0
    for item in items:
        if not isinstance(item, dict):
            _finding(findings, "error", "power_queue_item_type", "Power queue item is not an object")
            continue
        metadata = item.get("metadata")
        identity = _identity(item)
        scope = metadata.get("identity_scope") if isinstance(metadata, dict) else None
        if identity is None and scope not in {"suite_aggregate", "shard_aggregate", "batch_aggregate"}:
            _finding(findings, "error", "power_queue_aggregate_scope_missing", f"{item.get('work_id')}: aggregate lacks identity_scope")
        count = metadata.get("n_scenarios") if isinstance(metadata, dict) else None
        if not isinstance(count, int) or count < 1:
            _finding(findings, "error", "power_queue_scenario_count_missing", f"{item.get('work_id')}: n_scenarios missing")
        else:
            n_scenarios += count
    return {"path": str(queue_path), "sha256": _sha256(queue_path), "n_items": len(items), "n_scenarios": n_scenarios}


def _audit_power_plan(
    plan_path: Path,
    queue_path: Path,
    base_core_path: Path,
    current_tree: str,
    queue_items: int,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    plan = _load(plan_path)
    if plan.get("schema_version") != "protocol21-candidate-batch-plan-v1":
        _finding(findings, "error", "power_plan_schema", "Power plan schema is not protocol21-candidate-batch-plan-v1")
    if plan.get("queue_path") != str(queue_path):
        _finding(findings, "error", "power_plan_queue_path_mismatch", "plan queue path differs from audit input")
    if plan.get("queue_artifact_sha256") != _sha256(queue_path):
        _finding(findings, "error", "power_plan_queue_hash_mismatch", "plan queue hash is stale")
    if plan.get("base_core_path") != str(base_core_path):
        _finding(findings, "error", "power_plan_core_path_mismatch", "plan base Core path differs from audit input")
    if plan.get("base_core_sha256") != _sha256(base_core_path):
        _finding(findings, "error", "power_plan_core_hash_mismatch", "plan base Core hash is stale")
    expected_tree = plan.get("implementation_tree_sha256")
    if expected_tree != current_tree:
        _finding(findings, "error", "plan_implementation_tree_stale", "Power plan was generated under a stale implementation tree")
    if plan.get("n_queue_items") != queue_items:
        _finding(findings, "error", "power_plan_queue_count_mismatch", "plan n_queue_items differs from queue")
    shards = plan.get("shards")
    planned_work_ids: list[str] = []
    if not isinstance(shards, list):
        _finding(findings, "error", "power_plan_shards_missing", "Power plan shards are missing")
        shards = []
    for shard in shards:
        if not isinstance(shard, dict):
            _finding(findings, "error", "power_plan_shard_type", "Power plan shard is not an object")
            continue
        for item in shard.get("items") or []:
            if isinstance(item, dict):
                planned_work_ids.append(str(item.get("work_id") or ""))
    if len(planned_work_ids) != plan.get("n_scheduled"):
        _finding(findings, "error", "power_plan_scheduled_count_mismatch", "plan n_scheduled differs from shard items")
    return {
        "path": str(plan_path),
        "sha256": _sha256(plan_path),
        "n_scheduled": plan.get("n_scheduled"),
        "n_shards": len(shards),
        "implementation_tree_sha256": expected_tree,
    }


def audit_candidate_ledger(
    ledger_path: Path,
    *,
    result_root: Path | None = None,
    current_tree: str | None = None,
    expected_base_core_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit one protocol21 coordinator ledger and optionally its result tree."""
    ledger = _load(ledger_path)
    findings: list[dict[str, Any]] = []
    if ledger.get("schema_version") != "protocol21-candidate-batch-ledger-v1":
        _finding(findings, "warning", "ledger_schema_unexpected", "ledger is not the coordinator ledger schema")
    items = ledger.get("items")
    if not isinstance(items, list):
        _finding(findings, "error", "ledger_items_missing", "ledger items must be a list")
        items = []
    result_ids: set[str] = set()
    referenced_paths: set[Path] = set()
    for item in items:
        if not isinstance(item, dict):
            _finding(findings, "error", "ledger_item_type", "ledger item is not an object")
            continue
        work_id = str(item.get("work_id") or "")
        if not work_id or work_id in result_ids:
            _finding(findings, "error", "ledger_work_id_duplicate", f"duplicate or missing work_id: {work_id!r}")
        result_ids.add(work_id)
        result_path = item.get("result_path")
        if not isinstance(result_path, str) or not result_path:
            _finding(findings, "error", "ledger_result_missing", f"{work_id}: result_path missing")
            continue
        resolved_result = _resolve(result_path)
        referenced_paths.add(resolved_result)
        if not resolved_result.is_file():
            _finding(findings, "error", "ledger_result_missing", f"{work_id}: result artifact missing: {resolved_result}")
            continue
        try:
            result = _load(resolved_result)
        except (OSError, ValueError, json.JSONDecodeError):
            _finding(findings, "error", "ledger_result_invalid", f"{work_id}: result artifact is not valid JSON: {resolved_result}")
            continue
        if result.get("work_id") not in {None, work_id}:
            _finding(findings, "error", "ledger_result_work_id_mismatch", f"{work_id}: result work_id differs")
    declared = ledger.get("n_scheduled")
    if isinstance(declared, int) and declared != len(items):
        _finding(findings, "error", "ledger_scheduled_count_mismatch", f"ledger n_scheduled={declared}, items={len(items)}")
    if ledger.get("status") == "complete" and findings:
        _finding(findings, "error", "ledger_complete_with_findings", "ledger claims complete despite accounting findings")
    if current_tree is not None and ledger.get("implementation_tree_sha256") != current_tree:
        _finding(findings, "warning", "ledger_implementation_tree_stale", "ledger was generated under a different implementation tree")
    if expected_base_core_sha256 and ledger.get("base_core_sha256") != expected_base_core_sha256:
        _finding(findings, "warning", "ledger_base_core_stale", "ledger is bound to a different base Core hash")
    if result_root is not None and result_root.is_dir():
        for path in result_root.rglob("*.json"):
            resolved = path.resolve()
            if resolved not in referenced_paths:
                _finding(findings, "warning", "ledger_orphan_result", f"unreferenced result artifact: {path}")
    return {
        "path": str(ledger_path),
        "sha256": _sha256(ledger_path),
        "status": ledger.get("status"),
        "n_scheduled": declared,
        "n_items": len(items),
        "n_referenced_results": len(referenced_paths),
        "findings": findings,
    }


def _audit_power_prefilter(
    suite_path: Path,
    prefilter_paths: list[Path],
    current_tree: str,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    suite = _load(suite_path)
    expected_ids = _suite_identities(suite, findings, "power_prefilter_suite")
    observed: set[tuple[str, str]] = set()
    rows_by_report: dict[str, int] = {}
    for path in prefilter_paths:
        if not path.is_file():
            _finding(findings, "error", "power_prefilter_missing", f"prefilter report missing: {path}")
            continue
        report = _load(path)
        binding = report.get("input_bindings", {}).get("source_suite") if isinstance(report.get("input_bindings"), dict) else None
        if not isinstance(binding, dict):
            _finding(findings, "error", "power_prefilter_binding_missing", f"source suite binding missing: {path}")
        else:
            if binding.get("path") != str(path.parent.parent / "suites" / path.name):
                # The generated report normally binds the absolute suite path;
                # exact path checking is performed from the report itself below.
                bound_path = _resolve(str(binding.get("path") or ""))
                if not bound_path.is_file():
                    _finding(findings, "error", "power_prefilter_bound_suite_missing", f"bound suite missing: {bound_path}")
            bound_path = _resolve(str(binding.get("path") or ""))
            if bound_path.is_file() and binding.get("sha256") != _sha256(bound_path):
                _finding(findings, "error", "power_prefilter_bound_suite_hash_stale", f"prefilter source suite hash is stale: {path}")
        if report.get("implementation_tree_sha256") != current_tree:
            _finding(findings, "warning", "power_prefilter_implementation_tree_stale", f"prefilter report is from a different implementation tree: {path}")
        results = report.get("results")
        if not isinstance(results, list):
            _finding(findings, "error", "power_prefilter_results_missing", f"prefilter results missing: {path}")
            continue
        rows_by_report[str(path)] = len(results)
        expected = report.get("n_expected")
        completed = report.get("n_completed")
        if expected != len(results) or completed != len(results):
            _finding(findings, "error", "power_prefilter_count_mismatch", f"prefilter count mismatch: {path}")
        for row in results:
            if not isinstance(row, dict):
                _finding(findings, "error", "power_prefilter_row_type", f"prefilter result is not an object: {path}")
                continue
            identity = _identity(row)
            if identity is None:
                _finding(findings, "error", "power_prefilter_identity_missing", f"prefilter result lacks identity: {path}")
                continue
            if identity in observed:
                _finding(findings, "error", "power_prefilter_duplicate_identity", f"identity appears in multiple prefilter reports: {identity[0]}")
            observed.add(identity)
    missing = expected_ids - observed
    extra = observed - expected_ids
    if missing:
        _finding(findings, "error", "power_prefilter_missing_terminal", f"{len(missing)} suite identities lack prefilter results")
    if extra:
        _finding(findings, "error", "power_prefilter_unmatched", f"{len(extra)} prefilter identities are absent from suite")
    return {"n_expected": len(expected_ids), "n_observed": len(observed), "reports": rows_by_report}


def audit_power_artifacts(
    *,
    candidate_path: Path,
    suite_path: Path,
    queue_path: Path,
    plan_path: Path,
    ledger_path: Path | None,
    prefilter_paths: list[Path],
    current_tree: str,
    base_core_path: Path | None = None,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    candidate = _load(candidate_path)
    suite = _load(suite_path)
    base_core = base_core_path or _resolve(str(candidate.get("base_core", {}).get("path") or ""))
    if not base_core.is_file():
        _finding(findings, "error", "power_base_core_missing", f"base Core artifact missing: {base_core}")
    else:
        recorded = candidate.get("base_core", {}).get("sha256") if isinstance(candidate.get("base_core"), dict) else None
        if recorded != _sha256(base_core):
            _finding(findings, "error", "power_candidate_core_hash_stale", "Power candidate report base Core hash is stale")
    candidate_rows = candidate.get("scenarios")
    if not isinstance(candidate_rows, list):
        _finding(findings, "error", "power_candidate_rows_missing", "Power candidate scenarios must be a list")
        candidate_rows = []
    candidate_identity_report = _audit_exact_rows(candidate_rows, label="candidate", findings=findings)
    source_units_report = _audit_source_units(candidate.get("source_units"), findings, "power_source_units")
    suite_ids = _suite_identities(suite, findings, "power_suite")
    candidate_ids = {_identity(row) for row in candidate_rows if isinstance(row, dict) and _identity(row) is not None}
    if candidate_ids != suite_ids:
        _finding(findings, "error", "power_candidate_suite_identity_mismatch", "candidate report and suite identities differ")
    artifact = next(iter(suite.get("source_artifacts") or []), None)
    if isinstance(artifact, dict):
        artifact_path = _resolve(str(artifact.get("path") or ""))
        if not artifact_path.is_file() or artifact.get("sha256") != _sha256(artifact_path):
            _finding(findings, "error", "power_suite_source_artifact_stale", "suite source artifact binding is stale")
        elif artifact_path.resolve() != candidate_path.resolve():
            _finding(findings, "error", "power_suite_source_artifact_mismatch", "suite source artifact does not point to candidate report")
    else:
        _finding(findings, "error", "power_suite_source_artifact_missing", "suite source artifact binding is missing")
    queue_report = _audit_power_queue(queue_path, suite_path, base_core, findings)
    if queue_report["n_scenarios"] != len(suite_ids):
        _finding(findings, "error", "power_queue_scenario_total_mismatch", "queue shard totals do not equal suite rows")
    plan_report = _audit_power_plan(plan_path, queue_path, base_core, current_tree, queue_report["n_items"], findings)
    ledger_report = None
    if ledger_path is not None:
        ledger_report = audit_candidate_ledger(
            ledger_path,
            result_root=ledger_path.parent / "results",
            current_tree=current_tree,
            expected_base_core_sha256=_sha256(base_core) if base_core.is_file() else None,
        )
        findings.extend(ledger_report["findings"])
        if ledger_report.get("n_scheduled") != plan_report.get("n_scheduled"):
            _finding(findings, "error", "power_ledger_plan_count_mismatch", "ledger and plan scheduled counts differ")
    prefilter_report = _audit_power_prefilter(suite_path, prefilter_paths, current_tree, findings) if prefilter_paths else None
    return {
        "scope": "power_grid_candidate_wave",
        "candidate": {"path": str(candidate_path), "sha256": _sha256(candidate_path), "identity": candidate_identity_report, "source_units": source_units_report},
        "suite": {"path": str(suite_path), "sha256": _sha256(suite_path), "n_identities": len(suite_ids)},
        "queue": queue_report,
        "plan": plan_report,
        "ledger": ledger_report,
        "prefilter": prefilter_report,
        "findings": findings,
        "status": "blocked" if any(item["severity"] == "error" for item in findings) else "passed",
    }


def audit_latest_wave(path: Path, *, current_tree: str) -> dict[str, Any]:
    wave = _load(path)
    findings: list[dict[str, Any]] = []
    if wave.get("schema_version") != "latest-benchmark-candidate-wave-v1":
        _finding(findings, "error", "latest_schema", "latest benchmark wave schema is unexpected")
    sources = wave.get("sources")
    if not isinstance(sources, list):
        _finding(findings, "error", "latest_sources_missing", "latest wave sources must be a list")
        sources = []
    source_ids = [str(source.get("source_id") or "") for source in sources if isinstance(source, dict)]
    for source_id, count in Counter(source_ids).items():
        if not source_id or count > 1:
            _finding(findings, "error", "latest_duplicate_source_id", f"duplicate or missing latest source_id: {source_id!r}")
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            _finding(findings, "error", "latest_source_type", f"latest source {index} is not an object")
            continue
        if source.get("work_state") != "terminal":
            _finding(findings, "error", "latest_nonterminal_source", f"latest source {index} is not terminal")
        if source.get("disposition") not in VALID_DISPOSITIONS:
            _finding(findings, "error", "latest_disposition_invalid", f"latest source {index} has invalid disposition")
    counts = wave.get("counts")
    if isinstance(counts, dict):
        if counts.get("terminal_rows") != len(sources):
            _finding(findings, "error", "latest_terminal_count_mismatch", "latest terminal_rows differs from source rows")
        observed_dispositions = Counter(str(source.get("disposition")) for source in sources if isinstance(source, dict))
        if counts.get("dispositions") != dict(sorted(observed_dispositions.items())):
            _finding(findings, "error", "latest_disposition_count_mismatch", "latest disposition counts differ from source rows")
        for key in ("native_replay_rows", "full_protocol21_ready_rows"):
            if key in counts:
                observed = sum(int(source.get(key, 0) or 0) for source in sources if isinstance(source, dict))
                if counts[key] != observed:
                    _finding(findings, "error", f"latest_{key}_mismatch", f"latest {key} differs from source rows")
    binding = wave.get("implementation_binding")
    if not isinstance(binding, dict) or binding.get("implementation_tree_sha256") != current_tree:
        _finding(findings, "error", "latest_implementation_tree_stale", "latest wave is bound to a stale implementation tree")
    input_bindings = wave.get("input_bindings")
    if not isinstance(input_bindings, dict):
        _finding(findings, "error", "latest_input_bindings_missing", "latest wave input bindings are missing")
    else:
        for label, raw in input_bindings.items():
            if not isinstance(raw, dict):
                _finding(findings, "error", "latest_input_binding_invalid", f"latest binding {label} is not an object")
                continue
            raw_path = raw.get("path")
            if not isinstance(raw_path, str):
                _finding(findings, "error", "latest_input_path_missing", f"latest binding {label} lacks path")
                continue
            input_path = _resolve(raw_path)
            if not input_path.is_file():
                _finding(findings, "error", "latest_input_path_missing", f"latest input does not exist: {input_path}")
            elif raw.get("sha256") != _sha256(input_path):
                _finding(findings, "error", "latest_input_hash_stale", f"latest input binding is stale: {input_path}")
    if wave.get("core_admission_claimed"):
        _finding(findings, "error", "latest_core_admission_claimed", "latest wave must remain candidate-only")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "n_sources": len(sources),
        "counts": counts,
        "findings": findings,
        "status": "blocked" if any(item["severity"] == "error" for item in findings) else "passed",
    }


def audit_existing_candidate_ledgers(
    root: Path,
    *,
    current_tree: str,
    base_core_sha256: str | None = None,
    exclude: set[Path] | None = None,
) -> dict[str, Any]:
    exclude = {path.resolve() for path in (exclude or set())}
    reports: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for path in sorted(root.rglob("candidate_batch_ledger.json")):
        if path.resolve() in exclude:
            continue
        try:
            report = audit_candidate_ledger(path, current_tree=current_tree, expected_base_core_sha256=base_core_sha256)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _finding(findings, "error", "ledger_unreadable", f"cannot read candidate ledger {path}: {exc}")
            continue
        reports.append(report)
        findings.extend(report["findings"])
    return {
        "root": str(root),
        "n_ledgers": len(reports),
        "n_stale": sum(any(item["code"] == "ledger_implementation_tree_stale" for item in report["findings"]) for report in reports),
        "n_incomplete": sum(report.get("status") != "complete" for report in reports),
        "reports": reports,
        "findings": findings,
    }


def build_report(
    *,
    power_candidate: Path,
    power_suite: Path,
    power_queue: Path,
    power_plan: Path,
    power_ledger: Path | None,
    power_prefilter: list[Path],
    latest_wave: Path,
    base_core: Path,
    existing_root: Path,
) -> dict[str, Any]:
    current_tree = implementation_identity()["implementation_tree_sha256"]
    power = audit_power_artifacts(
        candidate_path=power_candidate.resolve(),
        suite_path=power_suite.resolve(),
        queue_path=power_queue.resolve(),
        plan_path=power_plan.resolve(),
        ledger_path=power_ledger.resolve() if power_ledger else None,
        prefilter_paths=[path.resolve() for path in power_prefilter],
        current_tree=current_tree,
        base_core_path=base_core.resolve(),
    )
    latest = audit_latest_wave(latest_wave.resolve(), current_tree=current_tree)
    existing = audit_existing_candidate_ledgers(
        existing_root.resolve(),
        current_tree=current_tree,
        base_core_sha256=_sha256(base_core.resolve()) if base_core.is_file() else None,
        exclude={power_ledger.resolve()} if power_ledger else set(),
    )
    findings = power["findings"] + latest["findings"] + existing["findings"]
    return {
        "schema_version": "candidate-artifact-inventory-audit-v1",
        "read_only": True,
        "current_implementation_tree_sha256": current_tree,
        "base_core": {"path": str(base_core.resolve()), "sha256": _sha256(base_core.resolve()) if base_core.is_file() else None},
        "power": power,
        "latest_wave": latest,
        "existing_candidate_ledgers": existing,
        "findings": findings,
        "status": "blocked" if any(item["severity"] == "error" for item in findings) else "passed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-candidate", type=Path, default=Path("reports/powergrid_candidate_batch_20260813.json"))
    parser.add_argument("--power-suite", type=Path, default=Path("reports/powergrid_candidate_batch_source_suite_20260813.json"))
    parser.add_argument("--power-queue", type=Path, default=Path("reports/powergrid_candidate_queue_20260813.json"))
    parser.add_argument("--power-plan", type=Path, default=Path("reports/powergrid_candidate_plan_20260813.json"))
    parser.add_argument("--power-ledger", type=Path, default=Path("reports/powergrid_candidate_execution_20260813/candidate_batch_ledger.json"))
    parser.add_argument("--power-prefilter", type=Path, action="append", default=[])
    parser.add_argument("--latest-wave", type=Path, default=Path("reports/latest_benchmark_candidate_wave_20260813.json"))
    parser.add_argument("--base-core", type=Path, default=Path("release/dt_sched_bench_v0_51_0/core_suite.json"))
    parser.add_argument("--existing-root", type=Path, default=Path("reports"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = build_report(
            power_candidate=args.power_candidate,
            power_suite=args.power_suite,
            power_queue=args.power_queue,
            power_plan=args.power_plan,
            power_ledger=args.power_ledger,
            power_prefilter=args.power_prefilter,
            latest_wave=args.latest_wave,
            base_core=args.base_core,
            existing_root=args.existing_root,
        )
        if args.output:
            output = args.output.resolve()
            if output.exists():
                raise ValueError(f"refusing to overwrite existing report: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "n_findings": len(report["findings"])}, sort_keys=True))
    return 0 if report["status"] == "passed" else 4


if __name__ == "__main__":
    raise SystemExit(main())
