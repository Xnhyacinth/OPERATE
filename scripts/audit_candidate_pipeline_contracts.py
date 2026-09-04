#!/usr/bin/env python3
"""Audit candidate pipeline artifacts without executing or mutating them.

The candidate coordinator intentionally accepts aggregate suite/shard commands,
so a return code alone cannot prove that every source row was accounted for.
This audit makes the missing proof explicit: exact identities (or an explicit
aggregate scope), one result per scheduled work item, current input hashes,
and no stale/unbound prefilter reports.

The command is diagnostic only.  It never calls a simulator, changes a queue,
or rewrites an existing report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402

STAGES = {
    "inventory",
    "conversion",
    "static_preflight",
    "native_prefilter",
    "full_protocol21",
    "evidence_freeze",
    "final_union",
}
WORK_STATES = {"pending", "running", "passed", "failed_retryable", "terminal"}
DISPOSITIONS = {
    "core_locked_increment",
    "held_repair",
    "held_runtime",
    "held_license_or_terms",
    "transfer_only",
    "secondary_duplicate",
    "retired_intrinsic",
}
SCHEDULABLE = {"pending", "failed_retryable"}


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


def _identity(row: dict[str, Any]) -> tuple[str, str] | None:
    scenario_id = str(row.get("scenario_id") or "")
    signature = str(row.get("scenario_signature") or "")
    if not scenario_id and not signature:
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
    finding: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if path is not None:
        finding["path"] = path
    findings.append(finding)


def audit_queue(queue_path: Path, base_core_path: Path) -> dict[str, Any]:
    """Audit a candidate-batch queue and its exact locked-Core skip boundary."""
    queue = _load(queue_path)
    core = _load(base_core_path)
    findings: list[dict[str, Any]] = []
    if queue.get("schema_version") != "candidate-batch-queue-v1":
        _finding(findings, "error", "queue_schema", "unsupported queue schema", path=str(queue_path))
    items = queue.get("items")
    if not isinstance(items, list):
        _finding(findings, "error", "queue_items", "queue items must be a list", path=str(queue_path))
        items = []
    core_rows = core.get("scenarios")
    locked: set[tuple[str, str]] = set()
    if not isinstance(core_rows, list):
        _finding(findings, "error", "core_rows", "base Core scenarios must be a list", path=str(base_core_path))
        core_rows = []
    for index, row in enumerate(core_rows):
        if not isinstance(row, dict):
            _finding(findings, "error", "core_row_type", f"base Core row {index} is not an object")
            continue
        identity = _identity(row)
        if identity is None or not all(identity):
            _finding(findings, "error", "core_identity", f"base Core row {index} lacks exact identity")
            continue
        if identity in locked:
            _finding(findings, "error", "core_duplicate_identity", f"duplicate base Core identity {identity[0]}")
        locked.add(identity)

    work_ids: set[str] = set()
    scheduled_ids: set[str] = set()
    exact_identities: set[tuple[str, str]] = set()
    for index, raw in enumerate(items):
        if not isinstance(raw, dict):
            _finding(findings, "error", "queue_item_type", f"queue item {index} is not an object")
            continue
        work_id = str(raw.get("work_id") or "")
        if not work_id:
            _finding(findings, "error", "work_id_missing", f"queue item {index} lacks work_id")
        elif work_id in work_ids:
            _finding(findings, "error", "work_id_duplicate", f"duplicate work_id {work_id}")
        work_ids.add(work_id)
        stage = raw.get("stage")
        state = raw.get("work_state")
        disposition = raw.get("disposition")
        if stage not in STAGES:
            _finding(findings, "error", "stage_invalid", f"{work_id}: invalid stage {stage!r}")
        if state not in WORK_STATES:
            _finding(findings, "error", "work_state_invalid", f"{work_id}: invalid work_state {state!r}")
        if disposition is not None and disposition not in DISPOSITIONS:
            _finding(findings, "error", "disposition_invalid", f"{work_id}: invalid disposition {disposition!r}")
        if state == "terminal" and disposition is None:
            _finding(findings, "error", "terminal_without_disposition", f"{work_id}: terminal item lacks disposition")
        if state in SCHEDULABLE:
            scheduled_ids.add(work_id)
            command = raw.get("command")
            if not isinstance(command, list) or not command or not all(isinstance(part, str) and part for part in command):
                _finding(findings, "error", "command_missing", f"{work_id}: schedulable item lacks argv")
        identity = _identity(raw)
        metadata = raw.get("metadata")
        scope = metadata.get("identity_scope") if isinstance(metadata, dict) else None
        if identity is None:
            if scope not in {"suite_aggregate", "shard_aggregate", "batch_aggregate"}:
                _finding(
                    findings,
                    "error",
                    "aggregate_scope_missing",
                    f"{work_id}: missing identity without explicit aggregate identity_scope",
                )
        elif not all(identity):
            _finding(findings, "error", "partial_identity", f"{work_id}: scenario identity is incomplete")
        else:
            exact_identities.add(identity)
            if state in SCHEDULABLE and identity in locked:
                _finding(
                    findings,
                    "error",
                    "locked_core_scheduled",
                    f"{work_id}: exact locked-Core identity is schedulable",
                )

    return {
        "path": str(queue_path),
        "sha256": _sha256(queue_path),
        "base_core": {"path": str(base_core_path), "sha256": _sha256(base_core_path)},
        "n_items": len(items),
        "n_scheduled": len(scheduled_ids),
        "n_exact_identities": len(exact_identities),
        "findings": findings,
    }


def audit_plan(plan_path: Path) -> dict[str, Any]:
    """Check shard accounting and whether a plan's bound inputs are stale."""
    plan = _load(plan_path)
    findings: list[dict[str, Any]] = []
    shards = plan.get("shards")
    if not isinstance(shards, list):
        _finding(findings, "error", "plan_shards", "plan shards must be a list", path=str(plan_path))
        shards = []
    shard_ids: set[str] = set()
    scheduled_ids: set[str] = set()
    for shard in shards:
        if not isinstance(shard, dict):
            _finding(findings, "error", "shard_type", "plan shard is not an object")
            continue
        shard_id = str(shard.get("shard_id") or "")
        if not shard_id or shard_id in shard_ids:
            _finding(findings, "error", "shard_id_duplicate", f"duplicate or missing shard_id {shard_id!r}")
        shard_ids.add(shard_id)
        items = shard.get("items")
        if not isinstance(items, list):
            _finding(findings, "error", "shard_items", f"{shard_id}: items must be a list")
            continue
        for item in items:
            if not isinstance(item, dict):
                _finding(findings, "error", "shard_item_type", f"{shard_id}: item is not an object")
                continue
            work_id = str(item.get("work_id") or "")
            if not work_id or work_id in scheduled_ids:
                _finding(findings, "error", "planned_work_id_duplicate", f"duplicate/missing planned work_id {work_id!r}")
            scheduled_ids.add(work_id)
        tokens = shard.get("resource_tokens")
        if not isinstance(tokens, int) or tokens < 1:
            _finding(findings, "error", "resource_tokens", f"{shard_id}: invalid resource_tokens")
        elif tokens > int(plan.get("runtime_resource_tokens") or 0):
            _finding(findings, "error", "resource_capacity", f"{shard_id}: resource weight exceeds plan capacity")

    declared = plan.get("n_scheduled")
    if isinstance(declared, int) and declared != len(scheduled_ids):
        _finding(findings, "error", "planned_count_mismatch", f"plan n_scheduled={declared}, observed={len(scheduled_ids)}")
    elif not isinstance(declared, int):
        _finding(findings, "error", "planned_count_missing", "plan n_scheduled must be an integer")

    for key in ("queue_path", "base_core_path"):
        raw_path = plan.get(key)
        if not isinstance(raw_path, str) or not raw_path:
            _finding(findings, "error", f"{key}_missing", f"plan {key} is missing")
            continue
        path = Path(raw_path)
        if not path.is_file():
            _finding(findings, "warning", f"{key}_unavailable", f"plan {key} is not readable: {path}")
            continue
        digest_key = "queue_artifact_sha256" if key == "queue_path" else "base_core_sha256"
        expected = plan.get(digest_key)
        if expected != _sha256(path):
            _finding(findings, "error", f"{key}_hash_mismatch", f"plan {key} hash does not match bound artifact")
    expected_tree = plan.get("implementation_tree_sha256")
    if isinstance(expected_tree, str) and expected_tree:
        live_tree = implementation_identity()["implementation_tree_sha256"]
        if expected_tree != live_tree:
            _finding(findings, "error", "implementation_tree_stale", "plan implementation tree differs from live tree")
    else:
        _finding(findings, "error", "implementation_tree_missing", "plan implementation tree binding is missing")
    return {
        "path": str(plan_path),
        "sha256": _sha256(plan_path),
        "n_shards": len(shards),
        "n_scheduled": len(scheduled_ids),
        "findings": findings,
    }


def audit_ledger(ledger_path: Path, plan_path: Path | None = None) -> dict[str, Any]:
    """Ensure a coordinator ledger has exactly one result for every planned item."""
    ledger = _load(ledger_path)
    findings: list[dict[str, Any]] = []
    raw_items = ledger.get("items")
    if not isinstance(raw_items, list):
        _finding(findings, "error", "ledger_items", "ledger items must be a list", path=str(ledger_path))
        raw_items = []
    result_ids: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict):
            _finding(findings, "error", "ledger_item_type", "ledger item is not an object")
            continue
        work_id = str(item.get("work_id") or "")
        if not work_id or work_id in result_ids:
            _finding(findings, "error", "ledger_work_id_duplicate", f"duplicate/missing ledger work_id {work_id!r}")
        result_ids.add(work_id)
        result_path = item.get("result_path")
        if not isinstance(result_path, str) or not Path(result_path).is_file():
            _finding(findings, "error", "ledger_result_missing", f"{work_id}: result artifact is missing")
        if item.get("work_state") not in {"passed", "terminal", "failed_retryable"}:
            _finding(findings, "error", "ledger_state_invalid", f"{work_id}: invalid result work_state")
    planned_ids: set[str] = set()
    if plan_path is not None:
        plan = _load(plan_path)
        for shard in plan.get("shards") or []:
            if isinstance(shard, dict):
                for item in shard.get("items") or []:
                    if isinstance(item, dict):
                        planned_ids.add(str(item.get("work_id") or ""))
        missing = sorted(planned_ids - result_ids)
        extra = sorted(result_ids - planned_ids)
        if missing:
            _finding(findings, "error", "ledger_missing_items", f"ledger omits {len(missing)} planned work item(s)")
        if extra:
            _finding(findings, "error", "ledger_extra_items", f"ledger contains {len(extra)} unplanned work item(s)")
    if ledger.get("status") == "complete" and findings:
        _finding(findings, "error", "complete_with_findings", "ledger claims complete despite accounting findings")
    return {
        "path": str(ledger_path),
        "sha256": _sha256(ledger_path),
        "status": ledger.get("status"),
        "n_results": len(result_ids),
        "n_planned": len(planned_ids),
        "findings": findings,
    }


def audit_prefilter(source_path: Path, report_paths: list[Path]) -> dict[str, Any]:
    """Check one-terminal-per-source accounting for prefilter reports."""
    source = _load(source_path)
    source_rows = source.get("scenarios")
    findings: list[dict[str, Any]] = []
    source_ids = {
        _identity(row)
        for row in source_rows or []
        if isinstance(row, dict) and _identity(row) is not None and all(_identity(row) or ())
    }
    terminals: dict[tuple[str, str], Path] = {}
    source_sha = _sha256(source_path)
    for path in report_paths:
        report = _load(path)
        if report.get("source_suite_sha256") != source_sha:
            _finding(findings, "error", "prefilter_source_hash", f"prefilter report is bound to a different source suite: {path}")
        if not report.get("implementation_tree_sha256") or not report.get("runtime_version"):
            _finding(findings, "error", "prefilter_runtime_binding", f"prefilter report lacks implementation/runtime binding: {path}")
        rows = report.get("rows")
        if not isinstance(rows, list):
            _finding(findings, "error", "prefilter_rows", f"prefilter rows missing: {path}")
            continue
        for row in rows:
            if not isinstance(row, dict):
                _finding(findings, "error", "prefilter_row_type", f"prefilter row is not an object: {path}")
                continue
            identity = _identity(row)
            if identity is None or not all(identity):
                _finding(findings, "error", "prefilter_identity", f"prefilter row lacks exact identity: {path}")
                continue
            if identity in terminals:
                _finding(findings, "error", "prefilter_duplicate_terminal", f"identity appears in multiple reports: {identity[0]}")
            terminals[identity] = path
    missing = sorted(source_ids - set(terminals))
    unmatched = sorted(set(terminals) - source_ids)
    if missing:
        _finding(findings, "error", "prefilter_missing_terminal", f"{len(missing)} source row(s) have no terminal prefilter disposition")
    if unmatched:
        _finding(findings, "error", "prefilter_unmatched", f"{len(unmatched)} prefilter row(s) are absent from source suite")
    return {
        "source": {"path": str(source_path), "sha256": source_sha},
        "reports": [{"path": str(path), "sha256": _sha256(path)} for path in report_paths],
        "n_source_rows": len(source_ids),
        "n_terminal_rows": len(terminals),
        "findings": findings,
    }


def build_report(
    *,
    queue_path: Path,
    base_core_path: Path,
    plan_path: Path | None = None,
    ledger_path: Path | None = None,
    source_path: Path | None = None,
    prefilter_reports: list[Path] | None = None,
) -> dict[str, Any]:
    queue_report = audit_queue(queue_path.resolve(), base_core_path.resolve())
    plan_report = audit_plan(plan_path.resolve()) if plan_path else None
    ledger_report = (
        audit_ledger(ledger_path.resolve(), plan_path.resolve() if plan_path else None)
        if ledger_path
        else None
    )
    prefilter_report = (
        audit_prefilter(source_path.resolve(), [path.resolve() for path in prefilter_reports or []])
        if source_path and prefilter_reports
        else None
    )
    all_findings = list(queue_report["findings"])
    for report in (plan_report, ledger_report, prefilter_report):
        if report:
            all_findings.extend(report["findings"])
    return {
        "schema_version": "candidate-pipeline-contract-audit-v1",
        "status": "blocked" if any(item["severity"] == "error" for item in all_findings) else "passed",
        "read_only": True,
        "inputs": {
            "queue": queue_report,
            "plan": plan_report,
            "ledger": ledger_report,
            "prefilter": prefilter_report,
        },
        "findings": all_findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--base-core", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--source-suite", type=Path)
    parser.add_argument("--prefilter-report", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if bool(args.prefilter_report) != bool(args.source_suite):
        parser.error("--source-suite and --prefilter-report must be supplied together")
    try:
        report = build_report(
            queue_path=args.queue,
            base_core_path=args.base_core,
            plan_path=args.plan,
            ledger_path=args.ledger,
            source_path=args.source_suite,
            prefilter_reports=args.prefilter_report,
        )
        if args.output:
            output = args.output.resolve()
            if output.exists():
                raise ValueError(f"refusing to overwrite existing audit report: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "n_findings": len(report["findings"])}, sort_keys=True))
    return 0 if report["status"] == "passed" else 4


if __name__ == "__main__":
    raise SystemExit(main())
