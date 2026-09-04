#!/usr/bin/env python3
"""Summarize Wave-1 candidate gate outcomes without admitting Core rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = REPO_ROOT / "reports/protocol21_wave1_candidate_queue.json"
DEFAULT_LEDGER = (
    REPO_ROOT
    / "reports/protocol21_wave1_candidate_batches/candidate_batch_ledger.json"
)
DEFAULT_BASE_CORE = (
    REPO_ROOT
    / "reports/protocol21_pending_union_fresh_e18_realtraffic_v1"
    / "refined_core_selection_protocol2_v21.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "reports/protocol21_wave1_candidate_disposition_current.json"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_tree_sha256() -> str | None:
    try:
        from core.implementation_identity import implementation_identity

        return str(implementation_identity().get("implementation_tree_sha256") or "") or None
    except Exception:  # noqa: BLE001 - report remains explicit if unavailable
        return None


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _pipeline_summary(root: Path) -> dict[str, Any]:
    pipeline = root / "protocol21"
    core_path = pipeline / "refined_core_selection_protocol2_v21.json"
    readiness_path = pipeline / "protocol2_v21_core_readiness.json"
    contract_path = pipeline / "agentic_core_contract_protocol2_v21.json"
    core = _load(core_path) if core_path.is_file() else {}
    readiness = _load(readiness_path) if readiness_path.is_file() else {}
    contract = _load(contract_path) if contract_path.is_file() else {}
    rejected = core.get("rejected") if isinstance(core.get("rejected"), list) else []
    selected = core.get("scenarios") if isinstance(core.get("scenarios"), list) else []
    disposition_counts = Counter(
        str(row.get("disposition") or "unknown")
        for row in rejected
        if isinstance(row, dict)
    )
    reason_counts: Counter[str] = Counter()
    for row in rejected:
        if not isinstance(row, dict):
            continue
        values = row.get("reason_codes")
        if isinstance(values, list):
            reason_counts.update(str(value) for value in values)
        elif row.get("reason_code"):
            reason_counts[str(row["reason_code"])] += 1
    contract_summary = contract.get("summary")
    if isinstance(contract_summary, dict):
        for key, value in (contract_summary.get("blocker_counts") or {}).items():
            reason_counts[f"agentic:{key}"] += int(value)
    if selected:
        disposition = "candidate_survivor_requires_fresh_union"
    elif disposition_counts.get("retired_intrinsic"):
        disposition = "retired_intrinsic"
    else:
        disposition = "held_repair"
    return {
        "candidate_root": _display_path(root),
        "n_source": int(core.get("n_source") or len(rejected) + len(selected)),
        "n_selected": len(selected),
        "n_rejected": len(rejected),
        "n_secondary": int(core.get("n_secondary") or 0),
        "disposition": disposition,
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "pipeline_status": readiness.get("status") or "missing",
        "formal_evaluation_ready": bool(readiness.get("formal_evaluation_ready")),
        "leaderboard_eligible": bool(readiness.get("leaderboard_eligible")),
        "formal_run_blockers": list(readiness.get("formal_run_blockers") or []),
        "agentic_status_counts": (contract.get("summary") or {}).get("status_counts", {}),
    }


def build_report(
    *,
    queue_path: Path = DEFAULT_QUEUE,
    ledger_path: Path = DEFAULT_LEDGER,
    base_core_path: Path = DEFAULT_BASE_CORE,
) -> dict[str, Any]:
    queue = _load(queue_path)
    ledger = _load(ledger_path)
    base_core = _load(base_core_path)
    families: list[dict[str, Any]] = []
    for family in queue.get("families", []):
        if not isinstance(family, dict):
            continue
        root = Path(str(family["candidate_output_root"]))
        if not root.is_absolute():
            root = REPO_ROOT / root
        summary = _pipeline_summary(root)
        summary.update(
            {
                "family": family.get("family"),
                "domain": family.get("domain"),
                "backend": family.get("backend"),
                "source_suite_sha256": family.get("source_suite_sha256"),
            }
        )
        families.append(summary)
    items = ledger.get("items") if isinstance(ledger.get("items"), list) else []
    terminal = [
        item
        for item in items
        if isinstance(item, dict) and item.get("work_state") == "terminal"
    ]
    selected_total = sum(int(family["n_selected"]) for family in families)
    return {
        "schema_version": "protocol21-wave1-candidate-disposition-v1",
        "status": "candidate_terminal_no_increment" if selected_total == 0 else "candidate_survivors_require_final_union",
        "release_admission": False,
        "promotion_allowed": False,
        "base_core": {
            "path": _display_path(base_core_path),
            "sha256": _sha256(base_core_path),
            "n_core": len(base_core.get("scenarios") or []),
            "status": base_core.get("status"),
        },
        "implementation_tree_sha256": _implementation_tree_sha256(),
        "queue": {
            "path": _display_path(queue_path),
            "sha256": _sha256(queue_path),
            "schema_version": queue.get("schema_version"),
        },
        "ledger": {
            "path": _display_path(ledger_path),
            "sha256": _sha256(ledger_path),
            "status": ledger.get("status"),
            "n_scheduled": ledger.get("n_scheduled"),
            "n_terminal": len(terminal),
        },
        "families": sorted(families, key=lambda row: str(row.get("family"))),
        "totals": {
            "n_source": sum(int(row["n_source"]) for row in families),
            "n_selected": selected_total,
            "n_rejected": sum(int(row["n_rejected"]) for row in families),
            "n_secondary": sum(int(row["n_secondary"]) for row in families),
        },
        "policy": {
            "locked_core_changed": False,
            "llm_outcome_used_for_admission": False,
            "failed_science_gate_retried_identically": False,
            "final_union_required": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--base-core", type=Path, default=DEFAULT_BASE_CORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report(
        queue_path=args.queue,
        ledger_path=args.ledger,
        base_core_path=args.base_core,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "totals": report["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
