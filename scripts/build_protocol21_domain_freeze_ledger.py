#!/usr/bin/env python3
"""Build incremental domain evidence freezes without weakening atomic release."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.protocol21_admission_ledger import (  # noqa: E402
    row_prequalification_evidence_complete,
)
from core.protocol21_evidence import report_rows  # noqa: E402
from core.protocol21_qualification_cohort import (  # noqa: E402
    REQUIRED_FORMAL_DOMAINS,
)
from core.working_set_contract import (  # noqa: E402
    validate_protocol21_row_lineage,
)


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def build_domain_freeze_ledger(
    *,
    working_set: dict[str, Any],
    admission_ledger: dict[str, Any],
    backend_coverage: dict[str, Any],
) -> dict[str, Any]:
    """Freeze complete backend/domain evidence while keeping release atomic."""
    rows = report_rows(working_set)
    constraints = working_set.get("constraints")
    if not isinstance(constraints, dict):
        constraints = working_set.get("selection_constraints")
    constraints = constraints if isinstance(constraints, dict) else {}
    if constraints.get("candidate_evidence_merge_only") is True:
        freeze_scope = "candidate"
    elif constraints.get("backend_evidence_freeze_only") is True:
        freeze_scope = "backend"
    elif constraints.get("domain_evidence_freeze_only") is True:
        freeze_scope = "domain"
    else:
        freeze_scope = "suite"
    ledger_by_identity = {
        _identity(row): row
        for row in admission_ledger.get("rows") or []
        if isinstance(row, dict)
    }
    coverage = backend_coverage.get("by_backend")
    coverage = coverage if isinstance(coverage, dict) else {}
    rows_by_backend: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_backend.setdefault(
            str(row.get("backend_kind") or ""), []
        ).append(row)

    by_backend: dict[str, dict[str, Any]] = {}
    for backend, backend_rows in sorted(rows_by_backend.items()):
        reasons: set[str] = set()
        lineage_ready = sum(
            not validate_protocol21_row_lineage(row)
            for row in backend_rows
        )
        prequalified = sum(
            row_prequalification_evidence_complete(
                ledger_by_identity.get(_identity(row), {})
            )
            for row in backend_rows
        )
        item = coverage.get(backend)
        item = item if isinstance(item, dict) else {}
        if lineage_ready != len(backend_rows):
            reasons.add("selected_row_lineage_incomplete")
        if prequalified != len(backend_rows):
            reasons.add("selected_row_evidence_incomplete")
        if int(item.get("n_rows") or 0) != len(backend_rows):
            reasons.add("backend_coverage_denominator_mismatch")
        if int(item.get("n_runtime_validated_rows") or 0) != len(
            backend_rows
        ):
            reasons.add("selected_row_runtime_evidence_incomplete")
        if item.get("all_selected_rows_runtime_validated") is not True:
            reasons.add("selected_row_runtime_evidence_incomplete")
        if item.get("source_trace_complete") is not True:
            reasons.add("source_runtime_evidence_incomplete")
        if item.get("world_release_eligible") is not True:
            reasons.add("world_runtime_evidence_incomplete")
        if item.get("agent_action_backend_effect_observed") is not True:
            reasons.add("native_action_effect_evidence_incomplete")
        if item.get("deterministic_replay_verified") is not True:
            reasons.add("deterministic_replay_evidence_incomplete")
        if item.get("world_evolution_applicability") != "dynamic_native":
            reasons.add("dynamic_world_evolution_required")
        if item.get("fabricated_exogenous_events") is True:
            reasons.add("fabricated_exogenous_events_disallowed")
        by_backend[backend] = {
            "status": (
                "backend_evidence_frozen" if not reasons else "held"
            ),
            "n_selected_rows": len(backend_rows),
            "n_rows_lineage_ready": lineage_ready,
            "n_rows_prequalified": prequalified,
            "n_runtime_validated_rows": int(
                item.get("n_runtime_validated_rows") or 0
            ),
            "reason_codes": sorted(reasons),
        }

    rows_by_domain: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_domain.setdefault(str(row.get("domain") or ""), []).append(row)
    by_domain: dict[str, dict[str, Any]] = {}
    for domain, domain_rows in sorted(rows_by_domain.items()):
        backends = sorted(
            {str(row.get("backend_kind") or "") for row in domain_rows}
        )
        frozen = bool(backends) and all(
            by_backend[backend]["status"] == "backend_evidence_frozen"
            for backend in backends
        )
        domain_status = (
            "held"
            if not frozen
            else (
                "partial_backend_evidence"
                if freeze_scope == "backend"
                else (
                    "candidate_evidence_frozen"
                    if freeze_scope == "candidate"
                    else "domain_evidence_frozen"
                )
            )
        )
        by_domain[domain] = {
            "status": domain_status,
            "n_selected_rows": len(domain_rows),
            "backends": backends,
            "reason_codes": sorted(
                {
                    reason
                    for backend in backends
                    for reason in by_backend[backend]["reason_codes"]
                }
            ),
        }

    required = set(REQUIRED_FORMAL_DOMAINS)
    present = set(rows_by_domain)
    release_blockers = {
        f"missing_required_domain:{domain}" for domain in required - present
    }
    if freeze_scope != "suite":
        release_blockers.add(f"partial_freeze_scope:{freeze_scope}")
    if freeze_scope != "candidate":
        release_blockers.update(
            f"domain_evidence_not_frozen:{domain}"
            for domain in required & present
            if by_domain[domain]["status"] != "domain_evidence_frozen"
        )
    suite_prequalified = freeze_scope == "suite" and not release_blockers
    if suite_prequalified:
        release_blockers.add("formal_qualification_not_run")
    return {
        "schema_version": "2.1",
        "status": (
            "suite_prequalified"
            if suite_prequalified
            else "incremental_domain_evidence"
        ),
        "freeze_policy": (
            "all_selected_rows_current_dynamic_native_evidence_v1"
        ),
        "freeze_scope": freeze_scope,
        "required_formal_domains": list(REQUIRED_FORMAL_DOMAINS),
        "n_selected_rows": len(rows),
        "by_backend": by_backend,
        "by_domain": by_domain,
        "domain_status_summary": dict(
            sorted(
                Counter(item["status"] for item in by_domain.values()).items()
            )
        ),
        "suite_prequalified": suite_prequalified,
        "release_ready": False,
        "leaderboard_eligible": False,
        "release_blockers": sorted(release_blockers),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-set", type=Path, required=True)
    parser.add_argument("--admission-ledger", type=Path, required=True)
    parser.add_argument("--backend-coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_domain_freeze_ledger(
        working_set=json.loads(args.working_set.read_text(encoding="utf-8")),
        admission_ledger=json.loads(
            args.admission_ledger.read_text(encoding="utf-8")
        ),
        backend_coverage=json.loads(
            args.backend_coverage.read_text(encoding="utf-8")
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "suite_prequalified": report["suite_prequalified"],
                "release_ready": report["release_ready"],
                "release_blockers": report["release_blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
