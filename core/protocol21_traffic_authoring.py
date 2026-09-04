"""Fail-closed authoring boundary for Protocol-2.1 Traffic scenarios."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

TRAFFIC_AUTHORING_SCHEMA_VERSION = "1.0"

NATIVE_VERIFIED = "native_runtime_verified"
NATIVE_UNVERIFIED = "native_runtime_candidate_unverified"
PROVENANCE_BLOCKED = "provenance_blocked"
MOCK_DIAGNOSTIC = "mock_diagnostic_remaining"


def assess_authoring_eligibility(
    migration: Mapping[str, Any],
    *,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess authoring eligibility without granting admission."""
    migration_class = str(migration.get("primary_class") or "")
    evidence_state = {
        name: (
            binding.get(name, {}).get("status")
            if isinstance(binding, Mapping)
            else "missing"
        )
        for name in (
            "runtime_binding",
            "source_binding",
            "control_binding",
            "capture_binding",
            "task_binding",
        )
    }

    missing = []
    for name, status in evidence_state.items():
        if status != "complete":
            contract = name.removesuffix("_binding")
            if contract in {"capture", "task"}:
                contract = f"{contract}_contract"
            missing.append(contract)

    if migration_class == MOCK_DIAGNOSTIC:
        status = "diagnostic_only"
        actions = [
            "keep diagnostic exploration only",
            "prohibit formal benchmark authoring",
        ]
    elif migration_class == NATIVE_VERIFIED and not missing:
        status = "eligible"
        actions = ["request separate admission review"]
    elif migration_class == NATIVE_VERIFIED:
        status = "blocked"
        actions = [
            f"complete {contract}" for contract in missing
        ]
    elif migration_class == NATIVE_UNVERIFIED:
        status = "blocked"
        actions = ["produce candidate-specific native runtime proof"]
    else:
        status = "blocked"
        actions = ["recover provenance or retire as unavailable"]

    return {
        "schema_version": TRAFFIC_AUTHORING_SCHEMA_VERSION,
        "scenario_id": str(migration.get("scenario_id") or ""),
        "path": str(migration.get("path") or ""),
        "migration_class": migration_class,
        "authoring_status": status,
        "evidence_state": evidence_state,
        "missing_contracts": missing,
        "next_required_action": actions,
        "formal_authoring_allowed": status == "eligible",
        "admitted": False,
    }


def build_authoring_eligibility_report(
    migration_plan: Mapping[str, Any],
    *,
    bindings: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build eligibility rows and an evidence-first authoring queue."""
    supplied_bindings = bindings or {}
    scenarios = [
        assess_authoring_eligibility(
            row,
            binding=supplied_bindings.get(str(row.get("scenario_id") or "")),
        )
        for row in migration_plan.get("scenarios", [])
    ]
    status_counts = Counter(row["authoring_status"] for row in scenarios)
    class_rank = {
        NATIVE_VERIFIED: 0,
        NATIVE_UNVERIFIED: 1,
        PROVENANCE_BLOCKED: 2,
        MOCK_DIAGNOSTIC: 3,
    }
    source_rows = {
        str(row.get("scenario_id") or ""): row
        for row in migration_plan.get("scenarios", [])
    }
    queue = []
    ordered = sorted(
        scenarios,
        key=lambda row: (
            class_rank.get(row["migration_class"], 99),
            source_rows.get(row["scenario_id"], {}).get("path", ""),
        ),
    )
    for rank, row in enumerate(ordered, 1):
        queue.append(
            {
                "rank": rank,
                "scenario_id": row["scenario_id"],
                "scenario_ref": row["scenario_id"] or row["path"],
                "path": row["path"],
                "migration_class": row["migration_class"],
                "authoring_status": row["authoring_status"],
                "reason": (
                    "Evidence-first queue; queue is not admission."
                ),
                "missing_contracts": row["missing_contracts"],
                "next_required_action": row["next_required_action"],
            }
        )
    return {
        "schema_version": TRAFFIC_AUTHORING_SCHEMA_VERSION,
        "summary": {
            "total": len(scenarios),
            "eligible": status_counts["eligible"],
            "blocked": status_counts["blocked"],
            "diagnostic_only": status_counts["diagnostic_only"],
        },
        "scenarios": scenarios,
        "authoring_queue": queue,
    }
