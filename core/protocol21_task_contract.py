"""Fail-closed task semantics for Protocol-2.1 candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

TASK_CONTRACT_SCHEMA_VERSION = "1.0"


def validate_task_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute task-contract status without supplying missing semantics."""
    decision_space = contract.get("decision_space")
    if not isinstance(decision_space, Mapping):
        decision_space = {}
    metrics = contract.get("metrics")
    if not isinstance(metrics, Mapping):
        metrics = {}

    normalized = {
        "schema_version": str(
            contract.get("schema_version")
            or TASK_CONTRACT_SCHEMA_VERSION
        ),
        "scenario_id": str(contract.get("scenario_id") or ""),
        "objective": contract.get("objective"),
        "decision_space": dict(decision_space),
        "metrics": dict(metrics),
        "success_condition": contract.get("success_condition"),
        "failure_condition": contract.get("failure_condition"),
        "evaluation_horizon": contract.get("evaluation_horizon"),
    }
    required = {
        "scenario_id": normalized["scenario_id"],
        "objective": normalized["objective"],
        "metrics": normalized["metrics"],
        "success_condition": normalized["success_condition"],
        "failure_condition": normalized["failure_condition"],
        "evaluation_horizon": normalized["evaluation_horizon"],
        "decision_space.action_constraints": normalized[
            "decision_space"
        ].get("action_constraints"),
    }
    missing_fields = sorted(
        name for name, value in required.items() if not value
    )
    if missing_fields:
        status = "missing"
    elif contract.get("status") == "blocked":
        status = "blocked"
    else:
        status = "valid"
    return {
        **normalized,
        "status": status,
        "missing_fields": missing_fields,
    }
