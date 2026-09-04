"""Authoring contract for future Protocol-2.1 Traffic tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.protocol21_traffic_capture import (
    TRAFFIC_CAPTURE_SCHEMA_VERSION,
)

TRAFFIC_TASK_SCHEMA_VERSION = "1.0"
TRAFFIC_TASK_REQUIRED_FIELDS = (
    "scenario_id",
    "objective",
    "decision_space",
    "metrics",
    "success_condition",
    "failure_condition",
    "evaluation_horizon",
    "action_constraints",
    "capture_contract_version",
)


def traffic_task_schema() -> dict[str, Any]:
    """Return an empty field-only Traffic task template."""
    return {
        "schema_version": TRAFFIC_TASK_SCHEMA_VERSION,
        "scenario_id": "",
        "objective": None,
        "decision_space": {},
        "metrics": {},
        "success_condition": None,
        "failure_condition": None,
        "evaluation_horizon": {},
        "action_constraints": {},
        "capture_contract_version": TRAFFIC_CAPTURE_SCHEMA_VERSION,
        "status": "missing",
    }


def validate_traffic_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Validate author-supplied semantics without inferring an objective."""
    normalized = {
        "schema_version": str(
            task.get("schema_version") or TRAFFIC_TASK_SCHEMA_VERSION
        ),
        "scenario_id": str(task.get("scenario_id") or ""),
        "objective": task.get("objective"),
        "decision_space": (
            dict(task["decision_space"])
            if isinstance(task.get("decision_space"), Mapping)
            else {}
        ),
        "metrics": (
            dict(task["metrics"])
            if isinstance(task.get("metrics"), Mapping)
            else {}
        ),
        "success_condition": task.get("success_condition"),
        "failure_condition": task.get("failure_condition"),
        "evaluation_horizon": (
            dict(task["evaluation_horizon"])
            if isinstance(task.get("evaluation_horizon"), Mapping)
            else {}
        ),
        "action_constraints": (
            dict(task["action_constraints"])
            if isinstance(task.get("action_constraints"), Mapping)
            else {}
        ),
        "capture_contract_version": task.get(
            "capture_contract_version"
        ),
    }
    missing = [
        field
        for field in TRAFFIC_TASK_REQUIRED_FIELDS
        if normalized[field] in (None, "", [], {})
    ]
    semantic_keys = set(TRAFFIC_TASK_REQUIRED_FIELDS) & set(task)
    if not semantic_keys:
        reason = "task_contract_missing"
    elif normalized["objective"] is None:
        reason = "objective_missing"
    elif (
        normalized["capture_contract_version"]
        != TRAFFIC_CAPTURE_SCHEMA_VERSION
    ):
        reason = "task_capture_contract_mismatch"
    elif missing:
        reason = "task_contract_missing"
    else:
        reason = "traffic_task_contract_valid"
    status = "valid" if not missing and reason.endswith("_valid") else "blocked"
    return {
        **normalized,
        "status": status,
        "reason": reason,
        "missing_fields": sorted(missing),
    }
