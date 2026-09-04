"""Protocol-2.1 evidence binding without task-semantic inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PROTOCOL21_SCENARIO_BINDING_VERSION = "1.0"
TASK_BINDING_FIELDS = [
    "objective",
    "decision_space",
    "metrics",
    "success_condition",
    "failure_condition",
    "evaluation_horizon",
    "action_constraints",
    "capture_contract_version",
]


def build_scenario_binding(
    *,
    scenario_id: str,
    runtime_binding: Mapping[str, Any],
    source_binding: Mapping[str, Any],
    control_binding: Mapping[str, Any],
    capture_binding: Mapping[str, Any],
    task_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind supplied evidence while keeping admission fail-closed."""
    if task_binding is None:
        bound_task = {
            "status": "missing",
            "contract": None,
            "missing_fields": list(TASK_BINDING_FIELDS),
        }
    else:
        bound_task = dict(task_binding)

    reasons = []
    if runtime_binding.get("status") != "complete":
        reasons.append("runtime_binding_incomplete")
    if source_binding.get("status") != "complete":
        reasons.append("source_binding_incomplete")
    if control_binding.get("status") != "complete":
        reasons.append("control_binding_incomplete")
    if capture_binding.get("status") != "complete":
        reasons.append("capture_contract_incomplete")
    if "safety_attribution" in (capture_binding.get("blocks") or []):
        reasons.append("safety_context_missing")
    if bound_task.get("status") != "complete":
        reasons.append("task_contract_missing")

    evaluation_allowed = not reasons
    return {
        "schema_version": PROTOCOL21_SCENARIO_BINDING_VERSION,
        "scenario_id": scenario_id,
        "runtime_binding": dict(runtime_binding),
        "source_binding": dict(source_binding),
        "control_binding": dict(control_binding),
        "capture_binding": dict(capture_binding),
        "task_binding": bound_task,
        "evaluation_binding": {
            "status": "allowed" if evaluation_allowed else "blocked",
            "evaluation_allowed": evaluation_allowed,
            "reason": reasons,
        },
        "admission_status": "blocked",
        "admitted": False,
    }
