"""Read-only migration inventory for Protocol-2.1 Traffic scenarios."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.protocol21_traffic_capture import (
    TRAFFIC_CAPTURE_SCHEMA_VERSION,
)
from core.protocol21_traffic_task import validate_traffic_task

TRAFFIC_MIGRATION_SCHEMA_VERSION = "1.0"
TASK_FIELDS = (
    "objective",
    "decision_space",
    "metrics",
    "success_condition",
    "failure_condition",
    "evaluation_horizon",
    "action_constraints",
    "capture_contract_version",
)


def inventory_traffic_scenario(
    scenario: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    """Inventory explicit migration evidence without inferring task fields."""
    backend_config = scenario.get("backend_config")
    if not isinstance(backend_config, Mapping):
        backend_config = {}
    provenance = scenario.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    task_contract = scenario.get("task_contract")
    if not isinstance(task_contract, Mapping):
        task_contract = {}

    runtime_backend = (
        scenario.get("backend_kind")
        or backend_config.get("backend_kind")
    )
    backend_contract_present = bool(runtime_backend and backend_config)
    source_asset_present = bool(
        provenance.get("source_locked") is True
        and provenance.get("files")
    )
    world_evolution_present = bool(
        scenario.get("world_evolution_contract")
        or scenario.get("perturbations")
    )
    control_surface_present = bool(
        scenario.get("control_surface")
        or backend_config.get("live_phase_control") is True
        or backend_config.get("sumo_corridor_program_map")
    )
    task_result = validate_traffic_task(task_contract)
    task_contract_present = task_result["status"] == "valid"
    capture_version = (
        scenario.get("capture_contract_version")
        or task_contract.get("capture_contract_version")
    )
    capture_present = (
        capture_version == TRAFFIC_CAPTURE_SCHEMA_VERSION
    )

    missing = []
    if not runtime_backend:
        missing.append("runtime_backend")
    if not backend_contract_present:
        missing.append("backend_contract")
    if not source_asset_present:
        missing.append("source_contract")
    if not control_surface_present:
        missing.append("control_surface")
    if not task_contract_present:
        missing.append("task_contract")
    if not capture_present:
        missing.append("capture_contract")

    if (
        backend_contract_present
        and source_asset_present
        and control_surface_present
        and task_contract_present
        and capture_present
    ):
        status = "ready"
    elif (
        backend_contract_present
        and source_asset_present
        and control_surface_present
    ):
        status = "partial"
    else:
        status = "blocked"
    return {
        "scenario_id": str(scenario.get("scenario_id") or ""),
        "path": str(path),
        "runtime_backend": runtime_backend,
        "backend_contract_present": backend_contract_present,
        "runtime_identity_present": bool(
            scenario.get("runtime_identity")
        ),
        "source_asset_present": source_asset_present,
        "world_evolution_present": world_evolution_present,
        "capture_contract_version": capture_version,
        "task_contract_present": task_contract_present,
        "task_fields_present": {
            field: bool(task_contract.get(field))
            for field in TASK_FIELDS
        },
        "control_surface_present": control_surface_present,
        "family": str(scenario.get("family") or "unknown"),
        "migration_status": status,
        "missing_fields": missing,
    }


def build_migration_inventory(
    scenarios: list[tuple[Path, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build scenario rows and domain-level distributions."""
    rows = [
        inventory_traffic_scenario(body, path=path)
        for path, body in scenarios
    ]
    status_counts = Counter(row["migration_status"] for row in rows)
    missing_counts = Counter(
        field for row in rows for field in row["missing_fields"]
    )

    def distribution(field: str) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for row in rows:
            key = str(row.get(field) or "missing")
            bucket = result.setdefault(
                key,
                {"total": 0, "ready": 0, "partial": 0, "blocked": 0},
            )
            bucket["total"] += 1
            bucket[row["migration_status"]] += 1
        return dict(sorted(result.items()))

    summary = {
        "total": len(rows),
        "ready": status_counts["ready"],
        "partial": status_counts["partial"],
        "blocked": status_counts["blocked"],
        "missing_task_contract": missing_counts["task_contract"],
        "missing_capture_contract": missing_counts["capture_contract"],
        "missing_source_contract": missing_counts["source_contract"],
        "missing_runtime_contract": (
            missing_counts["runtime_backend"]
            + missing_counts["backend_contract"]
        ),
        "by_backend": distribution("runtime_backend"),
        "by_family": distribution("family"),
    }
    return {
        "schema_version": TRAFFIC_MIGRATION_SCHEMA_VERSION,
        "summary": summary,
        "scenarios": rows,
    }
