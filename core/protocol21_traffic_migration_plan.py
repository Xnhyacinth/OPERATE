"""Evidence-faithful migration plans for Protocol-2.1 Traffic scenarios."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

TRAFFIC_MIGRATION_PLAN_VERSION = "1.0"

NATIVE_RUNTIME_VERIFIED = "native_runtime_verified"
NATIVE_RUNTIME_CANDIDATE = "native_runtime_candidate_unverified"
PROVENANCE_BLOCKED = "provenance_blocked"
MOCK_DIAGNOSTIC = "mock_diagnostic_remaining"

MIGRATION_ACTIONS = {
    NATIVE_RUNTIME_VERIFIED: [
        "complete task contract",
        "complete capture contract",
        "validate safety attribution",
        "define task evaluation metrics",
        "run headroom diagnostic",
        "run qualification",
    ],
    NATIVE_RUNTIME_CANDIDATE: [
        "reproduce candidate-specific runtime identity",
        "bind sumocfg network and route hashes",
        "verify native launch",
        "verify native control surface",
        "upgrade only after runtime proof passes",
    ],
    PROVENANCE_BLOCKED: [
        "recover provenance or retire as unavailable",
    ],
    MOCK_DIAGNOSTIC: [
        "keep diagnostic only",
        "prohibit formal benchmark admission",
        "prohibit leaderboard eligibility",
    ],
}


def runtime_proof_is_complete(proof: Mapping[str, Any] | None) -> bool:
    """Accept only candidate-specific native launch and source-hash proof."""
    if not isinstance(proof, Mapping):
        return False
    identity = proof.get("runtime_identity")
    control = proof.get("native_control_effect")
    if not isinstance(identity, Mapping) or not isinstance(control, Mapping):
        return False
    return bool(
        proof.get("candidate_kind")
        == "protocol21_traffic_runtime_candidate"
        and identity.get("complete") is True
        and identity.get("native_launch_passed") is True
        and identity.get("sumocfg_sha256")
        and identity.get("network_sha256")
        and identity.get("ordered_route_sha256s")
        and identity.get("sumo_version")
        and identity.get("transport")
        and control.get("native_control_effect_observed") is True
    )


def plan_traffic_scenario(
    row: Mapping[str, Any],
    *,
    runtime_proof: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one inventory row without inferring task semantics."""
    backend = row.get("runtime_backend")
    verified = runtime_proof_is_complete(runtime_proof)
    blocked = row.get("migration_status") == "blocked"

    if backend == "sumo" and verified:
        primary_class = NATIVE_RUNTIME_VERIFIED
    elif backend == "sumo":
        primary_class = NATIVE_RUNTIME_CANDIDATE
    elif blocked:
        primary_class = PROVENANCE_BLOCKED
    else:
        primary_class = MOCK_DIAGNOSTIC

    labels = []
    if backend == "sumo":
        labels.append("native_sumo_declared")
    if backend == "mock_sumo":
        labels.append("mock_backend")
    if verified:
        labels.append("runtime_verified")
    elif backend == "sumo":
        labels.extend(
            [
                "runtime_identity_missing",
                "candidate_specific_runtime_proof_missing",
            ]
        )
    if row.get("control_surface_present"):
        labels.append("native_control_available")
    if primary_class == PROVENANCE_BLOCKED:
        labels.append("provenance_blocked")
    if not row.get("task_contract_present"):
        labels.append("task_contract_missing")
    if not row.get("capture_contract_version"):
        labels.append("capture_contract_missing")
    if not row.get("source_asset_present"):
        labels.append("source_contract_missing")
    if not row.get("world_evolution_present"):
        labels.append("world_evidence_missing")

    return {
        "scenario_id": str(row.get("scenario_id") or ""),
        "path": str(row.get("path") or ""),
        "family": str(row.get("family") or "unknown"),
        "primary_class": primary_class,
        "diagnostic_labels": labels,
        "current_capabilities": {
            "runtime_backend": backend,
            "source_asset_present": bool(
                row.get("source_asset_present")
            ),
            "world_evolution_present": bool(
                row.get("world_evolution_present")
            ),
            "control_surface_present": bool(
                row.get("control_surface_present")
            ),
            "runtime_proof_complete": verified,
        },
        "missing_contracts": list(row.get("missing_fields") or []),
        "migration_actions": list(MIGRATION_ACTIONS[primary_class]),
        "ready": False,
        "admitted": False,
        "formal_allowed": False,
    }


def build_traffic_migration_plan(
    inventory: Mapping[str, Any],
    *,
    runtime_proofs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build exclusive primary classes and overlapping diagnostic labels."""
    proofs = runtime_proofs or {}
    scenarios = [
        plan_traffic_scenario(
            row,
            runtime_proof=proofs.get(str(row.get("scenario_id") or "")),
        )
        for row in inventory.get("scenarios", [])
    ]
    primary_counts = Counter(row["primary_class"] for row in scenarios)
    label_counts = Counter(
        label for row in scenarios for label in row["diagnostic_labels"]
    )
    class_rank = {
        NATIVE_RUNTIME_VERIFIED: 0,
        NATIVE_RUNTIME_CANDIDATE: 1,
        PROVENANCE_BLOCKED: 2,
        MOCK_DIAGNOSTIC: 3,
    }

    def priority_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        capabilities = row["current_capabilities"]
        family_rank = 0 if row.get("family") == "signal_coordination" else 1
        return (
            class_rank[row["primary_class"]],
            family_rank,
            not capabilities["source_asset_present"],
            not capabilities["control_surface_present"],
            str(row.get("path") or ""),
        )

    priority_order = []
    for rank, row in enumerate(sorted(scenarios, key=priority_key), 1):
        priority_order.append(
            {
                "rank": rank,
                "scenario_id": row["scenario_id"],
                "path": row["path"],
                "primary_class": row["primary_class"],
                "reason": (
                    "Evidence-first migration order; priority is not "
                    "admission."
                ),
                "missing_contracts": row["missing_contracts"],
            }
        )

    return {
        "schema_version": TRAFFIC_MIGRATION_PLAN_VERSION,
        "summary": {
            "total": len(scenarios),
            "primary_class_counts": dict(sorted(primary_counts.items())),
            "diagnostic_label_counts": dict(sorted(label_counts.items())),
        },
        "classes": {
            name: {"migration_actions": list(actions)}
            for name, actions in MIGRATION_ACTIONS.items()
        },
        "scenarios": scenarios,
        "priority_order": priority_order,
    }
