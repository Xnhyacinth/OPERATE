#!/usr/bin/env python3
"""Classify Protocol-2.1 agentic-contract rejects without changing gates.

This is a report-only remediation queue.  It deliberately does not inspect
LLM trajectories or infer that a low model score means a bad scenario.  The
queue follows the evidence dependency order from source/runtime evidence to
post-change decisions, then labels each row as a repair candidate, a pending
repair, or a definitive retirement.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

# Ordered from prerequisites to downstream evidence.  Blockers in the source
# report are not sorted by dependency, so this table is the single ordering
# used for the ``first_missing_evidence_edge`` field.
EVIDENCE_EDGES: tuple[dict[str, Any], ...] = (
    {
        "code": "artifact_identity",
        "blockers": (
            "artifact_semantics_stale",
            "artifact_identity_mismatch",
            "scenario_signature_current",
        ),
        "checks": (
            "current_protocol_semantics",
            "identity_bound_across_artifacts",
            "scenario_signature_current",
        ),
        "required_evidence": (
            "current protocol/scoring semantics",
            "scenario_id and scenario_signature bound across every artifact",
        ),
        "repair_action": "re-run the same replay artifacts and bind one implementation fingerprint",
    },
    {
        "code": "source_lock",
        "blockers": (
            "source_lock_failed",
            "source_lock_unproven",
            "source_lock_evidence_missing",
            "check_failed:source_lock_passed",
        ),
        "checks": ("source_lock_passed",),
        "required_evidence": ("locked source path/hash/license/version",),
        "repair_action": "rebuild the source-lock evidence from the native asset graph",
    },
    {
        "code": "source_consumption",
        "blockers": (
            "source_consumption_failed",
            "source_consumption_unproven",
            "check_failed:source_consumption_passed",
        ),
        "checks": ("source_consumption_passed",),
        "required_evidence": (
            "runtime-consumed source channel, derived state, tick, and replay hash",
        ),
        "repair_action": "repair the backend trace or replace with a source-consuming adapter",
    },
    {
        "code": "deterministic_replay",
        "blockers": (
            "deterministic_replay_failed",
            "deterministic_replay_evidence_missing",
            "check_failed:deterministic_replay_passed",
        ),
        "checks": ("deterministic_replay_passed",),
        "required_evidence": ("same-seed deterministic replay fingerprints",),
        "repair_action": "rerun the bounded replay with locked seed and runtime inputs",
    },
    {
        "code": "native_backend_execution",
        "blockers": (
            "native_backend_execution_unproven",
            "check_failed:native_backend_executable",
        ),
        "checks": ("native_backend_executable",),
        "required_evidence": ("native simulator execution and state trace",),
        "repair_action": "repair the native adapter; replace only with an independently runnable source",
    },
    {
        "code": "event_reachability",
        "blockers": (
            "predesigned_event_unreachable",
            "predesigned_event_not_reached",
            "check_failed:predesigned_change_or_disruption_reached",
            "check_failed:event_or_change_occurs_after_initial_state",
        ),
        "checks": (
            "predesigned_change_or_disruption_reached",
            "event_or_change_occurs_after_initial_state",
        ),
        "required_evidence": (
            "declared event realized on a non-terminal tick",
            "at least one executable response opportunity after the event",
        ),
        "repair_action": "repair event timing/window from the same source or replace the static row",
    },
    {
        "code": "world_evolution",
        "blockers": (
            "evidence_missing_runtime_evolution",
            "world_change_contract_missing",
            "material_exogenous_change_unproven",
            "check_failed:material_exogenous_change_observed",
            "check_failed:exogenous_state_evolution_observed",
            "parallel_clock_not_observed",
            "check_failed:parallel_simulator_agent_clock_observed",
        ),
        "checks": (
            "world_change_contract_declared",
            "material_exogenous_change_observed",
            "exogenous_state_evolution_observed",
            "parallel_simulator_agent_clock_observed",
        ),
        "required_evidence": (
            "simulator-owned clock and evolving native state",
            "material exogenous event record consumed by the backend",
        ),
        "repair_action": "rerun the backend trace and add only source-grounded world-evolution evidence",
    },
    {
        "code": "post_change_decision",
        "blockers": (
            "post_change_decision_unproven",
            "check_failed:post_change_decision_observed",
        ),
        "checks": ("post_change_decision_observed",),
        "required_evidence": ("decision tick and action edge after the change",),
        "repair_action": "repair the response window and replay the existing task contract",
    },
    {
        "code": "agent_action_effect",
        "blockers": (
            "agent_action_backend_effect_unproven",
            "evidence_missing_native_control_use",
            "check_failed:agent_action_backend_effect_observed",
            "check_failed:successful_reference_used_native_control",
        ),
        "checks": (
            "agent_action_backend_effect_observed",
            "successful_reference_used_native_control",
        ),
        "required_evidence": (
            "state-changing native tool call",
            "later simulator state effect linked to that call",
        ),
        "repair_action": "replay the native control surface; do not treat acknowledgement as a state effect",
    },
    {
        "code": "adaptive_response",
        "blockers": (
            "adaptive_replanning_or_delegation_unproven",
            "standing_plan_response_unproven",
            "event_adaptive_cadence_undeclared",
            "check_failed:event_adaptive_cadence_declared",
            "check_failed:difficulty_appropriate_control_response_observed",
        ),
        "checks": (
            "event_adaptive_cadence_declared",
            "difficulty_appropriate_control_response_observed",
        ),
        "required_evidence": (
            "standing-plan monitoring at the declared cadence",
            "active replanning/delegation for High and Extreme",
        ),
        "repair_action": "rerun the same task with simulator-owned time and evidence-linked adaptive wakeups",
    },
    {
        "code": "task_contract",
        "blockers": (
            "task_contract_failed",
            "task_contract_evidence_missing",
            "check_failed:task_contract_passed",
        ),
        "checks": ("task_contract_passed",),
        "required_evidence": ("completed native task contract and terminal state",),
        "repair_action": "repair the task contract only when the native task remains solvable; otherwise replace",
    },
    {
        "code": "terminal_integrity",
        "blockers": (
            "terminal_integrity_failed",
            "check_failed:terminal_integrity_passed",
        ),
        "checks": ("terminal_integrity_passed",),
        "required_evidence": ("terminal integrity and response-window closure",),
        "repair_action": "repair terminal handling and replay; do not conceal unanswered terminal interrupts",
    },
    {
        "code": "reference_headroom",
        "blockers": (
            "reference_headroom_nonpositive",
            "reference_headroom_unproven",
            "check_failed:material_reference_headroom",
        ),
        "checks": ("material_reference_headroom",),
        "required_evidence": ("wait/greedy/oracle native headroom with aligned units",),
        "repair_action": "replace an inert source or repair the aligned baseline replay; never lower the threshold",
    },
    {
        "code": "depth_proof",
        "blockers": (
            "strategy_depth_contradicted",
            "check_failed:bounded_or_exact_depth_available",
            "check_failed:decision_graph_present",
            "check_failed:decision_graph_acyclic",
        ),
        "checks": (
            "bounded_or_exact_depth_available",
            "decision_graph_present",
            "decision_graph_acyclic",
            "difficulty_not_contradicted",
        ),
        "required_evidence": ("bounded/exact dependency depth and acyclic decision graph",),
        "repair_action": "rerun depth minimization with evidence edges; retire only a contradicted difficulty contract",
    },
    {
        "code": "reference_baselines",
        "blockers": (
            "check_failed:oracle_result_present",
            "check_failed:greedy_result_present",
            "check_failed:wait_reference_present",
        ),
        "checks": ("oracle_result_present", "greedy_result_present", "wait_reference_present"),
        "required_evidence": ("wait, greedy, and oracle baseline replays",),
        "repair_action": "rerun missing baselines under the same implementation fingerprint",
    },
)

_HARD_RETIRE_BLOCKERS = frozenset(
    {
        "deterministic_replay_failed",
        "task_contract_failed",
        "reference_headroom_nonpositive",
        "source_consumption_failed",
        "predesigned_event_unreachable",
        "strategy_depth_contradicted",
    }
)
_IDENTITY_BLOCKERS = frozenset(EVIDENCE_EDGES[0]["blockers"])

# A replay repair may only promote a row after every evidence edge is true.
# Keep this derived from ``EVIDENCE_EDGES`` so adding a new gate cannot
# accidentally leave the remediation pass weaker than the release pipeline.
REPLAY_REQUIRED_CHECKS: tuple[str, ...] = tuple(
    dict.fromkeys(
        str(check)
        for edge in EVIDENCE_EDGES
        for check in edge["checks"]
    )
)
_REPLAY_ORIGINS = frozenset({"source_schedule", "declared_perturbation"})
_LOGISTICS_REPLAY_BACKENDS = frozenset({"pyvrp_cvrp", "pyvrp_vrptw"})
_WORLD_REPAIR_CHECKS = frozenset(
    {
        "material_exogenous_change_observed",
        "post_change_decision_observed",
        "difficulty_appropriate_control_response_observed",
    }
)


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "samples", "scenarios"):
        value = report.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _check_blockers(row: dict[str, Any]) -> list[str]:
    blockers = [str(value) for value in row.get("blockers") or [] if str(value)]
    checks = row.get("checks") or {}
    if isinstance(checks, dict):
        blockers.extend(
            f"check_failed:{key}"
            for key, value in checks.items()
            if value is False
        )
    return list(dict.fromkeys(blockers))


def first_missing_evidence_edge(row: dict[str, Any]) -> dict[str, Any]:
    """Return the first unmet evidence edge in dependency order."""
    blockers = _check_blockers(row)
    for edge in EVIDENCE_EDGES:
        edge_blockers = set(edge["blockers"])
        matched = [blocker for blocker in blockers if blocker in edge_blockers]
        matched_checks = {
            f"check_failed:{check}"
            for check in edge["checks"]
            if f"check_failed:{check}" in blockers
        }
        matched = list(dict.fromkeys(matched + sorted(matched_checks)))
        if matched:
            return {
                "code": edge["code"],
                "blocked_by": matched,
                "required_evidence": list(edge["required_evidence"]),
                "repair_action": edge["repair_action"],
            }
    return {
        "code": "unclassified",
        "blocked_by": blockers,
        "required_evidence": ["review the first missing Protocol-2.1 evidence edge"],
        "repair_action": "manual evidence review; do not promote or relax a gate",
    }


def _only_identity_edge(row: dict[str, Any], edge: dict[str, Any]) -> bool:
    if edge["code"] != "artifact_identity":
        return False
    return set(_check_blockers(row)).issubset(_IDENTITY_BLOCKERS)


def _is_hard_retire(row: dict[str, Any]) -> bool:
    if str(row.get("status") or "") == "retired":
        return True
    return bool(_HARD_RETIRE_BLOCKERS.intersection(_check_blockers(row)))


def _model_failure_ignored(row: dict[str, Any]) -> bool:
    values = (
        row.get("model_status"),
        row.get("model_failure"),
        row.get("llm_status"),
    )
    return any(
        isinstance(value, str)
        and value.lower() in {"failed", "error", "timeout", "incomplete"}
        for value in values
    )


def _disposition(edge_code: str) -> str:
    if edge_code in {"source_lock", "native_backend_execution", "reference_headroom"}:
        return "replace"
    return "repair"


def build_remediation_queue(report: dict[str, Any]) -> dict[str, Any]:
    """Build a compact queue from one agentic-contract report."""
    if not isinstance(report, dict):
        raise TypeError("agentic report must be a mapping")
    rows = _rows(report)
    items: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: str(value.get("scenario_id") or "")):
        if str(row.get("status") or "") == "passed":
            continue
        edge = first_missing_evidence_edge(row)
        hard_retire = _is_hard_retire(row)
        if hard_retire:
            status = "retire"
            disposition = "retire"
        elif _only_identity_edge(row, edge):
            status = "repaired_candidate"
            disposition = "repair"
        else:
            status = "repair_pending"
            disposition = _disposition(str(edge["code"]))
        blockers = _check_blockers(row)
        items.append(
            {
                "scenario_id": str(row.get("scenario_id") or ""),
                "scenario_signature": str(row.get("scenario_signature") or ""),
                "domain": row.get("domain"),
                "backend_kind": row.get("backend_kind"),
                "difficulty_level": row.get("difficulty_level"),
                "current_contract_status": row.get("status"),
                "status": status,
                "disposition": disposition,
                "first_missing_evidence_edge": edge,
                "remaining_blockers": blockers,
                "model_failure_ignored": _model_failure_ignored(row),
            }
        )
    counts = Counter(str(item["status"]) for item in items)
    edge_counts = Counter(
        str(item["first_missing_evidence_edge"]["code"]) for item in items
    )
    expected = int(report.get("n_expected") or len(rows))
    return {
        "schema_version": "protocol21-agentic-remediation-v1",
        "status": "open" if items else "empty",
        "source_report_status": report.get("status"),
        "n_input": expected,
        "n_rows_seen": len(rows),
        "n_remediation_items": len(items),
        "passed_excluded": sum(
            str(row.get("status") or "") == "passed" for row in rows
        ),
        "status_counts": {
            "repair_pending": int(counts.get("repair_pending", 0)),
            "repaired_candidate": int(counts.get("repaired_candidate", 0)),
            "retire": int(counts.get("retire", 0)),
        },
        "first_missing_evidence_edge_counts": dict(sorted(edge_counts.items())),
        "policy": {
            "first_missing_edge_order": [str(edge["code"]) for edge in EVIDENCE_EDGES],
            "thresholds_unchanged": True,
            "frozen_core_mutation": False,
            "model_failure_policy": "model failures are diagnostic only and never trigger retire",
        },
        "model_failure_policy": {
            "model_failures_trigger_retire": False,
            "model_failures_are_data_quality_evidence": False,
        },
        "items": items,
    }


def _replay_identity(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )


def _strict_replay_proof(row: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate an explicit replay proof without inferring missing evidence.

    This intentionally accepts only a small, machine-readable contract.  A
    passed row or a source declaration by itself is not proof that the replay
    consumed a material event and then observed an action-linked consequence.
    """

    blockers: list[str] = []
    if str(row.get("status") or "") != "passed":
        blockers.append("replay_status_not_passed")
    checks = row.get("checks")
    if not isinstance(checks, dict):
        blockers.append("replay_checks_missing")
    else:
        blockers.extend(
            f"replay_check_failed:{check}"
            for check in REPLAY_REQUIRED_CHECKS
            if checks.get(check) is not True
        )

    evidence = row.get("replay_evidence")
    if not isinstance(evidence, dict):
        blockers.append("replay_evidence_missing")
        return False, list(dict.fromkeys(blockers)), {}

    replay_count = evidence.get("replay_count")
    if isinstance(replay_count, bool) or not isinstance(replay_count, int) or replay_count < 2:
        blockers.append("replay_count_insufficient")
    if evidence.get("deterministic") is not True:
        blockers.append("replay_determinism_unproven")

    digests = evidence.get("trace_semantic_digests")
    if (
        not isinstance(digests, list)
        or len(digests) < 2
        or any(not isinstance(digest, str) or not digest for digest in digests)
        or len(set(digests)) != 1
    ):
        blockers.append("replay_trace_digest_mismatch")

    if evidence.get("source_consumption_proven") is not True:
        blockers.append("replay_source_consumption_unproven")
    if evidence.get("agent_action_backend_effect_observed") is not True:
        blockers.append("replay_action_effect_unproven")
    native_action_effects = evidence.get("native_action_effects")
    if not isinstance(native_action_effects, list) or not any(
        isinstance(effect, dict)
        and isinstance(effect.get("action_to_outcome_edge"), dict)
        for effect in native_action_effects
    ):
        blockers.append("replay_native_action_edges_missing")

    records = evidence.get("world_evolution_records")
    global_event_edges = evidence.get("event_to_decision_action_edges")
    if not isinstance(global_event_edges, list):
        global_event_edges = []
    if not isinstance(records, list) or not records:
        blockers.append("world_evolution_records_missing")
        records = []
    else:
        for index, record in enumerate(records):
            prefix = f"world_record_{index}"
            if not isinstance(record, dict):
                blockers.append(f"{prefix}_malformed")
                continue
            if record.get("origin") not in _REPLAY_ORIGINS:
                blockers.append("world_event_origin_unproven")
            if record.get("material_exogenous") is not True:
                blockers.append(f"{prefix}_not_material_exogenous")
            changed_fields = record.get("changed_state_fields")
            if not isinstance(changed_fields, list) or not changed_fields:
                blockers.append(f"{prefix}_state_effect_missing")
            applied_tick = record.get("applied_tick")
            response_tick = record.get("response_opportunity_tick")
            decision_tick = record.get("later_decision_tick")
            if (
                isinstance(applied_tick, bool)
                or not isinstance(applied_tick, int)
                or isinstance(response_tick, bool)
                or not isinstance(response_tick, int)
                or response_tick <= applied_tick
            ):
                blockers.append(f"{prefix}_response_window_missing")
            if (
                isinstance(applied_tick, bool)
                or not isinstance(applied_tick, int)
                or isinstance(decision_tick, bool)
                or not isinstance(decision_tick, int)
                or decision_tick <= applied_tick
            ):
                blockers.append(f"{prefix}_post_change_decision_missing")
            action_edges = record.get("action_to_outcome_edges") or record.get(
                "event_to_action_edges"
            )
            linked_global_edges = [
                edge
                for edge in global_event_edges
                if isinstance(edge, dict)
                and str(edge.get("source_event_id") or "")
                == str(record.get("event_id") or "")
                and isinstance(edge.get("target_tick"), int)
                and edge["target_tick"] > applied_tick
            ]
            if (
                (not isinstance(action_edges, list) or not action_edges)
                and not linked_global_edges
            ):
                blockers.append(f"{prefix}_action_edge_missing")

    summary = {
        "replay_count": replay_count,
        "deterministic": evidence.get("deterministic") is True,
        "trace_semantic_digest": (
            digests[0] if isinstance(digests, list) and digests else None
        ),
        "n_world_evolution_records": len(records),
        "n_event_to_decision_action_edges": len(global_event_edges),
        "world_origins": sorted(
            {
                str(record.get("origin"))
                for record in records
                if isinstance(record, dict) and record.get("origin")
            }
        ),
        "event_ids": [
            str(record.get("event_id"))
            for record in records
            if isinstance(record, dict) and record.get("event_id")
        ],
        "applied_ticks": [
            record.get("applied_tick")
            for record in records
            if isinstance(record, dict)
        ],
        "response_ticks": [
            record.get("response_opportunity_tick")
            for record in records
            if isinstance(record, dict)
        ],
    }
    unique_blockers = list(dict.fromkeys(blockers))
    return not unique_blockers, unique_blockers, summary


def _report_index(report: dict[str, Any] | None) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not isinstance(report, dict):
        return index
    for row in _rows(report):
        index.setdefault(_replay_identity(row), []).append(row)
    return index


def _load_scenario_source(
    source_row: dict[str, Any], repo_root: Path
) -> tuple[dict[str, Any], Path | None, str | None]:
    """Load one source-locked YAML row without treating declarations as runtime proof."""

    raw_path = str(source_row.get("path") or "")
    if not raw_path:
        return {}, None, None
    path = Path(raw_path)
    if not path.is_absolute():
        path = repo_root / path
    try:
        resolved = path.resolve()
        payload = resolved.read_bytes()
    except OSError:
        return {}, None, None
    try:
        import yaml

        parsed = yaml.safe_load(payload) or {}
    except (ImportError, OSError, ValueError):
        return {}, resolved, hashlib.sha256(payload).hexdigest()
    return (
        parsed if isinstance(parsed, dict) else {},
        resolved,
        hashlib.sha256(payload).hexdigest(),
    )


def _logistics_changed_fields(
    *,
    event_kind: str,
    scenario: dict[str, Any],
    perturbation: dict[str, Any],
) -> list[str]:
    """Map a native route event to observable backend state fields.

    These names mirror ``RouteSim.snapshot`` and scoring records.  The mapping
    is deliberately limited to event kinds whose native implementation is
    present; unknown events are held rather than guessed.
    """

    config = scenario.get("backend_config") or {}
    network = config.get("network") or {}
    target = perturbation.get("target") or {}
    if event_kind == "vehicle_breakdown":
        vehicle_index = target.get("vehicle_index")
        if isinstance(vehicle_index, bool) or not isinstance(vehicle_index, int):
            return []
        return [
            f"entities.v{vehicle_index}.broken",
            "totals.n_failed_routes",
        ]
    if event_kind == "blocked_arc":
        customer_index = target.get("customer_index")
        customers = network.get("customers") or []
        if (
            isinstance(customer_index, bool)
            or not isinstance(customer_index, int)
            or not isinstance(customers, list)
            or not (0 <= customer_index < len(customers))
        ):
            return []
        customer = customers[customer_index]
        customer_id = str(customer.get("id") or "") if isinstance(customer, dict) else ""
        if not customer_id:
            return []
        return [
            f"entities.{customer_id}.blocked",
            "totals.n_failed_routes",
        ]
    if event_kind == "demand_surge":
        return ["totals.aggregate_demand_units", "totals.unmet_demand_units"]
    if event_kind == "traffic_delay":
        return ["totals.routing_operating_cost"]
    if event_kind == "urgent_order":
        return ["entities.urgent_order", "totals.aggregate_demand_units"]
    return []


def _logistics_materiality(
    event_kind: str, perturbation: dict[str, Any]
) -> tuple[str, float, float] | None:
    try:
        intensity = float(perturbation.get("intensity", 1.0))
    except (TypeError, ValueError):
        return None
    if event_kind in {"vehicle_breakdown", "blocked_arc"}:
        return "failed_route_count", 1.0, 1.0
    if event_kind == "demand_surge":
        return "demand_surge_fraction", max(0.0, intensity - 1.0), 0.01
    if event_kind == "traffic_delay":
        return "travel_cost_multiplier", max(0.0, intensity - 1.0), 0.01
    if event_kind == "urgent_order":
        return "priority_order_injection", intensity, 1.0
    return None


def _runtime_event_matches(
    event_kind: str, runtime_event_ids: list[Any]
) -> list[str]:
    matches: list[str] = []
    for raw_id in runtime_event_ids:
        event_id = str(raw_id or "")
        if event_id == event_kind or event_id.startswith(f"{event_kind}:"):
            matches.append(event_id)
    return matches


def _build_logistics_replay_evidence(
    *,
    agentic_row: dict[str, Any],
    behavioral_row: dict[str, Any],
    complexity_row: dict[str, Any],
    source_row: dict[str, Any],
    repo_root: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Build auditable replay evidence for native VRP perturbations only."""

    blockers: list[str] = []
    if agentic_row.get("backend_kind") not in _LOGISTICS_REPLAY_BACKENDS:
        blockers.append("unsupported_logistics_replay_backend")
        return None, blockers
    if str(behavioral_row.get("status") or "") != "passed":
        blockers.append("replay_behavior_not_passed")
    behavior_replay = behavioral_row.get("replay_evidence") or {}
    wait_first = str(behavior_replay.get("wait_fingerprint_first") or "")
    wait_second = str(behavior_replay.get("wait_fingerprint_second") or "")
    if not wait_first or wait_first != wait_second:
        blockers.append("replay_wait_fingerprint_mismatch")

    source_contract = (
        agentic_row.get("agentic_contract", {}).get("source_consumption_contract")
        or {}
    )
    world_contract = (
        agentic_row.get("agentic_contract", {}).get("world_evolution_contract")
        or {}
    )
    if source_contract.get("status") != "passed":
        blockers.append("replay_source_consumption_unproven")
    if source_contract.get("deterministic_across_replays") is not True:
        blockers.append("replay_source_determinism_unproven")
    if not source_contract.get("trace_semantic_digest"):
        blockers.append("replay_source_trace_digest_missing")
    if world_contract.get("world_change_contract_declared") is not True:
        blockers.append("replay_world_contract_missing")
    if world_contract.get("simulator_owned_clock_observed") is not True:
        blockers.append("replay_simulator_clock_missing")
    if world_contract.get("agent_action_backend_effect_observed") is not True:
        blockers.append("replay_action_effect_unproven")

    scenario, scenario_path, scenario_sha256 = _load_scenario_source(
        source_row, repo_root
    )
    perturbations = scenario.get("perturbations")
    if not isinstance(perturbations, list) or not perturbations:
        blockers.append("logistics_replay_event_proof_missing")
        return None, blockers
    runtime_ids = list(world_contract.get("realized_predesigned_event_ids") or [])
    exogenous_ticks = {
        int(tick)
        for tick in world_contract.get("exogenous_state_change_ticks") or []
        if isinstance(tick, int) and not isinstance(tick, bool)
    }
    complexity_observed = complexity_row.get("observed") or {}
    decision_ticks = sorted(
        {
            int(tick)
            for tick in complexity_observed.get("model_decision_ticks") or []
            if isinstance(tick, int) and not isinstance(tick, bool)
        }
    )
    action_records = [
        record
        for record in complexity_observed.get("agent_caused_event_records") or []
        if isinstance(record, dict)
    ]
    if not decision_ticks:
        blockers.append("replay_post_change_decision_missing")
    if not action_records:
        blockers.append("replay_action_edge_missing")
    horizon = int(scenario.get("horizon_ticks") or 0)
    records: list[dict[str, Any]] = []
    event_edges: list[dict[str, Any]] = []
    runtime_event_cursors: dict[str, int] = {}
    for perturbation in perturbations:
        if not isinstance(perturbation, dict):
            blockers.append("logistics_replay_event_declaration_malformed")
            continue
        event_kind = str(perturbation.get("kind") or "")
        matches = _runtime_event_matches(event_kind, runtime_ids)
        trigger_tick = perturbation.get("trigger_tick")
        if not event_kind or not isinstance(trigger_tick, int) or isinstance(trigger_tick, bool):
            blockers.append("logistics_replay_event_declaration_malformed")
            continue
        match_index = runtime_event_cursors.get(event_kind, 0)
        if match_index >= len(matches) or trigger_tick not in exogenous_ticks:
            blockers.append(f"logistics_replay_event_not_observed:{event_kind}")
            continue
        runtime_event_cursors[event_kind] = match_index + 1
        changed_fields = _logistics_changed_fields(
            event_kind=event_kind,
            scenario=scenario,
            perturbation=perturbation,
        )
        materiality = _logistics_materiality(event_kind, perturbation)
        if not changed_fields or materiality is None:
            blockers.append(f"logistics_replay_event_mapping_missing:{event_kind}")
            continue
        metric, value, threshold = materiality
        response_tick = trigger_tick + 1
        later_decisions = [tick for tick in decision_ticks if tick >= response_tick]
        if horizon <= response_tick or not later_decisions:
            blockers.append(f"logistics_replay_response_window_missing:{event_kind}")
            continue
        later_actions = [
            record
            for record in action_records
            if isinstance(record.get("applied_tick"), int)
            and record["applied_tick"] >= response_tick
        ]
        if not later_actions:
            blockers.append(f"logistics_replay_action_after_event_missing:{event_kind}")
            continue
        event_id = matches[match_index]
        record = {
            "event_id": event_id,
            "event_type": event_kind,
            "origin": "declared_perturbation",
            "declared_event": dict(perturbation),
            "applied_tick": trigger_tick,
            "visibility": "hidden" if perturbation.get("hidden") else "visible",
            "decision_required": True,
            "changed_state_fields": changed_fields,
            "materiality_metric": metric,
            "materiality_value": value,
            "materiality_threshold": threshold,
            "materiality_passed": value >= threshold,
            "material_exogenous": value >= threshold,
            "response_window_required": True,
            "response_opportunity_tick": response_tick,
            "later_decision_tick": later_decisions[0],
            "terminal_response_window_missing": False,
            "runtime_event_ids": matches,
        }
        if not record["material_exogenous"]:
            blockers.append(f"logistics_replay_event_not_material:{event_kind}")
        records.append(record)
        for decision_tick in later_decisions:
            event_edges.append(
                {
                    "kind": "event_to_post_change_decision",
                    "source_event_id": event_id,
                    "target_tick": decision_tick,
                }
            )

    if blockers or not records or not event_edges:
        return None, list(dict.fromkeys(blockers or ["logistics_replay_event_proof_missing"]))

    difficulty = str(
        agentic_row.get("difficulty_level")
        or source_row.get("difficulty_level")
        or "basic"
    ).lower()
    adaptive_observed = bool(
        world_contract.get("adaptive_replanning_observed")
        or world_contract.get("valid_plan_delegation_observed")
    )
    if difficulty in {"high", "extreme"} and not adaptive_observed:
        return None, ["logistics_replay_adaptive_response_missing"]
    action_effect_digest = hashlib.sha256(
        json.dumps(action_records, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "replay_count": 2,
        "deterministic": True,
        "trace_semantic_digests": [wait_first, wait_second],
        "source_consumption_proven": True,
        "agent_action_backend_effect_observed": True,
        "world_evolution_records": records,
        "event_to_decision_action_edges": event_edges,
        "native_action_effects": [
            {
                key: record[key]
                for key in (
                    "event_id",
                    "applied_tick",
                    "call_id",
                    "tool_name",
                    "action_to_outcome_edge",
                )
                if key in record
            }
            for record in action_records
        ],
        "normalization": {
            "schema_version": "protocol21-logistics-replay-normalizer-v1",
            "source_scenario_path": str(scenario_path) if scenario_path else None,
            "source_scenario_sha256": scenario_sha256,
            "runtime_predesigned_event_ids": [str(value) for value in runtime_ids],
            "runtime_exogenous_state_change_ticks": sorted(exogenous_ticks),
            "runtime_model_decision_ticks": decision_ticks,
            "runtime_action_effect_digest": action_effect_digest,
            "material_event_count": len(records),
        },
    }, []


def _compact_replay_repair_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep the repair artifact auditable without copying full trajectories."""

    keys = (
        "scenario_id",
        "scenario_signature",
        "domain",
        "backend_kind",
        "difficulty_level",
        "status",
        "blockers",
        "checks",
        "repair_blockers",
        "replay_evidence",
    )
    return {
        key: copy.deepcopy(row[key])
        for key in keys
        if key in row
    }


def build_logistics_replay_report(
    agentic_report: dict[str, Any],
    *,
    behavioral_report: dict[str, Any] | None = None,
    complexity_report: dict[str, Any] | None = None,
    source_suite: dict[str, Any] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Normalize existing v15 Logistics replay evidence without changing Core.

    Only native VRP rows with a source-declared perturbation, observed runtime
    event tick, deterministic wait replay, post-event decision, and native
    action-effect evidence are marked ``passed`` in this *repair report*.  The
    remediation queue still emits ``repaired_candidate`` and downstream gates
    must run again.  Static JSPLIB rows and all non-Logistics domains remain
    unchanged/held.
    """

    if not isinstance(agentic_report, dict):
        raise TypeError("agentic report must be a mapping")
    root = Path(repo_root or Path.cwd())
    behavior_index = _report_index(behavioral_report)
    complexity_index = _report_index(complexity_report)
    source_index = _report_index(source_suite)
    rows_seen = _rows(agentic_report)
    results: list[dict[str, Any]] = []
    n_normalized = 0
    n_held = 0
    for original in _rows(agentic_report):
        row = copy.deepcopy(original)
        if str(row.get("status") or "") != "held":
            continue
        if row.get("backend_kind") not in _LOGISTICS_REPLAY_BACKENDS:
            row["repair_blockers"] = ["unsupported_logistics_replay_backend"]
            n_held += 1
            results.append(_compact_replay_repair_row(row))
            continue
        identity = _replay_identity(row)
        source_matches = source_index.get(identity, [])
        behavior_matches = behavior_index.get(identity, [])
        complexity_matches = complexity_index.get(identity, [])
        source_row = source_matches[0] if len(source_matches) == 1 else {}
        behavioral_row = behavior_matches[0] if len(behavior_matches) == 1 else {}
        complexity_row = next(
            (
                candidate
                for candidate in complexity_matches
                if str(candidate.get("agent_name") or "").lower()
                == "oracle_offline"
            ),
            complexity_matches[0] if len(complexity_matches) == 1 else {},
        )
        if (
            len(source_matches) != 1
            or len(behavior_matches) != 1
            or len(complexity_matches) < 1
        ):
            row["repair_blockers"] = ["logistics_replay_join_missing"]
            n_held += 1
            results.append(_compact_replay_repair_row(row))
            continue
        evidence, blockers = _build_logistics_replay_evidence(
            agentic_row=row,
            behavioral_row=behavioral_row,
            complexity_row=complexity_row,
            source_row=source_row,
            repo_root=root,
        )
        if evidence is None:
            row["repair_blockers"] = blockers
            n_held += 1
            results.append(_compact_replay_repair_row(row))
            continue
        checks = dict(row.get("checks") or {})
        checks.update({check: True for check in _WORLD_REPAIR_CHECKS})
        row["checks"] = checks
        row["difficulty_level"] = (
            row.get("difficulty_level") or source_row.get("difficulty_level")
        )
        row["status"] = "passed"
        row["blockers"] = []
        row["repair_blockers"] = []
        row["replay_evidence"] = evidence
        n_normalized += 1
        results.append(_compact_replay_repair_row(row))
    return {
        "schema_version": "protocol21-logistics-replay-repair-v1",
        "status": "complete",
        "source_report_status": agentic_report.get("status"),
        "n_expected": int(agentic_report.get("n_expected") or len(results)),
        "n_rows_seen": len(rows_seen),
        "n_repair_rows": len(results),
        "n_passed_excluded": sum(
            str(row.get("status") or "") == "passed" for row in rows_seen
        ),
        "n_retired_excluded": sum(
            str(row.get("status") or "") == "retired" for row in rows_seen
        ),
        "n_normalized": n_normalized,
        "n_held": n_held,
        "results": results,
    }


def apply_replay_repairs(
    queue: dict[str, Any], replay_report: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Apply only independently proven replay repairs to an existing queue.

    The existing queue is copied and retired rows are immutable.  Missing,
    ambiguous, or incomplete replay evidence leaves a row held; this helper
    never fabricates evidence or changes a row to ``passed``/Core eligible.
    """

    if not isinstance(queue, dict):
        raise TypeError("remediation queue must be a mapping")
    items = queue.get("items")
    if not isinstance(items, list):
        raise ValueError("remediation queue must contain an items list")
    updated = copy.deepcopy(queue)

    report_rows = _rows(replay_report) if isinstance(replay_report, dict) else []
    replay_index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in report_rows:
        replay_index.setdefault(_replay_identity(row), []).append(row)

    promoted = 0
    held = 0
    retire_locked = 0
    for item in updated["items"]:
        if not isinstance(item, dict):
            continue
        item.setdefault("replay_blockers", [])
        status = str(item.get("status") or "")
        if status == "retire":
            item["repair_evidence_status"] = "not_applicable"
            item["replay_blockers"] = ["retire_locked"]
            retire_locked += 1
            continue
        if status != "repair_pending":
            item.setdefault("repair_evidence_status", "not_run")
            item.setdefault("replay_blockers", ["replay_not_required"])
            continue

        if replay_report is None:
            item["repair_evidence_status"] = "not_run"
            item["replay_blockers"] = ["replay_evidence_missing"]
            held += 1
            continue

        matches = replay_index.get(_replay_identity(item), [])
        if not matches:
            item["repair_evidence_status"] = "missing"
            item["replay_blockers"] = ["replay_row_missing"]
            held += 1
            continue
        if len(matches) != 1:
            item["repair_evidence_status"] = "incomplete"
            item["replay_blockers"] = ["replay_row_ambiguous"]
            held += 1
            continue

        proven, blockers, summary = _strict_replay_proof(matches[0])
        if proven:
            item["status"] = "repaired_candidate"
            item["disposition"] = "repair"
            item["remaining_blockers"] = []
            item["repair_evidence_status"] = "proven"
            item["replay_blockers"] = []
            item["repair_resolution"] = "replay_proven_all_required_edges"
            item["replay_evidence"] = summary
            promoted += 1
        else:
            item["repair_evidence_status"] = (
                "held" if "replay_status_not_passed" in blockers else "incomplete"
            )
            item["replay_blockers"] = blockers
            held += 1

    counts = Counter(str(item.get("status") or "") for item in updated["items"] if isinstance(item, dict))
    updated["status_counts"] = {
        "repair_pending": int(counts.get("repair_pending", 0)),
        "repaired_candidate": int(counts.get("repaired_candidate", 0)),
        "retire": int(counts.get("retire", 0)),
    }
    updated["repair_pass"] = {
        "schema_version": "protocol21-replay-repair-v1",
        "status": "not_run" if replay_report is None else "complete",
        "fail_closed": True,
        "n_replay_rows": len(report_rows),
        "n_promoted": promoted,
        "n_held": held,
        "n_retire_locked": retire_locked,
    }
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--agentic-report", type=Path)
    source.add_argument("--queue", type=Path)
    parser.add_argument("--replay-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.agentic_report is not None:
        report = json.loads(args.agentic_report.read_text(encoding="utf-8"))
        queue = build_remediation_queue(report)
    else:
        queue = json.loads(args.queue.read_text(encoding="utf-8"))
    replay_report = (
        json.loads(args.replay_report.read_text(encoding="utf-8"))
        if args.replay_report is not None
        else None
    )
    queue = apply_replay_repairs(queue, replay_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": queue["status"],
                "n_remediation_items": queue["n_remediation_items"],
                "status_counts": queue["status_counts"],
                "first_missing_evidence_edge_counts": queue[
                    "first_missing_evidence_edge_counts"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
