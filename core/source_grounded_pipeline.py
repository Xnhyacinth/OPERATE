"""Fail-closed admission contract for source-grounded scheduling tasks.

Candidate mining may use random walks, boundary search, or optimization, but
those methods only locate windows. Core admission still requires locked source
consumption, native control capability, deterministic replay, a successful
reference policy, a causal evidence/action graph, and replay-backed difficulty.
"""

from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlparse

from core.difficulty_contract import DIFFICULTY_REQUIREMENTS
from core.protocol21_admission import (
    resolve_protocol21_admission_profile,
    source_admission_failures,
)

PIPELINE_VERSION = "source_grounded_decision_graph_v1"
_PROVED_MINIMALITY = frozenset(
    {
        "one_minimal",
        "exact_global",
        "proved_lower_bound",
    }
)
_ORDERED_CONTRACT_DEPTH_PROOF = "task_contract_ordered_milestone_lower_bound"
_SINGLE_STAGE_MINIMALITY_STATUS = "one_minimal_single_stage_action_dag"
_SOURCE_CLOCK_SEMANTICS = frozenset(
    {
        "simulator_owned",
        "simulator_owned_substeps",
    }
)


def _nonempty_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_int(value: Any) -> int | None:
    number = _finite_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _resolvable_http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _decision_graph_is_valid(graph: dict[str, Any]) -> bool:
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not _nonempty_list(nodes) or not isinstance(edges, list):
        return False
    node_ids = [
        str(node.get("id") or "")
        for node in nodes
        if isinstance(node, dict)
    ]
    if len(node_ids) != len(nodes) or not all(node_ids):
        return False
    if len(set(node_ids)) != len(node_ids):
        return False
    node_kinds = {
        str(node.get("kind") or "")
        for node in nodes
        if isinstance(node, dict)
    }
    if not {"observation", "action", "outcome"}.issubset(node_kinds):
        return False

    adjacency = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in edges:
        if not isinstance(edge, dict):
            return False
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source not in adjacency or target not in adjacency or source == target:
            return False
        adjacency[source].append(target)
        indegree[target] += 1
    frontier = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while frontier:
        node_id = frontier.pop()
        visited += 1
        for target in adjacency[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
    return bool(
        visited == len(node_ids)
        and graph.get("successful_reference") is True
        and _nonempty_list(graph.get("required_tools"))
    )


def evaluate_source_grounded_candidate(
    candidate: dict[str, Any],
) -> dict[str, Any]:
    """Return deterministic, machine-readable Core admission gates."""
    boundary = _mapping(candidate.get("domain_boundary"))
    source = _mapping(candidate.get("source"))
    capability = _mapping(candidate.get("capability"))
    replay = _mapping(candidate.get("replay"))
    graph = _mapping(candidate.get("decision_graph"))
    proof = _mapping(candidate.get("difficulty_proof"))
    independence = _mapping(candidate.get("independence"))

    difficulty = str(candidate.get("difficulty_level") or "")
    requirements = DIFFICULTY_REQUIREMENTS.get(difficulty)
    depth_value = _nonnegative_int(graph.get("exact_dependency_depth"))
    lower_bound_value = _nonnegative_int(
        graph.get("required_depth_lower_bound")
        if graph.get("required_depth_lower_bound") is not None
        else proof.get("required_depth_lower_bound")
    )
    reversal_value = _nonnegative_int(graph.get("plan_reversal_count"))
    depth = depth_value if depth_value is not None else 0
    lower_bound = lower_bound_value if lower_bound_value is not None else 0
    reversals = reversal_value if reversal_value is not None else 0
    required_tools = set(
        graph.get("required_tools")
        if isinstance(graph.get("required_tools"), list)
        else []
    )

    source_lock = bool(
        _nonempty_list(source.get("files"))
        and _resolvable_http_url(source.get("url"))
        and source.get("version_lock")
        and source.get("license")
        and source.get("window_sha256")
    )
    capability_contract = bool(
        _nonempty_list(capability.get("native_state_fields"))
        and _nonempty_list(capability.get("observation_tools"))
        and _nonempty_list(capability.get("control_tools"))
        and capability.get("clock_semantics") in _SOURCE_CLOCK_SEMANTICS
        and capability.get("deterministic_seed") is True
        and capability.get("counterfactual_reset") is True
        and capability.get("adaptive_recovery_signal")
    )
    wait_task_loss = _finite_float(replay.get("wait_task_loss"))
    reference_task_loss = _finite_float(replay.get("reference_task_loss"))
    material_headroom = _mapping(replay.get("material_headroom"))
    task_headroom = bool(
        material_headroom.get("status") == "passed"
        or (
            replay.get("reference_task_completed") is True
            and wait_task_loss is not None
            and reference_task_loss is not None
            and wait_task_loss > reference_task_loss
        )
    )
    single_stage_minimal_proof = bool(
        difficulty in {"basic", "medium"}
        and graph.get("dependency_depth_status")
        == _SINGLE_STAGE_MINIMALITY_STATUS
        and proof.get("minimality_status") == "one_minimal"
    )
    exact_difficulty_proof = bool(
        requirements is not None
        and depth_value is not None
        and reversal_value is not None
        and proof.get("contract_passed") is True
        and proof.get("minimality_status") in _PROVED_MINIMALITY
        and (
            graph.get("dependency_depth_status")
            == "declared_evidence_action_dag"
            or single_stage_minimal_proof
        )
        and depth >= requirements.min_dependency_depth
        and len(required_tools) >= requirements.min_physical_tools
        and reversals >= requirements.min_strategy_switches
    )
    proof_kinds = set(
        str(value)
        for value in (
            graph.get("depth_proof_kinds")
            if isinstance(graph.get("depth_proof_kinds"), list)
            else proof.get("depth_proof_kinds")
            if isinstance(proof.get("depth_proof_kinds"), list)
            else []
        )
    )
    ordered_contract_difficulty_proof = bool(
        requirements is not None
        and lower_bound_value is not None
        and proof.get("contract_passed") is True
        and _ORDERED_CONTRACT_DEPTH_PROOF in proof_kinds
        and graph.get("dependency_depth_status")
        == _ORDERED_CONTRACT_DEPTH_PROOF
        and lower_bound >= requirements.min_dependency_depth
        and len(required_tools) >= requirements.min_physical_tools
        and reversals >= requirements.min_strategy_switches
    )
    difficulty_proof = exact_difficulty_proof or ordered_contract_difficulty_proof

    gates = {
        "domain_boundary": bool(
            boundary.get("allowed") is True
            and boundary.get("classification") == candidate.get("domain")
        ),
        "source_lock": source_lock,
        "source_consumption": bool(
            source.get("consumed_by_backend") is True
            and _nonempty_list(source.get("consumed_fields"))
        ),
        "capability_contract": capability_contract,
        "deterministic_replay": bool(
            replay.get("wait_fingerprint_first")
            and replay.get("wait_fingerprint_first")
            == replay.get("wait_fingerprint_second")
        ),
        "task_headroom": task_headroom,
        "decision_graph": _decision_graph_is_valid(graph),
        "difficulty_proof": difficulty_proof,
        "counterfactual": replay.get("counterfactual_supported") is True,
        "independence": bool(
            independence.get("structural_fingerprint")
            and independence.get("semantic_fingerprint")
            and independence.get("is_duplicate") is False
        ),
    }
    failures = [name for name, passed in gates.items() if not passed]
    admission_profile = resolve_protocol21_admission_profile(candidate)
    admission_failures = source_admission_failures(
        failures,
        profile=admission_profile,
    )
    diagnostic_failures = [
        failure for failure in failures if failure not in admission_failures
    ]
    gate_evidence = {
        "source_consumption": {
            "status": (
                "passed"
                if gates["source_consumption"]
                else (
                    "failed"
                    if source.get("consumed_by_backend") is False
                    else "missing"
                )
            )
        },
        "deterministic_replay": {
            "status": (
                "passed"
                if gates["deterministic_replay"]
                else (
                    "failed"
                    if replay.get("wait_fingerprint_first")
                    and replay.get("wait_fingerprint_second")
                    else "missing"
                )
            )
        },
        "task_headroom": {
            "status": (
                "passed"
                if gates["task_headroom"]
                else (
                    "failed"
                    if replay.get("reference_task_completed") is True
                    and wait_task_loss is not None
                    and reference_task_loss is not None
                    else "missing"
                )
            )
            ,
            "material_headroom": material_headroom,
        },
    }
    return {
        "pipeline_version": PIPELINE_VERSION,
        "scenario_id": candidate.get("scenario_id"),
        "status": (
            "admitted_for_core_review" if not admission_failures else "held"
        ),
        "admission_profile": admission_profile,
        "gates": gates,
        "gate_evidence": gate_evidence,
        "failures": failures,
        "admission_failures": admission_failures,
        "diagnostic_failures": diagnostic_failures,
        "mining_method": (candidate.get("mining") or {}).get("method"),
        "difficulty_evidence": {
            "level": difficulty,
            "exact_dependency_depth": depth,
            "required_depth_lower_bound": (
                lower_bound_value if lower_bound_value is not None else None
            ),
            "required_tools": sorted(required_tools),
            "plan_reversal_count": reversals,
            "minimality_status": proof.get("minimality_status"),
            "depth_proof_kinds": sorted(proof_kinds),
        },
    }
