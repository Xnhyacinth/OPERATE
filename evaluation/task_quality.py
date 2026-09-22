"""Mission-native task quality for OPERATE 0.23, revision 2.

The caller authenticates episode artifacts and supplies predeclared acceptance
and reference contracts. This evaluator rechecks the terminal trace hash and
uses recorded native objectives, never a wait-relative success flag as an
absolute completion target. Feasibility tasks require complete native service;
optimization tasks retain their original continuous objective.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from evaluation.native_objectives import NATIVE_OBJECTIVES, extract_native_objective
from evaluation.native_quality import score_native_quality

VERSION = "task_quality.v2"
PROTOCOL_REVISION = 2
FJSP = "dynasched_flexible_job_shop"


def backend_readiness() -> dict[str, dict[str, str]]:
    """Describe implemented contracts, not whether every episode has evidence."""
    return {
        name: {
            "status": "absolute_count_contract_available"
            if name == FJSP
            else "native_optimization_contract_available",
            "reason": "terminal_native_operations_and_source_arrival_counts"
            if name == FJSP
            else "native_objective_with_existing_hard_gate_not_binary_completion",
        }
        for name in NATIVE_OBJECTIVES
    }


def _finite(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _value(observation: dict, path: Any) -> Any:
    if (
        not isinstance(path, list)
        or not path
        or not all(isinstance(key, str) and key for key in path)
    ):
        return None
    current: Any = observation
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _declared_completion(observation: dict, contract: dict) -> tuple[float, bool]:
    """Interpret only numeric/bool predicates over the authenticated observation."""
    fraction = contract.get("completion_fraction") or {}
    numerator = _value(observation, fraction.get("numerator_path"))
    denominator = fraction.get("denominator")
    if not (
        _finite(numerator)
        and _finite(denominator)
        and denominator > 0
        and 0 <= numerator <= denominator
    ):
        raise ValueError("absolute_completion_fraction_invalid")
    predicates = contract.get("mandatory")
    if not isinstance(predicates, list) or not predicates:
        raise ValueError("absolute_mandatory_predicates_missing")
    outcomes = []
    for predicate in predicates:
        if not isinstance(predicate, dict):
            raise ValueError("absolute_mandatory_predicate_invalid")
        actual = _value(observation, predicate.get("path"))
        target = predicate.get("target")
        op = predicate.get("op")
        if op == "eq" and type(actual) is bool and type(target) is bool:
            passed = actual == target
        elif _finite(actual) and _finite(target) and op in {"eq", "le", "ge"}:
            passed = {
                "eq": actual == target,
                "le": actual <= target,
                "ge": actual >= target,
            }[op]
        else:
            raise ValueError("absolute_mandatory_predicate_invalid")
        outcomes.append(passed)
    return numerator / denominator, all(outcomes) and numerator == denominator


def evaluate_task_quality(
    episode: dict[str, Any],
    *,
    artifact_binding: dict[str, Any],
    reference_contract: dict[str, Any] | None = None,
    acceptance_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score R or retain an explicitly incomplete absolute-completion reading.

    An optional acceptance contract must be predeclared in the caller's bound
    suite/manifest, not generated from model results. Paths are lists of dict
    keys relative to the terminal native observation. The frozen denominator,
    mandatory predicates and quality bounds are part of that contract.
    """
    result: dict[str, Any] = {
        "schema_version": VERSION,
        "evaluation_version": "0.23.0",
        "protocol_revision": PROTOCOL_REVISION,
        "mission_type": "feasibility"
        if episode.get("backend_kind") == FJSP
        else "optimization",
        "applicable": False,
        "score": None,
        "completion_fraction": None,
        "mandatory_complete": None,
        "quality_fraction": None,
        "hard_failure": None,
        "evidence_ids": [],
        "reason": None,
        "formal_run_certified": False,
        "utility_contract": {
            "normalization": "affine_native_quality_0_100",
            "feasibility_gate": "incomplete_native_feasibility_receives_zero",
            "optimization_semantics": "native_objective_not_absolute_completion",
            "legacy_completion_flag_used": False,
        },
    }

    def unavailable(reason: str) -> dict[str, Any]:
        return {**result, "reason": reason}

    if (
        artifact_binding.get("verified") is not True
        or artifact_binding.get("native_cost_bound") is not True
    ):
        return unavailable("unbound_episode_artifacts")
    backend = episode.get("backend_kind")
    if backend not in NATIVE_OBJECTIVES:
        return unavailable("unsupported_backend_or_domain")
    if acceptance_contract is not None:
        if (
            acceptance_contract.get("schema_version") != "task_acceptance.v1"
            or not isinstance(acceptance_contract.get("source"), str)
            or not acceptance_contract["source"]
            or any(
                acceptance_contract.get(key) != episode.get(key)
                for key in ("scenario_signature", "seed", "backend_kind")
            )
        ):
            return unavailable("absolute_acceptance_contract_identity_mismatch")
        try:
            encoded = json.dumps(
                acceptance_contract, sort_keys=True, allow_nan=False
            ).encode()
        except (TypeError, ValueError):
            return unavailable("absolute_acceptance_contract_invalid")
        result["acceptance_contract_sha256"] = hashlib.sha256(encoded).hexdigest()
    measurement = extract_native_objective(episode)
    if measurement.get("applicable") is not True:
        return unavailable(measurement.get("reason", "native_measurement_unavailable"))
    result["hard_failure"] = measurement["hard_failure"]
    result["evidence_ids"] = list(measurement["evidence_ids"])
    if measurement["hard_failure"]:
        return {
            **result,
            "applicable": True,
            "score": 0.0,
            "reason": "task_hard_failure",
        }

    def reference_quality() -> dict[str, Any]:
        if not reference_contract or reference_contract.get("status") != "ready":
            return unavailable("quality_reference_unavailable")
        if any(
            reference_contract.get(key) != episode.get(key)
            for key in ("scenario_signature", "seed", "backend_kind")
        ):
            return unavailable("quality_reference_identity_mismatch")
        relative = score_native_quality(measurement, reference_contract)
        if relative.get("score") is None:
            return unavailable(relative.get("reason", "quality_reference_unavailable"))
        score = (relative["score"] + 100.0) / 2.0
        return {
            **result,
            "applicable": True,
            "score": score,
            "quality_fraction": score / 100.0,
            "native_cost": measurement["actual_cost"],
            "quality_basis": "fixed_reference_native_objective_normalization",
            "reference_contract_sha256": reference_contract.get("contract_sha256"),
            "reason": "mission_native_quality_verified",
        }

    if acceptance_contract is None and backend != FJSP:
        return reference_quality()

    descriptor = (artifact_binding.get("artifacts") or {}).get(
        "trajectory_artifact"
    ) or {}
    try:
        raw = Path(descriptor["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != descriptor.get("sha256"):
            return unavailable("terminal_trace_hash_mismatch")
        trace = [json.loads(line) for line in raw.splitlines() if line.strip()]
        terminal = trace[-1]
        observation = terminal["observation"]
    except (OSError, KeyError, ValueError, IndexError, TypeError):
        return unavailable("terminal_native_observation_missing")
    if not isinstance(observation, dict):
        return unavailable("terminal_native_observation_missing")
    ids = terminal.get("evidence_ids")
    valid_ids = (episode.get("trajectory_summary") or {}).get(
        "operational_agency_valid_evidence_ids"
    ) or []
    if (
        not isinstance(ids, list)
        or not ids
        or not all(isinstance(item, str) and item in valid_ids for item in ids)
    ):
        return unavailable("terminal_native_evidence_missing")
    result["evidence_ids"] = list(dict.fromkeys(result["evidence_ids"] + ids))
    result["terminal_trace_sha256"] = descriptor["sha256"]
    if acceptance_contract is not None:
        result["mission_type"] = "declared_feasibility"
    if acceptance_contract is None:
        names = (
            "operations_total",
            "operations_cancelled",
            "operations_completed",
            "operations_scheduled",
            "jobs_total",
            "jobs_arrived",
        )
        if any(
            type(observation.get(key)) is not int or observation[key] < 0
            for key in names
        ):
            return unavailable("absolute_operation_counts_invalid")
        if (
            observation["jobs_total"] <= 0
            or observation["jobs_arrived"] > observation["jobs_total"]
        ):
            return unavailable("absolute_operation_counts_invalid")
        if observation["jobs_arrived"] < observation["jobs_total"]:
            # The known source job denominator proves incomplete service, but
            # does not reveal the unarrived operation count. Do not invent T.
            return {
                **result,
                "applicable": True,
                "score": 0.0,
                "mandatory_complete": False,
                "reason": "incomplete_source_obligations",
            }
        required = observation["operations_total"] - observation["operations_cancelled"]
        completed, scheduled = (
            observation["operations_completed"],
            observation["operations_scheduled"],
        )
        if required <= 0 or not 0 <= completed <= scheduled <= required:
            return unavailable("absolute_operation_counts_invalid")
        evidence = (episode.get("task_completion") or {}).get("evidence") or {}
        if (
            any(evidence.get(key) != observation[key] for key in names[:4])
            or evidence.get("operations_required") != required
        ):
            return unavailable("completion_trace_mismatch")
        fraction, complete = completed / required, completed == scheduled == required
        result["acceptance_contract"] = "native_all_required_operations_completed.v1"
    else:
        try:
            fraction, complete = _declared_completion(observation, acceptance_contract)
        except ValueError as exc:
            return unavailable(str(exc))
    result.update(completion_fraction=fraction, mandatory_complete=complete)
    if not complete:
        return {
            **result,
            "applicable": True,
            "score": 0.0,
            "reason": "incomplete_absolute_obligations",
        }
    quality = None
    if acceptance_contract is not None:
        spec = acceptance_contract.get("quality") or {}
        cost = _value(observation, spec.get("cost_path"))
        good, bad = spec.get("good_cost"), spec.get("bad_cost")
        if (
            spec.get("excludes_unfulfilled_penalties") is not True
            or not all(_finite(value) for value in (cost, good, bad))
            or good >= bad
            or not _finite(bad - good)
            or not _finite(bad - cost)
        ):
            return unavailable("absolute_quality_bounds_missing_or_invalid")
        quality = max(0.0, min(1.0, (bad - cost) / (bad - good)))
        result["quality_basis"] = "predeclared_native_service_cost_bounds"
    else:
        makespan = observation.get("makespan")
        components = episode["ground_truth_summary"]["cost_components"]
        if (
            not _finite(makespan)
            or makespan < 0
            or set(components) != {"production_cost"}
            or not math.isclose(
                makespan, components["production_cost"], rel_tol=1e-10, abs_tol=1e-7
            )
        ):
            return unavailable("complete_schedule_cost_not_pure_makespan")
        return reference_quality()
    return {
        **result,
        "applicable": True,
        "score": 100.0 * quality,
        "quality_fraction": quality,
        "reason": "absolute_obligations_and_quality_verified",
    }
