"""Matched analysis contracts for OPERATE supplementary experiments."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any


E1_ARMS = frozenset({"logical_persistent", "logical_stateless"})
E3_DELAYS = frozenset({0.0, 1.0, 5.0})
E4_ARMS = frozenset({"reveal", "withhold"})


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _path(row: Mapping[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _score_dimension(row: Mapping[str, Any], name: str) -> float | None:
    for dimension in _path(row, "score", "dimensions") or []:
        if dimension.get("name") == name and dimension.get("applicable") is True:
            return _number(dimension.get("calibrated_score"))
    return None


def _numeric_delta(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, float | None]:
    keys = sorted(set(left) | set(right))
    return {
        key: (
            round(float(left[key]) - float(right[key]), 6)
            if _number(left.get(key)) is not None and _number(right.get(key)) is not None
            else None
        )
        for key in keys
    }


def native_tick_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, float | None]:
    """Aggregate the shared native tick fields without inventing a new score."""

    rows = list(records)
    additive = {
        "production_cost": "native_production_cost_sum",
        "startup_cost": "native_startup_cost_sum",
        "shed_penalty": "native_shed_penalty_sum",
        "balance_error_mw": "native_balance_error_sum",
        "safety_violation_severity": "native_safety_severity_sum",
    }
    result: dict[str, float | None] = {"native_ticks": float(len(rows))}
    for source, target in additive.items():
        values = [_number(row.get(source)) for row in rows]
        result[target] = (
            round(sum(value for value in values if value is not None), 6)
            if values and all(value is not None for value in values) else None
        )
    result["native_catastrophic_ticks"] = (
        float(sum(row["catastrophic_failure"] for row in rows))
        if rows and all(isinstance(row.get("catastrophic_failure"), bool) for row in rows)
        else None
    )
    return result


def supplementary_artifact_bytes(row: Mapping[str, Any], descriptor: Mapping[str, Any]) -> bytes:
    """Resolve an archive-local sidecar and verify its frozen byte hash."""
    raw, expected = descriptor.get("path"), descriptor.get("sha256")
    if not isinstance(raw, str) or not raw or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("artifact_descriptor_missing")
    path = Path(raw)
    root_value = row.get("_supplementary_batch_root")
    if root_value:
        root = Path(str(root_value)).resolve()
        # Old machine absolute paths may be relocated only by their declared
        # trajectory suffix inside this journal's own run, never by basename.
        if "trajectories" in path.parts:
            path = root.joinpath(*path.parts[path.parts.index("trajectories"):])
        elif not path.is_absolute():
            path = root / path
        if not path.resolve().is_relative_to(root):
            raise ValueError("artifact_outside_batch")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("artifact_hash_mismatch")
    return payload


def _e1_identity(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
    config = _path(row, "agent_config", "config")
    fields = (row.get("model"), row.get("implementation_tree_sha256"),
              row.get("suite_manifest_sha256"), _path(row, "score", "scenario_signature"),
              _path(row, "score", "scoring_version"))
    if any(not isinstance(value, str) or not value for value in fields):
        return None
    if isinstance(row.get("seed"), bool) or not isinstance(row.get("seed"), int):
        return None
    if not isinstance(config, Mapping) or any(
        key not in config or config[key] is None
        for key in ("model", "provider", "api_mode", "temperature", "max_tokens",
                    "prompt_mode", "persistent_context_max_chars")
    ):
        return None
    if (config.get("context_ablation_mode") != "matched_transcript_v1"
            or config["model"] != row["model"]
            or config.get("interaction_mode") != row.get("interaction_mode")):
        return None
    return (*fields, row["seed"], {k: v for k, v in config.items() if k != "interaction_mode"})


def _provider_identity_exact(stats: Mapping[str, Any]) -> bool:
    requests = stats.get("provider_model_identity_request_count")
    return bool(type(requests) is int and requests > 0 and all(
        type(stats.get(key)) is int and stats[key] == requests
        for key in ("provider_model_identity_closed_count", "provider_model_identity_exact_count")
    ) and not any(stats.get(key) for key in (
        "provider_model_identity_missing_count", "provider_model_identity_mismatch_count",
        "provider_model_identity_failed_request_count"
    )))


def _e1_arm_problem(row: Mapping[str, Any]) -> str | None:
    from scripts.batch_llm_eval import _llm_call_failure_eligibility_reasons

    llm = _path(row, "trajectory_summary", "llm") or {}
    if row.get("status") != "ok":
        return "non_terminal_e1_arm"
    if row.get("_supplementary_artifact_problem"):
        return "e1_artifact_missing_or_invalid"
    if _llm_call_failure_eligibility_reasons(llm) or llm.get("ticks_wait_fallback"):
        return "e1_provider_or_prompt_failure"
    if not _provider_identity_exact(llm):
        return "provider_identity_not_exact"
    keys = ["trajectory_artifact", "evidence_ledger_artifact", "provider_audit_artifact",
            "completed_runtime_artifact"]
    if row.get("interaction_mode") == "logical_persistent":
        keys.append("semantic_ledger_artifact")
    try:
        for key in keys:
            descriptor = _path(row, "trajectory_summary", key)
            if not isinstance(descriptor, Mapping):
                return "e1_artifact_missing_or_invalid"
            supplementary_artifact_bytes(row, descriptor)
    except (OSError, ValueError):
        return "e1_artifact_missing_or_invalid"
    return None


def audit_e1_matrix(
    cells: Iterable[Mapping[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_model: str,
    expected_scoring_version: str,
) -> dict[str, Any]:
    """Accept E1 rows by frozen scenario signature rather than display ID."""

    candidates: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "in_flight":
            continue
        candidates[(str(_path(row, "score", "scenario_signature")),
                    str(row.get("interaction_mode")))].append(row)
    expected = list(cells)
    findings = []
    for cell in expected:
        key = (str(cell["scenario_signature"]), str(cell["condition"]))
        matches = candidates.get(key, [])
        row = matches[0] if len(matches) == 1 else None
        problem = "duplicate_e1_arm" if len(matches) > 1 else None
        llm = _path(row or {}, "trajectory_summary", "llm") or {}
        requests = int(llm.get("provider_model_identity_request_count") or 0)
        if row is None:
            problem = problem or "missing"
        elif row.get("status") != "ok":
            problem = f"status:{row.get('status')}"
        elif row.get("model") != expected_model:
            problem = "model_binding_mismatch"
        elif _path(row, "score", "scoring_version") != expected_scoring_version:
            problem = "scoring_version_mismatch"
        elif not (
            requests > 0
            and int(llm.get("provider_model_identity_closed_count") or 0) == requests
            and int(llm.get("provider_model_identity_exact_count") or 0) == requests
            and int(llm.get("provider_model_identity_missing_count") or 0) == 0
            and int(llm.get("provider_model_identity_mismatch_count") or 0) == 0
            and int(llm.get("provider_model_identity_failed_request_count") or 0) == 0
        ):
            problem = "provider_identity_not_exact"
        if not problem and row is not None:
            problem = "e1_identity_missing" if _e1_identity(row) is None else _e1_arm_problem(row)
        if problem:
            findings.append({
                "pair_id": cell["pair_id"],
                "condition": cell["condition"],
                "problem": problem,
            })
    for pair_id in {str(cell["pair_id"]) for cell in expected}:
        if any(str(finding["pair_id"]) == pair_id for finding in findings):
            continue
        pair = [candidates[(str(cell["scenario_signature"]), str(cell["condition"]))][0]
                for cell in expected if str(cell["pair_id"]) == pair_id]
        result = analyze_e1_pair(pair)
        if not result["valid"]:
            findings.append({"pair_id": pair_id, "condition": "pair", "problem": result["problem"]})
    failed_pairs = {row["pair_id"] for row in findings}
    expected_pairs = {str(cell["pair_id"]) for cell in expected}
    return {
        "schema_version": "operate-supplementary-e1-acceptance/2.0",
        "join_key": "scenario_signature+interaction_mode",
        "expected_cells": len(expected),
        "accepted_cells": sum(str(cell["pair_id"]) not in failed_pairs for cell in expected),
        "expected_pairs": len(expected_pairs),
        "accepted_pairs": len(expected_pairs - failed_pairs),
        "findings": findings,
        "accepted": bool(expected) and not findings,
    }


def analyze_e1_pair(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare one persistent/stateless pair without changing the base score."""

    samples = list(rows)
    by_arm = {str(row.get("interaction_mode")): row for row in samples}
    if len(samples) != 2 or set(by_arm) != E1_ARMS:
        return {"valid": False, "problem": "incomplete_or_duplicate_e1_pair"}
    persistent = by_arm["logical_persistent"]
    stateless = by_arm["logical_stateless"]
    identities = [_e1_identity(row) for row in samples]
    if any(identity is None for identity in identities) or identities[0] != identities[1]:
        return {"valid": False, "problem": "e1_treatment_identity_missing_or_mismatch"}
    for row in samples:
        if problem := _e1_arm_problem(row):
            return {"valid": False, "problem": problem}
    identity = {
        (row.get("scenario_id"), _path(row, "score", "scenario_signature"))
        for row in by_arm.values()
    }
    versions = {_path(row, "score", "scoring_version") for row in by_arm.values()}
    if any(row.get("status") != "ok" for row in by_arm.values()):
        return {"valid": False, "problem": "non_terminal_e1_arm"}
    if len(identity) != 1 or len(versions) != 1 or None in versions:
        return {"valid": False, "problem": "e1_identity_or_scorer_mismatch"}

    def metrics(row: Mapping[str, Any]) -> dict[str, float | None]:
        from scripts.batch_llm_eval import _ranking_primary_for_analysis

        summary = row.get("trajectory_summary") or {}
        llm = summary.get("llm") or {}
        return {
            "primary_score": _ranking_primary_for_analysis(dict(row)),
            "fixed_score": _number(_path(row, "score", "total_score")),
            "applicable_score": _number(
                _path(row, "score", "score_views", "adaptive_applicable", "total_score")
            ),
            "economic_cost_score": _score_dimension(row, "economic_cost"),
            "safety_score": _score_dimension(row, "safety_violation"),
            "counterfactual_prevention_score": _score_dimension(
                row, "counterfactual_prevention"
            ),
            "tool_use_efficiency_score": _score_dimension(row, "tool_use_efficiency"),
            "model_calls": _number(llm.get("llm_calls_ok")),
            "tool_calls": _number(summary.get("n_tool_calls")),
            "wait_actions": _number(summary.get("n_wait_actions")),
            "provider_retries": _number(llm.get("retry_attempts_total")),
            "protocol_repairs": _number(llm.get("protocol_repair_attempts")),
            "plan_commits": _number(llm.get("plan_commits_confirmed")),
            "plan_revisions": _number(llm.get("plan_revisions_confirmed")),
            **{
                key: _number(value)
                for key, value in (row.get("supplementary_native_metrics") or {}).items()
            },
        }

    persistent_metrics = metrics(persistent)
    stateless_metrics = metrics(stateless)
    return {
        "valid": True,
        "scenario_id": persistent.get("scenario_id"),
        "scoring_version": next(iter(versions)),
        "score_contract": {
            "primary_score": "wait_relative_primary_missing_is_null",
            "fixed_score": "fixed_all_dimensions_composite_diagnostic",
        },
        "arms": {
            "logical_persistent": persistent_metrics,
            "logical_stateless": stateless_metrics,
        },
        "persistent_minus_stateless": _numeric_delta(
            persistent_metrics, stateless_metrics
        ),
    }


def analyze_e3_group(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize a matched 0/1/5-second realtime delay group."""

    by_delay: dict[float, Mapping[str, Any]] = {}
    for row in rows:
        delay = _number(_path(row, "treatment_identity", "clock", "response_delivery_delay_s"))
        if delay is None or delay in by_delay:
            return {"valid": False, "problem": "invalid_or_duplicate_e3_delay"}
        by_delay[delay] = row
    if set(by_delay) != E3_DELAYS:
        return {"valid": False, "problem": "incomplete_e3_delay_group"}
    identities = []
    for row in by_delay.values():
        treatment = deepcopy(row.get("treatment_identity"))
        if (not isinstance(row.get("scenario_signature"), str) or not row["scenario_signature"]
                or isinstance(row.get("seed"), bool) or not isinstance(row.get("seed"), int)
                or not _path(treatment, "implementation_contract", "implementation_tree_sha256")
                or not _path(treatment, "provider_public_config", "model")
                or not _path(treatment, "provider_public_config", "provider")
                or not _path(treatment, "clock", "tick_interval_s")
                or not treatment.get("harness")):
            return {"valid": False, "problem": "e3_identity_missing"}
        treatment["clock"].pop("response_delivery_delay_s")
        identities.append((row.get("scenario_id"), row["scenario_signature"], row["seed"], treatment))
    if any(identity != identities[0] for identity in identities[1:]):
        return {"valid": False, "problem": "e3_identity_mismatch"}
    from runner.realtime_episode import response_delivery_delay_violations

    if any(response_delivery_delay_violations(dict(row)) for row in by_delay.values()):
        return {"valid": False, "problem": "e3_delivery_delay_not_evidenced"}
    if any(not row.get("provider_audit")
           or _path(row, "provider_audit_contract", "complete") is not True
           for row in by_delay.values()):
        return {"valid": False, "problem": "e3_provider_audit_missing_or_incomplete"}
    if any(
        row.get("episode_status") != "complete"
        or row.get("evaluation_ready") is not True
        or _path(row, "artifact_validation", "valid") is not True
        for row in by_delay.values()
    ):
        return {"valid": False, "problem": "e3_artifact_not_evaluation_ready"}

    def metrics(row: Mapping[str, Any]) -> dict[str, float | None]:
        diagnostics = row.get("diagnostics") or {}
        return {
            "response_missed": _number(_path(diagnostics, "trigger_response", "response_missed")),
            "actions_effected": _number(_path(diagnostics, "action_lifecycle", "effected")),
            "actions_expired": _number(_path(diagnostics, "action_lifecycle", "expired")),
            "late_responses_discarded": _number(
                _path(diagnostics, "action_lifecycle", "late_response_discarded")
            ),
            "turns_superseded": _number(_path(diagnostics, "action_lifecycle", "turn_superseded")),
            "model_turns": _number(_path(diagnostics, "autonomy", "model_turns")),
            "controlled_holds": _number(_path(diagnostics, "safety", "controlled_holds")),
            "takeovers": _number(_path(diagnostics, "safety", "takeovers")),
            "native_valid_responses": _number(
                _path(diagnostics, "provider_protocol", "native_valid_without_repair")
            ),
            "repair_attempts": _number(_path(diagnostics, "provider_protocol", "repair_attempts")),
        }

    delay_metrics = {str(int(delay)): metrics(by_delay[delay]) for delay in sorted(by_delay)}
    return {
        "valid": True,
        "scenario_id": next(iter(by_delay.values())).get("scenario_id"),
        "metric_contract": "independent_realtime_diagnostics_no_primary_score",
        "delays_s": delay_metrics,
        "delay_1_minus_0": _numeric_delta(delay_metrics["1"], delay_metrics["0"]),
        "delay_5_minus_0": _numeric_delta(delay_metrics["5"], delay_metrics["0"]),
    }


def analyze_e4_pair(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare one reveal/withhold information-ablation pair."""

    samples = list(rows)
    by_arm = {str(_path(row, "cell", "condition")): row for row in samples}
    if len(samples) != 2 or set(by_arm) != E4_ARMS:
        return {"valid": False, "problem": "incomplete_or_duplicate_e4_pair"}
    from runner.supplementary_e4 import digest, native_validation

    reveal, withhold = by_arm["reveal"], by_arm["withhold"]
    if any(row.get("schema_version") != "native_prefix_information_v1" for row in samples):
        return {"valid": False, "problem": "e4_unsupported_producer_contract"}
    for row in samples:
        runtime = row.get("runtime_validation")
        if not isinstance(runtime, Mapping):
            return {"valid": False, "problem": "e4_runtime_validation_missing_or_invalid"}
        try:
            validation = native_validation(runtime)
        except (AttributeError, TypeError):
            return {"valid": False, "problem": "e4_runtime_validation_missing_or_invalid"}
        if validation["valid"] is not True or row.get("artifact_validation") != validation:
            return {"valid": False, "problem": "e4_runtime_validation_missing_or_invalid"}
        cell = row["cell"]
        state = row.get("state_artifact")
        try:
            state_hash = digest(state) if isinstance(state, Mapping) and state else None
        except (TypeError, ValueError):
            state_hash = None
        if (state_hash is None or state_hash != cell.get("state_artifact_sha256")
                or state_hash != _path(row, "intervention", "native_prefix_sha256")):
            return {"valid": False, "problem": "e4_state_artifact_hash_mismatch"}
        if (any(not isinstance(cell.get(key), str) or not cell[key]
                for key in ("model", "scenario_signature", "domain", "backend_kind"))
                or any(isinstance(cell.get(key), bool) or not isinstance(cell.get(key), int)
                       or cell[key] < 0 for key in ("seed", "response_deadline_tick"))):
            return {"valid": False, "problem": "e4_identity_missing"}
        from scripts.batch_llm_eval import _llm_call_failure_eligibility_reasons

        provider = row.get("provider_stats") or {}
        if (not provider.get("llm_calls_ok")
                or not _provider_identity_exact(provider)
                or _llm_call_failure_eligibility_reasons(provider)
                or provider.get("ticks_wait_fallback")
                or any(str(key).startswith("provider_") and count
                       for key, count in (provider.get("retry_by_reason") or {}).items())):
            return {"valid": False, "problem": "e4_provider_failure_or_missing_measurement"}
    identity_fields = (
        "model", "seed", "scenario_signature", "domain", "backend_kind", "response_deadline_tick",
        "pair_id",
        "state_artifact_sha256",
        "runtime_implementation_tree_sha256",
        "provider_config_sha256",
    )
    if any(_path(row, "cell", field) in (None, "", {}, [])
           for row in samples for field in (*identity_fields, "harness_sha256")):
        return {"valid": False, "problem": "e4_identity_missing"}
    if any(
        _path(reveal, "cell", field) != _path(withhold, "cell", field)
        for field in identity_fields
    ) or _path(reveal, "cell", "harness_sha256") != _path(
        withhold, "cell", "harness_sha256"
    ):
        return {"valid": False, "problem": "e4_identity_mismatch"}
    if any(row.get("status") != "native_trial_completed" for row in by_arm.values()):
        return {"valid": False, "problem": "non_terminal_e4_arm"}

    def metrics(row: Mapping[str, Any]) -> dict[str, float | None]:
        decision = row.get("decision_epoch") or {}
        provider = row.get("provider_stats") or {}
        native = native_tick_metrics(
            _path(row, "decision_epoch", "native_outcome", "native_records") or []
        )
        return {
            "investigation_actions": _number(decision.get("investigation_actions")),
            "commit_attempted": float(decision.get("commit_attempted") is True),
            "window_exhausted_after_query": (
                float(decision["window_exhausted_after_query"])
                if isinstance(decision.get("window_exhausted_after_query"), bool) else None
            ),
            "visible_evidence_count": float(len(decision.get("visible_evidence_ids_end") or [])),
            "model_calls": _number(provider.get("llm_calls_ok")),
            "tool_calls_requested": _number(provider.get("tool_calls_requested")),
            "provider_retries": _number(provider.get("retry_attempts_total")),
            "native_actual_cost": _number(_path(decision, "native_outcome", "actual_cost")),
            **{
                f"native_cost_component:{key}": _number(value)
                for key, value in (
                    _path(decision, "native_outcome", "cost_components")
                    or _path(decision, "native_outcome", "ground_truth", "cost_components") or {}
                ).items()
            },
            **native,
        }

    reveal_metrics, withhold_metrics = metrics(reveal), metrics(withhold)
    return {
        "valid": True,
        "pair_id": _path(reveal, "cell", "pair_id"),
        "metric_contract": "paired_information_ablation_no_primary_score",
        "first_choice": {
            "reveal": _path(reveal, "decision_epoch", "first_choice"),
            "withhold": _path(withhold, "decision_epoch", "first_choice"),
        },
        "arms": {"reveal": reveal_metrics, "withhold": withhold_metrics},
        "reveal_minus_withhold": _numeric_delta(reveal_metrics, withhold_metrics),
    }
