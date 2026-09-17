"""Matched analysis contracts for OPERATE supplementary experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


E1_ARMS = frozenset({"logical_persistent", "logical_stateless"})
E3_DELAYS = frozenset({0.0, 1.0, 5.0})
E4_ARMS = frozenset({"reveal", "withhold"})


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


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


def native_tick_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate the shared native tick fields without inventing a new score."""

    rows = list(records)
    additive = {
        "production_cost": "native_production_cost_sum",
        "startup_cost": "native_startup_cost_sum",
        "shed_penalty": "native_shed_penalty_sum",
        "balance_error_mw": "native_balance_error_sum",
        "safety_violation_severity": "native_safety_severity_sum",
    }
    result = {"native_ticks": float(len(rows))}
    for source, target in additive.items():
        values = [_number(row.get(source)) for row in rows]
        result[target] = round(sum(value for value in values if value is not None), 6)
    result["native_catastrophic_ticks"] = float(
        sum(row.get("catastrophic_failure") is True for row in rows)
    )
    return result


def audit_e1_matrix(
    cells: Iterable[Mapping[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_model: str,
    expected_scoring_version: str,
) -> dict[str, Any]:
    """Accept E1 rows by frozen scenario signature rather than display ID."""

    latest = {
        (
            str(_path(row, "score", "scenario_signature")),
            str(row.get("interaction_mode")),
        ): row
        for row in rows
    }
    expected = list(cells)
    findings = []
    for cell in expected:
        key = (str(cell["scenario_signature"]), str(cell["condition"]))
        row = latest.get(key)
        problem = None
        llm = _path(row or {}, "trajectory_summary", "llm") or {}
        requests = int(llm.get("provider_model_identity_request_count") or 0)
        if row is None:
            problem = "missing"
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
        if problem:
            findings.append({
                "pair_id": cell["pair_id"],
                "condition": cell["condition"],
                "problem": problem,
            })
    failed_pairs = {row["pair_id"] for row in findings}
    expected_pairs = {str(cell["pair_id"]) for cell in expected}
    return {
        "schema_version": "operate-supplementary-e1-acceptance/2.0",
        "join_key": "scenario_signature+interaction_mode",
        "expected_cells": len(expected),
        "accepted_cells": len(expected) - len(findings),
        "expected_pairs": len(expected_pairs),
        "accepted_pairs": len(expected_pairs - failed_pairs),
        "findings": findings,
        "accepted": bool(expected) and not findings,
    }


def analyze_e1_pair(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare one persistent/stateless pair without changing the base score."""

    by_arm = {str(row.get("interaction_mode")): row for row in rows}
    if set(by_arm) != E1_ARMS:
        return {"valid": False, "problem": "incomplete_or_duplicate_e1_pair"}
    persistent = by_arm["logical_persistent"]
    stateless = by_arm["logical_stateless"]
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
        summary = row.get("trajectory_summary") or {}
        llm = summary.get("llm") or {}
        return {
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
    identity = {(row.get("scenario_id"), row.get("agent_name")) for row in by_delay.values()}
    if len(identity) != 1:
        return {"valid": False, "problem": "e3_identity_mismatch"}
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

    by_arm = {str(_path(row, "cell", "condition")): row for row in rows}
    if set(by_arm) != E4_ARMS:
        return {"valid": False, "problem": "incomplete_or_duplicate_e4_pair"}
    reveal, withhold = by_arm["reveal"], by_arm["withhold"]
    identity_fields = (
        "pair_id",
        "state_artifact_sha256",
        "runtime_implementation_tree_sha256",
        "provider_config_sha256",
    )
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
            "query_deadline_exhausted": float(
                decision.get("query_receipt_deadline_exhausted") is True
            ),
            "visible_evidence_count": float(len(decision.get("visible_evidence_ids_end") or [])),
            "model_calls": _number(provider.get("llm_calls_ok")),
            "tool_calls_requested": _number(provider.get("tool_calls_requested")),
            "provider_retries": _number(provider.get("retry_attempts_total")),
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
