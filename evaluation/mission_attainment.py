"""Source-obligation attainment with fixed, explicitly limited support.

No opponent, reference, process count or model-written claim sets a task score.
Contracts come from mission_contracts' source-only compiler. A supported-subset
score is never promoted to a full-suite score when acceptance is undefined.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from evaluation.native_scorecard import _read
from evaluation.scorer import score_system_survival
from evaluation.target_attainment import aggregate_target_attainment

VERSION = "0.25.0"


def evaluate_mission(row: dict, contract: dict) -> dict:
    result = dict(
        score=None,
        attained=None,
        completion_fraction=None,
        evidence_ids=[],
        reason="acceptance_not_defined",
        contract_sha256=contract.get("contract_sha256"),
        interpretation=contract.get("interpretation"),
        safety=None,
    )
    if contract.get("eligible") is not True:
        return result

    def missing(reason):
        return {**result, "reason": reason}

    if any(
        row.get(k) != contract.get(k)
        for k in ("scenario_signature", "seed", "backend_kind", "domain")
    ):
        return missing("mission_identity_mismatch")
    binding = row.get("artifact_binding") or {}
    if row.get("blocker") and row["blocker"] not in (
        "incomplete_observation_window",
        "multiple_comparison_scopes",
    ):
        return missing(row["blocker"])
    if not (
        binding.get("verified") is True and binding.get("native_cost_bound") is True
    ):
        return missing("unbound_episode_artifacts")
    if (
        row.get("blocker") == "multiple_comparison_scopes"
        and row.get("individual_execution_verified") is not True
    ):
        return missing("individual_execution_identity_unproven")
    snapshot = _read(binding, "scoring_inputs_artifact")["payload"]
    if any(
        snapshot.get("identity", {}).get(k) != contract[k]
        for k in ("scenario_signature", "seed")
    ):
        return missing("snapshot_mission_identity_mismatch")
    inputs = snapshot["inputs"]
    ledger = _read(binding, "evidence_ledger_artifact")
    by_id = {e["evidence_id"]: e for e in ledger}
    valid = set(by_id)
    snapshot_items = (inputs.get("evidence_logger") or {}).get("items")
    if (
        len(by_id) != len(ledger)
        or not isinstance(snapshot_items, list)
        or any(
            not isinstance(e, dict) or by_id.get(e.get("evidence_id")) != e
            for e in snapshot_items
        )
    ):
        return missing("snapshot_evidence_mismatch")
    state_ids = [
        e["evidence_id"]
        for e in ledger
        if e.get("kind") == "backend_tick" and e.get("source") == "engine"
    ]
    records = inputs.get("backend_tick_records")
    fatal = inputs.get("chose_fatal_option")
    if (
        not isinstance(records, list)
        or not records
        or type(fatal) is not bool
        or not state_ids
        or any(
            not isinstance(r, dict) or type(r.get("catastrophic_failure")) is not bool
            for r in records
        )
    ):
        return missing("mission_safety_evidence_missing")
    survival = asdict(score_system_survival(records, evidence_ids=state_ids))
    if survival["applicable"] is not True:
        return missing("mission_safety_gate_unknown")
    hard = fatal or survival["floor_violation"]
    result["safety"] = dict(verified=True, hard_failure=hard, evidence_ids=state_ids)
    result["evidence_ids"] = state_ids
    if hard:
        return {
            **result,
            "score": 0.0,
            "attained": False,
            "reason": "native_hard_failure",
        }
    if contract["acceptance_kind"] != "complete_source_operations_by_horizon":
        return missing("unsupported_acceptance_kind")
    trace = _read(binding, "trajectory_artifact")
    if (
        not trace
        or [r.get("tick") for r in trace] != list(range(1, len(trace) + 1))
        or len(trace) != binding.get("trace_ticks")
        or len(records) != len(trace)
    ):
        return missing("mission_trace_clock_invalid")
    terminal = trace[-1]
    ids = terminal.get("evidence_ids")
    if not isinstance(ids, list) or not ids or any(e not in valid for e in ids):
        return missing("mission_terminal_evidence_missing")
    obs = terminal.get("observation") or {}
    req = contract["requirements"]
    fields = (
        "jobs_total",
        "jobs_arrived",
        "operations_total",
        "operations_completed",
        "operations_scheduled",
        "operations_cancelled",
    )
    if any(type(obs.get(k)) is not int or obs[k] < 0 for k in fields):
        return missing("mission_counts_missing_or_invalid")
    jobs, operations = req["source_jobs_total"], req["source_operations_total"]
    if (
        type(jobs) is not int
        or jobs <= 0
        or type(operations) is not int
        or operations <= 0
        or obs["jobs_total"] != jobs
        or obs["jobs_arrived"] > jobs
        or obs["operations_total"] > operations
        or not 0
        <= obs["operations_completed"]
        <= obs["operations_scheduled"]
        <= operations
        or obs["operations_scheduled"] + obs["operations_cancelled"]
        > obs["operations_total"]
        or (obs["jobs_arrived"] == jobs and obs["operations_total"] != operations)
        or obs["operations_cancelled"] > operations
    ):
        return missing("mission_source_counts_mismatch")
    complete = (
        obs["jobs_arrived"] == jobs
        and obs["operations_total"] == operations
        and obs["operations_completed"] == operations
        and obs["operations_cancelled"] == 0
    )
    deadline = req["deadline_boundary"]
    if type(deadline) is not int or deadline <= 0 or len(trace) > deadline:
        return missing("mission_deadline_evidence_invalid")
    if len(trace) < deadline and not complete:
        return missing("incomplete_observation_window")
    return {
        **result,
        "score": 100.0 if complete else 0.0,
        "attained": complete,
        "completion_fraction": obs["operations_completed"] / operations,
        "observed_counts": {k: obs[k] for k in fields},
        "source_jobs_total": jobs,
        "source_operations_total": operations,
        "evidence_ids": list(dict.fromkeys(state_ids + ids)),
        "reason": "source_obligations_completed"
        if complete
        else "source_obligations_unfulfilled",
        "quality_target_status": "no_absolute_cost_budget_declared",
        "deadline_basis": "episode_tick_budget_not_native_job_due_dates",
    }


def aggregate_missions(
    rows: list[dict], contracts: list[dict], *, models: list[str]
) -> dict:
    if not contracts:
        raise ValueError("mission suite must be nonempty")
    suite = {(c["scenario_signature"], c["seed"]): c for c in contracts}
    if len(suite) != len(contracts):
        raise ValueError("duplicate mission suite case")
    if any((r["scenario_signature"], r["seed"]) not in suite for r in rows):
        raise ValueError("mission row outside suite")
    eligible = [c for c in contracts if c["eligible"]]
    keys = {(c["scenario_signature"], c["seed"]) for c in eligible}
    measured = []
    for r in rows:
        if (r["scenario_signature"], r["seed"]) in keys:
            m = r["mission"]
            measured.append(
                dict(
                    model=r["model"],
                    scenario_signature=r["scenario_signature"],
                    seed=r["seed"],
                    measurement=m,
                    safety=m.get("safety") or {},
                )
            )

    def aggregate(scope):
        if not scope:
            return dict(
                n_expected=0,
                models=[],
                leaderboard=[],
                reason="no_source_defined_acceptance",
            )
        selected = {(c["scenario_signature"], c["seed"]) for c in scope}
        result = aggregate_target_attainment(
            [r for r in measured if (r["scenario_signature"], r["seed"]) in selected],
            scope,
            models=models,
        )
        result.update(
            schema_version="mission_attainment.v1",
            evaluation_version=VERSION,
            score_units="percent_source_obligation_attainment",
        )
        return result

    result = aggregate(eligible)
    for item in result["models"]:
        item["scorable_task_score"] = item.pop("score")
        item["scorable_task_rank"] = item.pop("rank")
        item["primary_score"] = (
            item["scorable_task_score"] if len(eligible) == len(contracts) else None
        )
        item["primary_rank"] = (
            item["scorable_task_rank"] if len(eligible) == len(contracts) else None
        )
    if not eligible:
        result["models"] = [
            dict(
                model=m,
                scorable_task_score=None,
                scorable_task_rank=None,
                primary_score=None,
                primary_rank=None,
                n_expected=0,
                n_scored=0,
            )
            for m in models
        ]
    result["slices"] = {
        name: aggregate(
            [
                c
                for c in eligible
                if c.get("strata", {}).get(name, {}).get("status") == "source_declared"
            ]
        )
        for name in sorted({name for c in contracts for name in c.get("strata", {})})
    }
    result.update(
        schema_version="mission_attainment.v1",
        evaluation_version=VERSION,
        suite_cases=len(contracts),
        scorable_cases=len(eligible),
        acceptance_coverage_complete=len(eligible) == len(contracts),
        acceptance_kind_counts=dict(Counter(c["acceptance_kind"] for c in contracts)),
        score_interpretation="fixed_source_acceptance_subset_not_full_suite_unless_contract_coverage_complete",
        full_suite_leaderboard=[
            x for x in result.get("leaderboard", []) if x["primary_score"] is not None
        ],
        formal_run_certified=False,
        leaderboard_eligible=False,
    )
    return result
