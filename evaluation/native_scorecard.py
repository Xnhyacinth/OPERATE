"""Reference-free, read-only measurements for OPERATE 0.24.

Artifact authentication belongs to the caller. Native hard gates and costs are
recomputed from the authenticated scoring snapshot, not copied score summaries.
No optimizer, environment or model is executed. Absence of a diagnostic never
invalidates a separate native measurement.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from core.counterfactual import _cost_components_are_usable
from evaluation.native_objectives import extract_native_objective
from evaluation.scorer import score_system_survival
from evaluation.task_completion import _microgrid_native_task_loss
from evaluation.task_quality import FJSP, evaluate_task_quality

EVALUATION_VERSION = "0.24.0"


def _read(binding: dict, name: str) -> Any:
    descriptor = binding["artifacts"][name]
    raw = Path(descriptor["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise ValueError("bound artifact changed during native evaluation")
    return (
        json.loads(raw)
        if name == "scoring_inputs_artifact"
        else [json.loads(line) for line in raw.splitlines() if line.strip()]
    )


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _wait_diagnostic(row: dict, native: dict, binding: dict) -> dict:
    result = dict(
        applicable=False,
        cost_reduction=None,
        wait_cost=None,
        reason="existing_wait_evidence_unavailable",
        score=None,
    )
    cf = row.get("counterfactual") or {}
    if not (
        binding.get("counterfactual_bound") is True
        and native.get("applicable") is True
        and cf.get("applicable") is True
        and cf.get("masking_policy") == "wait_only"
    ):
        return result
    if row.get("backend_kind") == "pymgrid_economic_dispatch":
        return {**result, "reason": "economic_wait_is_not_native_state_loss"}
    components = cf.get("counterfactual_components")
    wait = cf.get("counterfactual_cost")
    if not (
        native.get("counterfactual_consistent") is True
        and isinstance(components, dict)
        and components
        and set(components) == set(cf.get("actual_components") or {})
        and _cost_components_are_usable(
            components,
            (row.get("ground_truth_summary") or {}).get(
                "cost_component_value_domains", {}
            ),
        )
        and all(_finite(v) for v in components.values())
        and _finite(wait)
        and math.isclose(sum(components.values()), wait, rel_tol=1e-10, abs_tol=1e-7)
    ):
        return {**result, "reason": "wait_objective_or_components_mismatch"}
    delta = wait - native["actual_cost"]
    if not _finite(delta):
        return {**result, "reason": "wait_difference_nonfinite"}
    return {
        **result,
        "applicable": True,
        "cost_reduction": delta,
        "wait_cost": wait,
        "reason": "existing_bound_wait_native_difference",
        "objective_id": native["objective_id"],
        "unit": native["unit"],
        "evidence_ids": native["evidence_ids"],
        "snapshot_sha256": binding["artifacts"]["scoring_inputs_artifact"]["sha256"],
    }


def evaluate_native_scorecard(
    episode: dict, spec: dict, *, artifact_binding: dict
) -> dict:
    """Produce three views, without manufacturing three numeric capability scores."""
    result = dict(
        schema_version="native_scorecard.v1",
        evaluation_version=EVALUATION_VERSION,
        absolute_quality_score=None,
        comparison_blocker=None,
        native_outcome=dict(applicable=False, reason="unbound_episode_artifacts"),
        wait_diagnostic=dict(
            applicable=False,
            score=None,
            cost_reduction=None,
            reason="unbound_episode_artifacts",
        ),
        operational_decisions=dict(score=None, reason="unbound_episode_artifacts"),
        sustained_operation=dict(score=None, reason="unbound_episode_artifacts"),
    )
    if not (
        artifact_binding.get("verified") is True
        and artifact_binding.get("native_cost_bound") is True
    ):
        result["comparison_blocker"] = "unbound_episode_artifacts"
        return result
    inputs = _read(artifact_binding, "scoring_inputs_artifact")["payload"]["inputs"]
    ledger = _read(artifact_binding, "evidence_ledger_artifact")
    records = inputs.get("backend_tick_records")
    fatal = inputs.get("chose_fatal_option")
    if (
        not isinstance(records, list)
        or not records
        or not all(isinstance(r, dict) for r in records)
        or type(fatal) is not bool
    ):
        result["native_outcome"]["reason"] = "native_snapshot_state_missing"
        result["comparison_blocker"] = "native_snapshot_state_missing"
        return result
    # Use the snapshot's evidence inventory; model text and cached scores cannot
    # declare a safety gate or fabricate cost evidence.
    state_ids = [r["evidence_id"] for r in ledger if r.get("kind") == "backend_tick"]
    cost_ids = [
        r["evidence_id"] for r in ledger if r.get("kind") == "cost_summary"
    ] or state_ids
    survival = asdict(score_system_survival(records, evidence_ids=state_ids))
    row = deepcopy(episode)
    row["ground_truth_summary"]["chose_fatal_option"] = fatal
    row["score"] = {
        "dimensions": [
            survival,
            dict(name="economic_cost", applicable=True, evidence_ids=cost_ids),
        ]
    }
    if row.get("backend_kind") == "pymgrid_economic_dispatch":
        keys = ("balance_error_mw", "shed_penalty")
        if any(not _finite(record.get(key)) for record in records for key in keys):
            result["native_outcome"]["reason"] = "native_state_loss_records_invalid"
            result["comparison_blocker"] = "native_state_loss_records_invalid"
            return result
        row["ground_truth_summary"]["_task_tick_records"] = records
        row["task_completion"]["evidence"] = {
            **(row["task_completion"].get("evidence") or {}),
            "actual_task_loss": _microgrid_native_task_loss(
                inputs["cost_components"], records, keys
            ),
            "task_loss_component_keys": list(keys),
        }
    native = extract_native_objective(
        row,
        terminal_runtime_verified=artifact_binding.get(
            "native_completed_tree_drift_verified"
        )
        is True,
    )
    # Legacy task_success is a mixture of wait-relative mitigation and service
    # contracts; never relabel it as absolute task attainment.
    native.pop("task_success", None)
    result["native_outcome"] = native
    if not native.get("applicable"):
        result["comparison_blocker"] = native.get("reason")
    if row.get("backend_kind") == FJSP and native.get("applicable"):
        completion = evaluate_task_quality(row, artifact_binding=artifact_binding)
        result["source_obligations"] = {
            key: completion.get(key)
            for key in (
                "mandatory_complete",
                "completion_fraction",
                "evidence_ids",
                "reason",
            )
        }
        if not native["hard_failure"]:
            if type(completion.get("mandatory_complete")) is not bool:
                result["comparison_blocker"] = completion.get("reason")
            else:
                native["feasible"] = completion["mandatory_complete"]
                native["evidence_ids"] = list(
                    dict.fromkeys(native["evidence_ids"] + completion["evidence_ids"])
                )
    result["wait_diagnostic"] = _wait_diagnostic(row, native, artifact_binding)
    calls = [
        r
        for r in ledger
        if r.get("kind") == "tool_call" and r.get("source") in ("engine", "tool")
    ]
    controls = [
        r for r in calls if (r.get("payload") or {}).get("state_changing") is True
    ]
    result["operational_decisions"] = dict(
        score=None,
        recorded_tool_attempts=len(calls),
        recorded_control_attempts=len(controls),
        evidence_ids=[r["evidence_id"] for r in calls],
        reason="observed_attempt_inventory_not_effectiveness_or_proactivity",
        opportunity_success_rate=None,
    )
    ticks, horizon = artifact_binding.get("trace_ticks"), spec.get("horizon_ticks")
    result["sustained_operation"] = dict(
        score=None,
        recorded_ticks=ticks,
        configured_ticks=horizon,
        full_observation_window=type(horizon) is int and ticks == horizon,
        reason="authenticated_time_coverage_not_persistence_or_memory_score",
        trajectory_sha256=artifact_binding["artifacts"]["trajectory_artifact"][
            "sha256"
        ],
    )
    if (
        type(horizon) is not int
        or horizon <= 0
        or type(ticks) is not int
        or (
            ticks != horizon
            and not native.get("hard_failure")
            and not (row.get("backend_kind") == FJSP and native.get("feasible"))
        )
    ):
        result["comparison_blocker"] = (
            result["comparison_blocker"] or "incomplete_observation_window"
        )
    return result
