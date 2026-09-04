#!/usr/bin/env python3
"""Bounded causal-agency diagnostic for the held JSPLIB SWV06 candidate.

This script does not change the scenario or reinterpret missing runtime evidence.
It records whether existing event, observation, action, backend-effect, and masked
replay evidence naturally forms the chain required by the agentic contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from run import load_scenario_yaml  # noqa: E402
from runner.episode import run_one  # noqa: E402

SCENARIO_ID = (
    "logistics/job_shop_dispatch/time_pressure/extreme/jobshop_swv06_dynamic_recovery_extreme_s44"
)
DEFAULT_SCENARIO = (
    REPO_ROOT / "scenarios/staging/held33_external_refine/logistics/job_shop_dispatch/"
    "time_pressure/extreme/jobshop_swv06_dynamic_recovery_extreme_s44.yaml"
)
DEFAULT_AGENTIC = (
    REPO_ROOT / "reports/protocol21_pending_union_fresh_current_20260812_wave2_realtraffic_stable/"
    "agentic_core_contract_protocol2_v21.json"
)
DEFAULT_BEHAVIORAL = (
    REPO_ROOT / "reports/held33_jsplib_swv06_behavioral_prefilter_current_20260812.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "reports/held33_jsplib_swv06_agency_diagnostic_current_20260812.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _find_row(payload: dict[str, Any], scenario_id: str) -> dict[str, Any]:
    for row in payload.get("results") or []:
        if isinstance(row, dict) and row.get("scenario_id") == scenario_id:
            return row
    raise ValueError(f"scenario absent from artifact: {scenario_id}")


def build_agency_diagnostic(result: dict[str, Any]) -> dict[str, Any]:
    """Summarize evidence without manufacturing an official causal edge."""
    trajectory = result.get("trajectory_summary") or {}
    complexity = trajectory.get("complexity") or {}
    events = complexity.get("material_exogenous_event_records") or []
    effects = complexity.get("agent_caused_event_records") or []
    official_edges = complexity.get("event_to_action_edges") or []
    per_action = (result.get("counterfactual") or {}).get("per_action") or []
    positive_actions = [
        row for row in per_action if float(row.get("marginal_prevented_loss") or 0.0) > 0.0
    ]
    plan_revision_count = int(complexity.get("explicit_plan_revision_count") or 0)
    valid_delegation = bool(
        (trajectory.get("event_adaptive_autonomy") or {}).get("valid_plan_delegation_observed")
    )
    effects_by_call: dict[str, list[dict[str, Any]]] = {}
    for effect in effects:
        call_id = str(effect.get("call_id") or "")
        if call_id:
            effects_by_call.setdefault(call_id, []).append(effect)

    records = []
    for event in events:
        event_tick = int(event.get("applied_tick"))
        candidates = [row for row in positive_actions if int(row.get("tick", -1)) >= event_tick]
        response = min(candidates, key=lambda row: int(row["tick"])) if candidates else None
        call_id = str((response or {}).get("call_id") or "")
        response_effects = effects_by_call.get(call_id, [])
        edge = next(
            (
                row
                for row in official_edges
                if row.get("source_event_id") == event.get("event_id")
                and row.get("call_id") == call_id
            ),
            None,
        )
        event_evidence_ids = list(event.get("evidence_ids") or [])
        observation_bound = bool(event_evidence_ids and edge)
        plan_response_present = bool(plan_revision_count or valid_delegation)
        records.append(
            {
                "event_id": event.get("event_id"),
                "event_tick": event_tick,
                "event_visibility": event.get("visibility"),
                "event_origin": event.get("origin"),
                "event_evidence_ids": event_evidence_ids,
                "first_positive_masked_action_tick": (response or {}).get("tick"),
                "first_positive_masked_action_call_id": call_id or None,
                "first_positive_masked_action_tool": (response or {}).get("tool_name"),
                "masked_marginal_prevented_loss": (response or {}).get("marginal_prevented_loss"),
                "backend_effect_count_for_call": len(response_effects),
                "backend_effect_event_ids": [row.get("event_id") for row in response_effects[:5]],
                "observation_binding_present": observation_bound,
                "validated_event_action_effect_edge_present": bool(edge),
                "plan_replacement_or_delegation_present": plan_response_present,
                "causal_chain_complete": bool(
                    response and response_effects and observation_bound and plan_response_present
                ),
                "diagnostic_only": True,
            }
        )

    tool_counts = Counter(str(row.get("tool_name") or "") for row in per_action)
    positive_tool_counts = Counter(str(row.get("tool_name") or "") for row in positive_actions)
    event_response_records = trajectory.get("event_response_records") or []
    blockers = []
    if not event_response_records:
        blockers.append("runtime_event_response_records_missing")
    if any(not row.get("event_evidence_ids") for row in records):
        blockers.append("event_observation_binding_missing")
    if not official_edges:
        blockers.append("validated_event_to_action_edge_missing")
    if not plan_revision_count and not valid_delegation:
        blockers.append("plan_replacement_or_delegation_missing")
    repair_deltas = [
        float(row.get("marginal_prevented_loss") or 0.0)
        for row in per_action
        if row.get("tool_name") == "repair_machine"
    ]
    if repair_deltas and max(repair_deltas) <= 0.0:
        blockers.append("repair_machine_masked_delta_nonpositive")

    return {
        "scenario_id": result.get("scenario_id"),
        "scenario_signature": result.get("scenario_signature"),
        "task_completed": bool((result.get("task_completion") or {}).get("completed")),
        "runtime_event_response_record_count": len(event_response_records),
        "material_event_count": len(events),
        "validated_event_to_action_edge_count": len(official_edges),
        "explicit_plan_revision_count": plan_revision_count,
        "valid_plan_delegation_observed": valid_delegation,
        "per_action_masked_replay": {
            "status": (result.get("counterfactual") or {}).get("per_action_status"),
            "expected": (result.get("counterfactual") or {}).get("per_action_expected"),
            "completed": (result.get("counterfactual") or {}).get("per_action_completed"),
            "total_actions": len(per_action),
            "positive_actions": len(positive_actions),
            "actions_by_tool": dict(sorted(tool_counts.items())),
            "positive_actions_by_tool": dict(sorted(positive_tool_counts.items())),
            "repair_machine_max_delta": max(repair_deltas) if repair_deltas else None,
        },
        "diagnostic_event_response_records": records,
        "blockers": blockers,
        "native_outcome_influence_observed": bool(positive_actions),
        "natural_adaptive_replanning_proof": not blockers,
        "full_protocol21_replay_recommended": not blockers,
        "disposition": "held_repair" if blockers else "candidate_prefilter_survivor",
        "core_admission_claimed": False,
    }


def run_diagnostic(
    *,
    scenario_path: Path,
    agentic_path: Path,
    behavioral_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    scenario_hash_before = _sha256(scenario_path)
    old_agentic = _find_row(_load(agentic_path), SCENARIO_ID)
    behavioral = _find_row(_load(behavioral_path), SCENARIO_ID)
    relative = scenario_path.relative_to(REPO_ROOT / "scenarios").with_suffix("").as_posix()
    scenario = load_scenario_yaml(relative)
    result = run_one(
        scenario,
        "oracle_offline",
        per_action_attribution=True,
        per_action_cap=None,
    )
    if result.get("scenario_id") != SCENARIO_ID:
        raise RuntimeError("runtime scenario identity mismatch")
    diagnostic = build_agency_diagnostic(result)
    scenario_hash_after = _sha256(scenario_path)
    if scenario_hash_after != scenario_hash_before:
        raise RuntimeError("scenario changed during diagnostic")
    report = {
        "schema_version": "jsplib_swv06_agency_diagnostic_v1",
        "candidate_only": True,
        "bounded_rows": 1,
        "implementation_identity": implementation_identity(REPO_ROOT),
        "input_binding": {
            "scenario_path": scenario_path.relative_to(REPO_ROOT).as_posix(),
            "scenario_file_sha256": scenario_hash_before,
            "scenario_identity_preserved": True,
            "agentic_report_path": agentic_path.relative_to(REPO_ROOT).as_posix(),
            "agentic_report_sha256": _sha256(agentic_path),
            "behavioral_report_path": behavioral_path.relative_to(REPO_ROOT).as_posix(),
            "behavioral_report_sha256": _sha256(behavioral_path),
            "old_agentic_status": old_agentic.get("status"),
            "old_agentic_blockers": old_agentic.get("blockers") or [],
            "new_behavioral_status": behavioral.get("status"),
        },
        "diagnostic": diagnostic,
        "scientific_interpretation": {
            "runtime_instrumentation_gap": True,
            "reference_plan_contract_gap": True,
            "contract_relaxation_supported": False,
            "reason": (
                "Positive post-event masked backend effects exist, but runtime observation "
                "binding and explicit plan replacement/delegation evidence are absent."
            ),
        },
        "full_protocol21_executed": False,
        "eligible_as_post_221_increment": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--agentic-report", type=Path, default=DEFAULT_AGENTIC)
    parser.add_argument("--behavioral-report", type=Path, default=DEFAULT_BEHAVIORAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_diagnostic(
        scenario_path=args.scenario.resolve(),
        agentic_path=args.agentic_report.resolve(),
        behavioral_path=args.behavioral_report.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
