#!/usr/bin/env python3
"""Run a source-grounded, diagnostic-only operational-agency positive control.

The control overlays one deterministic early machine outage on a locked JSPLIB
instance.  It never claims source independence or Core admission.  Its sole
purpose is to prove that the runtime evaluator recognizes a replay-bound native
response whose substitutable repair calls have positive *group* effect even
when every leave-one-call-out delta is zero.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines import REGISTRY  # noqa: E402
from baselines.oracle_offline import OracleOfflineAgent  # noqa: E402
from core import Action  # noqa: E402
from core.implementation_identity import implementation_identity  # noqa: E402
from runner.episode import run_one  # noqa: E402

DEFAULT_BASE = (
    REPO_ROOT / "scenarios/operate_v0_58_0/logistics/job_shop_dispatch/"
    "time_pressure/extreme/jobshop_swv06_dynamic_recovery_extreme_s44.yaml"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "reports/operate_v0_58_0/agency/generic_runtime.json"
)
CONTROL_SCENARIO_ID = "diagnostic/agency_positive_control/jsplib_swv06_early_machine_breakdown"
CONTROL_AGENT_NAME = "agency_positive_control_oracle"


def _bind_visible_declared_event_evidence(
    action: Action, observation: dict[str, Any]
) -> Action:
    visible_events = [
        event
        for event in observation.get("__last_realized_events__") or []
        if isinstance(event, dict)
        and event.get("origin") == "declared_perturbation"
        and event.get("hidden") is not True
        and event.get("evidence_ids")
    ]
    if not visible_events:
        return action
    evidence_ids = [str(value) for value in visible_events[-1]["evidence_ids"]]
    for call in action.tool_calls:
        if call.name in {"wait", "noop"}:
            continue
        call.consumes_evidence_ids = list(
            dict.fromkeys([*(call.consumes_evidence_ids or []), *evidence_ids])
        )
    return action


class AgencyPositiveControlOracle(OracleOfflineAgent):
    name = CONTROL_AGENT_NAME

    def act(
        self,
        observation: dict[str, Any],
        tool_specs: list[dict[str, Any]],
    ) -> Action:
        return _bind_visible_declared_event_evidence(
            super().act(observation, tool_specs), observation
        )


@contextmanager
def _registered_control_agent() -> Iterator[None]:
    previous = REGISTRY.get(CONTROL_AGENT_NAME)
    REGISTRY[CONTROL_AGENT_NAME] = AgencyPositiveControlOracle
    try:
        yield
    finally:
        if previous is None:
            REGISTRY.pop(CONTROL_AGENT_NAME, None)
        else:
            REGISTRY[CONTROL_AGENT_NAME] = previous


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_parent_scenario(base: dict[str, Any]) -> None:
    if base.get("domain") != "logistics" or base.get("backend_kind") != "jsplib_job_shop":
        raise ValueError("runtime positive control requires logistics/jsplib_job_shop")
    scenario_id = base.get("scenario_id")
    if not isinstance(scenario_id, str) or base.get("seed_id") != scenario_id:
        raise ValueError("runtime positive control requires bound parent scenario identity")
    horizon = base.get("horizon_ticks")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 2:
        raise ValueError("runtime positive control requires a valid parent horizon")
    source_contract = base.get("source_contract")
    if not isinstance(source_contract, dict) or not source_contract.get("runtime_input"):
        raise ValueError("runtime positive control requires source-bound runtime input")
    backend = base.get("backend_config")
    if not isinstance(backend, dict) or not isinstance(backend.get("task_contract"), dict):
        raise ValueError("runtime positive control requires parent task_contract")


def build_control_scenario(base: dict[str, Any]) -> dict[str, Any]:
    """Return a detached diagnostic overlay without mutating the base row."""
    scenario = copy.deepcopy(base)
    scenario["seed_id"] = CONTROL_SCENARIO_ID
    scenario["scenario_id"] = CONTROL_SCENARIO_ID
    scenario.pop("scenario_signature", None)
    scenario["perturbations"] = [
        {
            "kind": "machine_breakdown",
            "trigger_tick": 1,
            "duration_ticks": 100,
            "hidden": False,
            "target": {
                "machine_id": 1,
                "source": "locked_jsplib_machine_set",
            },
            "intensity": 1.0,
            "notes": (
                "Diagnostic-only deterministic early outage over a locked JSPLIB machine identity."
            ),
        }
    ]
    backend = scenario.setdefault("backend_config", {})
    dynamic = backend.setdefault("dynamic_job_shop", {})
    dynamic["recovery_clearance_ticks"] = 1
    backend["release_ready"] = False
    backend["release_reentry_ready"] = False
    backend["agency_positive_control"] = {
        "diagnostic_only": True,
        "declared_perturbation": True,
        "source_independence_credit": False,
        "purpose": "runtime_evaluator_sensitivity_calibration",
    }
    backend["task_contract"]["event_response_window"] = {
        "first_tick": 2,
        "last_tick": int(scenario["horizon_ticks"]) - 1,
    }
    backend["task_requirements"] = {
        "min_distinct_control_ticks": 2,
        "min_distinct_physical_tools": 2,
        "ordered_tool_milestones": [
            {"tool": "dispatch_ready_operations", "not_after_tick": 2},
            {
                "tool": "repair_machine",
                "not_before_tick": 3,
                "not_after_tick": int(scenario["horizon_ticks"]) - 1,
            },
            {
                "tool": "dispatch_ready_operations",
                "not_before_tick": 5,
                "not_after_tick": int(scenario["horizon_ticks"]) - 1,
            },
        ],
    }
    return scenario


def _source_bindings(scenario: dict[str, Any]) -> dict[str, str]:
    runtime_inputs = (scenario.get("source_contract") or {}).get("runtime_input") or []
    bindings: dict[str, str] = {}
    for value in runtime_inputs:
        if not isinstance(value, str) or not value:
            raise ValueError("runtime source path must be a non-empty string")
        path = (REPO_ROOT / value).resolve()
        path.relative_to(REPO_ROOT.resolve())
        if not path.is_file():
            raise FileNotFoundError(value)
        bindings[value] = _sha256(path)
    if not bindings:
        raise ValueError("positive control requires a runtime source binding")
    return bindings


def _coverage_complete(counterfactual: dict[str, Any], prefix: str) -> bool:
    expected = counterfactual.get(f"{prefix}_expected")
    return bool(
        isinstance(expected, int)
        and not isinstance(expected, bool)
        and expected >= 0
        and counterfactual.get(f"{prefix}_attempted") == expected
        and counterfactual.get(f"{prefix}_completed") == expected
        and counterfactual.get(f"{prefix}_failures") == []
        and counterfactual.get(f"{prefix}_status") == "complete"
        and isinstance(
            counterfactual.get(
                "per_action" if prefix == "per_action" else "per_action_groups"
            ),
            list,
        )
        and len(
            counterfactual[
                "per_action" if prefix == "per_action" else "per_action_groups"
            ]
        )
        == expected
    )


def _terminal_integrity(result: dict[str, Any]) -> dict[str, Any]:
    trajectory = result.get("trajectory_summary") or {}
    raw = trajectory.get("terminal_integrity") or {}
    ground_truth = result.get("ground_truth_summary") or {}
    if not (
        isinstance(raw, dict)
        and raw.get("release_ready") is True
        and raw.get("unresolved_pending_actions") == {}
        and raw.get("unanswered_interrupt_reasons") == []
        and result.get("status") != "error"
        and ground_truth.get("chose_fatal_option") is not True
    ):
        raise ValueError("positive-control terminal integrity is incomplete")
    return {
        **raw,
        "terminal": True,
        "fatal": False,
        "fatal_error": None,
        "orphan_process_count": 0,
    }


def _authoritative_declared_event(
    result: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    event_id = str(record.get("event_id") or "")
    realized = (result.get("ground_truth_summary") or {}).get("realized_events") or []
    source = next(
        (
            event
            for event in realized
            if isinstance(event, dict)
            and event.get("event_id") == event_id
            and event.get("type") == "machine_breakdown"
            and event.get("origin") == "declared_perturbation"
            and event.get("hidden") is not True
        ),
        None,
    )
    source_ids = set((source or {}).get("evidence_ids") or [])
    trigger_ids = set(record.get("trigger_evidence_ids") or [])
    consumed_ids = set(record.get("action_consumes_evidence_ids") or [])
    if not (
        source is not None
        and record.get("causal_parent_event_id") == event_id
        and source_ids.intersection(trigger_ids).intersection(consumed_ids)
    ):
        raise ValueError("positive control lacks authoritative declared event binding")
    return {
        **record,
        "event_origin": "declared_perturbation",
        "declared_perturbation": True,
    }


def summarize_positive_control(result: dict[str, Any]) -> dict[str, Any]:
    """Validate and retain only the evidence needed by the readiness gate."""
    counterfactual = result.get("counterfactual") or {}
    if counterfactual.get("per_action_capped") is not False or not _coverage_complete(
        counterfactual, "per_action"
    ):
        raise ValueError("positive control requires complete uncapped per-action replay")
    groups = counterfactual.get("per_action_groups") or []
    repair_groups = [
        row for row in groups if isinstance(row, dict) and row.get("tool_name") == "repair_machine"
    ]
    if counterfactual.get("per_action_group_status") != "complete":
        raise ValueError("positive control requires complete action-group replay")
    expected = int(counterfactual.get("per_action_group_expected") or 0)
    attempted = int(counterfactual.get("per_action_group_attempted") or 0)
    completed = int(counterfactual.get("per_action_group_completed") or 0)
    if not expected or not expected == attempted == completed:
        raise ValueError("positive control action-group coverage is incomplete")
    if counterfactual.get("per_action_group_failures"):
        raise ValueError("positive control action-group replay has failures")
    if len(repair_groups) != 1:
        raise ValueError("exactly one repair action group is required")
    group = repair_groups[0]
    delta = float(group.get("masked_action_group_delta") or 0.0)
    call_ids = group.get("call_ids")
    if delta <= 0.0 or not isinstance(call_ids, list) or len(call_ids) < 2:
        raise ValueError("repair action group must have positive replay effect")

    trajectory = result.get("trajectory_summary") or {}
    records = trajectory.get("event_response_records") or []
    valid_evidence_ids = trajectory.get(
        "operational_agency_valid_evidence_ids"
    )
    if not isinstance(valid_evidence_ids, list):
        raise ValueError("positive control lacks authoritative evidence IDs")
    matches = [
        row
        for row in records
        if isinstance(row, dict)
        and row.get("masked_action_group_id") == group.get("group_id")
        and row.get("masked_action_group_call_ids") == call_ids
        and float(row.get("masked_action_group_delta") or 0.0) == delta
    ]
    if len(matches) != 1:
        raise ValueError("runtime event response does not bind the replayed group")
    record = _authoritative_declared_event(result, matches[0])
    if not record.get("backend_effect_evidence_ids"):
        raise ValueError("positive control lacks native backend effect evidence")

    profile = trajectory.get("operational_agency_profile") or {}
    outcome = (profile.get("dimensions") or {}).get("outcome_influence") or {}
    if (
        profile.get("runtime_binding_verified") is not True
        or profile.get("runtime_evidence_binding_verified") is not True
        or profile.get("masked_replay_binding_verified") is not True
        or int(profile.get("causal_record_count") or 0) < 1
        or outcome.get("applicable") is not True
        or not outcome.get("evidence_ids")
    ):
        raise ValueError("operational-agency profile did not pass runtime binding")
    task = result.get("task_completion") or {}
    if task.get("completed") is not True:
        raise ValueError("positive-control native task did not complete")
    terminal = _terminal_integrity(result)
    return {
        "scenario_id": result.get("scenario_id"),
        "scenario_signature": result.get("scenario_signature"),
        "agent_name": result.get("agent_name"),
        "seed": result.get("seed"),
        "task_completion": task,
        "terminal_integrity": terminal,
        "counterfactual": {**counterfactual, "repair_group": group},
        "event_response_record": record,
        "event_response_records": records,
        "operational_agency_valid_evidence_ids": valid_evidence_ids,
        "operational_agency_profile": profile,
    }


def run_positive_control(
    *,
    base_path: Path,
    output_path: Path,
    repeats: int,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("positive control requires at least two deterministic repeats")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("base scenario YAML must contain an object")
    validate_parent_scenario(base)
    scenario = build_control_scenario(base)
    source_bindings = _source_bindings(scenario)
    implementation_before = implementation_identity(REPO_ROOT)
    with _registered_control_agent():
        summaries = [
            summarize_positive_control(
                run_one(
                    scenario,
                    CONTROL_AGENT_NAME,
                    per_action_attribution=True,
                    per_action_cap=None,
                    per_action_group_attribution=True,
                    per_action_group_cap=None,
                    within_tick_interaction=True,
                )
            )
            for _ in range(repeats)
        ]
    deterministic_fields = (
        "scenario_signature",
        "task_completion",
        "counterfactual",
        "event_response_record",
        "operational_agency_profile",
    )
    reference = {key: summaries[0][key] for key in deterministic_fields}
    if any(
        {key: summary[key] for key in deterministic_fields} != reference
        for summary in summaries[1:]
    ):
        raise RuntimeError("runtime positive control is not deterministic")
    implementation_after = implementation_identity(REPO_ROOT)
    implementation_stable = (
        implementation_before["implementation_tree_sha256"]
        == implementation_after["implementation_tree_sha256"]
    )
    if not implementation_stable:
        raise RuntimeError("implementation tree changed during runtime positive control")
    report = {
        "schema_version": "operational-agency-runtime-positive-control-v1",
        "status": "passed",
        "diagnostic_only": True,
        "release_admission": False,
        "core_admission_claimed": False,
        "source_independence_credit": False,
        "implementation_identity": implementation_before,
        "implementation_stability": {
            "before": implementation_before,
            "after": implementation_after,
            "passed": True,
        },
        "attribution_contract": {
            "mode": "complete_uncapped",
            "per_action_cap": None,
            "per_action_group_cap": None,
        },
        "base_scenario_binding": {
            "path": base_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(base_path),
            "scenario_id": base.get("scenario_id"),
        },
        "source_file_bindings": source_bindings,
        "overlay_contract": {
            "origin": "declared_perturbation",
            "event_kind": "machine_breakdown",
            "trigger_tick": 1,
            "duration_ticks": 100,
            "target_machine_id": 1,
            "source_identity": "locked_jsplib_machine_set",
            "source_independence_credit": False,
        },
        "determinism": {"repeats": repeats, "passed": True},
        "result": summaries[0],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scenario", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    report = run_positive_control(
        base_path=args.base_scenario.resolve(),
        output_path=args.output.resolve(),
        repeats=args.repeats,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "causal_record_count": report["result"]["operational_agency_profile"][
                    "causal_record_count"
                ],
                "masked_action_group_delta": report["result"]["counterfactual"]["repair_group"][
                    "masked_action_group_delta"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
