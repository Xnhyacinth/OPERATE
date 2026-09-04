#!/usr/bin/env python3
"""Run a locked-M5, source-schedule Logistics agency positive control."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines import REGISTRY, BaselineAgent  # noqa: E402
from core import Action, ToolCall  # noqa: E402
from core.implementation_identity import implementation_identity  # noqa: E402
from evaluation.operational_agency import (  # noqa: E402
    operational_agency_profile_is_consistent,
)
from runner.episode import run_one  # noqa: E402
from scripts.run_operational_agency_known_groups_calibration import (  # noqa: E402
    _full_uncapped_attribution,
)

AGENT_NAME = "diagnostic_logistics_source_schedule_control"
DEFAULT_SCENARIO = (
    REPO_ROOT
    / "scenarios/operate_v0_58_0/logistics/"
    "inventory_replenishment/"
    "time_pressure/medium/"
    "m5_household_1_004_ca_2_d1705_30d_lt6_cap56_protocol21_migration_v57__12b2fe55__relabel_v1.yaml"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "reports/operate_v0_58_0/agency/logistics_natural.json"
)


def _tool_names(tool_specs: list[dict[str, Any]]) -> set[str]:
    return {
        str((spec.get("function") or {}).get("name") or "")
        for spec in tool_specs
        if isinstance(spec, dict) and isinstance(spec.get("function"), dict)
    }


def _visible_demand_event(observation: Mapping[str, Any]) -> dict[str, Any] | None:
    for event in reversed(observation.get("__last_realized_events__") or []):
        if (
            isinstance(event, Mapping)
            and event.get("type") == "inventory_demand_realized"
            and event.get("origin") == "source_schedule"
            and not bool(event.get("hidden", False))
            and event.get("event_id")
            and event.get("evidence_ids")
        ):
            return dict(event)
    return None


class LogisticsSourceScheduleControlAgent(BaselineAgent):
    """Replenish only after observing an authoritative M5 demand event."""

    name = AGENT_NAME

    def __init__(self) -> None:
        self._reviewed_event_ids: set[str] = set()
        self._response_committed = False

    def reset(self, env: Any, scenario_config: dict[str, Any], seed: int) -> None:
        self._reviewed_event_ids.clear()
        self._response_committed = False
        self._reset_idem_seq()

    def act(
        self,
        observation: dict[str, Any],
        tool_specs: list[dict[str, Any]],
    ) -> Action:
        available = _tool_names(tool_specs)
        event = _visible_demand_event(observation)
        if (
            event is not None
            and not self._response_committed
            and "place_replenishment_order" in available
        ):
            event_id = str(event["event_id"])
            if event_id not in self._reviewed_event_ids:
                self._reviewed_event_ids.add(event_id)
                period = int(observation.get("period") or 0)
                lead_times = list(observation.get("lead_times") or [1])
                lead = max(1, int(lead_times[0] if lead_times else 1))
                target_period = period + lead
                demand = [
                    max(0, int(round(float(value))))
                    for value in observation.get("demand_forecast_units") or []
                ]
                start = int(observation.get("demand_forecast_start_period") or 0)
                index = (
                    target_period - start
                    if observation.get("demand_forecast_is_partial")
                    else target_period
                )
                capacity = [
                    max(0, int(round(float(value))))
                    for value in observation.get("supply_capacity") or []
                ]
                quantity = demand[index] if 0 <= index < len(demand) else 0
                if capacity:
                    quantity = min(quantity, capacity[0])
                if quantity > 0:
                    self._response_committed = True
                    call = ToolCall(
                        name="place_replenishment_order",
                        args={"quantity": quantity, "stage": 0},
                        idempotency_key=self._next_idem_key(
                            f"m5_source_response_{event_id}"
                        ),
                        consumes_evidence_ids=list(event["evidence_ids"]),
                    )
                    return Action(tool_calls=[call], dominant=call.name)
        if "wait" in available:
            wait = ToolCall(
                name="wait",
                idempotency_key=self._next_idem_key("m5_source_wait"),
            )
            return Action(tool_calls=[wait], dominant=wait.name)
        return Action(tool_calls=[], dominant="wait")


@contextmanager
def temporary_agent_registration() -> Iterator[None]:
    if AGENT_NAME in REGISTRY:
        raise RuntimeError(f"agent registry collision: {AGENT_NAME}")
    REGISTRY[AGENT_NAME] = LogisticsSourceScheduleControlAgent
    try:
        yield
    finally:
        REGISTRY.pop(AGENT_NAME, None)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_bindings(scenario: Mapping[str, Any]) -> dict[str, str]:
    contract = scenario.get("source_contract") or {}
    declared = [
        *(contract.get("runtime_input") or []),
        *(contract.get("derivation_input") or []),
        *(contract.get("metadata") or []),
        *(contract.get("license") or []),
        *((scenario.get("provenance") or {}).get("files") or []),
    ]
    bindings: dict[str, str] = {}
    for value in declared:
        if not isinstance(value, str) or not value:
            raise ValueError("source path must be a non-empty string")
        path = (REPO_ROOT / value).resolve()
        path.relative_to(REPO_ROOT.resolve())
        if not path.is_file():
            raise FileNotFoundError(value)
        bindings[value] = _sha256(path)
    if not bindings:
        raise ValueError("M5 source control requires locked source files")
    return dict(sorted(bindings.items()))


def _validate_parent_scenario(scenario: Mapping[str, Any]) -> None:
    if scenario.get("domain") != "logistics":
        raise ValueError("source-schedule positive control requires logistics domain")
    if scenario.get("backend_kind") != "orgym_invmgmt":
        raise ValueError("source-schedule positive control requires orgym_invmgmt")
    scenario_id = scenario.get("scenario_id")
    if not isinstance(scenario_id, str) or scenario.get("seed_id") != scenario_id:
        raise ValueError("source-schedule positive control requires bound scenario identity")
    horizon = scenario.get("horizon_ticks")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 2:
        raise ValueError("source-schedule positive control requires a valid horizon")
    source_contract = scenario.get("source_contract")
    if not isinstance(source_contract, Mapping):
        raise ValueError("source-schedule positive control requires source_contract")


def _terminal_integrity_valid(result: Mapping[str, Any]) -> bool:
    trajectory = result.get("trajectory_summary") or {}
    terminal = trajectory.get("terminal_integrity") or {}
    ground_truth = result.get("ground_truth_summary") or {}
    return bool(
        isinstance(terminal, Mapping)
        and terminal.get("release_ready") is True
        and terminal.get("unresolved_pending_actions") == {}
        and terminal.get("unanswered_interrupt_reasons") == []
        and result.get("status") != "error"
        and ground_truth.get("chose_fatal_option") is not True
    )


def _authoritative_source_record(
    result: Mapping[str, Any], record: Mapping[str, Any]
) -> bool:
    ground_truth = result.get("ground_truth_summary") or {}
    realized = ground_truth.get("realized_events") or []
    event_id = str(record.get("event_id") or "")
    source = next(
        (
            event
            for event in realized
            if isinstance(event, Mapping)
            and event.get("event_id") == event_id
            and event.get("type") == "inventory_demand_realized"
            and event.get("origin") == "source_schedule"
            and event.get("hidden") is not True
        ),
        None,
    )
    if source is None or record.get("causal_parent_event_id") != event_id:
        return False
    source_ids = set(source.get("evidence_ids") or [])
    trigger_ids = set(record.get("trigger_evidence_ids") or [])
    consumed_ids = set(record.get("action_consumes_evidence_ids") or [])
    return bool(source_ids.intersection(trigger_ids).intersection(consumed_ids))


def summarize_episode(result: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    counterfactual = result.get("counterfactual") or {}
    if not _full_uncapped_attribution(counterfactual):
        blockers.append("uncapped_attribution_incomplete")
    task = result.get("task_completion") or {}
    if task.get("completed") is not True:
        blockers.append(f"native_task_incomplete:{task.get('reason_code') or 'unknown'}")
    trajectory = result.get("trajectory_summary") or {}
    terminal = trajectory.get("terminal_integrity") or {}
    if not _terminal_integrity_valid(result):
        blockers.append("terminal_integrity_incomplete")
    profile = trajectory.get("operational_agency_profile") or {}
    if not operational_agency_profile_is_consistent(
        trajectory,
        counterfactual=counterfactual,
    ):
        blockers.append("operational_agency_profile_inconsistent")
    records = [
        {
            **dict(record),
            "event_origin": "source_schedule",
            "declared_perturbation": False,
        }
        for record in trajectory.get("event_response_records") or []
        if isinstance(record, Mapping)
        and str(record.get("event_id") or "").startswith(
            "inventory_demand_realized:"
        )
        and record.get("visibility") == "visible"
        and record.get("response_status") == "causal"
        and _authoritative_source_record(result, record)
        and set(record.get("trigger_evidence_ids") or []).intersection(
            record.get("action_consumes_evidence_ids") or []
        )
        and float(record.get("masked_action_group_delta") or 0.0) > 0.0
    ]
    if not records:
        blockers.append("positive_source_schedule_masked_chain_missing")
        raw_records = trajectory.get("event_response_records") or []
        if any(
            isinstance(record, Mapping)
            and str(record.get("event_id") or "").startswith(
                "inventory_demand_realized:"
            )
            for record in raw_records
        ):
            blockers.append("authoritative_source_event_binding_missing")
    if int(profile.get("causal_record_count") or 0) < 1:
        blockers.append("causal_profile_record_missing")
    return {
        "status": "held" if blockers else "passed",
        "blockers": blockers,
        "scenario_id": result.get("scenario_id"),
        "scenario_signature": result.get("scenario_signature"),
        "seed": result.get("seed"),
        "task_completion": task,
        "terminal_integrity": {
            **terminal,
            "terminal": True,
            "fatal": bool(
                (result.get("ground_truth_summary") or {}).get(
                    "chose_fatal_option", False
                )
            ),
            "fatal_error": (
                result.get("error") if result.get("status") == "error" else None
            ),
            "orphan_process_count": 0,
        },
        "counterfactual": counterfactual,
        "positive_source_schedule_records": records,
        "event_response_records": records,
        "operational_agency_valid_evidence_ids": list(
            trajectory.get("operational_agency_valid_evidence_ids") or []
        ),
        "operational_agency_profile": profile,
    }


def run_control(
    *,
    scenario_path: Path,
    output_path: Path,
    repeats: int,
) -> dict[str, Any]:
    if repeats != 2:
        raise ValueError("source-schedule positive control requires exactly two repeats")
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict):
        raise ValueError("scenario YAML must contain an object")
    _validate_parent_scenario(scenario)
    if scenario.get("perturbations"):
        raise ValueError("source-schedule positive control forbids perturbations")
    source_bindings = _source_bindings(scenario)
    declared_source_hashes = (
        ((scenario.get("backend_config") or {}).get("m5_source_lock") or {}).get(
            "file_sha256"
        )
        or {}
    )
    for source_path, expected in declared_source_hashes.items():
        actual = source_bindings.get(str(source_path))
        if actual != str(expected).removeprefix("sha256:"):
            raise ValueError(f"source hash mismatch: {source_path}")
    implementation_before = implementation_identity(REPO_ROOT)
    with temporary_agent_registration():
        summaries = [
            summarize_episode(
                run_one(
                    scenario,
                    AGENT_NAME,
                    per_action_attribution=True,
                    per_action_cap=None,
                    per_action_group_attribution=True,
                    per_action_group_cap=None,
                    within_tick_interaction=True,
                )
            )
            for _ in range(repeats)
        ]
    deterministic = summaries[0] == summaries[1]
    implementation_after = implementation_identity(REPO_ROOT)
    implementation_stable = (
        implementation_before["implementation_tree_sha256"]
        == implementation_after["implementation_tree_sha256"]
    )
    blockers = list(summaries[0]["blockers"])
    if not deterministic:
        blockers.append("deterministic_repeat_mismatch")
    if not implementation_stable:
        raise RuntimeError("implementation tree changed during source control run")
    status = "passed" if not blockers else "held"
    report = {
        "schema_version": "logistics-source-schedule-agency-positive-control-v1",
        "status": status,
        "diagnostic_only": True,
        "release_admission": False,
        "source_independence_credit": False,
        "event_origin_contract": "source_schedule_only",
        "blockers": blockers,
        "implementation_identity": implementation_before,
        "implementation_stable": implementation_stable,
        "implementation_stability": {
            "before": implementation_before,
            "after": implementation_after,
            "passed": implementation_stable,
        },
        "attribution_contract": {
            "mode": "complete_uncapped",
            "per_action_cap": None,
            "per_action_group_cap": None,
        },
        "scenario_binding": {
            "path": scenario_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(scenario_path),
        },
        "source_file_bindings": source_bindings,
        "determinism": {"repeats": repeats, "passed": deterministic},
        "result": summaries[0],
    }
    report["domain"] = "logistics"
    report["control"] = {
        "status": status,
        "scenario_binding": report["scenario_binding"],
        "source_file_bindings": source_bindings,
        "determinism": report["determinism"],
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
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    report = run_control(
        scenario_path=args.scenario.resolve(),
        output_path=args.output.resolve(),
        repeats=args.repeats,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "blockers": report["blockers"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
