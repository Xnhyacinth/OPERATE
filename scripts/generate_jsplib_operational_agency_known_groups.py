#!/usr/bin/env python3
"""Generate one source-bound JSPLIB operational-agency known-group experiment.

All cells execute through ``runner.episode.run_one`` with complete, uncapped
per-action and action-group counterfactual attribution.  The generator only
adds experimental cell metadata after the runtime result passes authoritative
profile validation; it never writes or adjusts an agency score.  A comparison
that does not arise naturally is retained as a fail-closed held artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines import REGISTRY, BaselineAgent  # noqa: E402
from core import Action, ToolCall  # noqa: E402
from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import portable_repo_path  # noqa: E402
from evaluation.operational_agency import (  # noqa: E402
    operational_agency_profile_is_consistent,
)
from runner.episode import run_one  # noqa: E402
from scripts.run_operational_agency_known_groups_calibration import (  # noqa: E402
    _full_uncapped_attribution,
    build_known_groups_report,
)
from scripts.run_operational_agency_runtime_positive_control import (  # noqa: E402
    DEFAULT_BASE,
    build_control_scenario,
    validate_parent_scenario,
)

DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "reports/protocol21_operational_agency_jsplib_known_groups_current"
)

ADAPTIVE_PLAN_AGENT = "diagnostic_jsplib_adaptive_plan"
REACTIVE_AGENT = "diagnostic_jsplib_reactive"
OPEN_LOOP_AGENT = "diagnostic_jsplib_open_loop"
PARTIAL_AGENT = "diagnostic_jsplib_partial_adaptive_plan"
_MAX_DISPATCH_BATCH = 4


@dataclass(frozen=True)
class CellSpec:
    """One executable condition and its explicit known-group binding."""

    agent_name: str
    cells: tuple[str, ...]
    pair_ids: dict[str, str]
    positive_control: bool
    scenario: dict[str, Any]


def _available_tool_names(tool_specs: list[dict[str, Any]]) -> set[str]:
    return {
        name
        for spec in tool_specs
        if isinstance(spec, dict)
        for name in (
            str(spec.get("name") or ""),
            str((spec.get("function") or {}).get("name") or "")
            if isinstance(spec.get("function"), dict)
            else "",
        )
        if name
    }


def _visible_machine_events(
    observation: Mapping[str, Any],
) -> dict[int, list[str]]:
    """Return only event IDs actually delivered in the agent-facing stream."""
    visible: dict[int, list[str]] = {}
    for event in observation.get("__last_realized_events__") or []:
        if not isinstance(event, Mapping):
            continue
        if str(event.get("type") or event.get("kind") or "") != "machine_breakdown":
            continue
        try:
            machine_id = int(event.get("machine_id"))
        except (TypeError, ValueError):
            continue
        evidence_ids = [
            str(value)
            for value in event.get("evidence_ids") or []
            if isinstance(value, str) and value
        ]
        if evidence_ids:
            visible[machine_id] = list(dict.fromkeys(evidence_ids))
    return visible


def _dispatch_call(
    observation: Mapping[str, Any],
    *,
    policy: str,
    idempotency_key: str,
) -> ToolCall | None:
    ready = observation.get("ready_operations") or {}
    if not isinstance(ready, Mapping):
        return None

    def key(item: tuple[Any, Any]) -> tuple[Any, ...]:
        job_id, operation = item
        row = operation if isinstance(operation, Mapping) else {}
        if policy == "open_loop":
            return (str(job_id),)
        return (
            -float(row.get("urgency") or 0.0),
            int(row.get("duration") or 0),
            int(row.get("machine_id") or 0),
            str(job_id),
        )

    operations = [
        {
            "job_id": str(job_id),
            "operation_index": int(operation.get("operation_index") or 0),
        }
        for job_id, operation in sorted(ready.items(), key=key)[:_MAX_DISPATCH_BATCH]
        if isinstance(operation, Mapping)
    ]
    if not operations:
        return None
    return ToolCall(
        name="dispatch_ready_operations",
        args={"operations": operations},
        idempotency_key=idempotency_key,
    )


class _ObservedDispatchAgent(BaselineAgent):
    """Observation-only JSPLIB dispatcher with no outage-recovery lever."""

    name = "diagnostic_jsplib_observed_dispatch"

    def __init__(self, *, policy: str = "reactive") -> None:
        if policy not in {"reactive", "open_loop"}:
            raise ValueError(f"unsupported observed-dispatch policy: {policy}")
        self._policy = policy
        self._tick = 0

    def reset(self, env: Any, scenario_config: dict[str, Any], seed: int) -> None:
        self._tick = 0
        self._reset_idem_seq()

    def act(
        self,
        observation: dict[str, Any],
        tool_specs: list[dict[str, Any]],
    ) -> Action:
        self._tick += 1
        available = _available_tool_names(tool_specs)
        dispatch = (
            _dispatch_call(
                observation,
                policy=self._policy,
                idempotency_key=self._next_idem_key(
                    f"jsplib_{self._policy}_dispatch"
                ),
            )
            if "dispatch_ready_operations" in available
            else None
        )
        if dispatch is not None:
            return Action(tool_calls=[dispatch], dominant=dispatch.name)
        wait = (
            ToolCall(
                name="wait",
                idempotency_key=self._next_idem_key(f"jsplib_{self._policy}_wait"),
            )
            if "wait" in available
            else None
        )
        return Action(
            tool_calls=[wait] if wait is not None else [],
            dominant="wait",
        )


class _ReactiveAgent(_ObservedDispatchAgent):
    name = REACTIVE_AGENT

    def __init__(self) -> None:
        super().__init__(policy="reactive")


class _OpenLoopAgent(_ObservedDispatchAgent):
    name = OPEN_LOOP_AGENT

    def __init__(self) -> None:
        super().__init__(policy="open_loop")


class _AdaptivePlanAgent(_ObservedDispatchAgent):
    """Dispatch, plan, and recover only from authoritative visible events."""

    name = ADAPTIVE_PLAN_AGENT

    def __init__(self) -> None:
        super().__init__(policy="reactive")
        self._visible_event_evidence: dict[int, list[str]] = {}
        self._plan_started = False
        self._revised_machines: set[int] = set()

    def reset(self, env: Any, scenario_config: dict[str, Any], seed: int) -> None:
        super().reset(env, scenario_config, seed)
        self._visible_event_evidence.clear()
        self._plan_started = False
        self._revised_machines.clear()

    def act(
        self,
        observation: dict[str, Any],
        tool_specs: list[dict[str, Any]],
    ) -> Action:
        self._tick += 1
        available = _available_tool_names(tool_specs)
        self._visible_event_evidence.update(_visible_machine_events(observation))
        active = observation.get("active_machine_disruptions") or {}
        active_machine_ids = {
            int(machine_id)
            for machine_id, until_tick in active.items()
            if int(until_tick) > int(observation.get("tick") or 0)
        } if isinstance(active, Mapping) else set()
        visible_active = sorted(
            active_machine_ids.intersection(self._visible_event_evidence)
        )
        calls: list[ToolCall] = []

        if "commit_to_plan" in available and not self._plan_started:
            calls.append(
                ToolCall(
                    name="commit_to_plan",
                    args={
                        "plan_id": "jsplib-standing-plan-v1",
                        "rationale": (
                            "Dispatch the locked JSPLIB precedence graph and review "
                            "the plan when an observed native disruption arrives."
                        ),
                        "predicted_events": [],
                    },
                    idempotency_key=self._next_idem_key("jsplib_plan_v1"),
                )
            )
            self._plan_started = True

        for machine_id in visible_active:
            evidence_ids = self._visible_event_evidence[machine_id]
            if (
                "commit_to_plan" in available
                and machine_id not in self._revised_machines
            ):
                calls.append(
                    ToolCall(
                        name="commit_to_plan",
                        args={
                            "plan_id": f"jsplib-recovery-plan-machine-{machine_id}",
                            "replaces_plan_id": "jsplib-standing-plan-v1",
                            "revision_reason": "observed_native_machine_breakdown",
                            "trigger_evidence_ids": evidence_ids,
                            "rationale": (
                                "Replace the standing dispatch plan after the "
                                "source-bound machine outage became observable."
                            ),
                            "predicted_events": [],
                        },
                        idempotency_key=self._next_idem_key(
                            f"jsplib_plan_repair_{machine_id}"
                        ),
                        consumes_evidence_ids=evidence_ids,
                    )
                )
                self._revised_machines.add(machine_id)
            if "repair_machine" in available:
                calls.append(
                    ToolCall(
                        name="repair_machine",
                        args={"machine_id": machine_id},
                        idempotency_key=self._next_idem_key(
                            f"jsplib_repair_{machine_id}"
                        ),
                        consumes_evidence_ids=evidence_ids,
                    )
                )

        dispatch = (
            _dispatch_call(
                observation,
                policy="reactive",
                idempotency_key=self._next_idem_key("jsplib_adaptive_dispatch"),
            )
            if "dispatch_ready_operations" in available
            else None
        )
        if dispatch is not None:
            calls.append(dispatch)
        if not calls and "wait" in available:
            calls.append(
                ToolCall(
                    name="wait",
                    idempotency_key=self._next_idem_key("jsplib_adaptive_wait"),
                )
            )
        return Action(tool_calls=calls, dominant=calls[0].name if calls else "wait")


class _PartialAdaptivePlanAgent(_AdaptivePlanAgent):
    name = PARTIAL_AGENT


_DIAGNOSTIC_AGENTS: dict[str, type[BaselineAgent]] = {
    ADAPTIVE_PLAN_AGENT: _AdaptivePlanAgent,
    REACTIVE_AGENT: _ReactiveAgent,
    OPEN_LOOP_AGENT: _OpenLoopAgent,
    PARTIAL_AGENT: _PartialAdaptivePlanAgent,
}


@contextmanager
def temporary_agent_registrations() -> Iterator[None]:
    """Register diagnostic-only policies for ``run_one`` and restore exactly."""
    collisions = sorted(set(_DIAGNOSTIC_AGENTS).intersection(REGISTRY))
    if collisions:
        raise RuntimeError(f"agent registry collision: {collisions}")
    REGISTRY.update(_DIAGNOSTIC_AGENTS)
    try:
        yield
    finally:
        for name in _DIAGNOSTIC_AGENTS:
            REGISTRY.pop(name, None)


def build_cell_specs(base: dict[str, Any]) -> list[CellSpec]:
    """Build six detached conditions over one locked physical source."""
    seed = int(base.get("seed", 44))
    pair_ids = {
        "adaptive_gt_reactive": f"jsplib-swv06-adaptive-reactive-s{seed}",
        "adaptive_plan_gt_open_loop": f"jsplib-swv06-plan-open-loop-s{seed}",
        "full_observation_gte_partial": f"jsplib-swv06-full-partial-s{seed}",
    }
    definitions = (
        (
            "adaptive-plan-full",
            ADAPTIVE_PLAN_AGENT,
            ("adaptive", "adaptive_plan", "full_observation"),
            dict(pair_ids),
            True,
            False,
        ),
        (
            "reactive",
            REACTIVE_AGENT,
            ("reactive",),
            {"adaptive_gt_reactive": pair_ids["adaptive_gt_reactive"]},
            False,
            False,
        ),
        (
            "open-loop",
            OPEN_LOOP_AGENT,
            ("open_loop",),
            {"adaptive_plan_gt_open_loop": pair_ids["adaptive_plan_gt_open_loop"]},
            False,
            False,
        ),
        (
            "partial-observation",
            PARTIAL_AGENT,
            ("partial_observation",),
            {"full_observation_gte_partial": pair_ids["full_observation_gte_partial"]},
            False,
            True,
        ),
        ("wait", "wait_only", ("wait_only",), {}, False, False),
        ("random", "random", ("random",), {}, False, False),
    )
    specs: list[CellSpec] = []
    for condition, agent_name, cells, cell_pairs, positive, hidden in definitions:
        scenario = build_control_scenario(base)
        scenario_id = f"diagnostic/agency_known_groups/jsplib_swv06/{condition}"
        scenario["seed_id"] = scenario_id
        scenario["scenario_id"] = scenario_id
        scenario.pop("scenario_signature", None)
        scenario["perturbations"][0]["hidden"] = hidden
        scenario["backend_config"]["agency_positive_control"]["condition"] = condition
        specs.append(
            CellSpec(
                agent_name=agent_name,
                cells=cells,
                pair_ids=cell_pairs,
                positive_control=positive,
                scenario=scenario,
            )
        )
    return specs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_file_bindings(scenario: Mapping[str, Any]) -> dict[str, str]:
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
        raise ValueError("known-groups generator requires a runtime source binding")
    return bindings


def _validate_runtime_episode(
    result: Mapping[str, Any],
    *,
    positive_control: bool,
) -> list[str]:
    trajectory = result.get("trajectory_summary")
    counterfactual = result.get("counterfactual")
    if not isinstance(trajectory, Mapping) or not isinstance(counterfactual, Mapping):
        raise ValueError("runtime episode lacks trajectory or counterfactual output")
    if not _full_uncapped_attribution(counterfactual):
        raise ValueError("runtime episode lacks complete uncapped attribution")
    if not operational_agency_profile_is_consistent(
        trajectory,
        counterfactual=counterfactual,
    ):
        raise ValueError("runtime episode lacks authoritative agency evidence")
    terminal = trajectory.get("terminal_integrity") or {}
    ground_truth = result.get("ground_truth_summary") or {}
    if not (
        isinstance(terminal, Mapping)
        and terminal.get("release_ready") is True
        and terminal.get("unresolved_pending_actions") == {}
        and terminal.get("unanswered_interrupt_reasons") == []
        and result.get("status") != "error"
        and ground_truth.get("chose_fatal_option") is not True
    ):
        raise ValueError("runtime episode terminal integrity is incomplete")
    scientific_blockers: list[str] = []
    if positive_control:
        profile = trajectory["operational_agency_profile"]
        task = result.get("task_completion") or {}
        semantic_coverage = trajectory.get("tool_semantic_coverage") or {}
        registered_tools = set(semantic_coverage.get("registered_tool_names") or [])
        tool_histogram = trajectory.get("tool_histogram") or {}
        if "commit_to_plan" not in registered_tools:
            scientific_blockers.append("adaptive_plan_tool_not_registered")
        elif int(tool_histogram.get("commit_to_plan") or 0) < 1:
            scientific_blockers.append("adaptive_plan_tool_not_executed")
        if int(tool_histogram.get("repair_machine") or 0) < 1:
            scientific_blockers.append("adaptive_recovery_control_not_executed")
        if int(profile.get("causal_record_count") or 0) < 1:
            scientific_blockers.append("adaptive_positive_control_no_causal_record")
        records = trajectory.get("event_response_records") or []
        realized = ground_truth.get("realized_events") or []
        authoritative = False
        for record in records:
            if not isinstance(record, Mapping):
                continue
            event_id = str(record.get("event_id") or "")
            source = next(
                (
                    event
                    for event in realized
                    if isinstance(event, Mapping)
                    and event.get("event_id") == event_id
                    and event.get("type") == "machine_breakdown"
                    and event.get("origin") == "declared_perturbation"
                    and event.get("hidden") is not True
                ),
                None,
            )
            source_ids = set((source or {}).get("evidence_ids") or [])
            if (
                source is not None
                and record.get("causal_parent_event_id") == event_id
                and source_ids.intersection(record.get("trigger_evidence_ids") or [])
                .intersection(record.get("action_consumes_evidence_ids") or [])
                and record.get("backend_effect_evidence_ids")
            ):
                authoritative = True
                break
        if not authoritative:
            scientific_blockers.append(
                "adaptive_positive_control_event_not_authoritative"
            )
        if task.get("completed") is not True:
            scientific_blockers.append("adaptive_positive_control_task_incomplete")
    return scientific_blockers


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_known_groups(
    *,
    base_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute, validate, bind, and recompute the one-source experiment."""
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("base scenario YAML must contain an object")
    validate_parent_scenario(base)
    cell_specs = build_cell_specs(base)
    source_bindings = _source_file_bindings(cell_specs[0].scenario)
    physical_source_lock = {
        "contract": "runtime_input_sha256_set_v1",
        "backend_kind": "jsplib_job_shop",
        "instance_name": str(
            (cell_specs[0].scenario.get("backend_config") or {}).get(
                "instance_name", "swv06"
            )
        ),
        "source_file_bindings": source_bindings,
    }
    start_identity = implementation_identity(REPO_ROOT)
    episodes: list[dict[str, Any]] = []
    runtime_cell_blockers: dict[str, list[str]] = {}
    with temporary_agent_registrations():
        for cell in cell_specs:
            result = run_one(
                cell.scenario,
                cell.agent_name,
                seed_override=int(cell.scenario.get("seed", 44)),
                per_action_attribution=True,
                per_action_cap=None,
                per_action_group_attribution=True,
                per_action_group_cap=None,
                within_tick_interaction=True,
            )
            cell_blockers = _validate_runtime_episode(
                result,
                positive_control=cell.positive_control,
            ) or []
            if cell_blockers:
                runtime_cell_blockers[str(result.get("scenario_id") or "")] = (
                    cell_blockers
                )
            result["status"] = (
                "held_scientific" if cell_blockers else "ok"
            )
            result["operational_agency_known_groups"] = {
                "cells": list(cell.cells),
                "pair_ids": dict(cell.pair_ids),
                "positive_control": cell.positive_control,
            }
            episodes.append(result)

    end_identity = implementation_identity(REPO_ROOT)
    if end_identity["implementation_tree_sha256"] != start_identity[
        "implementation_tree_sha256"
    ]:
        raise RuntimeError("implementation tree changed during known-groups execution")

    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.jsonl"
    episodes_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in episodes),
        encoding="utf-8",
    )
    slice_path = output_dir / "slice.json"
    slice_payload = {
        "schema_version": "operational-agency-known-groups-jsplib-slice-v1",
        "diagnostic_only": True,
        "source_independence_credit": False,
        "source_file_bindings": source_bindings,
        "scenarios": [
            {
                "scenario_id": result["scenario_id"],
                "scenario_signature": result["scenario_signature"],
                "domain": "logistics",
                "case_ledger": {"physical_source_lock": physical_source_lock},
            }
            for result in episodes
        ],
    }
    _write_json(slice_path, slice_payload)
    artifact = build_known_groups_report(
        repo_root=REPO_ROOT,
        slice_path=slice_path,
        episode_paths=[episodes_path],
        implementation_tree_sha256=start_identity["implementation_tree_sha256"],
        required_domains={"logistics"},
    )
    artifact_path = output_dir / "known_groups.json"
    _write_json(artifact_path, artifact)
    run_report = {
        "schema_version": "operational-agency-known-groups-jsplib-generator-v1",
        "status": artifact["status"],
        "diagnostic_only": True,
        "release_admission": False,
        "source_independence_credit": False,
        "implementation_identity": start_identity,
        "base_scenario_binding": {
            "path": base_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(base_path),
        },
        "source_file_bindings": source_bindings,
        "physical_source_lock": physical_source_lock,
        "n_runtime_episodes": len(episodes),
        "runtime_contract": {
            "runner": "runner.episode.run_one",
            "per_action_attribution": True,
            "per_action_cap": None,
            "per_action_group_attribution": True,
            "per_action_group_cap": None,
            "within_tick_interaction": True,
            "score_mutation": False,
        },
        "output_bindings": {
            "episodes": {
                "path": portable_repo_path(episodes_path, repo_root=REPO_ROOT),
                "sha256": _sha256(episodes_path),
            },
            "slice": {
                "path": portable_repo_path(slice_path, repo_root=REPO_ROOT),
                "sha256": _sha256(slice_path),
            },
            "known_groups": {
                "path": portable_repo_path(artifact_path, repo_root=REPO_ROOT),
                "sha256": _sha256(artifact_path),
            },
        },
        "comparison_status": {
            name: row["status"]
            for name, row in artifact["comparisons"].items()
        },
        "runtime_cell_blockers": runtime_cell_blockers,
        "blockers": artifact["blockers"],
    }
    _write_json(output_dir / "run_report.json", run_report)
    return run_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-scenario", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    report = generate_known_groups(
        base_path=args.base_scenario.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_runtime_episodes": report["n_runtime_episodes"],
                "comparison_status": report["comparison_status"],
                "blocker_codes": sorted(
                    {row["code"] for row in report["blockers"]}
                ),
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
