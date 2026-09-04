#!/usr/bin/env python3
"""Run fail-closed Power and Microgrid operational-agency controls."""

from __future__ import annotations

import argparse
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

CONTROL_AGENT_NAME = "domain_agency_positive_control"
DEFAULT_POWER_SCENARIO = REPO_ROOT / (
    "scenarios/operate_v0_58_0/power_grid/"
    "opendss_fresh_feeders_solar_ramp/deep_planning/basic/"
    "opendss_ieee123_solar_ramp_s42.yaml"
)
DEFAULT_MICROGRID_SCENARIO = REPO_ROOT / (
    "scenarios/operate_v0_58_0/microgrid/"
    "microgrid_economic_dispatch_24h/deep_planning/high/"
    "native_state_loss_chicago_high_s61.yaml"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "reports/operate_v0_58_0/agency/power_microgrid.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bind_visible_source_evidence(
    action: Action, observation: dict[str, Any]
) -> Action:
    """Explicitly bind controls to the latest agent-visible source event."""
    visible_source_events = [
        event
        for event in observation.get("__last_realized_events__") or []
        if isinstance(event, dict)
        and event.get("origin") == "source_schedule"
        and event.get("visibility") != "hidden"
        and event.get("evidence_ids")
    ]
    if not visible_source_events:
        return action
    evidence_ids = [
        str(evidence_id)
        for evidence_id in visible_source_events[-1].get("evidence_ids") or []
        if evidence_id
    ]
    for call in action.tool_calls:
        if call.name in {"wait", "noop"}:
            continue
        call.consumes_evidence_ids = list(
            dict.fromkeys([*(call.consumes_evidence_ids or []), *evidence_ids])
        )
    return action


class SourceBoundOracleAgent(OracleOfflineAgent):
    """Oracle policy with explicit, observation-derived evidence edges."""

    name = CONTROL_AGENT_NAME

    def act(
        self,
        observation: dict[str, Any],
        tool_specs: list[dict[str, Any]],
    ) -> Action:
        return _bind_visible_source_evidence(
            super().act(observation, tool_specs), observation
        )


@contextmanager
def _registered_control_agent() -> Iterator[None]:
    previous = REGISTRY.get(CONTROL_AGENT_NAME)
    REGISTRY[CONTROL_AGENT_NAME] = SourceBoundOracleAgent
    try:
        yield
    finally:
        if previous is None:
            REGISTRY.pop(CONTROL_AGENT_NAME, None)
        else:
            REGISTRY[CONTROL_AGENT_NAME] = previous


def _attribution_blockers(counterfactual: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for prefix in ("per_action", "per_action_group"):
        status = str(counterfactual.get(f"{prefix}_status") or "")
        expected = int(counterfactual.get(f"{prefix}_expected") or 0)
        attempted = int(counterfactual.get(f"{prefix}_attempted") or 0)
        completed = int(counterfactual.get(f"{prefix}_completed") or 0)
        failures = list(counterfactual.get(f"{prefix}_failures") or [])
        if status != "complete":
            blockers.append(f"{prefix}_status:{status or 'missing'}")
        if not expected == attempted == completed:
            blockers.append(
                f"{prefix}_coverage:{expected}/{attempted}/{completed}"
            )
        if failures:
            blockers.append(f"{prefix}_replay_failures:{len(failures)}")
    if int(counterfactual.get("per_action_expected") or 0) <= 0:
        blockers.append("per_action_attribution_empty")
    return blockers


def summarize_domain_repeat(result: dict[str, Any]) -> dict[str, Any]:
    """Validate one native run without manufacturing a passing control."""
    task = dict(result.get("task_completion") or {})
    counterfactual = dict(result.get("counterfactual") or {})
    trajectory = dict(result.get("trajectory_summary") or {})
    events_by_id = {
        str(event.get("event_id")): dict(event)
        for event in trajectory.get("world_evolution_records") or []
        if isinstance(event, dict) and event.get("event_id")
    }
    events_by_id.update(
        {
            str(event.get("event_id")): dict(event)
            for event in (
                (result.get("ground_truth_summary") or {}).get(
                    "realized_events"
                )
                or []
            )
            if isinstance(event, dict) and event.get("event_id")
        }
    )
    records = [
        {
            **dict(record),
            "event_origin": (
                events_by_id.get(str(record.get("event_id") or ""), {}).get(
                    "origin"
                )
            ),
            "declared_perturbation": bool(
                events_by_id.get(
                    str(record.get("event_id") or ""), {}
                ).get("declared_perturbation", False)
            )
            or (
                events_by_id.get(str(record.get("event_id") or ""), {}).get(
                    "origin"
                )
                == "declared_perturbation"
            ),
        }
        for record in trajectory.get("event_response_records") or []
        if isinstance(record, dict)
    ]
    positive_records = [
        record
        for record in records
        if record.get("response_status") == "causal"
        and float(record.get("masked_action_group_delta") or 0.0) > 0.0
        and set(record.get("trigger_evidence_ids") or []).intersection(
            record.get("action_consumes_evidence_ids") or []
        )
        and record.get("backend_effect_evidence_ids")
    ]
    profile = dict(trajectory.get("operational_agency_profile") or {})
    blockers = _attribution_blockers(counterfactual)
    if task.get("completed") is not True:
        blockers.append(
            f"native_task_incomplete:{task.get('reason_code') or 'unknown'}"
        )
    if not positive_records:
        blockers.append("positive_masked_replay_missing")
    if not all(
        profile.get(field) is True
        for field in (
            "runtime_binding_verified",
            "runtime_evidence_binding_verified",
            "masked_replay_binding_verified",
        )
    ):
        blockers.append("operational_agency_runtime_binding_incomplete")
    if int(profile.get("causal_record_count") or 0) <= 0:
        blockers.append("operational_agency_causal_record_missing")
    attribution = {
        key: counterfactual.get(key)
        for key in (
            "per_action_status",
            "per_action_expected",
            "per_action_attempted",
            "per_action_completed",
            "per_action_failures",
            "per_action_capped",
            "per_action_group_status",
            "per_action_group_expected",
            "per_action_group_attempted",
            "per_action_group_completed",
            "per_action_group_failures",
            "per_action",
            "per_action_groups",
        )
    }
    terminal = dict(trajectory.get("terminal_integrity") or {})
    terminal.update(
        {
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
        }
    )
    return {
        "status": "passed" if not blockers else "held",
        "blockers": sorted(set(blockers)),
        "scenario_id": result.get("scenario_id"),
        "scenario_signature": result.get("scenario_signature"),
        "agent_name": result.get("agent_name"),
        "seed": result.get("seed"),
        "task_completion": task,
        "attribution": attribution,
        "counterfactual": attribution,
        "positive_event_response": (
            max(
                positive_records,
                key=lambda row: float(
                    row.get("masked_action_group_delta") or 0.0
                ),
            )
            if positive_records
            else None
        ),
        "event_response_records": records,
        "operational_agency_valid_evidence_ids": trajectory.get(
            "operational_agency_valid_evidence_ids"
        ),
        "operational_agency_profile": profile,
        "terminal_integrity": terminal,
    }


def _source_file_bindings(scenario: dict[str, Any]) -> dict[str, str]:
    candidates: list[str] = []
    source_contract = scenario.get("source_contract") or {}
    for key in ("runtime_input", "derivation_input"):
        candidates.extend(source_contract.get(key) or [])
    if not candidates:
        candidates.extend((scenario.get("provenance") or {}).get("files") or [])
    case_file = (scenario.get("backend_config") or {}).get("case_file")
    if case_file:
        candidates.append(str(case_file))
    declared_hashes = source_contract.get("file_sha256s") or {}
    bindings: dict[str, str] = {}
    for value in dict.fromkeys(candidates):
        value = str(value)
        if "://" in value:
            digest = str(declared_hashes.get(value) or "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"virtual source asset lacks SHA-256 lock: {value}")
            bindings[value] = digest
            continue
        path = (REPO_ROOT / str(value)).resolve()
        path.relative_to(REPO_ROOT.resolve())
        if not path.is_file():
            raise FileNotFoundError(value)
        ref = path.relative_to(REPO_ROOT).as_posix()
        bindings[ref] = _sha256(path)
    if not bindings:
        raise ValueError("domain control requires source-file bindings")
    return bindings


def _run_repeat(scenario: dict[str, Any]) -> dict[str, Any]:
    return summarize_domain_repeat(
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


def _run_domain(path: Path, *, repeats: int) -> dict[str, Any]:
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict):
        raise ValueError(f"scenario must be a mapping: {path}")
    summaries = [_run_repeat(scenario) for _ in range(repeats)]
    deterministic = all(summary == summaries[0] for summary in summaries[1:])
    blockers = sorted(
        {
            blocker
            for summary in summaries
            for blocker in summary["blockers"]
        }
    )
    if not deterministic:
        blockers.append("nondeterministic_repeats")
    return {
        "status": "passed" if not blockers else "held",
        "blockers": blockers,
        "scenario_binding": {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(path),
        },
        "source_file_bindings": _source_file_bindings(scenario),
        "determinism": {"repeats": repeats, "passed": deterministic},
        "result": summaries[0],
    }


def run_controls(
    *,
    power_scenario: Path,
    microgrid_scenario: Path,
    output: Path,
    repeats: int = 2,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("per-domain controls require at least two repeats")
    implementation_before = implementation_identity(REPO_ROOT)
    with _registered_control_agent():
        domains = {
            "Power Grid": _run_domain(power_scenario, repeats=repeats),
            "Microgrid": _run_domain(microgrid_scenario, repeats=repeats),
        }
    implementation_after = implementation_identity(REPO_ROOT)
    implementation_stable = (
        implementation_before["implementation_tree_sha256"]
        == implementation_after["implementation_tree_sha256"]
    )
    if not implementation_stable:
        for row in domains.values():
            row["status"] = "held"
            row["result"]["status"] = "held"
            row["result"].setdefault("blockers", []).append(
                "implementation_tree_changed_during_run"
            )
    report = {
        "schema_version": "power-microgrid-agency-positive-controls-v1",
        "status": (
            "passed"
            if all(row["status"] == "passed" for row in domains.values())
            else "held"
        ),
        "diagnostic_only": True,
        "release_admission": False,
        "implementation_identity": implementation_after,
        "implementation_stability": {
            "before": implementation_before,
            "after": implementation_after,
            "passed": implementation_stable,
        },
        "run_contract": {
            "repeats_per_domain": repeats,
            "per_action_attribution": True,
            "per_action_cap": None,
            "per_action_group_attribution": True,
            "per_action_group_cap": None,
            "evidence_binding": "explicit_agent_visible_source_evidence",
        },
        "domains": domains,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-scenario", type=Path, default=DEFAULT_POWER_SCENARIO)
    parser.add_argument(
        "--microgrid-scenario", type=Path, default=DEFAULT_MICROGRID_SCENARIO
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)
    report = run_controls(
        power_scenario=args.power_scenario.resolve(),
        microgrid_scenario=args.microgrid_scenario.resolve(),
        output=args.output.resolve(),
        repeats=args.repeats,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "domains": {
                    domain: row["status"]
                    for domain, row in report["domains"].items()
                },
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
