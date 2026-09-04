#!/usr/bin/env python3
"""Run the active OPERATE natural-source Microgrid agency positive control."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from scripts.run_power_microgrid_agency_positive_controls import (  # noqa: E402
    _registered_control_agent,
    _run_domain,
)

SCENARIO_ID = (
    "microgrid/microgrid_economic_dispatch_24h/deep_planning/high/"
    "native_state_loss_chicago_high_s61"
)
DEFAULT_SCENARIO = REPO_ROOT / (
    "scenarios/operate_v0_58_0/microgrid/"
    "microgrid_economic_dispatch_24h/deep_planning/high/"
    "native_state_loss_chicago_high_s61.yaml"
)
DEFAULT_CORE_SELECTION = REPO_ROOT / (
    "release/operate_v0_58_0_candidate/operate_v058_formal/"
    "refined_core_selection_protocol2_v21.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "reports/operate_v0_58_0/agency/microgrid_natural.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coverage_blockers(attribution: dict[str, Any], prefix: str) -> list[str]:
    blockers: list[str] = []
    status = str(attribution.get(f"{prefix}_status") or "missing")
    expected = int(attribution.get(f"{prefix}_expected") or 0)
    attempted = int(attribution.get(f"{prefix}_attempted") or 0)
    completed = int(attribution.get(f"{prefix}_completed") or 0)
    if status != "complete":
        blockers.append(f"{prefix}_status:{status}")
    if not expected == attempted == completed:
        blockers.append(
            f"{prefix}_coverage:{expected}/{attempted}/{completed}"
        )
    if attribution.get(f"{prefix}_failures"):
        blockers.append(f"{prefix}_failures")
    return blockers


def control_blockers(domain: dict[str, Any]) -> list[str]:
    """Apply the natural-source positive-control contract fail-closed."""
    blockers: list[str] = []
    if domain.get("status") != "passed":
        blockers.extend(str(value) for value in domain.get("blockers") or [])
    determinism = dict(domain.get("determinism") or {})
    if int(determinism.get("repeats") or 0) < 2:
        blockers.append("fewer_than_two_repeats")
    if determinism.get("passed") is not True:
        blockers.append("nondeterministic_repeats")
    if not domain.get("source_file_bindings"):
        blockers.append("source_file_bindings_missing")

    result = dict(domain.get("result") or {})
    if (result.get("task_completion") or {}).get("completed") is not True:
        blockers.append("native_task_incomplete")
    positive = dict(result.get("positive_event_response") or {})
    if (
        positive.get("event_origin") != "source_schedule"
        or positive.get("declared_perturbation") is not False
        or float(positive.get("masked_action_group_delta") or 0.0) <= 0.0
    ):
        blockers.append("natural_source_schedule_positive_control_missing")
    elif (
        not set(positive.get("trigger_evidence_ids") or []).intersection(
            positive.get("action_consumes_evidence_ids") or []
        )
        or not positive.get("backend_effect_evidence_ids")
    ):
        blockers.append("source_action_effect_evidence_chain_incomplete")

    attribution = dict(result.get("attribution") or {})
    blockers.extend(_coverage_blockers(attribution, "per_action"))
    blockers.extend(_coverage_blockers(attribution, "per_action_group"))
    if attribution.get("per_action_capped") is not False:
        blockers.append("per_action_attribution_capped")

    terminal = dict(result.get("terminal_integrity") or {})
    if terminal.get("terminal") is not True or terminal.get("release_ready") is not True:
        blockers.append("terminal_incomplete")
    if terminal.get("unresolved_pending_actions"):
        blockers.append("terminal_pending_actions")
    if terminal.get("fatal") or terminal.get("fatal_error"):
        blockers.append("terminal_fatal")
    if int(terminal.get("orphan_process_count") or 0) != 0:
        blockers.append("terminal_orphan_process")
    return sorted(set(blockers))


def _core_selection_binding(
    *, scenario_path: Path, core_selection_path: Path
) -> dict[str, Any]:
    selection = json.loads(core_selection_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in selection.get("scenarios") or []
        if isinstance(row, dict) and row.get("scenario_id") == SCENARIO_ID
    ]
    if len(rows) != 1 or rows[0].get("status") != "core_locked":
        raise ValueError("Chicago Microgrid row is not uniquely core_locked")
    row = rows[0]
    if (REPO_ROOT / str(row.get("path") or "")).resolve() != scenario_path.resolve():
        raise ValueError("Core scenario path drift")
    return {
        "path": core_selection_path.relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha256(core_selection_path),
        "scenario_id": SCENARIO_ID,
        "status": row["status"],
        "scenario_signature": row.get("scenario_signature"),
        "source_denominator_key": row.get("source_denominator_key"),
    }


def run_control(
    *,
    output: Path,
    repeats: int = 2,
    scenario_path: Path = DEFAULT_SCENARIO,
    core_selection_path: Path = DEFAULT_CORE_SELECTION,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("Microgrid positive control requires at least two repeats")
    scenario_path = scenario_path.resolve()
    core_selection_path = core_selection_path.resolve()
    scenario_path.relative_to(REPO_ROOT.resolve())
    core_selection_path.relative_to(REPO_ROOT.resolve())
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict) or scenario.get("scenario_id") != SCENARIO_ID:
        raise ValueError("Microgrid positive-control scenario identity drift")
    core_binding = _core_selection_binding(
        scenario_path=scenario_path,
        core_selection_path=core_selection_path,
    )
    before = implementation_identity(REPO_ROOT)
    with _registered_control_agent():
        domain = _run_domain(scenario_path, repeats=repeats)
    after = implementation_identity(REPO_ROOT)
    blockers = control_blockers(domain)
    if before["implementation_tree_sha256"] != after["implementation_tree_sha256"]:
        blockers.append("implementation_tree_changed_during_run")
    if domain["result"].get("scenario_signature") != scenario.get(
        "scenario_signature"
    ):
        blockers.append("scenario_signature_drift")
    blockers = sorted(set(blockers))
    report = {
        "schema_version": (
            "microgrid-source-schedule-agency-positive-control-v1"
        ),
        "status": "passed" if not blockers else "held",
        "diagnostic_only": True,
        "release_admission": False,
        "blockers": blockers,
        "implementation_stability": {
            "before": before,
            "after": after,
            "passed": before["implementation_tree_sha256"]
            == after["implementation_tree_sha256"],
        },
        "run_contract": {
            "repeats": repeats,
            "trigger_origin": "source_schedule",
            "declared_perturbation_trigger_allowed": False,
            "per_action_attribution": True,
            "per_action_cap": None,
            "per_action_group_attribution": True,
            "per_action_group_cap": None,
        },
        "core_selection_binding": core_binding,
        "scenario_binding": {
            "path": scenario_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(scenario_path),
            "scenario_id": SCENARIO_ID,
            "scenario_signature": scenario.get("scenario_signature"),
        },
        "domain": domain,
    }
    resolved_output = output.resolve()
    if not resolved_output.is_relative_to((REPO_ROOT / "reports").resolve()):
        raise ValueError("positive-control output must stay under reports")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--core-selection", type=Path, default=DEFAULT_CORE_SELECTION
    )
    args = parser.parse_args(argv)
    report = run_control(
        output=args.output,
        repeats=args.repeats,
        scenario_path=args.scenario,
        core_selection_path=args.core_selection,
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
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
