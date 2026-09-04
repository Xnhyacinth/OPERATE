#!/usr/bin/env python3
"""Turn replay-minimization evidence into fail-closed difficulty dispositions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.difficulty_contract import (  # noqa: E402
    DIFFICULTY_CONTRACT_VERSION,
    calibrate_difficulty_level,
)
from core.implementation_identity import implementation_identity  # noqa: E402

DEFAULT_RELEASE_DIR = (
    REPO_ROOT / "release" / "operate_v0_58_0_candidate" / "operate_v058_formal"
)
DEFAULT_INPUT = DEFAULT_RELEASE_DIR / "complexity_protocol2_v21.json"
DEFAULT_OUTPUT = DEFAULT_RELEASE_DIR / "strategy_depth_protocol2_v21.json"
DEPTH_FLOORS = {"basic": 1, "medium": 1, "high": 2, "extreme": 3}


def _scenario_body(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row.get("path") or ""))
    if not path.is_absolute():
        path = REPO_ROOT / path
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"scenario is not a mapping: {path}")
    return body


def build_report(
    *,
    calibration: dict[str, Any],
    source_suite: dict[str, Any] | None = None,
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    config = calibration.get("config") or {}
    strict_protocol21_scope = (
        str(config.get("evaluation_protocol_version") or "") == "2.1"
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in calibration.get("results") or []:
        grouped[str(result.get("scenario_id") or "")].append(result)
    source_rows = {
        str(row.get("scenario_id") or ""): row
        for row in (source_suite or {}).get("scenarios") or []
    }
    for scenario_id, results in sorted(grouped.items()):
        level = str(results[0].get("difficulty_level") or "")
        scenario_signatures = {
            str(result.get("scenario_signature") or "")
            for result in results
            if result.get("scenario_signature")
        }
        floor = DEPTH_FLOORS.get(level, 1)
        successful: list[tuple[dict[str, Any], dict[str, Any], int]] = []
        for result in results:
            minimization = result.get("replay_minimization") or {}
            ticks = minimization.get("successful_decision_tick_upper_bound")
            if ticks is None:
                ticks = minimization.get("one_minimal_decision_ticks")
            if (
                result.get("status") == "complete"
                and isinstance(ticks, list)
                and minimization.get("status") != "initial_trace_not_successful"
            ):
                successful.append(
                    (
                        result,
                        minimization,
                        len(set(int(value) for value in ticks)),
                    )
                )
        successful_upper_bound = (
            min(item[2] for item in successful) if successful else None
        )
        lower_bound = max(
            (
                int(item[1].get("non_meta_call_count_lower_bound") or 0)
                for item in successful
            ),
            default=0,
        )
        required_depth_lower_bound = max(
            (
                int(item[1].get("required_depth_lower_bound") or 0)
                for item in successful
            ),
            default=0,
        )
        depth_proof_kinds = sorted(
            {
                str(item[1].get("depth_proof_kind") or "")
                for item in successful
                if item[1].get("depth_proof_kind")
            }
        )
        exact_task_dependency_depth = max(
            (
                int(item[1].get("exact_dependency_depth") or 0)
                for item in successful
                if item[1].get("dependency_depth_status")
                == "one_minimal_task_contract_phase_dag"
            ),
            default=0,
        )
        agent_counts = Counter(
            str(result.get("agent_name") or result.get("agent") or "")
            for result in results
        )
        multiplicity_error = strict_protocol21_scope and (
            len(scenario_signatures) != 1
            or sorted(agent_counts) != [
                "greedy_heuristic",
                "oracle_offline",
                "wait_only",
            ]
            or any(count != 1 for count in agent_counts.values())
        )
        if multiplicity_error:
            disposition = "complexity_identity_multiplicity_error"
            core_action = "hold_out"
        elif not any(result.get("status") == "complete" for result in results):
            disposition = "calibration_error"
            core_action = "hold_out"
        elif successful_upper_bound is None:
            disposition = "successful_reference_trace_not_proven"
            core_action = "hold_pending_reference_repair"
        elif successful_upper_bound < floor:
            disposition = "replace_or_retire_depth_contradicted"
            core_action = "replace_or_retire"
        elif (
            exact_task_dependency_depth >= floor
            or required_depth_lower_bound >= floor
            or floor == 1
            and lower_bound >= 1
        ):
            disposition = "required_depth_lower_bound_met"
            core_action = "keep"
        else:
            disposition = "required_depth_not_proven"
            core_action = "hold_pending_lower_bound"
        difficulty_calibration: dict[str, Any] | None = None
        if source_suite is not None:
            source_row = source_rows.get(scenario_id)
            source_signature = str(
                (source_row or {}).get("scenario_signature") or ""
            )
            if (
                source_row is None
                or not successful
                or len(scenario_signatures) != 1
                or source_signature != next(iter(scenario_signatures))
            ):
                difficulty_calibration = {
                    "version": DIFFICULTY_CONTRACT_VERSION,
                    "status": "held",
                    "failure": (
                        "difficulty_source_signature_mismatch"
                        if source_row is not None and successful
                        else "difficulty_evidence_missing"
                    ),
                }
            else:
                result, minimization, _ticks = min(
                    successful,
                    key=lambda item: item[2],
                )
                observed = result.get("observed") or {}
                observed_endpoints = set(
                    observed.get("observed_physical_actuator_endpoint_set") or []
                )
                minimality_for_contract = dict(minimization)
                minimal_endpoints = set(
                    minimization.get(
                        "one_minimal_physical_actuator_endpoint_set"
                    )
                    or []
                )
                physical_tools = set(
                    observed.get("observed_state_changing_tool_set") or []
                ) | observed_endpoints | minimal_endpoints
                try:
                    difficulty_calibration = calibrate_difficulty_level(
                        _scenario_body(source_row),
                        trajectory_complexity=observed,
                        replay_minimality=minimality_for_contract,
                        physical_tool_names=physical_tools,
                    )
                except (OSError, ValueError, yaml.YAMLError) as exc:
                    difficulty_calibration = {
                        "version": DIFFICULTY_CONTRACT_VERSION,
                        "status": "held",
                        "failure": "difficulty_scenario_unreadable",
                        "detail": str(exc),
                    }
            if difficulty_calibration.get("status") != "passed":
                disposition = "difficulty_evidence_missing"
                core_action = "hold_relabel_or_redesign"
            elif (
                difficulty_calibration.get("declared_level_matches_evidence")
                is not True
            ):
                disposition = "difficulty_relabel_required"
                core_action = "hold_relabel"
        samples.append(
            {
                "scenario_id": scenario_id,
                "scenario_signature": (
                    next(iter(scenario_signatures))
                    if len(scenario_signatures) == 1
                    else None
                ),
                "difficulty_level": level,
                "tier_floor": floor,
                "successful_strategy_tick_upper_bound": successful_upper_bound,
                "non_meta_call_count_lower_bound": lower_bound or None,
                "required_depth_lower_bound": (
                    required_depth_lower_bound or None
                ),
                "depth_proof_kinds": depth_proof_kinds,
                "exact_task_dependency_depth": (
                    exact_task_dependency_depth or None
                ),
                "successful_reference_agents": sorted(
                    {
                        str(result.get("agent_name") or "unknown")
                        for result, _minimization, _ticks in successful
                    }
                ),
                "trace_subset_minimum_statuses": sorted(
                    {
                        str(minimization.get("trace_subset_minimum_status"))
                        for _result, minimization, _ticks in successful
                        if minimization.get("trace_subset_minimum_status")
                    }
                ),
                "disposition": disposition,
                "core_action": core_action,
                "difficulty_calibration": difficulty_calibration,
                "claim": (
                    "A successful trace below the tier floor disproves the required "
                    "temporal depth. A trace at or above the floor is only an upper "
                    "bound and cannot prove that the depth is necessary; an explicit "
                    "ordered task contract may contribute a conservative necessary "
                    "stage-count lower bound, but never a global-shortest claim."
                ),
            }
        )
    counts = Counter(str(row["disposition"]) for row in samples)
    return {
        "schema_version": "1.0",
        "scope": "bounded_replay_strategy_depth_difficulty_gate",
        "difficulty_contract_version": DIFFICULTY_CONTRACT_VERSION,
        "calibration_status": calibration.get("status"),
        "evaluation_semantics": {
            "protocol_version": config.get("evaluation_protocol_version"),
            "implementation_fingerprint": config.get(
                "evaluation_implementation_fingerprint"
            ),
            "scoring_version": config.get("scoring_version"),
        },
        "implementation_tree_sha256": implementation_identity()[
            "implementation_tree_sha256"
        ],
        "n_expected": len(grouped),
        "n_completed": len(samples),
        "n_reference_results": len(calibration.get("results") or []),
        "complete": (
            calibration.get("status") == "complete"
            and len(calibration.get("results") or [])
            == int(calibration.get("n_expected", 0) or 0)
            and all(
                row.get("disposition")
                != "complexity_identity_multiplicity_error"
                for row in samples
            )
        ),
        "summary": {"disposition_counts": dict(sorted(counts.items()))},
        "policy": {
            "below_floor_successful_upper_bound_is_contradictory": True,
            "above_floor_upper_bound_is_not_a_required_depth_proof": True,
            "difficulty_relabel_without_task_redesign_allowed": True,
            "difficulty_relabel_requires_replay_supported_public_level": True,
            "difficulty_relabel_requires_new_identity_and_full_replay": True,
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--suite", type=Path)
    args = parser.parse_args()
    calibration = json.loads(args.input.read_text(encoding="utf-8"))
    source_suite = (
        json.loads(args.suite.read_text(encoding="utf-8"))
        if args.suite
        else None
    )
    report = build_report(
        calibration=calibration,
        source_suite=source_suite,
    )
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "n_completed": report["n_completed"],
                "n_expected": report["n_expected"],
                "dispositions": report["summary"]["disposition_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
