#!/usr/bin/env python3
"""Build a per-sample Hy3 setting audit without conflating model and data failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner import (  # noqa: E402
    EVALUATION_IMPLEMENTATION_FINGERPRINT,
    EVALUATION_PROTOCOL_VERSION,
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _trajectory_path(row: dict[str, Any], repo_root: Path) -> Path | None:
    raw = str((row.get("trajectory_summary") or {}).get("trajectory_path") or "")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = repo_root / path
    candidate = path.with_suffix(".trajectory.jsonl")
    return candidate if candidate.is_file() else None


def _tool_diagnostics(
    row: dict[str, Any], repo_root: Path
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    error_codes: Counter[str] = Counter()
    errors_by_tool: Counter[str] = Counter()
    meta_action_issues: Counter[str] = Counter()
    path = _trajectory_path(row, repo_root)
    if path is None:
        return error_codes, errors_by_tool, meta_action_issues
    for step in _load_jsonl(path):
        for result in step.get("tool_results") or []:
            error_code = str(result.get("error_code") or "")
            tool_name = str(result.get("name") or "")
            latency = int(result.get("latency_ticks", 0) or 0)
            if error_code:
                error_codes[error_code] += 1
                errors_by_tool[f"{tool_name}:{error_code}"] += 1
            if tool_name in {"wait", "noop"} and (
                error_code == "INJECTED_FAILURE" or latency > 0
            ):
                meta_action_issues[error_code or "DELAYED"] += 1
    return error_codes, errors_by_tool, meta_action_issues


def build_report(
    *,
    suite: dict[str, Any],
    operational: dict[str, Any],
    episodes: list[dict[str, Any]],
    repo_root: Path,
) -> dict[str, Any]:
    operational_by_id = {
        str(row["scenario_id"]): row for row in operational.get("samples") or []
    }
    scenario_by_signature = {
        str(row.get("scenario_signature")): row for row in suite.get("scenarios") or []
    }
    episode_by_signature = {
        str(row.get("scenario_signature")): row
        for row in episodes
        if row.get("status") == "ok" and row.get("scenario_signature")
    }

    samples: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    aggregate_errors: Counter[str] = Counter()
    for signature, scenario in scenario_by_signature.items():
        scenario_id = str(scenario["scenario_id"])
        episode = episode_by_signature.get(signature)
        admission = operational_by_id.get(scenario_id, {})
        if episode is None:
            samples.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_signature": signature,
                    "status": "missing_hy3_episode",
                    "core_disposition": "keep_if_reference_gates_pass_rerun_required",
                }
            )
            totals["missing_hy3_episode"] += 1
            totals["rerun_required"] += 1
            continue

        error_codes, errors_by_tool, meta_action_issues = _tool_diagnostics(
            episode, repo_root
        )
        aggregate_errors.update(error_codes)
        llm = (episode.get("trajectory_summary") or {}).get("llm") or {}
        provider_contaminated = int(llm.get("llm_calls_failed", 0) or 0) > 0
        hidden_cost_budget_encountered = (
            error_codes["TICK_COST_BUDGET_EXHAUSTED"] > 0
        )
        old_meta_action_semantics_encountered = bool(meta_action_issues)
        task_completed = bool((episode.get("task_completion") or {}).get("completed"))
        reference_gates_passed = not bool(admission.get("hard_issues"))
        protocol = episode.get("evaluation_protocol") or {}
        protocol_current = (
            str(protocol.get("version") or "") == EVALUATION_PROTOCOL_VERSION
            and str(protocol.get("implementation_fingerprint") or "")
            == EVALUATION_IMPLEMENTATION_FINGERPRINT
            and bool(protocol.get("within_tick_interaction"))
        )

        totals["covered"] += 1
        totals["task_completed" if task_completed else "task_failed"] += 1
        totals["provider_contaminated"] += int(provider_contaminated)
        totals["hidden_cost_budget_encountered"] += int(
            hidden_cost_budget_encountered
        )
        totals["old_meta_action_semantics_encountered"] += int(
            old_meta_action_semantics_encountered
        )
        totals["reference_gates_passed"] += int(reference_gates_passed)
        totals["protocol_current"] += int(protocol_current)

        framework_confounds = []
        if not protocol_current:
            framework_confounds.append("stale_or_incomplete_protocol")
        if provider_contaminated:
            framework_confounds.append("provider_fallback")
        if hidden_cost_budget_encountered and not protocol_current:
            framework_confounds.append("undisclosed_cost_budget")
        if old_meta_action_semantics_encountered:
            framework_confounds.append("fallible_or_delayed_meta_action")
        rerun_required = bool(framework_confounds)
        totals["rerun_required"] += int(rerun_required)

        samples.append(
            {
                "scenario_id": scenario_id,
                "scenario_signature": signature,
                "domain": scenario.get("domain"),
                "family": scenario.get("family"),
                "backend_kind": scenario.get("backend_kind"),
                "difficulty_level": scenario.get("difficulty_level"),
                "reference_gates_passed": reference_gates_passed,
                "protocol_current": protocol_current,
                "hy3_episode": {
                    "status": episode.get("status"),
                    "task_completed": task_completed,
                    "total_score": (episode.get("score") or {}).get("total_score"),
                    "n_ticks": (episode.get("trajectory_summary") or {}).get("n_ticks"),
                    "n_tool_calls": (episode.get("trajectory_summary") or {}).get(
                        "n_tool_calls"
                    ),
                    "effective_control_ticks": (
                        (episode.get("trajectory_summary") or {}).get("complexity")
                        or {}
                    ).get("n_effective_control_ticks"),
                    "provider_contaminated": provider_contaminated,
                    "tool_error_codes": dict(sorted(error_codes.items())),
                    "tool_errors_by_tool": dict(sorted(errors_by_tool.items())),
                },
                "framework_confounds": framework_confounds,
                "core_disposition": (
                    (
                        "keep_reference_valid_rerun_required"
                        if rerun_required
                        else "keep_reference_valid_protocol_current"
                    )
                    if reference_gates_passed
                    else "retire_or_replace_reference_gate_failure"
                ),
                "setting_decision": (
                    "do_not_mutate_sample_from_single_model_failure"
                    if reference_gates_passed
                    else "repair_or_replace_before_core"
                ),
            }
        )

    return {
        "schema_version": "1.0",
        "suite_id": suite.get("suite_id"),
        "model": "hy3-ioa",
        "n_core_samples": len(scenario_by_signature),
        "summary": {
            **dict(sorted(totals.items())),
            "tool_error_codes": dict(sorted(aggregate_errors.items())),
        },
        "interpretation": {
            "core_admission_basis": (
                "Reference-policy provenance, determinism, native leverage, task "
                "contract, duplicate and difficulty gates."
            ),
            "single_model_boundary": (
                "A Hy3 task failure or schema error measures the model; it does not "
                "alone justify changing or retiring a sample."
            ),
            "protocol_rerun_reason": (
                "Only stale-protocol, provider-contaminated, or framework-confounded "
                "rows require rerun; clean protocol-current rows are retained."
            ),
        },
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--operational", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    report = build_report(
        suite=_load_json(args.suite),
        operational=_load_json(args.operational),
        episodes=_load_jsonl(args.episodes),
        repo_root=repo_root,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
