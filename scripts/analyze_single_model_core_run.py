#!/usr/bin/env python3
"""Build a resumable refinement report from a single-model core run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner import EVALUATION_IMPLEMENTATION_FINGERPRINT  # noqa: E402
from scripts.batch_llm_eval import effective_episode_rows_for_analysis  # noqa: E402

DEFAULT_SUITE = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "validated_core_suite.json"
)
DEFAULT_EPISODES = (
    REPO_ROOT
    / "batch_results"
    / "v0_52_apiyi_deepseek_flash_full_1pass"
    / "episodes.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_SUITE.with_name("deepseek_single_model_calibration.json")
DEFAULT_MODEL = "deepseek-v4-flash"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _relocated_artifact_base(raw: str, artifact_root: Path | None) -> Path:
    path = Path(raw)
    if Path(str(path) + ".trajectory.jsonl").is_file() or artifact_root is None:
        return path
    marker = "/trajectories/"
    normalized = str(path).replace("\\", "/")
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidate = artifact_root / "trajectories" / suffix
        if Path(str(candidate) + ".trajectory.jsonl").is_file():
            return candidate
    return path


def _relocated_artifact_path(raw: str, artifact_root: Path | None) -> Path:
    path = Path(raw)
    if path.is_file() or artifact_root is None:
        return path
    marker = "/trajectories/"
    normalized = str(path).replace("\\", "/")
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidate = artifact_root / "trajectories" / suffix
        if candidate.is_file():
            return candidate
    return path


def _artifact_issue_codes(
    row: dict[str, Any], artifact_root: Path | None = None
) -> tuple[list[str], dict[str, Any]]:
    trajectory = row.get("trajectory_summary") or {}
    trajectory_base = trajectory.get("trajectory_path")
    if not trajectory_base:
        return [], {}
    resolved_base = _relocated_artifact_base(str(trajectory_base), artifact_root)
    trajectory_path = Path(str(resolved_base) + ".trajectory.jsonl")
    header_path = Path(str(resolved_base) + ".header.json")
    evidence_raw = trajectory.get("evidence_path")
    issues = []
    steps = _read_rows(trajectory_path)
    if not trajectory_path.is_file():
        issues.append("trajectory_file_missing")
    expected_ticks = row.get("n_ticks_ran")
    if expected_ticks is not None and len(steps) != int(expected_ticks):
        issues.append("trajectory_tick_count_mismatch")
    call_ids = [
        str(call["call_id"])
        for step in steps
        for call in ((step.get("action") or {}).get("actions") or [])
        if isinstance(call, dict) and call.get("call_id")
    ]
    if len(call_ids) != len(set(call_ids)):
        issues.append("duplicate_trajectory_call_id")
    if header_path.is_file():
        try:
            header = json.loads(header_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            header = {}
        if (
            row.get("scenario_signature")
            and header.get("scenario_signature")
            and row["scenario_signature"] != header["scenario_signature"]
        ):
            issues.append("trajectory_signature_mismatch")
    else:
        issues.append("trajectory_header_missing")
    artifact_metrics = {
        "trajectory_ticks": len(steps),
        "duplicate_trajectory_call_ids": len(call_ids) - len(set(call_ids)),
    }
    if not evidence_raw:
        issues.append("evidence_ledger_missing")
        return issues, {
            **artifact_metrics,
            "score_evidence_refs": 0,
            "unresolved_score_evidence_refs": 0,
        }
    evidence_path = _relocated_artifact_path(str(evidence_raw), artifact_root)
    if not evidence_path.is_file():
        issues.append("evidence_ledger_missing")
        return issues, {
            **artifact_metrics,
            "score_evidence_refs": 0,
            "unresolved_score_evidence_refs": 0,
        }
    evidence_ids = {
        str(item.get("evidence_id"))
        for item in _read_rows(evidence_path)
        if item.get("evidence_id")
    }
    dimensions = (row.get("score") or {}).get("dimensions") or []
    if isinstance(dimensions, dict):
        dimensions = list(dimensions.values())
    score_refs = [
        str(evidence_id)
        for dimension in dimensions
        if isinstance(dimension, dict)
        for evidence_id in (dimension.get("evidence_ids") or [])
    ]
    unresolved = [evidence_id for evidence_id in score_refs if evidence_id not in evidence_ids]
    if unresolved:
        issues.append("score_evidence_reference_unresolved")
    return issues, {
        **artifact_metrics,
        "score_evidence_refs": len(score_refs),
        "unresolved_score_evidence_refs": len(unresolved),
    }


def _issue_codes(
    row: dict[str, Any], artifact_root: Path | None = None
) -> tuple[list[str], dict[str, Any]]:
    if row.get("status") != "ok":
        return ["episode_error"], {}
    trajectory = row.get("trajectory_summary") or {}
    llm = trajectory.get("llm") or {}
    impact = row.get("decision_impact") or {}
    complexity = trajectory.get("complexity") or {}
    tool_ok = int(trajectory.get("tool_results_ok", 0) or 0)
    tool_failed = int(trajectory.get("tool_results_failed", 0) or 0)
    tool_total = tool_ok + tool_failed
    failure_ratio = tool_failed / max(1, tool_total)
    investigations = int(impact.get("n_investigation_calls", 0) or 0)
    controls = int(impact.get("n_control_calls", 0) or 0)
    prevented_loss = float(impact.get("prevented_loss", 0.0) or 0.0)
    task_completion = row.get("task_completion") or {}
    issues = []
    if int(llm.get("llm_calls_failed", 0) or 0) > 0 or float(
        llm.get("fallback_wait_ratio", 0.0) or 0.0
    ) > 0.1:
        issues.append("provider_instability")
    if failure_ratio > 0.25:
        issues.append("tool_failure_ratio_gt_25pct")
    if investigations >= 6 and investigations >= 4 * max(1, controls):
        issues.append("query_storm_observed")
    if controls == 0:
        issues.append("zero_control_observed")
    if float(complexity.get("dependency_metadata_coverage", 0.0) or 0.0) < 1.0:
        issues.append("dependency_metadata_incomplete")
    if (
        row.get("suite_scenario_signature")
        and row.get("scenario_signature") != row.get("suite_scenario_signature")
    ):
        issues.append("suite_runtime_signature_mismatch")
    implementation = (row.get("evaluation_protocol") or {}).get(
        "implementation_fingerprint"
    )
    if not implementation:
        issues.append("evaluation_implementation_unverified")
    elif implementation != EVALUATION_IMPLEMENTATION_FINGERPRINT:
        issues.append("evaluation_implementation_mismatch")
    artifact_issues, artifact_metrics = _artifact_issue_codes(row, artifact_root)
    issues.extend(artifact_issues)
    metrics = {
        "llm_calls_ok": int(llm.get("llm_calls_ok", 0) or 0),
        "llm_calls_failed": int(llm.get("llm_calls_failed", 0) or 0),
        "fallback_wait_ratio": float(llm.get("fallback_wait_ratio", 0.0) or 0.0),
        "tool_calls": int(trajectory.get("n_tool_calls", 0) or 0),
        "tool_results_ok": tool_ok,
        "tool_results_failed": tool_failed,
        "tool_failure_ratio": round(failure_ratio, 6),
        "investigation_calls": investigations,
        "control_calls": controls,
        "prevented_loss": prevented_loss,
        "outcome_changed": bool(impact.get("outcome_changed", False)),
        "task_applicable": bool(task_completion.get("applicable")),
        "task_completed": bool(task_completion.get("completed")),
        "task_reason_code": task_completion.get("reason_code"),
        "dependency_metadata_coverage": float(
            complexity.get("dependency_metadata_coverage", 0.0) or 0.0
        ),
        **artifact_metrics,
    }
    return issues, metrics


def _resolve_scenario_id(
    row: dict[str, Any],
    expected_by_id: dict[str, dict[str, Any]],
    expected_by_leaf: dict[str, list[str]],
) -> tuple[str, str]:
    """Resolve both successful and pre-episode error rows to a suite scenario."""
    runtime_id = str(row.get("scenario_id") or "")
    slug = str(row.get("scenario_slug") or "")
    candidates = [value for value in (runtime_id, slug) if value]
    for candidate in candidates:
        if candidate in expected_by_id:
            return candidate, runtime_id or candidate.rsplit("/", 1)[-1]
        suffix_matches = [
            expected_id
            for expected_id in expected_by_id
            if candidate.endswith(f"/{expected_id}")
        ]
        if len(suffix_matches) == 1:
            expected_id = suffix_matches[0]
            return expected_id, runtime_id or expected_id.rsplit("/", 1)[-1]
        leaf_matches = expected_by_leaf.get(candidate.rsplit("/", 1)[-1]) or []
        if len(leaf_matches) == 1:
            expected_id = leaf_matches[0]
            return expected_id, runtime_id or expected_id.rsplit("/", 1)[-1]
    return runtime_id or slug, runtime_id or slug


def build_report(
    episodes_path: Path,
    suite_path: Path,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    expected_rows = list(suite.get("scenarios") or [])
    expected_by_id = {str(row["scenario_id"]): row for row in expected_rows}
    expected_by_leaf: dict[str, list[str]] = {}
    for expected_id in expected_by_id:
        expected_by_leaf.setdefault(expected_id.rsplit("/", 1)[-1], []).append(expected_id)
    episodes = effective_episode_rows_for_analysis(_read_rows(episodes_path))
    results = []
    counts: dict[str, int] = {}
    for row in episodes:
        scenario_id, runtime_scenario_id = _resolve_scenario_id(
            row,
            expected_by_id,
            expected_by_leaf,
        )
        expected = expected_by_id.get(scenario_id) or {}
        issues, metrics = _issue_codes(row, episodes_path.parent)
        if row.get("status") != "ok" or any(
            code
            in {
                "provider_instability",
                "tool_failure_ratio_gt_25pct",
                "trajectory_file_missing",
                "trajectory_header_missing",
                "trajectory_tick_count_mismatch",
                "duplicate_trajectory_call_id",
                "trajectory_signature_mismatch",
                "evidence_ledger_missing",
                "score_evidence_reference_unresolved",
                "suite_runtime_signature_mismatch",
                "evaluation_implementation_unverified",
                "evaluation_implementation_mismatch",
            }
            for code in issues
        ):
            disposition = "protocol_or_runtime_review"
        elif "query_storm_observed" in issues or "zero_control_observed" in issues:
            disposition = "model_behavior_observation_not_sample_rejection"
        else:
            disposition = "clean_single_model_observation"
        for code in issues:
            counts[code] = counts.get(code, 0) + 1
        results.append(
            {
                "scenario_id": scenario_id,
                "runtime_scenario_id": runtime_scenario_id,
                "domain": row.get("domain") or expected.get("domain"),
                "backend_kind": row.get("backend_kind")
                or expected.get("backend_kind"),
                "difficulty_level": row.get("difficulty_level")
                or expected.get("difficulty_level"),
                "status": row.get("status"),
                "score": (row.get("score") or {}).get("total_score"),
                "issue_codes": issues,
                "disposition": disposition,
                "metrics": metrics,
            }
        )
    attempted_ids = {
        row["scenario_id"] for row in results if row["scenario_id"] in expected_by_id
    }
    missing = sorted(set(expected_by_id) - attempted_ids)
    n_expected = len(expected_rows)
    if not attempted_ids:
        status = "not_started"
    elif len(attempted_ids) == n_expected:
        status = "complete"
    else:
        status = "partial"
    model_scope = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    formal_blockers = {
        "episode_error",
        "provider_instability",
        "trajectory_file_missing",
        "trajectory_header_missing",
        "trajectory_tick_count_mismatch",
        "duplicate_trajectory_call_id",
        "trajectory_signature_mismatch",
        "evidence_ledger_missing",
        "score_evidence_reference_unresolved",
        "suite_runtime_signature_mismatch",
        "evaluation_implementation_unverified",
        "evaluation_implementation_mismatch",
    }
    formal_eligible = (
        status == "complete"
        and not any(counts.get(code, 0) for code in formal_blockers)
    )
    task_applicable = [
        row for row in results if row["metrics"].get("task_applicable")
    ]
    task_completed = sum(
        bool(row["metrics"].get("task_completed")) for row in task_applicable
    )
    llm_calls_ok = sum(
        int(row["metrics"].get("llm_calls_ok", 0) or 0) for row in results
    )
    llm_calls_failed = sum(
        int(row["metrics"].get("llm_calls_failed", 0) or 0) for row in results
    )
    return {
        "schema_version": "0.2",
        "scope": f"{model_scope}_single_model_single_pass_core_calibration",
        "model": model,
        "status": status,
        "episodes_path": str(episodes_path),
        "episodes_source_exists": episodes_path.is_file(),
        "not_a_multi_model_release_calibration": True,
        "n_expected": n_expected,
        "n_completed": len(attempted_ids),
        "n_missing": len(missing),
        "n_ok": sum(row["status"] == "ok" for row in results),
        "n_error": sum(row["status"] != "ok" for row in results),
        "formal_core_calibration_eligible": formal_eligible,
        "protocol_status": (
            "formal_core_calibration_eligible"
            if formal_eligible
            else "diagnostic_only_or_incomplete"
        ),
        "issue_counts": dict(sorted(counts.items())),
        "task_completion": {
            "n_applicable": len(task_applicable),
            "n_completed": task_completed,
            "solve_rate": round(
                task_completed / max(1, len(task_applicable)), 6
            ),
        },
        "provider_health": {
            "llm_calls_ok": llm_calls_ok,
            "llm_calls_failed": llm_calls_failed,
            "failure_rate": round(
                llm_calls_failed / max(1, llm_calls_ok + llm_calls_failed),
                6,
            ),
        },
        "missing_scenario_ids": missing,
        "results": sorted(results, key=lambda row: row["scenario_id"]),
        "interpretation": {
            "query_storm_or_zero_control": (
                "A single model's poor policy is measurement evidence, not by itself "
                "a reason to reject a sample. Review the protocol only when failures, "
                "invalid schemas, missing metadata, or evaluator defects are present."
            ),
            "promotion_rule": (
                "Do not change core membership from this report alone; combine with "
                "oracle/greedy calibration, duplicate checks, and other model families."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    report = build_report(
        args.episodes.resolve(),
        args.suite.resolve(),
        model=args.model,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "n_expected", "n_completed", "n_ok", "n_error", "issue_counts")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
