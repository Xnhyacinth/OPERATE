#!/usr/bin/env python3
"""Post-process batch_results/<session>/episodes.jsonl into extra stats + issue flags."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.batch_status import execution_status_counts  # noqa: E402


def _coverage_entry(applicable: int, total: int) -> dict[str, float | int]:
    return {
        "applicable": applicable,
        "total": total,
        "coverage": float(applicable / total) if total else 0.0,
    }


def _construct_coverage(rows: list[dict]) -> dict[str, object]:
    """Report what a batch actually exercised, without changing its scores.

    Applicability is part of the scorer/evaluator contract.  Surfacing its
    denominator prevents a sparse proactive or planning diagnostic from being
    described as if it covered the entire Core.  ``difficulty_mode`` is used
    only as the existing task stratum label; no new archetype is inferred from
    model behavior.
    """

    score_names = sorted(
        {
            str(dimension.get("name"))
            for row in rows
            for dimension in (row.get("score") or {}).get("dimensions") or []
            if isinstance(dimension, dict) and dimension.get("name")
        }
    )
    agency_names = sorted(
        {
            str(name)
            for row in rows
            for name in (
                (
                    (row.get("trajectory_summary") or {}).get(
                        "operational_agency_profile"
                    )
                    or {}
                ).get("dimensions")
                or {}
            )
        }
    )

    def summarize(group: list[dict]) -> dict[str, object]:
        score_dimensions = {}
        for name in score_names:
            applicable = sum(
                any(
                    isinstance(dimension, dict)
                    and dimension.get("name") == name
                    and dimension.get("applicable") is True
                    for dimension in (row.get("score") or {}).get("dimensions") or []
                )
                for row in group
            )
            score_dimensions[name] = _coverage_entry(applicable, len(group))
        operational_agency = {}
        for name in agency_names:
            applicable = sum(
                (
                    (
                        (
                            (row.get("trajectory_summary") or {}).get(
                                "operational_agency_profile"
                            )
                            or {}
                        ).get("dimensions")
                        or {}
                    ).get(name)
                    or {}
                ).get("applicable")
                is True
                for row in group
            )
            operational_agency[name] = _coverage_entry(applicable, len(group))
        return {
            "episodes": len(group),
            "score_dimensions": score_dimensions,
            "operational_agency": operational_agency,
        }

    overall = summarize(rows)
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row.get("difficulty_mode") or "unknown")].append(row)
    return {
        "n_episodes": len(rows),
        "task_stratum_field": "difficulty_mode",
        "score_dimensions": overall["score_dimensions"],
        "operational_agency": overall["operational_agency"],
        "by_task_stratum": {
            name: summarize(group) for name, group in sorted(by_stratum.items())
        },
    }


def _load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if idx == len(raw_lines) - 1:
                break
            raise
    return rows


def analyze_output_dir(
    out_dir: Path, rows: list[dict] | None = None
) -> dict[str, object]:
    ep_path = out_dir / "episodes.jsonl"
    if rows is None:
        if not ep_path.exists():
            raise FileNotFoundError(f"no {ep_path}")
        rows = _load_jsonl_rows(ep_path)
    ok = [r for r in rows if r.get("status") == "ok"]
    execution_counts = execution_status_counts(rows)

    by_model: dict[str, list[float]] = defaultdict(list)
    tool_hist: dict[str, Counter[str]] = defaultdict(Counter)
    autonomy_totals: dict[str, Counter[str]] = defaultdict(Counter)
    flags: list[str] = []

    for r in ok:
        model = str(r.get("model", r.get("agent_name", ""))).replace("llm_agent/", "")
        by_model[model].append(float(r["score"]["total_score"]))
        traj = r.get("trajectory_summary") or {}
        for name, cnt in (traj.get("tool_histogram") or {}).items():
            tool_hist[model][name] += int(cnt)
        llm = traj.get("llm") or {}
        autonomy = traj.get("event_adaptive_autonomy") or {}
        model_totals = autonomy_totals[model]
        model_totals["episodes"] += 1
        model_totals["simulator_ticks"] += int(r.get("n_ticks_ran", 0) or 0)
        model_totals["model_decision_ticks"] += int(
            autonomy.get("model_decision_ticks", r.get("n_ticks_ran", 0)) or 0
        )
        model_totals["autonomous_hold_ticks"] += int(
            autonomy.get("autonomous_hold_ticks", 0) or 0
        )
        model_totals["decision_budget_hold_ticks"] += int(
            autonomy.get("decision_budget_hold_ticks", 0) or 0
        )
        model_totals["pending_action_hold_ticks"] += int(
            autonomy.get("pending_action_hold_ticks", 0) or 0
        )
        model_totals["decision_budget_exhausted_episodes"] += int(
            bool(autonomy.get("decision_budget_exhausted"))
        )
        model_totals["autonomy_windows"] += sum(
            record.get("kind") == "autonomy_window_opened"
            for record in autonomy.get("records") or []
        )
        model_totals["early_wakes"] += sum(
            record.get("kind") == "early_wake"
            for record in autonomy.get("records") or []
        )
        model_totals["plan_commits_confirmed"] += int(
            llm.get("plan_commits_confirmed", 0) or 0
        )
        model_totals["plan_revisions_confirmed"] += int(
            llm.get("plan_revisions_confirmed", 0) or 0
        )
        completed = (r.get("task_completion") or {}).get("completed")
        model_totals["task_outcome_observed"] += int(isinstance(completed, bool))
        model_totals["task_completed"] += int(completed is True)
        model_totals["tool_results_failed"] += int(
            traj.get("tool_results_failed", 0) or 0
        )
        argument_parse_failures = int(
            llm.get("tool_argument_parse_failures", 0) or 0
        )
        model_totals["tool_argument_parse_failures"] += argument_parse_failures
        model_totals["dependency_metadata_missing_calls"] += int(
            llm.get("dependency_metadata_missing_calls", 0) or 0
        )
        model_totals["dependency_metadata_invalid_calls"] += int(
            llm.get("dependency_metadata_invalid_calls", 0) or 0
        )
        model_totals["native_calls_dependency_metadata_cleared"] += int(
            llm.get("native_calls_dependency_metadata_cleared", 0) or 0
        )
        model_totals["protocol_repair_calls_dependency_metadata_cleared"] += int(
            llm.get(
                "protocol_repair_calls_dependency_metadata_cleared", 0
            )
            or 0
        )
        model_totals["tool_argument_truncation_failures"] += int(
            llm.get("tool_argument_truncation_failures", 0) or 0
        )
        model_totals["provider_output_truncation_count"] += int(
            llm.get("provider_output_truncation_count", 0) or 0
        )
        model_totals["provider_tool_call_failures"] += int(
            llm.get("provider_tool_call_failures", 0) or 0
        )
        model_totals["retry_attempts_total"] += int(
            llm.get("retry_attempts_total", 0) or 0
        )
        if argument_parse_failures > 0:
            flags.append(
                f"{model} {r.get('scenario_id')} s{r.get('seed')}: "
                f"tool_argument_parse_failures={argument_parse_failures}"
            )
        if int(llm.get("llm_calls_failed", 0)) > 0:
            flags.append(
                f"{model} {r.get('scenario_id')} s{r.get('seed')}: "
                f"llm_calls_failed={llm.get('llm_calls_failed')}"
            )
        provider_tool_failures = int(llm.get("provider_tool_call_failures", 0) or 0)
        if provider_tool_failures > 0:
            flags.append(
                f"{model} {r.get('scenario_id')} s{r.get('seed')}: "
                f"provider_tool_call_failures={provider_tool_failures} "
                f"fallback_without_tools_count={llm.get('fallback_without_tools_count', 0)}"
            )
        retry_attempts = int(llm.get("retry_attempts_total", 0) or 0)
        if retry_attempts > 0:
            flags.append(
                f"{model} {r.get('scenario_id')} s{r.get('seed')}: "
                f"retry_attempts_total={retry_attempts} retry_by_reason={llm.get('retry_by_reason', {})}"
            )
        if (
            int(traj.get("n_tool_calls", 0) or 0) == 0
            and int(llm.get("llm_calls_ok", 0) or 0) > 0
        ):
            flags.append(
                f"{model} {r.get('scenario_id')}: LLM ok but zero tool calls (check parsing)"
            )

    autonomy_by_model: dict[str, dict[str, float | int | None]] = {}
    for model, counts in autonomy_totals.items():
        simulator_ticks = int(counts["simulator_ticks"])
        autonomy_by_model[model] = {
            **dict(counts),
            "autonomous_hold_share": (
                float(
                    counts["autonomous_hold_ticks"]
                    + counts["pending_action_hold_ticks"]
                )
                / simulator_ticks
                if simulator_ticks
                else 0.0
            ),
            "task_completion_rate": (
                float(counts["task_completed"]) / counts["task_outcome_observed"]
                if counts["task_outcome_observed"]
                else None
            ),
        }

    return {
        "n_total": len(rows),
        "n_ok": len(ok),
        "n_error": execution_counts["n_episodes_error"],
        **execution_counts,
        "measurement_scope": "execution_ok_diagnostic_including_contaminated_rows",
        "mean_by_model": {m: sum(s) / len(s) for m, s in by_model.items()},
        "tool_histogram_by_model": {m: dict(c) for m, c in tool_hist.items()},
        "event_adaptive_autonomy_by_model": autonomy_by_model,
        "construct_coverage": _construct_coverage(ok),
        "flags": flags[:200],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("output_dir", type=Path, help="batch_results session directory")
    args = p.parse_args()
    out_dir = args.output_dir
    try:
        report = analyze_output_dir(out_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    (out_dir / "analysis_deep.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
