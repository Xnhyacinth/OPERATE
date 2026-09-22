"""Offline Lite episode reporting with complete denominators and separate cohorts.

Reads explicit immutable input snapshots; does not select successful attempts,
relabel historical scores, authenticate artifacts, or grant formal eligibility.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.leaderboard import (  # noqa: E402
    _physical_source_identity,
    aggregate_primary_leaderboard,
    infer_primary_leaderboard,
)  # noqa: E402
from evaluation.outcome_diagnostics import summarize_native_outcomes  # noqa: E402
from scripts.summarize_leaderboard_results import (  # noqa: E402
    _capability_diagnostics,
    _ok_row_cleanliness,
)  # noqa: E402

SCOPE_FIELDS = (
    "implementation_tree_sha256",
    "suite_manifest_sha256",
    "interaction_mode",
)


def _semantics(row):
    value = row.get("run_semantics_fingerprint")
    profile = row.get("agent_profile_sha256")
    suffix = f":agent-{profile}"
    return (
        value[: -len(suffix)]
        if isinstance(value, str) and profile and value.endswith(suffix)
        else value
    )


def _scope(row):
    return tuple(row.get(k) for k in SCOPE_FIELDS) + (
        (row.get("score") or {}).get("scoring_version"),
        _semantics(row),
    )


def _valid(row):
    reasons = []
    if row.get("status") != "ok":
        reasons.append("status_not_ok")
    if not all(isinstance(v, str) and v for v in _scope(row)):
        reasons.append("comparison_identity_missing")
    tree = row.get("implementation_tree_sha256")
    if any(
        row.get(k) != tree
        for k in ("implementation_tree_sha256_start", "implementation_tree_sha256_end")
    ):
        reasons.append("runtime_changed_or_unproven")
    ranking = row.get("ranking") or {}
    score = ranking.get("primary_score")
    if (
        ranking.get("aggregation") != "wait_relative_outcome_v1"
        or ranking.get("formal_score_eligible") is not True
        or type(score) not in (int, float)
        or not math.isfinite(score)
        or not 0 <= score <= 100
    ):
        reasons.append("primary_measurement_unavailable")
    completion = row.get("task_completion") or {}
    if (
        completion.get("applicable") is not True
        or type(completion.get("completed")) is not bool
        or not completion.get("evidence")
    ):
        reasons.append("completion_measurement_unavailable")
    terminal = (row.get("trajectory_summary") or {}).get("terminal_integrity") or {}
    if (
        terminal.get("release_ready") is not True
        or terminal.get("collection_complete") is not True
    ):
        reasons.append("terminal_evidence_incomplete")
    clean, reason = _ok_row_cleanliness(row)
    if not clean:
        reasons.append(reason)
    return reasons


def build_report(suite_rows, inputs, *, bootstrap=0):
    expected = {r["scenario_signature"]: r for r in suite_rows}
    if (
        not expected
        or len(expected) != len(suite_rows)
        or any(not isinstance(k, str) or not k.strip() for k in expected)
    ):
        raise ValueError("suite must have unique nonempty scenario signatures")
    reports, admitted, cohort_models = {}, {}, defaultdict(list)
    observed_admitted, observed_groups = {}, defaultdict(list)
    for model, episodes in sorted(inputs.items()):
        grouped = defaultdict(list)
        blockers = []
        for row in episodes:
            signature = row.get("scenario_signature")
            if signature not in expected:
                blockers.append(
                    {"reason": "outside_suite", "scenario_signature": signature}
                )
            else:
                grouped[signature].append(row)
        model_rows, cells, scopes, profiles, passes = [], [], set(), set(), set()
        observed_rows = []
        for signature, spec in expected.items():
            attempts = grouped[signature]
            reasons = []
            if len(attempts) != 1:
                reasons.append(
                    "missing_attempt" if not attempts else "duplicate_attempts"
                )
                cells.append(
                    {
                        "scenario_signature": signature,
                        "reasons": reasons,
                        "n_attempts": len(attempts),
                    }
                )
                continue
            row = attempts[0]
            reasons.extend(_valid(row))
            scopes.add(_scope(row))
            profile = (
                row.get("agent_profile_sha256"),
                row.get("run_semantics_fingerprint"),
                row.get("agent_treatment_sha256"),
                row.get("model"),
            )
            if not all(isinstance(v, str) and v for v in profile):
                reasons.append("model_profile_missing")
            profiles.add(profile)
            if type(row.get("seed")) is not int or row.get("pass_id") in (None, ""):
                reasons.append("repeat_identity_missing")
            passes.add(str(row.get("pass_id")))
            for key in (
                "domain",
                "backend_kind",
                "source_denominator_key",
                "physical_source_key",
                "scenario_id",
            ):
                if not spec.get(key) or (
                    row.get(key) not in (None, "") and row[key] != spec[key]
                ):
                    reasons.append(f"suite_lineage_mismatch:{key}")
            ledger = row.get("case_ledger") or {}
            if ledger.get("source_denominator_key") not in (
                None,
                "",
                spec.get("source_denominator_key"),
            ):
                reasons.append("nested_source_lineage_mismatch")
            if ledger.get("physical_source_key") not in (
                None,
                "",
                spec.get("physical_source_key"),
            ):
                reasons.append("nested_physical_source_lineage_mismatch")
            physical = _physical_source_identity(row)
            expected_physical = _physical_source_identity(spec)

            def decoded(value):
                for _ in range(2):
                    if not isinstance(value, str):
                        break
                    try:
                        value = json.loads(value)
                    except (ValueError, TypeError):
                        break
                return value

            if physical is not None and decoded(physical) != decoded(expected_physical):
                reasons.append("physical_source_lineage_mismatch")
            cells.append(
                {
                    "scenario_signature": signature,
                    "reasons": reasons,
                    "primary_score": (row.get("ranking") or {}).get("primary_score"),
                }
            )
            score_measured = (
                "primary_measurement_unavailable" not in reasons
                and "completion_measurement_unavailable" not in reasons
            )
            if score_measured:
                normalized = deepcopy(row)
                # Suite lineage enriches absent fields, never conflicting ones.
                for key in (
                    "domain",
                    "backend_kind",
                    "source_denominator_key",
                    "physical_source_key",
                    "scenario_id",
                ):
                    if key in spec:
                        normalized[key] = spec[key]
                normalized.update(
                    model=model,
                    model_id=model,
                    discriminative_core_score=row["ranking"]["primary_score"],
                    task_completion_raw=float(row["task_completion"]["completed"]),
                )
                observed_rows.append(normalized)
                if not reasons:
                    model_rows.append(normalized)
        if len(scopes) != 1 or len(profiles) != 1 or len(passes) != 1:
            blockers.append({"reason": "mixed_or_missing_model_run_identity"})
        complete = len(model_rows) == len(expected) and not blockers
        identity_reasons = {
            "comparison_identity_missing",
            "runtime_changed_or_unproven",
            "model_profile_missing",
            "repeat_identity_missing",
            "nested_source_lineage_mismatch",
            "physical_source_lineage_mismatch",
            "nested_physical_source_lineage_mismatch",
        }
        observed_identity_ok = all(
            not any(
                reason in identity_reasons
                or reason.startswith("suite_lineage_mismatch:")
                for reason in cell["reasons"]
            )
            for cell in cells
        )
        observed_complete = (
            observed_identity_ok
            and len(observed_rows) == len(expected)
            and not blockers
            and all(all(isinstance(v, str) and v for v in scope) for scope in scopes)
        )
        if observed_complete:
            observed_admitted[model] = observed_rows
            observed_groups[
                (
                    next(iter(scopes)),
                    tuple(
                        sorted(
                            (r["scenario_signature"], r["seed"], str(r["pass_id"]))
                            for r in observed_rows
                        )
                    ),
                )
            ].append(model)
        reports[model] = {
            "diagnostic_observed_subset": aggregate_primary_leaderboard(observed_rows)[
                "leaderboard"
            ][0]
            if observed_rows
            else None,
            "diagnostic_capabilities": _capability_diagnostics(
                [{**r, "_capability_episode": r} for r in observed_rows], [model]
            )
            if observed_rows
            else None,
            "n_observed_scores": len(observed_rows),
            "observed_score_complete": observed_complete,
            "diagnostic_valid_subset": aggregate_primary_leaderboard(model_rows)[
                "leaderboard"
            ][0]
            if model_rows
            else None,
            "diagnostic_native_outcomes": summarize_native_outcomes(model_rows)
            if model_rows
            else {},
            "complete": complete,
            "n_expected": len(expected),
            "n_valid": len(model_rows),
            "blockers": blockers,
            "cells": cells,
            "scopes": [list(s) for s in sorted(scopes, key=str)],
            "profiles": [list(s) for s in sorted(profiles, key=str)],
        }
        for subset_key, subset_rows in (
            ("diagnostic_valid_subset", model_rows),
            ("diagnostic_observed_subset", observed_rows),
        ):
            subset = reports[model][subset_key]
            if subset:
                versions = {
                    (r.get("score") or {}).get("scoring_version") for r in subset_rows
                }
                subset["scoring_version"] = (
                    next(iter(versions)) if len(versions) == 1 else None
                )
                subset["not_a_complete_suite_index"] = not (
                    complete
                    if subset_key == "diagnostic_valid_subset"
                    else observed_complete
                )
                subset["audit_passed"] = complete
        if complete:
            admitted[model] = model_rows
            cohort_models[
                (
                    next(iter(scopes)),
                    tuple(
                        sorted(
                            (r["scenario_signature"], r["seed"], str(r["pass_id"]))
                            for r in model_rows
                        )
                    ),
                )
            ].append(model)
    cohorts = _cohorts(admitted, cohort_models, expected, bootstrap)
    observed_cohorts = _cohorts(observed_admitted, observed_groups, expected, bootstrap)
    for cohort in observed_cohorts:
        cohort["audit_passed"] = all(reports[m]["complete"] for m in cohort["models"])
        cohort["interpretation"] = (
            "stored_outcome_description_not_audit_cleared_ranking"
        )
    return {
        "schema_version": "lite_trajectory_report.v1",
        "formal_eligible": False,
        "score_basis": "stored_wait_relative_primary_not_native_quality_candidate",
        "n_suite_rows": len(expected),
        "models": reports,
        "cohorts": cohorts,
        "observed_cohorts": observed_cohorts,
        "native_quality_index": {
            "status": "requires_independently_calibrated_references",
            "model_trajectories_are_not_reference_anchors": True,
        },
    }


def _cohorts(admitted, cohort_models, expected, bootstrap):
    cohorts = []
    for (scope, repeat), models in sorted(cohort_models.items(), key=str):
        rows = [row for model in models for row in admitted[model]]
        aggregation = aggregate_primary_leaderboard(rows)
        # Aggregator labels with live version; retain actual stored score version.
        aggregation["scoring_version"] = scope[-2]
        for item in aggregation["leaderboard"]:
            item["scoring_version"] = scope[-2]
        inference = None
        inference_error = None
        if bootstrap:
            try:
                inference_rows = [
                    {
                        k: r[k]
                        for k in (
                            "model",
                            "model_id",
                            "domain",
                            "backend_kind",
                            "source_denominator_key",
                            "physical_source_key",
                            "discriminative_core_score",
                            "task_completion_raw",
                            "seed",
                        )
                    }
                    for r in rows
                ]
                inference = infer_primary_leaderboard(
                    inference_rows, n_bootstrap=bootstrap, seed=1729
                )
                inference["scoring_version"] = scope[-2]
                for item in inference["leaderboard"]:
                    item["scoring_version"] = scope[-2]
            except ValueError as exc:
                inference_error = str(exc)
        discrimination = []
        for signature in expected:
            values = {
                model: next(
                    r["discriminative_core_score"]
                    for r in admitted[model]
                    if r["scenario_signature"] == signature
                )
                for model in models
            }
            span = max(values.values()) - min(values.values())
            discrimination.append(
                {
                    "scenario_signature": signature,
                    "scores": values,
                    "range": span,
                    "all_equal": span == 0 if len(models) > 1 else None,
                    "all_at_least_90": min(values.values()) >= 90
                    if len(models) > 1
                    else None,
                }
            )
        cohorts.append(
            {
                "scope": dict(
                    zip(
                        (*SCOPE_FIELDS, "scoring_version", "comparison_semantics"),
                        scope,
                    )
                ),
                "repeat_membership": [list(r) for r in repeat],
                "models": models,
                "ranking": aggregation["leaderboard"],
                "inference": inference,
                "inference_error": inference_error,
                "case_discrimination": discrimination,
                "native_outcomes": summarize_native_outcomes(rows),
                "capability_diagnostics": _capability_diagnostics(
                    [{**r, "_capability_episode": r} for r in rows], models
                ),
            }
        )
    return cohorts


def select_latest_attempts(rows):
    """Explicit descriptive snapshot by start time, never by outcome quality."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (row.get("scenario_signature"), row.get("seed"), row.get("pass_id"))
        ].append(row)
    selected, history = [], []
    for key, attempts in grouped.items():
        if len(attempts) == 1:
            selected.extend(attempts)
            continue
        try:
            times = [
                datetime.fromisoformat(r["invocation_started_at_utc"]) for r in attempts
            ]
            if any(t.tzinfo is None for t in times) or any(
                not r.get("execution_attempt_id") for r in attempts
            ):
                raise ValueError("missing attempt identity")
            newest = max(times)
            indices = [i for i, t in enumerate(times) if t == newest]
            if len(indices) != 1:
                raise ValueError("ambiguous latest attempt")
        except (KeyError, TypeError, ValueError):
            selected.extend(attempts)
            continue
        index = indices[0]
        selected.append(attempts[index])
        history.append(
            {
                "scenario_signature": key[0],
                "seed": key[1],
                "pass_id": key[2],
                "selected_attempt_id": attempts[index]["execution_attempt_id"],
                "excluded_attempt_ids": [
                    r["execution_attempt_id"]
                    for i, r in enumerate(attempts)
                    if i != index
                ],
            }
        )
    return selected, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="JSON with suite path and runs mapping label to episodes.jsonl paths",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()
    config_bytes = args.manifest.read_bytes()
    config = json.loads(config_bytes)
    suite_path = Path(config["suite"])
    suite_bytes = suite_path.read_bytes()
    suite = json.loads(suite_bytes)
    inputs, provenance = {}, []
    for model, paths in config["runs"].items():
        inputs[model] = []
        for name in paths:
            raw = Path(name).read_bytes()
            # A partial final line is an error, never silently discarded.
            parsed = [json.loads(line) for line in raw.splitlines() if line.strip()]
            filters = config.get("row_filters", {}).get(model, {})
            allowed = {
                "implementation_tree_sha256",
                "agent_profile_sha256",
                "agent_treatment_sha256",
                "pass_id",
            }
            if set(filters) - allowed:
                raise ValueError("only explicit run identity filters are allowed")
            selected = [
                r for r in parsed if all(r.get(k) == v for k, v in filters.items())
            ]
            inputs[model].extend(selected)
            provenance.append(
                {
                    "model": model,
                    "path": name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "identity_filters": filters,
                    "n_input": len(parsed),
                    "n_selected": len(selected),
                    "n_excluded": len(parsed) - len(selected),
                }
            )
    attempt_policy = config.get("attempt_policy", "retain_all")
    if attempt_policy not in {"retain_all", "latest_started"}:
        raise ValueError("unknown attempt policy")
    history = {}
    if attempt_policy == "latest_started":
        for model, rows in inputs.items():
            inputs[model], history[model] = select_latest_attempts(rows)
    report = build_report(suite["scenarios"], inputs, bootstrap=args.bootstrap)
    report["attempt_policy"] = attempt_policy
    report["attempt_selection_history"] = history
    report["inputs"] = {
        "manifest_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "suite_sha256": hashlib.sha256(suite_bytes).hexdigest(),
        "episodes": provenance,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    with (args.output_dir / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["run", "scenario_signature", "primary_score", "audit_reasons"],
        )
        writer.writeheader()
        for label, model_report in report["models"].items():
            for cell in model_report["cells"]:
                writer.writerow(
                    {
                        "run": label,
                        "scenario_signature": cell["scenario_signature"],
                        "primary_score": cell.get("primary_score"),
                        "audit_reasons": "|".join(cell["reasons"]),
                    }
                )
    lines = [
        "# Lite trajectory analysis",
        "",
        "Stored primary, complete denominators, separate execution cohorts. Not formal certification or a calibrated native-quality index.",
        "",
        "| run | observed/expected | audit-valid | full observed scores |",
        "|---|---:|---:|---|",
    ]
    lines += [
        f"| {m} | {r['n_observed_scores']}/{r['n_expected']} | {r['n_valid']} | {r['observed_score_complete']} |"
        for m, r in report["models"].items()
    ]
    for i, cohort in enumerate(report["observed_cohorts"], 1):
        lines += [
            "",
            f"## Observed cohort {i} (audit passed: {cohort['audit_passed']})",
            "",
            "| order | model | stored primary macro |",
            "|---:|---|---:|",
        ]
        lines += [
            f"| {j} | {r['model']} | {r['primary_leaderboard_score']:.4f} |"
            for j, r in enumerate(cohort["ranking"], 1)
        ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "complete_runs": sum(r["complete"] for r in report["models"].values()),
                "cohorts": len(report["cohorts"]),
            }
        )
    )


if __name__ == "__main__":
    main()
