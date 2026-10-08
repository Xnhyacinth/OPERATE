#!/usr/bin/env python3
"""Verify and materialize the 0.26.1 Lite141 offline main table.

This reads frozen reports and source-suite metadata. It does not run a model,
reference policy, solver, or environment replay. Core Full is an optional,
separately scored extension, not a prerequisite for Lite publication.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "0.26.1"
EXPECTED_REVISION = "source_obligation_outcome.v2"
EXPECTED_CASES = 141
EXPECTED_COMPLETION_CASES = 131
EXPECTED_ECONOMIC_CASES = 10
DEFAULT_POLICY = Path("release/operate_v0_62_0/lite_main_scoring_0261.json")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _finite_score(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8)


def validate_lite_policy(
    policy: dict, report: dict, suite: dict, suite_raw: bytes
) -> None:
    """Bind the official Lite track without relabeling historical Core flags."""
    expected = {
        "schema_version": "operate_lite_main_scoring_policy.v1",
        "release_namespace": suite.get("parent_release_id"),
        "main_track": "lite141_offline",
        "suite_sha256": _sha256(suite_raw),
        "case_count": EXPECTED_CASES,
        "scoring_version": EXPECTED_VERSION,
        "protocol_revision": EXPECTED_REVISION,
        "weight_manifest_sha256": report["task_weight_manifest"]["manifest_sha256"],
        "completion_contracts_sha256": report["completion_contracts_sha256"],
        "scoring_entry_sha256": report["scoring_entry_sha256"],
        "suite_selection": "model_informed_lite_development_panel",
        "model_outcomes_used_for_suite_selection": True,
        "requires_all_lite_cases_per_ranked_model": True,
        "retrospective_offline_scoring_allowed": True,
        "strict_binary_attainment_claim": False,
        "full_core_extension_required_for_lite_main_release": False,
        "full_core_extension_status": "optional_separately_scored_track",
    }
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("lite_main_policy_mismatch")
    if (
        suite.get("track") != "efficiency_development"
        or suite.get("formal_full_leaderboard_eligible") is not False
        or policy.get("suite_path") != "release/operate_v0_62_0/lite_suite.json"
    ):
        raise ValueError("lite_main_scope_mismatch")


def verify_case_rows(report: dict, source: dict, weights: dict) -> dict:
    """Independently recalculate every available point score and model index."""
    old_rows = {
        (row["model"], row["scenario_signature"], row["seed"]): row
        for row in source["graded_episodes"]
    }
    if len(old_rows) != len(source["graded_episodes"]):
        raise ValueError("duplicate_source_episode")
    rows = {}
    scored = Counter()
    model_totals = Counter()
    cap_totals = Counter()
    for row in report["graded_episodes"]:
        model = row["model"]
        case = row["scenario_signature"], row["seed"]
        identity = model, *case
        if identity in rows or identity not in old_rows or case not in weights:
            raise ValueError("duplicate_or_foreign_episode")
        rows[identity] = row
        original = old_rows[identity]
        if row["artifact_binding"] != original["artifact_binding"]:
            raise ValueError("selected_artifact_changed")
        mission = row["mission"]
        old_quality = original["mission"].get("score")
        if mission.get("native_quality_025") != old_quality:
            raise ValueError("native_quality_changed")
        quality = mission.get("native_quality_025")
        score = mission.get("score")
        completion = row["completion"]
        if score is None:
            if quality is not None:
                raise ValueError("scored_native_case_missing_joint_score")
            continue
        if not (_finite_score(score) and _finite_score(quality)):
            raise ValueError("invalid_case_score")
        safety = mission.get("safety") or {}
        if (
            safety.get("verified") is not True
            or type(safety.get("hard_failure")) is not bool
        ):
            raise ValueError("unverified_hard_gate")
        if not (
            mission.get("native_evidence_ids")
            and row["artifact_binding"].get("verified") is True
        ):
            raise ValueError("unbound_native_evidence")
        cost, scale = mission.get("native_objective"), mission.get("source_scale")
        if (
            type(cost) not in (int, float)
            or type(scale) not in (int, float)
            or not math.isfinite(cost)
            or not math.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("invalid_native_cost_or_scale")
        if cost < 0 and mission.get("signed_objective") is not True:
            raise ValueError("unsigned_negative_native_cost")
        expected_quality = (
            0.0
            if safety["hard_failure"]
            else 100.0
            if cost < 0
            else 100.0 / (1.0 + cost / scale)
        )
        if not _close(quality, expected_quality):
            raise ValueError("native_quality_formula_mismatch")
        if completion.get("applicable") is False:
            if completion.get("score") is not None:
                raise ValueError("structural_completion_has_score")
            expected_score = quality
        else:
            fraction = completion.get("score")
            if not (_finite_score(fraction) and completion.get("evidence_ids")):
                raise ValueError("completion_evidence_missing")
            expected_score = min(quality, fraction)
        if safety["hard_failure"]:
            expected_score = 0.0
        if not _close(score, expected_score):
            raise ValueError("joint_score_formula_mismatch")
        interval = mission.get("score_interval")
        if not (
            isinstance(interval, list)
            and len(interval) == 2
            and all(_finite_score(value) for value in interval)
            and interval[0] <= score <= interval[1]
        ):
            raise ValueError("joint_score_interval_invalid")
        weight = weights[case]
        scored[model] += 1
        model_totals[model] += weight * score
        cap_totals[model] += weight * (quality - score)
    if set(rows) != set(old_rows):
        raise ValueError("selected_episode_set_changed")
    for item in report["models"]:
        model = item["model"]
        complete = scored[model] == len(weights)
        if item["n_scored"] != scored[model] or item["complete"] is not complete:
            raise ValueError("model_coverage_mismatch")
        if complete:
            if not (
                _finite_score(item["primary_score"])
                and _close(item["primary_score"], model_totals[model])
                and _close(
                    item["native_quality_025"] - item["primary_score"],
                    cap_totals[model],
                )
            ):
                raise ValueError("model_aggregate_mismatch")
        elif item["primary_score"] is not None or item["primary_rank"] is not None:
            raise ValueError("partial_model_has_full_index")
    complete_models = [item for item in report["models"] if item["complete"]]
    ordered = sorted(
        complete_models, key=lambda item: (-item["primary_score"], item["model"])
    )
    previous = None
    rank = None
    for index, item in enumerate(ordered, 1):
        if item["primary_score"] != previous:
            rank = index
        if item["primary_rank"] != rank:
            raise ValueError("model_rank_mismatch")
        previous = item["primary_score"]
    return {
        "selected_rows": len(rows),
        "scored_rows": sum(scored.values()),
        "complete_models": len(complete_models),
        "incomplete_models": len(report["models"]) - len(complete_models),
    }


def audit(
    *,
    report_path: Path,
    suite_path: Path,
    policy_path: Path = DEFAULT_POLICY,
    core_path: Path | None = None,
    root: Path = ROOT,
) -> tuple[dict, list[dict]]:
    report_raw = report_path.read_bytes()
    report = json.loads(report_raw)
    suite_raw = suite_path.read_bytes()
    suite = json.loads(suite_raw)
    if not policy_path.is_absolute():
        policy_path = root / policy_path
    policy_raw = policy_path.read_bytes()
    policy = json.loads(policy_raw)
    if (
        report.get("evaluation_version") != EXPECTED_VERSION
        or report.get("protocol_revision") != EXPECTED_REVISION
        or report.get("n_expected") != EXPECTED_CASES
        or report.get("completion_applicable_cases") != EXPECTED_COMPLETION_CASES
        or report.get("native_economic_only_cases") != EXPECTED_ECONOMIC_CASES
        or report.get("strict_attainment_status") != "not_calibrated"
    ):
        raise ValueError("unexpected_scoring_protocol")
    if (
        suite.get("n_scenarios") != EXPECTED_CASES
        or len(suite.get("scenarios") or []) != EXPECTED_CASES
        or report["source_contracts"].get("suite_sha256") != _sha256(suite_raw)
        or report["task_weight_manifest"].get("suite_sha256") != _sha256(suite_raw)
    ):
        raise ValueError("lite_suite_identity_mismatch")
    validate_lite_policy(policy, report, suite, suite_raw)
    source_ref = report["source_report"]
    source_path = Path(source_ref["path"])
    if not source_path.is_absolute():
        source_path = root / source_path
    source_raw = source_path.read_bytes()
    if _sha256(source_raw) != source_ref["sha256"]:
        raise ValueError("source_report_hash_mismatch")
    source = json.loads(source_raw)
    if source.get("evaluation_version") != "0.25.0":
        raise ValueError("source_report_version_mismatch")
    if report["source_contracts"] != source.get("source_contracts") or report[
        "canonical_models"
    ] != source.get("canonical_models"):
        raise ValueError("source_contract_or_model_roster_changed")
    model_ids = [item["model"] for item in report["models"]]
    if (
        len(model_ids) != len(set(model_ids))
        or set(model_ids) != set(report["canonical_models"])
        or any(item["n_expected"] != EXPECTED_CASES for item in report["models"])
    ):
        raise ValueError("model_roster_or_denominator_mismatch")
    if _sha256((root / "scripts/evaluate_completion026.py").read_bytes()) != report.get(
        "scoring_entry_sha256"
    ):
        raise ValueError("scoring_entry_hash_mismatch")
    for path, digest in report["scoring_modules_sha256"].items():
        if _sha256((root / path).read_bytes()) != digest:
            raise ValueError("scoring_module_hash_mismatch")
    ledger_ref = report["score_audit_ledger"]
    ledger_path = report_path.parent / ledger_ref["path"]
    ledger_raw = ledger_path.read_bytes()
    if _sha256(ledger_raw) != ledger_ref["sha256"]:
        raise ValueError("score_ledger_hash_mismatch")
    if len(ledger_raw.splitlines()) != ledger_ref["rows"]:
        raise ValueError("score_ledger_row_count_mismatch")
    from evaluation.operational_weights import validate_task_weights

    weights = validate_task_weights(
        report["task_weight_manifest"], report["source_contracts"]
    )
    if len(weights) != EXPECTED_CASES:
        raise ValueError("weight_denominator_mismatch")
    case_keys = {(row["scenario_signature"], row["seed"]) for row in suite["scenarios"]}
    if case_keys != set(weights):
        raise ValueError("suite_case_set_mismatch")
    suite_rows = {
        (row["scenario_signature"], row["seed"]): row for row in suite["scenarios"]
    }
    source_contracts = {
        (row["scenario_signature"], row["seed"]): row
        for row in report["source_contracts"]["contracts"]
    }
    if set(source_contracts) != case_keys or any(
        source_contracts[key]["scenario_sha256"] != suite_rows[key]["yaml_sha256"]
        or source_contracts[key]["scenario_path"] != suite_rows[key]["path"]
        for key in case_keys
    ):
        raise ValueError("source_contract_suite_case_mismatch")
    completion_contracts = report["completion_contracts"]
    if (
        len(completion_contracts) != EXPECTED_CASES
        or {(row["scenario_signature"], row["seed"]) for row in completion_contracts}
        != case_keys
    ):
        raise ValueError("completion_contract_coverage_mismatch")
    from evaluation.mission_contracts import _digest
    from evaluation.operational_completion import compile_completion_contract

    if (
        _digest({"contracts": completion_contracts})
        != report["completion_contracts_sha256"]
    ):
        raise ValueError("completion_contract_digest_mismatch")
    for contract in completion_contracts:
        key = contract["scenario_signature"], contract["seed"]
        raw = (root / source_contracts[key]["scenario_path"]).read_bytes()
        if _sha256(raw) != source_contracts[key]["scenario_sha256"]:
            raise ValueError("source_scenario_hash_mismatch")
        expected = compile_completion_contract(
            yaml.safe_load(raw), source_contracts[key]
        )
        if contract != expected:
            raise ValueError("completion_contract_source_mismatch")
    ledger = [json.loads(line) for line in ledger_raw.splitlines()]
    ledger_rows = {
        (row["model"], row["scenario_signature"], row["seed"]): row for row in ledger
    }
    if len(ledger_rows) != len(ledger):
        raise ValueError("duplicate_ledger_row")
    report_row_ids = {
        (row["model"], row["scenario_signature"], row["seed"])
        for row in report["graded_episodes"]
    }
    if set(ledger_rows) != report_row_ids:
        raise ValueError("ledger_episode_set_mismatch")
    for row in report["graded_episodes"]:
        identity = row["model"], row["scenario_signature"], row["seed"]
        entry = ledger_rows.get(identity)
        if entry is None or any(
            entry[field] != expected
            for field, expected in (
                ("score_026", row["mission"]["score"]),
                ("completion_score", row["completion"]["score"]),
                ("native_quality_025", row["mission"]["native_quality_025"]),
                ("completion_contract_sha256", row["completion"]["contract_sha256"]),
            )
        ):
            raise ValueError("score_ledger_case_mismatch")
    checks = verify_case_rows(report, source, weights)
    lite_backends = {row["backend_kind"] for row in suite["scenarios"]}
    core_extension = {"status": "not_evaluated_optional", "blocking_lite": False}
    core_digest = None
    if core_path is not None:
        core_raw = core_path.read_bytes()
        core = json.loads(core_raw)
        core_backends = {row["backend_kind"] for row in core["scenarios"]}
        core_digest = _sha256(core_raw)
        core_extension = {
            "status": "separate_full_track_not_scored_by_this_lite_report",
            "blocking_lite": False,
            "case_count": len(core["scenarios"]),
            "backends_without_0261_lite_adapter": sorted(core_backends - lite_backends),
        }
    historical_flags = {
        "suite_formal_full_leaderboard_eligible": suite.get(
            "formal_full_leaderboard_eligible"
        ),
        "scoring_report_formal_run_certified": report.get("formal_run_certified"),
        "scoring_report_leaderboard_eligible": report.get("leaderboard_eligible"),
        "retrospective_only": report.get("retrospective_only"),
    }
    if checks["complete_models"] < 1:
        raise ValueError("no_complete_lite_model")
    summary = {
        "schema_version": "operate_offline_main_table_audit.v2",
        "scope": "Lite141 offline main leaderboard; model-informed development panel",
        "evaluation_version": EXPECTED_VERSION,
        "protocol_revision": EXPECTED_REVISION,
        "score_formula": "hard_zero_else_casewise_min_source_fulfillment_and_native_quality;citylearn_native_only",
        "weight_policy": report["task_weight_manifest"]["weight_policy"],
        "task_weight_manifest_sha256": report["task_weight_manifest"][
            "manifest_sha256"
        ],
        "input_hashes": {
            "suite": _sha256(suite_raw),
            "lite_main_policy": _sha256(policy_raw),
            "optional_core_suite": core_digest,
            "source_025_report": _sha256(source_raw),
            "scored_0261_report": _sha256(report_raw),
            "score_ledger": _sha256(ledger_raw),
            "scoring_entry": report["scoring_entry_sha256"],
            "scoring_modules": report["scoring_modules_sha256"],
            "audit_entry": _sha256(
                (root / "scripts/audit_offline_main_table026.py").read_bytes()
            ),
        },
        "checks": checks,
        "status": {
            "lite_main_scoring_release_ready": True,
            "public_distribution_performed": False,
            "core_full_extension": core_extension,
            "historical_core_formal_flags": historical_flags,
        },
        "interpretation_limits": [
            "model_informed_lite_development_selection_not_unbiased_holdout",
            "index_not_binary_mission_attainment",
            "one_trajectory_per_case_not_repeat_reliability",
            "gpu_sla_penalizes_queued_jobs_past_procedural_due_tick_not_job_completion_deadline",
            "routing_uses_dispatch_wave_not_source_native_time_windows",
            "sumo_recovery_uses_terminal_supervisory_tick_modes_not_all_physics_substeps",
            "completion_metrics_heterogeneous_by_backend",
        ],
    }
    table = []
    aliases = report["canonical_models"]
    for model in sorted(
        report["models"],
        key=lambda item: (
            item["primary_score"] is None,
            -(item["primary_score"] or 0),
            aliases[item["model"]],
        ),
    ):
        table.append(
            {
                "model": aliases[model["model"]],
                "coverage": f"{model['n_scored']}/{model['n_expected']}",
                "index_0261": model["primary_score"],
                "rank_0261": model["primary_rank"],
                "native_quality_025": model["native_quality_025"],
                "completion_applicable_131": model["source_completion"]["score"],
                "hard_failure_rate": model["hard_failure_rate"],
                "completion_cap_reduction": model["completion_cap_score_reduction"],
            }
        )
    return summary, table


def render(summary: dict, rows: list[dict]) -> str:
    def number(value: object) -> str:
        return "N/A" if value is None else f"{value:.2f}"

    lines = [
        "# OPERATE Lite141 0.26.1 offline main table",
        "",
        "Lite141 is the release denominator; Core Full is an optional separate extension.",
        "The Lite panel was selected using development-model outcomes, so it is not an unbiased holdout.",
        "These scores retrospectively re-evaluate existing trajectories; public distribution has not occurred here.",
        "The 0.26.1 index is native operational quality capped by source-obligation progress",
        "where measurable. It is not a mission-pass percentage.",
        "",
        "| Model | Coverage | 0.26.1 rank | 0.26.1 Index | 0.25 quality | F on 131 cases | Hard failure % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['coverage']} | {row['rank_0261'] or '—'} "
            f"| {number(row['index_0261'])} | {number(row['native_quality_025'])} "
            f"| {number(row['completion_applicable_131'])} "
            f"| {number(row['hard_failure_rate'])} |"
        )
    lines += ["", "Lite main scoring release gate: PASS", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--suite", type=Path, default=Path("release/operate_v0_62_0/lite_suite.json")
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--core", type=Path, help="Optional Core Full scope diagnostic")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-lite-main", action="store_true")
    parser.add_argument("--require-formal-core", action="store_true")
    args = parser.parse_args()
    summary, rows = audit(
        report_path=args.report,
        suite_path=args.suite,
        policy_path=args.policy,
        core_path=args.core,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "table.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output_dir / "table.md").write_text(render(summary, rows))
    print(render(summary, rows))
    if (
        args.require_lite_main
        and not summary["status"]["lite_main_scoring_release_ready"]
    ):
        raise SystemExit("lite_main_scoring_release_gate_failed")
    if args.require_formal_core:
        raise SystemExit("optional_core_full_requires_separate_score_report")


if __name__ == "__main__":
    main()
