#!/usr/bin/env python3
"""Offline 0.26 source-obligation outcomes from a frozen 0.25 native report.

No agent, backend, reference controller, or simulator is executed. The input
report fixes the trajectory selection and authenticates each native artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.mission_contracts import _digest as contract_digest, _read_locked  # noqa: E402
from evaluation.native_scorecard import _read  # noqa: E402
from evaluation.operational_completion import (  # noqa: E402
    compile_completion_contract,
    measure_completion,
    score_joint_outcome,
)
from evaluation.operational_contracts import compile_operational_suite  # noqa: E402
from evaluation.operational_reporting import aggregate_native_outcome  # noqa: E402
from evaluation.operational_service import (  # noqa: E402
    SERVICE_BACKENDS,
    compile_service_contract,
)
from evaluation.operational_weights import validate_task_weights  # noqa: E402


VERSION = "0.26.1"
REVISION = "source_obligation_outcome.v2"


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _valid_score(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 100


def _authenticated_service(row: dict, scenario: dict, contract: dict) -> dict:
    """Reuse the 0.25 service result already bound to native evidence."""
    service = row["mission"].get("service") or {}
    expected = compile_service_contract(scenario, source_contract=contract)
    numerator, denominator = service.get("numerator"), service.get("denominator")
    if not (
        service.get("applicable") is True
        and service.get("evidence_ids")
        and _valid_score(service.get("score"))
        and type(numerator) in (int, float)
        and math.isfinite(numerator)
        and type(denominator) in (int, float)
        and math.isfinite(denominator)
        and denominator > 0
        and denominator == expected["denominator"]
        and 0 <= numerator <= denominator + max(1e-6, denominator * 1e-6)
        and math.isclose(
            service["score"],
            100 * numerator / denominator,
            rel_tol=0,
            abs_tol=1e-6,
        )
    ):
        return {
            "applicable": True,
            "score": None,
            "reason": "source_service_evidence_inconsistent",
        }
    return service


def _weighted_completion(
    rows: list[dict], weights: dict, applicable: set, models: list[str]
) -> dict:
    mass = math.fsum(weights[key] for key in applicable)
    by_model = {}
    for row in rows:
        key = row["scenario_signature"], row["seed"]
        if key in applicable:
            by_model.setdefault(row["model"], {})[key] = row["completion"]
    result = {}
    for model in models:
        values = by_model.get(model, {})
        valid = {
            key: item["score"]
            for key, item in values.items()
            if _valid_score(item["score"])
        }
        result[model] = {
            "score": math.fsum(weights[key] * valid[key] for key in applicable) / mass
            if len(valid) == len(applicable)
            else None,
            "n_scored": len(valid),
            "n_expected": len(applicable),
            "frozen_weight_mass": mass,
            "interpretation": "source_obligation_fulfillment_not_binary_task_attainment",
        }
    return result


def evaluate(source_report: dict, *, suite_path: Path, root: Path = ROOT) -> dict:
    """Rescore selected episodes; source obligations and weights are model-free."""
    if (
        source_report.get("evaluation_version") != "0.25.0"
        or source_report.get("protocol_revision") != "source_grounded_native_outcome.v3"
        or source_report.get("scoring_mode") != "native_outcome"
    ):
        raise ValueError("unsupported_source_native_report")
    compiled = compile_operational_suite(suite_path, root=root)
    reported = source_report["source_contracts"]
    if compiled["suite_sha256"] != reported.get("suite_sha256"):
        raise ValueError("source_suite_hash_mismatch")
    contracts = {(c["scenario_signature"], c["seed"]): c for c in compiled["contracts"]}
    reported_contracts = {
        (c["scenario_signature"], c["seed"]): c for c in reported["contracts"]
    }
    if (
        len(contracts) != len(reported_contracts)
        or contracts.keys() != reported_contracts.keys()
    ):
        raise ValueError("source_contract_coverage_mismatch")
    for key, contract in contracts.items():
        previous = reported_contracts[key]
        for field in ("scenario_sha256", "scenario_path", "backend_kind", "service"):
            if contract[field] != previous[field]:
                raise ValueError("source_contract_changed")
    manifest = source_report["task_weight_manifest"]
    if (
        manifest.get("model_outcomes_used") is not False
        or manifest.get("weight_policy")
        != "equal_domain_family_physical_source_then_case"
    ):
        raise ValueError("nonindependent_task_weight_policy")
    weights = validate_task_weights(manifest, compiled)
    models = list(source_report["canonical_models"])
    if len(models) != len(set(models)) or not models:
        raise ValueError("invalid_model_population")
    selected = source_report["graded_episodes"]
    seen = set()
    scenarios = {}
    completion_contracts = {}
    for key, contract in contracts.items():
        scenarios[key] = yaml.safe_load(
            _read_locked(
                root,
                {
                    "path": contract["scenario_path"],
                    "sha256": contract["scenario_sha256"],
                },
                label="scenario",
            )
        )
        completion_contracts[key] = compile_completion_contract(
            scenarios[key], contract
        )
    output = []
    for original in selected:
        key = original["scenario_signature"], original["seed"]
        identity = original["model"], *key
        if identity in seen or key not in contracts or identity[0] not in models:
            raise ValueError("duplicate_or_foreign_selected_episode")
        seen.add(identity)
        row = deepcopy(original)
        mission = row["mission"]
        binding = row["artifact_binding"]
        backend = scenarios[key]["backend_kind"]
        completion = (
            measure_completion(
                scenarios[key],
                source_contract=contracts[key],
                snapshot_inputs={},
                trace=[],
                root=root,
            )
            if backend == "citylearn"
            else {
                "applicable": True,
                "score": None,
                "reason": "source_native_score_missing",
            }
        )
        quality = mission.get("score")
        quality_interval = mission.get("score_interval")
        safety = mission.get("safety") or {}
        eligible = (
            _valid_score(quality)
            and binding.get("verified") is True
            and binding.get("native_cost_bound") is True
            and safety.get("verified") is True
            and type(safety.get("hard_failure")) is bool
        )
        if eligible and backend in SERVICE_BACKENDS and backend != "sumo_ego":
            service = _authenticated_service(row, scenarios[key], contracts[key])
            diagnostic_trace = (
                _read(binding, "trajectory_artifact")
                if backend == "dynasched_flexible_job_shop"
                else []
            )
            if diagnostic_trace and len(diagnostic_trace) != binding.get("trace_ticks"):
                raise ValueError("fjsp_job_diagnostic_trace_window_mismatch")
            completion = measure_completion(
                scenarios[key],
                source_contract=contracts[key],
                snapshot_inputs={},
                trace=diagnostic_trace,
                existing_service=service,
                root=root,
            )
        elif eligible and backend != "citylearn":
            inputs = _read(binding, "scoring_inputs_artifact")["payload"]["inputs"]
            trace = _read(binding, "trajectory_artifact")
            ledger = _read(binding, "evidence_ledger_artifact")
            evidence = {item.get("evidence_id") for item in ledger}
            if (
                inputs.get("scenario_signature") != key[0]
                or len(trace) != binding.get("trace_ticks")
                or not set(mission.get("native_evidence_ids") or []) <= evidence
            ):
                raise ValueError("selected_native_evidence_mismatch")
            service = (
                _authenticated_service(row, scenarios[key], contracts[key])
                if backend == "sumo_ego"
                else None
            )
            completion = measure_completion(
                scenarios[key],
                source_contract=contracts[key],
                snapshot_inputs=inputs,
                trace=trace,
                existing_service=service,
                root=root,
            )
        outcome = score_joint_outcome(
            completion,
            quality,
            quality_interval,
            hard_failure=eligible and safety["hard_failure"],
        )
        mode = (
            "native_economic_quality_only"
            if completion["applicable"] is False
            else "completion_capped_native_quality"
        )
        mission.update(outcome)
        completion["completion_kind"] = completion_contracts[key]["completion_kind"]
        completion["contract_sha256"] = completion_contracts[key]["contract_sha256"]
        if _valid_score(completion.get("score")):
            completion["evidence_ids"] = (
                service["evidence_ids"]
                if backend in SERVICE_BACKENDS
                else mission.get("native_evidence_ids") or []
            )
        mission.update(
            protocol_revision=REVISION,
            interpretation="noncompensatory_completion_and_native_quality",
            native_quality_025=quality,
            completion=completion,
            scoring_mode_026=mode,
        )
        row["completion"] = completion
        output.append(row)

    scored = aggregate_native_outcome(
        output, reported["contracts"], weights, models=models
    )
    applicable = {
        key
        for key, contract in contracts.items()
        if contract["service"]["kind"] != "automatic_exogenous_load_service"
    }
    if not applicable:
        raise ValueError("no_source_completion_cases")
    completion_only = _weighted_completion(output, weights, applicable, models)
    old = {item["model"]: item for item in source_report["models"]}
    caps = {model: [] for model in models}
    cap_contributions = {model: [] for model in models}
    for row in output:
        completion = row["completion"]
        quality = row["mission"]["native_quality_025"]
        if (
            completion["applicable"] is True
            and _valid_score(completion.get("score"))
            and _valid_score(quality)
            and completion["score"] + 1e-9 < quality
            and row["mission"]["safety"]["hard_failure"] is False
        ):
            caps[row["model"]].append((row["scenario_signature"], row["seed"]))
            key = row["scenario_signature"], row["seed"]
            cap_contributions[row["model"]].append(
                weights[key] * (quality - completion["score"])
            )
    for item in scored["models"]:
        model = item["model"]
        item["source_completion"] = completion_only[model]
        item["native_quality_025"] = old[model]["primary_score"]
        item["completion_cap_binding_cases"] = len(caps[model])
        item["completion_cap_binding_weight_mass"] = math.fsum(
            weights[key] for key in caps[model]
        )
        item["completion_cap_score_reduction"] = (
            math.fsum(cap_contributions[model]) if item["complete"] else None
        )
        if item["complete"] and not math.isclose(
            item["native_quality_025"] - item["primary_score"],
            item["completion_cap_score_reduction"],
            abs_tol=1e-8,
            rel_tol=0,
        ):
            raise ValueError("completion_cap_aggregate_identity_failed")
    scored.update(
        evaluation_version=VERSION,
        protocol_revision=REVISION,
        primary_interpretation="noncompensatory_completion_and_native_quality_with_economic_only_cases",
        primary_is_binary_task_attainment=False,
        strict_attainment_status="not_calibrated",
        completion_applicable_cases=len(applicable),
        native_economic_only_cases=len(contracts) - len(applicable),
        completion_only=completion_only,
        completion_contracts=[completion_contracts[key] for key in sorted(contracts)],
        completion_contracts_sha256=contract_digest(
            {"contracts": [completion_contracts[key] for key in sorted(contracts)]}
        ),
        source_contracts=reported,
        task_weight_manifest=source_report["task_weight_manifest"],
        graded_episodes=output,
        canonical_models=source_report["canonical_models"],
        comparison_policy=source_report.get("comparison_policy"),
        input_manifest_sha256=source_report.get("input_manifest_sha256"),
        outcome_modes=dict(Counter(r["mission"]["scoring_mode_026"] for r in output)),
        execution_policy={
            "provider_calls": 0,
            "environment_replays": 0,
            "solver_calls": 0,
            "model_outcomes_used_for_calibration": False,
        },
        retrospective_only=True,
    )
    return scored


def render(report: dict) -> str:
    def number(value):
        return "N/A" if value is None else f"{value:.2f}"

    lines = [
        "# OPERATE 0.26 离线任务结果评分",
        "",
        "主分：131 题取源任务进度／服务结果与 0.25 结算原生质量的较低者；SUMO 采用源契约规定的末段稳定恢复，不再以跑完全路线为完成条件。10 题 CityLearn 仅沿用结算原生经济质量。硬失败归零；这不是 141 题的纯完成率或二元通过率。",
        "",
        "|模型|排名|0.26 任务结果分|完成度 131 题|0.25 原生质量|覆盖|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in sorted(
        report["models"],
        key=lambda x: (
            x["primary_score"] is None,
            -(x["primary_score"] or 0),
            x["model"],
        ),
    ):
        lines.append(
            f"|{report['canonical_models'][item['model']]}|{item['primary_rank'] or '—'}|"
            f"{number(item['primary_score'])}|{number(item['source_completion']['score'])}|"
            f"{number(item['native_quality_025'])}|{item['n_scored']}/{item['n_expected']}|"
        )
    lines.extend(
        [
            "",
            "分母包括初始义务、源中未来到达工作与窗口内外生扰动；取消、拒收或延迟不缩小分母。完整分数只对固定 141 题都有有效证据的模型发布。",
            "",
        ]
    )
    return "\n".join(lines)


def score_audit_ledger(report: dict):
    """Expose the exact task-local cap responsible for each 0.25→0.26 change."""
    weights = {
        (c["scenario_signature"], c["seed"]): c
        for c in report["task_weight_manifest"]["cases"]
    }
    for row in report["graded_episodes"]:
        key = row["scenario_signature"], row["seed"]
        task = weights[key]
        mission = row["mission"]
        completion = row["completion"]
        quality = mission.get("native_quality_025")
        fraction = completion.get("score")
        score = mission.get("score")
        cap = (
            max(0.0, quality - fraction)
            if _valid_score(quality)
            and _valid_score(fraction)
            and (mission.get("safety") or {}).get("hard_failure") is False
            else 0.0
            if _valid_score(score)
            else None
        )
        artifacts = (row.get("artifact_binding") or {}).get("artifacts") or {}
        yield {
            "model": row["model"],
            "scenario_signature": key[0],
            "seed": key[1],
            "backend_kind": task["backend_kind"],
            "task_family": task["task_family"],
            "physical_source_cluster": task["physical_source_cluster"],
            "score_weight": task["score_weight"],
            "completion_kind": completion["completion_kind"],
            "completion_contract_sha256": completion["contract_sha256"],
            "completion_score": fraction,
            "job_completion_diagnostic": completion.get("job_completion_diagnostic"),
            "route_progress_diagnostic": completion.get("route_progress_diagnostic"),
            "terminal_nominal_ticks": completion.get("terminal_nominal_ticks"),
            "native_mrm_observed": completion.get("mrm_observed"),
            "guarded_recovery_required": completion.get("recovery_required"),
            "native_cost": mission.get("native_objective"),
            "source_scale": mission.get("source_scale"),
            "native_quality_025": quality,
            "score_026": score,
            "verified_hard_failure": (mission.get("safety") or {}).get("hard_failure"),
            "cap_reduction": cap,
            "weighted_cap_reduction": task["score_weight"] * cap
            if cap is not None
            else None,
            "completion_evidence_ids": completion.get("evidence_ids") or [],
            "native_evidence_ids": mission.get("native_evidence_ids") or [],
            "trajectory_sha256": (artifacts.get("trajectory_artifact") or {}).get(
                "sha256"
            ),
            "native_snapshot_sha256": (
                artifacts.get("scoring_inputs_artifact") or {}
            ).get("sha256"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument(
        "--suite", type=Path, default=Path("release/operate_v0_62_0/lite_suite.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source_report.read_bytes()
    report = evaluate(json.loads(raw), suite_path=args.suite, root=ROOT)
    report["source_report"] = {"path": str(args.source_report), "sha256": _digest(raw)}
    report["scoring_entry_sha256"] = _digest(Path(__file__).read_bytes())
    report["scoring_modules_sha256"] = {
        path: _digest((ROOT / path).read_bytes())
        for path in (
            "evaluation/operational_completion.py",
            "evaluation/operational_service.py",
            "evaluation/operational_reporting.py",
            "evaluation/operational_weights.py",
        )
    }
    ledger = "".join(
        json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
        for item in score_audit_ledger(report)
    ).encode()
    report["score_audit_ledger"] = {
        "path": "score_audit_ledger.jsonl",
        "sha256": _digest(ledger),
        "rows": len(report["graded_episodes"]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "score_audit_ledger.jsonl").write_bytes(ledger)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output_dir / "report.md").write_text(render(report))
    print(render(report))


if __name__ == "__main__":
    main()
