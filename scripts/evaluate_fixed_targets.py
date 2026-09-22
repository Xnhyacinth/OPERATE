#!/usr/bin/env python3
"""Rescore bound historical episodes against a frozen fixed-target suite."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.fixed_targets import TARGET_POLICIES, score_target  # noqa: E402
from evaluation.native_objectives import extract_native_objective  # noqa: E402
from evaluation.target_attainment import aggregate_target_attainment  # noqa: E402
from evaluation.task_quality import FJSP, evaluate_task_quality  # noqa: E402
from scripts.evaluate_native_quality import evaluate as evaluate_legacy  # noqa: E402
from scripts.evaluate_operate_quality import _execution_usable, _input_rows  # noqa: E402


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_targets(descriptor, suite_sha256, specs):
    if not isinstance(descriptor, dict):
        raise ValueError("fixed_targets hash-bound descriptor required")
    raw = Path(descriptor["path"]).read_bytes()
    if not descriptor.get("sha256") or digest(raw) != descriptor["sha256"]:
        raise ValueError("fixed target hash missing or mismatched")
    payload = json.loads(raw)
    if (
        payload.get("schema_version") != "fixed_target_suite.v1"
        or payload.get("suite_sha256") != suite_sha256
    ):
        raise ValueError("fixed target suite identity mismatch")
    if not isinstance(payload.get("targets"), list):
        raise ValueError("fixed targets must be a list")
    targets = {}
    for target in payload["targets"]:
        key = (target.get("scenario_signature"), target.get("seed"))
        if key not in specs or key in targets:
            raise ValueError("duplicate or outside fixed target")
        if target.get("backend_kind") != specs[key]["backend_kind"]:
            raise ValueError("fixed target backend mismatch")
        if target.get("status") == "ready":
            if target.get("schema_version") != "fixed_expert_target.v1" or target.get(
                "candidate_policies"
            ) != list(TARGET_POLICIES):
                raise ValueError("fixed target protocol mismatch")
            if any(
                target.get(k) != specs[key].get(k) for k in ("domain", "horizon_ticks")
            ) or target.get("source_yaml_sha256") != specs[key].get("yaml_sha256"):
                raise ValueError("fixed target source identity mismatch")
            encoded = json.dumps(
                {k: v for k, v in target.items() if k != "contract_sha256"},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            if (
                not target.get("contract_sha256")
                or digest(encoded) != target["contract_sha256"]
            ):
                raise ValueError("fixed target contract hash mismatch")
        targets[key] = target
    return targets, payload


def evaluate(config):
    suite_raw = Path(config["suite"]).read_bytes()
    suite = json.loads(suite_raw)["scenarios"]
    specs = {(s["scenario_signature"], s["seed"]): s for s in suite}
    if len(specs) != len(suite):
        raise ValueError("duplicate suite case")
    targets, target_payload = load_targets(
        config.get("fixed_targets"), digest(suite_raw), specs
    )
    inputs = _input_rows(config)
    legacy = evaluate_legacy(
        {
            **config,
            "references": [],
            "reference_compatibility": [],
            "require_input_hashes": True,
        }
    )
    measurements, details = [], []
    for label, originals in inputs.items():
        group = legacy["by_model"][label]
        bound = {
            (r["scenario_signature"], r["seed"]): r for r in group["graded_episodes"]
        }
        for original in originals:
            key = (original.get("scenario_signature"), original.get("seed"))
            if key not in specs:
                if config.get("select_suite_members") is True:
                    continue
                raise ValueError("input case outside fixed suite")
            spec, old = specs[key], bound[key]
            row = {**spec, **original}
            binding = old["artifact_binding"]
            native = extract_native_objective(row)
            target = targets.get(key, {})
            blocker = group.get("identity_blocker")
            if not blocker and (
                binding.get("verified") is not True
                or binding.get("native_cost_bound") is not True
            ):
                blocker = "unbound_episode_artifacts"
            if not blocker and not _execution_usable(row, legacy["comparison_policy"]):
                blocker = "execution_identity_unproven_or_changed"
            if not blocker and any(
                row.get(k) != spec[k] for k in ("domain", "backend_kind")
            ):
                blocker = "episode_backend_mismatch"
            if (
                not blocker
                and legacy["comparison_policy"] == "strict"
                and target.get("runtime_identity")
                != row.get("implementation_tree_sha256")
            ):
                blocker = "fixed_target_runtime_identity_mismatch"
            if blocker:
                result = dict(
                    score=None, attained=None, evidence_ids=[], reason=blocker
                )
            else:
                result = score_target(native, target)
                if row.get("backend_kind") == FJSP and result.get("score") is not None:
                    completion = evaluate_task_quality(
                        row, artifact_binding=binding, reference_contract=None
                    )
                    if completion.get("mandatory_complete") is False and completion.get(
                        "evidence_ids"
                    ):
                        result.update(
                            score=0.0,
                            attained=False,
                            reason="incomplete_absolute_obligations",
                            evidence_ids=list(
                                dict.fromkeys(
                                    result["evidence_ids"] + completion["evidence_ids"]
                                )
                            ),
                        )
                    elif (
                        completion.get("mandatory_complete") is not True
                        and native.get("hard_failure") is not True
                    ):
                        result.update(
                            score=None,
                            attained=None,
                            reason=completion.get(
                                "reason", "completion_evidence_missing"
                            ),
                        )
                    result["absolute_completion"] = completion
            safety = dict(
                verified=not bool(blocker) and native.get("applicable") is True,
                hard_failure=native.get("hard_failure"),
                evidence_ids=native.get("evidence_ids", []),
            )
            scored = dict(
                model=label,
                scenario_signature=key[0],
                seed=key[1],
                measurement=result,
                safety=safety,
            )
            measurements.append(scored)
            details.append(
                {
                    **scored,
                    "native_measurement": native,
                    "fixed_target": target,
                    "artifact_binding": binding,
                    "execution_provenance": old.get("execution_provenance"),
                    "source_episode_path": original.get("_offline_source_path"),
                }
            )
    report = aggregate_target_attainment(measurements, suite, models=list(inputs))
    report.update(
        suite_sha256=digest(suite_raw),
        fixed_targets=config["fixed_targets"],
        target_provenance=target_payload.get("target_provenance", []),
        target_selection_policy=target_payload.get("selection_policy"),
        comparison_policy=legacy["comparison_policy"],
        compatibility_user_assumed=legacy["compatibility_user_assumed"],
        observed_execution_identities={
            m: g.get("observed_execution_identities")
            for m, g in legacy["by_model"].items()
        },
        episode_inputs=legacy["episode_inputs"],
        graded_episodes=details,
        formal_run_certified=False,
        leaderboard_eligible=False,
        semantics="fixed_expert_target_attainment_not_optimality_or_original_task_success_or_general_agency",
        evaluation_scope="retrospective_development_suite_with_frozen_model_independent_targets",
        reason_counts=dict(Counter(r["measurement"].get("reason") for r in details)),
    )
    return report


def render_summary(report):
    lines = [
        "# Fixed-target attainment 0.23.1",
        "",
        "Fixed expert targets, not certified optima, original-task success rates, or a general agency score.",
        "Retrospective development-set evaluation; not a formal public leaderboard.",
        "",
        "| Model | Rank | Macro attainment % | Passed | Scored / expected |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["leaderboard"] + report["incomplete_models"]:
        score = "N/A" if row["score"] is None else f"{row['score']:.2f}"
        lines.append(
            f"| {row['model']} | {row['rank'] or '—'} | {score} | {row['n_passed']} | {row['n_scored']}/{row['n_expected']} |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if ".hl" not in args.output.resolve().parts:
        parser.error("output must be under .hl")
    if args.output.exists():
        parser.error("output already exists; frozen results cannot be overwritten")
    raw = args.manifest.read_bytes()
    entry_hash = digest(Path(__file__).read_bytes())
    from core.implementation_identity import implementation_identity

    before = implementation_identity(ROOT)
    report = evaluate(json.loads(raw))
    after = implementation_identity(ROOT)
    if before["evaluation_runtime_sha256"] != after[
        "evaluation_runtime_sha256"
    ] or entry_hash != digest(Path(__file__).read_bytes()):
        raise RuntimeError("fixed target evaluator changed during scoring")
    report.update(
        input_manifest_sha256=digest(raw),
        scoring_implementation=before,
        scoring_entry_sha256=entry_hash,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    (args.output / "report.md").write_text("\n".join(render_summary(report)) + "\n")
    print(
        json.dumps(
            {"output": str(args.output), "complete_models": len(report["leaderboard"])}
        )
    )


if __name__ == "__main__":
    main()
