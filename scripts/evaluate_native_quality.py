#!/usr/bin/env python3
"""Score existing episodes with authenticated fixed-policy native references.

No provider calls, historical rewrites, or automatic reference-policy tuning.
Paths in the input manifest are relative to the working directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.native_objectives import extract_native_objective  # noqa: E402
from evaluation.capability_report import (  # noqa: E402
    build_capability_report,
    build_long_task_report,
)
from evaluation.capability_evidence import bind_capability_evidence  # noqa: E402
from evaluation.native_quality import (  # noqa: E402
    VERSION,
    aggregate_native_rows,
    score_native_quality,
)  # noqa: E402
from evaluation.native_reference import load_reference_report  # noqa: E402
from evaluation.reference_compatibility import verify_reference_compatibility  # noqa: E402
from tools.lite_trajectory_report import _scope, select_latest_attempts  # noqa: E402

EVALUATION_VERSION = "0.22.0"
USER_ASSUMED_POLICY = "latest_framework_user_assumed"


def grade_bound_episode(row, spec, contracts, source_path, comparison_policy="strict"):
    native = grade_episode(row, spec, contracts, comparison_policy)
    capability = build_capability_report({**spec, **row}, scenario_spec=spec)
    binding = bind_capability_evidence(
        row, spec, source_path=source_path, comparison_policy=comparison_policy
    )
    if binding["verified"] is not True:
        if native.get("score") is not None:
            native = {
                "score": None,
                "reason": "unbound_episode_artifacts",
                "artifact_binding_reason": binding["reason"],
            }
        if "native_outcome" in capability:
            capability["native_outcome"].update(
                applicable=False,
                actual_cost=None,
                feasible=None,
                hard_failure=None,
                task_success=None,
                reason="unbound_episode_artifacts",
            )
        agency = capability.get("operational_agency")
        if agency is not None:
            agency.update(
                verified=False,
                dimensions=None,
                causal_record_count=None,
                reason="unbound_episode_artifacts",
            )
        if "temporal_evidence" in capability:
            capability["temporal_evidence"]["chains"] = None
    elif "horizon" in capability:
        capability["horizon"]["trace_coverage_verified"] = True
        capability["horizon"]["recorded_ticks"] = binding["trace_ticks"]
    return native, capability, binding


def grade_episode(episode, spec, contracts, comparison_policy="strict"):
    if comparison_policy not in {"strict", USER_ASSUMED_POLICY}:
        raise ValueError("unknown comparison policy")
    row = dict(episode)
    for field in ("domain", "backend_kind"):
        if row.get(field) not in (None, spec[field]):
            return {"score": None, "reason": "episode_backend_mismatch"}
        row[field] = spec[field]
    tree = row.get("implementation_tree_sha256")
    if not tree or (comparison_policy == "strict" and any(
        row.get(k) != tree
        for k in ("implementation_tree_sha256_start", "implementation_tree_sha256_end")
    )):
        return {"score": None, "reason": "execution_identity_unproven_or_changed"}
    if not all(
        isinstance(row.get(k), str) and row[k]
        for k in (
            "agent_profile_sha256",
            "agent_treatment_sha256",
            "model",
            "interaction_mode",
            "run_semantics_fingerprint",
        )
    ):
        return {"score": None, "reason": "model_treatment_identity_missing"}
    if not all(isinstance(v, str) and v for v in _scope(row)):
        return {"score": None, "reason": "comparison_scope_missing"}
    key = (row.get("scenario_signature"), row.get("seed"), tree)
    if comparison_policy == USER_ASSUMED_POLICY:
        candidates = {
            value["contract_sha256"]: value
            for candidate, value in contracts.items()
            if candidate[:2] == key[:2]
        }
        if len(candidates) > 1:
            raise ValueError("ambiguous user-assumed reference; select one calibration")
        contract = next(iter(candidates.values()), None)
    else:
        contract = contracts.get(key)
    if contract is None:
        return {"score": None, "reason": "matching_reference_not_available"}
    measurement = extract_native_objective(row)
    result = score_native_quality(measurement, contract)
    return {
        **result,
        "measurement": measurement,
        "reference_contract_sha256": contract["contract_sha256"],
        "comparison_policy": comparison_policy,
        "compatibility_user_assumed": comparison_policy == USER_ASSUMED_POLICY,
        "execution_runtime_identity": tree,
        "reference_runtime_identity": contract["runtime_identity"],
        "reference_compatibility_receipt_sha256": contract.get(
            "compatibility_receipt_sha256"
        ),
        "original_primary_score": (row.get("ranking") or {}).get("primary_score"),
    }


def validated_model_aliases(aliases, comparison_policy):
    """Allow explicitly requested namespace spelling aliases, never model revisions."""
    from baselines.llm_agent import frozen_model_response_aliases

    if not isinstance(aliases, dict):
        raise ValueError("model aliases must be an explicit mapping")
    if aliases and comparison_policy != USER_ASSUMED_POLICY:
        raise ValueError("model aliases require explicit user-assumed comparison")
    for raw, canonical in aliases.items():
        if (not isinstance(raw, str) or not isinstance(canonical, str)
                or raw != canonical.rsplit("/", 1)[-1]
                or raw not in frozen_model_response_aliases(canonical)):
            raise ValueError("model alias is not an existing same-name namespace spelling")
    return dict(aliases)


def evaluate(config, *, bootstrap=0):
    comparison_policy = config.get("comparison_policy", "strict")
    if comparison_policy not in {"strict", USER_ASSUMED_POLICY}:
        raise ValueError("unknown comparison policy")
    user_assumed = comparison_policy == USER_ASSUMED_POLICY
    model_aliases = validated_model_aliases(config.get("model_aliases", {}), comparison_policy)
    suite_path = Path(config["suite"])
    suite_raw = suite_path.read_bytes()
    suite = json.loads(suite_raw)["scenarios"]
    specs = {(r["scenario_signature"], r["seed"]): r for r in suite}
    if len(specs) != len(suite):
        raise ValueError("duplicate native evaluation suite row")
    contracts = {}
    calibration = []
    for source in config["references"]:
        bundle = load_reference_report(Path(source["root"]), source["report"])
        calibration.append(
            {
                "root": source["root"],
                "report": source["report"],
                "report_sha256": bundle["report_sha256"],
            }
        )
        include = source.get("include_signatures")
        if include is not None and (
            not isinstance(include, list)
            or set(include) - {c["scenario_signature"] for c in bundle["contracts"]}
        ):
            raise ValueError("reference selection contains unknown signatures")
        for contract in bundle["contracts"]:
            if include is not None and contract["scenario_signature"] not in include:
                continue
            key = (
                contract["scenario_signature"],
                contract["seed"],
                contract["runtime_identity"],
            )
            if key in contracts:
                raise ValueError(
                    "duplicate reference contract; choose one fixed calibration"
                )
            contracts[key] = contract
    reference_lookup = dict(contracts)
    if user_assumed:
        by_case = defaultdict(set)
        for key, contract in contracts.items():
            by_case[key[:2]].add(contract["contract_sha256"])
        if any(len(values) > 1 for values in by_case.values()):
            raise ValueError("ambiguous user-assumed reference; select one calibration")
    compatibility = []
    for pair in ([] if user_assumed else config.get("reference_compatibility", [])):
        receipt = verify_reference_compatibility(
            Path(pair["reference_root"]), Path(pair["model_root"])
        )
        compatibility.append(receipt)
        for key, contract in contracts.items():
            if key[2] == receipt["reference_runtime_identity"]:
                target = (key[0], key[1], receipt["model_runtime_identity"])
                if target in contracts:
                    continue  # The explicitly supplied exact-runtime contract wins.
                if (
                    target in reference_lookup
                    and reference_lookup[target]["contract_sha256"]
                    != contract["contract_sha256"]
                ):
                    raise ValueError(
                        "ambiguous compatible reference; select one calibration"
                    )
                reference_lookup.setdefault(
                    target,
                    {
                        **contract,
                        "compatibility_receipt_sha256": receipt["receipt_sha256"],
                    },
                )
    inputs, provenance, history = {}, [], {}
    episode_scoring_versions = set()
    for model, paths in config["runs"].items():
        episodes = []
        filters = config.get("row_filters", {}).get(model, {})
        if set(filters) - {
            "implementation_tree_sha256",
            "agent_profile_sha256",
            "agent_treatment_sha256",
            "pass_id",
        }:
            raise ValueError("only fixed identity filters are supported")
        for path in paths:
            raw = Path(path).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            expected = config.get("input_sha256", {}).get(str(path))
            if (config.get("require_input_hashes") is True and not expected) or (
                expected is not None and expected != digest
            ):
                raise ValueError(f"episode input hash missing or mismatched: {path}")
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            selected = [
                r for r in rows if all(r.get(k) == v for k, v in filters.items())
            ]
            episodes.extend(
                {
                    **row,
                    "_offline_source_path": row.get("_offline_source_path", str(path)),
                }
                for row in selected
            )
            provenance.append(
                {
                    "model": model,
                    "path": path,
                    "sha256": digest,
                    "expected_hash_verified": expected is not None,
                    "n_input": len(rows),
                    "n_selected": len(selected),
                }
            )
        policy = config.get("attempt_policy", "retain_all")
        if policy == "latest_started":
            episodes, history[model] = select_latest_attempts(episodes)
        elif policy != "retain_all":
            raise ValueError("unknown attempt policy")
        inputs[model] = episodes
        episode_scoring_versions.update(
            str(version)
            for row in episodes
            if (version := (row.get("score") or {}).get("scoring_version"))
        )
    results, groups = {}, defaultdict(list)
    for model, episodes in inputs.items():
        selected, outside = [], []
        scopes, profiles, repeats = set(), set(), set()
        observed_scopes, actual_models, interaction_modes = set(), set(), set()
        raw_models = set()
        for row in episodes:
            key = (row.get("scenario_signature"), row.get("seed"))
            if key not in specs:
                outside.append({"scenario_signature": key[0], "seed": key[1]})
                continue
            scope = _scope(row)
            observed_scopes.add(scope)
            raw_models.add(row.get("model"))
            actual_models.add(model_aliases.get(row.get("model"), row.get("model")))
            interaction_modes.add(row.get("interaction_mode"))
            if all(isinstance(v, str) and v for v in scope):
                scopes.add(scope)
            profiles.add(
                (
                    row.get("model"),
                    row.get("agent_profile_sha256"),
                    row.get("agent_treatment_sha256"),
                )
            )
            repeats.add(str(row.get("pass_id")))
            native, capability, binding = grade_bound_episode(
                row, specs[key], reference_lookup, row.get("_offline_source_path"),
                comparison_policy,
            )
            selected.append(
                {
                    "model": model,
                    "scenario_signature": key[0],
                    "seed": key[1],
                    "native_quality": native,
                    "capability_report": capability,
                    "artifact_binding": binding,
                    "comparison_policy": comparison_policy,
                    "compatibility_user_assumed": user_assumed,
                    "execution_provenance": {
                        field: row.get(field)
                        for field in (
                            "model", "interaction_mode", "implementation_tree_sha256",
                            "implementation_tree_sha256_start", "implementation_tree_sha256_end",
                            "agent_profile_sha256", "agent_treatment_sha256",
                            "run_semantics_fingerprint", "suite_manifest_sha256", "pass_id",
                            "_offline_source_path", "_offline_selection_provenance",
                            "_derived_recovery_projection", "status", "error",
                        )
                    },
                }
            )
        result = aggregate_native_rows(selected, suite, [model])
        if (
            (outside and config.get("select_suite_members") is not True)
            or (not user_assumed and (len(scopes) != 1 or len(profiles) != 1))
            or len(actual_models) != 1
            or len(interaction_modes) != 1
            or any(not isinstance(value, str) or not value
                   for value in actual_models | interaction_modes)
            or len(repeats) != 1
            or any(not all(isinstance(v, str) and v for v in p) for p in profiles)
            or "None" in repeats
        ):
            for field in (
                "index",
                "domain_scores",
                "backend_scores",
                "source_scores",
                "task_success_rate",
                "constraint_failure_rate",
                "upper_bound_fraction",
                "lower_bound_fraction",
                "above_strong_reference_fraction",
            ):
                result["models"][model][field] = None
            result["models"][model]["complete"] = False
            result["ranking"] = []
            result["complete"] = False
            result["identity_blocker"] = "outside_suite_or_mixed_execution_identity"
        result["comparison_policy"] = comparison_policy
        result["compatibility_user_assumed"] = user_assumed
        result["observed_execution_identities"] = {
            "scopes": [list(value) for value in sorted(observed_scopes, key=repr)],
            "profiles": [list(value) for value in sorted(profiles, key=repr)],
            "pass_ids": sorted(repeats),
            "actual_models": sorted(actual_models, key=repr),
            "raw_models": sorted(raw_models, key=repr),
            "model_aliases": model_aliases,
            "interaction_modes": sorted(interaction_modes, key=repr),
        }
        result["outside_suite"] = outside
        result["graded_episodes"] = selected
        result["long_task_diagnostic"] = build_long_task_report(selected, suite, model)
        if "identity_blocker" in result:
            result["long_task_diagnostic"].update(
                index=None, complete=False, reason=result["identity_blocker"]
            )
        results[model] = result
        if result["complete"]:
            comparison_scope = (
                (USER_ASSUMED_POLICY, next(iter(interaction_modes)))
                if user_assumed else next(iter(scopes))
            )
            groups[(comparison_scope, next(iter(repeats)))].append(model)
    cohorts = []
    for (scope, repeat), models in groups.items():
        rows = [r for m in models for r in results[m]["graded_episodes"]]
        cohorts.append(
            {
                "execution_scope": list(scope) if not user_assumed else None,
                "comparison_scope": list(scope),
                "comparison_policy": comparison_policy,
                "compatibility_user_assumed": user_assumed,
                "pass_id": repeat,
                **aggregate_native_rows(rows, suite, models, bootstrap=bootstrap),
            }
        )
    ready_cases = {
        (contract["scenario_signature"], contract["seed"])
        for contract in contracts.values()
        if contract["status"] == "ready"
    }
    return {
        "schema_version": VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "comparison_policy": comparison_policy,
        "compatibility_user_assumed": user_assumed,
        "model_aliases": model_aliases,
        "episode_scoring_versions": sorted(episode_scoring_versions),
        "formal_run_certified": False,
        "suite_sha256": hashlib.sha256(suite_raw).hexdigest(),
        "n_suite_rows": len(suite),
        "reference_contracts": list(contracts.values()),
        "reference_inputs": calibration,
        "reference_compatibility": compatibility,
        "reference_compatibility_requests_not_applied":
            config.get("reference_compatibility", []) if user_assumed else [],
        "episode_inputs": provenance,
        "attempt_selection_history": history,
        "by_model": results,
        "cohorts": cohorts,
        "n_calibrated_reference_rows": sum(
            c["status"] == "ready" for c in contracts.values()
        ),
        "reference_coverage": {
            "expected_suite_cases": len(specs),
            "ready_suite_cases": len(set(specs) & ready_cases),
            "missing_suite_cases": [
                {"scenario_signature": key[0], "seed": key[1]}
                for key in sorted(set(specs) - ready_cases)
            ],
            "ready_runtime_contracts": sum(
                c["status"] == "ready" for c in contracts.values()
            ),
        },
    }


def model_summary_lines(report):
    """Keep input coverage, measurement gaps and cohort blockers distinct."""
    lines = [
        "| model | index | input cases/expected | measured/expected | cohort blocker | unavailable row reasons |",
        "|---|---:|---:|---:|---|---|",
    ]
    for model, group in report["by_model"].items():
        item = group["models"][model]
        rows = group["graded_episodes"]
        cases = {(row["scenario_signature"], row["seed"]) for row in rows}
        reasons = Counter(
            row["native_quality"].get("reason", "unspecified")
            for row in rows
            if row["native_quality"].get("score") is None
        )
        bindings = Counter(
            row["artifact_binding"]["reason"]
            for row in rows
            if row.get("artifact_binding", {}).get("verified") is False
        )
        reason_text = "; ".join(f"{key}={n}" for key, n in sorted(reasons.items()))
        if bindings:
            reason_text += (
                " (binding: "
                + "; ".join(f"{key}={n}" for key, n in sorted(bindings.items()))
                + ")"
            )
        lines.append(
            f"| {model} | {item['index'] if item['index'] is not None else 'N/A'} "
            f"| {len(cases)}/{item['n_expected']} | {item['n_measured']}/{item['n_expected']} "
            f"| {group.get('identity_blocker', 'none')} | {reason_text or 'none'} |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error("--bootstrap must be nonnegative")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / ".hl"):
        parser.error("output must be under .hl")
    raw = args.manifest.read_bytes()
    from core.implementation_identity import implementation_identity

    scoring_implementation = implementation_identity(ROOT)
    scoring_entry_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = evaluate(json.loads(raw), bootstrap=args.bootstrap)
    ending_implementation = implementation_identity(ROOT)
    if (
        scoring_implementation["evaluation_runtime_sha256"]
        != ending_implementation["evaluation_runtime_sha256"]
        or scoring_entry_sha256
        != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise RuntimeError("evaluation implementation changed during offline scoring")
    report["scoring_implementation"] = scoring_implementation
    report["scoring_entry_sha256"] = scoring_entry_sha256
    report["input_manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    lines = [
        "# Native operational quality",
        "",
        f"Evaluation: {EVALUATION_VERSION}; native score: {VERSION}; expected cases: {report['n_suite_rows']}.",
        "",
        f"Comparison policy: {report.get('comparison_policy', 'strict')}. "
        + ("Cross-runtime/profile compatibility is user-assumed, not verified; formal certification remains false."
           if report.get("compatibility_user_assumed") else "Execution and reference identity checks are strict."),
        "",
        "Missing references or measurements withhold the index. Run certification is separate.",
        "",
        "Input cases are not authenticated completion or formal eligibility. Cohort blockers apply even when individual rows have scores.",
        "",
        *model_summary_lines(report),
    ]
    with (output / "scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "scenario_signature",
                "seed",
                "native_score",
                "raw_quality",
                "old_primary",
                "reason",
            ],
        )
        writer.writeheader()
        for model, group in report["by_model"].items():
            for row in group["graded_episodes"]:
                score = row["native_quality"]
                writer.writerow(
                    {
                        "model": model,
                        "scenario_signature": row["scenario_signature"],
                        "seed": row["seed"],
                        "native_score": score.get("score"),
                        "raw_quality": score.get("raw_score"),
                        "old_primary": score.get("original_primary_score"),
                        "reason": score.get("reason", ""),
                    }
                )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "calibrated_rows": report["n_calibrated_reference_rows"],
                "complete_models": sum(
                    x["complete"] for x in report["by_model"].values()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
