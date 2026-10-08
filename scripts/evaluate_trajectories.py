#!/usr/bin/env python3
"""Default Lite141 0.28 reader for raw closed episode directories.

No legacy score report, reference calibration, provider call or native backend
construction is needed. Original journals and runtime identities stay intact.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from evaluation.operational_outcome027 import aggregate_outcome027  # noqa: E402
from evaluation.operational_outcome028 import aggregate_outcome028  # noqa: E402
from evaluation.operational_weights import validate_task_weights  # noqa: E402
from evaluation.trajectory_outcome027 import (  # noqa: E402
    artifact_ref,
    canonical_digest,
    grade_episode027,
    read_bound,
    terminal_tree_drift_candidate,
)
from evaluation.trajectory_outcome028 import (  # noqa: E402
    DEFAULT_POLICY,
    grade_episode028,
    load_policy,
)


def _resolve(ref: dict, root: Path) -> dict:
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ValueError("input_artifact_descriptor_shape_invalid")
    path = Path(ref["path"])
    return {**ref, "path": str(path if path.is_absolute() else root / path)}


def input_manifest(
    run_dirs: list[Path],
    *,
    policy_path: Path = DEFAULT_POLICY,
    aliases: dict | None = None,
) -> dict:
    runs, models = [], set()
    for directory in run_dirs:
        journal, config = directory / "episodes.jsonl", directory / "run_config.json"
        models.update(json.loads(config.read_bytes())["models"])
        runs.append(
            {"episodes": artifact_ref(journal), "run_config": artifact_ref(config)}
        )
    aliases = aliases or {}
    if (
        not isinstance(aliases, dict)
        or not set(aliases) <= models
        or any(not isinstance(value, str) or not value for value in aliases.values())
    ):
        raise ValueError("undeclared_or_empty_model_alias")
    labels = {}
    for actual in models:
        label = aliases.get(actual, actual)
        if label in labels and labels[label] != actual:
            raise ValueError("model_alias_merges_distinct_actual_identities")
        labels[label] = actual
    return {
        "schema_version": "operate_trajectory_outcome027_inputs.v1",
        "policy": artifact_ref(policy_path),
        "runs": runs,
        "models": sorted({aliases.get(model, model) for model in models}),
        "model_aliases": aliases,
        "comparison_policy": "descriptive_whole_episode",
        "selection": "latest_status_ok_whole_episode_before_scoring",
        "pass_index": 0,
        "treatment": "logical_persistent",
    }


def _unknown(model: str, contract: dict, reason: str, *, version="0.27.0") -> dict:
    row = {
        "model": model,
        "scenario_signature": contract["scenario_signature"],
        "seed": contract["seed"],
        "C": None,
        "S": contract["scales"]["native_cost"]["value"],
        "F": None,
        "N": None,
        "Q": None,
        "status": "unavailable",
        "origin": {"reason": reason},
        "outcome027": {
            "evaluation_version": "0.27.0",
            "numeric_determined": False,
            "publish_eligible": False,
            "evidence_qualified": False,
            "completion_measured": False,
            "completion_score": None,
            "completion_applicable": contract["backend_kind"] != "citylearn",
            "native_quality": None,
            "hard_failure": None,
            "native_catastrophe_applicable": None,
            "hard_gate_signal_source": None,
            "score": None,
            "missing_measurement_bounds": [0.0, 100.0],
            "reason": reason,
            "evidence_ids": [],
        },
    }
    if version == "0.28.0":
        row["outcome028"] = {**row["outcome027"], "evaluation_version": version}
    return row


def evaluate(
    manifest: dict, *, root: Path = ROOT, artifact_root: Path | None = None
) -> dict:
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "operate_trajectory_outcome027_inputs.v1"
    ):
        raise ValueError("unsupported_raw_trajectory_manifest")
    if manifest.get("comparison_policy") not in {
        "descriptive_whole_episode",
        "latest_framework_user_assumed",
    } or manifest.get("selection") not in {
        "latest_status_ok_whole_episode_before_scoring",
        "latest_status_ok_or_terminal_tree_drift_whole_episode_before_scoring",
    }:
        raise ValueError("unsupported_whole_episode_selection_policy")
    comparison = (
        "latest_framework_user_assumed"
        if manifest["comparison_policy"] == "latest_framework_user_assumed"
        else "strict"
    )
    include_tree_drift = (
        manifest["selection"]
        == "latest_status_ok_or_terminal_tree_drift_whole_episode_before_scoring"
    )
    if include_tree_drift and comparison != "latest_framework_user_assumed":
        raise ValueError(
            "terminal_tree_drift_selection_requires_explicit_user_assumption"
        )
    models = manifest["models"]
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(model, str) or not model for model in models)
        or len(set(models)) != len(models)
    ):
        raise ValueError("empty_or_duplicate_model_population")
    root = Path(root).resolve()
    relocations = manifest.get("artifact_relocations") or {}
    if not isinstance(relocations, dict):
        raise ValueError("artifact_relocations_shape_invalid")
    relocations = dict(relocations)
    if artifact_root is not None:
        for path in Path(artifact_root).iterdir():
            if (
                path.is_file()
                and len(path.name) == 64
                and all(c in "0123456789abcdef" for c in path.name)
            ):
                relocations[path.name] = str(path.resolve())
    policy_raw, policy_ref = read_bound(
        _resolve(manifest["policy"], root), relocations=relocations
    )
    policy = load_policy(Path(policy_ref["path"]), root=root)
    version = policy.get("evaluation_version", "0.27.0")
    modern = version == "0.28.0"
    outcome_key = "outcome028" if modern else "outcome027"
    normalizations = {
        (c["scenario_signature"], c["seed"]): c
        for c in policy.get("normalization_contracts", [])
    }
    compiled = policy["source_contracts"]
    contracts = {
        (row["scenario_signature"], row["seed"]): row for row in compiled["contracts"]
    }
    weights = validate_task_weights(policy["task_weight_manifest"], compiled)
    populations = {
        (row["scenario_signature"], row["seed"]): row
        for row in policy["voltage_population_contracts"]
    }
    scenarios = {}
    for key, contract in contracts.items():
        raw, _ = read_bound(
            {
                "path": str(root / contract["scenario_path"]),
                "sha256": contract["scenario_sha256"],
            }
        )
        scenarios[key] = yaml.safe_load(raw)
    start = implementation_identity(root)
    aliases = manifest.get("model_aliases") or {}
    if not isinstance(aliases, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not v
        for k, v in aliases.items()
    ):
        raise ValueError("invalid_model_alias_mapping")
    candidates, attempts, frozen, input_runs = {}, [], [policy_ref], []
    if modern:
        frozen.append(policy["base_policy_artifact"])
    out_of_panel = []
    recoveries = {}
    for recovery in manifest.get("recoveries", []):
        if (
            not isinstance(recovery, dict)
            or not {
                "model",
                "scenario_signature",
                "seed",
                "canonical_episode_sha256",
                "sidecar",
            }
            <= recovery.keys()
        ):
            raise ValueError("native_recovery_input_shape_invalid")
        key = (
            recovery["model"],
            recovery["scenario_signature"],
            recovery["seed"],
            recovery["canonical_episode_sha256"],
        )
        if (
            key in recoveries
            or recovery["model"] not in models
            or key[1:3] not in contracts
        ):
            raise ValueError("native_recovery_duplicate_or_foreign_input")
        _, ref = read_bound(
            _resolve(recovery["sidecar"], root), relocations=relocations
        )
        frozen.append(ref)
        recoveries[key] = {**recovery, "sidecar": ref}
    for run in manifest["runs"]:
        journal_raw, journal_ref = read_bound(
            _resolve(run["episodes"], root), relocations=relocations
        )
        config_raw, config_ref = read_bound(
            _resolve(run["run_config"], root), relocations=relocations
        )
        config = json.loads(config_raw)
        if not journal_raw.endswith(b"\n"):
            raise ValueError("journal_not_closed_at_record_boundary")
        frozen.extend([journal_ref, config_ref])
        input_runs.append({"episodes": journal_ref, "run_config": config_ref})
        for number, line in enumerate(journal_raw.splitlines(), 1):
            if not line.strip():
                continue
            episode = json.loads(line)
            actual = episode["model"]
            model = aliases.get(actual, actual)
            key = episode["scenario_signature"], episode["seed"]
            if actual not in config["models"] or model not in models:
                raise ValueError("journal_foreign_model_or_source_case")
            if key not in contracts:
                if manifest.get("source_case_scope") != "project_frozen_Lite141":
                    raise ValueError("journal_foreign_model_or_source_case")
                out_of_panel.append(
                    {
                        "model": model,
                        "scenario_signature": key[0],
                        "seed": key[1],
                        "journal": journal_ref,
                        "line": number,
                        "canonical_episode_sha256": canonical_digest(episode),
                    }
                )
                continue
            if episode.get("pass_index", 0) != manifest.get("pass_index", 0):
                continue
            origin = {
                "actual_model": actual,
                "journal": journal_ref,
                # Selection uses the frozen declared path, independent of storage relocation.
                "selection_journal_path": run["episodes"]["path"],
                "config": config_ref,
                "line": number,
                "canonical_episode_sha256": canonical_digest(episode),
                "execution_attempt_id": episode.get("execution_attempt_id"),
                "invocation_started_at_utc": episode.get("invocation_started_at_utc"),
                "original_episode_status": episode.get("status"),
                "original_episode_error_type": episode.get("error_type"),
                "original_episode_error": episode.get("error"),
            }
            if episode.get("status") == "ok":
                candidates.setdefault((model, *key), []).append(
                    (episode, config, origin)
                )
            else:
                if include_tree_drift and terminal_tree_drift_candidate(episode):
                    candidates.setdefault((model, *key), []).append(
                        (episode, config, origin)
                    )
                attempts.append(
                    {
                        "model": model,
                        "scenario_signature": key[0],
                        "seed": key[1],
                        "status": episode.get("status"),
                        "error_type": episode.get("error_type"),
                        "termination_category": episode.get("termination_category"),
                        "origin": origin,
                    }
                )
    rows = []
    for model in models:
        for key, contract in sorted(contracts.items()):
            options = candidates.get((model, *key))
            if not options:
                rows.append(
                    _unknown(
                        model,
                        contract,
                        "no_closed_successful_whole_episode",
                        version=version,
                    )
                )
                continue
            episode, config, origin = max(
                options,
                key=lambda item: (
                    item[2]["invocation_started_at_utc"] or "",
                    item[2]["line"],
                    item[2]["selection_journal_path"],
                ),
            )
            try:
                recovery = recoveries.get(
                    (model, *key, origin["canonical_episode_sha256"])
                )
                grader = grade_episode028 if modern else grade_episode027
                extra = (
                    {"normalization_contract": normalizations.get(key)}
                    if modern
                    else {}
                )
                row = grader(
                    episode,
                    config,
                    Path(origin["journal"]["path"]),
                    contract,
                    scenarios[key],
                    populations.get(key),
                    root=root,
                    treatment=manifest.get("treatment", "logical_persistent"),
                    relocations=relocations,
                    recovery_ref=recovery["sidecar"] if recovery else None,
                    recovery_equivalence=recovery.get("recovery_equivalence")
                    if recovery
                    else None,
                    comparison_policy=comparison,
                    **extra,
                )
            except (ValueError, OSError) as error:
                row = _unknown(model, contract, str(error), version=version)
                row["status"] = "execution_or_native_evidence_unqualified"
                row["error_type"] = type(error).__name__
            else:
                row["model"] = model
                row["status"] = (
                    "scored"
                    if row["Q"] is not None
                    else "native_measurement_incomplete"
                )
            row["origin"].update(origin)
            rows.append(row)
    actual_ids = {
        model: {
            r["origin"]["actual_model"]
            for r in rows
            if r["model"] == model and r["origin"].get("actual_model")
        }
        for model in models
    }
    if any(len(ids) > 1 for ids in actual_ids.values()):
        raise ValueError("model_alias_merges_distinct_actual_identities")
    aggregate = aggregate_outcome028 if modern else aggregate_outcome027
    report = aggregate(rows, contracts, weights, models=models)
    observed = Counter(
        row["model"] for row in rows if row["origin"].get("canonical_episode_sha256")
    )
    for model in report["models"]:
        model["n_observed"] = observed[model["model"]]
        model["selected_actual_model_ids"] = sorted(actual_ids[model["model"]])
    for ref in frozen:
        read_bound(ref)
    if Path(policy_ref["path"]).read_bytes() != policy_raw:
        raise ValueError("source_policy_changed_during_scoring")
    if (
        implementation_identity(root)["implementation_tree_sha256"]
        != start["implementation_tree_sha256"]
    ):
        raise ValueError("scoring_runtime_changed_during_evaluation")
    report.update(
        schema_version=f"operate_raw_trajectory_outcome{version[:4].replace('.', '')}_report.v1",
        policy=policy_ref,
        source_contracts=compiled,
        task_weight_manifest=policy["task_weight_manifest"],
        graded_episodes=rows,
        attempts=attempts,
        excluded_out_of_panel_records=out_of_panel,
        input_runs=input_runs,
        input_recoveries=list(recoveries.values()),
        input_manifest_sha256=canonical_digest(manifest),
        input_manifest_digest_policy="canonical_json",
        current_implementation=start,
        comparison_policy=manifest["comparison_policy"],
        compatibility_user_assumed=comparison == "latest_framework_user_assumed",
        formal_run_certified=False,
        same_run_141_merge_certified=False,
        selection=manifest["selection"],
        model_aliases=aliases,
        normalization_contracts=policy.get("normalization_contracts", []),
        counts={
            "models": len(models),
            "fixed_targets": len(rows),
            "numeric_Q": sum(row["Q"] is not None for row in rows),
            "complete_models": sum(row["complete"] for row in report["models"]),
            "states": dict(Counter(row["status"] for row in rows)),
            "verified_recovery_sidecars_used": sum(
                row.get("recovery_binding") is not None for row in rows
            ),
            "strict_execution_identity_qualified": sum(
                (row.get("execution_binding") or {}).get(
                    "strict_execution_identity_qualified"
                )
                is True
                for row in rows
            ),
            "native_completed_tree_drift_qualified": sum(
                (row.get("execution_binding") or {}).get(
                    "native_completed_tree_drift_qualified"
                )
                is True
                for row in rows
            ),
        },
        missing_by_model={
            model: dict(
                Counter(
                    (row.get("completion") or {}).get("reason")
                    or (row.get(outcome_key) or {}).get("reason")
                    or row["status"]
                    for row in rows
                    if row["model"] == model and row["Q"] is None
                )
            )
            for model in models
        },
        execution={
            "provider_calls": 0,
            "native_episode_replays": 0,
            "source_constructors": 0,
        },
    )
    return report


def render(report: dict) -> str:
    lines = [
        f"# Lite141 {report.get('evaluation_version', '0.27.0')} trajectory evaluation",
        "",
        "Fixed development panel; Q is continuous source-grounded utility, not binary success. Missing evidence remains N/A.",
        "",
        "Ranks describe complete models within this report only; execution providers/profiles/trees remain in the case ledger.",
        "",
        "| Model | Scored /141 | Q | N | F | Rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if report.get("compatibility_user_assumed"):
        lines.insert(
            4,
            "Cross-framework compatibility is user-assumed. These are descriptive recorded deployment outcomes; implementation effects were not experimentally controlled.",
        )

    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"

    for row in report["models"]:
        lines.append(
            f"| {row['model']} | {row['n_scored']} | {fmt(row['primary_score'])} | {fmt(row['native_quality'])} | {fmt(row['completion_only']['score'])} | {row['primary_rank'] or '—'} |"
        )
    return "\n".join(lines) + "\n"


def _transport_independent(value):
    if isinstance(value, dict):
        result = {key: _transport_independent(item) for key, item in value.items()}
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            result["path"] = "sha256:" + value["sha256"]
        return result
    if isinstance(value, list):
        return [_transport_independent(item) for item in value]
    return value


def audit_report(
    report: dict,
    manifest: dict,
    *,
    root: Path = ROOT,
    artifact_root: Path | None = None,
) -> dict:
    """Rebuild every case and aggregate; preserve the original report identity."""
    rebuilt = evaluate(manifest, root=root, artifact_root=artifact_root)
    original_runtime = report.get("current_implementation") or {}
    actual_runtime = rebuilt["current_implementation"]
    if (
        original_runtime.get("implementation_tree_sha256")
        != actual_runtime["implementation_tree_sha256"]
    ):
        raise ValueError("trajectory_report_scoring_runtime_mismatch")
    original, expected = (
        {k: v for k, v in value.items() if k != "current_implementation"}
        for value in (report, rebuilt)
    )
    if _transport_independent(original) != _transport_independent(expected):
        raise ValueError("trajectory_report_reproduction_mismatch")
    return {
        "schema_version": f"operate_raw_trajectory_outcome{report.get('evaluation_version', '0.27.0')[:4].replace('.', '')}_audit.v1",
        "verified": True,
        "report_canonical_sha256": canonical_digest(report),
        "input_manifest_sha256": canonical_digest(manifest),
        "original_scoring_implementation": original_runtime,
        "audit_implementation": actual_runtime,
        "counts": rebuilt["counts"],
        "execution": rebuilt["execution"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path)
    group.add_argument("--run-dir", type=Path, action="append")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--model-alias", action="append", default=[], metavar="ACTUAL=LABEL"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument(
        "--artifact-root", type=Path, help="restored SHA-256-named artifact directory"
    )
    args = parser.parse_args()
    aliases = dict(value.split("=", 1) for value in args.model_alias)
    if args.manifest and (aliases or args.policy != DEFAULT_POLICY):
        parser.error("manifest already freezes its policy and model aliases")
    if args.audit_report and not args.manifest:
        parser.error("report audit requires the original frozen input manifest")
    manifest_raw = args.manifest.read_bytes() if args.manifest else None
    manifest = (
        json.loads(manifest_raw)
        if manifest_raw
        else input_manifest(args.run_dir, policy_path=args.policy, aliases=aliases)
    )
    output = args.output_dir.resolve()
    if output.exists() or any(
        output.is_relative_to(ROOT / directory)
        for directory in ("release", "scenarios", "sources")
    ):
        raise ValueError("output_must_be_new_and_outside_frozen_release_inputs")
    if args.audit_report:
        report_raw = args.audit_report.read_bytes()
        receipt = audit_report(
            json.loads(report_raw), manifest, artifact_root=args.artifact_root
        )
        if (
            args.audit_report.read_bytes() != report_raw
            or args.manifest.read_bytes() != manifest_raw
        ):
            raise ValueError("trajectory_audit_input_changed")
        output.mkdir(parents=True, exist_ok=False)
        (output / "audit_receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps({"verified": True, "counts": receipt["counts"]}))
        return
    report = evaluate(manifest, artifact_root=args.artifact_root)
    if args.manifest and args.manifest.read_bytes() != manifest_raw:
        raise ValueError("input_manifest_changed_during_evaluation")
    output.mkdir(parents=True, exist_ok=False)
    (output / "input_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (output / "report.md").write_text(render(report))
    (output / "case_rows.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
            for row in report["graded_episodes"]
        )
    )
    with (output / "table.csv").open("x", newline="") as stream:
        columns = [
            "model",
            "n_scored",
            "primary_score",
            "native_quality",
            "primary_rank",
            "complete",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in report["models"])
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
