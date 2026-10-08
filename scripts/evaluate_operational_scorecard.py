#!/usr/bin/env python3
"""Build the 0.29 outcome/capability scorecard from original raw trajectories.

The 0.28 outcome index is preserved. No provider call, source constructor or
native episode replay is performed; previous score reports are not inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from evaluation.capability_measurement029 import AXES  # noqa: E402
from evaluation.scorecard029 import build_scorecard, portable_report  # noqa: E402
from evaluation.trajectory_outcome027 import canonical_digest  # noqa: E402
from scripts.evaluate_trajectories import (  # noqa: E402
    evaluate as evaluate_outcomes,
    input_manifest,
)


def evaluate(
    manifest: dict, *, root: Path = ROOT, artifact_root: Path | None = None
) -> tuple[dict, dict]:
    before = implementation_identity(root)["implementation_tree_sha256"]
    base = evaluate_outcomes(manifest, root=root, artifact_root=artifact_root)
    scorecard = build_scorecard(base, root=root)
    if implementation_identity(root)["implementation_tree_sha256"] != before:
        raise ValueError("scorecard_runtime_changed_during_evaluation")
    return scorecard, base


def summary_rows(report: dict) -> list[dict]:
    ranking = report["retrospective_ranking"]
    ranked = {r["model"]: r for r in ranking["rows"]}
    capabilities = {r["model"]: r["axes"] for r in report["capability_models"]}
    analyses = {r["model"]: r for r in report["outcome_analysis"]["models"]}
    ranges = report["outcome_analysis"]["rank_sensitivity"]["physical_source_cluster"][
        "rank_ranges"
    ]
    result = []
    for model in report["outcome_models"]:
        name = model["model"]
        analysis = analyses[name]
        interval = model.get("score_interval")
        row = {
            "model": name,
            "evaluation_version": report["evaluation_version"],
            "outcome_index_version": report["outcome_index_version"],
            "comparison_policy": report["comparison_policy"],
            "n_expected": model["n_expected"],
            "n_scored": model["n_scored"],
            "Q": model["primary_score"],
            "N": model["native_quality"],
            "F": model["completion_only"]["score"],
            "F_expected": model["completion_only"]["n_expected"],
            "F_measured": model["completion_only"]["n_measured"],
            "outcome_point_rank": model["primary_rank"],
            "rank_cohort_count": len(ranking["cohort"]),
            "total_declared_model_count": len(report["outcome_models"]),
            "declared_panel_rank_lower": ranked[name]["declared_panel_rank_range"][0],
            "declared_panel_rank_upper": ranked[name]["declared_panel_rank_range"][1],
            "retrospective_rank_eligible": ranked[name]["complete_eligible"],
            "missing_score_reasons": json.dumps(model["reason_counts"], sort_keys=True),
            "measurement_rank_min": ranked[name]["measurement_rank_range"][0]
            if ranked[name]["measurement_rank_range"] is not None
            else None,
            "measurement_rank_max": ranked[name]["measurement_rank_range"][1]
            if ranked[name]["measurement_rank_range"] is not None
            else None,
            "Q_combined_lower": analysis["overall"]["Q"]["bounds"][0],
            "Q_combined_upper": analysis["overall"]["Q"]["bounds"][1],
            "retrospective_protocol_sha256": report["retrospective_protocol"][
                "protocol_sha256"
            ],
            "Q_missing_lower": model["missing_measurement_bounds"][0],
            "Q_missing_upper": model["missing_measurement_bounds"][1],
            "Q_precision_lower": interval[0] if interval else None,
            "Q_precision_upper": interval[1] if interval else None,
            "recorded_hard_failure_rate": model["recorded_hard_failure_rate"],
            "catastrophe_unmodeled_weight": model["safety_coverage"][
                "native_catastrophe"
            ]["unmodeled"]["weight_mass"],
            "declared_hard_gate_known_cases": model["safety_coverage"][
                "declared_hard_gate_known_cases"
            ],
            "declared_hard_gate_known_weight": model["safety_coverage"][
                "declared_hard_gate_known_weight_mass"
            ],
            "negative_cost_saturation_cases": analysis[
                "negative_cost_saturation_cases"
            ],
            "source_deletion_rank_min": ranges[name][0] if name in ranges else None,
            "source_deletion_rank_max": ranges[name][1] if name in ranges else None,
            "rank_range_interpretation": "deterministic_sensitivity_not_confidence_interval",
            "formal_run_certified": report["formal_run_certified"],
            "same_run_141_merge_certified": report["same_run_141_merge_certified"],
            "capability_ranking_ready": report["capability_ranking_ready"],
        }
        for scope, coverage in model["safety_coverage"]["native_catastrophe"].items():
            row[f"catastrophe_{scope}_weight"] = coverage["weight_mass"]
            row[f"catastrophe_{scope}_cases"] = coverage["cases"]
        for axis in AXES:
            value = capabilities[name][axis]
            row[axis] = value["score"]
            row[axis + "_measured_cases"] = value["measured_cases"]
            row[axis + "_expected_cases"] = value["expected_cases"]
            row[axis + "_reasons"] = json.dumps(value["reason_counts"], sort_keys=True)
        result.append(row)
    return sorted(
        result,
        key=lambda row: (
            row["outcome_point_rank"] is None,
            row["outcome_point_rank"] or 0,
            row["model"],
        ),
    )


def render(report: dict) -> str:
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"

    lines = [
        "# OPERATE 0.29 fixed-panel retrospective outcome results",
        "",
        "Q/N/F retain the 0.28 fixed-panel outcome definition. The point rank is a declared utility ordering, not a capability rank.",
        "Missing source capability contracts or measurements remain N/A. No six-axis composite or repeated-run confidence interval is manufactured.",
        "",
        f"Protocol: `{report['retrospective_protocol']['protocol_id']}`. Outcome ranks compare {len(report['retrospective_ranking']['cohort'])} complete models out of {len(report['outcome_models'])} declared models; incomplete models are unranked. Optional six-axis calibration is not an outcome-ranking prerequisite.",
        f"Comparison policy: `{report['comparison_policy']}`. Original execution provenance remains available in the full report.",
        "",
        "| Model | Q | N | F | Scored / expected | Hard failure % | Outcome rank | Measurement rank range | Delete-source rank range |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    rows = summary_rows(report)
    for row in rows:
        name = row["model"].replace("|", "\\|").replace("\n", " ")
        bounds = (
            "N/A"
            if row["source_deletion_rank_min"] is None
            else f"{row['source_deletion_rank_min']}–{row['source_deletion_rank_max']}"
        )
        lines.append(
            f"| {name} | {fmt(row['Q'])} | {fmt(row['N'])} | {fmt(row['F'])} | {row['n_scored']}/{row['n_expected']} | {fmt(row['recorded_hard_failure_rate'])} | {row['outcome_point_rank'] or '—'} | {str(row['measurement_rank_min']) + '–' + str(row['measurement_rank_max']) if row['measurement_rank_min'] is not None else 'N/A'} | {bounds} |"
        )
    lines.extend(
        [
            "",
            "Source-deletion ranges change panel composition; they are sensitivity analyses, not uncertainty from repeated model executions.",
            "Measurement rank ranges cover recorded numeric precision within the complete cohort. Neither type is a confidence interval. Pairwise precision-unresolved relations are not merged into transitive tie groups. Point ranks use unrounded scores; exact ties share competition rank.",
            "",
            "All-declared-model bounds include missing evidence. They are hypothetical possible competition ranks, not assigned ranks for incomplete models. Safety coverage is a fraction of fixed case weight; known declared gates do not imply comprehensive safety.",
            "",
            "| Model | Combined Q bounds | Hypothetical rank among all declared | Known gate weight | Catastrophe modeled / unmodeled / unknown weight |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in rows:
        name = row["model"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {name} | {fmt(row['Q_combined_lower'])}–{fmt(row['Q_combined_upper'])} | {row['declared_panel_rank_lower']}–{row['declared_panel_rank_upper']} | {fmt(row['declared_hard_gate_known_weight'])} | {fmt(row['catastrophe_applicable_weight'])} / {fmt(row['catastrophe_unmodeled_weight'])} / {fmt(row['catastrophe_unknown_weight'])} |"
        )
    lines.extend(
        [
            "",
            "Optional capability diagnostics (outside the retrospective primary ranking):",
            "",
            "| Model | " + " | ".join(AXES) + " |",
            "|---|" + "---:|" * len(AXES),
        ]
    )
    for row in rows:
        name = row["model"].replace("|", "\\|").replace("\n", " ")
        values = [
            f"{fmt(row[a])} ({row[a + '_measured_cases']}/{row[a + '_expected_cases']})"
            for a in AXES
        ]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "Capability parentheses show measured/expected cases. Absent contracts are unknown scope, not structural non-applicability. Native outcomes and effect association do not independently prove causal improvement or construct calibration.",
            "",
            "See `domain_summary.csv`, `capability_summary.csv`, `task_semantics.json`, and the complete `report.json` for fixed weights, raw objectives, Pareto tradeoffs, evidence links, missingness and exact measurement boundaries.",
        ]
    )
    return "\n".join(lines) + "\n"


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output: Path, manifest: dict, report: dict, base: dict) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("input_manifest.json", manifest),
        ("report.json", report),
        ("outcome028.json", base),
        ("task_semantics.json", report["task_semantics"]),
        ("retrospective_protocol.json", report["retrospective_protocol"]),
        ("retrospective_ranking.json", report["retrospective_ranking"]),
        ("rank_sensitivity.json", report["outcome_analysis"]["rank_sensitivity"]),
    ):
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
    (output / "report.md").write_text(render(report))
    _csv(output / "table.csv", summary_rows(report))
    domains = []
    for model in report["outcome_analysis"]["models"]:
        for domain in model["domain"]:
            domains.append(
                {
                    "model": model["model"],
                    "domain": domain["domain"],
                    **{
                        f"{m}_{field}": domain[m][field]
                        for m in ("Q", "N", "F")
                        for field in (
                            "score",
                            "expected_cases",
                            "measured_cases",
                            "expected_weight_mass",
                        )
                    },
                    "capability_ranking_ready": False,
                }
            )
    _csv(output / "domain_summary.csv", domains)
    capability_rows = []
    for model in report["capability_models"]:
        for axis, value in model["axes"].items():
            capability_rows.append(
                {
                    "model": model["model"],
                    "capability": axis,
                    **{
                        k: v
                        for k, v in value.items()
                        if k not in {"evidence_ids_by_case", "reason_counts"}
                    },
                    "reasons": json.dumps(value["reason_counts"], sort_keys=True),
                }
            )
    _csv(output / "capability_summary.csv", capability_rows)
    # All-model component relations remain available, including incomplete
    # comparisons. The separately exported ranking fixes the complete cohort.
    pairwise_rows = [
        {
            "a": pair["a"],
            "b": pair["b"],
            "Q_difference": pair["Q_difference"],
            "Q_difference_lower": pair["Q_difference_bounds"][0],
            "Q_difference_upper": pair["Q_difference_bounds"][1],
            **{
                key + "_weight": value
                for key, value in pair["relation_weight_mass"].items()
            },
        }
        for pair in report["outcome_analysis"]["pairwise"]
    ]
    if pairwise_rows:
        _csv(output / "pairwise_summary.csv", pairwise_rows)
    checksums = {
        p.name: {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    (output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--run-dir", action="append", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path)
    args = parser.parse_args()
    if args.audit_report and not args.manifest:
        parser.error("audit requires the original input manifest")
    output = args.output_dir.resolve()
    if output.exists() or any(
        output.is_relative_to(ROOT / p) for p in ("release", "scenarios", "sources")
    ):
        raise ValueError("output_must_be_new_and_outside_frozen_release_inputs")
    manifest_raw = args.manifest.read_bytes() if args.manifest else None
    manifest = (
        json.loads(manifest_raw) if manifest_raw else input_manifest(args.run_dir)
    )
    original_raw = args.audit_report.read_bytes() if args.audit_report else None
    report, base = evaluate(manifest, artifact_root=args.artifact_root)
    if args.manifest and args.manifest.read_bytes() != manifest_raw:
        raise ValueError("scorecard_manifest_changed")
    if original_raw is not None:
        original = json.loads(original_raw)
        if args.audit_report.read_bytes() != original_raw:
            raise ValueError("scorecard_audit_input_changed")
        # Git bookkeeping may differ; runtime bytes must still match.
        if (
            original["current_implementation"]["implementation_tree_sha256"]
            != report["current_implementation"]["implementation_tree_sha256"]
        ):
            raise ValueError("scorecard_audit_runtime_mismatch")
        for value in (original, report):
            value.pop("current_implementation", None)
        if portable_report(original) != portable_report(report):
            raise ValueError("scorecard_reproduction_mismatch")
        output.mkdir(parents=True, exist_ok=False)
        receipt = {
            "verified": True,
            "report_sha256": canonical_digest(json.loads(original_raw)),
            "execution": report["execution"],
        }
        (output / "audit_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    else:
        write_outputs(output, manifest, report, base)
    print(
        json.dumps(
            {
                "output": str(output),
                "raw_reconstruction_completed": True,
                "authenticated_cases": sum(
                    r.get("native_measurement_qualified") is True
                    for r in base["graded_episodes"]
                ),
                "unavailable_or_unqualified_cases": len(base["models"])
                * base["n_expected"]
                - sum(
                    r.get("native_measurement_qualified") is True
                    for r in base["graded_episodes"]
                ),
                "capability_ranking_ready": False,
                "execution": report["execution"],
            }
        )
    )


if __name__ == "__main__":
    main()
