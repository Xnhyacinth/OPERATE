#!/usr/bin/env python3
"""Reconstruct versioned 0.30 task outcomes from original trajectory evidence.

No cached score report is accepted as an input. No provider or native replay
is invoked. Original 0.28 outcomes are exported alongside the new scoring.
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
from evaluation.task_outcome030 import build_task_outcomes  # noqa: E402
from evaluation.task_scorecard030 import analyze_task_outcomes, score_changes  # noqa: E402
from scripts.evaluate_trajectories import evaluate as evaluate_outcomes  # noqa: E402


def evaluate(manifest: dict, *, root=ROOT, artifact_root=None) -> tuple[dict, dict]:
    before = implementation_identity(root)["implementation_tree_sha256"]
    base = evaluate_outcomes(manifest, root=root, artifact_root=artifact_root)
    report = build_task_outcomes(base, root=root)
    report.update(analyze_task_outcomes(report))
    report["score_changes"] = score_changes(base, report)
    if implementation_identity(root)["implementation_tree_sha256"] != before:
        raise ValueError("task_scoring_runtime_changed_during_evaluation")
    return report, base


def summary_rows(report: dict) -> list[dict]:
    changes = {r["model"]: r for r in report["score_changes"]["models"]}
    rankings = {r["model"]: r for r in report["retrospective_ranking"]["rows"]}
    rows = []
    for model in report["models"]:
        name = model["model"]
        rank = rankings[name]
        rows.append(
            {
                "model": name,
                "rank": model["primary_rank"],
                "Q": model["primary_score"],
                "N": model["native_quality"],
                "F": model["completion_only"]["score"],
                "n_scored": model["n_scored"],
                "n_expected": model["n_expected"],
                "old_Q": changes[name]["Q028"],
                "old_rank": changes[name]["rank028"],
                "delta": changes[name]["delta"],
                "Q_lower": rank["measurement_bounds"][0],
                "Q_upper": rank["measurement_bounds"][1],
                "recorded_hard_failure_rate": model["recorded_hard_failure_rate"],
                "missing_reasons": json.dumps(model["reason_counts"], sort_keys=True),
                "historical_feedback_repaired": False,
            }
        )
    return sorted(rows, key=lambda r: (r["rank"] is None, r["rank"] or 0, r["model"]))


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(report: dict) -> str:
    def fmt(v):
        return "N/A" if v is None else f"{v:.4f}"

    lines = [
        "# OPERATE 0.30 fixed-panel task outcomes",
        "",
        "Task-specific outcome corrections with the original fixed hierarchical weights.",
        "Historical driving feedback remains affected by the old token clock; rescoring does not repair past decisions.",
        "Only complete models receive point ranks. Bounds describe missingness and recorded precision, not repeated-run uncertainty.",
        "",
        "| Rank | Model | Q030 | Q028 | Delta | N030 | F030 | Coverage |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows(report):
        lines.append(
            f"| {row['rank'] or 'N/A'} | {row['model']} | {fmt(row['Q'])} | "
            f"{fmt(row['old_Q'])} | {fmt(row['delta'])} | {fmt(row['N'])} | "
            f"{fmt(row['F'])} | {row['n_scored']}/{row['n_expected']} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(output: Path, manifest: dict, report: dict, base: dict) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("input_manifest.json", manifest),
        ("report.json", report),
        ("outcome028.json", base),
        ("score_changes.json", report["score_changes"]),
        ("retrospective_ranking.json", report["retrospective_ranking"]),
        ("rank_sensitivity.json", report["outcome_analysis"]["rank_sensitivity"]),
    ):
        (output / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    (output / "report.md").write_text(render(report))
    _csv(output / "table.csv", summary_rows(report))
    _csv(output / "changed_cases.csv", report["score_changes"]["changed_cases"])
    domains = []
    for model in report["outcome_analysis"]["models"]:
        for domain in model["domain"]:
            domains.append(
                {
                    "model": model["model"],
                    "domain": domain["domain"],
                    **{m: domain[m]["score"] for m in ("Q", "N", "F")},
                    "measured_cases": domain["Q"]["measured_cases"],
                    "expected_cases": domain["Q"]["expected_cases"],
                }
            )
    _csv(output / "domain_summary.csv", domains)
    checksums = {
        path.name: {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(output.iterdir())
        if path.is_file()
    }
    (output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() or any(
        output.is_relative_to(ROOT / p) for p in ("release", "scenarios", "sources")
    ):
        raise ValueError("output_must_be_new_and_outside_frozen_release_inputs")
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    report, base = evaluate(manifest, artifact_root=args.artifact_root)
    if args.manifest.read_bytes() != raw:
        raise ValueError("task_scoring_manifest_changed")
    write_outputs(output, manifest, report, base)
    print(
        json.dumps(
            {
                "output": str(output),
                "models": len(report["models"]),
                "execution": report["execution"],
            }
        )
    )


if __name__ == "__main__":
    main()
