#!/usr/bin/env python3
"""Reference-anchored 0.31 outcomes from authenticated 0.30 trajectory evidence.

The base 0.30 report is either rebuilt from a raw input manifest or supplied as
a hash-pinned report (descriptive; recorded as not re-authenticated here).
An optional domain specification substitutes explicitly mapped corrected-retest
rows, each rebuilt from its own raw manifest. No provider call, native replay
or source construction is made.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from evaluation.process_scorecard031 import build_process_scorecard  # noqa: E402
from evaluation.reference_anchors031 import validate_reference_anchors  # noqa: E402
from evaluation.task_outcome031 import (  # noqa: E402
    VERSION_0312,
    build_anchored_outcomes,
    load_policy031,
    substitute_domain_rows,
)
from evaluation.trajectory_outcome027 import canonical_digest  # noqa: E402
from evaluation.uncertainty031 import (  # noqa: E402
    bootstrap_uncertainty,
    domain_variance_share,
    influence_diagnostics,
    retest_agreement,
)
from scripts.evaluate_task_outcomes import evaluate as evaluate030  # noqa: E402

DEFAULT_POLICY = ROOT / "evaluation/policies/lite141_0312.json"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _report030_from_manifest(
    path: Path, artifact_root: Path | None
) -> tuple[dict, dict]:
    raw = path.read_bytes()
    report, _ = evaluate030(json.loads(raw), root=ROOT, artifact_root=artifact_root)
    if path.read_bytes() != raw:
        raise ValueError("input_manifest_changed_during_evaluation")
    manifest = json.loads(raw)
    binding = dict(
        kind="raw_manifest_reauthenticated",
        path=str(path),
        sha256=_sha256(raw),
        report_canonical_sha256=canonical_digest(report),
        raw_reauthenticated=True,
    )
    return report, {
        **binding,
        "journals": [r["episodes"]["path"] for r in manifest["runs"]],
    }


def _report030_from_file(path: Path, expected_sha256: str) -> tuple[dict, dict]:
    raw = path.read_bytes()
    if _sha256(raw) != expected_sha256:
        raise ValueError("pinned_report_sha256_mismatch")
    report = json.loads(raw)
    return report, dict(
        kind="hash_pinned_report_not_reauthenticated_here",
        path=str(path),
        sha256=expected_sha256,
        raw_reauthenticated=False,
        journals=[],
    )


def load_inputs(args) -> tuple[dict, dict, dict | None, list]:
    if args.manifest:
        base, base_binding = _report030_from_manifest(args.manifest, args.artifact_root)
    else:
        base, base_binding = _report030_from_file(
            args.base_report, args.base_report_sha256
        )
    historical, spec_binding, supplements = None, None, []
    if args.domain_spec:
        raw = args.domain_spec.read_bytes()
        spec = json.loads(raw)
        for item in spec["supplements"]:
            path = Path(item["manifest"])
            report, binding = _report030_from_manifest(
                path if path.is_absolute() else ROOT / path, args.artifact_root
            )
            supplements.append(
                dict(
                    report=report,
                    sha256=binding["report_canonical_sha256"],
                    label=item["label"],
                    models=item["models"],
                    binding=binding,
                )
            )
        historical = base
        base = substitute_domain_rows(base, supplements, domain=spec["domain"])
        spec_binding = dict(
            path=str(args.domain_spec),
            sha256=_sha256(raw),
            domain=spec["domain"],
            supplements=[
                {k: s[k] for k in ("label", "models", "binding")} for s in supplements
            ],
        )
    return base, base_binding, (historical, spec_binding), supplements


def substitute_cases(base: dict, spec_path: Path, artifact_root: Path | None):
    """Replace explicitly listed whole episodes with reauthenticated supplements.

    Every listed case must exist in the base and be evidence-qualified in its
    supplement; nothing else changes. Provenance keeps the superseded digest.
    """
    raw = spec_path.read_bytes()
    spec = json.loads(raw)
    report = deepcopy(base)
    index = {
        (r["model"], r["scenario_signature"], r["seed"]): i
        for i, r in enumerate(report["graded_episodes"])
    }
    bindings, journals, changed = [], [], []
    for item in spec["supplements"]:
        path = Path(item["manifest"])
        supplement, binding = _report030_from_manifest(
            path if path.is_absolute() else ROOT / path, artifact_root
        )
        if supplement["source_contracts"] != base["source_contracts"]:
            raise ValueError("case_supplement_source_contract_mismatch")
        rows = {
            (r["model"], r["scenario_signature"], r["seed"]): r
            for r in supplement["graded_episodes"]
            if (r.get("outcome030") or {}).get("evidence_qualified") is True
        }
        for case in item["cases"]:
            target = (case["model"], case["scenario_signature"], case["seed"])
            source = (case.get("supplement_model", case["model"]), *target[1:])
            if target not in index or source not in rows:
                raise ValueError(f"case_substitution_unavailable:{target}")
            previous = report["graded_episodes"][index[target]]
            replacement = deepcopy(rows[source])
            replacement["model"] = target[0]
            replacement["substitution"] = dict(
                label=item["label"],
                source_report_sha256=binding["report_canonical_sha256"],
                supplement_model=source[0],
                superseded_episode_sha256=(previous.get("origin") or {}).get(
                    "canonical_episode_sha256"
                ),
                interpretation="descriptive_cross_implementation_whole_episode_substitution",
            )
            report["graded_episodes"][index[target]] = replacement
            changed.append(dict(zip(("model", "scenario_signature", "seed"), target)))
        bindings.append(dict(label=item["label"], binding=binding, cases=item["cases"]))
        journals += binding["journals"]
    return (
        report,
        dict(
            path=str(spec_path),
            sha256=_sha256(raw),
            supplements=bindings,
            cases=changed,
        ),
        journals,
    )


def _episodes(paths: list[str]) -> dict:
    episodes = {}
    for path in paths:
        resolved = Path(path)
        if not resolved.exists():
            continue
        for line in resolved.read_text().splitlines():
            if line.strip():
                episode = json.loads(line)
                episodes[canonical_digest(episode)] = episode
    return episodes


def evaluate(args) -> dict:
    before = implementation_identity(ROOT)["implementation_tree_sha256"]
    # This newer entrypoint is not in the shared historical runtime-file list.
    entrypoint_hash = _sha256(Path(__file__).read_bytes())
    policy = load_policy031(args.policy, root=ROOT)
    base, base_binding, (historical, spec_binding), supplements = load_inputs(args)
    case_binding, case_journals = None, []
    if args.case_spec:
        base, case_binding, case_journals = substitute_cases(
            base, args.case_spec, args.artifact_root
        )
    report = build_anchored_outcomes(base, policy)
    report["input"] = dict(
        base=base_binding,
        domain_substitution=spec_binding,
        case_substitution=case_binding,
        policy=dict(path=str(args.policy), sha256=_sha256(args.policy.read_bytes())),
    )
    if report["evaluation_version"] == VERSION_0312:
        report["certification"] = certification(base, report["input"])
        report["rank_sensitivity"] = rank_sensitivity(report)
    clusters = {
        (c["scenario_signature"], c["seed"]): c["physical_source_cluster"]
        or f"unidentified:{c['scenario_signature']}:{c['seed']}"
        for c in policy["weight_manifest"]["cases"]
    }
    report["uncertainty"] = bootstrap_uncertainty(
        report,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        clusters=clusters,
    )
    report["uncertainty_case_bootstrap"] = bootstrap_uncertainty(
        report, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
    )
    report["domain_variance_share"] = domain_variance_share(report)
    report["influence"] = influence_diagnostics(report)
    if historical is not None:
        before_substitution = build_anchored_outcomes(
            historical, policy, sensitivity=False
        )
        report["test_retest"] = retest_agreement(
            before_substitution["cases"], report["cases"], domain=spec_binding["domain"]
        )
        report["historical_domain_models"] = [
            {k: m[k] for k in ("model", "primary_score", "primary_rank", "complete")}
            for m in before_substitution["models"]
        ]
    journals = [
        *base_binding["journals"],
        *[j for s in supplements for j in s["binding"]["journals"]],
        *case_journals,
        *map(str, args.episodes_journal),
    ]
    episodes = _episodes(journals)
    anchors = validate_reference_anchors(
        policy["reference_anchors"], base["source_contracts"]
    )
    scenarios = {
        (c["scenario_signature"], c["seed"]): yaml.safe_load(
            (ROOT / c["scenario_path"]).read_bytes()
        )
        for c in base["source_contracts"]["contracts"]
    }
    weights = {
        (c["scenario_signature"], c["seed"]): c["score_weight"]
        for c in policy["weight_manifest"]["cases"]
    }
    report["process_scorecard"] = build_process_scorecard(
        report, episodes, scenarios, anchors, weights
    )
    report["execution"] = dict(
        provider_calls=0, native_episode_replays=0, source_constructors=0
    )
    report["evaluator_implementation_tree_sha256"] = before
    if report["evaluation_version"] == VERSION_0312:
        report["evaluator_entrypoint_sha256"] = entrypoint_hash
    if implementation_identity(ROOT)["implementation_tree_sha256"] != before:
        raise ValueError("outcome031_runtime_changed_during_evaluation")
    if _sha256(Path(__file__).read_bytes()) != entrypoint_hash:
        raise ValueError("outcome031_entrypoint_changed_during_evaluation")
    return report


def certification(base: dict, inputs: dict) -> dict:
    """Rescoring and complete coverage never certify a historical formal run."""
    substituted = bool(inputs["domain_substitution"] or inputs["case_substitution"])
    raw = inputs["base"]["raw_reauthenticated"] and all(
        s["binding"]["raw_reauthenticated"]
        for name in ("domain_substitution", "case_substitution")
        for s in (inputs[name] or {}).get("supplements", [])
    )
    formal = base.get("formal_run_certified") is True and not substituted
    same_run = base.get("same_run_141_merge_certified") is True and not substituted
    reasons = []
    if not raw:
        reasons.append("base_report_hash_pinned_not_raw_reauthenticated")
    if substituted:
        reasons.append("explicit_cross_execution_episode_substitutions")
    if base.get("formal_run_certified") is not True:
        reasons.append("source_report_not_formal_run_certified")
    return dict(
        ranking_scope="fixed_panel_retrospective",
        raw_reauthenticated=raw,
        formal_run_certified=formal and raw,
        same_run_141_merge_certified=same_run and raw,
        reasons=reasons,
        rank_interpretation="point_order_not_statistically_resolved_total_order",
        held_out_generalization_certified=False,
    )


def rank_sensitivity(report: dict) -> dict:
    """Expose rank movements, not just correlations that can hide them."""
    complete = {m["model"]: m for m in report["models"] if m["complete"]}
    variants = {}
    for name, values in report["sensitivity"].items():
        if not name.startswith(("weights:", "exclude_domain:")):
            continue
        ordered = sorted(complete, key=lambda m: (-values[m], m))
        ranks, previous, rank = {}, None, None
        for position, model in enumerate(ordered, 1):
            if values[model] != previous:
                rank = position
            ranks[model], previous = rank, values[model]
        variants[name] = ranks
    rows = []
    for model, row in complete.items():
        ranks = {name: v[model] for name, v in variants.items()}
        all_ranks = [row["primary_rank"], *ranks.values()]
        rows.append(
            dict(
                model=model,
                point_rank=row["primary_rank"],
                specification_rank_range=[min(all_ranks), max(all_ranks)],
                max_rank_shift=max(abs(r - row["primary_rank"]) for r in all_ranks),
                variant_ranks=ranks,
            )
        )
    return dict(
        interpretation="declared_weight_and_domain_scope_sensitivity_not_a_confidence_interval",
        models=sorted(rows, key=lambda r: (r["point_rank"], r["model"])),
    )


def _fmt(value) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def table_rows(report: dict) -> list[dict]:
    uncertainty = {m["model"]: m for m in report["uncertainty"]["models"]}
    sensitivity = report["sensitivity"]
    rows = []
    for m in report["models"]:
        u = uncertainty.get(m["model"], {})
        rows.append(
            dict(
                rank=m["primary_rank"],
                letters=u.get("letters"),
                heuristic_group=u.get("heuristic_group"),
                model=m["model"],
                score031=m["primary_score"],
                interval95_low=(u.get("interval95") or [None, None])[0],
                interval95_high=(u.get("interval95") or [None, None])[1],
                rank_interval=json.dumps(u.get("rank_interval95")),
                n_scored=m["n_scored"],
                n_expected=m["n_expected"],
                bounds_low=m["missing_measurement_bounds"][0],
                bounds_high=m["missing_measurement_bounds"][1],
                hard_failure_cases=m["hard_failure_cases"],
                Q030=m["Q030"],
                excluding_driving=sensitivity["exclude_domain:autonomous_driving"].get(
                    m["model"]
                ),
                **{f"domain:{d}": v["score"] for d, v in m["domains"].items()},
            )
        )
    return sorted(rows, key=lambda r: (r["rank"] is None, r["rank"] or 0, r["model"]))


def render(report: dict) -> str:
    semantics = report["anchor_semantics"]
    uncertainty = report["uncertainty"]
    lines = [
        f"# OPERATE {report['evaluation_version']} reference-anchored outcomes",
        "",
        "Per case: 0 = seeded do-nothing controller; 100 = closing the do-nothing-to-reference",
        "cost gap. The joint score reaches 100 only with full service: the reference itself scores",
        f"100 on {semantics['reference_joint_score_100_cases']} of {semantics['eligible_cases']} "
        "eligible cases. Upper anchors are heuristics, "
        f"{semantics['privileged_upper_anchor_cases']} of them privileged (future/native information),",
        "so scores are not a percentage of optimal capability. Verified hard failure = -100.",
        "",
        "Letters: models sharing a letter are not resolved at "
        f"P >= {uncertainty['resolution_probability']:g} "
        f"({uncertainty['resolved_pairs']}/{uncertainty['total_pairs']} pairs resolved). Intervals come",
        "from a domain-stratified source-cluster bootstrap of tasks, not from repeated runs.",
        "",
        "| Rank | Letters | Model | Score | 95% task interval | Coverage | Excl. driving (co-primary) | 0.30 Q |",
        "|---:|---|---|---:|---|---:|---:|---:|",
    ]
    cert = report.get("certification")
    if cert:
        lines[2:4] = [
            "Per case: 0 = seeded do-nothing controller. 100 requires full service where applicable,",
            "and reference cost or better on an informative cost axis (otherwise the cost regression",
            "guard applies). The reference itself scores",
        ]
        lines[2:2] = [
            "Scope: fixed-panel retrospective point ranking. "
            f"Raw evidence reauthenticated: {cert['raw_reauthenticated']}. "
            f"Formal run certified: {cert['formal_run_certified']}; "
            f"same-run merge certified: {cert['same_run_141_merge_certified']}.",
            "Bootstrap letters are descriptive pairwise stability summaries without "
            "family-wise multiple-comparison correction. They do not establish equivalence.",
            "",
        ]
    for r in table_rows(report):
        interval = (
            "N/A"
            if r["interval95_low"] is None
            else f"[{r['interval95_low']:.1f}, {r['interval95_high']:.1f}]"
        )
        lines.append(
            f"| {r['rank'] or 'N/A'} | {r['letters'] or 'N/A'} | {r['model']} | {_fmt(r['score031'])} | "
            f"{interval} | {r['n_scored']}/{r['n_expected']} | {_fmt(r['excluding_driving'])} | "
            f"{_fmt(r['Q030'])} |"
        )
    shares = report["domain_variance_share"].get("shares") or {}
    lines += [
        "",
        "Domain share of between-model variance: "
        + ", ".join(f"{d} {v:.2f}" for d, v in shares.items() if v is not None),
    ]
    retest = report.get("test_retest")
    if retest:
        lines.append(
            "Driving cross-harness agreement (not same-condition repeatability): case Pearson "
            f"{_fmt(retest['case_pearson'])}, model Spearman {_fmt(retest['model_mean_spearman'])}."
        )
    influence = report.get("influence") or {}
    if influence.get("leave_one_case_out"):
        loo = influence["leave_one_case_out"]
        lines.append(
            "Leave-one-case-out: rank-shift distribution "
            f"{loo['rank_shift_distribution']}, minimum Spearman {loo['min_spearman']:.3f}; "
            f"top {len(influence['case_variance_share']['top'])} cases carry "
            f"{influence['case_variance_share']['top_share_total']:.1%} of between-model variance."
        )
    if report.get("rank_sensitivity"):
        lines += [
            "",
            "Specification rank ranges (weights and domain exclusions; not confidence intervals):",
        ]
        lines += [
            f"- {r['model']}: {r['specification_rank_range'][0]}–{r['specification_rank_range'][1]}."
            for r in report["rank_sensitivity"]["models"]
        ]
    return "\n".join(lines) + "\n"


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(output: Path, report: dict) -> None:
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("report.json", report),
        ("uncertainty.json", report["uncertainty"]),
        ("uncertainty_case_bootstrap.json", report["uncertainty_case_bootstrap"]),
        ("influence.json", report["influence"]),
        ("sensitivity.json", report["sensitivity"]),
        ("process_scorecard.json", report["process_scorecard"]),
    ):
        (output / name).write_text(
            json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
        )
    (output / "report.md").write_text(render(report))
    for name in ("certification", "rank_sensitivity"):
        if name in report:
            (output / f"{name}.json").write_text(
                json.dumps(report[name], indent=1, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
    _csv(output / "table.csv", table_rows(report))
    case_fields = (
        "model",
        "scenario_signature",
        "seed",
        "domain",
        "task_family",
        "backend_kind",
        "headline_eligible",
        "S",
        "g",
        "f",
        "binding_axis",
        "reason",
        "C",
        "C_wait",
        "C_ref",
        "ref_policy",
        "F",
        "F_wait",
        "hard_failure",
        "Q030",
        "canonical_episode_sha256",
    )
    _csv(
        output / "cases.csv",
        [{k: c.get(k) for k in case_fields} for c in report["cases"]],
    )
    _csv(
        output / "process_summary.csv",
        [
            dict(
                model=m["model"],
                episodes_available=m["episodes_available"],
                **{f"{k}": v["value"] for k, v in m["metrics"].items()},
                **{
                    f"{k}_coverage": f"{v['measured_cases']}/{v['applicable_cases']}"
                    for k, v in m["metrics"].items()
                },
            )
            for m in report["process_scorecard"]["models"]
        ],
    )
    checksums = {
        p.name: dict(sha256=_sha256(p.read_bytes()), bytes=p.stat().st_size)
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    (output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="raw 0.28 input manifest")
    source.add_argument("--base-report", type=Path, help="hash-pinned 0.30 report.json")
    parser.add_argument("--base-report-sha256")
    parser.add_argument(
        "--domain-spec", type=Path, help="corrected-retest substitution spec"
    )
    parser.add_argument(
        "--case-spec", type=Path, help="explicit whole-episode case substitutions"
    )
    parser.add_argument("--episodes-journal", type=Path, action="append", default=[])
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.base_report and not args.base_report_sha256:
        parser.error("--base-report requires --base-report-sha256")
    output = args.output_dir.resolve()
    if output.exists() or any(
        output.is_relative_to(ROOT / p) for p in ("release", "scenarios", "sources")
    ):
        raise ValueError("output_must_be_new_and_outside_frozen_release_inputs")
    report = evaluate(args)
    write_outputs(output, report)
    print(
        json.dumps(
            dict(
                output=str(output),
                complete_models=len(report["leaderboard"]),
                execution=report["execution"],
            )
        )
    )


if __name__ == "__main__":
    main()
