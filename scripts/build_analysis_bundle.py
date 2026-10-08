#!/usr/bin/env python3
"""Read-only analysis of a checksum-bound retrospective 0.30 result package.

This consumes an existing scoring package; it is not a new scorer or a formal
same-run certification. Original journals and selected whole episodes are
verified before attaching behavioral diagnostics. No provider or native backend
is imported or invoked.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = (
    "deepseek-ai/deepseek-v4-pro-0813",
    "deepseek-ai/deepseek-v4.1-flash",
    "deepseek-ai/deepseek-v4-flash-0731",
    "moonshotai/kimi-k3",
    "glm-5.2",
    "zai-org/glm-5.3",
    "Qwen/Qwen3.6-27B",
    "qwen/qwen3.8-27b",
    "qwen/qwen3.8-max",
    "gpt-6.1-sol",
    "gpt-6-sol",
    "gpt-6-luna",
)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bound_path(path, root):
    path = Path(path)
    return path if path.is_absolute() else root / path


def verify_file(path, expected_sha, expected_size=None):
    path = Path(path)
    if not expected_sha or not path.is_file():
        raise ValueError(f"missing_bound_file: {path}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise ValueError(f"bound_file_size_mismatch: {path}")
    actual = sha(path)
    if actual != expected_sha:
        raise ValueError(f"bound_file_hash_mismatch: {path}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def verify_package(directory):
    directory = Path(directory)
    checksums = json.loads((directory / "checksums.json").read_bytes())
    required = {"report.json", "input_manifest.json", "outcome028.json"}
    if not required <= checksums.keys():
        raise ValueError("incomplete_result_checksum_inventory")
    records = []
    for name, spec in checksums.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("checksum_path_outside_package")
        records.append(verify_file(path, spec["sha256"], spec.get("bytes")))
    report = json.loads((directory / "report.json").read_bytes())
    if report.get("evaluation_version") != "0.30.0":
        raise ValueError("analysis_requires_030")
    return report, records


def select_original(row, lines):
    origin = row["origin"]
    line = origin.get("line")
    if type(line) is not int or not 1 <= line <= len(lines):
        raise ValueError("original_episode_line_out_of_range")
    episode = json.loads(lines[line - 1])
    if digest(episode) != origin["canonical_episode_sha256"]:
        raise ValueError("original_episode_digest_mismatch")
    if any(episode.get(k) != row[k] for k in ("scenario_signature", "seed")):
        raise ValueError("original_case_identity_mismatch")
    if episode.get("execution_attempt_id") != origin.get("execution_attempt_id"):
        raise ValueError("original_attempt_identity_mismatch")
    if episode.get("model") != origin.get("actual_model"):
        raise ValueError("original_model_identity_mismatch")
    return episode


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def weighted_summary(rows, field):
    """A fixed-slice point exists only when all slice measurements are known."""
    total = math.fsum(r["weight"] for r in rows)
    measured = [r for r in rows if finite(r.get(field))]
    mass = math.fsum(r["weight"] for r in measured)
    return {
        "value": math.fsum(r["weight"] * r[field] for r in measured) / total
        if total and len(measured) == len(rows)
        else None,
        "expected_cases": len(rows),
        "measured_cases": len(measured),
        "expected_weight": total,
        "measured_weight": mass,
    }


def validate_analysis_cases(cases, contracts, weights):
    keys = [(c["scenario_signature"], c["seed"]) for c in cases]
    if len(set(keys)) != len(keys) or set(keys) != set(contracts):
        raise ValueError("analysis_case_identity_coverage_mismatch")
    for case, key in zip(cases, keys, strict=True):
        value = case.get("weight")
        if not finite(value) or value <= 0 or value != weights[key]:
            raise ValueError("analysis_frozen_weight_mismatch")


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False, allow_nan=False)
                    if isinstance(v, (dict, list))
                    else v
                    for k, v in row.items()
                }
            )


def build_bundle(report, *, models, root=ROOT, extractor=None):
    from evaluation.operational_weights import validate_task_weights

    weights = validate_task_weights(
        report["task_weight_manifest"], report["source_contracts"]
    )
    declared = {m["model"]: m for m in report["models"]}
    if len(set(models)) != len(models) or not set(models) <= declared.keys():
        raise ValueError("unknown_or_duplicate_model_selection")
    analysis = {m["model"]: m for m in report["outcome_analysis"]["models"]}
    contracts = {
        (c["scenario_signature"], c["seed"]): c
        for c in report["source_contracts"]["contracts"]
    }
    rows = {
        (r["model"], r["scenario_signature"], r["seed"]): r
        for r in report["graded_episodes"]
        if r["model"] in models
    }
    expected = len(models) * len(contracts)
    if len(rows) != expected:
        raise ValueError("selected_case_matrix_incomplete_or_duplicate")
    selected = [r for r in report["graded_episodes"] if r["model"] in models]
    if len(selected) != expected:
        raise ValueError("duplicate_selected_case")
    verified, journals, sources = {}, {}, {}
    for row in selected:
        origin = row.get("origin") or {}
        for kind in ("journal", "config"):
            spec = origin.get(kind)
            if not spec:
                raise ValueError(f"missing_original_{kind}")
            p = bound_path(spec["path"], root)
            key = (str(p), spec["sha256"])
            if key not in verified:
                verified[key] = verify_file(p, spec["sha256"], spec.get("byte_count"))
        p = bound_path(origin["journal"]["path"], root)
        if str(p) not in journals:
            journals[str(p)] = p.read_bytes().splitlines()
    import yaml

    for key, contract in contracts.items():
        path = bound_path(contract["scenario_path"], root)
        spec = verify_file(path, contract["scenario_sha256"])
        verified[(str(path), spec["sha256"])] = spec
        scenario = yaml.safe_load(path.read_text())
        sources[key] = {
            "difficulty_mode": scenario.get("difficulty_mode"),
            "difficulty_level": scenario.get("difficulty_level"),
        }
    cases, behavior, timeline, lifecycle, evidence, issues = [], [], [], [], [], []
    summary = []
    for name in models:
        original = declared[name]
        model_cases = analysis[name]["cases"]
        validate_analysis_cases(model_cases, contracts, weights)
        if len(model_cases) != len(contracts):
            raise ValueError("analysis_case_denominator_mismatch")
        for case in model_cases:
            key = (case["scenario_signature"], case["seed"])
            row = rows[(name, *key)]
            contract = contracts[key]
            if any(case.get(k) != row.get(k) for k in ("Q", "N", "C")):
                raise ValueError("cached_analysis_metric_mismatch")
            c = {
                **case,
                "model": name,
                "backend_kind": contract["backend_kind"],
                "horizon_ticks": contract["horizon_ticks"],
                "source_denominator_key": contract["source_denominator_key"],
                "Q028": row.get("legacy_Q028"),
                "N028": row.get("legacy_N028"),
                "F028": row.get("legacy_F028"),
                **sources[key],
            }
            cases.append(c)
            path = bound_path(row["origin"]["journal"]["path"], root)
            episode = select_original(row, journals[str(path)])
            context = {
                "model": name,
                "scenario_signature": key[0],
                "seed": key[1],
                "domain": c["domain"],
                "backend_kind": c["backend_kind"],
                "task_family": c["task_family"],
                "weight": c["weight"],
                "Q": c["Q"],
                "horizon_ticks": c["horizon_ticks"],
            }
            if extractor:
                extracted = extractor(episode, root=root, journal_dir=path.parent)
                behavior.append(
                    {
                        **extracted["summary"],
                        **context,
                        "execution_attempt_id": episode.get("execution_attempt_id"),
                        "origin": row["origin"],
                    }
                )
                timeline.extend({**v, **context} for v in extracted["time_rows"])
                lifecycle.extend({**v, **context} for v in extracted["lifecycle_rows"])
                evidence.extend({**v, **context} for v in extracted["evidence"])
                issues.extend({"issue": v, **context} for v in extracted["issues"])
        q = weighted_summary([c for c in cases if c["model"] == name], "Q")
        if original["primary_score"] is not None and (
            q["value"] is None
            or not math.isclose(q["value"], original["primary_score"], abs_tol=1e-10)
        ):
            raise ValueError("weighted_Q_does_not_reproduce_report")
        summary.append(
            {
                "model": name,
                "Q": original["primary_score"],
                "N": original["native_quality"],
                "F": original["completion_only"]["score"],
                "original_rank": original["primary_rank"],
                "n_scored": original["n_scored"],
                "n_expected": original["n_expected"],
                "hard_failure_rate": original["recorded_hard_failure_rate"],
                "safety_coverage": original["safety_coverage"],
            }
        )
        print(f"analyzed {name}: {len(model_cases)} cases", file=sys.stderr, flush=True)
    groups = defaultdict(list)
    for c in cases:
        for field in (
            "domain",
            "task_family",
            "backend_kind",
            "difficulty_mode",
            "difficulty_level",
        ):
            groups[c["model"], field, str(c.get(field))].append(c)
    slices = [
        {
            "model": model,
            "field": field,
            "stratum": stratum,
            **{metric: weighted_summary(rs, metric) for metric in ("Q", "N")},
        }
        for (model, field, stratum), rs in sorted(groups.items())
    ]
    paired = []
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            for key, contract in contracts.items():
                left, right = rows[(a, *key)], rows[(b, *key)]
                weight = next(
                    c["weight"]
                    for c in analysis[a]["cases"]
                    if (c["scenario_signature"], c["seed"]) == key
                )
                delta = (
                    left["Q"] - right["Q"]
                    if finite(left.get("Q")) and finite(right.get("Q"))
                    else None
                )
                paired.append(
                    {
                        "model_a": a,
                        "model_b": b,
                        "scenario_signature": key[0],
                        "seed": key[1],
                        "domain": contract["domain"],
                        "task_family": contract["task_family"],
                        "physical_source_cluster": contract["physical_source_cluster"],
                        "weight": weight,
                        "delta_Q": delta,
                        "weighted_delta_Q": weight * delta
                        if delta is not None
                        else None,
                    }
                )
    # Recheck originals to detect concurrent changes while reading diagnostics.
    for record in verified.values():
        verify_file(record["path"], record["sha256"], record["bytes"])
    return {
        "schema_version": "operate_offline_analysis030.v1",
        "evaluation_version": "0.30.0",
        "interpretation": "selected_recorded_outcomes_not_new_scoring_or_same_run_certification",
        "models": summary,
        "cases": cases,
        "behavior": behavior,
        "timeline": timeline,
        "lifecycle": lifecycle,
        "evidence": evidence,
        "issues": issues,
        "strata": slices,
        "pairwise_case_differences": paired,
        "rank_sensitivity": report["outcome_analysis"]["rank_sensitivity"],
        "rank_sensitivity_cohort": "original_complete_cohort_not_selected_12_reranking",
        "native_pairwise": [
            p
            for p in report["outcome_analysis"]["pairwise"]
            if p.get("model_a", p.get("a")) in models
            and p.get("model_b", p.get("b")) in models
        ],
        "score_changes": report["score_changes"],
        "verified_inputs": list(verified.values()),
        "execution": {
            "provider_calls": 0,
            "native_replays": 0,
            "environment_constructors": 0,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() or any(
        out.is_relative_to(ROOT / p) for p in ("release", "scenarios", "sources")
    ):
        raise ValueError("output_must_be_new_outside_frozen_inputs")
    report, package = verify_package(args.result_dir)
    from evaluation.offline_behavior import extract_behavior

    bundle = build_bundle(report, models=args.models, extractor=extract_behavior)
    for record in package:
        verify_file(record["path"], record["sha256"], record["bytes"])
    bundle["result_package"] = package
    bundle["analysis_runtime"] = {
        str(p.relative_to(ROOT)): sha(p)
        for p in [Path(__file__).resolve(), ROOT / "evaluation/offline_behavior.py"]
    }
    out.mkdir(parents=True)
    (out / "bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    for key in (
        "models",
        "cases",
        "behavior",
        "timeline",
        "lifecycle",
        "issues",
        "strata",
        "pairwise_case_differences",
    ):
        write_csv(out / f"{key}.csv", bundle[key])
    checks = {
        p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(out.iterdir())
        if p.is_file()
    }
    (out / "checksums.json").write_text(json.dumps(checks, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(out),
                "models": len(bundle["models"]),
                "cases": len(bundle["cases"]),
                "behavior": len(bundle["behavior"]),
                "timeline_rows": len(bundle["timeline"]),
                "issues": len(bundle["issues"]),
            }
        )
    )


if __name__ == "__main__":
    main()
