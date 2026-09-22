"""Offline fixed-anchor quality candidate; never changes live scoring or eligibility.

Inputs are explicitly prepared objective measurements, not arbitrary episode
summaries. This calculator checks matching declarations; it does not authenticate
the referenced replay artifacts or qualify anchor policies. Anchor admission is
a separate, required calibration step before any promotion.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

VERSION = "native_quality_index_candidate.v1"
MATCH_FIELDS = (
    "scenario_signature",
    "seed",
    "objective_id",
    "runtime_identity",
    "unit",
    "suite_sha256",
    "interaction_mode",
    "backend_kind",
    "source_denominator_key",
    "workload_contract_hash",
)


def _number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _ids(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def score_quality(sample: dict, contract: dict) -> dict:
    """0 = weak policy; 100 = strong reference; negative = worse than weak.

    Bounds limit a single case's contribution; raw normalized regret and native
    cost remain visible. A verified hard failure or infeasible schedule receives
    -100 when its cost and evidence are complete. Missing gates, costs, identities
    or evidence links receive no score (an evidence failure, not a task failure).
    """
    base = {"schema_version": VERSION, "formal_eligible": False}

    def unavailable(reason: str) -> dict:
        return {**base, "score": None, "reason": reason}

    for key in MATCH_FIELDS:
        value = contract.get(key)
        if key == "seed":
            valid = type(value) is int and type(sample.get(key)) is int
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid or sample.get(key) != value:
            return unavailable("objective_identity_mismatch")
    if not _ids(contract.get("anchor_evidence_ids")) or not _ids(
        sample.get("evidence_ids")
    ):
        return unavailable("evidence_links_missing")
    if contract.get("anchor_kind") != "executed_policies":
        return unavailable("unsupported_anchor_kind")
    values = [
        contract.get(key) for key in ("weak_cost", "strong_cost", "minimum_anchor_gap")
    ]
    if not all(_number(value) for value in values):
        return unavailable("invalid_anchor_costs")
    weak, strong, minimum = values
    gap = weak - strong
    if not math.isfinite(gap) or minimum <= 0 or gap < minimum:
        return unavailable("insufficient_anchor_separation")
    if any(type(sample.get(key)) is not bool for key in ("feasible", "hard_failure")):
        return unavailable("constraint_status_missing")
    actual = sample.get("actual_cost")
    if not _number(actual):
        return unavailable("native_objective_missing")
    difference = weak - actual
    if math.isfinite(difference):
        raw = 100.0 * (difference / gap)
    else:
        # Scale only the overflowing subtraction, preserving normal precision.
        scale = max(abs(weak), abs(strong), abs(actual), gap)
        raw = (
            100.0 * ((weak / scale - actual / scale) / (gap / scale))
            if gap / scale
            else math.inf
        )
    if not math.isfinite(raw):
        return unavailable("nonfinite_normalized_quality")
    failed = sample["hard_failure"] or not sample["feasible"]
    return {
        **base,
        "score": -100.0 if failed else max(-100.0, min(100.0, raw)),
        "raw_score": raw,
        "constraint_failed": failed,
        "range_clipped": raw < -100 or raw > 100,
        "actual_cost": actual,
        "weak_cost": weak,
        "strong_cost": strong,
        "reference_is_optimum": False,
        "evidence_ids": list(sample["evidence_ids"]),
        "anchor_evidence_ids": list(contract["anchor_evidence_ids"]),
    }


def summarize(samples: list[dict], contracts: list[dict], models: list[str]) -> dict:
    """Fixed complete case denominator, then source/backend/domain equal means.

    One sample per model/case/seed is required. Repeats must be analyzed in
    separate declared pass strata. No best-attempt selection or missing-case
    renormalization is performed.
    """
    expected = {}
    for contract in contracts:
        key = (contract["scenario_signature"], contract["seed"])
        if key in expected:
            raise ValueError("duplicate objective contract")
        for field in ("domain", "backend_kind", "source_denominator_key"):
            if not isinstance(contract.get(field), str) or not contract[field].strip():
                raise ValueError("missing fixed aggregation identity")
        expected[key] = contract
    if not expected or not models or len(set(models)) != len(models):
        raise ValueError("nonempty unique model and case scopes required")
    for field in ("suite_sha256", "runtime_identity", "interaction_mode"):
        if len({c.get(field) for c in contracts}) != 1:
            raise ValueError(f"mixed comparison scope: {field}")
    grouped = defaultdict(list)
    for sample in samples:
        key = (sample.get("scenario_signature"), sample.get("seed"))
        if sample.get("model") not in models or key not in expected:
            raise ValueError("sample outside declared scope")
        grouped[(sample["model"], *key)].append(sample)
    output = {}
    for model in models:
        rows = []
        source_scores = defaultdict(list)
        for key, contract in expected.items():
            attempts = grouped[(model, *key)]
            result = (
                score_quality(attempts[0], contract)
                if len(attempts) == 1
                else {
                    "score": None,
                    "reason": "missing_or_duplicate_attempt",
                }
            )
            rows.append({"scenario_signature": key[0], "seed": key[1], **result})
            if result["score"] is not None:
                source_scores[
                    tuple(
                        contract[k]
                        for k in ("domain", "backend_kind", "source_denominator_key")
                    )
                ].append(result["score"])
        complete = all(row["score"] is not None for row in rows)
        backend_scores = defaultdict(list)
        for (domain, backend, _), scores in source_scores.items():
            backend_scores[(domain, backend)].append(mean(scores))
        domain_scores = defaultdict(list)
        for (domain, _), scores in backend_scores.items():
            domain_scores[domain].append(mean(scores))
        output[model] = {
            "index": mean(mean(scores) for scores in domain_scores.values())
            if complete
            else None,
            "domain_scores": {d: mean(v) for d, v in domain_scores.items()}
            if complete
            else None,
            "upper_bound_fraction": mean(row["score"] == 100 for row in rows)
            if complete
            else None,
            "lower_bound_fraction": mean(row["score"] == -100 for row in rows)
            if complete
            else None,
            "constraint_failure_fraction": mean(
                row["constraint_failed"] for row in rows
            )
            if complete
            else None,
            "complete": complete,
            "n_expected": len(expected),
            "n_measured": sum(row["score"] is not None for row in rows),
            "rows": rows,
        }
    pairwise = []
    for left, right in combinations(models, 2):
        a, b = output[left], output[right]
        complete = a["complete"] and b["complete"]
        pairs = list(zip(a["rows"], b["rows"]))
        pairwise.append(
            {
                "left": left,
                "right": right,
                "index_difference": a["index"] - b["index"] if complete else None,
                "n_clipping_hidden_differences": sum(
                    x["score"] == y["score"]
                    and x["raw_score"] != y["raw_score"]
                    and (x["range_clipped"] or y["range_clipped"])
                    and not x["constraint_failed"]
                    and not y["constraint_failed"]
                    for x, y in pairs
                )
                if complete
                else None,
                "inference": "descriptive_only_no_significance_claim",
            }
        )
    return {
        "schema_version": VERSION,
        "formal_eligible": False,
        "pairwise": pairwise,
        "aggregation": "effective_source_backend_domain_macro_v1",
        "score_range": [-100, 100],
        "models": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON: samples, contracts, models")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    payload = json.loads(raw)
    report = summarize(payload["samples"], payload["contracts"], payload["models"])
    report["input_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
