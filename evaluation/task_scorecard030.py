"""Shared fixed-weight diagnostics for the versioned task outcome corrections."""

from copy import deepcopy

from evaluation.outcome_analysis029 import build_outcome_analysis
from evaluation.retrospective_ranking029 import build_retrospective_ranking
from evaluation.trajectory_outcome027 import canonical_digest


def analyze_task_outcomes(report: dict) -> dict:
    """Reuse structural diagnostics without changing or exporting legacy scores.

    The internal projection supplies the established engine's field names only;
    its utilities are the authenticated 0.30 outcomes. It is never an 0.28
    evidence report and is not serialized as one.
    """
    if report.get("evaluation_version") != "0.30.0":
        raise ValueError("task_analysis_requires_outcome030")
    projected = {
        **report,
        "evaluation_version": "0.28.0",
        "graded_episodes": [
            {
                **row,
                "outcome028": row["outcome030"],
                "normalization028": row.get("normalization030"),
            }
            for row in report["graded_episodes"]
        ],
    }
    analysis = build_outcome_analysis(projected)
    analysis.update(
        evaluation_version="0.30.0",
        base_evaluation_version="0.30.0",
        headline_score_changed=False,
    )
    ranking = build_retrospective_ranking(analysis)
    ranking.update(evaluation_version="0.30.0", base_evaluation_version="0.30.0")
    return {"outcome_analysis": analysis, "retrospective_ranking": ranking}


def score_changes(base: dict, report: dict) -> dict:
    """Decompose the exact fixed-weight delta; no model-fitted calibration."""
    weights = {
        (r["scenario_signature"], r["seed"]): r["score_weight"]
        for r in report["task_weight_manifest"]["cases"]
    }
    old = {r["model"]: r for r in base["models"]}
    models = []
    cases = []
    for row in report["graded_episodes"]:
        prior, current = row.get("legacy_Q028"), row.get("Q")
        if all(
            row.get(f"legacy_{metric}028") == row.get(metric)
            for metric in ("Q", "N", "F")
        ):
            continue
        weight = weights[row["scenario_signature"], row["seed"]]
        cases.append(
            {
                "model": row["model"],
                "scenario_signature": row["scenario_signature"],
                "seed": row["seed"],
                "backend_kind": row["backend_kind"],
                "weight": weight,
                "Q028": prior,
                "Q030": current,
                "weighted_delta": weight * (current - prior)
                if prior is not None and current is not None
                else None,
                "N028": row.get("legacy_N028"),
                "N030": row.get("N"),
                "F028": row.get("legacy_F028"),
                "F030": row.get("F"),
                "evidence_ids": deepcopy(row["outcome030"]["evidence_ids"]),
            }
        )
    for row in report["models"]:
        prior = old[row["model"]]
        q0, q1 = prior["primary_score"], row["primary_score"]
        models.append(
            {
                "model": row["model"],
                "Q028": q0,
                "Q030": q1,
                "delta": q1 - q0 if q0 is not None and q1 is not None else None,
                "rank028": prior["primary_rank"],
                "rank030": row["primary_rank"],
            }
        )
    return {
        "base_version": "0.28.0",
        "evaluation_version": "0.30.0",
        "models": models,
        "changed_cases": cases,
        "weight_manifest_sha256": report["task_weight_manifest"]["manifest_sha256"],
        "source_contracts_sha256": canonical_digest(report["source_contracts"]),
        "historical_feedback_repaired": False,
    }
