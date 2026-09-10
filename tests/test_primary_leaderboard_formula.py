from __future__ import annotations

import pytest

from evaluation.leaderboard import (
    PRIMARY_LEADERBOARD_FORMULA_VERSION,
    PrimaryLeaderboardContractError,
    aggregate_primary_leaderboard,
    infer_primary_leaderboard,
)
from evaluation.scorer import SCORING_VERSION
from runner import EVALUATION_IMPLEMENTATION_FINGERPRINT, EVALUATION_PROTOCOL_VERSION


def _row(
    *,
    score: float,
    source: str,
    domain: str,
    backend: str,
    completion: float = 1.0,
    model: str = "model-a",
    physical_source: object | None = None,
) -> dict:
    row = {
        "model": model,
        "domain": domain,
        "backend_kind": backend,
        "source_denominator_key": source,
        "discriminative_core_score": score,
        "task_completion_raw": completion,
        "formal_score_eligible": True,
        "scoring_version": SCORING_VERSION,
        "evaluation_protocol": {
            "version": EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
    }
    if physical_source is not None:
        row["case_ledger"] = {
            "physical_source_lock": physical_source,
        }
    return row


def _imbalanced_rows() -> list[dict]:
    return [
        _row(score=100.0, source="S1", domain="D1", backend="B1"),
        _row(score=0.0, source="S1", domain="D1", backend="B1"),
        _row(score=100.0, source="S2", domain="D1", backend="B1"),
        _row(score=0.0, source="S3", domain="D1", backend="B2"),
        _row(score=100.0, source="S4", domain="D2", backend="B3"),
    ]


def test_primary_leaderboard_uses_four_level_hierarchical_macro() -> None:
    report = aggregate_primary_leaderboard(_imbalanced_rows())
    row = report["leaderboard"][0]

    assert (
        report["primary_leaderboard_formula_version"]
        == PRIMARY_LEADERBOARD_FORMULA_VERSION
        == "effective_source_backend_domain_macro_v1"
    )
    assert row["primary_leaderboard_score"] == 68.75
    assert row["diagnostic_sample_weighted_mean"] == 60.0
    assert row["domain_scores"] == {"D1": 37.5, "D2": 100.0}
    assert row["backend_scores"] == {
        "D1/B1": 75.0,
        "D1/B2": 0.0,
        "D2/B3": 100.0,
    }
    assert row["effective_source_scores"]["D1/B1/S1"] == 50.0
    assert row["primary_task_completion_rate"] == 1.0
    assert "headline" not in row
    assert "recommended_headline" not in report


def test_duplicate_sample_does_not_change_effective_source_weight() -> None:
    once = aggregate_primary_leaderboard(_imbalanced_rows())
    duplicated = aggregate_primary_leaderboard(
        [
            *_imbalanced_rows(),
            _row(score=50.0, source="S1", domain="D1", backend="B1"),
        ]
    )

    assert (
        once["leaderboard"][0]["primary_leaderboard_score"]
        == duplicated["leaderboard"][0]["primary_leaderboard_score"]
    )


def test_physical_source_macro_is_diagnostic_and_deduplicates_windows() -> None:
    report = aggregate_primary_leaderboard(
        [
            _row(
                score=100.0,
                source="S1",
                domain="D",
                backend="B",
                physical_source={"asset": "P1", "sha256": "a" * 64},
            ),
            _row(
                score=100.0,
                source="S2",
                domain="D",
                backend="B",
                physical_source={"sha256": "a" * 64, "asset": "P1"},
            ),
            _row(
                score=0.0,
                source="S3",
                domain="D",
                backend="B",
                physical_source={"asset": "P2", "sha256": "b" * 64},
            ),
        ]
    )
    row = report["leaderboard"][0]

    assert row["primary_leaderboard_score"] == pytest.approx(200.0 / 3.0)
    assert row["diagnostic_physical_source_macro_applicable"] is True
    assert row["diagnostic_physical_source_macro_score"] == 50.0
    assert row["diagnostic_physical_source_task_completion_rate"] == 1.0
    assert row["n_physical_sources"] == 2
    assert row["n_samples_missing_physical_source_identity"] == 0


def test_physical_source_macro_prefers_canonical_lock_over_legacy_key() -> None:
    physical_lock = {"asset": "P1", "sha256": "a" * 64}
    first = _row(
        score=100.0,
        source="S1",
        domain="D",
        backend="B",
        physical_source=physical_lock,
    )
    first["physical_source_key"] = "a" * 64
    second = _row(
        score=0.0,
        source="S2",
        domain="D",
        backend="B",
        physical_source=physical_lock,
    )
    second["physical_source_key"] = '{"legacy":"asset-graph"}'

    report = aggregate_primary_leaderboard([first, second])
    row = report["leaderboard"][0]

    assert row["n_physical_sources"] == 1
    assert row["diagnostic_physical_source_macro_score"] == 50.0


def test_physical_source_macro_fails_closed_when_identity_is_incomplete() -> None:
    report = aggregate_primary_leaderboard(
        [
            _row(
                score=80.0,
                source="S1",
                domain="D",
                backend="B",
                physical_source="P1",
            ),
            _row(score=20.0, source="S2", domain="D", backend="B"),
        ]
    )
    row = report["leaderboard"][0]

    assert row["primary_leaderboard_score"] == 50.0
    assert row["diagnostic_physical_source_macro_applicable"] is False
    assert row["diagnostic_physical_source_macro_score"] is None
    assert row["diagnostic_physical_source_task_completion_rate"] is None
    assert row["n_physical_sources"] is None
    assert row["n_samples_missing_physical_source_identity"] == 1


def test_primary_inference_bootstraps_the_hierarchical_estimand() -> None:
    rows = []
    for model, d1_score, d2_score in (
        ("model-a", 100.0, 0.0),
        ("model-b", 40.0, 80.0),
    ):
        for index in range(20):
            rows.append(
                _row(
                    score=d1_score,
                    source=f"D1-{index}",
                    domain="D1",
                    backend="B1",
                    model=model,
                    physical_source={"asset": f"D1-P{index}"},
                )
            )
        rows.append(
            _row(
                score=d2_score,
                source="D2-only",
                domain="D2",
                backend="B2",
                model=model,
                physical_source={"asset": "D2-P"},
            )
        )

    report = infer_primary_leaderboard(rows, n_bootstrap=200, seed=7)
    boards = {row["model"]: row for row in report["leaderboard"]}

    # Flat rows favour model-a (20 high rows vs one low row), while the
    # equal-domain primary estimand favours model-b. Inference must follow the
    # latter rather than silently reverting to a flat scenario mean.
    assert boards["model-a"]["diagnostic_sample_weighted_mean"] > boards[
        "model-b"
    ]["diagnostic_sample_weighted_mean"]
    assert boards["model-a"]["primary_leaderboard_score"] == 50.0
    assert boards["model-b"]["primary_leaderboard_score"] == 60.0
    assert all("primary_cluster_ci" in row for row in boards.values())
    pair = report["primary_pairwise"][0]
    assert pair["pair"] == ["model-a", "model-b"]
    assert pair["mean_diff"] == -10.0
    assert pair["estimand"] == PRIMARY_LEADERBOARD_FORMULA_VERSION
    assert pair["cluster_unit"] == "physical_source_lock"
    assert pair["raw_p_value_method"] in {
        "exact_cluster_label_swap",
        "monte_carlo_cluster_label_swap",
    }


def test_primary_pairwise_p_value_uses_paired_cluster_randomization() -> None:
    rows = []
    for index in range(10):
        for model, score in (("model-a", 100.0), ("model-b", 0.0)):
            rows.append(
                _row(
                    score=score,
                    source=f"source-{index}",
                    domain="D",
                    backend="B",
                    model=model,
                    physical_source={"asset": f"physical-{index}"},
                )
            )

    report = infer_primary_leaderboard(
        rows,
        n_bootstrap=50,
        n_randomization=50,
        seed=7,
    )

    pair = report["primary_pairwise"][0]
    assert pair["raw_p_value_method"] == "exact_cluster_label_swap"
    assert pair["n_randomization"] == 2**10
    assert pair["raw_p_value"] == 2 / (2**10)


def test_primary_pairwise_holm_is_reported_even_when_low_power() -> None:
    rows = []
    for index in range(3):
        for model, score in (("model-a", 100.0), ("model-b", 0.0)):
            rows.append(
                _row(
                    score=score,
                    source=f"source-{index}",
                    domain="D",
                    backend="B",
                    model=model,
                    physical_source={"asset": f"physical-{index}"},
                )
            )

    report = infer_primary_leaderboard(rows, n_bootstrap=20, seed=7)

    pair = report["primary_pairwise"][0]
    assert pair["low_power"] is True
    assert pair["holm_rank"] == 1
    assert pair["holm_adjusted_p_value"] == pair["raw_p_value"]
    assert pair["significant"] is False


def test_primary_pairwise_low_power_never_claims_significance() -> None:
    rows = []
    for index in range(6):
        for model, score in (("model-a", 100.0), ("model-b", 0.0)):
            rows.append(
                _row(
                    score=score,
                    source=f"source-{index}",
                    domain="D",
                    backend="B",
                    model=model,
                    physical_source={"asset": f"physical-{index}"},
                )
            )

    report = infer_primary_leaderboard(rows, n_bootstrap=20, seed=7)

    pair = report["primary_pairwise"][0]
    assert pair["low_power"] is True
    assert pair["holm_adjusted_p_value"] < 0.05
    assert pair["significant"] is False


def test_primary_inference_requires_physical_source_identity() -> None:
    with pytest.raises(
        PrimaryLeaderboardContractError,
        match="physical source identity",
    ):
        infer_primary_leaderboard(_imbalanced_rows(), n_bootstrap=10)


def test_primary_inference_rejects_missing_source_within_shared_cluster() -> None:
    rows = [
        _row(score=100.0, source="s1", domain="D", backend="B", model="a"),
        _row(score=0.0, source="s2", domain="D", backend="B", model="a"),
        _row(score=100.0, source="s1", domain="D", backend="B", model="b"),
    ]
    for row in rows:
        row["physical_source_key"] = "shared-physical"
        row["seed"] = 42

    with pytest.raises(
        PrimaryLeaderboardContractError,
        match="identical effective-source and repeat membership",
    ):
        infer_primary_leaderboard(rows, n_bootstrap=10)


def test_primary_inference_rejects_missing_repeat_within_shared_cluster() -> None:
    rows = [
        _row(score=100.0, source="s1", domain="D", backend="B", model="a"),
        _row(score=0.0, source="s1", domain="D", backend="B", model="a"),
        _row(score=100.0, source="s1", domain="D", backend="B", model="b"),
        _row(score=0.0, source="s1", domain="D", backend="B", model="b"),
    ]
    for row, repeat_id in zip(rows, (0, 1, 0, 2), strict=True):
        row["physical_source_key"] = "shared-physical"
        row["repeat_id"] = repeat_id

    with pytest.raises(
        PrimaryLeaderboardContractError,
        match="identical effective-source and repeat membership",
    ):
        infer_primary_leaderboard(rows, n_bootstrap=10)


def test_primary_inference_rejects_physical_source_split_across_strata() -> None:
    rows = []
    for model in ("model-a", "model-b"):
        for backend in ("backend-a", "backend-b"):
            rows.append(
                _row(
                    score=50.0,
                    source=f"{backend}-source",
                    domain="D",
                    backend=backend,
                    model=model,
                    physical_source={"asset": "shared-grid"},
                )
            )

    with pytest.raises(
        PrimaryLeaderboardContractError,
        match="physical source identity may not span domain/backend strata",
    ):
        infer_primary_leaderboard(rows, n_bootstrap=10)


def test_nested_case_ledger_denominator_is_canonical() -> None:
    row = _row(
        score=80.0,
        source="remove-me",
        domain="D",
        backend="B",
    )
    row.pop("source_denominator_key")
    row["case_ledger"] = {"source_denominator_key": "nested"}

    result = aggregate_primary_leaderboard([row])

    assert result["leaderboard"][0]["effective_source_scores"] == {
        "D/B/nested": 80.0
    }


@pytest.mark.parametrize(
    "missing",
    [
        "model",
        "domain",
        "backend_kind",
        "source_denominator_key",
        "discriminative_core_score",
    ],
)
def test_formal_primary_aggregation_fails_closed_on_missing_fields(
    missing: str,
) -> None:
    row = _row(
        score=80.0,
        source="S",
        domain="D",
        backend="B",
    )
    row.pop(missing)

    with pytest.raises(PrimaryLeaderboardContractError):
        aggregate_primary_leaderboard([row])


@pytest.mark.parametrize(
    "fallback_field",
    ["scenario_id", "path", "family", "seed", "scenario_signature"],
)
def test_source_identity_never_falls_back_to_noncanonical_fields(
    fallback_field: str,
) -> None:
    row = _row(
        score=80.0,
        source="S",
        domain="D",
        backend="B",
    )
    row.pop("source_denominator_key")
    row[fallback_field] = "not-a-source-key"

    with pytest.raises(PrimaryLeaderboardContractError):
        aggregate_primary_leaderboard([row])


def test_summary_uses_the_canonical_primary_helper() -> None:
    from scripts.summarize_leaderboard_results import (
        _primary_leaderboard_payload as summary,
    )

    assert aggregate_primary_leaderboard(_imbalanced_rows()) == summary(
        _imbalanced_rows()
    )


def test_formal_batch_payload_rejects_missing_five_group_evidence() -> None:
    from scripts.batch_llm_eval import _primary_leaderboard_payload

    raw = {
        "status": "ok",
        "model": "model-a",
        "domain": "D",
        "backend_kind": "B",
        "source_denominator_key": "S",
        "difficulty_level": "basic",
        "task_completion": {
            "applicable": True,
            "completed": True,
            "contract": "native.task.v1",
            "evidence": {"objective_met": True},
        },
        "score": {
            "dimensions": [
                {
                    "name": "system_survival",
                    "applicable": True,
                    "calibrated_score": 100.0,
                    "evidence_ids": ["survival:e1"],
                }
            ]
        },
    }

    with pytest.raises(
        PrimaryLeaderboardContractError,
        match="outcome evidence",
    ):
        _primary_leaderboard_payload([raw])


def test_formal_batch_payload_rejects_stale_precomputed_score() -> None:
    from scripts.batch_llm_eval import _primary_leaderboard_payload

    row = _row(score=99.0, source="S", domain="D", backend="B")
    row["scoring_version"] = "0.10.0"
    row["evaluation_protocol"]["implementation_fingerprint"] = "stale-v5"

    with pytest.raises(
        PrimaryLeaderboardContractError,
        match="precomputed formal score",
    ):
        _primary_leaderboard_payload([row])


def test_formal_batch_payload_rejects_tampered_current_precomputed_score() -> None:
    from evaluation.scorer import DISCRIMINATIVE_CORE_DIMENSIONS
    from scripts.batch_llm_eval import _primary_leaderboard_payload

    row = _row(score=99.0, source="S", domain="D", backend="B")
    row.update(
        {
            "status": "ok",
            "difficulty_level": "basic",
            "task_completion": {
                "applicable": True,
                "completed": True,
                "contract": "native.task.v1",
                "evidence": {"objective_met": True},
            },
            "score": {
                "dimensions": [
                    {
                        "name": name,
                        "applicable": True,
                        "calibrated_score": 50.0,
                        "evidence_ids": [f"evidence:{name}"],
                    }
                    for name in DISCRIMINATIVE_CORE_DIMENSIONS
                    if name != "task_completion"
                ]
            },
        }
    )

    with pytest.raises(
        PrimaryLeaderboardContractError,
        match="precomputed formal score mismatch",
    ):
        _primary_leaderboard_payload([row])
