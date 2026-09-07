from __future__ import annotations

import hashlib
import json

import pytest

from evaluation.leaderboard import PrimaryLeaderboardContractError
from evaluation.scorer import HEADLINE_SCORE_GROUPS, SCORING_VERSION
from runner import EVALUATION_IMPLEMENTATION_FINGERPRINT, EVALUATION_PROTOCOL_VERSION
from scripts import batch_llm_eval as batch


def _row(tmp_path):
    contract = {"adaptive_replanning": False, "foresight_score": False}
    ledger = tmp_path / "episode.evidence.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "kind": "dimension_applicability_contract",
                "source": "engine",
                "payload": {"dimensions": contract},
            }
        )
        + "\n"
    )
    return {
        "status": "ok",
        "model": "test",
        "domain": "logistics",
        "backend_kind": "jsplib_job_shop",
        "source_denominator_key": "source",
        "case_ledger": {"physical_source_lock": {"asset": "fixture"}},
        "scoring_version": SCORING_VERSION,
        "evaluation_protocol": {
            "version": EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
        },
        "task_completion": {
            "applicable": True,
            "completed": True,
            "contract": "task.v1",
            "evidence": {"objective_met": True},
        },
        "score": {
            "dimension_applicability": contract,
            "dimensions": [
                {
                    "name": name,
                    "applicable": name not in contract,
                    "calibrated_score": 50.0,
                    "evidence_ids": [f"e:{name}"],
                }
                for group in HEADLINE_SCORE_GROUPS.values()
                for name in group["dimensions"]
            ],
        },
        "trajectory_summary": {
            "trajectory_path": str(tmp_path / "episode"),
            "evidence_ledger_artifact": {
                "path": str(ledger),
                "schema_version": "evidence_ledger_jsonl_v1",
                "event_count": 1,
                "byte_count": len(ledger.read_bytes()),
                "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            },
        },
    }


@pytest.mark.parametrize("precomputed", [False, True])
def test_primary_uses_only_evidence_bound_structural_na(tmp_path, precomputed):
    row = _row(tmp_path)
    expected = 57.5 / 0.85
    if precomputed:
        row.update(
            discriminative_core_score=expected,
            task_completion_raw=1.0,
            formal_score_eligible=True,
        )
    result = batch._primary_leaderboard_payload([row])
    assert result["leaderboard"][0]["primary_leaderboard_score"] == pytest.approx(
        expected
    )


def test_primary_rejects_tampered_applicability(tmp_path):
    row = _row(tmp_path)
    row["score"]["dimension_applicability"]["information_efficiency"] = False
    with pytest.raises(PrimaryLeaderboardContractError, match="applicability"):
        batch._primary_leaderboard_payload([row])


def test_primary_rejects_unbound_applicability(tmp_path):
    row = _row(tmp_path)
    del row["trajectory_summary"]
    with pytest.raises(PrimaryLeaderboardContractError, match="applicability"):
        batch._primary_leaderboard_payload([row])


def test_diagnostic_headline_uses_same_structural_na_formula(tmp_path):
    row = _row(tmp_path)
    assert batch._score_for_leaderboard_view(
        row, "discriminative_core"
    ) == pytest.approx(57.5 / 0.85)


def test_primary_verifies_portable_evidence_relative_to_batch_root(tmp_path):
    row = _row(tmp_path)
    row["trajectory_summary"]["trajectory_path"] = "episode"
    row["trajectory_summary"]["evidence_ledger_artifact"]["path"] = (
        "episode.evidence.jsonl"
    )
    result = batch._primary_leaderboard_payload([row], batch_root=tmp_path)
    assert result["leaderboard"][0]["primary_leaderboard_score"] == pytest.approx(
        57.5 / 0.85
    )


def test_primary_rejects_changed_ledger_bytes(tmp_path):
    row = _row(tmp_path)
    (tmp_path / "episode.evidence.jsonl").write_text("{}\n")
    with pytest.raises(PrimaryLeaderboardContractError, match="sha256"):
        batch._primary_leaderboard_payload([row])
