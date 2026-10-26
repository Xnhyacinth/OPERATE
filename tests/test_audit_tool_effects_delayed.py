import json
from pathlib import Path

from scripts.audit_tool_effects import build_report


def _row(path: Path, *, evidence_id: str) -> dict:
    return {
        "status": "ok",
        "scenario_id": "inventory/test",
        "scenario_slug": "inventory/test",
        "model": "model",
        "domain": "logistics",
        "backend_kind": "orgym_invmgmt",
        "family": "inventory_replenishment",
        "trajectory_summary": {"trajectory_path": str(path)[:-17]},
        "decision_impact": {"outcome_changed": True},
        "score": {
            "dimensions": [
                {
                    "name": "tool_use_efficiency",
                    "applicable": True,
                    "evidence_ids": [evidence_id],
                }
            ]
        },
    }


def test_delayed_result_may_link_evidence_from_original_step(tmp_path: Path) -> None:
    evidence_id = "ev-effect"
    path = tmp_path / "episode.trajectory.jsonl"
    entries = [
        {
            "tick": 0,
            "action": {"actions": []},
            "tool_results": [],
            "evidence_ids": [evidence_id],
        },
        {
            "tick": 1,
            "action": {"actions": []},
            "tool_results": [
                {
                    "name": "place_replenishment_order",
                    "ok": True,
                    "state_changing": True,
                    "idempotency_key": "call-1",
                    "evidence_id": evidence_id,
                    "latency_ticks": 1,
                    "payload": {"_status": "order_placed"},
                }
            ],
            "evidence_ids": ["ev-current"],
        },
    ]
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )

    totals = build_report([_row(path, evidence_id=evidence_id)])[
        "tool_effect_audit"
    ]["totals"]

    assert totals["delayed_state_changing_evidence_from_prior_step"] == 1
    assert totals["state_changing_evidence_not_in_step"] == 0
    assert totals["blocking_issues"] == 0


def test_non_delayed_result_still_blocks_on_orphaned_evidence(tmp_path: Path) -> None:
    path = tmp_path / "episode.trajectory.jsonl"
    path.write_text(
        json.dumps(
            {
                "tick": 0,
                "action": {"actions": []},
                "tool_results": [
                    {
                        "name": "place_replenishment_order",
                        "ok": True,
                        "state_changing": True,
                        "idempotency_key": "call-1",
                        "evidence_id": "ev-orphan",
                        "latency_ticks": 0,
                        "payload": {"_status": "order_placed"},
                    }
                ],
                "evidence_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    totals = build_report([_row(path, evidence_id="ev-orphan")])[
        "tool_effect_audit"
    ]["totals"]

    assert totals["state_changing_evidence_not_in_step"] == 1
    assert totals["blocking_issues"] == 1
