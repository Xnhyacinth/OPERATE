"""Extending sampling budgets must retain independent provider cooldowns."""

import json

import pytest

from scripts import run_eval_campaign as campaign
from tests.test_run_eval_campaign import job, result, state, summary


@pytest.mark.parametrize("pending", [1, 2])
def test_budget_extension_preserves_cooldown_and_other_cells(tmp_path, pending):
    lane = state()
    spec = job()
    terminal = result(
        status="error",
        retryable_infrastructure=True,
        execution_attempt_id="attempt-1",
        retry_at="1970-01-01T00:16:40+00:00",
    )
    ledger = tmp_path / "attempts.jsonl"
    campaign.apply_invocation(
        lane,
        spec,
        summary([terminal], pending=pending),
        now=100,
        cooldown=60,
        success_cooldown=0,
        max_attempts=1,
        ledger_path=ledger,
    )
    key = campaign.attempt_key(terminal)
    row = lane["jobs"][spec["id"]]
    assert row["held_cells"][key] == "episode_retry_budget_exhausted"

    campaign.extend_attempt_budget(ledger, lane, spec["id"], key, 1, "transport fixed")
    campaign.update_cell_holds(row, max_attempts=1)

    assert row["held_cells"][key] == "provider_cooldown"
    assert row["cell_not_before"][key] == 1000
    # A lane with another pending cell can dispatch it while the held-cell
    # allowlist continues excluding this cell. A one-cell lane must wait.
    selected = campaign.choose_job([spec], lane, now=101)
    assert selected == (spec if pending == 2 else None)
    assert row["pending"] == pending
    assert campaign.choose_job([spec], lane, now=1000) == spec
    assert row["held_cells"] == row["cell_not_before"] == {}

    campaign.restore_attempt_ledger(ledger, lane)
    assert row["attempt_failures"][key] == 1
    assert row["attempt_extensions"][key] == 1
    assert [json.loads(line)["event"] for line in ledger.read_text().splitlines()] == [
        "attempt_charged",
        "attempt_budget_extended",
    ]


def test_budget_extension_does_not_replace_a_repair_hold_with_cooldown(tmp_path):
    lane = state()
    spec = job()
    key = campaign.attempt_key(result())
    row = lane["jobs"][spec["id"]] = {
        "status": "needs_attention",
        "reason": "cells_held",
        "pending": 1,
        "attempt_failures": {key: 1},
        "held_cells": {key: "needs_repair"},
        "cell_not_before": {key: 1000},
    }
    campaign.extend_attempt_budget(
        tmp_path / "attempts.jsonl", lane, spec["id"], key, 1, "more budget"
    )
    campaign.update_cell_holds(row, max_attempts=1)
    assert row["held_cells"][key] == "needs_repair"
    assert campaign.choose_job([spec], lane, now=1001) is None
