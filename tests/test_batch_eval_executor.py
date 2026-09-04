from __future__ import annotations

import sys
from pathlib import Path

import pytest

import batch_eval
from batch_eval import _execution_plan


def test_live_sumo_uses_process_isolation() -> None:
    assert _execution_plan({"sumo"}, 4) == ("process", 4)
    assert _execution_plan({"sumo_ego"}, 2) == ("process", 2)


def test_mock_sumo_can_remain_threaded() -> None:
    assert _execution_plan({"mock_sumo"}, 4) == ("thread", 4)


def test_grid2op_forces_single_worker_even_in_mixed_batch() -> None:
    assert _execution_plan({"grid2op", "sumo"}, 8) == ("process", 1)


@pytest.mark.parametrize("agent", ["llm_agent", "react_llm", "reflexion_llm"])
def test_legacy_batch_rejects_llm_agents_before_creating_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent: str,
) -> None:
    output_dir = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_eval.py",
            "--agents",
            agent,
            "--scenarios",
            "unused",
            "--output-dir",
            str(output_dir),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        batch_eval.main()

    assert not output_dir.exists()
