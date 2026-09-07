from pathlib import Path

import pytest

from runner import batch


@pytest.mark.parametrize("with_log", [False, True])
def test_scenario_load_failure_is_recorded_without_masking_original_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_log: bool,
) -> None:
    def fail(_slug: str) -> dict:
        raise FileNotFoundError("missing scenario fixture")

    monkeypatch.setattr("run.load_scenario_yaml", fail)
    log_path = tmp_path / "logs" / "episode.log"
    options = {"episode_log_path": log_path} if with_log else {}
    row = batch.run_one_safe(("missing", "llm_agent", 42, {}, options))
    assert row["status"] == "error"
    assert row["error_type"] == "FileNotFoundError"
    assert "missing scenario fixture" in row["error"]
    if with_log:
        assert "missing scenario fixture" in log_path.read_text()
