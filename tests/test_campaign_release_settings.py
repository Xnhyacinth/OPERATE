from pathlib import Path
import json
import sys

import pytest

import run_lite
from scripts.run_eval_campaign import build_command


def job(**updates):
    return dict(
        id="model",
        suite="full",
        model="hy3-ioa",
        api_key_env="API_KEY",
        context_window=192000,
        max_output=64000,
        reasoning_effort="high",
        workers=16,
        chunk_jobs=16,
        **updates,
    )


def test_campaign_defaults_to_public_catalog():
    command = build_command(Path("/runtime"), Path("/out"), job())
    assert (
        command[command.index("--formal-manifest") + 1]
        == "/runtime/benchmark/manifest.json"
    )


def test_campaign_uses_explicit_new_manifest():
    command = build_command(
        Path("/runtime"),
        Path("/out"),
        job(formal_manifest="release/operate_v0_62_0/manifest.json"),
    )
    assert (
        command[command.index("--formal-manifest") + 1]
        == "/runtime/release/operate_v0_62_0/manifest.json"
    )


def test_realtime_campaign_uses_distinct_cli_and_bounded_chunk():
    command = build_command(
        Path("/runtime"),
        Path("/out"),
        job(
            setting="realtime_persistent",
            formal_manifest="release/new/manifest.json",
            suite_path="release/new/core_suite.json",
        ),
    )
    assert command[1].endswith("scripts/batch_realtime_llm_eval.py")
    assert "--model" in command and "--models" not in command
    assert "--max-jobs" in command and "--no-finalize" in command
    assert (
        command[command.index("--suite") + 1] == "/runtime/release/new/core_suite.json"
    )


def test_campaign_rejects_unknown_setting():
    with pytest.raises(ValueError, match="setting"):
        build_command(Path("/runtime"), Path("/out"), job(setting="typo"))


def test_lite_explicit_suite_is_consumed_before_scope_guard(tmp_path, monkeypatch):
    suite = tmp_path / "lite_suite.json"
    suite.write_text(
        json.dumps({"scenarios": [{"path": "scenarios/new/example.yaml"}]})
    )
    monkeypatch.setattr(
        sys, "argv", ["run_lite.py", "--lite-suite", str(suite), "--models", "example"]
    )
    captured = []
    monkeypatch.setattr(
        run_lite.batch_llm_eval, "main", lambda: captured.extend(sys.argv) or 0
    )
    assert run_lite.main() == 0
    assert captured[captured.index("--lite-lineage-suite") + 1] == str(suite)
    assert "--lite-suite" not in captured
