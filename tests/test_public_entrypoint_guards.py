from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import run_lite


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("option", [
    "--scenarios=other", "--scenario-slice=full", "--formal-r",
    "--formal-manifest=full.json", "--finalize-only", "--finalize-o",
    "--retry-cells=other.json",
])
def test_lite_rejects_scope_overrides(monkeypatch, option):
    monkeypatch.setattr(sys, "argv", ["run_lite.py", option])
    monkeypatch.setattr(run_lite.batch_llm_eval, "main", lambda: 0)
    with pytest.raises(SystemExit, match="fixes its scenario set"):
        run_lite.main()


def test_lite_allows_exact_finalize_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_lite.py", "--finalize"])
    monkeypatch.setattr(run_lite.batch_llm_eval, "main", lambda: 0)
    assert run_lite.main() == 0


def test_lite_does_not_append_user_positionals_to_fixed_scenarios(monkeypatch):
    import argparse

    def parse_scope():
        parser = argparse.ArgumentParser()
        parser.add_argument("--scenario-slice")
        parser.add_argument("--scenarios", nargs="*")
        parser.parse_args()
        return 0

    monkeypatch.setattr(sys, "argv", ["run_lite.py", "other-scenario"])
    monkeypatch.setattr(run_lite.batch_llm_eval, "main", parse_scope)
    with pytest.raises(SystemExit):
        run_lite.main()


def test_setup_backend_import_failure_is_fatal():
    source = (ROOT / "scripts/setup_eval_env.sh").read_text()
    section = source.split('log "verify locked backend imports"', 1)[1]
    section = section.split('if [ "$SKIP_DATA"', 1)[0]
    result = subprocess.run(
        ["bash"], input="set -euo pipefail\nPY=false\nok() { :; }\nwarn() { :; }\n" + section,
        text=True, capture_output=True,
    )
    assert result.returncode != 0


def test_setup_smoke_uses_existing_runner_and_propagates_failures():
    source = (ROOT / "scripts/setup_eval_env.sh").read_text()
    section = source.split('if [ "$SMOKE" -eq 1 ]; then', 1)[1]
    section = section.split('\nfi', 1)[0]
    assert "scripts/smoke_backends.py" not in section
    result = subprocess.run(
        ["bash"], input="set -euo pipefail\nPY=false\nlog() { :; }\nwarn() { :; }\n" + section,
        text=True, capture_output=True,
    )
    assert result.returncode != 0
