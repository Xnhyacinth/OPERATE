from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


SETUP = Path(__file__).resolve().parents[1] / "scripts/setup_eval_env.sh"


def _preflight():
    source = SETUP.read_text()
    marker = '"$PY" - <<\'SUMO_PREFLIGHT\'\n'
    assert marker in source, "setup must execute native SUMO before reporting installed"
    assert source.index(marker) < source.index('if [ "$SKIP_DATA"')
    return source.split(marker, 1)[1].split("\nSUMO_PREFLIGHT", 1)[0]


@pytest.fixture
def launch(monkeypatch):
    module = ModuleType("core.sidecar.sumo_sidecar")
    module._resolve_traci_launch = lambda: ("/native/sumo", {"SUMO_HOME": "/wheel"})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_setup_native_sumo_reports_resolved_binary_and_version(launch, monkeypatch, capsys):
    def run(command, **kwargs):
        assert command == ["/native/sumo", "--version"]
        assert kwargs == {
            "env": {"SUMO_HOME": "/wheel"}, "check": True,
            "capture_output": True, "text": True, "timeout": 30,
        }
        return subprocess.CompletedProcess(command, 0, "Eclipse SUMO sumo 1.27.1\ndetails\n", "")
    monkeypatch.setattr(subprocess, "run", run)
    exec(_preflight(), {})
    output = capsys.readouterr().out
    assert "/native/sumo" in output
    assert "Eclipse SUMO sumo 1.27.1" in output


@pytest.mark.parametrize("failure", [
    FileNotFoundError("native missing"),
    subprocess.CalledProcessError(1, ["sumo"], stderr="dynamic library missing"),
    subprocess.TimeoutExpired(["sumo"], 30, stderr="startup stalled"),
])
def test_setup_native_sumo_failure_is_fatal(launch, monkeypatch, capsys, failure):
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(SystemExit) as stopped:
        exec(_preflight(), {})
    assert stopped.value.code == 1
    output = capsys.readouterr().err
    assert "FATAL" in output
    assert str(getattr(failure, "stderr", None) or failure) in output


def test_setup_native_sumo_resolver_failure_is_fatal(launch, capsys):
    def missing():
        raise RuntimeError("no SUMO executable")
    launch._resolve_traci_launch = missing
    with pytest.raises(SystemExit) as stopped:
        exec(_preflight(), {})
    assert stopped.value.code == 1
    assert "no SUMO executable" in capsys.readouterr().err
