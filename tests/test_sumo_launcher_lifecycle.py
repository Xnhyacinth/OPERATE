"""Exercise real launcher/child lifetimes without a live simulator or provider."""

import importlib.metadata
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from core.sidecar.sumo_sidecar import SumoSidecar, SumoSidecarUnavailable, _resolve_traci_launch


def _alive(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True,
    )
    return result.returncode == 0 and not result.stdout.strip().startswith("Z")


@pytest.mark.parametrize("ignore_term", [False, True])
def test_failed_handshake_leaves_no_native_child_of_wheel_wrapper(tmp_path, monkeypatch, ignore_term):
    package = tmp_path / "site-packages"
    native = package / "sumo/bin/sumo"
    native.parent.mkdir(parents=True)
    pid_file = tmp_path / "native.pid"
    native.write_text(
        f"#!{sys.executable}\n"
        "import os, signal, time\n"
        f"if {ignore_term!r}: signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    native.chmod(0o755)
    wrapper = tmp_path / "bin/sumo"
    wrapper.parent.mkdir()
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import subprocess, sys\n"
        f"sys.exit(subprocess.call([{str(native)!r}] + sys.argv[1:]))\n"
    )
    wrapper.chmod(0o755)
    distribution = SimpleNamespace(
        files=[Path("../bin/sumo"), Path("sumo/bin/sumo")],
        entry_points=[SimpleNamespace(name="sumo", value="sumo:sumo", group="console_scripts")],
        locate_file=lambda path: package / path,
    )
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    monkeypatch.setenv("PATH", str(wrapper.parent) + os.pathsep + os.environ["PATH"])
    owned_pid = None

    def fail_connect(**kwargs):
        nonlocal owned_pid
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if pid_file.exists() and pid_file.read_text().isdigit():
                break
            time.sleep(0.01)
        assert pid_file.exists(), "native child did not start"
        owned_pid = int(pid_file.read_text())
        raise RuntimeError("injected handshake failure")

    monkeypatch.setitem(sys.modules, "traci", SimpleNamespace(connect=fail_connect))
    sidecar = SumoSidecar("missing.net.xml", "missing.rou.xml", traci_port=45678)
    sidecar._transport = "traci"
    try:
        with pytest.raises(RuntimeError, match="injected handshake failure"):
            sidecar.start()
        assert owned_pid is not None
        assert not _alive(owned_pid), "native child survived wrapper cleanup"
        assert sidecar._proc is None
    finally:
        sidecar.close()
        # Only the PID written by this test's private fixture may be cleaned up.
        if owned_pid is not None and _alive(owned_pid):
            os.kill(owned_pid, signal.SIGKILL)


def test_deliberate_system_sumo_keeps_path_precedence(tmp_path, monkeypatch):
    binary = tmp_path / "system/sumo"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary.parent))
    assert _resolve_traci_launch() == (str(binary), None)


@pytest.mark.parametrize("custom_environment", [False, True])
def test_wheel_fallback_preserves_launcher_environment(tmp_path, monkeypatch, custom_environment):
    package = tmp_path / "sumo"
    native = package / "bin/sumo"
    native.parent.mkdir(parents=True)
    native.write_text("#!/bin/sh\nexit 0\n")
    native.chmod(0o755)
    distribution = SimpleNamespace(locate_file=lambda path: tmp_path / path)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    for name in ("SUMO_HOME", "PROJ_LIB", "PROJ_DATA"):
        monkeypatch.delenv(name, raising=False)
    if custom_environment:
        monkeypatch.setenv("SUMO_HOME", "/chosen/sumo")
        monkeypatch.setenv("PROJ_DATA", "/chosen/proj")
    binary, environment = _resolve_traci_launch()
    assert binary == str(native)
    assert environment["SUMO_HOME"] == ("/chosen/sumo" if custom_environment else str(package))
    assert environment["PROJ_DATA"] == ("/chosen/proj" if custom_environment else str(package / "data/proj"))
    assert ("PROJ_LIB" not in environment) if custom_environment else environment["PROJ_LIB"] == str(package / "data/proj")
    assert os.environ.get("SUMO_HOME") == ("/chosen/sumo" if custom_environment else None)


def test_known_wheel_with_missing_native_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: SimpleNamespace(
        locate_file=lambda path: tmp_path / path,
    ))
    with pytest.raises(SumoSidecarUnavailable, match="native binary unavailable"):
        _resolve_traci_launch()
