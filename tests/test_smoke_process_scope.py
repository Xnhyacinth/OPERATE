from types import SimpleNamespace

import pytest

from scripts import run_protocol21_diagnostic_smoke as smoke


def test_sumo_inventory_excludes_other_process_groups(monkeypatch):
    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="111\n222\n333\n"
    ))
    monkeypatch.setattr(smoke.os, "getpgrp", lambda: 50)
    monkeypatch.setattr(smoke.os, "getpgid", lambda pid: {111: 50, 222: 60, 333: 50}[pid])
    assert smoke._sumo_process_ids() == ({111, 333}, True)


@pytest.mark.parametrize("error,available", [(ProcessLookupError(), True), (PermissionError(), False)])
def test_sumo_inventory_does_not_hide_permission_failures(monkeypatch, error, available):
    monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="111\n"
    ))

    def getpgid(pid):
        raise error

    monkeypatch.setattr(smoke.os, "getpgid", getpgid)
    assert smoke._sumo_process_ids() == (set(), available)


def test_episode_ignores_unrelated_sumo_but_keeps_owned_orphans(monkeypatch):
    monkeypatch.setattr(smoke, "implementation_identity", lambda root: {
        "implementation_tree_sha256": "tree"
    })
    monkeypatch.setattr(smoke, "_load_bound_scenario", lambda row: {
        "scenario_id": "canonical", "seed_id": "runtime"
    })
    monkeypatch.setattr(smoke, "run_one", lambda *a, **k: {
        "scenario_id": "runtime", "scenario_signature": "sig"
    })
    monkeypatch.setattr(smoke.os, "getpgrp", lambda: 50)
    monkeypatch.setattr(smoke.os, "getpgid", lambda pid: 60 if pid == 222 else 50)
    row = {"path": "test.yaml", "scenario_id": "canonical", "scenario_signature": "sig"}
    for after, expected_orphans in [("111\n222\n", []), ("111\n222\n333\n", [333])]:
        inventories = iter(["111\n", after])
        monkeypatch.setattr(smoke.subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=next(inventories)
        ))
        result = smoke._run_episode(row, "wait_only", 42, timeout_seconds=5)
        assert result["diagnostic_runtime_integrity"]["orphan_pids"] == expected_orphans
        assert result["status"] == ("error" if expected_orphans else "ok")


@pytest.mark.parametrize("eof", [False, True])
def test_isolated_failure_checks_worker_process_group(monkeypatch, eof):
    def recv():
        raise EOFError

    receive = SimpleNamespace(poll=lambda timeout: eof, recv=recv, close=lambda: None)
    send = SimpleNamespace(close=lambda: None)
    process = SimpleNamespace(pid=900, exitcode=1, start=lambda: None)
    context = SimpleNamespace(Pipe=lambda **k: (receive, send), Process=lambda **k: process)
    monkeypatch.setattr(smoke.mp, "get_context", lambda mode: context)
    monkeypatch.setattr(smoke, "_terminate_isolated_process", lambda p: None)
    monkeypatch.setattr(smoke, "implementation_identity", lambda root: {
        "implementation_tree_sha256": "tree"
    })
    observed = []

    def inventory(*, pgid=None):
        observed.append(pgid)
        return {901}, True

    monkeypatch.setattr(smoke, "_sumo_process_ids", inventory)
    result = smoke._run_episode_isolated({}, "wait_only", 42, timeout_seconds=1,
                                         expected_implementation_tree_sha256="tree")
    assert observed == [900]
    assert result["diagnostic_runtime_integrity"]["orphan_pids"] == [901]
