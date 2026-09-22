from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from runner.worker_deadline import WorkerDeadlineError, run_with_deadline


def _native_hang(job):
    from runner.worker_deadline import mark_postprocessing_started

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    Path(job["pid_file"]).write_text(f"{os.getpid()} {child.pid}")
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if job.get("postprocessing"):
        mark_postprocessing_started()
    ctypes.CDLL(None).sleep(600)
    return {"status": "ok"}


def _successful(job):
    return {"status": "ok", "value": job["value"]}


def _crash(job):
    os._exit(9)


def _running(pid):
    path = Path(f"/proc/{pid}/stat")
    return path.exists() and path.read_text().split(")", 1)[1].strip().split()[0] != "Z"


@pytest.mark.parametrize("postprocessing", [False, True])
def test_native_hang_is_killed_including_descendants(tmp_path, postprocessing):
    started = time.monotonic()
    pid_file = tmp_path / "pids"
    with pytest.raises(WorkerDeadlineError) as caught:
        run_with_deadline(
            _native_hang,
            {"pid_file": str(pid_file), "postprocessing": postprocessing},
            episode_timeout_s=10 if postprocessing else 2,
            postprocessing_timeout_s=0.25,
        )
    assert caught.value.phase == ("postprocessing" if postprocessing else "episode")
    assert time.monotonic() - started < 8
    pids = [int(value) for value in pid_file.read_text().split()]
    deadline = time.monotonic() + 2
    while any(_running(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not any(_running(pid) for pid in pids)
    # A poisoned worker must not prevent the next queued job from completing.
    assert run_with_deadline(
        _successful,
        {"value": 42},
        episode_timeout_s=5,
        postprocessing_timeout_s=1,
    ) == {"status": "ok", "value": 42}


def test_child_process_crash_is_explicit():
    from runner.worker_deadline import WorkerProcessError

    with pytest.raises(WorkerProcessError, match="exited"):
        run_with_deadline(_crash, {}, episode_timeout_s=5, postprocessing_timeout_s=1)


@pytest.mark.parametrize("bad", [0, -1, float("inf"), float("nan")])
def test_deadlines_must_be_finite_positive(bad):
    with pytest.raises(ValueError, match="finite and positive"):
        run_with_deadline(
            _successful, {}, episode_timeout_s=bad, postprocessing_timeout_s=1
        )


def test_large_result_uses_nonblocking_notification():
    payload = "x" * 2_000_000
    assert run_with_deadline(
        _successful, {"value": payload}, episode_timeout_s=10, postprocessing_timeout_s=1,
    )["value"] == payload


def _deadline_result(job):
    try:
        return run_with_deadline(
            _native_hang if job.get("hang") else _successful, job,
            episode_timeout_s=2, postprocessing_timeout_s=1,
        )
    except WorkerDeadlineError:
        return {"status": "error", "error_type": "WorkerDeadlineError"}


def test_batch_pool_continues_after_native_worker_hang(tmp_path, monkeypatch):
    import json
    from scripts import batch_llm_eval as batch

    monkeypatch.setattr(batch, "_run_llm_episode_job_with_deadline", _deadline_result)
    jobs = [
        {"model": "fixture", "scenario_slug": "hang", "seed": 42,
         "hang": True, "pid_file": str(tmp_path / "pids")},
        {"model": "fixture", "scenario_slug": "next", "seed": 42, "value": 42},
    ]
    path = tmp_path / "episodes.jsonl"
    batch._run_global_jobs(jobs, path, "w", 1)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    terminal = [row for row in rows if row["status"] != "in_flight"]
    assert [row["status"] for row in terminal] == ["error", "ok"]
    assert terminal[-1]["value"] == 42
