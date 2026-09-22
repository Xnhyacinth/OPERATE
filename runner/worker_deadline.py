"""Wall-clock supervision that also interrupts a blocked native backend.

Each job gets a fresh spawned process. The supervising pool worker stays alive
after a deadline, so an unresponsive backend cannot poison the rest of a batch.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import signal
import tempfile
import time
import traceback
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

DEFAULT_EPISODE_TIMEOUT_S = 64_800.0
DEFAULT_POSTPROCESSING_TIMEOUT_S = 28_800.0
_phase_connection: Connection | None = None


class WorkerProcessError(RuntimeError):
    """The isolated worker exited without producing a terminal result."""


class WorkerDeadlineError(WorkerProcessError):
    def __init__(self, phase: str, budget_s: float) -> None:
        self.phase = phase
        self.budget_s = budget_s
        super().__init__(f"{phase} wall-clock budget exceeded ({budget_s:g}s)")


def mark_postprocessing_started() -> None:
    """Start the postprocessing clock; a no-op outside an isolated job."""
    if _phase_connection is not None:
        _phase_connection.send(("postprocessing", time.monotonic()))


def _child_main(
    connection: Connection, function: Callable, job: dict[str, Any], result_path: str
) -> None:
    global _phase_connection
    os.setsid()
    _phase_connection = connection
    try:
        result = function(job)
        path = Path(result_path)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result), encoding="utf-8")
        temporary.replace(path)
        connection.send(("result", None))
    except BaseException as exc:
        # Provider exceptions can embed secrets. Keep frames, omit their message.
        frames = "".join(traceback.format_tb(exc.__traceback__, limit=12))[-8192:]
        connection.send(("error", f"{type(exc).__name__}\n{frames}"))
    finally:
        connection.close()


def _stop_process(process: multiprocessing.Process) -> None:
    """Kill the dedicated group, including native simulator subprocesses."""
    if process.pid is None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        if process.is_alive():
            process.kill()
    process.join(timeout=5)
    if process.is_alive():
        raise WorkerProcessError(f"worker {process.pid} did not exit after SIGKILL")


def run_with_deadline(
    function: Callable[[dict[str, Any]], dict[str, Any]],
    job: dict[str, Any],
    *,
    episode_timeout_s: float,
    postprocessing_timeout_s: float,
) -> dict[str, Any]:
    """Execute one job with episode and independently signaled scoring limits.

    Call from the main thread of a POSIX supervisor process. The episode
    budget includes startup and scoring. Neither limit drops scores
    or attribution rows: expiry fails the attempt and preserves on-disk evidence.
    """
    for value in (episode_timeout_s, postprocessing_timeout_s):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("worker deadlines must be finite and positive")
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    temporary = tempfile.TemporaryDirectory(prefix="operate-worker-")
    result_path = Path(temporary.name) / "result.json"
    process = context.Process(target=_child_main, args=(send, function, job, str(result_path)))
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    started = time.monotonic()
    postprocessing_started: float | None = None
    try:
        process.start()
        send.close()
        while True:
            now = time.monotonic()
            if now - started >= episode_timeout_s:
                raise WorkerDeadlineError("episode", episode_timeout_s)
            if (
                postprocessing_started is not None
                and now - postprocessing_started >= postprocessing_timeout_s
            ):
                raise WorkerDeadlineError("postprocessing", postprocessing_timeout_s)
            remaining = episode_timeout_s - (now - started)
            if postprocessing_started is not None:
                remaining = min(
                    remaining, postprocessing_timeout_s - (now - postprocessing_started)
                )
            if receive.poll(min(0.1, remaining)):
                try:
                    kind, payload = receive.recv()
                except EOFError as exc:
                    process.join(timeout=0.1)
                    raise WorkerProcessError(
                        f"worker exited without a result (exitcode={process.exitcode})"
                    ) from exc
                if kind == "postprocessing":
                    if postprocessing_started is None:
                        postprocessing_started = payload
                elif kind == "result":
                    return json.loads(result_path.read_text(encoding="utf-8"))
                elif kind == "error":
                    raise WorkerProcessError(payload)
            elif not process.is_alive():
                raise WorkerProcessError(
                    f"worker exited without a result (exitcode={process.exitcode})"
                )
    finally:
        try:
            _stop_process(process)
        finally:
            receive.close()
            send.close()
            signal.signal(signal.SIGTERM, previous_handler)
            temporary.cleanup()
