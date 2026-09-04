#!/usr/bin/env python3
"""Probe deterministic policy headroom in Alibaba's reference simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SIMULATOR = (
    REPO_ROOT
    / "works"
    / "clusterdata"
    / "cluster-trace-gpu-v2020"
    / "simulator"
)
DEFAULT_TRACE = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "alibaba_trace_gpu_jobs_1000.csv"
)
DEFAULT_OUTPUT = DEFAULT_TRACE.with_name("alibaba_trace_runtime_probe.json")


def _relative_headroom(baseline: float, candidate: float) -> float:
    if baseline <= 0:
        return 0.0
    return round((baseline - candidate) / baseline, 9)


def _metrics(result: tuple[int, float, float, int]) -> dict[str, Any]:
    completed, average_jct, wait_time, makespan = result
    return {
        "completed_work_units": completed,
        "average_jct_seconds": round(average_jct, 9),
        "average_wait_seconds": round(wait_time, 9),
        "makespan_seconds": makespan,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_report(
    *,
    fifo_first: tuple[int, float, float, int],
    fifo_second: tuple[int, float, float, int],
    sjf: tuple[int, float, float, int],
    config: dict[str, Any],
    trace_sha256: str,
) -> dict[str, Any]:
    jct_headroom = _relative_headroom(fifo_first[1], sjf[1])
    wait_headroom = _relative_headroom(fifo_first[2], sjf[2])
    checks = {
        "deterministic_fifo_replay": fifo_first == fifo_second,
        "same_completed_work": fifo_first[0] == sjf[0],
        "positive_average_jct_headroom": jct_headroom > 0,
        "positive_wait_time_headroom": wait_headroom > 0,
    }
    return {
        "schema_version": "0.1",
        "status": (
            "runtime_reference_headroom_ready_adapter_pending"
            if all(checks.values())
            else "runtime_or_headroom_blocked"
        ),
        "source_id": "alibaba_cluster_trace_gpu_v2020",
        "trace_sha256": trace_sha256,
        "config": config,
        "checks": checks,
        "policies": {
            "fifo_first": _metrics(fifo_first),
            "fifo_second": _metrics(fifo_second),
            "sjf": _metrics(sjf),
        },
        "headroom": {
            "average_jct_relative": jct_headroom,
            "wait_time_relative": wait_headroom,
        },
        "claim_boundary": (
            "This proves deterministic reference-policy headroom in the locked "
            "upstream simulator. It does not prove a DT-Sched native adapter, "
            "per-tick LLM control leverage, task completion, or Core eligibility."
        ),
    }


def run_policy(trace: Path, *, allocation_policy: int, config: dict[str, Any]) -> tuple[int, float, float, int]:
    if str(UPSTREAM_SIMULATOR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_SIMULATOR))
    from simulator import Simulator

    simulator = Simulator(
        csv_file=trace,
        alloc_policy=allocation_policy,
        preempt_policy=2,
        sort_node_policy=3,
        num_nodes=1,
        random_seed=int(config["seed"]),
        max_time=10**8,
        num_spare_node=0,
        pattern=0,
        hetero=False,
        num_gpus=int(config["gpu_capacity_units"]),
        num_cpus=int(config["cpu_capacity_units"]),
        describe_file=None,
        log_file=Path("/tmp/dt_sched_alibaba_trace_probe.log"),
        export_job_stats=False,
        export_cluster_util=False,
        arrival_rate=int(config["arrival_rate_per_minute"]),
        num_jobs_limit=int(config["job_limit"]),
        gpu_type_matching=0,
        verbose=0,
    )
    return simulator.simulator_go(repeat=1)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--job-limit", type=int, default=50)
    parser.add_argument("--arrival-rate", type=int, default=20)
    parser.add_argument("--gpu-capacity-units", type=int, default=5000)
    parser.add_argument("--cpu-capacity-units", type=int, default=156576)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = {
        "job_limit": args.job_limit,
        "arrival_rate_per_minute": args.arrival_rate,
        "gpu_capacity_units": args.gpu_capacity_units,
        "cpu_capacity_units": args.cpu_capacity_units,
        "seed": args.seed,
        "fifo_allocation_policy": 8,
        "sjf_allocation_policy": 0,
        "preemption_policy": 2,
        "node_sort_policy": 3,
    }
    fifo_first = run_policy(args.trace, allocation_policy=8, config=config)
    fifo_second = run_policy(args.trace, allocation_policy=8, config=config)
    sjf = run_policy(args.trace, allocation_policy=0, config=config)
    report = build_report(
        fifo_first=fifo_first,
        fifo_second=fifo_second,
        sjf=sjf,
        config=config,
        trace_sha256=_sha256(args.trace),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"].startswith("runtime_reference") else 1


if __name__ == "__main__":
    raise SystemExit(main())
