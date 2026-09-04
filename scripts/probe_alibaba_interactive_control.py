#!/usr/bin/env python3
"""Probe a deterministic mid-episode scheduling-policy control on Alibaba trace."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_SIMULATOR = (
    REPO_ROOT / "works" / "clusterdata" / "cluster-trace-gpu-v2020" / "simulator"
)
DEFAULT_RELEASE_DIR = REPO_ROOT / "release" / "dt_sched_bench_v0_52_0_candidate"
DEFAULT_TRACE = DEFAULT_RELEASE_DIR / "alibaba_trace_gpu_jobs_1000.csv"
DEFAULT_OUTPUT = DEFAULT_RELEASE_DIR / "alibaba_trace_interactive_control_probe.json"

from scripts.probe_alibaba_trace_runtime import (  # noqa: E402
    _metrics,
    _relative_headroom,
    _sha256,
    run_policy,
)


def build_report(
    *,
    fifo: tuple[int, float, float, int],
    switched_first: tuple[int, float, float, int],
    switched_second: tuple[int, float, float, int],
    config: dict[str, Any],
    trace_sha256: str,
    control_action_applied: bool,
) -> dict[str, Any]:
    jct_headroom = _relative_headroom(fifo[1], switched_first[1])
    wait_headroom = _relative_headroom(fifo[2], switched_first[2])
    checks = {
        "deterministic_switched_replay": switched_first == switched_second,
        "same_completed_work": fifo[0] == switched_first[0],
        "positive_average_jct_headroom": jct_headroom > 0,
        "positive_wait_time_headroom": wait_headroom > 0,
        "state_changing_policy_switch_applied": control_action_applied,
    }
    return {
        "schema_version": "0.1",
        "status": (
            "interactive_control_headroom_ready_adapter_pending"
            if all(checks.values())
            else "interactive_control_or_headroom_blocked"
        ),
        "source_id": "alibaba_cluster_trace_gpu_v2020",
        "trace_sha256": trace_sha256,
        "config": config,
        "control": {
            "name": "set_queue_policy",
            "from": "fifo",
            "to": "shortest_job_first",
            "state_changing": True,
        },
        "checks": checks,
        "policies": {
            "fifo": _metrics(fifo),
            "switched_first": _metrics(switched_first),
            "switched_second": _metrics(switched_second),
        },
        "headroom": {
            "average_jct_relative": jct_headroom,
            "wait_time_relative": wait_headroom,
        },
        "claim_boundary": (
            "This proves that a deterministic mid-episode policy mutation in the "
            "locked upstream simulator changes native scheduling outcomes. It does "
            "not yet prove ToolRegistry wiring, observation fog, task completion, "
            "counterfactual replay through a DT-Sched adapter, or Core eligibility."
        ),
    }


def run_switched_policy(
    trace: Path, *, config: dict[str, Any]
) -> tuple[tuple[int, float, float, int], bool]:
    if str(UPSTREAM_SIMULATOR) not in sys.path:
        sys.path.insert(0, str(UPSTREAM_SIMULATOR))
    from simulator import Simulator

    simulator = Simulator(
        csv_file=trace,
        alloc_policy=8,
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
        log_file=Path("/tmp/dt_sched_alibaba_interactive_probe.log"),
        export_job_stats=False,
        export_cluster_util=False,
        arrival_rate=int(config["arrival_rate_per_minute"]),
        num_jobs_limit=int(config["job_limit"]),
        gpu_type_matching=0,
        verbose=0,
    )
    simulator.init_go()
    switch_applied = False
    switch_time = int(config["switch_time_seconds"])
    while not simulator.exit_flag:
        if not switch_applied and int(simulator.cur_time) >= switch_time:
            simulator.scheduler.alloc_policy = 0
            switch_applied = True
        simulator.tic(simulator.delta)
    history = simulator.cluster.job_history
    completed = int(history.num_jobs_done)
    return (
        (
            completed,
            float(history.jct_summary) / completed,
            float(history.wait_time_summary) / completed,
            int(simulator.cur_time),
        ),
        switch_applied,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--job-limit", type=int, default=50)
    parser.add_argument("--arrival-rate", type=int, default=20)
    parser.add_argument("--gpu-capacity-units", type=int, default=5000)
    parser.add_argument("--cpu-capacity-units", type=int, default=156576)
    parser.add_argument("--switch-time-seconds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = {
        "job_limit": args.job_limit,
        "arrival_rate_per_minute": args.arrival_rate,
        "gpu_capacity_units": args.gpu_capacity_units,
        "cpu_capacity_units": args.cpu_capacity_units,
        "switch_time_seconds": args.switch_time_seconds,
        "seed": args.seed,
        "initial_allocation_policy": 8,
        "switched_allocation_policy": 0,
        "preemption_policy": 2,
        "node_sort_policy": 3,
    }
    fifo = run_policy(args.trace, allocation_policy=8, config=config)
    switched_first, first_applied = run_switched_policy(args.trace, config=config)
    switched_second, second_applied = run_switched_policy(args.trace, config=config)
    report = build_report(
        fifo=fifo,
        switched_first=switched_first,
        switched_second=switched_second,
        config=config,
        trace_sha256=_sha256(args.trace),
        control_action_applied=first_applied and second_applied,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"].startswith("interactive_control_headroom") else 1


if __name__ == "__main__":
    raise SystemExit(main())
