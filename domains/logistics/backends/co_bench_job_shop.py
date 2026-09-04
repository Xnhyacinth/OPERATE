"""Prototype CO-Bench Job-Shop backend adapter (STAGING ONLY).

This module is a staging prototype — it is NOT wired into the release
pipeline, the adapter's ``_build_backend``, or any registry. It exists to
prove that CO-Bench's text-format job-shop instances can be loaded and
driven through the same ``schedule_next_operation`` / ``tick`` /
``snapshot`` interface used by ``jsplib_job_shop``.

Status: PROTOTYPE / STAGING
- Instance format: CO-Bench "Nb of jobs / Times / Machines" text format
- Machine indexing: CO-Bench uses 1-indexed machines; we normalize to 0-indexed
- Reference: CO-Bench provides upper_bound / lower_bound per test case
- Oracle: dispatching-rule baseline (SPT — Shortest Processing Time first)

DO NOT import this from production code paths.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import EvidenceLogger, ToolContext, ToolRegistry, ToolSpec
from domains.logistics.seeds.schema import LogisticsScenarioSeed

_UNSCHEDULED_OPERATION_PENALTY = 100.0
_MAX_DISPATCH_BATCH_SIZE = 20


def _normalize_job_id(raw: Any, valid_ids: Iterable[str]) -> str:
    """Map a loosely-formatted job id to the canonical ``j{n}`` id.

    Agents routinely emit ``J29`` / ``job_29`` / ``29`` for the canonical
    ``j29`` surfaced in the observation. These aliases identify the same
    job, so accept them instead of rejecting with ``unknown_job``. The real
    decision under test is *which* job / operation to schedule, not the
    string format; genuinely unknown jobs still fail.
    """
    s = str(raw).strip()
    valid = set(valid_ids)
    if s in valid:
        return s
    match = re.search(r"(\d+)", s)
    if match is not None:
        candidate = f"j{int(match.group(1))}"
        if candidate in valid:
            return candidate
    return s


@dataclass(frozen=True)
class CoBenchJobShopOperation:
    """One ordered operation in a CO-Bench job-shop routing plan."""

    machine_id: int  # 0-indexed
    duration: int


@dataclass(frozen=True)
class ScheduledOperation:
    """Scheduled operation with deterministic machine/job timing."""

    job_id: str
    operation_index: int
    machine_id: int
    start_time: int
    end_time: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "operation_index": self.operation_index,
            "machine_id": self.machine_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class CoBenchJobShopBackend:
    """Deterministic pure-Python Job-Shop simulator for CO-Bench samples.

    Interface-compatible with ``JsplibJobShopBackend`` so the same tool
    surface and scoring records work with minimal adaptation.

    Key differences from JSPLIB:
    - Instances come from CO-Bench text format (Nb of jobs / Times / Machines)
    - Machines are 1-indexed in the source; normalized to 0-indexed internally
    - Each .txt file contains MULTIPLE test cases (10 by default for Taillard)
    - Reference data is per-case upper_bound / lower_bound (not known_optimum)
    """

    backend_kind = "co_bench_job_shop"

    def __init__(
        self,
        *,
        instance_id: str,
        case_index: int,
        jobs: dict[str, list[CoBenchJobShopOperation]],
        n_machines: int,
        upper_bound: int | None = None,
        lower_bound: int | None = None,
        time_seed: int | None = None,
        machine_seed: int | None = None,
    ) -> None:
        if not jobs:
            raise ValueError("job-shop instance must contain at least one job")
        if n_machines <= 0:
            raise ValueError("job-shop instance must contain at least one machine")
        self.instance_id = str(instance_id)
        self.case_index = int(case_index)
        self._jobs = {job_id: list(ops) for job_id, ops in jobs.items()}
        self._n_machines = int(n_machines)
        self._upper_bound = upper_bound
        self._lower_bound = lower_bound
        self._time_seed = time_seed
        self._machine_seed = machine_seed
        self._next_operation = {job_id: 0 for job_id in self._jobs}
        self._job_available_at = {job_id: 0 for job_id in self._jobs}
        self._machine_available_at = {
            machine_id: 0 for machine_id in range(self._n_machines)
        }
        self._scheduled: list[ScheduledOperation] = []
        self._tick_records: list[dict[str, Any]] = []
        self._current_tick = 0
        self._reference_optimum: float | None = None

    # ── Class methods: instance loading ──────────────────────────────────

    @classmethod
    def from_co_bench_file(
        cls,
        file_path: str | Path,
        case_index: int = 0,
    ) -> CoBenchJobShopBackend:
        """Load a single test case from a CO-Bench job-shop .txt file.

        CO-Bench files contain multiple test cases. ``case_index`` selects
        which one to load (0-based).
        """
        cases = parse_co_bench_job_shop_file(file_path)
        if case_index < 0 or case_index >= len(cases):
            raise ValueError(
                f"case_index {case_index} out of range [0, {len(cases) - 1}]"
            )
        case = cases[case_index]
        return cls._from_parsed_case(
            file_path=Path(file_path),
            case_index=case_index,
            case=case,
        )

    @classmethod
    def from_parsed_case(
        cls,
        instance_id: str,
        case_index: int,
        case: dict[str, Any],
    ) -> CoBenchJobShopBackend:
        """Build a backend from a parsed CO-Bench case dict."""
        return cls._from_parsed_case(
            file_path=None,
            case_index=case_index,
            case=case,
            instance_id_override=instance_id,
        )

    @classmethod
    def from_seed(cls, seed: LogisticsScenarioSeed) -> CoBenchJobShopBackend:
        """Build a backend from a staging CO-Bench Job-Shop seed skeleton."""
        cfg = seed.backend_config.get("co_bench_job_shop") or {}
        if seed.backend_kind != cls.backend_kind:
            raise ValueError(
                f"expected co_bench_job_shop seed, got {seed.backend_kind}"
            )
        rows = cfg.get("jobs_detail")
        if not isinstance(rows, list):
            raise ValueError(
                "co-bench job-shop seed is missing "
                "backend_config.co_bench_job_shop.jobs_detail"
            )
        jobs: dict[str, list[CoBenchJobShopOperation]] = {}
        for job_idx, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValueError(
                    f"co-bench job-shop seed job row {job_idx} is not a list"
                )
            jobs[f"j{job_idx}"] = [
                CoBenchJobShopOperation(
                    machine_id=int(op["machine"]),
                    duration=int(op["duration"]),
                )
                for op in row
            ]
        backend = cls(
            instance_id=str(
                seed.backend_config.get("instance_name") or seed.seed_id
            ),
            case_index=int(cfg.get("case_index", 0)),
            jobs=jobs,
            n_machines=int(cfg.get("machines", 0)),
            upper_bound=_opt_int(cfg.get("upper_bound")),
            lower_bound=_opt_int(cfg.get("lower_bound")),
            time_seed=_opt_int(cfg.get("time_seed")),
            machine_seed=_opt_int(cfg.get("machine_seed")),
        )
        backend.reset(seed)
        return backend

    @classmethod
    def _from_parsed_case(
        cls,
        file_path: Path | None,
        case_index: int,
        case: dict[str, Any],
        instance_id_override: str | None = None,
    ) -> CoBenchJobShopBackend:
        n_jobs = int(case["n_jobs"])
        n_machines = int(case["n_machines"])
        times = case["times"]
        machines = case["machines"]

        jobs: dict[str, list[CoBenchJobShopOperation]] = {}
        for job_idx in range(n_jobs):
            ops: list[CoBenchJobShopOperation] = []
            for op_idx in range(n_machines):
                ops.append(
                    CoBenchJobShopOperation(
                        # CO-Bench machines are 1-indexed; normalize to 0-indexed
                        machine_id=int(machines[job_idx][op_idx]) - 1,
                        duration=int(times[job_idx][op_idx]),
                    )
                )
            jobs[f"j{job_idx}"] = ops

        instance_id = (
            instance_id_override
            or f"{file_path.stem if file_path else 'unknown'}_case{case_index}"
        )
        return cls(
            instance_id=instance_id,
            case_index=case_index,
            jobs=jobs,
            n_machines=n_machines,
            upper_bound=_opt_int(case.get("upper_bound")),
            lower_bound=_opt_int(case.get("lower_bound")),
            time_seed=_opt_int(case.get("time_seed")),
            machine_seed=_opt_int(case.get("machine_seed")),
        )

    # ── Reset / state ────────────────────────────────────────────────────

    def reset(self, scenario_seed: LogisticsScenarioSeed) -> None:
        """Reset dynamic state from the parsed seed-backed job shop."""
        cfg = scenario_seed.backend_config.get("co_bench_job_shop") or {}
        rows = cfg.get("jobs_detail")
        if not isinstance(rows, list):
            raise ValueError(
                "co-bench job-shop seed is missing "
                "backend_config.co_bench_job_shop.jobs_detail"
            )

        self.instance_id = str(
            scenario_seed.backend_config.get("instance_name")
            or scenario_seed.seed_id
        )
        self.case_index = int(cfg.get("case_index", 0))
        self._jobs = {}
        for job_idx, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValueError(
                    f"co-bench job-shop seed job row {job_idx} is not a list"
                )
            self._jobs[f"j{job_idx}"] = [
                CoBenchJobShopOperation(
                    machine_id=int(op["machine"]),
                    duration=int(op["duration"]),
                )
                for op in row
            ]
        self._n_machines = int(cfg.get("machines", 0))
        self._upper_bound = _opt_int(cfg.get("upper_bound"))
        self._lower_bound = _opt_int(cfg.get("lower_bound"))
        self._time_seed = _opt_int(cfg.get("time_seed"))
        self._machine_seed = _opt_int(cfg.get("machine_seed"))
        if not self._jobs:
            raise ValueError("job-shop instance must contain at least one job")
        if self._n_machines <= 0:
            raise ValueError("job-shop instance must contain at least one machine")
        self._next_operation = {job_id: 0 for job_id in self._jobs}
        self._job_available_at = {job_id: 0 for job_id in self._jobs}
        self._machine_available_at = {
            machine_id: 0 for machine_id in range(self._n_machines)
        }
        self._scheduled = []
        self._tick_records = []
        self._current_tick = 0
        self._reference_optimum = _reference_optimum_from_seed(scenario_seed)

    # ── Scheduling operations ────────────────────────────────────────────

    def schedule_next_operation(
        self, *, job_id: str, operation_index: int
    ) -> dict[str, Any]:
        """Schedule the next operation for a job at earliest feasible time."""
        job_id = _normalize_job_id(job_id, self._jobs)
        if job_id not in self._jobs:
            return {"_status": "error", "error": "unknown_job", "job_id": job_id}
        expected_index = self._next_operation[job_id]
        if int(operation_index) != expected_index:
            return {
                "_status": "error",
                "error": "operation_not_ready",
                "job_id": job_id,
                "operation_index": int(operation_index),
                "expected_operation_index": expected_index,
            }
        operations = self._jobs[job_id]
        if expected_index >= len(operations):
            return {
                "_status": "error",
                "error": "job_complete",
                "job_id": job_id,
                "operation_index": int(operation_index),
            }

        op = operations[expected_index]
        start = max(
            self._job_available_at[job_id],
            self._machine_available_at[op.machine_id],
        )
        end = start + op.duration
        scheduled = ScheduledOperation(
            job_id=job_id,
            operation_index=expected_index,
            machine_id=op.machine_id,
            start_time=start,
            end_time=end,
        )
        self._scheduled.append(scheduled)
        self._next_operation[job_id] = expected_index + 1
        self._job_available_at[job_id] = end
        self._machine_available_at[op.machine_id] = end

        return {
            "_status": "scheduled",
            **scheduled.to_dict(),
            "makespan": self.makespan,
        }

    @property
    def makespan(self) -> int:
        if not self._scheduled:
            return 0
        return max(op.end_time for op in self._scheduled)

    def ready_operations(self) -> dict[str, dict[str, int]]:
        ready: dict[str, dict[str, int]] = {}
        for job_id, idx in self._next_operation.items():
            operations = self._jobs[job_id]
            if idx >= len(operations):
                continue
            op = operations[idx]
            ready[job_id] = {
                "operation_index": idx,
                "machine_id": op.machine_id,
                "duration": op.duration,
            }
        return ready

    # ── Snapshot / tick ──────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        max_operations_per_job = max(
            (len(operations) for operations in self._jobs.values()),
            default=1,
        )
        ready_operations = self.ready_operations()
        return {
            "domain": "logistics",
            "backend_kind": self.backend_kind,
            "instance_id": self.instance_id,
            "case_index": self.case_index,
            "jobs": len(self._jobs),
            "machines": self._n_machines,
            "operations_total": sum(len(ops) for ops in self._jobs.values()),
            "operations_scheduled": len(self._scheduled),
            "decision_opportunity": bool(ready_operations),
            "decision_cadence": {
                "max_operations_per_dispatch": _MAX_DISPATCH_BATCH_SIZE,
                "minimum_dispatch_waves": max_operations_per_job,
                "hold_while_actions_pending": True,
                "model_decision_budget": 2 * max_operations_per_job + 8,
            },
            "makespan": self.makespan,
            "upper_bound": self._upper_bound,
            "lower_bound": self._lower_bound,
            "machine_available_at": dict(self._machine_available_at),
            "job_available_at": dict(self._job_available_at),
            "ready_operations": ready_operations,
            "scheduled_operations": [op.to_dict() for op in self._scheduled[-12:]],
            "scheduled_operations_available": len(self._scheduled),
        }

    def tick(self, current_tick: int) -> dict[str, Any]:
        """Advance one supervisory tick and emit a scorer-facing record."""
        self._current_tick = int(current_tick)
        record = self._canonical_row_for_tick(self._current_tick)
        self._tick_records.append(record)
        operations_total = sum(len(ops) for ops in self._jobs.values())
        completed = len(self._scheduled) == operations_total
        return {
            "tick": self._current_tick,
            "routing_cost": float(record["production_cost"]),
            "dispatch_fixed_cost": 0.0,
            "drop_penalty": float(record["shed_penalty"]),
            "unmet_demand": float(record["balance_error_mw"]),
            "done": completed,
            "realized_events": [],
        }

    # ── Scoring records ──────────────────────────────────────────────────

    def scoring_records(self) -> list[dict[str, Any]]:
        if self._tick_records:
            return list(self._tick_records)
        if self._scheduled:
            return self.canonical_scoring_records()
        return [self._canonical_row_for_tick(0)]

    def canonical_scoring_records(self) -> list[dict[str, Any]]:
        """Map job-shop progress into the benchmark's canonical 14-key shape."""
        operations_total = sum(len(ops) for ops in self._jobs.values())
        rows: list[dict[str, Any]] = []
        for idx, _op in enumerate(self._scheduled, start=1):
            makespan = max(prev.end_time for prev in self._scheduled[:idx])
            unscheduled = max(0, operations_total - idx)
            rows.append(
                {
                    "tick": idx - 1,
                    "aggregate_demand_mw": operations_total,
                    "aggregate_generation_mw": idx,
                    "balance_error_mw": unscheduled,
                    "reserves_required_mw": 0,
                    "reserves_procured_mw": 0,
                    "production_cost": makespan,
                    "startup_cost": 0,
                    "shed_penalty": unscheduled,
                    "rho_max": round(
                        max(0.0, idx / max(1, operations_total)),
                        6,
                    ),
                    "n_overloads": 0,
                    "n_voltage_violations": 0,
                    "n_disconnected_lines": 0,
                    # Successful completion must not be scored as a
                    # catastrophic terminal failure.
                    "done": False,
                    "catastrophic_failure": False,
                    "safety_violation_severity": 0.0,
                }
            )
        return rows

    def native_scheduling_records(self) -> list[dict[str, Any]]:
        """Return detailed per-operation schedule rows for candidate reports."""
        return [
            {
                "step": idx,
                "job_id": op.job_id,
                "operation_index": op.operation_index,
                "machine_id": op.machine_id,
                "start_time": op.start_time,
                "end_time": op.end_time,
                "makespan": max(
                    prev.end_time for prev in self._scheduled[:idx]
                ),
            }
            for idx, op in enumerate(self._scheduled, start=1)
        ]

    def ground_truth_costs(self) -> dict[str, float]:
        return {"production_cost": round(self._objective_cost(), 3)}

    def per_customer_unmet_units(self) -> dict[str, float]:
        return {}

    # ── Oracle: dispatching rules ────────────────────────────────────────

    def dispatching_rule_oracle(
        self, rule: str = "spt"
    ) -> dict[str, Any]:
        """Run a dispatching-rule oracle and return the full schedule.

        Supported rules:
        - ``spt``: Shortest Processing Time first (default)
        - ``lpt``: Longest Processing Time first
        - ``fifo``: First-In-First-Out (job index order)
        - ``mwkr``: Most Work Remaining

        Returns a dict with ``makespan``, ``schedule`` (list of
        ScheduledOperation dicts), and ``rule`` name.
        """
        # Save current state
        saved_next = dict(self._next_operation)
        saved_job_avail = dict(self._job_available_at)
        saved_machine_avail = dict(self._machine_available_at)
        saved_scheduled = list(self._scheduled)

        try:
            # Reset for oracle run
            self._next_operation = {job_id: 0 for job_id in self._jobs}
            self._job_available_at = {job_id: 0 for job_id in self._jobs}
            self._machine_available_at = {
                m: 0 for m in range(self._n_machines)
            }
            self._scheduled = []

            n_ops = sum(len(ops) for ops in self._jobs.values())
            for _ in range(n_ops):
                ready = self.ready_operations()
                if not ready:
                    break
                job_id = _select_job_by_rule(ready, self._jobs, rule)
                op_idx = ready[job_id]["operation_index"]
                self.schedule_next_operation(
                    job_id=job_id, operation_index=op_idx
                )

            result = {
                "rule": rule,
                "makespan": self.makespan,
                "schedule": [op.to_dict() for op in self._scheduled],
                "n_operations": len(self._scheduled),
            }
            return result
        finally:
            # Restore state
            self._next_operation = saved_next
            self._job_available_at = saved_job_avail
            self._machine_available_at = saved_machine_avail
            self._scheduled = saved_scheduled

    # ── Internals ────────────────────────────────────────────────────────

    def _canonical_row_for_tick(self, tick: int) -> dict[str, Any]:
        operations_total = sum(len(ops) for ops in self._jobs.values())
        scheduled = len(self._scheduled)
        unscheduled = max(0, operations_total - scheduled)
        return {
            "tick": int(tick),
            "aggregate_demand_mw": operations_total,
            "aggregate_generation_mw": scheduled,
            "balance_error_mw": unscheduled,
            "reserves_required_mw": 0,
            "reserves_procured_mw": 0,
            "production_cost": round(self._objective_cost(), 3),
            "startup_cost": 0,
            "shed_penalty": round(
                unscheduled * _UNSCHEDULED_OPERATION_PENALTY, 3
            ),
            "rho_max": round(scheduled / max(1, operations_total), 6),
            "n_overloads": 0,
            "n_voltage_violations": 0,
            "n_disconnected_lines": 0,
            "done": False,
            "catastrophic_failure": False,
            "safety_violation_severity": 0.0,
        }

    def _objective_cost(self) -> float:
        operations_total = sum(len(ops) for ops in self._jobs.values())
        unscheduled = max(0, operations_total - len(self._scheduled))
        return float(self.makespan + unscheduled * _UNSCHEDULED_OPERATION_PENALTY)


# ─────────────────────────────────────────────────────────────────────────────
# CO-Bench file parser
# ─────────────────────────────────────────────────────────────────────────────


def parse_co_bench_job_shop_file(file_path: str | Path) -> list[dict[str, Any]]:
    """Parse a CO-Bench job-shop scheduling .txt file.

    File format (repeating blocks):

        Nb of jobs, Nb of Machines, Time seed, Machine seed, Upper bound, Lower bound
          n_jobs  n_machines  time_seed  machine_seed  upper_bound  lower_bound
        Times
         t11 t12 ... t1m
         ...
         tn1 tn2 ... tnm
        Machines
         m11 m12 ... m1m
         ...
         mn1 mn2 ... mnm

    Returns a list of case dicts, each with keys:
    ``n_jobs``, ``n_machines``, ``time_seed``, ``machine_seed``,
    ``upper_bound``, ``lower_bound``, ``times``, ``machines``.

    Machine IDs in the returned ``machines`` list are 1-indexed (as in source).
    """
    path = Path(file_path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    lines = [line for line in lines if line]  # remove blanks

    cases: list[dict[str, Any]] = []
    i = 0
    header_re = re.compile(r"[Nn]b of jobs")

    while i < len(lines):
        if header_re.search(lines[i]):
            # Next line: n_jobs n_machines time_seed machine_seed upper_bound lower_bound
            i += 1
            if i >= len(lines):
                raise ValueError(f"unexpected EOF after header at line {i}")
            tokens = lines[i].split()
            if len(tokens) < 6:
                raise ValueError(
                    f"expected 6 values on header data line, got {len(tokens)}"
                )
            n_jobs = int(tokens[0])
            n_machines = int(tokens[1])
            time_seed = int(tokens[2])
            machine_seed = int(tokens[3])
            upper_bound = int(tokens[4])
            lower_bound = int(tokens[5])

            # Times section
            i += 1
            if i >= len(lines) or not lines[i].lower().startswith("times"):
                raise ValueError(
                    f"expected 'Times' section at line {i}, got {lines[i][:40]}"
                )
            i += 1
            times: list[list[int]] = []
            for _ in range(n_jobs):
                if i >= len(lines):
                    raise ValueError(f"unexpected EOF in Times section at line {i}")
                row = list(map(int, lines[i].split()))
                if len(row) != n_machines:
                    raise ValueError(
                        f"Times row {len(times)} has {len(row)} values, "
                        f"expected {n_machines}"
                    )
                times.append(row)
                i += 1

            # Machines section
            if i >= len(lines) or not lines[i].lower().startswith("machines"):
                raise ValueError(
                    f"expected 'Machines' section at line {i}, got {lines[i][:40]}"
                )
            i += 1
            machines: list[list[int]] = []
            for _ in range(n_jobs):
                if i >= len(lines):
                    raise ValueError(
                        f"unexpected EOF in Machines section at line {i}"
                    )
                row = list(map(int, lines[i].split()))
                if len(row) != n_machines:
                    raise ValueError(
                        f"Machines row {len(machines)} has {len(row)} values, "
                        f"expected {n_machines}"
                    )
                machines.append(row)
                i += 1

            cases.append(
                {
                    "n_jobs": n_jobs,
                    "n_machines": n_machines,
                    "time_seed": time_seed,
                    "machine_seed": machine_seed,
                    "upper_bound": upper_bound,
                    "lower_bound": lower_bound,
                    "times": times,
                    "machines": machines,
                }
            )
        else:
            i += 1

    if not cases:
        raise ValueError(f"no job-shop test cases found in {file_path}")
    return cases


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration (mirrors jsplib_job_shop pattern)
# ─────────────────────────────────────────────────────────────────────────────


def register_co_bench_job_shop_tools(
    reg: ToolRegistry, backend: CoBenchJobShopBackend
) -> None:
    """Register the CO-Bench Job-Shop tool surface through core protocol."""

    reg.register(
        ToolSpec(
            name="dispatch_ready_operations",
            description=(
                "Atomically submit an ordered batch of 1-20 mutually "
                "independent operations copied from the current "
                "observation's ready_operations. Each job may appear at "
                "most once in a batch. Operations are scheduled in the "
                "provided order at their earliest precedence- and "
                "machine-feasible times; invalid entries are reported "
                "without rolling back valid entries."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "job_id": {"type": "string"},
                                "operation_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                },
                            },
                            "required": ["job_id", "operation_index"],
                        },
                    }
                },
                "required": ["operations"],
            },
            handler=_h_dispatch_ready_operations(backend),
            state_changing=True,
            semantic_role="control",
            native_target_kind="ready_operation_batch",
            actuator_family="job_shop_dispatch",
            cost_units=1.0,
        )
    )
    reg.register(
        ToolSpec(
            name="dispatch_job_operation",
            description=(
                "Schedule the next unscheduled operation of a CO-Bench "
                "job at the earliest feasible time respecting job "
                "precedence and single-machine capacity. Copy job_id and "
                "operation_index from the current observation's "
                "ready_operations; never repeat a pair after it schedules "
                "successfully."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "operation_index": {"type": "integer", "minimum": 0},
                },
                "required": ["job_id", "operation_index"],
            },
            handler=_h_dispatch_job_operation(backend),
            state_changing=True,
            semantic_role="control",
            native_target_kind="job_operation",
            actuator_family="job_shop_dispatch",
            cost_units=0.25,
        )
    )
    reg.register(
        ToolSpec(
            name="wait",
            description=(
                "Advance without scheduling a CO-Bench job-shop operation."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_h_wait,
            state_changing=False,
            semantic_role="meta",
            native_target_kind="simulation_clock",
            cost_units=0.0,
        )
    )


def _h_dispatch_ready_operations(backend: CoBenchJobShopBackend):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        requested = args.get("operations")
        if not isinstance(requested, list) or not 1 <= len(requested) <= 20:
            result: dict[str, Any] = {
                "_status": "error",
                "error": "operations_must_contain_1_to_20_items",
                "requested_count": len(requested) if isinstance(requested, list) else 0,
            }
        else:
            initially_ready = {
                (job_id, int(op["operation_index"]))
                for job_id, op in backend.ready_operations().items()
            }
            seen_jobs: set[str] = set()
            item_results: list[dict[str, Any]] = []
            scheduled_count = 0
            for item in requested:
                if not isinstance(item, dict):
                    item_result = {
                        "_status": "error",
                        "error": "operation_item_must_be_object",
                    }
                else:
                    job_id = str(item.get("job_id", ""))
                    try:
                        operation_index = int(item.get("operation_index", -1))
                    except (TypeError, ValueError):
                        operation_index = -1
                    if job_id in seen_jobs:
                        item_result = {
                            "_status": "error",
                            "error": "duplicate_job_in_batch",
                            "job_id": job_id,
                            "operation_index": operation_index,
                        }
                    elif (job_id, operation_index) not in initially_ready:
                        item_result = {
                            "_status": "error",
                            "error": "operation_not_ready_at_batch_start",
                            "job_id": job_id,
                            "operation_index": operation_index,
                        }
                    else:
                        seen_jobs.add(job_id)
                        item_result = backend.schedule_next_operation(
                            job_id=job_id,
                            operation_index=operation_index,
                        )
                item_results.append(item_result)
                if item_result.get("_status") == "scheduled":
                    scheduled_count += 1
            rejected_count = len(item_results) - scheduled_count
            result = {
                "_status": "scheduled_batch" if scheduled_count else "error",
                "error": None if scheduled_count else "no_operations_scheduled",
                "requested_count": len(requested),
                "scheduled_count": scheduled_count,
                "rejected_count": rejected_count,
                "results": item_results,
                "makespan": backend.makespan,
            }

        evidence = ctx.extra.get("evidence")
        if isinstance(evidence, EvidenceLogger):
            result["evidence_id"] = evidence.log(
                "co_bench_job_shop_tool_call",
                ctx.tick,
                payload={
                    "tool": "dispatch_ready_operations",
                    "ok": result.get("_status") != "error",
                    **result,
                },
                source="tool",
            )
        return result

    return handler


def _h_dispatch_job_operation(backend: CoBenchJobShopBackend):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        result = backend.schedule_next_operation(
            job_id=str(args.get("job_id", "")),
            operation_index=int(args.get("operation_index", -1)),
        )
        evidence = ctx.extra.get("evidence")
        if isinstance(evidence, EvidenceLogger):
            evidence_id = evidence.log(
                "co_bench_job_shop_tool_call",
                ctx.tick,
                payload={
                    "tool": "dispatch_job_operation",
                    "ok": result.get("_status") != "error",
                    **result,
                },
                source="tool",
            )
            result["evidence_id"] = evidence_id
        return result

    return handler


def _h_wait(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    evidence = ctx.extra.get("evidence")
    result = {"_status": "waited"}
    if isinstance(evidence, EvidenceLogger):
        result["evidence_id"] = evidence.log(
            "co_bench_job_shop_tool_call",
            ctx.tick,
            payload={"tool": "wait", "ok": True, **result},
            source="tool",
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _opt_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _reference_optimum_from_seed(seed: LogisticsScenarioSeed) -> float | None:
    reference = seed.backend_config.get("reference") or {}
    try:
        if reference.get("type") == "known_optimum":
            return float(reference["makespan"])
        if reference.get("type") == "best_known_bounds":
            return float(reference["lower_bound"])
    except (KeyError, TypeError, ValueError):
        return None
    # Fallback: CO-Bench lower_bound
    cfg = seed.backend_config.get("co_bench_job_shop") or {}
    lb = cfg.get("lower_bound")
    if lb is not None:
        try:
            return float(lb)
        except (TypeError, ValueError):
            pass
    return None


def _select_job_by_rule(
    ready: dict[str, dict[str, int]],
    jobs: dict[str, list[CoBenchJobShopOperation]],
    rule: str,
) -> str:
    """Select a ready job according to the dispatching rule."""
    job_ids = list(ready.keys())

    if rule == "spt":
        # Shortest Processing Time: pick the ready op with smallest duration
        return min(job_ids, key=lambda j: ready[j]["duration"])
    if rule == "lpt":
        # Longest Processing Time
        return max(job_ids, key=lambda j: ready[j]["duration"])
    if rule == "fifo":
        # First-In-First-Out by job index
        return min(job_ids, key=lambda j: int(j[1:]) if j[1:].isdigit() else 0)
    if rule == "mwkr":
        # Most Work Remaining: total remaining processing time for the job
        def remaining_work(job_id: str) -> int:
            ops = jobs[job_id]
            idx = ready[job_id]["operation_index"]
            return sum(op.duration for op in ops[idx:])

        return max(job_ids, key=remaining_work)

    # Default to SPT
    return min(job_ids, key=lambda j: ready[j]["duration"])
