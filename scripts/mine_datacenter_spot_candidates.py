#!/usr/bin/env python3
"""Mine bounded, source-locked Alibaba Spot GPU candidate recipes.

This is a streaming structural prefilter.  Its output is deliberately
candidate-only: empirical difficulty, headroom, and Core admission require
separate executable behavioral gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOB_TRACE = (
    ROOT / "works/clusterdata/cluster-trace-v2026-spot-gpu/job_info_df.csv"
)
DEFAULT_NODE_INVENTORY = (
    ROOT / "works/clusterdata/cluster-trace-v2026-spot-gpu/node_info_df.csv"
)
DEFAULT_OUTPUT = (
    ROOT / ".hl/artifacts/datacenter_spot_candidate_ledger_20260828.json"
)

LEDGER_SCHEMA = "operate-datacenter-spot-candidate-ledger-v1"
SOURCE_SCHEMA = "alibaba-spot-gpu-v2026-v1"
CONTIGUOUS_WINDOW_RECIPE = "alibaba-spot-gpu-window-v1"
ROW_INDICES_RECIPE = "alibaba-spot-gpu-row-indices-v2"
ROW_INDICES_ORDERING = "raw_csv_zero_based_strictly_increasing"
# Backward-compatible name used by v1 callers.
WINDOW_RECIPE = CONTIGUOUS_WINDOW_RECIPE
UPSTREAM_COMMIT = "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71"
UPSTREAM_URL = (
    "https://github.com/alibaba/clusterdata/tree/"
    f"{UPSTREAM_COMMIT}/cluster-trace-v2026-spot-gpu"
)
REQUIRED_JOB_FIELDS = {
    "job_name",
    "organization",
    "gpu_model",
    "cpu_request",
    "gpu_request",
    "worker_num",
    "submit_time",
    "duration",
    "job_type",
}
REQUIRED_NODE_FIELDS = {
    "gpu_model",
    "gpu_capacity_num",
    "cpu_num",
}


@dataclass(frozen=True)
class MiningConfig:
    """Structural thresholds; none of these asserts a difficulty label."""

    window_size: int = 8
    stride: int = 1
    horizon_ticks: int = 16
    seconds_per_tick: float = 600.0
    min_organizations: int = 2
    min_arrival_epochs: int = 3
    min_pressure_ratio: float = 1.5
    max_pressure_ratio: float = 8.0
    min_duration_ratio: float = 4.0
    max_candidates: int = 12
    max_candidates_per_gpu_model: int = 2
    recipe_version: str = ROW_INDICES_RECIPE

    def validate(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size_must_be_at_least_two")
        if self.stride < 1:
            raise ValueError("stride_must_be_positive")
        if self.horizon_ticks < 3:
            raise ValueError("horizon_ticks_must_be_at_least_three")
        if not math.isfinite(self.seconds_per_tick) or self.seconds_per_tick <= 0:
            raise ValueError("seconds_per_tick_must_be_positive")
        if self.min_organizations < 2:
            raise ValueError("min_organizations_must_be_at_least_two")
        if not 3 <= self.min_arrival_epochs <= self.horizon_ticks:
            raise ValueError("min_arrival_epochs_out_of_range")
        if not 1.0 < self.min_pressure_ratio <= self.max_pressure_ratio:
            raise ValueError("pressure_ratio_bounds_invalid")
        if self.min_duration_ratio <= 1.0:
            raise ValueError("min_duration_ratio_must_exceed_one")
        if self.max_candidates < 1 or self.max_candidates_per_gpu_model < 1:
            raise ValueError("candidate_limits_must_be_positive")
        if self.recipe_version not in {
            CONTIGUOUS_WINDOW_RECIPE,
            ROW_INDICES_RECIPE,
        }:
            raise ValueError("source_window_recipe_unsupported")


@dataclass(frozen=True)
class NodeCapacity:
    gpu_model: str
    gpu_capacity: float
    cpu_capacity: float
    ordinal: int


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _require_fields(
    fieldnames: Iterable[str] | None,
    required: set[str],
    *,
    source: str,
) -> None:
    observed = set(fieldnames or ())
    missing = sorted(required - observed)
    if missing:
        raise ValueError(f"{source}_fields_missing:{','.join(missing)}")


def _finite_float(value: Any, *, field: str, row_index: int) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"job_row_{row_index}_{field}_invalid") from exc
    if not math.isfinite(converted):
        raise ValueError(f"job_row_{row_index}_{field}_invalid")
    return converted


def _normalize_job(row: dict[str, Any], *, row_index: int) -> dict[str, Any]:
    """Match the runtime backend's v2026 source-window normalization."""
    job_id = str(row["job_name"])
    submit_time = _finite_float(
        row["submit_time"], field="submit_time", row_index=row_index
    )
    duration = _finite_float(
        row["duration"], field="duration", row_index=row_index
    )
    worker_num = int(
        _finite_float(row["worker_num"], field="worker_num", row_index=row_index)
    )
    if duration <= 0 or worker_num <= 0:
        raise ValueError(f"job_row_{row_index}_nonpositive_work")
    return {
        "job_id": job_id,
        "inst_id": job_id,
        "user": str(row["organization"]),
        "start_time": submit_time,
        "end_time": submit_time + duration,
        "duration_seconds": duration,
        "instance_count": worker_num,
        "requested_cpu_percent": _finite_float(
            row["cpu_request"], field="cpu_request", row_index=row_index
        ),
        "requested_memory_units": 0.0,
        "requested_gpu_units": _finite_float(
            row["gpu_request"], field="gpu_request", row_index=row_index
        ),
        "gpu_types": str(row["gpu_model"]),
        "priority_class": str(row["job_type"]),
    }


def _iter_job_windows(
    job_trace: Path,
    *,
    window_size: int,
    stride: int,
) -> Iterator[tuple[int, list[dict[str, Any]]]]:
    """Yield fixed windows while retaining only ``window_size`` source rows."""
    buffer: deque[dict[str, Any]] = deque(maxlen=window_size)
    with job_trace.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require_fields(reader.fieldnames, REQUIRED_JOB_FIELDS, source="job_trace")
        for row_index, row in enumerate(reader):
            buffer.append(_normalize_job(row, row_index=row_index))
            start = row_index - window_size + 1
            if len(buffer) == window_size and start % stride == 0:
                yield start, list(buffer)


def _iter_gpu_model_windows(
    job_trace: Path,
    *,
    window_size: int,
    stride: int,
) -> Iterator[tuple[list[int], list[dict[str, Any]]]]:
    """Yield bounded per-model windows with exact raw CSV row identities."""
    buffers: dict[str, deque[tuple[int, dict[str, Any]]]] = defaultdict(
        lambda: deque(maxlen=window_size)
    )
    model_row_counts: Counter[str] = Counter()
    with job_trace.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require_fields(reader.fieldnames, REQUIRED_JOB_FIELDS, source="job_trace")
        for row_index, row in enumerate(reader):
            normalized = _normalize_job(row, row_index=row_index)
            model = str(normalized["gpu_types"])
            buffer = buffers[model]
            buffer.append((row_index, normalized))
            model_row_counts[model] += 1
            model_start = model_row_counts[model] - window_size
            if len(buffer) == window_size and model_start % stride == 0:
                yield (
                    [raw_index for raw_index, _ in buffer],
                    [job for _, job in buffer],
                )


def _load_nodes(node_inventory: Path) -> dict[str, list[NodeCapacity]]:
    nodes: dict[str, list[NodeCapacity]] = defaultdict(list)
    with node_inventory.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require_fields(
            reader.fieldnames,
            REQUIRED_NODE_FIELDS,
            source="node_inventory",
        )
        for row_index, row in enumerate(reader):
            model = str(row["gpu_model"])
            gpu_capacity = _finite_float(
                row["gpu_capacity_num"],
                field="gpu_capacity_num",
                row_index=row_index,
            )
            cpu_capacity = _finite_float(
                row["cpu_num"], field="cpu_num", row_index=row_index
            )
            if not model or gpu_capacity <= 0 or cpu_capacity <= 0:
                continue
            nodes[model].append(
                NodeCapacity(
                    gpu_model=model,
                    gpu_capacity=gpu_capacity,
                    cpu_capacity=cpu_capacity,
                    ordinal=len(nodes[model]),
                )
            )
    return dict(nodes)


def _arrival_epochs(
    jobs: list[dict[str, Any]],
    *,
    seconds_per_tick: float,
    horizon_ticks: int,
) -> list[int]:
    first_submit = min(float(job["start_time"]) for job in jobs)
    return sorted(
        {
            min(
                horizon_ticks - 1,
                max(
                    0,
                    int(
                        (float(job["start_time"]) - first_submit)
                        // seconds_per_tick
                    ),
                ),
            )
            for job in jobs
        }
    )


def _job_demands(job: dict[str, Any]) -> tuple[float, float]:
    workers = float(job["instance_count"])
    return (
        float(job["requested_gpu_units"]) * workers,
        float(job["requested_cpu_percent"]) * workers,
    )


def _select_node(
    jobs: list[dict[str, Any]],
    nodes: list[NodeCapacity],
    *,
    config: MiningConfig,
) -> tuple[NodeCapacity, float, float] | None:
    demands = [_job_demands(job) for job in jobs]
    total_gpu = sum(gpu for gpu, _ in demands)
    total_cpu = sum(cpu for _, cpu in demands)
    compatible: list[tuple[tuple[float, float, int], NodeCapacity, float, float]] = []
    target = (config.min_pressure_ratio + config.max_pressure_ratio) / 2.0
    for node in nodes:
        if any(
            gpu > node.gpu_capacity or cpu > node.cpu_capacity
            for gpu, cpu in demands
        ):
            continue
        gpu_ratio = total_gpu / node.gpu_capacity
        cpu_ratio = total_cpu / node.cpu_capacity
        pressure = max(gpu_ratio, cpu_ratio)
        if not config.min_pressure_ratio <= pressure <= config.max_pressure_ratio:
            continue
        # Prefer a central pressure regime, then the tighter GPU ratio, then
        # source order.  This is structural selection, not a difficulty score.
        rank = (abs(pressure - target), -gpu_ratio, node.ordinal)
        compatible.append((rank, node, gpu_ratio, cpu_ratio))
    if not compatible:
        return None
    _, node, gpu_ratio, cpu_ratio = min(compatible, key=lambda item: item[0])
    return node, gpu_ratio, cpu_ratio


def _individually_compatible_nodes(
    jobs: list[dict[str, Any]], nodes: list[NodeCapacity]
) -> list[NodeCapacity]:
    demands = [_job_demands(job) for job in jobs]
    return [
        node
        for node in nodes
        if all(
            gpu <= node.gpu_capacity and cpu <= node.cpu_capacity
            for gpu, cpu in demands
        )
    ]


def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, ...]:
    evidence = candidate["evidence"]
    counts = evidence["priority_counts"]
    priority_balance = min(counts["HP"], counts["Spot"])
    return (
        float(priority_balance),
        float(evidence["organization_count"]),
        float(evidence["arrival_epoch_count"]),
        min(float(evidence["duration_ratio"]), 100.0),
        -float(candidate["source_window_start"]),
    )


def _suite_recipe(
    *,
    candidate_id: str,
    job_trace: Path,
    node_inventory: Path,
    job_trace_sha256: str,
    node_trace_sha256: str,
    source_row_indices: list[int],
    jobs: list[dict[str, Any]],
    node: NodeCapacity,
    window_sha256: str,
    node_inventory_sha256: str,
    config: MiningConfig,
) -> dict[str, Any]:
    job_ref = _repo_ref(job_trace)
    node_ref = _repo_ref(node_inventory)
    window_start = source_row_indices[0]
    if config.recipe_version == ROW_INDICES_RECIPE:
        time_window = {
            "selection_kind": "explicit_raw_row_indices",
            "source_row_indices": list(source_row_indices),
            "source_window_sha256": window_sha256,
        }
        selection_transform = {
            "source_window_start": window_start,
            "source_row_indices": list(source_row_indices),
            "source_row_ordering": ROW_INDICES_ORDERING,
            "selected_rows_sha256": window_sha256,
        }
    else:
        time_window = {
            "source_start": window_start,
            "source_end_exclusive": window_start + len(jobs),
            "source_window_sha256": window_sha256,
        }
        selection_transform = {"source_window_start": window_start}
    return {
        "scenario_id": candidate_id,
        "seed_id": candidate_id,
        "family": "gpu_cluster_spot_sla_control",
        "domain": "datacenter",
        "backend_kind": "alibaba_trace_sim",
        "seed": int(_stable_digest(candidate_id)[:8], 16),
        "difficulty_mode": "time_pressure",
        "difficulty_level": "basic",
        "candidate_only": True,
        "release_admission": "candidate_only",
        "horizon_ticks": config.horizon_ticks,
        "tick_minutes": config.seconds_per_tick / 60.0,
        "provenance": {
            "data_source": "Alibaba cluster-trace-v2026 Spot GPU",
            "files": [job_ref, node_ref],
            "commit": UPSTREAM_COMMIT,
            "url": UPSTREAM_URL,
            "lock_strategy": "upstream_git_commit_and_raw_sha256",
            "time_window": time_window,
        },
        "source_contract": {
            "runtime_input": [job_ref, node_ref],
            "derivation_input": [],
            "implementation_asset": [],
            "metadata": [],
            "license": [],
            "file_sha256s": {
                job_ref: job_trace_sha256,
                node_ref: node_trace_sha256,
            },
            "derived_window": {
                "sha256": window_sha256,
                "recipe_version": config.recipe_version,
            },
        },
        "backend_config": {
            "gpu_capacity_units": node.gpu_capacity,
            "cpu_capacity_units": node.cpu_capacity,
            "source_time_scale": {
                "mode": "source_seconds",
                "seconds_per_tick": config.seconds_per_tick,
            },
            "source_transform": {
                "recipe_version": config.recipe_version,
                "source_schema": SOURCE_SCHEMA,
                "source_file_roles": {
                    "job_trace": job_ref,
                    "node_inventory": node_ref,
                },
                "job_trace_sha256": job_trace_sha256,
                "node_trace_sha256": node_trace_sha256,
                **selection_transform,
                "window_size": len(jobs),
                "arrival_horizon_ticks": config.horizon_ticks,
                "source_window_sha256": window_sha256,
                "selected_gpu_model": node.gpu_model,
                "node_selection_recipe": {
                    "kind": "gpu_model_ordinal",
                    "gpu_model": node.gpu_model,
                    "ordinal": node.ordinal,
                    "ordering": "source_row_order_after_gpu_model_filter",
                },
                "selected_node_inventory_sha256": node_inventory_sha256,
            },
        },
    }


def _evaluate_window(
    *,
    source_row_indices: list[int],
    jobs: list[dict[str, Any]],
    nodes_by_model: dict[str, list[NodeCapacity]],
    job_trace: Path,
    node_inventory: Path,
    job_trace_sha256: str,
    node_trace_sha256: str,
    config: MiningConfig,
) -> tuple[dict[str, Any] | None, str | None]:
    window_start = source_row_indices[0]
    gpu_models = {str(job["gpu_types"]) for job in jobs}
    if len(gpu_models) != 1 or not next(iter(gpu_models), ""):
        return None, "mixed_gpu_models"
    gpu_model = next(iter(gpu_models))
    priorities = Counter(str(job["priority_class"]) for job in jobs)
    if priorities["HP"] == 0 or priorities["Spot"] == 0:
        return None, "missing_hp_spot_mix"
    organizations = {str(job["user"]) for job in jobs}
    if len(organizations) < config.min_organizations:
        return None, "insufficient_organizations"
    arrival_epochs = _arrival_epochs(
        jobs,
        seconds_per_tick=config.seconds_per_tick,
        horizon_ticks=config.horizon_ticks,
    )
    if len(arrival_epochs) < config.min_arrival_epochs:
        return None, "insufficient_arrival_epochs"
    model_nodes = nodes_by_model.get(gpu_model, [])
    if not model_nodes:
        return None, "gpu_model_missing_in_node_inventory"
    compatible_nodes = _individually_compatible_nodes(jobs, model_nodes)
    if not compatible_nodes:
        return None, "no_compatible_node_capacity"
    node_match = _select_node(
        jobs,
        compatible_nodes,
        config=config,
    )
    if node_match is None:
        return None, "resource_pressure_out_of_range"
    node, gpu_ratio, cpu_ratio = node_match
    durations = [float(job["duration_seconds"]) for job in jobs]
    duration_ratio = max(durations) / min(durations)
    if duration_ratio < config.min_duration_ratio:
        return None, "insufficient_duration_conflict"

    normalized_inventory = [
        {
            "gpu_model": node.gpu_model,
            "gpu_capacity_num": node.gpu_capacity,
            "cpu_num": node.cpu_capacity,
        }
    ]
    window_sha256 = _stable_digest(jobs)
    node_inventory_sha256 = _stable_digest(normalized_inventory)
    identity = _stable_digest(
        {
            "job_trace_sha256": job_trace_sha256,
            "node_trace_sha256": node_trace_sha256,
            "source_window_start": window_start,
            "source_row_indices": source_row_indices,
            "window_size": len(jobs),
            "window_sha256": window_sha256,
            "node_inventory_sha256": node_inventory_sha256,
        }
    )
    candidate_id = (
        "datacenter/alibaba_spot_gpu_candidate/structural_prefilter/"
        f"{gpu_model.lower().replace('/', '-')}_w{window_start}_{identity[:12]}"
    )
    recipe = _suite_recipe(
        candidate_id=candidate_id,
        job_trace=job_trace,
        node_inventory=node_inventory,
        job_trace_sha256=job_trace_sha256,
        node_trace_sha256=node_trace_sha256,
        source_row_indices=source_row_indices,
        jobs=jobs,
        node=node,
        window_sha256=window_sha256,
        node_inventory_sha256=node_inventory_sha256,
        config=config,
    )
    return (
        {
            "candidate_id": candidate_id,
            "candidate_only": True,
            "source_window_start": window_start,
            "source_end_exclusive": source_row_indices[-1] + 1,
            "source_row_indices": list(source_row_indices),
            "source_window_sha256": window_sha256,
            "selected_node_inventory_sha256": node_inventory_sha256,
            "difficulty_label": "pending_behavioral_calibration",
            "decision_depth_claimed": False,
            "independent_decision_axes": [
                "cross_epoch_arrivals",
                "duration_heterogeneity",
                "hp_spot_priority_tradeoff",
                "multi_tenant_resource_contention",
            ],
            "evidence": {
                "gpu_model": gpu_model,
                "priority_counts": {
                    "HP": priorities["HP"],
                    "Spot": priorities["Spot"],
                },
                "organization_count": len(organizations),
                "arrival_epoch_count": len(arrival_epochs),
                "arrival_epochs": arrival_epochs,
                "gpu_demand_capacity_ratio": round(gpu_ratio, 6),
                "cpu_demand_capacity_ratio": round(cpu_ratio, 6),
                "duration_ratio": round(duration_ratio, 6),
                "all_jobs_individually_schedulable": True,
            },
            "required_next_gates": [
                "runtime_source_consumption",
                "deterministic_replay",
                "bounded_action_timing_reference",
                "positive_material_headroom",
                "native_task_completion",
                "process_capability",
                "safety_non_regression",
                "effective_source_deduplication",
            ],
            "suite_recipe": recipe,
        },
        None,
    )


def mine_candidates(
    *,
    job_trace: Path,
    node_inventory: Path,
    config: MiningConfig | None = None,
) -> dict[str, Any]:
    """Stream the job trace and return a bounded candidate-only ledger."""
    config = config or MiningConfig()
    config.validate()
    job_trace = job_trace.resolve()
    node_inventory = node_inventory.resolve()
    if not job_trace.is_file():
        raise FileNotFoundError(job_trace)
    if not node_inventory.is_file():
        raise FileNotFoundError(node_inventory)

    job_trace_sha256 = _sha256_file(job_trace)
    node_trace_sha256 = _sha256_file(node_inventory)
    nodes_by_model = _load_nodes(node_inventory)
    retained: dict[
        str,
        list[tuple[tuple[float, ...], str, dict[str, Any]]],
    ] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()
    windows_evaluated = 0
    job_rows_streamed = 0
    if config.recipe_version == ROW_INDICES_RECIPE:
        windows = _iter_gpu_model_windows(
            job_trace,
            window_size=config.window_size,
            stride=config.stride,
        )
    else:
        windows = (
            (list(range(window_start, window_start + len(jobs))), jobs)
            for window_start, jobs in _iter_job_windows(
                job_trace,
                window_size=config.window_size,
                stride=config.stride,
            )
        )
    for source_row_indices, jobs in windows:
        windows_evaluated += 1
        job_rows_streamed = max(job_rows_streamed, source_row_indices[-1] + 1)
        candidate, reason = _evaluate_window(
            source_row_indices=source_row_indices,
            jobs=jobs,
            nodes_by_model=nodes_by_model,
            job_trace=job_trace,
            node_inventory=node_inventory,
            job_trace_sha256=job_trace_sha256,
            node_trace_sha256=node_trace_sha256,
            config=config,
        )
        if candidate is None:
            rejection_counts[str(reason)] += 1
            continue
        model = str(candidate["evidence"]["gpu_model"])
        item = (_candidate_rank(candidate), candidate["candidate_id"], candidate)
        heap = retained[model]
        heapq.heappush(heap, item)
        pool_limit = config.max_candidates_per_gpu_model * 8
        if len(heap) > pool_limit:
            heapq.heappop(heap)

    # The final partial source tail never forms a window.  Count all rows by a
    # second streaming scalar pass only when stride/window geometry hid a tail.
    with job_trace.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _require_fields(reader.fieldnames, REQUIRED_JOB_FIELDS, source="job_trace")
        job_rows_streamed = sum(1 for _ in reader)

    ranked = [item for heap in retained.values() for item in heap]
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    candidates: list[dict[str, Any]] = []
    model_counts: Counter[str] = Counter()
    for _, _, candidate in ranked:
        if len(candidates) >= config.max_candidates:
            break
        model = str(candidate["evidence"]["gpu_model"])
        if model_counts[model] >= config.max_candidates_per_gpu_model:
            continue
        candidate_indices = set(candidate["source_row_indices"])
        if any(
            candidate_indices.intersection(selected["source_row_indices"])
            for selected in candidates
        ):
            continue
        candidates.append(candidate)
        model_counts[model] += 1
    candidates.sort(key=lambda row: (row["source_window_start"], row["candidate_id"]))
    source_suite_recipe = {
        "schema_version": "operate-datacenter-spot-candidate-suite-recipe-v1",
        "candidate_only": True,
        "core_admission_claimed": False,
        "scenarios": [candidate["suite_recipe"] for candidate in candidates],
    }
    return {
        "schema_version": LEDGER_SCHEMA,
        "status": "complete_candidate_only",
        "candidate_only": True,
        "core_admission_claimed": False,
        "leaderboard_eligible": False,
        "source": {
            "dataset": "Alibaba cluster-trace-v2026 Spot GPU",
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_url": UPSTREAM_URL,
            "license_status": "research_use_terms_no_explicit_redistribution_license",
            "public_packaging": "fetch_build_only",
            "lock_strategy": "upstream_git_commit_and_raw_sha256",
            "job_trace": _repo_ref(job_trace),
            "job_trace_sha256": job_trace_sha256,
            "node_inventory": _repo_ref(node_inventory),
            "node_trace_sha256": node_trace_sha256,
        },
        "selection_policy": {
            "difficulty_basis": "independent_decision_axes_not_event_count",
            "window_size": config.window_size,
            "stride": config.stride,
            "horizon_ticks": config.horizon_ticks,
            "seconds_per_tick": config.seconds_per_tick,
            "recipe_version": config.recipe_version,
            "required_priority_classes": ["HP", "Spot"],
            "min_organizations": config.min_organizations,
            "min_arrival_epochs": config.min_arrival_epochs,
            "pressure_ratio_range": [
                config.min_pressure_ratio,
                config.max_pressure_ratio,
            ],
            "min_duration_ratio": config.min_duration_ratio,
            "max_candidates": config.max_candidates,
            "max_candidates_per_gpu_model": config.max_candidates_per_gpu_model,
            "source_overlap_policy": "greedy_disjoint_windows",
        },
        "statistics": {
            "candidate_count": len(candidates),
            "job_rows_streamed": job_rows_streamed,
            "windows_evaluated": windows_evaluated,
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "candidates": candidates,
        "source_suite_recipe": source_suite_recipe,
    }


def write_candidate_ledger(
    report: dict[str, Any],
    output: Path,
    *,
    repo_root: Path = ROOT,
) -> Path:
    """Write only outside the release tree and return the resolved path."""
    root = repo_root.resolve()
    resolved = output if output.is_absolute() else root / output
    resolved = resolved.resolve()
    if resolved.is_relative_to(root / "release"):
        raise ValueError("candidate_output_must_not_target_release")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-trace", type=Path, default=DEFAULT_JOB_TRACE)
    parser.add_argument("--node-inventory", type=Path, default=DEFAULT_NODE_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--horizon-ticks", type=int, default=16)
    parser.add_argument("--seconds-per-tick", type=float, default=600.0)
    parser.add_argument("--min-organizations", type=int, default=2)
    parser.add_argument("--min-arrival-epochs", type=int, default=3)
    parser.add_argument("--min-pressure-ratio", type=float, default=1.5)
    parser.add_argument("--max-pressure-ratio", type=float, default=8.0)
    parser.add_argument("--min-duration-ratio", type=float, default=4.0)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--max-candidates-per-gpu-model", type=int, default=2)
    parser.add_argument(
        "--recipe-version",
        choices=(CONTIGUOUS_WINDOW_RECIPE, ROW_INDICES_RECIPE),
        default=ROW_INDICES_RECIPE,
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write the candidate-only ledger; otherwise print a dry-run summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = MiningConfig(
        window_size=args.window_size,
        stride=args.stride,
        horizon_ticks=args.horizon_ticks,
        seconds_per_tick=args.seconds_per_tick,
        min_organizations=args.min_organizations,
        min_arrival_epochs=args.min_arrival_epochs,
        min_pressure_ratio=args.min_pressure_ratio,
        max_pressure_ratio=args.max_pressure_ratio,
        min_duration_ratio=args.min_duration_ratio,
        max_candidates=args.max_candidates,
        max_candidates_per_gpu_model=args.max_candidates_per_gpu_model,
        recipe_version=args.recipe_version,
    )
    report = mine_candidates(
        job_trace=args.job_trace,
        node_inventory=args.node_inventory,
        config=config,
    )
    if args.execute:
        path = write_candidate_ledger(report, args.output)
        print(path)
    else:
        print(
            json.dumps(
                {
                    "candidate_only": True,
                    "candidate_count": len(report["candidates"]),
                    "output_not_written": str(args.output),
                    "statistics": report["statistics"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
