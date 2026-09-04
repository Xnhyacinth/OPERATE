"""Deterministic, trace-backed GPU cluster scheduling simulator."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..seeds.schema import DatacenterScenarioSeed

RESERVED_GPU_UNIT_TICK_COST = 0.1
SOURCE_WINDOW_RECIPE = "alibaba-trace-window-v1"
SPOT_SOURCE_WINDOW_RECIPE = "alibaba-spot-gpu-window-v1"
SPOT_SOURCE_ROW_INDICES_RECIPE = "alibaba-spot-gpu-row-indices-v2"
SPOT_SOURCE_ROW_ORDERING = "raw_csv_zero_based_strictly_increasing"
SPOT_SOURCE_SCHEMA = "alibaba-spot-gpu-v2026-v1"
QUEUE_POLICIES = (
    "fifo",
    "shortest_job_first",
    "least_gpu_first",
    "deadline_criticality_first",
)
PERTURBATION_EVENT_CLASS = MappingProxyType(
    {
        "capacity_reduction": "alarm",
        "queue_burst": "alarm",
        "sla_deadline_pressure": "alarm",
    }
)
REPO_ROOT = Path(__file__).resolve().parents[3]


def _coerce_source_job(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "job_id": str(row["job_id"]),
        "inst_id": str(row["inst_id"]),
        "user": str(row["user"]),
        "start_time": float(row["start_time"]),
        "end_time": float(row["end_time"]),
        "duration_seconds": float(row["duration_seconds"]),
        "instance_count": int(float(row["instance_count"])),
        "requested_cpu_percent": float(row["requested_cpu_percent"]),
        "requested_memory_units": float(row["requested_memory_units"]),
        "requested_gpu_units": float(row["requested_gpu_units"]),
        "gpu_types": str(row["gpu_types"]),
    }
    priority_class = row.get("priority_class", row.get("job_type"))
    if priority_class not in (None, ""):
        normalized["priority_class"] = str(priority_class)
    return normalized


def _coerce_spot_gpu_job(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one row from the official v2026 Spot GPU job trace."""
    job_id = str(row["job_name"])
    submit_time = float(row["submit_time"])
    duration = float(row["duration"])
    return {
        "job_id": job_id,
        "inst_id": job_id,
        "user": str(row["organization"]),
        "start_time": submit_time,
        "end_time": submit_time + duration,
        "duration_seconds": duration,
        "instance_count": int(row["worker_num"]),
        "requested_cpu_percent": float(row["cpu_request"]),
        "requested_memory_units": 0.0,
        "requested_gpu_units": float(row["gpu_request"]),
        "gpu_types": str(row["gpu_model"]),
        "priority_class": str(row["job_type"]),
    }


def _stable_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def resolve_alibaba_source_window(
    *,
    provenance_files: list[str],
    time_window: dict[str, Any],
    backend_config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Resolve one contiguous Alibaba trace window from locked source identity."""
    source_transform = dict(backend_config.get("source_transform") or {})
    if not source_transform:
        raise ValueError("source_contract_window_metadata_missing")
    if source_transform.get("source_schema") == SPOT_SOURCE_SCHEMA:
        return _resolve_spot_gpu_source_window(
            provenance_files=provenance_files,
            time_window=time_window,
            source_transform=source_transform,
            repo_root=repo_root,
        )
    if len(provenance_files) != 1:
        raise ValueError("source_file_count_invalid")
    source_path = str(provenance_files[0])
    resolved = Path(source_path)
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise ValueError("source_file_missing")
    source_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    expected_source_sha256 = str(
        source_transform.get("trace_sha256") or ""
    ).removeprefix("sha256:")
    if not expected_source_sha256 or source_sha256 != expected_source_sha256:
        raise ValueError("source_hash_mismatch")

    first_job_id = str(time_window.get("first_job_id") or "")
    last_job_id = str(time_window.get("last_job_id") or "")
    window_size = int(source_transform.get("window_size") or 0)
    if not first_job_id or not last_job_id or window_size <= 0:
        raise ValueError("source_contract_window_metadata_missing")
    with resolved.open(newline="", encoding="utf-8") as stream:
        jobs = [_coerce_source_job(row) for row in csv.DictReader(stream)]
    starts = [index for index, row in enumerate(jobs) if row["job_id"] == first_job_id]
    ends = [index for index, row in enumerate(jobs) if row["job_id"] == last_job_id]
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("source_window_boundary_not_unique")
    row_start, row_end = starts[0], ends[0]
    if row_end < row_start or row_end - row_start + 1 != window_size:
        raise ValueError("source_window_not_contiguous")
    normalized_jobs = jobs[row_start : row_end + 1]
    window_sha256 = _stable_digest(normalized_jobs)
    expected_window_sha256 = str(
        source_transform.get("source_window_sha256")
        or time_window.get("source_window_sha256")
        or ""
    ).removeprefix("sha256:")
    if expected_window_sha256 and window_sha256 != expected_window_sha256:
        raise ValueError("source_window_hash_mismatch")
    return {
        "source_path": source_path,
        "resolved_source_path": str(resolved),
        "source_sha256": source_sha256,
        "recipe_version": SOURCE_WINDOW_RECIPE,
        "row_start": row_start,
        "row_end": row_end,
        "job_id_start": first_job_id,
        "job_id_end": last_job_id,
        "normalized_jobs": normalized_jobs,
        "sha256": window_sha256,
        "runtime_window_digest": window_sha256,
        "runtime_opened_assets": [
            {
                "source_path": source_path,
                "resolved_source_path": str(resolved),
                "sha256": source_sha256,
                "role": "derivation_input",
            }
        ],
        "consumed_channels": [
            "arrival_time",
            "cpu_demand",
            "duration",
            "gpu_demand",
        ],
    }


def _resolve_source_asset(
    source_path: str,
    *,
    expected_sha256: str,
    repo_root: Path,
) -> tuple[Path, str]:
    resolved = Path(source_path)
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise ValueError("source_file_missing")
    observed_sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if not expected_sha256 or observed_sha256 != expected_sha256.removeprefix(
        "sha256:"
    ):
        raise ValueError("source_hash_mismatch")
    return resolved, observed_sha256


def _resolve_spot_gpu_source_window(
    *,
    provenance_files: list[str],
    time_window: dict[str, Any],
    source_transform: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Open the v2026 Spot GPU job trace and physical node inventory."""
    roles = dict(source_transform.get("source_file_roles") or {})
    job_source = str(roles.get("job_trace") or "")
    node_source = str(roles.get("node_inventory") or "")
    if (
        len(provenance_files) != 2
        or not job_source
        or not node_source
        or set(provenance_files) != {job_source, node_source}
    ):
        raise ValueError("source_file_roles_invalid")
    job_path, job_sha256 = _resolve_source_asset(
        job_source,
        expected_sha256=str(source_transform.get("job_trace_sha256") or ""),
        repo_root=repo_root,
    )
    node_path, node_sha256 = _resolve_source_asset(
        node_source,
        expected_sha256=str(source_transform.get("node_trace_sha256") or ""),
        repo_root=repo_root,
    )
    recipe_version = str(
        source_transform.get("recipe_version") or SPOT_SOURCE_WINDOW_RECIPE
    )
    window_size = int(source_transform.get("window_size") or 0)
    normalized_jobs: list[dict[str, Any]] = []
    source_row_indices: list[int] | None = None
    source_row_ordering: str | None = None
    if recipe_version == SPOT_SOURCE_WINDOW_RECIPE:
        raw_row_start = source_transform.get("source_window_start")
        row_start = -1 if raw_row_start is None else int(raw_row_start)
        expected_end = int(
            time_window.get("source_end_exclusive") or row_start + window_size
        )
        if row_start < 0 or window_size <= 0 or expected_end != row_start + window_size:
            raise ValueError("source_contract_window_metadata_missing")
        row_end = row_start + window_size - 1
        with job_path.open(newline="", encoding="utf-8") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                if index < row_start:
                    continue
                if index > row_end:
                    break
                normalized_jobs.append(_coerce_spot_gpu_job(row))
        if len(normalized_jobs) != window_size:
            raise ValueError("source_window_not_contiguous")
    elif recipe_version == SPOT_SOURCE_ROW_INDICES_RECIPE:
        raw_indices = source_transform.get("source_row_indices")
        if (
            not isinstance(raw_indices, list)
            or not raw_indices
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_indices
            )
        ):
            raise ValueError("source_row_indices_invalid")
        source_row_indices = list(raw_indices)
        if (
            any(value < 0 for value in source_row_indices)
            or source_row_indices != sorted(set(source_row_indices))
            or len(source_row_indices) != window_size
        ):
            raise ValueError("source_row_indices_invalid")
        source_row_ordering = str(source_transform.get("source_row_ordering") or "")
        if source_row_ordering != SPOT_SOURCE_ROW_ORDERING:
            raise ValueError("source_row_ordering_invalid")
        if (
            time_window.get("selection_kind") != "explicit_raw_row_indices"
            or time_window.get("source_row_indices") != source_row_indices
        ):
            raise ValueError("source_row_indices_binding_mismatch")
        row_start, row_end = source_row_indices[0], source_row_indices[-1]
        selected = set(source_row_indices)
        with job_path.open(newline="", encoding="utf-8") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                if index > row_end:
                    break
                if index in selected:
                    normalized_jobs.append(_coerce_spot_gpu_job(row))
        if len(normalized_jobs) != window_size:
            raise ValueError("source_selected_rows_missing")
    else:
        raise ValueError("source_window_recipe_unsupported")
    window_sha256 = _stable_digest(normalized_jobs)
    expected_window_sha256 = str(
        source_transform.get("source_window_sha256")
        or time_window.get("source_window_sha256")
        or ""
    ).removeprefix("sha256:")
    if not expected_window_sha256 or window_sha256 != expected_window_sha256:
        raise ValueError("source_window_hash_mismatch")
    selected_rows_sha256: str | None = None
    if recipe_version == SPOT_SOURCE_ROW_INDICES_RECIPE:
        selected_rows_sha256 = str(
            source_transform.get("selected_rows_sha256") or ""
        ).removeprefix("sha256:")
        if selected_rows_sha256 != window_sha256:
            raise ValueError("source_selected_rows_hash_mismatch")
        time_window_sha256 = str(
            time_window.get("source_window_sha256") or ""
        ).removeprefix("sha256:")
        if time_window_sha256 != window_sha256:
            raise ValueError("source_window_hash_mismatch")

    selected_gpu_model = str(source_transform.get("selected_gpu_model") or "")
    node_selection_recipe = dict(source_transform.get("node_selection_recipe") or {})
    expected_recipe = {
        "kind": "gpu_model_ordinal",
        "gpu_model": selected_gpu_model,
        "ordinal": int(node_selection_recipe.get("ordinal") or 0),
        "ordering": "source_row_order_after_gpu_model_filter",
    }
    if (
        not selected_gpu_model
        or node_selection_recipe != expected_recipe
        or expected_recipe["ordinal"] < 0
    ):
        raise ValueError("source_node_selection_missing")
    with node_path.open(newline="", encoding="utf-8") as stream:
        matching_nodes = [
            dict(row)
            for row in csv.DictReader(stream)
            if str(row.get("gpu_model") or "") == selected_gpu_model
        ]
    ordinal = int(expected_recipe["ordinal"])
    if ordinal >= len(matching_nodes):
        raise ValueError("source_node_selection_missing")
    selected_node = matching_nodes[ordinal]
    selected_inventory = [
        {
            "gpu_model": str(selected_node["gpu_model"]),
            "gpu_capacity_num": float(selected_node["gpu_capacity_num"]),
            "cpu_num": float(selected_node["cpu_num"]),
        }
    ]
    if any(
        str(job.get("gpu_types") or "") != selected_gpu_model for job in normalized_jobs
    ):
        raise ValueError("source_node_inventory_mismatch")
    gpu_capacity = float(selected_node["gpu_capacity_num"])
    cpu_capacity = float(selected_node["cpu_num"])
    node_inventory_sha256 = _stable_digest(selected_inventory)
    expected_inventory_sha256 = str(
        source_transform.get("selected_node_inventory_sha256") or ""
    ).removeprefix("sha256:")
    if (
        not expected_inventory_sha256
        or node_inventory_sha256 != expected_inventory_sha256
    ):
        raise ValueError("source_node_inventory_hash_mismatch")
    return {
        "source_path": job_source,
        "resolved_source_path": str(job_path),
        "source_sha256": job_sha256,
        "recipe_version": recipe_version,
        "row_start": row_start,
        "row_end": row_end,
        **(
            {
                "source_row_indices": source_row_indices,
                "source_row_ordering": source_row_ordering,
                "selected_rows_sha256": selected_rows_sha256,
            }
            if source_row_indices is not None
            else {}
        ),
        "normalized_jobs": normalized_jobs,
        "sha256": window_sha256,
        "runtime_window_digest": window_sha256,
        "runtime_opened_assets": [
            {
                "source_path": job_source,
                "resolved_source_path": str(job_path),
                "sha256": job_sha256,
                "role": "runtime_job_trace",
            },
            {
                "source_path": node_source,
                "resolved_source_path": str(node_path),
                "sha256": node_sha256,
                "role": "runtime_node_inventory",
            },
        ],
        "node_selection_recipe": expected_recipe,
        "selected_node_inventory_sha256": node_inventory_sha256,
        "source_gpu_capacity": gpu_capacity,
        "source_cpu_capacity": cpu_capacity,
        "consumed_channels": [
            "arrival_time",
            "cpu_demand",
            "duration",
            "gpu_demand",
            "priority_class",
            "node_gpu_capacity",
            "node_cpu_capacity",
        ],
    }


@dataclass
class _Job:
    job_id: str
    user: str
    submit_tick: int
    duration_ticks: int
    gpu_units: float
    cpu_units: float
    criticality: float
    due_tick: int
    remaining_ticks: int
    status: str = "future"
    wait_ticks: int = 0
    completion_tick: int | None = None
    preemptions: int = 0


@dataclass
class DatacenterTickRecord:
    tick: int
    gpu_capacity: float = 0.0
    cpu_capacity: float = 0.0
    arrived_jobs: int = 0
    queued_jobs: int = 0
    running_jobs: int = 0
    completed_jobs: int = 0
    gpu_demand: float = 0.0
    gpu_allocated: float = 0.0
    cpu_demand: float = 0.0
    cpu_allocated: float = 0.0
    queue_wait_cost: float = 0.0
    compute_cost: float = 0.0
    sla_violation_cost: float = 0.0
    preemption_waste_cost: float = 0.0
    reserve_capacity_cost: float = 0.0
    realized_events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False


class AlibabaTraceBackend:
    """Schedules immutable Alibaba-trace jobs with live native controls."""

    backend_kind = "alibaba_trace_sim"

    def __init__(self) -> None:
        self._seed_obj: DatacenterScenarioSeed | None = None
        self._tick = 0
        self._horizon = 8
        self._jobs: dict[str, _Job] = {}
        self._queue_policy = "fifo"
        self._queue_policy_generation = 1
        self._gpu_capacity = 1.0
        self._cpu_capacity = 1.0
        self._reserved_gpu = 0.0
        self._capacity_factor = 1.0
        self._capacity_restore_tick = -1
        self._queue_pressure_factor = 1.0
        self._queue_pressure_restore_tick = -1
        self._sla_deadline_shift = 0
        self._sla_pressure_restore_tick = -1
        self._records: list[DatacenterTickRecord] = []
        self._preemption_waste_total = 0.0
        self._reserve_cost_total = 0.0
        self._recorded_preemption_waste_total = 0.0
        self._recorded_reserve_cost_total = 0.0
        self._events: list[dict[str, Any]] = []
        self._control_events: list[dict[str, Any]] = []
        self._pending_reservations: list[dict[str, Any]] = []
        self._active_reservation_expiries: list[tuple[int, float]] = []
        self._source_window: dict[str, Any] | None = None
        self._source_consumption_ticks: list[int] = []
        self._post_source_state_digests: list[str] = []
        self._initial_state_digest = ""
        self._pending_action_effects: list[dict[str, Any]] = []
        self._action_effects_by_token: dict[str, dict[str, Any]] = {}
        self._unbound_effect_tokens: dict[str, list[str]] = {}
        self._effect_sequence = 0
        self._perturbation_events: list[dict[str, Any]] = []
        self._strategy_reversal_ticks: set[int] = set()
        self._policy_review_ledger: list[dict[str, Any]] = []
        self._reviewed_event_ids: set[str] = set()
        self._policy_effect_evidence_by_generation: dict[int, str] = {}
        self._source_gpu_reservation_limit: float | None = None
        self._tenant_ids: tuple[str, ...] = ()

    def reset(self, seed_obj: DatacenterScenarioSeed) -> None:
        self._seed_obj = seed_obj
        self._tick = 0
        self._horizon = int(seed_obj.horizon_ticks)
        initial_queue_policy = str(
            seed_obj.backend_config.get("initial_queue_policy") or "fifo"
        )
        if initial_queue_policy not in QUEUE_POLICIES:
            raise ValueError("unknown_queue_policy")
        self._queue_policy = initial_queue_policy
        self._queue_policy_generation = 1
        self._gpu_capacity = float(
            seed_obj.backend_config.get("gpu_capacity_units") or 1.0
        )
        self._cpu_capacity = float(
            seed_obj.backend_config.get("cpu_capacity_units") or 1.0
        )
        self._reserved_gpu = 0.0
        self._capacity_factor = 1.0
        self._capacity_restore_tick = -1
        self._queue_pressure_factor = 1.0
        self._queue_pressure_restore_tick = -1
        self._sla_deadline_shift = 0
        self._sla_pressure_restore_tick = -1
        self._records = []
        self._preemption_waste_total = 0.0
        self._reserve_cost_total = 0.0
        self._recorded_preemption_waste_total = 0.0
        self._recorded_reserve_cost_total = 0.0
        self._events = []
        self._control_events = []
        self._pending_reservations = []
        self._active_reservation_expiries = []
        self._source_consumption_ticks = []
        self._post_source_state_digests = []
        self._pending_action_effects = []
        self._action_effects_by_token = {}
        self._unbound_effect_tokens = {}
        self._effect_sequence = 0
        self._perturbation_events = []
        self._strategy_reversal_ticks = set()
        self._policy_review_ledger = []
        self._reviewed_event_ids = set()
        self._policy_effect_evidence_by_generation = {}
        self._source_gpu_reservation_limit = None
        self._tenant_ids = ()
        baked_jobs = list(seed_obj.backend_config.get("jobs") or [])
        source_transform = dict(seed_obj.backend_config.get("source_transform") or {})
        if source_transform:
            self._source_window = resolve_alibaba_source_window(
                provenance_files=list(seed_obj.provenance.files),
                time_window=dict(seed_obj.provenance.time_window),
                backend_config=seed_obj.backend_config,
                repo_root=REPO_ROOT,
            )
            source_jobs = list(self._source_window["normalized_jobs"])
            if source_transform.get("source_schema") == SPOT_SOURCE_SCHEMA:
                if baked_jobs:
                    raise ValueError("spot_source_baked_jobs_forbidden")
            else:
                normalized_baked = [_coerce_source_job(job) for job in baked_jobs]
                if normalized_baked != source_jobs:
                    raise ValueError("source_baked_job_mismatch")
            source_gpu_capacity = self._source_window.get("source_gpu_capacity")
            source_cpu_capacity = self._source_window.get("source_cpu_capacity")
            if source_gpu_capacity is not None and (
                not math.isclose(
                    self._gpu_capacity,
                    float(source_gpu_capacity),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or not math.isclose(
                    self._cpu_capacity,
                    float(source_cpu_capacity),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError("source_node_capacity_mismatch")
            if source_gpu_capacity is not None:
                self._source_gpu_reservation_limit = float(source_gpu_capacity)
        else:
            self._source_window = None
            source_jobs = baked_jobs
        if not source_jobs:
            raise ValueError("alibaba_trace_sim requires source-locked jobs")
        starts = [float(job.get("start_time") or 0.0) for job in source_jobs]
        min_start = min(starts)
        tick_seconds = max(1.0, float(seed_obj.tick_minutes) * 60.0)
        source_time = dict(seed_obj.backend_config.get("source_time_scale") or {})
        spot_source = source_transform.get("source_schema") == SPOT_SOURCE_SCHEMA
        use_source_clock = source_time.get("mode") == "source_seconds"
        source_seconds_per_tick = float(
            source_time.get("seconds_per_tick") or tick_seconds
        )
        if use_source_clock and not math.isclose(
            source_seconds_per_tick,
            tick_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("source_time_scale_tick_mismatch")
        span = max(1.0, max(starts) - min_start)
        arrival_horizon_ticks = int(
            source_transform.get("arrival_horizon_ticks") or self._horizon
        )
        if not 2 <= arrival_horizon_ticks <= self._horizon:
            raise ValueError("source_arrival_horizon_invalid")
        stakeholders = {item.load_id: item for item in seed_obj.load_assignments}
        self._jobs = {}
        spot_tenant_ids: dict[str, str] = {}
        for index, row in enumerate(source_jobs):
            if spot_source:
                job_id = f"trace_job_{index:03d}"
                raw_user = str(row.get("user") or "unknown")
                user = spot_tenant_ids.setdefault(
                    raw_user,
                    f"trace_tenant_{len(spot_tenant_ids):03d}",
                )
            else:
                job_id = str(row.get("job_id") or row.get("inst_id") or index)
                user = str(row.get("user") or "unknown")
            duration_seconds = max(1.0, float(row.get("duration_seconds") or 1.0))
            source_offset_seconds = (
                float(row.get("start_time") or min_start) - min_start
            )
            if use_source_clock:
                submit_tick = min(
                    arrival_horizon_ticks - 1,
                    max(0, int(source_offset_seconds // source_seconds_per_tick)),
                )
                duration_ticks = max(
                    1,
                    int(math.ceil(duration_seconds / source_seconds_per_tick)),
                )
            else:
                submit_tick = min(
                    arrival_horizon_ticks - 1,
                    max(
                        0,
                        int(
                            round(
                                source_offset_seconds
                                / span
                                * max(1, arrival_horizon_ticks - 2)
                            )
                        ),
                    ),
                )
                duration_ticks = max(1, int(math.ceil(duration_seconds / tick_seconds)))
            assignment = stakeholders.get(job_id)
            priority_class = str(row.get("priority_class") or "")
            source_criticality = {"HP": 0.9, "Spot": 0.25}.get(priority_class)
            if source_criticality is not None:
                if assignment is not None and not math.isclose(
                    float(assignment.criticality),
                    source_criticality,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError("source_priority_assignment_mismatch")
                criticality = source_criticality
            else:
                criticality = (
                    float(assignment.criticality) if assignment is not None else 0.3
                )
            gpu = max(
                0.0,
                float(row.get("requested_gpu_units") or row.get("num_gpu") or 0.0)
                * float(row.get("instance_count") or row.get("num_inst") or 1.0),
            )
            cpu = max(
                0.0,
                float(row.get("requested_cpu_percent") or row.get("num_cpu") or 0.0)
                * float(row.get("instance_count") or row.get("num_inst") or 1.0),
            )
            self._jobs[job_id] = _Job(
                job_id=job_id,
                user=user,
                submit_tick=submit_tick,
                duration_ticks=duration_ticks,
                gpu_units=gpu,
                cpu_units=cpu,
                criticality=criticality,
                due_tick=min(
                    self._horizon - 1,
                    submit_tick + max(1, int(math.ceil(duration_ticks * 1.5))),
                ),
                remaining_ticks=duration_ticks,
            )
        self._tenant_ids = tuple(sorted({job.user for job in self._jobs.values()}))
        self._initial_state_digest = self._source_state_digest()
        if self._source_window is not None:
            initial_arrivals = [
                job
                for job in self._jobs.values()
                if job.status == "future" and job.submit_tick == 0
            ]
            for job in initial_arrivals:
                job.status = "queued"
            if initial_arrivals:
                self._source_consumption_ticks.append(0)
                self._post_source_state_digests.append(self._source_state_digest())

    def tenant_ids(self) -> tuple[str, ...]:
        """Return public runtime tenant identities used by native stakeholders."""
        return self._tenant_ids

    def tick(self, current_tick: int) -> DatacenterTickRecord:
        self._tick = int(current_tick)
        self._events = [
            self._action_effect_event(effect) for effect in self._pending_action_effects
        ]
        self._pending_action_effects = []
        self._expire_reservations()
        self._drain_reservations()
        if (
            self._capacity_restore_tick >= 0
            and self._tick >= self._capacity_restore_tick
        ):
            self._capacity_factor = 1.0
            self._capacity_restore_tick = -1
            self._events.append(
                {
                    "type": "capacity_restored",
                    "event_class": "lifecycle",
                    "origin": "endogenous_completion",
                    "decision_required": False,
                    "actionable": False,
                    "tick": self._tick,
                }
            )
        if (
            self._queue_pressure_restore_tick >= 0
            and self._tick >= self._queue_pressure_restore_tick
        ):
            self._queue_pressure_factor = 1.0
            self._queue_pressure_restore_tick = -1
            self._events.append(
                {
                    "type": "queue_pressure_restored",
                    "event_class": "lifecycle",
                    "origin": "endogenous_completion",
                    "decision_required": False,
                    "actionable": False,
                    "tick": self._tick,
                }
            )
        if (
            self._sla_pressure_restore_tick >= 0
            and self._tick >= self._sla_pressure_restore_tick
        ):
            self._sla_deadline_shift = 0
            self._sla_pressure_restore_tick = -1
            self._events.append(
                {
                    "type": "sla_deadline_pressure_restored",
                    "event_class": "lifecycle",
                    "origin": "endogenous_completion",
                    "decision_required": False,
                    "actionable": False,
                    "tick": self._tick,
                }
            )
        self._apply_perturbations()
        self._reconcile_active_reservations()

        completed_now = 0
        for job in self._jobs.values():
            if job.status != "running":
                continue
            job.remaining_ticks -= 1
            if job.remaining_ticks <= 0:
                job.status = "done"
                job.completion_tick = self._tick
                completed_now += 1
                self._events.append(
                    {
                        "type": "job_completed",
                        "event_class": "lifecycle",
                        "origin": "endogenous_completion",
                        "decision_required": False,
                        "actionable": False,
                        "job_id": job.job_id,
                        "user": job.user,
                        "tick": self._tick,
                    }
                )

        arrived_now = 0
        for job in self._jobs.values():
            if job.status == "future" and job.submit_tick <= self._tick:
                job.status = "queued"
                arrived_now += 1
                self._events.append(
                    {
                        "type": "job_arrival",
                        "event_class": "task",
                        "event_id": (
                            f"datacenter:job_arrival:{job.job_id}:{self._tick}"
                        ),
                        "origin": "source_schedule",
                        "decision_required": self._tick + 1 < self._horizon,
                        "actionable": self._tick + 1 < self._horizon,
                        "job_id": job.job_id,
                        "tick": self._tick,
                        "changed_state_fields": [
                            "queued_jobs",
                            "unfinished_jobs",
                            "sla_risk",
                        ],
                        "materiality_metric": "arrived_jobs",
                        "materiality_value": 1,
                        "materiality_threshold": 1,
                        "materiality_passed": True,
                    }
                )

        self._schedule_queued_jobs()
        for job in self._jobs.values():
            if job.status == "queued":
                job.wait_ticks += 1

        running = [job for job in self._jobs.values() if job.status == "running"]
        queued = [job for job in self._jobs.values() if job.status == "queued"]
        completed = [job for job in self._jobs.values() if job.status == "done"]
        gpu_capacity, cpu_capacity = self._available_capacity()
        gpu_allocated = sum(job.gpu_units for job in running)
        cpu_allocated = sum(job.cpu_units for job in running)
        queue_wait_cost = sum(
            self._queue_pressure_factor * (1.0 + 2.0 * job.criticality)
            for job in queued
        )
        sla_cost = sum(
            5.0 * (1.0 + job.criticality)
            for job in queued
            if self._tick + self._sla_deadline_shift > job.due_tick
        )
        preemption_waste_cost = (
            self._preemption_waste_total - self._recorded_preemption_waste_total
        )
        reserve_capacity_cost = (
            self._reserve_cost_total - self._recorded_reserve_cost_total
        )
        self._recorded_preemption_waste_total = self._preemption_waste_total
        self._recorded_reserve_cost_total = self._reserve_cost_total
        record = DatacenterTickRecord(
            tick=self._tick,
            gpu_capacity=gpu_capacity,
            cpu_capacity=cpu_capacity,
            arrived_jobs=arrived_now,
            queued_jobs=len(queued),
            running_jobs=len(running),
            completed_jobs=len(completed),
            gpu_demand=sum(job.gpu_units for job in queued + running),
            gpu_allocated=gpu_allocated,
            cpu_demand=sum(job.cpu_units for job in queued + running),
            cpu_allocated=cpu_allocated,
            queue_wait_cost=queue_wait_cost,
            compute_cost=gpu_allocated * 0.05,
            sla_violation_cost=sla_cost,
            preemption_waste_cost=preemption_waste_cost,
            reserve_capacity_cost=reserve_capacity_cost,
            realized_events=list(self._events),
            done=len(completed) == len(self._jobs),
        )
        if arrived_now:
            self._source_consumption_ticks.append(self._tick)
            self._post_source_state_digests.append(self._source_state_digest())
        self._records.append(record)
        return record

    def queue_gpu_reservation(
        self,
        *,
        due_tick: int,
        gpu_units: float,
        duration_ticks: int,
    ) -> str:
        requested_units = float(gpu_units)
        if not self._reservation_within_source_limit(requested_units):
            raise ValueError("source_gpu_reservation_limit_exceeded")
        token = self._new_effect_token("reserve_gpu_capacity")
        effect = {
            "token": token,
            "tool_name": "reserve_gpu_capacity",
            "requested_action": {
                "gpu_units": requested_units,
                "duration_ticks": max(1, int(duration_ticks)),
            },
            "before_state_digest": None,
            "changed_state_fields": ["reserved_gpu_units"],
            "call_id": None,
            "evidence_ids": [],
        }
        self._action_effects_by_token[token] = effect
        self._pending_reservations.append(
            {
                "due_tick": int(due_tick),
                "gpu_units": requested_units,
                "duration_ticks": max(1, int(duration_ticks)),
                "effect_token": token,
            }
        )
        return token

    def _reservation_headroom(self, *, at_tick: int | None = None) -> float:
        if self._source_gpu_reservation_limit is None:
            return math.inf
        capacity_factor = self._capacity_factor
        if (
            at_tick is not None
            and self._capacity_restore_tick >= 0
            and int(at_tick) >= self._capacity_restore_tick
        ):
            capacity_factor = 1.0
        pending_units = sum(
            float(reservation["gpu_units"])
            for reservation in self._pending_reservations
        )
        return max(
            0.0,
            self._source_gpu_reservation_limit
            - self._gpu_capacity * capacity_factor
            - self._reserved_gpu
            - pending_units,
        )

    def _reservation_within_source_limit(
        self,
        requested_units: float,
        *,
        at_tick: int | None = None,
    ) -> bool:
        return requested_units <= self._reservation_headroom(at_tick=at_tick) + 1e-9

    def _expire_reservations(self) -> None:
        active: list[tuple[int, float]] = []
        for expiry_tick, units in self._active_reservation_expiries:
            if expiry_tick > self._tick:
                active.append((expiry_tick, units))
                continue
            self._reserved_gpu = max(0.0, self._reserved_gpu - units)
            self._events.append(
                {
                    "type": "gpu_reservation_expired",
                    "event_class": "lifecycle",
                    "origin": "endogenous_completion",
                    "decision_required": False,
                    "actionable": False,
                    "tick": self._tick,
                    "gpu_units": units,
                }
            )
        self._active_reservation_expiries = active

    def _drain_reservations(self) -> None:
        pending: list[dict[str, Any]] = []
        for reservation in self._pending_reservations:
            due_tick = int(reservation["due_tick"])
            units = float(reservation["gpu_units"])
            duration_ticks = int(reservation["duration_ticks"])
            if due_tick > self._tick:
                pending.append(reservation)
                continue
            before_state_digest = self._action_state_digest()
            applied_units = units
            if self._source_gpu_reservation_limit is not None:
                effective_capacity_factor = self._capacity_factor
                if (
                    self._capacity_restore_tick >= 0
                    and self._tick >= self._capacity_restore_tick
                ):
                    effective_capacity_factor = 1.0
                due_headroom = max(
                    0.0,
                    self._source_gpu_reservation_limit
                    - self._gpu_capacity * effective_capacity_factor
                    - self._reserved_gpu,
                )
                if units > due_headroom + 1e-9:
                    applied_units = 0.0
            if applied_units <= 1e-9:
                self._events.append(
                    {
                        "type": "gpu_reservation_capacity_clamped",
                        "event_class": "agent_outcome",
                        "origin": "agent_caused",
                        "decision_required": False,
                        "actionable": False,
                        "tick": self._tick,
                        "requested_gpu_units": units,
                        "applied_gpu_units": 0.0,
                    }
                )
                token = str(reservation["effect_token"])
                effect = self._action_effects_by_token.pop(token, None)
                if effect is not None:
                    effect["before_state_digest"] = before_state_digest
                    effect["applied_action"] = {
                        "gpu_units": 0.0,
                        "duration_ticks": duration_ticks,
                    }
                    effect["changed_state_fields"] = []
                    effect["after_state_digest"] = self._action_state_digest()
                    self._events.append(self._action_effect_event(effect))
                continue
            self._reserved_gpu += applied_units
            expiry_tick = self._tick + duration_ticks
            self._active_reservation_expiries.append((expiry_tick, applied_units))
            cost = applied_units * duration_ticks * RESERVED_GPU_UNIT_TICK_COST
            self._reserve_cost_total += cost
            self._events.append(
                {
                    "type": "gpu_reservation_arrived",
                    "event_class": "agent_outcome",
                    "origin": "agent_caused",
                    "decision_required": False,
                    "actionable": False,
                    "tick": self._tick,
                    "gpu_units": applied_units,
                    "expiry_tick": expiry_tick,
                }
            )
            self._control_events.append(
                {
                    "type": "gpu_reservation_arrived",
                    "tick": self._tick,
                    "gpu_units": applied_units,
                    "physical_actuator_id": "gpu_capacity_pool",
                }
            )
            self._record_response_control(self._tick)
            token = str(reservation["effect_token"])
            effect = self._action_effects_by_token.pop(token, None)
            if effect is not None:
                effect["before_state_digest"] = before_state_digest
                effect["applied_action"] = {
                    "gpu_units": applied_units,
                    "duration_ticks": duration_ticks,
                }
                effect["after_state_digest"] = self._action_state_digest()
                self._events.append(self._action_effect_event(effect))
        self._pending_reservations = pending

    def _reconcile_active_reservations(self) -> None:
        if self._source_gpu_reservation_limit is None:
            return
        maximum_reserved = max(
            0.0,
            self._source_gpu_reservation_limit
            - self._gpu_capacity * self._capacity_factor,
        )
        if self._reserved_gpu <= maximum_reserved + 1e-9:
            return
        remaining = maximum_reserved
        reconciled: list[tuple[int, float]] = []
        for expiry_tick, units in sorted(self._active_reservation_expiries):
            retained = min(units, remaining)
            if retained > 1e-9:
                reconciled.append((expiry_tick, retained))
                remaining -= retained
        trimmed = self._reserved_gpu - maximum_reserved
        self._active_reservation_expiries = reconciled
        self._reserved_gpu = maximum_reserved
        self._events.append(
            {
                "type": "gpu_reservation_capacity_clamped",
                "event_class": "agent_outcome",
                "origin": "agent_caused",
                "decision_required": False,
                "actionable": False,
                "tick": self._tick,
                "requested_gpu_units": self._reserved_gpu + trimmed,
                "applied_gpu_units": self._reserved_gpu,
            }
        )

    def _record_response_control(self, tick: int) -> None:
        """Attach a native control tick to each active event it can address."""
        for event in self._perturbation_events:
            trigger = int(event.get("tick") or 0)
            end = int(event.get("response_window_end_tick") or self._horizon - 1)
            if not trigger < tick <= end:
                continue
            response_ticks = event.setdefault("response_control_ticks", [])
            if tick not in response_ticks:
                response_ticks.append(tick)

    def _available_capacity(self) -> tuple[float, float]:
        gpu_capacity = self._gpu_capacity * self._capacity_factor + self._reserved_gpu
        if self._source_gpu_reservation_limit is not None:
            gpu_capacity = min(gpu_capacity, self._source_gpu_reservation_limit)
        return (gpu_capacity, self._cpu_capacity)

    def _schedule_queued_jobs(self) -> None:
        running = [job for job in self._jobs.values() if job.status == "running"]
        used_gpu = sum(job.gpu_units for job in running)
        used_cpu = sum(job.cpu_units for job in running)
        cap_gpu, cap_cpu = self._available_capacity()
        queued = self._ordered_queued_jobs()
        for job in queued:
            if (
                used_gpu + job.gpu_units <= cap_gpu + 1e-9
                and used_cpu + job.cpu_units <= cap_cpu + 1e-9
            ):
                job.status = "running"
                used_gpu += job.gpu_units
                used_cpu += job.cpu_units
        self._record_policy_review_outcomes()

    def _record_policy_review_outcomes(self) -> None:
        """Record later simulator ticks where the reviewed policy remains active."""
        if not self._policy_review_ledger or not any(
            job.status in {"queued", "running"} for job in self._jobs.values()
        ):
            return
        for entry in self._policy_review_ledger:
            if self._tick <= int(entry["review_tick"]):
                continue
            if int(entry["policy_generation"]) != self._queue_policy_generation:
                continue
            outcome_ticks = entry.setdefault("outcome_effect_ticks", [])
            if self._tick not in outcome_ticks:
                outcome_ticks.append(self._tick)

    def _queue_sort_key(self, job: _Job) -> tuple[Any, ...]:
        """Return the deterministic native scheduler key for one arrived job."""
        if self._queue_policy == "deadline_criticality_first":
            return (
                job.due_tick - self._tick,
                -job.criticality,
                job.remaining_ticks,
                job.submit_tick,
                job.job_id,
            )
        if self._queue_policy == "shortest_job_first":
            return (job.remaining_ticks, job.submit_tick, job.job_id)
        if self._queue_policy == "least_gpu_first":
            return (job.gpu_units, job.submit_tick, job.job_id)
        return (job.submit_tick, job.job_id)

    def _ordered_jobs(self, *, status: str | None = None) -> list[_Job]:
        jobs = [
            job
            for job in self._jobs.values()
            if job.status != "future" and (status is None or job.status == status)
        ]
        return sorted(jobs, key=self._queue_sort_key)

    def _ordered_queued_jobs(self) -> list[_Job]:
        return self._ordered_jobs(status="queued")

    def _queue_priority_rationale(self, job: _Job) -> dict[str, Any]:
        order = {
            "fifo": ["submit_tick", "job_id"],
            "shortest_job_first": [
                "remaining_ticks",
                "submit_tick",
                "job_id",
            ],
            "least_gpu_first": ["gpu_units", "submit_tick", "job_id"],
            "deadline_criticality_first": [
                "due_slack",
                "criticality_desc",
                "remaining_ticks",
                "submit_tick",
                "job_id",
            ],
        }[self._queue_policy]
        return {
            "policy": self._queue_policy,
            "policy_generation": self._queue_policy_generation,
            "order": order,
            "due_tick": job.due_tick,
            "due_slack": job.due_tick - self._tick,
            "criticality": job.criticality,
            "remaining_ticks": job.remaining_ticks,
            "submit_tick": job.submit_tick,
            "job_id": job.job_id,
        }

    def _apply_perturbations(self) -> None:
        assert self._seed_obj is not None
        for ordinal, perturbation in enumerate(self._seed_obj.perturbations):
            if perturbation.trigger_tick != self._tick:
                continue
            event_class = PERTURBATION_EVENT_CLASS.get(perturbation.kind)
            if event_class is None:
                continue
            before_digest = self._source_state_digest()
            duration = max(1, int(perturbation.duration_ticks))
            event_id = (
                f"datacenter:{perturbation.kind}:{self._seed_obj.seed_id}:"
                f"{self._tick}:{ordinal}"
            )
            if perturbation.kind == "capacity_reduction":
                previous_factor = self._capacity_factor
                self._capacity_factor = max(0.1, 1.0 - float(perturbation.intensity))
                self._capacity_restore_tick = self._tick + duration
                changed_state_fields = [
                    "gpu_capacity_units",
                    "capacity_factor",
                ]
                materiality_metric = "capacity_factor_delta"
                materiality_value = abs(previous_factor - self._capacity_factor)
            elif perturbation.kind == "queue_burst":
                previous_factor = self._queue_pressure_factor
                self._queue_pressure_factor = max(
                    1.0, 1.0 + float(perturbation.intensity)
                )
                self._queue_pressure_restore_tick = self._tick + duration
                changed_state_fields = ["queue_pressure_factor"]
                materiality_metric = "queue_pressure_factor_delta"
                materiality_value = abs(previous_factor - self._queue_pressure_factor)
            elif perturbation.kind == "sla_deadline_pressure":
                previous_shift = self._sla_deadline_shift
                self._sla_deadline_shift = max(
                    1, int(round(float(perturbation.intensity)))
                )
                self._sla_pressure_restore_tick = self._tick + duration
                changed_state_fields = ["sla_deadline_shift_ticks"]
                materiality_metric = "sla_deadline_shift_ticks"
                materiality_value = abs(previous_shift - self._sla_deadline_shift)
            else:
                continue
            after_digest = self._source_state_digest()
            has_response_window = self._tick + 1 < self._horizon
            event = {
                "type": perturbation.kind,
                "event_id": event_id,
                "origin": "declared_perturbation",
                "event_class": event_class,
                "declared_perturbation": True,
                "declared_event": {
                    "kind": perturbation.kind,
                    "trigger_tick": int(perturbation.trigger_tick),
                    "duration_ticks": int(perturbation.duration_ticks),
                    "target": dict(perturbation.target or {}),
                    "intensity": float(perturbation.intensity),
                },
                "tick": self._tick,
                "hidden": perturbation.hidden,
                "decision_required": has_response_window,
                "changed_state_fields": changed_state_fields,
                "materiality_metric": materiality_metric,
                "materiality_value": materiality_value,
                "materiality_threshold": 0.01,
                "materiality_passed": materiality_value >= 0.01,
                "response_window_required": True,
                "response_opportunity_tick": (
                    self._tick + 1 if has_response_window else None
                ),
                "response_window_end_tick": min(
                    self._horizon - 1, self._tick + duration
                ),
                "terminal_response_window_missing": not has_response_window,
                "before_state_digest": before_digest,
                "after_state_digest": after_digest,
                "response_control_ticks": [],
            }
            self._perturbation_events.append(event)
            self._events.append(event)

    def apply_tool_effect(
        self,
        name: str,
        args: dict[str, Any],
        *,
        current_tick: int | None = None,
    ) -> dict[str, Any]:
        action_tick = self._tick if current_tick is None else int(current_tick)
        if name == "set_queue_policy":
            policy = str(args.get("policy") or "")
            if policy not in QUEUE_POLICIES:
                return {"_status": "error", "error": "unknown_queue_policy"}
            previous = self._queue_policy
            before_digest = self._action_state_digest()
            self._queue_policy = policy
            if previous != policy:
                self._queue_policy_generation += 1
                self._control_events.append(
                    {
                        "type": "queue_policy_changed",
                        "tick": action_tick,
                        "previous_policy": previous,
                        "queue_policy": policy,
                        "policy_generation": self._queue_policy_generation,
                        "physical_actuator_id": "queue_scheduler",
                    }
                )
                self._record_response_control(action_tick)
                self._strategy_reversal_ticks.add(action_tick)
                token = self._queue_immediate_action_effect(
                    tool_name=name,
                    requested_action=dict(args),
                    applied_action={
                        "queue_policy": policy,
                        "policy_generation": self._queue_policy_generation,
                    },
                    before_state_digest=before_digest,
                    changed_state_fields=[
                        "queue_policy",
                        "queue_policy_generation",
                    ],
                )
            else:
                token = None
            return {
                "previous_policy": previous,
                "queue_policy": policy,
                "policy_generation": self._queue_policy_generation,
                "physical_actuator_id": "queue_scheduler",
                "_backend_effect_token": token,
            }
        if name == "review_persistent_policy":
            return self._review_persistent_policy(args, action_tick)
        if name == "preempt_job":
            job = self._jobs.get(str(args.get("job_id") or ""))
            if job is None or job.status != "running":
                return {"_status": "error", "error": "job_not_running"}
            before_digest = self._action_state_digest()
            elapsed = max(0, job.duration_ticks - job.remaining_ticks)
            waste = float(elapsed) * max(1.0, job.gpu_units)
            self._preemption_waste_total += waste
            job.preemptions += 1
            job.remaining_ticks = job.duration_ticks
            job.status = "queued"
            token = self._queue_immediate_action_effect(
                tool_name=name,
                requested_action=dict(args),
                applied_action={
                    "job_id": job.job_id,
                    "wasted_gpu_ticks": waste,
                },
                before_state_digest=before_digest,
                changed_state_fields=[
                    "running_jobs",
                    "queued_jobs",
                    "preemption_waste",
                ],
            )
            return {
                "job_id": job.job_id,
                "wasted_gpu_ticks": waste,
                "physical_actuator_id": f"running_job:{job.job_id}",
                "_backend_effect_token": token,
            }
        if name == "reserve_gpu_capacity":
            units = max(0.0, float(args.get("gpu_units") or 0.0))
            if units <= 0:
                return {"_status": "error", "error": "gpu_units_must_be_positive"}
            if not self._reservation_within_source_limit(
                units,
                at_tick=action_tick,
            ):
                return {
                    "_status": "error",
                    "error": "source_gpu_reservation_limit_exceeded",
                }
            duration_ticks = max(1, int(args.get("duration_ticks") or 1))
            before_digest = self._action_state_digest()
            self._reserved_gpu += units
            self._active_reservation_expiries.append(
                (action_tick + duration_ticks, units)
            )
            cost = units * duration_ticks * RESERVED_GPU_UNIT_TICK_COST
            self._reserve_cost_total += cost
            self._control_events.append(
                {
                    "type": "gpu_reservation_arrived",
                    "tick": action_tick,
                    "gpu_units": units,
                    "physical_actuator_id": "gpu_capacity_pool",
                }
            )
            self._record_response_control(action_tick)
            token = self._queue_immediate_action_effect(
                tool_name=name,
                requested_action=dict(args),
                applied_action={
                    "gpu_units": units,
                    "duration_ticks": duration_ticks,
                },
                before_state_digest=before_digest,
                changed_state_fields=["reserved_gpu_units"],
            )
            return {
                "reserved_gpu_units": units,
                "reservation_cost": cost,
                "duration_ticks": duration_ticks,
                "physical_actuator_id": "gpu_capacity_pool",
                "_backend_effect_token": token,
            }
        return {"_status": "error", "error": "unknown_tool"}

    def _review_persistent_policy(
        self,
        args: dict[str, Any],
        action_tick: int,
    ) -> dict[str, Any]:
        raw_event_ids = args.get("event_ids")
        if not isinstance(raw_event_ids, (list, tuple)):
            return {"_status": "error", "error": "event_ids_must_be_nonempty"}
        event_ids = [str(event_id).strip() for event_id in raw_event_ids]
        if not event_ids or any(not event_id for event_id in event_ids):
            return {"_status": "error", "error": "event_ids_must_be_nonempty"}
        if len(set(event_ids)) != len(event_ids):
            return {"_status": "error", "error": "duplicate_event_id"}
        try:
            policy_generation = int(args.get("policy_generation"))
        except (TypeError, ValueError):
            return {"_status": "error", "error": "policy_generation_invalid"}
        if policy_generation != self._queue_policy_generation:
            return {"_status": "error", "error": "stale_policy_generation"}
        rationale = str(args.get("rationale") or "").strip()
        if not rationale:
            return {"_status": "error", "error": "review_rationale_required"}

        runtime_events = {
            str(event.get("event_id")): event
            for record in self._records
            for event in record.realized_events
            if event.get("event_id")
        }
        reviewed_events: list[dict[str, Any]] = []
        for event_id in event_ids:
            if event_id in self._reviewed_event_ids:
                return {"_status": "error", "error": "duplicate_review"}
            event = runtime_events.get(event_id)
            if event is None:
                return {"_status": "error", "error": "unknown_runtime_event"}
            if event.get("hidden") is True:
                return {"_status": "error", "error": "event_not_visible"}
            event_tick_value = event.get("tick")
            if event_tick_value is None:
                event_tick_value = event.get("outcome_tick")
            event_tick = int(event_tick_value if event_tick_value is not None else -1)
            if action_tick <= event_tick:
                return {"_status": "error", "error": "review_must_follow_event"}
            reviewed_events.append(event)

        queue = self.queue_state()
        queue_order_digest = _stable_digest(
            {
                "queue_policy": queue["queue_policy"],
                "policy_generation": self._queue_policy_generation,
                "dispatch_order": queue["dispatch_order"],
            }
        )
        policy_digest = _stable_digest(
            {
                "queue_policy": self._queue_policy,
                "policy_generation": self._queue_policy_generation,
            }
        )
        review_id = (
            f"datacenter:policy_review:{action_tick}:"
            f"{self._queue_policy_generation}:{len(self._policy_review_ledger)}"
        )
        ledger_entry = {
            "review_id": review_id,
            "review_tick": action_tick,
            "event_ids": list(event_ids),
            "event_ticks": [
                int(
                    event.get("tick")
                    if event.get("tick") is not None
                    else (
                        event.get("outcome_tick")
                        if event.get("outcome_tick") is not None
                        else -1
                    )
                )
                for event in reviewed_events
            ],
            "policy": self._queue_policy,
            "review_tool_name": "review_persistent_policy",
            "policy_tool_name": "set_queue_policy",
            "policy_generation": self._queue_policy_generation,
            "policy_digest": policy_digest,
            "policy_effect_evidence_id": self._policy_effect_evidence_by_generation.get(
                self._queue_policy_generation, ""
            ),
            "decision": "keep",
            "queue_order_digest": queue_order_digest,
            "dispatch_order": list(queue["dispatch_order"]),
            "outcome_effect_ticks": [],
            "evidence_ids": [],
            "rationale": rationale,
        }
        self._policy_review_ledger.append(ledger_entry)
        self._reviewed_event_ids.update(event_ids)
        return {
            "review_status": "accepted",
            **ledger_entry,
        }

    def _new_effect_token(self, tool_name: str) -> str:
        token = f"{tool_name}:{self._tick}:{self._effect_sequence}"
        self._effect_sequence += 1
        self._unbound_effect_tokens.setdefault(tool_name, []).append(token)
        return token

    def _queue_immediate_action_effect(
        self,
        *,
        tool_name: str,
        requested_action: dict[str, Any],
        applied_action: dict[str, Any],
        before_state_digest: str,
        changed_state_fields: list[str],
    ) -> str:
        token = self._new_effect_token(tool_name)
        effect = {
            "token": token,
            "tool_name": tool_name,
            "requested_action": requested_action,
            "applied_action": applied_action,
            "before_state_digest": before_state_digest,
            "after_state_digest": self._action_state_digest(),
            "changed_state_fields": changed_state_fields,
            "call_id": None,
            "evidence_ids": [],
        }
        self._action_effects_by_token[token] = effect
        self._pending_action_effects.append(effect)
        return token

    def bind_tool_result(
        self,
        *,
        name: str,
        call_id: str | None,
        evidence_id: str | None,
        payload: dict[str, Any],
        causal_parent_event_id: str | None = None,
    ) -> None:
        if name == "review_persistent_policy":
            review_id = str(payload.get("review_id") or "")
            for entry in self._policy_review_ledger:
                if entry.get("review_id") != review_id:
                    continue
                if evidence_id and evidence_id not in entry["evidence_ids"]:
                    entry["evidence_ids"].append(evidence_id)
                if evidence_id and evidence_id not in payload.get("evidence_ids", []):
                    payload.setdefault("evidence_ids", []).append(evidence_id)
                return
        token = str(payload.get("_backend_effect_token") or "")
        if not token:
            unbound = self._unbound_effect_tokens.get(name) or []
            token = unbound[0] if unbound else ""
        effect = self._action_effects_by_token.get(token)
        if effect is None:
            return
        effect["call_id"] = call_id
        if causal_parent_event_id:
            effect["causal_parent_event_id"] = causal_parent_event_id
        else:
            effect.pop("causal_parent_event_id", None)
        if evidence_id and evidence_id not in effect["evidence_ids"]:
            effect["evidence_ids"].append(evidence_id)
        if name == "set_queue_policy" and evidence_id:
            generation = int(
                (effect.get("applied_action") or {}).get("policy_generation", 0) or 0
            )
            if generation > 0:
                self._policy_effect_evidence_by_generation[generation] = evidence_id
        unbound = self._unbound_effect_tokens.get(name) or []
        if token in unbound:
            unbound.remove(token)

    def _action_effect_event(self, effect: dict[str, Any]) -> dict[str, Any]:
        event_id = f"{effect['token']}@{self._tick}"
        call_id = str(effect.get("call_id") or "")
        return {
            "type": "control_effect",
            "event_id": event_id,
            "origin": "agent_caused",
            "agent_caused": True,
            "tool_name": effect["tool_name"],
            "call_id": call_id,
            "requested_action": dict(effect["requested_action"]),
            "applied_action": dict(effect["applied_action"]),
            "before_state_digest": effect["before_state_digest"],
            "after_state_digest": effect["after_state_digest"],
            "changed_state_fields": list(effect["changed_state_fields"]),
            "outcome_tick": self._tick,
            "evidence_ids": list(effect["evidence_ids"]),
            **(
                {"causal_parent_event_id": effect["causal_parent_event_id"]}
                if effect.get("causal_parent_event_id")
                else {}
            ),
            "action_to_outcome_edge": {
                "source": f"call:{call_id}",
                "target": f"outcome:{event_id}",
                "kind": "action_to_outcome",
            },
        }

    def _action_state_digest(self) -> str:
        return _stable_digest(
            {
                "queue_policy": self._queue_policy,
                "queue_policy_generation": self._queue_policy_generation,
                "reserved_gpu_units": self._reserved_gpu,
                "jobs": {
                    job_id: {
                        "status": job.status,
                        "remaining_ticks": job.remaining_ticks,
                        "preemptions": job.preemptions,
                    }
                    for job_id, job in sorted(self._jobs.items())
                },
            }
        )

    def _source_state_digest(self) -> str:
        queued = sum(job.status == "queued" for job in self._jobs.values())
        running = sum(job.status == "running" for job in self._jobs.values())
        unfinished = sum(job.status != "done" for job in self._jobs.values())
        gpu, _cpu = self._available_capacity()
        return _stable_digest(
            {
                "queued_jobs": queued,
                "running_jobs": running,
                "available_gpu_units": gpu,
                "capacity_factor": self._capacity_factor,
                "queue_pressure_factor": self._queue_pressure_factor,
                "sla_deadline_shift": self._sla_deadline_shift,
                "unfinished_jobs": unfinished,
                "sla_risk": sum(
                    job.status == "queued" and self._tick > job.due_tick
                    for job in self._jobs.values()
                ),
            }
        )

    def protocol21_source_trace(self) -> dict[str, Any]:
        if self._source_window is None:
            return {
                "status": "held",
                "proof_kind": "derived_source_window",
                "evidence_from_scenario_config_only": True,
                "runtime_trace_observed": False,
                "blockers": ["source_contract_window_metadata_missing"],
            }
        window = self._source_window
        runtime_assets = list(window.get("runtime_opened_assets") or [])
        source_hashes = {
            str(asset["source_path"]): str(asset["sha256"]) for asset in runtime_assets
        }
        opened_hashes = {
            str(asset["resolved_source_path"]): str(asset["sha256"])
            for asset in runtime_assets
        }
        trace_payload = {
            "window_sha256": window["sha256"],
            "source_hashes": source_hashes,
            "consumption_ticks": self._source_consumption_ticks,
            "post_source_state_digests": self._post_source_state_digests,
        }
        return {
            "status": "passed",
            "proof_kind": "derived_source_window",
            "runtime_opened_assets": [
                {
                    "path": asset["resolved_source_path"],
                    "sha256": asset["sha256"],
                    "role": asset["role"],
                }
                for asset in runtime_assets
            ],
            "opened_source_paths": list(opened_hashes),
            "opened_source_sha256": opened_hashes,
            "locked_derivation_source_hashes": source_hashes,
            "consumed_source_hashes": source_hashes,
            "lineage_source_hashes": source_hashes,
            "source_window": {
                "recipe_version": window["recipe_version"],
                "row_start": window["row_start"],
                "row_end": window["row_end"],
                "source_window_sha256": window["sha256"],
                "runtime_window_digest": window["runtime_window_digest"],
                **(
                    {
                        "source_row_indices": list(window["source_row_indices"]),
                        "source_row_ordering": window["source_row_ordering"],
                        "selected_rows_sha256": window["selected_rows_sha256"],
                    }
                    if window.get("source_row_indices") is not None
                    else {}
                ),
                **(
                    {
                        "node_selection_recipe": window["node_selection_recipe"],
                        "selected_node_inventory_sha256": window[
                            "selected_node_inventory_sha256"
                        ],
                    }
                    if window.get("node_selection_recipe")
                    else {
                        "job_id_start": window["job_id_start"],
                        "job_id_end": window["job_id_end"],
                    }
                ),
            },
            "consumed_window_sha256": window["sha256"],
            "runtime_job_window_digest": window["runtime_window_digest"],
            "recipe_version": window["recipe_version"],
            "consumed_channels": list(window["consumed_channels"]),
            "derived_backend_state_fields": [
                "available_gpu_units",
                "queued_jobs",
                "running_jobs",
                "sla_risk",
                "unfinished_jobs",
            ],
            "consumption_ticks": list(self._source_consumption_ticks),
            "initial_state_digest": self._initial_state_digest,
            "post_source_state_digests": list(self._post_source_state_digests),
            "state_effect_observed": bool(self._source_consumption_ticks),
            "source_state_effect_observed": bool(self._source_consumption_ticks),
            "deterministic_source_trace": True,
            "trace_semantic_digest": _stable_digest(trace_payload),
            "runtime_trace_observed": True,
            "evidence_from_scenario_config_only": False,
            "blockers": [],
        }

    def queue_state(self) -> dict[str, Any]:
        arrived = self._ordered_jobs()
        queued = [job for job in arrived if job.status == "queued"]
        dispatch_ranks = {
            job.job_id: index for index, job in enumerate(arrived, start=1)
        }
        return {
            "queue_policy": self._queue_policy,
            "policy_generation": self._queue_policy_generation,
            "dispatch_order": [job.job_id for job in arrived][:20],
            "running_job_ids": [
                job.job_id for job in arrived if job.status == "running"
            ],
            "dispatch_rationale": {
                job.job_id: self._queue_priority_rationale(job) for job in arrived[:20]
            },
            "queued_jobs": [
                {
                    "job_id": job.job_id,
                    "user": job.user,
                    "remaining_ticks": job.remaining_ticks,
                    "gpu_units": job.gpu_units,
                    "criticality": job.criticality,
                    "due_tick": job.due_tick,
                    "dispatch_order": dispatch_ranks[job.job_id],
                    "priority_rationale": self._queue_priority_rationale(job),
                }
                for job in queued[:20]
            ],
        }

    def capacity_state(self) -> dict[str, Any]:
        gpu, cpu = self._available_capacity()
        running = [job for job in self._jobs.values() if job.status == "running"]
        return {
            "gpu_capacity_units": gpu,
            "cpu_capacity_units": cpu,
            "gpu_allocated_units": sum(job.gpu_units for job in running),
            "cpu_allocated_units": sum(job.cpu_units for job in running),
            "capacity_factor": self._capacity_factor,
            "reserved_gpu_units": self._reserved_gpu,
            "pending_reserved_gpu_units": sum(
                float(item["gpu_units"]) for item in self._pending_reservations
            ),
        }

    def arrival_forecast(self, horizon_ticks: int) -> dict[str, Any]:
        end_tick = min(self._horizon, self._tick + max(1, int(horizon_ticks)) + 1)
        arrivals = [
            job
            for job in self._jobs.values()
            if job.status == "future" and self._tick < job.submit_tick < end_tick
        ]
        return {
            "from_tick": self._tick + 1,
            "to_tick": end_tick - 1,
            "expected_job_count": len(arrivals),
            "expected_gpu_units": sum(job.gpu_units for job in arrivals),
            "expected_cpu_units": sum(job.cpu_units for job in arrivals),
            "source": "locked_trace_future_arrivals",
        }

    def snapshot(self) -> dict[str, Any]:
        queue = self.queue_state()
        dispatch_ranks = {
            job_id: index
            for index, job_id in enumerate(queue["dispatch_order"], start=1)
        }
        return {
            "tick": self._tick,
            "decision_opportunity": bool(queue["queued_jobs"]),
            "queue": queue,
            "capacity": self.capacity_state(),
            "jobs": {
                job.job_id: {
                    "kind": "gpu_job",
                    "user": job.user,
                    "status": job.status,
                    "submit_tick": job.submit_tick,
                    "remaining_ticks": job.remaining_ticks,
                    "gpu_units": job.gpu_units,
                    "cpu_units": job.cpu_units,
                    "criticality": job.criticality,
                    "due_tick": job.due_tick,
                    "wait_ticks": job.wait_ticks,
                    "preemptions": job.preemptions,
                    "dispatch_order": dispatch_ranks.get(job.job_id),
                    "priority_rationale": queue["dispatch_rationale"].get(job.job_id),
                }
                for job in self._jobs.values()
                if job.status != "future"
            },
        }

    def ground_truth_costs(self) -> dict[str, float]:
        unfinished = sum(
            job.remaining_ticks * (1.0 + job.criticality) * 5.0
            for job in self._jobs.values()
            if job.status != "done"
        )
        return {
            "compute_cost": round(sum(row.compute_cost for row in self._records), 6),
            "queue_wait_cost": round(
                sum(row.queue_wait_cost for row in self._records), 6
            ),
            "sla_violation_cost": round(
                sum(row.sla_violation_cost for row in self._records), 6
            ),
            "unfinished_work_penalty": round(unfinished, 6),
            "preemption_waste_cost": round(self._preemption_waste_total, 6),
            "reserve_capacity_cost": round(self._reserve_cost_total, 6),
        }

    def control_summary(self) -> dict[str, Any]:
        event_tools = {
            "queue_policy_changed": "set_queue_policy",
            "gpu_reservation_arrived": "reserve_gpu_capacity",
            "job_preempted": "preempt_job",
        }
        physical_tools = sorted(
            {
                event_tools.get(str(event.get("type") or ""), "")
                for event in self._control_events
                if event_tools.get(str(event.get("type") or ""), "")
            }
        )
        tool_ticks = {
            tool: sorted(
                {
                    int(event.get("tick") or 0)
                    for event in self._control_events
                    if event_tools.get(str(event.get("type") or "")) == tool
                }
            )
            for tool in physical_tools
        }
        endpoint_map = {
            "set_queue_policy": "queue_scheduler",
            "reserve_gpu_capacity": "gpu_capacity_pool",
            "preempt_job": "running_job_pool",
        }
        return {
            "queue_policy_changes": sum(
                event["type"] == "queue_policy_changed"
                for event in self._control_events
            ),
            "reservation_arrivals": sum(
                event["type"] == "gpu_reservation_arrived"
                for event in self._control_events
            ),
            "policy_review_count": len(self._policy_review_ledger),
            "policy_review_ledger": [
                dict(entry) for entry in self._policy_review_ledger
            ],
            "distinct_control_ticks": sorted(
                {int(event["tick"]) for event in self._control_events}
            ),
            "tool_ticks": tool_ticks,
            "distinct_physical_tools": physical_tools,
            "distinct_physical_actuator_endpoints": sorted(
                endpoint_map[name] for name in physical_tools
            ),
            "strategy_reversal_count": len(self._strategy_reversal_ticks),
            "strategy_reversal_ticks": sorted(self._strategy_reversal_ticks),
            "response_windows": [
                {
                    "event_id": str(event.get("event_id") or ""),
                    "event_type": str(event.get("type") or ""),
                    "trigger_tick": int(event.get("tick") or 0),
                    "opportunity_tick": event.get("response_opportunity_tick"),
                    "end_tick": event.get("response_window_end_tick"),
                    "control_ticks": sorted(
                        int(tick) for tick in event.get("response_control_ticks") or []
                    ),
                }
                for event in self._perturbation_events
            ],
        }

    def per_job_sla_violation_minutes(self) -> dict[str, float]:
        tick_minutes = int(self._seed_obj.tick_minutes if self._seed_obj else 1)
        return {
            job.job_id: float(
                max(0, job.wait_ticks - max(0, job.due_tick - job.submit_tick))
                * tick_minutes
            )
            for job in self._jobs.values()
        }

    def scoring_records(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for record in self._records:
            capacity_excess = max(
                0.0,
                record.gpu_allocated / max(1.0, record.gpu_capacity) - 1.0,
            )
            queued_sla_exposure = max(
                0.0,
                record.queue_wait_cost * 5.0,
            )
            sla_breach_fraction = max(
                0.0,
                record.sla_violation_cost / max(1.0, queued_sla_exposure),
            )
            rows.append(
                {
                    "tick": record.tick,
                    "aggregate_demand_mw": record.gpu_demand,
                    "aggregate_generation_mw": record.gpu_allocated,
                    "balance_error_mw": max(
                        0.0, record.gpu_demand - record.gpu_allocated
                    ),
                    "reserves_required_mw": record.gpu_demand,
                    "reserves_procured_mw": record.gpu_capacity,
                    "production_cost": record.compute_cost,
                    "startup_cost": record.reserve_capacity_cost,
                    "shed_penalty": (
                        record.queue_wait_cost + record.sla_violation_cost
                    ),
                    "rho_max": (record.gpu_allocated / max(1.0, record.gpu_capacity)),
                    "n_overloads": 0,
                    "n_voltage_violations": 0,
                    "n_disconnected_lines": 0,
                    # Canonical ``done`` means catastrophic early termination.
                    # Finishing all queued work early is a successful terminal
                    # state, so it must never trigger the survival penalty.
                    "done": False,
                    "catastrophic_failure": False,
                    "safety_violation_severity": min(
                        1.0, max(sla_breach_fraction, capacity_excess)
                    ),
                }
            )
        return rows
