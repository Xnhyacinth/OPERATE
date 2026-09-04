"""Deterministic OpenB GPU-sharing placement backend.

The backend consumes exact rows from Alibaba's v2023 node and pod traces.  It
models CPU, memory, physical GPU type, per-GPU milli-share fragmentation, QoS,
arrivals and lifetimes without translating them into generic queue jobs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..seeds.schema import DatacenterScenarioSeed
from .alibaba_trace_backend import DatacenterTickRecord


OPENB_SOURCE_SCHEMA = "alibaba-openb-v2023-v1"
OPENB_RECIPE = "alibaba-openb-explicit-row-graph-v1"
PLACEMENT_POLICIES = ("first_fit", "fragmentation_aware")
REPO_ROOT = Path(__file__).resolve().parents[3]
NODE_FIELDS = {"sn", "cpu_milli", "memory_mib", "gpu", "model"}
POD_FIELDS = {
    "name",
    "cpu_milli",
    "memory_mib",
    "num_gpu",
    "gpu_milli",
    "gpu_spec",
    "qos",
    "pod_phase",
    "creation_time",
    "deletion_time",
    "scheduled_time",
}
QOS_CRITICALITY = {"LS": 0.9, "Guaranteed": 0.8, "Burstable": 0.5, "BE": 0.2}


def _stable_digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def _finite_float(value: Any, *, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"openb_{field}_invalid") from exc
    if not math.isfinite(converted):
        raise ValueError(f"openb_{field}_invalid")
    return converted


def _normalize_node(row: dict[str, Any]) -> dict[str, Any]:
    node_id = str(row.get("sn") or "")
    cpu = _finite_float(row.get("cpu_milli"), field="node_cpu_milli")
    memory = _finite_float(row.get("memory_mib"), field="node_memory_mib")
    gpu_count = int(_finite_float(row.get("gpu"), field="node_gpu_count"))
    model = str(row.get("model") or "")
    if not node_id or cpu <= 0 or memory <= 0 or gpu_count < 0:
        raise ValueError("openb_node_schema_invalid")
    if gpu_count and not model:
        raise ValueError("openb_node_gpu_model_missing")
    return {
        "node_id": node_id,
        "cpu_milli": cpu,
        "memory_mib": memory,
        "gpu_count": gpu_count,
        "gpu_model": model,
    }


def _normalize_pod(row: dict[str, Any]) -> dict[str, Any]:
    pod_id = str(row.get("name") or "")
    cpu = _finite_float(row.get("cpu_milli"), field="pod_cpu_milli")
    memory = _finite_float(row.get("memory_mib"), field="pod_memory_mib")
    gpu_count = int(_finite_float(row.get("num_gpu"), field="pod_gpu_count"))
    gpu_milli = _finite_float(row.get("gpu_milli"), field="pod_gpu_milli")
    creation = _finite_float(row.get("creation_time"), field="pod_creation_time")
    deletion = _finite_float(row.get("deletion_time"), field="pod_deletion_time")
    scheduled_raw = row.get("scheduled_time")
    scheduled = (
        None
        if scheduled_raw in (None, "")
        else _finite_float(scheduled_raw, field="pod_scheduled_time")
    )
    if (
        not pod_id
        or cpu < 0
        or memory < 0
        or gpu_count < 0
        or not 0 <= gpu_milli <= 1000
        or deletion < creation
        or (scheduled is not None and scheduled < creation)
        or (scheduled is not None and scheduled > deletion)
    ):
        raise ValueError("openb_pod_schema_invalid")
    if gpu_count == 0 and gpu_milli != 0:
        raise ValueError("openb_cpu_pod_gpu_share_invalid")
    return {
        "pod_id": pod_id,
        "cpu_milli": cpu,
        "memory_mib": memory,
        "gpu_count": gpu_count,
        "gpu_milli_per_gpu": gpu_milli,
        "compatible_gpu_models": sorted(
            {item for item in str(row.get("gpu_spec") or "").split("|") if item}
        ),
        "qos": str(row.get("qos") or "unknown"),
        "source_phase": str(row.get("pod_phase") or "unknown"),
        "creation_time": creation,
        "scheduled_time": scheduled,
        "deletion_time": deletion,
    }


def _read_selected_rows(
    path: Path,
    *,
    required_fields: set[str],
    indices: list[int],
    normalizer: Any,
) -> list[dict[str, Any]]:
    selected = set(indices)
    output: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required_fields - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"openb_source_fields_missing:{','.join(sorted(missing))}")
        for index, row in enumerate(reader):
            if index in selected:
                output.append(normalizer(row))
            if index > indices[-1]:
                break
    if len(output) != len(indices):
        raise ValueError("openb_selected_rows_missing")
    return output


def _validated_indices(value: Any, *, kind: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"openb_{kind}_row_indices_invalid")
    indices = list(value)
    if indices != sorted(set(indices)) or indices[0] < 0:
        raise ValueError(f"openb_{kind}_row_indices_invalid")
    return indices


def _resolve_asset(
    path: str, expected_sha256: str, repo_root: Path
) -> tuple[Path, str]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise ValueError("source_file_missing")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if not expected_sha256 or digest != expected_sha256.removeprefix("sha256:"):
        raise ValueError("source_hash_mismatch")
    return resolved, digest


def resolve_openb_source_graph(
    *,
    provenance_files: list[str],
    source_transform: dict[str, Any],
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Resolve and verify an exact node/pod subgraph from locked source bytes."""
    if source_transform.get("source_schema") != OPENB_SOURCE_SCHEMA:
        raise ValueError("openb_source_schema_invalid")
    if source_transform.get("recipe_version") != OPENB_RECIPE:
        raise ValueError("openb_source_recipe_invalid")
    roles = dict(source_transform.get("source_file_roles") or {})
    node_source = str(roles.get("node_inventory") or "")
    pod_source = str(roles.get("pod_trace") or "")
    if set(provenance_files) != {node_source, pod_source} or len(provenance_files) != 2:
        raise ValueError("openb_source_file_roles_invalid")
    node_path, node_sha = _resolve_asset(
        node_source, str(source_transform.get("node_trace_sha256") or ""), repo_root
    )
    pod_path, pod_sha = _resolve_asset(
        pod_source, str(source_transform.get("pod_trace_sha256") or ""), repo_root
    )
    node_indices = _validated_indices(
        source_transform.get("node_row_indices"), kind="node"
    )
    pod_indices = _validated_indices(
        source_transform.get("pod_row_indices"), kind="pod"
    )
    nodes = _read_selected_rows(
        node_path,
        required_fields=NODE_FIELDS,
        indices=node_indices,
        normalizer=_normalize_node,
    )
    pods = _read_selected_rows(
        pod_path,
        required_fields=POD_FIELDS,
        indices=pod_indices,
        normalizer=_normalize_pod,
    )
    node_digest = _stable_digest(nodes)
    pod_digest = _stable_digest(pods)
    graph_digest = _stable_digest({"nodes": nodes, "pods": pods})
    expected = {
        "selected_nodes_sha256": node_digest,
        "selected_pods_sha256": pod_digest,
        "source_graph_sha256": graph_digest,
    }
    if any(
        str(source_transform.get(key) or "").removeprefix("sha256:") != value
        for key, value in expected.items()
    ):
        raise ValueError("openb_source_graph_hash_mismatch")
    return {
        "recipe_version": OPENB_RECIPE,
        "nodes": nodes,
        "pods": pods,
        "node_row_indices": node_indices,
        "pod_row_indices": pod_indices,
        "selected_nodes_sha256": node_digest,
        "selected_pods_sha256": pod_digest,
        "source_graph_sha256": graph_digest,
        "runtime_opened_assets": [
            {
                "path": str(node_path),
                "source_path": node_source,
                "sha256": node_sha,
                "role": "node_inventory",
            },
            {
                "path": str(pod_path),
                "source_path": pod_source,
                "sha256": pod_sha,
                "role": "pod_trace",
            },
        ],
        "consumed_channels": [
            "node_cpu_capacity",
            "node_gpu_count",
            "node_gpu_model",
            "node_memory_capacity",
            "pod_arrival_time",
            "pod_cpu_request",
            "pod_gpu_count",
            "pod_gpu_model_constraint",
            "pod_gpu_share",
            "pod_lifetime",
            "pod_memory_request",
            "pod_qos",
        ],
    }


@dataclass
class _Node:
    node_id: str
    cpu_milli: float
    memory_mib: float
    gpu_model: str
    gpu_slots: tuple[float, ...]


@dataclass
class _Pod:
    pod_id: str
    cpu_milli: float
    memory_mib: float
    gpu_count: int
    gpu_milli_per_gpu: float
    compatible_gpu_models: tuple[str, ...]
    qos: str
    submit_tick: int
    duration_ticks: int
    due_tick: int
    status: str = "future"
    node_id: str | None = None
    gpu_slot_indices: tuple[int, ...] = ()
    wait_ticks: int = 0
    remaining_ticks: int = 1
    migrations: int = 0


class AlibabaOpenBBackend:
    """Executable multi-resource placement kernel over OpenB source rows."""

    backend_kind = "alibaba_openb_gpu_placement"

    def __init__(self) -> None:
        self._seed_obj: DatacenterScenarioSeed | None = None
        self._tick = 0
        self._horizon = 8
        self._nodes: dict[str, _Node] = {}
        self._pods: dict[str, _Pod] = {}
        self._source_graph: dict[str, Any] | None = None
        self._placement_policy = "first_fit"
        self._autonomous = True
        self._events: list[dict[str, Any]] = []
        self._records: list[DatacenterTickRecord] = []
        self._control_events: list[dict[str, Any]] = []
        self._pending_effects: list[dict[str, Any]] = []
        self._effects: dict[str, dict[str, Any]] = {}
        self._effect_sequence = 0
        self._source_consumption_ticks: list[int] = []
        self._post_source_state_digests: list[str] = []
        self._initial_state_digest = ""
        self._migration_cost_total = 0.0

    def reset(self, seed_obj: DatacenterScenarioSeed) -> None:
        self._seed_obj = seed_obj
        self._tick = 0
        self._horizon = int(seed_obj.horizon_ticks)
        self._placement_policy = str(
            seed_obj.backend_config.get("initial_placement_policy") or "first_fit"
        )
        if self._placement_policy not in PLACEMENT_POLICIES:
            raise ValueError("unknown_placement_policy")
        self._autonomous = bool(
            seed_obj.backend_config.get("autonomous_placement", True)
        )
        source_transform = dict(seed_obj.backend_config.get("source_transform") or {})
        self._source_graph = resolve_openb_source_graph(
            provenance_files=list(seed_obj.provenance.files),
            source_transform=source_transform,
        )
        self._nodes = {
            row["node_id"]: _Node(
                node_id=str(row["node_id"]),
                cpu_milli=float(row["cpu_milli"]),
                memory_mib=float(row["memory_mib"]),
                gpu_model=str(row["gpu_model"]),
                gpu_slots=tuple(1000.0 for _ in range(int(row["gpu_count"]))),
            )
            for row in self._source_graph["nodes"]
        }
        if not self._nodes:
            raise ValueError("openb_selected_nodes_empty")
        source_pods = list(self._source_graph["pods"])
        starts = [float(row["creation_time"]) for row in source_pods]
        minimum = min(starts)
        span = max(1.0, max(starts) - minimum)
        tick_seconds = max(1.0, float(seed_obj.tick_minutes) * 60.0)
        self._pods = {}
        for row in source_pods:
            submit_tick = min(
                self._horizon - 2,
                max(
                    0,
                    int(
                        round(
                            (float(row["creation_time"]) - minimum)
                            / span
                            * (self._horizon - 2)
                        )
                    ),
                ),
            )
            scheduled_time = row["scheduled_time"]
            execution_start_time = (
                float(row["creation_time"])
                if scheduled_time is None
                else float(scheduled_time)
            )
            duration = max(
                1,
                int(
                    math.ceil(
                        (float(row["deletion_time"]) - execution_start_time)
                        / tick_seconds
                    )
                ),
            )
            source_delay = (
                self._horizon - 1 - submit_tick
                if scheduled_time is None
                else max(
                    0,
                    int(
                        math.ceil(
                            (float(scheduled_time) - float(row["creation_time"]))
                            / tick_seconds
                        )
                    ),
                )
            )
            pod = _Pod(
                pod_id=str(row["pod_id"]),
                cpu_milli=float(row["cpu_milli"]),
                memory_mib=float(row["memory_mib"]),
                gpu_count=int(row["gpu_count"]),
                gpu_milli_per_gpu=float(row["gpu_milli_per_gpu"]),
                compatible_gpu_models=tuple(row["compatible_gpu_models"]),
                qos=str(row["qos"]),
                submit_tick=submit_tick,
                duration_ticks=duration,
                due_tick=min(self._horizon - 1, submit_tick + source_delay),
                remaining_ticks=duration,
            )
            self._pods[pod.pod_id] = pod
        self._events = []
        self._records = []
        self._control_events = []
        self._pending_effects = []
        self._effects = {}
        self._effect_sequence = 0
        self._source_consumption_ticks = []
        self._post_source_state_digests = []
        self._migration_cost_total = 0.0
        self._initial_state_digest = self._source_state_digest()

    def tenant_ids(self) -> tuple[str, ...]:
        return tuple(sorted({pod.qos for pod in self._pods.values()}))

    def _node_usage(
        self, node_id: str, *, exclude_pod: str | None = None
    ) -> dict[str, Any]:
        cpu = memory = 0.0
        gpu = [0.0 for _ in self._nodes[node_id].gpu_slots]
        for pod in self._pods.values():
            if (
                pod.node_id != node_id
                or pod.pod_id == exclude_pod
                or pod.status != "running"
            ):
                continue
            cpu += pod.cpu_milli
            memory += pod.memory_mib
            for slot in pod.gpu_slot_indices:
                gpu[slot] += pod.gpu_milli_per_gpu
        return {"cpu_milli": cpu, "memory_mib": memory, "gpu_milli": gpu}

    def _compatible_node_ids(self, pod: _Pod) -> list[str]:
        return [
            node.node_id
            for node in self._nodes.values()
            if not pod.compatible_gpu_models
            or node.gpu_model in pod.compatible_gpu_models
        ]

    def _slot_plan(self, pod: _Pod, node_id: str) -> tuple[int, ...] | None:
        node = self._nodes.get(node_id)
        if node is None or (
            pod.compatible_gpu_models
            and node.gpu_model not in pod.compatible_gpu_models
        ):
            return None
        usage = self._node_usage(node_id, exclude_pod=pod.pod_id)
        if (
            usage["cpu_milli"] + pod.cpu_milli > node.cpu_milli + 1e-9
            or usage["memory_mib"] + pod.memory_mib > node.memory_mib + 1e-9
        ):
            return None
        available = [1000.0 - used for used in usage["gpu_milli"]]
        eligible = [
            index
            for index, remaining in enumerate(available)
            if remaining + 1e-9 >= pod.gpu_milli_per_gpu
        ]
        if len(eligible) < pod.gpu_count:
            return None
        if self._placement_policy == "fragmentation_aware":
            eligible.sort(
                key=lambda index: (available[index] - pod.gpu_milli_per_gpu, index)
            )
        return tuple(eligible[: pod.gpu_count])

    def _place(self, pod: _Pod, node_id: str, *, migrated: bool = False) -> bool:
        plan = self._slot_plan(pod, node_id)
        if plan is None:
            return False
        pod.node_id = node_id
        pod.gpu_slot_indices = plan
        pod.status = "running"
        if migrated:
            pod.migrations += 1
        return True

    def _node_order(self, pod: _Pod) -> list[str]:
        candidates = self._compatible_node_ids(pod)
        if self._placement_policy == "first_fit":
            return sorted(candidates)

        def score(node_id: str) -> tuple[float, str]:
            node = self._nodes[node_id]
            usage = self._node_usage(node_id)
            remaining = sum(1000.0 - value for value in usage["gpu_milli"])
            return (remaining, node.node_id)

        return sorted(candidates, key=score)

    def _feasible_node_ids(self, pod: _Pod) -> list[str]:
        return [
            node_id
            for node_id in self._node_order(pod)
            if self._slot_plan(pod, node_id) is not None
        ]

    def _autoplace(self) -> None:
        if not self._autonomous:
            return
        queued = sorted(
            (pod for pod in self._pods.values() if pod.status == "queued"),
            key=lambda pod: (
                -QOS_CRITICALITY.get(pod.qos, 0.3),
                pod.submit_tick,
                pod.pod_id,
            ),
        )
        for pod in queued:
            for node_id in self._node_order(pod):
                if self._place(pod, node_id):
                    break

    def tick(self, current_tick: int) -> DatacenterTickRecord:
        self._tick = int(current_tick)
        self._events = [self._effect_event(effect) for effect in self._pending_effects]
        self._pending_effects = []
        for pod in self._pods.values():
            if pod.status == "running":
                pod.remaining_ticks -= 1
                if pod.remaining_ticks <= 0:
                    pod.status = "done"
                    pod.node_id = None
                    pod.gpu_slot_indices = ()
                    self._events.append(
                        {
                            "type": "pod_completed",
                            "origin": "endogenous_completion",
                            "tick": self._tick,
                            "pod_id": pod.pod_id,
                            "user": pod.qos,
                            "actionable": False,
                            "decision_required": False,
                        }
                    )
        arrived = 0
        for pod in self._pods.values():
            if pod.status == "future" and pod.submit_tick <= self._tick:
                pod.status = "queued"
                arrived += 1
                self._events.append(
                    {
                        "type": "pod_arrival",
                        "event_class": "task",
                        "event_id": f"datacenter:openb:pod_arrival:{pod.pod_id}:{self._tick}",
                        "origin": "source_schedule",
                        "tick": self._tick,
                        "pod_id": pod.pod_id,
                        "changed_state_fields": [
                            "queued_pods",
                            "placement_fragmentation",
                            "qos_delay_risk",
                        ],
                        "materiality_metric": "arrived_pods",
                        "materiality_value": 1,
                        "materiality_threshold": 1,
                        "materiality_passed": True,
                        "decision_required": self._tick + 1 < self._horizon,
                        "actionable": self._tick + 1 < self._horizon,
                    }
                )
        self._autoplace()
        for pod in self._pods.values():
            if pod.status == "queued":
                pod.wait_ticks += 1
        queued = [pod for pod in self._pods.values() if pod.status == "queued"]
        running = [pod for pod in self._pods.values() if pod.status == "running"]
        done = [pod for pod in self._pods.values() if pod.status == "done"]
        gpu_capacity = sum(len(node.gpu_slots) for node in self._nodes.values())
        gpu_allocated = sum(
            pod.gpu_count * pod.gpu_milli_per_gpu / 1000.0 for pod in running
        )
        queue_cost = sum(
            1.0 + 3.0 * QOS_CRITICALITY.get(pod.qos, 0.3) for pod in queued
        )
        sla_cost = sum(
            8.0 * (1.0 + QOS_CRITICALITY.get(pod.qos, 0.3))
            for pod in queued
            if self._tick > pod.due_tick
        )
        record = DatacenterTickRecord(
            tick=self._tick,
            gpu_capacity=float(gpu_capacity),
            cpu_capacity=sum(node.cpu_milli for node in self._nodes.values()),
            arrived_jobs=arrived,
            queued_jobs=len(queued),
            running_jobs=len(running),
            completed_jobs=len(done),
            gpu_demand=sum(
                pod.gpu_count * pod.gpu_milli_per_gpu / 1000.0
                for pod in queued + running
            ),
            gpu_allocated=gpu_allocated,
            cpu_demand=sum(pod.cpu_milli for pod in queued + running),
            cpu_allocated=sum(pod.cpu_milli for pod in running),
            queue_wait_cost=queue_cost,
            compute_cost=gpu_allocated * 0.05,
            sla_violation_cost=sla_cost,
            preemption_waste_cost=0.0,
            reserve_capacity_cost=0.0,
            realized_events=list(self._events),
            done=len(done) == len(self._pods),
        )
        if arrived:
            self._source_consumption_ticks.append(self._tick)
            self._post_source_state_digests.append(self._source_state_digest())
        self._records.append(record)
        return record

    def placement_state(self) -> dict[str, Any]:
        queued = sorted(
            (pod for pod in self._pods.values() if pod.status == "queued"),
            key=lambda pod: pod.pod_id,
        )
        assignments = {
            pod.pod_id: {
                "node_id": pod.node_id,
                "gpu_slot_indices": list(pod.gpu_slot_indices),
            }
            for pod in self._pods.values()
            if pod.status == "running"
        }
        node_rows = []
        for node in sorted(self._nodes.values(), key=lambda item: item.node_id):
            usage = self._node_usage(node.node_id)
            remaining = [1000.0 - value for value in usage["gpu_milli"]]
            node_rows.append(
                {
                    "node_id": node.node_id,
                    "gpu_model": node.gpu_model,
                    "cpu_capacity_milli": node.cpu_milli,
                    "cpu_allocated_milli": usage["cpu_milli"],
                    "memory_capacity_mib": node.memory_mib,
                    "memory_allocated_mib": usage["memory_mib"],
                    "gpu_slot_remaining_milli": remaining,
                    "fragmented_gpu_milli": sum(
                        value for value in remaining if value < 1000.0
                    ),
                }
            )
        return {
            "policy": self._placement_policy,
            "autonomous_placement": self._autonomous,
            "assignment_digest": _stable_digest(assignments),
            "assignments": assignments,
            "nodes": node_rows,
            "queued_pods": [
                {
                    "pod_id": pod.pod_id,
                    "qos": pod.qos,
                    "cpu_milli": pod.cpu_milli,
                    "memory_mib": pod.memory_mib,
                    "gpu_count": pod.gpu_count,
                    "gpu_milli_per_gpu": pod.gpu_milli_per_gpu,
                    "compatible_gpu_models": list(pod.compatible_gpu_models),
                    "compatible_node_ids": self._compatible_node_ids(pod),
                    "feasible_node_ids": self._feasible_node_ids(pod),
                    "wait_ticks": pod.wait_ticks,
                    "due_tick": pod.due_tick,
                }
                for pod in queued[:50]
            ],
            "fragmented_gpu_milli": sum(
                row["fragmented_gpu_milli"] for row in node_rows
            ),
        }

    def queue_state(self) -> dict[str, Any]:
        placement = self.placement_state()
        return {
            "queue_policy": self._placement_policy,
            "policy_generation": 1,
            "dispatch_order": [row["pod_id"] for row in placement["queued_pods"]],
            "running_job_ids": sorted(placement["assignments"]),
            "dispatch_rationale": {},
            "queued_jobs": [
                {"job_id": row["pod_id"], **row} for row in placement["queued_pods"]
            ],
        }

    def capacity_state(self) -> dict[str, Any]:
        placement = self.placement_state()
        return {
            "node_count": len(self._nodes),
            "gpu_capacity_units": sum(
                len(node.gpu_slots) for node in self._nodes.values()
            ),
            "cpu_capacity_milli": sum(node.cpu_milli for node in self._nodes.values()),
            "memory_capacity_mib": sum(
                node.memory_mib for node in self._nodes.values()
            ),
            "fragmented_gpu_milli": placement["fragmented_gpu_milli"],
        }

    def arrival_forecast(self, horizon_ticks: int) -> dict[str, Any]:
        end = min(self._horizon, self._tick + max(1, int(horizon_ticks)) + 1)
        pods = [
            pod
            for pod in self._pods.values()
            if pod.status == "future" and self._tick < pod.submit_tick < end
        ]
        return {
            "from_tick": self._tick + 1,
            "to_tick": end - 1,
            "expected_pod_count": len(pods),
            "expected_gpu_milli": sum(
                pod.gpu_count * pod.gpu_milli_per_gpu for pod in pods
            ),
            "qos_counts": dict(sorted(Counter(pod.qos for pod in pods).items())),
            "source": "locked_openb_future_arrivals",
        }

    def apply_tool_effect(
        self, name: str, args: dict[str, Any], current_tick: int | None = None
    ) -> dict[str, Any]:
        if current_tick is not None:
            self._tick = int(current_tick)
        before = self._action_state_digest()
        changed: list[str] = []
        applied: dict[str, Any] = {}
        actuator = "node_placement_engine"
        if name == "set_placement_policy":
            policy = str(args.get("policy") or "")
            if policy not in PLACEMENT_POLICIES:
                return {"_status": "error", "error": "unknown_placement_policy"}
            previous = self._placement_policy
            if previous == policy:
                return {
                    "_status": "no_effect",
                    "error": "placement_policy_unchanged",
                    "policy": policy,
                    "previous_policy": previous,
                }
            self._placement_policy = policy
            applied = {"policy": policy, "previous_policy": previous}
            changed = ["placement_policy"]
            actuator = "placement_policy_controller"
        elif name == "place_pod":
            pod = self._pods.get(str(args.get("pod_id") or ""))
            node_id = str(args.get("node_id") or "")
            if pod is None or pod.status != "queued":
                return {"_status": "error", "error": "pod_not_queued"}
            if not self._place(pod, node_id):
                return {"_status": "error", "error": "placement_infeasible"}
            applied = {"pod_id": pod.pod_id, "node_id": node_id}
            changed = [
                "pod_assignment",
                "node_resource_allocation",
                "placement_fragmentation",
            ]
        elif name == "migrate_pod":
            pod = self._pods.get(str(args.get("pod_id") or ""))
            target = str(args.get("node_id") or "")
            if pod is None or pod.status != "running" or not pod.node_id:
                return {"_status": "error", "error": "pod_not_running"}
            previous_node = pod.node_id
            previous_slots = pod.gpu_slot_indices
            pod.node_id = None
            pod.gpu_slot_indices = ()
            pod.status = "queued"
            if not self._place(pod, target, migrated=True):
                pod.node_id = previous_node
                pod.gpu_slot_indices = previous_slots
                pod.status = "running"
                return {"_status": "error", "error": "migration_infeasible"}
            self._migration_cost_total += (
                1.0 + pod.gpu_count * pod.gpu_milli_per_gpu / 1000.0
            )
            applied = {
                "pod_id": pod.pod_id,
                "from_node_id": previous_node,
                "node_id": target,
            }
            changed = [
                "pod_assignment",
                "node_resource_allocation",
                "placement_fragmentation",
                "migration_cost",
            ]
        else:
            return {"_status": "error", "error": "unknown_tool"}
        token = self._new_effect(
            name=name,
            requested=dict(args),
            applied=applied,
            before=before,
            changed=changed,
        )
        self._control_events.append(
            {"type": name, "tick": self._tick, "physical_actuator_id": actuator}
        )
        return {
            "_status": "ok",
            "_backend_effect_token": token,
            "physical_actuator_id": actuator,
            **applied,
        }

    def _new_effect(
        self,
        *,
        name: str,
        requested: dict[str, Any],
        applied: dict[str, Any],
        before: str,
        changed: list[str],
    ) -> str:
        self._effect_sequence += 1
        token = f"datacenter:openb:{name}:{self._effect_sequence}"
        effect = {
            "token": token,
            "tool_name": name,
            "requested_action": requested,
            "applied_action": applied,
            "before_state_digest": before,
            "after_state_digest": self._action_state_digest(),
            "changed_state_fields": changed,
            "call_id": None,
            "evidence_ids": [],
        }
        self._effects[token] = effect
        self._pending_effects.append(effect)
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
        token = str(payload.get("_backend_effect_token") or "")
        effect = self._effects.get(token)
        if effect is None:
            return
        effect["call_id"] = call_id
        if evidence_id:
            effect["evidence_ids"].append(evidence_id)
        if causal_parent_event_id:
            effect["causal_parent_event_id"] = causal_parent_event_id

    def _effect_event(self, effect: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "control_effect",
            "event_id": f"{effect['token']}@{self._tick}",
            "origin": "agent_caused",
            "agent_caused": True,
            "tool_name": effect["tool_name"],
            "call_id": str(effect.get("call_id") or ""),
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
        }

    def _action_state_digest(self) -> str:
        return _stable_digest(
            {
                "policy": self._placement_policy,
                "assignments": {
                    pod.pod_id: [pod.node_id, pod.gpu_slot_indices]
                    for pod in self._pods.values()
                    if pod.status == "running"
                },
            }
        )

    def _source_state_digest(self) -> str:
        return _stable_digest(
            {
                "nodes": [
                    {
                        "node_id": node.node_id,
                        "cpu_milli": node.cpu_milli,
                        "memory_mib": node.memory_mib,
                        "gpu_model": node.gpu_model,
                        "gpu_slots": list(node.gpu_slots),
                    }
                    for node in sorted(
                        self._nodes.values(), key=lambda item: item.node_id
                    )
                ],
                "pods": [
                    {
                        "pod_id": pod.pod_id,
                        "cpu_milli": pod.cpu_milli,
                        "memory_mib": pod.memory_mib,
                        "gpu_count": pod.gpu_count,
                        "gpu_milli_per_gpu": pod.gpu_milli_per_gpu,
                        "compatible_gpu_models": list(pod.compatible_gpu_models),
                        "qos": pod.qos,
                        "submit_tick": pod.submit_tick,
                        "duration_ticks": pod.duration_ticks,
                        "due_tick": pod.due_tick,
                    }
                    for pod in sorted(self._pods.values(), key=lambda item: item.pod_id)
                ],
                "queued": sorted(
                    pod.pod_id for pod in self._pods.values() if pod.status == "queued"
                ),
                "running": sorted(
                    pod.pod_id for pod in self._pods.values() if pod.status == "running"
                ),
                "fragmented_gpu_milli": self.placement_state()["fragmented_gpu_milli"],
            }
        )

    def snapshot(self) -> dict[str, Any]:
        placement = self.placement_state()
        return {
            "tick": self._tick,
            "decision_opportunity": bool(placement["queued_pods"]),
            "placement": placement,
            "queue": self.queue_state(),
            "capacity": self.capacity_state(),
            "jobs": {
                pod.pod_id: {
                    "kind": "gpu_pod",
                    "user": pod.qos,
                    "qos": pod.qos,
                    "status": pod.status,
                    "submit_tick": pod.submit_tick,
                    "remaining_ticks": pod.remaining_ticks,
                    "gpu_units": pod.gpu_count * pod.gpu_milli_per_gpu / 1000.0,
                    "cpu_units": pod.cpu_milli,
                    "memory_mib": pod.memory_mib,
                    "gpu_model_constraints": list(pod.compatible_gpu_models),
                    "criticality": QOS_CRITICALITY.get(pod.qos, 0.3),
                    "due_tick": pod.due_tick,
                    "wait_ticks": pod.wait_ticks,
                    "node_id": pod.node_id,
                    "migrations": pod.migrations,
                }
                for pod in self._pods.values()
                if pod.status != "future"
            },
        }

    def ground_truth_costs(self) -> dict[str, float]:
        unfinished = sum(
            pod.remaining_ticks * (1.0 + QOS_CRITICALITY.get(pod.qos, 0.3))
            for pod in self._pods.values()
            if pod.status != "done"
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
            "migration_cost": round(self._migration_cost_total, 6),
        }

    def per_job_sla_violation_minutes(self) -> dict[str, float]:
        tick_minutes = int(self._seed_obj.tick_minutes if self._seed_obj else 1)
        return {
            pod.pod_id: float(
                max(0, pod.wait_ticks - max(0, pod.due_tick - pod.submit_tick))
                * tick_minutes
            )
            for pod in self._pods.values()
        }

    def control_summary(self) -> dict[str, Any]:
        tools = sorted({event["type"] for event in self._control_events})
        return {
            "distinct_physical_tools": tools,
            "distinct_control_ticks": sorted(
                {int(event["tick"]) for event in self._control_events}
            ),
            "distinct_physical_actuator_endpoints": sorted(
                {str(event["physical_actuator_id"]) for event in self._control_events}
            ),
            "tool_ticks": {
                tool: sorted(
                    {
                        int(event["tick"])
                        for event in self._control_events
                        if event["type"] == tool
                    }
                )
                for tool in tools
            },
            "migration_count": sum(pod.migrations for pod in self._pods.values()),
        }

    def protocol21_source_trace(self) -> dict[str, Any]:
        assert self._source_graph is not None
        assets = list(self._source_graph["runtime_opened_assets"])
        source_hashes = {
            str(item["source_path"]): str(item["sha256"]) for item in assets
        }
        opened_hashes = {str(item["path"]): str(item["sha256"]) for item in assets}
        trace_semantic_digest = _stable_digest(
            {
                "source_graph_sha256": self._source_graph["source_graph_sha256"],
                "consumption_ticks": self._source_consumption_ticks,
                "post_source_state_digests": self._post_source_state_digests,
            }
        )
        observed = bool(
            self._source_consumption_ticks and self._post_source_state_digests
        )
        return {
            "status": "passed" if observed else "held",
            "proof_kind": "direct_runtime_files",
            "runtime_opened_assets": [
                {"path": item["path"], "sha256": item["sha256"], "role": item["role"]}
                for item in assets
            ],
            "opened_source_paths": sorted(opened_hashes),
            "opened_source_sha256": opened_hashes,
            "consumed_source_hashes": source_hashes,
            "lineage_source_hashes": {},
            "source_graph_sha256": self._source_graph["source_graph_sha256"],
            "parser_output_digest": self._source_graph["source_graph_sha256"],
            "trace_semantic_digest": trace_semantic_digest,
            "consumed_channels": list(self._source_graph["consumed_channels"]),
            "derived_backend_state_fields": [
                "pod_assignments",
                "node_resource_allocation",
                "placement_fragmentation",
                "qos_delay_risk",
            ],
            "consumption_ticks": list(self._source_consumption_ticks),
            "initial_state_digest": self._initial_state_digest,
            "post_source_state_digests": list(self._post_source_state_digests),
            "state_effect_observed": observed,
            "source_state_effect_observed": observed,
            "deterministic_source_trace": True,
            "runtime_trace_observed": observed,
            "evidence_from_scenario_config_only": False,
            "blockers": [] if observed else ["source_transition_not_observed"],
        }

    def scoring_records(self) -> list[dict[str, Any]]:
        rows = []
        for record in self._records:
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
                    "startup_cost": 0.0,
                    "shed_penalty": record.queue_wait_cost + record.sla_violation_cost,
                    "rho_max": record.gpu_allocated / max(1.0, record.gpu_capacity),
                    "n_overloads": 0,
                    "n_voltage_violations": 0,
                    "n_disconnected_lines": 0,
                    "done": False,
                    "catastrophic_failure": False,
                    "safety_violation_severity": min(
                        1.0,
                        record.sla_violation_cost
                        / max(1.0, record.queue_wait_cost * 8.0),
                    ),
                }
            )
        return rows
