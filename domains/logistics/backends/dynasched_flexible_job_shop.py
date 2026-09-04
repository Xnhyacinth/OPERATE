"""Source-locked adapter for DynaSchedBench dynamic flexible job shop data.

This module deliberately does not translate DynaSchedBench bundles into the
fixed-machine JSPLIB representation.  The official ``dsbx`` runtime remains
the state-transition authority and every dispatch preserves its explicit
machine-group and concrete-machine choice.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core import EvidenceLogger, ToolContext, ToolRegistry, ToolSpec
from core.source_asset_contract import resolve_source_asset_contract

DYNASCHED_BACKEND_KIND = "dynasched_flexible_job_shop"
DYNASCHED_RUNTIME_VERSION = "0.3.0"
DYNASCHED_RUNTIME_COMMIT = "08975bf4a0473c5dff9177393bc6743db9ddc946"
DYNASCHED_RUNTIME_CODE_TREE_SHA256 = (
    "35f60f494e386f864d5be905d09dd87ccbad2a25878d35f6dee5458f79e8cff2"
)
REPO_ROOT = Path(__file__).resolve().parents[3]

_EVENT_NAMES = {
    "ARRIVAL": "job_arrival",
    "DUE_DATE_SET": "due_date_set",
    "BREAKDOWN": "machine_breakdown",
    "REPAIR_COMPLETION": "machine_repair_completion",
    "PTIME_CHANGE": "process_time_change",
    "PRIORITY_CHANGE": "priority_change",
    "ORDER_CANCELLATION": "order_cancellation",
    "PREVENTIVE_MAINTENANCE": "preventive_maintenance",
    "ROUTE_CHANGE": "route_change",
    "DUE_DATE_CHANGE": "due_date_change",
}

_EVENT_STATE_FIELDS = {
    "ARRIVAL": ["jobs", "ready_operations", "operations_arrived"],
    "DUE_DATE_SET": ["jobs.due_date"],
    "BREAKDOWN": ["machines.status", "machines.available_from", "ready_operations"],
    "REPAIR_COMPLETION": ["machines.status", "machines.available_from"],
    "PTIME_CHANGE": ["jobs.ops.proc_time_realized"],
    "PRIORITY_CHANGE": ["jobs.priority"],
    "ORDER_CANCELLATION": ["jobs.status", "operations_cancelled"],
    "PREVENTIVE_MAINTENANCE": ["machines.status", "machines.available_from"],
    "ROUTE_CHANGE": ["jobs.ops.machine_group", "ready_operations"],
    "DUE_DATE_CHANGE": ["jobs.due_date"],
}


def validate_source_event_registry(
    config: dict[str, Any], event_types: set[str]
) -> dict[str, dict[str, Any]]:
    """Return a complete typed registry for the source stream or fail closed."""
    raw = config.get("source_event_registry")
    registry = raw if isinstance(raw, dict) else {}
    declared_counts = config.get("source_event_counts")
    declared_types = (
        set(declared_counts) if isinstance(declared_counts, dict) else set()
    )
    invalid = []
    for event_type in sorted(event_types | declared_types):
        declaration = registry.get(event_type)
        if (
            not isinstance(declaration, dict)
            or not str(declaration.get("type") or "").strip()
            or not str(declaration.get("event_class") or "").strip()
            or not isinstance(declaration.get("actionable"), bool)
        ):
            invalid.append(event_type)
    if invalid:
        raise ValueError(
            "dynasched_source_event_registry_incomplete:" + ",".join(invalid)
        )
    return {
        event_type: dict(registry[event_type])
        for event_type in sorted(event_types | declared_types)
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _python_tree_sha256(root: Path) -> str:
    files = sorted(
        root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix()
    )
    if not files:
        raise RuntimeError("dynasched_runtime_code_tree_empty")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _resolve_asset(raw: str) -> Path:
    path = Path(raw)
    candidates = [path] if path.is_absolute() else [REPO_ROOT / path, path]
    resolved = next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()), None
    )
    if resolved is None:
        raise FileNotFoundError(f"required_source_file_missing:{raw}")
    return resolved


@dataclass(frozen=True)
class DynaSchedTickRecord:
    tick: int
    routing_cost: float
    dispatch_fixed_cost: float
    drop_penalty: float
    unmet_demand: float
    done: bool
    realized_events: list[dict[str, Any]]


class DynaSchedFlexibleJobShopBackend:
    """Thin deterministic wrapper around the official ``DynaSchedSim``."""

    backend_kind = DYNASCHED_BACKEND_KIND

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = dict(config or {})
        self._sim: Any = None
        self._source_paths: dict[str, Path] = {}
        self._source_hashes: dict[str, str] = {}
        self._runtime_source_hashes: dict[str, str] = {}
        self._metadata: dict[str, Any] = {}
        self._operations_total = 0
        self._current_tick = 0
        self._tick_records: list[dict[str, Any]] = []
        self._post_source_state_digests: list[str] = []
        self._source_event_effects: list[dict[str, Any]] = []
        self._consumption_ticks: list[int] = []
        self._parser_output_digest = ""
        self._initial_state_digest = ""
        self._runtime_event_types: set[str] = set()
        self._pending_action_effects: list[dict[str, Any]] = []
        self._source_event_ids: dict[int, str] = {}
        self._expected_source_event_ids: list[str] = []
        self._events_source_sha256 = ""

    @classmethod
    def from_seed(cls, seed: Any) -> DynaSchedFlexibleJobShopBackend:
        return cls(dict(seed.backend_config))

    @classmethod
    def from_config(cls, scenario: dict[str, Any]) -> DynaSchedFlexibleJobShopBackend:
        return cls(dict(scenario.get("backend_config") or {}))

    def reset(self, scenario_seed: Any) -> None:
        self._config = dict(
            getattr(scenario_seed, "backend_config", self._config) or {}
        )
        if self._config.get("runtime_version") != DYNASCHED_RUNTIME_VERSION:
            raise ValueError("dynasched_runtime_version_lock_mismatch")
        try:
            installed = importlib.metadata.version("dsbx")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "dynasched_runtime_unavailable: install source-locked dsbx==0.3.0"
            ) from exc
        if installed != DYNASCHED_RUNTIME_VERSION:
            raise RuntimeError(
                "dynasched_runtime_version_mismatch:"
                f"expected={DYNASCHED_RUNTIME_VERSION} actual={installed}"
            )
        runtime_source_lock = self._config.get("runtime_source_lock") or {}
        configured_commit = str(runtime_source_lock.get("commit") or "")
        configured_tree = str(
            self._config.get("runtime_code_tree_sha256")
            or runtime_source_lock.get("python_package_tree_sha256")
            or ""
        ).removeprefix("sha256:")
        if configured_commit != DYNASCHED_RUNTIME_COMMIT:
            raise ValueError("dynasched_runtime_source_lock_mismatch")
        if configured_tree != DYNASCHED_RUNTIME_CODE_TREE_SHA256:
            raise ValueError("dynasched_runtime_code_tree_lock_mismatch")
        if (
            str(runtime_source_lock.get("python_package_tree_sha256") or "")
            != configured_tree
        ):
            raise ValueError("dynasched_runtime_source_lock_mismatch")

        import dsbx

        installed_tree = _python_tree_sha256(Path(dsbx.__file__).resolve().parent)
        if installed_tree != configured_tree:
            raise ValueError(
                "dynasched_runtime_code_tree_lock_mismatch:"
                f"expected={configured_tree}:actual={installed_tree}"
            )

        locks = self._config.get("source_assets") or {}
        required = ("input_model_json", "events_jsonl")
        if any(not isinstance(locks.get(name), dict) for name in required):
            raise ValueError("dynasched_source_asset_lock_missing")
        self._source_paths = {}
        self._source_hashes = {}
        self._runtime_source_hashes = {}
        for name, lock in locks.items():
            if not isinstance(lock, dict) or not lock.get("path"):
                continue
            declared = str(lock["path"])
            path = _resolve_asset(declared)
            actual = _sha256(path)
            expected = str(lock.get("sha256") or "").removeprefix("sha256:")
            if actual != expected:
                raise ValueError(
                    f"source_hash_mismatch:{name}:expected={expected}:actual={actual}"
                )
            self._source_paths[name] = path
            self._source_hashes[declared] = actual
            if name in required:
                self._runtime_source_hashes[declared] = actual

        from dsbx.Gen import load_input_model
        from dsbx.Sim.Loader import load_events_from_jsonl
        from dsbx.Sim.Simulator import DynaSchedSim

        model = load_input_model(self._source_paths["input_model_json"])
        events = load_events_from_jsonl(self._source_paths["events_jsonl"])
        if not events:
            raise ValueError("dynasched_event_stream_empty")
        event_types = {str(getattr(event, "event_type", "")) for event in events}
        if "ARRIVAL" not in event_types:
            raise ValueError("dynasched_arrival_events_missing")
        validate_source_event_registry(self._config, event_types)

        static_jobs_path = self._source_paths.get("static_jobs_json")
        static_machines_path = self._source_paths.get("static_machines_json")
        if static_jobs_path is None or static_machines_path is None:
            raise ValueError("dynasched_bundle_crosscheck_assets_missing")
        static_jobs = json.loads(static_jobs_path.read_text(encoding="utf-8"))
        static_machines = json.loads(static_machines_path.read_text(encoding="utf-8"))
        jobs = dict(static_jobs.get("jobs") or {})
        machine_rows = list(static_machines.get("machines") or [])
        model_machine_ids = {str(machine.id) for machine in model.plant.machines}
        static_machine_ids = {str(machine.get("id")) for machine in machine_rows}
        if model_machine_ids != static_machine_ids:
            raise ValueError("dynasched_static_machine_inventory_mismatch")
        arrival_ids = {
            str(event.job_id)
            for event in events
            if str(getattr(event, "event_type", "")) == "ARRIVAL"
        }
        if set(jobs) != arrival_ids:
            raise ValueError("dynasched_static_job_inventory_mismatch")

        self._metadata = {
            "jobs": len(jobs),
            "machines": len(machine_rows),
            "machine_groups": len({str(row.get("group")) for row in machine_rows}),
            "events": len(events),
            "event_types": sorted(event_types),
            "source_horizon": float(model.scale.horizon),
        }
        self._operations_total = sum(
            len(list(job.get("routing") or [])) for job in jobs.values()
        )
        self._parser_output_digest = _digest(
            {
                "metadata": self._metadata,
                "machine_inventory": sorted(model_machine_ids),
                "event_stream": [event.model_dump(mode="json") for event in events],
            }
        )
        self._sim = DynaSchedSim(model, events)
        self._current_tick = 0
        self._tick_records = []
        self._post_source_state_digests = []
        self._source_event_effects = []
        self._consumption_ticks = []
        self._runtime_event_types = set()
        self._pending_action_effects = []
        self._events_source_sha256 = _sha256(self._source_paths["events_jsonl"])
        self._prepare_source_event_contract()
        before_reset_state = self._source_state()
        self._sim.reset()
        reset_event_index = int(self._sim._event_index)
        if reset_event_index:
            reset_events = list(self._sim._events[:reset_event_index])
            after_reset_state = self._source_state()
            self._record_source_event_effects(
                reset_events,
                start_index=0,
                before_state=before_reset_state,
                after_state=after_reset_state,
            )
            self._runtime_event_types.update(
                str(getattr(event, "event_type", "")) for event in reset_events
            )
            self._post_source_state_digests.append(_digest(after_reset_state))
            self._consumption_ticks.append(0)
        self._initial_state_digest = self._state_digest()

    @property
    def makespan(self) -> float:
        if self._sim is None:
            return 0.0
        return max(
            (
                float(segment.end)
                for machine in self._sim.state.machines.values()
                for segment in machine.schedule_segments
                if segment.job_id != "__DOWNTIME__"
            ),
            default=0.0,
        )

    def max_dispatch_batch_size(self) -> int:
        return 50

    def ready_operations(self) -> list[dict[str, Any]]:
        if self._sim is None:
            return []
        return [
            {
                "job_id": str(operation.job_id),
                "operation_index": int(operation.index),
                "operation_id": str(operation.op_id),
                "machine_group": str(operation.machine_group),
                "candidate_machines": list(operation.candidate_machines),
                "processing_time": float(operation.proc_time_realized),
                "previous_machine_id": operation.prev_machine_id,
                "rework_count": int(operation.rework_count),
            }
            for operation in self._sim.get_ready_operations()
        ]

    def native_oracle_reference_dispatch(
        self, *, max_operations: int | None = None
    ) -> list[dict[str, Any]]:
        """Return a deterministic native-state earliest-completion dispatch.

        This is an executable reference policy, not an optimum claim.  It uses
        the official simulator's machine availability, speed, downtime blocks,
        job priority, and ready-operation state.  The method is deliberately
        kept on the backend because the offline oracle may inspect hidden
        simulator state while publishable agents may not.
        """

        if self._sim is None:
            return []
        remaining = self.ready_operations()
        limit = min(
            self.max_dispatch_batch_size(),
            max(
                0,
                int(
                    self.max_dispatch_batch_size()
                    if max_operations is None
                    else max_operations
                ),
            ),
        )
        provisional_available = {
            str(machine_id): float(machine.available_from)
            for machine_id, machine in self._sim.state.machines.items()
        }
        selected: list[dict[str, Any]] = []
        while remaining and len(selected) < limit:
            candidates: list[
                tuple[tuple[float, float, float, str, int, str], int, str, float]
            ] = []
            for index, operation in enumerate(remaining):
                job_id = str(operation["job_id"])
                job = self._sim.state.jobs[job_id]
                processing_time = float(operation["processing_time"])
                for machine_id_value in operation["candidate_machines"]:
                    machine_id = str(machine_id_value)
                    machine = self._sim.state.machines[machine_id]
                    start = max(
                        float(self._sim.state.time),
                        float(job.release_time),
                        float(job.next_available_time),
                        provisional_available[machine_id],
                        float(
                            self._sim._machine_block_until(
                                machine_id, str(machine.group)
                            )
                        ),
                    )
                    end = start + processing_time / max(1e-9, float(machine.speed))
                    # Match DynaSchedSim.estimate_action_score: earliest end,
                    # with the native hidden priority as a small correction.
                    rank = (
                        end - 1e-3 * float(job.priority),
                        end,
                        float(job.due_date),
                        job_id,
                        int(operation["operation_index"]),
                        machine_id,
                    )
                    candidates.append((rank, index, machine_id, end))
            if not candidates:
                break
            _rank, index, machine_id, end = min(candidates, key=lambda row: row[0])
            operation = remaining.pop(index)
            selected.append(
                {
                    "job_id": str(operation["job_id"]),
                    "operation_index": int(operation["operation_index"]),
                    "machine_id": machine_id,
                }
            )
            provisional_available[machine_id] = end
        return selected

    def dispatch_operation(
        self, *, job_id: str, operation_index: int, machine_id: str
    ) -> dict[str, Any]:
        ready = {
            (row["job_id"], row["operation_index"]): row
            for row in self.ready_operations()
        }
        row = ready.get((job_id, operation_index))
        if row is None:
            return {
                "_status": "error",
                "error": "operation_not_ready",
                "job_id": job_id,
                "operation_index": operation_index,
            }
        if machine_id not in row["candidate_machines"]:
            return {
                "_status": "error",
                "error": "machine_not_in_operation_candidate_set",
                "job_id": job_id,
                "operation_index": operation_index,
                "machine_id": machine_id,
                "candidate_machines": row["candidate_machines"],
            }
        before = self._state_digest()
        self._sim.step_action(
            {
                "job_id": job_id,
                "machine_group": row["machine_group"],
                "machine_id": machine_id,
            }
        )
        after = self._state_digest()
        result = {
            "_status": "scheduled",
            "job_id": job_id,
            "operation_index": operation_index,
            "operation_id": row["operation_id"],
            "machine_group": row["machine_group"],
            "machine_id": machine_id,
            "candidate_machines": row["candidate_machines"],
            "rework_count": row["rework_count"],
            "before_state_digest": before,
            "after_state_digest": after,
        }
        self._pending_action_effects.append(
            {
                "type": "operation_dispatched",
                "origin": "agent_caused",
                "agent_caused": True,
                "tool_name": "dispatch_flexible_operations",
                "requested_action": {
                    "job_id": job_id,
                    "operation_index": operation_index,
                    "machine_id": machine_id,
                },
                "applied_action": dict(result),
                "before_state_digest": before,
                "after_state_digest": after,
                "changed_state_fields": [
                    "jobs.current_op_index",
                    "machines.schedule_segments",
                    "machines.available_from",
                ],
                "materiality_metric": "native_state_digest_change",
                "materiality_value": 1.0 if before != after else 0.0,
                "materiality_threshold": 1.0,
            }
        )
        return result

    def bind_tool_result(
        self,
        *,
        name: str,
        call_id: str,
        evidence_id: str,
        payload: dict[str, Any],
    ) -> None:
        if name != "dispatch_flexible_operations":
            return
        for effect in self._pending_action_effects:
            if effect.get("call_id"):
                continue
            effect["call_id"] = call_id
            effect["causal_call_id"] = call_id
            effect["evidence_ids"] = [evidence_id]

    def tick(self, current_tick: int) -> DynaSchedTickRecord:
        if self._sim is None:
            raise RuntimeError("dynasched_backend_not_reset")
        self._current_tick = int(current_tick)
        realized: list[dict[str, Any]] = []
        consumed_source_this_tick = False
        hidden_types = self._hidden_source_event_types()
        while True:
            before_event_index = int(self._sim._event_index)
            current_time = float(self._sim.state.time)
            next_event_time = (
                float(self._sim._events[before_event_index].time)
                if before_event_index < len(self._sim._events)
                else None
            )
            native_boundaries = [
                float(job.next_available_time)
                for job in self._sim.state.jobs.values()
                if float(job.next_available_time) > current_time + 1e-9
                and job.status not in {"completed", "cancelled"}
            ]
            native_boundaries.extend(
                float(machine.available_from)
                for machine in self._sim.state.machines.values()
                if float(machine.available_from) > current_time + 1e-9
            )
            candidates = list(native_boundaries)
            if next_event_time is not None:
                candidates.append(next_event_time)
            target = (
                min(candidates)
                if candidates
                else min(float(self._sim.state.horizon), current_time + 1.0)
            )
            native_boundary_reached = any(
                abs(value - target) <= 1e-9 for value in native_boundaries
            )
            before_state = self._source_state()
            before_digest = _digest(before_state)
            self._sim._process_events_until(target)
            after_state = self._source_state()
            after_digest = _digest(after_state)
            after_event_index = int(self._sim._event_index)
            source_events = list(
                self._sim._events[before_event_index:after_event_index]
            )
            if source_events:
                consumed_source_this_tick = True
                event_types = [
                    str(getattr(event, "event_type", "")) for event in source_events
                ]
                source_effects = self._record_source_event_effects(
                    source_events,
                    start_index=before_event_index,
                    before_state=before_state,
                    after_state=after_state,
                )
                self._runtime_event_types.update(event_types)
                self._post_source_state_digests.append(after_digest)
                for index, event in enumerate(source_events, start=before_event_index):
                    if str(getattr(event, "event_type", "")) in hidden_types:
                        continue
                    runtime_event = self._runtime_event(
                        event,
                        index=index,
                        before_state_digest=before_digest,
                        after_state_digest=after_digest,
                        before_state=before_state,
                        after_state=after_state,
                    )
                    if source_effects:
                        runtime_event["source_transition_event_ids"] = [
                            str(row["event_id"]) for row in source_effects
                        ]
                    realized.append(runtime_event)
            if realized or native_boundary_reached or not source_events:
                break

        realized.extend(self._pending_action_effects)
        self._pending_action_effects = []

        if consumed_source_this_tick and (
            not self._consumption_ticks
            or self._consumption_ticks[-1] != self._current_tick
        ):
            self._consumption_ticks.append(self._current_tick)

        snapshot = self.snapshot()
        unscheduled = float(snapshot["operations_remaining"])
        record = self._canonical_row_for_tick(current_tick)
        self._tick_records.append(record)
        done = bool(
            int(self._sim._event_index) >= len(self._sim._events)
            and snapshot["operations_remaining"] == 0
        )
        return DynaSchedTickRecord(
            tick=int(current_tick),
            routing_cost=float(record["production_cost"]),
            dispatch_fixed_cost=0.0,
            drop_penalty=float(record["shed_penalty"]),
            unmet_demand=unscheduled,
            done=done,
            realized_events=realized,
        )

    def snapshot(self) -> dict[str, Any]:
        if self._sim is None:
            return {}
        native = self._sim.export_snapshot()
        stats = native.system_stats
        operations = [
            operation
            for job in self._sim.state.jobs.values()
            for operation in job.ops
        ]
        completed = sum(operation.status == "done" for operation in operations)
        cancelled = sum(operation.status == "cancelled" for operation in operations)
        resolved = int(completed) + int(cancelled)
        operations_total = len(operations)
        machine_groups = {
            group: {
                "candidate_machines": list(values.get("machine_ids") or []),
                "total_speed": float(values.get("total_speed") or 0.0),
            }
            for group, values in sorted(self._sim.state.machine_groups.items())
        }
        hidden_types = self._hidden_source_event_types()
        event_counters = {
            str(name): value
            for name, value in dict(stats.event_counters).items()
            if str(name)
            not in {_EVENT_NAMES.get(value, value.lower()) for value in hidden_types}
        }
        next_visible_event_time = next(
            (
                float(event.time)
                for event in self._sim._events[self._sim._event_index :]
                if str(getattr(event, "event_type", "")) not in hidden_types
            ),
            None,
        )
        return {
            "backend_kind": self.backend_kind,
            "instance_id": str(self._config.get("bundle_id") or "dynasched_bundle"),
            "simulator_time": float(native.time),
            "source_horizon": float(native.horizon),
            "ready_operations": self.ready_operations(),
            "operations_total": operations_total,
            "operations_completed": int(completed),
            "operations_cancelled": int(cancelled),
            # Cancellations terminate work but never receive scheduling credit.
            "operations_scheduled": int(completed),
            "operations_remaining": max(0, operations_total - resolved),
            "jobs_total": int(self._metadata.get("jobs") or 0),
            "jobs_arrived": int(stats.num_jobs_arrived),
            "jobs_completed": int(stats.num_jobs_completed),
            "jobs_cancelled": int(stats.num_jobs_cancelled),
            "machine_groups": machine_groups,
            "machines": {
                str(machine.machine_id): {
                    "group": str(machine.group),
                    "speed": float(machine.speed),
                    "status": str(machine.status),
                    "available_from": float(machine.available_from),
                }
                for machine in native.machines
            },
            "event_counters": event_counters,
            "makespan": self.makespan,
            "machine_alternatives_preserved": all(
                len(row["candidate_machines"]) > 1 for row in machine_groups.values()
            ),
            "decision_cadence": {
                "mode": "hybrid",
                "simulator_owned_clock": True,
                "current_time": float(native.time),
                "next_visible_source_event_time": next_visible_event_time,
            },
        }

    def ground_truth_costs(self) -> dict[str, float]:
        snapshot = self.snapshot()
        unscheduled = float(snapshot.get("operations_remaining") or 0.0)
        return {"production_cost": self.makespan + unscheduled * 1000.0}

    def per_customer_unmet_units(self) -> dict[str, float]:
        if self._sim is None:
            return {}
        return {
            str(job.job_id): float(
                sum(
                    operation.status not in {"done", "cancelled"}
                    for operation in job.ops
                )
            )
            for job in self._sim.state.jobs.values()
        }

    def scoring_records(self) -> list[dict[str, Any]]:
        return list(self._tick_records) or [self._canonical_row_for_tick(0)]

    def protocol21_source_trace(self) -> dict[str, Any]:
        channel_by_event_type = {
            "ARRIVAL": "events.arrivals",
            "DUE_DATE_SET": "events.due_dates",
            "DUE_DATE_CHANGE": "events.due_dates",
            "BREAKDOWN": "events.breakdowns",
            "REPAIR_COMPLETION": "events.repairs",
            "PTIME_CHANGE": "events.process_time_changes",
            "PRIORITY_CHANGE": "events.priority_changes",
            "ORDER_CANCELLATION": "events.order_cancellations",
            "PREVENTIVE_MAINTENANCE": "events.preventive_maintenance",
            "ROUTE_CHANGE": "events.route_changes",
        }
        consumed_channels = {
            "plant.machine_groups",
            "plant.machine_speeds",
            *(
                channel_by_event_type[event_type]
                for event_type in self._runtime_event_types
            ),
        }
        derived_fields = {"machines", "machine_groups"}
        for event_type in self._runtime_event_types:
            derived_fields.update(_EVENT_STATE_FIELDS.get(event_type, []))
        expected_events = list(self._expected_source_event_ids)
        observed_events = [
            str(row["event_id"]) for row in self._source_event_effects
        ]
        material_effects = [
            row
            for row in self._source_event_effects
            if row.get("materiality_passed") is True
            and bool(row.get("changed_state_fields"))
            and bool(row.get("before_state_digest"))
            and bool(row.get("after_state_digest"))
            and row.get("before_state_digest") != row.get("after_state_digest")
        ]
        material_events = [str(row["event_id"]) for row in material_effects]
        event_coverage_complete = bool(expected_events) and (
            len(expected_events) == len(set(expected_events))
            and observed_events == expected_events
            and material_events == expected_events
        )
        trace_payload = {
            "parser_output_digest": self._parser_output_digest,
            "initial_state_digest": self._initial_state_digest,
            "post_source_state_digests": self._post_source_state_digests,
            "consumption_ticks": self._consumption_ticks,
            "event_types": sorted(self._runtime_event_types),
            "source_event_effects": self._source_event_effects,
        }
        result = {
            "status": "passed" if event_coverage_complete else "held",
            "proof_kind": "direct_runtime_files",
            "consumed_source_hashes": dict(sorted(self._runtime_source_hashes.items())),
            "lineage_source_hashes": {},
            "consumed_channels": sorted(consumed_channels),
            "derived_backend_state_fields": sorted(derived_fields),
            "consumption_ticks": list(self._consumption_ticks),
            "state_effect_observed": bool(material_effects),
            "source_state_effect_observed": bool(material_effects),
            "opened_source_paths": sorted(
                str(self._source_paths[name])
                for name in ("input_model_json", "events_jsonl")
            ),
            "opened_source_sha256": {
                str(self._source_paths[name]): _sha256(self._source_paths[name])
                for name in ("input_model_json", "events_jsonl")
            },
            "runtime_opened_assets": sorted(
                str(self._source_paths[name])
                for name in ("input_model_json", "events_jsonl")
            ),
            "metadata_crosscheck_assets": sorted(
                str(path)
                for name, path in self._source_paths.items()
                if name not in {"input_model_json", "events_jsonl"}
            ),
            "parser_output_digest": self._parser_output_digest,
            "instance_kind": "dynamic_flexible_job_shop_bundle",
            "initial_state_digest": self._initial_state_digest,
            "post_source_state_digests": list(self._post_source_state_digests),
            "source_event_effects": list(self._source_event_effects),
            "material_source_event_effects": material_effects,
            "source_field_to_state_field_map": {
                **{"plant.machines": ["machines", "machine_groups"]},
                **{
                    f"events.{event_type}": list(_EVENT_STATE_FIELDS[event_type])
                    for event_type in sorted(self._runtime_event_types)
                    if event_type in _EVENT_STATE_FIELDS
                },
            },
            "deterministic_source_trace": True,
            "trace_semantic_digest": _digest(trace_payload),
            "runtime_trace_observed": bool(self._source_event_effects),
            "evidence_from_scenario_config_only": False,
            "runtime_identity": {
                "package": "dsbx",
                "version": DYNASCHED_RUNTIME_VERSION,
                "upstream_commit": DYNASCHED_RUNTIME_COMMIT,
                "python_package_tree_sha256": DYNASCHED_RUNTIME_CODE_TREE_SHA256,
            },
        }
        if expected_events:
            result.update(
                {
                    "expected_source_event_ids": expected_events,
                    "observed_source_event_ids": observed_events,
                    "material_source_event_ids": material_events,
                    "source_event_materiality": list(self._source_event_effects),
                    "named_events_causally_proven": event_coverage_complete,
                    "blockers": (
                        []
                        if event_coverage_complete
                        else [
                            "named_source_event_materiality_unproven"
                            if observed_events == expected_events
                            else "named_source_event_replay_incomplete"
                        ]
                    ),
                }
            )
        else:
            result["blockers"] = ["source_event_contract_missing"]
        return result

    def _runtime_event(
        self,
        event: Any,
        *,
        index: int,
        before_state_digest: str,
        after_state_digest: str,
        before_state: dict[str, Any] | None = None,
        after_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = event.model_dump(mode="json")
        upstream_type = str(payload.get("event_type") or "UNKNOWN")
        registry = self._config.get("source_event_registry") or {}
        declaration = registry.get(upstream_type)
        if not isinstance(declaration, dict):
            declaration = {}
        event_type = str(
            declaration.get("type")
            or _EVENT_NAMES.get(upstream_type, upstream_type.lower())
        )
        changed = list(_EVENT_STATE_FIELDS.get(upstream_type, []))
        event_class = str(declaration.get("event_class") or "lifecycle")
        decision_required = declaration.get("actionable") is True
        before_projection = (
            self._event_state_projection(payload, before_state)
            if before_state is not None
            else None
        )
        after_projection = (
            self._event_state_projection(payload, after_state)
            if after_state is not None
            else None
        )
        if before_projection is not None and after_projection is not None:
            event_before_digest = _digest(before_projection)
            event_after_digest = _digest(after_projection)
        else:
            event_before_digest = before_state_digest
            event_after_digest = after_state_digest
        material = event_before_digest != event_after_digest
        return {
            "type": event_type,
            "event_class": event_class,
            "origin": "source_schedule",
            "decision_required": decision_required,
            "actionable": decision_required,
            "event_id": self._source_event_id(event, index=index),
            "source_event_ids": [self._source_event_id(event, index=index)],
            "source_event_type": upstream_type,
            "source_event_index": int(index),
            "source_time": float(event.time),
            "hidden": False,
            "changed_state_fields": changed,
            "before_state_digest": event_before_digest,
            "after_state_digest": event_after_digest,
            "source_boundary_before_state_digest": before_state_digest,
            "source_boundary_after_state_digest": after_state_digest,
            "before_state_projection": before_projection,
            "after_state_projection": after_projection,
            "materiality_metric": (
                "event_attributed_native_state_projection_change"
                if before_projection is not None
                else "native_state_digest_change"
            ),
            "materiality_value": 1.0 if material else 0.0,
            "materiality_threshold": 1.0,
            "materiality_passed": material,
            "state_observation_kind": "native_backend_readback",
            "response_window_required": upstream_type
            in {"BREAKDOWN", "PTIME_CHANGE", "ROUTE_CHANGE", "PRIORITY_CHANGE"},
            "declared_event": payload,
        }

    def _prepare_source_event_contract(self) -> None:
        """Bind declared proof events to exact rows in the locked event stream."""
        raw_contract = self._config.get("source_event_contract")
        if not isinstance(raw_contract, list) or not raw_contract:
            self._source_event_ids = {}
            self._expected_source_event_ids = []
            return
        selected: dict[int, str] = {}
        for declaration in raw_contract:
            if not isinstance(declaration, dict):
                raise ValueError("dynasched_source_event_contract_invalid")
            try:
                index = int(declaration["source_event_index"])
                event = self._sim._events[index]
                source_time = float(declaration["source_time"])
            except (IndexError, KeyError, TypeError, ValueError):
                raise ValueError("dynasched_source_event_contract_invalid") from None
            source_type = str(getattr(event, "event_type", ""))
            if (
                index < 0
                or index in selected
                or source_type != str(declaration.get("source_event_type") or "")
                or abs(float(event.time) - source_time) > 1e-9
            ):
                raise ValueError("dynasched_source_event_contract_mismatch")
            selected[index] = self._source_event_id(event, index=index)
        self._source_event_ids = dict(sorted(selected.items()))
        self._expected_source_event_ids = list(self._source_event_ids.values())

    def _source_event_id(self, event: Any, *, index: int) -> str:
        payload = event.model_dump(mode="json")
        identity = _digest(
            {
                "events_jsonl_sha256": self._events_source_sha256,
                "source_event_index": int(index),
                "source_event": payload,
            }
        )
        return f"dynasched:{identity}"

    def _record_source_event_effects(
        self,
        source_events: list[Any],
        *,
        start_index: int,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Record exact declared events against one native transition boundary."""
        before_state_digest = _digest(before_state)
        after_state_digest = _digest(after_state)
        effects: list[dict[str, Any]] = []
        for index, event in enumerate(source_events, start=start_index):
            if index not in self._source_event_ids:
                continue
            effect = self._runtime_event(
                event,
                index=index,
                before_state_digest=before_state_digest,
                after_state_digest=after_state_digest,
                before_state=before_state,
                after_state=after_state,
            )
            effect.update(
                {
                    "benchmark_tick": self._current_tick,
                    "source_event_indices": list(
                        range(start_index, start_index + len(source_events))
                    ),
                    "source_event_types": [
                        str(getattr(row, "event_type", ""))
                        for row in source_events
                    ],
                    "source_times": [float(row.time) for row in source_events],
                    "material_state_change": effect["materiality_passed"],
                }
            )
            self._source_event_effects.append(effect)
            effects.append(effect)
        return effects

    @staticmethod
    def _native_row(
        rows: Any, *, identity_field: str, identity: str
    ) -> dict[str, Any] | None:
        values = rows.values() if isinstance(rows, dict) else rows or []
        return next(
            (
                row
                for row in values
                if isinstance(row, dict)
                and str(row.get(identity_field) or "") == identity
            ),
            None,
        )

    def _event_state_projection(
        self, event: dict[str, Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        """Project only the native entity fields attributable to one source event."""
        event_type = str(event.get("event_type") or "")
        job_id = str(event.get("job_id") or "")
        machine_id = str(event.get("machine_id") or "")
        job = self._native_row(
            state.get("jobs"), identity_field="job_id", identity=job_id
        )
        machine = self._native_row(
            state.get("machines"), identity_field="machine_id", identity=machine_id
        )
        if event_type == "ARRIVAL":
            if job is None:
                return {"job": None}
            return {
                "job": {
                    key: job.get(key)
                    for key in (
                        "job_id",
                        "family",
                        "release_time",
                        "status",
                        "total_ops",
                        "ops",
                    )
                }
            }
        if event_type in {"DUE_DATE_SET", "DUE_DATE_CHANGE"}:
            return {"due_date": None if job is None else job.get("due_date")}
        if event_type == "PRIORITY_CHANGE":
            return {"priority": None if job is None else job.get("priority")}
        if event_type == "ORDER_CANCELLATION":
            return {
                "job_status": None if job is None else job.get("status"),
                "operation_statuses": (
                    None
                    if job is None
                    else [row.get("status") for row in job.get("ops") or []]
                ),
            }
        if event_type in {"BREAKDOWN", "REPAIR_COMPLETION", "PREVENTIVE_MAINTENANCE"}:
            return {
                "machine": (
                    None
                    if machine is None
                    else {
                        key: machine.get(key)
                        for key in (
                            "machine_id",
                            "status",
                            "available_from",
                            "schedule_segments",
                        )
                    }
                ),
                "machine_block": (state.get("machine_blocks") or {}).get(machine_id),
            }
        if event_type == "PTIME_CHANGE":
            step_index = int(event.get("step_index") or 0)
            operations = [] if job is None else list(job.get("ops") or [])
            operation = next(
                (
                    row
                    for row in operations
                    if int(row.get("index", -1)) == step_index
                ),
                None,
            )
            return {
                "proc_time_realized": (
                    None if operation is None else operation.get("proc_time_realized")
                )
            }
        if event_type == "ROUTE_CHANGE":
            from_step = int(event.get("from_step") or 0)
            operations = [] if job is None else list(job.get("ops") or [])
            return {
                "route": [
                    {
                        key: row.get(key)
                        for key in (
                            "index",
                            "machine_group",
                            "candidate_machines",
                            "proc_time_realized",
                        )
                    }
                    for row in operations
                    if int(row.get("index", -1)) >= from_step
                ]
            }
        return {"unsupported_event_type": event_type}

    def _source_state(self) -> dict[str, Any]:
        """Read native operational state without treating clock advance as effect."""
        if self._sim is None:
            return {}
        snapshot = self._sim.export_snapshot().model_dump(mode="json")
        return {
            "jobs": snapshot.get("jobs"),
            "machines": snapshot.get("machines"),
            "machine_blocks": dict(self._sim._block_until_machine),
            "group_blocks": dict(self._sim._block_until_group),
        }

    def _source_state_digest(self) -> str:
        return _digest(self._source_state())

    def _hidden_source_event_types(self) -> set[str]:
        return {
            str(value) for value in self._config.get("hidden_source_event_types") or []
        }

    def _canonical_row_for_tick(self, tick: int) -> dict[str, Any]:
        snapshot = self.snapshot()
        remaining = int(snapshot.get("operations_remaining") or 0)
        scheduled = int(snapshot.get("operations_scheduled") or 0)
        return {
            "tick": int(tick),
            "aggregate_demand_mw": int(snapshot.get("operations_total") or 0),
            "aggregate_generation_mw": scheduled,
            "balance_error_mw": remaining,
            "reserves_required_mw": 0,
            "reserves_procured_mw": 0,
            "production_cost": round(self.makespan + remaining * 1000.0, 6),
            "startup_cost": 0,
            "shed_penalty": float(remaining),
            "rho_max": round(
                scheduled / max(1, int(snapshot.get("operations_total") or 0)), 6
            ),
            "n_overloads": 0,
            "n_voltage_violations": 0,
            "n_disconnected_lines": 0,
            "done": False,
            "catastrophic_failure": False,
            "safety_violation_severity": 0.0,
        }

    def _state_digest(self) -> str:
        if self._sim is None:
            return _digest({})
        return _digest(self._sim.export_snapshot().model_dump(mode="json"))


def register_dynasched_flexible_job_shop_tools(
    registry: ToolRegistry,
    backend: DynaSchedFlexibleJobShopBackend,
) -> None:
    """Register the one native DynaSched action class and observation tools."""

    registry.register(
        ToolSpec(
            name="query_flexible_job_shop",
            description=(
                "Inspect source-driven flexible-job-shop jobs, concrete machines, "
                "candidate-machine sets, and event counters."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_query_handler(backend),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="flexible_job_shop_state",
            cost_units=0.1,
        )
    )
    registry.register(
        ToolSpec(
            name="dispatch_flexible_operations",
            description=(
                "Assign currently ready operations to explicit eligible concrete "
                "machines. The official DynaSched simulator enforces precedence, "
                "machine alternatives, downtime, speed, and rescheduling."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": backend.max_dispatch_batch_size(),
                        "items": {
                            "type": "object",
                            "properties": {
                                "job_id": {"type": "string"},
                                "operation_index": {"type": "integer", "minimum": 0},
                                "machine_id": {"type": "string"},
                            },
                            "required": ["job_id", "operation_index", "machine_id"],
                        },
                    }
                },
                "required": ["operations"],
            },
            handler=_dispatch_handler(backend),
            state_changing=True,
            semantic_role="control",
            native_target_kind="flexible_operation_batch",
            actuator_family="flexible_job_shop_dispatch",
            cost_units=1.0,
        )
    )
    registry.register(
        ToolSpec(
            name="wait",
            description=(
                "Apply no new scheduling control. The DynaSched simulator clock "
                "and source event stream continue to advance."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda _args, _ctx: {"_status": "waiting"},
            state_changing=False,
            semantic_role="meta",
            native_target_kind="simulation_clock",
            cost_units=0.0,
        )
    )


def _query_handler(backend: DynaSchedFlexibleJobShopBackend):
    def handler(_args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        result = {"_status": "observed", **backend.snapshot()}
        evidence = ctx.extra.get("evidence")
        if isinstance(evidence, EvidenceLogger):
            result["evidence_id"] = evidence.log(
                "job_shop_observation", ctx.tick, result, source="tool"
            )
        return result

    return handler


def _dispatch_handler(backend: DynaSchedFlexibleJobShopBackend):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        requested = args.get("operations")
        if not isinstance(requested, list) or not requested:
            result: dict[str, Any] = {
                "_status": "error",
                "error": "operations_must_be_nonempty_array",
                "results": [],
            }
        elif len(requested) > backend.max_dispatch_batch_size():
            result = {
                "_status": "error",
                "error": "dispatch_batch_size_exceeded",
                "max_batch_size": backend.max_dispatch_batch_size(),
                "results": [],
            }
        else:
            initially_ready = {
                (row["job_id"], row["operation_index"])
                for row in backend.ready_operations()
            }
            seen_jobs: set[str] = set()
            item_results: list[dict[str, Any]] = []
            for item in requested:
                if not isinstance(item, dict):
                    item_results.append(
                        {"_status": "error", "error": "operation_item_must_be_object"}
                    )
                    continue
                job_id = str(item.get("job_id") or "")
                try:
                    operation_index = int(item.get("operation_index", -1))
                except (TypeError, ValueError):
                    operation_index = -1
                if job_id in seen_jobs:
                    item_results.append(
                        {
                            "_status": "error",
                            "error": "duplicate_job_in_batch",
                            "job_id": job_id,
                        }
                    )
                    continue
                seen_jobs.add(job_id)
                if (job_id, operation_index) not in initially_ready:
                    item_results.append(
                        {
                            "_status": "error",
                            "error": "operation_not_ready_at_batch_start",
                            "job_id": job_id,
                            "operation_index": operation_index,
                        }
                    )
                    continue
                item_results.append(
                    backend.dispatch_operation(
                        job_id=job_id,
                        operation_index=operation_index,
                        machine_id=str(item.get("machine_id") or ""),
                    )
                )
            scheduled = sum(row.get("_status") == "scheduled" for row in item_results)
            result = {
                "_status": "scheduled_batch" if scheduled else "error",
                "error": None if scheduled else "no_operations_scheduled",
                "requested_count": len(requested),
                "scheduled_count": scheduled,
                "rejected_count": len(item_results) - scheduled,
                "results": item_results,
                "makespan": backend.makespan,
            }
        evidence = ctx.extra.get("evidence")
        if isinstance(evidence, EvidenceLogger):
            result["evidence_id"] = evidence.log(
                "job_shop_tool_call",
                ctx.tick,
                {
                    "tool": "dispatch_flexible_operations",
                    "ok": result.get("_status") != "error",
                    **result,
                },
                source="tool",
            )
        return result

    return handler


def build_dynasched_source_contract(
    scenario: dict[str, Any], repo_root: Path
) -> dict[str, Any]:
    del repo_root
    assets = (scenario.get("backend_config") or {}).get("source_assets") or {}
    runtime = [
        str(assets[name]["path"])
        for name in ("input_model_json", "events_jsonl")
        if isinstance(assets.get(name), dict) and assets[name].get("path")
    ]
    metadata = [
        str(lock["path"])
        for name, lock in sorted(assets.items())
        if name not in {"input_model_json", "events_jsonl", "license"}
        and isinstance(lock, dict)
        and lock.get("path")
    ]
    license_lock = assets.get("license")
    license_files = (
        [str(license_lock["path"])]
        if isinstance(license_lock, dict) and license_lock.get("path")
        else []
    )
    return {
        "runtime_input": runtime,
        "derivation_input": [],
        "implementation_asset": [],
        "metadata": metadata,
        "license": license_files,
    }


def extract_dynasched_source_evidence(
    *, env: Any, scenario: dict[str, Any]
) -> dict[str, Any]:
    contract = resolve_source_asset_contract(scenario, repo_root=REPO_ROOT)
    if contract.contract_errors or contract.missing_required_files:
        return {
            "status": "held",
            "proof_kind": "direct_runtime_files",
            "blockers": sorted(
                {
                    *contract.contract_errors,
                    *(
                        ["required_source_file_missing"]
                        if contract.missing_required_files
                        else []
                    ),
                }
            ),
        }
    backend = getattr(env, "_backend", None)
    if not isinstance(backend, DynaSchedFlexibleJobShopBackend):
        return {
            "status": "held",
            "proof_kind": "direct_runtime_files",
            "blockers": ["dynasched_backend_not_reset"],
        }
    return backend.protocol21_source_trace()
