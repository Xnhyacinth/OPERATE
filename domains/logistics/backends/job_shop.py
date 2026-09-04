"""JSPLIB Job-Shop backend (``jsplib_job_shop``).

Adapts the anchored ``works/JSPLIB-Instances`` samples into a deterministic,
logistics-native state transition and tool surface. Dispatched by
``domains/logistics/adapter.py``'s ``build_backend()`` for scenarios with
``backend_kind == "jsplib_job_shop"`` and released in the core suite
(51 core rows as of v0.51.0).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from core import EventDecisionClass, EvidenceLogger, ToolContext, ToolRegistry, ToolSpec
from core.common_tools import commit_to_plan_handler, plan_autonomy_properties
from domains.logistics.seeds.schema import LogisticsScenarioSeed

_UNSCHEDULED_OPERATION_PENALTY = 100.0
_MAX_DISPATCH_BATCH_SIZE = 50
_REPO_ROOT = Path(__file__).resolve().parents[3]
_NATIVE_EVENT_REGISTRY = MappingProxyType(
    {
        "machine_breakdown": MappingProxyType(
            {
                "type": "machine_breakdown",
                "event_class": "safety",
                "actionable": True,
            }
        ),
        "demand_surge": MappingProxyType(
            {
                "type": "demand_surge",
                "event_class": "alarm",
                "actionable": True,
            }
        ),
        "urgent_order": MappingProxyType(
            {
                "type": "urgent_order",
                "event_class": "task",
                "actionable": True,
            }
        ),
    }
)


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
class JobShopOperation:
    """One ordered operation in a job-shop routing plan."""

    machine_id: int
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


class JsplibJobShopBackend:
    """Deterministic pure-Python Job-Shop simulator for JSPLIB samples."""

    backend_kind = "jsplib_job_shop"

    def __init__(
        self,
        *,
        instance_id: str,
        jobs: dict[str, list[JobShopOperation]],
        n_machines: int,
    ) -> None:
        if not jobs:
            raise ValueError("job-shop instance must contain at least one job")
        if n_machines <= 0:
            raise ValueError("job-shop instance must contain at least one machine")
        self.instance_id = str(instance_id)
        self._jobs = {job_id: list(ops) for job_id, ops in jobs.items()}
        self._n_machines = int(n_machines)
        self._next_operation = {job_id: 0 for job_id in self._jobs}
        self._job_available_at = {job_id: 0 for job_id in self._jobs}
        self._machine_available_at = {
            machine_id: 0 for machine_id in range(self._n_machines)
        }
        self._scheduled: list[ScheduledOperation] = []
        self._tick_records: list[dict[str, Any]] = []
        self._current_tick = 0
        self._reference_optimum: float | None = None
        self._source_trace: dict[str, Any] | None = None
        self._pending_action_effects: list[dict[str, Any]] = []
        self._completed_operations: set[tuple[str, int]] = set()
        self._operation_lineage: dict[tuple[str, int], dict[str, Any]] = {}
        self._dynamic_mode = False
        self._dynamic_config: dict[str, Any] = {}
        self._active_machine_disruptions: dict[int, int] = {}
        self._active_machine_disruption_event_ids: dict[int, str] = {}
        self._active_job_urgencies: dict[str, float] = {}
        self._seed_perturbations: list[Any] = []
        self._horizon = 0
        self._j2_sidecar: dict[str, Any] | None = None
        self._j2_sidecar_effects: list[dict[str, Any]] = []
        self._source_event_registry: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_parsed_instance(
        cls, instance_id: str, parsed: dict[str, Any]
    ) -> JsplibJobShopBackend:
        """Build a backend from ``parse_jsplib_instance()`` output."""
        rows = parsed.get("jobs_detail")
        if not isinstance(rows, list):
            raise ValueError("parsed JSPLIB instance is missing jobs_detail")
        n_machines = int(parsed.get("machines", 0) or 0)
        jobs: dict[str, list[JobShopOperation]] = {}
        for job_idx, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValueError(f"job row {job_idx} is not a list")
            job_id = f"j{job_idx}"
            jobs[job_id] = [
                JobShopOperation(
                    machine_id=int(op["machine"]),
                    duration=int(op["duration"]),
                )
                for op in row
            ]
        return cls(instance_id=instance_id, jobs=jobs, n_machines=n_machines)

    @classmethod
    def from_seed(cls, seed: LogisticsScenarioSeed) -> JsplibJobShopBackend:
        """Build a backend from the release-grade Job-Shop seed skeleton."""
        cfg = seed.backend_config.get("job_shop") or {}
        rows = cfg.get("jobs_detail")
        if seed.backend_kind != cls.backend_kind:
            raise ValueError(f"expected jsplib_job_shop seed, got {seed.backend_kind}")
        if not isinstance(rows, list):
            raise ValueError(
                "job-shop seed is missing backend_config.job_shop.jobs_detail"
            )
        jobs: dict[str, list[JobShopOperation]] = {}
        for job_idx, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValueError(f"job-shop seed job row {job_idx} is not a list")
            jobs[f"j{job_idx}"] = [
                JobShopOperation(
                    machine_id=int(op["machine"]),
                    duration=int(op["duration"]),
                )
                for op in row
            ]
        backend = cls(
            instance_id=str(seed.backend_config.get("instance_name") or seed.seed_id),
            jobs=jobs,
            n_machines=int(cfg.get("machines", 0)),
        )
        backend.reset(seed)
        return backend

    def reset(self, scenario_seed: LogisticsScenarioSeed) -> None:
        """Reset dynamic state from the exact locked JSPLIB instance."""
        self._j2_sidecar = None
        self._j2_sidecar_effects = []
        raw_event_registry = scenario_seed.backend_config.get("source_event_registry")
        if raw_event_registry is None:
            self._source_event_registry = {}
        elif not isinstance(raw_event_registry, dict):
            raise ValueError("job_shop_source_event_registry_invalid")
        else:
            self._source_event_registry = {}
            for event_name, declaration in raw_event_registry.items():
                if not isinstance(declaration, dict):
                    raise ValueError("job_shop_source_event_registry_invalid")
                event_class = declaration.get("event_class")
                actionable = declaration.get("actionable")
                if not isinstance(event_class, str) or not isinstance(
                    actionable, bool
                ):
                    raise ValueError("job_shop_source_event_registry_incomplete")
                try:
                    EventDecisionClass(event_class)
                except ValueError:
                    raise ValueError(
                        "job_shop_source_event_registry_unknown_class"
                    ) from None
                self._source_event_registry[str(event_name)] = dict(declaration)
        cfg = scenario_seed.backend_config.get("job_shop") or {}
        self.instance_id = str(
            scenario_seed.backend_config.get("instance_name") or scenario_seed.seed_id
        )
        canonical_j2 = _canonical_j2_runtime_enabled(scenario_seed)
        source_path: Path | None = None
        source_label: str | None = None
        source_sha256: str | None = None
        if not canonical_j2:
            source_path, source_label = _resolve_instance_source(
                scenario_seed, self.instance_id
            )
            source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
            expected_sha256 = str(
                scenario_seed.backend_config.get("expected_sha256") or ""
            ).removeprefix("sha256:")
            if not expected_sha256 or source_sha256 != expected_sha256:
                raise ValueError(
                    "source_hash_mismatch: "
                    f"expected={expected_sha256 or 'missing'} actual={source_sha256}"
                )

        # Local import avoids the seed-builder/backend import cycle.
        from domains.logistics.seeds.from_jsplib import parse_jsplib_instance

        parsed = None if canonical_j2 else parse_jsplib_instance(source_path)
        j2_sidecar = _load_j2_event_sidecar(scenario_seed, parsed)
        if canonical_j2:
            if j2_sidecar is None:
                raise ValueError("realm_j2_canonical_source_missing")
            parsed = j2_sidecar["parsed"]
        if parsed is None:
            raise ValueError("job-shop source graph missing")
        baked = {
            key: cfg.get(key)
            for key in (
                "jobs",
                "machines",
                "operations",
                "machine_ids",
                "total_processing_time",
                "min_processing_time",
                "max_processing_time",
                "jobs_detail",
            )
        }
        if baked != parsed:
            raise ValueError("source_baked_operation_mismatch")

        rows = parsed["jobs_detail"]
        self._jobs = {
            f"j{job_idx}": [
                JobShopOperation(
                    machine_id=int(op["machine"]),
                    duration=int(op["duration"]),
                )
                for op in row
            ]
            for job_idx, row in enumerate(rows)
        }
        self._n_machines = int(parsed["machines"])
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
        self._pending_action_effects = []
        self._completed_operations = set()
        self._operation_lineage = {}
        self._dynamic_config = dict(
            scenario_seed.backend_config.get("dynamic_job_shop") or {}
        )
        self._horizon = int(getattr(scenario_seed, "horizon_ticks", 0) or 0)
        self._dynamic_mode = bool(
            self._dynamic_config.get("enabled")
            or scenario_seed.backend_config.get("dynamic_mode")
        )
        self._active_machine_disruptions = {}
        self._active_machine_disruption_event_ids = {}
        self._active_job_urgencies = {}
        self._seed_perturbations = list(
            getattr(scenario_seed, "perturbations", []) or []
        )
        self._j2_sidecar = j2_sidecar
        self._j2_sidecar_effects = []
        self._validate_dynamic_contract(scenario_seed)
        parser_digest = _digest(parsed)
        initial_state_digest = self._state_digest()
        runtime_assets: list[dict[str, Any]] = []
        opened_source_paths: list[str] = []
        opened_source_sha256: dict[str, str] = {}
        consumed_source_hashes: dict[str, str] = {}
        consumed_channels: list[str] = []
        source_field_to_state_field_map: dict[str, list[str]] = {}
        if source_path is not None and source_label is not None and source_sha256:
            resolved_source_path = str(source_path.resolve())
            runtime_assets.append(
                {
                    "path": source_label,
                    "sha256": source_sha256,
                    "role": "runtime_input",
                }
            )
            opened_source_paths.append(resolved_source_path)
            opened_source_sha256[resolved_source_path] = source_sha256
            consumed_source_hashes[source_label] = source_sha256
            consumed_channels.extend(
                [
                    "job_precedence",
                    "operation_duration",
                    "operation_machine",
                ]
            )
            source_field_to_state_field_map = {
                "jobs_detail[].machine": [
                    "ready_operations",
                    "machine_available_at",
                ],
                "jobs_detail[].duration": [
                    "ready_operations",
                    "current_makespan",
                ],
                "jobs_detail[] order": [
                    "job_next_operation",
                    "unfinished_operations",
                ],
            }
        if j2_sidecar is not None:
            runtime_assets.append(dict(j2_sidecar["runtime_asset"]))
            opened_source_paths.append(j2_sidecar["resolved_path"])
            opened_source_sha256[j2_sidecar["resolved_path"]] = j2_sidecar["sha256"]
            consumed_source_hashes[j2_sidecar["path"]] = j2_sidecar["sha256"]
            consumed_channels.extend(
                [
                    "j2_event_sidecar.operation_graph",
                    "j2_event_sidecar.machine_breakdown",
                ]
            )
        trace_payload = {
            "proof_kind": "direct_runtime_files",
            "runtime_opened_assets": runtime_assets,
            "opened_source_paths": opened_source_paths,
            "opened_source_sha256": opened_source_sha256,
            "consumed_source_hashes": consumed_source_hashes,
            "lineage_source_hashes": {},
            "consumed_channels": consumed_channels,
            "derived_backend_state_fields": [
                "current_makespan",
                "job_next_operation",
                "machine_available_at",
                "ready_operations",
                "unfinished_operations",
            ],
            "source_field_to_state_field_map": source_field_to_state_field_map,
            "consumption_ticks": [0],
            "parser_output_digest": parser_digest,
            "initial_state_digest": initial_state_digest,
            "post_source_state_digests": [initial_state_digest],
            "state_effect_observed": True,
            "source_state_effect_observed": True,
            "deterministic_source_trace": True,
            "runtime_trace_observed": True,
            "evidence_from_scenario_config_only": False,
        }
        if j2_sidecar is not None:
            trace_payload["source_field_to_state_field_map"].update(
                {
                    "j2_event_sidecar.instances[].jobs": [
                        "ready_operations",
                        "job_next_operation",
                        "unfinished_operations",
                    ],
                    "j2_event_sidecar.instances[].disruptions[].machine_id": [
                        "active_machine_disruptions",
                        "machine_available_at",
                        "ready_operations",
                    ],
                    "j2_event_sidecar.instances[].disruptions[].start_time": [
                        "active_machine_disruptions",
                    ],
                    "j2_event_sidecar.instances[].disruptions[].duration": [
                        "active_machine_disruptions",
                    ],
                }
            )
            trace_payload["sidecar_event"] = dict(j2_sidecar["event_trace"])
            trace_payload["sidecar_operation_graph_digest"] = j2_sidecar[
                "operation_graph_digest"
            ]
            trace_payload["sidecar_runtime_input"] = {
                "path": j2_sidecar["path"],
                "sha256": j2_sidecar["sha256"],
                "git_commit": j2_sidecar["git_commit"],
                "selected_instance_id": j2_sidecar["selected_instance_id"],
            }
            source_event_id = str(j2_sidecar["source_event_id"])
            trace_payload.update(
                {
                    "expected_source_event_ids": [source_event_id],
                    "observed_source_event_ids": [],
                    "material_source_event_ids": [],
                    "source_event_materiality": [],
                    "named_events_causally_proven": False,
                }
            )
        self._source_trace = {
            "status": "held" if j2_sidecar is not None else "passed",
            **trace_payload,
            **(
                {"blockers": ["named_source_event_replay_incomplete"]}
                if j2_sidecar is not None
                else {}
            ),
            "trace_semantic_digest": _digest(trace_payload),
        }
        if self._dynamic_mode:
            self._source_trace["dynamic_event_contract"] = {
                "mode": "procedural_machine_outage_over_locked_jsplib",
                "source_identity": self.instance_id,
                "event_kinds": [
                    str(getattr(p, "kind", ""))
                    for p in self._seed_perturbations
                    if str(getattr(p, "kind", ""))
                    in {"machine_breakdown", "demand_surge", "urgent_order"}
                ],
                "event_target_identity": "source_machine_id",
                "event_parameters_locked_in_seed": True,
                "runtime_state_fields": [
                    "active_machine_disruptions",
                    "machine_available_at",
                    "ready_operations",
                ],
            }
            self._source_trace["trace_semantic_digest"] = _digest(
                {
                    key: value
                    for key, value in self._source_trace.items()
                    if key != "trace_semantic_digest"
                }
            )

    def schedule_next_operation(
        self, *, job_id: str, operation_index: int
    ) -> dict[str, Any]:
        """Schedule the next operation for a job at earliest feasible time."""
        job_id = _normalize_job_id(job_id, self._jobs)
        if job_id not in self._jobs:
            return {
                "_status": "error",
                "error": "unknown_job",
                "job_id": job_id,
                "valid_job_ids": sorted(self._jobs.keys()),
            }
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
        before_digest = self._state_digest()
        self._scheduled.append(scheduled)
        self._next_operation[job_id] = expected_index + 1
        self._job_available_at[job_id] = end
        self._machine_available_at[op.machine_id] = end
        self._pending_action_effects.append(
            {
                "type": "operation_dispatched",
                "origin": "agent_caused",
                "agent_caused": True,
                "event_id": (
                    f"operation_dispatched:{job_id}:{expected_index}:"
                    f"{len(self._scheduled)}"
                ),
                "job_id": job_id,
                "operation_index": expected_index,
                "machine_id": op.machine_id,
                "before_state_digest": before_digest,
                "after_state_digest": self._state_digest(),
                "changed_state_fields": [
                    "current_makespan",
                    "job_next_operation",
                    "machine_available_at",
                    "ready_operations",
                    "unfinished_operations",
                ],
                "materiality_metric": "scheduled_operations",
                "materiality_value": 1,
                "materiality_threshold": 1,
                "materiality_passed": True,
                "call_id": None,
                "evidence_ids": [],
            }
        )
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

    def ready_operations(self) -> dict[str, dict[str, Any]]:
        ready: dict[str, dict[str, Any]] = {}
        for job_id, idx in self._next_operation.items():
            operations = self._jobs[job_id]
            if idx >= len(operations):
                continue
            op = operations[idx]
            item: dict[str, int | float] = {
                "operation_index": idx,
                "machine_id": op.machine_id,
                "duration": op.duration,
            }
            if self._dynamic_mode and self._active_job_urgencies.get(job_id, 0.0) > 0.0:
                item["urgency"] = float(self._active_job_urgencies[job_id])
            ready[job_id] = item
        return ready

    def snapshot(self) -> dict[str, Any]:
        max_operations_per_job = max(
            (len(operations) for operations in self._jobs.values()),
            default=1,
        )
        operations_total = sum(len(ops) for ops in self._jobs.values())
        batch_limit = self.max_dispatch_batch_size()
        minimum_dispatch_waves = max(
            max_operations_per_job,
            (operations_total + batch_limit - 1) // batch_limit,
        )
        ready_operations = self.ready_operations()
        return {
            "domain": "logistics",
            "backend_kind": self.backend_kind,
            "instance_id": self.instance_id,
            "jobs": len(self._jobs),
            "machines": self._n_machines,
            "operations_total": operations_total,
            "operations_scheduled": len(self._scheduled),
            "operations_completed": len(self._completed_operations),
            "unfinished_operations": (
                sum(len(ops) for ops in self._jobs.values())
                - len(self._completed_operations)
            ),
            "decision_opportunity": bool(ready_operations),
            "decision_cadence": {
                "max_operations_per_dispatch": batch_limit,
                "minimum_dispatch_waves": minimum_dispatch_waves,
                # A submitted dispatch is an asynchronous shop-floor command
                # under medium+ tool latency. Advance until it materializes
                # before asking the planner for another scheduling wave.
                "hold_while_actions_pending": True,
                # Allow one full scheduling pass plus replanning slack. Once
                # exhausted, the runner advances the backend without further
                # provider calls instead of turning a large JSPLIB instance
                # into hundreds of artificial LLM prompts.
                "model_decision_budget": 2 * minimum_dispatch_waves + 8,
            },
            "makespan": self.makespan,
            "current_makespan": self.makespan,
            "machine_available_at": dict(self._machine_available_at),
            "job_available_at": dict(self._job_available_at),
            "job_next_operation": dict(self._next_operation),
            "ready_operations": ready_operations,
            "scheduled_operations": [op.to_dict() for op in self._scheduled[-12:]],
            "scheduled_operations_available": len(self._scheduled),
            "dynamic_job_shop": self._dynamic_mode,
            "active_machine_disruptions": {
                str(machine_id): int(until_tick)
                for machine_id, until_tick in sorted(
                    self._active_machine_disruptions.items()
                )
                if until_tick > self._current_tick
            },
            "active_job_urgencies": {
                job_id: float(value)
                for job_id, value in sorted(self._active_job_urgencies.items())
                if value > 0.0
            },
        }

    def tick(self, current_tick: int) -> dict[str, Any]:
        """Advance one supervisory tick and emit a scorer-facing record."""
        self._current_tick = int(current_tick)
        self._expire_machine_disruptions()
        self._apply_machine_breakdowns_at_tick(self._current_tick)
        self._apply_demand_surges_at_tick(self._current_tick)
        self._apply_urgent_orders_at_tick(self._current_tick)
        record = self._canonical_row_for_tick(self._current_tick)
        self._tick_records.append(record)
        operations_total = sum(len(ops) for ops in self._jobs.values())
        realized_events = list(self._pending_action_effects)
        self._pending_action_effects = []
        for event in realized_events:
            if str(event.get("origin") or "") == "agent_caused":
                # The adapter returns the state after this backend tick as
                # current_tick + 1. Record the effect on that same observable
                # boundary instead of the pre-step tool-execution clock.
                event["outcome_tick"] = self._current_tick + 1
        schedule_complete = len(self._scheduled) == operations_total
        for operation in sorted(
            self._scheduled,
            key=lambda item: (
                item.end_time,
                item.job_id,
                item.operation_index,
            ),
        ):
            identity = (operation.job_id, operation.operation_index)
            if identity in self._completed_operations:
                continue
            if not schedule_complete:
                continue
            self._completed_operations.add(identity)
            realized_events.append(
                {
                    "type": "operation_completed",
                    "origin": "endogenous_completion",
                    "event_id": (
                        f"operation_completed:{operation.job_id}:"
                        f"{operation.operation_index}"
                    ),
                    "job_id": operation.job_id,
                    "operation_index": operation.operation_index,
                    "machine_id": operation.machine_id,
                    "changed_state_fields": [
                        "completed_operations",
                        "unfinished_operations",
                    ],
                    "materiality_metric": "completed_operations",
                    "materiality_value": 1,
                    "materiality_threshold": 1,
                    "materiality_passed": True,
                    "schedule_end_time": operation.end_time,
                    **self._operation_lineage.get(identity, {}),
                }
            )
        completed = schedule_complete
        return {
            "tick": self._current_tick,
            "routing_cost": float(record["production_cost"]),
            "dispatch_fixed_cost": 0.0,
            "drop_penalty": float(record["shed_penalty"]),
            "unmet_demand": float(record["balance_error_mw"]),
            "done": completed,
            "realized_events": realized_events,
        }

    @property
    def dynamic_mode(self) -> bool:
        """Whether this seed explicitly opts into dynamic job-shop events."""
        return self._dynamic_mode

    def max_dispatch_batch_size(self) -> int:
        """Return the seed-locked dispatch batch bound."""
        try:
            value = int(self._dynamic_config.get("max_dispatch_batch_size", 50))
        except (TypeError, ValueError):
            value = 50
        return max(1, min(50, value))

    def repair_machine(self, *, machine_id: int) -> dict[str, Any]:
        """Request bounded native recovery of an active machine outage."""
        if not self._dynamic_mode:
            return {"_status": "error", "error": "dynamic_mode_required"}
        try:
            machine_id = int(machine_id)
        except (TypeError, ValueError):
            return {"_status": "error", "error": "invalid_machine_id"}
        if machine_id not in self._machine_available_at:
            return {
                "_status": "error",
                "error": "unknown_machine",
                "machine_id": machine_id,
            }
        until_tick = self._active_machine_disruptions.get(machine_id)
        if until_tick is None or until_tick <= self._current_tick:
            return {
                "_status": "error",
                "error": "no_active_machine_breakdown",
                "machine_id": machine_id,
            }
        causal_parent_event_id = self._active_machine_disruption_event_ids.get(
            machine_id
        )
        if not causal_parent_event_id:
            return {
                "_status": "error",
                "error": "machine_breakdown_lineage_missing",
                "machine_id": machine_id,
            }
        try:
            clearance = int(self._dynamic_config.get("recovery_clearance_ticks", 1))
        except (TypeError, ValueError):
            clearance = 1
        clearance = max(1, clearance)
        recovered_until = min(until_tick, self._current_tick + clearance)
        scheduled_end = max(
            (op.end_time for op in self._scheduled if op.machine_id == machine_id),
            default=0,
        )
        before_digest = self._state_digest()
        previous_available = self._machine_available_at[machine_id]
        self._active_machine_disruptions[machine_id] = recovered_until
        self._machine_available_at[machine_id] = max(scheduled_end, recovered_until)
        effect = {
            "type": "machine_repaired",
            "origin": "agent_caused",
            "agent_caused": True,
            "event_id": (
                f"machine_repaired:{self.instance_id}:{machine_id}:{self._current_tick}"
            ),
            "machine_id": machine_id,
            "causal_parent_event_id": causal_parent_event_id,
            "before_state_digest": before_digest,
            "after_state_digest": self._state_digest(),
            "changed_state_fields": [
                "active_machine_disruptions",
                "machine_available_at",
                "ready_operations",
            ],
            "materiality_metric": "machine_outage_reduction_ticks",
            "materiality_value": max(0, until_tick - recovered_until),
            "materiality_threshold": 1,
            "materiality_passed": until_tick - recovered_until >= 1,
            "previous_machine_available_at": previous_available,
            "recovered_until_tick": recovered_until,
            "call_id": None,
            "evidence_ids": [],
        }
        self._pending_action_effects.append(effect)
        return {
            "_status": "machine_repair_requested",
            "machine_id": machine_id,
            "previous_outage_until_tick": until_tick,
            "recovered_until_tick": recovered_until,
            "machine_available_at": self._machine_available_at[machine_id],
        }

    def bind_tool_result(
        self,
        *,
        name: str,
        call_id: str | None,
        evidence_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        """Attach tool-protocol lineage to a realized dispatch effect."""
        if payload.get("_status") == "machine_repair_requested":
            unbound = [
                effect
                for effect in self._pending_action_effects
                if effect.get("type") == "machine_repaired"
                and effect.get("call_id") is None
                and int(effect.get("machine_id", -1))
                == int(payload.get("machine_id", -2))
            ]
            for effect in unbound:
                effect["tool_name"] = name
                effect["call_id"] = call_id
                effect["evidence_ids"] = [evidence_id] if evidence_id else []
                effect["requested_action"] = {"machine_id": effect["machine_id"]}
                effect["applied_action"] = {
                    "machine_id": effect["machine_id"],
                    "recovered_until_tick": effect["recovered_until_tick"],
                }
                effect["action_to_outcome_edge"] = {
                    "source": f"call:{call_id}",
                    "target": f"outcome:{effect['event_id']}",
                    "kind": "action_to_outcome",
                }
            return
        if payload.get("_status") not in {"scheduled", "scheduled_batch"}:
            return
        result_rows = (
            list(payload.get("results") or [])
            if payload.get("_status") == "scheduled_batch"
            else [payload]
        )
        identities = {
            (str(row.get("job_id") or ""), int(row.get("operation_index", -1)))
            for row in result_rows
            if isinstance(row, dict) and row.get("_status") == "scheduled"
        }
        unbound = [
            effect
            for effect in self._pending_action_effects
            if effect.get("call_id") is None
            and (
                str(effect.get("job_id") or ""),
                int(effect.get("operation_index", -1)),
            )
            in identities
        ]
        for effect in unbound:
            effect["tool_name"] = name
            effect["call_id"] = call_id
            effect["evidence_ids"] = [evidence_id] if evidence_id else []
            effect["requested_action"] = {
                "job_id": effect["job_id"],
                "operation_index": effect["operation_index"],
            }
            effect["applied_action"] = {
                **effect["requested_action"],
                "machine_id": effect["machine_id"],
            }
            effect["action_to_outcome_edge"] = {
                "source": f"call:{call_id}",
                "target": f"outcome:{effect['event_id']}",
                "kind": "action_to_outcome",
            }
            identity = (
                str(effect["job_id"]),
                int(effect["operation_index"]),
            )
            self._operation_lineage[identity] = {
                "causal_parent_event_id": effect["event_id"],
                "causal_call_id": call_id,
                "evidence_ids": list(effect["evidence_ids"]),
            }

    def protocol21_source_trace(self) -> dict[str, Any]:
        if self._source_trace is None:
            return {
                "status": "held",
                "proof_kind": "direct_runtime_files",
                "runtime_trace_observed": False,
                "evidence_from_scenario_config_only": True,
                "blockers": ["source_instance_path_missing"],
            }
        return json.loads(json.dumps(self._source_trace))

    def _refresh_source_trace_digest(self) -> None:
        if self._source_trace is None:
            return
        payload = {
            key: value
            for key, value in self._source_trace.items()
            if key != "trace_semantic_digest"
        }
        self._source_trace["trace_semantic_digest"] = _digest(payload)

    def _state_digest(self) -> str:
        return _digest(
            {
                "instance_id": self.instance_id,
                "machine_available_at": self._machine_available_at,
                "job_next_operation": self._next_operation,
                "ready_operations": self.ready_operations(),
                "unfinished_operations": (
                    sum(len(ops) for ops in self._jobs.values()) - len(self._scheduled)
                ),
                "current_makespan": self.makespan,
                "active_machine_disruptions": self._active_machine_disruptions,
                "active_job_urgencies": self._active_job_urgencies,
            }
        )

    def _validate_dynamic_contract(self, scenario_seed: LogisticsScenarioSeed) -> None:
        """Reject malformed dynamic rows before they enter a replay."""
        events = list(self._seed_perturbations)
        unknown_kinds = sorted(
            {
                str(getattr(event, "kind", ""))
                for event in events
                if str(getattr(event, "kind", "")) not in _NATIVE_EVENT_REGISTRY
            }
        )
        if unknown_kinds:
            raise ValueError(
                "dynamic_job_shop_unknown_event_kind:" + ",".join(unknown_kinds)
            )
        if not self._dynamic_mode:
            if events:
                raise ValueError("dynamic_machine_event_requires_dynamic_mode")
            return
        if not any(
            str(getattr(event, "kind", "")) == "machine_breakdown" for event in events
        ):
            raise ValueError("dynamic_job_shop_requires_machine_breakdown")
        horizon = int(getattr(scenario_seed, "horizon_ticks", 0))
        for event in events:
            kind = str(getattr(event, "kind", ""))
            target = getattr(event, "target", {}) or {}
            if kind == "machine_breakdown":
                try:
                    machine_id = int(target.get("machine_id"))
                except (TypeError, ValueError):
                    raise ValueError("machine_breakdown_machine_id_missing") from None
                if machine_id not in self._machine_available_at:
                    raise ValueError("machine_breakdown_unknown_machine")
            elif kind == "demand_surge":
                job_id = _normalize_job_id(target.get("job_id"), self._jobs)
                if job_id not in self._jobs:
                    raise ValueError("demand_surge_unknown_job")
            elif kind == "urgent_order":
                job_id = _normalize_job_id(target.get("job_id"), self._jobs)
                if job_id not in self._jobs:
                    raise ValueError("urgent_order_unknown_job")
            trigger = int(getattr(event, "trigger_tick", -1))
            duration = int(getattr(event, "duration_ticks", 0))
            if trigger <= 0 or trigger >= horizon:
                raise ValueError("machine_breakdown_trigger_outside_response_window")
            if duration < 1 or trigger + duration > horizon:
                raise ValueError("machine_breakdown_duration_outside_horizon")

    def _expire_machine_disruptions(self) -> None:
        for machine_id, until_tick in list(self._active_machine_disruptions.items()):
            if until_tick <= self._current_tick:
                self._active_machine_disruptions.pop(machine_id, None)
                self._active_machine_disruption_event_ids.pop(machine_id, None)

    def _apply_machine_breakdowns_at_tick(self, tick: int) -> None:
        for event in self._seed_perturbations:
            if (
                str(getattr(event, "kind", "")) != "machine_breakdown"
                or int(getattr(event, "trigger_tick", -1)) != tick
            ):
                continue
            target = getattr(event, "target", {}) or {}
            machine_id = int(target["machine_id"])
            duration = max(1, int(getattr(event, "duration_ticks", 1)))
            until_tick = tick + duration
            source_native = bool(
                self._j2_sidecar is not None
                and self._j2_sidecar["breakdown"]
                == {
                    "machine_id": machine_id,
                    "trigger_tick": tick,
                    "duration_ticks": duration,
                }
            )
            event_id = (
                str(self._j2_sidecar["source_event_id"])
                if source_native
                else f"machine_breakdown:{self.instance_id}:{machine_id}:{tick}"
            )
            before_digest = self._state_digest()
            previous_until = self._active_machine_disruptions.get(machine_id, tick)
            self._active_machine_disruptions[machine_id] = max(
                previous_until, until_tick
            )
            self._active_machine_disruption_event_ids[machine_id] = event_id
            self._machine_available_at[machine_id] = max(
                self._machine_available_at[machine_id], until_tick
            )
            after_digest = self._state_digest()
            declaration = self._source_event_registry.get(
                "machine_breakdown"
            ) or _NATIVE_EVENT_REGISTRY["machine_breakdown"]
            event_class = declaration.get("event_class")
            actionable = bool(
                declaration.get("actionable") is True
                and not bool(getattr(event, "hidden", False))
                and tick + 1 < self._horizon
            )
            self._pending_action_effects.append(
                {
                    "type": "machine_breakdown",
                    **(
                        {
                            "event_class": event_class,
                            "actionable": actionable,
                            "decision_required": actionable,
                        }
                        if isinstance(event_class, str)
                        and isinstance(actionable, bool)
                        else {"decision_required": True}
                    ),
                    "origin": (
                        "source_schedule" if source_native else "declared_perturbation"
                    ),
                    "declared_perturbation": True,
                    "source_native": source_native,
                    "event_id": event_id,
                    **(
                        {"source_event_ids": [event_id]}
                        if source_native
                        else {}
                    ),
                    "declared_event": {
                        "kind": "machine_breakdown",
                        "trigger_tick": tick,
                        "duration_ticks": duration,
                        "target": dict(target),
                    },
                    "tick": tick,
                    "hidden": bool(getattr(event, "hidden", False)),
                    "machine_id": machine_id,
                    "changed_state_fields": [
                        "active_machine_disruptions",
                        "machine_available_at",
                        "ready_operations",
                    ],
                    "materiality_metric": "machine_outage_ticks",
                    "materiality_value": duration,
                    "materiality_threshold": 1,
                    "materiality_passed": before_digest != after_digest,
                    "response_window_required": actionable,
                    "response_opportunity_tick": tick + 1 if actionable else None,
                    "before_state_digest": before_digest,
                    "after_state_digest": after_digest,
                    "state_observation_kind": "native_backend_readback",
                    "evidence_ids": [],
                }
            )
            self._record_j2_sidecar_effect(
                machine_id=machine_id,
                tick=tick,
                duration_ticks=duration,
                event_id=event_id,
                before_state_digest=before_digest,
                after_state_digest=after_digest,
            )

    def _record_j2_sidecar_effect(
        self,
        *,
        machine_id: int,
        tick: int,
        duration_ticks: int,
        event_id: str,
        before_state_digest: str,
        after_state_digest: str,
    ) -> None:
        """Attach runtime state-effect evidence to the locked J2 event."""
        sidecar = self._j2_sidecar
        if sidecar is None:
            return
        breakdown = sidecar["breakdown"]
        if (
            int(breakdown["machine_id"]) != int(machine_id)
            or int(breakdown["trigger_tick"]) != int(tick)
            or int(breakdown["duration_ticks"]) != int(duration_ticks)
        ):
            return
        if any(
            effect.get("event_id") == event_id for effect in self._j2_sidecar_effects
        ):
            return
        effect = {
            "event_id": event_id,
            "event_type": "machine_breakdown",
            "machine_id": int(machine_id),
            "trigger_tick": int(tick),
            "duration_ticks": int(duration_ticks),
            "before_state_digest": before_state_digest,
            "after_state_digest": after_state_digest,
            "state_effect_observed": before_state_digest != after_state_digest,
            "state_effect_fields": [
                "active_machine_disruptions",
                "machine_available_at",
                "ready_operations",
            ],
        }
        self._j2_sidecar_effects.append(effect)
        if self._source_trace is None:
            return
        sidecar_event = dict(self._source_trace.get("sidecar_event") or {})
        sidecar_event.update(
            {
                "state_effect_observed": effect["state_effect_observed"],
                "state_effect_applied": True,
                "state_effect_before_digest": before_state_digest,
                "state_effect_after_digest": after_state_digest,
                "state_effect_event_id": event_id,
            }
        )
        self._source_trace["sidecar_event"] = sidecar_event
        self._source_trace["sidecar_state_effects"] = list(self._j2_sidecar_effects)
        materiality = {
            "event_id": event_id,
            "event_type": "machine_breakdown",
            "changed_state_fields": list(effect["state_effect_fields"]),
            "before_state_digest": before_state_digest,
            "after_state_digest": after_state_digest,
            "state_observation_kind": "native_backend_readback",
            "materiality_metric": "native_state_digest_change",
            "materiality_value": 1.0 if effect["state_effect_observed"] else 0.0,
            "materiality_threshold": 1.0,
            "materiality_passed": effect["state_effect_observed"],
        }
        self._source_trace["observed_source_event_ids"] = [event_id]
        self._source_trace["material_source_event_ids"] = (
            [event_id] if effect["state_effect_observed"] else []
        )
        self._source_trace["source_event_materiality"] = [materiality]
        self._source_trace["named_events_causally_proven"] = effect[
            "state_effect_observed"
        ]
        self._source_trace["status"] = (
            "passed" if effect["state_effect_observed"] else "held"
        )
        self._source_trace["blockers"] = (
            []
            if effect["state_effect_observed"]
            else ["named_source_event_materiality_unproven"]
        )
        ticks = {
            int(value) for value in self._source_trace.get("consumption_ticks") or []
        }
        ticks.add(int(tick))
        self._source_trace["consumption_ticks"] = sorted(ticks)
        digests = list(self._source_trace.get("post_source_state_digests") or [])
        if after_state_digest not in digests:
            digests.append(after_state_digest)
        self._source_trace["post_source_state_digests"] = digests
        self._refresh_source_trace_digest()

    def _apply_demand_surges_at_tick(self, tick: int) -> None:
        for event in self._seed_perturbations:
            if (
                str(getattr(event, "kind", "")) != "demand_surge"
                or int(getattr(event, "trigger_tick", -1)) != tick
            ):
                continue
            target = getattr(event, "target", {}) or {}
            job_id = _normalize_job_id(target.get("job_id"), self._jobs)
            intensity = max(1.0, float(getattr(event, "intensity", 1.0)))
            before_digest = self._state_digest()
            self._active_job_urgencies[job_id] = max(
                self._active_job_urgencies.get(job_id, 0.0), intensity
            )
            declaration = self._source_event_registry.get(
                "demand_surge"
            ) or _NATIVE_EVENT_REGISTRY["demand_surge"]
            actionable = bool(
                declaration.get("actionable") is True
                and not bool(getattr(event, "hidden", False))
                and tick + 1 < self._horizon
            )
            self._pending_action_effects.append(
                {
                    "type": "demand_surge",
                    "event_class": str(declaration["event_class"]),
                    "actionable": actionable,
                    "origin": "declared_perturbation",
                    "declared_perturbation": True,
                    "event_id": (f"demand_surge:{self.instance_id}:{job_id}:{tick}"),
                    "declared_event": {
                        "kind": "demand_surge",
                        "trigger_tick": tick,
                        "duration_ticks": int(getattr(event, "duration_ticks", 1)),
                        "target": dict(target),
                    },
                    "tick": tick,
                    "hidden": bool(getattr(event, "hidden", False)),
                    "decision_required": actionable,
                    "job_id": job_id,
                    "changed_state_fields": [
                        "active_job_urgencies",
                        "ready_operations",
                    ],
                    "materiality_metric": "job_urgency_multiplier",
                    "materiality_value": intensity,
                    "materiality_threshold": 1,
                    "materiality_passed": intensity >= 1.0,
                    "response_window_required": actionable,
                    "response_opportunity_tick": tick + 1 if actionable else None,
                    "before_state_digest": before_digest,
                    "after_state_digest": self._state_digest(),
                    "evidence_ids": [],
                }
            )

    def _apply_urgent_orders_at_tick(self, tick: int) -> None:
        """Promote a source-locked job into an explicit urgent-order queue.

        This is a logistics-native priority overlay, not a synthetic arrival:
        the job identity and operation graph are consumed from JSPLIB, while
        the deterministic overlay creates a separate response obligation for
        extreme recovery tasks.
        """
        for event in self._seed_perturbations:
            if (
                str(getattr(event, "kind", "")) != "urgent_order"
                or int(getattr(event, "trigger_tick", -1)) != tick
            ):
                continue
            target = getattr(event, "target", {}) or {}
            job_id = _normalize_job_id(target.get("job_id"), self._jobs)
            intensity = max(1.0, float(getattr(event, "intensity", 1.0)))
            before_digest = self._state_digest()
            self._active_job_urgencies[job_id] = max(
                self._active_job_urgencies.get(job_id, 0.0), intensity
            )
            duration = max(1, int(getattr(event, "duration_ticks", 1)))
            declaration = self._source_event_registry.get(
                "urgent_order"
            ) or _NATIVE_EVENT_REGISTRY["urgent_order"]
            actionable = bool(
                declaration.get("actionable") is True
                and not bool(getattr(event, "hidden", False))
                and tick + 1 < self._horizon
            )
            self._pending_action_effects.append(
                {
                    "type": "urgent_order",
                    "event_class": str(declaration["event_class"]),
                    "actionable": actionable,
                    "origin": "declared_perturbation",
                    "declared_perturbation": True,
                    "event_id": f"urgent_order:{self.instance_id}:{job_id}:{tick}",
                    "declared_event": {
                        "kind": "urgent_order",
                        "trigger_tick": tick,
                        "duration_ticks": duration,
                        "target": dict(target),
                    },
                    "tick": tick,
                    "hidden": bool(getattr(event, "hidden", False)),
                    "decision_required": actionable,
                    "job_id": job_id,
                    "changed_state_fields": [
                        "active_job_urgencies",
                        "ready_operations",
                    ],
                    "materiality_metric": "job_priority_multiplier",
                    "materiality_value": intensity,
                    "materiality_threshold": 1,
                    "materiality_passed": intensity >= 1.0,
                    "response_window_required": actionable,
                    "response_opportunity_tick": tick + 1 if actionable else None,
                    "response_window_end_tick": min(
                        self._current_tick + duration, self._horizon - 1
                    ),
                    "before_state_digest": before_digest,
                    "after_state_digest": self._state_digest(),
                    "evidence_ids": [],
                }
            )

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
                    # Canonical ``done`` is reserved for catastrophic early
                    # termination, not successful schedule completion.
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
                "makespan": max(prev.end_time for prev in self._scheduled[:idx]),
            }
            for idx, op in enumerate(self._scheduled, start=1)
        ]

    def ground_truth_costs(self) -> dict[str, float]:
        return {"production_cost": round(self._objective_cost(), 3)}

    def per_customer_unmet_units(self) -> dict[str, float]:
        return {}

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
            "shed_penalty": round(unscheduled * _UNSCHEDULED_OPERATION_PENALTY, 3),
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


def register_job_shop_tools(
    reg: ToolRegistry,
    backend: JsplibJobShopBackend,
    env: Any | None = None,
) -> None:
    """Register the dev-only Job-Shop tool surface through core protocol."""

    batch_limit = backend.max_dispatch_batch_size()

    reg.register(
        ToolSpec(
            name="query_job_queue",
            description=(
                "Inspect the currently ready JSPLIB operations and scheduling "
                "progress without changing the job-shop state."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_h_query_job_queue(backend),
            state_changing=False,
            semantic_role="investigation",
            native_target_kind="ready_operation_queue",
            cost_units=0.1,
        )
    )
    reg.register(
        ToolSpec(
            name="dispatch_ready_operations",
            description=(
                f"Atomically submit an ordered batch of 1-{batch_limit} mutually "
                "independent operations copied from the current "
                "observation's ready_operations. Each job may appear at "
                "most once in a batch. Operations are scheduled in the "
                "provided order at their earliest precedence- and "
                "machine-feasible times; invalid entries are reported "
                "without rolling back valid entries. Use as many currently "
                "ready jobs as possible so large instances are scheduled in "
                "decision waves rather than one prompt per operation."
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
                "Schedule the next unscheduled operation of a JSPLIB job at "
                "the earliest feasible time respecting job precedence and "
                "single-machine capacity. Copy job_id and operation_index "
                "from the current observation's ready_operations; never "
                "repeat a pair after it schedules successfully."
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
    if backend.dynamic_mode and env is not None:
        reg.register(
            ToolSpec(
                name="commit_to_plan",
                description=(
                    "Record a standing JSPLIB dispatch plan or replace it after "
                    "an observed native shop-floor disruption."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "plan_id": {"type": "string"},
                        "rationale": {"type": "string"},
                        "replaces_plan_id": {"type": "string"},
                        "revision_reason": {"type": "string"},
                        "trigger_evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        **plan_autonomy_properties(),
                        "predicted_events": {
                            "type": "array",
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["plan_id"],
                },
                handler=commit_to_plan_handler(
                    env,
                    include_horizon_ticks=False,
                ),
                state_changing=False,
                semantic_role="planning",
                native_target_kind="standing_plan",
                cost_units=0.0,
            )
        )
    if backend.dynamic_mode:
        reg.register(
            ToolSpec(
                name="repair_machine",
                description=(
                    "Request native recovery of an active machine breakdown. "
                    "The simulator preserves a seed-declared clearance interval "
                    "and rejects unknown or inactive machine outages."
                ),
                parameters={
                    "type": "object",
                    "properties": {"machine_id": {"type": "integer", "minimum": 0}},
                    "required": ["machine_id"],
                },
                handler=_h_repair_machine(backend),
                state_changing=True,
                semantic_role="control",
                native_target_kind="machine_breakdown",
                actuator_family="machine_repair",
                cost_units=0.5,
            )
        )
    reg.register(
        ToolSpec(
            name="wait",
            description="Advance without scheduling a job-shop operation.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=_h_wait,
            state_changing=False,
            semantic_role="meta",
            native_target_kind="simulation_clock",
            cost_units=0.0,
        )
    )


def _h_query_job_queue(backend: JsplibJobShopBackend):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        snapshot = backend.snapshot()
        operations_total = int(snapshot["operations_total"])
        operations_scheduled = int(snapshot["operations_scheduled"])
        result = {
            "_status": "observed",
            "instance_id": snapshot["instance_id"],
            "ready_operations": snapshot["ready_operations"],
            "operations_total": operations_total,
            "operations_scheduled": operations_scheduled,
            "operations_remaining": max(0, operations_total - operations_scheduled),
            "makespan": snapshot["makespan"],
            "machine_available_at": snapshot["machine_available_at"],
        }
        evidence = ctx.extra.get("evidence")
        if isinstance(evidence, EvidenceLogger):
            result["evidence_id"] = evidence.log(
                "job_shop_observation",
                ctx.tick,
                payload={"tool": "query_job_queue", "ok": True, **result},
                source="tool",
            )
        return result

    return handler


def _h_dispatch_ready_operations(backend: JsplibJobShopBackend):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        requested = args.get("operations")
        max_batch_size = backend.max_dispatch_batch_size()
        if (
            not isinstance(requested, list)
            or not requested
            or (len(requested) > max_batch_size and not backend.dynamic_mode)
        ):
            result: dict[str, Any] = {
                "_status": "error",
                "error": (
                    "operations_must_contain_1_to_"
                    f"{backend.max_dispatch_batch_size()}_items"
                ),
                "requested_count": len(requested) if isinstance(requested, list) else 0,
            }
        else:
            overflow = list(requested[max_batch_size:])
            processable = list(requested[:max_batch_size])
            initially_ready = {
                (job_id, int(op["operation_index"]))
                for job_id, op in backend.ready_operations().items()
            }
            seen_jobs: set[str] = set()
            item_results: list[dict[str, Any]] = []
            scheduled_count = 0
            for item in processable:
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
            item_results.extend(
                {
                    "_status": "error",
                    "error": "batch_size_exceeded",
                    "max_batch_size": max_batch_size,
                }
                for _ in overflow
            )
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
                "job_shop_tool_call",
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


def _h_dispatch_job_operation(backend: JsplibJobShopBackend):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        result = backend.schedule_next_operation(
            job_id=str(args.get("job_id", "")),
            operation_index=int(args.get("operation_index", -1)),
        )
        evidence = ctx.extra.get("evidence")
        if isinstance(evidence, EvidenceLogger):
            evidence_id = evidence.log(
                "job_shop_tool_call",
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


def _h_repair_machine(backend: JsplibJobShopBackend):
    def handler(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        result = backend.repair_machine(machine_id=args.get("machine_id", -1))
        evidence = ctx.extra.get("evidence")
        if isinstance(evidence, EvidenceLogger):
            evidence_id = evidence.log(
                "job_shop_tool_call",
                ctx.tick,
                payload={
                    "tool": "repair_machine",
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
            "job_shop_tool_call",
            ctx.tick,
            payload={"tool": "wait", "ok": True, **result},
            source="tool",
        )
    return result


def _reference_optimum_from_seed(seed: LogisticsScenarioSeed) -> float | None:
    reference = seed.backend_config.get("reference") or {}
    try:
        if reference.get("type") == "known_optimum":
            return float(reference["makespan"])
        if reference.get("type") == "best_known_bounds":
            return float(reference["lower_bound"])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _canonical_j2_runtime_enabled(seed: LogisticsScenarioSeed) -> bool:
    """Return whether a REALM J2 JSON lock is the canonical runtime source."""
    config = seed.backend_config
    assets = config.get("external_source_assets") or {}
    lock = assets.get("j2_event_sidecar") if isinstance(assets, dict) else None
    return bool(
        config.get("source_mode") == "realm_j2_json"
        and isinstance(lock, dict)
        and lock.get("canonical_runtime_source") is True
    )


def _load_j2_event_sidecar(
    seed: LogisticsScenarioSeed,
    parsed: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Open and validate an optional REALM J2 event sidecar at runtime.

    The sidecar is an independently locked runtime input.  Its selected row
    must reproduce the raw JSSP operation graph and the source-observed
    machine-breakdown perturbation already declared by the YAML seed.  Any
    missing lock, changed bytes, row ambiguity, graph drift, or event drift is
    rejected before the backend can enter a replay.
    """
    backend_config = seed.backend_config
    assets = backend_config.get("external_source_assets") or {}
    if not isinstance(assets, dict) or "j2_event_sidecar" not in assets:
        return None
    lock = assets.get("j2_event_sidecar")
    if not isinstance(lock, dict):
        raise ValueError("j2_event_sidecar_lock_invalid")

    raw_path = lock.get("path")
    expected_sha256 = str(lock.get("sha256") or "").removeprefix("sha256:")
    git_commit = str(lock.get("git_commit") or "")
    selected_instance_id = str(lock.get("selected_instance_id") or "")
    if (
        not raw_path
        or not expected_sha256
        or not git_commit
        or not selected_instance_id
    ):
        raise ValueError("j2_event_sidecar_lock_incomplete")
    canonical_runtime = _canonical_j2_runtime_enabled(seed)
    if canonical_runtime and str(lock.get("license") or "") != "CC-BY-4.0":
        raise ValueError("realm_j2_canonical_license_missing")
    provenance_commit = str(seed.provenance.commit or "")
    if not provenance_commit or git_commit != provenance_commit:
        raise ValueError(
            "j2_event_sidecar_commit_mismatch: "
            f"asset={git_commit or 'missing'} provenance={provenance_commit or 'missing'}"
        )

    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"j2_event_sidecar_path_missing: {raw_path}")
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "j2_event_sidecar_hash_mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("j2_event_sidecar_json_invalid") from exc
    rows = payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("j2_event_sidecar_instances_missing")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and str(row.get("instance_id") or "") == selected_instance_id
    ]
    if len(matches) != 1:
        raise ValueError("j2_event_sidecar_selected_instance_mismatch")
    row = matches[0]
    if str(row.get("tier") or "") != "J2":
        raise ValueError("j2_event_sidecar_selected_instance_mismatch")

    graph = _normalize_j2_operation_graph(row)
    sidecar_parsed = _parsed_from_j2_graph(row, graph)
    if parsed is not None:
        if graph != parsed.get("jobs_detail"):
            raise ValueError("j2_event_sidecar_operation_graph_mismatch")
        if int(row.get("num_jobs", 0) or 0) != int(parsed.get("jobs", 0) or 0):
            raise ValueError("j2_event_sidecar_operation_graph_mismatch")
        if int(row.get("num_machines", 0) or 0) != int(parsed.get("machines", 0) or 0):
            raise ValueError("j2_event_sidecar_operation_graph_mismatch")
    elif not canonical_runtime:
        raise ValueError("j2_event_sidecar_operation_graph_mismatch")

    breakdown = _normalize_j2_breakdown(row)
    parsed_for_checks = parsed if parsed is not None else sidecar_parsed
    machine_ids = {int(value) for value in parsed_for_checks.get("machine_ids") or []}
    if breakdown["machine_id"] not in machine_ids:
        raise ValueError("j2_event_sidecar_breakdown_mismatch")
    perturbation = _matching_j2_perturbation(
        seed,
        expected_sha256=expected_sha256,
        selected_instance_id=selected_instance_id,
    )
    if perturbation is None:
        raise ValueError("j2_event_sidecar_breakdown_mismatch")
    target = getattr(perturbation, "target", {}) or {}
    try:
        yaml_breakdown = {
            "machine_id": int(target["machine_id"]),
            "trigger_tick": int(perturbation.trigger_tick),
            "duration_ticks": int(perturbation.duration_ticks),
        }
    except (KeyError, TypeError, ValueError):
        raise ValueError("j2_event_sidecar_breakdown_mismatch") from None
    if yaml_breakdown != breakdown:
        raise ValueError("j2_event_sidecar_breakdown_mismatch")
    if "source_sidecar_sha256" in target:
        target_sha = str(target.get("source_sidecar_sha256") or "").removeprefix(
            "sha256:"
        )
        if target_sha != expected_sha256:
            raise ValueError("j2_event_sidecar_breakdown_mismatch")
    if (
        "source_instance_id" in target
        and str(target.get("source_instance_id")) != selected_instance_id
    ):
        raise ValueError("j2_event_sidecar_breakdown_mismatch")

    sidecar_path = str(raw_path)
    event_trace = {
        "source": "j2_event_sidecar",
        "event_type": "machine_breakdown",
        "selected_instance_id": selected_instance_id,
        "machine_id": breakdown["machine_id"],
        "trigger_tick": breakdown["trigger_tick"],
        "duration_ticks": breakdown["duration_ticks"],
        "state_effect_observed": False,
        "state_effect_applied": False,
        "state_effect_expected": True,
        "state_effect_fields": [
            "active_machine_disruptions",
            "machine_available_at",
            "ready_operations",
        ],
    }
    selected_row_sha256 = _digest(row)
    source_event_id = "realm-j2:" + _digest(
        {
            "source_sha256": actual_sha256,
            "selected_row_sha256": selected_row_sha256,
            "selected_instance_id": selected_instance_id,
            "breakdown": breakdown,
        }
    )
    event_trace["event_id"] = source_event_id
    runtime_asset = {
        "path": sidecar_path,
        "sha256": actual_sha256,
        "role": "runtime_input",
        "git_commit": git_commit,
        "selected_instance_id": selected_instance_id,
    }
    if canonical_runtime:
        runtime_asset.update(
            {
                "canonical_runtime_source": True,
                "license": str(lock["license"]),
            }
        )
    return {
        "path": sidecar_path,
        "resolved_path": str(path),
        "sha256": actual_sha256,
        "git_commit": git_commit,
        "selected_instance_id": selected_instance_id,
        "operation_graph_digest": _digest(graph),
        "source_event_id": source_event_id,
        "breakdown": breakdown,
        "event_trace": event_trace,
        "runtime_asset": runtime_asset,
        "parsed": sidecar_parsed,
    }


def _parsed_from_j2_graph(
    row: dict[str, Any], graph: list[list[dict[str, int]]]
) -> dict[str, Any]:
    """Convert one locked REALM J2 row to the native JSPLIB parsed shape."""
    try:
        jobs = int(row["num_jobs"])
        machines = int(row["num_machines"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("j2_event_sidecar_operation_graph_mismatch") from None
    if jobs <= 0 or machines <= 0 or len(graph) != jobs:
        raise ValueError("j2_event_sidecar_operation_graph_mismatch")
    processing_times = [
        int(operation["duration"]) for job in graph for operation in job
    ]
    machine_ids = sorted(
        {int(operation["machine"]) for job in graph for operation in job}
    )
    if not processing_times or any(
        machine < 0 or machine >= machines
        for job in graph
        for machine in (int(operation["machine"]) for operation in job)
    ):
        raise ValueError("j2_event_sidecar_operation_graph_mismatch")
    return {
        "jobs": jobs,
        "machines": machines,
        "operations": len(processing_times),
        "machine_ids": machine_ids,
        "total_processing_time": int(sum(processing_times)),
        "min_processing_time": int(min(processing_times)),
        "max_processing_time": int(max(processing_times)),
        "jobs_detail": graph,
    }


def _normalize_j2_operation_graph(row: dict[str, Any]) -> list[list[dict[str, int]]]:
    jobs = row.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("j2_event_sidecar_operation_graph_mismatch")
    normalized: list[list[dict[str, int]]] = []
    try:
        for job in jobs:
            if not isinstance(job, list):
                raise ValueError
            normalized.append(
                [
                    {
                        "machine": int(operation["machine"]) - 1,
                        "duration": int(operation["processing_time"]),
                    }
                    for operation in job
                ]
            )
    except (KeyError, TypeError, ValueError):
        raise ValueError("j2_event_sidecar_operation_graph_mismatch") from None
    if any(
        operation["machine"] < 0 or operation["duration"] <= 0
        for job in normalized
        for operation in job
    ):
        raise ValueError("j2_event_sidecar_operation_graph_mismatch")
    return normalized


def _normalize_j2_breakdown(row: dict[str, Any]) -> dict[str, int]:
    disruptions = row.get("disruptions")
    if not isinstance(disruptions, list):
        raise ValueError("j2_event_sidecar_breakdown_mismatch")
    matches = [
        item
        for item in disruptions
        if isinstance(item, dict) and item.get("type") == "machine_breakdown"
    ]
    if len(matches) != 1:
        raise ValueError("j2_event_sidecar_breakdown_mismatch")
    try:
        machine_id = int(matches[0]["machine_id"]) - 1
        trigger_tick = int(matches[0]["start_time"])
        duration_ticks = int(matches[0]["duration"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("j2_event_sidecar_breakdown_mismatch") from None
    if machine_id < 0 or trigger_tick <= 0 or duration_ticks <= 0:
        raise ValueError("j2_event_sidecar_breakdown_mismatch")
    return {
        "machine_id": machine_id,
        "trigger_tick": trigger_tick,
        "duration_ticks": duration_ticks,
    }


def _matching_j2_perturbation(
    seed: LogisticsScenarioSeed,
    *,
    expected_sha256: str,
    selected_instance_id: str,
) -> Any | None:
    events = [
        event
        for event in list(getattr(seed, "perturbations", []) or [])
        if str(getattr(event, "kind", "")) == "machine_breakdown"
    ]
    source_events = []
    for event in events:
        target = getattr(event, "target", {}) or {}
        target_sha = str(target.get("source_sidecar_sha256") or "").removeprefix(
            "sha256:"
        )
        if (
            target.get("source_observed") is True
            or target_sha == expected_sha256
            or str(target.get("source_instance_id") or "") == selected_instance_id
        ):
            source_events.append(event)
    if source_events:
        return source_events[0] if len(source_events) == 1 else None
    return events[0] if len(events) == 1 else None


def _resolve_instance_source(
    seed: LogisticsScenarioSeed,
    instance_id: str,
) -> tuple[Path, str]:
    candidates: list[tuple[Path, str]] = []
    for raw in seed.provenance.files:
        label = str(raw)
        if Path(label).name != instance_id:
            continue
        path = Path(label).expanduser()
        if not path.is_absolute():
            path = _REPO_ROOT / path
        if path.is_file():
            candidates.append((path.resolve(), label))
    if len(candidates) != 1:
        raise ValueError("source_instance_path_missing")
    return candidates[0]
