"""Datacenter scheduling POMDP adapter for locked Alibaba trace windows."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from core import (
    Action,
    BeliefStateTracker,
    EthicalDilemmaManager,
    EvidenceLogger,
    FogOfWarPolicy,
    POMDPEnvironment,
    StakeholderTrustManager,
    StepInfo,
    StepReturn,
    TickBudget,
    ToolContext,
    ToolRegistry,
    safe_dataclass_to_dict,
)
from core.difficulty_levels import canonical_difficulty_level
from core.evidence import control_summary_from_evidence
from core.world_evolution_contract import canonicalize_runtime_events
from domains.registry import apply_supervisory_cadence

from .backends.alibaba_openb_backend import AlibabaOpenBBackend
from .backends.alibaba_trace_backend import AlibabaTraceBackend
from .native_stakeholders import build_stakeholder_groups
from .native_tools import register_datacenter_tools
from .seeds.schema import (
    DatacenterPerturbation,
    DatacenterScenarioSeed,
    JobStakeholder,
    Provenance,
)


def _authoritative_source_event(
    event: dict[str, Any],
    evidence_id: str,
    visible_source_events_by_evidence_id: dict[str, dict[str, Any]],
) -> None:
    """Attach logger evidence and retain only passively visible source events."""

    if str(event.get("origin") or "") not in {
        "source_schedule",
        "source_trace",
        "declared_perturbation",
    }:
        return
    event.setdefault("event_id", f"datacenter-source-event:{evidence_id}")
    evidence_ids = event.setdefault("evidence_ids", [])
    if evidence_id not in evidence_ids:
        evidence_ids.append(evidence_id)
    if event.get("hidden") is True:
        return
    visible_source_events_by_evidence_id[evidence_id] = {
        "event_id": str(event["event_id"]),
        "visible_from_request_tick": int(event.get("tick") or 0) + 1,
    }


def _visible_causal_parent_event_id(
    action: Action,
    call_id: str | None,
    request_tick: int,
    visible_source_events_by_evidence_id: dict[str, dict[str, Any]],
) -> str | None:
    """Resolve an explicitly consumed, already-visible source event."""

    call = next(
        (candidate for candidate in action.tool_calls if candidate.call_id == call_id),
        None,
    )
    if call is None:
        return None
    for evidence_id in call.consumes_evidence_ids or []:
        event = visible_source_events_by_evidence_id.get(str(evidence_id))
        if event is not None and int(event["visible_from_request_tick"]) <= int(
            request_tick
        ):
            return str(event["event_id"])
    return None


class DatacenterEnvironment(POMDPEnvironment):
    """Interactive GPU cluster scheduler with deterministic counterfactual replay."""

    domain = "datacenter"

    def __init__(self) -> None:
        self._seed_obj: DatacenterScenarioSeed | None = None
        self._tick = 0
        self._horizon = 8
        self._backend: AlibabaTraceBackend | AlibabaOpenBBackend | None = None
        self._fog: FogOfWarPolicy | None = None
        self._belief: BeliefStateTracker | None = None
        self._tools: ToolRegistry | None = None
        self._stakeholders: StakeholderTrustManager | None = None
        self._dilemmas: EthicalDilemmaManager | None = None
        self._evidence: EvidenceLogger | None = None
        self._episode_id = ""
        self._last_runtime_events: list[dict[str, Any]] = []
        self._visible_runtime_event_history: list[dict[str, Any]] = []
        self._visible_source_events_by_evidence_id: dict[str, dict[str, Any]] = {}

    def reset(self, scenario_config: dict[str, Any], seed: int) -> dict[str, Any]:
        seed_obj = _rebuild_seed_from_dict(scenario_config, override_seed=seed)
        if seed_obj.backend_kind not in {
            "alibaba_trace_sim",
            "alibaba_openb_gpu_placement",
        }:
            raise ValueError(
                f"unknown datacenter backend_kind: {seed_obj.backend_kind}"
            )
        self._seed_obj = seed_obj
        self._tick = 0
        self._horizon = seed_obj.horizon_ticks
        self._episode_id = f"dt_{seed_obj.signature()}_s{seed}"
        self._last_runtime_events = []
        self._visible_runtime_event_history = []
        self._visible_source_events_by_evidence_id = {}

        self._backend = (
            AlibabaOpenBBackend()
            if seed_obj.backend_kind == "alibaba_openb_gpu_placement"
            else AlibabaTraceBackend()
        )
        self._backend.reset(seed_obj)
        self._fog = FogOfWarPolicy(
            hide_rules=[], noise_rules=[], staleness_rules=[], seed=seed
        )
        self._fog.reset(seed=seed)
        self._belief = BeliefStateTracker()
        self._belief.reset()
        self._tools = ToolRegistry(
            budget=_build_tick_budget(seed_obj),
            seed=seed,
            difficulty_level=seed_obj.difficulty_level,
        )
        self._tools.reset(seed=seed)
        register_datacenter_tools(self._tools, self._backend, self)
        self._stakeholders = StakeholderTrustManager()
        for group in build_stakeholder_groups(seed_obj, self._backend.tenant_ids()):
            self._stakeholders.register(group)
        self._dilemmas = EthicalDilemmaManager()
        self._dilemmas.reset()
        self._evidence = EvidenceLogger(episode_id=self._episode_id)

        observation = self.snapshot()
        self._belief.update_from_observation(observation, tick=0)
        return observation

    def step(self, action: Action) -> StepReturn:
        assert self._tools is not None and self._backend is not None
        assert self._evidence is not None and self._belief is not None
        context = ToolContext(
            tick=self._tick,
            seed=int(self._seed_obj.seed if self._seed_obj else 0),
            backend=self._backend,
            extra={
                "fog": self._fog,
                "stakeholders": self._stakeholders,
                "dilemmas": self._dilemmas,
                "evidence": self._evidence,
                "env": self,
            },
        )
        tool_results = self._tools.execute_action(
            action,
            context,
            begin_tick=not self._consume_within_tick_budget_state(),
        )
        for result in tool_results:
            linked = result.evidence_id
            payload = {
                "name": result.name,
                "ok": result.ok,
                "error_code": result.error_code,
                "cost_units": result.cost_units,
                "call_id": result.call_id,
                "state_changing": result.state_changing,
                "payload": result.payload,
                **self.tool_dependency_payload(action, result),
            }
            if linked:
                payload["linked_result_evidence_id"] = linked
            evidence_id = self._evidence.log(
                kind="tool_call",
                tick=self._tick,
                payload=payload,
                source="tool",
            )
            if not linked:
                result.evidence_id = evidence_id
            if result.ok:
                causal_parent_event_id = _visible_causal_parent_event_id(
                    action,
                    result.call_id,
                    self._tick,
                    self._visible_source_events_by_evidence_id,
                )
                self._backend.bind_tool_result(
                    name=result.name,
                    call_id=result.call_id,
                    evidence_id=result.evidence_id,
                    payload=result.payload,
                    causal_parent_event_id=causal_parent_event_id,
                )

        record = self._backend.tick(self._tick)
        record_payload = safe_dataclass_to_dict(record)
        self._evidence.log(
            kind="backend_tick",
            tick=self._tick,
            payload=record_payload,
            source="engine",
        )
        self._evidence.log(
            kind="cost_summary",
            tick=self._tick,
            payload=self._backend.ground_truth_costs(),
            source="engine",
        )
        for event in record.realized_events:
            evidence_id = self._evidence.log(
                kind="realized_event",
                tick=self._tick,
                payload=dict(event),
                source="engine",
            )
            _authoritative_source_event(
                event,
                evidence_id,
                self._visible_source_events_by_evidence_id,
            )
        self._last_runtime_events = [
            {
                key: event[key]
                for key in (
                    "event_id",
                    "type",
                    "origin",
                    "tick",
                    "changed_state_fields",
                    "materiality_metric",
                    "materiality_value",
                    "materiality_threshold",
                    "materiality_passed",
                    "decision_required",
                    "evidence_ids",
                )
                if key in event
            }
            for event in record.realized_events
            if isinstance(event, dict) and event.get("hidden") is not True
        ]
        known_event_ids = {
            str(event.get("event_id"))
            for event in self._visible_runtime_event_history
            if event.get("event_id")
        }
        for event in self._last_runtime_events:
            event_id = str(event.get("event_id") or "")
            if event_id and event_id not in known_event_ids:
                self._visible_runtime_event_history.append(dict(event))
                known_event_ids.add(event_id)
        world_evolution_records = canonicalize_runtime_events(
            record.realized_events,
            applied_tick=self._tick,
        )
        self._update_stakeholders(record.realized_events)
        if self._stakeholders is not None:
            self._stakeholders.tick(self._tick)

        self._tick += 1
        done = bool(record.done or self._tick >= self._horizon)
        observation = self.snapshot()
        observation["tick"] = self._tick
        self._belief.update_from_observation(observation, tick=self._tick)
        reward = (
            -(
                record.queue_wait_cost
                + record.sla_violation_cost
                + record.preemption_waste_cost
                + record.reserve_capacity_cost
            )
            / 100.0
        )
        return StepReturn(
            observation=observation,
            tool_results=tool_results,
            reward=float(reward),
            done=done,
            info=StepInfo(
                realized_events=list(record.realized_events),
                evidence_ids=[
                    item.evidence_id
                    for item in self._evidence.items()
                    if item.tick == self._tick - 1
                ],
                extra={
                    "backend_tick_record": record_payload,
                    "world_evolution_records": world_evolution_records,
                },
            ),
        )

    def snapshot(self) -> dict[str, Any]:
        assert self._backend is not None
        observation = self._backend.snapshot()
        backend_kind = str(getattr(self._backend, "backend_kind", "") or "")
        if backend_kind:
            with suppress(KeyError):
                observation["decision_cadence"] = apply_supervisory_cadence(
                    backend_kind,
                    observation.get("decision_cadence"),
                )
        if self._stakeholders is not None:
            observation["stakeholder_trust"] = {
                group_id: {
                    "trust": reading.trust,
                    "tier": reading.tier,
                }
                for group_id, reading in self._stakeholders.snapshot().items()
            }
        if self._fog is not None:
            self._fog.set_tick(self._tick)
            observation = self._fog.filter(observation)
        observation["runtime_events"] = [
            dict(event) for event in self._visible_runtime_event_history
        ]
        return observation

    def ground_truth(self) -> dict[str, Any]:
        assert self._backend is not None
        truth = self._backend.snapshot()
        truth["per_job_sla_violation_minutes"] = (
            self._backend.per_job_sla_violation_minutes()
        )
        truth["cost_components"] = self._backend.ground_truth_costs()
        backend_summary = self._backend.control_summary()
        strict_summary = (
            control_summary_from_evidence(self._evidence)
            if self._evidence is not None
            else {
                "distinct_control_ticks": [],
                "distinct_physical_tools": [],
                "tool_ticks": {},
                "effect_tool_ticks": {},
            }
        )
        backend_tools = set(backend_summary.get("distinct_physical_tools") or [])
        strict_tools = set(strict_summary["distinct_physical_tools"])
        backend_summary.update(strict_summary)
        if backend_tools != strict_tools:
            backend_summary["distinct_physical_actuator_endpoints"] = []
        truth["control_summary"] = backend_summary
        if self._stakeholders is not None:
            truth["stakeholder_trust"] = {
                group_id: reading.trust
                for group_id, reading in self._stakeholders.snapshot().items()
            }
            truth["stakeholder_equity_gini"] = round(
                self._stakeholders.equity_gini(), 4
            )
        return truth

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def budget(self) -> TickBudget:
        return self._tools._budget if self._tools else TickBudget(horizon=self._horizon)

    def get_tool_specs(self) -> list[dict[str, Any]]:
        return self._tools.openai_schemas() if self._tools else []

    @property
    def evidence(self) -> EvidenceLogger | None:
        return self._evidence

    @property
    def stakeholders(self) -> StakeholderTrustManager | None:
        return self._stakeholders

    @property
    def dilemmas(self) -> EthicalDilemmaManager | None:
        return self._dilemmas

    @property
    def seed_obj(self) -> DatacenterScenarioSeed | None:
        return self._seed_obj

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def close(self) -> None:
        self._backend = None

    def _update_stakeholders(self, realized_events: list[dict[str, Any]]) -> None:
        assert self._backend is not None
        assert self._stakeholders is not None
        assert self._evidence is not None
        completed_users = {
            str(event.get("user") or "unknown")
            for event in realized_events
            if event.get("type") in {"job_completed", "pod_completed"}
        }
        overdue_users = {
            str(job.get("user") or "unknown")
            for job in (self._backend.snapshot().get("jobs") or {}).values()
            if job.get("status") == "queued"
            and self._tick > int(job.get("due_tick") or 0)
        }
        for group_id in sorted(completed_users | overdue_users):
            event = (
                "delayed_response" if group_id in overdue_users else "timely_response"
            )
            trust = self._stakeholders.record_event(group_id, event, self._tick)
            self._evidence.log(
                kind="trust_event",
                tick=self._tick,
                payload={
                    "group_id": group_id,
                    "event": event,
                    "trust": trust,
                },
                source="engine",
            )


def _build_tick_budget(seed_obj: DatacenterScenarioSeed) -> TickBudget:
    per_tick = {"basic": 5, "medium": 7, "high": 9, "extreme": 11}.get(
        canonical_difficulty_level(seed_obj.difficulty_level), 7
    )
    return TickBudget(
        max_tool_calls_per_tick=per_tick,
        max_cost_units_per_tick=max(2.0, per_tick * 0.75),
        max_total_tool_calls=per_tick * seed_obj.horizon_ticks,
        duplicate_suppression_window=2,
        cooldown_after_failure=1,
    )


def _rebuild_seed_from_dict(
    data: dict[str, Any], override_seed: int
) -> DatacenterScenarioSeed:
    return DatacenterScenarioSeed(
        seed_id=str(data.get("seed_id") or "anon"),
        family=str(data.get("family") or "gpu_cluster_scheduling"),
        domain=str(data.get("domain") or "datacenter"),
        backend_kind=str(data.get("backend_kind") or "alibaba_trace_sim"),
        backend_config=dict(data.get("backend_config") or {}),
        horizon_ticks=int(data.get("horizon_ticks") or 8),
        tick_minutes=int(data.get("tick_minutes") or 30),
        seed=int(override_seed),
        load_assignments=[
            JobStakeholder(**row) for row in data.get("load_assignments") or []
        ],
        perturbations=[
            DatacenterPerturbation(**row) for row in data.get("perturbations") or []
        ],
        dilemmas=list(data.get("dilemmas") or []),
        difficulty_mode=data.get("difficulty_mode") or "time_pressure",
        difficulty_level=data.get("difficulty_level") or "basic",
        provenance=Provenance(
            **(
                data.get("provenance")
                or {"data_source": "alibaba_cluster_trace_gpu_v2020"}
            )
        ),
    )
