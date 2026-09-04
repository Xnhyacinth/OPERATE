"""
domains.logistics.adapter — Logistics fleet-dispatch POMDP environment.

Mirrors the structure of ``domains.disaster.adapter`` /
``domains.power_grid.adapter`` — backend + fog + belief + tool registry +
(family-gated) stakeholder manager + dilemma manager + evidence logger +
cascade-bus subscriber — but uses logistics-native types and never imports
from another domain (Red Line #3).

The backend is selected by ``backend_kind`` (``pyvrp_cvrp`` /
``pyvrp_vrptw`` / ``pyvrp_lastmile``); all three are the same pure-Python
``RouteDemandSimulator`` with different family flags, so stepping never
requires PyVRP. PyVRP is used only on the optional fixed-plan cost-eval path.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from core import (
    Action,
    BeliefStateTracker,
    CascadeBus,
    CascadeEvent,
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
    arm_dilemmas,
    safe_dataclass_to_dict,
)
from core.difficulty_levels import canonical_difficulty_level
from core.evidence import control_summary_from_evidence
from core.world_evolution_contract import canonicalize_runtime_events
from domains.registry import apply_supervisory_cadence

from .backends.pyvrp_cvrp import PyvrpCvrpBackend
from .backends.pyvrp_lastmile import PyvrpLastmileBackend
from .backends.pyvrp_vrptw import PyvrpVrptwBackend
from .oracle import compute_reference_optimum
from .seeds.schema import (
    CustomerPriority,
    DilemmaSeed,
    LogisticsScenarioSeed,
    Perturbation,
    Provenance,
)

# Backend-emitted event type → canonical cascade event_type.
_CASCADE_EVENT_MAPPING: dict[str, str] = {
    "vehicle_breakdown": "logistics.vehicle.breakdown",
    "blocked_arc": "logistics.arc.blocked",
    "demand_surge": "logistics.demand.surge",
    "urgent_order": "logistics.order.urgent",
    "traffic_delay": "logistics.traffic.delay",
    "vehicle_dispatched": "logistics.vehicle.dispatched",
    "spot_carrier_arrived": "logistics.carrier.arrived",
    "fuel_delivery_delay": "logistics.fuel.delay",
}


def _visible_causal_parent_event_id(
    action: Action,
    call_id: str | None,
    visible_events_by_evidence_id: dict[str, str],
    *,
    materialized_consumes_evidence_ids: list[str] | None = None,
) -> str | None:
    """Resolve one explicitly consumed, previously visible source event."""
    call = next(
        (
            candidate
            for candidate in action.tool_calls
            if str(candidate.call_id or "") == str(call_id or "")
        ),
        None,
    )
    consumed = (
        list(call.consumes_evidence_ids or [])
        if call is not None
        else list(materialized_consumes_evidence_ids or [])
    )
    parents = {
        visible_events_by_evidence_id[evidence_id]
        for evidence_id in consumed
        if evidence_id in visible_events_by_evidence_id
    }
    return next(iter(parents)) if len(parents) == 1 else None


class LogisticsEnvironment(POMDPEnvironment):
    """OPERATE environment for the logistics / VRP-dispatch domain."""

    domain = "logistics"

    def __init__(self, cascade_bus: CascadeBus | None = None) -> None:
        self._seed_obj: LogisticsScenarioSeed | None = None
        self._tick = 0
        self._horizon = 8
        self._backend: Any = None
        self._fog: FogOfWarPolicy | None = None
        self._belief: BeliefStateTracker | None = None
        self._tools: ToolRegistry | None = None
        self._stakeholders: StakeholderTrustManager | None = None
        self._dilemmas: EthicalDilemmaManager | None = None
        self._evidence: EvidenceLogger | None = None
        self._cascade_bus = cascade_bus or CascadeBus()
        self._episode_id = ""
        self._pending_cascade_perturbations: list[Perturbation] = []
        self._visible_source_events_by_evidence_id: dict[str, str] = {}
        self._cascade_bus.subscribe(
            "power_grid.line.outage", self._on_power_grid_line_outage
        )
        self._cascade_bus.subscribe(
            "microgrid.storage.outage", self._on_microgrid_storage_outage
        )
        self._cascade_bus.subscribe(
            "microgrid.der.failure", self._on_microgrid_storage_outage
        )

    # ── POMDPEnvironment surface ────────────────────────────────────────

    def reset(self, scenario_config: dict[str, Any], seed: int) -> dict[str, Any]:
        seed_obj = _rebuild_seed_from_dict(scenario_config, override_seed=seed)
        self._seed_obj = seed_obj
        self._tick = 0
        self._horizon = seed_obj.horizon_ticks
        self._episode_id = f"dt_{seed_obj.signature()}_s{seed}"
        self._pending_cascade_perturbations = []
        self._visible_source_events_by_evidence_id = {}

        self._backend = _build_backend(seed_obj)
        self._backend.reset(seed_obj)

        # Cache the deterministic routing reference optimum so the scorer's
        # optimality_gap reads a replay-stable cost floor off
        # ``backend_config['reference_optimum']`` using the bounded,
        # dependency-invariant routing reference. Pure function of the seed.
        try:
            if seed_obj.backend_kind not in {
                "jsplib_job_shop",
                "co_bench_job_shop",
                "dynasched_flexible_job_shop",
                "orgym_invmgmt",
            } and not seed_obj.backend_config.get("reference_optimum"):
                compute_reference_optimum(seed_obj, cache=True)
        except Exception:
            pass
        if seed_obj.backend_kind == "jsplib_job_shop":
            _cache_job_shop_reference_optimum(seed_obj)
        if seed_obj.backend_kind == "co_bench_job_shop":
            _cache_co_bench_job_shop_reference_optimum(seed_obj)

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
        _register_native_tools(self._tools, self._backend, self)

        self._stakeholders = StakeholderTrustManager()
        _register_native_stakeholders(self._stakeholders, seed_obj)

        self._dilemmas = EthicalDilemmaManager()
        self._dilemmas.reset()
        arm_dilemmas(self._dilemmas, seed_obj.dilemmas)

        self._evidence = EvidenceLogger(episode_id=self._episode_id)

        obs = self.snapshot()
        self._belief.update_from_observation(obs, tick=0)
        return obs

    def step(self, action: Action) -> StepReturn:
        assert self._tools is not None and self._backend is not None
        assert self._fog is not None and self._evidence is not None
        assert self._dilemmas is not None and self._belief is not None

        ctx = ToolContext(
            tick=self._tick,
            seed=int(self._seed_obj.seed if self._seed_obj else 0),
            backend=self._backend,
            extra={
                "fog": self._fog,
                "stakeholders": self._stakeholders,
                "dilemmas": self._dilemmas,
                "evidence": self._evidence,
                "cascade_bus": self._cascade_bus,
                "env": self,
            },
        )
        tool_results = self._tools.execute_action(
            action, ctx, begin_tick=not self._consume_within_tick_budget_state()
        )
        for r in tool_results:
            result_evidence_id = r.evidence_id
            tool_call_payload = {
                "name": r.name,
                "ok": r.ok,
                "error_code": r.error_code,
                "cost_units": r.cost_units,
                "call_id": r.call_id,
                "state_changing": r.state_changing,
                "payload": r.payload,
                **self.tool_dependency_payload(action, r),
            }
            if result_evidence_id:
                tool_call_payload["linked_result_evidence_id"] = result_evidence_id
            tool_call_evidence_id = self._evidence.log(
                kind="tool_call",
                tick=self._tick,
                payload=tool_call_payload,
                source="tool",
            )
            if not result_evidence_id:
                r.evidence_id = tool_call_evidence_id
            bind_tool_result = getattr(
                self._backend, "bind_tool_result", None
            )
            if r.ok and callable(bind_tool_result):
                kwargs = {
                    "name": r.name,
                    "call_id": r.call_id,
                    "evidence_id": r.evidence_id,
                    "payload": r.payload,
                }
                if getattr(self._backend, "backend_kind", "") == "orgym_invmgmt":
                    kwargs["causal_parent_event_id"] = (
                        _visible_causal_parent_event_id(
                            action,
                            r.call_id,
                            self._visible_source_events_by_evidence_id,
                            materialized_consumes_evidence_ids=list(
                                r.consumes_evidence_ids or []
                            ),
                        )
                    )
                bind_tool_result(**kwargs)

        if self._pending_cascade_perturbations and self._seed_obj is not None:
            for p in self._pending_cascade_perturbations:
                self._seed_obj.perturbations.append(p)
                self._evidence.log(
                    kind="cascade_effect",
                    tick=self._tick,
                    payload={
                        "kind": p.kind,
                        "intensity": p.intensity,
                        "source": p.target.get("source_event", "cascade_bus"),
                    },
                    source="cascade_bus",
                )
            self._pending_cascade_perturbations = []

        backend_tick_record = self._backend.tick(self._tick)
        backend_tick_payload = safe_dataclass_to_dict(backend_tick_record)
        if not isinstance(backend_tick_payload, dict):
            backend_tick_payload = {}

        def record_value(key: str, default: Any) -> Any:
            return backend_tick_payload.get(key, default)

        self._evidence.log(
            kind="backend_tick",
            tick=self._tick,
            payload=backend_tick_payload,
            source="engine",
        )
        self._evidence.log(
            kind="cost_summary",
            tick=self._tick,
            payload={
                "routing_operating_cost": float(
                    record_value("routing_cost", 0.0)
                ),
                "vehicle_dispatch_fixed_cost": float(
                    record_value("dispatch_fixed_cost", 0.0)
                ),
                "drop_order_penalty": float(
                    record_value("drop_penalty", 0.0)
                ),
                "unmet_demand_units": float(
                    record_value("unmet_demand", 0.0)
                ),
            },
            source="engine",
        )
        if float(record_value("unmet_demand", 0.0)) > 0.0:
            self._evidence.log(
                kind="unmet_customer_demand",
                tick=self._tick,
                payload={
                    "unmet_demand_units": float(
                        record_value("unmet_demand", 0.0)
                    ),
                    "per_customer_unmet_units": {
                        cid: units
                        for cid, units in self._backend.per_customer_unmet_units().items()
                        if float(units) > 0.0
                    },
                },
                source="engine",
            )
        for ev in record_value("realized_events", []) or []:
            is_source_schedule = (
                str(ev.get("origin") or "") == "source_schedule"
            )
            requires_authoritative_source_binding = (
                is_source_schedule
                and getattr(self._backend, "backend_kind", "")
                == "orgym_invmgmt"
            )
            event_evidence_ids = [
                str(item)
                for item in ev.get("evidence_ids") or []
                if str(item)
            ]
            if requires_authoritative_source_binding:
                source_event_evidence_id = self._evidence.log(
                    kind="source_schedule_event",
                    tick=self._tick,
                    payload=dict(ev),
                    source="engine",
                )
                if source_event_evidence_id not in event_evidence_ids:
                    event_evidence_ids.append(source_event_evidence_id)
            ev["evidence_ids"] = event_evidence_ids
            evidence_id = self._evidence.log(
                kind="realized_event",
                tick=self._tick,
                payload=dict(ev),
                source="engine",
            )
            if (
                not requires_authoritative_source_binding
                and evidence_id not in event_evidence_ids
            ):
                event_evidence_ids.append(evidence_id)
            ev["evidence_ids"] = event_evidence_ids
            if (
                is_source_schedule
                and not bool(ev.get("hidden", False))
                and str(ev.get("event_id") or "")
            ):
                for event_evidence_id in event_evidence_ids:
                    self._visible_source_events_by_evidence_id[
                        event_evidence_id
                    ] = str(ev["event_id"])
            if str(ev.get("origin") or "") == "agent_caused":
                for result in tool_results:
                    if (
                        result.ok
                        and result.call_id
                        and str(result.call_id) == str(ev.get("call_id") or "")
                    ):
                        result.effect_tick = int(
                            ev.get("outcome_tick", self._tick)
                        )
                        produced = [
                            str(item)
                            for item in (
                                list(result.produces_evidence_ids or [])
                                + [result.evidence_id]
                                + event_evidence_ids
                            )
                            if item
                        ]
                        result.produces_evidence_ids = list(
                            dict.fromkeys(produced)
                        )
            self._publish_cascade_for_event(ev, evidence_id)
        world_evolution_records = canonicalize_runtime_events(
            list(record_value("realized_events", []) or []),
            applied_tick=self._tick,
        )

        # Dilemma triggers + deadline defaults (last-mile only in practice).
        current_snap = self.snapshot()
        triggered = self._dilemmas.maybe_trigger(self._tick, current_snap)
        for d in triggered:
            self._evidence.log(
                kind="dilemma_triggered",
                tick=self._tick,
                payload={"dilemma_id": d.dilemma_id, "description": d.description},
                source="engine",
            )
        defaults_fired = self._dilemmas.fire_defaults_for_missed_deadlines(self._tick)
        for choice in defaults_fired:
            self._evidence.log(
                kind="moral_choice",
                tick=self._tick,
                payload={
                    "dilemma_id": choice.dilemma_id,
                    "option_id": choice.chosen_option_id,
                    "rationale": choice.rationale,
                    "source": "default_option_fired",
                },
                source="engine",
            )

        if self._stakeholders is not None:
            self._stakeholders.tick(self._tick)

        self._tick += 1
        backend_done = bool(record_value("done", False))
        done = backend_done or self._tick >= self._horizon

        snap = self.snapshot()
        snap["tick"] = self._tick
        self._belief.update_from_observation(snap, tick=self._tick)

        reward = self._reward_signal(backend_tick_record)
        info = StepInfo(
            realized_events=list(
                record_value("realized_events", []) or []
            ),
            evidence_ids=[
                i.evidence_id
                for i in self._evidence.items()
                if i.tick == self._tick - 1
            ],
            extra={
                "dilemmas_triggered": [d.dilemma_id for d in triggered],
                "backend_tick_record": backend_tick_payload,
                "world_evolution_records": world_evolution_records,
            },
        )
        return StepReturn(
            observation=snap,
            tool_results=tool_results,
            reward=float(reward),
            done=done,
            info=info,
        )

    def snapshot(self) -> dict[str, Any]:
        assert self._backend is not None
        raw = self._backend.snapshot()
        if self._fog:
            self._fog.set_tick(self._tick)
            raw = self._fog.filter(raw)
        backend_kind = str(getattr(self._backend, "backend_kind", "") or "")
        if backend_kind:
            with suppress(KeyError):
                raw["decision_cadence"] = apply_supervisory_cadence(
                    backend_kind,
                    raw.get("decision_cadence"),
                )
        if self._stakeholders is not None and self._stakeholders.snapshot():
            raw["stakeholder_trust"] = {
                gid: {"trust": r.trust, "tier": r.tier}
                for gid, r in self._stakeholders.snapshot().items()
            }
        if self._dilemmas is not None:
            raw["active_dilemmas"] = [
                {
                    "dilemma_id": d.dilemma_id,
                    "description": d.description,
                    "options": [
                        {"option_id": o.option_id, "label": o.label, "fatal": o.fatal}
                        for o in d.options
                    ],
                    "deadline_tick": d.trigger_tick + d.resolution_deadline_ticks,
                }
                for d in self._dilemmas.record.dilemmas_triggered
                if d.dilemma_id not in self._dilemmas.record.choices
            ]
        return raw

    def ground_truth(self) -> dict[str, Any]:
        assert self._backend is not None and self._seed_obj is not None
        gt = self._backend.snapshot()
        gt["per_customer_unmet_units"] = self._backend.per_customer_unmet_units()
        gt["cost_components"] = self._backend.ground_truth_costs()
        if self._seed_obj.backend_config.get("task_requirements") and self._evidence:
            gt["control_summary"] = control_summary_from_evidence(
                self._evidence
            )
        if self._stakeholders is not None and self._stakeholders.snapshot():
            gt["stakeholder_trust"] = {
                gid: r.trust for gid, r in self._stakeholders.snapshot().items()
            }
        if self._dilemmas is not None:
            gt["dilemmas_triggered"] = [
                d.dilemma_id for d in self._dilemmas.record.dilemmas_triggered
            ]
            gt["choices"] = {
                did: {"option_id": c.chosen_option_id, "tick_chosen": c.tick_chosen}
                for did, c in self._dilemmas.record.choices.items()
            }
        return gt

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

    def close(self) -> None:
        pass

    # ── Public accessors (runner / scorer / tools) ──────────────────────

    @property
    def evidence(self) -> EvidenceLogger | None:
        return self._evidence

    @property
    def dilemmas(self) -> EthicalDilemmaManager | None:
        return self._dilemmas

    @property
    def stakeholders(self) -> StakeholderTrustManager | None:
        return self._stakeholders

    @property
    def cascade_bus(self) -> CascadeBus:
        return self._cascade_bus

    @property
    def seed_obj(self) -> LogisticsScenarioSeed | None:
        return self._seed_obj

    @property
    def episode_id(self) -> str:
        return self._episode_id

    # ── Internals ───────────────────────────────────────────────────────

    def _reward_signal(self, rec: Any) -> float:
        routing = float(getattr(rec, "routing_cost", 0.0))
        unmet = float(getattr(rec, "unmet_demand", 0.0))
        drop = float(getattr(rec, "drop_penalty", 0.0))
        return -(routing + unmet * 50.0 + drop) / 1000.0

    def _on_power_grid_line_outage(self, event: CascadeEvent) -> None:
        """Cascade-bus subscriber: grid outage -> logistics warehouse disruption."""
        severity = float(event.severity or 0.5)
        self._pending_cascade_perturbations.append(
            Perturbation(
                kind="traffic_delay",
                trigger_tick=max(self._tick, int(event.tick) + 1),
                duration_ticks=max(1, int(2 + 4 * severity)),
                hidden=False,
                target={
                    "region": (event.location or {}).get("region", "warehouse"),
                    "source_event": "power_grid.line.outage",
                },
                intensity=max(1.1, 1.0 + severity),
                notes=(
                    "Cascaded from power_grid.line.outage as warehouse power "
                    f"and yard-traffic disruption (correlation_id={event.correlation_id})"
                ),
            )
        )

    def _on_microgrid_storage_outage(self, event: CascadeEvent) -> None:
        """Cascade-bus subscriber: microgrid outage -> cold-chain logistics delay."""
        severity = float(event.severity or 0.5)
        self._pending_cascade_perturbations.append(
            Perturbation(
                kind="traffic_delay",
                trigger_tick=max(self._tick, int(event.tick) + 1),
                duration_ticks=max(1, int(2 + 3 * severity)),
                hidden=False,
                target={
                    "region": (event.location or {}).get("site", "cold_chain_hub"),
                    "source_event": event.event_type,
                },
                intensity=max(1.1, 1.0 + severity),
                notes=(
                    "Cascaded from microgrid outage as cold-chain loading delay "
                    f"(correlation_id={event.correlation_id})"
                ),
            )
        )

    def _publish_cascade_for_event(
        self, event: dict[str, Any], evidence_id: str
    ) -> None:
        canonical = _CASCADE_EVENT_MAPPING.get(str(event.get("type", "")))
        if canonical is None:
            return
        severity = float(event.get("intensity", 0.5) or 0.5)
        self._cascade_bus.publish(
            CascadeEvent(
                event_type=canonical,
                source_domain="logistics",
                tick=int(event.get("tick", self._tick)),
                severity=max(0.0, min(1.0, severity)),
                location={"region": event.get("region")},
                payload={**event, "evidence_id": evidence_id},
                correlation_id=evidence_id,
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factories / helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_backend(kind: str) -> Any:
    if isinstance(kind, LogisticsScenarioSeed):
        seed_obj = kind
        kind = seed_obj.backend_kind
    else:
        seed_obj = None
    if kind == "pyvrp_cvrp":
        return PyvrpCvrpBackend()
    if kind == "pyvrp_vrptw":
        return PyvrpVrptwBackend()
    if kind == "pyvrp_lastmile":
        return PyvrpLastmileBackend()
    if kind == "jsplib_job_shop":
        if seed_obj is None:
            raise ValueError("jsplib_job_shop backend requires a LogisticsScenarioSeed")
        from .backends.job_shop import JsplibJobShopBackend

        return JsplibJobShopBackend.from_seed(seed_obj)
    if kind == "co_bench_job_shop":
        if seed_obj is None:
            raise ValueError(
                "co_bench_job_shop backend requires a LogisticsScenarioSeed"
            )
        from .backends.co_bench_job_shop import CoBenchJobShopBackend

        return CoBenchJobShopBackend.from_seed(seed_obj)
    if kind == "dynasched_flexible_job_shop":
        if seed_obj is None:
            raise ValueError(
                "dynasched_flexible_job_shop backend requires a LogisticsScenarioSeed"
            )
        from .backends.dynasched_flexible_job_shop import (
            DynaSchedFlexibleJobShopBackend,
        )

        return DynaSchedFlexibleJobShopBackend.from_seed(seed_obj)
    if kind == "orgym_invmgmt":
        from .backends.orgym_invmgmt import OrgymInvmgmtBackend

        return OrgymInvmgmtBackend()
    raise ValueError(f"unknown logistics backend_kind: {kind}")


def _register_native_tools(
    reg: ToolRegistry, backend: Any, env: LogisticsEnvironment
) -> None:
    if getattr(backend, "backend_kind", "") == "jsplib_job_shop":
        from .backends.job_shop import register_job_shop_tools

        register_job_shop_tools(reg, backend, env)
        return
    if getattr(backend, "backend_kind", "") == "co_bench_job_shop":
        from .backends.co_bench_job_shop import register_co_bench_job_shop_tools

        register_co_bench_job_shop_tools(reg, backend)
        return
    if getattr(backend, "backend_kind", "") == "dynasched_flexible_job_shop":
        from .backends.dynasched_flexible_job_shop import (
            register_dynasched_flexible_job_shop_tools,
        )

        register_dynasched_flexible_job_shop_tools(reg, backend)
        return
    if getattr(backend, "backend_kind", "") == "orgym_invmgmt":
        from .backends.orgym_invmgmt import register_orgym_inventory_tools

        register_orgym_inventory_tools(reg, backend)
        return
    from .native_tools import register_logistics_tools

    register_logistics_tools(reg, backend, env)


def _cache_job_shop_reference_optimum(seed_obj: LogisticsScenarioSeed) -> None:
    if seed_obj.backend_config.get("reference_optimum"):
        return
    reference = seed_obj.backend_config.get("reference") or {}
    value: float | None = None
    method = str(reference.get("type") or "unknown")
    try:
        if reference.get("type") == "known_optimum":
            value = float(reference["makespan"])
        elif reference.get("type") == "best_known_bounds":
            value = float(reference["lower_bound"])
    except (KeyError, TypeError, ValueError):
        value = None
    if value is not None and value > 0:
        seed_obj.backend_config["reference_optimum"] = {
            "reference_optimum": value,
            "method": f"jsplib_{method}",
            "objective": "minimize_makespan",
            "objective_component": "production_cost",
        }


def _cache_co_bench_job_shop_reference_optimum(seed_obj: LogisticsScenarioSeed) -> None:
    if seed_obj.backend_config.get("reference_optimum"):
        return
    # Try structured reference block first
    reference = seed_obj.backend_config.get("reference") or {}
    value: float | None = None
    method = str(reference.get("type") or "unknown")
    try:
        if reference.get("type") == "known_optimum":
            value = float(reference["makespan"])
        elif reference.get("type") == "best_known_bounds":
            value = float(reference["lower_bound"])
    except (KeyError, TypeError, ValueError):
        value = None
    # Fallback: CO-Bench lower_bound in the backend_config.co_bench_job_shop block
    if value is None or value <= 0:
        cfg = seed_obj.backend_config.get("co_bench_job_shop") or {}
        lb = cfg.get("lower_bound")
        try:
            if lb is not None and float(lb) > 0:
                value = float(lb)
                method = "co_bench_lower_bound"
        except (TypeError, ValueError):
            pass
    if value is not None and value > 0:
        seed_obj.backend_config["reference_optimum"] = {
            "reference_optimum": value,
            "method": f"co_bench_{method}",
            "objective": "minimize_makespan",
            "objective_component": "production_cost",
        }


def _register_native_stakeholders(
    mgr: StakeholderTrustManager, seed_obj: LogisticsScenarioSeed
) -> None:
    from .native_stakeholders import build_stakeholder_groups

    for group in build_stakeholder_groups(seed_obj):
        mgr.register(group)


def _build_tick_budget(seed_obj: LogisticsScenarioSeed) -> TickBudget:
    per_tick = {"basic": 6, "medium": 8, "high": 10, "extreme": 12}.get(
        canonical_difficulty_level(seed_obj.difficulty_level), 8
    )
    max_cost_units = max(1.0, per_tick * 0.5)
    return TickBudget(
        max_tool_calls_per_tick=per_tick,
        max_cost_units_per_tick=max_cost_units,
        max_total_tool_calls=per_tick * seed_obj.horizon_ticks,
        duplicate_suppression_window=2,
        cooldown_after_failure=1,
    )


def _rebuild_seed_from_dict(
    d: dict[str, Any], override_seed: int
) -> LogisticsScenarioSeed:
    perturbations = [Perturbation(**p) for p in d.get("perturbations", [])]
    load_assignments = [CustomerPriority(**c) for c in d.get("load_assignments", [])]
    dilemmas = [DilemmaSeed(**ds) for ds in d.get("dilemmas", [])]
    provenance = Provenance(**d.get("provenance", {"data_source": "unspecified"}))
    return LogisticsScenarioSeed(
        seed_id=str(d.get("seed_id", "anon")),
        family=str(d.get("family", "cvrp_dispatch")),
        domain=str(d.get("domain", "logistics")),
        backend_kind=str(d.get("backend_kind", "pyvrp_cvrp")),
        backend_config=dict(d.get("backend_config", {})),
        horizon_ticks=int(d.get("horizon_ticks", 8)),
        tick_minutes=int(d.get("tick_minutes", 30)),
        seed=int(override_seed),
        load_assignments=load_assignments,
        perturbations=perturbations,
        dilemmas=dilemmas,
        difficulty_mode=d.get("difficulty_mode", "time_pressure"),
        difficulty_level=d.get("difficulty_level", "basic"),
        provenance=provenance,
    )
