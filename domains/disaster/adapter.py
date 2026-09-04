"""
domains.disaster.adapter — Disaster-response POMDP environment.

Mirrors the structure of ``domains.power_grid.adapter`` line-for-line —
backend + fog + belief + tool registry + stakeholder manager + dilemma
manager + evidence logger + cascade-bus subscriber — but uses
disaster-native types and never imports from ``domains.power_grid``
(Red Line #3, ``.hl/policy.md``).

The default backend is ``MockRcrsBackend``. The seed's
``backend_config['backend_kind']`` switches to the real
``RcrsBackend`` (which, in the v0.3 spike, raises ``NotImplementedError``
on every method except its constructor).

Cross-domain coupling: subscribes to ``power_grid.line.outage`` on the
cascade bus and injects a ``comms_blackout`` perturbation for any zone
whose ``has_hospital=True`` so a grid fault that takes out a hospital
substation propagates to a disaster-side comms outage.
"""

from __future__ import annotations

from typing import Any

from core import (
    Action,
    BeliefStateTracker,
    CascadeBus,
    CascadeEvent,
    EthicalDilemmaManager,
    EvidenceLogger,
    FogOfWarPolicy,
    HideRule,
    NoiseRule,
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

from .backends.mock_rcrs import MockRcrsBackend
from .backends.rcrs_backend import RcrsBackend
from .seeds.schema import (
    DilemmaSeed,
    DisasterScenarioSeed,
    Perturbation,
    Provenance,
    ZoneAssignment,
)

# ─────────────────────────────────────────────────────────────────────────────
# Event-name mapping (backend → canonical cascade event_type)
# ─────────────────────────────────────────────────────────────────────────────


_CASCADE_EVENT_MAPPING: dict[str, str] = {
    "building_collapse": "disaster.building.collapse",
    "aftershock": "disaster.hazard.aftershock",
    "fire_spread": "disaster.fire.spread",
    "medical_surge": "disaster.casualty.surge",
    "comms_blackout": "disaster.comms.blackout",
    "tsunami_inundation": "disaster.hazard.tsunami",
    "road_blockage": "disaster.transport.blocked",
    "bridge_failure": "disaster.transport.bridge_failed",
    "gas_leak": "disaster.hazard.gas_leak",
    "hazard_shake": "disaster.hazard.shake",
    "mutual_aid_arrived": "disaster.aid.arrived",
}


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────


class DisasterEnvironment(POMDPEnvironment):
    """OPERATE environment for the disaster-response domain.

    Constructor takes an optional ``cascade_bus`` so a cross-domain run
    can share a single bus across PowerGridEnvironment + DisasterEnvironment.
    """

    domain = "disaster"

    def __init__(self, cascade_bus: CascadeBus | None = None) -> None:
        self._seed_obj: DisasterScenarioSeed | None = None
        self._tick: int = 0
        self._horizon: int = 144
        self._backend: Any = None
        self._fog: FogOfWarPolicy | None = None
        self._belief: BeliefStateTracker | None = None
        self._tools: ToolRegistry | None = None
        self._stakeholders: StakeholderTrustManager | None = None
        self._dilemmas: EthicalDilemmaManager | None = None
        self._evidence: EvidenceLogger | None = None
        self._cascade_bus = cascade_bus or CascadeBus()
        self._episode_id: str = ""
        # Pending grid→disaster cascade injections: list of
        # ``Perturbation`` to add at the next ``tick()``. Used by the
        # ``power_grid.line.outage`` subscriber.
        self._pending_cascade_perturbations: list[Perturbation] = []
        # Wire the subscriber once per environment instance.
        self._cascade_bus.subscribe(
            "power_grid.line.outage", self._on_power_grid_line_outage
        )

    # ── POMDPEnvironment surface ────────────────────────────────────────

    def reset(self, scenario_config: dict[str, Any], seed: int) -> dict[str, Any]:
        seed_obj = _rebuild_seed_from_dict(scenario_config, override_seed=seed)
        self._seed_obj = seed_obj
        self._tick = 0
        self._horizon = seed_obj.horizon_ticks
        self._episode_id = f"dt_{seed_obj.signature()}_s{seed}"

        # Backend selection: mock by default; real only if explicitly
        # requested via backend_config (in v0.3 the real backend raises).
        kind = str(seed_obj.backend_config.get("backend_kind", "mock_rcrs"))
        self._backend = _build_backend(kind, seed_obj.backend_config)
        self._backend.reset(seed_obj)

        # Fog of war
        self._fog = _build_fog(seed_obj)
        self._fog.reset(seed=seed)

        # Belief
        self._belief = BeliefStateTracker()
        self._belief.reset()

        # Tool registry
        self._tools = ToolRegistry(
            budget=_build_tick_budget(seed_obj),
            seed=seed,
            difficulty_level=seed_obj.difficulty_level,
        )
        self._tools.reset(seed=seed)
        _register_native_tools(self._tools, self._backend, self)

        # Stakeholders
        self._stakeholders = StakeholderTrustManager()
        _register_native_stakeholders(self._stakeholders, seed_obj)

        # Dilemmas
        self._dilemmas = EthicalDilemmaManager()
        self._dilemmas.reset()
        arm_dilemmas(self._dilemmas, seed_obj.dilemmas)

        # Evidence
        self._evidence = EvidenceLogger(episode_id=self._episode_id)

        obs = self.snapshot()
        self._belief.update_from_observation(obs, tick=0)
        return obs

    def step(self, action: Action) -> StepReturn:
        assert self._tools is not None and self._backend is not None
        assert self._fog is not None and self._evidence is not None
        assert self._dilemmas is not None and self._stakeholders is not None
        assert self._belief is not None

        # 1) Execute tool calls (mutates backend via tool handlers).
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
        tool_results = self._tools.execute_action(action, ctx)
        for r in tool_results:
            r.evidence_id = self._evidence.log(
                kind="tool_call",
                tick=self._tick,
                payload={
                    "name": r.name,
                    "ok": r.ok,
                    "error_code": r.error_code,
                    "cost_units": r.cost_units,
                    "call_id": r.call_id,
                    "state_changing": r.state_changing,
                    "payload": r.payload,
                },
                source="tool",
            )

        # 2) Inject any pending cascade-bus perturbations onto the seed
        # BEFORE the backend ticks, so the backend sees them this tick.
        if self._pending_cascade_perturbations and self._seed_obj is not None:
            self._seed_obj.perturbations.extend(self._pending_cascade_perturbations)
            self._pending_cascade_perturbations = []

        # 3) Advance backend.
        backend_tick_record = self._backend.tick(self._tick)
        backend_tick_payload = safe_dataclass_to_dict(backend_tick_record)
        self._evidence.log(
            kind="backend_tick",
            tick=self._tick,
            payload=backend_tick_payload,
            source="engine",
        )
        self._evidence.log(
            kind="cost_summary",
            tick=self._tick,
            payload=self._cost_summary_payload(backend_tick_record),
            source="engine",
        )
        for ev in getattr(backend_tick_record, "realized_events", []) or []:
            evidence_id = self._evidence.log(
                kind="realized_event",
                tick=self._tick,
                payload=dict(ev),
                source="engine",
            )
            self._publish_cascade_for_event(ev, evidence_id)

        # 4) Dilemma triggers.
        current_snap = self.snapshot()
        triggered = self._dilemmas.maybe_trigger(self._tick, current_snap)
        for d in triggered:
            self._evidence.log(
                kind="dilemma_triggered",
                tick=self._tick,
                payload={"dilemma_id": d.dilemma_id, "description": d.description},
                source="engine",
            )
        # Enforce dilemma deadlines.
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

        # 5) Stakeholder natural drift.
        self._stakeholders.tick(self._tick)

        self._tick += 1
        backend_done = bool(getattr(backend_tick_record, "done", False))
        done = backend_done or self._tick >= self._horizon

        snap = self.snapshot()
        snap["tick"] = self._tick

        self._belief.update_from_observation(snap, tick=self._tick)

        reward = self._reward_signal(backend_tick_record)

        info = StepInfo(
            realized_events=list(
                getattr(backend_tick_record, "realized_events", []) or []
            ),
            evidence_ids=[
                i.evidence_id
                for i in self._evidence.items()
                if i.tick == self._tick - 1
            ],
            extra={
                "dilemmas_triggered": [d.dilemma_id for d in triggered],
                "stakeholder_trust": {
                    gid: r.trust for gid, r in self._stakeholders.snapshot().items()
                },
                "backend_tick_record": safe_dataclass_to_dict(backend_tick_record),
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
        if self._stakeholders:
            raw["stakeholder_trust"] = {
                gid: {"trust": r.trust, "tier": r.tier}
                for gid, r in self._stakeholders.snapshot().items()
            }
        if self._dilemmas:
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
        """Full state, including hidden perturbations and per-zone unserved minutes."""
        assert self._backend is not None and self._seed_obj is not None
        gt = self._backend.snapshot()
        gt["per_zone_unserved_minutes"] = self._backend.per_zone_unserved_minutes()
        gt["cost_components"] = self._backend.ground_truth_costs()
        if self._stakeholders:
            gt["stakeholder_trust"] = {
                gid: r.trust for gid, r in self._stakeholders.snapshot().items()
            }
            gt["stakeholder_equity_gini"] = round(self._stakeholders.equity_gini(), 4)
        if self._dilemmas:
            gt["dilemmas_triggered"] = [
                d.dilemma_id for d in self._dilemmas.record.dilemmas_triggered
            ]
            gt["choices"] = {
                did: {"option_id": c.chosen_option_id, "tick_chosen": c.tick_chosen}
                for did, c in self._dilemmas.record.choices.items()
            }
            gt["chose_fatal_option"] = self._dilemmas.chose_fatal_option()
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
        if not self._tools:
            return []
        return self._tools.openai_schemas()

    def close(self) -> None:
        # Mock backend has no external resources; real backend would
        # shut down its Docker container here.
        pass

    # ── Public accessors used by the runner/scorer ──────────────────────

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
    def seed_obj(self) -> DisasterScenarioSeed | None:
        return self._seed_obj

    @property
    def episode_id(self) -> str:
        return self._episode_id

    # ── Internals ───────────────────────────────────────────────────────

    def _cost_summary_payload(self, backend_tick_record: Any) -> dict[str, float]:
        return {
            "unserved_minutes": float(
                getattr(backend_tick_record, "unserved_minutes", 0.0)
            ),
            "responder_cost": float(
                getattr(backend_tick_record, "responder_cost", 0.0)
            ),
            "casualty_count": float(
                getattr(backend_tick_record, "casualty_count", 0.0)
            ),
            "zones_cleared": float(getattr(backend_tick_record, "zones_cleared", 0.0)),
        }

    def _reward_signal(self, backend_tick_record: Any) -> float:
        """Per-tick reward (negative cost). Informative only."""
        response = float(getattr(backend_tick_record, "response_cost_this_tick", 0.0))
        unserved = float(getattr(backend_tick_record, "unserved_cost_this_tick", 0.0))
        return -(response + unserved) / 1000.0

    def _publish_cascade_for_event(
        self, event: dict[str, Any], evidence_id: str
    ) -> None:
        """Map a backend-emitted event onto a canonical cascade event_type
        and publish it. Mirrors the power-grid adapter's helper exactly.
        """
        type_str = str(event.get("type", ""))
        canonical = _CASCADE_EVENT_MAPPING.get(type_str)
        if canonical is None:
            return
        severity = float(event.get("intensity", 0.5) or 0.5)
        self._cascade_bus.publish(
            CascadeEvent(
                event_type=canonical,
                source_domain="disaster",
                tick=int(event.get("tick", self._tick)),
                severity=max(0.0, min(1.0, severity)),
                location={"zone": event.get("zone") or event.get("epicenter")},
                payload={**event, "evidence_id": evidence_id},
                correlation_id=evidence_id,
            )
        )

    def _on_power_grid_line_outage(self, event: CascadeEvent) -> None:
        """Cascade-bus subscriber: grid line outage → hospital comms blackout.

        For every zone whose ``has_hospital=True``, enqueue a
        ``comms_blackout`` perturbation that fires on the next tick and
        runs for 6 ticks. We do not mutate backend state directly — we
        enqueue a ``Perturbation`` that ``step()`` will splice into the
        seed before the backend ticks, so the realized event flows
        through the normal evidence + cascade path.
        """
        if self._seed_obj is None:
            return
        trigger_tick = max(self._tick, int(event.tick) + 1)
        for za in self._seed_obj.zone_assignments:
            if not za.has_hospital:
                continue
            self._pending_cascade_perturbations.append(
                Perturbation(
                    kind="comms_blackout",
                    trigger_tick=trigger_tick,
                    duration_ticks=6,
                    hidden=False,
                    target={
                        "zone": za.zone_id,
                        "cause": "cascaded_from_power_grid.line.outage",
                    },
                    intensity=float(event.severity or 0.5),
                    notes=(
                        f"Cascaded from power_grid.line.outage "
                        f"(correlation_id={event.correlation_id})"
                    ),
                )
            )


# ─────────────────────────────────────────────────────────────────────────────
# Backend factory
# ─────────────────────────────────────────────────────────────────────────────


def _build_backend(kind: str, cfg: dict[str, Any]) -> Any:
    if kind == "mock_rcrs":
        return MockRcrsBackend()
    if kind == "rcrs":
        return RcrsBackend(
            docker_image=str(cfg.get("docker_image", "rcrs-server:0.7.x")),
            tcp_host=str(cfg.get("tcp_host", "127.0.0.1")),
            tcp_port=int(cfg.get("tcp_port", 7000)),
        )
    raise ValueError(f"unknown disaster backend_kind: {kind}")


# ─────────────────────────────────────────────────────────────────────────────
# Fog policy from seed
# ─────────────────────────────────────────────────────────────────────────────


def _build_fog(seed_obj: DisasterScenarioSeed) -> FogOfWarPolicy:
    """Hide buried counts under comms_blackout + add sensor noise.

    - Zones under a hidden ``comms_blackout`` perturbation hide their
      ``buried`` and ``fire_intensity`` counters until investigated.
    - Baseline sensor uncertainty: ``buried`` is reported ±5% noise to
      simulate triage uncertainty; ``fire_intensity`` is ±10% noise.
    - Hidden ``building_collapse`` perturbations: hide ``buried`` for
      the target zone until ``survey_zone`` or ``dispatch_recon`` is
      called on it.
    """
    hide_rules: list[HideRule] = []
    noise_rules: list[NoiseRule] = []

    noise_rules.append(NoiseRule(entity_kind="zone", attr="buried", sigma_rel=0.05))
    noise_rules.append(
        NoiseRule(entity_kind="zone", attr="fire_intensity", sigma_rel=0.10)
    )

    has_hidden_blackout = any(
        p.kind == "comms_blackout" and p.hidden for p in seed_obj.perturbations
    )
    has_hidden_collapse = any(
        p.kind == "building_collapse" and p.hidden for p in seed_obj.perturbations
    )
    if has_hidden_blackout or has_hidden_collapse:
        hide_rules.append(
            HideRule(
                entity_kind="zone",
                hidden_attrs=["buried", "fire_intensity"],
                reveal_on=["survey_zone", "dispatch_recon"],
            )
        )

    return FogOfWarPolicy(
        hide_rules=hide_rules,
        noise_rules=noise_rules,
        staleness_rules=[],
        seed=seed_obj.seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stakeholders + dilemmas (delegate)
# ─────────────────────────────────────────────────────────────────────────────


def _register_native_stakeholders(
    mgr: StakeholderTrustManager, seed_obj: DisasterScenarioSeed
) -> None:
    from .native_stakeholders import build_stakeholder_groups

    for group in build_stakeholder_groups(seed_obj):
        mgr.register(group)


# ─────────────────────────────────────────────────────────────────────────────
# Tool registration (delegate)
# ─────────────────────────────────────────────────────────────────────────────


def _register_native_tools(
    reg: ToolRegistry, backend: Any, env: DisasterEnvironment
) -> None:
    from .native_tools import register_disaster_tools

    register_disaster_tools(reg, backend, env)


# ─────────────────────────────────────────────────────────────────────────────
# Tick budget from difficulty
# ─────────────────────────────────────────────────────────────────────────────


def _build_tick_budget(seed_obj: DisasterScenarioSeed) -> TickBudget:
    level = canonical_difficulty_level(seed_obj.difficulty_level)
    per_tick = {"basic": 12, "medium": 10, "high": 8, "extreme": 6}.get(level, 8)
    total = per_tick * seed_obj.horizon_ticks
    # v0.51 fix: disaster was the only one of the five domains whose
    # ``_build_tick_budget`` never set ``max_cost_units_per_tick`` —
    # logistics / traffic / microgrid all set ``max(1.0, per_tick * 0.5)``
    # for the equivalent difficulty tiers (power_grid uses the same
    # formula over its own, wider tier table). This looked like an
    # unintentional omission rather than a deliberate disaster-specific
    # design choice, so it is repaired here to match the other domains.
    max_cost_units = max(1.0, per_tick * 0.5)
    return TickBudget(
        max_tool_calls_per_tick=per_tick,
        max_cost_units_per_tick=max_cost_units,
        max_total_tool_calls=total,
        duplicate_suppression_window=2,
        cooldown_after_failure=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Rebuild a DisasterScenarioSeed from its dict serialization
# ─────────────────────────────────────────────────────────────────────────────


def _rebuild_seed_from_dict(
    d: dict[str, Any], override_seed: int
) -> DisasterScenarioSeed:
    perturbations = [Perturbation(**p) for p in d.get("perturbations", [])]
    zone_assignments = [ZoneAssignment(**za) for za in d.get("zone_assignments", [])]
    dilemmas = [DilemmaSeed(**ds) for ds in d.get("dilemmas", [])]
    provenance = Provenance(**d.get("provenance", {"data_source": "unspecified"}))
    return DisasterScenarioSeed(
        seed_id=str(d.get("seed_id", "anon")),
        family=str(d.get("family", "urban_earthquake_M6_24h")),
        domain=str(d.get("domain", "disaster")),
        backend_kind=str(d.get("backend_kind", "mock_rcrs")),
        backend_config=dict(d.get("backend_config", {})),
        horizon_ticks=int(d.get("horizon_ticks", 144)),
        tick_minutes=int(d.get("tick_minutes", 10)),
        seed=int(override_seed),
        zone_assignments=zone_assignments,
        perturbations=perturbations,
        dilemmas=dilemmas,
        hazard_time_series_ref=d.get("hazard_time_series_ref"),
        difficulty_mode=d.get("difficulty_mode", "time_pressure"),
        difficulty_level=d.get("difficulty_level", "basic"),
        provenance=provenance,
    )
