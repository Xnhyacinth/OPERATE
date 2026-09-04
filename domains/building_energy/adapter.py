"""POMDP adapter for native CityLearn Building Energy candidates."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from core import (
    Action,
    BeliefStateTracker,
    CascadeBus,
    EvidenceLogger,
    FogOfWarPolicy,
    HideRule,
    NoiseRule,
    POMDPEnvironment,
    StalenessRule,
    StepInfo,
    StepReturn,
    TickBudget,
    ToolContext,
    ToolRegistry,
)
from core.difficulty_levels import canonical_difficulty_level
from domains.registry import apply_supervisory_cadence

from .backends.citylearn import CityLearnBackend
from .seeds.schema import BuildingEnergyScenarioSeed, rebuild_seed_from_dict
from .tools import register_building_energy_tools


def _state_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _build_citylearn_fog_policy(
    seed_obj: BuildingEnergyScenarioSeed, seed: int
) -> FogOfWarPolicy:
    """Build only the source-declared Building Energy observation contract."""

    contract = seed_obj.backend_config.get("observation_contract") or {}
    hidden = [str(value) for value in contract.get("hide_building_attrs") or []]
    noise = contract.get("noise_sigma_rel") or {}
    stale = contract.get("staleness_ticks") or {}
    if not isinstance(noise, dict) or not isinstance(stale, dict):
        raise ValueError("CityLearn observation contract must use attribute mappings")
    return FogOfWarPolicy(
        hide_rules=(
            [HideRule(entity_kind="building", hidden_attrs=hidden)]
            if hidden
            else []
        ),
        noise_rules=[
            NoiseRule(entity_kind="building", attr=str(attr), sigma_rel=float(value))
            for attr, value in sorted(noise.items())
        ],
        staleness_rules=[
            StalenessRule(
                entity_kind="building",
                attr=str(attr),
                staleness_ticks=int(value),
            )
            for attr, value in sorted(stale.items())
        ],
        seed=seed,
    )


def _authoritative_source_event(
    event: dict[str, Any],
    evidence_id: str,
    visible_source_events_by_evidence_id: dict[str, dict[str, Any]],
) -> None:
    """Attach logger evidence and retain only passively visible source events."""

    if str(event.get("origin") or "") == "agent_caused":
        return
    event.setdefault("event_id", f"building-source-event:{evidence_id}")
    evidence_ids = event.setdefault("evidence_ids", [])
    if evidence_id not in evidence_ids:
        evidence_ids.append(evidence_id)
    if event.get("hidden") is True:
        return
    visible_source_events_by_evidence_id[evidence_id] = {
        "event_id": str(event["event_id"]),
        "visible_from_request_tick": int(
            event.get("observed_after_control_tick", event.get("tick", 0))
            or 0
        )
        + 1,
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


class BuildingEnergyEnvironment(POMDPEnvironment):
    """Drive CityLearn through the benchmark protocol and formal evidence gates."""

    domain = "building_energy"
    formal_core_allowed = True

    def __init__(self, cascade_bus: CascadeBus | None = None) -> None:
        self._seed_obj: BuildingEnergyScenarioSeed | None = None
        self._tick = 0
        self._horizon = 24
        self._backend = CityLearnBackend()
        self._tools: ToolRegistry | None = None
        self._evidence: EvidenceLogger | None = None
        self._fog: FogOfWarPolicy | None = None
        self._belief: BeliefStateTracker | None = None
        self._cascade_bus = cascade_bus or CascadeBus()
        self._episode_id = ""
        self._visible_source_events_by_evidence_id: dict[str, dict[str, Any]] = {}
        # This Building Energy task has no source-grounded stakeholder/ethics manager
        # yet.  Expose explicit null managers so the generic runner can
        # diagnose the pilot without accidentally fabricating social scores.
        self.stakeholders = None
        self.dilemmas = None

    def reset(self, scenario_config: dict[str, Any], seed: int) -> dict[str, Any]:
        seed_obj = rebuild_seed_from_dict(scenario_config, override_seed=seed)
        if seed_obj.domain != self.domain or seed_obj.backend_kind != "citylearn":
            raise ValueError("BuildingEnergyEnvironment requires backend_kind=citylearn")
        self._seed_obj = seed_obj
        self._tick = 0
        self._horizon = seed_obj.horizon_ticks
        self._episode_id = f"dt_{seed_obj.signature()}_s{seed}"
        self._visible_source_events_by_evidence_id = {}
        self._backend.reset(seed_obj)
        self._fog = _build_citylearn_fog_policy(seed_obj, seed)
        self._fog.reset(seed=seed)
        self._belief = BeliefStateTracker()
        self._belief.reset()
        self._tools = ToolRegistry(
            budget=self._build_tick_budget(seed_obj),
            seed=seed,
            difficulty_level=seed_obj.difficulty_level,
        )
        self._tools.reset(seed=seed)
        register_building_energy_tools(self._tools, self._backend, self)
        self._evidence = EvidenceLogger(episode_id=self._episode_id)
        self._evidence.log(
            kind="source_lock_runtime",
            tick=0,
            payload=self._backend.source_consumption_evidence(),
            source="engine",
        )
        observation = self.snapshot()
        self._belief.update_from_observation(observation, tick=0)
        return observation

    def step(self, action: Action) -> StepReturn:
        if self._tools is None or self._evidence is None or self._seed_obj is None:
            raise RuntimeError("BuildingEnergyEnvironment is not reset")
        ctx = ToolContext(
            tick=self._tick,
            seed=self._seed_obj.seed,
            backend=self._backend,
            extra={"evidence": self._evidence, "env": self},
        )
        tool_results = self._tools.execute_action(action, ctx)
        for result in tool_results:
            result.evidence_id = self._evidence.log(
                kind="tool_call",
                tick=self._tick,
                payload={
                    "name": result.name,
                    "ok": result.ok,
                    "call_id": result.call_id,
                    "state_changing": result.state_changing,
                    "payload": result.payload,
                    **self.tool_dependency_payload(action, result),
                },
                source="tool",
            )
            if result.name == "set_storage_dispatch" and result.ok:
                causal_parent_event_id = _visible_causal_parent_event_id(
                    action,
                    result.call_id,
                    self._tick,
                    self._visible_source_events_by_evidence_id,
                )
                self._backend.bind_control_evidence(
                    call_id=result.call_id,
                    evidence_id=result.evidence_id,
                    payload=result.payload,
                    causal_parent_event_id=causal_parent_event_id,
                )
        record = self._backend.tick(self._tick)
        self._evidence.log(
            kind="backend_tick",
            tick=self._tick,
            payload={
                "tick": record.tick,
                "simulator_time_step": record.simulator_time_step,
                "reward": record.reward,
                "district_net_electricity_consumption": (
                    record.district_net_electricity_consumption
                ),
                "source_event_peak_response_burden": (
                    record.source_event_peak_response_burden
                ),
                "native_storage_charging_burden": (
                    record.native_storage_charging_burden
                ),
                "action_vector": record.action_vector,
                "source_consumed": record.source_consumed,
                "state_effect_observed": record.state_effect_observed,
            },
            source="engine",
        )
        for event in record.realized_events:
            evidence_id = self._evidence.log(
                kind="realized_event", tick=self._tick, payload=event, source="engine"
            )
            _authoritative_source_event(
                event,
                evidence_id,
                self._visible_source_events_by_evidence_id,
            )
        self._tick += 1
        done = bool(record.done or self._tick >= self._horizon)
        observation = self.snapshot()
        if self._belief is not None:
            self._belief.update_from_observation(observation, tick=self._tick)
        info = StepInfo(
            realized_events=list(record.realized_events),
            evidence_ids=[
                item.evidence_id
                for item in self._evidence.items()
                if item.tick == self._tick - 1
            ],
            extra={
                "native_state_effect_observed": record.state_effect_observed,
                "source_consumption_evidence": self._backend.source_consumption_evidence(),
                "trajectory_state_digest": _state_digest(observation),
            },
        )
        return StepReturn(
            observation=observation,
            tool_results=tool_results,
            reward=record.reward,
            done=done,
            info=info,
        )

    def snapshot(self) -> dict[str, Any]:
        observation = self._backend.snapshot()
        observation["tick"] = self._tick
        cadence = observation.get("decision_cadence")
        if isinstance(cadence, dict):
            observation["decision_cadence"] = apply_supervisory_cadence(
                self._backend.backend_kind,
                cadence,
            )
        observation["formal_core_allowed"] = True
        observation["release_admission"] = "protocol21_gated"
        if self._fog is not None:
            self._fog.set_tick(self._tick)
            observation = self._fog.filter(observation, entity_key="buildings")
        return observation

    def ground_truth(self) -> dict[str, Any]:
        truth = self._backend.ground_truth()
        truth["tick"] = self._tick
        truth["formal_core_allowed"] = True
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

    def source_consumption_evidence(self, *, scenario: dict[str, Any]) -> dict[str, Any]:
        evidence = self._backend.source_consumption_evidence()
        bounded_probe = evidence.get("bounded_source_probe")
        if evidence.get("status") == "held" and not (
            isinstance(bounded_probe, dict)
            and bounded_probe.get("executed") is True
        ):
            self._backend.run_bounded_source_probe()
            evidence = self._backend.source_consumption_evidence()
        evidence["scenario_domain"] = scenario.get("domain")
        evidence["backend_kind"] = scenario.get("backend_kind")
        return evidence

    @property
    def evidence(self) -> EvidenceLogger | None:
        return self._evidence

    @property
    def seed_obj(self) -> BuildingEnergyScenarioSeed | None:
        return self._seed_obj

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def close(self) -> None:
        self._backend.close()

    @staticmethod
    def _build_tick_budget(seed_obj: BuildingEnergyScenarioSeed) -> TickBudget:
        per_tick = {"basic": 5, "medium": 7, "high": 9, "extreme": 11}.get(
            canonical_difficulty_level(seed_obj.difficulty_level), 7
        )
        return TickBudget(
            max_tool_calls_per_tick=per_tick,
            max_total_tool_calls=per_tick * seed_obj.horizon_ticks,
            max_cost_units_per_tick=max(2.0, per_tick * 0.75),
            duplicate_suppression_window=2,
            cooldown_after_failure=1,
            horizon=seed_obj.horizon_ticks,
        )


def _rebuild_seed_from_dict(
    data: dict[str, Any], override_seed: int
) -> BuildingEnergyScenarioSeed:
    return rebuild_seed_from_dict(data, override_seed)
