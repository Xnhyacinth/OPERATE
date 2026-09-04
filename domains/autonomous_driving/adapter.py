"""POMDP adapter for autonomous-driving tactical supervision."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from core import (
    Action,
    BeliefStateTracker,
    EvidenceLogger,
    FogOfWarPolicy,
    HideRule,
    NoiseRule,
    POMDPEnvironment,
    StepInfo,
    StepReturn,
    TickBudget,
    ToolContext,
    ToolRegistry,
)
from core.difficulty_levels import canonical_difficulty_level
from domains.registry import apply_supervisory_cadence

from .backends.sumo_ego import (
    SumoEgoBackend,
    build_sumo_ego_backend,
    formal_sumo_ego_source_evidence_ready,
)
from .native_tools import register_autonomous_driving_tools


@dataclass
class DrivingSeed:
    seed_id: str
    family: str = "highway_cut_in_braking"
    domain: str = "autonomous_driving"
    backend_kind: str = "sumo_ego"
    backend_config: dict[str, Any] = field(default_factory=dict)
    horizon_ticks: int = 4
    tick_seconds: float = 5.0
    seed: int = 42
    difficulty_level: str = "basic"
    difficulty_mode: str = "time_pressure"
    provenance: dict[str, Any] = field(default_factory=dict)
    clock_contract: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _rebuild_seed_from_dict(data: dict[str, Any], override_seed: int) -> DrivingSeed:
    if "tick_minutes" in data and "tick_seconds" not in data:
        raise ValueError("autonomous_driving scenarios require tick_seconds")
    backend_config = dict(data.get("backend_config") or {})
    for key in (
        "candidate_id",
        "start_time_ms",
        "end_time_ms_exclusive",
        "actor_ids",
    ):
        if key in data and key not in backend_config:
            backend_config[key] = data[key]
    return DrivingSeed(
        seed_id=str(data.get("seed_id") or "autonomous_driving_pilot"),
        family=str(data.get("family") or "highway_cut_in_braking"),
        domain=str(data.get("domain") or "autonomous_driving"),
        backend_kind=str(data.get("backend_kind") or "sumo_ego"),
        backend_config=backend_config,
        horizon_ticks=max(1, int(data.get("horizon_ticks") or 4)),
        tick_seconds=max(0.01, float(data.get("tick_seconds") or 5.0)),
        seed=int(override_seed),
        difficulty_level=str(data.get("difficulty_level") or "basic"),
        difficulty_mode=str(data.get("difficulty_mode") or "time_pressure"),
        provenance=dict(data.get("provenance") or {}),
        clock_contract=dict(data.get("clock_contract") or {}),
    )


class AutonomousDrivingEnvironment(POMDPEnvironment):
    """Run long-horizon tactical decisions over backend-owned vehicle control."""

    domain = "autonomous_driving"
    formal_core_allowed = True

    def __init__(self) -> None:
        self._seed_obj: DrivingSeed | None = None
        self._tick = 0
        self._horizon = 4
        self._backend: SumoEgoBackend | None = None
        self._tools: ToolRegistry | None = None
        self._evidence: EvidenceLogger | None = None
        self._fog: FogOfWarPolicy | None = None
        self._belief: BeliefStateTracker | None = None
        self._episode_id = ""
        self._last_runtime_assurance_evidence_ids: list[str] = []
        self.stakeholders = None
        self.dilemmas = None

    def reset(self, scenario_config: dict[str, Any], seed: int) -> dict[str, Any]:
        seed_obj = _rebuild_seed_from_dict(scenario_config, seed)
        if seed_obj.domain != self.domain or seed_obj.backend_kind != "sumo_ego":
            raise ValueError(
                "AutonomousDrivingEnvironment requires domain=autonomous_driving "
                "and backend_kind=sumo_ego"
            )
        self._seed_obj = seed_obj
        self._tick = 0
        self._horizon = seed_obj.horizon_ticks
        self._episode_id = f"dt_{seed_obj.signature()}_s{seed}"
        self._backend = build_sumo_ego_backend(seed_obj.backend_config)
        self._backend.reset(seed_obj)
        self._fog = self._build_fog(seed_obj)
        self._fog.reset(seed)
        self._belief = BeliefStateTracker()
        self._belief.reset()
        self._tools = ToolRegistry(
            budget=self._build_tick_budget(seed_obj),
            seed=seed,
            difficulty_level=seed_obj.difficulty_level,
        )
        self._tools.reset(seed)
        register_autonomous_driving_tools(self._tools, self._backend, self)
        self._evidence = EvidenceLogger(self._episode_id)
        initialization_evidence_id = self._evidence.log(
            kind="runtime_assurance_initialized",
            tick=0,
            payload=self._backend.runtime_assurance_summary(),
            source="engine",
        )
        self._last_runtime_assurance_evidence_ids = [
            initialization_evidence_id
        ]
        observation = self.snapshot()
        self._belief.update_from_observation(observation, tick=0)
        return observation

    def step(self, action: Action) -> StepReturn:
        if (
            self._seed_obj is None
            or self._backend is None
            or self._tools is None
            or self._evidence is None
        ):
            raise RuntimeError("AutonomousDrivingEnvironment is not reset")
        context = ToolContext(
            tick=self._tick,
            seed=self._seed_obj.seed,
            backend=self._backend,
            extra={
                "evidence": self._evidence,
                "fog": self._fog,
                "env": self,
                "episode_horizon": self._horizon,
            },
        )
        tool_results = self._tools.execute_action(
            action,
            context,
            begin_tick=not self._consume_within_tick_budget_state(),
        )
        for result in tool_results:
            result.evidence_id = self._evidence.log(
                kind="tool_call",
                tick=self._tick,
                payload={
                    "name": result.name,
                    "ok": result.ok,
                    "error_code": result.error_code,
                    "call_id": result.call_id,
                    "state_changing": result.state_changing,
                    "payload": result.payload,
                    **self.tool_dependency_payload(action, result),
                },
                source="tool",
            )
            if result.ok and result.state_changing:
                self._backend.bind_tool_result(
                    call_id=result.call_id,
                    evidence_id=result.evidence_id,
                    payload=result.payload,
                )

        record = self._backend.tick(self._tick)
        backend_evidence_id = self._evidence.log(
            kind="backend_tick",
            tick=self._tick,
            payload=record.to_dict(),
            source="engine",
        )
        runtime_assurance_ids: list[str] = []
        for row in record.runtime_assurance_records:
            runtime_assurance_ids.append(
                self._evidence.log(
                    kind="runtime_assurance",
                    tick=self._tick,
                    payload=dict(row),
                    source="engine",
                )
            )
        runtime_assurance_ids.append(
            self._evidence.log(
                kind="runtime_assurance_observation",
                tick=self._tick,
                payload=self._backend.inspect_safety_state(),
                source="engine",
            )
        )
        self._last_runtime_assurance_evidence_ids = list(
            runtime_assurance_ids
        )
        for event in record.realized_events:
            self._evidence.log(
                kind="realized_event",
                tick=self._tick,
                payload=dict(event),
                source="engine",
            )
        self._tick += 1
        observation = self.snapshot()
        if self._belief is not None:
            self._belief.update_from_observation(observation, tick=self._tick)
        early_warnings: list[str] = []
        if record.runtime_assurance_records:
            early_warnings.append("runtime_assurance_intervention")
        if record.mrm_active:
            early_warnings.append("minimal_risk_maneuver_active")
        reward = -sum(
            (
                record.route_delay_seconds,
                record.residual_risk_burden * 20.0,
                record.comfort_jerk_burden * 0.01,
                record.collision_count * 10000.0,
                record.road_departure_count * 10000.0,
            )
        )
        assurance_summary = self._backend.runtime_assurance_summary()
        info = StepInfo(
            realized_events=list(record.realized_events),
            early_stop_warnings=early_warnings,
            evidence_ids=[
                item.evidence_id for item in self._evidence.items() if item.tick == self._tick - 1
            ],
            extra={
                "backend_tick_evidence_id": backend_evidence_id,
                "runtime_assurance": {
                    "schema_version": self._backend.runtime_assurance_schema_version,
                    "low_level_control_owner": assurance_summary["low_level_control_owner"],
                    "mode": assurance_summary["mode"],
                    "shadow_assurance_mode": assurance_summary["shadow_assurance_mode"],
                    "shield_mode": assurance_summary["shield_mode"],
                    "shield_enforcing": assurance_summary["shield_enforcing"],
                    "diagnostic_only": assurance_summary["diagnostic_only"],
                    "intervention_records": list(record.runtime_assurance_records),
                    "evidence_ids": runtime_assurance_ids,
                },
                "source_consumption_evidence": (self._backend.source_consumption_evidence()),
            },
        )
        return StepReturn(
            observation=observation,
            tool_results=tool_results,
            reward=float(reward),
            done=record.done,
            info=info,
        )

    def snapshot(self) -> dict[str, Any]:
        if self._backend is None:
            raise RuntimeError("AutonomousDrivingEnvironment is not reset")
        observation = self._backend.snapshot()
        observation["tick"] = self._tick
        cadence = observation.get("decision_cadence")
        if isinstance(cadence, dict):
            observation["decision_cadence"] = apply_supervisory_cadence(
                self._seed_obj.backend_kind if self._seed_obj is not None else "sumo_ego",
                cadence,
            )
        runtime_ready = self._formal_runtime_ready()
        observation["formal_core_allowed"] = runtime_ready
        observation["release_admission"] = (
            "native_live_sumo_evidence_verified"
            if runtime_ready
            else "runtime_evidence_required"
        )
        if self._fog is not None:
            self._fog.set_tick(self._tick)
            observation = self._fog.filter(observation)
        if self._seed_obj is not None and canonical_difficulty_level(
            self._seed_obj.difficulty_level
        ) in {"high", "extreme"}:
            safety = dict(observation.get("safety_state") or {})
            safety["min_ttc_seconds"] = None
            safety["detail_available_via"] = "inspect_safety_state"
            observation["safety_state"] = safety
        safety_state = observation.get("safety_state")
        if isinstance(safety_state, dict):
            safety_state["evidence_ids"] = list(
                self._last_runtime_assurance_evidence_ids
            )
        return observation

    def ground_truth(self) -> dict[str, Any]:
        if self._backend is None:
            raise RuntimeError("AutonomousDrivingEnvironment is not reset")
        truth = self._backend.ground_truth()
        truth["tick"] = self._tick
        truth["formal_core_allowed"] = self._formal_runtime_ready()
        return truth

    def _formal_runtime_ready(self) -> bool:
        """Require live source consumption and enforcing assurance per run."""
        if self._backend is None:
            return False
        evidence = self._backend.source_consumption_evidence()
        assurance = self._backend.runtime_assurance_summary()
        return bool(
            formal_sumo_ego_source_evidence_ready(evidence)
            and assurance.get("shield_enforcing") is True
            and assurance.get("diagnostic_only") is False
        )

    def source_consumption_evidence(self, *, scenario: dict[str, Any]) -> dict[str, Any]:
        if self._backend is None:
            return {"status": "held", "blockers": ["backend_not_reset"]}
        evidence = self._backend.source_consumption_evidence()
        evidence["scenario_domain"] = scenario.get("domain")
        evidence["backend_kind"] = scenario.get("backend_kind")
        return evidence

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
    def seed_obj(self) -> DrivingSeed | None:
        return self._seed_obj

    @property
    def episode_id(self) -> str:
        return self._episode_id

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()

    @staticmethod
    def _build_tick_budget(seed_obj: DrivingSeed) -> TickBudget:
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

    @staticmethod
    def _build_fog(seed_obj: DrivingSeed) -> FogOfWarPolicy:
        hidden = canonical_difficulty_level(seed_obj.difficulty_level) in {
            "high",
            "extreme",
        }
        return FogOfWarPolicy(
            hide_rules=(
                [
                    HideRule(
                        entity_kind="road_actor",
                        hidden_attrs=["relative_speed_mps"],
                        reveal_on=["inspect_local_scene"],
                    )
                ]
                if hidden
                else []
            ),
            noise_rules=[
                NoiseRule(
                    entity_kind="road_actor",
                    attr="relative_distance_m",
                    sigma_rel=0.01,
                )
            ],
            staleness_rules=[],
            seed=seed_obj.seed,
        )
