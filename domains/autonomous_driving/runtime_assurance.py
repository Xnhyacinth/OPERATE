"""Deterministic, simulator-side runtime assurance for vehicle control.

This module is a benchmark safety layer, not a regulatory minimal-risk
maneuver implementation.  It uses simulator state and deliberately simple P0
prediction so that decisions are deterministic, inspectable, and inexpensive.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum


class AssuranceState(StrEnum):
    """Latched runtime-assurance state."""

    NOMINAL = "nominal"
    DEGRADED = "degraded"
    EMERGENCY_OVERRIDE = "emergency_override"
    MRM_ACTIVE = "mrm_active"
    MINIMAL_RISK_CONDITION = "minimal_risk_condition"
    RECOVERY_PENDING = "recovery_pending"


class InterventionKind(StrEnum):
    """How the applied command relates to the nominal command."""

    PASS = "pass"  # nosec B105
    CLIP = "clip"
    OVERRIDE = "override"
    FALLBACK = "fallback"


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class RoadBoundary:
    """A simple, ordered drivable-area polygon without holes."""

    polygon: tuple[Point2D, ...]


@dataclass(frozen=True)
class VehicleState:
    actor_id: str
    x: float
    y: float
    heading_rad: float
    speed_mps: float
    acceleration_mps2: float
    length_m: float
    width_m: float


@dataclass(frozen=True)
class SafetyState:
    sim_time: float
    observed_at: float
    state_version: int
    ego: VehicleState
    actors: tuple[VehicleState, ...]
    road_boundary: RoadBoundary


@dataclass(frozen=True)
class LateralSafetyVerification:
    """A verifier mark scoped to one state and exact lateral command."""

    verifier_id: str
    state_version: int
    command_id: str
    acceleration_mps2: float
    steering_rad: float
    valid_until: float


@dataclass(frozen=True)
class ControlCommand:
    command_id: str
    sequence: int
    state_version: int
    issued_at: float
    valid_from: float
    valid_until: float
    acceleration_mps2: float
    steering_rad: float
    lateral_escape: bool = False
    lateral_verification: LateralSafetyVerification | None = None


@dataclass(frozen=True)
class CandidateAssessment:
    name: str
    collision_free: bool
    on_road: bool
    conflict_time_s: float | None
    boundary_margin_m: float
    predicted_impact_speed_mps: float
    progress_m: float


@dataclass(frozen=True)
class SafetyEvidence:
    reason_codes: tuple[str, ...]
    minimum_ttc_s: float | None
    minimum_time_headway_s: float | None
    nearest_longitudinal_gap_m: float | None
    required_stopping_distance_m: float | None
    candidates: tuple[CandidateAssessment, ...]
    selected_candidate: str
    rollout_step_s: float
    rollout_horizon_s: float


@dataclass(frozen=True)
class SafetyDecision:
    nominal_command: ControlCommand
    applied_command: ControlCommand
    intervention: InterventionKind
    assurance_state: AssuranceState
    evidence: SafetyEvidence


@dataclass(frozen=True)
class RuntimeAssuranceConfig:
    """P0 parameters; defaults define a 0.1 s, 3 s simulation monitor."""

    step_s: float = 0.1
    horizon_s: float = 3.0
    max_state_age_s: float = 0.2
    max_command_age_s: float = 0.5
    warning_ttc_s: float = 2.0
    emergency_ttc_s: float = 1.0
    min_time_headway_s: float = 2.0
    reaction_time_s: float = 0.2
    standstill_gap_m: float = 2.0
    uncertainty_margin_m: float = 0.5
    collision_margin_m: float = 0.1
    comfortable_deceleration_mps2: float = 3.0
    emergency_deceleration_mps2: float = 8.0
    assumed_lead_deceleration_mps2: float = 8.0
    maximum_acceleration_mps2: float = 3.0
    maximum_steering_rad: float = 0.5
    wheelbase_m: float = 2.8
    stopped_speed_mps: float = 0.1
    recovery_healthy_steps: int = 3
    consider_rear_collision: bool = False
    recovery_token: str | None = None

    def __post_init__(self) -> None:
        positive = (
            self.step_s,
            self.horizon_s,
            self.max_state_age_s,
            self.max_command_age_s,
            self.comfortable_deceleration_mps2,
            self.emergency_deceleration_mps2,
            self.assumed_lead_deceleration_mps2,
            self.maximum_acceleration_mps2,
            self.maximum_steering_rad,
            self.wheelbase_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("runtime-assurance positive parameters must be finite")
        nonnegative = (
            self.reaction_time_s,
            self.standstill_gap_m,
            self.uncertainty_margin_m,
            self.collision_margin_m,
            self.stopped_speed_mps,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in nonnegative):
            raise ValueError("runtime-assurance nonnegative parameters must be finite")
        if not math.isfinite(self.min_time_headway_s) or self.min_time_headway_s <= 0.0:
            raise ValueError("min_time_headway_s must be positive")
        if (
            not math.isfinite(self.emergency_ttc_s)
            or self.emergency_ttc_s <= 0.0
            or not math.isfinite(self.warning_ttc_s)
            or self.warning_ttc_s < self.emergency_ttc_s
        ):
            raise ValueError("warning_ttc_s must be at least emergency_ttc_s")
        if not math.isclose(
            self.horizon_s / self.step_s,
            round(self.horizon_s / self.step_s),
            abs_tol=1e-9,
        ):
            raise ValueError("horizon_s must be an integer multiple of step_s")
        if self.recovery_healthy_steps < 1:
            raise ValueError("recovery_healthy_steps must be positive")
        if self.recovery_token == "":  # nosec B105
            raise ValueError("recovery_token cannot be empty")


@dataclass(frozen=True)
class _Candidate:
    name: str
    command: ControlCommand
    order: int


class RuntimeAssurance:
    """Stateful command gate called immediately before every physics step."""

    def __init__(self, config: RuntimeAssuranceConfig | None = None) -> None:
        self.config = config or RuntimeAssuranceConfig()
        self._assurance_state = AssuranceState.NOMINAL
        self._last_sequence: int | None = None
        self._healthy_recovery_steps = 0
        self._mrm_requested = False
        self._authorized_recovery_token: str | None = None

    @property
    def assurance_state(self) -> AssuranceState:
        return self._assurance_state

    @property
    def mode(self) -> AssuranceState:
        """Compatibility alias for integrations that call the state a mode."""
        return self._assurance_state

    @property
    def recovery_ready(self) -> bool:
        """Whether the backend may issue a guarded recovery token.

        Token issuance is intentionally separate from authorization.  A token
        is only issued after the vehicle has remained in a minimal-risk
        condition for the configured healthy physics steps; authorization is
        still checked again by ``step`` against the current state.
        """
        return bool(
            self._assurance_state is AssuranceState.MINIMAL_RISK_CONDITION
            and self._healthy_recovery_steps >= self.config.recovery_healthy_steps
        )

    def request_mrm(self, *, reason: str) -> None:
        """Latch an explicit benchmark minimal-risk-stop request."""
        if not reason.strip():
            raise ValueError("an explicit MRM request requires a reason")
        self._mrm_requested = True
        self._authorized_recovery_token = None
        self._assurance_state = AssuranceState.MRM_ACTIVE

    def authorize_recovery(self, *, token: str) -> bool:
        """Queue a token for guarded recovery; it never bypasses health dwell."""
        if not token:
            raise ValueError("a recovery authorization requires a token")
        if self._assurance_state not in {
            AssuranceState.MRM_ACTIVE,
            AssuranceState.MINIMAL_RISK_CONDITION,
            AssuranceState.RECOVERY_PENDING,
        }:
            return False
        self._authorized_recovery_token = token
        return True

    def step(
        self,
        state: SafetyState,
        command: ControlCommand,
        *,
        recovery_token: str | None = None,
    ) -> SafetyDecision:
        """Validate, predict, arbitrate, and return the command to apply."""
        invalid_reasons = self._validate_inputs(state, command)
        if self._mrm_requested:
            self._mrm_requested = False
            self._healthy_recovery_steps = 0
            return self._fallback_decision(
                state,
                command,
                _ordered_unique((*invalid_reasons, "mrm_requested")),
            )
        if invalid_reasons:
            self._healthy_recovery_steps = 0
            return self._fallback_decision(state, command, invalid_reasons)

        self._last_sequence = command.sequence
        normalized, normalization_reasons = self._normalize_command(state, command)
        candidates = self._build_candidates(state, normalized)
        assessments = tuple(self._assess_candidate(state, candidate) for candidate in candidates)
        nominal_assessment = assessments[0]
        minimum_ttc, minimum_headway, gap, required_distance = self._longitudinal_risk(state)
        risk_reasons = list(normalization_reasons)

        braking_risk = (
            gap is not None and required_distance is not None and gap <= required_distance
        )
        emergency_ttc = minimum_ttc is not None and minimum_ttc <= self.config.emergency_ttc_s
        if braking_risk:
            risk_reasons.append("braking_distance_insufficient")
        if emergency_ttc:
            risk_reasons.append("emergency_ttc")
        headway_risk = (
            minimum_headway is not None and minimum_headway <= self.config.min_time_headway_s
        )
        if headway_risk:
            risk_reasons.append("time_headway_insufficient")
        if not nominal_assessment.collision_free:
            risk_reasons.append("trajectory_conflict")
        if not nominal_assessment.on_road:
            risk_reasons.append("road_boundary_violation")

        evidence = SafetyEvidence(
            reason_codes=_ordered_unique(risk_reasons),
            minimum_ttc_s=minimum_ttc,
            minimum_time_headway_s=minimum_headway,
            nearest_longitudinal_gap_m=gap,
            required_stopping_distance_m=required_distance,
            candidates=assessments,
            selected_candidate="nominal",
            rollout_step_s=self.config.step_s,
            rollout_horizon_s=self.config.horizon_s,
        )
        hard_risk = (
            braking_risk
            or emergency_ttc
            or not nominal_assessment.collision_free
            or not nominal_assessment.on_road
        )
        healthy = not hard_risk and not normalization_reasons

        effective_recovery_token = (
            recovery_token if recovery_token is not None else self._authorized_recovery_token
        )
        latched = self._handle_latched_state(
            state,
            command,
            normalized,
            evidence,
            healthy=healthy,
            recovery_token=effective_recovery_token,
        )
        if latched is not None:
            return latched

        if hard_risk:
            selected, _ = self._select_candidate(
                candidates[1:],
                assessments[1:],
            )
            self._assurance_state = AssuranceState.EMERGENCY_OVERRIDE
            evidence = replace(
                evidence,
                selected_candidate=selected.name,
            )
            return SafetyDecision(
                nominal_command=command,
                applied_command=selected.command,
                intervention=InterventionKind.OVERRIDE,
                assurance_state=self._assurance_state,
                evidence=evidence,
            )

        warning_risk = (
            minimum_ttc is not None and minimum_ttc <= self.config.warning_ttc_s
        ) or headway_risk
        if warning_risk:
            comfortable_index = next(
                index
                for index, candidate in enumerate(candidates)
                if candidate.name == "comfortable_brake"
            )
            selected = candidates[comfortable_index]
            selected_assessment = assessments[comfortable_index]
            if not (selected_assessment.collision_free and selected_assessment.on_road):
                selected, _ = self._select_candidate(
                    candidates[1:],
                    assessments[1:],
                )
            self._assurance_state = AssuranceState.DEGRADED
            evidence = replace(
                evidence,
                reason_codes=_ordered_unique((*evidence.reason_codes, "warning_ttc")),
                selected_candidate=selected.name,
            )
            return SafetyDecision(
                nominal_command=command,
                applied_command=selected.command,
                intervention=InterventionKind.CLIP,
                assurance_state=self._assurance_state,
                evidence=evidence,
            )

        if normalization_reasons:
            self._assurance_state = AssuranceState.DEGRADED
            return SafetyDecision(
                nominal_command=command,
                applied_command=normalized,
                intervention=InterventionKind.CLIP,
                assurance_state=self._assurance_state,
                evidence=evidence,
            )

        self._assurance_state = AssuranceState.NOMINAL
        return SafetyDecision(
            nominal_command=command,
            applied_command=command,
            intervention=InterventionKind.PASS,
            assurance_state=self._assurance_state,
            evidence=evidence,
        )

    def _validate_inputs(
        self,
        state: SafetyState,
        command: ControlCommand,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        state_values = (
            state.sim_time,
            state.observed_at,
            state.ego.x,
            state.ego.y,
            state.ego.heading_rad,
            state.ego.speed_mps,
            state.ego.acceleration_mps2,
            state.ego.length_m,
            state.ego.width_m,
        )
        actor_values = tuple(
            value
            for actor in state.actors
            for value in (
                actor.x,
                actor.y,
                actor.heading_rad,
                actor.speed_mps,
                actor.acceleration_mps2,
                actor.length_m,
                actor.width_m,
            )
        )
        road_values = tuple(
            value for point in state.road_boundary.polygon for value in (point.x, point.y)
        )
        if (
            not all(math.isfinite(value) for value in (*state_values, *actor_values))
            or state.ego.speed_mps < 0.0
            or state.ego.length_m <= 0.0
            or state.ego.width_m <= 0.0
            or any(
                actor.speed_mps < 0.0 or actor.length_m <= 0.0 or actor.width_m <= 0.0
                for actor in state.actors
            )
            or len(state.road_boundary.polygon) < 3
            or not all(math.isfinite(value) for value in road_values)
        ):
            reasons.append("state_invalid")
        elif state.observed_at > state.sim_time + 1e-9:
            reasons.append("state_from_future")
        elif state.sim_time - state.observed_at > self.config.max_state_age_s:
            reasons.append("state_stale")

        command_values = (
            command.issued_at,
            command.valid_from,
            command.valid_until,
            command.acceleration_mps2,
            command.steering_rad,
        )
        if (
            not command.command_id
            or command.sequence < 0
            or not all(math.isfinite(value) for value in command_values)
            or command.valid_until <= command.valid_from
        ):
            reasons.append("command_invalid")
        else:
            if command.state_version != state.state_version:
                reasons.append("command_state_version_mismatch")
            if command.issued_at > state.sim_time + 1e-9:
                reasons.append("command_from_future")
            elif state.sim_time - command.issued_at > self.config.max_command_age_s:
                reasons.append("command_stale")
            if state.sim_time < command.valid_from:
                reasons.append("command_not_yet_valid")
            if state.sim_time >= command.valid_until:
                reasons.append("command_expired")
            if self._last_sequence is not None and command.sequence <= self._last_sequence:
                reasons.append("command_out_of_sequence")
        return _ordered_unique(reasons)

    def _normalize_command(
        self,
        state: SafetyState,
        command: ControlCommand,
    ) -> tuple[ControlCommand, tuple[str, ...]]:
        reasons: list[str] = []
        acceleration = min(
            self.config.maximum_acceleration_mps2,
            max(-self.config.emergency_deceleration_mps2, command.acceleration_mps2),
        )
        steering = min(
            self.config.maximum_steering_rad,
            max(-self.config.maximum_steering_rad, command.steering_rad),
        )
        if acceleration != command.acceleration_mps2 or steering != command.steering_rad:
            reasons.append("control_bounds_clipped")

        if command.lateral_escape and not self._valid_lateral_mark(state, command):
            steering = 0.0
            reasons.append("lateral_escape_unverified")
        return (
            replace(
                command,
                acceleration_mps2=acceleration,
                steering_rad=steering,
            ),
            _ordered_unique(reasons),
        )

    def _valid_lateral_mark(
        self,
        state: SafetyState,
        command: ControlCommand,
    ) -> bool:
        mark = command.lateral_verification
        return bool(
            mark is not None
            and mark.verifier_id.strip()
            and mark.state_version == state.state_version
            and mark.command_id == command.command_id
            and mark.acceleration_mps2 == command.acceleration_mps2
            and mark.steering_rad == command.steering_rad
            and math.isfinite(mark.valid_until)
            and state.sim_time < mark.valid_until
        )

    def _build_candidates(
        self,
        state: SafetyState,
        normalized: ControlCommand,
    ) -> tuple[_Candidate, ...]:
        comfortable = replace(
            normalized,
            acceleration_mps2=-self.config.comfortable_deceleration_mps2,
            steering_rad=0.0,
        )
        emergency = replace(
            normalized,
            acceleration_mps2=-self.config.emergency_deceleration_mps2,
            steering_rad=0.0,
        )
        candidates = [
            _Candidate("nominal", normalized, 0),
            _Candidate("comfortable_brake", comfortable, 1),
            _Candidate("emergency_brake", emergency, 2),
        ]
        if normalized.lateral_escape and self._valid_lateral_mark(state, normalized):
            candidates.append(
                _Candidate(
                    "verified_lateral_escape",
                    replace(
                        normalized,
                        acceleration_mps2=(-self.config.emergency_deceleration_mps2),
                    ),
                    3,
                )
            )
        return tuple(candidates)

    def _assess_candidate(
        self,
        state: SafetyState,
        candidate: _Candidate,
    ) -> CandidateAssessment:
        ego_rollout = self._rollout(
            state.ego,
            acceleration=candidate.command.acceleration_mps2,
            steering=candidate.command.steering_rad,
        )
        relevant_actors = tuple(
            actor
            for actor in sorted(state.actors, key=lambda item: item.actor_id)
            if self._could_reach_ego(
                state.ego,
                actor,
                ego_steering_rad=candidate.command.steering_rad,
            )
        )
        actor_rollouts = tuple(
            self._rollout(
                actor,
                acceleration=actor.acceleration_mps2,
                steering=0.0,
            )
            for actor in relevant_actors
        )
        collision_free = True
        conflict_time: float | None = None
        impact_speed = 0.0
        on_road = True
        boundary_margin = math.inf

        initial_polygon = _oriented_box(
            ego_rollout[0],
            self.config.collision_margin_m,
        )
        initial_margin = _footprint_boundary_margin(
            initial_polygon,
            state.road_boundary.polygon,
        )
        boundary_margin = min(boundary_margin, initial_margin)
        on_road = on_road and initial_margin >= 0.0
        for actor_rollout in actor_rollouts:
            actor_polygon = _oriented_box(
                actor_rollout[0],
                self.config.collision_margin_m,
            )
            if _convex_polygons_overlap(initial_polygon, actor_polygon):
                collision_free = False
                conflict_time = 0.0
                impact_speed = _relative_speed(
                    ego_rollout[0],
                    actor_rollout[0],
                )
                break

        for index in range(1, len(ego_rollout)):
            ego_before = ego_rollout[index - 1]
            ego_after = ego_rollout[index]
            ego_swept = _convex_hull(
                (
                    *_oriented_box(
                        ego_before,
                        self.config.collision_margin_m,
                    ),
                    *_oriented_box(
                        ego_after,
                        self.config.collision_margin_m,
                    ),
                )
            )
            current_margin = _footprint_boundary_margin(
                ego_swept,
                state.road_boundary.polygon,
            )
            boundary_margin = min(boundary_margin, current_margin)
            on_road = on_road and current_margin >= 0.0

            for actor_rollout in actor_rollouts:
                actor_before = actor_rollout[index - 1]
                actor_after = actor_rollout[index]
                actor_swept = _convex_hull(
                    (
                        *_oriented_box(
                            actor_before,
                            self.config.collision_margin_m,
                        ),
                        *_oriented_box(
                            actor_after,
                            self.config.collision_margin_m,
                        ),
                    )
                )
                if _convex_polygons_overlap(ego_swept, actor_swept):
                    collision_free = False
                    if conflict_time is None:
                        conflict_time = index * self.config.step_s
                        impact_speed = _relative_speed(ego_after, actor_after)

        progress = math.hypot(
            ego_rollout[-1].x - state.ego.x,
            ego_rollout[-1].y - state.ego.y,
        )
        return CandidateAssessment(
            name=candidate.name,
            collision_free=collision_free,
            on_road=on_road,
            conflict_time_s=conflict_time,
            boundary_margin_m=boundary_margin,
            predicted_impact_speed_mps=impact_speed,
            progress_m=progress,
        )

    def _could_reach_ego(
        self,
        ego: VehicleState,
        actor: VehicleState,
        *,
        ego_steering_rad: float,
    ) -> bool:
        """Conservative broad phase before expensive swept-polygon rollout."""
        if (
            abs(ego_steering_rad) <= 1e-12
            and abs(_normalize_angle(actor.heading_rad - ego.heading_rad)) <= 1e-6
        ):
            sine = math.sin(ego.heading_rad)
            cosine = math.cos(ego.heading_rad)
            longitudinal = (actor.x - ego.x) * cosine + (actor.y - ego.y) * sine
            lateral = -(actor.x - ego.x) * sine + (actor.y - ego.y) * cosine
            lateral_overlap = 0.5 * (ego.width_m + actor.width_m)
            if abs(lateral) > lateral_overlap + self.config.collision_margin_m:
                return False
            if not self.config.consider_rear_collision and longitudinal < -0.5 * (
                ego.length_m + actor.length_m
            ):
                return False
        separation = math.hypot(actor.x - ego.x, actor.y - ego.y)
        closing_bound = (
            max(0.0, ego.speed_mps)
            + max(0.0, actor.speed_mps)
            + abs(ego.acceleration_mps2) * self.config.horizon_s
            + abs(actor.acceleration_mps2) * self.config.horizon_s
        ) * self.config.horizon_s
        footprint = 0.5 * (ego.length_m + ego.width_m + actor.length_m + actor.width_m)
        return separation <= (closing_bound + footprint + self.config.collision_margin_m)

    def _rollout(
        self,
        initial: VehicleState,
        *,
        acceleration: float,
        steering: float,
    ) -> tuple[VehicleState, ...]:
        result = [initial]
        current = initial
        steps = round(self.config.horizon_s / self.config.step_s)
        for _ in range(steps):
            next_speed = max(
                0.0,
                current.speed_mps + acceleration * self.config.step_s,
            )
            yaw_rate = current.speed_mps / self.config.wheelbase_m * math.tan(steering)
            heading_delta = yaw_rate * self.config.step_s
            travel = 0.5 * (current.speed_mps + next_speed) * self.config.step_s
            mid_heading = current.heading_rad + 0.5 * heading_delta
            current = replace(
                current,
                x=current.x + travel * math.cos(mid_heading),
                y=current.y + travel * math.sin(mid_heading),
                heading_rad=_normalize_angle(current.heading_rad + heading_delta),
                speed_mps=next_speed,
                acceleration_mps2=acceleration,
            )
            result.append(current)
        return tuple(result)

    def _longitudinal_risk(
        self,
        state: SafetyState,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        cosine = math.cos(state.ego.heading_rad)
        sine = math.sin(state.ego.heading_rad)
        minimum_ttc: float | None = None
        minimum_headway: float | None = None
        nearest_gap: float | None = None
        required_at_nearest: float | None = None
        for actor in sorted(state.actors, key=lambda item: item.actor_id):
            relative_x = actor.x - state.ego.x
            relative_y = actor.y - state.ego.y
            longitudinal = relative_x * cosine + relative_y * sine
            lateral = -relative_x * sine + relative_y * cosine
            lane_threshold = (
                0.5 * (state.ego.width_m + actor.width_m) + self.config.uncertainty_margin_m
            )
            if longitudinal <= 0.0 or abs(lateral) > lane_threshold:
                continue
            gap = max(
                0.0,
                longitudinal - 0.5 * (state.ego.length_m + actor.length_m),
            )
            actor_longitudinal_speed = max(
                0.0,
                actor.speed_mps * math.cos(actor.heading_rad - state.ego.heading_rad),
            )
            closing_speed = state.ego.speed_mps - actor_longitudinal_speed
            ttc = 0.0 if gap == 0.0 else None
            if gap > 0.0 and closing_speed > 0.0:
                ttc = gap / closing_speed
            if ttc is not None and (minimum_ttc is None or ttc < minimum_ttc):
                minimum_ttc = ttc
            headway = gap / max(state.ego.speed_mps, 1e-9)
            if minimum_headway is None or headway < minimum_headway:
                minimum_headway = headway

            required = max(
                0.0,
                state.ego.speed_mps * self.config.reaction_time_s
                + state.ego.speed_mps**2 / (2.0 * self.config.emergency_deceleration_mps2)
                - actor_longitudinal_speed**2 / (2.0 * self.config.assumed_lead_deceleration_mps2),
            )
            required += self.config.standstill_gap_m + self.config.uncertainty_margin_m
            if nearest_gap is None or gap < nearest_gap:
                nearest_gap = gap
                required_at_nearest = required
        return minimum_ttc, minimum_headway, nearest_gap, required_at_nearest

    def _select_candidate(
        self,
        candidates: tuple[_Candidate, ...],
        assessments: tuple[CandidateAssessment, ...],
    ) -> tuple[_Candidate, CandidateAssessment]:
        paired = tuple(zip(candidates, assessments, strict=True))
        return max(
            paired,
            key=lambda pair: (
                int(pair[1].collision_free),
                int(pair[1].on_road),
                -pair[1].predicted_impact_speed_mps,
                pair[1].progress_m,
                -abs(pair[0].command.steering_rad),
                -pair[0].order,
            ),
        )

    def _handle_latched_state(
        self,
        state: SafetyState,
        nominal: ControlCommand,
        normalized: ControlCommand,
        evidence: SafetyEvidence,
        *,
        healthy: bool,
        recovery_token: str | None,
    ) -> SafetyDecision | None:
        if self._assurance_state is AssuranceState.EMERGENCY_OVERRIDE:
            if state.ego.speed_mps <= self.config.stopped_speed_mps:
                self._assurance_state = AssuranceState.MINIMAL_RISK_CONDITION
            else:
                self._assurance_state = AssuranceState.MRM_ACTIVE
            self._healthy_recovery_steps = 0
            return self._hold_or_stop_decision(state, nominal, evidence)

        if self._assurance_state is AssuranceState.MRM_ACTIVE:
            if state.ego.speed_mps <= self.config.stopped_speed_mps:
                self._assurance_state = AssuranceState.MINIMAL_RISK_CONDITION
            return self._hold_or_stop_decision(state, nominal, evidence)

        if self._assurance_state is AssuranceState.MINIMAL_RISK_CONDITION:
            if not healthy:
                self._healthy_recovery_steps = 0
                self._assurance_state = AssuranceState.MRM_ACTIVE
                self._authorized_recovery_token = None
                return self._hold_or_stop_decision(state, nominal, evidence)
            self._healthy_recovery_steps += 1
            token_valid = self._valid_recovery_token(recovery_token)
            reasons = list(evidence.reason_codes)
            if recovery_token is not None and not token_valid:
                reasons.append("recovery_token_invalid")
            if token_valid and self._healthy_recovery_steps >= self.config.recovery_healthy_steps:
                self._assurance_state = AssuranceState.RECOVERY_PENDING
                reasons.append("recovery_armed")
            return self._hold_or_stop_decision(
                state,
                nominal,
                replace(evidence, reason_codes=_ordered_unique(reasons)),
            )

        if self._assurance_state is AssuranceState.RECOVERY_PENDING:
            if not healthy or not self._valid_recovery_token(recovery_token):
                self._assurance_state = AssuranceState.MINIMAL_RISK_CONDITION
                self._healthy_recovery_steps = 0
                self._authorized_recovery_token = None
                guard_reasons = (*evidence.reason_codes, "recovery_guard_failed")
                return self._hold_or_stop_decision(
                    state,
                    nominal,
                    replace(
                        evidence,
                        reason_codes=_ordered_unique(guard_reasons),
                    ),
                )
            self._assurance_state = AssuranceState.NOMINAL
            self._healthy_recovery_steps = 0
            self._authorized_recovery_token = None
            return SafetyDecision(
                nominal_command=nominal,
                applied_command=normalized,
                intervention=(
                    InterventionKind.PASS if normalized == nominal else InterventionKind.CLIP
                ),
                assurance_state=self._assurance_state,
                evidence=replace(
                    evidence,
                    reason_codes=_ordered_unique((*evidence.reason_codes, "recovery_complete")),
                ),
            )
        return None

    def _valid_recovery_token(self, recovery_token: str | None) -> bool:
        return bool(
            self.config.recovery_token is not None
            and recovery_token is not None
            and recovery_token == self.config.recovery_token
        )

    def _fallback_decision(
        self,
        state: SafetyState,
        command: ControlCommand,
        reasons: tuple[str, ...],
    ) -> SafetyDecision:
        if state.ego.speed_mps <= self.config.stopped_speed_mps:
            self._assurance_state = AssuranceState.MINIMAL_RISK_CONDITION
        else:
            self._assurance_state = AssuranceState.MRM_ACTIVE
        applied = self._fallback_command(
            command,
            at_rest=(state.ego.speed_mps <= self.config.stopped_speed_mps),
        )
        evidence = SafetyEvidence(
            reason_codes=reasons,
            minimum_ttc_s=None,
            minimum_time_headway_s=None,
            nearest_longitudinal_gap_m=None,
            required_stopping_distance_m=None,
            candidates=(),
            selected_candidate="mrm_fallback",
            rollout_step_s=self.config.step_s,
            rollout_horizon_s=self.config.horizon_s,
        )
        return SafetyDecision(
            nominal_command=command,
            applied_command=applied,
            intervention=InterventionKind.FALLBACK,
            assurance_state=self._assurance_state,
            evidence=evidence,
        )

    def _hold_or_stop_decision(
        self,
        state: SafetyState,
        command: ControlCommand,
        evidence: SafetyEvidence,
    ) -> SafetyDecision:
        at_rest = state.ego.speed_mps <= self.config.stopped_speed_mps
        applied = self._fallback_command(command, at_rest=at_rest)
        return SafetyDecision(
            nominal_command=command,
            applied_command=applied,
            intervention=InterventionKind.FALLBACK,
            assurance_state=self._assurance_state,
            evidence=replace(evidence, selected_candidate="mrm_hold"),
        )

    def _fallback_command(
        self,
        command: ControlCommand,
        *,
        at_rest: bool,
    ) -> ControlCommand:
        return replace(
            command,
            acceleration_mps2=(0.0 if at_rest else -self.config.emergency_deceleration_mps2),
            steering_rad=0.0,
            lateral_escape=False,
            lateral_verification=None,
        )


def _ordered_unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _oriented_box(
    vehicle: VehicleState,
    margin: float,
) -> tuple[Point2D, ...]:
    half_length = 0.5 * vehicle.length_m + margin
    half_width = 0.5 * vehicle.width_m + margin
    cosine = math.cos(vehicle.heading_rad)
    sine = math.sin(vehicle.heading_rad)
    result = []
    for longitudinal, lateral in (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ):
        result.append(
            Point2D(
                vehicle.x + longitudinal * cosine - lateral * sine,
                vehicle.y + longitudinal * sine + lateral * cosine,
            )
        )
    return tuple(result)


def _convex_hull(points: tuple[Point2D, ...]) -> tuple[Point2D, ...]:
    ordered = sorted({(point.x, point.y) for point in points})
    if len(ordered) <= 1:
        return tuple(Point2D(*point) for point in ordered)

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return tuple(Point2D(*point) for point in lower[:-1] + upper[:-1])


def _convex_polygons_overlap(
    first: tuple[Point2D, ...],
    second: tuple[Point2D, ...],
) -> bool:
    if len(first) < 3 or len(second) < 3:
        return False
    for polygon in (first, second):
        for index, point in enumerate(polygon):
            following = polygon[(index + 1) % len(polygon)]
            axis_x = -(following.y - point.y)
            axis_y = following.x - point.x
            first_projection = tuple(item.x * axis_x + item.y * axis_y for item in first)
            second_projection = tuple(item.x * axis_x + item.y * axis_y for item in second)
            if (
                max(first_projection) < min(second_projection) - 1e-9
                or max(second_projection) < min(first_projection) - 1e-9
            ):
                return False
    return True


def _footprint_boundary_margin(
    footprint: tuple[Point2D, ...],
    boundary: tuple[Point2D, ...],
) -> float:
    footprint_edges = _polygon_edges(footprint)
    boundary_edges = _polygon_edges(boundary)
    inside = all(_point_in_polygon(point, boundary) for point in footprint)
    crosses_boundary = any(
        _segments_properly_intersect(
            footprint_start,
            footprint_end,
            boundary_start,
            boundary_end,
        )
        for footprint_start, footprint_end in footprint_edges
        for boundary_start, boundary_end in boundary_edges
    )
    distance = min(
        _segment_distance(
            footprint_start,
            footprint_end,
            boundary_start,
            boundary_end,
        )
        for footprint_start, footprint_end in footprint_edges
        for boundary_start, boundary_end in boundary_edges
    )
    return distance if inside and not crosses_boundary else -max(distance, 1e-12)


def _point_in_polygon(point: Point2D, polygon: tuple[Point2D, ...]) -> bool:
    inside = False
    for first, second in _polygon_edges(polygon):
        if _point_segment_distance(point, first, second) <= 1e-9:
            return True
        if (first.y > point.y) == (second.y > point.y):
            continue
        intersection_x = (second.x - first.x) * (point.y - first.y) / (second.y - first.y) + first.x
        if point.x < intersection_x:
            inside = not inside
    return inside


def _polygon_edges(
    polygon: tuple[Point2D, ...],
) -> tuple[tuple[Point2D, Point2D], ...]:
    return tuple(
        (point, polygon[(index + 1) % len(polygon)]) for index, point in enumerate(polygon)
    )


def _point_segment_distance(
    point: Point2D,
    first: Point2D,
    second: Point2D,
) -> float:
    delta_x = second.x - first.x
    delta_y = second.y - first.y
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared == 0.0:
        return math.hypot(point.x - first.x, point.y - first.y)
    projection = ((point.x - first.x) * delta_x + (point.y - first.y) * delta_y) / length_squared
    projection = min(1.0, max(0.0, projection))
    closest_x = first.x + projection * delta_x
    closest_y = first.y + projection * delta_y
    return math.hypot(point.x - closest_x, point.y - closest_y)


def _segment_distance(
    first_start: Point2D,
    first_end: Point2D,
    second_start: Point2D,
    second_end: Point2D,
) -> float:
    if _segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        _point_segment_distance(first_start, second_start, second_end),
        _point_segment_distance(first_end, second_start, second_end),
        _point_segment_distance(second_start, first_start, first_end),
        _point_segment_distance(second_end, first_start, first_end),
    )


def _segments_properly_intersect(
    first_start: Point2D,
    first_end: Point2D,
    second_start: Point2D,
    second_end: Point2D,
) -> bool:
    first_orientation = _orientation(first_start, first_end, second_start)
    second_orientation = _orientation(first_start, first_end, second_end)
    third_orientation = _orientation(second_start, second_end, first_start)
    fourth_orientation = _orientation(second_start, second_end, first_end)
    return (
        first_orientation * second_orientation < -1e-18
        and third_orientation * fourth_orientation < -1e-18
    )


def _segments_intersect(
    first_start: Point2D,
    first_end: Point2D,
    second_start: Point2D,
    second_end: Point2D,
) -> bool:
    if _segments_properly_intersect(
        first_start,
        first_end,
        second_start,
        second_end,
    ):
        return True
    return any(
        abs(_orientation(segment_start, segment_end, point)) <= 1e-9
        and _point_on_segment(point, segment_start, segment_end)
        for point, segment_start, segment_end in (
            (second_start, first_start, first_end),
            (second_end, first_start, first_end),
            (first_start, second_start, second_end),
            (first_end, second_start, second_end),
        )
    )


def _orientation(first: Point2D, second: Point2D, third: Point2D) -> float:
    return (second.x - first.x) * (third.y - first.y) - (second.y - first.y) * (third.x - first.x)


def _point_on_segment(
    point: Point2D,
    start: Point2D,
    end: Point2D,
) -> bool:
    return (
        min(start.x, end.x) - 1e-9 <= point.x <= max(start.x, end.x) + 1e-9
        and min(start.y, end.y) - 1e-9 <= point.y <= max(start.y, end.y) + 1e-9
    )


def _relative_speed(first: VehicleState, second: VehicleState) -> float:
    first_x = first.speed_mps * math.cos(first.heading_rad)
    first_y = first.speed_mps * math.sin(first.heading_rad)
    second_x = second.speed_mps * math.cos(second.heading_rad)
    second_y = second.speed_mps * math.sin(second.heading_rad)
    return math.hypot(first_x - second_x, first_y - second_y)


__all__ = [
    "AssuranceState",
    "CandidateAssessment",
    "ControlCommand",
    "InterventionKind",
    "LateralSafetyVerification",
    "Point2D",
    "RoadBoundary",
    "RuntimeAssurance",
    "RuntimeAssuranceConfig",
    "SafetyDecision",
    "SafetyEvidence",
    "SafetyState",
    "VehicleState",
]
