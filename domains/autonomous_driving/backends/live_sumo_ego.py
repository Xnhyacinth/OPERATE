"""Optional libsumo sibling for the ``sumo_ego`` backend.

Live execution is deliberately opt-in.  It reuses the deterministic tactical
and runtime-assurance logic from the fixture backend while sourcing the initial
ego/actor state from a pinned SUMO configuration.  Low-level SUMO vehicle
commands remain backend-owned and are never exposed as agent tools.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Any

from core.sidecar.sumo_sidecar import (
    SumoSidecar,
    probe_sumo_transport,
)
from domains.autonomous_driving.lane_geometry import (
    ExecutedStateSample,
    ExecutedTrajectoryCertificate,
    LaneChangeVerification,
    NativeVehicleState,
    PinnedLaneIndexMap,
    certify_executed_trajectory,
    load_pinned_lane_index_map,
    verify_lane_change_trajectory,
)

from .sumo_ego import (
    ActorState,
    EgoState,
    SumoEgoBackend,
    _digest,
    _source_event_decision_fields,
    formal_sumo_ego_source_evidence_ready,
)


def _source_state_digest(*, lane_index: int, speed_mps: float) -> str:
    """Hash source state at enough precision to preserve material changes."""
    return _digest(
        {
            "lane_index": int(lane_index),
            "speed_mps": round(float(speed_mps), 12),
        }
    )


def _rejected_lateral_verification(reason: str) -> LaneChangeVerification:
    return LaneChangeVerification(
        verified=False,
        reason_codes=(reason,),
        sample_count=0,
        trajectory_digest=("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    )


def _could_reach_lateral_rollout(
    ego: NativeVehicleState,
    actor: NativeVehicleState,
    *,
    duration_s: float,
    ego_acceleration_mps2: float,
) -> bool:
    separation = math.hypot(actor.x - ego.x, actor.y - ego.y)
    travel_bound = (ego.speed_mps + actor.speed_mps) * duration_s + 0.5 * abs(
        ego_acceleration_mps2
    ) * duration_s * duration_s
    footprint_bound = 0.5 * (ego.length_m + ego.width_m + actor.length_m + actor.width_m)
    return separation <= travel_bound + footprint_bound + 0.2


def _sumo_start_command(
    *, binary: str, config_path: str, seed: int, step_seconds: float
) -> list[str]:
    """Build the fully pinned live-SUMO command used by the pilot."""
    step = str(step_seconds)
    return [
        binary,
        "-c",
        config_path,
        "--seed",
        str(seed),
        "--step-length",
        step,
        "--default.action-step-length",
        step,
        "--collision.action",
        "warn",
        "--collision.check-junctions",
        "true",
        "--collision.mingap-factor",
        "0",
        "--time-to-teleport",
        "-1",
        "--step-method.ballistic",
        "true",
    ]


def live_sumo_available() -> bool:
    """Return whether explicitly enabled native SUMO transport is reachable.

    ``libsumo`` is preferred, but the official ``eclipse-sumo`` wheel on
    macOS arm64 commonly ships only the standalone binary plus TraCI tools.
    The generic sidecar already owns that TCP transport, so it is a valid
    native smoke path even when the in-process binding is absent.
    """
    if os.getenv("OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL") != "1":
        return False
    if importlib.util.find_spec("libsumo") is not None:
        return True
    try:
        import sumo  # type: ignore[import-not-found,import-untyped]
    except ImportError:
        return False
    tools_dir = Path(str(sumo.SUMO_HOME)) / "tools"
    if tools_dir.is_dir() and str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    return probe_sumo_transport() in {"libsumo", "traci"}


class LiveSumoEgoBackend(SumoEgoBackend):
    """SUMO-backed state sibling; physics and safety ownership stay local."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._sumo: Any = None
        self._sidecar: SumoSidecar | None = None
        self._sumo_started = False
        self._ego_vehicle_id = str(self._config.get("ego_vehicle_id") or "")
        self._pending_source_lane_targets: dict[str, tuple[str, int, dict[str, Any]]] = {}
        self._pending_source_speed_targets: dict[
            str, tuple[str, int, float, float, dict[str, Any]]
        ] = {}
        self._last_live_lateral_verification: LaneChangeVerification | None = None
        self._pinned_lane_maps: dict[str, PinnedLaneIndexMap] = {}

    @property
    def native_transport(self) -> str | None:
        """Expose the selected native transport for diagnostic reports."""
        return self._sidecar.transport if self._sidecar is not None else None

    @property
    def native_runtime_version(self) -> str | None:
        """Expose the connected SUMO version while the sibling is running."""
        if self._sidecar is None or not self._sumo_started:
            return None
        return self._sidecar.runtime_version()

    def source_consumption_evidence(self) -> dict[str, Any]:
        """Promote source evidence only after native SUMO consumed the trace.

        The fixture/emulator intentionally remains ``held``.  A live sibling
        has a separate evidence disposition once the locked bundle was opened,
        source events were applied in TraCI/libsumo, and post-event state
        digests were recorded. Readiness remains fail-closed on those runtime
        facts while treating the host platform as recorded metadata.
        """
        evidence = super().source_consumption_evidence()
        evidence["runtime_fidelity"] = "native_live_sumo_reactive"
        evidence["blockers"] = [] if self._sumo_started else ["native_sumo_not_started"]
        if formal_sumo_ego_source_evidence_ready(evidence):
            evidence["status"] = "verified"
            evidence["formal_admission"] = "native_live_sumo_source_consumption"
        else:
            evidence["status"] = "held"
            evidence["formal_admission"] = "held_incomplete_native_source_evidence"
            if not evidence["blockers"]:
                evidence["blockers"] = ["native_source_evidence_incomplete"]
        return evidence

    def reset(self, seed_obj: Any) -> None:
        if self._config.get("allow_live_lateral_maneuvers") is True:
            raise ValueError(
                "live lateral maneuvers are held: verified candidate trajectory "
                "is not bound to SUMO lane-change execution"
            )
        if not live_sumo_available():
            raise RuntimeError(
                "live sumo_ego requires OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1 and a native SUMO transport"
            )
        if not self._ego_vehicle_id:
            raise ValueError("live sumo_ego requires backend_config.ego_vehicle_id")
        config_path = str(self._config.get("sumo_config_path") or "")
        if not config_path:
            raise ValueError("live sumo_ego requires backend_config.sumo_config_path")
        self._reset_common(seed_obj)
        self._last_live_lateral_verification = None
        self._pinned_lane_maps.clear()
        self._pending_source_lane_targets.clear()
        self._pending_source_speed_targets.clear()
        # A live sibling may consume the same locked event schedule as the
        # source bundle.  The event identities are checked by the parent
        # loader; this path only applies the resulting high-level actor update
        # through TraCI and never exposes it as an agent action.
        if self._config.get("source_bundle"):
            fixture = self._load_fixture()
            self._source_events = [
                dict(row) for row in fixture.get("source_events") or [] if isinstance(row, dict)
            ]
        self._sidecar = SumoSidecar(
            net_path=str(self._config.get("sumo_net_path") or ""),
            route_path=str(self._config.get("sumo_route_path") or ""),
            config_path=config_path,
            seed=int(getattr(seed_obj, "seed", 0)),
            step_length=self._physics_dt_s,
            extra_args=(
                "--collision.action",
                "warn",
                "--collision.check-junctions",
                "true",
                "--collision.mingap-factor",
                "0",
                "--step-method.ballistic",
                "true",
            ),
        )
        try:
            self._sidecar.start()
            self._sumo = self._sidecar.connection
            self._sumo_started = True
        except Exception:
            self.close()
            raise
        from domains.autonomous_driving.data.contracts import file_sha256

        native_paths = [
            Path(config_path).expanduser().resolve(),
            Path(str(self._config.get("sumo_net_path") or "")).expanduser().resolve(),
            Path(str(self._config.get("sumo_route_path") or "")).expanduser().resolve(),
        ]
        if not all(path.is_file() for path in native_paths):
            self.close()
            raise ValueError("live sumo_ego native source asset missing")
        for path in native_paths:
            if path not in self._source_paths:
                self._source_paths.append(path)
            self._source_sha256s[str(path)] = file_sha256(path)
        self._prime_until_ego()
        self._sumo.vehicle.setSpeedMode(
            self._ego_vehicle_id,
            int(self._config.get("speed_mode") or 31),
        )
        self._sumo.vehicle.setLaneChangeMode(
            self._ego_vehicle_id,
            int(self._config.get("lane_change_mode") or 0),
        )
        # Disable autonomous lane changes for every source actor while keeping
        # TraCI-only source disturbances possible. Mode 512 means a vehicle
        # changes lane only when this backend applies a locked source event;
        # SUMO's own strategic/cooperative model cannot rewrite the hazard.
        for actor_id in self._sumo.vehicle.getIDList():
            actor_id = str(actor_id)
            if actor_id == self._ego_vehicle_id:
                continue
            self._sumo.vehicle.setLaneChangeMode(actor_id, 512)
        # Source event actors are held at their normalized initial velocity
        # until the locked disturbance fires.  Otherwise SUMO's free-road
        # acceleration can erase the mined lead/follower relation before the
        # event tick.  The actor still advances in native SUMO and the event
        # applies its source acceleration through ``setAcceleration``.
        scheduled_actor_ids = {
            str(row.get("actor_id") or "")
            for row in self._source_events
            if str(row.get("actor_id") or "")
        }
        for actor_id in sorted(scheduled_actor_ids):
            if actor_id not in set(self._sumo.vehicle.getIDList()):
                raise RuntimeError(f"live source event actor is absent: {actor_id}")
            current_speed = float(self._sumo.vehicle.getSpeed(actor_id))
            self._sumo.vehicle.setSpeedMode(actor_id, 0)
            self._sumo.vehicle.setSpeed(actor_id, current_speed)
        self._refresh_from_sumo()
        route = list(self._sumo.vehicle.getRoute(self._ego_vehicle_id))
        route_length = self._route_length(route)
        self._route_length_m = max(1.0, route_length)
        self._initialize_assurance()
        self._source_initial_state_digest = _digest(self._source_state())

    def request_tactical_maneuver(self, args: dict[str, Any]) -> dict[str, Any]:
        """Accept a live lane change only with a current native certificate."""
        maneuver = str(args.get("maneuver") or "")
        if maneuver not in {"change_lane_left", "change_lane_right"}:
            return super().request_tactical_maneuver(args)
        verification = _rejected_lateral_verification(
            "native_lateral_execution_binding_unavailable"
        )
        self._last_live_lateral_verification = verification
        return {
            "_status": "error",
            "error": "live_lateral_maneuvers_disabled",
            "reason_codes": list(verification.reason_codes),
        }

    def inspect_safety_state(self) -> dict[str, Any]:
        state = super().inspect_safety_state()
        verification = self._last_live_lateral_verification
        if verification is not None:
            state["live_lateral_verification"] = {
                "verified": verification.verified,
                "reason_codes": list(verification.reason_codes),
                "sample_count": verification.sample_count,
                "trajectory_digest": verification.trajectory_digest,
                "conflict_time_s": verification.conflict_time_s,
            }
        return state

    def _missing_native_lateral_apis(self) -> tuple[str, ...]:
        if self._sumo is None:
            return ("runtime_connection",)
        required = (
            (self._sumo.lane, "getEdgeID"),
            (self._sumo.lane, "getShape"),
            (self._sumo.lane, "getWidth"),
            (self._sumo.vehicle, "getPosition"),
            (self._sumo.vehicle, "getAngle"),
            (self._sumo.vehicle, "getLength"),
            (self._sumo.vehicle, "getWidth"),
        )
        return tuple(name for owner, name in required if not callable(getattr(owner, name, None)))

    def _verify_live_lane_change(
        self,
        *,
        target_lane: int,
        acceleration_mps2: float,
    ) -> tuple[str | None, LaneChangeVerification]:
        """Build a certificate from the active network and active vehicles."""
        if self._sumo is None or self._sidecar is None:
            return None, _rejected_lateral_verification("native_runtime_not_started")
        if self._missing_native_lateral_apis():
            return None, _rejected_lateral_verification("native_lane_geometry_unavailable")
        try:
            current_lane_id = str(self._sumo.vehicle.getLaneID(self._ego_vehicle_id))
            current_lane = int(self._sumo.vehicle.getLaneIndex(self._ego_vehicle_id))
            edge_id = str(self._sumo.lane.getEdgeID(current_lane_id))
            if edge_id.startswith(":"):
                return None, _rejected_lateral_verification("internal_edge_lane_change_unsupported")
            lane_ids = self._sidecar.edge_lane_ids(edge_id)
            lane_map = self._pinned_lane_map(edge_id)
            if set(lane_ids) != {lane_id for _, lane_id in lane_map.lanes}:
                return None, _rejected_lateral_verification(
                    "pinned_runtime_lane_identity_mismatch"
                )
            if lane_map.lane_id(current_lane) != current_lane_id:
                return None, _rejected_lateral_verification("native_lane_identity_mismatch")
            if abs(target_lane - current_lane) != 1:
                return None, _rejected_lateral_verification("target_lane_not_adjacent")
            target_lane_id = lane_map.lane_id(target_lane)
            current_shape = tuple(
                (float(x), float(y)) for x, y in self._sumo.lane.getShape(current_lane_id)
            )
            target_shape = tuple(
                (float(x), float(y)) for x, y in self._sumo.lane.getShape(target_lane_id)
            )
            current_lane_width = float(self._sumo.lane.getWidth(current_lane_id))
            target_lane_width = float(self._sumo.lane.getWidth(target_lane_id))
            ego = self._native_vehicle_state(self._ego_vehicle_id)
            actors = tuple(
                self._native_vehicle_state(actor_id)
                for actor_id in sorted(str(value) for value in self._sumo.vehicle.getIDList())
                if actor_id != self._ego_vehicle_id
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None, _rejected_lateral_verification("native_lane_geometry_unavailable")

        duration_s = max(self._physics_dt_s, 1.0)
        relevant_actors = tuple(
            actor
            for actor in actors
            if _could_reach_lateral_rollout(
                ego,
                actor,
                duration_s=duration_s,
                ego_acceleration_mps2=acceleration_mps2,
            )
        )
        if len(relevant_actors) > 64:
            return target_lane_id, _rejected_lateral_verification(
                "lateral_actor_verification_budget_exceeded"
            )
        verification = verify_lane_change_trajectory(
            current_lane_shape=current_shape,
            target_lane_shape=target_shape,
            current_lane_width_m=current_lane_width,
            target_lane_width_m=target_lane_width,
            ego=ego,
            actors=relevant_actors,
            duration_s=duration_s,
            step_s=self._physics_dt_s,
            acceleration_mps2=acceleration_mps2,
        )
        return target_lane_id, verification

    def _pinned_lane_map(self, edge_id: str) -> PinnedLaneIndexMap:
        cached = self._pinned_lane_maps.get(edge_id)
        if cached is not None:
            return cached
        if self._sidecar is None:
            raise RuntimeError("live SUMO sidecar is not started")
        mapping = load_pinned_lane_index_map(
            self._sidecar.net_path,
            edge_id=edge_id,
            expected_sha256=(
                str(self._config["sumo_net_sha256"])
                if self._config.get("sumo_net_sha256")
                else None
            ),
        )
        self._pinned_lane_maps[edge_id] = mapping
        return mapping

    def _native_vehicle_state(self, vehicle_id: str) -> NativeVehicleState:
        if self._sumo is None:
            raise RuntimeError("live SUMO is not started")
        front_x, front_y = self._sumo.vehicle.getPosition(vehicle_id)
        heading = math.radians(90.0 - float(self._sumo.vehicle.getAngle(vehicle_id)))
        length = float(self._sumo.vehicle.getLength(vehicle_id))
        # SUMO/TraCI reports the centre of the front bumper, whereas the
        # footprint verifier uses a geometric centre.
        x = float(front_x) - 0.5 * length * math.cos(heading)
        y = float(front_y) - 0.5 * length * math.sin(heading)
        return NativeVehicleState(
            vehicle_id=vehicle_id,
            x=x,
            y=y,
            heading_rad=heading,
            speed_mps=float(self._sumo.vehicle.getSpeed(vehicle_id)),
            length_m=length,
            width_m=float(self._sumo.vehicle.getWidth(vehicle_id)),
        )

    def _native_lateral_readback(self, *, elapsed_s: float) -> ExecutedStateSample:
        state = self._native_vehicle_state(self._ego_vehicle_id)
        if self._sumo is None:
            raise RuntimeError("live SUMO is not started")
        return ExecutedStateSample(
            elapsed_s=float(elapsed_s),
            x=state.x,
            y=state.y,
            heading_rad=state.heading_rad,
            lane_id=str(self._sumo.vehicle.getLaneID(self._ego_vehicle_id)),
        )

    def _certify_live_lateral_shadow(
        self,
        *,
        candidate: LaneChangeVerification,
        readbacks: tuple[ExecutedStateSample, ...],
        current_lane_id: str,
        target_lane_id: str,
    ) -> ExecutedTrajectoryCertificate:
        if self._sumo is None:
            raise RuntimeError("live SUMO is not started")
        edge_id = str(self._sumo.lane.getEdgeID(current_lane_id))
        lane_map = self._pinned_lane_map(edge_id)
        if abs(
            lane_map.index_for(current_lane_id) - lane_map.index_for(target_lane_id)
        ) != 1:
            raise ValueError("lateral shadow certificate requires adjacent pinned lanes")
        return certify_executed_trajectory(
            candidate=candidate,
            readbacks=readbacks,
            network_sha256=lane_map.network_sha256,
            current_lane_id=current_lane_id,
            target_lane_id=target_lane_id,
        )

    def _apply_source_events(self, *, substep: int) -> list[dict[str, Any]]:
        """Apply locked source disturbances to native SUMO at outer ticks."""
        if self._sumo is None:
            raise RuntimeError("live SUMO is not started")
        realized: list[dict[str, Any]] = []
        for index, raw in enumerate(self._source_events):
            trigger = int(raw.get("trigger_tick") or 0)
            event_id = str(raw.get("event_id") or f"source-event-{index}")
            offset_ms = max(0, int(raw.get("trigger_offset_ms_within_tick") or 0))
            target_substep = min(
                self._substeps - 1,
                int(offset_ms / max(self._physics_dt_s * 1000.0, 1.0)),
            )
            if (
                trigger != self._tick
                or target_substep != substep
                or event_id in self._applied_source_events
            ):
                continue
            self._applied_source_events.add(event_id)
            actor_id = str(raw.get("actor_id") or "")
            if actor_id not in set(self._sumo.vehicle.getIDList()):
                raise RuntimeError(f"live source event actor is absent: {actor_id}")
            before_speed = float(self._sumo.vehicle.getSpeed(actor_id))
            before_lane = int(self._sumo.vehicle.getLaneIndex(actor_id))
            target_lane = before_lane
            kind = str(raw.get("kind") or "actor_state_update")
            if kind == "lead_vehicle_braking":
                acceleration = float(raw.get("acceleration_mps2") or 0.0)
                duration = max(0.1, float(raw.get("control_duration_s") or 0.1))
                self._sumo.vehicle.setSpeedMode(actor_id, 0)
                set_acceleration = getattr(self._sumo.vehicle, "setAcceleration", None)
                if not callable(set_acceleration):
                    raise RuntimeError("live SUMO vehicle.setAcceleration is unavailable")
                set_acceleration(actor_id, acceleration, duration)
                target_speed = max(0.0, before_speed + acceleration * duration)
            elif kind in {
                "actor_state_update",
                "cut_in_gap_boundary",
                "short_time_headway_boundary",
            }:
                target_speed = max(0.0, float(raw.get("speed_mps") or before_speed))
                self._sumo.vehicle.setSpeedMode(actor_id, 0)
                self._sumo.vehicle.setSpeed(actor_id, target_speed)
            elif kind == "lane_change_conflict":
                target_speed = max(0.0, float(raw.get("speed_mps") or before_speed))
                raw_target_lane = raw.get("lane_index")
                target_lane = before_lane if raw_target_lane is None else int(raw_target_lane)
                lane_count = int(
                    self._sumo.edge.getLaneNumber(self._sumo.vehicle.getRoadID(actor_id))
                )
                if target_lane < 0 or target_lane >= lane_count:
                    raise RuntimeError(f"live source event target lane is invalid: {target_lane}")
                self._sumo.vehicle.setSpeedMode(actor_id, 0)
                self._sumo.vehicle.setSpeed(actor_id, target_speed)
                if target_lane != before_lane:
                    # This is a source-derived actor disturbance, not an LLM
                    # maneuver. The target lane is locked by the source row;
                    # the ego's unvalidated live lateral control remains off.
                    # ``moveTo`` is used for the logged actor transition so a
                    # source event cannot be silently rejected by SUMO's
                    # autonomous lane-change safety policy.
                    current_lane_id = str(self._sumo.vehicle.getLaneID(actor_id))
                    lane_prefix = current_lane_id.rsplit("_", 1)[0]
                    move_to = getattr(self._sumo.vehicle, "moveTo", None)
                    if callable(move_to):
                        move_to(
                            actor_id,
                            f"{lane_prefix}_{target_lane}",
                            float(self._sumo.vehicle.getLanePosition(actor_id)),
                            2,
                        )
                    else:
                        self._sumo.vehicle.changeLane(
                            actor_id,
                            target_lane,
                            max(self._physics_dt_s, 1.0),
                        )
            else:
                raise RuntimeError(f"unsupported live source event kind: {kind}")
            after_speed = float(self._sumo.vehicle.getSpeed(actor_id))
            after_lane = int(self._sumo.vehicle.getLaneIndex(actor_id))
            # SUMO may reject or quantize a requested speed.  Materiality is
            # based on the state read back from SUMO, never on the command
            # target, so a no-op cannot masquerade as a source event.
            speed_changed = not math.isclose(before_speed, after_speed, rel_tol=0.0, abs_tol=1e-12)
            lane_changed = before_lane != after_lane
            changed = speed_changed or lane_changed
            materiality_passed = changed and (kind != "lane_change_conflict" or lane_changed)
            hidden = bool(raw.get("hidden", False))
            decision_fields = _source_event_decision_fields(
                kind=kind,
                changed=materiality_passed,
                hidden=hidden,
                tick=self._tick,
                horizon=self._horizon,
            )
            realized.append(
                {
                    "event_id": event_id,
                    "type": kind,
                    "origin": "source_schedule",
                    "tick": self._tick,
                    "physics_substep": substep,
                    "simulation_time_seconds": round(self._simulation_time_seconds, 6),
                    "hidden": hidden,
                    **decision_fields,
                    "changed_state_fields": (
                        (
                            ["speed_mps", "acceleration_mps2"]
                            if kind == "lead_vehicle_braking"
                            else []
                        )
                        + (["lane_index"] if lane_changed else [])
                        + (
                            ["speed_mps"]
                            if speed_changed and kind != "lead_vehicle_braking"
                            else []
                        )
                    ),
                    "materiality_metric": (
                        "native_source_actor_acceleration_applied"
                        if kind == "lead_vehicle_braking"
                        else "native_source_actor_lane_and_speed_changed"
                        if kind == "lane_change_conflict"
                        else "native_source_actor_speed_changed"
                    ),
                    "materiality_value": int(materiality_passed),
                    "materiality_threshold": 1,
                    "materiality_passed": materiality_passed,
                    "before_state_digest": _source_state_digest(
                        lane_index=before_lane, speed_mps=before_speed
                    ),
                    "after_state_digest": _source_state_digest(
                        lane_index=after_lane, speed_mps=after_speed
                    ),
                    "state_observation_kind": "native_backend_readback",
                    "source_event_ids": list(raw.get("source_event_ids") or [event_id]),
                    "actor_id": actor_id,
                }
            )
            self._source_event_evidence[event_id] = realized[-1]
            if kind == "lane_change_conflict" and target_lane != before_lane:
                self._pending_source_lane_targets[event_id] = (
                    actor_id,
                    target_lane,
                    realized[-1],
                )
            if kind in {
                "lead_vehicle_braking",
                "actor_state_update",
                "cut_in_gap_boundary",
                "short_time_headway_boundary",
                "lane_change_conflict",
            }:
                self._pending_source_speed_targets[event_id] = (
                    actor_id,
                    before_lane,
                    before_speed,
                    target_speed,
                    realized[-1],
                )
        return realized

    def _route_length(self, route: list[str]) -> float:
        """Read route length from the exact loaded lane graph.

        TraCI exposes lane lengths, while ``edge.getLength`` is a libsumo-only
        convenience on some versions.  Using the sidecar's parsed topology
        keeps the libsumo and TraCI paths semantically identical.
        """
        if self._sumo is None or self._sidecar is None:
            raise RuntimeError("live SUMO is not started")
        total = 0.0
        for edge_id in route:
            total += self._edge_length(edge_id)
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError("runtime route has no positive length")
        return total

    def _edge_length(self, edge_id: str) -> float:
        if self._sumo is None or self._sidecar is None:
            raise RuntimeError("live SUMO is not started")
        lane_ids = self._sidecar.edge_lane_ids(edge_id)
        lengths = [float(self._sumo.lane.getLength(lane_id)) for lane_id in lane_ids]
        if not lengths or any(not math.isfinite(value) or value <= 0.0 for value in lengths):
            raise RuntimeError(f"runtime route edge has invalid lane lengths: {edge_id}")
        return max(lengths)

    def _vehicle_route_position(self, vehicle_id: str, lane_id: str) -> float:
        """Return absolute route position, including delayed-departure prefix."""
        if self._sumo is None:
            raise RuntimeError("live SUMO is not started")
        current_edge = str(self._sumo.lane.getEdgeID(lane_id))
        route = [str(value) for value in self._sumo.vehicle.getRoute(vehicle_id)]
        try:
            edge_index = route.index(current_edge)
        except ValueError as exc:
            raise RuntimeError(
                f"vehicle {vehicle_id} is on edge {current_edge!r} outside its route"
            ) from exc
        prefix = sum(self._edge_length(edge_id) for edge_id in route[:edge_index])
        return prefix + float(self._sumo.vehicle.getLanePosition(vehicle_id))

    def _prime_until_ego(self) -> None:
        if self._sumo is None:
            raise RuntimeError("live SUMO is not started")
        limit = max(1, int(self._config.get("ego_departure_timeout_steps") or 1000))
        for _ in range(limit):
            if self._ego_vehicle_id in set(self._sumo.vehicle.getIDList()):
                return
            self._sumo.simulationStep()
        raise RuntimeError("configured ego vehicle did not depart within timeout")

    def _refresh_from_sumo(self) -> None:
        if self._sumo is None:
            raise RuntimeError("live SUMO is not started")
        vehicle_ids = tuple(str(value) for value in self._sumo.vehicle.getIDList())
        if self._ego_vehicle_id not in vehicle_ids:
            raise RuntimeError("ego vehicle absent from live SUMO state")
        lane_id = str(self._sumo.vehicle.getLaneID(self._ego_vehicle_id))
        lane_index = int(self._sumo.vehicle.getLaneIndex(self._ego_vehicle_id))
        self._lane_width_m = float(self._sumo.lane.getWidth(lane_id))
        lateral = lane_index * self._lane_width_m + float(
            self._sumo.vehicle.getLateralLanePosition(self._ego_vehicle_id)
        )
        speed = float(self._sumo.vehicle.getSpeed(self._ego_vehicle_id))
        # ``SumoEgoBackend.tick`` keeps a reference to the ego object across
        # physics substeps.  Refresh that object in place; replacing it here
        # would make the outer loop combine stale ego state with fresh SUMO
        # actor state and can manufacture a trajectory conflict.
        ego: EgoState | None = getattr(self, "_ego", None)
        if ego is None:
            ego = EgoState(
                vehicle_id=self._ego_vehicle_id,
                route_position_m=0.0,
                lateral_position_m=0.0,
                lane_index=lane_index,
                speed_mps=speed,
            )
            self._ego = ego
        ego.vehicle_id = self._ego_vehicle_id
        ego.route_position_m = self._vehicle_route_position(self._ego_vehicle_id, lane_id)
        ego.lateral_position_m = lateral
        ego.lane_index = lane_index
        ego.speed_mps = speed
        ego.heading_rad = math.radians(
            90.0 - float(self._sumo.vehicle.getAngle(self._ego_vehicle_id))
        )
        self._actors = {}
        for actor_id in vehicle_ids:
            if actor_id == self._ego_vehicle_id:
                continue
            actor_lane = int(self._sumo.vehicle.getLaneIndex(actor_id))
            actor_lateral = actor_lane * self._lane_width_m + float(
                self._sumo.vehicle.getLateralLanePosition(actor_id)
            )
            self._actors[actor_id] = ActorState(
                actor_id=actor_id,
                route_position_m=self._vehicle_route_position(
                    actor_id,
                    str(self._sumo.vehicle.getLaneID(actor_id)),
                ),
                lateral_position_m=actor_lateral,
                lane_index=actor_lane,
                speed_mps=float(self._sumo.vehicle.getSpeed(actor_id)),
            )
        self._lane_count = max(
            1, int(self._sumo.edge.getLaneNumber(self._sumo.lane.getEdgeID(lane_id)))
        )
        self._speed_limit_mps = max(0.1, float(self._sumo.lane.getMaxSpeed(lane_id)))

    def _advance_physics(self, acceleration_mps2: float, steering_rad: float) -> None:
        if self._sumo is None:
            raise RuntimeError("live SUMO is not started")
        ego = self._require_ego()
        speed = max(
            0.0,
            min(
                self._speed_limit_mps,
                ego.speed_mps + acceleration_mps2 * self._physics_dt_s,
            ),
        )
        self._sumo.vehicle.setSpeed(self._ego_vehicle_id, speed)
        # Lateral intent is applied through SUMO's lane-change controller only;
        # the agent never receives a direct steering actuator.
        request: dict[str, Any] = dict(getattr(self, "_pending_maneuver", None) or {})
        if request.get("maneuver") in {"change_lane_left", "change_lane_right"}:
            verification = _rejected_lateral_verification(
                "native_lateral_execution_binding_unavailable"
            )
            self._last_live_lateral_verification = verification
            self._pending_maneuver = None
            if self._assurance is not None:
                self._assurance.request_mrm(
                    "live_lateral_verification_failed:native_lateral_execution_binding_unavailable"
                )
        # A source schedule can intentionally disturb the ego vehicle (the
        # logged risk-boundary state). Re-apply that one-substep target after
        # the nominal controller so the controller cannot erase the source
        # event before SUMO integrates it.
        for (
            actor_id,
            _before_lane,
            _before_speed,
            target_speed,
            _event,
        ) in self._pending_source_speed_targets.values():
            if actor_id == self._ego_vehicle_id:
                self._sumo.vehicle.setSpeedMode(actor_id, 0)
                self._sumo.vehicle.setSpeed(actor_id, target_speed)
        self._sumo.simulationStep()
        self._refresh_from_sumo()
        for event_id, (actor_id, target_lane, event) in list(
            self._pending_source_lane_targets.items()
        ):
            if actor_id not in set(self._sumo.vehicle.getIDList()):
                continue
            if int(self._sumo.vehicle.getLaneIndex(actor_id)) != target_lane:
                continue
            event["materiality_passed"] = True
            event["materiality_value"] = 1
            if "lane_index" not in event["changed_state_fields"]:
                event["changed_state_fields"].append("lane_index")
            event["after_state_digest"] = _source_state_digest(
                lane_index=target_lane,
                speed_mps=float(self._sumo.vehicle.getSpeed(actor_id)),
            )
            del self._pending_source_lane_targets[event_id]
        for event_id, (actor_id, before_lane, before_speed, _target_speed, event) in list(
            self._pending_source_speed_targets.items()
        ):
            if actor_id not in set(self._sumo.vehicle.getIDList()):
                continue
            after_speed = float(self._sumo.vehicle.getSpeed(actor_id))
            after_lane = int(self._sumo.vehicle.getLaneIndex(actor_id))
            speed_changed = not math.isclose(before_speed, after_speed, rel_tol=0.0, abs_tol=1e-12)
            lane_changed = before_lane != after_lane
            event["after_state_digest"] = _source_state_digest(
                lane_index=after_lane, speed_mps=after_speed
            )
            if speed_changed and "speed_mps" not in event["changed_state_fields"]:
                event["changed_state_fields"].append("speed_mps")
            if event["type"] == "lane_change_conflict":
                passed = speed_changed and lane_changed
            else:
                passed = speed_changed
            event["materiality_passed"] = bool(passed)
            event["materiality_value"] = int(passed)
            del self._pending_source_speed_targets[event_id]

    def _detect_safety_failures(self) -> None:
        """Combine geometry checks with SUMO-native collision/teleport facts."""
        super()._detect_safety_failures()
        if self._sumo is None:
            return
        colliding = {str(value) for value in self._sumo.simulation.getCollidingVehiclesIDList()}
        if self._ego_vehicle_id in colliding:
            self._collision_ids.update(colliding - {self._ego_vehicle_id})
            if len(colliding) == 1:
                self._collision_ids.add("sumo_collision_unknown_counterparty")
        teleporting = {str(value) for value in self._sumo.simulation.getStartingTeleportIDList()}
        if self._ego_vehicle_id in teleporting:
            self._road_departures = 1

    def close(self) -> None:
        if self._sidecar is not None:
            try:
                self._sidecar.close()
            finally:
                self._sumo_started = False
                self._sumo = None
                self._sidecar = None
