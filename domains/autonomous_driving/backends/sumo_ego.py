"""Deterministic ego-vehicle backend with an optional live SUMO sibling.

The benchmark agent only sets tactical intent.  A domain-local runtime-
assurance component owns emergency overrides and the backend owns all
sub-second actuator commands.  The pure-Python path consumes a bounded fixture
derived from a public source; the live sibling is loaded only when explicitly
requested and usable.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast


_SOURCE_EVENT_CLASS = MappingProxyType(
    {
        "actor_state_update": "telemetry",
        "cut_in": "safety",
        "cut_in_gap_boundary": "safety",
        "lane_change_conflict": "safety",
        "lead_vehicle_braking": "safety",
        "short_time_headway_boundary": "safety",
        "stopped_vehicle": "safety",
    }
)


def _source_event_decision_fields(
    *, kind: str, changed: bool, hidden: bool, tick: int, horizon: int
) -> dict[str, Any]:
    event_class = _SOURCE_EVENT_CLASS.get(kind, "telemetry")
    decision_required = bool(
        changed and not hidden and event_class == "safety" and tick + 1 < horizon
    )
    return {
        "event_class": event_class,
        "decision_required": decision_required,
        "actionable": decision_required,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(cast(Any, value)).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _native_source_event_evidence_ready(evidence: dict[str, Any]) -> bool:
    expected = evidence.get("expected_source_event_ids")
    observed = evidence.get("observed_source_event_ids")
    material = evidence.get("material_source_event_ids")
    rows = evidence.get("source_event_materiality")
    if not all(isinstance(value, list) for value in (expected, observed, material, rows)):
        return False
    if (
        not expected
        or not all(isinstance(value, str) and value for value in expected)
        or len(expected) != len(set(expected))
        or observed != expected
        or material != expected
        or len(rows) != len(expected)
    ):
        return False
    rows_by_id = {
        str(row.get("event_id") or ""): row
        for row in rows
        if isinstance(row, dict) and row.get("event_id")
    }
    if len(rows_by_id) != len(rows) or list(rows_by_id) != expected:
        return False
    return all(
        row.get("state_observation_kind") == "native_backend_readback"
        and row.get("materiality_passed") is True
        and bool(row.get("changed_state_fields"))
        and bool(row.get("before_state_digest"))
        and bool(row.get("after_state_digest"))
        and row.get("before_state_digest") != row.get("after_state_digest")
        for row in rows
    )


def formal_sumo_ego_source_evidence_ready(evidence: dict[str, Any]) -> bool:
    """Return whether one native run contains complete formal source evidence."""
    return bool(
        evidence.get("proof_kind") == "direct_runtime_files"
        and evidence.get("runtime_trace_observed") is True
        and evidence.get("evidence_from_scenario_config_only") is False
        and evidence.get("runtime_fidelity") == "native_live_sumo_reactive"
        and evidence.get("deterministic_source_trace") is True
        and evidence.get("state_effect_observed") is True
        and evidence.get("source_state_effect_observed") is True
        and evidence.get("named_events_causally_proven") is True
        and bool(evidence.get("runtime_opened_assets"))
        and bool(evidence.get("consumed_source_hashes"))
        and bool(evidence.get("lineage_source_hashes"))
        and bool(evidence.get("consumed_window_sha256"))
        and bool(evidence.get("recipe_version"))
        and bool(evidence.get("consumed_channels"))
        and bool(evidence.get("derived_backend_state_fields"))
        and bool(evidence.get("source_field_to_state_field_map"))
        and bool(evidence.get("initial_state_digest"))
        and bool(evidence.get("post_source_state_digests"))
        and bool(evidence.get("trace_semantic_digest"))
        and _native_source_event_evidence_ready(evidence)
        and not evidence.get("blockers")
    )


@dataclass
class EgoState:
    vehicle_id: str
    route_position_m: float
    lateral_position_m: float
    lane_index: int
    speed_mps: float
    acceleration_mps2: float = 0.0
    heading_rad: float = 0.0
    length_m: float = 4.8
    width_m: float = 1.9


@dataclass
class ActorState:
    actor_id: str
    route_position_m: float
    lane_index: int
    speed_mps: float
    lateral_position_m: float | None = None
    length_m: float = 4.8
    width_m: float = 1.9


@dataclass
class DrivingEnvelope:
    target_speed_min_mps: float
    target_speed_max_mps: float
    min_time_headway_s: float = 2.0
    max_acceleration_mps2: float = 2.0
    max_deceleration_mps2: float = 4.0
    expires_at_tick: int | None = None


@dataclass
class SumoEgoTickRecord:
    tick: int
    simulation_time_seconds: float
    route_progress: float
    route_delay_seconds: float
    min_ttc_seconds: float | None
    safety_violation_severity: float
    catastrophic_failure: bool
    collision_count: int
    road_departure_count: int
    comfort_jerk_burden: float
    residual_risk_burden: float
    shield_intervention_count: int
    mrm_active: bool
    assurance_mode: str = "nominal"
    realized_events: list[dict[str, Any]] = field(default_factory=list)
    runtime_assurance_records: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuntimeAssuranceUnavailable(RuntimeError):
    """Raised when the mandatory domain safety implementation is absent."""


class _RuntimeAssuranceBridge:
    """Small compatibility adapter around the domain safety module.

    There is deliberately no local fallback safety policy.  If the safety
    implementation is unavailable or incompatible, reset fails closed instead
    of silently running an unshielded vehicle.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        try:
            module = importlib.import_module("domains.autonomous_driving.runtime_assurance")
        except (ImportError, AttributeError) as exc:
            raise RuntimeAssuranceUnavailable(
                "autonomous_driving runtime assurance is mandatory"
            ) from exc
        implementation = getattr(module, "SafetySupervisor", None) or getattr(
            module, "RuntimeAssurance", None
        )
        if implementation is None:
            raise RuntimeAssuranceUnavailable(
                "runtime_assurance module exposes neither SafetySupervisor nor RuntimeAssurance"
            )
        config_type = getattr(module, "RuntimeAssuranceConfig", None)
        runtime_config: Any = dict(config)
        if config_type is not None:
            runtime_config = self._construct(config_type, config)
        try:
            self._impl = implementation(runtime_config)
        except TypeError:
            self._impl = implementation(config=runtime_config)
        self._module = module

    @staticmethod
    def _construct(factory: Any, values: dict[str, Any]) -> Any:
        signature = inspect.signature(factory)
        accepted = {name: value for name, value in values.items() if name in signature.parameters}
        return factory(**accepted)

    @property
    def mode(self) -> str:
        value = getattr(self._impl, "mode", "unknown")
        return str(getattr(value, "value", value))

    def request_mrm(self, reason: str) -> None:
        request = getattr(self._impl, "request_mrm", None)
        if not callable(request):
            raise RuntimeAssuranceUnavailable(
                "runtime assurance does not support explicit MRM requests"
            )
        request(reason=reason)

    def authorize_recovery(self, token: str) -> None:
        authorize = getattr(self._impl, "authorize_recovery", None)
        if not callable(authorize):
            raise RuntimeAssuranceUnavailable("runtime assurance does not support guarded recovery")
        accepted = authorize(token=token)
        if accepted is False:
            raise RuntimeAssuranceUnavailable("runtime assurance rejected recovery authorization")

    def set_recovery_token(self, token: str) -> None:
        """Rotate the configured token without resetting the latched mode."""
        config = getattr(self._impl, "config", None)
        if config is None or not is_dataclass(config):
            raise RuntimeAssuranceUnavailable(
                "runtime assurance does not expose a replaceable recovery config"
            )
        try:
            self._impl.config = replace(cast(Any, config), recovery_token=token)
        except (TypeError, ValueError) as exc:
            raise RuntimeAssuranceUnavailable(
                "runtime assurance rejected recovery token rotation"
            ) from exc

    @property
    def recovery_ready(self) -> bool:
        return bool(getattr(self._impl, "recovery_ready", False))

    def step(
        self,
        *,
        ego: EgoState,
        actors: list[ActorState],
        lane_count: int,
        lane_width_m: float,
        acceleration_mps2: float,
        steering_rad: float,
        sequence: int,
        simulation_time_seconds: float,
    ) -> tuple[float, float, dict[str, Any]]:
        point_type = self._module.Point2D
        vehicle_type = self._module.VehicleState
        road_type = self._module.RoadBoundary
        state_type = self._module.SafetyState
        command_type = self._module.ControlCommand

        ego_vehicle = self._construct(
            vehicle_type,
            {
                "actor_id": ego.vehicle_id,
                "x": ego.route_position_m,
                "y": ego.lateral_position_m,
                "heading_rad": ego.heading_rad,
                "speed_mps": ego.speed_mps,
                "acceleration_mps2": ego.acceleration_mps2,
                "length_m": ego.length_m,
                "width_m": ego.width_m,
            },
        )
        obstacles = [
            self._construct(
                vehicle_type,
                {
                    "actor_id": actor.actor_id,
                    "x": actor.route_position_m,
                    "y": (
                        actor.lateral_position_m
                        if actor.lateral_position_m is not None
                        else actor.lane_index * lane_width_m
                    ),
                    "heading_rad": 0.0,
                    "speed_mps": actor.speed_mps,
                    "acceleration_mps2": 0.0,
                    "length_m": actor.length_m,
                    "width_m": actor.width_m,
                },
            )
            for actor in actors
        ]
        lower = -lane_width_m / 2.0
        upper = (lane_count - 0.5) * lane_width_m
        x_min = ego.route_position_m - 1000.0
        x_max = ego.route_position_m + 1000.0
        road = self._construct(
            road_type,
            {
                "polygon": (
                    self._construct(point_type, {"x": x_min, "y": lower}),
                    self._construct(point_type, {"x": x_max, "y": lower}),
                    self._construct(point_type, {"x": x_max, "y": upper}),
                    self._construct(point_type, {"x": x_min, "y": upper}),
                ),
            },
        )
        safety_state = self._construct(
            state_type,
            {
                "sim_time": simulation_time_seconds,
                "observed_at": simulation_time_seconds,
                "state_version": sequence,
                "ego": ego_vehicle,
                "actors": tuple(obstacles),
                "road_boundary": road,
            },
        )
        command = self._construct(
            command_type,
            {
                "command_id": f"sumo-ego-{sequence}",
                "acceleration_mps2": acceleration_mps2,
                "steering_rad": steering_rad,
                "sequence": sequence,
                "state_version": sequence,
                "issued_at": simulation_time_seconds,
                "valid_from": simulation_time_seconds,
                "valid_until": simulation_time_seconds + 1.0,
                "lateral_escape": False,
                "lateral_verification": None,
            },
        )
        decision = self._impl.step(safety_state, command)
        applied = (
            getattr(decision, "applied_command", None)
            or getattr(decision, "command", None)
            or getattr(decision, "control", None)
        )
        if applied is None:
            raise RuntimeAssuranceUnavailable(
                "runtime assurance decision omitted the applied command"
            )
        record = _jsonable(decision)
        if not isinstance(record, dict):
            record = {"decision": record}
        record.setdefault("mode", self.mode)
        return (
            float(applied.acceleration_mps2),
            float(getattr(applied, "steering_rad", 0.0)),
            record,
        )


class SumoEgoBackend:
    """Fixture-backed deterministic ego-vehicle simulator."""

    backend_kind = "sumo_ego"
    runtime_assurance_schema_version = "1.1"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = dict(config or {})
        self._seed_obj: Any = None
        self._ego: EgoState | None = None
        self._actors: dict[str, ActorState] = {}
        self._source_events: list[dict[str, Any]] = []
        self._applied_source_events: set[str] = set()
        self._source_event_evidence: dict[str, dict[str, Any]] = {}
        self._records: list[SumoEgoTickRecord] = []
        self._tick = 0
        self._horizon = 0
        self._simulation_time_seconds = 0.0
        self._physics_dt_s = 0.1
        self._decision_interval_s = 60.0
        self._substeps = 600
        self._max_review_after_ticks = 2
        self._lane_width_m = 3.6
        self._lane_lateral_origin_m = 0.0
        self._lane_count = 2
        self._route_length_m = 3000.0
        self._speed_limit_mps = 20.0
        self._envelope = DrivingEnvelope(
            target_speed_min_mps=0.0,
            target_speed_max_mps=15.0,
        )
        self._pending_maneuver: dict[str, Any] | None = None
        self._pending_effects: list[dict[str, Any]] = []
        self._bound_effects: dict[str, dict[str, Any]] = {}
        self._effect_sequence = 0
        self._assurance: _RuntimeAssuranceBridge | None = None
        self._collision_ids: set[str] = set()
        self._road_departures = 0
        self._source_paths: list[Path] = []
        self._source_sha256s: dict[str, str] = {}
        self._lineage_source_sha256s: dict[str, str] = {}
        self._source_window_sha256 = ""
        self._source_recipe_version = ""
        self._source_initial_state_digest = ""
        self._source_consumption_ticks: list[int] = []
        self._source_state_digests: list[str] = []
        self._recovery_token = ""  # nosec B105
        self._recovery_token_issued = False
        self._last_supervisory_sequence = 0
        self._recovery_token_expires_tick: int | None = None
        self._recovery_token_state_digest: str | None = None
        self._mrm_cycle = 0
        self._mrm_cycle_active = False
        self._recovery_action_trace: list[str] = []
        self._tactical_action_trace: list[dict[str, Any]] = []
        self._investigation_trace: list[dict[str, Any]] = []
        self._diagnostic_shield_mode = "active"

    def reset(self, seed_obj: Any) -> None:
        self._reset_common(seed_obj)
        fixture = self._load_fixture()
        ego = dict(fixture.get("ego") or {})
        if not ego:
            raise ValueError("sumo_ego fixture requires an ego record")
        self._lane_width_m = float(fixture.get("lane_width_m") or 3.6)
        self._lane_count = max(1, int(fixture.get("lane_count") or 2))
        self._route_length_m = max(1.0, float(fixture.get("route_length_m") or 3000.0))
        self._speed_limit_mps = max(0.1, float(fixture.get("speed_limit_mps") or 20.0))
        self._ego = EgoState(
            vehicle_id=str(ego.get("vehicle_id") or "ego"),
            route_position_m=float(ego.get("route_position_m") or 0.0),
            lateral_position_m=float(
                cast(
                    Any,
                    ego.get("lateral_position_m")
                    if ego.get("lateral_position_m") is not None
                    else int(ego.get("lane_index") or 0) * self._lane_width_m,
                )
            ),
            lane_index=int(ego.get("lane_index") or 0),
            speed_mps=float(ego.get("speed_mps") or 0.0),
            length_m=float(ego.get("length_m") or 4.8),
            width_m=float(ego.get("width_m") or 1.9),
        )
        # NGSIM lane identifiers need not start at zero.  Preserve the
        # source lane index while still allowing the emulated lateral model to
        # update it when a validated maneuver is requested.
        self._lane_lateral_origin_m = (
            self._ego.lateral_position_m - self._ego.lane_index * self._lane_width_m
        )
        self._actors = {
            str(row["actor_id"]): ActorState(
                actor_id=str(row["actor_id"]),
                route_position_m=float(row.get("route_position_m") or 0.0),
                lane_index=int(row.get("lane_index") or 0),
                speed_mps=float(row.get("speed_mps") or 0.0),
                lateral_position_m=(
                    float(row["lateral_position_m"])
                    if row.get("lateral_position_m") is not None
                    else None
                ),
                length_m=float(row.get("length_m") or 4.8),
                width_m=float(row.get("width_m") or 1.9),
            )
            for row in fixture.get("actors") or []
            if isinstance(row, dict) and row.get("actor_id")
        }
        self._source_events = [
            dict(row) for row in fixture.get("source_events") or [] if isinstance(row, dict)
        ]
        initial_maximum = min(
            self._speed_limit_mps,
            float(
                self._config.get("initial_target_speed_max_mps")
                or self._config.get("initial_target_speed_mps")
                or self._speed_limit_mps
            ),
        )
        self._envelope = DrivingEnvelope(
            target_speed_min_mps=max(
                0.0,
                min(
                    initial_maximum,
                    float(self._config.get("initial_target_speed_min_mps") or 0.0),
                ),
            ),
            target_speed_max_mps=initial_maximum,
            min_time_headway_s=float(self._config.get("initial_min_time_headway_s") or 2.0),
        )
        self._initialize_assurance()
        if self._source_paths:
            self._source_initial_state_digest = _digest(self._source_state())

    def _reset_common(self, seed_obj: Any) -> None:
        self._seed_obj = seed_obj
        self._tick = 0
        self._horizon = max(1, int(getattr(seed_obj, "horizon_ticks", 1)))
        self._simulation_time_seconds = 0.0
        self._records.clear()
        self._actors.clear()
        self._applied_source_events.clear()
        self._source_event_evidence.clear()
        self._pending_maneuver = None
        self._pending_effects.clear()
        self._bound_effects.clear()
        self._effect_sequence = 0
        self._collision_ids.clear()
        self._road_departures = 0
        self._source_consumption_ticks.clear()
        self._source_state_digests.clear()
        self._source_paths.clear()
        self._source_sha256s.clear()
        self._lineage_source_sha256s.clear()
        self._source_window_sha256 = ""
        self._source_recipe_version = ""
        self._source_initial_state_digest = ""
        self._mrm_cycle = 0
        self._mrm_cycle_active = False
        self._recovery_action_trace.clear()
        self._tactical_action_trace.clear()
        self._investigation_trace.clear()
        self._recovery_token = self._recovery_token_for_cycle(0)
        self._recovery_token_issued = False
        self._last_supervisory_sequence = 0
        self._recovery_token_expires_tick = None
        self._recovery_token_state_digest = None
        self._diagnostic_shield_mode = str(
            self._config.get("diagnostic_shield_mode") or "active"
        ).lower()
        if self._diagnostic_shield_mode not in {"active", "shadow"}:
            raise ValueError("unsupported sumo_ego diagnostic shield mode")
        if (
            self._diagnostic_shield_mode != "active"
            and self._config.get("unsafe_diagnostic_acknowledged") is not True
        ):
            raise ValueError("non-enforcing shield mode requires unsafe diagnostic acknowledgement")
        self._physics_dt_s = max(0.01, float(self._config.get("physics_step_seconds") or 0.1))
        default_interval = max(0.01, float(getattr(seed_obj, "tick_seconds", 5.0)))
        self._decision_interval_s = max(
            self._physics_dt_s,
            float(self._config.get("decision_interval_seconds") or default_interval),
        )
        if not math.isclose(
            self._decision_interval_s,
            default_interval,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("sumo_ego decision clock does not match tick_seconds")
        self._substeps = max(
            1,
            int(
                self._config.get("physics_substeps_per_tick")
                or round(self._decision_interval_s / self._physics_dt_s)
            ),
        )
        if not math.isclose(
            self._substeps * self._physics_dt_s,
            self._decision_interval_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("sumo_ego physics substeps do not span one decision interval")
        requirements = dict(self._config.get("task_requirements") or {})
        declared_review_interval = requirements.get(
            "required_review_interval_ticks",
            self._config.get("max_review_after_ticks", 2),
        )
        try:
            self._max_review_after_ticks = int(declared_review_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("sumo_ego review interval must be an integer") from exc
        if self._max_review_after_ticks < 1 or self._max_review_after_ticks > 2:
            raise ValueError("sumo_ego review interval must be between one and two ticks")
        self._validate_clock_contract(seed_obj)

    def _validate_clock_contract(self, seed_obj: Any) -> None:
        contract = dict(getattr(seed_obj, "clock_contract", {}) or {})
        if not contract:
            return
        expected_substeps = int(contract.get("substeps_per_supervisory_tick") or 0)
        if contract.get("schema_version") != "driving_clock_v1":
            raise ValueError("sumo_ego clock contract schema mismatch")
        if not math.isclose(
            float(contract.get("physics_step_seconds") or 0.0),
            self._physics_dt_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("sumo_ego physics clock does not match clock_contract")
        if not math.isclose(
            float(contract.get("shield_step_seconds") or 0.0),
            self._physics_dt_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("sumo_ego shield clock does not match physics clock")
        if expected_substeps != self._substeps:
            raise ValueError("sumo_ego substeps do not match clock_contract")
        if contract.get("provider_wall_clock_advances_simulation") is not False:
            raise ValueError("sumo_ego wall clock must not advance simulation")

    def _load_fixture(self) -> dict[str, Any]:
        inline = self._config.get("fixture")
        if isinstance(inline, dict):
            return dict(inline)
        source_bundle = str(self._config.get("source_bundle") or "").strip()
        if source_bundle:
            return self._load_source_bundle(Path(source_bundle).expanduser().resolve())
        source = str(
            self._config.get("generated_fixture_path") or self._config.get("fixture_path") or ""
        ).strip()
        if not source:
            raise ValueError(
                "sumo_ego requires fixture, generated_fixture_path, fixture_path, or source_bundle"
            )
        path = Path(source).expanduser().resolve()
        payload = path.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        expected = str(
            self._config.get("generated_fixture_sha256") or self._config.get("fixture_sha256") or ""
        ).removeprefix("sha256:")
        if not expected:
            raise ValueError("sumo_ego fixture_path requires fixture_sha256")
        if observed != expected:
            raise ValueError("sumo_ego fixture sha256 mismatch")
        self._source_paths = [path]
        self._source_sha256s = {str(path): observed}
        loaded = json.loads(payload)
        if not isinstance(loaded, dict):
            raise ValueError("sumo_ego fixture must deserialize to an object")
        return loaded

    def _load_source_bundle(self, bundle_dir: Path) -> dict[str, Any]:
        from domains.autonomous_driving.data.contracts import file_sha256
        from domains.autonomous_driving.data.ngsim import verify_bundle

        verify_bundle(bundle_dir)
        bundle_path = bundle_dir / "bundle.json"
        bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        database_path = bundle_dir / "normalized/trajectories.sqlite3"
        candidates_path = bundle_dir / "mining/candidates.json"
        report = json.loads(candidates_path.read_text(encoding="utf-8"))
        runtime_fixture = json.loads(
            (bundle_dir / "runtime/fixture.json").read_text(encoding="utf-8")
        )
        derivation = dict(runtime_fixture.get("derivation") or {})
        candidates = [dict(row) for row in report.get("candidates") or []]
        requested_id = str(self._config.get("candidate_id") or "")
        if not requested_id:
            raise ValueError("sumo_ego source bundle requires candidate_id")
        selected = next(
            (row for row in candidates if row.get("candidate_id") == requested_id),
            None,
        )
        if selected is None:
            raise ValueError("sumo_ego source bundle candidate not found")
        hazard_context = dict(selected.get("hazard_context") or {})
        phase_complete = bool(hazard_context.get("phase_window_complete"))
        if requested_id != str(derivation.get("candidate_id") or ""):
            if phase_complete:
                raise ValueError("sumo_ego phase-complete candidate override rejected")
            raise ValueError("sumo_ego runtime fixture candidate identity mismatch")
        if derivation.get("source_window_sha256") != selected.get("source_window_sha256"):
            raise ValueError("sumo_ego runtime fixture source window mismatch")
        locked_actor_ids = tuple(str(value) for value in selected.get("actor_ids") or [])
        requested_actor_ids = tuple(str(value) for value in self._config.get("actor_ids") or [])
        if requested_actor_ids and requested_actor_ids != locked_actor_ids:
            raise ValueError("sumo_ego source bundle actor_ids override rejected")
        actor_ids = locked_actor_ids
        if not actor_ids:
            raise ValueError("sumo_ego source bundle candidate has no actors")
        start_ms = int(selected.get("start_time_ms") or 0)
        end_ms = int(selected.get("end_time_ms_exclusive") or 0)
        for key, locked_value in (
            ("start_time_ms", start_ms),
            ("end_time_ms_exclusive", end_ms),
        ):
            if key in self._config and int(self._config[key]) != locked_value:
                raise ValueError(f"sumo_ego source bundle {key} override rejected")
        query = """
            SELECT actor_id, timestamp_ms, local_x_m, local_y_m, speed_mps,
                   acceleration_mps2, lane_id, length_m, width_m
              FROM states
             WHERE timestamp_ms >= ? AND timestamp_ms < ?
             ORDER BY timestamp_ms, actor_id
        """
        with sqlite3.connect(database_path) as connection:
            rows = [
                row
                for row in connection.execute(query, (start_ms, end_ms))
                if str(row[0]) in actor_ids
            ]
        if not rows:
            raise ValueError("sumo_ego source bundle window has no runtime states")
        lane_ids = sorted({int(row[6]) for row in rows})
        lane_map = {lane_id: index for index, lane_id in enumerate(lane_ids)}
        initial_time = min(int(row[1]) for row in rows)
        initial_rows = {str(row[0]): row for row in rows if int(row[1]) == initial_time}
        explicit_ego_actor_id = str(self._config.get("ego_actor_id") or "")
        if phase_complete:
            hazard_ego_actor_id = str(hazard_context.get("ego_actor_id") or "")
            if self._config.get("source_window_sha256") not in {
                None,
                selected.get("source_window_sha256"),
            }:
                raise ValueError("sumo_ego phase-complete source_window_sha256 override rejected")
            fixture_ego_actor_id = str((runtime_fixture.get("ego") or {}).get("vehicle_id") or "")
            if (
                derivation.get("candidate_hazard_context_bound") is not True
                or str(derivation.get("ego_actor_id") or "") != hazard_ego_actor_id
                or fixture_ego_actor_id != hazard_ego_actor_id
            ):
                raise ValueError("sumo_ego phase-complete hazard ego lock mismatch")
            if explicit_ego_actor_id and explicit_ego_actor_id != hazard_ego_actor_id:
                raise ValueError("sumo_ego phase-complete hazard ego override rejected")
            ego_actor_id = hazard_ego_actor_id
        else:
            diagnostic_ego_actor_id = str(derivation.get("ego_actor_id") or "")
            if explicit_ego_actor_id != diagnostic_ego_actor_id:
                raise ValueError("sumo_ego diagnostic source bundle requires declared fixture ego")
            ego_actor_id = explicit_ego_actor_id
        if ego_actor_id not in initial_rows:
            raise ValueError("sumo_ego source bundle ego missing at window start")
        conflict_actor_id = str(hazard_context.get("conflict_actor_id") or "")
        if phase_complete:
            if str(derivation.get("conflict_actor_id") or "") != conflict_actor_id:
                raise ValueError("sumo_ego phase-complete conflict actor lock mismatch")
            explicit_conflict_actor_id = str(self._config.get("conflict_actor_id") or "")
            if explicit_conflict_actor_id and explicit_conflict_actor_id != conflict_actor_id:
                raise ValueError("sumo_ego phase-complete conflict actor override rejected")
        if phase_complete and conflict_actor_id not in actor_ids:
            raise ValueError("sumo_ego phase-complete conflict actor missing from candidate")
        if phase_complete and conflict_actor_id not in initial_rows:
            raise ValueError("sumo_ego source bundle conflict actor missing at window start")
        fixture_events = [
            dict(event)
            for event in runtime_fixture.get("source_events") or []
            if isinstance(event, dict)
        ]
        if phase_complete and not fixture_events:
            raise ValueError("sumo_ego phase-complete source events missing")
        for event in fixture_events:
            event_actor_id = str(event.get("actor_id") or "")
            source_event_ids = event.get("source_event_ids")
            provenance = dict(event.get("source_provenance") or {})
            if event_actor_id not in actor_ids:
                raise ValueError("sumo_ego source event actor is outside locked candidate")
            if (
                not isinstance(source_event_ids, list)
                or not source_event_ids
                or not all(isinstance(value, str) and value for value in source_event_ids)
            ):
                raise ValueError("sumo_ego source event ids are missing")
            if provenance.get("candidate_id") != requested_id or provenance.get(
                "source_window_sha256"
            ) != selected.get("source_window_sha256"):
                raise ValueError("sumo_ego source event provenance lock mismatch")

        def actor_payload(row: tuple[Any, ...], *, vehicle_id: str) -> dict[str, Any]:
            return {
                "vehicle_id" if vehicle_id == ego_actor_id else "actor_id": vehicle_id,
                "route_position_m": float(row[3]),
                "lateral_position_m": float(row[2]),
                "lane_index": lane_map[int(row[6])],
                "speed_mps": max(0.0, float(row[4])),
                "length_m": float(row[7]),
                "width_m": float(row[8]),
            }

        ego_row = initial_rows[ego_actor_id]
        reactive_ids = {
            str(value) for value in derivation.get("reactive_actor_ids") or [] if str(value)
        }
        if phase_complete:
            reactive_ids.update({ego_actor_id, conflict_actor_id})
        candidate_rows = [
            (actor_id, row)
            for actor_id, row in sorted(initial_rows.items())
            if actor_id != ego_actor_id and (not reactive_ids or actor_id in reactive_ids)
        ]
        # A naturalistic log can contain two vehicles whose projected source
        # boxes overlap at the selected timestamp even though the log is not
        # a collision record.  Keep the locked ego/conflict pair and add only
        # non-overlapping context actors, so the deterministic emulator does
        # not create an artificial collision before the first source event.
        required_ids = {conflict_actor_id} if phase_complete else set()
        kept: list[tuple[str, tuple[Any, ...]]] = []
        ego_geometry = actor_payload(ego_row, vehicle_id=ego_actor_id)
        for actor_id, row in candidate_rows:
            if actor_id in required_ids:
                kept.append((actor_id, row))
                continue
            lane = lane_map[int(row[6])]
            position = float(row[3])
            length = float(row[7])

            occupied = [
                (
                    int(ego_geometry.get("lane_index") or 0),
                    float(ego_geometry.get("route_position_m") or 0.0),
                    float(ego_geometry.get("length_m") or 4.8),
                ),
                *[(lane_map[int(other[6])], float(other[3]), float(other[7])) for _, other in kept],
            ]
            if any(
                lane == other_lane
                and abs(position - other_position) < 0.5 * (length + other_length)
                for other_lane, other_position, other_length in occupied
            ):
                continue
            kept.append((actor_id, row))
        actors = [actor_payload(row, vehicle_id=actor_id) for actor_id, row in kept]
        source_files = [
            database_path,
            candidates_path,
            bundle_dir / "runtime/fixture.json",
            bundle_path,
        ]
        self._source_paths = source_files
        self._source_sha256s = {str(path): file_sha256(path) for path in source_files}
        raw_contract = dict(bundle_payload.get("source_contract") or {})
        lineage_paths = [
            bundle_dir / str(path) for path in raw_contract.get("derivation_input") or []
        ]
        self._lineage_source_sha256s = {
            str(path): file_sha256(path) for path in lineage_paths
        }
        self._source_window_sha256 = str(derivation.get("source_window_sha256") or "")
        self._source_recipe_version = "ngsim_phase_complete_window_v1"
        return {
            "lane_count": max(1, len(lane_ids)),
            "lane_width_m": float(self._config.get("lane_width_m") or 3.6),
            "route_length_m": max(
                1.0,
                float(self._config.get("route_length_m") or 3000.0),
            ),
            "speed_limit_mps": max(
                0.1,
                float(self._config.get("speed_limit_mps") or 30.0),
            ),
            "ego": actor_payload(ego_row, vehicle_id=ego_actor_id),
            "actors": actors,
            "source_events": fixture_events,
        }

    def _initialize_assurance(self) -> None:
        assurance_config = dict(self._config.get("runtime_assurance") or {})
        declared_step = float(assurance_config.get("step_s") or self._physics_dt_s)
        if not math.isclose(
            declared_step,
            self._physics_dt_s,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("sumo_ego runtime assurance step must match physics step")
        assurance_config.setdefault("step_s", self._physics_dt_s)
        assurance_config.setdefault("physics_step_seconds", self._physics_dt_s)
        assurance_config.setdefault("min_time_headway_s", self._envelope.min_time_headway_s)
        assurance_config.setdefault("recovery_token", self._recovery_token)
        self._assurance = _RuntimeAssuranceBridge(assurance_config)

    def inspect_ego_state(self) -> dict[str, Any]:
        return _jsonable(self._require_ego())

    def record_investigation(self, tool_name: str) -> None:
        """Record paid supervisory inspection without exposing engine truth."""
        self._investigation_trace.append({"tool_name": str(tool_name), "tick": self._tick})

    def inspect_local_scene(self) -> dict[str, Any]:
        return {
            "ego": self.inspect_ego_state(),
            "actors": [_jsonable(actor) for actor in self._actors.values()],
            "lane_count": self._lane_count,
            "lane_width_m": self._lane_width_m,
            "speed_limit_mps": self._speed_limit_mps,
        }

    def inspect_odd_status(self) -> dict[str, Any]:
        return {
            "status": "inside_odd",
            "road_geometry_available": True,
            "runtime_assurance_available": self._assurance is not None,
        }

    def inspect_safety_state(self) -> dict[str, Any]:
        shield_enforcing = self._diagnostic_shield_mode == "active"
        return {
            "estimate_kind": "current_observation_derived",
            "future_trajectory_included": False,
            "mode": self.assurance_mode if shield_enforcing else "nominal",
            "shadow_assurance_mode": (None if shield_enforcing else self.assurance_mode),
            "shield_active": shield_enforcing,
            "shield_mode": self._diagnostic_shield_mode,
            "min_ttc_seconds": self._minimum_ttc(),
            "collision_count": len(self._collision_ids),
            "road_departure_count": self._road_departures,
            "recovery_ready": bool(
                self._assurance and self._assurance.recovery_ready
            ),
            "low_level_control_owner": (
                "backend_runtime_assurance"
                if shield_enforcing
                else "nominal_controller_unshielded_diagnostic"
            ),
        }

    def set_driving_envelope(self, args: dict[str, Any]) -> dict[str, Any]:
        before = _jsonable(self._envelope)
        legacy_target = args.get("target_speed_mps")
        target_minimum = float(args.get("target_speed_min_mps", 0.0))
        target_maximum = float(
            args.get(
                "target_speed_max_mps",
                legacy_target if legacy_target is not None else math.nan,
            )
        )
        headway = float(args.get("min_time_headway_s", self._envelope.min_time_headway_s))
        max_acceleration = float(
            args.get("max_acceleration_mps2", self._envelope.max_acceleration_mps2)
        )
        max_deceleration = float(
            args.get("max_deceleration_mps2", self._envelope.max_deceleration_mps2)
        )
        if not all(
            math.isfinite(value)
            for value in (
                target_minimum,
                target_maximum,
                headway,
                max_acceleration,
                max_deceleration,
            )
        ):
            return {"_status": "error", "error": "non_finite_driving_envelope"}
        if target_minimum < 0.0 or target_minimum > target_maximum:
            return {"_status": "error", "error": "invalid_target_speed_range"}
        if target_maximum > self._speed_limit_mps:
            return {"_status": "error", "error": "target_speed_exceeds_runtime_limit"}
        validation = self._validate_supervisory_command(args)
        if validation is not None:
            return validation
        self._envelope = DrivingEnvelope(
            target_speed_min_mps=target_minimum,
            target_speed_max_mps=target_maximum,
            min_time_headway_s=max(0.5, headway),
            max_acceleration_mps2=max(0.1, max_acceleration),
            max_deceleration_mps2=max(0.1, max_deceleration),
            expires_at_tick=int(args["expires_at_tick"]),
        )
        return self._queue_effect("set_driving_envelope", args, before, _jsonable(self._envelope))

    def request_tactical_maneuver(self, args: dict[str, Any]) -> dict[str, Any]:
        maneuver = str(args.get("maneuver") or "")
        allowed = {"keep_lane", "change_lane_left", "change_lane_right", "slow_for_hazard"}
        if maneuver not in allowed:
            return {"_status": "error", "error": "unknown_tactical_maneuver"}
        validation = self._validate_supervisory_command(args)
        if validation is not None:
            return validation
        before = dict(self._pending_maneuver or {})
        self._pending_maneuver = {
            "maneuver": maneuver,
            "target_lane": args.get("target_lane"),
            "requested_at_tick": self._tick,
            "expires_at_tick": int(args["expires_at_tick"]),
        }
        return self._queue_effect("request_tactical_maneuver", args, before, self._pending_maneuver)

    def request_minimal_risk_maneuver(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._assurance is None:
            return {"_status": "error", "error": "runtime_assurance_unavailable"}
        reason = str(args.get("reason") or "operator_request")
        try:
            self._begin_mrm_cycle()
            self._assurance.request_mrm(reason)
        except RuntimeAssuranceUnavailable as exc:
            return {"_status": "error", "error": str(exc)}
        self._recovery_action_trace.append("request_minimal_risk_maneuver")
        return self._queue_effect(
            "request_minimal_risk_maneuver",
            args,
            {"mode": self.assurance_mode},
            {"mode": self.assurance_mode, "request_accepted": True},
        )

    def request_recovery_check(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = self.assurance_mode.lower()
        if mode != "minimal_risk_condition":
            return {"_status": "error", "error": "recovery_check_not_ready"}
        if self._assurance is None or not self._assurance.recovery_ready:
            return {"_status": "error", "error": "recovery_health_dwell_incomplete"}
        self._recovery_token_issued = True
        self._recovery_token_expires_tick = self._tick + 1
        self._recovery_token_state_digest = self._recovery_state_digest()
        self._recovery_action_trace.append("request_recovery_check")
        return self._queue_effect(
            "request_recovery_check",
            args,
            {"recovery_token_issued": False, "mode": mode},
            {"recovery_token_issued": True, "mode": mode},
            response_extra={
                "recovery_token": self._recovery_token,
                "health_dwell_required": True,
                "expires_at_tick": self._recovery_token_expires_tick,
            },
        )

    def authorize_recovery(self, args: dict[str, Any]) -> dict[str, Any]:
        if self._assurance is None:
            return {"_status": "error", "error": "runtime_assurance_unavailable"}
        token = str(args.get("recovery_token") or "")
        if (
            not self._recovery_token_issued
            or token != self._recovery_token
            or self._recovery_token_expires_tick is None
            or self._tick > self._recovery_token_expires_tick
            or self._recovery_token_state_digest != self._recovery_state_digest()
        ):
            return {"_status": "error", "error": "recovery_token_not_issued"}
        try:
            self._assurance.authorize_recovery(token)
        except RuntimeAssuranceUnavailable as exc:
            return {"_status": "error", "error": str(exc)}
        self._recovery_token_issued = False
        self._recovery_token_expires_tick = None
        self._recovery_token_state_digest = None
        self._recovery_action_trace.append("authorize_recovery")
        return self._queue_effect(
            "authorize_recovery",
            {"recovery_token_present": bool(token)},
            {"mode": self.assurance_mode},
            {"mode": self.assurance_mode, "authorization_queued": True},
        )

    def _queue_effect(
        self,
        tool_name: str,
        requested: dict[str, Any],
        before: Any,
        after: Any,
        response_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        token = f"sumo-ego-effect-{self._effect_sequence}"
        self._effect_sequence += 1
        row = {
            "effect_token": token,
            "tool_name": tool_name,
            "requested_action": dict(requested),
            "before_state_digest": _digest(before),
            "after_state_digest": _digest(after),
            "changed_state_fields": [tool_name],
        }
        self._pending_effects.append(row)
        self._tactical_action_trace.append(
            {
                "tool_name": tool_name,
                "tick": self._tick,
                "effect_token": token,
                "before_state_digest": row["before_state_digest"],
                "after_state_digest": row["after_state_digest"],
            }
        )
        return {
            "_status": "accepted",
            "effect_token": token,
            "requested_action": dict(requested),
            "applied_tactical_state": after,
            "low_level_control_owner": "backend_runtime_assurance",
            **dict(response_extra or {}),
        }

    def _recovery_state_digest(self) -> str:
        return _digest(
            {
                "ego": _jsonable(self._require_ego()),
                "actors": [
                    _jsonable(actor)
                    for actor in sorted(self._actors.values(), key=lambda value: value.actor_id)
                ],
                "mode": self.assurance_mode,
                "collision_ids": sorted(self._collision_ids),
                "road_departures": self._road_departures,
                "mrm_cycle": self._mrm_cycle,
            }
        )

    def bind_tool_result(
        self,
        *,
        call_id: str | None,
        evidence_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        token = str(payload.get("effect_token") or "")
        if token:
            self._bound_effects[token] = {
                "call_id": call_id,
                "evidence_id": evidence_id,
            }

    def tick(self, current_tick: int) -> SumoEgoTickRecord:
        ego = self._require_ego()
        if self._assurance is None:
            raise RuntimeAssuranceUnavailable("runtime assurance not initialized")
        self._tick = int(current_tick)
        self._expire_supervisory_commands()
        source_events: list[dict[str, Any]] = []
        effect_events = self._materialize_effect_events()
        min_ttc: float | None = None
        comfort_burden = 0.0
        assurance_rows: list[dict[str, Any]] = []
        interventions = 0
        prior_acceleration = ego.acceleration_mps2
        for substep in range(self._substeps):
            source_events.extend(self._apply_source_events(substep=substep))
            state_digest_before = _digest(self._source_state())
            requested_acceleration, requested_steering = self._nominal_command()
            shield_acceleration, shield_steering, assurance = self._assurance.step(
                ego=ego,
                actors=list(self._actors.values()),
                lane_count=self._lane_count,
                lane_width_m=self._lane_width_m,
                acceleration_mps2=requested_acceleration,
                steering_rad=requested_steering,
                sequence=self._tick * self._substeps + substep,
                simulation_time_seconds=self._simulation_time_seconds,
            )
            original_intervention = str(
                assurance.get("intervention_kind")
                or assurance.get("intervention")
                or assurance.get("kind")
                or "pass"
            ).lower()
            would_intervene = original_intervention not in {"pass", "none"}
            enforcement_applied = self._diagnostic_shield_mode == "active"
            if enforcement_applied:
                applied_acceleration = shield_acceleration
                applied_steering = shield_steering
            else:
                applied_acceleration = requested_acceleration
                applied_steering = requested_steering
            assurance = {
                **assurance,
                "diagnostic_shield_mode": self._diagnostic_shield_mode,
                "enforcement_applied": enforcement_applied,
                "would_intervene": would_intervene,
                "original_intervention": original_intervention,
            }
            mode = str(assurance.get("mode") or self.assurance_mode).lower()
            if enforcement_applied and would_intervene:
                interventions += 1
            assurance_record: dict[str, Any] | None = None
            if would_intervene or mode not in {"nominal", "unknown"}:
                assurance_record = {
                    **assurance,
                    "controller_id": "runtime_assurance",
                    "controller_version": self.runtime_assurance_schema_version,
                    "source_candidate_id": self._config.get("candidate_id"),
                    "intervention_id": f"ra:{self._tick}:{substep}",
                    "outer_tick": self._tick,
                    "physics_substep": substep,
                    "simulation_time_seconds": self._simulation_time_seconds,
                    "ego_id": ego.vehicle_id,
                    "hazard_actor_ids": [
                        actor.actor_id
                        for actor in sorted(self._actors.values(), key=lambda item: item.actor_id)
                        if abs(actor.route_position_m - ego.route_position_m) <= 100.0
                    ][:64],
                    "lane_index": ego.lane_index,
                    "route_position_m": ego.route_position_m,
                    "requested_acceleration_mps2": requested_acceleration,
                    "applied_acceleration_mps2": applied_acceleration,
                    "requested_steering_rad": requested_steering,
                    "applied_steering_rad": applied_steering,
                    "candidate_trajectory_digest": _digest(assurance.get("evidence") or assurance),
                    "state_digest_before": state_digest_before,
                }
                assurance_rows.append(assurance_record)
            comfort_burden += abs(applied_acceleration - prior_acceleration)
            prior_acceleration = applied_acceleration
            self._advance_physics(applied_acceleration, applied_steering)
            self._simulation_time_seconds += self._physics_dt_s
            if assurance_record is not None:
                assurance_record["state_digest_after"] = _digest(self._source_state())
                assurance_record["effect_simulation_time_seconds"] = self._simulation_time_seconds
            current_ttc = self._minimum_ttc()
            if current_ttc is not None:
                min_ttc = current_ttc if min_ttc is None else min(min_ttc, current_ttc)
            self._detect_safety_failures()
            if self._collision_ids:
                break
        route_progress = min(1.0, ego.route_position_m / self._route_length_m)
        ttc_threshold = max(0.1, self._envelope.min_time_headway_s)
        ttc_deficit = (
            max(0.0, ttc_threshold - min_ttc) / ttc_threshold if min_ttc is not None else 0.0
        )
        severity = 1.0 if self._collision_ids or self._road_departures else ttc_deficit
        route_delay = (
            max(0.0, self._speed_limit_mps - ego.speed_mps)
            * self._decision_interval_s
            / max(self._speed_limit_mps, 0.1)
        )
        catastrophic = bool(self._collision_ids or self._road_departures)
        done = catastrophic or route_progress >= 1.0 or self._tick + 1 >= self._horizon
        physical_assurance_mode = (
            self.assurance_mode.lower() if self._diagnostic_shield_mode == "active" else "nominal"
        )
        record = SumoEgoTickRecord(
            tick=self._tick,
            simulation_time_seconds=round(self._simulation_time_seconds, 6),
            route_progress=round(route_progress, 6),
            route_delay_seconds=round(route_delay, 6),
            min_ttc_seconds=round(min_ttc, 6) if min_ttc is not None else None,
            safety_violation_severity=round(max(0.0, min(1.0, severity)), 6),
            catastrophic_failure=catastrophic,
            collision_count=len(self._collision_ids),
            road_departure_count=self._road_departures,
            comfort_jerk_burden=round(comfort_burden, 6),
            residual_risk_burden=round(ttc_deficit, 6),
            shield_intervention_count=interventions,
            mrm_active=physical_assurance_mode
            in {
                "mrm_active",
                "minimal_risk_condition",
                "recovery_pending",
            },
            assurance_mode=physical_assurance_mode,
            realized_events=[*source_events, *effect_events],
            runtime_assurance_records=assurance_rows,
            done=done,
        )
        self._records.append(record)
        mode_after_tick = physical_assurance_mode
        if mode_after_tick in {
            "mrm_active",
            "minimal_risk_condition",
            "recovery_pending",
        }:
            if not self._mrm_cycle_active:
                self._begin_mrm_cycle()
        elif mode_after_tick == "nominal" and self._mrm_cycle_active:
            self._mrm_cycle_active = False
            self._invalidate_recovery_token_issue()
        if self._source_paths:
            self._source_consumption_ticks.append(self._tick)
            self._source_state_digests.append(_digest(self._source_state()))
        return record

    def _begin_mrm_cycle(self) -> None:
        self._invalidate_recovery_token_issue()
        if self._mrm_cycle_active:
            return
        if self._assurance is None:
            raise RuntimeAssuranceUnavailable("runtime assurance not initialized")
        self._mrm_cycle += 1
        self._mrm_cycle_active = True
        self._recovery_token = self._recovery_token_for_cycle(self._mrm_cycle)
        self._assurance.set_recovery_token(self._recovery_token)

    def _invalidate_recovery_token_issue(self) -> None:
        self._recovery_token_issued = False
        self._recovery_token_expires_tick = None
        self._recovery_token_state_digest = None

    def _recovery_token_for_cycle(self, cycle: int) -> str:
        return hashlib.sha256(
            (
                f"{getattr(self._seed_obj, 'seed_id', '')}|"
                f"{getattr(self._seed_obj, 'seed', 0)}|recovery|{cycle}"
            ).encode()
        ).hexdigest()[:24]

    def _apply_source_events(self, *, substep: int) -> list[dict[str, Any]]:
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
            kind = str(raw.get("kind") or "actor_state_change")
            actor_id = str(raw.get("actor_id") or "")
            actor: ActorState | EgoState | None = self._actors.get(actor_id)
            if actor is None and self._ego is not None and actor_id == self._ego.vehicle_id:
                actor = self._ego
            before = _jsonable(actor) if actor is not None else {}
            if kind == "cut_in" and actor is not None:
                actor.lane_index = self._require_ego().lane_index
                actor.lateral_position_m = actor.lane_index * self._lane_width_m
                actor.route_position_m = self._require_ego().route_position_m + float(
                    raw.get("gap_m") or 8.0
                )
            elif kind == "lead_vehicle_braking" and actor is not None:
                acceleration = float(raw.get("acceleration_mps2") or 0.0)
                duration = max(0.1, float(raw.get("control_duration_s") or 0.1))
                actor.speed_mps = max(0.0, actor.speed_mps + acceleration * duration)
            elif kind == "stopped_vehicle" and actor is not None:
                actor.speed_mps = 0.0
            elif (
                kind
                in {
                    "actor_state_update",
                    "lane_change_conflict",
                    "cut_in_gap_boundary",
                    "short_time_headway_boundary",
                }
                and actor is not None
            ):
                actor.route_position_m = float(raw.get("route_position_m", actor.route_position_m))
                actor.lateral_position_m = float(
                    raw.get(
                        "lateral_position_m",
                        actor.lateral_position_m
                        if actor.lateral_position_m is not None
                        else actor.lane_index * self._lane_width_m,
                    )
                )
                actor.lane_index = int(raw.get("lane_index", actor.lane_index))
                actor.speed_mps = max(0.0, float(raw.get("speed_mps", actor.speed_mps)))
            after = _jsonable(actor) if actor is not None else {}
            changed = ["road_actor_state"] if before != after else []
            hidden = bool(raw.get("hidden", False))
            decision_fields = _source_event_decision_fields(
                kind=kind,
                changed=bool(changed),
                hidden=hidden,
                tick=self._tick,
                horizon=self._horizon,
            )
            event = {
                "event_id": event_id,
                "type": kind,
                "origin": "source_schedule",
                "tick": self._tick,
                "physics_substep": substep,
                "simulation_time_seconds": round(self._simulation_time_seconds, 6),
                "hidden": hidden,
                **decision_fields,
                "changed_state_fields": changed,
                "materiality_metric": "source_actor_state_changed",
                "materiality_value": 1 if changed else 0,
                "materiality_threshold": 1,
                "materiality_passed": bool(changed),
                "before_state_digest": _digest(before),
                "after_state_digest": _digest(after),
                "state_observation_kind": "deterministic_emulator_state",
                "source_event_ids": list(raw.get("source_event_ids") or [event_id]),
                "actor_id": actor_id,
            }
            realized.append(event)
            self._source_event_evidence[event_id] = event
        return realized

    def _materialize_effect_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for row in self._pending_effects:
            token = str(row["effect_token"])
            binding = self._bound_effects.get(token, {})
            call_id = binding.get("call_id")
            evidence_id = binding.get("evidence_id")
            events.append(
                {
                    "event_id": f"{token}:tick-{self._tick}",
                    "type": f"{row['tool_name']}_applied",
                    "origin": "agent_caused",
                    "agent_caused": True,
                    "tick": self._tick,
                    "decision_required": False,
                    "changed_state_fields": list(row["changed_state_fields"]),
                    "materiality_metric": "tactical_state_digest_changed",
                    "materiality_value": int(
                        row["before_state_digest"] != row["after_state_digest"]
                    ),
                    "materiality_threshold": 1,
                    "materiality_passed": row["before_state_digest"] != row["after_state_digest"],
                    "call_id": call_id,
                    "tool_name": row["tool_name"],
                    "requested_action": row["requested_action"],
                    "applied_action": {"effect_token": token},
                    "before_state_digest": row["before_state_digest"],
                    "after_state_digest": row["after_state_digest"],
                    "outcome_tick": self._tick,
                    "evidence_ids": [evidence_id] if evidence_id else [],
                    "action_to_outcome_edge": {
                        "source_call_id": call_id,
                        "outcome_event_id": f"{token}:tick-{self._tick}",
                    },
                }
            )
        self._pending_effects.clear()
        return events

    def _nominal_command(self) -> tuple[float, float]:
        ego = self._require_ego()
        target_speed = self._envelope.target_speed_max_mps
        steering = 0.0
        request = self._pending_maneuver or {}
        maneuver = request.get("maneuver")
        if maneuver == "slow_for_hazard":
            target_speed = min(target_speed, self._speed_limit_mps * 0.35)
        elif maneuver in {"change_lane_left", "change_lane_right"}:
            direction = 1 if maneuver == "change_lane_left" else -1
            target_lane = request.get("target_lane")
            if target_lane is None:
                target_lane = ego.lane_index + direction
            target_lane = max(0, min(self._lane_count - 1, int(target_lane)))
            target_y = target_lane * self._lane_width_m
            steering = max(-0.25, min(0.25, (target_y - ego.lateral_position_m) * 0.1))
            if abs(target_y - ego.lateral_position_m) < 0.1:
                ego.lane_index = target_lane
                self._pending_maneuver = None
        acceleration = (target_speed - ego.speed_mps) / self._physics_dt_s
        return (
            max(
                -self._envelope.max_deceleration_mps2,
                min(self._envelope.max_acceleration_mps2, acceleration),
            ),
            steering,
        )

    def _validate_supervisory_command(self, args: dict[str, Any]) -> dict[str, Any] | None:
        sequence = args.get("command_sequence")
        expires_at = args.get("expires_at_tick")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            return {"_status": "error", "error": "command_sequence_required"}
        if sequence <= self._last_supervisory_sequence:
            return {"_status": "error", "error": "command_sequence_out_of_order"}
        if not isinstance(expires_at, int) or isinstance(expires_at, bool):
            return {"_status": "error", "error": "expires_at_tick_required"}
        if expires_at < self._tick:
            return {"_status": "error", "error": "supervisory_command_expired"}
        self._last_supervisory_sequence = sequence
        return None

    def _expire_supervisory_commands(self) -> None:
        if self._assurance is None:
            return
        if (
            self._envelope.expires_at_tick is not None
            and self._tick > self._envelope.expires_at_tick
        ):
            self._assurance.request_mrm("driving_envelope_expired")
            self._envelope.expires_at_tick = None
        if self._pending_maneuver is not None and self._tick > int(
            self._pending_maneuver["expires_at_tick"]
        ):
            self._assurance.request_mrm("tactical_maneuver_expired")
            self._pending_maneuver = None

    def _advance_physics(self, acceleration_mps2: float, steering_rad: float) -> None:
        ego = self._require_ego()
        ego.acceleration_mps2 = acceleration_mps2
        ego.speed_mps = max(
            0.0,
            min(
                self._speed_limit_mps,
                ego.speed_mps + acceleration_mps2 * self._physics_dt_s,
            ),
        )
        ego.route_position_m += ego.speed_mps * self._physics_dt_s
        ego.lateral_position_m += ego.speed_mps * math.tan(steering_rad) * self._physics_dt_s
        if abs(steering_rad) > 1e-9:
            ego.lane_index = max(
                0,
                min(
                    self._lane_count - 1,
                    int(
                        round(
                            (ego.lateral_position_m - self._lane_lateral_origin_m)
                            / self._lane_width_m
                        )
                    ),
                ),
            )
        for actor in self._actors.values():
            actor.route_position_m += actor.speed_mps * self._physics_dt_s

    def _minimum_ttc(self) -> float | None:
        ego = self._require_ego()
        values: list[float] = []
        for actor in self._actors.values():
            if actor.lane_index != ego.lane_index:
                continue
            gap = (
                actor.route_position_m
                - ego.route_position_m
                - (actor.length_m + ego.length_m) / 2.0
            )
            closing = ego.speed_mps - actor.speed_mps
            if gap > 0 and closing > 1e-9:
                values.append(gap / closing)
        return min(values) if values else None

    def _detect_safety_failures(self) -> None:
        ego = self._require_ego()
        for actor in self._actors.values():
            longitudinal_overlap = (
                abs(actor.route_position_m - ego.route_position_m)
                <= (actor.length_m + ego.length_m) / 2.0
            )
            lateral_overlap = actor.lane_index == ego.lane_index
            if longitudinal_overlap and lateral_overlap:
                self._collision_ids.add(actor.actor_id)
        lower = -self._lane_width_m / 2.0
        upper = (self._lane_count - 0.5) * self._lane_width_m
        if not lower <= ego.lateral_position_m <= upper:
            self._road_departures = 1

    @property
    def assurance_mode(self) -> str:
        return self._assurance.mode if self._assurance is not None else "unavailable"

    def snapshot(self) -> dict[str, Any]:
        ego = self._require_ego()
        actors = {
            actor.actor_id: {
                "kind": "road_actor",
                "actor_id": actor.actor_id,
                "relative_distance_m": actor.route_position_m - ego.route_position_m,
                "relative_speed_mps": actor.speed_mps - ego.speed_mps,
                "lane_index": actor.lane_index,
            }
            for actor in self._actors.values()
        }
        return {
            "domain": "autonomous_driving",
            "backend": self.backend_kind,
            "clock_semantics": "simulator_owned_with_physics_substeps",
            "tick": self._tick,
            "horizon": self._horizon,
            "simulation_time_seconds": round(self._simulation_time_seconds, 6),
            "physics_step_seconds": self._physics_dt_s,
            "physics_substeps_per_tick": self._substeps,
            "decision_opportunity": True,
            "decision_cadence": {
                "mode": "hybrid",
                "periodic_scan_every_ticks": 1,
                "max_review_after_ticks": self._max_review_after_ticks,
                "hold_while_actions_pending": False,
            },
            "ego": _jsonable(ego),
            "entities": actors,
            "driving_envelope": _jsonable(self._envelope),
            "pending_maneuver": dict(self._pending_maneuver or {}),
            "safety_state": self.inspect_safety_state(),
            "route": {
                "length_m": self._route_length_m,
                "progress": min(1.0, ego.route_position_m / self._route_length_m),
                "lane_count": self._lane_count,
                "lane_width_m": self._lane_width_m,
                "speed_limit_mps": self._speed_limit_mps,
            },
        }

    def ground_truth(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        snapshot["entities"] = {
            actor.actor_id: {"kind": "road_actor", **_jsonable(actor)}
            for actor in self._actors.values()
        }
        snapshot["cost_components"] = self.ground_truth_costs()
        snapshot["road_user_harm"] = {actor_id: 1.0 for actor_id in sorted(self._collision_ids)}
        snapshot["runtime_assurance"] = self.runtime_assurance_summary()
        snapshot["investigation_trace"] = list(self._investigation_trace)
        snapshot["tactical_action_trace"] = list(self._tactical_action_trace)
        snapshot["_task_tick_records"] = self.scoring_records()
        return snapshot

    def ground_truth_costs(self) -> dict[str, float]:
        return {
            "collision_cost": 10000.0
            * max((row.collision_count for row in self._records), default=0),
            "road_departure_cost": 10000.0
            * max((row.road_departure_count for row in self._records), default=0),
            "risk_exposure_cost": sum(row.residual_risk_burden * 20.0 for row in self._records),
            "route_delay_cost": sum(row.route_delay_seconds for row in self._records),
            "comfort_cost": sum(row.comfort_jerk_burden * 0.01 for row in self._records),
            "mrm_cost": sum(5.0 for row in self._records if row.mrm_active),
            "mrm_failure_cost": 10000.0
            * int(
                bool(self._records)
                and self._records[-1].catastrophic_failure
                and any(row.mrm_active for row in self._records)
            ),
            "shield_intervention_cost": sum(
                row.shield_intervention_count * 0.01 for row in self._records
            ),
        }

    def scoring_records(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records]

    def runtime_assurance_summary(self) -> dict[str, Any]:
        shield_enforcing = self._diagnostic_shield_mode == "active"
        mode_trace = [
            {"tick": record.tick, "mode": record.assurance_mode} for record in self._records
        ]
        mrm_ticks = [record.tick for record in self._records if record.mrm_active]
        mrc_ticks = [
            record.tick
            for record in self._records
            if record.assurance_mode == "minimal_risk_condition"
        ]
        recovery_ticks = [
            record.tick for record in self._records if record.assurance_mode == "recovery_pending"
        ]
        nominal_after_mrm = bool(
            mrm_ticks
            and any(
                record.tick > max(mrm_ticks) and record.assurance_mode == "nominal"
                for record in self._records
            )
        )
        return {
            "schema_version": self.runtime_assurance_schema_version,
            "shield_mode": self._diagnostic_shield_mode,
            "shield_enforcing": shield_enforcing,
            "shield_active": shield_enforcing,
            "diagnostic_only": not shield_enforcing,
            "low_level_control_owner": (
                "backend_runtime_assurance"
                if shield_enforcing
                else "nominal_controller_unshielded_diagnostic"
            ),
            "mode": self.assurance_mode if shield_enforcing else "nominal",
            "shadow_assurance_mode": None if shield_enforcing else self.assurance_mode,
            "physics_step_seconds": self._physics_dt_s,
            "physics_substeps_per_tick": self._substeps,
            "intervention_count": sum(record.shield_intervention_count for record in self._records),
            "mrm_ticks": mrm_ticks,
            "mrc_ticks": mrc_ticks,
            "recovery_pending_ticks": recovery_ticks,
            "mode_trace": mode_trace,
            "nominal_after_mrm": nominal_after_mrm,
            "recovery_completed": nominal_after_mrm,
            "recovery_ready": bool(self._assurance and self._assurance.recovery_ready),
            "recovery_health_steps": int(
                getattr(getattr(self._assurance, "_impl", None), "_healthy_recovery_steps", 0)
            ),
            "recovery_healthy_steps_required": int(
                getattr(
                    getattr(getattr(self._assurance, "_impl", None), "config", None),
                    "recovery_healthy_steps",
                    0,
                )
            ),
            "recovery_action_trace": list(self._recovery_action_trace),
        }

    def source_consumption_evidence(self) -> dict[str, Any]:
        if not self._source_paths:
            return {
                "status": "held",
                "proof_kind": "inline_fixture_diagnostic",
                "blockers": ["source_locked_fixture_path_missing"],
                "runtime_trace_observed": bool(self._records),
                "evidence_from_scenario_config_only": True,
            }
        state_effect = bool(
            self._source_initial_state_digest
            and any(
                digest != self._source_initial_state_digest
                for digest in self._source_state_digests
            )
        )
        expected_events = [
            str(event.get("event_id") or f"source-event-{index}")
            for index, event in enumerate(self._source_events)
        ]
        observed_events = list(self._source_event_evidence)
        source_event_materiality = [
            dict(self._source_event_evidence[event_id]) for event_id in observed_events
        ]
        material_events = [
            str(event["event_id"])
            for event in source_event_materiality
            if event.get("materiality_passed") is True
            and bool(event.get("changed_state_fields"))
            and bool(event.get("before_state_digest"))
            and bool(event.get("after_state_digest"))
            and event.get("before_state_digest") != event.get("after_state_digest")
        ]
        event_coverage_complete = bool(expected_events) and (
            len(expected_events) == len(set(expected_events))
            and observed_events == expected_events
            and material_events == expected_events
        )
        return {
            "status": "held",
            "proof_kind": "direct_runtime_files",
            "opened_source_paths": [str(path) for path in self._source_paths],
            "opened_source_sha256": dict(self._source_sha256s),
            "runtime_opened_assets": [str(path) for path in self._source_paths],
            "consumed_source_hashes": dict(self._source_sha256s),
            "lineage_source_hashes": dict(self._lineage_source_sha256s),
            "consumed_window_sha256": self._source_window_sha256,
            "recipe_version": self._source_recipe_version,
            "consumed_channels": [
                "ngsim.states.actor_id",
                "ngsim.states.timestamp_ms",
                "ngsim.states.local_x_m",
                "ngsim.states.local_y_m",
                "ngsim.states.speed_mps",
                "ngsim.states.lane_id",
                "ngsim.source_events",
            ],
            "derived_backend_state_fields": [
                "ego.route_position_m",
                "ego.lateral_position_m",
                "ego.lane_index",
                "ego.speed_mps",
                "actors.route_position_m",
                "actors.lane_index",
                "actors.speed_mps",
            ],
            "source_field_to_state_field_map": {
                "local_y_m": "route_position_m",
                "local_x_m": "lateral_position_m",
                "lane_id": "lane_index",
                "speed_mps": "speed_mps",
            },
            "initial_state_digest": self._source_initial_state_digest,
            "consumption_ticks": list(self._source_consumption_ticks),
            "post_source_state_digests": list(self._source_state_digests),
            "runtime_trace_observed": bool(self._source_consumption_ticks),
            "evidence_from_scenario_config_only": False,
            "deterministic_source_trace": True,
            "trace_semantic_digest": _digest(self._source_state_digests),
            "state_effect_observed": state_effect,
            "source_state_effect_observed": event_coverage_complete,
            "expected_source_event_ids": expected_events,
            "observed_source_event_ids": observed_events,
            "material_source_event_ids": material_events,
            "source_event_materiality": source_event_materiality,
            "named_events_causally_proven": event_coverage_complete,
            "runtime_fidelity": (
                "source_initialized_constant_velocity_diagnostic"
                if self._config.get("source_bundle")
                else "file_backed_diagnostic_fixture"
            ),
            "formal_admission": "held_pending_live_sumo_reactive_validation",
            "blockers": (
                ["reactive_closed_loop_not_validated"]
                if self._source_consumption_ticks
                else ["source_trace_empty", "reactive_closed_loop_not_validated"]
            ),
        }

    def _source_state(self) -> dict[str, Any]:
        return {
            "ego": _jsonable(self._require_ego()),
            "actors": [_jsonable(actor) for actor in self._actors.values()],
        }

    def _require_ego(self) -> EgoState:
        if self._ego is None:
            raise RuntimeError("sumo_ego backend is not reset")
        return self._ego

    def close(self) -> None:
        return None


def build_sumo_ego_backend(config: dict[str, Any] | None = None) -> SumoEgoBackend:
    """Select live SUMO only when explicitly usable; otherwise use emulation."""
    cfg = dict(config or {})
    mode = str(cfg.get("execution_mode") or "auto").lower()
    if mode not in {"auto", "live", "emulated_source_initialized"}:
        raise ValueError(f"unsupported sumo_ego execution_mode: {mode}")
    if cfg.get("allow_live_lateral_maneuvers") is True:
        raise ValueError("live lateral maneuvers are held pending lane-geometry validation")
    config_declared = bool(cfg.get("sumo_config_path"))
    ego_declared = bool(cfg.get("ego_vehicle_id"))
    if mode == "auto" and config_declared != ego_declared:
        raise ValueError(
            "incomplete live sumo_ego intent requires sumo_config_path and ego_vehicle_id"
        )
    live_ready = config_declared and ego_declared
    if mode == "live" or (mode == "auto" and live_ready):
        try:
            live_module = importlib.import_module(
                "domains.autonomous_driving.backends.live_sumo_ego"
            )
            available = bool(live_module.live_sumo_available())
        except (ImportError, AttributeError):
            available = False
        if available:
            return live_module.LiveSumoEgoBackend(cfg)
        if mode == "live" or live_ready:
            raise RuntimeError("live sumo_ego requested but no native SUMO transport is available")
    return SumoEgoBackend(cfg)
