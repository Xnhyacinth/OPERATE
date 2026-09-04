"""Fail-closed native lane-change trajectory verification.

The verifier consumes lane centre lines and widths read from the running SUMO
network.  It deliberately has no synthetic-road fallback: malformed or too
short native geometry yields a rejected verification.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

PointTuple = tuple[float, float]


@dataclass(frozen=True)
class NativeVehicleState:
    """One vehicle state read from the active native simulator."""

    vehicle_id: str
    x: float
    y: float
    heading_rad: float
    speed_mps: float
    length_m: float
    width_m: float


@dataclass(frozen=True)
class TrajectorySample:
    elapsed_s: float
    x: float
    y: float
    heading_rad: float


@dataclass(frozen=True)
class ExecutedStateSample:
    elapsed_s: float
    x: float
    y: float
    heading_rad: float
    lane_id: str


@dataclass(frozen=True)
class PinnedLaneIndexMap:
    network_path: str
    network_sha256: str
    edge_id: str
    lanes: tuple[tuple[int, str], ...]

    def lane_id(self, index: int) -> str:
        for lane_index, lane_id in self.lanes:
            if lane_index == index:
                return lane_id
        raise ValueError(f"lane index is absent from pinned edge {self.edge_id}: {index}")

    def index_for(self, lane_id: str) -> int:
        for lane_index, candidate in self.lanes:
            if candidate == lane_id:
                return lane_index
        raise ValueError(f"lane id is absent from pinned edge {self.edge_id}: {lane_id}")


@dataclass(frozen=True)
class LaneChangeVerification:
    """Deterministic result for one complete native lane-change rollout."""

    verified: bool
    reason_codes: tuple[str, ...]
    sample_count: int
    trajectory_digest: str
    conflict_actor_id: str | None = None
    conflict_time_s: float | None = None
    minimum_boundary_margin_m: float | None = None
    trajectory: tuple[TrajectorySample, ...] = ()


@dataclass(frozen=True)
class ExecutedTrajectoryCertificate:
    verified: bool
    reason_codes: tuple[str, ...]
    sample_count: int
    candidate_trajectory_digest: str
    executed_trajectory_digest: str
    network_sha256: str
    current_lane_id: str
    target_lane_id: str
    maximum_position_error_m: float | None = None
    maximum_heading_error_rad: float | None = None


def load_pinned_lane_index_map(
    network_path: str | Path,
    *,
    edge_id: str,
    expected_sha256: str | None = None,
) -> PinnedLaneIndexMap:
    """Parse explicit lane indices from the exact SUMO network asset."""
    path = Path(network_path).resolve()
    if not path.is_file():
        raise ValueError(f"pinned SUMO network is missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError("pinned SUMO network sha256 mismatch")
    matches: list[tuple[tuple[int, str], ...]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rb") as stream:
            payload = stream.read(256 * 1024 * 1024 + 1)
        if len(payload) > 256 * 1024 * 1024:
            raise ValueError("pinned SUMO network exceeds the parser size limit")
        upper_payload = payload.upper()
        if b"<!DOCTYPE" in upper_payload or b"<!ENTITY" in upper_payload:
            raise ValueError("pinned SUMO network contains forbidden XML declarations")
        # The bounded payload has already rejected DTD and entity declarations.
        root = ET.fromstring(payload)  # nosec B314
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] != "edge":
                continue
            if str(element.attrib.get("id") or "") == edge_id:
                lanes: list[tuple[int, str]] = []
                for child in element:
                    if child.tag.rsplit("}", 1)[-1] != "lane":
                        continue
                    lane_id = str(child.attrib.get("id") or "").strip()
                    raw_index = child.attrib.get("index")
                    if not lane_id or raw_index is None:
                        raise ValueError("pinned SUMO lane index or identity is missing")
                    try:
                        lane_index = int(raw_index)
                    except ValueError as exc:
                        raise ValueError("pinned SUMO lane index is invalid") from exc
                    if lane_index < 0:
                        raise ValueError("pinned SUMO lane index is invalid")
                    lanes.append((lane_index, lane_id))
                matches.append(tuple(sorted(lanes)))
    except ET.ParseError as exc:
        raise ValueError("pinned SUMO network XML is invalid") from exc
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"pinned SUMO edge identity is missing or ambiguous: {edge_id}")
    parsed_lanes = matches[0]
    indices = tuple(index for index, _ in parsed_lanes)
    lane_ids = tuple(lane_id for _, lane_id in parsed_lanes)
    if len(set(indices)) != len(indices):
        raise ValueError("pinned SUMO lane index is duplicated")
    if len(set(lane_ids)) != len(lane_ids):
        raise ValueError("pinned SUMO lane identity is duplicated")
    return PinnedLaneIndexMap(
        network_path=str(path),
        network_sha256=digest,
        edge_id=edge_id,
        lanes=parsed_lanes,
    )


def verify_lane_change_trajectory(
    *,
    current_lane_shape: Sequence[PointTuple],
    target_lane_shape: Sequence[PointTuple],
    current_lane_width_m: float,
    target_lane_width_m: float,
    ego: NativeVehicleState,
    actors: Sequence[NativeVehicleState],
    duration_s: float,
    step_s: float,
    acceleration_mps2: float = 0.0,
    collision_margin_m: float = 0.1,
) -> LaneChangeVerification:
    """Verify footprint containment and collision freedom over every sample.

    Longitudinal motion follows both native lane centre lines.  A cubic smooth
    step blends between those centre lines for the lateral transition.  Every
    consecutive pair of vehicle footprints is checked as a swept polygon; an
    endpoint-only check is never sufficient.
    """

    empty_digest = _trajectory_digest(())
    if not _valid_scalar_inputs(
        current_lane_width_m=current_lane_width_m,
        target_lane_width_m=target_lane_width_m,
        duration_s=duration_s,
        step_s=step_s,
        acceleration_mps2=acceleration_mps2,
        collision_margin_m=collision_margin_m,
        ego=ego,
        actors=actors,
    ):
        return LaneChangeVerification(
            verified=False,
            reason_codes=("lane_geometry_input_invalid",),
            sample_count=0,
            trajectory_digest=empty_digest,
        )

    current_line = _native_line(current_lane_shape)
    target_line = _native_line(target_lane_shape)
    if current_line is None or target_line is None:
        return LaneChangeVerification(
            verified=False,
            reason_codes=("lane_geometry_invalid",),
            sample_count=0,
            trajectory_digest=empty_digest,
        )

    steps = max(1, math.ceil(duration_s / step_s))
    times = tuple(duration_s * index / steps for index in range(steps + 1))
    distances = _travel_distances(
        times,
        initial_speed_mps=ego.speed_mps,
        acceleration_mps2=acceleration_mps2,
    )
    start = Point(ego.x, ego.y)
    current_start = current_line.project(start)
    target_start = target_line.project(start)
    required_distance = distances[-1] + ego.length_m / 2.0 + collision_margin_m
    if (
        current_start + required_distance > current_line.length + 1e-9
        or target_start + required_distance > target_line.length + 1e-9
    ):
        return LaneChangeVerification(
            verified=False,
            reason_codes=("lane_shape_horizon_insufficient",),
            sample_count=len(times),
            trajectory_digest=empty_digest,
        )

    positions: list[PointTuple] = []
    for elapsed, distance in zip(times, distances, strict=True):
        current_point = current_line.interpolate(current_start + distance)
        target_point = target_line.interpolate(target_start + distance)
        fraction = elapsed / duration_s
        blend = fraction * fraction * (3.0 - 2.0 * fraction)
        positions.append(
            (
                current_point.x + blend * (target_point.x - current_point.x),
                current_point.y + blend * (target_point.y - current_point.y),
            )
        )
    positions[0] = (ego.x, ego.y)
    headings = _trajectory_headings(positions, ego.heading_rad)
    footprints = tuple(
        _vehicle_polygon(
            x=position[0],
            y=position[1],
            heading_rad=heading,
            length_m=ego.length_m,
            width_m=ego.width_m,
            margin_m=collision_margin_m,
        )
        for position, heading in zip(positions, headings, strict=True)
    )
    digest = _trajectory_digest(
        tuple(
            (round(x, 9), round(y, 9), round(heading, 9))
            for (x, y), heading in zip(positions, headings, strict=True)
        )
    )
    trajectory = tuple(
        TrajectorySample(
            elapsed_s=elapsed,
            x=position[0],
            y=position[1],
            heading_rad=heading,
        )
        for elapsed, position, heading in zip(times, positions, headings, strict=True)
    )

    corridor = unary_union(
        (
            current_line.buffer(current_lane_width_m / 2.0, cap_style=2, join_style=2),
            target_line.buffer(target_lane_width_m / 2.0, cap_style=2, join_style=2),
        )
    )
    minimum_margin = math.inf
    swept_footprints = [footprints[0]]
    swept_footprints.extend(
        unary_union((before, after)).convex_hull
        for before, after in zip(footprints, footprints[1:], strict=False)
    )
    for swept in swept_footprints:
        if not corridor.buffer(1e-9).covers(swept):
            return LaneChangeVerification(
                verified=False,
                reason_codes=("native_road_boundary_violation",),
                sample_count=len(times),
                trajectory_digest=digest,
                minimum_boundary_margin_m=-math.sqrt(max(0.0, swept.difference(corridor).area)),
                trajectory=trajectory,
            )
        minimum_margin = min(minimum_margin, corridor.boundary.distance(swept))

    for actor in sorted(actors, key=lambda value: value.vehicle_id):
        actor_footprints = tuple(
            _vehicle_polygon(
                x=actor.x + actor.speed_mps * elapsed * math.cos(actor.heading_rad),
                y=actor.y + actor.speed_mps * elapsed * math.sin(actor.heading_rad),
                heading_rad=actor.heading_rad,
                length_m=actor.length_m,
                width_m=actor.width_m,
                margin_m=collision_margin_m,
            )
            for elapsed in times
        )
        if footprints[0].intersects(actor_footprints[0]):
            return LaneChangeVerification(
                verified=False,
                reason_codes=("trajectory_conflict",),
                sample_count=len(times),
                trajectory_digest=digest,
                conflict_actor_id=actor.vehicle_id,
                conflict_time_s=0.0,
                minimum_boundary_margin_m=minimum_margin,
                trajectory=trajectory,
            )
        for index, (ego_before, ego_after, actor_before, actor_after) in enumerate(
            zip(
                footprints,
                footprints[1:],
                actor_footprints,
                actor_footprints[1:],
                strict=False,
            ),
            start=1,
        ):
            ego_swept = unary_union((ego_before, ego_after)).convex_hull
            actor_swept = unary_union((actor_before, actor_after)).convex_hull
            if ego_swept.intersects(actor_swept):
                return LaneChangeVerification(
                    verified=False,
                    reason_codes=("trajectory_conflict",),
                    sample_count=len(times),
                    trajectory_digest=digest,
                    conflict_actor_id=actor.vehicle_id,
                    conflict_time_s=times[index],
                    minimum_boundary_margin_m=minimum_margin,
                    trajectory=trajectory,
                )

    return LaneChangeVerification(
        verified=True,
        reason_codes=(),
        sample_count=len(times),
        trajectory_digest=digest,
        minimum_boundary_margin_m=minimum_margin,
        trajectory=trajectory,
    )


def certify_executed_trajectory(
    *,
    candidate: LaneChangeVerification,
    readbacks: Sequence[ExecutedStateSample],
    network_sha256: str,
    current_lane_id: str,
    target_lane_id: str,
    position_tolerance_m: float = 0.15,
    heading_tolerance_rad: float = 0.05,
    time_tolerance_s: float = 1e-6,
) -> ExecutedTrajectoryCertificate:
    """Compare every native executed-state readback with one candidate.

    This is deliberately post-hoc evidence.  It cannot authorize the same
    maneuver whose executed states it observes.
    """
    executed_digest = _executed_trajectory_digest(readbacks)
    def build_certificate(
        *,
        verified: bool,
        reason_codes: tuple[str, ...],
        maximum_position_error_m: float | None = None,
        maximum_heading_error_rad: float | None = None,
    ) -> ExecutedTrajectoryCertificate:
        return ExecutedTrajectoryCertificate(
            verified=verified,
            reason_codes=reason_codes,
            sample_count=len(readbacks),
            candidate_trajectory_digest=candidate.trajectory_digest,
            executed_trajectory_digest=executed_digest,
            network_sha256=network_sha256,
            current_lane_id=current_lane_id,
            target_lane_id=target_lane_id,
            maximum_position_error_m=maximum_position_error_m,
            maximum_heading_error_rad=maximum_heading_error_rad,
        )
    tolerances = (position_tolerance_m, heading_tolerance_rad, time_tolerance_s)
    network_digest_valid = bool(
        len(network_sha256) == 64
        and all(character in "0123456789abcdef" for character in network_sha256.lower())
    )
    if (
        not all(math.isfinite(value) and value >= 0.0 for value in tolerances)
        or not network_digest_valid
        or not current_lane_id
        or not target_lane_id
        or current_lane_id == target_lane_id
    ):
        return build_certificate(
            verified=False,
            reason_codes=("executed_certificate_input_invalid",),
        )
    if not candidate.verified or not candidate.trajectory:
        return build_certificate(
            verified=False,
            reason_codes=("candidate_trajectory_unverified",),
        )
    if len(readbacks) != len(candidate.trajectory):
        return build_certificate(
            verified=False,
            reason_codes=("executed_sample_count_mismatch",),
        )

    reasons: list[str] = []
    maximum_position_error = 0.0
    maximum_heading_error = 0.0
    allowed_lane_ids = {current_lane_id, target_lane_id}
    for expected, observed in zip(candidate.trajectory, readbacks, strict=True):
        values = (
            observed.elapsed_s,
            observed.x,
            observed.y,
            observed.heading_rad,
        )
        if not observed.lane_id or not all(math.isfinite(value) for value in values):
            reasons.append("executed_state_invalid")
            continue
        if observed.lane_id not in allowed_lane_ids:
            reasons.append("executed_lane_identity_mismatch")
        if abs(observed.elapsed_s - expected.elapsed_s) > time_tolerance_s:
            reasons.append("executed_time_mismatch")
        position_error = math.hypot(observed.x - expected.x, observed.y - expected.y)
        heading_error = abs(_normalize_angle(observed.heading_rad - expected.heading_rad))
        maximum_position_error = max(maximum_position_error, position_error)
        maximum_heading_error = max(maximum_heading_error, heading_error)
    if maximum_position_error > position_tolerance_m:
        reasons.append("executed_position_mismatch")
    if maximum_heading_error > heading_tolerance_rad:
        reasons.append("executed_heading_mismatch")
    if readbacks[-1].lane_id != target_lane_id:
        reasons.append("executed_target_lane_not_reached")
    ordered_reasons = tuple(dict.fromkeys(reasons))
    return build_certificate(
        verified=not ordered_reasons,
        reason_codes=ordered_reasons,
        maximum_position_error_m=maximum_position_error,
        maximum_heading_error_rad=maximum_heading_error,
    )


def _native_line(shape: Sequence[PointTuple]) -> LineString | None:
    try:
        points = tuple((float(x), float(y)) for x, y in shape)
    except (TypeError, ValueError):
        return None
    if len(points) < 2 or any(not math.isfinite(value) for point in points for value in point):
        return None
    line = LineString(points)
    if not line.is_valid or not line.is_simple or line.length <= 0.0:
        return None
    return line


def _valid_scalar_inputs(
    *,
    current_lane_width_m: float,
    target_lane_width_m: float,
    duration_s: float,
    step_s: float,
    acceleration_mps2: float,
    collision_margin_m: float,
    ego: NativeVehicleState,
    actors: Sequence[NativeVehicleState],
) -> bool:
    positive = (
        current_lane_width_m,
        target_lane_width_m,
        duration_s,
        step_s,
        ego.length_m,
        ego.width_m,
    )
    nonnegative = (collision_margin_m, ego.speed_mps)
    ego_values = (ego.x, ego.y, ego.heading_rad, acceleration_mps2)
    return bool(
        all(math.isfinite(value) and value > 0.0 for value in positive)
        and all(math.isfinite(value) and value >= 0.0 for value in nonnegative)
        and all(math.isfinite(value) for value in ego_values)
        and ego.vehicle_id
        and all(_valid_vehicle(actor) for actor in actors)
    )


def _valid_vehicle(vehicle: NativeVehicleState) -> bool:
    values = (
        vehicle.x,
        vehicle.y,
        vehicle.heading_rad,
        vehicle.speed_mps,
        vehicle.length_m,
        vehicle.width_m,
    )
    return bool(
        vehicle.vehicle_id
        and all(math.isfinite(value) for value in values)
        and vehicle.speed_mps >= 0.0
        and vehicle.length_m > 0.0
        and vehicle.width_m > 0.0
    )


def _travel_distances(
    times: Sequence[float],
    *,
    initial_speed_mps: float,
    acceleration_mps2: float,
) -> tuple[float, ...]:
    stop_time = initial_speed_mps / -acceleration_mps2 if acceleration_mps2 < 0.0 else math.inf
    return tuple(
        initial_speed_mps * min(elapsed, stop_time)
        + 0.5 * acceleration_mps2 * min(elapsed, stop_time) ** 2
        for elapsed in times
    )


def _trajectory_headings(
    positions: Sequence[PointTuple],
    initial_heading_rad: float,
) -> tuple[float, ...]:
    headings: list[float] = []
    for index, position in enumerate(positions):
        if index + 1 < len(positions):
            target = positions[index + 1]
        elif index > 0:
            target = position
            position = positions[index - 1]
        else:
            headings.append(initial_heading_rad)
            continue
        delta_x = target[0] - position[0]
        delta_y = target[1] - position[1]
        headings.append(
            math.atan2(delta_y, delta_x)
            if math.hypot(delta_x, delta_y) > 1e-12
            else initial_heading_rad
        )
    return tuple(headings)


def _vehicle_polygon(
    *,
    x: float,
    y: float,
    heading_rad: float,
    length_m: float,
    width_m: float,
    margin_m: float,
) -> Polygon:
    half_length = length_m / 2.0 + margin_m
    half_width = width_m / 2.0 + margin_m
    cosine = math.cos(heading_rad)
    sine = math.sin(heading_rad)
    points = []
    for longitudinal, lateral in (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ):
        points.append(
            (
                x + longitudinal * cosine - lateral * sine,
                y + longitudinal * sine + lateral * cosine,
            )
        )
    return Polygon(points)


def _trajectory_digest(trajectory: Sequence[tuple[float, float, float]]) -> str:
    payload = json.dumps(tuple(trajectory), separators=(",", ":"), sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _executed_trajectory_digest(readbacks: Sequence[ExecutedStateSample]) -> str:
    payload = tuple(
        (
            round(sample.elapsed_s, 9),
            round(sample.x, 9),
            round(sample.y, 9),
            round(sample.heading_rad, 9),
            sample.lane_id,
        )
        for sample in readbacks
    )
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=False).encode()
    ).hexdigest()


def _normalize_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi
