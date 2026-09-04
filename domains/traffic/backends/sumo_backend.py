"""
domains.traffic.backends.sumo_backend — Real SUMO sidecar backend.

v0.7 traffic spike strategy (per ``docs/v0.7_traffic_spec.md`` §2/§11),
mirroring the disaster RCRS approach in
``domains/disaster/backends/rcrs_backend.py``:

- The traffic vertical slice ships a fully-working
  :class:`~domains.traffic.backends.mock_sumo.MockSumoBackend` (pure
  Python, deterministic, no SUMO) so the benchmark is RUNNABLE
  end-to-end on a clean checkout (stage 1–3).
- This module drives a live microsimulation through
  :class:`core.sidecar.sumo_sidecar.SumoSidecar` (``libsumo`` →
  ``traci`` → Docker ``eclipse/sumo``) behind an explicit env gate.

Why a separate class instead of feature-flagging the mock:

- The real backend has external preconditions (a reachable SUMO
  transport, a compiled ``.net.xml`` / ``.rou.xml`` pair). Encoding
  these in a single class would force the mock to ship the same
  failure modes, breaking the "spike must run on a laptop without
  SUMO" invariant in ``.hl/policy.md``.
- A separate class lets the adapter select mock vs real via
  ``backend_kind`` in the seed without runtime branches in the hot
  path.

Activation contract (stage 4):

- The adapter constructs ``SumoBackend(cfg)`` only when
  ``seed.backend_kind == 'sumo'``.
- ``reset()`` requires ``OPERATE_TRAFFIC_BACKEND_REAL=1`` *and* a reachable
  SUMO transport (``sumo_available()``). Without the env flag it raises
  ``RuntimeError`` so an audit can confirm no scoring run accidentally
  used the stub; with the flag but no transport it raises
  :class:`~core.sidecar.sumo_sidecar.SumoSidecarUnavailable` so live
  runners can *gracefully skip* (the s4-traffic gate), exactly like
  ``audit._egret_available()`` gating the microgrid AC-OPF path.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from core.sidecar.sumo_sidecar import (
    SumoSidecar,
    SumoSidecarUnavailable,
    sumo_available,
)
from domains.traffic.runtime_control_contract import (
    RuntimeControlContractError,
    classify_vehicle_movement_link_indices,
    parsed_program_ids_by_tls,
    resolve_sumocfg_asset_graph,
    validate_phase_duration_request,
    validate_program_logic,
)
from domains.traffic.source_identity import (
    build_sumo_source_identity_payload,
    compute_sumo_source_identity,
    normalize_sumo_version,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..seeds.schema import TrafficScenarioSeed

_SPEC_DOC = "docs/v0.7_traffic_spec.md"
_PHASE_NOTE = (
    "Live SUMO adapter has only implemented the signal-plan control mapping; "
    f"see {_SPEC_DOC} §2 (transport) / §11 (live-execution gate)."
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_DECLARED_PERTURBATION_EVENT_CLASS = MappingProxyType(
    {
        "demand_surge": "alarm",
        "lane_blockage": "safety",
        "weather_capacity_drop": "alarm",
    }
)


@dataclass
class _LiveTrafficTickRecord:
    tick: int
    aggregate_offered: float = 0.0
    aggregate_served: float = 0.0
    aggregate_queue: float = 0.0
    aggregate_delay_minutes: float = 0.0
    travel_cost_this_tick: float = 0.0
    shed_cost_this_tick: float = 0.0
    actuation_cost_this_tick: float = 0.0
    mutual_aid_cost_this_tick: float = 0.0
    reserves_required: float = 0.0
    reserves_procured: float = 0.0
    rho_max: float = 0.0
    n_overloads: int = 0
    n_gridlocked: int = 0
    n_blocked_edges: int = 0
    realized_events: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False


class SumoBackend:
    """Real SUMO sidecar transport backend.

    Reset and tick execute the SUMO sidecar. The formal control surface is
    restricted to runtime-enumerated programs and bounded phase-duration
    supervision; controls without a native mapping remain blocked.
    """

    backend_kind = "sumo"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = dict(cfg or {})
        self._cfg = cfg
        # Artifact handles the sidecar will need (compiled offline from the
        # seed's network provenance; absent in the mock-only stage).
        self._net_path = cfg.get("sumo_net_path")
        self._route_path = cfg.get("sumo_route_path")
        self._config_path = cfg.get("sumo_config_path")
        self._sidecar: Any = None
        self._seed_obj: TrafficScenarioSeed | None = None
        self._tick: int = 0
        self._horizon: int = 0
        self._tick_records: list[_LiveTrafficTickRecord] = []
        self._last_snapshot: dict[str, Any] | None = None
        self._transport: str | None = None
        self._sumo_substeps_per_tick: int = 1
        self._physics_step_seconds: float = 1.0
        self._decision_interval_seconds: float = 30.0
        self._max_agent_vehicle_records = max(
            1,
            min(256, int(cfg.get("max_agent_vehicle_records", 24))),
        )
        self._runtime_control_contract: dict[str, Any] | None = None
        self._pending_signal_controls: list[dict[str, Any]] = []
        self._materialized_signal_controls: list[dict[str, Any]] = []
        self._applied_declared_perturbations: set[int] = set()
        self._pending_declared_perturbation_events: list[dict[str, Any]] = []
        self._edge_speed_overlays: dict[int, dict[str, Any]] = {}
        self._lane_blockage_overlays: dict[int, dict[str, Any]] = {}
        self._source_trace_rows: list[dict[str, Any]] = []
        self._corridor_tls_map: dict[str, str] = {
            str(k): str(v) for k, v in dict(cfg.get("corridor_tls_map") or {}).items()
        }
        self._signal_program_map: dict[str, str] = {
            str(k): str(v)
            for k, v in dict(cfg.get("sumo_signal_program_map") or {}).items()
        }
        # Per-corridor benchmark-program → real present-program-id map. Each
        # corridor's bound TLS exposes its own program ids on the real net, so a
        # single global map cannot bind every TLS; the locked binding artifact
        # supplies a distinct, net-derived program map per corridor.
        self._corridor_program_map: dict[str, dict[str, str]] = {
            str(corridor): {str(k): str(v) for k, v in dict(progs or {}).items()}
            for corridor, progs in dict(
                cfg.get("sumo_corridor_program_map") or {}
            ).items()
        }
        self._live_signal_state: dict[str, dict[str, Any]] = {}
        # corridor_id -> de-duplicated SUMO lanes its bound TLS controls, derived
        # live at reset(). Per-corridor live physical feedback (queue/delay) is
        # aggregated over these lanes each tick so the equity dimension
        # (per_corridor_delay_minutes) is real + evidence-linked, not a stub.
        self._corridor_lanes: dict[str, tuple[str, ...]] = {}
        # corridor_id -> cumulative delay-minutes (time integral of queued
        # vehicles * tick_minutes), the per-corridor decomposition of the
        # aggregate delay metric. Reset per episode.
        self._corridor_delay_minutes: dict[str, float] = {}
        self._corridor_last_metrics: dict[str, dict[str, float]] = {}

    def _attribution_coverage(self) -> dict[str, Any]:
        unique_lanes = {
            str(lane)
            for lanes in self._corridor_lanes.values()
            for lane in tuple(lanes or ())
        }
        network_counts = (
            (self._last_snapshot or {}).get("network_counts")
            if isinstance(self._last_snapshot, dict)
            else {}
        )
        if not isinstance(network_counts, dict):
            network_counts = {}
        n_lanes_raw = network_counts.get("n_lanes")
        n_edges_raw = network_counts.get("n_edges")
        n_lanes = int(n_lanes_raw) if n_lanes_raw is not None else None
        n_edges = int(n_edges_raw) if n_edges_raw is not None else None
        unique_count = len(unique_lanes)
        return {
            "bound_corridor_count": len(self._corridor_tls_map),
            "bound_tls_count": len(set(self._corridor_tls_map.values())),
            "tls_with_controlled_lanes": sum(
                1 for lanes in self._corridor_lanes.values() if lanes
            ),
            "unique_controlled_lanes": unique_count,
            "network_lane_count": n_lanes,
            "network_edge_count": n_edges,
            "controlled_lane_share_of_network": (
                round(unique_count / n_lanes, 3) if n_lanes else None
            ),
            "unattributed_network_lanes_estimate": (
                max(0, n_lanes - unique_count) if n_lanes is not None else None
            ),
            "zero_lane_corridors": sorted(
                corridor
                for corridor, lanes in self._corridor_lanes.items()
                if not lanes
            ),
            "missing_network_lane_denominator": n_lanes is None,
        }

    def _snapshot_with_network_counts(self, tick: int) -> dict[str, Any]:
        if self._sidecar is None:
            raise SumoSidecarUnavailable("snapshot() before reset()/sidecar start")
        raw = dict(self._sidecar.snapshot(tick=tick))
        if isinstance(raw.get("network_counts"), dict):
            return raw
        getter = getattr(self._sidecar, "network_counts", None)
        if callable(getter):
            try:
                raw["network_counts"] = getter()
            except Exception:
                raw["network_counts"] = {"n_lanes": None, "n_edges": None}
        else:
            raw["network_counts"] = {"n_lanes": None, "n_edges": None}
        return raw

    def reset(self, scenario_seed: TrafficScenarioSeed) -> None:
        """Launch SUMO via the sidecar, apply the seed, prime tick 0.

        Two-stage gate so audits and live runners get distinct signals:

        1. ``OPERATE_TRAFFIC_BACKEND_REAL != "1"`` → ``RuntimeError``. The
           real path is opt-in; the default checkout must never reach a
           half-built backend.
        2. flag set but ``sumo_available() is False`` → raise
           :class:`SumoSidecarUnavailable` so the stage-4 runner records
           ``executed_with_live_backend=False`` and *skips* rather than
           failing the suite (matches the EGRET / RCRS graceful-skip
           contract).
        """
        if self._sidecar is not None:
            self.close()
        if os.environ.get("OPERATE_TRAFFIC_BACKEND_REAL") != "1":
            raise SumoSidecarUnavailable(
                "SumoBackend.reset() called without "
                "OPERATE_TRAFFIC_BACKEND_REAL=1; real SUMO execution is opt-in. "
                f"See {_SPEC_DOC}. To run a traffic scenario on a clean "
                "checkout, use the default backend_kind='mock_sumo'."
            )
        if not sumo_available():
            raise SumoSidecarUnavailable(
                "OPERATE_TRAFFIC_BACKEND_REAL=1 but no SUMO transport is "
                "reachable (libsumo / traci / docker all absent). Install "
                f"SUMO 1.20+ or unset the flag. See {_SPEC_DOC} §2."
            )

        net_path = _resolve_required_path(
            self._net_path or scenario_seed.net_ref,
            "SUMO network",
        )
        route_path = _resolve_required_path(
            self._route_path or scenario_seed.route_ref,
            "SUMO route/demand",
        )
        config_path = (
            _resolve_required_path(self._config_path, "SUMO config")
            if self._config_path
            else None
        )
        step_length = float(self._cfg.get("sumo_step_length_seconds", 1.0) or 1.0)
        self._physics_step_seconds = float(
            self._cfg.get("physics_step_seconds", step_length) or step_length
        )
        self._decision_interval_seconds = float(
            self._cfg.get(
                "decision_interval_seconds",
                int(scenario_seed.tick_minutes or 5) * 60,
            )
            or 30.0
        )
        default_substeps = max(
            1,
            int(round(self._decision_interval_seconds / self._physics_step_seconds)),
        )
        self._sumo_substeps_per_tick = int(
            self._cfg.get("sumo_substeps_per_tick", default_substeps)
            or default_substeps
        )
        if self._cfg.get("enforce_decision_interval_contract") is True:
            declared_tick_seconds = float(scenario_seed.tick_minutes or 0) * 60.0
            if not math.isclose(
                self._decision_interval_seconds, declared_tick_seconds
            ) or not math.isclose(
                self._sumo_substeps_per_tick * self._physics_step_seconds,
                self._decision_interval_seconds,
            ):
                raise RuntimeError(
                    "traffic_decision_interval_substep_mismatch: tick_minutes, "
                    "decision_interval_seconds, physics_step_seconds, and "
                    "sumo_substeps_per_tick must describe one interval"
                )
        self._seed_obj = scenario_seed
        self._tick = 0
        self._horizon = int(scenario_seed.horizon_ticks)
        self._tick_records.clear()
        self._last_snapshot = None
        self._live_signal_state.clear()
        self._pending_signal_controls.clear()
        self._materialized_signal_controls.clear()
        self._applied_declared_perturbations.clear()
        self._pending_declared_perturbation_events.clear()
        self._edge_speed_overlays.clear()
        self._lane_blockage_overlays.clear()
        self._source_trace_rows.clear()
        self._sidecar = SumoSidecar(
            net_path=str(net_path),
            route_path=str(route_path),
            seed=int(scenario_seed.seed),
            step_length=step_length,
            routing_algorithm=str(self._cfg.get("sumo_routing_algorithm", "dijkstra")),
            docker_image=str(self._cfg.get("sumo_docker_image", "eclipse/sumo:1.20.0")),
            traci_port=(
                int(self._cfg["sumo_traci_port"])
                if self._cfg.get("sumo_traci_port") is not None
                else None
            ),
            extra_args=tuple(self._cfg.get("sumo_extra_args", ()) or ()),
            config_path=str(config_path) if config_path is not None else None,
        )
        try:
            self._transport = str(self._sidecar.start())
            self._last_snapshot = self._snapshot_with_network_counts(tick=0)
            self._runtime_control_contract = self._build_runtime_control_contract(
                config_path
            )
            self._record_source_trace(tick=0, snapshot=self._last_snapshot)
            binding_network_sha = str(
                self._cfg.get("sumo_tls_binding_net_sha256") or ""
            )
            runtime_network_sha = str(
                (self._runtime_control_contract or {})
                .get("source_assets", {})
                .get("network", {})
                .get("sha256", "")
            )
            if binding_network_sha and binding_network_sha != runtime_network_sha:
                # The old corridor binding belongs to another physical network.
                # It remains available for its exact source but cannot be reused.
                self._corridor_tls_map = {}
                self._corridor_program_map = {}
            # Derive each mapped corridor's controlled lanes from its bound TLS so
            # per-corridor live queue/delay can be aggregated each tick. Done once at
            # reset (topology is static within an episode).
            self._corridor_lanes = {
                corridor: self._sidecar.controlled_lanes(tls_id)
                for corridor, tls_id in self._corridor_tls_map.items()
            }
            self._validate_signal_program_bindings()
            self._corridor_delay_minutes = {
                corridor: 0.0 for corridor in self._corridor_tls_map
            }
            self._corridor_last_metrics = {}
        except Exception:
            self.close()
            raise

    def _build_runtime_control_contract(
        self, config_path: Path | None
    ) -> dict[str, Any] | None:
        if self._sidecar is None or config_path is None:
            return None
        assets = resolve_sumocfg_asset_graph(config_path)
        parsed = parsed_program_ids_by_tls(assets)
        tls_rows: dict[str, Any] = {}
        rejected: list[dict[str, Any]] = []
        for tls_id in self._sidecar.traffic_light_ids():
            row = self._sidecar.traffic_light_contract(tls_id)
            classification = classify_vehicle_movement_link_indices(
                row["controlled_links"]
            )
            safe_programs: list[str] = []
            for program_id, program in row["programs"].items():
                if classification["status"] != "passed":
                    rejected.append(
                        {
                            "tls_id": tls_id,
                            "program_id": program_id,
                            "reason_code": (
                                "traffic_vehicle_movement_link_classification_invalid"
                            ),
                            "details": classification,
                        }
                    )
                    continue
                try:
                    validate_program_logic(
                        tls_id=tls_id,
                        program_id=program_id,
                        controlled_link_count=len(row["controlled_links"]),
                        parsed_program_ids=parsed.get(tls_id, set()),
                        phases=program.get("phases") or [],
                        vehicle_movement_link_indices=classification[
                            "vehicle_movement_link_indices"
                        ],
                    )
                except RuntimeControlContractError as exc:
                    rejected.append(
                        {
                            "tls_id": tls_id,
                            "program_id": program_id,
                            "reason_code": exc.code,
                            "details": exc.details,
                        }
                    )
                else:
                    safe_programs.append(program_id)
            tls_rows[tls_id] = {
                **row,
                "vehicle_movement_link_classification": classification,
                "safe_selectable_program_ids": safe_programs,
            }
        source_assets = assets
        version = self._sidecar.runtime_version()
        normalized_version = normalize_sumo_version(version)
        transport = {
            "traci": "traci_tcp",
            "libsumo": "libsumo_in_process",
            "docker": "docker_traci_tcp",
        }.get(str(self._transport), str(self._transport or "unknown"))
        required_version = str(self._cfg.get("required_sumo_version") or "").strip()
        required_transport = str(
            self._cfg.get("required_sumo_transport") or ""
        ).strip()
        if required_version and normalized_version != normalize_sumo_version(
            required_version
        ):
            raise RuntimeError(
                "traffic_sumo_runtime_version_mismatch: "
                f"required={required_version}, observed={normalized_version}"
            )
        if required_transport and transport != required_transport:
            raise RuntimeError(
                "traffic_sumo_transport_identity_mismatch: "
                f"required={required_transport}, observed={transport}"
            )
        identity_payload = build_sumo_source_identity_payload(
            assets,
            service_date=str(self._cfg.get("service_date") or ""),
            sumo_version=version,
            transport=transport,
        )
        identity = compute_sumo_source_identity(identity_payload)
        return {
            "schema_version": "1.0",
            "complete_source_identity_sha256": identity,
            "complete_source_identity_payload": identity_payload,
            "source_assets": source_assets,
            "service_date": str(self._cfg.get("service_date") or ""),
            "sumo_version": normalized_version,
            "traci_server_version_raw": version,
            "traci_api_version": (
                int(version.split()[0])
                if version.split()
                and version.split()[0].isdigit()
                else None
            ),
            "transport": transport,
            "physics_step_seconds": self._physics_step_seconds,
            "decision_interval_seconds": self._decision_interval_seconds,
            "tls": tls_rows,
            "rejected_programs": rejected,
        }

    def _validate_signal_program_bindings(self) -> None:
        """Reject scenario controls that name programs absent from live SUMO."""
        if self._sidecar is None:
            raise RuntimeError("SUMO sidecar is not started")
        invalid: dict[str, dict[str, Any]] = {}
        for corridor, mapping in self._corridor_program_map.items():
            tls_id = self._corridor_tls_map.get(corridor)
            if not tls_id:
                invalid[corridor] = {
                    "reason": "missing_sumo_tls_mapping",
                    "declared_programs": sorted(set(mapping.values())),
                }
                continue
            available = set(self._sidecar.traffic_light_program_ids(tls_id))
            missing = sorted(set(mapping.values()) - available)
            if missing:
                invalid[corridor] = {
                    "reason": "declared_program_absent_from_live_tls",
                    "tls_id": tls_id,
                    "missing_programs": missing,
                    "available_programs": sorted(available),
                }
        if invalid:
            raise RuntimeError(
                "live_sumo_signal_program_contract_invalid: "
                + json.dumps(invalid, sort_keys=True)
            )

    def tick(self, current_tick: int) -> Any:
        if self._sidecar is None:
            raise SumoSidecarUnavailable("tick() before reset()/sidecar start")
        self._tick = int(current_tick)
        self._apply_declared_perturbations_at_tick()
        self._materialize_pending_signal_controls()
        interval_arrived = 0
        interval_departed = 0
        counts_fn = getattr(self._sidecar, "simulation_counts", None)
        for _ in range(self._sumo_substeps_per_tick):
            self._sidecar.simulation_step()
            self._materialize_pending_signal_controls()
            if callable(counts_fn):
                counts = counts_fn()
                interval_arrived += int(counts.get("arrived", 0) or 0)
                interval_departed += int(counts.get("departed", 0) or 0)
        raw = self._snapshot_with_network_counts(tick=self._tick)
        if callable(counts_fn):
            raw["interval_arrived"] = interval_arrived
            raw["interval_departed"] = interval_departed
        self._last_snapshot = raw
        self._record_source_trace(tick=self._tick, snapshot=raw)
        realized_events = self._finalize_declared_perturbation_events(raw)
        n_vehicles = float(raw.get("n_vehicles", 0.0) or 0.0)
        arrived = float(
            raw.get("interval_arrived", raw.get("arrived", 0.0)) or 0.0
        )
        departed = float(
            raw.get("interval_departed", raw.get("departed", 0.0)) or 0.0
        )
        served = max(0.0, arrived)
        queue = max(0.0, n_vehicles)
        offered = max(n_vehicles, departed + arrived)
        # Per-corridor live physical feedback: aggregate queue/delay over each
        # corridor's controlled lanes and integrate delay-minutes across ticks.
        tick_minutes = float(self._tick_minutes())
        per_corridor: dict[str, dict[str, float]] = {}
        for corridor, lanes in self._corridor_lanes.items():
            metrics = self._sidecar.lane_group_metrics(lanes)
            corridor_queue = float(metrics["halting"])
            delay_increment = round(corridor_queue * tick_minutes, 3)
            self._corridor_delay_minutes[corridor] = round(
                self._corridor_delay_minutes.get(corridor, 0.0) + delay_increment, 3
            )
            per_corridor[corridor] = {
                "queue": round(corridor_queue, 3),
                "vehicles": round(float(metrics["vehicles"]), 3),
                "delay_minutes_increment": delay_increment,
                "cumulative_delay_minutes": self._corridor_delay_minutes[corridor],
                "waiting_time_s": round(float(metrics["waiting_time_s"]), 3),
                "n_lanes": int(metrics["n_lanes"]),
            }
        self._corridor_last_metrics = per_corridor
        realized_events.insert(
            0,
            {
                "type": "sumo_live_snapshot",
                "tick": self._tick,
                "actionable": False,
                "decision_required": False,
                "transport": self._transport,
                "native_physics_step_count": self._sumo_substeps_per_tick,
                "decision_interval_seconds": self._decision_interval_seconds,
                "n_vehicles": int(n_vehicles),
                "arrived": int(arrived),
                "departed": int(departed),
                "per_corridor": per_corridor,
                "runtime_signal_control": self._snapshot_runtime_tls_state(),
                "materialized_signal_controls": [
                    row
                    for row in self._materialized_signal_controls
                    if row.get("applied_at_tick") == self._tick
                ],
                "attribution_coverage": self._attribution_coverage(),
            }
        )
        if arrived > 0 or departed > 0:
            source_identity = str(
                (self._runtime_control_contract or {}).get(
                    "complete_source_identity_sha256"
                )
                or ""
            )
            realized_events.append(
                {
                    "event_id": (
                        f"sumo-source-flow:{source_identity}:{self._tick}"
                    ),
                    "type": "traffic_demand_change",
                    "origin": "source_schedule",
                    "tick": self._tick,
                    "actionable": False,
                    "decision_required": False,
                    "interval_arrived": int(arrived),
                    "interval_departed": int(departed),
                    "changed_state_fields": [
                        "n_vehicles",
                        "interval_arrived",
                        "interval_departed",
                        "controlled_lane_queues",
                    ],
                    "materiality_metric": "interval_vehicle_flow",
                    "materiality_value": int(arrived + departed),
                    "materiality_threshold": 1,
                    "response_window_required": False,
                    "evidence_ids": [
                        f"sumo:{source_identity}:source-flow:{self._tick}"
                    ],
                }
            )
        record = _LiveTrafficTickRecord(
            tick=self._tick,
            aggregate_offered=round(offered, 3),
            aggregate_served=round(served, 3),
            aggregate_queue=round(queue, 3),
            aggregate_delay_minutes=round(queue * float(self._tick_minutes()), 3),
            travel_cost_this_tick=round(queue * float(self._tick_minutes()) * 0.30, 3),
            rho_max=round((queue / max(1.0, served)), 4),
            n_overloads=0,
            realized_events=realized_events,
            done=False,
        )
        self._tick_records.append(record)
        return record

    def snapshot(self) -> dict[str, Any]:
        raw = self._last_snapshot
        if raw is None:
            if self._sidecar is None:
                raise SumoSidecarUnavailable("snapshot() before reset()/sidecar start")
            raw = self._snapshot_with_network_counts(tick=self._tick)
            self._last_snapshot = raw
        last = self._tick_records[-1] if self._tick_records else None
        entities = {}
        if self._seed_obj is not None:
            entities = {
                c.corridor_id: {
                    "kind": "corridor",
                    "district": c.district,
                    "demand_veh": c.demand_veh,
                    "queue": self._corridor_last_metrics.get(c.corridor_id, {}).get(
                        "queue", 0.0
                    ),
                    "delay_minutes": self._corridor_delay_minutes.get(
                        c.corridor_id, 0.0
                    ),
                    "criticality": c.criticality,
                    "carries_ems_corridor": c.carries_ems_corridor,
                    "carries_vip_route": c.carries_vip_route,
                    "live_sumo_mapped": c.corridor_id in self._corridor_tls_map,
                    **self._live_signal_state.get(c.corridor_id, {}),
                }
                for c in self._seed_obj.corridors
            }
        return {
            "backend": "sumo",
            "source_integration_rung": "executed_with_live_backend_probe",
            "live_control_surface": (
                "signal_plan_mapping_available"
                if self._corridor_tls_map
                else "blocked_pending_native_sumo_control_mapping"
            ),
            "sumo": dict(raw),
            "runtime_signal_control": {
                "complete_source_identity_sha256": (
                    self._runtime_control_contract or {}
                ).get("complete_source_identity_sha256"),
                "legal_tls_ids": sorted(
                    (self._runtime_control_contract or {}).get("tls") or {}
                ),
                "tls": self._snapshot_runtime_tls_state(),
                "pending_controls": list(self._pending_signal_controls),
                "recent_materialized_controls": list(
                    self._materialized_signal_controls[-10:]
                ),
                "physics_step_seconds": self._physics_step_seconds,
                "decision_interval_seconds": self._decision_interval_seconds,
                "next_native_decision_opportunity_seconds": (
                    float(raw.get("sim_time", 0.0))
                    + self._decision_interval_seconds
                ),
            },
            "vehicle_control_capture": self._agent_vehicle_control_capture(),
            "entities": entities,
            "totals": {
                "aggregate_offered": last.aggregate_offered if last else 0.0,
                "aggregate_served": last.aggregate_served if last else 0.0,
                "aggregate_queue": last.aggregate_queue if last else 0.0,
                "aggregate_delay_minutes": (
                    last.aggregate_delay_minutes if last else 0.0
                ),
            },
            "tick": self._tick,
            "horizon": self._horizon,
        }

    def _snapshot_runtime_tls_state(self) -> dict[str, dict[str, Any]]:
        """Expose only the current native state needed to form a legal action."""
        if self._sidecar is None:
            return {}
        contract_rows = (
            (self._runtime_control_contract or {}).get("tls") or {}
        )
        states: dict[str, dict[str, Any]] = {}
        for tls_id in sorted(set(self._corridor_tls_map.values())):
            if tls_id not in contract_rows:
                continue
            try:
                runtime = self._sidecar.traffic_light_contract(tls_id)
            except (AttributeError, SumoSidecarUnavailable):
                continue
            states[tls_id] = {
                "current_program": runtime.get("current_program"),
                "current_phase": runtime.get("current_phase"),
                "current_state": runtime.get("current_state"),
                "remaining_duration": runtime.get("remaining_duration"),
                "current_phase_bounds": dict(
                    runtime.get("current_phase_bounds") or {}
                ),
                "safe_selectable_program_ids": list(
                    contract_rows[tls_id].get(
                        "safe_selectable_program_ids"
                    )
                    or []
                ),
            }
        return states

    def _capture_vehicle_control_context(self) -> dict[str, Any]:
        """Bind live vehicle/link/phase context to the exact SUMO source identity."""
        source_identity = str(
            (self._runtime_control_contract or {}).get(
                "complete_source_identity_sha256"
            )
            or ""
        )
        if self._sidecar is None:
            return {
                "status": "unavailable",
                "complete_source_identity_sha256": source_identity,
                "records": [],
                "blockers": ["sumo_sidecar_not_started"],
            }
        tls_ids = sorted(set(self._corridor_tls_map.values()))
        if not tls_ids:
            return {
                "status": "missing",
                "complete_source_identity_sha256": source_identity,
                "records": [],
                "blockers": ["traffic_binding_tls_missing"],
            }
        records_by_tls: dict[str, list[dict[str, Any]]] = {}
        try:
            for tls_id in tls_ids:
                tls_records: list[dict[str, Any]] = []
                for raw in self._sidecar.vehicle_control_context(tls_id):
                    record = dict(raw)
                    vehicle_id = str(record.get("vehicle_id") or "")
                    record.setdefault("tls_context", {"tls_id": tls_id})
                    evidence_id = (
                        f"sumo:{source_identity}:{tls_id}:{vehicle_id}"
                    )
                    record["evidence_ids"] = [evidence_id]
                    lineage = dict(record.get("source_lineage") or {})
                    lineage["evidence_ids"] = [evidence_id]
                    record["source_lineage"] = lineage
                    tls_records.append(record)
                records_by_tls[tls_id] = tls_records
        except SumoSidecarUnavailable as exc:
            return {
                "status": "unavailable",
                "complete_source_identity_sha256": source_identity,
                "records": [],
                "blockers": [str(exc)],
            }
        record_count = sum(len(rows) for rows in records_by_tls.values())
        records: list[dict[str, Any]] = []
        offset = 0
        while len(records) < self._max_agent_vehicle_records:
            appended = False
            for tls_id in tls_ids:
                tls_records = records_by_tls.get(tls_id, [])
                if offset < len(tls_records):
                    records.append(tls_records[offset])
                    appended = True
                    if len(records) >= self._max_agent_vehicle_records:
                        break
            if not appended:
                break
            offset += 1
        return {
            "status": "complete" if record_count else "empty",
            "complete_source_identity_sha256": source_identity,
            "record_count": record_count,
            "returned_record_count": len(records),
            "truncated": record_count > len(records),
            "max_agent_vehicle_records": self._max_agent_vehicle_records,
            "records_by_tls": {
                tls_id: len(rows) for tls_id, rows in records_by_tls.items()
            },
            "records": records,
            "blockers": (
                [] if record_count else ["no_controlled_vehicle_observed"]
            ),
        }

    def _agent_vehicle_control_capture(self) -> dict[str, Any]:
        """Agent-facing vehicle rows: keep control state, drop provenance dumps."""
        capture = self._capture_vehicle_control_context()
        records = capture.get("records")
        if not isinstance(records, list):
            return capture
        capture = dict(capture)
        capture["records"] = [
            self._agent_vehicle_record(row) if isinstance(row, dict) else row
            for row in records
        ]
        return capture

    @staticmethod
    def _agent_vehicle_record(record: dict[str, Any]) -> dict[str, Any]:
        tls = record.get("tls_context") or {}
        phase = record.get("phase_context") or {}
        return {
            "vehicle_id": record.get("vehicle_id"),
            "edge_id": record.get("edge_id"),
            "lane_id": record.get("lane_id"),
            "tls_context": {"tls_id": tls.get("tls_id") if isinstance(tls, dict) else tls},
            "phase_context": {
                "program_id": phase.get("program_id") if isinstance(phase, dict) else None,
                "phase_index": phase.get("phase_index") if isinstance(phase, dict) else None,
                "link_signal_state": (
                    phase.get("link_signal_state") if isinstance(phase, dict) else None
                ),
            },
            "evidence_ids": list(record.get("evidence_ids") or [])[:2],
        }

    @staticmethod
    def _declared_event_state_digest(snapshot: dict[str, Any]) -> str:
        payload = {
            key: snapshot.get(key)
            for key in (
                "sim_time",
                "n_vehicles",
                "arrived",
                "departed",
                "interval_arrived",
                "interval_departed",
            )
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def _event_state_digest(state: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _restore_edge_speed_overlays_at_tick(
        self, *, source_identity: str
    ) -> None:
        if self._sidecar is None:
            return
        for index, overlay in list(self._edge_speed_overlays.items()):
            if int(overlay["restore_tick"]) != self._tick:
                continue
            mutations = [
                self._sidecar.set_edge_lane_max_speeds(
                    edge_id=edge_id,
                    lane_max_speeds=original_speeds,
                )
                for edge_id, original_speeds in overlay[
                    "original_lane_speeds"
                ].items()
            ]
            before_state = {
                "edge_lane_max_speeds": {
                    row["edge_id"]: row["before_lane_max_speeds"]
                    for row in mutations
                }
            }
            after_state = {
                "edge_lane_max_speeds": {
                    row["edge_id"]: row["after_lane_max_speeds"]
                    for row in mutations
                }
            }
            self._pending_declared_perturbation_events.append(
                {
                    "type": "weather_capacity_restored",
                    "event_id": f"{overlay['event_id']}:restore",
                    "origin": "endogenous_completion",
                    "declared_perturbation": True,
                    "tick": self._tick,
                    "hidden": bool(overlay["hidden"]),
                    "event_class": "lifecycle",
                    "decision_required": False,
                    "actionable": False,
                    "response_window_required": False,
                    "changed_state_fields": ["edge_speed_limits"],
                    "materiality_metric": "restored_runtime_edge_count",
                    "materiality_threshold": 1,
                    "source_identity_sha256": source_identity,
                    "evidence_ids": [
                        f"sumo:{source_identity}:declared:{overlay['trigger_tick']}:{index}:restore"
                    ],
                    "native_mutations": mutations,
                    "before_state": before_state,
                    "after_state": after_state,
                    "before_state_digest": self._event_state_digest(before_state),
                    "after_state_digest": self._event_state_digest(after_state),
                    "event_effect_kind": "edge_speed",
                    "sumo_state_mutated": any(
                        row["sumo_state_mutated"] for row in mutations
                    ),
                    "application_status": "accepted",
                }
            )
            del self._edge_speed_overlays[index]

    def _restore_lane_blockages_at_tick(self, *, source_identity: str) -> None:
        if self._sidecar is None:
            return
        for index, overlay in list(self._lane_blockage_overlays.items()):
            if int(overlay["restore_tick"]) != self._tick:
                continue
            mutations = [
                self._sidecar.set_lane_disallowed(
                    lane_id=lane_id,
                    disallowed_classes=tuple(original_classes),
                )
                for lane_id, original_classes in overlay[
                    "original_disallowed_classes"
                ].items()
            ]
            before_state = {
                "lane_disallowed_classes": {
                    row["lane_id"]: row["before_disallowed_classes"]
                    for row in mutations
                }
            }
            after_state = {
                "lane_disallowed_classes": {
                    row["lane_id"]: row["after_disallowed_classes"]
                    for row in mutations
                }
            }
            self._pending_declared_perturbation_events.append(
                {
                    "type": "traffic_lane_blockage_restored",
                    "event_id": f"{overlay['event_id']}:restore",
                    "origin": "endogenous_completion",
                    "declared_perturbation": True,
                    "tick": self._tick,
                    "hidden": bool(overlay["hidden"]),
                    "event_class": "lifecycle",
                    "decision_required": False,
                    "actionable": False,
                    "response_window_required": False,
                    "changed_state_fields": ["lane_permissions"],
                    "materiality_metric": "restored_runtime_lane_count",
                    "materiality_threshold": 1,
                    "source_identity_sha256": source_identity,
                    "evidence_ids": [
                        f"sumo:{source_identity}:declared:{overlay['trigger_tick']}:{index}:restore"
                    ],
                    "native_mutations": mutations,
                    "before_state": before_state,
                    "after_state": after_state,
                    "before_state_digest": self._event_state_digest(before_state),
                    "after_state_digest": self._event_state_digest(after_state),
                    "event_effect_kind": "lane_disallowed",
                    "sumo_state_mutated": any(
                        row["sumo_state_mutated"] for row in mutations
                    ),
                    "application_status": "accepted",
                }
            )
            del self._lane_blockage_overlays[index]

    def _apply_lane_blockage(
        self,
        *,
        index: int,
        perturbation: Any,
        target: dict[str, Any],
        source_identity: str,
        horizon: int,
    ) -> dict[str, Any]:
        lane_ids = sorted(
            {
                str(value).strip()
                for value in target.get("lane_ids") or []
                if str(value).strip()
            }
        )
        disallowed_classes = tuple(
            sorted(
                {
                    str(value).strip()
                    for value in target.get("disallowed_classes")
                    or ["passenger"]
                    if str(value).strip()
                }
            )
        )
        tls_id = str(target.get("tls_id") or "").strip()
        route_id = str(target.get("route_id") or "").strip()
        event_id = f"sumo-declared:{source_identity}:{self._tick}:{index}"
        decision_required = (
            not bool(perturbation.hidden) and self._tick + 1 < horizon
        )
        event: dict[str, Any] = {
            "type": "traffic_lane_blockage",
            "event_id": event_id,
            "origin": "declared_perturbation",
            "declared_perturbation": True,
            "declared_event": {
                "kind": "lane_blockage",
                "trigger_tick": int(perturbation.trigger_tick),
                "duration_ticks": int(perturbation.duration_ticks),
                "target": target,
                "intensity": float(perturbation.intensity),
            },
            "tick": self._tick,
            "hidden": bool(perturbation.hidden),
            "event_class": _DECLARED_PERTURBATION_EVENT_CLASS["lane_blockage"],
            "decision_required": decision_required,
            "actionable": decision_required,
            "changed_state_fields": ["lane_permissions"],
            "materiality_metric": "runtime_lane_permission_mutations",
            "materiality_threshold": 1,
            "response_window_required": True,
            "response_opportunity_tick": (
                self._tick + 1 if self._tick + 1 < horizon else None
            ),
            "terminal_response_window_missing": self._tick + 1 >= horizon,
            "source_identity_sha256": source_identity,
            "evidence_ids": [
                f"sumo:{source_identity}:declared:{self._tick}:{index}"
            ],
            "application_status": "held",
            "event_effect_kind": "lane_disallowed",
            "sumo_state_mutated": False,
        }
        try:
            if not lane_ids:
                raise ValueError("lane_blockage requires source-locked lane_ids")
            if not disallowed_classes:
                raise ValueError("lane_blockage requires disallowed_classes")
            if not tls_id or not route_id:
                raise ValueError(
                    "lane_blockage requires exact tls_id and route_id coupling"
                )
            if self._sidecar is None:
                raise SumoSidecarUnavailable("SUMO sidecar is not started")
            controlled_lanes = set(self._sidecar.controlled_lanes(tls_id))
            route_edges = set(self._sidecar.route_edges(route_id))
            lane_edge_ids = {
                lane_id: self._sidecar.lane_edge_id(lane_id)
                for lane_id in lane_ids
            }
            if not set(lane_ids).issubset(controlled_lanes):
                raise ValueError(
                    "lane_blockage lanes are not controlled by the declared TLS"
                )
            if not set(lane_edge_ids.values()).issubset(route_edges):
                raise ValueError(
                    "lane_blockage edges are absent from the declared runtime route"
                )
            passenger_enabled_before = {
                lane_id: self._sidecar.lane_allows_vehicle_class(
                    lane_id, "passenger"
                )
                for lane_id in lane_ids
            }
            if not all(passenger_enabled_before.values()):
                raise ValueError(
                    "lane_blockage target is not passenger-enabled before mutation"
                )
            original_disallowed = {
                lane_id: self._sidecar.lane_disallowed_classes(lane_id)
                for lane_id in lane_ids
            }
            mutations = [
                self._sidecar.set_lane_disallowed(
                    lane_id=lane_id,
                    disallowed_classes=tuple(
                        sorted(
                            set(original_disallowed[lane_id]).union(
                                disallowed_classes
                            )
                        )
                    ),
                )
                for lane_id in lane_ids
            ]
            passenger_enabled_after = {
                lane_id: self._sidecar.lane_allows_vehicle_class(
                    lane_id, "passenger"
                )
                for lane_id in lane_ids
            }
            capacity_reduction = sum(
                passenger_enabled_before[lane_id]
                and not passenger_enabled_after[lane_id]
                for lane_id in lane_ids
            )
            if capacity_reduction != len(lane_ids):
                raise RuntimeError(
                    "lane_blockage did not reduce passenger-enabled lane capacity"
                )
            before_state = {
                "lane_disallowed_classes": {
                    row["lane_id"]: row["before_disallowed_classes"]
                    for row in mutations
                }
            }
            after_state = {
                "lane_disallowed_classes": {
                    row["lane_id"]: row["after_disallowed_classes"]
                    for row in mutations
                }
            }
            event.update(
                {
                    "application_status": "accepted",
                    "native_mutations": mutations,
                    "before_state": before_state,
                    "after_state": after_state,
                    "before_state_digest": self._event_state_digest(before_state),
                    "after_state_digest": self._event_state_digest(after_state),
                    "sumo_state_mutated": any(
                        row["sumo_state_mutated"] for row in mutations
                    ),
                    "source_coupling": {
                        "tls_id": tls_id,
                        "route_id": route_id,
                        "lane_edge_ids": sorted(set(lane_edge_ids.values())),
                        "controlled_lane_match": True,
                        "route_edge_match": True,
                    },
                    "passenger_enabled_lane_count_before": sum(
                        passenger_enabled_before.values()
                    ),
                    "passenger_enabled_lane_count_after": sum(
                        passenger_enabled_after.values()
                    ),
                    "passenger_flow_capacity_reduction_lanes": capacity_reduction,
                }
            )
            self._lane_blockage_overlays[index] = {
                "event_id": event_id,
                "trigger_tick": self._tick,
                "restore_tick": self._tick
                + max(1, int(perturbation.duration_ticks)),
                "hidden": bool(perturbation.hidden),
                "original_disallowed_classes": original_disallowed,
            }
        except (ValueError, SumoSidecarUnavailable, RuntimeError) as exc:
            event.update(
                {
                    "application_error": f"{type(exc).__name__}: {exc}",
                    "error_code": "traffic_lane_blockage_not_implemented",
                }
            )
        return event

    def _apply_declared_perturbations_at_tick(self) -> None:
        """Apply only runtime-supported procedural traffic perturbations.

        A declaration is not evidence by itself.  ``demand_surge`` is realized
        by injecting vehicles onto a route that SUMO reports as loaded from the
        locked route graph.  Unsupported kinds are recorded as held events so
        audits fail closed instead of treating inert YAML as a shock.
        """
        if self._seed_obj is None or self._sidecar is None:
            return
        # Unit-level/fake-sidecar probes may install ``_seed_obj`` directly
        # without going through ``reset()``.  Keep the event contract tied to
        # the declared scenario horizon in that case rather than treating the
        # uninitialised runtime horizon (0) as terminal.
        horizon = self._horizon or int(
            getattr(self._seed_obj, "horizon_ticks", 0) or 0
        )
        source_identity = str(
            (self._runtime_control_contract or {}).get(
                "complete_source_identity_sha256"
            )
            or ""
        )
        self._restore_edge_speed_overlays_at_tick(source_identity=source_identity)
        self._restore_lane_blockages_at_tick(source_identity=source_identity)
        perturbations = getattr(self._seed_obj, "perturbations", ()) or ()
        for index, perturbation in enumerate(perturbations):
            if index in self._applied_declared_perturbations:
                continue
            if int(perturbation.trigger_tick) != self._tick:
                continue
            self._applied_declared_perturbations.add(index)
            before = self._snapshot_with_network_counts(tick=self._tick)
            target = dict(perturbation.target or {})
            kind = str(perturbation.kind)
            if kind == "weather_capacity_drop":
                event = self._apply_weather_capacity_drop(
                    index=index,
                    perturbation=perturbation,
                    target=target,
                    source_identity=source_identity,
                    horizon=horizon,
                )
                self._pending_declared_perturbation_events.append(event)
                continue
            if kind == "lane_blockage":
                event = self._apply_lane_blockage(
                    index=index,
                    perturbation=perturbation,
                    target=target,
                    source_identity=source_identity,
                    horizon=horizon,
                )
                self._pending_declared_perturbation_events.append(event)
                continue
            supported_kind = kind in _DECLARED_PERTURBATION_EVENT_CLASS
            decision_required = bool(
                supported_kind
                and not bool(perturbation.hidden)
                and self._tick + 1 < horizon
            )
            event: dict[str, Any] = {
                "type": (
                    "traffic_demand_surge"
                    if kind == "demand_surge"
                    else "traffic_procedural_perturbation_failed"
                ),
                "event_id": (
                    f"sumo-declared:{source_identity}:{self._tick}:{index}"
                ),
                "origin": "declared_perturbation",
                "declared_perturbation": True,
                "declared_event": {
                    "kind": str(perturbation.kind),
                    "trigger_tick": int(perturbation.trigger_tick),
                    "duration_ticks": int(perturbation.duration_ticks),
                    "target": target,
                    "intensity": float(perturbation.intensity),
                },
                "tick": self._tick,
                "hidden": bool(perturbation.hidden),
                "event_class": (
                    _DECLARED_PERTURBATION_EVENT_CLASS[kind]
                    if supported_kind
                    else "routine"
                ),
                "decision_required": decision_required,
                "actionable": decision_required,
                "changed_state_fields": [
                    "n_vehicles",
                    "runtime_vehicle_ids",
                    "route_departures",
                ],
                "materiality_metric": "injected_runtime_vehicle_count",
                "materiality_threshold": 1,
                "response_window_required": True,
                "response_opportunity_tick": (
                    self._tick + 1 if self._tick + 1 < horizon else None
                ),
                "terminal_response_window_missing": self._tick + 1 >= horizon,
                "before_state_digest": self._declared_event_state_digest(before),
                "source_identity_sha256": source_identity,
                "evidence_ids": [
                    f"sumo:{source_identity}:declared:{self._tick}:{index}"
                ],
                "application_status": "held",
                "application_error": None,
                "injected_vehicle_ids": [],
                "runtime_route_id": None,
            }
            if kind != "demand_surge":
                # Without a native vehicle inventory there is no runtime
                # identity from which to attribute this declaration.  Keep the
                # explicit ``unknown`` origin in that fail-closed case; a
                # runtime-aware test double/back-end retains the declared
                # origin so downstream audits can distinguish a held event.
                if not callable(getattr(self._sidecar, "vehicle_ids", None)):
                    event["origin"] = "unknown"
                event["runtime_origin"] = "unknown"
                event["status"] = "held"
                event["error_code"] = "traffic_procedural_kind_not_implemented"
                event["sumo_state_mutated"] = False
                event["application_error"] = (
                    "traffic_runtime_perturbation_not_implemented"
                )
                self._pending_declared_perturbation_events.append(event)
                continue
            try:
                requested_route = str(target.get("route_id") or "").strip()
                count = max(1, int(target.get("vehicle_count") or 1))
                route_inventory = getattr(self._sidecar, "route_ids", None)
                inject_one = getattr(
                    self._sidecar, "inject_vehicle_from_route", None
                )
                if callable(route_inventory) and callable(inject_one):
                    route_ids = route_inventory()
                    route_id = requested_route or (route_ids[0] if route_ids else "")
                    if not route_id:
                        raise ValueError("runtime route inventory is empty")
                else:
                    legacy_inject = getattr(
                        self._sidecar, "inject_demand_surge", None
                    )
                    if not callable(legacy_inject):
                        raise SumoSidecarUnavailable(
                            "SUMO native demand injection API is unavailable"
                        )
                    source_vehicle_id = str(target.get("source_vehicle_id") or "")
                    if not source_vehicle_id:
                        context_getter = getattr(
                            self._sidecar, "vehicle_control_context", None
                        )
                        if callable(context_getter):
                            context = context_getter(str(target.get("tls_id") or ""))
                            if context:
                                source_vehicle_id = str(
                                    (context[0] or {}).get("vehicle_id") or ""
                                )
                    legacy = legacy_inject(
                        source_vehicle_id=source_vehicle_id,
                        event_id=str(event["event_id"]),
                        vehicle_count=count,
                    )
                    event.update(
                        {
                            "application_status": (
                                "accepted"
                                if legacy.get("sumo_state_mutated") is True
                                else "held"
                            ),
                            "injected_vehicle_ids": list(
                                legacy.get("overlay_vehicle_ids") or []
                            ),
                            "native_mutations": [legacy],
                            "runtime_route_id": legacy.get("overlay_route_id")
                            or legacy.get("source_route_id"),
                            "sumo_state_mutated": bool(
                                legacy.get("sumo_state_mutated") is True
                            ),
                            "before_state": legacy.get("before_state"),
                            "after_state": legacy.get("after_state"),
                        }
                    )
                    self._pending_declared_perturbation_events.append(event)
                    continue
                vehicle_ids: list[str] = []
                mutations: list[dict[str, Any]] = []
                for offset in range(count):
                    vehicle_id = (
                        f"dt-procedural-{source_identity[:12]}-"
                        f"{self._tick}-{index}-{offset}"
                    )
                    mutations.append(
                        self._sidecar.inject_vehicle_from_route(
                            vehicle_id=vehicle_id,
                            route_id=route_id,
                        )
                    )
                    vehicle_ids.append(vehicle_id)
                event.update(
                    {
                        "application_status": "accepted",
                        "runtime_route_id": route_id,
                        "injected_vehicle_ids": vehicle_ids,
                        "native_mutations": mutations,
                        "sumo_state_mutated": True,
                    }
                )
            except (ValueError, SumoSidecarUnavailable, RuntimeError) as exc:
                event["application_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
            self._pending_declared_perturbation_events.append(event)

    def _apply_weather_capacity_drop(
        self,
        *,
        index: int,
        perturbation: Any,
        target: dict[str, Any],
        source_identity: str,
        horizon: int,
    ) -> dict[str, Any]:
        edge_ids = sorted({str(value) for value in target.get("edge_ids") or [] if str(value)})
        factor = float(target.get("capacity_factor") or 0.0)
        tls_id = str(target.get("tls_id") or "").strip()
        route_id = str(target.get("route_id") or "").strip()
        event_id = f"sumo-declared:{source_identity}:{self._tick}:{index}"
        decision_required = (
            not bool(perturbation.hidden) and self._tick + 1 < horizon
        )
        event: dict[str, Any] = {
            "type": "weather_capacity_drop",
            "event_id": event_id,
            "origin": "declared_perturbation",
            "declared_perturbation": True,
            "declared_event": {
                "kind": "weather_capacity_drop",
                "trigger_tick": int(perturbation.trigger_tick),
                "duration_ticks": int(perturbation.duration_ticks),
                "target": target,
                "intensity": float(perturbation.intensity),
            },
            "tick": self._tick,
            "hidden": bool(perturbation.hidden),
            "event_class": _DECLARED_PERTURBATION_EVENT_CLASS[
                "weather_capacity_drop"
            ],
            "decision_required": decision_required,
            "actionable": decision_required,
            "changed_state_fields": [
                "edge_speed_limits",
                "controlled_lane_queues",
                "waiting_time",
            ],
            "materiality_metric": "runtime_edge_speed_mutations",
            "materiality_threshold": 1,
            "response_window_required": True,
            "response_opportunity_tick": (
                self._tick + 1 if self._tick + 1 < horizon else None
            ),
            "terminal_response_window_missing": self._tick + 1 >= horizon,
            "source_identity_sha256": source_identity,
            "evidence_ids": [f"sumo:{source_identity}:declared:{self._tick}:{index}"],
            "application_status": "held",
            "event_effect_kind": "edge_speed",
            "sumo_state_mutated": False,
        }
        try:
            if not edge_ids:
                raise ValueError("weather_capacity_drop requires source-locked edge_ids")
            if not 0.0 < factor < 1.0:
                raise ValueError("capacity_factor must be strictly between zero and one")
            if self._sidecar is None:
                raise SumoSidecarUnavailable("SUMO sidecar is not started")
            runtime_edges = set(self._sidecar.edge_ids())
            missing = sorted(set(edge_ids) - runtime_edges)
            if missing:
                raise ValueError(
                    f"weather edge_ids are absent from the runtime graph: {missing}"
                )
            source_coupling: dict[str, Any] | None = None
            if tls_id or route_id:
                if not tls_id or not route_id:
                    raise ValueError(
                        "weather_capacity_drop coupling requires tls_id and route_id"
                    )
                controlled_edges = {
                    self._sidecar.lane_edge_id(lane_id)
                    for lane_id in self._sidecar.controlled_lanes(tls_id)
                }
                route_edges = set(self._sidecar.route_edges(route_id))
                if not set(edge_ids).issubset(controlled_edges):
                    raise ValueError(
                        "weather edges are not controlled by the declared TLS"
                    )
                if not set(edge_ids).issubset(route_edges):
                    raise ValueError(
                        "weather edges are absent from the declared runtime route"
                    )
                source_coupling = {
                    "tls_id": tls_id,
                    "route_id": route_id,
                    "edge_ids": edge_ids,
                    "controlled_edge_match": True,
                    "route_edge_match": True,
                }
            original_speeds = {
                edge_id: self._sidecar.edge_lane_max_speeds(edge_id)
                for edge_id in edge_ids
            }
            mutations = [
                self._sidecar.set_edge_lane_max_speeds(
                    edge_id=edge_id,
                    lane_max_speeds={
                        lane_id: speed * factor
                        for lane_id, speed in original_speeds[edge_id].items()
                    },
                )
                for edge_id in edge_ids
            ]
            before_state = {
                "edge_lane_max_speeds": {
                    row["edge_id"]: row["before_lane_max_speeds"]
                    for row in mutations
                }
            }
            after_state = {
                "edge_lane_max_speeds": {
                    row["edge_id"]: row["after_lane_max_speeds"]
                    for row in mutations
                }
            }
            event.update(
                {
                    "application_status": "accepted",
                    "native_mutations": mutations,
                    "before_state": before_state,
                    "after_state": after_state,
                    "before_state_digest": self._event_state_digest(before_state),
                    "after_state_digest": self._event_state_digest(after_state),
                    "sumo_state_mutated": any(
                        row["sumo_state_mutated"] for row in mutations
                    ),
                    **(
                        {"source_coupling": source_coupling}
                        if source_coupling is not None
                        else {}
                    ),
                }
            )
            self._edge_speed_overlays[index] = {
                "event_id": event_id,
                "trigger_tick": self._tick,
                "restore_tick": self._tick + max(1, int(perturbation.duration_ticks)),
                "hidden": bool(perturbation.hidden),
                "original_lane_speeds": original_speeds,
            }
        except (ValueError, SumoSidecarUnavailable, RuntimeError) as exc:
            event.update(
                {
                    "application_error": f"{type(exc).__name__}: {exc}",
                    "error_code": "traffic_weather_capacity_drop_not_implemented",
                }
            )
        return event

    def _finalize_declared_perturbation_events(
        self, snapshot: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not self._pending_declared_perturbation_events:
            return []
        after_digest = self._declared_event_state_digest(snapshot)
        active_vehicle_ids: set[str] = set()
        vehicle_inventory = getattr(self._sidecar, "vehicle_ids", None)
        if callable(vehicle_inventory):
            try:
                active_vehicle_ids = set(vehicle_inventory())
            except (SumoSidecarUnavailable, AttributeError):
                active_vehicle_ids = set()
        finalized: list[dict[str, Any]] = []
        for event in self._pending_declared_perturbation_events:
            if event.get("event_effect_kind") in {"edge_speed", "lane_disallowed"}:
                mutations = list(event.get("native_mutations") or [])
                changed = [
                    row
                    for row in mutations
                    if isinstance(row, dict) and row.get("sumo_state_mutated") is True
                ]
                event.update(
                    {
                        "materiality_value": len(changed),
                        "materiality_passed": bool(
                            event.get("application_status") == "accepted" and changed
                        ),
                        "native_state_effect_observed": bool(changed),
                        "application_status": (
                            "passed"
                            if event.get("application_status") == "accepted" and changed
                            else event.get("application_status")
                        ),
                    }
                )
                finalized.append(event)
                continue
            injected = [str(value) for value in event.get("injected_vehicle_ids") or []]
            after_state = event.get("after_state") or {}
            fallback_ids = {
                str(value) for value in after_state.get("vehicle_ids") or []
            }
            observed = sorted(
                set(injected).intersection(active_vehicle_ids or fallback_ids)
            )
            material_value = len(observed)
            event.update(
                {
                    "after_state_digest": after_digest,
                    "observed_runtime_vehicle_ids": observed,
                    "materiality_value": material_value,
                    "materiality_passed": bool(
                        event.get("application_status") == "accepted"
                        and observed
                        and event.get("before_state_digest") != after_digest
                    ),
                    "native_state_effect_observed": bool(observed),
                    "sumo_state_mutated": bool(
                        event.get("sumo_state_mutated") is True
                        or event.get("application_status") == "accepted"
                    ),
                    "application_status": (
                        "passed"
                        if event.get("application_status") == "accepted"
                        and observed
                        else event.get("application_status")
                    ),
                }
            )
            finalized.append(event)
        self._pending_declared_perturbation_events.clear()
        return finalized

    def apply_tool_effect(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name in ("wait", "noop"):
            return {"_status": "waited" if name == "wait" else "noop"}
        if name in ("query_network_state", "query_detector", "inspect_intersection"):
            return {
                "_status": "ok",
                "tool": name,
                "corridor": str(args.get("corridor") or args.get("corridor_id") or ""),
                "sumo_snapshot": dict(self._last_snapshot or {}),
                "live_sumo_mapped": False,
            }
        if name == "query_signal_control":
            return self._query_signal_control(args)
        if name == "set_signal_program":
            return self._set_signal_program(args)
        if name == "set_signal_phase_duration":
            return self._set_signal_phase_duration(args)
        if name == "change_signal_plan":
            return self._change_signal_plan(args)
        if name == "extend_current_green_phase":
            return self._extend_current_green_phase(args)
        return {
            "_status": "error",
            "error": "live_sumo_control_not_implemented",
            "tool": name,
            "sumo_state_mutated": False,
        }

    def _query_signal_control(self, args: dict[str, Any]) -> dict[str, Any]:
        contract = self._runtime_control_contract or {}
        assets = contract.get("source_assets") or {}
        opened = [
            row
            for key in ("sumocfg", "network")
            if isinstance((row := assets.get(key)), dict)
        ]
        for key in ("route_files", "additional_files", "recursive_inputs"):
            opened.extend(
                row
                for row in assets.get(key) or []
                if isinstance(row, dict)
            )
        tls_rows = contract.get("tls") or {}
        tls_id = str(args.get("tls_id") or "")
        if tls_id and tls_id not in tls_rows:
            return {
                "_status": "error",
                "error": "traffic_binding_tls_missing",
                "known_tls_ids": sorted(tls_rows),
                "sumo_state_mutated": False,
            }
        selected = [tls_id] if tls_id else sorted(tls_rows)[:25]
        rows: dict[str, Any] = {}
        for current_tls in selected:
            runtime = (
                self._sidecar.traffic_light_contract(current_tls)
                if self._sidecar is not None
                else dict(tls_rows[current_tls])
            )
            lanes = tuple(runtime.get("controlled_lanes") or ())
            controlled_edges = (
                {
                    self._sidecar.lane_edge_id(lane_id)
                    for lane_id in lanes
                }
                if self._sidecar is not None
                else set()
            )
            metrics = (
                self._sidecar.lane_group_metrics(lanes)
                if self._sidecar is not None
                else {}
            )
            active_disruptions = [
                {
                    "kind": "weather_capacity_drop",
                    "event_id": overlay.get("event_id"),
                    "started_tick": overlay.get("trigger_tick"),
                    "affected_edge_ids": sorted(
                        set(overlay.get("original_lane_speeds") or {})
                        & controlled_edges
                    ),
                }
                for overlay in self._edge_speed_overlays.values()
                if set(overlay.get("original_lane_speeds") or {})
                & controlled_edges
            ]
            active_disruptions.extend(
                {
                    "kind": "lane_blockage",
                    "event_id": overlay.get("event_id"),
                    "started_tick": overlay.get("trigger_tick"),
                    "affected_lane_ids": sorted(
                        set(overlay.get("original_disallowed_classes") or {})
                        & set(lanes)
                    ),
                }
                for overlay in self._lane_blockage_overlays.values()
                if set(overlay.get("original_disallowed_classes") or {})
                & set(lanes)
            )
            rows[current_tls] = {
                **runtime,
                "safe_selectable_program_ids": tls_rows[current_tls].get(
                    "safe_selectable_program_ids", []
                ),
                "controlled_lane_metrics": metrics,
                "active_disruptions": active_disruptions,
            }
        return {
            "_status": "ok",
            "tool": "query_signal_control",
            "complete_source_identity_sha256": contract.get(
                "complete_source_identity_sha256"
            ),
            "source_identity": {
                "complete_source_identity_sha256": contract.get(
                    "complete_source_identity_sha256"
                ),
                "payload": contract.get(
                    "complete_source_identity_payload"
                ),
            },
            "runtime_opened_assets": opened,
            "tls": rows,
            "pending_controls": list(self._pending_signal_controls),
            "recent_materialized_controls": list(
                self._materialized_signal_controls[-10:]
            ),
            "next_native_decision_opportunity_seconds": (
                self._decision_interval_seconds
            ),
            "sumo_state_mutated": False,
        }

    def _set_signal_program(self, args: dict[str, Any]) -> dict[str, Any]:
        tls_id = str(args.get("tls_id") or "")
        program_id = str(args.get("program_id") or "")
        tls = ((self._runtime_control_contract or {}).get("tls") or {}).get(tls_id)
        if tls is None:
            return self._control_error("traffic_binding_tls_missing")
        if program_id not in set(tls.get("safe_selectable_program_ids") or ()):
            return self._control_error("traffic_binding_program_missing_on_tls")
        if self._sidecar is None:
            return self._control_error("sumo_sidecar_not_started")
        runtime = self._sidecar.traffic_light_contract(tls_id)
        runtime_program = (runtime.get("programs") or {}).get(program_id)
        classification = classify_vehicle_movement_link_indices(
            runtime["controlled_links"]
        )
        if classification["status"] != "passed":
            return self._control_error(
                "traffic_vehicle_movement_link_classification_invalid"
            )
        try:
            validate_program_logic(
                tls_id=tls_id,
                program_id=program_id,
                controlled_link_count=len(runtime["controlled_links"]),
                parsed_program_ids=parsed_program_ids_by_tls(
                    (self._runtime_control_contract or {}).get(
                        "source_assets", {}
                    )
                ).get(tls_id, set()),
                phases=(runtime_program or {}).get("phases") or [],
                vehicle_movement_link_indices=classification[
                    "vehicle_movement_link_indices"
                ],
            )
        except RuntimeControlContractError as exc:
            return self._control_error(exc.code)
        if program_id == runtime["current_program"]:
            return self._control_error("traffic_control_noop")
        pending = {
            "tls_id": tls_id,
            "program_id": program_id,
            "requested_at_sim_time": float(
                (self._last_snapshot or {}).get("sim_time", 0.0)
            ),
            "due_condition": "verified_cycle_transition",
            "prior_program": runtime["current_program"],
            "prior_phase": runtime["current_phase"],
            "prior_state": runtime["current_state"],
            "last_observed_phase": runtime["current_phase"],
        }
        self._pending_signal_controls.append(pending)
        return {
            "_status": "pending",
            "tool": "set_signal_program",
            **pending,
            "complete_source_identity_sha256": (
                self._runtime_control_contract or {}
            ).get("complete_source_identity_sha256"),
            "sumo_state_mutated": False,
        }

    def _set_signal_phase_duration(self, args: dict[str, Any]) -> dict[str, Any]:
        tls_id = str(args.get("tls_id") or "")
        if self._sidecar is None:
            return self._control_error("sumo_sidecar_not_started")
        tls = ((self._runtime_control_contract or {}).get("tls") or {}).get(tls_id)
        if tls is None:
            return self._control_error("traffic_binding_tls_missing")
        before = self._sidecar.traffic_light_contract(tls_id)
        safe_program_ids = set(tls.get("safe_selectable_program_ids") or ())
        if str(before.get("current_program")) not in safe_program_ids:
            return self._control_error("traffic_current_program_not_safe")
        classification = classify_vehicle_movement_link_indices(
            before.get("controlled_links") or []
        )
        if classification["status"] != "passed":
            return self._control_error(
                "traffic_vehicle_movement_link_classification_invalid"
            )
        bounds = dict(before.get("current_phase_bounds") or {})
        try:
            phase = int(args.get("observed_phase", before["current_phase"]))
            program = str(
                args.get("observed_program", before["current_program"])
            )
            validate_phase_duration_request(
                observed_program=program,
                observed_phase=phase,
                runtime_program=before["current_program"],
                runtime_phase=before["current_phase"],
                runtime_state={
                    "state": before["current_state"],
                    "min_duration": bounds.get("min_duration"),
                    "max_duration": bounds.get("max_duration"),
                    "controlled_lanes": before["controlled_lanes"],
                    "controlled_links": before["controlled_links"],
                },
                requested_remaining_duration=float(
                    args.get("remaining_duration_seconds")
                ),
            )
        except (RuntimeControlContractError, TypeError, ValueError) as exc:
            code = (
                exc.code
                if isinstance(exc, RuntimeControlContractError)
                else "traffic_phase_duration_out_of_range"
            )
            return self._control_error(code)
        result = self._sidecar.set_traffic_light_phase_duration(
            tls_id, float(args["remaining_duration_seconds"])
        )
        after = self._sidecar.traffic_light_contract(tls_id)
        return {
            "_status": "ok",
            "tool": "set_signal_phase_duration",
            "before_runtime_state": before,
            "after_runtime_state": after,
            **result,
            "complete_source_identity_sha256": (
                self._runtime_control_contract or {}
            ).get("complete_source_identity_sha256"),
            "evidence_ids": [
                f"sumo:{(self._runtime_control_contract or {}).get('complete_source_identity_sha256')}:{tls_id}"
            ],
        }

    @staticmethod
    def _control_error(code: str) -> dict[str, Any]:
        return {
            "_status": "error",
            "error": code,
            "sumo_state_mutated": False,
        }

    def _materialize_pending_signal_controls(self) -> None:
        if self._sidecar is None or not self._pending_signal_controls:
            return
        remaining: list[dict[str, Any]] = []
        for pending in self._pending_signal_controls:
            tls_id = pending["tls_id"]
            current = self._sidecar.traffic_light_contract(tls_id)
            prior_phase = int(pending["last_observed_phase"])
            current_phase = int(current["current_phase"])
            is_cycle_transition = current_phase == 0 and prior_phase != 0
            pending["last_observed_phase"] = current_phase
            if not is_cycle_transition:
                remaining.append(pending)
                continue
            result = self._sidecar.set_traffic_light_program(
                tls_id, pending["program_id"]
            )
            pending.update(
                {
                    "applied_at_tick": self._tick,
                    "applied_at_sim_time": current["sim_time"],
                    "resulting_runtime_state": self._sidecar.traffic_light_contract(
                        tls_id
                    ),
                    "evidence_ids": [
                        "sumo:"
                        f"{(self._runtime_control_contract or {}).get('complete_source_identity_sha256')}:"
                        f"{tls_id}:program_materialized"
                    ],
                    **result,
                }
            )
            self._materialized_signal_controls.append(dict(pending))
        self._pending_signal_controls = remaining

    def _record_source_trace(
        self, *, tick: int, snapshot: dict[str, Any] | None
    ) -> None:
        contract = self._runtime_control_contract or {}
        if not contract or snapshot is None:
            return
        state_payload = {
            "sim_time": snapshot.get("sim_time"),
            "n_vehicles": snapshot.get("n_vehicles"),
            "arrived": snapshot.get("arrived"),
            "departed": snapshot.get("departed"),
            "interval_arrived": snapshot.get("interval_arrived"),
            "interval_departed": snapshot.get("interval_departed"),
        }
        digest = json.dumps(
            state_payload, sort_keys=True, separators=(",", ":")
        ).encode()
        self._source_trace_rows.append(
            {
                "tick": int(tick),
                "sim_time": snapshot.get("sim_time"),
                "state_digest": hashlib.sha256(digest).hexdigest(),
                "derived_state": state_payload,
            }
        )

    def protocol21_source_trace(self) -> dict[str, Any]:
        contract = self._runtime_control_contract or {}
        assets = contract.get("source_assets") or {}
        opened: list[dict[str, Any]] = []
        for key in ("sumocfg", "network"):
            row = assets.get(key)
            if isinstance(row, dict):
                opened.append(row)
        for key in ("route_files", "additional_files", "recursive_inputs"):
            opened.extend(
                row for row in assets.get(key) or [] if isinstance(row, dict)
            )
        trace = list(self._source_trace_rows)
        state_effect_observed = (
            len({row["state_digest"] for row in trace}) > 1
        )
        trace_ready = len(trace) >= 2 and state_effect_observed
        trace_semantic_payload = {
            "complete_source_identity_sha256": contract.get(
                "complete_source_identity_sha256"
            ),
            "runtime_trace": trace,
        }
        trace_semantic_digest = hashlib.sha256(
            json.dumps(
                trace_semantic_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        return {
            "status": (
                "passed"
                if trace_ready
                else "held"
            ),
            "proof_kind": "direct_runtime_files",
            "complete_source_identity_sha256": contract.get(
                "complete_source_identity_sha256"
            ),
            "opened_source_paths": [row.get("path") for row in opened],
            "opened_source_sha256": {
                str(row.get("path")): row.get("sha256") for row in opened
            },
            "runtime_opened_assets": opened,
            "consumed_channels": [
                "route_departures",
                "declared_runtime_perturbations",
                "tls_program_phase",
                "controlled_lane_queue_wait_throughput",
            ],
            "derived_backend_state_fields": [
                "corridor_queues",
                "signal_programs",
                "runtime_vehicle_ids",
                "travel_time_cost",
            ],
            "consumption_ticks": [row["tick"] for row in trace],
            "state_effect_observed": state_effect_observed,
            "source_state_effect_observed": state_effect_observed,
            "runtime_departure_counts": [
                {
                    "tick": row["tick"],
                    "departed": row["derived_state"].get(
                        "interval_departed",
                        row["derived_state"].get("departed"),
                    ),
                }
                for row in trace
            ],
            "initial_state_digest": trace[0]["state_digest"] if trace else None,
            "post_warmup_state_digest": (
                trace[1]["state_digest"] if len(trace) > 1 else None
            ),
            "post_source_change_state_digest": (
                trace[-1]["state_digest"] if len(trace) > 1 else None
            ),
            "post_source_state_digests": [
                row["state_digest"] for row in trace[1:]
            ],
            "trace_semantic_digest": trace_semantic_digest,
            # Determinism requires a second replay comparison.  This
            # single-episode trace proves only material source evolution.
            "trace_materiality_ready": trace_ready,
            "deterministic_source_trace": None,
            "subsequent_state_digests": [
                row["state_digest"] for row in trace[1:]
            ],
            "runtime_trace": trace,
            "evidence_from_scenario_config_only": False,
            "blockers": [] if trace_ready else ["runtime_trace_too_short"],
        }

    def _change_signal_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        corridor = str(args.get("corridor") or args.get("corridor_id") or "")
        if not corridor:
            return {
                "_status": "error",
                "error": "missing_corridor",
                "tool": "change_signal_plan",
                "sumo_state_mutated": False,
            }
        tls_id = self._corridor_tls_map.get(corridor)
        if not tls_id:
            return {
                "_status": "error",
                "error": "missing_sumo_tls_mapping",
                "tool": "change_signal_plan",
                "corridor": corridor,
                "known_corridors": sorted(self._corridor_tls_map),
                "sumo_state_mutated": False,
            }
        program = str(args.get("program") or args.get("program_id") or "default")
        # Resolve the benchmark program slot to a real present program id:
        # per-corridor net-derived map (preferred) → global map → identity.
        corridor_progs = self._corridor_program_map.get(corridor)
        if corridor_progs and program in corridor_progs:
            sumo_program = corridor_progs[program]
        else:
            sumo_program = self._signal_program_map.get(program, program)
        if self._sidecar is None:
            return {
                "_status": "error",
                "error": "sumo_sidecar_not_started",
                "tool": "change_signal_plan",
                "corridor": corridor,
                "sumo_tls_id": tls_id,
                "sumo_state_mutated": False,
            }
        try:
            control_result = self._sidecar.set_traffic_light_program(
                tls_id, sumo_program
            )
        except Exception as exc:
            return {
                "_status": "error",
                "error": "sumo_signal_program_apply_failed",
                "tool": "change_signal_plan",
                "corridor": corridor,
                "sumo_tls_id": tls_id,
                "sumo_program_id": sumo_program,
                "exception": f"{type(exc).__name__}: {exc}",
                "sumo_state_mutated": False,
            }
        if control_result.get("sumo_program_readback_matches") is False:
            return {
                "_status": "error",
                "error": "sumo_signal_program_readback_mismatch",
                "tool": "change_signal_plan",
                "corridor": corridor,
                "program": program,
                **control_result,
                "sumo_state_mutated": False,
            }
        self._live_signal_state[corridor] = {
            "signal_program": program,
            "live_sumo_mapped": True,
            **control_result,
        }
        return {
            "_status": "ok",
            "tool": "change_signal_plan",
            "corridor": corridor,
            "program": program,
            **control_result,
            "stakeholder_class": _stakeholder_for_corridor_id(corridor),
        }

    def _extend_current_green_phase(self, args: dict[str, Any]) -> dict[str, Any]:
        corridor = str(args.get("corridor") or args.get("corridor_id") or "")
        if not corridor:
            return {
                "_status": "error",
                "error": "missing_corridor",
                "tool": "extend_current_green_phase",
                "sumo_state_mutated": False,
            }
        tls_id = self._corridor_tls_map.get(corridor)
        if not tls_id:
            return {
                "_status": "error",
                "error": "missing_sumo_tls_mapping",
                "tool": "extend_current_green_phase",
                "corridor": corridor,
                "known_corridors": sorted(self._corridor_tls_map),
                "sumo_state_mutated": False,
            }
        duration_s = float(args.get("duration_s") or 180.0)
        if self._sidecar is None:
            return {
                "_status": "error",
                "error": "sumo_sidecar_not_started",
                "tool": "extend_current_green_phase",
                "corridor": corridor,
                "sumo_tls_id": tls_id,
                "sumo_state_mutated": False,
            }
        try:
            control_result = self._sidecar.set_traffic_light_phase_duration(
                tls_id, duration_s
            )
        except Exception as exc:
            return {
                "_status": "error",
                "error": "sumo_phase_duration_apply_failed",
                "tool": "extend_current_green_phase",
                "corridor": corridor,
                "sumo_tls_id": tls_id,
                "duration_s": duration_s,
                "exception": f"{type(exc).__name__}: {exc}",
                "sumo_state_mutated": False,
            }
        self._live_signal_state[corridor] = {
            "signal_program": "phase_duration_override",
            "live_sumo_mapped": True,
            **control_result,
        }
        return {
            "_status": "ok",
            "tool": "extend_current_green_phase",
            "corridor": corridor,
            "duration_s": duration_s,
            **control_result,
            "stakeholder_class": _stakeholder_for_corridor_id(corridor),
        }

    def ground_truth_costs(self) -> dict[str, float]:
        travel = sum(r.travel_cost_this_tick for r in self._tick_records)
        return {
            "travel_time_cost": round(travel, 3),
            "shed_delay_cost": 0.0,
            "actuation_cost": 0.0,
            "mutual_aid_cost": 0.0,
        }

    def scoring_records(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for r in self._tick_records:
            corridor_count = max(1, len(self._corridor_tls_map))
            network_violation_fraction = (
                r.n_gridlocked + r.n_blocked_edges
            ) / (2.0 * corridor_count)
            rows.append(
                {
                    "tick": r.tick,
                    "aggregate_demand_mw": float(r.aggregate_offered),
                    "aggregate_generation_mw": float(r.aggregate_served),
                    "balance_error_mw": float(r.aggregate_queue),
                    "reserves_required_mw": float(r.reserves_required),
                    "reserves_procured_mw": float(r.reserves_procured),
                    "production_cost": float(r.travel_cost_this_tick),
                    "startup_cost": float(r.actuation_cost_this_tick),
                    "shed_penalty": float(r.shed_cost_this_tick),
                    "rho_max": float(r.rho_max),
                    "n_overloads": int(r.n_overloads),
                    "n_voltage_violations": int(r.n_gridlocked),
                    "n_disconnected_lines": int(r.n_blocked_edges),
                    "done": False,
                    "catastrophic_failure": False,
                    "safety_violation_severity": min(
                        1.0,
                        max(0.0, network_violation_fraction),
                    ),
                }
            )
        return rows

    def native_scoring_records(self) -> list[dict[str, Any]]:
        return [
            {
                "tick": r.tick,
                "aggregate_offered": r.aggregate_offered,
                "aggregate_served": r.aggregate_served,
                "aggregate_queue": r.aggregate_queue,
                "aggregate_delay_minutes": r.aggregate_delay_minutes,
                "travel_cost_this_tick": r.travel_cost_this_tick,
                "shed_cost_this_tick": r.shed_cost_this_tick,
            }
            for r in self._tick_records
        ]

    def per_corridor_delay_minutes(self) -> dict[str, float]:
        """Cumulative per-corridor delay-minutes from live SUMO lane metrics.

        Corridors with a live TLS binding report the integrated queue*tick_minutes
        delay aggregated over their controlled lanes; corridors without a binding
        (no controlled-lane mapping on this net) report 0.0 honestly rather than a
        fabricated value.
        """
        if self._seed_obj is None:
            return {}
        return {
            c.corridor_id: round(
                float(self._corridor_delay_minutes.get(c.corridor_id, 0.0)), 3
            )
            for c in self._seed_obj.corridors
        }

    def forecast_for(self, horizon: int) -> list[dict[str, Any]]:
        return [
            {
                "tick": self._tick + k,
                "predicted_offered_veh": None,
                "forecast_index": k,
                "reason": "live_sumo_forecast_not_implemented",
            }
            for k in range(int(horizon))
        ]

    def queue_mutual_aid_effect(self, *, due_tick: int, mw: float) -> dict[str, Any]:
        return {
            "_status": "error",
            "error": "live_sumo_delayed_incident_response_not_implemented",
            "due_tick": int(due_tick),
            "mw": float(mw),
            "sumo_state_mutated": False,
        }

    def close(self) -> None:
        if self._sidecar is not None:
            self._sidecar.close()
        self._sidecar = None
        self._transport = None

    def _tick_minutes(self) -> int:
        return int(self._seed_obj.tick_minutes if self._seed_obj is not None else 5)


def _resolve_required_path(raw_path: Any, label: str) -> Path:
    if not raw_path:
        raise FileNotFoundError(f"{label} path is required for live SUMO backend")
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"{label} path does not exist: {path}")
    return path


def _stakeholder_for_corridor_id(corridor_id: str) -> str:
    if corridor_id in {"cbd_ring", "hospital_access"}:
        return "emergency_services"
    if corridor_id == "industrial_freight":
        return "freight_operator"
    if corridor_id == "west_lowincome":
        return "transit_agency"
    return "commuter"
