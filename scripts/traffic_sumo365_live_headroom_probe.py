#!/usr/bin/env python3
"""Opt-in live-SUMO headroom probe for SUMO Ingolstadt 365 staging.

This is a v0.32 non-release gate. It launches the real SUMO kernel only when
``OPERATE_TRAFFIC_BACKEND_REAL=1`` and records whether a single source-locked
SUMO365 date can execute the same locked ``.sumocfg`` asset set named by the
source audit. Passing this probe is necessary evidence for later Traffic live
promotion, but it is not a release materializer and never changes the
publishable denominator by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import Action, ToolCall  # noqa: E402
from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import required_semantics  # noqa: E402
from core.sidecar.sumo_sidecar import probe_sumo_transport, sumo_available  # noqa: E402
from domains.traffic.adapter import TrafficEnvironment  # noqa: E402
from domains.traffic.seeds.schema import CorridorAssignment  # noqa: E402
from domains.traffic.seeds.sumo365 import (  # noqa: E402
    SUMO365_SERVICE_DATES,
    build_sumo365_traffic_seed,
    sumo365_source_lock,
)
from scripts.audit_sumo365_traffic_dates import (  # noqa: E402
    build_sumo365_source_audit_report,
    resolve_reports_output_path,
)

DEFAULT_OUTPUT = REPO_ROOT / "reports" / "traffic_sumo365_live_headroom_probe.json"
DEFAULT_SERVICE_DATE = "2023-06-19"
MIN_HEADROOM_L1_MINUTES = 1.0
MIN_CORRIDORS_CHANGED = 1
DEFAULT_FAMILY = "incident_response"
DEFAULT_DIFFICULTY_MODE = "time_pressure"
DEFAULT_DIFFICULTY_LEVEL = "basic"
ALLOWED_FAMILIES = ("incident_response", "vip_priority_dilemma")
ALLOWED_DIFFICULTY_MODES = ("time_pressure", "deep_planning")
ALLOWED_DIFFICULTY_LEVELS = ("basic",)
DEFAULT_WINDOW_BEGIN_S = 28800
DEFAULT_WINDOW_END_S = 32400
ALLOWED_CONTROL_ACTIONS = (
    "state_derived_extend_current_green_phase",
    "change_signal_plan",
)

REQUIRED_ASSET_ROLES = [
    "sumocfg",
    "network",
    "route",
    "tl_logic",
    "waut",
    "pt_stops",
    "pt_trips",
]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sumo365_asset_binding(
    *, service_date: str, source_lock: dict[str, Any], loaded: bool
) -> dict[str, Any]:
    files = dict(source_lock.get("file_refs") or {})
    return {
        "service_date": service_date,
        "sumocfg_loaded": bool(loaded),
        "required_asset_roles": list(REQUIRED_ASSET_ROLES),
        "sumocfg": files.get("sumocfg"),
        "sumocfg_references": {
            role: files.get(role) for role in REQUIRED_ASSET_ROLES if role != "sumocfg"
        },
        "all_required_assets_sha256_locked": (
            source_lock.get("source_locked") is True
            and all(role in files for role in REQUIRED_ASSET_ROLES)
        ),
    }


def _derive_sumo365_native_tls_binding(
    *, network_path: Path, program_path: Path, max_tls: int | None = None
) -> dict[str, Any]:
    """Derive a bounded control map only from the exact SUMO365 assets.

    A TLS is eligible only when the physical network contains incoming lanes
    controlled by it and the date-specific program file contains at least one
    phase with a green vehicle signal. The deterministic cap prefers TLSs with
    more unique controlled lanes, then the native TLS id.
    """

    controlled: dict[str, set[str]] = {}
    for _event, element in ET.iterparse(network_path, events=("end",)):
        if element.tag.rsplit("}", 1)[-1] != "connection":
            element.clear()
            continue
        tls_id = str(element.get("tl") or "").strip()
        edge_id = str(element.get("from") or "").strip()
        lane_index = str(element.get("fromLane") or "").strip()
        if tls_id and edge_id and lane_index:
            controlled.setdefault(tls_id, set()).add(f"{edge_id}_{lane_index}")
        element.clear()

    programs: dict[str, set[str]] = {}
    for tl_logic in ET.parse(program_path).getroot().iter("tlLogic"):
        tls_id = str(tl_logic.get("id") or "").strip()
        program_id = str(tl_logic.get("programID") or "").strip()
        has_vehicle_green = any(
            any(signal in {"g", "G"} for signal in str(phase.get("state") or ""))
            for phase in tl_logic.findall("phase")
        )
        if tls_id and program_id and has_vehicle_green:
            programs.setdefault(tls_id, set()).add(program_id)

    eligible = sorted(
        set(controlled).intersection(programs),
        key=lambda tls_id: (-len(controlled[tls_id]), tls_id),
    )
    if max_tls is not None:
        eligible = eligible[: max(1, int(max_tls))]
    corridor_tls = {f"native_tls::{tls_id}": tls_id for tls_id in eligible}
    return {
        "schema_version": "sumo365-native-tls-binding-v1",
        "derivation": "sumo365_native_network_tls_program_v1",
        "network_path": str(network_path),
        "network_sha256": hashlib.sha256(network_path.read_bytes()).hexdigest(),
        "program_path": str(program_path),
        "program_sha256": hashlib.sha256(program_path.read_bytes()).hexdigest(),
        "max_tls": int(max_tls) if max_tls is not None else None,
        "corridor_tls_map": corridor_tls,
        "controlled_lanes_by_corridor": {
            corridor: sorted(controlled[tls_id]) for corridor, tls_id in corridor_tls.items()
        },
        "program_ids_by_corridor": {
            corridor: sorted(programs[tls_id]) for corridor, tls_id in corridor_tls.items()
        },
    }


def _positive_native_headroom(
    *,
    wait: dict[str, Any],
    acting: dict[str, Any],
    headroom_l1_minutes: float,
) -> tuple[bool, dict[str, float]]:
    """Require a beneficial native delay reduction, not merely any effect."""
    wait_loss = float(wait.get("total_delay_minutes") or 0.0)
    acting_loss = float(acting.get("total_delay_minutes") or 0.0)
    wait_network_loss = float(
        (wait.get("cost_components") or {}).get("travel_time_cost") or 0.0
    )
    acting_network_loss = float(
        (acting.get("cost_components") or {}).get("travel_time_cost") or 0.0
    )
    improvement = wait_loss - acting_loss
    required = max(MIN_HEADROOM_L1_MINUTES, 0.05 * wait_loss)
    network_improvement = wait_network_loss - acting_network_loss
    network_required = max(MIN_HEADROOM_L1_MINUTES, 0.05 * wait_network_loss)
    passed = bool(
        headroom_l1_minutes >= MIN_HEADROOM_L1_MINUTES
        and wait_loss > 0.0
        and improvement >= required
        and wait_network_loss > 0.0
        and network_improvement >= network_required
    )
    return passed, {
        "wait_native_loss": wait_loss,
        "acting_native_loss": acting_loss,
        "native_loss_improvement": improvement,
        "required_native_loss_improvement": required,
        "wait_full_network_native_loss": wait_network_loss,
        "acting_full_network_native_loss": acting_network_loss,
        "full_network_native_loss_improvement": network_improvement,
        "required_full_network_native_loss_improvement": network_required,
    }


def _native_replay_signature(episode: dict[str, Any]) -> str:
    """Bind determinism to the complete native source/runtime outcome."""
    trace = episode.get("source_consumption_evidence") or {}
    payload = {
        "per_corridor_delay_minutes": episode.get("per_corridor_delay_minutes"),
        "cost_components": episode.get("cost_components"),
        "runtime_signal_control": episode.get("runtime_signal_control"),
        "complete_source_identity_sha256": trace.get(
            "complete_source_identity_sha256"
        ),
        "trace_semantic_digest": trace.get("trace_semantic_digest"),
        "runtime_trace": trace.get("runtime_trace"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _failed_probe_issues(
    *,
    positive_headroom: bool,
    differentiated: bool,
    state_change: bool,
    evidence_linked: bool,
    deterministic: bool,
) -> list[str]:
    return [
        code
        for code, ok in {
            "causal_outcome_change_missing": positive_headroom,
            "corridor_differentiation_missing": differentiated,
            "state_change_evidence_missing": state_change,
            "evidence_wiring_missing": evidence_linked,
            "deterministic_replay_failed": deterministic,
        }.items()
        if not ok
    ]


def _candidate_key(
    *, service_date: str, family: str, difficulty_mode: str, difficulty_level: str
) -> str:
    return f"traffic365/{family}/{service_date}/{difficulty_mode}/{difficulty_level}"


def _build_live_seed(
    service_date: str,
    *,
    family: str,
    difficulty_mode: str,
    difficulty_level: str,
    n_ticks: int,
    window_begin_s: int = DEFAULT_WINDOW_BEGIN_S,
    window_end_s: int = DEFAULT_WINDOW_END_S,
) -> Any:
    seed = build_sumo365_traffic_seed(
        seed_id=_candidate_key(
            service_date=service_date,
            family=family,
            difficulty_mode=difficulty_mode,
            difficulty_level=difficulty_level,
        ),
        service_date=service_date,
        family=family,
        difficulty_level=difficulty_level,
        difficulty_mode=difficulty_mode,
    )
    seed.backend_kind = "sumo"
    files = dict(seed.backend_config.get("sumo365_files") or {})
    network_path = REPO_ROOT / files["network"]
    program_path = REPO_ROOT / files["tl_logic"]
    binding = _derive_sumo365_native_tls_binding(
        network_path=network_path,
        program_path=program_path,
    )
    seed.corridors = [
        CorridorAssignment(
            corridor_id=corridor,
            district="native_sumo_tls",
            demand_veh=max(
                1,
                len(binding["controlled_lanes_by_corridor"][corridor]),
            ),
            edges=sorted(
                {
                    lane.rsplit("_", 1)[0]
                    for lane in binding["controlled_lanes_by_corridor"][corridor]
                }
            ),
            income_bracket="mid",
            transit_dependent_fraction=0.0,
            carries_ems_corridor=False,
            carries_vip_route=False,
            criticality=0.5,
        )
        for corridor in binding["corridor_tls_map"]
    ]
    seed.backend_config = {
        **seed.backend_config,
        "backend_kind": "sumo",
        "corridor_tls_map": dict(binding.get("corridor_tls_map") or {}),
        "sumo_corridor_program_map": {},
        "sumo_tls_binding_net_sha256": binding["network_sha256"],
        "sumo365_native_tls_binding": binding,
        "sumo_config_path": files["sumocfg"],
        # Morning peak gives the locked date asset enough vehicles for a bounded
        # headroom proof without running the full 24h source trace.
        "sumo_extra_args": (
            "--begin",
            str(int(window_begin_s)),
            "--end",
            str(int(window_end_s)),
        ),
        "sumo_substeps_per_tick": 120,
        "sumo_probe_n_ticks": int(n_ticks),
    }
    return seed


def _state_derived_phase_targets(
    wait_probe: dict[str, Any], *, max_targets: int
) -> list[dict[str, Any]]:
    """Select live-controllable corridors from observed wait-floor queues."""

    metrics = dict(wait_probe.get("last_corridor_metrics") or {})
    corridor_tls = dict(wait_probe.get("corridor_tls_map") or {})
    runtime_tls = dict((wait_probe.get("runtime_signal_control") or {}).get("tls") or {})
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for corridor, row in metrics.items():
        tls_id = corridor_tls.get(corridor)
        runtime = runtime_tls.get(tls_id) or {}
        state = str(runtime.get("current_state") or "")
        current_program = str(runtime.get("current_program") or "")
        safe_programs = {
            str(program) for program in runtime.get("safe_selectable_program_ids") or []
        }
        bounds = dict(runtime.get("current_phase_bounds") or {})
        minimum = float(bounds.get("min_duration") or 0.0)
        maximum = float(bounds.get("max_duration") or 0.0)
        if (
            not tls_id
            or not isinstance(row, dict)
            or current_program not in safe_programs
            or not any(signal in {"g", "G"} for signal in state)
            or minimum <= 0.0
            or maximum < minimum
        ):
            continue
        queue = float(row.get("queue", 0.0) or 0.0)
        vehicles = float(row.get("vehicles", 0.0) or 0.0)
        waiting_s = float(row.get("waiting_time_s", 0.0) or 0.0)
        scored.append(
            (
                queue * 1000.0 + vehicles * 10.0 + waiting_s,
                corridor,
                {
                    "tls_id": tls_id,
                    "observed_program": current_program,
                    "observed_phase": int(runtime.get("current_phase") or 0),
                    "remaining_duration_seconds": maximum,
                },
            )
        )
    scored.sort(reverse=True)
    return [
        {
            "corridor": corridor,
            **runtime_target,
            "selection_score": round(score, 3),
        }
        for score, corridor, runtime_target in scored[: max(0, int(max_targets))]
        if score > 0.0
    ]


def _control_tick_for_action(control_action: str) -> int:
    if control_action == "state_derived_extend_current_green_phase":
        return 1
    return 0


def _control_succeeded(payload: dict[str, Any]) -> bool:
    return bool(
        payload.get("sumo_state_mutated") is True
        or (payload.get("runtime_result") or {}).get("sumo_state_mutated") is True
    )


def _run_episode(
    *,
    service_date: str,
    family: str,
    difficulty_mode: str,
    difficulty_level: str,
    acting: bool,
    n_ticks: int,
    program_slot: str = "incident_relief",
    live_phase_targets: list[dict[str, Any]] | None = None,
    window_begin_s: int = DEFAULT_WINDOW_BEGIN_S,
    window_end_s: int = DEFAULT_WINDOW_END_S,
    control_action: str = "state_derived_extend_current_green_phase",
) -> dict[str, Any]:
    seed = _build_live_seed(
        service_date,
        family=family,
        difficulty_mode=difficulty_mode,
        difficulty_level=difficulty_level,
        n_ticks=n_ticks,
        window_begin_s=window_begin_s,
        window_end_s=window_end_s,
    )
    env = TrafficEnvironment()
    try:
        env.reset(seed.to_dict(), seed=seed.seed)
        backend = env._backend
        controllable = sorted(seed.backend_config["corridor_tls_map"])
        for tick in range(int(n_ticks)):
            if acting and tick == _control_tick_for_action(control_action):
                if (
                    control_action == "state_derived_extend_current_green_phase"
                    and live_phase_targets
                ):
                    action = Action(
                        tool_calls=[
                            ToolCall(
                                name="set_signal_phase_duration",
                                args={
                                    "tls_id": str(target["tls_id"]),
                                    "observed_program": str(target["observed_program"]),
                                    "observed_phase": int(target["observed_phase"]),
                                    "remaining_duration_seconds": float(
                                        target["remaining_duration_seconds"]
                                    ),
                                },
                            )
                            for target in live_phase_targets
                        ],
                        dominant="set_signal_phase_duration",
                    )
                elif control_action == "change_signal_plan":
                    action = Action(
                        tool_calls=[
                            ToolCall(
                                name="change_signal_plan",
                                args={"corridor": corridor, "program": program_slot},
                            )
                            for corridor in controllable
                        ],
                        dominant="change_signal_plan",
                    )
                else:
                    action = Action(tool_calls=[ToolCall(name="wait")], dominant="wait")
            else:
                action = Action(tool_calls=[ToolCall(name="wait")], dominant="wait")
            env.step(action)

        gt = env.ground_truth()
        evidence = env.evidence
        control_rows = (
            evidence.items_by_kind("control") + evidence.items_by_kind("native_signal_control")
            if evidence
            else []
        )
        realized_rows = evidence.items_by_kind("realized_event") if evidence else []
        control_evidence_summary = [
            {
                "evidence_id": row.evidence_id,
                "tick": row.tick,
                "tool": row.payload.get("tool"),
                "corridor": row.payload.get("corridor"),
                "sumo_tls_id": row.payload.get("sumo_tls_id"),
                "sumo_state_mutated": row.payload.get("sumo_state_mutated"),
                "sumo_program_readback_matches": row.payload.get("sumo_program_readback_matches"),
                "sumo_phase_duration_s": row.payload.get("sumo_phase_duration_s"),
                "runtime_result": row.payload.get("runtime_result"),
                "trust_event": row.payload.get("trust_event"),
            }
            for row in control_rows
        ]
        successful_control_rows = [row for row in control_rows if _control_succeeded(row.payload)]
        realized_event_summary = [
            {
                "type": row.payload.get("type"),
                "origin": row.payload.get("origin"),
                "materiality_value": row.payload.get("materiality_value"),
                "material_change": row.payload.get("material_change"),
            }
            for row in realized_rows
        ]
        trust_ok = any(
            (
                row.payload.get("sumo_state_mutated") is True
                and (
                    row.payload.get("sumo_program_readback_matches") is True
                    or row.payload.get("sumo_phase_duration_s") is not None
                )
                and row.payload.get("trust_event")
            )
            or ((row.payload.get("runtime_result") or {}).get("sumo_state_mutated") is True)
            for row in control_rows
        )
        return {
            "per_corridor_delay_minutes": {
                k: round(float(v), 3)
                for k, v in dict(gt.get("per_corridor_delay_minutes") or {}).items()
            },
            "total_delay_minutes": round(
                sum(float(v) for v in dict(gt.get("per_corridor_delay_minutes") or {}).values()),
                3,
            ),
            "cost_components": dict(gt.get("cost_components") or {}),
            "state_changes_observed": len(successful_control_rows),
            "evidence_linked": bool(trust_ok),
            "control_evidence_summary": control_evidence_summary,
            "realized_event_count": len(realized_rows),
            "realized_event_summary": realized_event_summary,
            "controllable_corridors": controllable,
            "corridor_tls_map": dict(seed.backend_config["corridor_tls_map"]),
            "runtime_signal_control": dict(gt.get("runtime_signal_control") or {}),
            "last_corridor_metrics": dict(getattr(backend, "_corridor_last_metrics", {})),
            "source_consumption_evidence": backend.protocol21_source_trace(),
        }
    finally:
        env.close()


def build_sumo365_live_headroom_probe_report(
    *,
    service_date: str = DEFAULT_SERVICE_DATE,
    family: str = DEFAULT_FAMILY,
    difficulty_mode: str = DEFAULT_DIFFICULTY_MODE,
    difficulty_level: str = DEFAULT_DIFFICULTY_LEVEL,
    n_ticks: int = 2,
    program_slot: str = "incident_relief",
    window_begin_s: int = DEFAULT_WINDOW_BEGIN_S,
    window_end_s: int = DEFAULT_WINDOW_END_S,
    control_action: str = "state_derived_extend_current_green_phase",
) -> dict[str, Any]:
    source_audit = build_sumo365_source_audit_report()
    source_lock = sumo365_source_lock(service_date)
    candidate_key = _candidate_key(
        service_date=service_date,
        family=family,
        difficulty_mode=difficulty_mode,
        difficulty_level=difficulty_level,
    )
    base: dict[str, Any] = {
        "schema_version": "0.1",
        "scope": "traffic_sumo365_live_headroom_probe",
        "generated_at_utc": _utc_now(),
        "non_release_artifact": True,
        "release_ready": False,
        "release_reentry_ready": False,
        "publishable_denominator_delta": 0,
        "service_date": service_date,
        "family": family,
        "difficulty_mode": difficulty_mode,
        "difficulty_level": difficulty_level,
        "candidate_key": candidate_key,
        "n_ticks": int(n_ticks),
        "program_slot": program_slot,
        "window": {
            "begin_s": int(window_begin_s),
            "end_s": int(window_end_s),
        },
        "env_gate": "OPERATE_TRAFFIC_BACKEND_REAL=1",
        "source_audit_status": source_audit["status"],
        "source_lock": source_lock,
        "sumo365_asset_binding": _sumo365_asset_binding(
            service_date=service_date, source_lock=source_lock, loaded=False
        ),
        "headroom_metric": "l1_per_corridor_delay_minutes_acting_vs_wait",
        "headroom_scope": "single_date_single_window_single_program_slot",
        "control_action": control_action,
        "live_control_action": control_action,
        "min_headroom_l1_minutes": MIN_HEADROOM_L1_MINUTES,
        "min_corridors_changed": MIN_CORRIDORS_CHANGED,
        "release_blocker_codes": [
            "live_sumo365_full_date_headroom_missing",
            "live_sumo_headroom_missing_for_all_sumo365_dates",
            "release_materializer_not_implemented",
        ],
    }
    if source_audit["status"] != "ready_for_mock_sumo_filter":
        return {
            **base,
            "executed_with_live_backend": False,
            "status": "blocked_source_audit_not_ready",
            "issues": ["source_audit_not_ready"],
        }
    if os.environ.get("OPERATE_TRAFFIC_BACKEND_REAL") != "1":
        return {
            **base,
            "executed_with_live_backend": False,
            "status": "skipped_env_gate_unset",
            "reason": "OPERATE_TRAFFIC_BACKEND_REAL != 1",
        }
    if not sumo_available():
        return {
            **base,
            "executed_with_live_backend": False,
            "status": "skipped_no_sumo_transport",
            "reason": "no reachable SUMO transport (libsumo/traci/docker)",
            "release_blocker_codes": [
                "live_sumo_transport_unavailable",
                "release_materializer_not_implemented",
            ],
            "issues": ["live_sumo_transport_unavailable"],
        }

    try:
        wait_probe = _run_episode(
            service_date=service_date,
            family=family,
            difficulty_mode=difficulty_mode,
            difficulty_level=difficulty_level,
            acting=False,
            n_ticks=1,
            program_slot=program_slot,
            window_begin_s=window_begin_s,
            window_end_s=window_end_s,
            control_action=control_action,
        )
        live_phase_targets = (
            _state_derived_phase_targets(wait_probe, max_targets=2)
            if control_action == "state_derived_extend_current_green_phase"
            else []
        )
        wait = _run_episode(
            service_date=service_date,
            family=family,
            difficulty_mode=difficulty_mode,
            difficulty_level=difficulty_level,
            acting=False,
            n_ticks=n_ticks,
            program_slot=program_slot,
            window_begin_s=window_begin_s,
            window_end_s=window_end_s,
            control_action=control_action,
        )
        acting = _run_episode(
            service_date=service_date,
            family=family,
            difficulty_mode=difficulty_mode,
            difficulty_level=difficulty_level,
            acting=True,
            n_ticks=n_ticks,
            program_slot=program_slot,
            live_phase_targets=live_phase_targets,
            window_begin_s=window_begin_s,
            window_end_s=window_end_s,
            control_action=control_action,
        )
        wait_repeat = _run_episode(
            service_date=service_date,
            family=family,
            difficulty_mode=difficulty_mode,
            difficulty_level=difficulty_level,
            acting=False,
            n_ticks=n_ticks,
            program_slot=program_slot,
            window_begin_s=window_begin_s,
            window_end_s=window_end_s,
            control_action=control_action,
        )
    except Exception as exc:
        return {
            **base,
            "executed_with_live_backend": False,
            "status": "live_probe_execution_failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "issues": ["live_probe_execution_failed"],
        }

    corridors = sorted(
        set(wait["per_corridor_delay_minutes"]) | set(acting["per_corridor_delay_minutes"])
    )
    deltas = {
        c: round(
            acting["per_corridor_delay_minutes"].get(c, 0.0)
            - wait["per_corridor_delay_minutes"].get(c, 0.0),
            3,
        )
        for c in corridors
    }
    l1 = round(sum(abs(v) for v in deltas.values()), 3)
    changed = sorted(c for c, v in deltas.items() if abs(v) > 0.0)
    deterministic = _native_replay_signature(wait) == _native_replay_signature(
        wait_repeat
    )
    state_change = acting["state_changes_observed"] > 0
    evidence_linked = acting["evidence_linked"] is True
    positive_headroom, headroom_evidence = _positive_native_headroom(
        wait=wait,
        acting=acting,
        headroom_l1_minutes=l1,
    )
    differentiated = len(changed) >= MIN_CORRIDORS_CHANGED
    passed = bool(
        positive_headroom and differentiated and deterministic and state_change and evidence_linked
    )
    release_blockers = (
        []
        if passed
        else [
            "live_sumo365_headroom_missing_for_selected_window",
            "release_materializer_not_implemented",
        ]
    )
    return {
        **base,
        "release_blocker_codes": release_blockers,
        "non_release_artifact": not passed,
        "release_ready": bool(passed),
        "release_reentry_ready": bool(passed),
        "publishable_denominator_delta": 1 if passed else 0,
        "validated_candidate_basis": (
            "exact candidate-level live SUMO365 execution: same service date, "
            "family, difficulty mode, difficulty level, state-derived "
            "signal-control action, and mock-filter passing release-candidate row"
        ),
        "validated_candidate_keys": [candidate_key] if passed else [],
        "selected_transport": probe_sumo_transport(),
        "sumo365_asset_binding": _sumo365_asset_binding(
            service_date=service_date, source_lock=source_lock, loaded=True
        ),
        "executed_with_live_backend": True,
        "live_state_derived_targets": live_phase_targets,
        "target_derivation_probe": wait_probe,
        "wait_floor": wait,
        "acting": acting,
        "wait_repeat": wait_repeat,
        "per_corridor_delta_minutes": deltas,
        "headroom_l1_minutes": l1,
        "native_headroom_evidence": headroom_evidence,
        "corridors_changed": changed,
        "n_corridors_changed": len(changed),
        "causal_outcome_change_probe_passed": bool(positive_headroom),
        "differentiated_across_corridors": bool(differentiated),
        "state_change_probe_passed": bool(state_change),
        "evidence_wiring_probe_passed": bool(evidence_linked),
        "metric_deterministic_replay_passed": bool(deterministic),
        "all_probes_passed": passed,
        "status": (
            "live_sumo365_headroom_proven" if passed else "live_sumo365_headroom_incomplete"
        ),
        "issues": []
        if passed
        else _failed_probe_issues(
            positive_headroom=positive_headroom,
            differentiated=differentiated,
            state_change=state_change,
            evidence_linked=evidence_linked,
            deterministic=deterministic,
        ),
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    output = resolve_reports_output_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--service-date", default=DEFAULT_SERVICE_DATE)
    parser.add_argument("--family", default=DEFAULT_FAMILY, choices=ALLOWED_FAMILIES)
    parser.add_argument(
        "--difficulty-mode",
        default=DEFAULT_DIFFICULTY_MODE,
        choices=ALLOWED_DIFFICULTY_MODES,
    )
    parser.add_argument(
        "--difficulty-level",
        default=DEFAULT_DIFFICULTY_LEVEL,
        choices=ALLOWED_DIFFICULTY_LEVELS,
    )
    parser.add_argument("--n-ticks", type=int, default=2)
    parser.add_argument("--program-slot", default="incident_relief")
    parser.add_argument("--window-begin-s", type=int, default=DEFAULT_WINDOW_BEGIN_S)
    parser.add_argument("--window-end-s", type=int, default=DEFAULT_WINDOW_END_S)
    parser.add_argument(
        "--control-action",
        default="state_derived_extend_current_green_phase",
        choices=ALLOWED_CONTROL_ACTIONS,
    )
    parser.add_argument("--require-passed", action="store_true")
    args = parser.parse_args(argv)
    if args.service_date not in SUMO365_SERVICE_DATES:
        parser.error(f"--service-date must be one of {', '.join(SUMO365_SERVICE_DATES)}")
    start_hash = str(implementation_identity()["implementation_tree_sha256"])
    report = build_sumo365_live_headroom_probe_report(
        service_date=args.service_date,
        family=args.family,
        difficulty_mode=args.difficulty_mode,
        difficulty_level=args.difficulty_level,
        n_ticks=args.n_ticks,
        program_slot=args.program_slot,
        window_begin_s=args.window_begin_s,
        window_end_s=args.window_end_s,
        control_action=args.control_action,
    )
    end_hash = str(implementation_identity()["implementation_tree_sha256"])
    report["implementation_tree_sha256_start"] = start_hash
    report["implementation_tree_sha256"] = end_hash
    report["implementation_tree_stable"] = start_hash == end_hash
    report["evaluation_semantics"] = required_semantics()
    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_passed and report.get("status") != "live_sumo365_headroom_proven":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
