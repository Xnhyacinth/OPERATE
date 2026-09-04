#!/usr/bin/env python3
"""Audit native SUMO source graphs and bounded live control evidence.

The audit is intentionally fail-closed.  Source identity is derived from the
same recursive ``sumocfg`` graph used by the live backend; a control is only
credited when the observed runtime program/phase, native TraCI mutation, and a
post-action queue/delay observation are all present.  No program or TLS id is
invented by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from domains.traffic.runtime_control_contract import (  # noqa: E402
    RuntimeControlContractError,
    require_compatible_binding,
    validate_phase_duration_request,
)
from domains.traffic.source_identity import (  # noqa: E402
    build_sumo_source_identity_payload,
    compute_sumo_source_identity,
    resolve_sumo_input_graph,
)

SCHEMA_VERSION = "1.0"


def validate_response_window(
    *,
    event_time: float,
    decision_times: Iterable[float],
    horizon_seconds: float,
) -> dict[str, Any]:
    """Require a decision strictly after an event and before terminal time."""
    event = float(event_time)
    horizon = float(horizon_seconds)
    later = sorted(
        float(value)
        for value in decision_times
        if event < float(value) < horizon
    )
    if not later:
        return {
            "status": "held",
            "reason_code": "post_change_decision_missing",
            "event_time": event,
            "horizon_seconds": horizon,
        }
    return {
        "status": "passed",
        "event_time": event,
        "next_response_time": later[0],
        "horizon_seconds": horizon,
    }


def _binding_audit(
    *,
    runtime_network_sha256: str,
    binding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not binding:
        return {
            "status": "held",
            "reason_code": "traffic_binding_unverified",
        }
    required = (
        "network_sha256",
        "declared_tls_id",
        "declared_program_ids",
        "runtime_programs_by_tls",
    )
    missing = [key for key in required if key not in binding]
    if missing:
        return {
            "status": "held",
            "reason_code": "traffic_binding_incomplete",
            "missing_fields": missing,
        }
    try:
        require_compatible_binding(
            binding_network_sha256=str(binding["network_sha256"]),
            runtime_network_sha256=str(runtime_network_sha256),
            declared_tls_id=str(binding["declared_tls_id"]),
            declared_program_ids=(
                str(value) for value in binding["declared_program_ids"]
            ),
            runtime_programs_by_tls={
                str(tls): [str(value) for value in programs]
                for tls, programs in dict(
                    binding["runtime_programs_by_tls"]
                ).items()
            },
        )
    except RuntimeControlContractError as exc:
        return {
            "status": "held",
            "reason_code": exc.code,
            "details": dict(exc.details),
        }
    return {
        "status": "passed",
        "network_sha256": str(runtime_network_sha256),
        "tls_id": str(binding["declared_tls_id"]),
        "program_ids": sorted(
            {str(value) for value in binding["declared_program_ids"]}
        ),
    }


def audit_sumocfg(
    sumocfg: Path,
    *,
    service_date: str,
    sumo_version: Any,
    transport: str = "traci_tcp",
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the canonical recursive source graph and binding disposition."""
    try:
        graph = resolve_sumo_input_graph(Path(sumocfg))
        payload = build_sumo_source_identity_payload(
            graph,
            service_date=service_date,
            sumo_version=sumo_version,
            transport=transport,
        )
        identity = compute_sumo_source_identity(payload)
    except Exception as exc:  # source audit must report, not hide, bad assets
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "reason_code": getattr(exc, "code", "traffic_source_identity_mismatch"),
            "error": str(exc),
            "source_graph": None,
            "source_identity": None,
            "binding": {"status": "blocked", "reason_code": "source_graph_invalid"},
            "traci_port": {"requested": None, "mode": "dynamic"},
        }

    binding_report = _binding_audit(
        runtime_network_sha256=str(graph["network"]["sha256"]),
        binding=binding,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if binding_report["status"] == "passed" else "held",
        "source_graph": graph,
        "source_identity": {
            "sha256": identity,
            "payload": payload,
        },
        "binding": binding_report,
        "traci_port": {"requested": None, "mode": "dynamic"},
    }


def _queue_delay_effect(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    fields = (
        "halting",
        "waiting_time_s",
        "vehicles",
        "occupancy",
        "queue",
        "delay",
        "delay_minutes",
    )
    deltas: dict[str, float] = {}
    for field in fields:
        if field not in before or field not in after:
            continue
        try:
            deltas[field] = float(after[field]) - float(before[field])
        except (TypeError, ValueError):
            continue
    observed = any(abs(delta) > 1e-12 for delta in deltas.values())
    return {"observed": observed, "deltas": deltas}


def audit_native_control_evidence(
    *,
    runtime_program_ids: Sequence[str],
    observed_program: str,
    observed_phase: int,
    before_runtime_state: Mapping[str, Any],
    runtime_mutation: Mapping[str, Any],
    after_runtime_state: Mapping[str, Any],
    queue_before: Mapping[str, Any],
    queue_after: Mapping[str, Any],
    state_digest_before: str,
    state_digest_after: str,
    event_time: float,
    decision_times: Iterable[float],
    horizon_seconds: float,
) -> dict[str, Any]:
    """Validate one native phase-duration mutation and its nonterminal effect."""
    reasons: list[str] = []
    program_id = str(observed_program)
    runtime_programs = {str(value) for value in runtime_program_ids}
    if program_id not in runtime_programs:
        reasons.append("traffic_binding_program_missing_on_tls")

    before = dict(before_runtime_state)
    bounds = dict(before.get("current_phase_bounds") or {})
    if "min_duration" not in before:
        before["min_duration"] = bounds.get("min_duration")
    if "max_duration" not in before:
        before["max_duration"] = bounds.get("max_duration")
    before.setdefault("state", before.get("current_state"))
    before.setdefault("current_program", program_id)
    before.setdefault("current_phase", observed_phase)
    requested = runtime_mutation.get("sumo_phase_duration_s")
    validation: dict[str, Any] | None = None
    if requested is None:
        reasons.append("traffic_phase_duration_missing")
    else:
        try:
            validation = validate_phase_duration_request(
                observed_program=program_id,
                observed_phase=int(observed_phase),
                runtime_program=str(before["current_program"]),
                runtime_phase=int(before["current_phase"]),
                runtime_state=before,
                requested_remaining_duration=float(requested),
            )
        except RuntimeControlContractError as exc:
            reasons.append(exc.code)

    queue_effect = _queue_delay_effect(queue_before, queue_after)
    response = validate_response_window(
        event_time=event_time,
        decision_times=decision_times,
        horizon_seconds=horizon_seconds,
    )
    if response["status"] != "passed":
        reasons.append(str(response["reason_code"]))
    if runtime_mutation.get("sumo_state_mutated") is not True:
        reasons.append("traffic_native_mutation_missing")
    if not queue_effect["observed"]:
        reasons.append("traffic_native_queue_delay_effect_missing")
    if not state_digest_before or not state_digest_after or state_digest_before == state_digest_after:
        reasons.append("traffic_native_state_effect_missing")

    state = str(before.get("state") or "")
    safe_clearance = bool(
        any(signal in {"g", "G"} for signal in state)
        and not any(signal in {"y", "Y"} for signal in state)
        and not set(state) <= {"r", "R"}
    )
    if not safe_clearance and "traffic_phase_not_green" not in reasons:
        reasons.append("traffic_phase_not_green")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not reasons else "held",
        "reason_codes": sorted(set(reasons)),
        "program_id": program_id,
        "runtime_program_ids": sorted(runtime_programs),
        "observed_phase": int(observed_phase),
        "validation": validation,
        "safe_clearance": safe_clearance,
        "runtime_mutation": dict(runtime_mutation),
        "before_runtime_state": before,
        "after_runtime_state": dict(after_runtime_state),
        "queue_delay_effect": queue_effect,
        "state_effect_observed": bool(
            state_digest_before
            and state_digest_after
            and state_digest_before != state_digest_after
        ),
        "response_window": response,
    }


def _native_probe(
    sumocfg: Path,
    *,
    service_date: str,
    steps: int,
) -> dict[str, Any]:
    """Run a small real TraCI probe using only runtime-enumerated controls."""
    from core.sidecar.sumo_sidecar import SumoSidecar

    graph = resolve_sumo_input_graph(sumocfg)
    routes = graph["route_files"]
    if not routes:
        return {"status": "held", "reason_code": "traffic_route_input_missing"}
    sidecar = SumoSidecar(
        str(graph["network"]["path"]),
        str(routes[0]["path"]),
        config_path=str(Path(sumocfg).resolve()),
        traci_port=None,
    )
    try:
        transport = sidecar.start()
        runtime_version = sidecar.runtime_version()
        candidates: list[tuple[str, dict[str, Any]]] = []
        for tls_id in sidecar.traffic_light_ids():
            runtime = sidecar.traffic_light_contract(tls_id)
            state = str(runtime.get("current_state") or "")
            bounds = runtime.get("current_phase_bounds") or {}
            if (
                runtime.get("current_program")
                and any(ch in {"g", "G"} for ch in state)
                and not any(ch in {"y", "Y"} for ch in state)
                and bounds.get("min_duration") is not None
                and bounds.get("max_duration") is not None
                and runtime.get("controlled_lanes")
            ):
                candidates.append((str(tls_id), runtime))
        if not candidates:
            return {
                "status": "held",
                "reason_code": "traffic_no_safe_runtime_phase",
                "transport": transport,
                "sumo_version": runtime_version,
                "traci_port": {"requested": None, "mode": "dynamic"},
            }
        tls_id, before = candidates[0]
        lanes = tuple(str(value) for value in before["controlled_lanes"])
        queue_before = sidecar.lane_group_metrics(lanes)
        simulation = getattr(getattr(sidecar, "_conn", None), "simulation", None)
        time_getter = getattr(simulation, "getTime", None)
        event_time = float(time_getter()) if callable(time_getter) else 0.0
        minimum = float((before.get("current_phase_bounds") or {})["min_duration"])
        maximum = float((before.get("current_phase_bounds") or {})["max_duration"])
        spent = float(before.get("spent_duration") or 0.0)
        remaining_min = max(0.0, minimum - spent)
        remaining_max = max(remaining_min, maximum - spent)
        requested = remaining_min + (remaining_max - remaining_min) * 0.75
        mutation = sidecar.set_traffic_light_phase_duration(tls_id, requested)
        state_before = hashlib.sha256(
            json.dumps({"runtime": before, "queue": queue_before}, sort_keys=True).encode()
        ).hexdigest()
        for _ in range(max(1, int(steps))):
            sidecar.simulation_step()
        after = sidecar.traffic_light_contract(tls_id)
        queue_after = sidecar.lane_group_metrics(lanes)
        response_time = (
            float(time_getter()) if callable(time_getter) else event_time + max(1, steps)
        )
        state_after = hashlib.sha256(
            json.dumps({"runtime": after, "queue": queue_after}, sort_keys=True).encode()
        ).hexdigest()
        return audit_native_control_evidence(
            runtime_program_ids=sorted((before.get("programs") or {}).keys()),
            observed_program=str(before["current_program"]),
            observed_phase=int(before["current_phase"]),
            before_runtime_state={
                **before,
                "min_duration": minimum,
                "max_duration": maximum,
            },
            runtime_mutation=mutation,
            after_runtime_state=after,
            queue_before=queue_before,
            queue_after=queue_after,
            state_digest_before=state_before,
            state_digest_after=state_after,
            event_time=event_time,
            decision_times=[response_time],
            horizon_seconds=response_time + 1.0,
        ) | {
            "transport": transport,
            "sumo_version": runtime_version,
            "binding": {
                "network_sha256": str(graph["network"]["sha256"]),
                "declared_tls_id": tls_id,
                "declared_program_ids": sorted((before.get("programs") or {}).keys()),
                "runtime_programs_by_tls": {
                    tls_id: sorted((before.get("programs") or {}).keys())
                },
            },
            "traci_port": {"requested": None, "mode": "dynamic"},
            "tls_id": tls_id,
            "service_date": service_date,
        }
    finally:
        sidecar.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sumocfg", type=Path, required=True)
    parser.add_argument("--service-date", required=True)
    parser.add_argument("--sumo-version", default="SUMO 1.20.0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-native-probe", action="store_true")
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    report = audit_sumocfg(
        args.sumocfg,
        service_date=args.service_date,
        sumo_version=args.sumo_version,
    )
    if args.run_native_probe and report["status"] != "blocked":
        native_control = _native_probe(
            args.sumocfg,
            service_date=args.service_date,
            steps=args.steps,
        )
        report["native_control"] = native_control
        if native_control.get("status") == "passed" and native_control.get(
            "binding"
        ):
            report = audit_sumocfg(
                args.sumocfg,
                service_date=args.service_date,
                sumo_version=native_control.get("sumo_version", args.sumo_version),
                transport="traci_tcp",
                binding=native_control["binding"],
            ) | {"native_control": native_control}
        else:
            report["status"] = "held"
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
