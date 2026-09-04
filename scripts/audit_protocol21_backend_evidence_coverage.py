#!/usr/bin/env python3
"""Inventory backend-native source/world evidence for a working set."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.pomdp import Action, ToolCall  # noqa: E402
from core.protocol21_evidence import report_rows, required_semantics  # noqa: E402
from domains.registry import (  # noqa: E402
    get_backend_capability,
    get_domain_spec,
    resolve_backend_source_contract_builder,
)

_REQUIRED_RUNTIME_FIELDS = (
    "opened_source_paths",
    "opened_source_sha256",
    "consumed_channels",
    "derived_backend_state_fields",
    "consumption_ticks",
    "state_effect_observed",
)
_REQUIRED_WORLD_FIELDS = (
    "exogenous_change_records",
    "material_exogenous_event_records",
    "post_change_decision_ticks",
    "event_to_action_edges",
    "adaptive_replanning_observed",
    "agent_action_backend_effect_observed",
)


class _ProbeTimeout(TimeoutError):
    pass


def _timeout_handler(_signum: int, _frame: object) -> None:
    raise _ProbeTimeout("runtime probe timed out")


def _load_scenario(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row.get("path") or ""))
    resolved = path if path.is_absolute() else REPO_ROOT / path
    loaded = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("scenario YAML is not a mapping")
    return loaded


def _traffic_phase_duration_probe_call(
    *,
    scenario: dict[str, Any],
    observation: dict[str, Any],
    tick: int,
) -> ToolCall | None:
    """Build one bounded native SUMO control call from current runtime state."""
    backend_config = scenario.get("backend_config") or {}
    if not isinstance(backend_config, dict):
        return None
    if backend_config.get("live_phase_control") is not True:
        return None
    requested = backend_config.get("reference_phase_duration_seconds")
    try:
        requested_seconds = float(requested)
    except (TypeError, ValueError):
        return None
    if requested_seconds <= 0:
        return None
    runtime_tls = (
        (observation.get("runtime_signal_control") or {}).get("tls") or {}
    )
    if not isinstance(runtime_tls, dict):
        return None
    candidates: list[tuple[str, dict[str, Any], int, float]] = []
    for tls_id, runtime in runtime_tls.items():
        if not isinstance(runtime, dict):
            continue
        state = str(runtime.get("current_state") or "")
        if (
            not any(signal in {"g", "G"} for signal in state)
            or any(signal in {"y", "Y"} for signal in state)
        ):
            continue
        bounds = runtime.get("current_phase_bounds") or {}
        if not isinstance(bounds, dict):
            continue
        try:
            minimum = float(bounds.get("min_duration"))
            maximum = float(bounds.get("max_duration"))
            phase = int(runtime.get("current_phase"))
        except (TypeError, ValueError):
            continue
        if minimum <= 0 or maximum < minimum:
            continue
        candidates.append(
            (
                str(tls_id),
                runtime,
                phase,
                max(minimum, min(maximum, requested_seconds)),
            )
        )
    if not candidates:
        return None
    tls_id, runtime, phase, duration = min(
        candidates, key=lambda item: item[0]
    )
    return ToolCall(
        name="set_signal_phase_duration",
        args={
            "tls_id": tls_id,
            "observed_program": str(runtime.get("current_program") or ""),
            "observed_phase": phase,
            "remaining_duration_seconds": duration,
        },
        idempotency_key=f"protocol21-runtime-probe-phase-{tls_id}-{tick}",
    )


def _acopf_reserve_probe_call(
    *,
    observation: dict[str, Any],
    tick: int,
) -> ToolCall | None:
    """Build a small reserve commitment from a visible native requirement."""
    entities = observation.get("entities") or {}
    totals = observation.get("totals") or {}
    if not isinstance(entities, dict) or not isinstance(totals, dict):
        return None
    reserve = entities.get("reserve_commitment")
    if not isinstance(reserve, dict) or reserve.get("kind") != "reserve_commitment":
        return None
    try:
        required_mw = float(totals.get("reserves_required_mw"))
    except (TypeError, ValueError):
        return None
    if required_mw <= 0.0:
        return None
    # The protocol exposes no reservation ceiling.  A bounded 0.1%-of-need
    # request (capped at 1 MW) is therefore a safe, state-derived probe that
    # cannot masquerade as an optimal controller or alter task difficulty.
    requested_mw = min(max(required_mw * 0.001, 0.001), 1.0)
    return ToolCall(
        name="commit_reserve",
        args={"mw": round(requested_mw, 6)},
        idempotency_key=f"protocol21-runtime-probe-reserve-{tick}",
    )


def _opendss_transformer_tap_probe_call(
    *,
    observation: dict[str, Any],
    tick: int,
) -> ToolCall | None:
    """Build one bounded transformer-tap probe from visible regulator state."""
    entities = observation.get("entities") or {}
    if not isinstance(entities, dict):
        return None
    candidates: list[tuple[int, int]] = []
    for regulator in entities.values():
        if (
            not isinstance(regulator, dict)
            or regulator.get("kind") != "voltage_regulator"
        ):
            continue
        try:
            trafo_id = int(regulator["trafo_id"])
            current_tap = int(regulator["tap_number"])
            tap_min = int(regulator["tap_min"])
            tap_max = int(regulator["tap_max"])
        except (KeyError, TypeError, ValueError):
            continue
        if tap_min > tap_max:
            continue
        target_tap = (
            current_tap + 1
            if current_tap < tap_max
            else current_tap - 1
            if current_tap > tap_min
            else None
        )
        if target_tap is not None:
            candidates.append((trafo_id, target_tap))
    if not candidates:
        return None
    trafo_id, tap_pos = min(candidates)
    return ToolCall(
        name="set_transformer_tap",
        args={"trafo_id": trafo_id, "tap_pos": tap_pos},
        idempotency_key=(
            f"protocol21-runtime-probe-transformer-tap-{trafo_id}-{tick}"
        ),
    )


def _microgrid_battery_probe_call(
    *,
    observation: dict[str, Any],
    tick: int,
) -> ToolCall | None:
    """Build one bounded battery probe using only visible native state."""
    entities = observation.get("entities") or {}
    if not isinstance(entities, dict):
        return None
    candidates: list[tuple[str, float, float]] = []
    for battery_id, battery in entities.items():
        if not isinstance(battery, dict) or battery.get("kind") != "battery":
            continue
        try:
            soc_mwh = float(battery.get("soc_mwh"))
            max_discharge_mw = float(battery.get("max_discharge_mw"))
        except (TypeError, ValueError):
            continue
        if soc_mwh <= 1e-6 or max_discharge_mw <= 1e-6:
            continue
        candidates.append((str(battery_id), soc_mwh, max_discharge_mw))
    if not candidates:
        return None
    battery_id, soc_mwh, max_discharge_mw = min(candidates, key=lambda item: item[0])
    discharge_mw = min(max_discharge_mw * 0.5, soc_mwh * 0.5)
    if discharge_mw <= 1e-6:
        return None
    return ToolCall(
        name="set_battery_dispatch",
        args={"battery_id": battery_id, "p_mw": -round(discharge_mw, 6)},
        idempotency_key=(
            f"protocol21-runtime-probe-battery-{battery_id}-{tick}"
        ),
    )


def _runtime_probe_cache_key(row: dict[str, Any]) -> str:
    scenario_path = Path(str(row.get("path") or ""))
    scenario_path = (
        scenario_path
        if scenario_path.is_absolute()
        else REPO_ROOT / scenario_path
    )
    payload = {
        "schema_version": "protocol21_backend_runtime_probe_cache_v2",
        "scenario_id": str(row.get("scenario_id") or ""),
        "scenario_signature": str(row.get("scenario_signature") or ""),
        "domain": str(row.get("domain") or ""),
        "backend_kind": str(row.get("backend_kind") or ""),
        "scenario_file_sha256": (
            hashlib.sha256(scenario_path.read_bytes()).hexdigest()
            if scenario_path.is_file()
            else None
        ),
        "audit_implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "implementation_tree_sha256": implementation_identity()[
            "implementation_tree_sha256"
        ],
        "evaluation_semantics": required_semantics(),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _probe_with_cache(
    row: dict[str, Any],
    *,
    timeout_seconds: float,
    cache_dir: Path | None,
    cache_counts: Counter[str],
) -> dict[str, Any]:
    key = _runtime_probe_cache_key(row)
    cache_path = cache_dir / f"{key}.json" if cache_dir is not None else None
    if cache_path is not None and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cached = {}
        if cached.get("cache_key") == key and isinstance(
            cached.get("result"), dict
        ):
            cache_counts["hits"] += 1
            return dict(cached["result"])
    cache_counts["misses"] += 1
    result = _probe_backend(row, timeout_seconds=timeout_seconds)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {"cache_key": key, "result": result},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(cache_path)
        cache_counts["writes"] += 1
    return result


def _probe_backend(
    row: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    previous_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        scenario = _load_scenario(row)
        capability = get_backend_capability(
            str(row.get("backend_kind") or "")
        )
        contract_builder = resolve_backend_source_contract_builder(
            capability
        )
        scenario["source_contract"] = contract_builder(
            scenario,
            REPO_ROOT,
        )
        replay_results = [
            _exercise_backend(row=row, scenario=scenario)
            for _ in range(2)
        ]
        evidence, world_evolution = replay_results[0]
        if not isinstance(evidence, dict):
            raise TypeError("source evidence is not a mapping")
        deterministic = all(
            replay_results[0][0].get(field)
            == replay_results[1][0].get(field)
            for field in (
                "trace_semantic_digest",
                "consumption_ticks",
                "post_source_state_digests",
            )
        )
        blockers = list(evidence.get("blockers") or [])
        trace_observed = evidence.get("runtime_trace_observed") is True
        missing = [
            field
            for field in _REQUIRED_RUNTIME_FIELDS
            if field not in evidence
            or evidence.get(field) in (None, "", [], {})
            and field != "state_effect_observed"
        ]
        world_missing = [
            field
            for field in _REQUIRED_WORLD_FIELDS
            if field not in world_evolution
            or world_evolution.get(field) in (None, "", [], {})
            and field
            not in {
                "adaptive_replanning_observed",
                "agent_action_backend_effect_observed",
            }
        ]
        source_passed = (
            evidence.get("status") == "passed"
            and trace_observed
            and not missing
            and evidence.get("state_effect_observed") is True
            and deterministic
        )
        material_records = list(
            world_evolution.get("material_exogenous_event_records") or []
        )
        world_passed = bool(
            source_passed
            and not world_missing
            and material_records
            and world_evolution.get("agent_action_backend_effect_observed")
        )
        passed = source_passed and world_passed
        if trace_observed:
            classification = "backend_specific_runtime_trace"
        elif "backend_runtime_source_trace_unimplemented" in blockers:
            classification = "runtime_trace_unimplemented"
        else:
            classification = "generic_runtime_trace_fallback"
        evidence_blockers = list(blockers)
        if source_passed and not material_records:
            evidence_blockers = ["material_exogenous_change_missing"]
        return {
            "probe_status": (
                "runtime_probe_passed"
                if passed
                else "runtime_probe_held"
            ),
            "source_probe_status": (
                "runtime_probe_passed"
                if source_passed
                else "runtime_probe_held"
            ),
            "world_probe_status": (
                "runtime_probe_passed"
                if world_passed
                else "runtime_probe_held"
            ),
            "source_trace_complete": source_passed,
            "world_release_eligible": world_passed,
            "agent_action_backend_effect_observed": bool(
                world_evolution.get("agent_action_backend_effect_observed")
            ),
            "deterministic_replay_verified": deterministic,
            "world_evolution_applicability": (
                "dynamic_native" if material_records else "static_planning"
            ),
            "fabricated_exogenous_events": False,
            "world_evolution_reason_code": (
                None
                if world_passed
                else (
                    "material_exogenous_change_missing"
                    if source_passed and not material_records
                    else "world_evolution_evidence_incomplete"
                )
            ),
            "disposition": (
                "runtime_evidence_ready"
                if passed
                else (
                    "source_trace_ready_world_evolution_data_blocked"
                    if source_passed
                    else "runtime_source_trace_blocked"
                )
            ),
            "runtime_trace_classification": classification,
            "source_evidence_missing_fields": missing,
            "world_evolution_missing_fields": world_missing,
            "evidence_blockers": evidence_blockers,
            "evidence_status": str(evidence.get("status") or "unknown"),
            "runtime_validation_pending": not passed,
        }
    except _ProbeTimeout as exc:
        return {
            "probe_status": "runtime_probe_error",
            "source_probe_status": "runtime_probe_error",
            "world_probe_status": "runtime_probe_error",
            "source_trace_complete": False,
            "world_release_eligible": False,
            "agent_action_backend_effect_observed": False,
            "deterministic_replay_verified": False,
            "world_evolution_applicability": "unknown",
            "fabricated_exogenous_events": False,
            "world_evolution_reason_code": "runtime_probe_error",
            "disposition": "runtime_probe_error",
            "runtime_trace_classification": "runtime_trace_unimplemented",
            "source_evidence_missing_fields": list(_REQUIRED_RUNTIME_FIELDS),
            "world_evolution_missing_fields": list(_REQUIRED_WORLD_FIELDS),
            "evidence_blockers": ["runtime_probe_timeout"],
            "detail": str(exc),
            "runtime_validation_pending": True,
        }
    except Exception as exc:
        return {
            "probe_status": "runtime_probe_error",
            "source_probe_status": "runtime_probe_error",
            "world_probe_status": "runtime_probe_error",
            "source_trace_complete": False,
            "world_release_eligible": False,
            "agent_action_backend_effect_observed": False,
            "deterministic_replay_verified": False,
            "world_evolution_applicability": "unknown",
            "fabricated_exogenous_events": False,
            "world_evolution_reason_code": "runtime_probe_error",
            "disposition": "runtime_probe_error",
            "runtime_trace_classification": "runtime_trace_unimplemented",
            "source_evidence_missing_fields": list(_REQUIRED_RUNTIME_FIELDS),
            "world_evolution_missing_fields": list(_REQUIRED_WORLD_FIELDS),
            "evidence_blockers": ["runtime_probe_exception"],
            "detail": f"{type(exc).__name__}: {exc}",
            "runtime_validation_pending": True,
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _exercise_backend(
    *,
    row: dict[str, Any],
    scenario: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    env = get_domain_spec(str(row.get("domain") or "")).env_factory()()
    records: list[dict[str, Any]] = []
    agent_effect_seen = False
    acopf_control_submitted = False
    opendss_control_submitted = False
    microgrid_control_submitted = False
    try:
        observation = env.reset(scenario, int(scenario.get("seed") or 0))
        horizon = int(scenario.get("horizon_ticks") or 1)
        for tick in range(horizon):
            if (
                row.get("backend_kind") == "alibaba_trace_sim"
                and not agent_effect_seen
            ):
                policy = (
                    "shortest_job_first"
                    if tick % 2 == 0
                    else "least_gpu_first"
                )
                action = Action(
                    tool_calls=[
                        ToolCall(
                            name="set_queue_policy",
                            args={"policy": policy},
                            idempotency_key=(
                                f"protocol21-runtime-probe-{tick}"
                            ),
                        )
                    ]
                )
            elif (
                row.get("backend_kind") == "orgym_invmgmt"
                and tick == 0
            ):
                action = Action(
                    tool_calls=[
                        ToolCall(
                            name="place_replenishment_order",
                            args={"quantity": 1, "stage": 0},
                            idempotency_key="protocol21-runtime-probe",
                        )
                    ]
                )
            elif (
                row.get("backend_kind") == "pglib_uc_synthetic"
                and tick == 0
            ):
                action = Action(
                    tool_calls=[
                        ToolCall(
                            name="commit_reserve",
                            args={"mw": 1.0},
                            idempotency_key="protocol21-runtime-probe",
                        )
                    ]
                )
            elif (
                row.get("backend_kind") == "pandapower_acopf"
                and not agent_effect_seen
                and not acopf_control_submitted
            ):
                call = _acopf_reserve_probe_call(
                    observation=observation,
                    tick=tick,
                )
                action = Action(
                    tool_calls=(
                        [call] if call is not None else [ToolCall(name="wait")]
                    )
                )
                acopf_control_submitted = call is not None
            elif (
                row.get("backend_kind") == "opendss_fresh_feeders"
                and not agent_effect_seen
                and not opendss_control_submitted
                and any(
                    record.get("origin") == "source_schedule"
                    and record.get("material_exogenous") is True
                    for record in records
                )
            ):
                call = _opendss_transformer_tap_probe_call(
                    observation=observation,
                    tick=tick,
                )
                action = Action(
                    tool_calls=(
                        [call] if call is not None else [ToolCall(name="wait")]
                    )
                )
                opendss_control_submitted = call is not None
            elif (
                row.get("backend_kind") == "sumo"
                and not agent_effect_seen
            ):
                call = _traffic_phase_duration_probe_call(
                    scenario=scenario,
                    observation=observation,
                    tick=tick,
                )
                action = Action(
                    tool_calls=(
                        [call] if call is not None else [ToolCall(name="wait")]
                    )
                )
            elif (
                row.get("domain") == "microgrid"
                and not agent_effect_seen
                and not microgrid_control_submitted
            ):
                call = _microgrid_battery_probe_call(
                    observation=observation,
                    tick=tick,
                )
                action = Action(
                    tool_calls=(
                        [call] if call is not None else [ToolCall(name="wait")]
                    )
                )
                microgrid_control_submitted = call is not None
            else:
                action = Action(tool_calls=[ToolCall(name="wait")])
            result = env.step(action)
            observation = result.observation
            records.extend(
                result.info.extra.get("world_evolution_records") or []
            )
            agent_effect_seen = any(
                record.get("origin") == "agent_caused"
                for record in records
            )
            if not agent_effect_seen and microgrid_control_submitted:
                microgrid_results = [
                    tool_result
                    for tool_result in result.tool_results
                    if str(getattr(tool_result, "name", ""))
                    == "set_battery_dispatch"
                ]
                # Keep one call in flight while the protocol reports a
                # pending acknowledgement. Once it fails or resolves without
                # an observed native effect, allow a later native tick to
                # retry with fresh visible battery bounds.
                if microgrid_results and not any(
                    getattr(tool_result, "ok", False)
                    and isinstance(getattr(tool_result, "payload", None), dict)
                    and tool_result.payload.get("_status") == "pending"
                    for tool_result in microgrid_results
                ):
                    microgrid_control_submitted = False
            if not agent_effect_seen and opendss_control_submitted:
                opendss_results = [
                    tool_result
                    for tool_result in result.tool_results
                    if str(getattr(tool_result, "name", ""))
                    == "set_transformer_tap"
                ]
                if opendss_results and not any(
                    getattr(tool_result, "ok", False)
                    and isinstance(getattr(tool_result, "payload", None), dict)
                    and tool_result.payload.get("_status") == "pending"
                    for tool_result in opendss_results
                ):
                    opendss_control_submitted = False
            if result.done:
                break
        evidence = env.source_consumption_evidence(scenario=scenario)
    finally:
        env.close()

    source_records = [
        record
        for record in records
        if record.get("origin") == "source_schedule"
    ]
    declared_records = [
        record
        for record in records
        if record.get("origin") == "declared_perturbation"
    ]
    material_records = [
        record
        for record in records
        if record.get("material_exogenous") is True
    ]
    agent_records = [
        record
        for record in records
        if record.get("origin") == "agent_caused"
    ]
    edges = [
        record["action_to_outcome_edge"]
        for record in agent_records
        if record.get("action_to_outcome_edge")
    ]
    post_change_ticks = sorted(
        {
            int(record["applied_tick"]) + 1
            for record in material_records
            if int(record["applied_tick"]) + 1 < horizon
        }
    )
    world_evolution = {
        "source_scheduled_change_records": source_records,
        "declared_perturbation_change_records": declared_records,
        # Both records are explicit, material runtime changes.  Keeping them
        # separate avoids calling procedural variants source schedules while
        # allowing the audit to accept either permitted exogenous mechanism.
        "exogenous_change_records": [*source_records, *declared_records],
        "material_exogenous_event_records": material_records,
        "post_change_decision_ticks": post_change_ticks,
        "event_to_action_edges": edges,
        "adaptive_replanning_observed": any(
            int(agent.get("applied_tick") or 0)
            > int(source.get("applied_tick") or 0)
            for agent in agent_records
            for source in material_records
        ),
        "agent_action_backend_effect_observed": bool(agent_records),
    }
    return evidence, world_evolution


def build_backend_evidence_coverage(
    *,
    suite: dict[str, Any],
    exercise: bool,
    exercise_scope: str = "representative",
    sample_timeout_seconds: float = 180.0,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    if exercise_scope not in {"representative", "all"}:
        raise ValueError(f"invalid exercise_scope: {exercise_scope}")
    rows = report_rows(suite)
    cache_counts: Counter[str] = Counter(
        {"hits": 0, "misses": 0, "writes": 0}
    )
    by_backend_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_backend_rows.setdefault(
            str(row.get("backend_kind") or ""), []
        ).append(row)

    by_backend: dict[str, dict[str, Any]] = {}
    for backend, backend_rows in sorted(by_backend_rows.items()):
        if exercise:
            selected_rows = (
                backend_rows
                if exercise_scope == "all"
                else backend_rows[:1]
            )
            row_results = [
                {
                    "scenario_id": str(row.get("scenario_id") or ""),
                    "scenario_signature": str(
                        row.get("scenario_signature") or ""
                    ),
                    **_probe_with_cache(
                        row,
                        timeout_seconds=sample_timeout_seconds,
                        cache_dir=cache_dir,
                        cache_counts=cache_counts,
                    ),
                }
                for row in selected_rows
            ]
            if exercise_scope == "representative":
                probe = {
                    key: value
                    for key, value in row_results[0].items()
                    if key not in {"scenario_id", "scenario_signature"}
                }
            else:
                all_passed = all(
                    row["probe_status"] == "runtime_probe_passed"
                    for row in row_results
                )
                classifications = {
                    str(row["runtime_trace_classification"])
                    for row in row_results
                }
                applicability = {
                    str(row["world_evolution_applicability"])
                    for row in row_results
                }
                probe = {
                    "probe_status": (
                        "runtime_probe_passed"
                        if all_passed
                        else "runtime_probe_held"
                    ),
                    "source_probe_status": (
                        "runtime_probe_passed"
                        if all(
                            row["source_probe_status"]
                            == "runtime_probe_passed"
                            for row in row_results
                        )
                        else "runtime_probe_held"
                    ),
                    "world_probe_status": (
                        "runtime_probe_passed"
                        if all(
                            row["world_probe_status"]
                            == "runtime_probe_passed"
                            for row in row_results
                        )
                        else "runtime_probe_held"
                    ),
                    "source_trace_complete": all(
                        row["source_trace_complete"] is True
                        for row in row_results
                    ),
                    "world_release_eligible": all(
                        row["world_release_eligible"] is True
                        for row in row_results
                    ),
                    "agent_action_backend_effect_observed": all(
                        row["agent_action_backend_effect_observed"] is True
                        for row in row_results
                    ),
                    "deterministic_replay_verified": all(
                        row["deterministic_replay_verified"] is True
                        for row in row_results
                    ),
                    "world_evolution_applicability": (
                        next(iter(applicability))
                        if len(applicability) == 1
                        else "mixed"
                    ),
                    "fabricated_exogenous_events": any(
                        row["fabricated_exogenous_events"] is True
                        for row in row_results
                    ),
                    "world_evolution_reason_code": (
                        None
                        if all_passed
                        else "world_evolution_evidence_incomplete"
                    ),
                    "disposition": (
                        "runtime_evidence_ready"
                        if all_passed
                        else "runtime_evidence_incomplete"
                    ),
                    "runtime_trace_classification": (
                        next(iter(classifications))
                        if len(classifications) == 1
                        else "mixed"
                    ),
                    "source_evidence_missing_fields": sorted(
                        {
                            field
                            for row in row_results
                            for field in row[
                                "source_evidence_missing_fields"
                            ]
                        }
                    ),
                    "world_evolution_missing_fields": sorted(
                        {
                            field
                            for row in row_results
                            for field in row[
                                "world_evolution_missing_fields"
                            ]
                        }
                    ),
                    "evidence_blockers": sorted(
                        {
                            blocker
                            for row in row_results
                            for blocker in row["evidence_blockers"]
                        }
                    ),
                    "evidence_status": (
                        "passed" if all_passed else "incomplete"
                    ),
                    "runtime_validation_pending": not all_passed,
                }
        else:
            row_results = []
            probe = {
                "probe_status": "runtime_probe_held",
                "source_probe_status": "runtime_probe_held",
                "world_probe_status": "runtime_probe_held",
                "source_trace_complete": False,
                "world_release_eligible": False,
                "agent_action_backend_effect_observed": False,
                "deterministic_replay_verified": False,
                "world_evolution_applicability": "unknown",
                "fabricated_exogenous_events": False,
                "world_evolution_reason_code": "runtime_probe_not_requested",
                "disposition": "runtime_probe_not_requested",
                "runtime_trace_classification": (
                    "runtime_trace_unimplemented"
                ),
                "source_evidence_missing_fields": list(
                    _REQUIRED_RUNTIME_FIELDS
                ),
                "world_evolution_missing_fields": list(
                    _REQUIRED_WORLD_FIELDS
                ),
                "evidence_blockers": ["runtime_probe_not_requested"],
                "runtime_validation_pending": True,
            }
        by_backend[backend] = {
            "n_rows": len(backend_rows),
            "n_runtime_validated_rows": sum(
                row["probe_status"] == "runtime_probe_passed"
                for row in row_results
            ),
            "all_selected_rows_runtime_validated": bool(
                exercise
                and len(row_results) == len(backend_rows)
                and all(
                    row["probe_status"] == "runtime_probe_passed"
                    for row in row_results
                )
            ),
            "runtime_exercise_scope": exercise_scope,
            "row_runtime_results": row_results,
            "domains": dict(
                sorted(
                    Counter(
                        str(row.get("domain") or "")
                        for row in backend_rows
                    ).items()
                )
            ),
            "representative_scenario_id": str(
                backend_rows[0].get("scenario_id") or ""
            ),
            **probe,
        }

    source_pending = [
        (backend, item)
        for backend, item in by_backend.items()
        if not item["source_trace_complete"]
    ]
    candidates = source_pending or [
        (backend, item)
        for backend, item in by_backend.items()
        if item["runtime_validation_pending"]
    ]
    candidates = candidates or list(by_backend.items())
    recommended = (
        sorted(
            candidates,
            key=lambda pair: (-int(pair[1]["n_rows"]), pair[0]),
        )[0][0]
        if candidates
        else None
    )
    lineage = suite.get("lineage_contract") or {}
    runtime_exceptions = sorted(
        backend
        for backend, item in by_backend.items()
        if item["source_probe_status"] == "runtime_probe_passed"
    )
    return {
        "schema_version": "2.1",
        "status": "complete",
        "n_working_rows": len(rows),
        "working_set_lineage_status": str(
            lineage.get("status") or suite.get("status") or "unknown"
        ),
        "runtime_probe_is_diagnostic": True,
        "cache_summary": {
            key: cache_counts[key] for key in ("hits", "misses", "writes")
        },
        "by_backend": by_backend,
        "runtime_probe_summary": dict(
            sorted(
                Counter(
                    item["probe_status"] for item in by_backend.values()
                ).items()
            )
        ),
        "review_claim": {
            "scope": (
                "all_selected_rows"
                if exercise and exercise_scope == "all"
                else "one_representative_runtime_probe_per_backend"
            ),
            "row_coverage_is_inventory_not_runtime_validation": not (
                exercise and exercise_scope == "all"
            ),
            "formal_source_consumption_proven": all(
                item["probe_status"] == "runtime_probe_passed"
                for item in by_backend.values()
            )
            and bool(by_backend),
            "only_sumo_defines_protocol21_source_trace": "rejected",
            "runtime_trace_exceptions": runtime_exceptions,
            "exceptions": runtime_exceptions,
        },
        "recommended_first_backend": recommended,
        "recommendation_basis": {
            "policy": "largest_pending_backend_first",
            "n_rows": (
                by_backend[recommended]["n_rows"] if recommended else 0
            ),
            "probe_status": (
                by_backend[recommended]["probe_status"]
                if recommended
                else None
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exercise", action="store_true")
    parser.add_argument(
        "--exercise-scope",
        choices=("representative", "all"),
        default="representative",
    )
    parser.add_argument(
        "--sample-timeout-seconds",
        type=float,
        default=180.0,
    )
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    report = build_backend_evidence_coverage(
        suite=json.loads(args.working_set.read_text(encoding="utf-8")),
        exercise=args.exercise,
        exercise_scope=args.exercise_scope,
        sample_timeout_seconds=args.sample_timeout_seconds,
        cache_dir=args.cache_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
