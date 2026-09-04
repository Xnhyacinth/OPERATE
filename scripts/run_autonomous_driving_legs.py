#!/usr/bin/env python3
"""Run deterministic native-SUMO baseline legs for one driving source bundle.

This is a calibration/provenance harness, not an LLM leaderboard runner.  It
keeps the simulator and shield identical across legs and reports the
incremental tactical policy result.  A provider-backed LLM run remains under
the normal episode runner and is not silently substituted here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import signal
import subprocess  # nosec B404 - fixed argv, no shell
import sys
from collections.abc import Callable
from contextlib import redirect_stdout, suppress
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.autonomous_driving_policy import greedy_action, oracle_action  # noqa: E402
from core import Action  # noqa: E402
from domains.autonomous_driving.adapter import AutonomousDrivingEnvironment  # noqa: E402


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_relative(path_value: Any, bundle: Path) -> str:
    path = Path(str(path_value))
    resolved = path.resolve() if path.is_absolute() else (bundle / path).resolve()
    try:
        return resolved.relative_to(bundle.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("autonomous_driving_report_path_outside_bundle") from error


def _portable_scenario(scenario: dict[str, Any], bundle: Path) -> dict[str, Any]:
    portable = deepcopy(scenario)
    backend = dict(portable.get("backend_config") or {})
    backend["source_bundle"] = "."
    for name in ("sumo_config_path", "sumo_net_path", "sumo_route_path"):
        if backend.get(name):
            backend[name] = _bundle_relative(backend[name], bundle)
    portable["backend_config"] = backend
    return portable


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value or ""))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _load_bound_scenario_yaml(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Load and bind the exact formal scenario artifact to its source bundle."""
    from core.scenario_validator import validate_scenario_yaml

    scenario_path = path.resolve()
    raw = scenario_path.read_bytes()
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ValueError("autonomous_driving_scenario_yaml_object_required")
    scenario = dict(value)
    errors = validate_scenario_yaml(scenario, scenario_path)
    if errors:
        raise ValueError("autonomous_driving_scenario_yaml_invalid:" + ";".join(errors))
    backend = dict(scenario.get("backend_config") or {})
    provenance = dict(scenario.get("provenance") or {})
    task_contract = scenario.get("task_contract")
    if (
        scenario.get("domain") != "autonomous_driving"
        or scenario.get("backend_kind") != "sumo_ego"
        or backend.get("execution_mode") != "live"
        or not isinstance(task_contract, dict)
        or not task_contract
    ):
        raise ValueError("autonomous_driving_scenario_formal_contract_invalid")
    candidate_id = str(backend.get("candidate_id") or "")
    ego_actor_id = str(backend.get("ego_actor_id") or "")
    source_window_sha256 = str(scenario.get("source_window_sha256") or "")
    source_event_chain_sha256 = str(provenance.get("source_event_chain_sha256") or "")
    if (
        not candidate_id
        or not ego_actor_id
        or not _valid_sha256(source_window_sha256)
        or not _valid_sha256(source_event_chain_sha256)
    ):
        raise ValueError("autonomous_driving_scenario_source_identity_missing")
    if str(provenance.get("candidate_id") or "") != candidate_id:
        raise ValueError("autonomous_driving_scenario_provenance_candidate_mismatch")
    bundle = _resolve_repo_path(backend.get("source_bundle"))
    if not bundle.is_dir():
        raise ValueError("autonomous_driving_scenario_bundle_missing")
    expected_assets = {
        "sumo_config_path": bundle / "sumo/run.sumocfg",
        "sumo_net_path": bundle / "sumo/network.net.xml",
        "sumo_route_path": bundle / "sumo/routes.rou.xml",
    }
    for name, expected in expected_assets.items():
        observed = _resolve_repo_path(backend.get(name))
        if observed != expected.resolve() or not observed.is_file():
            raise ValueError(f"autonomous_driving_scenario_{name}_mismatch")
    fixture = json.loads((bundle / "runtime/fixture.json").read_text(encoding="utf-8"))
    derivation = dict(fixture.get("derivation") or {})
    if str(derivation.get("candidate_id") or "") != candidate_id:
        raise ValueError("autonomous_driving_scenario_fixture_candidate_mismatch")
    if str(derivation.get("ego_actor_id") or "") != ego_actor_id:
        raise ValueError("autonomous_driving_scenario_fixture_ego_mismatch")
    if str(derivation.get("source_window_sha256") or "") != source_window_sha256:
        raise ValueError("autonomous_driving_scenario_fixture_window_mismatch")
    manifest = json.loads((bundle / "bundle.json").read_text(encoding="utf-8"))
    if (
        str((manifest.get("evidence") or {}).get("runtime_source_events_sha256") or "")
        != source_event_chain_sha256
    ):
        raise ValueError("autonomous_driving_scenario_event_chain_mismatch")
    runtime = deepcopy(scenario)
    runtime_backend = dict(runtime["backend_config"])
    runtime_backend["source_bundle"] = str(bundle)
    for name, expected in expected_assets.items():
        runtime_backend[name] = str(expected.resolve())
    runtime["backend_config"] = runtime_backend
    artifact = {
        "schema_version": "autonomous_driving_scenario_artifact_v1",
        "scenario_id": str(scenario.get("scenario_id") or scenario.get("seed_id") or ""),
        "scenario_yaml_sha256": hashlib.sha256(raw).hexdigest(),
        "semantic_contract_sha256": _digest(scenario),
        "candidate_id": candidate_id,
        "source_window_sha256": source_window_sha256,
        "source_event_chain_sha256": source_event_chain_sha256,
    }
    return runtime, bundle, artifact


def _portable_source_consumption(
    source_consumption: dict[str, Any], bundle: Path
) -> dict[str, Any]:
    portable = deepcopy(source_consumption)
    paths = portable.get("opened_source_paths")
    if isinstance(paths, list):
        portable["opened_source_paths"] = [_bundle_relative(value, bundle) for value in paths]
    hashes = portable.get("opened_source_sha256")
    if isinstance(hashes, dict):
        portable["opened_source_sha256"] = {
            _bundle_relative(name, bundle): value for name, value in hashes.items()
        }
    return portable


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    """Write a versioned evidence report atomically without overwriting."""
    output = path.resolve()
    partial = output.with_name(f"{output.name}.part")
    if output.exists() or partial.exists():
        raise FileExistsError("autonomous_driving_calibration_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output)


def _post_change_runtime_evidence(
    *,
    leg_name: str,
    ego_actor_id: str,
    conflict_actor_id: str,
    events: list[dict[str, Any]],
    tactical_action_trace: list[dict[str, Any]],
    state_digests: list[str],
) -> list[dict[str, Any]]:
    """Bind a material source event to later native action/state evidence."""
    material_events = sorted(
        (
            row
            for row in events
            if row.get("origin") == "source_schedule"
            and row.get("materiality_passed") is True
            and isinstance(row.get("tick"), int)
        ),
        key=lambda row: int(row["tick"]),
    )
    effect_by_token = {
        str((row.get("applied_action") or {}).get("effect_token") or ""): row
        for row in events
        if row.get("origin") == "agent_caused" and row.get("materiality_passed") is True
    }
    evidence: list[dict[str, Any]] = []
    for source_event in material_events:
        material_tick = int(source_event["tick"])
        action = next(
            (
                row
                for row in tactical_action_trace
                if isinstance(row.get("tick"), int)
                and int(row["tick"]) > material_tick
                and str(row.get("effect_token") or "") in effect_by_token
            ),
            None,
        )
        if action is None:
            continue
        decision_tick = int(action["tick"])
        native_effect_tick = decision_tick + 1
        if native_effect_tick >= len(state_digests):
            continue
        before_digest = str(state_digests[decision_tick])
        after_digest = str(state_digests[native_effect_tick])
        if len(before_digest) != 64 or len(after_digest) != 64 or before_digest == after_digest:
            continue
        action_id = str(action["effect_token"])
        effect = effect_by_token[action_id]
        evidence.append(
            {
                "source_event_id": str(source_event.get("event_id") or ""),
                "ego_actor_id": ego_actor_id,
                "conflict_actor_id": conflict_actor_id,
                "material_event_tick": material_tick,
                "decision_tick": decision_tick,
                "control_tick": decision_tick,
                "native_effect_tick": native_effect_tick,
                "decision": {
                    "origin": "agent_policy",
                    "policy_leg": leg_name,
                    "action_id": action_id,
                },
                "control": {
                    "action_id": action_id,
                    "call_id": (effect.get("action_to_outcome_edge") or {}).get("source_call_id"),
                    "applied_by_native_backend": True,
                    "backend_kind": "sumo_ego",
                    "tool_name": str(action.get("tool_name") or ""),
                },
                "native_effect": {
                    "observed_from_backend_step": True,
                    "state_digest_before": before_digest,
                    "state_digest_after": after_digest,
                },
            }
        )
        break
    return evidence


def _scenario(args: argparse.Namespace, *, level: str) -> dict[str, Any]:
    bundle = args.bundle.resolve()
    candidate_id = args.candidate_id
    stem = bundle / "sumo"
    required_review_interval = 1 if level in {"high", "extreme"} else 2
    stable_dwell = {"basic": 2, "medium": 3, "high": 4, "extreme": 6}[level]
    mode = "deep_planning" if level in {"high", "extreme"} else "time_pressure"
    return {
        "domain": "autonomous_driving",
        "family": "sustained_highway_risk_supervision",
        "backend_kind": "sumo_ego",
        "horizon_ticks": args.ticks,
        "tick_seconds": args.tick_seconds,
        "difficulty_level": level,
        "difficulty_mode": mode,
        "backend_config": {
            "source_bundle": str(bundle),
            "candidate_id": candidate_id,
            "ego_actor_id": args.ego,
            "execution_mode": "live",
            "sumo_config_path": str((stem / "run.sumocfg").resolve()),
            "sumo_net_path": str((stem / "network.net.xml").resolve()),
            "sumo_route_path": str((stem / "routes.rou.xml").resolve()),
            "ego_vehicle_id": args.ego,
            "task_requirements": {
                "required_stable_dwell_ticks": stable_dwell,
                "guarded_recovery_required_if_mrm": True,
                "recovery_state": "nominal_after_guarded_recovery",
                "recovery_sequence": [
                    "request_minimal_risk_maneuver",
                    "request_recovery_check",
                    "authorize_recovery",
                ],
                "required_review_interval_ticks": required_review_interval,
            },
        },
        "clock_contract": {
            "schema_version": "driving_clock_v1",
            "physics_step_seconds": args.physics_step_seconds,
            "shield_step_seconds": args.physics_step_seconds,
            "substeps_per_supervisory_tick": round(args.tick_seconds / args.physics_step_seconds),
            "provider_wall_clock_advances_simulation": False,
        },
    }


def _action_provider(
    name: str,
) -> Callable[[dict[str, Any], list[dict[str, Any]], dict[str, Any]], Action]:
    if name == "shield_only":
        return lambda _observation, _specs, _scenario: Action(tool_calls=[])
    if name == "rule_tactical":
        return lambda observation, specs, _scenario: greedy_action(observation, specs)
    if name == "oracle_offline":
        return lambda observation, specs, scenario: oracle_action(observation, specs, scenario)
    raise ValueError(f"unsupported calibration leg: {name}")


def _run_leg(name: str, scenario: dict[str, Any], seed: int) -> dict[str, Any]:
    env = AutonomousDrivingEnvironment()
    provider = _action_provider(name)
    records: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    startup_messages: list[str] = []
    try:
        startup = io.StringIO()
        with redirect_stdout(startup):
            observation = env.reset(scenario, seed)
            specs = env.get_tool_specs()
            for _ in range(int(scenario["horizon_ticks"])):
                result = env.step(provider(observation, specs, scenario))
                records.append(dict((result.info.extra or {}).get("runtime_assurance") or {}))
                events.extend(dict(event) for event in result.info.realized_events)
                observation = result.observation
                if result.done:
                    break
        startup_messages = startup.getvalue().splitlines()
        truth = env.ground_truth()
        source_consumption = env.source_consumption_evidence(scenario=scenario)
        costs = dict(truth.get("cost_components") or {})
        source = dict(truth.get("runtime_assurance") or {})
        backend_config = dict(scenario.get("backend_config") or {})
        bundle = Path(str(backend_config.get("source_bundle") or "")).resolve()
        fixture_path = bundle / "runtime/fixture.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        derivation = dict(fixture.get("derivation") or {})
        post_change_runtime_evidence = _post_change_runtime_evidence(
            leg_name=name,
            ego_actor_id=str(backend_config.get("ego_actor_id") or ""),
            conflict_actor_id=str(derivation.get("conflict_actor_id") or ""),
            events=events,
            tactical_action_trace=list(truth.get("tactical_action_trace") or []),
            state_digests=list(source_consumption.get("post_source_state_digests") or []),
        )
        source_consumption = _portable_source_consumption(source_consumption, bundle)
        return {
            "leg": name,
            "status": "completed",
            "records": len(records),
            "startup_messages": startup_messages,
            "collision_count": max(
                int(row.get("collision_count") or 0)
                for row in truth.get("_task_tick_records") or [{"collision_count": 0}]
            ),
            "road_departure_count": max(
                int(row.get("road_departure_count") or 0)
                for row in truth.get("_task_tick_records") or [{"road_departure_count": 0}]
            ),
            "route_progress": float((truth.get("ego") or {}).get("route_position_m") or 0.0)
            / max(float((truth.get("route") or {}).get("length_m") or 1.0), 1.0),
            "shield_interventions": int(source.get("intervention_count") or 0),
            "mrm_ticks": list(source.get("mrm_ticks") or []),
            "source_events": [
                {
                    "event_id": event.get("event_id"),
                    "type": event.get("type"),
                    "tick": event.get("tick"),
                    "actor_id": event.get("actor_id"),
                    "materiality_passed": event.get("materiality_passed"),
                }
                for event in events
                if event.get("origin") == "source_schedule"
            ],
            "cost_components": costs,
            "semantic_digest": _digest(
                {
                    "leg": name,
                    "records": truth.get("_task_tick_records"),
                    "events": events,
                    "costs": costs,
                    "source_consumption": source_consumption,
                }
            ),
            "source_consumption": source_consumption,
            "post_change_runtime_evidence": post_change_runtime_evidence,
        }
    finally:
        env.close()


def _total_cost(row: dict[str, Any]) -> float:
    return sum(float(value) for value in (row.get("cost_components") or {}).values())


def _attribution(legs: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(row.get("leg")): row for row in legs}
    shield = by_name.get("shield_only")
    if shield is None:
        return {"status": "held", "reason": "shield_only_leg_missing"}
    shield_cost = _total_cost(shield)
    shield_collision = int(shield.get("collision_count") or 0)
    shield_departure = int(shield.get("road_departure_count") or 0)
    comparison: dict[str, Any] = {}
    for name, row in by_name.items():
        if name == "shield_only":
            continue
        collision = int(row.get("collision_count") or 0)
        departure = int(row.get("road_departure_count") or 0)
        comparison[name] = {
            "loss_total_cost": round(_total_cost(row), 6),
            "agent_incremental_value_vs_shield_only": round(shield_cost - _total_cost(row), 6),
            "shield_intervention_delta": int(row.get("shield_interventions") or 0)
            - int(shield.get("shield_interventions") or 0),
            "safety_regression_vs_shield_only": bool(
                collision > shield_collision or departure > shield_departure
            ),
            "prevention_credit_eligible": bool(
                collision <= shield_collision and departure <= shield_departure
            ),
        }
    return {
        "status": "diagnostic",
        "shield_only_total_cost": round(shield_cost, 6),
        "comparisons": comparison,
        "interpretation": "cost and shield burden are diagnostic until replay, source, safety, and headroom gates pass",
    }


def _run_leg_isolated(args: argparse.Namespace, name: str) -> dict[str, Any]:
    """Run one native leg in a fresh process and return its single result.

    TraCI/libsumo bindings are process-global on some platforms.  Keeping
    calibration legs in one Python process can therefore leave a stale label
    or native connection after the first leg.  A child process makes each leg
    an independent replay while preserving the same bundle, candidate, seed,
    and clock contract.
    """
    command = [sys.executable, str(Path(__file__).resolve())]
    if args.scenario_yaml is not None:
        command.extend(("--scenario-yaml", str(args.scenario_yaml)))
    else:
        command.extend(
            (
                "--bundle",
                str(args.bundle),
                "--candidate-id",
                str(args.candidate_id),
                "--ego",
                str(args.ego),
                "--seed",
                str(args.seed),
                "--ticks",
                str(args.ticks),
                "--tick-seconds",
                str(args.tick_seconds),
                "--physics-step-seconds",
                str(args.physics_step_seconds),
                "--difficulty-level",
                str(args.difficulty_level),
            )
        )
    command.extend(("--legs", name, "--_single-leg"))
    child_env = os.environ.copy()
    child_env["OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL"] = "1"
    child = subprocess.Popen(  # nosec B603 - fixed argv, no shell
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env,
        start_new_session=(os.name == "posix"),
    )
    timeout_seconds = max(60.0, float(args.ticks) * float(args.tick_seconds) * 4.0)
    try:
        stdout, stderr = child.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        if os.name == "posix":
            with suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGTERM)
        else:
            child.terminate()
        try:
            child.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
            child.wait(timeout=5.0)
        raise RuntimeError(f"isolated leg timed out: {name}") from exc
    if child.returncode != 0:
        detail = stderr.strip() or stdout.strip() or "no child output"
        raise RuntimeError(f"isolated leg failed ({name}, rc={child.returncode}): {detail}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"isolated leg returned non-JSON output: {name}") from exc
    legs = payload.get("legs")
    if (
        payload.get("status") != "diagnostic_complete"
        or not isinstance(legs, list)
        or len(legs) != 1
    ):
        raise RuntimeError(f"isolated leg returned an invalid report: {name}")
    return dict(legs[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-yaml", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--ego")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int)
    parser.add_argument("--tick-seconds", type=float)
    parser.add_argument("--physics-step-seconds", type=float)
    parser.add_argument("--difficulty-level", choices=("basic", "medium", "high", "extreme"))
    parser.add_argument(
        "--legs",
        nargs="+",
        choices=("shield_only", "rule_tactical", "oracle_offline"),
        default=("shield_only", "rule_tactical", "oracle_offline"),
    )
    parser.add_argument(
        "--_single-leg",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    scenario_artifact: dict[str, Any] | None = None
    try:
        if args.scenario_yaml is not None:
            if any(
                value is not None
                for value in (
                    args.bundle,
                    args.candidate_id,
                    args.ego,
                    args.seed,
                    args.ticks,
                    args.tick_seconds,
                    args.physics_step_seconds,
                    args.difficulty_level,
                )
            ):
                raise ValueError("autonomous_driving_scenario_yaml_override_forbidden")
            args.scenario_yaml = args.scenario_yaml.resolve()
            scenario, args.bundle, scenario_artifact = _load_bound_scenario_yaml(args.scenario_yaml)
            backend = dict(scenario["backend_config"])
            clock = dict(scenario["clock_contract"])
            args.candidate_id = str(backend["candidate_id"])
            args.ego = str(backend["ego_actor_id"])
            args.seed = int(scenario["seed"])
            args.ticks = int(scenario["horizon_ticks"])
            args.tick_seconds = float(scenario["tick_seconds"])
            args.physics_step_seconds = float(clock["physics_step_seconds"])
            args.difficulty_level = str(scenario["difficulty_level"])
            evidence_tier = "formal_yaml_bound_v1"
        else:
            if args.bundle is None or not args.candidate_id or not args.ego:
                raise ValueError("autonomous_driving_legacy_identity_required")
            args.bundle = args.bundle.resolve()
            args.seed = 42 if args.seed is None else args.seed
            args.ticks = 8 if args.ticks is None else args.ticks
            args.tick_seconds = 5.0 if args.tick_seconds is None else args.tick_seconds
            args.physics_step_seconds = (
                0.1 if args.physics_step_seconds is None else args.physics_step_seconds
            )
            args.difficulty_level = args.difficulty_level or "basic"
            scenario = _scenario(args, level=args.difficulty_level)
            evidence_tier = "diagnostic_legacy_generic_scenario_v1"
        if args.ticks < 1 or args.physics_step_seconds <= 0.0 or args.tick_seconds <= 0.0:
            raise ValueError("ticks and clock durations must be positive")
        substeps = round(args.tick_seconds / args.physics_step_seconds)
        if abs(substeps * args.physics_step_seconds - args.tick_seconds) > 1e-9:
            raise ValueError("tick-seconds must be an exact multiple of physics-step-seconds")
        os.environ["OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL"] = "1"
        if len(args.legs) > 1 and not args._single_leg:
            legs = [_run_leg_isolated(args, name) for name in args.legs]
            leg_isolation = "subprocess_per_leg"
        else:
            legs = [_run_leg(name, scenario, args.seed) for name in args.legs]
            leg_isolation = "in_process_single_leg"
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(
            json.dumps(
                {
                    "status": "held",
                    "reason": str(error),
                    "platform": {
                        "system": platform.system(),
                        "machine": platform.machine(),
                        "python_implementation": platform.python_implementation(),
                        "python_version": platform.python_version(),
                        "policy": "cross_platform_runtime_fingerprint_v1",
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    report = {
        "schema_version": (
            "autonomous_driving_calibration_legs_v2"
            if scenario_artifact is not None
            else "autonomous_driving_calibration_legs_v1"
        ),
        "status": "diagnostic_complete",
        "evidence_tier": evidence_tier,
        "admission": "held_until_data_runtime_and_headroom_gates",
        "baseline_fingerprints": {
            "policy_module_sha256": _file_digest(
                REPO_ROOT / "baselines/autonomous_driving_policy.py"
            ),
            "runner_sha256": _file_digest(Path(__file__).resolve()),
        },
        "leg_isolation": leg_isolation,
        "seed": args.seed,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "policy": "cross_platform_runtime_fingerprint_v1",
        },
        "scenario": _portable_scenario(scenario, args.bundle.resolve()),
        "legs": legs,
        "attribution": _attribution(legs),
    }
    if scenario_artifact is not None:
        report["scenario_artifact"] = scenario_artifact
    if {str(value.get("leg") or "") for value in legs} == {
        "shield_only",
        "rule_tactical",
        "oracle_offline",
    }:
        try:
            from domains.autonomous_driving.evidence_binding import (
                calibration_evidence_binding,
                sumo_runtime_binding,
            )

            report["evidence_binding"] = calibration_evidence_binding(
                repo_root=REPO_ROOT,
                bundle=args.bundle.resolve(),
                candidate_id=args.candidate_id,
                legs=[dict(value) for value in legs],
            )
            report["sumo_runtime"] = sumo_runtime_binding(REPO_ROOT)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            print(
                json.dumps(
                    {
                        "status": "held",
                        "reason": f"portable_evidence_binding_failed: {error}",
                    },
                    sort_keys=True,
                )
            )
            return 2
    report["report_digest_sha256"] = _digest(report)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        try:
            _write_json_exclusive(output, report)
        except OSError as error:
            print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True))
            return 1
        print(
            json.dumps(
                {
                    "status": "verified",
                    "output": str(output.resolve()),
                    "report_digest_sha256": report["report_digest_sha256"],
                },
                sort_keys=True,
            )
        )
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
