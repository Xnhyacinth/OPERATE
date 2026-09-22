#!/usr/bin/env python3
"""Run per-sample behavioral gates for the active OPERATE Core candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import signal
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import]

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit._common import _resolve_scenario_path  # noqa: E402
from audit.episode_cache import (  # noqa: E402
    AUDIT_EPISODE_CONTRACT_VERSION,
    EpisodeCache,
    use_episode_cache,
)
from audit.self_consistency import episode_metrics  # noqa: E402
from baselines import (  # noqa: E402
    GreedyHeuristicAgent,
    OracleOfflineAgent,
    RandomAgent,
    WaitOnlyAgent,
)
from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_admission import (  # noqa: E402
    declared_protocol21_admission_profile,
    partition_behavioral_check_failures,
    resolve_protocol21_admission_profile,
)
from evaluation import SCORING_VERSION  # noqa: E402
from runner import (  # noqa: E402
    EVALUATION_IMPLEMENTATION_FINGERPRINT,
    EVALUATION_PROTOCOL_VERSION,
)

DEFAULT_SUITE = (
    REPO_ROOT / "release" / "operate_v0_58_0" / "protocol21_source_suite.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release"
    / "operate_v0_58_0_candidate"
    / "operate_v058_formal"
    / "behavioral_calibration.json"
)
DEFAULT_CACHE = REPO_ROOT / ".audit-cache" / "operate-v058-candidate-episodes"
AGENTS = {
    "wait_only": WaitOnlyAgent,
    "random": RandomAgent,
    "greedy_heuristic": GreedyHeuristicAgent,
    "oracle_offline": OracleOfflineAgent,
}
NATIVE_DIMENSION_IMPROVEMENT = 1.0
MIN_PREVENTED_LOSS = 1.0
MAX_CRITICAL_NATIVE_REGRESSION = 5.0
CRITICAL_NATIVE_DIMENSIONS = {"system_survival", "safety_violation"}
HARD_ISOLATION_BACKENDS = frozenset(
    {
        "cigre_distribution",
        "dynasched_flexible_job_shop",
        "grid2op",
        "opendss_fresh_feeders",
        "opendss_ieee13",
        "pandapower_acopf",
        "pandapower_lv",
        "pyvrp_cvrp",
        "pyvrp_lastmile",
        "pyvrp_vrptw",
        "sumo",
    }
)
BOUNDED_ISOLATION_WORKERS = {"sumo": 1}
NATURAL_REAP_SECONDS = 0.5
# Live SUMO is process-owned and can expose a transient counterfactual result
# under concurrent native instances even when the source trace is identical.
# Serialize SUMO, and recompute its episodes every time so the paired wait
# replay cannot depend on another worker or an earlier worker's lifecycle.
# Other native backends retain their requested workers and content-addressed
# cache.
UNCACHEABLE_RUNTIME_BACKENDS = frozenset({"sumo"})


def _episode_cache_for_row(
    cache_dir: Path | None, row: dict[str, Any]
) -> EpisodeCache | None:
    if cache_dir is None or str(row.get("backend_kind") or "") in (
        UNCACHEABLE_RUNTIME_BACKENDS
    ):
        return None
    return EpisodeCache(cache_dir)


def _filter_rows(
    rows: list[dict[str, Any]],
    *,
    scenario_ids: set[str] | None,
    domains: set[str] | None,
    levels: set[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if (not scenario_ids or str(row.get("scenario_id")) in scenario_ids)
        and (not domains or str(row.get("domain")) in domains)
        and (not levels or str(row.get("difficulty_level")) in levels)
    ]
    return selected[:limit] if limit is not None else selected


def _classify_result(result: dict[str, Any]) -> dict[str, Any]:
    """Apply the current fail-closed behavioral gate to saved episode metrics."""
    if result.get("status") == "error" or "episodes" not in result:
        return result
    episodes = result["episodes"]
    wait = episodes["wait_only"]
    oracle = episodes["oracle_offline"]
    native_task_contracts = {
        str((episodes[name].get("task_completion") or {}).get("contract") or "")
        for name in ("wait_only", "greedy_heuristic", "oracle_offline")
    }
    native_state_loss_contract = (
        "microgrid.native_state_loss.v1" in native_task_contracts
    )

    def _native_task_loss(episode: dict[str, Any]) -> float | None:
        evidence = episode.get("task_completion", {}).get("evidence") or {}
        actual = evidence.get("actual_task_loss")
        if actual is None:
            return None
        try:
            return float(actual)
        except (TypeError, ValueError):
            return None

    def _native_task_headroom(episode: dict[str, Any]) -> float:
        evidence = episode.get("task_completion", {}).get("evidence") or {}
        try:
            return float(evidence.get("counterfactual_task_loss", 0.0)) - float(
                evidence.get("actual_task_loss", 0.0)
            )
        except (TypeError, ValueError):
            return 0.0

    def _critical_native_values(
        scores: dict[str, Any],
    ) -> dict[str, float] | None:
        values: dict[str, float] = {}
        for dimension in CRITICAL_NATIVE_DIMENSIONS:
            try:
                value = float(scores[dimension])
            except (KeyError, TypeError, ValueError):
                return None
            if not math.isfinite(value):
                return None
            values[dimension] = value
        return values

    raw_headroom = max(
        float(episodes[name]["raw_total"]) - float(wait["raw_total"])
        for name in ("greedy_heuristic", "oracle_offline")
    )
    score_headroom = max(
        float(episodes[name]["total_score"]) - float(wait["total_score"])
        for name in ("greedy_heuristic", "oracle_offline")
    )
    native_task_headroom = max(
        (
            _native_task_headroom(episodes[name])
            for name in ("greedy_heuristic", "oracle_offline")
        ),
        default=0.0,
    )
    beneficial_cost_gap = max(
        0.0,
        *(
            (float(wait["cost"]) - float(episodes[name]["cost"]))
            / max(1.0, abs(float(wait["cost"])))
            for name in ("greedy_heuristic", "oracle_offline")
        ),
    )
    wait_native = dict(wait.get("native_dimension_scores") or {})
    wait_critical = _critical_native_values(wait_native)
    native_improvements: dict[str, float] = {}
    native_leverage_agents: list[str] = []
    process_capable_reference_agents: list[str] = []
    safe_native_agents: list[str] = []
    critical_native_evidence_agents: list[str] = []
    critical_regressions: dict[str, float] = {}
    for name in ("greedy_heuristic", "oracle_offline"):
        episode = episodes[name]
        task_completed = bool((episode.get("task_completion") or {}).get("completed"))
        task_completion = dict(episode.get("task_completion") or {})
        process_capability_met = (
            task_completion.get("process_capability_satisfied") is True
            if task_completion.get("process_capability_applicable") is True
            else True
        )
        if task_completed and process_capability_met:
            process_capable_reference_agents.append(name)
        improvements = [
            float(value) - float(wait_native[dimension])
            for dimension, value in (
                episode.get("native_dimension_scores") or {}
            ).items()
            if dimension in wait_native
        ]
        best_improvement = max(improvements, default=0.0)
        native_improvements[name] = best_improvement
        episode_critical = _critical_native_values(
            dict(episode.get("native_dimension_scores") or {})
        )
        critical_evidence_complete = (
            wait_critical is not None and episode_critical is not None
        )
        critical_deltas = (
            [
                episode_critical[dimension] - wait_critical[dimension]
                for dimension in CRITICAL_NATIVE_DIMENSIONS
            ]
            if critical_evidence_complete
            else []
        )
        worst_critical_delta = min(
            critical_deltas,
            default=-(MAX_CRITICAL_NATIVE_REGRESSION + 1.0),
        )
        critical_regressions[name] = worst_critical_delta
        no_critical_regression = (
            critical_evidence_complete
            and worst_critical_delta >= -MAX_CRITICAL_NATIVE_REGRESSION
        )
        if critical_evidence_complete:
            critical_native_evidence_agents.append(name)
        if no_critical_regression:
            safe_native_agents.append(name)
        cost_gap = (float(wait["cost"]) - float(episode["cost"])) / max(
            1.0, abs(float(wait["cost"]))
        )
        has_state_change = (
            int(episode.get("successful_state_changing_calls", 0) or 0) > 0
        )
        has_native_benefit = (
            cost_gap >= 0.005
            or float(episode.get("prevented_loss", 0.0) or 0.0) > MIN_PREVENTED_LOSS
            or best_improvement >= NATIVE_DIMENSION_IMPROVEMENT
        )
        if (
            has_state_change
            and has_native_benefit
            and no_critical_regression
            and task_completed
        ):
            native_leverage_agents.append(name)
    checks = dict(result.get("checks") or {})
    required_replays = {
        "wait_only",
        "greedy_heuristic",
        "oracle_offline",
    }
    checks["native_backend_executable"] = required_replays.issubset(episodes) and all(
        isinstance(episodes.get(name), dict)
        and "cost" in episodes[name]
        and "total_score" in episodes[name]
        for name in required_replays
    )
    if native_state_loss_contract:
        wait_loss = _native_task_loss(wait)
        oracle_loss = _native_task_loss(oracle)
        checks["oracle_not_worse_than_wait"] = (
            wait_loss is not None
            and oracle_loss is not None
            and oracle_loss <= wait_loss * 1.1 + 1e-9
        )
    else:
        checks["oracle_not_worse_than_wait"] = (
            float(oracle["cost"]) <= float(wait["cost"]) * 1.1 + 1e-9
        )
    checks["aggregate_decision_headroom"] = (
        raw_headroom > 0.0
        or beneficial_cost_gap > 0.0
        or native_task_headroom > 0.0
        or max(native_improvements.values(), default=0.0) > 0.0
    )
    checks["native_state_changing_leverage"] = bool(native_leverage_agents)
    checks["task_contract_completed_by_reference"] = any(
        bool((episodes[name].get("task_completion") or {}).get("completed"))
        for name in ("greedy_heuristic", "oracle_offline")
    )
    checks["reference_process_capability_satisfied"] = bool(
        process_capable_reference_agents
    )
    checks["wait_only_task_contract_not_completed"] = not bool(
        (wait.get("task_completion") or {}).get("completed")
    )
    checks["no_critical_native_regression"] = bool(safe_native_agents)
    checks["positive_decision_headroom"] = (
        checks["aggregate_decision_headroom"]
        and checks["native_state_changing_leverage"]
    )
    result["checks"] = checks
    result["metrics"] = {
        "best_greedy_or_oracle_raw_headroom": round(raw_headroom, 9),
        "best_greedy_or_oracle_score_headroom": round(score_headroom, 9),
        "best_beneficial_relative_cost_gap": round(beneficial_cost_gap, 9),
        "best_native_task_loss_headroom": round(native_task_headroom, 9),
        "best_native_dimension_improvement": round(
            max(native_improvements.values(), default=0.0), 9
        ),
        "native_leverage_agents": native_leverage_agents,
        "process_capable_reference_agents": process_capable_reference_agents,
        "safe_native_agents": safe_native_agents,
        "critical_native_evidence_agents": critical_native_evidence_agents,
        "worst_critical_native_dimension_delta": round(
            min(critical_regressions.values(), default=0.0), 9
        ),
        "oracle_wait_score_headroom": round(
            float(oracle["total_score"]) - float(wait["total_score"]), 9
        ),
        "oracle_wait_raw_headroom": round(
            float(oracle["raw_total"]) - float(wait["raw_total"]), 9
        ),
    }
    admission_profile = resolve_protocol21_admission_profile(result)
    admission_failures, diagnostic_failures = partition_behavioral_check_failures(
        checks,
        profile=admission_profile,
    )
    result["admission_profile"] = admission_profile
    result["admission_failures"] = admission_failures
    result["diagnostic_failures"] = diagnostic_failures
    result["status"] = "passed" if not admission_failures else "failed"
    return result


def _episode(
    row: dict[str, Any],
    body: dict[str, Any],
    name: str,
    *,
    replay_index: int = 0,
) -> dict[str, Any]:
    metrics = episode_metrics(
        body,
        row,
        AGENTS[name],
        difficulty_level=str(row["difficulty_level"]),
        scenario_signature=str(row["scenario_signature"]),
        replay_index=replay_index,
    )
    return {
        "cost": round(float(metrics["cost"]), 9),
        "total_score": round(float(metrics["total_score"]), 9),
        "raw_total": round(float(metrics["raw_total"]), 9),
        "tool_calls": int(metrics["tool_calls"]),
        "successful_non_wait_calls": int(metrics["successful_non_wait_calls"]),
        "successful_state_changing_calls": int(
            metrics.get("successful_state_changing_calls", 0) or 0
        ),
        "effective_tool_names": list(metrics.get("effective_tool_names") or []),
        "effective_state_changing_ticks": list(
            metrics.get("effective_state_changing_ticks") or []
        ),
        "interaction_ticks": list(metrics.get("interaction_ticks") or []),
        "effective_decision_ticks": int(
            metrics.get("effective_decision_ticks", 0) or 0
        ),
        "phase_depth_proxy": int(metrics.get("phase_depth_proxy", 0) or 0),
        "shortest_strategy_status": str(
            metrics.get("shortest_strategy_status", "unknown")
        ),
        "required_tool_set_status": str(
            metrics.get("required_tool_set_status", "unknown")
        ),
        "prevented_loss": round(float(metrics.get("prevented_loss", 0.0) or 0.0), 9),
        "normalized_prevention": round(
            float(metrics.get("normalized_prevention", 0.0) or 0.0), 9
        ),
        "native_dimension_scores": {
            str(key): round(float(value), 9)
            for key, value in (metrics.get("native_dimension_scores") or {}).items()
        },
        "task_completion": dict(metrics.get("task_completion") or {}),
        "world_evolution": dict(metrics.get("world_evolution") or {}),
        "event_adaptive_autonomy": dict(metrics.get("event_adaptive_autonomy") or {}),
        "terminal_integrity": dict(metrics.get("terminal_integrity") or {}),
        "traffic_capture": dict(metrics.get("traffic_capture") or {}),
        "source_consumption_evidence": dict(
            metrics.get("source_consumption_evidence") or {}
        ),
    }


def _sumo_episode_child(
    connection: Any,
    row: dict[str, Any],
    body: dict[str, Any],
    name: str,
    replay_index: int,
    inherit_sample_process_group: bool,
) -> None:
    """Compute one live-SUMO episode in a disposable native process."""
    if not inherit_sample_process_group:
        _claim_process_group()
    try:
        try:
            payload = {
                "status": "ok",
                "episode": _episode(
                    row,
                    body,
                    name,
                    replay_index=replay_index,
                ),
            }
        except Exception as exc:
            payload = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        connection.send(payload)
    finally:
        connection.close()


def _run_sumo_episode_isolated(
    row: dict[str, Any],
    body: dict[str, Any],
    name: str,
    *,
    replay_index: int = 0,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Return one SUMO episode without reusing process-global libsumo state."""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    inherit_sample_process_group = _current_process_owns_group()
    process = context.Process(
        target=_sumo_episode_child,
        args=(
            child,
            row,
            body,
            name,
            replay_index,
            inherit_sample_process_group,
        ),
        daemon=False,
    )
    process.start()
    child.close()
    try:
        if timeout_seconds and timeout_seconds > 0:
            ready = parent.poll(float(timeout_seconds))
        else:
            ready = True
        if not ready:
            raise TimeoutError(
                f"SUMO {name} episode exceeded {timeout_seconds}s in isolated process"
            )
        try:
            payload = parent.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"SUMO {name} episode worker exited without a result"
            ) from exc
        if payload.get("status") != "ok":
            raise RuntimeError(
                f"SUMO {name} episode failed in isolated process: "
                f"{payload.get('error') or 'unknown error'}"
            )
        return dict(payload["episode"])
    finally:
        parent.close()
        _stop_isolated_process(
            process,
            terminate_group=not inherit_sample_process_group,
        )


def _determinism_differences(
    first: Any,
    second: Any,
    *,
    limit: int = 64,
) -> list[dict[str, Any]]:
    """Return bounded, exact leaf differences for paired-replay diagnosis."""
    differences: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, path: str) -> None:
        if len(differences) >= limit:
            return
        if type(left) is not type(right):
            differences.append({"path": path or "/", "first": left, "second": right})
        elif isinstance(left, dict):
            for key in sorted(set(left) | set(right)):
                child_path = f"{path}/{key}"
                if key not in left or key not in right:
                    differences.append(
                        {
                            "path": child_path,
                            "first": left.get(key),
                            "second": right.get(key),
                        }
                    )
                else:
                    visit(left[key], right[key], child_path)
                if len(differences) >= limit:
                    return
        elif isinstance(left, list):
            if len(left) != len(right):
                differences.append(
                    {
                        "path": f"{path}/length",
                        "first": len(left),
                        "second": len(right),
                    }
                )
            for index, (left_item, right_item) in enumerate(
                zip(left, right, strict=False)
            ):
                visit(left_item, right_item, f"{path}/{index}")
                if len(differences) >= limit:
                    return
        elif left != right:
            differences.append({"path": path or "/", "first": left, "second": right})

    visit(first, second, "")
    return differences


@contextmanager
def _sample_timeout(seconds: int | None):
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"behavioral sample exceeded {seconds}s")

    prior_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prior_handler)


def _claim_process_group() -> None:
    """Put an isolated replay and its native children in one killable group."""
    if not hasattr(os, "setpgid"):
        return
    try:
        os.setpgid(0, 0)
    except OSError:
        # Some multiprocessing start methods already make the child a group
        # leader; the parent-side cleanup remains safe in that case.
        return


def _current_process_owns_group() -> bool:
    if not hasattr(os, "getpgrp"):
        return False
    try:
        return int(os.getpgrp()) == int(os.getpid())
    except OSError:
        return False


def _terminate_process_group(process: Any) -> None:
    """Terminate native descendants that outlive an isolated Python child."""
    process_id = getattr(process, "pid", None)
    if process_id is None or not hasattr(os, "killpg"):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(int(process_id), sig)
        except OSError:
            # The process group may already have exited; cleanup is
            # intentionally best effort and must not hide the sample result.
            continue


def _signal_process_group(process: Any, sig: int) -> None:
    process_id = getattr(process, "pid", None)
    if process_id is None or not hasattr(os, "killpg"):
        return
    try:
        os.killpg(int(process_id), sig)
    except OSError:
        return


def _process_group_exists(process: Any) -> bool:
    process_id = getattr(process, "pid", None)
    if process_id is None or not hasattr(os, "killpg"):
        return False
    try:
        os.killpg(int(process_id), 0)
    except OSError:
        return False
    return True


def _stop_isolated_process(process: Any, *, terminate_group: bool = True) -> bool:
    """Reap one isolated replay and every native child in its process group."""
    process.join(timeout=NATURAL_REAP_SECONDS)
    if not process.is_alive():
        if not terminate_group or not _process_group_exists(process):
            return False
        _signal_process_group(process, signal.SIGTERM)
        _signal_process_group(process, signal.SIGKILL)
        return True
    if terminate_group:
        _signal_process_group(process, signal.SIGTERM)
    process.join(timeout=1.0)
    # A native child can survive after the Python group leader exits.  Address
    # the group while its original pgid is still known, before per-process
    # fallbacks make that lifecycle harder to observe.
    if terminate_group:
        _signal_process_group(process, signal.SIGKILL)
    process.join(timeout=1.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1.0)
    if process.is_alive() and hasattr(process, "kill"):
        process.kill()
        process.join(timeout=1.0)
    return True


def _calibrate_one(
    row: dict[str, Any],
    cache_dir: Path | None = None,
    sample_timeout_seconds: int | None = 600,
) -> dict[str, Any]:
    path = _resolve_scenario_path(row["path"])
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        with _sample_timeout(sample_timeout_seconds):
            cache = _episode_cache_for_row(cache_dir, row)
            with use_episode_cache(cache):

                def run_episode(name: str, *, replay_index: int = 0) -> dict[str, Any]:
                    if str(row.get("backend_kind") or "") == "sumo":
                        return _run_sumo_episode_isolated(
                            row,
                            body,
                            name,
                            replay_index=replay_index,
                            timeout_seconds=sample_timeout_seconds,
                        )
                    return _episode(
                        row,
                        body,
                        name,
                        replay_index=replay_index,
                    )

                wait_a = run_episode("wait_only", replay_index=0)
                wait_b = run_episode("wait_only", replay_index=1)
                episodes = {
                    "wait_only": wait_a,
                    "random": run_episode("random"),
                    "greedy_heuristic": run_episode("greedy_heuristic"),
                    "oracle_offline": run_episode("oracle_offline"),
                }
        deterministic = wait_a == wait_b
        wait_fingerprint_first = hashlib.sha256(
            json.dumps(
                wait_a,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        wait_fingerprint_second = hashlib.sha256(
            json.dumps(
                wait_b,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        oracle_headroom = episodes["oracle_offline"]["raw_total"] - wait_a["raw_total"]
        checks = {
            "deterministic_replay": deterministic,
        }
        return _classify_result(
            {
                "scenario_id": row["scenario_id"],
                "scenario_signature": row["scenario_signature"],
                "admission_profile": resolve_protocol21_admission_profile(row),
                "path": row["path"],
                "domain": row.get("domain", "power_grid"),
                "backend_kind": row["backend_kind"],
                "family": row["family"],
                "difficulty_mode": row["difficulty_mode"],
                "difficulty_level": row["difficulty_level"],
                "status": "pending_classification",
                "checks": checks,
                "replay_evidence": {
                    "wait_fingerprint_first": wait_fingerprint_first,
                    "wait_fingerprint_second": wait_fingerprint_second,
                    "wait_differences": (
                        []
                        if deterministic
                        else _determinism_differences(wait_a, wait_b)
                    ),
                    "source_consumption_first": wait_a.get(
                        "source_consumption_evidence"
                    ),
                    "source_consumption_second": wait_b.get(
                        "source_consumption_evidence"
                    ),
                },
                "metrics": {"oracle_wait_raw_headroom": round(oracle_headroom, 9)},
                "episodes": episodes,
            }
        )
    except Exception as exc:
        return {
            "scenario_id": row["scenario_id"],
            "scenario_signature": row.get("scenario_signature"),
            "admission_profile": resolve_protocol21_admission_profile(row),
            "path": row["path"],
            "domain": row.get("domain", "power_grid"),
            "backend_kind": row["backend_kind"],
            "family": row["family"],
            "difficulty_mode": row["difficulty_mode"],
            "difficulty_level": row["difficulty_level"],
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _calibrate_one_child(
    connection: Any,
    row: dict[str, Any],
    cache_dir: Path | None,
) -> None:
    """Run one sample in a disposable process so native solvers are killable."""
    _claim_process_group()
    try:
        connection.send(_calibrate_one(row, cache_dir, None))
    finally:
        connection.close()


def _behavioral_worker_exit_result(
    row: dict[str, Any], process: Any, *, pipe_state: str = "eof"
) -> dict[str, Any]:
    process.join(timeout=NATURAL_REAP_SECONDS)
    still_alive = bool(process.is_alive())
    exitcode = getattr(process, "exitcode", None)
    signal_name = None
    if isinstance(exitcode, int) and exitcode < 0:
        try:
            signal_name = signal.Signals(-exitcode).name
        except ValueError:
            signal_name = "UNKNOWN"
    details = f"exitcode={exitcode}"
    if signal_name is not None:
        details += f", signal={signal_name}"
    error = (
        "NativeProcessProtocolError: behavioral worker closed its result pipe "
        "but remained alive after bounded natural reap"
        if still_alive
        else (
            "NativeProcessExit: behavioral worker exited without a result "
            f"({details})"
        )
    )
    return {
        **row,
        "status": "error",
        "error": error,
        "worker_exit_evidence": {
            "pipe_state": pipe_state,
            "natural_reap": "still_alive" if still_alive else "exited",
            "exitcode": exitcode,
            "signal": signal_name,
            "cleanup_forced": False,
        },
    }


def _record_worker_cleanup(result: dict[str, Any], process: Any) -> None:
    forced = _stop_isolated_process(process)
    evidence = result.get("worker_exit_evidence")
    if isinstance(evidence, dict):
        evidence["cleanup_forced"] = forced


def _calibrate_one_isolated(
    row: dict[str, Any],
    cache_dir: Path | None,
    sample_timeout_seconds: int | None,
) -> dict[str, Any]:
    if not sample_timeout_seconds or sample_timeout_seconds <= 0:
        return _calibrate_one(row, cache_dir, sample_timeout_seconds)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_calibrate_one_child,
        args=(child, row, cache_dir),
        daemon=False,
    )
    process.start()
    child.close()
    result: dict[str, Any] = {}
    try:
        if parent.poll(float(sample_timeout_seconds)):
            try:
                result = parent.recv()
            except EOFError:
                result = _behavioral_worker_exit_result(row, process)
        else:
            result = {
                "scenario_id": row["scenario_id"],
                "scenario_signature": row.get("scenario_signature"),
                "path": row.get("path"),
                "domain": row.get("domain", "power_grid"),
                "backend_kind": row.get("backend_kind"),
                "family": row.get("family"),
                "difficulty_mode": row.get("difficulty_mode"),
                "difficulty_level": row.get("difficulty_level"),
                "status": "error",
                "error": (
                    "TimeoutError: behavioral sample exceeded "
                    f"{sample_timeout_seconds}s in isolated process"
                ),
                "worker_exit_evidence": {
                    "pipe_state": "timeout",
                    "natural_reap": "still_alive",
                    "exitcode": getattr(process, "exitcode", None),
                    "signal": None,
                    "cleanup_forced": False,
                },
            }
    finally:
        parent.close()
        _record_worker_cleanup(result, process)
    return result


def _calibrate_one_dispatched(
    row: dict[str, Any],
    cache_dir: Path | None,
    sample_timeout_seconds: int | None,
) -> dict[str, Any]:
    """Use a killable child only for native solver/runtime backends."""
    if str(row.get("backend_kind") or "") in HARD_ISOLATION_BACKENDS or (
        sample_timeout_seconds
        and sample_timeout_seconds > 0
        and not hasattr(signal, "SIGALRM")
    ):
        return _calibrate_one_isolated(
            row,
            cache_dir,
            sample_timeout_seconds,
        )
    return _calibrate_one(row, cache_dir, sample_timeout_seconds)


def _run_hard_isolated_batch(
    rows: list[dict[str, Any]],
    *,
    cache_dir: Path | None,
    workers: int,
    sample_timeout_seconds: int | None,
    on_result: Any,
) -> None:
    """Run native backends in one killable process layer with checkpoints."""
    context = multiprocessing.get_context("spawn")
    pending = deque(rows)
    active: dict[int, tuple[Any, Any, dict[str, Any], float]] = {}

    def terminate(process: Any, result: dict[str, Any]) -> None:
        _record_worker_cleanup(result, process)

    try:
        while pending or active:
            while pending and len(active) < max(1, workers):
                row = pending.popleft()
                parent, child = context.Pipe(duplex=False)
                process = context.Process(
                    target=_calibrate_one_child,
                    args=(child, row, cache_dir),
                    daemon=False,
                )
                process.start()
                child.close()
                active[int(process.pid)] = (
                    process,
                    parent,
                    row,
                    time.monotonic(),
                )

            progressed = False
            for pid, (process, parent, row, started) in list(active.items()):
                result = None
                if parent.poll():
                    try:
                        result = parent.recv()
                    except EOFError:
                        result = _behavioral_worker_exit_result(row, process)
                elif not process.is_alive():
                    result = _behavioral_worker_exit_result(row, process)
                elif (
                    sample_timeout_seconds
                    and sample_timeout_seconds > 0
                    and time.monotonic() - started >= sample_timeout_seconds
                ):
                    result = {
                        **row,
                        "status": "error",
                        "error": (
                            "TimeoutError: behavioral sample exceeded "
                            f"{sample_timeout_seconds}s in isolated process"
                        ),
                        "worker_exit_evidence": {
                            "pipe_state": "timeout",
                            "natural_reap": "still_alive",
                            "exitcode": getattr(process, "exitcode", None),
                            "signal": None,
                            "cleanup_forced": False,
                        },
                    }
                if result is None:
                    continue
                terminate(process, result)
                parent.close()
                del active[pid]
                on_result(result)
                progressed = True
            if not progressed and active:
                time.sleep(0.05)
    finally:
        for process, parent, _row, _started in active.values():
            _stop_isolated_process(process)
            parent.close()


def _write_report(
    path: Path,
    results: dict[str, dict[str, Any]],
    expected: int,
    *,
    implementation_tree_sha256: str,
    core_release_pipeline_sha256: str,
    admission_profile: str,
) -> None:
    ordered = [results[key] for key in sorted(results)]
    counts: dict[str, int] = {}
    for row in ordered:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = {
        "schema_version": "0.3",
        "audit_episode_contract_version": AUDIT_EPISODE_CONTRACT_VERSION,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "evaluation_implementation_fingerprint": (
            EVALUATION_IMPLEMENTATION_FINGERPRINT
        ),
        "scoring_version": SCORING_VERSION,
        "evaluation_semantics": {
            "protocol_version": EVALUATION_PROTOCOL_VERSION,
            "implementation_fingerprint": (EVALUATION_IMPLEMENTATION_FINGERPRINT),
            "scoring_version": SCORING_VERSION,
        },
        "implementation_tree_sha256": implementation_tree_sha256,
        "core_release_pipeline_sha256": core_release_pipeline_sha256,
        "admission_profile": admission_profile,
        "suite_id": "operate_v0_58_0_candidate",
        "status": "complete" if len(ordered) == expected else "partial",
        "n_expected": expected,
        "n_completed": len(ordered),
        "status_counts": dict(sorted(counts.items())),
        "results": ordered,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _result_with_inherited_admission_profile(
    result: dict[str, Any],
    *,
    admission_profile: str,
    artifact_kind: str,
) -> dict[str, Any]:
    declared_profile = declared_protocol21_admission_profile(result)
    if declared_profile not in (None, admission_profile):
        scenario_id = str(result.get("scenario_id") or "<missing-scenario-id>")
        raise ValueError(
            f"behavioral calibration {artifact_kind} row admission profile mismatch: "
            f"{scenario_id}: {declared_profile} != {admission_profile}"
        )
    return {**result, "admission_profile": admission_profile}


def calibrate(
    suite_path: Path,
    output_path: Path,
    workers: int,
    limit: int | None,
    cache_dir: Path | None = DEFAULT_CACHE,
    scenario_ids: set[str] | None = None,
    domains: set[str] | None = None,
    levels: set[str] | None = None,
    sample_timeout_seconds: int | None = 600,
    import_result_paths: list[Path] | None = None,
) -> dict[str, Any]:
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    admission_profile = resolve_protocol21_admission_profile(suite)
    identity = implementation_identity()
    implementation_tree_sha256 = identity["implementation_tree_sha256"]
    core_release_pipeline_sha256 = identity["core_release_pipeline_sha256"]
    rows = _filter_rows(
        list(suite["scenarios"]),
        scenario_ids=scenario_ids,
        domains=domains,
        levels=levels,
        limit=limit,
    )
    row_profile_mismatches = [
        str(row.get("scenario_id") or "")
        for row in rows
        if declared_protocol21_admission_profile(row) not in (None, admission_profile)
    ]
    if row_profile_mismatches:
        raise ValueError(
            "behavioral row admission profile mismatch: "
            + ", ".join(sorted(row_profile_mismatches))
        )
    rows = [{**row, "admission_profile": admission_profile} for row in rows]
    desired_ids = {str(row["scenario_id"]) for row in rows}
    desired_signatures = {
        str(row["scenario_id"]): str(row["scenario_signature"]) for row in rows
    }
    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            prior.get("schema_version") == "0.3"
            and prior.get("audit_episode_contract_version")
            == AUDIT_EPISODE_CONTRACT_VERSION
            and prior.get("evaluation_protocol_version") == EVALUATION_PROTOCOL_VERSION
            and prior.get("evaluation_implementation_fingerprint")
            == EVALUATION_IMPLEMENTATION_FINGERPRINT
            and prior.get("scoring_version") == SCORING_VERSION
            and prior.get("implementation_tree_sha256") == implementation_tree_sha256
            and prior.get("core_release_pipeline_sha256")
            == core_release_pipeline_sha256
            and resolve_protocol21_admission_profile(prior) == admission_profile
        ):
            prior_results = [
                _result_with_inherited_admission_profile(
                    row,
                    admission_profile=admission_profile,
                    artifact_kind="resume",
                )
                for row in prior.get("results", [])
            ]
            existing = {
                row["scenario_id"]: _classify_result(row)
                for row in prior_results
                if row["scenario_id"] in desired_ids
                and row.get("scenario_signature")
                == desired_signatures[row["scenario_id"]]
                and row.get("status") in {"passed", "failed"}
                and all(
                    "effective_tool_names" in (episode or {})
                    and "task_completion" in (episode or {})
                    and "world_evolution" in (episode or {})
                    and "terminal_integrity" in (episode or {})
                    for episode in (row.get("episodes") or {}).values()
                )
            }
    for import_path in import_result_paths or []:
        imported = json.loads(import_path.read_text(encoding="utf-8"))
        if not (
            imported.get("schema_version") == "0.3"
            and imported.get("audit_episode_contract_version")
            == AUDIT_EPISODE_CONTRACT_VERSION
            and imported.get("evaluation_protocol_version")
            == EVALUATION_PROTOCOL_VERSION
            and imported.get("evaluation_implementation_fingerprint")
            == EVALUATION_IMPLEMENTATION_FINGERPRINT
            and imported.get("scoring_version") == SCORING_VERSION
            and imported.get("implementation_tree_sha256") == implementation_tree_sha256
            and imported.get("core_release_pipeline_sha256")
            == core_release_pipeline_sha256
            and resolve_protocol21_admission_profile(imported) == admission_profile
        ):
            raise ValueError(f"behavioral calibration import is stale: {import_path}")
        imported_results = [
            _result_with_inherited_admission_profile(
                result,
                admission_profile=admission_profile,
                artifact_kind=f"import {import_path}",
            )
            for result in imported.get("results") or []
        ]
        for result in imported_results:
            scenario_id = str(result.get("scenario_id") or "")
            if (
                scenario_id in desired_ids
                and str(result.get("scenario_signature") or "")
                == desired_signatures[scenario_id]
                and result.get("status") in {"passed", "failed"}
                and all(
                    "effective_tool_names" in (episode or {})
                    and "task_completion" in (episode or {})
                    and "world_evolution" in (episode or {})
                    and "terminal_integrity" in (episode or {})
                    for episode in (result.get("episodes") or {}).values()
                )
            ):
                existing.setdefault(
                    scenario_id,
                    _classify_result(result),
                )
    pending = [row for row in rows if row["scenario_id"] not in existing]
    hard_pending = [
        row
        for row in pending
        if str(row.get("backend_kind") or "") in HARD_ISOLATION_BACKENDS
        or (
            sample_timeout_seconds
            and sample_timeout_seconds > 0
            and not hasattr(signal, "SIGALRM")
        )
    ]
    hard_ids = {str(row["scenario_id"]) for row in hard_pending}
    safe_pending = [row for row in pending if str(row["scenario_id"]) not in hard_ids]

    def save_result(result: dict[str, Any]) -> None:
        existing[str(result["scenario_id"])] = result
        _write_report(
            output_path,
            existing,
            len(rows),
            implementation_tree_sha256=implementation_tree_sha256,
            core_release_pipeline_sha256=core_release_pipeline_sha256,
            admission_profile=admission_profile,
        )

    if workers <= 1:
        for row in pending:
            save_result(
                _calibrate_one_dispatched(row, cache_dir, sample_timeout_seconds)
            )
    else:
        if safe_pending:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _calibrate_one_dispatched,
                        row,
                        cache_dir,
                        sample_timeout_seconds,
                    ): row
                    for row in safe_pending
                }
                try:
                    for future in as_completed(futures):
                        save_result(future.result())
                except BaseException:
                    for future in futures:
                        future.cancel()
                    terminate_workers = getattr(executor, "terminate_workers", None)
                    if callable(terminate_workers):
                        terminate_workers()
                    else:
                        for process in (
                            getattr(executor, "_processes", None) or {}
                        ).values():
                            process.terminate()
                    raise
        bounded_hard_pending = [
            row
            for row in hard_pending
            if str(row.get("backend_kind") or "") in BOUNDED_ISOLATION_WORKERS
        ]
        parallel_hard_pending = [
            row
            for row in hard_pending
            if str(row.get("backend_kind") or "") not in BOUNDED_ISOLATION_WORKERS
        ]
        if parallel_hard_pending:
            _run_hard_isolated_batch(
                parallel_hard_pending,
                cache_dir=cache_dir,
                workers=workers,
                sample_timeout_seconds=sample_timeout_seconds,
                on_result=save_result,
            )
        if bounded_hard_pending:
            _run_hard_isolated_batch(
                bounded_hard_pending,
                cache_dir=cache_dir,
                workers=min(workers, BOUNDED_ISOLATION_WORKERS["sumo"]),
                sample_timeout_seconds=sample_timeout_seconds,
                on_result=save_result,
            )
    _write_report(
        output_path,
        existing,
        len(rows),
        implementation_tree_sha256=implementation_tree_sha256,
        core_release_pipeline_sha256=core_release_pipeline_sha256,
        admission_profile=admission_profile,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--scenario-ids-file", type=Path)
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--levels", nargs="+")
    parser.add_argument("--sample-timeout-seconds", type=int, default=600)
    parser.add_argument("--import-results", type=Path, action="append")
    args = parser.parse_args()
    scenario_ids = None
    if args.scenario_ids_file:
        values = json.loads(args.scenario_ids_file.read_text(encoding="utf-8"))
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError("--scenario-ids-file must contain a JSON string array")
        scenario_ids = set(values)
    report = calibrate(
        args.suite.resolve(),
        args.output.resolve(),
        args.workers,
        args.limit,
        None if args.no_cache else args.cache_dir.resolve(),
        scenario_ids=scenario_ids,
        domains=set(args.domains) if args.domains else None,
        levels=set(args.levels) if args.levels else None,
        sample_timeout_seconds=args.sample_timeout_seconds,
        import_result_paths=(
            [path.resolve() for path in args.import_results]
            if args.import_results
            else None
        ),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "n_expected", "n_completed", "status_counts")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
