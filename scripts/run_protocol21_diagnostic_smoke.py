#!/usr/bin/env python3
"""Run a strict baseline smoke over the non-release 40-row diagnostic slice.

This runner intentionally calls the same single-episode path as formal
evaluation, but writes only a diagnostic report.  It never updates a Core,
manifest, leaderboard, or release artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import signal
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import (  # noqa: E402
    canonicalize_repo_owned_paths,
)
from core.scenario_validator import validate_scenario_yaml  # noqa: E402
from runner.episode import run_one  # noqa: E402

DEFAULT_SLICE = REPO_ROOT / (
    "release/operate_v0_58_0_candidate/operate_v058_formal/diagnostic/"
    "diagnostic_slice.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "release/operate_v0_58_0_candidate/operate_v058_formal/diagnostic/smoke"
)
DEFAULT_AGENTS = ("wait_only", "random", "greedy_heuristic", "oracle_offline")
REQUIRED_RESULT_KEYS = (
    "scenario_id",
    "scenario_signature",
    "score",
    "counterfactual",
    "trajectory_summary",
    "task_completion",
)


class EpisodeTimeout(TimeoutError):
    """Raised when one diagnostic episode exceeds its explicit wall-time cap."""


@contextmanager
def _episode_timeout(seconds: float):
    if seconds <= 0:
        raise ValueError("episode timeout must be positive")
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("episode timeout requires POSIX SIGALRM support")

    def _raise_timeout(_signum: int, _frame: object) -> None:
        raise EpisodeTimeout(f"episode exceeded {seconds:g} seconds")

    prior_handler = signal.signal(signal.SIGALRM, _raise_timeout)
    prior_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *prior_timer)
        signal.signal(signal.SIGALRM, prior_handler)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("diagnostic slice path must be inside the repository") from exc


def _load_bound_scenario(row: dict[str, Any]) -> dict[str, Any]:
    """Load the exact repo-relative YAML bound by the diagnostic slice."""
    declared_path = Path(str(row.get("path") or ""))
    if not declared_path.as_posix() or declared_path.is_absolute():
        raise ValueError("diagnostic scenario path must be repo-relative")
    root = REPO_ROOT.resolve()
    path = (root / declared_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("diagnostic scenario path escapes repository") from exc
    if not path.is_file() or path.suffix != ".yaml":
        raise FileNotFoundError(f"scenario not found: {declared_path.as_posix()}")
    scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(scenario, dict):
        raise ValueError("diagnostic scenario YAML must contain an object")
    errors = validate_scenario_yaml(scenario, source_path=path)
    if str(scenario.get("domain") or "") == "traffic":
        from domains.traffic.scenario_validation import validate_traffic_scenario_yaml

        errors.extend(validate_traffic_scenario_yaml(scenario))
    if errors:
        raise ValueError(f"scenario YAML validation failed: {'; '.join(errors)}")
    expected_id = str(row.get("scenario_id") or "")
    if str(scenario.get("scenario_id") or "") != expected_id:
        raise ValueError("yaml_scenario_identity_mismatch")
    return scenario


def _coverage_errors(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [key for key in REQUIRED_RESULT_KEYS if key not in result]
    if missing:
        errors.append(f"missing_result_keys:{','.join(missing)}")
    trajectory = result.get("trajectory_summary")
    if not isinstance(trajectory, dict):
        return errors + ["trajectory_summary_missing"]
    coverage = trajectory.get("tool_semantic_coverage")
    if not isinstance(coverage, dict):
        errors.append("tool_semantic_coverage_missing")
    else:
        if coverage.get("covered") is not True:
            errors.append("tool_semantic_coverage_not_covered")
        if coverage.get("unknown_tool_names"):
            errors.append("unknown_tools")
        if coverage.get("unclassified_tool_names"):
            errors.append("unclassified_tools")
    counterfactual = result.get("counterfactual")
    if not isinstance(counterfactual, dict):
        errors.append("counterfactual_missing")
    else:
        for prefix in ("per_action", "per_action_group"):
            expected = counterfactual.get(f"{prefix}_expected")
            if (
                counterfactual.get(f"{prefix}_status") != "complete"
                or not isinstance(expected, int)
                or isinstance(expected, bool)
                or expected < 0
                or counterfactual.get(f"{prefix}_attempted") != expected
                or counterfactual.get(f"{prefix}_completed") != expected
                or counterfactual.get(f"{prefix}_failures") != []
            ):
                errors.append(f"{prefix}_attribution_incomplete")
        if counterfactual.get("per_action_capped") is not False:
            errors.append("per_action_attribution_capped")
    if not isinstance(result.get("task_completion"), dict):
        errors.append("task_completion_missing")
    input_binding = result.get("diagnostic_input_binding")
    if not isinstance(input_binding, dict) or input_binding.get("verified") is not True:
        errors.append("diagnostic_input_binding_unverified")
    records = trajectory.get("event_response_records")
    profile = trajectory.get("operational_agency_profile")
    if not isinstance(records, list):
        errors.append("event_response_records_missing")
    if not isinstance(profile, dict):
        errors.append("operational_agency_profile_missing")
    else:
        if profile.get("schema_version") != "operational_agency_profile_v1":
            errors.append("operational_agency_profile_schema")
        if profile.get("runtime_evidence_binding_verified") is not True:
            errors.append("runtime_evidence_binding_unverified")
        if profile.get("masked_replay_binding_verified") is not True:
            errors.append("masked_replay_binding_unverified")
        if isinstance(records, list) and profile.get("event_response_record_count") != len(records):
            errors.append("event_response_record_count_mismatch")
    terminal = trajectory.get("terminal_integrity")
    if not isinstance(terminal, dict) or terminal.get("release_ready") is not True:
        errors.append("terminal_integrity_not_ready")
    runtime = result.get("diagnostic_runtime_integrity")
    tree = result.get("implementation_tree_sha256")
    if (
        not isinstance(runtime, dict)
        or not isinstance(tree, str)
        or not tree
        or runtime.get("implementation_tree_stable") is not True
        or runtime.get("implementation_tree_sha256_start") != tree
        or runtime.get("implementation_tree_sha256_end") != tree
    ):
        errors.append("implementation_tree_drift")
    if not isinstance(runtime, dict) or runtime.get("process_check_available") is not True:
        errors.append("orphan_process_check_unavailable")
    elif runtime.get("orphan_pids") != []:
        errors.append("orphan_processes")
    return errors


def _strict_errors(result: dict[str, Any]) -> list[str]:
    errors = _coverage_errors(result)
    ground_truth = result.get("ground_truth_summary") or {}
    if bool(ground_truth.get("chose_fatal_option")):
        errors.append("fatal_option")
    if result.get("status") == "error":
        errors.append("episode_error")
    return errors


def validate_result(
    result: dict[str, Any], *, check_profile: str = "strict",
) -> dict[str, Any]:
    """Return machine-readable smoke checks for one episode result."""
    errors = _strict_errors(result)
    strict_errors = list(errors)
    warnings: list[str] = []
    if check_profile not in {"strict", "runtime_installation"}:
        raise ValueError(f"unknown check profile: {check_profile}")
    if check_profile == "runtime_installation":
        if result.get("agent_name") != "wait_only":
            raise ValueError("runtime_installation requires wait_only")
        terminal = (result.get("trajectory_summary") or {}).get("terminal_integrity")
        if (
            result.get("status") == "ok"
            and isinstance(terminal, dict)
            and terminal.get("release_ready") is False
            and terminal.get("unresolved_pending_actions") == {}
            and terminal.get("terminal_feedback_reasons") == []
            and isinstance(terminal.get("unanswered_interrupt_reasons"), list)
            and bool(terminal["unanswered_interrupt_reasons"])
            and all(isinstance(reason, str) and reason for reason in terminal["unanswered_interrupt_reasons"])
            and "terminal_integrity_not_ready" in errors
        ):
            errors.remove("terminal_integrity_not_ready")
            warnings.append("wait_only_unanswered_interrupts")
    return {
        "passed": not errors,
        "errors": errors,
        "check_profile": check_profile,
        "strict_errors": strict_errors,
        "warnings": warnings,
        "unknown_tool_names": list(
            ((result.get("trajectory_summary") or {}).get("tool_semantic_coverage") or {}).get(
                "unknown_tool_names", []
            )
        ),
        "unclassified_tool_names": list(
            ((result.get("trajectory_summary") or {}).get("tool_semantic_coverage") or {}).get(
                "unclassified_tool_names", []
            )
        ),
        "fatal": bool((result.get("ground_truth_summary") or {}).get("chose_fatal_option")),
    }


def _sumo_process_ids(*, pgid: int | None = None) -> tuple[set[int], bool]:
    """Inspect only SUMO children in the isolated episode's process group."""
    try:
        owner_pgid = os.getpgrp() if pgid is None else pgid
        completed = subprocess.run(
            ["pgrep", "-f", r"(^|/)sumo( |$)"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return set(), False
    if completed.returncode not in (0, 1):
        return set(), False
    pids = {
        int(value)
        for value in completed.stdout.split()
        if value.isdigit()
    }
    owned: set[int] = set()
    for pid in pids:
        try:
            if os.getpgid(pid) == owner_pgid:
                owned.add(pid)
        except ProcessLookupError:
            # A process that exited between inventory and lookup is not orphaned.
            continue
        except OSError:
            return set(), False
    return owned, True


def _run_episode(
    row: dict[str, Any],
    agent: str,
    seed: int,
    *,
    timeout_seconds: float,
    expected_implementation_tree_sha256: str | None = None,
) -> dict[str, Any]:
    path = str(row.get("path") or "")
    if not path:
        return {
            "status": "error",
            "agent_name": agent,
            "seed": seed,
            "scenario_id": row.get("scenario_id"),
            "error": "scenario_path_missing",
        }
    expected_id = str(row.get("scenario_id") or "")
    expected_signature = str(row.get("scenario_signature") or "")
    start_tree = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    expected_tree = expected_implementation_tree_sha256 or start_tree
    before_pids, before_check_available = _sumo_process_ids()
    binding: dict[str, Any] = {
        "verified": False,
        "expected_scenario_id": expected_id,
        "expected_scenario_signature": expected_signature,
        "runtime_scenario_id": None,
        "runtime_scenario_signature": None,
        "yaml_scenario_id": None,
        "yaml_seed_id": None,
        "yaml_scenario_id_verified": False,
        "expected_implementation_tree_sha256": expected_tree,
    }
    result: dict[str, Any]
    try:
        if start_tree != expected_tree:
            raise RuntimeError("implementation_tree_changed_before_episode")
        with _episode_timeout(timeout_seconds):
            scenario = _load_bound_scenario(row)
            binding["yaml_scenario_id"] = scenario.get("scenario_id")
            binding["yaml_seed_id"] = scenario.get("seed_id")
            binding["yaml_scenario_id_verified"] = scenario.get("scenario_id") == expected_id
            result = run_one(
                scenario,
                agent,
                seed_override=seed,
                per_action_attribution=True,
                per_action_cap=None,
                per_action_group_attribution=True,
                per_action_group_cap=None,
                within_tick_interaction=True,
            )
        binding["runtime_scenario_id"] = result.get("scenario_id")
        binding["runtime_scenario_signature"] = result.get("scenario_signature")
        binding["verified"] = (
            binding["yaml_scenario_id_verified"] is True
            and result.get("scenario_id") == scenario.get("seed_id")
            and result.get("scenario_signature") == expected_signature
        )
        result["diagnostic_input_binding"] = binding
        if binding["verified"] is not True:
            raise ValueError("runtime_scenario_identity_mismatch")
        result["scenario_id"] = expected_id
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - diagnostic runner must ledger failures
        result = {
            "status": "error",
            "agent_name": agent,
            "seed": seed,
            "scenario_id": row.get("scenario_id"),
            "scenario_signature": row.get("scenario_signature"),
            "error": f"{type(exc).__name__}: {exc}",
            "diagnostic_input_binding": binding,
        }
    end_tree = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    after_pids, after_check_available = _sumo_process_ids()
    orphan_pids = sorted(after_pids.difference(before_pids))
    stable = start_tree == expected_tree == end_tree
    result["implementation_tree_sha256"] = expected_tree
    result["diagnostic_runtime_integrity"] = {
        "implementation_tree_sha256_start": start_tree,
        "implementation_tree_sha256_end": end_tree,
        "implementation_tree_stable": stable,
        "process_check_available": (
            before_check_available and after_check_available
        ),
        "orphan_pids": orphan_pids,
    }
    if result.get("status") == "ok" and (not stable or orphan_pids):
        result["status"] = "error"
        result["error"] = (
            "implementation_tree_drift"
            if not stable
            else f"orphan_processes:{orphan_pids}"
        )
    return result


def _isolated_episode_worker(
    connection: Any,
    runner: Callable[..., dict[str, Any]],
    row: dict[str, Any],
    agent: str,
    seed: int,
    timeout_seconds: float,
    expected_implementation_tree_sha256: str,
) -> None:
    if hasattr(os, "setsid"):
        os.setsid()
    try:
        connection.send(
            runner(
                row,
                agent,
                seed,
                timeout_seconds=timeout_seconds,
                expected_implementation_tree_sha256=(
                    expected_implementation_tree_sha256
                ),
            )
        )
    finally:
        connection.close()


def _terminate_isolated_process(process: mp.Process) -> None:
    if not process.is_alive():
        process.join()
        return
    used_process_group = False
    if process.pid is not None and hasattr(os, "killpg"):
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGTERM)
                used_process_group = True
        except (OSError, ProcessLookupError):
            pass
    if not used_process_group:
        process.terminate()
    process.join(timeout=2.0)
    if not process.is_alive():
        return
    if used_process_group and process.pid is not None:
        with suppress(OSError, ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()
    process.join(timeout=2.0)


def _run_episode_isolated(
    row: dict[str, Any],
    agent: str,
    seed: int,
    *,
    timeout_seconds: float,
    expected_implementation_tree_sha256: str,
    runner: Callable[..., dict[str, Any]] = _run_episode,
) -> dict[str, Any]:
    """Run one episode in a killable process so native calls obey the timeout."""
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_episode_worker,
        args=(
            send,
            runner,
            row,
            agent,
            seed,
            timeout_seconds,
            expected_implementation_tree_sha256,
        ),
    )
    process.start()
    send.close()
    try:
        if receive.poll(timeout_seconds):
            result = receive.recv()
            process.join(timeout=2.0)
            if process.is_alive():
                _terminate_isolated_process(process)
            if isinstance(result, dict):
                return result
            error = "isolated_episode_result_not_object"
        else:
            _terminate_isolated_process(process)
            error = f"EpisodeTimeout: episode exceeded {timeout_seconds:g} seconds"
    except EOFError:
        _terminate_isolated_process(process)
        error = f"isolated_episode_worker_exit:{process.exitcode}"
    finally:
        receive.close()

    end_tree = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    # The worker calls setsid before launching native backends; its PID is the
    # owned PGID even after worker exit. Other simultaneous runs are unrelated.
    after_pids, after_check_available = _sumo_process_ids(pgid=process.pid)
    return {
        "status": "error",
        "agent_name": agent,
        "seed": seed,
        "scenario_id": row.get("scenario_id"),
        "scenario_signature": row.get("scenario_signature"),
        "error": error,
        "diagnostic_input_binding": {
            "verified": False,
            "expected_scenario_id": row.get("scenario_id"),
            "expected_scenario_signature": row.get("scenario_signature"),
            "expected_implementation_tree_sha256": (
                expected_implementation_tree_sha256
            ),
        },
        "implementation_tree_sha256": expected_implementation_tree_sha256,
        "diagnostic_runtime_integrity": {
            "implementation_tree_sha256_start": (
                expected_implementation_tree_sha256
            ),
            "implementation_tree_sha256_end": end_tree,
            "implementation_tree_stable": (
                end_tree == expected_implementation_tree_sha256
            ),
            "process_check_available": after_check_available,
            "orphan_pids": sorted(after_pids),
        },
    }


def _numeric(result: dict[str, Any], *keys: str) -> float | None:
    current: Any = result
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    try:
        return float(current) if current is not None else None
    except (TypeError, ValueError):
        return None


def build_report(
    *,
    slice_payload: dict[str, Any],
    results: Iterable[dict[str, Any]],
    requested_agents: list[str],
    repeats: int,
    check_profile: str = "strict",
) -> dict[str, Any]:
    if check_profile == "runtime_installation" and requested_agents != ["wait_only"]:
        raise ValueError("runtime_installation requires only wait_only")
    rows = list(results)
    checks = [validate_result(result, check_profile=check_profile) for result in rows]
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result, check in zip(rows, checks, strict=True):
        result["diagnostic_smoke_check"] = check
        by_agent[str(result.get("agent_name") or "unknown")].append(result)
    agent_summary: dict[str, Any] = {}
    for agent in requested_agents:
        agent_rows = by_agent.get(agent, [])
        scores = [
            value
            for value in (_numeric(row, "score", "total_score") for row in agent_rows)
            if value is not None
        ]
        prevented = [
            value
            for value in (_numeric(row, "counterfactual", "prevented_loss") for row in agent_rows)
            if value is not None
        ]
        agent_summary[agent] = {
            "n_episodes": len(agent_rows),
            "n_ok": sum(row.get("status") == "ok" for row in agent_rows),
            "n_failed": sum(row.get("status") != "ok" for row in agent_rows),
            "n_fatal": sum(
                check["fatal"] for check in (validate_result(row) for row in agent_rows)
            ),
            "n_incomplete": sum(not row["diagnostic_smoke_check"]["passed"] for row in agent_rows),
            "mean_total_score": statistics.fmean(scores) if scores else None,
            "mean_prevented_loss": statistics.fmean(prevented) if prevented else None,
        }
    expected = len(slice_payload.get("scenarios") or []) * len(requested_agents) * repeats
    check_failures = [
        {
            "scenario_id": row.get("scenario_id"),
            "agent_name": row.get("agent_name"),
            "errors": check["errors"],
        }
        for row, check in zip(rows, checks, strict=True)
        if not check["passed"]
    ]
    strict_failures = [
        {"scenario_id": row.get("scenario_id"), "agent_name": row.get("agent_name"),
         "errors": check["strict_errors"]}
        for row, check in zip(rows, checks, strict=True) if check["strict_errors"]
    ]
    return {
        "schema_version": "protocol21-diagnostic-smoke-v1",
        "status": "passed"
        if len(rows) == expected and not check_failures
        else "blocked_quality_gate",
        "diagnostic_only": True,
        "release_admission": False,
        "check_profile": check_profile,
        "model_success_claimed": False,
        "n_check_failures": len(check_failures),
        "check_failures": check_failures,
        "strict_prompt": True,
        "provider_fallback": False,
        "slice_schema_version": slice_payload.get("schema_version"),
        "slice_status": slice_payload.get("status"),
        "n_scenarios": len(slice_payload.get("scenarios") or []),
        "requested_agents": requested_agents,
        "repeats": repeats,
        "n_expected": expected,
        "n_results": len(rows),
        "n_strict_failures": len(strict_failures),
        "strict_failures": strict_failures,
        "agent_summary": agent_summary,
        "known_groups_calibration": {
            "adaptive_gt_reactive": "not_run_no_llm_agent_in_smoke",
            "adaptive_plan_gt_open_loop": "not_run_no_llm_agent_in_smoke",
            "full_observation_ge_partial": "not_applicable_to_single_slice",
            "wait_random_agency_near_zero": "diagnostic_check_requires_event_response_records",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, default=DEFAULT_SLICE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--agents", nargs="+", default=list(DEFAULT_AGENTS))
    parser.add_argument("--check-profile", choices=["strict", "runtime_installation"], default="strict")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--episode-timeout-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.check_profile == "runtime_installation" and args.agents != ["wait_only"]:
        parser.error("runtime_installation requires --agents wait_only")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.max_workers != 1:
        parser.error("diagnostic smoke requires --max-workers=1 for native backend determinism")
    if args.episode_timeout_seconds <= 0:
        parser.error("--episode-timeout-seconds must be positive")
    payload = _load(args.slice)
    rows = payload.get("scenarios")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        parser.error("slice must contain a scenarios list")
    implementation_start = implementation_identity(REPO_ROOT)[
        "implementation_tree_sha256"
    ]
    results: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for row in rows:
            seed = int(row.get("seed", 42)) + repeat
            for agent in args.agents:
                results.append(
                    _run_episode_isolated(
                        row,
                        str(agent),
                        seed,
                        timeout_seconds=args.episode_timeout_seconds,
                        expected_implementation_tree_sha256=implementation_start,
                    )
                )
    implementation_end = implementation_identity(REPO_ROOT)[
        "implementation_tree_sha256"
    ]
    portable_results = canonicalize_repo_owned_paths(
        results,
        repo_root=REPO_ROOT,
    )
    report = build_report(
        slice_payload=payload,
        results=portable_results,
        requested_agents=[str(agent) for agent in args.agents],
        repeats=args.repeats,
        check_profile=args.check_profile,
    )
    report["input_bindings"] = {
        "slice": {
            "path": _repo_relative(args.slice),
            "sha256": hashlib.sha256(args.slice.read_bytes()).hexdigest(),
        }
    }
    report["implementation_binding"] = {
        "start": implementation_start,
        "end": implementation_end,
        "stable": implementation_start == implementation_end,
    }
    if implementation_start != implementation_end:
        report["status"] = "blocked_quality_gate"
        report["n_check_failures"] += 1
        report["check_failures"].append({
            "scenario_id": None, "agent_name": None,
            "errors": ["implementation_tree_drift"],
        })
        report["n_strict_failures"] += 1
        report["strict_failures"].append(
            {
                "scenario_id": None,
                "agent_name": None,
                "errors": ["implementation_tree_drift"],
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "episodes.jsonl").write_text(
        "".join(
            json.dumps(result, sort_keys=True) + "\n"
            for result in portable_results
        ),
        encoding="utf-8",
    )
    (args.output_dir / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_results": report["n_results"],
                "n_expected": report["n_expected"],
                "n_strict_failures": report["n_strict_failures"],
                "check_profile": report["check_profile"],
                "n_check_failures": report["n_check_failures"],
                "model_success_claimed": False,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
