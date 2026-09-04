#!/usr/bin/env python3
"""Validate task contracts from behavioral evidence or direct reference replay."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import signal
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audit._common import _resolve_scenario_path  # noqa: E402
from audit.episode_cache import AUDIT_EPISODE_CONTRACT_VERSION  # noqa: E402
from core.implementation_identity import implementation_identity  # noqa: E402
from core.material_headroom import (  # noqa: E402
    TRAFFIC_NATIVE_SIGNAL_HEADROOM_V2,
    build_traffic_native_signal_headroom_v2,
    material_headroom_from_task_completion,
)
from evaluation import SCORING_VERSION  # noqa: E402
from evaluation.task_completion import task_completion_contract  # noqa: E402
from run import run_one  # noqa: E402
from runner import (  # noqa: E402
    EVALUATION_IMPLEMENTATION_FINGERPRINT,
    EVALUATION_PROTOCOL_VERSION,
)
from scripts.calibrate_core_candidate import (  # noqa: E402
    HARD_ISOLATION_BACKENDS,
    _claim_process_group,
    _stop_isolated_process,
)

BOUNDED_ISOLATION_WORKERS = {"sumo": 1}


def _result_from_episode_summary(
    row: dict[str, Any],
    agent_name: str,
    *,
    completion: dict[str, Any],
    terminal_integrity: dict[str, Any],
    evaluation_protocol: dict[str, Any],
    scoring_version: str | None,
    evidence_source: str,
) -> dict[str, Any]:
    terminal_ready = bool(terminal_integrity.get("release_ready"))
    task_completed = bool(completion.get("completed"))
    process_capability_met = bool(
        completion.get("process_capability_applicable") is not True
        or completion.get("process_capability_satisfied") is True
    )
    evidence = completion.get("evidence") or {}
    if completion.get("contract") == TRAFFIC_NATIVE_SIGNAL_HEADROOM_V2:
        material_headroom = build_traffic_native_signal_headroom_v2(
            baseline_metrics=dict(evidence.get("baseline_metrics") or {}),
            baseline_repeat_metrics=dict(
                evidence.get("baseline_repeat_metrics") or {}
            ),
            reference_metrics=dict(evidence.get("reference_metrics") or {}),
            reference_repeat_metrics=dict(
                evidence.get("reference_repeat_metrics") or {}
            ),
            native_control_effect=bool(evidence.get("native_control_effect")),
            safety=evidence.get("safety"),
        )
    else:
        material_headroom = material_headroom_from_task_completion(completion)
    native_material_ready = (
        completion.get("contract") != TRAFFIC_NATIVE_SIGNAL_HEADROOM_V2
        or material_headroom.get("status") == "passed"
    )
    status = (
        "passed"
        if (
            task_completed
            and process_capability_met
            and terminal_ready
            and native_material_ready
        )
        else "failed"
    )
    if not task_completed:
        reason_code = completion.get("reason_code")
    elif not process_capability_met:
        reason_code = "reference_process_capability_unsatisfied"
    elif not terminal_ready:
        reason_code = (
            "terminal_integrity_failure"
            if terminal_integrity
            else "terminal_integrity_missing"
        )
    elif not native_material_ready:
        reason_code = material_headroom.get("reason_code")
    else:
        reason_code = completion.get("reason_code")
    return {
        "scenario_id": row["scenario_id"],
        "scenario_signature": row.get("scenario_signature"),
        "domain": row.get("domain"),
        "family": row.get("family"),
        "difficulty_level": row.get("difficulty_level"),
        "agent_name": agent_name,
        "status": status,
        "applicable": bool(completion.get("applicable")),
        "completed": task_completed,
        "process_capability_applicable": bool(
            completion.get("process_capability_applicable")
        ),
        "process_capability_satisfied": completion.get(
            "process_capability_satisfied"
        ),
        "process_capability_checks": completion.get("process_capability_checks")
        or {},
        "contract": completion.get("contract"),
        "reason_code": reason_code,
        "task_reason_code": completion.get("reason_code"),
        "evidence": evidence,
        "material_headroom": material_headroom,
        "terminal_integrity": terminal_integrity,
        "evaluation_protocol": evaluation_protocol,
        "scoring_version": scoring_version,
        "evidence_source": evidence_source,
    }


def _run(row: dict[str, Any], agent_name: str) -> dict[str, Any]:
    try:
        body = yaml.safe_load(
            _resolve_scenario_path(str(row["path"])).read_text(encoding="utf-8")
        )
        episode = run_one(body, agent_name=agent_name)
        completion = episode.get("task_completion") or {}
        terminal_integrity = (episode.get("trajectory_summary") or {}).get(
            "terminal_integrity"
        ) or {}
        return _result_from_episode_summary(
            row,
            agent_name,
            completion=completion,
            terminal_integrity=terminal_integrity,
            evaluation_protocol=episode.get("evaluation_protocol") or {},
            scoring_version=(episode.get("score") or {}).get("scoring_version"),
            evidence_source="backend_replay",
        )
    except Exception as exc:
        return {
            "scenario_id": row["scenario_id"],
            "scenario_signature": row.get("scenario_signature"),
            "agent_name": agent_name,
            "status": "error",
            "applicable": False,
            "completed": False,
            "reason_code": "reference_episode_error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_reference_agents(
    row: dict[str, Any], agent_names: list[str]
) -> dict[str, Any]:
    attempts: list[dict[str, str]] = []
    selected: dict[str, Any] | None = None
    for agent_name in agent_names:
        result = _run(row, agent_name)
        attempts.append(
            {"agent_name": agent_name, "status": str(result.get("status") or "error")}
        )
        if selected is None or (
            selected.get("status") == "error" and result.get("status") != "error"
        ):
            selected = result
        if result.get("status") == "passed":
            selected = result
            break
    assert selected is not None
    return {**selected, "reference_agents_attempted": attempts}


@contextmanager
def _sample_timeout(seconds: int | None):
    if not seconds or seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"task-contract sample exceeded {seconds}s")

    prior_handler = signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, prior_handler)


def _reference_error(
    row: dict[str, Any],
    agent_names: list[str],
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "scenario_id": row["scenario_id"],
        "scenario_signature": row.get("scenario_signature"),
        "domain": row.get("domain"),
        "family": row.get("family"),
        "difficulty_level": row.get("difficulty_level"),
        "agent_name": agent_names[0],
        "status": "error",
        "applicable": False,
        "completed": False,
        "reason_code": "reference_episode_error",
        "error": f"{type(exc).__name__}: {exc}",
        "reference_agents_attempted": [],
    }


def _derive_reference_agents_from_behavioral(
    row: dict[str, Any],
    behavioral: dict[str, Any],
    agent_names: list[str],
    *,
    evaluation_protocol: dict[str, Any],
    scoring_version: str,
) -> dict[str, Any]:
    attempts: list[dict[str, str]] = []
    selected: dict[str, Any] | None = None
    episodes = behavioral.get("episodes") or {}
    for agent_name in agent_names:
        summary = episodes.get(agent_name)
        if not isinstance(summary, dict) or not isinstance(
            summary.get("task_completion"), dict
        ):
            result = _reference_error(
                row,
                agent_names,
                RuntimeError(
                    f"behavioral episode evidence missing for {agent_name}"
                ),
            )
            result["reason_code"] = "behavioral_reference_evidence_missing"
            result["evidence_source"] = "behavioral_episode"
        else:
            result = _result_from_episode_summary(
                row,
                agent_name,
                completion=dict(summary["task_completion"]),
                terminal_integrity=dict(summary.get("terminal_integrity") or {}),
                evaluation_protocol=evaluation_protocol,
                scoring_version=scoring_version,
                evidence_source="behavioral_episode",
            )
        attempts.append(
            {"agent_name": agent_name, "status": str(result.get("status") or "error")}
        )
        if selected is None or (
            selected.get("status") == "error" and result.get("status") != "error"
        ):
            selected = result
        if result.get("status") == "passed":
            selected = result
            break
    assert selected is not None
    selected["behavioral_scenario_signature"] = behavioral.get(
        "scenario_signature"
    )
    return {**selected, "reference_agents_attempted": attempts}


def _run_reference_agents_child(
    connection: Any,
    row: dict[str, Any],
    agent_names: list[str],
) -> None:
    _claim_process_group()
    try:
        connection.send(_run_reference_agents(row, agent_names))
    finally:
        connection.close()


def _run_reference_agents_isolated(
    row: dict[str, Any],
    agent_names: list[str],
    sample_timeout_seconds: int | None,
) -> dict[str, Any]:
    if not sample_timeout_seconds or sample_timeout_seconds <= 0:
        return _run_reference_agents(row, agent_names)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_reference_agents_child,
        args=(child, row, agent_names),
        daemon=False,
    )
    process.start()
    child.close()
    try:
        if parent.poll(float(sample_timeout_seconds)):
            try:
                return parent.recv()
            except EOFError:
                pass
        return {
            "scenario_id": row["scenario_id"],
            "scenario_signature": row.get("scenario_signature"),
            "domain": row.get("domain"),
            "family": row.get("family"),
            "difficulty_level": row.get("difficulty_level"),
            "agent_name": agent_names[0],
            "status": "error",
            "applicable": False,
            "completed": False,
            "reason_code": "reference_episode_timeout",
            "error": (
                "TimeoutError: task-contract sample exceeded "
                f"{sample_timeout_seconds}s in isolated process"
            ),
            "reference_agents_attempted": [],
        }
    finally:
        parent.close()
        _stop_isolated_process(process)


def _run_reference_agents_dispatched(
    row: dict[str, Any],
    agent_names: list[str],
    sample_timeout_seconds: int | None,
) -> dict[str, Any]:
    """Isolate native solvers while reusing workers for safe Python backends."""
    if str(row.get("backend_kind") or "") in HARD_ISOLATION_BACKENDS or (
        sample_timeout_seconds
        and sample_timeout_seconds > 0
        and not hasattr(signal, "SIGALRM")
    ):
        return _run_reference_agents_isolated(
            row,
            agent_names,
            sample_timeout_seconds,
        )
    try:
        with _sample_timeout(sample_timeout_seconds):
            return _run_reference_agents(row, agent_names)
    except Exception as exc:
        return _reference_error(row, agent_names, exc)


def _run_hard_isolated_batch(
    rows: list[dict[str, Any]],
    *,
    agent_names: list[str],
    workers: int,
    sample_timeout_seconds: int | None,
    on_result: Any,
) -> None:
    """Run native task contracts in one killable process layer."""
    context = multiprocessing.get_context("spawn")
    queued = deque(rows)
    active: dict[int, tuple[Any, Any, dict[str, Any], float]] = {}

    try:
        while queued or active:
            while queued and len(active) < max(1, workers):
                row = queued.popleft()
                parent, child = context.Pipe(duplex=False)
                process = context.Process(
                    target=_run_reference_agents_child,
                    args=(child, row, agent_names),
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
                        result = _reference_error(
                            row,
                            agent_names,
                            RuntimeError(
                                "task-contract worker exited without a result"
                            ),
                        )
                elif not process.is_alive():
                    result = _reference_error(
                        row,
                        agent_names,
                        RuntimeError("task-contract worker exited without a result"),
                    )
                elif (
                    sample_timeout_seconds
                    and sample_timeout_seconds > 0
                    and time.monotonic() - started >= sample_timeout_seconds
                ):
                    result = _reference_error(
                        row,
                        agent_names,
                        TimeoutError(
                            "task-contract sample exceeded "
                            f"{sample_timeout_seconds}s in isolated process"
                        ),
                    )
                    result["reason_code"] = "reference_episode_timeout"
                if result is None:
                    continue
                _stop_isolated_process(process)
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


def _write(
    path: Path,
    results: dict[str, dict[str, Any]],
    expected: int,
    reference_agent_names: list[str],
    *,
    implementation_tree_sha256: str,
    core_release_pipeline_sha256: str,
) -> None:
    ordered = [results[key] for key in sorted(results)]
    report = {
        "schema_version": "1.0",
        "status": "complete" if len(ordered) == expected else "partial",
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
        "n_expected": expected,
        "n_completed": len(ordered),
        "n_passed": sum(row["status"] == "passed" for row in ordered),
        "n_failed": sum(row["status"] == "failed" for row in ordered),
        "n_errors": sum(row["status"] == "error" for row in ordered),
        "admission_policy": "reference_policy_must_complete_else_replace_or_retire",
        "reference_agent_names": reference_agent_names,
        "results": ordered,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _definitive_reference_failure(
    row: dict[str, Any], reference_agent_names: list[str]
) -> bool:
    attempts = list(row.get("reference_agents_attempted") or [])
    return (
        row.get("status") == "failed"
        and [str(item.get("agent_name")) for item in attempts] == reference_agent_names
        and all(item.get("status") == "failed" for item in attempts)
    )


def calibrate(
    suite_path: Path,
    output_path: Path,
    *,
    agent_name: str = "oracle_offline",
    workers: int = 1,
    scenario_ids: set[str] | None = None,
    eligible_results_path: Path | None = None,
    fallback_agent_names: list[str] | None = None,
    sample_timeout_seconds: int | None = None,
) -> dict[str, Any]:
    identity = implementation_identity()
    implementation_tree_sha256 = identity["implementation_tree_sha256"]
    core_release_pipeline_sha256 = identity["core_release_pipeline_sha256"]
    reference_agent_names = list(
        dict.fromkeys([agent_name, *(fallback_agent_names or [])])
    )
    rows = list(json.loads(suite_path.read_text(encoding="utf-8"))["scenarios"])
    behavioral_results: dict[str, dict[str, Any]] = {}
    behavioral_evaluation_protocol = {
        "version": EVALUATION_PROTOCOL_VERSION,
        "implementation_fingerprint": EVALUATION_IMPLEMENTATION_FINGERPRINT,
    }
    behavioral_scoring_version = SCORING_VERSION
    if eligible_results_path is not None:
        eligibility = json.loads(eligible_results_path.read_text(encoding="utf-8"))
        if eligibility.get("schema_version") != "0.3":
            raise ValueError("eligible results require behavioral schema 0.3")
        if eligibility.get("status") != "complete":
            raise ValueError("eligible behavioral results must be complete")
        if (
            eligibility.get("implementation_tree_sha256") != implementation_tree_sha256
            or eligibility.get("core_release_pipeline_sha256")
            != core_release_pipeline_sha256
        ):
            raise ValueError("eligible behavioral results are stale")
        if (
            eligibility.get("evaluation_protocol_version")
            != EVALUATION_PROTOCOL_VERSION
            or eligibility.get("evaluation_implementation_fingerprint")
            != EVALUATION_IMPLEMENTATION_FINGERPRINT
            or eligibility.get("scoring_version") != SCORING_VERSION
        ):
            raise ValueError("eligible behavioral evaluation semantics are stale")
        eligible_ids = {
            str(row["scenario_id"])
            for row in eligibility.get("results") or []
            if row.get("status") == "passed"
            and bool((row.get("checks") or {}).get("native_state_changing_leverage"))
        }
        behavioral_results = {
            str(row["scenario_id"]): row
            for row in eligibility.get("results") or []
            if str(row.get("scenario_id") or "") in eligible_ids
        }
        rows = [row for row in rows if str(row.get("scenario_id")) in eligible_ids]
    if scenario_ids:
        rows = [row for row in rows if str(row.get("scenario_id")) in scenario_ids]
    desired = {str(row["scenario_id"]) for row in rows}
    expected_contracts = {
        str(row["scenario_id"]): task_completion_contract(
            str(row.get("domain") or "unknown"),
            str(row.get("family") or "unknown"),
        )
        for row in rows
    }
    expected_signatures = {
        str(row["scenario_id"]): str(row.get("scenario_signature") or "")
        for row in rows
    }
    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        prior = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            prior.get("schema_version") == "1.0"
            and prior.get("implementation_tree_sha256") == implementation_tree_sha256
            and prior.get("core_release_pipeline_sha256")
            == core_release_pipeline_sha256
        ):
            existing = {
                str(row["scenario_id"]): row
                for row in prior.get("results") or []
                if str(row.get("scenario_id")) in desired
                and str(row.get("scenario_signature") or "")
                == expected_signatures[str(row.get("scenario_id"))]
                and row.get("agent_name") in reference_agent_names
                and (
                    row.get("status") == "passed"
                    or _definitive_reference_failure(row, reference_agent_names)
                )
                and row.get("contract")
                == expected_contracts[str(row.get("scenario_id"))]
                and bool((row.get("terminal_integrity") or {}).get("release_ready"))
                and (row.get("evaluation_protocol") or {}).get("version")
                == EVALUATION_PROTOCOL_VERSION
                and (row.get("evaluation_protocol") or {}).get(
                    "implementation_fingerprint"
                )
                == EVALUATION_IMPLEMENTATION_FINGERPRINT
                and row.get("scoring_version") == SCORING_VERSION
            }
    pending = [row for row in rows if str(row["scenario_id"]) not in existing]

    def save_result(result: dict[str, Any]) -> None:
        existing[str(result["scenario_id"])] = result
        _write(
            output_path,
            existing,
            len(rows),
            reference_agent_names,
            implementation_tree_sha256=implementation_tree_sha256,
            core_release_pipeline_sha256=core_release_pipeline_sha256,
        )

    if eligible_results_path is not None:
        for row in pending:
            scenario_id = str(row["scenario_id"])
            behavioral = behavioral_results.get(scenario_id)
            if behavioral is None or str(
                behavioral.get("scenario_signature") or ""
            ) != expected_signatures[scenario_id]:
                result = _reference_error(
                    row,
                    reference_agent_names,
                    RuntimeError("behavioral scenario evidence missing or stale"),
                )
                result["reason_code"] = "behavioral_reference_evidence_missing"
                result["evidence_source"] = "behavioral_episode"
                save_result(result)
                continue
            save_result(
                _derive_reference_agents_from_behavioral(
                    row,
                    behavioral,
                    reference_agent_names,
                    evaluation_protocol=behavioral_evaluation_protocol,
                    scoring_version=behavioral_scoring_version,
                )
            )
        pending = []

    if workers <= 1:
        for row in pending:
            save_result(
                _run_reference_agents_dispatched(
                    row,
                    reference_agent_names,
                    sample_timeout_seconds,
                )
            )
    else:
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
        safe_pending = [
            row for row in pending if str(row["scenario_id"]) not in hard_ids
        ]
        if safe_pending:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _run_reference_agents_dispatched,
                        row,
                        reference_agent_names,
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
                agent_names=reference_agent_names,
                workers=workers,
                sample_timeout_seconds=sample_timeout_seconds,
                on_result=save_result,
            )
        if bounded_hard_pending:
            _run_hard_isolated_batch(
                bounded_hard_pending,
                agent_names=reference_agent_names,
                workers=min(workers, BOUNDED_ISOLATION_WORKERS["sumo"]),
                sample_timeout_seconds=sample_timeout_seconds,
                on_result=save_result,
            )
    _write(
        output_path,
        existing,
        len(rows),
        reference_agent_names,
        implementation_tree_sha256=implementation_tree_sha256,
        core_release_pipeline_sha256=core_release_pipeline_sha256,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent", default="oracle_offline")
    parser.add_argument("--fallback-agents", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-timeout-seconds", type=int, default=600)
    parser.add_argument("--scenario-ids-file", type=Path)
    parser.add_argument(
        "--eligible-results",
        type=Path,
        help=(
            "Complete behavioral schema-0.3 report; derive task contracts for "
            "native passes without replaying the backend."
        ),
    )
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
        agent_name=args.agent,
        workers=args.workers,
        scenario_ids=scenario_ids,
        fallback_agent_names=args.fallback_agents,
        sample_timeout_seconds=args.sample_timeout_seconds,
        eligible_results_path=(
            args.eligible_results.resolve() if args.eligible_results else None
        ),
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "n_expected",
                    "n_completed",
                    "n_passed",
                    "n_failed",
                    "n_errors",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
