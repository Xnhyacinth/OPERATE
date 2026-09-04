#!/usr/bin/env python3
"""Plan and execute isolated Protocol-2.1 candidate work batches.

The coordinator never mutates its input queue or the locked Core.  Commands
come from the queue, are grouped into deterministic domain/backend shards, and
produce one immutable coordinator result per work item before a ledger is
assembled.  Scientific admission remains the responsibility of the shared
Protocol-2.1 pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from evaluation import SCORING_VERSION  # noqa: E402

STAGES = (
    "inventory",
    "conversion",
    "static_preflight",
    "native_prefilter",
    "full_protocol21",
    "evidence_freeze",
    "final_union",
)
WORK_STATES = (
    "pending",
    "running",
    "passed",
    "failed_retryable",
    "terminal",
)
DISPOSITIONS = (
    "core_locked_increment",
    "held_repair",
    "held_runtime",
    "held_license_or_terms",
    "transfer_only",
    "secondary_duplicate",
    "retired_intrinsic",
)
SCHEDULABLE_STATES = {"pending", "failed_retryable"}
DOMAIN_PRIORITY = {
    "power_grid": 0,
    "traffic": 1,
    "microgrid": 2,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: Any, *, length: int | None = None) -> str:
    digest = hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _scenario_rows(payload: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    for key in ("scenarios", "core_scenarios", "rows", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            if not all(isinstance(row, dict) for row in value):
                raise ValueError(f"{path}: {key} must contain only objects")
            return value
    raise ValueError(f"{path}: scenario rows missing")


def _identity(row: dict[str, Any]) -> tuple[str, str] | None:
    scenario_id = str(row.get("scenario_id") or "")
    signature = str(row.get("scenario_signature") or "")
    if not scenario_id and not signature:
        return None
    if not scenario_id or not signature:
        raise ValueError("scenario identity must include id and signature")
    return scenario_id, signature


def _resource_tokens(backend: str) -> int:
    normalized = backend.lower()
    if "sumo" in normalized or "sidecar" in normalized:
        return 4
    if any(name in normalized for name in ("pandapower", "grid2op", "opendss")):
        return 2
    return 1


def _dependency_shard(item: dict[str, Any]) -> str:
    explicit = item.get("dependency_shard")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError("dependency_shard must be a non-empty string")
        return explicit.strip()
    identity = _identity(item)
    if identity is None:
        return f"work:{item['work_id']}"
    return _digest(
        {
            "domain": item["domain"],
            "backend": item["backend"],
            "identity": identity,
        }
    )


def _validate_item(item: dict[str, Any]) -> None:
    work_id = item.get("work_id")
    if not isinstance(work_id, str) or not work_id:
        raise ValueError("queue work_id must be a non-empty string")
    if item.get("stage") not in STAGES:
        raise ValueError(f"{work_id}: invalid stage")
    work_state = item.get("work_state")
    if work_state not in WORK_STATES:
        raise ValueError(f"{work_id}: invalid work_state")
    disposition = item.get("disposition")
    if disposition is not None and disposition not in DISPOSITIONS:
        raise ValueError(f"{work_id}: invalid disposition")
    if work_state == "terminal" and disposition is None:
        raise ValueError(f"{work_id}: terminal work requires a disposition")
    for key in ("domain", "backend"):
        if not isinstance(item.get(key), str) or not item[key]:
            raise ValueError(f"{work_id}: {key} must be a non-empty string")
    _identity(item)
    _dependency_shard(item)
    if work_state in SCHEDULABLE_STATES:
        command = item.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) and part for part in command)
        ):
            raise ValueError(f"{work_id}: schedulable work requires command argv")


def build_candidate_plan(
    queue_path: Path,
    base_core_path: Path,
    output_root: Path,
    *,
    io_workers: int = 32,
    conversion_shard_size: int = 64,
    runtime_resource_tokens: int = 4,
) -> dict[str, Any]:
    """Build a deterministic plan without running candidate commands."""
    if io_workers < 1:
        raise ValueError("io_workers must be >= 1")
    if conversion_shard_size < 1:
        raise ValueError("conversion_shard_size must be >= 1")
    if runtime_resource_tokens < 1:
        raise ValueError("runtime_resource_tokens must be >= 1")
    queue_path = queue_path.resolve()
    base_core_path = base_core_path.resolve()
    output_root = output_root.resolve()
    queue = _load_object(queue_path)
    if queue.get("schema_version") != "candidate-batch-queue-v1":
        raise ValueError("unsupported candidate queue schema_version")
    items = queue.get("items")
    if not isinstance(items, list):
        raise ValueError("candidate queue items missing")
    if not all(isinstance(item, dict) for item in items):
        raise ValueError("candidate queue items must be objects")
    for item in items:
        _validate_item(item)
    work_ids = [str(item["work_id"]) for item in items]
    if len(set(work_ids)) != len(work_ids):
        raise ValueError("candidate queue work_id values must be unique")

    base_payload = _load_object(base_core_path)
    locked_identities = {
        identity
        for row in _scenario_rows(base_payload, base_core_path)
        if (identity := _identity(row)) is not None
    }
    scheduled: list[dict[str, Any]] = []
    locked_skips: list[dict[str, str]] = []
    for item in sorted(items, key=lambda value: str(value["work_id"])):
        identity = _identity(item)
        if identity is not None and identity in locked_identities:
            locked_skips.append(
                {
                    "work_id": str(item["work_id"]),
                    "scenario_id": identity[0],
                    "scenario_signature": identity[1],
                    "reason": "exact_identity_already_in_locked_core",
                }
            )
            continue
        if item["work_state"] in SCHEDULABLE_STATES:
            scheduled.append(item)

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in scheduled:
        key = (str(item["stage"]), str(item["domain"]), str(item["backend"]))
        groups.setdefault(key, []).append(item)
    shards: list[dict[str, Any]] = []
    ordered_groups = sorted(
        groups.items(),
        key=lambda entry: (
            STAGES.index(entry[0][0]),
            DOMAIN_PRIORITY.get(entry[0][1], len(DOMAIN_PRIORITY)),
            entry[0][1],
            entry[0][2],
        ),
    )
    for (stage, domain, backend), group in ordered_groups:
        ordered = sorted(group, key=lambda value: str(value["work_id"]))
        shard_size = conversion_shard_size if stage == "conversion" else len(ordered)
        for offset in range(0, len(ordered), shard_size):
            chunk = ordered[offset : offset + shard_size]
            identity_payload = {
                "stage": stage,
                "domain": domain,
                "backend": backend,
                "work": [
                    {
                        "work_id": item["work_id"],
                        "identity": _identity(item),
                        "command": item["command"],
                    }
                    for item in chunk
                ],
            }
            shard_id = f"{stage}-{domain}-{backend}-{_digest(identity_payload, length=16)}"
            shards.append(
                {
                    "shard_id": shard_id,
                    "stage": stage,
                    "domain": domain,
                    "backend": backend,
                    "resource_tokens": _resource_tokens(backend),
                    "items": chunk,
                }
            )
    implementation_sha = implementation_identity()["implementation_tree_sha256"]
    semantic_queue = sorted(items, key=lambda value: str(value["work_id"]))
    return {
        "schema_version": "protocol21-candidate-batch-plan-v1",
        "status": "planned",
        "release_admission": False,
        "queue_path": str(queue_path),
        "queue_sha256": _digest(
            {
                "schema_version": queue["schema_version"],
                "items": semantic_queue,
            }
        ),
        "queue_artifact_sha256": _sha256(queue_path),
        "base_core_path": str(base_core_path),
        "base_core_sha256": _sha256(base_core_path),
        "implementation_tree_sha256": implementation_sha,
        "scoring_version": SCORING_VERSION,
        "runtime_version": {"python": platform.python_version()},
        "output_root": str(output_root),
        "io_workers": io_workers,
        "conversion_shard_size": conversion_shard_size,
        "runtime_resource_tokens": runtime_resource_tokens,
        "n_queue_items": len(items),
        "n_scheduled": len(scheduled),
        "n_core_locked_skipped": len(locked_skips),
        "core_locked_skips": locked_skips,
        "shards": shards,
    }


class _WeightedSemaphore:
    def __init__(self, capacity: int) -> None:
        self._available = capacity
        self._condition = threading.Condition()

    def acquire(self, weight: int) -> None:
        with self._condition:
            self._condition.wait_for(lambda: self._available >= weight)
            self._available -= weight

    def release(self, weight: int) -> None:
        with self._condition:
            self._available += weight
            self._condition.notify_all()


def _run_item(
    item: dict[str, Any],
    *,
    shard: dict[str, Any],
    result_path: Path,
    semaphore: _WeightedSemaphore,
    timeout_seconds: int,
    max_retries: int,
    execution_binding_sha256: str,
) -> dict[str, Any]:
    weight = int(shard["resource_tokens"])
    semaphore.acquire(weight)
    try:
        attempts = 0
        timed_out = False
        completed: subprocess.CompletedProcess[str] | None = None
        while attempts <= max_retries:
            attempts += 1
            try:
                completed = subprocess.run(
                    item["command"],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
                timed_out = False
                break
            except subprocess.TimeoutExpired:
                timed_out = True
        passed = completed is not None and completed.returncode == 0
        if passed:
            work_state = "passed"
            disposition = None
        elif timed_out:
            work_state = "terminal"
            disposition = "held_runtime"
        else:
            # A completed scientific gate failure is terminal for this queue
            # item. It must be repaired explicitly rather than retried with
            # identical scientific inputs.
            work_state = "terminal"
            disposition = item.get("disposition") or "held_repair"
        result = {
            "schema_version": "protocol21-candidate-work-result-v1",
            "work_id": item["work_id"],
            "shard_id": shard["shard_id"],
            "stage": item["stage"],
            "domain": item["domain"],
            "backend": item["backend"],
            "command": item["command"],
            "command_sha256": _digest(item["command"]),
            "execution_binding_sha256": execution_binding_sha256,
            "attempts": attempts,
            "timed_out": timed_out,
            "command_return_code": completed.returncode if completed else None,
            "stdout": completed.stdout if completed else "",
            "stderr": completed.stderr if completed else "",
            "work_state": work_state,
            "disposition": disposition,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return {**result, "result_path": str(result_path), "reused": False}
    finally:
        semaphore.release(weight)


def _reuse_result(
    path: Path,
    item: dict[str, Any],
    shard_id: str,
    execution_binding_sha256: str,
) -> dict[str, Any]:
    result = _load_object(path)
    if result.get("work_id") != item["work_id"]:
        raise RuntimeError(f"resume result work_id mismatch: {path}")
    if result.get("shard_id") != shard_id:
        raise RuntimeError(f"resume result shard mismatch: {path}")
    if result.get("command_sha256") != _digest(item["command"]):
        raise RuntimeError(f"resume result command mismatch: {path}")
    if result.get("execution_binding_sha256") != execution_binding_sha256:
        raise RuntimeError(f"resume result execution binding mismatch: {path}")
    return {**result, "result_path": str(path), "reused": True}


def _assert_plan_bindings(plan: dict[str, Any]) -> None:
    """Reject a plan as soon as any queue/core/code binding drifts."""
    queue_path = Path(str(plan.get("queue_path") or ""))
    base_core_path = Path(str(plan.get("base_core_path") or ""))
    if not queue_path.is_file() or not base_core_path.is_file():
        raise RuntimeError("candidate plan input artifact is missing")
    if _sha256(queue_path) != plan.get("queue_artifact_sha256"):
        raise RuntimeError("candidate plan queue artifact hash drift")
    if _sha256(base_core_path) != plan.get("base_core_sha256"):
        raise RuntimeError("candidate plan base Core hash drift")
    live_tree = implementation_identity()["implementation_tree_sha256"]
    if live_tree != plan.get("implementation_tree_sha256"):
        raise RuntimeError("candidate plan implementation tree drift")


def _write_dependency_barrier_result(
    *,
    item: dict[str, Any],
    shard: dict[str, Any],
    result_path: Path,
    execution_binding_sha256: str,
    upstream: dict[str, Any],
) -> dict[str, Any]:
    """Record one unstarted dependent item without affecting other shards."""
    disposition = (
        "held_runtime"
        if upstream.get("disposition") == "held_runtime"
        else "held_repair"
    )
    reason = (
        "dependency_barrier_after_"
        f"{upstream.get('stage')}:{upstream.get('work_id')}"
    )
    result = {
        "schema_version": "protocol21-candidate-work-result-v1",
        "work_id": item["work_id"],
        "shard_id": shard["shard_id"],
        "stage": item["stage"],
        "domain": item["domain"],
        "backend": item["backend"],
        "command": item["command"],
        "command_sha256": _digest(item["command"]),
        "execution_binding_sha256": execution_binding_sha256,
        "attempts": 0,
        "timed_out": False,
        "command_return_code": None,
        "stdout": "",
        "stderr": reason,
        "work_state": "terminal",
        "disposition": disposition,
        "terminal_reason": reason,
        "blocked_by": {
            "work_id": upstream.get("work_id"),
            "stage": upstream.get("stage"),
            "disposition": upstream.get("disposition"),
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return {**result, "result_path": str(result_path), "reused": False}


def execute_plan(
    plan: dict[str, Any],
    *,
    resume: bool,
    timeout_seconds: int,
    max_retries: int,
) -> dict[str, Any]:
    """Execute plan commands and assemble a ledger after all isolated writes."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be > 0")
    if not 0 <= max_retries <= 2:
        raise ValueError("max_retries must be between 0 and 2")
    _assert_plan_bindings(plan)
    output_root = Path(str(plan["output_root"]))
    semaphore = _WeightedSemaphore(int(plan["runtime_resource_tokens"]))
    completed_results: list[dict[str, Any]] = []
    tasks_by_stage: dict[
        str, list[tuple[dict[str, Any], dict[str, Any], Path, str, str]]
    ] = {}
    reusable_by_stage: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for shard in plan["shards"]:
        if int(shard["resource_tokens"]) > int(plan["runtime_resource_tokens"]):
            raise ValueError(f"{shard['shard_id']}: resource weight exceeds capacity")
        for item in shard["items"]:
            execution_binding_sha256 = _digest(
                {
                    "queue_sha256": plan["queue_sha256"],
                    "base_core_sha256": plan["base_core_sha256"],
                    "implementation_tree_sha256": plan["implementation_tree_sha256"],
                    "runtime_version": plan["runtime_version"],
                    "scoring_version": plan["scoring_version"],
                    "shard_id": shard["shard_id"],
                    "work_id": item["work_id"],
                    "command": item["command"],
                }
            )
            result_path = (
                output_root
                / "results"
                / str(shard["shard_id"])
                / f"{_digest(str(item['work_id']))}.json"
            )
            dependency_shard = _dependency_shard(item)
            if result_path.exists():
                if not resume:
                    raise FileExistsError(f"immutable result already exists: {result_path}")
                reusable_by_stage.setdefault(str(shard["stage"]), []).append(
                    (
                        _reuse_result(
                            result_path,
                            item,
                            str(shard["shard_id"]),
                            execution_binding_sha256,
                        ),
                        dependency_shard,
                    )
                )
                continue
            tasks_by_stage.setdefault(str(shard["stage"]), []).append(
                (
                    shard,
                    item,
                    result_path,
                    execution_binding_sha256,
                    dependency_shard,
                )
            )

    blocked_dependencies: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=int(plan["io_workers"])) as executor:
        for stage in STAGES:
            _assert_plan_bindings(plan)
            reused = reusable_by_stage.get(stage, [])
            reused_stage_results = [result for result, _dependency in reused]
            completed_results.extend(reused_stage_results)
            runnable = []
            barrier_results: list[dict[str, Any]] = []
            for task in tasks_by_stage.get(stage, []):
                shard, item, result_path, binding_sha256, dependency_shard = task
                upstream = blocked_dependencies.get(dependency_shard)
                if upstream is None:
                    runnable.append(task)
                    continue
                barrier_results.append(
                    _write_dependency_barrier_result(
                        item=item,
                        shard=shard,
                        result_path=result_path,
                        execution_binding_sha256=binding_sha256,
                        upstream=upstream,
                    )
                )
            completed_results.extend(barrier_results)
            futures = [
                executor.submit(
                    _run_item,
                    item,
                    shard=shard,
                    result_path=result_path,
                    semaphore=semaphore,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    execution_binding_sha256=binding_sha256,
                )
                for shard, item, result_path, binding_sha256, _dependency in runnable
            ]
            stage_results = [future.result() for future in as_completed(futures)]
            completed_results.extend(stage_results)
            current_dependencies = {
                str(item["work_id"]): dependency
                for _shard, item, _path, _binding, dependency in runnable
            }
            for result, dependency_shard in [
                *reused,
                *[
                    (result, current_dependencies[str(result["work_id"])])
                    for result in stage_results
                ],
            ]:
                if result.get("work_state") == "terminal":
                    blocked_dependencies.setdefault(dependency_shard, result)
            _assert_plan_bindings(plan)
    completed_results.sort(key=lambda item: str(item["work_id"]))
    n_passed = sum(item["work_state"] == "passed" for item in completed_results)
    n_held = sum(item["work_state"] == "terminal" for item in completed_results)
    if n_held == 0:
        status = "complete"
    elif n_passed == 0:
        status = "held"
    else:
        status = "partial"
    dispositions = Counter(
        str(item["disposition"])
        for item in completed_results
        if item.get("disposition") is not None
    )
    ledger = {
        "schema_version": "protocol21-candidate-batch-ledger-v1",
        "status": status,
        "release_admission": False,
        "queue_sha256": plan["queue_sha256"],
        "base_core_sha256": plan["base_core_sha256"],
        "implementation_tree_sha256": plan["implementation_tree_sha256"],
        "scoring_version": plan["scoring_version"],
        "runtime_version": plan["runtime_version"],
        "n_scheduled": plan["n_scheduled"],
        "n_core_locked_skipped": plan["n_core_locked_skipped"],
        "summary": {
            "n_passed": n_passed,
            "n_held": n_held,
            "dispositions": dict(sorted(dispositions.items())),
        },
        "items": completed_results,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "candidate_batch_ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ledger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--base-core", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--io-workers", type=int, default=32)
    parser.add_argument("--conversion-shard-size", type=int, default=64)
    parser.add_argument("--runtime-resource-tokens", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        plan = build_candidate_plan(
            args.queue,
            args.base_core,
            args.output_root,
            io_workers=args.io_workers,
            conversion_shard_size=args.conversion_shard_size,
            runtime_resource_tokens=args.runtime_resource_tokens,
        )
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            print("NO_COMMANDS_EXECUTED=true")
            return 0
        ledger = execute_plan(
            plan,
            resume=args.resume,
            timeout_seconds=args.timeout_seconds,
            max_retries=args.max_retries,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": ledger["status"], "n_scheduled": ledger["n_scheduled"]}))
    return 0 if ledger["status"] == "complete" else 4


if __name__ == "__main__":
    raise SystemExit(main())
