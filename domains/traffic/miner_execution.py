"""Bounded deterministic process execution for native Traffic mining."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

MIN_WORKERS = 1
MAX_WORKERS = 4


class MinerExecutionError(ValueError):
    pass


def semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_worker_count(workers: int) -> int:
    if workers < MIN_WORKERS or workers > MAX_WORKERS:
        raise MinerExecutionError(
            "traffic_parallel_workers_out_of_range"
        )
    return workers


def _canonical_key(row: dict[str, Any]) -> tuple[Any, ...]:
    semantic = row["semantic_input"]
    return (
        int(row["wave_index"]),
        str(semantic["service_date"]),
        str(semantic["complete_source_identity_sha256"]),
        str(semantic["tls_id"]),
        str(semantic["control_type"]),
        semantic_sha256(semantic["action"]),
        str(semantic["source_window_key"]),
        str(row["semantic_input_sha256"]),
    )


def build_trial_plan(
    primary_specs: list[dict[str, Any]],
    fallback_specs: list[dict[str, Any]],
    *,
    workers: int,
) -> dict[str, Any]:
    """Build a worker-independent complete two-wave trial plan."""
    validate_worker_count(workers)
    rows: list[dict[str, Any]] = []
    for wave_index, (wave, specs) in enumerate(
        (("primary", primary_specs), ("fallback", fallback_specs))
    ):
        for spec in specs:
            semantic_input = dict(spec["semantic_input"])
            semantic_input_sha256 = semantic_sha256(semantic_input)
            rows.append(
                {
                    "wave": wave,
                    "wave_index": wave_index,
                    "semantic_input": semantic_input,
                    "semantic_input_sha256": semantic_input_sha256,
                    "trial_id": (
                        "traffic-"
                        f"{semantic_input_sha256[:24]}"
                    ),
                }
            )
    trial_ids = [row["trial_id"] for row in rows]
    if len(set(trial_ids)) != len(trial_ids):
        raise MinerExecutionError(
            "traffic_parallel_duplicate_trial_id"
        )
    rows.sort(key=_canonical_key)
    for trial_index, row in enumerate(rows):
        row["trial_index"] = trial_index
    primary_wave = [
        row for row in rows if row["wave"] == "primary"
    ]
    fallback_wave = [
        row for row in rows if row["wave"] == "fallback"
    ]
    semantic_plan = {
        "headroom_contract_id": (
            "traffic.native_signal_supervision.v2"
        ),
        "primary_wave": [
            row["semantic_input"] for row in primary_wave
        ],
        "fallback_wave": [
            row["semantic_input"] for row in fallback_wave
        ],
    }
    return {
        "schema_version": "1.0",
        "plan_sha256": semantic_sha256(semantic_plan),
        "primary_wave": primary_wave,
        "fallback_wave": fallback_wave,
        "canonical_order": [
            row["trial_id"] for row in rows
        ],
        "trial_count_primary": len(primary_wave),
        "trial_count_fallback_if_needed": len(fallback_wave),
    }


def _worker_failure(
    spec: dict[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    message = f"{type(exc).__name__}: {exc}"
    lowered = message.lower()
    reason_code = (
        "traffic_parallel_port_collision"
        if "address already in use" in lowered
        or "port collision" in lowered
        else "traffic_parallel_worker_failure"
    )
    return {
        "trial_id": spec["trial_id"],
        "trial_index": spec["trial_index"],
        "wave": spec["wave"],
        "status": "failed",
        "reason_code": reason_code,
        "error": message,
    }


def execute_paired_trials(
    specs: list[dict[str, Any]],
    *,
    workers: int,
    worker_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Run whole paired trials and return canonical parent-side results."""
    validate_worker_count(workers)
    trial_ids = [str(row["trial_id"]) for row in specs]
    if len(set(trial_ids)) != len(trial_ids):
        raise MinerExecutionError(
            "traffic_parallel_duplicate_trial_id"
        )
    effective = min(workers, len(specs)) if specs else 0
    if effective <= 1:
        results = []
        failures = []
        for spec in specs:
            try:
                results.append(worker_fn(spec))
            except Exception as exc:
                failure = _worker_failure(spec, exc)
                results.append(failure)
                failures.append(failure)
    else:
        results = []
        failures = []
        executor = ProcessPoolExecutor(
            max_workers=effective,
            mp_context=multiprocessing.get_context("spawn"),
        )
        futures = {
            executor.submit(worker_fn, spec): spec for spec in specs
        }
        try:
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    failure = _worker_failure(spec, exc)
                    results.append(failure)
                    failures.append(failure)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=False)
    results.sort(
        key=lambda row: (
            int(row["trial_index"]),
            str(row["trial_id"]),
        )
    )
    failures.sort(
        key=lambda row: (
            int(row["trial_index"]),
            str(row["trial_id"]),
        )
    )
    return {
        "results": results,
        "worker_failures": failures,
        "workers_effective": effective,
        "multiprocessing_start_method": "spawn",
        "canonical_result_order": True,
    }
