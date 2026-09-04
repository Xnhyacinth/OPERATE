#!/usr/bin/env python3
"""Build a bounded, source-locked Alibaba GPU trace subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

JOB_MEMBER = "pai_job_table.csv"
TASK_MEMBER = "pai_task_table.csv"
JOB_FIELDS = ("job_name", "inst_id", "user", "status", "start_time", "end_time")
TASK_FIELDS = (
    "job_name",
    "task_name",
    "inst_num",
    "status",
    "start_time",
    "end_time",
    "plan_cpu",
    "plan_mem",
    "plan_gpu",
    "gpu_type",
)
OFFICIAL_SHA256 = {
    "pai_job_table.tar.gz": "5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0",
    "pai_task_table.tar.gz": "cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5",
}
OUTPUT_FIELDS = (
    "job_name",
    "inst_id",
    "user",
    "start_time",
    "end_time",
    "duration_seconds",
    "task_count",
    "instance_count",
    "requested_cpu_percent",
    "requested_memory_units",
    "requested_gpu_units",
    "gpu_types",
    "job_id",
    "duration",
    "num_cpu",
    "num_gpu",
    "submit_time",
    "num_inst",
)


def _float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _rows(path: Path, member: str, fields: tuple[str, ...]) -> Iterator[dict[str, str]]:
    with tarfile.open(path, "r:gz") as archive:
        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member} is missing from {path}")
        with io.TextIOWrapper(extracted, encoding="utf-8", newline="") as stream:
            yield from csv.DictReader(stream, fieldnames=fields)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_subset(
    job_tar: Path,
    task_tar: Path,
    *,
    max_jobs: int,
    candidate_multiplier: int = 10,
) -> list[dict[str, Any]]:
    """Join a deterministic prefix of completed jobs to completed task resources."""
    if max_jobs <= 0:
        raise ValueError("max_jobs must be positive")
    if candidate_multiplier <= 0:
        raise ValueError("candidate_multiplier must be positive")

    candidates: list[dict[str, Any]] = []
    candidate_limit = max_jobs * candidate_multiplier
    for row in _rows(job_tar, JOB_MEMBER, JOB_FIELDS):
        start = _float(row.get("start_time"))
        end = _float(row.get("end_time"))
        if row.get("status") != "Terminated" or start is None or end is None or end <= start:
            continue
        candidates.append(
            {
                "job_name": row["job_name"],
                "inst_id": row["inst_id"],
                "user": row["user"],
                "start_time": start,
                "end_time": end,
                "duration_seconds": end - start,
            }
        )
        if len(candidates) >= candidate_limit:
            break

    candidate_names = {row["job_name"] for row in candidates}
    resources: dict[str, dict[str, Any]] = {}
    for row in _rows(task_tar, TASK_MEMBER, TASK_FIELDS):
        job_name = row.get("job_name")
        if job_name not in candidate_names or row.get("status") != "Terminated":
            continue
        instances = _float(row.get("inst_num"))
        cpu = _float(row.get("plan_cpu"))
        memory = _float(row.get("plan_mem"))
        gpu = _float(row.get("plan_gpu"))
        if None in (instances, cpu, memory, gpu) or instances <= 0:
            continue
        aggregate = resources.setdefault(
            str(job_name),
            {
                "task_count": 0,
                "instance_count": 0.0,
                "requested_cpu_percent": 0.0,
                "requested_memory_units": 0.0,
                "requested_gpu_units": 0.0,
                "gpu_types": set(),
            },
        )
        aggregate["task_count"] += 1
        aggregate["instance_count"] += instances
        aggregate["requested_cpu_percent"] += instances * cpu
        aggregate["requested_memory_units"] += instances * memory
        aggregate["requested_gpu_units"] += instances * gpu / 100.0
        if row.get("gpu_type"):
            aggregate["gpu_types"].add(row["gpu_type"])

    subset: list[dict[str, Any]] = []
    for job in candidates:
        aggregate = resources.get(job["job_name"])
        if aggregate is None or aggregate["requested_gpu_units"] <= 0:
            continue
        subset.append(
            {
                **job,
                "task_count": aggregate["task_count"],
                "instance_count": round(aggregate["instance_count"]),
                "requested_cpu_percent": round(
                    aggregate["requested_cpu_percent"], 6
                ),
                "requested_memory_units": round(
                    aggregate["requested_memory_units"], 6
                ),
                "requested_gpu_units": round(
                    aggregate["requested_gpu_units"], 6
                ),
                "gpu_types": sorted(aggregate["gpu_types"]),
                "job_id": job["inst_id"],
                "duration": max(1, round(job["duration_seconds"])),
                "num_cpu": round(
                    aggregate["requested_cpu_percent"] / 100.0, 6
                ),
                "num_gpu": round(aggregate["requested_gpu_units"], 6),
                "submit_time": round(job["start_time"]),
                "num_inst": round(aggregate["instance_count"]),
            }
        )
        if len(subset) >= max_jobs:
            break
    return subset


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "gpu_types": "|".join(row["gpu_types"])})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-tar", type=Path, required=True)
    parser.add_argument("--task-tar", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int, default=1000)
    parser.add_argument("--candidate-multiplier", type=int, default=10)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    input_hashes = {
        "pai_job_table.tar.gz": _sha256(args.job_tar),
        "pai_task_table.tar.gz": _sha256(args.task_tar),
    }
    for name, expected in OFFICIAL_SHA256.items():
        if input_hashes.get(name) != expected:
            raise ValueError(f"{name} does not match the official SHA-256")

    rows = build_subset(
        args.job_tar,
        args.task_tar,
        max_jobs=args.max_jobs,
        candidate_multiplier=args.candidate_multiplier,
    )
    if len(rows) < args.max_jobs:
        raise ValueError(
            f"only {len(rows)} eligible jobs found; requested {args.max_jobs}"
        )
    _write_csv(args.output_csv, rows)
    manifest = {
        "schema_version": "0.2",
        "status": "source_subset_ready_adapter_pending",
        "source_id": "alibaba_cluster_trace_gpu_v2020",
        "source_url": "https://github.com/alibaba/clusterdata",
        "asset_urls": {
            "pai_job_table.tar.gz": (
                "https://aliopentrace.oss-cn-beijing.aliyuncs.com/"
                "v2020GPUTraces/pai_job_table.tar.gz"
            ),
            "pai_task_table.tar.gz": (
                "https://aliopentrace.oss-cn-beijing.aliyuncs.com/"
                "v2020GPUTraces/pai_task_table.tar.gz"
            ),
        },
        "upstream_sha256": input_hashes,
        "selection": {
            "method": "deterministic_completed_job_prefix_with_completed_gpu_tasks",
            "max_jobs": args.max_jobs,
            "candidate_multiplier": args.candidate_multiplier,
            "requires_job_status": "Terminated",
            "requires_positive_gpu_request": True,
        },
        "n_jobs": len(rows),
        "time_bounds": {
            "start_time_min": min(row["start_time"] for row in rows),
            "start_time_max": max(row["start_time"] for row in rows),
            "end_time_max": max(row["end_time"] for row in rows),
        },
        "derived_csv": {
            "path": str(args.output_csv),
            "sha256": _sha256(args.output_csv),
            "size_bytes": args.output_csv.stat().st_size,
            "upstream_simulator_compatible_fields": [
                "job_id",
                "user",
                "duration",
                "num_cpu",
                "num_gpu",
                "submit_time",
                "num_inst",
            ],
        },
        "promotion_blockers": [
            "native_datacenter_adapter_pending",
            "tool_protocol_and_evidence_pending",
            "deterministic_replay_and_behavioral_headroom_pending",
            "difficulty_duplicate_and_multi_model_gates_pending",
        ],
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
