#!/usr/bin/env python3
"""Build a bounded source-locked subset of Alibaba's 100K simulator trace."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

UPSTREAM_SHA256 = (
    "04b563bdf706dbb8d0dd167dec19e0a56a1b4c404d64df31e28bcb204ee1ac30"
)
UPSTREAM_COMMIT = "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71"
UPSTREAM_URL = (
    "https://github.com/alibaba/clusterdata/blob/"
    f"{UPSTREAM_COMMIT}/cluster-trace-gpu-v2020/simulator/traces/pai/"
    "pai_job_duration_estimate_100K.csv"
)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_rows(source: Path, *, max_jobs: int) -> list[dict[str, Any]]:
    if max_jobs <= 0:
        raise ValueError("max_jobs must be positive")
    rows: list[dict[str, Any]] = []
    with source.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            duration = float(raw["duration"])
            submit_time = float(raw["submit_time"])
            num_gpu = float(raw["num_gpu"])
            num_cpu = float(raw["num_cpu"])
            num_inst = float(raw["num_pod"])
            if duration <= 0 or num_gpu <= 0 or num_inst <= 0:
                continue
            job_id = str(raw["job_id"])
            rows.append(
                {
                    "job_name": str(raw["job_name"]),
                    "inst_id": job_id,
                    "user": str(raw["user"]),
                    "start_time": submit_time,
                    "end_time": submit_time + duration,
                    "duration_seconds": duration,
                    "task_count": 1,
                    "instance_count": num_inst,
                    "requested_cpu_percent": num_cpu / num_inst,
                    "requested_memory_units": 0.0,
                    "requested_gpu_units": num_gpu / num_inst,
                    "gpu_types": str(raw["gpu_type"]),
                    "job_id": job_id,
                    "duration": duration,
                    "num_cpu": num_cpu,
                    "num_gpu": num_gpu,
                    "submit_time": submit_time,
                    "num_inst": num_inst,
                }
            )
            if len(rows) >= max_jobs:
                break
    if len(rows) != max_jobs:
        raise ValueError(f"only {len(rows)} eligible GPU jobs; requested {max_jobs}")
    return rows


def write_subset(
    source: Path,
    output_csv: Path,
    output_manifest: Path,
    *,
    max_jobs: int,
) -> dict[str, Any]:
    observed_sha256 = _sha256(source)
    if observed_sha256 != UPSTREAM_SHA256:
        raise ValueError("Alibaba 100K simulator trace SHA-256 mismatch")
    rows = build_rows(source, max_jobs=max_jobs)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    try:
        portable_output_path = str(output_csv.relative_to(Path.cwd()))
    except ValueError:
        portable_output_path = str(output_csv)
    manifest = {
        "schema_version": "1.0",
        "status": "source_subset_ready",
        "source_id": "alibaba_cluster_trace_gpu_v2020_simulator_100k",
        "source_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_sha256": UPSTREAM_SHA256,
        "selection": {
            "method": "first_eligible_positive_gpu_jobs_in_upstream_order",
            "max_jobs": max_jobs,
            "requires_positive_gpu_request": True,
        },
        "field_mapping": {
            "start_time": "submit_time",
            "duration_seconds": "duration",
            "instance_count": "num_pod",
            "requested_cpu_percent": "num_cpu / num_pod",
            "requested_gpu_units": "num_gpu / num_pod",
            "requested_memory_units": "unavailable_in_simulator_trace; not consumed",
        },
        "derived_csv": {
            "path": portable_output_path,
            "sha256": _sha256(output_csv),
            "size_bytes": output_csv.stat().st_size,
            "n_jobs": len(rows),
        },
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--max-jobs", type=int, default=6000)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    report = write_subset(
        args.source.resolve(),
        args.output_csv.resolve(),
        args.output_manifest.resolve(),
        max_jobs=args.max_jobs,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
