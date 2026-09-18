"""Scenario-level provenance locks for the Alibaba cluster-trace sources.

Mirrors ``domains.power_grid.seeds.source_locks`` /
``domains.logistics.seeds.source_locks`` (same ``SourceLock`` dataclass +
``provenance_lock_kwargs`` helper) so the audit gate reads datacenter
provenance the same way it reads power-grid / logistics provenance. This
module does NOT import from another domain.

Why the upstream anchors exist
------------------------------
The released ``alibaba_trace_sim`` rows consume *repo-tracked* derived CSVs
under ``sources/alibaba/``. Those bytes are a lossy subset of an upstream
archive, so a lock that only repeats the subset's own SHA-256 is a closed
loop: it can never detect a substituted upstream. ``alibaba_local_lock_record``
therefore pairs the repo-tracked bytes the runtime actually opens with the
upstream archive hashes the subset builders assert before emitting a single
row (``scripts/build_alibaba_trace_subset.py`` for the official 2020 GPU
trace archives, ``scripts/build_alibaba_simulator_trace_subset.py`` for the
100K simulator trace). Both builders fail closed on a hash mismatch, so the
anchor is a real gate and not a restatement.

The anchor constants are restated here — this module deliberately does not
import ``scripts`` — and ``tests/test_datacenter_source_locks.py`` asserts
they still equal the builders' constants, so drift is caught.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SourceLock:
    url: str
    commit: str
    lock_strategy: str
    version: str | None = None
    data_release: str | None = None
    license: str | None = None


SOURCE_LOCKS: dict[str, SourceLock] = {
    # Alibaba Cluster Trace GPU (v2020) — the official 100K simulator trace
    # plus the job/task tables used to derive the ``sources/alibaba`` subsets.
    "alibaba_cluster_trace_gpu_v2020": SourceLock(
        url="https://github.com/alibaba/clusterdata",
        commit="0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71",
        lock_strategy="upstream_archives_sha256_plus_derived_csv_sha256",
        license="Apache-2.0 upstream repository; trace terms apply",
    ),
    "alibaba_cluster_trace_gpu_v2020_simulator_100k": SourceLock(
        url=(
            "https://github.com/alibaba/clusterdata/blob/"
            "0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71/cluster-trace-gpu-v2020/"
            "simulator/traces/pai/pai_job_duration_estimate_100K.csv"
        ),
        commit="0d0f3f1efdbf1add6a7bcc63676eafbd1eb11f71",
        lock_strategy="upstream_git_commit_and_sha256_plus_derived_subset_sha256",
        license="Apache-2.0 upstream repository; trace terms apply",
    ),
}


def provenance_lock_kwargs(*source_ids: str) -> dict[str, str]:
    """Return common scenario provenance lock fields for one or more sources.

    Signature-identical to the other domains' helper so seed factories and
    audit code use one calling convention across domains.
    """
    locks = [SOURCE_LOCKS[source_id] for source_id in source_ids]
    out = {
        "url": " + ".join(lock.url for lock in locks),
        "commit": " + ".join(lock.commit for lock in locks),
        "lock_strategy": " + ".join(lock.lock_strategy for lock in locks),
    }
    licenses = [lock.license for lock in locks if lock.license]
    if licenses:
        out["license"] = " + ".join(licenses)
    return out


def sha256_file(path: Path) -> str:
    """Hash a file without importing another domain's helper."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Upstream archive hashes asserted by the subset builders before emission.
# Kept literal (no ``scripts`` import) and drift-checked by
# ``tests/test_datacenter_source_locks.py``.
UPSTREAM_ARCHIVE_SHA256: dict[str, str] = {
    "upstream_sha256": (
        "04b563bdf706dbb8d0dd167dec19e0a56a1b4c404d64df31e28bcb204ee1ac30"
    ),
    "pai_job_table.tar.gz": (
        "5aad7f7caac501136d14ed6a48e40546f825d7b0617a3a4f337e2348fe0a6cb0"
    ),
    "pai_task_table.tar.gz": (
        "cd1d6dc3215d2a8607ccf6b6dd952b5db776df86926c73259fea7c1499ac40e5"
    ),
}

# Repo-tracked derived subset → (source id, builder, anchors the builder
# asserts, primary anchor, row-selection recipe). Every anchor the builder
# verifies is recorded, not just the one file the rows happen to be read from:
# ``build_alibaba_trace_subset`` reads job rows from ``pai_job_table.tar.gz``
# but selects them using task-side fields from ``pai_task_table.tar.gz``, so
# the job-table digest alone cannot pin the emitted CSV. The recipe digest
# records the selection function the builder applies, so a reader can tell
# which rows the anchors produced.
ALIBABA_LOCAL_LOCKS: dict[str, tuple[str, str, tuple[str, ...], str, dict[str, Any]]] = {
    "sources/alibaba/alibaba_trace_gpu_simulator_jobs_6000.csv": (
        "alibaba_cluster_trace_gpu_v2020_simulator_100k",
        "scripts/build_alibaba_simulator_trace_subset.py",
        ("upstream_sha256",),
        "upstream_sha256",
        {
            "method": "first_eligible_positive_gpu_jobs_in_upstream_order",
            "max_jobs": 6000,
            "requires_positive_gpu_request": True,
            "derived_fields": {
                "start_time": "submit_time",
                "duration_seconds": "duration",
                "instance_count": "num_pod",
                "requested_cpu_percent": "num_cpu / num_pod",
                "requested_gpu_units": "num_gpu / num_pod",
            },
        },
    ),
    "sources/alibaba/alibaba_trace_gpu_jobs_1000.csv": (
        "alibaba_cluster_trace_gpu_v2020",
        "scripts/build_alibaba_trace_subset.py",
        ("pai_job_table.tar.gz", "pai_task_table.tar.gz"),
        "pai_job_table.tar.gz",
        {
            "method": (
                "deterministic_completed_job_prefix_with_completed_gpu_tasks"
            ),
            "max_jobs": 1000,
            "candidate_multiplier": 10,
            "requires_job_status": "Terminated",
            "requires_positive_gpu_request": True,
        },
    ),
}


def alibaba_local_lock_record(
    *,
    repo_root: Path | None = None,
    verify_local: bool = True,
) -> dict[str, dict[str, Any]]:
    """Machine-checkable derivation record for the tracked Alibaba CSVs.

    Each entry ties the repo-tracked bytes (``derived_asset_sha256``) to the
    complete set of upstream archives the subset builder verified before
    emitting them (``upstream_anchored_sha256s``, with
    ``upstream_anchored_sha256`` retained as the primary anchor for existing
    readers), so the lock neither self-references the subset it produces nor
    under-declares the builder's gate.
    """
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    record: dict[str, dict[str, Any]] = {}
    for relative, (
        source_id,
        builder,
        anchor_keys,
        primary,
        recipe,
    ) in sorted(ALIBABA_LOCAL_LOCKS.items()):
        anchors: dict[str, str] = {}
        for anchor_key in anchor_keys:
            anchor = UPSTREAM_ARCHIVE_SHA256.get(anchor_key)
            if anchor is None:
                raise KeyError(f"alibaba_upstream_anchor_missing:{anchor_key}")
            anchors[anchor_key] = anchor
        path = root / relative
        if verify_local and not path.is_file():
            raise FileNotFoundError(f"alibaba_local_asset_missing:{relative}")
        entry: dict[str, Any] = {
            "source_id": source_id,
            "subset_builder": builder,
            "upstream_anchor": primary,
            "upstream_anchored_sha256": anchors[primary],
            "upstream_anchor_keys": sorted(anchors),
            "upstream_anchored_sha256s": anchors,
            "row_selection_recipe": dict(recipe),
            "row_selection_recipe_sha256": hashlib.sha256(
                json.dumps(
                    recipe, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
        }
        if path.is_file():
            entry["derived_asset_sha256"] = sha256_file(path)
        record[relative] = entry
    return record
