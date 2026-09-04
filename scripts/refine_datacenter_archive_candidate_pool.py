#!/usr/bin/env python3
"""Close the Datacenter candidate pool without mutating an active release.

The refiner keeps one high-pressure Alibaba Spot window per GPU model, keeps
lower-pressure windows as visible secondary variants, and assigns terminal
repair/redesign dispositions to source releases and canonical archived
families.  ``ready_for_full_admission`` means ready for the bounded delta replay
listed in the report; it is not release admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.datacenter.source_native_candidates import (  # noqa: E402
    build_openb_candidate,
    probe_dlrm_source,
    probe_genai_source,
    probe_openb_candidate,
)
from core.protocol21_evidence import canonicalize_repo_owned_paths  # noqa: E402


DEFAULT_CLUSTER_ROOT = ROOT / "works/clusterdata"
DEFAULT_SPOT_LEDGER = (
    ROOT / ".hl/artifacts/datacenter_spot_candidate_ledger_20260828.json"
)
DEFAULT_OUTPUT = (
    ROOT / ".hl/artifacts/operate_v058_datacenter_archive_candidate_refinement.json"
)
ARCHIVE_REF = "ecd2a4fc^"
ARCHIVE_PATHS = (
    "release/dt_sched_bench_v0_52_0_candidate/external_source_catalog.json",
    "release/dt_sched_bench_v0_52_0_candidate/"
    "protocol21_agentic_remediation_queue_v1.json",
)
FINAL_DISPOSITIONS = {
    "ready_for_full_admission",
    "held_repair",
    "redesign",
    "secondary",
    "rejected",
}
SPOT_AXES = {
    "cross_epoch_arrivals",
    "duration_heterogeneity",
    "hp_spot_priority_tradeoff",
    "multi_tenant_resource_contention",
}


SOURCE_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "key": "alibaba-v2020-pai",
        "source_id": "alibaba_cluster_trace_gpu_v2020_simulator_100k",
        "paths": (
            "cluster-trace-gpu-v2020/simulator/traces/pai/"
            "pai_job_duration_estimate_100K.csv",
        ),
        "available_disposition": "secondary",
        "available_reasons": [
            "source_family_already_contributes_core",
            "current_backend_consumes_pai_queue_semantics",
        ],
        "adapter_outcome": "passed",
        "adapter_detail": (
            "The released Alibaba trace backend already consumes PAI job arrivals, "
            "durations and resource requests; another window batch would pad the same "
            "physical source."
        ),
    },
    {
        "key": "alibaba-v2023-openb",
        "source_id": "alibaba_cluster_trace_gpu_v2023_openb",
        "paths": (
            "cluster-trace-gpu-v2023/csv/openb_node_list_gpu_node.csv",
            "cluster-trace-gpu-v2023/csv/openb_pod_list_gpuspec33.csv",
        ),
        "available_disposition": "ready_for_full_admission",
        "available_reasons": [
            "bounded_delta_replay_required",
            "executable_openb_placement_probe_passed",
            "independent_node_pod_source_graph",
        ],
        "adapter_outcome": "passed",
        "adapter_detail": (
            "The dedicated OpenB backend consumes node/pod rows, CPU, memory, GPU "
            "model/share, QoS and lifetimes and exposes placement, migration and "
            "placement-policy controls."
        ),
    },
    {
        "key": "alibaba-v2025-dlrm",
        "source_id": "alibaba_cluster_trace_gpu_v2025_dlrm",
        "paths": ("cluster-trace-gpu-v2025/disaggregated_DLRM_trace.csv",),
        "available_disposition": "redesign",
        "available_reasons": [
            "disaggregated_training_adapter_required",
            "independent_pipeline_contention_source",
        ],
        "adapter_outcome": "adapter_redesign_required",
        "adapter_detail": (
            "The queue backend cannot infer DLRM stage dependencies, communication "
            "contention or disaggregated placement from generic queue jobs."
        ),
        "redesign_spec": {
            "backend_contract": "alibaba_disaggregated_dlrm.v1",
            "decision_axes": [
                "compute_communication_overlap",
                "cross_stage_backpressure",
                "disaggregated_stage_placement",
                "training_deadline_and_energy_cost",
            ],
            "native_controls": [
                "place_training_stage",
                "reserve_interconnect_capacity",
                "reschedule_pipeline_stage",
            ],
        },
    },
    {
        "key": "alibaba-v2026-genai-lora",
        "source_id": "alibaba_cluster_trace_v2026_genai",
        "paths": ("cluster-trace-v2026-GenAI/lora_request_trace.csv",),
        "available_disposition": "redesign",
        "available_reasons": [
            "genai_serving_update_adapter_required",
            "independent_request_latency_source",
        ],
        "adapter_outcome": "adapter_redesign_required",
        "adapter_detail": (
            "A generic batch queue loses LoRA/base-model identity, serving latency, "
            "replica routing and online-update interference."
        ),
        "redesign_spec": {
            "backend_contract": "alibaba_genai_lora_serving.v1",
            "decision_axes": [
                "base_model_update_interference",
                "latency_slo_and_queue_cost",
                "lora_batching_and_replica_routing",
                "shared_gpu_memory_and_compute_contention",
            ],
            "native_controls": [
                "route_lora_request",
                "set_dynamic_batch",
                "schedule_model_update",
                "scale_model_replica",
            ],
        },
    },
    {
        "key": "alibaba-v2026-spot",
        "source_id": "alibaba_cluster_trace_v2026_spot_gpu",
        "paths": (
            "cluster-trace-v2026-spot-gpu/job_info_df.csv",
            "cluster-trace-v2026-spot-gpu/node_info_df.csv",
        ),
        "available_disposition": "secondary",
        "available_reasons": [
            "represented_by_window_candidates",
            "source_bundle_not_an_independent_candidate",
        ],
        "adapter_outcome": "passed",
        "adapter_detail": (
            "The current backend consumes the typed Spot source schema and exposes queue "
            "policy, preemption and capacity reservation controls."
        ),
    },
)


ARCHIVED_FAMILIES: dict[str, dict[str, Any]] = {
    "online_batch_colocation": {
        "slug": "online-batch-colocation",
        "source_key": "alibaba-v2020-pai",
        "available_disposition": "secondary",
        "reasons": [
            "covered_by_current_executable_queue_family",
            "historical_family_canonicalized",
        ],
        "detail": (
            "This historical proposal is already represented by the source-grounded PAI "
            "queue backend; old window copies do not add a physical source or control axis."
        ),
    },
    "dag_job_scheduling": {
        "slug": "dag-job-scheduling",
        "source_key": "alibaba-v2020-pai",
        "available_disposition": "redesign",
        "reasons": [
            "dag_dependency_semantics_not_consumed",
            "historical_family_canonicalized",
        ],
        "detail": (
            "Preserve source-native task dependencies and add precedence-aware scheduling; "
            "flattening a DAG into independent queue jobs is not an admissible repair."
        ),
        "redesign_spec": {
            "backend_contract": "alibaba_dag_scheduler.v1",
            "decision_axes": [
                "critical_path_deadline",
                "dependency_aware_preemption",
                "gang_and_stage_resource_contention",
            ],
            "native_controls": ["prioritize_stage", "place_task_group", "preempt_stage"],
        },
    },
    "gpu_placement_and_rescheduling": {
        "slug": "gpu-placement-and-rescheduling",
        "source_key": "alibaba-v2023-openb",
        "available_disposition": "secondary",
        "reasons": [
            "covered_by_executable_openb_candidate",
            "historical_family_canonicalized",
        ],
        "detail": (
            "Use the OpenB node/pod graph and native placement state; queue ordering alone "
            "cannot test fragmentation-aware rescheduling."
        ),
    },
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _attempt(code: str, outcome: str, detail: str) -> dict[str, str]:
    return {
        "code": code,
        "outcome": outcome,
        "detail": detail,
        "phase": "proposed" if outcome == "design_required" else "executed",
    }


def _git_blob(root: Path, relative: str) -> dict[str, Any] | None:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(root), "ls-tree", "-l", "HEAD", "--", relative],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not output:
        return None
    prefix, path = output.split("\t", 1)
    mode, kind, object_id, size = prefix.split()
    return {
        "path": path,
        "availability": "git_object",
        "git_object_id": object_id,
        "git_object_kind": kind,
        "git_mode": mode,
        "size_bytes": int(size) if size != "-" else None,
    }


def _file_evidence(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if path.is_file():
        return {
            "path": str(path),
            "availability": "working_tree",
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    tracked = _git_blob(root, relative)
    if tracked is not None:
        return tracked
    return {"path": str(path), "availability": "missing"}


def _row(
    *,
    candidate_id: str,
    source_id: str,
    source_unit: str,
    classification_scope: str,
    final_disposition: str,
    reason_codes: Iterable[str],
    repair_attempts: list[dict[str, str]],
    evidence: dict[str, Any],
    redesign_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if final_disposition not in FINAL_DISPOSITIONS:
        raise ValueError(f"invalid disposition: {final_disposition}")
    row = {
        "candidate_id": candidate_id,
        "source_id": source_id,
        "source_family": source_id,
        "source_unit": source_unit,
        "domain": "datacenter",
        "classification_scope": classification_scope,
        "final_disposition": final_disposition,
        "reason_codes": sorted(set(reason_codes)),
        "repair_attempts": repair_attempts,
        "evidence": evidence,
    }
    # Keep the legacy alias while pool aggregators migrate to final_disposition.
    row["disposition"] = final_disposition
    if redesign_spec is not None:
        row["redesign_spec"] = redesign_spec
    return row


def _source_rows(cluster_root: Path) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    availability: dict[str, bool] = {}
    for spec in SOURCE_FAMILIES:
        files = [_file_evidence(cluster_root, relative) for relative in spec["paths"]]
        available = all(item["availability"] != "missing" for item in files)
        availability[str(spec["key"])] = available
        attempts = [
            _attempt(
                "source_bundle_discovery",
                "passed" if available else "blocked_external",
                (
                    "All required source objects are available in the working tree or its "
                    "locked Git object database."
                    if available
                    else "Fetch or restore the locked source objects; absence is an "
                    "environment repair, not an intrinsic task rejection."
                ),
            )
        ]
        semantic_probe: dict[str, Any] | None = None
        candidate_recipe: dict[str, Any] | None = None
        local_paths = [cluster_root / relative for relative in spec["paths"]]
        if available and all(path.is_file() for path in local_paths):
            try:
                if spec["key"] == "alibaba-v2023-openb":
                    candidate_recipe = build_openb_candidate(
                        node_path=local_paths[0],
                        pod_path=local_paths[1],
                    )
                    semantic_probe = probe_openb_candidate(candidate_recipe)
                elif spec["key"] == "alibaba-v2025-dlrm":
                    semantic_probe = probe_dlrm_source(local_paths[0])
                elif spec["key"] == "alibaba-v2026-genai-lora":
                    semantic_probe = probe_genai_source(
                        local_paths[0], source_root=local_paths[0].parent
                    )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                semantic_probe = {
                    "parser_status": "failed",
                    "disposition": "redesign",
                    "blockers": [f"source_native_probe_failed:{type(exc).__name__}"],
                }
        if available:
            disposition = str(spec["available_disposition"])
            reasons = list(spec["available_reasons"])
            if semantic_probe is not None and spec["key"] == "alibaba-v2023-openb":
                if semantic_probe.get("status") != "passed":
                    disposition = "redesign"
                    reasons = ["openb_executable_probe_failed"]
            attempts.append(
                _attempt(
                    "native_backend_semantic_fit",
                    str(spec["adapter_outcome"]),
                    str(spec["adapter_detail"]),
                )
            )
        else:
            disposition = "held_repair"
            reasons = ["source_bundle_unavailable"]
            # The final attempt is intentionally the actionable blocking edge.
            attempts.append(
                _attempt(
                    "locked_source_fetch",
                    "blocked_external",
                    "Restore the exact upstream commit/source bundle, then rerun this refiner.",
                )
            )
        rows.append(
            _row(
                candidate_id=f"datacenter/source/{spec['key']}",
                source_id=str(spec["source_id"]),
                source_unit="+".join(spec["paths"]),
                classification_scope="candidate",
                final_disposition=disposition,
                reason_codes=reasons,
                repair_attempts=attempts,
                evidence={
                    "source_available": available,
                    "files": files,
                    "semantic_fit_evaluated": available,
                    "source_native_probe": semantic_probe,
                    "candidate_recipe": candidate_recipe,
                },
                redesign_spec=(
                    dict(spec["redesign_spec"])
                    if available and spec.get("redesign_spec")
                    else None
                ),
            )
        )
    return rows, availability


def _pressure(candidate: dict[str, Any]) -> float:
    evidence = candidate.get("evidence") or {}
    return (
        float(evidence.get("gpu_demand_capacity_ratio") or 0.0)
        + math.log1p(float(evidence.get("duration_ratio") or 0.0))
        + float(evidence.get("arrival_epoch_count") or 0.0) / 10.0
        + float(evidence.get("organization_count") or 0.0) / 10.0
    )


def _spot_contract_complete(candidate: dict[str, Any]) -> bool:
    evidence = candidate.get("evidence") or {}
    transform = ((candidate.get("suite_recipe") or {}).get("backend_config") or {}).get(
        "source_transform"
    ) or {}
    digest = str(candidate.get("source_window_sha256") or "")
    return bool(
        candidate.get("candidate_id")
        and re.fullmatch(r"[0-9a-f]{64}", digest)
        and SPOT_AXES <= set(candidate.get("independent_decision_axes") or [])
        and evidence.get("all_jobs_individually_schedulable") is True
        and float(evidence.get("gpu_demand_capacity_ratio") or 0.0) > 1.0
        and int(evidence.get("arrival_epoch_count") or 0) >= 3
        and int(evidence.get("organization_count") or 0) >= 2
        and {"HP", "Spot"} <= set((evidence.get("priority_counts") or {}))
        and transform.get("source_schema") == "alibaba-spot-gpu-v2026-v1"
        and transform.get("source_window_sha256") == digest
    )


def _spot_rows(
    path: Path,
    reference_probe: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    payload = _load(path)
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("spot candidate ledger must contain candidates")
    valid = [row for row in candidates if isinstance(row, dict) and _spot_contract_complete(row)]
    winners: dict[str, dict[str, Any]] = {}
    for candidate in valid:
        model = str((candidate.get("evidence") or {}).get("gpu_model") or "unknown")
        previous = winners.get(model)
        if previous is None or (_pressure(candidate), str(candidate["candidate_id"])) > (
            _pressure(previous),
            str(previous["candidate_id"]),
        ):
            winners[model] = candidate

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("candidate_id") or "spot/missing-id")
        evidence = dict(candidate.get("evidence") or {})
        model = str(evidence.get("gpu_model") or "unknown")
        complete = _spot_contract_complete(candidate)
        winner = complete and winners.get(model) is candidate
        attempts = [
            _attempt(
                "spot_source_and_control_contract",
                "passed" if complete else "failed",
                "Require exact source row identity, HP/Spot mix, multi-tenant arrivals, "
                "resource contention and current backend source-schema consumption.",
            )
        ]
        if not complete:
            disposition = "rejected"
            reasons = ["spot_candidate_contract_incomplete"]
        elif winner:
            if reference_probe is None:
                from scripts.materialize_datacenter_candidate_delta import (
                    bounded_spot_reference_probe,
                )

                reference_probe = bounded_spot_reference_probe
            calibration = reference_probe(dict(candidate["suite_recipe"]))
            evidence["bounded_reference_calibration"] = calibration
            if calibration.get("material_headroom") is True:
                disposition = "ready_for_full_admission"
                reasons = [
                    "bounded_delta_replay_required",
                    "hardest_independent_gpu_model_axis",
                    "spot_preemption_slo_cost_contention",
                ]
            else:
                disposition = "secondary"
                reasons = ["no_realizable_decision_headroom"]
            attempts.append(
                _attempt(
                    "same_trace_padding_control",
                    (
                        "selected"
                        if disposition == "ready_for_full_admission"
                        else "retained_secondary"
                    ),
                    (
                        "Highest structural pressure for this GPU model; bounded native "
                        "action/timing calibration determines whether executable decision "
                        "headroom exists."
                    ),
                )
            )
        else:
            disposition = "secondary"
            reasons = ["same_gpu_model_lower_pressure_variant"]
            attempts.append(
                _attempt(
                    "same_trace_padding_control",
                    "retained_secondary",
                    "A harder source window already represents this GPU-model axis; preserve "
                    "this window for variance analysis without inflating Core.",
                )
            )
        rows.append(
            _row(
                candidate_id=candidate_id,
                source_id="alibaba_cluster_trace_v2026_spot_gpu",
                source_unit=(
                    f"job_info_df.csv#rows={candidate.get('source_row_indices') or []}"
                ),
                classification_scope="candidate",
                final_disposition=disposition,
                reason_codes=reasons,
                repair_attempts=attempts,
                evidence={
                    **evidence,
                    "pressure_score": round(_pressure(candidate), 9),
                    "source_window_sha256": candidate.get("source_window_sha256"),
                    "independent_decision_axes": candidate.get(
                        "independent_decision_axes"
                    )
                    or [],
                    "current_backend_control_axes": [
                        "preempt_job",
                        "reserve_gpu_capacity",
                        "set_queue_policy",
                    ],
                },
            )
        )
    return rows


def _archive_records(payload: Any) -> tuple[set[str], int]:
    families: set[str] = set()
    count = 0

    def visit(value: Any) -> None:
        nonlocal count
        if isinstance(value, dict):
            candidate_families = value.get("candidate_families")
            direct_datacenter = (
                str(value.get("domain") or "").lower() == "datacenter"
                or str(value.get("scenario_id") or "").startswith("datacenter/")
                or any(
                    "datacenter" in str(item).lower()
                    for item in value.get("domains") or []
                )
            )
            if direct_datacenter and isinstance(candidate_families, list):
                for family in candidate_families:
                    if str(family).strip():
                        families.add(str(family).strip())
                        count += 1
            elif direct_datacenter:
                count += 1
            for nested in value.values():
                if isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    return families, count


def _archived_rows(
    payloads: list[dict[str, Any]], source_available: dict[str, bool]
) -> tuple[list[dict[str, Any]], int]:
    families = set(ARCHIVED_FAMILIES)
    canonicalized_count = 0
    for payload in payloads:
        discovered, count = _archive_records(payload)
        families.update(discovered)
        canonicalized_count += count
    rows: list[dict[str, Any]] = []
    for family in sorted(families):
        spec = ARCHIVED_FAMILIES.get(family)
        if spec is None:
            rows.append(
                _row(
                    candidate_id=f"datacenter/archive/{_slug(family)}",
                    source_id="archived_datacenter_family",
                    source_unit=family,
                    classification_scope="candidate",
                    final_disposition="held_repair",
                    reason_codes=["archived_family_contract_unmapped"],
                    repair_attempts=[
                        _attempt(
                            "canonical_family_contract",
                            "design_required",
                            "Bind a real source graph, native control surface and reference "
                            "policy before this historical proposal can enter conversion.",
                        )
                    ],
                    evidence={"archive_ref": ARCHIVE_REF},
                )
            )
            continue
        available = source_available.get(str(spec["source_key"]), False)
        if available:
            disposition = str(spec["available_disposition"])
            reasons = list(spec["reasons"])
            outcome = (
                "covered_without_new_core_row"
                if disposition == "secondary"
                else "adapter_redesign_required"
            )
        else:
            disposition = "held_repair"
            reasons = ["archived_family_source_bundle_unavailable"]
            outcome = "blocked_external"
        rows.append(
            _row(
                candidate_id=f"datacenter/archive/{spec['slug']}",
                source_id="alibaba_cluster_trace_official",
                source_unit=family,
                classification_scope="candidate",
                final_disposition=disposition,
                reason_codes=reasons,
                repair_attempts=[
                    _attempt(
                        "canonicalize_historical_copies",
                        "passed",
                        "Collapse scenario/window copies into one scientific family decision.",
                    ),
                    _attempt("native_semantic_repair", outcome, str(spec["detail"])),
                ],
                evidence={
                    "archive_ref": ARCHIVE_REF,
                    "source_bundle_available": available,
                    "source_key": spec["source_key"],
                },
                redesign_spec=(
                    dict(spec["redesign_spec"])
                    if available and spec.get("redesign_spec")
                    else None
                ),
            )
        )
    return rows, canonicalized_count


def _git_archived_payloads() -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in ARCHIVE_PATHS:
        try:
            raw = subprocess.check_output(
                ["git", "show", f"{ARCHIVE_REF}:{path}"],
                cwd=ROOT,
                stderr=subprocess.DEVNULL,
            )
            value = json.loads(raw)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            payloads.append(value)
    return payloads


def build_refinement(
    *,
    cluster_root: Path,
    spot_ledger_path: Path,
    archived_registry_paths: list[Path] | None = None,
    archived_registry_payloads: list[dict[str, Any]] | None = None,
    spot_reference_probe: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, terminal Datacenter candidate ledger."""
    source_rows, source_available = _source_rows(cluster_root)
    spot_rows = _spot_rows(spot_ledger_path, spot_reference_probe)
    archive_payloads = list(archived_registry_payloads or [])
    archive_inputs: list[dict[str, Any]] = []
    for path in archived_registry_paths or []:
        payload = _load(path)
        archive_payloads.append(payload)
        archive_inputs.append({"path": str(path), "sha256": _sha256(path)})
    archive_rows, archived_count = _archived_rows(archive_payloads, source_available)
    rows = sorted(
        [*source_rows, *spot_rows, *archive_rows],
        key=lambda row: row["candidate_id"],
    )
    candidate_ids = [row["candidate_id"] for row in rows]
    duplicate_ids = len(candidate_ids) - len(set(candidate_ids))
    if duplicate_ids:
        raise ValueError("candidate refinement contains duplicate candidate IDs")
    if any(
        not row["candidate_id"]
        or not row["source_id"]
        or not row["source_unit"]
        or not row["reason_codes"]
        or not row["repair_attempts"]
        or row["final_disposition"] not in FINAL_DISPOSITIONS
        for row in rows
    ):
        raise ValueError("every Datacenter unit must have a complete terminal row")
    dispositions = Counter(row["final_disposition"] for row in rows)
    summary = {
        "n_discovered": len(rows),
        "n_terminal": len(rows),
        "n_unresolved": 0,
        "n_duplicate_candidate_ids": duplicate_ids,
        "n_ready_for_full_admission": dispositions["ready_for_full_admission"],
        "n_archived_families": len(archive_rows),
        "n_archived_rows_canonicalized": archived_count,
        "dispositions": dict(sorted(dispositions.items())),
    }
    report = {
        "schema_version": "operate-datacenter-archive-candidate-refinement-v1",
        "status": "terminal_candidate_refinement_complete",
        "candidate_only": True,
        "release_admission": False,
        "no_pending_decisions": True,
        "policy": {
            "one_hard_window_per_gpu_model": True,
            "same_trace_variants_remain_secondary": True,
            "missing_sources_are_environment_repairs": True,
            "redesign_preserves_source_native_semantics": True,
            "ready_for_full_admission_requires_delta_replay": True,
            "required_delta_replay": [
                "source_values_drive_state",
                "deterministic_seeded_replay",
                "reference_beats_no_action",
                "native_control_changes_backend",
                "no_safety_regression",
            ],
        },
        "inputs": {
            "cluster_root": str(cluster_root),
            "spot_ledger": {
                "path": str(spot_ledger_path),
                "sha256": _sha256(spot_ledger_path),
            },
            "archived_registries": archive_inputs,
            "n_embedded_archived_payloads": len(archived_registry_payloads or []),
        },
        "rows": rows,
        "ready_for_full_admission_candidates": [
            row
            for row in rows
            if row["final_disposition"] == "ready_for_full_admission"
        ],
        "summary": summary,
    }
    return canonicalize_repo_owned_paths(report, repo_root=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-root", type=Path, default=DEFAULT_CLUSTER_ROOT)
    parser.add_argument("--spot-ledger", type=Path, default=DEFAULT_SPOT_LEDGER)
    parser.add_argument("--archive-registry", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    embedded = [] if args.archive_registry else _git_archived_payloads()
    report = build_refinement(
        cluster_root=args.cluster_root,
        spot_ledger_path=args.spot_ledger,
        archived_registry_paths=args.archive_registry,
        archived_registry_payloads=embedded,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
