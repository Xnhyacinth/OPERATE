"""Candidate builders and executable probes for source-native Alibaba traces."""

from __future__ import annotations

import csv
import hashlib
import tarfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from .backends.alibaba_openb_backend import (
    NODE_FIELDS,
    OPENB_RECIPE,
    OPENB_SOURCE_SCHEMA,
    POD_FIELDS,
    AlibabaOpenBBackend,
    _normalize_node,
    _normalize_pod,
    _stable_digest,
)
from .seeds.schema import DatacenterScenarioSeed, Provenance


REPO_ROOT = Path(__file__).resolve().parents[2]


DLRM_FIELDS = {
    "instance_sn",
    "role",
    "app_name",
    "cpu_request",
    "cpu_limit",
    "gpu_request",
    "gpu_limit",
    "rdma_request",
    "rdma_limit",
    "memory_request",
    "memory_limit",
    "disk_request",
    "disk_limit",
    "max_instance_per_node",
    "creation_time",
    "scheduled_time",
    "deletion_time",
}
GENAI_FIELDS = {
    "gmt_create",
    "predict_type",
    "predict_status",
    "exec_time_seconds",
    "groupId",
    "prompt_length",
    "negative_prompt_length",
    "num_images_per_prompt",
    "num_inference_steps",
    "checkpoint_model_version_id",
    "num_lora",
}
GENAI_RUNTIME_ARCHIVES = (
    "basemodel_update_latency_anon.tar.gz",
    "lora_update_latency_anon.tar.gz",
    "pipeline_inference_data_anon.tar.gz",
    "pipeline_update_latency_anon.tar.gz",
    "pod_gpu_duty_cycle_anon.tar.gz",
    "pod_gpu_memory_used_bytes_anon.tar.gz",
    "queue_rt_raw_anon.tar.gz",
    "queue_size_raw_anon.tar.gz",
)

ALIBABA_TRACE_DIMENSION_APPLICABILITY: dict[str, dict[str, Any]] = {
    "system_survival": {
        "applicable": True,
        "reason": "gpu_queue_preemption_and_reservation_tick_records_available",
    },
    "economic_cost": {
        "applicable": True,
        "reason": "trace_wait_sla_and_wait_counterfactual_costs_available",
    },
    "safety_violation": {
        "applicable": True,
        "reason": "sla_deadline_and_capacity_shortfall_records_available",
    },
    "weighted_equity_score": {
        "applicable": False,
        "reason": "alibaba_trace_sim_has_no_source_grounded_load_class_criticality_ledger",
    },
    "ethical_quality": {
        "applicable": False,
        "reason": "alibaba_trace_sim_has_no_moral_dilemma_payload",
    },
    "stakeholder_management": {
        "applicable": True,
        "reason": "datacenter_native_job_class_trust_manager_available",
    },
    "adaptive_replanning": {
        "applicable": True,
        "reason": "state_changing_queue_preempt_and_reserve_tools_available",
    },
    "information_efficiency": {
        "applicable": True,
        "reason": "partial_cluster_observation_and_inspection_tools_available",
    },
    "foresight_score": {
        "applicable": False,
        "reason": "baseline_oracle_does_not_emit_commit_to_plan_predictions",
    },
    "optimality_gap": {
        "applicable": False,
        "reason": "no_lp_or_milp_reference_optimum_for_alibaba_trace_sim",
    },
    "counterfactual_prevention": {
        "applicable": True,
        "reason": "deterministic_masked_action_replay_over_same_trace_window",
    },
    "stakeholder_equity": {
        "applicable": True,
        "reason": "cross_job_class_outcome_balance_recorded",
    },
    "tool_use_efficiency": {
        "applicable": True,
        "reason": "tool_protocol_call_outcome_and_budget_evidence_available",
    },
}

ALIBABA_OPENB_DIMENSION_APPLICABILITY: dict[str, dict[str, Any]] = {
    "system_survival": {
        "applicable": True,
        "reason": "openb_pod_queue_and_unfinished_work_tick_records_available",
    },
    "economic_cost": {
        "applicable": True,
        "reason": "openb_compute_queue_sla_migration_and_unfinished_costs_available",
    },
    "safety_violation": {
        "applicable": True,
        "reason": "openb_qos_sla_and_capacity_shortfall_records_available",
    },
    "weighted_equity_score": {
        "applicable": False,
        "reason": "openb_has_no_scenario_load_assignment_criticality_ledger",
    },
    "ethical_quality": {
        "applicable": False,
        "reason": "openb_has_no_moral_dilemma_payload",
    },
    "stakeholder_management": {
        "applicable": True,
        "reason": "openb_source_qos_tenant_trust_manager_available",
    },
    "adaptive_replanning": {
        "applicable": True,
        "reason": "state_changing_placement_policy_place_and_migrate_tools_available",
    },
    "information_efficiency": {
        "applicable": True,
        "reason": "partial_pod_node_observation_and_arrival_forecast_tools_available",
    },
    "foresight_score": {
        "applicable": False,
        "reason": "baseline_oracle_does_not_emit_commit_to_plan_predictions",
    },
    "optimality_gap": {
        "applicable": False,
        "reason": "no_certified_openb_placement_optimum",
    },
    "counterfactual_prevention": {
        "applicable": True,
        "reason": "deterministic_masked_action_replay_over_same_openb_source_graph",
    },
    "stakeholder_equity": {
        "applicable": True,
        "reason": "cross_qos_tenant_outcome_balance_recorded",
    },
    "tool_use_efficiency": {
        "applicable": True,
        "reason": "tool_protocol_call_outcome_and_budget_evidence_available",
    },
}


def datacenter_dimension_applicability(
    backend_kind: str,
) -> dict[str, dict[str, Any]]:
    """Return the evidence-surface contract for a source-native backend."""
    templates = {
        "alibaba_trace_sim": ALIBABA_TRACE_DIMENSION_APPLICABILITY,
        "alibaba_openb_gpu_placement": ALIBABA_OPENB_DIMENSION_APPLICABILITY,
    }
    try:
        return deepcopy(templates[backend_kind])
    except KeyError as exc:
        raise ValueError(
            f"datacenter_dimension_applicability_unsupported:{backend_kind}"
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_source_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _read_rows(
    path: Path, required_fields: set[str], normalizer: Any | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        missing = required_fields - set(fields)
        if missing:
            raise ValueError(f"source_fields_missing:{','.join(sorted(missing))}")
        rows = [normalizer(row) if normalizer else dict(row) for row in reader]
    return rows, fields


def _pod_window_score(rows: list[dict[str, Any]]) -> tuple[float, ...]:
    gpu_pressure = sum(
        int(row["gpu_count"]) * float(row["gpu_milli_per_gpu"]) / 1000.0 for row in rows
    )
    fractional = sum(0 < float(row["gpu_milli_per_gpu"]) < 1000.0 for row in rows)
    specified = sum(bool(row["compatible_gpu_models"]) for row in rows)
    return (
        float(fractional),
        float(specified),
        float(len({row["qos"] for row in rows})),
        gpu_pressure,
        float(len({row["creation_time"] for row in rows})),
    )


def _select_pod_indices(pods: list[dict[str, Any]], window_size: int) -> list[int]:
    if not 2 <= window_size <= len(pods):
        raise ValueError("openb_pod_window_size_invalid")
    best: tuple[tuple[float, ...], int] | None = None
    for start in range(0, len(pods) - window_size + 1):
        score = _pod_window_score(pods[start : start + window_size])
        candidate = (score, -start)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    start = -best[1]
    return list(range(start, start + window_size))


def _select_node_indices(
    nodes: list[dict[str, Any]],
    pods: list[dict[str, Any]],
    max_nodes: int,
) -> list[int]:
    if max_nodes < 1:
        raise ValueError("openb_max_nodes_invalid")
    model_needs = Counter(
        model for pod in pods for model in pod["compatible_gpu_models"]
    )
    chosen: list[int] = []
    for model, _count in sorted(
        model_needs.items(), key=lambda item: (-item[1], item[0])
    ):
        compatible = [
            (int(node["gpu_count"]), index)
            for index, node in enumerate(nodes)
            if node["gpu_model"] == model and index not in chosen
        ]
        if compatible and len(chosen) < max_nodes:
            chosen.append(min(compatible)[1])
    remaining = sorted(
        (
            int(node["gpu_count"]),
            float(node["cpu_milli"]),
            index,
        )
        for index, node in enumerate(nodes)
        if index not in chosen and int(node["gpu_count"]) > 0
    )
    for _gpu, _cpu, index in remaining:
        if len(chosen) >= max_nodes:
            break
        chosen.append(index)
    selected_models = {nodes[index]["gpu_model"] for index in chosen}
    incompatible = [
        pod["pod_id"]
        for pod in pods
        if pod["compatible_gpu_models"]
        and not selected_models.intersection(pod["compatible_gpu_models"])
    ]
    if incompatible:
        raise ValueError("openb_selected_nodes_do_not_cover_pod_gpu_models")
    return sorted(chosen)


def build_openb_candidate(
    *,
    node_path: Path,
    pod_path: Path,
    max_nodes: int = 8,
    pod_window_size: int = 64,
    horizon_ticks: int = 16,
    tick_minutes: int = 10,
    seed: int = 42,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build a source-locked OpenB candidate without claiming release admission."""
    nodes, _ = _read_rows(node_path, NODE_FIELDS, _normalize_node)
    pods, _ = _read_rows(pod_path, POD_FIELDS, _normalize_pod)
    pod_indices = _select_pod_indices(pods, min(pod_window_size, len(pods)))
    selected_pods = [pods[index] for index in pod_indices]
    node_indices = _select_node_indices(nodes, selected_pods, max_nodes)
    selected_nodes = [nodes[index] for index in node_indices]
    node_source = _portable_source_path(node_path, repo_root)
    pod_source = _portable_source_path(pod_path, repo_root)
    source_graph_sha256 = _stable_digest(
        {"nodes": selected_nodes, "pods": selected_pods}
    )
    transform = {
        "source_schema": OPENB_SOURCE_SCHEMA,
        "recipe_version": OPENB_RECIPE,
        "source_file_roles": {
            "node_inventory": node_source,
            "pod_trace": pod_source,
        },
        "node_trace_sha256": _sha256(node_path),
        "pod_trace_sha256": _sha256(pod_path),
        "node_row_indices": node_indices,
        "pod_row_indices": pod_indices,
        "selected_nodes_sha256": _stable_digest(selected_nodes),
        "selected_pods_sha256": _stable_digest(selected_pods),
        "source_graph_sha256": source_graph_sha256,
    }
    identity = source_graph_sha256[:16]
    seed_obj = DatacenterScenarioSeed(
        seed_id=f"alibaba_openb_gpu_placement_{identity}",
        family="gpu_sharing_placement_and_rescheduling",
        backend_kind="alibaba_openb_gpu_placement",
        backend_config={
            "source_transform": transform,
            "initial_placement_policy": "first_fit",
            "autonomous_placement": True,
        },
        horizon_ticks=max(4, int(horizon_ticks)),
        tick_minutes=max(1, int(tick_minutes)),
        seed=int(seed),
        difficulty_mode="deep_planning",
        difficulty_level="high",
        provenance=Provenance(
            data_source="Alibaba cluster-trace-gpu-v2023 OpenB",
            files=[node_source, pod_source],
            url="https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2023",
            lock_strategy="upstream_git_commit_raw_sha256_and_explicit_row_graph",
            license="research trace terms; upstream repository license applies",
            time_window={
                "selection_kind": "explicit_raw_row_graph",
                "node_row_indices": node_indices,
                "pod_row_indices": pod_indices,
                "source_graph_sha256": source_graph_sha256,
            },
        ),
    )
    scenario = seed_obj.to_dict()
    scenario.update(
        {
            "scenario_id": (
                "datacenter/alibaba_openb_gpu_placement_candidate/"
                f"deep_planning/high/{identity}"
            ),
            "scenario_signature": seed_obj.signature(),
            "source_contract": {
                "schema_version": "source_contract.v1",
                "runtime_input": [node_source, pod_source],
                "derived_window": {
                    "recipe_version": OPENB_RECIPE,
                    "sha256": source_graph_sha256,
                },
                "derived_source_graph": {
                    "recipe_version": OPENB_RECIPE,
                    "sha256": source_graph_sha256,
                },
                "source_values_drive_state": True,
                "procedural_stress": None,
            },
            "candidate_gate": {
                "status": "requires_bounded_delta_replay",
                "core_admission_claimed": False,
                "decision_axes": [
                    "fractional_gpu_fragmentation",
                    "gpu_model_compatibility",
                    "multi_resource_node_contention",
                    "qos_aware_placement_and_migration",
                ],
                "required_gates": [
                    "source_values_drive_state",
                    "deterministic_seeded_replay",
                    "reference_beats_no_action",
                    "native_control_changes_backend",
                    "no_safety_regression",
                ],
            },
        }
    )
    return scenario


def probe_openb_candidate(scenario: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic source and native-control probes without admission."""
    seed = DatacenterScenarioSeed(
        **{
            key: value
            for key, value in scenario.items()
            if key in DatacenterScenarioSeed.__dataclass_fields__
        }
    )
    # Rebuild nested provenance when a JSON-style mapping is supplied.
    if isinstance(seed.provenance, dict):
        seed.provenance = Provenance(**seed.provenance)
    first = AlibabaOpenBBackend()
    second = AlibabaOpenBBackend()
    first.reset(seed)
    second.reset(seed)
    first_rows = []
    second_rows = []
    for tick in range(seed.horizon_ticks):
        first_rows.append(first.tick(tick).__dict__)
        second_rows.append(second.tick(tick).__dict__)
    deterministic = first_rows == second_rows and first.snapshot() == second.snapshot()

    no_action_scenario = deepcopy(scenario)
    no_action_scenario["backend_config"]["autonomous_placement"] = False
    no_action_seed = DatacenterScenarioSeed(
        **{
            key: value
            for key, value in no_action_scenario.items()
            if key in DatacenterScenarioSeed.__dataclass_fields__
        }
    )
    if isinstance(no_action_seed.provenance, dict):
        no_action_seed.provenance = Provenance(**no_action_seed.provenance)
    no_action = AlibabaOpenBBackend()
    no_action.reset(no_action_seed)
    for tick in range(no_action_seed.horizon_ticks):
        no_action.tick(tick)
    reference_cost = sum(first.ground_truth_costs().values())
    no_action_cost = sum(no_action.ground_truth_costs().values())
    reference_headroom = no_action_cost - reference_cost

    manual_scenario = deepcopy(scenario)
    manual_scenario["backend_config"]["autonomous_placement"] = False
    manual_seed = DatacenterScenarioSeed(
        **{
            key: value
            for key, value in manual_scenario.items()
            if key in DatacenterScenarioSeed.__dataclass_fields__
        }
    )
    if isinstance(manual_seed.provenance, dict):
        manual_seed.provenance = Provenance(**manual_seed.provenance)
    controlled = AlibabaOpenBBackend()
    controlled.reset(manual_seed)
    controlled.tick(0)
    queued = controlled.placement_state()["queued_pods"]
    control_changed_state = False
    if queued and queued[0]["compatible_node_ids"]:
        before = controlled.snapshot()["placement"]["assignment_digest"]
        result = controlled.apply_tool_effect(
            "place_pod",
            {
                "pod_id": queued[0]["pod_id"],
                "node_id": queued[0]["compatible_node_ids"][0],
            },
            current_tick=0,
        )
        after = controlled.snapshot()["placement"]["assignment_digest"]
        control_changed_state = result.get("_status") == "ok" and before != after
    trace = controlled.protocol21_source_trace()
    return {
        "status": (
            "passed"
            if deterministic
            and control_changed_state
            and trace["state_effect_observed"]
            and reference_headroom > 0
            else "held"
        ),
        "deterministic_replay": deterministic,
        "native_control_changes_backend": control_changed_state,
        "source_values_drive_state": trace["state_effect_observed"],
        "reference_beats_no_action": reference_headroom > 0,
        "reference_cost": round(reference_cost, 6),
        "no_action_cost": round(no_action_cost, 6),
        "reference_headroom": round(reference_headroom, 6),
        "source_graph_sha256": trace["source_graph_sha256"],
        "release_admission": False,
    }


def probe_dlrm_source(path: Path) -> dict[str, Any]:
    """Parse DLRM rows and report why the OpenB placement kernel is insufficient."""
    rows, fields = _read_rows(path, DLRM_FIELDS)
    roles = Counter(str(row["role"]) for row in rows)
    apps = {str(row["app_name"]) for row in rows}
    missing_creation = sum(not str(row["creation_time"]).strip() for row in rows)
    return {
        "parser_status": "passed",
        "source_sha256": _sha256(path),
        "row_count": len(rows),
        "fields": fields,
        "roles": dict(sorted(roles.items())),
        "application_count": len(apps),
        "preexisting_instances_without_creation_time": missing_creation,
        "required_native_dimensions": [
            "app_density",
            "cn_hn_role_coupling",
            "disk",
            "rdma",
        ],
        "required_native_controls": [
            "place_cn_instance",
            "place_hn_instance",
            "reserve_rdma_capacity",
            "reschedule_application_group",
        ],
        "control_probe": {
            "openb_kernel_reusable_without_semantic_loss": False,
            "reason": (
                "OpenB has CPU/memory/GPU nodes but no RDMA/disk capacity, application "
                "density constraint or coupled CN/HN service state."
            ),
        },
        "disposition": "redesign",
        "blockers": [
            "openb_kernel_missing_rdma_disk_density_and_role_coupling",
            "source_node_topology_absent",
        ],
    }


def _archive_csv_fields(path: Path) -> list[str] | None:
    try:
        with tarfile.open(path, "r:gz") as archive:
            member = next(
                (item for item in archive.getmembers() if item.isfile()), None
            )
            if member is None:
                return None
            stream = archive.extractfile(member)
            if stream is None:
                return None
            first_line = stream.readline().decode("utf-8").strip()
    except (OSError, tarfile.TarError, UnicodeDecodeError):
        return None
    return next(csv.reader([first_line]), [])


def probe_genai_source(
    path: Path, *, source_root: Path | None = None
) -> dict[str, Any]:
    """Parse GenAI request rows and test whether serving-state joins are executable."""
    rows, fields = _read_rows(path, GENAI_FIELDS)
    root = source_root or path.parent
    runtime_headers = {
        name: _archive_csv_fields(root / name)
        for name in GENAI_RUNTIME_ARCHIVES
        if (root / name).is_file()
    }
    correlated_available = len(runtime_headers) == len(GENAI_RUNTIME_ARCHIVES) and all(
        runtime_headers.values()
    )
    request_has_container = "container_ip" in fields
    runtime_has_container = bool(runtime_headers) and all(
        "container_ip" in (header or []) for header in runtime_headers.values()
    )
    blockers = ["serving_capacity_topology_absent"]
    if not correlated_available:
        blockers.append("correlated_runtime_channels_unavailable")
    elif runtime_has_container and not request_has_container:
        blockers.append("request_runtime_join_key_absent")
    blockers.append("openb_kernel_missing_model_lifecycle_batching_and_replica_state")
    return {
        "parser_status": "passed",
        "source_sha256": _sha256(path),
        "row_count": len(rows),
        "fields": fields,
        "request_types": dict(
            sorted(Counter(row["predict_type"] for row in rows).items())
        ),
        "model_count": len({row["checkpoint_model_version_id"] for row in rows}),
        "group_count": len({row["groupId"] for row in rows}),
        "runtime_archive_headers": runtime_headers,
        "correlated_runtime_channels_available": correlated_available,
        "request_to_runtime_join_available": request_has_container
        and runtime_has_container,
        "required_native_controls": [
            "route_lora_request",
            "schedule_model_update",
            "set_dynamic_batch",
            "scale_model_replica",
        ],
        "control_probe": {
            "openb_kernel_reusable_without_semantic_loss": False,
            "reason": (
                "Serving requires replica/model residency, batching, cold-start/update "
                "latency and request routing state absent from the placement kernel."
            ),
        },
        "disposition": "redesign",
        "blockers": sorted(blockers),
    }
