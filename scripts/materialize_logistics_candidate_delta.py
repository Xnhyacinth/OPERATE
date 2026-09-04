#!/usr/bin/env python3
"""Materialize source-locked logistics/manufacturing candidate deltas.

Only independent candidate rows already classified
``ready_for_full_admission`` by the refinement ledger are considered. Historical
ledgers that used the legacy disposition remain readable. Every selected row is
either materialized through an existing current-tree seed builder or retained
as a machine-readable blocker. This script never edits the active release or
active Core scenarios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.suite_identity import recompute_signature_with_seed  # noqa: E402
from core.protocol21_evidence import canonicalize_repo_owned_paths  # noqa: E402
from evaluation.dimension_applicability import (  # noqa: E402
    dimension_applicability_contract_issue,
)
from domains.logistics.backends.dynasched_flexible_job_shop import (  # noqa: E402
    DYNASCHED_RUNTIME_CODE_TREE_SHA256,
    DYNASCHED_RUNTIME_COMMIT,
    DYNASCHED_RUNTIME_VERSION,
)
from domains.logistics.backends.job_shop import (  # noqa: E402
    _normalize_j2_breakdown,
    _normalize_j2_operation_graph,
    _parsed_from_j2_graph,
)
from domains.logistics.seeds.from_jsplib import (  # noqa: E402
    build_job_shop_dispatch_seed,
)
from domains.logistics.seeds.from_m5_orgym import (  # noqa: E402
    M5_ROOT,
    M5_REQUIRED_FILES,
    M5OrgymWindow,
    _read_first_prices,
    _read_sales_rows,
    build_m5_orgym_inventory_seed,
    verify_m5_orgym_source_lock,
)
from domains.logistics.seeds.from_vrplib import (  # noqa: E402
    build_cvrp_dispatch_seed,
    build_vrptw_dispatch_seed,
)
from scripts.build_external_direct_pilot_v2 import (  # noqa: E402
    DEFAULT_REALM_LOCK,
    REALM_COMMIT,
)
from scripts.build_protocol21_candidate_source_suite import build_suite  # noqa: E402


DEFAULT_LEDGER = (
    REPO_ROOT / ".hl/artifacts/operate_v058_logistics_manufacturing_refinement.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".hl/artifacts/operate_v058_logistics_candidate_delta"

_SUPPORTED_SCOPE = "candidate"
_SUPPORTED_DISPOSITION = "ready_for_full_admission"
_LEGACY_SUPPORTED_DISPOSITION = "core_ready"
_READABLE_DISPOSITIONS = {
    _SUPPORTED_DISPOSITION,
    _LEGACY_SUPPORTED_DISPOSITION,
}
DYNASCHED_EVENT_DRIVEN_REPLAY_CONTRACT = "dynasched_native_boundary_work_ticks_v1"
DYNASCHED_EVENT_DRIVEN_MAX_WORK_TICKS = 32768


class MaterializationBlocked(Exception):
    """A selected candidate cannot use a semantically exact existing builder."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(candidate_id: str) -> int:
    return int.from_bytes(hashlib.sha256(candidate_id.encode()).digest()[:4], "big")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_.-")
    return slug or hashlib.sha256(value.encode()).hexdigest()[:16]


def _source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("source_metadata")
    if not isinstance(value, dict):
        raise MaterializationBlocked(
            "source_metadata_missing", "selected row has no source_metadata object"
        )
    return value


def _source_path(metadata: dict[str, Any]) -> tuple[str, Path, str]:
    declared = str(metadata.get("path") or "").strip()
    expected = str(metadata.get("sha256") or "").strip().lower()
    if not declared or len(expected) != 64:
        raise MaterializationBlocked(
            "source_lock_incomplete", "source path and sha256 are required"
        )
    path = Path(declared)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        raise MaterializationBlocked(
            "source_asset_unavailable", f"locked source is unavailable: {declared}"
        )
    actual = _sha256(path)
    if actual != expected:
        raise MaterializationBlocked(
            "source_hash_mismatch",
            f"locked source hash mismatch: expected={expected} actual={actual}",
        )
    return declared, path, expected


def _build_jsplib_seed_from_locked_source(
    *,
    instance: str,
    source_path: Path,
    source_sha256: str,
    seed: int,
) -> tuple[Any, bool]:
    """Use the row-bound source lock when the optional mirror manifest is absent."""
    kwargs = {
        "instance": instance,
        "seed": seed,
        "difficulty_mode": "deep_planning",
        "difficulty_level": "high",
    }
    try:
        return build_job_shop_dispatch_seed(**kwargs), False
    except ValueError as exc:
        if "missing CHECKSUMS entry" not in str(exc):
            raise

    metadata_path = REPO_ROOT / "works/JSPLIB-Instances/instances.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    matching = [
        dict(row)
        for row in metadata
        if isinstance(row, dict) and str(row.get("name") or "") == instance
    ]
    if len(matching) != 1:
        raise MaterializationBlocked(
            "jsplib_metadata_identity_missing",
            f"expected one JSPLIB metadata row for {instance!r}",
        )

    with tempfile.TemporaryDirectory(prefix="operate-jsplib-lock-") as temporary:
        root = Path(temporary)
        relative = Path("instances") / source_path.name
        target = root / relative
        target.parent.mkdir(parents=True)
        shutil.copyfile(source_path, target)
        matching[0]["path"] = relative.as_posix()
        (root / "instances.json").write_text(
            json.dumps(matching, sort_keys=True), encoding="utf-8"
        )
        (root / "CHECKSUMS.txt").write_text(
            f"{source_sha256}  {relative.as_posix()}\n", encoding="utf-8"
        )
        return build_job_shop_dispatch_seed(root=root, **kwargs), True


def _source_contract(
    runtime_input: list[str],
    hashes: dict[str, str],
    *,
    derivation_input: list[str] | None = None,
    implementation_asset: list[str] | None = None,
    metadata: list[str] | None = None,
    license_files: list[str] | None = None,
    derived_window: dict[str, str] | None = None,
) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "runtime_input": runtime_input,
        "derivation_input": derivation_input or [],
        "implementation_asset": implementation_asset or [],
        "metadata": metadata or [],
        "license": license_files or [],
        "file_sha256s": hashes,
    }
    if derived_window is not None:
        contract["derived_window"] = derived_window
    return contract


def _routing_dimension_applicability() -> dict[str, dict[str, Any]]:
    """Applicability shared by the native CVRP and VRPTW control surfaces."""
    return {
        "system_survival": {
            "applicable": True,
            "reason": "failed_route_and_unserved_customer_tick_records_available",
        },
        "economic_cost": {
            "applicable": True,
            "reason": "route_distance_duration_and_wait_counterfactual_available",
        },
        "safety_violation": {
            "applicable": True,
            "reason": "capacity_time_window_and_failed_route_records_available",
        },
        "weighted_equity_score": {
            "applicable": False,
            "reason": "vrplib_candidate_has_no_source_grounded_priority_classes",
        },
        "ethical_quality": {
            "applicable": False,
            "reason": "vrplib_candidate_has_no_moral_dilemma_payload",
        },
        "stakeholder_management": {
            "applicable": False,
            "reason": "vrplib_candidate_has_no_trust_manager_delta",
        },
        "adaptive_replanning": {
            "applicable": True,
            "reason": "state_changing_route_dispatch_tools_available",
        },
        "information_efficiency": {
            "applicable": True,
            "reason": "partial_route_observation_and_inspection_tools_available",
        },
        "foresight_score": {
            "applicable": False,
            "reason": "reference_agents_do_not_emit_commit_to_plan_predictions",
        },
        "optimality_gap": {
            "applicable": True,
            "reason": "deterministic_offline_routing_reference_available",
        },
        "counterfactual_prevention": {
            "applicable": True,
            "reason": "deterministic_masked_action_replay_over_same_vrplib_seed",
        },
        "stakeholder_equity": {
            "applicable": False,
            "reason": "vrplib_candidate_has_no_cross_stakeholder_outcome_ledger",
        },
        "tool_use_efficiency": {
            "applicable": True,
            "reason": "tool_protocol_call_outcome_and_budget_evidence_available",
        },
    }


def _complete_dimension_applicability(body: dict[str, Any]) -> None:
    """Attach the backend-owned scorer contract without model-derived inference."""
    config = body.get("backend_config")
    if not isinstance(config, dict):
        raise MaterializationBlocked(
            "backend_config_missing", "seed builder did not emit backend_config"
        )
    backend_kind = str(body.get("backend_kind") or "")
    applicability = config.get("dimension_applicability")
    if backend_kind in {"pyvrp_cvrp", "pyvrp_vrptw"}:
        applicability = _routing_dimension_applicability()
    elif backend_kind == "dynasched_flexible_job_shop" and isinstance(
        applicability, dict
    ):
        applicability = {
            **applicability,
            "system_survival": {
                "applicable": True,
                "reason": "unfinished_operation_and_machine_downtime_tick_records_available",
            },
            "economic_cost": {
                "applicable": True,
                "reason": "native_makespan_cost_and_wait_counterfactual_available",
            },
            "safety_violation": {
                "applicable": True,
                "reason": "precedence_machine_eligibility_and_downtime_constraints_available",
            },
            "weighted_equity_score": {
                "applicable": False,
                "reason": "dynasched_bundle_has_no_source_grounded_criticality_classes",
            },
            "ethical_quality": {
                "applicable": False,
                "reason": "dynasched_bundle_has_no_moral_dilemma_payload",
            },
            "stakeholder_management": {
                "applicable": False,
                "reason": "dynasched_bundle_has_no_stakeholder_trust_model",
            },
            "adaptive_replanning": {
                "applicable": True,
                "reason": "source_driven_shop_events_require_dynamic_rescheduling",
            },
            "information_efficiency": {
                "applicable": True,
                "reason": "partial_shop_state_is_exposed_by_native_investigation_tool",
            },
            "foresight_score": {
                "applicable": False,
                "reason": "reference_agents_do_not_emit_commit_to_plan_predictions",
            },
            "stakeholder_equity": {
                "applicable": False,
                "reason": "dynasched_bundle_has_no_cross_stakeholder_outcome_ledger",
            },
            "tool_use_efficiency": {
                "applicable": True,
                "reason": "tool_protocol_call_outcome_and_budget_evidence_available",
            },
        }
    elif backend_kind == "orgym_invmgmt" and isinstance(applicability, dict):
        applicability = dict(applicability)
        applicability["stakeholder_equity"] = {
            "applicable": False,
            "reason": "orgym_inventory_m5_release_has_no_cross_stakeholder_outcome_ledger",
        }
        applicability["tool_use_efficiency"] = {
            "applicable": True,
            "reason": "tool_protocol_call_outcome_and_budget_evidence_available",
        }
    elif (
        backend_kind == "jsplib_job_shop"
        and config.get("source_mode") == "realm_j2_json"
        and isinstance(applicability, dict)
    ):
        applicability = {
            **applicability,
            "system_survival": {
                "applicable": True,
                "reason": "unscheduled_operation_and_machine_outage_tick_records_available",
            },
            "economic_cost": {
                "applicable": True,
                "reason": "makespan_cost_and_wait_counterfactual_available",
            },
            "safety_violation": {
                "applicable": True,
                "reason": "precedence_machine_capacity_and_outage_constraints_available",
            },
            "weighted_equity_score": {
                "applicable": False,
                "reason": "source_instance_has_no_stakeholder_priority_classes",
            },
            "ethical_quality": {
                "applicable": False,
                "reason": "source_instance_has_no_moral_dilemma_payload",
            },
            "stakeholder_management": {
                "applicable": False,
                "reason": "source_instance_has_no_stakeholder_trust_model",
            },
            "adaptive_replanning": {
                "applicable": True,
                "reason": "source_native_machine_breakdown_requires_repair_and_rescheduling",
            },
            "information_efficiency": {
                "applicable": True,
                "reason": "partial_shop_observation_and_inspection_tools_available",
            },
            "foresight_score": {
                "applicable": False,
                "reason": "reference_agents_do_not_emit_commit_to_plan_predictions",
            },
            "stakeholder_equity": {
                "applicable": False,
                "reason": "source_instance_has_no_cross_stakeholder_outcome_ledger",
            },
            "tool_use_efficiency": {
                "applicable": True,
                "reason": "tool_protocol_call_outcome_and_budget_evidence_available",
            },
        }
    elif (
        backend_kind == "jsplib_job_shop"
        and config.get("source_mode") != "realm_j2_json"
        and isinstance(applicability, dict)
    ):
        applicability = {
            **applicability,
            "system_survival": {
                "applicable": True,
                "reason": "unscheduled_operation_and_makespan_tick_records_available",
            },
            "economic_cost": {
                "applicable": True,
                "reason": "makespan_cost_and_wait_counterfactual_available",
            },
            "adaptive_replanning": {
                "applicable": False,
                "reason": "static_job_shop_has_no_exogenous_disruption_window",
            },
            "information_efficiency": {
                "applicable": True,
                "reason": "partial_shop_observation_and_inspection_tools_available",
            },
            "foresight_score": {
                "applicable": False,
                "reason": "reference_agents_do_not_emit_commit_to_plan_predictions",
            },
            "stakeholder_equity": {
                "applicable": False,
                "reason": "classic_jsplib_has_no_cross_stakeholder_outcome_ledger",
            },
            "tool_use_efficiency": {
                "applicable": True,
                "reason": "tool_protocol_call_outcome_and_budget_evidence_available",
            },
        }
    applicability_issue = dimension_applicability_contract_issue(applicability)
    if applicability_issue is not None and applicability_issue[0] == "incomplete":
        raise MaterializationBlocked(
            "dimension_applicability_incomplete",
            f"{backend_kind} did not emit the complete scorer contract",
        )
    if applicability_issue is not None:
        raise MaterializationBlocked(
            "dimension_applicability_invalid",
            f"invalid scorer contract for {backend_kind}:{applicability_issue[1]}",
        )
    config["dimension_applicability"] = applicability


def _materialize_jsplib(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _source_metadata(row)
    instance = str(metadata.get("instance_name") or "").strip()
    declared, source_path, digest = _source_path(metadata)
    if not instance:
        raise MaterializationBlocked(
            "jsplib_instance_missing", "JSPLIB instance_name is required"
        )
    seed, used_row_lock_fallback = _build_jsplib_seed_from_locked_source(
        instance=instance,
        source_path=source_path,
        source_sha256=digest,
        seed=_seed(str(row["candidate_id"])),
    )
    body = seed.to_dict()
    metadata_files = [
        path
        for path in ("works/JSPLIB-Instances/instances.json",)
        if (REPO_ROOT / path).is_file()
    ]
    license_files = [
        path
        for path in ("works/JSPLIB-Instances/LICENSE",)
        if (REPO_ROOT / path).is_file()
    ]
    if used_row_lock_fallback:
        body["backend_config"].pop("checksum_manifest", None)
        body["backend_config"]["source_lock_strategy"] = (
            "candidate_refinement_ledger_sha256"
        )
        body["provenance"]["files"] = [declared, *metadata_files]
    horizon = int(body["horizon_ticks"])
    body["backend_config"]["task_contract"] = {
        "contract": "logistics.job_shop.multi_wave_dispatch.v1",
        "native_controls": [
            "dispatch_ready_operations",
            "dispatch_job_operation",
        ],
    }
    body["backend_config"]["task_requirements"] = {
        "min_distinct_control_ticks": 2,
        "min_distinct_physical_tools": 1,
        "ordered_tool_milestones": [
            {
                "tool": "dispatch_ready_operations",
                "not_after_tick": horizon - 1,
            },
            {
                "tool": "dispatch_ready_operations",
                "not_after_tick": horizon - 1,
            },
        ],
    }
    body["source_contract"] = _source_contract(
        [declared],
        {declared: digest},
        metadata=metadata_files,
        license_files=license_files,
    )
    body["procedural_stress"] = {
        "label": "none_required",
        "source_native": True,
    }
    return body


def _routing_builder_source(path: str) -> str:
    if path.startswith("works/PyVRP-Instances/"):
        return "vrplib"
    if re.fullmatch(r"works/VRPLIB/tests/data/[^/]+\.vrp", path):
        return "vrplib_package_test_data"
    if path.startswith("works/VRPLIB/tests/data/cvrplib/Vrp-Set-Solomon/"):
        return "vrplib_package_solomon"
    if path.startswith("works/VRPLIB/tests/data/cvrplib/Vrp-Set-X/X/"):
        return "vrplib_package_x_set"
    if path.startswith(
        "works/VRPLIB/tests/data/lkh-3/CVRPTW/INSTANCES/"
    ) and path.endswith(".vrptw"):
        return "vrplib_package_lkh_cvrptw"
    if path.startswith(
        "works/VRPLIB/tests/data/lkh-3/CVRP/INSTANCES/"
    ) and path.endswith(".vrp"):
        return "vrplib_package_lkh_cvrp"
    if re.fullmatch(r"works/VRPLIB/tests/data/cvrplib/[^/]+\.vrp", path):
        return "vrplib_package_cvrplib_root"
    raise MaterializationBlocked(
        "routing_source_path_not_supported_by_seed_builder",
        f"no exact current-tree routing seed resolver for {path}",
    )


def _materialize_routing(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _source_metadata(row)
    instance = str(metadata.get("instance_name") or "").strip()
    variant = str(metadata.get("routing_variant") or "").strip().upper()
    declared, _, digest = _source_path(metadata)
    if not instance:
        raise MaterializationBlocked(
            "routing_instance_missing", "routing instance_name is required"
        )
    source_id = _routing_builder_source(declared)
    resolver_instance = Path(declared).stem if source_id != "vrplib" else instance
    kwargs = {
        "instance": resolver_instance,
        "source_id": source_id,
        "seed": _seed(str(row["candidate_id"])),
        "difficulty_mode": "deep_planning",
        "difficulty_level": "high",
    }
    if variant == "CVRP":
        seed = build_cvrp_dispatch_seed(**kwargs)
    elif variant in {"VRPTW", "CVRPTW"}:
        seed = build_vrptw_dispatch_seed(**kwargs)
    else:
        raise MaterializationBlocked(
            "routing_variant_not_supported_by_seed_builder",
            f"existing seed builders do not preserve routing variant {variant!r}",
        )
    body = seed.to_dict()
    if not body.get("perturbations"):
        raise MaterializationBlocked(
            "routing_typed_event_missing", "routing builder emitted no event"
        )
    provenance_files = (body.get("provenance") or {}).get("files") or []
    if not provenance_files or provenance_files[0] != declared:
        raise MaterializationBlocked(
            "routing_builder_source_identity_mismatch",
            f"builder resolved {provenance_files[:1]!r}, ledger locked {declared!r}",
        )
    body["source_contract"] = _source_contract([declared], {declared: digest})
    body["procedural_stress"] = {
        "label": "procedural",
        "source_native": False,
    }
    return body


def _materialize_m5(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _source_metadata(row)
    start = int(metadata.get("start_day") or 0)
    end = int(metadata.get("end_day") or 0)
    window_length = end - start + 1
    if window_length < 2:
        raise MaterializationBlocked(
            "m5_window_contract_invalid",
            f"candidate window must contain at least two days: d{start}_d{end}",
        )
    key = str(metadata.get("key") or "").strip()
    item = str(metadata.get("item_id") or "").strip()
    store = str(metadata.get("store_id") or "").strip()
    if not key or not item or not store:
        raise MaterializationBlocked(
            "m5_window_identity_incomplete", "M5 key/item_id/store_id are required"
        )
    sales_rows, prices, source_lock = _m5_shared_builder_inputs()
    seed = build_m5_orgym_inventory_seed(
        window=M5OrgymWindow(
            sku_store_key=key,
            item_id=item,
            store_id=store,
            start_day=start,
            seed=_seed(str(row["candidate_id"])),
            category_id=str(metadata.get("dept_id") or ""),
            difficulty_level="high",
            window_length_days=window_length,
        ),
        sales_rows=sales_rows,
        prices=prices,
        source_lock=source_lock,
    )
    body = seed.to_dict()
    body["backend_config"]["source_event_registry"] = {
        "demand_observation": {
            "actionable": True,
            "event_class": "task",
            "origin": "locked_m5_demand_stream",
            "unknown_events_actionable": False,
        }
    }
    body["backend_config"]["reference"] = {
        "type": "native_heuristic_policy",
        "policy": "base_stock_replenishment_v1",
        "formal_optimality_applicable": False,
    }
    body["backend_config"]["task_requirements"] = {
        "min_distinct_control_ticks": 2,
        "min_distinct_physical_tools": 1,
        "ordered_tool_milestones": [
            {
                "tool": "place_replenishment_order",
                "not_after_tick": window_length - 1,
            },
            {
                "tool": "place_replenishment_order",
                "not_after_tick": window_length - 1,
            },
        ],
    }
    required = list(M5_REQUIRED_FILES)
    hashes = {path: _sha256(REPO_ROOT / path) for path in required}
    demand_digest = str(body["backend_config"]["demand_stream_hash"]).removeprefix(
        "sha256:"
    )
    body["source_contract"] = _source_contract(
        required,
        hashes,
        metadata=["works/M5/source_lock.json"],
        implementation_asset=[
            "works/OR-Gym/or_gym/envs/supply_chain/inventory_management.py"
        ],
        license_files=["works/OR-Gym/LICENSE"],
        derived_window={
            "sha256": demand_digest,
            "recipe_version": "m5_orgym_inventory_window_v1",
        },
    )
    body["procedural_stress"] = {
        "label": "none_required",
        "source_native": True,
    }
    return body


@lru_cache(maxsize=1)
def _m5_shared_builder_inputs() -> tuple[
    dict[str, dict[str, str]], dict[tuple[str, str], float], dict[str, Any]
]:
    """Parse the shared M5 files once for a bulk candidate materialization."""
    return (
        _read_sales_rows(M5_ROOT / "sales_train_evaluation.csv"),
        _read_first_prices(M5_ROOT / "sell_prices.csv"),
        verify_m5_orgym_source_lock(source_root=M5_ROOT),
    )


def _materialize_realm(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _source_metadata(row)
    instance = str(metadata.get("instance_id") or "").strip()
    if not instance:
        raise MaterializationBlocked(
            "realm_instance_id_missing", "REALM selected instance_id is required"
        )
    declared, path, digest = _source_path(
        {**metadata, "sha256": metadata.get("source_sha256") or metadata.get("sha256")}
    )
    if declared != "works/REALM-Bench-direct-pilot/datasets/clean/JSSP/J2.json":
        raise MaterializationBlocked(
            "realm_j2_source_path_unexpected", f"unexpected J2 source: {declared}"
        )
    if digest != DEFAULT_REALM_LOCK.j2_sha256:
        raise MaterializationBlocked(
            "realm_j2_source_lock_mismatch", "J2 source differs from current lock"
        )
    payload = _realm_j2_payload(path)
    matches = [
        item
        for item in payload.get("instances") or []
        if isinstance(item, dict) and str(item.get("instance_id") or "") == instance
    ]
    if len(matches) != 1:
        raise MaterializationBlocked(
            "realm_j2_selected_instance_missing", f"selected={instance}"
        )
    sidecar_row = matches[0]
    graph = _normalize_j2_operation_graph(sidecar_row)
    parsed = _parsed_from_j2_graph(sidecar_row, graph)
    disruption_types = sorted(
        {
            str(event.get("type") or "")
            for event in sidecar_row.get("disruptions") or []
            if isinstance(event, dict)
        }
    )
    if disruption_types != ["machine_breakdown"]:
        raise MaterializationBlocked(
            "realm_source_event_backend_semantics_unimplemented",
            "current job-shop state transition implements exact J2 "
            f"machine_breakdown only; selected event types={disruption_types}",
        )
    breakdown = _normalize_j2_breakdown(sidecar_row)
    # A delayed native dispatch can resolve only one critical-path operation
    # every two supervisory ticks; leave the same fixed event-response margin.
    horizon = max(
        2 * int(parsed["operations"]) + 16,
        breakdown["trigger_tick"] + breakdown["duration_ticks"] + 3,
    )
    source_identity = f"realm_j2_ccby:{REALM_COMMIT}:{instance}:{digest}"
    selected_digest = hashlib.sha256(
        json.dumps(
            sidecar_row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    body = {
        "seed_id": f"realm_{_slug(instance)}",
        "family": "job_shop_dispatch",
        "domain": "logistics",
        "backend_kind": "jsplib_job_shop",
        "backend_config": {
            "instance_name": instance,
            "source_mode": "realm_j2_json",
            "source_integration_rung": "canonical_runtime_j2_sidecar",
            "release_ready": False,
            "release_reentry_ready": False,
            "source_denominator_key": source_identity,
            "job_shop": parsed,
            "reference": {
                "type": "native_heuristic_policy",
                "policy": "earliest_completion_ready_operation_v1",
                "formal_optimality_applicable": False,
                "headroom_contract": "executable_policy_vs_no_action_replay",
            },
            "external_source_assets": {
                "j2_event_sidecar": {
                    "path": declared,
                    "sha256": digest,
                    "git_commit": REALM_COMMIT,
                    "selected_instance_id": instance,
                    "canonical_runtime_source": True,
                    "license": "CC-BY-4.0",
                    "runtime_consumed": True,
                    "converter_consumed": False,
                }
            },
            "source_axes": {
                "benchmark_tier": "J2",
                "benchmark_instance_id": instance,
                "jobs": parsed["jobs"],
                "machines": parsed["machines"],
                "operations": parsed["operations"],
                "source_event_type": "machine_breakdown",
            },
            "dynamic_job_shop": {
                "enabled": True,
                "event_source": "realm_j2_source_native_v1",
                "max_dispatch_batch_size": 2,
                "recovery_clearance_ticks": 1,
                "source_observed_events": True,
            },
            "source_event_registry": {
                "machine_breakdown": {
                    "type": "machine_breakdown",
                    "event_class": "safety",
                    "origin": "source_schedule",
                    "actionable": True,
                    "unknown_events_actionable": False,
                }
            },
            "source_event_contract": {
                "source_sidecar_runtime_consumed": True,
                "source_sidecar_converter_consumed": False,
                "sidecar_path": declared,
                "sidecar_sha256": digest,
                "selected_instance_id": instance,
                "source_event_type": "machine_breakdown",
                "runtime_effect_required": True,
            },
            "task_contract": {
                "contract": "logistics.job_shop.realm_j2_recovery.v1",
                "event_response_window": {
                    "first_tick": breakdown["trigger_tick"] + 1,
                    "last_tick": horizon - 1,
                },
                "native_controls": [
                    "dispatch_ready_operations",
                    "repair_machine",
                ],
            },
            "task_requirements": {
                "min_distinct_control_ticks": 3,
                "min_distinct_physical_tools": 2,
                "ordered_tool_milestones": [
                    {
                        "tool": "dispatch_ready_operations",
                        "not_after_tick": breakdown["trigger_tick"],
                    },
                    {
                        "tool": "repair_machine",
                        "not_before_tick": breakdown["trigger_tick"] + 1,
                        "not_after_tick": horizon - 1,
                    },
                    {
                        "tool": "dispatch_ready_operations",
                        "not_before_tick": breakdown["trigger_tick"] + 1,
                        "not_after_tick": horizon - 1,
                    },
                ],
            },
            "dimension_applicability": {
                "optimality_gap": {
                    "applicable": False,
                    "reason": "no_certified_dynamic_j2_optimum",
                },
                "counterfactual_prevention": {
                    "applicable": True,
                    "reason": "deterministic_no_action_replay_over_same_j2_event",
                },
            },
        },
        "horizon_ticks": horizon,
        "tick_minutes": 1,
        "seed": _seed(str(row["candidate_id"])),
        "load_assignments": [],
        "perturbations": [
            {
                "kind": "machine_breakdown",
                "trigger_tick": breakdown["trigger_tick"],
                "duration_ticks": breakdown["duration_ticks"],
                "hidden": False,
                "target": {
                    "machine_id": breakdown["machine_id"],
                    "source_observed": True,
                    "source_instance_id": instance,
                    "source_sidecar_sha256": digest,
                },
                "intensity": 1.0,
                "notes": "Exact source-observed REALM J2 machine breakdown.",
            }
        ],
        "dilemmas": [],
        "difficulty_mode": "deep_planning",
        "difficulty_level": "high",
        "provenance": {
            "data_source": "realm_bench_j2_ccby",
            "files": [declared],
            "commit": REALM_COMMIT,
            "url": "https://github.com/genglongling/REALM-Bench",
            "lock_strategy": (
                "git_commit+file_sha256+selected_row_id+cc-by-runtime-source"
            ),
            "time_window": {"selected_instance_id": instance},
            "license": "CC-BY-4.0",
            "notes": "The selected J2 row is the canonical runtime source.",
        },
        "source_contract": _source_contract(
            [declared],
            {declared: digest},
            derived_window={
                "sha256": selected_digest,
                "recipe_version": "realm-j2-selected-instance-v1",
                "selected_instance_id": instance,
            },
        ),
        "procedural_stress": {"label": "source_native", "source_native": True},
    }
    return body


@lru_cache(maxsize=1)
def _realm_j2_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("instances"), list):
        raise MaterializationBlocked(
            "realm_j2_schema_invalid", "J2 source must contain an instances list"
        )
    return payload


def _materialize_dynasched(row: dict[str, Any]) -> dict[str, Any]:
    metadata = _source_metadata(row)
    bundle = str(metadata.get("bundle") or "").strip()
    if not bundle or Path(bundle).is_absolute() or ".." in Path(bundle).parts:
        raise MaterializationBlocked(
            "dynasched_bundle_identity_invalid", f"invalid source bundle: {bundle!r}"
        )
    root = REPO_ROOT / "works/DynaSchedBench/data" / bundle
    required_names = (
        "input_model.json",
        "events.jsonl",
        "static_jobs.json",
        "static_machines.json",
    )
    optional_names = ("final_metrics.json", "meta.json")
    missing = [name for name in required_names if not (root / name).is_file()]
    if missing:
        raise MaterializationBlocked(
            "dynasched_bundle_crosscheck_assets_missing", ",".join(missing)
        )
    paths = {
        # Preserve the portable source namespace even when a local bundle
        # install exposes ``works/DynaSchedBench`` as a symlink. Resolving the
        # link here leaks the machine-local install root into release YAML.
        name: root / name
        for name in (*required_names, *optional_names)
        if (root / name).is_file()
    }
    declared = {
        name: path.relative_to(REPO_ROOT).as_posix() for name, path in paths.items()
    }
    hashes = {name: _sha256(path) for name, path in paths.items()}
    if str(metadata.get("input_sha256") or "") != hashes["input_model.json"]:
        raise MaterializationBlocked(
            "dynasched_input_hash_mismatch", "ledger/input_model hash mismatch"
        )
    if str(metadata.get("events_sha256") or "") != hashes["events.jsonl"]:
        raise MaterializationBlocked(
            "dynasched_events_hash_mismatch", "ledger/events stream hash mismatch"
        )
    if str(metadata.get("static_jobs_sha256") or "") != hashes["static_jobs.json"]:
        raise MaterializationBlocked(
            "dynasched_static_jobs_hash_mismatch",
            "ledger/static job graph hash mismatch",
        )
    if (
        str(metadata.get("static_machines_sha256") or "")
        != hashes["static_machines.json"]
    ):
        raise MaterializationBlocked(
            "dynasched_static_machines_hash_mismatch",
            "ledger/static machine graph hash mismatch",
        )

    event_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        paths["events.jsonl"].read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MaterializationBlocked(
                "dynasched_event_json_invalid", f"line={line_number}: {exc}"
            ) from exc
        if not isinstance(event, dict) or not str(event.get("event_type") or ""):
            raise MaterializationBlocked(
                "dynasched_event_schema_invalid", f"line={line_number}"
            )
        event_rows.append(event)
    event_count = len(event_rows)
    if event_count != int(metadata.get("event_count") or -1):
        raise MaterializationBlocked(
            "dynasched_event_count_mismatch",
            f"ledger={metadata.get('event_count')} runtime={event_count}",
        )
    event_types = sorted({str(event["event_type"]) for event in event_rows})
    if event_types != sorted(str(value) for value in metadata.get("event_types") or []):
        raise MaterializationBlocked(
            "dynasched_event_type_mismatch", "ledger/runtime event types differ"
        )
    actionable = {str(value) for value in metadata.get("actionable_event_types") or []}
    if not actionable or not actionable.issubset(event_types):
        raise MaterializationBlocked(
            "dynasched_actionable_event_registry_invalid",
            "actionable event types must be a nonempty subset of runtime events",
        )

    jobs_payload = json.loads(paths["static_jobs.json"].read_text(encoding="utf-8"))
    machine_payload = json.loads(
        paths["static_machines.json"].read_text(encoding="utf-8")
    )
    jobs = dict(jobs_payload.get("jobs") or {})
    machines = list(machine_payload.get("machines") or [])
    operations = sum(len(list(job.get("routing") or [])) for job in jobs.values())
    if not jobs or not machines or operations <= 0:
        raise MaterializationBlocked(
            "dynasched_static_inventory_invalid", "empty job/machine operation graph"
        )
    route_lengths = {
        str(job_id): len(list(job.get("routing") or []))
        for job_id, job in jobs.items()
    }
    operation_work_count = operations
    for event in event_rows:
        if str(event.get("event_type") or "") != "ROUTE_CHANGE":
            continue
        job_id = str(event.get("job_id") or "")
        new_routing = event.get("new_routing")
        from_step = event.get("from_step")
        if job_id not in route_lengths or not isinstance(new_routing, list):
            raise MaterializationBlocked(
                "dynasched_route_change_graph_mismatch",
                "route change does not match the static job graph",
            )
        if isinstance(from_step, bool) or not isinstance(from_step, int):
            raise MaterializationBlocked(
                "dynasched_route_change_graph_mismatch",
                "route change lacks an integer from_step",
            )
        new_length = max(0, from_step) + len(new_routing)
        operation_work_count += max(0, new_length - route_lengths[job_id])
        route_lengths[job_id] = new_length
    horizon = event_count + operation_work_count + 2
    if horizon > DYNASCHED_EVENT_DRIVEN_MAX_WORK_TICKS:
        raise MaterializationBlocked(
            "dynasched_event_driven_work_budget_exceeded",
            (
                f"estimated_work_ticks={horizon} exceeds "
                f"max_work_ticks={DYNASCHED_EVENT_DRIVEN_MAX_WORK_TICKS}"
            ),
        )
    declared_replay_contract = metadata.get("event_driven_replay_contract")
    if declared_replay_contract is not None and (
        not isinstance(declared_replay_contract, dict)
        or declared_replay_contract.get("contract")
        != DYNASCHED_EVENT_DRIVEN_REPLAY_CONTRACT
        or int(declared_replay_contract.get("estimated_work_ticks") or -1)
        != horizon
        or int(declared_replay_contract.get("max_work_ticks") or -1)
        != DYNASCHED_EVENT_DRIVEN_MAX_WORK_TICKS
        or declared_replay_contract.get("within_budget") is not True
    ):
        raise MaterializationBlocked(
            "dynasched_event_driven_replay_contract_mismatch",
            "refinement/runtime event-driven replay contracts differ",
        )
    registry = {
        event_type: {
            "type": {
                "ARRIVAL": "job_arrival",
                "DUE_DATE_SET": "due_date_set",
                "BREAKDOWN": "machine_breakdown",
                "REPAIR_COMPLETION": "machine_repair_completion",
                "PTIME_CHANGE": "process_time_change",
                "PRIORITY_CHANGE": "priority_change",
                "ORDER_CANCELLATION": "order_cancellation",
                "PREVENTIVE_MAINTENANCE": "preventive_maintenance",
                "ROUTE_CHANGE": "route_change",
                "DUE_DATE_CHANGE": "due_date_change",
            }.get(event_type, event_type.lower()),
            "event_class": (
                "safety"
                if event_type in {"BREAKDOWN", "PREVENTIVE_MAINTENANCE"}
                else "lifecycle"
                if event_type in {"DUE_DATE_SET", "REPAIR_COMPLETION"}
                else "task"
            ),
            "origin": "source_schedule",
            "actionable": event_type in actionable,
            "unknown_events_actionable": False,
        }
        for event_type in event_types
    }
    first_by_type: dict[str, dict[str, Any]] = {}
    for index, event in enumerate(event_rows):
        event_type = str(event["event_type"])
        if event_type in actionable and event_type not in first_by_type:
            first_by_type[event_type] = {
                "source_event_index": index,
                "source_time": float(event.get("time") or 0.0),
                "source_event_type": event_type,
            }
    license_path = "works/DynaSchedBench/LICENSE"
    license_digest = _sha256(REPO_ROOT / license_path)
    source_assets = {
        "input_model_json": {
            "path": declared["input_model.json"],
            "sha256": hashes["input_model.json"],
            "role": "runtime_input",
        },
        "events_jsonl": {
            "path": declared["events.jsonl"],
            "sha256": hashes["events.jsonl"],
            "role": "runtime_input",
        },
        "static_jobs_json": {
            "path": declared["static_jobs.json"],
            "sha256": hashes["static_jobs.json"],
            "role": "bundle_crosscheck_metadata",
        },
        "static_machines_json": {
            "path": declared["static_machines.json"],
            "sha256": hashes["static_machines.json"],
            "role": "bundle_crosscheck_metadata",
        },
        **{
            name.removesuffix(".json") + "_json": {
                "path": declared[name],
                "sha256": hashes[name],
                "role": "bundle_crosscheck_metadata",
            }
            for name in optional_names
            if name in declared
        },
        "license": {
            "path": license_path,
            "sha256": license_digest,
            "role": "license",
        },
    }
    runtime = [declared["input_model.json"], declared["events.jsonl"]]
    source_contract = _source_contract(
        runtime,
        {
            path: hashes[name]
            for name, path in (
                ("input_model.json", runtime[0]),
                ("events.jsonl", runtime[1]),
            )
        },
        metadata=[
            declared[name]
            for name in ("static_jobs.json", "static_machines.json", *optional_names)
            if name in declared
        ],
        license_files=[license_path],
    )
    return {
        "seed_id": f"dynasched_{_slug(bundle)}",
        "family": "job_shop_dispatch",
        "domain": "logistics",
        "backend_kind": "dynasched_flexible_job_shop",
        "backend_config": {
            "bundle_id": bundle,
            "source_integration_rung": "native_runtime_direct_candidate",
            "release_ready": False,
            "release_reentry_ready": False,
            "runtime_package": "dsbx",
            "runtime_version": DYNASCHED_RUNTIME_VERSION,
            "runtime_code_tree_sha256": DYNASCHED_RUNTIME_CODE_TREE_SHA256,
            "runtime_source_lock": {
                "repository": "https://github.com/dsbx7/DynaSchedBench",
                "commit": DYNASCHED_RUNTIME_COMMIT,
                "python_package_tree_sha256": DYNASCHED_RUNTIME_CODE_TREE_SHA256,
            },
            "direct_data_use": True,
            "upstream_bundle_path": f"data/{bundle}",
            "machine_alternatives_preserved": True,
            "source_assets": source_assets,
            "source_event_counts": dict(
                sorted(
                    Counter(str(event["event_type"]) for event in event_rows).items()
                )
            ),
            "source_event_registry": registry,
            "event_driven_replay_contract": {
                "contract": DYNASCHED_EVENT_DRIVEN_REPLAY_CONTRACT,
                "estimated_work_ticks": horizon,
                "max_work_ticks": DYNASCHED_EVENT_DRIVEN_MAX_WORK_TICKS,
                "advance_semantics": "next_native_boundary_or_source_event",
            },
            "hidden_source_event_types": [],
            "source_denominator_key": (
                f"dynasched:{DYNASCHED_RUNTIME_COMMIT}:data/{bundle}"
            ),
            "task_requirements": {
                "min_distinct_control_ticks": 2,
                "min_distinct_physical_tools": 1,
                "ordered_tool_milestones": [
                    {
                        "tool": "dispatch_flexible_operations",
                        "not_after_tick": horizon - 1,
                    },
                    {
                        "tool": "dispatch_flexible_operations",
                        "not_after_tick": horizon - 1,
                    },
                ],
            },
            "source_event_contract": list(first_by_type.values()),
            "reference": {
                "type": "native_oracle_policy",
                "policy": "native_oracle_earliest_completion_v1",
                "runtime_authority": "dsbx.Sim.Simulator.DynaSchedSim",
                "formal_optimality_applicable": False,
            },
            "dimension_applicability": {
                "optimality_gap": {
                    "applicable": False,
                    "reason": "bundle_has_no_certified_dynamic_fjsp_optimum",
                },
                "counterfactual_prevention": {
                    "applicable": True,
                    "reason": "deterministic_masked_action_replay_over_same_source_event_stream",
                },
            },
        },
        "horizon_ticks": horizon,
        "tick_minutes": 1,
        "seed": _seed(str(row["candidate_id"])),
        "load_assignments": [],
        "perturbations": [],
        "dilemmas": [],
        "difficulty_mode": "deep_planning",
        "difficulty_level": "high",
        "provenance": {
            "data_source": "DynaSchedBench",
            "files": [*runtime, *source_contract["metadata"], license_path],
            "commit": DYNASCHED_RUNTIME_COMMIT,
            "url": "https://github.com/dsbx7/DynaSchedBench",
            "lock_strategy": "git_commit_plus_per_file_sha256",
            "time_window": {
                "source_event_count": event_count,
                "source_event_types": event_types,
            },
            "license": "Apache-2.0",
            "notes": "Official DynaSchedSim consumes the locked model and event stream.",
        },
        "source_contract": source_contract,
        "procedural_stress": {"label": "source_native", "source_native": True},
    }


def _materialize_pglib(row: dict[str, Any]) -> dict[str, Any]:
    raise MaterializationBlocked(
        "source_family_owned_by_power_grid_track",
        "PGLib-UC candidate ownership is exclusive to the power-grid track",
    )


_BUILDERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "jsplib": _materialize_jsplib,
    "routing": _materialize_routing,
    "m5_orgym": _materialize_m5,
    "realm_j2": _materialize_realm,
    "dynasched": _materialize_dynasched,
    "pglib_uc": _materialize_pglib,
}


def _finalize(row: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    source_family = str(row.get("source_family") or "unknown")
    domain = str(body.get("domain") or row.get("domain") or "unknown")
    scenario_id = (
        f"candidate_delta/{_slug(domain)}/{_slug(source_family)}/{_slug(candidate_id)}"
    )
    body["seed_id"] = scenario_id
    body["scenario_id"] = scenario_id
    body["policy_contract"] = {
        "strict_prompt": True,
        "benchmark_side_hints": False,
    }
    config = body.get("backend_config")
    if not isinstance(config, dict):
        raise MaterializationBlocked(
            "backend_config_missing", "seed builder did not emit backend_config"
        )
    config["candidate_materialization"] = {
        "candidate_id": candidate_id,
        "source_family": source_family,
        "active_core": False,
    }
    _complete_dimension_applicability(body)
    perturbations = body.get("perturbations")
    if not isinstance(perturbations, list):
        raise MaterializationBlocked(
            "typed_event_registry_missing", "perturbations must be a list"
        )
    for index, event in enumerate(perturbations):
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("kind"), str)
            or not event["kind"]
            or isinstance(event.get("trigger_tick"), bool)
            or not isinstance(event.get("trigger_tick"), int)
            or event["trigger_tick"] < 0
            or isinstance(event.get("duration_ticks"), bool)
            or not isinstance(event.get("duration_ticks"), int)
            or event["duration_ticks"] <= 0
        ):
            raise MaterializationBlocked(
                "typed_event_invalid", f"invalid perturbation at index {index}"
            )
    contract = body.get("source_contract")
    if not isinstance(contract, dict):
        raise MaterializationBlocked(
            "source_contract_missing", "seed builder adaptation has no source contract"
        )
    required = list(contract.get("runtime_input") or []) + list(
        contract.get("derivation_input") or []
    )
    if not required or set(contract.get("file_sha256s") or {}) != set(required):
        raise MaterializationBlocked(
            "source_contract_incomplete", "source input/hash binding is incomplete"
        )
    body.pop("scenario_signature", None)
    body["scenario_signature"] = recompute_signature_with_seed(body, int(body["seed"]))
    return body


def _blocker(row: dict[str, Any], code: str, detail: str) -> dict[str, Any]:
    return {
        "candidate_id": str(row.get("candidate_id") or ""),
        "source_id": str(row.get("source_id") or ""),
        "source_family": str(row.get("source_family") or ""),
        "source_unit": str(row.get("source_unit") or ""),
        "domain": str(row.get("domain") or ""),
        "materialization_status": "blocked",
        "blocker_code": code,
        "detail": detail,
        "evidence": row.get("evidence")
        if isinstance(row.get("evidence"), dict)
        else {},
    }


def _safe_output_root(output_root: Path) -> Path:
    resolved = output_root.resolve()
    protected = (
        REPO_ROOT / "release/operate_v0_58_0",
        REPO_ROOT / "scenarios/operate_v0_58_0",
    )
    if any(
        resolved == path.resolve() or path.resolve() in resolved.parents
        for path in protected
    ):
        raise ValueError(
            "candidate delta output cannot be inside the active release/Core"
        )
    return resolved


def materialize_candidate_delta(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    execute: bool = False,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build a deterministic plan and optionally write its candidate artifacts."""
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("refinement ledger must contain a rows list")
    selected = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("classification_scope") == _SUPPORTED_SCOPE
        and row.get("final_disposition") in _READABLE_DISPOSITIONS
    ]
    candidate_ids = [str(row.get("candidate_id") or "") for row in selected]
    duplicates = sorted(
        candidate_id
        for candidate_id, count in Counter(candidate_ids).items()
        if not candidate_id or count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate selected candidate_id: {duplicates[0]!r}")

    output_root = _safe_output_root(output_root)
    scenario_bodies: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    blockers: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: str(item["candidate_id"])):
        family = str(row.get("source_family") or "")
        builder = _BUILDERS.get(family)
        if builder is None:
            blockers.append(
                _blocker(
                    row,
                    "source_family_builder_unavailable",
                    f"no exact current-tree seed builder registered for {family!r}",
                )
            )
            continue
        try:
            body = _finalize(row, builder(row))
        except MaterializationBlocked as exc:
            blockers.append(_blocker(row, exc.code, exc.detail))
            continue
        # Candidate builders parse heterogeneous external formats. A malformed
        # or unsupported individual source must be explicit evidence without
        # aborting accounting for every other selected candidate.
        except Exception as exc:  # noqa: BLE001
            blockers.append(
                _blocker(
                    row,
                    "seed_builder_validation_failed",
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        path = output_root / "scenarios" / f"{_slug(str(row['candidate_id']))}.yaml"
        scenario_bodies.append((row, body, path))

    scenario_rows = [
        {
            "candidate_id": str(row["candidate_id"]),
            "source_id": str(row.get("source_id") or ""),
            "source_family": str(row.get("source_family") or ""),
            "source_unit": str(row.get("source_unit") or ""),
            "scenario_id": str(body["scenario_id"]),
            "scenario_signature": str(body["scenario_signature"]),
            "path": str(path),
            "domain": str(body["domain"]),
            "family": str(body["family"]),
            "backend_kind": str(body["backend_kind"]),
            "difficulty_mode": str(body["difficulty_mode"]),
            "difficulty_level": str(body["difficulty_level"]),
            "materialization_status": "materialized_candidate",
        }
        for row, body, path in scenario_bodies
    ]
    selected_count = len(selected)
    accounted = len(scenario_rows) + len(blockers)
    if accounted != selected_count:
        raise RuntimeError(
            f"selected candidate accounting failure: {accounted} != {selected_count}"
        )
    report = {
        "schema_version": "operate-logistics-candidate-delta-v1",
        "status": "candidate_only_requires_protocol21_admission",
        "input_bindings": {
            "refinement_ledger": {
                "path": str(ledger_path.resolve()),
                "sha256": _sha256(ledger_path),
            }
        },
        "selection": {
            "classification_scope": _SUPPORTED_SCOPE,
            "final_disposition": _SUPPORTED_DISPOSITION,
        },
        "summary": {
            "n_selected_ready_for_full_admission": selected_count,
            "n_materialized": len(scenario_rows),
            "n_blocked": len(blockers),
            "all_selected_accounted": accounted == selected_count,
            "active_core_modified": False,
        },
        "materialized_by_source_family": dict(
            sorted(Counter(row["source_family"] for row in scenario_rows).items())
        ),
        "blocked_by_code": dict(
            sorted(Counter(row["blocker_code"] for row in blockers).items())
        ),
        "scenarios": scenario_rows,
        "blockers": blockers,
    }
    portable_report = canonicalize_repo_owned_paths(report, repo_root=repo_root)
    if not execute:
        return portable_report
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite candidate delta: {output_root}")
    output_root.mkdir(parents=True)
    for _, body, path in scenario_bodies:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(body, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    report_path = output_root / "materialization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        suite = canonicalize_repo_owned_paths(
            build_suite(report_path), repo_root=repo_root
        )
    finally:
        report_path.write_text(
            json.dumps(portable_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output_root / "source_suite.json").write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return portable_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    report = materialize_candidate_delta(
        ledger_path=args.ledger.resolve(),
        output_root=args.output_root.resolve(),
        execute=args.execute,
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
