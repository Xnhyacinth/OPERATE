"""Registry-bound source-consumption extractors.

These functions are deliberately backend-specific entry points.  They run
only after a successful backend reset and emit the narrow facts consumed by
the protocol-2.1 source gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from core.source_asset_contract import (
    is_virtual_source_reference,
    resolve_source_asset_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

_DERIVED_WINDOW_KEYS: dict[str, tuple[str, ...]] = {
    "alibaba_trace_sim": ("jobs", "source_transform"),
    "jsplib_job_shop": ("job_shop", "instance_name"),
    "orgym_invmgmt": (
        "demand_profile_id",
        "m5_demand_handle",
        "m5_start_day",
        "m5_end_day",
        "orgym_env_config",
    ),
    "pyvrp_cvrp": ("network", "instance_name"),
    "pyvrp_vrptw": ("network", "instance_name"),
    "pymgrid_economic_dispatch": ("profiles", "site"),
}


def _window_payload(backend_kind: str, config: dict[str, Any]) -> dict[str, Any]:
    keys = _DERIVED_WINDOW_KEYS.get(backend_kind, ())
    return {key: config.get(key) for key in keys}


def derive_window_sha256_for_scenario(scenario: dict[str, Any]) -> str | None:
    backend_kind = str(scenario.get("backend_kind") or "")
    if backend_kind == "pandapower_lv":
        return str(
            (
                (scenario.get("backend_config") or {}).get("derivation_recipe")
                or {}
            ).get("source_window_sha256")
            or ""
        ) or None
    if backend_kind not in _DERIVED_WINDOW_KEYS:
        return None
    payload = _window_payload(
        backend_kind, dict(scenario.get("backend_config") or {})
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _contract_evidence(
    *,
    env: Any,
    scenario: dict[str, Any],
    proof_kind: str,
    channels: tuple[str, ...],
    state_fields: tuple[str, ...],
    dynamic: bool,
    actual_runtime_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    contract = resolve_source_asset_contract(scenario, repo_root=REPO_ROOT)
    virtual_missing = bool(contract.missing_required_files) and all(
        is_virtual_source_reference(path)
        for path in contract.missing_required_files
    )
    if contract.contract_errors or (
        contract.missing_required_files and not virtual_missing
    ):
        return {
            "status": "held",
            "proof_kind": proof_kind,
            "blockers": sorted(
                {
                    *contract.contract_errors,
                    *(
                        ("required_source_file_missing",)
                        if contract.missing_required_files
                        else ()
                    ),
                }
            ),
        }
    backend = getattr(env, "_backend", None)
    if backend is None:
        return {
            "status": "held",
            "proof_kind": proof_kind,
            "blockers": ["backend_not_reset"],
        }
    direct = proof_kind in {"direct_runtime_files", "native_include_graph"}
    actual_resolved = {
        str(path.resolve()) for path in actual_runtime_paths if path.is_file()
    }
    consumed_hashes = {
        raw: digest
        for raw, digest in contract.locked_source_hashes.items()
        if contract.resolved_source_paths.get(raw) in actual_resolved
    }
    opened_hashes = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in actual_runtime_paths
        if path.is_file()
    }
    native_trace = getattr(backend, "protocol21_source_trace", None)
    if callable(native_trace):
        evidence = native_trace()
        if isinstance(evidence, dict):
            native_opened_paths = list(
                evidence.get("opened_source_paths") or []
            )
            native_opened_hashes = dict(
                evidence.get("opened_source_sha256") or {}
            )
            return {
                **evidence,
                "proof_kind": evidence.get("proof_kind") or proof_kind,
                "opened_source_paths": sorted(
                    set(native_opened_paths).union(opened_hashes)
                ),
                "opened_source_sha256": {
                    **native_opened_hashes,
                    **opened_hashes,
                },
                "runtime_trace_observed": True,
                "evidence_from_scenario_config_only": False,
            }
    # Fail closed. Open-file hashes are useful diagnostics, but do not prove
    # that a parser output affected backend state.
    return {
        "status": "held",
        "proof_kind": proof_kind,
        "consumed_source_hashes": consumed_hashes if direct else {},
        "lineage_source_hashes": {},
        "consumed_window_sha256": None,
        "recipe_version": contract.recipe_version,
        "consumed_channels": [],
        "derived_backend_state_fields": [],
        "consumption_ticks": [],
        "state_effect_observed": False,
        "opened_source_paths": sorted(opened_hashes),
        "opened_source_sha256": opened_hashes,
        "runtime_trace_observed": False,
        "evidence_from_scenario_config_only": False,
        "blockers": ["backend_runtime_source_trace_unimplemented"],
    }
    return evidence


def alibaba_trace_sim(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="derived_source_window",
        channels=("trace_jobs", "future_arrivals"),
        state_fields=("queued_jobs", "running_jobs", "available_gpu_units"),
        dynamic=True,
    )


def alibaba_openb_gpu_placement(
    *, env: Any, scenario: dict[str, Any]
) -> dict[str, Any]:
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="direct_runtime_files",
        channels=("openb_node_inventory", "openb_pod_trace"),
        state_fields=(
            "pod_assignments",
            "node_resource_allocation",
            "placement_fragmentation",
            "qos_delay_risk",
        ),
        dynamic=True,
    )


def jsplib_job_shop(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="direct_runtime_files",
        channels=(
            "job_precedence",
            "operation_duration",
            "operation_machine",
        ),
        state_fields=(
            "ready_operations",
            "machine_available_at",
            "job_next_operation",
            "unfinished_operations",
            "current_makespan",
        ),
        dynamic=False,
    )


def orgym_invmgmt(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="derived_source_window",
        channels=("calendar_day", "demand_units", "sell_price"),
        state_fields=(
            "on_hand_inventory",
            "inventory_position",
            "pipeline_orders",
            "realized_demand",
            "lost_sales",
        ),
        dynamic=True,
    )


def pyvrp_cvrp(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="direct_runtime_files",
        channels=(),
        state_fields=(),
        dynamic=False,
    )


def pyvrp_vrptw(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="direct_runtime_files",
        channels=(),
        state_fields=(),
        dynamic=False,
    )


def pymgrid_economic_dispatch(
    *, env: Any, scenario: dict[str, Any]
) -> dict[str, Any]:
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="derived_source_window",
        channels=("load_profile", "renewable_profile", "tariff_profile"),
        state_fields=("battery_soc", "grid_exchange_mw", "unmet_load_mw"),
        dynamic=True,
    )


def pandapower_lv(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="derived_source_window",
        channels=("load_profile", "generation_profile"),
        state_fields=("bus_voltage_pu", "line_loading_percent", "der_dispatch"),
        dynamic=True,
    )


def cigre_distribution(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    """Require a runtime trace for pandapower-generated CIGRE feeders.

    The network is created by a locked upstream constructor, so provenance
    alone is not source-consumption evidence.  Until the native backend emits
    an opened/consumed trace for that constructor, this adapter deliberately
    returns a held result rather than upgrading a URI declaration to proof.
    """
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="derived_source_window",
        channels=("pandapower_network_constructor", "load_profile"),
        state_fields=(
            "bus_voltage_pu",
            "line_loading_percent",
            "der_dispatch",
        ),
        dynamic=True,
    )


def pandapower_acopf(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    from domains.power_grid.source_paths import resolve_source_ref

    case_file = str(
        (scenario.get("backend_config") or {}).get("case_file") or ""
    )
    paths = (
        (resolve_source_ref(case_file, description="PGLib-OPF case"),)
        if case_file
        else ()
    )
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="direct_runtime_files",
        channels=("pglib_opf_case",),
        state_fields=("bus_voltage_pu", "line_loading_percent", "generator_dispatch_mw"),
        dynamic=False,
        actual_runtime_paths=paths,
    )


def pglib_uc_synthetic(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    from domains.power_grid.source_paths import resolve_source_ref

    case_file = str(
        (scenario.get("backend_config") or {}).get("case_file") or ""
    )
    paths = (
        (resolve_source_ref(case_file, description="pglib-uc case"),)
        if case_file
        else ()
    )
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="direct_runtime_files",
        channels=("pglib_uc_case", "demand_profile"),
        state_fields=("generator_dispatch_mw", "demand_mw", "reserve_shortfall_mw"),
        dynamic=True,
        actual_runtime_paths=paths,
    )


_DSS_INCLUDE_RE = re.compile(
    r"(?i)(?:^|\s)(?:redirect|compile)\s+[\"']?([^\"'!\\s]+)"
    r"|file\\s*=\\s*[\"']?([^\"')\\s]+)"
)


def _opendss_include_graph(master: Path) -> tuple[Path, ...]:
    queued = [master.resolve()]
    visited: set[Path] = set()
    while queued:
        path = queued.pop()
        if path in visited or not path.is_file():
            continue
        visited.add(path)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in _DSS_INCLUDE_RE.finditer(text):
            raw = next(
                (value for value in match.groups() if value),
                "",
            )
            if not raw:
                continue
            candidate = (path.parent / raw).resolve()
            if candidate.is_file() and candidate not in visited:
                queued.append(candidate)
    return tuple(sorted(visited))


def opendss_fresh_feeders(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    backend = getattr(env, "_backend", None)
    master = getattr(backend, "master_file", None)
    paths = _opendss_include_graph(Path(master)) if master else ()
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="native_include_graph",
        channels=("master_dss", "redirected_dss_assets"),
        state_fields=("bus_voltage_pu", "line_loading_percent", "tap_positions"),
        dynamic=False,
        actual_runtime_paths=paths,
    )


def opendss_ieee13(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    return opendss_fresh_feeders(env=env, scenario=scenario)


def sumo(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    backend = getattr(env, "_backend", None)
    sidecar = getattr(backend, "_sidecar", None)
    if sidecar is None:
        return {
            "status": "held",
            "proof_kind": "direct_runtime_files",
            "blockers": ["live_sumo_runtime_not_verified"],
        }
    contract = getattr(backend, "_runtime_control_contract", None) or {}
    source_assets = contract.get("source_assets") or {}
    asset_rows: list[dict[str, Any]] = []
    for key in ("sumocfg", "network"):
        if isinstance(source_assets.get(key), dict):
            asset_rows.append(source_assets[key])
    for key in ("route_files", "additional_files", "recursive_inputs"):
        asset_rows.extend(
            row
            for row in source_assets.get(key) or []
            if isinstance(row, dict)
        )
    paths = tuple(Path(str(row["path"])) for row in asset_rows if row.get("path"))
    return _contract_evidence(
        env=env,
        scenario=scenario,
        proof_kind="direct_runtime_files",
        channels=(
            "sumo_network",
            "sumo_routes",
            "traffic_light_programs",
            "controlled_lane_metrics",
        ),
        state_fields=(
            "corridor_queues",
            "signal_programs",
            "travel_time_cost",
            "native_throughput",
        ),
        dynamic=True,
        actual_runtime_paths=paths,
    )


def mock_sumo(*, env: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    del env, scenario
    return {
        "status": "failed",
        "proof_kind": None,
        "declared_source_unused": True,
        "blockers": ["backend_formal_fidelity_not_allowed"],
    }
