#!/usr/bin/env python3
"""Build the Track-C external benchmark conversion ledger.

This report is intentionally a source-first, non-release artifact.  External
benchmarks may contribute an interaction/evaluation method, but an external
raw QA corpus or generated instance can never become a Protocol-2.1 row.  A
candidate is ready for the ordinary Protocol-2.1 pipeline only when a
separately locked native source has evidence for every required gate.

The inventory is computed from the current checkout rather than copied from an
older source-lock report.  This matters for ignored ``works/`` checkouts: a
dirty tree, missing file, package mismatch, or stale sidecar is recorded as a
hold instead of being silently treated as source consumption.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.audit_citylearn_sources import (
    _auxiliary_file_paths,
    _required_file_reports,
    _source_file_paths,
    _source_lock_sidecar_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_C_DIR = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "protocol21_expansion_trials"
    / "track_c_external_conversion_v1"
)
DEFAULT_OUTPUT = TRACK_C_DIR / "conversion_catalog.json"
REVIEW_DATE = "2026-08-06"

REQUIRED_GATES = (
    "source_lock",
    "runtime_consumption",
    "native_state_effect",
    "temporal_evolution",
    "task_contract",
    "response_window",
    "deterministic_counterfactual_replay",
    "difficulty_depth",
    "effective_source_independence",
)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _relative(path: Path, repo_root: Path = REPO_ROOT) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _git(root: Path, *args: str) -> str | None:
    if not root.is_dir():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _package(package: str, module: str | None = None) -> dict[str, Any]:
    module = module or package
    try:
        importable = importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        importable = False
    version: str | None = None
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        version = importlib.metadata.version(package)
    return {
        "distribution": package,
        "module": module,
        "importable": importable,
        "version": version,
    }


def _asset(path: str, *, role: str = "native_source_asset") -> dict[str, Any]:
    candidate = REPO_ROOT / path
    return {
        "path": path,
        "exists": candidate.exists(),
        "size_bytes": candidate.stat().st_size if candidate.is_file() else None,
        "sha256": _sha256(candidate),
        "role": role,
    }


def _evidence(path: str, note: str) -> dict[str, Any]:
    candidate = REPO_ROOT / path
    return {
        "path": path,
        "exists": candidate.is_file(),
        "sha256": _sha256(candidate),
        "note": note,
    }


def _gates(**values: bool) -> dict[str, bool]:
    return {gate: bool(values.get(gate, False)) for gate in REQUIRED_GATES}


def _gate_status(gates: dict[str, bool]) -> tuple[str, list[str]]:
    missing = [gate for gate in REQUIRED_GATES if not gates.get(gate, False)]
    return ("ready_for_full_protocol21" if not missing else "held", missing)


def _external_source(
    *,
    source_id: str,
    title: str,
    urls: list[str],
    license_value: str,
    license_status: str,
    version_value: str | None,
    version_status: str,
    raw_asset_consumed: bool = False,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "title": title,
        "urls": urls,
        "url_evidence": "public_reference" if urls else "missing",
        "license": {
            "value": license_value,
            "status": license_status,
        },
        "version": {
            "value": version_value,
            "status": version_status,
        },
        "raw_asset_consumed": raw_asset_consumed,
        "raw_asset_admission": "forbidden",
    }


def _recipe_base(
    *,
    source: dict[str, Any],
    target_domain: str,
    target_backend: str,
    native_tools: list[str],
    local_assets: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    gates: dict[str, bool],
    disposition: str,
    transfer_method: str,
    blockers: list[str],
    notes: str,
) -> dict[str, Any]:
    status, missing = _gate_status(gates)
    all_blockers = sorted(set(blockers) | {f"{gate}_unproven" for gate in missing})
    return {
        "source_id": source["source_id"],
        "title": source["title"],
        "external_source": source,
        "target": {
            "domain": target_domain,
            "backend": target_backend,
            "native_tools": native_tools,
        },
        "native_source_assets": local_assets,
        "evidence_refs": evidence,
        "protocol21_gates": gates,
        "required_protocol21_gates": list(REQUIRED_GATES),
        "transfer_method": transfer_method,
        "disposition": disposition,
        "status": status,
        "ready_for_full_protocol21": status == "ready_for_full_protocol21",
        "direct_core_admission": False,
        "blocker_codes": all_blockers,
        "notes": notes,
    }


def _dyna_recipe(repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = _external_source(
        source_id="dynaschedbench",
        title="DynaSchedBench",
        urls=["https://arxiv.org/abs/2605.27566"],
        license_value="not_applicable_to_method_transfer; raw asset not consumed",
        license_status="method_only_no_external_asset_claim",
        version_value="arXiv:2605.27566",
        version_status="paper_identifier_evidenced",
    )
    native_path = "works/JSPLIB-Instances/instances/la09"
    native_source = repo_root / native_path
    expected_hash = "0c1730baa9e9480efff7d062d5af5758f056a5889326d877bc7d5b43d71a2cc4"
    probe = _read_json(
        repo_root
        / "release/dt_sched_bench_v0_52_0_candidate/protocol21_expansion_trials/external_dyna_native_backend_probe_v1.json"
    )
    queue = _read_json(
        repo_root
        / "release/dt_sched_bench_v0_52_0_candidate/protocol21_expansion_trials/external_native_pilot_queue_v1.json"
    )
    row = next(
        (
            item
            for item in queue.get("candidates", [])
            if isinstance(item, dict)
            and item.get("candidate_id") == "dynaschedbench__jsplib__high"
        ),
        {},
    )
    queue_gates = ((row.get("pilot_gate_evidence") or {}).get("gates") or {})
    gates = _gates(
        source_lock=(
            native_source.is_file()
            and _sha256(native_source) == expected_hash
            and _git(repo_root / "works/JSPLIB-Instances", "rev-parse", "HEAD")
            == "eea2b60dd7e2f5c907ff7302662c61812eb7efdf"
        ),
        runtime_consumption=bool(queue_gates.get("native_runtime")),
        native_state_effect=bool(queue_gates.get("state_changing_control")),
        temporal_evolution=bool(
            probe.get("runtime", {}).get("runtime_trace_observed")
            and probe.get("response_window", {}).get("repair_in_window")
        ),
        task_contract=bool(queue_gates.get("response_window")),
        response_window=bool(queue_gates.get("response_window")),
        deterministic_counterfactual_replay=bool(
            queue_gates.get("deterministic_counterfactual_replay")
        ),
        difficulty_depth=bool(queue_gates.get("difficulty_depth")),
        effective_source_independence=bool(
            queue_gates.get("effective_source_independence")
        ),
    )
    recipe = _recipe_base(
        source=source,
        target_domain="logistics",
        target_backend="jsplib_job_shop",
        native_tools=["dispatch_job_operation", "resequence_job_operations"],
        local_assets=[
            _asset(native_path),
            _asset(
                "works/JSPLIB-Instances/CHECKSUMS.txt",
                role="native_source_lock_manifest",
            ),
            _asset("works/JSPLIB-Instances/README.md", role="native_source_metadata"),
        ],
        evidence=[
            _evidence(
                "release/dt_sched_bench_v0_52_0_candidate/protocol21_expansion_trials/external_dyna_native_backend_probe_v1.json",
                "direct JSPLIB runtime trace, state-changing dispatch/repair, and response window",
            ),
            _evidence(
                "release/dt_sched_bench_v0_52_0_candidate/protocol21_expansion_trials/external_native_pilot_queue_v1.json",
                "all ordinary native pilot gates and candidate identity",
            ),
            _evidence(
                "release/dt_sched_bench_v0_52_0_candidate/protocol21_unified_five_domain_v5_71_dyna/source_consumption_protocol2_v21.json",
                "source consumption and deterministic replay evidence",
            ),
            _evidence(
                "release/dt_sched_bench_v0_52_0_candidate/protocol21_unified_five_domain_v5_71_dyna/source_grounded_protocol2_v21.json",
                "source lock, headroom, independence, and difficulty evidence",
            ),
            _evidence(
                "release/dt_sched_bench_v0_52_0_candidate/protocol21_unified_five_domain_v5_71_dyna/task_contracts_protocol2_v21.json",
                "native task contract and terminal response window",
            ),
            _evidence(
                "release/dt_sched_bench_v0_52_0_candidate/protocol21_unified_five_domain_v5_71_dyna/strategy_depth_protocol2_v21.json",
                "observed lower-bound strategy depth",
            ),
        ],
        gates=gates,
        disposition="admitted_method_transfer",
        transfer_method=(
            "Apply DynaSchedBench event calibration and observability axes to "
            "the locked JSPLIB la09 instance; no DynaSchedBench raw/generated "
            "asset is consumed."
        ),
        blockers=[],
        notes=(
            "The native source, not the external paper or generated DFJSP data, "
            "is the denominator. Ready means eligible to enter the ordinary "
            "Protocol-2.1 pipeline only; it is not Core admission."
        ),
    )
    scenario_id = "logistics/job_shop_dispatch/time_pressure/high/jobshop_la09_dynamic_recovery_high_s44"
    candidate = {
        "candidate_id": scenario_id,
        "external_source_id": "dynaschedbench",
        "transfer_mode": "native_method_transfer",
        "native_source": {
            "path": native_path,
            "sha256": _sha256(native_source),
            "expected_sha256": expected_hash,
            "repository_url": "https://github.com/tamy0612/JSPLIB",
            "git_commit": _git(repo_root / "works/JSPLIB-Instances", "rev-parse", "HEAD"),
            "raw_external_asset_consumed": False,
        },
        "native_backend": {
            "domain": "logistics",
            "backend": "jsplib_job_shop",
            "tools": ["dispatch_job_operation", "repair_machine"],
            "runtime_probe": "external_dyna_native_backend_probe_v1.json",
        },
        "scenario_id": scenario_id,
        "source_denominator_key": "jsplib_job_shop:la09:method:dynaschedbench",
        "structural_fingerprint": row.get("structural_fingerprint"),
        "protocol21_gates": gates,
        "ready_for_full_protocol21": all(gates.values()),
        "direct_core_admission": False,
        "status": "ready_for_full_protocol21" if all(gates.values()) else "held",
        "blocker_codes": [] if all(gates.values()) else [
            f"{gate}_unproven" for gate, passed in gates.items() if not passed
        ],
        "evidence_refs": recipe["evidence_refs"],
        "observed": {
            "breakdown_trigger_tick": probe.get("response_window", {}).get(
                "breakdown_trigger_tick"
            ),
            "repair_in_window": probe.get("response_window", {}).get(
                "repair_in_window"
            ),
            "consumed_source_hash": probe.get("runtime", {}).get(
                "consumed_source_hash"
            ),
            "operations_scheduled": (
                (row.get("pilot_gate_evidence") or {}).get("observed") or {}
            ).get("operations_scheduled"),
            "depth_lower_bound": (
                (row.get("pilot_gate_evidence") or {}).get("observed") or {}
            ).get("depth_lower_bound"),
        },
    }
    return recipe, [candidate]


def _cityflow_recipe(repo_root: Path) -> dict[str, Any]:
    root = repo_root / "works/CityFlow"
    source = _external_source(
        source_id="cityflow_official",
        title="CityFlow",
        urls=[
            "https://github.com/cityflow-project/CityFlow",
            "https://cityflow.readthedocs.io/en/latest/",
        ],
        license_value="Apache-2.0 code; roadnet/flow asset terms require lock",
        license_status="code_license_file_present_asset_terms_pending",
        version_value=_git(root, "rev-parse", "HEAD"),
        version_status="local_git_commit_observed" if root.is_dir() else "missing",
    )
    probe_path = "release/dt_sched_bench_v0_52_0_candidate/cityflow_runtime_probe.json"
    probe = _read_json(repo_root / probe_path)
    runtime = probe.get("runtime") or {}
    p = probe.get("probe") or {}
    adapter_exists = (repo_root / "domains/traffic/backends/cityflow_backend.py").is_file()
    gates = _gates(
        source_lock=(
            root.is_dir()
            and bool(_git(root, "rev-parse", "HEAD"))
            and _git(root, "status", "--porcelain") == ""
            and all(
                (_sha256(root / name) is not None)
                for name in ("LICENSE.txt", "examples/config.json", "examples/flow.json", "examples/roadnet.json")
            )
        ),
        runtime_consumption=bool(
            probe.get("status") == "runtime_deterministic_control_headroom_verified"
            and runtime.get("module_path")
        ),
        native_state_effect=bool(p.get("state_changing_control_observed")),
        temporal_evolution=int(p.get("steps", 0) or 0) > 1,
        task_contract=False,
        response_window=False,
        deterministic_counterfactual_replay=bool(p.get("same_seed_replay_equal")),
        difficulty_depth=False,
        effective_source_independence=False,
    )
    blockers = ["cityflow_native_adapter_missing"] if not adapter_exists else []
    blockers.extend(["cityflow_worktree_dirty"] if _git(root, "status", "--porcelain") else [])
    return _recipe_base(
        source=source,
        target_domain="traffic",
        target_backend="cityflow",
        native_tools=["query_signal_control", "set_signal_phase_duration"],
        local_assets=[
            _asset("works/CityFlow/LICENSE.txt", role="source_license"),
            _asset("works/CityFlow/examples/config.json"),
            _asset("works/CityFlow/examples/flow.json"),
            _asset("works/CityFlow/examples/roadnet.json"),
        ],
        evidence=[
            _evidence(probe_path, "temporary live CityFlow control/headroom probe"),
            _evidence(
                "release/dt_sched_bench_v0_52_0_candidate/external_source_locks.json",
                "historical source-lock inventory; current worktree state is rechecked above",
            ),
        ],
        gates=gates,
        disposition="pilot_only",
        transfer_method="Transfer CityFlow signal-control state/action semantics to a native traffic adapter; do not relabel CityFlow as SUMO.",
        blockers=blockers,
        notes="The temporary runtime probe proves a state change and replay only. It does not prove a tool-protocol adapter, task contract, response window, depth, or independent candidate.",
    )


def _simbench_recipe(repo_root: Path) -> dict[str, Any]:
    source = _external_source(
        source_id="simbench_official",
        title="SimBench",
        urls=[
            "https://github.com/e2nIEE/simbench",
            "https://simbench.readthedocs.io/en/stable/",
        ],
        license_value="BSD-3-Clause code; ODbL/DbCL data terms",
        license_status="source_lock_artifact_declared",
        version_value="simbench==1.6.2; v1.6.2@615135cbc04f4576bba6edad8528c1aa7e0a0b10",
        version_status="package_and_commit_observed",
    )
    package = _package("simbench")
    calibration_path = "release/dt_sched_bench_v0_52_0_candidate/simbench_candidate_calibration.json"
    task_path = "release/dt_sched_bench_v0_52_0_candidate/simbench_task_contracts_protocol1_4.json"
    calibration = _read_json(repo_root / calibration_path)
    task = _read_json(repo_root / task_path)
    rows = [row for row in calibration.get("candidates", []) if isinstance(row, dict)]
    passed = [row for row in rows if row.get("status") == "passed"]
    one = passed[0] if passed else {}
    gates = _gates(
        source_lock=bool(
            package.get("version") == "1.6.2"
            and (calibration.get("source_lock") or {}).get("commit")
            == "615135cbc04f4576bba6edad8528c1aa7e0a0b10"
        ),
        runtime_consumption=bool(one.get("source_profile_consumed")),
        native_state_effect=bool((one.get("checks") or {}).get("native_state_changing_leverage")),
        temporal_evolution=bool(one.get("episodes", {}).get("oracle_offline", {}).get("n_ticks", 0) > 1),
        task_contract=bool(task.get("n_passed", 0) > 0),
        response_window=False,
        deterministic_counterfactual_replay=bool(one.get("deterministic_wait_replay")),
        difficulty_depth=False,
        effective_source_independence=False,
    )
    return _recipe_base(
        source=source,
        target_domain="power_grid",
        target_backend="pandapower_simbench",
        native_tools=["set_topology", "redispatch_generation", "set_storage_power"],
        local_assets=[
            _asset(calibration_path, role="native_behavioral_calibration"),
            _asset(task_path, role="native_task_contract_report"),
            _asset("requirements-backends.txt", role="package_dependency_lock"),
        ],
        evidence=[
            _evidence(calibration_path, "12 real-profile windows, one native-leverage pass"),
            _evidence(task_path, "task contract failed: n_passed=0"),
            _evidence(
                "release/dt_sched_bench_v0_52_0_candidate/backend_conversion_status.json",
                "existing five-backend conversion ledger",
            ),
        ],
        gates=gates,
        disposition="pilot_only",
        transfer_method="Use SimBench annual profiles and topology through the native AC feeder adapter; keep profile windows and network codes as the effective source axes.",
        blockers=["simbench_task_contract_failed", "simbench_depth_and_independence_pending"],
        notes="Profile consumption and deterministic wait replay are evidenced, but the sole task-contract probe has insufficient task-loss mitigation and no candidate is ready for full Protocol-2.1.",
    )


def _citylearn_recipe(repo_root: Path) -> dict[str, Any]:
    root = repo_root / "works/CityLearn"
    dataset = root / "data/datasets/citylearn_challenge_2022_phase_3"
    source = _external_source(
        source_id="citylearn_official",
        title="CityLearn",
        urls=[
            "https://github.com/intelligent-environments-lab/CityLearn",
            "https://www.citylearn.net/",
        ],
        license_value="MIT code; bundled dataset/timeseries terms require lock",
        license_status="code_license_present_dataset_terms_pending",
        version_value=_git(root, "rev-parse", "HEAD"),
        version_status="local_git_commit_observed" if root.is_dir() else "missing",
    )
    required_reports = _required_file_reports(dataset)
    selected_lock = _source_lock_sidecar_report(dataset, required_reports)
    runtime_package = _package("citylearn", "citylearn.citylearn")
    source_files = {**_source_file_paths(dataset), **_auxiliary_file_paths(dataset)}
    asset_paths = [
        _relative(path)
        for path in source_files.values()
        if path.is_file()
    ]
    asset_hashes_match = all(
        row.get("exists") is True and row.get("matches_current_file") is True
        for row in required_reports
    )
    source_lock_passed = bool(
        root.is_dir()
        and _git(root, "status", "--porcelain") in (None, "")
        and selected_lock.get("closed") is True
        and asset_hashes_match
    )
    gates = _gates(
        source_lock=source_lock_passed,
        runtime_consumption=False,
        native_state_effect=False,
        temporal_evolution=False,
        task_contract=False,
        response_window=False,
        deterministic_counterfactual_replay=False,
        difficulty_depth=False,
        effective_source_independence=False,
    )
    return _recipe_base(
        source=source,
        target_domain="building_energy",
        target_backend="citylearn",
        native_tools=[
            "set_battery_charge_rate",
            "set_dhw_storage_charge_rate",
            "shift_flexible_building_load",
            "set_cooling_or_heating_setpoint",
        ],
        local_assets=[
            _asset("works/CityLearn/LICENSE", role="source_license"),
            *[_asset(path) for path in asset_paths],
            _asset(
                "release/dt_sched_bench_v0_52_0_candidate/citylearn_source_lock.json",
                role="declared_source_lock_sidecar",
            ),
        ],
        evidence=[
            _evidence(
                "release/dt_sched_bench_v0_52_0_candidate/citylearn_source_lock.json",
                "declared lock is revalidated against schema, Building_*.csv, and runtime sizing assets",
            ),
            _evidence("reports/citylearn_source_preflight_latest.json", "preflight blocker report"),
            _evidence(
                "release/dt_sched_bench_v0_52_0_candidate/external_source_locks.json",
                "local checkout and asset inventory",
            ),
            _evidence(
                "domains/building_energy/adapter.py",
                "pilot-only native adapter with simulator-owned clock",
            ),
            _evidence(
                "tests/test_citylearn_building_energy_adapter.py",
                "native state-effect, deterministic replay, and source-lock regression tests",
            ),
        ],
        gates=gates,
        disposition="pilot_only",
        transfer_method="Use the pilot building-energy-native adapter around the locked CityLearn schema/timeseries; do not map HVAC/occupant tasks into power-grid entities or admit this source before release gates.",
        blockers=[
            *([] if runtime_package["importable"] else ["citylearn_runtime_module_not_importable"]),
            "citylearn_release_backend_formalization_pending",
            "citylearn_masked_action_replay_not_implemented",
            "citylearn_scorer_evidence_not_wired",
            "citylearn_protocol21_full_gates_not_run",
        ],
        notes=(
            "CityLearn schema, Building_*.csv timeseries, and PV/battery sizing "
            "assets are source-locked. A native pilot adapter now proves storage "
            "state effects and deterministic simulator-owned time, but scorer "
            "wiring, masked replay, task/depth gates, and release materialization "
            "remain incomplete."
        ),
    )


def _flatland_recipe(repo_root: Path) -> dict[str, Any]:
    root = repo_root / "works/flatland-rl"
    source = _external_source(
        source_id="flatland_official",
        title="Flatland",
        urls=[
            "https://github.com/flatland-association/flatland-rl",
            "https://flatland-association.github.io/flatland-book/",
        ],
        license_value="MIT",
        license_status="license_file_present",
        version_value=_git(root, "rev-parse", "HEAD"),
        version_status="local_git_commit_observed" if root.is_dir() else "missing",
    )
    package = _package("flatland")
    clean = _git(root, "status", "--porcelain") == "" if root.is_dir() else False
    assets = sorted((root / "env_data").glob("**/*.pkl")) if root.is_dir() else []
    gates = _gates(
        source_lock=bool(root.is_dir() and clean and (root / "LICENSE").is_file() and assets),
        runtime_consumption=False,
        native_state_effect=False,
        temporal_evolution=False,
        task_contract=False,
        response_window=False,
        deterministic_counterfactual_replay=False,
        difficulty_depth=False,
        effective_source_independence=False,
    )
    return _recipe_base(
        source=source,
        target_domain="rail_transport",
        target_backend="flatland",
        native_tools=[
            "set_train_route_choice",
            "hold_train_at_signal",
            "reroute_train_after_malfunction",
        ],
        local_assets=[
            _asset("works/flatland-rl/LICENSE", role="source_license"),
            _asset("works/flatland-rl/env_data/railway/example_flatland_000.pkl"),
            _asset("works/flatland-rl/env_data/railway/example_flatland_001.pkl"),
        ],
        evidence=[
            _evidence(
                "release/dt_sched_bench_v0_52_0_candidate/external_source_locks.json",
                "local Flatland commit/license/asset inventory",
            ),
        ],
        gates=gates,
        disposition="pilot_only",
        transfer_method="Add a rail-native domain adapter; never remap trains, blocks, or switches into road Traffic.",
        blockers=[
            "flatland_runtime_not_importable",
            "flatland_native_adapter_missing",
            "rail_task_contract_unproven",
        ],
        notes=f"Flatland source files are locally locked, but the current Python runtime reports importable={package['importable']} and no rail adapter/evidence path exists.",
    )


def _method_only_recipe(
    *,
    repo_root: Path,
    source: dict[str, Any],
    target_domain: str,
    target_backend: str,
    native_tools: list[str],
    disposition: str,
    blockers: list[str],
    transfer_method: str,
    notes: str,
    local_assets: list[dict[str, Any]] | None = None,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return _recipe_base(
        source=source,
        target_domain=target_domain,
        target_backend=target_backend,
        native_tools=native_tools,
        local_assets=local_assets or [],
        evidence=evidence or [],
        gates=_gates(),
        disposition=disposition,
        transfer_method=transfer_method,
        blockers=blockers,
        notes=notes,
    )


def build_track_c_report(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Build a deterministic, fail-closed Track-C conversion catalog."""

    recipes: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    dyna, dyna_candidates = _dyna_recipe(repo_root)
    recipes.append(dyna)
    candidates.extend(dyna_candidates)

    recipes.append(
        _method_only_recipe(
            repo_root=repo_root,
            source=_external_source(
                source_id="oragentbench",
                title="ORAgentBench",
                urls=[
                    "https://oragentbench.github.io/",
                    "https://github.com/ORAgentBench/ORAgentBench",
                    "https://arxiv.org/abs/2606.19787",
                ],
                license_value="task-specific terms require review",
                license_status="not_verified",
                version_value=None,
                version_status="release_or_commit_unlocked",
            ),
            target_domain="native_backend_per_task",
            target_backend="native_backend_per_task",
            native_tools=["backend_native_tools_only_until_adapter_exists"],
            disposition="method_transfer_only",
            blockers=[
                "offline_task_not_native_environment",
                "source_asset_license_unresolved",
                "native_clock_unresolved",
                "native_state_effect_unproven",
            ],
            transfer_method="Reuse executable-artifact, hidden-feasibility, and normalized-objective validator separation only.",
            notes="Offline OR workflow tasks and QA answers are not simulator-owned, replayable Protocol-2.1 environments.",
        )
    )
    recipes.append(
        _method_only_recipe(
            repo_root=repo_root,
            source=_external_source(
                source_id="elecbench",
                title="ElecBench",
                urls=["https://arxiv.org/abs/2407.05365"],
                license_value="dataset/question terms require review",
                license_status="not_verified",
                version_value="arXiv:2407.05365",
                version_status="paper_identifier_evidenced",
            ),
            target_domain="power_grid",
            target_backend="grid2op_or_pandapower_native",
            native_tools=["set_topology", "redispatch_generation", "set_storage_power"],
            disposition="method_transfer_only",
            blockers=[
                "QA_only_not_replayable",
                "external_runtime_unresolved",
                "native_state_effect_unproven",
                "source_asset_unresolved",
            ],
            transfer_method="Map safety/stability/security/fairness rubric concepts to evidence-linked native power metrics only.",
            notes="QA prompts and textual dispatch labels never become native power-grid episodes.",
        )
    )
    recipes.append(
        _method_only_recipe(
            repo_root=repo_root,
            source=_external_source(
                source_id="realm_bench",
                title="REALM-Bench",
                urls=[
                    "https://github.com/genglongling/REALM-Bench",
                    "https://huggingface.co/datasets/genglongling/REALM-Bench",
                    "https://arxiv.org/abs/2502.18836",
                ],
                license_value="MIT code; CC-BY-4.0 dataset; upstream instance terms apply",
                license_status="public_claimed_upstream_chain_not_locked_locally",
                version_value="9c3aa2ae97d65198f6ee29fe942d99f9b3a9c6eb",
                version_status="pilot_manifest_claim_not_local_checkout",
            ),
            target_domain="logistics",
            target_backend="jsplib_job_shop_or_pyvrp_cvrp",
            native_tools=["dispatch_job_operation", "dispatch_route_stop", "resequence_vehicle_route"],
            disposition="pilot_only",
            blockers=[
                "external_instance_lineage_unresolved",
                "source_license_chain_incomplete",
                "event_runtime_unproven",
                "counterfactual_replay_pending",
            ],
            transfer_method="Use multi-step planning/disruption decomposition on an unused, separately locked JSPLIB or VRPLIB source key.",
            notes="Local JSPLIB/PyVRP directories are context-only; no REALM-Bench asset is claimed as consumed.",
            local_assets=[
                _asset("works/JSPLIB-Instances/instances", role="context_only_native_anchor"),
                _asset("works/PyVRP-Instances", role="context_only_native_anchor"),
            ],
        )
    )
    recipes.append(
        _method_only_recipe(
            repo_root=repo_root,
            source=_external_source(
                source_id="frontier_eng",
                title="Frontier-Eng",
                urls=[
                    "https://arxiv.org/abs/2604.12290",
                    "https://lab.einsia.ai/frontier-eng/",
                    "https://github.com/EinsiaLab/Frontier-Engineering",
                ],
                license_value="task/repository terms require per-file review",
                license_status="pilot_manifest_claim_not_locally_verified",
                version_value="e3fa29c193356af2ce1ec8b3d23ab1a2e2410071",
                version_status="pilot_manifest_claim_not_local_checkout",
            ),
            target_domain="microgrid",
            target_backend="ev2gym",
            native_tools=["set_ev_charging_power", "inspect_transformer_loading", "inspect_ev_soc_and_departures"],
            disposition="pilot_only",
            blockers=[
                "adapter_not_implemented",
                "local_asset_lock_pending",
                "native_state_effect_unproven",
                "replay_and_depth_unproven",
            ],
            transfer_method="Reuse the bounded propose/execute/evaluate/revise loop on a native EV charging simulator after locking task and profile files.",
            notes="EV2Gym upstream commit and task path are listed in the pilot manifest, but no local asset lock or native adapter exists.",
            evidence=[
                _evidence(
                    "release/dt_sched_bench_v0_52_0_candidate/protocol21_long_horizon_state/external_benchmark_pilot_manifest_v1.json",
                    "Frontier-Eng EV2Gym pilot manifest; lock and adapter remain pending",
                )
            ],
        )
    )
    recipes.append(
        _method_only_recipe(
            repo_root=repo_root,
            source=_external_source(
                source_id="edgebench",
                title="EdgeBench",
                urls=["https://arxiv.org/abs/2607.05155", "https://edge-bench.org/"],
                license_value="heterogeneous task-release terms require review",
                license_status="not_verified",
                version_value="arXiv:2607.05155",
                version_status="paper_identifier_evidenced",
            ),
            target_domain="native_backend_per_task",
            target_backend="native_backend_per_task",
            native_tools=["backend_native_tools_only_until_adapter_exists"],
            disposition="diagnostic_only",
            blockers=[
                "heterogeneous_task_not_domain_native",
                "local_task_asset_missing",
                "native_control_leverage_unproven",
                "source_independence_unresolved",
            ],
            transfer_method="Apply ultra-long-horizon checkpoints and time-normalized trajectory views to already locked native episodes only.",
            notes="Duration and feedback cadence do not establish a native scheduling control surface or independent source.",
        )
    )

    recipes.extend(
        [
            _cityflow_recipe(repo_root),
            _simbench_recipe(repo_root),
            _citylearn_recipe(repo_root),
            _method_only_recipe(
                repo_root=repo_root,
                source=_external_source(
                    source_id="acnsim_official",
                    title="ACN-Sim / ACNPortal",
                    urls=[
                        "https://github.com/zach401/acnportal",
                        "https://acnportal.readthedocs.io/",
                        "https://pypi.org/project/acnportal/",
                    ],
                    license_value="BSD-3-Clause (upstream claim; verify LICENSE)",
                    license_status="not_verified_no_local_checkout",
                    version_value=None,
                    version_status="package_version_unlocked",
                ),
                target_domain="ev_charging_scheduling",
                target_backend="acnsim",
                native_tools=[
                    "allocate_charging_current",
                    "set_evse_pilot",
                    "prioritize_departing_session",
                ],
                disposition="pilot_only",
                blockers=[
                    "acnsim_source_checkout_missing",
                    "acnsim_runtime_not_importable",
                    "session_snapshot_and_network_lock_missing",
                    "native_state_effect_unproven",
                ],
                transfer_method="Build an EV-charging-native adapter around locked ACN-Data sessions and ACN-Sim network constraints.",
                notes="No ACNPortal/ACN-Sim checkout or package is available in the current environment; do not substitute NREL traces or synthetic sessions.",
            ),
            _method_only_recipe(
                repo_root=repo_root,
                source=_external_source(
                    source_id="batsim_hpc_official",
                    title="Batsim / HPC scheduling",
                    urls=[
                        "https://github.com/oar-team/batsim",
                        "https://batsim.readthedocs.io/",
                    ],
                    license_value="CeCILL-C (upstream claim; file-level lock required)",
                    license_status="not_verified_no_local_checkout",
                    version_value=None,
                    version_status="release_or_commit_unlocked",
                ),
                target_domain="datacenter_hpc_scheduling",
                target_backend="batsim",
                native_tools=[
                    "submit_hpc_job",
                    "preempt_hpc_job",
                    "place_hpc_job_on_node",
                ],
                disposition="pilot_only",
                blockers=[
                    "batsim_source_checkout_missing",
                    "batsim_runtime_not_available",
                    "workload_and_platform_file_lock_missing",
                    "native_adapter_missing",
                ],
                transfer_method="Transfer Batsim workload/platform event semantics only to a separately locked HPC trace/runtime; do not relabel Alibaba trace rows as Batsim.",
                notes="The local Alibaba clusterdata checkout is a distinct native source and remains context-only for this Batsim recipe.",
                local_assets=[
                    _asset("works/clusterdata", role="context_only_non_batsim_hpc_source"),
                    _asset(
                        "release/dt_sched_bench_v0_52_0_candidate/backend_conversion_status.json",
                        role="separate_alibaba_native_backend_evidence",
                    ),
                ],
            ),
            _flatland_recipe(repo_root),
        ]
    )

    if len(recipes) != 12:
        raise AssertionError(f"Track C requires 12 source recipes, got {len(recipes)}")
    canonical = json.dumps(recipes, sort_keys=True, separators=(",", ":"))
    ready_sources = [recipe["source_id"] for recipe in recipes if recipe["ready_for_full_protocol21"]]
    direct_external = [
        row
        for recipe in recipes
        for row in recipe.get("candidate_rows", [])
        if row.get("external_source", {}).get("raw_asset_consumed")
    ]
    # Candidate rows are kept in a separate top-level list so downstream tools
    # do not have to infer admission state from a recipe's prose.
    return {
        "schema_version": "protocol21-track-c-external-conversion-v1",
        "generated_on": REVIEW_DATE,
        "status": "staging_only",
        "scope": "DynaSchedBench, ORAgentBench, ElecBench, REALM-Bench, Frontier-Eng, EdgeBench, CityFlow, SimBench, CityLearn, ACN-Sim, Batsim/HPC, and Flatland",
        "direct_core_admission": False,
        "external_raw_asset_admission": False,
        "required_protocol21_gates": list(REQUIRED_GATES),
        "promotion_rule": "External methods may transfer only to a separately locked native source. Raw QA, paper, hand-authored, or generated external rows never enter Core. Only a candidate with every required gate true may enter the ordinary Protocol-2.1 pipeline; readiness never implies Core admission.",
        "n_sources": len(recipes),
        "n_ready_for_full_protocol21": len(ready_sources),
        "ready_source_ids": ready_sources,
        "n_candidate_rows": len(candidates),
        "n_direct_external_core_admitted": 0,
        "source_ids": [recipe["source_id"] for recipe in recipes],
        "recipes": recipes,
        "candidate_rows": candidates,
        "summary": {
            "dispositions": {
                disposition: sum(recipe["disposition"] == disposition for recipe in recipes)
                for disposition in sorted({recipe["disposition"] for recipe in recipes})
            },
            "ready_candidates": [row["candidate_id"] for row in candidates if row["ready_for_full_protocol21"]],
            "blocked_sources": {
                recipe["source_id"]: recipe["blocker_codes"]
                for recipe in recipes
                if not recipe["ready_for_full_protocol21"]
            },
            "external_raw_rows_forbidden": True,
            "direct_external_rows": len(direct_external),
        },
        "catalog_fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_track_c_report(args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_sources": report["n_sources"],
                "n_ready_for_full_protocol21": report["n_ready_for_full_protocol21"],
                "n_direct_external_core_admitted": report["n_direct_external_core_admitted"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
