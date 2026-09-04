#!/usr/bin/env python3
"""Build a source-first external-benchmark conversion queue.

This is a staging report, not a scenario materializer.  External benchmark
instances, QA prompts, puzzles, and generated worlds are never admitted to
Core directly.  A row is promoted only after a native backend consumes a
locked asset and emits runtime evidence for state-changing control, task
completion, response windows, difficulty depth, and counterfactual replay.

The queue intentionally includes high/extreme blueprints for underrepresented
domains.  Their presence records concrete work to perform; it does not turn
unexecuted plans into release data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "external_transfer_candidates_v2"
    / "queue.json"
)
DEFAULT_WORKING_SET = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "protocol21_expansion_trials"
    / "working_set_resco_v2"
    / "source_suite.json"
)

DIFFICULTY_LEVELS = ("basic", "medium", "high", "extreme")
REQUIRED_GATES = (
    "source_lock",
    "runtime_consumption",
    "native_state_effect",
    "task_contract",
    "response_window",
    "difficulty_depth",
    "counterfactual_replay",
)

_DOMAIN_EVENT_AXES: dict[str, list[str]] = {
    "datacenter": [
        "job_arrival_burst",
        "capacity_reduction",
        "queue_deadline_change",
    ],
    "logistics": [
        "job_release_or_arrival",
        "machine_or_vehicle_unavailability",
        "processing_or_travel_time_change",
    ],
    "microgrid": [
        "pv_or_load_ramp",
        "battery_or_der_derating",
        "islanding_or_grid_alarm",
    ],
    "power_grid": [
        "load_step",
        "line_or_generator_outage",
        "voltage_or_reserve_alarm",
    ],
    "traffic": [
        "demand_peak",
        "queue_spillback",
        "incident_overlay",
    ],
}

_DEPTH_FLOORS: dict[str, dict[str, int]] = {
    "basic": {"decision_ticks": 1, "control_types": 1, "plan_switches": 0},
    "medium": {"decision_ticks": 2, "control_types": 1, "plan_switches": 0},
    "high": {"decision_ticks": 2, "control_types": 2, "plan_switches": 1},
    "extreme": {"decision_ticks": 3, "control_types": 2, "plan_switches": 2},
}


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_digest(rows: list[tuple[str, str]]) -> str:
    body = "".join(f"{path}\0{digest}\n" for path, digest in sorted(rows))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _git_commit(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    return commit or None


def _read_source_lock(root: Path) -> tuple[dict[str, str], str | None]:
    """Read an explicit local source lock without treating it as runtime proof."""

    lock_path = root / "source_lock.json"
    if not lock_path.is_file():
        return {}, None
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, str(lock_path)
    if not isinstance(payload, dict):
        return {}, str(lock_path)
    values = {
        key: str(payload[key])
        for key in ("source_url", "license", "git_commit_or_release_tag")
        if payload.get(key)
    }
    files = payload.get("files")
    if isinstance(files, dict):
        rows = [(str(path), str(digest)) for path, digest in files.items()]
        values["manifest_digest"] = _canonical_digest(rows)
    return values, str(lock_path)


def _asset_inventory(
    *, repo_root: Path, relative_root: str, max_paths: int = 24
) -> dict[str, Any]:
    """Return a bounded, source-identifying inventory for one local family."""

    root = (repo_root / relative_root).resolve()
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and ".git" not in path.parts
            and path.name != ".DS_Store"
        )
    else:
        files = []

    config = _SOURCE_FAMILY_BY_ROOT.get(relative_root, {})
    lock_root = (repo_root / str(config.get("lock_root", relative_root))).resolve()
    lock_values, lock_path = _read_source_lock(lock_root)
    git_commit = _git_commit(lock_root)
    commit = git_commit or str(config.get("commit") or "") or None
    digest_scope = "none"
    tree_digest: str | None = None
    provenance_files = sorted(root.glob("*.provenance.json"))
    if git_commit:
        tree_result = subprocess.run(
            ["git", "-C", str(lock_root), "rev-parse", "HEAD^{tree}"],
            check=False,
            capture_output=True,
            text=True,
        )
        if tree_result.returncode == 0:
            tree_digest = tree_result.stdout.strip() or None
            digest_scope = "locked_git_tree"
        else:
            # A static commit is still a provenance lock, but cannot stand in
            # for an observed repository tree when the local checkout is not
            # a git worktree.
            tree_digest = hashlib.sha256(commit.encode("utf-8")).hexdigest()
            digest_scope = "declared_commit_only"
    elif lock_values.get("manifest_digest"):
        tree_digest = lock_values["manifest_digest"]
        digest_scope = "declared_source_lock_manifest"
    elif provenance_files:
        provenance_rows: list[tuple[str, str]] = []
        for provenance_path in provenance_files:
            try:
                payload = json.loads(provenance_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            digest = str(payload.get("sha256") or "")
            locks = payload.get("source_file_locks")
            if isinstance(locks, dict):
                digest += json.dumps(locks, sort_keys=True)
            if digest:
                provenance_rows.append((provenance_path.name, digest))
        if provenance_rows:
            tree_digest = _canonical_digest(provenance_rows)
            digest_scope = "provenance_manifest"
        elif files:
            sampled = [
                (str(path.relative_to(root)), _sha256_file(path) or "")
                for path in files[:max_paths]
            ]
            tree_digest = _canonical_digest(sampled)
            digest_scope = "sampled_asset_manifest"
    elif commit:
        tree_digest = hashlib.sha256(commit.encode("utf-8")).hexdigest()
        digest_scope = "declared_commit_only"
    elif files:
        # This fallback is intentionally bounded and marked sampled.  A
        # sampled digest is not sufficient for Core admission by itself.
        sampled = [
            (str(path.relative_to(root)), _sha256_file(path) or "")
            for path in files[:max_paths]
        ]
        tree_digest = _canonical_digest(sampled)
        digest_scope = "sampled_asset_manifest"

    rel_paths = [str(path.relative_to(repo_root)) for path in files[:max_paths]]
    lock_manifest_relative = None
    if lock_path:
        lock_candidate = Path(lock_path)
        try:
            lock_manifest_relative = str(lock_candidate.resolve().relative_to(repo_root))
        except ValueError:
            lock_manifest_relative = str(lock_candidate)
    adapter = config.get("runtime_adapter")
    adapter_path = repo_root / str(adapter) if adapter else None
    adapter_sha = _sha256_file(adapter_path) if adapter_path else None
    source_family = config.get("source_family", relative_root.replace("/", "_"))
    source_token = tree_digest or "missing"
    return {
        "source_family": source_family,
        "root": relative_root,
        "path_exists": root.exists(),
        "exists": bool(files),
        "asset_file_count": len(files),
        "asset_paths": rel_paths,
        "asset_paths_truncated": len(files) > max_paths,
        "tree_sha256": tree_digest,
        "digest_scope": digest_scope,
        "source_lock_manifest": lock_manifest_relative,
        "source_lock_status": (
            "locked"
            if digest_scope
            in {
                "locked_git_tree",
                "declared_source_lock_manifest",
                "provenance_manifest",
            }
            else "sampled_not_release" if digest_scope == "sampled_asset_manifest" else "missing"
        ),
        "source_lock_values": lock_values,
        "commit": commit,
        "source_denominator_key": f"{source_family}:source:{source_token}",
        "physical_source_key": f"{source_family}:physical:{source_token}",
        "runtime_key": (
            f"backend:{config.get('backend')}"
            f":adapter:{adapter_sha or 'missing'}"
        ),
        "runtime_adapter": adapter,
        "runtime_adapter_exists": bool(adapter_path and adapter_path.is_file()),
    }


_SOURCE_FAMILY_BY_ROOT: dict[str, dict[str, Any]] = {
    "works/JSPLIB-Instances/instances": {
        "source_family": "jsplib",
        "lock_root": "works/JSPLIB-Instances",
        "domain": "logistics",
        "backend": "jsplib_job_shop",
        "source_url": "https://github.com/tamy0612/JSPLIB",
        "commit": "eea2b60dd7e2f5c907ff7302662c61812eb7efdf",
        "license": "public academic OR benchmark mirror; see upstream LICENSE",
        "runtime_adapter": "domains/logistics/backends/job_shop.py",
        "native_tools": ["dispatch_job_operation", "resequence_job_operations"],
        "target_external": {"dynaschedbench", "realm_bench", "oragentbench"},
    },
    "works/M5": {
        "source_family": "orgym_m5",
        "domain": "logistics",
        "backend": "orgym_invmgmt",
        "source_url": "https://www.kaggle.com/competitions/m5-forecasting-accuracy",
        "commit": "m5-forecasting-accuracy 2020-06-01 files",
        "license": "Kaggle competition rules; non-redistributable raw files",
        "runtime_adapter": "domains/logistics/backends/orgym_invmgmt.py",
        "native_tools": ["place_replenishment_order", "inspect_inventory_state"],
        "target_external": {"oragentbench", "frontier_eng", "edgebench"},
    },
    "works/PyVRP-Instances": {
        "source_family": "pyvrp",
        "domain": "logistics",
        "backend": "pyvrp_cvrp",
        "source_url": "https://github.com/PyVRP/Instances",
        "commit": "1cf23a5969fabf23c80f8002e42ed501a47aca61",
        "license": "mirror MIT; underlying instances research-use-only",
        "runtime_adapter": "domains/logistics/backends/pyvrp_cvrp.py",
        "native_tools": ["dispatch_route_stop", "resequence_vehicle_route"],
        "target_external": {"realm_bench", "oragentbench", "edgebench"},
    },
    "works/OpenDSS-IEEE13": {
        "source_family": "opendss_ieee13",
        "domain": "power_grid",
        "backend": "opendss_ieee13",
        "source_url": "https://github.com/dss-extensions/OpenDSSDirect.py",
        "commit": "3b20839",
        "license": "upstream feeder assets; verify individual data terms",
        "runtime_adapter": "domains/power_grid/backends/opendss_ieee13.py",
        "native_tools": ["set_tap_position", "set_capacitor_state", "redispatch_generation"],
        "target_external": {"elecbench", "frontier_eng", "edgebench"},
    },
    "works/Grid2Op_cache": {
        "source_family": "grid2op",
        "domain": "power_grid",
        "backend": "grid2op",
        "source_url": "https://github.com/Grid2Op/grid2op",
        "commit": "d74b8e11a238ebea40fd17694529347bb4854d3c",
        "license": "Grid2Op license; scenario data terms must be locked separately",
        "runtime_adapter": "domains/power_grid/backends/grid2op_backend.py",
        "native_tools": ["set_topology", "redispatch_generation", "set_storage_power"],
        "target_external": {"elecbench", "frontier_eng", "edgebench"},
    },
    "works/PGLib-OPF": {
        "source_family": "pglib_opf",
        "domain": "power_grid",
        "backend": "pandapower_acopf",
        "source_url": "https://github.com/power-grid-lib/pglib-opf",
        "commit": "dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3",
        "license": "PGLib-OPF license and case-specific terms",
        "runtime_adapter": "domains/power_grid/backends/pandapower_acopf.py",
        "native_tools": ["set_tap_position", "redispatch_generation", "set_shunt_state"],
        "target_external": {"elecbench", "oragentbench", "frontier_eng"},
    },
    "works/nrel-microgrid": {
        "source_family": "nrel_microgrid",
        "domain": "microgrid",
        "backend": "pandapower_lv",
        "source_url": "https://github.com/NREL/ComStock",
        "commit": "local-provenance-locked",
        "license": "NREL profile terms recorded per provenance file",
        "runtime_adapter": "domains/microgrid/backends/pandapower_lv.py",
        "native_tools": ["set_storage_power", "set_pv_curtailment", "shed_load"],
        "target_external": {"elecbench", "frontier_eng", "edgebench"},
    },
    "works/RESCO": {
        "source_family": "resco_sumo",
        "domain": "traffic",
        "backend": "sumo",
        "source_url": "https://github.com/PepMS/RESCO",
        "commit": "f1ed9a1",
        "license": "RESCO repository license; scenario assets locked per network",
        "runtime_adapter": "domains/traffic/backends/sumo_backend.py",
        "native_tools": ["query_signal_control", "set_signal_phase_duration"],
        "target_external": {"frontier_eng", "edgebench", "oragentbench"},
    },
    "works/clusterdata": {
        "source_family": "alibaba_clusterdata",
        "domain": "datacenter",
        "backend": "alibaba_trace_sim",
        "source_url": "https://github.com/alibaba/clusterdata",
        "commit": "0d0f3f1",
        "license": "Alibaba trace release terms; release-specific files locked",
        "runtime_adapter": "domains/datacenter/backends/alibaba_trace_backend.py",
        "native_tools": ["assign_job", "preempt_job", "inspect_cluster_state"],
        "target_external": {"oragentbench", "frontier_eng", "edgebench"},
    },
}


_EXTERNAL_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "dynaschedbench",
        "title": "DynaSchedBench",
        "url": "https://arxiv.org/abs/2605.27566",
        "raw_kind": "synthetic_dfjsp_instances",
        "target_domain": "logistics",
        "target_backend": "jsplib_job_shop",
        "target_families": ("jsplib",),
        "reusable_methods": ["event-stream stress axes", "dynamic release metadata"],
        "raw_policy": "method_transfer_only",
    },
    {
        "source_id": "oragentbench",
        "title": "ORAgentBench",
        "url": "https://oragentbench.github.io/",
        "raw_kind": "offline_or_task_corpus",
        "target_domain": "cross_domain",
        "target_backend": "native_backend_per_task",
        "target_families": ("jsplib", "orgym_m5", "pyvrp", "pglib_opf", "resco_sumo"),
        "reusable_methods": ["hidden feasibility validator", "objective decomposition"],
        "raw_policy": "validator_method_only",
    },
    {
        "source_id": "elecbench",
        "title": "ElecBench",
        "url": "https://arxiv.org/abs/2407.05365",
        "raw_kind": "power_qa",
        "target_domain": "power_grid",
        "target_backend": "native_power_grid_backend",
        "target_families": ("opendss_ieee13", "grid2op", "pglib_opf"),
        "reusable_methods": ["safety rubric", "professional scenario taxonomy"],
        "raw_policy": "qa_not_replayable",
    },
    {
        "source_id": "realm_bench",
        "title": "REALM-Bench",
        "url": "https://github.com/genglongling/REALM-Bench",
        "raw_kind": "handwritten_planning_puzzles",
        "target_domain": "logistics",
        "target_backend": "native_logistics_backend",
        "target_families": ("jsplib", "pyvrp"),
        "reusable_methods": ["long-horizon decomposition", "disruption/replanning labels"],
        "raw_policy": "puzzle_not_source_data",
    },
    {
        "source_id": "frontier_eng",
        "title": "Frontier-Eng",
        "url": "https://arxiv.org/abs/2604.12290",
        "raw_kind": "mixed_engineering_tasks",
        "target_domain": "cross_domain",
        "target_backend": "native_backend_per_task",
        "target_families": (
            "opendss_ieee13",
            "grid2op",
            "pglib_opf",
            "nrel_microgrid",
            "clusterdata",
            "resco_sumo",
        ),
        "reusable_methods": ["propose/execute/evaluate/revise loop", "verifier separation"],
        "raw_policy": "mixed_assets_require_independent_locks",
    },
    {
        "source_id": "edgebench",
        "title": "EdgeBench",
        "url": "https://arxiv.org/abs/2607.05155",
        "raw_kind": "long_horizon_agent_traces",
        "target_domain": "cross_domain",
        "target_backend": "native_backend_per_task",
        "target_families": ("orgym_m5", "pyvrp", "nrel_microgrid", "clusterdata", "resco_sumo"),
        "reusable_methods": ["long-horizon logging", "effective-source aggregation"],
        "raw_policy": "trace_method_only",
    },
)


def _load_working_set(path: Path, repo_root: Path) -> dict[str, Any]:
    resolved = path if path.is_absolute() else repo_root / path
    if not resolved.is_file():
        return {
            "path": str(resolved),
            "exists": False,
            "sha256": None,
            "status": "missing",
            "n_scenarios": 0,
        }
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    return {
        "path": str(resolved),
        "exists": True,
        "sha256": _sha256_file(resolved),
        "status": payload.get("status", "unknown") if isinstance(payload, dict) else "invalid",
        "n_scenarios": len(payload.get("scenarios", [])) if isinstance(payload, dict) else 0,
    }


def _family_inventory(repo_root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for relative_root, config in _SOURCE_FAMILY_BY_ROOT.items():
        row = _asset_inventory(repo_root=repo_root, relative_root=relative_root)
        row.update(
            {
                "domain": config["domain"],
                "backend": config["backend"],
                "source_url": config["source_url"],
                "source_commit": config["commit"],
                "license": config["license"],
                "native_tools": list(config["native_tools"]),
                "event_axes": list(_DOMAIN_EVENT_AXES[config["domain"]]),
            }
        )
        inventory[str(config["source_family"])] = row
    return inventory


def _candidate(
    *,
    external: dict[str, Any],
    level: str,
    inventory: dict[str, dict[str, Any]],
    local_family: str | None,
) -> dict[str, Any]:
    if local_family is None:
        source_key = f"external:{external['source_id']}:raw:{level}"
        physical_key = f"external:{external['source_id']}:unresolved"
        runtime_key = f"pending-native-adapter:{external['target_backend']}"
        domain = str(external["target_domain"])
        backend = str(external["target_backend"])
        tools = ["backend_native_tools_only"]
        source_files: list[str] = []
        asset_exists = False
        reason_codes = [
            "external_raw_not_admissible",
            "external_source_asset_not_downloaded",
            "native_runtime_pending",
        ]
    else:
        source = inventory[local_family]
        source_key = f"{source['source_denominator_key']}:method:{external['source_id']}"
        physical_key = str(source["physical_source_key"])
        runtime_key = str(source["runtime_key"])
        domain = str(source["domain"])
        backend = str(source["backend"])
        tools = list(source["native_tools"])
        source_files = list(source["asset_paths"])
        asset_exists = bool(source["exists"])
        reason_codes = []
        if not asset_exists:
            reason_codes.append("source_asset_missing")
        if source["source_lock_status"] != "locked":
            reason_codes.append("source_lock_fingerprint_incomplete")
        if not source["runtime_adapter_exists"]:
            reason_codes.append("runtime_adapter_missing")
        reason_codes.append("native_runtime_evidence_pending")

    reason_codes.extend(
        [
            "native_state_effect_unproven",
            "task_contract_pending",
            "response_window_pending",
            "difficulty_depth_pending",
            "counterfactual_replay_pending",
        ]
    )
    candidate_id = "__".join(
        [external["source_id"], local_family or "external_raw", level]
    )
    structural_payload = {
        "external_source": external["source_id"],
        "local_family": local_family,
        "level": level,
        "physical_source_key": physical_key,
        "event_axes": _DOMAIN_EVENT_AXES.get(domain, []),
    }
    structural_fp = hashlib.sha256(
        json.dumps(structural_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    return {
        "candidate_id": candidate_id,
        "status": "held",
        "transfer_mode": "external_raw_asset" if local_family is None else "native_method_transfer",
        "external_source_id": external["source_id"],
        "external_source_url": external["url"],
        "external_raw_policy": external["raw_policy"],
        "target_domain": domain,
        "target_backend": backend,
        "difficulty_level": level,
        "source_denominator_key": source_key,
        "physical_source_key": physical_key,
        "runtime_key": runtime_key,
        "structural_fingerprint": structural_fp,
        "source_asset_paths": source_files,
        "source_assets_exist": asset_exists,
        "native_tools": tools,
        "event_contract": {
            "event_axes": _DOMAIN_EVENT_AXES.get(domain, []),
            "simulator_owns_clock": True,
            "response_window_required": True,
            "response_window_proven": False,
        },
        "difficulty_contract": {
            "level": level,
            "minimums": dict(_DEPTH_FLOORS[level]),
            "observed": False,
        },
        "required_protocol21_gates": list(REQUIRED_GATES),
        "reason_codes": sorted(set(reason_codes)),
        "promotion": {
            "direct_core_admission": False,
            "requires_native_replay": True,
            "requires_independent_physical_source": True,
            "requires_all_gates": True,
        },
    }


def build_external_transfer_queue(
    *,
    repo_root: Path = REPO_ROOT,
    working_set_path: Path = DEFAULT_WORKING_SET,
) -> dict[str, Any]:
    """Build a deterministic candidate queue without promoting any row."""

    repo_root = repo_root.resolve()
    inventory = _family_inventory(repo_root)
    candidates: list[dict[str, Any]] = []
    for external in _EXTERNAL_SOURCES:
        for level in DIFFICULTY_LEVELS:
            candidates.append(
                _candidate(
                    external=external,
                    level=level,
                    inventory=inventory,
                    local_family=None,
                )
            )
        for family in external["target_families"]:
            if family not in inventory:
                continue
            for level in DIFFICULTY_LEVELS:
                candidates.append(
                    _candidate(
                        external=external,
                        level=level,
                        inventory=inventory,
                        local_family=family,
                    )
                )

    by_domain = Counter(str(row["target_domain"]) for row in candidates)
    by_difficulty = Counter(str(row["difficulty_level"]) for row in candidates)
    by_mode = Counter(str(row["transfer_mode"]) for row in candidates)
    by_reason = Counter(
        reason
        for row in candidates
        for reason in row["reason_codes"]
    )
    return {
        "schema_version": "protocol21-external-transfer-candidates-v2",
        "status": "candidate_queue_ready_no_core_admission",
        "direct_core_admission": False,
        "n_candidates": len(candidates),
        "n_ready": sum(row["status"] == "ready" for row in candidates),
        "n_held": sum(row["status"] == "held" for row in candidates),
        "n_external_sources": len(_EXTERNAL_SOURCES),
        "n_local_source_families": len(inventory),
        "working_set": _load_working_set(working_set_path, repo_root),
        "external_sources": [
            {
                **external,
                "direct_core_admission": False,
                "source_lock_status": "reference_only_until_asset_lock",
            }
            for external in _EXTERNAL_SOURCES
        ],
        "local_source_inventory": list(inventory.values()),
        "candidate_summary": {
            "by_domain": dict(sorted(by_domain.items())),
            "by_difficulty": dict(sorted(by_difficulty.items())),
            "by_transfer_mode": dict(sorted(by_mode.items())),
            "top_reason_codes": dict(by_reason.most_common()),
        },
        "promotion_policy": {
            "external_raw_data_direct_admission": False,
            "synthetic_or_qa_or_puzzle_direct_admission": False,
            "native_method_transfer_may_use_local_source": True,
            "all_protocol21_gates_required": True,
            "no_count_target": True,
            "high_extreme_require_observed_depth": True,
            "duplicate_physical_source_key_is_not_new_source": True,
        },
        "candidates": candidates,
    }


def _write_report(report: dict[str, Any], output: Path, summary_output: Path | None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": report["schema_version"],
            "status": report["status"],
            "n_candidates": report["n_candidates"],
            "n_ready": report["n_ready"],
            "n_held": report["n_held"],
            "n_external_sources": report["n_external_sources"],
            "n_local_source_families": report["n_local_source_families"],
            "working_set": report["working_set"],
            "candidate_summary": report["candidate_summary"],
        }
        summary_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--working-set", type=Path, default=DEFAULT_WORKING_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args(argv)
    report = build_external_transfer_queue(
        repo_root=args.repo_root,
        working_set_path=args.working_set,
    )
    _write_report(report, args.output, args.summary_output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_candidates": report["n_candidates"],
                "n_ready": report["n_ready"],
                "n_held": report["n_held"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
