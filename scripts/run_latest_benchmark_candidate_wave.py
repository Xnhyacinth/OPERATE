#!/usr/bin/env python3
"""Build a deterministic terminal ledger for the latest benchmark wave.

This command is deliberately candidate-only.  It inventories every requested
external benchmark, binds local assets and license evidence when present, and
merges already-produced native prefilter reports without rerunning a backend.
It never copies raw data, changes the frozen Core, or turns a paper/benchmark
recipe into a release scenario.  Rows that lack a source lock, native runtime,
or redistribution terms receive an explicit terminal disposition.

The expensive native conversion commands remain separate (for example
``run_external_native_conversion_wave.py``).  This ledger is the coordinator's
bounded, parallel static layer: it makes the whole wave auditable and prevents
silent drops while those commands are run in isolated shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPO_ROOT / ".hl/artifacts/works_candidate_inventory_2026-08-12.json"
DEFAULT_EXISTING_WAVE = REPO_ROOT / "reports/external_native_conversion_wave_20260813.json"
DEFAULT_REPORT = REPO_ROOT / "reports/latest_benchmark_candidate_wave_20260813.json"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports/latest_benchmark_candidate_wave_20260813"
PIPELINE_VERSION = "latest_benchmark_candidate_wave_v1"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _git_status(path: Path) -> list[str]:
    if not (path / ".git").exists():
        return []
    completed = subprocess.run(
        ("git", "-C", str(path), "status", "--short"),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return [f"git_status_failed:{completed.returncode}"]
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _root_info(roots: list[str], *, repo_root: Path) -> dict[str, Any]:
    resolved = [(repo_root / root).resolve() for root in roots]
    present = next((path for path in resolved if path.is_dir()), None)
    files = []
    if present is not None:
        files = [path for path in present.rglob("*") if path.is_file()]
    return {
        "declared_roots": roots,
        "present_root": _repo_relative(present) if present else None,
        "asset_count": len(files),
        "git_dirty_entries": _git_status(present) if present else [],
    }


def latest_benchmark_catalog() -> tuple[dict[str, Any], ...]:
    """Return the external sources that must get one explicit terminal row.

    URLs/licenses are source metadata, not admission evidence.  Missing local
    assets and unresolved terms intentionally stay visible as held rows.
    """

    return (
        {
            "source_id": "acn_ev_charging",
            "title": "ACN-Data / ACN-Sim",
            "domain": "vehicle",
            "backend_kind": "acn_sim",
            "source_url": "https://ev.caltech.edu/dataset.html",
            "license": "dataset-specific-research-terms",
            "local_roots": ["works/ACN-Data", "works/acn-data"],
            "planned_conversion": "native_charge_control",
            "default_disposition": "held_license_or_terms",
            "required_native_gates": [
                "source_lock",
                "station_cluster_identity",
                "timestamp_window_consumed",
                "native_charge_control_effect",
                "deterministic_counterfactual_replay",
            ],
        },
        {
            "source_id": "building_data_genome_2",
            "title": "Building Data Genome Project 2",
            "domain": "building_energy",
            "backend_kind": "boptest_or_citylearn",
            "source_url": "https://github.com/buds-lab/building-data-genome-project-2",
            "license": "CC-BY-SA-4.0-terms-to-verify",
            "local_roots": ["works/building-data-genome-project-2"],
            "planned_conversion": "trace_driven_boptest_or_citylearn",
            "default_disposition": "held_license_or_terms",
            "required_native_gates": [
                "source_lock",
                "building_graph_lock",
                "closed_loop_backend",
                "native_hvac_or_storage_effect",
                "comfort_energy_tradeoff",
            ],
        },
        {
            "source_id": "boptest",
            "title": "IBPSA BOPTEST",
            "domain": "building_energy",
            "backend_kind": "boptest",
            "source_url": "https://github.com/ibpsa/project1-boptest",
            "license": "BSD-3-Clause-code-model-terms-to-verify",
            "local_roots": ["works/BOPTEST", "works/boptest"],
            "planned_conversion": "native_boptest_control",
            "default_disposition": "held_missing_assets",
            "required_native_gates": [
                "versioned_testcase_lock",
                "container_runtime_lock",
                "native_control_effect",
                "terminal_parity",
            ],
        },
        {
            "source_id": "cityflow_examples",
            "title": "CityFlow",
            "domain": "traffic",
            "backend_kind": "cityflow",
            "source_url": "https://github.com/cityflow-project/CityFlow",
            "license": "Apache-2.0-code-example-data-terms-to-verify",
            "local_roots": ["works/CityFlow"],
            "planned_conversion": "native_signal_control_probe",
            "default_disposition": "held_license_or_terms",
            "required_native_gates": [
                "network_and_flow_lock",
                "native_signal_phase_effect",
                "deterministic_replay",
                "queue_headroom",
            ],
        },
        {
            "source_id": "flatland",
            "title": "Flatland railway rescheduling",
            "domain": "rail",
            "backend_kind": "flatland",
            "source_url": "https://github.com/flatland-association/flatland-rl",
            "license": "MIT-code-generated-environment",
            "local_roots": ["works/flatland-rl"],
            "planned_conversion": "method_transfer_only_until_rail_core",
            "default_disposition": "method_transfer_only",
            "required_native_gates": [
                "paired_real_rail_graph",
                "timetable_lock",
                "native_route_replan_effect",
                "terminal_parity",
            ],
        },
        {
            "source_id": "batsim",
            "title": "Batsim scheduling simulator",
            "domain": "datacenter",
            "backend_kind": "batsim",
            "source_url": "https://github.com/oar-team/batsim",
            "license": "CeCILL-C-code-workload-terms-to-verify",
            "local_roots": ["works/Batsim", "works/batsim"],
            "planned_conversion": "native_scheduler_conversion",
            "default_disposition": "held_missing_assets",
            "required_native_gates": [
                "workload_license_lock",
                "native_scheduler_effect",
                "deterministic_replay",
                "counterfactual_replay",
            ],
        },
        {
            "source_id": "oragentbench",
            "title": "ORAgentBench",
            "domain": "cross_domain",
            "backend_kind": "native_backend_per_task",
            "source_url": "https://oragentbench.github.io/",
            "license": "task-specific-terms-to-verify",
            "local_roots": ["works/ORAgentBench", "works/oragentbench"],
            "planned_conversion": "validator_method_transfer_only",
            "default_disposition": "method_transfer_only",
            "required_native_gates": [
                "source_locked_native_environment",
                "backend_native_action_effect",
                "evidence_linkage",
            ],
        },
        {
            "source_id": "simbench_official",
            "title": "SimBench",
            "domain": "power_grid",
            "backend_kind": "cigre_distribution",
            "source_url": "https://github.com/e2nIEE/simbench",
            "license": "ODbL-1.0-DbCL-1.0",
            "local_roots": ["works/SimBench", "works/simbench"],
            "planned_conversion": "native_prefilter_existing_wave",
            "default_disposition": "candidate_prefilter",
            "required_native_gates": [
                "profile_window_consumed",
                "native_action_effect",
                "positive_headroom",
                "full_protocol21_replay",
            ],
        },
    )


def _inventory_rows(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = inventory.get("sources")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("inventory.sources must be a list of objects")
    result = {}
    for row in rows:
        source_id = str(row.get("source_id") or "").strip()
        if not source_id or source_id in result:
            raise ValueError(f"inventory source_id is missing or duplicated: {source_id}")
        result[source_id] = dict(row)
    return result


def _terminal_static(
    descriptor: Mapping[str, Any],
    *,
    inventory_row: Mapping[str, Any] | None,
    repo_root: Path,
) -> dict[str, Any]:
    source_id = str(descriptor["source_id"])
    row = dict(inventory_row or {})
    roots = list(descriptor.get("local_roots") or [])
    if row.get("source_root") and str(row["source_root"]) not in roots:
        roots.insert(0, str(row["source_root"]))
    root_info = _root_info(roots, repo_root=repo_root)
    declared = str(row.get("disposition") or descriptor.get("default_disposition") or "held")
    blockers: list[str] = []
    disposition = declared
    if not root_info["present_root"]:
        blockers.append("source_assets_missing")
        if declared not in {"method_transfer_only", "held_license_or_terms"}:
            disposition = "held_missing_assets"
    if declared == "held_source_tree_dirty":
        blockers.append("source_tree_dirty")
        disposition = "held_source_tree_dirty"
    license_info = row.get("license") if isinstance(row.get("license"), dict) else {}
    if license_info and license_info.get("evidence_bound") is False:
        blockers.append("license_evidence_unbound")
    if license_info and license_info.get("redistribution_cleared") is False:
        blockers.append("redistribution_terms_unresolved")
    if declared == "held_license_or_terms":
        blockers.append("license_or_terms_review")
        disposition = "held_license_or_terms"
    if declared == "method_transfer_only":
        blockers.append("native_backend_not_in_release_domain")
        disposition = "method_transfer_only"
    if declared == "candidate_prefilter":
        blockers.append("native_prefilter_and_full_protocol21_pending")
        disposition = "candidate_prefilter"
    blockers = sorted(set(blockers))
    return {
        "source_id": source_id,
        "title": descriptor.get("title"),
        "domain": descriptor.get("domain") or row.get("domain"),
        "backend_kind": descriptor.get("backend_kind") or row.get("backend_kind"),
        "stage": "inventory",
        "work_state": "terminal",
        "disposition": disposition,
        "attempted": False,
        "materialized_rows": 0,
        "native_replay_executed": False,
        "native_passed_rows": 0,
        "full_protocol21_ready_rows": 0,
        "blockers": blockers,
        "source": {
            "url": row.get("source_url") or descriptor.get("source_url"),
            "license": row.get("license") or descriptor.get("license"),
            "local_assets": root_info,
            "source_root": row.get("source_root"),
            "source_unit_kind": row.get("source_unit_kind"),
        },
        "conversion": {
            "recipe": descriptor.get("planned_conversion"),
            "required_native_gates": list(descriptor.get("required_native_gates") or []),
            "candidate_only": True,
            "raw_data_copied_or_redistributed": False,
        },
        "core_admission_claimed": False,
    }


def _existing_family_rows(
    existing_wave: Mapping[str, Any], *, current_tree_sha256: str | None
) -> list[dict[str, Any]]:
    rows = existing_wave.get("families")
    if not isinstance(rows, list):
        return []
    report_tree = (existing_wave.get("implementation_identity") or {}).get(
        "implementation_tree_sha256"
    )
    stale = bool(current_tree_sha256 and report_tree and report_tree != current_tree_sha256)
    result: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict) or not raw.get("source_id"):
            continue
        row = {
            "source_id": str(raw["source_id"]),
            "title": str(raw["source_id"]),
            "domain": (raw.get("source") or {}).get("domain"),
            "backend_kind": (raw.get("runtime") or {}).get("backend"),
            "stage": raw.get("stage") or "unknown",
            "work_state": "terminal",
            "disposition": "held_stale_evidence" if stale else raw.get("disposition"),
            "attempted": bool(raw.get("attempted")),
            "materialized_rows": int(raw.get("materialized_rows") or 0),
            "native_replay_executed": bool(raw.get("native_replay_executed")),
            "native_passed_rows": int(raw.get("native_passed_rows") or 0),
            "full_protocol21_ready_rows": 0,
            "blockers": sorted(
                set(
                    [str(item) for item in raw.get("blockers") or []]
                    + (["implementation_tree_drift"] if stale else [])
                )
            ),
            "source": raw.get("source") or {},
            "runtime": raw.get("runtime") or {},
            "artifacts": raw.get("artifacts") or {},
            "implementation_tree_sha256": raw.get("implementation_tree_sha256")
            or report_tree,
            "core_admission_claimed": False,
            "candidate_only": True,
        }
        result.append(row)
    return result


def build_wave_ledger(
    *,
    inventory: Mapping[str, Any],
    existing_wave: Mapping[str, Any] | None = None,
    repo_root: Path = REPO_ROOT,
    current_tree_sha256: str | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    """Build one terminal row for every inventory/catalog/family input."""

    inventory_by_id = _inventory_rows(inventory)
    descriptors = list(latest_benchmark_catalog())
    descriptor_ids = {str(item["source_id"]) for item in descriptors}
    descriptors.extend(
        {
            "source_id": source_id,
            "title": source_id,
            "domain": str(row.get("domain") or "unknown"),
            "backend_kind": row.get("backend_kind"),
            "source_url": row.get("source_url"),
            "license": row.get("license"),
            "local_roots": [str(row["source_root"])] if row.get("source_root") else [],
            "planned_conversion": "inventory_native_prefilter",
            "default_disposition": str(row.get("disposition") or "held"),
            "required_native_gates": [],
        }
        for source_id, row in sorted(inventory_by_id.items())
        if source_id not in descriptor_ids
    )
    # Inventory is authoritative for already-known source identities.  A
    # descriptor is still retained for newly requested benchmarks.
    def make_row(descriptor: dict[str, Any]) -> dict[str, Any]:
        return _terminal_static(
            descriptor,
            inventory_row=inventory_by_id.get(str(descriptor["source_id"])),
            repo_root=repo_root,
        )

    max_workers = max(1, min(int(workers), len(descriptors) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        static_rows = list(pool.map(make_row, descriptors))
    static_rows.sort(key=lambda row: str(row["source_id"]))

    existing_rows = _existing_family_rows(
        existing_wave or {}, current_tree_sha256=current_tree_sha256
    )
    existing_rows.sort(key=lambda row: str(row["source_id"]))
    # A native prefilter family is the more specific terminal for the same
    # source identity.  Replace its static catalog placeholder instead of
    # emitting two competing terminal rows.
    existing_ids = {str(row["source_id"]) for row in existing_rows}
    rows = [row for row in static_rows if str(row["source_id"]) not in existing_ids]
    rows.extend(existing_rows)
    source_ids = [str(row["source_id"]) for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("candidate wave has duplicate terminal source identities")
    counts: dict[str, int] = {}
    for row in rows:
        disposition = str(row.get("disposition") or "unknown")
        counts[disposition] = counts.get(disposition, 0) + 1
    return {
        "schema_version": "latest-benchmark-candidate-wave-v1",
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete_candidate_only",
        "scope": "latest_external_benchmarks_and_existing_candidate_prefilters",
        "candidate_only": True,
        "core_admission_claimed": False,
        "implementation_binding": {"implementation_tree_sha256": current_tree_sha256},
        "policy": {
            "raw_data_copied_or_redistributed": False,
            "method_transfer_is_not_source_consumption": True,
            "declared_perturbation_is_not_source_independence": True,
            "full_protocol21_required_before_core": True,
            "model_outcomes_used_for_filtering": False,
        },
        "counts": {
            "inventory_inputs": len(inventory_by_id),
            "catalog_inputs": len(descriptors),
            "existing_candidate_family_inputs": len(existing_rows),
            "terminal_rows": len(rows),
            "dispositions": dict(sorted(counts.items())),
            "native_replay_rows": sum(bool(row.get("native_replay_executed")) for row in rows),
            "current_hash_native_replay_rows": sum(
                bool(row.get("native_replay_executed"))
                and row.get("disposition") != "held_stale_evidence"
                for row in rows
            ),
            "stale_evidence_rows": sum(
                row.get("disposition") == "held_stale_evidence" for row in rows
            ),
            "full_protocol21_ready_rows": 0,
        },
        "sources": rows,
    }


def run_wave(
    *,
    inventory_path: Path = DEFAULT_INVENTORY,
    existing_wave_path: Path = DEFAULT_EXISTING_WAVE,
    report_path: Path = DEFAULT_REPORT,
    report_root: Path = DEFAULT_REPORT_ROOT,
    repo_root: Path = REPO_ROOT,
    current_tree_sha256: str | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    for path in (report_path, report_root):
        resolved = path.resolve()
        if not resolved.is_relative_to((repo_root / "reports").resolve()):
            raise ValueError(f"candidate output outside reports/: {path}")
    inventory = _load(inventory_path.resolve())
    existing = _load(existing_wave_path.resolve()) if existing_wave_path.is_file() else {}
    if current_tree_sha256 is None:
        # Avoid importing the whole runner here; callers that already hold an
        # implementation identity can pass it explicitly for hash binding.
        current_tree_sha256 = str(
            (existing.get("implementation_identity") or {}).get("implementation_tree_sha256")
            or "unknown"
        )
    ledger = build_wave_ledger(
        inventory=inventory,
        existing_wave=existing,
        repo_root=repo_root,
        current_tree_sha256=current_tree_sha256,
        workers=workers,
    )
    ledger["input_bindings"] = {
        "inventory": {"path": _repo_relative(inventory_path), "sha256": _sha256(inventory_path)},
        "existing_wave": {
            "path": _repo_relative(existing_wave_path),
            "sha256": _sha256(existing_wave_path),
        },
    }
    report_root.mkdir(parents=True, exist_ok=True)
    _write(report_root / "source_catalog.json", {"sources": list(latest_benchmark_catalog())})
    for row in ledger["sources"]:
        _write(report_root / "terminals" / f"{row['source_id']}.json", row)
    _write(report_path, ledger)
    return ledger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--existing-wave", type=Path, default=DEFAULT_EXISTING_WAVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--implementation-tree-sha256")
    args = parser.parse_args()
    report = run_wave(
        inventory_path=args.inventory.resolve(),
        existing_wave_path=args.existing_wave.resolve(),
        report_path=args.output.resolve(),
        report_root=args.report_root.resolve(),
        current_tree_sha256=args.implementation_tree_sha256,
        workers=args.workers,
    )
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
