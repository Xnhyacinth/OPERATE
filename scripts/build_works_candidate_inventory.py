#!/usr/bin/env python3
"""Inventory downloaded benchmark assets against the active OPERATE release.

The command hashes representative local assets and binds git/runtime/license
metadata.  Scientific candidacy, environment closure, and public redistribution
are reported separately: a repairable runtime or packaging issue never becomes
an intrinsic task rejection.  The command never downloads data, executes a
simulator, mutates Core, or claims candidate admission.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.opendss_candidate_inventory import (  # noqa: E402
    opendss_entrypoint_units,
)

UNIT_COVERAGE_REQUIRED = {
    "rts_gmlc",
    "pglib_opf",
    "pglib_uc",
    "sumo365_ingolstadt",
    "resco",
    "nrel_microgrid",
    "citylearn",
    "dynaschedbench",
    "jsplib",
    "realm_j2",
    "vrplib",
    "m5_forecasting",
    "pyvrp_instances",
    "opendss_ieee_testcases",
    "grid2op_cache",
}
DEFAULT_LOCKED_CORE = (
    REPO_ROOT / "release" / "operate_v0_58_0" / "protocol21_source_suite.json"
)
DEFAULT_NEAR_CORE = REPO_ROOT / "scenarios/candidates/near_core_registry.json"
DEFAULT_OUTPUT = REPO_ROOT / ".hl/artifacts/operate_v058_candidate_inventory.json"


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    path: str
    domain: str
    backend: str | None
    url: str
    license_id: str
    license_evidence: str | None
    redistribution_cleared: bool
    unit_glob: str
    asset_globs: tuple[str, ...]
    runtime: str
    overlap_tokens: tuple[str, ...]
    method_transfer_only: bool = False
    unit_kind: str = "source_unit"
    unit_count_strategy: str = "files"
    support_only: bool = False


SOURCE_SPECS: tuple[SourceSpec, ...] = (
    SourceSpec(
        "rts_gmlc",
        "works/RTS-GMLC",
        "power_grid",
        "pglib_uc_synthetic",
        "https://github.com/GridMod/RTS-GMLC",
        "NREL-RTS-data-use-notice",
        "works/RTS-GMLC/README.md",
        True,
        "RTS_Data/FormattedData/pandapower/pandapower_net.json",
        (
            "RTS_Data/FormattedData/pandapower/pandapower_net.json",
            "RTS_Data/timeseries_data_files/Load/REAL_TIME_regional_Load.csv",
            "README.md",
        ),
        "pandapower",
        ("rts_gmlc", "real_forecast"),
        unit_kind="coherent_grid_and_timeseries_graph",
    ),
    SourceSpec(
        "pglib_opf",
        "works/PGLib-OPF",
        "power_grid",
        "pandapower_acopf",
        "https://github.com/power-grid-lib/pglib-opf",
        "CC-BY-4.0-data+MIT-software",
        "works/PGLib-OPF/LICENSE",
        True,
        "pglib_opf_case*.m",
        ("pglib_opf_case*.m", "LICENSE"),
        "pandapower",
        ("pglib_opf", "opf_case"),
        unit_kind="original_opf_case",
    ),
    SourceSpec(
        "pglib_uc",
        "works/pglib-uc",
        "power_grid",
        "pglib_uc_synthetic",
        "https://github.com/power-grid-lib/pglib-uc",
        "CC-BY-4.0-data+MIT-software",
        "works/pglib-uc/LICENSE",
        True,
        "*/*.json",
        ("rts_gmlc/*.json", "ferc/*.json", "ca/*.json", "LICENSE"),
        "python",
        ("pglib_uc", "uc_source_window"),
        unit_kind="unit_commitment_instance",
    ),
    SourceSpec(
        "grid2op_cache",
        "works/Grid2Op_cache/rte_case14_realistic",
        "power_grid",
        None,
        "https://github.com/Grid2op/grid2op",
        "environment-data-license-unresolved",
        None,
        False,
        "config.py",
        ("config.py", "grid.json", "case14_realistic.json", "prods_charac.csv"),
        "grid2op",
        ("grid2op",),
        unit_kind="grid2op_environment",
    ),
    SourceSpec(
        "sumo365_ingolstadt",
        "works/sumo_ingolstadt",
        "traffic",
        "sumo",
        "https://github.com/TUM-VT/sumo_ingolstadt",
        "Apache-2.0",
        "works/sumo_ingolstadt/LICENSE.md",
        True,
        "simulation/Ingolstadt SUMO 365/*.sumocfg",
        (
            "simulation/Ingolstadt SUMO 365/*.sumocfg",
            "simulation/Ingolstadt SUMO 365/ingolstadt_net.net.xml",
            "LICENSE.md",
        ),
        "sumo",
        ("sumo_ingolstadt_365", "traffic_live"),
        unit_kind="service_date",
    ),
    SourceSpec(
        "resco",
        "works/RESCO",
        "traffic",
        "sumo",
        "https://github.com/Pi-Star-Lab/RESCO",
        "GPL-3.0-repository",
        "works/RESCO/LICENSE",
        True,
        "resco_benchmark/environments/*/*.sumocfg",
        (
            "resco_benchmark/environments/*/*.sumocfg",
            "resco_benchmark/environments/*/*.net.xml",
            "LICENSE",
        ),
        "sumo",
        ("resco", "cologne", "ingolstadt"),
        unit_kind="traffic_network",
    ),
    SourceSpec(
        "nrel_microgrid",
        "works/nrel-microgrid",
        "microgrid",
        "pandapower_lv",
        "https://data.nrel.gov/",
        "mixed-NREL-OEDI-OpenEI-terms-review",
        "works/nrel-microgrid/sources/source_lock_vnext.json",
        False,
        "*.npz",
        ("*.npz", "*.provenance.json", "sources/source_lock_vnext.json"),
        "pandapower",
        ("nrel", "microgrid_lv", "pymgrid"),
        unit_kind="site_profile",
    ),
    SourceSpec(
        "citylearn",
        "works/CityLearn",
        "building_energy",
        "citylearn",
        "https://github.com/intelligent-environments-lab/CityLearn",
        "MIT-software;dataset-terms-unresolved",
        "works/CityLearn/LICENSE",
        False,
        "data/datasets/*/schema.json",
        ("data/datasets/*/schema.json", "LICENSE"),
        "citylearn",
        ("citylearn",),
        unit_kind="citylearn_dataset",
    ),
    SourceSpec(
        "building_data_genome_2",
        "works/building-data-genome-project-2",
        "building_energy",
        None,
        "https://github.com/buds-lab/building-data-genome-project-2",
        "CC-BY-SA-4.0-not-locally-locked",
        None,
        False,
        "**/*.csv",
        (),
        "citylearn",
        ("building_data_genome", "bdg2"),
        unit_kind="building_meter_source",
    ),
    SourceSpec(
        "cityflow_examples",
        "works/CityFlow",
        "traffic",
        None,
        "https://github.com/cityflow-project/CityFlow",
        "Apache-2.0-software;example-data-terms-unresolved",
        "works/CityFlow/LICENSE.txt",
        False,
        "examples/config.json",
        (
            "examples/config.json",
            "examples/roadnet.json",
            "examples/flow.json",
            "LICENSE.txt",
        ),
        "cityflow",
        ("cityflow",),
        method_transfer_only=True,
        unit_kind="example_simulation_graph",
    ),
    SourceSpec(
        "flatland",
        "works/flatland-rl",
        "rail",
        None,
        "https://github.com/flatland-association/flatland-rl",
        "MIT-software;generated-environments",
        "works/flatland-rl/LICENSE",
        True,
        "env_data/railway/*.pkl",
        ("env_data/railway/*.pkl", "LICENSE"),
        "flatland-rl",
        ("flatland",),
        method_transfer_only=True,
        unit_kind="generated_rail_environment",
    ),
    SourceSpec(
        "dynaschedbench",
        "works/DynaSchedBench",
        "logistics",
        "dynasched_flexible_job_shop",
        "https://github.com/dsbx7/DynaSchedBench",
        "Apache-2.0",
        "works/DynaSchedBench/LICENSE",
        True,
        "data/**/input_model.json",
        ("data/JMS-Bench/*.jsonl", "data/MK-Bench/*.jsonl", "LICENSE"),
        "dsbx",
        ("dynasched",),
        unit_kind="generated_dynamic_scheduling_instance",
    ),
    SourceSpec(
        "jsplib",
        "works/JSPLIB-Instances",
        "logistics",
        "jsplib_job_shop",
        "https://github.com/tamy0612/JSPLIB",
        "upstream-instance-terms-review",
        "works/JSPLIB-Instances/README.md",
        False,
        "instances/*",
        ("instances/*", "README.md"),
        "python",
        ("jsplib_job_shop", "jobshop_"),
        unit_kind="job_shop_instance",
    ),
    SourceSpec(
        "realm_j2",
        "works/REALM-Bench-direct-pilot",
        "logistics",
        "jsplib_job_shop",
        "https://github.com/genglongling/REALM-Bench",
        "CC-BY-4.0-selected-J2-data",
        "works/REALM-Bench-direct-pilot/README.md",
        True,
        "datasets/clean/JSSP/J2.json",
        ("datasets/clean/JSSP/J2.json", "README.md"),
        "python",
        ("realm_j2", "realm_"),
        unit_kind="disrupted_job_shop_dataset",
        unit_count_strategy="json_instances",
    ),
    SourceSpec(
        "vrplib",
        "works/VRPLIB",
        "logistics",
        "pyvrp_cvrp",
        "https://github.com/PyVRP/VRPLIB",
        "MIT-software;instance-provenance-per-file",
        "works/VRPLIB/LICENSE.md",
        True,
        "tests/data/**/*",
        ("tests/data/**/*.vrp", "tests/data/**/*.vrptw", "LICENSE.md"),
        "pyvrp",
        ("pyvrp_cvrp", "cvrp_"),
        unit_kind="routing_instance",
        unit_count_strategy="vrplib_routes",
    ),
    SourceSpec(
        "m5_forecasting",
        "works/M5",
        "logistics",
        "orgym_inventory",
        "https://www.kaggle.com/competitions/m5-forecasting-accuracy",
        "M5-competition-terms-private-hash-locked",
        "works/M5/source_lock.json",
        False,
        "sales_train_evaluation.csv",
        (
            "calendar.csv",
            "sales_train_evaluation.csv",
            "sell_prices.csv",
            "source_lock.json",
        ),
        "or-gym",
        ("orgym_inventory", "m5"),
        unit_kind="item_store_timeseries",
        unit_count_strategy="csv_rows",
    ),
    SourceSpec(
        "pyvrp_instances",
        "works/PyVRP-Instances",
        "logistics",
        "pyvrp_cvrp",
        "https://github.com/PyVRP/Instances",
        "MIT-repository;instance-provenance-per-family",
        "works/PyVRP-Instances/LICENSE",
        False,
        "**/*.vrp",
        ("**/*.vrp", "LICENSE"),
        "pyvrp",
        ("pyvrp_cvrp", "pyvrp_instances"),
        unit_kind="routing_instance",
    ),
    SourceSpec(
        "opendss_ieee_testcases",
        "works/OpenDSS-IEEE13",
        "power_grid",
        "opendss_fresh_feeders",
        "https://github.com/dss-extensions/electricdss-tst",
        "BSD-3-Clause",
        "works/OpenDSS-IEEE13/License.txt",
        True,
        "**/*.dss",
        ("**/*.dss", "License.txt"),
        "dss-python",
        ("opendss_fresh", "opendss", "ieee13"),
        unit_kind="coherent_feeder_entrypoint",
        unit_count_strategy="opendss_entrypoints",
    ),
    SourceSpec(
        "orgym_runtime",
        "works/OR-Gym",
        "logistics",
        "orgym_inventory",
        "https://github.com/hubbs5/or-gym",
        "MIT",
        "works/OR-Gym/LICENSE",
        True,
        "or_gym/__init__.py",
        ("or_gym/__init__.py", "LICENSE"),
        "or-gym",
        ("orgym_inventory",),
        unit_kind="runtime_support",
        support_only=True,
    ),
    SourceSpec(
        "alibaba_clusterdata",
        "works/clusterdata",
        "datacenter",
        "alibaba_trace_sim",
        "https://github.com/alibaba/clusterdata",
        "v2020-CC-BY-4.0;later-families-research-terms-only",
        "works/clusterdata/cluster-trace-gpu-v2020/LICENSE",
        False,
        "cluster-trace-*",
        (
            "cluster-trace-gpu-v2025/*.csv",
            "cluster-trace-v2026-GenAI/*.csv",
            "cluster-trace-v2026-spot-gpu/*.csv",
            "cluster-trace-gpu-v2020/LICENSE",
        ),
        "python",
        ("clusterdata", "alibaba"),
        unit_kind="trace_release_family",
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _git_metadata(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {"is_git_checkout": False, "commit": None, "dirty": None, "remote": None}

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    return {
        "is_git_checkout": True,
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
        "remote": run("remote", "get-url", "origin"),
    }


def _runtime_binding(name: str) -> dict[str, Any]:
    if name == "python":
        return {
            "runtime": name,
            "available": True,
            "version": platform.python_version(),
        }
    if name == "sumo":
        executable = shutil.which("sumo")
        return {"runtime": name, "available": executable is not None, "version": None}
    try:
        installed = version(name)
    except PackageNotFoundError:
        installed = None
    return {"runtime": name, "available": installed is not None, "version": installed}


def _glob_entries(root: Path, pattern: str) -> list[Path]:
    return sorted(
        path for path in root.glob(pattern) if path.is_file() or path.is_dir()
    )


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _canonical_source_units(
    spec: SourceSpec,
    *,
    root: Path,
    units: list[Path],
) -> list[str]:
    """Mirror the exact ``source_unit`` identity emitted by each refiner."""
    source_id = spec.source_id
    if source_id == "rts_gmlc":
        values = ["coherent_grid_forecast_reserve_graph"]
    elif source_id == "pglib_opf":
        values = [path.stem for path in units]
    elif source_id == "pglib_uc":
        values = [path.relative_to(root).as_posix() for path in units]
    elif source_id == "grid2op_cache":
        values = [root.name]
    elif source_id == "sumo365_ingolstadt":
        values = [path.stem for path in units]
    elif source_id == "resco":
        values = [path.parent.name for path in units]
    elif source_id == "nrel_microgrid":
        values = [path.stem for path in units]
    elif source_id == "citylearn":
        values = [path.parent.name for path in units]
    elif source_id == "dynaschedbench":
        data_root = root / "data"
        values = [path.parent.relative_to(data_root).as_posix() for path in units]
    elif source_id == "jsplib":
        metadata = json.loads((root / "instances.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, list):
            raise ValueError("JSPLIB instances.json must contain a list")
        values = [
            _repo_relative(root / str(row.get("path") or ""))
            for row in metadata
            if isinstance(row, dict)
        ]
    elif source_id == "realm_j2":
        if len(units) != 1:
            raise ValueError("REALM J2 inventory requires exactly one container")
        payload = _read_json(units[0])
        instances = payload.get("instances")
        if not isinstance(instances, list):
            raise ValueError("REALM J2 instances must contain a list")
        prefix = _repo_relative(units[0])
        values = [
            f"{prefix}#{str(row.get('instance_id') or '')}"
            for row in instances
            if isinstance(row, dict)
        ]
    elif source_id == "vrplib":
        values = [
            _repo_relative(path)
            for path in units
            if path.suffix.lower() in {".vrp", ".vrptw"}
            or (path.suffix.lower() == ".txt" and "Vrp-Set-Solomon" in path.parts)
        ]
    elif source_id == "m5_forecasting":
        if len(units) != 1:
            raise ValueError("M5 inventory requires exactly one sales table")
        with units[0].open(newline="", encoding="utf-8") as stream:
            values = [
                f"sales_train_evaluation.csv#{str(row.get('id') or '')}"
                for row in csv.DictReader(stream)
            ]
    elif source_id == "pyvrp_instances":
        values = [_repo_relative(path) for path in units]
    elif source_id == "opendss_ieee_testcases":
        values = opendss_entrypoint_units(root)
    else:
        raise ValueError(f"canonical source-unit policy missing for {source_id}")
    return sorted(set(values))


def _source_unit_manifest_sha256(source_units: list[str]) -> str:
    return hashlib.sha256("\n".join(source_units).encode()).hexdigest()


def _source_unit_count(spec: SourceSpec, units: list[Path]) -> int:
    if spec.unit_count_strategy == "files":
        return len(units)
    if spec.unit_count_strategy == "csv_rows":
        if len(units) != 1:
            return 0
        with units[0].open("r", encoding="utf-8") as stream:
            return max(sum(1 for _ in stream) - 1, 0)
    if spec.unit_count_strategy == "json_instances":
        if len(units) != 1:
            return 0
        payload = _read_json(units[0])
        instances = payload.get("instances")
        return len(instances) if isinstance(instances, list) else 0
    if spec.unit_count_strategy == "vrplib_routes":
        return sum(
            path.suffix.lower() in {".vrp", ".vrptw"}
            or (path.suffix.lower() == ".txt" and "Vrp-Set-Solomon" in path.parts)
            for path in units
        )
    if spec.unit_count_strategy == "opendss_entrypoints":
        return len(opendss_entrypoint_units(REPO_ROOT / spec.path))
    raise ValueError(
        f"unsupported source unit count strategy: {spec.unit_count_strategy}"
    )


def _representative_assets(
    root: Path, patterns: tuple[str, ...]
) -> list[dict[str, Any]]:
    selected: list[Path] = []
    for pattern in patterns:
        selected.extend(path for path in _glob_entries(root, pattern) if path.is_file())
    rows = []
    for path in sorted(set(selected))[:12]:
        rows.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return rows


def _matches(row: dict[str, Any], tokens: tuple[str, ...]) -> bool:
    serialized = json.dumps(row, sort_keys=True).lower()
    return any(token.lower() in serialized for token in tokens)


def _source_row(
    spec: SourceSpec,
    *,
    core_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    root = REPO_ROOT / spec.path
    present = root.is_dir()
    units = _glob_entries(root, spec.unit_glob) if present else []
    exact_unit_coverage = spec.source_id in UNIT_COVERAGE_REQUIRED
    source_units: list[str] | None = None
    source_unit_parse_error: dict[str, str] | None = None
    try:
        if exact_unit_coverage and present:
            source_units = _canonical_source_units(spec, root=root, units=units)
            source_unit_count = len(source_units)
        else:
            source_unit_count = _source_unit_count(spec, units)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        source_unit_count = 0
        source_units = [] if exact_unit_coverage else None
        source_unit_parse_error = {
            "error_type": type(exc).__name__,
            "detail": str(exc)[:1000],
        }
    assets = _representative_assets(root, spec.asset_globs) if present else []
    license_path = REPO_ROOT / spec.license_evidence if spec.license_evidence else None
    license_bound = bool(license_path and license_path.is_file())
    git = (
        _git_metadata(root)
        if present
        else {
            "is_git_checkout": False,
            "commit": None,
            "dirty": None,
            "remote": None,
        }
    )
    runtime = _runtime_binding(spec.runtime)
    related_core_rows = sum(_matches(row, spec.overlap_tokens) for row in core_rows)
    environment_reasons: list[str] = []
    if source_unit_parse_error is not None:
        environment_reasons.append("source_inventory_parse_failed")
    elif not present or not source_unit_count or not assets:
        environment_reasons.append("source_assets_missing")
    if not runtime["available"]:
        environment_reasons.append("runtime_unavailable")
    reproducibility_warnings = ["source_tree_dirty"] if git["dirty"] is True else []
    scientific_disposition = (
        "support_only"
        if spec.support_only
        else "method_transfer_only"
        if spec.method_transfer_only
        else "candidate_prefilter"
    )
    execution_state = "ready" if not environment_reasons else "held_repair"
    return {
        "source_id": spec.source_id,
        "domain": spec.domain,
        "backend_kind": spec.backend,
        "source_root": spec.path,
        "source_url": spec.url,
        "present": present,
        "source_unit_kind": spec.unit_kind,
        "source_unit_count": source_unit_count,
        "source_unit_count_strategy": spec.unit_count_strategy,
        **(
            {
                "source_units": source_units or [],
                "source_unit_manifest_sha256": _source_unit_manifest_sha256(
                    source_units or []
                ),
            }
            if exact_unit_coverage
            else {}
        ),
        "source_unit_parse_error": source_unit_parse_error,
        "unit_coverage_required": exact_unit_coverage,
        "source_lock": {
            "git": git,
            "representative_assets": assets,
            "hash_scope": "representative_static_preflight_only",
        },
        "license": {
            "id": spec.license_id,
            "evidence_path": spec.license_evidence,
            "evidence_bound": license_bound,
            "redistribution_cleared": spec.redistribution_cleared,
        },
        "runtime_binding": runtime,
        "related_core_rows": related_core_rows,
        # A source-family/token match is not an exact candidate identity.  Exact
        # duplicate checks happen only after a converter emits scenario id,
        # signature, and canonical effective-source identity.
        "exact_identity_skip_required": False,
        "scientific_disposition": scientific_disposition,
        "disposition": scientific_disposition,
        "execution_state": execution_state,
        "environment_closure": {
            "closed": not environment_reasons,
            "reason_codes": sorted(environment_reasons),
        },
        "reproducibility_warnings": reproducibility_warnings,
        "distribution": {
            "public_redistribution_cleared": bool(
                license_bound and spec.redistribution_cleared
            ),
            "private_hash_locked_evaluation_allowed": bool(present and assets),
            "reason_codes": (
                []
                if license_bound and spec.redistribution_cleared
                else ["public_redistribution_terms_unresolved"]
            ),
        },
        "candidate_only": True,
        "release_admission": False,
    }


def _candidate_domain(row: dict[str, Any]) -> str:
    domain = str(row.get("domain") or "").strip()
    if domain:
        return domain
    scenario_id = str(row.get("scenario_id") or "").strip()
    parts = scenario_id.split("/")
    if len(parts) < 2 or any(not part for part in parts):
        raise ValueError("candidate without domain requires a canonical scenario_id")
    return parts[0]


def _candidate_records(
    *,
    release_held: list[dict[str, Any]],
    release_abandoned: list[dict[str, Any]],
    near_core: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in release_held:
        records.append(
            {
                "origin": "active_source_suite_held_candidates",
                "scenario_id": str(row.get("scenario_id") or ""),
                "domain": _candidate_domain(row),
                "backend_kind": str(row.get("backend_kind") or ""),
                "inventory_state": str(row.get("disposition") or "held_repair"),
                "reason_codes": sorted(map(str, row.get("reason_codes") or [])),
                "terminal": False,
                "final_disposition": None,
                "automatic_admission": False,
            }
        )
    for row in release_abandoned:
        disposition = str(row.get("disposition") or "abandoned_terminal")
        records.append(
            {
                "origin": "active_source_suite_abandoned_candidates",
                "scenario_id": str(row.get("scenario_id") or ""),
                "domain": _candidate_domain(row),
                "backend_kind": str(row.get("backend_kind") or ""),
                "inventory_state": disposition,
                "reason_codes": sorted(map(str, row.get("reason_codes") or [])),
                "terminal": True,
                "final_disposition": disposition,
                "automatic_admission": False,
            }
        )
    for row in near_core:
        recovery = str(row.get("recovery_class") or "review_required")
        negative_evidence = row.get("later_negative_evidence")
        terminal = (
            recovery == "redesign_required_after_negative_evidence"
            and isinstance(negative_evidence, dict)
            and bool(negative_evidence.get("code"))
        )
        state = (
            "abandoned_terminal"
            if terminal
            else
            "ready_for_current_replay"
            if recovery == "current_full_replay_required"
            else "redesign_required"
        )
        record = {
            "origin": "near_core_registry",
            "scenario_id": str(row.get("scenario_id") or ""),
            "domain": _candidate_domain(row),
            "backend_kind": str(row.get("backend_kind") or ""),
            "inventory_state": state,
            "recovery_class": recovery,
            "terminal": terminal,
            "final_disposition": "abandoned_terminal" if terminal else None,
            "automatic_admission": False,
        }
        for field in (
            "source_asset_path",
            "source_asset_sha256",
            "source_denominator_key",
        ):
            if row.get(field):
                record[field] = str(row[field])
        if terminal:
            record["reason_codes"] = [str(negative_evidence["code"])]
            record["evidence"] = {
                "later_negative_evidence": deepcopy(negative_evidence),
                "historical_path": str(row.get("historical_path") or ""),
                "historical_file_sha256": str(
                    row.get("historical_file_sha256") or ""
                ),
            }
        records.append(record)
    return sorted(records, key=lambda row: (row["origin"], row["scenario_id"]))


def build_inventory(
    *,
    source_suite_path: Path = DEFAULT_LOCKED_CORE,
    near_core_path: Path = DEFAULT_NEAR_CORE,
    specs: tuple[SourceSpec, ...] = SOURCE_SPECS,
) -> dict[str, Any]:
    locked_core = _read_json(source_suite_path)
    near_core_payload = _read_json(near_core_path)
    core_rows = locked_core.get("scenarios")
    release_held = locked_core.get("held_candidates") or []
    release_abandoned = locked_core.get("abandoned_candidates") or []
    near_core = near_core_payload.get("candidates") or []
    if not isinstance(core_rows, list):
        raise ValueError("active source suite scenarios must be a list")
    if (
        not isinstance(release_held, list)
        or not isinstance(release_abandoned, list)
        or not isinstance(near_core, list)
    ):
        raise ValueError(
            "held, abandoned, and near-Core candidate inventories must be lists"
        )
    rows = [_source_row(spec, core_rows=core_rows) for spec in specs]
    dispositions = Counter(str(row["scientific_disposition"]) for row in rows)
    execution_states = Counter(str(row["execution_state"]) for row in rows)
    candidate_records = _candidate_records(
        release_held=[row for row in release_held if isinstance(row, dict)],
        release_abandoned=[row for row in release_abandoned if isinstance(row, dict)],
        near_core=[row for row in near_core if isinstance(row, dict)],
    )
    queue = [
        {
            "work_id": f"works-wave2:{row['source_id']}",
            "source_id": row["source_id"],
            "domain": row["domain"],
            "backend_kind": row["backend_kind"],
            "status": (
                "pending"
                if row["scientific_disposition"] == "candidate_prefilter"
                and row["execution_state"] == "ready"
                else "held_repair"
                if row["scientific_disposition"] == "candidate_prefilter"
                else "support_only"
                if row["scientific_disposition"] == "support_only"
                else "transfer_only"
            ),
            "scientific_disposition": row["scientific_disposition"],
            "execution_state": row["execution_state"],
            "execute": False,
            "next_gate": (
                "source_native_static_prefilter_then_runtime_smoke"
                if row["scientific_disposition"] == "candidate_prefilter"
                and row["execution_state"] == "ready"
                else "environment_closure"
                if row["scientific_disposition"] == "candidate_prefilter"
                else row["scientific_disposition"]
            ),
            "exact_identity_skip_required": row["exact_identity_skip_required"],
        }
        for row in rows
    ]
    return {
        "schema_version": "operate-candidate-inventory-v2",
        "status": "candidate_inventory_complete_non_admitting",
        "candidate_only": True,
        "release_admission": False,
        "locked_core": {
            "path": source_suite_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(source_suite_path),
            "implementation_tree_sha256": locked_core.get("implementation_tree_sha256"),
            "n_rows": len(core_rows),
            "n_held_candidates": len(release_held),
            "n_abandoned_candidates": len(release_abandoned),
            "mutated": False,
        },
        "near_core_registry": {
            "path": near_core_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(near_core_path),
            "n_rows": len(near_core),
        },
        "policy": {
            "downloads_performed": False,
            "simulator_calls_performed": False,
            "representative_hashes_are_not_full_source_consumption_proof": True,
            "source_unit_counts_are_inventory_units_not_candidate_counts": True,
            "environment_or_distribution_closure_never_rejects_task_quality": True,
            "family_overlap_is_not_exact_identity": True,
            "full_protocol21_and_hash_bound_fresh_union_required": True,
        },
        "sources": rows,
        "candidate_records": candidate_records,
        "queue": queue,
        "summary": {
            "n_source_families": len(rows),
            "n_present": sum(row["present"] for row in rows),
            "n_missing": sum(not row["present"] for row in rows),
            "n_source_units": sum(int(row["source_unit_count"]) for row in rows),
            "n_active_core_rows": len(core_rows),
            "n_release_held_candidates": len(release_held),
            "n_release_abandoned_candidates": len(release_abandoned),
            "n_near_core_candidates": len(near_core),
            "n_related_core_rows": sum(int(row["related_core_rows"]) for row in rows),
            "scientific_dispositions": dict(sorted(dispositions.items())),
            "execution_states": dict(sorted(execution_states.items())),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", type=Path, default=DEFAULT_LOCKED_CORE)
    parser.add_argument("--near-core", type=Path, default=DEFAULT_NEAR_CORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = build_inventory(
        source_suite_path=args.source_suite.resolve(),
        near_core_path=args.near_core.resolve(),
    )
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"dry_run": args.dry_run, **report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
