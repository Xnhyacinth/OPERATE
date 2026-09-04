#!/usr/bin/env python3
"""Build row-level structural candidate metadata from local source inventories.

This command performs no simulator replay and makes no quality or admission
claim.  It converts raw inventory observations into a bounded set of concrete
source identities that the v0.58 conversion-wave selector can schedule for
source-consumption and behavioral calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.source_asset_contract import canonical_physical_source_asset_key  # noqa: E402


DEFAULT_SOURCE_SUITE = ROOT / "release/operate_v0_61_0/protocol21_source_suite.json"
DEFAULT_INVENTORY = ROOT / ".hl/artifacts/operate_v058_candidate_inventory.json"
DEFAULT_REALM = (
    ROOT / "works/REALM-Bench-direct-pilot/datasets/clean/JSSP/J2.json"
)
DEFAULT_JSPLIB = ROOT / "works/JSPLIB-Instances/instances.json"
DEFAULT_PGLIB = ROOT / "works/PGLib-OPF"
DEFAULT_CITYLEARN = ROOT / "works/CityLearn/data/datasets"
DEFAULT_CITYLEARN_LOCK = (
    ROOT / ".hl/artifacts/citylearn_baeda_3dem_source_lock.json"
)
DEFAULT_DATACENTER = (
    ROOT / ".hl/artifacts/datacenter_spot_candidate_ledger_20260828.json"
)
DEFAULT_SIMBENCH = (
    ROOT
    / ".hl/artifacts/operate_v058_simbench_commercial_p19680_candidate_metadata.json"
)
DEFAULT_OUTPUT = ROOT / ".hl/artifacts/operate_v058_candidate_source_metadata.json"

PREFERRED_PGLIB_CASES = (
    "pglib_opf_case14_ieee",
    "pglib_opf_case57_ieee",
    "pglib_opf_case73_ieee_rts",
    "pglib_opf_case118_ieee",
    "pglib_opf_case300_ieee",
    "pglib_opf_case162_ieee_dtc",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _physical_asset_graph_key(
    backend_kind: str, assets: list[tuple[str, str]]
) -> str:
    return canonical_physical_source_asset_key(
        {
            "schema_version": "source_asset_graph_v1",
            "backend_kind": backend_kind,
            "required_source_assets": [
                {"declared_path": path, "sha256": digest}
                for path, digest in assets
            ],
        }
    )


def _identity_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _referenced_dataset_files(schema: Path) -> list[Path]:
    dataset_root = schema.parent
    referenced = {schema.resolve()}

    def visit(value: Any) -> None:
        if isinstance(value, str):
            candidate = dataset_root / value
            if candidate.is_file():
                referenced.add(candidate.resolve())
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(_load(schema))
    return sorted(referenced, key=lambda path: path.name)


def _environment_states(inventory: dict[str, Any]) -> dict[str, str]:
    return {
        str(row.get("source_id") or ""): str(
            row.get("execution_state") or "held_repair"
        )
        for row in inventory.get("sources") or []
        if isinstance(row, dict) and row.get("source_id")
    }


def _active_source_tokens(rows: list[Any]) -> set[str]:
    """Collect exact source identities without substring-based exclusions."""

    tokens: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            tokens.add(value)
            tokens.update(part for part in re.split(r"[:/|@]", value) if part)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(rows)
    return tokens


def _quality(environment: str, *, procedural: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "environment_status": environment,
        "behavioral_headroom": "unknown",
        "source_consumption": "unknown",
        "safety": "unknown",
        "evidence_stage": "structural_prefilter",
    }
    if procedural:
        result["procedural_stress_contract"] = "planned_unverified"
    return result


def _candidate(
    *,
    candidate_id: str,
    source_family: str,
    domain: str,
    backend_kind: str,
    source_denominator_key: str,
    physical_source_key: str,
    structural_axes: list[str],
    environment: str,
    source_metadata: dict[str, Any],
    priority: float,
    procedural: bool = False,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_family": source_family,
        "domain": domain,
        "backend_kind": backend_kind,
        "source_denominator_key": source_denominator_key,
        "physical_source_key": physical_source_key,
        "structural_axes": sorted(set(structural_axes)),
        "priority": priority,
        "quality": _quality(environment, procedural=procedural),
        "source_metadata": source_metadata,
        "candidate_only": True,
        "release_admission": False,
    }


def _realm_candidates(
    path: Path,
    *,
    active_tokens: set[str],
    environment: str,
    limit: int,
) -> list[dict[str, Any]]:
    payload = _load(path)
    rows = payload.get("instances") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("REALM J2 instances must be a list")
    source_sha = _sha256(path)
    physical_source_key = _physical_asset_graph_key(
        "jsplib_job_shop", [(path.name, source_sha)]
    )
    eligible: list[tuple[str, int, str, dict[str, Any]]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        instance_id = str(row.get("instance_id") or "")
        disruptions = row.get("disruptions") or []
        disruption = disruptions[0] if disruptions and isinstance(disruptions[0], dict) else {}
        kind = str(disruption.get("type") or "unknown")
        jobs = int(row.get("num_jobs") or 0)
        machines = int(row.get("num_machines") or 0)
        if not instance_id or instance_id in active_tokens or kind == "unknown":
            continue
        eligible.append((kind, jobs * machines, instance_id, row))
    by_kind: dict[str, list[tuple[str, int, str, dict[str, Any]]]] = defaultdict(list)
    for item in eligible:
        by_kind[item[0]].append(item)
    selected: list[tuple[str, int, str, dict[str, Any]]] = []
    for kind in sorted(by_kind):
        selected.append(max(by_kind[kind], key=lambda item: (item[1], item[2])))
    remaining = sorted(
        (item for item in eligible if item not in selected),
        key=lambda item: (-item[1], item[0], item[2]),
    )
    selected = (selected + remaining)[:limit]
    output = []
    for kind, complexity, instance_id, row in selected:
        jobs = int(row.get("num_jobs") or 0)
        machines = int(row.get("num_machines") or 0)
        output.append(
            _candidate(
                candidate_id=f"realm_j2/{instance_id}",
                source_family="realm_j2",
                domain="logistics",
                backend_kind="jsplib_job_shop",
                source_denominator_key=f"realm_j2:{instance_id}:{source_sha}",
                physical_source_key=physical_source_key,
                structural_axes=[
                    f"source_native_disruption:{kind}",
                    f"job_machine_shape:{jobs}x{machines}",
                    "source_native_operation_graph",
                ],
                environment=environment,
                source_metadata={
                    "path": str(path.resolve()),
                    "sha256": source_sha,
                    "instance_id": instance_id,
                    "disruption_type": kind,
                    "num_jobs": jobs,
                    "num_machines": machines,
                },
                priority=float(complexity),
            )
        )
    return output


def _jsplib_candidates(
    path: Path,
    *,
    active_tokens: set[str],
    environment: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = _load(path)
    if not isinstance(rows, list):
        raise ValueError("JSPLIB metadata must be a list")
    eligible = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("name")
        and str(row["name"]) not in active_tokens
        and (row.get("optimum") is not None or isinstance(row.get("bounds"), dict))
    ]
    eligible.sort(
        key=lambda row: (
            -(int(row.get("jobs") or 0) * int(row.get("machines") or 0)),
            str(row["name"]),
        )
    )
    selected: list[dict[str, Any]] = []
    seen_shapes: set[tuple[int, int]] = set()
    for row in eligible:
        shape = (int(row.get("jobs") or 0), int(row.get("machines") or 0))
        if shape in seen_shapes:
            continue
        selected.append(row)
        seen_shapes.add(shape)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        selected.extend(row for row in eligible if row not in selected)
        selected = selected[:limit]
    source_sha = _sha256(path)
    output = []
    for row in selected:
        name = str(row["name"])
        jobs = int(row.get("jobs") or 0)
        machines = int(row.get("machines") or 0)
        instance_path = (path.parent / str(row.get("path") or "")).resolve()
        if not instance_path.is_file():
            raise FileNotFoundError(
                f"JSPLIB candidate instance is missing: {instance_path}"
            )
        instance_sha = _sha256(instance_path)
        physical_source_key = _physical_asset_graph_key(
            "jsplib_job_shop", [(instance_path.name, instance_sha)]
        )
        output.append(
            _candidate(
                candidate_id=f"jsplib/{name}/typed_breakdown_prefilter",
                source_family="jsplib",
                domain="logistics",
                backend_kind="jsplib_job_shop",
                source_denominator_key=f"jsplib_job_shop:{name}",
                physical_source_key=physical_source_key,
                structural_axes=[
                    f"job_machine_shape:{jobs}x{machines}",
                    "source_native_precedence_graph",
                    "typed_seeded_breakdown_stress",
                ],
                environment=environment,
                source_metadata={
                    "metadata_path": str(path.resolve()),
                    "metadata_sha256": source_sha,
                    "instance": name,
                    "instance_path": str(instance_path),
                    "instance_sha256": instance_sha,
                    "num_jobs": jobs,
                    "num_machines": machines,
                },
                priority=float(jobs * machines),
                procedural=True,
            )
        )
    return output


def _pglib_candidates(
    root: Path,
    *,
    active_tokens: set[str],
    environment: str,
    limit: int,
) -> list[dict[str, Any]]:
    output = []
    for case in PREFERRED_PGLIB_CASES:
        path = root / f"{case}.m"
        if not path.is_file() or case in active_tokens:
            continue
        source_sha = _sha256(path)
        physical_source_key = _physical_asset_graph_key(
            "pandapower_acopf", [(path.name, source_sha)]
        )
        output.append(
            _candidate(
                candidate_id=f"pglib_opf/{case}/reserve_ramp_recovery_redesign",
                source_family="pglib_opf",
                domain="power_grid",
                backend_kind="pandapower_acopf",
                source_denominator_key=f"pandapower_acopf:{case}:reserve_ramp_recovery_v1",
                physical_source_key=physical_source_key,
                structural_axes=[
                    f"independent_topology:{case}",
                    "reserve_ramp_recovery_redesign",
                ],
                environment=environment,
                source_metadata={"path": str(path.resolve()), "sha256": source_sha},
                priority=float(
                    len(PREFERRED_PGLIB_CASES) - PREFERRED_PGLIB_CASES.index(case)
                ),
            )
        )
        if len(output) == limit:
            break
    return output


def _citylearn_candidates(
    root: Path,
    *,
    environment: str,
    source_lock_path: Path | None,
) -> list[dict[str, Any]]:
    dataset = root / "baeda_3dem"
    schema = dataset / "schema.json"
    if not schema.is_file():
        return []
    source_lock = None
    if source_lock_path is not None and source_lock_path.is_file():
        source_lock = _load(source_lock_path)
        if source_lock.get("source_id") != "baeda_3dem":
            raise ValueError("CityLearn source lock dataset identity mismatch")
        asset_bindings = []
        for group in ("runtime_files", "derivation_files"):
            rows = source_lock.get(group) or {}
            if not isinstance(rows, dict):
                raise ValueError(f"CityLearn source lock {group} must be an object")
            for declared_path, row in sorted(rows.items()):
                digest = row.get("sha256") if isinstance(row, dict) else None
                if not isinstance(digest, str) or len(digest) != 64:
                    raise ValueError(
                        f"CityLearn source lock hash invalid: {declared_path}"
                    )
                asset_bindings.append((str(declared_path), digest))
    else:
        assets = _referenced_dataset_files(schema)
        asset_bindings = [(path.name, _sha256(path)) for path in assets]
    effective_environment = environment
    if source_lock is not None and "carbon_intensity.csv" in (
        source_lock.get("optional_runtime_assets_absent") or []
    ):
        effective_environment = "held_runtime"
    physical_source_key = _physical_asset_graph_key("citylearn", asset_bindings)
    source_sha = _identity_digest(physical_source_key)
    return [
        _candidate(
            candidate_id="citylearn/baeda_3dem/independent_graph_probe",
            source_family="citylearn",
            domain="building_energy",
            backend_kind="citylearn",
            source_denominator_key=f"citylearn:baeda_3dem:{source_sha}",
            physical_source_key=physical_source_key,
            structural_axes=["independent_building_graph", "storage_comfort_energy_tradeoff"],
            environment=effective_environment,
            source_metadata={
                "path": str(dataset.resolve()),
                "asset_graph_sha256": source_sha,
                "required_source_assets": [
                    {"declared_path": path, "sha256": digest}
                    for path, digest in asset_bindings
                ],
                "source_lock": (
                    _file_binding(source_lock_path)
                    if source_lock_path is not None
                    and source_lock_path.is_file()
                    else None
                ),
            },
            priority=1.0,
        )
    ]


def _datacenter_candidates(
    path: Path | None, *, environment: str
) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    ledger = _load(path)
    source = ledger.get("source") or {}
    physical = _physical_asset_graph_key(
        "alibaba_trace_sim",
        [
            (str(source.get("job_trace") or "job_info_df.csv"), str(source.get("job_trace_sha256") or "")),
            (
                str(source.get("node_inventory") or "node_info_df.csv"),
                str(source.get("node_trace_sha256") or ""),
            ),
        ],
    )
    output = []
    for row in ledger.get("candidates") or []:
        if not isinstance(row, dict):
            continue
        window_sha = str(row.get("source_window_sha256") or "")
        output.append(
            _candidate(
                candidate_id=str(row.get("candidate_id") or ""),
                source_family="datacenter_hard",
                domain="datacenter",
                backend_kind="alibaba_trace_sim",
                source_denominator_key=f"alibaba_spot_window:{window_sha}",
                physical_source_key=physical,
                structural_axes=list(row.get("independent_decision_axes") or []),
                environment=environment,
                source_metadata={
                    "ledger_path": str(path.resolve()),
                    "window_sha256": window_sha,
                    "gpu_model": (row.get("evidence") or {}).get("gpu_model"),
                    "suite_recipe": row.get("suite_recipe"),
                },
                priority=float((row.get("evidence") or {}).get("duration_ratio") or 0.0),
            )
        )
    return output


def _simbench_candidates(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    report = _load(path)
    if report.get("candidate_only") is not True or report.get("release_admission") is not False:
        raise ValueError("SimBench metadata must be candidate-only")
    output: list[dict[str, Any]] = []
    for raw in report.get("candidates") or []:
        if not isinstance(raw, dict):
            raise ValueError("SimBench metadata candidates must be objects")
        candidate = dict(raw)
        candidate["source_family"] = "simbench_commercial"
        candidate["source_metadata"] = {
            **dict(candidate.get("source_metadata") or {}),
            "candidate_metadata_path": str(path.resolve()),
            "candidate_metadata_sha256": _sha256(path),
            "upstream_source_family": raw.get("source_family"),
        }
        output.append(candidate)
    return output


def build_source_metadata(
    *,
    source_suite_path: Path,
    candidate_inventory_path: Path,
    realm_j2_path: Path,
    jsplib_metadata_path: Path,
    pglib_opf_root: Path,
    citylearn_datasets_root: Path,
    citylearn_source_lock_path: Path | None,
    datacenter_ledger_path: Path | None,
    simbench_metadata_path: Path | None = None,
    realm_limit: int = 6,
    jsplib_limit: int = 8,
    pglib_limit: int = 5,
) -> dict[str, Any]:
    if min(realm_limit, jsplib_limit, pglib_limit) < 0:
        raise ValueError("candidate limits must be non-negative")
    suite = _load(source_suite_path)
    inventory = _load(candidate_inventory_path)
    active_rows = suite.get("scenarios") or []
    active_tokens = _active_source_tokens(active_rows)
    environments = _environment_states(inventory)
    candidates = [
        *_realm_candidates(
            realm_j2_path,
            active_tokens=active_tokens,
            environment=environments.get("realm_j2", "held_repair"),
            limit=realm_limit,
        ),
        *_jsplib_candidates(
            jsplib_metadata_path,
            active_tokens=active_tokens,
            environment=environments.get("jsplib", "held_repair"),
            limit=jsplib_limit,
        ),
        *_pglib_candidates(
            pglib_opf_root,
            active_tokens=active_tokens,
            environment=environments.get("pglib_opf", "held_repair"),
            limit=pglib_limit,
        ),
        *_citylearn_candidates(
            citylearn_datasets_root,
            environment=environments.get("citylearn", "held_repair"),
            source_lock_path=citylearn_source_lock_path,
        ),
        *_datacenter_candidates(
            datacenter_ledger_path,
            environment=environments.get("alibaba_clusterdata", "held_repair"),
        ),
        *_simbench_candidates(simbench_metadata_path),
    ]
    candidates.sort(key=lambda row: (row["source_family"], row["candidate_id"]))
    by_family = Counter(str(row["source_family"]) for row in candidates)
    return {
        "schema_version": "operate-candidate-source-metadata-v1",
        "status": "complete_structural_prefilter",
        "candidate_only": True,
        "release_admission": False,
        "quality_claims": "structural_prefilter_only",
        "inputs": {
            "source_suite": _file_binding(source_suite_path),
            "candidate_inventory": _file_binding(candidate_inventory_path),
            "realm_j2": _file_binding(realm_j2_path),
            "jsplib_metadata": _file_binding(jsplib_metadata_path),
            "pglib_opf_root": {
                "path": str(pglib_opf_root.resolve()),
                "binding": "selected_candidate_file_sha256",
            },
            "citylearn_datasets_root": {
                "path": str(citylearn_datasets_root.resolve()),
                "binding": "selected_candidate_schema_sha256",
            },
            "citylearn_source_lock": (
                _file_binding(citylearn_source_lock_path)
                if citylearn_source_lock_path is not None
                and citylearn_source_lock_path.is_file()
                else None
            ),
            "datacenter_ledger": (
                _file_binding(datacenter_ledger_path)
                if datacenter_ledger_path is not None
                and datacenter_ledger_path.is_file()
                else None
            ),
            "simbench_metadata": (
                _file_binding(simbench_metadata_path)
                if simbench_metadata_path is not None
                and simbench_metadata_path.is_file()
                else None
            ),
        },
        "policy": {
            "raw_inventory_units_are_not_candidates": True,
            "behavioral_headroom_not_inferred": True,
            "source_consumption_not_inferred": True,
            "environment_failure_not_scientific_rejection": True,
            "active_effective_sources_excluded": True,
        },
        "candidates": candidates,
        "summary": {
            "n_candidates": len(candidates),
            "by_source_family": dict(sorted(by_family.items())),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", type=Path, default=DEFAULT_SOURCE_SUITE)
    parser.add_argument("--candidate-inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--realm-j2", type=Path, default=DEFAULT_REALM)
    parser.add_argument("--jsplib-metadata", type=Path, default=DEFAULT_JSPLIB)
    parser.add_argument("--pglib-opf-root", type=Path, default=DEFAULT_PGLIB)
    parser.add_argument("--citylearn-datasets-root", type=Path, default=DEFAULT_CITYLEARN)
    parser.add_argument(
        "--citylearn-source-lock", type=Path, default=DEFAULT_CITYLEARN_LOCK
    )
    parser.add_argument("--datacenter-ledger", type=Path, default=DEFAULT_DATACENTER)
    parser.add_argument("--simbench-metadata", type=Path, default=DEFAULT_SIMBENCH)
    parser.add_argument("--realm-limit", type=int, default=6)
    parser.add_argument("--jsplib-limit", type=int, default=8)
    parser.add_argument("--pglib-limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_source_metadata(
        source_suite_path=args.source_suite,
        candidate_inventory_path=args.candidate_inventory,
        realm_j2_path=args.realm_j2,
        jsplib_metadata_path=args.jsplib_metadata,
        pglib_opf_root=args.pglib_opf_root,
        citylearn_datasets_root=args.citylearn_datasets_root,
        citylearn_source_lock_path=args.citylearn_source_lock,
        datacenter_ledger_path=args.datacenter_ledger,
        simbench_metadata_path=args.simbench_metadata,
        realm_limit=args.realm_limit,
        jsplib_limit=args.jsplib_limit,
        pglib_limit=args.pglib_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
