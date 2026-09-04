#!/usr/bin/env python3
"""Materialize a source-locked, candidate-only Power Grid conversion batch.

This command is the Power Grid side of the ``works`` expansion queue.  It
converts every locally available PGLib-UC source unit and the matching RTS-GMLC
DA/RT windows into native Protocol-2.1 scenario YAMLs.  Static PGLib-OPF
snapshots are inventoried with an explicit repair disposition; they are not
given an unrelated time series merely to inflate the candidate count.  Native
CIGRE/OpenDSS/Grid2Op/SimBench staging rows are imported by reference when
their source contract is complete.

The output is deliberately non-admitting.  Materialization proves source
identity, licensing, graph/task compilation, and deterministic signatures; the
normal native prefilter and full Protocol-2.1 replay remain the only path to
Core.  The script never edits a release manifest or a frozen Core suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.suite_identity import recompute_signature_with_seed  # noqa: E402
from domains.power_grid.seeds.from_pglib_uc import (  # noqa: E402
    build_critical_winter_peak_seed,
    build_reserve_stress_seed,
    build_wind_uncertainty_seed,
)
from domains.power_grid.seeds.from_rts_real_ts import (  # noqa: E402
    build_daily_ops_real_forecast_seed,
)
from scripts.build_protocol21_candidate_source_suite import build_suite  # noqa: E402
from scripts.prepare_protocol21_working_set import _source_contract  # noqa: E402

DEFAULT_UC_ROOT = ROOT / "works/pglib-uc"
DEFAULT_OPF_ROOT = ROOT / "works/PGLib-OPF"
DEFAULT_RTS_ROOT = ROOT / "works/RTS-GMLC"
DEFAULT_BASE_CORE = ROOT / "release/dt_sched_bench_v0_51_0/core_suite.json"
DEFAULT_STAGING = ROOT / "scenarios/staging/powergrid_bulk_candidates_20260813"
DEFAULT_REPORT = ROOT / "reports/powergrid_candidate_batch_20260813.json"
DEFAULT_SUITE = ROOT / "reports/powergrid_candidate_batch_source_suite_20260813.json"

_DISPOSITIONS = {
    "core_locked_increment",
    "held_repair",
    "held_runtime",
    "held_license_or_terms",
    "transfer_only",
    "secondary_duplicate",
    "retired_intrinsic",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any, length: int = 32) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:length]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML object required: {path}")
    return value


def _locked_identity_sets(base_core: Path) -> tuple[set[tuple[str, str]], set[str], set[str]]:
    payload = _load_json(base_core)
    exact: set[tuple[str, str]] = set()
    source_keys: set[str] = set()
    source_files: set[str] = set()
    for row in payload.get("scenarios") or []:
        if not isinstance(row, dict):
            continue
        scenario_id = str(row.get("scenario_id") or row.get("seed_id") or "")
        signature = str(row.get("scenario_signature") or "")
        if scenario_id and signature:
            exact.add((scenario_id, signature))
        for key in ("source_denominator_key", "source_key"):
            value = row.get(key)
            if value not in (None, "", {}, []):
                source_keys.add(str(value))
        contract = row.get("source_contract") or {}
        for key in ("runtime_input", "derivation_input", "implementation_asset", "metadata"):
            for value in contract.get(key) or []:
                source_files.add(str(value))
        for value in (row.get("provenance") or {}).get("files") or []:
            source_files.add(str(value))
    return exact, source_keys, source_files


def _source_hashes(paths: Iterable[str], *, root: Path = ROOT) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw in paths:
        path = Path(str(raw))
        resolved = path if path.is_absolute() else root / path
        if resolved.is_file():
            hashes[str(raw)] = _sha256(resolved)
    return hashes


def _native_graph_contract(
    *, backend: str, source_key: str, source_kind: str, horizon_ticks: int, task: dict[str, Any]
) -> dict[str, Any]:
    """Describe the compiled native environment/task graph, not a narrative."""
    return {
        "schema_version": "power_grid_native_graph_task_v1",
        "source_kind": source_kind,
        "source_key": source_key,
        "backend": backend,
        "state_graph": {
            "nodes": [
                "locked_source_window",
                "native_network_state",
                "controllable_generators",
                "source_schedule_events",
                "observations_and_tool_results",
                "native_outcomes",
            ],
            "edges": [
                ["locked_source_window", "native_network_state"],
                ["native_network_state", "source_schedule_events"],
                ["source_schedule_events", "observations_and_tool_results"],
                ["observations_and_tool_results", "controllable_generators"],
                ["controllable_generators", "native_outcomes"],
            ],
        },
        "task_schedule": {
            "horizon_ticks": int(horizon_ticks),
            "phase_order": ["inspect", "anticipate", "commit", "dispatch", "review"],
            "ordered_native_milestones": list(task.get("ordered_tool_milestones") or []),
            "task_completion_is_backend_predicate": True,
        },
        "event_policy": {
            "source_schedule_drives_state": True,
            "declared_perturbations_are_seeded_overlays": True,
            "declared_perturbations_do_not_create_source_independence": True,
            "hidden_or_delayed_observation_is_runtime_enforced": True,
        },
    }


def _attach_contract(body: dict[str, Any], *, source_files: list[str], source_kind: str) -> dict[str, Any]:
    """Attach hashes and native graph metadata before recomputing the signature."""
    body = deepcopy(body)
    config = body.setdefault("backend_config", {})
    source_key = str(config.get("source_denominator_key") or "")
    task = config.get("task_requirements") or {}
    config["source_denominator_key"] = source_key
    config["candidate_conversion_contract"] = _native_graph_contract(
        backend=str(body.get("backend_kind") or ""),
        source_key=source_key,
        source_kind=source_kind,
        horizon_ticks=int(body.get("horizon_ticks") or 0),
        task=task,
    )
    provenance = body.setdefault("provenance", {})
    provenance["files"] = list(dict.fromkeys(str(item) for item in source_files))
    contract = _source_contract(body)
    runtime = list(contract.get("runtime_input") or [])
    derivation = [item for item in provenance["files"] if item not in runtime]
    contract["derivation_input"] = list(dict.fromkeys(derivation))
    required = [*contract.get("runtime_input", []), *contract.get("derivation_input", [])]
    contract["file_sha256s"] = _source_hashes(required)
    body["source_contract"] = contract
    body.pop("scenario_signature", None)
    body["scenario_signature"] = recompute_signature_with_seed(body, int(body.get("seed") or 0))
    return body


def _candidate_row(body: dict[str, Any], path: Path, *, reason_codes: list[str]) -> dict[str, Any]:
    config = body.get("backend_config") or {}
    source_key = str(config.get("source_denominator_key") or "")
    return {
        "scenario_id": str(body.get("scenario_id") or body.get("seed_id") or ""),
        "scenario_signature": str(body.get("scenario_signature") or ""),
        "path": _relative(path),
        "domain": str(body.get("domain") or "power_grid"),
        "backend_kind": str(body.get("backend_kind") or ""),
        "family": str(body.get("family") or ""),
        "difficulty_mode": str(body.get("difficulty_mode") or ""),
        "difficulty_level": str(body.get("difficulty_level") or ""),
        "horizon_ticks": int(body.get("horizon_ticks") or 0),
        "seed": int(body.get("seed") or 0),
        "source_denominator_key": source_key,
        "physical_source_key": _digest((body.get("source_contract") or {}).get("file_sha256s") or {}),
        "status": "pending_protocol21_full_admission",
        "reason_codes": ["candidate_only", *reason_codes],
    }


def _source_unit(
    *, source_family: str, source_unit: str, source_sha256: str, disposition: str,
    reason_codes: list[str], candidate_ids: list[str] | None = None, source_kind: str = ""
) -> dict[str, Any]:
    if disposition not in _DISPOSITIONS:
        raise ValueError(f"invalid disposition: {disposition}")
    return {
        "source_family": source_family,
        "source_unit": source_unit,
        "source_sha256": source_sha256,
        "source_kind": source_kind or source_family,
        "work_state": "terminal",
        "disposition": disposition,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "candidate_scenario_ids": list(candidate_ids or []),
        "simulator_calls": 0,
    }


def _seed_for_uc(case_path: Path, *, seed: int) -> Any:
    subset = case_path.parent.name
    slug = _slug(case_path.stem)
    kwargs = {
        "seed_id": f"power_grid/pglib_uc_{subset}/deep_planning/extreme/{slug}_s{seed}",
        "seed": seed,
        "difficulty_mode": "deep_planning",
        "difficulty_level": "extreme",
    }
    if subset == "ca":
        return build_reserve_stress_seed(case_path, **kwargs)
    if subset == "ferc":
        return build_wind_uncertainty_seed(case_path, **kwargs)
    if subset == "rts_gmlc":
        return build_critical_winter_peak_seed(case_path, **kwargs)
    raise ValueError(f"unsupported pglib-uc subset: {subset}")


def _materialize_uc(
    *, case_path: Path, staging_root: Path, seed: int, locked_exact: set[tuple[str, str]], locked_keys: set[str], locked_files: set[str]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    relative = _relative(case_path)
    source_hash = _sha256(case_path)
    source_key = f"pglib_uc:{case_path.parent.name}:{case_path.name}:long_horizon"
    if source_key in locked_keys or relative in locked_keys or relative in locked_files:
        return None, _source_unit(
            source_family="pglib_uc",
            source_unit=relative,
            source_sha256=source_hash,
            disposition="secondary_duplicate",
            reason_codes=["source_denominator_already_locked"],
            source_kind="unit_commitment_instance",
        )
    body = _seed_for_uc(case_path, seed=seed).to_dict()
    body["scenario_id"] = str(body["seed_id"])
    body.setdefault("backend_config", {})["source_denominator_key"] = source_key
    body = _attach_contract(body, source_files=[relative, "works/pglib-uc/LICENSE"], source_kind="pglib_uc_unit_commitment")
    output = staging_root / "pglib_uc" / str(body["family"]) / f"{_slug(case_path.stem)}_extreme_s{seed}.yaml"
    row = _candidate_row(body, output, reason_codes=["source_schedule_compiled", "multi_unit_native_task_contract"])
    if (row["scenario_id"], row["scenario_signature"]) in locked_exact:
        return None, _source_unit(
            source_family="pglib_uc", source_unit=relative, source_sha256=source_hash,
            disposition="secondary_duplicate", reason_codes=["exact_identity_already_locked"], source_kind="unit_commitment_instance",
        )
    unit = _source_unit(
        source_family="pglib_uc", source_unit=relative, source_sha256=source_hash,
        disposition="held_repair", reason_codes=["candidate_materialized_requires_native_prefilter", "source_locked"],
        candidate_ids=[row["scenario_id"]], source_kind="unit_commitment_instance",
    )
    return {"body": body, "row": row, "output": output}, unit


def _materialize_rts(
    *, case_path: Path, staging_root: Path, rts_root: Path, seed: int, locked_exact: set[tuple[str, str]], locked_keys: set[str], locked_files: set[str]
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    relative_case = _relative(case_path)
    date = case_path.stem
    source_key = f"rts_gmlc:real_forecast_window:{date}:24h"
    source_files = [
        relative_case,
        _relative(rts_root / "RTS_Data/timeseries_data_files/Load/DAY_AHEAD_regional_Load.csv"),
        _relative(rts_root / "RTS_Data/timeseries_data_files/Load/REAL_TIME_regional_Load.csv"),
        "works/RTS-GMLC/README.md",
    ]
    source_hash = _sha256(case_path)
    if source_key in locked_keys or relative_case in locked_files:
        return None, _source_unit(
            source_family="rts_gmlc_real_forecast", source_unit=date, source_sha256=source_hash,
            disposition="secondary_duplicate", reason_codes=["source_denominator_already_locked"], source_kind="da_rt_forecast_window",
        )
    body = build_daily_ops_real_forecast_seed(
        case_path,
        seed_id=f"power_grid/rts_gmlc_real_forecast_24h/deep_planning/extreme/rts_{date}_s{seed}",
        seed=seed,
        difficulty_mode="deep_planning",
        difficulty_level="extreme",
    ).to_dict()
    body["scenario_id"] = str(body["seed_id"])
    body.setdefault("backend_config", {})["source_denominator_key"] = source_key
    body["backend_config"]["source_window"] = {"calendar_day": date, "window_hours": 24, "decision_axis": "da_to_rt_forecast_error"}
    body = _attach_contract(body, source_files=source_files, source_kind="rts_gmlc_da_rt_forecast_window")
    output = staging_root / "rts_gmlc_real_forecast" / f"rts_{date}_extreme_s{seed}.yaml"
    row = _candidate_row(body, output, reason_codes=["real_da_rt_profile_compiled", "source_schedule_compiled"])
    if (row["scenario_id"], row["scenario_signature"]) in locked_exact:
        return None, _source_unit(
            source_family="rts_gmlc_real_forecast", source_unit=date, source_sha256=source_hash,
            disposition="secondary_duplicate", reason_codes=["exact_identity_already_locked"], source_kind="da_rt_forecast_window",
        )
    unit = _source_unit(
        source_family="rts_gmlc_real_forecast", source_unit=date, source_sha256=source_hash,
        disposition="held_repair", reason_codes=["candidate_materialized_requires_native_prefilter", "real_da_rt_window"],
        candidate_ids=[row["scenario_id"]], source_kind="da_rt_forecast_window",
    )
    return {"body": body, "row": row, "output": output}, unit


def _inventory_static_opf(*, opf_root: Path, locked_keys: set[str]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for path in sorted(opf_root.glob("pglib_opf_case*.m")):
        relative = _relative(path)
        duplicate = relative in locked_keys or path.stem in locked_keys
        units.append(
            _source_unit(
                source_family="pglib_opf",
                source_unit=relative,
                source_sha256=_sha256(path),
                disposition="secondary_duplicate" if duplicate else "held_repair",
                reason_codes=(
                    ["source_denominator_already_locked"]
                    if duplicate
                    else [
                        "static_snapshot_requires_paired_source_timeseries",
                        "unrelated_timeseries_forbidden",
                        "native_acopf_conversion_recipe_available",
                    ]
                ),
                source_kind="static_opf_snapshot",
            )
        )
    return units


def _import_existing_staging(
    *, staging_roots: Iterable[Path], locked_exact: set[tuple[str, str]], seen: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for root in sorted({path.resolve() for path in staging_roots if path.exists()}):
        for path in sorted(root.rglob("*.yaml")):
            try:
                body = _load_yaml(path)
            except (OSError, ValueError, yaml.YAMLError):
                continue
            if str(body.get("domain") or "") != "power_grid":
                continue
            scenario_id = str(body.get("scenario_id") or body.get("seed_id") or "")
            if not scenario_id:
                continue
            signature = str(body.get("scenario_signature") or "")
            if not signature:
                try:
                    signature = recompute_signature_with_seed(body, int(body.get("seed") or 0))
                except Exception:
                    continue
            identity = (scenario_id, signature)
            source_unit = _relative(path)
            if identity in locked_exact or identity in seen:
                units.append(_source_unit(
                    source_family="existing_power_staging", source_unit=source_unit,
                    source_sha256=_sha256(path), disposition="secondary_duplicate",
                    reason_codes=["existing_staging_identity_already_seen"], source_kind="materialized_power_scenario",
                ))
                continue
            seen.add(identity)
            contract = body.get("source_contract") or {}
            required = [*(contract.get("runtime_input") or []), *(contract.get("derivation_input") or [])]
            if not required or not isinstance(contract.get("file_sha256s"), dict):
                units.append(_source_unit(
                    source_family="existing_power_staging", source_unit=source_unit,
                    source_sha256=_sha256(path), disposition="held_repair",
                    reason_codes=["source_contract_incomplete"], source_kind="materialized_power_scenario",
                ))
                continue
            rows.append(_candidate_row(body, path, reason_codes=["existing_native_staging_row", "candidate_replay_required"]))
            units.append(_source_unit(
                source_family="existing_power_staging", source_unit=source_unit,
                source_sha256=_sha256(path), disposition="held_repair",
                reason_codes=["existing_native_staging_row", "candidate_replay_required"],
                candidate_ids=[scenario_id], source_kind="materialized_power_scenario",
            ))
    return rows, units


def build(
    *,
    uc_root: Path = DEFAULT_UC_ROOT,
    opf_root: Path = DEFAULT_OPF_ROOT,
    rts_root: Path = DEFAULT_RTS_ROOT,
    base_core: Path = DEFAULT_BASE_CORE,
    staging_root: Path = DEFAULT_STAGING,
    existing_staging_roots: Iterable[Path] = (),
    seed: int = 42,
) -> tuple[dict[str, Any], dict[Path, dict[str, Any]]]:
    locked_exact, locked_keys, locked_files = _locked_identity_sets(base_core)
    files: dict[Path, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for case_path in sorted(uc_root.glob("*/*.json")):
        materialized, unit = _materialize_uc(
            case_path=case_path, staging_root=staging_root, seed=seed,
            locked_exact=locked_exact, locked_keys=locked_keys, locked_files=locked_files,
        )
        units.append(unit)
        if materialized is not None:
            files[materialized["output"]] = materialized["body"]
            rows.append(materialized["row"])
            seen.add((materialized["row"]["scenario_id"], materialized["row"]["scenario_signature"]))

    rts_cases = sorted((rts_root.parent / "pglib-uc" / "rts_gmlc").glob("*.json"))
    for case_path in rts_cases:
        materialized, unit = _materialize_rts(
            case_path=case_path, staging_root=staging_root, rts_root=rts_root, seed=seed,
            locked_exact=locked_exact, locked_keys=locked_keys, locked_files=locked_files,
        )
        units.append(unit)
        if materialized is not None:
            identity = (materialized["row"]["scenario_id"], materialized["row"]["scenario_signature"])
            if identity not in seen:
                files[materialized["output"]] = materialized["body"]
                rows.append(materialized["row"])
                seen.add(identity)

    units.extend(_inventory_static_opf(opf_root=opf_root, locked_keys=locked_keys))
    imported_rows, imported_units = _import_existing_staging(
        staging_roots=existing_staging_roots, locked_exact=locked_exact, seen=seen,
    )
    rows.extend(imported_rows)
    units.extend(imported_units)
    rows.sort(key=lambda row: (row["scenario_id"], row["scenario_signature"]))
    units.sort(key=lambda row: (row["source_family"], row["source_unit"]))

    # Source units are terminal in this materialization report; the candidate
    # rows themselves are intentionally pending the next replay stages.
    assert all(unit["work_state"] == "terminal" and unit["disposition"] in _DISPOSITIONS for unit in units)
    report = {
        "schema_version": "powergrid-candidate-batch-v1",
        "pipeline_version": "powergrid_source_native_graph_task_v1",
        "status": "candidate_materialization_complete_prefilter_pending",
        "candidate_only": True,
        "release_admission": False,
        "base_core": {"path": _relative(base_core), "sha256": _sha256(base_core)},
        "input_policy": {
            "frozen_core_untouched": True,
            "model_outcomes_used_for_filtering": False,
            "static_opf_without_native_timeseries": "held_repair",
            "declared_perturbations_create_source_independence": False,
            "all_source_units_terminal": True,
        },
        "source_units": units,
        "scenarios": rows,
        "summary": {
            "n_source_units": len(units),
            "n_candidate_rows": len(rows),
            "n_materialized_pglib_uc": sum(
                1 for row in units
                if row["source_family"] == "pglib_uc" and row["candidate_scenario_ids"]
            ),
            "n_materialized_rts_windows": sum(
                1 for row in units
                if row["source_family"] == "rts_gmlc_real_forecast" and row["candidate_scenario_ids"]
            ),
            "n_static_opf_units": sum(row["source_family"] == "pglib_opf" for row in units),
            "n_existing_staging_rows": len(imported_rows),
            "dispositions": dict(sorted(Counter(row["disposition"] for row in units).items())),
            "all_candidate_rows_have_unique_identity": len({(row["scenario_id"], row["scenario_signature"]) for row in rows}) == len(rows),
            "all_candidate_rows_source_bound": all(row["source_denominator_key"] and row["physical_source_key"] for row in rows),
        },
    }
    suite_marker = {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "pending_report_write",
        "n_scenarios": len(rows),
        "selection_policy": "powergrid_candidate_batch_source_locked_v1",
        "release_ready": False,
        "leaderboard_eligible": False,
    }
    return report, {"files": files, "suite_marker": suite_marker}


def execute(*, report: dict[str, Any], files: dict[Path, dict[str, Any]], report_path: Path, suite_path: Path) -> dict[str, Any]:
    for path, body in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    suite = build_suite(report_path)
    suite_path.parent.mkdir(parents=True, exist_ok=True)
    suite_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uc-root", type=Path, default=DEFAULT_UC_ROOT)
    parser.add_argument("--opf-root", type=Path, default=DEFAULT_OPF_ROOT)
    parser.add_argument("--rts-root", type=Path, default=DEFAULT_RTS_ROOT)
    parser.add_argument("--base-core", type=Path, default=DEFAULT_BASE_CORE)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--existing-staging", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    report, payload = build(
        uc_root=args.uc_root.resolve(), opf_root=args.opf_root.resolve(), rts_root=args.rts_root.resolve(),
        base_core=args.base_core.resolve(), staging_root=args.staging_root.resolve(),
        existing_staging_roots=[path.resolve() for path in args.existing_staging], seed=args.seed,
    )
    suite = payload["suite_marker"]
    if args.execute:
        suite = execute(report=report, files=payload["files"], report_path=args.report.resolve(), suite_path=args.suite.resolve())
    print(json.dumps({**report["summary"], "status": report["status"], "suite_rows": suite.get("n_scenarios", 0)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
