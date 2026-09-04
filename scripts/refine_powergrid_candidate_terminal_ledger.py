#!/usr/bin/env python3
"""Build an exact, non-admitting terminal ledger for Power Grid candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402

POWER_BACKENDS = {
    "cigre_distribution",
    "grid2op",
    "opendss_fresh_feeders",
    "opendss_ieee13",
    "pandapower_acopf",
    "pglib_uc_synthetic",
}
HARD_GATES = [
    "source_consumption",
    "determinism",
    "native_positive_benefit",
    "safety",
    "task_completion",
    "identity",
    "high_extreme_post_change_response",
    "strategy_depth",
]


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, str]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}


def _source_paths(rows: list[dict[str, Any]], prefix: str) -> set[str]:
    return {
        str(path)
        for row in rows
        for path in row.get("provenance_files", [])
        if str(path).startswith(prefix)
    }


def build_ledger(
    *,
    published_core: dict[str, Any],
    protocol_core: dict[str, Any],
    bulk_suite: dict[str, Any],
    behavioral: dict[str, Any],
    uc_files: list[Path],
    opf_files: list[Path],
    benchmark_terminals: list[dict[str, Any]],
    probe_start_hash: str,
    ledger_build_hash: str,
) -> dict[str, Any]:
    core_rows = [
        row
        for row in published_core.get("scenarios", [])
        if row.get("backend_kind") in POWER_BACKENDS
    ]
    protocol_rows = [
        row for row in protocol_core.get("scenarios", []) if row.get("domain") == "power_grid"
    ]
    protected_uc = _source_paths(core_rows, "works/pglib-uc/")
    protected_opf = _source_paths(core_rows, "works/PGLib-OPF/")

    suite_by_id = {
        str(row["scenario_id"]): row for row in bulk_suite.get("scenarios", [])
    }
    result_by_source: dict[str, dict[str, Any]] = {}
    for result in behavioral.get("results", []):
        scenario = suite_by_id.get(str(result.get("scenario_id")))
        if scenario is None:
            raise ValueError(f"behavioral result absent from suite: {result.get('scenario_id')}")
        runtime_inputs = (
            scenario.get("case_ledger", {})
            .get("physical_source_lock", {})
            .get("runtime_input", [])
        )
        if len(runtime_inputs) != 1:
            raise ValueError(f"one UC runtime source required: {scenario.get('scenario_id')}")
        source = str(runtime_inputs[0])
        if source in result_by_source:
            raise ValueError(f"duplicate behavioral source result: {source}")
        result_by_source[source] = result

    source_units: list[dict[str, Any]] = []
    for path in sorted(uc_files):
        source = path.relative_to(ROOT).as_posix()
        if source in protected_uc:
            disposition = "excluded_existing_core"
            reasons = ["published_v0_51_core_source"]
            result = None
        else:
            disposition = "held_repair"
            result = result_by_source.get(source)
            if result is None:
                reasons = ["representative_gate_not_green_do_not_expand_to_56"]
            elif result.get("status") == "error":
                reasons = ["representative_probe_error_fail_closed"]
            else:
                failed = sorted(
                    key for key, passed in (result.get("checks") or {}).items() if passed is False
                )
                reasons = failed or ["representative_probe_not_admission_evidence"]
        source_units.append(
            {
                "source_family": "pglib_uc",
                "source_unit": source,
                "source_sha256": _sha256(path),
                "work_state": "terminal",
                "disposition": disposition,
                "reason_codes": reasons,
                "representative_result": None
                if result is None
                else {
                    "scenario_id": result.get("scenario_id"),
                    "status": result.get("status"),
                    "checks": result.get("checks"),
                    "error": result.get("error"),
                },
            }
        )

    for path in sorted(opf_files):
        source = path.relative_to(ROOT).as_posix()
        protected = source in protected_opf
        source_units.append(
            {
                "source_family": "pglib_opf",
                "source_unit": source,
                "source_sha256": _sha256(path),
                "work_state": "terminal",
                "disposition": "excluded_existing_core" if protected else "held_repair",
                "reason_codes": ["published_v0_51_core_source"]
                if protected
                else [
                    "static_snapshot_missing_paired_source_timeseries",
                    "synthetic_or_unrelated_timeseries_forbidden",
                ],
            }
        )

    keys = [(row["source_family"], row["source_unit"]) for row in source_units]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate source unit in terminal ledger")
    if any(row["work_state"] != "terminal" or not row["disposition"] for row in source_units):
        raise ValueError("every source unit requires exactly one terminal disposition")
    benchmark_keys = [str(row.get("source_id") or "") for row in benchmark_terminals]
    if not all(benchmark_keys):
        raise ValueError("every benchmark track requires a non-empty source_id")
    if len(benchmark_keys) != len(set(benchmark_keys)):
        raise ValueError("duplicate benchmark track in terminal ledger")
    if any(
        row.get("work_state") != "terminal" or not row.get("disposition")
        for row in benchmark_terminals
    ):
        raise ValueError("every benchmark track requires exactly one terminal disposition")

    artifact_hash = str(behavioral.get("implementation_tree_sha256") or "")
    probe_stale = len({probe_start_hash, artifact_hash, ledger_build_hash}) != 1
    dispositions = Counter(str(row["disposition"]) for row in source_units)
    uc_rows = [row for row in source_units if row["source_family"] == "pglib_uc"]
    opf_rows = [row for row in source_units if row["source_family"] == "pglib_opf"]
    return {
        "schema_version": "powergrid-candidate-terminal-ledger-v1",
        "status": "complete_non_admitting",
        "constraints": {
            "candidate_only": True,
            "release_admission": False,
            "core_admission_profile": "quality_core_v2",
            "protected_core_immutable": True,
            "hard_gates_unchanged": HARD_GATES,
            "full_protocol21_replay_executed": False,
        },
        "protected_core": {
            "published_release": "dt_sched_bench_v0_51_0",
            "n_power_grid_rows": len(core_rows),
            "n_pglib_uc_sources": len(protected_uc),
            "n_pglib_opf_sources": len(protected_opf),
            "protocol21_frozen_union_power_rows": [
                {
                    "scenario_id": row.get("scenario_id"),
                    "backend_kind": row.get("backend_kind"),
                    "source_denominator_key": row.get("source_denominator_key"),
                }
                for row in protocol_rows
            ],
        },
        "probe_identity": {
            "implementation_tree_sha256_start": probe_start_hash,
            "implementation_tree_sha256_artifact": artifact_hash,
            "implementation_tree_sha256_ledger_build": ledger_build_hash,
            "stale": probe_stale,
            "stale_policy": "fail_closed_no_admission",
        },
        "pglib_opf_pairing_audit": {
            "status": "blocked",
            "finding": "PGLib-OPF contains static MATPOWER cases, not paired operating series.",
            "compatible_alternative": (
                "RTS-GMLC contains a coherent topology and native load/PV/wind/reserve series; "
                "it is a separate source and must not be padded onto unrelated PGLib-OPF cases."
            ),
        },
        "source_units": source_units,
        "benchmark_tracks": benchmark_terminals,
        "candidate_working_set": {
            "core_admission_profile": "quality_core_v2",
            "n_admission_eligible": 0,
            "scenario_ids": [],
            "reason": "no current-hash representative cleared all unchanged scientific gates",
        },
        "summary": {
            "n_source_units": len(source_units),
            "n_terminal": len(source_units),
            "n_benchmark_tracks": len(benchmark_terminals),
            "n_all_inputs": len(source_units) + len(benchmark_terminals),
            "n_all_terminal": len(source_units) + len(benchmark_terminals),
            "n_uc": len(uc_rows),
            "n_uc_excluded_existing_core": sum(
                row["disposition"] == "excluded_existing_core" for row in uc_rows
            ),
            "n_uc_noncore_held": sum(row["disposition"] == "held_repair" for row in uc_rows),
            "n_uc_representative_results": len(result_by_source),
            "n_uc_representative_passed": sum(
                row.get("status") == "passed" for row in result_by_source.values()
            ),
            "n_opf": len(opf_rows),
            "n_opf_excluded_existing_core": sum(
                row["disposition"] == "excluded_existing_core" for row in opf_rows
            ),
            "n_opf_noncore_held_missing_paired_timeseries": sum(
                row["disposition"] == "held_repair" for row in opf_rows
            ),
            "dispositions": dict(sorted(dispositions.items())),
            "all_inputs_have_exactly_one_terminal": True,
            "do_not_expand_pglib_uc_to_56": True,
            "core_mutated": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/powergrid_candidate_refine_20260814/terminal_ledger.json",
    )
    parser.add_argument("--probe-start-hash", required=True)
    args = parser.parse_args()

    core_path = ROOT / "release/dt_sched_bench_v0_51_0/core_suite.json"
    protocol_path = (
        ROOT
        / "reports/protocol21_pending_union_fresh_current_20260811_realtraffic"
        / "refined_core_selection_protocol2_v21.json"
    )
    suite_path = ROOT / "reports/pglib_bulk_protocol21_source_suite_20260812.json"
    behavioral_path = (
        ROOT
        / "reports/powergrid_candidate_refine_20260814"
        / "pglib_uc_noncore_representative_behavioral.json"
    )
    terminal_paths = [
        ROOT / "reports/latest_benchmark_candidate_wave_20260813/terminals" / name
        for name in ("rts_gmlc.json", "grid2op_cache.json", "simbench_official.json")
    ]
    terminals = []
    for path in terminal_paths:
        row = _load(path)
        terminals.append(
            {
                "source_id": row.get("source_id"),
                "work_state": "terminal",
                "disposition": row.get("disposition"),
                "blockers": row.get("blockers", []),
                "binding": _binding(path),
            }
        )
    current_hash = implementation_identity(ROOT)["implementation_tree_sha256"]
    payload = build_ledger(
        published_core=_load(core_path),
        protocol_core=_load(protocol_path),
        bulk_suite=_load(suite_path),
        behavioral=_load(behavioral_path),
        uc_files=sorted((ROOT / "works/pglib-uc").glob("**/*.json")),
        opf_files=sorted((ROOT / "works/PGLib-OPF").glob("pglib_opf_case*.m")),
        benchmark_terminals=terminals,
        probe_start_hash=args.probe_start_hash,
        ledger_build_hash=current_hash,
    )
    if payload["summary"]["n_uc"] != 56 or payload["summary"]["n_opf"] != 66:
        raise ValueError("expected exactly 56 PGLib-UC and 66 PGLib-OPF source units")
    if payload["protected_core"]["n_power_grid_rows"] != 89:
        raise ValueError("published Power Grid Core count drifted from 89")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
