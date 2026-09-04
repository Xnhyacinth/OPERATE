#!/usr/bin/env python3
"""Prioritize held-33 and external families for evidence-based refinement.

The planner is candidate-only. It does not change difficulty labels, invent
events, or admit Core rows. Optional materialization copies exactly one held
scenario into staging while preserving its locked physical-source identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402

PIPELINE_VERSION = "held33_external_refine_priority_v1"
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / "reports"
    / "protocol21_pending_union_fresh_current_20260812_wave2_realtraffic_stable"
)
DEFAULT_CORE = DEFAULT_RUN_ROOT / "refined_core_selection_protocol2_v21.json"
DEFAULT_SUITE = (
    REPO_ROOT / "reports" / "protocol21_pending_union_build_v1" / "source_suite_with_wave1.json"
)
DEFAULT_TRACK_C = REPO_ROOT / "reports" / "track_c_external_conversion_current_20260812.json"
DEFAULT_WORKS = REPO_ROOT / ".hl" / "artifacts" / "works_candidate_inventory_2026-08-12.json"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "held33_external_refine_plan_current_20260812.json"
DEFAULT_CANDIDATE_REPORT = (
    REPO_ROOT / "reports" / "held33_jsplib_swv06_candidate_report_current_20260812.json"
)
DEFAULT_CANDIDATE_SUITE = (
    REPO_ROOT / "reports" / "held33_jsplib_swv06_source_suite_current_20260812.json"
)
DEFAULT_SCENARIO = (
    REPO_ROOT
    / "scenarios"
    / "staging"
    / "held33_external_refine"
    / "logistics"
    / "job_shop_dispatch"
    / "time_pressure"
    / "extreme"
    / "jobshop_swv06_dynamic_recovery_extreme_s44.yaml"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _index(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("results") or report.get("rows") or []
    return {
        str(row["scenario_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("scenario_id")
    }


def _failed_checks(row: dict[str, Any], names: tuple[str, ...]) -> list[str]:
    checks = row.get("checks") or {}
    return sorted(name for name in names if checks.get(name) is False)


def _difficulty_evidence(row: dict[str, Any]) -> dict[str, Any]:
    direct = row.get("difficulty_evidence")
    if isinstance(direct, dict):
        return direct
    pipeline = row.get("source_grounded_pipeline") or {}
    nested = pipeline.get("difficulty_evidence")
    return nested if isinstance(nested, dict) else {}


def _priority(row: dict[str, Any]) -> tuple[str, int]:
    blockers = row["blockers"]
    total = sum(len(values) for values in blockers.values())
    agentic_only = bool(blockers["agentic"]) and not any(
        blockers[name] for name in ("source", "runtime", "license", "native", "headroom", "depth")
    )
    difficulty_bonus = {
        "extreme": 100,
        "high": 60,
        "medium": 20,
        "basic": 0,
    }.get(str(row.get("difficulty_level") or ""), 0)
    if agentic_only:
        return "P0_single_gate_repair", 1000 + difficulty_bonus - total
    if not blockers["source"] and not blockers["runtime"] and not blockers["license"]:
        return "P1_native_headroom_or_depth", 700 + difficulty_bonus - total
    if not blockers["license"] and not blockers["runtime"]:
        return "P2_adapter_or_source_evidence", 400 - total
    return "P3_held_external_dependency", 100 - total


def build_held_priority_rows(
    *,
    rejected: list[dict[str, Any]],
    suite_rows: list[dict[str, Any]],
    behavioral: dict[str, Any],
    source_consumption: dict[str, Any],
    tasks: dict[str, Any],
    source_grounded: dict[str, Any],
    agentic: dict[str, Any],
) -> list[dict[str, Any]]:
    suite = {str(row["scenario_id"]): row for row in suite_rows}
    behavioral_by_id = _index(behavioral)
    source_by_id = _index(source_consumption)
    task_by_id = _index(tasks)
    grounded_by_id = _index(source_grounded)
    agentic_by_id = _index(agentic)
    rows = []
    for rejected_row in rejected:
        scenario_id = str(rejected_row["scenario_id"])
        source_row = source_by_id.get(scenario_id, {})
        behavior_row = behavioral_by_id.get(scenario_id, {})
        task_row = task_by_id.get(scenario_id, {})
        grounded_row = grounded_by_id.get(scenario_id, {})
        agentic_row = agentic_by_id.get(scenario_id, {})
        suite_row = suite.get(scenario_id, {})
        source_blockers = list(source_row.get("blockers") or [])
        if source_row.get("status") != "passed" and not source_blockers:
            source_blockers = ["source_consumption_not_passed"]
        runtime_blockers = []
        if behavior_row.get("status") not in {"passed", "complete"}:
            runtime_blockers = ["behavioral_replay_not_passed"]
        native_blockers = _failed_checks(
            behavior_row,
            ("native_backend_executable", "native_state_changing_leverage"),
        )
        headroom_blockers = _failed_checks(
            behavior_row,
            ("aggregate_decision_headroom", "positive_decision_headroom"),
        )
        if task_row.get("status") != "passed":
            headroom_blockers.append("task_contract_not_passed")
        depth = _difficulty_evidence(grounded_row)
        grounded_failures = set(grounded_row.get("failures") or [])
        depth_blockers = ["difficulty_proof"] if "difficulty_proof" in grounded_failures else []
        agentic_blockers = list(agentic_row.get("blockers") or [])
        path = str(suite_row.get("path") or "")
        license_blockers = [] if path else ["scenario_path_or_license_unresolved"]
        blockers = {
            "source": sorted(set(source_blockers)),
            "runtime": sorted(set(runtime_blockers)),
            "license": license_blockers,
            "native": sorted(set(native_blockers)),
            "headroom": sorted(set(headroom_blockers)),
            "depth": depth_blockers,
            "agentic": sorted(set(agentic_blockers)),
        }
        result = {
            "scenario_id": scenario_id,
            "scenario_signature": rejected_row.get("scenario_signature"),
            "domain": suite_row.get("domain") or scenario_id.split("/", 1)[0],
            "backend_kind": suite_row.get("backend_kind"),
            "family": suite_row.get("family"),
            "difficulty_level": suite_row.get("difficulty_level"),
            "path": path,
            "effective_source_key": suite_row.get("source_denominator_key")
            or suite_row.get("source_key"),
            "blockers": blockers,
            "evidence": {
                "behavioral_status": behavior_row.get("status"),
                "source_consumption_status": source_row.get("status"),
                "task_status": task_row.get("status"),
                "source_grounded_status": grounded_row.get("status"),
                "agentic_status": agentic_row.get("status"),
                "required_depth_lower_bound": depth.get("required_depth_lower_bound"),
                "exact_dependency_depth": depth.get("exact_dependency_depth"),
                "plan_reversal_count": depth.get("plan_reversal_count"),
            },
            "terminal_disposition": "held_repair",
            "core_admission_claimed": False,
        }
        priority_class, score = _priority(result)
        result["priority_class"] = priority_class
        result["priority_score"] = score
        rows.append(result)
    return sorted(rows, key=lambda row: (-row["priority_score"], row["scenario_id"]))


def _external_rows(track_c: dict[str, Any], works: dict[str, Any]) -> list[dict[str, Any]]:
    def categorize(codes: list[str]) -> dict[str, list[str]]:
        categorized = {
            "source": [],
            "native": [],
            "runtime": [],
            "license": [],
            "headroom": [],
            "depth": [],
        }
        for code in codes:
            lowered = code.lower()
            if "license" in lowered or "terms" in lowered:
                key = "license"
            elif "runtime" in lowered or "clock" in lowered:
                key = "runtime"
            elif "depth" in lowered:
                key = "depth"
            elif any(
                token in lowered for token in ("task", "response", "counterfactual", "headroom")
            ):
                key = "headroom"
            elif any(token in lowered for token in ("native", "adapter", "state_effect")):
                key = "native"
            else:
                key = "source"
            categorized[key].append(code)
        return {key: sorted(set(values)) for key, values in categorized.items()}

    rows = []
    for recipe in track_c.get("recipes") or []:
        blockers = sorted(set(recipe.get("blocker_codes") or []))
        rows.append(
            {
                "kind": "external_benchmark",
                "source_id": recipe.get("source_id"),
                "domain": (recipe.get("target") or {}).get("domain"),
                "backend_kind": (recipe.get("target") or {}).get("backend"),
                "disposition": recipe.get("disposition"),
                "status": recipe.get("status"),
                "blockers": blockers,
                "blockers_by_gate": categorize(blockers),
                "ready_for_full_protocol21": bool(recipe.get("ready_for_full_protocol21")),
                "priority_score": 350 - 20 * len(blockers),
                "raw_asset_consumed": bool(
                    (recipe.get("external_source") or {}).get("raw_asset_consumed")
                ),
            }
        )
    for source in works.get("sources") or []:
        disposition = str(source.get("disposition") or "")
        blockers = []
        if not (source.get("license") or {}).get("evidence_bound"):
            blockers.append("license_unresolved")
        if not (source.get("runtime_binding") or {}).get("available"):
            blockers.append("runtime_unavailable")
        git = (source.get("source_lock") or {}).get("git") or {}
        if git.get("dirty"):
            blockers.append("source_tree_dirty")
        if disposition not in {"candidate_prefilter"}:
            blockers.append(disposition or "inventory_held")
        rows.append(
            {
                "kind": "works_inventory",
                "source_id": source.get("source_id"),
                "domain": source.get("domain"),
                "backend_kind": source.get("backend_kind"),
                "disposition": disposition,
                "blockers": sorted(set(blockers)),
                "blockers_by_gate": categorize(sorted(set(blockers))),
                "source_unit_count": source.get("source_unit_count"),
                "priority_score": 500 - 40 * len(set(blockers)),
                "candidate_only": True,
            }
        )
    return sorted(rows, key=lambda row: (-row["priority_score"], str(row["source_id"])))


def materialize_top_candidate(
    *,
    row: dict[str, Any],
    scenario_path: Path,
    report_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if row.get("priority_class") != "P0_single_gate_repair":
        raise ValueError("only a P0 single-gate repair may be materialized")
    source_path = Path(str(row.get("path") or ""))
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    body = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError("source scenario must be a YAML object")
    if body.get("scenario_id") != row.get("scenario_id"):
        raise ValueError("source scenario identity mismatch")
    if body.get("scenario_signature") != row.get("scenario_signature"):
        raise ValueError("source scenario signature mismatch")
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    report = {
        "schema_version": "candidate-report-v1",
        "status": "candidate_materialized_pending_protocol21_prefilter",
        "n_scenarios": 1,
        "release_membership_changed": False,
        "constraints": {
            "candidate_only": True,
            "one_per_effective_source_identity": True,
            "declared_events_add_independence": False,
            "difficulty_relabel_allowed": False,
            "core_admission_claimed": False,
        },
        "scenarios": [
            {
                "scenario_id": body["scenario_id"],
                "scenario_signature": body["scenario_signature"],
                "path": (
                    scenario_path.resolve().relative_to(repo_root.resolve()).as_posix()
                    if scenario_path.resolve().is_relative_to(repo_root.resolve())
                    else str(scenario_path.resolve())
                ),
                "source_denominator_key": row.get("effective_source_key"),
                "effective_source_key": row.get("effective_source_key"),
                "remaining_blockers": row["blockers"],
            }
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def build_report() -> dict[str, Any]:
    core = _load(DEFAULT_CORE)
    suite = _load(DEFAULT_SUITE)
    held = build_held_priority_rows(
        rejected=list(core.get("rejected") or []),
        suite_rows=list(suite.get("scenarios") or []),
        behavioral=_load(DEFAULT_RUN_ROOT / "behavioral_calibration_protocol2_v21.json"),
        source_consumption=_load(DEFAULT_RUN_ROOT / "source_consumption_protocol2_v21.json"),
        tasks=_load(DEFAULT_RUN_ROOT / "task_contracts_protocol2_v21.json"),
        source_grounded=_load(DEFAULT_RUN_ROOT / "source_grounded_protocol2_v21.json"),
        agentic=_load(DEFAULT_RUN_ROOT / "agentic_core_contract_protocol2_v21.json"),
    )
    for row in held:
        path = REPO_ROOT / str(row.get("path") or "")
        if not path.is_file():
            continue
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        license_value = (
            (body.get("provenance") or {}).get("license") if isinstance(body, dict) else None
        )
        row["evidence"]["license_declared"] = bool(license_value)
        if not license_value:
            row["blockers"]["license"] = ["source_license_not_declared"]
        priority_class, score = _priority(row)
        row["priority_class"] = priority_class
        row["priority_score"] = score
    held.sort(key=lambda row: (-row["priority_score"], row["scenario_id"]))
    external = _external_rows(_load(DEFAULT_TRACK_C), _load(DEFAULT_WORKS))
    counts = Counter(row["priority_class"] for row in held)
    return {
        "schema_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete_non_admitting",
        "implementation_tree_sha256": implementation_identity()["implementation_tree_sha256"],
        "bindings": {
            str(path.relative_to(REPO_ROOT)): _sha256(path)
            for path in (DEFAULT_CORE, DEFAULT_SUITE, DEFAULT_TRACK_C, DEFAULT_WORKS)
        },
        "n_held": len(held),
        "n_external_inventory_rows": len(external),
        "priority_counts": dict(sorted(counts.items())),
        "selection_rule": (
            "prefer local source-locked executable families with native effect, "
            "task/headroom/depth evidence; never add random events or relabel difficulty"
        ),
        "selected_refine_target": held[0] if held else None,
        "held_rows": held,
        "external_rows": external,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--materialize-top", action="store_true")
    parser.add_argument("--candidate-report", type=Path, default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--candidate-suite", type=Path, default=DEFAULT_CANDIDATE_SUITE)
    parser.add_argument("--scenario-output", type=Path, default=DEFAULT_SCENARIO)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.materialize_top:
        materialize_top_candidate(
            row=report["selected_refine_target"],
            scenario_path=args.scenario_output,
            report_path=args.candidate_report,
        )
        from scripts.build_protocol21_candidate_source_suite import build_suite

        suite = build_suite(args.candidate_report.resolve())
        args.candidate_suite.write_text(
            json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "n_held": report["n_held"],
                "selected": (report.get("selected_refine_target") or {}).get("scenario_id"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
