#!/usr/bin/env python3
"""Triage and stage a bounded refresh of four held native Power rows.

This batch deliberately performs no scenario transform.  The prior held plan
was built from an incomplete/stale merged run and assigned the same 30 derived
blockers to every remaining native Power row.  We copy four distinct physical
sources byte-identically into a candidate-only suite, verify current scenario
signatures and backend/source contracts, and require fresh preflight,
behavioral, source-consumption, and task evidence before any deeper refine.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from domains.registry import (  # noqa: E402
    get_backend_capability,
    get_domain_spec,
)

BASE_INPUT_ROOT = Path(__file__).resolve().parents[1]
HELD_PLAN = BASE_INPUT_ROOT / "reports/held33_external_refine_plan_current_20260812.json"
BASE_SUITE = BASE_INPUT_ROOT / "reports/protocol21_effective_dedup_v1/source_suite.json"
DEFAULT_OUTPUT_ROOT = BASE_INPUT_ROOT / "scenarios/staging/held33_native_power_refresh"
DEFAULT_TRIAGE_REPORT = BASE_INPUT_ROOT / "reports/held33_native_power_refresh_triage.json"
DEFAULT_SURVIVOR_SUITE = (
    BASE_INPUT_ROOT
    / "reports/held33_native_power_refresh_current_20260813_run1/"
    "survivor_source_suite.json"
)
FULL_PROTOCOL21_STAGE_OUTPUTS = {
    "preflight": "protocol2_v21_working_set_preflight.json",
    "behavioral": "behavioral_calibration_protocol2_v21.json",
    "source_consumption": "source_consumption_protocol2_v21.json",
    "task_contracts": "task_contracts_protocol2_v21.json",
    "complexity": "complexity_protocol2_v21.json",
    "observed_reference_depth": "observed_reference_depth_protocol2_v21.json",
    "strategy_depth": "strategy_depth_protocol2_v21.json",
    "source_grounded": "source_grounded_protocol2_v21.json",
    "agentic_contract": "agentic_core_contract_protocol2_v21.json",
    "materialize_core": "refined_core_selection_protocol2_v21.json",
    "release_coverage": "release_coverage_protocol2_v21.json",
    "readiness": "protocol2_v21_core_readiness.json",
}

TARGET_SCENARIO_IDS = (
    (
        "power_grid/distribution_volt_var/deep_planning/medium/"
        "power_grid__distribution_volt_var_native_state_loss__deep_planning__"
        "high__native_cigre_high_s73__d6e07c9a__relabel_v1"
    ),
    (
        "power_grid/distribution_volt_var_oberrhein/deep_planning/high/"
        "power_grid__distribution_volt_var_native_state_loss__deep_planning__"
        "extreme__native_oberrhein_extreme_s74__98002b89__relabel_v1"
    ),
    (
        "power_grid/opendss_ieee13_volt_var/time_pressure/basic/"
        "opendss_ieee13_capacitor_voltage_dynamic_s42"
    ),
    (
        "power_grid/opendss_fresh_feeders_volt_var/deep_planning/high/"
        "opendss_fresh_ieee123_volt_var_basic_s42_native_response_high_"
        "tap_cap_windows"
    ),
)


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    """Create a JSON result exactly once; never overwrite prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_immutable_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)


def wait_for_stable_implementation(
    *,
    minimum_stable_seconds: float = 60.0,
    maximum_wait_seconds: float = 300.0,
    poll_seconds: float = 20.0,
    identity_fn: Callable[[], dict[str, Any]] | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Require one unchanged implementation hash for a bounded time window."""
    if minimum_stable_seconds < 60.0:
        raise ValueError("candidate replay requires a >=60s stability window")
    if maximum_wait_seconds < minimum_stable_seconds:
        raise ValueError("maximum wait must cover the stability window")
    if poll_seconds <= 0:
        raise ValueError("poll seconds must be positive")
    get_identity = identity_fn or (lambda: implementation_identity(REPO_ROOT))
    started = monotonic_fn()
    stable_since = started
    identity = get_identity()
    current_hash = str(identity.get("implementation_tree_sha256") or "")
    if not current_hash:
        raise ValueError("implementation identity is missing tree hash")
    hash_sequence = [current_hash]
    observations = [
        {"elapsed_seconds": 0.0, "implementation_tree_sha256": current_hash}
    ]
    while True:
        now = monotonic_fn()
        stable_seconds = now - stable_since
        elapsed_seconds = now - started
        if stable_seconds >= minimum_stable_seconds:
            return {
                "implementation_tree_sha256": current_hash,
                "stable_seconds": round(stable_seconds, 6),
                "elapsed_seconds": round(elapsed_seconds, 6),
                "hash_sequence": hash_sequence,
                "observations": observations,
            }
        if elapsed_seconds >= maximum_wait_seconds:
            raise TimeoutError(
                "implementation tree did not remain stable for "
                f"{minimum_stable_seconds:.0f}s"
            )
        sleep_for = min(
            poll_seconds,
            minimum_stable_seconds - stable_seconds,
            maximum_wait_seconds - elapsed_seconds,
        )
        sleep_fn(sleep_for)
        observed_at = monotonic_fn()
        observed = get_identity()
        observed_hash = str(observed.get("implementation_tree_sha256") or "")
        observations.append(
            {
                "elapsed_seconds": round(observed_at - started, 6),
                "implementation_tree_sha256": observed_hash,
            }
        )
        if observed_hash != current_hash:
            current_hash = observed_hash
            stable_since = observed_at
            hash_sequence.append(current_hash)


def _absolute(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else BASE_INPUT_ROOT / path


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    return (
        resolved.relative_to(root).as_posix()
        if resolved.is_relative_to(root)
        else resolved.as_posix()
    )


def _index(rows: list[object]) -> dict[str, dict[str, Any]]:
    return {
        str(row["scenario_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("scenario_id")
    }


def build_exact_blocker_triage() -> dict[str, Any]:
    """Select four independent source/backend rows for an as-is fresh filter."""
    plan = _load_json(HELD_PLAN)
    suite = _load_json(BASE_SUITE)
    held_by_id = _index(list(plan.get("held_rows") or []))
    suite_by_id = _index(list(suite.get("scenarios") or []))
    domain_spec = get_domain_spec("power_grid")
    selected: list[dict[str, Any]] = []
    for scenario_id in TARGET_SCENARIO_IDS:
        held = held_by_id.get(scenario_id)
        source_row = suite_by_id.get(scenario_id)
        if held is None or source_row is None:
            raise ValueError(f"held/source row missing: {scenario_id}")
        path = _absolute(str(source_row.get("path") or ""))
        body = _load_yaml(path)
        signature = domain_spec.scenario_signature(body, int(body.get("seed") or 42))
        capability = get_backend_capability(str(body.get("backend_kind") or ""))
        blocker_groups = held.get("blockers") or {}
        prior_blockers = sorted(
            f"{group}:{code}"
            for group, codes in blocker_groups.items()
            for code in codes or []
        )
        source_contract = body.get("source_contract") or {}
        runtime_inputs = [str(value) for value in source_contract.get("runtime_input") or []]
        derivation_inputs = [
            str(value) for value in source_contract.get("derivation_input") or []
        ]
        missing_runtime_inputs = [
            value for value in runtime_inputs if not _absolute(value).is_file()
        ]
        source_denominator_key = str(
            (source_row.get("case_ledger") or {}).get("source_denominator_key")
            or source_row.get("source_denominator_key")
            or source_row.get("source_key")
            or ""
        )
        selected.append(
            {
                "scenario_id": scenario_id,
                "scenario_signature": str(body.get("scenario_signature") or ""),
                "scenario_signature_current": bool(
                    body.get("scenario_signature") == signature
                ),
                "absolute_path": path.resolve().as_posix(),
                "path": _display_path(path),
                "scenario_sha256": _sha256(path),
                "backend_kind": capability.backend_kind,
                "backend_formal_core_allowed": capability.formal_core_allowed,
                "backend_runtime_fidelity": capability.runtime_fidelity,
                "source_consumption_mode": capability.source_consumption_mode,
                "source_contract_present": bool(runtime_inputs or derivation_inputs),
                "runtime_inputs": runtime_inputs,
                "derivation_inputs": derivation_inputs,
                "missing_runtime_inputs": missing_runtime_inputs,
                "source_denominator_key": source_denominator_key,
                "difficulty_level": str(body.get("difficulty_level") or ""),
                "horizon_ticks": int(body.get("horizon_ticks") or 0),
                "prior_blocker_count": len(prior_blockers),
                "prior_blockers": prior_blockers,
                "prior_evidence_interpretation": (
                    "stale_or_incomplete_merged_evidence_requires_fresh_refresh"
                ),
                "candidate_action": "fresh_as_is_native_prefilter",
                "scenario_transform": "none",
                "core_admission_claimed": False,
            }
        )
    ready = all(
        row["scenario_signature_current"]
        and row["backend_formal_core_allowed"]
        and row["source_contract_present"]
        and not row["missing_runtime_inputs"]
        and row["source_denominator_key"]
        for row in selected
    ) and len({row["source_denominator_key"] for row in selected}) == len(selected)
    return {
        "schema_version": "held33_native_power_blocker_triage.v1",
        "status": "ready_for_fresh_native_prefilter" if ready else "held",
        "candidate_only": True,
        "selection_rule": (
            "after_excluding_delegated_jsplib_and_known_or_blocked_families,"
            "select_distinct_runtime-contained_native_power_sources_for_as-is_refresh"
        ),
        "excluded": {
            "datacenter_held3": "completed_by_separate_tail_refine",
            "jsplib_swv06": "owned_by_parallel_jsplib_known_groups_repair",
            "microgrid_las_vegas_portland": "known_survivors_owned_elsewhere",
            "pglib_uc_56": "native_safety_task_blocker_owned_elsewhere",
            "sumo365": "native_headroom_blocker_owned_elsewhere",
        },
        "n_selected": len(selected),
        "selected": selected,
        "required_next_gate": (
            "fresh_preflight_behavioral_source_consumption_task_contracts"
        ),
    }


def materialize_refresh_suite(
    *, output_root: Path = DEFAULT_OUTPUT_ROOT
) -> dict[str, Any]:
    """Copy the selected YAMLs byte-identically into a candidate-only suite."""
    out = output_root.resolve()
    allowed = (REPO_ROOT / "scenarios/staging").resolve()
    if not out.is_relative_to(allowed):
        raise ValueError("refresh output must stay under scenarios/staging")
    triage = build_exact_blocker_triage()
    if triage["status"] != "ready_for_fresh_native_prefilter":
        raise ValueError("triage is not ready for native prefilter")
    base = _load_json(BASE_SUITE)
    rows_by_id = _index(list(base.get("scenarios") or []))
    suite_rows: list[dict[str, Any]] = []
    scenario_artifacts: list[dict[str, Any]] = []
    for triage_row in triage["selected"]:
        scenario_id = str(triage_row["scenario_id"])
        source_path = Path(str(triage_row["absolute_path"]))
        backend = str(triage_row["backend_kind"])
        candidate_path = out / backend / source_path.name
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, candidate_path)
        source_sha256 = _sha256(source_path)
        candidate_sha256 = _sha256(candidate_path)
        if source_sha256 != candidate_sha256:
            raise ValueError(f"candidate copy drift: {scenario_id}")
        row = copy.deepcopy(rows_by_id[scenario_id])
        row["path"] = _display_path(candidate_path)
        row["candidate_gate"] = {
            "status": "requires_fresh_native_prefilter",
            "reason_codes": [
                "prior_merged_evidence_stale_or_incomplete",
                "scenario_copied_byte_identically",
            ],
        }
        row["protocol21_lineage"] = {
            "status": "candidate_refresh",
            "ready": False,
            "scenario_transform": "none",
            "source_identity_preserved": True,
            "requires_full_protocol21_replay": True,
        }
        suite_rows.append(row)
        scenario_artifacts.append(
            {
                "scenario_id": scenario_id,
                "source_path": source_path.resolve().as_posix(),
                "candidate_path": candidate_path.resolve().as_posix(),
                "source_sha256": source_sha256,
                "candidate_sha256": candidate_sha256,
                "scenario_transform": "none",
                "terminal_disposition": "held_repair",
            }
        )
    suite = {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "working_set",
        "selection_policy": "quality_maximal_v1",
        "constraints": {
            "candidate_only": True,
            "formal_evaluation_ready": False,
            "model_outcomes_used_for_filtering": False,
            "one_per_effective_source_identity": True,
            "scenario_transform": "none",
        },
        "leaderboard_eligible": False,
        "release_ready": False,
        "n_scenarios": len(suite_rows),
        "scenarios": suite_rows,
    }
    suite_path = out / "source_suite.json"
    suite_path.write_text(
        json.dumps(suite, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ledger = {
        "schema_version": "held33_native_power_refresh_ledger.v1",
        "status": "pending_fresh_native_prefilter",
        "candidate_only": True,
        "core_admission_claimed": False,
        "n_scenarios": len(scenario_artifacts),
        "scenarios": scenario_artifacts,
        "source_suite_path": suite_path.resolve().as_posix(),
        "source_suite_sha256": _sha256(suite_path),
    }
    ledger_path = out / "refresh_ledger.json"
    ledger_path.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **ledger,
        "source_suite_path": suite_path.resolve().as_posix(),
        "refresh_ledger_path": ledger_path.resolve().as_posix(),
    }


def _artifact_index(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return _index(
        list(
            artifact.get("results")
            or artifact.get("rows")
            or artifact.get("scenarios")
            or []
        )
    )


def _stage_failure_codes(stage: str, row: dict[str, Any]) -> list[str]:
    if stage == "behavioral":
        failed_checks = sorted(
            str(key)
            for key, value in (row.get("checks") or {}).items()
            if value is False
        )
        if failed_checks:
            return [f"{stage}:{value}" for value in failed_checks]
    values = (
        row.get("blockers")
        or row.get("failure_reasons")
        or ([row.get("reason_code")] if row.get("reason_code") else None)
        or [row.get("status") or "failed"]
    )
    return [f"{stage}:{value}" for value in values]


def summarize_prefilter(
    *,
    preflight: dict[str, Any],
    behavioral: dict[str, Any],
    source_consumption: dict[str, Any],
    task_contracts: dict[str, Any],
    implementation_tree_sha256_start: str,
    implementation_tree_sha256_end: str,
) -> dict[str, Any]:
    """Give every input exactly one terminal prefilter state, fail-closed."""
    artifacts = {
        "preflight": preflight,
        "behavioral": behavioral,
        "source_consumption": source_consumption,
        "task_contracts": task_contracts,
    }
    stable = implementation_tree_sha256_start == implementation_tree_sha256_end
    bound = stable and all(
        str(artifact.get("implementation_tree_sha256") or "")
        == implementation_tree_sha256_start
        for artifact in artifacts.values()
    )
    indexes = {name: _artifact_index(value) for name, value in artifacts.items()}
    results: list[dict[str, Any]] = []
    for scenario_id in TARGET_SCENARIO_IDS:
        blockers: list[str] = []
        stage_status: dict[str, str] = {}
        for stage, rows in indexes.items():
            row = rows.get(scenario_id)
            status = str((row or {}).get("status") or "missing")
            behavioral_status = str(
                (indexes["behavioral"].get(scenario_id) or {}).get("status")
                or "missing"
            )
            if stage == "task_contracts" and row is None and behavioral_status != "passed":
                stage_status[stage] = "not_run_behavioral_failed"
                continue
            stage_status[stage] = status
            if row is None:
                blockers.append(f"{stage}:missing_terminal_row")
            elif stage == "preflight":
                if (row.get("fatal_blockers") or []) or status not in {
                    "passed",
                    "runtime_pending",
                }:
                    blockers.append(f"{stage}:{status}")
            elif status != "passed":
                blockers.extend(_stage_failure_codes(stage, row))
        terminal = "passed" if not blockers else "held_repair"
        if not bound:
            terminal = "held_runtime"
            blockers = ["implementation_tree_drift"]
        results.append(
            {
                "scenario_id": scenario_id,
                "terminal_state": terminal,
                "stage_status": stage_status,
                "blockers": sorted(set(blockers)),
            }
        )
    n_passed = sum(row["terminal_state"] == "passed" for row in results)
    if not bound:
        status = "held"
    elif n_passed == len(results):
        status = "passed"
    elif n_passed:
        status = "partial_survivors"
    else:
        status = "held"
    return {
        "schema_version": "held33_native_power_prefilter_summary.v1",
        "status": status,
        "implementation_tree_sha256_start": implementation_tree_sha256_start,
        "implementation_tree_sha256_end": implementation_tree_sha256_end,
        "implementation_tree_bound": bound,
        "n_expected": len(results),
        "n_passed": n_passed,
        "n_held": len(results) - n_passed,
        "blockers": [] if bound else ["implementation_tree_drift"],
        "results": results,
        "full_protocol21_allowed": bool(bound and n_passed == len(results)),
    }


def build_exact_survivor_suite(
    *,
    source_suite: dict[str, Any],
    summary: dict[str, Any],
    implementation_tree_sha256: str,
) -> dict[str, Any]:
    """Build a candidate-only suite from rows passing every bounded gate."""
    terminal = {
        str(row.get("scenario_id")): str(row.get("terminal_state") or "")
        for row in summary.get("results") or []
        if isinstance(row, dict)
    }
    rows = [
        copy.deepcopy(row)
        for row in source_suite.get("scenarios") or []
        if isinstance(row, dict)
        and terminal.get(str(row.get("scenario_id"))) == "passed"
    ]
    for row in rows:
        row["candidate_gate"] = {
            "status": "bounded_native_prefilter_passed",
            "implementation_tree_sha256": implementation_tree_sha256,
            "core_admission_claimed": False,
        }
        row["protocol21_lineage"] = {
            **dict(row.get("protocol21_lineage") or {}),
            "status": "candidate_prefilter_survivor",
            "ready": False,
            "requires_full_protocol21_replay": True,
        }
    return {
        "schema_version": "protocol2.1-working-set-v1",
        "status": "candidate_prefilter_survivors",
        "candidate_only": True,
        "leaderboard_eligible": False,
        "release_ready": False,
        "implementation_tree_sha256": implementation_tree_sha256,
        "constraints": {
            "candidate_only": True,
            "formal_evaluation_ready": False,
            "model_outcomes_used_for_filtering": False,
            "scenario_transform": "none",
        },
        "n_scenarios": len(rows),
        "scenarios": rows,
    }


def build_full_protocol21_source_suite(
    *,
    survivor_suite: dict[str, Any],
    survivor_source_path: Path,
    survivor_source_sha256: str,
    implementation_tree_sha256: str,
) -> dict[str, Any]:
    """Promote one exact prefilter survivor to a runnable candidate suite."""
    rows = survivor_suite.get("scenarios")
    if (
        survivor_suite.get("status") != "candidate_prefilter_survivors"
        or survivor_suite.get("candidate_only") is not True
        or not isinstance(rows, list)
        or len(rows) != 1
    ):
        raise ValueError("exactly one candidate prefilter survivor is required")
    row = rows[0]
    if (
        not isinstance(row, dict)
        or row.get("scenario_id") != TARGET_SCENARIO_IDS[2]
        or (row.get("candidate_gate") or {}).get("status")
        != "bounded_native_prefilter_passed"
    ):
        raise ValueError("the natural IEEE13 survivor is not bound")
    runnable = copy.deepcopy(survivor_suite)
    runnable["status"] = "working_set"
    runnable["selection_policy"] = "exact_prefilter_survivor_full_protocol21_v1"
    runnable["implementation_tree_sha256"] = implementation_tree_sha256
    runnable["input_bindings"] = {
        "prefilter_survivor": {
            "path": _display_path(survivor_source_path),
            "sha256": survivor_source_sha256,
        }
    }
    runnable["constraints"] = {
        **dict(runnable.get("constraints") or {}),
        "candidate_only": True,
        "formal_evaluation_ready": False,
        "fresh_union_replay_required": True,
        "physical_sources_are_inference_clusters": True,
    }
    return runnable


def build_exact_selection_source_pair(
    *,
    source_suite: dict[str, Any],
    source_suite_path: Path,
    source_suite_sha256: str,
    selection: dict[str, Any],
    selection_path: Path,
    selection_sha256: str,
    implementation_tree_sha256: str,
) -> dict[str, Any]:
    """Bind the exact selected row and source accepted by the union builder."""
    source_rows = source_suite.get("scenarios") or []
    selected_rows = selection.get("scenarios") or []
    if (
        source_suite.get("status") != "working_set"
        or len(source_rows) != 1
        or selection.get("status") != "protocol21_core_candidate"
        or selection.get("n_selected") != 1
        or len(selected_rows) != 1
    ):
        raise ValueError("a one-row Protocol-2.1 selection is required")
    identity = lambda row: (  # noqa: E731
        str(row.get("scenario_id") or ""),
        str(row.get("scenario_signature") or ""),
    )
    if identity(source_rows[0]) != identity(selected_rows[0]):
        raise ValueError("selection/source identity mismatch")
    binding = (
        (selection.get("input_bindings") or {}).get("source_suite") or {}
    ).get("sha256")
    if binding != source_suite_sha256:
        raise ValueError("selection/source hash mismatch")
    return {
        "schema_version": "protocol21-selected-source-pair.v1",
        "status": "selected_candidate_source_pair",
        "candidate_only": True,
        "core_or_release_mutated": False,
        "implementation_tree_sha256": implementation_tree_sha256,
        "n_selected": 1,
        "scenario_identity": {
            "scenario_id": identity(source_rows[0])[0],
            "scenario_signature": identity(source_rows[0])[1],
        },
        "source_suite": {
            "path": _display_path(source_suite_path),
            "sha256": source_suite_sha256,
        },
        "selection": {
            "path": _display_path(selection_path),
            "sha256": selection_sha256,
        },
        "physical_source_policy": "shared_cluster_allowed_in_inference",
        "required_next_step": "selected_union_builder_then_fresh_union_replay",
    }


def _full_stage_summary(
    *, result_root: Path, pipeline_manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    records = {
        str(row.get("name") or ""): row
        for row in pipeline_manifest.get("stages") or []
        if isinstance(row, dict)
    }
    summaries: list[dict[str, Any]] = []
    for name, filename in FULL_PROTOCOL21_STAGE_OUTPUTS.items():
        path = result_root / filename
        artifact = _load_json(path) if path.is_file() else {}
        rows = artifact.get("results") or artifact.get("scenarios") or []
        row_statuses: dict[str, int] = {}
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict):
                status = str(
                    row.get("status")
                    or row.get("terminal_state")
                    or row.get("disposition")
                    or "unspecified"
                )
                row_statuses[status] = row_statuses.get(status, 0) + 1
        record = records.get(name) or {}
        summaries.append(
            {
                "stage": name,
                "return_code": record.get("return_code"),
                "artifact_present": path.is_file(),
                "artifact_path": _display_path(path),
                "artifact_sha256": _sha256(path) if path.is_file() else None,
                "artifact_status": artifact.get("status"),
                "complete": artifact.get("complete"),
                "n_expected": artifact.get("n_expected"),
                "n_completed": artifact.get("n_completed"),
                "n_selected": artifact.get("n_selected"),
                "row_statuses": row_statuses,
                "implementation_tree_sha256": (
                    artifact.get("implementation_tree_sha256")
                    or record.get("implementation_tree_sha256")
                ),
            }
        )
    return summaries


def run_full_protocol21_survivor(
    *,
    survivor_source_path: Path,
    result_root: Path,
    stability_window_seconds: float = 60.0,
    maximum_wait_seconds: float = 600.0,
    poll_seconds: float = 20.0,
    sample_timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Run one exact survivor through all twelve candidate-only gates."""
    result_root = result_root.resolve()
    if not result_root.is_relative_to((REPO_ROOT / "reports").resolve()):
        raise ValueError("full replay output must stay under reports")
    if result_root.exists():
        raise FileExistsError(result_root)
    stability = wait_for_stable_implementation(
        minimum_stable_seconds=stability_window_seconds,
        maximum_wait_seconds=maximum_wait_seconds,
        poll_seconds=poll_seconds,
    )
    required_hash = str(stability["implementation_tree_sha256"])
    survivor_source_path = survivor_source_path.resolve()
    survivor_sha256 = _sha256(survivor_source_path)
    runnable = build_full_protocol21_source_suite(
        survivor_suite=_load_json(survivor_source_path),
        survivor_source_path=survivor_source_path,
        survivor_source_sha256=survivor_sha256,
        implementation_tree_sha256=required_hash,
    )
    result_root.mkdir(parents=True, exist_ok=False)
    runnable_path = result_root / "survivor_source_suite.json"
    _write_immutable_json(runnable_path, runnable)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_protocol21_core_pipeline.py"),
        "--source-suite",
        str(runnable_path),
        "--release-dir",
        str(result_root),
        "--workers",
        "1",
        "--sample-timeout-seconds",
        str(sample_timeout_seconds),
        "--expected-count",
        "1",
        "--execute",
    ]
    before = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    if before != required_hash:
        raise RuntimeError("full_protocol21:implementation_tree_drift_before_run")
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    _write_immutable_text(result_root / "pipeline.stdout.txt", completed.stdout)
    _write_immutable_text(result_root / "pipeline.stderr.txt", completed.stderr)
    end_hash = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    pipeline_manifest_path = result_root / "protocol2_v21_pipeline_manifest.json"
    pipeline_manifest = (
        _load_json(pipeline_manifest_path)
        if pipeline_manifest_path.is_file()
        else {"status": "missing", "stages": []}
    )
    gate_summaries = _full_stage_summary(
        result_root=result_root, pipeline_manifest=pipeline_manifest
    )
    stage_names = [
        str(row.get("name") or "")
        for row in pipeline_manifest.get("stages") or []
        if isinstance(row, dict)
    ]
    stage_hashes_bound = all(
        row.get("implementation_tree_sha256") == required_hash
        for row in pipeline_manifest.get("stages") or []
        if isinstance(row, dict)
    )
    full_stage_coverage = stage_names == list(FULL_PROTOCOL21_STAGE_OUTPUTS)
    runtime_bound = bool(
        before == required_hash
        and end_hash == required_hash
        and stage_hashes_bound
        and full_stage_coverage
        and completed.returncode in {0, 4}
    )
    selection_path = result_root / FULL_PROTOCOL21_STAGE_OUTPUTS["materialize_core"]
    selection = _load_json(selection_path) if selection_path.is_file() else {}
    selected = bool(
        runtime_bound
        and selection.get("status") == "protocol21_core_candidate"
        and selection.get("n_selected") == 1
    )
    pair_path = result_root / "selected_source_pair.json"
    if selected:
        pair = build_exact_selection_source_pair(
            source_suite=runnable,
            source_suite_path=runnable_path,
            source_suite_sha256=_sha256(runnable_path),
            selection=selection,
            selection_path=selection_path,
            selection_sha256=_sha256(selection_path),
            implementation_tree_sha256=required_hash,
        )
        _write_immutable_json(pair_path, pair)
    coverage = next(
        row for row in gate_summaries if row["stage"] == "release_coverage"
    )
    terminal = {
        "schema_version": "held33-ieee13-full-protocol21.v1",
        "status": (
            "selected_candidate_pending_union"
            if selected
            else "held_scientific"
            if runtime_bound
            else "held_runtime"
        ),
        "candidate_only": True,
        "core_or_release_mutated": False,
        "implementation_tree_sha256_start": required_hash,
        "implementation_tree_sha256_end": end_hash,
        "implementation_tree_bound": runtime_bound,
        "stability_window": stability,
        "pipeline_return_code": completed.returncode,
        "pipeline_manifest_status": pipeline_manifest.get("status"),
        "full_stage_coverage": full_stage_coverage,
        "stage_order": stage_names,
        "gates": gate_summaries,
        "selection": {
            "selected": selected,
            "n_selected": selection.get("n_selected"),
            "n_rejected": selection.get("n_rejected"),
            "disposition_counts": selection.get("disposition_counts"),
        },
        "isolated_domain_release_coverage": {
            "artifact_status": coverage.get("artifact_status"),
            "diagnostic_only": True,
            "not_a_scientific_failure": True,
        },
        "source": {
            "prefilter_survivor_path": _display_path(survivor_source_path),
            "prefilter_survivor_sha256": survivor_sha256,
            "runnable_source_suite_path": _display_path(runnable_path),
            "runnable_source_suite_sha256": _sha256(runnable_path),
        },
        "selected_source_pair": (
            {
                "path": _display_path(pair_path),
                "sha256": _sha256(pair_path),
            }
            if selected
            else None
        ),
        "blockers": (
            []
            if selected
            else [
                "protocol21_scientific_selection_failed"
                if runtime_bound
                else "implementation_or_stage_drift"
            ]
        ),
    }
    _write_immutable_json(result_root / "terminal_summary.json", terminal)
    artifact_hashes = {
        path.relative_to(result_root).as_posix(): _sha256(path)
        for path in sorted(result_root.rglob("*"))
        if path.is_file() and path.name != "immutable_evidence_manifest.json"
    }
    _write_immutable_json(
        result_root / "immutable_evidence_manifest.json",
        {
            "schema_version": "immutable-candidate-evidence-root.v1",
            "status": "frozen",
            "candidate_only": True,
            "implementation_tree_sha256": required_hash,
            "artifacts": artifact_hashes,
        },
    )
    return terminal


def _run_stage(
    *,
    name: str,
    argv: list[str],
    output: Path,
    required_implementation_tree_sha256: str,
) -> dict[str, Any]:
    before = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    if before != required_implementation_tree_sha256:
        raise RuntimeError(f"{name}:implementation_tree_drift_before_stage")
    completed = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    after = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    if after != required_implementation_tree_sha256:
        raise RuntimeError(f"{name}:implementation_tree_drift_after_stage")
    if completed.returncode != 0:
        raise RuntimeError(f"{name}:process_exit_{completed.returncode}")
    artifact = _load_json(output)
    if artifact.get("implementation_tree_sha256") != required_implementation_tree_sha256:
        raise RuntimeError(f"{name}:artifact_implementation_tree_drift")
    return artifact


def _runtime_hold_summary(
    *,
    required_hash: str,
    current_hash: str,
    blocker: str,
) -> dict[str, Any]:
    return {
        "schema_version": "held33_native_power_prefilter_summary.v1",
        "status": "held",
        "implementation_tree_sha256_start": required_hash,
        "implementation_tree_sha256_end": current_hash,
        "implementation_tree_bound": False,
        "n_expected": len(TARGET_SCENARIO_IDS),
        "n_passed": 0,
        "n_held": len(TARGET_SCENARIO_IDS),
        "blockers": [blocker],
        "results": [
            {
                "scenario_id": scenario_id,
                "terminal_state": "held_runtime",
                "stage_status": {},
                "blockers": [blocker],
            }
            for scenario_id in TARGET_SCENARIO_IDS
        ],
        "full_protocol21_allowed": False,
    }


def run_bounded_refresh(
    *,
    result_root: Path,
    stability_window_seconds: float = 60.0,
    maximum_wait_seconds: float = 300.0,
    poll_seconds: float = 20.0,
    workers: int = 4,
    sample_timeout_seconds: int = 900,
) -> dict[str, Any]:
    """Run four current-tree candidate gates and freeze one immutable result."""
    result_root = result_root.resolve()
    if not result_root.is_relative_to((REPO_ROOT / "reports").resolve()):
        raise ValueError("bounded replay output must stay under reports")
    if result_root.exists():
        raise FileExistsError(result_root)
    stability = wait_for_stable_implementation(
        minimum_stable_seconds=stability_window_seconds,
        maximum_wait_seconds=maximum_wait_seconds,
        poll_seconds=poll_seconds,
    )
    required_hash = str(stability["implementation_tree_sha256"])
    triage = build_exact_blocker_triage()
    if triage["status"] != "ready_for_fresh_native_prefilter":
        raise ValueError("current triage is not ready for bounded replay")
    result_root.mkdir(parents=True, exist_ok=False)
    source_suite_path = DEFAULT_OUTPUT_ROOT / "source_suite.json"
    source_suite = _load_json(source_suite_path)
    outputs = {
        "preflight": result_root / "protocol2_v21_working_set_preflight.json",
        "behavioral": result_root / "behavioral_calibration_protocol2_v21.json",
        "source_consumption": result_root / "source_consumption_protocol2_v21.json",
        "task_contracts": result_root / "task_contracts_protocol2_v21.json",
    }
    python = sys.executable
    commands = {
        "preflight": [
            python,
            str(REPO_ROOT / "scripts/preflight_protocol21_working_set.py"),
            "--source-suite",
            str(source_suite_path),
            "--output",
            str(outputs["preflight"]),
            "--expected-count",
            str(len(TARGET_SCENARIO_IDS)),
            "--require-source-consumption-adapters",
            "--require-formal-core-backends",
            "--exercise-source-adapters",
        ],
        "behavioral": [
            python,
            str(REPO_ROOT / "scripts/calibrate_core_candidate.py"),
            "--suite",
            str(source_suite_path),
            "--output",
            str(outputs["behavioral"]),
            "--workers",
            str(workers),
            "--sample-timeout-seconds",
            str(sample_timeout_seconds),
            "--cache-dir",
            str(result_root / "cache/behavioral"),
        ],
        "source_consumption": [
            python,
            str(REPO_ROOT / "scripts/audit_protocol21_source_consumption.py"),
            "--suite",
            str(source_suite_path),
            "--behavioral",
            str(outputs["behavioral"]),
            "--output",
            str(outputs["source_consumption"]),
        ],
        "task_contracts": [
            python,
            str(REPO_ROOT / "scripts/calibrate_task_contracts.py"),
            "--suite",
            str(source_suite_path),
            "--output",
            str(outputs["task_contracts"]),
            "--agent",
            "oracle_offline",
            "--fallback-agents",
            "greedy_heuristic",
            "--workers",
            str(workers),
            "--sample-timeout-seconds",
            str(sample_timeout_seconds),
            "--eligible-results",
            str(outputs["behavioral"]),
        ],
    }
    artifacts: dict[str, dict[str, Any]] = {}
    try:
        for name in (
            "preflight",
            "behavioral",
            "source_consumption",
            "task_contracts",
        ):
            artifacts[name] = _run_stage(
                name=name,
                argv=commands[name],
                output=outputs[name],
                required_implementation_tree_sha256=required_hash,
            )
    except RuntimeError as exc:
        current_hash = implementation_identity(REPO_ROOT)[
            "implementation_tree_sha256"
        ]
        summary = _runtime_hold_summary(
            required_hash=required_hash,
            current_hash=current_hash,
            blocker=str(exc),
        )
        _write_immutable_json(result_root / "terminal_summary.json", summary)
        _write_immutable_json(
            result_root / "run_manifest.json",
            {
                "schema_version": "held33_native_power_bounded_refresh_run.v1",
                "status": "held_runtime",
                "stability_window": stability,
                "source_suite_path": _display_path(source_suite_path),
                "source_suite_sha256": _sha256(source_suite_path),
                "completed_stages": list(artifacts),
                "terminal_summary_sha256": _sha256(
                    result_root / "terminal_summary.json"
                ),
            },
        )
        return summary
    end_hash = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    summary = summarize_prefilter(
        preflight=artifacts["preflight"],
        behavioral=artifacts["behavioral"],
        source_consumption=artifacts["source_consumption"],
        task_contracts=artifacts["task_contracts"],
        implementation_tree_sha256_start=required_hash,
        implementation_tree_sha256_end=end_hash,
    )
    survivor_suite = build_exact_survivor_suite(
        source_suite=source_suite,
        summary=summary,
        implementation_tree_sha256=required_hash,
    )
    _write_immutable_json(result_root / "terminal_summary.json", summary)
    _write_immutable_json(result_root / "survivor_source_suite.json", survivor_suite)
    _write_immutable_json(
        result_root / "run_manifest.json",
        {
            "schema_version": "held33_native_power_bounded_refresh_run.v1",
            "status": "complete",
            "candidate_only": True,
            "core_admission_claimed": False,
            "stability_window": stability,
            "source_suite_path": _display_path(source_suite_path),
            "source_suite_sha256": _sha256(source_suite_path),
            "stage_artifacts": {
                name: {
                    "path": _display_path(path),
                    "sha256": _sha256(path),
                }
                for name, path in outputs.items()
            },
            "terminal_summary_sha256": _sha256(
                result_root / "terminal_summary.json"
            ),
            "survivor_source_suite_sha256": _sha256(
                result_root / "survivor_source_suite.json"
            ),
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--triage-report", type=Path, default=DEFAULT_TRIAGE_REPORT)
    parser.add_argument("--execute-bounded", action="store_true")
    parser.add_argument("--execute-full-survivor", action="store_true")
    parser.add_argument(
        "--survivor-source-suite", type=Path, default=DEFAULT_SURVIVOR_SUITE
    )
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--stability-window-seconds", type=float, default=60.0)
    parser.add_argument("--maximum-wait-seconds", type=float, default=300.0)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample-timeout-seconds", type=int, default=900)
    args = parser.parse_args()
    if args.execute_bounded and args.execute_full_survivor:
        parser.error("choose only one execute mode")
    if args.execute_full_survivor:
        if args.result_root is None:
            parser.error("--execute-full-survivor requires --result-root")
        terminal = run_full_protocol21_survivor(
            survivor_source_path=args.survivor_source_suite,
            result_root=args.result_root,
            stability_window_seconds=args.stability_window_seconds,
            maximum_wait_seconds=args.maximum_wait_seconds,
            poll_seconds=args.poll_seconds,
            sample_timeout_seconds=args.sample_timeout_seconds,
        )
        print(
            json.dumps(
                {
                    "status": terminal["status"],
                    "selected": terminal["selection"]["selected"],
                    "implementation_tree_bound": terminal[
                        "implementation_tree_bound"
                    ],
                    "result_root": str(args.result_root),
                },
                sort_keys=True,
            )
        )
        return 0 if terminal["implementation_tree_bound"] else 2
    if args.execute_bounded:
        if args.result_root is None:
            parser.error("--execute-bounded requires --result-root")
        summary = run_bounded_refresh(
            result_root=args.result_root,
            stability_window_seconds=args.stability_window_seconds,
            maximum_wait_seconds=args.maximum_wait_seconds,
            poll_seconds=args.poll_seconds,
            workers=args.workers,
            sample_timeout_seconds=args.sample_timeout_seconds,
        )
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "n_passed": summary["n_passed"],
                    "n_held": summary["n_held"],
                    "result_root": str(args.result_root),
                },
                sort_keys=True,
            )
        )
        return 0 if summary["implementation_tree_bound"] else 2
    start_hash = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    triage = build_exact_blocker_triage()
    artifacts = materialize_refresh_suite(output_root=args.output_root)
    end_hash = implementation_identity(REPO_ROOT)["implementation_tree_sha256"]
    triage["implementation_tree_sha256_start"] = start_hash
    triage["implementation_tree_sha256_end"] = end_hash
    triage["implementation_tree_stable"] = start_hash == end_hash
    triage["materialized_artifacts"] = {
        "source_suite_path": artifacts["source_suite_path"],
        "source_suite_sha256": artifacts["source_suite_sha256"],
        "refresh_ledger_path": artifacts["refresh_ledger_path"],
    }
    if start_hash != end_hash:
        triage["status"] = "held"
        triage["blockers"] = ["implementation_tree_drift"]
    report_path = args.triage_report.resolve()
    if not report_path.is_relative_to((REPO_ROOT / "reports").resolve()):
        raise ValueError("triage report must stay under reports")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(triage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(triage, indent=2, sort_keys=True))
    return 0 if triage["status"] == "ready_for_fresh_native_prefilter" else 2


if __name__ == "__main__":
    raise SystemExit(main())
