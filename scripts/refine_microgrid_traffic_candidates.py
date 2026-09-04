#!/usr/bin/env python3
"""Run bounded Microgrid prefilters and reconcile Traffic candidate evidence.

The output is an exact, candidate-only terminal ledger.  It deliberately keeps
the frozen Core immutable and uses a compact admission screen: distinct source
identity, locked/consumed native source, deterministic replay, beneficial
native loss, safe task-completing control, and complete terminal execution.
Full Protocol replay remains a later admission step.

SUMO365 dates are never advanced from the existing native ledger unless that
ledger proves beneficial full-network loss.  Old RESCO evidence is retained as
useful evidence but is marked stale when its implementation-tree binding does
not match the current tree.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import required_semantics  # noqa: E402
from domains.microgrid.backends.pymgrid_backend import (  # noqa: E402
    PymgridEconomicDispatchBackend,
)
from domains.microgrid.native_tools import (  # noqa: E402
    _h_apply,
    _h_connect_pcc,
    _h_dispatch_genset,
)
from domains.microgrid.seeds.schema import MicrogridLoad, Perturbation  # noqa: E402
from scripts.build_microgrid_held_refine import build as build_lv_refine  # noqa: E402
from scripts.build_microgrid_native_state_loss_candidates import (  # noqa: E402
    build as build_native_state_loss,
)

DEFAULT_CORE = (
    REPO_ROOT
    / "reports/protocol21_pending_union_fresh_current_20260811_realtraffic"
    / "refined_core_selection_protocol2_v21.json"
)
DEFAULT_LV_SUITE = (
    REPO_ROOT / "reports/microgrid_held_refine_current_20260814/source_suite.json"
)
DEFAULT_LV_BEHAVIORAL = (
    REPO_ROOT
    / "reports/track_a_underrepresented_20260812/microgrid/protocol21_a00df"
    / "behavioral_calibration_protocol2_v21.json"
)
DEFAULT_SUMO365 = REPO_ROOT / "reports/sumo365_native_tls_candidate_9date_v3/ledger.json"
DEFAULT_RESCO = (
    REPO_ROOT / "reports/protocol21_expansion/resco_replacement_calibration_v1.json"
)
DEFAULT_RESCO_4X4 = REPO_ROOT / "reports/traffic_resco_4x4_held_v1.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "reports/microgrid_traffic_candidate_refine_20260814_v2/terminal_ledger.json"
)

REQUIRED_NATIVE_GATES = (
    "source_lock",
    "source_consumption",
    "determinism",
    "beneficial_native_loss",
    "safe_reference",
    "task_completion",
    "terminal_integrity",
    "high_or_extreme_long_horizon",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _core_identity_index(core_suite: dict[str, Any]) -> dict[str, set[str]]:
    index = {
        "scenario_id": set(),
        "scenario_signature": set(),
        "source_key": set(),
        "source_denominator_key": set(),
    }
    for row in core_suite.get("scenarios") or []:
        if not isinstance(row, dict):
            raise ValueError("frozen Core scenarios must contain objects")
        ledger = row.get("case_ledger") or {}
        values = {
            "scenario_id": row.get("scenario_id"),
            "scenario_signature": row.get("scenario_signature"),
            "source_key": row.get("source_key"),
            "source_denominator_key": row.get("source_denominator_key")
            or ledger.get("source_denominator_key"),
        }
        for key, value in values.items():
            if isinstance(value, str) and value:
                index[key].add(value)
    return index


def _enrich_resco_source_identities(calibration: dict[str, Any]) -> dict[str, Any]:
    """Recover effective-source keys omitted by older calibration reports."""
    enriched = copy.deepcopy(calibration)
    for result in enriched.get("results") or []:
        if result.get("source_denominator_key"):
            continue
        raw_path = result.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            continue
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(body, dict):
            continue
        backend = body.get("backend_config") or {}
        source_denominator_key = body.get("source_denominator_key") or backend.get(
            "source_denominator_key"
        )
        if isinstance(source_denominator_key, str) and source_denominator_key:
            result["source_denominator_key"] = source_denominator_key
    return enriched


def _matches_core(row: dict[str, Any], index: dict[str, set[str]]) -> bool:
    for key in index:
        value = row.get(key)
        if isinstance(value, str) and value and value in index[key]:
            return True
    return False


def _terminal(
    *,
    source_id: str,
    domain: str,
    backend_kind: str,
    source_family: str,
    disposition: str,
    blockers: list[str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "domain": domain,
        "backend_kind": backend_kind,
        "source_family": source_family,
        "work_state": "terminal",
        "disposition": disposition,
        "candidate_only": True,
        "core_admission_claimed": False,
        "blockers": sorted(set(blockers)),
        "evidence": evidence,
    }


def _seed_from_body(body: dict[str, Any]) -> SimpleNamespace:
    provenance = SimpleNamespace(files=list((body.get("provenance") or {}).get("files") or []))
    return SimpleNamespace(
        horizon_ticks=int(body["horizon_ticks"]),
        tick_minutes=int(body["tick_minutes"]),
        seed=int(body["seed"]),
        seed_id=str(body["seed_id"]),
        backend_config=body["backend_config"],
        load_assignments=[MicrogridLoad(**raw) for raw in body["load_assignments"]],
        perturbations=[Perturbation(**raw) for raw in body["perturbations"]],
        provenance=provenance,
    )


def _run_native_state_loss(body: dict[str, Any], *, respond: bool) -> dict[str, Any]:
    seed = _seed_from_body(body)
    backend = PymgridEconomicDispatchBackend()
    backend.reset(seed)
    recipe = body["backend_config"]["response_window_recipe"]
    event_tick = int(recipe["event_trigger_tick"])
    recovery_tick = int(recipe["recovery_opportunity_tick"])
    restore_tick = int(recipe["restoration_opportunity_tick"])
    max_discharge = float(body["backend_config"]["battery"]["max_discharge_mw"])
    genset_max = float(body["backend_config"]["genset"]["max_mw"])
    env = SimpleNamespace(evidence=None)
    dispatch_battery = _h_apply(backend, env, "set_battery_dispatch")
    dispatch_genset = _h_dispatch_genset(backend)
    connect_pcc = _h_connect_pcc(backend)
    controls: list[dict[str, Any]] = []

    for tick in range(seed.horizon_ticks):
        context = SimpleNamespace(tick=tick)
        if respond and tick == event_tick - 1:
            result = dispatch_genset(
                {"genset_id": "genset0", "p_mw": genset_max}, context
            )
            controls.append({"tick": tick, "tool": "dispatch_genset", "result": result})
        if respond and tick == recovery_tick:
            result = dispatch_battery(
                {"battery_id": "batt0", "p_mw": -max_discharge}, context
            )
            controls.append(
                {"tick": tick, "tool": "set_battery_dispatch", "result": result}
            )
        if respond and tick == restore_tick:
            result = connect_pcc({"connect": True}, context)
            controls.append({"tick": tick, "tool": "connect_pcc", "result": result})
        backend.tick(tick)

    records = backend.scoring_records()
    trace = backend.protocol21_source_trace()
    return {
        "costs": backend.ground_truth_costs(),
        "records": records,
        "trace": trace,
        "controls": controls,
    }


def _task_loss(records: list[dict[str, Any]]) -> float:
    return sum(
        abs(float(row.get("balance_error_mw") or 0.0)) * 200.0
        + float(row.get("shed_penalty") or 0.0)
        for row in records
    )


def _native_state_loss_result(body: dict[str, Any]) -> dict[str, Any]:
    baseline = _run_native_state_loss(body, respond=False)
    reference = _run_native_state_loss(body, respond=True)
    replay = _run_native_state_loss(body, respond=True)
    baseline_loss = _task_loss(baseline["records"])
    reference_loss = _task_loss(reference["records"])
    task = body["backend_config"]["native_state_loss_task"]
    source_lock = body["backend_config"]["source_lock"]
    trace = reference["trace"]
    records = reference["records"]
    source_files = [REPO_ROOT / path for path in source_lock.get("files") or []]
    controls_ok = bool(reference["controls"]) and all(
        (control.get("result") or {}).get("_status") not in {"error", "unsupported"}
        for control in reference["controls"]
    )
    expected_ticks = list(range(int(body["horizon_ticks"])))
    terminal_integrity = (
        len(records) == len(expected_ticks)
        and [int(row.get("tick", -1)) for row in records] == expected_ticks
    )
    improvement = baseline_loss - reference_loss
    gates = {
        "source_lock": bool(source_files)
        and all(path.is_file() for path in source_files)
        and bool(source_lock.get("window_sha256"))
        and bool((body.get("provenance") or {}).get("commit")),
        "source_consumption": trace.get("status") == "passed"
        and trace.get("source_state_effect_observed") is True
        and trace.get("consumed_window_sha256") == source_lock.get("window_sha256")
        and trace.get("consumption_ticks") == expected_ticks,
        "determinism": reference["costs"] == replay["costs"]
        and reference["records"] == replay["records"]
        and (reference["trace"] or {}).get("trace_semantic_digest")
        == (replay["trace"] or {}).get("trace_semantic_digest"),
        "beneficial_native_loss": baseline_loss
        >= float(task["minimum_baseline_task_loss"])
        and improvement >= float(task["minimum_task_loss_reduction"]),
        "safe_reference": all(
            int(row.get(key) or 0) == 0
            for row in records
            for key in ("n_overloads", "n_voltage_violations", "n_disconnected_lines")
        )
        and not any(row.get("done") is True for row in records),
        "task_completion": controls_ok
        and improvement >= float(task["minimum_task_loss_reduction"]),
        "terminal_integrity": terminal_integrity,
        "high_or_extreme_long_horizon": body.get("difficulty_level") in {"high", "extreme"}
        and int(body["horizon_ticks"]) >= 24
        and len(body.get("perturbations") or []) >= 2
        and any(event.get("hidden") is True for event in body.get("perturbations") or []),
    }
    return {
        "scenario_id": body["scenario_id"],
        "scenario_signature": body["scenario_signature"],
        "source_denominator_key": body["backend_config"]["source_denominator_key"],
        "backend_kind": body["backend_kind"],
        "difficulty_level": body["difficulty_level"],
        "horizon_ticks": body["horizon_ticks"],
        "gates": gates,
        "metrics": {
            "baseline_native_task_loss": round(baseline_loss, 6),
            "reference_native_task_loss": round(reference_loss, 6),
            "native_task_loss_improvement": round(improvement, 6),
            "control_ticks": [row["tick"] for row in reference["controls"]],
            "control_tools": [row["tool"] for row in reference["controls"]],
            "reference_terminal_tick": records[-1]["tick"] if records else None,
            "source_trace_semantic_digest": trace.get("trace_semantic_digest"),
            "source_window_sha256": source_lock.get("window_sha256"),
        },
    }


def _lv_rows(
    report: dict[str, Any],
    *,
    core_index: dict[str, set[str]],
    implementation_tree_sha256: str,
    run_tree_stable: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    policy = report.get("policy") or {}
    for outcome in report.get("outcomes") or []:
        source_id = str(outcome["scenario_id"])
        upstream_disposition = str(outcome.get("disposition") or "held_repair")
        if upstream_disposition == "secondary_duplicate" or _matches_core(
            {"scenario_id": source_id}, core_index
        ):
            disposition = "secondary_duplicate"
            blockers = ["effective_source_already_represented_in_frozen_core"]
        elif upstream_disposition != "candidate_pending_full_protocol21":
            disposition = "held_repair"
            blockers = [str(value) for value in outcome.get("reason_codes") or []]
        else:
            selected = outcome.get("selected_probe") or {}
            behavioral = outcome.get("full_behavioral") or {}
            checks = behavioral.get("checks") or {}
            episodes = behavioral.get("episodes") or {}
            replay_episodes = [
                episodes.get("wait_only") or {},
                episodes.get("oracle_offline") or {},
            ]
            evidence_errors = []
            if report.get("admission_profile") != "quality_core_v2":
                evidence_errors.append("admission_profile_mismatch")
            if report.get("evaluation_semantics") != required_semantics():
                evidence_errors.append("evaluation_semantics_mismatch")
            if report.get("implementation_tree_sha256") != implementation_tree_sha256:
                evidence_errors.append("report_implementation_tree_mismatch")
            if report.get("implementation_tree_stable") is not True:
                evidence_errors.append("report_implementation_tree_unstable")
            if behavioral.get("admission_profile") != "quality_core_v2":
                evidence_errors.append("behavioral_admission_profile_mismatch")
            if behavioral.get("evaluation_semantics") != required_semantics():
                evidence_errors.append("behavioral_evaluation_semantics_mismatch")
            if behavioral.get("implementation_tree_sha256") != implementation_tree_sha256:
                evidence_errors.append("behavioral_implementation_tree_mismatch")
            if behavioral.get("implementation_tree_stable") is not True:
                evidence_errors.append("behavioral_implementation_tree_unstable")
            if not outcome.get("scenario_signature") or behavioral.get(
                "scenario_signature"
            ) != outcome.get("scenario_signature"):
                evidence_errors.append("behavioral_scenario_signature_mismatch")
            if not outcome.get("source_denominator_key") or behavioral.get(
                "source_denominator_key"
            ) != outcome.get("source_denominator_key"):
                evidence_errors.append("behavioral_source_identity_mismatch")
            gates = {
                "source_profile_preserved": policy.get("source_profile_unchanged") is True,
                "event_schedule_preserved": policy.get("event_schedule_unchanged") is True,
                "source_consumption": all(
                    (episode.get("source_consumption_evidence") or {}).get("status")
                    == "passed"
                    and (
                        episode.get("source_consumption_evidence") or {}
                    ).get("source_state_effect_observed")
                    is True
                    for episode in replay_episodes
                ),
                "determinism": checks.get("deterministic_replay") is True,
                "native_execution": checks.get("native_backend_executable") is True,
                "beneficial_native_loss": selected.get("screen_passed") is True
                and float(selected.get("oracle_cost") or 0.0)
                < float(selected.get("wait_cost") or 0.0),
                "safe_reference": checks.get("no_critical_native_regression") is True
                and float(selected.get("oracle_system_survival") or 0.0) == 100.0,
                "task_completion": selected.get("oracle_task_completed") is True
                and checks.get("task_contract_completed_by_reference") is True,
                "positive_headroom": checks.get("positive_decision_headroom") is True,
                "terminal_integrity": all(
                    (episode.get("terminal_integrity") or {}).get("release_ready") is True
                    for episode in replay_episodes
                ),
            }
            failed = [key for key, value in gates.items() if not value]
            if not run_tree_stable:
                disposition = "held_stale_evidence"
                blockers = ["implementation_tree_drift_during_native_prefilter"]
            elif evidence_errors:
                disposition = "held_stale_evidence"
                blockers = evidence_errors
            elif failed:
                disposition = "held_repair"
                blockers = [f"gate_failed:{key}" for key in failed]
            else:
                disposition = "candidate_prefilter"
                blockers = ["fresh_full_protocol_replay_pending"]
            outcome = {**outcome, "minimal_native_gates": gates}
        rows.append(
            _terminal(
                source_id=source_id,
                domain="microgrid",
                backend_kind="pandapower_lv",
                source_family="nrel_oedi_lv_voltage",
                disposition=disposition,
                blockers=blockers,
                evidence={"bounded_refine": outcome},
            )
        )
    return rows


def _native_microgrid_rows(
    results: list[dict[str, Any]],
    *,
    core_index: dict[str, set[str]],
    run_tree_stable: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        source_id = str(result["scenario_id"])
        if _matches_core(result, core_index):
            disposition = "secondary_duplicate"
            blockers = ["effective_source_already_represented_in_frozen_core"]
        elif not run_tree_stable:
            disposition = "held_stale_evidence"
            blockers = ["implementation_tree_drift_during_native_prefilter"]
        else:
            failed = [
                key for key in REQUIRED_NATIVE_GATES if (result.get("gates") or {}).get(key) is not True
            ]
            disposition = "candidate_prefilter" if not failed else "held_repair"
            blockers = (
                ["fresh_full_protocol_replay_pending"]
                if not failed
                else [f"gate_failed:{key}" for key in failed]
            )
        rows.append(
            _terminal(
                source_id=source_id,
                domain="microgrid",
                backend_kind=str(result.get("backend_kind") or "pymgrid_economic_dispatch"),
                source_family="nrel_oedi_ems_native_state_loss",
                disposition=disposition,
                blockers=blockers,
                evidence={"native_prefilter": result},
            )
        )
    return rows


def _sumo365_rows(
    ledger: dict[str, Any], *, core_index: dict[str, set[str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in ledger.get("results") or []:
        service_date = str(result["service_date"])
        source_id = f"traffic/sumo365/{service_date}"
        candidate = {
            "scenario_id": source_id,
            "source_denominator_key": f"sumo_ingolstadt_365:{service_date}",
        }
        gates = result.get("gates") or {}
        full_network_positive = gates.get("positive_headroom") is True and float(
            result.get("native_loss_improvement") or 0.0
        ) > 0.0
        if _matches_core(candidate, core_index):
            disposition = "secondary_duplicate"
            blockers = ["effective_source_already_represented_in_frozen_core"]
        elif not full_network_positive:
            disposition = "held_repair"
            blockers = sorted(
                set(str(value) for value in result.get("reason_codes") or [])
                | {"full_network_positive_headroom_unproven"}
            )
        else:
            disposition = "candidate_prefilter"
            blockers = ["fresh_full_protocol_replay_pending"]
        rows.append(
            _terminal(
                source_id=source_id,
                domain="traffic",
                backend_kind="sumo",
                source_family="sumo_ingolstadt_365",
                disposition=disposition,
                blockers=blockers,
                evidence={"native_prefilter": result, "headroom_scope": "full_network"},
            )
        )
    return rows


def _resco_rows(
    calibration: dict[str, Any],
    *,
    core_index: dict[str, set[str]],
    implementation_tree_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evidence_hash = calibration.get("implementation_tree_sha256")
    for result in calibration.get("results") or []:
        source_id = str(result["scenario_id"])
        if _matches_core(result, core_index):
            disposition = "secondary_duplicate"
            blockers = ["effective_source_already_represented_in_frozen_core"]
        else:
            checks = result.get("checks") or {}
            required = (
                "deterministic_replay",
                "native_backend_executable",
                "native_state_changing_leverage",
                "task_contract_completed_by_reference",
                "no_critical_native_regression",
                "positive_decision_headroom",
            )
            failed = [key for key in required if checks.get(key) is not True]
            if failed:
                disposition = "held_repair"
                blockers = [f"gate_failed:{key}" for key in failed]
                if evidence_hash != implementation_tree_sha256:
                    blockers.append("native_evidence_implementation_hash_stale")
            elif evidence_hash != implementation_tree_sha256:
                disposition = "held_stale_evidence"
                blockers = ["native_evidence_implementation_hash_stale"]
            else:
                disposition = "candidate_prefilter"
                blockers = ["fresh_full_protocol_replay_pending"]
        rows.append(
            _terminal(
                source_id=source_id,
                domain="traffic",
                backend_kind=str(result.get("backend_kind") or "sumo"),
                source_family="resco_locked_network",
                disposition=disposition,
                blockers=blockers,
                evidence={
                    "native_calibration": result,
                    "evidence_implementation_tree_sha256": evidence_hash,
                },
            )
        )
    return rows


def _resco_4x4_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in report.get("candidates") or []:
        network = str(candidate.get("network") or "unknown")
        probe = candidate.get("probe") or {}
        headroom = probe.get("headroom") or {}
        blockers = [str((candidate.get("hold") or {}).get("code") or "held_repair")]
        if headroom.get("status") != "passed" or float(
            headroom.get("absolute_improvement") or 0.0
        ) <= 0.0:
            blockers.append("full_network_positive_headroom_unproven")
        rows.append(
            _terminal(
                source_id=f"traffic/resco/{network}",
                domain="traffic",
                backend_kind="sumo",
                source_family="resco_locked_network",
                disposition="held_repair",
                blockers=blockers,
                evidence={"native_prefilter": candidate},
            )
        )
    return rows


def build_terminal_ledger(
    *,
    core_suite: dict[str, Any],
    lv_refine_report: dict[str, Any],
    microgrid_native_results: list[dict[str, Any]],
    sumo365_ledger: dict[str, Any],
    resco_calibration: dict[str, Any],
    resco_4x4: dict[str, Any],
    implementation_tree_sha256: str,
    run_tree_stable: bool,
) -> dict[str, Any]:
    core_index = _core_identity_index(core_suite)
    rows: list[dict[str, Any]] = []
    rows.extend(
        _lv_rows(
            lv_refine_report,
            core_index=core_index,
            implementation_tree_sha256=implementation_tree_sha256,
            run_tree_stable=run_tree_stable,
        )
    )
    rows.extend(
        _native_microgrid_rows(
            microgrid_native_results,
            core_index=core_index,
            run_tree_stable=run_tree_stable,
        )
    )
    rows.extend(_sumo365_rows(sumo365_ledger, core_index=core_index))
    rows.extend(
        _resco_rows(
            resco_calibration,
            core_index=core_index,
            implementation_tree_sha256=implementation_tree_sha256,
        )
    )
    rows.extend(_resco_4x4_rows(resco_4x4))
    rows.sort(key=lambda row: row["source_id"])
    identities = [str(row["source_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("candidate terminal ledger contains duplicate source identities")
    dispositions = Counter(str(row["disposition"]) for row in rows)
    domains = Counter(str(row["domain"]) for row in rows)
    return {
        "schema_version": "microgrid-traffic-candidate-refine.v2",
        "status": "complete_candidate_only",
        "candidate_only": True,
        "core_admission_claimed": False,
        "implementation_tree_sha256": implementation_tree_sha256,
        "run_tree_stable": run_tree_stable,
        "constraints": {
            "core_admission_profile": "quality_core_v2",
            "source_grounded_difficulty_deduplication": True,
            "redundant_agentic_checks_deduplicated": True,
            "strategy_depth_required_in_full_protocol": True,
            "source_consumption_required": True,
            "determinism_required": True,
            "native_beneficial_direction_headroom_required": True,
            "native_safety_and_task_completion_required": True,
            "effective_source_identity_required": True,
            "high_extreme_post_change_response_required": True,
        },
        "minimal_core_candidate_gate": {
            "purpose": "remove redundant reruns while preserving scientific validity",
            "required": list(REQUIRED_NATIVE_GATES),
            "full_protocol_replay_still_required_before_core": True,
            "model_performance_used_for_filtering": False,
            "random_events_added_for_admission": False,
            "frozen_core_modified": False,
        },
        "counts": {
            "terminal_rows": len(rows),
            "microgrid_rows": domains.get("microgrid", 0),
            "traffic_rows": domains.get("traffic", 0),
            **dict(sorted(dispositions.items())),
        },
        "rows": rows,
    }


def run(
    *,
    core_path: Path = DEFAULT_CORE,
    lv_suite_path: Path = DEFAULT_LV_SUITE,
    lv_behavioral_path: Path = DEFAULT_LV_BEHAVIORAL,
    sumo365_path: Path = DEFAULT_SUMO365,
    resco_path: Path = DEFAULT_RESCO,
    resco_4x4_path: Path = DEFAULT_RESCO_4X4,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    inputs = (
        core_path,
        lv_suite_path,
        lv_behavioral_path,
        sumo365_path,
        resco_path,
        resco_4x4_path,
    )
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    output_path = output_path.resolve()
    output_path.relative_to((REPO_ROOT / "reports").resolve())

    start_hash = str(implementation_identity()["implementation_tree_sha256"])
    lv_output_root = output_path.parent / "microgrid_lv_candidates"
    lv_report, lv_suite, lv_files = build_lv_refine(
        suite_path=lv_suite_path.resolve(),
        behavioral_path=lv_behavioral_path.resolve(),
        output_root=lv_output_root,
    )
    _native_report, native_files = build_native_state_loss(repo_root=REPO_ROOT)
    native_results = [
        _native_state_loss_result(body)
        for _path, body in sorted(native_files.items(), key=lambda item: str(item[0]))
    ]
    end_hash = str(implementation_identity()["implementation_tree_sha256"])
    if start_hash == end_hash and lv_report.get("implementation_tree_stable") is True:
        for path, body in lv_files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
        _write(lv_output_root / "refine_report.json", lv_report)
        _write(lv_output_root / "source_suite.json", lv_suite)
    report = build_terminal_ledger(
        core_suite=_load(core_path),
        lv_refine_report=lv_report,
        microgrid_native_results=native_results,
        sumo365_ledger=_load(sumo365_path),
        resco_calibration=_enrich_resco_source_identities(_load(resco_path)),
        resco_4x4=_load(resco_4x4_path),
        implementation_tree_sha256=end_hash,
        run_tree_stable=start_hash == end_hash,
    )
    report["run_binding"] = {
        "implementation_tree_sha256_start": start_hash,
        "implementation_tree_sha256_end": end_hash,
        "stable": start_hash == end_hash,
    }
    report["input_bindings"] = {
        path.name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}
        for path in inputs
    }
    _write(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--lv-suite", type=Path, default=DEFAULT_LV_SUITE)
    parser.add_argument("--lv-behavioral", type=Path, default=DEFAULT_LV_BEHAVIORAL)
    parser.add_argument("--sumo365", type=Path, default=DEFAULT_SUMO365)
    parser.add_argument("--resco", type=Path, default=DEFAULT_RESCO)
    parser.add_argument("--resco-4x4", type=Path, default=DEFAULT_RESCO_4X4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(
        core_path=args.core.resolve(),
        lv_suite_path=args.lv_suite.resolve(),
        lv_behavioral_path=args.lv_behavioral.resolve(),
        sumo365_path=args.sumo365.resolve(),
        resco_path=args.resco.resolve(),
        resco_4x4_path=args.resco_4x4.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
