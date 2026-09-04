#!/usr/bin/env python3
"""Compare deterministic mock-SUMO traces against recorded live-SUMO traces.

This is a non-release audit helper. It never promotes live SUMO to the scored
backend and never falls back from a missing live trace to the mock trace. The
intent is to make the mock-vs-live gap explicit before any future live-scoring
promotion work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import Action, ToolCall  # noqa: E402
from core.sidecar.sumo_sidecar import sumo_available  # noqa: E402
from domains.traffic.adapter import TrafficEnvironment  # noqa: E402
from domains.traffic.seeds.from_lust import build_traffic_seed  # noqa: E402

REPORT_SCOPE = "traffic_mock_live_fidelity_probe"
REBASELINE_REPORT_SCOPE = "traffic_fidelity_rebaseline_probe"
LIVE_ATTRIBUTION_REPORT_SCOPE = "traffic_live_attribution_probe"
TARGET_DECISION_REPORT_SCOPE = "traffic_fidelity_target_decision_review"
LIVE_CASE_LEDGER_PREVIEW_REPORT_SCOPE = "traffic_live_case_ledger_materializer_preview"
CORRIDOR_SCOPED_PROMOTION_PREVIEW_REPORT_SCOPE = (
    "traffic_corridor_scoped_live_promotion_preview"
)
CORRIDOR_SCOPED_LIVE_RUNNER_REPORT_SCOPE = "traffic_corridor_scoped_live_runner"
LIVE_REPLACEMENT_MATERIALIZATION_PLAN_SCOPE = (
    "traffic_live_replacement_materialization_plan"
)
LIVE_REPLACEMENT_PILOT_RELEASE_DRY_RUN_SCOPE = (
    "traffic_live_replacement_pilot_release_dry_run"
)
DEFAULT_OUTPUT = Path("reports") / "traffic_mock_live_fidelity_probe.json"
LIVE_TRACE_CAPTURE_ENV_GATE = "OPERATE_TRAFFIC_BACKEND_REAL=1"
FIDELITY_GATE_THRESHOLDS = {
    "aggregate_queue_abs_delta": 10.0,
    "aggregate_delay_minutes_abs_delta": 25.0,
    "per_corridor_delay_l1_minutes": 25.0,
    "per_corridor_delay_max_abs_delta_minutes": 10.0,
    "per_corridor_delay_changed_corridors": 3,
    "signal_program_mismatch_count": 0,
    "priority_outcome_l1_delta": 5.0,
    "spillback_proxy_abs_delta": 2.0,
}
REBASELINE_THRESHOLDS = {
    "aggregate_queue_abs_delta": FIDELITY_GATE_THRESHOLDS["aggregate_queue_abs_delta"],
    "aggregate_delay_minutes_abs_delta": FIDELITY_GATE_THRESHOLDS[
        "aggregate_delay_minutes_abs_delta"
    ],
    "aggregate_ratio_abs_delta": 0.10,
    "queue_to_delay_minutes_abs_delta": 0.01,
    "corridor_delay_coverage_min": 0.80,
    "live_controlled_lane_queue_share_min": 0.80,
}
LIVE_ATTRIBUTION_THRESHOLDS = {
    "network_to_corridor_delay_coverage_min": 0.80,
    "network_to_tls_controlled_queue_share_min": 0.80,
    "network_to_tls_controlled_vehicle_share_min": 0.80,
    "network_to_tls_controlled_lane_share_min": 0.80,
}


def build_traffic_mock_live_fidelity_report(
    *,
    mock_trace: dict[str, Any] | list[dict[str, Any]] | None = None,
    live_trace: dict[str, Any] | list[dict[str, Any]] | None = None,
    mock_trace_path: Path | str | None = None,
    live_trace_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build a machine-readable mock-vs-live fidelity report.

    ``mock_trace`` and ``live_trace`` are plain JSON-like trace summaries. The
    function also accepts paths to such JSON artifacts for CLI use. Supplying a
    mock trace without a live trace produces a blocked report, not a fabricated
    live result.
    """

    if mock_trace is None and mock_trace_path is not None:
        mock_trace = _load_json(Path(mock_trace_path))
    if live_trace is None and live_trace_path is not None:
        live_trace = _load_json(Path(live_trace_path))

    base = _base_report()
    blockers: list[str] = []
    if mock_trace is None:
        blockers.append("mock_trace_missing")
    if live_trace is None:
        blockers.append("live_trace_missing")
    if os.environ.get("OPERATE_TRAFFIC_BACKEND_REAL") != "1" and live_trace is None:
        blockers.insert(0, "live_runtime_not_gated")

    if blockers:
        status = (
            "blocked_live_runtime_not_gated"
            if "live_runtime_not_gated" in blockers
            else "blocked_missing_live_probe"
        )
        return {
            **base,
            "status": status,
            "executed_with_live_backend": False,
            "blocker_codes": sorted(dict.fromkeys(blockers)),
            "fidelity_metrics": {},
            "fidelity_target_split": _fidelity_target_split_without_live_trace(
                blockers
            ),
        }

    assert mock_trace is not None and live_trace is not None
    metrics = _compare_traces(mock_trace, live_trace)
    fidelity_gate = _fidelity_gate(metrics)
    target_split = _fidelity_target_split_from_metrics(metrics, fidelity_gate)
    split_gate = _split_fidelity_gate_from_target_split(target_split)
    return {
        **base,
        "status": "fidelity_probe_complete",
        "executed_with_live_backend": bool(
            _trace_dict(live_trace).get("executed_with_live_backend", True)
        ),
        "blocker_codes": [],
        "fidelity_metrics": metrics,
        "fidelity_gate": fidelity_gate,
        "rebaseline_diagnostics": _rebaseline_diagnostics(metrics, fidelity_gate),
        "fidelity_target_split": target_split,
        "split_fidelity_gate": split_gate,
        "mock_trace_summary": mock_trace,
        "live_trace_summary": live_trace,
    }


def build_traffic_mock_live_replay_report(
    *,
    family: str = "incident_response",
    difficulty_level: str = "basic",
    difficulty_mode: str = "time_pressure",
    seed: int = 42,
    n_ticks: int = 6,
    action_stream: list[Any] | None = None,
    run_live: bool = False,
) -> dict[str, Any]:
    """Replay the same action stream on mock SUMO and, when gated, live SUMO."""

    actions = (
        action_stream
        if action_stream is not None
        else _default_fidelity_action_stream(int(n_ticks))
    )
    mock_seed = build_traffic_seed(
        seed_id=f"fidelity/{family}/mock",
        family=family,
        seed=int(seed),
        difficulty_level=difficulty_level,
        difficulty_mode=difficulty_mode,
    )
    mock_seed.backend_kind = "mock_sumo"
    mock_trace = _replay_traffic_seed(
        scenario_config=mock_seed.to_dict(),
        seed=int(seed),
        action_stream=actions,
        n_ticks=int(n_ticks),
    )

    live_trace: dict[str, Any] | None = None
    if (
        run_live
        and os.environ.get("OPERATE_TRAFFIC_BACKEND_REAL") == "1"
        and sumo_available()
    ):
        live_seed = build_traffic_seed(
            seed_id=f"fidelity/{family}/live",
            family=family,
            seed=int(seed),
            difficulty_level=difficulty_level,
            difficulty_mode=difficulty_mode,
        )
        live_seed.backend_kind = "sumo"
        live_seed.backend_config = {
            **live_seed.backend_config,
            "backend_kind": "sumo",
        }
        live_trace = _replay_traffic_seed(
            scenario_config=live_seed.to_dict(),
            seed=int(seed),
            action_stream=actions,
            n_ticks=int(n_ticks),
        )

    report = build_traffic_mock_live_fidelity_report(
        mock_trace=mock_trace,
        live_trace=live_trace,
    )
    report.update(
        {
            "seed_request": {
                "family": family,
                "difficulty_level": difficulty_level,
                "difficulty_mode": difficulty_mode,
                "seed": int(seed),
                "n_ticks": int(n_ticks),
            },
            "mock_trace_summary": mock_trace,
            "live_trace_summary": live_trace,
            "live_trace_capture": _live_trace_capture_contract(
                family=family,
                difficulty_level=difficulty_level,
                difficulty_mode=difficulty_mode,
                seed=int(seed),
                n_ticks=int(n_ticks),
                run_live=bool(run_live),
                live_trace=live_trace,
            ),
            "run_live_requested": bool(run_live),
        }
    )
    return report


def _default_fidelity_action_stream(n_ticks: int) -> list[list[dict[str, Any]]]:
    actions: list[list[dict[str, Any]]] = []
    if int(n_ticks) <= 0:
        return actions
    actions.append(
        [
            {
                "name": "change_signal_plan",
                "args": {
                    "corridor": "hospital_access",
                    "program": "incident_relief",
                },
            }
        ]
    )
    actions.extend([[{"name": "wait"}] for _ in range(max(0, int(n_ticks) - 1))])
    return actions


def build_traffic_corridor_scoped_live_runner_report(
    *,
    target_decision_report: dict[str, Any] | None = None,
    target_decision_report_path: Path | str | None = None,
    family: str = "incident_response",
    difficulty_level: str = "basic",
    difficulty_mode: str = "time_pressure",
    seed: int = 42,
    n_ticks: int = 6,
    candidate_action_stream: list[Any] | None = None,
    release_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Generate same-seed live traces for the corridor-scoped preview gate.

    This runner is deliberately reports-only. It executes only when the real
    SUMO env gate and runtime are present, never falls back to mock SUMO, and
    then feeds the resulting candidate/wait/oracle-like/counterfactual traces
    into ``build_traffic_corridor_scoped_promotion_preview_report``.
    """

    if target_decision_report is None and target_decision_report_path is not None:
        target_decision_report = _load_json(Path(target_decision_report_path))

    base = {
        **_base_report(),
        "scope": CORRIDOR_SCOPED_LIVE_RUNNER_REPORT_SCOPE,
        "seed_request": {
            "family": family,
            "difficulty_level": difficulty_level,
            "difficulty_mode": difficulty_mode,
            "seed": int(seed),
            "n_ticks": int(n_ticks),
        },
        "writes_release_artifacts": False,
        "writes_scenario_yaml": False,
    }
    env_gated = os.environ.get("OPERATE_TRAFFIC_BACKEND_REAL") == "1"
    runtime_available = sumo_available() if env_gated else False
    if not env_gated or not runtime_available:
        status = (
            "blocked_env_gate_unset"
            if not env_gated
            else "blocked_sumo_runtime_unavailable"
        )
        preview = build_traffic_corridor_scoped_promotion_preview_report(
            target_decision_report=target_decision_report,
            release_dir=release_dir,
        )
        return {
            **base,
            "status": "blocked_live_runner_not_gated",
            "executed_with_live_backend": False,
            "blocker_codes": [status],
            "trace_status": {
                "candidate_live": status,
                "wait_live": status,
                "oracle_like_live": status,
                "counterfactual_live": status,
            },
            "live_trace_capture": {
                "env_gate": LIVE_TRACE_CAPTURE_ENV_GATE,
                "env_gate_satisfied": env_gated,
                "sumo_runtime_available": runtime_available,
                "silent_mock_fallback_allowed": False,
            },
            "promotion_preview": preview,
            "release_promotion_decision": _release_promotion_decision(
                live_scoring_allowed=False,
                case_ledger_preview_allowed=False,
                release_materializer_allowed=False,
                reason="live_runner_not_gated",
            ),
        }

    candidate_actions = (
        candidate_action_stream
        if candidate_action_stream is not None
        else _default_fidelity_action_stream(int(n_ticks))
    )
    wait_actions = _wait_action_stream(int(n_ticks))
    oracle_actions = _oracle_like_corridor_action_stream(
        target_decision_report=target_decision_report,
        candidate_action_stream=candidate_actions,
        n_ticks=int(n_ticks),
    )

    traces = {
        "candidate_live": _capture_live_trace(
            family=family,
            difficulty_level=difficulty_level,
            difficulty_mode=difficulty_mode,
            seed=int(seed),
            n_ticks=int(n_ticks),
            action_stream=candidate_actions,
            trace_role="candidate_live",
        ),
        "wait_live": _capture_live_trace(
            family=family,
            difficulty_level=difficulty_level,
            difficulty_mode=difficulty_mode,
            seed=int(seed),
            n_ticks=int(n_ticks),
            action_stream=wait_actions,
            trace_role="wait_live",
        ),
        "oracle_like_live": _capture_live_trace(
            family=family,
            difficulty_level=difficulty_level,
            difficulty_mode=difficulty_mode,
            seed=int(seed),
            n_ticks=int(n_ticks),
            action_stream=oracle_actions,
            trace_role="oracle_like_live",
        ),
        "counterfactual_live": _capture_live_trace(
            family=family,
            difficulty_level=difficulty_level,
            difficulty_mode=difficulty_mode,
            seed=int(seed),
            n_ticks=int(n_ticks),
            action_stream=wait_actions,
            trace_role="counterfactual_live",
        ),
    }
    _attach_counterfactual_score_evidence_ids(
        traces["candidate_live"], traces["counterfactual_live"]
    )
    target_decision_source = "provided_target_decision_report"
    if target_decision_report is None:
        target_decision_report = _derive_corridor_scoped_target_decision_from_trace(
            traces["candidate_live"]
        )
        target_decision_source = "derived_from_live_runner_traces"
    trace_status = {
        role: (
            "captured_live_trace"
            if trace.get("executed_with_live_backend") is True
            else "blocked_live_trace_missing"
        )
        for role, trace in traces.items()
    }
    preview = build_traffic_corridor_scoped_promotion_preview_report(
        target_decision_report=target_decision_report,
        live_trace=traces["candidate_live"],
        wait_trace=traces["wait_live"],
        oracle_trace=traces["oracle_like_live"],
        counterfactual_trace=traces["counterfactual_live"],
        release_dir=release_dir,
    )
    core_blockers = [
        blocker
        for blocker in preview.get("release_blockers", [])
        if blocker
        not in {
            "case_ledger_materialization_gate_not_run",
            "release_wrapper_materialization_gate_not_run",
        }
    ]
    return {
        **base,
        "status": (
            "live_runner_complete_release_blocked"
            if not core_blockers
            else "live_runner_complete_preview_blocked"
        ),
        "executed_with_live_backend": all(
            trace.get("executed_with_live_backend") is True for trace in traces.values()
        ),
        "blocker_codes": core_blockers,
        "trace_status": trace_status,
        "live_trace_capture": {
            "env_gate": LIVE_TRACE_CAPTURE_ENV_GATE,
            "env_gate_satisfied": True,
            "sumo_runtime_available": True,
            "silent_mock_fallback_allowed": False,
        },
        "traces": traces,
        "target_decision_source": target_decision_source,
        "promotion_preview": preview,
        "release_promotion_decision": preview.get("release_promotion_decision", {}),
    }


def build_traffic_rebaseline_probe_report(
    *,
    fidelity_report: dict[str, Any] | None = None,
    fidelity_report_path: Path | str | None = None,
    mock_trace: dict[str, Any] | list[dict[str, Any]] | None = None,
    live_trace: dict[str, Any] | list[dict[str, Any]] | None = None,
    mock_trace_path: Path | str | None = None,
    live_trace_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build a non-release report for live/mock rebaseline axes."""

    if fidelity_report is None and fidelity_report_path is not None:
        fidelity_report = _load_json(Path(fidelity_report_path))
    if mock_trace is None and mock_trace_path is not None:
        mock_trace = _load_json(Path(mock_trace_path))
    if live_trace is None and live_trace_path is not None:
        live_trace = _load_json(Path(live_trace_path))
    if fidelity_report:
        mock_trace = mock_trace or fidelity_report.get("mock_trace_summary")
        live_trace = live_trace or fidelity_report.get("live_trace_summary")

    base = _base_report()
    if mock_trace is None or live_trace is None:
        blockers = []
        if mock_trace is None:
            blockers.append("mock_trace_missing")
        if live_trace is None:
            blockers.append("live_trace_missing")
        return {
            **base,
            "scope": REBASELINE_REPORT_SCOPE,
            "status": "blocked_missing_trace",
            "blocker_codes": blockers,
            "release_promotion_implication": "blocked_missing_rebaseline_inputs",
            "thresholds": dict(REBASELINE_THRESHOLDS),
        }

    aggregate = _aggregate_rebaseline_alignment(mock_trace, live_trace)
    corridor = _corridor_rebaseline_attribution(mock_trace, live_trace)
    blockers = _rebaseline_blockers(aggregate, corridor)
    split = _fidelity_target_split_from_rebaseline(
        aggregate=aggregate,
        corridor=corridor,
        blockers=blockers,
    )
    return {
        **base,
        "scope": REBASELINE_REPORT_SCOPE,
        "status": "blocked_rebaseline_required" if blockers else "rebaseline_aligned",
        "blocker_codes": blockers,
        "release_promotion_implication": (
            "blocked_rebaseline_before_live_scoring"
            if blockers
            else "rebaseline_probe_passed"
        ),
        "aggregate_alignment": aggregate,
        "corridor_attribution": corridor,
        "fidelity_target_split": split,
        "source_fidelity_gate_status": (
            (fidelity_report or {}).get("fidelity_gate") or {}
        ).get("status"),
        "thresholds": dict(REBASELINE_THRESHOLDS),
    }


def build_traffic_live_attribution_probe_report(
    *,
    fidelity_report: dict[str, Any] | None = None,
    fidelity_report_path: Path | str | None = None,
    live_trace: dict[str, Any] | list[dict[str, Any]] | None = None,
    live_trace_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build a non-release report for live SUMO attribution coverage."""

    if fidelity_report is None and fidelity_report_path is not None:
        fidelity_report = _load_json(Path(fidelity_report_path))
    if live_trace is None and live_trace_path is not None:
        live_trace = _load_json(Path(live_trace_path))
    if fidelity_report:
        live_trace = live_trace or fidelity_report.get("live_trace_summary")

    base = _base_report()
    if live_trace is None:
        return {
            **base,
            "scope": LIVE_ATTRIBUTION_REPORT_SCOPE,
            "status": "blocked_missing_live_trace",
            "blocker_codes": ["live_trace_missing"],
            "release_promotion_implication": "blocked_missing_attribution_inputs",
            "thresholds": dict(LIVE_ATTRIBUTION_THRESHOLDS),
        }

    network = _live_network_totals(live_trace)
    corridor = _benchmark_corridor_attribution(live_trace, network)
    tls = _tls_controlled_lane_attribution(live_trace, network)
    mapping = _live_mapping_coverage(live_trace)
    blockers = _live_attribution_blockers(corridor, tls)
    measurement_basis = {
        "aggregate_queue_basis": "network_vehicle_count",
        "aggregate_delay_basis": "network_vehicle_count_x_tick_minutes",
        "corridor_delay_basis": "tls_controlled_lane_halting_integral",
        "mapping_coverage_basis": "sumo_tls_controlled_lanes_vs_network_lane_count",
        "basis_mismatch_if_undercovered": True,
    }
    return {
        **base,
        "scope": LIVE_ATTRIBUTION_REPORT_SCOPE,
        "status": (
            "blocked_attribution_rebaseline_required"
            if blockers
            else "attribution_probe_aligned"
        ),
        "blocker_codes": blockers,
        "release_promotion_implication": (
            "blocked_attribution_rebaseline_before_live_scoring"
            if blockers
            else "attribution_probe_passed"
        ),
        "network_totals": network,
        "benchmark_corridor_attribution": corridor,
        "tls_controlled_lane_attribution": tls,
        "mapping_coverage": mapping,
        "measurement_basis": measurement_basis,
        "fidelity_target_split": _fidelity_target_split_from_live_attribution(
            network=network,
            corridor=corridor,
            tls=tls,
            measurement_basis=measurement_basis,
            blockers=blockers,
        ),
        "recommended_next_action": (
            "separate_network_total_from_tls_controlled_corridor_target"
            if blockers
            else "keep_current_attribution_target"
        ),
        "thresholds": dict(LIVE_ATTRIBUTION_THRESHOLDS),
    }


def build_traffic_fidelity_target_decision_report(
    *,
    fidelity_report: dict[str, Any] | None = None,
    fidelity_report_path: Path | str | None = None,
    rebaseline_report: dict[str, Any] | None = None,
    rebaseline_report_path: Path | str | None = None,
    live_attribution_report: dict[str, Any] | None = None,
    live_attribution_report_path: Path | str | None = None,
) -> dict[str, Any]:
    """Choose the next non-release Traffic live-fidelity target path."""

    if fidelity_report is None and fidelity_report_path is not None:
        fidelity_report = _load_json(Path(fidelity_report_path))
    if rebaseline_report is None and rebaseline_report_path is not None:
        rebaseline_report = _load_json(Path(rebaseline_report_path))
    if live_attribution_report is None and live_attribution_report_path is not None:
        live_attribution_report = _load_json(Path(live_attribution_report_path))

    base = {**_base_report(), "scope": TARGET_DECISION_REPORT_SCOPE}
    if fidelity_report is None:
        return {
            **base,
            "status": "blocked_missing_inputs",
            "blocker_codes": ["fidelity_report_missing"],
            "release_promotion_decision": _release_promotion_decision(
                live_scoring_allowed=False,
                case_ledger_preview_allowed=False,
                release_materializer_allowed=False,
                reason="missing_fidelity_report",
            ),
            "next_actions": [
                "capture_live_trace_then_run_fidelity_target_decision_review"
            ],
        }

    rebaseline_report = rebaseline_report or build_traffic_rebaseline_probe_report(
        fidelity_report=fidelity_report
    )
    live_attribution_report = (
        live_attribution_report
        or build_traffic_live_attribution_probe_report(fidelity_report=fidelity_report)
    )
    target_assessment = _target_decision_assessment(
        fidelity_report=fidelity_report,
        rebaseline_report=rebaseline_report,
        live_attribution_report=live_attribution_report,
    )
    basis_mismatch = target_assessment["basis_mismatch_detected"] is True
    fidelity_passed = (fidelity_report.get("fidelity_gate") or {}).get("passed") is True
    blockers = _target_decision_blockers(
        target_assessment=target_assessment,
        basis_mismatch=basis_mismatch,
        fidelity_passed=fidelity_passed,
    )
    chosen_path = _target_decision_chosen_path(
        target_assessment=target_assessment,
        basis_mismatch=basis_mismatch,
        fidelity_passed=fidelity_passed,
    )
    decision_options = _target_decision_options(
        target_assessment=target_assessment,
        chosen_path=chosen_path,
    )
    return {
        **base,
        "status": (
            "decision_ready_live_scoring_blocked"
            if blockers
            else "decision_ready_fidelity_target_aligned"
        ),
        "blocker_codes": blockers,
        "chosen_path": chosen_path,
        "target_assessment": target_assessment,
        "decision_options": decision_options,
        "corridor_scoped_live_scoring_contract": (
            _corridor_scoped_live_scoring_contract(target_assessment)
        ),
        "release_promotion_decision": _release_promotion_decision(
            live_scoring_allowed=False if blockers else fidelity_passed,
            case_ledger_preview_allowed=bool(fidelity_report),
            release_materializer_allowed=False,
            reason=(
                "live_network_wide_denominator_not_represented"
                if "live_network_wide_scoring_denominator_not_represented" in blockers
                else (
                    "fidelity_target_split_required"
                    if basis_mismatch
                    else (
                        "fidelity_gate_blocked"
                        if blockers
                        else "fidelity_target_aligned_preview_only"
                    )
                )
            ),
        ),
        "next_actions": _target_decision_next_actions(
            target_assessment=target_assessment,
            chosen_path=chosen_path,
            blockers=blockers,
        ),
        "source_reports": {
            "fidelity_status": fidelity_report.get("status"),
            "fidelity_gate_status": (fidelity_report.get("fidelity_gate") or {}).get(
                "status"
            ),
            "rebaseline_status": rebaseline_report.get("status"),
            "live_attribution_status": live_attribution_report.get("status"),
        },
    }


def build_traffic_live_case_ledger_preview_report(
    *,
    target_decision_report: dict[str, Any] | None = None,
    target_decision_report_path: Path | str | None = None,
    live_runner_report: dict[str, Any] | None = None,
    live_runner_report_path: Path | str | None = None,
    release_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Preview live-scored Traffic case-ledger deltas without release writes."""

    if target_decision_report is None and target_decision_report_path is not None:
        target_decision_report = _load_json(Path(target_decision_report_path))
    if live_runner_report is None and live_runner_report_path is not None:
        live_runner_report = _load_json(Path(live_runner_report_path))

    release_path = (
        Path(release_dir) if release_dir is not None else _latest_traffic_release_dir()
    )
    release = _traffic_release_snapshot(release_path)
    traffic_primary = release["traffic_primary_rows"]
    live_runner_summary = _live_runner_summary(live_runner_report)
    live_runner_ready = live_runner_summary.get("ready_for_case_ledger_preview") is True
    promotion_preview = (
        live_runner_report.get("promotion_preview")
        if isinstance(live_runner_report, dict)
        else {}
    )
    corridor_gate_summary = _corridor_scoped_live_gate_summary(promotion_preview)
    target_decision = target_decision_report or {}
    if not target_decision and isinstance(promotion_preview, dict):
        target_decision = dict(promotion_preview.get("target_decision_summary") or {})
    target_release_decision = target_decision.get("release_promotion_decision") or {}
    target_blockers = [str(v) for v in target_decision.get("blocker_codes") or []]
    blocking_release_gates = (
        _live_case_ledger_blocking_gates_from_runner(live_runner_summary)
        if live_runner_ready
        else _live_case_ledger_blocking_gates(target_decision)
    )
    case_preview_allowed = (
        target_release_decision.get("case_ledger_preview_allowed") is True
        or target_decision.get("status")
        in {
            "decision_ready_live_scoring_blocked",
            "decision_ready_fidelity_target_aligned",
        }
        or live_runner_ready
    )
    live_scoring_allowed = (
        target_release_decision.get("live_scoring_allowed") is True
        and not blocking_release_gates
        and not live_runner_ready
    )
    release_materializer_allowed = (
        target_release_decision.get("release_materializer_allowed") is True
        and live_scoring_allowed
    )
    recommended_live_rows = (
        len(traffic_primary)
        if live_scoring_allowed and release_materializer_allowed
        else 0
    )
    candidates = [
        _live_case_ledger_candidate_preview(
            row,
            release=release,
            target_blockers=target_blockers,
            blocking_release_gates=blocking_release_gates,
            live_release_eligible=live_scoring_allowed and release_materializer_allowed,
            live_runner_evidence_ready=live_runner_ready,
            corridor_gate_summary=corridor_gate_summary,
        )
        for row in traffic_primary
    ]
    source_keys = sorted(
        {
            str(row.get("source_denominator_key"))
            for row in candidates
            if row.get("source_denominator_key")
        }
    )
    source_variant_keys = sorted(
        {
            f"{row.get('source_denominator_key')}::{row.get('decision_variant_key')}"
            for row in candidates
            if row.get("source_denominator_key") and row.get("decision_variant_key")
        }
    )
    status = (
        "blocked_missing_target_decision"
        if not target_decision and not live_runner_report
        else (
            "preview_ready_live_runner_release_blocked"
            if live_runner_ready
            else "preview_ready_live_scoring_blocked"
            if blocking_release_gates or not live_scoring_allowed
            else "preview_ready_release_materialization_blocked"
        )
    )
    return {
        **_base_report(),
        "scope": LIVE_CASE_LEDGER_PREVIEW_REPORT_SCOPE,
        "status": status,
        "writes_release_artifacts": False,
        "writes_scenario_yaml": False,
        "release_dir": str(release_path),
        "recommended_live_rows": recommended_live_rows,
        "blocking_release_gates": blocking_release_gates,
        "source_release": {
            "release_id": release["release_id"],
            "scoring_version": release["scoring_version"],
            "traffic_registry_rows": len(release["traffic_registry_rows"]),
            "traffic_primary_rows": len(traffic_primary),
            "traffic_core_rows": len(release["traffic_core_rows"]),
            "current_scored_backend": "mock_sumo",
        },
        "target_decision_summary": {
            "scope": target_decision.get("scope"),
            "status": target_decision.get("status"),
            "chosen_path": target_decision.get("chosen_path"),
            "blocker_codes": target_blockers,
        },
        "live_runner_summary": live_runner_summary,
        "corridor_scoped_live_gate_summary": corridor_gate_summary,
        "release_promotion_decision": _release_promotion_decision(
            live_scoring_allowed=live_scoring_allowed,
            case_ledger_preview_allowed=case_preview_allowed,
            release_materializer_allowed=release_materializer_allowed,
            reason=(
                "live_scoring_blocked_preview_only"
                if blocking_release_gates or not live_scoring_allowed
                else "live_case_ledger_preview_only"
            ),
        ),
        "candidate_rows_preview": candidates,
        "release_delta_if_unblocked": {
            "recommended_live_rows_now": recommended_live_rows,
            "candidate_live_primary_rows_if_unblocked": len(traffic_primary),
            "candidate_live_core_rows_if_unblocked": len(release["traffic_core_rows"]),
            "candidate_source_denominator_keys": source_keys,
            "candidate_source_variant_keys": source_variant_keys,
            "candidate_effective_source_delta_if_replacing_mock": 0,
            "candidate_physical_source_delta_if_replacing_mock": 0,
            "delta_basis": (
                "live scoring would replace the existing source-locked "
                "sumo_ingolstadt mock-scored Traffic rows; it must not pad a "
                "second denominator over the same net/family/level keys."
            ),
        },
        "validation": {
            "loaded_release_manifest": bool(release["release_id"]),
            "traffic_primary_rows_present": bool(traffic_primary),
            "candidate_signatures_unique": len(
                {
                    row.get("scenario_signature")
                    for row in candidates
                    if row.get("scenario_signature")
                }
            )
            == len(candidates),
            "candidate_structural_fingerprints_unique": len(
                {
                    row.get("structural_fingerprint")
                    for row in candidates
                    if row.get("structural_fingerprint")
                }
            )
            == len(candidates),
            "candidate_source_variant_keys_unique": (
                len(source_variant_keys) == len(candidates)
            ),
            "candidate_ledgers_present": all(
                row.get("dimension_applicability") for row in candidates
            ),
            "candidate_source_keys_present": all(
                row.get("source_denominator_key") for row in candidates
            ),
            "no_release_writes": True,
            "recommended_live_rows_zero_while_blocked": (
                recommended_live_rows == 0
                if blocking_release_gates or not live_scoring_allowed
                else True
            ),
        },
        "policy": {
            "non_release_artifact": True,
            "release_artifact_mutation_allowed": False,
            "scenario_yaml_mutation_allowed": False,
            "mock_as_live_evidence_allowed": False,
            "live_trace_required_for_live_scoring": True,
            "split_fidelity_gate_required_before_live_scoring": True,
            "signal_program_readback_required_before_live_scoring": True,
            "live_rows_replace_existing_mock_traffic_denominator": True,
        },
        "next_actions": _live_case_ledger_next_actions(blocking_release_gates),
    }


def build_traffic_live_replacement_materialization_plan_report(
    *,
    live_runner_report: dict[str, Any] | None = None,
    live_runner_report_path: Path | str | None = None,
    live_case_ledger_preview_report: dict[str, Any] | None = None,
    live_case_ledger_preview_report_path: Path | str | None = None,
    release_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Plan the next live-traffic release boundary without writing artifacts.

    This is deliberately stricter than the case-ledger preview. A single live
    runner can prove one corridor-scoped cell, but it cannot certify every
    Traffic row in the v0.8 denominator. The plan therefore authorizes at most a
    pilot materializer design and keeps full replacement blocked until per-row
    live evidence exists.
    """

    if live_runner_report is None and live_runner_report_path is not None:
        live_runner_report = _load_json(Path(live_runner_report_path))
    if (
        live_case_ledger_preview_report is None
        and live_case_ledger_preview_report_path is not None
    ):
        live_case_ledger_preview_report = _load_json(
            Path(live_case_ledger_preview_report_path)
        )

    release_path = (
        Path(release_dir) if release_dir is not None else _latest_traffic_release_dir()
    )
    release = _traffic_release_snapshot(release_path)
    if live_case_ledger_preview_report is None and isinstance(live_runner_report, dict):
        live_case_ledger_preview_report = build_traffic_live_case_ledger_preview_report(
            live_runner_report=live_runner_report,
            release_dir=release_path,
        )

    live_runner_summary = _live_runner_summary(live_runner_report)
    ledger_preview = live_case_ledger_preview_report or {}
    ledger_ready = ledger_preview.get("status") in {
        "preview_ready_live_runner_release_blocked",
        "preview_ready_live_scoring_blocked",
        "preview_ready_release_materialization_blocked",
    }
    ledger_materialization_only = set(
        str(v) for v in ledger_preview.get("blocking_release_gates") or []
    ).issubset(
        {
            "live_scoring_case_ledger_not_materialized",
            "real_release_materializer_not_run_into_release_wrapper",
        }
    )
    runner_ready = live_runner_summary.get("ready_for_case_ledger_preview") is True
    pilot_rows = _live_replacement_pilot_rows(
        live_runner_report=live_runner_report,
        ledger_preview=ledger_preview,
        release=release,
    )
    full_replacement_rows = _full_live_replacement_ready_rows(
        ledger_preview=ledger_preview,
        release=release,
    )
    blockers = _live_replacement_plan_blockers(
        runner_ready=runner_ready,
        ledger_ready=ledger_ready,
        ledger_materialization_only=ledger_materialization_only,
        pilot_rows=pilot_rows,
        full_replacement_rows=full_replacement_rows,
        live_runner_summary=live_runner_summary,
        ledger_preview=ledger_preview,
    )
    status = (
        "ready_for_authorization_pilot_only"
        if not blockers
        else "blocked_live_replacement_plan"
    )
    return {
        **_base_report(),
        "scope": LIVE_REPLACEMENT_MATERIALIZATION_PLAN_SCOPE,
        "status": status,
        "writes_release_artifacts": False,
        "writes_scenario_yaml": False,
        "release_dir": str(release_path),
        "source_release": {
            "release_id": release["release_id"],
            "scoring_version": release["scoring_version"],
            "traffic_registry_rows": len(release["traffic_registry_rows"]),
            "traffic_primary_rows": len(release["traffic_primary_rows"]),
            "traffic_core_rows": len(release["traffic_core_rows"]),
            "current_scored_backend": "mock_sumo",
        },
        "live_runner_summary": live_runner_summary,
        "case_ledger_preview_summary": {
            "scope": ledger_preview.get("scope"),
            "status": ledger_preview.get("status"),
            "recommended_live_rows": int(
                ledger_preview.get("recommended_live_rows") or 0
            ),
            "blocking_release_gates": list(
                ledger_preview.get("blocking_release_gates") or []
            ),
            "candidate_rows_previewed": len(
                ledger_preview.get("candidate_rows_preview") or []
            ),
        },
        "replacement_semantics": {
            "mode": "replace_existing_mock_traffic_denominator",
            "candidate_live_backend": "sumo",
            "current_scored_backend": "mock_sumo",
            "source_id": "sumo_ingolstadt",
            "live_corridor_denominator_key": _live_replacement_denominator_key(
                live_runner_report=live_runner_report,
                ledger_preview=ledger_preview,
            ),
            "network_wide_live_scoring_allowed": False,
            "mock_and_live_double_count_allowed": False,
            "effective_source_delta_if_replacing_mock": 0,
            "physical_source_delta_if_replacing_mock": 0,
        },
        "gates": _live_replacement_plan_gates(
            runner_ready=runner_ready,
            ledger_ready=ledger_ready,
            ledger_materialization_only=ledger_materialization_only,
            pilot_rows=pilot_rows,
            full_replacement_rows=full_replacement_rows,
            live_runner_summary=live_runner_summary,
            ledger_preview=ledger_preview,
        ),
        "authorization_recommendation": {
            "status": (
                "authorize_single_cell_pilot_design"
                if status == "ready_for_authorization_pilot_only"
                else "do_not_authorize_materialization"
            ),
            "release_artifact_mutation_allowed": False,
            "scenario_yaml_mutation_allowed": False,
            "release_materializer_allowed": False,
            "recommended_scope": (
                "single_cell_pilot_replacement_plan" if pilot_rows else "none"
            ),
            "pilot_rows_ready_for_authorization": len(pilot_rows),
            "full_traffic_replacement_rows_ready": len(full_replacement_rows),
            "full_replacement_blocker": (
                None
                if len(full_replacement_rows) == len(release["traffic_primary_rows"])
                else "per_row_live_runner_evidence_missing"
            ),
            "do_not_pad_denominator": True,
            "replacement_semantics": (
                "future live SUMO rows replace the existing mock_sumo Traffic "
                "rows for the same sumo_ingolstadt source keys; they must not "
                "add a second Traffic denominator."
            ),
        },
        "evidence_scope": _live_replacement_evidence_scope(live_runner_report),
        "pilot_replacement_rows": pilot_rows,
        "full_replacement_rows_ready": full_replacement_rows,
        "required_before_any_release_write": [
            "explicit_user_authorization_for_new_release_boundary",
            "implement_v0_9_live_traffic_materializer",
            "implement_v0_9_live_traffic_behavioral_audit_wrapper",
            "run_per_row_live_runner_or_documented_equivalence_gate",
            "materialize_case_ledgers_with_live_counterfactual_citations",
            "run_readiness_gate_for_new_release",
        ],
        "future_write_set_if_authorized": [
            "release/dt_sched_bench_v0_50_0/registry.json",
            "release/dt_sched_bench_v0_50_0/primary_suite.json",
            "release/dt_sched_bench_v0_50_0/core_suite.json",
            "release/dt_sched_bench_v0_50_0/manifest.json",
            "scenarios/releases/dt_sched_bench_v0_50_0/traffic/*",
            "scripts/audit_v0_9_live_traffic_behavioral.py",
            "scripts/check_v0_9_readiness.py",
            "docs/RELEASE_NOTES_v0.9.0.md",
        ],
        "audit_design_requirements": {
            "fresh_live_traffic_partitions_required": True,
            "v0_8_mock_sumo_fresh_audit_not_reused_for_live_rows": True,
            "inherited_power_grid_logistics_rows_may_reuse_v0_8_1_only_if_fingerprints_match": True,
            "counterfactual_replay_required_or_explicit_na": True,
            "scoring_version_review_required": True,
            "backend_descriptor_must_quote_live_scope": (
                "corridor-scoped live SUMO signal control over TLS-controlled "
                "corridors only; not network-wide live SUMO scoring and not "
                "vehicle routing."
            ),
        },
        "blocker_codes": blockers,
        "policy": {
            "non_release_artifact": True,
            "release_artifact_mutation_allowed": False,
            "scenario_yaml_mutation_allowed": False,
            "mock_as_live_evidence_allowed": False,
            "live_rows_replace_existing_mock_traffic_denominator": True,
            "single_runner_must_not_certify_all_rows": True,
        },
        "next_actions": _live_replacement_plan_next_actions(
            status=status, blockers=blockers, pilot_rows=pilot_rows
        ),
    }


def build_traffic_live_replacement_pilot_release_dry_run_report(
    *,
    replacement_plan_report: dict[str, Any] | None = None,
    replacement_plan_report_path: Path | str | None = None,
    release_dir: Path | str | None = None,
    target_release_id: str = "dt_sched_bench_v0_50_0",
) -> dict[str, Any]:
    """Build a release-shaped pilot skeleton without authorizing writes.

    The dry run is the last non-release gate before a human can authorize a
    real v0.9 materializer. It intentionally models replacement of the current
    mock-SUMO Traffic row, so the expected core/effective-source delta is zero.
    """

    if replacement_plan_report is None and replacement_plan_report_path is not None:
        replacement_plan_report = _load_json(Path(replacement_plan_report_path))

    release_path = (
        Path(release_dir) if release_dir is not None else _latest_traffic_release_dir()
    )
    release = _traffic_release_snapshot(release_path)
    plan = replacement_plan_report or {}
    pilot_rows = list(plan.get("pilot_replacement_rows") or [])
    plan_ready = plan.get("status") == "ready_for_authorization_pilot_only"
    replacement = plan.get("replacement_semantics") or {}
    replacement_not_addition = (
        replacement.get("mode") == "replace_existing_mock_traffic_denominator"
        and replacement.get("mock_and_live_double_count_allowed") is False
    )
    single_cell_scope = len(pilot_rows) == 1
    blockers = _live_replacement_pilot_dry_run_blockers(
        plan_ready=plan_ready,
        pilot_rows=pilot_rows,
        replacement_not_addition=replacement_not_addition,
    )
    status = (
        "ready_for_explicit_authorization"
        if plan_ready and single_cell_scope and replacement_not_addition
        else "blocked_pilot_release_dry_run"
    )
    replacement_rows = _live_replacement_pilot_dry_run_rows(
        pilot_rows=pilot_rows,
        target_release_id=target_release_id,
    )
    return {
        **_base_report(),
        "scope": LIVE_REPLACEMENT_PILOT_RELEASE_DRY_RUN_SCOPE,
        "status": status,
        "writes_release_artifacts": False,
        "writes_scenario_yaml": False,
        "release_dir": str(release_path),
        "source_release": {
            "release_id": release["release_id"],
            "scoring_version": release["scoring_version"],
            "traffic_registry_rows": len(release["traffic_registry_rows"]),
            "traffic_primary_rows": len(release["traffic_primary_rows"]),
            "traffic_core_rows": len(release["traffic_core_rows"]),
            "scored_backend": "mock_sumo",
        },
        "target_release": {
            "release_id": target_release_id,
            "release_type": "single_cell_live_traffic_replacement_pilot",
            "fresh_release_required": True,
            "inherits_non_traffic_rows_from": release["release_id"],
            "requires_new_live_traffic_behavioral_audit": True,
        },
        "expected_release_delta": {
            "registry": 0,
            "primary": 0,
            "core": 0,
            "effective_sources": 0,
            "physical_sources": 0,
            "count_policy": "replacement_not_addition",
            "rationale": (
                "The live pilot replaces one existing mock_sumo Traffic row "
                "over the same sumo_ingolstadt source denominator."
            ),
        },
        "replacement_rows": replacement_rows,
        "required_authorization": {
            "authorization_required": True,
            "exact_prompt": _live_replacement_pilot_authorization_prompt(
                replacement_rows
            ),
            "authorized_release_writes": False,
        },
        "gates": {
            "replacement_plan_gate": {
                "passed": bool(plan_ready),
                "status": "passed" if plan_ready else "blocked",
                "blocker_codes": []
                if plan_ready
                else ["replacement_plan_not_ready_for_pilot"],
            },
            "single_cell_scope_gate": {
                "passed": len(pilot_rows) == 1,
                "status": "passed" if len(pilot_rows) == 1 else "blocked",
                "pilot_rows": len(pilot_rows),
                "blocker_codes": []
                if len(pilot_rows) == 1
                else ["pilot_scope_not_single_cell"],
            },
            "double_count_prevention_gate": {
                "passed": bool(replacement_not_addition),
                "status": "passed" if replacement_not_addition else "blocked",
                "blocker_codes": []
                if replacement_not_addition
                else ["replacement_semantics_not_proven"],
            },
            "v0_9_materializer_gate": {
                "passed": False,
                "status": "blocked",
                "blocker_codes": ["v0_9_live_traffic_materializer_not_implemented"],
            },
            "v0_9_audit_wrapper_gate": {
                "passed": False,
                "status": "blocked",
                "blocker_codes": [
                    "v0_9_live_traffic_behavioral_audit_wrapper_not_implemented"
                ],
            },
        },
        "required_write_set_if_authorized": _live_replacement_pilot_write_set(
            target_release_id
        ),
        "forbidden_without_authorization": [
            "write_release_manifest",
            "write_scenario_yaml",
            "write_registry_primary_core",
            "materialize_release",
            "tag_release",
        ],
        "blocker_codes": blockers,
        "policy": {
            "non_release_artifact": True,
            "release_artifact_mutation_allowed": False,
            "scenario_yaml_mutation_allowed": False,
            "mock_and_live_double_count_allowed": False,
            "pilot_replaces_existing_core_row": True,
            "full_traffic_replacement_allowed": False,
        },
        "next_actions": _live_replacement_pilot_dry_run_next_actions(
            status=status, replacement_rows=replacement_rows
        ),
    }


def build_traffic_corridor_scoped_promotion_preview_report(
    *,
    target_decision_report: dict[str, Any] | None = None,
    target_decision_report_path: Path | str | None = None,
    live_trace: dict[str, Any] | list[dict[str, Any]] | None = None,
    live_trace_path: Path | str | None = None,
    wait_trace: dict[str, Any] | list[dict[str, Any]] | None = None,
    wait_trace_path: Path | str | None = None,
    oracle_trace: dict[str, Any] | list[dict[str, Any]] | None = None,
    oracle_trace_path: Path | str | None = None,
    counterfactual_trace: dict[str, Any] | list[dict[str, Any]] | None = None,
    counterfactual_trace_path: Path | str | None = None,
    release_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Preview a corridor-scoped live SUMO scoring promotion without writes.

    The preview intentionally separates a candidate live target from release
    materialization. It can pass the live evidence/headroom/counterfactual
    contract, but it still keeps ``live_scoring_allowed`` false until a later
    release wrapper writes case ledgers and suite artifacts.
    """

    if target_decision_report is None and target_decision_report_path is not None:
        target_decision_report = _load_json(Path(target_decision_report_path))
    if live_trace is None and live_trace_path is not None:
        live_trace = _load_json(Path(live_trace_path))
    if wait_trace is None and wait_trace_path is not None:
        wait_trace = _load_json(Path(wait_trace_path))
    if oracle_trace is None and oracle_trace_path is not None:
        oracle_trace = _load_json(Path(oracle_trace_path))
    if counterfactual_trace is None and counterfactual_trace_path is not None:
        counterfactual_trace = _load_json(Path(counterfactual_trace_path))

    release_path = (
        Path(release_dir) if release_dir is not None else _latest_traffic_release_dir()
    )
    target_decision = target_decision_report or {}
    contract = dict(target_decision.get("corridor_scoped_live_scoring_contract") or {})
    headroom_gate = _corridor_scoped_headroom_gate(
        live_trace=live_trace,
        wait_trace=wait_trace,
        oracle_trace=oracle_trace,
    )
    counterfactual_gate = _live_counterfactual_replay_gate(
        live_trace=live_trace,
        wait_trace=wait_trace,
        counterfactual_trace=counterfactual_trace,
    )
    evidence_contract = _corridor_scoped_evidence_contract(
        contract=contract,
        live_trace=live_trace,
        counterfactual_gate=counterfactual_gate,
    )
    score_gate = _score_consumption_evidence_gate(
        contract=contract,
        live_trace=live_trace,
        counterfactual_gate=counterfactual_gate,
    )
    ledger_preview = _corridor_scoped_case_ledger_delta_preview(
        contract=contract,
        release_dir=release_path,
    )
    blockers = _corridor_scoped_release_blockers(
        target_decision=target_decision,
        contract=contract,
        headroom_gate=headroom_gate,
        counterfactual_gate=counterfactual_gate,
        evidence_contract=evidence_contract,
        score_gate=score_gate,
        ledger_preview=ledger_preview,
    )
    core_gates_passed = not [
        blocker
        for blocker in blockers
        if blocker
        not in {
            "case_ledger_materialization_gate_not_run",
            "release_wrapper_materialization_gate_not_run",
        }
    ]
    status = (
        "promotion_preview_ready_live_release_blocked"
        if core_gates_passed
        else "blocked_corridor_scoped_release_requirements"
    )
    return {
        **_base_report(),
        "scope": CORRIDOR_SCOPED_PROMOTION_PREVIEW_REPORT_SCOPE,
        "status": status,
        "writes_release_artifacts": False,
        "writes_scenario_yaml": False,
        "target_decision_summary": {
            "scope": target_decision.get("scope"),
            "status": target_decision.get("status"),
            "chosen_path": target_decision.get("chosen_path"),
            "blocker_codes": list(target_decision.get("blocker_codes") or []),
        },
        "corridor_scoped_contract_summary": {
            "status": contract.get("status"),
            "target_id": contract.get("target_id"),
            "scoring_denominator": contract.get("scoring_denominator") or {},
            "required_evidence_kinds": list(
                contract.get("required_evidence_kinds") or []
            ),
            "release_gates_required": list(
                contract.get("release_gates_required") or []
            ),
        },
        "corridor_scoped_baseline_oracle_headroom_gate": headroom_gate,
        "live_counterfactual_replay_gate": counterfactual_gate,
        "evidence_contract": evidence_contract,
        "score_consumption_evidence_gate": score_gate,
        "case_ledger_delta_preview": ledger_preview,
        "release_blockers": blockers,
        "release_promotion_decision": _release_promotion_decision(
            live_scoring_allowed=False,
            case_ledger_preview_allowed=(
                core_gates_passed
                or target_decision.get("release_promotion_decision", {}).get(
                    "case_ledger_preview_allowed"
                )
                is True
            ),
            release_materializer_allowed=False,
            reason=(
                "corridor_scoped_contract_ready_but_release_materialization_blocked"
                if core_gates_passed
                else "corridor_scoped_contract_requirements_not_met"
            ),
        ),
        "policy": {
            "non_release_artifact": True,
            "release_artifact_mutation_allowed": False,
            "scenario_yaml_mutation_allowed": False,
            "mock_as_live_evidence_allowed": False,
            "network_wide_live_scoring_allowed": False,
            "live_rows_replace_existing_mock_traffic_denominator": True,
        },
        "next_actions": _corridor_scoped_promotion_next_actions(blockers),
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _traffic_release_snapshot(release_dir: Path) -> dict[str, Any]:
    manifest = _load_json(release_dir / "manifest.json")
    registry = _load_json(release_dir / "registry.json")
    primary = _load_json(release_dir / "primary_suite.json")
    core = _load_json(release_dir / "core_suite.json")
    registry_rows = list(registry.get("scenarios") or [])
    primary_rows = list(primary.get("scenarios") or [])
    core_rows = list(core.get("scenarios") or [])
    return {
        "release_id": manifest.get("release_id"),
        "scoring_version": manifest.get("scoring_version"),
        "traffic_registry_rows": _traffic_rows(registry_rows),
        "traffic_primary_rows": _traffic_rows(primary_rows),
        "traffic_core_rows": _traffic_rows(core_rows),
        "diagnostic_cells": list(
            (manifest.get("leaderboard_eligibility") or {}).get("diagnostic_cells")
            or []
        ),
    }


def _latest_traffic_release_dir() -> Path:
    release_root = REPO_ROOT / "release"
    candidates: list[Path] = []
    for release_dir in release_root.glob("dt_sched_bench_v*"):
        registry_path = release_dir / "registry.json"
        if not registry_path.exists():
            continue
        try:
            registry = _load_json(registry_path)
        except (OSError, json.JSONDecodeError):
            continue
        rows = list(registry.get("scenarios") or [])
        if _traffic_rows(rows):
            candidates.append(release_dir)
    if not candidates:
        raise FileNotFoundError("no release directory with Traffic rows found")
    return max(candidates, key=_release_version_key)


def _release_version_key(path: Path) -> tuple[int, ...]:
    raw = path.name.removeprefix("dt_sched_bench_v")
    out: list[int] = []
    for part in raw.split("_"):
        try:
            out.append(int(part))
        except ValueError:
            out.append(-1)
    return tuple(out)


def _traffic_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("domain") == "traffic" or row.get("backend_kind") == "mock_sumo"
    ]


def _live_case_ledger_blocking_gates(
    target_decision_report: dict[str, Any],
) -> list[str]:
    if not target_decision_report:
        return ["target_decision_report_missing"]

    blockers = set(str(v) for v in target_decision_report.get("blocker_codes") or [])
    gates: list[str] = []
    if "live_scoring_blocked_pending_split_fidelity_gate" in blockers:
        gates.append("split_fidelity_gate_not_passed")
    if "mock_live_fidelity_gap_exceeds_threshold" in blockers:
        gates.append("mock_live_fidelity_gap_exceeds_threshold")
    if "signal_program_readback_not_verified" in blockers:
        gates.append("signal_program_readback_not_verified")
    if "live_network_wide_scoring_denominator_not_represented" in blockers:
        gates.append("live_network_wide_scoring_denominator_not_represented")
    if "priority_semantics_not_verified" in blockers:
        gates.append("priority_semantics_not_verified")
    if "spillback_proxy_not_verified" in blockers:
        gates.append("spillback_proxy_not_verified")

    release_decision = target_decision_report.get("release_promotion_decision") or {}
    if release_decision.get("live_scoring_allowed") is not True:
        gates.append("live_scoring_fidelity_target_not_approved")
    gates.extend(
        [
            "live_scoring_case_ledger_not_materialized",
            "real_release_materializer_not_run_into_release_wrapper",
        ]
    )
    return sorted(dict.fromkeys(gates))


def _live_case_ledger_blocking_gates_from_runner(
    live_runner_summary: dict[str, Any],
) -> list[str]:
    if live_runner_summary.get("ready_for_case_ledger_preview") is not True:
        return ["live_runner_report_not_ready_for_case_ledger_preview"]
    return [
        "live_scoring_case_ledger_not_materialized",
        "real_release_materializer_not_run_into_release_wrapper",
    ]


def _live_runner_summary(
    live_runner_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(live_runner_report, dict):
        return {
            "ready_for_case_ledger_preview": False,
            "promotion_preview_release_blockers": [],
        }

    promotion_preview = live_runner_report.get("promotion_preview")
    if not isinstance(promotion_preview, dict):
        promotion_preview = {}
    release_blockers = [str(v) for v in promotion_preview.get("release_blockers") or []]
    materialization_blockers = {
        "case_ledger_materialization_gate_not_run",
        "release_wrapper_materialization_gate_not_run",
    }
    trace_status = dict(live_runner_report.get("trace_status") or {})
    ready = (
        live_runner_report.get("executed_with_live_backend") is True
        and promotion_preview.get("status")
        == "promotion_preview_ready_live_release_blocked"
        and bool(trace_status)
        and all(status == "captured_live_trace" for status in trace_status.values())
        and bool(release_blockers)
        and set(release_blockers).issubset(materialization_blockers)
    )
    return {
        "scope": live_runner_report.get("scope"),
        "status": live_runner_report.get("status"),
        "target_decision_source": live_runner_report.get("target_decision_source"),
        "executed_with_live_backend": live_runner_report.get(
            "executed_with_live_backend"
        )
        is True,
        "trace_status": trace_status,
        "promotion_preview_status": promotion_preview.get("status"),
        "promotion_preview_release_blockers": release_blockers,
        "ready_for_case_ledger_preview": ready,
    }


def _corridor_scoped_live_gate_summary(
    promotion_preview: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(promotion_preview, dict):
        return {
            "headroom_passed": False,
            "counterfactual_passed": False,
            "evidence_contract_passed": False,
            "score_consumption_passed": False,
            "live_corridor_denominator_key": None,
            "required_live_ledger_fields": [],
        }

    ledger = promotion_preview.get("case_ledger_delta_preview") or {}
    return {
        "headroom_passed": (
            promotion_preview.get("corridor_scoped_baseline_oracle_headroom_gate") or {}
        ).get("passed")
        is True,
        "counterfactual_passed": (
            promotion_preview.get("live_counterfactual_replay_gate") or {}
        ).get("passed")
        is True,
        "evidence_contract_passed": (
            promotion_preview.get("evidence_contract") or {}
        ).get("passed")
        is True,
        "score_consumption_passed": (
            promotion_preview.get("score_consumption_evidence_gate") or {}
        ).get("passed")
        is True,
        "live_corridor_denominator_key": ledger.get("live_corridor_denominator_key"),
        "required_live_ledger_fields": list(ledger.get("required_fields") or []),
    }


def _live_case_ledger_candidate_preview(
    row: dict[str, Any],
    *,
    release: dict[str, Any],
    target_blockers: list[str],
    blocking_release_gates: list[str],
    live_release_eligible: bool,
    live_runner_evidence_ready: bool = False,
    corridor_gate_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ledger = row.get("case_ledger") or {}
    demotion_rule = _live_case_ledger_demotion_rule(row, release)
    corridor_summary = corridor_gate_summary or {}
    return {
        "scenario_id": row.get("scenario_id"),
        "family": row.get("family"),
        "difficulty_mode": row.get("difficulty_mode"),
        "difficulty_level": row.get("difficulty_level"),
        "scenario_signature": row.get("scenario_signature"),
        "structural_fingerprint": row.get("structural_fingerprint"),
        "current_backend_kind": row.get("backend_kind"),
        "future_live_backend_kind": "sumo",
        "source_denominator_key": ledger.get("source_denominator_key"),
        "independence_axis": ledger.get("independence_axis"),
        "decision_pressure_axis": ledger.get("decision_pressure_axis"),
        "decision_variant_key": ledger.get("decision_variant_key"),
        "additional_decision_axis": ledger.get("additional_decision_axis"),
        "complexity_tags": list(ledger.get("complexity_tags") or []),
        "dimension_applicability": ledger.get("dimension_applicability") or {},
        "live_headroom_citation": ledger.get("live_headroom_citation"),
        "diagnostic_demotion_rule": demotion_rule,
        "live_release_eligible": bool(live_release_eligible),
        "live_runner_evidence_ready": bool(live_runner_evidence_ready),
        "live_corridor_denominator_key": corridor_summary.get(
            "live_corridor_denominator_key"
        ),
        "live_scoring_blockers": list(blocking_release_gates),
        "target_decision_blockers": list(target_blockers),
        "case_ledger_delta_needed": [
            "live_scoring_case_ledger_delta_required",
            "scored_backend_kind_update_mock_sumo_to_sumo",
            "live_fidelity_target_split_evidence_required",
            "signal_program_readback_evidence_required",
            "live_counterfactual_or_explicit_opt_out_required",
            "release_materializer_row_write_required",
        ],
        "live_case_ledger_fields_required": [
            "live_scored_backend_descriptor",
            "live_fidelity_gate_citation",
            "signal_program_readback_citation",
            "live_replay_determinism_citation",
            "live_tool_effect_evidence_ids",
            "live_score_consumption_evidence_ids",
            *[
                field
                for field in corridor_summary.get("required_live_ledger_fields", [])
                if field
                not in {
                    "live_scored_backend_descriptor",
                    "live_fidelity_gate_citation",
                    "signal_program_readback_citation",
                    "live_replay_determinism_citation",
                    "live_tool_effect_evidence_ids",
                    "live_score_consumption_evidence_ids",
                }
            ],
        ],
    }


def _live_replacement_pilot_rows(
    *,
    live_runner_report: dict[str, Any] | None,
    ledger_preview: dict[str, Any],
    release: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(live_runner_report, dict):
        return []
    seed_request = live_runner_report.get("seed_request")
    if not isinstance(seed_request, dict):
        return []
    runner_family = str(seed_request.get("family") or "")
    runner_mode = str(seed_request.get("difficulty_mode") or "")
    runner_level = str(seed_request.get("difficulty_level") or "")
    if not runner_family or not runner_mode or not runner_level:
        return []

    core_ids = {
        str(row.get("scenario_id"))
        for row in release.get("traffic_core_rows") or []
        if row.get("scenario_id")
    }
    pilot_rows: list[dict[str, Any]] = []
    for row in ledger_preview.get("candidate_rows_preview") or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("family")) != runner_family:
            continue
        if str(row.get("difficulty_level")) != runner_level:
            continue
        evidence_match = "exact_primary_row"
        if str(row.get("difficulty_mode")) != runner_mode:
            demotion = row.get("diagnostic_demotion_rule") or {}
            if demotion.get("demoted_twin_mode") != runner_mode:
                continue
            evidence_match = "decision_equivalent_mode_twin"
        pilot_rows.append(
            {
                "scenario_id": row.get("scenario_id"),
                "family": row.get("family"),
                "difficulty_mode": row.get("difficulty_mode"),
                "difficulty_level": row.get("difficulty_level"),
                "source_denominator_key": row.get("source_denominator_key"),
                "scenario_signature": row.get("scenario_signature"),
                "structural_fingerprint": row.get("structural_fingerprint"),
                "decision_variant_key": row.get("decision_variant_key"),
                "live_corridor_denominator_key": row.get(
                    "live_corridor_denominator_key"
                ),
                "independence_axis": row.get("independence_axis"),
                "decision_pressure_axis": row.get("decision_pressure_axis"),
                "dimension_applicability": row.get("dimension_applicability") or {},
                "current_backend_kind": row.get("current_backend_kind"),
                "future_live_backend_kind": row.get("future_live_backend_kind"),
                "in_current_core_suite": str(row.get("scenario_id")) in core_ids,
                "evidence_match": evidence_match,
                "runner_seed_request": {
                    "family": runner_family,
                    "difficulty_mode": runner_mode,
                    "difficulty_level": runner_level,
                    "seed": seed_request.get("seed"),
                    "n_ticks": seed_request.get("n_ticks"),
                },
                "authorization_scope": "pilot_design_only_not_release_write",
            }
        )
    return pilot_rows


def _full_live_replacement_ready_rows(
    *, ledger_preview: dict[str, Any], release: dict[str, Any]
) -> list[dict[str, Any]]:
    del release
    # A row needs row-local live proof to count as full replacement ready. The
    # current live-runner bridge can prove one cell and intentionally marks all
    # candidate rows as previewed; that is not enough for full replacement.
    rows: list[dict[str, Any]] = []
    for row in ledger_preview.get("candidate_rows_preview") or []:
        if not isinstance(row, dict):
            continue
        if (
            row.get("live_release_eligible") is True
            and row.get("per_row_live_runner_evidence_ready") is True
        ):
            rows.append(
                {
                    "scenario_id": row.get("scenario_id"),
                    "family": row.get("family"),
                    "difficulty_mode": row.get("difficulty_mode"),
                    "difficulty_level": row.get("difficulty_level"),
                    "source_denominator_key": row.get("source_denominator_key"),
                }
            )
    return rows


def _live_replacement_plan_blockers(
    *,
    runner_ready: bool,
    ledger_ready: bool,
    ledger_materialization_only: bool,
    pilot_rows: list[dict[str, Any]],
    full_replacement_rows: list[dict[str, Any]],
    live_runner_summary: dict[str, Any],
    ledger_preview: dict[str, Any],
) -> list[str]:
    del full_replacement_rows
    blockers: list[str] = []
    if not live_runner_summary.get("scope"):
        blockers.append("live_runner_report_missing")
    elif not runner_ready:
        blockers.append("live_runner_report_not_ready_for_case_ledger_preview")
    if not ledger_preview:
        blockers.append("live_case_ledger_preview_missing")
    elif not ledger_ready:
        blockers.append("live_case_ledger_preview_not_ready")
    if ledger_preview and not ledger_materialization_only:
        blockers.append("live_case_ledger_preview_has_non_materialization_blockers")
    if not pilot_rows:
        blockers.append("pilot_live_row_not_matched_to_release_primary_or_core")
    return list(dict.fromkeys(blockers))


def _live_replacement_denominator_key(
    *,
    live_runner_report: dict[str, Any] | None,
    ledger_preview: dict[str, Any],
) -> str | None:
    preview = {}
    if isinstance(live_runner_report, dict) and isinstance(
        live_runner_report.get("promotion_preview"), dict
    ):
        preview = live_runner_report.get("promotion_preview") or {}
    runner_key = (
        (preview.get("case_ledger_delta_preview") or {}).get(
            "live_corridor_denominator_key"
        )
        if isinstance(preview, dict)
        else None
    )
    if runner_key:
        return str(runner_key)
    for row in ledger_preview.get("candidate_rows_preview") or []:
        if isinstance(row, dict) and row.get("live_corridor_denominator_key"):
            return str(row.get("live_corridor_denominator_key"))
    return None


def _live_replacement_plan_gates(
    *,
    runner_ready: bool,
    ledger_ready: bool,
    ledger_materialization_only: bool,
    pilot_rows: list[dict[str, Any]],
    full_replacement_rows: list[dict[str, Any]],
    live_runner_summary: dict[str, Any],
    ledger_preview: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    return {
        "live_runner_gate": {
            "passed": bool(runner_ready),
            "status": "passed" if runner_ready else "blocked",
            "blocker_codes": []
            if runner_ready
            else ["live_runner_report_not_ready_for_case_ledger_preview"],
            "trace_status": dict(live_runner_summary.get("trace_status") or {}),
        },
        "case_ledger_preview_gate": {
            "passed": bool(ledger_ready),
            "status": "passed" if ledger_ready else "blocked",
            "blocker_codes": []
            if ledger_ready
            else ["live_case_ledger_preview_not_ready"],
            "preview_status": ledger_preview.get("status"),
        },
        "materialization_blocker_scope_gate": {
            "passed": bool(ledger_materialization_only),
            "status": "passed" if ledger_materialization_only else "blocked",
            "blocker_codes": []
            if ledger_materialization_only
            else ["live_case_ledger_preview_has_non_materialization_blockers"],
            "blocking_release_gates": list(
                ledger_preview.get("blocking_release_gates") or []
            ),
        },
        "single_cell_pilot_match_gate": {
            "passed": bool(pilot_rows),
            "status": "passed" if pilot_rows else "blocked",
            "blocker_codes": []
            if pilot_rows
            else ["pilot_live_row_not_matched_to_release_primary_or_core"],
            "matched_rows": len(pilot_rows),
        },
        "full_replacement_per_row_evidence_gate": {
            "passed": False,
            "status": "blocked",
            "blocker_codes": ["per_row_live_runner_evidence_missing"],
            "matched_rows": len(full_replacement_rows),
            "required_scope": "one_live_runner_or_equivalence_gate_per_live_primary_row",
        },
    }


def _live_replacement_evidence_scope(
    live_runner_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(live_runner_report, dict):
        return {
            "scope": "no_live_runner_evidence",
            "applies_to": "none",
            "trace_roles": {},
        }
    preview = live_runner_report.get("promotion_preview")
    if not isinstance(preview, dict):
        preview = {}
    ledger = preview.get("case_ledger_delta_preview")
    if not isinstance(ledger, dict):
        ledger = {}
    return {
        "scope": "single_live_runner_seed_request",
        "applies_to": "single_cell_only_until_per_row_live_evidence_exists",
        "seed_request": live_runner_report.get("seed_request") or {},
        "trace_roles": dict(live_runner_report.get("trace_status") or {}),
        "target_decision_source": live_runner_report.get("target_decision_source"),
        "promotion_preview_status": preview.get("status"),
        "live_corridor_denominator_key": ledger.get("live_corridor_denominator_key"),
    }


def _live_replacement_plan_next_actions(
    *,
    status: str,
    blockers: list[str],
    pilot_rows: list[dict[str, Any]],
) -> list[str]:
    if status == "ready_for_authorization_pilot_only":
        return [
            "request_explicit_authorization_for_v0_9_single_cell_live_replacement_plan",
            "keep_v0_8_1_mock_sumo_canonical_until_new_release_passes_gates",
            "design_v0_9_live_traffic_materializer_for_pilot_rows_only",
            "run_per_row_live_runner_before_full_traffic_replacement",
        ]
    actions: list[str] = []
    if "live_runner_report_missing" in blockers:
        actions.append("run_corridor_scoped_live_runner_first")
    if "live_runner_report_not_ready_for_case_ledger_preview" in blockers:
        actions.append("repair_live_runner_preview_gate_before_materialization_plan")
    if "live_case_ledger_preview_has_non_materialization_blockers" in blockers:
        actions.append("repair_live_case_ledger_preview_non_materialization_blockers")
    if "pilot_live_row_not_matched_to_release_primary_or_core" in blockers:
        actions.append("rerun_live_runner_on_a_primary_or_decision_equivalent_cell")
    if not pilot_rows:
        actions.append("keep_mock_sumo_canonical_and_move_to_next_source_locked_track")
    return list(dict.fromkeys(actions))


def _live_replacement_pilot_dry_run_blockers(
    *,
    plan_ready: bool,
    pilot_rows: list[dict[str, Any]],
    replacement_not_addition: bool,
) -> list[str]:
    blockers = [
        "explicit_user_authorization_missing",
        "v0_9_live_traffic_materializer_not_implemented",
        "v0_9_live_traffic_behavioral_audit_wrapper_not_implemented",
    ]
    if not plan_ready:
        blockers.append("replacement_plan_not_ready_for_pilot")
    if len(pilot_rows) != 1:
        blockers.append("pilot_scope_not_single_cell")
    if not replacement_not_addition:
        blockers.append("replacement_semantics_not_proven")
    return list(dict.fromkeys(blockers))


def _live_replacement_pilot_dry_run_rows(
    *, pilot_rows: list[dict[str, Any]], target_release_id: str
) -> list[dict[str, Any]]:
    if len(pilot_rows) != 1:
        return []
    row = pilot_rows[0]
    scenario_id = str(row.get("scenario_id") or "")
    target_scenario_id = scenario_id.replace("traffic/", "traffic_live/", 1)
    if target_scenario_id == scenario_id:
        target_scenario_id = f"traffic_live/{scenario_id}"
    return [
        {
            "current_scenario_id": scenario_id,
            "target_scenario_id": target_scenario_id,
            "target_release_id": target_release_id,
            "family": row.get("family"),
            "difficulty_mode": row.get("difficulty_mode"),
            "difficulty_level": row.get("difficulty_level"),
            "current_backend_kind": row.get("current_backend_kind") or "mock_sumo",
            "target_backend_kind": row.get("future_live_backend_kind") or "sumo",
            "source_denominator_key": row.get("source_denominator_key"),
            "live_corridor_denominator_key": row.get("live_corridor_denominator_key"),
            "source_denominator_key_preserved": True,
            "core_membership_preserved": bool(row.get("in_current_core_suite")),
            "scenario_signature_to_replace": row.get("scenario_signature"),
            "structural_fingerprint_to_replace": row.get("structural_fingerprint"),
            "live_evidence_match": row.get("evidence_match"),
            "runner_seed_request": row.get("runner_seed_request") or {},
            "case_ledger_delta": {
                "scored_backend_kind": "sumo",
                "replaces_backend_kind": row.get("current_backend_kind") or "mock_sumo",
                "live_scored_backend_descriptor_required": True,
                "live_corridor_denominator_key_required": True,
                "live_counterfactual_replay_citation_required": True,
                "live_score_consumption_evidence_ids_required": True,
                "mock_live_double_count_forbidden": True,
            },
        }
    ]


def _live_replacement_pilot_authorization_prompt(
    replacement_rows: list[dict[str, Any]],
) -> str:
    if not replacement_rows:
        return (
            "Do not authorize Traffic live SUMO replacement materialization until "
            "a single pilot row is matched."
        )
    scenario_id = replacement_rows[0].get("current_scenario_id")
    return (
        "Authorize v0.9 single-cell Traffic live SUMO replacement pilot "
        f"materialization for {scenario_id}."
    )


def _live_replacement_pilot_write_set(target_release_id: str) -> list[str]:
    suffix = target_release_id.removeprefix("dt_sched_bench_")
    release_dir = f"release/{target_release_id}"
    scenario_dir = f"scenarios/releases/{target_release_id}"
    return [
        f"{release_dir}/manifest.json",
        f"{release_dir}/registry.json",
        f"{release_dir}/primary_suite.json",
        f"{release_dir}/core_suite.json",
        f"{scenario_dir}/traffic_live/incident_response/deep_planning/basic.yaml",
        f"scripts/audit_{suffix}_live_traffic_behavioral.py",
        f"scripts/check_{suffix}_readiness.py",
        f"docs/RELEASE_NOTES_{suffix}.md",
    ]


def _live_replacement_pilot_dry_run_next_actions(
    *, status: str, replacement_rows: list[dict[str, Any]]
) -> list[str]:
    if status == "ready_for_explicit_authorization":
        return [
            "ask_user_to_authorize_exact_single_cell_v0_9_live_replacement",
            "keep_v0_8_1_mock_sumo_canonical_until_authorized_release_passes",
            "implement_v0_9_materializer_only_after_authorization",
        ]
    if replacement_rows:
        return [
            "repair_replacement_plan_gates_before_requesting_authorization",
            "keep_v0_8_1_mock_sumo_canonical",
        ]
    return [
        "run_traffic_live_replacement_materialization_plan_first",
        "keep_v0_8_1_mock_sumo_canonical",
    ]


def _live_case_ledger_demotion_rule(
    row: dict[str, Any], release: dict[str, Any]
) -> dict[str, Any]:
    family = row.get("family")
    mode = row.get("difficulty_mode")
    level = row.get("difficulty_level")
    for cell in release.get("diagnostic_cells") or []:
        if (
            cell.get("domain") == "traffic"
            and cell.get("family") == family
            and cell.get("difficulty_mode") != mode
            and cell.get("difficulty_level") == level
        ):
            reason = cell.get("reason") or {}
            return {
                "applies_to_future_live_twins": True,
                "reason_code": str(
                    reason.get("code") or "decision_equivalent_difficulty_mode_twin"
                ),
                "current_primary_keeps_mode": mode,
                "demoted_twin_mode": cell.get("difficulty_mode"),
                "decision_equivalent_to_cell": reason.get(
                    "decision_equivalent_to_cell"
                ),
            }
    return {
        "applies_to_future_live_twins": False,
        "reason_code": None,
        "current_primary_keeps_mode": mode,
    }


def _live_case_ledger_next_actions(blocking_release_gates: list[str]) -> list[str]:
    actions: list[str] = []
    if "split_fidelity_gate_not_passed" in blocking_release_gates:
        actions.extend(
            [
                "implement_split_fidelity_gate",
                "rerun_mock_live_fidelity_with_split_targets",
            ]
        )
    if "signal_program_readback_not_verified" in blocking_release_gates:
        actions.append("verify_live_signal_program_readback")
    actions.extend(
        [
            "materialize_live_case_ledger_only_after_gates_pass",
            "keep_release_artifacts_unchanged",
        ]
    )
    return list(dict.fromkeys(actions))


def _corridor_scoped_headroom_gate(
    *,
    live_trace: dict[str, Any] | list[dict[str, Any]] | None,
    wait_trace: dict[str, Any] | list[dict[str, Any]] | None,
    oracle_trace: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if live_trace is None or wait_trace is None:
        missing = []
        if live_trace is None:
            missing.append("live_trace")
        if wait_trace is None:
            missing.append("wait_trace")
        return {
            "passed": False,
            "status": "blocked_missing_trace",
            "blocker_codes": [f"{name}_missing" for name in missing],
            "denominator": "tls_controlled_corridor_set",
        }
    wait_cost = _corridor_scoped_delay_cost(wait_trace)
    live_cost = _corridor_scoped_delay_cost(live_trace)
    oracle_cost = (
        _corridor_scoped_delay_cost(oracle_trace)
        if oracle_trace is not None
        else live_cost
    )
    baseline_gap = wait_cost - live_cost
    oracle_vs_wait_headroom = wait_cost - oracle_cost
    passed = baseline_gap > 0.0 and oracle_vs_wait_headroom > 0.0
    blockers: list[str] = []
    if baseline_gap <= 0.0:
        blockers.append("corridor_scoped_baseline_gap_non_positive")
    if oracle_vs_wait_headroom <= 0.0:
        blockers.append("corridor_scoped_oracle_headroom_non_positive")
    return {
        "passed": passed,
        "status": "passed" if passed else "blocked_no_decision_headroom",
        "denominator": "tls_controlled_corridor_set",
        "metric": "sum_per_corridor_delay_minutes",
        "wait_cost": _round3(wait_cost),
        "candidate_live_cost": _round3(live_cost),
        "oracle_like_cost": _round3(oracle_cost),
        "baseline_gap_wait_minus_candidate": _round3(baseline_gap),
        "oracle_vs_wait_headroom": _round3(oracle_vs_wait_headroom),
        "blocker_codes": blockers,
    }


def _live_counterfactual_replay_gate(
    *,
    live_trace: dict[str, Any] | list[dict[str, Any]] | None,
    wait_trace: dict[str, Any] | list[dict[str, Any]] | None,
    counterfactual_trace: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if live_trace is None:
        return {
            "passed": False,
            "status": "blocked_missing_live_trace",
            "blocker_codes": ["live_trace_missing"],
            "fallback_if_unavailable": (
                "counterfactual_prevention_applicable_false_required"
            ),
        }
    if counterfactual_trace is None:
        return {
            "passed": False,
            "status": "blocked_missing_counterfactual_trace",
            "blocker_codes": ["counterfactual_trace_missing"],
            "basis": "same_seed_same_live_sumo_backend_action_stream_masked",
            "fallback_if_unavailable": (
                "counterfactual_prevention_applicable_false_required"
            ),
        }
    counterfactual_cost = _corridor_scoped_delay_cost(counterfactual_trace)
    wait_cost = (
        _corridor_scoped_delay_cost(wait_trace)
        if wait_trace is not None
        else counterfactual_cost
    )
    live_cost = _corridor_scoped_delay_cost(live_trace)
    deterministic_replay_passed = abs(counterfactual_cost - wait_cost) <= 1e-6
    prevented_loss = counterfactual_cost - live_cost
    passed = deterministic_replay_passed and prevented_loss > 0.0
    blockers: list[str] = []
    if not deterministic_replay_passed:
        blockers.append("masked_counterfactual_replay_not_deterministic")
    if prevented_loss <= 0.0:
        blockers.append("counterfactual_prevented_loss_non_positive")
    return {
        "passed": passed,
        "status": "passed" if passed else "blocked_counterfactual_replay",
        "basis": "same_seed_same_live_sumo_backend_action_stream_masked",
        "counterfactual_cost": _round3(counterfactual_cost),
        "replay_no_action_cost": _round3(wait_cost),
        "candidate_live_cost": _round3(live_cost),
        "prevented_loss": _round3(prevented_loss),
        "deterministic_replay_passed": deterministic_replay_passed,
        "blocker_codes": blockers,
        "fallback_if_unavailable": (
            "counterfactual_prevention_applicable_false_required"
        ),
    }


def _corridor_scoped_evidence_contract(
    *,
    contract: dict[str, Any],
    live_trace: dict[str, Any] | list[dict[str, Any]] | None,
    counterfactual_gate: dict[str, Any],
) -> dict[str, Any]:
    required = list(contract.get("required_evidence_kinds") or [])
    observed = set(_observed_live_evidence_kinds(live_trace))
    if counterfactual_gate.get("passed") is True:
        observed.add("counterfactual_replay_evidence")
    missing = [kind for kind in required if kind not in observed]
    return {
        "passed": not missing and bool(required),
        "status": "passed" if not missing and required else "blocked_missing_evidence",
        "required_evidence_kinds": required,
        "observed_evidence_kinds": sorted(observed),
        "missing_evidence_kinds": missing,
        "evidence_ids": _live_trace_evidence_ids(live_trace),
    }


def _score_consumption_evidence_gate(
    *,
    contract: dict[str, Any],
    live_trace: dict[str, Any] | list[dict[str, Any]] | None,
    counterfactual_gate: dict[str, Any],
) -> dict[str, Any]:
    del contract
    score_ids = _score_consumption_evidence_ids(live_trace)
    required_dimensions = [
        "weighted_equity_score",
        "stakeholder_management",
        "counterfactual_prevention",
    ]
    consumed = {
        dim: list(score_ids.get(dim) or [])
        for dim in required_dimensions
        if score_ids.get(dim)
    }
    missing = [dim for dim in required_dimensions if dim not in consumed]
    if (
        counterfactual_gate.get("passed") is not True
        and "counterfactual_prevention" not in missing
    ):
        missing.append("counterfactual_prevention")
    return {
        "passed": not missing,
        "status": "passed" if not missing else "blocked_missing_score_evidence",
        "required_dimensions": required_dimensions,
        "consumed_dimensions": consumed,
        "missing_dimensions": missing,
        "score_consumption_evidence_ids": score_ids,
    }


def _corridor_scoped_case_ledger_delta_preview(
    *, contract: dict[str, Any], release_dir: Path
) -> dict[str, Any]:
    required = list(contract.get("case_ledger_fields_required") or [])
    denominator = dict(contract.get("scoring_denominator") or {})
    return {
        "required_fields_present": bool(required),
        "required_fields": required,
        "live_corridor_denominator_key": "sumo_ingolstadt:tls_controlled_corridor_set",
        "source_release_dir": str(release_dir),
        "denominator": denominator,
        "delta_basis": (
            "future live SUMO rows replace the current mock_sumo Traffic "
            "denominator for the same source/family/level keys; they must not "
            "pad a second denominator."
        ),
        "writes_release_artifacts": False,
    }


def _corridor_scoped_release_blockers(
    *,
    target_decision: dict[str, Any],
    contract: dict[str, Any],
    headroom_gate: dict[str, Any],
    counterfactual_gate: dict[str, Any],
    evidence_contract: dict[str, Any],
    score_gate: dict[str, Any],
    ledger_preview: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not target_decision:
        blockers.append("target_decision_report_missing")
    if contract.get("status") != "draft_ready_non_release":
        blockers.append("corridor_scoped_live_contract_not_ready")
    for name, gate in (
        ("corridor_scoped_baseline_oracle_headroom_gate", headroom_gate),
        ("live_counterfactual_replay_gate", counterfactual_gate),
        ("evidence_contract", evidence_contract),
        ("score_consumption_evidence_gate", score_gate),
    ):
        if gate.get("passed") is not True:
            blockers.append(f"{name}_not_passed")
    if ledger_preview.get("required_fields_present") is not True:
        blockers.append("case_ledger_delta_preview_incomplete")
    blockers.extend(
        [
            "case_ledger_materialization_gate_not_run",
            "release_wrapper_materialization_gate_not_run",
        ]
    )
    return sorted(dict.fromkeys(blockers))


def _corridor_scoped_promotion_next_actions(blockers: list[str]) -> list[str]:
    actions: list[str] = []
    if "target_decision_report_missing" in blockers:
        actions.append("run_target_decision_report_first")
    if "corridor_scoped_baseline_oracle_headroom_gate_not_passed" in blockers:
        actions.append("rerun_live_wait_candidate_oracle_corridor_headroom_probe")
    if "live_counterfactual_replay_gate_not_passed" in blockers:
        actions.append("rerun_same_seed_masked_live_counterfactual_replay")
    if "evidence_contract_not_passed" in blockers:
        actions.append("wire_live_corridor_queue_delay_and_tool_effect_evidence")
    if "score_consumption_evidence_gate_not_passed" in blockers:
        actions.append("prove_scorer_consumes_live_evidence_ids")
    actions.extend(
        [
            "keep_release_artifacts_unchanged",
            "request_release_materialization_only_after_all_preview_gates_pass",
        ]
    )
    return list(dict.fromkeys(actions))


def _corridor_scoped_delay_cost(
    trace: dict[str, Any] | list[dict[str, Any]] | None,
) -> float:
    if trace is None:
        return 0.0
    delay = _per_corridor_delay(trace)
    if delay:
        return float(sum(delay.values()))
    details = _live_per_corridor_details(trace)
    return float(
        sum(_as_float(row.get("cumulative_delay_minutes")) for row in details.values())
    )


def _observed_live_evidence_kinds(
    trace: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[str]:
    if trace is None:
        return []
    observed: set[str] = set()
    if _trace_dict(trace).get("executed_with_live_backend") is True:
        observed.add("live_backend_execution")
    for event in _last_record(trace).get("realized_events") or []:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "sumo_live_snapshot":
            observed.add("sumo_live_snapshot")
            if isinstance(event.get("per_corridor"), dict) and event.get(
                "per_corridor"
            ):
                observed.add("per_corridor_delay_and_queue")
    if _control_readback_match_rate(trace) is not None:
        observed.add("signal_program_readback")
    if _live_tool_effect_evidence_ids(trace):
        observed.add("tool_effect_evidence")
    if _score_consumption_evidence_ids(trace):
        observed.add("score_consumption_evidence")
    return sorted(observed)


def _live_trace_evidence_ids(
    trace: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, list[str]]:
    if trace is None:
        return {}
    ids: dict[str, list[str]] = {}
    tool_ids = _live_tool_effect_evidence_ids(trace)
    if tool_ids:
        ids["tool_effect_evidence"] = tool_ids
    snapshot_ids: list[str] = []
    for event in _last_record(trace).get("realized_events") or []:
        if (
            isinstance(event, dict)
            and event.get("type") == "sumo_live_snapshot"
            and event.get("evidence_id")
        ):
            snapshot_ids.append(str(event.get("evidence_id")))
    if snapshot_ids:
        ids["sumo_live_snapshot"] = snapshot_ids
    for kind, values in _score_consumption_evidence_ids(trace).items():
        ids[f"score_consumption:{kind}"] = list(values)
    return ids


def _live_tool_effect_evidence_ids(
    trace: dict[str, Any] | list[dict[str, Any]] | None,
) -> list[str]:
    if trace is None:
        return []
    out: list[str] = []
    for batch in _trace_dict(trace).get("tool_results") or []:
        rows = batch if isinstance(batch, list) else [batch]
        for row in rows:
            if not isinstance(row, dict):
                continue
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            if row.get("name") != "change_signal_plan":
                continue
            if (
                row.get("ok") is not True
                or payload.get("sumo_state_mutated") is not True
            ):
                continue
            if row.get("evidence_id"):
                out.append(str(row.get("evidence_id")))
    return out


def _score_consumption_evidence_ids(
    trace: dict[str, Any] | list[dict[str, Any]] | None,
) -> dict[str, list[str]]:
    if trace is None:
        return {}
    raw = _trace_dict(trace).get("score_consumption_evidence_ids")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key, values in raw.items():
        if isinstance(values, list):
            out[str(key)] = [str(v) for v in values]
        elif values:
            out[str(key)] = [str(values)]
    return out


def _live_trace_capture_contract(
    *,
    family: str,
    difficulty_level: str,
    difficulty_mode: str,
    seed: int,
    n_ticks: int,
    run_live: bool,
    live_trace: dict[str, Any] | None,
) -> dict[str, Any]:
    env_gated = os.environ.get("OPERATE_TRAFFIC_BACKEND_REAL") == "1"
    runtime_available = sumo_available() if env_gated else False
    if live_trace is not None:
        status = "captured_live_trace"
    elif not env_gated:
        status = "blocked_env_gate_unset"
    elif not runtime_available:
        status = "blocked_sumo_runtime_unavailable"
    elif not run_live:
        status = "blocked_run_live_not_requested"
    else:
        status = "blocked_live_trace_missing"
    return {
        "status": status,
        "env_gate": LIVE_TRACE_CAPTURE_ENV_GATE,
        "env_gate_satisfied": env_gated,
        "sumo_runtime_available": runtime_available,
        "run_live_requested": run_live,
        "live_trace_required": True,
        "silent_mock_fallback_allowed": False,
        "generate_command": _live_trace_generate_command(
            family=family,
            difficulty_level=difficulty_level,
            difficulty_mode=difficulty_mode,
            seed=seed,
            n_ticks=n_ticks,
        ),
    }


def _live_trace_generate_command(
    *,
    family: str,
    difficulty_level: str,
    difficulty_mode: str,
    seed: int,
    n_ticks: int,
) -> str:
    return (
        f"{LIVE_TRACE_CAPTURE_ENV_GATE} .venv/bin/python "
        "scripts/traffic_mock_live_fidelity_probe.py --replay-seed "
        f"--run-live --family {family} --difficulty-level {difficulty_level} "
        f"--difficulty-mode {difficulty_mode} --seed {int(seed)} "
        f"--n-ticks {int(n_ticks)} "
        "--output reports/traffic_mock_live_fidelity_probe.json"
    )


def _wait_action_stream(n_ticks: int) -> list[list[dict[str, Any]]]:
    return [[{"name": "wait"}] for _ in range(max(0, int(n_ticks)))]


def _oracle_like_corridor_action_stream(
    *,
    target_decision_report: dict[str, Any] | None,
    candidate_action_stream: list[Any],
    n_ticks: int,
) -> list[Any]:
    """Bounded live policy for headroom proof, not a global SUMO optimum."""

    first_tick: list[dict[str, Any]] = []
    candidate_first = candidate_action_stream[0] if candidate_action_stream else []
    candidate_calls = (
        candidate_first if isinstance(candidate_first, list) else [candidate_first]
    )
    for call in candidate_calls:
        if isinstance(call, dict) and call.get("name") == "change_signal_plan":
            first_tick.append(dict(call))
    if not first_tick:
        first_tick = _default_fidelity_action_stream(1)[0]

    contract = (target_decision_report or {}).get(
        "corridor_scoped_live_scoring_contract"
    ) or {}
    denominator = contract.get("scoring_denominator") or {}
    bound_count = int(denominator.get("bound_corridor_count") or 0)
    if bound_count <= 1:
        return [first_tick, *_wait_action_stream(max(0, int(n_ticks) - 1))]

    try:
        seed = build_traffic_seed(
            seed_id="live-runner/oracle-like",
            family="incident_response",
        )
        corridors = sorted(dict(seed.backend_config.get("corridor_tls_map") or {}))
    except Exception:
        corridors = []
    if not corridors:
        return [first_tick, *_wait_action_stream(max(0, int(n_ticks) - 1))]

    program = str((first_tick[0].get("args") or {}).get("program") or "incident_relief")
    first_tick = [
        {
            "name": "change_signal_plan",
            "args": {"corridor": corridor, "program": program},
        }
        for corridor in corridors
    ]
    return [first_tick, *_wait_action_stream(max(0, int(n_ticks) - 1))]


def _derive_corridor_scoped_target_decision_from_trace(
    live_trace: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the corridor-scoped non-release contract from a captured live trace."""

    network = _live_network_totals(live_trace)
    corridor = _benchmark_corridor_attribution(live_trace, network)
    mapping = _mapping_coverage_assessment(
        {"mapping_coverage": _live_mapping_coverage(live_trace)}
    )
    signal_readback = _signal_program_readback_assessment(
        {
            "fidelity_metrics": {
                "signal_program_mismatch_count": 0,
                "live_control_readback_match_rate": _control_readback_match_rate(
                    live_trace
                ),
            }
        }
    )
    corridor_target = _corridor_scoped_live_target_candidate(
        fidelity_report={"executed_with_live_backend": True},
        corridor=corridor,
        mapping=mapping,
        signal_readback=signal_readback,
    )
    target_assessment = {
        "network_total_fidelity": {
            "status": "not_evaluated_by_corridor_scoped_live_runner",
            **network,
        },
        "tls_controlled_corridor_attribution": {
            "status": "corridor_scoped_target",
            **corridor,
        },
        "mapping_coverage": mapping,
        "corridor_scoped_live_target_candidate": corridor_target,
        "signal_program_readback": signal_readback,
        "priority_semantics": {"status": "not_evaluated_by_live_runner"},
        "spillback_proxy": {"status": "not_evaluated_by_live_runner"},
        "basis_mismatch_detected": mapping.get("network_wide_scoring_represented")
        is False,
        "measurement_basis": _target_split_measurement_basis(),
    }
    blockers = ["live_network_wide_scoring_denominator_not_represented"]
    if corridor_target.get("candidate_ready") is not True:
        blockers.extend(
            f"corridor_scoped_live_target_missing_{name}"
            for name in corridor_target.get("missing_requirements") or []
        )
    return {
        **_base_report(),
        "scope": TARGET_DECISION_REPORT_SCOPE,
        "status": "decision_ready_live_scoring_blocked",
        "blocker_codes": sorted(dict.fromkeys(blockers)),
        "chosen_path": "keep_mock_scoring_until_live_denominator_redefined",
        "target_assessment": target_assessment,
        "decision_options": {
            "corridor_scoped_live_target": _corridor_scoped_live_target_option(
                target_assessment
            ),
            "keep_mock_scoring_as_canonical": {
                "status": (
                    "selected_until_live_denominator_redefined_or_coverage_expanded"
                ),
                "controlled_lane_share_of_network": mapping.get(
                    "controlled_lane_share_of_network"
                ),
                "network_lane_count": mapping.get("network_lane_count"),
                "unique_controlled_lanes": mapping.get("unique_controlled_lanes"),
            },
        },
        "corridor_scoped_live_scoring_contract": (
            _corridor_scoped_live_scoring_contract(target_assessment)
        ),
        "release_promotion_decision": _release_promotion_decision(
            live_scoring_allowed=False,
            case_ledger_preview_allowed=True,
            release_materializer_allowed=False,
            reason="live_network_wide_denominator_not_represented",
        ),
        "policy": {
            "non_release_artifact": True,
            "release_artifact_mutation_allowed": False,
            "scenario_yaml_mutation_allowed": False,
            "mock_as_live_evidence_allowed": False,
            "network_wide_live_scoring_allowed": False,
        },
    }


def _capture_live_trace(
    *,
    family: str,
    difficulty_level: str,
    difficulty_mode: str,
    seed: int,
    n_ticks: int,
    action_stream: list[Any],
    trace_role: str,
) -> dict[str, Any]:
    live_seed = build_traffic_seed(
        seed_id=f"corridor_live_runner/{trace_role}",
        family=family,
        seed=int(seed),
        difficulty_level=difficulty_level,
        difficulty_mode=difficulty_mode,
    )
    live_seed.backend_kind = "sumo"
    live_seed.backend_config = {
        **live_seed.backend_config,
        "backend_kind": "sumo",
    }
    trace = _replay_traffic_seed(
        scenario_config=live_seed.to_dict(),
        seed=int(seed),
        action_stream=action_stream,
        n_ticks=int(n_ticks),
    )
    trace.update(
        {
            "trace_role": trace_role,
            "family": family,
            "difficulty_level": difficulty_level,
            "difficulty_mode": difficulty_mode,
            "seed": int(seed),
            "n_ticks": int(n_ticks),
            "action_stream": action_stream,
        }
    )
    _attach_score_consumption_evidence_ids(trace)
    return trace


def _attach_score_consumption_evidence_ids(trace: dict[str, Any]) -> None:
    tool_ids = _live_tool_effect_evidence_ids(trace)
    snapshot_ids = []
    for event in _last_record(trace).get("realized_events") or []:
        if (
            isinstance(event, dict)
            and event.get("type") == "sumo_live_snapshot"
            and event.get("evidence_id")
        ):
            snapshot_ids.append(str(event.get("evidence_id")))

    score_ids: dict[str, list[str]] = {}
    if snapshot_ids:
        score_ids["weighted_equity_score"] = list(snapshot_ids)
    if tool_ids:
        score_ids["stakeholder_management"] = list(tool_ids)
    if score_ids:
        trace["score_consumption_evidence_ids"] = score_ids


def _attach_counterfactual_score_evidence_ids(
    candidate_trace: dict[str, Any],
    counterfactual_trace: dict[str, Any],
) -> None:
    score_ids = dict(candidate_trace.get("score_consumption_evidence_ids") or {})
    evidence_ids = _snapshot_evidence_ids(candidate_trace) + _snapshot_evidence_ids(
        counterfactual_trace
    )
    if evidence_ids:
        score_ids["counterfactual_prevention"] = list(dict.fromkeys(evidence_ids))
        candidate_trace["score_consumption_evidence_ids"] = score_ids


def _snapshot_evidence_ids(trace: dict[str, Any] | list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for record in _records(trace):
        for event in record.get("realized_events") or []:
            if (
                isinstance(event, dict)
                and event.get("type") == "sumo_live_snapshot"
                and event.get("evidence_id")
            ):
                out.append(str(event.get("evidence_id")))
    return out


def _base_report() -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "scope": REPORT_SCOPE,
        "non_release_artifact": True,
        "release_ready": False,
        "release_reentry_ready": False,
        "proceed_commands": [],
        "scored_backend": "mock_sumo_deterministic",
        "candidate_live_backend": "sumo",
        "live_scoring_promotion_ready": False,
        "used_mock_fallback_for_live": False,
        "generated_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "policy": {
            "live_trace_required": True,
            "silent_mock_fallback_allowed": False,
            "release_artifact_mutation_allowed": False,
        },
    }


def _replay_traffic_seed(
    *,
    scenario_config: dict[str, Any],
    seed: int,
    action_stream: list[Any],
    n_ticks: int,
) -> dict[str, Any]:
    env = TrafficEnvironment()
    env.reset(scenario_config, seed=seed)
    records: list[dict[str, Any]] = []
    tool_results: list[list[dict[str, Any]]] = []
    control_readbacks: list[dict[str, Any]] = []
    try:
        for tick in range(int(n_ticks)):
            raw_action = (
                action_stream[tick] if tick < len(action_stream) else [{"name": "wait"}]
            )
            ret = env.step(_action_from_raw(raw_action))
            serialized_tool_results = [r.to_dict() for r in ret.tool_results]
            control_readbacks.extend(
                _control_readbacks_from_tool_results(serialized_tool_results)
            )
            backend_record = ret.info.extra.get("backend_tick_record") or {}
            realized_events = _events_with_evidence_ids(
                ret.info.realized_events,
                _realized_event_evidence_ids(env, tick),
            )
            records.append(
                {
                    "tick": tick,
                    "aggregate_queue": _as_float(backend_record.get("aggregate_queue")),
                    "aggregate_delay_minutes": _as_float(
                        backend_record.get("aggregate_delay_minutes")
                    ),
                    "per_corridor_delay_minutes": env.ground_truth().get(
                        "per_corridor_delay_minutes", {}
                    ),
                    "n_gridlocked": int(backend_record.get("n_gridlocked") or 0),
                    "realized_events": realized_events,
                }
            )
            tool_results.append(serialized_tool_results)
            if ret.done:
                break
        gt = env.ground_truth()
        return {
            "backend": str(scenario_config.get("backend_kind", "mock_sumo")),
            "records": records,
            "tool_results": tool_results,
            "signal_program_by_corridor": _signal_program_by_corridor(gt),
            "control_readbacks": control_readbacks,
            "per_corridor_delay_minutes": gt.get("per_corridor_delay_minutes", {}),
            "cost_components": gt.get("cost_components", {}),
            "executed_with_live_backend": scenario_config.get("backend_kind") == "sumo",
        }
    finally:
        env.close()


def _events_with_evidence_ids(
    events: list[dict[str, Any]],
    evidence_ids: list[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    candidates = [str(eid) for eid in evidence_ids]
    cursor = 0
    for event in events:
        row = dict(event)
        if row.get("evidence_id"):
            out.append(row)
            continue
        if cursor < len(candidates):
            row["evidence_id"] = candidates[cursor]
            cursor += 1
        out.append(row)
    return out


def _realized_event_evidence_ids(env: TrafficEnvironment, tick: int) -> list[str]:
    if env.evidence is None:
        return []
    return [
        item.evidence_id
        for item in env.evidence.items_by_kind("realized_event")
        if item.tick == int(tick)
    ]


def _action_from_raw(raw: Any) -> Action:
    if isinstance(raw, dict) and "tool_calls" in raw:
        raw_calls = raw.get("tool_calls") or []
        dominant = raw.get("dominant")
    elif isinstance(raw, dict):
        raw_calls = [raw]
        dominant = raw.get("name")
    elif isinstance(raw, list):
        raw_calls = raw
        dominant = raw[0].get("name") if raw and isinstance(raw[0], dict) else None
    else:
        raw_calls = [{"name": "wait"}]
        dominant = "wait"
    calls = [
        ToolCall(
            name=str(call.get("name", "wait")),
            args=dict(call.get("args") or {}),
            idempotency_key=call.get("idempotency_key"),
            rationale=call.get("rationale"),
        )
        for call in raw_calls
        if isinstance(call, dict)
    ]
    return Action(tool_calls=calls or [ToolCall(name="wait")], dominant=dominant)


def _signal_program_by_corridor(ground_truth: dict[str, Any]) -> dict[str, str]:
    entities = ground_truth.get("entities")
    if not isinstance(entities, dict):
        return {}
    programs: dict[str, str] = {}
    for corridor, entity in entities.items():
        if not isinstance(entity, dict):
            continue
        program = entity.get("signal_program")
        if not isinstance(program, str) or not program:
            continue
        if program == "default" and entity.get("live_sumo_mapped") is not True:
            continue
        programs[str(corridor)] = program
    return programs


def _control_readbacks_from_tool_results(
    tool_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    readbacks: list[dict[str, Any]] = []
    for row in tool_results:
        if not isinstance(row, dict) or row.get("name") != "change_signal_plan":
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        if (
            "sumo_program_readback_matches" not in payload
            and "live_readback_matches_resolved" not in payload
        ):
            continue
        readbacks.append(
            {
                "corridor": payload.get("corridor"),
                "requested_program": payload.get("program"),
                "sumo_tls_id": payload.get("sumo_tls_id"),
                "sumo_program_id": payload.get("sumo_program_id"),
                "sumo_program_readback": payload.get("sumo_program_readback"),
                "sumo_program_readback_available": payload.get(
                    "sumo_program_readback_available"
                ),
                "sumo_program_readback_matches": payload.get(
                    "sumo_program_readback_matches"
                ),
                "live_readback_matches_resolved": payload.get(
                    "live_readback_matches_resolved"
                ),
                "sumo_state_mutated": payload.get("sumo_state_mutated"),
                "ok": row.get("ok"),
            }
        )
    return readbacks


def _compare_traces(
    mock_trace: dict[str, Any] | list[dict[str, Any]],
    live_trace: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    mock_queue = _metric(mock_trace, "aggregate_queue")
    live_queue = _metric(live_trace, "aggregate_queue")
    mock_delay = _metric(mock_trace, "aggregate_delay_minutes")
    live_delay = _metric(live_trace, "aggregate_delay_minutes")

    per_corridor_delta = _per_corridor_delta_minutes(mock_trace, live_trace)
    abs_deltas = {k: abs(v) for k, v in per_corridor_delta.items()}
    priority_delta = _dict_l1_delta(
        _trace_dict(mock_trace).get("priority_outcomes"),
        _trace_dict(live_trace).get("priority_outcomes"),
    )
    spillback_delta = abs(
        _metric(live_trace, "n_gridlocked") - _metric(mock_trace, "n_gridlocked")
    )

    return {
        "aggregate_queue_abs_delta": _round3(abs(live_queue - mock_queue)),
        "aggregate_delay_minutes_abs_delta": _round3(abs(live_delay - mock_delay)),
        "per_corridor_delay_delta_minutes": per_corridor_delta,
        "per_corridor_delay_l1_minutes": _round3(sum(abs_deltas.values())),
        "per_corridor_delay_max_abs_delta_minutes": _round3(
            max(abs_deltas.values(), default=0.0)
        ),
        "per_corridor_delay_changed_corridors": sorted(
            k for k, v in per_corridor_delta.items() if abs(v) > 0.0
        ),
        "signal_program_mismatch_count": _signal_program_mismatch_count(
            mock_trace, live_trace
        ),
        "live_control_readback_match_rate": _control_readback_match_rate(live_trace),
        "priority_outcome_l1_delta": _round3(priority_delta),
        "spillback_proxy_abs_delta": _round3(spillback_delta),
    }


def _fidelity_gate(metrics: dict[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for metric, threshold in FIDELITY_GATE_THRESHOLDS.items():
        if metric == "per_corridor_delay_changed_corridors":
            observed = len(metrics.get(metric) or [])
        else:
            observed = _as_float(metrics.get(metric))
        if observed > threshold:
            failures.append(
                {
                    "metric": metric,
                    "observed": _round3(observed),
                    "threshold": threshold,
                }
            )
    if failures:
        return {
            "status": "blocked_live_rebaseline_required",
            "passed": False,
            "blocker_codes": ["mock_live_fidelity_gap_exceeds_threshold"],
            "metric_failures": failures,
            "thresholds": dict(FIDELITY_GATE_THRESHOLDS),
        }
    return {
        "status": "passed",
        "passed": True,
        "blocker_codes": [],
        "metric_failures": [],
        "thresholds": dict(FIDELITY_GATE_THRESHOLDS),
    }


def _rebaseline_diagnostics(
    metrics: dict[str, Any], fidelity_gate: dict[str, Any]
) -> dict[str, Any]:
    if fidelity_gate.get("passed") is True:
        return {
            "status": "not_required",
            "release_promotion_implication": "fidelity_gate_passed",
            "likely_gap_sources": [],
        }

    failed_metrics = {
        str(row.get("metric"))
        for row in fidelity_gate.get("metric_failures") or []
        if isinstance(row, dict)
    }
    sources: list[dict[str, Any]] = []
    if {
        "aggregate_queue_abs_delta",
        "aggregate_delay_minutes_abs_delta",
    } & failed_metrics:
        sources.append(
            {
                "axis": "aggregate_queue_delay_scale_or_tick_alignment",
                "evidence_metrics": {
                    "aggregate_queue_abs_delta": metrics.get(
                        "aggregate_queue_abs_delta"
                    ),
                    "aggregate_delay_minutes_abs_delta": metrics.get(
                        "aggregate_delay_minutes_abs_delta"
                    ),
                },
                "next_probe": (
                    "Compare mock tick_minutes, SUMO substeps, demand scale, "
                    "and aggregate queue-to-delay conversion on the same replay."
                ),
            }
        )
    if {
        "per_corridor_delay_l1_minutes",
        "per_corridor_delay_max_abs_delta_minutes",
        "per_corridor_delay_changed_corridors",
    } & failed_metrics:
        sources.append(
            {
                "axis": "per_corridor_delay_attribution_mismatch",
                "evidence_metrics": {
                    "per_corridor_delay_l1_minutes": metrics.get(
                        "per_corridor_delay_l1_minutes"
                    ),
                    "per_corridor_delay_max_abs_delta_minutes": metrics.get(
                        "per_corridor_delay_max_abs_delta_minutes"
                    ),
                    "per_corridor_delay_changed_corridors": metrics.get(
                        "per_corridor_delay_changed_corridors"
                    ),
                },
                "next_probe": (
                    "Compare live TLS controlled-lane coverage against "
                    "benchmark corridor definitions and reconcile "
                    "per-corridor delay with aggregate delay."
                ),
            }
        )
    if "signal_program_mismatch_count" in failed_metrics:
        sources.append(
            {
                "axis": "signal_program_mapping_or_readback_mismatch",
                "evidence_metrics": {
                    "signal_program_mismatch_count": metrics.get(
                        "signal_program_mismatch_count"
                    ),
                    "live_control_readback_match_rate": metrics.get(
                        "live_control_readback_match_rate"
                    ),
                },
                "next_probe": (
                    "Re-derive corridor TLS bindings and program ids from the "
                    "locked net, then verify live readback after each control."
                ),
            }
        )
    if "priority_outcome_l1_delta" in failed_metrics:
        sources.append(
            {
                "axis": "priority_outcome_semantics_mismatch",
                "evidence_metrics": {
                    "priority_outcome_l1_delta": metrics.get(
                        "priority_outcome_l1_delta"
                    )
                },
                "next_probe": (
                    "Replay EMS/VIP priority controls and reconcile stakeholder "
                    "delay extraction between mock and live traces."
                ),
            }
        )
    if "spillback_proxy_abs_delta" in failed_metrics:
        sources.append(
            {
                "axis": "spillback_proxy_semantics_mismatch",
                "evidence_metrics": {
                    "spillback_proxy_abs_delta": metrics.get(
                        "spillback_proxy_abs_delta"
                    )
                },
                "next_probe": (
                    "Compare mock gridlock proxy against live blocked/halting "
                    "lane metrics before using spillback for release scoring."
                ),
            }
        )
    if not sources:
        sources.append(
            {
                "axis": "unknown_fidelity_metric_gap",
                "evidence_metrics": dict(metrics),
                "next_probe": (
                    "Inspect the raw mock and live traces and add a narrower "
                    "diagnostic axis before promotion."
                ),
            }
        )
    return {
        "status": "rebaseline_required",
        "release_promotion_implication": "blocked_rebaseline_before_live_scoring",
        "likely_gap_sources": sources,
    }


def _aggregate_rebaseline_alignment(
    mock_trace: dict[str, Any] | list[dict[str, Any]],
    live_trace: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    mock_queue = _metric(mock_trace, "aggregate_queue")
    live_queue = _metric(live_trace, "aggregate_queue")
    mock_delay = _metric(mock_trace, "aggregate_delay_minutes")
    live_delay = _metric(live_trace, "aggregate_delay_minutes")
    mock_conversion = _safe_ratio(mock_delay, mock_queue)
    live_conversion = _safe_ratio(live_delay, live_queue)
    queue_ratio = _safe_ratio(live_queue, mock_queue)
    delay_ratio = _safe_ratio(live_delay, mock_delay)
    ratio_delta = abs(queue_ratio - delay_ratio)
    conversion_delta = abs(live_conversion - mock_conversion)
    return {
        "mock_aggregate_queue": _round3(mock_queue),
        "live_aggregate_queue": _round3(live_queue),
        "mock_aggregate_delay_minutes": _round3(mock_delay),
        "live_aggregate_delay_minutes": _round3(live_delay),
        "aggregate_queue_abs_delta": _round3(abs(live_queue - mock_queue)),
        "aggregate_delay_minutes_abs_delta": _round3(abs(live_delay - mock_delay)),
        "queue_ratio_live_over_mock": _round3(queue_ratio),
        "delay_ratio_live_over_mock": _round3(delay_ratio),
        "queue_to_delay_minutes_mock": _round3(mock_conversion),
        "queue_to_delay_minutes_live": _round3(live_conversion),
        "queue_to_delay_minutes_abs_delta": _round3(conversion_delta),
        "queue_delay_ratio_abs_delta": _round3(ratio_delta),
        "tick_conversion_status": (
            "aligned"
            if conversion_delta
            <= REBASELINE_THRESHOLDS["queue_to_delay_minutes_abs_delta"]
            else "mismatch"
        ),
    }


def _corridor_rebaseline_attribution(
    mock_trace: dict[str, Any] | list[dict[str, Any]],
    live_trace: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    mock_delay = _per_corridor_delay(mock_trace)
    live_delay = _per_corridor_delay(live_trace)
    live_details = _live_per_corridor_details(live_trace)
    mock_sum = sum(mock_delay.values())
    live_sum = sum(live_delay.values())
    mock_total = _metric(mock_trace, "aggregate_delay_minutes")
    live_total = _metric(live_trace, "aggregate_delay_minutes")
    live_lane_queue = sum(_as_float(row.get("queue")) for row in live_details.values())
    live_aggregate_queue = _metric(live_trace, "aggregate_queue")
    live_corridors = set(live_delay) | set(live_details)
    mock_corridors = set(mock_delay)
    binding_corridors = _binding_corridors()
    return {
        "mock_corridor_delay_sum_minutes": _round3(mock_sum),
        "live_corridor_delay_sum_minutes": _round3(live_sum),
        "mock_corridor_delay_coverage_ratio": _round3(
            _safe_ratio(mock_sum, mock_total)
        ),
        "live_corridor_delay_coverage_ratio": _round3(
            _safe_ratio(live_sum, live_total)
        ),
        "live_unattributed_delay_minutes": _round3(max(0.0, live_total - live_sum)),
        "live_controlled_lane_queue": _round3(live_lane_queue),
        "live_controlled_lane_queue_share": _round3(
            _safe_ratio(live_lane_queue, live_aggregate_queue)
        ),
        "binding_corridor_count": len(binding_corridors),
        "mock_observed_corridor_count": len(mock_corridors),
        "live_observed_corridor_count": len(live_corridors),
        "missing_live_corridors": sorted(binding_corridors - live_corridors),
        "extra_live_corridors": sorted(live_corridors - binding_corridors),
        "per_corridor_live_details": live_details,
    }


def _rebaseline_blockers(
    aggregate: dict[str, Any], corridor: dict[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if (
        _as_float(aggregate.get("aggregate_queue_abs_delta"))
        > REBASELINE_THRESHOLDS["aggregate_queue_abs_delta"]
        or _as_float(aggregate.get("aggregate_delay_minutes_abs_delta"))
        > REBASELINE_THRESHOLDS["aggregate_delay_minutes_abs_delta"]
        or _as_float(aggregate.get("queue_delay_ratio_abs_delta"))
        > REBASELINE_THRESHOLDS["aggregate_ratio_abs_delta"]
        or _as_float(aggregate.get("queue_to_delay_minutes_abs_delta"))
        > REBASELINE_THRESHOLDS["queue_to_delay_minutes_abs_delta"]
    ):
        blockers.append("aggregate_queue_or_delay_gap_exceeds_threshold")
    if (
        _as_float(corridor.get("live_corridor_delay_coverage_ratio"))
        < REBASELINE_THRESHOLDS["corridor_delay_coverage_min"]
    ):
        blockers.append("live_corridor_delay_under_attributed")
    if (
        _as_float(corridor.get("live_controlled_lane_queue_share"))
        < REBASELINE_THRESHOLDS["live_controlled_lane_queue_share_min"]
    ):
        blockers.append("live_controlled_lane_queue_under_coverage")
    if corridor.get("missing_live_corridors"):
        blockers.append("missing_live_corridor_bindings")
    return blockers


def _live_per_corridor_details(
    trace: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    for event in _last_record(trace).get("realized_events") or []:
        if not isinstance(event, dict):
            continue
        per_corridor = event.get("per_corridor")
        if not isinstance(per_corridor, dict):
            continue
        for corridor, raw in per_corridor.items():
            if not isinstance(raw, dict):
                continue
            details[str(corridor)] = {
                "queue": _round3(_as_float(raw.get("queue"))),
                "vehicles": _round3(_as_float(raw.get("vehicles"))),
                "n_lanes": int(_as_float(raw.get("n_lanes"))),
                "cumulative_delay_minutes": _round3(
                    _as_float(raw.get("cumulative_delay_minutes"))
                ),
                "delay_minutes_increment": _round3(
                    _as_float(raw.get("delay_minutes_increment"))
                ),
                "waiting_time_s": _round3(_as_float(raw.get("waiting_time_s"))),
            }
    return details


def _live_network_totals(
    trace: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    snapshot = _last_live_snapshot_event(trace)
    return {
        "aggregate_queue": _round3(_metric(trace, "aggregate_queue")),
        "aggregate_delay_minutes": _round3(_metric(trace, "aggregate_delay_minutes")),
        "snapshot_n_vehicles": _round3(_as_float(snapshot.get("n_vehicles"))),
        "snapshot_departed": _round3(_as_float(snapshot.get("departed"))),
        "snapshot_arrived": _round3(_as_float(snapshot.get("arrived"))),
        "snapshot_tick": int(_as_float(snapshot.get("tick"))),
        "transport": snapshot.get("transport"),
    }


def _benchmark_corridor_attribution(
    trace: dict[str, Any] | list[dict[str, Any]],
    network: dict[str, Any],
) -> dict[str, Any]:
    delay = _per_corridor_delay(trace)
    binding_corridors = _binding_corridors()
    observed_corridors = set(delay)
    delay_sum = sum(delay.values())
    aggregate_delay = _as_float(network.get("aggregate_delay_minutes"))
    return {
        "delay_sum_minutes": _round3(delay_sum),
        "delay_coverage_ratio": _round3(_safe_ratio(delay_sum, aggregate_delay)),
        "unattributed_delay_minutes": _round3(max(0.0, aggregate_delay - delay_sum)),
        "binding_corridor_count": len(binding_corridors),
        "observed_corridor_count": len(observed_corridors),
        "missing_bound_corridors": sorted(binding_corridors - observed_corridors),
        "extra_observed_corridors": sorted(observed_corridors - binding_corridors),
        "per_corridor_delay_minutes": {
            corridor: _round3(value) for corridor, value in sorted(delay.items())
        },
    }


def _tls_controlled_lane_attribution(
    trace: dict[str, Any] | list[dict[str, Any]],
    network: dict[str, Any],
) -> dict[str, Any]:
    details = _live_per_corridor_details(trace)
    queue = sum(_as_float(row.get("queue")) for row in details.values())
    vehicles = sum(_as_float(row.get("vehicles")) for row in details.values())
    delay = sum(
        _as_float(row.get("cumulative_delay_minutes")) for row in details.values()
    )
    waiting_s = sum(_as_float(row.get("waiting_time_s")) for row in details.values())
    n_lanes = sum(int(_as_float(row.get("n_lanes"))) for row in details.values())
    return {
        "queue": _round3(queue),
        "vehicles": _round3(vehicles),
        "cumulative_delay_minutes": _round3(delay),
        "waiting_time_s": _round3(waiting_s),
        "n_lanes": n_lanes,
        "corridor_count": len(details),
        "queue_share_of_network": _round3(
            _safe_ratio(queue, _as_float(network.get("aggregate_queue")))
        ),
        "vehicle_share_of_network": _round3(
            _safe_ratio(vehicles, _as_float(network.get("snapshot_n_vehicles")))
        ),
        "delay_share_of_network": _round3(
            _safe_ratio(delay, _as_float(network.get("aggregate_delay_minutes")))
        ),
        "per_corridor_details": details,
    }


def _live_mapping_coverage(
    trace: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    snapshot = _last_live_snapshot_event(trace)
    raw = snapshot.get("attribution_coverage")
    if isinstance(raw, dict):
        n_lanes = raw.get("network_lane_count")
        unique_lanes = int(_as_float(raw.get("unique_controlled_lanes")))
        return {
            "bound_corridor_count": int(_as_float(raw.get("bound_corridor_count"))),
            "bound_tls_count": int(_as_float(raw.get("bound_tls_count"))),
            "tls_with_controlled_lanes": int(
                _as_float(raw.get("tls_with_controlled_lanes"))
            ),
            "unique_controlled_lanes": unique_lanes,
            "network_lane_count": (
                int(_as_float(n_lanes)) if n_lanes is not None else None
            ),
            "network_edge_count": (
                int(_as_float(raw.get("network_edge_count")))
                if raw.get("network_edge_count") is not None
                else None
            ),
            "controlled_lane_share_of_network": (
                _round3(_as_float(raw.get("controlled_lane_share_of_network")))
                if raw.get("controlled_lane_share_of_network") is not None
                else None
            ),
            "unattributed_network_lanes_estimate": (
                int(_as_float(raw.get("unattributed_network_lanes_estimate")))
                if raw.get("unattributed_network_lanes_estimate") is not None
                else None
            ),
            "zero_lane_corridors": [
                str(v) for v in list(raw.get("zero_lane_corridors") or [])
            ],
            "missing_network_lane_denominator": bool(
                raw.get("missing_network_lane_denominator")
            ),
            "denominator_status": (
                "network_lane_denominator_missing"
                if raw.get("missing_network_lane_denominator")
                else "network_lane_denominator_available"
            ),
        }
    details = _live_per_corridor_details(trace)
    unique_lanes = sum(int(_as_float(row.get("n_lanes"))) for row in details.values())
    return {
        "bound_corridor_count": len(_binding_corridors()),
        "bound_tls_count": None,
        "tls_with_controlled_lanes": sum(1 for row in details.values() if row),
        "unique_controlled_lanes": unique_lanes,
        "network_lane_count": None,
        "network_edge_count": None,
        "controlled_lane_share_of_network": None,
        "unattributed_network_lanes_estimate": None,
        "zero_lane_corridors": [
            corridor
            for corridor, row in sorted(details.items())
            if int(_as_float(row.get("n_lanes"))) == 0
        ],
        "missing_network_lane_denominator": True,
        "denominator_status": "network_lane_denominator_missing",
    }


def _live_attribution_blockers(
    corridor: dict[str, Any], tls: dict[str, Any]
) -> list[str]:
    blockers: list[str] = []
    if (
        _as_float(corridor.get("delay_coverage_ratio"))
        < LIVE_ATTRIBUTION_THRESHOLDS["network_to_corridor_delay_coverage_min"]
    ):
        blockers.append("network_to_corridor_delay_under_coverage")
    if (
        _as_float(tls.get("queue_share_of_network"))
        < LIVE_ATTRIBUTION_THRESHOLDS["network_to_tls_controlled_queue_share_min"]
    ):
        blockers.append("network_to_tls_controlled_queue_under_coverage")
    if (
        _as_float(tls.get("vehicle_share_of_network"))
        < LIVE_ATTRIBUTION_THRESHOLDS["network_to_tls_controlled_vehicle_share_min"]
    ):
        blockers.append("network_to_tls_controlled_vehicle_under_coverage")
    if blockers:
        blockers.append(
            "attribution_basis_mismatch_network_total_vs_tls_controlled_lanes"
        )
    return blockers


def _fidelity_target_split_from_rebaseline(
    *,
    aggregate: dict[str, Any],
    corridor: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    aggregate_blocked = "aggregate_queue_or_delay_gap_exceeds_threshold" in blockers
    corridor_blocked = bool(
        {
            "live_corridor_delay_under_attributed",
            "live_controlled_lane_queue_under_coverage",
            "missing_live_corridor_bindings",
        }
        & set(blockers)
    )
    basis_mismatch = _corridor_basis_mismatch_from_values(
        delay_coverage_ratio=_as_float(
            corridor.get("live_corridor_delay_coverage_ratio")
        ),
        queue_share=_as_float(corridor.get("live_controlled_lane_queue_share")),
    )
    split_blockers = list(blockers)
    if basis_mismatch:
        split_blockers.append(
            "attribution_basis_mismatch_network_total_vs_tls_controlled_lanes"
        )
    return {
        "status": "blocked" if split_blockers else "aligned",
        "network_total_fidelity": {
            "status": "rebaseline_required" if aggregate_blocked else "aligned",
            "mock_aggregate_queue": aggregate.get("mock_aggregate_queue"),
            "live_aggregate_queue": aggregate.get("live_aggregate_queue"),
            "mock_aggregate_delay_minutes": aggregate.get(
                "mock_aggregate_delay_minutes"
            ),
            "live_aggregate_delay_minutes": aggregate.get(
                "live_aggregate_delay_minutes"
            ),
            "queue_ratio_live_over_mock": aggregate.get("queue_ratio_live_over_mock"),
            "delay_ratio_live_over_mock": aggregate.get("delay_ratio_live_over_mock"),
            "tick_conversion_status": aggregate.get("tick_conversion_status"),
        },
        "tls_controlled_corridor_attribution": {
            "status": "under_attributed" if corridor_blocked else "aligned",
            "live_corridor_delay_coverage_ratio": corridor.get(
                "live_corridor_delay_coverage_ratio"
            ),
            "live_controlled_lane_queue_share": corridor.get(
                "live_controlled_lane_queue_share"
            ),
            "live_unattributed_delay_minutes": corridor.get(
                "live_unattributed_delay_minutes"
            ),
            "missing_live_corridors": list(
                corridor.get("missing_live_corridors") or []
            ),
        },
        "measurement_basis": _target_split_measurement_basis(),
        "basis_mismatch_detected": basis_mismatch,
        "blocker_codes": sorted(dict.fromkeys(split_blockers)),
        "recommended_next_action": (
            "separate_network_total_from_tls_controlled_corridor_target"
            if basis_mismatch
            else (
                "rebaseline_network_total_fidelity"
                if aggregate_blocked
                else "keep_current_attribution_target"
            )
        ),
    }


def _fidelity_target_split_from_metrics(
    metrics: dict[str, Any], fidelity_gate: dict[str, Any]
) -> dict[str, Any]:
    failed_metrics = {
        str(row.get("metric"))
        for row in fidelity_gate.get("metric_failures") or []
        if isinstance(row, dict)
    }
    aggregate_blocked = bool(
        {"aggregate_queue_abs_delta", "aggregate_delay_minutes_abs_delta"}
        & failed_metrics
    )
    corridor_blocked = bool(
        {
            "per_corridor_delay_l1_minutes",
            "per_corridor_delay_max_abs_delta_minutes",
            "per_corridor_delay_changed_corridors",
        }
        & failed_metrics
    )
    return {
        "status": "blocked" if fidelity_gate.get("passed") is not True else "aligned",
        "network_total_fidelity": {
            "status": "rebaseline_required" if aggregate_blocked else "aligned",
            "aggregate_queue_abs_delta": metrics.get("aggregate_queue_abs_delta"),
            "aggregate_delay_minutes_abs_delta": metrics.get(
                "aggregate_delay_minutes_abs_delta"
            ),
        },
        "tls_controlled_corridor_attribution": {
            "status": "under_attributed" if corridor_blocked else "aligned",
            "per_corridor_delay_l1_minutes": metrics.get(
                "per_corridor_delay_l1_minutes"
            ),
            "per_corridor_delay_max_abs_delta_minutes": metrics.get(
                "per_corridor_delay_max_abs_delta_minutes"
            ),
            "per_corridor_delay_changed_corridors": list(
                metrics.get("per_corridor_delay_changed_corridors") or []
            ),
        },
        "measurement_basis": _target_split_measurement_basis(),
        "basis_mismatch_detected": corridor_blocked,
        "blocker_codes": list(fidelity_gate.get("blocker_codes") or []),
        "recommended_next_action": (
            "separate_network_total_from_tls_controlled_corridor_target"
            if corridor_blocked
            else (
                "rebaseline_network_total_fidelity"
                if aggregate_blocked
                else "keep_current_attribution_target"
            )
        ),
    }


def _fidelity_target_split_from_live_attribution(
    *,
    network: dict[str, Any],
    corridor: dict[str, Any],
    tls: dict[str, Any],
    measurement_basis: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    basis_mismatch = (
        "attribution_basis_mismatch_network_total_vs_tls_controlled_lanes" in blockers
    )
    return {
        "status": "blocked" if blockers else "aligned",
        "network_total_fidelity": {
            "status": "not_evaluated_without_mock_trace",
            "live_aggregate_queue": network.get("aggregate_queue"),
            "live_aggregate_delay_minutes": network.get("aggregate_delay_minutes"),
            "snapshot_n_vehicles": network.get("snapshot_n_vehicles"),
        },
        "tls_controlled_corridor_attribution": {
            "status": "under_attributed" if blockers else "aligned",
            "delay_coverage_ratio": corridor.get("delay_coverage_ratio"),
            "queue_share_of_network": tls.get("queue_share_of_network"),
            "vehicle_share_of_network": tls.get("vehicle_share_of_network"),
            "delay_share_of_network": tls.get("delay_share_of_network"),
            "unattributed_delay_minutes": corridor.get("unattributed_delay_minutes"),
        },
        "measurement_basis": {
            **_target_split_measurement_basis(),
            "source_measurement_basis": measurement_basis,
        },
        "basis_mismatch_detected": basis_mismatch,
        "blocker_codes": sorted(dict.fromkeys(blockers)),
        "recommended_next_action": (
            "separate_network_total_from_tls_controlled_corridor_target"
            if basis_mismatch
            else "keep_current_attribution_target"
        ),
    }


def _split_fidelity_gate_from_target_split(
    target_split: dict[str, Any],
) -> dict[str, Any]:
    network = dict(target_split.get("network_total_fidelity") or {})
    tls = dict(target_split.get("tls_controlled_corridor_attribution") or {})
    network_status = str(network.get("status") or "not_evaluated")
    tls_status = str(tls.get("status") or "not_evaluated")
    network_passed = network_status == "aligned"
    tls_passed = tls_status == "aligned"
    blockers: list[str] = []
    if not network_passed:
        blockers.append(
            "traffic_network_total_fidelity_rebaseline_required"
            if network_status == "rebaseline_required"
            else "traffic_network_total_fidelity_not_verified"
        )
    if not tls_passed:
        blockers.append(
            "traffic_tls_corridor_attribution_under_coverage"
            if tls_status == "under_attributed"
            else "traffic_tls_corridor_attribution_not_verified"
        )
    passed = not blockers
    return {
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "blocker_codes": sorted(dict.fromkeys(blockers)),
        "network_total_fidelity": {
            **network,
            "passed": network_passed,
        },
        "tls_controlled_corridor_attribution": {
            **tls,
            "passed": tls_passed,
        },
        "measurement_basis": dict(target_split.get("measurement_basis") or {}),
        "basis_mismatch_detected": target_split.get("basis_mismatch_detected") is True,
        "legacy_target_split_status": target_split.get("status"),
    }


def _target_decision_assessment(
    *,
    fidelity_report: dict[str, Any],
    rebaseline_report: dict[str, Any],
    live_attribution_report: dict[str, Any],
) -> dict[str, Any]:
    split = fidelity_report.get("fidelity_target_split") or {}
    rebaseline_split = rebaseline_report.get("fidelity_target_split") or {}
    attribution_split = live_attribution_report.get("fidelity_target_split") or {}
    network = _first_dict(
        rebaseline_split.get("network_total_fidelity"),
        split.get("network_total_fidelity"),
        attribution_split.get("network_total_fidelity"),
    )
    corridor = _first_dict(
        rebaseline_split.get("tls_controlled_corridor_attribution"),
        attribution_split.get("tls_controlled_corridor_attribution"),
        split.get("tls_controlled_corridor_attribution"),
    )
    signal_readback = _signal_program_readback_assessment(fidelity_report)
    priority = _thresholded_metric_assessment(
        fidelity_report=fidelity_report,
        metric="priority_outcome_l1_delta",
        threshold=FIDELITY_GATE_THRESHOLDS["priority_outcome_l1_delta"],
        label="priority_semantics",
    )
    spillback = _thresholded_metric_assessment(
        fidelity_report=fidelity_report,
        metric="spillback_proxy_abs_delta",
        threshold=FIDELITY_GATE_THRESHOLDS["spillback_proxy_abs_delta"],
        label="spillback_proxy",
    )
    basis_mismatch = any(
        report.get("basis_mismatch_detected") is True
        for report in (split, rebaseline_split, attribution_split)
        if isinstance(report, dict)
    ) or (
        "attribution_basis_mismatch_network_total_vs_tls_controlled_lanes"
        in set(live_attribution_report.get("blocker_codes") or [])
    )
    mapping = _mapping_coverage_assessment(live_attribution_report)
    corridor_target = _corridor_scoped_live_target_candidate(
        fidelity_report=fidelity_report,
        corridor=corridor,
        mapping=mapping,
        signal_readback=signal_readback,
    )
    return {
        "network_total_fidelity": network or {"status": "not_evaluated"},
        "tls_controlled_corridor_attribution": corridor or {"status": "not_evaluated"},
        "mapping_coverage": mapping,
        "corridor_scoped_live_target_candidate": corridor_target,
        "signal_program_readback": signal_readback,
        "priority_semantics": priority,
        "spillback_proxy": spillback,
        "basis_mismatch_detected": bool(
            basis_mismatch or mapping.get("network_wide_scoring_represented") is False
        ),
        "measurement_basis": _target_split_measurement_basis(),
    }


def _target_decision_blockers(
    *,
    target_assessment: dict[str, Any],
    basis_mismatch: bool,
    fidelity_passed: bool,
) -> list[str]:
    blockers: list[str] = []
    if basis_mismatch:
        blockers.append("live_scoring_blocked_pending_split_fidelity_gate")
    if (target_assessment.get("mapping_coverage") or {}).get(
        "network_wide_scoring_represented"
    ) is False:
        blockers.append("live_network_wide_scoring_denominator_not_represented")
    if not fidelity_passed:
        blockers.append("mock_live_fidelity_gap_exceeds_threshold")
    for key, blocker in (
        ("signal_program_readback", "signal_program_readback_not_verified"),
        ("priority_semantics", "priority_semantics_not_verified"),
        ("spillback_proxy", "spillback_proxy_not_verified"),
    ):
        status = (target_assessment.get(key) or {}).get("status")
        if status in {"mismatch", "blocked", "not_evaluated"}:
            blockers.append(blocker)
    return sorted(dict.fromkeys(blockers))


def _target_decision_chosen_path(
    *,
    target_assessment: dict[str, Any],
    basis_mismatch: bool,
    fidelity_passed: bool,
) -> str:
    if (target_assessment.get("mapping_coverage") or {}).get(
        "network_wide_scoring_represented"
    ) is False:
        return "keep_mock_scoring_until_live_denominator_redefined"
    if basis_mismatch:
        return "split_network_total_and_tls_controlled_corridor_targets"
    if not fidelity_passed:
        return "rebaseline_current_fidelity_target"
    if (target_assessment.get("signal_program_readback") or {}).get(
        "status"
    ) == "mismatch":
        return "repair_signal_program_readback"
    return "keep_current_fidelity_target"


def _target_decision_options(
    *,
    target_assessment: dict[str, Any],
    chosen_path: str,
) -> dict[str, Any]:
    tls = target_assessment.get("tls_controlled_corridor_attribution") or {}
    mapping = target_assessment.get("mapping_coverage") or {}
    delay_coverage = _as_float(
        tls.get("delay_coverage_ratio") or tls.get("live_corridor_delay_coverage_ratio")
    )
    queue_share = _as_float(
        tls.get("queue_share_of_network") or tls.get("live_controlled_lane_queue_share")
    )
    return {
        "split_fidelity_gate": {
            "status": (
                "selected"
                if chosen_path
                == "split_network_total_and_tls_controlled_corridor_targets"
                else "available"
            ),
            "description": (
                "Gate network-total queue/delay separately from "
                "TLS-controlled corridor attribution before live scoring."
            ),
        },
        "expand_tls_lane_attribution": {
            "status": (
                "blocked_current_coverage_below_threshold"
                if delay_coverage
                < LIVE_ATTRIBUTION_THRESHOLDS["network_to_corridor_delay_coverage_min"]
                or queue_share
                < LIVE_ATTRIBUTION_THRESHOLDS[
                    "network_to_tls_controlled_queue_share_min"
                ]
                else "available_for_review"
            ),
            "delay_coverage_ratio": _round3(delay_coverage),
            "queue_share_of_network": _round3(queue_share),
        },
        "corridor_scoped_live_target": _corridor_scoped_live_target_option(
            target_assessment
        ),
        "keep_live_scoring_blocked": {
            "status": "selected_until_target_gate_passes",
            "description": (
                "Live SUMO remains cited/proof-only until the selected target "
                "gate passes and case ledgers are materialized."
            ),
        },
        "keep_mock_scoring_as_canonical": {
            "status": (
                "selected_until_live_denominator_redefined_or_coverage_expanded"
                if mapping.get("network_wide_scoring_represented") is False
                else "available_for_review"
            ),
            "controlled_lane_share_of_network": mapping.get(
                "controlled_lane_share_of_network"
            ),
            "network_lane_count": mapping.get("network_lane_count"),
            "unique_controlled_lanes": mapping.get("unique_controlled_lanes"),
        },
    }


def _target_decision_next_actions(
    *,
    target_assessment: dict[str, Any],
    chosen_path: str,
    blockers: list[str],
) -> list[str]:
    if not blockers:
        return ["review_case_ledger_preview_before_release_materialization"]
    if "live_network_wide_scoring_denominator_not_represented" in blockers:
        actions = []
        corridor_candidate = (
            target_assessment.get("corridor_scoped_live_target_candidate") or {}
        )
        candidate = (
            chosen_path == "keep_mock_scoring_until_live_denominator_redefined"
            and corridor_candidate.get("candidate_ready") is True
        )
        if candidate:
            actions.append("design_corridor_scoped_live_scoring_contract")
        actions.extend(
            [
                "define_tls_controlled_corridor_as_live_scoring_target_or_expand_mapping",
                "rerun_live_attribution_probe_with_denominator_coverage",
                "keep_mock_sumo_scoring_canonical_for_network_wide_traffic_release",
            ]
        )
        return actions
    if chosen_path == "split_network_total_and_tls_controlled_corridor_targets":
        return [
            "implement_split_fidelity_gate",
            "rerun_mock_live_fidelity_with_split_targets",
            "keep_live_scoring_out_of_release_until_split_gate_passes",
        ]
    if chosen_path == "rebaseline_current_fidelity_target":
        return [
            "rebaseline_current_fidelity_thresholds",
            "rerun_mock_live_fidelity_probe",
            "keep_live_scoring_out_of_release_until_fidelity_gate_passes",
        ]
    return ["repair_live_control_semantics_then_rerun_fidelity_gate"]


def _mapping_coverage_assessment(
    live_attribution_report: dict[str, Any],
) -> dict[str, Any]:
    mapping = live_attribution_report.get("mapping_coverage") or {}
    if not isinstance(mapping, dict) or not mapping:
        return {
            "status": "not_evaluated",
            "network_wide_scoring_represented": None,
            "threshold": LIVE_ATTRIBUTION_THRESHOLDS[
                "network_to_tls_controlled_lane_share_min"
            ],
        }
    share = mapping.get("controlled_lane_share_of_network")
    missing_denominator = mapping.get("missing_network_lane_denominator") is True
    represented = (
        False
        if missing_denominator
        else _as_float(share)
        >= LIVE_ATTRIBUTION_THRESHOLDS["network_to_tls_controlled_lane_share_min"]
    )
    return {
        **mapping,
        "status": (
            "under_represented_for_network_wide_scoring"
            if represented is False
            else "represented_for_network_wide_scoring"
        ),
        "threshold": LIVE_ATTRIBUTION_THRESHOLDS[
            "network_to_tls_controlled_lane_share_min"
        ],
        "network_wide_scoring_represented": represented,
    }


def _corridor_scoped_live_target_candidate(
    *,
    fidelity_report: dict[str, Any],
    corridor: dict[str, Any],
    mapping: dict[str, Any],
    signal_readback: dict[str, Any],
) -> dict[str, Any]:
    bound = int(_as_float(mapping.get("bound_corridor_count")))
    with_lanes = int(_as_float(mapping.get("tls_with_controlled_lanes")))
    zero_lane_corridors = [str(v) for v in mapping.get("zero_lane_corridors") or []]
    network_represented = mapping.get("network_wide_scoring_represented") is True
    live_executed = fidelity_report.get("executed_with_live_backend") is True
    readback_aligned = signal_readback.get("status") == "aligned"
    all_bound_corridors_have_lanes = bool(bound) and with_lanes == bound
    observed_corridors = int(
        _as_float(
            corridor.get("observed_corridor_count")
            or corridor.get("live_observed_corridor_count")
        )
    )
    candidate_ready = (
        live_executed
        and readback_aligned
        and all_bound_corridors_have_lanes
        and not zero_lane_corridors
        and not network_represented
    )
    missing: list[str] = []
    if not live_executed:
        missing.append("live_backend_execution")
    if not readback_aligned:
        missing.append("signal_program_readback_alignment")
    if not all_bound_corridors_have_lanes or zero_lane_corridors:
        missing.append("tls_controlled_lane_binding_for_all_corridors")
    return {
        "status": "candidate_ready" if candidate_ready else "blocked",
        "target_id": "tls_controlled_corridor_delay",
        "candidate_ready": candidate_ready,
        "release_scope": "live_corridor_scoped_not_network_wide",
        "scoring_basis": "per_corridor_delay_minutes_from_tls_controlled_lanes",
        "counterfactual_basis": "same_seed_same_action_stream_on_live_sumo",
        "bound_corridor_count": bound,
        "observed_corridor_count": observed_corridors,
        "tls_with_controlled_lanes": with_lanes,
        "unique_controlled_lanes": mapping.get("unique_controlled_lanes"),
        "zero_lane_corridors": zero_lane_corridors,
        "signal_program_readback_status": signal_readback.get("status"),
        "live_control_readback_match_rate": signal_readback.get(
            "live_control_readback_match_rate"
        ),
        "network_wide_blockers_remaining": (
            ["network_wide_denominator_under_represented"]
            if not network_represented
            else []
        ),
        "evidence_requirements_met": [
            name
            for name, ok in (
                ("live_backend_executed", live_executed),
                ("signal_program_readback_aligned", readback_aligned),
                (
                    "all_bound_corridors_have_controlled_lanes",
                    all_bound_corridors_have_lanes,
                ),
            )
            if ok
        ],
        "missing_requirements": missing,
        "next_required_contract": (
            "Define a live Traffic scoring contract whose denominator is the "
            "TLS-controlled corridor set, then materialize live case ledgers "
            "and counterfactual replay evidence before release scoring."
        ),
    }


def _corridor_scoped_live_target_option(
    target_assessment: dict[str, Any],
) -> dict[str, Any]:
    candidate = target_assessment.get("corridor_scoped_live_target_candidate") or {}
    ready = candidate.get("candidate_ready") is True
    return {
        "status": (
            "candidate_ready_for_design" if ready else "blocked_until_candidate_ready"
        ),
        "target_id": candidate.get("target_id", "tls_controlled_corridor_delay"),
        "release_scope": candidate.get("release_scope"),
        "scoring_basis": candidate.get("scoring_basis"),
        "missing_requirements": list(candidate.get("missing_requirements") or []),
    }


def _corridor_scoped_live_scoring_contract(
    target_assessment: dict[str, Any],
) -> dict[str, Any]:
    candidate = target_assessment.get("corridor_scoped_live_target_candidate") or {}
    ready = candidate.get("candidate_ready") is True
    mapping = target_assessment.get("mapping_coverage") or {}
    return {
        "status": (
            "draft_ready_non_release" if ready else "blocked_candidate_not_ready"
        ),
        "target_id": candidate.get("target_id", "tls_controlled_corridor_delay"),
        "contract_scope": "non_release_live_traffic_target_definition",
        "live_scoring_allowed": False,
        "release_materializer_allowed": False,
        "release_scope": candidate.get("release_scope"),
        "scoring_denominator": {
            "kind": "tls_controlled_corridor_set",
            "basis": "corridors_with_source_locked_tls_lane_bindings",
            "bound_corridor_count": candidate.get("bound_corridor_count"),
            "tls_with_controlled_lanes": candidate.get("tls_with_controlled_lanes"),
            "unique_controlled_lanes": candidate.get("unique_controlled_lanes"),
            "network_lane_count": mapping.get("network_lane_count"),
            "controlled_lane_share_of_network": mapping.get(
                "controlled_lane_share_of_network"
            ),
            "network_wide_denominator_excluded": True,
            "network_wide_exclusion_reason": (
                "tls_controlled_lane_mapping_under_represented_for_network_wide_scoring"
            ),
        },
        "scoring_metrics": [
            "per_corridor_delay_minutes",
            "per_corridor_queue",
            "priority_outcome_delay_minutes",
            "signal_plan_readback_match",
        ],
        "required_evidence_kinds": [
            "live_backend_execution",
            "signal_program_readback",
            "sumo_live_snapshot",
            "per_corridor_delay_and_queue",
            "tool_effect_evidence",
            "score_consumption_evidence",
            "counterfactual_replay_evidence",
        ],
        "counterfactual_contract": {
            "required_before_release": True,
            "basis": "same_seed_same_live_sumo_backend_action_stream_masked",
            "fallback_if_unavailable": "mark_counterfactual_prevention_not_applicable_with_machine_reason",
        },
        "case_ledger_fields_required": [
            "live_scored_backend_descriptor",
            "live_corridor_denominator_key",
            "live_fidelity_gate_citation",
            "live_replay_determinism_citation",
            "live_tool_effect_evidence_ids",
            "live_score_consumption_evidence_ids",
            "live_counterfactual_replay_citation",
        ],
        "release_gates_required": [
            "corridor_scoped_baseline_oracle_headroom_gate",
            "live_replay_determinism_gate",
            "live_counterfactual_replay_gate",
            "score_consumption_evidence_gate",
            "case_ledger_materialization_gate",
            "release_wrapper_materialization_gate",
        ],
        "missing_requirements": list(candidate.get("missing_requirements") or []),
        "policy": {
            "non_release_artifact": True,
            "release_artifact_mutation_allowed": False,
            "scenario_yaml_mutation_allowed": False,
            "mock_as_live_evidence_allowed": False,
            "network_wide_live_scoring_allowed": False,
        },
    }


def _release_promotion_decision(
    *,
    live_scoring_allowed: bool,
    case_ledger_preview_allowed: bool,
    release_materializer_allowed: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "live_scoring_allowed": bool(live_scoring_allowed),
        "case_ledger_preview_allowed": bool(case_ledger_preview_allowed),
        "release_materializer_allowed": bool(release_materializer_allowed),
        "reason": reason,
    }


def _signal_program_readback_assessment(
    fidelity_report: dict[str, Any],
) -> dict[str, Any]:
    metrics = fidelity_report.get("fidelity_metrics") or {}
    mismatch_count = _as_float(metrics.get("signal_program_mismatch_count"))
    readback_rate = metrics.get("live_control_readback_match_rate")
    if readback_rate is None:
        return {
            "status": "not_evaluated",
            "signal_program_mismatch_count": int(mismatch_count),
            "live_control_readback_match_rate": None,
        }
    readback_mismatch = _as_float(readback_rate) < 1.0
    return {
        "status": "mismatch" if mismatch_count > 0 or readback_mismatch else "aligned",
        "signal_program_mismatch_count": int(mismatch_count),
        "live_control_readback_match_rate": readback_rate,
    }


def _thresholded_metric_assessment(
    *,
    fidelity_report: dict[str, Any],
    metric: str,
    threshold: float,
    label: str,
) -> dict[str, Any]:
    metrics = fidelity_report.get("fidelity_metrics") or {}
    if metric not in metrics:
        return {"status": "not_evaluated", "metric": metric, "label": label}
    observed = _as_float(metrics.get(metric))
    return {
        "status": "mismatch" if observed > threshold else "aligned",
        "metric": metric,
        "label": label,
        "observed": _round3(observed),
        "threshold": threshold,
    }


def _first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return dict(value)
    return {}


def _fidelity_target_split_without_live_trace(blockers: list[str]) -> dict[str, Any]:
    return {
        "status": "blocked_missing_live_trace",
        "network_total_fidelity": {
            "status": "not_evaluated_without_live_trace",
        },
        "tls_controlled_corridor_attribution": {
            "status": "not_evaluated_without_live_trace",
        },
        "measurement_basis": _target_split_measurement_basis(),
        "basis_mismatch_detected": False,
        "blocker_codes": sorted(dict.fromkeys(blockers)),
        "recommended_next_action": "capture_live_trace_then_evaluate_target_split",
    }


def _target_split_measurement_basis() -> dict[str, Any]:
    return {
        "network_total_basis": "network_vehicle_count_x_tick_minutes",
        "corridor_attribution_basis": "tls_controlled_lane_halting_integral",
        "direct_comparison_requires_explicit_target_split": True,
    }


def _corridor_basis_mismatch_from_values(
    *, delay_coverage_ratio: float, queue_share: float
) -> bool:
    return (
        delay_coverage_ratio
        < LIVE_ATTRIBUTION_THRESHOLDS["network_to_corridor_delay_coverage_min"]
        or queue_share
        < LIVE_ATTRIBUTION_THRESHOLDS["network_to_tls_controlled_queue_share_min"]
    )


def _last_live_snapshot_event(
    trace: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, Any]:
    for event in _last_record(trace).get("realized_events") or []:
        if isinstance(event, dict) and event.get("type") == "sumo_live_snapshot":
            return event
    return {}


def _binding_corridors() -> set[str]:
    try:
        seed = build_traffic_seed(
            seed_id="rebaseline/binding", family="incident_response"
        )
    except Exception:
        return set()
    return set(dict(seed.backend_config.get("corridor_tls_map") or {}))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-9:
        return 0.0
    return float(numerator) / float(denominator)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_dict(trace: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(trace, dict):
        return trace
    return {"records": trace}


def _records(trace: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(trace, list):
        return [r for r in trace if isinstance(r, dict)]
    raw = trace.get("records") or trace.get("native_scoring_records") or []
    return [r for r in raw if isinstance(r, dict)]


def _last_record(trace: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    records = _records(trace)
    return records[-1] if records else {}


def _metric(trace: dict[str, Any] | list[dict[str, Any]], key: str) -> float:
    t = _trace_dict(trace)
    last = _last_record(trace)
    if key in last:
        return _as_float(last.get(key))
    if key in t:
        return _as_float(t.get(key))
    totals = t.get("totals")
    if isinstance(totals, dict) and key in totals:
        return _as_float(totals.get(key))
    return 0.0


def _per_corridor_delay(
    trace: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, float]:
    t = _trace_dict(trace)
    for candidate in (
        t.get("per_corridor_delay_minutes"),
        _last_record(trace).get("per_corridor_delay_minutes"),
    ):
        if isinstance(candidate, dict):
            return {str(k): _as_float(v) for k, v in candidate.items()}

    for event in _last_record(trace).get("realized_events") or []:
        if not isinstance(event, dict):
            continue
        per_corridor = event.get("per_corridor")
        if isinstance(per_corridor, dict):
            return {
                str(k): _as_float(v.get("cumulative_delay_minutes"))
                for k, v in per_corridor.items()
                if isinstance(v, dict)
            }
    return {}


def _per_corridor_delta_minutes(
    mock_trace: dict[str, Any] | list[dict[str, Any]],
    live_trace: dict[str, Any] | list[dict[str, Any]],
) -> dict[str, float]:
    mock = _per_corridor_delay(mock_trace)
    live = _per_corridor_delay(live_trace)
    return {
        k: _round3(live.get(k, 0.0) - mock.get(k, 0.0))
        for k in sorted(set(mock) | set(live))
    }


def _signal_program_mismatch_count(
    mock_trace: dict[str, Any] | list[dict[str, Any]],
    live_trace: dict[str, Any] | list[dict[str, Any]],
) -> int:
    mock = _trace_dict(mock_trace).get("signal_program_by_corridor")
    live = _trace_dict(live_trace).get("signal_program_by_corridor")
    if not isinstance(mock, dict) or not isinstance(live, dict):
        return 0
    return sum(1 for k in sorted(set(mock) | set(live)) if mock.get(k) != live.get(k))


def _control_readback_match_rate(
    live_trace: dict[str, Any] | list[dict[str, Any]],
) -> float | None:
    readbacks = _trace_dict(live_trace).get("control_readbacks")
    if not isinstance(readbacks, list) or not readbacks:
        return None
    matches = 0
    total = 0
    for row in readbacks:
        if not isinstance(row, dict):
            continue
        total += 1
        if (
            row.get("sumo_program_readback_matches") is True
            or row.get("live_readback_matches_resolved") is True
        ):
            matches += 1
    return _round3(matches / total) if total else None


def _dict_l1_delta(mock: Any, live: Any) -> float:
    if not isinstance(mock, dict) or not isinstance(live, dict):
        return 0.0
    return sum(
        abs(_as_float(live.get(k)) - _as_float(mock.get(k)))
        for k in sorted(set(mock) | set(live))
    )


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _round3(value: float) -> float:
    return round(float(value), 3)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mock-trace", type=Path)
    parser.add_argument("--live-trace", type=Path)
    parser.add_argument("--fidelity-report", type=Path)
    parser.add_argument("--rebaseline-report", type=Path)
    parser.add_argument("--live-attribution-report", type=Path)
    parser.add_argument("--target-decision-report", type=Path)
    parser.add_argument("--live-runner-report", type=Path)
    parser.add_argument("--replay-seed", action="store_true")
    parser.add_argument("--rebaseline-probe", action="store_true")
    parser.add_argument("--live-attribution-probe", action="store_true")
    parser.add_argument("--target-decision", action="store_true")
    parser.add_argument("--live-case-ledger-preview", action="store_true")
    parser.add_argument(
        "--live-replacement-plan",
        "--live-replacement-materialization-plan",
        dest="live_replacement_plan",
        action="store_true",
    )
    parser.add_argument(
        "--live-replacement-pilot-dry-run",
        "--live-replacement-pilot-release-dry-run",
        dest="live_replacement_pilot_dry_run",
        action="store_true",
    )
    parser.add_argument("--corridor-scoped-promotion-preview", action="store_true")
    parser.add_argument("--corridor-scoped-live-runner", action="store_true")
    parser.add_argument("--live-case-ledger-preview-report", type=Path)
    parser.add_argument("--live-replacement-plan-report", type=Path)
    parser.add_argument("--wait-trace", type=Path)
    parser.add_argument("--oracle-trace", type=Path)
    parser.add_argument("--counterfactual-trace", type=Path)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--family", default="incident_response")
    parser.add_argument("--difficulty-level", default="basic")
    parser.add_argument("--difficulty-mode", default="time_pressure")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ticks", type=int, default=6)
    parser.add_argument("--action-stream", type=Path)
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    if args.corridor_scoped_live_runner:
        actions = _load_json(args.action_stream) if args.action_stream else None
        report = build_traffic_corridor_scoped_live_runner_report(
            target_decision_report_path=(
                args.target_decision_report or args.fidelity_report
            ),
            family=args.family,
            difficulty_level=args.difficulty_level,
            difficulty_mode=args.difficulty_mode,
            seed=args.seed,
            n_ticks=args.n_ticks,
            candidate_action_stream=actions,
            release_dir=args.release_dir,
        )
    elif args.corridor_scoped_promotion_preview:
        report = build_traffic_corridor_scoped_promotion_preview_report(
            target_decision_report_path=(
                args.target_decision_report or args.fidelity_report
            ),
            live_trace_path=args.live_trace,
            wait_trace_path=args.wait_trace,
            oracle_trace_path=args.oracle_trace,
            counterfactual_trace_path=args.counterfactual_trace,
            release_dir=args.release_dir,
        )
    elif args.live_case_ledger_preview:
        report = build_traffic_live_case_ledger_preview_report(
            target_decision_report_path=(
                args.target_decision_report or args.fidelity_report
            ),
            live_runner_report_path=args.live_runner_report,
            release_dir=args.release_dir,
        )
    elif args.live_replacement_plan:
        report = build_traffic_live_replacement_materialization_plan_report(
            live_runner_report_path=args.live_runner_report,
            live_case_ledger_preview_report_path=args.live_case_ledger_preview_report,
            release_dir=args.release_dir,
        )
    elif args.live_replacement_pilot_dry_run:
        report = build_traffic_live_replacement_pilot_release_dry_run_report(
            replacement_plan_report_path=args.live_replacement_plan_report,
            release_dir=args.release_dir,
        )
    elif args.target_decision:
        report = build_traffic_fidelity_target_decision_report(
            fidelity_report_path=args.fidelity_report,
            rebaseline_report_path=args.rebaseline_report,
            live_attribution_report_path=args.live_attribution_report,
        )
    elif args.live_attribution_probe:
        report = build_traffic_live_attribution_probe_report(
            fidelity_report_path=args.fidelity_report,
            live_trace_path=args.live_trace,
        )
    elif args.rebaseline_probe:
        report = build_traffic_rebaseline_probe_report(
            fidelity_report_path=args.fidelity_report,
            mock_trace_path=args.mock_trace,
            live_trace_path=args.live_trace,
        )
    elif args.replay_seed:
        actions = _load_json(args.action_stream) if args.action_stream else None
        report = build_traffic_mock_live_replay_report(
            family=args.family,
            difficulty_level=args.difficulty_level,
            difficulty_mode=args.difficulty_mode,
            seed=args.seed,
            n_ticks=args.n_ticks,
            action_stream=actions,
            run_live=args.run_live,
        )
    else:
        report = build_traffic_mock_live_fidelity_report(
            mock_trace_path=args.mock_trace,
            live_trace_path=args.live_trace,
        )
    write_report(report, args.output)
    if args.require_complete and report.get("status") != "fidelity_probe_complete":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
