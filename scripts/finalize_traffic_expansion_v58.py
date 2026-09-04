#!/usr/bin/env python3
"""Freeze bounded Traffic expansion evidence without promoting weak rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_NETWORK_GAIN_FLOOR = 0.05


def _denominator(row: dict[str, Any]) -> str:
    return str(
        row.get("source_denominator_key")
        or (row.get("case_ledger") or {}).get("source_denominator_key")
        or ""
    )


def _trial_result(trial: dict[str, Any]) -> dict[str, Any]:
    baseline = float(
        (trial.get("baseline_metrics") or {}).get("network_vehicle_time_auc_s")
        or 0.0
    )
    reference = float(
        (trial.get("reference_metrics") or {}).get("network_vehicle_time_auc_s")
        or 0.0
    )
    improvement = baseline - reference
    required = max(1.0, FULL_NETWORK_GAIN_FLOOR * baseline) if baseline else 1.0
    gates = {
        "full_network_positive_5pct": baseline > 0 and improvement >= required,
        "deterministic_replay_twice": bool(
            trial.get("baseline_repeat_deterministic") is True
            and trial.get("reference_repeat_deterministic") is True
        ),
        "source_consumption": (trial.get("source_consumption") or {}).get("status")
        == "passed",
        "native_control_effect": (trial.get("native_control_effect") or {}).get(
            "native_control_effect_observed"
        )
        is True,
        "safety_no_regression": (trial.get("safety") or {}).get("status")
        == "passed",
        "post_change_decision_control_effect": bool(
            (trial.get("world_evolution") or {}).get(
                "post_change_decision_observed"
            )
            and (trial.get("causal_chain") or {}).get("action_after_source_change")
        ),
        "task_complete": (trial.get("task_completion") or {}).get("status")
        == "passed",
    }
    return {
        "trial_id": trial.get("trial_id"),
        "improvement_vehicle_seconds": improvement,
        "required_improvement_vehicle_seconds": required,
        "gates": gates,
        "essential_survivor": all(gates.values()),
    }


def build_terminal_ledger(
    *,
    selected_rows: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    resco_rows: list[dict[str, Any]],
    replay_command: list[str],
) -> dict[str, Any]:
    exclusions = sorted(
        (
            {
                "scenario_id": str(row.get("scenario_id") or ""),
                "source_denominator_key": _denominator(row),
            }
            for row in selected_rows
            if row.get("domain") == "traffic"
        ),
        key=lambda row: (row["source_denominator_key"], row["scenario_id"]),
    )
    trial_results = [(_trial_result(row), row) for row in trials]
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        service_date = str(source.get("service_date") or "")
        matching = [item for item in trial_results if item[1].get("service_date") == service_date]
        best = max(
            matching,
            key=lambda item: item[0]["improvement_vehicle_seconds"],
            default=(
                {
                    "improvement_vehicle_seconds": 0.0,
                    "required_improvement_vehicle_seconds": 1.0,
                    "gates": {},
                    "essential_survivor": False,
                    "trial_id": None,
                },
                {},
            ),
        )[0]
        survivor = next((item[0] for item in matching if item[0]["essential_survivor"]), None)
        rows.append(
            {
                "source_family": "sumo_ingolstadt_365",
                "source_unit": service_date,
                "work_state": "terminal",
                "disposition": "essential_survivor" if survivor else "held_native_headroom",
                "best_trial": survivor or best,
                "reason_codes": []
                if survivor
                else ["no_safe_full_network_5pct_task_complete_survivor"],
            }
        )
    rows.extend(resco_rows)
    survivor_rows = [row for row in rows if row["disposition"] == "essential_survivor"]
    return {
        "schema_version": "traffic-expansion-terminal-ledger-v58.1",
        "status": "essential_survivors_available" if survivor_rows else "complete_no_survivors",
        "candidate_only": True,
        "core_admission_claimed": False,
        "full_protocol21_replay_executed": False,
        "essential_gate": {
            "full_network_native_loss_improvement_fraction": FULL_NETWORK_GAIN_FLOOR,
            "deterministic_replays_per_policy": 2,
            "safety_regression_allowed": False,
            "task_completion_required": True,
            "post_change_decision_control_effect_required_for_high_extreme": True,
        },
        "exclusion_set": exclusions,
        "n_excluded_v57_traffic_rows": len(exclusions),
        "n_terminal": len(rows),
        "n_essential_survivors": len(survivor_rows),
        "rows": rows,
        "survivor_scenario_yamls": [],
        "source_suite": None,
        "replay_command": replay_command,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resco_terminal_rows() -> list[dict[str, Any]]:
    return [
        {
            "source_family": "resco",
            "source_unit": "ingolstadt7",
            "work_state": "terminal",
            "disposition": "held_native_control_surface",
            "reason_codes": ["no_legal_variable_phase_or_alternate_program"],
            "native_replay_executed": False,
        },
        {
            "source_family": "resco",
            "source_unit": "arterial4x4",
            "work_state": "terminal",
            "disposition": "held_source_runtime_graph",
            "reason_codes": ["route_member_archived_not_direct_sumocfg_input", "prior_native_policy_negative"],
            "native_replay_executed": False,
        },
        {
            "source_family": "resco",
            "source_unit": "grid4x4",
            "work_state": "terminal",
            "disposition": "held_source_runtime_graph",
            "reason_codes": ["route_member_archived_not_direct_sumocfg_input", "prior_native_policy_negative"],
            "native_replay_executed": False,
        },
        {
            "source_family": "resco",
            "source_unit": "saltlake2_400sX200w",
            "work_state": "terminal",
            "disposition": "held_source_runtime_demand",
            "reason_codes": ["sumocfg_route_definitions_have_no_consumed_vehicle_demand"],
            "native_replay_executed": False,
        },
        {
            "source_family": "resco",
            "source_unit": "saltlake2_stateXuniversity",
            "work_state": "terminal",
            "disposition": "held_source_runtime_demand",
            "reason_codes": ["sumocfg_route_definitions_have_no_consumed_vehicle_demand"],
            "native_replay_executed": False,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--mining-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite immutable ledger: {args.output}")
    selection = _load(args.selection)
    mining = args.mining_dir
    input_paths = {
        "v57_selection": args.selection,
        "execution_manifest": mining / "execution_manifest.json",
        "source_identities": mining / "source_identities.json",
        "headroom_trials": mining / "headroom_trials.json",
        "source_identity_crosscheck": mining / "source_identity_crosscheck.json",
    }
    dates = sorted(
        str(row.get("service_date"))
        for row in _load(input_paths["source_identities"])["results"]
    )
    ledger = build_terminal_ledger(
        selected_rows=selection["scenarios"],
        trials=_load(input_paths["headroom_trials"])["results"],
        source_rows=_load(input_paths["source_identities"])["results"],
        resco_rows=_resco_terminal_rows(),
        replay_command=[
            ".venv/bin/python",
            "scripts/mine_sumo365_native_traffic.py",
            "--output-dir",
            "<fresh-output-dir>",
            "--primary-dates",
            *dates,
            "--event-selection",
            "latest_material",
            "--decision-interval-seconds",
            "30",
            "--warmup-seconds",
            "300",
            "--horizon-seconds",
            "900",
            "--max-tls-per-source",
            "3",
            "--max-duration-actions-per-tls",
            "4",
            "--trial-budget-per-source",
            "8",
            "--tail-seconds",
            "120",
            "--tail-mode",
            "until_clear_or_max",
            "--max-tail-seconds",
            "1200",
            "--independent-asset-graph",
            "reports/traffic_sumo365_refine_20260814/source_identity_graph.json",
            "--repeats",
            "2",
            "--workers",
            "2",
        ],
    )
    ledger["input_bindings"] = {
        key: {
            "path": str(path.resolve().relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
        }
        for key, path in input_paths.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": ledger["status"], "n_essential_survivors": ledger["n_essential_survivors"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
