#!/usr/bin/env python3
"""Finalize bounded native Traffic mining into a terminal candidate ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _full_network_result(trial: dict[str, Any]) -> tuple[float, float, bool]:
    baseline = float(
        (trial.get("baseline_metrics") or {}).get("network_vehicle_time_auc_s")
        or 0.0
    )
    reference = float(
        (trial.get("reference_metrics") or {}).get("network_vehicle_time_auc_s")
        or 0.0
    )
    improvement = baseline - reference
    required = max(1.0, 0.05 * baseline) if baseline > 0.0 else 1.0
    passed = bool(
        trial.get("status") == "passed"
        and baseline > 0.0
        and improvement >= required
        and trial.get("baseline_repeat_deterministic") is True
        and trial.get("reference_repeat_deterministic") is True
        and (trial.get("source_consumption") or {}).get("status") == "passed"
        and (trial.get("native_control_effect") or {}).get(
            "native_control_effect_observed"
        )
        is True
        and (trial.get("world_evolution") or {}).get(
            "post_change_decision_observed"
        )
        is True
        and (trial.get("safety") or {}).get("status") == "passed"
    )
    return improvement, required, passed


def build_terminal_ledger(
    *,
    execution: dict[str, Any],
    crosscheck: dict[str, Any],
    trials: list[dict[str, Any]],
    requested_dates: list[str],
    replay_command: list[str],
) -> dict[str, Any]:
    identity_by_date = {
        str(row["service_date"]): row for row in crosscheck.get("results") or []
    }
    rows: list[dict[str, Any]] = []
    for service_date in sorted(set(requested_dates)):
        date_trials = [
            row for row in trials if row.get("service_date") == service_date
        ]
        scored = [(*_full_network_result(row), row) for row in date_trials]
        best = max(scored, key=lambda item: item[0], default=(0.0, 1.0, False, {}))
        native_survivor = any(item[2] for item in scored)
        identity = identity_by_date.get(service_date) or {}
        reasons = sorted(
            {
                str(reason)
                for trial in date_trials
                for reason in (
                    trial.get("reason_codes")
                    or [trial.get("reason_code")]
                )
                if reason
            }
        )
        safe_trial_exists = any(
            (trial.get("safety") or {}).get("status") == "passed"
            for trial in date_trials
        )
        rows.append(
            {
                "service_date": service_date,
                "work_state": "terminal",
                "disposition": (
                    "candidate_pending_tool_protocol_materialization"
                    if native_survivor
                    else "held_native_headroom"
                    if safe_trial_exists
                    else "held_safety"
                ),
                "reason_codes": (
                    ["native_survivor_requires_tool_protocol_materialization"]
                    if native_survivor
                    else sorted({*reasons, "no_full_network_positive_survivor"})
                ),
                "n_trials": len(date_trials),
                "best_full_network_improvement": best[0],
                "required_full_network_improvement": best[1],
                "gates": {
                    "source_identity_crosscheck": bool(
                        identity.get("identity_equality") is True
                        and identity.get("payload_equality") is True
                    ),
                    "source_consumption": any(
                        (trial.get("source_consumption") or {}).get("status")
                        == "passed"
                        for trial in date_trials
                    ),
                    "determinism": any(
                        trial.get("baseline_repeat_deterministic") is True
                        and trial.get("reference_repeat_deterministic") is True
                        for trial in date_trials
                    ),
                    "native_control_effect": any(
                        (trial.get("native_control_effect") or {}).get(
                            "native_control_effect_observed"
                        )
                        is True
                        for trial in date_trials
                    ),
                    "full_network_positive_5pct": native_survivor,
                    "safety_no_regression": safe_trial_exists,
                    "post_change_response": any(
                        (trial.get("world_evolution") or {}).get(
                            "post_change_decision_observed"
                        )
                        is True
                        for trial in date_trials
                    ),
                    "tool_protocol_native_action": False,
                },
            }
        )
    n_survivors = sum(
        row["disposition"] == "candidate_pending_tool_protocol_materialization"
        for row in rows
    )
    execution_complete = bool(
        execution.get("status") == "complete"
        and execution.get("planned_primary_trials")
        == execution.get("completed_primary_trials")
        and execution.get("orphan_process_count") == 0
    )
    return {
        "schema_version": "traffic-native-essential-gate-ledger-v1",
        "status": (
            "native_survivors_pending_materialization"
            if n_survivors
            else "complete_no_survivors"
        ),
        "candidate_only": True,
        "core_admission_claimed": False,
        "execution_complete": execution_complete,
        "n_expected": len(rows),
        "n_terminal": len(rows),
        "n_native_survivors": n_survivors,
        "rows": rows,
        "next_stage": (
            "materialize_survivor_through_ToolProtocol_then_quality_core_v2"
            if n_survivors
            else None
        ),
        "replay_command": replay_command,
    }


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mining-dir", type=Path, required=True)
    parser.add_argument("--service-dates", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mining_dir = args.mining_dir.resolve()
    paths = {
        "execution_manifest": mining_dir / "execution_manifest.json",
        "source_identity_crosscheck": mining_dir / "source_identity_crosscheck.json",
        "headroom_trials": mining_dir / "headroom_trials.json",
    }
    ledger = build_terminal_ledger(
        execution=_load(paths["execution_manifest"]),
        crosscheck=_load(paths["source_identity_crosscheck"]),
        trials=_load(paths["headroom_trials"])["results"],
        requested_dates=args.service_dates,
        replay_command=[
            ".venv/bin/python",
            "scripts/mine_sumo365_native_traffic.py",
            "--output-dir",
            "<fresh-output-dir>",
            "--primary-dates",
            *sorted(set(args.service_dates)),
            "--decision-interval-seconds",
            "30",
            "--warmup-seconds",
            "300",
            "--horizon-seconds",
            "600",
            "--max-tls-per-source",
            "2",
            "--max-duration-actions-per-tls",
            "3",
            "--trial-budget-per-source",
            "6",
            "--minimum-positive-sources",
            "1",
            "--tail-seconds",
            "120",
            "--independent-asset-graph",
            "reports/traffic_sumo365_refine_20260814/source_identity_graph.json",
            "--repeats",
            "2",
            "--workers",
            "2",
        ],
    )
    ledger["input_bindings"] = {
        name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}
        for name, path in paths.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(ledger, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
