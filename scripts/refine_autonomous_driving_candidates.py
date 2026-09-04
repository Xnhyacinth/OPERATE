#!/usr/bin/env python3
"""Select NGSIM candidate survivors with only essential exploratory gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = (
    REPO_ROOT
    / "works/autonomous_driving/ngsim/derived/ngsim_multisite_core_slice_v13/catalog_full_v1.json"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _index_reports(path: Path, *, candidate_field: str) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for report_path in sorted(path.glob("*.json")):
        report = _load(report_path)
        candidate_id = str(report.get(candidate_field) or "")
        if not candidate_id and candidate_field == "candidate_id":
            candidate_id = str(
                ((report.get("scenario") or {}).get("backend_config") or {}).get("candidate_id")
                or ""
            )
        if candidate_id:
            reports[candidate_id] = report
    return reports


def _cost(leg: dict[str, Any]) -> float:
    return round(sum(float(value) for value in (leg.get("cost_components") or {}).values()), 6)


def _valid_post_change_chain(leg: dict[str, Any]) -> bool:
    for row in leg.get("post_change_runtime_evidence") or []:
        ticks = [row.get(name) for name in ("material_event_tick", "decision_tick", "control_tick", "native_effect_tick")]
        if not all(isinstance(value, int) for value in ticks):
            continue
        material, decision, control, effect = (int(value) for value in ticks)
        if (
            material < decision <= control < effect
            and (row.get("decision") or {}).get("origin") == "agent_policy"
            and (row.get("control") or {}).get("applied_by_native_backend") is True
            and (row.get("native_effect") or {}).get("observed_from_backend_step") is True
        ):
            return True
    return False


def _candidate_row(
    catalog_row: dict[str, Any],
    calibration: dict[str, Any] | None,
    replay: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_id = str(catalog_row.get("candidate_id") or "")
    blockers: list[str] = []
    if calibration is None:
        return {"candidate_id": candidate_id, "disposition": "held", "blockers": ["calibration_missing"]}
    scenario = dict(calibration.get("scenario") or {})
    backend = dict(scenario.get("backend_config") or {})
    if (
        str(backend.get("candidate_id") or "") != candidate_id
        or len(str(catalog_row.get("source_event_chain_sha256") or "")) != 64
    ):
        blockers.append("source_identity_mismatch")
    if backend.get("execution_mode") != "live":
        blockers.append("native_live_sumo_missing")
    if scenario.get("difficulty_level") not in {"high", "extreme"}:
        blockers.append("high_extreme_task_missing")
    legs = {
        str(row.get("leg") or ""): row
        for row in calibration.get("legs") or []
        if isinstance(row, dict)
    }
    if set(legs) != {"shield_only", "rule_tactical", "oracle_offline"}:
        blockers.append("three_leg_attribution_missing")
    source_native = bool(legs) and all(
        row.get("status") == "completed"
        and (row.get("source_consumption") or {}).get("status") == "verified"
        and (row.get("source_consumption") or {}).get("runtime_trace_observed") is True
        and (row.get("source_consumption") or {}).get("deterministic_source_trace") is True
        and bool(row.get("source_events"))
        for row in legs.values()
    )
    if not source_native:
        blockers.append("source_native_effect_missing")
    safe = bool(legs) and all(
        int(row.get("collision_count") or 0) == 0
        and int(row.get("road_departure_count") or 0) == 0
        for row in legs.values()
    )
    if not safe:
        blockers.append("safety_gate_failed")
    horizon = int(scenario.get("horizon_ticks") or 0)
    if not legs or horizon <= 0 or any(int(row.get("records") or 0) != horizon for row in legs.values()):
        blockers.append("task_execution_incomplete")
    agent_rows = [legs[name] for name in ("rule_tactical", "oracle_offline") if name in legs]
    if not any(_valid_post_change_chain(row) for row in agent_rows):
        blockers.append("post_event_control_chain_missing")
    shield_cost = _cost(legs["shield_only"]) if "shield_only" in legs else 0.0
    agent_costs = {str(row["leg"]): _cost(row) for row in agent_rows}
    best_agent = min(agent_costs, key=agent_costs.get) if agent_costs else ""
    headroom = round(shield_cost - agent_costs.get(best_agent, shield_cost), 6)
    comparison = dict(((calibration.get("attribution") or {}).get("comparisons") or {}).get(best_agent) or {})
    if (
        headroom <= 0.0
        or round(float(comparison.get("agent_incremental_value_vs_shield_only") or 0.0), 6)
        != headroom
        or comparison.get("safety_regression_vs_shield_only") is not False
    ):
        blockers.append("positive_agent_headroom_missing")
    if (
        replay is None
        or replay.get("status") != "verified"
        or replay.get("deterministic_semantic_replay") is not True
        or str(replay.get("candidate_id") or "") != candidate_id
    ):
        blockers.append("deterministic_replay_missing")
    blockers = sorted(set(blockers))
    return {
        "candidate_id": candidate_id,
        "recording_id": catalog_row.get("recording_id"),
        "hazard_kind": catalog_row.get("hazard_kind"),
        "difficulty_level": scenario.get("difficulty_level"),
        "shield_only_cost": shield_cost,
        "best_agent_leg": best_agent,
        "best_agent_cost": agent_costs.get(best_agent),
        "agent_headroom_vs_shield_only": headroom,
        "disposition": "candidate_survivor" if not blockers else "held",
        "blockers": blockers,
    }


def _survivor_scenarios(
    *, scenario_root: Path | None, survivor_ids: set[str]
) -> list[dict[str, Any]]:
    if scenario_root is None:
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(scenario_root.rglob("*.yaml")):
        body = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(body, dict):
            continue
        candidate_id = str((body.get("backend_config") or {}).get("candidate_id") or "")
        if candidate_id not in survivor_ids or body.get("difficulty_level") not in {
            "high",
            "extreme",
        }:
            continue
        try:
            relative = path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            relative = str(path.resolve())
        rows.append(
            {
                "scenario_id": str(body.get("scenario_id") or body.get("seed_id") or ""),
                "path": relative,
                "domain": "autonomous_driving",
                "backend_kind": "sumo_ego",
                "family": body.get("family"),
                "difficulty_mode": body.get("difficulty_mode"),
                "difficulty_level": body.get("difficulty_level"),
                "candidate_id": candidate_id,
            }
        )
    return rows


def build_refine_ledger(
    *,
    catalog: dict[str, Any],
    calibration_dir: Path,
    replay_dir: Path,
    scenario_root: Path | None = None,
) -> dict[str, Any]:
    calibrations = _index_reports(calibration_dir, candidate_field="candidate_id")
    replays = _index_reports(replay_dir, candidate_field="candidate_id")
    rows = [
        _candidate_row(row, calibrations.get(str(row.get("candidate_id") or "")), replays.get(str(row.get("candidate_id") or "")))
        for row in catalog.get("bundles") or []
        if isinstance(row, dict)
    ]
    survivors = [row for row in rows if row["disposition"] == "candidate_survivor"]
    held = [row for row in rows if row["disposition"] == "held"]
    scenarios = _survivor_scenarios(
        scenario_root=scenario_root,
        survivor_ids={str(row["candidate_id"]) for row in survivors},
    )
    return {
        "schema_version": "autonomous_driving_essential_refine_ledger_v1",
        "status": "candidate_survivors_found" if survivors else "held",
        "formal_core_allowed": False,
        "evidence_scope": "mutable_worktree_candidate_evidence",
        "essential_gates": [
            "source_identity",
            "deterministic_native_replay",
            "native_source_effect",
            "positive_agent_headroom",
            "zero_collision_and_road_departure",
            "high_or_extreme_task",
            "post_event_control_chain",
        ],
        "summary": {
            "candidate_count": len(rows),
            "survivor_count": len(survivors),
            "held_count": len(held),
        },
        "survivors": survivors,
        "held": held,
        "scenarios": scenarios,
        "stable_tree_rerun": (
            "python scripts/run_autonomous_driving_core_calibration_batch.py "
            "--catalog <catalog.json> --output-root <empty-output-dir> "
            "--difficulty-level extreme --ticks 14 --repeats 2"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--scenario-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_refine_ledger(
        catalog=_load(args.catalog),
        calibration_dir=args.calibration_dir,
        replay_dir=args.replay_dir,
        scenario_root=args.scenario_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if report["survivors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
