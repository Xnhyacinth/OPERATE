#!/usr/bin/env python3
"""Run a bounded, non-admission Datacenter policy-review preflight.

The batch is deliberately smaller than Protocol-2.1: it binds the declared
trace/window, runs native wait and oracle trajectories, checks visible event
and policy-review evidence, compares costs, and verifies deterministic wait
replay.  A candidate result is only a shortlist for a fresh Protocol-2.1 run.
No YAML, release manifest, Core suite or queue file is modified.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.oracle_offline import OracleOfflineAgent  # noqa: E402
from core import Action, ToolCall  # noqa: E402
from domains.datacenter.adapter import DatacenterEnvironment  # noqa: E402

REPORT_SCHEMA = "protocol21-datacenter-preflight-v1"
DEFAULT_QUEUE = ROOT / ".hl/source_conversion_queue_v56.json"
DEFAULT_OUTPUT = ROOT / "reports/protocol21_datacenter_preflight.json"


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _source_identity(row: dict[str, Any], scenario: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    provenance = dict(scenario.get("provenance") or {})
    declared_files = [str(item) for item in provenance.get("files") or []]
    files = []
    for declared in declared_files:
        path = _resolve_path(declared, repo_root)
        files.append({"declared_path": declared, "exists": path.is_file(), "sha256": sha256_file(path)})
    metadata_ok = all(provenance.get(key) for key in ("data_source", "url", "license", "commit", "lock_strategy"))
    file_ok = bool(files) and all(item["exists"] for item in files)
    expected = str(row.get("scenario_sha256") or "")
    scenario_path = _resolve_path(str(row.get("source_path") or ""), repo_root)
    scenario_hash = sha256_file(scenario_path)
    scenario_ok = bool(scenario_hash) and (not expected or scenario_hash == expected)
    status = "passed" if metadata_ok and file_ok and scenario_ok else "held"
    reason = None
    if not metadata_ok:
        reason = "source_lock_metadata_incomplete"
    elif not file_ok:
        reason = "source_asset_missing"
    elif not scenario_ok:
        reason = "scenario_yaml_hash_mismatch"
    return {
        "status": status,
        "reason_code": reason,
        "scenario_sha256": scenario_hash,
        "expected_scenario_sha256": expected or None,
        "files": files,
        "commit": provenance.get("commit"),
        "license": provenance.get("license"),
    }


def _wait_action(tick: int) -> Action:
    return Action(tool_calls=[ToolCall(name="wait", call_id=f"dc-preflight-wait-{tick}")], dominant="wait")


def _rollout(scenario: dict[str, Any], *, oracle: bool) -> dict[str, Any]:
    env = DatacenterEnvironment()
    observations: list[str] = []
    events: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    try:
        observation = env.reset(scenario, seed=int(scenario.get("seed") or 0))
        agent = OracleOfflineAgent()
        if oracle:
            agent.reset(env, scenario, seed=int(scenario.get("seed") or 0))
        specs = env.get_tool_specs()
        for tick in range(max(1, int(scenario.get("horizon_ticks") or 1))):
            observations.append(stable_digest(observation))
            action = agent.act(observation, specs) if oracle else _wait_action(tick)
            result = env.step(action)
            events.extend(list(result.info.realized_events or []))
            events.extend(list(result.info.extra.get("world_evolution_records") or []))
            for tool in result.tool_results:
                if tool.ok and tool.state_changing:
                    effects.append({"name": tool.name, "tick": tick, "effect_tick": tool.effect_tick, "payload": dict(tool.payload)})
            observation = result.observation
            if result.done:
                break
        truth = env.ground_truth()
        reviews = list((truth.get("control_summary") or {}).get("policy_review_ledger") or [])
        costs = {str(k): float(v) for k, v in dict(truth.get("cost_components") or {}).items() if isinstance(v, (int, float))}
        return {
            "status": "passed",
            "n_ticks": len(observations),
            "trajectory_digest": stable_digest({"observations": observations, "events": events, "effects": effects, "truth": truth}),
            "final_state_digest": stable_digest(truth),
            "events": events,
            "effects": effects,
            "reviews": reviews,
            "cost_components": costs,
            "scalar_cost": round(sum(costs.values()), 6),
        }
    except Exception as exc:  # keep one bad candidate from aborting the batch
        return {"status": "held", "reason_code": "runtime_preflight_error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        env.close()


def classify_datacenter_preflight(row: dict[str, Any]) -> dict[str, Any]:
    """Classify a preflight row; this function never admits Core."""
    checks = (("source_identity", "held_source_lock"), ("runtime", "held_runtime"), ("behavior", "held_behavior"), ("review", "held_review"), ("effect", "held_effect"), ("headroom", "held_headroom"), ("determinism", "held_determinism"))
    disposition = "preflight_candidate"
    for name, held in checks:
        if (row.get(name) or {}).get("status") != "passed":
            disposition = held
            break
    return {**row, "disposition": disposition, "core_admission": False, "full_protocol21_required": True}


def _target_rows(queue_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(queue_path.read_text(encoding="utf-8"))
    lane = next(item for item in payload.get("priority_lanes", []) if item.get("lane_id") == "P0_datacenter_and_opendss_agentic_repair")
    return [dict(row) for row in lane.get("candidate_rows") or [] if row.get("backend") == "alibaba_trace_sim"]


def build_report(*, queue_path: Path = DEFAULT_QUEUE, repo_root: Path = ROOT) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for target in _target_rows(queue_path):
        scenario_path = _resolve_path(str(target["source_path"]), repo_root)
        row: dict[str, Any] = {"scenario_id": target.get("scenario_id"), "path": str(target["source_path"]), "source_identity": {"status": "held", "reason_code": "scenario_yaml_missing"}}
        if not scenario_path.is_file():
            rows.append(classify_datacenter_preflight(row))
            continue
        scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
        row["scenario_signature"] = scenario.get("scenario_signature")
        row["source_identity"] = _source_identity(target, scenario, repo_root)
        wait_one = _rollout(scenario, oracle=False)
        wait_two = _rollout(scenario, oracle=False)
        oracle = _rollout(scenario, oracle=True)
        material_events = [event for event in oracle.get("events", []) if event.get("material_exogenous") is True or (event.get("materiality") or {}).get("passed") is True]
        reviews = [review for review in oracle.get("reviews", []) if review.get("outcome_effect_ticks")]
        row["runtime"] = {"status": oracle.get("status"), "n_ticks": oracle.get("n_ticks"), "error": oracle.get("error")}
        row["behavior"] = {"status": "passed" if material_events else "held", "material_event_count": len(material_events), "event_types": sorted({str(event.get("event_type") or event.get("type") or "") for event in material_events})}
        row["review"] = {"status": "passed" if oracle.get("reviews") else "held", "review_count": len(oracle.get("reviews") or []), "review_ticks": [review.get("review_tick") for review in oracle.get("reviews") or []]}
        row["effect"] = {"status": "passed" if reviews else "held", "reason_code": None if reviews else "policy_effect_missing", "outcome_effect_ticks": [tick for review in reviews for tick in review.get("outcome_effect_ticks") or []], "native_effect_count": len(oracle.get("effects") or [])}
        wait_cost = wait_one.get("scalar_cost")
        oracle_cost = oracle.get("scalar_cost")
        headroom = (float(wait_cost) - float(oracle_cost)) if wait_cost is not None and oracle_cost is not None else None
        row["headroom"] = {"status": "passed" if headroom is not None and headroom > 0 else "held", "raw_headroom": round(headroom, 6) if headroom is not None else None, "wait_cost": wait_cost, "oracle_cost": oracle_cost}
        deterministic = wait_one.get("status") == "passed" and wait_one.get("trajectory_digest") == wait_two.get("trajectory_digest")
        row["determinism"] = {"status": "passed" if deterministic else "held", "first": wait_one.get("trajectory_digest"), "second": wait_two.get("trajectory_digest")}
        row["probe"] = {"oracle_trajectory_digest": oracle.get("trajectory_digest"), "oracle_cost_components": oracle.get("cost_components")}
        rows.append(classify_datacenter_preflight(row))
    dispositions = Counter(str(row["disposition"]) for row in rows)
    return {"schema_version": REPORT_SCHEMA, "status": "preflight_complete", "release_ready": False, "core_admission": False, "target": {"lane": "P0_datacenter_and_opendss_agentic_repair", "backend": "alibaba_trace_sim", "n_expected": 6, "queue_sha256": sha256_file(queue_path)}, "n_rows": len(rows), "disposition_counts": dict(sorted(dispositions.items())), "rows": rows, "non_goals": ["does_not_modify_yaml_release_core_or_queue", "does_not_replace_full_protocol21", "does_not_rebind_old_evidence"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(queue_path=args.queue.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "n_rows": report["n_rows"], "disposition_counts": report["disposition_counts"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
