"""Recompute existing native voltage-recovery stages from bound tick records.

This measures the scenario's declared multi-stage fulfillment, not unaided
memory retention, plan prose, tool choice, or a new long-horizon construct.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evaluation.native_reference import load_reference_report, read_verified

VERSION = "native_multi_stage_fulfillment.v1"
CONTRACTS = {"microgrid.lv_voltage.staged_recovery.v2",
             "microgrid.lv_voltage.cross_tick_recovery.v2"}


def _hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def compile_legacy_persistence_contract(
    scenario: dict[str, Any], *, scenario_sha256: str,
) -> dict[str, Any]:
    """Compile only already-declared source phases; caller verifies YAML hash."""
    result = {"schema_version": VERSION, "applicable": False,
              "interpretation": "native_multi_stage_fulfillment_not_memory_retention"}
    spec = (scenario.get("backend_config") or {}).get("task_contract") or {}
    if (scenario.get("backend_kind") != "pandapower_lv"
            or scenario.get("domain") != "microgrid" or spec.get("contract") not in CONTRACTS):
        return {**result, "reason": "no_existing_native_multi_stage_contract"}
    phases, minimum = spec.get("phase_ticks"), spec.get("minimum_reduction_each_phase")
    horizon = scenario.get("horizon_ticks")
    if (not _digest(scenario_sha256) or not scenario.get("scenario_signature")
            or type(scenario.get("seed")) is not int or type(horizon) is not int
            or not isinstance(phases, list) or len(phases) < 2
            or any(type(t) is not int or not 0 <= t < horizon for t in phases)
            or phases != sorted(set(phases)) or type(minimum) is not int or minimum <= 0):
        return {**result, "reason": "invalid_existing_native_phase_contract"}
    result.update(applicable=True, reason=None, scenario_signature=scenario["scenario_signature"],
                  seed=scenario["seed"], backend_kind="pandapower_lv", domain="microgrid",
                  scenario_sha256=scenario_sha256, source_contract=spec["contract"],
                  horizon_ticks=horizon, phase_ticks=list(phases),
                  minimum_reduction_each_phase=minimum,
                  definition_source="evaluation/task_completion.py: native phase_reductions")
    result["contract_sha256"] = _hash(result)
    return result


def load_verified_wait_baseline(reference_contract: dict | None, reference_inputs: list[dict]) -> dict:
    """Resolve a certified wait-only episode and its original scoring snapshot."""
    def fail(reason: str) -> dict:
        return {"verified": False, "reason": reason}

    if not isinstance(reference_contract, dict):
        return fail("reference_contract_unavailable")
    digest = reference_contract.get("reference_report_sha256")
    candidates = [item for item in reference_inputs if item.get("report_sha256") == digest]
    if len(candidates) != 1:
        return fail("unique_reference_report_not_available")
    source = candidates[0]
    try:
        root = Path(source["root"])
        read_verified(root, source["report"], digest)
        bundle = load_reference_report(root, source["report"])
        contracts = [item for item in bundle["contracts"]
                     if item["contract_sha256"] == reference_contract.get("contract_sha256")]
        if len(contracts) != 1:
            return fail("reference_contract_binding_mismatch")
        contract = contracts[0]
        choices = [item for item in contract["reference_artifacts"]
                   if item["policy"] == "wait_only" and item["repetition"] == 0]
        if len(choices) != 1:
            return fail("wait_reference_episode_missing")
        selected = choices[0]
        raw, _ = read_verified(root, selected["episode_path"], selected["episode_sha256"])
        episode = json.loads(raw)
        if episode.get("agent_name") != "wait_only" or any(
            episode.get(key) != contract[key] for key in ("scenario_signature", "seed")
        ):
            return fail("wait_reference_episode_identity_mismatch")
        descriptor = dict(episode["trajectory_summary"]["scoring_inputs_artifact"])
        snapshot_path = Path(descriptor["path"])
        if not snapshot_path.is_absolute():
            snapshot_path = root / snapshot_path
        if not snapshot_path.resolve().is_relative_to(root.resolve()):
            return fail("wait_reference_snapshot_outside_root")
        descriptor["path"] = str(snapshot_path)
        return {"verified": True, "scenario_signature": contract["scenario_signature"],
                "seed": contract["seed"], "runtime_identity": contract["runtime_identity"],
                "reference_report_sha256": digest, "reference_contract_sha256": contract["contract_sha256"],
                "reference_episode_sha256": selected["episode_sha256"],
                "scoring_inputs_artifact": dict(descriptor)}
    except (OSError, ValueError, KeyError, TypeError):
        return fail("wait_reference_authentication_failed")


def _snapshot(descriptor: dict, *, prefix: str) -> tuple[dict | None, str | None]:
    try:
        raw = Path(descriptor["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != descriptor.get("sha256") or (
            "byte_count" in descriptor and len(raw) != descriptor["byte_count"]
        ):
            return None, f"{prefix}_snapshot_hash_mismatch"
        payload = json.loads(raw)["payload"]
        if not isinstance(payload.get("identity"), dict) or not isinstance(payload.get("inputs"), dict):
            return None, f"{prefix}_snapshot_invalid"
        return payload, None
    except (OSError, KeyError, ValueError, TypeError):
        return None, f"{prefix}_snapshot_invalid"


def _records(payload: dict) -> dict[int, dict] | None:
    records = payload["inputs"].get("backend_tick_records")
    if not isinstance(records, list) or not records:
        return None
    for tick, row in enumerate(records):
        if (not isinstance(row, dict) or type(row.get("tick")) is not int
                or row["tick"] != tick or type(row.get("n_voltage_violations")) is not int
                or row["n_voltage_violations"] < 0):
            return None
    return {row["tick"]: row for row in records}


def _record_evidence(payload: dict, records: dict[int, dict]) -> dict[int, str] | None:
    items = (payload["inputs"].get("evidence_logger") or {}).get("items")
    if not isinstance(items, list):
        return None
    ids = {}
    for tick, row in records.items():
        matched = [item for item in items if isinstance(item, dict)
                   and item.get("kind") == "backend_tick" and item.get("tick") == tick]
        if (len(matched) != 1 or not isinstance(matched[0].get("evidence_id"), str)
                or not matched[0]["evidence_id"]
                or (matched[0].get("payload") or {}).get("n_voltage_violations") != row["n_voltage_violations"]):
            return None
        ids[tick] = matched[0]["evidence_id"]
    return ids


def score_legacy_persistence(
    episode: dict, contract: dict, *, artifact_binding: dict, baseline: dict,
) -> dict[str, Any]:
    """Count fixed native stages met; missing evidence never shrinks scope."""
    result = {"schema_version": VERSION, "evaluation_version": "0.23.0",
              "applicable": False, "score": None, "expected_phases": None,
              "successful_phases": None, "phases": [], "reason": None,
              "evidence_ids": [], "baseline_evidence_ids": [],
              "interpretation": "native_multi_stage_fulfillment_not_memory_retention",
              "formal_run_certified": False}

    def fail(reason: str) -> dict:
        return {**result, "reason": reason}

    if contract.get("applicable") is not True:
        return fail(contract.get("reason", "native_phase_contract_unavailable"))
    try:
        if _hash({k: v for k, v in contract.items() if k != "contract_sha256"}) != contract.get("contract_sha256"):
            return fail("compiled_contract_hash_mismatch")
    except (ValueError, TypeError):
        return fail("compiled_contract_hash_mismatch")
    if (episode.get("status") != "ok" or any(episode.get(key) != contract.get(key)
            for key in ("scenario_signature", "seed", "domain", "backend_kind"))):
        return fail("episode_contract_identity_mismatch")
    if artifact_binding.get("verified") is not True or artifact_binding.get("native_cost_bound") is not True:
        return fail("unbound_episode_artifacts")
    if (baseline.get("verified") is not True or not _digest(baseline.get("reference_report_sha256"))
            or any(baseline.get(key) != contract.get(key) for key in ("scenario_signature", "seed"))):
        return fail("unbound_wait_reference")
    actual_descriptor = (artifact_binding.get("artifacts") or {}).get("scoring_inputs_artifact") or {}
    actual, reason = _snapshot(actual_descriptor, prefix="actual")
    if reason:
        return fail(reason)
    wait_descriptor = baseline.get("scoring_inputs_artifact") or {}
    wait, reason = _snapshot(wait_descriptor, prefix="baseline")
    if reason:
        return fail(reason)
    metadata = artifact_binding.get("execution_metadata") or {}
    user_assumed = (metadata.get("comparison_policy") == "latest_framework_user_assumed"
                    and metadata.get("compatibility_user_assumed") is True)
    actual_runtime = metadata.get("snapshot_runtime_identity") if user_assumed else episode.get("implementation_tree_sha256")
    for label, payload, runtime in (("actual", actual, actual_runtime),
                                     ("baseline", wait, baseline.get("runtime_identity"))):
        identity = payload["identity"]
        if (not runtime or any(identity.get(key) != contract[key] for key in ("scenario_signature", "seed"))
                or (identity.get("implementation") or {}).get("implementation_tree_sha256") != runtime
                or (label == "baseline" and identity.get("agent_name") != "wait_only")):
            return fail(f"{label}_snapshot_identity_mismatch")
    actual_records, wait_records = _records(actual), _records(wait)
    if actual_records is None or wait_records is None:
        return fail("native_phase_records_invalid")
    if (len(actual_records) != episode.get("n_ticks_ran")
            or len(actual_records) != artifact_binding.get("trace_ticks")):
        return fail("actual_phase_record_coverage_mismatch")
    actual_ids, wait_ids = _record_evidence(actual, actual_records), _record_evidence(wait, wait_records)
    if actual_ids is None or wait_ids is None:
        return fail("native_phase_evidence_binding_missing")
    phases = contract["phase_ticks"]
    result["expected_phases"] = len(phases)
    if any(tick not in wait_records for tick in phases):
        return fail("baseline_required_phase_missing")
    fatal = actual["inputs"].get("chose_fatal_option")
    if type(fatal) is not bool:
        return fail("native_phase_safety_unknown")
    hard_failure = fatal or any(row.get("catastrophic_failure") is True for row in actual_records.values())
    successes = 0
    for tick in phases:
        row = actual_records.get(tick)
        reduction = None if row is None else wait_records[tick]["n_voltage_violations"] - row["n_voltage_violations"]
        passed = row is not None and not hard_failure and reduction >= contract["minimum_reduction_each_phase"]
        successes += bool(passed)
        actual_id = actual_ids[tick] if row is not None else actual_ids[max(actual_ids)]
        result["evidence_ids"].append(actual_id)
        result["baseline_evidence_ids"].append(wait_ids[tick])
        result["phases"].append({"tick": tick, "baseline_violations": wait_records[tick]["n_voltage_violations"],
                                 "actual_violations": None if row is None else row["n_voltage_violations"],
                                 "reduction": reduction, "fulfilled": bool(passed),
                                 "evidence_ids": [actual_id], "baseline_evidence_ids": [wait_ids[tick]],
                                 "reason": "episode_ended_before_required_phase" if row is None else
                                 "task_hard_failure" if hard_failure else "native_phase_met" if passed else "native_phase_not_met"})
    result["evidence_ids"] = list(dict.fromkeys(result["evidence_ids"]))
    result["baseline_evidence_ids"] = list(dict.fromkeys(result["baseline_evidence_ids"]))
    result.update(comparison_policy="latest_framework_user_assumed" if user_assumed else "strict",
                  applicable=True, score=100.0 * successes / len(phases), successful_phases=successes,
                  hard_failure=hard_failure, reason="native_multi_stage_fulfillment_recomputed",
                  contract_sha256=contract["contract_sha256"], scenario_sha256=contract["scenario_sha256"],
                  actual_snapshot_sha256=actual_descriptor["sha256"], baseline_snapshot_sha256=wait_descriptor["sha256"],
                  reference_report_sha256=baseline["reference_report_sha256"])
    return result
