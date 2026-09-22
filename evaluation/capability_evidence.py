"""Bind offline measurements to the runner's immutable local artifacts.

This authenticates against recorded hashes, not an adversary able to rewrite
both artifacts and manifests. It does not certify providers or formal runs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def bind_capability_evidence(
    episode: dict[str, Any], spec: dict[str, Any],
    source_path: str | Path | None = None,
    *, comparison_policy: str = "strict",
) -> dict[str, Any]:
    """Check trace, evidence, scoring inputs and their episode-local identity."""
    if comparison_policy not in {"strict", "latest_framework_user_assumed"}:
        raise ValueError("unknown comparison policy")
    user_assumed = comparison_policy == "latest_framework_user_assumed"
    result: dict[str, Any] = {
        "verified": False, "reason": None,
        "validation_scope": "recorded_local_artifact_hashes_and_episode_binding",
        "artifacts": {}, "trace_ticks": None, "evidence_count": None,
        "native_cost_bound": False, "counterfactual_bound": False,
        "formal_run_certified": False,
    }

    def fail(reason: str) -> dict[str, Any]:
        result["reason"] = reason
        return result

    for key in ("scenario_signature", "seed"):
        if episode.get(key) is None or episode.get(key) != spec.get(key):
            return fail("episode_suite_identity_mismatch")
    for key in ("backend_kind", "domain"):
        if key in spec and episode.get(key) not in (None, spec[key]):
            return fail("episode_suite_identity_mismatch")
    summary = episode.get("trajectory_summary") or {}
    loaded: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for key in ("trajectory_artifact", "evidence_ledger_artifact", "scoring_inputs_artifact"):
        descriptor = summary.get(key)
        if not isinstance(descriptor, dict) or not descriptor.get("sha256") or not descriptor.get("path"):
            return fail("required_artifact_descriptor_missing")
        declared = Path(descriptor["path"])
        candidates = [declared]
        if source_path is not None:
            root = Path(source_path).parent
            if not declared.is_absolute():
                candidates.append(root / declared)
            if "trajectories" in declared.parts:
                index = declared.parts.index("trajectories")
                candidates.append(root.joinpath(*declared.parts[index:]))
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            return fail("artifact_file_missing")
        try:
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != descriptor["sha256"]:
                return fail("artifact_hash_mismatch")
            if "byte_count" in descriptor and descriptor["byte_count"] != len(raw):
                return fail("artifact_byte_count_mismatch")
            value = (json.loads(raw) if key == "scoring_inputs_artifact" else
                     [json.loads(line) for line in raw.splitlines() if line.strip()])
        except (OSError, ValueError):
            return fail("artifact_unreadable_or_invalid_json")
        if key != "scoring_inputs_artifact":
            if not all(isinstance(row, dict) for row in value):
                return fail("artifact_record_shape_invalid")
            if "event_count" in descriptor and descriptor["event_count"] != len(value):
                return fail("artifact_event_count_mismatch")
        elif not isinstance(value, dict):
            return fail("artifact_record_shape_invalid")
        loaded[key] = value
        paths[key] = path
        result["artifacts"][key] = {"path": str(path), "sha256": digest, "byte_count": len(raw)}

    trace = loaded["trajectory_artifact"]
    count = len(trace)
    if (type(episode.get("n_ticks_ran")) is not int or
            episode["n_ticks_ran"] != count or
            summary.get("n_ticks") != count):
        return fail("trajectory_tick_count_mismatch")
    if [row.get("tick") for row in trace] != list(range(1, count + 1)):
        return fail("trajectory_tick_sequence_invalid")
    horizon = spec.get("horizon_ticks")
    if type(horizon) is int and count > horizon:
        return fail("trajectory_exceeds_configured_horizon")
    result["trace_ticks"] = count
    evidence = loaded["evidence_ledger_artifact"]
    ids = [row.get("evidence_id") for row in evidence]
    if not all(isinstance(value, str) and value for value in ids) or len(set(ids)) != len(ids):
        return fail("evidence_ledger_ids_invalid")
    by_id = dict(zip(ids, evidence))
    claimed = summary.get("operational_agency_valid_evidence_ids")
    if not isinstance(claimed, list) or not all(
        isinstance(value, str) and value in by_id for value in claimed
    ):
        return fail("summary_evidence_not_in_ledger")
    for row in trace:
        refs = row.get("evidence_ids") or []
        if not isinstance(refs, list) or not all(
            isinstance(value, str) and value in by_id for value in refs
        ):
            return fail("trajectory_evidence_not_in_ledger")
    result["evidence_count"] = len(evidence)
    snapshot = loaded["scoring_inputs_artifact"].get("payload")
    if not isinstance(snapshot, dict):
        return fail("scoring_snapshot_payload_missing")
    identity = snapshot.get("identity") or {}
    if any(identity.get(key) != episode.get(key) for key in ("scenario_signature", "seed")):
        return fail("scoring_snapshot_identity_mismatch")
    tree = (identity.get("implementation") or {}).get("implementation_tree_sha256")
    result["execution_metadata"] = {
        "snapshot_runtime_identity": tree,
        "episode_runtime_identity": episode.get("implementation_tree_sha256"),
        "episode_runtime_start": episode.get("implementation_tree_sha256_start"),
        "episode_runtime_end": episode.get("implementation_tree_sha256_end"),
        "comparison_policy": comparison_policy,
        "compatibility_user_assumed": user_assumed,
        "runtime_identity_matches": bool(tree) and tree == episode.get("implementation_tree_sha256"),
    }
    if not tree or (not user_assumed and tree != episode.get("implementation_tree_sha256")):
        return fail("scoring_snapshot_runtime_mismatch")
    inputs = snapshot.get("inputs") or {}
    costs = inputs.get("cost_components")
    if not isinstance(costs, dict) or not costs or costs != (episode.get("ground_truth_summary") or {}).get("cost_components"):
        return fail("native_cost_snapshot_mismatch")
    result["native_cost_bound"] = True
    replay = inputs.get("counterfactual_report")
    if not isinstance(replay, dict) or replay != episode.get("counterfactual"):
        return fail("counterfactual_snapshot_mismatch")
    result["counterfactual_bound"] = True
    snapshot_evidence = (inputs.get("evidence_logger") or {}).get("items")
    if not isinstance(snapshot_evidence, list) or not all(
        isinstance(item, dict) and by_id.get(item.get("evidence_id")) == item
        for item in snapshot_evidence
    ):
        return fail("scoring_snapshot_evidence_mismatch")
    trace_path = paths["trajectory_artifact"]
    header_path = trace_path.with_name(trace_path.name.replace(".trajectory.jsonl", ".header.json"))
    if header_path != trace_path and header_path.is_file():
        try:
            header = json.loads(header_path.read_bytes())
        except (OSError, ValueError):
            return fail("trajectory_header_unreadable")
        if not isinstance(header, dict) or any(
            header.get(key) != episode.get(key)
            for key in ("scenario_signature", "seed")
        ) or header.get("total_ticks") != count:
            return fail("trajectory_header_identity_mismatch")
        if any(key in spec and header.get(key) != spec[key]
               for key in ("horizon_ticks", "backend_kind", "domain")):
            return fail("trajectory_header_suite_mismatch")
    result.update(verified=True, reason="bound_local_execution_artifacts")
    return result
