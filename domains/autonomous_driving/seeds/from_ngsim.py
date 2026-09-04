"""Create provenance-complete, non-release seeds from mined NGSIM windows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from domains.autonomous_driving.data.contracts import object_sha256


def build_seed_records(
    mining_report: Mapping[str, Any],
    *,
    source_evidence_sha256: str,
) -> dict[str, Any]:
    """Build stable seed descriptors without claiming a simulator backend exists."""
    if mining_report.get("schema_version") != "ngsim_mining_report_v1":
        raise ValueError("ngsim_seed_mining_schema_mismatch")
    mining_evidence = str(mining_report.get("mining_evidence_sha256") or "")
    if len(source_evidence_sha256) != 64 or len(mining_evidence) != 64:
        raise ValueError("ngsim_seed_evidence_invalid")
    seeds: list[dict[str, Any]] = []
    for candidate in mining_report.get("candidates") or []:
        candidate_id = str(candidate.get("candidate_id") or "")
        window_sha256 = str(candidate.get("source_window_sha256") or "")
        if not candidate_id or len(window_sha256) != 64:
            raise ValueError("ngsim_seed_candidate_invalid")
        identity = {
            "source_evidence_sha256": source_evidence_sha256,
            "mining_evidence_sha256": mining_evidence,
            "candidate_id": candidate_id,
            "source_window_sha256": window_sha256,
            "hazard_kind": candidate.get("hazard_kind"),
            "hazard_context": dict(candidate.get("hazard_context") or {}),
            "window_semantics": dict(candidate.get("window_semantics") or {}),
        }
        seed_digest = object_sha256(identity)
        seed: dict[str, Any] = {
            "schema_version": "autonomous_driving_seed_v1",
            "scenario_id": f"autonomous_driving/ngsim/{seed_digest[:20]}",
            "seed_id": seed_digest[:16],
            "seed": int(seed_digest[:8], 16),
            "domain": "autonomous_driving",
            "backend_kind": "sumo_ego",
            "admission_status": "held",
            "held_reason": "live_sumo_reactive_validation_pending",
            "candidate_id": candidate_id,
            "start_time_ms": int(candidate["start_time_ms"]),
            "end_time_ms_exclusive": int(candidate["end_time_ms_exclusive"]),
            "actor_ids": list(candidate["actor_ids"]),
            "source_window_sha256": window_sha256,
            "hazard_kind": candidate.get("hazard_kind"),
            "hazard_context": dict(candidate.get("hazard_context") or {}),
            "window_semantics": dict(candidate.get("window_semantics") or {}),
            "decision_axes": {
                "naturalistic_window": True,
                "actor_count": int(candidate["actor_count"]),
                "max_concurrent_actors": int(candidate["risk_features"]["max_concurrent_actors"]),
                "lane_change_count": int(candidate["risk_features"]["lane_change_count"]),
                "hard_brake_magnitude_milli_mps2": int(
                    candidate["risk_features"]["hard_brake_magnitude_milli_mps2"]
                ),
            },
            "provenance": identity,
        }
        seed["scenario_signature"] = object_sha256(seed)
        seeds.append(seed)
    result: dict[str, Any] = {
        "schema_version": "autonomous_driving_seed_set_v1",
        "source_evidence_sha256": source_evidence_sha256,
        "mining_evidence_sha256": mining_evidence,
        "seeds": seeds,
    }
    result["seed_set_sha256"] = object_sha256(result)
    return result
