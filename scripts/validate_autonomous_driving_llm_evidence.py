#!/usr/bin/env python3
"""Validate provider-backed autonomous-driving episode evidence.

Baseline calibration reports deliberately cannot satisfy this contract.  The
manifest must bind each provider result to a source candidate and difficulty
slice, while the result must prove strict prompting, a successful provider
call, source-event consumption, and a safe backend outcome.  A held report is
still useful for diagnostics but can never open the Core LLM gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_AGENT_NAMES = {"llm_agent", "react_llm", "reflexion_llm"}
DEFAULT_DIFFICULTIES = ("basic", "medium", "high", "extreme")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _valid_hash(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _run_report(
    *, manifest: dict[str, Any], entry: dict[str, Any], manifest_dir: Path
) -> dict[str, Any]:
    blockers: list[str] = []
    result_path = Path(str(entry.get("result_path") or ""))
    if not result_path.is_absolute():
        result_path = manifest_dir / result_path
    try:
        relative_path = result_path.resolve().relative_to(manifest_dir.resolve()).as_posix()
    except ValueError:
        return {
            "candidate_id": str(entry.get("candidate_id") or ""),
            "difficulty_level": str(entry.get("difficulty_level") or ""),
            "scenario_id": str(entry.get("scenario_id") or ""),
            "result_path": str(result_path),
            "status": "held",
            "blockers": ["result_path_outside_manifest"],
        }
    if not result_path.is_file():
        return {
            "candidate_id": str(entry.get("candidate_id") or ""),
            "difficulty_level": str(entry.get("difficulty_level") or ""),
            "scenario_id": str(entry.get("scenario_id") or ""),
            "result_path": relative_path,
            "status": "held",
            "blockers": ["result_file_missing"],
        }
    result = _load(result_path)
    result_hash = _sha256(result_path)
    declared_hash = entry.get("result_sha256")
    if declared_hash is not None and str(declared_hash) != result_hash:
        blockers.append("result_hash_mismatch")
    candidate_id = str(entry.get("candidate_id") or "")
    difficulty = str(entry.get("difficulty_level") or "")
    scenario_id = str(entry.get("scenario_id") or "")
    scenario_yaml_sha256 = str(entry.get("scenario_yaml_sha256") or "")
    semantic_contract_sha256 = str(entry.get("semantic_contract_sha256") or "")
    horizon_ticks = entry.get("horizon_ticks")
    if not candidate_id:
        blockers.append("candidate_binding_missing")
    if difficulty not in DEFAULT_DIFFICULTIES:
        blockers.append("difficulty_level_invalid")
    if str(result.get("scenario_id") or "") != scenario_id:
        blockers.append("scenario_binding_mismatch")
    if str(result.get("difficulty_level") or "") != difficulty:
        blockers.append("difficulty_binding_mismatch")
    if not _valid_hash(scenario_yaml_sha256) or not _valid_hash(
        semantic_contract_sha256
    ):
        blockers.append("scenario_artifact_binding_missing")
    if not isinstance(horizon_ticks, int) or isinstance(horizon_ticks, bool) or horizon_ticks <= 0:
        blockers.append("scenario_horizon_binding_missing")
    if (
        manifest.get("requires_live_backend") is True
        and str(entry.get("execution_mode") or "") != "live"
    ):
        blockers.append("live_backend_requirement_not_proven")
    ground_truth = result.get("ground_truth_summary")
    if not isinstance(ground_truth, dict):
        ground_truth = {}
        blockers.append("ground_truth_summary_missing")
    if str(ground_truth.get("candidate_id") or "") != candidate_id:
        blockers.append("candidate_identity_not_consumed_by_backend")
    expected_window_hash = str(entry.get("source_window_sha256") or "")
    observed_window_hash = str(ground_truth.get("source_window_sha256") or "")
    if not _valid_hash(expected_window_hash) or observed_window_hash != expected_window_hash:
        blockers.append("source_window_binding_mismatch")
    expected_event_hash = str(entry.get("source_event_chain_sha256") or "")
    observed_event_hash = str(ground_truth.get("source_event_chain_sha256") or "")
    if not _valid_hash(expected_event_hash) or observed_event_hash != expected_event_hash:
        blockers.append("source_event_chain_binding_mismatch")
    realized = ground_truth.get("realized_events")
    material_events = [
        event
        for event in realized or []
        if isinstance(event, dict)
        and event.get("origin") == "source_schedule"
        and event.get("materiality_passed") is True
    ]
    if not material_events:
        blockers.append("material_source_event_not_realized")
    agent_name = str(result.get("agent_name") or "")
    if agent_name != str(manifest.get("agent_name") or ""):
        blockers.append("agent_identity_mismatch")
    if agent_name not in REQUIRED_AGENT_NAMES:
        blockers.append("agent_name_not_provider_backed")
    agent_config = result.get("agent_config")
    config = agent_config.get("config") if isinstance(agent_config, dict) else None
    config = config if isinstance(config, dict) else {}
    declared_prompt_mode = str(manifest.get("prompt_mode") or "")
    if declared_prompt_mode != "strict" or str(config.get("prompt_mode") or "") != "strict":
        blockers.append("strict_prompt_not_proven")
    if str(config.get("provider") or "") != str(manifest.get("provider") or ""):
        blockers.append("provider_identity_mismatch")
    if str(config.get("model") or "") != str(manifest.get("model") or ""):
        blockers.append("model_identity_mismatch")
    trajectory = result.get("trajectory_summary")
    llm = trajectory.get("llm") if isinstance(trajectory, dict) else None
    llm = llm if isinstance(llm, dict) else {}
    calls_ok = _number(llm.get("llm_calls_ok"))
    calls_failed = _number(llm.get("llm_calls_failed"))
    fallbacks = _number(llm.get("ticks_wait_fallback"))
    request_count = _number(llm.get("provider_request_count"))
    if calls_ok is None or calls_ok < 1 or request_count is None or request_count < 1:
        blockers.append("provider_call_missing")
    if calls_failed not in (None, 0.0):
        blockers.append("provider_call_failed")
    if fallbacks not in (None, 0.0):
        blockers.append("provider_fallback_observed")
    provider_models = llm.get("provider_models")
    if not isinstance(provider_models, list) or not provider_models:
        blockers.append("provider_model_telemetry_missing")
    elif str(manifest.get("model") or "") not in {str(value) for value in provider_models}:
        blockers.append("provider_model_telemetry_mismatch")
    collision = _number(ground_truth.get("collision_count"))
    departure = _number(ground_truth.get("road_departure_count"))
    if collision is None or departure is None or collision > 0 or departure > 0:
        blockers.append("collision_or_departure_observed")
    assurance = ground_truth.get("runtime_assurance")
    if not isinstance(assurance, dict) or not isinstance(assurance.get("mode_trace"), list):
        blockers.append("runtime_assurance_trace_missing")
    tactical_trace = ground_truth.get("tactical_action_trace")
    if not isinstance(tactical_trace, list):
        blockers.append("tactical_action_trace_missing")
    elif any(
        not isinstance(row, dict)
        or not str(row.get("tool_name") or "")
        or not isinstance(row.get("tick"), int | float)
        or isinstance(row.get("tick"), bool)
        for row in tactical_trace
    ):
        blockers.append("tactical_action_trace_invalid")
    counterfactual = result.get("counterfactual")
    if not isinstance(counterfactual, dict) or counterfactual.get("applicable") is not True:
        blockers.append("counterfactual_replay_missing")
    task_completion = result.get("task_completion")
    if not isinstance(task_completion, dict):
        blockers.append("task_completion_evidence_missing")
    else:
        task_evidence = task_completion.get("evidence")
        if not isinstance(task_evidence, dict):
            blockers.append("task_completion_native_evidence_missing")
        elif task_evidence.get("native_control_requirements_met") is not True:
            blockers.append("task_completion_native_requirements_failed")
    return {
        "candidate_id": candidate_id,
        "difficulty_level": difficulty,
        "scenario_id": scenario_id,
        "scenario_yaml_sha256": scenario_yaml_sha256,
        "semantic_contract_sha256": semantic_contract_sha256,
        "horizon_ticks": horizon_ticks,
        "result_path": relative_path,
        "result_sha256": result_hash,
        "status": "verified" if not blockers else "held",
        "blockers": sorted(set(blockers)),
        "material_source_event_count": len(material_events),
        "provider_calls_ok": int(calls_ok or 0),
        "provider_request_count": int(request_count or 0),
    }


def build_evidence_report(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = _load(manifest_path)
    if manifest.get("schema_version") != "autonomous_driving_llm_evidence_manifest_v1":
        raise ValueError("autonomous_driving_llm_evidence_manifest_schema_invalid")
    required = tuple(
        str(level) for level in manifest.get("required_difficulties") or DEFAULT_DIFFICULTIES
    )
    if not required or any(level not in DEFAULT_DIFFICULTIES for level in required):
        raise ValueError("autonomous_driving_llm_evidence_required_difficulties_invalid")
    entries = manifest.get("runs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("autonomous_driving_llm_evidence_runs_missing")
    reports = [
        _run_report(manifest=manifest, entry=dict(entry), manifest_dir=manifest_path.parent)
        for entry in entries
        if isinstance(entry, dict)
    ]
    suite_binding = {
        f"{str(entry.get('candidate_id') or '')}:{str(entry.get('difficulty_level') or '')}": {
            "scenario_yaml_sha256": str(entry.get("scenario_yaml_sha256") or ""),
            "semantic_contract_sha256": str(entry.get("semantic_contract_sha256") or ""),
            "horizon_ticks": entry.get("horizon_ticks"),
        }
        for entry in entries
        if isinstance(entry, dict)
    }
    observed_suite_digest = _digest(suite_binding)
    declared_suite_digest = str(manifest.get("suite_semantic_coverage_sha256") or "")
    blockers: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for report in reports:
        key = (str(report.get("candidate_id") or ""), str(report.get("difficulty_level") or ""))
        if key in seen:
            blockers.append({"code": "duplicate_run_binding", "scope": ":".join(key)})
        seen.add(key)
        if report.get("status") != "verified":
            for code in report.get("blockers") or ["run_not_verified"]:
                blockers.append({"code": str(code), "scope": ":".join(key)})
    candidate_ids = sorted(
        {str(report.get("candidate_id") or "") for report in reports if report.get("candidate_id")}
    )
    if declared_suite_digest != observed_suite_digest:
        blockers.append({"code": "suite_semantic_coverage_mismatch", "scope": "manifest"})
    for candidate_id in candidate_ids:
        levels = {
            str(report.get("difficulty_level") or "")
            for report in reports
            if report.get("candidate_id") == candidate_id
        }
        missing = sorted(set(required) - levels)
        if missing:
            blockers.append(
                {
                    "code": "difficulty_coverage_missing",
                    "scope": f"{candidate_id}:{','.join(missing)}",
                }
            )
    payload: dict[str, Any] = {
        "schema_version": "autonomous_driving_llm_evidence_v1",
        "status": "verified" if not blockers else "held",
        "manifest_sha256": _sha256(manifest_path),
        "agent_name": str(manifest.get("agent_name") or ""),
        "provider": str(manifest.get("provider") or ""),
        "model": str(manifest.get("model") or ""),
        "prompt_mode": str(manifest.get("prompt_mode") or ""),
        "required_difficulties": list(required),
        "suite_semantic_coverage_sha256": observed_suite_digest,
        "candidate_ids": candidate_ids,
        "runs": sorted(
            reports,
            key=lambda value: (str(value.get("candidate_id")), str(value.get("difficulty_level"))),
        ),
        "blockers": sorted(blockers, key=lambda value: (value["scope"], value["code"])),
    }
    payload["evidence_digest_sha256"] = _digest(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    manifest = args.manifest if args.manifest.is_absolute() else REPO_ROOT / args.manifest
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    try:
        report = build_evidence_report(manifest)
        if args.check:
            verified = output.is_file() and _load(output) == report
        else:
            if output.exists():
                raise FileExistsError("autonomous_driving_llm_evidence_output_exists")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            verified = True
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True))
        return 1
    print(
        json.dumps(
            {"status": "verified" if verified else "stale", "evidence_status": report["status"]},
            sort_keys=True,
        )
    )
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
