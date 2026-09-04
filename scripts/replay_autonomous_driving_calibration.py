#!/usr/bin/env python3
"""Repeat native driving calibration legs and audit semantic determinism."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess  # nosec B404 - fixed argv, no shell
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    output = path.resolve()
    partial = output.with_name(f"{output.name}.part")
    if output.exists() or partial.exists():
        raise FileExistsError("autonomous_driving_replay_output_exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(output)


def _candidate_from_bundle(bundle: Path, candidate_id: str) -> tuple[str, str]:
    fixture = _load(bundle / "runtime/fixture.json")
    derivation = dict(fixture.get("derivation") or {})
    if str(derivation.get("candidate_id") or "") != candidate_id:
        raise ValueError("autonomous_driving_replay_candidate_binding_mismatch")
    ego = str(derivation.get("ego_actor_id") or "")
    if not ego:
        raise ValueError("autonomous_driving_replay_ego_missing")
    return candidate_id, ego


def _formal_scenario_inputs(
    path: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], int, int, str, str, str]:
    from scripts.run_autonomous_driving_legs import _load_bound_scenario_yaml

    scenario, bundle, artifact = _load_bound_scenario_yaml(path.resolve())
    backend = dict(scenario.get("backend_config") or {})
    return (
        scenario,
        bundle,
        artifact,
        int(scenario["seed"]),
        int(scenario["horizon_ticks"]),
        str(scenario["difficulty_level"]),
        str(backend["candidate_id"]),
        str(backend["ego_actor_id"]),
    )


def _reference_calibration(
    path: Path,
    *,
    bundle: Path,
    candidate_id: str,
    ego: str,
    seed: int,
    ticks: int,
    difficulty_level: str,
    scenario_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = _load(path.resolve())
    unsigned = dict(report)
    expected_digest = str(unsigned.pop("report_digest_sha256", ""))
    scenario = dict(report.get("scenario") or {})
    backend = dict(scenario.get("backend_config") or {})
    source_bundle_ref = str(backend.get("source_bundle") or "")
    source_bundle_matches = (
        source_bundle_ref == "." or Path(source_bundle_ref).resolve() == bundle.resolve()
    )
    legs = report.get("legs")
    leg_names = {str(value.get("leg") or "") for value in legs or [] if isinstance(value, dict)}
    formal = scenario_artifact is not None
    schema_valid = report.get("schema_version") == (
        "autonomous_driving_calibration_legs_v2"
        if formal
        else "autonomous_driving_calibration_legs_v1"
    )
    artifact_valid = not formal or (
        report.get("evidence_tier") == "formal_yaml_bound_v1"
        and report.get("scenario_artifact") == scenario_artifact
    )
    if (
        not schema_valid
        or not artifact_valid
        or report.get("status") != "diagnostic_complete"
        or int(report.get("seed") or -1) != seed
        or int(scenario.get("horizon_ticks") or 0) != ticks
        or scenario.get("difficulty_level") != difficulty_level
        or str(backend.get("candidate_id") or "") != candidate_id
        or str(backend.get("ego_actor_id") or "") != ego
        or not source_bundle_matches
        or backend.get("execution_mode") != "live"
        or leg_names != {"shield_only", "rule_tactical", "oracle_offline"}
        or len(legs or []) != 3
        or not expected_digest
        or expected_digest != _digest(unsigned)
    ):
        raise ValueError("autonomous_driving_replay_reference_calibration_invalid")
    return report


def _run_once(
    *,
    bundle: Path,
    candidate_id: str,
    ego: str,
    seed: int,
    ticks: int,
    difficulty_level: str,
    scenario_yaml: Path | None = None,
) -> dict[str, Any]:
    command = [sys.executable, str(REPO_ROOT / "scripts/run_autonomous_driving_legs.py")]
    if scenario_yaml is not None:
        command.extend(("--scenario-yaml", str(scenario_yaml.resolve())))
    else:
        command.extend(
            (
                "--bundle",
                str(bundle),
                "--candidate-id",
                candidate_id,
                "--ego",
                ego,
                "--seed",
                str(seed),
                "--ticks",
                str(ticks),
                "--difficulty-level",
                difficulty_level,
            )
        )
    command.extend(("--legs", "shield_only", "rule_tactical", "oracle_offline"))
    completed = subprocess.run(  # nosec B603
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(180.0, ticks * 30.0),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no child output"
        raise RuntimeError(f"autonomous_driving_replay_leg_failed: {detail}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("autonomous_driving_replay_report_not_json") from exc
    if not isinstance(report, dict) or report.get("status") != "diagnostic_complete":
        raise ValueError("autonomous_driving_replay_report_not_complete")
    return report


def build_replay_report(
    *,
    bundle: Path | None = None,
    candidate_id: str | None = None,
    scenario_yaml: Path | None = None,
    seed: int | None = None,
    ticks: int | None = None,
    repeats: int = 3,
    difficulty_level: str | None = None,
    reference_calibration: Path | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    output: Path | None = None,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("autonomous_driving_replay_repeats_too_small")
    scenario_artifact: dict[str, Any] | None = None
    if scenario_yaml is not None:
        if any(
            value is not None for value in (bundle, candidate_id, seed, ticks, difficulty_level)
        ):
            raise ValueError("autonomous_driving_replay_scenario_yaml_override_forbidden")
        (
            _scenario,
            bundle,
            scenario_artifact,
            seed,
            ticks,
            difficulty_level,
            candidate_id,
            ego,
        ) = _formal_scenario_inputs(scenario_yaml)
        scenario_yaml = scenario_yaml.resolve()
        evidence_tier = "formal_yaml_bound_v1"
    else:
        if bundle is None or not candidate_id:
            raise ValueError("autonomous_driving_replay_legacy_identity_required")
        seed = 42 if seed is None else seed
        ticks = 8 if ticks is None else ticks
        difficulty_level = difficulty_level or "high"
        bundle = bundle.resolve()
        candidate_id, ego = _candidate_from_bundle(bundle, candidate_id)
        evidence_tier = "diagnostic_legacy_generic_scenario_v1"
    if (
        bundle is None
        or candidate_id is None
        or seed is None
        or ticks is None
        or difficulty_level is None
    ):
        raise ValueError("autonomous_driving_replay_resolved_inputs_missing")
    if ticks < 1:
        raise ValueError("autonomous_driving_replay_ticks_invalid")
    reports: list[dict[str, Any]] = []
    repeat_sources: list[str] = []
    repeat_evidence: list[str] = []
    if reference_calibration is not None:
        reports.append(
            _reference_calibration(
                reference_calibration,
                bundle=bundle,
                candidate_id=candidate_id,
                ego=ego,
                seed=seed,
                ticks=ticks,
                difficulty_level=difficulty_level,
                scenario_artifact=scenario_artifact,
            )
        )
        repeat_sources.append("reference_calibration")
        repeat_evidence.append(reference_calibration.name)
    fresh_repeats = repeats - len(reports)
    for _ in range(fresh_repeats):
        repeat_number = len(reports) + 1
        checkpoint = (
            checkpoint_dir.resolve() / f"repeat_{repeat_number:03d}.json"
            if checkpoint_dir is not None
            else None
        )
        if checkpoint is not None and checkpoint.exists():
            if not resume:
                raise FileExistsError("autonomous_driving_replay_checkpoint_exists")
            report = _reference_calibration(
                checkpoint,
                bundle=bundle,
                candidate_id=candidate_id,
                ego=ego,
                seed=seed,
                ticks=ticks,
                difficulty_level=difficulty_level,
                scenario_artifact=scenario_artifact,
            )
        else:
            report = _run_once(
                bundle=bundle,
                candidate_id=candidate_id,
                ego=ego,
                seed=seed,
                ticks=ticks,
                difficulty_level=difficulty_level,
                scenario_yaml=scenario_yaml,
            )
            if checkpoint is not None:
                _write_json_exclusive(checkpoint, report)
        reports.append(report)
        repeat_sources.append("fresh_native_replay")
        repeat_evidence.append(checkpoint.name if checkpoint is not None else "inline")
    leg_digests: dict[str, list[str]] = {}
    for report in reports:
        for leg in report.get("legs") or []:
            if isinstance(leg, dict):
                leg_digests.setdefault(str(leg.get("leg") or ""), []).append(
                    str(leg.get("semantic_digest") or "")
                )
    deterministic = bool(leg_digests) and all(
        len(values) == repeats and len(set(values)) == 1 and values[0]
        for values in leg_digests.values()
    )
    artifact_consistent = scenario_artifact is None or all(
        report.get("schema_version") == "autonomous_driving_calibration_legs_v2"
        and report.get("evidence_tier") == "formal_yaml_bound_v1"
        and report.get("scenario_artifact") == scenario_artifact
        for report in reports
    )
    from domains.autonomous_driving.evidence_binding import (
        runtime_implementation_binding,
        sumo_runtime_binding,
    )

    current_implementation = runtime_implementation_binding(REPO_ROOT)
    current_runtime = sumo_runtime_binding(REPO_ROOT)
    evidence_bindings = [dict(report.get("evidence_binding") or {}) for report in reports]
    binding = evidence_bindings[0] if evidence_bindings else {}
    binding_consistent = bool(binding) and all(value == binding for value in evidence_bindings[1:])
    binding_current = (
        binding.get("candidate_id") == candidate_id
        and binding.get("implementation_sha256")
        == current_implementation["autonomous_driving_slice_sha256"]
        and binding.get("semantics_sha256") == current_implementation["semantics_sha256"]
        and binding.get("runtime_sha256") == current_runtime["runtime_sha256"]
        and all(
            isinstance(binding.get(name), str) and len(str(binding[name])) == 64
            for name in (
                "input_sha256",
                "runtime_sha256",
                "source_window_sha256",
                "source_event_chain_sha256",
            )
        )
    )
    deterministic = deterministic and binding_consistent and binding_current and artifact_consistent
    calibration_report_digest = str(reports[0].get("report_digest_sha256") or "")
    result: dict[str, Any] = {
        "schema_version": (
            "autonomous_driving_replay_audit_v2"
            if scenario_artifact is not None
            else "autonomous_driving_replay_audit_v1"
        ),
        "status": "verified" if deterministic else "held",
        "evidence_tier": evidence_tier,
        "formal_core_allowed": False,
        "candidate_id": candidate_id,
        "ego_actor_id": ego,
        "bundle": ".",
        "seed": seed,
        "ticks": ticks,
        "difficulty_level": difficulty_level,
        "repeats": repeats,
        "repeat_sources": repeat_sources,
        "repeat_evidence": repeat_evidence,
        "reference_calibration": reference_calibration.name if reference_calibration else None,
        "leg_semantic_digests": leg_digests,
        "deterministic_semantic_replay": deterministic,
        "evidence_binding": {
            **binding,
            "calibration_report_digest_sha256": calibration_report_digest,
        }
        if binding_consistent and binding_current and len(calibration_report_digest) == 64
        else {},
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "policy": "cross_platform_runtime_fingerprint_v1",
        },
        "sumo_runtime": current_runtime,
    }
    if scenario_artifact is not None:
        result["scenario_artifact"] = scenario_artifact
    result["replay_digest_sha256"] = _digest(result)
    if output is not None:
        _write_json_exclusive(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-yaml", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-calibration", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--difficulty-level", choices=("basic", "medium", "high", "extreme"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    bundle = (
        None
        if args.bundle is None
        else args.bundle
        if args.bundle.is_absolute()
        else REPO_ROOT / args.bundle
    )
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    try:
        report = build_replay_report(
            bundle=bundle,
            candidate_id=args.candidate_id,
            scenario_yaml=(
                args.scenario_yaml
                if args.scenario_yaml is None or args.scenario_yaml.is_absolute()
                else REPO_ROOT / args.scenario_yaml
            ),
            seed=args.seed,
            ticks=args.ticks,
            repeats=args.repeats,
            difficulty_level=args.difficulty_level,
            reference_calibration=(
                args.reference_calibration
                if args.reference_calibration is None or args.reference_calibration.is_absolute()
                else REPO_ROOT / args.reference_calibration
            ),
            checkpoint_dir=(
                args.checkpoint_dir
                if args.checkpoint_dir is None or args.checkpoint_dir.is_absolute()
                else REPO_ROOT / args.checkpoint_dir
            ),
            resume=args.resume,
            output=None if args.check else output,
        )
        verified = report["status"] == "verified" and report["deterministic_semantic_replay"]
        if args.check:
            if not output.is_file():
                raise ValueError("autonomous_driving_replay_report_missing")
            verified = verified and _load(output) == report
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"status": "verified" if verified else "held"}, sort_keys=True))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
