#!/usr/bin/env python3
"""Run resumable native-SUMO calibration and replay for a driving catalog.

The runner is deliberately serial: every candidate is executed in isolated
child processes, and only evidence whose catalog, fixture, clock, and digest
bindings match is eligible for resume. It does not run or emulate an LLM.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess  # nosec B404 - fixed argv, no shell
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEG_NAMES = {"shield_only", "rule_tactical", "oracle_offline"}
TICK_SECONDS = 5.0
PHYSICS_STEP_SECONDS = 0.1


def object_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _digest_without(payload: dict[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return object_digest(unsigned)


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _portable_bundle_reference_matches(value: Any, bundle: Path) -> bool:
    reference = str(value or "")
    return reference == "." or _resolve_repo_path(reference) == bundle.resolve()


def _portable_artifact_reference_matches(value: Any, expected: Path) -> bool:
    reference = Path(str(value or ""))
    if reference.parent == Path("."):
        return reference.name == expected.name
    return _resolve_repo_path(reference) == expected.resolve()


def _current_calibration_evidence_binding(
    *, bundle: Path, candidate_id: str, legs: list[dict[str, Any]]
) -> dict[str, str]:
    from domains.autonomous_driving.evidence_binding import calibration_evidence_binding

    return calibration_evidence_binding(
        repo_root=REPO_ROOT,
        bundle=bundle,
        candidate_id=candidate_id,
        legs=legs,
    )


def candidate_file_stem(candidate_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate_id).strip("._-")
    if not stem:
        raise ValueError("autonomous_driving_batch_candidate_id_invalid")
    return stem


def load_catalog_rows(catalog_path: Path) -> list[dict[str, Any]]:
    catalog = _load(catalog_path.resolve())
    if catalog.get("schema_version") != "autonomous_driving_candidate_catalog_v1":
        raise ValueError("autonomous_driving_batch_catalog_schema_invalid")
    raw_rows = catalog.get("bundles")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("autonomous_driving_batch_catalog_empty")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in raw_rows:
        if not isinstance(value, dict):
            raise ValueError("autonomous_driving_batch_catalog_row_invalid")
        candidate_id = str(value.get("candidate_id") or "")
        ego_actor_id = str(value.get("ego_actor_id") or "")
        bundle_value = str(value.get("bundle_path") or "")
        if not candidate_id or not ego_actor_id or not bundle_value:
            raise ValueError("autonomous_driving_batch_catalog_identity_missing")
        if candidate_id in seen:
            raise ValueError("autonomous_driving_batch_catalog_candidate_duplicate")
        bundle = _resolve_repo_path(bundle_value)
        fixture = _load(bundle / "runtime/fixture.json")
        derivation = dict(fixture.get("derivation") or {})
        if str(derivation.get("candidate_id") or "") != candidate_id:
            raise ValueError("autonomous_driving_batch_catalog_fixture_candidate_mismatch")
        if str(derivation.get("ego_actor_id") or "") != ego_actor_id:
            raise ValueError("autonomous_driving_batch_catalog_fixture_ego_mismatch")
        seen.add(candidate_id)
        rows.append(
            {
                "candidate_id": candidate_id,
                "ego_actor_id": ego_actor_id,
                "bundle": bundle,
                "recording_id": str(value.get("recording_id") or ""),
                "hazard_kind": str(value.get("hazard_kind") or ""),
            }
        )
    return rows


def _load_bound_scenario_yaml(path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    from scripts.run_autonomous_driving_legs import _load_bound_scenario_yaml as load

    return load(path)


def bind_suite_scenarios(
    rows: list[dict[str, Any]], suite_dir: Path, *, difficulty_level: str
) -> list[dict[str, Any]]:
    """Bind each catalog row to exactly one validated suite scenario YAML."""
    suite_root = suite_dir.resolve()
    reports_by_candidate: dict[str, list[Path]] = {}
    for report_path in sorted(suite_root.glob("*/suite_report.json")):
        report = _load(report_path)
        candidate_id = str(report.get("candidate_id") or "")
        reports_by_candidate.setdefault(candidate_id, []).append(report_path)

    bound: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        reports = reports_by_candidate.get(candidate_id, [])
        if len(reports) != 1:
            raise ValueError("autonomous_driving_batch_suite_report_not_unique")
        report = _load(reports[0])
        slices = report.get("difficulty_slices")
        if not isinstance(slices, list):
            raise ValueError("autonomous_driving_batch_suite_slices_invalid")
        matches: list[tuple[Path, dict[str, Any], Path, dict[str, Any]]] = []
        for relative in slices:
            scenario_path = (reports[0].parent / str(relative)).resolve()
            if not scenario_path.is_relative_to(suite_root) or not scenario_path.is_file():
                raise ValueError("autonomous_driving_batch_suite_scenario_missing")
            scenario, bundle, artifact = _load_bound_scenario_yaml(scenario_path)
            if scenario.get("difficulty_level") == difficulty_level:
                matches.append((scenario_path, scenario, bundle, artifact))
        if len(matches) != 1:
            raise ValueError("autonomous_driving_batch_suite_difficulty_not_unique")
        scenario_path, scenario, bundle, artifact = matches[0]
        backend = dict(scenario.get("backend_config") or {})
        if (
            str(backend.get("candidate_id") or "") != candidate_id
            or str(backend.get("ego_actor_id") or "") != str(row["ego_actor_id"])
            or bundle.resolve() != Path(row["bundle"]).resolve()
            or str(artifact.get("candidate_id") or "") != candidate_id
        ):
            raise ValueError("autonomous_driving_batch_suite_identity_mismatch")
        bound.append(
            {
                **row,
                "scenario_yaml": scenario_path,
                "scenario_artifact": artifact,
                "seed": int(scenario["seed"]),
                "ticks": int(scenario["horizon_ticks"]),
                "difficulty_level": difficulty_level,
            }
        )
    return bound


def validate_calibration_report(
    path: Path,
    *,
    candidate_id: str,
    ego_actor_id: str,
    bundle: Path,
    seed: int,
    ticks: int,
    difficulty_level: str,
    scenario_artifact: dict[str, Any] | None = None,
    expected_current_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = _load(path)
    scenario = dict(report.get("scenario") or {})
    backend = dict(scenario.get("backend_config") or {})
    if (
        str(backend.get("candidate_id") or "") != candidate_id
        or str(backend.get("ego_actor_id") or "") != ego_actor_id
        or not _portable_bundle_reference_matches(backend.get("source_bundle"), bundle)
    ):
        raise ValueError("autonomous_driving_batch_calibration_identity_mismatch")
    if backend.get("execution_mode") != "live":
        raise ValueError("autonomous_driving_batch_calibration_not_native_live")
    clock = dict(scenario.get("clock_contract") or {})
    if (
        int(report.get("seed") or -1) != seed
        or float(scenario.get("tick_seconds") or 0.0) != TICK_SECONDS
        or float(clock.get("physics_step_seconds") or 0.0) != PHYSICS_STEP_SECONDS
        or float(clock.get("shield_step_seconds") or 0.0) != PHYSICS_STEP_SECONDS
    ):
        raise ValueError("autonomous_driving_batch_calibration_clock_or_seed_mismatch")
    if int(scenario.get("horizon_ticks") or 0) != ticks:
        raise ValueError("autonomous_driving_batch_calibration_ticks_mismatch")
    if scenario.get("difficulty_level") != difficulty_level:
        raise ValueError("autonomous_driving_batch_calibration_difficulty_mismatch")
    legs = report.get("legs")
    if not isinstance(legs, list):
        raise ValueError("autonomous_driving_batch_calibration_incomplete")
    leg_names = {str(value.get("leg") or "") for value in legs if isinstance(value, dict)}
    expected_schema = (
        "autonomous_driving_calibration_legs_v2"
        if scenario_artifact is not None
        else "autonomous_driving_calibration_legs_v1"
    )
    if (
        report.get("schema_version") != expected_schema
        or report.get("status") != "diagnostic_complete"
        or leg_names != LEG_NAMES
        or len(legs) != len(LEG_NAMES)
    ):
        raise ValueError("autonomous_driving_batch_calibration_incomplete")
    if scenario_artifact is not None and (
        report.get("evidence_tier") != "formal_yaml_bound_v1"
        or report.get("scenario_artifact") != scenario_artifact
    ):
        raise ValueError("autonomous_driving_batch_calibration_scenario_artifact_mismatch")
    current_binding = expected_current_binding or _current_calibration_evidence_binding(
        bundle=bundle,
        candidate_id=candidate_id,
        legs=[dict(value) for value in legs if isinstance(value, dict)],
    )
    if report.get("evidence_binding") != current_binding:
        raise ValueError("autonomous_driving_batch_calibration_evidence_binding_mismatch")
    expected = str(report.get("report_digest_sha256") or "")
    if not expected or expected != _digest_without(report, "report_digest_sha256"):
        raise ValueError("autonomous_driving_batch_calibration_digest_mismatch")
    return report


def validate_replay_report(
    path: Path,
    *,
    candidate_id: str,
    ego_actor_id: str,
    bundle: Path,
    calibration_path: Path,
    seed: int,
    ticks: int,
    repeats: int,
    difficulty_level: str,
    scenario_artifact: dict[str, Any] | None = None,
    expected_current_binding: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = _load(path)
    if (
        str(report.get("candidate_id") or "") != candidate_id
        or str(report.get("ego_actor_id") or "") != ego_actor_id
        or not _portable_bundle_reference_matches(report.get("bundle"), bundle)
    ):
        raise ValueError("autonomous_driving_batch_replay_identity_mismatch")
    expected_schema = (
        "autonomous_driving_replay_audit_v2"
        if scenario_artifact is not None
        else "autonomous_driving_replay_audit_v1"
    )
    if (
        report.get("schema_version") != expected_schema
        or report.get("status") != "verified"
        or report.get("deterministic_semantic_replay") is not True
    ):
        raise ValueError("autonomous_driving_batch_replay_not_verified")
    if scenario_artifact is not None and (
        report.get("evidence_tier") != "formal_yaml_bound_v1"
        or report.get("scenario_artifact") != scenario_artifact
    ):
        raise ValueError("autonomous_driving_batch_replay_scenario_artifact_mismatch")
    if (
        int(report.get("seed") or -1) != seed
        or int(report.get("ticks") or 0) != ticks
        or int(report.get("repeats") or 0) != repeats
        or report.get("difficulty_level") != difficulty_level
    ):
        raise ValueError("autonomous_driving_batch_replay_parameters_mismatch")
    digests = report.get("leg_semantic_digests")
    if not isinstance(digests, dict) or set(digests) != LEG_NAMES:
        raise ValueError("autonomous_driving_batch_replay_legs_incomplete")
    if any(
        not isinstance(values, list)
        or len(values) != repeats
        or len(set(str(item) for item in values)) != 1
        or not str(values[0])
        for values in digests.values()
    ):
        raise ValueError("autonomous_driving_batch_replay_digest_drift")
    if report.get("repeat_sources") != [
        "reference_calibration",
        *("fresh_native_replay" for _ in range(repeats - 1)),
    ]:
        raise ValueError("autonomous_driving_batch_replay_sources_invalid")
    if not _portable_artifact_reference_matches(
        report.get("reference_calibration"), calibration_path
    ):
        raise ValueError("autonomous_driving_batch_replay_reference_mismatch")
    evidence = report.get("repeat_evidence")
    checkpoint_dir = path.parent / "checkpoints" / candidate_file_stem(candidate_id)
    expected_checkpoint_names = [
        f"repeat_{repeat_index:03d}.json" for repeat_index in range(2, repeats + 1)
    ]
    if (
        not isinstance(evidence, list)
        or len(evidence) != repeats
        or not _portable_artifact_reference_matches(evidence[0], calibration_path)
        or [Path(str(value or "")).name for value in evidence[1:]] != expected_checkpoint_names
        or any(
            not _portable_artifact_reference_matches(value, checkpoint_dir / str(value or ""))
            or not (checkpoint_dir / Path(str(value or "")).name).is_file()
            for value in evidence[1:]
        )
    ):
        raise ValueError("autonomous_driving_batch_replay_evidence_missing")
    evidence_paths = [
        calibration_path,
        *(checkpoint_dir / Path(str(value)).name for value in evidence[1:]),
    ]
    calibration_reports = [
        validate_calibration_report(
            evidence_paths[0],
            candidate_id=candidate_id,
            ego_actor_id=ego_actor_id,
            bundle=bundle,
            seed=seed,
            ticks=ticks,
            difficulty_level=difficulty_level,
            scenario_artifact=scenario_artifact,
            expected_current_binding=expected_current_binding,
        )
    ]
    current_binding = dict(calibration_reports[0].get("evidence_binding") or {})
    calibration_reports.extend(
        validate_calibration_report(
            evidence_path,
            candidate_id=candidate_id,
            ego_actor_id=ego_actor_id,
            bundle=bundle,
            seed=seed,
            ticks=ticks,
            difficulty_level=difficulty_level,
            scenario_artifact=scenario_artifact,
            expected_current_binding=current_binding,
        )
        for evidence_path in evidence_paths[1:]
    )
    for repeat_index, calibration_report in enumerate(calibration_reports):
        checkpoint_digests = {
            str(leg.get("leg") or ""): str(leg.get("semantic_digest") or "")
            for leg in calibration_report.get("legs") or []
            if isinstance(leg, dict)
        }
        if any(
            checkpoint_digests.get(leg_name) != str(digests[leg_name][repeat_index])
            for leg_name in LEG_NAMES
        ):
            raise ValueError("autonomous_driving_batch_replay_checkpoint_digest_mismatch")
    calibration = calibration_reports[0]
    expected_binding = {
        **current_binding,
        "calibration_report_digest_sha256": str(calibration.get("report_digest_sha256") or ""),
    }
    if report.get("evidence_binding") != expected_binding:
        raise ValueError("autonomous_driving_batch_replay_evidence_binding_mismatch")
    expected = str(report.get("replay_digest_sha256") or "")
    if not expected or expected != _digest_without(report, "replay_digest_sha256"):
        raise ValueError("autonomous_driving_batch_replay_digest_mismatch")
    return report


def _atomic_replace_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def run_command(command: list[str], *, timeout: float) -> None:
    completed = subprocess.run(  # nosec B603 - fixed argv, no shell
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no child output"
        raise RuntimeError(f"autonomous_driving_batch_child_failed: {detail}")


def _calibration_command(
    row: dict[str, Any], output: Path, *, seed: int, ticks: int, difficulty_level: str
) -> list[str]:
    if row.get("scenario_yaml"):
        return [
            sys.executable,
            str(REPO_ROOT / "scripts/run_autonomous_driving_legs.py"),
            "--scenario-yaml",
            str(row["scenario_yaml"]),
            "--legs",
            "shield_only",
            "rule_tactical",
            "oracle_offline",
            "--output",
            str(output),
        ]
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/run_autonomous_driving_legs.py"),
        "--bundle",
        str(row["bundle"]),
        "--candidate-id",
        str(row["candidate_id"]),
        "--ego",
        str(row["ego_actor_id"]),
        "--seed",
        str(seed),
        "--ticks",
        str(ticks),
        "--difficulty-level",
        difficulty_level,
        "--legs",
        "shield_only",
        "rule_tactical",
        "oracle_offline",
        "--output",
        str(output),
    ]


def _replay_command(
    row: dict[str, Any],
    calibration_path: Path,
    checkpoint_dir: Path,
    output: Path,
    *,
    seed: int,
    ticks: int,
    repeats: int,
    difficulty_level: str,
    resume: bool,
) -> list[str]:
    if row.get("scenario_yaml"):
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/replay_autonomous_driving_calibration.py"),
            "--scenario-yaml",
            str(row["scenario_yaml"]),
            "--repeats",
            str(repeats),
            "--reference-calibration",
            str(calibration_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--output",
            str(output),
        ]
    else:
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/replay_autonomous_driving_calibration.py"),
            "--bundle",
            str(row["bundle"]),
            "--candidate-id",
            str(row["candidate_id"]),
            "--seed",
            str(seed),
            "--ticks",
            str(ticks),
            "--repeats",
            str(repeats),
            "--reference-calibration",
            str(calibration_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--difficulty-level",
            difficulty_level,
            "--output",
            str(output),
        ]
    if resume:
        command.append("--resume")
    return command


def run_batch(
    *,
    catalog_path: Path,
    calibration_dir: Path,
    replay_dir: Path,
    output: Path,
    seed: int = 42,
    ticks: int = 14,
    repeats: int = 3,
    difficulty_level: str = "extreme",
    suite_dir: Path | None = None,
    resume: bool = False,
    candidate_ids: set[str] | None = None,
) -> dict[str, Any]:
    if ticks < 1 or repeats < 2:
        raise ValueError("autonomous_driving_batch_parameters_invalid")
    rows = load_catalog_rows(catalog_path)
    if candidate_ids is not None:
        known = {str(row["candidate_id"]) for row in rows}
        unknown = candidate_ids - known
        if unknown:
            raise ValueError("autonomous_driving_batch_candidate_filter_unknown")
        rows = [row for row in rows if row["candidate_id"] in candidate_ids]
    if suite_dir is not None:
        rows = bind_suite_scenarios(rows, suite_dir, difficulty_level=difficulty_level)
        if any(
            row["seed"] != seed
            or row["ticks"] != ticks
            or row["difficulty_level"] != difficulty_level
            for row in rows
        ):
            raise ValueError("autonomous_driving_batch_suite_parameters_mismatch")
    if not rows:
        raise ValueError("autonomous_driving_batch_selection_empty")
    if output.exists() and not resume:
        raise FileExistsError("autonomous_driving_batch_output_exists")

    calibration_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    reused = 0
    for row in rows:
        candidate_id = str(row["candidate_id"])
        row_seed = int(row.get("seed", seed))
        row_ticks = int(row.get("ticks", ticks))
        row_difficulty = str(row.get("difficulty_level", difficulty_level))
        scenario_artifact = row.get("scenario_artifact")
        stem = candidate_file_stem(candidate_id)
        calibration_path = calibration_dir / f"{stem}.json"
        replay_path = replay_dir / f"{stem}.json"
        replay_checkpoint_dir = replay_dir / "checkpoints" / stem
        execution = "executed"
        try:
            calibration_reused = calibration_path.exists()
            if calibration_path.exists():
                if not resume:
                    raise FileExistsError("autonomous_driving_batch_calibration_output_exists")
                calibration_report = validate_calibration_report(
                    calibration_path,
                    candidate_id=candidate_id,
                    ego_actor_id=str(row["ego_actor_id"]),
                    bundle=Path(row["bundle"]),
                    seed=row_seed,
                    ticks=row_ticks,
                    difficulty_level=row_difficulty,
                    scenario_artifact=scenario_artifact,
                )
            else:
                run_command(
                    _calibration_command(
                        row,
                        calibration_path,
                        seed=row_seed,
                        ticks=row_ticks,
                        difficulty_level=row_difficulty,
                    ),
                    timeout=max(300.0, ticks * 60.0),
                )
                calibration_report = validate_calibration_report(
                    calibration_path,
                    candidate_id=candidate_id,
                    ego_actor_id=str(row["ego_actor_id"]),
                    bundle=Path(row["bundle"]),
                    seed=row_seed,
                    ticks=row_ticks,
                    difficulty_level=row_difficulty,
                    scenario_artifact=scenario_artifact,
                )

            if replay_path.exists():
                replay_reused = True
                if not resume:
                    raise FileExistsError("autonomous_driving_batch_replay_output_exists")
                validate_replay_report(
                    replay_path,
                    candidate_id=candidate_id,
                    ego_actor_id=str(row["ego_actor_id"]),
                    bundle=Path(row["bundle"]),
                    calibration_path=calibration_path,
                    seed=row_seed,
                    ticks=row_ticks,
                    repeats=repeats,
                    difficulty_level=row_difficulty,
                    scenario_artifact=scenario_artifact,
                    expected_current_binding=dict(calibration_report.get("evidence_binding") or {}),
                )
                if calibration_reused:
                    execution = "reused"
                    reused += 1
            else:
                replay_reused = False
                run_command(
                    _replay_command(
                        row,
                        calibration_path,
                        replay_checkpoint_dir,
                        replay_path,
                        seed=row_seed,
                        ticks=row_ticks,
                        repeats=repeats,
                        difficulty_level=row_difficulty,
                        resume=resume,
                    ),
                    timeout=max(900.0, repeats * ticks * 60.0),
                )
                validate_replay_report(
                    replay_path,
                    candidate_id=candidate_id,
                    ego_actor_id=str(row["ego_actor_id"]),
                    bundle=Path(row["bundle"]),
                    calibration_path=calibration_path,
                    seed=row_seed,
                    ticks=row_ticks,
                    repeats=repeats,
                    difficulty_level=row_difficulty,
                    scenario_artifact=scenario_artifact,
                    expected_current_binding=dict(calibration_report.get("evidence_binding") or {}),
                )
            if calibration_reused != replay_reused:
                execution = "resumed_partial"
            results.append(
                {
                    "candidate_id": candidate_id,
                    "recording_id": row["recording_id"],
                    "hazard_kind": row["hazard_kind"],
                    "status": "verified",
                    "execution": execution,
                    "calibration_path": str(calibration_path.resolve()),
                    "replay_path": str(replay_path.resolve()),
                    "scenario_yaml": (
                        str(Path(row["scenario_yaml"]).resolve())
                        if row.get("scenario_yaml")
                        else None
                    ),
                }
            )
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            results.append(
                {
                    "candidate_id": candidate_id,
                    "recording_id": row["recording_id"],
                    "hazard_kind": row["hazard_kind"],
                    "status": "failed",
                    "reason": str(error),
                    "execution": execution,
                }
            )
        partial_report = _build_report(
            catalog_path=catalog_path,
            results=results,
            expected_count=len(rows),
            reused=reused,
            seed=seed,
            ticks=ticks,
            repeats=repeats,
            difficulty_level=difficulty_level,
        )
        _atomic_replace_json(output, partial_report)
    return partial_report


def _build_report(
    *,
    catalog_path: Path,
    results: list[dict[str, Any]],
    expected_count: int,
    reused: int,
    seed: int,
    ticks: int,
    repeats: int,
    difficulty_level: str,
) -> dict[str, Any]:
    completed = sum(row.get("status") == "verified" for row in results)
    failed = sum(row.get("status") == "failed" for row in results)
    report: dict[str, Any] = {
        "schema_version": "autonomous_driving_core_calibration_batch_v1",
        "status": "verified" if completed == expected_count and failed == 0 else "held",
        "formal_core_allowed": False,
        "catalog_path": str(catalog_path.resolve()),
        "parameters": {
            "seed": seed,
            "ticks": ticks,
            "repeats": repeats,
            "difficulty_level": difficulty_level,
            "execution": "serial_native_sumo_isolated_processes",
        },
        "summary": {
            "candidate_count": expected_count,
            "completed": completed,
            "failed": failed,
            "reused": reused,
        },
        "candidates": results,
    }
    report["report_digest_sha256"] = object_digest(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ticks", type=int, default=14)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--difficulty-level",
        choices=("basic", "medium", "high", "extreme"),
        default="extreme",
    )
    parser.add_argument("--candidate-id", action="append", dest="candidate_ids")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_batch(
            catalog_path=_resolve_repo_path(args.catalog),
            suite_dir=(_resolve_repo_path(args.suite_dir) if args.suite_dir is not None else None),
            calibration_dir=_resolve_repo_path(args.calibration_dir),
            replay_dir=_resolve_repo_path(args.replay_dir),
            output=_resolve_repo_path(args.output),
            seed=args.seed,
            ticks=args.ticks,
            repeats=args.repeats,
            difficulty_level=args.difficulty_level,
            resume=args.resume,
            candidate_ids=set(args.candidate_ids) if args.candidate_ids else None,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "failed", "reason": str(error)}, sort_keys=True))
        return 1
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, sort_keys=True))
    return 0 if report["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
