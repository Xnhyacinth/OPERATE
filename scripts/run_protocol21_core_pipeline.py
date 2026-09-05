#!/usr/bin/env python3
"""Run or print the fail-closed Protocol-2.1 Core evidence pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_admission import (  # noqa: E402
    QUALITY_CORE_V2_ADMISSION_PROFILE,
    resolve_protocol21_admission_profile,
)
from core.protocol21_evidence import (  # noqa: E402
    canonicalize_repo_owned_paths,
    extract_semantics,
    required_semantics,
    resolve_binding_path,
    verify_artifact_binding,
)
from evaluation import SCORING_VERSION  # noqa: E402
from runner import (  # noqa: E402
    EVALUATION_IMPLEMENTATION_FINGERPRINT,
    EVALUATION_PROTOCOL_VERSION,
)

DEFAULT_SOURCE = (
    REPO_ROOT / "release" / "operate_v0_61_0" / "protocol21_source_suite.json"
)
DEFAULT_RELEASE_DIR = (
    REPO_ROOT / ".hl" / "release_rebuild" / "operate_v0_61_0" / "replay_from_promoted_suite"
)
FINGERPRINT = EVALUATION_IMPLEMENTATION_FINGERPRINT
STAGE_ORDER = (
    "preflight",
    "behavioral",
    "source_consumption",
    "task_contracts",
    "complexity",
    "observed_reference_depth",
    "strategy_depth",
    "source_grounded",
    "agentic_contract",
    "materialize_core",
    "release_coverage",
    "readiness",
)
DEFAULT_STOP_AFTER = STAGE_ORDER[-1]


def _runtime_binding(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind required native runtime switches before any stage can run."""
    backend_kinds = {
        str(row.get("backend_kind") or "").strip().lower()
        for row in rows
        if isinstance(row, dict)
    }
    requires_real_sumo = "sumo" in backend_kinds
    traffic_real_enabled = os.environ.get("OPERATE_TRAFFIC_BACKEND_REAL") == "1"
    requires_real_autonomous_driving_sumo = "sumo_ego" in backend_kinds
    autonomous_driving_real_enabled = (
        os.environ.get("OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL") == "1"
    )
    if requires_real_sumo and not traffic_real_enabled:
        raise ValueError(
            "source suite contains backend_kind=sumo; set "
            "OPERATE_TRAFFIC_BACKEND_REAL=1 before Protocol-2.1 replay"
        )
    if requires_real_autonomous_driving_sumo and not autonomous_driving_real_enabled:
        raise ValueError(
            "source suite contains backend_kind=sumo_ego; set "
            "OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL=1 before Protocol-2.1 replay"
        )
    return {
        "requires_real_sumo": requires_real_sumo,
        "OPERATE_TRAFFIC_BACKEND_REAL": "1" if traffic_real_enabled else None,
        "requires_real_autonomous_driving_sumo": (
            requires_real_autonomous_driving_sumo
        ),
        "OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL": (
            "1" if autonomous_driving_real_enabled else None
        ),
    }


def _output_paths(release_dir: Path) -> dict[str, Path]:
    return {
        "preflight": release_dir / "protocol2_v21_working_set_preflight.json",
        "behavioral": release_dir / "behavioral_calibration_protocol2_v21.json",
        "source_consumption": release_dir / "source_consumption_protocol2_v21.json",
        "task_contracts": release_dir / "task_contracts_protocol2_v21.json",
        "complexity": release_dir / "complexity_protocol2_v21.json",
        "observed_reference_depth": release_dir
        / "observed_reference_depth_protocol2_v21.json",
        "strategy_depth": release_dir / "strategy_depth_protocol2_v21.json",
        "source_grounded": release_dir / "source_grounded_protocol2_v21.json",
        "agentic_contract": release_dir / "agentic_core_contract_protocol2_v21.json",
        "materialize_core": release_dir / "refined_core_selection_protocol2_v21.json",
        "release_coverage": release_dir / "release_coverage_protocol2_v21.json",
        "readiness": release_dir / "protocol2_v21_core_readiness.json",
        "pipeline_manifest": release_dir / "protocol2_v21_pipeline_manifest.json",
    }


def _stage(
    name: str,
    *argv: str | Path,
    expected_output: Path | None = None,
) -> dict[str, Any]:
    runtime_argv = [str(value) for value in argv]
    portable_argv = canonicalize_repo_owned_paths(runtime_argv, repo_root=REPO_ROOT)
    if portable_argv and Path(portable_argv[0]).is_absolute():
        portable_argv[0] = Path(portable_argv[0]).name
    return {
        "name": name,
        "argv": portable_argv,
        "runtime_argv": runtime_argv,
        "expected_output": (
            canonicalize_repo_owned_paths(str(expected_output), repo_root=REPO_ROOT)
            if expected_output
            else None
        ),
        "runtime_expected_output": (str(expected_output) if expected_output else None),
    }


def _bind_scenario_yaml_graph(
    rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Bind every planned scenario YAML without expanding implementation identity."""
    bindings: list[dict[str, str]] = []
    for row in rows:
        scenario_id = str(row.get("scenario_id") or "")
        raw_path = str(row.get("path") or "").strip()
        if not raw_path:
            raise ValueError(f"{scenario_id}: scenario YAML path missing")
        try:
            path = resolve_binding_path(raw_path, repo_root=REPO_ROOT)
        except ValueError as exc:
            raise ValueError(
                f"{scenario_id}: scenario YAML path invalid: {raw_path}"
            ) from exc
        if not path.is_file():
            raise ValueError(f"{scenario_id}: scenario YAML missing: {path}")
        bindings.append(
            {
                "scenario_id": scenario_id,
                "path": canonicalize_repo_owned_paths(str(path), repo_root=REPO_ROOT),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return bindings


def _verify_planned_source_graph(plan: dict[str, Any]) -> None:
    """Fail closed if the planned suite or any scenario YAML has drifted."""
    source_raw = plan.get("runtime_source_suite") or plan.get("source_suite")
    graph = plan.get("scenario_yaml_graph")
    if not source_raw or not isinstance(graph, list) or not graph:
        raise RuntimeError("planned source graph drift: binding missing")
    source_suite = Path(str(source_raw)).resolve()
    if not source_suite.is_file():
        raise RuntimeError("planned source graph drift: source suite missing")
    if hashlib.sha256(source_suite.read_bytes()).hexdigest() != plan.get(
        "source_suite_sha256"
    ):
        raise RuntimeError("planned source graph drift: source suite sha256 mismatch")
    for binding in graph:
        if not isinstance(binding, dict):
            raise RuntimeError(
                "planned source graph drift: invalid scenario YAML binding"
            )
        scenario_id = str(binding.get("scenario_id") or "")
        raw_path = str(binding.get("path") or "")
        expected_sha256 = str(binding.get("sha256") or "")
        if not scenario_id or not raw_path or not expected_sha256:
            raise RuntimeError(
                "planned source graph drift: invalid scenario YAML binding"
            )
        try:
            path = resolve_binding_path(raw_path, repo_root=REPO_ROOT)
        except ValueError as exc:
            raise RuntimeError(
                f"planned source graph drift: {scenario_id}: path invalid"
            ) from exc
        if not path.is_file():
            raise RuntimeError(
                f"planned source graph drift: {scenario_id}: YAML missing"
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError(
                f"planned source graph drift: {scenario_id}: YAML sha256 mismatch"
            )


def build_pipeline_plan(
    *,
    source_suite: Path,
    release_dir: Path,
    workers: int,
    sample_timeout_seconds: int,
    expected_count: int | None = None,
    max_replays: int = 64,
    max_replay_work_ticks: int = 32768,
    exact_max_calls: int = 6,
    exact_max_replays: int = 4096,
    per_action_cap: int = -1,
    stop_after: str = DEFAULT_STOP_AFTER,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if sample_timeout_seconds <= 0:
        raise ValueError("sample-timeout-seconds must be > 0")
    if stop_after not in STAGE_ORDER:
        raise ValueError(f"unknown stop-after stage: {stop_after}")
    source_suite = source_suite.resolve()
    release_dir = release_dir.resolve()
    source_suite_bytes = source_suite.read_bytes()
    payload = json.loads(source_suite_bytes)
    if (
        payload.get("status") != "working_set"
        or payload.get("leaderboard_eligible") is not False
    ):
        raise ValueError(
            "source suite must be a non-eligible completed protocol-2.1 working set"
        )
    admission_profile = resolve_protocol21_admission_profile(payload)
    complexity_mode = "full_minimality"
    if admission_profile == QUALITY_CORE_V2_ADMISSION_PROFILE:
        complexity_mode = "bounded_diagnostic"
        max_replays = min(max_replays, 1)
        exact_max_calls = 0
        exact_max_replays = 0
        per_action_cap = 0
    rows = payload.get("scenarios")
    if not isinstance(rows, list) or not rows:
        raise ValueError("source suite must contain a non-empty scenarios list")
    inferred_count = len(rows)
    if expected_count is not None and inferred_count != expected_count:
        raise ValueError(
            f"source suite must contain exactly {expected_count} scenarios"
        )
    expected_count = inferred_count
    identities = [
        (
            str(row.get("scenario_id") or ""),
            str(row.get("scenario_signature") or ""),
        )
        for row in rows
        if isinstance(row, dict)
    ]
    if (
        len(identities) != expected_count
        or any(
            not scenario_id or not signature for scenario_id, signature in identities
        )
        or len(set(identities)) != expected_count
    ):
        raise ValueError("source suite identities must be complete and unique")
    runtime_binding = _runtime_binding(rows)
    source_sha256 = hashlib.sha256(source_suite_bytes).hexdigest()
    scenario_yaml_graph = _bind_scenario_yaml_graph(rows)
    identity = implementation_identity()
    implementation_tree_sha256 = identity["implementation_tree_sha256"]
    core_release_pipeline_sha256 = identity["core_release_pipeline_sha256"]
    release_tooling_sha256 = identity["release_tooling_sha256"]
    outputs = _output_paths(release_dir)
    budget_key = (
        f"r{max_replays}-t{max_replay_work_ticks}-c{exact_max_calls}"
        f"-e{exact_max_replays}-a{per_action_cap}"
    )
    cache_root = (
        REPO_ROOT
        / ".audit-cache"
        / FINGERPRINT
        / core_release_pipeline_sha256
        / budget_key
    )
    python = Path(sys.executable)
    stages = [
        _stage(
            "preflight",
            python,
            REPO_ROOT / "scripts/preflight_protocol21_working_set.py",
            "--source-suite",
            source_suite,
            "--output",
            outputs["preflight"],
            "--expected-count",
            str(expected_count),
            "--require-source-consumption-adapters",
            "--require-formal-core-backends",
            "--exercise-source-adapters",
            expected_output=outputs["preflight"],
        ),
        _stage(
            "behavioral",
            python,
            REPO_ROOT / "scripts/calibrate_core_candidate.py",
            "--suite",
            source_suite,
            "--output",
            outputs["behavioral"],
            "--workers",
            str(workers),
            "--sample-timeout-seconds",
            str(sample_timeout_seconds),
            "--cache-dir",
            cache_root / "behavioral",
            expected_output=outputs["behavioral"],
        ),
        _stage(
            "source_consumption",
            python,
            REPO_ROOT / "scripts/audit_protocol21_source_consumption.py",
            "--suite",
            source_suite,
            "--behavioral",
            outputs["behavioral"],
            "--output",
            outputs["source_consumption"],
            expected_output=outputs["source_consumption"],
        ),
        _stage(
            "task_contracts",
            python,
            REPO_ROOT / "scripts/calibrate_task_contracts.py",
            "--suite",
            source_suite,
            "--output",
            outputs["task_contracts"],
            "--agent",
            "oracle_offline",
            "--fallback-agents",
            "greedy_heuristic",
            "--workers",
            str(workers),
            "--sample-timeout-seconds",
            str(sample_timeout_seconds),
            "--eligible-results",
            outputs["behavioral"],
            expected_output=outputs["task_contracts"],
        ),
        _stage(
            "complexity",
            python,
            REPO_ROOT / "scripts/calibrate_core_complexity.py",
            "--suite",
            source_suite,
            "--output",
            outputs["complexity"],
            "--agents",
            "oracle_offline",
            "greedy_heuristic",
            "wait_only",
            "--workers",
            str(workers),
            "--sample-timeout-seconds",
            str(sample_timeout_seconds),
            "--cache-dir",
            cache_root / "complexity",
            "--eligible-task-contracts",
            outputs["task_contracts"],
            "--max-replays",
            str(max_replays),
            "--max-replay-work-ticks",
            str(max_replay_work_ticks),
            "--exact-max-calls",
            str(exact_max_calls),
            "--exact-max-replays",
            str(exact_max_replays),
            "--per-action-cap",
            str(per_action_cap),
            expected_output=outputs["complexity"],
        ),
        _stage(
            "observed_reference_depth",
            python,
            REPO_ROOT / "scripts/audit_observed_reference_depth.py",
            "--behavioral",
            outputs["behavioral"],
            "--task-contracts",
            outputs["task_contracts"],
            "--output",
            outputs["observed_reference_depth"],
            expected_output=outputs["observed_reference_depth"],
        ),
        _stage(
            "strategy_depth",
            python,
            REPO_ROOT / "scripts/audit_strategy_depth_calibration.py",
            "--input",
            outputs["complexity"],
            "--suite",
            source_suite,
            "--output",
            outputs["strategy_depth"],
            expected_output=outputs["strategy_depth"],
        ),
        _stage(
            "source_grounded",
            python,
            REPO_ROOT / "scripts/audit_source_grounded_pipeline.py",
            "--input",
            source_suite,
            "--behavioral",
            outputs["behavioral"],
            "--source-consumption",
            outputs["source_consumption"],
            "--task-contracts",
            outputs["task_contracts"],
            "--complexity",
            outputs["complexity"],
            "--strategy-depth",
            outputs["strategy_depth"],
            "--require-protocol21-evidence",
            "--output",
            outputs["source_grounded"],
            expected_output=outputs["source_grounded"],
        ),
        _stage(
            "agentic_contract",
            python,
            REPO_ROOT / "scripts/audit_protocol21_core_contract.py",
            "--source-suite",
            source_suite,
            "--behavioral",
            outputs["behavioral"],
            "--task-contracts",
            outputs["task_contracts"],
            "--complexity",
            outputs["complexity"],
            "--observed-depth",
            outputs["observed_reference_depth"],
            "--strategy-depth",
            outputs["strategy_depth"],
            "--source-grounded",
            outputs["source_grounded"],
            "--source-consumption",
            outputs["source_consumption"],
            "--output",
            outputs["agentic_contract"],
            expected_output=outputs["agentic_contract"],
        ),
        _stage(
            "materialize_core",
            python,
            REPO_ROOT / "scripts/materialize_protocol2_core.py",
            "--source",
            source_suite,
            "--behavioral",
            outputs["behavioral"],
            "--tasks",
            outputs["task_contracts"],
            "--observed-depth",
            outputs["observed_reference_depth"],
            "--depth",
            outputs["strategy_depth"],
            "--source-gate",
            outputs["source_grounded"],
            "--agentic-contract",
            outputs["agentic_contract"],
            "--require-protocol21-gates",
            "--output",
            outputs["materialize_core"],
            expected_output=outputs["materialize_core"],
        ),
        _stage(
            "release_coverage",
            python,
            REPO_ROOT / "scripts/audit_protocol21_release_coverage.py",
            "--core",
            outputs["materialize_core"],
            "--output",
            outputs["release_coverage"],
            expected_output=outputs["release_coverage"],
        ),
        _stage(
            "readiness",
            python,
            REPO_ROOT / "scripts/build_protocol21_core_readiness.py",
            "--core",
            outputs["materialize_core"],
            "--source-suite",
            source_suite,
            "--preflight",
            outputs["preflight"],
            "--behavioral",
            outputs["behavioral"],
            "--source-consumption",
            outputs["source_consumption"],
            "--task-contracts",
            outputs["task_contracts"],
            "--complexity",
            outputs["complexity"],
            "--observed-depth",
            outputs["observed_reference_depth"],
            "--strategy-depth",
            outputs["strategy_depth"],
            "--source-gate",
            outputs["source_grounded"],
            "--agentic-contract",
            outputs["agentic_contract"],
            "--release-coverage",
            outputs["release_coverage"],
            "--output",
            outputs["readiness"],
            expected_output=outputs["readiness"],
        ),
    ]
    for stage in stages:
        stage["core_release_pipeline_sha256"] = core_release_pipeline_sha256
    stages = stages[: STAGE_ORDER.index(stop_after) + 1]
    return {
        "source_suite": canonicalize_repo_owned_paths(
            str(source_suite), repo_root=REPO_ROOT
        ),
        "runtime_source_suite": str(source_suite),
        "source_suite_sha256": source_sha256,
        "scenario_yaml_graph": scenario_yaml_graph,
        "n_source_scenarios": expected_count,
        "evaluation_implementation_fingerprint": FINGERPRINT,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "scoring_version": SCORING_VERSION,
        "admission_profile": admission_profile,
        "complexity_mode": complexity_mode,
        "execution_profile": (
            "full_release" if stop_after == DEFAULT_STOP_AFTER else "candidate_replay"
        ),
        "stop_after": stop_after,
        "implementation_tree_sha256": implementation_tree_sha256,
        "core_release_pipeline_sha256": core_release_pipeline_sha256,
        "release_tooling_sha256": release_tooling_sha256,
        "runtime_binding": runtime_binding,
        "pipeline_manifest": canonicalize_repo_owned_paths(
            str(outputs["pipeline_manifest"]), repo_root=REPO_ROOT
        ),
        "runtime_pipeline_manifest": str(outputs["pipeline_manifest"]),
        "stages": stages,
    }


def _validate_stage_output(stage: dict[str, Any]) -> None:
    raw = stage.get("runtime_expected_output") or stage.get("expected_output")
    if not raw:
        return
    path = Path(raw)
    if not path.is_file():
        raise RuntimeError(f"{stage['name']}: output missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    current_identity = implementation_identity()
    current_tree = current_identity["implementation_tree_sha256"]
    current_core_pipeline = current_identity["core_release_pipeline_sha256"]
    planned_tree = str(stage.get("plan_implementation_tree_sha256") or "")
    if planned_tree and planned_tree != current_tree:
        raise RuntimeError(f"{stage['name']}: plan implementation tree drift")
    planned_core_pipeline = str(
        stage.get("core_release_pipeline_sha256")
        or stage.get("plan_core_release_pipeline_sha256")
        or ""
    )
    if planned_core_pipeline and planned_core_pipeline != current_core_pipeline:
        raise RuntimeError(f"{stage['name']}: core release pipeline drift")
    if (
        planned_core_pipeline
        and payload.get("core_release_pipeline_sha256") != planned_core_pipeline
    ):
        raise RuntimeError(f"{stage['name']}: artifact core release pipeline mismatch")
    if stage["name"] == "preflight":
        if payload.get("status") != "passed" or payload.get("n_fatal") != 0:
            raise RuntimeError("preflight: fatal blockers remain")
        if payload.get("implementation_tree_sha256") != current_tree:
            raise RuntimeError("preflight: implementation tree drift")
    if stage["name"] == "materialize_core":
        if payload.get("status") != "protocol21_core_candidate":
            raise RuntimeError("materialize_core: invalid status")
        if payload.get("implementation_tree_sha256") != current_tree:
            raise RuntimeError("materialize_core: implementation tree drift")
    if stage["name"] == "readiness":
        if payload.get("status") not in {"formal_evaluation_ready", "blocked"}:
            raise RuntimeError("readiness: invalid status")
        if not isinstance(payload.get("formal_evaluation_ready"), bool):
            raise RuntimeError("readiness: formal_evaluation_ready must be boolean")
        if payload.get("implementation_tree_sha256") != current_tree:
            raise RuntimeError("readiness: implementation tree drift")
    if (
        stage["name"] != "preflight"
        and extract_semantics(payload) != required_semantics()
    ):
        raise RuntimeError(f"{stage['name']}: stale evaluation semantics")
    if payload.get("implementation_tree_sha256") != current_tree:
        raise RuntimeError(f"{stage['name']}: implementation tree drift")
    if stage["name"] not in {"preflight", "materialize_core", "readiness"} and not (
        payload.get("status") == "complete" or payload.get("complete") is True
    ):
        raise RuntimeError(f"{stage['name']}: report is incomplete")
    expected = payload.get("n_expected")
    completed = payload.get("n_completed")
    if (
        expected is not None
        and completed is not None
        and int(expected) != int(completed)
    ):
        raise RuntimeError(f"{stage['name']}: expected/completed mismatch")
    if stage["name"] == "behavioral":
        status_counts = payload.get("status_counts")
        if (
            type(completed) is not int
            or not isinstance(status_counts, dict)
            or any(
                not isinstance(status, str)
                or not status
                or type(count) is not int
                or count < 0
                for status, count in status_counts.items()
            )
            or sum(status_counts.values()) != completed
            or status_counts.get("passed") != completed
        ):
            raise RuntimeError("behavioral: non-passing rows remain")
    if stage["name"] == "preflight":
        argv = [
            str(value)
            for value in (stage.get("runtime_argv") or stage.get("argv") or [])
        ]
        try:
            expected_index = argv.index("--expected-count") + 1
            planned_expected = int(argv[expected_index])
        except (ValueError, IndexError, TypeError):
            planned_expected = None
        if planned_expected is not None and expected != planned_expected:
            raise RuntimeError("preflight: expected count binding mismatch")
        try:
            source_index = argv.index("--source-suite") + 1
            planned_source = Path(argv[source_index]).resolve()
        except (ValueError, IndexError, TypeError):
            planned_source = None
        source_binding = (payload.get("input_bindings") or {}).get("source_suite")
        try:
            bound_source = (
                resolve_binding_path(str(source_binding.get("path") or ""))
                if isinstance(source_binding, dict)
                else None
            )
        except ValueError:
            bound_source = None
        if planned_source is not None and bound_source != planned_source:
            raise RuntimeError("preflight: source suite binding mismatch")
    binding_maps = [
        payload.get("input_bindings") or {},
        payload.get("artifact_bindings") or {},
    ]
    for binding_map in binding_maps:
        if not isinstance(binding_map, dict):
            raise RuntimeError(f"{stage['name']}: invalid input binding map")
        for binding in binding_map.values():
            if not isinstance(binding, dict) or not binding.get("path"):
                raise RuntimeError(f"{stage['name']}: invalid input binding")
            try:
                binding_path = resolve_binding_path(str(binding["path"]))
                errors = verify_artifact_binding(
                    binding,
                    binding_path,
                    implementation_tree_sha256=current_tree,
                )
            except ValueError:
                errors = ["artifact_path_mismatch"]
            if errors:
                raise RuntimeError(f"{stage['name']}: invalid input binding: {errors}")


def _print_plan(plan: dict[str, Any]) -> None:
    print(f"SOURCE_SUITE_SHA256={plan['source_suite_sha256']}")
    print("STAGE_ORDER=" + " -> ".join(stage["name"] for stage in plan["stages"]))
    for index, stage in enumerate(plan["stages"], start=1):
        print(f"[{index:02d}] {stage['name']}")
        print("  " + " ".join(stage.get("runtime_argv") or stage["argv"]))


def _write_pipeline_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest = canonicalize_repo_owned_paths(manifest, repo_root=REPO_ROOT)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _bind_stage_artifact(
    path: Path,
    core_release_pipeline_sha256: str,
) -> None:
    """Bind an orchestrated Core stage artifact before downstream use."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"stage artifact must be an object: {path}")
    existing = payload.get("core_release_pipeline_sha256")
    if existing not in {None, core_release_pipeline_sha256}:
        raise RuntimeError(f"stage artifact core release pipeline mismatch: {path}")
    payload["core_release_pipeline_sha256"] = core_release_pipeline_sha256
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_resume_manifest(
    path: Path,
    *,
    source_suite: str,
    source_suite_sha256: str,
    scenario_yaml_graph: list[dict[str, str]],
    implementation_tree_sha256: str,
    core_release_pipeline_sha256: str,
    runtime_binding: dict[str, Any],
) -> dict[str, Any]:
    """Load a compatible stage checkpoint, failing closed on suite/code drift."""
    if not path.is_file():
        raise RuntimeError(f"resume manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_suite") != source_suite:
        raise RuntimeError("resume manifest source suite path mismatch")
    if payload.get("source_suite_sha256") != source_suite_sha256:
        raise RuntimeError("resume manifest source suite mismatch")
    if payload.get("scenario_yaml_graph") != scenario_yaml_graph:
        raise RuntimeError("resume manifest scenario YAML graph mismatch")
    if payload.get("implementation_tree_sha256") != implementation_tree_sha256:
        raise RuntimeError("resume manifest implementation tree mismatch")
    if payload.get("core_release_pipeline_sha256") != core_release_pipeline_sha256:
        raise RuntimeError("resume manifest core release pipeline mismatch")
    # Publisher and maintenance tooling is provenance, not stage compatibility.
    # Runtime, pipeline, source and output identities remain checked separately.
    if payload.get("runtime_binding") != runtime_binding:
        raise RuntimeError("resume manifest runtime binding mismatch")
    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise RuntimeError("resume manifest stages must be a list")
    seen_names: set[str] = set()
    for entry in stages:
        if not isinstance(entry, dict):
            raise RuntimeError("resume manifest stage entry must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("resume manifest stage name missing")
        if name in seen_names:
            raise RuntimeError(f"resume manifest stage duplicated: {name}")
        if entry.get("core_release_pipeline_sha256") != core_release_pipeline_sha256:
            raise RuntimeError(
                f"resume manifest stage core release pipeline mismatch: {name}"
            )
        seen_names.add(name)
    return payload


def _stage_return_code_is_reusable(stage: dict[str, Any], return_code: Any) -> bool:
    if not isinstance(return_code, int):
        return False
    if stage["name"] == "readiness":
        return return_code in {0, 4}
    return return_code == 0


def _resume_stage_is_reusable(
    stage: dict[str, Any],
    prior_stage: dict[str, Any],
) -> bool:
    """Return whether an exact, currently-valid stage checkpoint may be reused."""
    if prior_stage.get("argv") != stage["argv"]:
        return False
    if prior_stage.get("core_release_pipeline_sha256") != stage.get(
        "core_release_pipeline_sha256"
    ):
        return False
    if not _stage_return_code_is_reusable(stage, prior_stage.get("return_code")):
        return False
    expected_output = stage.get("runtime_expected_output") or stage.get(
        "expected_output"
    )
    if expected_output:
        output_path = Path(str(expected_output))
        if not output_path.is_file():
            return False
        expected_sha256 = prior_stage.get("output_sha256")
        if not isinstance(expected_sha256, str) or not expected_sha256:
            return False
        actual_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            return False
    try:
        _validate_stage_output(stage)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-suite", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sample-timeout-seconds", type=int, default=900)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--max-replays", type=int, default=64)
    parser.add_argument("--max-replay-work-ticks", type=int, default=32768)
    parser.add_argument("--exact-max-calls", type=int, default=6)
    parser.add_argument("--exact-max-replays", type=int, default=4096)
    parser.add_argument("--per-action-cap", type=int, default=-1)
    parser.add_argument(
        "--stop-after",
        choices=STAGE_ORDER,
        default=DEFAULT_STOP_AFTER,
        help=(
            "stop after a verified stage; use materialize_core for candidate "
            "admission replay without release composition/readiness gates"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse only contiguous, hash-verified completed stages from the "
            "same source suite and implementation tree"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        plan = build_pipeline_plan(
            source_suite=args.source_suite,
            release_dir=args.release_dir,
            workers=args.workers,
            sample_timeout_seconds=args.sample_timeout_seconds,
            expected_count=args.expected_count,
            max_replays=args.max_replays,
            max_replay_work_ticks=args.max_replay_work_ticks,
            exact_max_calls=args.exact_max_calls,
            exact_max_replays=args.exact_max_replays,
            per_action_cap=args.per_action_cap,
            stop_after=args.stop_after,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print_plan(plan)
    if not args.execute:
        print("NO_COMMANDS_EXECUTED=true")
        return 0
    pipeline_manifest_path = Path(
        plan.get("runtime_pipeline_manifest") or plan["pipeline_manifest"]
    )
    if not args.resume:
        release_dir = args.release_dir.resolve()
        try:
            if release_dir.exists() and (
                not release_dir.is_dir() or any(release_dir.iterdir())
            ):
                raise RuntimeError(
                    "fresh run release directory is not empty; use --resume "
                    "or choose a new --release-dir"
                )
        except (OSError, RuntimeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3
    prior_stages: dict[str, dict[str, Any]] = {}
    prior_manifest_sha256: str | None = None
    prior_release_tooling_sha256: str | None = None
    if args.resume:
        try:
            prior_manifest = _load_resume_manifest(
                pipeline_manifest_path,
                source_suite=plan["source_suite"],
                source_suite_sha256=plan["source_suite_sha256"],
                scenario_yaml_graph=plan["scenario_yaml_graph"],
                implementation_tree_sha256=plan["implementation_tree_sha256"],
                core_release_pipeline_sha256=plan["core_release_pipeline_sha256"],
                runtime_binding=plan["runtime_binding"],
            )
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3
        prior_manifest_sha256 = hashlib.sha256(
            pipeline_manifest_path.read_bytes()
        ).hexdigest()
        prior_release_tooling_sha256 = prior_manifest.get("release_tooling_sha256")
        prior_stages = {str(entry["name"]): entry for entry in prior_manifest["stages"]}
        print(f"RESUME_MANIFEST={pipeline_manifest_path}")
    manifest = {
        "schema_version": "1.0",
        "status": "running",
        "source_suite": plan["source_suite"],
        "source_suite_sha256": plan["source_suite_sha256"],
        "scenario_yaml_graph": plan["scenario_yaml_graph"],
        "implementation_tree_sha256": plan["implementation_tree_sha256"],
        "core_release_pipeline_sha256": plan["core_release_pipeline_sha256"],
        "release_tooling_sha256": plan.get("release_tooling_sha256"),
        "runtime_binding": plan["runtime_binding"],
        "execution_profile": plan.get("execution_profile", "full_release"),
        "stop_after": plan.get("stop_after", DEFAULT_STOP_AFTER),
        "resume": {
            "enabled": bool(args.resume),
            "prior_manifest_sha256": prior_manifest_sha256,
            "prior_release_tooling_sha256": prior_release_tooling_sha256,
        },
        "stages": [],
    }
    may_reuse_stage = bool(args.resume)
    for stage in plan["stages"]:
        if not stage["argv"]:
            continue
        try:
            _verify_planned_source_graph(plan)
        except (OSError, RuntimeError) as exc:
            manifest["status"] = "blocked"
            manifest["formal_run_blockers"] = ["source_input_drift"]
            _write_pipeline_manifest(pipeline_manifest_path, manifest)
            print(f"ERROR: {stage['name']}: {exc}", file=sys.stderr)
            return 3
        identity_start = implementation_identity()
        if (
            identity_start["implementation_tree_sha256"]
            != plan["implementation_tree_sha256"]
            or identity_start["core_release_pipeline_sha256"]
            != plan["core_release_pipeline_sha256"]
        ):
            manifest["status"] = "blocked"
            manifest["formal_run_blockers"] = ["core_release_pipeline_drift"]
            _write_pipeline_manifest(pipeline_manifest_path, manifest)
            print(
                f"ERROR: {stage['name']}: core release pipeline drift",
                file=sys.stderr,
            )
            return 3
        expected_output = (
            Path(stage.get("runtime_expected_output") or stage["expected_output"])
            if stage.get("runtime_expected_output") or stage.get("expected_output")
            else None
        )
        prior_stage = prior_stages.get(str(stage["name"]))
        reused = bool(
            may_reuse_stage
            and prior_stage is not None
            and _resume_stage_is_reusable(stage, prior_stage)
        )
        if reused:
            stage_record = dict(prior_stage)
            stage_record["reused"] = True
            stage_record["resumed_at"] = datetime.now(UTC).isoformat()
            manifest["stages"].append(stage_record)
            _write_pipeline_manifest(pipeline_manifest_path, manifest)
            print(f"RESUMED_STAGE={stage['name']}")
            return_code = int(stage_record["return_code"])
        else:
            may_reuse_stage = False
            started = datetime.now(UTC).isoformat()
            completed = subprocess.run(
                stage.get("runtime_argv") or stage["argv"],
                cwd=REPO_ROOT,
                check=False,
            )
            finished = datetime.now(UTC).isoformat()
            return_code = completed.returncode
            identity_end = implementation_identity()
            drifted = (
                identity_end["implementation_tree_sha256"]
                != plan["implementation_tree_sha256"]
                or identity_end["core_release_pipeline_sha256"]
                != plan["core_release_pipeline_sha256"]
            )
            source_drift_error: str | None = None
            try:
                _verify_planned_source_graph(plan)
            except (OSError, RuntimeError) as exc:
                source_drift_error = str(exc)
            artifact_error: str | None = None
            if (
                not drifted
                and source_drift_error is None
                and _stage_return_code_is_reusable(stage, return_code)
                and expected_output is not None
            ):
                try:
                    _bind_stage_artifact(
                        expected_output,
                        plan["core_release_pipeline_sha256"],
                    )
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    json.JSONDecodeError,
                ) as exc:
                    artifact_error = str(exc)
            stage_record = {
                "name": stage["name"],
                "argv": stage["argv"],
                "started_at": started,
                "finished_at": finished,
                "return_code": return_code,
                "output_sha256": (
                    hashlib.sha256(expected_output.read_bytes()).hexdigest()
                    if expected_output is not None and expected_output.is_file()
                    else None
                ),
                "implementation_tree_sha256": plan["implementation_tree_sha256"],
                "implementation_tree_sha256_start": identity_start[
                    "implementation_tree_sha256"
                ],
                "implementation_tree_sha256_end": identity_end[
                    "implementation_tree_sha256"
                ],
                "core_release_pipeline_sha256": plan["core_release_pipeline_sha256"],
                "core_release_pipeline_sha256_start": identity_start[
                    "core_release_pipeline_sha256"
                ],
                "core_release_pipeline_sha256_end": identity_end[
                    "core_release_pipeline_sha256"
                ],
                "reused": False,
            }
            manifest["stages"].append(stage_record)
            if drifted or source_drift_error is not None or artifact_error is not None:
                manifest["status"] = "blocked"
                if drifted:
                    manifest["formal_run_blockers"] = ["core_release_pipeline_drift"]
                elif source_drift_error is not None:
                    manifest["formal_run_blockers"] = ["source_input_drift"]
                else:
                    manifest["formal_run_blockers"] = ["stage_artifact_binding_failed"]
            _write_pipeline_manifest(pipeline_manifest_path, manifest)
            if drifted:
                print(
                    f"ERROR: {stage['name']}: core release pipeline drift",
                    file=sys.stderr,
                )
                return 3
            if source_drift_error is not None:
                print(
                    f"ERROR: {stage['name']}: {source_drift_error}",
                    file=sys.stderr,
                )
                return 3
            if artifact_error is not None:
                print(
                    f"ERROR: {stage['name']}: {artifact_error}",
                    file=sys.stderr,
                )
                return 3
        try:
            _verify_planned_source_graph(plan)
        except (OSError, RuntimeError) as exc:
            manifest["status"] = "blocked"
            manifest["formal_run_blockers"] = ["source_input_drift"]
            _write_pipeline_manifest(pipeline_manifest_path, manifest)
            print(f"ERROR: {stage['name']}: {exc}", file=sys.stderr)
            return 3
        if return_code != 0 and not (stage["name"] == "readiness" and return_code == 4):
            print(
                f"ERROR: {stage['name']} failed with {return_code}",
                file=sys.stderr,
            )
            return return_code
        try:
            validation_stage = dict(stage)
            validation_stage["plan_implementation_tree_sha256"] = plan[
                "implementation_tree_sha256"
            ]
            validation_stage["plan_core_release_pipeline_sha256"] = plan[
                "core_release_pipeline_sha256"
            ]
            _validate_stage_output(validation_stage)
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            manifest["status"] = "blocked"
            manifest["formal_run_blockers"] = [
                "core_release_pipeline_drift"
                if "pipeline drift" in str(exc)
                else "stage_validation_failed"
            ]
            _write_pipeline_manifest(pipeline_manifest_path, manifest)
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3
        if stage["name"] == "readiness":
            readiness = json.loads(expected_output.read_text(encoding="utf-8"))
            if return_code == 4:
                manifest["status"] = "blocked"
                _write_pipeline_manifest(pipeline_manifest_path, manifest)
                print("PIPELINE_STATUS=blocked")
                print(
                    "FORMAL_RUN_BLOCKERS="
                    + json.dumps(readiness.get("formal_run_blockers") or [])
                )
                return 4
            if readiness.get("formal_evaluation_ready") is not True:
                manifest["status"] = "blocked"
                manifest["formal_run_blockers"] = list(
                    readiness.get("formal_run_blockers") or []
                )
                _write_pipeline_manifest(pipeline_manifest_path, manifest)
                print("PIPELINE_STATUS=blocked")
                return 4
    try:
        _verify_planned_source_graph(plan)
    except (OSError, RuntimeError) as exc:
        manifest["status"] = "blocked"
        manifest["formal_run_blockers"] = ["source_input_drift"]
        _write_pipeline_manifest(pipeline_manifest_path, manifest)
        print(f"ERROR: final {exc}", file=sys.stderr)
        return 3
    final_identity = implementation_identity()
    if (
        final_identity["implementation_tree_sha256"]
        != plan["implementation_tree_sha256"]
        or final_identity["core_release_pipeline_sha256"]
        != plan["core_release_pipeline_sha256"]
    ):
        manifest["status"] = "blocked"
        manifest["formal_run_blockers"] = ["core_release_pipeline_drift"]
        _write_pipeline_manifest(pipeline_manifest_path, manifest)
        print("ERROR: final core release pipeline drift", file=sys.stderr)
        return 3
    if plan.get("execution_profile") == "candidate_replay":
        terminal_stage = manifest["stages"][-1]
        manifest["status"] = "candidate_replay_complete"
        manifest["completed_stage"] = terminal_stage["name"]
        manifest["terminal_stage_artifact"] = {
            "path": next(
                (
                    stage.get("expected_output")
                    for stage in plan["stages"]
                    if stage["name"] == terminal_stage["name"]
                ),
                None,
            ),
            "sha256": terminal_stage.get("output_sha256"),
        }
        _write_pipeline_manifest(pipeline_manifest_path, manifest)
        print("PIPELINE_STATUS=candidate_replay_complete")
        print(f"REPLAY_LEDGER={pipeline_manifest_path}")
        return 0
    manifest["status"] = "formal_evaluation_ready"
    _write_pipeline_manifest(pipeline_manifest_path, manifest)
    print("PIPELINE_STATUS=formal_evaluation_ready")
    print(
        ".venv/bin/python scripts/batch_llm_eval.py "
        "--output-dir '<NEW_OUTPUT_DIR>' "
        "--formal-manifest '<VERSIONED_RELEASE_MANIFEST>' "
        "--models '<ONE_MODEL>' --interaction-mode logical_persistent "
        "--seed-mode scenario --pass-k 1 --prompt-mode strict "
        "--scheduler-mode global --max-workers '<1_TO_32>' --temperature 0 "
        "--max-tokens 32768 --protocol-repair-max-tokens 8192 "
        "--provider-timeout-s 300 --persistent-history-max-messages 64 "
        "--persistent-context-max-chars 512000 "
        "--persistent-memory-max-items 128 --stream-chat-completions "
        "--save-trajectories --formal-run --finalize"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
