#!/usr/bin/env python3
"""Run strict-prompt LLM episodes for a source-bound driving suite.

This is an execution harness, not a baseline.  It writes provider results and
an immutable manifest consumed by
``validate_autonomous_driving_llm_evidence.py``.  Missing credentials or
provider failures are preserved as held evidence; they are never replaced by
wait-only output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess  # nosec B404 - fixed argv, no shell
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DIFFICULTIES = ("basic", "medium", "high", "extreme")
AGENTS = ("llm_agent", "react_llm", "reflexion_llm")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _suite_entries(suite_dir: Path, *, require_live: bool = False) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for report_path in sorted(suite_dir.glob("*/suite_report.json")):
        report = _load(report_path)
        candidate_id = str(report.get("candidate_id") or "")
        source_hash = str(report.get("source_window_sha256") or "")
        for scenario_ref in report.get("difficulty_slices") or []:
            scenario_path = report_path.parent / str(scenario_ref)
            if not candidate_id or not scenario_path.is_file():
                raise ValueError("autonomous_driving_llm_suite_entry_invalid")
            scenario = _load_yaml(scenario_path)
            difficulty = str(scenario.get("difficulty_level") or "")
            if difficulty not in DIFFICULTIES:
                raise ValueError("autonomous_driving_llm_difficulty_invalid")
            backend = scenario.get("backend_config") or {}
            if str(backend.get("candidate_id") or "") != candidate_id:
                raise ValueError("autonomous_driving_llm_candidate_binding_mismatch")
            execution_mode = str(backend.get("execution_mode") or "")
            if require_live and execution_mode != "live":
                raise ValueError("autonomous_driving_llm_requires_live_suite")
            if source_hash and str(scenario.get("source_window_sha256") or "") != source_hash:
                raise ValueError("autonomous_driving_llm_source_window_binding_mismatch")
            entries.append(
                {
                    "candidate_id": candidate_id,
                    "difficulty_level": difficulty,
                    "scenario_id": str(scenario.get("seed_id") or ""),
                    "scenario_path": str(scenario_path.resolve()),
                    "scenario_yaml_sha256": _sha256(scenario_path),
                    "semantic_contract_sha256": _digest(scenario),
                    "horizon_ticks": str(scenario.get("horizon_ticks") or ""),
                    "source_window_sha256": str(scenario.get("source_window_sha256") or ""),
                    "source_event_chain_sha256": str(
                        (scenario.get("provenance") or {}).get("source_event_chain_sha256") or ""
                    ),
                    "execution_mode": execution_mode,
                }
            )
    if not entries:
        raise ValueError("autonomous_driving_llm_suite_empty")
    return entries


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml  # type: ignore[import-untyped]

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def build_manifest(
    *,
    suite_dir: Path,
    result_dir: Path,
    agent_name: str,
    provider: str,
    model: str,
    prompt_mode: str = "strict",
    require_live: bool = False,
) -> dict[str, Any]:
    if agent_name not in AGENTS:
        raise ValueError("autonomous_driving_llm_agent_invalid")
    if not provider or not model or prompt_mode != "strict":
        raise ValueError("autonomous_driving_llm_provider_or_prompt_invalid")
    entries = _suite_entries(suite_dir.resolve(), require_live=require_live)
    runs: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        result_path = (
            result_dir
            / f"{index:04d}-{entry['candidate_id'].replace(':', '_')}-{entry['difficulty_level']}.json"
        )
        runs.append(
            {
                "candidate_id": entry["candidate_id"],
                "difficulty_level": entry["difficulty_level"],
                "scenario_id": entry["scenario_id"],
                "scenario_path": entry["scenario_path"],
                "scenario_yaml_sha256": entry["scenario_yaml_sha256"],
                "semantic_contract_sha256": entry["semantic_contract_sha256"],
                "horizon_ticks": int(entry["horizon_ticks"] or 0),
                "source_window_sha256": entry["source_window_sha256"],
                "source_event_chain_sha256": entry["source_event_chain_sha256"],
                "execution_mode": entry["execution_mode"],
                "result_path": result_path.name,
            }
        )
    suite_binding = {
        f"{entry['candidate_id']}:{entry['difficulty_level']}": {
            "scenario_yaml_sha256": entry["scenario_yaml_sha256"],
            "semantic_contract_sha256": entry["semantic_contract_sha256"],
            "horizon_ticks": int(entry["horizon_ticks"] or 0),
        }
        for entry in entries
    }
    return {
        "schema_version": "autonomous_driving_llm_evidence_manifest_v1",
        "status": "planned",
        "agent_name": agent_name,
        "provider": provider,
        "model": model,
        "prompt_mode": prompt_mode,
        "requires_live_backend": require_live,
        "required_difficulties": list(DIFFICULTIES),
        "suite_dir": str(suite_dir.resolve()),
        "result_dir": str(result_dir.resolve()),
        "suite_semantic_coverage_sha256": _digest(suite_binding),
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agent", choices=AGENTS, default="llm_agent")
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="reject suites whose backend_config.execution_mode is not live",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    suite_dir = args.suite_dir if args.suite_dir.is_absolute() else REPO_ROOT / args.suite_dir
    result_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    manifest_path = result_dir / "manifest.json"
    try:
        manifest = build_manifest(
            suite_dir=suite_dir,
            result_dir=result_dir,
            agent_name=args.agent,
            provider=args.provider,
            model=args.model,
            require_live=args.require_live,
        )
        if args.check:
            current = _load(manifest_path)
            ok = current == manifest
            print(json.dumps({"status": "verified" if ok else "stale"}, sort_keys=True))
            return 0 if ok else 1
        if result_dir.exists() and any(result_dir.iterdir()):
            raise FileExistsError("autonomous_driving_llm_result_dir_not_empty")
        result_dir.mkdir(parents=True, exist_ok=True)
        execution_error: str | None = None
        if not os.getenv(str(args.api_key_env)):
            execution_error = f"provider_api_key_missing:{args.api_key_env}"
            manifest["runs"][0]["status"] = "held"
            manifest["runs"][0]["error"] = execution_error
            for pending in manifest["runs"][1:]:
                pending["status"] = "not_attempted"
                pending["error"] = "provider_failure_before_execution"
        else:
            for index, entry in enumerate(manifest["runs"]):
                if execution_error is not None:
                    entry["status"] = "not_attempted"
                    entry["error"] = "provider_failure_before_execution"
                    continue
                command = [
                    sys.executable,
                    str(REPO_ROOT / "run.py"),
                    "--scenario",
                    str(
                        next(
                            row["scenario_path"]
                            for row in _suite_entries(
                                suite_dir, require_live=bool(manifest.get("requires_live_backend"))
                            )
                            if row["scenario_id"] == entry["scenario_id"]
                            and row["difficulty_level"] == entry["difficulty_level"]
                            and row["candidate_id"] == entry["candidate_id"]
                        )
                    ),
                    "--agent",
                    args.agent,
                    "--provider",
                    args.provider,
                    "--model",
                    args.model,
                    "--api-key-env",
                    args.api_key_env,
                    "--prompt-mode",
                    "strict",
                    "--seed",
                    str(args.seed),
                    "--output",
                    str(result_dir / str(entry["result_path"])),
                ]
                completed = subprocess.run(  # nosec B603 - fixed argv, no shell
                    command,
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                    env={**os.environ, "OPERATE_AUTONOMOUS_DRIVING_SUMO_REAL": "1"},
                )
                if completed.returncode != 0:
                    execution_error = (
                        f"autonomous_driving_llm_episode_failed:{entry['scenario_id']}:"
                        f"{completed.stderr.strip() or completed.stdout.strip()}"
                    )
                    entry["status"] = "held"
                    entry["error"] = execution_error
                    for pending in manifest["runs"][index + 1 :]:
                        pending["status"] = "not_attempted"
                        pending["error"] = "provider_failure_before_execution"
                    break
        for entry in manifest["runs"]:
            path = result_dir / str(entry["result_path"])
            if path.is_file():
                entry["result_sha256"] = _sha256(path)
        manifest["status"] = "held" if execution_error else "completed"
        if execution_error:
            manifest["execution_error"] = execution_error
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": "held" if execution_error else "completed",
                    "runs": len(manifest["runs"]),
                    "manifest": str(manifest_path),
                },
                sort_keys=True,
            )
        )
        return 2 if execution_error else 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "held", "reason": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
