#!/usr/bin/env python3
"""Run one immutable, process-isolated native SUMO365 prefilter per date."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.implementation_identity import implementation_identity  # noqa: E402
from core.protocol21_evidence import required_semantics  # noqa: E402
from domains.traffic.seeds.sumo365 import SUMO365_SERVICE_DATES  # noqa: E402

TARGET_VERSION = "1.27.1"
DEFAULT_SUITE = REPO_ROOT / "reports/wave2_traffic_sumo365_source_suite_v2.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/sumo365_serial_prefilter_1_27_1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    serialized = _canonical(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"immutable result already exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def _sumo_binary() -> Path | None:
    spec = importlib.util.find_spec("sumo")
    if spec is None or not spec.submodule_search_locations:
        return None
    candidate = Path(next(iter(spec.submodule_search_locations))) / "bin" / "sumo"
    return candidate.resolve() if candidate.is_file() and os.access(candidate, os.X_OK) else None


def _runtime_contract() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("libsumo", "traci", "sumolib", "sumo-data"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    binary = _sumo_binary()
    version_line = ""
    binary_sha = None
    error = None
    if binary is None:
        error = "sumo_binary_missing"
    else:
        binary_sha = _sha256(binary)
        completed = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        version_line = (completed.stdout or completed.stderr).splitlines()[0].strip()
        if completed.returncode != 0:
            error = f"sumo_version_probe_exit_{completed.returncode}"
    exact_packages = all(packages[name] == TARGET_VERSION for name in packages)
    exact_binary = version_line.endswith(TARGET_VERSION)
    return {
        "target_version": TARGET_VERSION,
        "packages": packages,
        "binary": str(binary) if binary else None,
        "binary_sha256": binary_sha,
        "binary_version_line": version_line,
        "transport": "traci_tcp",
        "forced_transport_env": "traci",
        "exact_version_match": bool(exact_packages and exact_binary and not error),
        "error": error,
    }


def _classify_probe(
    service_date: str,
    report: dict[str, Any],
    *,
    runtime_ok: bool,
    current_tree: str | None = None,
    source_suite_binding_ok: bool = True,
) -> dict[str, Any]:
    episodes = [report.get(name) or {} for name in ("wait_floor", "acting", "wait_repeat")]
    traces = [episode.get("source_consumption_evidence") or {} for episode in episodes]
    material_source_events = [
        sum(
            1
            for event in episode.get("realized_event_summary") or []
            if isinstance(event, dict)
            and event.get("origin") in {"source_schedule", "declared_perturbation"}
            and event.get("type") != "sumo_live_snapshot"
            and (
                event.get("material_change") is True
                or float(event.get("materiality_value") or 0.0) > 0.0
            )
        )
        for episode in episodes
    ]
    current_tree = current_tree or str(
        implementation_identity()["implementation_tree_sha256"]
    )
    headroom = report.get("native_headroom_evidence") or {}

    def _finite(name: str) -> float | None:
        try:
            value = float(headroom[name])
        except (KeyError, TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    wait_loss = _finite("wait_native_loss")
    acting_loss = _finite("acting_native_loss")
    wait_full = _finite("wait_full_network_native_loss")
    acting_full = _finite("acting_full_network_native_loss")
    corridor_improvement = (
        wait_loss - acting_loss if wait_loss is not None and acting_loss is not None else None
    )
    full_improvement = (
        wait_full - acting_full if wait_full is not None and acting_full is not None else None
    )
    required_corridor = (
        max(1.0, 0.05 * wait_loss) if wait_loss is not None and wait_loss > 0.0 else None
    )
    required_full = (
        max(1.0, 0.05 * wait_full)
        if wait_full is not None and wait_full > 0.0
        else None
    )
    beneficial_native_headroom = bool(
        corridor_improvement is not None
        and full_improvement is not None
        and required_corridor is not None
        and required_full is not None
        and corridor_improvement >= required_corridor
        and full_improvement >= required_full
    )
    evidence_current = bool(
        report.get("implementation_tree_stable") is True
        and report.get("implementation_tree_sha256_start") == current_tree
        and report.get("implementation_tree_sha256") == current_tree
        and report.get("evaluation_semantics") == required_semantics()
    )
    gates = {
        "current_evidence": evidence_current,
        "source_suite_binding": source_suite_binding_ok,
        "runtime_exact_1_27_1": runtime_ok,
        "native_execution": report.get("executed_with_live_backend") is True,
        "traci_transport": report.get("selected_transport") == "traci",
        "source_consumption": bool(traces)
        and all(
            trace.get("status") == "passed" and trace.get("source_state_effect_observed") is True
            for trace in traces
        ),
        "determinism": report.get("metric_deterministic_replay_passed") is True
        and all(trace.get("trace_materiality_ready") is True for trace in traces),
        "events_realized": bool(episodes)
        and all(count > 0 for count in material_source_events),
        "native_control_effect": report.get("state_change_probe_passed") is True
        and report.get("evidence_wiring_probe_passed") is True,
        "positive_headroom": beneficial_native_headroom,
    }
    if not runtime_ok or report.get("status") in {
        "live_probe_execution_failed",
        "skipped_no_sumo_transport",
        "skipped_env_gate_unset",
    }:
        disposition = "held_runtime"
        reasons = ["runtime_unavailable_or_probe_failed"]
    elif not evidence_current or not source_suite_binding_ok:
        disposition = "held_stale_evidence"
        reasons = [
            "implementation_or_semantics_binding_stale"
            if not evidence_current
            else "source_suite_binding_stale"
        ]
    elif all(gates.values()):
        disposition = "full_protocol21_pending"
        reasons = ["all_native_prefilter_gates_passed"]
    else:
        disposition = "held_repair"
        reason_map = {
            "source_consumption": "source_consumption_unproven",
            "determinism": "determinism_unproven",
            "events_realized": "runtime_events_unproven",
            "native_control_effect": "native_control_effect_unproven",
            "positive_headroom": "positive_headroom_unproven",
            "traci_transport": "transport_identity_mismatch",
            "native_execution": "native_execution_unproven",
        }
        reasons = [
            reason_map[key] for key, value in gates.items() if not value and key in reason_map
        ]
    return {
        "schema_version": "sumo365-serial-native-prefilter-v1",
        "service_date": service_date,
        "work_state": "terminal",
        "disposition": disposition,
        "candidate_only": True,
        "release_admission": False,
        "gates": gates,
        "reason_codes": reasons,
        "probe_report": report,
        "implementation_tree_sha256": current_tree,
    }


def _terminate_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            continue


def _run_date(
    service_date: str,
    *,
    runtime: dict[str, Any],
    output_root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    if runtime.get("exact_version_match") is not True:
        return _classify_probe(
            service_date,
            {"status": "runtime_contract_failed", "runtime_contract": runtime},
            runtime_ok=False,
        )
    binary = Path(str(runtime["binary"]))
    env = dict(os.environ)
    env.update(
        {
            "OPERATE_TRAFFIC_BACKEND_REAL": "1",
            "OPERATE_TRAFFIC_FORCE_TRANSPORT": "traci",
            "PATH": f"{binary.parent}{os.pathsep}{env.get('PATH', '')}",
        }
    )
    scratch_root = output_root / ".scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"sumo365-{service_date}-", dir=scratch_root) as temp:
        probe_path = Path(temp) / "probe.json"
        command = [
            str(REPO_ROOT / ".venv/bin/python"),
            str(REPO_ROOT / "scripts/traffic_sumo365_live_headroom_probe.py"),
            "--output",
            str(probe_path),
            "--service-date",
            service_date,
            "--n-ticks",
            "2",
        ]
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_group(process)
            report = {
                "status": "live_probe_execution_failed",
                "reason": f"TimeoutError: serial probe exceeded {timeout_seconds}s",
                "stdout_tail": "",
            }
        else:
            report = (
                json.loads(probe_path.read_text(encoding="utf-8"))
                if probe_path.is_file()
                else {
                    "status": "live_probe_execution_failed",
                    "reason": f"probe exit={process.returncode} without JSON",
                    "stdout_tail": stdout[-4000:],
                }
            )
        finally:
            _terminate_group(process)
    result = _classify_probe(service_date, report, runtime_ok=True)
    result["runtime_contract"] = runtime
    return result


def _ledger_result_row(row: dict[str, Any]) -> dict[str, Any]:
    probe = row.get("probe_report") or {}
    headroom = probe.get("native_headroom_evidence") or {}
    return {
        "service_date": row["service_date"],
        "work_state": row["work_state"],
        "disposition": row["disposition"],
        "reason_codes": row["reason_codes"],
        "gates": row["gates"],
        "safe_target_count": len(probe.get("live_state_derived_targets") or []),
        "wait_native_loss": float(headroom.get("wait_native_loss") or 0.0),
        "acting_native_loss": float(headroom.get("acting_native_loss") or 0.0),
        "native_loss_improvement": float(headroom.get("native_loss_improvement") or 0.0),
        "required_native_loss_improvement": float(
            headroom.get("required_native_loss_improvement") or 0.0
        ),
        "headroom_l1_minutes": float(probe.get("headroom_l1_minutes") or 0.0),
        "result_path": f"dates/{row['service_date']}.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--date", action="append", dest="dates")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    suite_path = args.suite.resolve()
    output_root = args.output_root.resolve()
    output_root.relative_to(REPO_ROOT.resolve())
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite_dates = sorted(
        str((row.get("scenario_id") or "").split("/")[-2]) for row in suite.get("scenarios") or []
    )
    dates = sorted(args.dates or suite_dates)
    if any(date not in SUMO365_SERVICE_DATES or date not in suite_dates for date in dates):
        raise ValueError("requested date is not present in the locked source suite")
    runtime = _runtime_contract()
    current_tree = str(implementation_identity()["implementation_tree_sha256"])
    suite_binding = {"path": str(suite_path), "sha256": _sha256(suite_path)}
    results: list[dict[str, Any]] = []
    for service_date in dates:
        path = output_root / "dates" / f"{service_date}.json"
        if path.exists() and args.resume:
            saved = json.loads(path.read_text(encoding="utf-8"))
            result = _classify_probe(
                service_date,
                saved.get("probe_report") or {},
                runtime_ok=runtime.get("exact_version_match") is True
                and saved.get("runtime_contract") == runtime,
                current_tree=current_tree,
                source_suite_binding_ok=saved.get("source_suite") == suite_binding,
            )
            result["runtime_contract"] = runtime
            result["source_suite"] = suite_binding
        elif path.exists():
            raise FileExistsError(f"immutable result already exists: {path}")
        else:
            result = _run_date(
                service_date,
                runtime=runtime,
                output_root=output_root,
                timeout_seconds=args.timeout_seconds,
            )
            result["source_suite"] = suite_binding
            _write_immutable(path, result)
        results.append(result)
        print(
            f"{service_date} {result['work_state']} {result['disposition']}",
            flush=True,
        )
    ledger = {
        "schema_version": "sumo365-serial-native-prefilter-ledger-v1",
        "status": "complete",
        "workers": 1,
        "candidate_only": True,
        "release_admission": False,
        "implementation_tree_sha256": current_tree,
        "evaluation_semantics": required_semantics(),
        "source_suite": suite_binding,
        "runtime_contract": runtime,
        "n_expected": len(dates),
        "n_terminal": sum(row.get("work_state") == "terminal" for row in results),
        "disposition_counts": {
            disposition: sum(row["disposition"] == disposition for row in results)
            for disposition in sorted({row["disposition"] for row in results})
        },
        "results": [_ledger_result_row(row) for row in results],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "ledger.json").write_text(_canonical(ledger), encoding="utf-8")
    print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
