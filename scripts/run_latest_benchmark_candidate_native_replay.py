#!/usr/bin/env python3
"""Run bounded native probes for the latest benchmark candidate wave.

The command is intentionally candidate-only.  It reuses the existing
CityLearn source/runtime probe and the bounded SimBench native prefilter, but
binds their evidence to the current implementation tree and writes immutable
per-source terminals.  A runtime, source-lock, license, or implementation
drift is represented as a terminal held disposition; no row is promoted to
Core and no release artifact is touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CITYLEARN_ROOT = (
    REPO_ROOT
    / "works"
    / "CityLearn"
    / "data"
    / "datasets"
    / "citylearn_challenge_2022_phase_3"
)
DEFAULT_CITYLEARN_LOCK = (
    REPO_ROOT
    / "release"
    / "dt_sched_bench_v0_52_0_candidate"
    / "citylearn_source_lock.json"
)
DEFAULT_REPORT_ROOT = (
    REPO_ROOT
    / "reports"
    / "latest_benchmark_candidate_wave_20260813"
    / "native_conversion"
    / "current_hash_replay"
)
PIPELINE_VERSION = "latest_benchmark_candidate_native_replay_v1"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _identity() -> dict[str, Any]:
    from core.implementation_identity import implementation_identity

    return dict(implementation_identity())


def _held_row(
    *,
    source_id: str,
    domain: str,
    backend_kind: str,
    disposition: str,
    blockers: list[str],
    implementation: Mapping[str, Any],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_id": source_id,
        "domain": domain,
        "backend_kind": backend_kind,
        "stage": "native_prefilter",
        "work_state": "terminal",
        "disposition": disposition,
        "attempted": True,
        "materialized_rows": 0,
        "native_replay_executed": False,
        "native_passed_rows": 0,
        "full_protocol21_ready_rows": 0,
        "gate_failures": sorted(set(str(value) for value in blockers)),
        "blockers": sorted(set(str(value) for value in blockers)),
        "implementation_tree_sha256": implementation.get("implementation_tree_sha256"),
        "candidate_only": True,
        "core_admission_claimed": False,
    }
    if details:
        row["details"] = dict(details)
    return row


def _run_citylearn(
    *,
    source_root: Path,
    source_lock: Path,
    seed: int,
    ticks: int,
    implementation: Mapping[str, Any],
    preflight_fn: Callable[..., dict[str, Any]] | None = None,
    probe_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the bounded source and runtime checks for one CityLearn family."""

    if preflight_fn is None:
        from scripts.audit_citylearn_sources import build_citylearn_source_preflight_report

        preflight_fn = build_citylearn_source_preflight_report
    if probe_fn is None:
        from scripts.probe_citylearn_runtime import run_probe

        probe_fn = run_probe
    try:
        preflight = preflight_fn(source_root=source_root)
        probe = probe_fn(
            source_root=source_root,
            source_lock_path=source_lock,
            seed=int(seed),
            n_ticks=int(ticks),
        )
    except Exception as exc:  # pragma: no cover - runtime/dependency specific
        return _held_row(
            source_id="citylearn",
            domain="building_energy",
            backend_kind="citylearn_gymnasium",
            disposition="held_runtime",
            blockers=[f"native_probe_exception:{type(exc).__name__}"],
            implementation=implementation,
            details={"exception": str(exc)},
        )

    preflight_blockers = [
        str(value)
        for value in preflight.get("release_blocker_codes") or []
    ]
    probe_blockers = [str(value) for value in probe.get("release_blockers") or []]
    source_lock_blockers = [
        str(value) for value in probe.get("source_lock_blockers") or []
    ]
    all_blockers = sorted(set(preflight_blockers + probe_blockers + source_lock_blockers))
    runtime = probe.get("runtime") if isinstance(probe.get("runtime"), dict) else {}
    source_consumption = (
        probe.get("source_consumption")
        if isinstance(probe.get("source_consumption"), dict)
        else {}
    )
    native_replay_executed = probe.get("status") == "runtime_probe_passed"
    deterministic = bool(source_consumption.get("deterministic_replay"))
    native_effect = bool(runtime.get("native_state_effect_observed"))
    source_assets_complete = bool(source_consumption.get("runtime_opened_assets_complete"))
    source_lock_closed = not source_lock_blockers
    if not native_replay_executed:
        disposition = "held_runtime"
    elif not source_lock_closed or any("license" in blocker for blocker in all_blockers):
        disposition = "held_license_or_terms"
    elif not (deterministic and native_effect and source_assets_complete):
        disposition = "held_repair"
    else:
        disposition = "held_repair"
    if not all_blockers:
        all_blockers = ["citylearn_protocol21_full_gates_pending"]
    return {
        "source_id": "citylearn",
        "domain": "building_energy",
        "backend_kind": "citylearn_gymnasium",
        "stage": "native_prefilter",
        "work_state": "terminal",
        "disposition": disposition,
        "attempted": True,
        "materialized_rows": 1,
        "native_replay_executed": native_replay_executed,
        "native_passed_rows": int(native_replay_executed and deterministic and native_effect),
        "full_protocol21_ready_rows": 0,
        "gate_failures": all_blockers,
        "blockers": all_blockers,
        "implementation_tree_sha256": implementation.get("implementation_tree_sha256"),
        "candidate_only": True,
        "core_admission_claimed": False,
        "details": {
            "source_root": _repo_relative(source_root),
            "source_lock": _repo_relative(source_lock),
            "source_lock_sha256": _sha256(source_lock),
            "preflight_status": preflight.get("status"),
            "runtime_probe_status": probe.get("status"),
            "release_ready": bool(probe.get("release_ready")) and bool(preflight.get("release_ready")),
            "deterministic_replay": deterministic,
            "native_state_effect_observed": native_effect,
            "runtime_opened_assets_complete": source_assets_complete,
            "consumed_source_hashes": source_consumption.get("consumed_source_hashes") or {},
            "runtime": {
                "backend": runtime.get("backend"),
                "n_ticks": runtime.get("action_run", {}).get("n_ticks"),
                "package_version": probe.get("package_version"),
            },
            "preflight_report": preflight,
            "probe_report": probe,
        },
    }


def _run_simbench(
    *,
    implementation: Mapping[str, Any],
    report_fn: Callable[[], dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the bounded three-row SimBench native prefilter."""

    if report_fn is None:
        from scripts.build_simbench_protocol21_long_horizon_candidates import build_report

        report_fn = build_report
    try:
        prefilter = report_fn()
    except Exception as exc:  # pragma: no cover - runtime/dependency specific
        row = _held_row(
            source_id="simbench_official",
            domain="power_grid",
            backend_kind="cigre_distribution",
            disposition="held_runtime",
            blockers=[f"native_prefilter_exception:{type(exc).__name__}"],
            implementation=implementation,
            details={"exception": str(exc)},
        )
        return row, []
    children = [dict(item) for item in prefilter.get("results") or [] if isinstance(item, dict)]
    passed = [
        row
        for row in children
        if row.get("gate_failures") == ["full_protocol21_pending"]
    ]
    blockers = sorted(
        {
            str(failure)
            for row in children
            for failure in row.get("gate_failures") or []
        }
    )
    if passed:
        blockers.append("simbench_protocol21_full_gate_chain_pending")
        disposition = "candidate_prefilter"
    else:
        disposition = "held_repair"
    blockers = sorted(set(blockers))
    row = {
        "source_id": "simbench_official",
        "domain": "power_grid",
        "backend_kind": "cigre_distribution",
        "stage": "native_prefilter",
        "work_state": "terminal",
        "disposition": disposition,
        "attempted": True,
        "materialized_rows": len(children),
        "native_replay_executed": bool(children),
        "native_passed_rows": len(passed),
        "full_protocol21_ready_rows": 0,
        "gate_failures": blockers,
        "blockers": blockers,
        "implementation_tree_sha256": implementation.get("implementation_tree_sha256"),
        "candidate_only": True,
        "core_admission_claimed": False,
        "details": {
            "prefilter_status": prefilter.get("status"),
            "runtime_versions": prefilter.get("runtime_versions") or {},
            "source_suite_sha256": prefilter.get("source_suite_sha256"),
            "prefilter_implementation_sha256": prefilter.get("implementation_sha256"),
            "children": children,
        },
    }
    return row, children


def run_native_replay(
    *,
    report_root: Path = DEFAULT_REPORT_ROOT,
    citylearn_root: Path = DEFAULT_CITYLEARN_ROOT,
    citylearn_lock: Path = DEFAULT_CITYLEARN_LOCK,
    citylearn_seed: int = 2022,
    citylearn_ticks: int = 8,
    identity_fn: Callable[[], dict[str, Any]] | None = None,
    citylearn_runner: Callable[..., dict[str, Any]] | None = None,
    simbench_runner: Callable[..., tuple[dict[str, Any], list[dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    """Run native probes and write a fail-closed candidate-only report."""

    if identity_fn is None:
        identity_fn = _identity
    start_identity = identity_fn()
    if citylearn_runner is None:
        citylearn_runner = _run_citylearn
    if simbench_runner is None:
        simbench_runner = _run_simbench
    citylearn = citylearn_runner(
        source_root=citylearn_root,
        source_lock=citylearn_lock,
        seed=citylearn_seed,
        ticks=citylearn_ticks,
        implementation=start_identity,
    )
    simbench, children = simbench_runner(implementation=start_identity)
    end_identity = identity_fn()
    stable = (
        start_identity.get("implementation_tree_sha256")
        == end_identity.get("implementation_tree_sha256")
    )
    rows = [citylearn, simbench]
    if not stable:
        for row in rows:
            row["disposition"] = "held_stale_evidence"
            row["gate_failures"] = sorted(
                set(row.get("gate_failures") or []) | {"implementation_tree_drift"}
            )
            row["blockers"] = list(row["gate_failures"])
            row["native_passed_rows"] = 0
            row["full_protocol21_ready_rows"] = 0
    disposition_counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("disposition") or "unknown")
        disposition_counts[key] = disposition_counts.get(key, 0) + 1
    report: dict[str, Any] = {
        "schema_version": "latest-benchmark-candidate-native-replay-v1",
        "pipeline_version": PIPELINE_VERSION,
        "status": "complete_candidate_only" if stable else "stale_fail_closed",
        "scope": "bounded_native_prefilter_no_core_admission",
        "candidate_only": True,
        "core_admission_claimed": False,
        "implementation_identity_start": start_identity,
        "implementation_identity_end": end_identity,
        "implementation_tree_stable": stable,
        "counts": {
            "source_families": len(rows),
            "terminal_rows": len(rows),
            "native_replay_families": sum(bool(row.get("native_replay_executed")) for row in rows),
            "native_passed_rows": sum(int(row.get("native_passed_rows") or 0) for row in rows),
            "full_protocol21_ready_rows": 0,
            "dispositions": dict(sorted(disposition_counts.items())),
            "simbench_child_rows": len(children),
        },
        "policy": {
            "release_membership_changed": False,
            "raw_data_copied_or_redistributed": False,
            "full_protocol21_required_before_core": True,
            "implementation_drift_fails_closed": True,
        },
        "sources": rows,
    }
    report_root = report_root.resolve()
    reports_root = (REPO_ROOT / "reports").resolve()
    if not report_root.is_relative_to(reports_root):
        raise ValueError(f"candidate report root must stay under reports/: {report_root}")
    report_root.mkdir(parents=True, exist_ok=True)
    _write(report_root / "native_replay_summary.json", report)
    for row in rows:
        _write(report_root / "terminals" / f"{row['source_id']}.json", row)
    for child in children:
        child_id = _stable_hash(child.get("scenario_id") or child)[:16]
        _write(report_root / "terminals" / f"simbench_{child_id}.json", child)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--citylearn-root", type=Path, default=DEFAULT_CITYLEARN_ROOT)
    parser.add_argument("--citylearn-lock", type=Path, default=DEFAULT_CITYLEARN_LOCK)
    parser.add_argument("--citylearn-seed", type=int, default=2022)
    parser.add_argument("--citylearn-ticks", type=int, default=8)
    args = parser.parse_args()
    report = run_native_replay(
        report_root=args.report_root,
        citylearn_root=args.citylearn_root,
        citylearn_lock=args.citylearn_lock,
        citylearn_seed=args.citylearn_seed,
        citylearn_ticks=args.citylearn_ticks,
    )
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 0 if report["status"] == "complete_candidate_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
