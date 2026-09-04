#!/usr/bin/env python3
"""Build a report-only Protocol-2.1 expansion queue.

The queue joins existing domain reports and the external method-transfer
catalog.  It does not create scenarios, execute models, or promote Core rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_expansion_priority_matrix import (  # noqa: E402
    build_method_transfer_catalog,
)

TRACK_REPORTS: dict[str, tuple[tuple[str, str], ...]] = {
    "microgrid": (
        ("candidate", "grounded_microgrid_candidate_admission.json"),
        ("candidate", "grounded_microgrid_candidate_admission_v4.json"),
        ("candidate", "grounded_microgrid_post_minimality_v5.json"),
        ("candidate", "grounded_microgrid_model_diagnostic_v5.json"),
        ("reports", "microgrid_overlay_preflight.json"),
    ),
    "power_grid": (
        ("candidate", "grounded_power_candidate_admission_v1.json"),
        ("candidate", "grounded_power_post_minimality_v1.json"),
        ("candidate", "grounded_power_model_diagnostic_v1.json"),
        ("reports", "opendss_fresh_feeders_behavioral_gate.json"),
        ("reports", "acopf_cross_tick_behavioral_gate.json"),
    ),
    "traffic": (
        ("reports", "traffic_protocol21_feasibility_report.json"),
        ("candidate", "traffic_protocol21_feasibility_report.json"),
        ("reports", "sumo365_traffic_source_audit_current.json"),
        ("reports", "traffic_sumo365_live_headroom_probe_20230619_current.json"),
        ("reports", "resco_source_audit_six.json"),
        ("reports", "resco_replacement_calibration_v1.json"),
        ("reports", "resco_replacement_gate_v1.json"),
        ("root_reports", "sumo365_traffic_source_audit_current.json"),
        ("root_reports", "traffic_sumo365_live_headroom_probe_20230619_current.json"),
        ("root_reports", "traffic_protocol21_feasibility_report.json"),
        ("candidate", "protocol21_expansion_resco_candidates_v1.json"),
        ("candidate", "resco_replacement_gate_report.json"),
    ),
}

_ACCEPTED_ROW_STATUSES = {
    "admitted",
    "admitted_for_downstream_gates",
    "preadmitted_pending_one_minimal",
    "passed",
    "ready",
}
_STATUS_PRIORITY = {
    "passed": 100,
    "ready": 95,
    "admitted": 90,
    "admitted_for_downstream_gates": 90,
    "preadmitted_pending_one_minimal": 85,
    "failed": 80,
    "held": 70,
    "pending_model_discrimination": 60,
    "model_calibration_pending": 50,
    "pre_model_quality_failure": 40,
}


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "error": str(exc)}
    return payload if isinstance(payload, dict) else {"status": "invalid_payload"}


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("results", "rows", "scenarios", "candidates", "samples"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _has_native_evidence(row: dict[str, Any]) -> bool:
    explicit = row.get("native_evidence")
    if explicit is not None:
        return explicit is True
    for key in ("native_behavioral_validation", "behavioral_validation"):
        validation = row.get(key)
        if isinstance(validation, dict):
            checks = validation.get("checks") or {}
            return (
                validation.get("status") == "passed"
                and checks.get("native_backend_executable", True) is True
                and checks.get("native_state_changing_leverage", True) is True
            )
    checks = row.get("behavioral_checks") or row.get("checks")
    if isinstance(checks, dict):
        return (
            str(row.get("status") or "") == "passed"
            and checks.get("native_backend_executable") is True
            and checks.get("native_state_changing_leverage") is True
            and checks.get("task_contract_completed_by_reference") is True
        )
    runtime = row.get("runtime")
    if isinstance(runtime, dict):
        return (
            runtime.get("native") is True
            and runtime.get("status") == "passed"
            and runtime.get("state_effect_observed") is True
        )
    return False


def _row_status(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "held")
    if status in _ACCEPTED_ROW_STATUSES and not _has_native_evidence(row):
        return "held"
    return status


def _resolve_report(
    kind: str,
    filename: str,
    *,
    candidate_dir: Path,
    reports_dir: Path,
    repo_root: Path,
) -> Path:
    if kind == "candidate":
        return candidate_dir / filename
    if kind == "root_reports":
        return repo_root / "reports" / filename
    return reports_dir / filename


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _summarize_report(path: Path, *, repo_root: Path) -> dict[str, Any]:
    payload = _load(path)
    rows = _rows(payload)
    row_statuses = []
    for index, row in enumerate(rows):
        identifier = next(
            (
                str(row[key])
                for key in (
                    "scenario_id",
                    "seed_id",
                    "source_id",
                    "environment",
                    "path",
                )
                if row.get(key)
            ),
            f"row-{index}",
        )
        row_statuses.append(
            {"id": identifier, "status": _row_status(row)}
        )
    statuses = Counter(item["status"] for item in row_statuses)
    top_status = str(payload.get("status") or "missing")
    return {
        "path": _relative(path, repo_root),
        "exists": path.is_file(),
        "status": top_status,
        "n_rows": len(rows),
        "candidate_status_counts": dict(sorted(statuses.items())),
        "n_admitted": int(sum(statuses.get(key, 0) for key in _ACCEPTED_ROW_STATUSES)),
        "n_held": int(statuses.get("held", 0)),
        "row_statuses": row_statuses,
        "error": payload.get("error"),
    }


def _track_status(reports: list[dict[str, Any]]) -> str:
    statuses = [str(report["status"]) for report in reports]
    if not statuses or all(status == "missing" for status in statuses):
        return "missing"
    if any(status in {"blocked", "invalid", "invalid_payload"} for status in statuses):
        return "blocked"
    return "staging_only"


def build_queue(
    *,
    candidate_dir: Path,
    reports_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Build a deterministic, non-release expansion queue."""
    tracks: dict[str, dict[str, Any]] = {}
    for domain, entries in TRACK_REPORTS.items():
        reports = [
            _summarize_report(
                _resolve_report(
                    kind,
                    filename,
                    candidate_dir=candidate_dir,
                    reports_dir=reports_dir,
                    repo_root=repo_root,
                ),
                repo_root=repo_root,
            )
            for kind, filename in entries
        ]
        combined_rows: dict[str, str] = {}
        for report in reports:
            for item in report["row_statuses"]:
                identifier = str(item["id"])
                status = str(item["status"])
                prior = combined_rows.get(identifier)
                if prior is None or _STATUS_PRIORITY.get(
                    status, 1
                ) > _STATUS_PRIORITY.get(prior, 1):
                    combined_rows[identifier] = status
        combined = Counter(combined_rows.values())
        tracks[domain] = {
            "status": _track_status(reports),
            "reports": reports,
            "candidate_status_counts": dict(sorted(combined.items())),
            "n_rows": int(len(combined_rows)),
            "n_admitted": int(
                sum(combined.get(key, 0) for key in _ACCEPTED_ROW_STATUSES)
            ),
            "n_held": int(combined.get("held", 0)),
        }
    return {
        "schema_version": "protocol21-expansion-queue-1",
        "status": "staging_only",
        "tracks": tracks,
        "method_transfer_catalog": build_method_transfer_catalog(),
        "promotion_policy": {
            "all_protocol21_gates_required": True,
            "external_benchmark_data_direct_admission": False,
            "model_api_calls_permitted": False,
            "frozen_release_mutation_permitted": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_queue(
        candidate_dir=args.candidate_dir,
        reports_dir=args.reports_dir,
        repo_root=args.repo_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "tracks": {
                    key: value["status"] for key, value in report["tracks"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
