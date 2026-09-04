#!/usr/bin/env python3
"""Build a conservative, non-release Gate-0 report for Wave-1 sources.

Gate-0 only answers whether a source is sufficiently identified to enter a
candidate probe.  It deliberately does not admit scenarios, write release
manifests, or infer Core eligibility from package availability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "protocol21_wave1_gate0_current.json"
DEFAULT_CORE = (
    REPO_ROOT
    / "reports/protocol21_pending_union_fresh_e18_realtraffic_v1"
    / "refined_core_selection_protocol2_v21.json"
)
DEFAULT_FRONTIER = REPO_ROOT / "reports/frontier_domain_candidates.json"
DEFAULT_EXTERNAL = REPO_ROOT / "reports/external_source_locks.json"

DEFAULT_SOURCE_REPORTS = {
    "acn_ev_charging": REPO_ROOT / "reports/acn_ev_charging_source_preflight.json",
    "citylearn": REPO_ROOT / "reports/citylearn_source_preflight.json",
    "grid2op": REPO_ROOT / "reports/grid2op_source_audit.json",
}


def _implementation_tree_sha256() -> str | None:
    try:
        from core.implementation_identity import implementation_identity

        return str(implementation_identity().get("implementation_tree_sha256") or "") or None
    except Exception:  # noqa: BLE001 - a missing identity is a blocker, not a crash
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _status(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "missing"
    declared = payload.get("status")
    if declared:
        return str(declared)
    if isinstance(payload.get("summary"), dict):
        return "source_audit_completed"
    return "unknown"


def _blockers(payload: dict[str, Any] | None) -> list[str]:
    if payload is None:
        return ["report_missing"]
    values = payload.get("release_blocker_codes")
    if not isinstance(values, list):
        values = payload.get("blockers")
    if not isinstance(values, list):
        return []
    return sorted({str(value) for value in values if str(value)})


def _core_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        raise FileNotFoundError(path)
    rows = payload.get("scenarios")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: scenarios must be a list")
    identities: list[tuple[str, str]] = []
    physical: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{path}: scenario row must be an object")
        scenario_id = str(row.get("scenario_id") or "")
        signature = str(row.get("scenario_signature") or "")
        if not scenario_id or not signature:
            raise ValueError(f"{path}: Core row has incomplete identity")
        identities.append((scenario_id, signature))
        ledger = row.get("case_ledger")
        if isinstance(ledger, dict):
            lock = ledger.get("physical_source_lock")
            if isinstance(lock, (dict, list)):
                physical.add(json.dumps(lock, sort_keys=True, separators=(",", ":")))
            elif lock not in (None, ""):
                physical.add(str(lock))
    return {
        "path": _display_path(path),
        "sha256": _sha256(path),
        "status": str(payload.get("status") or "unknown"),
        "n_core": len(rows),
        "n_effective_identities": len(set(identities)),
        "n_physical_source_locks": len(physical),
        "implementation_tree_sha256": payload.get("implementation_tree_sha256"),
    }


def _source_summary(name: str, path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    summary: dict[str, Any] = {
        "source_family": name,
        "path": _display_path(path),
        "present": payload is not None,
        "status": _status(payload),
        "release_ready": bool(payload.get("release_ready")) if payload else False,
        "release_reentry_ready": bool(payload.get("release_reentry_ready")) if payload else False,
        "blocker_codes": _blockers(payload),
    }
    if payload is not None:
        for key in ("source_id", "selected_source_candidate", "package_preflight", "summary"):
            if key in payload:
                summary[key] = payload[key]
    return summary


def _frontier_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        return {"path": str(path), "present": False, "status": "missing"}
    candidates = payload.get("candidates")
    rows = candidates if isinstance(candidates, list) else []
    return {
        "path": _display_path(path),
        "present": True,
        "status": str(payload.get("status") or "unknown"),
        "release_ready": bool(payload.get("release_ready")),
        "n_candidates": len(rows),
        "release_blocker_codes": sorted(
            {str(value) for value in payload.get("release_blocker_codes", [])}
        ),
        "candidate_ids": sorted(
            str(row.get("candidate_id"))
            for row in rows
            if isinstance(row, dict) and row.get("candidate_id")
        ),
    }


def _external_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload is None:
        return {"path": str(path), "present": False, "status": "missing"}
    rows = payload.get("sources")
    rows = rows if isinstance(rows, list) else []
    return {
        "path": _display_path(path),
        "present": True,
        "status": str(payload.get("status") or "unknown"),
        "n_sources": len(rows),
        "n_source_lock_verified": sum(
            bool(row.get("source_lock_verified"))
            for row in rows
            if isinstance(row, dict)
        ),
        "blocked_source_ids": sorted(
            str(row.get("source_id"))
            for row in rows
            if isinstance(row, dict) and not row.get("source_lock_verified")
        ),
    }


def build_report(
    *,
    core_path: Path = DEFAULT_CORE,
    frontier_path: Path = DEFAULT_FRONTIER,
    external_path: Path = DEFAULT_EXTERNAL,
    source_reports: dict[str, Path] | None = None,
) -> dict[str, Any]:
    reports = source_reports or DEFAULT_SOURCE_REPORTS
    source_rows = [_source_summary(name, path) for name, path in sorted(reports.items())]
    return {
        "schema_version": "protocol21-wave1-gate0-v1",
        "scope": "wave1_candidate_source_inventory_and_runtime_preflight",
        "release_admission": False,
        "candidate_only": True,
        "status": "candidate_gate0_open" if source_rows else "candidate_gate0_empty",
        "promotion_allowed": False,
        "policy": {
            "locked_core_mutated": False,
            "raw_restricted_data_repackaged": False,
            "llm_outcomes_used_for_admission": False,
            "source_native_probe_required_before_materialization": True,
            "final_union_required_before_formal_readiness": True,
        },
        "implementation_tree_sha256": _implementation_tree_sha256(),
        "locked_core": _core_summary(core_path),
        "source_reports": source_rows,
        "frontier_inventory": _frontier_summary(frontier_path),
        "external_source_locks": _external_summary(external_path),
        "gate_contract": {
            "required_before_conversion_batch": [
                "verified_source_url_and_license_or_access_policy",
                "physical_identity_and_effective_window",
                "backend_runtime_version_and_seed",
                "source_consumption_evidence",
            ],
            "required_before_protocol21_core_admission": [
                "native_action_effect",
                "deterministic_replay",
                "no_action_counterfactual_or_explicit_opt_out",
                "positive_reference_headroom",
                "complete_event_and_evidence_lineage",
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument(
        "--source-report",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="override/add a source report; may be repeated",
    )
    args = parser.parse_args(argv)
    source_reports = dict(DEFAULT_SOURCE_REPORTS)
    for raw in args.source_report:
        if "=" not in raw:
            parser.error("--source-report must use NAME=PATH")
        name, raw_path = raw.split("=", 1)
        if not name or not raw_path:
            parser.error("--source-report must use NAME=PATH")
        source_reports[name] = Path(raw_path)
    report = build_report(
        core_path=args.core,
        frontier_path=args.frontier,
        external_path=args.external,
        source_reports=source_reports,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "release_admission": report["release_admission"],
                "n_core": report["locked_core"]["n_core"],
                "source_reports": len(report["source_reports"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
