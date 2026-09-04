#!/usr/bin/env python3
"""Merge Microgrid/Traffic/latest-benchmark candidate evidence.

This is a candidate-only coordinator report.  It does not call a simulator,
rewrite a scenario, or modify a Core/release artifact.  The expensive native
probes are deliberately separate (for example ``run_sumo365_serial_prefilter``
and ``build_microgrid_held_refine``); this command makes their immutable
outputs comparable and fail-closed.  In particular, a native result without an
implementation-tree binding is visible as stale evidence rather than being
counted as a current survivor.

Every input row gets exactly one terminal row.  Existing source/route hashes,
candidate YAML hashes, and source-suite hashes are retained so a later fresh
replay can consume this ledger without guessing which artifact was used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = REPO_ROOT / "reports"

DEFAULT_MICROGRID = (
    REPORTS_ROOT / "microgrid_held_refine_current_20260814" / "refine_report.json"
)
DEFAULT_TRAFFIC_QUEUE = (
    REPORTS_ROOT / "traffic_batch_candidates_v3_20260812" / "traffic_candidate_queue.json"
)
DEFAULT_TRAFFIC_NATIVE = (
    REPORTS_ROOT / "sumo365_native_tls_candidate_9date_v3" / "ledger.json"
)
DEFAULT_LATEST = REPORTS_ROOT / "latest_benchmark_candidate_wave_20260813.json"
DEFAULT_OUTPUT_ROOT = REPORTS_ROOT / "underrepresented_candidate_terminal_20260814"


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


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _implementation_tree_sha256() -> str | None:
    try:
        from core.implementation_identity import implementation_identity

        return str(implementation_identity().get("implementation_tree_sha256") or "") or None
    except Exception:  # pragma: no cover - identity is optional in report-only use
        return None


def _terminal(
    *,
    source_id: str,
    domain: str,
    backend_kind: str | None,
    stage: str,
    disposition: str,
    blockers: list[str],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_id": source_id,
        "domain": domain,
        "backend_kind": backend_kind,
        "stage": stage,
        "work_state": "terminal",
        "disposition": disposition,
        "candidate_only": True,
        "core_admission_claimed": False,
        "blockers": sorted(set(str(value) for value in blockers)),
    }
    if details:
        row["details"] = dict(details)
    return row


def _microgrid_rows(report: Mapping[str, Any], *, report_path: Path) -> list[dict[str, Any]]:
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, list):
        raise ValueError("microgrid refine report must contain outcomes")
    rows: list[dict[str, Any]] = []
    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise ValueError("microgrid outcome must be an object")
        scenario_id = str(outcome.get("scenario_id") or "").strip()
        if not scenario_id:
            raise ValueError("microgrid outcome is missing scenario_id")
        status = str(outcome.get("work_state") or "terminal")
        disposition = str(outcome.get("disposition") or "held_repair")
        if disposition == "candidate_pending_full_protocol21":
            # A bounded native survivor is still a candidate until the fresh
            # full gate chain runs.  Do not label it as a Core increment.
            disposition = "candidate_prefilter"
            stage = "full_protocol21"
            blockers = ["fresh_full_protocol21_replay_pending"]
        elif disposition == "secondary_duplicate":
            stage = "evidence_freeze"
            blockers = ["effective_source_already_represented"]
        else:
            stage = "native_prefilter"
            blockers = [str(item) for item in outcome.get("reason_codes") or []]
        selected = outcome.get("selected_probe")
        details: dict[str, Any] = {
            "scenario_id": scenario_id,
            "source_report": _relative(report_path),
            "source_report_sha256": _sha256(report_path),
            "upstream_work_state": status,
            "native_refine": {
                "screen_passed": bool(selected and selected.get("screen_passed")),
                "selected_probe": selected,
            },
            "candidate_paths": [
                str((outcome.get("full_behavioral") or {}).get("path"))
            ]
            if (outcome.get("full_behavioral") or {}).get("path")
            else [],
        }
        rows.append(
            _terminal(
                source_id=scenario_id,
                domain="microgrid",
                backend_kind="pandapower_lv",
                stage=stage,
                disposition=disposition,
                blockers=blockers,
                details=details,
            )
        )
    return rows


def _traffic_rows(
    queue: Mapping[str, Any], native: Mapping[str, Any], *, queue_path: Path, native_path: Path
) -> list[dict[str, Any]]:
    items = queue.get("items")
    native_results = native.get("results")
    if not isinstance(items, list) or not isinstance(native_results, list):
        raise ValueError("traffic queue/native ledger must contain lists")
    by_date = {
        str(row.get("service_date")): row
        for row in native_results
        if isinstance(row, dict) and row.get("service_date")
    }
    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("traffic queue item must be an object")
        source_id = str(item.get("scenario_id") or item.get("work_id") or "").strip()
        if not source_id:
            raise ValueError("traffic queue item is missing scenario identity")
        parts = source_id.split("/")
        service_date = next((part for part in parts if part[:4].isdigit() and len(part) == 10), "")
        native_row = by_date.get(service_date)
        candidate_ref = item.get("candidate_yaml")
        candidate_path = (
            _resolve(candidate_ref, base=queue_path.parent) if isinstance(candidate_ref, str) else None
        )
        blockers = ["positive_headroom_unproven", "native_prefilter_not_current_hash_bound"]
        native_executed = bool(native_row)
        if native_row:
            blockers = sorted(
                set(str(value) for value in native_row.get("reason_codes") or [])
                | {"native_prefilter_not_current_hash_bound"}
            )
        else:
            blockers = ["missing_matching_native_prefilter_result"]
        details = {
            "scenario_id": source_id,
            "service_date": service_date or None,
            "queue_path": _relative(queue_path),
            "queue_sha256": _sha256(queue_path),
            "native_ledger_path": _relative(native_path),
            "native_ledger_sha256": _sha256(native_path),
            "effective_source_identity": item.get("effective_source_identity"),
            "physical_source_identity": item.get("physical_source_identity"),
            "candidate_yaml": _relative(candidate_path) if candidate_path else None,
            "candidate_yaml_sha256": _sha256(candidate_path) if candidate_path else None,
            "static_conversion": {
                "stage": item.get("stage"),
                "work_state": item.get("work_state"),
                "disposition": item.get("disposition"),
            },
            "native_prefilter": native_row,
            "native_replay_executed": native_executed,
            "current_hash_bound": False,
        }
        rows.append(
            _terminal(
                source_id=source_id,
                domain="traffic",
                backend_kind="sumo",
                stage="native_prefilter",
                disposition="held_repair",
                blockers=blockers,
                details=details,
            )
        )
    return rows


def _latest_rows(report: Mapping[str, Any], *, report_path: Path) -> list[dict[str, Any]]:
    sources = report.get("sources")
    if not isinstance(sources, list):
        raise ValueError("latest benchmark report must contain sources")
    rows: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("latest benchmark source row must be an object")
        source_id = str(source.get("source_id") or "").strip()
        if not source_id:
            raise ValueError("latest benchmark source is missing source_id")
        # SimBench and CityLearn family rows are already represented by their
        # dedicated domain/conversion reports.  Keep their latest terminal as
        # a single source row only when no explicit downstream row is present.
        rows.append(
            _terminal(
                source_id=f"latest::{source_id}",
                domain=str(source.get("domain") or "unknown"),
                backend_kind=source.get("backend_kind"),
                stage=str(source.get("stage") or "inventory"),
                disposition=str(source.get("disposition") or "held"),
                blockers=[str(value) for value in source.get("blockers") or []],
                details={
                    "source_report": _relative(report_path),
                    "source_report_sha256": _sha256(report_path),
                    "upstream": source,
                },
            )
        )
    return rows


def build_ledger(
    *,
    microgrid_path: Path,
    traffic_queue_path: Path,
    traffic_native_path: Path,
    latest_path: Path,
    implementation_tree_sha256: str | None = None,
) -> dict[str, Any]:
    paths = (microgrid_path, traffic_queue_path, traffic_native_path, latest_path)
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(", ".join(missing))
    rows = []
    rows.extend(_microgrid_rows(_load(microgrid_path), report_path=microgrid_path))
    rows.extend(
        _traffic_rows(
            _load(traffic_queue_path),
            _load(traffic_native_path),
            queue_path=traffic_queue_path,
            native_path=traffic_native_path,
        )
    )
    rows.extend(_latest_rows(_load(latest_path), report_path=latest_path))
    rows.sort(key=lambda row: str(row["source_id"]))
    identities = [str(row["source_id"]) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("candidate terminal ledger contains duplicate source identities")
    counts = Counter(str(row.get("disposition") or "unknown") for row in rows)
    domains = Counter(str(row.get("domain") or "unknown") for row in rows)
    return {
        "schema_version": "underrepresented-candidate-terminal-ledger-v1",
        "status": "complete_candidate_only",
        "candidate_only": True,
        "core_admission_claimed": False,
        "implementation_binding": {
            "implementation_tree_sha256": implementation_tree_sha256,
            "native_evidence_without_matching_binding_is_stale": True,
        },
        "policy": {
            "frozen_core_modified": False,
            "release_artifacts_modified": False,
            "declared_perturbations_do_not_create_source_independence": True,
            "method_transfer_is_not_source_consumption": True,
            "full_protocol21_required_before_core": True,
            "model_outcomes_used_for_filtering": False,
        },
        "counts": {
            "terminal_rows": len(rows),
            "microgrid_rows": sum(row["domain"] == "microgrid" for row in rows),
            "traffic_rows": sum(row["domain"] == "traffic" for row in rows),
            "latest_benchmark_rows": sum(
                str(row["source_id"]).startswith("latest::") for row in rows
            ),
            "native_replay_executed_rows": sum(
                bool((row.get("details") or {}).get("native_replay_executed"))
                for row in rows
            ),
            "current_hash_native_survivor_rows": 0,
            "full_protocol21_ready_rows": 0,
            "dispositions": dict(sorted(counts.items())),
            "domains": dict(sorted(domains.items())),
        },
        "input_bindings": {
            "microgrid_refine": {"path": _relative(microgrid_path), "sha256": _sha256(microgrid_path)},
            "traffic_queue": {"path": _relative(traffic_queue_path), "sha256": _sha256(traffic_queue_path)},
            "traffic_native": {"path": _relative(traffic_native_path), "sha256": _sha256(traffic_native_path)},
            "latest_benchmark": {"path": _relative(latest_path), "sha256": _sha256(latest_path)},
        },
        "rows": rows,
    }


def run(
    *,
    microgrid_path: Path = DEFAULT_MICROGRID,
    traffic_queue_path: Path = DEFAULT_TRAFFIC_QUEUE,
    traffic_native_path: Path = DEFAULT_TRAFFIC_NATIVE,
    latest_path: Path = DEFAULT_LATEST,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    if not output_root.is_relative_to(REPORTS_ROOT.resolve()):
        raise ValueError("output_root must stay below reports/")
    report = build_ledger(
        microgrid_path=microgrid_path.resolve(),
        traffic_queue_path=traffic_queue_path.resolve(),
        traffic_native_path=traffic_native_path.resolve(),
        latest_path=latest_path.resolve(),
        implementation_tree_sha256=_implementation_tree_sha256(),
    )
    report["ledger_sha256"] = _stable_hash(report)
    _write(output_root / "terminal_ledger.json", report)
    for row in report["rows"]:
        _write(output_root / "terminals" / f"{_stable_hash(row['source_id'])[:16]}.json", row)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microgrid", type=Path, default=DEFAULT_MICROGRID)
    parser.add_argument("--traffic-queue", type=Path, default=DEFAULT_TRAFFIC_QUEUE)
    parser.add_argument("--traffic-native", type=Path, default=DEFAULT_TRAFFIC_NATIVE)
    parser.add_argument("--latest", type=Path, default=DEFAULT_LATEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    report = run(
        microgrid_path=args.microgrid,
        traffic_queue_path=args.traffic_queue,
        traffic_native_path=args.traffic_native,
        latest_path=args.latest,
        output_root=args.output_root,
    )
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
